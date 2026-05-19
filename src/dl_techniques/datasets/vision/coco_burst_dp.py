"""Reference-set wrapper around the local COCO 2017 multi-task loader.

Per-sample, generates:
    - a *corrupted reference* (the model's input ref image)
    - a *variable-size* (1..n_max) set of *auxiliary views* of the same scene
      under different geometric warps + photometric/blur/occlusion corruptions
    - the *clean reference* as the reconstruction target
    - segmentation labels (unchanged from the underlying loader)

The aux-view distribution is the synthetic stand-in for real high-speed
multi-view capture (drone bursts, sensor sweeps): same scene, different
nuisance parameters. Anchor and aux corruptions are sampled independently.

Notes
-----
* Geometric warps applied to aux views are *not* inverted back to the anchor
  frame. The dense heads only predict in the anchor coordinate system; the
  aux views are read-only context that the model fuses via attention.
* All corruptions are implemented in numpy + PIL only — no opencv dependency
  is assumed. Rotation/translation is done with PIL's affine transform.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

import keras

from dl_techniques.datasets.vision.coco_multitask_local import (
    COCO_DEFAULT_ROOT,
    COCOMultiTaskConfig,
    COCO2017MultiTaskLoader,
    NUM_COCO_CLASSES,
)
from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------------
# Distortion utilities
# ---------------------------------------------------------------------------


def _add_gaussian_noise(img: np.ndarray, sigma: float, rng: random.Random) -> np.ndarray:
    """img in [0, 1], sigma in [0, 1] units (pre-scaled by 1/255)."""
    if sigma <= 0:
        return img
    # Use numpy's rng seeded from the python rng for reproducibility.
    seed = rng.randrange(0, 2**31 - 1)
    noise = np.random.default_rng(seed).normal(0.0, sigma, size=img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0).astype(np.float32)


def _adjust_brightness_contrast(
    img: np.ndarray,
    brightness: float,
    contrast: float,
) -> np.ndarray:
    """Multiplicative brightness and contrast in [0, 1] space."""
    out = img * contrast + brightness
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    pil = Image.fromarray((img * 255).astype(np.uint8))
    pil = pil.filter(ImageFilter.GaussianBlur(radius=sigma))
    return (np.asarray(pil, dtype=np.float32) / 255.0).astype(np.float32)


def _motion_blur(img: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    """Cheap directional motion blur via repeated rotated stripe averaging."""
    if length <= 1:
        return img
    # Build a length-pixel line kernel rotated by angle.
    k = np.zeros((length, length), dtype=np.float32)
    k[length // 2, :] = 1.0 / length
    pil_k = Image.fromarray((k * 255).astype(np.uint8))
    pil_k = pil_k.rotate(angle_deg, resample=Image.BILINEAR)
    k = np.asarray(pil_k, dtype=np.float32)
    k /= max(k.sum(), 1e-8)

    # Per-channel 2D conv via FFT.
    out = np.zeros_like(img)
    for c in range(img.shape[-1]):
        ch = img[..., c]
        # Pad to avoid wrap-around.
        pad = length
        ch_p = np.pad(ch, pad, mode="reflect")
        k_p = np.zeros_like(ch_p)
        kh, kw = k.shape
        k_p[:kh, :kw] = k
        k_p = np.roll(k_p, -kh // 2, axis=0)
        k_p = np.roll(k_p, -kw // 2, axis=1)
        f_img = np.fft.rfft2(ch_p)
        f_k = np.fft.rfft2(k_p)
        conv = np.fft.irfft2(f_img * f_k, s=ch_p.shape)
        conv = conv[pad:-pad, pad:-pad]
        out[..., c] = conv
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _affine_warp(
    img: np.ndarray,
    angle_deg: float,
    tx_frac: float,
    ty_frac: float,
    scale: float,
) -> np.ndarray:
    """Apply rotation + translation + scale around the image center."""
    h, w = img.shape[:2]
    pil = Image.fromarray((img * 255).astype(np.uint8))
    # PIL.rotate handles rotation + translation; scale via resize-and-paste.
    # Compose: first scale, then rotate with translation.
    if not math.isclose(scale, 1.0, abs_tol=1e-3):
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        pil = pil.resize((new_w, new_h), resample=Image.BILINEAR)
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        off_x = (w - new_w) // 2
        off_y = (h - new_h) // 2
        canvas.paste(pil, (off_x, off_y))
        pil = canvas
    tx = int(round(tx_frac * w))
    ty = int(round(ty_frac * h))
    pil = pil.rotate(
        angle_deg, resample=Image.BILINEAR, translate=(tx, ty), fillcolor=(0, 0, 0)
    )
    return (np.asarray(pil, dtype=np.float32) / 255.0).astype(np.float32)


def _random_occlusion(
    img: np.ndarray,
    max_frac: float,
    num_boxes_range: Tuple[int, int],
    rng: random.Random,
) -> np.ndarray:
    """Paint up to N rectangles in mean-grey covering up to ``max_frac`` of pixels."""
    out = img.copy()
    h, w = out.shape[:2]
    n = rng.randint(num_boxes_range[0], num_boxes_range[1])
    if n <= 0 or max_frac <= 0:
        return out
    fill = np.array([0.5, 0.5, 0.5], dtype=np.float32)  # neutral grey in [0,1]
    target_area = max_frac * h * w
    placed_area = 0.0
    for _ in range(n):
        if placed_area >= target_area:
            break
        box_h = rng.randint(max(2, h // 16), max(3, h // 4))
        box_w = rng.randint(max(2, w // 16), max(3, w // 4))
        y0 = rng.randint(0, h - box_h)
        x0 = rng.randint(0, w - box_w)
        out[y0 : y0 + box_h, x0 : x0 + box_w, :] = fill
        placed_area += box_h * box_w
    return out


# ---------------------------------------------------------------------------
# Distortion-spec dataclasses (per role: anchor vs aux)
# ---------------------------------------------------------------------------


@dataclass
class DistortionSpec:
    """Probability + parameter range bundle for a single role (ref or aux).

    All ``*_prob`` values gate independent Bernoulli draws of the
    corresponding effect. Geometric warp is applied unconditionally to aux
    (the *whole point* of aux views) and never to the anchor.
    """

    noise_sigma_range: Tuple[float, float] = (0.0, 0.0)
    brightness_jitter: float = 0.0      # +- additive
    contrast_jitter: float = 0.0        # +- around 1.0
    blur_sigma_range: Tuple[float, float] = (0.0, 0.0)
    motion_blur_prob: float = 0.0
    motion_blur_length_range: Tuple[int, int] = (1, 1)
    occlusion_prob: float = 0.0
    occlusion_max_frac: float = 0.0
    occlusion_num_boxes_range: Tuple[int, int] = (1, 1)
    # Geometric (aux only). Setting any range to (0,0) disables that axis.
    affine_angle_range_deg: Tuple[float, float] = (0.0, 0.0)
    affine_translate_frac_range: Tuple[float, float] = (0.0, 0.0)
    affine_scale_range: Tuple[float, float] = (1.0, 1.0)
    # Anchor: forbid geometric warp.
    allow_affine: bool = True


def default_anchor_spec() -> DistortionSpec:
    """Moderate corruption applied to the reference (input to the model)."""
    return DistortionSpec(
        noise_sigma_range=(0.02, 0.12),     # ~5..30 / 255
        brightness_jitter=0.05,
        contrast_jitter=0.1,
        blur_sigma_range=(0.0, 1.2),
        motion_blur_prob=0.2,
        motion_blur_length_range=(3, 11),
        occlusion_prob=0.1,
        occlusion_max_frac=0.05,
        occlusion_num_boxes_range=(1, 2),
        allow_affine=False,
    )


def default_aux_spec() -> DistortionSpec:
    """Stronger corruption + always-on geometric warp for aux views."""
    return DistortionSpec(
        noise_sigma_range=(0.02, 0.16),     # ~5..40 / 255
        brightness_jitter=0.12,
        contrast_jitter=0.2,
        blur_sigma_range=(0.0, 2.5),
        motion_blur_prob=0.4,
        motion_blur_length_range=(3, 21),
        occlusion_prob=0.3,
        occlusion_max_frac=0.15,
        occlusion_num_boxes_range=(1, 3),
        affine_angle_range_deg=(-10.0, 10.0),
        affine_translate_frac_range=(-0.08, 0.08),
        affine_scale_range=(0.95, 1.05),
        allow_affine=True,
    )


def apply_distortion(img: np.ndarray, spec: DistortionSpec, rng: random.Random) -> np.ndarray:
    """Apply a stochastic distortion stack defined by ``spec`` to ``img`` in [0,1]."""
    out = img

    # Geometric (aux only)
    if spec.allow_affine and (
        spec.affine_angle_range_deg != (0.0, 0.0)
        or spec.affine_translate_frac_range != (0.0, 0.0)
        or spec.affine_scale_range != (1.0, 1.0)
    ):
        angle = rng.uniform(*spec.affine_angle_range_deg)
        tx = rng.uniform(*spec.affine_translate_frac_range)
        ty = rng.uniform(*spec.affine_translate_frac_range)
        scale = rng.uniform(*spec.affine_scale_range)
        out = _affine_warp(out, angle, tx, ty, scale)

    # Brightness + contrast (always-on, small)
    brightness = rng.uniform(-spec.brightness_jitter, spec.brightness_jitter) if spec.brightness_jitter > 0 else 0.0
    contrast = 1.0 + (rng.uniform(-spec.contrast_jitter, spec.contrast_jitter) if spec.contrast_jitter > 0 else 0.0)
    if not (math.isclose(brightness, 0.0, abs_tol=1e-4) and math.isclose(contrast, 1.0, abs_tol=1e-4)):
        out = _adjust_brightness_contrast(out, brightness, contrast)

    # Gaussian blur
    if spec.blur_sigma_range[1] > 0:
        b_sigma = rng.uniform(*spec.blur_sigma_range)
        out = _gaussian_blur(out, b_sigma)

    # Motion blur
    if rng.random() < spec.motion_blur_prob:
        length = rng.randint(*spec.motion_blur_length_range)
        angle = rng.uniform(-180.0, 180.0)
        out = _motion_blur(out, length, angle)

    # Photometric noise (last, so blur doesn't smear it)
    if spec.noise_sigma_range[1] > 0:
        n_sigma = rng.uniform(*spec.noise_sigma_range)
        out = _add_gaussian_noise(out, n_sigma, rng)

    # Occlusion
    if rng.random() < spec.occlusion_prob:
        out = _random_occlusion(
            out,
            max_frac=spec.occlusion_max_frac,
            num_boxes_range=spec.occlusion_num_boxes_range,
            rng=rng,
        )

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


@dataclass
class COCOBurstDPConfig:
    """Configuration for :class:`COCO2017BurstDPLoader`."""

    coco_root: str = COCO_DEFAULT_ROOT
    split: str = "train2017"
    image_size: int = 256
    batch_size: int = 4

    n_max: int = 5                 # padding dimension for aux
    n_min: int = 1                 # min aux views per sample
    sample_n_per_sample: bool = True   # if False, every sample has exactly n_max

    max_images: Optional[int] = None
    shuffle: bool = True
    seed: int = 0
    workers: int = 4
    use_multiprocessing: bool = True

    anchor_spec: DistortionSpec = field(default_factory=default_anchor_spec)
    aux_spec: DistortionSpec = field(default_factory=default_aux_spec)


class COCO2017BurstDPLoader(keras.utils.PyDataset):
    """Yields ``({ref, aux, aux_mask}, {recon, segmentation})``."""

    def __init__(self, config: COCOBurstDPConfig, **kwargs: Any) -> None:
        super().__init__(
            workers=config.workers,
            use_multiprocessing=config.use_multiprocessing,
            **kwargs,
        )
        self.cfg = config
        self._rng = random.Random(config.seed)

        # Underlying classification + segmentation loader.
        # We use it with batch_size=1 and pull a single (image, labels) at a
        # time, then build the ref/aux stack ourselves.
        underlying_cfg = COCOMultiTaskConfig(
            coco_root=config.coco_root,
            split=config.split,
            image_size=config.image_size,
            batch_size=1,
            max_images=config.max_images,
            shuffle=config.shuffle,
            seed=config.seed,
            augment=False,           # we control augmentation here
            workers=1,
            use_multiprocessing=False,
            emit_boxes=False,
        )
        self._underlying = COCO2017MultiTaskLoader(underlying_cfg)
        if config.n_min < 0 or config.n_min > config.n_max:
            raise ValueError(
                f"Require 0 <= n_min ({config.n_min}) <= n_max ({config.n_max})."
            )

    def __len__(self) -> int:
        n_imgs = len(self._underlying.image_ids)
        return (n_imgs + self.cfg.batch_size - 1) // self.cfg.batch_size

    def _sample_n(self) -> int:
        if not self.cfg.sample_n_per_sample:
            return self.cfg.n_max
        return self._rng.randint(self.cfg.n_min, self.cfg.n_max)

    def _build_one_sample(self, image_id: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        size = self.cfg.image_size
        n_max = self.cfg.n_max

        # Load and resize the clean image at the working resolution.
        clean = self._underlying._load_image(image_id, size)  # noqa: SLF001 — intentional
        seg = self._underlying._build_mask(image_id, (size, size))  # noqa: SLF001

        ref_corrupt = apply_distortion(clean, self.cfg.anchor_spec, self._rng)

        n = self._sample_n()
        aux = np.zeros((n_max, size, size, 3), dtype=np.float32)
        aux_mask = np.zeros((n_max,), dtype=np.float32)
        for k in range(n):
            aux[k] = apply_distortion(clean, self.cfg.aux_spec, self._rng)
            aux_mask[k] = 1.0

        inputs: Dict[str, np.ndarray] = {
            "ref": ref_corrupt.astype(np.float32),
            "aux": aux,
            "aux_mask": aux_mask,
        }
        labels: Dict[str, np.ndarray] = {
            "recon": clean.astype(np.float32),
            "segmentation": seg.astype(np.int32),
        }
        return inputs, labels

    def __getitem__(self, idx: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        bs = self.cfg.batch_size
        start = idx * bs
        end = min(start + bs, len(self._underlying.image_ids))
        batch_ids = self._underlying.image_ids[start:end]
        B = len(batch_ids)

        size = self.cfg.image_size
        n_max = self.cfg.n_max

        ref_b = np.empty((B, size, size, 3), dtype=np.float32)
        aux_b = np.empty((B, n_max, size, size, 3), dtype=np.float32)
        aux_mask_b = np.empty((B, n_max), dtype=np.float32)
        recon_b = np.empty((B, size, size, 3), dtype=np.float32)
        seg_b = np.empty((B, size, size), dtype=np.int32)

        for i, image_id in enumerate(batch_ids):
            inputs, labels = self._build_one_sample(image_id)
            ref_b[i] = inputs["ref"]
            aux_b[i] = inputs["aux"]
            aux_mask_b[i] = inputs["aux_mask"]
            recon_b[i] = labels["recon"]
            seg_b[i] = labels["segmentation"]

        inputs_batch: Dict[str, np.ndarray] = {
            "ref": ref_b,
            "aux": aux_b,
            "aux_mask": aux_mask_b,
        }
        labels_batch: Dict[str, np.ndarray] = {
            "recon": recon_b,
            "segmentation": seg_b,
        }
        return inputs_batch, labels_batch

    def on_epoch_end(self) -> None:
        if self.cfg.shuffle:
            self._rng.shuffle(self._underlying.image_ids)

    def probe(self) -> Dict[str, Any]:
        x, y = self[0]
        return {
            "ref_shape": tuple(x["ref"].shape),
            "aux_shape": tuple(x["aux"].shape),
            "aux_mask_shape": tuple(x["aux_mask"].shape),
            "n_aux_per_sample": x["aux_mask"].sum(axis=-1).tolist(),
            "recon_shape": tuple(y["recon"].shape),
            "segmentation_shape": tuple(y["segmentation"].shape),
            "ref_range": (float(x["ref"].min()), float(x["ref"].max())),
            "recon_range": (float(y["recon"].min()), float(y["recon"].max())),
            "seg_unique_classes_sample0": int(len(np.unique(y["segmentation"][0]))),
        }


def build_coco_burst_dp_datasets(
    coco_root: str = COCO_DEFAULT_ROOT,
    image_size: int = 256,
    batch_size: int = 4,
    n_max: int = 5,
    n_min: int = 1,
    max_train_images: Optional[int] = None,
    max_val_images: Optional[int] = None,
    workers: int = 4,
    aux_spec: Optional[DistortionSpec] = None,
    seed: int = 0,
) -> Tuple[COCO2017BurstDPLoader, COCO2017BurstDPLoader]:
    """Build (train, val) COCO Burst-DP loaders.

    Parameters
    ----------
    aux_spec : DistortionSpec, optional
        If provided, overrides the default aux distortion spec for both
        train and val loaders. ``None`` (default) keeps the current
        behavior (``default_aux_spec()`` via the config dataclass).
    """
    train_kwargs: Dict[str, Any] = dict(
        coco_root=coco_root,
        split="train2017",
        image_size=image_size,
        batch_size=batch_size,
        n_max=n_max,
        n_min=n_min,
        max_images=max_train_images,
        shuffle=True,
        seed=seed,
        workers=workers,
    )
    val_kwargs: Dict[str, Any] = dict(
        coco_root=coco_root,
        split="val2017",
        image_size=image_size,
        batch_size=batch_size,
        n_max=n_max,
        n_min=n_min,
        max_images=max_val_images,
        shuffle=False,
        seed=seed + 1,
        workers=max(1, workers // 2),
    )
    if aux_spec is not None:
        train_kwargs["aux_spec"] = aux_spec
        val_kwargs["aux_spec"] = aux_spec
    train_cfg = COCOBurstDPConfig(**train_kwargs)
    val_cfg = COCOBurstDPConfig(**val_kwargs)
    return COCO2017BurstDPLoader(train_cfg), COCO2017BurstDPLoader(val_cfg)
