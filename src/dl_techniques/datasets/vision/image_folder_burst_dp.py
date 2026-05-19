"""Image-folder Burst-DP loader for fidelity-only datasets (DIV2K, VGG-Face2).

Mirrors :class:`COCO2017BurstDPLoader` semantics — per-sample emits a
corrupted reference, a variable-size aux-view stack, and a clean
reconstruction target — but draws clean images from a flat list of image
paths instead of the COCO multi-task loader. Segmentation labels are
emitted as all-zero ``int32`` arrays so the dual-head :class:`BurstDP`
model can be trained on fidelity-only data without any model-side change.

This module is intentionally a thin sibling of ``coco_burst_dp.py``: the
distortion primitives (``apply_distortion``, ``DistortionSpec``,
``default_anchor_spec``, ``default_aux_spec``) are **imported**, not
duplicated.

# DECISION plan_2026-05-19_64f2a17b/D-001
Fidelity-only datasets (DIV2K, VGG-Face2) lack segmentation labels. We
emit dummy all-zero seg labels here and rely on the trainer to zero out
``loss_weights["segmentation"]`` and drop seg metrics. This keeps
``BurstDP``, ``BurstDPConfig``, checkpoints, and existing tests unchanged.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import keras

from dl_techniques.datasets.vision.coco_burst_dp import (
    DistortionSpec,
    apply_distortion,
    default_anchor_spec,
    default_aux_spec,
)
from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------------
# Config + loader
# ---------------------------------------------------------------------------


@dataclass
class ImageFolderBurstDPConfig:
    """Configuration for :class:`ImageFolderBurstDPLoader`.

    Parallel to :class:`COCOBurstDPConfig` but takes an explicit list of
    image paths instead of a COCO root + split.
    """

    image_paths: List[str] = field(default_factory=list)
    image_size: int = 256
    batch_size: int = 4

    n_max: int = 5
    n_min: int = 1
    sample_n_per_sample: bool = True

    shuffle: bool = True
    seed: int = 0
    workers: int = 4
    use_multiprocessing: bool = True

    anchor_spec: DistortionSpec = field(default_factory=default_anchor_spec)
    aux_spec: DistortionSpec = field(default_factory=default_aux_spec)


def _load_image_from_path(path: str, size: int) -> np.ndarray:
    """Load + resize one image from disk to a [0, 1] float32 array.

    Raises any underlying PIL exception to the caller; per-sample retry
    is handled in :meth:`ImageFolderBurstDPLoader._build_one_sample`.
    """
    with Image.open(path) as pil:
        pil = pil.convert("RGB")
        if pil.size != (size, size):
            pil = pil.resize((size, size), resample=Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
    return arr.astype(np.float32)


class ImageFolderBurstDPLoader(keras.utils.PyDataset):
    """Yields ``({ref, aux, aux_mask}, {recon, segmentation})`` from a flat path list.

    Segmentation labels are all-zero ``int32`` of shape ``(B, H, W)``.
    """

    def __init__(self, config: ImageFolderBurstDPConfig, **kwargs: Any) -> None:
        super().__init__(
            workers=max(1, min(config.workers, max(1, len(config.image_paths)))),
            use_multiprocessing=config.use_multiprocessing,
            **kwargs,
        )
        if not config.image_paths:
            raise FileNotFoundError(
                "ImageFolderBurstDPLoader: empty image_paths. "
                "Check dataset root / discovery globs."
            )
        if config.n_min < 0 or config.n_min > config.n_max:
            raise ValueError(
                f"Require 0 <= n_min ({config.n_min}) <= n_max ({config.n_max})."
            )
        self.cfg = config
        self._rng = random.Random(config.seed)
        # Mutable copy — shuffled in-place on epoch end if shuffle=True.
        self.image_paths: List[str] = list(config.image_paths)
        if config.shuffle:
            self._rng.shuffle(self.image_paths)

        logger.info(
            f"ImageFolderBurstDPLoader: #images={len(self.image_paths)}, "
            f"image_size={config.image_size}, batch_size={config.batch_size}, "
            f"n_min={config.n_min}, n_max={config.n_max}, "
            f"workers={config.workers}, mp={config.use_multiprocessing}"
        )

    # -- helpers -----------------------------------------------------------

    def __len__(self) -> int:
        n = len(self.image_paths)
        return (n + self.cfg.batch_size - 1) // self.cfg.batch_size

    def _sample_n(self) -> int:
        if not self.cfg.sample_n_per_sample:
            return self.cfg.n_max
        return self._rng.randint(self.cfg.n_min, self.cfg.n_max)

    def _load_with_retry(self, idx: int, size: int, max_retries: int = 3) -> np.ndarray:
        """Try ``max_retries`` consecutive paths starting at ``idx``.

        On per-file load failure (e.g. corrupt JPG in VGG-Face2), advance
        the index and try the next one. After ``max_retries`` failures,
        re-raise.
        """
        n = len(self.image_paths)
        last_exc: Optional[BaseException] = None
        for offset in range(max_retries):
            j = (idx + offset) % n
            path = self.image_paths[j]
            try:
                return _load_image_from_path(path, size)
            except Exception as exc:  # noqa: BLE001 — broad on purpose
                last_exc = exc
                logger.warning(
                    f"ImageFolderBurstDPLoader: failed to load '{path}' "
                    f"(attempt {offset + 1}/{max_retries}): {exc}"
                )
        # All retries exhausted.
        raise RuntimeError(
            f"ImageFolderBurstDPLoader: {max_retries} consecutive load failures "
            f"starting at idx={idx}. Last error: {last_exc!r}"
        )

    def _build_one_sample(
        self, idx: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        size = self.cfg.image_size
        n_max = self.cfg.n_max

        clean = self._load_with_retry(idx, size)

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
            # Fidelity-only: dummy seg labels, all zeros, int32.
            "segmentation": np.zeros((size, size), dtype=np.int32),
        }
        return inputs, labels

    def __getitem__(
        self, idx: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        bs = self.cfg.batch_size
        n = len(self.image_paths)
        start = idx * bs
        end = min(start + bs, n)
        B = end - start

        size = self.cfg.image_size
        n_max = self.cfg.n_max

        ref_b = np.empty((B, size, size, 3), dtype=np.float32)
        aux_b = np.empty((B, n_max, size, size, 3), dtype=np.float32)
        aux_mask_b = np.empty((B, n_max), dtype=np.float32)
        recon_b = np.empty((B, size, size, 3), dtype=np.float32)
        seg_b = np.zeros((B, size, size), dtype=np.int32)

        for i in range(B):
            inputs, labels = self._build_one_sample(start + i)
            ref_b[i] = inputs["ref"]
            aux_b[i] = inputs["aux"]
            aux_mask_b[i] = inputs["aux_mask"]
            recon_b[i] = labels["recon"]
            seg_b[i] = labels["segmentation"]  # already zeros — kept for symmetry

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
            self._rng.shuffle(self.image_paths)

    def probe(self) -> Dict[str, Any]:
        x, y = self[0]
        return {
            "ref_shape": tuple(x["ref"].shape),
            "aux_shape": tuple(x["aux"].shape),
            "aux_mask_shape": tuple(x["aux_mask"].shape),
            "n_aux_per_sample": x["aux_mask"].sum(axis=-1).tolist(),
            "recon_shape": tuple(y["recon"].shape),
            "segmentation_shape": tuple(y["segmentation"].shape),
            "segmentation_dtype": str(y["segmentation"].dtype),
            "segmentation_all_zero": bool((y["segmentation"] == 0).all()),
            "ref_range": (float(x["ref"].min()), float(x["ref"].max())),
            "recon_range": (float(y["recon"].min()), float(y["recon"].max())),
        }


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


_IMG_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def discover_div2k_paths(div2k_root: str) -> Tuple[List[str], List[str]]:
    """Discover DIV2K train/validation PNG paths.

    Expects layout ``<div2k_root>/{train,validation}/*.png``.

    Returns
    -------
    train_paths, val_paths : sorted lists of absolute paths.

    Raises
    ------
    FileNotFoundError : if either subdir is missing or empty.
    """
    train_dir = os.path.join(div2k_root, "train")
    val_dir = os.path.join(div2k_root, "validation")

    def _list_pngs(d: str) -> List[str]:
        if not os.path.isdir(d):
            return []
        return sorted(
            os.path.join(d, f)
            for f in os.listdir(d)
            if f.endswith(_IMG_EXTS)
        )

    train_paths = _list_pngs(train_dir)
    val_paths = _list_pngs(val_dir)

    if not train_paths:
        raise FileNotFoundError(
            f"discover_div2k_paths: no images under '{train_dir}'. "
            f"Expected DIV2K layout '<root>/train/*.png'."
        )
    if not val_paths:
        raise FileNotFoundError(
            f"discover_div2k_paths: no images under '{val_dir}'. "
            f"Expected DIV2K layout '<root>/validation/*.png'."
        )
    logger.info(
        f"discover_div2k_paths: train={len(train_paths)}, val={len(val_paths)} "
        f"(root='{div2k_root}')"
    )
    return train_paths, val_paths


def discover_vggface2_paths(vggface2_root: str) -> Tuple[List[str], List[str]]:
    """Discover VGG-Face2 train/test image paths from pre-built list files.

    Expects layout::

        <vggface2_root>/train_list.txt
        <vggface2_root>/test_list.txt
        <vggface2_root>/train/<id>/<file>.jpg
        <vggface2_root>/test/<id>/<file>.jpg

    Returns
    -------
    train_paths, val_paths : lists of absolute paths.

    Raises
    ------
    FileNotFoundError : if either list file is missing or empty.
    """
    train_list = os.path.join(vggface2_root, "train_list.txt")
    test_list = os.path.join(vggface2_root, "test_list.txt")

    def _read_list(list_path: str, subdir: str) -> List[str]:
        if not os.path.isfile(list_path):
            return []
        out: List[str] = []
        with open(list_path, "rt", encoding="utf-8") as fh:
            for line in fh:
                rel = line.strip()
                if not rel:
                    continue
                out.append(os.path.join(vggface2_root, subdir, rel))
        return out

    train_paths = _read_list(train_list, "train")
    val_paths = _read_list(test_list, "test")

    if not train_paths:
        raise FileNotFoundError(
            f"discover_vggface2_paths: missing or empty '{train_list}'."
        )
    if not val_paths:
        raise FileNotFoundError(
            f"discover_vggface2_paths: missing or empty '{test_list}'."
        )
    logger.info(
        f"discover_vggface2_paths: train={len(train_paths)}, val={len(val_paths)} "
        f"(root='{vggface2_root}')"
    )
    return train_paths, val_paths


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _slice(paths: List[str], cap: Optional[int]) -> List[str]:
    if cap is None or cap <= 0:
        return paths
    return paths[:cap]


def _build_pair(
    train_paths: List[str],
    val_paths: List[str],
    *,
    image_size: int,
    batch_size: int,
    n_max: int,
    n_min: int,
    workers: int,
    seed: int,
    aux_spec: Optional[DistortionSpec] = None,
) -> Tuple[ImageFolderBurstDPLoader, ImageFolderBurstDPLoader]:
    train_kwargs: Dict[str, Any] = dict(
        image_paths=train_paths,
        image_size=image_size,
        batch_size=batch_size,
        n_max=n_max,
        n_min=n_min,
        shuffle=True,
        seed=seed,
        workers=workers,
    )
    val_kwargs: Dict[str, Any] = dict(
        image_paths=val_paths,
        image_size=image_size,
        batch_size=batch_size,
        n_max=n_max,
        n_min=n_min,
        shuffle=False,
        seed=seed + 1,
        workers=max(1, workers // 2),
    )
    if aux_spec is not None:
        train_kwargs["aux_spec"] = aux_spec
        val_kwargs["aux_spec"] = aux_spec
    train_cfg = ImageFolderBurstDPConfig(**train_kwargs)
    val_cfg = ImageFolderBurstDPConfig(**val_kwargs)
    return ImageFolderBurstDPLoader(train_cfg), ImageFolderBurstDPLoader(val_cfg)


def build_div2k_burst_dp_datasets(
    div2k_root: str,
    image_size: int = 256,
    batch_size: int = 4,
    n_max: int = 5,
    n_min: int = 1,
    max_train_images: Optional[int] = None,
    max_val_images: Optional[int] = None,
    workers: int = 4,
    aux_spec: Optional[DistortionSpec] = None,
    seed: int = 0,
) -> Tuple[ImageFolderBurstDPLoader, ImageFolderBurstDPLoader]:
    """Build (train, val) Burst-DP loaders for DIV2K.

    Parameters
    ----------
    aux_spec : DistortionSpec, optional
        If provided, overrides the default aux distortion spec for both
        loaders. ``None`` keeps the current default behavior.
    """
    train_paths, val_paths = discover_div2k_paths(div2k_root)
    train_paths = _slice(train_paths, max_train_images)
    val_paths = _slice(val_paths, max_val_images)
    return _build_pair(
        train_paths,
        val_paths,
        image_size=image_size,
        batch_size=batch_size,
        n_max=n_max,
        n_min=n_min,
        workers=workers,
        seed=seed,
        aux_spec=aux_spec,
    )


def build_vggface2_burst_dp_datasets(
    vggface2_root: str,
    image_size: int = 256,
    batch_size: int = 4,
    n_max: int = 5,
    n_min: int = 1,
    max_train_images: Optional[int] = None,
    max_val_images: Optional[int] = None,
    workers: int = 4,
    aux_spec: Optional[DistortionSpec] = None,
    seed: int = 0,
) -> Tuple[ImageFolderBurstDPLoader, ImageFolderBurstDPLoader]:
    """Build (train, val) Burst-DP loaders for VGG-Face2.

    Parameters
    ----------
    aux_spec : DistortionSpec, optional
        If provided, overrides the default aux distortion spec for both
        loaders. ``None`` keeps the current default behavior.
    """
    train_paths, val_paths = discover_vggface2_paths(vggface2_root)
    train_paths = _slice(train_paths, max_train_images)
    val_paths = _slice(val_paths, max_val_images)
    return _build_pair(
        train_paths,
        val_paths,
        image_size=image_size,
        batch_size=batch_size,
        n_max=n_max,
        n_min=n_min,
        workers=workers,
        seed=seed,
        aux_spec=aux_spec,
    )
