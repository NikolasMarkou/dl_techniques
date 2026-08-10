"""Reconstruction-error anomaly detector — **currently non-functional**.

.. warning::

   This application has no model to run. It was built entirely on
   ``ConvNeXtPatchVAE``, and ``dl_techniques.models.convnext_patch_vae`` was
   removed from the library (along with ``src/train/convnext_patch_vae/``).
   :meth:`PatchReconstructionAnomalyDetector.from_pretrained` therefore raises
   ``RuntimeError`` unconditionally, and no surviving model in
   ``dl_techniques`` implements the ``sample_from(x, temperature=...)``
   deterministic-decode API that :meth:`anomaly_maps` calls. The scoring,
   thresholding and overlay code below is architecture-agnostic and still
   correct, so the module is kept for a caller who supplies their own
   compatible model object — but there is no in-repo way to obtain one.

The detector scores every patch by how poorly the trained VAE **reconstructs**
it: the squared error between the input and a deterministic decode, average-
pooled to the patch grid. Patches the model reconstructs poorly are flagged as
anomalous. This signal is sign-correct by construction (high reconstruction
error = anomalous) and **sampler-agnostic** — it does not depend on whether the
latent prior is Gaussian or vMF, because the only model call is a deterministic
decode (``sample_from(x, temperature=0.0)``):

* for a ``vmf`` checkpoint, ``temperature=0.0`` decodes the unit-normalized mean
  direction;
* for a ``gaussian`` checkpoint, ``temperature=0.0`` decodes ``mu``.

For a bce-trained checkpoint the decode is in ``[0, 1]`` (the input is also in
``[0, 1]``), so the per-pixel MSE is well-scaled. Errors are mean-pooled over
non-overlapping ``patch_size x patch_size`` blocks → an ``(Hp, Wp)`` map (e.g.
``32x32`` for a 256px image at ``patch_size=8``).

This module deliberately has **no GUI dependency** — it stays importable and
usable headless. The Streamlit front-end lives in ``streamlit_app.py``.

Example (requires a model object you bring yourself — ``from_pretrained``
raises)::

    from applications.anomaly_detection.anomaly_detector import (
        PatchReconstructionAnomalyDetector,
    )

    det = PatchReconstructionAnomalyDetector(my_model)  # must expose sample_from
    x, (h, w) = det.preprocess("photo.jpg")             # native res, padded to /patch
    amap = det.anomaly_maps(x, orig_hw=(h, w))["anomaly"]  # (Hp, Wp)
    mask, thr = det.anomaly_mask(amap)                  # boolean (Hp, Wp)
    overlay = det.overlay(x[0][:h, :w], amap)           # uint8 (h, w, 3)
"""

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import keras
from keras import ops
import matplotlib

from dl_techniques.utils.logger import logger

ImageLike = Union[str, np.ndarray]


class PatchReconstructionAnomalyDetector:
    """Per-patch reconstruction-error anomaly detector.

    Scores each patch by the mean squared error between the input and the
    trained VAE's deterministic decode, pooled to the ``(Hp, Wp)`` patch grid.
    The signal is sampler-agnostic (no branch on the latent prior type).

    There is no in-repo way to obtain a compatible ``model``: the only
    architecture that ever provided one (``ConvNeXtPatchVAE``) was removed from
    the library, and :meth:`from_pretrained` now raises. Construct this class
    directly only if you already hold your own model object.

    Args:
        model: A model exposing ``sample_from(x, temperature)`` and, optionally,
            a ``.config`` carrying ``patch_size`` / ``recon_loss_type`` /
            ``img_size`` (each falls back to a default when absent).
    """

    def __init__(
        self,
        model: keras.Model,
    ) -> None:
        self.model = model
        cfg = getattr(model, "config", None)
        self.patch_size = int(getattr(cfg, "patch_size", 4))
        self.recon_loss_type = str(getattr(cfg, "recon_loss_type", "bce"))
        self.default_image_size = int(getattr(cfg, "img_size", 128) or 128)
        logger.info(
            "PatchReconstructionAnomalyDetector ready: per-patch reconstruction "
            "error, patch_size=%d, recon_loss=%s",
            self.patch_size,
            self.recon_loss_type,
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    # DECISION plan-2026-08-10T130454-3649c19e/D-016: this method raises
    # unconditionally and MUST NOT be "repaired" by re-adding a load path.
    # ``ConvNeXtPatchVAE`` was the ONLY architecture it ever supported, and
    # ``dl_techniques.models.convnext_patch_vae`` no longer exists in the
    # library, so there is no surviving branch to fall through to. Do NOT
    # replace this with `keras.models.load_model(...)` without custom objects:
    # a `.keras` checkpoint of that model deserializes its registered custom
    # classes by name, so it would fail deep inside Keras with an opaque
    # unknown-object error instead of naming the removed architecture. Do NOT
    # return ``None`` either — every caller (`streamlit_app.load_detector`, the
    # README's programmatic example) immediately dereferences the result. See
    # decisions.md D-016; the sibling app's D-009 set this precedent.
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
    ) -> "PatchReconstructionAnomalyDetector":
        """Unsupported: the only architecture this detector could load is gone.

        Args:
            model_path: Path to the ``.keras`` checkpoint. Reported back in the
                error message; never opened.

        Returns:
            Never returns.

        Raises:
            RuntimeError: always. ``ConvNeXtPatchVAE`` was the sole architecture
                this loader supported and its model module was removed from the
                library, so no checkpoint can be loaded through this path.
        """
        logger.error(
            "PatchReconstructionAnomalyDetector.from_pretrained is no longer "
            "supported; refusing to load %s",
            model_path,
        )
        raise RuntimeError(
            "PatchReconstructionAnomalyDetector.from_pretrained cannot load "
            f"'{model_path}'. This detector only ever supported the "
            "ConvNeXtPatchVAE architecture, and its model module "
            "(dl_techniques.models.convnext_patch_vae) has been REMOVED from "
            "the library, together with its trainers "
            "(src/train/convnext_patch_vae/). There is no alternative "
            "architecture to fall back to: no surviving model in "
            "dl_techniques implements the sample_from(x, temperature=...) "
            "deterministic-decode API this detector scores with, so the whole "
            "application is non-functional, not just this loader. Your "
            "options: (1) construct PatchReconstructionAnomalyDetector(model) "
            "directly with your own already-loaded keras.Model that exposes "
            "sample_from(x, temperature=0.0) and a .config carrying "
            "patch_size; or (2) recover the convnext_patch_vae package from "
            "git history if you still hold a trained checkpoint."
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def preprocess(
        self,
        image: ImageLike,
        multiple: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Load an image at native resolution and pad to a model-ready batch.

        The VAE is resolution-agnostic: any spatial size divisible by
        ``patch_size`` works. Rather than square-resizing (which distorts
        aspect ratio), the image keeps its native size and is reflect-padded on
        the bottom/right up to the next multiple of ``multiple``.

        Args:
            image: A file path or an ``HxW``/``HxWx3``/``HxWx4`` array
                (uint8 in [0, 255] or float in [0, 1]).
            multiple: Pad each side up to a multiple of this. Defaults to
                ``patch_size``.
            max_size: If set and the longer side exceeds it, downscale
                (aspect-preserving) before padding — caps GPU memory. ``None``
                or ``0`` keeps native resolution.

        Returns:
            ``(x, (orig_h, orig_w))`` — ``x`` is a ``(1, H', W', 3)`` float32
            batch in ``[0, 1]`` padded to multiples of ``multiple``;
            ``(orig_h, orig_w)`` is the unpadded content size for cropping.
        """
        mult = int(multiple or self.patch_size)

        if isinstance(image, str):
            img = keras.utils.img_to_array(
                keras.utils.load_img(image, color_mode="rgb")
            )
        else:
            img = np.asarray(image)

        img = img.astype("float32")
        if img.ndim == 2:  # grayscale -> RGB
            img = np.stack([img] * 3, axis=-1)
        if img.ndim == 3 and img.shape[-1] == 4:  # RGBA -> RGB
            img = img[..., :3]
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"Cannot interpret image of shape {img.shape}")

        if img.max() > 1.0:  # uint8-range -> [0, 1]
            img = img / 255.0

        # Optional aspect-preserving downscale to cap memory.
        if max_size:
            h0, w0 = img.shape[:2]
            longest = max(h0, w0)
            if longest > max_size:
                scale = float(max_size) / float(longest)
                new_hw = (max(1, round(h0 * scale)), max(1, round(w0 * scale)))
                img = np.asarray(
                    ops.image.resize(img, size=new_hw, antialias=True),
                    dtype="float32",
                )

        h, w = img.shape[:2]
        pad_h = (-h) % mult
        pad_w = (-w) % mult
        if pad_h or pad_w:
            # reflect needs pad < dim; fall back to edge for tiny images.
            mode = "reflect" if (pad_h < h and pad_w < w) else "edge"
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)

        x = np.clip(img[None].astype("float32"), 0.0, 1.0)
        return x, (h, w)

    # ------------------------------------------------------------------
    # Reconstruction-error scoring
    # ------------------------------------------------------------------
    def anomaly_maps(
        self, x: np.ndarray, orig_hw: Optional[Tuple[int, int]] = None
    ) -> Dict[str, np.ndarray]:
        """Per-patch reconstruction-error anomaly map for one image.

        The model is decoded deterministically (``sample_from`` at
        ``temperature=0.0``), which is sampler-agnostic: for a ``vmf``
        checkpoint ``t=0`` decodes the unit-normalized mean direction; for a
        ``gaussian`` checkpoint ``t=0`` decodes ``mu``. A bce-trained decode is
        in ``[0, 1]`` (as is the input), so the per-pixel MSE is well-scaled.
        The pixel errors are mean-pooled over non-overlapping
        ``patch_size x patch_size`` blocks to the patch grid.

        Args:
            x: A ``(1, H', W', 3)`` float32 batch in ``[0, 1]`` from
                :meth:`preprocess` (spatial dims multiples of ``patch_size``).
            orig_hw: Optional unpadded ``(H, W)`` (from :meth:`preprocess`).
                When given, the map is cropped to its valid patch region
                (``ceil(dim / patch_size)``) so reflect-padded patches do not
                pollute scores or overlays.

        Returns:
            ``{"anomaly": (Hp, Wp)}`` numpy float32 array — the per-patch mean
            squared reconstruction error.
        """
        # Deterministic decode, (1, H, W, 3) in [0, 1] for a bce checkpoint.
        recon = self.model.sample_from(x, temperature=0.0)
        err = ops.mean(
            ops.square(ops.cast(x, "float32") - ops.cast(recon, "float32")),
            axis=-1,
        )  # (1, H, W)

        patch = self.patch_size
        shape = ops.shape(err)
        h_dim, w_dim = shape[1], shape[2]
        hp = h_dim // patch
        wp = w_dim // patch
        # Average-pool over non-overlapping patch x patch blocks (keras.ops only).
        blocks = ops.reshape(err, (1, hp, patch, wp, patch))
        pooled = ops.mean(blocks, axis=(2, 4))  # (1, Hp, Wp)
        amap = np.asarray(pooled, dtype="float32")[0]

        if orig_hw is not None:
            h, w = orig_hw
            amap = amap[
                : math.ceil(h / self.patch_size),
                : math.ceil(w / self.patch_size),
            ]

        return {"anomaly": amap}

    # ------------------------------------------------------------------
    # Thresholding / scoring
    # ------------------------------------------------------------------
    def anomaly_mask(
        self,
        score_map: np.ndarray,
        method: str = "zscore",
        k: float = 3.0,
        percentile: float = 95.0,
        abs_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """Threshold a reconstruction-error map into a boolean anomaly mask.

        Args:
            score_map: A per-patch reconstruction-error map ``(Hp, Wp)``.
            method: ``"zscore"`` (``mean + k*std``), ``"percentile"``, or
                ``"absolute"`` (requires ``abs_threshold``).
            k: z-score multiplier for ``"zscore"``.
            percentile: percentile (0-100) for ``"percentile"``.
            abs_threshold: reconstruction-error (MSE units) cut-off for
                ``"absolute"``.

        Returns:
            ``(mask, threshold)`` — boolean array matching ``score_map`` and the
            scalar threshold used.

        Raises:
            ValueError: for an unknown method or missing ``abs_threshold``.
        """
        m = np.asarray(score_map, dtype="float32")
        if method == "zscore":
            thr = float(m.mean() + k * m.std())
        elif method == "percentile":
            thr = float(np.percentile(m, percentile))
        elif method == "absolute":
            if abs_threshold is None:
                raise ValueError("method='absolute' requires abs_threshold")
            thr = float(abs_threshold)
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        return m > thr, thr

    def score(
        self, score_map: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Scalar summaries of a reconstruction-error map (and optional mask).

        Args:
            score_map: A per-patch reconstruction-error map ``(Hp, Wp)``.
            mask: Optional boolean mask for anomalous-patch counts.

        Returns:
            Dict with ``mean_score, max_score, p95_score`` and, if ``mask``
            given, ``n_anomalous`` and ``frac_anomalous``.
        """
        m = np.asarray(score_map, dtype="float32")
        out: Dict[str, float] = {
            "mean_score": float(m.mean()),
            "max_score": float(m.max()),
            "p95_score": float(np.percentile(m, 95.0)),
        }
        if mask is not None:
            out["n_anomalous"] = int(np.count_nonzero(mask))
            out["frac_anomalous"] = float(np.mean(mask))
        return out

    # ------------------------------------------------------------------
    # Visualization helpers (matplotlib colormaps only — no pyplot/backend)
    # ------------------------------------------------------------------
    @staticmethod
    def _resize_nearest(arr: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
        """Nearest-neighbour upsample a 2D ``(h, w)`` array to ``(H, W)``."""
        h, w = arr.shape[:2]
        out_h, out_w = out_hw
        yi = (np.arange(out_h) * h // out_h).clip(0, h - 1)
        xi = (np.arange(out_w) * w // out_w).clip(0, w - 1)
        return arr[yi][:, xi]

    @staticmethod
    def _as_uint8(image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype="float32")
        if img.max() > 1.0:
            img = img / 255.0
        return np.clip(img, 0.0, 1.0)

    def overlay(
        self,
        image01: np.ndarray,
        score_map: np.ndarray,
        cmap: str = "inferno",
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Alpha-blend a colormapped reconstruction-error heatmap over the image.

        Args:
            image01: ``(H, W, 3)`` image (``[0, 1]`` or ``[0, 255]``).
            score_map: per-patch reconstruction-error map ``(Hp, Wp)``.
            cmap: matplotlib colormap name.
            alpha: heatmap opacity in ``[0, 1]``.

        Returns:
            ``(H, W, 3)`` uint8 RGB image.
        """
        base = self._as_uint8(image01)
        h, w = base.shape[:2]
        big = self._resize_nearest(np.asarray(score_map, "float32"), (h, w))
        ptp = float(big.max() - big.min())
        norm = (big - big.min()) / (ptp + 1e-8)
        heat = matplotlib.colormaps[cmap](norm)[..., :3]
        blended = (1.0 - alpha) * base + alpha * heat
        return (np.clip(blended, 0.0, 1.0) * 255.0).astype("uint8")

    def mask_overlay(
        self,
        image01: np.ndarray,
        mask: np.ndarray,
        color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Tint anomalous patches over the image.

        Args:
            image01: ``(H, W, 3)`` image (``[0, 1]`` or ``[0, 255]``).
            mask: boolean per-patch mask ``(Hp, Wp)``.
            color: RGB tint in ``[0, 1]``.
            alpha: tint opacity.

        Returns:
            ``(H, W, 3)`` uint8 RGB image.
        """
        base = self._as_uint8(image01).copy()
        h, w = base.shape[:2]
        big = self._resize_nearest(np.asarray(mask, "float32"), (h, w)) > 0.5
        tint = np.asarray(color, dtype="float32")
        base[big] = (1.0 - alpha) * base[big] + alpha * tint
        return (np.clip(base, 0.0, 1.0) * 255.0).astype("uint8")
