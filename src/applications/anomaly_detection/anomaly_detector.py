"""Patch-entropy anomaly detector built on ``ConvNeXtPatchVAE``.

The detector runs the trained VAE's **encoder only** (the decoder is never
executed) and scores every patch by its KL divergence — the number of nats the
encoder spends to describe that patch beyond what the prior predicts. Patches
with high KL are "surprising" / high-entropy and flagged as anomalous.

The single-scale model produces one per-patch latent grid
``z(B, Hp, Wp, latent_dim)``; the anomaly signal is the standard per-patch KL
against ``N(0, I)``: ``KL(q(z|x) || N(0, I))`` summed over the latent dimension
→ an ``(Hp, Wp)`` map (e.g. ``32x32`` for a 128px image at ``patch_size=4``).

This module deliberately has **no GUI dependency** — it stays importable and
usable headless. The Streamlit front-end lives in ``streamlit_app.py``.

Example::

    from applications.anomaly_detection.anomaly_detector import (
        PatchEntropyAnomalyDetector,
    )

    det = PatchEntropyAnomalyDetector.from_pretrained("results/.../best_model.keras")
    x, (h, w) = det.preprocess("photo.jpg")        # native res, padded to /patch
    kl = det.kl_maps(x, orig_hw=(h, w))["kl"]      # (Hp, Wp)
    mask, thr = det.anomaly_mask(kl)               # boolean (Hp, Wp)
    overlay = det.overlay(x[0][:h, :w], kl)        # uint8 (h, w, 3)
"""

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import keras
from keras import ops
import matplotlib

from dl_techniques.utils.logger import logger

ImageLike = Union[str, np.ndarray]

# Mirror the training-time numerical guard: log_var is clipped to [-10, +10]
# in every KL computation inside the model (float32 stability, H18 fix).
_DEFAULT_LOG_VAR_CLIP = 10.0


class PatchEntropyAnomalyDetector:
    """Encoder-only, KL-divergence per-patch anomaly detector.

    Args:
        model: A loaded ``ConvNeXtPatchVAE`` instance.
        log_var_clip: Symmetric clip applied to ``log_var`` before KL math.
            Must match the model's internal clip (default ``10.0``).
    """

    def __init__(
        self,
        model: keras.Model,
        log_var_clip: float = _DEFAULT_LOG_VAR_CLIP,
    ) -> None:
        self.model = model
        self.log_var_clip = float(log_var_clip)
        cfg = getattr(model, "config", None)
        self.patch_size = int(getattr(cfg, "patch_size", 4))
        self.recon_loss_type = str(getattr(cfg, "recon_loss_type", "bce"))
        self.default_image_size = int(getattr(cfg, "img_size", 128) or 128)
        logger.info(
            "PatchEntropyAnomalyDetector ready: KL vs N(0, I), "
            "patch_size=%d, recon_loss=%s, log_var_clip=%.1f",
            self.patch_size,
            self.recon_loss_type,
            self.log_var_clip,
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        log_var_clip: float = _DEFAULT_LOG_VAR_CLIP,
    ) -> "PatchEntropyAnomalyDetector":
        """Load a trained checkpoint and wrap it in a detector.

        Args:
            model_path: Path to the ``.keras`` checkpoint.
            log_var_clip: See :class:`PatchEntropyAnomalyDetector`.

        Returns:
            A ready detector.

        Raises:
            Exception: re-raised after logging if ``load_model`` fails.
        """
        # Import so the @register_keras_serializable classes resolve by name.
        from dl_techniques.models.convnext_patch_vae.model import (
            ConvNeXtPatchVAE,
        )

        custom_objects = {
            "ConvNeXtPatchVAE": ConvNeXtPatchVAE,
        }
        logger.info("Loading anomaly-detection model from %s", model_path)
        try:
            model = keras.models.load_model(
                model_path, custom_objects=custom_objects, compile=False
            )
        except Exception as exc:  # noqa: BLE001 - log then re-raise
            logger.error("Failed to load model %s: %s", model_path, exc)
            raise
        return cls(model, log_var_clip=log_var_clip)

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
    # KL math (mirrors model._compute_kl exactly)
    # ------------------------------------------------------------------
    def _kl_standard(self, mu: Any, log_var: Any) -> Any:
        """Per-patch KL against ``N(0, I)``; sum over latent dim -> (B, Hp, Wp)."""
        mu_f = ops.cast(mu, "float32")
        lv = ops.clip(
            ops.cast(log_var, "float32"), -self.log_var_clip, self.log_var_clip
        )
        return -0.5 * ops.sum(
            1.0 + lv - ops.square(mu_f) - ops.exp(lv), axis=-1
        )

    def kl_maps(
        self, x: np.ndarray, orig_hw: Optional[Tuple[int, int]] = None
    ) -> Dict[str, np.ndarray]:
        """Per-patch KL "surprise" map for one image.

        Args:
            x: A ``(1, H', W', 3)`` float32 batch in ``[0, 1]`` from
                :meth:`preprocess` (spatial dims multiples of ``patch_size``).
            orig_hw: Optional unpadded ``(H, W)`` (from :meth:`preprocess`).
                When given, the map is cropped to its valid patch region
                (``ceil(dim / patch_size)``) so reflect-padded patches do not
                pollute scores or overlays.

        Returns:
            ``{"kl": (Hp, Wp)}`` numpy float32 array — the per-patch KL against
            ``N(0, I)`` (decoder is NOT executed).
        """
        mu, log_var = self.model.encode(x)
        kl = self._kl_standard(mu, log_var)
        kl_np = np.asarray(kl, dtype="float32")[0]

        if orig_hw is not None:
            h, w = orig_hw
            kl_np = kl_np[
                : math.ceil(h / self.patch_size),
                : math.ceil(w / self.patch_size),
            ]

        return {"kl": kl_np}

    # ------------------------------------------------------------------
    # Thresholding / scoring
    # ------------------------------------------------------------------
    def anomaly_mask(
        self,
        kl_map: np.ndarray,
        method: str = "zscore",
        k: float = 3.0,
        percentile: float = 95.0,
        abs_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """Threshold a KL map into a boolean anomaly mask.

        Args:
            kl_map: A per-patch KL map ``(Hp, Wp)``.
            method: ``"zscore"`` (``mean + k*std``), ``"percentile"``, or
                ``"absolute"`` (requires ``abs_threshold``).
            k: z-score multiplier for ``"zscore"``.
            percentile: percentile (0-100) for ``"percentile"``.
            abs_threshold: nats cut-off for ``"absolute"``.

        Returns:
            ``(mask, threshold)`` — boolean array matching ``kl_map`` and the
            scalar threshold used.

        Raises:
            ValueError: for an unknown method or missing ``abs_threshold``.
        """
        m = np.asarray(kl_map, dtype="float32")
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
        self, kl_map: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Scalar summaries of a KL map (and optional mask).

        Args:
            kl_map: A per-patch KL map ``(Hp, Wp)``.
            mask: Optional boolean mask for anomalous-patch counts.

        Returns:
            Dict with ``mean_kl, max_kl, p95_kl`` and, if ``mask`` given,
            ``n_anomalous`` and ``frac_anomalous``.
        """
        m = np.asarray(kl_map, dtype="float32")
        out: Dict[str, float] = {
            "mean_kl": float(m.mean()),
            "max_kl": float(m.max()),
            "p95_kl": float(np.percentile(m, 95.0)),
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
        kl_map: np.ndarray,
        cmap: str = "inferno",
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Alpha-blend a colormapped KL heatmap over the image.

        Args:
            image01: ``(H, W, 3)`` image (``[0, 1]`` or ``[0, 255]``).
            kl_map: per-patch KL map ``(Hp, Wp)``.
            cmap: matplotlib colormap name.
            alpha: heatmap opacity in ``[0, 1]``.

        Returns:
            ``(H, W, 3)`` uint8 RGB image.
        """
        base = self._as_uint8(image01)
        h, w = base.shape[:2]
        big = self._resize_nearest(np.asarray(kl_map, "float32"), (h, w))
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
