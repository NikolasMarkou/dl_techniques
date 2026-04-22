"""JEPA latent-masking visualization callbacks.

Keras callbacks for monitoring JEPA-style (latent-masked prediction) training:

- :class:`LatentMaskOverlayCallback` — saves input frames with the tube-mask
  tinted over masked patches at regular intervals.
- :class:`PatchPredictionErrorCallback` — saves a per-patch L2-error heatmap
  comparing ``predictor(z_masked)`` against the encoder target
  ``encode_frames(pixels)``.

Both are model-agnostic beyond the JEPA latent-masking contract: the model
must expose ``encode_frames``, ``predictor``, ``mask_gen``, ``mask_token``
and a ``config`` with ``patches_per_side`` / ``mask_prediction_enabled``
(see ``VideoJEPA``).

Both callbacks:

* lazy-import matplotlib — callers should set ``MPLBACKEND=Agg`` for
  headless environments;
* swallow all exceptions and log a warning via ``dl_techniques.utils.logger``
  so visualization failures never take down training;
* cache a fixed ``eval_pixels`` batch at construction time for cross-epoch
  visual stability.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, Optional

import keras
import keras.ops as ops
import numpy as np

from dl_techniques.utils.logger import logger


def _import_matplotlib():
    """Import matplotlib.pyplot with lazy loading."""
    import matplotlib.pyplot as plt
    return plt


def _normalize_pixels(eval_pixels: Any) -> np.ndarray:
    """Coerce ``eval_pixels`` to a numpy array of shape ``(B, T, H, W, C)``.

    Accepts rank-4 ``(B, H, W, C)`` — interpreted as a single-frame clip —
    or rank-5 ``(B, T, H, W, C)``. Everything else raises.
    """
    arr = np.asarray(eval_pixels)
    if arr.ndim == 4:
        arr = arr[:, None, ...]  # insert T=1 axis
    if arr.ndim != 5:
        raise ValueError(
            f"eval_pixels must be rank-4 or rank-5, got shape {arr.shape}"
        )
    return arr


def _to_display_image(frame: np.ndarray) -> np.ndarray:
    """Map a single-frame array of shape ``(H, W, C)`` into ``[0, 1]``.

    Tolerant of the common conventions used in this repo:
    * already in ``[0, 1]`` → identity;
    * in ``[-1, 1]`` → shift + scale;
    * in ``[0, 255]`` → divide.
    """
    f = frame.astype(np.float32)
    lo, hi = float(np.nanmin(f)), float(np.nanmax(f))
    if hi > 1.5 and hi <= 260.0:
        f = f / 255.0
    elif lo < -0.01:
        f = (f + 1.0) / 2.0
    return np.clip(f, 0.0, 1.0)


def _upsample_mask(mask_hp: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour upsample a 2-D mask ``(Hp, Wp)`` → ``(H, W)``."""
    Hp, Wp = mask_hp.shape
    H, W = target_hw
    rh = H // Hp
    rw = W // Wp
    if rh * Hp == H and rw * Wp == W:
        # exact integer upsample — fast path
        return np.repeat(np.repeat(mask_hp, rh, axis=0), rw, axis=1)
    # generic fallback
    ys = (np.arange(H) * Hp // H).clip(0, Hp - 1)
    xs = (np.arange(W) * Wp // W).clip(0, Wp - 1)
    return mask_hp[np.ix_(ys, xs)]


# =====================================================================
# LatentMaskOverlayCallback
# =====================================================================


class LatentMaskOverlayCallback(keras.callbacks.Callback):
    """Save input frames with the tube-mask tinted over masked patches.

    At the end of every *frequency* epochs, calls ``model.mask_gen`` once to
    sample a spatial mask, upsamples it to image resolution, and blends a
    coloured tint over the masked regions of the cached ``eval_pixels``.

    Args:
        eval_pixels: Numpy-compatible batch of shape ``(B, T, H, W, C)`` or
            ``(B, H, W, C)``.
        output_dir: Directory to write PNGs into. Created if missing.
        frequency: Write every *frequency* epochs (0-indexed). Defaults to 1.
        max_samples: Max rows in the grid. Defaults to 4.
        max_frames: Max columns (frames) in the grid. Defaults to 4.
        tint: RGB triple in ``[0, 1]`` used to tint masked pixels.
            Defaults to red ``(1.0, 0.0, 0.0)``.
        alpha: Blend weight for the tint. Defaults to 0.4.
    """

    def __init__(
        self,
        eval_pixels: Any,
        output_dir: str,
        frequency: int = 1,
        max_samples: int = 4,
        max_frames: int = 4,
        tint: tuple[float, float, float] = (1.0, 0.0, 0.0),
        alpha: float = 0.4,
    ) -> None:
        super().__init__()
        self.eval_pixels = _normalize_pixels(eval_pixels)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frequency = max(1, int(frequency))
        self.max_samples = int(max_samples)
        self.max_frames = int(max_frames)
        self.tint = tuple(float(c) for c in tint)
        self.alpha = float(alpha)
        self._disabled: bool = False

    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        try:
            cfg = getattr(self.model, "config", None)
            enabled = bool(getattr(cfg, "mask_prediction_enabled", True))
            if not enabled:
                self._disabled = True
                logger.warning(
                    "LatentMaskOverlayCallback disabled: "
                    "model.config.mask_prediction_enabled is False."
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"LatentMaskOverlayCallback on_train_begin error: {e}")
            self._disabled = True

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if self._disabled:
            return
        if epoch % self.frequency != 0:
            return

        try:
            B = self.eval_pixels.shape[0]
            mask_spatial = self.model.mask_gen(B, training=False)
            mask_np = np.asarray(mask_spatial).astype(np.float32)  # (B, Hp, Wp)
            self._save_grid(epoch, mask_np)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"LatentMaskOverlayCallback error at epoch {epoch}: {e}"
            )
        finally:
            gc.collect()

    def _save_grid(self, epoch: int, mask_np: np.ndarray) -> None:
        plt = _import_matplotlib()
        fig = None
        try:
            B, T, H, W, C = self.eval_pixels.shape
            n_samples = min(self.max_samples, B)
            n_frames = min(self.max_frames, T)

            fig, axes = plt.subplots(
                n_samples, n_frames,
                figsize=(2.0 * n_frames, 2.0 * n_samples),
                squeeze=False,
            )
            fig.suptitle(
                f"Latent Mask Overlay — Epoch {epoch}",
                fontsize=12, y=0.995,
            )

            tint = np.asarray(self.tint, dtype=np.float32).reshape(1, 1, 3)

            for i in range(n_samples):
                up_mask = _upsample_mask(mask_np[i], (H, W))  # (H, W)
                up_mask = up_mask[..., None]  # (H, W, 1)
                for t in range(n_frames):
                    img = _to_display_image(self.eval_pixels[i, t])
                    # Handle grayscale input by tiling to RGB for overlay
                    if img.shape[-1] == 1:
                        img = np.repeat(img, 3, axis=-1)
                    elif img.shape[-1] > 3:
                        img = img[..., :3]
                    blend = img * (1.0 - self.alpha * up_mask) + \
                            tint * (self.alpha * up_mask)
                    blend = np.clip(blend, 0.0, 1.0)
                    ax = axes[i, t]
                    ax.imshow(blend)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if t == 0:
                        ax.set_ylabel(f"s{i}", fontsize=8, rotation=0,
                                      ha="right", va="center")
                    if i == 0:
                        ax.set_title(f"t={t}", fontsize=8)

            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            path = self.output_dir / f"epoch_{epoch:03d}_mask_overlay.png"
            plt.savefig(str(path), dpi=120, bbox_inches="tight")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to save mask overlay grid: {e}")
        finally:
            if fig is not None:
                try:
                    plt.close(fig)
                except Exception:  # noqa: BLE001
                    pass


# =====================================================================
# PatchPredictionErrorCallback
# =====================================================================


class PatchPredictionErrorCallback(keras.callbacks.Callback):
    """Save a per-patch L2-error heatmap of JEPA latent prediction.

    At the end of every *frequency* epochs, runs ``encode_frames`` and
    ``predictor`` on the cached ``eval_pixels`` with ``training=False`` and
    plots the mean-squared error along the embedding axis for each
    ``(sample, frame)`` pair. If masking is enabled, optionally overlays a
    thin white border where ``mask == 1``.

    Args:
        eval_pixels: Numpy-compatible batch of shape ``(B, T, H, W, C)`` or
            ``(B, H, W, C)``.
        output_dir: Directory to write PNGs into. Created if missing.
        frequency: Write every *frequency* epochs (0-indexed). Defaults to 1.
        max_samples: Max rows in the grid. Defaults to 4.
        max_frames: Max columns (frames) in the grid. Defaults to 4.
    """

    def __init__(
        self,
        eval_pixels: Any,
        output_dir: str,
        frequency: int = 1,
        max_samples: int = 4,
        max_frames: int = 4,
    ) -> None:
        super().__init__()
        self.eval_pixels = _normalize_pixels(eval_pixels)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frequency = max(1, int(frequency))
        self.max_samples = int(max_samples)
        self.max_frames = int(max_frames)
        self._disabled: bool = False
        self._mask_enabled: bool = True

    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        try:
            cfg = getattr(self.model, "config", None)
            if cfg is None:
                self._disabled = True
                logger.warning(
                    "PatchPredictionErrorCallback disabled: model has no config."
                )
                return
            self._mask_enabled = bool(
                getattr(cfg, "mask_prediction_enabled", True)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"PatchPredictionErrorCallback on_train_begin error: {e}"
            )
            self._disabled = True

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if self._disabled:
            return
        if epoch % self.frequency != 0:
            return

        try:
            pixels = ops.convert_to_tensor(self.eval_pixels)
            z = self.model.encode_frames(pixels)  # (B, T, Hp, Wp, D)
            if self._mask_enabled:
                B = self.eval_pixels.shape[0]
                mask_spatial = self.model.mask_gen(B, training=False)  # (B,Hp,Wp)
                Hp = int(self.model.config.patches_per_side)
                D = int(self.model.config.embed_dim)
                M = ops.reshape(
                    ops.cast(mask_spatial, z.dtype),
                    (B, 1, Hp, Hp, 1),
                )
                token = ops.reshape(self.model.mask_token, (1, 1, 1, 1, D))
                z_masked = (1.0 - M) * z + M * token
                mask_np = np.asarray(mask_spatial).astype(np.float32)
            else:
                z_masked = z
                mask_np = None

            pred = self.model.predictor(z_masked, training=False)
            err = ops.mean(ops.square(pred - z), axis=-1)  # (B, T, Hp, Wp)
            err_np = np.asarray(err).astype(np.float32)

            self._save_grid(epoch, err_np, mask_np)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"PatchPredictionErrorCallback error at epoch {epoch}: {e}"
            )
        finally:
            gc.collect()

    def _save_grid(
        self,
        epoch: int,
        err_np: np.ndarray,
        mask_np: Optional[np.ndarray],
    ) -> None:
        plt = _import_matplotlib()
        fig = None
        try:
            B, T, Hp, Wp = err_np.shape
            n_samples = min(self.max_samples, B)
            n_frames = min(self.max_frames, T)

            # Shared colour scale across the visible slice for comparability.
            vmin = float(np.nanmin(err_np[:n_samples, :n_frames]))
            vmax = float(np.nanmax(err_np[:n_samples, :n_frames]))
            if vmax <= vmin:
                vmax = vmin + 1e-8

            fig, axes = plt.subplots(
                n_samples, n_frames,
                figsize=(2.0 * n_frames, 2.0 * n_samples),
                squeeze=False,
            )
            fig.suptitle(
                f"Patch Prediction Error (mean L2) — Epoch {epoch}",
                fontsize=12, y=0.995,
            )

            im = None
            for i in range(n_samples):
                for t in range(n_frames):
                    ax = axes[i, t]
                    im = ax.imshow(
                        err_np[i, t], cmap="viridis",
                        vmin=vmin, vmax=vmax,
                        interpolation="nearest",
                    )
                    if mask_np is not None:
                        # Outline masked patches with a thin white border.
                        mi = mask_np[i]  # (Hp, Wp)
                        ys, xs = np.where(mi > 0.5)
                        for y, x in zip(ys, xs):
                            ax.add_patch(plt.Rectangle(
                                (x - 0.5, y - 0.5), 1, 1,
                                fill=False, edgecolor="white",
                                linewidth=0.5,
                            ))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if t == 0:
                        ax.set_ylabel(f"s{i}", fontsize=8, rotation=0,
                                      ha="right", va="center")
                    if i == 0:
                        ax.set_title(f"t={t}", fontsize=8)

            if im is not None:
                cbar = fig.colorbar(
                    im, ax=axes.ravel().tolist(),
                    shrink=0.7, pad=0.02,
                )
                cbar.ax.tick_params(labelsize=7)

            path = self.output_dir / f"epoch_{epoch:03d}_patch_error.png"
            plt.savefig(str(path), dpi=120, bbox_inches="tight")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to save patch error heatmap: {e}")
        finally:
            if fig is not None:
                try:
                    plt.close(fig)
                except Exception:  # noqa: BLE001
                    pass
