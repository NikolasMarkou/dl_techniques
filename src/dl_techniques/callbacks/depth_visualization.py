"""
Depth estimation visualization callbacks.

Keras callbacks for monitoring monocular depth estimation training:

- :class:`DepthPredictionGridCallback` — saves RGB | GT depth | predicted
  depth comparison grids at regular intervals.
- :class:`DepthMetricsCurveCallback` — plots training/validation metric
  curves (loss, AbsRel, delta accuracy) during training.

Both callbacks write PNG images to a configurable output directory and
are model-agnostic — they work with any depth estimator that takes an
RGB tensor and outputs a single-channel depth map.
"""

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np

from dl_techniques.utils.logger import logger

# Lazy-import matplotlib to avoid import-time side effects.
# Callers should set MPLBACKEND=Agg before launching training.


def _import_matplotlib():
    """Import matplotlib.pyplot with lazy loading."""
    import matplotlib.pyplot as plt
    return plt


# =====================================================================
# DepthPredictionGridCallback
# =====================================================================


class DepthPredictionGridCallback(keras.callbacks.Callback):
    """Save RGB | GT depth | predicted depth comparison grids.

    At the end of every *frequency* epochs, runs the model on a fixed
    set of validation samples and saves a 3-row comparison grid:

    - Row 1: RGB input (de-normalized to ``[0, 1]``)
    - Row 2: Ground-truth depth (invalid pixels shown in gray)
    - Row 3: Predicted depth

    The callback is model-agnostic — it calls ``self.model(rgb)`` and
    expects a single-channel depth output.

    Args:
        val_rgb: Validation RGB tensor ``(N, H, W, 3)`` in ``[-1, 1]``.
        val_depth: Validation depth tensor ``(N, H, W, 1)`` in ``[-1, 1]``.
        val_mask: Validation mask tensor ``(N, H, W, 1)`` with 1=valid.
        output_dir: Directory for saving grid PNGs.
        frequency: Save every *frequency* epochs.  Defaults to 5.
        max_samples: Maximum number of samples in the grid.
            Defaults to 8.
        title: Title prefix for the grid.  Defaults to
            ``"Depth Estimation"``.
    """

    def __init__(
        self,
        val_rgb: Any,
        val_depth: Any,
        val_mask: Any,
        output_dir: str,
        frequency: int = 5,
        max_samples: int = 8,
        title: str = "Depth Estimation",
    ) -> None:
        super().__init__()
        self.val_rgb = val_rgb
        self.val_depth = val_depth
        self.val_mask = val_mask
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frequency = frequency
        self.max_samples = max_samples
        self.title = title

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if (epoch + 1) % self.frequency != 0:
            return
        if self.val_rgb is None:
            return

        try:
            pred_depth = self.model(self.val_rgb, training=False)

            # Log masked MSE for quick monitoring
            import keras.ops as ops
            valid_count = ops.sum(self.val_mask)
            if valid_count > 0:
                masked_mse = (
                    ops.sum(
                        ops.square(pred_depth - self.val_depth) * self.val_mask
                    ) / valid_count
                )
                logger.info(
                    f"Epoch {epoch + 1} depth monitor — "
                    f"masked MSE: {float(masked_mse):.6f}"
                )

            self._save_grid(epoch + 1, pred_depth)

            del pred_depth
            gc.collect()
        except Exception as e:
            logger.warning(
                f"Depth grid callback error at epoch {epoch + 1}: {e}"
            )

    def _save_grid(
        self,
        epoch: int,
        pred_depth: Any,
    ) -> None:
        """Render and save a 3-row comparison grid."""
        plt = _import_matplotlib()
        try:
            n = min(self.max_samples, self.val_rgb.shape[0])
            fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 7.5))
            fig.suptitle(
                f"{self.title} — Epoch {epoch}",
                fontsize=16, y=0.98,
            )

            # Handle single-sample edge case (axes shape differs)
            if n == 1:
                axes = axes[:, np.newaxis]

            rgb_np = np.array(self.val_rgb)
            gt_np = np.array(self.val_depth)
            mask_np = np.array(self.val_mask)
            pred_np = np.array(pred_depth)

            for i in range(n):
                # RGB: [-1, 1] → [0, 1]
                rgb_img = np.clip((rgb_np[i] + 1.0) / 2.0, 0, 1)

                # GT depth: [-1, 1] → [0, 1], gray for invalid
                gt = gt_np[i].squeeze(-1)
                gt_vis = np.clip((gt + 1.0) / 2.0, 0, 1)
                m = mask_np[i].squeeze(-1)
                gt_vis = np.where(m > 0, gt_vis, 0.5)

                # Predicted depth: [-1, 1] → [0, 1]
                pred = pred_np[i].squeeze(-1)
                pred_vis = np.clip((pred + 1.0) / 2.0, 0, 1)

                labels = ["RGB", "GT Depth", "Predicted"]
                images = [rgb_img, gt_vis, pred_vis]
                cmaps = [None, "viridis", "viridis"]

                for row, (img, cmap) in enumerate(zip(images, cmaps)):
                    axes[row, i].imshow(img, cmap=cmap, vmin=0, vmax=1)
                    if i == 0:
                        axes[row, i].set_ylabel(
                            labels[row], fontsize=12, rotation=0,
                            ha="right", va="center",
                        )
                    axes[row, i].axis("off")

            plt.tight_layout()
            plt.subplots_adjust(top=0.92, left=0.10)
            path = self.output_dir / f"epoch_{epoch:03d}_depth.png"
            plt.savefig(str(path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to save depth grid: {e}")


# =====================================================================
# DepthMetricsCurveCallback
# =====================================================================


class DepthMetricsCurveCallback(keras.callbacks.Callback):
    """Plot training/validation metric curves during training.

    Collects specified metric values from the Keras ``logs`` dict at the
    end of each epoch and periodically saves a multi-panel plot showing
    the evolution of each metric over training.

    Args:
        output_dir: Directory for saving curve PNGs.
        train_metrics: List of training metric keys to track
            (e.g. ``["loss", "abs_rel", "delta_1.25"]``).
        val_metrics: List of validation metric keys to track
            (e.g. ``["val_loss", "val_abs_rel", "val_delta_1.25"]``).
            If ``None``, auto-generated by prepending ``"val_"`` to
            each training metric.
        frequency: Save plots every *frequency* epochs.  Defaults to 5.
    """

    def __init__(
        self,
        output_dir: str,
        train_metrics: Optional[List[str]] = None,
        val_metrics: Optional[List[str]] = None,
        frequency: int = 5,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frequency = frequency

        if train_metrics is None:
            train_metrics = ["loss", "abs_rel", "delta_1.25"]
        if val_metrics is None:
            val_metrics = [f"val_{m}" for m in train_metrics]

        self.train_keys = list(train_metrics)
        self.val_keys = list(val_metrics)
        self.history: Dict[str, List[float]] = {
            k: [] for k in self.train_keys + self.val_keys
        }

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if logs is None:
            logs = {}

        for key in self.history:
            if key in logs:
                self.history[key].append(float(logs[key]))

        if (epoch + 1) % self.frequency == 0 or epoch == 0:
            self._create_plots(epoch + 1)

    def _create_plots(self, epoch: int) -> None:
        """Render metric curves as a multi-panel figure."""
        plt = _import_matplotlib()
        try:
            # Pair train/val metrics for plotting
            pairs = list(zip(self.train_keys, self.val_keys))
            n_panels = len(pairs)
            if n_panels == 0:
                return

            fig, axes = plt.subplots(
                1, n_panels, figsize=(5 * n_panels, 4), squeeze=False,
            )

            for idx, (train_key, val_key) in enumerate(pairs):
                ax = axes[0, idx]
                train_vals = self.history.get(train_key, [])
                val_vals = self.history.get(val_key, [])
                epochs_range = range(1, len(train_vals) + 1)

                if train_vals:
                    ax.plot(epochs_range, train_vals, label="train")
                if val_vals:
                    val_epochs = range(1, len(val_vals) + 1)
                    ax.plot(val_epochs, val_vals, label="val")

                ax.set_title(train_key)
                ax.set_xlabel("Epoch")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            fig.suptitle(f"Training Metrics — Epoch {epoch}", fontsize=14)
            plt.tight_layout()
            path = self.output_dir / f"epoch_{epoch:03d}_metrics.png"
            plt.savefig(str(path), dpi=100, bbox_inches="tight")
            plt.close(fig)
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to create metric curve plots: {e}")
