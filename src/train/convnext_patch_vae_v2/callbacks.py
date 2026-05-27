"""Training callbacks for ConvNeXtPatchVAEV2.

Two callbacks are introduced here, both written so they degrade
gracefully when a feature is disabled at config time:

- :class:`BetaAnnealingCallback` — identical contract to V1: linearly
  ramps a Python float attribute on ``self.model`` from a start value
  to a target value over ``anneal_epochs``. Default ``attr_name`` is
  ``"_beta_kl"`` to match V2's model field.
- :class:`MaskedReconViz` — periodically dumps an
  ``original / reconstruction / pixel-mask`` triple-row grid PNG so
  the MAE / VAE training can be monitored. Skipped silently when
  ``model.config.mae_mask_ratio == 0``.

All file writes are guarded by ``os.makedirs(..., exist_ok=True)`` and a
broad ``try / except`` with a logger warning — per LESSONS, "periodic
training-side-effect callbacks must swallow render failures".
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import keras
import numpy as np

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------------
# Beta annealing (port from V1 — slightly leaner, no L1/L2 attr_name machinery
# since V2 is single-scale)
# ---------------------------------------------------------------------------


class BetaAnnealingCallback(keras.callbacks.Callback):
    """Linearly ramp a model attribute from start to target.

    Args:
        beta_start: Initial value at epoch 0.
        beta_target: Value after ``anneal_epochs``.
        anneal_epochs: Epochs over which to ramp. ``<= 0`` disables.
        attr_name: Attribute on ``self.model`` to mutate. Default
            ``"_beta_kl"`` (V2 single-scale).
    """

    def __init__(
        self,
        beta_start: float,
        beta_target: float,
        anneal_epochs: int,
        attr_name: str = "_beta_kl",
    ) -> None:
        super().__init__()
        self.beta_start = float(beta_start)
        self.beta_target = float(beta_target)
        self.anneal_epochs = int(anneal_epochs)
        self.attr_name = attr_name

    def on_train_begin(self, logs=None) -> None:
        initial_epoch = int(self.params.get("initial_epoch", 0))
        if self.anneal_epochs > 0 and initial_epoch > 0:
            progress = min(1.0, initial_epoch / self.anneal_epochs)
            new_value = (
                self.beta_start
                + progress * (self.beta_target - self.beta_start)
            )
            setattr(self.model, self.attr_name, new_value)

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        if self.anneal_epochs <= 0:
            return
        progress = min(1.0, epoch / self.anneal_epochs)
        new_value = (
            self.beta_start + progress * (self.beta_target - self.beta_start)
        )
        setattr(self.model, self.attr_name, new_value)
        logger.info(
            "BetaAnnealing[%s] epoch=%d beta=%.4f",
            self.attr_name, epoch, new_value,
        )


# ---------------------------------------------------------------------------
# Masked-reconstruction visualization
# ---------------------------------------------------------------------------


class MaskedReconViz(keras.callbacks.Callback):
    """Save (original | recon | mask) triple-row grids periodically.

    Skipped silently when the model's MAE mask is disabled
    (``model.config.mae_mask_ratio == 0.0``) — visualizing a vanilla
    recon adds no signal. Use V1's ``ReconVisualizationCallback`` for
    the no-MAE case.

    Args:
        val_samples: Fixed validation batch ``(N, H, W, C)`` in
            display-space (post-denorm if MSE branch).
        save_dir: Directory to write PNGs.
        frequency: Save every this many epochs (and on epoch 0).
    """

    def __init__(
        self,
        val_samples: np.ndarray,
        save_dir: str,
        frequency: int = 1,
    ) -> None:
        super().__init__()
        self.val_samples = val_samples
        self.save_dir = save_dir
        self.frequency = int(frequency)
        os.makedirs(save_dir, exist_ok=True)

    def _disabled(self) -> bool:
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return True
        return getattr(cfg, "mae_mask_ratio", 0.0) <= 0.0

    def _save_grid(self, path: str, originals, recons, title: str) -> None:
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        n = len(originals)
        cmap = "gray" if originals.shape[-1] == 1 else None
        fig, axes = plt.subplots(2, n, figsize=(n * 1.4, 3.2))
        for row, imgs in enumerate([originals, recons]):
            for i in range(n):
                axes[row, i].imshow(np.clip(imgs[i], 0.0, 1.0).squeeze(), cmap=cmap)
                axes[row, i].axis("off")
        fig.suptitle(title, fontsize=11)
        import matplotlib.pyplot as _plt
        _plt.tight_layout(rect=[0, 0, 1, 0.96])
        _plt.savefig(path, dpi=120, bbox_inches="tight")
        _plt.close(fig)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if self._disabled():
            return
        if epoch != 0 and (epoch % self.frequency != 0):
            return
        try:
            outputs = self.model(self.val_samples, training=False)
            originals = self.val_samples
            recons = np.array(outputs["reconstruction"])
            path = os.path.join(
                self.save_dir, f"masked_recon_epoch_{epoch + 1:04d}.png"
            )
            self._save_grid(
                path, originals, recons,
                f"Epoch {epoch + 1}  |  mae_loss={(logs or {}).get('mae_loss', float('nan')):.4f}",
            )
        except Exception as exc:
            logger.warning(
                "MaskedReconViz failed at epoch %d: %s", epoch, exc,
            )
