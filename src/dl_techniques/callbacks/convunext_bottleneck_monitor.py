"""Bottleneck monitoring callback for the bias-free ConvUNeXt denoiser.

Provides :class:`ConvUnextBottleneckMonitorCallback`, a Keras callback that
inspects the deepest-stage latent (the bottleneck) of a bias-free ConvUNeXt
denoiser built with ``expose_bottleneck=True`` over a *fixed* validation batch
every ``monitor_freq`` epochs (and once again at the end of training).

When ``expose_bottleneck=True``, ``create_convunext_denoiser`` returns a model
whose call yields ``[denoised, ..., bottleneck]`` (the bottleneck is the
trailing output by factory contract). The trainer fits a single-output training
*view* that shares weights with this full model, so a callback that attached via
``self.model`` would see the WRONG (single-output) model. To avoid that, the
FULL 2-output model is passed explicitly into the constructor and stored as
``self.full_model``; the callback reads the bottleneck from it directly.

Each monitored epoch the callback logs latent health statistics (mean / std /
min / max, mean per-sample L2 norm, dead-channel fraction, activation sparsity)
and writes one lightweight feature-map grid PNG of the bottleneck channels for
the first sample.

The callback imports nothing from ``src/train/`` (HARD library-layering rule):
the trainer builds the fixed validation batch and passes it in. matplotlib is
lazy-imported inside the plotting method; callers should set ``MPLBACKEND=Agg``
for headless environments. The entire emit path is wrapped in a try/except so a
plotting or forward-pass failure can never raise into the training loop.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import keras
import numpy as np

from dl_techniques.utils.logger import logger


class ConvUnextBottleneckMonitorCallback(keras.callbacks.Callback):
    """Monitor the ConvUNeXt denoiser bottleneck over a fixed val batch.

    Reads the bottleneck latent (the trailing output of an
    ``expose_bottleneck=True`` ConvUNeXt denoiser) from an explicitly-passed
    full 2-output model over a fixed validation batch every ``monitor_freq``
    epochs and once at the end of training. For each monitored point it logs
    aggregate latent health statistics and writes a per-channel feature-map grid
    PNG for the first sample. All emit work is wrapped in a try/except and never
    propagates into the training loop.

    Args:
        full_model: The FULL 2-output (or DS-list + bottleneck) ConvUNeXt model
            whose trailing output is the bottleneck. This must be the full model,
            NOT the single-output training view the callback attaches to.
        val_batch: Fixed clean validation batch ``(B, H, W, C)`` (tensor or numpy
            array) in model-input space. Used for a forward pass each monitored
            epoch.
        output_dir: Base directory for PNG files. A ``<name_prefix>/``
            subdirectory is created beneath it.
        monitor_freq: Emit stats + PNG every this many epochs. Defaults to 5.
        max_featuremap_channels: Cap on the number of bottleneck channels tiled
            in the feature-map grid. Defaults to 16.
        name_prefix: Subdirectory name and filename/log prefix. Defaults to
            ``"convunext_bottleneck"``.
    """

    def __init__(
        self,
        full_model: keras.Model,
        val_batch: Any,
        output_dir: Union[str, Path],
        monitor_freq: int = 5,
        max_featuremap_channels: int = 16,
        name_prefix: str = "convunext_bottleneck",
    ) -> None:
        super().__init__()
        # Store the FULL 2-output model explicitly: the callback attaches to the
        # single-output training view, so self.model would be the WRONG model.
        self.full_model = full_model
        self.val_batch = val_batch
        self.monitor_freq = monitor_freq
        self.max_featuremap_channels = max_featuremap_channels
        self.name_prefix = name_prefix
        self._subdir = Path(output_dir) / name_prefix
        self._subdir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Bottleneck extraction
    # -----------------------------------------------------------------
    def _get_bottleneck(self) -> np.ndarray:
        """Forward the fixed batch and return the bottleneck latent as numpy.

        The full model's call returns tensors positionally; by the factory
        contract the bottleneck is the LAST output, so we take ``out[-1]``.

        Returns:
            Bottleneck latent ``(B, h, w, C)`` as a numpy array.
        """
        out = self.full_model(self.val_batch, training=False)
        # Single-output models return a bare tensor; multi-output return a
        # list/tuple. The bottleneck is the trailing output by factory contract.
        if isinstance(out, (list, tuple)):
            bottleneck = out[-1]
        else:
            bottleneck = out
        return np.array(bottleneck)

    # -----------------------------------------------------------------
    # Emit (stats + feature-map grid)
    # -----------------------------------------------------------------
    def _emit(self, tag: Union[int, str]) -> None:
        """Compute + log bottleneck stats and save a feature-map grid PNG.

        Args:
            tag: Epoch number (1-based int) or the string ``'final'``. Ints are
                zero-padded in the filename; ``'final'`` is used verbatim.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        feat = self._get_bottleneck()  # (B, h, w, C)
        n = feat.shape[0]

        # Aggregate health statistics.
        flat = feat.reshape(n, -1)
        l2 = np.sqrt(np.sum(flat ** 2, axis=1))
        chan_mean_abs = np.mean(np.abs(feat), axis=(0, 1, 2))  # (C,)
        dead_frac = float(np.mean(chan_mean_abs < 1e-6))
        sparsity = float(np.mean(np.abs(feat) < 1e-4))

        logger.info(
            f"{self.name_prefix} [{tag}]: shape={tuple(feat.shape)} "
            f"mean={float(feat.mean()):.6f} std={float(feat.std()):.6f} "
            f"min={float(feat.min()):.6f} max={float(feat.max()):.6f} "
            f"mean_l2={float(l2.mean()):.4f} dead_frac={dead_frac:.4f} "
            f"sparsity={sparsity:.4f}"
        )

        # Feature-map grid for the first sample, up to max_featuremap_channels.
        sample = feat[0]  # (h, w, C)
        h, w, c = sample.shape
        k = int(min(c, self.max_featuremap_channels))
        cols = int(np.ceil(np.sqrt(k)))
        rows = int(np.ceil(k / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
        axes = np.atleast_1d(axes).ravel()
        for i in range(len(axes)):
            ax = axes[i]
            if i < k:
                ax.imshow(sample[:, :, i], cmap="viridis")
                ax.set_title(f"ch {i}", fontsize=7)
            ax.axis("off")

        tag_str = f"{tag:04d}" if isinstance(tag, int) else str(tag)
        fig.suptitle(
            f"ConvUNeXt bottleneck feature maps "
            f"(sample 0, {k}/{c} ch) - epoch {tag_str}",
            fontsize=10,
        )
        plt.tight_layout()
        out_path = self._subdir / f"epoch_{tag_str}_bottleneck.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"{self.name_prefix} [{tag}]: saved feature-map grid to {out_path}")

    # -----------------------------------------------------------------
    # Keras hooks
    # -----------------------------------------------------------------
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Emit bottleneck stats + PNG on monitored epochs.

        Args:
            epoch: 0-based epoch index from Keras.
            logs: Keras logs dict (unused).
        """
        if (epoch + 1) % self.monitor_freq != 0:
            return
        try:
            self._emit(epoch + 1)
        except Exception as exc:
            logger.warning(
                f"{self.name_prefix}: monitor failed at epoch {epoch + 1}: {exc}",
                exc_info=True,
            )

    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        """Re-emit bottleneck stats + PNG at the end of training."""
        try:
            self._emit("final")
        except Exception as exc:
            logger.warning(
                f"{self.name_prefix}: monitor failed on train end: {exc}",
                exc_info=True,
            )
