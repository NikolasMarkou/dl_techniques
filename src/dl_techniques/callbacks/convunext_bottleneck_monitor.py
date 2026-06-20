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
and writes three PNGs for the first sample / fixed batch:

1. ``epoch_NNNN_bottleneck_first.png`` — a grid of the first ``min(C, 64)``
   bottleneck channels (fixed indices), drawn with a CONSTANT absolute color
   scale (``featuremap_vmin``/``vmax``) so colors are comparable across epochs.
2. ``epoch_NNNN_bottleneck_energy.png`` — a grid of the ``min(C, 64)`` highest
   mean-square-energy channels, recomputed each epoch, each tile titled with
   its channel index (same fixed color scale).
3. ``bottleneck_health.png`` — cumulative line charts of the five health
   statistics versus epoch, overwritten each monitored epoch and at train end.

The callback imports nothing from ``src/train/`` (HARD library-layering rule):
the trainer builds the fixed validation batch and passes it in. matplotlib is
lazy-imported inside the plotting method; callers should set ``MPLBACKEND=Agg``
for headless environments. The entire emit path is wrapped in a try/except so a
plotting or forward-pass failure can never raise into the training loop.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
        output_dir: Directory PNG files are written into directly (created if
            missing). The caller chooses the location (e.g. a shared
            ``visualizations/`` dir); no extra subdirectory is created.
        monitor_freq: Emit stats + PNG every this many epochs. Defaults to 5.
        max_featuremap_channels: Cap on the number of bottleneck channels tiled
            in each feature-map grid. Defaults to 64 (an 8x8 tile grid).
        featuremap_vmin: Lower bound of the FIXED absolute color scale applied
            to every feature-map tile at every epoch. Holding this constant
            across epochs makes a given color mean the same activation value
            over training (see D-001). Defaults to -3.0.
        featuremap_vmax: Upper bound of the fixed absolute color scale.
            Defaults to 3.0.
        name_prefix: Filename and log-message prefix. Defaults to
            ``"convunext_bottleneck"``.
    """

    def __init__(
        self,
        full_model: keras.Model,
        val_batch: Any,
        output_dir: Union[str, Path],
        monitor_freq: int = 5,
        max_featuremap_channels: int = 64,
        featuremap_vmin: float = -3.0,
        featuremap_vmax: float = 3.0,
        name_prefix: str = "convunext_bottleneck",
    ) -> None:
        super().__init__()
        # Store the FULL 2-output model explicitly: the callback attaches to the
        # single-output training view, so self.model would be the WRONG model.
        self.full_model = full_model
        self.val_batch = val_batch
        self.monitor_freq = monitor_freq
        self.max_featuremap_channels = max_featuremap_channels
        self.featuremap_vmin = featuremap_vmin
        self.featuremap_vmax = featuremap_vmax
        self.name_prefix = name_prefix
        self._subdir = Path(output_dir)
        self._subdir.mkdir(parents=True, exist_ok=True)

        # Cumulative health statistics, keyed by metric name -> list of values.
        # Accumulated only on integer-tagged monitored epochs (the ``'final'``
        # re-emit is de-duplicated) so the epoch x-axis stays monotone.
        self.history: Dict[str, List[float]] = {
            "mean_activation": [],
            "std_activation": [],
            "mean_l2_norm": [],
            "dead_unit_frac": [],
            "sparsity": [],
        }
        self.epochs_seen: List[int] = []

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
    # Statistics
    # -----------------------------------------------------------------
    def _compute_stats(self, feat: np.ndarray) -> Dict[str, float]:
        """Compute batch-aggregate bottleneck health statistics.

        Args:
            feat: Bottleneck latent ``(B, h, w, C)``.

        Returns:
            Dict of scalar statistics: ``mean_activation``, ``std_activation``,
            ``mean_l2_norm``, ``dead_unit_frac``, ``sparsity``. Names match the
            ``self.history`` keys so accumulation is a direct mapping.
        """
        n = feat.shape[0]
        flat = feat.reshape(n, -1)
        l2 = np.sqrt(np.sum(flat ** 2, axis=1))
        chan_mean_abs = np.mean(np.abs(feat), axis=(0, 1, 2))  # (C,)
        return {
            "mean_activation": float(feat.mean()),
            "std_activation": float(feat.std()),
            "mean_l2_norm": float(l2.mean()),
            "dead_unit_frac": float(np.mean(chan_mean_abs < 1e-6)),
            "sparsity": float(np.mean(np.abs(feat) < 1e-4)),
        }

    # -----------------------------------------------------------------
    # Feature-map grid (fixed color scale, titled tiles)
    # -----------------------------------------------------------------
    def _save_grid(
        self,
        sample: np.ndarray,
        indices: Any,
        tag: Union[int, str],
        suffix: str,
        title: str,
    ) -> None:
        """Tile the given channel indices of one sample as a heatmap grid.

        Args:
            sample: One bottleneck sample ``(h, w, C)`` (sample 0).
            indices: Channel indices to tile (list or numpy array).
            tag: Epoch number (1-based int) or the string ``'final'``.
            suffix: Filename suffix (``"first"`` or ``"energy"``).
            title: Figure super-title.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        k = len(indices)
        cols = int(np.ceil(np.sqrt(k)))
        rows = int(np.ceil(k / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
        axes = np.atleast_1d(axes).ravel()
        for i in range(len(axes)):
            ax = axes[i]
            if i < k:
                # DECISION plan_2026-06-20_0f7a354e/D-001: FIXED absolute color
                # scale (vmin/vmax constant across all epochs). Do NOT drop these
                # kwargs / let imshow autoscale per tile -- that per-epoch remap
                # is the exact "colors shift between iterations" bug (F2). See
                # decisions.md D-001.
                ax.imshow(
                    sample[:, :, indices[i]],
                    cmap="viridis",
                    vmin=self.featuremap_vmin,
                    vmax=self.featuremap_vmax,
                )
                ax.set_title(f"ch {int(indices[i])}", fontsize=7)
            ax.axis("off")

        tag_str = f"{tag:04d}" if isinstance(tag, int) else str(tag)
        fig.suptitle(title, fontsize=10)
        plt.tight_layout()
        out_path = self._subdir / f"epoch_{tag_str}_bottleneck_{suffix}.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    # -----------------------------------------------------------------
    # Health curves (cumulative)
    # -----------------------------------------------------------------
    def _save_health_curves(self) -> None:
        """Plot each cumulative health metric versus epoch (overwrites)."""
        import matplotlib.pyplot as plt  # noqa: PLC0415

        metrics = list(self.history.keys())
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.ravel()
        for idx, name in enumerate(metrics):
            ax = axes[idx]
            ax.plot(self.epochs_seen, self.history[name], marker="o", ms=3)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("epoch", fontsize=8)
            ax.grid(True, alpha=0.3)
        # Hide any unused panels.
        for idx in range(len(metrics), len(axes)):
            axes[idx].axis("off")
        fig.suptitle("ConvUNeXt bottleneck health over training", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            self._subdir / "bottleneck_health.png", dpi=120, bbox_inches="tight"
        )
        plt.close(fig)

    # -----------------------------------------------------------------
    # Emit (stats + two feature-map grids + health curves)
    # -----------------------------------------------------------------
    def _emit(self, tag: Union[int, str]) -> None:
        """Compute + log bottleneck stats and save grids + health curves.

        Args:
            tag: Epoch number (1-based int) or the string ``'final'``. Ints are
                zero-padded in the filename; ``'final'`` is used verbatim.
        """
        feat = self._get_bottleneck()  # (B, h, w, C)
        n = feat.shape[0]

        stats = self._compute_stats(feat)

        logger.info(
            f"{self.name_prefix} [{tag}]: shape={tuple(feat.shape)} "
            f"mean={stats['mean_activation']:.6f} "
            f"std={stats['std_activation']:.6f} "
            f"min={float(feat.min()):.6f} max={float(feat.max()):.6f} "
            f"mean_l2={stats['mean_l2_norm']:.4f} "
            f"dead_frac={stats['dead_unit_frac']:.4f} "
            f"sparsity={stats['sparsity']:.4f}"
        )

        # History accumulation: integer epochs only, and only when strictly
        # newer than the last seen epoch. This de-duplicates the ``'final'``
        # re-emit (and any repeat) so the curve x-axis stays monotone.
        if isinstance(tag, int) and (
            not self.epochs_seen or tag > self.epochs_seen[-1]
        ):
            for name, value in stats.items():
                self.history[name].append(value)
            self.epochs_seen.append(tag)

        sample = feat[0]  # (h, w, C)
        c = sample.shape[-1]
        k = int(min(c, self.max_featuremap_channels))
        tag_str = f"{tag:04d}" if isinstance(tag, int) else str(tag)

        # First grid: fixed channel indices 0..k-1 (temporal anchor).
        first_idx = list(range(k))

        # DECISION plan_2026-06-20_0f7a354e/D-002: top-energy channel set is
        # RECOMPUTED each epoch (not frozen at first emit). Cell positions move
        # between epochs by design -- the fixed color scale + the first-k grid
        # are the temporal anchors; this grid is the "what matters now" view.
        # Each tile is titled with its channel index so the moving identity
        # stays legible. See decisions.md D-002.
        energy = np.mean(feat ** 2, axis=(0, 1, 2))  # (C,)
        energy_idx = list(np.argsort(energy)[::-1][:k])

        self._save_grid(
            sample,
            first_idx,
            tag,
            "first",
            title=(
                f"ConvUNeXt bottleneck first {k}/{c} ch "
                f"(sample 0) - epoch {tag_str}"
            ),
        )
        self._save_grid(
            sample,
            energy_idx,
            tag,
            "energy",
            title=(
                f"ConvUNeXt bottleneck top-{k}/{c} energy ch "
                f"(sample 0) - epoch {tag_str}"
            ),
        )

        self._save_health_curves()

        logger.info(
            f"{self.name_prefix} [{tag}]: saved first/energy feature-map grids "
            f"+ health curves to {self._subdir}"
        )

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
