"""Bottleneck monitoring callback for the CliffordLaplacianUNet autoencoder.

Provides :class:`CliffordBottleneckMonitorCallback`, a Keras callback that
inspects the deepest-stage latent (the bottleneck) of a
``CliffordLaplacianUNet`` over a *fixed* validation batch every
``monitor_freq`` epochs and writes four diagnostic PNGs:

1. ``bottleneck_health.png`` — cumulative line charts of latent health
   statistics (mean / std activation, mean L2 norm, dead-unit fraction,
   sparsity) versus epoch. Overwritten each monitored epoch.
2. ``bottleneck_featuremap_epoch_NNNN.png`` — a grid of grayscale
   per-channel feature maps for one fixed sample.
3. ``bottleneck_pca_epoch_NNNN.png`` — a 2-D PCA scatter of the flattened
   latent, colored by per-sample reconstruction MSE.
4. ``bottleneck_histogram_epoch_NNNN.png`` — a histogram of all latent
   activations.

The callback acquires the model via ``self.model`` and calls the public
``model.encode(batch, training=False)`` accessor. It imports nothing from
``src/train/`` (HARD library-layering rule): the trainer builds the fixed
validation batch and passes it into the constructor.

matplotlib and sklearn are lazy-imported inside the plotting methods;
callers should set ``MPLBACKEND=Agg`` for headless environments. The entire
``on_epoch_end`` body is wrapped in a try/except so a plotting failure can
never raise into the training loop.
"""

import os
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import keras
import numpy as np

from dl_techniques.utils.logger import logger
from dl_techniques.utils.visualization import collage


class CliffordBottleneckMonitorCallback(keras.callbacks.Callback):
    """Monitor the CliffordLaplacianUNet bottleneck over a fixed val batch.

    Every ``monitor_freq`` epochs (gated on ``(epoch + 1) % monitor_freq``),
    encodes a fixed validation batch through ``model.encode`` and emits four
    PNGs (health curves, per-channel feature-map tiles, PCA scatter colored by
    per-sample reconstruction MSE, and an activation histogram). All plotting
    is wrapped in a try/except and never propagates into training.

    Args:
        val_batch: Fixed validation batch ``(N, H, W, C)`` (tensor or numpy
            array) in model-input space, or ``None``. When ``None`` the
            callback no-ops with a logged warning.
        output_dir: Base directory for PNG files. A ``bottleneck/``
            subdirectory is created beneath it.
        monitor_freq: Emit plots every this many epochs. Defaults to 5.
        max_featuremap_channels: Cap on the number of latent channels tiled
            in the feature-map grid. Defaults to 64.
    """

    def __init__(
        self,
        val_batch: Optional[Any],
        output_dir: Union[str, Path],
        monitor_freq: int = 5,
        max_featuremap_channels: int = 64,
    ) -> None:
        super().__init__()
        self.val_batch = val_batch
        self.monitor_freq = monitor_freq
        self.max_featuremap_channels = max_featuremap_channels
        self.output_dir = Path(output_dir) / "bottleneck"
        os.makedirs(self.output_dir, exist_ok=True)

        # Cumulative health statistics, keyed by metric name -> list of values.
        self.history: Dict[str, List[float]] = {
            "mean_activation": [],
            "std_activation": [],
            "mean_l2_norm": [],
            "dead_unit_frac": [],
            "sparsity": [],
        }
        self.epochs_seen: List[int] = []

    # -----------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------
    def _compute_health_stats(self, feat: np.ndarray) -> Dict[str, float]:
        """Compute batch-aggregate latent health statistics.

        Args:
            feat: Bottleneck latent ``(N, h, w, C)``.

        Returns:
            Dict of scalar statistics: ``mean_activation``, ``std_activation``,
            ``mean_l2_norm``, ``dead_unit_frac``, ``sparsity``.
        """
        n = feat.shape[0]
        flat = feat.reshape(n, -1)
        l2 = np.sqrt(np.sum(flat ** 2, axis=1))
        # Per-channel max-abs across batch + spatial dims.
        chan_max_abs = np.max(np.abs(feat), axis=(0, 1, 2))
        return {
            "mean_activation": float(feat.mean()),
            "std_activation": float(feat.std()),
            "mean_l2_norm": float(l2.mean()),
            "dead_unit_frac": float(np.mean(chan_max_abs < 1e-6)),
            "sparsity": float(np.mean(np.abs(feat) < 1e-3)),
        }

    # -----------------------------------------------------------------
    # Plot 1: health curves (cumulative)
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
        fig.suptitle("Bottleneck health over training", fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / "bottleneck_health.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    # -----------------------------------------------------------------
    # Plot 2: feature-map tiles
    # -----------------------------------------------------------------
    def _save_featuremap(self, feat: np.ndarray, epoch: int) -> None:
        """Tile sample-0 latent channels as a grid of grayscale heatmaps.

        Args:
            feat: Bottleneck latent ``(N, h, w, C)``.
            epoch: 1-based epoch number for filename / title.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        sample = feat[0]  # (h, w, C)
        h, w, c = sample.shape
        k = int(min(c, self.max_featuremap_channels))

        # Per-channel normalize for display, then tile via collage().
        tiles = np.empty((k, h, w, 1), dtype=np.float32)
        for i in range(k):
            ch = sample[:, :, i]
            lo, hi = float(ch.min()), float(ch.max())
            tiles[i, :, :, 0] = (ch - lo) / (hi - lo + 1e-8)

        grid = collage(tiles)  # (gh, gw, 1) sqrt-grid
        if grid is None:
            # collage() returns None when k does not tile cleanly; fall back
            # to a single-channel direct display of the first channel.
            grid = tiles[0]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(np.squeeze(grid), cmap="viridis")
        ax.set_title(f"Bottleneck feature maps (sample 0, {k}/{c} ch) — epoch {epoch}", fontsize=10)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"bottleneck_featuremap_epoch_{epoch:04d}.png",
            dpi=120,
            bbox_inches="tight",
        )
        plt.close(fig)

    # -----------------------------------------------------------------
    # Plot 3: PCA scatter
    # -----------------------------------------------------------------
    def _save_pca_scatter(self, feat: np.ndarray, mse: np.ndarray, epoch: int) -> None:
        """2-D PCA scatter of the flattened latent, colored by recon MSE.

        Args:
            feat: Bottleneck latent ``(N, h, w, C)``.
            mse: Per-sample reconstruction MSE ``(N,)``.
            epoch: 1-based epoch number for filename / title.
        """
        from sklearn.decomposition import PCA  # noqa: PLC0415
        import matplotlib.pyplot as plt  # noqa: PLC0415

        n_samples = feat.shape[0]
        if n_samples < 2:
            logger.warning(
                f"CliffordBottleneckMonitorCallback: PCA skipped at epoch {epoch} "
                f"(need >=2 samples, got {n_samples})"
            )
            return

        latent_flat = feat.reshape(n_samples, -1)
        n_dims = latent_flat.shape[1]
        n_components = min(2, n_dims)
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
        coords = pca.fit_transform(latent_flat)
        var_ratio = pca.explained_variance_ratio_

        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1] if n_components > 1 else np.zeros(len(coords)),
            c=mse,
            cmap="plasma",
            alpha=0.7,
            s=20,
            vmin=0.0,
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("recon MSE per sample", fontsize=9)
        ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%} var)", fontsize=9)
        if n_components > 1:
            ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%} var)", fontsize=9)
        ax.set_title(f"Bottleneck PCA — epoch {epoch}", fontsize=10)
        ax.text(
            0.02, 0.02,
            f"n={n_samples}  flat_dim={n_dims}",
            transform=ax.transAxes, fontsize=7, color="gray",
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"bottleneck_pca_epoch_{epoch:04d}.png",
            dpi=120,
            bbox_inches="tight",
        )
        plt.close(fig)

    # -----------------------------------------------------------------
    # Plot 4: histogram
    # -----------------------------------------------------------------
    def _save_histogram(self, feat: np.ndarray, epoch: int) -> None:
        """Histogram of all latent activations.

        Args:
            feat: Bottleneck latent ``(N, h, w, C)``.
            epoch: 1-based epoch number for filename / title.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        vals = feat.ravel()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(vals, bins=80, log=True)
        ax.set_xlabel("activation", fontsize=9)
        ax.set_ylabel("count (log)", fontsize=9)
        ax.set_title(
            f"Bottleneck activation histogram — epoch {epoch}  "
            f"(mean={vals.mean():.4f}, std={vals.std():.4f})",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"bottleneck_histogram_epoch_{epoch:04d}.png",
            dpi=120,
            bbox_inches="tight",
        )
        plt.close(fig)

    # -----------------------------------------------------------------
    # Keras hooks
    # -----------------------------------------------------------------
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Emit the four bottleneck PNGs on monitored epochs.

        Args:
            epoch: 0-based epoch index from Keras.
            logs: Keras logs dict (unused; reconstruction MSE is recomputed).
        """
        if (epoch + 1) % self.monitor_freq != 0:
            return
        if self.val_batch is None:
            return

        try:
            ep = epoch + 1
            feat = np.array(self.model.encode(self.val_batch, training=False))
            recon = np.array(self.model(self.val_batch, training=False)["reconstruction"])
            x = np.array(self.val_batch)
            mse = np.mean((recon - x) ** 2, axis=(1, 2, 3))

            stats = self._compute_health_stats(feat)
            for name, value in stats.items():
                self.history[name].append(value)
            self.epochs_seen.append(ep)

            self._save_health_curves()
            self._save_featuremap(feat, ep)
            self._save_pca_scatter(feat, mse, ep)
            self._save_histogram(feat, ep)

            gc.collect()
            logger.info(
                f"CliffordBottleneckMonitorCallback epoch {ep}: "
                f"dead_unit_frac={stats['dead_unit_frac']:.4f} "
                f"mean_l2={stats['mean_l2_norm']:.4f}"
            )
        except Exception as exc:
            logger.error(
                f"CliffordBottleneckMonitorCallback failed at epoch {epoch}: {exc}",
                exc_info=True,
            )

    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        """Re-emit the cumulative health curves at the end of training."""
        if not self.epochs_seen:
            return
        try:
            self._save_health_curves()
            logger.info(
                f"CliffordBottleneckMonitorCallback: final health curves saved to "
                f"{self.output_dir / 'bottleneck_health.png'}"
            )
        except Exception as exc:
            logger.error(
                f"CliffordBottleneckMonitorCallback.on_train_end failed: {exc}",
                exc_info=True,
            )
