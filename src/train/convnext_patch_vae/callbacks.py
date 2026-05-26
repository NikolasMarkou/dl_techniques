"""Visualization callbacks for ConvNeXtPatchVAE training.

Two callbacks:
- ``LatentSpaceCallback``: PCA scatter of encoded latent means, colored by
  per-sample KL divergence.
- ``LatentInterpolationCallback``: linear interpolation grid between adjacent
  pairs of validation images through the latent space.
"""

import os
import gc
from typing import Dict, Optional

import numpy as np
import keras

from dl_techniques.utils.logger import logger


class LatentSpaceCallback(keras.callbacks.Callback):
    """PCA scatter of the latent means every ``frequency`` epochs.

    Encodes ``val_images`` through the model encoder, flattens the 4-D
    spatial latent ``(B, Hp, Wp, latent_dim)`` to ``(B, Hp*Wp*latent_dim)``
    (no mean-pooling — patch structure is preserved), then projects to 2-D
    with randomised-SVD PCA.  Points are coloured by per-sample KL divergence,
    which requires no class labels and gives a direct diagnostic of posterior
    collapse or coverage.

    Args:
        val_images: Fixed validation images, shape ``(N, H, W, C)`` in
            model-input space.
        save_dir: Directory for PNG files.
        frequency: Save every this many epochs (and at ``on_train_end``).
        cifar_mean: Per-channel mean for MSE de-normalisation.  ``None``
            means images are already in ``[0, 1]``.
        cifar_std: Per-channel std for MSE de-normalisation.
    """

    def __init__(
        self,
        val_images: np.ndarray,
        save_dir: str,
        frequency: int = 5,
        cifar_mean: Optional[np.ndarray] = None,
        cifar_std: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self._val_images = val_images
        self.save_dir = save_dir
        self.frequency = frequency
        self.cifar_mean = cifar_mean
        self.cifar_std = cifar_std
        os.makedirs(save_dir, exist_ok=True)

    def _encode_all(self):
        """Return ``(latent_flat, kl_per_sample)`` arrays.

        Processes in batches of 32 to bound GPU memory usage.
        """
        import tensorflow as tf  # noqa: PLC0415  -- lazy import

        batch_size = 32
        n = len(self._val_images)
        flats, kls = [], []

        for start in range(0, n, batch_size):
            batch = keras.ops.convert_to_tensor(
                self._val_images[start : start + batch_size], dtype="float32"
            )
            mu, log_var = self.model.encode(batch)
            mu_np = np.array(mu)       # (B, Hp, Wp, D)
            lv_np = np.array(log_var)  # (B, Hp, Wp, D)

            B = mu_np.shape[0]
            flat_dim = mu_np.shape[1] * mu_np.shape[2] * mu_np.shape[3]
            flats.append(mu_np.reshape(B, flat_dim))

            # Per-sample KL: -0.5 * mean_over_(Hp,Wp,D)(1 + lv - mu^2 - exp(lv))
            lv_clipped = np.clip(lv_np, -10.0, 10.0)
            kl = -0.5 * np.mean(
                1.0 + lv_clipped - mu_np ** 2 - np.exp(lv_clipped),
                axis=(1, 2, 3),
            )
            kls.append(kl)

        return np.concatenate(flats, axis=0), np.concatenate(kls, axis=0)

    def _save_scatter(self, path: str, title: str) -> None:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt  # noqa: PLC0415

        latent_flat, kl_vals = self._encode_all()
        n_samples, n_dims = latent_flat.shape

        n_components = min(2, n_dims)
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
        coords = pca.fit_transform(latent_flat)
        var_ratio = pca.explained_variance_ratio_

        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1] if n_components > 1 else np.zeros(len(coords)),
            c=kl_vals,
            cmap="plasma",
            alpha=0.7,
            s=20,
            vmin=0.0,
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("KL divergence per sample", fontsize=9)
        ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%} var)", fontsize=9)
        if n_components > 1:
            ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%} var)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.text(
            0.02, 0.02,
            f"n={n_samples}  flat_dim={n_dims}",
            transform=ax.transAxes, fontsize=7, color="gray",
        )
        plt.tight_layout()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        gc.collect()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if epoch % self.frequency != 0:
            return
        try:
            loss_val = (logs or {}).get("loss", float("nan"))
            path = os.path.join(self.save_dir, f"latent_epoch_{epoch + 1:04d}.png")
            self._save_scatter(path, f"Latent PCA — Epoch {epoch + 1}  |  loss={loss_val:.4f}")
        except Exception as exc:
            logger.warning(f"LatentSpaceCallback failed at epoch {epoch}: {exc}")

    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        try:
            path = os.path.join(self.save_dir, "latent_final.png")
            self._save_scatter(path, "Latent PCA — Final")
            logger.info(f"Final latent scatter saved: {path}")
        except Exception as exc:
            logger.warning(f"LatentSpaceCallback.on_train_end failed: {exc}")


class LatentInterpolationCallback(keras.callbacks.Callback):
    """Linear interpolation grid between adjacent pairs of val images.

    Encodes each image in a pair to its posterior mean ``mu``, then linearly
    interpolates ``z = (1-alpha)*mu_A + alpha*mu_B`` across ``num_steps``
    alpha values and decodes each point.  Saves a grid PNG with one row per
    pair and one column per interpolation step.

    Args:
        val_samples: Fixed validation images, shape ``(N, H, W, C)`` in
            model-input space.  Must contain at least 2 images; only
            ``(N // 2) * 2`` images are used (pairs).
        save_dir: Directory for PNG files.
        frequency: Save every this many epochs (and at ``on_train_end``).
        num_steps: Number of interpolation steps (including both endpoints).
        cifar_mean: Per-channel mean for MSE de-normalisation.
        cifar_std: Per-channel std.
    """

    def __init__(
        self,
        val_samples: np.ndarray,
        save_dir: str,
        frequency: int = 5,
        num_steps: int = 8,
        cifar_mean: Optional[np.ndarray] = None,
        cifar_std: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        n_pairs = len(val_samples) // 2
        self._val_samples = val_samples[: n_pairs * 2]
        self.save_dir = save_dir
        self.frequency = frequency
        self.num_steps = num_steps
        self.cifar_mean = cifar_mean
        self.cifar_std = cifar_std
        os.makedirs(save_dir, exist_ok=True)

    def _to_display(self, x: np.ndarray) -> np.ndarray:
        """Undo normalisation and clip to ``[0, 1]``."""
        if self.cifar_mean is not None:
            x = x * self.cifar_std + self.cifar_mean
        return np.clip(x, 0.0, 1.0)

    def _build_grid(self):
        """Return ``(n_pairs, num_steps, H, W, C)`` decoded interpolations."""
        n_pairs = len(self._val_samples) // 2
        alphas = np.linspace(0.0, 1.0, self.num_steps, dtype=np.float32)
        grid = []

        for p in range(n_pairs):
            x_a = keras.ops.convert_to_tensor(
                self._val_samples[p * 2 : p * 2 + 1], dtype="float32"
            )
            x_b = keras.ops.convert_to_tensor(
                self._val_samples[p * 2 + 1 : p * 2 + 2], dtype="float32"
            )
            mu_a = np.array(self.model.encode(x_a)[0])  # (1, Hp, Wp, D)
            mu_b = np.array(self.model.encode(x_b)[0])

            row = []
            for alpha in alphas:
                z = keras.ops.convert_to_tensor(
                    (1.0 - alpha) * mu_a + alpha * mu_b, dtype="float32"
                )
                decoded = np.array(self.model.decode(z))  # (1, H, W, C)
                row.append(self._to_display(decoded[0]))
            grid.append(row)

        return grid  # list[list[np.ndarray(H,W,C)]]

    def _save_grid(self, path: str, title: str) -> None:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        grid = self._build_grid()
        n_pairs = len(grid)
        num_steps = self.num_steps
        cmap = "gray" if self._val_samples.shape[-1] == 1 else None
        alphas = np.linspace(0.0, 1.0, num_steps)

        fig, axes = plt.subplots(
            n_pairs, num_steps,
            figsize=(num_steps * 1.4, n_pairs * 1.4 + 0.5),
            squeeze=False,
        )
        for r in range(n_pairs):
            for c in range(num_steps):
                axes[r, c].imshow(grid[r][c].squeeze(), cmap=cmap, interpolation="bilinear")
                axes[r, c].axis("off")
            axes[r, 0].set_ylabel(f"pair {r + 1}", fontsize=7)

        for c, alpha in enumerate(alphas):
            axes[0, c].set_title(f"α={alpha:.2f}", fontsize=7)

        fig.suptitle(title, fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        gc.collect()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if epoch % self.frequency != 0:
            return
        try:
            loss_val = (logs or {}).get("loss", float("nan"))
            path = os.path.join(self.save_dir, f"interp_epoch_{epoch + 1:04d}.png")
            self._save_grid(
                path,
                f"Latent Interpolations — Epoch {epoch + 1}  |  loss={loss_val:.4f}",
            )
        except Exception as exc:
            logger.warning(f"LatentInterpolationCallback failed at epoch {epoch}: {exc}")

    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        try:
            path = os.path.join(self.save_dir, "interp_final.png")
            self._save_grid(path, "Latent Interpolations — Final")
            logger.info(f"Final interpolation grid saved: {path}")
        except Exception as exc:
            logger.warning(f"LatentInterpolationCallback.on_train_end failed: {exc}")
