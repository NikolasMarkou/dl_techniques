"""
Training script for Keras-compliant VAE with comprehensive visualizations.

Handles data loading, model creation, training, evaluation, and generates
per-epoch reconstructions and latent space visualizations.
"""

import os
import keras
import argparse
import matplotlib
import numpy as np
from typing import Tuple, Dict, Any, Optional

matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.vae.model import VAE, create_vae
from dl_techniques.layers.sampling import Sampling, HypersphereSampling

from train.common import (
    setup_gpu,
    create_base_argument_parser,
    create_callbacks,
    generate_training_curves,
    load_dataset,
    set_seeds,
    save_config_json,
)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

CUSTOM_OBJECTS = {"VAE": VAE, "Sampling": Sampling, "HypersphereSampling": HypersphereSampling}


# ---------------------------------------------------------------------
# VAE-specific visualizations (domain-specific, stays local)
# ---------------------------------------------------------------------


def plot_reconstruction_comparison(
        original: np.ndarray, reconstructed: np.ndarray,
        save_path: str, epoch: Optional[int] = None, dataset: str = 'mnist',
) -> None:
    """Plot original vs reconstructed images."""
    n = min(10, len(original))
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.8))
    cmap = 'gray' if dataset.lower() == 'mnist' else None

    # Label each row once on the left (set_title over every cell melds the
    # bottom row's caption into the originals above it). Hide ticks/spines so
    # the ylabel reads as a clean row header without drawing a frame.
    row_labels = ('Original', 'Reconstructed')
    for row in range(2):
        imgs = original if row == 0 else np.clip(reconstructed, 0, 1)
        for i in range(n):
            ax = axes[row, i]
            ax.imshow(imgs[i].squeeze(), cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(row_labels[row], fontsize=10, fontweight='bold')

    title = f'Reconstruction - Epoch {epoch}' if epoch else 'Final Reconstruction'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    # Explicit spacing: leave headroom for the suptitle and a real gap between
    # the two rows so labels never overlap the images.
    fig.subplots_adjust(top=0.88, hspace=0.15, wspace=0.05)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_latent_space(
        model: VAE, data: np.ndarray, labels: np.ndarray,
        save_path: str, epoch: Optional[int] = None, batch_size: int = 128,
) -> None:
    """Plot the latent space colored by class labels (any latent_dim).

    For latent_dim==2 the native 2 coordinates are used. For latent_dim>2 the
    coordinates are projected to 2D with PCA so the plot is always produced
    (the old `latent_dim != 2: return` guard silently emitted nothing for
    higher-dim runs). For hypersphere modes the plotted quantity is the
    ON-SPHERE direction (the real latent), not raw z_mean.
    """
    outputs = model.predict(data, batch_size=batch_size, verbose=0)
    z_mean = np.asarray(outputs['z_mean'])
    labels = labels.flatten() if labels.ndim > 1 else labels

    # DECISION plan_2026-06-04_7ff8ea8b/D-002: for hypersphere modes the decoder
    # consumes the ON-SPHERE latent z = radius * normalize(z_mean + eps); the raw
    # z_mean is the UNNORMALIZED mean and can have ||z_mean|| >> 4, so the old hard
    # [-4,4] clamp rendered a healthy direction-spread as a "collapsed point". Plot
    # the on-sphere direction u = z_mean/||z_mean|| (the REAL latent), NOT raw z_mean.
    is_sphere = str(model.sampling_type).startswith("hypersphere")
    if is_sphere:
        norm = np.linalg.norm(z_mean, axis=1, keepdims=True)
        coords = z_mean / np.maximum(norm, 1e-12)
        title_q = "on-sphere direction"
    else:
        coords = z_mean
        title_q = "z_mean"

    # DECISION plan_2026-06-04_7ff8ea8b/D-007: plot ANY latent_dim. dim==2 uses the
    # native coords (circle for hypersphere, autoscaled plane for gaussian); dim>2 is
    # projected to 2D via PCA so latent_space_per_epoch is populated for every run.
    if coords.shape[1] == 2:
        coords2 = coords
        xlabel, ylabel = "Latent Dim 1", "Latent Dim 2"
        lim = 1.3 if is_sphere else None
    else:
        from sklearn.decomposition import PCA
        coords2 = PCA(n_components=2).fit_transform(coords)
        xlabel, ylabel = "PCA-1", "PCA-2"
        title_q = f"{title_q}, PCA-2 of {coords.shape[1]}D"
        lim = None

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(coords2[:, 0], coords2[:, 1], c=labels, cmap='viridis', s=5, alpha=0.7)
    plt.colorbar(scatter, ticks=np.arange(len(np.unique(labels)))).set_label('Class', rotation=270, labelpad=15)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    base = f'Latent Space [{title_q}]'
    plt.title(f'{base} - Epoch {epoch}' if epoch else f'Final {base}', fontweight='bold')
    if lim is not None:
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_interpolation(
        model: VAE, data: np.ndarray, save_path: str,
        n_interp: int = 10, n_steps: int = 10,
        epoch: Optional[int] = None, dataset: str = 'mnist',
) -> None:
    """Plot interpolations between sample pairs in latent space."""
    indices = np.random.choice(len(data), size=n_interp * 2, replace=False)
    pairs = data[indices].reshape(n_interp, 2, *data.shape[1:])
    fig, axes = plt.subplots(n_interp, n_steps, figsize=(n_steps * 1.2, n_interp * 1.2))
    cmap = 'gray' if dataset.lower() == 'mnist' else None

    for i, (img1, img2) in enumerate(pairs):
        z1 = model.predict(img1[np.newaxis], verbose=0)['z_mean']
        z2 = model.predict(img2[np.newaxis], verbose=0)['z_mean']
        for j, alpha in enumerate(np.linspace(0, 1, n_steps)):
            z = (1 - alpha) * z1 + alpha * z2
            recon = np.array(keras.ops.squeeze(model.decode(z)[0]))
            ax = axes[j] if n_interp == 1 else axes[i, j]
            ax.imshow(np.clip(recon, 0, 1), cmap=cmap)
            ax.axis('off')
            if i == 0:
                ax.set_title(f'α={alpha:.1f}', fontsize=8)

    fig.suptitle(f'Interpolation - Epoch {epoch}' if epoch else 'Final Interpolation', fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


class VisualizationCallback(keras.callbacks.Callback):
    """Generate per-epoch VAE visualizations."""

    def __init__(self, val_data, save_dir, dataset, frequency=5, batch_size=128):
        super().__init__()
        self.x_val, self.y_val = val_data
        self.save_dir = save_dir
        self.dataset = dataset
        self.frequency = frequency
        self.batch_size = batch_size

        self.recon_dir = os.path.join(save_dir, 'reconstructions_per_epoch')
        self.latent_dir = os.path.join(save_dir, 'latent_space_per_epoch')
        self.interp_dir = os.path.join(save_dir, 'interpolations_per_epoch')
        for d in [self.recon_dir, self.latent_dir, self.interp_dir]:
            os.makedirs(d, exist_ok=True)

        self.samples = self.x_val[np.random.choice(len(self.x_val), 10, replace=False)]

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.frequency == 0 or epoch == 0:
            outputs = self.model.predict(self.samples, verbose=0)
            plot_reconstruction_comparison(
                self.samples, outputs["reconstruction"],
                os.path.join(self.recon_dir, f'epoch_{epoch+1:03d}.png'), epoch+1, self.dataset)
            plot_latent_space(
                self.model, self.x_val[:5000], self.y_val[:5000],
                os.path.join(self.latent_dir, f'epoch_{epoch+1:03d}.png'), epoch+1, self.batch_size)
            plot_interpolation(
                self.model, self.x_val[:100],
                os.path.join(self.interp_dir, f'epoch_{epoch+1:03d}.png'),
                epoch=epoch+1, dataset=self.dataset)


def plot_training_history(history, save_dir):
    """Plot VAE training loss curves.

    ``generate_training_curves`` (train.common) hard-requires a ``'loss'``
    history key, but the VAE tracks ``'total_loss'`` (no plain ``'loss'``).
    Alias ``total_loss``/``val_total_loss`` to ``loss``/``val_loss`` so the
    shared plotting util works without modifying the common helper.
    """
    hist = dict(history.history) if not isinstance(history, dict) else dict(history)
    if 'loss' not in hist and 'total_loss' in hist:
        hist['loss'] = hist['total_loss']
    if 'val_loss' not in hist and 'val_total_loss' in hist:
        hist['val_loss'] = hist['val_total_loss']
    generate_training_curves(
        history=hist,
        results_dir=save_dir,
        filename="training_history",
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def train_model(args):
    """Main training function."""
    logger.info("Starting VAE training")
    setup_gpu(gpu_id=args.gpu)

    # Seed every RNG source BEFORE dataset load / model construction so
    # comparison arms are reproducible (compare_runs reads per-run logs).
    set_seeds(args.seed)

    # Load dataset — VAE keeps grayscale channel for MNIST
    if args.dataset.lower() == 'mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
        x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0
        input_shape = (28, 28, 1)
    elif args.dataset.lower() == 'cifar10':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        y_train, y_test = y_train.flatten(), y_test.flatten()
        input_shape = (32, 32, 3)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Smoke mode: subset the TRAIN set and force a single epoch so the full
    # path (train -> CSVLogger -> best_model.keras -> reload) runs in seconds.
    # Validation + create_callbacks (CSVLogger) stay INTACT so training_log.csv
    # is still produced for compare_runs.
    epochs = args.epochs
    if args.smoke:
        smoke_n = min(2000, x_train.shape[0])
        x_train = x_train[:smoke_n]
        epochs = 1
        logger.info(f"[smoke] train subset={smoke_n}, epochs forced to 1")

    logger.info(f"Data: {x_train.shape[0]} train, {x_test.shape[0]} test, shape={input_shape}")

    # Create model
    model = create_vae(
        input_shape=input_shape,
        latent_dim=args.latent_dim,
        variant="small",
        optimizer=args.optimizer,
        sampling_type=args.sampler,
    )
    model.summary(print_fn=logger.info)

    # Callbacks — VAE monitors val_total_loss, not val_accuracy.
    # Include the sampler arm in model_name so the 3 comparison arms produce
    # 3 distinct run dirs.
    callbacks, results_dir = create_callbacks(
        model_name=f"vae_{args.dataset}_{args.sampler}",
        results_dir_prefix="vae",
        monitor='val_total_loss',
        patience=args.patience,
        use_lr_schedule=False,
        include_analyzer=not args.no_epoch_analyzer,
    )

    # Persist the run config so the comparison's config-diff sees `sampler`.
    save_config_json(vars(args), results_dir)
    callbacks.append(
        VisualizationCallback((x_test, y_test), results_dir, args.dataset, args.viz_frequency, args.batch_size),
    )

    # Train
    history = model.fit(
        x_train, validation_data=(x_test, None),
        epochs=epochs, batch_size=args.batch_size,
        callbacks=callbacks, verbose=1,
    )

    # Load best model
    best_path = os.path.join(results_dir, 'best_model.keras')
    best_model = model
    if os.path.exists(best_path):
        try:
            best_model = keras.models.load_model(best_path, custom_objects=CUSTOM_OBJECTS)
            logger.info("Loaded best checkpoint")
        except Exception as e:
            logger.warning(f"Could not load best model: {e}")

    test_results = best_model.evaluate(x_test, batch_size=args.batch_size, verbose=1, return_dict=True)
    logger.info(f"Test results: {test_results}")

    # Final visualizations
    plot_training_history(history, results_dir)
    samples = x_test[np.random.choice(len(x_test), 10, replace=False)]
    outputs = best_model.predict(samples, verbose=0)
    plot_reconstruction_comparison(samples, outputs['reconstruction'],
                                    os.path.join(results_dir, 'final_reconstructions.png'), dataset=args.dataset)
    plot_latent_space(best_model, x_test[:5000], y_test[:5000],
                       os.path.join(results_dir, 'final_latent_space.png'), batch_size=args.batch_size)
    plot_interpolation(best_model, x_test[:100],
                        os.path.join(results_dir, 'final_interpolations.png'), dataset=args.dataset)

    # Save summary
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"VAE Training Summary\n{'=' * 30}\n")
        f.write(f"Dataset: {args.dataset}, Latent dim: {args.latent_dim}\n")
        f.write(f"Epochs: {len(history.history['total_loss'])}\n\n")
        for key, val in test_results.items():
            f.write(f"  {key}: {val:.4f}\n")

    logger.info(f"Training complete. Results: {results_dir}")


# ---------------------------------------------------------------------

def main():
    parser = create_base_argument_parser(
        description='Train VAE on image data',
        dataset_choices=['mnist', 'cifar10'],
    )
    parser.add_argument('--latent-dim', type=int, default=2, dest='latent_dim',
                        help='Latent space dimensionality (2 for visualization)')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--viz-frequency', type=int, default=5, dest='viz_frequency',
                        help='Visualization frequency (epochs)')
    parser.add_argument('--sampler', type=str, default='gaussian',
                        choices=['gaussian', 'hypersphere'],
                        help='Latent sampling mode passed to create_vae(sampling_type=...)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for all RNG sources (reproducible comparison arms)')
    parser.add_argument('--smoke', action='store_true', default=False,
                        help='Smoke mode: 1 epoch on a small train subset (fast end-to-end check)')
    parser.add_argument("--no-epoch-analyzer", action="store_true",
                        help="Disable the per-epoch WeightWatcher/ModelAnalyzer callback (speeds up sweeps/comparisons).")
    parser.set_defaults(epochs=10, batch_size=128, patience=10)
    args = parser.parse_args()

    try:
        train_model(args)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
