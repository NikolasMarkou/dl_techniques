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
from dl_techniques.layers.sampling import Sampling

from train.common import (
    setup_gpu,
    create_base_argument_parser,
    load_dataset,
)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

CUSTOM_OBJECTS = {"VAE": VAE, "Sampling": Sampling}


# ---------------------------------------------------------------------
# VAE-specific visualizations (domain-specific, stays local)
# ---------------------------------------------------------------------


def plot_reconstruction_comparison(
        original: np.ndarray, reconstructed: np.ndarray,
        save_path: str, epoch: Optional[int] = None, dataset: str = 'mnist',
) -> None:
    """Plot original vs reconstructed images."""
    n = min(10, len(original))
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.5))
    cmap = 'gray' if dataset.lower() == 'mnist' else None

    for i in range(n):
        axes[0, i].imshow(original[i].squeeze(), cmap=cmap)
        axes[0, i].set_title('Original', fontsize=8)
        axes[0, i].axis('off')
        axes[1, i].imshow(np.clip(reconstructed[i].squeeze(), 0, 1), cmap=cmap)
        axes[1, i].set_title('Reconstructed', fontsize=8)
        axes[1, i].axis('off')

    title = f'Reconstruction - Epoch {epoch}' if epoch else 'Final Reconstruction'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_latent_space(
        model: VAE, data: np.ndarray, labels: np.ndarray,
        save_path: str, epoch: Optional[int] = None, batch_size: int = 128,
) -> None:
    """Plot 2D latent space colored by class labels."""
    if model.latent_dim != 2:
        return

    outputs = model.predict(data, batch_size=batch_size, verbose=0)
    z_mean = outputs['z_mean']
    labels = labels.flatten() if labels.ndim > 1 else labels

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap='viridis', s=5, alpha=0.7)
    plt.colorbar(scatter, ticks=np.arange(len(np.unique(labels)))).set_label('Class', rotation=270, labelpad=15)
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.title(f'Latent Space - Epoch {epoch}' if epoch else 'Final Latent Space', fontweight='bold')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
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
    """Plot VAE training loss curves."""
    h = history.history
    if 'total_loss' not in h:
        logger.error(f"'total_loss' not in history. Keys: {list(h.keys())}")
        return

    epochs = range(1, len(h['total_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle("VAE Training Loss", fontsize=16, fontweight='bold')

    for ax, key, title in zip(axes, ['total_loss', 'reconstruction_loss', 'kl_loss'],
                                    ['Total Loss', 'Reconstruction Loss', 'KL Divergence']):
        if key in h:
            ax.plot(epochs, h[key], 'b-', label='Train')
            if f'val_{key}' in h:
                ax.plot(epochs, h[f'val_{key}'], 'r-', label='Val')
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def train_model(args):
    """Main training function."""
    logger.info("Starting VAE training")
    setup_gpu(gpu_id=args.gpu)

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

    logger.info(f"Data: {x_train.shape[0]} train, {x_test.shape[0]} test, shape={input_shape}")

    # Create model
    model = create_vae(
        input_shape=input_shape,
        latent_dim=args.latent_dim,
        variant="small",
        optimizer=args.optimizer,
    )
    model.summary(print_fn=logger.info)

    # Callbacks — VAE monitors val_total_loss, not val_accuracy
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"vae_{args.dataset}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_total_loss', patience=args.patience,
                                       restore_best_weights=True, verbose=1, mode='min'),
        keras.callbacks.ModelCheckpoint(os.path.join(results_dir, 'best_model.keras'),
                                         monitor='val_total_loss', save_best_only=True, verbose=1, mode='min'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_total_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        keras.callbacks.CSVLogger(os.path.join(results_dir, 'training_log.csv')),
        VisualizationCallback((x_test, y_test), results_dir, args.dataset, args.viz_frequency, args.batch_size),
    ]

    # Train
    history = model.fit(
        x_train, validation_data=(x_test, None),
        epochs=args.epochs, batch_size=args.batch_size,
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
