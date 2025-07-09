"""
Training script for Keras-compliant VAE with comprehensive visualizations.

This script demonstrates how to train the VAE model using standard Keras
workflows with model.compile() and model.fit(). The script handles data loading,
model creation, training, evaluation, and generates comprehensive visualizations
including per-epoch reconstructions and latent space distributions.

Usage:
    python vae/train.py [--dataset mnist] [--epochs 50] [--batch-size 128] [--latent-dim 2]
"""

import argparse
import os
import numpy as np
import keras
import tensorflow as tf
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.vae import VAE, create_vae

# ---------------------------------------------------------------------

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# ---------------------------------------------------------------------

def setup_gpu():
    """Configure GPU settings for optimal training."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")
    else:
        logger.info("No GPUs found, using CPU")

# ---------------------------------------------------------------------

def load_mnist_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess MNIST dataset for VAE."""
    logger.info("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0
    logger.info(f"Train data shape: {x_train.shape}")
    logger.info(f"Test data shape: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)

# ---------------------------------------------------------------------

def load_cifar10_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess CIFAR-10 dataset for VAE."""
    logger.info("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    logger.info(f"Train data shape: {x_train.shape}")
    logger.info(f"Test data shape: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)

# ---------------------------------------------------------------------

def create_model_config(dataset: str, latent_dim: int) -> Dict[str, Any]:
    """Create VAE model configuration based on dataset."""
    if dataset.lower() == 'mnist':
        return {
            'latent_dim': latent_dim,
            'input_shape': (28, 28, 1),
            'depths': 3,
            'steps_per_depth': 3,
            'filters': [32, 64, 128]
        }
    elif dataset.lower() == 'cifar10':
        return {
            'latent_dim': latent_dim,
            'input_shape': (32, 32, 3),
            'depths': 3,
            'steps_per_depth': 3,
            'filters': [32, 64, 128]
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

# ---------------------------------------------------------------------

def plot_reconstruction_comparison(
        original_images: np.ndarray,
        reconstructed_images: np.ndarray,
        save_path: str,
        epoch: Optional[int] = None,
        dataset: str = 'mnist'
) -> None:
    """Plot original vs reconstructed images comparison."""
    n_samples = min(10, len(original_images))
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 1.5, 3.5))
    cmap = 'gray' if dataset.lower() == 'mnist' else None

    for i in range(n_samples):
        img_orig = original_images[i].squeeze()
        img_recon = reconstructed_images[i].squeeze()
        axes[0, i].imshow(img_orig, cmap=cmap)
        axes[0, i].set_title('Original', fontsize=8)
        axes[0, i].axis('off')
        axes[1, i].imshow(np.clip(img_recon, 0, 1), cmap=cmap)
        axes[1, i].set_title('Reconstructed', fontsize=8)
        axes[1, i].axis('off')

    title = f'Reconstruction Comparison - Epoch {epoch}' if epoch is not None else 'Final Reconstruction Comparison'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------

def plot_latent_space(
        model: VAE,
        data: np.ndarray,
        labels: np.ndarray,
        save_path: str,
        epoch: Optional[int] = None,
        batch_size: int = 128
) -> None:
    """Plot the latent space colored by class labels."""
    if model.latent_dim != 2:
        logger.warning(f"Latent space plotting is only available for latent_dim=2, skipping visualization.")
        return

    outputs = model.predict(data, batch_size=batch_size, verbose=0)
    z_mean = outputs['z_mean']

    plt.figure(figsize=(12, 10))

    if labels.ndim > 1:
        labels = labels.flatten()

    scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap='viridis', s=5, alpha=0.7)

    # Create a discrete colorbar correctly.
    # The 'scatter' object itself is the "mappable" that the colorbar function needs.
    num_classes = len(np.unique(labels))
    cbar = plt.colorbar(scatter, ticks=np.arange(num_classes))
    cbar.set_label('Digit Class', rotation=270, labelpad=15, fontsize=12)

    plt.xlabel("Latent Dimension 1", fontsize=12)
    plt.ylabel("Latent Dimension 2", fontsize=12)
    title = f'Latent Space Distribution - Epoch {epoch}' if epoch is not None else 'Final Latent Space'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------

class VisualizationCallback(keras.callbacks.Callback):
    """Callback to generate VAE visualizations during training."""

    def __init__(
            self,
            validation_data: Tuple[np.ndarray, np.ndarray],
            save_dir: str,
            dataset: str,
            frequency: int = 5,
            batch_size: int = 128
    ):
        super().__init__()
        self.x_val, self.y_val = validation_data
        self.save_dir = save_dir
        self.dataset = dataset
        self.frequency = frequency
        self.batch_size = batch_size

        self.recon_dir = os.path.join(save_dir, 'reconstructions_per_epoch')
        self.latent_dir = os.path.join(save_dir, 'latent_space_per_epoch')
        os.makedirs(self.recon_dir, exist_ok=True)
        os.makedirs(self.latent_dir, exist_ok=True)

        indices = np.random.choice(len(self.x_val), size=10, replace=False)
        self.sample_images = self.x_val[indices]

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        if (epoch + 1) % self.frequency == 0 or epoch == 0:
            logger.info(f"Generating visualizations for epoch {epoch + 1}...")
            # Reconstruction visualization
            outputs = self.model.predict(self.sample_images, verbose=0)
            reconstructions = outputs["reconstruction"]
            recon_path = os.path.join(self.recon_dir, f'epoch_{epoch + 1:03d}.png')
            plot_reconstruction_comparison(self.sample_images, reconstructions, recon_path, epoch + 1, self.dataset)

            # Latent space visualization (if applicable)
            latent_path = os.path.join(self.latent_dir, f'epoch_{epoch + 1:03d}.png')
            # Use a subset of validation data for speed
            plot_latent_space(self.model, self.x_val[:5000], self.y_val[:5000], latent_path, epoch + 1, self.batch_size)

# ---------------------------------------------------------------------

def create_callbacks(
        model_name: str,
        validation_data: Tuple[np.ndarray, np.ndarray],
        dataset: str,
        batch_size: int,
        monitor: str = 'val_total_loss',  # Fixed: Use the correct metric name
        patience: int = 10,
        viz_frequency: int = 5
) -> Tuple[list, str]:
    """Create training callbacks."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"vae_{model_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True, verbose=1, mode='min'),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(results_dir, 'best_model.keras'),
            monitor=monitor, save_best_only=True, verbose=1, mode='min'
        ),
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=5, min_lr=1e-6, verbose=1, mode='min'),
        keras.callbacks.CSVLogger(filename=os.path.join(results_dir, 'training_log.csv')),
        keras.callbacks.TensorBoard(log_dir=os.path.join(results_dir, 'tensorboard'), histogram_freq=1),
        VisualizationCallback(validation_data, results_dir, dataset, viz_frequency, batch_size)
    ]
    logger.info(f"Results will be saved to: {results_dir}")
    return callbacks, results_dir

# ---------------------------------------------------------------------

def plot_training_history(history: keras.callbacks.History, save_dir: str):
    """Plots training history curves."""
    history_dict = history.history

    # Use the correct keys from the VAE metrics
    loss_key = 'total_loss'
    recon_loss_key = 'reconstruction_loss'
    kl_loss_key = 'kl_loss'

    # Check if the keys exist in the history
    if loss_key not in history_dict:
        logger.error(f"Key '{loss_key}' not found in history. Available keys: {list(history_dict.keys())}")
        return

    epochs = range(1, len(history_dict[loss_key]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=False)
    fig.suptitle("VAE Training and Validation Loss", fontsize=16, fontweight='bold')

    # Total Loss
    axes[0].plot(epochs, history_dict[loss_key], 'b-', label='Training Loss')
    if f'val_{loss_key}' in history_dict:
        axes[0].plot(epochs, history_dict[f'val_{loss_key}'], 'r-', label='Validation Loss')
    axes[0].set_title('Total Loss', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Reconstruction Loss
    if recon_loss_key in history_dict:
        axes[1].plot(epochs, history_dict[recon_loss_key], 'b-', label='Training')
        if f'val_{recon_loss_key}' in history_dict:
            axes[1].plot(epochs, history_dict[f'val_{recon_loss_key}'], 'r-', label='Validation')
        axes[1].set_title('Reconstruction Loss', fontsize=14)
        axes[1].set_xlabel('Epoch')
        axes[1].legend()

    # KL Loss
    if kl_loss_key in history_dict:
        axes[2].plot(epochs, history_dict[kl_loss_key], 'b-', label='Training')
        if f'val_{kl_loss_key}' in history_dict:
            axes[2].plot(epochs, history_dict[f'val_{kl_loss_key}'], 'r-', label='Validation')
        axes[2].set_title('KL Divergence Loss', fontsize=14)
        axes[2].set_xlabel('Epoch')
        axes[2].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()

# ---------------------------------------------------------------------

def train_model(args: argparse.Namespace):
    """Main training function."""
    logger.info("Starting VAE training script")
    setup_gpu()

    if args.dataset.lower() == 'mnist':
        (x_train, y_train), (x_test, y_test) = load_mnist_data()
    elif args.dataset.lower() == 'cifar10':
        (x_train, y_train), (x_test, y_test) = load_cifar10_data()
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    model_config = create_model_config(args.dataset, args.latent_dim)
    model = create_vae(
        optimizer=args.optimizer,
        **model_config
    )
    model.summary(print_fn=logger.info)

    callbacks, results_dir = create_callbacks(
        model_name=args.dataset,
        validation_data=(x_test, y_test),
        dataset=args.dataset,
        batch_size=args.batch_size,
        monitor='val_total_loss',  # Fixed: Use the correct metric name
        patience=args.patience,
        viz_frequency=args.viz_frequency
    )

    logger.info("Starting model training...")
    history = model.fit(
        x_train,
        validation_data=(x_test, None),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    logger.info("Training completed. Evaluating on test set...")
    best_model_path = os.path.join(results_dir, 'best_model.keras')
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from: {best_model_path}")
        # Load with custom objects to handle the VAE class
        from dl_techniques.layers.sampling import Sampling
        best_model = keras.models.load_model(
            best_model_path,
            custom_objects={"VAE": VAE, "Sampling": Sampling}
        )
    else:
        logger.warning("No best model found, using the final model state.")
        best_model = model

    test_results = best_model.evaluate(x_test, batch_size=args.batch_size, verbose=1, return_dict=True)
    logger.info(f"Final Test Results (from best model): {test_results}")

    logger.info("Generating final visualizations...")
    plot_training_history(history, results_dir)
    final_recon_path = os.path.join(results_dir, 'final_reconstructions.png')
    final_latent_path = os.path.join(results_dir, 'final_latent_space.png')

    sample_indices = np.random.choice(len(x_test), size=10, replace=False)
    sample_images = x_test[sample_indices]
    outputs = best_model.predict(sample_images, verbose=0)
    plot_reconstruction_comparison(sample_images, outputs['reconstruction'], final_recon_path, dataset=args.dataset)
    plot_latent_space(best_model, x_test[:5000], y_test[:5000], final_latent_path, batch_size=args.batch_size)

    final_model_path = os.path.join(results_dir, f"vae_{args.dataset}_final.keras")
    best_model.save(final_model_path)
    logger.info(f"Final best model saved to: {final_model_path}")

    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"VAE Training Summary\n====================\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Latent Dim: {args.latent_dim}\n")
        f.write(f"Stopped at Epoch: {len(history.history['total_loss'])}\n\n")
        f.write(f"Final Test Results (from best model):\n")
        for key, val in test_results.items():
            f.write(f"  {key.replace('_', ' ').title()}: {val:.4f}\n")

# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train a Variational Autoencoder (VAE) on image data.')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'], help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--latent-dim', type=int, default=2,
                        help='Dimensionality of the latent space (use 2 for visualization)')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--viz-frequency', type=int, default=5, help='Frequency of visualization callbacks (in epochs)')
    args = parser.parse_args()

    try:
        train_model(args)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise

# ---------------------------------------------------------------------

if __name__ == '__main__':
    main()