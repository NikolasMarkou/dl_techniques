"""
Enhanced training script for Spatial VAE with specialized visualizations.

This script demonstrates how to train the Spatial VAE model and generate
comprehensive visualizations that showcase the spatial latent structure,
including spatial latent maps and spatial interpolations.

Usage:
    python spatial_vae/train.py [--dataset mnist] [--epochs 50] [--batch-size 128] [--latent-dim 8]
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


from dl_techniques.models.spatial_vae import SpatialVAE, create_spatial_vae, SpatialSampling
from dl_techniques.utils.logger import logger

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")


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


def load_mnist_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess MNIST dataset for Spatial VAE."""
    logger.info("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0
    logger.info(f"Train data shape: {x_train.shape}")
    logger.info(f"Test data shape: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)


def load_cifar10_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess CIFAR-10 dataset for Spatial VAE."""
    logger.info("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    logger.info(f"Train data shape: {x_train.shape}")
    logger.info(f"Test data shape: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)


def create_model_config(dataset: str, latent_dim: int) -> Dict[str, Any]:
    """Create Spatial VAE model configuration based on dataset."""
    if dataset.lower() == 'mnist':
        return {
            'latent_dim': latent_dim,
            'input_shape': (28, 28, 1),
            'encoder_filters': [32, 64],
            'decoder_filters': [64, 32],
            'spatial_latent_size': (7, 7),  # 28/4 = 7 (2 conv layers with stride 2)
            'kl_loss_weight': 1.0,
            'use_batch_norm': True,
        }
    elif dataset.lower() == 'cifar10':
        return {
            'latent_dim': latent_dim,
            'input_shape': (32, 32, 3),
            'encoder_filters': [32, 64, 128],
            'decoder_filters': [128, 64, 32],
            'spatial_latent_size': (8, 8),  # 32/4 = 8 (2 conv layers with stride 2)
            'kl_loss_weight': 1.0,
            'use_batch_norm': True,
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


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
        axes[0, i].set_title('Original', fontsize=10)
        axes[0, i].axis('off')
        axes[1, i].imshow(np.clip(img_recon, 0, 1), cmap=cmap)
        axes[1, i].set_title('Reconstructed', fontsize=10)
        axes[1, i].axis('off')

    title = f'Reconstruction Comparison - Epoch {epoch}' if epoch is not None else 'Final Reconstruction Comparison'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_spatial_latent_maps(
        model: SpatialVAE,
        sample_images: np.ndarray,
        save_path: str,
        epoch: Optional[int] = None,
        dataset: str = 'mnist'
) -> None:
    """Plot spatial latent maps for sample images."""
    n_samples = min(3, len(sample_images))
    outputs = model.predict(sample_images[:n_samples], verbose=0)
    z_mean = outputs['z_mean']  # Shape: (n_samples, H, W, latent_dim)

    cmap = 'gray' if dataset.lower() == 'mnist' else None

    # Show first few latent dimensions
    n_latent_dims = min(4, model.latent_dim)

    fig, axes = plt.subplots(n_samples, n_latent_dims + 1,
                            figsize=((n_latent_dims + 1) * 2, n_samples * 2))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        # Original image
        axes[i, 0].imshow(sample_images[i].squeeze(), cmap=cmap)
        axes[i, 0].set_title(f'Original {i+1}', fontsize=10)
        axes[i, 0].axis('off')

        # Spatial latent maps for different dimensions
        for j in range(n_latent_dims):
            latent_map = z_mean[i, :, :, j]
            im = axes[i, j+1].imshow(latent_map, cmap='viridis')
            axes[i, j+1].set_title(f'Latent Dim {j+1}', fontsize=10)
            axes[i, j+1].axis('off')
            plt.colorbar(im, ax=axes[i, j+1], fraction=0.046, pad=0.04)

    title = f'Spatial Latent Maps - Epoch {epoch}' if epoch is not None else 'Final Spatial Latent Maps'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_spatial_interpolation(
        model: SpatialVAE,
        sample_images: np.ndarray,
        save_path: str,
        epoch: Optional[int] = None,
        dataset: str = 'mnist'
) -> None:
    """Plot interpolation between two images in spatial latent space."""
    if len(sample_images) < 2:
        return

    # Encode two images
    outputs1 = model.predict(sample_images[0:1], verbose=0)
    outputs2 = model.predict(sample_images[1:2], verbose=0)

    z1 = outputs1['z_mean'][0]  # Shape: (H, W, latent_dim)
    z2 = outputs2['z_mean'][0]  # Shape: (H, W, latent_dim)

    # Create interpolation
    n_steps = 7
    alphas = np.linspace(0, 1, n_steps)

    interpolated_images = []
    for alpha in alphas:
        z_interp = (1 - alpha) * z1 + alpha * z2
        z_interp = np.expand_dims(z_interp, 0)  # Add batch dimension
        recon = model.decode(z_interp)
        interpolated_images.append(recon[0].numpy())

    # Plot
    cmap = 'gray' if dataset.lower() == 'mnist' else None
    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 1.5, 2))

    for i, (img, alpha) in enumerate(zip(interpolated_images, alphas)):
        axes[i].imshow(np.clip(img.squeeze(), 0, 1), cmap=cmap)
        axes[i].set_title(f'α={alpha:.2f}', fontsize=10)
        axes[i].axis('off')

    title = f'Spatial Latent Interpolation - Epoch {epoch}' if epoch is not None else 'Final Spatial Interpolation'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_latent_statistics(
        model: SpatialVAE,
        data: np.ndarray,
        save_path: str,
        epoch: Optional[int] = None,
        batch_size: int = 128
) -> None:
    """Plot statistics of spatial latent representations."""
    # Encode a subset of data
    n_samples = min(1000, len(data))
    outputs = model.predict(data[:n_samples], batch_size=batch_size, verbose=0)
    z_mean = outputs['z_mean']  # Shape: (n_samples, H, W, latent_dim)
    z_log_var = outputs['z_log_var']

    # Calculate statistics
    mean_activations = np.mean(z_mean, axis=0)  # (H, W, latent_dim)
    std_activations = np.std(z_mean, axis=0)    # (H, W, latent_dim)
    var_activations = np.mean(np.exp(z_log_var), axis=0)  # (H, W, latent_dim)

    # Plot statistics for first few latent dimensions
    n_dims = min(3, model.latent_dim)
    fig, axes = plt.subplots(3, n_dims, figsize=(n_dims * 3, 9))

    if n_dims == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_dims):
        # Mean activations
        im1 = axes[0, i].imshow(mean_activations[:, :, i], cmap='viridis')
        axes[0, i].set_title(f'Mean - Latent Dim {i+1}', fontsize=12)
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)

        # Standard deviation
        im2 = axes[1, i].imshow(std_activations[:, :, i], cmap='viridis')
        axes[1, i].set_title(f'Std Dev - Latent Dim {i+1}', fontsize=12)
        axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)

        # Learned variance
        im3 = axes[2, i].imshow(var_activations[:, :, i], cmap='viridis')
        axes[2, i].set_title(f'Learned Var - Latent Dim {i+1}', fontsize=12)
        axes[2, i].axis('off')
        plt.colorbar(im3, ax=axes[2, i], fraction=0.046, pad=0.04)

    title = f'Spatial Latent Statistics - Epoch {epoch}' if epoch is not None else 'Final Spatial Latent Statistics'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


class SpatialVisualizationCallback(keras.callbacks.Callback):
    """Callback to generate Spatial VAE visualizations during training."""

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

        # Create visualization directories
        self.recon_dir = os.path.join(save_dir, 'reconstructions_per_epoch')
        self.spatial_dir = os.path.join(save_dir, 'spatial_latent_per_epoch')
        self.interp_dir = os.path.join(save_dir, 'interpolation_per_epoch')
        self.stats_dir = os.path.join(save_dir, 'latent_stats_per_epoch')

        for dir_path in [self.recon_dir, self.spatial_dir, self.interp_dir, self.stats_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Sample images for consistent visualization
        indices = np.random.choice(len(self.x_val), size=5, replace=False)
        self.sample_images = self.x_val[indices]

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        if (epoch + 1) % self.frequency == 0 or epoch == 0:
            logger.info(f"Generating Spatial VAE visualizations for epoch {epoch + 1}...")

            # Reconstruction visualization
            outputs = self.model.predict(self.sample_images, verbose=0)
            reconstructions = outputs["reconstruction"]
            recon_path = os.path.join(self.recon_dir, f'epoch_{epoch + 1:03d}.png')
            plot_reconstruction_comparison(self.sample_images, reconstructions, recon_path, epoch + 1, self.dataset)

            # Spatial latent maps
            spatial_path = os.path.join(self.spatial_dir, f'epoch_{epoch + 1:03d}.png')
            plot_spatial_latent_maps(self.model, self.sample_images, spatial_path, epoch + 1, self.dataset)

            # Spatial interpolation
            interp_path = os.path.join(self.interp_dir, f'epoch_{epoch + 1:03d}.png')
            plot_spatial_interpolation(self.model, self.sample_images, interp_path, epoch + 1, self.dataset)

            # Latent statistics
            stats_path = os.path.join(self.stats_dir, f'epoch_{epoch + 1:03d}.png')
            plot_latent_statistics(self.model, self.x_val[:500], stats_path, epoch + 1, self.batch_size)


def create_callbacks(
        model_name: str,
        validation_data: Tuple[np.ndarray, np.ndarray],
        dataset: str,
        batch_size: int,
        monitor: str = 'val_loss',
        patience: int = 10,
        viz_frequency: int = 5
) -> Tuple[list, str]:
    """Create training callbacks."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"spatial_vae_{model_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(results_dir, 'best_model.keras'),
            monitor=monitor, save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        keras.callbacks.CSVLogger(filename=os.path.join(results_dir, 'training_log.csv')),
        keras.callbacks.TensorBoard(log_dir=os.path.join(results_dir, 'tensorboard'), histogram_freq=1),
        SpatialVisualizationCallback(validation_data, results_dir, dataset, viz_frequency, batch_size)
    ]
    logger.info(f"Results will be saved to: {results_dir}")
    return callbacks, results_dir


def plot_training_history(history: keras.callbacks.History, save_dir: str):
    """Plot training history curves for Spatial VAE."""
    history_dict = history.history

    # Use the correct keys from the Spatial VAE metrics
    loss_key = 'total_loss'
    recon_loss_key = 'reconstruction_loss'
    kl_loss_key = 'kl_loss'

    # Check if keys exist in history
    if loss_key not in history_dict:
        logger.warning(f"Key '{loss_key}' not found in history. Available keys: {list(history_dict.keys())}")
        return

    epochs = range(1, len(history_dict[loss_key]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=False)
    fig.suptitle("Spatial VAE Training and Validation Loss", fontsize=16, fontweight='bold')

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
        axes[2].set_title('Spatial KL Divergence Loss', fontsize=14)
        axes[2].set_xlabel('Epoch')
        axes[2].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()


def train_model(args: argparse.Namespace):
    """Main training function for Spatial VAE."""
    logger.info("Starting Spatial VAE training script")
    setup_gpu()

    if args.dataset.lower() == 'mnist':
        (x_train, y_train), (x_test, y_test) = load_mnist_data()
    elif args.dataset.lower() == 'cifar10':
        (x_train, y_train), (x_test, y_test) = load_cifar10_data()
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    model_config = create_model_config(args.dataset, args.latent_dim)
    model = create_spatial_vae(
        optimizer=args.optimizer, learning_rate=args.learning_rate, **model_config
    )
    model.summary(print_fn=logger.info)

    callbacks, results_dir = create_callbacks(
        model_name=args.dataset,
        validation_data=(x_test, y_test),
        dataset=args.dataset,
        batch_size=args.batch_size,
        patience=args.patience,
        viz_frequency=args.viz_frequency
    )

    logger.info("Starting Spatial VAE training...")
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
        try:
            best_model = keras.models.load_model(
                best_model_path,
                custom_objects={'SpatialVAE': SpatialVAE, 'SpatialSampling': SpatialSampling}
            )
        except Exception as e:
            logger.warning(f"Failed to load best model: {e}. Using final model state.")
            best_model = model
    else:
        logger.warning("No best model found, using the final model state.")
        best_model = model

    test_results = best_model.evaluate(x_test, batch_size=args.batch_size, verbose=1, return_dict=True)
    logger.info(f"Final Test Results (from best model): {test_results}")

    logger.info("Generating final Spatial VAE visualizations...")
    try:
        plot_training_history(history, results_dir)
    except Exception as e:
        logger.error(f"Failed to plot training history: {e}")

    # Final visualizations
    try:
        sample_indices = np.random.choice(len(x_test), size=5, replace=False)
        sample_images = x_test[sample_indices]

        # Final reconstructions
        outputs = best_model.predict(sample_images, verbose=0)
        final_recon_path = os.path.join(results_dir, 'final_reconstructions.png')
        plot_reconstruction_comparison(sample_images, outputs['reconstruction'], final_recon_path, dataset=args.dataset)

        # Final spatial latent maps
        final_spatial_path = os.path.join(results_dir, 'final_spatial_latent_maps.png')
        plot_spatial_latent_maps(best_model, sample_images, final_spatial_path, dataset=args.dataset)

        # Final interpolation
        final_interp_path = os.path.join(results_dir, 'final_spatial_interpolation.png')
        plot_spatial_interpolation(best_model, sample_images, final_interp_path, dataset=args.dataset)

        # Final latent statistics
        final_stats_path = os.path.join(results_dir, 'final_latent_statistics.png')
        plot_latent_statistics(best_model, x_test[:1000], final_stats_path, batch_size=args.batch_size)

        logger.info("All visualizations generated successfully.")
    except Exception as e:
        logger.error(f"Failed to generate some visualizations: {e}")

    try:
        final_model_path = os.path.join(results_dir, f"spatial_vae_{args.dataset}_final.keras")
        best_model.save(final_model_path)
        logger.info(f"Final best model saved to: {final_model_path}")
    except Exception as e:
        logger.error(f"Failed to save final model: {e}")

    # Save training summary
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"Spatial VAE Training Summary\n============================\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Latent Dim: {args.latent_dim}\n")
        f.write(f"Spatial Latent Size: {model_config.get('spatial_latent_size', 'Auto')}\n")
        f.write(f"Stopped at Epoch: {len(history.history['loss'])}\n\n")
        f.write(f"Final Test Results (from best model):\n")
        for key, val in test_results.items():
            f.write(f"  {key.replace('_', ' ').title()}: {val:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description='Train a Spatial Variational Autoencoder on image data.')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'], help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--latent-dim', type=int, default=8,
                        help='Dimensionality of the latent space at each spatial location')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--viz-frequency', type=int, default=5, help='Frequency of visualization callbacks (in epochs)')
    args = parser.parse_args()

    try:
        train_model(args)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()