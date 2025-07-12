"""
Training script for Hierarchical Variational Autoencoder (HVAE) with comprehensive visualizations.

This script demonstrates how to train the HVAE model using standard Keras
workflows with model.compile() and model.fit(). The script handles data loading,
model creation, training, evaluation, and generates comprehensive visualizations
including per-epoch reconstructions, latent space distributions, and pyramid visualizations.

Usage:
    python hvae/train.py [--dataset mnist] [--epochs 50] [--batch-size 128] [--num-levels 3] [--latent-dims 64 32 16]
"""

import os
import keras
import argparse
import matplotlib
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List

matplotlib.use('Agg')  # Use non-interactive backend for server environments
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.hvae import HVAE, create_hvae

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
    """Load and preprocess MNIST dataset for HVAE."""
    logger.info("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0
    logger.info(f"Train data shape: {x_train.shape}")
    logger.info(f"Test data shape: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)

# ---------------------------------------------------------------------

def load_cifar10_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess CIFAR-10 dataset for HVAE."""
    logger.info("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    logger.info(f"Train data shape: {x_train.shape}")
    logger.info(f"Test data shape: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)

# ---------------------------------------------------------------------

def create_model_config(dataset: str, num_levels: int, latent_dims: List[int]) -> Dict[str, Any]:
    """Create HVAE model configuration based on dataset."""
    if dataset.lower() == 'mnist':
        return {
            'num_levels': num_levels,
            'latent_dims': latent_dims,
            'input_shape': (28, 28, 1),
            'vae_config': {
                'depths': 2,
                'steps_per_depth': 2,
                'filters': [32, 64]
            }
        }
    elif dataset.lower() == 'cifar10':
        return {
            'num_levels': num_levels,
            'latent_dims': latent_dims,
            'input_shape': (32, 32, 3),
            'vae_config': {
                'depths': 2,
                'steps_per_depth': 2,
                'filters': [32, 64]
            }
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

def plot_pyramid_visualization(
        gaussian_pyramid: List[np.ndarray],
        laplacian_pyramid: List[np.ndarray],
        intermediate_reconstructions: List[np.ndarray],
        save_path: str,
        epoch: Optional[int] = None,
        dataset: str = 'mnist',
        sample_idx: int = 0
) -> None:
    """Plot pyramid decomposition and reconstruction at each level."""
    num_levels = len(gaussian_pyramid)
    fig, axes = plt.subplots(3, num_levels, figsize=(num_levels * 2, 6))
    cmap = 'gray' if dataset.lower() == 'mnist' else None

    # Handle single level case
    if num_levels == 1:
        axes = axes.reshape(3, 1)

    for i in range(num_levels):
        # Gaussian pyramid
        gaussian_img = gaussian_pyramid[i][sample_idx].squeeze()
        axes[0, i].imshow(gaussian_img, cmap=cmap)
        axes[0, i].set_title(f'Gaussian L{i}', fontsize=8)
        axes[0, i].axis('off')

        # Laplacian pyramid
        laplacian_img = laplacian_pyramid[i][sample_idx].squeeze()
        # Normalize Laplacian for visualization
        laplacian_normalized = (laplacian_img - laplacian_img.min()) / (laplacian_img.max() - laplacian_img.min() + 1e-8)
        axes[1, i].imshow(laplacian_normalized, cmap=cmap)
        axes[1, i].set_title(f'Laplacian L{i}', fontsize=8)
        axes[1, i].axis('off')

        # Intermediate reconstructions
        recon_img = intermediate_reconstructions[i][sample_idx].squeeze()
        axes[2, i].imshow(np.clip(recon_img, 0, 1), cmap=cmap)
        axes[2, i].set_title(f'Recon L{i}', fontsize=8)
        axes[2, i].axis('off')

    # Add row labels
    axes[0, 0].set_ylabel('Gaussian\nPyramid', fontsize=10, rotation=0, labelpad=40)
    axes[1, 0].set_ylabel('Laplacian\nPyramid', fontsize=10, rotation=0, labelpad=40)
    axes[2, 0].set_ylabel('Hierarchical\nReconstruction', fontsize=10, rotation=0, labelpad=40)

    title = f'Pyramid Visualization - Epoch {epoch}' if epoch is not None else 'Final Pyramid Visualization'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------

def plot_latent_space(
        model: HVAE,
        data: np.ndarray,
        labels: np.ndarray,
        save_path: str,
        epoch: Optional[int] = None,
        batch_size: int = 128,
        level: int = 0
) -> None:
    """Plot the latent space for a specific level colored by class labels."""
    if model.latent_dims[level] != 2:
        logger.warning(f"Latent space plotting is only available for latent_dim=2, level {level} has dim={model.latent_dims[level]}, skipping visualization.")
        return

    z_means, z_log_vars = model.encode(data)
    z_mean_level = z_means[level]

    plt.figure(figsize=(12, 10))

    if labels.ndim > 1:
        labels = labels.flatten()

    scatter = plt.scatter(z_mean_level[:, 0], z_mean_level[:, 1], c=labels, cmap='viridis', s=5, alpha=0.7)

    # Create a discrete colorbar
    num_classes = len(np.unique(labels))
    cbar = plt.colorbar(scatter, ticks=np.arange(num_classes))
    cbar.set_label('Class', rotation=270, labelpad=15, fontsize=12)

    plt.xlabel(f"Latent Dimension 1 (Level {level})", fontsize=12)
    plt.ylabel(f"Latent Dimension 2 (Level {level})", fontsize=12)
    title = f'Latent Space Distribution Level {level} - Epoch {epoch}' if epoch is not None else f'Final Latent Space Level {level}'
    plt.title(title, fontsize=14, fontweight='bold')

    # Fix axes limits for consistent visualization
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------

def plot_interpolation(
        model: HVAE,
        data: np.ndarray,
        save_path: str,
        n_interpolations: int = 5,
        n_steps: int = 10,
        epoch: Optional[int] = None,
        dataset: str = 'mnist'
) -> None:
    """Plot interpolations between pairs of samples in hierarchical latent space."""
    # Select pairs of samples randomly
    indices = np.random.choice(len(data), size=n_interpolations * 2, replace=False)
    sample_pairs = data[indices].reshape(n_interpolations, 2, *data.shape[1:])

    fig, axes = plt.subplots(n_interpolations, n_steps, figsize=(n_steps * 1.2, n_interpolations * 1.2))
    cmap = 'gray' if dataset.lower() == 'mnist' else None

    for i, (img1, img2) in enumerate(sample_pairs):
        # Encode both images to hierarchical latent space
        img1_batch = np.expand_dims(img1, axis=0)
        img2_batch = np.expand_dims(img2, axis=0)

        z_means1, z_log_vars1 = model.encode(img1_batch)
        z_means2, z_log_vars2 = model.encode(img2_batch)

        # Create interpolation steps
        alphas = np.linspace(0, 1, n_steps)

        for j, alpha in enumerate(alphas):
            # Interpolate in each level's latent space
            z_interp = []
            for level in range(model.num_levels):
                z_level_interp = (1 - alpha) * z_means1[level] + alpha * z_means2[level]
                z_interp.append(z_level_interp)

            # Decode interpolated latent vectors
            reconstructed = model.decode(z_interp)

            # Handle single vs multiple interpolations
            if n_interpolations == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]

            # Plot interpolated image
            img_to_plot = keras.ops.squeeze(reconstructed[0])
            img_to_plot = np.array(img_to_plot)
            ax.imshow(np.clip(img_to_plot, 0, 1), cmap=cmap)
            ax.axis('off')

            # Add alpha value as title for the first row
            if i == 0:
                ax.set_title(f'α={alpha:.1f}', fontsize=8)

    title = f'Hierarchical Latent Space Interpolation - Epoch {epoch}' if epoch is not None else 'Final Hierarchical Latent Space Interpolation'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------

class HVAEVisualizationCallback(keras.callbacks.Callback):
    """Callback to generate HVAE visualizations during training."""

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
        self.pyramid_dir = os.path.join(save_dir, 'pyramids_per_epoch')
        self.latent_dir = os.path.join(save_dir, 'latent_space_per_epoch')
        self.interp_dir = os.path.join(save_dir, 'interpolations_per_epoch')

        os.makedirs(self.recon_dir, exist_ok=True)
        os.makedirs(self.pyramid_dir, exist_ok=True)
        os.makedirs(self.latent_dir, exist_ok=True)
        os.makedirs(self.interp_dir, exist_ok=True)

        indices = np.random.choice(len(self.x_val), size=10, replace=False)
        self.sample_images = self.x_val[indices]

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        if (epoch + 1) % self.frequency == 0 or epoch == 0:
            logger.info(f"Generating HVAE visualizations for epoch {epoch + 1}...")

            # Get model outputs
            outputs = self.model.predict(self.sample_images, verbose=0)
            reconstructions = outputs["reconstruction"]

            # Reconstruction comparison
            recon_path = os.path.join(self.recon_dir, f'epoch_{epoch + 1:03d}.png')
            plot_reconstruction_comparison(self.sample_images, reconstructions, recon_path, epoch + 1, self.dataset)

            # Pyramid visualization
            pyramid_path = os.path.join(self.pyramid_dir, f'epoch_{epoch + 1:03d}.png')
            plot_pyramid_visualization(
                outputs["gaussian_pyramid"],
                outputs["laplacian_pyramid"],
                outputs["intermediate_reconstructions"],
                pyramid_path, epoch + 1, self.dataset
            )

            # Latent space visualization for each level with 2D latent space
            for level in range(self.model.num_levels):
                if self.model.latent_dims[level] == 2:
                    latent_path = os.path.join(self.latent_dir, f'epoch_{epoch + 1:03d}_level_{level}.png')
                    plot_latent_space(self.model, self.x_val[:5000], self.y_val[:5000],
                                    latent_path, epoch + 1, self.batch_size, level)

            # Interpolation visualization
            interp_path = os.path.join(self.interp_dir, f'epoch_{epoch + 1:03d}.png')
            plot_interpolation(self.model, self.x_val[:100], interp_path, n_interpolations=5,
                               n_steps=10, epoch=epoch + 1, dataset=self.dataset)

# ---------------------------------------------------------------------

def create_callbacks(
        model_name: str,
        validation_data: Tuple[np.ndarray, np.ndarray],
        dataset: str,
        batch_size: int,
        monitor: str = 'val_total_loss',
        patience: int = 10,
        viz_frequency: int = 5
) -> Tuple[list, str]:
    """Create training callbacks."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"hvae_{model_name}_{timestamp}")
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
        HVAEVisualizationCallback(validation_data, results_dir, dataset, viz_frequency, batch_size)
    ]
    logger.info(f"Results will be saved to: {results_dir}")
    return callbacks, results_dir

# ---------------------------------------------------------------------

def plot_training_history(history: keras.callbacks.History, save_dir: str):
    """Plots training history curves."""
    history_dict = history.history

    # Use the correct keys from the HVAE metrics
    loss_key = 'total_loss'
    recon_loss_key = 'reconstruction_loss'
    kl_loss_key = 'kl_loss'

    # Check if the keys exist in the history
    if loss_key not in history_dict:
        logger.error(f"Key '{loss_key}' not found in history. Available keys: {list(history_dict.keys())}")
        return

    epochs = range(1, len(history_dict[loss_key]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=False)
    fig.suptitle("HVAE Training and Validation Loss", fontsize=16, fontweight='bold')

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
    logger.info("Starting HVAE training script")
    setup_gpu()

    if args.dataset.lower() == 'mnist':
        (x_train, y_train), (x_test, y_test) = load_mnist_data()
    elif args.dataset.lower() == 'cifar10':
        (x_train, y_train), (x_test, y_test) = load_cifar10_data()
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    model_config = create_model_config(args.dataset, args.num_levels, args.latent_dims)
    model = create_hvae(
        optimizer=args.optimizer,
        kl_loss_weight=args.kl_loss_weight,
        **model_config
    )
    model.summary(print_fn=logger.info)

    callbacks, results_dir = create_callbacks(
        model_name=args.dataset,
        validation_data=(x_test, y_test),
        dataset=args.dataset,
        batch_size=args.batch_size,
        monitor='val_total_loss',
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
        # Load with custom objects to handle the HVAE class
        from dl_techniques.layers.sampling import Sampling
        from dl_techniques.models.hvae import GaussianDownsample
        best_model = keras.models.load_model(
            best_model_path,
            custom_objects={
                "HVAE": HVAE,
                "Sampling": Sampling,
                "GaussianDownsample": GaussianDownsample
            }
        )
    else:
        logger.warning("No best model found, using the final model state.")
        best_model = model

    test_results = best_model.evaluate(x_test, batch_size=args.batch_size, verbose=1, return_dict=True)
    logger.info(f"Final Test Results (from best model): {test_results}")

    logger.info("Generating final visualizations...")
    plot_training_history(history, results_dir)

    # Final visualizations
    sample_indices = np.random.choice(len(x_test), size=10, replace=False)
    sample_images = x_test[sample_indices]
    outputs = best_model.predict(sample_images, verbose=0)

    # Final reconstruction comparison
    final_recon_path = os.path.join(results_dir, 'final_reconstructions.png')
    plot_reconstruction_comparison(sample_images, outputs['reconstruction'], final_recon_path, dataset=args.dataset)

    # Final pyramid visualization
    final_pyramid_path = os.path.join(results_dir, 'final_pyramid.png')
    plot_pyramid_visualization(
        outputs["gaussian_pyramid"],
        outputs["laplacian_pyramid"],
        outputs["intermediate_reconstructions"],
        final_pyramid_path, dataset=args.dataset
    )

    # Final latent space visualizations for 2D levels
    for level in range(best_model.num_levels):
        if best_model.latent_dims[level] == 2:
            final_latent_path = os.path.join(results_dir, f'final_latent_space_level_{level}.png')
            plot_latent_space(best_model, x_test[:5000], y_test[:5000],
                            final_latent_path, batch_size=args.batch_size, level=level)

    # Final interpolation
    final_interp_path = os.path.join(results_dir, 'final_interpolations.png')
    plot_interpolation(best_model, x_test[:100], final_interp_path, n_interpolations=5, n_steps=10, dataset=args.dataset)

    # Save final model
    final_model_path = os.path.join(results_dir, f"hvae_{args.dataset}_final.keras")
    best_model.save(final_model_path)
    logger.info(f"Final best model saved to: {final_model_path}")

    # Save training summary
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"HVAE Training Summary\n=====================\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Number of Levels: {args.num_levels}\n")
        f.write(f"Latent Dimensions: {args.latent_dims}\n")
        f.write(f"KL Loss Weight: {args.kl_loss_weight}\n")
        f.write(f"Stopped at Epoch: {len(history.history['total_loss'])}\n\n")
        f.write(f"Final Test Results (from best model):\n")
        for key, val in test_results.items():
            f.write(f"  {key.replace('_', ' ').title()}: {val:.4f}\n")

# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train a Hierarchical Variational Autoencoder (HVAE) on image data.')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'], help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--num-levels', type=int, default=3, help='Number of hierarchy levels')
    parser.add_argument('--latent-dims', type=int, nargs='+', default=[64, 32, 16],
                        help='Latent dimensions for each level (use 2 for a level to enable latent space visualization)')
    parser.add_argument('--kl-loss-weight', type=float, default=0.01, help='Weight for KL divergence loss')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--viz-frequency', type=int, default=5, help='Frequency of visualization callbacks (in epochs)')
    args = parser.parse_args()

    # Validate arguments
    if len(args.latent_dims) != args.num_levels:
        raise ValueError(f"Number of latent dimensions ({len(args.latent_dims)}) must match number of levels ({args.num_levels})")

    # Check power of 2 constraint for image dimensions
    if args.dataset.lower() == 'mnist':
        if 28 % (2 ** (args.num_levels - 1)) != 0:
            raise ValueError(f"MNIST dimensions (28x28) not compatible with {args.num_levels} levels. Try 3 levels or fewer.")
    elif args.dataset.lower() == 'cifar10':
        if 32 % (2 ** (args.num_levels - 1)) != 0:
            raise ValueError(f"CIFAR-10 dimensions (32x32) not compatible with {args.num_levels} levels. Try 3 levels or fewer.")

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