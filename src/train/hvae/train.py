"""
Training script for Hierarchical VAE (HVAE) with comprehensive visualizations.

This script demonstrates how to train the HVAE model using standard Keras
workflows with model.compile() and model.fit(). The script handles data loading,
model creation, training, evaluation, and generates comprehensive visualizations
including per-epoch reconstructions, hierarchical latent space distributions,
and multi-level interpolations.

Usage:
    python hvae/train.py [--dataset mnist] [--epochs 50] [--batch-size 128] [--latent-dims 32 32 16 16]
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
from dl_techniques.models.hvae import HierarchicalVAE, create_hvae
from dl_techniques.layers.sampling import Sampling

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

def create_model_config(dataset: str, latent_dims: List[int]) -> Dict[str, Any]:
    """Create HVAE model configuration based on dataset."""
    if dataset.lower() == 'mnist':
        return {
            'latent_dims': latent_dims,
            'input_shape': (28, 28, 1),
            'num_levels': 2,
            'hidden_dims': [64, 128, 256],
            'kl_loss_weights': [0.01, 0.005],
            'use_pyramid_downsampling': True
        }
    elif dataset.lower() == 'cifar10':
        return {
            'latent_dims': latent_dims,
            'input_shape': (32, 32, 3),
            'num_levels': 2,
            'hidden_dims': [64, 128, 256],
            'kl_loss_weights': [0.01, 0.005],
            'use_pyramid_downsampling': True
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

def plot_hierarchical_latent_space(
        model: HierarchicalVAE,
        data: np.ndarray,
        labels: np.ndarray,
        save_path: str,
        epoch: Optional[int] = None,
        batch_size: int = 128
) -> None:
    """Plot hierarchical latent spaces colored by class labels."""
    # Get hierarchical latent representations
    latent_dict = model.predict(data, batch_size=batch_size, verbose=0)

    # Determine number of latent pairs
    num_level1_pairs = len(latent_dict['level1_means'])
    num_level2_pairs = len(latent_dict['level2_means'])

    # Create subplots for each level
    fig, axes = plt.subplots(2, max(num_level1_pairs, num_level2_pairs),
                            figsize=(4 * max(num_level1_pairs, num_level2_pairs), 8))

    if labels.ndim > 1:
        labels = labels.flatten()

    # Plot Level 1 latent spaces
    for i in range(num_level1_pairs):
        ax = axes[0, i] if max(num_level1_pairs, num_level2_pairs) > 1 else axes[0]

        # Get the mean for this latent pair
        z_mean = latent_dict['level1_means'][i]

        # If latent dimension is 2, plot as 2D scatter
        if z_mean.shape[1] == 2:
            scatter = ax.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap='viridis', s=5, alpha=0.7)
            ax.set_xlabel(f"Level 1 Latent {i+1} - Dim 1")
            ax.set_ylabel(f"Level 1 Latent {i+1} - Dim 2")
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
        else:
            # For higher dimensions, plot first two dimensions
            scatter = ax.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap='viridis', s=5, alpha=0.7)
            ax.set_xlabel(f"Level 1 Latent {i+1} - Dim 1")
            ax.set_ylabel(f"Level 1 Latent {i+1} - Dim 2")
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)

        ax.set_title(f'Level 1 Latent {i+1}', fontsize=12)
        ax.grid(True, alpha=0.3)

    # Plot Level 2 latent spaces
    for i in range(num_level2_pairs):
        ax = axes[1, i] if max(num_level1_pairs, num_level2_pairs) > 1 else axes[1]

        # Get the mean for this higher-level latent
        z_mean = latent_dict['level2_means'][i]

        # If latent dimension is 2, plot as 2D scatter
        if z_mean.shape[1] == 2:
            scatter = ax.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap='viridis', s=5, alpha=0.7)
            ax.set_xlabel(f"Level 2 Latent {i+1} - Dim 1")
            ax.set_ylabel(f"Level 2 Latent {i+1} - Dim 2")
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
        else:
            # For higher dimensions, plot first two dimensions
            scatter = ax.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap='viridis', s=5, alpha=0.7)
            ax.set_xlabel(f"Level 2 Latent {i+1} - Dim 1")
            ax.set_ylabel(f"Level 2 Latent {i+1} - Dim 2")
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)

        ax.set_title(f'Level 2 Latent {i+1}', fontsize=12)
        ax.grid(True, alpha=0.3)

    # Add colorbar
    num_classes = len(np.unique(labels))
    cbar = plt.colorbar(scatter, ax=axes, ticks=np.arange(num_classes))
    cbar.set_label('Class', rotation=270, labelpad=15, fontsize=12)

    title = f'Hierarchical Latent Space - Epoch {epoch}' if epoch is not None else 'Final Hierarchical Latent Space'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------

def plot_hierarchical_interpolation(
        model: HierarchicalVAE,
        data: np.ndarray,
        save_path: str,
        n_interpolations: int = 5,
        n_steps: int = 10,
        epoch: Optional[int] = None,
        dataset: str = 'mnist'
) -> None:
    """Plot hierarchical interpolations between pairs of samples."""
    # Select pairs of samples randomly
    indices = np.random.choice(len(data), size=n_interpolations * 2, replace=False)
    sample_pairs = data[indices].reshape(n_interpolations, 2, *data.shape[1:])

    fig, axes = plt.subplots(n_interpolations, n_steps, figsize=(n_steps * 1.2, n_interpolations * 1.2))
    cmap = 'gray' if dataset.lower() == 'mnist' else None

    for i, (img1, img2) in enumerate(sample_pairs):
        # Encode both images to hierarchical latent space
        img1_batch = np.expand_dims(img1, axis=0)
        img2_batch = np.expand_dims(img2, axis=0)

        latent1 = model.encode(img1_batch)
        latent2 = model.encode(img2_batch)

        # Create interpolation steps
        alphas = np.linspace(0, 1, n_steps)

        for j, alpha in enumerate(alphas):
            # Interpolate in hierarchical latent space
            interp_latent = {}
            interp_latent['level1_latents'] = []
            interp_latent['level2_latents'] = []

            # Interpolate level 1 latents
            for k in range(len(latent1['level1_latents'])):
                z1 = latent1['level1_latents'][k]
                z2 = latent2['level1_latents'][k]
                z_interp = (1 - alpha) * z1 + alpha * z2
                interp_latent['level1_latents'].append(z_interp)

            # Interpolate level 2 latents
            for k in range(len(latent1['level2_latents'])):
                z1 = latent1['level2_latents'][k]
                z2 = latent2['level2_latents'][k]
                z_interp = (1 - alpha) * z1 + alpha * z2
                interp_latent['level2_latents'].append(z_interp)

            # Decode interpolated latent vectors
            reconstructed = model.decode(interp_latent)

            # Handle single vs multiple interpolations
            if n_interpolations == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]

            # Plot interpolated image
            img_to_plot = keras.ops.squeeze(reconstructed[0])
            img_to_plot = np.array(img_to_plot)  # Convert to numpy array
            ax.imshow(np.clip(img_to_plot, 0, 1), cmap=cmap)
            ax.axis('off')

            # Add alpha value as title for the first row
            if i == 0:
                ax.set_title(f'α={alpha:.1f}', fontsize=8)

    title = f'Hierarchical Interpolation - Epoch {epoch}' if epoch is not None else 'Final Hierarchical Interpolation'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------

def plot_level_comparison(
        model: HierarchicalVAE,
        data: np.ndarray,
        save_path: str,
        n_samples: int = 5,
        epoch: Optional[int] = None,
        dataset: str = 'mnist'
) -> None:
    """Plot reconstructions using different levels of the hierarchy."""
    # Select random samples
    indices = np.random.choice(len(data), size=n_samples, replace=False)
    sample_images = data[indices]

    # Get hierarchical latent representations
    latent_dict = model.encode(sample_images)

    # Create reconstructions using different levels
    fig, axes = plt.subplots(4, n_samples, figsize=(n_samples * 1.5, 6))
    cmap = 'gray' if dataset.lower() == 'mnist' else None

    for i in range(n_samples):
        # Original image
        axes[0, i].imshow(sample_images[i].squeeze(), cmap=cmap)
        axes[0, i].set_title('Original', fontsize=8)
        axes[0, i].axis('off')

        # Full reconstruction
        full_recon = model.decode(latent_dict)
        axes[1, i].imshow(np.clip(full_recon[i].numpy().squeeze(), 0, 1), cmap=cmap)
        axes[1, i].set_title('Full Hierarchy', fontsize=8)
        axes[1, i].axis('off')

        # Level 1 only reconstruction (set level 2 to zeros)
        level1_only = {
            'level1_latents': [lat[i:i+1] for lat in latent_dict['level1_latents']],
            'level2_latents': [np.zeros_like(lat[i:i+1]) for lat in latent_dict['level2_latents']]
        }
        level1_recon = model.decode(level1_only)
        axes[2, i].imshow(np.clip(level1_recon[0].squeeze(), 0, 1), cmap=cmap)
        axes[2, i].set_title('Level 1 Only', fontsize=8)
        axes[2, i].axis('off')

        # Level 2 only reconstruction (set level 1 to zeros)
        level2_only = {
            'level1_latents': [np.zeros_like(lat[i:i+1]) for lat in latent_dict['level1_latents']],
            'level2_latents': [lat[i:i+1] for lat in latent_dict['level2_latents']]
        }
        level2_recon = model.decode(level2_only)
        axes[3, i].imshow(np.clip(level2_recon[0].squeeze(), 0, 1), cmap=cmap)
        axes[3, i].set_title('Level 2 Only', fontsize=8)
        axes[3, i].axis('off')

    title = f'Hierarchical Level Comparison - Epoch {epoch}' if epoch is not None else 'Final Level Comparison'
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

        # Create visualization directories
        self.recon_dir = os.path.join(save_dir, 'reconstructions_per_epoch')
        self.latent_dir = os.path.join(save_dir, 'hierarchical_latent_per_epoch')
        self.interp_dir = os.path.join(save_dir, 'hierarchical_interpolations_per_epoch')
        self.level_dir = os.path.join(save_dir, 'level_comparisons_per_epoch')

        os.makedirs(self.recon_dir, exist_ok=True)
        os.makedirs(self.latent_dir, exist_ok=True)
        os.makedirs(self.interp_dir, exist_ok=True)
        os.makedirs(self.level_dir, exist_ok=True)

        # Select sample images for consistent visualization
        indices = np.random.choice(len(self.x_val), size=10, replace=False)
        self.sample_images = self.x_val[indices]

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        if (epoch + 1) % self.frequency == 0 or epoch == 0:
            logger.info(f"Generating HVAE visualizations for epoch {epoch + 1}...")

            # Reconstruction visualization
            outputs = self.model.predict(self.sample_images, verbose=0)
            reconstructions = outputs["reconstruction"]
            recon_path = os.path.join(self.recon_dir, f'epoch_{epoch + 1:03d}.png')
            plot_reconstruction_comparison(self.sample_images, reconstructions, recon_path, epoch + 1, self.dataset)

            # Hierarchical latent space visualization
            latent_path = os.path.join(self.latent_dir, f'epoch_{epoch + 1:03d}.png')
            plot_hierarchical_latent_space(self.model, self.x_val[:2000], self.y_val[:2000],
                                         latent_path, epoch + 1, self.batch_size)

            # Hierarchical interpolation visualization
            interp_path = os.path.join(self.interp_dir, f'epoch_{epoch + 1:03d}.png')
            plot_hierarchical_interpolation(self.model, self.x_val[:100], interp_path,
                                          n_interpolations=5, n_steps=10, epoch=epoch + 1, dataset=self.dataset)

            # Level comparison visualization
            # level_path = os.path.join(self.level_dir, f'epoch_{epoch + 1:03d}.png')
            # plot_level_comparison(self.model, self.x_val[:50], level_path,
            #                     n_samples=5, epoch=epoch + 1, dataset=self.dataset)

# ---------------------------------------------------------------------

def create_callbacks(
        model_name: str,
        validation_data: Tuple[np.ndarray, np.ndarray],
        dataset: str,
        batch_size: int,
        monitor: str = 'val_total_loss',
        patience: int = 15,
        viz_frequency: int = 5
) -> Tuple[list, str]:
    """Create training callbacks for HVAE."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"hvae_{model_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience, restore_best_weights=True, verbose=1, mode='min'
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(results_dir, 'best_model.keras'),
            monitor=monitor, save_best_only=True, verbose=1, mode='min'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=7, min_lr=1e-6, verbose=1, mode='min'
        ),
        keras.callbacks.CSVLogger(filename=os.path.join(results_dir, 'training_log.csv')),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(results_dir, 'tensorboard'), histogram_freq=1
        ),
        HVAEVisualizationCallback(validation_data, results_dir, dataset, viz_frequency, batch_size)
    ]
    logger.info(f"Results will be saved to: {results_dir}")
    return callbacks, results_dir

# ---------------------------------------------------------------------

def plot_training_history(history: keras.callbacks.History, save_dir: str):
    """Plot HVAE training history curves."""
    history_dict = history.history

    # HVAE-specific metric keys
    loss_key = 'total_loss'
    recon_loss_key = 'reconstruction_loss'
    kl_loss_keys = [key for key in history_dict.keys() if key.startswith('kl_loss_level_') and not key.startswith('val_')]

    # Check if the keys exist in the history
    if loss_key not in history_dict:
        logger.error(f"Key '{loss_key}' not found in history. Available keys: {list(history_dict.keys())}")
        return

    epochs = range(1, len(history_dict[loss_key]) + 1)

    # Create subplots: total loss, reconstruction loss, and KL losses for each level
    n_plots = 2 + len(kl_loss_keys)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    fig.suptitle("HVAE Training and Validation Loss", fontsize=16, fontweight='bold')

    # Total Loss
    axes[0].plot(epochs, history_dict[loss_key], 'b-', label='Training Loss')
    if f'val_{loss_key}' in history_dict:
        axes[0].plot(epochs, history_dict[f'val_{loss_key}'], 'r-', label='Validation Loss')
    axes[0].set_title('Total Loss', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reconstruction Loss
    if recon_loss_key in history_dict:
        axes[1].plot(epochs, history_dict[recon_loss_key], 'b-', label='Training')
        if f'val_{recon_loss_key}' in history_dict:
            axes[1].plot(epochs, history_dict[f'val_{recon_loss_key}'], 'r-', label='Validation')
        axes[1].set_title('Reconstruction Loss', fontsize=14)
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # KL Losses for each level
    for i, kl_key in enumerate(kl_loss_keys):
        ax_idx = 2 + i
        axes[ax_idx].plot(epochs, history_dict[kl_key], 'b-', label='Training')
        if f'val_{kl_key}' in history_dict:
            axes[ax_idx].plot(epochs, history_dict[f'val_{kl_key}'], 'r-', label='Validation')
        level_num = kl_key.split('_')[-1]
        axes[ax_idx].set_title(f'KL Loss Level {level_num}', fontsize=14)
        axes[ax_idx].set_xlabel('Epoch')
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------

def train_model(args: argparse.Namespace):
    """Main training function for HVAE."""
    logger.info("Starting HVAE training script")
    setup_gpu()

    # Load data
    if args.dataset.lower() == 'mnist':
        (x_train, y_train), (x_test, y_test) = load_mnist_data()
    elif args.dataset.lower() == 'cifar10':
        (x_train, y_train), (x_test, y_test) = load_cifar10_data()
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Create model
    model_config = create_model_config(args.dataset, args.latent_dims)
    model = create_hvae(
        optimizer=args.optimizer,
        **model_config
    )
    model.summary(print_fn=logger.info)

    # Create callbacks
    callbacks, results_dir = create_callbacks(
        model_name=args.dataset,
        validation_data=(x_test, y_test),
        dataset=args.dataset,
        batch_size=args.batch_size,
        monitor='val_total_loss',
        patience=args.patience,
        viz_frequency=args.viz_frequency
    )

    # Train model
    logger.info("Starting HVAE training...")
    history = model.fit(
        x_train,
        validation_data=(x_test, None),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate and save results
    logger.info("Training completed. Evaluating on test set...")
    best_model_path = os.path.join(results_dir, 'best_model.keras')
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from: {best_model_path}")
        best_model = keras.models.load_model(
            best_model_path,
            custom_objects={"HierarchicalVAE": HierarchicalVAE, "Sampling": Sampling}
        )
    else:
        logger.warning("No best model found, using the final model state.")
        best_model = model

    test_results = best_model.evaluate(x_test, batch_size=args.batch_size, verbose=1, return_dict=True)
    logger.info(f"Final Test Results (from best model): {test_results}")

    # Generate final visualizations
    logger.info("Generating final visualizations...")
    plot_training_history(history, results_dir)

    # Final visualizations
    final_recon_path = os.path.join(results_dir, 'final_reconstructions.png')
    final_latent_path = os.path.join(results_dir, 'final_hierarchical_latent.png')
    final_interp_path = os.path.join(results_dir, 'final_hierarchical_interpolations.png')
    final_level_path = os.path.join(results_dir, 'final_level_comparisons.png')

    sample_indices = np.random.choice(len(x_test), size=10, replace=False)
    sample_images = x_test[sample_indices]

    outputs = best_model.predict(sample_images, verbose=0)
    plot_reconstruction_comparison(sample_images, outputs['reconstruction'], final_recon_path, dataset=args.dataset)
    plot_hierarchical_latent_space(best_model, x_test[:2000], y_test[:2000], final_latent_path, batch_size=args.batch_size)
    plot_hierarchical_interpolation(best_model, x_test[:100], final_interp_path, n_interpolations=5, n_steps=10, dataset=args.dataset)
    #plot_level_comparison(best_model, x_test[:50], final_level_path, n_samples=5, dataset=args.dataset)

    # Save final model
    final_model_path = os.path.join(results_dir, f"hvae_{args.dataset}_final.keras")
    best_model.save(final_model_path)
    logger.info(f"Final best model saved to: {final_model_path}")

    # Create training summary
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"HVAE Training Summary\n====================\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Latent Dims: {args.latent_dims}\n")
        f.write(f"Num Levels: {model_config['num_levels']}\n")
        f.write(f"KL Loss Weights: {model_config['kl_loss_weights']}\n")
        f.write(f"Stopped at Epoch: {len(history.history['total_loss'])}\n\n")
        f.write(f"Final Test Results (from best model):\n")
        for key, val in test_results.items():
            f.write(f"  {key.replace('_', ' ').title()}: {val:.4f}\n")

# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train a Hierarchical Variational Autoencoder (HVAE) on image data.')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                        help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--latent-dims', type=int, nargs='+', default=[32, 16],
                        help='Latent dimensions for hierarchical levels (each represents one latent variable)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--viz-frequency', type=int, default=5,
                        help='Frequency of visualization callbacks (in epochs)')

    args = parser.parse_args()

    # Validate latent dimensions
    if len(args.latent_dims) < 2:
        raise ValueError("latent_dims must have at least 2 dimensions")

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