"""
Training script for MDN-VAE with comprehensive visualizations.

This script demonstrates how to train the MDN-VAE model using standard Keras
workflows with model.compile() and model.fit(). The script handles data loading,
model creation, training, evaluation, and generates comprehensive visualizations
including per-epoch reconstructions, latent space distributions, and mixture
component analysis.

Usage:
    python mdn_vae/train.py [--dataset mnist] [--epochs 50] [--batch-size 128] [--latent-dim 10] [--num-mixtures 5]
"""

import os
import keras
import argparse
import matplotlib
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

matplotlib.use('Agg')  # Use non-interactive backend for server environments
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.mdn_layer import MDNLayer
from dl_techniques.models.mdn_vae import MDN_VAE, create_mdn_vae

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

def create_model_config(dataset: str, latent_dim: int, num_mixtures: int) -> Dict[str, Any]:
    """Create MDN-VAE model configuration based on dataset."""
    if dataset.lower() == 'mnist':
        return {
            'latent_dim': latent_dim,
            'num_mixtures': num_mixtures,
            'input_shape': (28, 28, 1),
            'depths': 2,  # Reduced for stability
            'steps_per_depth': 2,
            'filters': [32, 64],
            'kl_loss_weight': 0.001  # Lower for MDN
        }
    elif dataset.lower() == 'cifar10':
        return {
            'latent_dim': latent_dim,
            'num_mixtures': num_mixtures,
            'input_shape': (32, 32, 3),
            'depths': 3,
            'steps_per_depth': 2,
            'filters': [32, 64, 128],
            'kl_loss_weight': 0.0001  # Even lower for more complex data
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

    title = f'MDN-VAE Reconstruction - Epoch {epoch}' if epoch is not None else 'Final MDN-VAE Reconstruction'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------

def plot_mixture_weights(
        model: MDN_VAE,
        data: np.ndarray,
        labels: np.ndarray,
        save_path: str,
        epoch: Optional[int] = None,
        batch_size: int = 128
) -> None:
    """Plot mixture weights distribution and specialization."""
    # Get MDN parameters
    mdn_params = model.encode(data)

    # Extract mixture weights
    _, _, pi_logits = model.mdn_layer.split_mixture_params(mdn_params)
    pi_np = keras.ops.convert_to_numpy(keras.activations.softmax(pi_logits, axis=-1))

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Average mixture weights
    ax = axes[0, 0]
    avg_weights = np.mean(pi_np, axis=0)
    ax.bar(range(model.num_mixtures), avg_weights, color='skyblue', edgecolor='navy')
    ax.set_xlabel('Mixture Component')
    ax.set_ylabel('Average Weight')
    ax.set_title('Average Mixture Weights Across All Samples')
    ax.set_ylim(0, 1)

    # 2. Mixture weights per class (heatmap)
    ax = axes[0, 1]
    if labels.ndim > 1:
        labels = labels.flatten()
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    weights_per_class = np.zeros((num_classes, model.num_mixtures))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        weights_per_class[i] = np.mean(pi_np[mask], axis=0)

    im = ax.imshow(weights_per_class, aspect='auto', cmap='YlOrRd')
    ax.set_xlabel('Mixture Component')
    ax.set_ylabel('Class')
    ax.set_title('Average Mixture Weights per Class')
    ax.set_xticks(range(model.num_mixtures))
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(unique_labels)
    plt.colorbar(im, ax=ax)

    # 3. Entropy of mixture weights
    ax = axes[1, 0]
    entropy = -np.sum(pi_np * np.log(pi_np + 1e-10), axis=1)
    ax.hist(entropy, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Mixture Weight Entropy')
    ax.axvline(np.mean(entropy), color='red', linestyle='--', label=f'Mean: {np.mean(entropy):.3f}')
    ax.legend()

    # 4. Dominant component per class
    ax = axes[1, 1]
    dominant_comp_per_sample = np.argmax(pi_np, axis=1)
    dominant_comp_counts = np.zeros((num_classes, model.num_mixtures))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        for j in range(model.num_mixtures):
            dominant_comp_counts[i, j] = np.sum(dominant_comp_per_sample[mask] == j)

    # Normalize to percentages
    dominant_comp_counts = dominant_comp_counts / dominant_comp_counts.sum(axis=1, keepdims=True) * 100

    im = ax.imshow(dominant_comp_counts, aspect='auto', cmap='Blues')
    ax.set_xlabel('Mixture Component')
    ax.set_ylabel('Class')
    ax.set_title('Dominant Component Distribution per Class (%)')
    ax.set_xticks(range(model.num_mixtures))
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(unique_labels)
    plt.colorbar(im, ax=ax, label='Percentage')

    title = f'MDN-VAE Mixture Analysis - Epoch {epoch}' if epoch is not None else 'Final MDN-VAE Mixture Analysis'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------

def plot_latent_space_mdn(
        model: MDN_VAE,
        data: np.ndarray,
        labels: np.ndarray,
        save_path: str,
        epoch: Optional[int] = None,
        batch_size: int = 128
) -> None:
    """Plot the MDN latent space with mixture components."""
    # Get MDN parameters and samples
    outputs = model.predict(data, batch_size=batch_size, verbose=0)
    z_samples = outputs['z']
    mdn_params = outputs['mdn_params']

    # Extract mixture parameters
    mu, sigma, pi_logits = model.mdn_layer.split_mixture_params(mdn_params)
    pi_np = keras.ops.convert_to_numpy(keras.activations.softmax(pi_logits, axis=-1))
    mu_np = keras.ops.convert_to_numpy(mu)

    # If latent dim > 2, use t-SNE
    if model.latent_dim > 2:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        z_2d = tsne.fit_transform(keras.ops.convert_to_numpy(z_samples))

        # Also project means
        all_means = mu_np.reshape(-1, model.latent_dim)
        means_2d = tsne.fit_transform(all_means)
        means_2d = means_2d.reshape(mu_np.shape[0], model.num_mixtures, 2)
    else:
        z_2d = keras.ops.convert_to_numpy(z_samples[:, :2])
        means_2d = mu_np[:, :, :2]

    if labels.ndim > 1:
        labels = labels.flatten()

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # 1. Samples colored by class
    ax = axes[0]
    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10', s=20, alpha=0.6)
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_title('MDN-VAE Latent Space Samples')
    plt.colorbar(scatter, ax=ax, label='Class')

    # 2. Mixture component means
    ax = axes[1]

    # Plot all component means with transparency based on their weights
    for i in range(len(data)):
        for j in range(model.num_mixtures):
            ax.scatter(
                means_2d[i, j, 0],
                means_2d[i, j, 1],
                c=[labels[i]],
                cmap='tab10',
                s=50 * pi_np[i, j],  # Size proportional to weight
                alpha=pi_np[i, j] * 0.8,  # Transparency based on weight
                edgecolors='black',
                linewidth=0.5
            )

    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_title('Mixture Component Means (size ∝ weight)')

    # Set consistent axis limits
    all_points = np.vstack([z_2d, means_2d.reshape(-1, 2)])
    xlim = (np.min(all_points[:, 0]) - 1, np.max(all_points[:, 0]) + 1)
    ylim = (np.min(all_points[:, 1]) - 1, np.max(all_points[:, 1]) + 1)

    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)

    title = f'MDN-VAE Latent Space - Epoch {epoch}' if epoch is not None else 'Final MDN-VAE Latent Space'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------

def plot_interpolation_mdn(
        model: MDN_VAE,
        data: np.ndarray,
        save_path: str,
        n_interpolations: int = 5,
        n_steps: int = 10,
        epoch: Optional[int] = None,
        dataset: str = 'mnist'
) -> None:
    """Plot interpolations in MDN latent space."""
    # Select pairs of samples
    indices = np.random.choice(len(data), size=n_interpolations * 2, replace=False)
    sample_pairs = data[indices].reshape(n_interpolations, 2, *data.shape[1:])

    fig, axes = plt.subplots(n_interpolations, n_steps, figsize=(n_steps * 1.2, n_interpolations * 1.2))
    if n_interpolations == 1:
        axes = axes.reshape(1, -1)

    cmap = 'gray' if dataset.lower() == 'mnist' else None

    for i, (img1, img2) in enumerate(sample_pairs):
        # Encode both images
        img1_batch = np.expand_dims(img1, axis=0)
        img2_batch = np.expand_dims(img2, axis=0)

        # Get MDN parameters
        mdn_params1 = model.encode(img1_batch)
        mdn_params2 = model.encode(img2_batch)

        # Extract mixture parameters
        mu1, sigma1, pi1 = model.mdn_layer.split_mixture_params(mdn_params1)
        mu2, sigma2, pi2 = model.mdn_layer.split_mixture_params(mdn_params2)

        # Convert to numpy
        mu1_np = keras.ops.convert_to_numpy(mu1)
        mu2_np = keras.ops.convert_to_numpy(mu2)

        # Use the mean of the most probable component for interpolation
        pi1_np = keras.ops.convert_to_numpy(keras.activations.softmax(pi1, axis=-1))
        pi2_np = keras.ops.convert_to_numpy(keras.activations.softmax(pi2, axis=-1))

        idx1 = np.argmax(pi1_np[0])
        idx2 = np.argmax(pi2_np[0])

        z1 = mu1_np[0, idx1, :]
        z2 = mu2_np[0, idx2, :]

        # Interpolate
        alphas = np.linspace(0, 1, n_steps)

        for j, alpha in enumerate(alphas):
            z_interp = (1 - alpha) * z1 + alpha * z2
            z_interp = np.expand_dims(z_interp, axis=0)

            # Decode
            reconstructed = model.decode(z_interp)

            # Plot
            ax = axes[i, j]
            img_to_plot = keras.ops.squeeze(reconstructed[0])
            img_to_plot = np.array(img_to_plot)
            ax.imshow(np.clip(img_to_plot, 0, 1), cmap=cmap)
            ax.axis('off')

            if i == 0:
                ax.set_title(f'α={alpha:.1f}', fontsize=8)

    title = f'MDN-VAE Latent Interpolation - Epoch {epoch}' if epoch is not None else 'Final MDN-VAE Interpolation'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------

class MDNVisualizationCallback(keras.callbacks.Callback):
    """Callback to generate MDN-VAE visualizations during training."""

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

        # Create directories
        self.recon_dir = os.path.join(save_dir, 'reconstructions_per_epoch')
        self.latent_dir = os.path.join(save_dir, 'latent_space_per_epoch')
        self.interp_dir = os.path.join(save_dir, 'interpolations_per_epoch')
        self.mixture_dir = os.path.join(save_dir, 'mixture_analysis_per_epoch')

        os.makedirs(self.recon_dir, exist_ok=True)
        os.makedirs(self.latent_dir, exist_ok=True)
        os.makedirs(self.interp_dir, exist_ok=True)
        os.makedirs(self.mixture_dir, exist_ok=True)

        # Sample images for consistent visualization
        indices = np.random.choice(len(self.x_val), size=10, replace=False)
        self.sample_images = self.x_val[indices]

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        if (epoch + 1) % self.frequency == 0 or epoch == 0:
            logger.info(f"Generating MDN-VAE visualizations for epoch {epoch + 1}...")

            # Reconstruction visualization
            outputs = self.model.predict(self.sample_images, verbose=0)
            reconstructions = outputs["reconstruction"]
            recon_path = os.path.join(self.recon_dir, f'epoch_{epoch + 1:03d}.png')
            plot_reconstruction_comparison(
                self.sample_images, reconstructions, recon_path,
                epoch + 1, self.dataset
            )

            # Mixture weights visualization
            mixture_path = os.path.join(self.mixture_dir, f'epoch_{epoch + 1:03d}.png')
            plot_mixture_weights(
                self.model, self.x_val[:2000], self.y_val[:2000],
                mixture_path, epoch + 1, self.batch_size
            )

            # Latent space visualization
            latent_path = os.path.join(self.latent_dir, f'epoch_{epoch + 1:03d}.png')
            plot_latent_space_mdn(
                self.model, self.x_val[:2000], self.y_val[:2000],
                latent_path, epoch + 1, self.batch_size
            )

            # Interpolation visualization
            interp_path = os.path.join(self.interp_dir, f'epoch_{epoch + 1:03d}.png')
            plot_interpolation_mdn(
                self.model, self.x_val[:100], interp_path,
                n_interpolations=5, n_steps=10, epoch=epoch + 1,
                dataset=self.dataset
            )

# ---------------------------------------------------------------------

def create_callbacks(
        model_name: str,
        validation_data: Tuple[np.ndarray, np.ndarray],
        dataset: str,
        batch_size: int,
        monitor: str = 'val_total_loss',
        patience: int = 15,  # Increased patience for MDN
        viz_frequency: int = 5
) -> Tuple[list, str]:
    """Create training callbacks."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"mdn_vae_{model_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience,
            restore_best_weights=True, verbose=1, mode='min'
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(results_dir, 'best_model.keras'),
            monitor=monitor, save_best_only=True, verbose=1, mode='min'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=7,
            min_lr=1e-7, verbose=1, mode='min'
        ),
        keras.callbacks.CSVLogger(
            filename=os.path.join(results_dir, 'training_log.csv')
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(results_dir, 'tensorboard'),
            histogram_freq=1
        ),
        MDNVisualizationCallback(
            validation_data, results_dir, dataset,
            viz_frequency, batch_size
        )
    ]

    logger.info(f"Results will be saved to: {results_dir}")
    return callbacks, results_dir

# ---------------------------------------------------------------------

def plot_training_history(history: keras.callbacks.History, save_dir: str):
    """Plot training history curves."""
    history_dict = history.history

    # Use the correct keys
    loss_key = 'total_loss'
    recon_loss_key = 'reconstruction_loss'
    kl_loss_key = 'kl_loss'

    if loss_key not in history_dict:
        logger.error(f"Key '{loss_key}' not found. Available: {list(history_dict.keys())}")
        return

    epochs = range(1, len(history_dict[loss_key]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle("MDN-VAE Training and Validation Loss", fontsize=16, fontweight='bold')

    # Total Loss
    axes[0].plot(epochs, history_dict[loss_key], 'b-', label='Training Loss')
    if f'val_{loss_key}' in history_dict:
        axes[0].plot(epochs, history_dict[f'val_{loss_key}'], 'r-', label='Validation Loss')
    axes[0].set_title('Total Loss', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_yscale('log')  # Log scale for better visualization

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
    logger.info("Starting MDN-VAE training script")
    logger.info(f"Configuration: latent_dim={args.latent_dim}, num_mixtures={args.num_mixtures}")
    setup_gpu()

    # Load data
    if args.dataset.lower() == 'mnist':
        (x_train, y_train), (x_test, y_test) = load_mnist_data()
    elif args.dataset.lower() == 'cifar10':
        (x_train, y_train), (x_test, y_test) = load_cifar10_data()
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Create model
    model_config = create_model_config(args.dataset, args.latent_dim, args.num_mixtures)

    # Configure optimizer with learning rate
    if args.optimizer == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    elif args.optimizer == 'adamw':
        optimizer = keras.optimizers.AdamW(learning_rate=args.learning_rate)
    else:
        optimizer = args.optimizer

    model = create_mdn_vae(
        optimizer=optimizer,
        **model_config
    )

    model.summary(print_fn=logger.info)
    logger.info(f"Total parameters: {model.count_params():,}")

    # Create callbacks
    callbacks, results_dir = create_callbacks(
        model_name=f"{args.dataset}_mix{args.num_mixtures}",
        validation_data=(x_test, y_test),
        dataset=args.dataset,
        batch_size=args.batch_size,
        monitor='val_total_loss',
        patience=args.patience,
        viz_frequency=args.viz_frequency
    )

    # Train model
    logger.info("Starting model training...")
    history = model.fit(
        x_train,
        validation_data=(x_test, None),  # VAE doesn't need y for validation
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate best model
    logger.info("Training completed. Evaluating on test set...")
    best_model_path = os.path.join(results_dir, 'best_model.keras')

    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from: {best_model_path}")
        # Load with custom objects
        best_model = keras.models.load_model(
            best_model_path,
            custom_objects={
                "MDN_VAE": MDN_VAE,
                "MDNLayer": MDNLayer,
                "SamplingMDN": model.sampling_layer.__class__
            }
        )
    else:
        logger.warning("No best model found, using final model state.")
        best_model = model

    # Evaluate
    test_results = best_model.evaluate(
        x_test, batch_size=args.batch_size, verbose=1, return_dict=True
    )
    logger.info(f"Final Test Results: {test_results}")

    # Generate final visualizations
    logger.info("Generating final visualizations...")
    plot_training_history(history, results_dir)

    # Final visualizations
    final_recon_path = os.path.join(results_dir, 'final_reconstructions.png')
    final_latent_path = os.path.join(results_dir, 'final_latent_space.png')
    final_interp_path = os.path.join(results_dir, 'final_interpolations.png')
    final_mixture_path = os.path.join(results_dir, 'final_mixture_analysis.png')

    # Sample images
    sample_indices = np.random.choice(len(x_test), size=10, replace=False)
    sample_images = x_test[sample_indices]
    outputs = best_model.predict(sample_images, verbose=0)

    # Generate visualizations
    plot_reconstruction_comparison(
        sample_images, outputs['reconstruction'],
        final_recon_path, dataset=args.dataset
    )
    plot_latent_space_mdn(
        best_model, x_test[:5000], y_test[:5000],
        final_latent_path, batch_size=args.batch_size
    )
    plot_interpolation_mdn(
        best_model, x_test[:100], final_interp_path,
        n_interpolations=5, n_steps=10, dataset=args.dataset
    )
    plot_mixture_weights(
        best_model, x_test[:5000], y_test[:5000],
        final_mixture_path, batch_size=args.batch_size
    )

    # Save final model
    final_model_path = os.path.join(results_dir, f"mdn_vae_{args.dataset}_final.keras")
    best_model.save(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")

    # Save training summary
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"MDN-VAE Training Summary\n")
        f.write(f"========================\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Latent Dim: {args.latent_dim}\n")
        f.write(f"Num Mixtures: {args.num_mixtures}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Stopped at Epoch: {len(history.history['total_loss'])}\n\n")
        f.write(f"Model Configuration:\n")
        for key, val in model_config.items():
            f.write(f"  {key}: {val}\n")
        f.write(f"\nFinal Test Results:\n")
        for key, val in test_results.items():
            f.write(f"  {key.replace('_', ' ').title()}: {val:.6f}\n")

# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train a Mixture Density Network VAE (MDN-VAE) on image data.'
    )
    parser.add_argument(
        '--dataset', type=str, default='mnist',
        choices=['mnist', 'cifar10'], help='Dataset to use'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=128,
        help='Training batch size'
    )
    parser.add_argument(
        '--latent-dim', type=int, default=10,
        help='Dimensionality of the latent space'
    )
    parser.add_argument(
        '--num-mixtures', type=int, default=5,
        help='Number of Gaussian mixture components'
    )
    parser.add_argument(
        '--optimizer', type=str, default='adam',
        choices=['adam', 'adamw'], help='Optimizer to use'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=1e-3,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--patience', type=int, default=15,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--viz-frequency', type=int, default=5,
        help='Frequency of visualization callbacks (in epochs)'
    )

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