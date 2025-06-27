#!/usr/bin/env python3
# train_vae.py

"""
Enhanced training script for Keras-compliant VAE with comprehensive visualizations.

This script demonstrates how to train the VAE model using standard Keras
workflows with model.compile() and model.fit(). The script handles data loading,
model creation, training, evaluation, and generates comprehensive visualizations
including per-epoch reconstructions and training plots.

Usage:
    python train_vae.py [--dataset mnist] [--epochs 50] [--batch-size 128] [--latent-dim 2]
"""

import argparse
import os
import numpy as np
import keras
import tensorflow as tf
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

from dl_techniques.models.vae import VAE, create_vae
from dl_techniques.utils.logger import logger

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


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
    """Load and preprocess MNIST dataset for VAE."""
    logger.info("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0
    logger.info(f"Train data shape: {x_train.shape}")
    logger.info(f"Test data shape: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)


def load_cifar10_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess CIFAR-10 dataset for VAE."""
    logger.info("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    logger.info(f"Train data shape: {x_train.shape}")
    logger.info(f"Test data shape: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)


def create_model_config(dataset: str, latent_dim: int) -> Dict[str, Any]:
    """Create VAE model configuration based on dataset."""
    if dataset.lower() == 'mnist':
        return {
            'latent_dim': latent_dim,
            'input_shape': (28, 28, 1),
            'encoder_filters': [32, 64],
            'decoder_filters': [64, 32],
            'kl_loss_weight': 1.0,
            'use_batch_norm': True,
        }
    elif dataset.lower() == 'cifar10':
        return {
            'latent_dim': latent_dim,
            'input_shape': (32, 32, 3),
            'encoder_filters': [32, 64, 128],
            'decoder_filters': [128, 64, 32],
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


def plot_latent_space(
        model: VAE,
        data: np.ndarray,
        labels: np.ndarray,
        save_path: str,
        epoch: Optional[int] = None
) -> None:
    """Plot the latent space colored by class labels."""
    if model.latent_dim != 2:
        logger.warning(f"Latent space plotting is only available for latent_dim=2, skipping.")
        return

    encoder_output = model.encoder.predict(data, batch_size=128, verbose=0)
    z_mean = encoder_output[:, :model.latent_dim]

    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap='viridis', s=5, alpha=0.7)
    plt.colorbar(label='Digit Class')
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    title = f'Latent Space Distribution - Epoch {epoch}' if epoch is not None else 'Final Latent Space'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


class VisualizationCallback(keras.callbacks.Callback):
    """Callback to generate VAE visualizations during training."""

    def __init__(
            self,
            validation_data: Tuple[np.ndarray, np.ndarray],
            save_dir: str,
            dataset: str,
            frequency: int = 5
    ):
        super().__init__()
        self.x_val, self.y_val = validation_data
        self.save_dir = save_dir
        self.dataset = dataset
        self.frequency = frequency

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
            plot_latent_space(self.model, self.x_val, self.y_val, latent_path, epoch + 1)


def create_callbacks(
        model_name: str,
        validation_data: Tuple[np.ndarray, np.ndarray],
        dataset: str,
        monitor: str = 'val_loss',
        patience: int = 10,
        save_best_only: bool = True,
        viz_frequency: int = 5
) -> Tuple[list, str]:
    """Create training callbacks."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/vae_{model_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(results_dir, 'best_model.keras'),
            monitor=monitor, save_best_only=save_best_only, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        keras.callbacks.CSVLogger(filename=os.path.join(results_dir, 'training_log.csv'), append=False),
        keras.callbacks.TensorBoard(log_dir=os.path.join(results_dir, 'tensorboard'), histogram_freq=1),
        VisualizationCallback(validation_data, results_dir, dataset, viz_frequency)
    ]
    logger.info(f"Results will be saved to: {results_dir}")
    return callbacks, results_dir


def plot_training_history(history: keras.callbacks.History, save_dir: str):
    """Plots training history curves."""
    history_dict = history.history
    epochs = range(1, len(history_dict['loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Total Loss
    axes[0].plot(epochs, history_dict['loss'], 'b-', label='Training Loss')
    axes[0].plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss')
    axes[0].set_title('Total Loss', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reconstruction Loss
    axes[1].plot(epochs, history_dict['reconstruction_loss'], 'b-', label='Training Recon Loss')
    axes[1].plot(epochs, history_dict['val_reconstruction_loss'], 'r-', label='Validation Recon Loss')
    axes[1].set_title('Reconstruction Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # KL Loss
    axes[2].plot(epochs, history_dict['kl_loss'], 'b-', label='Training KL Loss')
    axes[2].plot(epochs, history_dict['val_kl_loss'], 'r-', label='Validation KL Loss')
    axes[2].set_title('KL Divergence Loss', fontsize=14)
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()


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
        learning_rate=args.learning_rate,
        **model_config
    )

    sample_input = tf.zeros((1,) + model_config['input_shape'])
    _ = model(sample_input)
    model.summary(print_fn=logger.info)

    callbacks, results_dir = create_callbacks(
        model_name=args.dataset,
        validation_data=(x_test, y_test),
        dataset=args.dataset,
        patience=args.patience,
        viz_frequency=args.viz_frequency
    )

    logger.info("Starting model training...")
    history = model.fit(
        x_train,
        validation_data=(x_test, None),  # VAE is unsupervised, labels not used in val
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    logger.info("Training completed. Evaluating on test set...")
    test_results = model.evaluate(x_test, batch_size=args.batch_size, verbose=1)
    test_results_dict = dict(zip(model.metrics_names, test_results))
    logger.info(f"Final Test Results: {test_results_dict}")

    logger.info("Generating final visualizations...")
    plot_training_history(history, results_dir)
    final_recon_path = os.path.join(results_dir, 'final_reconstructions.png')
    final_latent_path = os.path.join(results_dir, 'final_latent_space.png')

    sample_indices = np.random.choice(len(x_test), size=10, replace=False)
    sample_images = x_test[sample_indices]
    outputs = model.predict(sample_images, verbose=0)
    plot_reconstruction_comparison(sample_images, outputs['reconstruction'], final_recon_path, dataset=args.dataset)
    plot_latent_space(model, x_test[:5000], y_test[:5000], final_latent_path)  # Plot a subset of test set

    final_model_path = os.path.join(results_dir, f"vae_{args.dataset}_final.keras")
    model.save(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")

    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"VAE Training Summary\n")
        f.write("====================\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Latent Dim: {args.latent_dim}\n")
        f.write(f"Epochs: {len(history.history['loss'])}\n")
        for key, val in test_results_dict.items():
            f.write(f"Final Test {key.replace('_', ' ').title()}: {val:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description='Train a Variational Autoencoder (VAE) on image data.')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'], help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--latent-dim', type=int, default=2, help='Dimensionality of the latent space')
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


if __name__ == '__main__':
    main()