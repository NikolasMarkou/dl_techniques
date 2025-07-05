"""
CapsNet with comprehensive visualizations.

This script demonstrates how to train the CapsNet model using standard Keras
workflows with model.compile() and model.fit(). The script handles data loading,
model creation, training, evaluation, and generates comprehensive visualizations
including per-epoch reconstructions and training plots.

Usage:
    python train.py [--dataset mnist] [--epochs 50] [--batch-size 32]
"""


import os
import keras
import argparse
import matplotlib
import numpy as np
import tensorflow as tf
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.models.capsnet import create_capsnet
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ---------------------------------------------------------------------

def setup_gpu():
    """Configure GPU settings for optimal training."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")
    else:
        logger.info("No GPUs found, using CPU")

# ---------------------------------------------------------------------

def load_mnist_data(
        validation_split: float = 0.1,
        normalize: bool = True
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess MNIST dataset.

    Args:
        validation_split: Fraction of training data to use for validation.
        normalize: Whether to normalize pixel values to [0, 1].

    Returns:
        Tuple of (train_data, val_data, test_data) where each is (x, y).
    """
    logger.info("Loading MNIST dataset...")

    # Load raw data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Add channel dimension
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Normalize if requested
    if normalize:
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

    # Convert labels to one-hot encoding
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Create validation split
    if validation_split > 0:
        val_size = int(len(x_train) * validation_split)
        x_val = x_train[:val_size]
        y_val = y_train[:val_size]
        x_train = x_train[val_size:]
        y_train = y_train[val_size:]
    else:
        x_val, y_val = x_test, y_test  # Use test set as validation

    logger.info(f"Dataset shapes:")
    logger.info(f"  Train: {x_train.shape}, {y_train.shape}")
    logger.info(f"  Validation: {x_val.shape}, {y_val.shape}")
    logger.info(f"  Test: {x_test.shape}, {y_test.shape}")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def load_cifar10_data(
        validation_split: float = 0.1,
        normalize: bool = True
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess CIFAR-10 dataset.

    Args:
        validation_split: Fraction of training data to use for validation.
        normalize: Whether to normalize pixel values to [0, 1].

    Returns:
        Tuple of (train_data, val_data, test_data) where each is (x, y).
    """
    logger.info("Loading CIFAR-10 dataset...")

    # Load raw data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize if requested
    if normalize:
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

    # Convert labels to one-hot encoding
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Create validation split
    if validation_split > 0:
        val_size = int(len(x_train) * validation_split)
        x_val = x_train[:val_size]
        y_val = y_train[:val_size]
        x_train = x_train[val_size:]
        y_train = y_train[val_size:]
    else:
        x_val, y_val = x_test, y_test  # Use test set as validation

    logger.info(f"Dataset shapes:")
    logger.info(f"  Train: {x_train.shape}, {y_train.shape}")
    logger.info(f"  Validation: {x_val.shape}, {y_val.shape}")
    logger.info(f"  Test: {x_test.shape}, {y_test.shape}")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def create_model_config(dataset: str) -> Dict[str, Any]:
    """Create model configuration based on dataset.

    Args:
        dataset: Name of dataset ('mnist' or 'cifar10').

    Returns:
        Dictionary of model configuration parameters.
    """
    if dataset.lower() == 'mnist':
        return {
            'num_classes': 10,
            'input_shape': (28, 28, 1),
            'conv_filters': [256, 256],
            'primary_capsules': 16,
            'primary_capsule_dim': 4,
            'digit_capsule_dim': 8,
            'reconstruction': True,
            'decoder_architecture': [256, 512],
            'positive_margin': 0.9,
            'negative_margin': 0.1,
            'downweight': 0.5,
            'reconstruction_weight': 0.1,
            'use_batch_norm': True
        }
    elif dataset.lower() == 'cifar10':
        return {
            'num_classes': 10,
            'input_shape': (32, 32, 3),
            'conv_filters': [128, 256, 256],  # More layers for complex images
            'primary_capsules': 32,
            'primary_capsule_dim': 8,
            'digit_capsule_dim': 16,
            'reconstruction': True,
            'decoder_architecture': [512, 1024],
            'positive_margin': 0.9,
            'negative_margin': 0.1,
            'downweight': 0.5,
            'reconstruction_weight': 0.1,
            'use_batch_norm': True
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def plot_reconstruction_comparison(
        original_images: np.ndarray,
        reconstructed_images: np.ndarray,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        save_path: str,
        epoch: Optional[int] = None,
        dataset: str = 'mnist'
) -> None:
    """Plot original vs reconstructed images comparison.

    Args:
        original_images: Original input images.
        reconstructed_images: Reconstructed images from CapsNet.
        true_labels: True class labels.
        predicted_labels: Predicted class labels.
        save_path: Path to save the plot.
        epoch: Current epoch number (for title).
        dataset: Dataset name for proper visualization.
    """
    n_samples = min(8, len(original_images))
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))

    if dataset.lower() == 'mnist':
        cmap = 'gray'
    else:
        cmap = None

    for i in range(n_samples):
        # Original image
        if dataset.lower() == 'mnist':
            img_orig = original_images[i].squeeze()
            img_recon = reconstructed_images[i].squeeze()
        else:
            img_orig = original_images[i]
            img_recon = reconstructed_images[i]

        axes[0, i].imshow(img_orig, cmap=cmap)
        axes[0, i].set_title(f'Original\nTrue: {true_labels[i]}', fontsize=10)
        axes[0, i].axis('off')

        # Reconstructed image
        axes[1, i].imshow(np.clip(img_recon, 0, 1), cmap=cmap)
        axes[1, i].set_title(f'Reconstructed\nPred: {predicted_labels[i]}', fontsize=10)
        axes[1, i].axis('off')

    if epoch is not None:
        fig.suptitle(f'Reconstruction Comparison - Epoch {epoch}', fontsize=14, fontweight='bold')
    else:
        fig.suptitle('Final Reconstruction Comparison', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(history: keras.callbacks.History, save_dir: str) -> None:
    """Plot training history including loss and accuracy curves.

    Args:
        history: Keras training history object.
        save_dir: Directory to save plots.
    """
    history_dict = history.history
    epochs = range(1, len(history_dict['loss']) + 1)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot training and validation loss
    axes[0, 0].plot(epochs, history_dict['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot training and validation accuracy
    if 'capsule_accuracy' in history_dict:
        axes[0, 1].plot(epochs, history_dict['capsule_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        if 'val_capsule_accuracy' in history_dict:
            axes[0, 1].plot(epochs, history_dict['val_capsule_accuracy'], 'r-', label='Validation Accuracy',
                            linewidth=2)
        else:
            # If no validation accuracy, add a note
            axes[0, 1].text(0.5, 0.5, 'Validation Accuracy\nNot Available',
                            transform=axes[0, 1].transAxes, ha='center', va='center',
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        # Hide the accuracy plot if no accuracy metrics are available
        axes[0, 1].text(0.5, 0.5, 'Accuracy Metrics\nNot Available',
                        transform=axes[0, 1].transAxes, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')

    # Plot reconstruction loss if available
    if 'reconstruction_loss' in history_dict:
        axes[1, 0].plot(epochs, history_dict['reconstruction_loss'], 'g-', label='Training Recon Loss', linewidth=2)
        axes[1, 0].plot(epochs, history_dict['val_reconstruction_loss'], 'm-', label='Validation Recon Loss',
                        linewidth=2)
        axes[1, 0].set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Reconstruction Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Reconstruction Loss\nNot Available',
                        transform=axes[1, 0].transAxes, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        axes[1, 0].set_title('Reconstruction Loss', fontsize=14, fontweight='bold')

    # Plot learning rate if available, otherwise show epoch vs combined metrics
    if 'lr' in history_dict:
        axes[1, 1].plot(epochs, history_dict['lr'], 'orange', linewidth=2)
        axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Plot margin loss if available, or loss components
        if 'margin_loss' in history_dict:
            axes[1, 1].plot(epochs, history_dict['margin_loss'], 'purple', label='Training Margin Loss', linewidth=2)
            if 'val_margin_loss' in history_dict:
                axes[1, 1].plot(epochs, history_dict['val_margin_loss'], 'orange', label='Val Margin Loss', linewidth=2)
            axes[1, 1].set_title('Margin Loss', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Margin Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Show loss difference plot
            loss_diff = np.array(history_dict['val_loss']) - np.array(history_dict['loss'])
            axes[1, 1].plot(epochs, loss_diff, 'red', linewidth=2)
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 1].set_title('Validation - Training Loss', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Difference')
            axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_summary(history: keras.callbacks.History, test_results: Dict[str, float], save_dir: str) -> None:
    """Plot a summary of all metrics.

    Args:
        history: Keras training history object.
        test_results: Dictionary of test results.
        save_dir: Directory to save plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Final metrics comparison
    available_metrics = []
    train_values = []
    val_values = []
    test_values = []

    # Always include loss
    available_metrics.append('Loss')
    train_values.append(history.history['loss'][-1])
    val_values.append(history.history['val_loss'][-1])
    test_values.append(test_results.get('loss', 0))

    # Include accuracy only if available
    if 'capsule_accuracy' in history.history:
        available_metrics.append('Accuracy')
        train_values.append(history.history['capsule_accuracy'][-1])
        val_values.append(
            history.history.get('val_capsule_accuracy', [0])[-1] if 'val_capsule_accuracy' in history.history else 0)
        test_values.append(test_results.get('capsule_accuracy', 0))

    # Include reconstruction loss if available
    if 'reconstruction_loss' in history.history:
        available_metrics.append('Recon Loss')
        train_values.append(history.history['reconstruction_loss'][-1])
        val_values.append(history.history['val_reconstruction_loss'][-1])
        test_values.append(test_results.get('reconstruction_loss', 0))

    x = np.arange(len(available_metrics))
    width = 0.25

    bars1 = axes[0, 0].bar(x - width, train_values, width, label='Train', alpha=0.8, color='lightblue')
    bars2 = axes[0, 0].bar(x, val_values, width, label='Validation', alpha=0.8, color='lightcoral')
    bars3 = axes[0, 0].bar(x + width, test_values, width, label='Test', alpha=0.8, color='lightgreen')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axes[0, 0].annotate(f'{height:.3f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=9)

    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Values')
    axes[0, 0].set_title('Final Metrics Comparison', fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(available_metrics)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Training progress
    epochs = range(1, len(history.history['loss']) + 1)
    axes[0, 1].plot(epochs, history.history['loss'], label='Train Loss', linewidth=2, color='blue')
    axes[0, 1].plot(epochs, history.history['val_loss'], label='Val Loss', linewidth=2, color='red')

    if 'capsule_accuracy' in history.history:
        ax2 = axes[0, 1].twinx()
        ax2.plot(epochs, history.history['capsule_accuracy'], '--', label='Train Acc', linewidth=2, color='lightblue')
        if 'val_capsule_accuracy' in history.history:
            ax2.plot(epochs, history.history['val_capsule_accuracy'], '--', label='Val Acc', linewidth=2,
                     color='lightcoral')
        ax2.set_ylabel('Accuracy')
        ax2.legend(loc='center right')
        ax2.set_ylim([0.95, 1.0])  # Focus on the accuracy range

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Progress', fontweight='bold')
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Loss components breakdown
    if 'margin_loss' in history.history:
        axes[1, 0].plot(epochs, history.history['loss'], label='Total Loss', linewidth=2)
        axes[1, 0].plot(epochs, history.history['margin_loss'], label='Margin Loss', linewidth=2)
        if 'reconstruction_loss' in history.history:
            # Scale reconstruction loss for comparison
            recon_scaled = np.array(history.history['reconstruction_loss']) * 0.0005  # reconstruction weight
            axes[1, 0].plot(epochs, recon_scaled, label='Recon Loss (scaled)', linewidth=2)
        axes[1, 0].set_title('Loss Components', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        # Show reconstruction loss if available
        if 'reconstruction_loss' in history.history:
            axes[1, 0].plot(epochs, history.history['reconstruction_loss'], 'g-', label='Training', linewidth=2)
            axes[1, 0].plot(epochs, history.history['val_reconstruction_loss'], 'm-', label='Validation', linewidth=2)
            axes[1, 0].set_title('Reconstruction Loss Detail', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Reconstruction Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Additional Loss\nComponents\nNot Available',
                            transform=axes[1, 0].transAxes, ha='center', va='center',
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            axes[1, 0].set_title('Loss Components', fontweight='bold')

    # Plot 4: Training stability metrics
    if len(epochs) > 5:
        # Calculate rolling averages for stability
        window = max(3, len(epochs) // 10)
        loss_rolling = pd.Series(history.history['loss']).rolling(window=window, center=True).mean()
        val_loss_rolling = pd.Series(history.history['val_loss']).rolling(window=window, center=True).mean()

        axes[1, 1].plot(epochs, history.history['loss'], alpha=0.3, color='blue', label='Raw Train Loss')
        axes[1, 1].plot(epochs, history.history['val_loss'], alpha=0.3, color='red', label='Raw Val Loss')
        axes[1, 1].plot(epochs, loss_rolling, linewidth=2, color='blue', label=f'Train Loss (MA-{window})')
        axes[1, 1].plot(epochs, val_loss_rolling, linewidth=2, color='red', label=f'Val Loss (MA-{window})')

        axes[1, 1].set_title('Training Stability (Moving Average)', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Not enough epochs for moving average, show final epochs summary
        final_epochs = min(5, len(epochs))
        final_train_loss = np.mean(history.history['loss'][-final_epochs:])
        final_val_loss = np.mean(history.history['val_loss'][-final_epochs:])

        summary_text = f"Final {final_epochs} epochs average:\n"
        summary_text += f"Train Loss: {final_train_loss:.4f}\n"
        summary_text += f"Val Loss: {final_val_loss:.4f}\n"
        summary_text += f"Overfitting: {final_val_loss / final_train_loss:.2f}x"

        axes[1, 1].text(0.5, 0.5, summary_text,
                        transform=axes[1, 1].transAxes, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                        fontsize=12)
        axes[1, 1].set_title('Training Summary', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()


class LearningRateLogger(keras.callbacks.Callback):
    """Custom callback to log learning rate to history."""

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        # Get learning rate from optimizer
        lr = float(self.model.optimizer.learning_rate)
        logs['lr'] = lr


class ValidationAccuracyLogger(keras.callbacks.Callback):
    """Custom callback to manually compute and log validation accuracy if not available."""

    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # Only compute if validation accuracy is not already available
        if 'val_capsule_accuracy' not in logs:
            x_val, y_val = self.validation_data
            predictions = self.model.predict(x_val, verbose=0)

            # Handle different prediction formats
            if isinstance(predictions, dict):
                # Find the length/capsule key
                pred_key = None
                for key in predictions.keys():
                    if 'length' in key.lower() or 'capsule' in key.lower() or 'output' in key.lower():
                        pred_key = key
                        break
                if pred_key is None:
                    pred_key = list(predictions.keys())[0]
                pred_classes = np.argmax(predictions[pred_key], axis=1)
            elif isinstance(predictions, (list, tuple)):
                pred_classes = np.argmax(predictions[0], axis=1)
            else:
                pred_classes = np.argmax(predictions, axis=1)

            true_classes = np.argmax(y_val, axis=1)
            accuracy = np.mean(pred_classes == true_classes)
            logs['val_capsule_accuracy'] = accuracy


class ReconstructionVisualizationCallback(keras.callbacks.Callback):
    """Custom callback to generate reconstruction visualizations during training."""

    def __init__(
            self,
            validation_data: Tuple[np.ndarray, np.ndarray],
            save_dir: str,
            dataset: str = 'mnist',
            n_samples: int = 8,
            frequency: int = 5
    ):
        """Initialize the callback.

        Args:
            validation_data: Tuple of (x_val, y_val) for generating reconstructions.
            save_dir: Directory to save visualization plots.
            dataset: Dataset name ('mnist' or 'cifar10').
            n_samples: Number of samples to visualize.
            frequency: How often to generate visualizations (every N epochs).
        """
        super().__init__()
        self.x_val, self.y_val = validation_data
        self.save_dir = save_dir
        self.dataset = dataset
        self.n_samples = n_samples
        self.frequency = frequency

        # Create reconstruction directory
        self.recon_dir = os.path.join(save_dir, 'reconstructions_per_epoch')
        os.makedirs(self.recon_dir, exist_ok=True)

        # Select random samples for consistent visualization
        indices = np.random.choice(len(self.x_val), size=self.n_samples, replace=False)
        self.sample_images = self.x_val[indices]
        self.sample_labels = self.y_val[indices]
        self.true_classes = np.argmax(self.sample_labels, axis=1)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Generate reconstructions at the end of each epoch."""
        if (epoch + 1) % self.frequency == 0:
            try:
                logger.info(f"Generating reconstructions for epoch {epoch + 1}...")

                # Generate predictions and reconstructions
                predictions = self.model.predict(self.sample_images, verbose=0)

                # Debug: print prediction structure
                if epoch == 0:  # Only log on first epoch
                    if isinstance(predictions, dict):
                        logger.info(f"Prediction keys: {list(predictions.keys())}")
                        for key, value in predictions.items():
                            logger.info(f"  {key}: shape {value.shape}")
                    elif isinstance(predictions, (list, tuple)):
                        logger.info(f"Prediction is list/tuple with {len(predictions)} elements")
                        for i, pred in enumerate(predictions):
                            logger.info(f"  Element {i}: shape {pred.shape}")
                    else:
                        logger.info(f"Prediction shape: {predictions.shape}")

                # Extract predictions and reconstructions
                if isinstance(predictions, dict):
                    # Dictionary format - look for common keys
                    length_key = None
                    recon_key = None

                    for key in predictions.keys():
                        if 'length' in key.lower() or 'capsule' in key.lower():
                            length_key = key
                        elif 'recon' in key.lower() or 'decoder' in key.lower():
                            recon_key = key

                    if length_key:
                        pred_classes = np.argmax(predictions[length_key], axis=1)
                    else:
                        # Fallback - use first key that looks like predictions
                        first_key = list(predictions.keys())[0]
                        pred_classes = np.argmax(predictions[first_key], axis=1)

                    reconstructions = predictions.get(recon_key) if recon_key else None

                elif isinstance(predictions, (list, tuple)):
                    # List/tuple format
                    pred_classes = np.argmax(predictions[0], axis=1)
                    reconstructions = predictions[1] if len(predictions) > 1 else None
                else:
                    # Single array format
                    pred_classes = np.argmax(predictions, axis=1)
                    reconstructions = None

                if reconstructions is not None:
                    # Save reconstruction plot
                    save_path = os.path.join(self.recon_dir, f'epoch_{epoch + 1:03d}.png')
                    plot_reconstruction_comparison(
                        self.sample_images,
                        reconstructions,
                        self.true_classes,
                        pred_classes,
                        save_path,
                        epoch=epoch + 1,
                        dataset=self.dataset
                    )

                    logger.info(f"Reconstruction visualization saved: {save_path}")
                else:
                    logger.warning(f"No reconstructions found in model output for epoch {epoch + 1}")

            except Exception as e:
                logger.error(f"Failed to generate reconstructions for epoch {epoch + 1}: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")


def create_callbacks(
        model_name: str,
        validation_data: Tuple[np.ndarray, np.ndarray],
        dataset: str,
        monitor: str = 'val_loss',
        patience: int = 10,
        save_best_only: bool = True,
        viz_frequency: int = 1
) -> Tuple[list, str]:
    """Create training callbacks including visualization callback.

    Args:
        model_name: Name for saved model files.
        validation_data: Validation data for reconstruction visualization.
        dataset: Dataset name.
        monitor: Metric to monitor for callbacks.
        patience: Patience for early stopping.
        save_best_only: Whether to save only the best model.
        viz_frequency: How often to generate visualizations (every N epochs).

    Returns:
        Tuple of (callbacks list, results directory path).
    """
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/capsnet_{model_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    callbacks = [
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            mode='min' if 'loss' in monitor else 'max',  # For loss, lower is better; for accuracy, higher is better
            verbose=1
        ),

        # Model checkpointing
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(results_dir, 'best_model.keras'),
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=False,
            mode='min' if 'loss' in monitor else 'max',  # For loss, lower is better; for accuracy, higher is better
            verbose=1
        ),

        # Learning rate reduction
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),

        # Learning rate logging
        LearningRateLogger(),

        # CSV logging
        keras.callbacks.CSVLogger(
            filename=os.path.join(results_dir, 'training_log.csv'),
            append=False
        ),

        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(results_dir, 'tensorboard'),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),

        # Custom reconstruction visualization callback
        ReconstructionVisualizationCallback(
            validation_data=validation_data,
            save_dir=results_dir,
            dataset=dataset,
            n_samples=8,
            frequency=viz_frequency
        )
    ]

    logger.info(f"Results will be saved to: {results_dir}")
    return callbacks, results_dir


def generate_final_visualizations(
        model: keras.Model,
        test_data: Tuple[np.ndarray, np.ndarray],
        history: keras.callbacks.History,
        results_dir: str,
        dataset: str,
        test_results_dict: Optional[Dict[str, float]] = None,
        n_samples: int = 16
) -> None:
    """Generate comprehensive final visualizations.

    Args:
        model: Trained CapsNet model.
        test_data: Test dataset.
        history: Training history.
        results_dir: Directory to save visualizations.
        dataset: Dataset name.
        test_results_dict: Pre-computed test results dictionary.
        n_samples: Number of samples for final reconstructions.
    """
    x_test, y_test = test_data

    logger.info("Generating final visualizations...")

    # Create final visualizations directory
    final_viz_dir = os.path.join(results_dir, 'final_visualizations')
    os.makedirs(final_viz_dir, exist_ok=True)

    # Evaluate model and get test results if not provided
    if test_results_dict is None:
        test_results = model.evaluate(x_test, y_test, verbose=0)
        if isinstance(test_results, (list, tuple)):
            test_results_dict = dict(zip(model.metrics_names, test_results))
        elif isinstance(test_results, dict):
            test_results_dict = test_results
        else:
            test_results_dict = {"test_loss": test_results}

    # Plot training history
    plot_training_history(history, final_viz_dir)

    # Plot metrics summary
    plot_metrics_summary(history, test_results_dict, final_viz_dir)

    # Generate final reconstructions
    sample_indices = np.random.choice(len(x_test), size=n_samples, replace=False)
    sample_images = x_test[sample_indices]
    sample_labels = y_test[sample_indices]
    true_classes = np.argmax(sample_labels, axis=1)

    # Get predictions and reconstructions
    predictions = model.predict(sample_images, verbose=0)

    # Handle different prediction formats
    if isinstance(predictions, dict):
        # Find the length/capsule key for predictions
        length_key = None
        recon_key = None

        for key in predictions.keys():
            if 'length' in key.lower() or 'capsule' in key.lower():
                length_key = key
            elif 'recon' in key.lower() or 'decoder' in key.lower():
                recon_key = key

        if length_key:
            pred_classes = np.argmax(predictions[length_key], axis=1)
        else:
            # Fallback to first key
            first_key = list(predictions.keys())[0]
            pred_classes = np.argmax(predictions[first_key], axis=1)

        reconstructions = predictions.get(recon_key) if recon_key else None

    elif isinstance(predictions, (list, tuple)):
        pred_classes = np.argmax(predictions[0], axis=1)
        reconstructions = predictions[1] if len(predictions) > 1 else None
    else:
        pred_classes = np.argmax(predictions, axis=1)
        reconstructions = None

    if reconstructions is not None:
        # Large final reconstruction comparison
        save_path = os.path.join(final_viz_dir, 'final_reconstructions.png')
        plot_reconstruction_comparison(
            sample_images,
            reconstructions,
            true_classes,
            pred_classes,
            save_path,
            dataset=dataset
        )

        # Save reconstruction statistics
        recon_error = np.mean(np.square(sample_images - reconstructions))
        with open(os.path.join(final_viz_dir, 'reconstruction_stats.txt'), 'w') as f:
            f.write(f"Reconstruction Statistics\n")
            f.write(f"========================\n")
            f.write(f"Mean Squared Error: {recon_error:.6f}\n")
            f.write(f"Samples analyzed: {n_samples}\n")
            f.write(f"Correct predictions: {np.sum(true_classes == pred_classes)}/{n_samples}\n")
            f.write(f"Accuracy on samples: {np.mean(true_classes == pred_classes):.4f}\n")

    # Save training summary
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"CapsNet Training Summary\n")
        f.write(f"=======================\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Total epochs: {len(history.history['loss'])}\n")
        f.write(f"Final training loss: {history.history['loss'][-1]:.6f}\n")
        f.write(f"Final validation loss: {history.history['val_loss'][-1]:.6f}\n")

        if 'capsule_accuracy' in history.history:
            f.write(f"Final training accuracy: {history.history['capsule_accuracy'][-1]:.4f}\n")
            if 'val_capsule_accuracy' in history.history:
                f.write(f"Final validation accuracy: {history.history['val_capsule_accuracy'][-1]:.4f}\n")
            else:
                f.write(f"Final validation accuracy: Not available\n")

        f.write(f"\nTest Results:\n")
        for metric_name, value in test_results_dict.items():
            f.write(f"{metric_name}: {value:.6f}\n")

    logger.info(f"Final visualizations saved to: {final_viz_dir}")


def train_model(args: argparse.Namespace) -> None:
    """Main training function.

    Args:
        args: Command line arguments.
    """
    logger.info("Starting CapsNet training with visualizations")
    logger.info(f"Arguments: {vars(args)}")

    # Setup GPU
    setup_gpu()

    # Load data
    if args.dataset.lower() == 'mnist':
        train_data, val_data, test_data = load_mnist_data(args.validation_split)
    elif args.dataset.lower() == 'cifar10':
        train_data, val_data, test_data = load_cifar10_data(args.validation_split)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    x_train, y_train = train_data
    x_val, y_val = val_data
    x_test, y_test = test_data

    # Create model
    logger.info("Creating CapsNet model...")
    model_config = create_model_config(args.dataset)

    model = create_capsnet(
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        **model_config
    )

    # Build model by calling it once
    sample_input = tf.zeros((1,) + model_config['input_shape'])
    _ = model(sample_input, training=False)

    # Print model summary
    logger.info("Model architecture:")
    model.summary()

    # Create callbacks with visualization
    callbacks, results_dir = create_callbacks(
        model_name=args.dataset,
        validation_data=(x_val, y_val),
        dataset=args.dataset,
        patience=args.patience,
        save_best_only=True,
        viz_frequency=args.viz_frequency
    )

    # Train model
    logger.info("Starting training...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = model.evaluate(x_test, y_test, verbose=1)

    # Print final results
    logger.info("Training completed!")
    logger.info("Final test results:")

    # Handle different result formats
    if isinstance(test_results, (list, tuple)):
        for metric_name, value in zip(model.metrics_names, test_results):
            if isinstance(value, (int, float, np.number)):
                logger.info(f"  {metric_name}: {value:.4f}")
            else:
                logger.info(f"  {metric_name}: {value}")
        test_results_dict = dict(zip(model.metrics_names, test_results))
    elif isinstance(test_results, dict):
        for metric_name, value in test_results.items():
            if isinstance(value, (int, float, np.number)):
                logger.info(f"  {metric_name}: {value:.4f}")
            else:
                logger.info(f"  {metric_name}: {value}")
        test_results_dict = test_results
    else:
        if isinstance(test_results, (int, float, np.number)):
            logger.info(f"  test_loss: {test_results:.4f}")
        else:
            logger.info(f"  test_loss: {test_results}")
        test_results_dict = {"test_loss": test_results}

    # Clean test_results_dict to ensure all values are numeric
    cleaned_test_results = {}
    for key, value in test_results_dict.items():
        if isinstance(value, (int, float, np.number)):
            cleaned_test_results[key] = float(value)
        elif isinstance(value, dict):
            # If value is a dict, try to extract a meaningful metric
            if 'loss' in value:
                cleaned_test_results[key + '_loss'] = float(value['loss'])
            elif len(value) == 1:
                # Single item dict, use its value
                sub_key, sub_value = next(iter(value.items()))
                if isinstance(sub_value, (int, float, np.number)):
                    cleaned_test_results[key] = float(sub_value)
        else:
            # Try to convert to float, fallback to 0
            try:
                cleaned_test_results[key] = float(value)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert {key}={value} to float, using 0")
                cleaned_test_results[key] = 0.0

    test_results_dict = cleaned_test_results

    # Generate comprehensive final visualizations
    generate_final_visualizations(
        model=model,
        test_data=(x_test, y_test),
        history=history,
        results_dir=results_dir,
        dataset=args.dataset,
        test_results_dict=test_results_dict,
        n_samples=16
    )

    # Save final model
    final_model_path = os.path.join(results_dir, f"capsnet_{args.dataset}_final.keras")
    model.save(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")

    # Generate some predictions for demonstration
    logger.info("Generating sample predictions...")
    sample_indices = np.random.choice(len(x_test), size=5, replace=False)
    sample_images = x_test[sample_indices]
    sample_labels = y_test[sample_indices]

    predictions = model.predict(sample_images, verbose=0)

    # Handle different prediction formats
    if isinstance(predictions, dict):
        # Find the length/capsule key
        length_key = None
        for key in predictions.keys():
            if 'length' in key.lower() or 'capsule' in key.lower():
                length_key = key
                break

        if length_key:
            predicted_classes = np.argmax(predictions[length_key], axis=1)
            confidences = predictions[length_key].max(axis=1)
        else:
            # Fallback to first key
            first_key = list(predictions.keys())[0]
            predicted_classes = np.argmax(predictions[first_key], axis=1)
            confidences = predictions[first_key].max(axis=1)

    elif isinstance(predictions, (list, tuple)):
        predicted_classes = np.argmax(predictions[0], axis=1)
        confidences = predictions[0].max(axis=1)
    else:
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = predictions.max(axis=1)

    true_classes = np.argmax(sample_labels, axis=1)

    logger.info("Sample predictions:")
    for i in range(5):
        logger.info(
            f"  Sample {i}: True={true_classes[i]}, Predicted={predicted_classes[i]}, Confidence={confidences[i]:.4f}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Train CapsNet on image classification datasets with visualizations')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                        help='Dataset to use for training (default: mnist)')
    parser.add_argument('--validation-split', type=float, default=0.1,
                        help='Fraction of training data to use for validation (default: 0.1)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size (default: 32)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use (default: adam)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience (default: 30)')

    # Model arguments
    parser.add_argument('--no-reconstruction', action='store_true',
                        help='Disable reconstruction decoder')
    parser.add_argument('--reconstruction-weight', type=float, default=0.0005,
                        help='Weight for reconstruction loss (default: 0.0005)')

    # Visualization arguments
    parser.add_argument('--viz-frequency', type=int, default=1,
                        help='How often to generate reconstruction visualizations (every N epochs, default: 1)')

    # Parse arguments
    args = parser.parse_args()

    # Start training
    try:
        train_model(args)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main()