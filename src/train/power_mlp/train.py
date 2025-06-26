#!/usr/bin/env python3
"""
Enhanced training script for PowerMLP with comprehensive visualizations.

This script demonstrates how to train the PowerMLP model using standard Keras
workflows with model.compile() and model.fit(). The script handles data loading,
model creation, training, evaluation, and generates comprehensive visualizations
including training plots and performance analysis.

PowerMLP is an efficient alternative to Kolmogorov-Arnold Networks (KAN) offering:
- ~40x faster training time
- ~10x fewer FLOPs
- Equal or better performance
- Simpler implementation

Usage:
    # Basic stable training:
    python train_powermlp.py --dataset mnist --epochs 50 --architecture small --k 2

    # Advanced training with regularization:
    python train_powermlp.py --dataset mnist --k 2 --dropout-rate 0.2 --batch-normalization

    # Conservative training for stability:
    python train_powermlp.py --dataset mnist --learning-rate 0.0001 --k 1 --architecture small
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
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
from sklearn.metrics import classification_report, confusion_matrix

from dl_techniques.models.power_mlp import PowerMLP, create_power_mlp, create_power_mlp_binary_classifier, \
    create_power_mlp_regressor
from dl_techniques.utils.logger import logger

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LearningRateLogger(keras.callbacks.Callback):
    """Custom callback to log learning rate to history."""

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        # Get learning rate from optimizer
        lr = float(self.model.optimizer.learning_rate)
        logs['lr'] = lr

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


def load_mnist_data(
        validation_split: float = 0.1,
        normalize: bool = True,
        flatten: bool = True
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess MNIST dataset for PowerMLP.

    Args:
        validation_split: Fraction of training data to use for validation.
        normalize: Whether to normalize pixel values to [0, 1].
        flatten: Whether to flatten images to 1D vectors.

    Returns:
        Tuple of (train_data, val_data, test_data) where each is (x, y).
    """
    logger.info("Loading MNIST dataset...")

    # Load raw data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Flatten if requested (PowerMLP expects 1D input)
    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    else:
        # Add channel dimension for consistency
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
        normalize: bool = True,
        flatten: bool = True
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess CIFAR-10 dataset for PowerMLP.

    Args:
        validation_split: Fraction of training data to use for validation.
        normalize: Whether to normalize pixel values to [0, 1].
        flatten: Whether to flatten images to 1D vectors.

    Returns:
        Tuple of (train_data, val_data, test_data) where each is (x, y).
    """
    logger.info("Loading CIFAR-10 dataset...")

    # Load raw data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Flatten if requested (PowerMLP expects 1D input)
    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

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


def create_model_config(
        dataset: str,
        architecture: str = "default",
        k: int = 2,  # Lower default k to prevent exploding gradients
        dropout_rate: float = 0.0,
        batch_normalization: bool = False
) -> Dict[str, Any]:
    """Create model configuration based on dataset and architecture.

    Args:
        dataset: Name of dataset ('mnist' or 'cifar10').
        architecture: Architecture type ('default', 'small', 'large', 'deep').
        k: Power for ReLU-k activation.
        dropout_rate: Dropout rate for regularization.
        batch_normalization: Whether to use batch normalization.

    Returns:
        Dictionary of model configuration parameters.
    """
    base_config = {
        'k': k,
        'dropout_rate': dropout_rate,
        'batch_normalization': batch_normalization,
        'output_activation': 'softmax',
        'kernel_initializer': 'glorot_normal',  # Better initialization for stability
        'bias_initializer': 'zeros',
        'kernel_regularizer': keras.regularizers.L2(1e-5) if dropout_rate == 0.0 else None,
        # Light regularization if no dropout
    }

    # Architecture configurations - more conservative sizes
    architectures = {
        'small': {
            'mnist': [128, 64, 10],
            'cifar10': [256, 128, 10]
        },
        'default': {
            'mnist': [256, 128, 64, 10],
            'cifar10': [512, 256, 128, 10]
        },
        'large': {
            'mnist': [512, 256, 128, 64, 10],
            'cifar10': [1024, 512, 256, 128, 10]
        },
        'deep': {
            'mnist': [256, 128, 128, 64, 64, 32, 10],
            'cifar10': [512, 256, 256, 128, 128, 64, 10]
        }
    }

    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}")

    if dataset.lower() not in architectures[architecture]:
        raise ValueError(f"Unsupported dataset: {dataset}")

    base_config['hidden_units'] = architectures[architecture][dataset.lower()]

    return base_config


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
    if 'accuracy' in history_dict:
        axes[0, 1].plot(epochs, history_dict['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history_dict:
            axes[0, 1].plot(epochs, history_dict['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Accuracy Metrics\nNot Available',
                        transform=axes[0, 1].transAxes, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')

    # Plot learning rate if available
    if 'lr' in history_dict:
        axes[1, 0].plot(epochs, history_dict['lr'], 'orange', linewidth=2)
        axes[1, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        # Show loss difference plot
        if 'val_loss' in history_dict:
            loss_diff = np.array(history_dict['val_loss']) - np.array(history_dict['loss'])
            axes[1, 0].plot(epochs, loss_diff, 'red', linewidth=2)
            axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].set_title('Validation - Training Loss', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss Difference')
            axes[1, 0].grid(True, alpha=0.3)

    # Plot training stability metrics
    if len(epochs) > 5:
        # Calculate rolling averages for stability
        window = max(3, len(epochs) // 10)
        loss_rolling = pd.Series(history_dict['loss']).rolling(window=window, center=True).mean()
        val_loss_rolling = pd.Series(history_dict['val_loss']).rolling(window=window, center=True).mean()

        axes[1, 1].plot(epochs, history_dict['loss'], alpha=0.3, color='blue', label='Raw Train Loss')
        axes[1, 1].plot(epochs, history_dict['val_loss'], alpha=0.3, color='red', label='Raw Val Loss')
        axes[1, 1].plot(epochs, loss_rolling, linewidth=2, color='blue', label=f'Train Loss (MA-{window})')
        axes[1, 1].plot(epochs, val_loss_rolling, linewidth=2, color='red', label=f'Val Loss (MA-{window})')

        axes[1, 1].set_title('Training Stability (Moving Average)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Show training summary for short runs
        final_epochs = min(5, len(epochs))
        final_train_loss = np.mean(history_dict['loss'][-final_epochs:])
        final_val_loss = np.mean(history_dict['val_loss'][-final_epochs:])

        summary_text = f"Final {final_epochs} epochs average:\n"
        summary_text += f"Train Loss: {final_train_loss:.4f}\n"
        summary_text += f"Val Loss: {final_val_loss:.4f}\n"
        summary_text += f"Overfitting: {final_val_loss / final_train_loss:.2f}x"

        axes[1, 1].text(0.5, 0.5, summary_text,
                        transform=axes[1, 1].transAxes, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                        fontsize=12)
        axes[1, 1].set_title('Training Summary', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_summary(
        history: keras.callbacks.History,
        test_results: Dict[str, float],
        save_dir: str,
        model_params: Dict[str, Any]
) -> None:
    """Plot a comprehensive summary of all metrics.

    Args:
        history: Keras training history object.
        test_results: Dictionary of test results.
        save_dir: Directory to save plots.
        model_params: Model parameters for display.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Final metrics comparison
    available_metrics = ['Loss', 'Accuracy']
    train_values = [
        history.history['loss'][-1],
        history.history.get('accuracy', [0])[-1]
    ]
    val_values = [
        history.history['val_loss'][-1],
        history.history.get('val_accuracy', [0])[-1]
    ]
    test_values = [
        test_results.get('loss', 0),
        test_results.get('accuracy', 0)
    ]

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
                                    xytext=(0, 3),
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

    if 'accuracy' in history.history:
        ax2 = axes[0, 1].twinx()
        ax2.plot(epochs, history.history['accuracy'], '--', label='Train Acc', linewidth=2, color='lightblue')
        if 'val_accuracy' in history.history:
            ax2.plot(epochs, history.history['val_accuracy'], '--', label='Val Acc', linewidth=2, color='lightcoral')
        ax2.set_ylabel('Accuracy')
        ax2.legend(loc='center right')

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Progress', fontweight='bold')
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Model architecture visualization
    hidden_units = model_params.get('hidden_units', [])
    if hidden_units:
        layer_names = [f'Layer {i + 1}' for i in range(len(hidden_units))]
        layer_sizes = hidden_units

        bars = axes[1, 0].bar(layer_names, layer_sizes, alpha=0.7, color='skyblue')
        axes[1, 0].set_title('Model Architecture', fontweight='bold')
        axes[1, 0].set_xlabel('Layers')
        axes[1, 0].set_ylabel('Number of Units')
        axes[1, 0].set_yscale('log')

        # Add value labels on bars
        for bar, size in zip(bars, layer_sizes):
            height = bar.get_height()
            axes[1, 0].annotate(f'{size}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)

        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Architecture\nNot Available',
                        transform=axes[1, 0].transAxes, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        axes[1, 0].set_title('Model Architecture', fontweight='bold')

    # Plot 4: Model parameters summary
    param_text = "PowerMLP Parameters:\n"
    param_text += f"Architecture: {model_params.get('hidden_units', 'N/A')}\n"
    param_text += f"ReLU-k power: {model_params.get('k', 'N/A')}\n"
    param_text += f"Dropout rate: {model_params.get('dropout_rate', 'N/A')}\n"
    param_text += f"Batch norm: {model_params.get('batch_normalization', 'N/A')}\n"
    param_text += f"Total params: {test_results.get('total_params', 'N/A'):,}\n"
    param_text += f"Trainable params: {test_results.get('trainable_params', 'N/A'):,}\n"
    param_text += f"\nTraining Results:\n"
    param_text += f"Best val accuracy: {max(history.history.get('val_accuracy', [0])):.4f}\n"
    param_text += f"Final test accuracy: {test_results.get('accuracy', 0):.4f}\n"
    param_text += f"Training epochs: {len(history.history['loss'])}\n"

    axes[1, 1].text(0.05, 0.95, param_text,
                    transform=axes[1, 1].transAxes, ha='left', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    fontsize=10, fontfamily='monospace')
    axes[1, 1].set_title('Model & Training Summary', fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_dir: str,
        class_names: Optional[List[str]] = None
) -> None:
    """Plot confusion matrix for classification results.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        save_dir: Directory to save plots.
        class_names: List of class names for labeling.
    """
    # Convert one-hot to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot raw confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0])
    axes[0].set_title('Confusion Matrix (Raw Counts)', fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # Plot normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1])
    axes[1].set_title('Confusion Matrix (Normalized)', fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_stable_optimizer(optimizer_name: str, learning_rate: float) -> keras.optimizers.Optimizer:
    """Create an optimizer with gradient clipping for stable training.

    Args:
        optimizer_name: Name of the optimizer.
        learning_rate: Learning rate.

    Returns:
        Configured optimizer with gradient clipping.
    """
    if optimizer_name.lower() == 'adam':
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,  # Gradient clipping
            epsilon=1e-7  # Smaller epsilon for numerical stability
        )
    elif optimizer_name.lower() == 'adamw':
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            clipnorm=1.0,
            weight_decay=0.01,
            epsilon=1e-7
        )
    elif optimizer_name.lower() == 'sgd':
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            clipnorm=1.0
        )
    else:
        # Fallback to basic optimizer
        optimizer = keras.optimizers.get(optimizer_name)
        optimizer.learning_rate = learning_rate
        if hasattr(optimizer, 'clipnorm'):
            optimizer.clipnorm = 1.0

    return optimizer
    """Custom callback to log learning rate to history."""

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        # Get learning rate from optimizer
        lr = float(self.model.optimizer.learning_rate)
        logs['lr'] = lr


class NaNStoppingCallback(keras.callbacks.Callback):
    """Custom callback to stop training when NaN values are detected."""

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # Check for NaN in any metric
        for key, value in logs.items():
            if np.isnan(value) or np.isinf(value):
                logger.error(f"NaN or Inf detected in {key}: {value}. Stopping training.")
                self.model.stop_training = True
                return


def create_callbacks(
        model_name: str,
        monitor: str = 'val_loss',
        patience: int = 15,
        save_best_only: bool = True,
        reduce_lr_patience: int = 7
) -> Tuple[List[keras.callbacks.Callback], str]:
    """Create training callbacks.

    Args:
        model_name: Name for saved model files.
        monitor: Metric to monitor for callbacks.
        patience: Patience for early stopping.
        save_best_only: Whether to save only the best model.
        reduce_lr_patience: Patience for learning rate reduction.

    Returns:
        Tuple of (callbacks list, results directory path).
    """
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/powermlp_{model_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    callbacks = [
        # NaN detection callback (should be first)
        NaNStoppingCallback(),

        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            mode='min' if 'loss' in monitor else 'max',
            verbose=1
        ),

        # Model checkpointing
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(results_dir, 'best_model.keras'),
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=False,
            mode='min' if 'loss' in monitor else 'max',
            verbose=1
        ),

        # Learning rate reduction
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
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
        )
    ]

    logger.info(f"Results will be saved to: {results_dir}")
    return callbacks, results_dir


def generate_final_visualizations(
        model: keras.Model,
        test_data: Tuple[np.ndarray, np.ndarray],
        history: keras.callbacks.History,
        results_dir: str,
        model_params: Dict[str, Any],
        dataset: str
) -> None:
    """Generate comprehensive final visualizations.

    Args:
        model: Trained PowerMLP model.
        test_data: Test dataset.
        history: Training history.
        results_dir: Directory to save visualizations.
        model_params: Model parameters.
        dataset: Dataset name.
    """
    x_test, y_test = test_data

    logger.info("Generating final visualizations...")

    # Create final visualizations directory
    final_viz_dir = os.path.join(results_dir, 'final_visualizations')
    os.makedirs(final_viz_dir, exist_ok=True)

    # Evaluate model and get test results
    test_results = model.evaluate(x_test, y_test, verbose=0)
    if isinstance(test_results, (list, tuple)):
        # Handle case where metrics_names might not be available or might be different
        if hasattr(model, 'metrics_names') and model.metrics_names and len(model.metrics_names) == len(test_results):
            test_results_dict = dict(zip(model.metrics_names, test_results))
        else:
            # Fallback naming scheme
            test_results_dict = {"loss": test_results[0]}
            if len(test_results) > 1:
                test_results_dict["accuracy"] = test_results[1]
            for i, value in enumerate(test_results[2:], 2):
                test_results_dict[f"metric_{i}"] = value
    else:
        test_results_dict = {"test_loss": test_results}

    # Add model parameter information
    test_results_dict['total_params'] = model.count_params()
    # Calculate trainable parameters correctly for Keras 3
    trainable_params = 0
    for weight in model.trainable_weights:
        if hasattr(weight, 'numpy'):
            # For Keras 3 - weight is a tensor
            trainable_params += np.prod(weight.shape)
        else:
            # Fallback for other cases
            trainable_params += np.prod(weight.get_shape().as_list())
    test_results_dict['trainable_params'] = trainable_params

    # Plot training history
    plot_training_history(history, final_viz_dir)

    # Plot metrics summary
    plot_metrics_summary(history, test_results_dict, final_viz_dir, model_params)

    # Generate predictions for confusion matrix
    try:
        predictions = model.predict(x_test, verbose=0)

        # Create class names
        if dataset.lower() == 'mnist':
            class_names = [str(i) for i in range(10)]
        elif dataset.lower() == 'cifar10':
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        else:
            class_names = None

        # Plot confusion matrix
        plot_confusion_matrix(y_test, predictions, final_viz_dir, class_names)

        # Generate classification report
        y_true_classes = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
        y_pred_classes = np.argmax(predictions, axis=1) if len(predictions.shape) > 1 else predictions

        # Save detailed classification report
        report = classification_report(y_true_classes, y_pred_classes,
                                       target_names=class_names, output_dict=True)

        # Save detailed classification report as JSON
        import json
        with open(os.path.join(final_viz_dir, 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

    except Exception as e:
        logger.warning(f"Failed to generate confusion matrix and classification report: {e}")
        report = {}

    # Save training summary
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"PowerMLP Training Summary\n")
        f.write(f"========================\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Architecture: {model_params.get('hidden_units', 'N/A')}\n")
        f.write(f"ReLU-k power: {model_params.get('k', 'N/A')}\n")
        f.write(f"Dropout rate: {model_params.get('dropout_rate', 'N/A')}\n")
        f.write(f"Batch normalization: {model_params.get('batch_normalization', 'N/A')}\n")
        f.write(f"Total epochs: {len(history.history['loss'])}\n")
        f.write(f"Total parameters: {test_results_dict['total_params']:,}\n")
        f.write(f"Trainable parameters: {test_results_dict['trainable_params']:,}\n")
        f.write(f"\nFinal Results:\n")
        f.write(f"Final training loss: {history.history['loss'][-1]:.6f}\n")
        f.write(f"Final validation loss: {history.history['val_loss'][-1]:.6f}\n")

        if 'accuracy' in history.history:
            f.write(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}\n")
            if 'val_accuracy' in history.history:
                f.write(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}\n")

        f.write(f"\nTest Results:\n")
        for metric_name, value in test_results_dict.items():
            if isinstance(value, (int, float, np.number)):
                f.write(f"{metric_name}: {value:.6f}\n")
            else:
                f.write(f"{metric_name}: {value}\n")

        f.write(f"\nPer-class Performance:\n")
        if report and isinstance(report, dict):
            for class_name, metrics in report.items():
                if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    f.write(f"{class_name}: precision={metrics.get('precision', 0):.3f}, "
                            f"recall={metrics.get('recall', 0):.3f}, f1-score={metrics.get('f1-score', 0):.3f}\n")
        else:
            f.write("Per-class performance not available due to errors in report generation.\n")

    logger.info(f"Final visualizations saved to: {final_viz_dir}")


def train_model(args: argparse.Namespace) -> None:
    """Main training function.

    Args:
        args: Command line arguments.
    """
    logger.info("Starting PowerMLP training with visualizations")
    logger.info(f"Arguments: {vars(args)}")

    # Setup GPU
    setup_gpu()

    # Load data
    if args.dataset.lower() == 'mnist':
        train_data, val_data, test_data = load_mnist_data(args.validation_split, flatten=True)
    elif args.dataset.lower() == 'cifar10':
        train_data, val_data, test_data = load_cifar10_data(args.validation_split, flatten=True)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    x_train, y_train = train_data
    x_val, y_val = val_data
    x_test, y_test = test_data

    # Create model configuration
    model_config = create_model_config(
        dataset=args.dataset,
        architecture=args.architecture,
        k=args.k,
        dropout_rate=args.dropout_rate,
        batch_normalization=args.batch_normalization
    )

    logger.info(f"Model configuration: {model_config}")

    # Create model with stable optimizer
    logger.info("Creating PowerMLP model...")
    stable_optimizer = create_stable_optimizer(args.optimizer, args.learning_rate)

    model = PowerMLP(**model_config)
    model.compile(
        optimizer=stable_optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.info(f"Using optimizer: {type(stable_optimizer).__name__} with clipnorm=1.0")

    # Build model by calling it once
    sample_input = tf.zeros((1, x_train.shape[1]))
    _ = model(sample_input, training=False)

    # Print model summary
    logger.info("Model architecture:")
    model.summary()

    # Quick sanity check
    logger.info("Performing quick sanity check...")
    try:
        sample_batch = x_train[:32]
        sample_labels = y_train[:32]
        test_loss = model.evaluate(sample_batch, sample_labels, verbose=0)
        logger.info(f"Initial model sanity check - loss: {test_loss[0]:.4f}, accuracy: {test_loss[1]:.4f}")

        if np.isnan(test_loss[0]) or np.isinf(test_loss[0]):
            logger.error("Model produces NaN/Inf values before training! Check model configuration.")
            logger.error("Try reducing learning rate, k value, or enabling batch normalization.")
            return

    except Exception as e:
        logger.warning(f"Sanity check failed: {e}. Proceeding with training...")

    logger.info(f"Model configuration summary:")
    logger.info(f"  - Architecture: {model_config['hidden_units']}")
    logger.info(f"  - ReLU-k power: {model_config['k']}")
    logger.info(f"  - Dropout rate: {model_config['dropout_rate']}")
    logger.info(f"  - Batch normalization: {model_config['batch_normalization']}")
    logger.info(f"  - Learning rate: {args.learning_rate}")
    logger.info(f"  - Total parameters: {model.count_params():,}")

    # Create callbacks
    callbacks, results_dir = create_callbacks(
        model_name=f"{args.dataset}_{args.architecture}",
        patience=args.patience,
        save_best_only=True,
        reduce_lr_patience=args.patience // 2
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

    if isinstance(test_results, (list, tuple)):
        # Handle case where metrics_names might not be available
        if hasattr(model, 'metrics_names') and model.metrics_names:
            for metric_name, value in zip(model.metrics_names, test_results):
                logger.info(f"  {metric_name}: {value:.4f}")
        else:
            # Fallback naming
            logger.info(f"  loss: {test_results[0]:.4f}")
            if len(test_results) > 1:
                logger.info(f"  accuracy: {test_results[1]:.4f}")
            for i, value in enumerate(test_results[2:], 2):
                logger.info(f"  metric_{i}: {value:.4f}")
    else:
        logger.info(f"  test_loss: {test_results:.4f}")

    # Generate comprehensive final visualizations
    generate_final_visualizations(
        model=model,
        test_data=(x_test, y_test),
        history=history,
        results_dir=results_dir,
        model_params=model_config,
        dataset=args.dataset
    )

    # Save final model
    final_model_path = os.path.join(results_dir, f"powermlp_{args.dataset}_final.keras")
    model.save(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")

    # Generate some predictions for demonstration
    logger.info("Generating sample predictions...")
    sample_indices = np.random.choice(len(x_test), size=10, replace=False)
    sample_images = x_test[sample_indices]
    sample_labels = y_test[sample_indices]

    predictions = model.predict(sample_images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(sample_labels, axis=1)
    confidences = np.max(predictions, axis=1)

    logger.info("Sample predictions:")
    for i in range(10):
        logger.info(
            f"  Sample {i}: True={true_classes[i]}, Predicted={predicted_classes[i]}, "
            f"Confidence={confidences[i]:.4f}"
        )


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Train PowerMLP on image classification datasets with visualizations'
    )

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                        help='Dataset to use for training (default: mnist)')
    parser.add_argument('--validation-split', type=float, default=0.1,
                        help='Fraction of training data to use for validation (default: 0.1)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Training batch size (default: 128)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use (default: adam)')
    parser.add_argument('--learning-rate', type=float, default=0.0003,
                        help='Learning rate (default: 0.0003)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')

    # Model arguments
    parser.add_argument('--architecture', type=str, default='default',
                        choices=['small', 'default', 'large', 'deep'],
                        help='Model architecture size (default: default)')
    parser.add_argument('--k', type=int, default=2,
                        help='Power for ReLU-k activation (default: 2)')
    parser.add_argument('--dropout-rate', type=float, default=0.1,
                        help='Dropout rate for regularization (default: 0.1)')
    parser.add_argument('--batch-normalization', action='store_true',
                        help='Enable batch normalization')

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