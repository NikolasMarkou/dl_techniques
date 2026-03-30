"""
CapsNet training with comprehensive visualizations.

Trains CapsNet on MNIST or CIFAR-10 with per-epoch reconstruction
visualizations, training curves, and metrics summaries.
"""

import os
import keras
import matplotlib
import numpy as np
import tensorflow as tf
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional

from dl_techniques.utils.logger import logger
from dl_techniques.models.capsnet import create_capsnet
from train.common import setup_gpu, create_base_argument_parser, create_callbacks

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# ---------------------------------------------------------------------
# Data loading (CapsNet-specific: uses one-hot labels)
# ---------------------------------------------------------------------

def load_mnist_data(
        validation_split: float = 0.1,
        normalize: bool = True
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess MNIST dataset with one-hot labels for CapsNet."""
    logger.info("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    if normalize:
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if validation_split > 0:
        val_size = int(len(x_train) * validation_split)
        x_val, y_val = x_train[:val_size], y_train[:val_size]
        x_train, y_train = x_train[val_size:], y_train[val_size:]
    else:
        x_val, y_val = x_test, y_test

    logger.info(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def load_cifar10_data(
        validation_split: float = 0.1,
        normalize: bool = True
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess CIFAR-10 dataset with one-hot labels for CapsNet."""
    logger.info("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    if normalize:
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if validation_split > 0:
        val_size = int(len(x_train) * validation_split)
        x_val, y_val = x_train[:val_size], y_train[:val_size]
        x_train, y_train = x_train[val_size:], y_train[val_size:]
    else:
        x_val, y_val = x_test, y_test

    logger.info(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------

def create_model_config(dataset: str) -> Dict[str, Any]:
    """Create CapsNet configuration based on dataset."""
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
            'conv_filters': [128, 256, 256],
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


# ---------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------

def _extract_predictions(predictions) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract predicted classes and reconstructions from model output.

    Returns:
        Tuple of (predicted_classes, reconstructions_or_None).
    """
    if isinstance(predictions, dict):
        length_key = None
        recon_key = None
        for key in predictions.keys():
            if 'length' in key.lower() or 'capsule' in key.lower():
                length_key = key
            elif 'recon' in key.lower() or 'decoder' in key.lower():
                recon_key = key
        pred_key = length_key or list(predictions.keys())[0]
        pred_classes = np.argmax(predictions[pred_key], axis=1)
        reconstructions = predictions.get(recon_key) if recon_key else None
    elif isinstance(predictions, (list, tuple)):
        pred_classes = np.argmax(predictions[0], axis=1)
        reconstructions = predictions[1] if len(predictions) > 1 else None
    else:
        pred_classes = np.argmax(predictions, axis=1)
        reconstructions = None
    return pred_classes, reconstructions


def _extract_predictions_with_confidence(predictions) -> Tuple[np.ndarray, np.ndarray]:
    """Extract predicted classes and confidences from model output."""
    if isinstance(predictions, dict):
        length_key = None
        for key in predictions.keys():
            if 'length' in key.lower() or 'capsule' in key.lower():
                length_key = key
                break
        pred_key = length_key or list(predictions.keys())[0]
        probs = predictions[pred_key]
    elif isinstance(predictions, (list, tuple)):
        probs = predictions[0]
    else:
        probs = predictions
    return np.argmax(probs, axis=1), probs.max(axis=1)


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def plot_reconstruction_comparison(
        original_images: np.ndarray,
        reconstructed_images: np.ndarray,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        save_path: str,
        epoch: Optional[int] = None,
        dataset: str = 'mnist'
) -> None:
    """Plot original vs reconstructed images comparison."""
    n_samples = min(8, len(original_images))
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
    cmap = 'gray' if dataset.lower() == 'mnist' else None

    for i in range(n_samples):
        img_orig = original_images[i].squeeze() if dataset.lower() == 'mnist' else original_images[i]
        img_recon = reconstructed_images[i].squeeze() if dataset.lower() == 'mnist' else reconstructed_images[i]

        axes[0, i].imshow(img_orig, cmap=cmap)
        axes[0, i].set_title(f'True: {true_labels[i]}', fontsize=10)
        axes[0, i].axis('off')

        axes[1, i].imshow(np.clip(img_recon, 0, 1), cmap=cmap)
        axes[1, i].set_title(f'Pred: {predicted_labels[i]}', fontsize=10)
        axes[1, i].axis('off')

    title = f'Reconstruction - Epoch {epoch}' if epoch is not None else 'Final Reconstruction'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(history: keras.callbacks.History, save_dir: str) -> None:
    """Plot training history including loss, accuracy, and reconstruction curves."""
    h = history.history
    epochs = range(1, len(h['loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(epochs, h['loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, h['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    if 'capsule_accuracy' in h:
        axes[0, 1].plot(epochs, h['capsule_accuracy'], 'b-', label='Train', linewidth=2)
        if 'val_capsule_accuracy' in h:
            axes[0, 1].plot(epochs, h['val_capsule_accuracy'], 'r-', label='Val', linewidth=2)
        axes[0, 1].set_title('Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Accuracy Not Available',
                        transform=axes[0, 1].transAxes, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        axes[0, 1].set_title('Accuracy', fontsize=14, fontweight='bold')

    # Reconstruction loss
    if 'reconstruction_loss' in h:
        axes[1, 0].plot(epochs, h['reconstruction_loss'], 'g-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, h['val_reconstruction_loss'], 'm-', label='Val', linewidth=2)
        axes[1, 0].set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Reconstruction Loss\nNot Available',
                        transform=axes[1, 0].transAxes, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        axes[1, 0].set_title('Reconstruction Loss', fontsize=14, fontweight='bold')

    # Learning rate / margin loss / loss difference
    if 'lr' in h:
        axes[1, 1].plot(epochs, h['lr'], 'orange', linewidth=2)
        axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    elif 'margin_loss' in h:
        axes[1, 1].plot(epochs, h['margin_loss'], 'purple', label='Train', linewidth=2)
        if 'val_margin_loss' in h:
            axes[1, 1].plot(epochs, h['val_margin_loss'], 'orange', label='Val', linewidth=2)
        axes[1, 1].set_title('Margin Loss', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        loss_diff = np.array(h['val_loss']) - np.array(h['loss'])
        axes[1, 1].plot(epochs, loss_diff, 'red', linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Val - Train Loss', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_summary(history: keras.callbacks.History, test_results: Dict[str, float], save_dir: str) -> None:
    """Plot a summary of all metrics."""
    h = history.history
    epochs = range(1, len(h['loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Final metrics comparison bar chart
    metrics, train_vals, val_vals, test_vals = ['Loss'], [h['loss'][-1]], [h['val_loss'][-1]], [test_results.get('loss', 0)]
    if 'capsule_accuracy' in h:
        metrics.append('Accuracy')
        train_vals.append(h['capsule_accuracy'][-1])
        val_vals.append(h.get('val_capsule_accuracy', [0])[-1])
        test_vals.append(test_results.get('capsule_accuracy', 0))
    if 'reconstruction_loss' in h:
        metrics.append('Recon Loss')
        train_vals.append(h['reconstruction_loss'][-1])
        val_vals.append(h['val_reconstruction_loss'][-1])
        test_vals.append(test_results.get('reconstruction_loss', 0))

    x = np.arange(len(metrics))
    width = 0.25
    for offset, vals, label, color in [(-width, train_vals, 'Train', 'lightblue'),
                                        (0, val_vals, 'Val', 'lightcoral'),
                                        (width, test_vals, 'Test', 'lightgreen')]:
        bars = axes[0, 0].bar(x + offset, vals, width, label=label, alpha=0.8, color=color)
        for bar in bars:
            ht = bar.get_height()
            if ht > 0:
                axes[0, 0].annotate(f'{ht:.3f}', xy=(bar.get_x() + bar.get_width() / 2, ht),
                                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    axes[0, 0].set_title('Final Metrics', fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Training progress (loss + accuracy)
    axes[0, 1].plot(epochs, h['loss'], label='Train Loss', linewidth=2, color='blue')
    axes[0, 1].plot(epochs, h['val_loss'], label='Val Loss', linewidth=2, color='red')
    if 'capsule_accuracy' in h:
        ax2 = axes[0, 1].twinx()
        ax2.plot(epochs, h['capsule_accuracy'], '--', label='Train Acc', linewidth=2, color='lightblue')
        if 'val_capsule_accuracy' in h:
            ax2.plot(epochs, h['val_capsule_accuracy'], '--', label='Val Acc', linewidth=2, color='lightcoral')
        ax2.set_ylabel('Accuracy')
        ax2.legend(loc='center right')
        ax2.set_ylim([0.95, 1.0])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Progress', fontweight='bold')
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)

    # Loss components
    if 'margin_loss' in h:
        axes[1, 0].plot(epochs, h['loss'], label='Total', linewidth=2)
        axes[1, 0].plot(epochs, h['margin_loss'], label='Margin', linewidth=2)
        if 'reconstruction_loss' in h:
            recon_scaled = np.array(h['reconstruction_loss']) * 0.0005
            axes[1, 0].plot(epochs, recon_scaled, label='Recon (scaled)', linewidth=2)
        axes[1, 0].set_title('Loss Components', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    elif 'reconstruction_loss' in h:
        axes[1, 0].plot(epochs, h['reconstruction_loss'], 'g-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, h['val_reconstruction_loss'], 'm-', label='Val', linewidth=2)
        axes[1, 0].set_title('Reconstruction Loss Detail', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Loss Components\nNot Available',
                        transform=axes[1, 0].transAxes, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        axes[1, 0].set_title('Loss Components', fontweight='bold')

    # Training stability
    if len(list(epochs)) > 5:
        window = max(3, len(list(epochs)) // 10)
        loss_ma = pd.Series(h['loss']).rolling(window=window, center=True).mean()
        val_loss_ma = pd.Series(h['val_loss']).rolling(window=window, center=True).mean()
        axes[1, 1].plot(epochs, h['loss'], alpha=0.3, color='blue', label='Raw Train')
        axes[1, 1].plot(epochs, h['val_loss'], alpha=0.3, color='red', label='Raw Val')
        axes[1, 1].plot(epochs, loss_ma, linewidth=2, color='blue', label=f'Train MA-{window}')
        axes[1, 1].plot(epochs, val_loss_ma, linewidth=2, color='red', label=f'Val MA-{window}')
        axes[1, 1].set_title('Training Stability', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        n = min(5, len(list(epochs)))
        summary = (f"Final {n} epochs avg:\n"
                   f"Train Loss: {np.mean(h['loss'][-n:]):.4f}\n"
                   f"Val Loss: {np.mean(h['val_loss'][-n:]):.4f}\n"
                   f"Overfit: {np.mean(h['val_loss'][-n:]) / np.mean(h['loss'][-n:]):.2f}x")
        axes[1, 1].text(0.5, 0.5, summary, transform=axes[1, 1].transAxes, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5), fontsize=12)
        axes[1, 1].set_title('Training Summary', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------
# Custom callbacks
# ---------------------------------------------------------------------

class LearningRateLogger(keras.callbacks.Callback):
    """Log learning rate to history each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        logs['lr'] = float(self.model.optimizer.learning_rate)


class ValidationAccuracyLogger(keras.callbacks.Callback):
    """Manually compute validation accuracy if not provided by model."""
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if 'val_capsule_accuracy' not in logs:
            x_val, y_val = self.validation_data
            predictions = self.model.predict(x_val, verbose=0)
            pred_classes, _ = _extract_predictions(predictions)
            true_classes = np.argmax(y_val, axis=1)
            logs['val_capsule_accuracy'] = np.mean(pred_classes == true_classes)


class ReconstructionVisualizationCallback(keras.callbacks.Callback):
    """Generate reconstruction visualizations during training."""

    def __init__(
            self,
            validation_data: Tuple[np.ndarray, np.ndarray],
            save_dir: str,
            dataset: str = 'mnist',
            n_samples: int = 8,
            frequency: int = 5
    ):
        super().__init__()
        self.dataset = dataset
        self.frequency = frequency
        self.recon_dir = os.path.join(save_dir, 'reconstructions_per_epoch')
        os.makedirs(self.recon_dir, exist_ok=True)

        x_val, y_val = validation_data
        indices = np.random.choice(len(x_val), size=n_samples, replace=False)
        self.sample_images = x_val[indices]
        self.sample_labels = y_val[indices]
        self.true_classes = np.argmax(self.sample_labels, axis=1)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        if (epoch + 1) % self.frequency != 0:
            return
        try:
            predictions = self.model.predict(self.sample_images, verbose=0)

            if epoch == 0:
                if isinstance(predictions, dict):
                    logger.info(f"Prediction keys: {list(predictions.keys())}")
                    for key, value in predictions.items():
                        logger.info(f"  {key}: shape {value.shape}")
                elif isinstance(predictions, (list, tuple)):
                    for i, pred in enumerate(predictions):
                        logger.info(f"  Prediction[{i}]: shape {pred.shape}")

            pred_classes, reconstructions = _extract_predictions(predictions)

            if reconstructions is not None:
                save_path = os.path.join(self.recon_dir, f'epoch_{epoch + 1:03d}.png')
                plot_reconstruction_comparison(
                    self.sample_images, reconstructions,
                    self.true_classes, pred_classes,
                    save_path, epoch=epoch + 1, dataset=self.dataset
                )
                logger.info(f"Reconstruction saved: {save_path}")
            else:
                logger.warning(f"No reconstructions in output for epoch {epoch + 1}")
        except Exception as e:
            import traceback
            logger.error(f"Reconstruction failed epoch {epoch + 1}: {e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------
# Callbacks factory
# ---------------------------------------------------------------------

def create_capsnet_callbacks(
        model_name: str,
        validation_data: Tuple[np.ndarray, np.ndarray],
        dataset: str,
        monitor: str = 'val_accuracy',
        patience: int = 10,
        save_best_only: bool = True,
        viz_frequency: int = 1
) -> Tuple[list, str]:
    """Create training callbacks including CapsNet-specific visualization callback.

    Returns:
        Tuple of (callbacks list, results directory path).
    """
    callbacks, results_dir = create_callbacks(
        model_name=model_name,
        results_dir_prefix="capsnet",
        monitor=monitor,
        patience=patience,
        use_lr_schedule=False,
        include_tensorboard=True,
        include_analyzer=True,
    )

    # CapsNet-specific callbacks
    callbacks.extend([
        LearningRateLogger(),
        ReconstructionVisualizationCallback(
            validation_data=validation_data, save_dir=results_dir,
            dataset=dataset, n_samples=8, frequency=viz_frequency
        ),
    ])

    return callbacks, results_dir


# ---------------------------------------------------------------------
# Final visualizations
# ---------------------------------------------------------------------

def generate_final_visualizations(
        model: keras.Model,
        test_data: Tuple[np.ndarray, np.ndarray],
        history: keras.callbacks.History,
        results_dir: str,
        dataset: str,
        test_results_dict: Optional[Dict[str, float]] = None,
        n_samples: int = 16
) -> None:
    """Generate comprehensive final visualizations."""
    x_test, y_test = test_data

    final_viz_dir = os.path.join(results_dir, 'final_visualizations')
    os.makedirs(final_viz_dir, exist_ok=True)

    if test_results_dict is None:
        test_results = model.evaluate(x_test, y_test, verbose=0)
        if isinstance(test_results, (list, tuple)):
            test_results_dict = dict(zip(model.metrics_names, test_results))
        elif isinstance(test_results, dict):
            test_results_dict = test_results
        else:
            test_results_dict = {"test_loss": test_results}

    plot_training_history(history, final_viz_dir)
    plot_metrics_summary(history, test_results_dict, final_viz_dir)

    # Final reconstruction comparison
    sample_indices = np.random.choice(len(x_test), size=n_samples, replace=False)
    sample_images = x_test[sample_indices]
    sample_labels = y_test[sample_indices]
    true_classes = np.argmax(sample_labels, axis=1)

    predictions = model.predict(sample_images, verbose=0)
    pred_classes, reconstructions = _extract_predictions(predictions)

    if reconstructions is not None:
        save_path = os.path.join(final_viz_dir, 'final_reconstructions.png')
        plot_reconstruction_comparison(
            sample_images, reconstructions, true_classes, pred_classes,
            save_path, dataset=dataset
        )
        recon_error = np.mean(np.square(sample_images - reconstructions))
        with open(os.path.join(final_viz_dir, 'reconstruction_stats.txt'), 'w') as f:
            f.write(f"Reconstruction Statistics\n{'=' * 24}\n")
            f.write(f"MSE: {recon_error:.6f}\n")
            f.write(f"Correct: {np.sum(true_classes == pred_classes)}/{n_samples}\n")
            f.write(f"Accuracy: {np.mean(true_classes == pred_classes):.4f}\n")

    # Training summary
    h = history.history
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"CapsNet Training Summary\n{'=' * 23}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Epochs: {len(h['loss'])}\n")
        f.write(f"Final train loss: {h['loss'][-1]:.6f}\n")
        f.write(f"Final val loss: {h['val_loss'][-1]:.6f}\n")
        if 'capsule_accuracy' in h:
            f.write(f"Final train acc: {h['capsule_accuracy'][-1]:.4f}\n")
            if 'val_capsule_accuracy' in h:
                f.write(f"Final val acc: {h['val_capsule_accuracy'][-1]:.4f}\n")
        f.write(f"\nTest Results:\n")
        for name, value in test_results_dict.items():
            f.write(f"  {name}: {value:.6f}\n")

    logger.info(f"Final visualizations saved to: {final_viz_dir}")


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def _clean_test_results(test_results, model) -> Dict[str, float]:
    """Convert model.evaluate output to a clean {metric: float} dict."""
    if isinstance(test_results, (list, tuple)):
        raw = dict(zip(model.metrics_names, test_results))
    elif isinstance(test_results, dict):
        raw = test_results
    else:
        return {"test_loss": float(test_results)}

    cleaned = {}
    for key, value in raw.items():
        if isinstance(value, (int, float, np.number)):
            cleaned[key] = float(value)
        elif isinstance(value, dict):
            if 'loss' in value:
                cleaned[key + '_loss'] = float(value['loss'])
            elif len(value) == 1:
                sub_val = next(iter(value.values()))
                if isinstance(sub_val, (int, float, np.number)):
                    cleaned[key] = float(sub_val)
        else:
            try:
                cleaned[key] = float(value)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert {key}={value} to float, using 0")
                cleaned[key] = 0.0
    return cleaned


def train_model(args) -> None:
    """Main CapsNet training function."""
    setup_gpu(gpu_id=args.gpu)

    logger.info("Starting CapsNet training")
    logger.info(f"Arguments: {vars(args)}")

    # Load data (CapsNet needs one-hot labels)
    if args.dataset.lower() == 'mnist':
        train_data, val_data, test_data = load_mnist_data(args.validation_split)
    elif args.dataset.lower() == 'cifar10':
        train_data, val_data, test_data = load_cifar10_data(args.validation_split)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    x_train, y_train = train_data
    x_val, y_val = val_data
    x_test, y_test = test_data

    # Create and build model
    model_config = create_model_config(args.dataset)
    model = create_capsnet(
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        **model_config
    )
    sample_input = tf.zeros((1,) + model_config['input_shape'])
    _ = model(sample_input, training=False)
    model.summary()

    # Callbacks
    callbacks, results_dir = create_capsnet_callbacks(
        model_name=args.dataset,
        validation_data=(x_val, y_val),
        dataset=args.dataset,
        patience=args.patience,
        save_best_only=True,
        viz_frequency=args.viz_frequency
    )

    # Train
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    test_results = model.evaluate(x_test, y_test, verbose=1)
    test_results_dict = _clean_test_results(test_results, model)
    logger.info("Test results:")
    for name, value in test_results_dict.items():
        logger.info(f"  {name}: {value:.4f}")

    # Visualizations
    generate_final_visualizations(
        model=model, test_data=(x_test, y_test), history=history,
        results_dir=results_dir, dataset=args.dataset,
        test_results_dict=test_results_dict, n_samples=16
    )

    # Save final model
    final_model_path = os.path.join(results_dir, f"capsnet_{args.dataset}_final.keras")
    model.save(final_model_path)
    logger.info(f"Final model saved: {final_model_path}")

    # Sample predictions
    sample_images = x_test[np.random.choice(len(x_test), size=5, replace=False)]
    predictions = model.predict(sample_images, verbose=0)
    pred_classes, confidences = _extract_predictions_with_confidence(predictions)
    true_classes = np.argmax(y_test[:5], axis=1)
    for i in range(5):
        logger.info(f"  Sample {i}: True={true_classes[i]}, Pred={pred_classes[i]}, Conf={confidences[i]:.4f}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = create_base_argument_parser(
        description='Train CapsNet on image classification datasets',
        default_dataset='mnist',
        dataset_choices=['mnist', 'cifar10']
    )

    # CapsNet-specific arguments
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use (default: adam)')
    parser.add_argument('--validation-split', type=float, default=0.1,
                        help='Fraction of training data for validation (default: 0.1)')
    parser.add_argument('--no-reconstruction', action='store_true',
                        help='Disable reconstruction decoder')
    parser.add_argument('--reconstruction-weight', type=float, default=0.0005,
                        help='Weight for reconstruction loss (default: 0.0005)')
    parser.add_argument('--viz-frequency', type=int, default=1,
                        help='Reconstruction visualization frequency in epochs (default: 1)')

    args = parser.parse_args()

    try:
        train_model(args)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
