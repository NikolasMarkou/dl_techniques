#!/usr/bin/env python3
"""
Enhanced training script for PowerMLP with comprehensive visualizations.

PowerMLP is an efficient alternative to Kolmogorov-Arnold Networks (KAN) offering:
- ~40x faster training time
- ~10x fewer FLOPs
- Equal or better performance
- Simpler implementation

Usage:
    python train.py --dataset mnist --epochs 50 --architecture small --k 2
    python train.py --dataset mnist --k 2 --dropout-rate 0.2 --batch-normalization
"""

import os
import json
import argparse
import numpy as np
import keras
import tensorflow as tf
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
from sklearn.metrics import classification_report, confusion_matrix

from dl_techniques.models.power_mlp.model import PowerMLP
from dl_techniques.utils.logger import logger

from train.common import (
    setup_gpu,
    load_dataset,
    get_class_names,
    create_callbacks,
    run_model_analysis,
)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# ---------------------------------------------------------------------
# PowerMLP-specific callbacks
# ---------------------------------------------------------------------

class LearningRateLogger(keras.callbacks.Callback):
    """Custom callback to log learning rate to history."""

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        lr = float(self.model.optimizer.learning_rate)
        logs['lr'] = lr


class NaNStoppingCallback(keras.callbacks.Callback):
    """Custom callback to stop training when NaN values are detected."""

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        for key, value in logs.items():
            if np.isnan(value) or np.isinf(value):
                logger.error(f"NaN or Inf detected in {key}: {value}. Stopping training.")
                self.model.stop_training = True
                return


# ---------------------------------------------------------------------
# PowerMLP-specific data preparation (flatten + one-hot + val split)
# ---------------------------------------------------------------------

def prepare_mlp_data(
        dataset_name: str,
        validation_split: float = 0.1,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], int]:
    """Load dataset via common loader and adapt for PowerMLP (flatten, one-hot, val split).

    Returns:
        Tuple of (train_data, val_data, test_data, num_classes).
    """
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = load_dataset(dataset_name)

    # Flatten for MLP input
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Validation split
    if validation_split > 0:
        val_size = int(len(x_train) * validation_split)
        x_val, y_val = x_train[:val_size], y_train[:val_size]
        x_train, y_train = x_train[val_size:], y_train[val_size:]
    else:
        x_val, y_val = x_test, y_test

    logger.info(f"  MLP data — Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), num_classes


# ---------------------------------------------------------------------

def create_model_config(
        dataset: str,
        architecture: str = "default",
        k: int = 2,
        dropout_rate: float = 0.0,
        batch_normalization: bool = False
) -> Dict[str, Any]:
    """Create model configuration based on dataset and architecture."""
    base_config = {
        'k': k,
        'dropout_rate': dropout_rate,
        'batch_normalization': batch_normalization,
        'output_activation': 'softmax',
        'kernel_initializer': 'glorot_normal',
        'bias_initializer': 'zeros',
        'kernel_regularizer': keras.regularizers.L2(1e-5) if dropout_rate == 0.0 else None,
    }

    architectures = {
        'small': {'mnist': [128, 64, 10], 'cifar10': [256, 128, 10]},
        'default': {'mnist': [256, 128, 64, 10], 'cifar10': [512, 256, 128, 10]},
        'large': {'mnist': [512, 256, 128, 64, 10], 'cifar10': [1024, 512, 256, 128, 10]},
        'deep': {'mnist': [256, 128, 128, 64, 64, 32, 10], 'cifar10': [512, 256, 256, 128, 128, 64, 10]},
    }

    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}")
    if dataset.lower() not in architectures[architecture]:
        raise ValueError(f"Unsupported dataset: {dataset}")

    base_config['hidden_units'] = architectures[architecture][dataset.lower()]
    return base_config


# ---------------------------------------------------------------------

def create_stable_optimizer(optimizer_name: str, learning_rate: float) -> keras.optimizers.Optimizer:
    """Create an optimizer with gradient clipping for stable training."""
    if optimizer_name.lower() == 'adam':
        return keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0, epsilon=1e-7)
    elif optimizer_name.lower() == 'adamw':
        return keras.optimizers.AdamW(learning_rate=learning_rate, clipnorm=1.0, weight_decay=0.01, epsilon=1e-7)
    elif optimizer_name.lower() == 'sgd':
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, clipnorm=1.0)
    else:
        return keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)


# ---------------------------------------------------------------------
# PowerMLP-specific visualization (custom matplotlib plots)
# ---------------------------------------------------------------------

def plot_training_history(history: keras.callbacks.History, save_dir: str) -> None:
    """Plot training history including loss and accuracy curves."""
    history_dict = history.history
    epochs = range(1, len(history_dict['loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(epochs, history_dict['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    if 'accuracy' in history_dict:
        axes[0, 1].plot(epochs, history_dict['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history_dict:
            axes[0, 1].plot(epochs, history_dict['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Learning rate or loss difference
    if 'lr' in history_dict:
        axes[1, 0].plot(epochs, history_dict['lr'], 'orange', linewidth=2)
        axes[1, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    elif 'val_loss' in history_dict:
        loss_diff = np.array(history_dict['val_loss']) - np.array(history_dict['loss'])
        axes[1, 0].plot(epochs, loss_diff, 'red', linewidth=2)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Validation - Training Loss', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss Difference')
        axes[1, 0].grid(True, alpha=0.3)

    # Stability (moving average)
    if len(epochs) > 5:
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

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_dir: str,
        class_names: Optional[List[str]] = None,
) -> None:
    """Plot confusion matrix for classification results."""
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Raw Counts)', fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Confusion Matrix (Normalized)', fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------

def train_model(args: argparse.Namespace) -> None:
    """Main training function."""
    logger.info("Starting PowerMLP training")
    setup_gpu()

    # Load data (flattened + one-hot for MLP)
    train_data, val_data, test_data, num_classes = prepare_mlp_data(
        args.dataset, args.validation_split
    )
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

    # Create model
    optimizer = create_stable_optimizer(args.optimizer, args.learning_rate)
    model = PowerMLP(**model_config)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Build model
    sample_input = tf.zeros((1, x_train.shape[1]))
    _ = model(sample_input, training=False)
    model.summary(print_fn=logger.info)

    # Sanity check
    try:
        test_loss = model.evaluate(x_train[:32], y_train[:32], verbose=0)
        logger.info(f"Initial sanity check - loss: {test_loss[0]:.4f}, accuracy: {test_loss[1]:.4f}")
        if np.isnan(test_loss[0]) or np.isinf(test_loss[0]):
            logger.error("Model produces NaN/Inf before training! Check configuration.")
            return
    except Exception as e:
        logger.warning(f"Sanity check failed: {e}. Proceeding...")

    logger.info(f"  Architecture: {model_config['hidden_units']}, k={model_config['k']}")
    logger.info(f"  Dropout: {model_config['dropout_rate']}, BatchNorm: {model_config['batch_normalization']}")
    logger.info(f"  LR: {args.learning_rate}, Optimizer: {args.optimizer}, Params: {model.count_params():,}")

    # Create callbacks (common + model-specific)
    callbacks, results_dir = create_callbacks(
        model_name=f"{args.dataset}_{args.architecture}",
        results_dir_prefix="powermlp",
        monitor='val_loss',
        patience=args.patience,
        use_lr_schedule=False,  # PowerMLP uses ReduceLROnPlateau
    )
    # Add PowerMLP-specific callbacks
    callbacks.insert(0, NaNStoppingCallback())
    callbacks.append(LearningRateLogger())

    # Train
    logger.info("Starting training...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    logger.info("Evaluating on test set...")
    test_results = model.evaluate(x_test, y_test, verbose=1, return_dict=True)
    logger.info(f"Test results: {test_results}")

    # Generate PowerMLP-specific visualizations
    viz_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    plot_training_history(history, viz_dir)

    predictions = model.predict(x_test, verbose=0)
    class_names = get_class_names(args.dataset, num_classes)

    plot_confusion_matrix(y_test, predictions, viz_dir, class_names)

    # Classification report
    y_true_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(predictions, axis=1)
    report = classification_report(y_true_classes, y_pred_classes,
                                   target_names=class_names, output_dict=True)
    with open(os.path.join(viz_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # Run model analysis (convert one-hot back to integer labels for analyzer)
    run_model_analysis(
        model=model,
        test_data=(x_test, y_true_classes),
        training_history=history,
        model_name=f"powermlp_{args.dataset}_{args.architecture}",
        results_dir=results_dir,
    )

    # Save final model
    final_model_path = os.path.join(results_dir, f"powermlp_{args.dataset}_final.keras")
    model.save(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")

    # Save training summary
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"PowerMLP Training Summary\n")
        f.write(f"========================\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Architecture: {model_config['hidden_units']}\n")
        f.write(f"ReLU-k power: {model_config['k']}\n")
        f.write(f"Dropout rate: {model_config['dropout_rate']}\n")
        f.write(f"Batch normalization: {model_config['batch_normalization']}\n")
        f.write(f"Total parameters: {model.count_params():,}\n")
        f.write(f"Epochs trained: {len(history.history['loss'])}\n\n")

        f.write(f"Test Results:\n")
        for key, val in test_results.items():
            f.write(f"  {key}: {val:.4f}\n")

        best_val_acc = max(history.history.get('val_accuracy', [0]))
        f.write(f"\nBest validation accuracy: {best_val_acc:.4f}\n")

    logger.info("Training completed successfully!")
    logger.info(f"Results saved to: {results_dir}")


# ---------------------------------------------------------------------

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train PowerMLP on image classification datasets'
    )

    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar10'], help='Dataset to use')
    parser.add_argument('--validation-split', type=float, default=0.1,
                        help='Validation split fraction')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use')
    parser.add_argument('--learning-rate', type=float, default=0.0003,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--architecture', type=str, default='default',
                        choices=['small', 'default', 'large', 'deep'],
                        help='Model architecture size')
    parser.add_argument('--k', type=int, default=2,
                        help='Power for ReLU-k activation')
    parser.add_argument('--dropout-rate', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--batch-normalization', action='store_true',
                        help='Enable batch normalization')

    args = parser.parse_args()

    try:
        train_model(args)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------

if __name__ == '__main__':
    main()
