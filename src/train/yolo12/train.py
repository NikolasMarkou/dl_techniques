#!/usr/bin/env python3
"""
Enhanced training script for YOLOv12 object detection model.

This script demonstrates how to train the YOLOv12 model using standard Keras
workflows with model.compile() and model.fit(). The script handles data loading,
model creation, training, evaluation, and generates comprehensive visualizations.

Features:
    - Complete training pipeline with proper data handling
    - Comprehensive metrics and visualization
    - Model saving and loading capabilities
    - Configurable hyperparameters
    - Support for different model scales
    - Integration with the dl_techniques framework

Usage:
    python train.py [--scale n] [--epochs 300] [--batch-size 16] [--img-size 640]

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
import json

from dl_techniques.models.yolo12 import create_yolov12
from dl_techniques.losses.yolo12_loss import create_yolov12_loss
from dl_techniques.utils.logger import logger

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


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


def create_dummy_coco_dataset(
        num_samples: int,
        img_size: int,
        num_classes: int = 80,
        max_boxes: int = 20,
        min_boxes: int = 1
) -> tf.data.Dataset:
    """Create a dummy COCO-style dataset for training demonstration.

    Args:
        num_samples: Number of samples to generate.
        img_size: Input image size.
        num_classes: Number of object classes.
        max_boxes: Maximum number of boxes per image.
        min_boxes: Minimum number of boxes per image.

    Returns:
        TensorFlow dataset with (image, labels) pairs.
    """

    def generator():
        for _ in range(num_samples):
            # Generate dummy image
            img = np.random.rand(img_size, img_size, 3).astype(np.float32)

            # Generate random number of boxes
            num_boxes = np.random.randint(min_boxes, max_boxes + 1)

            # Initialize labels array
            labels = np.zeros((max_boxes, 5), dtype=np.float32)

            for i in range(num_boxes):
                # Random class
                cls_id = np.random.randint(0, num_classes)

                # Random box coordinates (ensure valid boxes)
                x1 = np.random.uniform(0, img_size * 0.8)
                y1 = np.random.uniform(0, img_size * 0.8)
                x2 = np.random.uniform(x1 + 20, min(x1 + 200, img_size))
                y2 = np.random.uniform(y1 + 20, min(y1 + 200, img_size))

                labels[i] = [cls_id, x1, y1, x2, y2]

            yield img, labels

    output_signature = (
        tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(max_boxes, 5), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset


def create_model_config(scale: str, num_classes: int, img_size: int) -> Dict[str, Any]:
    """Create model configuration based on scale and dataset.

    Args:
        scale: Model scale ('n', 's', 'm', 'l', 'x').
        num_classes: Number of object classes.
        img_size: Input image size.

    Returns:
        Dictionary of model configuration parameters.
    """
    return {
        'num_classes': num_classes,
        'input_shape': (img_size, img_size, 3),
        'scale': scale,
        'reg_max': 16,
        'kernel_initializer': 'he_normal',
    }


def create_loss_config(num_classes: int, img_size: int) -> Dict[str, Any]:
    """Create loss function configuration.

    Args:
        num_classes: Number of object classes.
        img_size: Input image size.

    Returns:
        Dictionary of loss configuration parameters.
    """
    return {
        'num_classes': num_classes,
        'input_shape': (img_size, img_size),
        'reg_max': 16,
        'box_weight': 7.5,
        'cls_weight': 0.5,
        'dfl_weight': 1.5,
        'assigner_topk': 10,
        'assigner_alpha': 0.5,
        'assigner_beta': 6.0,
    }


class YOLOv12Metrics:
    """Custom metrics for YOLOv12 training."""

    def __init__(self, num_classes: int = 80):
        self.num_classes = num_classes
        self.reset_states()

    def reset_states(self):
        """Reset metric states."""
        self.total_loss = 0.0
        self.box_loss = 0.0
        self.cls_loss = 0.0
        self.dfl_loss = 0.0
        self.count = 0

    def update_state(self, loss_dict: Dict[str, float]):
        """Update metric states."""
        self.total_loss += loss_dict.get('total_loss', 0.0)
        self.box_loss += loss_dict.get('box_loss', 0.0)
        self.cls_loss += loss_dict.get('cls_loss', 0.0)
        self.dfl_loss += loss_dict.get('dfl_loss', 0.0)
        self.count += 1

    def result(self) -> Dict[str, float]:
        """Get metric results."""
        if self.count == 0:
            return {'total_loss': 0.0, 'box_loss': 0.0, 'cls_loss': 0.0, 'dfl_loss': 0.0}

        return {
            'total_loss': self.total_loss / self.count,
            'box_loss': self.box_loss / self.count,
            'cls_loss': self.cls_loss / self.count,
            'dfl_loss': self.dfl_loss / self.count,
        }


def plot_training_history(history: keras.callbacks.History, save_dir: str) -> None:
    """Plot training history including loss curves.

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
    if 'val_loss' in history_dict:
        axes[0, 0].plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot learning rate if available
    if 'lr' in history_dict:
        axes[0, 1].plot(epochs, history_dict['lr'], 'orange', linewidth=2)
        axes[0, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Learning Rate\nNot Available',
                        transform=axes[0, 1].transAxes, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        axes[0, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')

    # Plot loss components if available
    loss_components = ['box_loss', 'cls_loss', 'dfl_loss']
    available_components = [comp for comp in loss_components if comp in history_dict]

    if available_components:
        for i, comp in enumerate(available_components):
            color = plt.cm.Set1(i)
            axes[1, 0].plot(epochs, history_dict[comp], color=color, label=comp.replace('_', ' ').title(), linewidth=2)
        axes[1, 0].set_title('Loss Components', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Loss Components\nNot Available',
                        transform=axes[1, 0].transAxes, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        axes[1, 0].set_title('Loss Components', fontsize=14, fontweight='bold')

    # Plot training stability (moving average)
    if len(epochs) > 5:
        window = max(3, len(epochs) // 10)
        loss_rolling = pd.Series(history_dict['loss']).rolling(window=window, center=True).mean()

        axes[1, 1].plot(epochs, history_dict['loss'], alpha=0.3, color='blue', label='Raw Loss')
        axes[1, 1].plot(epochs, loss_rolling, linewidth=2, color='blue', label=f'Loss (MA-{window})')

        if 'val_loss' in history_dict:
            val_loss_rolling = pd.Series(history_dict['val_loss']).rolling(window=window, center=True).mean()
            axes[1, 1].plot(epochs, history_dict['val_loss'], alpha=0.3, color='red', label='Raw Val Loss')
            axes[1, 1].plot(epochs, val_loss_rolling, linewidth=2, color='red', label=f'Val Loss (MA-{window})')

        axes[1, 1].set_title('Training Stability (Moving Average)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Not Enough Epochs\nfor Moving Average',
                        transform=axes[1, 1].transAxes, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        axes[1, 1].set_title('Training Stability', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_architecture(model: keras.Model, save_dir: str) -> None:
    """Plot model architecture diagram.

    Args:
        model: YOLOv12 model.
        save_dir: Directory to save the plot.
    """
    try:
        keras.utils.plot_model(
            model,
            to_file=os.path.join(save_dir, 'model_architecture.png'),
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=150
        )
        logger.info("Model architecture diagram saved")
    except Exception as e:
        logger.warning(f"Failed to save model architecture diagram: {e}")


class LearningRateLogger(keras.callbacks.Callback):
    """Custom callback to log learning rate to history."""

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        # Get learning rate from optimizer
        lr = float(self.model.optimizer.learning_rate)
        logs['lr'] = lr


def create_callbacks(
        model_name: str,
        results_dir: str,
        monitor: str = 'val_loss',
        patience: int = 50,
        save_best_only: bool = True
) -> List[keras.callbacks.Callback]:
    """Create training callbacks.

    Args:
        model_name: Name for saved model files.
        results_dir: Results directory path.
        monitor: Metric to monitor for callbacks.
        patience: Patience for early stopping.
        save_best_only: Whether to save only the best model.

    Returns:
        List of callbacks.
    """
    callbacks = [
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            mode='min',
            verbose=1
        ),

        # Model checkpointing
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(results_dir, 'best_model.keras'),
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),

        # Learning rate reduction
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if 'val' in monitor else 'loss',
            factor=0.5,
            patience=max(10, patience // 3),
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
    ]

    return callbacks


def generate_final_visualizations(
        model: keras.Model,
        history: keras.callbacks.History,
        results_dir: str,
        model_config: Dict[str, Any],
        test_results: Optional[Dict[str, float]] = None
) -> None:
    """Generate comprehensive final visualizations.

    Args:
        model: Trained YOLOv12 model.
        history: Training history.
        results_dir: Directory to save visualizations.
        model_config: Model configuration dictionary.
        test_results: Test results dictionary.
    """
    logger.info("Generating final visualizations...")

    # Create final visualizations directory
    final_viz_dir = os.path.join(results_dir, 'final_visualizations')
    os.makedirs(final_viz_dir, exist_ok=True)

    # Plot training history
    plot_training_history(history, final_viz_dir)

    # Plot model architecture
    plot_model_architecture(model, final_viz_dir)

    # Save model summary
    with open(os.path.join(final_viz_dir, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Save training configuration
    config_dict = {
        'model_config': model_config,
        'training_epochs': len(history.history['loss']),
        'final_training_loss': float(history.history['loss'][-1]),
        'final_validation_loss': float(
            history.history.get('val_loss', [0])[-1]) if 'val_loss' in history.history else None,
        'best_validation_loss': float(
            min(history.history.get('val_loss', [float('inf')]))) if 'val_loss' in history.history else None,
    }

    if test_results:
        config_dict['test_results'] = test_results

    with open(os.path.join(final_viz_dir, 'training_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Save training summary
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"YOLOv12 Training Summary\n")
        f.write(f"========================\n")
        f.write(f"Model Scale: {model_config['scale']}\n")
        f.write(f"Number of Classes: {model_config['num_classes']}\n")
        f.write(f"Input Shape: {model_config['input_shape']}\n")
        f.write(f"Total Epochs: {len(history.history['loss'])}\n")
        f.write(f"Final Training Loss: {history.history['loss'][-1]:.6f}\n")

        if 'val_loss' in history.history:
            f.write(f"Final Validation Loss: {history.history['val_loss'][-1]:.6f}\n")
            f.write(f"Best Validation Loss: {min(history.history['val_loss']):.6f}\n")

        if test_results:
            f.write(f"\nTest Results:\n")
            for metric_name, value in test_results.items():
                f.write(f"{metric_name}: {value:.6f}\n")

        # Model parameters
        total_params = model.count_params()
        trainable_params = sum([keras.backend.count_params(w) for w in model.trainable_weights])
        f.write(f"\nModel Statistics:\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Non-trainable Parameters: {total_params - trainable_params:,}\n")

    logger.info(f"Final visualizations saved to: {final_viz_dir}")


def train_model(args: argparse.Namespace) -> None:
    """Main training function.

    Args:
        args: Command line arguments.
    """
    logger.info("Starting YOLOv12 training")
    logger.info(f"Arguments: {vars(args)}")

    # Setup GPU
    setup_gpu()

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/yolov12_{args.scale}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {results_dir}")

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = create_dummy_coco_dataset(
        num_samples=args.train_samples,
        img_size=args.img_size,
        num_classes=args.num_classes,
        max_boxes=args.max_boxes
    )

    val_dataset = create_dummy_coco_dataset(
        num_samples=args.val_samples,
        img_size=args.img_size,
        num_classes=args.num_classes,
        max_boxes=args.max_boxes
    )

    test_dataset = create_dummy_coco_dataset(
        num_samples=args.test_samples,
        img_size=args.img_size,
        num_classes=args.num_classes,
        max_boxes=args.max_boxes
    )

    # Prepare datasets
    train_dataset = train_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    logger.info(f"Train dataset: {train_dataset.element_spec}")
    logger.info(f"Validation dataset: {val_dataset.element_spec}")

    # Create model
    logger.info("Creating YOLOv12 model...")
    model_config = create_model_config(args.scale, args.num_classes, args.img_size)
    model = create_yolov12(**model_config)

    # Build model by calling it once
    sample_input = tf.zeros((1, args.img_size, args.img_size, 3))
    _ = model(sample_input, training=False)

    # Print model summary
    logger.info("Model architecture:")
    model.summary()

    # Create loss function
    logger.info("Creating loss function...")
    loss_config = create_loss_config(args.num_classes, args.img_size)
    loss_fn = create_yolov12_loss(**loss_config)

    # Create optimizer
    if args.optimizer.lower() == 'adamw':
        optimizer = keras.optimizers.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == 'sgd':
        optimizer = keras.optimizers.SGD(
            learning_rate=args.learning_rate,
            momentum=0.9,
            nesterov=True
        )
    else:
        optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        run_eagerly=args.run_eagerly
    )

    # Test the loss function with a single batch first
    logger.info("Testing loss function with sample data...")
    try:
        sample_batch = next(iter(train_dataset))
        test_loss = model.evaluate(sample_batch[0], sample_batch[1], steps=1, verbose=0)
        logger.info(f"✓ Loss function test successful: {test_loss:.6f}")
    except Exception as e:
        logger.error(f"✗ Loss function test failed: {e}")
        logger.info("Switching to eager execution mode for debugging...")
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            run_eagerly=True
        )

    # Create callbacks
    callbacks = create_callbacks(
        model_name=f"yolov12_{args.scale}",
        results_dir=results_dir,
        monitor='val_loss' if args.validation else 'loss',
        patience=args.patience,
        save_best_only=True
    )

    # Train model
    logger.info("Starting training...")
    validation_data = val_dataset if args.validation else None

    history = model.fit(
        train_dataset,
        validation_data=validation_data,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set if requested
    test_results = None
    if args.evaluate:
        logger.info("Evaluating on test set...")
        test_loss = model.evaluate(test_dataset, verbose=1)
        test_results = {'test_loss': float(test_loss)}

        logger.info("Test Results:")
        logger.info(f"  Test Loss: {test_loss:.6f}")

    # Generate final visualizations
    generate_final_visualizations(
        model=model,
        history=history,
        results_dir=results_dir,
        model_config=model_config,
        test_results=test_results
    )

    # Save final model
    final_model_path = os.path.join(results_dir, f"yolov12_{args.scale}_final.keras")
    model.save(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")

    logger.info("Training completed successfully!")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Train YOLOv12 object detection model')

    # Model arguments
    parser.add_argument('--scale', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='Model scale (default: n)')
    parser.add_argument('--num-classes', type=int, default=80,
                        help='Number of object classes (default: 80)')
    parser.add_argument('--img-size', type=int, default=320,
                        help='Input image size (default: 320)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs (default: 300)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Training batch size (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Initial learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay for AdamW optimizer (default: 5e-4)')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type (default: sgd)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (default: 50)')

    # Dataset arguments
    parser.add_argument('--train-samples', type=int, default=1000,
                        help='Number of training samples (default: 1000)')
    parser.add_argument('--val-samples', type=int, default=200,
                        help='Number of validation samples (default: 200)')
    parser.add_argument('--test-samples', type=int, default=100,
                        help='Number of test samples (default: 100)')
    parser.add_argument('--max-boxes', type=int, default=20,
                        help='Maximum boxes per image (default: 20)')

    # Control arguments
    parser.add_argument('--no-validation', action='store_true',
                        help='Disable validation during training')
    parser.add_argument('--no-evaluate', action='store_true',
                        help='Skip evaluation on test set')
    parser.add_argument('--run-eagerly', action='store_true',
                        help='Run model in eager mode (useful for debugging)')

    # Parse arguments
    args = parser.parse_args()
    args.validation = not args.no_validation
    args.evaluate = not args.no_evaluate

    # Start training
    try:
        train_model(args)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == '__main__':
    main()