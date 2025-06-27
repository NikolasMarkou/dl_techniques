"""
Comprehensive Training Script for YOLOv12 Multi-Task Model.

This script provides complete training pipeline for simultaneous object detection,
segmentation, and classification on the SUT-Crack dataset using patch-based learning.

Features:
    - Multi-task model training with shared backbone
    - Patch-based data loading for large images
    - Adaptive loss weighting and monitoring
    - Comprehensive evaluation and visualization
    - Model checkpointing and saving
    - Progress tracking and logging

Usage:
    python train.py --data-dir /path/to/SUT-Crack \
                             --tasks detection segmentation classification \
                             --scale n --epochs 100 --batch-size 16

File: src/train/yolo12_pavement_defect/train.py
"""


import os
import sys
import keras
import json
import argparse
import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.datasets.sut import SUTDataset
from dl_techniques.models.yolo12_multitask import create_yolov12_multitask
from dl_techniques.losses.yolo12_multitask_loss import create_multitask_loss

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


class MultiTaskMetrics:
    """Custom metrics tracking for multi-task training."""

    def __init__(self, tasks: List[str]):
        self.tasks = tasks
        self.reset_states()
        self.metrics = {}
        self.total_loss = 0.0
        self.total_count = 0

    def reset_states(self):
        """Reset all metric states."""
        self.metrics = {task: {'loss': 0.0, 'count': 0} for task in self.tasks}
        self.total_loss = 0.0
        self.total_count = 0

    def update_state(self, losses: Dict[str, float]):
        """Update metric states with batch losses."""
        for task in self.tasks:
            if f'{task}_loss' in losses:
                self.metrics[task]['loss'] += losses[f'{task}_loss']
                self.metrics[task]['count'] += 1

        if 'total_loss' in losses:
            self.total_loss += losses['total_loss']
            self.total_count += 1

    def result(self) -> Dict[str, float]:
        """Get current metric results."""
        results = {}

        for task in self.tasks:
            if self.metrics[task]['count'] > 0:
                results[f'{task}_loss'] = self.metrics[task]['loss'] / self.metrics[task]['count']

        if self.total_count > 0:
            results['total_loss'] = self.total_loss / self.total_count

        return results

# ---------------------------------------------------------------------


class MultiTaskTrainingCallback(keras.callbacks.Callback):
    """Custom callback for multi-task training monitoring."""

    def __init__(self, validation_dataset=None, validation_steps=None, log_dir=None):
        super().__init__()
        self.validation_dataset = validation_dataset
        self.validation_steps = validation_steps
        self.log_dir = log_dir
        self.epoch_losses = []

    def on_epoch_end(self, epoch, logs=None):
        """Log multi-task losses at epoch end."""
        if logs is None:
            logs = {}

        # Extract and log individual task losses if available
        if hasattr(self.model.loss, 'individual_losses'):
            individual_losses = self.model.loss.individual_losses
            for task, loss in individual_losses.items():
                loss_value = float(loss.numpy()) if hasattr(loss, 'numpy') else float(loss)
                logs[f'{task}_loss'] = loss_value

        # Log task weights if using uncertainty weighting
        if hasattr(self.model.loss, 'get_task_weights'):
            task_weights = self.model.loss.get_task_weights()
            for task, weight in task_weights.items():
                logs[f'{task}_weight'] = weight

        # Store epoch losses for plotting
        self.epoch_losses.append(logs.copy())

        # Log learning rate
        lr = float(self.model.optimizer.learning_rate)
        logs['learning_rate'] = lr

        logger.info(f"Epoch {epoch + 1} - Loss: {logs.get('loss', 0):.4f}")

# ---------------------------------------------------------------------

def create_dataset_splits(
    data_dir: str,
    patch_size: int = 256,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[SUTDataset, SUTDataset, SUTDataset]:
    """
    Create train/validation/test dataset splits.

    Args:
        data_dir: Path to SUT-Crack dataset.
        patch_size: Size of patches to extract.
        train_ratio: Ratio of data for training.
        val_ratio: Ratio of data for validation.
        test_ratio: Ratio of data for testing.
        random_seed: Random seed for reproducible splits.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    # Create full dataset to get annotations
    full_dataset = SUTDataset(
        data_dir=data_dir,
        patch_size=patch_size,
        patches_per_image=1,  # Just for getting annotations
        include_segmentation=True
    )

    # Split annotations
    np.random.seed(random_seed)
    annotations = full_dataset.annotations.copy()
    np.random.shuffle(annotations)

    n_total = len(annotations)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_annotations = annotations[:n_train]
    val_annotations = annotations[n_train:n_train + n_val]
    test_annotations = annotations[n_train + n_val:]

    logger.info(f"Dataset splits - Train: {len(train_annotations)}, "
               f"Val: {len(val_annotations)}, Test: {len(test_annotations)}")

    # Create dataset objects with split annotations
    train_dataset = SUTDataset(
        data_dir=data_dir,
        patch_size=patch_size,
        patches_per_image=16,  # More patches for training
        positive_ratio=0.7,
        include_segmentation=True
    )
    train_dataset.annotations = train_annotations

    val_dataset = SUTDataset(
        data_dir=data_dir,
        patch_size=patch_size,
        patches_per_image=8,  # Fewer patches for validation
        positive_ratio=0.5,
        include_segmentation=True
    )
    val_dataset.annotations = val_annotations

    test_dataset = SUTDataset(
        data_dir=data_dir,
        patch_size=patch_size,
        patches_per_image=8,
        positive_ratio=0.5,
        include_segmentation=True
    )
    test_dataset.annotations = test_annotations

    return train_dataset, val_dataset, test_dataset

# ---------------------------------------------------------------------


def create_model_and_loss(
    tasks: List[str],
    patch_size: int,
    scale: str,
    num_classes: int = 1,
    use_uncertainty_weighting: bool = False
) -> Tuple[keras.Model, keras.losses.Loss]:
    """
    Create multitask model and loss function.

    Args:
        tasks: List of tasks to enable.
        patch_size: Input patch size.
        scale: Model scale.
        num_classes: Number of classes.
        use_uncertainty_weighting: Whether to use uncertainty-based loss weighting.

    Returns:
        Tuple of (model, loss_function).
    """
    # Create multitask model
    model = create_yolov12_multitask(
        num_classes=num_classes,
        input_shape=(patch_size, patch_size, 3),
        scale=scale,
        tasks=tasks
    )

    # Create multitask loss
    loss_fn = create_multitask_loss(
        tasks=tasks,
        patch_size=patch_size,
        use_uncertainty_weighting=use_uncertainty_weighting
    )

    return model, loss_fn

# ---------------------------------------------------------------------


def train_model(args: argparse.Namespace) -> None:
    """Main training function."""
    logger.info("Starting YOLOv12 Multi-Task training")
    logger.info(f"Arguments: {vars(args)}")

    # Setup GPU
    setup_gpu()

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tasks_str = "_".join(args.tasks)
    results_dir = f"results/yolov12_multitask_{args.scale}_{tasks_str}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {results_dir}")

    # Create dataset splits
    logger.info("Creating dataset splits...")
    train_dataset, val_dataset, test_dataset = create_dataset_splits(
        data_dir=args.data_dir,
        patch_size=args.patch_size,
        random_seed=args.random_seed
    )

    # Print dataset information
    train_info = train_dataset.get_dataset_info()
    val_info = val_dataset.get_dataset_info()
    test_info = test_dataset.get_dataset_info()

    logger.info(f"Train dataset: {train_info}")
    logger.info(f"Validation dataset: {val_info}")
    logger.info(f"Test dataset: {test_info}")

    # Create TensorFlow datasets
    train_tf_dataset = train_dataset.create_tf_dataset(
        batch_size=args.batch_size,
        shuffle=True,
        repeat=True
    )

    val_tf_dataset = val_dataset.create_tf_dataset(
        batch_size=args.batch_size,
        shuffle=False,
        repeat=True
    )

    # Calculate steps
    steps_per_epoch = train_info['total_patches_per_epoch'] // args.batch_size
    validation_steps = val_info['total_patches_per_epoch'] // args.batch_size

    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Validation steps: {validation_steps}")

    # Create model and loss
    logger.info("Creating model and loss function...")
    model, loss_fn = create_model_and_loss(
        tasks=args.tasks,
        patch_size=args.patch_size,
        scale=args.scale,
        use_uncertainty_weighting=args.uncertainty_weighting
    )

    # Build model
    sample_input = tf.zeros((1, args.patch_size, args.patch_size, 3))
    _ = model(sample_input, training=False)

    # Print model summary
    logger.info("Model architecture:")
    model.summary()

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

    # Test compilation with sample data
    logger.info("Testing model compilation...")
    try:
        sample_batch = next(iter(train_tf_dataset))
        test_loss = model.evaluate(sample_batch[0], sample_batch[1], steps=1, verbose=0)
        logger.info(f"✓ Model compilation test successful: {test_loss:.6f}")
    except Exception as e:
        logger.error(f"✗ Model compilation test failed: {e}")
        if not args.run_eagerly:
            logger.info("Switching to eager execution mode...")
            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                run_eagerly=True
            )

    # Create callbacks
    callbacks = create_callbacks(
        results_dir=results_dir,
        monitor='val_loss',
        patience=args.patience,
        val_dataset=val_tf_dataset,
        validation_steps=validation_steps
    )

    # Train model
    logger.info("Starting training...")
    history = model.fit(
        train_tf_dataset,
        validation_data=val_tf_dataset,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    test_results = None
    if args.evaluate and test_dataset.annotations:
        logger.info("Evaluating on test set...")
        test_tf_dataset = test_dataset.create_tf_dataset(
            batch_size=args.batch_size,
            shuffle=False,
            repeat=False
        )

        test_steps = test_info['total_patches_per_epoch'] // args.batch_size
        test_loss = model.evaluate(test_tf_dataset, steps=test_steps, verbose=1)
        test_results = {'test_loss': float(test_loss)}

        logger.info(f"Test Results: Test Loss: {test_loss:.6f}")

    # Generate visualizations and save results
    save_training_results(
        model=model,
        history=history,
        results_dir=results_dir,
        args=args,
        train_info=train_info,
        val_info=val_info,
        test_info=test_info,
        test_results=test_results
    )

    logger.info("Training completed successfully!")


def create_callbacks(
    results_dir: str,
    monitor: str = 'val_loss',
    patience: int = 50,
    val_dataset=None,
    validation_steps=None
) -> List[keras.callbacks.Callback]:
    """Create training callbacks."""
    callbacks = [
        # Multi-task callback
        MultiTaskTrainingCallback(
            validation_dataset=val_dataset,
            validation_steps=validation_steps,
            log_dir=os.path.join(results_dir, 'logs')
        ),

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
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),

        # Learning rate reduction
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=patience // 3,
            min_lr=1e-7,
            verbose=1
        ),

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


def save_training_results(
    model: keras.Model,
    history: keras.callbacks.History,
    results_dir: str,
    args: argparse.Namespace,
    train_info: Dict,
    val_info: Dict,
    test_info: Dict,
    test_results: Optional[Dict] = None
):
    """Save comprehensive training results and visualizations."""
    logger.info("Saving training results and visualizations...")

    # Create visualizations directory
    viz_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # Plot training history
    plot_training_history(history, viz_dir)

    # Save model architecture diagram
    try:
        keras.utils.plot_model(
            model,
            to_file=os.path.join(viz_dir, 'model_architecture.png'),
            show_shapes=True,
            show_layer_names=True,
            expand_nested=False,
            dpi=150
        )
    except Exception as e:
        logger.warning(f"Failed to save model architecture: {e}")

    # Save model summary
    with open(os.path.join(results_dir, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Save configuration
    config = {
        'training_args': vars(args),
        'model_config': {
            'tasks': args.tasks,
            'scale': args.scale,
            'patch_size': args.patch_size,
            'total_parameters': model.count_params(),
        },
        'dataset_info': {
            'train': train_info,
            'validation': val_info,
            'test': test_info
        },
        'training_results': {
            'epochs_completed': len(history.history['loss']),
            'final_training_loss': float(history.history['loss'][-1]),
            'final_validation_loss': float(history.history.get('val_loss', [0])[-1]) if 'val_loss' in history.history else None,
            'best_validation_loss': float(min(history.history.get('val_loss', [float('inf')]))) if 'val_loss' in history.history else None,
        }
    }

    if test_results:
        config['test_results'] = test_results

    with open(os.path.join(results_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Save final model
    final_model_path = os.path.join(results_dir, 'final_model.keras')
    model.save(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")

    # Create training summary
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"YOLOv12 Multi-Task Training Summary\n")
        f.write(f"===================================\n")
        f.write(f"Tasks: {', '.join(args.tasks)}\n")
        f.write(f"Model Scale: {args.scale}\n")
        f.write(f"Patch Size: {args.patch_size}x{args.patch_size}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Optimizer: {args.optimizer}\n")
        f.write(f"Total Epochs: {len(history.history['loss'])}\n")
        f.write(f"Final Training Loss: {history.history['loss'][-1]:.6f}\n")

        if 'val_loss' in history.history:
            f.write(f"Final Validation Loss: {history.history['val_loss'][-1]:.6f}\n")
            f.write(f"Best Validation Loss: {min(history.history['val_loss']):.6f}\n")

        if test_results:
            f.write(f"\nTest Results:\n")
            for metric, value in test_results.items():
                f.write(f"  {metric}: {value:.6f}\n")

        f.write(f"\nModel Statistics:\n")
        f.write(f"  Total Parameters: {model.count_params():,}\n")
        f.write(f"  Trainable Parameters: {model.trainable_params:,}\n")


def plot_training_history(history: keras.callbacks.History, save_dir: str):
    """Plot comprehensive training history."""
    history_dict = history.history
    epochs = range(1, len(history_dict['loss']) + 1)

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Plot total loss
    axes[0, 0].plot(epochs, history_dict['loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history_dict:
        axes[0, 0].plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot task-specific losses
    task_losses = ['detection_loss', 'segmentation_loss', 'classification_loss']
    colors = ['green', 'orange', 'purple']

    for i, (task_loss, color) in enumerate(zip(task_losses, colors)):
        if task_loss in history_dict:
            axes[0, 1].plot(epochs, history_dict[task_loss], color=color,
                           label=task_loss.replace('_', ' ').title(), linewidth=2)

    if any(task_loss in history_dict for task_loss in task_losses):
        axes[0, 1].set_title('Task-Specific Losses', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Plot learning rate
    if 'learning_rate' in history_dict:
        axes[0, 2].plot(epochs, history_dict['learning_rate'], 'orange', linewidth=2)
        axes[0, 2].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)

    # Plot task weights (if using uncertainty weighting)
    task_weights = ['detection_weight', 'segmentation_weight', 'classification_weight']
    weight_colors = ['green', 'orange', 'purple']

    weights_plotted = False
    for task_weight, color in zip(task_weights, weight_colors):
        if task_weight in history_dict:
            axes[1, 0].plot(epochs, history_dict[task_weight], color=color,
                           label=task_weight.replace('_', ' ').title(), linewidth=2)
            weights_plotted = True

    if weights_plotted:
        axes[1, 0].set_title('Task Weights (Uncertainty)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Weight')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Task Weights\n(Fixed Weighting)',
                       ha='center', va='center', transform=axes[1, 0].transAxes)

    # Plot loss moving average
    if len(epochs) > 5:
        window = max(3, len(epochs) // 10)
        loss_ma = pd.Series(history_dict['loss']).rolling(window=window, center=True).mean()

        axes[1, 1].plot(epochs, history_dict['loss'], alpha=0.3, color='blue', label='Raw Loss')
        axes[1, 1].plot(epochs, loss_ma, linewidth=2, color='blue', label=f'Loss MA-{window}')

        if 'val_loss' in history_dict:
            val_loss_ma = pd.Series(history_dict['val_loss']).rolling(window=window, center=True).mean()
            axes[1, 1].plot(epochs, history_dict['val_loss'], alpha=0.3, color='red', label='Raw Val Loss')
            axes[1, 1].plot(epochs, val_loss_ma, linewidth=2, color='red', label=f'Val Loss MA-{window}')

        axes[1, 1].set_title('Training Stability', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    # Plot loss distribution
    final_losses = []
    for task_loss in task_losses:
        if task_loss in history_dict:
            final_losses.append((task_loss.replace('_loss', '').title(), history_dict[task_loss][-1]))

    if final_losses:
        tasks, losses = zip(*final_losses)
        axes[1, 2].bar(tasks, losses, color=['green', 'orange', 'purple'][:len(tasks)])
        axes[1, 2].set_title('Final Task Losses', fontsize=14, fontweight='bold')
        axes[1, 2].set_ylabel('Loss Value')
        axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Train YOLOv12 Multi-Task Model')

    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to SUT-Crack dataset directory')
    parser.add_argument('--tasks', nargs='+',
                       choices=['detection', 'segmentation', 'classification'],
                       default=['detection', 'segmentation', 'classification'],
                       help='Tasks to enable for training')

    # Model arguments
    parser.add_argument('--scale', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='Model scale')
    parser.add_argument('--patch-size', type=int, default=256,
                       help='Input patch size')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer type')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')

    # Loss arguments
    parser.add_argument('--uncertainty-weighting', action='store_true',
                       help='Use uncertainty-based task weighting')

    # Control arguments
    parser.add_argument('--no-evaluate', action='store_true',
                       help='Skip evaluation on test set')
    parser.add_argument('--run-eagerly', action='store_true',
                       help='Run model in eager mode')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')

    # Parse arguments
    args = parser.parse_args()
    args.evaluate = not args.no_evaluate

    # Validate arguments
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory not found: {args.data_dir}")

    if not args.tasks:
        raise ValueError("At least one task must be specified")

    # Set random seeds
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

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

# ---------------------------------------------------------------------


if __name__ == '__main__':
    main()