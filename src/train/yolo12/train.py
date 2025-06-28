"""
Enhanced YOLOv12 Object Detection Training Script using Multi-Task Architecture.

This script demonstrates how to train a YOLOv12 object detection model using the
multi-task architecture with only the detection task enabled. This approach provides
the benefits of the modular multi-task system while focusing on detection only.

Features:
    - Uses YOLOv12MultiTask model with detection-only configuration
    - TaskType enum-based configuration for type safety
    - Native Keras loss components with proper integration
    - Complete training pipeline with comprehensive monitoring
    - Professional visualizations and model analysis
    - Dummy COCO-style dataset generation for demonstration
    - Robust error handling and logging
    - Model saving in multiple formats

Usage:
    python train_detection.py --scale n --epochs 100 --batch-size 16 --img-size 640

File: src/train/yolo12/train.py
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
from typing import Tuple, Dict, Optional, List
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.vision_task_types import TaskType
from dl_techniques.models.yolo12_multitask import create_yolov12_multitask
from dl_techniques.losses.yolo12_multitask_loss import create_yolov12_multitask_loss

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

def create_dummy_coco_dataset(
    num_samples: int,
    img_size: int,
    num_classes: int = 80,
    max_boxes: int = 20,
    min_boxes: int = 1
) -> tf.data.Dataset:
    """
    Create a dummy COCO-style dataset for object detection training.

    Args:
        num_samples: Number of samples to generate.
        img_size: Input image size.
        num_classes: Number of object classes.
        max_boxes: Maximum number of boxes per image.
        min_boxes: Minimum number of boxes per image.

    Returns:
        TensorFlow dataset with (image, labels) pairs.
        Labels format: (class_id, x1, y1, x2, y2) in absolute coordinates.
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

# ---------------------------------------------------------------------

def create_detection_model_and_loss(
    scale: str,
    num_classes: int,
    img_size: int,
    reg_max: int = 16,
    **loss_kwargs
) -> Tuple[keras.Model, keras.losses.Loss]:
    """
    Create YOLOv12 detection model and loss using multi-task architecture.

    Args:
        scale: Model scale ('n', 's', 'm', 'l', 'x').
        num_classes: Number of object classes.
        img_size: Input image size.
        reg_max: Maximum value for DFL regression.
        **loss_kwargs: Additional loss function arguments.

    Returns:
        Tuple of (model, loss_function).
    """
    logger.info(f"Creating YOLOv12 detection model (scale: {scale}, classes: {num_classes})")

    # Create detection-only multi-task model using TaskType enum
    model = create_yolov12_multitask(
        num_classes=num_classes,
        input_shape=(img_size, img_size, 3),
        scale=scale,
        tasks=TaskType.DETECTION,  # Single task - returns tensor directly
        reg_max=reg_max
    )

    # Create detection-only loss function
    loss_fn = create_yolov12_multitask_loss(
        tasks=TaskType.DETECTION,  # Single task configuration
        num_classes=num_classes,
        input_shape=(img_size, img_size),
        reg_max=reg_max,
        **loss_kwargs
    )

    logger.info("✓ Model and loss function created successfully")
    logger.info(f"Model output type: Single tensor (detection-only)")

    return model, loss_fn

# ---------------------------------------------------------------------

def test_model_forward_pass(
    model: keras.Model,
    img_size: int,
    batch_size: int = 2
) -> bool:
    """
    Test model forward pass to ensure proper functionality.

    Args:
        model: YOLOv12 detection model.
        img_size: Input image size.
        batch_size: Test batch size.

    Returns:
        True if test passes, False otherwise.
    """
    try:
        logger.info("Testing model forward pass...")

        # Create dummy input
        dummy_input = tf.random.normal((batch_size, img_size, img_size, 3))

        # Forward pass
        output = model(dummy_input, training=False)

        # Log output information
        logger.info(f"✓ Forward pass successful")
        logger.info(f"  Input shape: {dummy_input.shape}")
        logger.info(f"  Output shape: {output.shape}")
        logger.info(f"  Output type: {type(output)}")

        # Verify output is a tensor (not dict) for single-task
        if isinstance(output, dict):
            logger.warning("⚠ Output is dict - expected tensor for single-task model")
            return False

        return True

    except Exception as e:
        logger.error(f"✗ Forward pass test failed: {e}")
        return False

# ---------------------------------------------------------------------

class EnhancedYOLOv12Callback(keras.callbacks.Callback):
    """Enhanced callback for YOLOv12 detection training monitoring."""

    def __init__(self,
                 loss_fn=None,
                 log_freq: int = 1,
                 save_dir: Optional[str] = None):
        super().__init__()
        self.loss_fn = loss_fn
        self.log_freq = log_freq
        self.save_dir = save_dir
        self.epoch_metrics = []

    def on_epoch_end(self, epoch, logs=None):
        """Enhanced logging for detection training."""
        if logs is None:
            logs = {}

        # Get individual loss components if available
        if self.loss_fn and hasattr(self.loss_fn, 'get_individual_losses'):
            try:
                individual_losses = self.loss_fn.get_individual_losses()
                for loss_name, loss_val in individual_losses.items():
                    logs[f'{loss_name}'] = float(loss_val)
            except Exception as e:
                logger.warning(f"Failed to get individual losses: {e}")

        # Log learning rate
        if hasattr(self.model.optimizer, 'learning_rate'):
            try:
                lr = float(self.model.optimizer.learning_rate)
                logs['learning_rate'] = lr
            except:
                pass

        # Store epoch metrics
        self.epoch_metrics.append(logs.copy())

        # Enhanced logging
        if (epoch + 1) % self.log_freq == 0:
            loss_str = f"Loss: {logs.get('loss', 0):.4f}"

            if 'val_loss' in logs:
                loss_str += f", Val Loss: {logs.get('val_loss', 0):.4f}"

            # Add detection-specific losses if available
            detection_losses = []
            for loss_type in ['detection']:
                if f'{loss_type}_loss' in logs:
                    detection_losses.append(f"{loss_type}: {logs[f'{loss_type}_loss']:.4f}")

            if detection_losses:
                loss_str += f" | {', '.join(detection_losses)}"

            logger.info(f"Epoch {epoch + 1}/{self.params.get('epochs', '?')} - {loss_str}")

    def get_metrics_history(self) -> List[Dict[str, float]]:
        """Get history of all metrics."""
        return self.epoch_metrics

# ---------------------------------------------------------------------

def create_enhanced_callbacks(
    model_name: str,
    results_dir: str,
    loss_fn: keras.losses.Loss,
    monitor: str = 'val_loss',
    patience: int = 50,
    save_best_only: bool = True
) -> List[keras.callbacks.Callback]:
    """
    Create enhanced training callbacks for detection training.

    Args:
        model_name: Name for saved model files.
        results_dir: Results directory path.
        loss_fn: Loss function instance for monitoring.
        monitor: Metric to monitor for callbacks.
        patience: Patience for early stopping.
        save_best_only: Whether to save only the best model.

    Returns:
        List of callbacks.
    """
    callbacks = [
        # Enhanced YOLOv12 callback
        EnhancedYOLOv12Callback(
            loss_fn=loss_fn,
            log_freq=1,
            save_dir=results_dir
        ),

        # Early stopping with improved configuration
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            mode='min',
            verbose=1,
            min_delta=1e-5
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

        # Learning rate reduction with adaptive settings
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if 'val' in monitor else 'loss',
            factor=0.5,
            patience=max(10, patience // 3),
            min_lr=1e-7,
            verbose=1,
            cooldown=2
        ),

        # CSV logging
        keras.callbacks.CSVLogger(
            filename=os.path.join(results_dir, 'training_log.csv'),
            append=False,
            separator=','
        ),

        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(results_dir, 'tensorboard'),
            histogram_freq=0,  # Disabled for performance
            write_graph=True,
            write_images=False,
            update_freq='epoch',
            profile_batch=0
        ),
    ]

    return callbacks

# ---------------------------------------------------------------------

def plot_detection_training_history(
    history: keras.callbacks.History,
    save_dir: str,
    callbacks: Optional[List] = None
) -> None:
    """
    Plot comprehensive training history for detection training.

    Args:
        history: Keras training history.
        save_dir: Directory to save plots.
        callbacks: List of callbacks (for additional metrics).
    """
    history_dict = history.history
    epochs = range(1, len(history_dict['loss']) + 1)

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Plot 1: Total loss
    ax_idx = 0
    axes[ax_idx].plot(epochs, history_dict['loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history_dict:
        axes[ax_idx].plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[ax_idx].set_title('Detection Loss', fontsize=14, fontweight='bold')
    axes[ax_idx].set_xlabel('Epoch')
    axes[ax_idx].set_ylabel('Loss')
    axes[ax_idx].legend()
    axes[ax_idx].grid(True, alpha=0.3)

    # Plot 2: Detection-specific losses
    ax_idx += 1
    detection_losses = ['detection_loss']  # Could be expanded to box_loss, cls_loss, dfl_loss
    detection_plotted = False

    for loss_name in detection_losses:
        if loss_name in history_dict:
            axes[ax_idx].plot(epochs, history_dict[loss_name],
                            color='green', label=f'{loss_name.replace("_", " ").title()}', linewidth=2)
            detection_plotted = True

    if detection_plotted:
        axes[ax_idx].set_title('Detection Loss Components', fontsize=14, fontweight='bold')
        axes[ax_idx].set_xlabel('Epoch')
        axes[ax_idx].set_ylabel('Loss')
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
    else:
        axes[ax_idx].text(0.5, 0.5, 'Detection Loss\nComponents\nNot Available',
                         ha='center', va='center', transform=axes[ax_idx].transAxes)

    # Plot 3: Learning rate
    ax_idx += 1
    if 'learning_rate' in history_dict:
        axes[ax_idx].plot(epochs, history_dict['learning_rate'], 'orange', linewidth=2)
        axes[ax_idx].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[ax_idx].set_xlabel('Epoch')
        axes[ax_idx].set_ylabel('Learning Rate')
        axes[ax_idx].set_yscale('log')
        axes[ax_idx].grid(True, alpha=0.3)
    else:
        axes[ax_idx].text(0.5, 0.5, 'Learning Rate\nNot Tracked',
                         ha='center', va='center', transform=axes[ax_idx].transAxes)

    # Plot 4: Training stability (moving average)
    ax_idx += 1
    if len(epochs) > 5:
        window = max(3, len(epochs) // 10)
        loss_ma = pd.Series(history_dict['loss']).rolling(window=window, center=True).mean()

        axes[ax_idx].plot(epochs, history_dict['loss'], alpha=0.3, color='blue', label='Raw Training Loss')
        axes[ax_idx].plot(epochs, loss_ma, linewidth=2, color='blue', label=f'Smoothed Training Loss (MA-{window})')

        if 'val_loss' in history_dict:
            val_loss_ma = pd.Series(history_dict['val_loss']).rolling(window=window, center=True).mean()
            axes[ax_idx].plot(epochs, history_dict['val_loss'], alpha=0.3, color='red', label='Raw Validation Loss')
            axes[ax_idx].plot(epochs, val_loss_ma, linewidth=2, color='red', label=f'Smoothed Validation Loss (MA-{window})')

        axes[ax_idx].set_title('Training Stability', fontsize=14, fontweight='bold')
        axes[ax_idx].set_xlabel('Epoch')
        axes[ax_idx].set_ylabel('Loss')
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)

    # Plot 5: Loss distribution over time
    ax_idx += 1
    if len(epochs) > 10:
        # Create violin plot of loss distribution over training phases
        early_phase = history_dict['loss'][:len(epochs)//3]
        mid_phase = history_dict['loss'][len(epochs)//3:2*len(epochs)//3]
        late_phase = history_dict['loss'][2*len(epochs)//3:]

        data_to_plot = [early_phase, mid_phase, late_phase]
        labels = ['Early', 'Mid', 'Late']

        axes[ax_idx].boxplot(data_to_plot, labels=labels)
        axes[ax_idx].set_title('Loss Distribution by Phase', fontsize=14, fontweight='bold')
        axes[ax_idx].set_ylabel('Loss')
        axes[ax_idx].grid(True, alpha=0.3)

    # Plot 6: Performance summary
    ax_idx += 1
    if 'val_loss' in history_dict:
        min_val_loss = min(history_dict['val_loss'])
        min_val_epoch = history_dict['val_loss'].index(min_val_loss) + 1
        final_val_loss = history_dict['val_loss'][-1]

        metrics = ['Min Val Loss', 'Final Val Loss', 'Final Train Loss']
        values = [min_val_loss, final_val_loss, history_dict['loss'][-1]]
        colors = ['green', 'orange', 'blue']

        bars = axes[ax_idx].bar(metrics, values, color=colors)
        axes[ax_idx].set_title('Final Performance Metrics', fontsize=14, fontweight='bold')
        axes[ax_idx].set_ylabel('Loss Value')
        axes[ax_idx].tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[ax_idx].text(bar.get_x() + bar.get_width()/2., height,
                            f'{value:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'detection_training_history.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------

def save_comprehensive_results(
    model: keras.Model,
    history: keras.callbacks.History,
    results_dir: str,
    args: argparse.Namespace,
    test_results: Optional[Dict[str, float]] = None,
    callbacks: Optional[List] = None
):
    """Save comprehensive training results and analysis."""
    logger.info("Saving comprehensive training results...")

    # Create directories
    viz_dir = os.path.join(results_dir, 'visualizations')
    model_dir = os.path.join(results_dir, 'models')
    analysis_dir = os.path.join(results_dir, 'analysis')

    for directory in [viz_dir, model_dir, analysis_dir]:
        os.makedirs(directory, exist_ok=True)

    # Plot training history
    plot_detection_training_history(history, viz_dir, callbacks)

    # Save model architecture
    try:
        keras.utils.plot_model(
            model,
            to_file=os.path.join(viz_dir, 'model_architecture.png'),
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=150
        )
    except Exception as e:
        logger.warning(f"Failed to save model architecture plot: {e}")

    # Save model summary
    with open(os.path.join(analysis_dir, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Create comprehensive configuration
    config = {
        'training_args': vars(args),
        'model_config': {
            'architecture': 'YOLOv12MultiTask (Detection Only)',
            'scale': args.scale,
            'num_classes': args.num_classes,
            'input_shape': (args.img_size, args.img_size, 3),
            'total_parameters': model.count_params(),
            'task_configuration': 'Detection Only (Single Task)',
            'output_format': 'Single Tensor'
        },
        'dataset_info': {
            'type': 'Dummy COCO',
            'train_samples': args.train_samples,
            'val_samples': args.val_samples,
            'test_samples': args.test_samples,
            'max_boxes_per_image': args.max_boxes
        },
        'training_results': {
            'epochs_completed': len(history.history['loss']),
            'final_training_loss': float(history.history['loss'][-1]),
            'final_validation_loss': float(history.history.get('val_loss', [0])[-1]) if 'val_loss' in history.history else None,
            'best_validation_loss': float(min(history.history.get('val_loss', [float('inf')]))) if 'val_loss' in history.history else None,
        },
        'loss_configuration': {
            'type': 'YOLOv12MultiTaskLoss (Detection Only)',
            'uses_native_keras_components': True,
            'single_task_mode': True
        }
    }

    if test_results:
        config['test_results'] = test_results

    # Save configuration
    with open(os.path.join(results_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Save models in multiple formats
    try:
        # Save final model in Keras format
        final_model_path = os.path.join(model_dir, 'final_model.keras')
        model.save(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")

        # Export in SavedModel format for deployment
        saved_model_path = os.path.join(model_dir, 'saved_model')
        model.export(saved_model_path)
        logger.info(f"SavedModel exported to: {saved_model_path}")

    except Exception as e:
        logger.error(f"Failed to save model: {e}")

    # Create detailed training summary
    create_detection_training_summary(results_dir, config, history)

    logger.info(f"All results saved to: {results_dir}")

# ---------------------------------------------------------------------

def create_detection_training_summary(
    results_dir: str,
    config: Dict,
    history: keras.callbacks.History
):
    """Create detailed training summary for detection training."""
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write("YOLOv12 Object Detection Training Summary\n")
        f.write("=" * 50 + "\n\n")

        # Architecture details
        f.write("Model Architecture:\n")
        f.write(f"  Type: {config['model_config']['architecture']}\n")
        f.write(f"  Scale: {config['model_config']['scale']}\n")
        f.write(f"  Task Configuration: {config['model_config']['task_configuration']}\n")
        f.write(f"  Output Format: {config['model_config']['output_format']}\n")
        f.write(f"  Total Parameters: {config['model_config']['total_parameters']:,}\n")

        # Training configuration
        f.write("Training Configuration:\n")
        f.write(f"  Input Size: {config['model_config']['input_shape']}\n")
        f.write(f"  Number of Classes: {config['model_config']['num_classes']}\n")
        f.write(f"  Batch Size: {config['training_args']['batch_size']}\n")
        f.write(f"  Learning Rate: {config['training_args']['learning_rate']}\n")
        f.write(f"  Optimizer: {config['training_args']['optimizer']}\n\n")

        # Dataset information
        f.write("Dataset Information:\n")
        f.write(f"  Type: {config['dataset_info']['type']}\n")
        f.write(f"  Training Samples: {config['dataset_info']['train_samples']:,}\n")
        f.write(f"  Validation Samples: {config['dataset_info']['val_samples']:,}\n")
        f.write(f"  Test Samples: {config['dataset_info']['test_samples']:,}\n\n")

        # Training results
        f.write("Training Results:\n")
        f.write(f"  Epochs Completed: {config['training_results']['epochs_completed']}\n")
        f.write(f"  Final Training Loss: {config['training_results']['final_training_loss']:.6f}\n")

        if config['training_results']['final_validation_loss']:
            f.write(f"  Final Validation Loss: {config['training_results']['final_validation_loss']:.6f}\n")
            f.write(f"  Best Validation Loss: {config['training_results']['best_validation_loss']:.6f}\n")

        # Test results
        if 'test_results' in config:
            f.write(f"\nTest Results:\n")
            for metric, value in config['test_results'].items():
                f.write(f"  {metric.replace('_', ' ').title()}: {value:.6f}\n")

# ---------------------------------------------------------------------

def train_detection_model(args: argparse.Namespace) -> None:
    """Main training function for YOLOv12 object detection."""
    logger.info("Starting YOLOv12 Object Detection Training (Multi-Task Architecture)")
    logger.info(f"Arguments: {vars(args)}")

    # Setup GPU
    setup_gpu()

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/yolov12_detection_{args.scale}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {results_dir}")

    # Create datasets
    logger.info("Creating dummy COCO datasets...")
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

    # Calculate training steps
    steps_per_epoch = max(args.train_samples // args.batch_size, 1)
    validation_steps = max(args.val_samples // args.batch_size, 1)

    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Validation steps: {validation_steps}")

    # Prepare datasets
    train_dataset = train_dataset.repeat().batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.repeat().batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # Create model and loss using multi-task architecture
    logger.info("Creating YOLOv12 detection model using multi-task architecture...")
    model, loss_fn = create_detection_model_and_loss(
        scale=args.scale,
        num_classes=args.num_classes,
        img_size=args.img_size,
        reg_max=16,
        # Additional loss parameters
        detection_box_weight=7.5,
        detection_cls_weight=0.5,
        detection_dfl_weight=1.5
    )

    # Test model forward pass
    forward_pass_success = test_model_forward_pass(model, args.img_size, args.batch_size)
    if not forward_pass_success:
        raise RuntimeError("Model forward pass test failed")

    # Print model information
    logger.info("Model architecture summary:")
    model.summary(print_fn=logger.info)
    logger.info(f"Total parameters: {model.count_params():,}")

    # Create optimizer
    if args.optimizer.lower() == 'adamw':
        optimizer = keras.optimizers.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            clipnorm=1.0
        )
    elif args.optimizer.lower() == 'sgd':
        optimizer = keras.optimizers.SGD(
            learning_rate=args.learning_rate,
            momentum=0.9,
            nesterov=True,
            clipnorm=1.0
        )
    else:
        optimizer = keras.optimizers.Adam(
            learning_rate=args.learning_rate,
            clipnorm=1.0
        )

    logger.info(f"Using optimizer: {type(optimizer).__name__}")

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        run_eagerly=args.run_eagerly
    )

    # Test compilation with sample data
    logger.info("Testing model compilation...")
    try:
        sample_batch = next(iter(train_dataset))
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

    # Create enhanced callbacks
    callbacks = create_enhanced_callbacks(
        model_name=f"yolov12_detection_{args.scale}",
        results_dir=results_dir,
        loss_fn=loss_fn,
        monitor='val_loss' if args.validation else 'loss',
        patience=args.patience,
        save_best_only=True
    )

    # Train model
    logger.info("Starting training...")
    try:
        validation_data = val_dataset if args.validation else None
        val_steps = validation_steps if args.validation else None

        history = model.fit(
            train_dataset,
            validation_data=validation_data,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")

        # Save partial results if available
        if 'history' in locals():
            logger.info("Saving partial training results...")
            save_comprehensive_results(
                model=model,
                history=history,
                results_dir=results_dir,
                args=args,
                callbacks=callbacks
            )
        raise

    # Evaluate on test set
    test_results = None
    if args.evaluate:
        logger.info("Evaluating on test set...")
        try:
            test_steps = max(args.test_samples // args.batch_size, 1)
            test_loss = model.evaluate(test_dataset, steps=test_steps, verbose=1)
            test_results = {'test_loss': float(test_loss)}

            logger.info(f"✓ Test Results: Test Loss: {test_loss:.6f}")
        except Exception as e:
            logger.error(f"Test evaluation failed: {e}")

    # Save comprehensive results
    save_comprehensive_results(
        model=model,
        history=history,
        results_dir=results_dir,
        args=args,
        test_results=test_results,
        callbacks=callbacks
    )

    logger.info("YOLOv12 Object Detection training completed successfully! 🎉")

# ---------------------------------------------------------------------

def main():
    """Main function with enhanced argument parsing."""
    parser = argparse.ArgumentParser(
        description='Train YOLOv12 Object Detection Model using Multi-Task Architecture',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument('--scale', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model scale')
    parser.add_argument('--num-classes', type=int, default=80,
                       help='Number of object classes')
    parser.add_argument('--img-size', type=int, default=320,
                       help='Input image size')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay for AdamW optimizer')
    parser.add_argument('--optimizer', type=str, default='sgd',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer type')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')

    # Dataset arguments
    parser.add_argument('--train-samples', type=int, default=1000,
                       help='Number of training samples')
    parser.add_argument('--val-samples', type=int, default=200,
                       help='Number of validation samples')
    parser.add_argument('--test-samples', type=int, default=100,
                       help='Number of test samples')
    parser.add_argument('--max-boxes', type=int, default=20,
                       help='Maximum boxes per image')

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

    # Log configuration
    logger.info("YOLOv12 Object Detection Training Configuration:")
    logger.info(f"  Architecture: Multi-Task (Detection Only)")
    logger.info(f"  Model Scale: {args.scale}")
    logger.info(f"  Image Size: {args.img_size}x{args.img_size}")
    logger.info(f"  Number of Classes: {args.num_classes}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Learning Rate: {args.learning_rate}")
    logger.info(f"  Optimizer: {args.optimizer}")

    # Start training with comprehensive error handling
    try:
        train_detection_model(args)
        logger.info("Training completed successfully! 🎉")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

# ---------------------------------------------------------------------

if __name__ == '__main__':
    main()