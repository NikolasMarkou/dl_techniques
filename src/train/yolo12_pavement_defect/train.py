"""
Comprehensive Training Script for YOLOv12 Multi-Task Model

This script provides complete training pipeline for simultaneous object detection,
segmentation, and classification on crack detection datasets using patch-based learning.

Features:
    - Multi-task model training with shared backbone using Named Outputs (Functional API)
    - TaskType enum-based configuration for type safety
    - Native Keras loss components with uncertainty weighting
    - Patch-based data loading for large images
    - Comprehensive evaluation and visualization
    - Model checkpointing and saving with proper serialization
    - Progress tracking and logging

Usage:
    python train.py --data-dir /path/to/dataset \
                    --tasks detection segmentation classification \
                    --scale n --epochs 100 --batch-size 16
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
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.datasets.sut import SUTDataset
from dl_techniques.models.yolo12_multitask import create_yolov12_multitask
from dl_techniques.utils.vision_task_types import (
    TaskType,
    TaskConfiguration,
    parse_task_list
)
from dl_techniques.losses.yolo12_multitask_loss import (
    create_yolov12_multitask_loss,
)

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

def parse_tasks(task_strings: List[str]) -> TaskConfiguration:
    """
    Parse task strings into TaskConfiguration.

    Args:
        task_strings: List of task names as strings.

    Returns:
        TaskConfiguration instance.
    """
    try:
        task_config = parse_task_list(task_strings)
        enabled_tasks = task_config.get_task_names()
        logger.info(f"Parsed tasks: {enabled_tasks}")
        return task_config
    except ValueError as e:
        logger.error(f"Invalid task configuration: {e}")
        logger.info(f"Valid tasks are: {[t.value for t in TaskType.all_tasks()]}")
        raise

# ---------------------------------------------------------------------

class EnhancedMultiTaskCallback(keras.callbacks.Callback):
    """Enhanced callback for multitask training monitoring with Named Outputs support."""

    def __init__(self,
                 loss_fn=None,
                 validation_dataset=None,
                 validation_steps=None,
                 log_dir=None,
                 log_freq: int = 1):
        super().__init__()
        self.loss_fn = loss_fn
        self.validation_dataset = validation_dataset
        self.validation_steps = validation_steps
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.epoch_losses = []
        self.task_weight_history = []

    def on_epoch_end(self, epoch, logs=None):
        """Enhanced logging for multi-task training."""
        if logs is None:
            logs = {}

        # Log task weights if using uncertainty weighting
        if self.loss_fn and hasattr(self.loss_fn, 'get_task_weights'):
            try:
                task_weights = self.loss_fn.get_task_weights()
                for task, weight in task_weights.items():
                    logs[f'{task}_weight'] = float(weight)
                self.task_weight_history.append(task_weights.copy())
            except Exception as e:
                logger.warning(f"Failed to get task weights: {e}")

        # Log individual task losses if available
        if self.loss_fn and hasattr(self.loss_fn, 'get_individual_losses'):
            try:
                individual_losses = self.loss_fn.get_individual_losses()
                for task, loss_val in individual_losses.items():
                    logs[f'{task}_loss'] = float(loss_val)
            except Exception as e:
                logger.warning(f"Failed to get individual losses: {e}")

        # Store epoch information
        self.epoch_losses.append(logs.copy())

        # Log learning rate
        if hasattr(self.model.optimizer, 'learning_rate'):
            try:
                lr = float(self.model.optimizer.learning_rate)
                logs['learning_rate'] = lr
            except:
                pass

        # Enhanced logging every log_freq epochs
        if (epoch + 1) % self.log_freq == 0:
            loss_str = f"Loss: {logs.get('loss', 0):.4f}"

            if 'val_loss' in logs:
                loss_str += f", Val Loss: {logs.get('val_loss', 0):.4f}"

            # Add task-specific losses if available
            task_losses = []
            for task in ['detection', 'segmentation', 'classification']:
                if f'{task}_loss' in logs:
                    task_losses.append(f"{task}: {logs[f'{task}_loss']:.4f}")

            if task_losses:
                loss_str += f" | {', '.join(task_losses)}"

            logger.info(f"Epoch {epoch + 1}/{self.params.get('epochs', '?')} - {loss_str}")

    def get_task_weight_history(self) -> List[Dict[str, float]]:
        """Get history of task weights over training."""
        return self.task_weight_history

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
        data_dir: Path to dataset.
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
    task_config: TaskConfiguration,
    patch_size: int,
    scale: str,
    num_classes: int = 1,
    use_uncertainty_weighting: bool = False,
    **loss_kwargs
) -> Tuple[keras.Model, keras.losses.Loss]:
    """
    Create multitask model and loss function with TaskType enum support.

    Args:
        task_config: TaskConfiguration instance.
        patch_size: Input patch size.
        scale: Model scale.
        num_classes: Number of classes.
        use_uncertainty_weighting: Whether to use uncertainty-based loss weighting.
        **loss_kwargs: Additional arguments for loss function.

    Returns:
        Tuple of (model, loss_function).
    """
    logger.info(f"Creating model with tasks: {task_config.get_task_names()}")

    # Create multitask model using TaskConfiguration
    model = create_yolov12_multitask(
        num_classes=num_classes,
        input_shape=(patch_size, patch_size, 3),
        scale=scale,
        tasks=task_config  # Pass TaskConfiguration directly
    )

    # Create multitask loss using TaskConfiguration
    loss_fn = create_yolov12_multitask_loss(
        tasks=task_config,  # Pass TaskConfiguration directly
        num_classes=num_classes,
        input_shape=(patch_size, patch_size),
        use_uncertainty_weighting=use_uncertainty_weighting,
        **loss_kwargs
    )

    return model, loss_fn

# ---------------------------------------------------------------------

def test_model_compilation(model: keras.Model,
                          train_dataset,
                          task_config: TaskConfiguration,
                          run_eagerly: bool = False) -> bool:
    """
    Test model compilation with sample data to ensure compatibility.

    Args:
        model: Compiled Keras model.
        train_dataset: Training dataset.
        task_config: Task configuration.
        run_eagerly: Whether to run in eager mode.

    Returns:
        True if compilation test succeeds, False otherwise.
    """
    logger.info("Testing model compilation...")

    try:
        # Get a sample batch
        sample_batch = next(iter(train_dataset))
        sample_x, sample_y = sample_batch

        # Test forward pass
        predictions = model(sample_x, training=False)

        # Log prediction format
        if isinstance(predictions, dict):
            pred_info = {k: v.shape for k, v in predictions.items()}
            logger.info(f"Model outputs (dict): {pred_info}")
        else:
            logger.info(f"Model output (tensor): {predictions.shape}")

        # Test loss computation
        loss_value = model.evaluate(sample_x, sample_y, steps=1, verbose=0)
        logger.info(f"✓ Model compilation test successful - Loss: {loss_value:.6f}")

        return True

    except Exception as e:
        logger.error(f"✗ Model compilation test failed: {e}")

        if not run_eagerly:
            logger.info("Suggestion: Try running with --run-eagerly flag")

        return False

# ---------------------------------------------------------------------

def create_callbacks(
    results_dir: str,
    loss_fn: keras.losses.Loss,
    task_config: TaskConfiguration,
    monitor: str = 'val_loss',
    patience: int = 20,
    val_dataset=None,
    validation_steps=None
) -> List[keras.callbacks.Callback]:
    """Create enhanced training callbacks with multi-task support."""

    callbacks = [
        # Enhanced multi-task callback
        EnhancedMultiTaskCallback(
            loss_fn=loss_fn,
            validation_dataset=val_dataset,
            validation_steps=validation_steps,
            log_dir=os.path.join(results_dir, 'logs'),
            log_freq=1
        ),

        # Early stopping with improved monitoring
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            mode='min',
            verbose=1,
            min_delta=1e-4
        ),

        # Model checkpointing with proper naming
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(results_dir, 'best_model.keras'),
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),

        # Learning rate reduction with adaptive settings
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=max(patience // 3, 5),
            min_lr=1e-7,
            verbose=1,
            cooldown=2
        ),

        # CSV logging with enhanced metrics
        keras.callbacks.CSVLogger(
            filename=os.path.join(results_dir, 'training_log.csv'),
            append=False,
            separator=','
        ),

        # TensorBoard logging with proper configuration
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(results_dir, 'tensorboard'),
            histogram_freq=0,  # Disabled for performance
            write_graph=True,
            write_images=False,
            update_freq='epoch',
            profile_batch=0  # Disabled for performance
        ),
    ]

    # Add task-specific callbacks if using uncertainty weighting
    if hasattr(loss_fn, 'use_uncertainty_weighting') and loss_fn.use_uncertainty_weighting:
        logger.info("Added uncertainty weighting monitoring")

    return callbacks

# ---------------------------------------------------------------------

def train_model(args: argparse.Namespace) -> None:
    """Enhanced main training function."""
    logger.info("Starting YOLOv12 Multi-Task training with Named Outputs")
    logger.info(f"Arguments: {vars(args)}")

    # Setup GPU
    setup_gpu()

    # Parse tasks into TaskConfiguration
    task_config = parse_tasks(args.tasks)

    # Create results directory with task information
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tasks_str = "_".join(task_config.get_task_names())
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
    steps_per_epoch = max(train_info['total_patches_per_epoch'] // args.batch_size, 1)
    validation_steps = max(val_info['total_patches_per_epoch'] // args.batch_size, 1)

    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Validation steps: {validation_steps}")

    # Create model and loss with enhanced configuration
    logger.info("Creating model and loss function...")
    model, loss_fn = create_model_and_loss(
        task_config=task_config,
        patch_size=args.patch_size,
        scale=args.scale,
        num_classes=1,  # Binary classification for crack detection
        use_uncertainty_weighting=args.uncertainty_weighting,
        # Additional loss parameters
        detection_weight=args.detection_weight,
        segmentation_weight=args.segmentation_weight,
        classification_weight=args.classification_weight
    )

    # Build model with proper input shape
    sample_input = tf.zeros((1, args.patch_size, args.patch_size, 3))
    _ = model(sample_input, training=False)

    # Print model information
    logger.info("Model architecture summary:")
    model.summary(print_fn=logger.info)
    logger.info(f"Total parameters: {model.count_params():,}")

    # Create optimizer with enhanced configuration
    if args.optimizer.lower() == 'adamw':
        optimizer = keras.optimizers.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            clipnorm=1.0  # Gradient clipping for stability
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

    # Test compilation
    compilation_success = test_model_compilation(
        model=model,
        train_dataset=train_tf_dataset,
        task_config=task_config,
        run_eagerly=args.run_eagerly
    )

    if not compilation_success and not args.run_eagerly:
        logger.warning("Compilation test failed, switching to eager execution...")
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            run_eagerly=True
        )

    # Create enhanced callbacks
    callbacks = create_callbacks(
        results_dir=results_dir,
        loss_fn=loss_fn,
        task_config=task_config,
        monitor='val_loss',
        patience=args.patience,
        val_dataset=val_tf_dataset,
        validation_steps=validation_steps
    )

    # Train model with enhanced error handling
    logger.info("Starting training...")
    try:
        history = model.fit(
            train_tf_dataset,
            validation_data=val_tf_dataset,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")

        # Save partial results if training was interrupted
        if 'history' in locals():
            logger.info("Saving partial training results...")
            save_training_results(
                model=model,
                history=history,
                results_dir=results_dir,
                args=args,
                task_config=task_config,
                train_info=train_info,
                val_info=val_info,
                test_info=test_info,
                callbacks=callbacks
            )
        raise

    # Evaluate on test set
    test_results = None
    if args.evaluate and test_dataset.annotations:
        logger.info("Evaluating on test set...")
        try:
            test_tf_dataset = test_dataset.create_tf_dataset(
                batch_size=args.batch_size,
                shuffle=False,
                repeat=False
            )

            test_steps = max(test_info['total_patches_per_epoch'] // args.batch_size, 1)
            test_loss = model.evaluate(test_tf_dataset, steps=test_steps, verbose=1)
            test_results = {'test_loss': float(test_loss)}

            logger.info(f"Test Results: Test Loss: {test_loss:.6f}")
        except Exception as e:
            logger.error(f"Test evaluation failed: {e}")

    # Generate comprehensive results
    save_training_results(
        model=model,
        history=history,
        results_dir=results_dir,
        args=args,
        task_config=task_config,
        train_info=train_info,
        val_info=val_info,
        test_info=test_info,
        test_results=test_results,
        callbacks=callbacks
    )

    logger.info("Training completed successfully!")

# ---------------------------------------------------------------------

def save_training_results(
    model: keras.Model,
    history: keras.callbacks.History,
    results_dir: str,
    args: argparse.Namespace,
    task_config: TaskConfiguration,
    train_info: Dict,
    val_info: Dict,
    test_info: Dict,
    test_results: Optional[Dict] = None,
    callbacks: Optional[List] = None
):
    """Save comprehensive training results with enhanced TaskType support."""
    logger.info("Saving training results and visualizations...")

    # Create directories
    viz_dir = os.path.join(results_dir, 'visualizations')
    model_dir = os.path.join(results_dir, 'models')
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Plot enhanced training history
    plot_enhanced_training_history(history, viz_dir, task_config)

    # Save model architecture
    try:
        keras.utils.plot_model(
            model,
            to_file=os.path.join(viz_dir, 'model_architecture.png'),
            show_shapes=True,
            show_layer_names=True,
            expand_nested=False,
            dpi=150,
            rankdir='TB'
        )
    except Exception as e:
        logger.warning(f"Failed to save model architecture plot: {e}")

    # Save comprehensive configuration
    config = {
        'training_args': vars(args),
        'task_configuration': {
            'enabled_tasks': task_config.get_task_names(),
            'is_single_task': task_config.is_single_task(),
            'is_multi_task': task_config.is_multi_task()
        },
        'model_config': {
            'scale': args.scale,
            'patch_size': args.patch_size,
            'total_parameters': model.count_params(),
            'trainable_parameters': sum([keras.backend.count_params(w) for w in model.trainable_weights]),
            'model_type': 'Named Outputs (Functional API)'
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
        },
        'loss_configuration': {
            'type': 'YOLOv12MultiTaskLoss',
            'uses_uncertainty_weighting': args.uncertainty_weighting,
            'task_weights': {
                'detection': args.detection_weight,
                'segmentation': args.segmentation_weight,
                'classification': args.classification_weight
            }
        }
    }

    if test_results:
        config['test_results'] = test_results

    # Add task weight history if available
    if callbacks:
        for callback in callbacks:
            if isinstance(callback, EnhancedMultiTaskCallback):
                weight_history = callback.get_task_weight_history()
                if weight_history:
                    config['task_weight_evolution'] = weight_history
                break

    # Save configuration
    with open(os.path.join(results_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Save models with proper naming
    try:
        # Save final model
        final_model_path = os.path.join(model_dir, 'final_model.keras')
        model.save(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")

        # Save model in SavedModel format for deployment
        saved_model_path = os.path.join(model_dir, 'saved_model')
        model.export(saved_model_path)
        logger.info(f"SavedModel exported to: {saved_model_path}")

    except Exception as e:
        logger.error(f"Failed to save model: {e}")

    # Save detailed training summary
    create_training_summary(results_dir, config, history, task_config)

    logger.info(f"All results saved to: {results_dir}")

# ---------------------------------------------------------------------

def plot_enhanced_training_history(
    history: keras.callbacks.History,
    save_dir: str,
    task_config: TaskConfiguration
):
    """Plot comprehensive training history with TaskType support."""
    history_dict = history.history
    epochs = range(1, len(history_dict['loss']) + 1)

    # Determine subplot layout based on available data
    n_plots = 4
    if task_config.is_multi_task():
        n_plots += 1  # Add task weights plot

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    # Plot 1: Total loss
    ax_idx = 0
    axes[ax_idx].plot(epochs, history_dict['loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history_dict:
        axes[ax_idx].plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[ax_idx].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[ax_idx].set_xlabel('Epoch')
    axes[ax_idx].set_ylabel('Loss')
    axes[ax_idx].legend()
    axes[ax_idx].grid(True, alpha=0.3)

    # Plot 2: Task-specific losses
    ax_idx += 1
    task_colors = {'detection': 'green', 'segmentation': 'orange', 'classification': 'purple'}
    tasks_plotted = False

    for task in task_config.get_task_names():
        task_loss_key = f'{task}_loss'
        if task_loss_key in history_dict:
            color = task_colors.get(task, 'gray')
            axes[ax_idx].plot(epochs, history_dict[task_loss_key],
                            color=color, label=f'{task.title()} Loss', linewidth=2)
            tasks_plotted = True

    if tasks_plotted:
        axes[ax_idx].set_title('Task-Specific Losses', fontsize=14, fontweight='bold')
        axes[ax_idx].set_xlabel('Epoch')
        axes[ax_idx].set_ylabel('Loss')
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
    else:
        axes[ax_idx].text(0.5, 0.5, 'Task-specific losses\nnot available',
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
        axes[ax_idx].text(0.5, 0.5, 'Learning rate\nnot tracked',
                         ha='center', va='center', transform=axes[ax_idx].transAxes)

    # Plot 4: Task weights (if using uncertainty weighting)
    ax_idx += 1
    weights_plotted = False
    for task in task_config.get_task_names():
        weight_key = f'{task}_weight'
        if weight_key in history_dict:
            color = task_colors.get(task, 'gray')
            axes[ax_idx].plot(epochs, history_dict[weight_key],
                            color=color, label=f'{task.title()} Weight', linewidth=2)
            weights_plotted = True

    if weights_plotted:
        axes[ax_idx].set_title('Task Weights (Uncertainty)', fontsize=14, fontweight='bold')
        axes[ax_idx].set_xlabel('Epoch')
        axes[ax_idx].set_ylabel('Weight')
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
    else:
        axes[ax_idx].text(0.5, 0.5, 'Fixed task weighting\n(No uncertainty)',
                         ha='center', va='center', transform=axes[ax_idx].transAxes)

    # Plot 5: Loss smoothing
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

    # Plot 6: Final loss comparison
    ax_idx += 1
    final_losses = []
    for task in task_config.get_task_names():
        task_loss_key = f'{task}_loss'
        if task_loss_key in history_dict:
            final_losses.append((task.title(), history_dict[task_loss_key][-1]))

    if final_losses:
        tasks, losses = zip(*final_losses)
        colors = [task_colors.get(task.lower(), 'gray') for task in tasks]
        bars = axes[ax_idx].bar(tasks, losses, color=colors)
        axes[ax_idx].set_title('Final Task Losses', fontsize=14, fontweight='bold')
        axes[ax_idx].set_ylabel('Loss Value')
        axes[ax_idx].tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, loss in zip(bars, losses):
            height = bar.get_height()
            axes[ax_idx].text(bar.get_x() + bar.get_width()/2., height,
                            f'{loss:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'enhanced_training_history.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------

def create_training_summary(
    results_dir: str,
    config: Dict,
    history: keras.callbacks.History,
    task_config: TaskConfiguration
):
    """Create detailed training summary."""
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write("YOLOv12 Multi-Task Training Summary\n")
        f.write("=" * 50 + "\n\n")

        # Task configuration
        f.write("Task Configuration:\n")
        f.write(f"  Enabled Tasks: {', '.join(task_config.get_task_names())}\n")
        f.write(f"  Task Type: {'Multi-task' if task_config.is_multi_task() else 'Single-task'}\n")
        f.write(f"  Model Architecture: Named Outputs (Functional API)\n\n")

        # Model details
        f.write("Model Details:\n")
        f.write(f"  Scale: {config['model_config']['scale']}\n")
        f.write(f"  Patch Size: {config['model_config']['patch_size']}x{config['model_config']['patch_size']}\n")
        f.write(f"  Total Parameters: {config['model_config']['total_parameters']:,}\n")
        f.write(f"  Trainable Parameters: {config['model_config']['trainable_parameters']:,}\n\n")

        # Training configuration
        f.write("Training Configuration:\n")
        f.write(f"  Batch Size: {config['training_args']['batch_size']}\n")
        f.write(f"  Learning Rate: {config['training_args']['learning_rate']}\n")
        f.write(f"  Optimizer: {config['training_args']['optimizer']}\n")
        f.write(f"  Uncertainty Weighting: {config['loss_configuration']['uses_uncertainty_weighting']}\n\n")

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

        # Task weight evolution
        if 'task_weight_evolution' in config and config['task_weight_evolution']:
            f.write(f"\nTask Weight Evolution (Final):\n")
            final_weights = config['task_weight_evolution'][-1]
            for task, weight in final_weights.items():
                f.write(f"  {task.title()}: {weight:.4f}\n")

# ---------------------------------------------------------------------

def main():
    """Enhanced main function with TaskType enum support."""
    parser = argparse.ArgumentParser(
        description='Train YOLOv12 Multi-Task Model with Named Outputs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--tasks', nargs='+',
                       choices=[task.value for task in TaskType.all_tasks()],
                       default=['detection', 'segmentation', 'classification'],
                       help='Tasks to enable for training')

    # Model arguments
    parser.add_argument('--scale', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
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

    # Loss configuration arguments
    parser.add_argument('--uncertainty-weighting', action='store_true',
                       help='Use uncertainty-based adaptive task weighting')
    parser.add_argument('--detection-weight', type=float, default=1.0,
                       help='Weight for detection loss')
    parser.add_argument('--segmentation-weight', type=float, default=1.0,
                       help='Weight for segmentation loss')
    parser.add_argument('--classification-weight', type=float, default=1.0,
                       help='Weight for classification loss')

    # Control arguments
    parser.add_argument('--no-evaluate', action='store_true',
                       help='Skip evaluation on test set')
    parser.add_argument('--run-eagerly', action='store_true',
                       help='Run model in eager mode (useful for debugging)')
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

    # Log configuration
    logger.info("YOLOv12 Multi-Task Training Configuration:")
    logger.info(f"  Tasks: {args.tasks}")
    logger.info(f"  Model Scale: {args.scale}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Learning Rate: {args.learning_rate}")
    logger.info(f"  Uncertainty Weighting: {args.uncertainty_weighting}")

    # Set random seeds for reproducibility
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    # Start training with comprehensive error handling
    try:
        train_model(args)
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