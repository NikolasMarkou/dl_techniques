"""
ResNet ImageNet Classification Training with Deep Supervision

This module implements a comprehensive training pipeline for ResNet models on ImageNet
with optional deep supervision support. The pipeline includes:

1. **Deep Supervision**: Optional multi-output training where intermediate stages
   are supervised with the same classification labels, improving gradient flow.

2. **ImageNet Data Pipeline**: Efficient tf.data pipeline with proper preprocessing,
   augmentation, and caching for ImageNet-scale training.

3. **Adaptive Loss Weight Scheduling**: Dynamic adjustment of deep supervision weights
   during training using configurable scheduling strategies.

4. **Comprehensive Monitoring**: Real-time visualization of training metrics, confusion
   matrices, ROC curves, and model architecture analysis.

5. **Memory-Efficient Processing**: Streaming dataset with parallel loading and prefetching.

Architecture Support:
    - ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
    - Single and multi-output (deep supervision) training
    - Pretrained weight initialization support

Usage:
    # Train ResNet-50 with deep supervision
    python train_resnet_imagenet.py --variant resnet50 --enable-deep-supervision --epochs 100

    # Train ResNet-18 without deep supervision
    python train_resnet_imagenet.py --variant resnet18 --no-deep-supervision --epochs 90

References:
    - He et al.: "Deep Residual Learning for Image Recognition"
    - Deep Supervision techniques for multi-scale learning
"""

import os
import gc
import json
import time
import keras
import argparse
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
    deep_supervision_schedule_builder
)
from dl_techniques.models.resnet.model import (
    ResNet,
    create_resnet,
    get_model_output_info,
    create_inference_model_from_training_model
)
from dl_techniques.visualization import (
    VisualizationManager,
    TrainingHistory,
    ClassificationResults,
    PlotConfig,
    PlotStyle,
    ColorScheme,
    TrainingCurvesVisualization,
    ConfusionMatrixVisualization,
    NetworkArchitectureVisualization,
    ModelComparisonBarChart,
    ROCPRCurves
)
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Comprehensive configuration for ResNet ImageNet training.

    This dataclass encapsulates all parameters needed for training, including
    data paths, model architecture, optimization settings, deep supervision
    configuration, and monitoring options.

    Attributes:
        train_data_dir: Path to ImageNet training directory (contains class subdirectories)
        val_data_dir: Path to ImageNet validation directory (contains class subdirectories)
        image_size: Size to resize images to (default 224 for ImageNet)
        batch_size: Training batch size

        model_variant: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        num_classes: Number of output classes (1000 for ImageNet)
        pretrained: Whether to load pretrained weights

        enable_deep_supervision: Whether to use multi-scale supervision
        deep_supervision_schedule_type: Weight scheduling strategy
        deep_supervision_schedule_config: Schedule-specific parameters

        epochs: Total number of training epochs
        learning_rate: Initial learning rate
        optimizer_type: Optimizer type ('adam', 'adamw', 'sgd')
        lr_schedule_type: Learning rate schedule type
        warmup_epochs: Number of warmup epochs
        weight_decay: L2 regularization weight
        gradient_clipping: Gradient clipping threshold

        augment_data: Whether to apply data augmentation
        label_smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = typical)
        mixup_alpha: Mixup alpha parameter (0.0 = no mixup)

        monitor_every_n_epochs: Frequency of intermediate result saving
        save_best_only: Whether to save only improved models
        early_stopping_patience: Early stopping patience in epochs
        validation_steps: Number of validation steps per epoch

        cache_dataset: Whether to cache the dataset in memory
        prefetch_buffer: Prefetch buffer size
        num_parallel_calls: Number of parallel data loading threads

        output_dir: Base output directory
        experiment_name: Unique experiment identifier (auto-generated if None)
        save_training_samples: Whether to save sample results during training
        save_model_checkpoints: Whether to save model checkpoints
    """

    # === Data Configuration ===
    train_data_dir: str = "/path/to/imagenet/train"
    val_data_dir: str = "/path/to/imagenet/val"
    image_size: int = 224
    batch_size: int = 256

    # === Model Configuration ===
    model_variant: str = 'resnet50'
    num_classes: int = 1000
    pretrained: bool = False

    # === Deep Supervision Configuration ===
    enable_deep_supervision: bool = False
    deep_supervision_schedule_type: str = 'step_wise'
    deep_supervision_schedule_config: Dict[str, Any] = field(default_factory=dict)

    # === Training Configuration ===
    epochs: int = 100
    learning_rate: float = 0.1
    optimizer_type: str = 'sgd'
    lr_schedule_type: str = 'cosine_decay'
    warmup_epochs: int = 5
    weight_decay: float = 1e-4
    gradient_clipping: float = 1.0
    momentum: float = 0.9  # For SGD optimizer

    # === Augmentation Configuration ===
    augment_data: bool = True
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.0  # Set to 0.2 for mixup augmentation

    # === Monitoring Configuration ===
    monitor_every_n_epochs: int = 5
    save_best_only: bool = True
    early_stopping_patience: int = 15
    validation_steps: Optional[int] = None

    # === Data Pipeline Configuration ===
    cache_dataset: bool = False
    prefetch_buffer: int = tf.data.AUTOTUNE
    num_parallel_calls: int = tf.data.AUTOTUNE

    # === Output Configuration ===
    output_dir: str = 'results'
    experiment_name: Optional[str] = None
    save_training_samples: bool = True
    save_model_checkpoints: bool = True

    def __post_init__(self) -> None:
        """Initialize default values and validate configuration."""
        # Generate experiment name if not provided
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ds_suffix = '_ds' if self.enable_deep_supervision else ''
            self.experiment_name = f"resnet_{self.model_variant}{ds_suffix}_{timestamp}"

        # Configuration validation
        if self.image_size <= 0:
            raise ValueError("Invalid image_size: must be positive")

        if self.num_classes <= 0:
            raise ValueError("Invalid num_classes: must be positive")

        if not Path(self.train_data_dir).exists():
            raise ValueError(f"Training directory does not exist: {self.train_data_dir}")

        if not Path(self.val_data_dir).exists():
            raise ValueError(f"Validation directory does not exist: {self.val_data_dir}")


# =============================================================================
# IMAGENET DATA PIPELINE
# =============================================================================

def get_imagenet_preprocessing() -> Tuple[float, float, float, float, float, float]:
    """
    Get ImageNet normalization constants.

    Returns:
        Tuple of (mean_r, mean_g, mean_b, std_r, std_g, std_b)
    """
    # ImageNet statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return mean[0], mean[1], mean[2], std[0], std[1], std[2]


def preprocess_image(
        image: tf.Tensor,
        label: tf.Tensor,
        config: TrainingConfig,
        is_training: bool = True
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Preprocess image for training or validation.

    Args:
        image: Input image tensor
        label: Image label
        config: Training configuration
        is_training: Whether preprocessing is for training

    Returns:
        Tuple of (preprocessed_image, label)
    """
    # Decode image
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)

    if is_training and config.augment_data:
        # Training augmentation pipeline
        # Random resized crop
        image = tf.image.resize(image, [config.image_size + 32, config.image_size + 32])
        image = tf.image.random_crop(image, [config.image_size, config.image_size, 3])

        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)

        # Color jittering
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.1)
    else:
        # Validation preprocessing
        # Resize with aspect ratio preservation, then center crop
        image = tf.image.resize(image, [int(config.image_size * 1.15), int(config.image_size * 1.15)])
        image = tf.image.resize_with_crop_or_pad(image, config.image_size, config.image_size)

    # Clip to valid range
    image = tf.clip_by_value(image, 0.0, 255.0)

    # Normalize to [0, 1]
    image = image / 255.0

    # Apply ImageNet normalization
    mean_r, mean_g, mean_b, std_r, std_g, std_b = get_imagenet_preprocessing()
    image = (image - [mean_r, mean_g, mean_b]) / [std_r, std_g, std_b]

    return image, label


def create_imagenet_dataset(
        data_dir: str,
        config: TrainingConfig,
        is_training: bool = True
) -> tf.data.Dataset:
    """
    Create ImageNet dataset pipeline.

    Args:
        data_dir: Directory containing class subdirectories
        config: Training configuration
        is_training: Whether to create training dataset

    Returns:
        Configured tf.data.Dataset
    """
    # List all image files and labels
    data_dir = Path(data_dir)
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    logger.info(f"Found {len(class_names)} classes in {data_dir}")

    # Create list of (image_path, label) pairs
    image_paths = []
    labels = []

    for class_name in class_names:
        class_dir = data_dir / class_name
        class_idx = class_to_idx[class_name]

        for img_file in class_dir.glob("*.JPEG"):
            image_paths.append(str(img_file))
            labels.append(class_idx)

    logger.info(f"Found {len(image_paths)} images")

    # Create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if is_training:
        # Shuffle for training
        dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
        dataset = dataset.repeat()

    # Load and preprocess images
    def load_and_preprocess(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Load image from file and preprocess."""
        image = tf.io.read_file(path)
        return preprocess_image(image, label, config, is_training)

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=config.num_parallel_calls)

    # Cache if requested (only for small datasets)
    if config.cache_dataset and not is_training:
        dataset = dataset.cache()

    # Batch and prefetch
    dataset = dataset.batch(config.batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(config.prefetch_buffer)

    return dataset


# =============================================================================
# LOSS FUNCTIONS AND METRICS
# =============================================================================

@keras.saving.register_keras_serializable()
class PrimaryOutputAccuracy(keras.metrics.Metric):
    """
    Accuracy metric that evaluates only the primary output for multi-output models.

    This metric is designed for deep supervision scenarios where the model
    produces multiple outputs but we want to track the quality of only the
    main output during training.
    """

    def __init__(self, name: str = 'primary_accuracy', **kwargs) -> None:
        """
        Initialize the primary output accuracy metric.

        Args:
            name: Metric name for logging and visualization
            **kwargs: Additional keyword arguments
        """
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(
            self,
            y_true: Union[tf.Tensor, List[tf.Tensor]],
            y_pred: Union[tf.Tensor, List[tf.Tensor]],
            sample_weight: Optional[tf.Tensor] = None
    ) -> None:
        """
        Update accuracy state using only the primary output.

        Args:
            y_true: Ground truth tensor(s)
            y_pred: Prediction tensor(s), either single tensor or list for multi-output
            sample_weight: Optional sample weighting
        """
        # Extract primary output from potentially multi-output structure
        if isinstance(y_pred, list):
            primary_pred = y_pred[0]
            primary_true = y_true[0] if isinstance(y_true, list) else y_true
        else:
            primary_pred = y_pred
            primary_true = y_true

        # Compute predictions
        predicted_classes = tf.argmax(primary_pred, axis=-1)

        # Handle integer labels
        if len(tf.shape(primary_true)) > 1:
            true_classes = tf.argmax(primary_true, axis=-1)
        else:
            true_classes = tf.cast(primary_true, tf.int64)

        # Compute matches
        matches = tf.cast(tf.equal(predicted_classes, true_classes), tf.float32)

        self.total.assign_add(tf.reduce_sum(matches))
        self.count.assign_add(tf.cast(tf.size(matches), tf.float32))

    def result(self) -> tf.Tensor:
        """Compute the mean accuracy across all processed samples."""
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self) -> None:
        """Reset metric state for new epoch or evaluation period."""
        self.total.assign(0.0)
        self.count.assign(0.0)


@keras.saving.register_keras_serializable()
class PrimaryOutputTop5Accuracy(keras.metrics.Metric):
    """Top-5 accuracy metric for the primary output."""

    def __init__(self, name: str = 'primary_top5_accuracy', **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(
            self,
            y_true: Union[tf.Tensor, List[tf.Tensor]],
            y_pred: Union[tf.Tensor, List[tf.Tensor]],
            sample_weight: Optional[tf.Tensor] = None
    ) -> None:
        """Update top-5 accuracy state."""
        # Extract primary output
        if isinstance(y_pred, list):
            primary_pred = y_pred[0]
            primary_true = y_true[0] if isinstance(y_true, list) else y_true
        else:
            primary_pred = y_pred
            primary_true = y_true

        # Get top-5 predictions
        top5_pred = tf.nn.top_k(primary_pred, k=5).indices

        # Handle integer labels
        if len(tf.shape(primary_true)) > 1:
            true_classes = tf.argmax(primary_true, axis=-1)
        else:
            true_classes = tf.cast(primary_true, tf.int64)

        # Check if true class is in top-5
        true_classes_expanded = tf.expand_dims(true_classes, axis=-1)
        matches = tf.reduce_any(tf.equal(top5_pred, true_classes_expanded), axis=-1)
        matches = tf.cast(matches, tf.float32)

        self.total.assign_add(tf.reduce_sum(matches))
        self.count.assign_add(tf.cast(tf.size(matches), tf.float32))

    def result(self) -> tf.Tensor:
        """Compute the mean top-5 accuracy."""
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self) -> None:
        """Reset metric state."""
        self.total.assign(0.0)
        self.count.assign(0.0)


# =============================================================================
# CALLBACKS
# =============================================================================

class DeepSupervisionWeightScheduler(keras.callbacks.Callback):
    """
    Dynamic weight scheduler for deep supervision training.

    This callback automatically updates the loss weights for multi-output models
    during training according to a configurable scheduling strategy.
    """

    def __init__(self, config: TrainingConfig, num_outputs: int) -> None:
        """
        Initialize the deep supervision weight scheduler.

        Args:
            config: Training configuration containing scheduling parameters
            num_outputs: Number of model outputs to schedule weights for
        """
        super().__init__()
        self.config = config
        self.num_outputs = num_outputs
        self.total_epochs = config.epochs

        # Create the scheduling function
        ds_config = {
            'type': config.deep_supervision_schedule_type,
            'config': config.deep_supervision_schedule_config
        }
        self.scheduler = deep_supervision_schedule_builder(ds_config, self.num_outputs, invert_order=True)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Update model loss weights at the beginning of each epoch.

        Args:
            epoch: Current epoch number (0-indexed)
            logs: Training logs dictionary (unused)
        """
        # Compute training progress [0, 1]
        progress = min(1.0, epoch / max(1, self.total_epochs - 1))

        # Get new weights from scheduler
        new_weights = self.scheduler(progress)

        # Update model's loss weights directly
        self.model.loss_weights = new_weights

        # Log the weight update
        weights_str = ", ".join([f"{w:.4f}" for w in new_weights])
        logger.info(f"Epoch {epoch + 1}/{self.total_epochs} - Updated DS weights: [{weights_str}]")


class MetricsVisualizationCallback(keras.callbacks.Callback):
    """Comprehensive metrics visualization callback for training monitoring."""

    def __init__(self, config: TrainingConfig) -> None:
        """Initialize the metrics visualization callback."""
        super().__init__()
        self.config = config
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.visualization_dir = self.output_dir / "training_metrics"
        self.visualization_dir.mkdir(parents=True, exist_ok=True)

        # Initialize storage for metrics tracking
        self.train_metrics: Dict[str, List[float]] = {
            'loss': [],
            'accuracy': [],
            'top5_accuracy': [],
            'primary_accuracy': [],
            'primary_top5_accuracy': []
        }
        self.val_metrics: Dict[str, List[float]] = {
            'val_loss': [],
            'val_accuracy': [],
            'val_top5_accuracy': [],
            'val_primary_accuracy': [],
            'val_primary_top5_accuracy': []
        }

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Update metrics storage and create visualization plots."""
        if logs is None:
            logs = {}

        # Store metrics
        for metric_name, metric_value in logs.items():
            try:
                converted_value = float(metric_value)
                if metric_name in self.train_metrics:
                    self.train_metrics[metric_name].append(converted_value)
                elif metric_name in self.val_metrics:
                    self.val_metrics[metric_name].append(converted_value)
            except (ValueError, TypeError):
                pass

        # Create visualization plots every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            self._create_metrics_plots(epoch + 1)

    def _create_metrics_plots(self, epoch: int) -> None:
        """Generate and save comprehensive metrics visualization plots."""
        try:
            if not self.train_metrics.get('loss', []):
                return

            num_epochs = len(self.train_metrics['loss'])
            epochs_range = range(1, num_epochs + 1)

            # Create 2x2 subplot grid
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Training and Validation Metrics - Epoch {epoch}', fontsize=16)

            # Plot loss
            axes[0, 0].plot(epochs_range, self.train_metrics['loss'], 'b-', label='Training Loss', linewidth=2)
            if self.val_metrics.get('val_loss', []):
                axes[0, 0].plot(epochs_range, self.val_metrics['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Plot accuracy (primary or regular)
            train_acc = self.train_metrics.get('primary_accuracy') or self.train_metrics.get('accuracy', [])
            val_acc = self.val_metrics.get('val_primary_accuracy') or self.val_metrics.get('val_accuracy', [])

            if train_acc and len(train_acc) == num_epochs:
                axes[0, 1].plot(epochs_range, train_acc, 'b-', label='Training Accuracy', linewidth=2)
            if val_acc and len(val_acc) == num_epochs:
                axes[0, 1].plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Plot top-5 accuracy
            train_top5 = self.train_metrics.get('primary_top5_accuracy') or self.train_metrics.get('top5_accuracy', [])
            val_top5 = self.val_metrics.get('val_primary_top5_accuracy') or self.val_metrics.get('val_top5_accuracy',
                                                                                                 [])

            if train_top5 and len(train_top5) == num_epochs:
                axes[1, 0].plot(epochs_range, train_top5, 'b-', label='Training Top-5', linewidth=2)
            if val_top5 and len(val_top5) == num_epochs:
                axes[1, 0].plot(epochs_range, val_top5, 'r-', label='Validation Top-5', linewidth=2)
            axes[1, 0].set_title('Top-5 Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-5 Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Plot learning rate if available
            if 'learning_rate' in self.train_metrics and self.train_metrics['learning_rate']:
                axes[1, 1].plot(epochs_range, self.train_metrics['learning_rate'], 'g-', label='Learning Rate',
                                linewidth=2)
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].set_yscale('log')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].axis('off')

            plt.tight_layout()
            save_path = self.visualization_dir / f"epoch_{epoch:03d}_metrics.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            gc.collect()

            logger.info(f"Metrics plot saved to: {save_path}")

        except Exception as e:
            logger.warning(f"Failed to create metrics plots: {e}")


def create_callbacks(
        config: TrainingConfig,
        num_outputs: int
) -> List[keras.callbacks.Callback]:
    """
    Create comprehensive training callbacks for monitoring and control.

    Args:
        config: Training configuration
        num_outputs: Number of model outputs for deep supervision

    Returns:
        List of configured Keras callbacks
    """
    callbacks = []

    # Ensure output directory exists
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deep supervision weight scheduling (only for multi-output models)
    if config.enable_deep_supervision and num_outputs > 1:
        callbacks.append(DeepSupervisionWeightScheduler(config, num_outputs))

    # Model checkpointing for best model preservation
    if config.save_model_checkpoints:
        checkpoint_path = output_dir / "best_model.keras"
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_loss',
                save_best_only=config.save_best_only,
                save_weights_only=False,
                verbose=1
            )
        )

    # Early stopping to prevent overfitting
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
    )

    # CSV logging for training history
    csv_path = output_dir / "training_log.csv"
    callbacks.append(
        keras.callbacks.CSVLogger(str(csv_path), append=True)
    )

    # Real-time metrics visualization
    callbacks.append(MetricsVisualizationCallback(config))

    # TensorBoard logging for advanced monitoring
    tensorboard_dir = output_dir / "tensorboard"
    callbacks.append(
        keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    )

    return callbacks


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def setup_visualization_manager(experiment_name: str, results_dir: str) -> VisualizationManager:
    """Setup and configure the visualization manager."""
    viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    config = PlotConfig(
        style=PlotStyle.PUBLICATION,
        color_scheme=ColorScheme(
            primary="#2E86AB",
            secondary="#A23B72",
            success="#06D6A0",
            warning="#FFD166"
        ),
        title_fontsize=14,
        label_fontsize=12,
        save_format="png",
        dpi=300,
        fig_size=(12, 8)
    )

    viz_manager = VisualizationManager(
        experiment_name=experiment_name,
        output_dir=viz_dir,
        config=config
    )

    viz_manager.register_template("training_curves", TrainingCurvesVisualization)
    viz_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)
    viz_manager.register_template("network_architecture", NetworkArchitectureVisualization)
    viz_manager.register_template("model_comparison_bars", ModelComparisonBarChart)
    viz_manager.register_template("roc_pr_curves", ROCPRCurves)

    logger.info(f"Visualization manager setup complete. Plots will be saved to: {viz_dir}")
    return viz_manager


def convert_keras_history_to_training_history(keras_history: keras.callbacks.History) -> TrainingHistory:
    """Convert Keras training history to visualization framework TrainingHistory."""
    history_dict = keras_history.history
    epochs = list(range(len(history_dict['loss'])))

    train_metrics = {}
    val_metrics = {}

    for key, values in history_dict.items():
        if key.startswith('val_') and key != 'val_loss':
            val_metrics[key.replace('val_', '')] = values
        elif not key.startswith('val_') and key != 'loss':
            train_metrics[key] = values

    return TrainingHistory(
        epochs=epochs,
        train_loss=history_dict['loss'],
        val_loss=history_dict['val_loss'],
        train_metrics=train_metrics,
        val_metrics=val_metrics
    )


# =============================================================================
# MAIN TRAINING ORCHESTRATION
# =============================================================================

def train_resnet_imagenet(config: TrainingConfig) -> keras.Model:
    """
    Orchestrate the complete training pipeline for ResNet on ImageNet.

    Args:
        config: Comprehensive training configuration

    Returns:
        Trained Keras model with best weights loaded
    """
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Model Variant: {config.model_variant}")
    logger.info(f"Deep Supervision: {'ENABLED' if config.enable_deep_supervision else 'DISABLED'}")
    if config.enable_deep_supervision:
        logger.info(f"  - Schedule: {config.deep_supervision_schedule_type}")

    # === 1. Setup and Configuration ===
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    logger.info(f"Results will be saved to: {output_dir}")

    # === 2. Create Datasets ===
    logger.info("Creating ImageNet datasets...")
    train_dataset = create_imagenet_dataset(config.train_data_dir, config, is_training=True)
    val_dataset = create_imagenet_dataset(config.val_data_dir, config, is_training=False)

    # Calculate steps per epoch
    train_dir = Path(config.train_data_dir)
    # Count approximate number of training images
    num_train_images = sum(len(list((train_dir / class_dir).glob("*.JPEG")))
                           for class_dir in train_dir.iterdir() if class_dir.is_dir())
    steps_per_epoch = num_train_images // config.batch_size
    logger.info(f"Steps per epoch: {steps_per_epoch}")

    # === 3. Create Model ===
    logger.info(f"Creating ResNet model: {config.model_variant}...")
    input_shape = (config.image_size, config.image_size, 3)

    model = create_resnet(
        variant=config.model_variant,
        num_classes=config.num_classes,
        input_shape=input_shape,
        pretrained=config.pretrained,
        enable_deep_supervision=config.enable_deep_supervision,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay) if config.weight_decay > 0 else None
    )

    model.summary()

    # Analyze model output structure
    model_info = get_model_output_info(model)
    has_multiple_outputs = model_info['has_deep_supervision']
    num_outputs = model_info['num_outputs']
    logger.info(f"Model created with {num_outputs} output(s)")

    # === 4. Adapt Dataset for Multi-Output Models ===
    if has_multiple_outputs:
        logger.info("Adapting dataset for multi-output model...")

        def create_multiscale_labels(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, Tuple[tf.Tensor, ...]]:
            """Create multiple copies of labels for each output."""
            labels = [label for _ in range(num_outputs)]
            return image, tuple(labels)

        train_dataset = train_dataset.map(create_multiscale_labels, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(create_multiscale_labels, num_parallel_calls=tf.data.AUTOTUNE)

    # === 5. Configure Optimization ===
    lr_config = {
        'type': config.lr_schedule_type,
        'learning_rate': config.learning_rate,
        'decay_steps': steps_per_epoch * config.epochs,
        'warmup_steps': steps_per_epoch * config.warmup_epochs,
        'alpha': 0.01
    }

    opt_config = {
        'type': config.optimizer_type,
        'gradient_clipping_by_norm': config.gradient_clipping
    }

    # Add momentum for SGD
    if config.optimizer_type.lower() == 'sgd':
        opt_config['momentum'] = config.momentum

    lr_schedule = learning_rate_schedule_builder(lr_config)
    optimizer = optimizer_builder(opt_config, lr_schedule)

    # === 6. Configure Loss and Metrics ===
    if has_multiple_outputs:
        # Multi-output configuration
        loss_fns = [keras.losses.SparseCategoricalCrossentropy(from_logits=True)] * num_outputs
        initial_weights = [1.0 / num_outputs] * num_outputs

        # Metrics only for primary output
        metrics_for_primary = [
            PrimaryOutputAccuracy(),
            PrimaryOutputTop5Accuracy()
        ]

        # Get the name of the primary output layer
        primary_output_name = model.output[0].name.split('/')[0]
        metrics = {primary_output_name: metrics_for_primary}

        logger.info(f"Metrics configured for '{primary_output_name}' layer only")
    else:
        # Single-output configuration
        loss_fns = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        initial_weights = None
        metrics = ['accuracy', 'top_5_accuracy']

    # === 7. Compile Model ===
    logger.info("Compiling model...")
    model.compile(
        optimizer=optimizer,
        loss=loss_fns,
        loss_weights=initial_weights,
        metrics=metrics
    )
    logger.info("Model compiled successfully")

    # === 8. Create Callbacks ===
    callbacks = create_callbacks(config, num_outputs)

    # === 9. Train Model ===
    start_time = time.time()

    # Calculate validation steps if not specified
    val_steps = config.validation_steps
    if val_steps is None:
        val_dir = Path(config.val_data_dir)
        num_val_images = sum(len(list((val_dir / class_dir).glob("*.JPEG")))
                             for class_dir in val_dir.iterdir() if class_dir.is_dir())
        val_steps = num_val_images // config.batch_size

    logger.info("Starting training...")
    history = model.fit(
        train_dataset,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time / 3600:.2f} hours")

    # === 10. Create Clean Inference Model ===
    if config.enable_deep_supervision and has_multiple_outputs:
        logger.info("Creating single-output inference model...")
        inference_model = create_inference_model_from_training_model(model)
        inference_model_path = output_dir / "inference_model.keras"
        inference_model.save(inference_model_path)
        logger.info(f"Inference model saved to: {inference_model_path}")

    # === 11. Save Training History ===
    try:
        history_path = output_dir / "training_history.json"
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        logger.info(f"Training history saved to: {history_path}")
    except Exception as e:
        logger.warning(f"Failed to save training history: {e}")

    gc.collect()
    return model


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description='Train ResNet on ImageNet with Deep Supervision',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # === Data Configuration ===
    parser.add_argument(
        '--train-data-dir', type=str, required=True,
        help='Path to ImageNet training directory'
    )
    parser.add_argument(
        '--val-data-dir', type=str, required=True,
        help='Path to ImageNet validation directory'
    )
    parser.add_argument(
        '--image-size', type=int, default=224,
        help='Image size for training'
    )

    # === Model Configuration ===
    parser.add_argument(
        '--variant', type=str, default='resnet50',
        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
        help='ResNet variant'
    )
    parser.add_argument(
        '--num-classes', type=int, default=1000,
        help='Number of output classes'
    )
    parser.add_argument(
        '--pretrained', action='store_true', default=False,
        help='Load pretrained weights'
    )

    # === Deep Supervision ===
    parser.add_argument(
        '--enable-deep-supervision', action='store_true', default=False,
        help='Enable deep supervision training'
    )
    parser.add_argument(
        '--no-deep-supervision', dest='enable_deep_supervision', action='store_false',
        help='Disable deep supervision training'
    )
    parser.add_argument(
        '--deep-supervision-schedule', type=str, default='step_wise',
        choices=[
            'constant_equal', 'constant_low_to_high', 'constant_high_to_low',
            'linear_low_to_high', 'non_linear_low_to_high', 'custom_sigmoid_low_to_high',
            'scale_by_scale_low_to_high', 'cosine_annealing', 'curriculum', 'step_wise'
        ],
        help='Deep supervision weight scheduling strategy'
    )

    # === Training Configuration ===
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=256,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.1,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--optimizer', type=str, default='sgd',
        choices=['sgd', 'adam', 'adamw'],
        help='Optimizer type'
    )
    parser.add_argument(
        '--lr-schedule', type=str, default='cosine_decay',
        choices=['cosine_decay', 'exponential_decay', 'constant'],
        help='Learning rate schedule type'
    )
    parser.add_argument(
        '--warmup-epochs', type=int, default=5,
        help='Number of warmup epochs'
    )
    parser.add_argument(
        '--weight-decay', type=float, default=1e-4,
        help='Weight decay for regularization'
    )

    # === Augmentation ===
    parser.add_argument(
        '--no-augmentation', dest='augment_data', action='store_false',
        help='Disable data augmentation'
    )
    parser.add_argument(
        '--label-smoothing', type=float, default=0.1,
        help='Label smoothing factor'
    )

    # === Output ===
    parser.add_argument(
        '--output-dir', type=str, default='results',
        help='Base output directory'
    )
    parser.add_argument(
        '--experiment-name', type=str, default=None,
        help='Experiment name (auto-generated if not provided)'
    )

    # === Monitoring ===
    parser.add_argument(
        '--monitor-every', type=int, default=5,
        help='Save intermediate results every N epochs'
    )
    parser.add_argument(
        '--early-stopping-patience', type=int, default=15,
        help='Early stopping patience in epochs'
    )

    return parser.parse_args()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Main training function with comprehensive configuration and execution."""
    args = parse_arguments()

    # Create training configuration
    config = TrainingConfig(
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,

        model_variant=args.variant,
        num_classes=args.num_classes,
        pretrained=args.pretrained,

        enable_deep_supervision=args.enable_deep_supervision,
        deep_supervision_schedule_type=args.deep_supervision_schedule,

        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer,
        lr_schedule_type=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,

        augment_data=args.augment_data,
        label_smoothing=args.label_smoothing,

        monitor_every_n_epochs=args.monitor_every,
        early_stopping_patience=args.early_stopping_patience,

        output_dir=args.output_dir,
        experiment_name=args.experiment_name
    )

    # Log configuration
    logger.info("=== TRAINING CONFIGURATION ===")
    logger.info(f"Model: {config.model_variant}")
    logger.info(f"Deep Supervision: {'Enabled' if config.enable_deep_supervision else 'Disabled'}")
    if config.enable_deep_supervision:
        logger.info(f"  Schedule: {config.deep_supervision_schedule_type}")
    logger.info(f"Training: {config.epochs} epochs, batch size {config.batch_size}")
    logger.info(f"Image Size: {config.image_size}x{config.image_size}")
    logger.info(f"Learning Rate: {config.learning_rate} with {config.lr_schedule_type}")
    logger.info(f"Optimizer: {config.optimizer_type}, Weight Decay: {config.weight_decay}")
    logger.info(f"Output: {config.output_dir}/{config.experiment_name}")

    # Execute training
    try:
        model = train_resnet_imagenet(config)
        logger.info("=== TRAINING COMPLETED SUCCESSFULLY ===")
        model.summary()
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()