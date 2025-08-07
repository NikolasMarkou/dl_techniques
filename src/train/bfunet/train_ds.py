"""
Comprehensive Training Pipeline for Bias-Free U-Net (BFU-Net) Denoisers with Deep Supervision.

This script implements a robust and highly configurable training pipeline for creating
"universal" image denoisers using deep supervision techniques. Deep supervision provides
intermediate supervision signals at multiple scales during training, enabling:

1. Better gradient flow to deeper layers
2. Multi-scale feature learning and supervision
3. More stable training for very deep networks
4. Curriculum learning capabilities through weight scheduling

The deep supervision scheduling allows dynamic weighting of different output scales
throughout training, implementing strategies like:
- Linear transition from coarse to fine features
- Curriculum learning with progressive output activation
- Custom sigmoid transitions for smooth focus shifts

The trained denoiser serves as an "implicit prior" for natural images and can be used
for solving various linear inverse problems as described in Kadkhodaie & Simoncelli (2021).

Key Features:
- Deep supervision with configurable weight scheduling
- Multiple output scales with automatic weight management
- Synthesis capabilities using the learned implicit prior
- Comprehensive monitoring and visualization
- Flexible model variants (tiny, small, base, large, xlarge)
- Robust training with early stopping and checkpointing

Training Philosophy:
1. **Bias-Free Architecture**: Scale-invariant network without additive constants
2. **Universal Noise Range**: Training across continuous noise level ranges
3. **Deep Supervision**: Multi-scale intermediate supervision for better learning
4. **Implicit Prior Learning**: MSE optimization learns gradient of log-probability
5. **Large-Scale Training**: Diverse datasets for robust natural image priors
"""

import gc
import json
import time
import keras
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.filesystem import count_available_files
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder
from dl_techniques.optimization.deep_supervision import schedule_builder as deep_supervision_schedule_builder
from dl_techniques.models.bfunet_denoiser import (
    create_bfunet_denoiser, BFUNET_CONFIGS, create_bfunet_variant,
    get_model_output_info,
)

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for bias-free U-Net denoiser training with deep supervision."""

    # === Data Configuration ===
    train_image_dirs: List[str] = field(default_factory=list)  # Directories containing training images
    val_image_dirs: List[str] = field(default_factory=list)  # Directories containing validation images
    patch_size: int = 64  # Size of training patches (patch_size x patch_size)
    channels: int = 1  # Number of input channels (1=grayscale, 3=RGB)
    image_extensions: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp'])

    # === Memory Management ===
    max_train_files: Optional[int] = None  # Limit training files (None = no limit)
    max_val_files: Optional[int] = None  # Limit validation files
    parallel_reads: int = 4  # Parallel file reading threads
    dataset_shuffle_buffer: int = 1000  # Shuffle buffer size

    # === Noise Configuration ===
    noise_sigma_min: float = 0.0  # Minimum noise standard deviation
    noise_sigma_max: float = 0.4  # Maximum noise standard deviation (universal range)
    noise_distribution: str = 'uniform'  # 'uniform' or 'log_uniform' sampling

    # === Model Configuration ===
    model_type: str = 'tiny'  # one of 'tiny', 'small', 'base', 'large', 'xlarge' or 'custom'
    depth: int = 3  # Number of depth levels (for custom model)
    blocks_per_level: int = 2  # Number of residual blocks (for custom model)
    filters: int = 64  # Number of filters (for custom model)
    kernel_size: int = 3  # Residual block kernel size
    activation: str = 'relu'  # Activation function

    # === Deep Supervision Configuration ===
    enable_deep_supervision: bool = True  # Enable deep supervision training
    deep_supervision_schedule: str = 'linear_low_to_high'  # Deep supervision weight schedule
    deep_supervision_config: Dict[str, Any] = field(default_factory=dict)  # Schedule-specific parameters

    # === Training Configuration ===
    batch_size: int = 32
    epochs: int = 100
    patches_per_image: int = 16  # Number of patches to extract per image
    augment_data: bool = True  # Apply data augmentation
    normalize_input: bool = True  # Normalize input to [0, 1]
    steps_per_epoch: Optional[int] = None  # Manual override for steps per epoch

    # === Optimization Configuration ===
    learning_rate: float = 1e-3
    optimizer_type: str = 'adam'
    lr_schedule_type: str = 'cosine_decay'
    warmup_epochs: int = 5
    weight_decay: float = 1e-5
    gradient_clipping: float = 1.0

    # === Monitoring Configuration ===
    monitor_every_n_epochs: int = 5  # Save intermediate results every N epochs
    save_best_only: bool = True  # Only save model if validation loss improves
    early_stopping_patience: int = 15  # Early stopping patience
    validation_steps: Optional[int] = 100  # Number of validation steps

    # === Image Synthesis Configuration ===
    enable_synthesis: bool = True  # Enable image synthesis during monitoring
    synthesis_samples: int = 10  # Number of images to synthesize
    synthesis_steps: int = 200  # Number of synthesis iterations
    synthesis_initial_step_size: float = 0.05  # Initial gradient step size
    synthesis_final_step_size: float = 0.8  # Final gradient step size
    synthesis_initial_noise: float = 0.4  # Initial noise injection level
    synthesis_final_noise: float = 0.005  # Final noise injection level

    # === Output Configuration ===
    output_dir: str = 'results'
    experiment_name: Optional[str] = None  # Auto-generated if None
    save_training_images: bool = True  # Save sample denoised images during training
    save_model_checkpoints: bool = True

    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ds_suffix = '_ds' if self.enable_deep_supervision else ''
            self.experiment_name = f"bfunet_{self.model_type}{ds_suffix}_{timestamp}"

        # Validation
        if self.noise_sigma_min < 0 or self.noise_sigma_max <= self.noise_sigma_min:
            raise ValueError("Invalid noise sigma range")

        if self.patch_size <= 0 or self.channels <= 0:
            raise ValueError("Invalid patch size or channel configuration")

        if not self.train_image_dirs:
            raise ValueError("No training directories specified")

        if not self.val_image_dirs:
            raise ValueError("No validation directories specified")

        if self.noise_distribution not in ['uniform', 'log_uniform']:
            raise ValueError(f"Invalid noise distribution: {self.noise_distribution}")

# ---------------------------------------------------------------------
# VISUALIZATION UTILITIES
# ---------------------------------------------------------------------

def visualize_synthesis_process(
        final_samples: tf.Tensor,
        intermediate_steps: List[tf.Tensor],
        save_path: Path,
        epoch: int
) -> None:
    """
    Visualize the image synthesis process showing evolution from noise to natural images.

    Args:
        final_samples: Final generated samples
        intermediate_steps: List of intermediate sampling steps
        save_path: Path to save the visualization
        epoch: Current training epoch
    """
    import gc
    import numpy as np
    import matplotlib.pyplot as plt

    try:
        num_samples = min(4, final_samples.shape[0])
        num_steps = len(intermediate_steps)

        # Create figure showing synthesis evolution
        fig, axes = plt.subplots(num_samples, num_steps, figsize=(3 * num_steps, 3 * num_samples))
        fig.suptitle(f'Deep Supervision Image Synthesis Evolution - Epoch {epoch}\n'
                     f'(Random Noise → Natural Images via Implicit Prior)', fontsize=16, y=0.98)

        if num_samples == 1:
            axes = axes.reshape(1, -1)
        if num_steps == 1:
            axes = axes.reshape(-1, 1)

        for sample_idx in range(num_samples):
            for step_idx, step_data in enumerate(intermediate_steps):
                img = step_data[sample_idx]

                # Handle grayscale vs RGB
                if img.shape[-1] == 1:
                    img = img.squeeze(-1)
                    cmap = 'gray'
                else:
                    cmap = None

                # Ensure valid range
                img = np.clip(img, 0.0, 1.0)

                axes[sample_idx, step_idx].imshow(img, cmap=cmap, vmin=0, vmax=1)
                axes[sample_idx, step_idx].axis('off')

                # Add step label
                if sample_idx == 0:
                    step_num = step_idx * (200 // (num_steps - 1)) if step_idx < num_steps - 1 else 200
                    axes[sample_idx, step_idx].set_title(f'Step {step_num}', fontsize=10)

                # Add sample label
                if step_idx == 0:
                    axes[sample_idx, step_idx].set_ylabel(f'Sample {sample_idx + 1}',
                                                          fontsize=12, rotation=0, ha='right', va='center')

        plt.tight_layout()
        plt.subplots_adjust(top=0.90, left=0.08)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plt.clf()
        gc.collect()

    except Exception as e:
        logger.warning(f"Failed to visualize synthesis process: {e}")

# ---------------------------------------------------------------------
# DATASET BUILDER (Updated for Deep Supervision)
# ---------------------------------------------------------------------

def load_and_preprocess_image(image_path: tf.Tensor, config: TrainingConfig) -> tf.Tensor:
    """
    Load and preprocess a single image using TensorFlow operations.

    Args:
        image_path: Tensor containing path to image file
        config: Training configuration

    Returns:
        Preprocessed image patch as tensor normalized to [0, 1]
    """
    try:
        # Read image file
        image_string = tf.io.read_file(image_path)

        # Decode image
        image = tf.image.decode_image(
            image_string,
            channels=config.channels,
            expand_animations=False
        )
        image.set_shape([None, None, config.channels])

        # Convert to float32 and normalize if requested
        image = tf.cast(image, tf.float32)
        if config.normalize_input:
            # Normalize from [0, 255] to [0, 1]
            image = image / 255.0

        # Get image dimensions
        shape = tf.shape(image)
        height, width = shape[0], shape[1]

        # Handle small images by resizing
        min_dim = tf.minimum(height, width)
        min_size = config.patch_size

        # Use tf.cond for data-dependent control flow
        def resize_if_small():
            """Logic to resize the image if it's smaller than the patch size."""
            scale_factor = tf.cast(min_size, tf.float32) / tf.cast(min_dim, tf.float32)
            new_height = tf.cast(tf.math.ceil(tf.cast(height, tf.float32) * scale_factor), tf.int32)
            new_width = tf.cast(tf.math.ceil(tf.cast(width, tf.float32) * scale_factor), tf.int32)
            return tf.image.resize(image, [new_height, new_width])

        def identity():
            """Logic to return the image as is."""
            return image

        # Use tf.cond to choose which function to execute
        image = tf.cond(
            tf.logical_or(height < min_size, width < min_size),
            true_fn=resize_if_small,
            false_fn=identity
        )

        # Extract one random patch
        patch = tf.image.random_crop(
            image,
            [config.patch_size, config.patch_size, config.channels]
        )

        return patch

    except tf.errors.InvalidArgumentError:
        # Return a black patch if image loading fails
        logger.warning(f"Failed to load image: {image_path}")
        return tf.zeros([config.patch_size, config.patch_size, config.channels], dtype=tf.float32)


def augment_patch(patch: tf.Tensor) -> tf.Tensor:
    """Apply data augmentation to a patch."""
    # Random flips
    patch = tf.image.random_flip_left_right(patch)
    patch = tf.image.random_flip_up_down(patch)

    # Random rotation (90 degree increments)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    patch = tf.image.rot90(patch, k)

    return patch


def add_noise_to_patch(patch: tf.Tensor, config: TrainingConfig) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Add Gaussian noise to a clean patch.

    Args:
        patch: Clean patch tensor [height, width, channels] in [0, 1] range
        config: Training configuration

    Returns:
        Tuple of (noisy_patch, clean_patch)
    """
    # Sample noise level
    if config.noise_distribution == 'uniform':
        noise_level = tf.random.uniform([], config.noise_sigma_min, config.noise_sigma_max)
    elif config.noise_distribution == 'log_uniform':
        log_min = tf.math.log(tf.maximum(config.noise_sigma_min, 1e-6))
        log_max = tf.math.log(config.noise_sigma_max)
        log_noise = tf.random.uniform([], log_min, log_max)
        noise_level = tf.exp(log_noise)
    else:
        raise ValueError(f"Unknown distribution: {config.noise_distribution}")

    # Generate Gaussian noise
    noisy_patch = (
            patch +
            tf.random.normal(shape=tf.shape(patch), mean=0.0, stddev=noise_level)
    )

    # Clip to valid range for [0, 1] normalized input
    noisy_patch = tf.clip_by_value(noisy_patch, 0.0, 1.0)

    return noisy_patch, patch


def create_dataset_for_deep_supervision(
    directories: List[str],
    config: TrainingConfig,
    num_outputs: int,
    is_training: bool = True
) -> tf.data.Dataset:
    """
    Create a dataset for deep supervision training with multiple target outputs.

    Args:
        directories: List of directories containing images
        config: Training configuration
        num_outputs: Number of model outputs (for deep supervision)
        is_training: Whether this is a training dataset

    Returns:
        Configured tf.data.Dataset that returns (inputs, targets) where:
        - inputs: noisy patches
        - targets: list of clean patches (one for each output)
    """
    logger.info(f"Creating {'training' if is_training else 'validation'} dataset for deep supervision")
    logger.info(f"Number of outputs: {num_outputs}")

    # Build a unified file list from all directories
    all_file_paths = []
    extensions_set = {ext.lower() for ext in config.image_extensions}
    extensions_set.update({ext.upper() for ext in config.image_extensions})

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.is_dir():
            logger.warning(f"Directory not found, skipping: {directory}")
            continue

        # Recursively find all image files
        try:
            for file_path in dir_path.rglob("*"):
                if file_path.is_file() and file_path.suffix in extensions_set:
                    all_file_paths.append(str(file_path))
        except Exception as e:
            logger.warning(f"Error scanning directory {directory}: {e}")
            continue

    if not all_file_paths:
        raise ValueError(f"No image files found in directories: {directories}")

    logger.info(f"Found a total of {len(all_file_paths)} files.")

    # Apply file limits if specified
    limit = config.max_train_files if is_training else config.max_val_files
    if limit and limit < len(all_file_paths):
        logger.info(f"Limiting to {limit} files as per configuration.")
        # Shuffle before limiting to get a random subset
        np.random.shuffle(all_file_paths)
        all_file_paths = all_file_paths[:limit]

    # Create dataset from the unified list of paths
    dataset = tf.data.Dataset.from_tensor_slices(all_file_paths)

    # Apply dataset transformations
    if is_training:
        dataset = dataset.shuffle(
            buffer_size=min(config.dataset_shuffle_buffer, len(all_file_paths)),
            reshuffle_each_iteration=True
        )
        dataset = dataset.repeat()  # Repeat indefinitely for training

    # For validation, repeat to avoid OutOfRangeError
    if not is_training:
        dataset = dataset.repeat()

    # Duplicate paths for multiple patches per image if training
    if is_training and config.patches_per_image > 1:
        dataset = dataset.flat_map(
            lambda path: tf.data.Dataset.from_tensors(path).repeat(config.patches_per_image)
        )

    # Load and preprocess images
    dataset = dataset.map(
        lambda path: load_and_preprocess_image(path, config),
        num_parallel_calls=config.parallel_reads
    )

    # Filter out failed loads (all zeros)
    dataset = dataset.filter(
        lambda x: tf.reduce_sum(tf.abs(x)) > 0
    )

    # Ensure shape consistency
    dataset = dataset.map(
        lambda x: tf.ensure_shape(x, [config.patch_size, config.patch_size, config.channels])
    )

    # Apply augmentation if training
    if is_training and config.augment_data:
        dataset = dataset.map(augment_patch, num_parallel_calls=tf.data.AUTOTUNE)

    # Add noise to create training pairs
    dataset = dataset.map(
        lambda patch: add_noise_to_patch(patch, config),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # For deep supervision: create multiple target outputs as a stacked tensor
    def create_multiple_targets(noisy, clean):
        # Stack the clean target num_outputs times (all targets are the same clean patch)
        targets = tf.stack([clean for _ in range(num_outputs)], axis=0)
        return noisy, targets

    dataset = dataset.map(create_multiple_targets, num_parallel_calls=tf.data.AUTOTUNE)

    # Ensure final shape consistency
    def ensure_shapes(noisy, targets_stacked):
        noisy = tf.ensure_shape(noisy, [config.patch_size, config.patch_size, config.channels])
        # targets_stacked has shape [num_outputs, patch_size, patch_size, channels] for each sample
        # After batching, it will be [batch_size, num_outputs, patch_size, patch_size, channels]
        targets_stacked = tf.ensure_shape(
            targets_stacked,
            [num_outputs, config.patch_size, config.patch_size, config.channels]
        )
        return noisy, targets_stacked

    dataset = dataset.map(ensure_shapes, num_parallel_calls=tf.data.AUTOTUNE)

    # Batching and prefetching
    dataset = dataset.prefetch(config.batch_size * 2)
    dataset = dataset.batch(config.batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ---------------------------------------------------------------------
# METRICS (Updated for Deep Supervision)
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
def psnr_metric_primary_output(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    PSNR metric for the primary output (index 0) in deep supervision training.

    Args:
        y_true: Ground truth images in [0, 1] range
        y_pred: Predicted/denoised images in [0, 1] range

    Returns:
        Mean PSNR value across the batch for primary output
    """
    return tf.reduce_mean(tf.image.psnr(y_pred, y_true, max_val=1.0))


# ---------------------------------------------------------------------
# CUSTOM TRAINING STEP WITH DEEP SUPERVISION
# ---------------------------------------------------------------------

class DeepSupervisionModel(keras.Model):
    """
    Custom model wrapper that handles deep supervision training with weight scheduling.
    """

    def __init__(self, base_model: keras.Model, config: TrainingConfig, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.config = config

        # Get model output information
        self.model_info = get_model_output_info(base_model)
        self.num_outputs = self.model_info['num_outputs']

        # Create deep supervision scheduler if enabled
        if config.enable_deep_supervision and self.num_outputs > 1:
            ds_config = {
                'type': config.deep_supervision_schedule,
                'config': config.deep_supervision_config
            }
            self.ds_scheduler = deep_supervision_schedule_builder(ds_config, self.num_outputs)
            logger.info(f"Created deep supervision scheduler: {config.deep_supervision_schedule}")
        else:
            self.ds_scheduler = None
            logger.info("Deep supervision scheduler not created (single output or disabled)")

        # Track training progress using simple Python variables (fixes device placement issues)
        self.current_epoch_value = 0
        self.total_epochs_value = config.epochs

        # Add loss tracker for proper loss reporting
        self._loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, inputs, training=None):
        """Forward pass through base model."""
        return self.base_model(inputs, training=training)

    def compute_deep_supervision_weights(self) -> tf.Tensor:
        """Compute current deep supervision weights based on training progress."""
        if self.ds_scheduler is None:
            # Equal weighting fallback when no scheduler is available
            equal_weight = 1.0 / tf.cast(self.num_outputs, tf.float32)
            return tf.fill([self.num_outputs], equal_weight)

        # Calculate training progress (0.0 to 1.0) using Python variables
        progress = self.current_epoch_value / max(1, self.total_epochs_value)
        progress = max(0.0, min(1.0, progress))  # Clip to [0, 1]

        # Get weights from scheduler directly (no tf.py_function needed)
        weights_np = self.ds_scheduler(progress).astype(np.float32)
        weights = tf.constant(weights_np, dtype=tf.float32)

        return weights

    def train_step(self, data):
        """Custom training step with deep supervision weighting."""
        x, y = data

        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self(x, training=True)

            # Handle single vs multiple outputs
            if not isinstance(predictions, list):
                predictions = [predictions]

            # Handle stacked targets format
            # y has shape [batch_size, num_outputs, height, width, channels]
            # We need to convert this to a list of [batch_size, height, width, channels] tensors
            if len(y.shape) == 5 and y.shape[1] == self.num_outputs:
                # Unstack along the num_outputs dimension (axis=1)
                y = tf.unstack(y, axis=1)
            elif not isinstance(y, list):
                # Single target, replicate for all outputs
                y = [y for _ in range(len(predictions))]

            # Compute weighted loss for each output
            total_loss = 0.0

            # Get deep supervision weights
            ds_weights = self.compute_deep_supervision_weights()

            for i, (pred, target) in enumerate(zip(predictions, y)):
                # Resize target to match prediction's spatial dimensions
                pred_shape = tf.shape(pred)
                target_height, target_width = pred_shape[1], pred_shape[2]

                # Resize target to match prediction size
                resized_target = tf.image.resize(
                    target,
                    [target_height, target_width],
                    method='bilinear'
                )

                # Compute MSE loss for this output
                mse_loss = tf.reduce_mean(tf.square(pred - resized_target))

                # Apply deep supervision weight
                weight = ds_weights[i]
                total_loss += weight * mse_loss

            # Add regularization losses (fixed for Keras 3.x)
            regularization_losses = self.base_model.losses
            if regularization_losses:
                total_loss += tf.add_n(regularization_losses)

        # Compute gradients and update weights
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics (using primary output - index 0)
        primary_pred = predictions[0]
        primary_target = y[0]

        # Update loss tracker with total weighted loss
        self._loss_tracker.update_state(total_loss)

        # Update compiled metrics manually - this ensures MAE, RMSE, PSNR are calculated
        if hasattr(self, 'compiled_metrics') and self.compiled_metrics is not None:
            # Update each metric individually for better error handling
            metrics_list = getattr(self.compiled_metrics, 'metrics', getattr(self.compiled_metrics, '_metrics', []))
            for metric in metrics_list:
                try:
                    if hasattr(metric, 'update_state'):
                        if 'psnr' in metric.name.lower():
                            # PSNR expects (y_true, y_pred) format and values in [0,1]
                            metric.update_state(primary_target, primary_pred)
                        elif metric.name in ['mae', 'mean_absolute_error']:
                            # MAE metric
                            metric.update_state(primary_target, primary_pred)
                        elif metric.name in ['rmse', 'root_mean_squared_error']:
                            # RMSE metric
                            metric.update_state(primary_target, primary_pred)
                        else:
                            # Other metrics
                            metric.update_state(primary_target, primary_pred)
                except Exception as e:
                    logger.warning(f"Failed to update metric {getattr(metric, 'name', 'unknown')}: {e}")

        # Create results dictionary - include all metrics
        results = {}

        # Add loss from our custom tracker
        results['loss'] = self._loss_tracker.result()

        # Add compiled metrics
        if hasattr(self, 'compiled_metrics') and self.compiled_metrics is not None:
            metrics_list = getattr(self.compiled_metrics, 'metrics', getattr(self.compiled_metrics, '_metrics', []))
            for metric in metrics_list:
                if hasattr(metric, 'name') and hasattr(metric, 'result'):
                    results[metric.name] = metric.result()

        # Add deep supervision info to logs
        results['ds_weight_primary'] = ds_weights[0]
        if self.num_outputs > 1:
            results['ds_weight_mean'] = tf.reduce_mean(ds_weights)

        return results

    def test_step(self, data):
        """Custom validation step using primary output only."""
        x, y = data

        # Forward pass
        predictions = self(x, training=False)

        # Handle single vs multiple outputs - use primary output for validation
        if isinstance(predictions, list):
            primary_pred = predictions[0]
        else:
            primary_pred = predictions

        # Handle stacked targets format - get primary target
        # y has shape [batch_size, num_outputs, height, width, channels]
        if len(y.shape) == 5 and y.shape[1] == self.num_outputs:
            # Get the first target (primary output target) along axis=1
            primary_target = y[:, 0, :, :, :]  # [batch_size, height, width, channels]
        elif isinstance(y, list):
            primary_target = y[0]
        else:
            primary_target = y

        # Compute validation loss (MSE on primary output only)
        val_loss = tf.reduce_mean(tf.square(primary_pred - primary_target))

        # Update loss tracker with validation loss
        self._loss_tracker.update_state(val_loss)

        # Update compiled metrics manually
        if hasattr(self, 'compiled_metrics') and self.compiled_metrics is not None:
            metrics_list = getattr(self.compiled_metrics, 'metrics', getattr(self.compiled_metrics, '_metrics', []))
            for metric in metrics_list:
                try:
                    if hasattr(metric, 'update_state'):
                        if 'psnr' in metric.name.lower():
                            metric.update_state(primary_target, primary_pred)
                        elif metric.name in ['mae', 'mean_absolute_error']:
                            metric.update_state(primary_target, primary_pred)
                        elif metric.name in ['rmse', 'root_mean_squared_error']:
                            metric.update_state(primary_target, primary_pred)
                        else:
                            metric.update_state(primary_target, primary_pred)
                except Exception as e:
                    logger.warning(f"Failed to update validation metric {getattr(metric, 'name', 'unknown')}: {e}")

        # Create results dictionary
        results = {}

        # Add loss
        results['loss'] = self._loss_tracker.result()

        # Add compiled metrics with 'val_' prefix for validation
        if hasattr(self, 'compiled_metrics') and self.compiled_metrics is not None:
            metrics_list = getattr(self.compiled_metrics, 'metrics', getattr(self.compiled_metrics, '_metrics', []))
            for metric in metrics_list:
                if hasattr(metric, 'name') and hasattr(metric, 'result'):
                    val_metric_name = f"val_{metric.name}" if not metric.name.startswith('val_') else metric.name
                    results[val_metric_name] = metric.result()

        return results

    def set_epoch(self, epoch: int):
        """Update current epoch for deep supervision scheduling."""
        self.current_epoch_value = epoch

    @property
    def metrics(self):
        """Return all metrics including custom loss tracker."""
        metrics = []

        # Add our custom loss tracker first
        if hasattr(self, '_loss_tracker'):
            metrics.append(self._loss_tracker)

        # Add compiled metrics if they exist
        if hasattr(self, 'compiled_metrics') and self.compiled_metrics is not None:
            metrics_list = getattr(self.compiled_metrics, 'metrics', getattr(self.compiled_metrics, '_metrics', []))
            metrics.extend(metrics_list)

        return metrics

    def reset_metrics(self):
        """Reset all metrics."""
        # Reset custom loss tracker
        if hasattr(self, '_loss_tracker'):
            self._loss_tracker.reset_state()

        # Reset compiled metrics
        if hasattr(self, 'compiled_metrics') and self.compiled_metrics is not None:
            # Try the new API first
            if hasattr(self.compiled_metrics, 'reset_state'):
                try:
                    self.compiled_metrics.reset_state()
                except:
                    # Fallback to individual metric reset
                    pass

            # Reset individual metrics
            metrics_list = getattr(self.compiled_metrics, 'metrics', getattr(self.compiled_metrics, '_metrics', []))
            for metric in metrics_list:
                if hasattr(metric, 'reset_state'):
                    try:
                        metric.reset_state()
                    except Exception as e:
                        logger.warning(f"Failed to reset metric {getattr(metric, 'name', 'unknown')}: {e}")

    def get_config(self):
        """Return the configuration of the model."""
        base_config = super().get_config()
        config = {
            'base_model_config': self.base_model.get_config(),
            'training_config': {
                'enable_deep_supervision': self.config.enable_deep_supervision,
                'deep_supervision_schedule': self.config.deep_supervision_schedule,
                'deep_supervision_config': self.config.deep_supervision_config,
                'epochs': self.config.epochs,
            },
            'num_outputs': self.num_outputs,
        }
        base_config.update(config)
        return base_config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Create a model from its configuration."""
        # This is a simplified version - in practice you'd need to reconstruct
        # the base model and config objects properly
        raise NotImplementedError(
            "DeepSupervisionModel.from_config() not implemented. "
            "Please recreate the model using the constructor with the original base_model and config."
        )

    def summary(self, *args, **kwargs):
        """Print a summary of the base model."""
        return self.base_model.summary(*args, **kwargs)

    def count_params(self):
        """Count the total number of parameters in the base model."""
        return self.base_model.count_params()

    def save_base_model(self, filepath, **kwargs):
        """Save the underlying base model."""
        return self.base_model.save(filepath, **kwargs)

    def get_base_model(self):
        """Get the underlying base model."""
        return self.base_model

    def get_weights(self):
        """Get weights from the base model."""
        return self.base_model.get_weights()

    def set_weights(self, weights):
        """Set weights to the base model."""
        return self.base_model.set_weights(weights)

    def trainable_weights(self):
        """Get trainable weights from the base model."""
        return self.base_model.trainable_weights

    def non_trainable_weights(self):
        """Get non-trainable weights from the base model."""
        return self.base_model.non_trainable_weights

# ---------------------------------------------------------------------
# IMAGE SYNTHESIS (Updated for Deep Supervision)
# ---------------------------------------------------------------------

@tf.function
def _sampling_step(denoiser, y, h_t, gamma_t):
    """Single sampling step compiled with tf.function for speed."""
    # Compute denoiser residual: d_t = D(y_t) - y_t
    denoised = denoiser(y, training=False)
    if isinstance(denoised, list):
        denoised = denoised[0]  # Use primary output

    d_t = denoised - y

    # Generate fresh Gaussian noise
    z_t = tf.random.normal(tf.shape(y))

    # Update rule: y_{t+1} = y_t + h_t * d_t + γ_t * z_t
    y_next = y + h_t * d_t + gamma_t * z_t

    # Clip to valid [0, 1] range
    y_next = tf.clip_by_value(y_next, 0.0, 1.0)

    return y_next


def unconditional_sampling_with_deep_supervision(
        denoiser: keras.Model,
        num_samples: int = 4,
        image_shape: Tuple[int, int, int] = (64, 64, 1),
        num_steps: int = 100,  # Reduced from 200 to 100
        initial_step_size: float = 0.05,  # Reduced from 0.1
        final_step_size: float = 0.8,  # Reduced from 1.0
        initial_noise_level: float = 0.3,  # Reduced from 0.5
        final_noise_level: float = 0.005,
        seed: Optional[int] = None,
        save_intermediate: bool = True
) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    """
    Optimized unconditional image synthesis using denoiser's implicit prior.

    Optimizations:
    - Compiled sampling step with tf.function
    - Reduced default number of steps from 200 to 100
    - Less frequent logging and intermediate saving
    - Optimized tensor operations

    Args:
        denoiser: Trained bias-free denoiser model (single or multi-output)
        num_samples: Number of images to generate
        image_shape: Shape of generated images (height, width, channels)
        num_steps: Number of sampling steps (reduced default: 100)
        initial_step_size: Initial step size h_0
        final_step_size: Final step size h_final
        initial_noise_level: Initial noise injection level γ_0
        final_noise_level: Final noise injection level γ_final
        seed: Random seed for reproducibility
        save_intermediate: Whether to save intermediate steps

    Returns:
        Tuple of (final_samples, intermediate_steps)
    """
    if seed is not None:
        tf.random.set_seed(seed)

    logger.info(f"Starting optimized sampling: {num_samples} samples, {num_steps} steps")

    # Initialize with random noise y_0
    y = tf.random.normal([num_samples] + list(image_shape), mean=0.5, stddev=0.25)
    y = tf.clip_by_value(y, 0.0, 1.0)

    # Pre-compute scheduling parameters (vectorized)
    step_sizes = tf.linspace(initial_step_size, final_step_size, num_steps)
    noise_levels = tf.linspace(initial_noise_level, final_noise_level, num_steps)

    intermediate_steps = []

    # Save initial state
    if save_intermediate:
        intermediate_steps.append(y.numpy().copy())

    # Determine how often to save intermediate steps and log
    save_interval = max(1, num_steps // 8)  # Save 8 intermediate steps max
    log_interval = max(1, num_steps // 5)  # Log 5 times max

    # Main sampling loop
    for t in range(num_steps):
        # Get current scheduling parameters
        h_t = step_sizes[t]
        gamma_t = noise_levels[t]

        # Perform sampling step (compiled for speed)
        y = _sampling_step(denoiser, y, h_t, gamma_t)

        # Save intermediate steps less frequently
        if save_intermediate and (t % save_interval == 0 or t == num_steps - 1):
            intermediate_steps.append(y.numpy().copy())

        # Log progress less frequently
        if t % log_interval == 0 or t == num_steps - 1:
            mean_intensity = tf.reduce_mean(y)
            logger.info(f"Step {t}/{num_steps}: mean={mean_intensity:.3f}, "
                        f"h_t={h_t:.4f}, γ_t={gamma_t:.4f}")

    logger.info(f"Sampling completed in {num_steps} steps. Generated {num_samples} samples.")

    return y, intermediate_steps


# ---------------------------------------------------------------------
# MONITORING AND CALLBACKS (Updated for Deep Supervision)
# ---------------------------------------------------------------------

class DeepSupervisionEpochCallback(keras.callbacks.Callback):
    """Callback to update epoch information for deep supervision scheduling."""

    def __init__(self, ds_model: DeepSupervisionModel):
        super().__init__()
        self.ds_model = ds_model

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Update epoch for deep supervision weight scheduling."""
        self.ds_model.set_epoch(epoch)


class MetricsVisualizationCallback(keras.callbacks.Callback):
    """Callback to visualize training and validation metrics (updated for deep supervision)."""

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.visualization_dir = self.output_dir / "visualization_plots"
        self.visualization_dir.mkdir(parents=True, exist_ok=True)

        # Storage for metrics
        self.train_metrics = {
            'loss': [],
            'mae': [],
            'rmse': [],
            'psnr_metric_primary_output': []
        }
        self.val_metrics = {
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'val_psnr_metric_primary_output': []
        }

        # Deep supervision metrics
        self.ds_metrics = {
            'ds_weight_primary': [],
            'ds_weight_mean': []
        }

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Update metrics and create visualization plots."""
        if logs is None:
            logs = {}

        # Store training metrics - only append if key exists in logs
        for key in self.train_metrics.keys():
            if key in logs:
                self.train_metrics[key].append(logs[key])
            # If this is the first time a metric is missing, pad with None or 0
            elif len(self.train_metrics[key]) < len(self.train_metrics['loss']):
                self.train_metrics[key].append(0.0)  # Use 0 as fallback

        # Store validation metrics - only append if key exists in logs
        for key in self.val_metrics.keys():
            if key in logs:
                self.val_metrics[key].append(logs[key])
            elif len(self.val_metrics[key]) < len(self.train_metrics['loss']):
                self.val_metrics[key].append(0.0)  # Use 0 as fallback

        # Store deep supervision metrics - only append if key exists in logs
        for key in self.ds_metrics.keys():
            if key in logs:
                self.ds_metrics[key].append(logs[key])
            elif len(self.ds_metrics[key]) < len(self.train_metrics['loss']):
                self.ds_metrics[key].append(0.0)  # Use 0 as fallback

        # Create plots every few epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            self._create_metrics_plots(epoch + 1)

    def _create_metrics_plots(self, epoch: int):
        """Create and save metrics visualization plots."""
        try:
            # Only proceed if we have loss data
            if not self.train_metrics['loss']:
                return

            epochs_range = range(1, len(self.train_metrics['loss']) + 1)

            # Create figure with additional subplot for deep supervision weights
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'Training Metrics with Deep Supervision - Epoch {epoch}', fontsize=16)

            # MSE (Loss) Plot
            axes[0, 0].plot(epochs_range, self.train_metrics['loss'], 'b-', label='Training MSE', linewidth=2)
            if self.val_metrics['val_loss'] and len(self.val_metrics['val_loss']) == len(epochs_range):
                axes[0, 0].plot(epochs_range, self.val_metrics['val_loss'], 'r-', label='Validation MSE', linewidth=2)
            axes[0, 0].set_title('Mean Squared Error (MSE)')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('MSE')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # MAE Plot
            if self.train_metrics['mae'] and len(self.train_metrics['mae']) == len(epochs_range):
                axes[0, 1].plot(epochs_range, self.train_metrics['mae'], 'b-', label='Training MAE', linewidth=2)
                if self.val_metrics['val_mae'] and len(self.val_metrics['val_mae']) == len(epochs_range):
                    axes[0, 1].plot(epochs_range, self.val_metrics['val_mae'], 'r-', label='Validation MAE',
                                    linewidth=2)
                axes[0, 1].set_title('Mean Absolute Error (MAE)')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('MAE')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, 'MAE data\nnot available', ha='center', va='center',
                                transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Mean Absolute Error (MAE)')

            # RMSE Plot
            if self.train_metrics['rmse'] and len(self.train_metrics['rmse']) == len(epochs_range):
                axes[0, 2].plot(epochs_range, self.train_metrics['rmse'], 'b-', label='Training RMSE', linewidth=2)
                if self.val_metrics['val_rmse'] and len(self.val_metrics['val_rmse']) == len(epochs_range):
                    axes[0, 2].plot(epochs_range, self.val_metrics['val_rmse'], 'r-', label='Validation RMSE',
                                    linewidth=2)
                axes[0, 2].set_title('Root Mean Squared Error (RMSE)')
                axes[0, 2].set_xlabel('Epoch')
                axes[0, 2].set_ylabel('RMSE')
                axes[0, 2].legend()
                axes[0, 2].grid(True, alpha=0.3)
            else:
                axes[0, 2].text(0.5, 0.5, 'RMSE data\nnot available', ha='center', va='center',
                                transform=axes[0, 2].transAxes)
                axes[0, 2].set_title('Root Mean Squared Error (RMSE)')

            # PSNR Plot
            if self.train_metrics['psnr_metric_primary_output'] and len(
                    self.train_metrics['psnr_metric_primary_output']) == len(epochs_range):
                axes[1, 0].plot(epochs_range, self.train_metrics['psnr_metric_primary_output'], 'b-',
                                label='Training PSNR', linewidth=2)
                if self.val_metrics['val_psnr_metric_primary_output'] and len(
                        self.val_metrics['val_psnr_metric_primary_output']) == len(epochs_range):
                    axes[1, 0].plot(epochs_range, self.val_metrics['val_psnr_metric_primary_output'], 'r-',
                                    label='Validation PSNR', linewidth=2)
                axes[1, 0].set_title('Peak Signal-to-Noise Ratio (PSNR)')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('PSNR (dB)')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'PSNR data\nnot available', ha='center', va='center',
                                transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Peak Signal-to-Noise Ratio (PSNR)')

            # Deep Supervision Primary Weight Plot
            if self.ds_metrics['ds_weight_primary'] and len(self.ds_metrics['ds_weight_primary']) == len(epochs_range):
                axes[1, 1].plot(epochs_range, self.ds_metrics['ds_weight_primary'], 'g-', label='Primary Output Weight',
                                linewidth=2)
                axes[1, 1].set_title('Deep Supervision: Primary Output Weight')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Weight')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_ylim(0, 1)
            else:
                axes[1, 1].text(0.5, 0.5, 'Deep Supervision\nPrimary Weight\nNot Available', ha='center', va='center',
                                transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Deep Supervision: Primary Weight')

            # Deep Supervision Mean Weight Plot
            if self.ds_metrics['ds_weight_mean'] and len(self.ds_metrics['ds_weight_mean']) == len(epochs_range):
                axes[1, 2].plot(epochs_range, self.ds_metrics['ds_weight_mean'], 'purple', label='Mean Weight',
                                linewidth=2)
                axes[1, 2].set_title('Deep Supervision: Mean Output Weight')
                axes[1, 2].set_xlabel('Epoch')
                axes[1, 2].set_ylabel('Mean Weight')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
            else:
                axes[1, 2].text(0.5, 0.5, 'Deep Supervision\nMean Weights\nNot Available', ha='center', va='center',
                                transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('DS Mean Weights')

            plt.tight_layout()
            save_path = self.visualization_dir / f"epoch_{epoch:03d}_metrics.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            plt.clf()
            gc.collect()

            # Save metrics data with safe conversion
            metrics_data = {
                'epoch': epoch,
                'train_metrics': {k: [float(x) if hasattr(x, 'numpy') else float(x) for x in v] for k, v in
                                  self.train_metrics.items()},
                'val_metrics': {k: [float(x) if hasattr(x, 'numpy') else float(x) for x in v] for k, v in
                                self.val_metrics.items()},
                'ds_metrics': {k: [float(x) if hasattr(x, 'numpy') else float(x) for x in v] for k, v in
                               self.ds_metrics.items()}
            }
            metrics_file = self.visualization_dir / "latest_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to create metrics plots: {e}")


class StreamingResultMonitor(keras.callbacks.Callback):
    """Memory-efficient monitoring using streaming data (updated for deep supervision)."""

    def __init__(self, config: TrainingConfig, val_directories: List[str]):
        super().__init__()
        self.config = config
        self.monitor_freq = config.monitor_every_n_epochs
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.results_dir = self.output_dir / "visualization_plots"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Create a small validation dataset for monitoring
        self._create_monitor_dataset(val_directories)

    def _create_monitor_dataset(self, val_directories: List[str]):
        """Create a small dataset for consistent monitoring."""
        monitor_files = []
        extensions_set = set(ext.lower() for ext in self.config.image_extensions)
        extensions_set.update(ext.upper() for ext in self.config.image_extensions)

        for directory in val_directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                continue

            try:
                for file_path in dir_path.rglob("*"):
                    if file_path.is_file() and file_path.suffix in extensions_set:
                        monitor_files.append(str(file_path))
                        if len(monitor_files) >= 10:
                            break
                if len(monitor_files) >= 10:
                    break
            except Exception as e:
                logger.error(f"Error getting monitor files from {directory}: {e}")
                continue

        if not monitor_files:
            logger.warning("No files found for monitoring")
            self.test_batch = None
            return

        try:
            clean_patches = []
            for file_path in monitor_files:
                path_tensor = tf.constant(file_path)
                patch = load_and_preprocess_image(path_tensor, self.config)
                clean_patches.append(patch)

            self.test_batch = tf.stack(clean_patches)
            logger.info(f"Created monitoring dataset with batch shape: {self.test_batch.shape}")
        except Exception as e:
            logger.error(f"Failed to create monitor dataset: {e}")
            self.test_batch = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Save intermediate results every N epochs."""
        if (epoch + 1) % self.monitor_freq != 0 or self.test_batch is None:
            return

        def _monitor_and_save(epoch_numpy):
            epoch_val = int(epoch_numpy)
            logger.info(f"Saving intermediate results for epoch {epoch_val}")

            try:
                # Add noise
                noisy_images, clean_images = add_noise_to_patch(self.test_batch, self.config)

                # Get model predictions (handle multi-output)
                model_outputs = self.model(noisy_images, training=False)

                # Use primary output (index 0) for visualization and metrics
                if isinstance(model_outputs, list):
                    denoised_images = model_outputs[0]  # Primary output
                else:
                    denoised_images = model_outputs

                # Save sample images
                if self.config.save_training_images:
                    self._save_image_samples(
                        epoch_val,
                        noisy_images,
                        clean_images,
                        denoised_images
                    )

                # Compute metrics using primary output
                mse_loss = tf.reduce_mean(tf.square(denoised_images - clean_images))
                psnr = tf.reduce_mean(tf.image.psnr(denoised_images, clean_images, max_val=1.0))

                logger.info(f"Epoch {epoch_val} - Primary Output MSE: {mse_loss.numpy():.6f}, PSNR: {psnr.numpy():.2f} dB")

                # Image synthesis using implicit prior
                if self.config.enable_synthesis:
                    logger.info(f"Generating synthetic images using implicit prior...")
                    try:
                        synthesis_shape = (self.config.patch_size, self.config.patch_size, self.config.channels)
                        generated_samples, intermediate_steps = unconditional_sampling_with_deep_supervision(
                            denoiser=self.model,
                            num_samples=self.config.synthesis_samples,
                            image_shape=synthesis_shape,
                            num_steps=self.config.synthesis_steps,
                            initial_step_size=self.config.synthesis_initial_step_size,
                            final_step_size=self.config.synthesis_final_step_size,
                            initial_noise_level=self.config.synthesis_initial_noise,
                            final_noise_level=self.config.synthesis_final_noise,
                            seed=epoch_val,
                            save_intermediate=True
                        )

                        # Visualize synthesis process
                        synthesis_save_path = self.results_dir / f"epoch_{epoch_val:03d}_synthesis.png"
                        visualize_synthesis_process(
                            final_samples=generated_samples,
                            intermediate_steps=intermediate_steps,
                            save_path=synthesis_save_path,
                            epoch=epoch_val
                        )

                        # Log synthesis metrics
                        generated_mean = tf.reduce_mean(generated_samples)
                        generated_std = tf.math.reduce_std(generated_samples)
                        logger.info(f"Generated images - Mean: {generated_mean:.3f}, Std: {generated_std:.3f}")

                    except Exception as synthesis_error:
                        logger.warning(f"Image synthesis failed at epoch {epoch_val}: {synthesis_error}")

                # Save metrics
                metrics = {
                    'epoch': epoch_val,
                    'val_mse': float(mse_loss),
                    'val_psnr': float(psnr),
                    'timestamp': datetime.now().isoformat()
                }
                metrics_file = self.results_dir / f"epoch_{epoch_val:03d}_metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)

                # Force garbage collection
                del noisy_images, clean_images, denoised_images
                gc.collect()

            except Exception as e:
                tf.print(f"Error during monitoring callback at epoch {epoch_val}: {e}")

            return 0

        # Wrap in tf.py_function
        tf.py_function(func=_monitor_and_save, inp=[epoch + 1], Tout=[tf.int32])

    def _save_image_samples(self, epoch: int, noisy: tf.Tensor,
                            clean: tf.Tensor, denoised: tf.Tensor):
        """Save sample images for visual inspection."""
        try:
            num_samples = min(10, noisy.shape[0])

            fig, axes = plt.subplots(3, 10, figsize=(25, 7.5))
            fig.suptitle(f'Deep Supervision Denoising Results - Epoch {epoch}', fontsize=20, y=0.98)

            for i in range(10):
                if i < num_samples:
                    clean_img = np.clip(clean[i].numpy(), 0.0, 1.0)
                    noisy_img = np.clip(noisy[i].numpy(), 0.0, 1.0)
                    denoised_img = np.clip(denoised[i].numpy(), 0.0, 1.0)

                    if clean_img.shape[-1] == 1:
                        clean_img = clean_img.squeeze(-1)
                        noisy_img = noisy_img.squeeze(-1)
                        denoised_img = denoised_img.squeeze(-1)
                        cmap = 'gray'
                    else:
                        cmap = None

                    # Top row: Clean images
                    axes[0, i].imshow(clean_img, cmap=cmap, vmin=0, vmax=1)
                    if i == 0:
                        axes[0, i].set_ylabel('Clean', fontsize=14, rotation=0, ha='right', va='center')
                    axes[0, i].set_title(f'Sample {i + 1}', fontsize=10)
                    axes[0, i].axis('off')

                    # Middle row: Noisy images
                    axes[1, i].imshow(noisy_img, cmap=cmap, vmin=0, vmax=1)
                    if i == 0:
                        axes[1, i].set_ylabel('Noisy', fontsize=14, rotation=0, ha='right', va='center')
                    axes[1, i].axis('off')

                    # Bottom row: Denoised images (primary output)
                    axes[2, i].imshow(denoised_img, cmap=cmap, vmin=0, vmax=1)
                    if i == 0:
                        axes[2, i].set_ylabel('Denoised\n(Primary)', fontsize=14, rotation=0, ha='right', va='center')
                    axes[2, i].axis('off')
                else:
                    # Hide unused subplots
                    axes[0, i].axis('off')
                    axes[1, i].axis('off')
                    axes[2, i].axis('off')

            plt.tight_layout()
            plt.subplots_adjust(top=0.92, left=0.08, right=0.98)
            save_path = self.results_dir / f"epoch_{epoch:03d}_samples.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            plt.clf()
            gc.collect()

        except Exception as e:
            logger.warning(f"Failed to save image samples: {e}")


def create_callbacks(
    config: TrainingConfig,
    val_directories: List[str],
    ds_model: Optional[DeepSupervisionModel] = None
) -> List[keras.callbacks.Callback]:
    """Create training callbacks for deep supervision training."""
    callbacks = []

    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deep supervision epoch callback
    if ds_model is not None:
        callbacks.append(DeepSupervisionEpochCallback(ds_model))

    # Model checkpointing
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

    # Early stopping
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
    )

    # CSV logging
    csv_path = output_dir / "training_log.csv"
    callbacks.append(
        keras.callbacks.CSVLogger(str(csv_path), append=True)
    )

    # Metrics visualization callback
    callbacks.append(
        MetricsVisualizationCallback(config)
    )

    # Streaming result monitoring
    callbacks.append(
        StreamingResultMonitor(config, val_directories)
    )

    # TensorBoard logging
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


# ---------------------------------------------------------------------
# MODEL CREATION HELPER (Updated for Deep Supervision)
# ---------------------------------------------------------------------

def create_model_instance(config: TrainingConfig, input_shape: Tuple[int, int, int]) -> keras.Model:
    """
    Create a model instance based on configuration with deep supervision support.

    Args:
        config: Training configuration
        input_shape: Input tensor shape

    Returns:
        Keras model instance
    """
    if config.model_type in BFUNET_CONFIGS:
        return create_bfunet_variant(
            variant=config.model_type,
            input_shape=input_shape,
            enable_deep_supervision=config.enable_deep_supervision
        )
    elif config.model_type == 'custom':
        return create_bfunet_denoiser(
            input_shape=input_shape,
            depth=config.depth,
            initial_filters=config.filters,
            blocks_per_level=config.blocks_per_level,
            kernel_size=config.kernel_size,
            activation=config.activation,
            enable_deep_supervision=config.enable_deep_supervision,
            model_name=f'bfunet_custom_{config.experiment_name}'
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


# ---------------------------------------------------------------------
# TRAINING FUNCTION (Updated for Deep Supervision)
# ---------------------------------------------------------------------

def train_bfunet_denoiser_with_deep_supervision(config: TrainingConfig) -> keras.Model:
    """
    Train a bias-free U-Net denoiser model with deep supervision and save clean inference model.

    Args:
        config: Training configuration

    Returns:
        Trained Keras model (DeepSupervisionModel wrapper)
    """
    logger.info("Starting bias-free U-Net denoiser training with deep supervision")
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Deep supervision enabled: {config.enable_deep_supervision}")
    logger.info(f"Deep supervision schedule: {config.deep_supervision_schedule}")

    # Create output directory and save config
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    # Validate directories exist
    for directory in config.train_image_dirs:
        if not Path(directory).exists():
            logger.warning(f"Training directory does not exist: {directory}")

    for directory in config.val_image_dirs:
        if not Path(directory).exists():
            logger.warning(f"Validation directory does not exist: {directory}")

    # Count available files
    logger.info("Counting available files...")
    try:
        train_file_count = count_available_files(
            config.train_image_dirs,
            config.image_extensions,
            config.max_train_files
        )
        val_file_count = count_available_files(
            config.val_image_dirs,
            config.image_extensions,
            config.max_val_files
        )
    except Exception as e:
        logger.warning(f"Error counting files: {e}")
        train_file_count = 1000  # Fallback estimate
        val_file_count = 100

    if train_file_count == 0:
        raise ValueError("No training files found!")
    if val_file_count == 0:
        raise ValueError("No validation files found!")

    logger.info(f"Found approximately {train_file_count} training files")
    logger.info(f"Found approximately {val_file_count} validation files")

    # Create base model
    logger.info(f"Creating {config.model_type} model...")
    input_shape = (config.patch_size, config.patch_size, config.channels)
    base_model = create_model_instance(config, input_shape)
    base_model.summary()

    # Get model output information
    model_info = get_model_output_info(base_model)
    num_outputs = model_info['num_outputs']

    logger.info(f"Model has {num_outputs} outputs")
    if model_info['has_deep_supervision']:
        logger.info("Deep supervision outputs detected")
        for i, shape in enumerate(model_info['output_shapes']):
            output_type = "Primary" if i == 0 else f"Supervision {i}"
            logger.info(f"  Output {i} ({output_type}): {shape}")
    else:
        logger.info("Single output model")

    # Create datasets for deep supervision
    logger.info("Creating datasets for deep supervision training...")
    train_dataset = create_dataset_for_deep_supervision(
        config.train_image_dirs, config, num_outputs, is_training=True
    )
    val_dataset = create_dataset_for_deep_supervision(
        config.val_image_dirs, config, num_outputs, is_training=False
    )

    # Calculate steps per epoch
    if config.steps_per_epoch is not None:
        steps_per_epoch = config.steps_per_epoch
    else:
        total_patches = train_file_count * config.patches_per_image
        steps_per_epoch = max(100, total_patches // config.batch_size)

    logger.info(f"Using {steps_per_epoch} steps per epoch")

    # Create deep supervision model wrapper
    ds_model = DeepSupervisionModel(base_model, config)

    # Create optimizer and learning rate schedule
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

    lr_schedule = learning_rate_schedule_builder(lr_config)
    optimizer = optimizer_builder(opt_config, lr_schedule)

    # Compile model with metrics for primary output
    ds_model.compile(
        optimizer=optimizer,
        loss='mse',  # This will be overridden by custom train_step
        metrics=[
            'mae',
            keras.metrics.RootMeanSquaredError(name='rmse'),
            psnr_metric_primary_output
        ]
    )

    logger.info(f"Deep supervision model compiled with {ds_model.base_model.count_params():,} parameters")

    # Create callbacks
    callbacks = create_callbacks(config, config.val_image_dirs, ds_model)

    # Train model
    logger.info("Starting training with deep supervision...")
    start_time = time.time()

    validation_steps = config.validation_steps or max(50, steps_per_epoch // 20)

    history = ds_model.fit(
        train_dataset,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # ---------------------------------------------------------------------
    # Save clean inference model (single output, flexible spatial dimensions)
    # ---------------------------------------------------------------------
    logger.info("Creating and saving clean inference model...")

    try:
        # Create inference model with flexible spatial dimensions
        inference_input_shape = (None, None, config.channels)
        inference_base_model = create_model_instance(config, inference_input_shape)

        # If the training model had deep supervision, create single-output inference model
        if model_info['has_deep_supervision']:
            inference_base_model = create_bfunet_variant(
                variant=config.model_type if config.model_type in BFUNET_CONFIGS else 'custom',
                input_shape=inference_input_shape,
                enable_deep_supervision=False,  # Disable for inference
                **({
                    'depth': config.depth,
                    'initial_filters': config.filters,
                    'blocks_per_level': config.blocks_per_level,
                    'kernel_size': config.kernel_size,
                    'activation': config.activation
                } if config.model_type == 'custom' else {})
            )

        # Copy weights from the trained model
        inference_base_model.set_weights(ds_model.base_model.get_weights())

        # Save the clean inference model
        inference_model_path = output_dir / "inference_model.keras"
        inference_base_model.save(inference_model_path)

        logger.info(f"Clean inference model saved to: {inference_model_path}")
        logger.info(f"Inference model input shape: {inference_input_shape}")
        logger.info("Inference model has single output (primary output only)")

        # Clean up inference model from memory
        del inference_base_model

    except Exception as e:
        logger.error(f"Failed to save clean inference model: {e}")

    # Save training history
    try:
        history_path = output_dir / "training_history.json"
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save training history: {e}")

    # Save deep supervision schedule information
    if config.enable_deep_supervision and ds_model.ds_scheduler is not None:
        try:
            # Create visualization of the weight schedule
            progress_points = np.linspace(0.0, 1.0, 100)
            schedule_data = {
                'schedule_type': config.deep_supervision_schedule,
                'schedule_config': config.deep_supervision_config,
                'num_outputs': num_outputs,
                'progress_points': progress_points.tolist(),
                'weight_evolution': [ds_model.ds_scheduler(p).tolist() for p in progress_points]
            }

            schedule_path = output_dir / "deep_supervision_schedule.json"
            with open(schedule_path, 'w') as f:
                json.dump(schedule_data, f, indent=2)

            logger.info(f"Deep supervision schedule saved to: {schedule_path}")

        except Exception as e:
            logger.warning(f"Failed to save deep supervision schedule: {e}")

    # Clean up memory
    gc.collect()

    return ds_model


# ---------------------------------------------------------------------
# ARGUMENT PARSING (Updated for Deep Supervision)
# ---------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for deep supervision training."""
    parser = argparse.ArgumentParser(
        description='Train Bias-Free U-Net Denoiser with Deep Supervision',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--model-type',
        choices=['tiny', 'small', 'base', 'large', 'xlarge', 'custom'],
        default='tiny',
        help='Model architecture type'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--patch-size',
        type=int,
        default=128,
        help='Size of training patches'
    )
    parser.add_argument(
        '--channels',
        type=int,
        choices=[1, 3],
        default=1,
        help='Number of input channels (1=grayscale, 3=RGB)'
    )

    # Deep supervision parameters
    parser.add_argument(
        '--enable-deep-supervision',
        action='store_true',
        default=True,
        help='Enable deep supervision training'
    )
    parser.add_argument(
        '--no-deep-supervision',
        dest='enable_deep_supervision',
        action='store_false',
        help='Disable deep supervision training'
    )
    parser.add_argument(
        '--ds-schedule',
        type=str,
        choices=[
            'constant_equal',
            'constant_low_to_high',
            'constant_high_to_low',
            'linear_low_to_high',
            'non_linear_low_to_high',
            'custom_sigmoid_low_to_high',
            'scale_by_scale_low_to_high',
            'cosine_annealing',
            'curriculum'
        ],
        default='linear_low_to_high',
        help='Deep supervision weight schedule'
    )

    # Output parameters
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name (auto-generated if not provided)'
    )

    # Other parameters
    parser.add_argument(
        '--patches-per-image',
        type=int,
        default=4,
        help='Number of patches to extract per image'
    )
    parser.add_argument(
        '--monitor-every',
        type=int,
        default=1,
        help='Save intermediate results every N epochs'
    )
    parser.add_argument(
        '--max-train-files',
        type=int,
        default=None,
        help='Maximum number of training files to use'
    )

    # Synthesis parameters
    parser.add_argument(
        '--enable-synthesis',
        action='store_true',
        default=True,
        help='Enable image synthesis during monitoring'
    )
    parser.add_argument(
        '--no-synthesis',
        dest='enable_synthesis',
        action='store_false',
        help='Disable image synthesis during monitoring'
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------

def main():
    """Main training function with deep supervision support."""
    # Parse command line arguments
    args = parse_arguments()

    # Configuration for training with deep supervision
    config = TrainingConfig(
        # Data paths - modify these for your setup
        train_image_dirs=[
            '/media/arxwn/data0_4tb/datasets/Megadepth',
            '/media/arxwn/data0_4tb/datasets/div2k/train',
            '/media/arxwn/data0_4tb/datasets/WFLW/images',
            '/media/arxwn/data0_4tb/datasets/bdd_data/train',
            '/media/arxwn/data0_4tb/datasets/COCO/train2017',
            '/media/arxwn/data0_4tb/datasets/VGG-Face2/data/train'
        ],
        val_image_dirs=[
            '/media/arxwn/data0_4tb/datasets/div2k/validation',
            '/media/arxwn/data0_4tb/datasets/COCO/val2017',
        ],

        # Training parameters from arguments
        patch_size=args.patch_size,
        channels=args.channels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patches_per_image=args.patches_per_image,

        # File limits for manageable training
        max_train_files=args.max_train_files,
        max_val_files=10000,
        parallel_reads=8,
        dataset_shuffle_buffer=1013,

        # Model configuration
        model_type=args.model_type,

        # Deep supervision configuration
        enable_deep_supervision=args.enable_deep_supervision,
        deep_supervision_schedule=args.ds_schedule,
        deep_supervision_config={},  # Use defaults for selected schedule

        # Noise configuration (universal range)
        noise_sigma_min=0.0,
        noise_sigma_max=0.5,
        noise_distribution='uniform',

        # Optimization
        learning_rate=1e-3,
        optimizer_type='adamw',
        lr_schedule_type='cosine_decay',
        warmup_epochs=5,

        # Monitoring
        monitor_every_n_epochs=args.monitor_every,
        save_training_images=True,
        validation_steps=500,

        # Image Synthesis
        enable_synthesis=args.enable_synthesis,
        synthesis_samples=10,
        synthesis_steps=200,

        # Output
        output_dir=args.output_dir,
        experiment_name=args.experiment_name
    )

    # Log the configuration
    logger.info("Deep Supervision Training Configuration:")
    logger.info(f"  Model Type: {config.model_type}")
    logger.info(f"  Deep Supervision: {config.enable_deep_supervision}")
    if config.enable_deep_supervision:
        logger.info(f"  DS Schedule: {config.deep_supervision_schedule}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Learning Rate: {config.learning_rate}")
    logger.info(f"  Patch Size: {config.patch_size}")
    logger.info(f"  Channels: {config.channels}")
    logger.info(f"  Noise Range: [{config.noise_sigma_min}, {config.noise_sigma_max}]")
    logger.info(f"  Monitor Every: {config.monitor_every_n_epochs} epochs")
    logger.info(f"  Image Synthesis: {'Enabled' if config.enable_synthesis else 'Disabled'}")
    logger.info(f"  Output Directory: {config.output_dir}")
    logger.info(f"  Experiment Name: {config.experiment_name}")

    try:
        model = train_bfunet_denoiser_with_deep_supervision(config)
        logger.info("Training completed successfully!")

        # Log model information
        model_info = get_model_output_info(model.base_model)
        logger.info(f"Final model has {model_info['num_outputs']} outputs")
        logger.info(f"Model parameters: {model.base_model.count_params():,}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()