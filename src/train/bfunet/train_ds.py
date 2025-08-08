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
from typing import Tuple, List, Optional, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.filesystem import count_available_files
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder, deep_supervision_schedule_builder
from dl_techniques.models.bfunet_denoiser import (
    create_bfunet_denoiser, BFUNET_CONFIGS, create_bfunet_variant
)

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for bias-free U-Net denoiser training with deep supervision support."""

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
    deep_supervision_schedule_type: str = 'linear_low_to_high'  # Deep supervision schedule
    deep_supervision_schedule_config: Dict[str, Any] = field(default_factory=dict)  # Schedule parameters

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
# DATASET BUILDER (unchanged from original)
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
    noise = tf.random.normal(tf.shape(patch)) * noise_level
    noisy_patch = patch + noise

    # Clip to valid range for [0, 1] normalized input
    noisy_patch = tf.clip_by_value(noisy_patch, 0.0, 1.0)

    return noisy_patch, patch


def create_dataset(directories: List[str], config: TrainingConfig, is_training: bool = True) -> tf.data.Dataset:
    """
    Create a dataset using unified file list approach to ensure uniform sampling.

    Args:
        directories: List of directories containing images
        config: Training configuration
        is_training: Whether this is a training dataset

    Returns:
        Configured tf.data.Dataset
    """
    logger.info(f"Creating {'training' if is_training else 'validation'} dataset from directories: {directories}")

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

    # Ensure final shape consistency
    dataset = dataset.map(
        lambda noisy, clean: (
            tf.ensure_shape(noisy, [config.patch_size, config.patch_size, config.channels]),
            tf.ensure_shape(clean, [config.patch_size, config.patch_size, config.channels])
        )
    )

    # Batching and prefetching
    dataset = dataset.prefetch(config.batch_size * 2)
    dataset = dataset.batch(config.batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ---------------------------------------------------------------------
# DEEP SUPERVISION LOSS FUNCTION (REVISED)
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ScaledMseLoss(keras.losses.Loss):
    """
    Computes Mean Squared Error, resizing y_true to match y_pred's shape.
    This is necessary for deep supervision where prediction tensors have
    different spatial dimensions than the ground truth.
    """

    def __init__(self, name="scaled_mse_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculates MSE loss after resizing y_true.

        Args:
            y_true: The ground truth tensor (full resolution).
            y_pred: The predicted tensor (potentially downscaled).

        Returns:
            The MSE loss.
        """
        pred_shape = tf.shape(y_pred)
        target_height, target_width = pred_shape[1], pred_shape[2]

        y_true_resized = tf.image.resize(y_true, (target_height, target_width))

        return tf.reduce_mean(tf.square(y_pred - y_true_resized))

# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
def psnr_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    PSNR (Peak Signal-to-Noise Ratio) metric for image denoising.
    Updated for [0, 1] normalization range.

    Args:
        y_true: Ground truth images in [0, 1] range
        y_pred: Predicted/denoised images in [0, 1] range

    Returns:
        Mean PSNR value across the batch
    """
    return tf.reduce_mean(tf.image.psnr(y_pred, y_true, max_val=1.0))


class PrimaryOutputPSNR(keras.metrics.Metric):
    """PSNR metric that only considers the primary output (index 0) for multi-output models."""

    def __init__(self, name: str = 'primary_psnr', **kwargs):
        super().__init__(name=name, **kwargs)
        self.psnr_sum = self.add_weight(name='psnr_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true: Union[tf.Tensor, List[tf.Tensor]],
                     y_pred: Union[tf.Tensor, List[tf.Tensor]],
                     sample_weight: Optional[tf.Tensor] = None):
        """Update PSNR state using only primary output."""
        # y_true and y_pred are now perfectly matching lists
        if isinstance(y_pred, list):
            primary_pred = y_pred[0]
            primary_true = y_true[0]
        else:
            primary_pred = y_pred
            primary_true = y_true

        psnr_batch = tf.image.psnr(primary_pred, primary_true, max_val=1.0)
        self.psnr_sum.assign_add(tf.reduce_sum(psnr_batch))
        self.count.assign_add(tf.cast(tf.size(psnr_batch), tf.float32))

    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self.psnr_sum, self.count)

    def reset_state(self):
        self.psnr_sum.assign(0.0)
        self.count.assign(0.0)

# ---------------------------------------------------------------------
# IMAGE SYNTHESIS USING IMPLICIT PRIOR (unchanged from original)
# ---------------------------------------------------------------------

def unconditional_sampling(
    denoiser: keras.Model,
    num_samples: int = 4,
    image_shape: Tuple[int, int, int] = (64, 64, 1),
    num_steps: int = 200,
    initial_step_size: float = 0.1,
    final_step_size: float = 1.0,
    initial_noise_level: float = 0.5,
    final_noise_level: float = 0.01,
    seed: Optional[int] = None,
    save_intermediate: bool = True
) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    """
    Unconditional image synthesis using the denoiser's implicit prior.
    For multi-output models, only uses the primary output (index 0).

    Implements Algorithm 1: Unconditional Sampling from the Kadkhodaie & Simoncelli paper.
    This performs stochastic coarse-to-fine gradient ascent to generate natural images
    from random noise using the learned implicit prior.

    Args:
        denoiser: Trained bias-free denoiser model
        num_samples: Number of images to generate
        image_shape: Shape of generated images (height, width, channels)
        num_steps: Number of sampling steps
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

    logger.info(f"Starting unconditional sampling: {num_samples} samples, {num_steps} steps")

    # Initialize with random noise y_0
    # Start with high noise level to be far from natural image manifold
    y = tf.random.normal([num_samples] + list(image_shape), mean=0.5, stddev=0.3)
    y = tf.clip_by_value(y, 0.0, 1.0)

    intermediate_steps = []

    # Coarse-to-fine scheduling
    step_sizes = tf.linspace(initial_step_size, final_step_size, num_steps)
    noise_levels = tf.linspace(initial_noise_level, final_noise_level, num_steps)

    for t in range(num_steps):
        # Current scheduling parameters
        h_t = step_sizes[t]  # Step size (increases over time)
        gamma_t = noise_levels[t]  # Noise level (decreases over time)

        # Compute denoiser residual: d_t = D(y_t) - y_t
        # This is proportional to ∇_y log p(y_t) according to Miyasawa's theorem
        model_output = denoiser(y, training=False)

        # Handle multi-output models - use only primary output for synthesis
        if isinstance(model_output, list):
            denoised = model_output[0]  # Primary output
        else:
            denoised = model_output

        d_t = denoised - y

        # Generate fresh Gaussian noise
        z_t = tf.random.normal(tf.shape(y))

        # Update rule: y_{t+1} = y_t + h_t * d_t + γ_t * z_t
        y = y + h_t * d_t + gamma_t * z_t

        # Clip to valid [0, 1] range
        y = tf.clip_by_value(y, 0.0, 1.0)

        # Save intermediate steps for visualization
        if save_intermediate and (t % (num_steps // 10) == 0 or t == num_steps - 1):
            intermediate_steps.append(y.numpy().copy())

        # Log progress
        if t % (num_steps // 5) == 0:
            mean_intensity = tf.reduce_mean(y)
            std_intensity = tf.math.reduce_std(y)
            logger.info(f"Step {t}/{num_steps}: mean={mean_intensity:.3f}, std={std_intensity:.3f}, "
                       f"h_t={h_t:.4f}, γ_t={gamma_t:.4f}")

    logger.info(f"Unconditional sampling completed. Generated {num_samples} samples.")

    return y, intermediate_steps


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
    try:
        num_samples = min(4, final_samples.shape[0])
        num_steps = len(intermediate_steps)

        # Create figure showing synthesis evolution
        fig, axes = plt.subplots(num_samples, num_steps, figsize=(3 * num_steps, 3 * num_samples))
        fig.suptitle(f'Image Synthesis Evolution - Epoch {epoch}\n'
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
# MONITORING AND CALLBACKS
# ---------------------------------------------------------------------

class MetricsVisualizationCallback(keras.callbacks.Callback):
    """Callback to visualize training and validation metrics."""

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
            'primary_psnr': []
        }
        self.val_metrics = {
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'val_primary_psnr': []
        }

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Update metrics and create visualization plots."""
        if logs is None:
            logs = {}

        # Store training metrics
        for key in self.train_metrics.keys():
            if key in logs:
                self.train_metrics[key].append(logs[key])

        # Store validation metrics
        for key in self.val_metrics.keys():
            if key in logs:
                self.val_metrics[key].append(logs[key])

        # Create plots every few epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            self._create_metrics_plots(epoch + 1)

    def _create_metrics_plots(self, epoch: int):
        """Create and save metrics visualization plots."""
        try:
            epochs_range = range(1, len(self.train_metrics['loss']) + 1)

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Training and Validation Metrics - Epoch {epoch}', fontsize=16)

            # MSE (Loss) Plot
            axes[0, 0].plot(epochs_range, self.train_metrics['loss'], 'b-', label='Training MSE', linewidth=2)
            if self.val_metrics['val_loss']:
                axes[0, 0].plot(epochs_range, self.val_metrics['val_loss'], 'r-', label='Validation MSE', linewidth=2)
            axes[0, 0].set_title('Mean Squared Error (MSE)')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('MSE')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # MAE Plot
            axes[0, 1].plot(epochs_range, self.train_metrics['mae'], 'b-', label='Training MAE', linewidth=2)
            if self.val_metrics['val_mae']:
                axes[0, 1].plot(epochs_range, self.val_metrics['val_mae'], 'r-', label='Validation MAE', linewidth=2)
            axes[0, 1].set_title('Mean Absolute Error (MAE)')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # RMSE Plot
            axes[1, 0].plot(epochs_range, self.train_metrics['rmse'], 'b-', label='Training RMSE', linewidth=2)
            if self.val_metrics['val_rmse']:
                axes[1, 0].plot(epochs_range, self.val_metrics['val_rmse'], 'r-', label='Validation RMSE', linewidth=2)
            axes[1, 0].set_title('Root Mean Squared Error (RMSE)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('RMSE')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # PSNR Plot
            axes[1, 1].plot(epochs_range, self.train_metrics['primary_psnr'], 'b-', label='Training PSNR', linewidth=2)
            if self.val_metrics['val_primary_psnr']:
                axes[1, 1].plot(epochs_range, self.val_metrics['val_primary_psnr'], 'r-', label='Validation PSNR', linewidth=2)
            axes[1, 1].set_title('Peak Signal-to-Noise Ratio (PSNR)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('PSNR (dB)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = self.visualization_dir / f"epoch_{epoch:03d}_metrics.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            plt.clf()
            gc.collect()

            # Also save the latest metrics data
            metrics_data = {
                'epoch': epoch,
                'train_metrics': self.train_metrics,
                'val_metrics': self.val_metrics
            }
            metrics_file = self.visualization_dir / "latest_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

        except Exception as e:
            logger.warning(f"Failed to create metrics plots: {e}")


class StreamingResultMonitor(keras.callbacks.Callback):
    """Memory-efficient monitoring using streaming data with deep supervision support."""

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

                # Denoise images
                model_output = self.model(noisy_images, training=False)

                # Handle multi-output models - use primary output for evaluation
                if isinstance(model_output, list):
                    denoised_images = model_output[0]  # Primary output
                    logger.info(f"Multi-output model: using primary output (shape: {denoised_images.shape})")
                else:
                    denoised_images = model_output

                # Save sample images
                if self.config.save_training_images:
                    self._save_image_samples(
                        epoch_val,
                        noisy_images,
                        clean_images,
                        denoised_images
                    )

                # Compute metrics - corrected for [0, 1] range
                mse_loss = tf.reduce_mean(tf.square(denoised_images - clean_images))
                psnr = tf.reduce_mean(tf.image.psnr(denoised_images, clean_images, max_val=1.0))

                logger.info(f"Epoch {epoch_val} - Validation MSE: {mse_loss.numpy():.6f}, PSNR: {psnr.numpy():.2f} dB")

                # === IMAGE SYNTHESIS USING IMPLICIT PRIOR ===
                if self.config.enable_synthesis:
                    logger.info(f"Generating synthetic images using implicit prior...")
                    try:
                        # Generate images using the denoiser's implicit prior
                        synthesis_shape = (self.config.patch_size, self.config.patch_size, self.config.channels)
                        generated_samples, intermediate_steps = unconditional_sampling(
                            denoiser=self.model,
                            num_samples=self.config.synthesis_samples,
                            image_shape=synthesis_shape,
                            num_steps=self.config.synthesis_steps,
                            initial_step_size=self.config.synthesis_initial_step_size,
                            final_step_size=self.config.synthesis_final_step_size,
                            initial_noise_level=self.config.synthesis_initial_noise,
                            final_noise_level=self.config.synthesis_final_noise,
                            seed=epoch_val,  # Use epoch as seed for reproducibility
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

                        # Compute synthesis quality metrics
                        generated_mean = tf.reduce_mean(generated_samples)
                        generated_std = tf.math.reduce_std(generated_samples)
                        generated_range = tf.reduce_max(generated_samples) - tf.reduce_min(generated_samples)

                        logger.info(f"Generated images - Mean: {generated_mean:.3f}, "
                                   f"Std: {generated_std:.3f}, Range: {generated_range:.3f}")

                        # Save synthesis metrics
                        synthesis_metrics = {
                            'epoch': epoch_val,
                            'synthesis_mean': float(generated_mean),
                            'synthesis_std': float(generated_std),
                            'synthesis_range': float(generated_range),
                            'num_synthesis_steps': self.config.synthesis_steps,
                            'num_samples': self.config.synthesis_samples,
                            'timestamp': datetime.now().isoformat()
                        }
                        synthesis_metrics_file = self.results_dir / f"epoch_{epoch_val:03d}_synthesis_metrics.json"
                        with open(synthesis_metrics_file, 'w') as f:
                            json.dump(synthesis_metrics, f, indent=2)

                    except Exception as synthesis_error:
                        logger.warning(f"Image synthesis failed at epoch {epoch_val}: {synthesis_error}")
                else:
                    logger.info("Image synthesis disabled in configuration")

                # Save denoising metrics
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
            fig.suptitle(f'Denoising Results - Epoch {epoch}', fontsize=20, y=0.98)

            for i in range(10):
                if i < num_samples:
                    # Images are already in [0, 1] range, no need for denormalization
                    clean_img = clean[i].numpy()
                    noisy_img = noisy[i].numpy()
                    denoised_img = denoised[i].numpy()

                    # Ensure values are in [0, 1] range
                    clean_img = np.clip(clean_img, 0.0, 1.0)
                    noisy_img = np.clip(noisy_img, 0.0, 1.0)
                    denoised_img = np.clip(denoised_img, 0.0, 1.0)

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

                    # Bottom row: Denoised images
                    axes[2, i].imshow(denoised_img, cmap=cmap, vmin=0, vmax=1)
                    if i == 0:
                        axes[2, i].set_ylabel('Denoised', fontsize=14, rotation=0, ha='right', va='center')
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


class DeepSupervisionWeightScheduler(keras.callbacks.Callback):
    """
    Callback to dynamically update loss_weights for deep supervision during training.
    """

    # The __init__ signature changes back to taking num_outputs
    def __init__(self, config: TrainingConfig, num_outputs: int):
        super().__init__()
        self.config = config
        self.num_outputs = num_outputs
        self.total_epochs = config.epochs

        ds_config = {
            'type': config.deep_supervision_schedule_type,
            'config': config.deep_supervision_schedule_config
        }
        self.scheduler = deep_supervision_schedule_builder(ds_config, self.num_outputs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Update the model's loss_weights at the start of each epoch."""
        progress = min(1.0, epoch / max(1, self.total_epochs - 1))

        # The scheduler returns a list (or numpy array) of weights
        new_weights = self.scheduler(progress)

        # Update the model's loss_weights directly with the list
        self.model.loss_weights = new_weights

        weights_str = ", ".join([f"{w:.4f}" for w in new_weights])
        logger.info(f"Epoch {epoch + 1}/{self.total_epochs} - Updated DS weights: [{weights_str}]")


def create_callbacks(config: TrainingConfig, val_directories: List[str], num_outputs: int) -> List[keras.callbacks.Callback]:
    """Create training callbacks."""
    callbacks = []

    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deep supervision weight scheduling callback
    if config.enable_deep_supervision and num_outputs > 1:
        # Instantiate with num_outputs
        callbacks.append(DeepSupervisionWeightScheduler(config, num_outputs))

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
# MODEL CREATION HELPER
# ---------------------------------------------------------------------

def create_model_instance(config: TrainingConfig, input_shape: Tuple[int, int, int]) -> keras.Model:
    """
    Create a model instance based on configuration.

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
# TRAINING FUNCTION
# ---------------------------------------------------------------------

def train_bfunet_denoiser(config: TrainingConfig) -> keras.Model:
    """
    Train a bias-free U-Net denoiser model with deep supervision support.

    This function orchestrates the entire training pipeline, including:
    1.  Model creation and inspection of its multi-output structure.
    2.  Dataset creation.
    3.  **Critical Adaptation**: A dataset mapping function is created to generate a
        tuple of multi-scale ground truth labels that perfectly match the model's
        output tensors in both structure and shape.
    4.  Setting up the optimizer, loss functions, and metrics.
    5.  Compiling the model in a way that Keras can handle the multi-output structure,
        specifically by targeting metrics to the primary output layer by name.
    6.  Executing the training loop with callbacks for monitoring and scheduling.
    7.  Saving the final, clean, single-output model for inference.

    Args:
        config: Training configuration dataclass instance.

    Returns:
        The trained Keras model, which holds the best weights from the training run.
    """
    logger.info("======================================================================")
    logger.info("===    STARTING BIAS-FREE U-NET DENOISER TRAINING PIPELINE     ===")
    logger.info("======================================================================")
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Deep Supervision: {'ENABLED' if config.enable_deep_supervision else 'DISABLED'}")
    if config.enable_deep_supervision:
        logger.info(f"  - Schedule: {config.deep_supervision_schedule_type}")

    # --- 1. Setup Output Directory and Save Configuration ---
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    logger.info(f"Results will be saved to: {output_dir}")

    # --- 2. Validate and Count Data Files ---
    logger.info("Validating and counting data files...")
    for directory in config.train_image_dirs:
        if not Path(directory).exists():
            logger.warning(f"Training directory does not exist: {directory}")
    for directory in config.val_image_dirs:
        if not Path(directory).exists():
            logger.warning(f"Validation directory does not exist: {directory}")
    logger.info(f"Training directories: {config.train_image_dirs}")
    logger.info(f"Validation directories: {config.val_image_dirs}")
    try:
        train_file_count = count_available_files(
            config.train_image_dirs, config.image_extensions, config.max_train_files
        )
        val_file_count = count_available_files(
            config.val_image_dirs, config.image_extensions, config.max_val_files
        )
    except Exception as e:
        logger.warning(f"Error counting files: {e}")
        train_file_count, val_file_count = 1000, 100
    if train_file_count == 0: raise ValueError("No training files found!")
    if val_file_count == 0: raise ValueError("No validation files found!")
    logger.info(f"Found approximately {train_file_count} training files and {val_file_count} validation files.")

    # --- 3. Create the Keras Model ---
    logger.info(f"Creating BFU-Net model variant: '{config.model_type}'...")
    input_shape = (config.patch_size, config.patch_size, config.channels)
    model = create_model_instance(config, input_shape)
    model.summary()

    has_multiple_outputs = isinstance(model.output, list)
    num_outputs = len(model.output) if has_multiple_outputs else 1
    logger.info(f"Model created with {num_outputs} output(s).")

    # --- 4. Create and Adapt Datasets ---
    logger.info("Creating and preparing datasets...")
    train_dataset = create_dataset(config.train_image_dirs, config, is_training=True)
    val_dataset = create_dataset(config.val_image_dirs, config, is_training=False)

    if has_multiple_outputs:
        concrete_output_dims = [(out.shape[1], out.shape[2]) for out in model.output]
        logger.info(f"Adapting dataset for multi-output model with target dimensions: {concrete_output_dims}")

        def create_multiscale_labels(noisy_patch, clean_patch):
            labels = [tf.image.resize(clean_patch, dim) for dim in concrete_output_dims]
            return noisy_patch, tuple(labels) # Return a tuple to prevent tf.data from stacking

        train_dataset = train_dataset.map(create_multiscale_labels, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(create_multiscale_labels, num_parallel_calls=tf.data.AUTOTUNE)

    # --- 5. Setup Optimizer, Losses, and Metrics ---
    if config.steps_per_epoch is not None:
        steps_per_epoch = config.steps_per_epoch
    else:
        total_patches = train_file_count * config.patches_per_image
        steps_per_epoch = max(100, total_patches // config.batch_size)
    logger.info(f"Calculated {steps_per_epoch} steps per epoch.")

    lr_config = {
        'type': config.lr_schedule_type, 'learning_rate': config.learning_rate,
        'decay_steps': steps_per_epoch * config.epochs, 'warmup_steps': steps_per_epoch * config.warmup_epochs,
        'alpha': 0.01
    }
    opt_config = {'type': config.optimizer_type, 'gradient_clipping_by_norm': config.gradient_clipping}
    optimizer = optimizer_builder(opt_config, learning_rate_schedule_builder(lr_config))

    # ** THE FIX FOR THE METRICS ERROR **
    if has_multiple_outputs:
        loss_fns = ['mse'] * num_outputs
        initial_weights = [1.0 / num_outputs] * num_outputs

        # Define the metrics we care about.
        metrics_for_primary_output = [
            'mae',
            keras.metrics.RootMeanSquaredError(name='rmse'),
            PrimaryOutputPSNR()
        ]

        # Assign this list of metrics ONLY to the 'final_output' layer by name.
        # This resolves the "list length mismatch" error.
        metrics = {'final_output': metrics_for_primary_output}
        logger.info("Metrics will be calculated for the 'final_output' layer only.")

    else: # Single-output case
        loss_fns = 'mse'
        initial_weights = None
        metrics = [
            'mae',
            keras.metrics.RootMeanSquaredError(name='rmse'),
            PrimaryOutputPSNR()
        ]

    # --- 6. Compile the Model ---
    logger.info("Compiling model...")
    model.compile(
        optimizer=optimizer,
        loss=loss_fns,
        loss_weights=initial_weights,
        metrics=metrics # Pass the dictionary of metrics
    )
    logger.info("Model compiled successfully.")

    # --- 7. Setup Callbacks and Start Training ---
    callbacks = create_callbacks(config, config.val_image_dirs, num_outputs)

    logger.info("======================================================================")
    logger.info(f"===              STARTING TRAINING FOR {config.epochs} EPOCHS              ===")
    logger.info("======================================================================")
    start_time = time.time()
    validation_steps = config.validation_steps or max(50, steps_per_epoch // 20)

    history = model.fit(
        train_dataset,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time / 3600:.2f} hours.")

    # --- 8. Save Final Inference Model ---
    logger.info("Creating and saving final single-output inference model...")
    try:
        inference_input_shape = (None, None, config.channels)
        if config.model_type in BFUNET_CONFIGS:
            inference_model = create_bfunet_variant(
                variant=config.model_type, input_shape=inference_input_shape, enable_deep_supervision=False
            )
        else: # Custom model
            inference_model = create_bfunet_denoiser(
                input_shape=inference_input_shape, depth=config.depth, initial_filters=config.filters,
                blocks_per_level=config.blocks_per_level, kernel_size=config.kernel_size,
                activation=config.activation, enable_deep_supervision=False,
                model_name=f'bfunet_custom_{config.experiment_name}_inference'
            )

        inference_model.set_weights(model.get_weights())
        logger.info("Successfully copied best weights to inference model.")

        inference_model_path = output_dir / "inference_model.keras"
        inference_model.save(inference_model_path)
        logger.info(f"Clean inference model saved to: {inference_model_path}")
        del inference_model
    except Exception as e:
        logger.error(f"Failed to save clean inference model: {e}", exc_info=True)
        raise

    # --- 9. Save Training History ---
    try:
        history_path = output_dir / "training_history.json"
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save training history: {e}")

    gc.collect()
    logger.info("======================================================================")
    logger.info("===                  TRAINING PIPELINE COMPLETED                 ===")
    logger.info("======================================================================")
    return model

# ---------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
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
        '--deep-supervision-schedule',
        choices=['constant_equal', 'constant_low_to_high', 'constant_high_to_low',
                'linear_low_to_high', 'non_linear_low_to_high', 'custom_sigmoid_low_to_high',
                'scale_by_scale_low_to_high', 'cosine_annealing', 'curriculum'],
        default='linear_low_to_high',
        help='Deep supervision weight schedule type'
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
        '--early-stopping-patience',
        type=int,
        default=15,
        help='Early stopping patience'
    )

    parser.add_argument(
        '--max-train-files',
        type=int,
        default=None,
        help='Maximum number of files to read'
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
    parser.add_argument(
        '--synthesis-samples',
        type=int,
        default=10,
        help='Number of images to synthesize'
    )
    parser.add_argument(
        '--synthesis-steps',
        type=int,
        default=200,
        help='Number of synthesis iteration steps'
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------

def main():
    """Main training function with deep supervision support."""
    # Parse command line arguments
    args = parse_arguments()

    # Configuration for training
    config = TrainingConfig(
        # Data paths - you should modify these for your setup
        train_image_dirs=[
            '/media/arxwn/data0_4tb/datasets/Megadepth',
            '/media/arxwn/data0_4tb/datasets/div2k/train',
            '/media/arxwn/data0_4tb/datasets/WFLW/images',
            '/media/arxwn/data0_4tb/datasets/bdd_data/train',
            '/media/arxwn/data0_4tb/datasets/COCO/train2017'
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
        deep_supervision_schedule_type=args.deep_supervision_schedule,
        deep_supervision_schedule_config={},  # Use defaults

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
        synthesis_samples=args.synthesis_samples,
        synthesis_steps=args.synthesis_steps,

        # Output
        output_dir=args.output_dir,
        experiment_name=args.experiment_name
    )

    # Log the configuration
    logger.info("Training Configuration:")
    logger.info(f"  Model Type: {config.model_type}")
    logger.info(f"  Deep Supervision: {'Enabled' if config.enable_deep_supervision else 'Disabled'}")
    if config.enable_deep_supervision:
        logger.info(f"    - Schedule Type: {config.deep_supervision_schedule_type}")
        logger.info(f"    - Schedule Config: {config.deep_supervision_schedule_config}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Learning Rate: {config.learning_rate}")
    logger.info(f"  Patch Size: {config.patch_size}")
    logger.info(f"  Channels: {config.channels}")
    logger.info(f"  Input Normalization: [0, 1] range")
    logger.info(f"  Noise Range: [{config.noise_sigma_min}, {config.noise_sigma_max}]")
    logger.info(f"  Monitor Every: {config.monitor_every_n_epochs} epochs")
    logger.info(f"  Image Synthesis: {'Enabled' if config.enable_synthesis else 'Disabled'}")
    if config.enable_synthesis:
        logger.info(f"    - Synthesis Samples: {config.synthesis_samples}")
        logger.info(f"    - Synthesis Steps: {config.synthesis_steps}")
        logger.info(f"    - Step Size Range: [{config.synthesis_initial_step_size}, {config.synthesis_final_step_size}]")
        logger.info(f"    - Noise Range: [{config.synthesis_final_noise}, {config.synthesis_initial_noise}]")
    logger.info(f"  Output Directory: {config.output_dir}")
    logger.info(f"  Experiment Name: {config.experiment_name}")

    try:
        model = train_bfunet_denoiser(config)
        logger.info("Training completed successfully!")
        model.summary()

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()