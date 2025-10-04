"""
Bias-Free U-Net Denoiser Training Pipeline with Deep Supervision

This module implements a comprehensive training pipeline for bias-free U-Net image denoisers
with optional deep supervision support. The pipeline includes:

1. **Multi-Scale Deep Supervision**: Optional training with multiple output scales where
   the model learns from both coarse and fine resolution targets simultaneously.

2. **Implicit Prior Image Synthesis**: Implementation of the unconditional sampling
   algorithm from Kadkhodaie & Simoncelli that uses the denoiser's learned implicit
   prior to generate natural images from random noise.

3. **Adaptive Loss Weight Scheduling**: Dynamic adjustment of deep supervision weights
   during training using configurable scheduling strategies.

4. **Comprehensive Monitoring**: Real-time visualization of training metrics, sample
   denoising results, and synthetic image generation.

5. **Memory-Efficient Data Pipeline**: Streaming dataset creation with parallel loading,
   augmentation, and noise injection.

Key Components:
    - TrainingConfig: Comprehensive configuration dataclass
    - Dataset pipeline with multi-scale label generation
    - Deep supervision loss functions and scheduling
    - Image synthesis using implicit neural priors
    - Advanced monitoring and visualization callbacks

Architecture Support:
    - Multiple BFU-Net variants (tiny, small, base, large, xlarge)
    - Custom architecture configuration
    - Single and multi-output model handling

Usage:
    python train_bfunet.py --model-type base --epochs 100 --enable-deep-supervision

References:
    - Kadkhodaie & Simoncelli: "Solving inverse problems with deep networks: The implicit prior"
    - Deep Supervision techniques for multi-scale learning
    - Bias-free convolutional architectures for image restoration
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
from typing import Tuple, List, Optional, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.filesystem import count_available_files
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
    deep_supervision_schedule_builder
)
from dl_techniques.models.bias_free_denoisers.bfunet import (
    create_bfunet_denoiser,
    BFUNET_CONFIGS,
    create_bfunet_variant
)


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Comprehensive configuration for bias-free U-Net denoiser training.

    This dataclass encapsulates all parameters needed for training, including
    data paths, model architecture, optimization settings, deep supervision
    configuration, and monitoring options.

    Attributes:
        train_image_dirs: List of directories containing training images
        val_image_dirs: List of directories containing validation images
        patch_size: Size of square training patches extracted from images
        channels: Number of input channels (1=grayscale, 3=RGB)
        image_extensions: Supported image file extensions

        max_train_files: Optional limit on training files (None = no limit)
        max_val_files: Optional limit on validation files
        parallel_reads: Number of parallel file reading threads
        dataset_shuffle_buffer: Buffer size for dataset shuffling

        noise_sigma_min: Minimum noise standard deviation for training
        noise_sigma_max: Maximum noise standard deviation (universal range)
        noise_distribution: Noise sampling distribution ('uniform' or 'log_uniform')

        model_type: Architecture variant or 'custom' for custom configuration
        depth: Network depth for custom models
        blocks_per_level: Residual blocks per level for custom models
        filters: Base number of filters for custom models
        kernel_size: Convolutional kernel size
        activation: Activation function name

        enable_deep_supervision: Whether to use multi-scale supervision
        deep_supervision_schedule_type: Weight scheduling strategy
        deep_supervision_schedule_config: Schedule-specific parameters

        batch_size: Training batch size
        epochs: Total number of training epochs
        patches_per_image: Number of patches extracted per image
        augment_data: Whether to apply data augmentation
        normalize_input: Whether to normalize inputs to [0,1] range
        steps_per_epoch: Manual override for steps per epoch

        learning_rate: Initial learning rate
        optimizer_type: Optimizer type ('adam', 'adamw', etc.)
        lr_schedule_type: Learning rate schedule type
        warmup_epochs: Number of warmup epochs
        weight_decay: L2 regularization weight
        gradient_clipping: Gradient clipping threshold

        monitor_every_n_epochs: Frequency of intermediate result saving
        save_best_only: Whether to save only improved models
        early_stopping_patience: Early stopping patience in epochs
        validation_steps: Number of validation steps per epoch

        enable_synthesis: Whether to enable image synthesis monitoring
        synthesis_samples: Number of images to synthesize
        synthesis_steps: Synthesis algorithm iterations
        synthesis_initial_step_size: Initial gradient step size for synthesis
        synthesis_final_step_size: Final gradient step size for synthesis
        synthesis_initial_noise: Initial noise injection level
        synthesis_final_noise: Final noise injection level

        output_dir: Base output directory
        experiment_name: Unique experiment identifier (auto-generated if None)
        save_training_images: Whether to save sample results during training
        save_model_checkpoints: Whether to save model checkpoints
    """

    # === Data Configuration ===
    train_image_dirs: List[str] = field(default_factory=list)
    val_image_dirs: List[str] = field(default_factory=list)
    patch_size: int = 64
    channels: int = 1
    image_extensions: List[str] = field(
        default_factory=lambda: ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp']
    )

    # === Memory Management ===
    max_train_files: Optional[int] = None
    max_val_files: Optional[int] = None
    parallel_reads: int = 4
    dataset_shuffle_buffer: int = 1000

    # === Noise Configuration ===
    noise_sigma_min: float = 0.0
    noise_sigma_max: float = 0.4
    noise_distribution: str = 'uniform'

    # === Model Configuration ===
    model_type: str = 'tiny'
    depth: int = 3
    blocks_per_level: int = 2
    filters: int = 64
    kernel_size: int = 3
    activation: str = 'relu'

    # === Deep Supervision Configuration ===
    enable_deep_supervision: bool = True
    deep_supervision_schedule_type: str = 'linear_low_to_high'
    deep_supervision_schedule_config: Dict[str, Any] = field(default_factory=dict)

    # === Training Configuration ===
    batch_size: int = 32
    epochs: int = 50
    patches_per_image: int = 16
    augment_data: bool = True
    normalize_input: bool = True
    steps_per_epoch: Optional[int] = None

    # === Optimization Configuration ===
    learning_rate: float = 1e-3
    optimizer_type: str = 'adam'
    lr_schedule_type: str = 'cosine_decay'
    warmup_epochs: int = 2
    weight_decay: float = 1e-5
    gradient_clipping: float = 1.0

    # === Monitoring Configuration ===
    monitor_every_n_epochs: int = 2
    save_best_only: bool = True
    early_stopping_patience: int = 15
    validation_steps: Optional[int] = 100

    # === Image Synthesis Configuration ===
    enable_synthesis: bool = False
    synthesis_samples: int = 10
    synthesis_steps: int = 200
    synthesis_initial_step_size: float = 0.05
    synthesis_final_step_size: float = 0.8
    synthesis_initial_noise: float = 0.4
    synthesis_final_noise: float = 0.005

    # === Output Configuration ===
    output_dir: str = 'results'
    experiment_name: Optional[str] = None
    save_training_images: bool = True
    save_model_checkpoints: bool = True

    def __post_init__(self) -> None:
        """
        Initialize default values and validate configuration after instantiation.

        Raises:
            ValueError: If configuration parameters are invalid
        """
        # Generate experiment name if not provided
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ds_suffix = '_ds' if self.enable_deep_supervision else ''
            self.experiment_name = f"bfunet_{self.model_type}{ds_suffix}_{timestamp}"

        # Configuration validation
        if self.noise_sigma_min < 0 or self.noise_sigma_max <= self.noise_sigma_min:
            raise ValueError("Invalid noise sigma range: min must be >= 0 and max > min")

        if self.patch_size <= 0 or self.channels <= 0:
            raise ValueError("Invalid patch size or channel configuration: must be positive")

        if not self.train_image_dirs:
            raise ValueError("No training directories specified")

        if not self.val_image_dirs:
            raise ValueError("No validation directories specified")

        if self.noise_distribution not in ['uniform', 'log_uniform']:
            raise ValueError(f"Invalid noise distribution: {self.noise_distribution}")


# =============================================================================
# DATASET PIPELINE FUNCTIONS
# =============================================================================

def load_and_preprocess_image(image_path: tf.Tensor, config: TrainingConfig) -> tf.Tensor:
    """
    Load and preprocess a single image using TensorFlow operations.

    This function handles image loading, format conversion, normalization,
    and patch extraction within the tf.data pipeline. It includes robust
    error handling for corrupt images and automatic resizing for small images.

    Args:
        image_path: Tensor containing the file system path to an image
        config: Training configuration containing preprocessing parameters

    Returns:
        Preprocessed image patch tensor of shape [patch_size, patch_size, channels]
        normalized to [0, 1] range if config.normalize_input is True

    Note:
        Uses tf.cond for efficient conditional execution within tf.function context.
        Failed image loads return zero tensors to maintain pipeline stability.
    """
    try:
        # Read and decode image file
        image_string = tf.io.read_file(image_path)
        image = tf.image.decode_image(
            image_string,
            channels=config.channels,
            expand_animations=False
        )
        image.set_shape([None, None, config.channels])

        # Convert to float32 and optionally normalize
        image = tf.cast(image, tf.float32)
        if config.normalize_input:
            image = image / 255.0  # Normalize from [0, 255] to [0, 1]

        # Handle images smaller than patch size by resizing
        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        min_dim = tf.minimum(height, width)
        min_size = config.patch_size

        def resize_if_small() -> tf.Tensor:
            """Resize image if any dimension is smaller than patch size."""
            scale_factor = tf.cast(min_size, tf.float32) / tf.cast(min_dim, tf.float32)
            new_height = tf.cast(tf.math.ceil(tf.cast(height, tf.float32) * scale_factor), tf.int32)
            new_width = tf.cast(tf.math.ceil(tf.cast(width, tf.float32) * scale_factor), tf.int32)
            return tf.image.resize(image, [new_height, new_width])

        def identity() -> tf.Tensor:
            """Return image unchanged."""
            return image

        # Conditionally resize using tf.cond for graph compatibility
        image = tf.cond(
            tf.logical_or(height < min_size, width < min_size),
            true_fn=resize_if_small,
            false_fn=identity
        )

        # Extract random patch from processed image
        patch = tf.image.random_crop(
            image,
            [config.patch_size, config.patch_size, config.channels]
        )

        return patch

    except tf.errors.InvalidArgumentError:
        # Return black patch for failed loads to maintain pipeline stability
        logger.warning(f"Failed to load image: {image_path}")
        return tf.zeros([config.patch_size, config.patch_size, config.channels], dtype=tf.float32)


def augment_patch(patch: tf.Tensor) -> tf.Tensor:
    """
    Apply data augmentation transformations to an image patch.

    Applies geometric augmentations that preserve image statistics:
    - Random horizontal and vertical flips
    - Random 90-degree rotations

    Args:
        patch: Input image patch tensor

    Returns:
        Augmented image patch with same shape as input
    """
    # Random flips
    patch = tf.image.random_flip_left_right(patch)
    patch = tf.image.random_flip_up_down(patch)

    # Random 90-degree rotation
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    patch = tf.image.rot90(patch, k)

    return patch


def add_noise_to_patch(patch: tf.Tensor, config: TrainingConfig) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Add Gaussian noise to a clean image patch for denoising training.

    Samples noise levels according to the configured distribution and adds
    zero-mean Gaussian noise to create noisy/clean training pairs.

    Args:
        patch: Clean image patch tensor in [0, 1] range
        config: Training configuration specifying noise parameters

    Returns:
        Tuple of (noisy_patch, clean_patch) where both tensors are clipped
        to valid [0, 1] range

    Note:
        Supports both uniform and log-uniform noise level sampling for
        different noise regime emphasis during training.
    """
    # Sample noise level according to configured distribution
    if config.noise_distribution == 'uniform':
        noise_level = tf.random.uniform([], config.noise_sigma_min, config.noise_sigma_max)
    elif config.noise_distribution == 'log_uniform':
        # Log-uniform sampling emphasizes lower noise levels
        log_min = tf.math.log(tf.maximum(config.noise_sigma_min, 1e-6))
        log_max = tf.math.log(config.noise_sigma_max)
        log_noise = tf.random.uniform([], log_min, log_max)
        noise_level = tf.exp(log_noise)
    else:
        raise ValueError(f"Unknown noise distribution: {config.noise_distribution}")

    # Generate and add Gaussian noise
    noise = tf.random.normal(tf.shape(patch)) * noise_level
    noisy_patch = patch + noise

    # Clip to valid range for normalized input
    noisy_patch = tf.clip_by_value(noisy_patch, 0.0, 1.0)

    return noisy_patch, patch


def create_dataset(
    directories: List[str],
    config: TrainingConfig,
    is_training: bool = True
) -> tf.data.Dataset:
    """
    Create a unified tf.data.Dataset from multiple image directories.

    This function builds a comprehensive dataset pipeline with:
    - Unified file listing from multiple directories
    - Optional file count limiting
    - Parallel image loading and preprocessing
    - Data augmentation for training sets
    - Noise injection for denoising pairs
    - Efficient batching and prefetching

    Args:
        directories: List of directory paths containing images
        config: Training configuration
        is_training: Whether to apply training-specific transformations

    Returns:
        Configured tf.data.Dataset yielding (noisy, clean) image pairs

    Raises:
        ValueError: If no valid image files are found in directories

    Note:
        The unified file listing approach ensures balanced sampling across
        directories regardless of their individual file counts.
    """
    logger.info(f"Creating {'training' if is_training else 'validation'} dataset from {len(directories)} directories")

    # Build unified file list from all directories
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

    logger.info(f"Found {len(all_file_paths)} total files")

    # Apply file count limits if specified
    limit = config.max_train_files if is_training else config.max_val_files
    if limit and limit < len(all_file_paths):
        logger.info(f"Limiting to {limit} files per configuration")
        # Shuffle before limiting for random subset selection
        np.random.shuffle(all_file_paths)
        all_file_paths = all_file_paths[:limit]

    # Create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices(all_file_paths)

    # Apply dataset transformations based on usage type
    if is_training:
        dataset = dataset.shuffle(
            buffer_size=min(config.dataset_shuffle_buffer, len(all_file_paths)),
            reshuffle_each_iteration=True
        )
        dataset = dataset.repeat()  # Infinite repetition for training
    else:
        dataset = dataset.repeat()  # Prevent OutOfRangeError during validation

    # Generate multiple patches per image for training efficiency
    if is_training and config.patches_per_image > 1:
        dataset = dataset.flat_map(
            lambda path: tf.data.Dataset.from_tensors(path).repeat(config.patches_per_image)
        )

    # Load and preprocess images with parallel execution
    dataset = dataset.map(
        lambda path: load_and_preprocess_image(path, config),
        num_parallel_calls=config.parallel_reads
    )

    # Filter out failed loads (all-zero tensors)
    dataset = dataset.filter(
        lambda x: tf.reduce_sum(tf.abs(x)) > 0
    )

    # Ensure consistent shape for downstream processing
    dataset = dataset.map(
        lambda x: tf.ensure_shape(x, [config.patch_size, config.patch_size, config.channels])
    )

    # Apply augmentation for training data
    if is_training and config.augment_data:
        dataset = dataset.map(augment_patch, num_parallel_calls=tf.data.AUTOTUNE)

    # Create noisy/clean pairs for denoising training
    dataset = dataset.map(
        lambda patch: add_noise_to_patch(patch, config),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Final shape enforcement and optimization
    dataset = dataset.map(
        lambda noisy, clean: (
            tf.ensure_shape(noisy, [config.patch_size, config.patch_size, config.channels]),
            tf.ensure_shape(clean, [config.patch_size, config.patch_size, config.channels])
        )
    )

    # Batch and prefetch for optimal performance
    dataset = dataset.prefetch(config.batch_size * 2)
    dataset = dataset.batch(config.batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# =============================================================================
# LOSS FUNCTIONS FOR DEEP SUPERVISION
# =============================================================================

@keras.saving.register_keras_serializable()
class ScaledMseLoss(keras.losses.Loss):
    """
    Mean Squared Error loss with automatic target resizing for multi-scale supervision.

    This loss function automatically resizes the ground truth tensor to match
    the prediction tensor's spatial dimensions. This is essential for deep
    supervision where intermediate outputs have different spatial resolutions
    than the full-resolution ground truth.

    The loss maintains the MSE formulation while handling the spatial mismatch
    through bilinear interpolation of the target tensor.
    """

    def __init__(self, name: str = "scaled_mse_loss", **kwargs) -> None:
        """
        Initialize the scaled MSE loss function.

        Args:
            name: Name identifier for the loss function
            **kwargs: Additional keyword arguments passed to parent class
        """
        super().__init__(name=name, **kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute MSE loss after resizing ground truth to match prediction.

        Args:
            y_true: Ground truth tensor at full resolution [batch, H, W, C]
            y_pred: Prediction tensor at potentially different resolution [batch, h, w, C]

        Returns:
            Scalar MSE loss value

        Note:
            Uses bilinear interpolation for resizing which preserves gradients
            and provides smooth loss landscapes for optimization.
        """
        pred_shape = tf.shape(y_pred)
        target_height, target_width = pred_shape[1], pred_shape[2]

        # Resize ground truth to match prediction resolution
        y_true_resized = tf.image.resize(y_true, (target_height, target_width))

        return tf.reduce_mean(tf.square(y_pred - y_true_resized))


# =============================================================================
# METRICS FOR EVALUATION
# =============================================================================

@keras.saving.register_keras_serializable()
def psnr_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) for image quality evaluation.

    PSNR is a standard metric for image restoration tasks, measuring the ratio
    between the maximum possible power of a signal and the power of corrupting
    noise that affects the fidelity of its representation.

    Args:
        y_true: Ground truth images in [0, 1] range
        y_pred: Predicted/denoised images in [0, 1] range

    Returns:
        Mean PSNR value across the batch in decibels (dB)

    Note:
        Higher PSNR values indicate better image quality. The max_val parameter
        is set to 1.0 to match the [0, 1] normalization range.
    """
    return tf.reduce_mean(tf.image.psnr(y_pred, y_true, max_val=1.0))


class PrimaryOutputPSNR(keras.metrics.Metric):
    """
    PSNR metric that evaluates only the primary output for multi-output models.

    This metric is designed for deep supervision scenarios where the model
    produces multiple outputs but we want to track the quality of only the
    main (typically highest resolution) output during training.
    """

    def __init__(self, name: str = 'primary_psnr', **kwargs) -> None:
        """
        Initialize the primary output PSNR metric.

        Args:
            name: Metric name for logging and visualization
            **kwargs: Additional keyword arguments
        """
        super().__init__(name=name, **kwargs)
        self.psnr_sum = self.add_weight(name='psnr_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(
        self,
        y_true: Union[tf.Tensor, List[tf.Tensor]],
        y_pred: Union[tf.Tensor, List[tf.Tensor]],
        sample_weight: Optional[tf.Tensor] = None
    ) -> None:
        """
        Update PSNR state using only the primary output.

        Args:
            y_true: Ground truth tensor(s), matching structure of y_pred
            y_pred: Prediction tensor(s), either single tensor or list for multi-output
            sample_weight: Optional sample weighting (currently unused)

        Note:
            For multi-output models, both y_true and y_pred are lists with
            matching structures. We extract the first element (primary output).
        """
        # Extract primary output from potentially multi-output structure
        if isinstance(y_pred, list):
            primary_pred = y_pred[0]
            primary_true = y_true[0]
        else:
            primary_pred = y_pred
            primary_true = y_true

        # Compute PSNR for the batch and accumulate
        psnr_batch = tf.image.psnr(primary_pred, primary_true, max_val=1.0)
        self.psnr_sum.assign_add(tf.reduce_sum(psnr_batch))
        self.count.assign_add(tf.cast(tf.size(psnr_batch), tf.float32))

    def result(self) -> tf.Tensor:
        """Compute the mean PSNR across all processed samples."""
        return tf.math.divide_no_nan(self.psnr_sum, self.count)

    def reset_state(self) -> None:
        """Reset metric state for new epoch or evaluation period."""
        self.psnr_sum.assign(0.0)
        self.count.assign(0.0)


# =============================================================================
# IMAGE SYNTHESIS USING IMPLICIT NEURAL PRIORS
# =============================================================================

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
    Generate images using the denoiser's implicit prior via stochastic sampling.

    Implements Algorithm 1 from Kadkhodaie & Simoncelli: "Solving inverse problems
    with deep networks: The implicit prior". This method performs coarse-to-fine
    stochastic gradient ascent in the image space, using the denoiser's learned
    implicit prior to generate natural-looking images from random noise.

    The algorithm works by iteratively:
    1. Computing the denoiser residual d_t = D(y_t) - y_t
    2. Updating via y_{t+1} = y_t + h_t * d_t + γ_t * z_t

    Where h_t increases (larger steps) and γ_t decreases (less noise injection)
    over time, implementing a coarse-to-fine sampling strategy.

    Args:
        denoiser: Trained denoiser model (handles multi-output automatically)
        num_samples: Number of images to generate simultaneously
        image_shape: Shape of generated images (height, width, channels)
        num_steps: Number of iterative sampling steps
        initial_step_size: Initial gradient step size h_0
        final_step_size: Final gradient step size h_final
        initial_noise_level: Initial noise injection level γ_0
        final_noise_level: Final noise injection level γ_final
        seed: Random seed for reproducible generation
        save_intermediate: Whether to save intermediate steps for visualization

    Returns:
        Tuple of (final_samples, intermediate_steps_list)
        - final_samples: Generated images tensor [num_samples, H, W, C]
        - intermediate_steps_list: List of intermediate tensors for visualization

    Note:
        For multi-output denoisers, automatically uses the primary (index 0) output
        for synthesis while ignoring auxiliary outputs used during training.
    """
    if seed is not None:
        tf.random.set_seed(seed)

    logger.info(f"Starting unconditional sampling: {num_samples} samples, {num_steps} steps")

    # Initialize with noise biased toward middle gray (more natural starting point)
    y = tf.random.normal([num_samples] + list(image_shape), mean=0.5, stddev=0.3)
    y = tf.clip_by_value(y, 0.0, 1.0)

    intermediate_steps = []

    # Create coarse-to-fine parameter schedules
    step_sizes = tf.linspace(initial_step_size, final_step_size, num_steps)
    noise_levels = tf.linspace(initial_noise_level, final_noise_level, num_steps)

    for t in range(num_steps):
        # Current scheduling parameters
        h_t = step_sizes[t]  # Step size (increases over time for finer details)
        gamma_t = noise_levels[t]  # Noise level (decreases over time for stability)

        # Compute denoiser residual: d_t = D(y_t) - y_t
        # This residual is proportional to ∇_y log p(y_t) according to Miyasawa's theorem
        model_output = denoiser(y, training=False)

        # Handle multi-output models by using only the primary output
        if isinstance(model_output, list):
            denoised = model_output[0]  # Primary output at full resolution
        else:
            denoised = model_output

        d_t = denoised - y

        # Generate fresh Gaussian noise for stochastic update
        z_t = tf.random.normal(tf.shape(y))

        # Stochastic gradient ascent update rule
        y = y + h_t * d_t + gamma_t * z_t

        # Maintain valid pixel range
        y = tf.clip_by_value(y, 0.0, 1.0)

        # Save intermediate steps for visualization
        if save_intermediate and (t % (num_steps // 10) == 0 or t == num_steps - 1):
            intermediate_steps.append(y.numpy().copy())

        # Log progress periodically
        if t % (num_steps // 5) == 0:
            mean_intensity = tf.reduce_mean(y)
            std_intensity = tf.math.reduce_std(y)
            logger.info(
                f"Step {t}/{num_steps}: mean={mean_intensity:.3f}, std={std_intensity:.3f}, "
                f"h_t={h_t:.4f}, γ_t={gamma_t:.4f}"
            )

    logger.info(f"Synthesis completed: generated {num_samples} samples")
    return y, intermediate_steps


def visualize_synthesis_process(
    final_samples: tf.Tensor,
    intermediate_steps: List[tf.Tensor],
    save_path: Path,
    epoch: int
) -> None:
    """
    Create visualization of the image synthesis evolution process.

    Generates a multi-panel figure showing how random noise evolves into
    natural images through the iterative sampling process. Each row shows
    a different sample, and each column shows a different time step.

    Args:
        final_samples: Final generated samples tensor
        intermediate_steps: List of intermediate sampling steps
        save_path: File path for saving the visualization
        epoch: Current training epoch for labeling

    Note:
        Handles both grayscale and RGB images automatically. Uses memory-efficient
        plotting with explicit garbage collection to prevent memory leaks during
        long training runs.
    """
    try:
        num_samples = min(4, final_samples.shape[0])
        num_steps = len(intermediate_steps)

        # Create multi-panel figure
        fig, axes = plt.subplots(num_samples, num_steps, figsize=(3 * num_steps, 3 * num_samples))
        fig.suptitle(
            f'Image Synthesis Evolution - Epoch {epoch}\n'
            f'(Random Noise → Natural Images via Implicit Prior)',
            fontsize=16, y=0.98
        )

        # Handle single row/column cases for consistent indexing
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        if num_steps == 1:
            axes = axes.reshape(-1, 1)

        for sample_idx in range(num_samples):
            for step_idx, step_data in enumerate(intermediate_steps):
                img = step_data[sample_idx]

                # Handle different image formats
                if img.shape[-1] == 1:
                    img = img.squeeze(-1)
                    cmap = 'gray'
                else:
                    cmap = None

                # Ensure valid pixel range for visualization
                img = np.clip(img, 0.0, 1.0)

                axes[sample_idx, step_idx].imshow(img, cmap=cmap, vmin=0, vmax=1)
                axes[sample_idx, step_idx].axis('off')

                # Add step labels to top row
                if sample_idx == 0:
                    step_num = step_idx * (200 // (num_steps - 1)) if step_idx < num_steps - 1 else 200
                    axes[sample_idx, step_idx].set_title(f'Step {step_num}', fontsize=10)

                # Add sample labels to first column
                if step_idx == 0:
                    axes[sample_idx, step_idx].set_ylabel(
                        f'Sample {sample_idx + 1}',
                        fontsize=12, rotation=0, ha='right', va='center'
                    )

        plt.tight_layout()
        plt.subplots_adjust(top=0.90, left=0.08)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plt.clf()
        gc.collect()

    except Exception as e:
        logger.warning(f"Failed to visualize synthesis process: {e}")


# =============================================================================
# MONITORING AND CALLBACK CLASSES
# =============================================================================

class MetricsVisualizationCallback(keras.callbacks.Callback):
    """
    Comprehensive metrics visualization callback for training monitoring.

    Creates real-time plots of training and validation metrics including:
    - Mean Squared Error (MSE) loss curves
    - Mean Absolute Error (MAE) progression
    - Root Mean Squared Error (RMSE) evolution
    - Peak Signal-to-Noise Ratio (PSNR) improvement

    Automatically saves plots and metrics data for post-training analysis.
    """

    def __init__(self, config: TrainingConfig) -> None:
        """
        Initialize the metrics visualization callback.

        Args:
            config: Training configuration containing output settings
        """
        super().__init__()
        self.config = config
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.visualization_dir = self.output_dir / "visualization_plots"
        self.visualization_dir.mkdir(parents=True, exist_ok=True)

        # Initialize storage for metrics tracking
        # Handle both single-output and multi-output model metric naming
        self.train_metrics: Dict[str, List[float]] = {
            'loss': [],
            'mae': [],
            'rmse': [],
            'primary_psnr': [],
            # Multi-output variants
            'final_output_mae': [],
            'final_output_rmse': [],
            'final_output_primary_psnr': []
        }
        self.val_metrics: Dict[str, List[float]] = {
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'val_primary_psnr': [],
            # Multi-output variants
            'val_final_output_mae': [],
            'val_final_output_rmse': [],
            'val_final_output_primary_psnr': []
        }

        # Track all available metrics dynamically
        self.available_metrics: set = set()

        logger.info(f"Metrics visualization callback initialized. Expected metrics:")
        logger.info(f"  Training: {list(self.train_metrics.keys())}")
        logger.info(f"  Validation: {list(self.val_metrics.keys())}")

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Log available metrics at the start of training for debugging.

        Args:
            logs: Training logs dictionary
        """
        if logs:
            self.available_metrics.update(logs.keys())
            logger.info(f"Available metrics at training start: {sorted(logs.keys())}")

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Track available metrics at epoch start.

        Args:
            epoch: Current epoch number
            logs: Training logs dictionary
        """
        if logs:
            self.available_metrics.update(logs.keys())
            if epoch == 0:  # Log once at first epoch
                logger.info(f"Available metrics at epoch begin: {sorted(logs.keys())}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Update metrics storage and create visualization plots.

        Args:
            epoch: Current epoch number (0-indexed)
            logs: Dictionary of metric values from training

        Note:
            Robust metric collection that handles both single-output and multi-output
            model naming conventions. Collects all available metrics dynamically.
        """
        if logs is None:
            logs = {}

        # Update available metrics set
        self.available_metrics.update(logs.keys())

        # Store all metrics that match our expected patterns
        all_expected_metrics = list(self.train_metrics.keys()) + list(self.val_metrics.keys())

        for metric_name, metric_value in logs.items():
            try:
                converted_value = float(metric_value)

                # Store training metrics
                if metric_name in self.train_metrics:
                    self.train_metrics[metric_name].append(converted_value)

                # Store validation metrics
                elif metric_name in self.val_metrics:
                    self.val_metrics[metric_name].append(converted_value)

                # Handle validation metrics that might not have 'val_' prefix in our keys
                # but do in the actual logs (e.g., val_final_output_mae)
                elif metric_name.startswith('val_'):
                    # Extract base name (e.g., 'val_final_output_mae' -> 'final_output_mae')
                    base_name = metric_name[4:]  # Remove 'val_' prefix
                    expected_key = f'val_{base_name}'
                    if expected_key in self.val_metrics:
                        self.val_metrics[expected_key].append(converted_value)

            except (ValueError, TypeError) as e:
                logger.debug(f"Failed to convert metric {metric_name}={metric_value}: {e}")

        # Log metrics collection status for debugging
        if (epoch + 1) % 5 == 0:  # Log every 5 epochs
            collected_train = [k for k, v in self.train_metrics.items() if v]
            collected_val = [k for k, v in self.val_metrics.items() if v]
            logger.debug(f"Epoch {epoch + 1}: Collected train metrics: {collected_train}")
            logger.debug(f"Epoch {epoch + 1}: Collected val metrics: {collected_val}")

        # Create visualization plots every 5 epochs or at start
        if (epoch + 1) % 5 == 0 or epoch == 0:
            self._create_metrics_plots(epoch + 1)

    def _get_metric_data(self, metric_type: str, metrics_dict: Dict[str, List[float]]) -> Tuple[List[float], str]:
        """
        Get metric data with fallback for different naming conventions.

        Args:
            metric_type: Base metric type ('mae', 'rmse', 'primary_psnr')
            metrics_dict: Dictionary of metrics to search in

        Returns:
            Tuple of (metric_data_list, actual_metric_name)
        """
        # For multi-output models, try prefixed version first
        prefixed_name = f'final_output_{metric_type}'
        if prefixed_name in metrics_dict and metrics_dict[prefixed_name]:
            return metrics_dict[prefixed_name], prefixed_name

        # Fallback to base name
        if metric_type in metrics_dict and metrics_dict[metric_type]:
            return metrics_dict[metric_type], metric_type

        # Return empty list if not found
        return [], f"{metric_type} (not found)"

    def _create_metrics_plots(self, epoch: int) -> None:
        """
        Generate and save comprehensive metrics visualization plots.

        Args:
            epoch: Current epoch number for plot labeling

        Note:
            Robust plotting that handles missing or incomplete metrics data.
            Automatically detects multi-output vs single-output metric naming.
        """
        try:
            # Check if we have any training loss data to plot
            if not self.train_metrics.get('loss', []):
                logger.debug("No training loss data available for plotting")
                return

            num_epochs = len(self.train_metrics['loss'])
            epochs_range = range(1, num_epochs + 1)

            # Create 2x2 subplot grid for different metrics
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Training and Validation Metrics - Epoch {epoch}', fontsize=16)

            # Helper function to safely plot metrics with smart name detection
            def safe_plot_metric(ax, train_type, val_type, title, ylabel, train_label, val_label):
                """Safely plot training and validation metrics with automatic name detection."""
                plots_added = False

                # Get training metric data
                train_data, train_name = self._get_metric_data(train_type, self.train_metrics)
                if train_data and len(train_data) == num_epochs:
                    ax.plot(epochs_range, train_data, 'b-',
                           label=f'{train_label} ({train_name})', linewidth=2)
                    plots_added = True

                # Get validation metric data
                val_data, val_name = self._get_metric_data(val_type, self.val_metrics)
                if val_data and len(val_data) == num_epochs:
                    ax.plot(epochs_range, val_data, 'r-',
                           label=f'{val_label} ({val_name})', linewidth=2)
                    plots_added = True

                ax.set_title(title)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(ylabel)
                if plots_added:
                    ax.legend()
                ax.grid(True, alpha=0.3)

                return plots_added

            # Plot MSE Loss (handled separately since it's always 'loss')
            axes[0, 0].set_title('Mean Squared Error (MSE)')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('MSE')

            if self.train_metrics['loss'] and len(self.train_metrics['loss']) == num_epochs:
                axes[0, 0].plot(epochs_range, self.train_metrics['loss'], 'b-',
                               label='Training MSE', linewidth=2)

            if (self.val_metrics.get('val_loss', []) and
                len(self.val_metrics['val_loss']) == num_epochs):
                axes[0, 0].plot(epochs_range, self.val_metrics['val_loss'], 'r-',
                               label='Validation MSE', linewidth=2)

            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Plot other metrics with smart detection
            safe_plot_metric(
                axes[0, 1], 'mae', 'val_mae',
                'Mean Absolute Error (MAE)', 'MAE',
                'Training MAE', 'Validation MAE'
            )

            safe_plot_metric(
                axes[1, 0], 'rmse', 'val_rmse',
                'Root Mean Squared Error (RMSE)', 'RMSE',
                'Training RMSE', 'Validation RMSE'
            )

            safe_plot_metric(
                axes[1, 1], 'primary_psnr', 'val_primary_psnr',
                'Peak Signal-to-Noise Ratio (PSNR)', 'PSNR (dB)',
                'Training PSNR', 'Validation PSNR'
            )

            plt.tight_layout()
            save_path = self.visualization_dir / f"epoch_{epoch:03d}_metrics.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            plt.clf()
            gc.collect()

            # Save comprehensive metrics data as JSON
            # Include diagnostic information about available metrics
            available_train_metrics = {k: len(v) for k, v in self.train_metrics.items() if v}
            available_val_metrics = {k: len(v) for k, v in self.val_metrics.items() if v}

            metrics_data = {
                'epoch': epoch,
                'train_metrics': self.train_metrics,
                'val_metrics': self.val_metrics,
                'diagnostics': {
                    'expected_epochs': num_epochs,
                    'available_train_metrics': available_train_metrics,
                    'available_val_metrics': available_val_metrics,
                    'all_available_metrics': sorted(list(self.available_metrics))
                }
            }
            metrics_file = self.visualization_dir / "latest_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2,
                         default=lambda x: float(x) if hasattr(x, 'item') else x)

            logger.info(f"Metrics plot saved with {len(available_train_metrics)} training and {len(available_val_metrics)} validation metrics")

        except Exception as e:
            logger.warning(f"Failed to create metrics plots: {e}")
            logger.debug(f"Available metrics: {sorted(list(self.available_metrics))}")
            logger.debug(f"Train metrics with data: {[k for k, v in self.train_metrics.items() if v]}")
            logger.debug(f"Val metrics with data: {[k for k, v in self.val_metrics.items() if v]}")


class StreamingResultMonitor(keras.callbacks.Callback):
    """
    Memory-efficient streaming monitor for denoising results and image synthesis.

    This callback provides comprehensive monitoring without storing large datasets
    in memory. It creates a small, consistent validation set for reproducible
    monitoring and handles both denoising evaluation and implicit prior synthesis.
    """

    def __init__(self, config: TrainingConfig, val_directories: List[str]) -> None:
        """
        Initialize the streaming result monitor.

        Args:
            config: Training configuration
            val_directories: List of validation directories for creating monitor set
        """
        super().__init__()
        self.config = config
        self.monitor_freq = config.monitor_every_n_epochs
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.results_dir = self.output_dir / "visualization_plots"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Create consistent monitoring dataset
        self._create_monitor_dataset(val_directories)

    def _create_monitor_dataset(self, val_directories: List[str]) -> None:
        """
        Create a small, consistent dataset for reproducible monitoring.

        Args:
            val_directories: List of validation directories to sample from

        Note:
            Limits to 10 images for memory efficiency while providing
            representative monitoring across different image types.
        """
        monitor_files = []
        extensions_set = set(ext.lower() for ext in self.config.image_extensions)
        extensions_set.update(ext.upper() for ext in self.config.image_extensions)

        # Collect representative files from validation directories
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
                logger.error(f"Error collecting monitor files from {directory}: {e}")
                continue

        if not monitor_files:
            logger.warning("No files found for monitoring")
            self.test_batch = None
            return

        # Load and preprocess monitoring images
        try:
            clean_patches = []
            for file_path in monitor_files:
                path_tensor = tf.constant(file_path)
                patch = load_and_preprocess_image(path_tensor, self.config)
                clean_patches.append(patch)

            self.test_batch = tf.stack(clean_patches)
            logger.info(f"Created monitoring dataset with shape: {self.test_batch.shape}")
        except Exception as e:
            logger.error(f"Failed to create monitor dataset: {e}")
            self.test_batch = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Perform monitoring and save results at specified intervals.

        Args:
            epoch: Current epoch number (0-indexed)
            logs: Training logs dictionary
        """
        if (epoch + 1) % self.monitor_freq != 0 or self.test_batch is None:
            return

        def _monitor_and_save(epoch_numpy: np.ndarray) -> tf.constant:
            """
            Internal monitoring function wrapped for tf.py_function compatibility.

            Args:
                epoch_numpy: Epoch number as numpy array

            Returns:
                Dummy return value for tf.py_function
            """
            epoch_val = int(epoch_numpy)
            logger.info(f"Saving intermediate results for epoch {epoch_val}")

            try:
                # Generate noisy/clean pairs for evaluation
                noisy_images, clean_images = add_noise_to_patch(self.test_batch, self.config)

                # Get model predictions
                model_output = self.model(noisy_images, training=False)

                # Handle multi-output models
                if isinstance(model_output, list):
                    denoised_images = model_output[0]  # Primary output
                    logger.info(f"Multi-output model: using primary output (shape: {denoised_images.shape})")
                else:
                    denoised_images = model_output

                # Save sample images for visual inspection
                if self.config.save_training_images:
                    self._save_image_samples(epoch_val, noisy_images, clean_images, denoised_images)

                # Compute and log denoising metrics
                mse_loss = tf.reduce_mean(tf.square(denoised_images - clean_images))
                psnr = tf.reduce_mean(tf.image.psnr(denoised_images, clean_images, max_val=1.0))

                logger.info(f"Epoch {epoch_val} - Validation MSE: {mse_loss.numpy():.6f}, "
                           f"PSNR: {psnr.numpy():.2f} dB")

                # === Image Synthesis Using Implicit Prior ===
                if self.config.enable_synthesis:
                    logger.info("Generating synthetic images using implicit prior...")
                    try:
                        # Configure synthesis parameters
                        synthesis_shape = (self.config.patch_size, self.config.patch_size, self.config.channels)

                        # Generate images through iterative sampling
                        generated_samples, intermediate_steps = unconditional_sampling(
                            denoiser=self.model,
                            num_samples=self.config.synthesis_samples,
                            image_shape=synthesis_shape,
                            num_steps=self.config.synthesis_steps,
                            initial_step_size=self.config.synthesis_initial_step_size,
                            final_step_size=self.config.synthesis_final_step_size,
                            initial_noise_level=self.config.synthesis_initial_noise,
                            final_noise_level=self.config.synthesis_final_noise,
                            seed=epoch_val,  # Reproducible generation
                            save_intermediate=True
                        )

                        # Create synthesis visualization
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

                # Save denoising evaluation metrics
                metrics = {
                    'epoch': epoch_val,
                    'val_mse': float(mse_loss),
                    'val_psnr': float(psnr),
                    'timestamp': datetime.now().isoformat()
                }
                metrics_file = self.results_dir / f"epoch_{epoch_val:03d}_metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)

                # Memory cleanup
                del noisy_images, clean_images, denoised_images
                gc.collect()

            except Exception as e:
                tf.print(f"Error during monitoring at epoch {epoch_val}: {e}")

            return tf.constant(0, dtype=tf.int32)

        # Execute monitoring within TensorFlow graph context
        tf.py_function(func=_monitor_and_save, inp=[epoch + 1], Tout=[tf.int32])

    def _save_image_samples(
        self,
        epoch: int,
        noisy: tf.Tensor,
        clean: tf.Tensor,
        denoised: tf.Tensor
    ) -> None:
        """
        Save visual comparison of noisy, clean, and denoised image samples.

        Args:
            epoch: Current epoch number for file naming
            noisy: Noisy input images
            clean: Ground truth clean images
            denoised: Model-denoised images
        """
        try:
            num_samples = min(10, noisy.shape[0])

            # Create 3-row comparison grid
            fig, axes = plt.subplots(3, 10, figsize=(25, 7.5))
            fig.suptitle(f'Denoising Results - Epoch {epoch}', fontsize=20, y=0.98)

            for i in range(10):
                if i < num_samples:
                    # Extract and clip images to valid range
                    clean_img = np.clip(clean[i].numpy(), 0.0, 1.0)
                    noisy_img = np.clip(noisy[i].numpy(), 0.0, 1.0)
                    denoised_img = np.clip(denoised[i].numpy(), 0.0, 1.0)

                    # Handle grayscale vs RGB format
                    if clean_img.shape[-1] == 1:
                        clean_img = clean_img.squeeze(-1)
                        noisy_img = noisy_img.squeeze(-1)
                        denoised_img = denoised_img.squeeze(-1)
                        cmap = 'gray'
                    else:
                        cmap = None

                    # Display images in three rows
                    axes[0, i].imshow(clean_img, cmap=cmap, vmin=0, vmax=1)
                    if i == 0:
                        axes[0, i].set_ylabel('Clean', fontsize=14, rotation=0, ha='right', va='center')
                    axes[0, i].set_title(f'Sample {i + 1}', fontsize=10)
                    axes[0, i].axis('off')

                    axes[1, i].imshow(noisy_img, cmap=cmap, vmin=0, vmax=1)
                    if i == 0:
                        axes[1, i].set_ylabel('Noisy', fontsize=14, rotation=0, ha='right', va='center')
                    axes[1, i].axis('off')

                    axes[2, i].imshow(denoised_img, cmap=cmap, vmin=0, vmax=1)
                    if i == 0:
                        axes[2, i].set_ylabel('Denoised', fontsize=14, rotation=0, ha='right', va='center')
                    axes[2, i].axis('off')
                else:
                    # Hide unused subplot slots
                    for row in range(3):
                        axes[row, i].axis('off')

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
    Dynamic weight scheduler for deep supervision training.

    This callback automatically updates the loss weights for multi-output models
    during training according to a configurable scheduling strategy. This allows
    the model to focus on different output scales at different training phases.

    Common strategies include:
    - Linear progression from coarse to fine outputs
    - Curriculum learning with progressive output activation
    - Custom sigmoid transitions for smooth weight changes
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
        self.scheduler = deep_supervision_schedule_builder(ds_config, self.num_outputs, invert_order=False)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Update model loss weights at the beginning of each epoch.

        Args:
            epoch: Current epoch number (0-indexed)
            logs: Training logs dictionary (unused)

        Note:
            The training progress is computed as epoch / (total_epochs - 1) to
            ensure the final epoch corresponds to progress = 1.0.
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


def create_callbacks(
    config: TrainingConfig,
    val_directories: List[str],
    num_outputs: int
) -> List[keras.callbacks.Callback]:
    """
    Create comprehensive training callbacks for monitoring and control.

    Args:
        config: Training configuration
        val_directories: Validation directories for monitoring
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

    # Comprehensive result monitoring with synthesis
    callbacks.append(StreamingResultMonitor(config, val_directories))

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
# MODEL CREATION UTILITIES
# =============================================================================

def create_model_instance(config: TrainingConfig, input_shape: Tuple[int, int, int]) -> keras.Model:
    """
    Create a model instance based on configuration specifications.

    Args:
        config: Training configuration specifying model architecture
        input_shape: Input tensor shape (height, width, channels)

    Returns:
        Configured Keras model instance

    Raises:
        ValueError: If model_type is not recognized
    """
    if config.model_type in BFUNET_CONFIGS:
        # Use predefined variant
        return create_bfunet_variant(
            variant=config.model_type,
            input_shape=input_shape,
            enable_deep_supervision=config.enable_deep_supervision
        )
    elif config.model_type == 'custom':
        # Create custom architecture
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


# =============================================================================
# MAIN TRAINING ORCHESTRATION
# =============================================================================

def train_bfunet_denoiser(config: TrainingConfig) -> keras.Model:
    """
    Orchestrate the complete training pipeline for bias-free U-Net denoisers.

    This function manages the entire training process including:
    1. Model creation and architecture inspection
    2. Multi-scale dataset preparation for deep supervision
    3. Optimizer and loss function configuration
    4. Training execution with comprehensive monitoring
    5. Final model saving for inference

    The key innovation is the automatic handling of multi-output models where
    the dataset is dynamically adapted to provide ground truth labels at
    multiple scales that match the model's output structure exactly.

    Args:
        config: Comprehensive training configuration

    Returns:
        Trained Keras model with best weights loaded

    Raises:
        ValueError: If configuration is invalid or no training data is found
        RuntimeError: If model creation or training fails

    Note:
        The function handles the complex case of deep supervision where the
        model outputs multiple tensors at different resolutions. The dataset
        pipeline automatically creates matching ground truth tensors through
        bilinear interpolation to ensure perfect structural alignment.
    """
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Deep Supervision: {'ENABLED' if config.enable_deep_supervision else 'DISABLED'}")
    if config.enable_deep_supervision:
        logger.info(f"  - Schedule: {config.deep_supervision_schedule_type}")

    # === 1. Setup and Configuration Validation ===
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration for reproducibility
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    logger.info(f"Results will be saved to: {output_dir}")

    # === 2. Data Validation and Counting ===
    logger.info("Validating training and validation data...")

    # Check directory existence
    for directory in config.train_image_dirs:
        if not Path(directory).exists():
            logger.warning(f"Training directory does not exist: {directory}")
    for directory in config.val_image_dirs:
        if not Path(directory).exists():
            logger.warning(f"Validation directory does not exist: {directory}")

    logger.info(f"Training directories: {config.train_image_dirs}")
    logger.info(f"Validation directories: {config.val_image_dirs}")

    # Count available files for dataset size estimation
    try:
        train_file_count = count_available_files(
            config.train_image_dirs, config.image_extensions, config.max_train_files
        )
        val_file_count = count_available_files(
            config.val_image_dirs, config.image_extensions, config.max_val_files
        )
    except Exception as e:
        logger.warning(f"Error counting files: {e}")
        train_file_count, val_file_count = 1000, 100  # Fallback estimates

    if train_file_count == 0:
        raise ValueError("No training files found!")
    if val_file_count == 0:
        raise ValueError("No validation files found!")

    logger.info(f"Found approximately {train_file_count} training and {val_file_count} validation files")

    # === 3. Model Creation and Architecture Analysis ===
    logger.info(f"Creating BFU-Net model variant: '{config.model_type}'...")
    input_shape = (config.patch_size, config.patch_size, config.channels)
    model = create_model_instance(config, input_shape)
    model.summary()

    # Analyze model output structure for deep supervision handling
    has_multiple_outputs = isinstance(model.output, list)
    num_outputs = len(model.output) if has_multiple_outputs else 1
    logger.info(f"Model created with {num_outputs} output(s)")

    # === 4. Dataset Creation and Multi-Scale Adaptation ===
    logger.info("Creating and configuring datasets...")
    train_dataset = create_dataset(config.train_image_dirs, config, is_training=True)
    val_dataset = create_dataset(config.val_image_dirs, config, is_training=False)

    # Critical adaptation for multi-output models: create matching ground truth structure
    if has_multiple_outputs:
        # Extract concrete output dimensions from model architecture
        concrete_output_dims = [(out.shape[1], out.shape[2]) for out in model.output]
        logger.info(f"Adapting dataset for multi-output model with dimensions: {concrete_output_dims}")

        def create_multiscale_labels(noisy_patch: tf.Tensor, clean_patch: tf.Tensor) -> Tuple[tf.Tensor, Tuple[tf.Tensor, ...]]:
            """
            Create multi-scale ground truth labels matching model output structure.

            This function is crucial for deep supervision training. It takes the
            full-resolution ground truth and creates multiple downsampled versions
            that exactly match each model output's spatial dimensions.

            Args:
                noisy_patch: Noisy input patch
                clean_patch: Full-resolution clean ground truth

            Returns:
                Tuple of (noisy_patch, tuple_of_multiscale_labels)

            Note:
                Returns a tuple instead of list to prevent tf.data from stacking
                the labels into a single tensor, maintaining the multi-output structure.
            """
            labels = [tf.image.resize(clean_patch, dim) for dim in concrete_output_dims]
            return noisy_patch, tuple(labels)

        # Apply multi-scale label generation to both datasets
        train_dataset = train_dataset.map(create_multiscale_labels, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(create_multiscale_labels, num_parallel_calls=tf.data.AUTOTUNE)

    # === 5. Optimization Configuration ===
    # Calculate training steps for learning rate scheduling
    if config.steps_per_epoch is not None:
        steps_per_epoch = config.steps_per_epoch
    else:
        total_patches = train_file_count * config.patches_per_image
        steps_per_epoch = max(100, total_patches // config.batch_size)
    logger.info(f"Calculated {steps_per_epoch} steps per epoch")

    # Configure learning rate schedule
    lr_config = {
        'type': config.lr_schedule_type,
        'learning_rate': config.learning_rate,
        'decay_steps': steps_per_epoch * config.epochs,
        'warmup_steps': steps_per_epoch * config.warmup_epochs,
        'alpha': 0.01  # Final learning rate as fraction of initial
    }

    # Configure optimizer with gradient clipping
    opt_config = {
        'type': config.optimizer_type,
        'gradient_clipping_by_norm': config.gradient_clipping
    }

    # Build optimizer with learning rate schedule
    lr_schedule = learning_rate_schedule_builder(lr_config)
    optimizer = optimizer_builder(opt_config, lr_schedule)

    # === 6. Loss Function and Metrics Configuration ===
    if has_multiple_outputs:
        # Configure for multi-output model with deep supervision
        loss_fns = ['mse'] * num_outputs
        initial_weights = [1.0 / num_outputs] * num_outputs  # Equal initial weights

        # Critical fix for metrics: target only the primary output by layer name
        # This prevents the "list length mismatch" error in Keras compilation
        metrics_for_primary_output = [
            'mae',
            keras.metrics.RootMeanSquaredError(name='rmse'),
            PrimaryOutputPSNR()
        ]

        # Assign metrics only to the 'final_output' layer (primary output)
        metrics = {'final_output': metrics_for_primary_output}
        logger.info("Metrics configured for 'final_output' layer only")

    else:
        # Single-output model configuration
        loss_fns = 'mse'
        initial_weights = None
        metrics = [
            'mae',
            keras.metrics.RootMeanSquaredError(name='rmse'),
            PrimaryOutputPSNR()
        ]

    # === 7. Model Compilation ===
    logger.info("Compiling model...")
    model.compile(
        optimizer=optimizer,
        loss=loss_fns,
        loss_weights=initial_weights,
        metrics=metrics
    )
    logger.info("Model compiled successfully")

    # === 8. Callback Setup and Training Execution ===
    callbacks = create_callbacks(config, config.val_image_dirs, num_outputs)

    start_time = time.time()
    validation_steps = config.validation_steps or max(50, steps_per_epoch // 20)

    # Execute training with comprehensive monitoring
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
    logger.info(f"Training completed in {training_time / 3600:.2f} hours")

    # === 9. Clean Inference Model Creation ===
    logger.info("Creating clean inference model...")
    try:
        # Create model with SAME architecture as training model (including deep supervision)
        inference_input_shape = (None, None, config.channels)  # Variable size for inference

        if config.model_type in BFUNET_CONFIGS:
            inference_model_full = create_bfunet_variant(
                variant=config.model_type,
                input_shape=inference_input_shape,
                enable_deep_supervision=config.enable_deep_supervision  # Match training configuration
            )
        else:
            inference_model_full = create_bfunet_denoiser(
                input_shape=inference_input_shape,
                depth=config.depth,
                initial_filters=config.filters,
                blocks_per_level=config.blocks_per_level,
                kernel_size=config.kernel_size,
                activation=config.activation,
                enable_deep_supervision=config.enable_deep_supervision,  # Match training configuration
                model_name=f'bfunet_custom_{config.experiment_name}_inference'
            )

        # Transfer trained weights to inference model (architectures now match)
        inference_model_full.set_weights(model.get_weights())
        logger.info("Successfully transferred weights to inference model")

        # If deep supervision was used, create a clean single-output model
        if config.enable_deep_supervision and isinstance(inference_model_full.output, list):
            logger.info("Creating single-output inference model from multi-output trained model...")

            # Create new functional model that only outputs the primary prediction
            inference_model = keras.Model(
                inputs=inference_model_full.input,
                outputs=inference_model_full.output[0],  # Only the primary/final output
                name=f"{inference_model_full.name}_single_output"
            )
            logger.info(f"Single-output inference model created with output shape: {inference_model.output.shape}")

            del inference_model_full  # Free memory
        else:
            # Single-output model, use as-is
            inference_model = inference_model_full
            logger.info("Single-output model, no modification needed")

        # Save clean inference model
        inference_model_path = output_dir / "inference_model.keras"
        inference_model.save(inference_model_path)
        logger.info(f"Clean inference model saved to: {inference_model_path}")

        del inference_model  # Free memory

    except Exception as e:
        logger.error(f"Failed to save clean inference model: {e}", exc_info=True)
        raise

    # === 10. Training History Preservation ===
    try:
        history_path = output_dir / "training_history.json"
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        logger.info(f"Training history saved to: {history_path}")
    except Exception as e:
        logger.warning(f"Failed to save training history: {e}")

    # Final cleanup
    gc.collect()

    return model


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for training configuration.

    Returns:
        Parsed arguments namespace with all training parameters
    """
    parser = argparse.ArgumentParser(
        description='Train Bias-Free U-Net Denoiser with Deep Supervision',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # === Core Training Parameters ===
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--model-type',
        choices=['tiny', 'small', 'base', 'large', 'xlarge', 'custom'],
        default='tiny',
        help='Model architecture variant'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--patch-size', type=int, default=128,
        help='Size of square training patches'
    )
    parser.add_argument(
        '--channels', type=int, choices=[1, 3], default=1,
        help='Number of input channels (1=grayscale, 3=RGB)'
    )

    # === Deep Supervision Configuration ===
    parser.add_argument(
        '--enable-deep-supervision', action='store_true', default=True,
        help='Enable deep supervision training'
    )
    parser.add_argument(
        '--no-deep-supervision', dest='enable_deep_supervision', action='store_false',
        help='Disable deep supervision training'
    )
    parser.add_argument(
        '--deep-supervision-schedule',
        choices=[
            'constant_equal', 'constant_low_to_high', 'constant_high_to_low',
            'linear_low_to_high', 'non_linear_low_to_high', 'custom_sigmoid_low_to_high',
            'scale_by_scale_low_to_high', 'cosine_annealing', 'curriculum', 'step_wise'
        ],
        default='step_wise',
        help='Deep supervision weight scheduling strategy'
    )

    # === Output and Monitoring ===
    parser.add_argument(
        '--output-dir', type=str, default='results',
        help='Base output directory for results'
    )
    parser.add_argument(
        '--experiment-name', type=str, default=None,
        help='Experiment identifier (auto-generated if not provided)'
    )
    parser.add_argument(
        '--patches-per-image', type=int, default=4,
        help='Number of patches to extract per image'
    )
    parser.add_argument(
        '--monitor-every', type=int, default=1,
        help='Save intermediate results every N epochs'
    )
    parser.add_argument(
        '--early-stopping-patience', type=int, default=15,
        help='Early stopping patience in epochs'
    )
    parser.add_argument(
        '--max-train-files', type=int, default=None,
        help='Maximum number of training files to use'
    )

    # === Image Synthesis Parameters ===
    parser.add_argument(
        '--enable-synthesis', action='store_true', default=True,
        help='Enable image synthesis during monitoring'
    )
    parser.add_argument(
        '--no-synthesis', dest='enable_synthesis', action='store_false',
        help='Disable image synthesis during monitoring'
    )
    parser.add_argument(
        '--synthesis-samples', type=int, default=4,
        help='Number of images to synthesize'
    )
    parser.add_argument(
        '--synthesis-steps', type=int, default=200,
        help='Number of synthesis iteration steps'
    )

    return parser.parse_args()


# =============================================================================
# MAIN EXECUTION ENTRY POINT
# =============================================================================

def main() -> None:
    """
    Main training function with comprehensive configuration and execution.

    This function demonstrates how to configure and execute the complete
    training pipeline for bias-free U-Net denoisers with deep supervision.

    Note:
        The data paths in this example should be modified to match your
        specific dataset locations and organizational structure.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Create comprehensive training configuration
    config = TrainingConfig(
        # === Data Paths (Modify these for your setup) ===
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

        # === Core Training Configuration ===
        patch_size=args.patch_size,
        channels=args.channels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patches_per_image=args.patches_per_image,

        # === Data Management ===
        max_train_files=args.max_train_files,
        max_val_files=10000,  # Reasonable validation set size
        parallel_reads=8,  # Optimize for system capabilities
        dataset_shuffle_buffer=1013,

        # === Model Architecture ===
        model_type=args.model_type,

        # === Deep Supervision Configuration ===
        enable_deep_supervision=args.enable_deep_supervision,
        deep_supervision_schedule_type=args.deep_supervision_schedule,
        deep_supervision_schedule_config={},  # Use default parameters

        # === Noise Configuration (Universal Range) ===
        noise_sigma_min=0.0,
        noise_sigma_max=0.5,  # Wide range for robust training
        noise_distribution='uniform',

        # === Optimization Strategy ===
        learning_rate=1e-3,
        optimizer_type='adamw',  # Often better than Adam for denoising
        lr_schedule_type='cosine_decay',
        warmup_epochs=5,

        # === Monitoring and Evaluation ===
        monitor_every_n_epochs=args.monitor_every,
        save_training_images=True,
        validation_steps=500,

        # === Image Synthesis Configuration ===
        enable_synthesis=args.enable_synthesis,
        synthesis_samples=args.synthesis_samples,
        synthesis_steps=args.synthesis_steps,

        # === Output Configuration ===
        output_dir=args.output_dir,
        experiment_name=args.experiment_name
    )

    # Log comprehensive configuration summary
    logger.info("=== TRAINING CONFIGURATION SUMMARY ===")
    logger.info(f"Model Type: {config.model_type}")
    logger.info(f"Deep Supervision: {'Enabled' if config.enable_deep_supervision else 'Disabled'}")
    if config.enable_deep_supervision:
        logger.info(f"  - Schedule: {config.deep_supervision_schedule_type}")
        logger.info(f"  - Config: {config.deep_supervision_schedule_config}")
    logger.info(f"Training: {config.epochs} epochs, batch size {config.batch_size}")
    logger.info(f"Patches: {config.patch_size}x{config.patch_size}x{config.channels}")
    logger.info(f"Learning Rate: {config.learning_rate} with {config.lr_schedule_type} schedule")
    logger.info(f"Noise Range: [{config.noise_sigma_min}, {config.noise_sigma_max}] ({config.noise_distribution})")
    logger.info(f"Monitoring: Every {config.monitor_every_n_epochs} epochs")
    logger.info(f"Synthesis: {'Enabled' if config.enable_synthesis else 'Disabled'}")
    if config.enable_synthesis:
        logger.info(f"  - Samples: {config.synthesis_samples}, Steps: {config.synthesis_steps}")
        logger.info(f"  - Step Size: [{config.synthesis_initial_step_size}, {config.synthesis_final_step_size}]")
        logger.info(f"  - Noise Level: [{config.synthesis_final_noise}, {config.synthesis_initial_noise}]")
    logger.info(f"Output: {config.output_dir}/{config.experiment_name}")

    # Execute training pipeline
    try:
        model = train_bfunet_denoiser(config)
        logger.info("=== TRAINING COMPLETED SUCCESSFULLY ===")
        model.summary()

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()