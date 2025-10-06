"""
Training Script for Conditional Bias-Free U-Net

Implements class-conditional denoising using Miyasawa's theorem for conditionals.
Supports Classifier-Free Guidance (CFG) training and sampling.

Key Features:
- Class-conditional denoising with embedding-based conditioning
- CFG training with unconditional token dropout
- CFG-guided sampling for enhanced generation
- Deep supervision support
- Conditional image synthesis visualization
"""

import gc
import keras
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any, Union

from dl_techniques.utils.train_vision.framework import (
    TrainingConfig,
    DatasetBuilder,
    TrainingPipeline,
    ModelBuilder,
)
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import deep_supervision_schedule_builder
from dl_techniques.models.bias_free_denoisers.bfunet_conditional import (
    create_conditional_bfunet_denoiser,
    create_conditional_bfunet_variant,
)
from dl_techniques.analyzer import DataInput


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ConditionalBFUNetConfig(TrainingConfig):
    """
    Configuration for conditional BFU-Net training.

    Attributes:
        train_image_dirs: Directories containing training images.
        val_image_dirs: Directories containing validation images.
        patch_size: Size of training patches.
        channels: Number of input channels.
        num_classes: Number of class labels (auto-detected if not specified).
        class_embedding_dim: Dimension of class embedding vectors.
        class_injection_method: Method for injecting class information.
        enable_cfg_training: Enable classifier-free guidance training.
        cfg_dropout_prob: Probability of dropping class labels for CFG.
        cfg_guidance_scale: Default guidance scale for CFG sampling.
        image_extensions: Valid image file extensions.
        max_train_files: Limit on training files.
        max_val_files: Limit on validation files.
        parallel_reads: Parallel file reading threads.
        dataset_shuffle_buffer: Shuffle buffer size.
        noise_sigma_min: Minimum noise standard deviation.
        noise_sigma_max: Maximum noise standard deviation.
        noise_distribution: Noise sampling distribution.
        patches_per_image: Patches to extract per image.
        augment_data: Apply data augmentation.
        normalize_input: Normalize input to [0, 1].
        enable_deep_supervision: Use multi-scale supervision.
        deep_supervision_schedule_type: Weight scheduling strategy.
        deep_supervision_schedule_config: Schedule parameters.
        depth: Network depth for custom models.
        blocks_per_level: Residual blocks per level.
        filters: Base number of filters.
        kernel_size: Convolutional kernel size.
        activation: Activation function.
        enable_synthesis: Enable conditional image synthesis monitoring.
        synthesis_samples_per_class: Number of samples per class.
        synthesis_steps: Synthesis iteration steps.
        synthesis_initial_step_size: Initial gradient step size.
        synthesis_final_step_size: Final gradient step size.
        synthesis_initial_noise: Initial noise injection.
        synthesis_final_noise: Final noise injection.
        synthesis_cfg_scales: CFG scales to test during synthesis.
        monitor_every_n_epochs: Visualization frequency.
        save_training_images: Save denoising samples.
        organize_by_subfolders: Whether images are organized in class subfolders.
    """
    # Data paths
    train_image_dirs: List[str] = field(default_factory=list)
    val_image_dirs: List[str] = field(default_factory=list)

    # Input shape
    patch_size: int = 128
    channels: int = 1

    # Conditional model specific
    num_classes: Optional[int] = None  # Auto-detected from data
    class_embedding_dim: int = 128
    class_injection_method: str = 'spatial_broadcast'

    # CFG training
    enable_cfg_training: bool = True
    cfg_dropout_prob: float = 0.1
    cfg_guidance_scale: float = 7.5

    # File handling
    image_extensions: List[str] = field(
        default_factory=lambda: ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp']
    )
    max_train_files: Optional[int] = None
    max_val_files: Optional[int] = None
    parallel_reads: int = 8
    dataset_shuffle_buffer: int = 1000

    # Noise configuration
    noise_sigma_min: float = 0.0
    noise_sigma_max: float = 0.5
    noise_distribution: str = 'uniform'

    # Training specifics
    patches_per_image: int = 16
    augment_data: bool = True
    normalize_input: bool = True

    # Deep supervision
    enable_deep_supervision: bool = True
    deep_supervision_schedule_type: str = 'step_wise'
    deep_supervision_schedule_config: Dict[str, Any] = field(default_factory=dict)

    # Custom model parameters
    depth: int = 3
    blocks_per_level: int = 2
    filters: int = 64
    kernel_size: int = 3
    activation: str = 'relu'

    # Conditional synthesis
    enable_synthesis: bool = True
    synthesis_samples_per_class: int = 2
    synthesis_steps: int = 200
    synthesis_initial_step_size: float = 0.05
    synthesis_final_step_size: float = 0.8
    synthesis_initial_noise: float = 0.4
    synthesis_final_noise: float = 0.005
    synthesis_cfg_scales: List[float] = field(default_factory=lambda: [0.0, 1.0, 3.0, 7.5])

    # Monitoring
    monitor_every_n_epochs: int = 2
    save_training_images: bool = True

    # Data organization
    organize_by_subfolders: bool = True

    def __post_init__(self):
        """Initialize and validate configuration."""
        self.input_shape = (self.patch_size, self.patch_size, self.channels)

        # Call parent
        super().__post_init__()

        # Additional validation
        if self.noise_sigma_min < 0 or self.noise_sigma_max <= self.noise_sigma_min:
            raise ValueError("Invalid noise sigma range")

        if not self.train_image_dirs:
            raise ValueError("No training directories specified")

        if not self.val_image_dirs:
            raise ValueError("No validation directories specified")

        if self.enable_cfg_training and self.cfg_dropout_prob <= 0:
            raise ValueError("CFG dropout probability must be positive when CFG training enabled")


# =============================================================================
# CONDITIONAL DATASET BUILDER
# =============================================================================

class ConditionalBFUNetDatasetBuilder(DatasetBuilder):
    """
    Dataset builder for conditional denoising with class labels.

    Supports two modes:
    1. Subfolder organization: Each class in its own subfolder
    2. Manual class mapping: Explicit class assignments
    """

    def __init__(
            self,
            config: ConditionalBFUNetConfig,
            model_output_dims: Optional[List[Tuple[int, int]]] = None,
            class_to_idx: Optional[Dict[str, int]] = None
    ):
        """
        Initialize conditional dataset builder.

        Args:
            config: ConditionalBFUNetConfig instance.
            model_output_dims: List of (height, width) tuples for each output scale.
            class_to_idx: Optional mapping from class names to indices.
        """
        super().__init__(config)
        self.config: ConditionalBFUNetConfig = config
        self.model_output_dims = model_output_dims
        self.class_to_idx = class_to_idx
        self.unconditional_token = None

    def _discover_classes(self, directories: List[str]) -> Dict[str, int]:
        """
        Discover class structure from directory organization.

        Returns:
            Dictionary mapping class names to indices.
        """
        if not self.config.organize_by_subfolders:
            raise ValueError("Class discovery requires subfolder organization")

        class_names = set()

        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.is_dir():
                continue

            # Get immediate subdirectories as class names
            subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
            class_names.update([d.name for d in subdirs])

        if not class_names:
            raise ValueError(f"No class subdirectories found in {directories}")

        # Sort for consistent ordering
        sorted_classes = sorted(class_names)
        class_to_idx = {name: idx for idx, name in enumerate(sorted_classes)}

        # Reserve last index for unconditional token if CFG enabled
        if self.config.enable_cfg_training:
            self.unconditional_token = len(class_to_idx)
            class_to_idx['__unconditional__'] = self.unconditional_token
            logger.info(f"Reserved class {self.unconditional_token} as unconditional token for CFG")

        logger.info(f"Discovered {len(sorted_classes)} classes: {sorted_classes}")

        return class_to_idx

    def _create_file_list_with_labels(
            self,
            directories: List[str],
            limit: Optional[int] = None
    ) -> Tuple[List[str], List[int]]:
        """
        Create file list with corresponding class labels.

        Returns:
            Tuple of (file_paths, class_labels)
        """
        all_files = []
        all_labels = []
        extensions_set = {ext.lower() for ext in self.config.image_extensions}
        extensions_set.update({ext.upper() for ext in self.config.image_extensions})

        if self.class_to_idx is None:
            self.class_to_idx = self._discover_classes(directories)

        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.is_dir():
                logger.warning(f"Directory not found: {directory}")
                continue

            if self.config.organize_by_subfolders:
                # Iterate through class subdirectories
                for class_name, class_idx in self.class_to_idx.items():
                    if class_name == '__unconditional__':
                        continue

                    class_dir = dir_path / class_name
                    if not class_dir.is_dir():
                        continue

                    try:
                        for file_path in class_dir.rglob("*"):
                            if file_path.is_file() and file_path.suffix in extensions_set:
                                all_files.append(str(file_path))
                                all_labels.append(class_idx)
                    except Exception as e:
                        logger.warning(f"Error scanning {class_dir}: {e}")
                        continue
            else:
                # Flat organization - assign class based on some criterion
                # (This would need custom logic based on your use case)
                raise NotImplementedError("Flat organization requires custom class assignment")

        if not all_files:
            raise ValueError(f"No files found in directories: {directories}")

        logger.info(f"Found {len(all_files)} files across {len(set(all_labels))} classes")

        # Log class distribution
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            class_name = [k for k, v in self.class_to_idx.items() if v == label][0]
            logger.info(f"  Class '{class_name}' ({label}): {count} files")

        if limit and limit < len(all_files):
            logger.info(f"Limiting to {limit} files")
            indices = np.random.choice(len(all_files), limit, replace=False)
            all_files = [all_files[i] for i in indices]
            all_labels = [all_labels[i] for i in indices]

        return all_files, all_labels

    def _load_and_preprocess_image(
            self,
            image_path: tf.Tensor,
            class_label: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Load and preprocess image with class label.

        Args:
            image_path: Path to image file.
            class_label: Integer class label.

        Returns:
            Tuple of (image_patch, class_label)
        """
        try:
            image_string = tf.io.read_file(image_path)
            image = tf.image.decode_image(
                image_string,
                channels=self.config.channels,
                expand_animations=False
            )
            image.set_shape([None, None, self.config.channels])

            image = tf.cast(image, tf.float32)
            if self.config.normalize_input:
                image = image / 255.0

            # Handle small images
            shape = tf.shape(image)
            height, width = shape[0], shape[1]
            min_dim = tf.minimum(height, width)
            min_size = self.config.patch_size

            def resize_if_small():
                scale = tf.cast(min_size, tf.float32) / tf.cast(min_dim, tf.float32)
                new_h = tf.cast(tf.math.ceil(tf.cast(height, tf.float32) * scale), tf.int32)
                new_w = tf.cast(tf.math.ceil(tf.cast(width, tf.float32) * scale), tf.int32)
                return tf.image.resize(image, [new_h, new_w])

            image = tf.cond(
                tf.logical_or(height < min_size, width < min_size),
                true_fn=resize_if_small,
                false_fn=lambda: image
            )

            patch = tf.image.random_crop(
                image,
                [self.config.patch_size, self.config.patch_size, self.config.channels]
            )

            return patch, class_label

        except tf.errors.InvalidArgumentError:
            logger.warning(f"Failed to load image: {image_path}")
            return (
                tf.zeros(
                    [self.config.patch_size, self.config.patch_size, self.config.channels],
                    dtype=tf.float32
                ),
                class_label
            )

    def _augment_patch(
            self,
            patch: tf.Tensor,
            class_label: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply data augmentation."""
        patch = tf.image.random_flip_left_right(patch)
        patch = tf.image.random_flip_up_down(patch)
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        patch = tf.image.rot90(patch, k)
        return patch, class_label

    def _add_noise(
            self,
            patch: tf.Tensor,
            class_label: tf.Tensor
    ) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        """
        Add Gaussian noise and prepare data for conditional model.

        Returns:
            ((noisy_patch, class_label), clean_patch)
        """
        if self.config.noise_distribution == 'uniform':
            noise_level = tf.random.uniform(
                [],
                self.config.noise_sigma_min,
                self.config.noise_sigma_max
            )
        else:  # log_uniform
            log_min = tf.math.log(tf.maximum(self.config.noise_sigma_min, 1e-6))
            log_max = tf.math.log(self.config.noise_sigma_max)
            log_noise = tf.random.uniform([], log_min, log_max)
            noise_level = tf.exp(log_noise)

        noise = tf.random.normal(tf.shape(patch)) * noise_level
        noisy_patch = patch + noise
        noisy_patch = tf.clip_by_value(noisy_patch, 0.0, 1.0)

        return (noisy_patch, class_label), patch

    def _apply_cfg_dropout(
            self,
            noisy_patch: tf.Tensor,
            class_label: tf.Tensor,
            clean_patch: tf.Tensor
    ) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        """
        Apply classifier-free guidance dropout during training.

        Randomly replaces class labels with unconditional token.
        """
        if not self.config.enable_cfg_training or self.unconditional_token is None:
            return (noisy_patch, class_label), clean_patch

        # Random dropout
        should_drop = tf.random.uniform([]) < self.config.cfg_dropout_prob

        # Replace with unconditional token if dropout
        class_label = tf.cond(
            should_drop,
            lambda: tf.constant(self.unconditional_token, dtype=class_label.dtype),
            lambda: class_label
        )

        return (noisy_patch, class_label), clean_patch

    def _create_multiscale_labels(
            self,
            inputs: Tuple[tf.Tensor, tf.Tensor],
            clean_patch: tf.Tensor
    ) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, ...]]:
        """Create multi-scale labels for deep supervision."""
        noisy_patch, class_label = inputs

        if self.model_output_dims is None:
            return (noisy_patch, class_label), clean_patch

        labels = [
            tf.image.resize(clean_patch, dim)
            for dim in self.model_output_dims
        ]
        return (noisy_patch, class_label), tuple(labels)

    def build(self) -> Tuple[
        tf.data.Dataset,
        tf.data.Dataset,
        Optional[int],
        Optional[int]
    ]:
        """Build training and validation datasets with class labels."""
        logger.info("Building conditional BFU-Net datasets")

        # Create file lists with labels
        train_files, train_labels = self._create_file_list_with_labels(
            self.config.train_image_dirs,
            self.config.max_train_files
        )
        val_files, val_labels = self._create_file_list_with_labels(
            self.config.val_image_dirs,
            self.config.max_val_files
        )

        # Store class info for later use
        if self.config.num_classes is None:
            self.config.num_classes = len(self.class_to_idx)
            logger.info(f"Auto-detected {self.config.num_classes} classes")

        # Create training dataset
        train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
        train_ds = train_ds.shuffle(
            buffer_size=min(self.config.dataset_shuffle_buffer, len(train_files)),
            reshuffle_each_iteration=True
        )
        train_ds = train_ds.repeat()

        if self.config.patches_per_image > 1:
            train_ds = train_ds.flat_map(
                lambda path, label: tf.data.Dataset.from_tensors((path, label)).repeat(
                    self.config.patches_per_image
                )
            )

        train_ds = train_ds.map(
            self._load_and_preprocess_image,
            num_parallel_calls=self.config.parallel_reads
        )
        train_ds = train_ds.filter(lambda x, y: tf.reduce_sum(tf.abs(x)) > 0)

        if self.config.augment_data:
            train_ds = train_ds.map(
                self._augment_patch,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        train_ds = train_ds.map(
            self._add_noise,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Apply CFG dropout
        if self.config.enable_cfg_training:
            train_ds = train_ds.map(
                lambda inputs, labels: self._apply_cfg_dropout(inputs[0], inputs[1], labels),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Add multi-scale labels if deep supervision enabled
        if self.config.enable_deep_supervision and self.model_output_dims:
            logger.info(f"Adding multi-scale labels for {len(self.model_output_dims)} outputs")
            train_ds = train_ds.map(
                self._create_multiscale_labels,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        train_ds = train_ds.batch(self.config.batch_size, drop_remainder=True)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        # Create validation dataset (similar to training but no CFG dropout)
        val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
        val_ds = val_ds.repeat()

        if self.config.patches_per_image > 1:
            val_ds = val_ds.flat_map(
                lambda path, label: tf.data.Dataset.from_tensors((path, label)).repeat(
                    self.config.patches_per_image
                )
            )

        val_ds = val_ds.map(
            self._load_and_preprocess_image,
            num_parallel_calls=self.config.parallel_reads
        )
        val_ds = val_ds.filter(lambda x, y: tf.reduce_sum(tf.abs(x)) > 0)
        val_ds = val_ds.map(self._add_noise, num_parallel_calls=tf.data.AUTOTUNE)

        if self.config.enable_deep_supervision and self.model_output_dims:
            val_ds = val_ds.map(
                self._create_multiscale_labels,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        val_ds = val_ds.batch(self.config.batch_size)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

        # Calculate steps
        total_train_patches = len(train_files) * self.config.patches_per_image
        steps_per_epoch = max(100, total_train_patches // self.config.batch_size)

        val_steps = self.config.validation_steps or max(
            50,
            (len(val_files) * self.config.patches_per_image) // self.config.batch_size
        )

        logger.info(f"Dataset ready: {steps_per_epoch} train steps, {val_steps} val steps")

        return train_ds, val_ds, steps_per_epoch, val_steps

    def get_test_data(self) -> Optional[DataInput]:
        """Get test data for analysis."""
        return None


# =============================================================================
# CLASSIFIER-FREE GUIDANCE SAMPLING
# =============================================================================

def conditional_sampling_with_cfg(
        denoiser: keras.Model,
        class_labels: np.ndarray,
        unconditional_token: int,
        num_samples: int = 4,
        image_shape: Tuple[int, int, int] = (64, 64, 1),
        num_steps: int = 200,
        initial_step_size: float = 0.1,
        final_step_size: float = 1.0,
        initial_noise_level: float = 0.5,
        final_noise_level: float = 0.01,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        save_intermediate: bool = True
) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    """
    Generate images using conditional denoiser with Classifier-Free Guidance.

    Implements the CFG formula:
        s_guided(y, c) = s(y) + w * (s(y, c) - s(y))

    Where s(y, c) is the conditional score and s(y) is the unconditional score.

    Args:
        denoiser: Conditional denoiser model.
        class_labels: Array of class indices (batch_size,).
        unconditional_token: Index of unconditional token.
        num_samples: Number of samples to generate.
        image_shape: Shape of generated images.
        num_steps: Number of sampling steps.
        initial_step_size: Initial step size.
        final_step_size: Final step size.
        initial_noise_level: Initial noise level.
        final_noise_level: Final noise level.
        guidance_scale: CFG guidance scale (w in formula).
        seed: Random seed.
        save_intermediate: Whether to save intermediate steps.

    Returns:
        Tuple of (final_samples, intermediate_steps)
    """
    if seed is not None:
        tf.random.set_seed(seed)

    logger.info(f"Starting conditional sampling with CFG (w={guidance_scale})")
    logger.info(f"Class labels: {class_labels}")

    # Initialize from random noise
    y = tf.random.normal([num_samples] + list(image_shape), mean=0.5, stddev=0.3)
    y = tf.clip_by_value(y, 0.0, 1.0)

    intermediate_steps = []
    step_sizes = tf.linspace(initial_step_size, final_step_size, num_steps)
    noise_levels = tf.linspace(initial_noise_level, final_noise_level, num_steps)

    # Prepare class labels for conditional and unconditional passes
    class_labels_tf = tf.constant(class_labels, dtype=tf.int32)
    class_labels_tf = tf.reshape(class_labels_tf, (-1, 1))

    unconditional_labels = tf.constant(
        [[unconditional_token]] * num_samples,
        dtype=tf.int32
    )

    for t in range(num_steps):
        h_t = step_sizes[t]
        gamma_t = noise_levels[t]

        if guidance_scale == 0.0:
            # Pure unconditional generation
            model_output = denoiser([y, unconditional_labels], training=False)
        elif guidance_scale == 1.0:
            # Pure conditional generation (no guidance)
            model_output = denoiser([y, class_labels_tf], training=False)
        else:
            # CFG: Combine conditional and unconditional predictions
            # Batch conditional and unconditional together for efficiency
            y_double = tf.concat([y, y], axis=0)
            labels_double = tf.concat([class_labels_tf, unconditional_labels], axis=0)

            output_double = denoiser([y_double, labels_double], training=False)

            if isinstance(output_double, list):
                denoised_double = output_double[0]
            else:
                denoised_double = output_double

            # Split back into conditional and unconditional
            denoised_cond, denoised_uncond = tf.split(denoised_double, 2, axis=0)

            # Compute guided prediction using CFG formula
            # x_guided = x_uncond + w * (x_cond - x_uncond)
            model_output = denoised_uncond + guidance_scale * (denoised_cond - denoised_uncond)

        if isinstance(model_output, list):
            denoised = model_output[0]
        else:
            denoised = model_output

        # Gradient ascent step
        d_t = denoised - y
        z_t = tf.random.normal(tf.shape(y))
        y = y + h_t * d_t + gamma_t * z_t
        y = tf.clip_by_value(y, 0.0, 1.0)

        if save_intermediate and (t % (num_steps // 10) == 0 or t == num_steps - 1):
            intermediate_steps.append(y.numpy().copy())

        if t % (num_steps // 5) == 0:
            logger.info(f"Step {t}/{num_steps}: h_t={h_t:.4f}, γ_t={gamma_t:.4f}")

    logger.info("Conditional sampling completed")
    return y, intermediate_steps


# =============================================================================
# METRICS
# =============================================================================

@keras.saving.register_keras_serializable()
def psnr_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """PSNR metric for [0, 1] range."""
    return tf.reduce_mean(tf.image.psnr(y_pred, y_true, max_val=1.0))


class PrimaryOutputPSNR(keras.metrics.Metric):
    """PSNR metric for primary output in multi-output models."""

    def __init__(self, name: str = 'primary_psnr', **kwargs):
        super().__init__(name=name, **kwargs)
        self.psnr_sum = self.add_weight(name='psnr_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(
            self,
            y_true: Union[tf.Tensor, List[tf.Tensor]],
            y_pred: Union[tf.Tensor, List[tf.Tensor]],
            sample_weight: Optional[tf.Tensor] = None
    ):
        if isinstance(y_pred, list):
            primary_pred = y_pred[0]
            primary_true = y_true[0]
        else:
            primary_pred = y_pred
            primary_true = y_true

        psnr_batch = tf.image.psnr(primary_pred, primary_true, max_val=1.0)
        self.psnr_sum.assign_add(tf.reduce_sum(psnr_batch))
        self.count.assign_add(tf.cast(tf.size(psnr_batch), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.psnr_sum, self.count)

    def reset_state(self):
        self.psnr_sum.assign(0.0)
        self.count.assign(0.0)


# =============================================================================
# CALLBACKS
# =============================================================================

class DeepSupervisionWeightScheduler(keras.callbacks.Callback):
    """Dynamic weight scheduler for deep supervision."""

    def __init__(self, config: ConditionalBFUNetConfig, num_outputs: int):
        super().__init__()
        self.config = config
        self.num_outputs = num_outputs
        self.total_epochs = config.epochs

        ds_config = {
            'type': config.deep_supervision_schedule_type,
            'config': config.deep_supervision_schedule_config
        }
        self.scheduler = deep_supervision_schedule_builder(
            ds_config,
            self.num_outputs,
            invert_order=False
        )

        logger.info("Deep supervision scheduler initialized")

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        progress = min(1.0, epoch / max(1, self.total_epochs - 1))
        new_weights = self.scheduler(progress)
        self.model.loss_weights = new_weights

        weights_str = ", ".join([f"{w:.4f}" for w in new_weights])
        logger.info(f"Epoch {epoch + 1} - DS weights: [{weights_str}]")


class ConditionalSynthesisMonitorCallback(keras.callbacks.Callback):
    """Monitor conditional image synthesis with CFG."""

    def __init__(
            self,
            config: ConditionalBFUNetConfig,
            output_dir: Path,
            class_to_idx: Dict[str, int],
            unconditional_token: int
    ):
        super().__init__()
        self.config = config
        self.output_dir = output_dir / "conditional_synthesis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_to_idx = class_to_idx
        self.unconditional_token = unconditional_token
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if not self.config.enable_synthesis:
            return

        if (epoch + 1) % self.config.monitor_every_n_epochs != 0:
            return

        try:
            logger.info(f"Generating conditional samples at epoch {epoch + 1}")

            shape = (self.config.patch_size, self.config.patch_size, self.config.channels)

            # Select classes to visualize (skip unconditional token)
            valid_classes = [
                idx for idx in self.class_to_idx.values()
                if idx != self.unconditional_token
            ]
            num_classes_to_viz = min(5, len(valid_classes))
            selected_classes = np.random.choice(
                valid_classes,
                size=num_classes_to_viz,
                replace=False
            )

            # Generate samples for each CFG scale
            results = {}
            for cfg_scale in self.config.synthesis_cfg_scales:
                logger.info(f"  Generating with CFG scale {cfg_scale}")

                # Create class labels: samples_per_class for each selected class
                class_labels = np.repeat(
                    selected_classes,
                    self.config.synthesis_samples_per_class
                )

                generated, intermediates = conditional_sampling_with_cfg(
                    denoiser=self.model,
                    class_labels=class_labels,
                    unconditional_token=self.unconditional_token,
                    num_samples=len(class_labels),
                    image_shape=shape,
                    num_steps=self.config.synthesis_steps,
                    initial_step_size=self.config.synthesis_initial_step_size,
                    final_step_size=self.config.synthesis_final_step_size,
                    initial_noise_level=self.config.synthesis_initial_noise,
                    final_noise_level=self.config.synthesis_final_noise,
                    guidance_scale=cfg_scale,
                    seed=epoch + int(cfg_scale * 100),
                    save_intermediate=True
                )

                results[cfg_scale] = (generated, class_labels, intermediates)

            self._visualize_conditional_synthesis(epoch + 1, results, selected_classes)

        except Exception as e:
            logger.warning(f"Conditional synthesis failed: {e}")

    def _visualize_conditional_synthesis(
            self,
            epoch: int,
            results: Dict[float, Tuple[tf.Tensor, np.ndarray, List[tf.Tensor]]],
            selected_classes: np.ndarray
    ):
        """Visualize conditional synthesis across different CFG scales."""
        try:
            num_cfg_scales = len(results)
            num_classes = len(selected_classes)
            samples_per_class = self.config.synthesis_samples_per_class

            # Create comparison grid: rows = CFG scales, columns = classes
            fig, axes = plt.subplots(
                num_cfg_scales,
                num_classes,
                figsize=(3 * num_classes, 3 * num_cfg_scales)
            )

            if num_cfg_scales == 1:
                axes = axes.reshape(1, -1)
            if num_classes == 1:
                axes = axes.reshape(-1, 1)

            fig.suptitle(
                f'Conditional Synthesis with CFG - Epoch {epoch}',
                fontsize=16
            )

            for row_idx, (cfg_scale, (generated, class_labels, _)) in enumerate(results.items()):
                for col_idx, class_idx in enumerate(selected_classes):
                    # Get first sample for this class
                    mask = class_labels == class_idx
                    class_samples = generated[mask]

                    if len(class_samples) > 0:
                        img = class_samples[0].numpy()

                        if img.shape[-1] == 1:
                            img = img.squeeze(-1)
                            cmap = 'gray'
                        else:
                            cmap = None

                        img = np.clip(img, 0, 1)
                        axes[row_idx, col_idx].imshow(img, cmap=cmap, vmin=0, vmax=1)
                    else:
                        axes[row_idx, col_idx].imshow(
                            np.zeros((self.config.patch_size, self.config.patch_size)),
                            cmap='gray'
                        )

                    axes[row_idx, col_idx].axis('off')

                    # Add labels
                    if row_idx == 0:
                        class_name = self.idx_to_class.get(class_idx, f'Class {class_idx}')
                        axes[row_idx, col_idx].set_title(class_name, fontsize=10)

                    if col_idx == 0:
                        axes[row_idx, col_idx].set_ylabel(
                            f'CFG={cfg_scale:.1f}',
                            fontsize=10,
                            rotation=0,
                            ha='right',
                            va='center'
                        )

            plt.tight_layout()
            save_path = self.output_dir / f"epoch_{epoch:03d}_cfg_comparison.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            gc.collect()

        except Exception as e:
            logger.warning(f"Visualization failed: {e}")


class ConditionalDenoisingVisualizationCallback(keras.callbacks.Callback):
    """Visualize conditional denoising results."""

    def __init__(
            self,
            config: ConditionalBFUNetConfig,
            val_files: List[str],
            val_labels: List[int],
            output_dir: Path,
            class_to_idx: Dict[str, int]
    ):
        super().__init__()
        self.config = config
        self.output_dir = output_dir / "conditional_denoising"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self._create_test_batch(val_files, val_labels)

    def _create_test_batch(self, val_files: List[str], val_labels: List[int]):
        """Create fixed test batch with class labels."""
        try:
            builder = ConditionalBFUNetDatasetBuilder(self.config)
            builder.class_to_idx = self.class_to_idx

            clean_patches = []
            class_labels = []

            for file_path, label in zip(val_files[:10], val_labels[:10]):
                path_tensor = tf.constant(file_path)
                label_tensor = tf.constant(label, dtype=tf.int32)
                patch, label_out = builder._load_and_preprocess_image(path_tensor, label_tensor)
                clean_patches.append(patch)
                class_labels.append(label_out)

            self.test_batch = tf.stack(clean_patches)
            self.test_labels = tf.stack(class_labels)
            self.test_labels = tf.reshape(self.test_labels, (-1, 1))

            logger.info(f"Test batch: {self.test_batch.shape}, labels: {self.test_labels.shape}")

        except Exception as e:
            logger.warning(f"Failed to create test batch: {e}")
            self.test_batch = None
            self.test_labels = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if self.test_batch is None or self.test_labels is None:
            return

        if (epoch + 1) % self.config.monitor_every_n_epochs != 0:
            return

        try:
            builder = ConditionalBFUNetDatasetBuilder(self.config)
            builder.class_to_idx = self.class_to_idx

            # Add noise
            noisy_with_labels, clean = builder._add_noise(self.test_batch, self.test_labels[:, 0])
            noisy, labels = noisy_with_labels

            # Reshape labels for model
            labels = tf.reshape(labels, (-1, 1))

            # Get predictions
            model_output = self.model([noisy, labels], training=False)

            if isinstance(model_output, list):
                denoised = model_output[0]
            else:
                denoised = model_output

            self._save_comparison(epoch + 1, noisy, clean, denoised, labels)

        except Exception as e:
            logger.warning(f"Denoising visualization failed: {e}")

    def _save_comparison(
            self,
            epoch: int,
            noisy: tf.Tensor,
            clean: tf.Tensor,
            denoised: tf.Tensor,
            labels: tf.Tensor
    ):
        """Save comparison visualization with class labels."""
        try:
            num_samples = min(10, noisy.shape[0])

            fig, axes = plt.subplots(3, num_samples, figsize=(25, 7.5))
            fig.suptitle(f'Conditional Denoising Results - Epoch {epoch}', fontsize=20)

            for i in range(num_samples):
                clean_img = np.clip(clean[i].numpy(), 0, 1)
                noisy_img = np.clip(noisy[i].numpy(), 0, 1)
                denoised_img = np.clip(denoised[i].numpy(), 0, 1)
                class_idx = int(labels[i].numpy()[0])
                class_name = self.idx_to_class.get(class_idx, f'Class {class_idx}')

                if clean_img.shape[-1] == 1:
                    clean_img = clean_img.squeeze(-1)
                    noisy_img = noisy_img.squeeze(-1)
                    denoised_img = denoised_img.squeeze(-1)
                    cmap = 'gray'
                else:
                    cmap = None

                axes[0, i].imshow(clean_img, cmap=cmap, vmin=0, vmax=1)
                axes[0, i].axis('off')
                axes[0, i].set_title(class_name, fontsize=10)
                if i == 0:
                    axes[0, i].set_ylabel('Clean', fontsize=12)

                axes[1, i].imshow(noisy_img, cmap=cmap, vmin=0, vmax=1)
                axes[1, i].axis('off')
                if i == 0:
                    axes[1, i].set_ylabel('Noisy', fontsize=12)

                axes[2, i].imshow(denoised_img, cmap=cmap, vmin=0, vmax=1)
                axes[2, i].axis('off')
                if i == 0:
                    axes[2, i].set_ylabel('Denoised', fontsize=12)

            plt.tight_layout()
            save_path = self.output_dir / f"epoch_{epoch:03d}_comparison.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            gc.collect()

        except Exception as e:
            logger.warning(f"Failed to save comparison: {e}")


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

class ConditionalBFUNetTrainingPipeline(TrainingPipeline):
    """Training pipeline for conditional BFU-Net."""

    def __init__(self, config: ConditionalBFUNetConfig):
        super().__init__(config)
        self.config: ConditionalBFUNetConfig = config
        self.num_outputs = 1
        self.model_output_dims = None
        self.class_to_idx = None
        self.unconditional_token = None

    def _compile_model(self, model: keras.Model, total_steps: int):
        """Compile with deep supervision support."""
        logger.info("Compiling conditional BFU-Net model")

        # Analyze model structure
        has_multiple_outputs = isinstance(model.output, list)
        self.num_outputs = len(model.output) if has_multiple_outputs else 1

        if has_multiple_outputs:
            self.model_output_dims = [
                (out.shape[1], out.shape[2]) for out in model.output
            ]
            logger.info(f"Multi-output model: {self.num_outputs} outputs")

        # Create optimizer
        lr_schedule = self._create_lr_schedule(total_steps)
        optimizer = self._create_optimizer(lr_schedule)

        # Configure loss and metrics
        if has_multiple_outputs:
            loss_fns = ['mse'] * self.num_outputs
            initial_weights = [1.0 / self.num_outputs] * self.num_outputs
            metrics_for_primary = [
                keras.metrics.MeanAbsoluteError(name='mae'),
                keras.metrics.RootMeanSquaredError(name='rmse'),
                PrimaryOutputPSNR()
            ]
            metrics = {'final_output': metrics_for_primary}
        else:
            loss_fns = 'mse'
            initial_weights = None
            metrics = [
                keras.metrics.MeanAbsoluteError(name='mae'),
                keras.metrics.RootMeanSquaredError(name='rmse'),
                PrimaryOutputPSNR()
            ]

        model.compile(
            optimizer=optimizer,
            loss=loss_fns,
            loss_weights=initial_weights,
            metrics=metrics
        )

        logger.info("Model compiled")

    def _create_callbacks(
            self,
            lr_schedule: Optional[Any] = None,
            custom_callbacks: Optional[List[keras.callbacks.Callback]] = None
    ) -> List[keras.callbacks.Callback]:
        """Create callbacks with conditional support."""
        callbacks = super()._create_callbacks(lr_schedule, custom_callbacks)

        # Add deep supervision scheduler
        if self.config.enable_deep_supervision and self.num_outputs > 1:
            callbacks.append(
                DeepSupervisionWeightScheduler(self.config, self.num_outputs)
            )

        # Add conditional synthesis monitor
        if self.config.enable_synthesis and self.class_to_idx is not None:
            callbacks.append(
                ConditionalSynthesisMonitorCallback(
                    self.config,
                    self.experiment_dir,
                    self.class_to_idx,
                    self.unconditional_token
                )
            )

        # Add conditional denoising visualization
        if self.config.save_training_images and self.class_to_idx is not None:
            try:
                builder = ConditionalBFUNetDatasetBuilder(self.config)
                builder.class_to_idx = self.class_to_idx
                val_files, val_labels = builder._create_file_list_with_labels(
                    self.config.val_image_dirs,
                    limit=100
                )
                callbacks.append(
                    ConditionalDenoisingVisualizationCallback(
                        self.config,
                        val_files,
                        val_labels,
                        self.experiment_dir,
                        self.class_to_idx
                    )
                )
            except Exception as e:
                logger.warning(f"Could not add denoising visualization: {e}")

        return callbacks

    def run(
            self,
            model_builder: ModelBuilder,
            dataset_builder: DatasetBuilder,
            custom_callbacks: Optional[List[keras.callbacks.Callback]] = None
    ) -> Tuple[keras.Model, keras.callbacks.History]:
        """Run conditional training with inference model creation."""
        # Build dataset first to get class info
        if isinstance(dataset_builder, ConditionalBFUNetDatasetBuilder):
            # Trigger class discovery
            temp_files, temp_labels = dataset_builder._create_file_list_with_labels(
                self.config.train_image_dirs,
                limit=10
            )
            self.class_to_idx = dataset_builder.class_to_idx
            self.unconditional_token = dataset_builder.unconditional_token

            # Update config if num_classes not set
            if self.config.num_classes is None:
                self.config.num_classes = len(self.class_to_idx)

        # Build model first to get output dimensions
        temp_model = model_builder(self.config)

        if isinstance(temp_model.output, list):
            self.model_output_dims = [
                (out.shape[1], out.shape[2]) for out in temp_model.output
            ]
            if isinstance(dataset_builder, ConditionalBFUNetDatasetBuilder):
                dataset_builder.model_output_dims = self.model_output_dims

        del temp_model
        gc.collect()

        # Run training
        model, history = super().run(
            model_builder,
            dataset_builder,
            custom_callbacks
        )

        # Save inference model
        self._save_inference_model(model, model_builder)

        # Save class mapping
        self._save_class_mapping()

        return model, history

    def _save_class_mapping(self):
        """Save class to index mapping."""
        if self.class_to_idx is None:
            return

        try:
            import json
            mapping_path = self.experiment_dir / "class_mapping.json"

            mapping_data = {
                'class_to_idx': self.class_to_idx,
                'idx_to_class': {v: k for k, v in self.class_to_idx.items()},
                'num_classes': len(self.class_to_idx),
                'unconditional_token': self.unconditional_token
            }

            with open(mapping_path, 'w') as f:
                json.dump(mapping_data, f, indent=2)

            logger.info(f"Class mapping saved: {mapping_path}")

        except Exception as e:
            logger.warning(f"Failed to save class mapping: {e}")

    def _save_inference_model(self, trained_model: keras.Model, model_builder: ModelBuilder):
        """Save clean inference model."""
        try:
            logger.info("Creating inference model")

            inference_config = ConditionalBFUNetConfig(**self.config.__dict__)
            inference_config.input_shape = (None, None, self.config.channels)
            inference_config.enable_deep_supervision = self.config.enable_deep_supervision

            inference_model_full = model_builder(inference_config)
            inference_model_full.set_weights(trained_model.get_weights())

            # If multi-output, create single-output version
            if self.config.enable_deep_supervision and isinstance(inference_model_full.output, list):
                logger.info("Creating single-output inference model")
                inference_model = keras.Model(
                    inputs=inference_model_full.input,
                    outputs=inference_model_full.output[0],
                    name=f"{inference_model_full.name}_single_output"
                )
                del inference_model_full
            else:
                inference_model = inference_model_full

            inference_path = self.experiment_dir / "inference_model.keras"
            inference_model.save(inference_path)

            logger.info(f"Inference model saved: {inference_path}")

            del inference_model
            gc.collect()

        except Exception as e:
            logger.error(f"Failed to save inference model: {e}")


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_conditional_bfunet_denoiser(config: ConditionalBFUNetConfig) -> keras.Model:
    """Build conditional BFU-Net denoiser model."""
    variant = config.model_args.get('variant', 'tiny')

    logger.info(f"Building Conditional BFU-Net: {variant}")
    logger.info(f"Number of classes: {config.num_classes}")
    logger.info(f"Deep supervision: {config.enable_deep_supervision}")
    logger.info(f"CFG training: {config.enable_cfg_training}")

    if config.num_classes is None:
        raise ValueError("num_classes must be set before building model")

    from dl_techniques.models.bias_free_denoisers.bfunet import BFUNET_CONFIGS

    if variant in BFUNET_CONFIGS:
        model = create_conditional_bfunet_variant(
            variant=variant,
            input_shape=config.input_shape,
            num_classes=config.num_classes,
            enable_deep_supervision=config.enable_deep_supervision,
            enable_cfg_training=config.enable_cfg_training,
            class_embedding_dim=config.class_embedding_dim,
            class_injection_method=config.class_injection_method,
            cfg_dropout_prob=config.cfg_dropout_prob
        )
    elif variant == 'custom':
        model = create_conditional_bfunet_denoiser(
            input_shape=config.input_shape,
            num_classes=config.num_classes,
            depth=config.depth,
            initial_filters=config.filters,
            blocks_per_level=config.blocks_per_level,
            kernel_size=config.kernel_size,
            activation=config.activation,
            enable_deep_supervision=config.enable_deep_supervision,
            enable_cfg_training=config.enable_cfg_training,
            class_embedding_dim=config.class_embedding_dim,
            class_injection_method=config.class_injection_method,
            cfg_dropout_prob=config.cfg_dropout_prob,
            model_name=f"conditional_bfunet_custom_{config.experiment_name}"
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    logger.info(f"Model created with {model.count_params():,} parameters")
    return model


# =============================================================================
# CLI
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='Train Conditional BFU-Net with CFG support',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model-variant', type=str, default='tiny')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--channels', type=int, default=1, choices=[1, 3])

    parser.add_argument('--enable-deep-supervision', action='store_true', default=True)
    parser.add_argument('--no-deep-supervision', dest='enable_deep_supervision', action='store_false')

    parser.add_argument('--enable-cfg-training', action='store_true', default=True)
    parser.add_argument('--no-cfg-training', dest='enable_cfg_training', action='store_false')
    parser.add_argument('--cfg-dropout-prob', type=float, default=0.1)

    parser.add_argument('--enable-synthesis', action='store_true', default=True)
    parser.add_argument('--no-synthesis', dest='enable_synthesis', action='store_false')

    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--experiment-name', type=str, default=None)

    return parser


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training function."""
    parser = create_argument_parser()
    args = parser.parse_args()

    config = ConditionalBFUNetConfig(
        # Data paths - MODIFY FOR YOUR SETUP
        # Images should be organized in subfolders by class:
        # train_dir/
        #   class_0/
        #     image1.jpg
        #     image2.jpg
        #   class_1/
        #     image1.jpg
        #     image2.jpg
        train_image_dirs=[
            '/path/to/train/data',  # MODIFY THIS
        ],
        val_image_dirs=[
            '/path/to/val/data',  # MODIFY THIS
        ],

        # Model
        patch_size=args.patch_size,
        channels=args.channels,
        model_args={'variant': args.model_variant},

        # Training
        epochs=args.epochs,
        batch_size=args.batch_size,
        patches_per_image=16,

        # Conditional settings
        enable_cfg_training=args.enable_cfg_training,
        cfg_dropout_prob=args.cfg_dropout_prob,

        # Deep supervision
        enable_deep_supervision=args.enable_deep_supervision,
        deep_supervision_schedule_type='step_wise',

        # Synthesis
        enable_synthesis=args.enable_synthesis,
        synthesis_samples_per_class=2,
        synthesis_steps=200,
        synthesis_cfg_scales=[0.0, 1.0, 3.0, 7.5],

        # Optimization
        learning_rate=1e-3,
        optimizer_type='adamw',
        lr_schedule_type='cosine',
        weight_decay=1e-5,

        # Output
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        enable_visualization=True,
        enable_analysis=False,

        # Data organization
        organize_by_subfolders=True,
    )

    logger.info("=== Conditional BFU-Net Training Configuration ===")
    logger.info(f"Model: {config.model_args['variant']}")
    logger.info(f"CFG Training: {config.enable_cfg_training}")
    if config.enable_cfg_training:
        logger.info(f"  Dropout prob: {config.cfg_dropout_prob}")
    logger.info(f"Deep Supervision: {config.enable_deep_supervision}")
    logger.info(f"Synthesis: {config.enable_synthesis}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Patch size: {config.patch_size}x{config.patch_size}")

    try:
        dataset_builder = ConditionalBFUNetDatasetBuilder(config)
        pipeline = ConditionalBFUNetTrainingPipeline(config)

        logger.info("Starting conditional training...")
        model, history = pipeline.run(
            model_builder=build_conditional_bfunet_denoiser,
            dataset_builder=dataset_builder
        )

        logger.info("Training completed successfully!")
        logger.info(f"Results: {pipeline.experiment_dir}")

        model.summary(print_fn=logger.info)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()