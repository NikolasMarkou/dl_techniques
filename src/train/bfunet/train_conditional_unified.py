"""
Training Script for Unified Conditional Bias-Free U-Net

Supports training with multiple conditioning modalities:
- Dense conditioning: RGB images → depth maps
- Discrete conditioning: Class labels → images
- Hybrid conditioning: RGB + class → depth maps

Implements generalized conditional Miyasawa's theorem for all modalities.
"""

import gc
import json
import keras
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any, Union
from enum import Enum

from dl_techniques.utils.train_vision.framework import (
    TrainingConfig,
    DatasetBuilder,
    TrainingPipeline,
    ModelBuilder,
)
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import deep_supervision_schedule_builder
from dl_techniques.models.bias_free_denoisers.bfunet_conditional_unified import (
    create_unified_conditional_bfunet,
)
from dl_techniques.analyzer import DataInput


# =============================================================================
# CONDITIONING MODE ENUM
# =============================================================================

class ConditioningMode(str, Enum):
    """Enumeration of supported conditioning modes."""
    DENSE_ONLY = 'dense_only'  # RGB → depth
    DISCRETE_ONLY = 'discrete_only'  # class → image
    HYBRID = 'hybrid'  # RGB + class → depth


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class UnifiedBFUNetConfig(TrainingConfig):
    """
    Configuration for unified conditional BFU-Net training.

    Supports three conditioning modes:
    1. Dense only: RGB → depth (monocular depth estimation)
    2. Discrete only: class → image (conditional generation)
    3. Hybrid: RGB + class → depth (semantic-aware depth)

    Attributes:
        conditioning_mode: Type of conditioning ('dense_only', 'discrete_only', 'hybrid').

        # Target signal
        target_shape: Shape of target signal (depth or image).
        target_channels: Number of channels in target.

        # Dense conditioning (for RGB → depth)
        dense_data_dirs: Directories containing paired (RGB, depth) data.
        dense_rgb_dirs: Directories with RGB images.
        dense_depth_dirs: Directories with depth maps.
        dense_conditioning_shape: Shape of dense conditioning signal.
        dense_conditioning_encoder_filters: Base filters for dense encoder.
        dense_injection_method: Method for injecting dense features.

        # Discrete conditioning (for class → image)
        discrete_data_dirs: Directories with class-organized images.
        num_classes: Number of discrete classes.
        class_embedding_dim: Dimension of class embeddings.
        discrete_injection_method: Method for injecting class embeddings.
        organize_by_subfolders: Whether data is organized in class subfolders.

        # Hybrid conditioning
        hybrid_data_format: Format for hybrid data ('triplets', 'separate').

        # CFG (for discrete conditioning)
        enable_cfg_training: Enable classifier-free guidance.
        cfg_dropout_prob: Probability of dropping class labels.
        cfg_guidance_scales: CFG scales to test during synthesis.

        # Common settings
        patch_size: Size of training patches.
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

        # Deep supervision
        enable_deep_supervision: Use multi-scale supervision.
        deep_supervision_schedule_type: Weight scheduling strategy.
        deep_supervision_schedule_config: Schedule parameters.

        # Model architecture
        depth: Network depth.
        blocks_per_level: Residual blocks per level.
        filters: Base number of filters.
        kernel_size: Convolutional kernel size.
        activation: Activation function.

        # Visualization
        enable_synthesis: Enable synthesis monitoring.
        synthesis_samples: Number of samples to generate.
        synthesis_steps: Synthesis iteration steps.
        monitor_every_n_epochs: Visualization frequency.
        save_training_images: Save denoising samples.
    """

    # Conditioning mode
    conditioning_mode: ConditioningMode = ConditioningMode.DISCRETE_ONLY

    # Target signal
    patch_size: int = 128
    target_channels: int = 1

    # Dense conditioning
    dense_data_dirs: List[str] = field(default_factory=list)
    dense_rgb_dirs: List[str] = field(default_factory=list)
    dense_depth_dirs: List[str] = field(default_factory=list)
    dense_conditioning_channels: int = 3
    dense_conditioning_encoder_filters: int = 64
    dense_injection_method: str = 'addition'

    # Discrete conditioning
    discrete_data_dirs: List[str] = field(default_factory=list)
    num_classes: Optional[int] = None
    class_embedding_dim: int = 128
    discrete_injection_method: str = 'spatial_broadcast'
    organize_by_subfolders: bool = True

    # Hybrid conditioning
    hybrid_data_format: str = 'triplets'  # or 'separate'

    # CFG
    enable_cfg_training: bool = False
    cfg_dropout_prob: float = 0.1
    cfg_guidance_scales: List[float] = field(default_factory=lambda: [0.0, 1.0, 3.0, 7.5])

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

    # Model architecture
    depth: int = 3
    blocks_per_level: int = 2
    filters: int = 64
    kernel_size: int = 3
    activation: str = 'relu'

    # Synthesis
    enable_synthesis: bool = True
    synthesis_samples: int = 4
    synthesis_steps: int = 200
    synthesis_initial_step_size: float = 0.05
    synthesis_final_step_size: float = 0.8
    synthesis_initial_noise: float = 0.4
    synthesis_final_noise: float = 0.005

    # Monitoring
    monitor_every_n_epochs: int = 2
    save_training_images: bool = True

    def __post_init__(self):
        """Initialize and validate configuration."""
        # Determine shapes based on conditioning mode
        self.target_shape = (self.patch_size, self.patch_size, self.target_channels)

        if self.conditioning_mode in [ConditioningMode.DENSE_ONLY, ConditioningMode.HYBRID]:
            self.dense_conditioning_shape = (
                self.patch_size,
                self.patch_size,
                self.dense_conditioning_channels
            )
        else:
            self.dense_conditioning_shape = None

        self.input_shape = self.target_shape  # For compatibility with base class

        # Call parent
        super().__post_init__()

        # Validation
        if self.noise_sigma_min < 0 or self.noise_sigma_max <= self.noise_sigma_min:
            raise ValueError("Invalid noise sigma range")

        # Validate data directories based on mode
        if self.conditioning_mode == ConditioningMode.DENSE_ONLY:
            if not (self.dense_data_dirs or (self.dense_rgb_dirs and self.dense_depth_dirs)):
                raise ValueError("Dense conditioning requires data directories")

        elif self.conditioning_mode == ConditioningMode.DISCRETE_ONLY:
            if not self.discrete_data_dirs:
                raise ValueError("Discrete conditioning requires data directories")

        elif self.conditioning_mode == ConditioningMode.HYBRID:
            if not self.dense_data_dirs:
                raise ValueError("Hybrid conditioning requires data directories")


# =============================================================================
# DATASET BUILDERS
# =============================================================================

class UnifiedBFUNetDatasetBuilder(DatasetBuilder):
    """
    Unified dataset builder supporting all conditioning modalities.
    """

    def __init__(
            self,
            config: UnifiedBFUNetConfig,
            model_output_dims: Optional[List[Tuple[int, int]]] = None,
            class_to_idx: Optional[Dict[str, int]] = None
    ):
        """
        Initialize unified dataset builder.

        Args:
            config: UnifiedBFUNetConfig instance.
            model_output_dims: List of (height, width) for each output scale.
            class_to_idx: Optional mapping from class names to indices.
        """
        super().__init__(config)
        self.config: UnifiedBFUNetConfig = config
        self.model_output_dims = model_output_dims
        self.class_to_idx = class_to_idx
        self.unconditional_token = None

    def _discover_classes(self, directories: List[str]) -> Dict[str, int]:
        """Discover class structure from directory organization."""
        if not self.config.organize_by_subfolders:
            raise ValueError("Class discovery requires subfolder organization")

        class_names = set()

        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.is_dir():
                continue

            subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
            class_names.update([d.name for d in subdirs])

        if not class_names:
            raise ValueError(f"No class subdirectories found in {directories}")

        sorted_classes = sorted(class_names)
        class_to_idx = {name: idx for idx, name in enumerate(sorted_classes)}

        # Reserve last index for unconditional token if CFG enabled
        if self.config.enable_cfg_training:
            self.unconditional_token = len(class_to_idx)
            class_to_idx['__unconditional__'] = self.unconditional_token
            logger.info(f"Reserved class {self.unconditional_token} as unconditional token")

        logger.info(f"Discovered {len(sorted_classes)} classes: {sorted_classes}")

        return class_to_idx

    def _load_and_preprocess_target(self, path: tf.Tensor) -> tf.Tensor:
        """Load and preprocess target signal (depth or image)."""
        try:
            data_string = tf.io.read_file(path)
            data = tf.image.decode_image(
                data_string,
                channels=self.config.target_channels,
                expand_animations=False
            )
            data.set_shape([None, None, self.config.target_channels])

            data = tf.cast(data, tf.float32)
            if self.config.normalize_input:
                data = data / 255.0

            # Handle small images
            shape = tf.shape(data)
            height, width = shape[0], shape[1]
            min_dim = tf.minimum(height, width)
            min_size = self.config.patch_size

            def resize_if_small():
                scale = tf.cast(min_size, tf.float32) / tf.cast(min_dim, tf.float32)
                new_h = tf.cast(tf.math.ceil(tf.cast(height, tf.float32) * scale), tf.int32)
                new_w = tf.cast(tf.math.ceil(tf.cast(width, tf.float32) * scale), tf.int32)
                return tf.image.resize(data, [new_h, new_w])

            data = tf.cond(
                tf.logical_or(height < min_size, width < min_size),
                true_fn=resize_if_small,
                false_fn=lambda: data
            )

            patch = tf.image.random_crop(
                data,
                [self.config.patch_size, self.config.patch_size, self.config.target_channels]
            )

            return patch

        except tf.errors.InvalidArgumentError:
            logger.warning(f"Failed to load target: {path}")
            return tf.zeros(
                [self.config.patch_size, self.config.patch_size, self.config.target_channels],
                dtype=tf.float32
            )

    def _load_and_preprocess_dense_condition(self, path: tf.Tensor) -> tf.Tensor:
        """Load and preprocess dense conditioning signal (RGB image)."""
        try:
            data_string = tf.io.read_file(path)
            data = tf.image.decode_image(
                data_string,
                channels=self.config.dense_conditioning_channels,
                expand_animations=False
            )
            data.set_shape([None, None, self.config.dense_conditioning_channels])

            data = tf.cast(data, tf.float32)
            if self.config.normalize_input:
                data = data / 255.0

            # Resize to match target dimensions
            data = tf.image.resize(data, [self.config.patch_size, self.config.patch_size])

            return data

        except tf.errors.InvalidArgumentError:
            logger.warning(f"Failed to load dense condition: {path}")
            return tf.zeros(
                [self.config.patch_size, self.config.patch_size,
                 self.config.dense_conditioning_channels],
                dtype=tf.float32
            )

    def _augment_data(self, *data_tensors):
        """Apply synchronized data augmentation."""
        if not self.config.augment_data:
            return data_tensors

        # Horizontal flip
        if tf.random.uniform([]) > 0.5:
            data_tensors = tuple(tf.image.flip_left_right(d) for d in data_tensors)

        # Vertical flip
        if tf.random.uniform([]) > 0.5:
            data_tensors = tuple(tf.image.flip_up_down(d) for d in data_tensors)

        # 90-degree rotations
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        data_tensors = tuple(tf.image.rot90(d, k) for d in data_tensors)

        return data_tensors

    def _add_noise(self, target: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Add Gaussian noise to target."""
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

        noise = tf.random.normal(tf.shape(target)) * noise_level
        noisy_target = target + noise
        noisy_target = tf.clip_by_value(noisy_target, 0.0, 1.0)

        return noisy_target, target

    def _apply_cfg_dropout(self, class_label: tf.Tensor) -> tf.Tensor:
        """Apply CFG dropout to class labels."""
        if not self.config.enable_cfg_training or self.unconditional_token is None:
            return class_label

        should_drop = tf.random.uniform([]) < self.config.cfg_dropout_prob

        return tf.cond(
            should_drop,
            lambda: tf.constant(self.unconditional_token, dtype=class_label.dtype),
            lambda: class_label
        )

    def _create_multiscale_labels(
            self,
            target: tf.Tensor
    ) -> Tuple[tf.Tensor, ...]:
        """Create multi-scale labels for deep supervision."""
        if self.model_output_dims is None:
            return (target,)

        labels = [
            tf.image.resize(target, dim)
            for dim in self.model_output_dims
        ]
        return tuple(labels)

    def build(self) -> Tuple[
        tf.data.Dataset,
        tf.data.Dataset,
        Optional[int],
        Optional[int]
    ]:
        """Build datasets based on conditioning mode."""
        logger.info(f"Building unified dataset: mode={self.config.conditioning_mode}")

        if self.config.conditioning_mode == ConditioningMode.DISCRETE_ONLY:
            return self._build_discrete_only()
        elif self.config.conditioning_mode == ConditioningMode.DENSE_ONLY:
            return self._build_dense_only()
        elif self.config.conditioning_mode == ConditioningMode.HYBRID:
            return self._build_hybrid()
        else:
            raise ValueError(f"Unknown conditioning mode: {self.config.conditioning_mode}")

    def _build_discrete_only(self):
        """Build dataset for discrete conditioning (class → image)."""
        logger.info("Building discrete-only conditioning dataset")

        # Discover classes
        if self.class_to_idx is None:
            self.class_to_idx = self._discover_classes(self.config.discrete_data_dirs)

        if self.config.num_classes is None:
            self.config.num_classes = len(self.class_to_idx)

        # Create file lists with labels
        def get_files_with_labels(dirs, limit):
            all_files = []
            all_labels = []
            extensions_set = {ext.lower() for ext in self.config.image_extensions}
            extensions_set.update({ext.upper() for ext in self.config.image_extensions})

            for directory in dirs:
                dir_path = Path(directory)
                if not dir_path.is_dir():
                    continue

                for class_name, class_idx in self.class_to_idx.items():
                    if class_name == '__unconditional__':
                        continue

                    class_dir = dir_path / class_name
                    if not class_dir.is_dir():
                        continue

                    for file_path in class_dir.rglob("*"):
                        if file_path.is_file() and file_path.suffix in extensions_set:
                            all_files.append(str(file_path))
                            all_labels.append(class_idx)

            if limit and limit < len(all_files):
                indices = np.random.choice(len(all_files), limit, replace=False)
                all_files = [all_files[i] for i in indices]
                all_labels = [all_labels[i] for i in indices]

            return all_files, all_labels

        train_files, train_labels = get_files_with_labels(
            self.config.discrete_data_dirs,
            self.config.max_train_files
        )

        # For validation, reuse training dirs (would typically be separate)
        val_files, val_labels = get_files_with_labels(
            self.config.discrete_data_dirs,
            min(len(train_files) // 10, 1000)  # 10% or 1000 samples
        )

        logger.info(f"Found {len(train_files)} training, {len(val_files)} validation samples")

        # Build training dataset
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

        # Load and process
        def process_discrete(path, label):
            target = self._load_and_preprocess_target(path)
            target, = self._augment_data(target)
            noisy, clean = self._add_noise(target)
            label = self._apply_cfg_dropout(label)

            if self.config.enable_deep_supervision and self.model_output_dims:
                clean_multiscale = self._create_multiscale_labels(clean)
                return (noisy, label), clean_multiscale
            else:
                return (noisy, label), clean

        train_ds = train_ds.map(process_discrete, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.batch(self.config.batch_size, drop_remainder=True)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        # Build validation dataset (similar but no CFG dropout)
        val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
        val_ds = val_ds.repeat()

        def process_val_discrete(path, label):
            target = self._load_and_preprocess_target(path)
            noisy, clean = self._add_noise(target)

            if self.config.enable_deep_supervision and self.model_output_dims:
                clean_multiscale = self._create_multiscale_labels(clean)
                return (noisy, label), clean_multiscale
            else:
                return (noisy, label), clean

        val_ds = val_ds.map(process_val_discrete, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.config.batch_size)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

        steps_per_epoch = len(train_files) * self.config.patches_per_image // self.config.batch_size
        val_steps = len(val_files) // self.config.batch_size

        return train_ds, val_ds, steps_per_epoch, val_steps

    def _build_dense_only(self):
        """Build dataset for dense conditioning (RGB → depth)."""
        logger.info("Building dense-only conditioning dataset")

        # TODO: Implement paired (RGB, depth) data loading
        # This requires specific dataset formats (e.g., NYU Depth V2, KITTI)
        raise NotImplementedError(
            "Dense-only conditioning requires custom dataset implementation. "
            "Please implement paired RGB-depth data loading for your specific dataset."
        )

    def _build_hybrid(self):
        """Build dataset for hybrid conditioning (RGB + class → depth)."""
        logger.info("Building hybrid conditioning dataset")

        # TODO: Implement triplet (RGB, class, depth) data loading
        raise NotImplementedError(
            "Hybrid conditioning requires custom dataset implementation. "
            "Please implement RGB-class-depth triplet loading for your specific dataset."
        )

    def get_test_data(self) -> Optional[DataInput]:
        """Get test data for analysis."""
        return None


# =============================================================================
# SYNTHESIS AND VISUALIZATION
# =============================================================================

def conditional_sampling_with_cfg(
        denoiser: keras.Model,
        class_labels: np.ndarray,
        unconditional_token: int,
        num_samples: int,
        image_shape: Tuple[int, int, int],
        num_steps: int = 200,
        initial_step_size: float = 0.1,
        final_step_size: float = 1.0,
        initial_noise_level: float = 0.5,
        final_noise_level: float = 0.01,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        save_intermediate: bool = True
) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    """Generate samples using Classifier-Free Guidance."""
    if seed is not None:
        tf.random.set_seed(seed)

    logger.info(f"Conditional sampling with CFG (w={guidance_scale})")

    y = tf.random.normal([num_samples] + list(image_shape), mean=0.5, stddev=0.3)
    y = tf.clip_by_value(y, 0.0, 1.0)

    intermediate_steps = []
    step_sizes = tf.linspace(initial_step_size, final_step_size, num_steps)
    noise_levels = tf.linspace(initial_noise_level, final_noise_level, num_steps)

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
            model_output = denoiser([y, unconditional_labels], training=False)
        elif guidance_scale == 1.0:
            model_output = denoiser([y, class_labels_tf], training=False)
        else:
            y_double = tf.concat([y, y], axis=0)
            labels_double = tf.concat([class_labels_tf, unconditional_labels], axis=0)

            output_double = denoiser([y_double, labels_double], training=False)

            if isinstance(output_double, list):
                denoised_double = output_double[0]
            else:
                denoised_double = output_double

            denoised_cond, denoised_uncond = tf.split(denoised_double, 2, axis=0)
            model_output = denoised_uncond + guidance_scale * (denoised_cond - denoised_uncond)

        if isinstance(model_output, list):
            denoised = model_output[0]
        else:
            denoised = model_output

        d_t = denoised - y
        z_t = tf.random.normal(tf.shape(y))
        y = y + h_t * d_t + gamma_t * z_t
        y = tf.clip_by_value(y, 0.0, 1.0)

        if save_intermediate and (t % (num_steps // 10) == 0 or t == num_steps - 1):
            intermediate_steps.append(y.numpy().copy())

        if t % (num_steps // 5) == 0:
            logger.info(f"Step {t}/{num_steps}: h_t={h_t:.4f}, γ_t={gamma_t:.4f}")

    logger.info("Sampling completed")
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

    def __init__(self, config: UnifiedBFUNetConfig, num_outputs: int):
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
    """Monitor conditional synthesis for discrete conditioning."""

    def __init__(
            self,
            config: UnifiedBFUNetConfig,
            output_dir: Path,
            class_to_idx: Optional[Dict[str, int]] = None,
            unconditional_token: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        self.output_dir = output_dir / "synthesis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_to_idx = class_to_idx
        self.unconditional_token = unconditional_token

        if class_to_idx:
            self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        else:
            self.idx_to_class = {}

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if not self.config.enable_synthesis:
            return

        if self.config.conditioning_mode != ConditioningMode.DISCRETE_ONLY:
            return

        if (epoch + 1) % self.config.monitor_every_n_epochs != 0:
            return

        if self.class_to_idx is None or self.unconditional_token is None:
            return

        try:
            logger.info(f"Generating conditional samples at epoch {epoch + 1}")

            shape = (
                self.config.patch_size,
                self.config.patch_size,
                self.config.target_channels
            )

            # Select classes to visualize
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

            # Generate for each CFG scale
            results = {}
            for cfg_scale in self.config.cfg_guidance_scales:
                logger.info(f"  CFG scale {cfg_scale}")

                class_labels = np.repeat(
                    selected_classes,
                    self.config.synthesis_samples
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

            self._visualize_synthesis(epoch + 1, results, selected_classes)

        except Exception as e:
            logger.warning(f"Synthesis failed: {e}")

    def _visualize_synthesis(
            self,
            epoch: int,
            results: Dict[float, Tuple],
            selected_classes: np.ndarray
    ):
        """Visualize synthesis results."""
        try:
            num_cfg_scales = len(results)
            num_classes = len(selected_classes)

            fig, axes = plt.subplots(
                num_cfg_scales,
                num_classes,
                figsize=(3 * num_classes, 3 * num_cfg_scales)
            )

            if num_cfg_scales == 1:
                axes = axes.reshape(1, -1)
            if num_classes == 1:
                axes = axes.reshape(-1, 1)

            fig.suptitle(f'Conditional Synthesis - Epoch {epoch}', fontsize=16)

            for row_idx, (cfg_scale, (generated, class_labels, _)) in enumerate(results.items()):
                for col_idx, class_idx in enumerate(selected_classes):
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

                    axes[row_idx, col_idx].axis('off')

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
            save_path = self.output_dir / f"epoch_{epoch:03d}_synthesis.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            gc.collect()

        except Exception as e:
            logger.warning(f"Visualization failed: {e}")


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

class UnifiedBFUNetTrainingPipeline(TrainingPipeline):
    """Training pipeline for unified conditional BFU-Net."""

    def __init__(self, config: UnifiedBFUNetConfig):
        super().__init__(config)
        self.config: UnifiedBFUNetConfig = config
        self.num_outputs = 1
        self.model_output_dims = None
        self.class_to_idx = None
        self.unconditional_token = None

    def _compile_model(self, model: keras.Model, total_steps: int):
        """Compile model."""
        logger.info("Compiling unified conditional BFU-Net")

        has_multiple_outputs = isinstance(model.output, list)
        self.num_outputs = len(model.output) if has_multiple_outputs else 1

        if has_multiple_outputs:
            self.model_output_dims = [
                (out.shape[1], out.shape[2]) for out in model.output
            ]
            logger.info(f"Multi-output model: {self.num_outputs} outputs")

        lr_schedule = self._create_lr_schedule(total_steps)
        optimizer = self._create_optimizer(lr_schedule)

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
        """Create callbacks."""
        callbacks = super()._create_callbacks(lr_schedule, custom_callbacks)

        if self.config.enable_deep_supervision and self.num_outputs > 1:
            callbacks.append(
                DeepSupervisionWeightScheduler(self.config, self.num_outputs)
            )

        if self.config.conditioning_mode == ConditioningMode.DISCRETE_ONLY:
            if self.config.enable_synthesis and self.class_to_idx is not None:
                callbacks.append(
                    ConditionalSynthesisMonitorCallback(
                        self.config,
                        self.experiment_dir,
                        self.class_to_idx,
                        self.unconditional_token
                    )
                )

        return callbacks

    def run(
            self,
            model_builder: ModelBuilder,
            dataset_builder: DatasetBuilder,
            custom_callbacks: Optional[List[keras.callbacks.Callback]] = None
    ) -> Tuple[keras.Model, keras.callbacks.History]:
        """Run training."""
        # Get class info if discrete conditioning
        if self.config.conditioning_mode in [ConditioningMode.DISCRETE_ONLY, ConditioningMode.HYBRID]:
            if isinstance(dataset_builder, UnifiedBFUNetDatasetBuilder):
                if self.config.conditioning_mode == ConditioningMode.DISCRETE_ONLY:
                    temp_files = list(Path(self.config.discrete_data_dirs[0]).rglob("*"))[:10]
                    dataset_builder.class_to_idx = dataset_builder._discover_classes(
                        self.config.discrete_data_dirs
                    )
                    self.class_to_idx = dataset_builder.class_to_idx
                    self.unconditional_token = dataset_builder.unconditional_token

                    if self.config.num_classes is None:
                        self.config.num_classes = len(self.class_to_idx)

        # Build model to get output dimensions
        temp_model = model_builder(self.config)

        if isinstance(temp_model.output, list):
            self.model_output_dims = [
                (out.shape[1], out.shape[2]) for out in temp_model.output
            ]
            if isinstance(dataset_builder, UnifiedBFUNetDatasetBuilder):
                dataset_builder.model_output_dims = self.model_output_dims

        del temp_model
        gc.collect()

        # Run training
        model, history = super().run(
            model_builder,
            dataset_builder,
            custom_callbacks
        )

        # Save artifacts
        self._save_inference_model(model, model_builder)
        self._save_metadata()

        return model, history

    def _save_metadata(self):
        """Save conditioning metadata."""
        try:
            metadata = {
                'conditioning_mode': self.config.conditioning_mode,
                'target_shape': self.config.target_shape,
            }

            if self.config.conditioning_mode in [ConditioningMode.DISCRETE_ONLY, ConditioningMode.HYBRID]:
                if self.class_to_idx:
                    metadata['class_to_idx'] = self.class_to_idx
                    metadata['idx_to_class'] = {v: k for k, v in self.class_to_idx.items()}
                    metadata['num_classes'] = len(self.class_to_idx)
                    metadata['unconditional_token'] = self.unconditional_token

            if self.config.conditioning_mode in [ConditioningMode.DENSE_ONLY, ConditioningMode.HYBRID]:
                metadata['dense_conditioning_shape'] = self.config.dense_conditioning_shape

            metadata_path = self.experiment_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Metadata saved: {metadata_path}")

        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")

    def _save_inference_model(self, trained_model: keras.Model, model_builder: ModelBuilder):
        """Save inference model."""
        try:
            logger.info("Creating inference model")

            inference_config = UnifiedBFUNetConfig(**self.config.__dict__)
            inference_config.target_shape = (None, None, self.config.target_channels)

            if self.config.conditioning_mode in [ConditioningMode.DENSE_ONLY, ConditioningMode.HYBRID]:
                inference_config.dense_conditioning_shape = (
                    None, None, self.config.dense_conditioning_channels
                )

            inference_model_full = model_builder(inference_config)
            inference_model_full.set_weights(trained_model.get_weights())

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

def build_unified_bfunet(config: UnifiedBFUNetConfig) -> keras.Model:
    """Build unified conditional BFU-Net based on configuration."""
    variant = config.model_args.get('variant', 'custom')

    logger.info(f"Building Unified BFU-Net: mode={config.conditioning_mode}")

    # Determine conditioning shapes
    dense_shape = config.dense_conditioning_shape if config.conditioning_mode in [
        ConditioningMode.DENSE_ONLY, ConditioningMode.HYBRID
    ] else None

    num_classes = config.num_classes if config.conditioning_mode in [
        ConditioningMode.DISCRETE_ONLY, ConditioningMode.HYBRID
    ] else None

    # Build appropriate model
    if variant == 'custom':
        model = create_unified_conditional_bfunet(
            target_shape=config.target_shape,
            dense_conditioning_shape=dense_shape,
            num_classes=num_classes,
            depth=config.depth,
            initial_filters=config.filters,
            blocks_per_level=config.blocks_per_level,
            kernel_size=config.kernel_size,
            activation=config.activation,
            dense_conditioning_encoder_filters=config.dense_conditioning_encoder_filters,
            dense_injection_method=config.dense_injection_method,
            class_embedding_dim=config.class_embedding_dim,
            discrete_injection_method=config.discrete_injection_method,
            enable_cfg_training=config.enable_cfg_training,
            cfg_dropout_prob=config.cfg_dropout_prob,
            enable_deep_supervision=config.enable_deep_supervision,
            model_name=f"unified_bfunet_{config.experiment_name}"
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
        description='Train Unified Conditional BFU-Net',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--conditioning-mode',
        type=str,
        choices=['dense_only', 'discrete_only', 'hybrid'],
        default='discrete_only',
        help='Conditioning modality'
    )
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--target-channels', type=int, default=1)

    parser.add_argument('--enable-deep-supervision', action='store_true', default=True)
    parser.add_argument('--no-deep-supervision', dest='enable_deep_supervision', action='store_false')

    parser.add_argument('--enable-cfg-training', action='store_true', default=False)
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

    # Example configuration for discrete-only (class-conditional)
    config = UnifiedBFUNetConfig(
        # Conditioning mode
        conditioning_mode=ConditioningMode(args.conditioning_mode),

        # Data paths - MODIFY FOR YOUR SETUP
        discrete_data_dirs=[
            '/path/to/class/organized/data',  # MODIFY
        ],

        # Target
        patch_size=args.patch_size,
        target_channels=args.target_channels,

        # Model
        model_args={'variant': 'custom'},
        depth=3,
        blocks_per_level=2,
        filters=64,

        # Training
        epochs=args.epochs,
        batch_size=args.batch_size,
        patches_per_image=16,

        # Conditioning
        enable_cfg_training=args.enable_cfg_training,
        cfg_dropout_prob=args.cfg_dropout_prob,

        # Deep supervision
        enable_deep_supervision=args.enable_deep_supervision,

        # Synthesis
        enable_synthesis=args.enable_synthesis,
        synthesis_samples=2,
        synthesis_steps=200,

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
    )

    logger.info("=== Unified Conditional BFU-Net Training ===")
    logger.info(f"Conditioning mode: {config.conditioning_mode}")
    logger.info(f"Deep supervision: {config.enable_deep_supervision}")
    logger.info(f"CFG training: {config.enable_cfg_training}")
    logger.info(f"Epochs: {config.epochs}")

    try:
        dataset_builder = UnifiedBFUNetDatasetBuilder(config)
        pipeline = UnifiedBFUNetTrainingPipeline(config)

        logger.info("Starting training...")
        model, history = pipeline.run(
            model_builder=build_unified_bfunet,
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