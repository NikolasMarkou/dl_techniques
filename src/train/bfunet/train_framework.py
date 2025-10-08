import gc
import keras
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any, Union

from dl_techniques.optimization.train_vision import (
    TrainingConfig,
    DatasetBuilder,
    TrainingPipeline,
    ModelBuilder
)
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    deep_supervision_schedule_builder,
    learning_rate_schedule_builder,
    optimizer_builder
)

from dl_techniques.models.bias_free_denoisers.bfunet import (
    create_bfunet_denoiser,
    BFUNET_CONFIGS,
    create_bfunet_variant,
)
from dl_techniques.analyzer import DataInput


# =============================================================================
# EXTENDED CONFIGURATION FOR DEEP SUPERVISION
# =============================================================================

@dataclass
class BFUNetConfig(TrainingConfig):
    """
    Extended configuration for BFU-Net with deep supervision.

    Attributes:
        train_image_dirs: Directories containing training images.
        val_image_dirs: Directories containing validation images.
        patch_size: Size of training patches.
        channels: Number of input channels.
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
        enable_synthesis: Enable image synthesis monitoring.
        synthesis_samples: Number of images to synthesize.
        synthesis_steps: Synthesis iteration steps.
        synthesis_initial_step_size: Initial gradient step size.
        synthesis_final_step_size: Final gradient step size.
        synthesis_initial_noise: Initial noise injection.
        synthesis_final_noise: Final noise injection.
        monitor_every_n_epochs: Visualization frequency.
        save_training_images: Save denoising samples.
    """
    # Data paths
    train_image_dirs: List[str] = field(default_factory=list)
    val_image_dirs: List[str] = field(default_factory=list)

    # Override input shape
    patch_size: int = 128
    channels: int = 1

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

    # Image synthesis
    enable_synthesis: bool = True
    synthesis_samples: int = 10
    synthesis_steps: int = 200
    synthesis_initial_step_size: float = 0.05
    synthesis_final_step_size: float = 0.8
    synthesis_initial_noise: float = 0.4
    synthesis_final_noise: float = 0.005

    # Monitoring
    monitor_every_n_epochs: int = 2
    save_training_images: bool = True
    monitor_metric: str = 'val_final_output_primary_psnr'
    monitor_mode: str = 'max'

    visualization_frequency: int = 1
    enable_gradient_tracking: bool = True
    enable_gradient_topology_viz: bool = True

    def __post_init__(self):
        """Initialize and validate configuration."""
        self.input_shape = (self.patch_size, self.patch_size, self.channels)
        self.num_classes = self.channels

        # Call parent
        super().__post_init__()

        # Additional validation
        if self.noise_sigma_min < 0 or self.noise_sigma_max <= self.noise_sigma_min:
            raise ValueError("Invalid noise sigma range")

        if not self.train_image_dirs:
            raise ValueError("No training directories specified")

        if not self.val_image_dirs:
            raise ValueError("No validation directories specified")


# =============================================================================
# DEEP SUPERVISION DATASET BUILDER
# =============================================================================

class BFUNetDatasetBuilder(DatasetBuilder):
    """
    Dataset builder with multi-scale label generation for deep supervision.

    This builder creates datasets that output multi-scale labels matching
    the model's output structure for deep supervision training.
    """

    def __init__(self, config: BFUNetConfig, model_output_dims: Optional[List[Tuple[int, int]]] = None):
        """
        Initialize dataset builder.

        Args:
            config: BFUNetConfig instance.
            model_output_dims: List of (height, width) tuples for each output scale.
        """
        super().__init__(config)
        self.config: BFUNetConfig = config
        self.model_output_dims = model_output_dims

    def _load_and_preprocess_image(self, image_path: tf.Tensor) -> tf.Tensor:
        """Load and preprocess image."""
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

            return patch

        except tf.errors.InvalidArgumentError:
            logger.warning(f"Failed to load image: {image_path}")
            return tf.zeros(
                [self.config.patch_size, self.config.patch_size, self.config.channels],
                dtype=tf.float32
            )

    def _augment_patch(self, patch: tf.Tensor) -> tf.Tensor:
        """Apply data augmentation."""
        patch = tf.image.random_flip_left_right(patch)
        patch = tf.image.random_flip_up_down(patch)
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        patch = tf.image.rot90(patch, k)
        return patch

    def _add_noise(self, patch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Add Gaussian noise."""
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

        return noisy_patch, patch

    def _create_file_list(
            self,
            directories: List[str],
            limit: Optional[int] = None
    ) -> List[str]:
        """Create unified file list."""
        all_files = []
        extensions_set = {ext.lower() for ext in self.config.image_extensions}
        extensions_set.update({ext.upper() for ext in self.config.image_extensions})

        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.is_dir():
                logger.warning(f"Directory not found: {directory}")
                continue

            try:
                for file_path in dir_path.rglob("*"):
                    if file_path.is_file() and file_path.suffix in extensions_set:
                        all_files.append(str(file_path))
            except Exception as e:
                logger.warning(f"Error scanning {directory}: {e}")
                continue

        if not all_files:
            raise ValueError(f"No files found in directories: {directories}")

        logger.info(f"Found {len(all_files)} files")

        if limit and limit < len(all_files):
            logger.info(f"Limiting to {limit} files")
            np.random.shuffle(all_files)
            all_files = all_files[:limit]

        return all_files

    def _create_multiscale_labels(
            self,
            noisy_patch: tf.Tensor,
            clean_patch: tf.Tensor
    ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, ...]]:
        """
        Create multi-scale labels for deep supervision.

        Args:
            noisy_patch: Noisy input.
            clean_patch: Clean ground truth.

        Returns:
            Tuple of (noisy, tuple_of_multiscale_labels).
        """
        if self.model_output_dims is None:
            return noisy_patch, clean_patch

        labels = [
            tf.image.resize(clean_patch, dim)
            for dim in self.model_output_dims
        ]
        return noisy_patch, tuple(labels)

    def build(self) -> Tuple[
        tf.data.Dataset,
        tf.data.Dataset,
        Optional[int],
        Optional[int]
    ]:
        """Build training and validation datasets."""
        logger.info("Building BFU-Net datasets")

        # Create file lists
        train_files = self._create_file_list(
            self.config.train_image_dirs,
            self.config.max_train_files
        )
        val_files = self._create_file_list(
            self.config.val_image_dirs,
            self.config.max_val_files
        )

        # Create training dataset
        train_ds = tf.data.Dataset.from_tensor_slices(train_files)
        train_ds = train_ds.shuffle(
            buffer_size=min(self.config.dataset_shuffle_buffer, len(train_files)),
            reshuffle_each_iteration=True
        )
        train_ds = train_ds.repeat()

        if self.config.patches_per_image > 1:
            train_ds = train_ds.flat_map(
                lambda path: tf.data.Dataset.from_tensors(path).repeat(
                    self.config.patches_per_image
                )
            )

        train_ds = train_ds.map(
            self._load_and_preprocess_image,
            num_parallel_calls=self.config.parallel_reads
        )
        train_ds = train_ds.filter(lambda x: tf.reduce_sum(tf.abs(x)) > 0)

        if self.config.augment_data:
            train_ds = train_ds.map(
                self._augment_patch,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        train_ds = train_ds.map(
            self._add_noise,
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

        # Create validation dataset
        val_ds = tf.data.Dataset.from_tensor_slices(val_files)
        val_ds = val_ds.repeat()

        if self.config.patches_per_image > 1:
            val_ds = val_ds.flat_map(
                lambda path: tf.data.Dataset.from_tensors(path).repeat(
                    self.config.patches_per_image
                )
            )

        val_ds = val_ds.map(
            self._load_and_preprocess_image,
            num_parallel_calls=self.config.parallel_reads
        )
        val_ds = val_ds.filter(lambda x: tf.reduce_sum(tf.abs(x)) > 0)
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
        return None  # Analysis not applicable for denoising


# =============================================================================
# METRICS AND LOSSES
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
# IMAGE SYNTHESIS
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
    Generate images using implicit prior sampling.

    Implements Kadkhodaie & Simoncelli algorithm for image synthesis.
    """
    if seed is not None:
        tf.random.set_seed(seed)

    logger.info(f"Starting synthesis: {num_samples} samples, {num_steps} steps")

    y = tf.random.normal([num_samples] + list(image_shape), mean=0.5, stddev=0.3)
    y = tf.clip_by_value(y, 0.0, 1.0)

    intermediate_steps = []
    step_sizes = tf.linspace(initial_step_size, final_step_size, num_steps)
    noise_levels = tf.linspace(initial_noise_level, final_noise_level, num_steps)

    for t in range(num_steps):
        h_t = step_sizes[t]
        gamma_t = noise_levels[t]

        model_output = denoiser(y, training=False)

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

    logger.info("Synthesis completed")
    return y, intermediate_steps


# =============================================================================
# CUSTOM CALLBACKS
# =============================================================================

class DeepSupervisionWeightScheduler(keras.callbacks.Callback):
    """Dynamic weight scheduler for deep supervision."""

    def __init__(self, config: BFUNetConfig, num_outputs: int):
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
        for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
            weights = self.scheduler(progress)
            logger.info(f"  Progress {progress:.2f}: {weights}")

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        progress = min(1.0, epoch / max(1, self.total_epochs - 1))
        new_weights = self.scheduler(progress)
        self.model.loss_weights = new_weights

        weights_str = ", ".join([f"{w:.4f}" for w in new_weights])
        logger.info(f"Epoch {epoch + 1} - DS weights: [{weights_str}]")


class SynthesisMonitorCallback(keras.callbacks.Callback):
    """Monitor image synthesis using implicit prior."""

    def __init__(self, config: BFUNetConfig, output_dir: Path):
        super().__init__()
        self.config = config
        self.output_dir = output_dir / "synthesis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if not self.config.enable_synthesis:
            return

        if (epoch + 1) % self.config.monitor_every_n_epochs != 0:
            return

        try:
            logger.info(f"Generating synthetic images at epoch {epoch + 1}")

            shape = (self.config.patch_size, self.config.patch_size, self.config.channels)

            generated, intermediates = unconditional_sampling(
                denoiser=self.model,
                num_samples=self.config.synthesis_samples,
                image_shape=shape,
                num_steps=self.config.synthesis_steps,
                initial_step_size=self.config.synthesis_initial_step_size,
                final_step_size=self.config.synthesis_final_step_size,
                initial_noise_level=self.config.synthesis_initial_noise,
                final_noise_level=self.config.synthesis_final_noise,
                seed=epoch + 1,
                save_intermediate=True
            )

            self._visualize_synthesis(epoch + 1, generated, intermediates)

            mean_val = tf.reduce_mean(generated)
            std_val = tf.math.reduce_std(generated)
            logger.info(f"Synthesis - Mean: {mean_val:.3f}, Std: {std_val:.3f}")

        except Exception as e:
            logger.warning(f"Synthesis failed: {e}")

    def _visualize_synthesis(
            self,
            epoch: int,
            final: tf.Tensor,
            intermediates: List[tf.Tensor]
    ):
        """Visualize synthesis evolution."""
        try:
            num_samples = min(4, final.shape[0])
            num_steps = len(intermediates)

            fig, axes = plt.subplots(num_samples, num_steps, figsize=(3 * num_steps, 3 * num_samples))
            fig.suptitle(f'Synthesis Evolution - Epoch {epoch}', fontsize=16)

            if num_samples == 1:
                axes = axes.reshape(1, -1)

            for i in range(num_samples):
                for j, step_data in enumerate(intermediates):
                    img = step_data[i]

                    if img.shape[-1] == 1:
                        img = img.squeeze(-1)
                        cmap = 'gray'
                    else:
                        cmap = None

                    img = np.clip(img, 0, 1)
                    axes[i, j].imshow(img, cmap=cmap, vmin=0, vmax=1)
                    axes[i, j].axis('off')

                    if i == 0:
                        step_num = j * (self.config.synthesis_steps // (
                                    num_steps - 1)) if j < num_steps - 1 else self.config.synthesis_steps
                        axes[i, j].set_title(f'Step {step_num}', fontsize=10)

            plt.tight_layout()
            save_path = self.output_dir / f"epoch_{epoch:03d}_synthesis.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            gc.collect()

        except Exception as e:
            logger.warning(f"Visualization failed: {e}")


class DenoisingVisualizationCallback(keras.callbacks.Callback):
    """Visualize denoising results."""

    def __init__(self, config: BFUNetConfig, val_files: List[str], output_dir: Path):
        super().__init__()
        self.config = config
        self.output_dir = output_dir / "denoising"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._create_test_batch(val_files)

    def _create_test_batch(self, val_files: List[str]):
        """Create fixed test batch."""
        try:
            builder = BFUNetDatasetBuilder(self.config)

            clean_patches = []
            for file_path in val_files[:10]:
                path_tensor = tf.constant(file_path)
                patch = builder._load_and_preprocess_image(path_tensor)
                clean_patches.append(patch)

            self.test_batch = tf.stack(clean_patches)
            logger.info(f"Test batch: {self.test_batch.shape}")

        except Exception as e:
            logger.warning(f"Failed to create test batch: {e}")
            self.test_batch = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if self.test_batch is None:
            return

        if (epoch + 1) % self.config.monitor_every_n_epochs != 0:
            return

        try:
            builder = BFUNetDatasetBuilder(self.config)
            noisy, clean = builder._add_noise(self.test_batch)

            model_output = self.model(noisy, training=False)

            if isinstance(model_output, list):
                denoised = model_output[0]
            else:
                denoised = model_output

            self._save_comparison(epoch + 1, noisy, clean, denoised)

            mse = tf.reduce_mean(tf.square(denoised - clean))
            psnr = tf.reduce_mean(tf.image.psnr(denoised, clean, max_val=1.0))
            logger.info(f"Epoch {epoch + 1} - MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")

        except Exception as e:
            logger.warning(f"Denoising visualization failed: {e}")

    def _save_comparison(self, epoch: int, noisy: tf.Tensor, clean: tf.Tensor, denoised: tf.Tensor):
        """Save comparison visualization."""
        try:
            num_samples = min(10, noisy.shape[0])

            fig, axes = plt.subplots(3, num_samples, figsize=(25, 7.5))
            fig.suptitle(f'Denoising Results - Epoch {epoch}', fontsize=20)

            for i in range(num_samples):
                clean_img = np.clip(clean[i].numpy(), 0, 1)
                noisy_img = np.clip(noisy[i].numpy(), 0, 1)
                denoised_img = np.clip(denoised[i].numpy(), 0, 1)

                if clean_img.shape[-1] == 1:
                    clean_img = clean_img.squeeze(-1)
                    noisy_img = noisy_img.squeeze(-1)
                    denoised_img = denoised_img.squeeze(-1)
                    cmap = 'gray'
                else:
                    cmap = None

                axes[0, i].imshow(clean_img, cmap=cmap, vmin=0, vmax=1)
                axes[0, i].axis('off')
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
# BFU-NET TRAINING PIPELINE
# =============================================================================

class BFUNetTrainingPipeline(TrainingPipeline):
    """Extended pipeline for BFU-Net with deep supervision."""

    def __init__(self, config: BFUNetConfig):
        super().__init__(config)
        self.config: BFUNetConfig = config
        self.num_outputs = 1
        self.model_output_dims = None

    def _compile_model(self, model: keras.Model, total_steps: int):
        """Compile with deep supervision support."""
        logger.info("Compiling BFU-Net model")

        # Analyze model structure
        has_multiple_outputs = isinstance(model.output, list)
        self.num_outputs = len(model.output) if has_multiple_outputs else 1

        if has_multiple_outputs:
            self.model_output_dims = [
                (out.shape[1], out.shape[2]) for out in model.output
            ]
            logger.info(f"Multi-output model: {self.num_outputs} outputs")
            logger.info(f"Output dimensions: {self.model_output_dims}")

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
            custom_callbacks: Optional[List[keras.callbacks.Callback]] = None,
            train_ds: Optional[tf.data.Dataset] = None
    ) -> List[keras.callbacks.Callback]:
        """Create callbacks with deep supervision support."""
        callbacks = super()._create_callbacks(
            lr_schedule, custom_callbacks, train_ds)

        # Add deep supervision scheduler
        if self.config.enable_deep_supervision and self.num_outputs > 1:
            callbacks.append(
                DeepSupervisionWeightScheduler(self.config, self.num_outputs)
            )

        # Add synthesis monitor
        if self.config.enable_synthesis:
            callbacks.append(
                SynthesisMonitorCallback(self.config, self.experiment_dir)
            )

        # Add denoising visualization
        if self.config.save_training_images:
            try:
                builder = BFUNetDatasetBuilder(self.config)
                val_files = builder._create_file_list(
                    self.config.val_image_dirs,
                    limit=100
                )
                callbacks.append(
                    DenoisingVisualizationCallback(self.config, val_files, self.experiment_dir)
                )
            except Exception as e:
                logger.warning(f"Could not add denoising visualization: {e}")

        return callbacks

    def _create_lr_schedule(self, total_steps: int) -> Any:
        """Creates the learning rate schedule from the configuration."""
        decay_steps = self.config.decay_steps

        if decay_steps is None:
            decay_steps = total_steps - self.config.warmup_steps


        if decay_steps <= 0:
            logger.warning(
                f"Calculated decay_steps is non-positive ({decay_steps}) because warmup_steps "
                f"({self.config.warmup_steps}) >= total_steps ({total_steps}). Setting decay_steps to 1."
            )
            decay_steps = 1

        # Collect all possible schedule parameters from the config
        schedule_config = {
            'type': self.config.lr_schedule_type,
            'learning_rate': self.config.learning_rate,
            'warmup_steps': self.config.warmup_steps,
            'warmup_start_lr': self.config.warmup_start_lr,
            'decay_steps': decay_steps,
            'alpha': self.config.alpha,
            'decay_rate': getattr(self.config, 'decay_rate', None),
            't_mul': getattr(self.config, 't_mul', None),
            'm_mul': getattr(self.config, 'm_mul', None),
        }

        return learning_rate_schedule_builder(schedule_config)


    def _create_optimizer(self, lr_schedule: Any) -> keras.optimizers.Optimizer:
        """Creates the optimizer from the configuration."""

        # Collect all possible optimizer parameters from the config
        optimizer_config = {
            'type': self.config.optimizer_type,
            'weight_decay': self.config.weight_decay,
            'beta_1': self.config.beta_1,
            'beta_2': self.config.beta_2,
            'epsilon': getattr(self.config, 'epsilon', 1e-7),
            'gradient_clipping_by_norm': self.config.gradient_clipping_norm_global,
            'gradient_clipping_by_local_norm': getattr(self.config, 'gradient_clipping_norm_local', None),
            'gradient_clipping_by_value': getattr(self.config, 'gradient_clipping_value', None),
        }

        return optimizer_builder(optimizer_config, lr_schedule=lr_schedule)

    def run(
            self,
            model_builder: ModelBuilder,
            dataset_builder: DatasetBuilder,
            custom_callbacks: Optional[List[keras.callbacks.Callback]] = None
    ) -> Tuple[keras.Model, keras.callbacks.History]:
        """Run training with inference model creation."""
        # Build model first to get output dimensions
        temp_model = model_builder(self.config)

        if isinstance(temp_model.output, list):
            self.model_output_dims = [
                (out.shape[1], out.shape[2]) for out in temp_model.output
            ]
            # Update dataset builder with output dimensions
            if isinstance(dataset_builder, BFUNetDatasetBuilder):
                dataset_builder.model_output_dims = self.model_output_dims

        del temp_model

        # Run training
        model, history = super().run(
            model_builder,
            dataset_builder,
            custom_callbacks
        )

        # Save inference model
        self._save_inference_model(model, model_builder)

        return model, history

    def _save_inference_model(self, trained_model: keras.Model, model_builder: ModelBuilder):
        """Save clean inference model."""
        try:
            logger.info("Creating inference model")

            inference_config = BFUNetConfig(**self.config.__dict__)
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

def build_bfunet_denoiser(config: BFUNetConfig) -> keras.Model:
    """Build BFU-Net denoiser model."""
    variant = config.model_args.get('variant', 'tiny')

    logger.info(f"Building BFU-Net: {variant}")
    logger.info(f"Deep supervision: {config.enable_deep_supervision}")

    if variant in BFUNET_CONFIGS:
        model = create_bfunet_variant(
            variant=variant,
            input_shape=config.input_shape,
            enable_deep_supervision=config.enable_deep_supervision
        )
    elif variant == 'custom':
        model = create_bfunet_denoiser(
            input_shape=config.input_shape,
            depth=config.depth,
            initial_filters=config.filters,
            blocks_per_level=config.blocks_per_level,
            kernel_size=config.kernel_size,
            activation=config.activation,
            enable_deep_supervision=config.enable_deep_supervision,
            model_name=f"bfunet_custom_{config.experiment_name}"
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    logger.info(f"Model created with {model.count_params():,} parameters")
    return model


# =============================================================================
# CLI
# =============================================================================

def create_bfunet_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='Train BFU-Net with deep supervision',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model-variant',
        choices=['tiny', 'small', 'base', 'large', 'xlarge', 'custom'],
        default='tiny',
        help='Model architecture variant'
    )

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--channels', type=int, default=1, choices=[1, 3])

    parser.add_argument('--enable-deep-supervision', action='store_true', default=True)
    parser.add_argument('--no-deep-supervision', dest='enable_deep_supervision', action='store_false')
    parser.add_argument('--deep-supervision-schedule', type=str, default='step_wise')

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
    parser = create_bfunet_argument_parser()
    args = parser.parse_args()

    config = BFUNetConfig(
        # Data paths - MODIFY FOR YOUR SETUP
        train_image_dirs=[
            #'/media/arxwn/data0_4tb/datasets/Megadepth',
            '/media/arxwn/data0_4tb/datasets/div2k/train',
            #'/media/arxwn/data0_4tb/datasets/WFLW/images',
            #'/media/arxwn/data0_4tb/datasets/bdd_data/train',
            #'/media/arxwn/data0_4tb/datasets/COCO/train2017'
        ],
        val_image_dirs=[
            '/media/arxwn/data0_4tb/datasets/div2k/validation',
            #'/media/arxwn/data0_4tb/datasets/COCO/val2017',
        ],

        # Model
        patch_size=args.patch_size,
        channels=args.channels,
        model_args={'variant': args.model_variant},

        # Training
        epochs=args.epochs,
        batch_size=args.batch_size,
        patches_per_image=16,

        # Deep supervision
        enable_deep_supervision=args.enable_deep_supervision,
        deep_supervision_schedule_type=args.deep_supervision_schedule,

        # Synthesis
        enable_synthesis=args.enable_synthesis,
        synthesis_samples=10,
        synthesis_steps=200,

        # Optimization
        learning_rate=1e-3,
        optimizer_type='adamw',
        lr_schedule_type='cosine_decay',
        weight_decay=1e-5,

        # Output
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        enable_visualization=True,
        enable_analysis=False,

        enable_gradient_tracking=True
    )

    logger.info("=== BFU-Net Training Configuration ===")
    logger.info(f"Model: {config.model_args['variant']}")
    logger.info(f"Deep Supervision: {config.enable_deep_supervision}")
    if config.enable_deep_supervision:
        logger.info(f"  Schedule: {config.deep_supervision_schedule_type}")
    logger.info(f"Synthesis: {config.enable_synthesis}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Patch size: {config.patch_size}x{config.patch_size}")

    try:
        dataset_builder = BFUNetDatasetBuilder(config)
        pipeline = BFUNetTrainingPipeline(config)

        logger.info("Starting training...")
        model, history = pipeline.run(
            model_builder=build_bfunet_denoiser,
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