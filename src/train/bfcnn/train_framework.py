import gc
import keras
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any

from dl_techniques.utils.train_vision.framework import (
    TrainingConfig,
    DatasetBuilder,
    TrainingPipeline,
    ModelBuilder,
)
from dl_techniques.utils.logger import logger
from dl_techniques.models.bias_free_denoisers.bfcnn import (
    create_bfcnn_denoiser,
    BFCNN_CONFIGS,
    create_bfcnn_variant,
)
from dl_techniques.analyzer import DataInput


# =============================================================================
# DENOISING-SPECIFIC CONFIGURATION
# =============================================================================

@dataclass
class DenoisingConfig(TrainingConfig):
    """
    Extended configuration for denoising tasks.

    This extends the base TrainingConfig with denoising-specific parameters
    while maintaining compatibility with the framework.

    Attributes:
        train_image_dirs: Directories containing training images.
        val_image_dirs: Directories containing validation images.
        patch_size: Size of training patches (overrides input_shape).
        channels: Number of input channels (1=grayscale, 3=RGB).
        image_extensions: Valid image file extensions.
        max_train_files: Limit on training files (None = no limit).
        max_val_files: Limit on validation files.
        parallel_reads: Parallel file reading threads.
        dataset_shuffle_buffer: Shuffle buffer size.
        noise_sigma_min: Minimum noise standard deviation.
        noise_sigma_max: Maximum noise standard deviation.
        noise_distribution: 'uniform' or 'log_uniform' sampling.
        patches_per_image: Number of patches to extract per image.
        augment_data: Apply data augmentation.
        normalize_input: Normalize input to [-1, 1].
        monitor_every_n_epochs: Save intermediate results every N epochs.
        save_training_images: Save sample denoised images during training.
    """
    # Data paths
    train_image_dirs: List[str] = field(default_factory=list)
    val_image_dirs: List[str] = field(default_factory=list)

    # Override input_shape with patch-based parameters
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

    # Monitoring
    monitor_every_n_epochs: int = 5
    save_training_images: bool = True

    def __post_init__(self):
        """Initialize and validate configuration."""
        # Set input_shape based on patch_size and channels
        self.input_shape = (self.patch_size, self.patch_size, self.channels)
        self.num_classes = self.channels  # Output has same channels as input

        # Call parent post_init
        super().__post_init__()

        # Additional validation
        if self.noise_sigma_min < 0 or self.noise_sigma_max <= self.noise_sigma_min:
            raise ValueError("Invalid noise sigma range")

        if not self.train_image_dirs:
            raise ValueError("No training directories specified")

        if not self.val_image_dirs:
            raise ValueError("No validation directories specified")

        if self.noise_distribution not in ['uniform', 'log_uniform']:
            raise ValueError(f"Invalid noise distribution: {self.noise_distribution}")


# =============================================================================
# DENOISING DATASET BUILDER
# =============================================================================

class DenoisingDatasetBuilder(DatasetBuilder):
    """
    Dataset builder for image denoising tasks.

    This builder loads images, extracts patches, applies augmentation,
    and adds synthetic noise to create training pairs.

    Attributes:
        config: DenoisingConfig instance with all parameters.
    """

    def __init__(self, config: DenoisingConfig):
        """
        Initialize the denoising dataset builder.

        Args:
            config: DenoisingConfig object.
        """
        super().__init__(config)
        self.config: DenoisingConfig = config

    def _load_and_preprocess_image(
            self,
            image_path: tf.Tensor
    ) -> tf.Tensor:
        """
        Load and preprocess a single image.

        Args:
            image_path: Tensor containing path to image file.

        Returns:
            Preprocessed image patch.
        """
        try:
            # Read and decode image
            image_string = tf.io.read_file(image_path)
            image = tf.image.decode_image(
                image_string,
                channels=self.config.channels,
                expand_animations=False
            )
            image.set_shape([None, None, self.config.channels])

            # Convert to float32
            image = tf.cast(image, tf.float32)

            # Normalize to [-1, 1] for bias-free networks
            if self.config.normalize_input:
                image = (image / 127.5) - 1.0
            else:
                image = image / 255.0

            # Handle small images by resizing
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

            # Extract random patch
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
        """
        Apply data augmentation.

        Args:
            patch: Input patch tensor.

        Returns:
            Augmented patch.
        """
        # Random flips
        patch = tf.image.random_flip_left_right(patch)
        patch = tf.image.random_flip_up_down(patch)

        # Random rotation (90 degree increments)
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        patch = tf.image.rot90(patch, k)

        return patch

    def _add_noise(
            self,
            patch: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Add Gaussian noise to create training pair.

        Args:
            patch: Clean patch tensor.

        Returns:
            Tuple of (noisy_patch, clean_patch).
        """
        # Sample noise level
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

        # Generate and add noise
        noise = tf.random.normal(tf.shape(patch)) * noise_level
        noisy_patch = patch + noise

        # Clip to valid range
        if self.config.normalize_input:
            noisy_patch = tf.clip_by_value(noisy_patch, -1.0, 1.0)
        else:
            noisy_patch = tf.clip_by_value(noisy_patch, 0.0, 1.0)

        return noisy_patch, patch

    def _create_file_list(
            self,
            directories: List[str],
            limit: Optional[int] = None
    ) -> List[str]:
        """
        Create unified file list from directories.

        Args:
            directories: List of directories to scan.
            limit: Optional limit on number of files.

        Returns:
            List of file paths.
        """
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

        # Apply limit
        if limit and limit < len(all_files):
            logger.info(f"Limiting to {limit} files")
            np.random.shuffle(all_files)
            all_files = all_files[:limit]

        return all_files

    def build(self) -> Tuple[
        tf.data.Dataset,
        tf.data.Dataset,
        Optional[int],
        Optional[int]
    ]:
        """
        Build training and validation datasets.

        Returns:
            Tuple of (train_dataset, val_dataset, steps_per_epoch, val_steps).
        """
        logger.info("Building denoising datasets")

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

        # Duplicate for multiple patches per image
        if self.config.patches_per_image > 1:
            train_ds = train_ds.flat_map(
                lambda path: tf.data.Dataset.from_tensors(path).repeat(
                    self.config.patches_per_image
                )
            )

        # Load and preprocess
        train_ds = train_ds.map(
            self._load_and_preprocess_image,
            num_parallel_calls=self.config.parallel_reads
        )

        # Filter failed loads
        train_ds = train_ds.filter(lambda x: tf.reduce_sum(tf.abs(x)) > 0)

        # Augmentation
        if self.config.augment_data:
            train_ds = train_ds.map(
                self._augment_patch,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Add noise
        train_ds = train_ds.map(
            self._add_noise,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Batch and prefetch
        train_ds = train_ds.batch(self.config.batch_size, drop_remainder=True)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        # Create validation dataset (similar but no augmentation)
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
        """
        Get test data for analysis.

        Returns:
            DataInput with test patches, or None.
        """
        try:
            val_files = self._create_file_list(
                self.config.val_image_dirs,
                limit=100
            )

            clean_patches = []
            noisy_patches = []

            for file_path in val_files[:50]:  # Use subset
                path_tensor = tf.constant(file_path)
                patch = self._load_and_preprocess_image(path_tensor)
                noisy, clean = self._add_noise(patch)

                clean_patches.append(clean.numpy())
                noisy_patches.append(noisy.numpy())

            # For denoising, we use noisy as input and clean as target
            return DataInput(
                x_data=np.array(noisy_patches),
                y_data=np.array(clean_patches)
            )

        except Exception as e:
            logger.warning(f"Could not create test data: {e}")
            return None


# =============================================================================
# DENOISING METRICS
# =============================================================================

@keras.saving.register_keras_serializable()
def psnr_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    PSNR metric for denoising ([-1, 1] range).

    Args:
        y_true: Ground truth images.
        y_pred: Predicted/denoised images.

    Returns:
        Mean PSNR value across batch.
    """
    return tf.reduce_mean(tf.image.psnr(y_pred, y_true, max_val=2.0))


# =============================================================================
# DENOISING VISUALIZATION CALLBACK
# =============================================================================

class DenoisingVisualizationCallback(keras.callbacks.Callback):
    """
    Callback for visualizing denoising results during training.

    Attributes:
        config: DenoisingConfig instance.
        output_dir: Directory for saving visualizations.
        test_batch: Fixed test batch for consistent monitoring.
    """

    def __init__(self, config: DenoisingConfig, val_files: List[str]):
        """
        Initialize the callback.

        Args:
            config: DenoisingConfig instance.
            val_files: List of validation file paths.
        """
        super().__init__()
        self.config = config
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create test batch
        self._create_test_batch(val_files)

    def _create_test_batch(self, val_files: List[str]):
        """Create a fixed test batch for monitoring."""
        try:
            # Use DenoisingDatasetBuilder helper
            builder = DenoisingDatasetBuilder(self.config)

            clean_patches = []
            for file_path in val_files[:10]:
                path_tensor = tf.constant(file_path)
                patch = builder._load_and_preprocess_image(path_tensor)
                clean_patches.append(patch)

            self.test_batch = tf.stack(clean_patches)
            logger.info(f"Created test batch: {self.test_batch.shape}")

        except Exception as e:
            logger.warning(f"Failed to create test batch: {e}")
            self.test_batch = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Save denoising visualizations."""
        if self.test_batch is None:
            return

        if (epoch + 1) % self.config.monitor_every_n_epochs != 0:
            return

        try:
            # Create noisy versions
            builder = DenoisingDatasetBuilder(self.config)
            noisy, clean = builder._add_noise(self.test_batch)

            # Denoise
            denoised = self.model(noisy, training=False)

            # Save visualization
            self._save_comparison(epoch + 1, noisy, clean, denoised)

            # Compute and log metrics
            mse = tf.reduce_mean(tf.square(denoised - clean))
            psnr = tf.reduce_mean(tf.image.psnr(denoised, clean, max_val=2.0))

            logger.info(
                f"Epoch {epoch + 1} - Test MSE: {mse.numpy():.6f}, "
                f"PSNR: {psnr.numpy():.2f} dB"
            )

        except Exception as e:
            logger.warning(f"Visualization failed: {e}")

    def _save_comparison(
            self,
            epoch: int,
            noisy: tf.Tensor,
            clean: tf.Tensor,
            denoised: tf.Tensor
    ):
        """Save comparison visualization."""
        num_samples = min(10, noisy.shape[0])

        fig, axes = plt.subplots(3, num_samples, figsize=(25, 7.5))
        fig.suptitle(f'Denoising Results - Epoch {epoch}', fontsize=20)

        for i in range(num_samples):
            # Denormalize from [-1, 1] to [0, 1]
            clean_img = (clean[i].numpy() + 1.0) / 2.0
            noisy_img = (noisy[i].numpy() + 1.0) / 2.0
            denoised_img = (denoised[i].numpy() + 1.0) / 2.0

            # Clip and prepare
            clean_img = np.clip(clean_img, 0, 1)
            noisy_img = np.clip(noisy_img, 0, 1)
            denoised_img = np.clip(denoised_img, 0, 1)

            if clean_img.shape[-1] == 1:
                clean_img = clean_img.squeeze(-1)
                noisy_img = noisy_img.squeeze(-1)
                denoised_img = denoised_img.squeeze(-1)
                cmap = 'gray'
            else:
                cmap = None

            # Plot
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


# =============================================================================
# DENOISING TRAINING PIPELINE
# =============================================================================

class DenoisingTrainingPipeline(TrainingPipeline):
    """
    Extended pipeline for denoising tasks.

    This overrides key methods to use denoising-specific loss,
    metrics, and callbacks.
    """

    def __init__(self, config: DenoisingConfig):
        """
        Initialize denoising pipeline.

        Args:
            config: DenoisingConfig instance.
        """
        super().__init__(config)
        self.config: DenoisingConfig = config

    def _compile_model(self, model: keras.Model, total_steps: int) -> None:
        """
        Compile model with denoising-specific loss and metrics.

        Args:
            model: Keras model to compile.
            total_steps: Total training steps.
        """
        logger.info("Compiling denoising model")

        # Create optimizer
        lr_schedule = self._create_lr_schedule(total_steps)
        optimizer = self._create_optimizer(lr_schedule)

        # Denoising uses MSE loss and specific metrics
        loss = keras.losses.MeanSquaredError()
        metrics = [
            keras.metrics.MeanAbsoluteError(name='mae'),
            keras.metrics.RootMeanSquaredError(name='rmse'),
            psnr_metric
        ]

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        logger.info("Denoising model compiled")

    def _create_callbacks(
            self,
            lr_schedule: Optional[Any] = None,
            custom_callbacks: Optional[List[keras.callbacks.Callback]] = None
    ) -> List[keras.callbacks.Callback]:
        """
        Create callbacks including denoising visualization.

        Args:
            lr_schedule: Learning rate schedule.
            custom_callbacks: Additional callbacks.

        Returns:
            List of callbacks.
        """
        # Get base callbacks
        callbacks = super()._create_callbacks(lr_schedule, custom_callbacks)

        # Add denoising visualization if enabled
        if self.config.save_training_images:
            try:
                # Get validation files for test batch
                builder = DenoisingDatasetBuilder(self.config)
                val_files = builder._create_file_list(
                    self.config.val_image_dirs,
                    limit=100
                )

                callbacks.append(
                    DenoisingVisualizationCallback(self.config, val_files)
                )
                logger.info("Added denoising visualization callback")

            except Exception as e:
                logger.warning(f"Could not add visualization callback: {e}")

        return callbacks

    def run(
            self,
            model_builder: ModelBuilder,
            dataset_builder: DatasetBuilder,
            custom_callbacks: Optional[List[keras.callbacks.Callback]] = None
    ) -> Tuple[keras.Model, keras.callbacks.History]:
        """
        Run training and save inference model.

        Args:
            model_builder: Function to create model.
            dataset_builder: Dataset builder instance.
            custom_callbacks: Optional additional callbacks.

        Returns:
            Tuple of (trained_model, history).
        """
        # Run standard training
        model, history = super().run(
            model_builder,
            dataset_builder,
            custom_callbacks
        )

        # Save inference model with flexible input shape
        self._save_inference_model(model, model_builder)

        return model, history

    def _save_inference_model(
            self,
            trained_model: keras.Model,
            model_builder: ModelBuilder
    ):
        """
        Save a clean inference model with flexible input shape.

        Args:
            trained_model: The trained model.
            model_builder: Function to create model.
        """
        try:
            logger.info("Creating inference model")

            # Create new model with flexible spatial dimensions
            inference_config = DenoisingConfig(**self.config.__dict__)
            inference_config.input_shape = (None, None, self.config.channels)

            inference_model = model_builder(inference_config)

            # Copy weights
            inference_model.set_weights(trained_model.get_weights())

            # Save
            inference_path = self.experiment_dir / "inference_model.keras"
            inference_model.save(inference_path)

            logger.info(f"Inference model saved: {inference_path}")
            logger.info(f"Accepts flexible input: (None, None, {self.config.channels})")

            del inference_model
            gc.collect()

        except Exception as e:
            logger.error(f"Failed to save inference model: {e}")


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_bfcnn_denoiser(config: DenoisingConfig) -> keras.Model:
    """
    Build BFCNN denoiser model.

    Args:
        config: DenoisingConfig with model parameters.

    Returns:
        Keras model for denoising.
    """
    logger.info(f"Building BFCNN denoiser: {config.model_args.get('variant', 'default')}")

    variant = config.model_args.get('variant', 'tiny')

    if variant in BFCNN_CONFIGS:
        model = create_bfcnn_variant(
            variant=variant,
            input_shape=config.input_shape
        )
    elif variant == 'custom':
        model = create_bfcnn_denoiser(
            input_shape=config.input_shape,
            num_blocks=config.model_args.get('num_blocks', 8),
            filters=config.model_args.get('filters', 64),
            initial_kernel_size=config.model_args.get('initial_kernel_size', 5),
            kernel_size=config.model_args.get('kernel_size', 3),
            activation=config.model_args.get('activation', 'relu'),
            model_name=f"bfcnn_custom_{config.experiment_name}"
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    logger.info(f"Model created with {model.count_params():,} parameters")
    return model


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def create_denoising_argument_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for denoising training.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description='Train BFCNN denoiser using vision framework',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument(
        '--model-variant',
        type=str,
        default='tiny',
        choices=['tiny', 'small', 'base', 'large', 'xlarge', 'custom'],
        help='Model variant'
    )

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--patch-size', type=int, default=128, help='Patch size')
    parser.add_argument(
        '--channels',
        type=int,
        default=1,
        choices=[1, 3],
        help='Channels (1=gray, 3=RGB)'
    )

    # Data arguments
    parser.add_argument('--patches-per-image', type=int, default=16)
    parser.add_argument('--max-train-files', type=int, default=None)
    parser.add_argument('--max-val-files', type=int, default=None)

    # Noise arguments
    parser.add_argument('--noise-min', type=float, default=0.0)
    parser.add_argument('--noise-max', type=float, default=0.5)

    # Optimization arguments
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--lr-schedule', type=str, default='cosine')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--experiment-name', type=str, default=None)

    return parser


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    """Main training function."""
    parser = create_denoising_argument_parser()
    args = parser.parse_args()

    # Create configuration
    config = DenoisingConfig(
        # Data paths - MODIFY THESE FOR YOUR SETUP
        train_image_dirs=[
            '/media/arxwn/data0_4tb/datasets/Megadepth',
            #'/media/arxwn/data0_4tb/datasets/div2k/train',
            #'/media/arxwn/data0_4tb/datasets/WFLW/images',
            #'/media/arxwn/data0_4tb/datasets/bdd_data/train',
            #'/media/arxwn/data0_4tb/datasets/COCO/train2017',
            #'/media/arxwn/data0_4tb/datasets/VGG-Face2/data/train',
        ],
        val_image_dirs=[
            '/media/arxwn/data0_4tb/datasets/div2k/validation',
            #'/media/arxwn/data0_4tb/datasets/COCO/val2017',
        ],

        # Model configuration
        patch_size=args.patch_size,
        channels=args.channels,
        model_args={'variant': args.model_variant},

        # Training configuration
        epochs=args.epochs,
        batch_size=args.batch_size,
        patches_per_image=args.patches_per_image,

        # File limits
        max_train_files=args.max_train_files,
        max_val_files=args.max_val_files,
        parallel_reads=8,

        # Noise configuration
        noise_sigma_min=args.noise_min,
        noise_sigma_max=args.noise_max,
        noise_distribution='uniform',

        # Optimization
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer,
        lr_schedule_type=args.lr_schedule,
        weight_decay=1e-5,
        gradient_clipping=1.0,

        # Monitoring
        monitor_every_n_epochs=5,
        save_training_images=True,
        validation_steps=200,
        early_stopping_patience=15,

        # Output
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        enable_visualization=True,
        enable_analysis=False,  # Disable for denoising (not classification)
    )

    # Log configuration
    logger.info("=== Denoising Training Configuration ===")
    logger.info(f"Model: BFCNN-{config.model_args['variant']}")
    logger.info(f"Patch size: {config.patch_size}x{config.patch_size}")
    logger.info(f"Channels: {config.channels}")
    logger.info(f"Noise range: [{config.noise_sigma_min}, {config.noise_sigma_max}]")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Output: {config.output_dir}/bfcnn_{config.experiment_name}")

    try:
        # Create components
        dataset_builder = DenoisingDatasetBuilder(config)
        pipeline = DenoisingTrainingPipeline(config)

        # Train
        logger.info("Starting training...")
        model, history = pipeline.run(
            model_builder=build_bfcnn_denoiser,
            dataset_builder=dataset_builder
        )

        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {pipeline.experiment_dir}")

        # Final summary
        model.summary(print_fn=logger.info)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()