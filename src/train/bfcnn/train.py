"""
Bias-Free CNN Denoiser Training Script - Memory-Efficient Version

Handles massive datasets that exceed available RAM through:
- Lazy file discovery and streaming data pipeline
- Memory-efficient patch extraction
- Progressive dataset sampling
- Configurable dataset limits per epoch

Based on "Robust and Interpretable Blind Image Denoising via Bias-Free
Convolutional Neural Networks" (Mohan et al., ICLR 2020).
"""

import os
import gc
import json
import time
import keras
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any, Union

from dl_techniques.utils.logger import logger
from dl_techniques.models.bfcnn_denoiser import create_bfcnn_denoiser, create_bfcnn_standard
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder

# =====================================================================
# CONFIGURATION
# =====================================================================

@dataclass
class TrainingConfig:
    """Memory-efficient configuration for bias-free CNN denoiser training."""

    # === Data Configuration ===
    train_image_dirs: List[str]  # Directories containing training images
    val_image_dirs: List[str]    # Directories containing validation images
    patch_size: int = 64         # Size of training patches (patch_size x patch_size)
    channels: int = 1            # Number of input channels (1=grayscale, 3=RGB)
    image_extensions: List[str] = None  # Supported image formats

    # === Memory Management ===
    max_images_per_epoch: Optional[int] = None    # Limit images per epoch (None = no limit)
    prefetch_buffer_size: int = 100               # Number of files to prefetch
    parallel_reads: int = 4                       # Parallel file reading threads
    dataset_shuffle_buffer: int = 1000            # Shuffle buffer size
    estimate_dataset_size: bool = True            # Estimate total dataset size
    max_estimation_samples: int = 1000            # Max files to sample for size estimation

    # === Noise Configuration ===
    noise_sigma_min: float = 0.0    # Minimum noise standard deviation
    noise_sigma_max: float = 0.4    # Maximum noise standard deviation (universal range)
    noise_distribution: str = 'uniform'  # 'uniform' or 'log_uniform' sampling
    blind_training: bool = True     # Train without providing noise level to model

    # === Model Configuration ===
    model_type: str = 'standard'   # 'light', 'standard', 'deep', or 'custom'
    num_blocks: int = 8            # Number of residual blocks (for custom model)
    filters: int = 64              # Number of filters (for custom model)
    initial_kernel_size: int = 5   # Initial convolution kernel size
    kernel_size: int = 3           # Residual block kernel size
    activation: str = 'relu'       # Activation function

    # === Training Configuration ===
    batch_size: int = 32
    epochs: int = 100
    patches_per_image: int = 16    # Number of patches to extract per image
    augment_data: bool = True      # Apply data augmentation
    normalize_input: bool = True   # Normalize input to [0, 1]
    steps_per_epoch: Optional[int] = None  # Manual override for steps per epoch

    # === Optimization Configuration ===
    learning_rate: float = 1e-3
    optimizer_type: str = 'adam'
    lr_schedule_type: str = 'cosine_decay'
    warmup_epochs: int = 5
    weight_decay: float = 1e-5
    gradient_clipping: float = 1.0

    # === Monitoring Configuration ===
    monitor_every_n_epochs: int = 5    # Save intermediate results every N epochs
    save_best_only: bool = True        # Only save model if validation loss improves
    early_stopping_patience: int = 15  # Early stopping patience
    reduce_lr_patience: int = 8        # Learning rate reduction patience
    validation_steps: Optional[int] = 100  # Number of validation steps

    # === Output Configuration ===
    output_dir: str = 'bfcnn_experiments'
    experiment_name: str = None        # Auto-generated if None
    save_training_images: bool = True  # Save sample denoised images during training
    save_model_checkpoints: bool = True

    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.image_extensions is None:
            self.image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']

        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"bfcnn_{self.model_type}_{timestamp}"

        # Validation
        if self.noise_sigma_min < 0 or self.noise_sigma_max <= self.noise_sigma_min:
            raise ValueError("Invalid noise sigma range")

        if self.patch_size <= 0 or self.channels <= 0:
            raise ValueError("Invalid patch size or channel configuration")

# Default configuration for large datasets
DEFAULT_CONFIG = TrainingConfig(
    train_image_dirs=['data/train'],
    val_image_dirs=['data/val'],
    patch_size=64,
    channels=1,
    noise_sigma_max=0.4,
    batch_size=32,
    epochs=100,
    learning_rate=1e-3,
    monitor_every_n_epochs=5,
    max_images_per_epoch=10000,  # Limit for memory efficiency
    prefetch_buffer_size=100,
    parallel_reads=4,
    dataset_shuffle_buffer=1000
)

# =====================================================================
# MEMORY-EFFICIENT DATA LOADING
# =====================================================================

class MemoryEfficientDatasetBuilder:
    """Builds memory-efficient datasets for massive image collections."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._train_file_patterns = self._build_file_patterns(config.train_image_dirs)
        self._val_file_patterns = self._build_file_patterns(config.val_image_dirs)
        self._estimated_sizes = None

    def _build_file_patterns(self, directories: List[str]) -> List[str]:
        """Build file patterns for tf.data.Dataset.list_files."""
        patterns = []
        for directory in directories:
            for ext in self.config.image_extensions:
                # Add patterns for both lowercase and uppercase extensions
                patterns.append(os.path.join(directory, f"**/*{ext}"))
                patterns.append(os.path.join(directory, f"**/*{ext.upper()}"))
        return patterns

    def estimate_dataset_sizes(self) -> Tuple[int, int]:
        """Estimate dataset sizes without loading all files into memory."""
        if not self.config.estimate_dataset_size:
            logger.info("Dataset size estimation disabled")
            return None, None

        logger.info("Estimating dataset sizes...")

        def estimate_size(patterns: List[str], dataset_type: str) -> int:
            """Estimate size for a set of file patterns."""
            if not patterns:
                return 0

            # Use tf.data to lazily discover files
            try:
                dataset = tf.data.Dataset.list_files(patterns, shuffle=False)

                # Sample a subset for estimation if dataset is too large
                sample_size = min(self.config.max_estimation_samples,
                                tf.data.experimental.cardinality(dataset).numpy())

                if sample_size == tf.data.experimental.UNKNOWN_CARDINALITY:
                    # For unknown cardinality, sample and extrapolate
                    sample_dataset = dataset.take(self.config.max_estimation_samples)
                    sampled_count = sum(1 for _ in sample_dataset)

                    # Very rough estimation - assume uniform distribution
                    if sampled_count == self.config.max_estimation_samples:
                        estimated_total = sampled_count * 10  # Conservative multiplier
                        logger.warning(f"Estimated {dataset_type} size (very rough): ~{estimated_total:,} files")
                    else:
                        estimated_total = sampled_count
                        logger.info(f"Estimated {dataset_type} size: {estimated_total:,} files")
                else:
                    estimated_total = int(sample_size)
                    logger.info(f"Estimated {dataset_type} size: {estimated_total:,} files")

                return estimated_total

            except Exception as e:
                logger.warning(f"Could not estimate {dataset_type} dataset size: {e}")
                return 1000  # Fallback estimate

        train_size = estimate_size(self._train_file_patterns, "training")
        val_size = estimate_size(self._val_file_patterns, "validation")

        self._estimated_sizes = (train_size, val_size)
        return train_size, val_size

    def create_training_dataset(self) -> tf.data.Dataset:
        """Create memory-efficient training dataset."""
        return self._create_dataset(
            self._train_file_patterns,
            is_training=True,
            max_images=self.config.max_images_per_epoch
        )

    def create_validation_dataset(self) -> tf.data.Dataset:
        """Create memory-efficient validation dataset."""
        val_limit = self.config.validation_steps * self.config.batch_size if self.config.validation_steps else None
        return self._create_dataset(
            self._val_file_patterns,
            is_training=False,
            max_images=val_limit
        )

    def _create_dataset(self, file_patterns: List[str], is_training: bool = True,
                       max_images: Optional[int] = None) -> tf.data.Dataset:
        """Create a memory-efficient dataset from file patterns."""

        if not file_patterns:
            raise ValueError("No file patterns provided")

        # Use list_files for lazy file discovery
        dataset = tf.data.Dataset.list_files(
            file_patterns,
            shuffle=is_training,
            seed=42 if not is_training else None  # Fixed seed for validation
        )

        # Limit number of images if specified
        if max_images is not None:
            dataset = dataset.take(max_images)

        # Shuffle files (for training)
        if is_training:
            dataset = dataset.shuffle(
                buffer_size=self.config.dataset_shuffle_buffer,
                reshuffle_each_iteration=True
            )

        # Repeat dataset for training
        if is_training:
            dataset = dataset.repeat()

        # Process files in parallel
        dataset = dataset.map(
            self._load_and_process_image,
            num_parallel_calls=self.config.parallel_reads
        )

        # Flatten patches (each image produces multiple patches)
        dataset = dataset.unbatch()

        # Additional data augmentation for training
        if is_training and self.config.augment_data:
            dataset = dataset.map(
                self._augment_patch,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Batch and prefetch
        dataset = dataset.batch(self.config.batch_size, drop_remainder=is_training)
        dataset = dataset.prefetch(self.config.prefetch_buffer_size)

        return dataset

    def _load_and_process_image(self, image_path: tf.Tensor) -> tf.Tensor:
        """Load and process a single image file efficiently."""

        @tf.function
        def process_image():
            # Read image file
            image_string = tf.io.read_file(image_path)

            # Decode image with error handling
            try:
                image = tf.image.decode_image(
                    image_string,
                    channels=self.config.channels,
                    expand_animations=False  # Don't expand animated GIFs
                )
                image.set_shape([None, None, self.config.channels])
            except:
                # Return dummy patch if image can't be decoded
                return tf.zeros([1, self.config.patch_size, self.config.patch_size, self.config.channels])

            # Convert to float32 and normalize
            image = tf.cast(image, tf.float32)
            if self.config.normalize_input:
                image = image / 255.0

            # Get image dimensions
            shape = tf.shape(image)
            height, width = shape[0], shape[1]

            # Skip images that are too small
            min_dim = tf.minimum(height, width)
            if min_dim < self.config.patch_size:
                # Resize small images
                scale_factor = tf.cast(self.config.patch_size, tf.float32) / tf.cast(min_dim, tf.float32)
                new_height = tf.cast(tf.cast(height, tf.float32) * scale_factor, tf.int32)
                new_width = tf.cast(tf.cast(width, tf.float32) * scale_factor, tf.int32)
                image = tf.image.resize(image, [new_height, new_width])

            # Extract random patches efficiently
            patches = []
            for _ in range(self.config.patches_per_image):
                # Random crop
                patch = tf.image.random_crop(
                    image,
                    [self.config.patch_size, self.config.patch_size, self.config.channels]
                )
                patches.append(patch)

            return tf.stack(patches)

        return process_image()

    def _augment_patch(self, patch: tf.Tensor) -> tf.Tensor:
        """Apply data augmentation to a patch."""
        # Random flips
        patch = tf.image.random_flip_left_right(patch)
        patch = tf.image.random_flip_up_down(patch)

        # Random 90-degree rotations
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        patch = tf.image.rot90(patch, k=k)

        return patch

class NoiseAugmentation:
    """Handles noise augmentation for bias-free denoiser training."""

    def __init__(self, sigma_min: float = 0.0, sigma_max: float = 0.4,
                 distribution: str = 'uniform'):
        """
        Initialize noise augmentation.

        Args:
            sigma_min: Minimum noise standard deviation
            sigma_max: Maximum noise standard deviation
            distribution: 'uniform' or 'log_uniform' sampling
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.distribution = distribution

    def sample_noise_level(self, batch_size: int) -> tf.Tensor:
        """Sample noise levels for a batch."""
        if self.distribution == 'uniform':
            noise_levels = tf.random.uniform(
                [batch_size, 1, 1, 1],
                self.sigma_min,
                self.sigma_max
            )
        elif self.distribution == 'log_uniform':
            log_min = tf.math.log(max(self.sigma_min, 1e-6))
            log_max = tf.math.log(self.sigma_max)
            log_noise = tf.random.uniform([batch_size, 1, 1, 1], log_min, log_max)
            noise_levels = tf.exp(log_noise)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        return noise_levels

    def add_noise(self, images: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Add Gaussian noise to clean images.

        Args:
            images: Clean images tensor [batch, height, width, channels]

        Returns:
            Tuple of (noisy_images, clean_images)
        """
        batch_size = tf.shape(images)[0]
        noise_levels = self.sample_noise_level(batch_size)

        # Generate Gaussian noise
        noise = tf.random.normal(tf.shape(images)) * noise_levels
        noisy_images = images + noise

        # Clip to valid range if input is normalized
        noisy_images = tf.clip_by_value(noisy_images, 0.0, 1.0)

        return noisy_images, images

# =====================================================================
# MONITORING AND CALLBACKS
# =====================================================================

class MemoryEfficientResultMonitor(keras.callbacks.Callback):
    """Memory-efficient monitoring of intermediate results."""

    def __init__(self, config: TrainingConfig, dataset_builder: MemoryEfficientDatasetBuilder):
        super().__init__()
        self.config = config
        self.dataset_builder = dataset_builder
        self.monitor_freq = config.monitor_every_n_epochs
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.results_dir = self.output_dir / "intermediate_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Create a small validation dataset for monitoring
        self._create_monitor_dataset()

        self.noise_augmenter = NoiseAugmentation(
            config.noise_sigma_min,
            config.noise_sigma_max,
            config.noise_distribution
        )

    def _create_monitor_dataset(self):
        """Create a small dataset for consistent monitoring."""
        # Create a small validation dataset (just a few batches)
        val_dataset = self.dataset_builder._create_dataset(
            self.dataset_builder._val_file_patterns,
            is_training=False,
            max_images=self.config.batch_size * 2  # Just 2 batches for monitoring
        )

        # Get one batch and cache it
        self.test_batch = next(iter(val_dataset.take(1)))
        logger.info(f"Created monitoring dataset with batch shape: {self.test_batch.shape}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Save intermediate results every N epochs."""
        if (epoch + 1) % self.monitor_freq != 0:
            return

        logger.info(f"Saving intermediate results for epoch {epoch + 1}")

        try:
            # Add noise to test batch
            noisy_images, clean_images = self.noise_augmenter.add_noise(self.test_batch)

            # Denoise images
            denoised_images = self.model(noisy_images, training=False)

            # Save sample images
            if self.config.save_training_images:
                self._save_image_samples(
                    epoch + 1,
                    noisy_images,
                    clean_images,
                    denoised_images
                )

            # Compute and log metrics
            mse_loss = tf.reduce_mean(tf.square(denoised_images - clean_images))
            psnr = tf.reduce_mean(tf.image.psnr(denoised_images, clean_images, max_val=1.0))

            logger.info(f"Epoch {epoch + 1} - Validation MSE: {mse_loss:.6f}, PSNR: {psnr:.2f} dB")

            # Save metrics
            metrics = {
                'epoch': epoch + 1,
                'val_mse': float(mse_loss),
                'val_psnr': float(psnr),
                'timestamp': datetime.now().isoformat()
            }

            metrics_file = self.results_dir / f"epoch_{epoch + 1:03d}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            # Force garbage collection to free memory
            del noisy_images, clean_images, denoised_images
            gc.collect()

        except Exception as e:
            logger.error(f"Error during monitoring at epoch {epoch + 1}: {e}")

    def _save_image_samples(self, epoch: int, noisy: tf.Tensor,
                          clean: tf.Tensor, denoised: tf.Tensor):
        """Save sample images for visual inspection."""
        num_samples = min(4, noisy.shape[0])

        fig, axes = plt.subplots(3, num_samples, figsize=(4 * num_samples, 12))
        if num_samples == 1:
            axes = axes.reshape(-1, 1)

        for i in range(num_samples):
            # Convert to numpy and handle channel dimension
            noisy_img = noisy[i].numpy()
            clean_img = clean[i].numpy()
            denoised_img = denoised[i].numpy()

            if noisy_img.shape[-1] == 1:
                noisy_img = noisy_img.squeeze(-1)
                clean_img = clean_img.squeeze(-1)
                denoised_img = denoised_img.squeeze(-1)
                cmap = 'gray'
            else:
                cmap = None

            # Plot images
            axes[0, i].imshow(noisy_img, cmap=cmap, vmin=0, vmax=1)
            axes[0, i].set_title(f'Noisy {i+1}')
            axes[0, i].axis('off')

            axes[1, i].imshow(clean_img, cmap=cmap, vmin=0, vmax=1)
            axes[1, i].set_title(f'Clean {i+1}')
            axes[1, i].axis('off')

            axes[2, i].imshow(denoised_img, cmap=cmap, vmin=0, vmax=1)
            axes[2, i].set_title(f'Denoised {i+1}')
            axes[2, i].axis('off')

        plt.tight_layout()
        save_path = self.results_dir / f"epoch_{epoch:03d}_samples.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Explicitly free memory
        plt.clf()
        gc.collect()

def create_callbacks(config: TrainingConfig, dataset_builder: MemoryEfficientDatasetBuilder) -> List[keras.callbacks.Callback]:
    """Create training callbacks for memory-efficient training."""
    callbacks = []

    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Learning rate reduction
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        )
    )

    # CSV logging
    csv_path = output_dir / "training_log.csv"
    callbacks.append(
        keras.callbacks.CSVLogger(str(csv_path), append=True)
    )

    # Memory-efficient intermediate result monitoring
    callbacks.append(
        MemoryEfficientResultMonitor(config, dataset_builder)
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

# =====================================================================
# TRAINING FUNCTION
# =====================================================================

def train_bfcnn_denoiser(config: TrainingConfig) -> keras.Model:
    """
    Train a bias-free CNN denoiser model with memory-efficient data loading.

    Args:
        config: Training configuration

    Returns:
        Trained Keras model
    """
    logger.info("Starting memory-efficient bias-free CNN denoiser training")
    logger.info(f"Experiment: {config.experiment_name}")

    # Create output directory and save config
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    # Create memory-efficient dataset builder
    logger.info("Creating memory-efficient dataset builder...")
    dataset_builder = MemoryEfficientDatasetBuilder(config)

    # Estimate dataset sizes
    train_size, val_size = dataset_builder.estimate_dataset_sizes()

    # Create datasets
    logger.info("Creating streaming datasets...")
    train_dataset = dataset_builder.create_training_dataset()
    val_dataset = dataset_builder.create_validation_dataset()

    # Add noise augmentation
    noise_augmenter = NoiseAugmentation(
        config.noise_sigma_min,
        config.noise_sigma_max,
        config.noise_distribution
    )

    def add_noise_to_batch(batch):
        return noise_augmenter.add_noise(batch)

    train_dataset = train_dataset.map(add_noise_to_batch, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(add_noise_to_batch, num_parallel_calls=tf.data.AUTOTUNE)

    # Calculate steps per epoch
    if config.steps_per_epoch is not None:
        steps_per_epoch = config.steps_per_epoch
    else:
        if train_size and config.max_images_per_epoch:
            effective_size = min(train_size, config.max_images_per_epoch)
        elif config.max_images_per_epoch:
            effective_size = config.max_images_per_epoch
        elif train_size:
            effective_size = train_size
        else:
            effective_size = 1000  # Fallback

        # Account for patches per image
        total_patches = effective_size * config.patches_per_image
        steps_per_epoch = max(1, total_patches // config.batch_size)

    logger.info(f"Using {steps_per_epoch} steps per epoch")

    # Create model
    logger.info(f"Creating {config.model_type} model...")
    input_shape = (config.patch_size, config.patch_size, config.channels)

    if config.model_type == 'light':
        from dl_techniques.models.bfcnn_denoiser import create_bfcnn_light
        model = create_bfcnn_light(input_shape)
    elif config.model_type == 'standard':
        model = create_bfcnn_standard(input_shape)
    elif config.model_type == 'deep':
        from dl_techniques.models.bfcnn_denoiser import create_bfcnn_deep
        model = create_bfcnn_deep(input_shape)
    elif config.model_type == 'custom':
        model = create_bfcnn_denoiser(
            input_shape=input_shape,
            num_blocks=config.num_blocks,
            filters=config.filters,
            initial_kernel_size=config.initial_kernel_size,
            kernel_size=config.kernel_size,
            activation=config.activation,
            model_name=f'bfcnn_custom_{config.experiment_name}'
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

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

    # Compile model with MSE loss (least-squares training as per paper)
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Mean Squared Error for Gaussian noise
        metrics=[
            'mae',
            keras.metrics.RootMeanSquaredError(name='rmse'),
            # PSNR metric
            lambda y_true, y_pred: tf.reduce_mean(
                tf.image.psnr(y_pred, y_true, max_val=1.0)
            )
        ]
    )

    logger.info(f"Model compiled with {model.count_params():,} parameters")

    # Create callbacks
    callbacks = create_callbacks(config, dataset_builder)

    # Train model
    logger.info("Starting training...")
    start_time = time.time()

    # Determine validation steps
    validation_steps = config.validation_steps
    if validation_steps is None:
        validation_steps = max(1, steps_per_epoch // 10)  # 10% of training steps

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
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # Save final model
    final_model_path = output_dir / "final_model.keras"
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # Save training history
    history_path = output_dir / "training_history.json"
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)

    # Clean up memory
    gc.collect()

    return model

# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    """Main training function for massive datasets."""
    # Memory-efficient configuration for large datasets
    config = TrainingConfig(
        # Data paths - UPDATE THESE FOR YOUR DATASET
        train_image_dirs=[
            'data/train',
            'data/additional_train',
            # Add more directories as needed
        ],
        val_image_dirs=['data/val'],

        # Training parameters
        patch_size=64,
        channels=1,  # 1 for grayscale, 3 for RGB
        batch_size=32,
        epochs=100,
        patches_per_image=16,

        # Memory management for massive datasets
        max_images_per_epoch=50000,  # Limit images per epoch to manage memory
        prefetch_buffer_size=100,    # Prefetch buffer
        parallel_reads=8,            # Parallel file reading
        dataset_shuffle_buffer=2000, # Shuffle buffer size
        estimate_dataset_size=True,  # Estimate total dataset size

        # Model configuration
        model_type='standard',  # 'light', 'standard', 'deep', or 'custom'

        # Noise configuration (universal range as per paper)
        noise_sigma_min=0.0,
        noise_sigma_max=0.4,
        noise_distribution='uniform',

        # Optimization
        learning_rate=1e-3,
        optimizer_type='adam',
        lr_schedule_type='cosine_decay',
        warmup_epochs=5,

        # Monitoring
        monitor_every_n_epochs=5,
        save_training_images=True,
        validation_steps=200,  # Fixed number of validation steps

        # Output
        output_dir='bfcnn_experiments',
        experiment_name=None  # Auto-generated with timestamp
    )

    try:
        # Train the model
        model = train_bfcnn_denoiser(config)
        logger.info("Training completed successfully!")

        # Print model summary
        model.summary()

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()