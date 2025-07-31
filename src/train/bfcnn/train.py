"""
Bias-Free CNN Denoiser Training Script

Comprehensive training script for the bias-free CNN denoiser model with:
- Universal noise handling (σ ∈ [0, 0.4])
- Least-squares training with MSE loss
- Blind operation capability
- Intermediate result monitoring
- Patch-based training from image directories

Based on "Robust and Interpretable Blind Image Denoising via Bias-Free
Convolutional Neural Networks" (Mohan et al., ICLR 2020).
"""

import json
import time
import keras
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any

# Local imports
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder
from dl_techniques.models.bfcnn_denoiser import create_bfcnn_denoiser, create_bfcnn_standard


# =====================================================================
# CONFIGURATION
# =====================================================================

@dataclass
class TrainingConfig:
    """Comprehensive configuration for bias-free CNN denoiser training."""

    # === Data Configuration ===
    train_image_dirs: List[str]  # Directories containing training images
    val_image_dirs: List[str]  # Directories containing validation images
    patch_size: int = 64  # Size of training patches (patch_size x patch_size)
    channels: int = 1  # Number of input channels (1=grayscale, 3=RGB)
    image_extensions: List[str] = None  # Supported image formats

    # === Noise Configuration ===
    noise_sigma_min: float = 0.0  # Minimum noise standard deviation
    noise_sigma_max: float = 0.4  # Maximum noise standard deviation (universal range)
    noise_distribution: str = 'uniform'  # 'uniform' or 'log_uniform' sampling
    blind_training: bool = True  # Train without providing noise level to model

    # === Model Configuration ===
    model_type: str = 'standard'  # 'light', 'standard', 'deep', or 'custom'
    num_blocks: int = 8  # Number of residual blocks (for custom model)
    filters: int = 64  # Number of filters (for custom model)
    initial_kernel_size: int = 5  # Initial convolution kernel size
    kernel_size: int = 3  # Residual block kernel size
    activation: str = 'relu'  # Activation function

    # === Training Configuration ===
    batch_size: int = 32
    epochs: int = 100
    patches_per_image: int = 16  # Number of patches to extract per image
    augment_data: bool = True  # Apply data augmentation
    normalize_input: bool = True  # Normalize input to [0, 1]

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
    reduce_lr_patience: int = 8  # Learning rate reduction patience

    # === Output Configuration ===
    output_dir: str = 'bfcnn_experiments'
    experiment_name: str = None  # Auto-generated if None
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


# Default configuration
DEFAULT_CONFIG = TrainingConfig(
    train_image_dirs=['data/train'],
    val_image_dirs=['data/val'],
    patch_size=64,
    channels=1,
    noise_sigma_max=0.4,  # Universal noise range as per paper
    batch_size=32,
    epochs=100,
    learning_rate=1e-3,
    monitor_every_n_epochs=5
)


# =====================================================================
# DATA LOADING AND PREPROCESSING
# =====================================================================

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


def load_images_from_directory(directory: str, extensions: List[str]) -> List[str]:
    """Load all image paths from a directory."""
    image_paths = []
    directory = Path(directory)

    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return image_paths

    for ext in extensions:
        image_paths.extend(list(directory.glob(f"**/*{ext}")))
        image_paths.extend(list(directory.glob(f"**/*{ext.upper()}")))

    logger.info(f"Found {len(image_paths)} images in {directory}")
    return [str(p) for p in image_paths]


def extract_patches(image: tf.Tensor, patch_size: int, num_patches: int) -> tf.Tensor:
    """Extract random patches from an image."""
    image_shape = tf.shape(image)
    height, width = image_shape[0], image_shape[1]

    # Ensure we can extract patches
    min_size = tf.minimum(height, width)
    if min_size < patch_size:
        # Resize image if too small
        scale = patch_size / tf.cast(min_size, tf.float32)
        new_height = tf.cast(tf.cast(height, tf.float32) * scale, tf.int32)
        new_width = tf.cast(tf.cast(width, tf.float32) * scale, tf.int32)
        image = tf.image.resize(image, [new_height, new_width])
        height, width = new_height, new_width

    # Random patch extraction
    patches = []
    for _ in range(num_patches):
        # Random top-left corner
        max_y = height - patch_size
        max_x = width - patch_size
        y = tf.random.uniform([], 0, max_y + 1, dtype=tf.int32)
        x = tf.random.uniform([], 0, max_x + 1, dtype=tf.int32)

        # Extract patch
        patch = tf.image.crop_to_bounding_box(image, y, x, patch_size, patch_size)
        patches.append(patch)

    return tf.stack(patches)


def create_dataset(config: TrainingConfig, is_training: bool = True) -> tf.data.Dataset:
    """Create a tf.data.Dataset for training or validation."""
    directories = config.train_image_dirs if is_training else config.val_image_dirs

    # Collect all image paths
    all_paths = []
    for directory in directories:
        paths = load_images_from_directory(directory, config.image_extensions)
        all_paths.extend(paths)

    if not all_paths:
        raise ValueError(f"No images found in directories: {directories}")

    def load_and_preprocess(image_path):
        """Load and preprocess a single image."""
        # Load image
        image_string = tf.io.read_file(image_path)

        # Decode image (handle different formats)
        try:
            image = tf.image.decode_image(image_string, channels=config.channels)
        except:
            # Fallback for problematic images
            return tf.zeros([config.patch_size, config.patch_size, config.channels])

        # Convert to float and normalize if requested
        image = tf.cast(image, tf.float32)
        if config.normalize_input:
            image = image / 255.0

        # Extract patches
        patches = extract_patches(image, config.patch_size, config.patches_per_image)

        return patches

    # Create dataset from paths
    dataset = tf.data.Dataset.from_tensor_slices(all_paths)

    if is_training:
        dataset = dataset.shuffle(len(all_paths))
        dataset = dataset.repeat()

    # Load and preprocess images
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Flatten patches (each image produces multiple patches)
    dataset = dataset.unbatch()

    # Data augmentation for training
    if is_training and config.augment_data:
        def augment(patch):
            # Random flips
            patch = tf.image.random_flip_left_right(patch)
            patch = tf.image.random_flip_up_down(patch)
            # Random rotation (90 degree increments)
            k = tf.random.uniform([], 0, 4, dtype=tf.int32)
            patch = tf.image.rot90(patch, k=k)
            return patch

        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the dataset
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    logger.info(f"Created {'training' if is_training else 'validation'} dataset with {len(all_paths)} images")

    return dataset


# =====================================================================
# MONITORING AND CALLBACKS
# =====================================================================

class IntermediateResultMonitor(keras.callbacks.Callback):
    """Monitor and save intermediate denoising results during training."""

    def __init__(self, config: TrainingConfig, val_dataset: tf.data.Dataset):
        super().__init__()
        self.config = config
        self.monitor_freq = config.monitor_every_n_epochs
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.results_dir = self.output_dir / "intermediate_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Get a fixed batch for consistent monitoring
        self.test_batch = next(iter(val_dataset.take(1)))
        self.noise_augmenter = NoiseAugmentation(
            config.noise_sigma_min,
            config.noise_sigma_max,
            config.noise_distribution
        )

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Save intermediate results every N epochs."""
        if (epoch + 1) % self.monitor_freq != 0:
            return

        logger.info(f"Saving intermediate results for epoch {epoch + 1}")

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
            axes[0, i].set_title(f'Noisy {i + 1}')
            axes[0, i].axis('off')

            axes[1, i].imshow(clean_img, cmap=cmap, vmin=0, vmax=1)
            axes[1, i].set_title(f'Clean {i + 1}')
            axes[1, i].axis('off')

            axes[2, i].imshow(denoised_img, cmap=cmap, vmin=0, vmax=1)
            axes[2, i].set_title(f'Denoised {i + 1}')
            axes[2, i].axis('off')

        plt.tight_layout()
        save_path = self.results_dir / f"epoch_{epoch:03d}_samples.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def create_callbacks(config: TrainingConfig, val_dataset: tf.data.Dataset) -> List[keras.callbacks.Callback]:
    """Create training callbacks."""
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

    # Intermediate result monitoring
    callbacks.append(
        IntermediateResultMonitor(config, val_dataset)
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
    Train a bias-free CNN denoiser model.

    Args:
        config: Training configuration

    Returns:
        Trained Keras model
    """
    logger.info("Starting bias-free CNN denoiser training")
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Configuration: {config}")

    # Create output directory and save config
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = create_dataset(config, is_training=True)
    val_dataset = create_dataset(config, is_training=False)

    # Add noise augmentation to training dataset
    noise_augmenter = NoiseAugmentation(
        config.noise_sigma_min,
        config.noise_sigma_max,
        config.noise_distribution
    )

    def add_noise_to_batch(batch):
        return noise_augmenter.add_noise(batch)

    train_dataset = train_dataset.map(add_noise_to_batch, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(add_noise_to_batch, num_parallel_calls=tf.data.AUTOTUNE)

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
    steps_per_epoch = len(load_images_from_directory(
        config.train_image_dirs[0], config.image_extensions
    )) * config.patches_per_image // config.batch_size

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
    callbacks = create_callbacks(config, val_dataset)

    # Train model
    logger.info("Starting training...")
    start_time = time.time()

    history = model.fit(
        train_dataset,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=max(1, steps_per_epoch // 10),  # 10% of training steps
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

    return model


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    """Main training function."""
    # Example configuration - modify as needed
    config = TrainingConfig(
        # Data paths - UPDATE THESE FOR YOUR DATASET
        train_image_dirs=['data/train', 'data/additional_train'],
        val_image_dirs=['data/val'],

        # Training parameters
        patch_size=64,
        channels=1,  # 1 for grayscale, 3 for RGB
        batch_size=32,
        epochs=100,
        patches_per_image=16,

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