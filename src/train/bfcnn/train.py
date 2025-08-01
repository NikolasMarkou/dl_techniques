"""
Bias-Free CNN Denoiser Training Script with Complete Fixes for Graph Mode Compatibility
"""

import gc
import json
import time
import keras
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any

from dl_techniques.utils.logger import logger
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder
from dl_techniques.models.bfcnn_denoiser import create_bfcnn_denoiser, create_bfcnn_standard

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for bias-free CNN denoiser training."""

    # === Data Configuration ===
    train_image_dirs: List[str]  # Directories containing training images
    val_image_dirs: List[str]    # Directories containing validation images
    patch_size: int = 64         # Size of training patches (patch_size x patch_size)
    channels: int = 1            # Number of input channels (1=grayscale, 3=RGB)
    image_extensions: List[str] = None  # Supported image formats

    # === Memory Management ===
    max_train_files: Optional[int] = None     # Limit training files (None = no limit)
    max_val_files: Optional[int] = None       # Limit validation files
    parallel_reads: int = 4                   # Parallel file reading threads
    dataset_shuffle_buffer: int = 1000        # Shuffle buffer size

    # === Noise Configuration ===
    noise_sigma_min: float = 0.0    # Minimum noise standard deviation
    noise_sigma_max: float = 0.4    # Maximum noise standard deviation (universal range)
    noise_distribution: str = 'uniform'  # 'uniform' or 'log_uniform' sampling

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
            self.image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp']

        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"bfcnn_{self.model_type}_{timestamp}"

        # Validation
        if self.noise_sigma_min < 0 or self.noise_sigma_max <= self.noise_sigma_min:
            raise ValueError("Invalid noise sigma range")

        if self.patch_size <= 0 or self.channels <= 0:
            raise ValueError("Invalid patch size or channel configuration")

# ---------------------------------------------------------------------
# FILE DISCOVERY UTILITIES
# ---------------------------------------------------------------------

def discover_image_files(directories: List[str],
                        extensions: List[str],
                        max_files: Optional[int] = None) -> List[str]:
    """
    Discover image files in directories.

    Args:
        directories: List of directories to search
        extensions: List of valid file extensions
        max_files: Maximum number of files to return

    Returns:
        List of discovered image file paths
    """
    if not directories:
        logger.warning("No directories provided for file discovery")
        return []

    extensions_set = set(ext.lower() for ext in extensions)
    extensions_set.update(ext.upper() for ext in extensions)

    discovered_files = []

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {directory}")
            continue

        logger.info(f"Discovering files in: {directory}")

        try:
            for file_path in dir_path.rglob("*"):
                if file_path.is_file() and file_path.suffix in extensions_set:
                    discovered_files.append(str(file_path))

                    if max_files and len(discovered_files) >= max_files:
                        logger.info(f"Reached max files limit: {max_files}")
                        return discovered_files

                    if len(discovered_files) % 1000 == 0:
                        logger.info(f"Discovered {len(discovered_files)} files...")

        except Exception as e:
            logger.error(f"Error discovering files in {directory}: {e}")
            continue

    logger.info(f"Total files discovered: {len(discovered_files)}")
    return discovered_files

# ---------------------------------------------------------------------
# DATASET BUILDER WITH GRAPH MODE FIXES
# ---------------------------------------------------------------------

def load_and_preprocess_image(image_path: tf.Tensor, config: TrainingConfig) -> tf.Tensor:
    """
    Load and preprocess a single image using TensorFlow operations.
    FIXED: Uses tf.cond for graph-compatible conditional execution and ceil for robust resizing.

    Args:
        image_path: Tensor containing path to image file
        config: Training configuration

    Returns:
        Preprocessed image patch as tensor
    """
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
        image = image / 255.0

    # Get image dimensions
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    # Handle small images by resizing
    min_dim = tf.minimum(height, width)
    min_size = config.patch_size

    # --- FIXED: Use tf.cond for data-dependent control flow ---
    def resize_if_small():
        """Logic to resize the image if it's smaller than the patch size."""
        scale_factor = tf.cast(min_size, tf.float32) / tf.cast(min_dim, tf.float32)

        # --- SOLUTION: Use tf.math.ceil to avoid truncation errors ---
        # This ensures the new dimensions are always large enough after scaling.
        new_height = tf.cast(tf.math.ceil(tf.cast(height, tf.float32) * scale_factor), tf.int32)
        new_width = tf.cast(tf.math.ceil(tf.cast(width, tf.float32) * scale_factor), tf.int32)

        return tf.image.resize(image, [new_height, new_width])

    def identity():
        """Logic to return the image as is."""
        return image

    # Use tf.cond to choose which function to execute
    # More explicit condition: resize if either dimension is too small
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

def augment_patch(patch: tf.Tensor) -> tf.Tensor:
    """Apply data augmentation to a patch."""
    # Random flips
    patch = tf.image.random_flip_left_right(patch)
    patch = tf.image.random_flip_up_down(patch)

    # Random 90-degree rotations
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    patch = tf.image.rot90(patch, k=k)

    return patch

def add_noise_to_patch(patch: tf.Tensor, config: TrainingConfig) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Add Gaussian noise to a clean patch.

    Args:
        patch: Clean patch tensor [height, width, channels]
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

    # Clip to valid range if input is normalized
    noisy_patch = tf.clip_by_value(noisy_patch, 0.0, 1.0)

    return noisy_patch, patch

def create_dataset(file_paths: List[str], config: TrainingConfig, is_training: bool = True) -> tf.data.Dataset:
    """
    Create a properly structured dataset that yields (x, y) pairs.

    Args:
        file_paths: List of image file paths
        config: Training configuration
        is_training: Whether this is for training (affects augmentation and repetition)

    Returns:
        TensorFlow Dataset yielding (noisy_patches, clean_patches) pairs
    """
    logger.info(f"Creating {'training' if is_training else 'validation'} dataset from {len(file_paths)} files")

    # Create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)

    if is_training:
        # Repeat files to get multiple patches per image
        dataset = dataset.repeat(config.patches_per_image)

        # Shuffle files
        dataset = dataset.shuffle(
            buffer_size=min(len(file_paths) * config.patches_per_image, config.dataset_shuffle_buffer),
            reshuffle_each_iteration=True
        )

        # Repeat for multiple epochs
        dataset = dataset.repeat()
    else:
        # For validation, just repeat to get multiple patches
        dataset = dataset.repeat(config.patches_per_image)

    # Load and preprocess images directly (graph-compatible)
    dataset = dataset.map(
        lambda path: load_and_preprocess_image(path, config),
        num_parallel_calls=config.parallel_reads
    )

    # Set shape explicitly after loading and cropping
    dataset = dataset.map(
        lambda x: tf.ensure_shape(x, [config.patch_size, config.patch_size, config.channels])
    )

    # Apply augmentation for training
    if is_training and config.augment_data:
        dataset = dataset.map(augment_patch, num_parallel_calls=tf.data.AUTOTUNE)

    # Add noise to create (input, target) pairs directly (graph-compatible)
    dataset = dataset.map(
        lambda patch: add_noise_to_patch(patch, config),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Set shapes for the tuple
    dataset = dataset.map(
        lambda noisy, clean: (
            tf.ensure_shape(noisy, [config.patch_size, config.patch_size, config.channels]),
            tf.ensure_shape(clean, [config.patch_size, config.patch_size, config.channels])
        )
    )

    # Batch and prefetch
    dataset = dataset.batch(config.batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def psnr_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    PSNR (Peak Signal-to-Noise Ratio) metric for image denoising.

    Args:
        y_true: Ground truth images
        y_pred: Predicted/denoised images

    Returns:
        Mean PSNR value across the batch
    """
    return tf.reduce_mean(tf.image.psnr(y_pred, y_true, max_val=1.0))

# ---------------------------------------------------------------------
# MONITORING AND CALLBACKS WITH GRAPH MODE FIXES
# ---------------------------------------------------------------------

class StreamingResultMonitor(keras.callbacks.Callback):
    """
    Memory-efficient monitoring using streaming data.
    FIXED: Uses tf.py_function for graph-compatible eager operations.
    """

    def __init__(self, config: TrainingConfig, val_file_paths: List[str]):
        super().__init__()
        self.config = config
        self.monitor_freq = config.monitor_every_n_epochs
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.results_dir = self.output_dir / "intermediate_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Create a small validation dataset for monitoring
        self._create_monitor_dataset(val_file_paths)

    def _create_monitor_dataset(self, val_file_paths: List[str]):
        """Create a small dataset for consistent monitoring."""
        monitor_files = val_file_paths[:min(4, len(val_file_paths))]
        clean_patches = []
        for file_path in monitor_files:
            path_tensor = tf.constant(file_path)
            patch = load_and_preprocess_image(path_tensor, self.config)
            clean_patches.append(patch)

        self.test_batch = tf.stack(clean_patches)
        logger.info(f"Created monitoring dataset with batch shape: {self.test_batch.shape}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Save intermediate results every N epochs, using tf.py_function for eager execution."""
        if (epoch + 1) % self.monitor_freq != 0:
            return

        # Define the function that contains non-graph compatible code
        def _monitor_and_save(epoch_numpy):
            epoch_val = int(epoch_numpy)
            logger.info(f"Saving intermediate results for epoch {epoch_val}")

            try:
                # Add noise (TF ops)
                noisy_images, clean_images = add_noise_to_patch(self.test_batch, self.config)

                # Denoise images (TF ops)
                denoised_images = self.model(noisy_images, training=False)

                # Save sample images (calls .numpy(), needs eager mode)
                if self.config.save_training_images:
                    self._save_image_samples(
                        epoch_val,
                        noisy_images,
                        clean_images,
                        denoised_images
                    )

                # Compute metrics (TF ops)
                mse_loss = tf.reduce_mean(tf.square(denoised_images - clean_images))
                psnr = tf.reduce_mean(tf.image.psnr(denoised_images, clean_images, max_val=1.0))

                # Log and save metrics (calls .numpy()/float(), needs eager mode)
                logger.info(f"Epoch {epoch_val} - Validation MSE: {mse_loss.numpy():.6f}, PSNR: {psnr.numpy():.2f} dB")

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
                # Use tf.print for better visibility of errors inside a py_function
                tf.print(f"Error during monitoring callback at epoch {epoch_val}: {e}")

            # tf.py_function must have a return value
            return 0

        # Wrap the eager function in tf.py_function
        tf.py_function(func=_monitor_and_save, inp=[epoch + 1], Tout=[tf.int32])

    def _save_image_samples(self, epoch: int, noisy: tf.Tensor,
                          clean: tf.Tensor, denoised: tf.Tensor):
        """Save sample images for visual inspection."""
        num_samples = min(4, noisy.shape[0])
        fig, axes = plt.subplots(3, num_samples, figsize=(4 * num_samples, 12))
        if num_samples == 1:
            axes = axes.reshape(-1, 1)

        for i in range(num_samples):
            # This part is now safe as it's called from within the tf.py_function
            noisy_img = noisy[i].numpy()
            clean_img = clean[i].numpy()
            denoised_img = denoised[i].numpy()

            if noisy_img.shape[-1] == 1:
                noisy_img, clean_img, denoised_img = map(lambda x: x.squeeze(-1), [noisy_img, clean_img, denoised_img])
                cmap = 'gray'
            else:
                cmap = None

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
        plt.close(fig)
        plt.clf()
        gc.collect()

def create_callbacks(config: TrainingConfig, val_file_paths: List[str]) -> List[keras.callbacks.Callback]:
    """Create training callbacks for streaming training."""
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

    # Streaming result monitoring with graph mode fixes
    callbacks.append(
        StreamingResultMonitor(config, val_file_paths)
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
# TRAINING FUNCTION
# ---------------------------------------------------------------------

def train_bfcnn_denoiser(config: TrainingConfig) -> keras.Model:
    """
    Train a bias-free CNN denoiser model with streaming data loading.

    Args:
        config: Training configuration

    Returns:
        Trained Keras model
    """
    logger.info("Starting streaming bias-free CNN denoiser training")
    logger.info(f"Experiment: {config.experiment_name}")

    # Create output directory and save config
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    # Discover files
    logger.info("Discovering training files...")
    train_files = discover_image_files(
        config.train_image_dirs,
        config.image_extensions,
        config.max_train_files
    )

    logger.info("Discovering validation files...")
    val_files = discover_image_files(
        config.val_image_dirs,
        config.image_extensions,
        config.max_val_files
    )

    if not train_files:
        raise ValueError("No training files found!")
    if not val_files:
        raise ValueError("No validation files found!")

    logger.info(f"Found {len(train_files)} training files")
    logger.info(f"Found {len(val_files)} validation files")

    # Create datasets that yield (x, y) pairs
    logger.info("Creating datasets...")
    train_dataset = create_dataset(train_files, config, is_training=True)
    val_dataset = create_dataset(val_files, config, is_training=False)

    # Calculate steps per epoch based on discovered files
    if config.steps_per_epoch is not None:
        steps_per_epoch = config.steps_per_epoch
    else:
        # Calculate based on actual discovered files
        total_patches = len(train_files) * config.patches_per_image
        steps_per_epoch = max(100, total_patches // config.batch_size)

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

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Mean Squared Error for Gaussian noise
        metrics=[
            'mae',
            keras.metrics.RootMeanSquaredError(name='rmse'),
            psnr_metric
        ]
    )

    logger.info(f"Model compiled with {model.count_params():,} parameters")

    # Create callbacks
    callbacks = create_callbacks(config, val_files)

    # Train model
    logger.info("Starting training...")
    start_time = time.time()

    # Determine validation steps
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

# ---------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------

def main():
    """Main training function with streaming file discovery and graph mode compatibility."""
    # Configuration for streaming training
    config = TrainingConfig(
        # Data paths
        train_image_dirs=[
            '/media/arxwn/data0_4tb/datasets/Megadepth',
            '/media/arxwn/data0_4tb/datasets/VGG-Face2/data/train',
            '/media/arxwn/data0_4tb/datasets/ade20k/images/ADE/training',
            '/media/arxwn/data0_4tb/datasets/KITTI/data/depth/raw_image_values'
        ],
        val_image_dirs=[
            '/media/arxwn/data0_4tb/datasets/ade20k/images/ADE/validation'
        ],

        # Training parameters
        patch_size=64,
        channels=1,  # 1 for grayscale, 3 for RGB
        batch_size=32,
        epochs=100,
        patches_per_image=16,

        # File limits for manageable training
        max_train_files=10000,      # Limit training files
        max_val_files=1000,         # Limit validation files
        parallel_reads=8,           # Parallel file processing
        dataset_shuffle_buffer=2000, # Shuffle buffer size

        # Model configuration
        model_type='standard',  # 'light', 'standard', 'deep', or 'custom'

        # Noise configuration (universal range as per paper)
        noise_sigma_min=0.0,
        noise_sigma_max=0.4,
        noise_distribution='uniform',

        # Optimization
        learning_rate=1e-3,
        optimizer_type='adamw',
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
        # Train the model with streaming data and graph mode fixes
        model = train_bfcnn_denoiser(config)
        logger.info("Training completed successfully!")

        # Print model summary
        model.summary()

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()