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

from train.common import setup_gpu, create_callbacks as create_common_callbacks
from dl_techniques.utils.logger import logger
from dl_techniques.utils.filesystem import count_available_files
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder
from dl_techniques.models.bias_free_denoisers.bfcnn import (
    create_bfcnn_denoiser, BFCNN_CONFIGS, create_bfcnn_variant
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for bias-free CNN denoiser training."""

    # Data
    train_image_dirs: List[str] = field(default_factory=list)
    val_image_dirs: List[str] = field(default_factory=list)
    patch_size: int = 64
    channels: int = 1
    image_extensions: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp'])

    # Memory
    max_train_files: Optional[int] = None
    max_val_files: Optional[int] = None
    parallel_reads: int = 4
    dataset_shuffle_buffer: int = 1000

    # Noise
    noise_sigma_min: float = 0.0
    noise_sigma_max: float = 0.4
    noise_distribution: str = 'uniform'

    # Model
    model_type: str = 'tiny'
    num_blocks: int = 8
    filters: int = 64
    initial_kernel_size: int = 5
    kernel_size: int = 3
    activation: str = 'relu'

    # Training
    batch_size: int = 32
    epochs: int = 100
    patches_per_image: int = 16
    augment_data: bool = True
    normalize_input: bool = True
    steps_per_epoch: Optional[int] = None

    # Optimization
    learning_rate: float = 1e-3
    optimizer_type: str = 'adam'
    lr_schedule_type: str = 'cosine_decay'
    warmup_epochs: int = 5
    weight_decay: float = 1e-5
    gradient_clipping: float = 1.0

    # Monitoring
    monitor_every_n_epochs: int = 5
    save_best_only: bool = True
    early_stopping_patience: int = 15
    validation_steps: Optional[int] = 100

    # Output
    output_dir: str = 'results'
    experiment_name: Optional[str] = None
    save_training_images: bool = True
    save_model_checkpoints: bool = True

    def __post_init__(self):
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"bfcnn_{self.model_type}_{timestamp}"

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
# DATASET BUILDER
# ---------------------------------------------------------------------

def load_and_preprocess_image(image_path: tf.Tensor, config: TrainingConfig) -> tf.Tensor:
    """Load, preprocess, and extract a random patch from an image."""
    try:
        image_string = tf.io.read_file(image_path)
        image = tf.image.decode_image(image_string, channels=config.channels, expand_animations=False)
        image.set_shape([None, None, config.channels])
        image = tf.cast(image, tf.float32)
        if config.normalize_input:
            image = (image / 127.5) - 1.0

        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        min_size = config.patch_size

        image = tf.cond(
            tf.logical_or(height < min_size, width < min_size),
            true_fn=lambda: tf.image.resize(image, [
                tf.cast(tf.math.ceil(tf.cast(height, tf.float32) * tf.cast(min_size, tf.float32) / tf.cast(tf.minimum(height, width), tf.float32)), tf.int32),
                tf.cast(tf.math.ceil(tf.cast(width, tf.float32) * tf.cast(min_size, tf.float32) / tf.cast(tf.minimum(height, width), tf.float32)), tf.int32)
            ]),
            false_fn=lambda: image
        )

        return tf.image.random_crop(image, [config.patch_size, config.patch_size, config.channels])

    except tf.errors.InvalidArgumentError:
        logger.warning(f"Failed to load image: {image_path}")
        return tf.zeros([config.patch_size, config.patch_size, config.channels], dtype=tf.float32)


def augment_patch(patch: tf.Tensor) -> tf.Tensor:
    """Apply random flips and 90-degree rotations."""
    patch = tf.image.random_flip_left_right(patch)
    patch = tf.image.random_flip_up_down(patch)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    return tf.image.rot90(patch, k)


def add_noise_to_patch(patch: tf.Tensor, config: TrainingConfig) -> Tuple[tf.Tensor, tf.Tensor]:
    """Add Gaussian noise with configurable distribution to a clean patch."""
    if config.noise_distribution == 'uniform':
        noise_level = tf.random.uniform([], config.noise_sigma_min, config.noise_sigma_max)
    elif config.noise_distribution == 'log_uniform':
        log_min = tf.math.log(tf.maximum(config.noise_sigma_min, 1e-6))
        log_max = tf.math.log(config.noise_sigma_max)
        noise_level = tf.exp(tf.random.uniform([], log_min, log_max))
    else:
        raise ValueError(f"Unknown distribution: {config.noise_distribution}")

    noisy_patch = patch + tf.random.normal(tf.shape(patch)) * noise_level
    return tf.clip_by_value(noisy_patch, -1.0, 1.0), patch


def create_dataset(directories: List[str], config: TrainingConfig, is_training: bool = True) -> tf.data.Dataset:
    """Create a tf.data.Dataset of noisy/clean pairs from image directories."""
    logger.info(f"Creating {'training' if is_training else 'validation'} dataset from directories: {directories}")

    all_file_paths = []
    extensions_set = {ext.lower() for ext in config.image_extensions}
    extensions_set.update({ext.upper() for ext in config.image_extensions})

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.is_dir():
            logger.warning(f"Directory not found, skipping: {directory}")
            continue
        try:
            for file_path in dir_path.rglob("*"):
                if file_path.is_file() and file_path.suffix in extensions_set:
                    all_file_paths.append(str(file_path))
        except Exception as e:
            logger.warning(f"Error scanning directory {directory}: {e}")

    if not all_file_paths:
        raise ValueError(f"No image files found in directories: {directories}")
    logger.info(f"Found a total of {len(all_file_paths)} files.")

    limit = config.max_train_files if is_training else config.max_val_files
    if limit and limit < len(all_file_paths):
        logger.info(f"Limiting to {limit} files as per configuration.")
        np.random.shuffle(all_file_paths)
        all_file_paths = all_file_paths[:limit]

    dataset = tf.data.Dataset.from_tensor_slices(all_file_paths)

    if is_training:
        dataset = dataset.shuffle(
            buffer_size=min(config.dataset_shuffle_buffer, len(all_file_paths)),
            reshuffle_each_iteration=True
        )
    dataset = dataset.repeat()

    if is_training and config.patches_per_image > 1:
        dataset = dataset.flat_map(
            lambda path: tf.data.Dataset.from_tensors(path).repeat(config.patches_per_image)
        )

    dataset = dataset.map(
        lambda path: load_and_preprocess_image(path, config),
        num_parallel_calls=config.parallel_reads
    )
    dataset = dataset.filter(lambda x: tf.reduce_sum(tf.abs(x)) > 0)
    dataset = dataset.map(
        lambda x: tf.ensure_shape(x, [config.patch_size, config.patch_size, config.channels])
    )

    if is_training and config.augment_data:
        dataset = dataset.map(augment_patch, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.map(
        lambda patch: add_noise_to_patch(patch, config),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.map(
        lambda noisy, clean: (
            tf.ensure_shape(noisy, [config.patch_size, config.patch_size, config.channels]),
            tf.ensure_shape(clean, [config.patch_size, config.patch_size, config.channels])
        )
    )

    dataset = dataset.prefetch(config.batch_size * 2)
    dataset = dataset.batch(config.batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
def psnr_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """PSNR metric for [-1, +1] normalized images."""
    return tf.reduce_mean(tf.image.psnr(y_pred, y_true, max_val=2.0))


# ---------------------------------------------------------------------
# MONITORING AND CALLBACKS
# ---------------------------------------------------------------------

class MetricsVisualizationCallback(keras.callbacks.Callback):
    """Visualizes training/validation metrics during training."""

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.visualization_dir = self.output_dir / "visualization_plots"
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.train_metrics = {'loss': [], 'mae': [], 'rmse': [], 'psnr_metric': []}
        self.val_metrics = {'val_loss': [], 'val_mae': [], 'val_rmse': [], 'val_psnr_metric': []}

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if logs is None:
            logs = {}
        for key in self.train_metrics:
            if key in logs:
                self.train_metrics[key].append(logs[key])
        for key in self.val_metrics:
            if key in logs:
                self.val_metrics[key].append(logs[key])

        if (epoch + 1) % 5 == 0 or epoch == 0:
            self._create_metrics_plots(epoch + 1)

    def _create_metrics_plots(self, epoch: int):
        try:
            epochs_range = range(1, len(self.train_metrics['loss']) + 1)
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Training and Validation Metrics - Epoch {epoch}', fontsize=16)

            plot_configs = [
                (axes[0, 0], 'loss', 'val_loss', 'Mean Squared Error (MSE)', 'MSE'),
                (axes[0, 1], 'mae', 'val_mae', 'Mean Absolute Error (MAE)', 'MAE'),
                (axes[1, 0], 'rmse', 'val_rmse', 'Root Mean Squared Error (RMSE)', 'RMSE'),
                (axes[1, 1], 'psnr_metric', 'val_psnr_metric', 'Peak Signal-to-Noise Ratio (PSNR)', 'PSNR (dB)'),
            ]
            for ax, train_key, val_key, title, ylabel in plot_configs:
                ax.plot(epochs_range, self.train_metrics[train_key], 'b-', label=f'Training {ylabel}', linewidth=2)
                if self.val_metrics[val_key]:
                    ax.plot(epochs_range, self.val_metrics[val_key], 'r-', label=f'Validation {ylabel}', linewidth=2)
                ax.set_title(title)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(ylabel)
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.visualization_dir / f"epoch_{epoch:03d}_metrics.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            plt.clf()
            gc.collect()

            metrics_data = {'epoch': epoch, 'train_metrics': self.train_metrics, 'val_metrics': self.val_metrics}
            with open(self.visualization_dir / "latest_metrics.json", 'w') as f:
                json.dump(metrics_data, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

        except Exception as e:
            logger.warning(f"Failed to create metrics plots: {e}")


class StreamingResultMonitor(keras.callbacks.Callback):
    """Memory-efficient monitoring using streaming validation data."""

    def __init__(self, config: TrainingConfig, val_directories: List[str]):
        super().__init__()
        self.config = config
        self.monitor_freq = config.monitor_every_n_epochs
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.results_dir = self.output_dir / "visualization_plots"
        self.results_dir.mkdir(parents=True, exist_ok=True)
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

        if not monitor_files:
            logger.warning("No files found for monitoring")
            self.test_batch = None
            return

        try:
            clean_patches = [load_and_preprocess_image(tf.constant(fp), self.config) for fp in monitor_files]
            self.test_batch = tf.stack(clean_patches)
            logger.info(f"Created monitoring dataset with batch shape: {self.test_batch.shape}")
        except Exception as e:
            logger.error(f"Failed to create monitor dataset: {e}")
            self.test_batch = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if (epoch + 1) % self.monitor_freq != 0 or self.test_batch is None:
            return

        def _monitor_and_save(epoch_numpy):
            epoch_val = int(epoch_numpy)
            logger.info(f"Saving intermediate results for epoch {epoch_val}")
            try:
                noisy_images, clean_images = add_noise_to_patch(self.test_batch, self.config)
                denoised_images = self.model(noisy_images, training=False)

                if self.config.save_training_images:
                    self._save_image_samples(epoch_val, noisy_images, clean_images, denoised_images)

                mse_loss = tf.reduce_mean(tf.square(denoised_images - clean_images))
                psnr = tf.reduce_mean(tf.image.psnr(denoised_images, clean_images, max_val=2.0))
                logger.info(f"Epoch {epoch_val} - Validation MSE: {mse_loss.numpy():.6f}, PSNR: {psnr.numpy():.2f} dB")

                metrics = {
                    'epoch': epoch_val, 'val_mse': float(mse_loss),
                    'val_psnr': float(psnr), 'timestamp': datetime.now().isoformat()
                }
                with open(self.results_dir / f"epoch_{epoch_val:03d}_metrics.json", 'w') as f:
                    json.dump(metrics, f, indent=2)

                del noisy_images, clean_images, denoised_images
                gc.collect()
            except Exception as e:
                tf.print(f"Error during monitoring callback at epoch {epoch_val}: {e}")
            return 0

        tf.py_function(func=_monitor_and_save, inp=[epoch + 1], Tout=[tf.int32])

    def _save_image_samples(self, epoch: int, noisy: tf.Tensor, clean: tf.Tensor, denoised: tf.Tensor):
        """Save noisy/clean/denoised comparison grid."""
        try:
            num_samples = min(10, noisy.shape[0])
            fig, axes = plt.subplots(3, 10, figsize=(25, 7.5))
            fig.suptitle(f'Denoising Results - Epoch {epoch}', fontsize=20, y=0.98)

            for i in range(10):
                if i < num_samples:
                    # Denormalize from [-1, +1] to [0, 1]
                    clean_img = np.clip((clean[i].numpy() + 1.0) / 2.0, 0.0, 1.0)
                    noisy_img = np.clip((noisy[i].numpy() + 1.0) / 2.0, 0.0, 1.0)
                    denoised_img = np.clip((denoised[i].numpy() + 1.0) / 2.0, 0.0, 1.0)

                    cmap = 'gray' if clean_img.shape[-1] == 1 else None
                    if clean_img.shape[-1] == 1:
                        clean_img, noisy_img, denoised_img = clean_img.squeeze(-1), noisy_img.squeeze(-1), denoised_img.squeeze(-1)

                    row_labels = ['Clean', 'Noisy', 'Denoised']
                    for row, img in enumerate([clean_img, noisy_img, denoised_img]):
                        axes[row, i].imshow(img, cmap=cmap, vmin=0, vmax=1)
                        if i == 0:
                            axes[row, i].set_ylabel(row_labels[row], fontsize=14, rotation=0, ha='right', va='center')
                        axes[row, i].axis('off')
                    axes[0, i].set_title(f'Sample {i + 1}', fontsize=10)
                else:
                    for row in range(3):
                        axes[row, i].axis('off')

            plt.tight_layout()
            plt.subplots_adjust(top=0.92, left=0.08, right=0.98)
            plt.savefig(self.results_dir / f"epoch_{epoch:03d}_samples.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            plt.clf()
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to save image samples: {e}")


def create_callbacks(config: TrainingConfig, val_directories: List[str]) -> List[keras.callbacks.Callback]:
    """Create training callbacks using common utilities plus domain-specific callbacks."""
    common_callbacks, _ = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix=str(Path(config.output_dir) / config.experiment_name),
        monitor="val_loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_tensorboard=True,
        include_analyzer=False,
    )

    # Domain-specific callbacks
    common_callbacks.append(MetricsVisualizationCallback(config))
    common_callbacks.append(StreamingResultMonitor(config, val_directories))

    return common_callbacks


# ---------------------------------------------------------------------
# MODEL CREATION
# ---------------------------------------------------------------------

def create_model_instance(config: TrainingConfig, input_shape: Tuple[int, int, int]) -> keras.Model:
    """Create a BFCNN model from config (variant or custom)."""
    if config.model_type in BFCNN_CONFIGS:
        return create_bfcnn_variant(variant=config.model_type, input_shape=input_shape)
    elif config.model_type == 'custom':
        return create_bfcnn_denoiser(
            input_shape=input_shape, num_blocks=config.num_blocks,
            filters=config.filters, initial_kernel_size=config.initial_kernel_size,
            kernel_size=config.kernel_size, activation=config.activation,
            model_name=f'bfcnn_custom_{config.experiment_name}'
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------

def train_bfcnn_denoiser(config: TrainingConfig) -> keras.Model:
    """Train a bias-free CNN denoiser and save clean inference model."""
    logger.info(f"Starting BFCNN training: {config.experiment_name}")

    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    for d in config.train_image_dirs:
        if not Path(d).exists():
            logger.warning(f"Training directory does not exist: {d}")
    for d in config.val_image_dirs:
        if not Path(d).exists():
            logger.warning(f"Validation directory does not exist: {d}")

    # Count files
    try:
        train_file_count = count_available_files(config.train_image_dirs, config.image_extensions, config.max_train_files)
        val_file_count = count_available_files(config.val_image_dirs, config.image_extensions, config.max_val_files)
    except Exception as e:
        logger.warning(f"Error counting files: {e}")
        train_file_count, val_file_count = 1000, 100

    if train_file_count == 0:
        raise ValueError("No training files found!")
    if val_file_count == 0:
        raise ValueError("No validation files found!")
    logger.info(f"Found ~{train_file_count} training, ~{val_file_count} validation files")

    # Create datasets
    train_dataset = create_dataset(config.train_image_dirs, config, is_training=True)
    val_dataset = create_dataset(config.val_image_dirs, config, is_training=False)

    # Steps per epoch
    if config.steps_per_epoch is not None:
        steps_per_epoch = config.steps_per_epoch
    else:
        steps_per_epoch = max(100, (train_file_count * config.patches_per_image) // config.batch_size)
    logger.info(f"Using {steps_per_epoch} steps per epoch")

    # Create model
    input_shape = (config.patch_size, config.patch_size, config.channels)
    model = create_model_instance(config, input_shape)
    model.summary()

    # Build optimizer with LR schedule
    lr_schedule = learning_rate_schedule_builder({
        'type': config.lr_schedule_type, 'learning_rate': config.learning_rate,
        'decay_steps': steps_per_epoch * config.epochs,
        'warmup_steps': steps_per_epoch * config.warmup_epochs, 'alpha': 0.01
    })
    optimizer = optimizer_builder({
        'type': config.optimizer_type,
        'gradient_clipping_by_norm': config.gradient_clipping
    }, lr_schedule)

    model.compile(
        optimizer=optimizer, loss='mse',
        metrics=['mae', keras.metrics.RootMeanSquaredError(name='rmse'), psnr_metric]
    )
    logger.info(f"Model compiled with {model.count_params():,} parameters")

    # Train
    callbacks = create_callbacks(config, config.val_image_dirs)
    start_time = time.time()
    validation_steps = config.validation_steps or max(50, steps_per_epoch // 20)

    history = model.fit(
        train_dataset, epochs=config.epochs, steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset, validation_steps=validation_steps,
        callbacks=callbacks, verbose=1
    )
    logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")

    # Save clean inference model with flexible spatial dims
    try:
        inference_model = create_model_instance(config, (None, None, config.channels))
        inference_model.set_weights(model.get_weights())
        inference_model_path = output_dir / "inference_model.keras"
        inference_model.save(inference_model_path)
        logger.info(f"Clean inference model saved to: {inference_model_path}")
        del inference_model
    except Exception as e:
        logger.error(f"Failed to save clean inference model: {e}")

    # Save training history
    try:
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(history_dict, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save training history: {e}")

    gc.collect()
    return model


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train Bias-Free CNN Denoiser',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model-type', choices=['tiny', 'small', 'base', 'large', 'xlarge', 'custom'], default='tiny')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--channels', type=int, choices=[1, 3], default=1)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--experiment-name', type=str, default=None)
    parser.add_argument('--patches-per-image', type=int, default=4)
    parser.add_argument('--monitor-every', type=int, default=5)
    parser.add_argument('--early-stopping-patience', type=int, default=15)
    parser.add_argument('--gpu', type=int, default=None, help='GPU device index')
    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_gpu(gpu_id=args.gpu)

    config = TrainingConfig(
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
        patch_size=args.patch_size, channels=args.channels,
        batch_size=args.batch_size, epochs=args.epochs,
        patches_per_image=args.patches_per_image,
        max_train_files=10000, max_val_files=1000,
        parallel_reads=8, dataset_shuffle_buffer=1013,
        model_type=args.model_type,
        noise_sigma_min=0.0, noise_sigma_max=0.5, noise_distribution='uniform',
        learning_rate=1e-3, optimizer_type='adamw', lr_schedule_type='cosine_decay',
        warmup_epochs=5,
        monitor_every_n_epochs=args.monitor_every,
        save_training_images=True, validation_steps=200,
        output_dir=args.output_dir, experiment_name=args.experiment_name
    )

    logger.info(f"Config: model={config.model_type}, epochs={config.epochs}, batch={config.batch_size}, "
                f"lr={config.learning_rate}, patch={config.patch_size}x{config.channels}, "
                f"noise=[{config.noise_sigma_min}, {config.noise_sigma_max}]")

    try:
        model = train_bfcnn_denoiser(config)
        logger.info("Training completed successfully!")
        model.summary()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
