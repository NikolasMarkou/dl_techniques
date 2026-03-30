"""BFU-Net denoiser training with deep supervision and implicit prior synthesis."""

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

from train.common import setup_gpu, create_callbacks as create_common_callbacks
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


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for bias-free U-Net denoiser training with deep supervision."""

    # Data
    train_image_dirs: List[str] = field(default_factory=list)
    val_image_dirs: List[str] = field(default_factory=list)
    patch_size: int = 64
    channels: int = 3
    image_extensions: List[str] = field(
        default_factory=lambda: ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp']
    )

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
    depth: int = 3
    blocks_per_level: int = 2
    filters: int = 64
    kernel_size: int = 3
    activation: str = 'relu'

    # Deep Supervision
    enable_deep_supervision: bool = True
    deep_supervision_schedule_type: str = 'linear_low_to_high'
    deep_supervision_schedule_config: Dict[str, Any] = field(default_factory=dict)

    # Training
    batch_size: int = 32
    epochs: int = 50
    patches_per_image: int = 16
    augment_data: bool = True
    normalize_input: bool = True
    steps_per_epoch: Optional[int] = None

    # Optimization
    learning_rate: float = 1e-3
    optimizer_type: str = 'adam'
    lr_schedule_type: str = 'cosine_decay'
    warmup_epochs: int = 2
    weight_decay: float = 1e-5
    gradient_clipping: float = 1.0

    # Monitoring
    monitor_every_n_epochs: int = 2
    save_best_only: bool = True
    early_stopping_patience: int = 15
    validation_steps: Optional[int] = 100

    # Image Synthesis
    enable_synthesis: bool = False
    synthesis_samples: int = 10
    synthesis_steps: int = 200
    synthesis_initial_step_size: float = 0.05
    synthesis_final_step_size: float = 0.8
    synthesis_initial_noise: float = 0.4
    synthesis_final_noise: float = 0.005

    # Output
    output_dir: str = 'results'
    experiment_name: Optional[str] = None
    save_training_images: bool = True
    save_model_checkpoints: bool = True

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ds_suffix = '_ds' if self.enable_deep_supervision else ''
            self.experiment_name = f"bfunet_{self.model_type}{ds_suffix}_{timestamp}"

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


# ---------------------------------------------------------------------
# DATASET PIPELINE
# ---------------------------------------------------------------------

def load_and_preprocess_image(image_path: tf.Tensor, config: TrainingConfig) -> tf.Tensor:
    """Load, normalize, and extract a random patch from an image."""
    try:
        image_string = tf.io.read_file(image_path)
        image = tf.image.decode_image(image_string, channels=config.channels, expand_animations=False)
        image.set_shape([None, None, config.channels])
        image = tf.cast(image, tf.float32)
        if config.normalize_input:
            image = image / 255.0

        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        min_dim = tf.minimum(height, width)
        min_size = config.patch_size

        image = tf.cond(
            tf.logical_or(height < min_size, width < min_size),
            true_fn=lambda: tf.image.resize(image, [
                tf.cast(tf.math.ceil(tf.cast(height, tf.float32) * tf.cast(min_size, tf.float32) / tf.cast(min_dim, tf.float32)), tf.int32),
                tf.cast(tf.math.ceil(tf.cast(width, tf.float32) * tf.cast(min_size, tf.float32) / tf.cast(min_dim, tf.float32)), tf.int32)
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
    """Add Gaussian noise with configurable distribution to a clean [0,1] patch."""
    if config.noise_distribution == 'uniform':
        noise_level = tf.random.uniform([], config.noise_sigma_min, config.noise_sigma_max)
    elif config.noise_distribution == 'log_uniform':
        log_min = tf.math.log(tf.maximum(config.noise_sigma_min, 1e-6))
        log_max = tf.math.log(config.noise_sigma_max)
        noise_level = tf.exp(tf.random.uniform([], log_min, log_max))
    else:
        raise ValueError(f"Unknown noise distribution: {config.noise_distribution}")

    noisy_patch = patch + tf.random.normal(tf.shape(patch)) * noise_level
    return tf.clip_by_value(noisy_patch, 0.0, 1.0), patch


def create_dataset(
    directories: List[str], config: TrainingConfig, is_training: bool = True
) -> tf.data.Dataset:
    """Create tf.data.Dataset of noisy/clean pairs from image directories."""
    logger.info(f"Creating {'training' if is_training else 'validation'} dataset from {len(directories)} directories")

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
    logger.info(f"Found {len(all_file_paths)} total files")

    limit = config.max_train_files if is_training else config.max_val_files
    if limit and limit < len(all_file_paths):
        logger.info(f"Limiting to {limit} files per configuration")
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
# LOSS FUNCTIONS FOR DEEP SUPERVISION
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ScaledMseLoss(keras.losses.Loss):
    """MSE loss with automatic target resizing for multi-scale supervision."""

    def __init__(self, name: str = "scaled_mse_loss", **kwargs) -> None:
        super().__init__(name=name, **kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        pred_shape = tf.shape(y_pred)
        y_true_resized = tf.image.resize(y_true, (pred_shape[1], pred_shape[2]))
        return tf.reduce_mean(tf.square(y_pred - y_true_resized))


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

class PrimaryOutputPSNR(keras.metrics.Metric):
    """PSNR metric evaluating only the primary output for multi-output models."""

    def __init__(self, name: str = 'primary_psnr', **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.psnr_sum = self.add_weight(name='psnr_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(
        self, y_true: Union[tf.Tensor, List[tf.Tensor]],
        y_pred: Union[tf.Tensor, List[tf.Tensor]],
        sample_weight: Optional[tf.Tensor] = None
    ) -> None:
        if isinstance(y_pred, list):
            primary_pred, primary_true = y_pred[0], y_true[0]
        else:
            primary_pred, primary_true = y_pred, y_true

        psnr_batch = tf.image.psnr(primary_pred, primary_true, max_val=1.0)
        self.psnr_sum.assign_add(tf.reduce_sum(psnr_batch))
        self.count.assign_add(tf.cast(tf.size(psnr_batch), tf.float32))

    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self.psnr_sum, self.count)

    def reset_state(self) -> None:
        self.psnr_sum.assign(0.0)
        self.count.assign(0.0)


# ---------------------------------------------------------------------
# IMAGE SYNTHESIS (IMPLICIT NEURAL PRIORS)
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
    """Generate images via stochastic coarse-to-fine sampling using the denoiser's implicit prior.

    Implements Kadkhodaie & Simoncelli's unconditional sampling algorithm.
    """
    if seed is not None:
        tf.random.set_seed(seed)

    logger.info(f"Starting unconditional sampling: {num_samples} samples, {num_steps} steps")

    y = tf.clip_by_value(
        tf.random.normal([num_samples] + list(image_shape), mean=0.5, stddev=0.3),
        0.0, 1.0
    )
    intermediate_steps = []
    step_sizes = tf.linspace(initial_step_size, final_step_size, num_steps)
    noise_levels = tf.linspace(initial_noise_level, final_noise_level, num_steps)

    for t in range(num_steps):
        h_t, gamma_t = step_sizes[t], noise_levels[t]

        model_output = denoiser(y, training=False)
        denoised = model_output[0] if isinstance(model_output, list) else model_output
        d_t = denoised - y

        y = tf.clip_by_value(y + h_t * d_t + gamma_t * tf.random.normal(tf.shape(y)), 0.0, 1.0)

        if save_intermediate and (t % (num_steps // 10) == 0 or t == num_steps - 1):
            intermediate_steps.append(y.numpy().copy())

        if t % (num_steps // 5) == 0:
            logger.info(
                f"Step {t}/{num_steps}: mean={tf.reduce_mean(y):.3f}, "
                f"std={tf.math.reduce_std(y):.3f}, h_t={h_t:.4f}, gamma_t={gamma_t:.4f}"
            )

    logger.info(f"Synthesis completed: {num_samples} samples generated")
    return y, intermediate_steps


def visualize_synthesis_process(
    final_samples: tf.Tensor, intermediate_steps: List[tf.Tensor],
    save_path: Path, epoch: int
) -> None:
    """Create multi-panel visualization of the synthesis evolution."""
    try:
        num_samples = min(4, final_samples.shape[0])
        num_steps = len(intermediate_steps)

        fig, axes = plt.subplots(num_samples, num_steps, figsize=(3 * num_steps, 3 * num_samples))
        fig.suptitle(f'Image Synthesis Evolution - Epoch {epoch}', fontsize=16, y=0.98)

        if num_samples == 1:
            axes = axes.reshape(1, -1)
        if num_steps == 1:
            axes = axes.reshape(-1, 1)

        for si in range(num_samples):
            for ti, step_data in enumerate(intermediate_steps):
                img = np.clip(step_data[si], 0.0, 1.0)
                cmap = 'gray' if img.shape[-1] == 1 else None
                if img.shape[-1] == 1:
                    img = img.squeeze(-1)

                axes[si, ti].imshow(img, cmap=cmap, vmin=0, vmax=1)
                axes[si, ti].axis('off')
                if si == 0:
                    step_num = ti * (200 // (num_steps - 1)) if ti < num_steps - 1 else 200
                    axes[si, ti].set_title(f'Step {step_num}', fontsize=10)
                if ti == 0:
                    axes[si, ti].set_ylabel(f'Sample {si + 1}', fontsize=12, rotation=0, ha='right', va='center')

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
    """Real-time visualization of training/validation metrics for single and multi-output models."""

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.visualization_dir = self.output_dir / "visualization_plots"
        self.visualization_dir.mkdir(parents=True, exist_ok=True)

        self.train_metrics: Dict[str, List[float]] = {
            'loss': [], 'mae': [], 'rmse': [], 'primary_psnr': [],
            'final_output_mae': [], 'final_output_rmse': [], 'final_output_primary_psnr': []
        }
        self.val_metrics: Dict[str, List[float]] = {
            'val_loss': [], 'val_mae': [], 'val_rmse': [], 'val_primary_psnr': [],
            'val_final_output_mae': [], 'val_final_output_rmse': [], 'val_final_output_primary_psnr': []
        }
        self.available_metrics: set = set()

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if logs:
            self.available_metrics.update(logs.keys())

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if logs:
            self.available_metrics.update(logs.keys())

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if logs is None:
            logs = {}
        self.available_metrics.update(logs.keys())

        for metric_name, metric_value in logs.items():
            try:
                val = float(metric_value)
                if metric_name in self.train_metrics:
                    self.train_metrics[metric_name].append(val)
                elif metric_name in self.val_metrics:
                    self.val_metrics[metric_name].append(val)
                elif metric_name.startswith('val_'):
                    expected_key = metric_name
                    if expected_key in self.val_metrics:
                        self.val_metrics[expected_key].append(val)
            except (ValueError, TypeError):
                pass

        if (epoch + 1) % 5 == 0 or epoch == 0:
            self._create_metrics_plots(epoch + 1)

    def _get_metric_data(self, metric_type: str, metrics_dict: Dict[str, List[float]]) -> Tuple[List[float], str]:
        """Get metric data with fallback for multi-output naming conventions."""
        prefixed = f'final_output_{metric_type}'
        if prefixed in metrics_dict and metrics_dict[prefixed]:
            return metrics_dict[prefixed], prefixed
        if metric_type in metrics_dict and metrics_dict[metric_type]:
            return metrics_dict[metric_type], metric_type
        return [], f"{metric_type} (not found)"

    def _create_metrics_plots(self, epoch: int) -> None:
        try:
            if not self.train_metrics.get('loss', []):
                return

            num_epochs = len(self.train_metrics['loss'])
            epochs_range = range(1, num_epochs + 1)
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Training and Validation Metrics - Epoch {epoch}', fontsize=16)

            def safe_plot(ax, train_type, val_type, title, ylabel, train_label, val_label):
                plots_added = False
                train_data, train_name = self._get_metric_data(train_type, self.train_metrics)
                if train_data and len(train_data) == num_epochs:
                    ax.plot(epochs_range, train_data, 'b-', label=f'{train_label} ({train_name})', linewidth=2)
                    plots_added = True
                val_data, val_name = self._get_metric_data(val_type, self.val_metrics)
                if val_data and len(val_data) == num_epochs:
                    ax.plot(epochs_range, val_data, 'r-', label=f'{val_label} ({val_name})', linewidth=2)
                    plots_added = True
                ax.set_title(title)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(ylabel)
                if plots_added:
                    ax.legend()
                ax.grid(True, alpha=0.3)

            # MSE Loss (always uses 'loss' key)
            axes[0, 0].set_title('Mean Squared Error (MSE)')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('MSE')
            if self.train_metrics['loss'] and len(self.train_metrics['loss']) == num_epochs:
                axes[0, 0].plot(epochs_range, self.train_metrics['loss'], 'b-', label='Training MSE', linewidth=2)
            if self.val_metrics.get('val_loss', []) and len(self.val_metrics['val_loss']) == num_epochs:
                axes[0, 0].plot(epochs_range, self.val_metrics['val_loss'], 'r-', label='Validation MSE', linewidth=2)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            safe_plot(axes[0, 1], 'mae', 'val_mae', 'MAE', 'MAE', 'Training MAE', 'Validation MAE')
            safe_plot(axes[1, 0], 'rmse', 'val_rmse', 'RMSE', 'RMSE', 'Training RMSE', 'Validation RMSE')
            safe_plot(axes[1, 1], 'primary_psnr', 'val_primary_psnr', 'PSNR', 'PSNR (dB)', 'Training PSNR', 'Validation PSNR')

            plt.tight_layout()
            plt.savefig(self.visualization_dir / f"epoch_{epoch:03d}_metrics.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            plt.clf()
            gc.collect()

            metrics_data = {
                'epoch': epoch,
                'train_metrics': self.train_metrics,
                'val_metrics': self.val_metrics,
                'diagnostics': {
                    'expected_epochs': num_epochs,
                    'available_train_metrics': {k: len(v) for k, v in self.train_metrics.items() if v},
                    'available_val_metrics': {k: len(v) for k, v in self.val_metrics.items() if v},
                    'all_available_metrics': sorted(list(self.available_metrics))
                }
            }
            with open(self.visualization_dir / "latest_metrics.json", 'w') as f:
                json.dump(metrics_data, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

        except Exception as e:
            logger.warning(f"Failed to create metrics plots: {e}")


class StreamingResultMonitor(keras.callbacks.Callback):
    """Memory-efficient monitor for denoising results and optional image synthesis."""

    def __init__(self, config: TrainingConfig, val_directories: List[str]) -> None:
        super().__init__()
        self.config = config
        self.monitor_freq = config.monitor_every_n_epochs
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.results_dir = self.output_dir / "visualization_plots"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._create_monitor_dataset(val_directories)

    def _create_monitor_dataset(self, val_directories: List[str]) -> None:
        """Create small consistent dataset (up to 10 images) for monitoring."""
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
                logger.error(f"Error collecting monitor files from {directory}: {e}")

        if not monitor_files:
            logger.warning("No files found for monitoring")
            self.test_batch = None
            return

        try:
            clean_patches = [load_and_preprocess_image(tf.constant(fp), self.config) for fp in monitor_files]
            self.test_batch = tf.stack(clean_patches)
            logger.info(f"Created monitoring dataset with shape: {self.test_batch.shape}")
        except Exception as e:
            logger.error(f"Failed to create monitor dataset: {e}")
            self.test_batch = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if (epoch + 1) % self.monitor_freq != 0 or self.test_batch is None:
            return

        def _monitor_and_save(epoch_numpy: np.ndarray) -> tf.constant:
            epoch_val = int(epoch_numpy)
            logger.info(f"Saving intermediate results for epoch {epoch_val}")
            try:
                noisy_images, clean_images = add_noise_to_patch(self.test_batch, self.config)
                model_output = self.model(noisy_images, training=False)
                denoised_images = model_output[0] if isinstance(model_output, list) else model_output

                if self.config.save_training_images:
                    self._save_image_samples(epoch_val, noisy_images, clean_images, denoised_images)

                mse_loss = tf.reduce_mean(tf.square(denoised_images - clean_images))
                psnr = tf.reduce_mean(tf.image.psnr(denoised_images, clean_images, max_val=1.0))
                logger.info(f"Epoch {epoch_val} - Validation MSE: {mse_loss.numpy():.6f}, PSNR: {psnr.numpy():.2f} dB")

                # Image synthesis using implicit prior
                if self.config.enable_synthesis:
                    logger.info("Generating synthetic images using implicit prior...")
                    try:
                        synthesis_shape = (self.config.patch_size, self.config.patch_size, self.config.channels)
                        generated_samples, intermediate_steps = unconditional_sampling(
                            denoiser=self.model, num_samples=self.config.synthesis_samples,
                            image_shape=synthesis_shape, num_steps=self.config.synthesis_steps,
                            initial_step_size=self.config.synthesis_initial_step_size,
                            final_step_size=self.config.synthesis_final_step_size,
                            initial_noise_level=self.config.synthesis_initial_noise,
                            final_noise_level=self.config.synthesis_final_noise,
                            seed=epoch_val, save_intermediate=True
                        )
                        visualize_synthesis_process(
                            generated_samples, intermediate_steps,
                            self.results_dir / f"epoch_{epoch_val:03d}_synthesis.png", epoch_val
                        )

                        gen_mean = tf.reduce_mean(generated_samples)
                        gen_std = tf.math.reduce_std(generated_samples)
                        gen_range = tf.reduce_max(generated_samples) - tf.reduce_min(generated_samples)
                        logger.info(f"Generated images - Mean: {gen_mean:.3f}, Std: {gen_std:.3f}, Range: {gen_range:.3f}")

                        synthesis_metrics = {
                            'epoch': epoch_val, 'synthesis_mean': float(gen_mean),
                            'synthesis_std': float(gen_std), 'synthesis_range': float(gen_range),
                            'num_synthesis_steps': self.config.synthesis_steps,
                            'num_samples': self.config.synthesis_samples,
                            'timestamp': datetime.now().isoformat()
                        }
                        with open(self.results_dir / f"epoch_{epoch_val:03d}_synthesis_metrics.json", 'w') as f:
                            json.dump(synthesis_metrics, f, indent=2)
                    except Exception as synthesis_error:
                        logger.warning(f"Image synthesis failed at epoch {epoch_val}: {synthesis_error}")

                metrics = {
                    'epoch': epoch_val, 'val_mse': float(mse_loss),
                    'val_psnr': float(psnr), 'timestamp': datetime.now().isoformat()
                }
                with open(self.results_dir / f"epoch_{epoch_val:03d}_metrics.json", 'w') as f:
                    json.dump(metrics, f, indent=2)

                del noisy_images, clean_images, denoised_images
                gc.collect()
            except Exception as e:
                tf.print(f"Error during monitoring at epoch {epoch_val}: {e}")
            return tf.constant(0, dtype=tf.int32)

        tf.py_function(func=_monitor_and_save, inp=[epoch + 1], Tout=[tf.int32])

    def _save_image_samples(self, epoch: int, noisy: tf.Tensor, clean: tf.Tensor, denoised: tf.Tensor) -> None:
        """Save noisy/clean/denoised comparison grid."""
        try:
            num_samples = min(10, noisy.shape[0])
            fig, axes = plt.subplots(3, 10, figsize=(25, 7.5))
            fig.suptitle(f'Denoising Results - Epoch {epoch}', fontsize=20, y=0.98)

            for i in range(10):
                if i < num_samples:
                    clean_img = np.clip(clean[i].numpy(), 0.0, 1.0)
                    noisy_img = np.clip(noisy[i].numpy(), 0.0, 1.0)
                    denoised_img = np.clip(denoised[i].numpy(), 0.0, 1.0)

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


class DeepSupervisionWeightScheduler(keras.callbacks.Callback):
    """Dynamically adjusts deep supervision loss weights during training."""

    def __init__(self, config: TrainingConfig, num_outputs: int) -> None:
        super().__init__()
        self.config = config
        self.num_outputs = num_outputs
        self.total_epochs = config.epochs

        ds_config = {'type': config.deep_supervision_schedule_type, 'config': config.deep_supervision_schedule_config}
        self.scheduler = deep_supervision_schedule_builder(ds_config, self.num_outputs, invert_order=False)

        logger.info("Expected scheduler: ")
        for progress in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
            logger.info(f"  -  at {progress} is {self.scheduler(progress)}")

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        progress = min(1.0, epoch / max(1, self.total_epochs - 1))
        new_weights = self.scheduler(progress)
        self.model.loss_weights = new_weights
        weights_str = ", ".join([f"{w:.4f}" for w in new_weights])
        logger.info(f"Epoch {epoch + 1}/{self.total_epochs} - Updated DS weights: [{weights_str}]")


def create_callbacks(
    config: TrainingConfig, val_directories: List[str], num_outputs: int
) -> List[keras.callbacks.Callback]:
    """Create training callbacks: common (checkpoint, early stop, CSV, analyzer) + domain-specific."""
    callbacks, _ = create_common_callbacks(
        model_name=config.experiment_name or config.model_type,
        results_dir_prefix="bfunet",
        monitor="val_loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
    )

    if config.enable_deep_supervision and num_outputs > 1:
        callbacks.append(DeepSupervisionWeightScheduler(config, num_outputs))
    callbacks.append(MetricsVisualizationCallback(config))
    callbacks.append(StreamingResultMonitor(config, val_directories))

    return callbacks


# ---------------------------------------------------------------------
# MODEL CREATION
# ---------------------------------------------------------------------

def create_model_instance(config: TrainingConfig, input_shape: Tuple[int, int, int]) -> keras.Model:
    """Create a BFU-Net model from config (variant or custom)."""
    if config.model_type in BFUNET_CONFIGS:
        return create_bfunet_variant(
            variant=config.model_type, input_shape=input_shape,
            enable_deep_supervision=config.enable_deep_supervision
        )
    elif config.model_type == 'custom':
        return create_bfunet_denoiser(
            input_shape=input_shape, depth=config.depth,
            initial_filters=config.filters, blocks_per_level=config.blocks_per_level,
            kernel_size=config.kernel_size, activation=config.activation,
            enable_deep_supervision=config.enable_deep_supervision,
            model_name=f'bfunet_custom_{config.experiment_name}'
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------

def train_bfunet_denoiser(config: TrainingConfig) -> keras.Model:
    """Full training pipeline for BFU-Net denoisers with deep supervision support."""
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Deep Supervision: {'ENABLED' if config.enable_deep_supervision else 'DISABLED'}")
    if config.enable_deep_supervision:
        logger.info(f"  - Schedule: {config.deep_supervision_schedule_type}")

    # Setup and save config
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    # Validate directories
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

    # Create model and analyze outputs
    input_shape = (config.patch_size, config.patch_size, config.channels)
    model = create_model_instance(config, input_shape)
    model.summary()

    has_multiple_outputs = isinstance(model.output, list)
    num_outputs = len(model.output) if has_multiple_outputs else 1
    logger.info(f"Model created with {num_outputs} output(s)")

    # Create datasets
    train_dataset = create_dataset(config.train_image_dirs, config, is_training=True)
    val_dataset = create_dataset(config.val_image_dirs, config, is_training=False)

    # Adapt dataset for multi-output models (multi-scale ground truth)
    if has_multiple_outputs:
        concrete_output_dims = [(out.shape[1], out.shape[2]) for out in model.output]
        logger.info(f"Adapting dataset for multi-output model with dimensions: {concrete_output_dims}")

        def create_multiscale_labels(noisy_patch: tf.Tensor, clean_patch: tf.Tensor):
            labels = [tf.image.resize(clean_patch, dim) for dim in concrete_output_dims]
            return noisy_patch, tuple(labels)

        train_dataset = train_dataset.map(create_multiscale_labels, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(create_multiscale_labels, num_parallel_calls=tf.data.AUTOTUNE)

    # Steps per epoch
    if config.steps_per_epoch is not None:
        steps_per_epoch = config.steps_per_epoch
    else:
        steps_per_epoch = max(100, (train_file_count * config.patches_per_image) // config.batch_size)
    logger.info(f"Calculated {steps_per_epoch} steps per epoch")

    # Build optimizer with LR schedule
    lr_schedule = learning_rate_schedule_builder({
        'type': config.lr_schedule_type, 'learning_rate': config.learning_rate,
        'decay_steps': steps_per_epoch * config.epochs,
        'warmup_steps': steps_per_epoch * config.warmup_epochs, 'alpha': 0.01
    })
    optimizer = optimizer_builder({
        'type': config.optimizer_type, 'gradient_clipping_by_norm': config.gradient_clipping
    }, lr_schedule)

    # Loss and metrics configuration
    if has_multiple_outputs:
        loss_fns = ['mse'] * num_outputs
        initial_weights = [1.0 / num_outputs] * num_outputs
        metrics = {'final_output': [
            'mae', keras.metrics.RootMeanSquaredError(name='rmse'), PrimaryOutputPSNR()
        ]}
    else:
        loss_fns = 'mse'
        initial_weights = None
        metrics = ['mae', keras.metrics.RootMeanSquaredError(name='rmse'), PrimaryOutputPSNR()]

    model.compile(optimizer=optimizer, loss=loss_fns, loss_weights=initial_weights, metrics=metrics)
    logger.info("Model compiled successfully")

    # Train
    callbacks = create_callbacks(config, config.val_image_dirs, num_outputs)
    start_time = time.time()
    validation_steps = config.validation_steps or max(50, steps_per_epoch // 20)

    history = model.fit(
        train_dataset, epochs=config.epochs, steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset, validation_steps=validation_steps,
        callbacks=callbacks, verbose=1
    )
    logger.info(f"Training completed in {(time.time() - start_time) / 3600:.2f} hours")

    # Save clean inference model
    try:
        inference_input_shape = (None, None, config.channels)
        if config.model_type in BFUNET_CONFIGS:
            inference_model_full = create_bfunet_variant(
                variant=config.model_type, input_shape=inference_input_shape,
                enable_deep_supervision=config.enable_deep_supervision
            )
        else:
            inference_model_full = create_bfunet_denoiser(
                input_shape=inference_input_shape, depth=config.depth,
                initial_filters=config.filters, blocks_per_level=config.blocks_per_level,
                kernel_size=config.kernel_size, activation=config.activation,
                enable_deep_supervision=config.enable_deep_supervision,
                model_name=f'bfunet_custom_{config.experiment_name}_inference'
            )

        inference_model_full.set_weights(model.get_weights())
        logger.info("Successfully transferred weights to inference model")

        if config.enable_deep_supervision and isinstance(inference_model_full.output, list):
            inference_model = keras.Model(
                inputs=inference_model_full.input, outputs=inference_model_full.output[0],
                name=f"{inference_model_full.name}_single_output"
            )
            del inference_model_full
        else:
            inference_model = inference_model_full

        inference_model_path = output_dir / "inference_model.keras"
        inference_model.save(inference_model_path)
        logger.info(f"Clean inference model saved to: {inference_model_path}")
        del inference_model
    except Exception as e:
        logger.error(f"Failed to save clean inference model: {e}", exc_info=True)
        raise

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
        description='Train Bias-Free U-Net Denoiser with Deep Supervision',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model-type', choices=['tiny', 'small', 'base', 'large', 'xlarge', 'custom'], default='tiny')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--channels', type=int, choices=[1, 3], default=1)

    # Deep supervision
    parser.add_argument('--enable-deep-supervision', action='store_true', default=True)
    parser.add_argument('--no-deep-supervision', dest='enable_deep_supervision', action='store_false')
    parser.add_argument('--deep-supervision-schedule', choices=[
        'constant_equal', 'constant_low_to_high', 'constant_high_to_low',
        'linear_low_to_high', 'non_linear_low_to_high', 'custom_sigmoid_low_to_high',
        'scale_by_scale_low_to_high', 'cosine_annealing', 'curriculum', 'step_wise'
    ], default='step_wise')

    # Output and monitoring
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--experiment-name', type=str, default=None)
    parser.add_argument('--patches-per-image', type=int, default=4)
    parser.add_argument('--monitor-every', type=int, default=1)
    parser.add_argument('--early-stopping-patience', type=int, default=15)
    parser.add_argument('--max-train-files', type=int, default=None)

    # Image synthesis
    parser.add_argument('--enable-synthesis', action='store_true', default=True)
    parser.add_argument('--no-synthesis', dest='enable_synthesis', action='store_false')
    parser.add_argument('--synthesis-samples', type=int, default=4)
    parser.add_argument('--synthesis-steps', type=int, default=200)

    # GPU
    parser.add_argument('--gpu', type=int, default=None, help='GPU device index')

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    setup_gpu(gpu_id=args.gpu)

    config = TrainingConfig(
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
        patch_size=args.patch_size, channels=args.channels,
        batch_size=args.batch_size, epochs=args.epochs,
        patches_per_image=args.patches_per_image,
        max_train_files=args.max_train_files, max_val_files=10000,
        parallel_reads=8, dataset_shuffle_buffer=1013,
        model_type=args.model_type,
        enable_deep_supervision=args.enable_deep_supervision,
        deep_supervision_schedule_type=args.deep_supervision_schedule,
        deep_supervision_schedule_config={},
        noise_sigma_min=0.0, noise_sigma_max=0.5, noise_distribution='uniform',
        learning_rate=1e-3, optimizer_type='adamw', lr_schedule_type='cosine_decay',
        warmup_epochs=5,
        monitor_every_n_epochs=args.monitor_every, save_training_images=True,
        validation_steps=500,
        enable_synthesis=args.enable_synthesis,
        synthesis_samples=args.synthesis_samples, synthesis_steps=args.synthesis_steps,
        output_dir=args.output_dir, experiment_name=args.experiment_name
    )

    logger.info(f"Config: model={config.model_type}, DS={'on' if config.enable_deep_supervision else 'off'}, "
                f"epochs={config.epochs}, batch={config.batch_size}, lr={config.learning_rate}, "
                f"patch={config.patch_size}x{config.channels}, noise=[{config.noise_sigma_min}, {config.noise_sigma_max}], "
                f"synthesis={'on' if config.enable_synthesis else 'off'}")

    try:
        model = train_bfunet_denoiser(config)
        logger.info("Training completed successfully!")
        model.summary()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
