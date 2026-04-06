"""BFCNN Denoiser Training using Vision Framework."""

import gc
import keras
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any

from dl_techniques.optimization.train_vision import (
    TrainingConfig, DatasetBuilder, TrainingPipeline, ModelBuilder,
)
from dl_techniques.utils.logger import logger
from dl_techniques.models.bias_free_denoisers.bfcnn import (
    create_bfcnn_denoiser, BFCNN_CONFIGS, create_bfcnn_variant,
)
from dl_techniques.analyzer import DataInput
from train.common import setup_gpu
from dl_techniques.metrics.psnr_metric import PsnrMetric


@dataclass
class DenoisingConfig(TrainingConfig):
    """Extended configuration for denoising tasks."""
    train_image_dirs: List[str] = field(default_factory=list)
    val_image_dirs: List[str] = field(default_factory=list)
    patch_size: int = 128
    channels: int = 1
    image_extensions: List[str] = field(
        default_factory=lambda: ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp']
    )
    max_train_files: Optional[int] = None
    max_val_files: Optional[int] = None
    parallel_reads: int = 8
    dataset_shuffle_buffer: int = 1000
    noise_sigma_min: float = 0.0
    noise_sigma_max: float = 0.5
    noise_distribution: str = 'uniform'
    patches_per_image: int = 16
    augment_data: bool = True
    normalize_input: bool = True
    monitor_every_n_epochs: int = 5
    save_training_images: bool = True

    def __post_init__(self):
        self.input_shape = (self.patch_size, self.patch_size, self.channels)
        self.num_classes = self.channels
        super().__post_init__()

        if self.noise_sigma_min < 0 or self.noise_sigma_max <= self.noise_sigma_min:
            raise ValueError("Invalid noise sigma range")
        if not self.train_image_dirs:
            raise ValueError("No training directories specified")
        if not self.val_image_dirs:
            raise ValueError("No validation directories specified")
        if self.noise_distribution not in ['uniform', 'log_uniform']:
            raise ValueError(f"Invalid noise distribution: {self.noise_distribution}")


class DenoisingDatasetBuilder(DatasetBuilder):
    """Dataset builder for image denoising tasks."""

    def __init__(self, config: DenoisingConfig):
        super().__init__(config)
        self.config: DenoisingConfig = config

    def _load_and_preprocess_image(self, image_path: tf.Tensor) -> tf.Tensor:
        try:
            image_string = tf.io.read_file(image_path)
            image = tf.image.decode_image(image_string, channels=self.config.channels, expand_animations=False)
            image.set_shape([None, None, self.config.channels])
            image = tf.cast(image, tf.float32)

            if self.config.normalize_input:
                image = (image / 127.5) - 1.0
            else:
                image = image / 255.0

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
                true_fn=resize_if_small, false_fn=lambda: image
            )

            patch = tf.image.random_crop(
                image, [self.config.patch_size, self.config.patch_size, self.config.channels]
            )
            return patch

        except tf.errors.InvalidArgumentError:
            logger.warning(f"Failed to load image: {image_path}")
            return tf.zeros([self.config.patch_size, self.config.patch_size, self.config.channels], dtype=tf.float32)

    def _augment_patch(self, patch: tf.Tensor) -> tf.Tensor:
        patch = tf.image.random_flip_left_right(patch)
        patch = tf.image.random_flip_up_down(patch)
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        patch = tf.image.rot90(patch, k)
        return patch

    def _add_noise(self, patch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.config.noise_distribution == 'uniform':
            noise_level = tf.random.uniform([], self.config.noise_sigma_min, self.config.noise_sigma_max)
        else:
            log_min = tf.math.log(tf.maximum(self.config.noise_sigma_min, 1e-6))
            log_max = tf.math.log(self.config.noise_sigma_max)
            noise_level = tf.exp(tf.random.uniform([], log_min, log_max))

        noise = tf.random.normal(tf.shape(patch)) * noise_level
        noisy_patch = patch + noise

        if self.config.normalize_input:
            noisy_patch = tf.clip_by_value(noisy_patch, -1.0, 1.0)
        else:
            noisy_patch = tf.clip_by_value(noisy_patch, 0.0, 1.0)

        return noisy_patch, patch

    def _create_file_list(self, directories: List[str], limit: Optional[int] = None) -> List[str]:
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

        if not all_files:
            raise ValueError(f"No files found in directories: {directories}")

        logger.info(f"Found {len(all_files)} files")
        if limit and limit < len(all_files):
            np.random.shuffle(all_files)
            all_files = all_files[:limit]

        return all_files

    def build(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[int], Optional[int]]:
        logger.info("Building denoising datasets")

        train_files = self._create_file_list(self.config.train_image_dirs, self.config.max_train_files)
        val_files = self._create_file_list(self.config.val_image_dirs, self.config.max_val_files)

        train_ds = tf.data.Dataset.from_tensor_slices(train_files)
        train_ds = train_ds.shuffle(
            buffer_size=min(self.config.dataset_shuffle_buffer, len(train_files)),
            reshuffle_each_iteration=True
        )
        train_ds = train_ds.repeat()

        if self.config.patches_per_image > 1:
            train_ds = train_ds.flat_map(
                lambda path: tf.data.Dataset.from_tensors(path).repeat(self.config.patches_per_image)
            )

        train_ds = train_ds.map(self._load_and_preprocess_image, num_parallel_calls=self.config.parallel_reads)
        train_ds = train_ds.filter(lambda x: tf.reduce_sum(tf.abs(x)) > 0)
        if self.config.augment_data:
            train_ds = train_ds.map(self._augment_patch, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.map(self._add_noise, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.batch(self.config.batch_size, drop_remainder=True)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices(val_files)
        val_ds = val_ds.repeat()
        if self.config.patches_per_image > 1:
            val_ds = val_ds.flat_map(
                lambda path: tf.data.Dataset.from_tensors(path).repeat(self.config.patches_per_image)
            )
        val_ds = val_ds.map(self._load_and_preprocess_image, num_parallel_calls=self.config.parallel_reads)
        val_ds = val_ds.filter(lambda x: tf.reduce_sum(tf.abs(x)) > 0)
        val_ds = val_ds.map(self._add_noise, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.config.batch_size)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

        total_train_patches = len(train_files) * self.config.patches_per_image
        steps_per_epoch = max(100, total_train_patches // self.config.batch_size)
        val_steps = self.config.validation_steps or max(
            50, (len(val_files) * self.config.patches_per_image) // self.config.batch_size
        )

        logger.info(f"Dataset ready: {steps_per_epoch} train steps, {val_steps} val steps")
        return train_ds, val_ds, steps_per_epoch, val_steps

    def get_test_data(self) -> Optional[DataInput]:
        try:
            val_files = self._create_file_list(self.config.val_image_dirs, limit=100)
            clean_patches, noisy_patches = [], []

            for file_path in val_files[:50]:
                patch = self._load_and_preprocess_image(tf.constant(file_path))
                noisy, clean = self._add_noise(patch)
                clean_patches.append(clean.numpy())
                noisy_patches.append(noisy.numpy())

            return DataInput(x_data=np.array(noisy_patches), y_data=np.array(clean_patches))
        except Exception as e:
            logger.warning(f"Could not create test data: {e}")
            return None



# PsnrMetric imported from dl_techniques.metrics (max_val=2.0 for [-1, +1] range)


class DenoisingVisualizationCallback(keras.callbacks.Callback):
    """Visualize denoising results during training."""

    def __init__(self, config: DenoisingConfig, val_files: List[str]):
        super().__init__()
        self.config = config
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._create_test_batch(val_files)

    def _create_test_batch(self, val_files: List[str]):
        try:
            builder = DenoisingDatasetBuilder(self.config)
            clean_patches = []
            for file_path in val_files[:10]:
                patch = builder._load_and_preprocess_image(tf.constant(file_path))
                clean_patches.append(patch)
            self.test_batch = tf.stack(clean_patches)
        except Exception as e:
            logger.warning(f"Failed to create test batch: {e}")
            self.test_batch = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if self.test_batch is None or (epoch + 1) % self.config.monitor_every_n_epochs != 0:
            return

        try:
            builder = DenoisingDatasetBuilder(self.config)
            noisy, clean = builder._add_noise(self.test_batch)
            denoised = self.model(noisy, training=False)
            self._save_comparison(epoch + 1, noisy, clean, denoised)

            mse = tf.reduce_mean(tf.square(denoised - clean))
            psnr = tf.reduce_mean(tf.image.psnr(denoised, clean, max_val=2.0))
            logger.info(f"Epoch {epoch + 1} - Test MSE: {mse.numpy():.6f}, PSNR: {psnr.numpy():.2f} dB")
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")

    def _save_comparison(self, epoch: int, noisy: tf.Tensor, clean: tf.Tensor, denoised: tf.Tensor):
        num_samples = min(10, noisy.shape[0])
        fig, axes = plt.subplots(3, num_samples, figsize=(25, 7.5))
        fig.suptitle(f'Denoising Results - Epoch {epoch}', fontsize=20)

        for i in range(num_samples):
            clean_img = np.clip((clean[i].numpy() + 1.0) / 2.0, 0, 1)
            noisy_img = np.clip((noisy[i].numpy() + 1.0) / 2.0, 0, 1)
            denoised_img = np.clip((denoised[i].numpy() + 1.0) / 2.0, 0, 1)

            if clean_img.shape[-1] == 1:
                clean_img, noisy_img, denoised_img = clean_img.squeeze(-1), noisy_img.squeeze(-1), denoised_img.squeeze(-1)
                cmap = 'gray'
            else:
                cmap = None

            axes[0, i].imshow(clean_img, cmap=cmap, vmin=0, vmax=1)
            axes[0, i].axis('off')
            if i == 0: axes[0, i].set_ylabel('Clean', fontsize=12)

            axes[1, i].imshow(noisy_img, cmap=cmap, vmin=0, vmax=1)
            axes[1, i].axis('off')
            if i == 0: axes[1, i].set_ylabel('Noisy', fontsize=12)

            axes[2, i].imshow(denoised_img, cmap=cmap, vmin=0, vmax=1)
            axes[2, i].axis('off')
            if i == 0: axes[2, i].set_ylabel('Denoised', fontsize=12)

        plt.tight_layout()
        plt.savefig(self.output_dir / f"epoch_{epoch:03d}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()


class DenoisingTrainingPipeline(TrainingPipeline):
    """Extended pipeline for denoising tasks."""

    def __init__(self, config: DenoisingConfig):
        super().__init__(config)
        self.config: DenoisingConfig = config

    def _compile_model(self, model: keras.Model, total_steps: int) -> None:
        lr_schedule = self._create_lr_schedule(total_steps)
        optimizer = self._create_optimizer(lr_schedule)
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.MeanSquaredError(),
            metrics=[
                keras.metrics.MeanAbsoluteError(name='mae'),
                keras.metrics.RootMeanSquaredError(name='rmse'),
                PsnrMetric(max_val=2.0, name='psnr_metric')
            ]
        )

    def _create_callbacks(
            self, lr_schedule: Optional[Any] = None,
            custom_callbacks: Optional[List[keras.callbacks.Callback]] = None
    ) -> List[keras.callbacks.Callback]:
        callbacks = super()._create_callbacks(lr_schedule, custom_callbacks)
        if self.config.save_training_images:
            try:
                builder = DenoisingDatasetBuilder(self.config)
                val_files = builder._create_file_list(self.config.val_image_dirs, limit=100)
                callbacks.append(DenoisingVisualizationCallback(self.config, val_files))
            except Exception as e:
                logger.warning(f"Could not add visualization callback: {e}")
        return callbacks

    def run(
            self, model_builder: ModelBuilder, dataset_builder: DatasetBuilder,
            custom_callbacks: Optional[List[keras.callbacks.Callback]] = None
    ) -> Tuple[keras.Model, keras.callbacks.History]:
        model, history = super().run(model_builder, dataset_builder, custom_callbacks)
        self._save_inference_model(model, model_builder)
        return model, history

    def _save_inference_model(self, trained_model: keras.Model, model_builder: ModelBuilder):
        try:
            inference_config = DenoisingConfig(**self.config.__dict__)
            inference_config.input_shape = (None, None, self.config.channels)
            inference_model = model_builder(inference_config)
            inference_model.set_weights(trained_model.get_weights())

            inference_path = self.experiment_dir / "inference_model.keras"
            inference_model.save(inference_path)
            logger.info(f"Inference model saved: {inference_path} (flexible input: (None, None, {self.config.channels}))")

            del inference_model
            gc.collect()
        except Exception as e:
            logger.error(f"Failed to save inference model: {e}")


def build_bfcnn_denoiser(config: DenoisingConfig) -> keras.Model:
    """Build BFCNN denoiser model."""
    variant = config.model_args.get('variant', 'tiny')
    logger.info(f"Building BFCNN denoiser: {variant}")

    if variant in BFCNN_CONFIGS:
        model = create_bfcnn_variant(variant=variant, input_shape=config.input_shape)
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


def create_denoising_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train BFCNN denoiser using vision framework',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model-variant', type=str, default='tiny', choices=['tiny', 'small', 'base', 'large', 'xlarge', 'custom'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--channels', type=int, default=1, choices=[1, 3])
    parser.add_argument('--patches-per-image', type=int, default=16)
    parser.add_argument('--max-train-files', type=int, default=None)
    parser.add_argument('--max-val-files', type=int, default=None)
    parser.add_argument('--noise-min', type=float, default=0.0)
    parser.add_argument('--noise-max', type=float, default=0.5)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--lr-schedule', type=str, default='cosine')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--experiment-name', type=str, default=None)
    return parser


def main():
    parser = create_denoising_argument_parser()
    args = parser.parse_args()

    config = DenoisingConfig(
        train_image_dirs=['/media/arxwn/data0_4tb/datasets/Megadepth'],
        val_image_dirs=['/media/arxwn/data0_4tb/datasets/div2k/validation'],
        patch_size=args.patch_size, channels=args.channels,
        model_args={'variant': args.model_variant},
        epochs=args.epochs, batch_size=args.batch_size,
        patches_per_image=args.patches_per_image,
        max_train_files=args.max_train_files, max_val_files=args.max_val_files,
        parallel_reads=8,
        noise_sigma_min=args.noise_min, noise_sigma_max=args.noise_max,
        noise_distribution='uniform',
        learning_rate=args.learning_rate, optimizer_type=args.optimizer,
        lr_schedule_type=args.lr_schedule, weight_decay=1e-5, gradient_clipping=1.0,
        monitor_every_n_epochs=5, save_training_images=True,
        validation_steps=200, early_stopping_patience=15,
        output_dir=args.output_dir, experiment_name=args.experiment_name,
        enable_visualization=True, enable_analysis=False,
    )

    logger.info(f"BFCNN-{config.model_args['variant']}, patch={config.patch_size}x{config.patch_size}, "
                f"channels={config.channels}, noise=[{config.noise_sigma_min}, {config.noise_sigma_max}], "
                f"epochs={config.epochs}, batch={config.batch_size}, lr={config.learning_rate}")

    try:
        dataset_builder = DenoisingDatasetBuilder(config)
        pipeline = DenoisingTrainingPipeline(config)

        model, history = pipeline.run(model_builder=build_bfcnn_denoiser, dataset_builder=dataset_builder)
        logger.info(f"Training complete. Results: {pipeline.experiment_dir}")
        model.summary(print_fn=logger.info)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
