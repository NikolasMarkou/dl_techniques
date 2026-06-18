"""CliffordLaplacianUNet DIV2K autoencoder training script.

Trains :class:`CliffordLaplacianUNet` as a deterministic image autoencoder on
DIV2K. The reconstruction target is the **identity** (input == target): each
patch is normalized to ``[-1, +1]``, augmented, and fed to the model as both
input and target. The model emits a dict ``{"reconstruction": ...}`` and is
compiled with ``loss="mse"``; the tf.data pipeline therefore yields dict-keyed
targets ``(patch, {"reconstruction": patch})`` so Keras binds the loss to the
correct output (plan Assumption A4). No noise, no VAE, no KL, no sampling.

Results are written to the repo-root ``results/`` directory.

Usage::

    MPLBACKEND=Agg python -m train.cliffordnet.train_cliffordnet_autoencoder \\
        --variant small \\
        --epochs 100 \\
        --batch-size 16 \\
        --patch-size 256 \\
        --gpu 1
"""

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

from train.common import (
    setup_gpu,
    create_callbacks as create_common_callbacks,
    generate_training_curves,
    save_config_json,
    collect_image_paths,
    augment_patch,
)
from dl_techniques.metrics.psnr_metric import PsnrMetric
from dl_techniques.utils.logger import logger
from dl_techniques.utils.filesystem import count_available_files
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)
from dl_techniques.models.cliffordnet import (
    create_clifford_laplacian_unet,
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for CliffordLaplacianUNet DIV2K autoencoder training."""

    # Data
    train_image_dirs: List[str] = field(
        default_factory=lambda: ["/media/arxwn/data0_4tb/datasets/div2k/train"]
    )
    val_image_dirs: List[str] = field(
        default_factory=lambda: ["/media/arxwn/data0_4tb/datasets/div2k/validation"]
    )
    patch_size: int = 256
    channels: int = 3
    image_extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"]
    )

    # Memory
    max_train_files: Optional[int] = None
    max_val_files: Optional[int] = None
    parallel_reads: int = 8
    dataset_shuffle_buffer: int = 1000

    # Model
    model_variant: str = "small"

    # Training
    batch_size: int = 16
    epochs: int = 100
    patches_per_image: int = 16
    augment_data: bool = True
    steps_per_epoch: Optional[int] = None

    # Optimization
    learning_rate: float = 1e-3
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "cosine_decay"
    warmup_epochs: int = 5
    weight_decay: float = 1e-5
    gradient_clipping: float = 1.0

    # Monitoring
    monitor_every_n_epochs: int = 5
    save_best_only: bool = True
    early_stopping_patience: int = 15
    validation_steps: Optional[int] = 200

    # Output
    output_dir: str = "results"
    experiment_name: Optional[str] = None
    save_training_images: bool = True

    def __post_init__(self):
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = (
                f"cliffordnet_autoencoder_{self.model_variant}_{timestamp}"
            )
        if self.patch_size <= 0 or self.channels <= 0:
            raise ValueError("Invalid patch size or channel configuration")
        if not self.train_image_dirs:
            raise ValueError("No training directories specified")
        if not self.val_image_dirs:
            raise ValueError("No validation directories specified")


# ---------------------------------------------------------------------
# DATASET BUILDER
# ---------------------------------------------------------------------


def load_and_preprocess_image(
    image_path: tf.Tensor, config: TrainingConfig
) -> tf.Tensor:
    """Load, preprocess, and extract a random patch from an image."""
    try:
        image_string = tf.io.read_file(image_path)
        image = tf.image.decode_image(
            image_string, channels=config.channels, expand_animations=False
        )
        image.set_shape([None, None, config.channels])
        image = tf.cast(image, tf.float32)

        # Normalize to [-1, +1]
        image = (image / 127.5) - 1.0

        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        min_size = config.patch_size

        image = tf.cond(
            tf.logical_or(height < min_size, width < min_size),
            true_fn=lambda: tf.image.resize(
                image,
                [
                    tf.cast(
                        tf.math.ceil(
                            tf.cast(height, tf.float32)
                            * tf.cast(min_size, tf.float32)
                            / tf.cast(tf.minimum(height, width), tf.float32)
                        ),
                        tf.int32,
                    ),
                    tf.cast(
                        tf.math.ceil(
                            tf.cast(width, tf.float32)
                            * tf.cast(min_size, tf.float32)
                            / tf.cast(tf.minimum(height, width), tf.float32)
                        ),
                        tf.int32,
                    ),
                ],
            ),
            false_fn=lambda: image,
        )

        return tf.image.random_crop(
            image, [config.patch_size, config.patch_size, config.channels]
        )

    except tf.errors.InvalidArgumentError:
        logger.warning(f"Failed to load image: {image_path}")
        return tf.zeros(
            [config.patch_size, config.patch_size, config.channels],
            dtype=tf.float32,
        )


def make_identity_target(
    patch: tf.Tensor,
) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    """Map a clean patch to an identity autoencoder example.

    The model output is a dict ``{"reconstruction": ...}`` compiled with
    ``loss="mse"``; the target is therefore dict-keyed so Keras binds the loss
    to the reconstruction head (plan Assumption A4).
    """
    return patch, {"reconstruction": patch}


def create_dataset(
    directories: List[str], config: TrainingConfig, is_training: bool = True
) -> tf.data.Dataset:
    """Create a tf.data.Dataset of identity (input, target) pairs."""
    logger.info(
        f"Creating {'training' if is_training else 'validation'} dataset "
        f"from directories: {directories}"
    )

    limit = config.max_train_files if is_training else config.max_val_files
    all_file_paths = collect_image_paths(
        directories, extensions=config.image_extensions, max_files=limit
    )

    if not all_file_paths:
        raise ValueError(f"No image files found in directories: {directories}")

    dataset = tf.data.Dataset.from_tensor_slices(all_file_paths)

    if is_training:
        dataset = dataset.shuffle(
            buffer_size=min(config.dataset_shuffle_buffer, len(all_file_paths)),
            reshuffle_each_iteration=True,
        )
    dataset = dataset.repeat()

    if is_training and config.patches_per_image > 1:
        dataset = dataset.flat_map(
            lambda path: tf.data.Dataset.from_tensors(path).repeat(
                config.patches_per_image
            )
        )

    dataset = dataset.map(
        lambda path: load_and_preprocess_image(path, config),
        num_parallel_calls=config.parallel_reads,
    )
    dataset = dataset.filter(lambda x: tf.reduce_sum(tf.abs(x)) > 0)
    dataset = dataset.map(
        lambda x: tf.ensure_shape(
            x, [config.patch_size, config.patch_size, config.channels]
        )
    )

    if is_training and config.augment_data:
        dataset = dataset.map(augment_patch, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.map(
        make_identity_target,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.map(
        lambda inp, tgt: (
            tf.ensure_shape(
                inp, [config.patch_size, config.patch_size, config.channels]
            ),
            {
                "reconstruction": tf.ensure_shape(
                    tgt["reconstruction"],
                    [config.patch_size, config.patch_size, config.channels],
                )
            },
        )
    )

    dataset = dataset.prefetch(config.batch_size * 2)
    dataset = dataset.batch(config.batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ---------------------------------------------------------------------
# MONITORING CALLBACKS
# ---------------------------------------------------------------------


class MetricsVisualizationCallback(keras.callbacks.Callback):
    """Visualizes training/validation metrics during training."""

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.visualization_dir = self.output_dir / "visualization_plots"
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.train_metrics = {"loss": [], "psnr_metric": []}
        self.val_metrics = {"val_loss": [], "val_psnr_metric": []}

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
            history_dict = {**self.train_metrics, **self.val_metrics}
            generate_training_curves(
                history=history_dict,
                results_dir=str(self.visualization_dir),
                filename=f"epoch_{epoch:03d}_metrics",
            )
            gc.collect()

            metrics_data = {
                "epoch": epoch,
                "train_metrics": self.train_metrics,
                "val_metrics": self.val_metrics,
            }
            with open(self.visualization_dir / "latest_metrics.json", "w") as f:
                json.dump(
                    metrics_data,
                    f,
                    indent=2,
                    default=lambda x: float(x) if hasattr(x, "item") else x,
                )
        except Exception as e:
            logger.warning(f"Failed to create metrics plots: {e}")


class StreamingResultMonitor(keras.callbacks.Callback):
    """Memory-efficient monitoring saving reconstruction comparison grids."""

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
            clean_patches = [
                load_and_preprocess_image(tf.constant(fp), self.config)
                for fp in monitor_files
            ]
            self.test_batch = tf.stack(clean_patches)
            logger.info(
                f"Created monitoring dataset with batch shape: {self.test_batch.shape}"
            )
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
                clean_images = self.test_batch
                outputs = self.model(clean_images, training=False)
                recon_images = (
                    outputs["reconstruction"]
                    if isinstance(outputs, dict)
                    else outputs
                )

                if self.config.save_training_images:
                    self._save_image_samples(epoch_val, clean_images, recon_images)

                mse_loss = tf.reduce_mean(tf.square(recon_images - clean_images))
                psnr = tf.reduce_mean(
                    tf.image.psnr(recon_images, clean_images, max_val=2.0)
                )
                logger.info(
                    f"Epoch {epoch_val} - Reconstruction MSE: {mse_loss.numpy():.6f}, "
                    f"PSNR: {psnr.numpy():.2f} dB"
                )

                metrics = {
                    "epoch": epoch_val,
                    "val_mse": float(mse_loss),
                    "val_psnr": float(psnr),
                    "timestamp": datetime.now().isoformat(),
                }
                with open(
                    self.results_dir / f"epoch_{epoch_val:03d}_metrics.json", "w"
                ) as f:
                    json.dump(metrics, f, indent=2)

                del clean_images, recon_images, outputs
                gc.collect()
            except Exception as e:
                tf.print(
                    f"Error during monitoring callback at epoch {epoch_val}: {e}"
                )
            return 0

        tf.py_function(func=_monitor_and_save, inp=[epoch + 1], Tout=[tf.int32])

    def _save_image_samples(
        self,
        epoch: int,
        clean: tf.Tensor,
        recon: tf.Tensor,
    ):
        """Save input / reconstruction / abs-difference comparison grid."""
        # Reconstruction errors in an identity AE are small; amplify the
        # absolute-difference heatmap so they are visible at a glance.
        diff_amplify = 5.0
        try:
            num_samples = min(10, clean.shape[0])
            fig, axes = plt.subplots(3, num_samples, figsize=(25, 7.5))
            fig.suptitle(
                f"CliffordLaplacianUNet Autoencoder - Epoch {epoch}",
                fontsize=20,
                y=0.98,
            )

            for i in range(num_samples):
                # Denormalize from [-1, +1] to [0, 1]
                clean_img = np.clip((clean[i].numpy() + 1.0) / 2.0, 0.0, 1.0)
                recon_img = np.clip((recon[i].numpy() + 1.0) / 2.0, 0.0, 1.0)

                # Absolute error in display space (collapse channels to a
                # single heatmap), amplified and clipped for visibility.
                diff_img = np.abs(clean_img - recon_img)
                if diff_img.ndim == 3 and diff_img.shape[-1] > 1:
                    diff_img = diff_img.mean(axis=-1)
                diff_img = np.clip(diff_img * diff_amplify, 0.0, 1.0)
                if diff_img.ndim == 3:  # grayscale: drop trailing channel
                    diff_img = diff_img.squeeze(-1)

                cmap = "gray" if clean_img.shape[-1] == 1 else None
                if clean_img.shape[-1] == 1:
                    clean_img = clean_img.squeeze(-1)
                    recon_img = recon_img.squeeze(-1)

                row_labels = [
                    "Input",
                    "Reconstruction",
                    f"|Diff| x{diff_amplify:g}",
                ]
                for row, img in enumerate([clean_img, recon_img]):
                    axes[row, i].imshow(img, cmap=cmap, vmin=0, vmax=1)
                    if i == 0:
                        axes[row, i].set_ylabel(
                            row_labels[row],
                            fontsize=14,
                            rotation=0,
                            ha="right",
                            va="center",
                        )
                    axes[row, i].axis("off")

                # Difference heatmap (perceptual colormap, fixed 0..1 scale).
                axes[2, i].imshow(diff_img, cmap="inferno", vmin=0, vmax=1)
                if i == 0:
                    axes[2, i].set_ylabel(
                        row_labels[2],
                        fontsize=14,
                        rotation=0,
                        ha="right",
                        va="center",
                    )
                axes[2, i].axis("off")

                axes[0, i].set_title(f"Sample {i + 1}", fontsize=10)

            plt.tight_layout()
            plt.subplots_adjust(top=0.90, left=0.08, right=0.98)
            plt.savefig(
                self.results_dir / f"epoch_{epoch:03d}_samples.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
            plt.clf()
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to save image samples: {e}")


def create_callbacks(
    config: TrainingConfig, val_directories: List[str]
) -> List[keras.callbacks.Callback]:
    """Create training callbacks: common utilities + domain-specific."""
    common_callbacks, _ = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="cliffordnet_autoencoder",
        run_dir=str(Path(config.output_dir) / config.experiment_name),
        monitor="val_loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_tensorboard=True,
        include_analyzer=False,
    )

    common_callbacks.append(MetricsVisualizationCallback(config))
    common_callbacks.append(StreamingResultMonitor(config, val_directories))

    return common_callbacks


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------


def train_cliffordnet_autoencoder(config: TrainingConfig) -> keras.Model:
    """Train a CliffordLaplacianUNet image autoencoder on DIV2K."""
    logger.info(
        f"Starting CliffordLaplacianUNet autoencoder training: "
        f"{config.experiment_name}"
    )

    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(config, str(output_dir), "config.json")

    for d in config.train_image_dirs:
        if not Path(d).exists():
            logger.warning(f"Training directory does not exist: {d}")
    for d in config.val_image_dirs:
        if not Path(d).exists():
            logger.warning(f"Validation directory does not exist: {d}")

    # Count files
    try:
        train_file_count = count_available_files(
            config.train_image_dirs, config.image_extensions, config.max_train_files
        )
        val_file_count = count_available_files(
            config.val_image_dirs, config.image_extensions, config.max_val_files
        )
    except Exception as e:
        logger.warning(f"Error counting files: {e}")
        train_file_count, val_file_count = 1000, 100

    if train_file_count == 0:
        raise ValueError("No training files found!")
    if val_file_count == 0:
        raise ValueError("No validation files found!")
    logger.info(
        f"Found ~{train_file_count} training, ~{val_file_count} validation files"
    )

    # Create datasets
    train_dataset = create_dataset(config.train_image_dirs, config, is_training=True)
    val_dataset = create_dataset(config.val_image_dirs, config, is_training=False)

    # Steps per epoch
    if config.steps_per_epoch is not None:
        steps_per_epoch = config.steps_per_epoch
    else:
        steps_per_epoch = max(
            100, (train_file_count * config.patches_per_image) // config.batch_size
        )
    logger.info(f"Using {steps_per_epoch} steps per epoch")

    # Create model
    model = create_clifford_laplacian_unet(
        variant=config.model_variant,
        in_channels=config.channels,
    )
    model.build((None, config.patch_size, config.patch_size, config.channels))
    model.summary()

    # Build optimizer with LR schedule
    lr_schedule = learning_rate_schedule_builder(
        {
            "type": config.lr_schedule_type,
            "learning_rate": config.learning_rate,
            "decay_steps": steps_per_epoch * config.epochs,
            "warmup_steps": steps_per_epoch * config.warmup_epochs,
            "alpha": 0.01,
        }
    )
    optimizer = optimizer_builder(
        {
            "type": config.optimizer_type,
            "gradient_clipping_by_norm": config.gradient_clipping,
        },
        lr_schedule,
    )

    # Deterministic reconstruction: single MSE term on the reconstruction head.
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[PsnrMetric(max_val=2.0, name="psnr_metric")],
    )
    logger.info(f"Model compiled with {model.count_params():,} parameters")

    # Train
    callbacks = create_callbacks(config, config.val_image_dirs)
    start_time = time.time()
    validation_steps = config.validation_steps or max(50, steps_per_epoch // 20)

    history = model.fit(
        train_dataset,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )
    logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")

    # Save final model
    try:
        model_path = output_dir / "final_model.keras"
        model.save(model_path)
        logger.info(f"Final model saved to: {model_path}")
    except Exception as e:
        logger.error(f"Failed to save final model: {e}")

    # Save training history
    try:
        history_dict = {
            k: [float(v) for v in vals] for k, vals in history.history.items()
        }
        with open(output_dir / "training_history.json", "w") as f:
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
        description="Train CliffordLaplacianUNet DIV2K Autoencoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--variant",
        choices=["small", "base", "large"],
        default="small",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patches-per-image", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--monitor-every", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-val-files", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument(
        "--gpu", type=int, default=None, help="GPU device index"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_gpu(gpu_id=args.gpu)

    config = TrainingConfig(
        patch_size=args.patch_size,
        channels=3,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patches_per_image=args.patches_per_image,
        max_train_files=args.max_train_files,
        max_val_files=args.max_val_files,
        parallel_reads=8,
        dataset_shuffle_buffer=1013,
        model_variant=args.variant,
        learning_rate=args.learning_rate,
        optimizer_type="adamw",
        lr_schedule_type="cosine_decay",
        warmup_epochs=5,
        monitor_every_n_epochs=args.monitor_every,
        early_stopping_patience=args.early_stopping_patience,
        save_training_images=True,
        validation_steps=200,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )

    logger.info(
        f"Config: variant={config.model_variant}, epochs={config.epochs}, "
        f"batch={config.batch_size}, lr={config.learning_rate}, "
        f"patch={config.patch_size}x{config.channels}"
    )

    try:
        model = train_cliffordnet_autoencoder(config)
        logger.info("Training completed successfully!")
        model.summary()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
