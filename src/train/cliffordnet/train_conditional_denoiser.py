"""CliffordNet conditional denoiser training script.

Trains a bias-free conditional CliffordNet denoiser following Miyasawa's
theorem for conditional denoising. Supports three conditioning modes:

- **Dense conditioning**: RGB image → depth map denoising
- **Discrete conditioning**: Class label → image denoising
- **Unconditional**: Standard image denoising (no conditioning)

Uses file-based paired/labeled image datasets with Gaussian noise
augmentation and [-1, +1] normalization.

Usage::

    # Dense conditioning (RGB → depth)
    python -m train.cliffordnet.train_conditional_denoiser \\
        --conditioning-mode dense \\
        --model-variant small \\
        --target-channels 1 \\
        --cond-channels 3 \\
        --epochs 100 \\
        --gpu 1

    # Discrete conditioning (class → image)
    python -m train.cliffordnet.train_conditional_denoiser \\
        --conditioning-mode discrete \\
        --model-variant tiny \\
        --target-channels 3 \\
        --num-classes 100 \\
        --dataset cifar100 \\
        --epochs 100 \\
        --gpu 1

    # Unconditional denoising
    python -m train.cliffordnet.train_conditional_denoiser \\
        --conditioning-mode none \\
        --model-variant tiny \\
        --target-channels 1 \\
        --epochs 100 \\
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
    load_dataset,
)
from dl_techniques.metrics.psnr_metric import PsnrMetric
from dl_techniques.utils.logger import logger
from dl_techniques.utils.filesystem import count_available_files
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)
from dl_techniques.models.cliffordnet.conditional_denoiser import (
    CliffordNetConditionalDenoiser,
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


@dataclass
class ConditionalTrainingConfig:
    """Configuration for conditional CliffordNet denoiser training."""

    # Conditioning mode: "none", "dense", "discrete"
    conditioning_mode: str = "none"

    # Data (file-based for dense conditioning)
    train_target_dirs: List[str] = field(default_factory=list)
    val_target_dirs: List[str] = field(default_factory=list)
    train_cond_dirs: List[str] = field(default_factory=list)
    val_cond_dirs: List[str] = field(default_factory=list)
    patch_size: int = 128
    target_channels: int = 1
    cond_channels: int = 3
    image_extensions: List[str] = field(
        default_factory=lambda: [
            ".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"
        ]
    )

    # Discrete conditioning
    dataset_name: str = "cifar100"
    num_classes: int = 100
    class_embedding_dim: int = 128

    # Memory
    max_train_files: Optional[int] = None
    max_val_files: Optional[int] = None
    parallel_reads: int = 8
    dataset_shuffle_buffer: int = 1000

    # Noise
    noise_sigma_min: float = 0.0
    noise_sigma_max: float = 0.5
    noise_distribution: str = "uniform"

    # Model
    model_variant: str = "tiny"
    stochastic_depth_rate: float = 0.1

    # Training
    batch_size: int = 32
    epochs: int = 100
    patches_per_image: int = 4
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
            mode_tag = {
                "none": "unconditional",
                "dense": "dense_cond",
                "discrete": "discrete_cond",
            }.get(self.conditioning_mode, self.conditioning_mode)
            self.experiment_name = (
                f"cliffordnet_denoiser_{mode_tag}_"
                f"{self.model_variant}_{timestamp}"
            )
        if self.conditioning_mode not in ("none", "dense", "discrete"):
            raise ValueError(
                f"Invalid conditioning_mode: {self.conditioning_mode}"
            )
        if self.noise_sigma_min < 0 or self.noise_sigma_max <= self.noise_sigma_min:
            raise ValueError("Invalid noise sigma range")


# ---------------------------------------------------------------------
# DATASET BUILDERS
# ---------------------------------------------------------------------


def _normalize_image(image: tf.Tensor) -> tf.Tensor:
    """Normalize image to [-1, +1] range."""
    return (tf.cast(image, tf.float32) / 127.5) - 1.0


def _sample_noise_level(config: ConditionalTrainingConfig) -> tf.Tensor:
    """Sample noise level from configured distribution."""
    if config.noise_distribution == "uniform":
        return tf.random.uniform(
            [], config.noise_sigma_min, config.noise_sigma_max
        )
    elif config.noise_distribution == "log_uniform":
        log_min = tf.math.log(tf.maximum(config.noise_sigma_min, 1e-6))
        log_max = tf.math.log(config.noise_sigma_max)
        return tf.exp(tf.random.uniform([], log_min, log_max))
    raise ValueError(f"Unknown distribution: {config.noise_distribution}")


def _augment_patch(patch: tf.Tensor) -> tf.Tensor:
    """Apply random flips and 90-degree rotations."""
    patch = tf.image.random_flip_left_right(patch)
    patch = tf.image.random_flip_up_down(patch)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    return tf.image.rot90(patch, k)


def _augment_pair(
    target: tf.Tensor, cond: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply synchronized augmentations to target/conditioning pair."""
    combined = tf.concat([target, cond], axis=-1)
    combined = tf.image.random_flip_left_right(combined)
    combined = tf.image.random_flip_up_down(combined)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    combined = tf.image.rot90(combined, k)
    t_ch = target.shape[-1]
    return combined[..., :t_ch], combined[..., t_ch:]


def create_discrete_dataset(
    config: ConditionalTrainingConfig,
    is_training: bool = True,
) -> tf.data.Dataset:
    """Create dataset for discrete conditioning from standard datasets.

    Returns batches of (inputs, targets) where:
    - inputs = (noisy_image, class_label)
    - targets = clean_image
    """
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = (
        load_dataset(config.dataset_name)
    )

    if is_training:
        images, labels = x_train, y_train
    else:
        images, labels = x_test, y_test

    # Normalize to [-1, +1] (load_dataset already maps to [0, 1])
    images = images.astype(np.float32) * 2.0 - 1.0

    def gen():
        indices = np.arange(len(images))
        if is_training:
            np.random.shuffle(indices)
        for i in indices:
            yield images[i], labels[i]

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    if is_training:
        ds = ds.shuffle(config.dataset_shuffle_buffer)
    ds = ds.repeat()

    def add_noise_and_format(image, label):
        sigma = _sample_noise_level(config)
        noise = tf.random.normal(tf.shape(image)) * sigma
        noisy_image = tf.clip_by_value(image + noise, -1.0, 1.0)
        class_label = tf.expand_dims(tf.cast(label, tf.int32), axis=0)
        return (noisy_image, class_label), image

    ds = ds.map(add_noise_and_format, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(config.batch_size, drop_remainder=is_training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, input_shape, num_classes


def _load_and_crop_image(
    image_path: tf.Tensor,
    channels: int,
    patch_size: int,
) -> tf.Tensor:
    """Load, preprocess, and extract a random patch from an image."""
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_image(
        image_string, channels=channels, expand_animations=False
    )
    image.set_shape([None, None, channels])
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1.0

    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    image = tf.cond(
        tf.logical_or(height < patch_size, width < patch_size),
        true_fn=lambda: tf.image.resize(
            image,
            [
                tf.maximum(height, patch_size),
                tf.maximum(width, patch_size),
            ],
        ),
        false_fn=lambda: image,
    )
    return tf.image.random_crop(image, [patch_size, patch_size, channels])


def create_dense_dataset(
    config: ConditionalTrainingConfig,
    is_training: bool = True,
) -> tf.data.Dataset:
    """Create paired dataset for dense conditioning.

    Expects matching file names in target_dirs and cond_dirs.
    Returns batches of (inputs, targets) where:
    - inputs = (noisy_target, cond_image)
    - targets = clean_target
    """
    dirs = config.train_target_dirs if is_training else config.val_target_dirs
    cond_dirs = config.train_cond_dirs if is_training else config.val_cond_dirs

    extensions = {ext.lower() for ext in config.image_extensions}
    extensions.update({ext.upper() for ext in config.image_extensions})

    target_files = []
    for d in dirs:
        dp = Path(d)
        if not dp.is_dir():
            continue
        for fp in sorted(dp.rglob("*")):
            if fp.is_file() and fp.suffix in extensions:
                target_files.append(str(fp))

    cond_files = []
    for d in cond_dirs:
        dp = Path(d)
        if not dp.is_dir():
            continue
        for fp in sorted(dp.rglob("*")):
            if fp.is_file() and fp.suffix in extensions:
                cond_files.append(str(fp))

    if not target_files or not cond_files:
        raise ValueError(
            f"No files found in target_dirs={dirs} or cond_dirs={cond_dirs}"
        )

    # Pair files by index (assumes matching order)
    n_pairs = min(len(target_files), len(cond_files))
    limit = config.max_train_files if is_training else config.max_val_files
    if limit and limit < n_pairs:
        n_pairs = limit

    target_files = target_files[:n_pairs]
    cond_files = cond_files[:n_pairs]
    logger.info(f"Created {n_pairs} paired samples for {'train' if is_training else 'val'}")

    ds = tf.data.Dataset.from_tensor_slices((target_files, cond_files))
    if is_training:
        ds = ds.shuffle(min(config.dataset_shuffle_buffer, n_pairs))
    ds = ds.repeat()

    def load_pair(target_path, cond_path):
        target = _load_and_crop_image(
            target_path, config.target_channels, config.patch_size
        )
        cond = _load_and_crop_image(
            cond_path, config.cond_channels, config.patch_size
        )
        return target, cond

    ds = ds.map(load_pair, num_parallel_calls=config.parallel_reads)

    if is_training and config.augment_data:
        ds = ds.map(
            lambda t, c: _augment_pair(t, c),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    def add_noise_and_format(target, cond):
        sigma = _sample_noise_level(config)
        noise = tf.random.normal(tf.shape(target)) * sigma
        noisy_target = tf.clip_by_value(target + noise, -1.0, 1.0)
        return (noisy_target, cond), target

    ds = ds.map(add_noise_and_format, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(config.batch_size, drop_remainder=is_training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def create_unconditional_dataset(
    config: ConditionalTrainingConfig,
    is_training: bool = True,
) -> tf.data.Dataset:
    """Create dataset for unconditional denoising.

    Returns batches of (noisy_image, clean_image).
    """
    dirs = config.train_target_dirs if is_training else config.val_target_dirs
    extensions = {ext.lower() for ext in config.image_extensions}
    extensions.update({ext.upper() for ext in config.image_extensions})

    all_files = []
    for d in dirs:
        dp = Path(d)
        if not dp.is_dir():
            continue
        for fp in dp.rglob("*"):
            if fp.is_file() and fp.suffix in extensions:
                all_files.append(str(fp))

    if not all_files:
        raise ValueError(f"No files found in {dirs}")

    limit = config.max_train_files if is_training else config.max_val_files
    if limit and limit < len(all_files):
        np.random.shuffle(all_files)
        all_files = all_files[:limit]

    logger.info(
        f"Found {len(all_files)} files for "
        f"{'train' if is_training else 'val'}"
    )

    ds = tf.data.Dataset.from_tensor_slices(all_files)
    if is_training:
        ds = ds.shuffle(min(config.dataset_shuffle_buffer, len(all_files)))
    ds = ds.repeat()

    ds = ds.map(
        lambda p: _load_and_crop_image(
            p, config.target_channels, config.patch_size
        ),
        num_parallel_calls=config.parallel_reads,
    )
    ds = ds.filter(lambda x: tf.reduce_sum(tf.abs(x)) > 0)

    if is_training and config.augment_data:
        ds = ds.map(_augment_patch, num_parallel_calls=tf.data.AUTOTUNE)

    def add_noise(patch):
        sigma = _sample_noise_level(config)
        noise = tf.random.normal(tf.shape(patch)) * sigma
        noisy = tf.clip_by_value(patch + noise, -1.0, 1.0)
        return noisy, patch

    ds = ds.map(add_noise, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(config.batch_size, drop_remainder=is_training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------
# MONITORING CALLBACKS
# ---------------------------------------------------------------------


class ConditionalMetricsVisualizationCallback(keras.callbacks.Callback):
    """Visualizes training/validation metrics during training."""

    def __init__(self, config: ConditionalTrainingConfig):
        super().__init__()
        self.config = config
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.visualization_dir = self.output_dir / "visualization_plots"
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.train_metrics = {
            "loss": [], "mae": [], "rmse": [], "psnr_metric": []
        }
        self.val_metrics = {
            "val_loss": [], "val_mae": [], "val_rmse": [],
            "val_psnr_metric": [],
        }

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
        except Exception as e:
            logger.warning(f"Failed to create metrics plots: {e}")


class StreamingResultMonitor(keras.callbacks.Callback):
    """Memory-efficient visual monitoring: saves noisy/clean/denoised grids."""

    def __init__(self, config: ConditionalTrainingConfig):
        super().__init__()
        self.config = config
        self.monitor_freq = config.monitor_every_n_epochs
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.results_dir = self.output_dir / "visualization_plots"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.test_batch = None
        self.test_labels = None
        self.test_cond = None

    def on_train_begin(self, logs=None):
        """Grab a small batch from the validation set for monitoring."""
        try:
            mode = self.config.conditioning_mode
            if mode == "discrete":
                # Grab samples from the standard dataset
                (_, _), (x_test, y_test), _, _ = load_dataset(
                    self.config.dataset_name
                )
                x_test = x_test.astype(np.float32) * 2.0 - 1.0
                indices = np.random.choice(len(x_test), 8, replace=False)
                self.test_batch = tf.constant(x_test[indices])
                self.test_labels = tf.constant(
                    y_test[indices].reshape(-1, 1).astype(np.int32)
                )
                logger.info(
                    f"Monitor: created discrete test batch "
                    f"{self.test_batch.shape}, labels {self.test_labels.shape}"
                )

            elif mode == "dense":
                # Grab paired samples from val dirs
                extensions = {
                    ext.lower() for ext in self.config.image_extensions
                }
                extensions.update(
                    ext.upper() for ext in self.config.image_extensions
                )
                target_files, cond_files = [], []
                for d in self.config.val_target_dirs:
                    dp = Path(d)
                    if not dp.exists():
                        continue
                    for fp in sorted(dp.rglob("*")):
                        if fp.is_file() and fp.suffix in extensions:
                            target_files.append(str(fp))
                            if len(target_files) >= 8:
                                break
                    if len(target_files) >= 8:
                        break
                for d in self.config.val_cond_dirs:
                    dp = Path(d)
                    if not dp.exists():
                        continue
                    for fp in sorted(dp.rglob("*")):
                        if fp.is_file() and fp.suffix in extensions:
                            cond_files.append(str(fp))
                            if len(cond_files) >= 8:
                                break
                    if len(cond_files) >= 8:
                        break
                n = min(len(target_files), len(cond_files))
                if n > 0:
                    t_patches, c_patches = [], []
                    for i in range(n):
                        t_patches.append(_load_and_crop_image(
                            tf.constant(target_files[i]),
                            self.config.target_channels,
                            self.config.patch_size,
                        ))
                        c_patches.append(_load_and_crop_image(
                            tf.constant(cond_files[i]),
                            self.config.cond_channels,
                            self.config.patch_size,
                        ))
                    self.test_batch = tf.stack(t_patches)
                    self.test_cond = tf.stack(c_patches)
                    logger.info(
                        f"Monitor: created dense test batch "
                        f"{self.test_batch.shape}, cond {self.test_cond.shape}"
                    )

            else:  # unconditional
                extensions = {
                    ext.lower() for ext in self.config.image_extensions
                }
                extensions.update(
                    ext.upper() for ext in self.config.image_extensions
                )
                files = []
                for d in self.config.val_target_dirs:
                    dp = Path(d)
                    if not dp.exists():
                        continue
                    for fp in dp.rglob("*"):
                        if fp.is_file() and fp.suffix in extensions:
                            files.append(str(fp))
                            if len(files) >= 8:
                                break
                    if len(files) >= 8:
                        break

                if files:
                    patches = []
                    for fp in files:
                        p = _load_and_crop_image(
                            tf.constant(fp),
                            self.config.target_channels,
                            self.config.patch_size,
                        )
                        patches.append(p)
                    self.test_batch = tf.stack(patches)
                    logger.info(
                        f"Monitor: created test batch {self.test_batch.shape}"
                    )
        except Exception as e:
            logger.warning(f"Monitor: failed to create test batch: {e}")
            self.test_batch = None

    def on_epoch_end(self, epoch: int, logs=None):
        if (epoch + 1) % self.monitor_freq != 0:
            return
        if self.test_batch is None:
            return

        try:
            clean = self.test_batch
            sigma = (
                self.config.noise_sigma_min + self.config.noise_sigma_max
            ) / 2.0
            noise = tf.random.normal(tf.shape(clean)) * sigma
            noisy = tf.clip_by_value(clean + noise, -1.0, 1.0)

            # Build model input based on conditioning mode
            mode = self.config.conditioning_mode
            if mode == "discrete" and self.test_labels is not None:
                model_input = (noisy, self.test_labels)
            elif mode == "dense" and self.test_cond is not None:
                model_input = (noisy, self.test_cond)
            else:
                model_input = noisy

            denoised = self.model(model_input, training=False)

            # Compute metrics
            mse = tf.reduce_mean(tf.square(denoised - clean)).numpy()
            psnr = tf.reduce_mean(
                tf.image.psnr(denoised, clean, max_val=2.0)
            ).numpy()
            logger.info(
                f"Epoch {epoch + 1} monitor — MSE: {mse:.6f}, "
                f"PSNR: {psnr:.2f} dB"
            )

            # Save image grid
            self._save_grid(epoch + 1, noisy, clean, denoised)

            # Save metrics
            metrics = {
                "epoch": epoch + 1,
                "monitor_mse": float(mse),
                "monitor_psnr": float(psnr),
                "sigma": float(sigma),
                "timestamp": datetime.now().isoformat(),
            }
            with open(
                self.results_dir / f"epoch_{epoch + 1:03d}_metrics.json",
                "w",
            ) as f:
                json.dump(metrics, f, indent=2)

            del noisy, denoised
            gc.collect()
        except Exception as e:
            logger.warning(f"Monitor callback error at epoch {epoch + 1}: {e}")

    def _save_grid(
        self,
        epoch: int,
        noisy: tf.Tensor,
        clean: tf.Tensor,
        denoised: tf.Tensor,
    ):
        """Save noisy/clean/denoised comparison grid."""
        try:
            n = min(8, noisy.shape[0])
            fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 7.5))
            fig.suptitle(
                f"CliffordNet Conditional Denoiser — Epoch {epoch}",
                fontsize=16,
                y=0.98,
            )

            for i in range(n):
                clean_img = np.clip((clean[i].numpy() + 1.0) / 2.0, 0, 1)
                noisy_img = np.clip((noisy[i].numpy() + 1.0) / 2.0, 0, 1)
                den_img = np.clip((denoised[i].numpy() + 1.0) / 2.0, 0, 1)

                cmap = "gray" if clean_img.shape[-1] == 1 else None
                if clean_img.shape[-1] == 1:
                    clean_img = clean_img.squeeze(-1)
                    noisy_img = noisy_img.squeeze(-1)
                    den_img = den_img.squeeze(-1)

                labels = ["Clean", "Noisy", "Denoised"]
                for row, img in enumerate([clean_img, noisy_img, den_img]):
                    axes[row, i].imshow(img, cmap=cmap, vmin=0, vmax=1)
                    if i == 0:
                        axes[row, i].set_ylabel(
                            labels[row], fontsize=12, rotation=0,
                            ha="right", va="center",
                        )
                    axes[row, i].axis("off")

            plt.tight_layout()
            plt.subplots_adjust(top=0.92, left=0.08)
            plt.savefig(
                self.results_dir / f"epoch_{epoch:03d}_samples.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
            plt.clf()
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to save image grid: {e}")


def create_callbacks(
    config: ConditionalTrainingConfig,
) -> Tuple[List[keras.callbacks.Callback], str]:
    """Create training callbacks: common utilities + domain-specific.

    Returns (callbacks, results_dir) so the training function uses the
    same single output directory for everything.
    """
    common_callbacks, results_dir = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="cliffordnet_cond_denoiser",
        monitor="val_loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_tensorboard=True,
        include_analyzer=False,
    )

    # Point config.output_dir / experiment_name at the common dir so
    # domain-specific callbacks write into the same place.
    config.output_dir = str(Path(results_dir).parent)
    config.experiment_name = Path(results_dir).name

    common_callbacks.append(ConditionalMetricsVisualizationCallback(config))
    common_callbacks.append(StreamingResultMonitor(config))

    return common_callbacks, results_dir


# ---------------------------------------------------------------------
# MODEL CREATION
# ---------------------------------------------------------------------


def create_model_instance(
    config: ConditionalTrainingConfig,
    input_shape: Optional[Tuple[int, int, int]] = None,
    num_classes: int = 0,
) -> CliffordNetConditionalDenoiser:
    """Create a conditional CliffordNet denoiser from config."""
    enable_dense = config.conditioning_mode == "dense"
    enable_discrete = config.conditioning_mode == "discrete"

    dense_cond_channels = config.cond_channels if enable_dense else None
    nc = num_classes if enable_discrete else 0

    in_channels = config.target_channels
    if input_shape is not None:
        in_channels = input_shape[-1]

    return CliffordNetConditionalDenoiser.from_variant(
        variant=config.model_variant,
        in_channels=in_channels,
        enable_dense_conditioning=enable_dense,
        dense_cond_channels=dense_cond_channels,
        enable_discrete_conditioning=enable_discrete,
        num_classes=nc,
        class_embedding_dim=config.class_embedding_dim,
        stochastic_depth_rate=config.stochastic_depth_rate,
    )


# ---------------------------------------------------------------------
# BIAS-FREE VERIFICATION
# ---------------------------------------------------------------------


def _verify_bias_free(model: keras.Model) -> None:
    """Log bias-free compliance check for the model."""
    bias_layers = []
    for layer in model._flatten_layers():
        if hasattr(layer, "use_bias") and layer.use_bias:
            bias_layers.append(layer.name)
        if isinstance(layer, keras.layers.BatchNormalization):
            if hasattr(layer, "center") and layer.center:
                bias_layers.append(f"{layer.name} (BN center=True)")
        if isinstance(layer, keras.layers.LayerNormalization):
            if hasattr(layer, "center") and layer.center:
                bias_layers.append(f"{layer.name} (LN center=True)")

    if bias_layers:
        logger.warning(
            f"Bias-free check: {len(bias_layers)} layer(s) have "
            f"bias/centering: {bias_layers[:10]}"
        )
    else:
        logger.info("Bias-free check: PASSED - all layers are bias-free")


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------


def train_conditional_denoiser(
    config: ConditionalTrainingConfig,
) -> keras.Model:
    """Train a conditional CliffordNet denoiser."""
    logger.info(
        f"Starting conditional CliffordNet denoiser training: "
        f"{config.experiment_name}"
    )
    logger.info(f"Conditioning mode: {config.conditioning_mode}")

    # Create datasets
    input_shape = None
    num_classes = 0

    if config.conditioning_mode == "discrete":
        train_ds, input_shape, num_classes = create_discrete_dataset(
            config, is_training=True
        )
        val_ds, _, _ = create_discrete_dataset(config, is_training=False)
        train_file_count = 50000  # Standard dataset size
        val_file_count = 10000

    elif config.conditioning_mode == "dense":
        train_ds = create_dense_dataset(config, is_training=True)
        val_ds = create_dense_dataset(config, is_training=False)
        train_file_count = min(
            count_available_files(
                config.train_target_dirs,
                config.image_extensions,
                config.max_train_files,
            ),
            count_available_files(
                config.train_cond_dirs,
                config.image_extensions,
                config.max_train_files,
            ),
        )
        val_file_count = min(
            count_available_files(
                config.val_target_dirs,
                config.image_extensions,
                config.max_val_files,
            ),
            count_available_files(
                config.val_cond_dirs,
                config.image_extensions,
                config.max_val_files,
            ),
        )

    else:  # unconditional
        train_ds = create_unconditional_dataset(config, is_training=True)
        val_ds = create_unconditional_dataset(config, is_training=False)
        train_file_count = count_available_files(
            config.train_target_dirs,
            config.image_extensions,
            config.max_train_files,
        )
        val_file_count = count_available_files(
            config.val_target_dirs,
            config.image_extensions,
            config.max_val_files,
        )

    logger.info(
        f"Dataset: ~{train_file_count} training, "
        f"~{val_file_count} validation samples"
    )

    # Steps per epoch
    if config.steps_per_epoch is not None:
        steps_per_epoch = config.steps_per_epoch
    else:
        steps_per_epoch = max(
            100,
            (train_file_count * config.patches_per_image)
            // config.batch_size,
        )
    logger.info(f"Using {steps_per_epoch} steps per epoch")

    # Create model
    model = create_model_instance(config, input_shape, num_classes)

    # Build model
    if config.conditioning_mode == "discrete" and input_shape is not None:
        model.build([
            (None, *input_shape),
            (None, 1),
        ])
    elif config.conditioning_mode == "dense":
        ps = config.patch_size
        model.build([
            (None, ps, ps, config.target_channels),
            (None, ps, ps, config.cond_channels),
        ])
    else:
        ps = config.patch_size
        model.build((None, ps, ps, config.target_channels))

    model.summary()

    # Build optimizer with LR schedule
    lr_schedule = learning_rate_schedule_builder({
        "type": config.lr_schedule_type,
        "learning_rate": config.learning_rate,
        "decay_steps": steps_per_epoch * config.epochs,
        "warmup_steps": steps_per_epoch * config.warmup_epochs,
        "alpha": 0.01,
    })
    optimizer = optimizer_builder(
        {
            "type": config.optimizer_type,
            "gradient_clipping_by_norm": config.gradient_clipping,
        },
        lr_schedule,
    )

    # MSE loss is critical for Miyasawa's theorem
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            "mae",
            keras.metrics.RootMeanSquaredError(name="rmse"),
            PsnrMetric(max_val=2.0, name="psnr_metric"),
        ],
    )
    logger.info(f"Model compiled with {model.count_params():,} parameters")

    # Verify bias-free compliance
    _verify_bias_free(model)

    # Train — create_callbacks returns a single results dir and redirects
    # config.output_dir / experiment_name to it so everything lands in one place.
    callbacks, results_dir = create_callbacks(config)
    output_dir = Path(results_dir)

    # Save config into the unified output directory
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    start_time = time.time()
    validation_steps = config.validation_steps or max(
        50, steps_per_epoch // 20
    )

    history = model.fit(
        train_ds,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )
    logger.info(
        f"Training completed in {time.time() - start_time:.2f} seconds"
    )

    # Save training history
    try:
        history_dict = {
            k: [float(v) for v in vals]
            for k, vals in history.history.items()
        }
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history_dict, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save training history: {e}")

    # Write training summary
    try:
        trained_epochs = len(history.history.get("loss", []))
        best_loss = min(history.history.get("val_loss", [float("nan")]))
        best_psnr = max(history.history.get("val_psnr_metric", [float("nan")]))
        summary_path = output_dir / "training_summary.txt"
        with open(summary_path, "w") as f:
            f.write("CliffordNet Conditional Denoiser Training Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Conditioning mode : {config.conditioning_mode}\n")
            f.write(f"Model variant     : {config.model_variant}\n")
            f.write(f"Parameters        : {model.count_params():,}\n")
            f.write(f"Epochs trained    : {trained_epochs}\n")
            f.write(f"Best val_loss     : {best_loss:.6f}\n")
            f.write(f"Best val_psnr     : {best_psnr:.2f} dB\n")
            f.write(f"Noise range       : [{config.noise_sigma_min}, {config.noise_sigma_max}]\n")
            f.write(f"Batch size        : {config.batch_size}\n")
            f.write(f"Learning rate     : {config.learning_rate}\n")
            f.write(f"Duration          : {time.time() - start_time:.1f}s\n")
        logger.info(f"Summary written to: {summary_path}")
    except Exception as e:
        logger.warning(f"Failed to write training summary: {e}")

    gc.collect()
    return model


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CliffordNet Conditional Denoiser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Conditioning
    parser.add_argument(
        "--conditioning-mode",
        choices=["none", "dense", "discrete"],
        default="none",
        help="Conditioning mode: none, dense (spatial), or discrete (class)",
    )

    # Model
    parser.add_argument(
        "--model-variant",
        choices=["tiny", "small", "base"],
        default="tiny",
    )
    parser.add_argument("--target-channels", type=int, default=1)
    parser.add_argument("--cond-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--class-embedding-dim", type=int, default=128)
    parser.add_argument("--stochastic-depth-rate", type=float, default=0.1)

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="cifar100",
        help="Standard dataset for discrete conditioning",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--noise-min", type=float, default=0.0)
    parser.add_argument("--noise-max", type=float, default=0.5)
    parser.add_argument("--patches-per-image", type=int, default=4)
    parser.add_argument("--monitor-every", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-val-files", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)

    # Output
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)

    # GPU
    parser.add_argument(
        "--gpu", type=int, default=None, help="GPU device index"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_gpu(gpu_id=args.gpu)

    config = ConditionalTrainingConfig(
        conditioning_mode=args.conditioning_mode,
        # File-based dirs (for dense/unconditional modes)
        train_target_dirs=[
            "/media/arxwn/data0_4tb/datasets/Megadepth",
            "/media/arxwn/data0_4tb/datasets/div2k/train",
            "/media/arxwn/data0_4tb/datasets/COCO/train2017",
        ],
        val_target_dirs=[
            "/media/arxwn/data0_4tb/datasets/div2k/validation",
            "/media/arxwn/data0_4tb/datasets/COCO/val2017",
        ],
        train_cond_dirs=[
            "/media/arxwn/data0_4tb/datasets/Megadepth/rgb",
        ],
        val_cond_dirs=[
            "/media/arxwn/data0_4tb/datasets/Megadepth/rgb_val",
        ],
        patch_size=args.patch_size,
        target_channels=args.target_channels,
        cond_channels=args.cond_channels,
        dataset_name=args.dataset,
        num_classes=args.num_classes,
        class_embedding_dim=args.class_embedding_dim,
        max_train_files=args.max_train_files or 10000,
        max_val_files=args.max_val_files or 1000,
        parallel_reads=8,
        dataset_shuffle_buffer=1013,
        model_variant=args.model_variant,
        stochastic_depth_rate=args.stochastic_depth_rate,
        noise_sigma_min=args.noise_min,
        noise_sigma_max=args.noise_max,
        noise_distribution="uniform",
        batch_size=args.batch_size,
        epochs=args.epochs,
        patches_per_image=args.patches_per_image,
        learning_rate=args.learning_rate,
        optimizer_type="adamw",
        lr_schedule_type="cosine_decay",
        warmup_epochs=5,
        monitor_every_n_epochs=args.monitor_every,
        save_training_images=True,
        validation_steps=200,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        steps_per_epoch=args.steps_per_epoch,
    )

    logger.info(
        f"Config: mode={config.conditioning_mode}, "
        f"model={config.model_variant}, "
        f"epochs={config.epochs}, "
        f"batch={config.batch_size}, "
        f"lr={config.learning_rate}, "
        f"target_ch={config.target_channels}, "
        f"noise=[{config.noise_sigma_min}, {config.noise_sigma_max}]"
    )

    train_conditional_denoiser(config)


if __name__ == "__main__":
    main()
