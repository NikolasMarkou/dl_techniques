"""CliffordNet confidence-interval denoiser training script.

Trains a bias-free conditional CliffordNet denoiser that outputs
**confidence intervals** instead of point estimates, following
Miyasawa's theorem.  Supports two uncertainty modes:

- **Gaussian heteroscedastic**: Dual heads predict mean + log-variance,
  trained with Gaussian NLL.
- **Quantile regression**: Multiple heads predict quantiles (e.g., 5th,
  50th, 95th percentile), trained with pinball loss.

Supports three conditioning modes (same as conditional denoiser):

- **Dense conditioning**: RGB image -> depth map denoising
- **Discrete conditioning**: Class label -> image denoising
- **Unconditional**: Standard image denoising

Usage::

    # Gaussian uncertainty (unconditional)
    python -m train.cliffordnet.train_confidence_denoiser \\
        --conditioning-mode none \\
        --uncertainty-mode gaussian \\
        --model-variant tiny \\
        --target-channels 1 \\
        --epochs 100 \\
        --gpu 1

    # Quantile regression (discrete conditioning)
    python -m train.cliffordnet.train_confidence_denoiser \\
        --conditioning-mode discrete \\
        --uncertainty-mode quantile \\
        --model-variant small \\
        --target-channels 3 \\
        --num-classes 100 \\
        --dataset cifar100 \\
        --epochs 100 \\
        --gpu 1

    # Dense conditioning with Gaussian uncertainty
    python -m train.cliffordnet.train_confidence_denoiser \\
        --conditioning-mode dense \\
        --uncertainty-mode gaussian \\
        --model-variant small \\
        --target-channels 1 \\
        --cond-channels 3 \\
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
from dl_techniques.models.cliffordnet.confidence_denoiser import (
    CliffordNetConfidenceDenoiser,
    GaussianNLLLoss,
    PinballLoss,
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


@dataclass
class ConfidenceTrainingConfig:
    """Configuration for confidence-interval CliffordNet denoiser training."""

    # Conditioning mode: "none", "dense", "discrete"
    conditioning_mode: str = "none"

    # Uncertainty mode: "gaussian" or "quantile"
    uncertainty_mode: str = "gaussian"
    quantiles: List[float] = field(
        default_factory=lambda: [0.05, 0.50, 0.95]
    )
    confidence_level: float = 0.90
    variance_regularization: float = 0.01

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

    # Noise — multiple regimes for visualization
    noise_sigma_min: float = 0.0
    noise_sigma_max: float = 0.5
    noise_distribution: str = "uniform"
    noise_regimes: List[float] = field(
        default_factory=lambda: [0.05, 0.15, 0.30, 0.50]
    )

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
                f"cliffordnet_confidence_{self.uncertainty_mode}_"
                f"{mode_tag}_{self.model_variant}_{timestamp}"
            )
        if self.conditioning_mode not in ("none", "dense", "discrete"):
            raise ValueError(
                f"Invalid conditioning_mode: {self.conditioning_mode}"
            )
        if self.uncertainty_mode not in ("gaussian", "quantile"):
            raise ValueError(
                f"Invalid uncertainty_mode: {self.uncertainty_mode}"
            )
        if self.noise_sigma_min < 0 or self.noise_sigma_max <= self.noise_sigma_min:
            raise ValueError("Invalid noise sigma range")


# ---------------------------------------------------------------------
# DATASET BUILDERS (same as conditional denoiser)
# ---------------------------------------------------------------------


def _normalize_image(image: tf.Tensor) -> tf.Tensor:
    """Normalize image to [-1, +1] range."""
    return (tf.cast(image, tf.float32) / 127.5) - 1.0


def _sample_noise_level(config: ConfidenceTrainingConfig) -> tf.Tensor:
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
    config: ConfidenceTrainingConfig,
    is_training: bool = True,
) -> Tuple[tf.data.Dataset, Tuple[int, ...], int]:
    """Create dataset for discrete conditioning from standard datasets."""
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = (
        load_dataset(config.dataset_name)
    )

    if is_training:
        images, labels = x_train, y_train
    else:
        images, labels = x_test, y_test

    images = (images.astype(np.float32) / 127.5) - 1.0

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
    config: ConfidenceTrainingConfig,
    is_training: bool = True,
) -> tf.data.Dataset:
    """Create paired dataset for dense conditioning."""
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

    n_pairs = min(len(target_files), len(cond_files))
    limit = config.max_train_files if is_training else config.max_val_files
    if limit and limit < n_pairs:
        n_pairs = limit

    target_files = target_files[:n_pairs]
    cond_files = cond_files[:n_pairs]
    logger.info(
        f"Created {n_pairs} paired samples for "
        f"{'train' if is_training else 'val'}"
    )

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
    config: ConfidenceTrainingConfig,
    is_training: bool = True,
) -> tf.data.Dataset:
    """Create dataset for unconditional denoising."""
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


class ConfidenceMetricsVisualizationCallback(keras.callbacks.Callback):
    """Visualizes training/validation metrics during training."""

    def __init__(self, config: ConfidenceTrainingConfig):
        super().__init__()
        self.config = config
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.visualization_dir = self.output_dir / "visualization_plots"
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.train_metrics: Dict[str, List[float]] = {}
        self.val_metrics: Dict[str, List[float]] = {}

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if logs is None:
            logs = {}
        for key, val in logs.items():
            if key.startswith("val_"):
                self.val_metrics.setdefault(key, []).append(float(val))
            else:
                self.train_metrics.setdefault(key, []).append(float(val))

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


def _z_score_for_level(level: float) -> float:
    """Compute z-score for a given confidence level (Abramowitz & Stegun)."""
    import math
    alpha = 1.0 - level
    p = 1.0 - alpha / 2.0
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    return t - (2.515517 + 0.802853 * t + 0.010328 * t**2) / (
        1.0 + 1.432788 * t + 0.189269 * t**2 + 0.001308 * t**3
    )


def _to_display(img: np.ndarray) -> np.ndarray:
    """Convert [-1,+1] image to [0,1] for display, squeezing single-channel."""
    disp = np.clip((img + 1.0) / 2.0, 0, 1)
    if disp.ndim == 3 and disp.shape[-1] == 1:
        disp = disp.squeeze(-1)
    return disp


def _extract_intervals(
    result: tf.Tensor,
    in_ch: int,
    uncertainty_mode: str,
    confidence_level: float,
    quantiles: List[float],
) -> Dict[str, tf.Tensor]:
    """Extract mean/median, lower, upper, uncertainty from raw model output."""
    if uncertainty_mode == "gaussian":
        mu = result[..., :in_ch]
        log_var = result[..., in_ch:]
        sigma_pred = tf.exp(0.5 * log_var)
        z = _z_score_for_level(confidence_level)
        return {
            "mu": mu,
            "lower": mu - z * sigma_pred,
            "upper": mu + z * sigma_pred,
            "uncertainty": sigma_pred,
        }
    # Quantile mode
    lower = result[..., :in_ch]
    upper = result[..., -in_ch:]
    median_idx = None
    for qi, q in enumerate(quantiles):
        if abs(q - 0.50) < 1e-6:
            median_idx = qi
            break
    if median_idx is not None:
        mu = result[..., median_idx * in_ch:(median_idx + 1) * in_ch]
    else:
        mu = (lower + upper) / 2.0
    return {
        "mu": mu,
        "lower": lower,
        "upper": upper,
        "uncertainty": upper - lower,
    }


class ConfidenceIntervalMonitor(keras.callbacks.Callback):
    """Comprehensive visualization of denoised results with confidence intervals.

    Generates five visualization artefacts per monitoring epoch:

    1. **Noise-regime grid** — rows = sigma levels, columns = Clean |
       Noisy | Denoised | |Error| | Uncertainty | CI Width.  Multiple
       samples shown as a column group.
    2. **Cross-section CI band** — picks a horizontal row from the image
       and draws the clean signal, the denoised mean, and the shaded
       CI band, one subplot per noise regime.
    3. **Error-vs-uncertainty scatter** — per-pixel scatter of absolute
       error vs predicted sigma / interval width.  Good calibration
       means the cloud hugs the diagonal.
    4. **Per-regime bar chart** — grouped bars for coverage and mean
       interval width across noise regimes.
    5. **Calibration curve** (Gaussian mode only) — predicted CI level
       on x-axis vs empirical coverage on y-axis for multiple levels.
    """

    # Number of test images to hold in memory
    _N_TEST_IMAGES: int = 4

    def __init__(self, config: ConfidenceTrainingConfig):
        super().__init__()
        self.config = config
        self.monitor_freq = config.monitor_every_n_epochs
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.results_dir = self.output_dir / "visualization_plots"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.test_batch = None
        self.test_labels = None
        # Accumulate per-regime metrics across epochs for trend plots
        self.regime_history: Dict[str, Dict[str, List[float]]] = {}

    # ------------------------------------------------------------------
    # Bootstrap: acquire a small fixed test batch
    # ------------------------------------------------------------------

    def on_train_begin(self, logs=None):
        """Grab a small batch from the validation set for monitoring."""
        n = self._N_TEST_IMAGES
        try:
            if self.config.conditioning_mode == "none":
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
                            if len(files) >= n:
                                break
                    if len(files) >= n:
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

            elif self.config.conditioning_mode == "discrete":
                (_, _), (x_test, y_test), _, _ = load_dataset(
                    self.config.dataset_name
                )
                images = (x_test[:n].astype(np.float32) / 127.5) - 1.0
                self.test_batch = tf.constant(images)
                self.test_labels = tf.constant(
                    y_test[:n].reshape(-1, 1).astype(np.int32)
                )

            if self.test_batch is not None:
                logger.info(
                    f"CI Monitor: test batch {self.test_batch.shape} "
                    f"({self.config.conditioning_mode})"
                )
        except Exception as e:
            logger.warning(f"CI Monitor: failed to create test batch: {e}")
            self.test_batch = None

    # ------------------------------------------------------------------
    # Epoch hook
    # ------------------------------------------------------------------

    def on_epoch_end(self, epoch: int, logs=None):
        if (epoch + 1) % self.monitor_freq != 0:
            return
        if self.test_batch is None:
            return
        try:
            regime_metrics = self._run_all_regimes(epoch + 1)
            self._save_regime_grid(epoch + 1, regime_metrics)
            self._save_crosssection_ci(epoch + 1, regime_metrics)
            self._save_error_vs_uncertainty(epoch + 1, regime_metrics)
            self._save_regime_bars(epoch + 1, regime_metrics)
            if self.config.uncertainty_mode == "gaussian":
                self._save_calibration_plot(epoch + 1)
            self._save_regime_metrics_json(epoch + 1, regime_metrics)
            gc.collect()
        except Exception as e:
            logger.warning(
                f"CI Monitor callback error at epoch {epoch + 1}: {e}"
            )

    # ------------------------------------------------------------------
    # Shared: run all noise regimes and collect results
    # ------------------------------------------------------------------

    def _make_model_input(self, noisy, idx=None):
        """Build model input list depending on conditioning mode."""
        if idx is None:
            batch = noisy
            labels = self.test_labels
        else:
            batch = noisy[idx:idx + 1]
            labels = self.test_labels[idx:idx + 1] if self.test_labels is not None else None

        if self.config.conditioning_mode == "discrete" and labels is not None:
            return [batch, labels]
        return batch

    def _run_all_regimes(self, epoch: int) -> List[Dict[str, Any]]:
        """Run inference for every noise regime and return collected data."""
        clean = self.test_batch
        in_ch = self.config.target_channels
        n_img = clean.shape[0]
        regime_data: List[Dict[str, Any]] = []

        for sigma in self.config.noise_regimes:
            noise = tf.random.normal(tf.shape(clean)) * sigma
            noisy = tf.clip_by_value(clean + noise, -1.0, 1.0)

            model_input = self._make_model_input(noisy)
            result = self.model(model_input, training=False)

            iv = _extract_intervals(
                result, in_ch,
                self.config.uncertainty_mode,
                self.config.confidence_level,
                self.config.quantiles,
            )

            # Metrics (over all test images)
            abs_error = tf.abs(iv["mu"] - clean)
            mse = float(tf.reduce_mean(tf.square(iv["mu"] - clean)).numpy())
            psnr = float(
                tf.reduce_mean(
                    tf.image.psnr(iv["mu"], clean, max_val=2.0)
                ).numpy()
            )
            covered = tf.logical_and(clean >= iv["lower"], clean <= iv["upper"])
            coverage = float(tf.reduce_mean(tf.cast(covered, tf.float32)).numpy())
            width = iv["upper"] - iv["lower"]
            mean_width = float(tf.reduce_mean(width).numpy())

            logger.info(
                f"Epoch {epoch} | sigma={sigma:.2f} — "
                f"MSE: {mse:.6f}, PSNR: {psnr:.2f} dB, "
                f"Coverage: {coverage:.3f}, Width: {mean_width:.4f}"
            )

            regime_data.append({
                "sigma": sigma,
                "clean": clean.numpy(),
                "noisy": noisy.numpy(),
                "mu": iv["mu"].numpy(),
                "lower": iv["lower"].numpy(),
                "upper": iv["upper"].numpy(),
                "uncertainty": iv["uncertainty"].numpy(),
                "abs_error": abs_error.numpy(),
                "mse": mse,
                "psnr": psnr,
                "coverage": coverage,
                "mean_width": mean_width,
            })

            # Track for trend plots
            key = f"sigma_{sigma:.2f}"
            if key not in self.regime_history:
                self.regime_history[key] = {
                    "epoch": [], "mse": [], "psnr": [],
                    "coverage": [], "width": [],
                }
            self.regime_history[key]["epoch"].append(epoch)
            self.regime_history[key]["mse"].append(mse)
            self.regime_history[key]["psnr"].append(psnr)
            self.regime_history[key]["coverage"].append(coverage)
            self.regime_history[key]["width"].append(mean_width)

        return regime_data

    # ------------------------------------------------------------------
    # Plot 1: noise-regime image grid
    # ------------------------------------------------------------------

    def _save_regime_grid(
        self, epoch: int, regime_data: List[Dict[str, Any]]
    ):
        """Grid: rows = sigma, cols = Clean | Noisy | Denoised | |Error| | Uncertainty | CI Width."""
        n_regimes = len(regime_data)
        n_img = min(self._N_TEST_IMAGES, regime_data[0]["clean"].shape[0])
        in_ch = self.config.target_channels

        # 7 semantic columns per image: clean, noisy, denoised, error, uncertainty, lower, upper
        col_labels = [
            "Clean", "Noisy", "Denoised", "|Error|",
            "Uncertainty", "Lower CI", "Upper CI",
        ]
        n_cols = len(col_labels)
        fig, axes = plt.subplots(
            n_regimes, n_cols,
            figsize=(2.8 * n_cols, 2.8 * n_regimes + 0.8),
        )
        if n_regimes == 1:
            axes = axes[np.newaxis, :]

        ci_pct = self.config.confidence_level * 100
        fig.suptitle(
            f"CliffordNet Confidence Denoiser — Epoch {epoch}   "
            f"[{self.config.uncertainty_mode}, {ci_pct:.0f}% CI]",
            fontsize=13, y=1.0,
        )

        for r_idx, rd in enumerate(regime_data):
            sigma = rd["sigma"]
            # Use first image in the batch
            imgs = {
                "Clean": rd["clean"][0],
                "Noisy": rd["noisy"][0],
                "Denoised": rd["mu"][0],
                "|Error|": rd["abs_error"][0],
                "Uncertainty": rd["uncertainty"][0],
                "Lower CI": rd["lower"][0],
                "Upper CI": rd["upper"][0],
            }

            for c_idx, label in enumerate(col_labels):
                ax = axes[r_idx, c_idx]
                raw = imgs[label]

                if label in ("|Error|", "Uncertainty"):
                    # Heatmap for error / uncertainty
                    if in_ch == 1:
                        disp = raw.squeeze(-1)
                    else:
                        disp = np.mean(raw, axis=-1)
                    vmax = max(float(np.percentile(disp, 99)), 1e-6)
                    cmap = "inferno" if label == "|Error|" else "hot"
                    im = ax.imshow(disp, cmap=cmap, vmin=0, vmax=vmax)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.imshow(_to_display(raw), cmap="gray" if in_ch == 1 else None,
                              vmin=0, vmax=1)

                if r_idx == 0:
                    ax.set_title(label, fontsize=9, fontweight="bold")
                if c_idx == 0:
                    ax.set_ylabel(
                        f"$\\sigma$={sigma:.2f}\n"
                        f"PSNR {rd['psnr']:.1f}\n"
                        f"Cov {rd['coverage']:.0%}",
                        fontsize=8, rotation=0, ha="right", va="center",
                        labelpad=50,
                    )
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(
            self.results_dir / f"epoch_{epoch:03d}_regime_grid.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
        plt.clf()

    # ------------------------------------------------------------------
    # Plot 2: cross-section CI band
    # ------------------------------------------------------------------

    def _save_crosssection_ci(
        self, epoch: int, regime_data: List[Dict[str, Any]]
    ):
        """For each regime, draw a 1-D horizontal cross-section through the
        middle of the first test image showing: clean signal, denoised mean,
        and shaded CI band.
        """
        n_regimes = len(regime_data)
        in_ch = self.config.target_channels
        channel = 0  # channel to plot

        fig, axes = plt.subplots(
            n_regimes, 1,
            figsize=(10, 2.5 * n_regimes + 0.6),
            sharex=True,
        )
        if n_regimes == 1:
            axes = [axes]

        fig.suptitle(
            f"Cross-section Confidence Intervals — Epoch {epoch}",
            fontsize=13, y=1.0,
        )

        for r_idx, rd in enumerate(regime_data):
            ax = axes[r_idx]
            sigma = rd["sigma"]
            h = rd["clean"].shape[1]
            row = h // 2  # middle row

            clean_row = rd["clean"][0, row, :, channel]
            noisy_row = rd["noisy"][0, row, :, channel]
            mu_row = rd["mu"][0, row, :, channel]
            lo_row = rd["lower"][0, row, :, channel]
            hi_row = rd["upper"][0, row, :, channel]

            x = np.arange(len(clean_row))

            ax.fill_between(
                x, lo_row, hi_row,
                alpha=0.25, color="C0", label="CI band",
            )
            ax.plot(x, clean_row, "k-", linewidth=1.0, label="Clean")
            ax.plot(x, noisy_row, color="0.65", linewidth=0.6,
                    alpha=0.6, label="Noisy")
            ax.plot(x, mu_row, "C0-", linewidth=1.0, label="Denoised")
            ax.set_ylabel(f"$\\sigma$={sigma:.2f}", fontsize=10)
            ax.set_xlim(0, len(x) - 1)
            if r_idx == 0:
                ax.legend(
                    loc="upper right", fontsize=8,
                    ncol=4, framealpha=0.7,
                )
            ax.tick_params(labelsize=8)
            # Add coverage annotation
            ax.text(
                0.01, 0.92,
                f"Coverage={rd['coverage']:.1%}  Width={rd['mean_width']:.3f}",
                transform=ax.transAxes, fontsize=7,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )

        axes[-1].set_xlabel("Pixel position", fontsize=10)
        plt.tight_layout()
        plt.savefig(
            self.results_dir / f"epoch_{epoch:03d}_crosssection_ci.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
        plt.clf()

    # ------------------------------------------------------------------
    # Plot 3: error vs uncertainty scatter
    # ------------------------------------------------------------------

    def _save_error_vs_uncertainty(
        self, epoch: int, regime_data: List[Dict[str, Any]]
    ):
        """Scatter plot of absolute error vs predicted uncertainty per pixel,
        one subplot per noise regime.  Well-calibrated models show the cloud
        centred on the diagonal.
        """
        n_regimes = len(regime_data)
        fig, axes = plt.subplots(
            1, n_regimes,
            figsize=(4 * n_regimes, 4),
            squeeze=False,
        )

        fig.suptitle(
            f"|Error| vs Predicted Uncertainty — Epoch {epoch}",
            fontsize=13, y=1.02,
        )

        for r_idx, rd in enumerate(regime_data):
            ax = axes[0, r_idx]
            sigma = rd["sigma"]

            err_flat = rd["abs_error"].flatten()
            unc_flat = rd["uncertainty"].flatten()
            # Subsample for speed (max 10k points)
            if len(err_flat) > 10000:
                idx = np.random.choice(len(err_flat), 10000, replace=False)
                err_flat = err_flat[idx]
                unc_flat = unc_flat[idx]

            ax.scatter(
                unc_flat, err_flat,
                s=1, alpha=0.15, c="C0", rasterized=True,
            )
            # Diagonal reference
            lim = max(float(np.percentile(unc_flat, 99)),
                      float(np.percentile(err_flat, 99)), 0.01)
            ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="y=x")
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_xlabel("Predicted uncertainty", fontsize=9)
            if r_idx == 0:
                ax.set_ylabel("|Error|", fontsize=9)
            ax.set_title(f"$\\sigma$={sigma:.2f}", fontsize=10)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=8)

            # Pearson correlation
            if np.std(unc_flat) > 0 and np.std(err_flat) > 0:
                corr = np.corrcoef(unc_flat, err_flat)[0, 1]
                ax.text(
                    0.05, 0.95, f"r={corr:.3f}",
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
                )

        plt.tight_layout()
        plt.savefig(
            self.results_dir / f"epoch_{epoch:03d}_error_vs_uncertainty.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
        plt.clf()

    # ------------------------------------------------------------------
    # Plot 4: per-regime bar chart (coverage + width)
    # ------------------------------------------------------------------

    def _save_regime_bars(
        self, epoch: int, regime_data: List[Dict[str, Any]]
    ):
        """Grouped bar chart: coverage and mean interval width per regime."""
        sigmas = [rd["sigma"] for rd in regime_data]
        coverages = [rd["coverage"] for rd in regime_data]
        widths = [rd["mean_width"] for rd in regime_data]
        psnrs = [rd["psnr"] for rd in regime_data]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.suptitle(
            f"Per-Regime Summary — Epoch {epoch}", fontsize=13, y=1.01
        )

        x = np.arange(len(sigmas))
        labels = [f"$\\sigma$={s:.2f}" for s in sigmas]

        # --- Coverage ---
        ax = axes[0]
        bars = ax.bar(x, coverages, color="C0", alpha=0.8)
        ax.axhline(
            self.config.confidence_level, color="red", linestyle="--",
            linewidth=1, label=f"Target {self.config.confidence_level:.0%}",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Empirical Coverage", fontsize=10)
        ax.set_title("Coverage", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        for bar, val in zip(bars, coverages):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", va="bottom", fontsize=8,
            )

        # --- Interval Width ---
        ax = axes[1]
        bars = ax.bar(x, widths, color="C1", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Mean Interval Width", fontsize=10)
        ax.set_title("Sharpness (lower is better)", fontsize=11)
        for bar, val in zip(bars, widths):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
            )

        # --- PSNR ---
        ax = axes[2]
        bars = ax.bar(x, psnrs, color="C2", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("PSNR (dB)", fontsize=10)
        ax.set_title("Denoising Quality", fontsize=11)
        for bar, val in zip(bars, psnrs):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(
            self.results_dir / f"epoch_{epoch:03d}_regime_bars.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
        plt.clf()

    # ------------------------------------------------------------------
    # Plot 5: calibration curve (Gaussian mode)
    # ------------------------------------------------------------------

    def _save_calibration_plot(self, epoch: int):
        """Predicted CI level vs empirical coverage, tested at multiple levels.

        Also shows per-regime calibration as separate curves so the user
        can see whether calibration degrades at high noise.
        """
        if self.test_batch is None:
            return

        clean = self.test_batch
        in_ch = self.config.target_channels
        levels = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

        fig, axes = plt.subplots(
            1, 2, figsize=(12, 5.5),
            gridspec_kw={"width_ratios": [1, 1]},
        )

        # --- Left: aggregate calibration ---
        ax = axes[0]
        ax.plot([0.45, 1.0], [0.45, 1.0], "k--", linewidth=1,
                label="Perfect")

        for sigma in self.config.noise_regimes:
            noise = tf.random.normal(tf.shape(clean)) * sigma
            noisy = tf.clip_by_value(clean + noise, -1.0, 1.0)
            model_input = self._make_model_input(noisy)
            result = self.model(model_input, training=False)
            mu = result[..., :in_ch]
            sigma_pred = tf.exp(0.5 * result[..., in_ch:])

            coverages = []
            for level in levels:
                z = _z_score_for_level(level)
                lo = mu - z * sigma_pred
                hi = mu + z * sigma_pred
                cov = float(tf.reduce_mean(
                    tf.cast(
                        tf.logical_and(clean >= lo, clean <= hi),
                        tf.float32,
                    )
                ).numpy())
                coverages.append(cov)

            ax.plot(levels, coverages, "o-", markersize=4,
                    label=f"$\\sigma$={sigma:.2f}")

        ax.set_xlabel("Predicted Confidence Level", fontsize=10)
        ax.set_ylabel("Empirical Coverage", fontsize=10)
        ax.set_title(f"Calibration — Epoch {epoch}", fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.45, 1.02)
        ax.set_ylim(0.45, 1.02)
        ax.set_aspect("equal")

        # --- Right: coverage trend across epochs ---
        ax = axes[1]
        target = self.config.confidence_level
        ax.axhline(target, color="red", linestyle="--", linewidth=1,
                    label=f"Target {target:.0%}")

        for key, hist in self.regime_history.items():
            if hist["epoch"]:
                ax.plot(
                    hist["epoch"], hist["coverage"],
                    "o-", markersize=3, label=key,
                )

        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Empirical Coverage", fontsize=10)
        ax.set_title("Coverage Trend Across Training", fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(
            self.results_dir / f"epoch_{epoch:03d}_calibration.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
        plt.clf()
        gc.collect()

    # ------------------------------------------------------------------
    # JSON metrics
    # ------------------------------------------------------------------

    def _save_regime_metrics_json(
        self, epoch: int, regime_data: List[Dict[str, Any]]
    ):
        """Persist per-regime numeric metrics to JSON."""
        metrics_per_regime = {}
        for rd in regime_data:
            metrics_per_regime[f"sigma_{rd['sigma']:.2f}"] = {
                "mse": rd["mse"],
                "psnr": rd["psnr"],
                "coverage": rd["coverage"],
                "mean_interval_width": rd["mean_width"],
            }
        with open(
            self.results_dir / f"epoch_{epoch:03d}_regime_metrics.json", "w"
        ) as f:
            json.dump(
                {
                    "epoch": epoch,
                    "uncertainty_mode": self.config.uncertainty_mode,
                    "confidence_level": self.config.confidence_level,
                    "regimes": metrics_per_regime,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )


def create_callbacks(
    config: ConfidenceTrainingConfig,
) -> List[keras.callbacks.Callback]:
    """Create training callbacks: common utilities + domain-specific."""
    common_callbacks, _ = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="cliffordnet_conf_denoiser",
        monitor="val_loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_tensorboard=True,
        include_analyzer=False,
    )

    common_callbacks.append(ConfidenceMetricsVisualizationCallback(config))
    common_callbacks.append(ConfidenceIntervalMonitor(config))

    return common_callbacks


# ---------------------------------------------------------------------
# MODEL CREATION
# ---------------------------------------------------------------------


def create_model_instance(
    config: ConfidenceTrainingConfig,
    input_shape: Optional[Tuple[int, int, int]] = None,
    num_classes: int = 0,
) -> CliffordNetConfidenceDenoiser:
    """Create a confidence CliffordNet denoiser from config."""
    enable_dense = config.conditioning_mode == "dense"
    enable_discrete = config.conditioning_mode == "discrete"

    dense_cond_channels = config.cond_channels if enable_dense else None
    nc = num_classes if enable_discrete else 0

    in_channels = config.target_channels
    if input_shape is not None:
        in_channels = input_shape[-1]

    return CliffordNetConfidenceDenoiser.from_variant(
        variant=config.model_variant,
        in_channels=in_channels,
        uncertainty_mode=config.uncertainty_mode,
        quantiles=config.quantiles,
        confidence_level=config.confidence_level,
        enable_dense_conditioning=enable_dense,
        dense_cond_channels=dense_cond_channels,
        enable_discrete_conditioning=enable_discrete,
        num_classes=nc,
        class_embedding_dim=config.class_embedding_dim,
        stochastic_depth_rate=config.stochastic_depth_rate,
        variance_regularization=config.variance_regularization,
    )


# ---------------------------------------------------------------------
# BIAS-FREE VERIFICATION
# ---------------------------------------------------------------------


def _verify_bias_free(model: keras.Model) -> None:
    """Log bias-free compliance check for the backbone."""
    bias_layers = []
    # Variance/quantile heads are allowed to have bias
    exempt_prefixes = ("var_", "quantile_")

    for layer in model._flatten_layers():
        name = layer.name
        # Skip uncertainty heads (they may use bias)
        if any(name.startswith(p) for p in exempt_prefixes):
            continue

        if hasattr(layer, "use_bias") and layer.use_bias:
            bias_layers.append(name)
        if isinstance(layer, keras.layers.BatchNormalization):
            if hasattr(layer, "center") and layer.center:
                bias_layers.append(f"{name} (BN center=True)")
        if isinstance(layer, keras.layers.LayerNormalization):
            if hasattr(layer, "center") and layer.center:
                bias_layers.append(f"{name} (LN center=True)")

    if bias_layers:
        logger.warning(
            f"Bias-free check: {len(bias_layers)} backbone layer(s) have "
            f"bias/centering: {bias_layers[:10]}"
        )
    else:
        logger.info(
            "Bias-free check: PASSED - backbone layers are bias-free "
            "(variance/quantile heads exempt)"
        )


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------


def train_confidence_denoiser(
    config: ConfidenceTrainingConfig,
) -> keras.Model:
    """Train a confidence-interval CliffordNet denoiser."""
    logger.info(
        f"Starting confidence CliffordNet denoiser training: "
        f"{config.experiment_name}"
    )
    logger.info(
        f"Conditioning: {config.conditioning_mode} | "
        f"Uncertainty: {config.uncertainty_mode}"
    )

    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    # Create datasets
    input_shape = None
    num_classes = 0

    if config.conditioning_mode == "discrete":
        train_ds, input_shape, num_classes = create_discrete_dataset(
            config, is_training=True
        )
        val_ds, _, _ = create_discrete_dataset(config, is_training=False)
        train_file_count = 50000
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

    # Use model's own loss and metrics (NLL or pinball, not MSE)
    loss = model.get_loss()
    metrics = model.get_metrics()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    logger.info(
        f"Model compiled with {model.count_params():,} parameters | "
        f"Loss: {loss.name} | "
        f"Uncertainty: {config.uncertainty_mode}"
    )

    # Verify bias-free compliance (backbone only)
    _verify_bias_free(model)

    # Train
    callbacks = create_callbacks(config)
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

    # Final calibration plot
    try:
        monitor = None
        for cb in callbacks:
            if isinstance(cb, ConfidenceIntervalMonitor):
                monitor = cb
                break
        if monitor and monitor.test_batch is not None:
            monitor._save_calibration_plot(config.epochs)
    except Exception as e:
        logger.warning(f"Failed to generate final calibration plot: {e}")

    # Write training summary
    try:
        trained_epochs = len(history.history.get("loss", []))
        best_loss = min(history.history.get("val_loss", [float("nan")]))
        best_cov = history.history.get("val_coverage", [float("nan")])
        best_cov = best_cov[-1] if best_cov else float("nan")
        best_width = history.history.get("val_interval_width", [float("nan")])
        best_width = best_width[-1] if best_width else float("nan")
        summary_path = output_dir / "training_summary.txt"
        with open(summary_path, "w") as f:
            f.write("CliffordNet Confidence Denoiser Training Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Uncertainty mode  : {config.uncertainty_mode}\n")
            f.write(f"Conditioning mode : {config.conditioning_mode}\n")
            f.write(f"Confidence level  : {config.confidence_level}\n")
            f.write(f"Model variant     : {config.model_variant}\n")
            f.write(f"Parameters        : {model.count_params():,}\n")
            f.write(f"Epochs trained    : {trained_epochs}\n")
            f.write(f"Best val_loss     : {best_loss:.6f}\n")
            f.write(f"Final coverage    : {best_cov:.4f}\n")
            f.write(f"Final width       : {best_width:.4f}\n")
            f.write(f"Noise range       : [{config.noise_sigma_min}, {config.noise_sigma_max}]\n")
            f.write(f"Noise regimes     : {config.noise_regimes}\n")
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
        description="Train CliffordNet Confidence Denoiser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Conditioning
    parser.add_argument(
        "--conditioning-mode",
        choices=["none", "dense", "discrete"],
        default="none",
        help="Conditioning mode: none, dense (spatial), or discrete (class)",
    )

    # Uncertainty
    parser.add_argument(
        "--uncertainty-mode",
        choices=["gaussian", "quantile"],
        default="gaussian",
        help="Uncertainty estimation mode",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.05, 0.50, 0.95],
        help="Quantile levels for quantile mode",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.90,
        help="Target confidence interval level",
    )
    parser.add_argument(
        "--variance-regularization",
        type=float,
        default=0.01,
        help="Variance regularization weight for Gaussian NLL",
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
    parser.add_argument(
        "--noise-regimes",
        type=float,
        nargs="+",
        default=[0.05, 0.15, 0.30, 0.50],
        help="Noise sigma values for regime visualization",
    )
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

    config = ConfidenceTrainingConfig(
        conditioning_mode=args.conditioning_mode,
        uncertainty_mode=args.uncertainty_mode,
        quantiles=args.quantiles,
        confidence_level=args.confidence_level,
        variance_regularization=args.variance_regularization,
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
        noise_sigma_min=args.noise_min,
        noise_sigma_max=args.noise_max,
        noise_distribution="uniform",
        noise_regimes=args.noise_regimes,
        model_variant=args.model_variant,
        stochastic_depth_rate=args.stochastic_depth_rate,
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
        f"uncertainty={config.uncertainty_mode}, "
        f"model={config.model_variant}, "
        f"epochs={config.epochs}, "
        f"batch={config.batch_size}, "
        f"lr={config.learning_rate}, "
        f"CI={config.confidence_level}, "
        f"noise=[{config.noise_sigma_min}, {config.noise_sigma_max}], "
        f"regimes={config.noise_regimes}"
    )

    train_confidence_denoiser(config)


if __name__ == "__main__":
    main()
