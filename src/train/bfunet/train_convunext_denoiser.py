"""Bias-free ConvNeXt (ConvUNext) denoiser trainer with a frozen Gabor stem and
a noise-sigma curriculum.

Trains ``create_convunext_denoiser`` (bias-free ConvNeXt U-Net) with an optional
NON-LEARNABLE Gabor depthwise stem on DIV2K + COCO. Patches of 256x256 are sampled
with geometric augmentation (flips + rot90), normalized to ``[-1, +1]`` (required
for bias-free / scaling-invariant denoising), and corrupted with additive Gaussian
noise whose per-image sigma is drawn from ``[sigma_min, sigma_max]``. The upper
bound ``sigma_max`` is a live ``tf.Variable`` widened every epoch by
``NoiseSigmaCurriculumCallback`` (curriculum: start with low noise, progressively
widen the spread).

Design notes (plan_2026-06-19_ed071c02):
- The noise function reads a captured ``tf.Variable`` (validated: a Variable read
  inside ``tf.data.map`` reflects per-epoch ``.assign``). This is a DELIBERATE
  Variable-backed variant of ``add_noise_to_patch`` (D-003), not a 4th blind copy.
- ConvNeXt V1 blocks are the default (strict bias-freedom: LayerNorm center=False).
  ConvNeXt V2 is opt-in but its GRN beta is trainable, so V2 is not strictly
  Mohan-compliant.
- Weighted COCO+DIV2K sourcing via ``select_weighted_image_paths`` prevents COCO's
  ~118K images from drowning DIV2K's ~800 (D-002 of plan_2026-06-18_1cca4fc1).

Reference PSNR baselines / SOTA (additive white Gaussian noise denoising)
------------------------------------------------------------------------
Published average PSNR (dB) on the standard AWGN denoising benchmarks. These are
REFERENCE TARGETS for interpreting this trainer's val-PSNR, NOT a like-for-like
leaderboard (see caveats below).

Noise-scale note: benchmark sigma is on the [0, 255] pixel scale. This trainer
adds noise in the [-1, +1] normalized space, so::

    sigma_255  =  sigma_here * 127.5

i.e. this run's curriculum ``sigma_max 0.05 -> 0.50`` corresponds to
``sigma_255 ~= 6.4 -> 63.75`` — it spans (and exceeds) the classic 15/25/50
benchmark regimes as a single *blind* model. PSNR is scale-invariant, so dB
computed here with ``max_val=2.0`` on [-1,+1] images is directly comparable to
the published ``max_val=255`` numbers.

Grayscale (1-ch), Set12 / BSD68:
    sigma=15:  DnCNN 32.86/31.73 | FFDNet 32.75/31.63 | DRUNet 33.25/31.91 |
               SwinIR 33.36/31.97 | Restormer 33.42/31.96
    sigma=25:  DnCNN 30.44/29.23 | FFDNet 30.43/29.19 | DRUNet 30.94/29.48 |
               SwinIR 31.01/29.50 | Restormer 31.08/29.52
    sigma=50:  DnCNN 27.18/26.23 | FFDNet 27.32/26.29 | DRUNet 27.90/26.59 |
               SwinIR 27.91/26.58 | Restormer 28.00/26.62

Color (3-ch) average PSNR (dB), Table 2 of Zhang et al. SCUNet (arXiv:2203.13278):
    | dataset  | s | DnCNN | FFDNet | DRUNet | SwinIR | Restormer | SCUNet |
    |----------|---|-------|--------|--------|--------|-----------|--------|
    | CBSD68   |15 | 33.90 | 33.87  | 34.30  | 34.42  | 34.40     | 34.40  |
    | CBSD68   |25 | 31.24 | 31.21  | 31.69  | 31.78  | 31.79     | 31.79  |
    | CBSD68   |50 | 27.95 | 27.96  | 28.51  | 28.56  | 28.60     | 28.61  |
    | Kodak24  |15 | 34.60 | 34.63  | 35.31  | 35.34  | 35.47     | 35.34  |
    | Kodak24  |25 | 32.14 | 32.13  | 32.89  | 32.89  | 33.04     | 32.92  |
    | Kodak24  |50 | 28.95 | 28.98  | 29.86  | 29.79  | 30.01     | 29.87  |
    | McMaster |15 | 33.45 | 34.66  | 35.40  | 35.61  | 35.61     | 35.60  |
    | McMaster |25 | 31.52 | 32.35  | 33.14  | 33.20  | 33.34     | 33.34  |
    | McMaster |50 | 28.62 | 29.18  | 30.08  | 30.22  | 30.30     | 30.29  |
    | Urban100 |15 | 32.98 | 33.83  | 34.81  | 35.13  | 35.13     | 35.18  |
    | Urban100 |25 | 30.81 | 31.40  | 32.60  | 32.90  | 32.96     | 33.03  |
    | Urban100 |50 | 27.59 | 28.05  | 29.61  | 29.82  | 30.02     | 30.14  |

Bias-free reference (most architecturally comparable to this model):
    Mohan et al., "Robust and Interpretable Blind Image Denoising via Bias-Free
    CNNs", ICLR 2020 (arXiv:1906.05478). BF-CNN MATCHES its biased DnCNN-style
    counterpart WITHIN the training noise range (e.g. ~29.2 dB on BSD68 s=25) but
    generalizes far better OUTSIDE it — a biased CNN collapses on unseen noise
    levels while the bias-free model degrades gracefully. That cross-noise
    robustness (not peak in-range PSNR) is the property this bias-free + noise-
    curriculum trainer targets.

Caveats (read before comparing):
  1. Different test set: numbers above are on Set12/BSD68/CBSD68/Kodak24/McMaster/
     Urban100. This trainer reports val-PSNR on DIV2K-validation patches — easier,
     cleaner content than Urban100, so absolute dB is NOT directly comparable.
  2. Blind wide-range vs fixed-sigma specialists: most SOTA rows are sigma-specific
     (or narrow blind). A single model trained over sigma_255 ~6..64 with a
     curriculum is solving a harder, broader task; expect lower peak PSNR at any
     single sigma than a specialist tuned for it.
  3. Capacity/training budget differ (these are trained to convergence on large
     corpora). Treat the table as orientation, not a target to "beat".

Usage::

    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m train.bfunet.train_convunext_denoiser \\
        --variant base --epochs 100 --batch-size 16 --patch-size 256 --gpu 1

    # Quick mechanism check (tiny, 2 epochs):
    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m train.bfunet.train_convunext_denoiser --smoke
"""

import gc
import csv
import json
import time
import keras
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # headless: avoid X11 crashes (LESSON)
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from train.common import (
    setup_gpu,
    create_callbacks as create_common_callbacks,
    save_config_json,
    collect_image_paths,
    augment_patch,
)
from train.superpoint.homographic_adaptation import select_weighted_image_paths
from dl_techniques.metrics.psnr_metric import PsnrMetric
from dl_techniques.metrics.ssim_metric import SsimMetric
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)
from dl_techniques.models.bias_free_denoisers.bfconvunext import (
    create_convunext_denoiser,
    CONVUNEXT_CONFIGS,
)
from dl_techniques.callbacks.noise_sigma_curriculum import (
    NoiseSigmaCurriculumCallback,
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for the bias-free ConvNeXt denoiser trainer."""

    # Data
    train_image_dirs: List[str] = field(
        default_factory=lambda: [
            "/media/arxwn/data0_4tb/datasets/COCO/train2017",
            "/media/arxwn/data0_4tb/datasets/div2k/train",
        ]
    )
    val_image_dirs: List[str] = field(
        default_factory=lambda: [
            "/media/arxwn/data0_4tb/datasets/div2k/validation",
        ]
    )
    dataset_weights: Optional[List[float]] = None  # None -> equal weight per dir
    patch_size: int = 256
    channels: int = 3
    image_extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"]
    )

    # Memory / sourcing
    max_train_files: Optional[int] = 10000
    max_val_files: Optional[int] = 500
    parallel_reads: int = 8
    dataset_shuffle_buffer: int = 1024
    seed: int = 42

    # Noise curriculum
    noise_sigma_min: float = 0.0
    sigma_max_start: float = 0.05   # narrow range at epoch 0 (low noise)
    sigma_max_end: float = 0.5      # wide range at the final curriculum epoch
    curriculum_epochs: Optional[int] = None  # None -> use `epochs`
    curriculum_schedule: str = "linear"      # linear | cosine | exp

    # Model (bias-free ConvNeXt U-Net)
    variant: str = "base"           # tiny | small | base | large | xlarge
    convnext_version: str = "v1"    # v1 = strict bias-free; v2 opt-in (GRN beta trains)
    use_gabor_stem: bool = True
    gabor_filters: int = 32
    gabor_kernel_size: int = 7
    enable_deep_supervision: bool = False

    # Training
    batch_size: int = 16
    epochs: int = 100
    patches_per_image: int = 8
    augment_data: bool = True
    steps_per_epoch: Optional[int] = None
    validation_steps: Optional[int] = 100

    # Optimization
    learning_rate: float = 1e-3
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "cosine_decay"
    warmup_epochs: int = 5
    gradient_clipping: float = 1.0
    early_stopping_patience: int = 15

    # Output
    output_dir: str = "results"
    experiment_name: Optional[str] = None
    viz_freq: int = 5  # save denoising grids every N epochs

    def __post_init__(self):
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"convunext_denoiser_{self.variant}_{timestamp}"
        if self.curriculum_epochs is None:
            self.curriculum_epochs = self.epochs
        if self.patch_size <= 0 or self.channels <= 0:
            raise ValueError("Invalid patch size or channel configuration")
        if self.noise_sigma_min < 0:
            raise ValueError("noise_sigma_min must be >= 0")
        if self.sigma_max_end <= self.noise_sigma_min:
            raise ValueError("sigma_max_end must exceed noise_sigma_min")
        if self.variant not in CONVUNEXT_CONFIGS:
            raise ValueError(
                f"Unknown variant {self.variant!r}; choices: {list(CONVUNEXT_CONFIGS)}"
            )
        if self.convnext_version not in ("v1", "v2"):
            raise ValueError("convnext_version must be 'v1' or 'v2'")
        if not self.train_image_dirs or not self.val_image_dirs:
            raise ValueError("train/val image dirs must be non-empty")


# ---------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------


def load_and_preprocess_image(
    image_path: tf.Tensor, config: TrainingConfig
) -> tf.Tensor:
    """Decode an image, normalize to [-1, +1], and crop a random patch."""
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_image(
        image_string, channels=config.channels, expand_animations=False
    )
    image.set_shape([None, None, config.channels])
    image = tf.cast(image, tf.float32)

    # Normalize to [-1, +1] (critical for bias-free architecture).
    image = (image / 127.5) - 1.0

    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    min_size = config.patch_size

    # Aspect-preserving upscale guard for images smaller than the patch.
    def _upscale():
        scale = tf.cast(min_size, tf.float32) / tf.cast(
            tf.minimum(height, width), tf.float32
        )
        new_h = tf.cast(tf.math.ceil(tf.cast(height, tf.float32) * scale), tf.int32)
        new_w = tf.cast(tf.math.ceil(tf.cast(width, tf.float32) * scale), tf.int32)
        return tf.image.resize(image, [new_h, new_w])

    image = tf.cond(
        tf.logical_or(height < min_size, width < min_size),
        true_fn=_upscale,
        false_fn=lambda: image,
    )

    return tf.image.random_crop(
        image, [config.patch_size, config.patch_size, config.channels]
    )


def make_curriculum_noise_fn(config: TrainingConfig, sigma_max_var: tf.Variable):
    """Build a noise function that samples per-image sigma from
    ``[noise_sigma_min, sigma_max_var]`` where the upper bound is a live
    ``tf.Variable`` widened per-epoch by the curriculum callback (D-003)."""

    sigma_min = float(config.noise_sigma_min)

    def add_curriculum_noise(patch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # sigma_max_var is read at graph-execution time -> reflects the per-epoch
        # .assign performed by NoiseSigmaCurriculumCallback (risk spike confirmed).
        noise_level = tf.random.uniform([], sigma_min, sigma_max_var)
        noisy = patch + tf.random.normal(tf.shape(patch)) * noise_level
        return tf.clip_by_value(noisy, -1.0, 1.0), patch

    return add_curriculum_noise


def collect_training_paths(config: TrainingConfig) -> List[str]:
    """Weighted COCO+DIV2K path worklist (prevents COCO drowning DIV2K)."""
    pairs = select_weighted_image_paths(
        image_dirs=config.train_image_dirs,
        weights=config.dataset_weights,
        num_images=config.max_train_files,
        seed=config.seed,
    )
    return [path for _name, path in pairs]


def create_dataset(
    file_paths: List[str],
    config: TrainingConfig,
    noise_fn,
    is_training: bool,
) -> tf.data.Dataset:
    """Build a tf.data pipeline of (noisy, clean) [-1,+1] patch pairs."""
    if not file_paths:
        raise ValueError("No image files found for the dataset")

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    if is_training:
        dataset = dataset.shuffle(
            buffer_size=min(config.dataset_shuffle_buffer, len(file_paths)),
            reshuffle_each_iteration=True,
        )
    dataset = dataset.repeat()

    if is_training and config.patches_per_image > 1:
        dataset = dataset.flat_map(
            lambda p: tf.data.Dataset.from_tensors(p).repeat(config.patches_per_image)
        )

    dataset = dataset.map(
        lambda p: load_and_preprocess_image(p, config),
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

    # Clip the clean patch back to [-1, +1] after augmentation. flips/rot90 preserve
    # range, but the aspect-safe bilinear upscale (small images) can overshoot; the
    # clean patch is both the model input and the regression target, so keep it in range.
    dataset = dataset.map(
        lambda x: tf.clip_by_value(x, -1.0, 1.0),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.map(noise_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(
        lambda noisy, clean: (
            tf.ensure_shape(
                noisy, [config.patch_size, config.patch_size, config.channels]
            ),
            tf.ensure_shape(
                clean, [config.patch_size, config.patch_size, config.channels]
            ),
        )
    )

    dataset = dataset.batch(config.batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------


def build_model(config: TrainingConfig) -> keras.Model:
    """Build the bias-free ConvNeXt denoiser from the variant config."""
    cfg = CONVUNEXT_CONFIGS[config.variant].copy()
    cfg.pop("description", None)
    cfg["convnext_version"] = config.convnext_version  # override variant default
    input_shape = (config.patch_size, config.patch_size, config.channels)
    return create_convunext_denoiser(
        input_shape=input_shape,
        use_gabor_stem=config.use_gabor_stem,
        gabor_filters=config.gabor_filters,
        gabor_kernel_size=config.gabor_kernel_size,
        enable_deep_supervision=config.enable_deep_supervision,
        final_activation="linear",  # MUST stay linear: bias-free homogeneity f(ax)=a*f(x)
        model_name=f"convunext_denoiser_{config.variant}",
        **cfg,
    )


def verify_bias_free(model: keras.Model) -> None:
    """Log a bias-free compliance check (informational)."""
    offenders = []
    for layer in model._flatten_layers():
        if getattr(layer, "use_bias", False):
            offenders.append(layer.name)
        if isinstance(layer, keras.layers.LayerNormalization) and getattr(
            layer, "center", False
        ):
            offenders.append(f"{layer.name} (LN center=True)")
    if offenders:
        logger.warning(
            f"Bias-free check: {len(offenders)} layer(s) carry bias/centering "
            f"(expected for ConvNeXt V2 GRN beta): {offenders[:10]}"
        )
    else:
        logger.info("Bias-free check: PASSED - all layers are bias-free")


# ---------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------


def _denorm(img: np.ndarray) -> np.ndarray:
    """Map a [-1, +1] image to [0, 1] for display."""
    return np.clip((img + 1.0) / 2.0, 0.0, 1.0)


def render_training_dashboard(history: dict, out_path: Path, title: str = "") -> None:
    """Render a single combined dashboard PNG of per-epoch training curves.

    Six panels: (1) Loss/MSE, (2) MSE (log), (3) PSNR, (4) MAE,
    (5) noise sigma_max (curriculum, with [0,255]-scale twin axis), (6) learning
    rate (log). ``history`` keys (any may be absent -> that panel is skipped):
    ``epoch, loss, val_loss, mae, val_mae, psnr, val_psnr, sigma_max, lr``.
    """
    ep = history.get("epoch")
    if not ep:
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    if title:
        fig.suptitle(title, fontsize=15, y=0.99)

    def _line(ax, ys, label, **kw):
        if ys is None:
            return
        n = min(len(ep), len(ys))
        ax.plot(ep[:n], ys[:n], label=label, lw=1.6, **kw)

    # (1) Loss / MSE (linear) -- loss IS mse (compiled loss='mse')
    ax = axes[0, 0]
    _line(ax, history.get("loss"), "train", color="#d62728")
    _line(ax, history.get("val_loss"), "val", color="#1f77b4")
    ax.set_title("Loss (MSE) per epoch"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3); ax.legend()

    # (2) MSE (log y) -- same data, log scale reveals late-epoch detail
    ax = axes[0, 1]
    _line(ax, history.get("loss"), "train", color="#d62728")
    _line(ax, history.get("val_loss"), "val", color="#1f77b4")
    ax.set_yscale("log")
    ax.set_title("MSE per epoch (log)"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE (log)")
    ax.grid(True, alpha=0.3, which="both"); ax.legend()

    # (3) PSNR
    ax = axes[0, 2]
    _line(ax, history.get("psnr"), "train", color="#d62728")
    _line(ax, history.get("val_psnr"), "val", color="#1f77b4")
    ax.set_title("PSNR per epoch"); ax.set_xlabel("epoch"); ax.set_ylabel("PSNR (dB)")
    ax.grid(True, alpha=0.3); ax.legend()

    # (4) MAE
    ax = axes[1, 0]
    _line(ax, history.get("mae"), "train", color="#d62728")
    _line(ax, history.get("val_mae"), "val", color="#1f77b4")
    ax.set_title("MAE per epoch"); ax.set_xlabel("epoch"); ax.set_ylabel("MAE")
    ax.grid(True, alpha=0.3); ax.legend()

    # (5) Noise level (curriculum) with [0,255]-scale twin axis
    ax = axes[1, 1]
    sig = history.get("sigma_max")
    _line(ax, sig, "sigma_max", color="#2ca02c", marker="o", markersize=3)
    ax.set_title("Noise sigma_max per epoch (curriculum)")
    ax.set_xlabel("epoch"); ax.set_ylabel("sigma_max  [-1,+1] units")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left")
    if sig is not None:
        twin = ax.twinx()
        twin.set_ylabel("sigma on [0,255] scale")
        lo, hi = ax.get_ylim()
        twin.set_ylim(lo * 127.5, hi * 127.5)

    # (6) Learning rate (log y)
    ax = axes[1, 2]
    _line(ax, history.get("lr"), "lr", color="#9467bd")
    ax.set_yscale("log")
    ax.set_title("Learning rate per epoch"); ax.set_xlabel("epoch"); ax.set_ylabel("lr (log)")
    ax.grid(True, alpha=0.3, which="both"); ax.legend()

    plt.tight_layout(rect=(0, 0, 1, 0.97) if title else None)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


class DenoisingVisualizationCallback(keras.callbacks.Callback):
    """Save clean / noisy / denoised comparison grids (and PSNR-vs-epoch curve).

    Every ``freq`` epochs, corrupts a FIXED clean validation batch with Gaussian
    noise at the CURRENT curriculum ``sigma_max`` and saves a 3-row panel
    (Clean | Noisy | Denoised) plus the running validation-PSNR curve. This is the
    denoising-specific visualization the scalar TensorBoard/CSV logs don't provide.
    """

    def __init__(
        self,
        clean_batch: tf.Tensor,
        sigma_max_var: tf.Variable,
        noise_sigma_min: float,
        out_dir: Path,
        freq: int = 5,
        max_samples: int = 8,
    ):
        super().__init__()
        self.clean_batch = clean_batch
        self.sigma_max_var = sigma_max_var
        self.noise_sigma_min = float(noise_sigma_min)
        self.viz_dir = Path(out_dir) / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.freq = max(1, freq)
        self.max_samples = max_samples
        self._psnr_history: List[float] = []
        self._hist = {k: [] for k in (
            "epoch", "loss", "val_loss", "mae", "val_mae",
            "psnr", "val_psnr", "sigma_max", "lr",
        )}

    def _current_lr(self) -> float:
        try:
            opt = self.model.optimizer
            lr = opt.learning_rate
            if isinstance(lr, keras.optimizers.schedules.LearningRateSchedule):
                return float(keras.ops.convert_to_numpy(lr(opt.iterations)))
            return float(keras.ops.convert_to_numpy(lr))
        except Exception:
            return float("nan")

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        if "val_psnr_metric" in logs:
            self._psnr_history.append(float(logs["val_psnr_metric"]))

        # Record per-epoch scalars for the combined dashboard. sigma_max is the
        # value the curriculum used this epoch; lr is read from the live optimizer
        # and also injected into logs so the CSVLogger captures it going forward.
        lr_val = self._current_lr()
        logs["lr"] = lr_val
        self._hist["epoch"].append(epoch + 1)
        self._hist["loss"].append(logs.get("loss", float("nan")))
        self._hist["val_loss"].append(logs.get("val_loss", float("nan")))
        self._hist["mae"].append(logs.get("mae", float("nan")))
        self._hist["val_mae"].append(logs.get("val_mae", float("nan")))
        self._hist["psnr"].append(logs.get("psnr_metric", float("nan")))
        self._hist["val_psnr"].append(logs.get("val_psnr_metric", float("nan")))
        self._hist["sigma_max"].append(float(self.sigma_max_var))
        self._hist["lr"].append(lr_val)

        try:
            render_training_dashboard(
                self._hist,
                self.viz_dir / "training_dashboard.png",
                title=f"Training dashboard - epoch {epoch + 1}",
            )
        except Exception as e:
            logger.warning(f"Dashboard render failed at epoch {epoch + 1}: {e}")

        if (epoch + 1) % self.freq != 0 and epoch != 0:
            return
        if self.clean_batch is None:
            return
        try:
            self._save_grid(epoch + 1)
            self._save_psnr_curve()
        except Exception as e:  # visualization must never break training
            logger.warning(f"Visualization failed at epoch {epoch + 1}: {e}")
        finally:
            gc.collect()

    def _save_grid(self, epoch: int):
        clean = self.clean_batch
        sigma = float(self.sigma_max_var)
        noise_level = tf.random.uniform([], self.noise_sigma_min, max(sigma, 1e-6))
        noisy = tf.clip_by_value(
            clean + tf.random.normal(tf.shape(clean)) * noise_level, -1.0, 1.0
        )
        denoised = self.model(noisy, training=False)
        if isinstance(denoised, (list, tuple)):
            denoised = denoised[0]  # deep-supervision: primary output

        clean_np = clean.numpy()
        noisy_np = noisy.numpy()
        den_np = np.asarray(denoised)
        n = min(self.max_samples, clean_np.shape[0])

        fig, axes = plt.subplots(3, n, figsize=(2.4 * n, 7.2))
        if n == 1:
            axes = axes.reshape(3, 1)
        fig.suptitle(
            f"ConvNeXt Denoiser - epoch {epoch} (sigma_max={sigma:.3f})",
            fontsize=14, y=0.99,
        )
        rows = ["Clean", f"Noisy", "Denoised"]
        for i in range(n):
            imgs = [_denorm(clean_np[i]), _denorm(noisy_np[i]), _denorm(den_np[i])]
            for r, img in enumerate(imgs):
                cmap = "gray" if img.shape[-1] == 1 else None
                disp = img.squeeze(-1) if img.shape[-1] == 1 else img
                axes[r, i].imshow(disp, cmap=cmap, vmin=0, vmax=1)
                axes[r, i].axis("off")
                if i == 0:
                    axes[r, i].set_title(rows[r], fontsize=11, loc="left")
        plt.tight_layout()
        path = self.viz_dir / f"epoch_{epoch:03d}_denoise_grid.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved denoising grid: {path}")

    def _save_psnr_curve(self):
        if not self._psnr_history:
            return
        fig = plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(self._psnr_history) + 1), self._psnr_history,
                 marker="o", lw=1.5)
        plt.xlabel("epoch")
        plt.ylabel("val PSNR (dB)")
        plt.title("Validation PSNR")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.viz_dir / "val_psnr_curve.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def build_fixed_val_batch(
    val_paths: List[str], config: TrainingConfig, n: int = 8
) -> Optional[tf.Tensor]:
    """Load a small FIXED batch of clean [-1,+1] patches for visualization."""
    if not val_paths:
        return None
    patches = []
    for p in val_paths[: max(n * 3, n)]:
        try:
            patch = load_and_preprocess_image(tf.constant(p), config)
            patch = tf.clip_by_value(patch, -1.0, 1.0)
            patches.append(patch)
            if len(patches) >= n:
                break
        except Exception:
            continue
    if not patches:
        return None
    return tf.stack(patches)


def build_dashboard_from_dir(exp_dir: str) -> Optional[Path]:
    """Rebuild the combined training dashboard from a finished/IN-PROGRESS run.

    Reads ``training_log.csv`` (per-epoch loss/mae/psnr) + ``config.json`` from an
    experiment directory, RECONSTRUCTS the per-epoch noise ``sigma_max`` (via the
    curriculum schedule) and learning rate (via the real LR-schedule builder), and
    writes ``visualizations/training_dashboard.png``. Lets us dashboard a live run
    without touching the running process. Returns the PNG path (or None)."""
    exp = Path(exp_dir)
    csv_path = exp / "training_log.csv"
    cfg_path = exp / "config.json"
    if not csv_path.exists():
        logger.error(f"No training_log.csv in {exp}")
        return None

    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        logger.error(f"Empty CSV: {csv_path}")
        return None

    def col(name):
        out = []
        for r in rows:
            try:
                out.append(float(r[name]))
            except (KeyError, ValueError):
                out.append(float("nan"))
        return out

    epochs = [int(float(r["epoch"])) + 1 for r in rows]  # 1-indexed for display
    hist = {
        "epoch": epochs,
        "loss": col("loss"), "val_loss": col("val_loss"),
        "mae": col("mae"), "val_mae": col("val_mae"),
        "psnr": col("psnr_metric"), "val_psnr": col("val_psnr_metric"),
        "sigma_max": None, "lr": None,
    }

    cfg = json.load(open(cfg_path)) if cfg_path.exists() else {}
    # Reconstruct noise sigma_max per epoch from the curriculum schedule.
    try:
        cur = NoiseSigmaCurriculumCallback(
            sigma_max_start=cfg.get("sigma_max_start", 0.05),
            sigma_max_end=cfg.get("sigma_max_end", 0.5),
            total_epochs=cfg.get("curriculum_epochs") or cfg.get("epochs", len(epochs)),
            schedule=cfg.get("curriculum_schedule", "linear"),
        )
        hist["sigma_max"] = [cur.sigma_max_at(e - 1) for e in epochs]
    except Exception as e:
        logger.warning(f"Could not reconstruct sigma_max: {e}")
    # Reconstruct learning rate per epoch from the real schedule builder.
    try:
        spe = cfg.get("steps_per_epoch")
        if spe:
            sched = learning_rate_schedule_builder({
                "type": cfg.get("lr_schedule_type", "cosine_decay"),
                "learning_rate": cfg.get("learning_rate", 1e-3),
                "decay_steps": spe * cfg.get("epochs", len(epochs)),
                "warmup_steps": spe * cfg.get("warmup_epochs", 0),
                "alpha": 0.01,
            })
            hist["lr"] = [float(keras.ops.convert_to_numpy(sched((e - 1) * spe)))
                          for e in epochs]
    except Exception as e:
        logger.warning(f"Could not reconstruct learning rate: {e}")

    out = exp / "visualizations" / "training_dashboard.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render_training_dashboard(
        hist, out, title=f"Training dashboard - {exp.name} ({len(epochs)} epochs)"
    )
    logger.info(f"Saved training dashboard: {out}")
    return out


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------


def train(config: TrainingConfig) -> keras.Model:
    """Train the bias-free ConvNeXt denoiser with the noise curriculum."""
    logger.info(f"Starting ConvNeXt denoiser training: {config.experiment_name}")
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(config, str(output_dir), "config.json")

    # Live, curriculum-controlled upper bound for the noise-sigma sampling range.
    sigma_max_var = tf.Variable(
        config.sigma_max_start, dtype=tf.float32, trainable=False, name="sigma_max"
    )
    noise_fn = make_curriculum_noise_fn(config, sigma_max_var)

    train_paths = collect_training_paths(config)
    val_paths = collect_image_paths(
        config.val_image_dirs,
        extensions=config.image_extensions,
        max_files=config.max_val_files,
    )
    logger.info(
        f"Sourced {len(train_paths)} train / {len(val_paths)} val image paths"
    )

    train_ds = create_dataset(train_paths, config, noise_fn, is_training=True)
    val_ds = create_dataset(val_paths, config, noise_fn, is_training=False)

    if config.steps_per_epoch is not None:
        steps_per_epoch = config.steps_per_epoch
    else:
        steps_per_epoch = max(
            100, (len(train_paths) * config.patches_per_image) // config.batch_size
        )
    validation_steps = config.validation_steps or max(
        10, len(val_paths) // config.batch_size
    )
    logger.info(
        f"steps_per_epoch={steps_per_epoch}, validation_steps={validation_steps}"
    )

    model = build_model(config)
    model.summary(print_fn=logger.info)
    verify_bias_free(model)

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

    # MSE loss -> least-squares-optimal (Miyasawa) denoiser.
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            "mae",
            PsnrMetric(max_val=2.0, name="psnr_metric"),
            SsimMetric(max_val=2.0, name="ssim_metric"),
        ],
    )
    logger.info(f"Model compiled with {model.count_params():,} parameters")

    callbacks, _ = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="convunext_denoiser",
        run_dir=str(output_dir),
        monitor="val_loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_tensorboard=True,
        include_analyzer=False,
    )
    callbacks.append(
        NoiseSigmaCurriculumCallback(
            sigma_max_var=sigma_max_var,
            sigma_max_start=config.sigma_max_start,
            sigma_max_end=config.sigma_max_end,
            total_epochs=config.curriculum_epochs,
            schedule=config.curriculum_schedule,
        )
    )

    # Denoising visualization: clean/noisy/denoised grids + PSNR curve.
    viz_batch = build_fixed_val_batch(val_paths, config, n=8)
    callbacks.append(
        DenoisingVisualizationCallback(
            clean_batch=viz_batch,
            sigma_max_var=sigma_max_var,
            noise_sigma_min=config.noise_sigma_min,
            out_dir=output_dir,
            freq=config.viz_freq,
        )
    )

    start = time.time()
    history = model.fit(
        train_ds,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )
    logger.info(f"Training completed in {time.time() - start:.2f}s")

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
        description="Train bias-free ConvNeXt denoiser (Gabor stem + noise curriculum)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--variant", choices=list(CONVUNEXT_CONFIGS), default="base")
    parser.add_argument("--convnext-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--no-gabor-stem", action="store_true",
                        help="Disable the frozen Gabor depthwise stem")
    parser.add_argument("--gabor-filters", type=int, default=32)
    parser.add_argument("--sigma-max-start", type=float, default=0.05)
    parser.add_argument("--sigma-max-end", type=float, default=0.5)
    parser.add_argument("--curriculum-schedule",
                        choices=["linear", "cosine", "exp"], default="linear")
    parser.add_argument("--curriculum-epochs", type=int, default=None)
    parser.add_argument("--deep-supervision", action="store_true")
    parser.add_argument("--viz-freq", type=int, default=5,
                        help="Save clean/noisy/denoised grids every N epochs")
    parser.add_argument("--dashboard", type=str, default=None,
                        help="Rebuild the combined training dashboard PNG from an "
                             "experiment dir (CSV+config) and exit; no training.")
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-val-files", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="Bound epoch length (default: files*patches//batch)")
    parser.add_argument("--validation-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Tiny end-to-end mechanism check (few steps/epochs, constant LR).",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Standalone dashboard rebuild (no training, no GPU needed): regenerate the
    # combined per-epoch dashboard from an experiment dir's CSV + config.
    if args.dashboard:
        build_dashboard_from_dir(args.dashboard)
        return

    setup_gpu(gpu_id=args.gpu)

    if args.smoke:
        # Mechanism check: tiny, fast, constant LR (avoid cosine collapse at 2 epochs).
        config = TrainingConfig(
            variant="tiny",
            convnext_version=args.convnext_version,
            use_gabor_stem=not args.no_gabor_stem,
            gabor_filters=8,
            epochs=2,
            curriculum_epochs=2,
            batch_size=2,
            patch_size=64,
            channels=3,
            patches_per_image=2,
            max_train_files=8,
            max_val_files=4,
            steps_per_epoch=3,
            validation_steps=2,
            warmup_epochs=0,
            # Mechanism check only; cosine_decay is the only supported schedule
            # (builder has no 'constant'). 2-epoch PSNR quality is NOT asserted.
            lr_schedule_type="cosine_decay",
            learning_rate=1e-3,
            sigma_max_start=0.05,
            sigma_max_end=0.5,
            curriculum_schedule="linear",
            viz_freq=1,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name or "convunext_denoiser_smoke",
        )
    else:
        config = TrainingConfig(
            variant=args.variant,
            convnext_version=args.convnext_version,
            use_gabor_stem=not args.no_gabor_stem,
            gabor_filters=args.gabor_filters,
            enable_deep_supervision=args.deep_supervision,
            epochs=args.epochs,
            curriculum_epochs=args.curriculum_epochs,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            channels=args.channels,
            patches_per_image=args.patches_per_image,
            learning_rate=args.learning_rate,
            sigma_max_start=args.sigma_max_start,
            sigma_max_end=args.sigma_max_end,
            curriculum_schedule=args.curriculum_schedule,
            max_train_files=args.max_train_files or 10000,
            max_val_files=args.max_val_files or 500,
            steps_per_epoch=args.steps_per_epoch,
            validation_steps=args.validation_steps if args.validation_steps is not None else 100,
            viz_freq=args.viz_freq,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
        )

    logger.info(
        f"Config: variant={config.variant} ({config.convnext_version}), "
        f"gabor_stem={config.use_gabor_stem}, epochs={config.epochs}, "
        f"patch={config.patch_size}x{config.channels}, "
        f"sigma_max {config.sigma_max_start}->{config.sigma_max_end} "
        f"({config.curriculum_schedule})"
    )

    try:
        train(config)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
