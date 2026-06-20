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

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from train.common import (
    setup_gpu,
    create_callbacks as create_common_callbacks,
    save_config_json,
    collect_image_paths,
    augment_patch,
    set_seeds,
)
from train.superpoint.homographic_adaptation import select_weighted_image_paths
from dl_techniques.metrics.psnr_metric import PsnrMetric
from dl_techniques.metrics.ssim_metric import SsimMetric
from dl_techniques.analyzer import AnalysisConfig
from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms.global_response_norm import GlobalResponseNormalization
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.utils.multiplicative_miyasawa import (
    apply_multiplicative_gaussian,
    apply_composite_gaussian,
)
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
    WWPGDProjectionCallback,
    WWTailConfig,
)
from dl_techniques.models.bias_free_denoisers.bfconvunext import (
    create_convunext_denoiser,
    CONVUNEXT_CONFIGS,
)
from dl_techniques.callbacks.noise_sigma_curriculum import (
    NoiseSigmaCurriculumCallback,
)
from dl_techniques.callbacks.convunext_bottleneck_monitor import (
    ConvUnextBottleneckMonitorCallback,
)
from dl_techniques.callbacks.self_iterate_pool import (
    SelfIteratePoolCallback,
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
    # Patch-level shuffle buffer applied AFTER the per-image patch flat_map, so the
    # `patches_per_image` consecutive crops of one image are interleaved across images.
    # Without it, batch_size <= patches_per_image yields batches drawn from a single image.
    patch_shuffle_buffer: int = 2048
    seed: int = 42

    # Noise curriculum
    noise_type: str = "additive"    # additive | multiplicative (y=x*(1+N*sigma)) | composite (y=x*n+a)
    composite_additive_ratio: float = 0.5  # composite: sigma_a = ratio * sigma_m (curriculum scalar)
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
    use_laplacian_pyramid: bool = False
    enable_deep_supervision: bool = False
    expose_bottleneck: bool = False
    # Opt-in depthwise-kernel init/regularization pass-through (plan_2026-06-20_353a3a76).
    # Both default None -> OFF, byte-identical to the pre-feature model (the blocks fall
    # back to their hardcoded TruncatedNormal(0,0.02) init + deepcopy(kernel_regularizer)).
    # depthwise_initializer: a keras init name; the alias 'orthonormal' maps to
    # Orthogonal(gain=1.0) via _resolve_depthwise_initializer (D-002). depthwise_l2: L2
    # weight on the depthwise kernels; None=off, else wired to keras.regularizers.L2.
    depthwise_initializer: Optional[str] = None
    depthwise_l2: Optional[float] = None

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
    warmup_epochs: Optional[int] = None  # None -> 10% of epochs (see __post_init__)
    gradient_clipping: float = 1.0
    # <= 0 disables early stopping. Default OFF: the noise-sigma curriculum makes val_loss
    # non-monotonic (it rises as the schedule ramps difficulty), so EarlyStopping on val_loss
    # fires prematurely and cuts training short. ModelCheckpoint still saves best_model.keras.
    early_stopping_patience: int = -1

    # Self-iterate (epoch-boundary pool regeneration; makes the denoiser self-iterable
    # so 2-5 sequential passes improve rather than over-smooth). Default OFF: the
    # streaming pipeline path is byte-identical when self_iterate=False.
    self_iterate: bool = False
    self_iterate_pool_size: int = 2048
    self_iterate_regen_freq: int = 1
    self_iterate_mix_ratio: float = 0.5

    # WW-PGD spectral tail-projection (epoch-boundary, non-differentiable). Default OFF:
    # when ww_pgd=False no callback is appended and training is byte-identical to before.
    # Hypers carry the ww_pgd_optimizer module defaults so the trainer is self-documenting.
    ww_pgd: bool = False
    ww_pgd_warmup_epochs: int = 0
    ww_pgd_ramp_epochs: int = 5
    ww_pgd_apply_every_epochs: int = 1
    ww_pgd_q: float = 1.0
    ww_pgd_blend_eta: float = 0.5
    ww_pgd_cayley_eta: float = 0.25
    ww_pgd_min_tail: int = 5
    # Per-epoch per-layer alpha logging (instrumentation, opt-in). When True (and
    # ww_pgd is also True) the WW-PGD callback writes a per-epoch alpha CSV to the
    # experiment dir. Default OFF: leaves the ww_pgd ON path byte-identical.
    ww_pgd_log_alpha: bool = False

    # Initialize model weights from a saved .keras checkpoint before training.
    # Primary use: self-iterate FINE-TUNING on top of a normally-trained denoiser
    # (the self-iterate pool's clean targets are fixed, so from-scratch self-iterate
    # has limited data diversity). Works in any mode; architecture must match the
    # checkpoint. None = train from random init (default).
    init_from: Optional[str] = None

    # Analysis (ModelAnalyzer, data-free weight + spectral). Default OFF (opt-in).
    enable_analyzer: bool = False
    analyzer_freq: int = 10        # run every N epochs
    analyzer_start_epoch: int = 1  # first epoch to analyze

    # Output
    output_dir: str = "results"
    experiment_name: Optional[str] = None
    viz_freq: int = 5  # save denoising grids every N epochs
    viz_samples: int = 8  # number of image columns in the eval grid

    def __post_init__(self):
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"convunext_denoiser_{self.variant}_{timestamp}"
        if self.curriculum_epochs is None:
            self.curriculum_epochs = self.epochs
        if self.warmup_epochs is None:
            # Default warmup = 10% of total training epochs (>=1).
            self.warmup_epochs = max(1, round(0.1 * self.epochs))
        if self.patch_size <= 0 or self.channels <= 0:
            raise ValueError("Invalid patch size or channel configuration")
        if self.noise_type not in {"additive", "multiplicative", "composite"}:
            raise ValueError(
                "noise_type must be 'additive', 'multiplicative', or 'composite'"
            )
        if self.composite_additive_ratio <= 0:
            raise ValueError("composite_additive_ratio must be > 0")
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
        # Self-iterate validation is guarded so the default-OFF config is unaffected.
        if self.self_iterate:
            if self.self_iterate_pool_size < self.batch_size:
                raise ValueError(
                    "self_iterate_pool_size "
                    f"({self.self_iterate_pool_size}) must be >= batch_size "
                    f"({self.batch_size}); a pool smaller than one batch cannot fill "
                    "a drop_remainder batch."
                )
            if self.noise_type != "additive":
                # Self-iterate is theory-bound to additive Gaussian noise: the Miyasawa
                # residual=score identity (and the clean-image fixed point it implies)
                # holds for additive noise ONLY; multiplicative/composite break the
                # linear-domain identity (D-003, research/miyasawas_theorem.md).
                raise ValueError(
                    "self_iterate requires noise_type='additive'; got "
                    f"{self.noise_type!r}. Multiplicative/composite noise breaks the "
                    "additive-only Miyasawa fixed-point theory the self-iterate "
                    "mechanism depends on (D-003)."
                )


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
    multiplicative = config.noise_type == "multiplicative"
    composite = config.noise_type == "composite"
    ratio = float(config.composite_additive_ratio)

    def add_curriculum_noise(patch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # sigma_max_var is read at graph-execution time -> reflects the per-epoch
        # .assign performed by NoiseSigmaCurriculumCallback (risk spike confirmed).
        # The per-image scalar sigma draw is SHARED by both noise types and is the
        # FIRST RNG op, so the curriculum behaves identically regardless of type.
        noise_level = tf.random.uniform([], sigma_min, sigma_max_var)
        if multiplicative:
            # DECISION plan_2026-06-20_4d26bdaf/D-001: multiplicative branch is opt-in
            # and lives AFTER the verbatim additive branch; do NOT refactor the additive
            # path's `patch + tf.random.normal(...) * noise_level` into a shared helper
            # -- that would reorder the additive RNG draws and break byte-identical
            # reproducibility of existing additive checkpoints (Pre-Mortem STOP-IF).
            noisy = apply_multiplicative_gaussian(patch, noise_level)
        elif composite:
            # DECISION plan_2026-06-20_f6ed2237/D-001: composite = multiplicative + additive floor.
            # sigma_a is tied to the curriculum scalar via composite_additive_ratio so the single
            # curriculum Variable drives both terms; the existing additive (else) and multiplicative
            # (if) blocks are kept VERBATIM so their RNG draw order -- and thus byte-identical
            # reproducibility of existing checkpoints -- is untouched (Pre-Mortem STOP-IF). Do NOT
            # fold the composite path into either existing block or reorder the shared noise_level draw.
            noisy = apply_composite_gaussian(patch, noise_level, ratio * noise_level)
        else:
            noisy = patch + tf.random.normal(tf.shape(patch)) * noise_level  # y = x + N(0, sigma^2)
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
        # Interleave the per-image patch copies across images BEFORE loading/cropping.
        # The flat_map above emits patches_per_image consecutive copies of each path, so
        # without this shuffle a batch of size <= patches_per_image would be all crops of
        # one image. Shuffling the cheap path copies here (strings, not patch tensors)
        # decorrelates batch composition with negligible memory.
        if config.patch_shuffle_buffer > 1:
            dataset = dataset.shuffle(
                buffer_size=config.patch_shuffle_buffer,
                reshuffle_each_iteration=True,
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
# SELF-ITERATE POOL (epoch-boundary regeneration data path; default OFF)
# ---------------------------------------------------------------------


def build_self_iterate_pool(
    file_paths: List[str],
    config: TrainingConfig,
    sigma_init: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a FIXED bounded RAM pool of clean patches + an initial noisy input pool.

    Loads ``config.self_iterate_pool_size`` clean patches ONCE (reusing the same
    ``load_and_preprocess_image`` load+crop logic the streaming pipeline uses, plus
    the same clip-to-[-1,+1] convention) into ``clean_pool``, then allocates a
    mutable ``current_input`` = ``clip(clean + N(0, sigma_init), -1, +1)`` as the
    epoch-1 additive-noise inputs.

    The ``current_input`` array is the LIVE buffer mutated IN PLACE by
    ``SelfIteratePoolCallback`` at epoch boundaries, and indexed (also in place) by
    the ``from_generator`` dataset built in ``create_self_iterate_dataset``.

    Args:
        file_paths: Clean image paths to crop patches from (cycled if too few).
        config: Trainer config (uses ``self_iterate_pool_size``, ``patch_size``,
            ``channels``, ``seed``).
        sigma_init: Additive-noise sigma for the initial ``current_input`` pool.

    Returns:
        ``(clean_pool, current_input)`` — two ``[P, patch, patch, C]`` float32 arrays
        in ``[-1, +1]``; ``clean_pool`` is the fixed target, ``current_input`` the
        mutable input buffer.

    Raises:
        ValueError: if no patches could be loaded from ``file_paths``.
    """
    if not file_paths:
        raise ValueError("No image files found for the self-iterate pool")

    pool_size = int(config.self_iterate_pool_size)
    rng = np.random.default_rng(config.seed or 42)

    patches: List[np.ndarray] = []
    idx = 0
    n_paths = len(file_paths)
    # Cap attempts so a directory of unreadable/degenerate images cannot loop forever.
    max_attempts = pool_size * 8 + n_paths
    attempts = 0
    while len(patches) < pool_size and attempts < max_attempts:
        path = file_paths[idx % n_paths]
        idx += 1
        attempts += 1
        try:
            # REUSE the streaming pipeline's load+random-crop+clip convention exactly.
            patch = load_and_preprocess_image(tf.constant(path), config)
            patch = tf.clip_by_value(patch, -1.0, 1.0)
            patch_np = np.asarray(patch, dtype=np.float32)
            if not np.any(np.abs(patch_np) > 0):
                continue  # mirror the streaming filter() that drops all-zero patches
            patches.append(patch_np)
        except Exception:
            continue

    if not patches:
        raise ValueError("Could not load any patches for the self-iterate pool")

    clean_pool = np.stack(patches[:pool_size]).astype(np.float32)
    if clean_pool.shape[0] < pool_size:
        logger.warning(
            "Self-iterate pool: requested %d patches but only %d were loadable.",
            pool_size,
            clean_pool.shape[0],
        )

    noise = rng.normal(size=clean_pool.shape).astype(np.float32)
    current_input = np.clip(
        clean_pool + noise * float(sigma_init), -1.0, 1.0
    ).astype(np.float32)

    logger.info(
        "Self-iterate pool built: %d patches of %dx%dx%d, sigma_init=%.4f",
        clean_pool.shape[0],
        config.patch_size,
        config.patch_size,
        config.channels,
        float(sigma_init),
    )
    return clean_pool, current_input


def create_self_iterate_dataset(
    clean_pool: np.ndarray,
    current_input: np.ndarray,
    config: TrainingConfig,
) -> Tuple[tf.data.Dataset, int]:
    """Build a tf.data source over the LIVE pool arrays via ``from_generator``.

    The generator indexes the SAME ``current_input`` / ``clean_pool`` numpy objects
    passed in (closes over the array objects, indexes inside the generator body), so
    in-place mutation by ``SelfIteratePoolCallback`` between epochs is reflected on
    the next fresh iterator that ``model.fit`` instantiates per epoch.

    The dataset is FINITE (no ``.repeat()``), sized to
    ``steps_per_epoch * batch_size`` items, so ``model.fit`` re-instantiates the
    generator each epoch and re-reads the mutated pool.

    Returns ``(dataset, steps_per_epoch)``.
    """
    # DECISION plan_2026-06-20_88705c63/D-004: build the pool-backed source with
    # `from_generator`, NOT `from_tensor_slices`. The Step-1 risk-spike empirically
    # FALSIFIED A1: `from_tensor_slices(numpy_array)` SNAPSHOTS the buffer into a
    # graph constant at construction, so in-place mutation of the original array is
    # NEVER re-read (same-iterator AND fresh-iterator both stale). `from_generator`
    # over a closure that indexes the LIVE array, sized FINITE (no `.repeat()`),
    # re-reads the mutated buffer on each fresh per-epoch iterator. Do NOT "optimize"
    # this back to from_tensor_slices/from_tensors — that silently kills regeneration
    # (the whole feature) with no error. See decisions.md D-004.
    pool_size = int(clean_pool.shape[0])
    batch_size = int(config.batch_size)
    steps_per_epoch = pool_size // batch_size
    if config.steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, int(config.steps_per_epoch))
    if steps_per_epoch < 1:
        raise ValueError(
            f"self-iterate pool ({pool_size}) too small for batch_size "
            f"({batch_size}): yields 0 steps_per_epoch."
        )
    n_items = steps_per_epoch * batch_size

    patch = int(config.patch_size)
    channels = int(config.channels)
    seed = int(config.seed or 42)

    def _gen():
        # Close over the LIVE array objects; index them HERE so each fresh iterator
        # sees the current (possibly callback-mutated) contents.
        order = np.random.default_rng(seed).permutation(pool_size)
        for j in range(n_items):
            i = int(order[j % pool_size])
            yield current_input[i], clean_pool[i]

    output_signature = (
        tf.TensorSpec(shape=(patch, patch, channels), dtype=tf.float32),
        tf.TensorSpec(shape=(patch, patch, channels), dtype=tf.float32),
    )
    dataset = tf.data.Dataset.from_generator(
        _gen, output_signature=output_signature
    )
    dataset = dataset.shuffle(
        buffer_size=min(config.patch_shuffle_buffer, n_items),
        reshuffle_each_iteration=True,
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, steps_per_epoch


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------


def _resolve_depthwise_initializer(name: Optional[str]):
    """Resolve the trainer's --depthwise-initializer string to an initializer.

    'orthonormal' -> keras.initializers.Orthogonal(gain=1.0) (the verified
    orthonormal path for a depthwise (K,K,C,1) kernel; unit-norm). None -> None
    (OFF, byte-identical). Any other string passes through unchanged for
    keras.initializers.get() resolution inside the block.
    """
    if name is None:
        return None
    if name == "orthonormal":
        return keras.initializers.Orthogonal(gain=1.0)
    return name


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
        use_laplacian_pyramid=config.use_laplacian_pyramid,
        enable_deep_supervision=config.enable_deep_supervision,
        expose_bottleneck=config.expose_bottleneck,
        final_activation="linear",  # MUST stay linear: bias-free homogeneity f(ax)=a*f(x)
        model_name=f"convunext_denoiser_{config.variant}",
        depthwise_initializer=_resolve_depthwise_initializer(config.depthwise_initializer),
        depthwise_regularizer=(
            keras.regularizers.L2(config.depthwise_l2)
            if config.depthwise_l2 is not None else None
        ),
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
        if isinstance(layer, GlobalResponseNormalization) and getattr(layer, "beta", None) is not None and layer.beta.trainable:
            offenders.append(f"{layer.name} (GRN beta - bias-like offset)")
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


# NOTE: batch-global RMSE PSNR (one RMSE over the whole batch). This intentionally
# differs from the per-image PsnrMetric used in training logs (which averages per-image
# PSNR), so eval-grid numbers will not match the val_psnr logged during fit.
def _mean_psnr(pred, clean) -> float:
    """Mean PSNR (dB) of ``pred`` vs ``clean`` on the [-1,+1] domain (max_val=2.0).

    Single source of truth for the trainer's PSNR convention: rmse over the whole
    batch, then ``20*log10(2.0/rmse)``. Both the eval grid and the multi-pass eval
    helpers call this so the formula lives in exactly one place (DRY).
    """
    mse = float(tf.reduce_mean(tf.square(tf.convert_to_tensor(pred) - clean)))
    return 20.0 * np.log10(2.0 / max(np.sqrt(mse), 1e-8))  # max_val=2.0


def denoise_k_passes(model: keras.Model, noisy, k: int) -> List[tf.Tensor]:
    """Apply ``model`` ``k`` times sequentially, clipping to [-1,+1] between passes.

    Returns the LIST of the k intermediate denoised tensors ``[pass1, ..., passk]``
    so callers can score each pass independently. The model is applied exactly once
    per pass (``training=False``); this is eval/inference only and never affects
    training. Domain [-1,+1] (clip after every pass).
    """
    outputs: List[tf.Tensor] = []
    x = tf.clip_by_value(tf.convert_to_tensor(noisy), -1.0, 1.0)
    for _ in range(int(k)):
        x = model(x, training=False)
        if isinstance(x, (list, tuple)):
            x = x[0]  # deep-supervision: primary output
        x = tf.clip_by_value(tf.convert_to_tensor(x), -1.0, 1.0)
        outputs.append(x)
    return outputs


def multi_pass_psnr(model: keras.Model, clean, noisy, k: int) -> List[float]:
    """Per-pass mean PSNR (dB) for k sequential applications of ``model``.

    Runs ``denoise_k_passes`` and scores each pass against ``clean`` using the SAME
    convention as the eval grid (``_mean_psnr``, max_val=2.0). Returns a list of k
    floats ``[psnr(pass1), ..., psnr(passk)]``. Eval-only; does not affect training.
    """
    clean_t = tf.convert_to_tensor(clean)
    return [_mean_psnr(p, clean_t) for p in denoise_k_passes(model, noisy, k)]


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
        val_ds=None,
        validation_steps: Optional[int] = None,
        noise_regimes: Optional[List[Tuple[str, float]]] = None,
        noise_type: str = "additive",
        composite_additive_ratio: float = 0.5,
        multi_pass_k: int = 3,
    ):
        super().__init__()
        self.clean_batch = clean_batch
        self.sigma_max_var = sigma_max_var
        self.noise_sigma_min = float(noise_sigma_min)
        self.noise_type = noise_type
        # Number of sequential passes scored/rendered in the multi-pass eval (additive
        # regimes only). Default 3 covers SC2 (passes 1->2->3). Eval/viz only.
        self.multi_pass_k = max(1, int(multi_pass_k))
        self.composite_additive_ratio = float(composite_additive_ratio)
        self.viz_dir = Path(out_dir) / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.freq = max(1, freq)
        self.max_samples = max_samples
        self.val_ds = val_ds
        self.validation_steps = validation_steps
        # Fixed reference noise regimes for the eval grid (sigma in [-1,+1] units),
        # DECOUPLED from the moving curriculum sigma so the grid is comparable across
        # epochs. Defaults map to the standard benchmark levels sigma_255 = 15/25/50.
        if noise_regimes is not None:
            self.noise_regimes = noise_regimes
        elif noise_type == "multiplicative":
            # Multiplicative-sigma regimes (n ~ N(1, sigma^2), dimensionless), spanning
            # the curriculum range; analog of the 15/25/50 AWGN benchmark triple.
            self.noise_regimes = [
                ("low", 0.10),
                ("medium", 0.25),
                ("high", 0.50),
            ]
        elif noise_type == "composite":
            # Composite regimes use the multiplicative sigma_m (same triple as the pure
            # multiplicative case); the additive floor sigma_a = ratio * sigma_m is derived
            # per regime in _save_grid, so only sigma_m is listed here.
            self.noise_regimes = [
                ("low", 0.10),
                ("medium", 0.25),
                ("high", 0.50),
            ]
        else:
            self.noise_regimes = [
                ("low", 15.0 / 127.5),
                ("medium", 25.0 / 127.5),
                ("high", 50.0 / 127.5),
            ]
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

    def on_train_begin(self, logs=None):
        """Epoch-0 baseline: visualize the UNTRAINED model (0 epochs completed).

        Saves the epoch-0 denoising grid and seeds the dashboard with an evaluated
        epoch-0 point so every curve starts from the untrained baseline. The x-axis
        is "epochs completed": 0 = untrained, then 1, 2, ... after each fit epoch.
        """
        # Epoch-0 grid on the fixed val batch (noise at the curriculum start sigma).
        if self.clean_batch is not None:
            try:
                self._save_grid(0)
            except Exception as e:
                logger.warning(f"Epoch-0 grid failed: {e}")
        # Epoch-0 dashboard point: evaluate the untrained model on the val set so the
        # baseline is computed the SAME way Keras computes epoch>=1 val metrics.
        if self.val_ds is not None:
            try:
                res = self.model.evaluate(
                    self.val_ds, steps=self.validation_steps,
                    verbose=0, return_dict=True,
                )
                lr_val = self._current_lr()
                self._hist["epoch"].append(0)
                self._hist["loss"].append(float("nan"))      # no train metric pre-fit
                self._hist["val_loss"].append(float(res.get("loss", float("nan"))))
                self._hist["mae"].append(float("nan"))
                self._hist["val_mae"].append(float(res.get("mae", float("nan"))))
                self._hist["psnr"].append(float("nan"))
                self._hist["val_psnr"].append(float(res.get("psnr_metric", float("nan"))))
                self._hist["sigma_max"].append(float(self.sigma_max_var))
                self._hist["lr"].append(lr_val)
                render_training_dashboard(
                    self._hist, self.viz_dir / "training_dashboard.png",
                    title="Training dashboard - epoch 0 (untrained baseline)",
                )
                logger.info(
                    f"Epoch-0 baseline: val_loss={res.get('loss'):.4f} "
                    f"val_psnr={res.get('psnr_metric', float('nan')):.2f} dB"
                )
            except Exception as e:
                logger.warning(f"Epoch-0 baseline eval failed: {e}")

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}

        # Record per-epoch scalars for the combined dashboard. sigma_max is the
        # value the curriculum used this epoch; lr is read from the live optimizer.
        # (CSV 'lr' is handled by LRLoggerCallback which runs before CSVLogger.)
        lr_val = self._current_lr()
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
        except Exception as e:  # visualization must never break training
            logger.warning(f"Visualization failed at epoch {epoch + 1}: {e}")
        finally:
            gc.collect()

    def _save_grid(self, epoch: int):
        """Eval grid: the SAME images under 3 fixed noise regimes.

        Row 0  : Clean (ground truth)
        Rows 1-2: Noisy(low)    / Denoised(low)
        Rows 3-4: Noisy(medium) / Denoised(medium)
        Rows 5-6: Noisy(high)   / Denoised(high)
        Both the Noisy- and Denoised-row labels carry the mean PSNR over the batch at
        that regime (noisy-vs-clean and denoised-vs-clean), so the improvement is visible.
        """
        clean = self.clean_batch
        clean_np = clean.numpy()
        n = min(self.max_samples, clean_np.shape[0])

        # Row 0 is clean; each regime contributes a (noisy, denoised) row pair.
        rows = [("Clean", clean_np)]
        multiplicative = self.noise_type == "multiplicative"
        composite = self.noise_type == "composite"
        ratio = self.composite_additive_ratio
        for label, sigma in self.noise_regimes:
            if multiplicative:
                # Per-pixel multiplicative regime: reuse the same noise primitive the
                # trainer uses, then the SAME [-1,+1] clip as the additive path.
                noisy = tf.clip_by_value(
                    apply_multiplicative_gaussian(clean, sigma), -1.0, 1.0
                )
            elif composite:
                # Composite regime: sigma here is sigma_m; the additive floor is
                # sigma_a = ratio * sigma_m. Reuse the trainer's composite primitive,
                # then the SAME [-1,+1] clip as the other paths.
                noisy = tf.clip_by_value(
                    apply_composite_gaussian(clean, sigma, ratio * sigma), -1.0, 1.0
                )
            else:
                noisy = tf.clip_by_value(
                    clean + tf.random.normal(tf.shape(clean)) * sigma, -1.0, 1.0
                )
            denoised = self.model(noisy, training=False)
            if isinstance(denoised, (list, tuple)):
                denoised = denoised[0]  # deep-supervision: primary output
            denoised = tf.convert_to_tensor(denoised)
            psnr = _mean_psnr(denoised, clean)  # max_val=2.0
            psnr_noisy = _mean_psnr(noisy, clean)  # max_val=2.0
            if multiplicative:
                noisy_label = f"Noisy {label}\n(mult σ={sigma:.2f}, PSNR {psnr_noisy:.1f} dB)"
            elif composite:
                noisy_label = (
                    f"Noisy {label}\n(comp σm={sigma:.2f} σa={ratio * sigma:.2f}, "
                    f"PSNR {psnr_noisy:.1f} dB)"
                )
            else:
                s255 = sigma * 127.5
                noisy_label = f"Noisy {label}\n(σ≈{s255:.0f}, PSNR {psnr_noisy:.1f} dB)"
            rows.append((noisy_label, noisy.numpy()))
            rows.append((f"Denoised {label}\n(PSNR {psnr:.1f} dB)", np.asarray(denoised)))

            # Multi-pass eval (SC2 surface): on ADDITIVE regimes only, score and render
            # passes 2..K so the monotone-ish improvement of self-iteration is visible
            # in the saved grid. Theory (Miyasawa fixed-point) is additive-only, so this
            # is skipped for multiplicative/composite. Pass-1 is already rendered above;
            # denoise_k_passes(...)[0] reproduces it, so we append [1:] (passes 2..K).
            if not multiplicative and not composite and self.multi_pass_k > 1:
                passes = denoise_k_passes(self.model, noisy, self.multi_pass_k)
                pass_psnrs = [_mean_psnr(p, clean) for p in passes]
                logger.info(
                    f"Multi-pass PSNR [{label}] (passes 1..{self.multi_pass_k}): "
                    + ", ".join(f"{v:.2f}" for v in pass_psnrs) + " dB"
                )
                for k_idx in range(1, self.multi_pass_k):  # passes 2..K
                    rows.append((
                        f"Denoised {label}\npass {k_idx + 1} "
                        f"(PSNR {pass_psnrs[k_idx]:.1f} dB)",
                        np.asarray(passes[k_idx]),
                    ))

        # 1 (clean) + 2*len(noise_regimes); + (K-1)*len(noise_regimes) extra denoised
        # rows when multi-pass eval is active (additive only).
        n_rows = len(rows)
        fig, axes = plt.subplots(n_rows, n, figsize=(2.3 * n, 2.3 * n_rows))
        axes = np.atleast_2d(axes)
        if n == 1:
            axes = axes.reshape(n_rows, 1)
        fig.suptitle(
            f"ConvUNeXt Denoiser - epoch {epoch} - same images, 3 noise regimes",
            fontsize=14, y=1.0,
        )
        for r, (label, arr) in enumerate(rows):
            for i in range(n):
                img = _denorm(arr[i])
                cmap = "gray" if img.shape[-1] == 1 else None
                disp = img.squeeze(-1) if img.shape[-1] == 1 else img
                ax = axes[r, i]
                ax.imshow(disp, cmap=cmap, vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if i == 0:
                    ax.set_ylabel(label, fontsize=9, rotation=0,
                                  ha="right", va="center", labelpad=12)
        plt.tight_layout()
        path = self.viz_dir / f"epoch_{epoch:03d}_denoise_grid.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved denoising grid (3 regimes): {path}")


class LRLoggerCallback(keras.callbacks.Callback):
    """Inject the current learning rate into ``logs`` so CSVLogger records an ``lr``
    column. MUST run BEFORE CSVLogger -> insert at the FRONT of the callbacks list
    (Keras runs callbacks in list order; an appended callback runs after CSVLogger
    and its ``logs`` edits never reach the already-written CSV row)."""

    def on_epoch_end(self, epoch: int, logs=None):
        if logs is None:
            return
        try:
            opt = self.model.optimizer
            lr = opt.learning_rate
            if isinstance(lr, keras.optimizers.schedules.LearningRateSchedule):
                logs["lr"] = float(keras.ops.convert_to_numpy(lr(opt.iterations)))
            else:
                logs["lr"] = float(keras.ops.convert_to_numpy(lr))
        except Exception:
            pass


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
    """Train the bias-free ConvUNeXt denoiser with the noise curriculum."""
    logger.info(f"Starting ConvUNeXt denoiser training: {config.experiment_name}")
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(config, str(output_dir), "config.json")

    set_seeds(config.seed)  # reproducible weight init (H8)

    # DECISION plan_2026-06-20_0433c2f2/D-002: --deep-supervision is NOT wired in this
    # trainer (no multi-scale targets, no per-output loss dict, no weight scheduler), so a
    # multi-output model would crash mid-fit with a cryptic KeyError (audit H2). Fail fast
    # with an actionable message instead. Full DS wiring is deferred to a future plan.
    if config.enable_deep_supervision:
        raise ValueError(
            "enable_deep_supervision is not supported by this trainer: it builds a "
            "multi-resolution multi-output model but no multi-scale targets / per-output "
            "loss dict / weight scheduler are wired, which crashes during fit. Remove "
            "--deep-supervision (full deep-supervision support is deferred; see D-002)."
        )

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

    # Validation pipeline is identical in BOTH branches (streaming, fixed-sigma
    # noise on a fixed val set); only the TRAIN source differs.
    # DECISION plan_2026-06-20_0433c2f2/D-001: validation noise is decoupled from the
    # curriculum. The training noise_fn reads the live, per-epoch-widened sigma_max_var,
    # which makes val_loss non-stationary -> ModelCheckpoint(monitor=val_loss) froze
    # best_model.keras near epoch 0 (audit H1). A separate FIXED-sigma Variable (never
    # .assign()-ed, not handed to NoiseSigmaCurriculumCallback) gives a stationary val
    # objective so the checkpoint tracks true denoising quality. Kept monitor=val_loss/min
    # (do NOT switch to val_psnr: create_callbacks forces mode=min unless 'accuracy' in name).
    sigma_fixed_var = tf.Variable(
        config.sigma_max_end, dtype=tf.float32, trainable=False, name="sigma_fixed_val"
    )
    val_noise_fn = make_curriculum_noise_fn(config, sigma_fixed_var)
    val_ds = create_dataset(val_paths, config, val_noise_fn, is_training=False)
    validation_steps = config.validation_steps or max(
        10, len(val_paths) // config.batch_size
    )

    # DECISION plan_2026-06-20_88705c63/D-004: branch the TRAIN data path on
    # config.self_iterate. ON -> a FINITE `from_generator` source over a bounded RAM
    # pool whose `current_input` buffer is mutated IN PLACE by
    # SelfIteratePoolCallback at epoch boundaries (the regeneration mechanism). OFF ->
    # the EXACT existing streaming `create_dataset` path (byte-identical to today).
    # Do NOT collapse these into the streaming pipeline: a `from_tensor_slices` /
    # streaming source SNAPSHOTS or never re-reads the pool, silently killing
    # regeneration with no error (see decisions.md D-004). The model is applied once
    # per batch in BOTH branches; compile/build/verify_bias_free/fit are identical.
    self_iterate_callback: Optional[SelfIteratePoolCallback] = None
    if config.self_iterate:
        # Initial pool sigma = the curriculum START value (the same value the live
        # sigma_max_var is initialized to at epoch 0), so epoch-1 inputs match the
        # streaming pipeline's epoch-0 noise level.
        sigma_init = float(config.sigma_max_start)
        clean_pool, current_input = build_self_iterate_pool(
            train_paths, config, sigma_init
        )
        train_ds, steps_per_epoch = create_self_iterate_dataset(
            clean_pool, current_input, config
        )
        # get_sigma reads the SAME live curriculum Variable the
        # NoiseSigmaCurriculumCallback advances each epoch, so fresh pool slots track
        # the curriculum. The pool dataset has NO tf.data noise closure, so this is
        # the only consumer of the curriculum variable in self-iterate mode.
        self_iterate_callback = SelfIteratePoolCallback(
            clean_pool=clean_pool,
            current_input=current_input,
            get_sigma=lambda: float(sigma_max_var),
            regen_freq=config.self_iterate_regen_freq,
            mix_ratio=config.self_iterate_mix_ratio,
            predict_batch_size=config.batch_size,
            seed=config.seed or 42,
        )
        logger.info(
            "Self-iterate mode ACTIVE: pool_size=%d, regen_freq=%d, mix_ratio=%.3f, "
            "steps_per_epoch=%d (finite from_generator pool path)",
            int(clean_pool.shape[0]),
            int(config.self_iterate_regen_freq),
            float(config.self_iterate_mix_ratio),
            int(steps_per_epoch),
        )
    else:
        train_ds = create_dataset(train_paths, config, noise_fn, is_training=True)
        if config.steps_per_epoch is not None:
            steps_per_epoch = config.steps_per_epoch
        else:
            steps_per_epoch = max(
                100, (len(train_paths) * config.patches_per_image) // config.batch_size
            )

    logger.info(
        f"steps_per_epoch={steps_per_epoch}, validation_steps={validation_steps}"
    )

    model = build_model(config)
    model.summary(print_fn=logger.info)
    verify_bias_free(model)

    # Optional warm-start from a saved .keras checkpoint (e.g. self-iterate
    # fine-tuning on top of a normally-trained denoiser). The functional model is
    # already built, so layer-by-layer transfer works. skip_prefixes=() loads ALL
    # layers (the denoiser has no head_ layers to skip, unlike CliffordNetUNet).
    if config.init_from is not None:
        logger.info(f"Initializing weights from checkpoint: {config.init_from}")
        report = load_weights_from_checkpoint(
            model,
            ckpt_path=config.init_from,
            skip_prefixes=(),
        )
        if len(report.loaded) == 0:
            raise ValueError(
                f"init_from loaded 0 layers from {config.init_from} — architecture "
                "mismatch? Ensure --variant/--convnext-version/--patch-size/gabor/"
                "laplacian flags match the checkpoint's model."
            )
        logger.info(
            f"init_from: loaded {len(report.loaded)} layer(s), "
            f"missing_in_source {len(report.missing_in_source)}, "
            f"shape_mismatch {len(report.shape_mismatch)}."
        )
        if report.shape_mismatch:
            logger.warning(
                f"init_from: {len(report.shape_mismatch)} layer(s) had shape "
                "mismatches and were left at init — check architecture flags."
            )
        verify_bias_free(model)  # re-check: transfer must not introduce bias

    # Keras requires a loss for every output, but the bottleneck is upstream of the decoder
    # so the denoiser MSE already trains it (F5/F6). Fit a single-output view that SHARES
    # weight objects with the full model; the full 2-output model is saved/monitored separately.
    if config.expose_bottleneck:
        # Robust filter: exclude ONLY the 'bottleneck' output, keep final_output (+ any DS outputs).
        bottleneck_tensor = model.get_layer("bottleneck").output
        train_outputs = [out for out in model.outputs if out is not bottleneck_tensor]
        # Use the single input TENSOR (model.input), not model.inputs (a 1-element LIST):
        # a list input-structure makes Keras warn "structure of `inputs` doesn't match"
        # on every bare-tensor call (fit + viz). Unwrap when there is exactly one input.
        train_inputs = model.inputs[0] if len(model.inputs) == 1 else model.inputs
        train_model = keras.Model(
            train_inputs,
            train_outputs[0] if len(train_outputs) == 1 else train_outputs,
            name=f"{model.name}_train_view",
        )
        logger.info(
            f"expose_bottleneck: fitting single-output training view "
            f"({len(train_outputs)} loss output(s)); full model has {len(model.outputs)} outputs."
        )
    else:
        train_model = model

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
    # Compile/fit operate on train_model (the single-output view when expose_bottleneck;
    # otherwise train_model IS model). The view shares weight objects with the full model.
    train_model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            "mae",
            PsnrMetric(max_val=2.0, name="psnr_metric"),
            SsimMetric(max_val=2.0, name="ssim_metric"),
        ],
    )
    logger.info(f"Model compiled with {model.count_params():,} parameters")

    disable_early_stopping = config.early_stopping_patience <= 0
    # ModelAnalyzer (opt-in): data-free weight + spectral analysis at the epoch cadence.
    # Calibration / information-flow / training-dynamics are data-dependent and
    # classification-oriented, so they are OFF for this image-to-image denoiser. The
    # EpochAnalyzerCallback attaches to the fitted model (the single-output training view
    # when expose_bottleneck is on), which shares all weights with the full model.
    analyzer_config = (
        AnalysisConfig(
            analyze_weights=True,
            analyze_spectral=True,
            analyze_calibration=False,
            analyze_information_flow=False,
            analyze_training_dynamics=False,
            verbose=False,
        )
        if config.enable_analyzer
        else None
    )
    callbacks, _ = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="convunext_denoiser",
        run_dir=str(output_dir),
        monitor="val_loss",
        patience=config.early_stopping_patience if not disable_early_stopping else 1,
        use_lr_schedule=True,
        include_terminate_on_nan=True,
        include_tensorboard=True,
        include_analyzer=config.enable_analyzer,
        analyzer_config=analyzer_config,
        analyzer_start_epoch=config.analyzer_start_epoch,
        analyzer_epoch_frequency=config.analyzer_freq,
    )
    if disable_early_stopping:
        # Curriculum learning makes val_loss non-monotonic, so early stopping on it would
        # cut training short. Strip the EarlyStopping callback; the full epoch budget runs
        # and ModelCheckpoint still writes best_model.keras.
        callbacks = [
            cb for cb in callbacks
            if not isinstance(cb, keras.callbacks.EarlyStopping)
        ]
        logger.info(
            "Early stopping disabled (early_stopping_patience <= 0); "
            "training the full schedule."
        )
    # Prepend so it runs BEFORE CSVLogger -> 'lr' lands in training_log.csv.
    callbacks.insert(0, LRLoggerCallback())
    callbacks.append(
        NoiseSigmaCurriculumCallback(
            sigma_max_var=sigma_max_var,
            sigma_max_start=config.sigma_max_start,
            sigma_max_end=config.sigma_max_end,
            total_epochs=config.curriculum_epochs,
            schedule=config.curriculum_schedule,
        )
    )
    # Self-iterate regeneration callback: appended AFTER the curriculum callback so
    # the curriculum's per-epoch sigma .assign runs first and get_sigma reads the
    # advanced value when refilling fresh pool slots. KEEP the curriculum callback
    # active in self-iterate mode: the pool dataset has no tf.data noise closure, so
    # the curriculum Variable is consumed ONLY by get_sigma here (intended).
    if self_iterate_callback is not None:
        callbacks.append(self_iterate_callback)

    # Denoising visualization: same images under 3 noise regimes.
    viz_batch = build_fixed_val_batch(val_paths, config, n=config.viz_samples)
    callbacks.append(
        DenoisingVisualizationCallback(
            clean_batch=viz_batch,
            sigma_max_var=sigma_max_var,
            noise_sigma_min=config.noise_sigma_min,
            out_dir=output_dir,
            freq=config.viz_freq,
            max_samples=config.viz_samples,
            val_ds=val_ds,
            validation_steps=validation_steps,
            noise_type=config.noise_type,
            composite_additive_ratio=config.composite_additive_ratio,
        )
    )

    # Bottleneck health monitor: only when the full model exposes the bottleneck
    # (reads the trailing bottleneck output from the FULL model, not the view).
    if config.expose_bottleneck:
        callbacks.append(
            ConvUnextBottleneckMonitorCallback(
                full_model=model,
                val_batch=viz_batch,
                output_dir=output_dir / "visualizations",  # share the existing viz dir
                monitor_freq=config.viz_freq,
                max_featuremap_channels=64,  # two 8x8 grids: first-64 + top-64 energy
            )
        )

    # DECISION plan_2026-06-20_73c91aad/D-001: WW-PGD is a callback (epoch-boundary
    # projection), passed the FULL model to dodge the train-view; default-OFF opt-in.
    if config.ww_pgd:
        # DECISION plan_2026-06-20_5fe67f0c/D-003: the alpha-trajectory CSV path is
        # supplied per-run as a callback ctor arg (NOT serialized into WWTailConfig),
        # so a run-specific filesystem path never leaks into the model/callback config.
        # csv_path stays None unless --ww-pgd-log-alpha is set, keeping the ww_pgd ON
        # path byte-identical when logging is off.
        ww_pgd_csv_path = (
            str(output_dir / "ww_pgd_layer_alpha.csv")
            if config.ww_pgd_log_alpha else None
        )
        callbacks.append(
            WWPGDProjectionCallback(
                config=WWTailConfig(
                    enable=True,
                    warmup_epochs=config.ww_pgd_warmup_epochs,
                    ramp_epochs=config.ww_pgd_ramp_epochs,
                    apply_every_epochs=config.ww_pgd_apply_every_epochs,
                    q=config.ww_pgd_q,
                    blend_eta=config.ww_pgd_blend_eta,
                    cayley_eta=config.ww_pgd_cayley_eta,
                    min_tail=config.ww_pgd_min_tail,
                    log_layer_stats=config.ww_pgd_log_alpha,
                ),
                num_epochs=config.epochs,
                model=model,  # the FULL model, NOT the train_model view
                csv_path=ww_pgd_csv_path,
            )
        )
        logger.info("WW-PGD spectral tail-projection ENABLED (epoch-boundary).")
        if config.ww_pgd_log_alpha:
            logger.info(f"WW-PGD alpha trajectory logging -> {ww_pgd_csv_path}")

    start = time.time()
    # DECISION plan_2026-06-20_88705c63/D-005: in self-iterate mode pass
    # steps_per_epoch=None to model.fit. The self-iterate train_ds is a FINITE
    # `from_generator` pool dataset sized to exactly `steps_per_epoch` batches
    # (no `.repeat()`). Passing `steps_per_epoch` alongside a finite dataset makes
    # Keras pull that many batches in epoch 1, EXHAUST the dataset, then hit
    # OUT_OF_RANGE ("input ran out of data") with loss=0.0 in epoch 2+ -- only
    # epoch 1 trains (the Bug-B smoke regression). With steps_per_epoch=None Keras
    # consumes the full finite dataset each epoch via a FRESH per-epoch iterator,
    # which BOTH fixes the exhaustion AND is exactly the fresh-iterator condition
    # the D-004 pool re-read mechanism depends on (the callback's in-place pool
    # mutation is re-read next epoch). Do NOT "fix" this with `.repeat()` on the
    # pool dataset: `.repeat()` keeps ONE iterator alive across epochs (spike
    # TEST-5 "same iter" stayed STALE), so callback pool mutations would NOT be
    # re-read -> silently breaks regeneration. The OFF path is unchanged: it keeps
    # passing its `steps_per_epoch` exactly as before. The `steps_per_epoch` value
    # is still used above for the LR decay_steps/warmup_steps math in BOTH modes.
    # validation_steps governs the val side independently of this arg. See D-005.
    fit_steps_per_epoch = None if config.self_iterate else steps_per_epoch
    history = train_model.fit(
        train_ds,
        epochs=config.epochs,
        steps_per_epoch=fit_steps_per_epoch,
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

    if config.expose_bottleneck:
        full_path = output_dir / "final_model_bottleneck.keras"
        model.save(full_path)
        logger.info(f"Saved full 2-output model (denoised, bottleneck) -> {full_path}")

    final_path = output_dir / "final_model.keras"
    model.save(final_path)
    logger.info(f"Saved final (last-epoch) model -> {final_path}")

    gc.collect()
    return model


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train bias-free ConvUNeXt denoiser (Gabor stem + noise curriculum)",
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
    parser.add_argument("--warmup-epochs", type=int, default=None,
                        help="LR warmup length (default: 10%% of --epochs)")
    parser.add_argument("--no-gabor-stem", action="store_true",
                        help="Disable the frozen Gabor depthwise stem")
    parser.add_argument("--laplacian-pyramid", action="store_true",
                        help="Enable the Laplacian-pyramid downsample/skip path (default OFF)")
    parser.add_argument("--expose-bottleneck", action="store_true",
                        help="Expose the bottleneck latent as an optional second model output (default OFF)")
    parser.add_argument("--analyzer", action="store_true",
                        help="Run ModelAnalyzer (data-free weight + spectral) during training (default OFF)")
    parser.add_argument("--analyzer-freq", type=int, default=10,
                        help="Run the analyzer every N epochs (with --analyzer)")
    parser.add_argument("--gabor-filters", type=int, default=32)
    parser.add_argument("--sigma-max-start", type=float, default=0.05)
    parser.add_argument("--sigma-max-end", type=float, default=0.5)
    parser.add_argument("--curriculum-schedule",
                        choices=["linear", "cosine", "exp"], default="linear")
    parser.add_argument("--curriculum-epochs", type=int, default=None)
    parser.add_argument("--deep-supervision", action="store_true")
    parser.add_argument("--viz-freq", type=int, default=5,
                        help="Save clean/noisy/denoised grids every N epochs")
    parser.add_argument("--viz-samples", type=int, default=8,
                        help="Number of image columns in the eval grid")
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
        "--multiplicative-noise", action="store_true",
        help="Opt-in: corrupt patches with per-pixel multiplicative Gaussian noise "
             "y=x*(1+N(0,1)*sigma) instead of additive AWGN. Default OFF (additive).",
    )
    parser.add_argument(
        "--composite-noise", action="store_true",
        help="Opt-in: corrupt patches with composite noise y=x*n+a (multiplicative "
             "n~N(1,sigma_m^2) plus additive a~N(0,sigma_a^2)). Takes precedence over "
             "--multiplicative-noise. Default OFF (additive).",
    )
    parser.add_argument(
        "--composite-additive-ratio", type=float, default=0.5,
        help="Composite mode only: additive floor as a fraction of the curriculum "
             "sigma_m (sigma_a = ratio * sigma_m). Must be > 0. Default 0.5.",
    )
    parser.add_argument(
        "--self-iterate", action="store_true",
        help="Opt-in: train the denoiser to be self-iterable (2-5 sequential passes "
             "improve PSNR instead of over-smoothing) via epoch-boundary regeneration "
             "over a bounded RAM patch pool. Additive noise ONLY (rejected with "
             "--multiplicative-noise/--composite-noise). Default OFF (streaming "
             "pipeline is byte-identical when off).",
    )
    parser.add_argument(
        "--self-iterate-pool-size", type=int, default=2048,
        help="Self-iterate mode only: number of clean patches in the RAM pool whose "
             "inputs are regenerated at epoch cadence. Must be >= batch_size. "
             "Default 2048 (~1.6GB at 256x256x3).",
    )
    parser.add_argument(
        "--self-iterate-regen-freq", type=int, default=1,
        help="Self-iterate mode only: regenerate the pool inputs every N epochs "
             "(model.predict over the pool). Default 1 (every epoch).",
    )
    parser.add_argument(
        "--self-iterate-mix-ratio", type=float, default=0.5,
        help="Self-iterate mode only: fraction of pool slots filled with regenerated "
             "(f(prev)->clean) pairs; the rest get fresh (clean+noise->clean). "
             "0.0 = fresh only, 1.0 = regenerated only. Default 0.5 (union).",
    )
    parser.add_argument(
        "--ww-pgd", action="store_true",
        help="Enable WW-PGD spectral tail-projection at epoch boundaries (default OFF).",
    )
    parser.add_argument(
        "--ww-pgd-log-alpha", action="store_true",
        help="Log a per-epoch, per-layer power-law alpha trajectory CSV "
             "(ww_pgd_layer_alpha.csv) to the experiment dir. Implies --ww-pgd "
             "(turns on the projection if not already set). Default OFF.",
    )
    parser.add_argument(
        "--init-from", type=str, default=None,
        help="Warm-start model weights from a saved .keras checkpoint before training. "
             "Primary use: self-iterate FINE-TUNING on top of a normally-trained "
             "denoiser. Architecture (variant/convnext-version/patch-size/gabor/"
             "laplacian) must match the checkpoint. Default: random init.",
    )
    parser.add_argument(
        "--depthwise-initializer", type=str, default=None,
        help="Opt-in initializer for the ConvNeXt depthwise kernels. The alias "
             "'orthonormal' maps to keras Orthogonal(gain=1.0) (unit-norm on the "
             "(K,K,C,1) depthwise shape); any other value passes through to "
             "keras.initializers.get(). Default None = byte-identical OFF "
             "(blocks keep their hardcoded TruncatedNormal(0,0.02)).",
    )
    parser.add_argument(
        "--depthwise-l2", type=float, default=None,
        help="Opt-in L2 weight on the ConvNeXt depthwise kernels (wired to "
             "keras.regularizers.L2). Default None = off (blocks keep their "
             "deepcopy(kernel_regularizer) default).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Tiny end-to-end mechanism check (few steps/epochs, constant LR).",
    )
    args = parser.parse_args()

    # DECISION plan_2026-06-20_88705c63/D-003: self-iterate is theory-bound to ADDITIVE
    # Gaussian noise. The Miyasawa residual=score identity (and the clean-image fixed
    # point that makes 2-5 passes non-decreasing) holds for additive noise ONLY;
    # multiplicative/composite noise breaks the linear-domain identity, so the
    # self-iterate objective has no theoretical justification there (decisions.md D-003,
    # research/miyasawas_theorem_multiplicative.md). Reject the combination at PARSE time
    # (fail fast, before any GPU/data work) rather than only at TrainingConfig.__post_init__.
    # Do NOT relax this to a warn-and-continue: a silently-wrong objective would train a
    # model that degrades under self-iteration with no error. Both guards are intentional
    # (belt-and-suspenders): parse-time error here for CLI users, ValueError in
    # __post_init__ for programmatic config construction.
    if args.self_iterate and (args.multiplicative_noise or args.composite_noise):
        parser.error(
            "--self-iterate requires additive noise; it is incompatible with "
            "--multiplicative-noise/--composite-noise (Miyasawa residual=score "
            "identity is additive-only)"
        )

    return args


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
            use_laplacian_pyramid=args.laplacian_pyramid,
            expose_bottleneck=args.expose_bottleneck,
            enable_deep_supervision=args.deep_supervision,
            enable_analyzer=args.analyzer,
            analyzer_freq=args.analyzer_freq,
            gabor_filters=8,
            epochs=2,
            curriculum_epochs=2,
            batch_size=2,
            patch_size=64,
            channels=3,
            patches_per_image=2,
            max_train_files=8,
            max_val_files=8,  # >= viz_samples so the smoke grid also shows 8 columns
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
            noise_type=(
                "composite" if args.composite_noise
                else "multiplicative" if args.multiplicative_noise
                else "additive"
            ),
            composite_additive_ratio=args.composite_additive_ratio,
            # Self-iterate: SMALL pool so a smoke run exercises >=1 regeneration cheaply.
            # Cap at 32 (>= smoke batch_size 2) and force regen_freq=1 so the single
            # epoch boundary triggers a regeneration; mix_ratio is honored from args.
            self_iterate=args.self_iterate,
            self_iterate_pool_size=min(args.self_iterate_pool_size, 32),
            self_iterate_regen_freq=1,
            self_iterate_mix_ratio=args.self_iterate_mix_ratio,
            ww_pgd=args.ww_pgd,
            ww_pgd_log_alpha=args.ww_pgd_log_alpha,
            init_from=args.init_from,
            depthwise_initializer=args.depthwise_initializer,
            depthwise_l2=args.depthwise_l2,
            viz_freq=1,
            viz_samples=args.viz_samples,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name or "convunext_denoiser_smoke",
        )
    else:
        config = TrainingConfig(
            variant=args.variant,
            convnext_version=args.convnext_version,
            use_gabor_stem=not args.no_gabor_stem,
            use_laplacian_pyramid=args.laplacian_pyramid,
            expose_bottleneck=args.expose_bottleneck,
            enable_analyzer=args.analyzer,
            analyzer_freq=args.analyzer_freq,
            gabor_filters=args.gabor_filters,
            enable_deep_supervision=args.deep_supervision,
            epochs=args.epochs,
            curriculum_epochs=args.curriculum_epochs,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            channels=args.channels,
            patches_per_image=args.patches_per_image,
            learning_rate=args.learning_rate,
            warmup_epochs=args.warmup_epochs,  # None -> 10% of epochs
            sigma_max_start=args.sigma_max_start,
            sigma_max_end=args.sigma_max_end,
            curriculum_schedule=args.curriculum_schedule,
            noise_type=(
                "composite" if args.composite_noise
                else "multiplicative" if args.multiplicative_noise
                else "additive"
            ),
            composite_additive_ratio=args.composite_additive_ratio,
            self_iterate=args.self_iterate,
            self_iterate_pool_size=args.self_iterate_pool_size,
            self_iterate_regen_freq=args.self_iterate_regen_freq,
            self_iterate_mix_ratio=args.self_iterate_mix_ratio,
            ww_pgd=args.ww_pgd,
            ww_pgd_log_alpha=args.ww_pgd_log_alpha,
            init_from=args.init_from,
            depthwise_initializer=args.depthwise_initializer,
            depthwise_l2=args.depthwise_l2,
            max_train_files=args.max_train_files or 10000,
            max_val_files=args.max_val_files or 500,
            steps_per_epoch=args.steps_per_epoch,
            validation_steps=args.validation_steps if args.validation_steps is not None else 100,
            viz_freq=args.viz_freq,
            viz_samples=args.viz_samples,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
        )

    # --ww-pgd-log-alpha implies --ww-pgd: enabling the alpha trajectory log turns
    # on the projection it instruments (so the flag is usable standalone).
    if config.ww_pgd_log_alpha:
        config.ww_pgd = True

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
