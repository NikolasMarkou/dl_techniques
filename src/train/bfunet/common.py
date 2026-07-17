"""Shared building blocks for the bfunet denoiser trainers (train_convunext_denoiser.py, train_cliffordunet_denoiser.py, and train_unet_denoiser.py). Data pipeline, curriculum noise, eval/PSNR helpers, dashboard, and callbacks live here once; each trainer imports/re-exports them."""

import gc
import csv
import json
import time
import keras
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
    augment_patch,
    create_callbacks as create_common_callbacks,
    save_config_json,
    set_seeds,
    validate_model_loading,
    collect_image_paths,
)
from train.superpoint.homographic_adaptation import select_weighted_image_paths
from dl_techniques.metrics.psnr_metric import PsnrMetric
from dl_techniques.metrics.ssim_metric import SsimMetric
from dl_techniques.losses.jacobian_symmetry import jacobian_symmetry_penalty
from dl_techniques.analyzer import AnalysisConfig
from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.utils.denoiser_provenance import require_unit_domain_checkpoint
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
# DATA DOMAIN (single source of truth for every clip/normalize/probe bound)
# ---------------------------------------------------------------------

# DECISION plan_2026-07-12_e56909cd/D-001: the pixel domain is [0,1], NOT the legacy
# zero-centered [-0.5,+0.5]. A bias-free network is degree-1 homogeneous with f(0)=0
# (structurally, not by learning). On a zero-centered domain a flat mid-grey patch IS
# the zero vector, so the net reproduces it FOR FREE -- the DC component, the very thing
# a denoiser's local filters must preserve, is never supervised at its most important
# operating point, and sum-to-one filters are never learned. On [0,1] a flat patch of
# value c is c*1, and homogeneity gives f(c*1) = c*f(1); reproducing it REQUIRES
# f(1) = 1, i.e. local weights that sum to one -- exactly the DC-preserving property the
# Miyasawa/Tweedie residual=score extraction depends on.
#
# CRITICAL: this is a pure DC SHIFT, NOT a rescale. Peak-to-peak width is 1.0 in BOTH
# domains, so sigma (`sigma_255 = sigma*255`, `noise_sigma_min`, `sigma_max_start/end`),
# `PsnrMetric(max_val=1.0)`, `SsimMetric(max_val=1.0)` and `_mean_psnr`'s 20*log10(1/rmse)
# are ALL still exactly correct. Do NOT "fix" any of them to match the new domain -- that
# would silently corrupt every reported dB number and nothing would fail. Do NOT add a
# domain switch / `pixel_domain` kwarg / compat shim either: legacy [-0.5,+0.5]
# checkpoints are knowingly invalidated (no partial-migration state works, INV-1).
# See plans/plan_2026-07-12_e56909cd/decisions.md D-001.
DATA_MIN: float = 0.0
DATA_MAX: float = 1.0


def decode_full_image(
    image_path: tf.Tensor, config: "BFUnetTrainingConfig"
) -> tf.Tensor:
    """Read + decode an image ONCE and normalize to [0, 1].

    Returns the full (variable-size) image, upscaled if smaller than the patch.
    This is the expensive step (HDD read + JPEG decode); the streaming pipeline
    calls it once per image and crops ``patches_per_image`` patches from the
    result, instead of re-reading/re-decoding the same file per patch.
    """
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_image(
        image_string, channels=config.channels, expand_animations=False
    )
    image.set_shape([None, None, config.channels])
    image = tf.cast(image, tf.float32)

    # Normalize to [0, 1] (DATA_MIN/DATA_MAX). THE single normalization site for the
    # whole training pipeline -- every other loader composes this one. Do NOT add a
    # second normalizer anywhere (INV-3).
    image = image / 255.0

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

    return tf.cond(
        tf.logical_or(height < min_size, width < min_size),
        true_fn=_upscale,
        false_fn=lambda: image,
    )


def random_crop_patch(image: tf.Tensor, config: "BFUnetTrainingConfig") -> tf.Tensor:
    """Extract one random ``patch_size`` crop from an already-decoded image."""
    return tf.image.random_crop(
        image, [config.patch_size, config.patch_size, config.channels]
    )


def load_and_preprocess_image(
    image_path: tf.Tensor, config: "BFUnetTrainingConfig"
) -> tf.Tensor:
    """Decode an image, normalize to [0, 1], and crop a single random patch.

    Thin compose of ``decode_full_image`` + ``random_crop_patch``, kept for the
    single-patch callers (self-iterate pool seed, fixed val-batch). The streaming
    ``create_dataset`` decodes once and crops many instead of calling this per patch.
    """
    return random_crop_patch(decode_full_image(image_path, config), config)


def collect_training_paths(config: "BFUnetTrainingConfig") -> List[str]:
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
    config: "BFUnetTrainingConfig",
    noise_fn,
    is_training: bool,
) -> tf.data.Dataset:
    """Build a tf.data pipeline of (noisy, clean) [0,1] patch pairs."""
    if not file_paths:
        raise ValueError("No image files found for the dataset")

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    if is_training:
        dataset = dataset.shuffle(
            buffer_size=min(config.dataset_shuffle_buffer, len(file_paths)),
            reshuffle_each_iteration=True,
        )
    dataset = dataset.repeat()

    # Decode each image ONCE — read + JPEG-decode is the bottleneck (datasets live on
    # a spinning HDD). `deterministic=False` lets fast reads flow to the GPU without
    # waiting on slower reads stuck behind HDD seek latency; AUTOTUNE scales the read
    # parallelism to keep the GPU fed (replaces the fixed `parallel_reads`).
    dataset = dataset.map(
        lambda p: decode_full_image(p, config),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    # Drop blank/corrupt decodes once, on the full image, before cropping.
    # SEMANTIC FLIP under [0,1] (no code change needed): this drops flat BLACK images
    # (the value that maps to 0), where on the legacy [-0.5,+0.5] domain it dropped flat
    # MID-GREY ones. Both are degenerate; black is the more sensible thing to drop.
    dataset = dataset.filter(lambda x: tf.reduce_sum(tf.abs(x)) > 0)

    if is_training and config.patches_per_image > 1:
        # Crop `patches_per_image` patches from the SAME decoded image (no re-read /
        # re-decode). This is the throughput fix: previously the path was repeated and
        # the file was read + decoded once PER patch.
        ppi = config.patches_per_image
        dataset = dataset.flat_map(
            lambda img: tf.data.Dataset.from_tensors(img)
            .repeat(ppi)
            .map(lambda im: random_crop_patch(im, config))
        )
        # The ppi crops above are consecutive same-image patches; shuffle the patch
        # tensors so a batch of size <= ppi is not all crops of one image. Buffer is
        # patch tensors now (not path strings), so it costs real host RAM
        # (~patch_shuffle_buffer * patch_size^2 * channels * 4 bytes).
        if config.patch_shuffle_buffer > 1:
            dataset = dataset.shuffle(
                buffer_size=config.patch_shuffle_buffer,
                reshuffle_each_iteration=True,
            )
    else:
        dataset = dataset.map(
            lambda img: random_crop_patch(img, config),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.map(
        lambda x: tf.ensure_shape(
            x, [config.patch_size, config.patch_size, config.channels]
        )
    )

    if is_training and config.augment_data:
        dataset = dataset.map(augment_patch, num_parallel_calls=tf.data.AUTOTUNE)

    # Clip the clean patch back to [DATA_MIN, DATA_MAX] after augmentation. flips/rot90
    # preserve range, but the aspect-safe bilinear upscale (small images) can overshoot;
    # the clean patch is both the model input and the regression target, so keep it in range.
    dataset = dataset.map(
        lambda x: tf.clip_by_value(x, DATA_MIN, DATA_MAX),
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
    config: "BFUnetTrainingConfig",
    sigma_init: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a FIXED bounded RAM pool of clean patches + an initial noisy input pool.

    Loads ``config.self_iterate_pool_size`` clean patches ONCE (reusing the same
    ``load_and_preprocess_image`` load+crop logic the streaming pipeline uses, plus
    the same clip-to-[0,1] convention) into ``clean_pool``, then allocates a
    mutable ``current_input`` = ``clip(clean + N(0, sigma_init), 0, 1)`` as the
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
        in ``[0, 1]``; ``clean_pool`` is the fixed target, ``current_input`` the
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
            patch = tf.clip_by_value(patch, DATA_MIN, DATA_MAX)
            patch_np = np.asarray(patch, dtype=np.float32)
            if not np.any(np.abs(patch_np) > 0):
                # Mirror the streaming filter() that drops all-zero patches. Same
                # semantic flip as create_dataset's filter: under [0,1] this drops flat
                # BLACK patches, not flat mid-grey ones. No code change needed.
                continue
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
        clean_pool + noise * float(sigma_init), DATA_MIN, DATA_MAX
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
    config: "BFUnetTrainingConfig",
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


def _denorm(img: np.ndarray) -> np.ndarray:
    """Map a model-domain image to [0, 1] for display.

    Under the [0,1] data domain (D-001) the model domain IS the display domain, so this
    collapses to a defensive clip (a model output can overshoot the domain). Kept as the
    ONE denormalizer so display call sites never inline their own conversion.
    """
    return np.clip(img, DATA_MIN, DATA_MAX)


def expected_noise_variance(sigma_min: float, sigma_max: float) -> float:
    """``E[sigma^2]`` for ``sigma ~ U(sigma_min, sigma_max)`` = ``(a^2 + ab + b^2)/3``.

    This is the per-epoch NOISE FLOOR: the MSE a do-nothing identity denoiser would
    score against the clean target. It is the natural difficulty normalizer for the
    curriculum (the training noise fn draws a fresh ``sigma`` per image from exactly
    this uniform range -- see ``make_curriculum_noise_fn``).

    Slight over-estimate in practice: the pipeline clips the noisy patch to
    ``[DATA_MIN, DATA_MAX]``, which removes a little noise energy near the boundaries.
    So the reported denoising gain is a mild UNDER-estimate. Additive noise only.
    """
    a, b = float(sigma_min), float(sigma_max)
    if b < a:
        a, b = b, a
    return (a * a + a * b + b * b) / 3.0


def expected_input_psnr(sigma_min: float, sigma_max: float) -> float:
    """Mean per-image PSNR (dB) of the NOISY INPUT for ``sigma ~ U(sigma_min, sigma_max)``.

    Per-image PSNR at ``max_val=1`` is ``-20*log10(sigma)``, and ``PsnrMetric`` averages
    per-image PSNR, so the reference is ``E[-20*log10(sigma)]``, NOT ``-20*log10(E[sigma])``
    (Jensen -- the two differ). With ``E[ln s] = (b(ln b - 1) - a(ln a - 1)) / (b - a)``.

    The ``a = 0`` case is finite even though a single ``sigma -> 0`` image has infinite
    PSNR: the limit of ``a*ln(a)`` is 0, giving ``E[ln s] -> ln(b) - 1``.
    """
    a, b = float(sigma_min), float(sigma_max)
    if b <= 0.0:
        return float("nan")
    if a <= 0.0:
        mean_ln = float(np.log(b)) - 1.0
    elif abs(b - a) < 1e-12:
        mean_ln = float(np.log(a))
    else:
        mean_ln = (b * (np.log(b) - 1.0) - a * (np.log(a) - 1.0)) / (b - a)
    return float(-20.0 / np.log(10.0) * mean_ln)


def _ratio_or_nan(num: List[float], den: List[float]) -> List[float]:
    """Elementwise ``num/den``, mapping non-finite / non-positive entries to NaN."""
    out = []
    for n, d in zip(num, den):
        n, d = float(n), float(d)
        if not (np.isfinite(n) and np.isfinite(d)) or n <= 0.0 or d <= 0.0:
            out.append(float("nan"))
        else:
            out.append(n / d)
    return out


def _curriculum_end_epoch(
    ep: List[int], sigma_max: List[float], final_sigma: float
) -> Optional[int]:
    """First epoch at which the curriculum has reached its final ``sigma_max``.

    After this epoch the TRAIN task stops moving and matches the (always-fixed) VAL
    task, so train/val curves become directly comparable. Returns None if the ramp
    never completes within the recorded epochs.
    """
    tol = 1e-6 + 1e-3 * abs(float(final_sigma))
    for e, s in zip(ep, sigma_max):
        if np.isfinite(s) and abs(float(s) - float(final_sigma)) <= tol:
            return int(e)
    return None


def render_training_dashboard(
    history: dict,
    out_path: Path,
    title: str = "",
    sigma_min: Optional[float] = None,
    val_sigma_max: Optional[float] = None,
    additive: bool = True,
) -> None:
    """Render a single combined dashboard PNG of per-epoch training curves.

    **Why this is not just "loss goes down".** Training uses a NOISE CURRICULUM: the
    per-image ``sigma`` is drawn from ``U(noise_sigma_min, sigma_max(epoch))`` and
    ``sigma_max`` RAMPS UP each epoch (``NoiseSigmaCurriculumCallback``). So the TRAIN
    task gets strictly HARDER over time, and raw train MSE / PSNR are confounded by
    task difficulty -- train loss can rise, or plateau, while the model is genuinely
    improving. VALIDATION does NOT ramp (``sigma_fixed_var = sigma_max_end``, a
    stationary task), so val curves ARE comparable across epochs but sit on a much
    harder task than early-epoch train.

    Reading raw train-vs-val on shared axes is therefore actively misleading. The
    dashboard fixes this by plotting, alongside the raw curves, the DIFFICULTY-
    NORMALIZED ones: each split's MSE divided by ITS OWN noise floor ``E[sigma^2]``
    (``expected_noise_variance``). Those normalized curves remove the curriculum and
    should improve monotonically; the raw ones need not.

    Eight panels. Top row = RAW (task-confounded, with noise-floor references):
    (1) MSE, (2) MSE (log), (3) PSNR, (4) denoising gain (dB).
    Bottom row = NORMALIZED + schedules: (5) residual noise fraction (log),
    (6) MAE, (7) sigma_max curriculum (with a [0,255] twin axis), (8) LR (log).

    ``history`` keys (any may be absent -> that panel/series is skipped):
    ``epoch, loss, val_loss, mae, val_mae, psnr, val_psnr, sigma_max, lr``.

    Args:
        history: Per-epoch scalar lists (see above).
        out_path: PNG destination.
        title: Figure suptitle.
        sigma_min: ``config.noise_sigma_min`` -- lower bound of the sampled sigma range.
        val_sigma_max: The FIXED upper bound used by the val pipeline
            (``config.sigma_max_end``). Together with ``sigma_min`` these enable the
            noise-floor references and the normalized panels.
        additive: Whether the run uses additive Gaussian noise. The ``E[sigma^2]``
            noise floor is an ADDITIVE-noise quantity, so for multiplicative/composite
            runs the floor references and normalized panels are SKIPPED rather than
            silently plotted wrong.
    """
    ep = history.get("epoch")
    if not ep:
        return

    sig = history.get("sigma_max")
    # Noise-floor machinery needs the sigma range AND additive noise. Anything missing
    # -> degrade gracefully to the raw-only dashboard rather than plotting a wrong floor.
    floors_ok = (
        additive
        and sig is not None
        and sigma_min is not None
        and val_sigma_max is not None
        and len(sig) >= len(ep)
    )

    train_floor = val_floor = None
    train_in_psnr = val_in_psnr = None
    ramp_end = None
    if floors_ok:
        train_floor = [expected_noise_variance(sigma_min, s) for s in sig[:len(ep)]]
        val_floor = [expected_noise_variance(sigma_min, val_sigma_max)] * len(ep)
        train_in_psnr = [expected_input_psnr(sigma_min, s) for s in sig[:len(ep)]]
        val_in_psnr = [expected_input_psnr(sigma_min, val_sigma_max)] * len(ep)
        ramp_end = _curriculum_end_epoch(ep, sig, val_sigma_max)

    fig, axes = plt.subplots(2, 4, figsize=(21, 9.5))
    if title:
        fig.suptitle(title, fontsize=15, y=0.99)

    TRAIN_C, VAL_C = "#d62728", "#1f77b4"

    def _line(ax, ys, label, **kw):
        if ys is None:
            return
        n = min(len(ep), len(ys))
        ax.plot(ep[:n], ys[:n], label=label, lw=1.6, **kw)

    def _floor(ax, ys, label, color):
        """A noise-floor reference: what an identity (do-nothing) denoiser would score."""
        if ys is None:
            return
        n = min(len(ep), len(ys))
        ax.plot(ep[:n], ys[:n], label=label, lw=1.2, ls=":", alpha=0.75, color=color)

    def _mark_ramp(ax):
        """Shade the curriculum ramp; past its end the train task is stationary too."""
        if ramp_end is None or ramp_end <= ep[0]:
            return
        ax.axvspan(ep[0], ramp_end, color="#2ca02c", alpha=0.06, zorder=0)
        ax.axvline(ramp_end, color="#2ca02c", ls="--", lw=1.0, alpha=0.6, zorder=0)

    # ---- Top row: RAW curves (task-confounded) + noise-floor references ----

    # (1) MSE (linear). loss IS mse (compiled loss='mse').
    ax = axes[0, 0]
    _line(ax, history.get("loss"), "train (curriculum sigma)", color=TRAIN_C)
    _line(ax, history.get("val_loss"), "val (fixed sigma)", color=VAL_C)
    _floor(ax, train_floor, "train noise floor E[s^2]", TRAIN_C)
    _floor(ax, val_floor, "val noise floor E[s^2]", VAL_C)
    _mark_ramp(ax)
    ax.set_title("MSE per epoch (RAW - train task is MOVING)")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7)

    # (2) MSE (log y). The floor gap IS the denoising gain, read directly.
    ax = axes[0, 1]
    _line(ax, history.get("loss"), "train", color=TRAIN_C)
    _line(ax, history.get("val_loss"), "val", color=VAL_C)
    _floor(ax, train_floor, "train noise floor", TRAIN_C)
    _floor(ax, val_floor, "val noise floor", VAL_C)
    _mark_ramp(ax)
    ax.set_yscale("log")
    ax.set_title("MSE per epoch (log) - gap to floor = gain")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (log)")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=7)

    # (3) PSNR + the mean per-image PSNR of the NOISY INPUT (the do-nothing baseline).
    ax = axes[0, 2]
    _line(ax, history.get("psnr"), "train", color=TRAIN_C)
    _line(ax, history.get("val_psnr"), "val", color=VAL_C)
    _floor(ax, train_in_psnr, "train input PSNR", TRAIN_C)
    _floor(ax, val_in_psnr, "val input PSNR", VAL_C)
    _mark_ramp(ax)
    ax.set_title("PSNR per epoch (RAW) vs noisy-input PSNR")
    ax.set_xlabel("epoch"); ax.set_ylabel("PSNR (dB)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7)

    # (4) DENOISING GAIN (dB) = 10*log10(E[sigma^2] / MSE). THE curriculum-free curve:
    # each split is normalized by ITS OWN noise floor, so this measures how much noise
    # energy the model removes -- independent of how hard the epoch's task was. THIS is
    # the panel to read for "is the model actually getting better?".
    ax = axes[0, 3]
    if floors_ok:
        for key, floor, lab, c in (
            ("loss", train_floor, "train", TRAIN_C),
            ("val_loss", val_floor, "val", VAL_C),
        ):
            mse = history.get(key)
            if mse is None:
                continue
            n = min(len(ep), len(mse), len(floor))
            gain = [10.0 * float(np.log10(r)) if np.isfinite(r) else float("nan")
                    for r in _ratio_or_nan(floor[:n], list(mse)[:n])]
            ax.plot(ep[:n], gain, label=lab, lw=1.8, color=c)
        ax.axhline(0.0, color="0.4", ls="--", lw=1.0, label="identity (no denoising)")
        _mark_ramp(ax)
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "needs additive noise\n+ sigma range",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="0.5")
    ax.set_title("Denoising gain (dB) - CURRICULUM-NORMALIZED")
    ax.set_xlabel("epoch"); ax.set_ylabel("10*log10(E[s^2] / MSE)  dB")
    ax.grid(True, alpha=0.3)

    # ---- Bottom row: normalized residual + MAE + schedules ----

    # (5) Residual noise fraction = MSE / E[sigma^2]. Same information as (4) in ratio
    # form: the fraction of input noise ENERGY still left in the output. 1.0 = identity.
    ax = axes[1, 0]
    if floors_ok:
        for key, floor, lab, c in (
            ("loss", train_floor, "train", TRAIN_C),
            ("val_loss", val_floor, "val", VAL_C),
        ):
            mse = history.get(key)
            if mse is None:
                continue
            n = min(len(ep), len(mse), len(floor))
            ax.plot(ep[:n], _ratio_or_nan(list(mse)[:n], floor[:n]),
                    label=lab, lw=1.8, color=c)
        ax.axhline(1.0, color="0.4", ls="--", lw=1.0, label="identity")
        ax.set_yscale("log")
        _mark_ramp(ax)
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "needs additive noise\n+ sigma range",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="0.5")
    ax.set_title("Residual noise fraction MSE/E[s^2] (log)")
    ax.set_xlabel("epoch"); ax.set_ylabel("fraction of noise energy left")
    ax.grid(True, alpha=0.3, which="both")

    # (6) MAE (raw; also task-confounded on the train split)
    ax = axes[1, 1]
    _line(ax, history.get("mae"), "train", color=TRAIN_C)
    _line(ax, history.get("val_mae"), "val", color=VAL_C)
    _mark_ramp(ax)
    ax.set_title("MAE per epoch (RAW)"); ax.set_xlabel("epoch"); ax.set_ylabel("MAE")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7)

    # (7) The curriculum itself, with a [0,255]-scale twin axis.
    ax = axes[1, 2]
    _line(ax, sig, "train sigma_max", color="#2ca02c", marker="o", markersize=3)
    if floors_ok:
        ax.axhline(float(val_sigma_max), color=VAL_C, ls=":", lw=1.3,
                   label="val sigma_max (fixed)")
    _mark_ramp(ax)
    ax.set_title("Noise sigma_max per epoch (curriculum)")
    ax.set_xlabel("epoch"); ax.set_ylabel("sigma_max  [0,1] units")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left", fontsize=7)
    if sig is not None:
        twin = ax.twinx()
        twin.set_ylabel("sigma on [0,255] scale")
        lo, hi = ax.get_ylim()
        twin.set_ylim(lo * 255.0, hi * 255.0)

    # (8) Learning rate (log y)
    ax = axes[1, 3]
    _line(ax, history.get("lr"), "lr", color="#9467bd")
    _mark_ramp(ax)
    ax.set_yscale("log")
    ax.set_title("Learning rate per epoch"); ax.set_xlabel("epoch"); ax.set_ylabel("lr (log)")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=7)

    plt.tight_layout(rect=(0, 0, 1, 0.97) if title else None)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# NOTE: batch-global RMSE PSNR (one RMSE over the whole batch). This intentionally
# differs from the per-image PsnrMetric used in training logs (which averages per-image
# PSNR), so eval-grid numbers will not match the val_psnr logged during fit.
def _mean_psnr(pred, clean) -> float:
    """Mean PSNR (dB) of ``pred`` vs ``clean`` on the [0,1] domain (max_val=1.0).

    Single source of truth for the trainer's PSNR convention: rmse over the whole
    batch, then ``20*log10(1.0/rmse)``. Both the eval grid and the multi-pass eval
    helpers call this so the formula lives in exactly one place (DRY).

    ``max_val=1.0`` is the PEAK-TO-PEAK WIDTH, which is 1.0 on [0,1] exactly as it was
    on the legacy [-0.5,+0.5] domain — the D-001 migration is a DC shift, not a rescale,
    so this formula is UNCHANGED by it (INV-2). Do NOT rescale it.
    """
    mse = float(tf.reduce_mean(tf.square(tf.convert_to_tensor(pred) - clean)))
    return 20.0 * np.log10(1.0 / max(np.sqrt(mse), 1e-8))  # max_val=1.0


def denoise_k_passes(model: keras.Model, noisy, k: int) -> List[tf.Tensor]:
    """Apply ``model`` ``k`` times sequentially, clipping to [0,1] between passes.

    Returns the LIST of the k intermediate denoised tensors ``[pass1, ..., passk]``
    so callers can score each pass independently. The model is applied exactly once
    per pass (``training=False``); this is eval/inference only and never affects
    training. Domain [0,1] (clip after every pass).
    """
    outputs: List[tf.Tensor] = []
    x = tf.clip_by_value(tf.convert_to_tensor(noisy), DATA_MIN, DATA_MAX)
    for _ in range(int(k)):
        x = model(x, training=False)
        if isinstance(x, (list, tuple)):
            x = x[0]  # deep-supervision: primary output
        x = tf.clip_by_value(tf.convert_to_tensor(x), DATA_MIN, DATA_MAX)
        outputs.append(x)
    return outputs


def multi_pass_psnr(model: keras.Model, clean, noisy, k: int) -> List[float]:
    """Per-pass mean PSNR (dB) for k sequential applications of ``model``.

    Runs ``denoise_k_passes`` and scores each pass against ``clean`` using the SAME
    convention as the eval grid (``_mean_psnr``, max_val=1.0). Returns a list of k
    floats ``[psnr(pass1), ..., psnr(passk)]``. Eval-only; does not affect training.
    """
    clean_t = tf.convert_to_tensor(clean)
    return [_mean_psnr(p, clean_t) for p in denoise_k_passes(model, noisy, k)]


def build_fixed_val_batch(
    val_paths: List[str], config: "BFUnetTrainingConfig", n: int = 8
) -> Optional[tf.Tensor]:
    """Load a small FIXED batch of clean [0,1] patches for visualization."""
    if not val_paths:
        return None
    patches = []
    for p in val_paths[: max(n * 3, n)]:
        try:
            patch = load_and_preprocess_image(tf.constant(p), config)
            patch = tf.clip_by_value(patch, DATA_MIN, DATA_MAX)
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
            sigma_max_start=cfg.get("sigma_max_start", 0.025),
            sigma_max_end=cfg.get("sigma_max_end", 0.25),
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
        hist, out, title=f"Training dashboard - {exp.name} ({len(epochs)} epochs)",
        # Sigma RANGE bounds -> per-epoch noise floor E[sigma^2] -> curriculum-normalized
        # panels. `sigma_max_end` is what the val pipeline pins its (fixed) range to.
        sigma_min=cfg.get("noise_sigma_min", 0.0),
        val_sigma_max=cfg.get("sigma_max_end", 0.25),
        additive=(cfg.get("noise_type", "additive") == "additive"),
    )
    logger.info(f"Saved training dashboard: {out}")
    return out


def make_curriculum_noise_fn(
    config: "BFUnetTrainingConfig",
    sigma_max_var: tf.Variable,
    clip_noise: bool = True,
):
    """Build a noise function that samples per-image sigma from
    ``[noise_sigma_min, sigma_max_var]`` where the upper bound is a live
    ``tf.Variable`` widened per-epoch by the curriculum callback (D-003).

    ``clip_noise`` (default ``True``) gates the return-value ``clip_by_value`` ONLY;
    when ``False`` the noisy input is returned unclipped (may exceed ``[0,1]``). The
    noise-generation draws above the clip are UNTOUCHED for all three branches, so the
    RNG draw order is preserved regardless of this flag (invariant 2)."""

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
        # Gate the return-value clip ONLY (invariant 2): nothing above this line moves,
        # so the RNG draw order is identical for clip_noise True/False. When False the
        # noisy input is returned unclipped (may exceed [0,1]); see decisions.md D-001.
        noisy = tf.clip_by_value(noisy, DATA_MIN, DATA_MAX) if clip_noise else noisy
        return noisy, patch

    return add_curriculum_noise


def _read_current_lr(model: keras.Model) -> float:
    """Read the current learning rate from ``model``'s live optimizer.

    Evaluates a ``LearningRateSchedule`` at the optimizer's current step, otherwise
    reads the scalar LR directly. Returns ``float('nan')`` if the optimizer/LR is
    unavailable. Single source of truth for both ``DenoisingVisualizationCallback.
    _current_lr`` (dashboard) and ``LRLoggerCallback`` (CSV ``lr`` column).
    """
    try:
        opt = model.optimizer
        lr = opt.learning_rate
        if isinstance(lr, keras.optimizers.schedules.LearningRateSchedule):
            return float(keras.ops.convert_to_numpy(lr(opt.iterations)))
        return float(keras.ops.convert_to_numpy(lr))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------
# CONFIGURATION (shared base dataclass)
# ---------------------------------------------------------------------


# DECISION plan_2026-07-02_8a5297be/D-001: shared config via dataclass INHERITANCE, not
# composition. Each trainer subclasses this and adds its disjoint model-only fields; do NOT
# collapse the two trainers' configs into one, and do NOT switch to a nested `model`
# sub-config — that would force editing every flat `config.foo` call site across both 2200+
# line trainers. Subclass __post_init__ MUST call super().__post_init__() first or all shared
# validation (sigma checks, dirs, self-iterate additive gate) silently vanishes. See decisions.md D-001.

# DECISION plan_2026-07-13_f44e2cb0/D-002: the admissible Gabor-stem activations.
# The generic builder (initializers/gabor_filters_initializer.py) deliberately does NOT
# validate its `activation` kwarg -- it also serves the non-bias-free cliffordnet
# autoencoder. This trainer IS the bias-free path, so the guard lives here. Only
# positively homogeneous activations may go on the stem: relu/leaky_relu/linear satisfy
# f(a*x) = a*f(x) for a >= 0, so D_sigma(y) = sigma*D(y/sigma) survives. Do NOT add gelu,
# elu, tanh, sigmoid, mish or swish to this set -- they have scale-dependent curvature and
# silently break Miyasawa compliance, which the existing homogeneity probes would NOT catch
# for the frozen stem alone.
GABOR_ACTIVATIONS = frozenset({"relu", "leaky_relu", "linear"})


@dataclass
class BFUnetTrainingConfig:
    """Shared configuration base for the bfunet bias-free denoiser trainers.

    Holds the ~55 fields that are byte-identical between the ConvUNeXt and Clifford
    trainers (Data / Memory / Noise-curriculum / shared U-Net topology / Training /
    Optimization / Self-iterate / WW-PGD / init_from / Analysis / Output) plus the
    shared ``__post_init__`` validation body. Each trainer subclasses this, adds its
    model-specific fields, and overrides ``__post_init__`` (calling
    ``super().__post_init__()`` first) with its model-specific validation. The
    ``experiment_prefix`` class attribute (NOT a dataclass field) is overridden per
    trainer so ``experiment_name`` defaults to ``f"{prefix}{variant}_{timestamp}"``.
    """

    # Overridable experiment-name prefix. Declared WITHOUT a type annotation so the
    # dataclass machinery ignores it (it is a plain class attribute, NOT a field);
    # subclasses set it to e.g. "convunext_denoiser_" / "cliffordunet_denoiser_".
    experiment_prefix = "denoiser_"

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
    # DECISION plan_2026-07-12_e56909cd/D-005: pixel-domain PROVENANCE STAMP, not a switch.
    # NOTHING in this repo branches on this value -- it exists solely so `save_config_json`
    # writes it into every checkpoint's config.json, letting a consumer tell a [0,1] model
    # from a legacy [-0.5,+0.5] one (which is otherwise indistinguishable at load time and
    # produces SILENT garbage if fed the wrong domain -- a bias-free net cannot subtract a
    # DC offset). Do NOT add an `if config.data_range == ...` anywhere: a domain-dispatch
    # branch is explicitly out of scope (INV-4, no compat shim). See decisions.md D-005.
    data_range: str = "[0,1]"
    image_extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"]
    )

    # Memory / sourcing
    max_train_files: Optional[int] = 10000
    max_val_files: Optional[int] = 500
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
    sigma_max_start: float = 0.025  # narrow range at epoch 0 (low noise)
    sigma_max_end: float = 0.25     # wide range at the final curriculum epoch
    curriculum_epochs: Optional[int] = None  # None -> use `epochs`
    curriculum_schedule: str = "linear"      # linear | cosine | exp
    # Clip the noisy input to [DATA_MIN, DATA_MAX] ([0,1]) after noise injection.
    # True (default) = today's behavior: the single return-value clip inside
    # make_curriculum_noise_fn fires for BOTH the train and val noise fns. False =
    # return the unclipped noisy input (may exceed [0,1]); mirrors eval_psnr_vs_noise's
    # --no-clip. Scope: gates ONLY the make_curriculum_noise_fn streaming path; the
    # self-iterate pool paths (common.py:325-327, self_iterate_pool.py) and the
    # dashboard viz clip are OUT of scope and still clip (train() logs a WARNING when
    # clip_noise=False and self_iterate=True). See decisions.md D-001.
    clip_noise: bool = True

    # Soft Jacobian-symmetry penalty (opt-in, default OFF). When symmetry_weight == 0.0
    # (default) training is byte-identical to today: stock compile(loss="mse")+fit(),
    # the model is NEVER wrapped. When > 0.0 the functional denoiser is wrapped in
    # BfunetSymmetryTrainingModel (step 4) whose train/test_step add
    # symmetry_weight * mean(||Jv - JTv||^2) over symmetry_probes random probe(s),
    # forced to float32. symmetry_weight>0 with mixed_precision=True is REFUSED in
    # __post_init__ (fail-closed): a second-order fp16/XLA path is in the repo's known
    # silent-training-death class. See decisions.md D-002 (wrapper) and D-003 (fp16 ban).
    symmetry_weight: float = 0.0
    symmetry_probes: int = 1

    # Model (shared bias-free U-Net topology)
    variant: str = "base"           # tiny | small | base | large | xlarge
    use_gabor_stem: bool = True
    gabor_filters: int = 32
    gabor_kernel_size: int = 11
    # Activation on the frozen Gabor stem. None -> linear passthrough (the raw signed
    # Gabor responses). Restricted to positively homogeneous activations: anything else
    # breaks the degree-1 homogeneity D(a*x) = a*D(x) the whole bias-free stack rests on.
    # Validated in __post_init__ against GABOR_ACTIVATIONS.
    gabor_activation: Optional[str] = None
    # Drop the mandatory bias-free 1x1 projection after the Gabor stem and feed the
    # depthwise bank straight into the encoder. Requires channels * gabor_filters ==
    # initial_filters EXACTLY (see initial_filters override below); the factory raises
    # otherwise. Default True = unchanged (projection kept).
    gabor_stem_projection: bool = True
    # Override the variant's initial_filters (level-0 width). None -> use the variant
    # default from the model CONFIGS. Primarily for the no-projection Gabor stem, where
    # initial_filters must equal channels * gabor_filters (e.g. 3 * 32 = 96).
    initial_filters: Optional[int] = None
    # Per-encoder-level channel-growth multiplier (>= 1). Channels at level i are
    # int(round(initial_filters * filter_multiplier ** i)). Default 2.0 doubles per
    # level (byte-identical to the historical int 2). Shared by both bfunet trainers;
    # passed into each factory by build_model.
    filter_multiplier: float = 2.0
    # Override the variant's number of U-Net levels (depth). None -> use the variant
    # default from the model CONFIGS. Shared by both bfunet trainers; passed into each
    # factory by build_model. Must be >= 2 when set.
    depth: Optional[int] = None
    # Override the variant's number of Clifford/ConvNeXt blocks per U-Net level. None ->
    # use the variant default from the model CONFIGS. Shared by both bfunet trainers;
    # passed into each factory by build_model. Must be >= 1 when set.
    blocks_per_level: Optional[int] = None
    # Groups for the final 1x1 output projection. 1 = standard dense conv (default,
    # byte-identical). -1 = one group per output channel (groups = channels), so each output
    # channel reads a disjoint feature group. Any >1 int sets the group count directly.
    # Requires initial_filters and channels both divisible by the resolved group count.
    final_projection_groups: int = 1
    use_laplacian_pyramid: bool = False
    # Parameter-free per-level channel matching (zero-pad on increase, slice-up+add-skip on decrease) instead of 1x1 channel-adjust convs; bias-free, default OFF. See the model factory.
    zero_pad_channels: bool = False
    high_freq_blocks: int = 0  # N bias-free blocks applied to the Laplacian high-frequency skip band per encoder level; ignored unless --laplacian-pyramid; default 0 = byte-identical to prior graph (D-001, plan_2026-07-06_b17c1f83)
    # Encoder downsample pooling for the non-Laplacian path: "max" (MaxPooling2D, default,
    # non-linear) or "average" (AveragePooling2D, LINEAR -> keeps the encoder path linear
    # for the Miyasawa/Tweedie residual-as-score interpretation). Ignored under Laplacian.
    downsample_pool_type: str = "max"
    # ConvNeXt block activation (inverted-bottleneck MLP). "leaky_relu" + alpha builds
    # keras.layers.LeakyReLU(negative_slope=alpha) in build_model (the bare "leaky_relu"
    # string resolves to slope 0.2, so the trainer constructs the instance to honor 0.1);
    # any other value is passed to the factory as a plain Keras activation string.
    block_activation: str = "leaky_relu"
    block_activation_alpha: float = 0.1
    # Pre-activation normalization inside every ConvNeXt block. "batchnorm" (default) =
    # variance-only BiasFreeBatchNorm (no mean, no beta) which restores degree-1
    # homogeneity f(ax)=a*f(x) at inference (pairs best with a homogeneous activation like
    # LeakyReLU); "layernorm" = per-input scale-invariant (degree-0), byte-identical to
    # legacy pre-batchnorm checkpoints. Wired to create_convunext_denoiser(block_normalization=...).
    block_normalization: str = "batchnorm"
    enable_deep_supervision: bool = False
    expose_bottleneck: bool = False

    # Training
    batch_size: int = 16
    epochs: int = 100
    patches_per_image: int = 4
    augment_data: bool = True
    # Mixed precision (mixed_float16): compute in fp16 on tensor cores, keep fp32 weights
    # + fp32 model output for stable MSE/PSNR/SSIM. Opt-in; numerics differ from fp32.
    mixed_precision: bool = False
    steps_per_epoch: Optional[int] = None
    validation_steps: Optional[int] = 100

    # Optimization
    learning_rate: float = 1e-3
    optimizer_type: str = "adamw"
    # Decoupled (AdamW) weight decay. Default mirrors optimizer_builder's adamw default
    # (0.004) so behavior is unchanged; surfaced here so it lands in the saved config.json
    # and is tunable via --weight-decay. AdamW WD only -- no kernel_regularizer L2 (would
    # double-penalize); see train/CLAUDE.md "Double Weight Decay".
    weight_decay: float = 0.004
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
            self.experiment_name = f"{self.experiment_prefix}{self.variant}_{timestamp}"
        if self.curriculum_epochs is None:
            self.curriculum_epochs = self.epochs
        if self.warmup_epochs is None:
            # Default warmup = 10% of total training epochs (>=1).
            self.warmup_epochs = max(1, round(0.1 * self.epochs))
        if self.patch_size <= 0 or self.channels <= 0:
            raise ValueError("Invalid patch size or channel configuration")
        if self.filter_multiplier < 1:
            raise ValueError(
                f"filter_multiplier must be >= 1, got {self.filter_multiplier}"
            )
        if self.gabor_activation is not None and self.gabor_activation not in GABOR_ACTIVATIONS:
            raise ValueError(
                f"gabor_activation must be None or one of {sorted(GABOR_ACTIVATIONS)}, "
                f"got {self.gabor_activation!r}. Non-homogeneous activations (gelu, elu, "
                f"tanh, sigmoid, mish, swish) break the degree-1 homogeneity "
                f"D(a*x) = a*D(x) that the bias-free denoisers rely on."
            )
        if self.depth is not None and self.depth < 2:
            raise ValueError(
                f"depth must be >= 2, got {self.depth}"
            )
        if self.blocks_per_level is not None and self.blocks_per_level < 1:
            raise ValueError(
                f"blocks_per_level must be >= 1, got {self.blocks_per_level}"
            )
        if self.block_normalization not in ("layernorm", "batchnorm"):
            raise ValueError(
                f"block_normalization must be 'layernorm' or 'batchnorm', "
                f"got {self.block_normalization!r}"
            )
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
        if self.symmetry_weight < 0:
            raise ValueError("symmetry_weight must be >= 0")
        if self.symmetry_probes < 1:
            raise ValueError("symmetry_probes must be >= 1")
        if self.symmetry_weight > 0 and self.mixed_precision:
            # DECISION plan-2026-07-17T112359-874b11cc/D-003: fail-closed. The penalty is a
            # JVP-of-VJP (second differentiation of the backward pass) and belongs to the
            # repo's fp16/XLA silent-training-death class (EnergyLayerNorm (var+eps)^(-3/2)
            # overflow; -1e9 -> fp16 -inf -> NaN). Do NOT relax this to a WARNING or try to
            # cast-around it here: a forward-only fp16 pass certifies a policy the BACKWARD
            # pass cannot honour, so the safe default is an outright refusal until a real
            # fit()-step weight-movement probe proves the fp16 combo safe. See decisions.md
            # D-003; the float32-forcing lives in the penalty module + the step-4 wrapper.
            raise ValueError(
                "symmetry_weight>0 requires float32 training; disable mixed_precision "
                "(second-order fp16/XLA is a known silent-training-death path)"
            )
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
            # Even under additive noise the always-on tf.clip_by_value(DATA_MIN, DATA_MAX)
            # in add_curriculum_noise makes the observed noise non-Gaussian at the [0,1]
            # boundaries, so residual=score (Miyasawa) no longer holds exactly there.
            # Under [0,1] the bias is RELOCATED, not removed: it now concentrates near
            # BLACK (0.0) and WHITE (1.0) rather than at the symmetric +-0.5 extremes.
            # Self-iterated passes can therefore drift from the theoretical clean-image
            # fixed point. WARNING (not ValueError): the drift is usually small and
            # self-iterate stays useful, so this is informational only.
            logger.warning(
                "self_iterate is ON: add_curriculum_noise always clips noisy inputs "
                "to [0, 1], which breaks the Miyasawa residual=score identity at "
                "the clip boundaries (near black and near white) even for additive "
                "noise. Self-iterated passes may drift from the theoretical "
                "clean-image fixed point."
            )


# ---------------------------------------------------------------------
# HOMOGENEITY PROBE (shared numeric black-box check)
# ---------------------------------------------------------------------


def _homogeneity_probe(model: keras.Model) -> None:
    """Numeric black-box degree-1 homogeneity probe (informational, NEVER raises).

    Architecture-agnostic half of ``verify_bias_free``: synthesizes an input,
    forwards ``x`` and ``alpha*x`` for ``alpha in (0.5, 2.0)``, and logs the relative
    error of ``f(alpha*x)`` vs ``alpha*f(x)``. Each trainer's ``verify_bias_free``
    runs its own model-specific offender scan and then calls this.
    """
    # ------------------------------------------------------------------
    # Numerical BLACK-BOX homogeneity probe (informational, NEVER raises;
    # verify_bias_free MUST return None). Detects ANY degree-1 break
    # f(a*x) != a*f(x) at inference — including breaks the static scan above
    # cannot see (GRN stem, GELU, non-homogeneous LayerNorm blocks).
    #
    # DECISION plan_2026-07-01_8054f023/D-005: a PASS here on an UNTRAINED model
    # does NOT prove homogeneity. LayerScale gamma_init=1e-5 makes each residual
    # branch near-identity at init, masking in-block norm/activation degree-0
    # breaks until training grows gamma (why the audit saw the break only on the
    # TRAINED checkpoint). The probe is most meaningful post-training, or for
    # GRN/GELU-driven breaks that ARE visible at init. Do NOT read an
    # untrained-model pass as evidence the model is homogeneous. See decisions.md
    # D-005; do NOT replace this with a hard assert / raise.
    try:
        in_shape = model.input_shape
        if isinstance(in_shape, list):
            in_shape = in_shape[0]
        spatial = tuple(d if d is not None else 64 for d in in_shape[1:])
        rng = np.random.default_rng(0)
        x = rng.uniform(DATA_MIN, DATA_MAX, size=(1,) + spatial).astype("float32")
    except Exception as e:
        logger.debug(f"Homogeneity probe skipped (could not synthesize input): {e}")
        return

    def _first_output(y):
        if isinstance(y, (list, tuple)):
            return y[0]
        if isinstance(y, dict):
            return list(y.values())[0]
        return y

    try:
        # fp16 compute needs a looser tolerance or the probe false-WARNs every run.
        try:
            compute_dtype = keras.mixed_precision.global_policy().compute_dtype
        except Exception:
            compute_dtype = "float32"
        tol = 5e-2 if compute_dtype == "float16" else 1e-2

        fx = np.asarray(_first_output(model(x, training=False)))
        max_rel = 0.0
        for alpha in (0.5, 2.0):
            f_ax = np.asarray(_first_output(model(alpha * x, training=False)))
            denom = max(float(np.max(np.abs(alpha * fx))), 1e-8)
            rel = float(np.max(np.abs(f_ax - alpha * fx)) / denom)
            max_rel = max(max_rel, rel)
            logger.info(f"Homogeneity probe: alpha={alpha} rel_err={rel:.3e}")
        if max_rel > tol:
            logger.warning(
                f"Homogeneity probe: model is NOT degree-1 homogeneous "
                f"(max rel_err={max_rel:.3e} > tol={tol:.1e}) — likely a "
                f"non-homogeneous norm/activation/context stream. INFORMATIONAL "
                f"only; on an untrained model LayerScale gamma=1e-5 can mask "
                f"in-block breaks (D-005)."
            )
    except Exception as e:
        logger.debug(f"Homogeneity probe skipped (forward pass failed): {e}")

    # The DC/sum-to-one probe is the [0,1] domain's reason for existing (D-001), so it
    # runs wherever the homogeneity probe runs. Chaining it here rather than adding a
    # second call in each of the four trainers' verify_bias_free keeps the call sites
    # (and the informational-only contract) in exactly one place.
    _dc_preservation_probe(model)


def _dc_preservation_probe(model: keras.Model) -> None:
    """Numeric DC / sum-to-one probe (informational, NEVER raises).

    Feeds FLAT constant images ``c * ones`` for several ``c`` in [0,1] and logs the
    relative error ``||f(c*1) - c*1|| / ||c*1||``. This is the diagnostic the [0,1]
    domain migration exists to make meaningful (D-001): a bias-free net is degree-1
    homogeneous with ``f(0) = 0``, so ``f(c*1) = c*f(1)``, and reproducing a flat patch
    REQUIRES ``f(1) = 1`` — local filter weights that sum to one, i.e. DC preservation.
    On the legacy zero-centered domain a flat mid-grey patch WAS the zero vector, so the
    property was satisfied structurally and this probe would have been vacuous.

    Interpretation: on an UNTRAINED model the error is EXPECTED to be large (random
    filter weights do not sum to 1) — that is not a failure. The number is only
    meaningful as a trend across training. Informational only, mirroring
    ``_homogeneity_probe``: it must never raise and never gate anything.
    """
    try:
        in_shape = model.input_shape
        if isinstance(in_shape, list):
            in_shape = in_shape[0]
        spatial = tuple(d if d is not None else 64 for d in in_shape[1:])
    except Exception as e:
        logger.debug(f"DC probe skipped (could not synthesize input): {e}")
        return

    def _first_output(y):
        if isinstance(y, (list, tuple)):
            return y[0]
        if isinstance(y, dict):
            return list(y.values())[0]
        return y

    try:
        for c in (0.1, 0.25, 0.5, 0.75, 0.9):
            flat = np.full((1,) + spatial, float(c), dtype="float32")
            out = np.asarray(_first_output(model(flat, training=False)), dtype="float64")
            denom = max(float(np.linalg.norm(flat)), 1e-8)
            rel = float(np.linalg.norm(out - flat) / denom)
            logger.info(f"DC/sum-to-one probe: c={c} rel_err={rel:.3e}")
    except Exception as e:
        logger.debug(f"DC probe skipped (forward pass failed): {e}")


# ---------------------------------------------------------------------
# VISUALIZATION CALLBACKS
# ---------------------------------------------------------------------


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
        out_dir: Path,
        freq: int = 5,
        max_samples: int = 8,
        val_ds=None,
        validation_steps: Optional[int] = None,
        noise_regimes: Optional[List[Tuple[str, float]]] = None,
        noise_type: str = "additive",
        composite_additive_ratio: float = 0.5,
        multi_pass_k: int = 3,
        model_label: str = "Denoiser",
        noise_sigma_min: float = 0.0,
        val_sigma_max: Optional[float] = None,
    ):
        super().__init__()
        self.clean_batch = clean_batch
        self.sigma_max_var = sigma_max_var
        self.noise_type = noise_type
        self.model_label = model_label
        # Sigma RANGE bounds, needed by the dashboard to compute the per-epoch noise
        # floor E[sigma^2] and so de-confound the curriculum from the raw curves.
        # `val_sigma_max` is the FIXED upper bound the val pipeline uses (sigma_max_end);
        # the train upper bound is the live, ramping `sigma_max_var`.
        self.noise_sigma_min = float(noise_sigma_min)
        self.val_sigma_max = None if val_sigma_max is None else float(val_sigma_max)
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
        # Fixed reference noise regimes for the eval grid (sigma in [0,1] units),
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
                ("low", 15.0 / 255.0),
                ("medium", 25.0 / 255.0),
                ("high", 50.0 / 255.0),
            ]
        self._hist = {k: [] for k in (
            "epoch", "loss", "val_loss", "mae", "val_mae",
            "psnr", "val_psnr", "sigma_max", "lr",
        )}

    def _current_lr(self) -> float:
        """Current LR from the live optimizer; ``float('nan')`` on failure.
        Delegates to the module-level ``_read_current_lr``."""
        return _read_current_lr(self.model)

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
                    sigma_min=self.noise_sigma_min,
                    val_sigma_max=self.val_sigma_max,
                    additive=(self.noise_type == "additive"),
                )
                logger.info(
                    f"Epoch-0 baseline: val_loss={res.get('loss'):.4f} "
                    f"val_psnr={res.get('psnr_metric', float('nan')):.2f} dB"
                )
            except Exception as e:
                logger.warning(f"Epoch-0 baseline eval failed: {e}")

    def on_epoch_end(self, epoch: int, logs=None):
        """Record per-epoch scalars for the combined dashboard and, at the viz
        cadence, save the clean/noisy/denoised grid plus the PSNR curve."""
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
                sigma_min=self.noise_sigma_min,
                val_sigma_max=self.val_sigma_max,
                additive=(self.noise_type == "additive"),
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
                # trainer uses, then the SAME [0,1] clip as the additive path.
                noisy = tf.clip_by_value(
                    apply_multiplicative_gaussian(clean, sigma), DATA_MIN, DATA_MAX
                )
            elif composite:
                # Composite regime: sigma here is sigma_m; the additive floor is
                # sigma_a = ratio * sigma_m. Reuse the trainer's composite primitive,
                # then the SAME [0,1] clip as the other paths.
                noisy = tf.clip_by_value(
                    apply_composite_gaussian(clean, sigma, ratio * sigma),
                    DATA_MIN,
                    DATA_MAX,
                )
            else:
                noisy = tf.clip_by_value(
                    clean + tf.random.normal(tf.shape(clean)) * sigma,
                    DATA_MIN,
                    DATA_MAX,
                )
            denoised = self.model(noisy, training=False)
            if isinstance(denoised, (list, tuple)):
                denoised = denoised[0]  # deep-supervision: primary output
            denoised = tf.convert_to_tensor(denoised)
            psnr = _mean_psnr(denoised, clean)  # max_val=1.0
            psnr_noisy = _mean_psnr(noisy, clean)  # max_val=1.0
            if multiplicative:
                noisy_label = f"Noisy {label}\n(mult σ={sigma:.2f}, PSNR {psnr_noisy:.1f} dB)"
            elif composite:
                noisy_label = (
                    f"Noisy {label}\n(comp σm={sigma:.2f} σa={ratio * sigma:.2f}, "
                    f"PSNR {psnr_noisy:.1f} dB)"
                )
            else:
                s255 = sigma * 255.0
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
            f"{self.model_label} - epoch {epoch} - same images, 3 noise regimes",
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
        """Inject the current LR into ``logs['lr']`` so CSVLogger records it; must
        run before CSVLogger (see the class docstring on callback ordering)."""
        if logs is None:
            return
        # Preserve the prior semantics: on read failure (nan) leave logs['lr'] unset
        # rather than writing a nan into the CSV row.
        lr = _read_current_lr(self.model)
        if np.isfinite(lr):
            logs["lr"] = lr


# ---------------------------------------------------------------------
# SOFT JACOBIAN-SYMMETRY PENALTY: training-only wrapper (opt-in, default OFF)
# ---------------------------------------------------------------------


class BfunetSymmetryTrainingModel(keras.Model):
    """Thin ``keras.Model`` training harness adding a soft Jacobian-symmetry penalty.

    Wraps an already-built, already-compiled functional bfunet denoiser
    (``inner_model``) and overrides ``train_step`` / ``test_step`` to add
    ``symmetry_weight * mean(||Jv - JTv||^2)`` (a reverse-mode double-VJP estimate
    of the denoiser's input-output Jacobian asymmetry, forced to float32) on top
    of the stock reconstruction MSE. Cloned from THERA's ``TheraTrainingModel``
    pattern (``src/train/thera/train_thera.py:88-269``, D-012): a proven, sanctioned
    custom-``train_step`` exception for a loss term that needs the model's OWN
    Jacobian inside the weight-gradient tape (D-002).

    **Discard-the-wrapper discipline (invariant 3).** ``inner_model`` is the sole
    DEPLOYABLE artifact — a functional denoiser carrying its ``config.json``
    ``data_range="[0,1]"`` provenance stamp. ``save()`` therefore delegates to
    ``self.inner.save()`` so ``best_model.keras`` (written by ``ModelCheckpoint``
    via ``model.save``) is the denoiser, NOT this training harness. The wrapper
    itself is never serialized.

    **Float32-forced penalty (invariant 4).** ``jacobian_symmetry_penalty`` casts
    its input to float32 and runs the whole nested-tape computation in float32; the
    trainer additionally sets ``jit_compile=False`` and the config ``__post_init__``
    REFUSES ``symmetry_weight>0 and mixed_precision`` (fail-closed), so this wrapper
    only ever runs under a float32 policy.

    Interface contract:
        - ``inner_model``: a callable, already-built ``keras.Model`` denoiser mapping
          a ``(B, H, W, C)`` input to a same-shaped output. Left UNMODIFIED (composed,
          not subclassed); its trainable variables ARE this wrapper's trainable
          variables (auto-tracked via the ``self.inner`` attribute).
        - ``symmetry_weight`` (float > 0): scalar weight on the penalty term.
        - ``symmetry_probes`` (int >= 1): number of random probes averaged by the
          penalty estimator.
        - ``train_step``/``test_step`` consume ``(x, y)`` (and ``sample_weight`` if
          present) exactly like the stock denoiser dataset (``noisy, clean``). Recon
          is computed via ``self.compute_loss`` so it matches the stock
          ``compile(loss="mse")`` reduction exactly; the ``"loss"`` metric reports the
          TOTAL (recon + penalty), and ``"symmetry_penalty"`` is reported separately.
        - ``save(*args, **kwargs)`` delegates to ``self.inner.save`` (returns the inner
          model's save result).

    Args:
        inner_model: The functional denoiser to wrap (deployable artifact).
        symmetry_weight: Weight on the Jacobian-symmetry penalty (> 0 when wrapped).
        symmetry_probes: Number of random probes for the penalty estimator (>= 1).
        **kwargs: Forwarded to ``keras.Model``.
    """

    def __init__(
        self,
        inner_model: keras.Model,
        symmetry_weight: float,
        symmetry_probes: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.inner = inner_model
        self.symmetry_weight = float(symmetry_weight)
        self.symmetry_probes = int(symmetry_probes)
        # Mean tracker so the penalty is epoch-averaged and auto-appears in logs /
        # history (auto-tracked via attribute assignment; verified present in
        # self.metrics_names). Distinct from the raw last-batch scalar.
        self.symmetry_tracker = keras.metrics.Mean(name="symmetry_penalty")

    def call(self, inputs, training=None):
        # Delegate so predict/evaluate/build and the train_step forward pass all
        # produce the inner denoiser's output.
        return self.inner(inputs, training=training)

    def build(self, input_shape):
        if not self.inner.built:
            self.inner.build(input_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return self.inner.compute_output_shape(input_shape)

    def _penalty(self, x) -> tf.Tensor:
        """Float32 Jacobian-symmetry penalty on the (noisy) input batch ``x``."""
        # Force float32 regardless of policy (the penalty fn already casts; this is
        # belt-and-braces so no fp16 leaks into the second-order tapes — D-003).
        pen = jacobian_symmetry_penalty(
            self.inner,
            tf.cast(x, tf.float32),
            num_probes=self.symmetry_probes,
        )
        return tf.cast(self.symmetry_weight, tf.float32) * pen

    def train_step(self, data):
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)

        # Single OUTER weight tape; the penalty's inner tapes compose with it for the
        # second-order term. Recon via compute_loss => matches stock compile(loss="mse").
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            recon = self.compute_loss(
                x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=True
            )
            penalty = self._penalty(x)
            total = recon + tf.cast(penalty, recon.dtype)

        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Report TOTAL (recon + penalty) as "loss" so it sits above bare MSE, mirroring
        # THERA; the compiled metrics (mae/psnr/ssim) track the denoised output.
        self._loss_tracker.update_state(total)
        self.symmetry_tracker.update_state(penalty)
        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def test_step(self, data):
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        y_pred = self(x, training=False)
        recon = self.compute_loss(
            x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=False
        )
        # Mirror the train objective so val_loss tracks the real (penalized) objective
        # (THERA test_step parity). THERA computes the penalty on val too; match it.
        penalty = self._penalty(x)
        total = recon + tf.cast(penalty, recon.dtype)
        self._loss_tracker.update_state(total)
        self.symmetry_tracker.update_state(penalty)
        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def save(self, *args, **kwargs):
        # DECISION plan-2026-07-17T112359-874b11cc/D-002: delegate to the inner denoiser
        # so ModelCheckpoint's best_model.keras is the DEPLOYABLE functional model (with its
        # config.json data_range="[0,1]" provenance stamp), NOT this discard-after-training
        # harness (invariant 3 / Pre-Mortem signal 2). Do NOT let the wrapper serialize itself:
        # require_unit_domain_checkpoint() would refuse a training-harness artifact.
        return self.inner.save(*args, **kwargs)


# ---------------------------------------------------------------------
# TRAINING ORCHESTRATION (shared)
# ---------------------------------------------------------------------


# DECISION plan_2026-07-02_8a5297be/D-003: train() takes BOTH build_model_fn AND verify_fn
# (the model-specific bias-free offender scan, invoked at the post-build AND post-init_from
# sites). Do NOT drop verify_fn and duplicate the two verify sites into each trainer's thin
# wrapper (re-forks ~30 lines of init_from/verify orchestration), and do NOT move
# verify_bias_free into common (it is model-specific AND frozen-imported from the trainer
# path). Only the architecture-agnostic numeric half lives here, in _homogeneity_probe. See decisions.md D-003.
def train(
    config: "BFUnetTrainingConfig",
    build_model_fn,
    verify_fn,
    *,
    model_label: str,
    results_dir_prefix: str,
    bottleneck_name_prefix: Optional[str] = None,
) -> keras.Model:
    """Train a bias-free bfunet denoiser with the noise curriculum.

    Shared orchestration for both trainers. ``build_model_fn(config)`` builds the
    model; ``verify_fn(model)`` runs the model-specific bias-free check (called at
    both the post-build and post-``init_from`` sites). ``model_label`` /
    ``results_dir_prefix`` / ``bottleneck_name_prefix`` are the per-trainer seams.
    """
    logger.info(f"Starting {model_label} denoiser training: {config.experiment_name}")

    # Provenance gate on the warm-start checkpoint, BEFORE anything expensive
    # (plan_2026-07-12_e56909cd/D-005). The "init_from loaded 0 layers" guard further
    # down does NOT cover this case: a LEGACY [-0.5,+0.5] checkpoint of the SAME
    # architecture loads 100% of its layers happily, and the run then warm-starts from
    # weights trained in the wrong pixel domain — no error, no signal, just a
    # systematically wrong init. Shared gate; see dl_techniques.utils.denoiser_provenance.
    if config.init_from is not None:
        require_unit_domain_checkpoint(config.init_from)

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
    noise_fn = make_curriculum_noise_fn(config, sigma_max_var, config.clip_noise)

    # Honesty guard (D-001): clip_noise only gates the streaming make_curriculum_noise_fn
    # path (train + val). The self-iterate pool init and its regen callback clip
    # independently and are OUT of scope for this flag, so under --self-iterate the pool
    # inputs are STILL clipped even with clip_noise=False. Warn rather than silently
    # leave a partial gap.
    if not config.clip_noise and config.self_iterate:
        logger.warning(
            "clip_noise=False has NO effect on the self-iterate pool paths: the "
            "self-iterate pool init and regeneration callback clip noisy inputs to "
            "[0, 1] independently of make_curriculum_noise_fn. Only the (unused-in-"
            "self-iterate) streaming train/val noise honours clip_noise=False here."
        )

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
    val_noise_fn = make_curriculum_noise_fn(config, sigma_fixed_var, config.clip_noise)
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
            clip_min=DATA_MIN,
            clip_max=DATA_MAX,
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

    # Mixed precision must be set BEFORE the model is built so every layer adopts the
    # mixed_float16 policy (fp16 compute, fp32 variables). Opt-in; off => fp32 as before.
    if config.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info(
            "Mixed precision ENABLED: global policy=mixed_float16 "
            "(compute fp16 on tensor cores, weights fp32, output cast back to fp32)"
        )
    else:
        keras.mixed_precision.set_global_policy("float32")

    model = build_model_fn(config)
    model.summary(print_fn=logger.info)
    verify_fn(model)

    # Optional warm-start from a saved .keras checkpoint (e.g. self-iterate
    # fine-tuning on top of a normally-trained denoiser). The functional model is
    # already built, so layer-by-layer transfer works. skip_prefixes=() loads ALL
    # layers (the denoiser has no head_ layers to skip, unlike CliffordNetUNet).
    if config.init_from is not None:
        # Its [0,1] provenance was already gated at the top of train() — a legacy-domain
        # checkpoint never reaches this transfer (D-005).
        logger.info(f"Initializing weights from checkpoint: {config.init_from}")
        report = load_weights_from_checkpoint(
            model,
            ckpt_path=config.init_from,
            skip_prefixes=(),
        )
        if len(report.loaded) == 0:
            raise ValueError(
                f"init_from loaded 0 layers from {config.init_from} — architecture "
                "mismatch? Ensure --variant/--patch-size and the model-specific flags "
                "match the checkpoint's architecture."
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
        verify_fn(model)  # re-check: transfer must not introduce bias

    # Under mixed_float16 the final Conv2D emits fp16; cast the model output(s) back to
    # fp32 so MSE + PSNR/SSIM are computed at full precision (Keras mixed-precision best
    # practice). The cast is a weight-free Activation, so bias-free homogeneity and the
    # weight transfer above are unaffected. expose_bottleneck rewrites the output set
    # below, so guard the (advanced, non-default) combination out for now.
    if config.mixed_precision:
        if config.expose_bottleneck:
            raise ValueError(
                "mixed_precision is not supported together with expose_bottleneck yet; "
                "run with only one of them."
            )
        f32_outputs = [
            keras.layers.Activation("linear", dtype="float32", name=f"output_f32_{i}")(o)
            for i, o in enumerate(model.outputs)
        ]
        model = keras.Model(
            inputs=model.inputs,
            outputs=f32_outputs if len(f32_outputs) > 1 else f32_outputs[0],
            name=model.name,
        )

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
            "weight_decay": config.weight_decay,
            "gradient_clipping_by_norm": config.gradient_clipping,
        },
        lr_schedule,
    )
    # Dynamic loss scaling: fp16 gradients underflow without it. Wraps the built
    # optimizer (gradient clipping stays on the inner optimizer, applied post-unscale).
    if config.mixed_precision:
        optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)

    # MSE loss -> least-squares-optimal (Miyasawa) denoiser.
    # Compile/fit operate on train_model (the single-output view when expose_bottleneck;
    # otherwise train_model IS model). The view shares weight objects with the full model.
    # XLA + mixed_float16 is incompatible here: the decoder's bilinear-upsample gradient
    # (ResizeBilinearGrad) emits fp32 regardless of input dtype, which XLA's strict dtype
    # checker rejects ("mixed precision disallowed") while the TF runtime auto-casts it.
    # Disable XLA for the mixed-precision path; fp16 tensor-core matmuls/convs still
    # accelerate it. fp32 keeps Keras' default jit ("auto").
    train_model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            "mae",
            PsnrMetric(max_val=1.0, name="psnr_metric"),
            SsimMetric(max_val=1.0, name="ssim_metric"),
        ],
        jit_compile=False if config.mixed_precision else "auto",
    )
    logger.info(f"Model compiled with {model.count_params():,} parameters")

    # Soft Jacobian-symmetry penalty (opt-in, default OFF). When symmetry_weight == 0.0
    # the model is NEVER wrapped and the stock compile/fit path above stays byte-identical
    # (invariant 1). When > 0, wrap the compiled denoiser in the THERA-style training
    # harness (D-002) whose train_step/test_step add the float32 penalty. jit_compile is
    # forced OFF for this path: the JVP-of-VJP is a second differentiation of the backward
    # pass and belongs to the repo's fp16/XLA silent-death class (D-003 also bans the fp16
    # combo at config time). The wrapper's save() delegates to the inner denoiser so
    # ModelCheckpoint's best_model.keras is the deployable model (invariant 3).
    if config.symmetry_weight > 0:
        train_model = BfunetSymmetryTrainingModel(
            train_model, config.symmetry_weight, config.symmetry_probes
        )
        train_model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=[
                "mae",
                PsnrMetric(max_val=1.0, name="psnr_metric"),
                SsimMetric(max_val=1.0, name="ssim_metric"),
            ],
            jit_compile=False,
        )
        logger.info(
            "Jacobian-symmetry penalty ENABLED "
            f"(symmetry_weight={config.symmetry_weight}, "
            f"symmetry_probes={config.symmetry_probes}, float32, jit_compile=False). "
            "Training the wrapper; best_model.keras remains the deployable denoiser."
        )

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
        results_dir_prefix=results_dir_prefix,
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
            out_dir=output_dir,
            freq=config.viz_freq,
            max_samples=config.viz_samples,
            val_ds=val_ds,
            validation_steps=validation_steps,
            noise_type=config.noise_type,
            composite_additive_ratio=config.composite_additive_ratio,
            model_label=f"{model_label} Denoiser",
            # Sigma RANGE bounds for the dashboard's noise floor E[sigma^2]. The val
            # pipeline pins its upper bound to sigma_max_end (see sigma_fixed_var), so
            # the val floor is CONSTANT while the train floor ramps with the curriculum.
            noise_sigma_min=config.noise_sigma_min,
            val_sigma_max=config.sigma_max_end,
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
                **({"name_prefix": bottleneck_name_prefix} if bottleneck_name_prefix else {}),
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
    # Capture a small synthetic sample + its prediction BEFORE saving so we can
    # verify the .keras round-trip reproduces outputs (sibling pattern:
    # train/convnext/train_convnext_v1.py:200-211). Log-only: a mismatch — or any
    # check failure — NEVER crashes the trainer tail; the in-memory model is fine.
    round_trip_sample = None
    round_trip_pred = None
    try:
        in_shape = model.input_shape
        if isinstance(in_shape, list):
            in_shape = in_shape[0]
        spatial = tuple(d if d is not None else config.patch_size for d in in_shape[1:])
        round_trip_sample = np.random.default_rng(0).uniform(
            DATA_MIN, DATA_MAX, size=(1,) + spatial
        ).astype("float32")
        round_trip_pred = model.predict(round_trip_sample, verbose=0)
    except Exception as e:
        logger.debug(f"Round-trip pre-save capture skipped: {e}")

    model.save(final_path)
    logger.info(f"Saved final (last-epoch) model -> {final_path}")

    if round_trip_pred is not None:
        try:
            # fp16 save/load drifts more; relax the tolerance so the check does not
            # false-WARN under mixed_precision.
            tol = 5e-2 if config.mixed_precision else 1e-4
            ok = validate_model_loading(
                str(final_path),
                round_trip_sample,
                round_trip_pred,
                custom_objects=None,  # all layers are @register_keras_serializable
                tolerance=tol,
            )
            if not ok:
                logger.warning(
                    "final_model.keras round-trip check FAILED: reloaded outputs "
                    f"differ from pre-save (tol={tol:.1e}). The saved file may not "
                    "reload identically."
                )
        except Exception as e:
            logger.warning(
                f"final_model.keras round-trip check errored (non-fatal): {e}"
            )

    gc.collect()
    return model


# ---------------------------------------------------------------------
# CLI (shared argument group)
# ---------------------------------------------------------------------


def add_common_arguments(parser) -> None:
    """Add the ~45 flags shared byte-identically by both trainers' ``parse_arguments``.

    EXCLUDES ``--variant`` (model-specific choices) and the model-only flags; each
    trainer adds those itself after calling this. Mutates ``parser`` in place.
    """
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--patches-per-image", type=int, default=4)
    parser.add_argument("--mixed-precision", action="store_true",
                        help="enable mixed_float16 (fp16 compute on tensor cores, fp32 "
                             "weights + fp32 output). NOTE: measured SLOWER than the fp32 "
                             "default for base@256/b4 on a 4090 (~22 vs 36 img/s): the "
                             "decoder's bilinear-upsample grad forces XLA off, and XLA "
                             "outweighs fp16 here. Off by default; may help at higher res "
                             "/ other GPUs / if the XLA upsample-grad issue is fixed.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.004,
                        help="AdamW decoupled weight decay (default 0.004, matching the "
                             "optimizer_builder default). AdamW WD only; no L2 "
                             "kernel_regularizer is added (avoids double weight decay).")
    parser.add_argument("--warmup-epochs", type=int, default=None,
                        help="LR warmup length (default: 10%% of --epochs)")
    parser.add_argument("--no-gabor-stem", action="store_true",
                        help="Disable the frozen Gabor depthwise stem")
    parser.add_argument("--laplacian-pyramid", action="store_true",
                        help="Enable the Laplacian-pyramid downsample/skip path (default OFF)")
    parser.add_argument("--no-clip", action="store_true",
                        help="Disable the [0,1] clip on the noisy input inside the "
                             "streaming train+val noise fn (default: clip ON). Matches "
                             "eval_psnr_vs_noise's --no-clip. Note: self-iterate pool "
                             "paths and dashboard images still clip (logged as a WARNING).")
    parser.add_argument("--symmetry-weight", type=float, default=0.0,
                        help="Weight of the soft Jacobian-symmetry penalty "
                             "symmetry_weight * mean(||Jv - JTv||^2) (default 0.0 = OFF, "
                             "stock compile/fit, byte-identical). When > 0 the denoiser is "
                             "wrapped in a float32 THERA-style training model; REFUSED "
                             "together with --mixed-precision (second-order fp16/XLA is a "
                             "known silent-training-death path).")
    parser.add_argument("--symmetry-probes", type=int, default=1,
                        help="Number of random probe vectors the symmetry penalty averages "
                             "over per step (default 1; ignored unless --symmetry-weight > 0).")
    parser.add_argument("--high-freq-blocks", type=int, default=0,
                        help="N bias-free blocks on the Laplacian high-frequency skip band per encoder level (default 0 = OFF; ignored unless --laplacian-pyramid)")
    parser.add_argument("--zero-pad-channels", action="store_true", help="Replace per-level channel-adjust 1x1 convs with parameter-free channel matching (zero-pad on increase; decoder slices the upsampled branch and adds the skip). Bias-free, fewer params; default OFF.")
    parser.add_argument("--mean-pooling", action="store_true",
                        help="Use AveragePooling2D (LINEAR) instead of MaxPooling2D for the "
                             "encoder downsample, keeping the encoder path linear for the "
                             "Miyasawa/Tweedie residual-as-score interpretation (MaxPooling is "
                             "non-linear). No effect under --laplacian-pyramid (already linear).")
    parser.add_argument(
        "--block-normalization", type=str, default="batchnorm",
        choices=["layernorm", "batchnorm"],
        help="Pre-activation normalization inside every ConvNeXt block (wired to "
             "create_convunext_denoiser block_normalization). 'batchnorm' (default) = "
             "variance-only BiasFreeBatchNorm (no mean, no beta) that restores degree-1 "
             "homogeneity f(ax)=a*f(x) at inference (pairs best with a homogeneous "
             "activation like LeakyReLU); 'layernorm' = per-input scale-invariant "
             "(degree-0), byte-identical to legacy pre-batchnorm checkpoints.",
    )
    parser.add_argument(
        "--block-activation", type=str, default="leaky_relu",
        help="Activation for the WHOLE denoiser: the ConvNeXt blocks AND the "
             "ConvUNextStem AND the deep-supervision heads all use it. 'leaky_relu' "
             "(default) builds LeakyReLU(negative_slope=--block-activation-alpha); any "
             "other Keras activation name is passed through as a string. The final "
             "activation stays linear (bias-free homogeneity).",
    )
    parser.add_argument(
        "--block-activation-alpha", type=float, default=0.1,
        help="Negative slope for LeakyReLU when --block-activation=leaky_relu. Applies "
             "to the whole denoiser (blocks + stem + deep-supervision). Default 0.1. "
             "Ignored for non-leaky activations.",
    )
    parser.add_argument("--expose-bottleneck", action="store_true",
                        help="Expose the bottleneck latent as an optional second model output (default OFF)")
    parser.add_argument("--analyzer", action="store_true",
                        help="Run ModelAnalyzer (data-free weight + spectral) during training (default OFF)")
    parser.add_argument("--analyzer-freq", type=int, default=10,
                        help="Run the analyzer every N epochs (with --analyzer)")
    parser.add_argument("--gabor-filters", type=int, default=32)
    parser.add_argument("--gabor-kernel-size", type=int, default=11,
                        help="Spatial size of the frozen Gabor depthwise stem (default 11).")
    parser.add_argument("--gabor-activation", type=str, default=None,
                        choices=sorted(GABOR_ACTIVATIONS),
                        help="Activation on the frozen Gabor stem. Default: none (linear "
                             "passthrough of the raw signed Gabor responses). Restricted to "
                             "positively homogeneous activations -- anything else breaks the "
                             "degree-1 homogeneity the bias-free denoisers rely on. Note the "
                             "bank has no phase-reversed filter pairs, so 'relu' discards each "
                             "filter's negative lobe with no sibling filter to recover it.")
    parser.add_argument("--no-gabor-projection", action="store_true",
                        help="Drop the 1x1 projection after the Gabor stem and feed the "
                             "depthwise bank straight into the encoder. Requires "
                             "channels*gabor_filters == initial_filters exactly (e.g. "
                             "--gabor-filters 32 --initial-filters 96 for 3-channel input).")
    parser.add_argument("--initial-filters", type=int, default=None,
                        help="Override the variant's level-0 width (initial_filters). "
                             "Default: variant value. Use with --no-gabor-projection so "
                             "channels*gabor_filters == initial_filters.")
    parser.add_argument("--filter-multiplier", type=float, default=2.0,
                        help="Per-encoder-level channel-growth multiplier (>=1). "
                             "channels[level]=round(initial_filters * multiplier**level). "
                             "Default 2.0 doubles per level.")
    parser.add_argument("--depth", type=int, default=None,
                        help="Override the variant's number of U-Net levels; "
                             "None = use variant preset (>=2).")
    parser.add_argument("--blocks-per-level", type=int, default=None,
                        dest="blocks_per_level",
                        help="Blocks per U-Net level; "
                             "None = use variant preset (>=1).")
    parser.add_argument("--final-projection-groups", type=int, default=1,
                        help="Groups for the final 1x1 output projection. 1=standard dense "
                             "(default). -1 = one group per output channel (groups=channels), "
                             "so each output channel reads a disjoint feature group. Any >1 "
                             "sets the group count directly. Requires initial_filters and "
                             "channels both divisible by the resolved group count.")
    parser.add_argument("--sigma-max-start", type=float, default=0.025)
    parser.add_argument("--sigma-max-end", type=float, default=0.25)
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
        "--smoke", action="store_true",
        help="Tiny end-to-end mechanism check (few steps/epochs, constant LR).",
    )


def reject_self_iterate_with_nonadditive(parser, args) -> None:
    """Parse-time guard: ``--self-iterate`` is incompatible with non-additive noise.

    Each trainer calls this right after ``parser.parse_args()`` (fail fast, before any
    GPU/data work). Mirrors the ``__post_init__`` ValueError belt-and-suspenders."""
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
