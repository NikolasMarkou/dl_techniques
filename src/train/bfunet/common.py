"""Shared building blocks for the bfunet denoiser trainers (train_convunext_denoiser.py and train_cliffordunet_denoiser.py). Data pipeline, curriculum noise, eval/PSNR helpers, dashboard, and callbacks live here once; each trainer imports/re-exports them."""

import csv
import json
import keras
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # headless: avoid X11 crashes (LESSON)
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from train.common import augment_patch
from train.superpoint.homographic_adaptation import select_weighted_image_paths
from dl_techniques.utils.logger import logger
from dl_techniques.utils.multiplicative_miyasawa import (
    apply_multiplicative_gaussian,
    apply_composite_gaussian,
)
from dl_techniques.optimization import learning_rate_schedule_builder
from dl_techniques.callbacks.noise_sigma_curriculum import (
    NoiseSigmaCurriculumCallback,
)


def decode_full_image(
    image_path: tf.Tensor, config: "TrainingConfig"
) -> tf.Tensor:
    """Read + decode an image ONCE and normalize to [-0.5, +0.5].

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

    # Normalize to [-0.5, +0.5] (critical for bias-free architecture).
    image = (image / 255.0) - 0.5

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


def random_crop_patch(image: tf.Tensor, config: "TrainingConfig") -> tf.Tensor:
    """Extract one random ``patch_size`` crop from an already-decoded image."""
    return tf.image.random_crop(
        image, [config.patch_size, config.patch_size, config.channels]
    )


def load_and_preprocess_image(
    image_path: tf.Tensor, config: "TrainingConfig"
) -> tf.Tensor:
    """Decode an image, normalize to [-0.5, +0.5], and crop a single random patch.

    Thin compose of ``decode_full_image`` + ``random_crop_patch``, kept for the
    single-patch callers (self-iterate pool seed, fixed val-batch). The streaming
    ``create_dataset`` decodes once and crops many instead of calling this per patch.
    """
    return random_crop_patch(decode_full_image(image_path, config), config)


def collect_training_paths(config: "TrainingConfig") -> List[str]:
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
    config: "TrainingConfig",
    noise_fn,
    is_training: bool,
) -> tf.data.Dataset:
    """Build a tf.data pipeline of (noisy, clean) [-0.5,+0.5] patch pairs."""
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

    # Clip the clean patch back to [-0.5, +0.5] after augmentation. flips/rot90 preserve
    # range, but the aspect-safe bilinear upscale (small images) can overshoot; the
    # clean patch is both the model input and the regression target, so keep it in range.
    dataset = dataset.map(
        lambda x: tf.clip_by_value(x, -0.5, 0.5),
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
    config: "TrainingConfig",
    sigma_init: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a FIXED bounded RAM pool of clean patches + an initial noisy input pool.

    Loads ``config.self_iterate_pool_size`` clean patches ONCE (reusing the same
    ``load_and_preprocess_image`` load+crop logic the streaming pipeline uses, plus
    the same clip-to-[-0.5,+0.5] convention) into ``clean_pool``, then allocates a
    mutable ``current_input`` = ``clip(clean + N(0, sigma_init), -0.5, +0.5)`` as the
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
        in ``[-0.5, +0.5]``; ``clean_pool`` is the fixed target, ``current_input`` the
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
            patch = tf.clip_by_value(patch, -0.5, 0.5)
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
        clean_pool + noise * float(sigma_init), -0.5, 0.5
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
    config: "TrainingConfig",
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
    """Map a [-0.5, +0.5] image to [0, 1] for display."""
    return np.clip(img + 0.5, 0.0, 1.0)


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
    ax.set_xlabel("epoch"); ax.set_ylabel("sigma_max  [-0.5,+0.5] units")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left")
    if sig is not None:
        twin = ax.twinx()
        twin.set_ylabel("sigma on [0,255] scale")
        lo, hi = ax.get_ylim()
        twin.set_ylim(lo * 255.0, hi * 255.0)

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
    """Mean PSNR (dB) of ``pred`` vs ``clean`` on the [-0.5,+0.5] domain (max_val=1.0).

    Single source of truth for the trainer's PSNR convention: rmse over the whole
    batch, then ``20*log10(1.0/rmse)``. Both the eval grid and the multi-pass eval
    helpers call this so the formula lives in exactly one place (DRY).
    """
    mse = float(tf.reduce_mean(tf.square(tf.convert_to_tensor(pred) - clean)))
    return 20.0 * np.log10(1.0 / max(np.sqrt(mse), 1e-8))  # max_val=1.0


def denoise_k_passes(model: keras.Model, noisy, k: int) -> List[tf.Tensor]:
    """Apply ``model`` ``k`` times sequentially, clipping to [-0.5,+0.5] between passes.

    Returns the LIST of the k intermediate denoised tensors ``[pass1, ..., passk]``
    so callers can score each pass independently. The model is applied exactly once
    per pass (``training=False``); this is eval/inference only and never affects
    training. Domain [-0.5,+0.5] (clip after every pass).
    """
    outputs: List[tf.Tensor] = []
    x = tf.clip_by_value(tf.convert_to_tensor(noisy), -0.5, 0.5)
    for _ in range(int(k)):
        x = model(x, training=False)
        if isinstance(x, (list, tuple)):
            x = x[0]  # deep-supervision: primary output
        x = tf.clip_by_value(tf.convert_to_tensor(x), -0.5, 0.5)
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
    val_paths: List[str], config: "TrainingConfig", n: int = 8
) -> Optional[tf.Tensor]:
    """Load a small FIXED batch of clean [-0.5,+0.5] patches for visualization."""
    if not val_paths:
        return None
    patches = []
    for p in val_paths[: max(n * 3, n)]:
        try:
            patch = load_and_preprocess_image(tf.constant(p), config)
            patch = tf.clip_by_value(patch, -0.5, 0.5)
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
        hist, out, title=f"Training dashboard - {exp.name} ({len(epochs)} epochs)"
    )
    logger.info(f"Saved training dashboard: {out}")
    return out


def make_curriculum_noise_fn(config: "TrainingConfig", sigma_max_var: tf.Variable):
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
        return tf.clip_by_value(noisy, -0.5, 0.5), patch

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
