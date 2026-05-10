"""Production training script for ACC-UNet (binary / multi-class segmentation).

Pattern-4 trainer (denoising / detection family — segmentation is the third
member). Uses ``train.common.{setup_gpu, create_callbacks}`` for the standard
production callback stack and ``dl_techniques.optimization`` for the
optimizer / LR schedule builders.

Two data modes are supported out of the box:

* ``--data-mode synthetic`` — geometric circles + optional rectangles, no
  external dependency. Suitable for smoke-testing on CPU.
* ``--data-mode oxford_pets`` — TFDS ``oxford_iiit_pet`` collapsed to binary
  foreground/background masks. Resized to a multiple of 16 (AccUNet contract).

Usage
-----
Smoke (CPU, ~1 minute):

    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" .venv/bin/python -m train.accunet.train_accunet \\
        --data-mode synthetic --epochs 1 --batch-size 4 \\
        --image-size 64 --num-samples 32 --num-classes 1 \\
        --output-dir /tmp/accunet_smoke

Oxford-IIIT-Pets (single GPU):

    MPLBACKEND=Agg .venv/bin/python -m train.accunet.train_accunet \\
        --data-mode oxford_pets --image-size 128 --batch-size 16 \\
        --epochs 30 --gpu 0
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import keras
import matplotlib

# Force non-interactive backend before any pyplot import to support headless runs.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# dl_techniques imports
# ---------------------------------------------------------------------

from train.common import setup_gpu, create_callbacks
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)
from dl_techniques.models.accunet import (
    create_acc_unet_binary,
    create_acc_unet_multiclass,
)
from dl_techniques.losses.segmentation_loss import (
    create_loss_function,
    LossConfig,
)

# `weight_transfer` is optional — only needed when --init-from is passed.
try:
    from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
except ImportError:  # pragma: no cover - defensive
    load_weights_from_checkpoint = None  # type: ignore[assignment]


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass
class AccUNetTrainingConfig:
    """All knobs surfaced by the CLI, plus a few internal-only fields."""

    # Data
    data_mode: str = "synthetic"  # 'synthetic' | 'oxford_pets'
    num_classes: int = 1
    input_channels: int = 3
    image_size: int = 128
    num_samples: int = 1000  # synthetic only
    max_train_samples: Optional[int] = None  # oxford_pets cap
    max_val_samples: Optional[int] = None

    # Model
    base_filters: int = 32
    mlfc_iterations: int = 3

    # Optimization
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: float = 2.0
    lr_schedule_type: str = "cosine_decay"
    gradient_clipping: float = 1.0
    loss_name: str = "combo"
    monitor: str = "val_loss"
    early_stopping_patience: int = 15

    # System
    gpu: Optional[int] = None
    seed: int = 42
    output_dir: str = "results"
    init_from: Optional[str] = None
    grid_every_n_epochs: int = 5

    # Derived (filled by run_training)
    results_dir_prefix: str = field(default="accunet")


# ---------------------------------------------------------------------
# Synthetic data generator (port of train_convunext.generate_synthetic_data,
# adapted to (B, H, W, num_classes) labels for binary, (B, H, W) integer
# labels for multi-class, matching the loss expectations)
# ---------------------------------------------------------------------


def generate_synthetic_segmentation_data(
    num_samples: int,
    height: int,
    width: int,
    num_classes: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (image, mask) pairs of geometric shapes.

    Returns
    -------
    x : np.ndarray
        ``(N, H, W, 3)`` float32 in ``[0, 1]``.
    y : np.ndarray
        For ``num_classes == 1``: ``(N, H, W, 1)`` float32 in ``{0, 1}``.
        For ``num_classes >= 2``: ``(N, H, W)`` int32 with class IDs in
        ``[0, num_classes)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    logger.info(
        f"Generating {num_samples} synthetic samples "
        f"({height}x{width}, num_classes={num_classes})"
    )
    x = np.zeros((num_samples, height, width, 3), dtype="float32")
    y_int = np.zeros((num_samples, height, width), dtype="int32")

    for i in range(num_samples):
        x[i] = rng.random((height, width, 3), dtype="float32") * 0.1

        # Circle (class 1)
        cy = int(rng.integers(0, height))
        cx = int(rng.integers(0, width))
        cr = int(rng.integers(10, max(11, height // 4)))
        yy, xx = np.ogrid[:height, :width]
        circle = ((yy - cy) ** 2 + (xx - cx) ** 2) <= cr ** 2
        x[i][circle] += np.array([0.5, 0.0, 0.0], dtype="float32")
        y_int[i][circle] = 1

        # Rectangle (class 2) — only if multi-class
        if num_classes > 2:
            ry = int(rng.integers(0, max(1, height - 20)))
            rx = int(rng.integers(0, max(1, width - 20)))
            rh = int(rng.integers(10, 50))
            rw = int(rng.integers(10, 50))
            rect = np.zeros((height, width), dtype=bool)
            rect[ry:ry + rh, rx:rx + rw] = True
            x[i][rect] = (
                rng.random((int(rect.sum()), 3), dtype="float32") * 0.2
                + np.array([0.0, 0.5, 0.0], dtype="float32")
            )
            y_int[i][rect] = 2

    np.clip(x, 0.0, 1.0, out=x)

    if num_classes == 1:
        y = (y_int > 0).astype("float32")[..., np.newaxis]
    else:
        y = y_int  # sparse integer labels
    return x, y


# ---------------------------------------------------------------------
# Oxford-IIIT-Pet loader (binary foreground vs background)
# ---------------------------------------------------------------------


def _require_multiple_of_16(image_size: int) -> None:
    if image_size % 16 != 0:
        raise ValueError(
            f"--image-size must be divisible by 16 for AccUNet, got {image_size}."
        )


def load_oxford_pets_dataset(
    image_size: int,
    batch_size: int,
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    seed: int = 42,
):
    """Load Oxford-IIIT-Pet via TFDS, collapsed to binary foreground masks.

    Returns ``(train_ds, val_ds, steps_per_epoch, val_steps)``.
    Raises ``RuntimeError`` if ``tensorflow_datasets`` is not installed.
    """
    _require_multiple_of_16(image_size)

    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "oxford_pets data mode requires tensorflow_datasets. "
            "Install with `pip install tensorflow-datasets`."
        ) from exc

    logger.info(f"Loading oxford_iiit_pet via TFDS at {image_size}x{image_size}")
    (train_raw, val_raw), info = tfds.load(
        "oxford_iiit_pet:3.*.*",
        split=["train", "test"],
        with_info=True,
        as_supervised=False,
    )

    n_train_full = info.splits["train"].num_examples
    n_val_full = info.splits["test"].num_examples
    n_train = min(max_train, n_train_full) if max_train else n_train_full
    n_val = min(max_val, n_val_full) if max_val else n_val_full
    logger.info(f"oxford_pets: using {n_train} train / {n_val} val examples")

    def _preprocess(example):
        img = tf.image.resize(example["image"], (image_size, image_size))
        img = tf.cast(img, tf.float32) / 255.0
        # TFDS pet masks: 1=foreground, 2=background, 3=ambiguous-edge.
        mask = tf.image.resize(
            example["segmentation_mask"],
            (image_size, image_size),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )
        mask = tf.cast(mask, tf.int32)
        binary = tf.cast(tf.equal(mask, 1), tf.float32)  # 1 where pet
        return img, binary

    if max_train:
        train_raw = train_raw.take(max_train)
    if max_val:
        val_raw = val_raw.take(max_val)

    train_ds = (
        train_raw.shuffle(1024, seed=seed, reshuffle_each_iteration=True)
        .map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val_raw.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    steps_per_epoch = max(1, n_train // batch_size)
    val_steps = max(1, n_val // batch_size)
    return train_ds, val_ds, steps_per_epoch, val_steps


# ---------------------------------------------------------------------
# Binary Dice metric
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BinaryDiceMetric(keras.metrics.Metric):
    """Differentiable / running Dice coefficient for binary segmentation.

    Treats predictions in ``[0, 1]`` (post-sigmoid) and labels in ``{0, 1}``.
    Aggregates intersection and union across batches for a stable epoch-level
    estimate (rather than averaging per-batch Dice values, which is noisy on
    small batches).
    """

    def __init__(self, name: str = "dice", smooth: float = 1.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.union = self.add_weight(name="union", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = keras.ops.cast(y_true, "float32")
        y_pred = keras.ops.cast(y_pred, "float32")
        y_pred_b = keras.ops.cast(y_pred > 0.5, "float32")
        inter = keras.ops.sum(y_true * y_pred_b)
        uni = keras.ops.sum(y_true) + keras.ops.sum(y_pred_b)
        self.intersection.assign_add(inter)
        self.union.assign_add(uni)

    def result(self):
        return (2.0 * self.intersection + self.smooth) / (self.union + self.smooth)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"smooth": float(self.smooth)})
        return config


# ---------------------------------------------------------------------
# Visualization callback
# ---------------------------------------------------------------------


class SegmentationGridCallback(keras.callbacks.Callback):
    """Save a small ``[image | gt | pred]`` PNG grid every N epochs.

    Operates on a fixed pinned validation slice (``val_x``, ``val_y``) so the
    visual evolves smoothly across epochs. Only renders for binary or 2-class
    multi-class models (multi-class > 2 falls back to argmax-as-grayscale).
    """

    def __init__(
        self,
        val_x: np.ndarray,
        val_y: np.ndarray,
        output_dir: str,
        every_n_epochs: int = 5,
        max_samples: int = 4,
    ):
        super().__init__()
        self.val_x = np.asarray(val_x[:max_samples])
        self.val_y = np.asarray(val_y[:max_samples])
        self.output_dir = output_dir
        self.every_n_epochs = max(1, int(every_n_epochs))
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs=None):
        if (epoch + 1) % self.every_n_epochs != 0:
            return
        try:
            preds = self.model.predict(self.val_x, verbose=0)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"SegmentationGridCallback: predict failed: {exc}")
            return

        n = self.val_x.shape[0]
        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
        if n == 1:
            axes = axes[np.newaxis, :]
        for i in range(n):
            axes[i, 0].imshow(np.clip(self.val_x[i], 0, 1))
            axes[i, 0].set_title("image")
            axes[i, 0].axis("off")

            gt = self.val_y[i]
            if gt.ndim == 3 and gt.shape[-1] == 1:
                gt = gt[..., 0]
            axes[i, 1].imshow(gt, cmap="gray", vmin=0, vmax=max(1, int(gt.max())))
            axes[i, 1].set_title("gt")
            axes[i, 1].axis("off")

            p = preds[i]
            if p.ndim == 3 and p.shape[-1] == 1:
                p = p[..., 0]
            elif p.ndim == 3:
                p = np.argmax(p, axis=-1)
            axes[i, 2].imshow(p, cmap="gray", vmin=0, vmax=1 if p.dtype.kind == "f" else max(1, int(p.max())))
            axes[i, 2].set_title("pred")
            axes[i, 2].axis("off")

        path = os.path.join(self.output_dir, f"epoch_{epoch + 1:03d}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=80, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved segmentation grid -> {path}")


# ---------------------------------------------------------------------
# Build / compile / train
# ---------------------------------------------------------------------


def build_model(config: AccUNetTrainingConfig) -> keras.Model:
    """Construct AccUNet via the appropriate factory + materialize weights."""
    if config.num_classes == 1:
        model = create_acc_unet_binary(
            input_channels=config.input_channels,
            input_shape=(config.image_size, config.image_size),
            base_filters=config.base_filters,
            mlfc_iterations=config.mlfc_iterations,
        )
    else:
        model = create_acc_unet_multiclass(
            input_channels=config.input_channels,
            num_classes=config.num_classes,
            input_shape=(config.image_size, config.image_size),
            base_filters=config.base_filters,
            mlfc_iterations=config.mlfc_iterations,
        )

    # Force build with a dummy forward so weight counts / load_weights work.
    dummy = np.zeros(
        (1, config.image_size, config.image_size, config.input_channels),
        dtype="float32",
    )
    _ = model(dummy, training=False)

    if config.init_from:
        if load_weights_from_checkpoint is None:
            raise RuntimeError(
                "weight_transfer.load_weights_from_checkpoint not importable; "
                "cannot honor --init-from."
            )
        logger.info(f"Loading initial weights from {config.init_from}")
        load_weights_from_checkpoint(model, config.init_from)

    logger.info(f"Model built: {model.count_params():,} parameters")
    return model


def compile_model(
    model: keras.Model,
    config: AccUNetTrainingConfig,
    steps_per_epoch: int,
) -> None:
    decay_steps = max(1, int(steps_per_epoch * config.epochs))
    warmup_steps = max(0, int(steps_per_epoch * config.warmup_epochs))
    lr_schedule = learning_rate_schedule_builder({
        "type": config.lr_schedule_type,
        "learning_rate": float(config.lr),
        "decay_steps": decay_steps,
        "warmup_steps": warmup_steps,
    })
    optimizer = optimizer_builder(
        {
            "type": "adamw",
            "weight_decay": float(config.weight_decay),
            "gradient_clipping_by_norm": float(config.gradient_clipping),
        },
        lr_schedule,
    )

    loss_fn = create_loss_function(
        config.loss_name,
        LossConfig(num_classes=config.num_classes),
    )

    if config.num_classes == 1:
        metrics: List = ["binary_accuracy", BinaryDiceMetric(name="dice")]
    else:
        metrics = ["sparse_categorical_accuracy"]

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    logger.info(
        f"Compiled with loss={config.loss_name}, optimizer=adamw "
        f"(wd={config.weight_decay}, lr={config.lr}), schedule={config.lr_schedule_type}"
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:  # pragma: no cover
        pass
    keras.utils.set_random_seed(seed)


def run_training(config: AccUNetTrainingConfig) -> Tuple[keras.Model, str]:
    setup_gpu(config.gpu)
    _seed_everything(config.seed)

    # --- Data ---
    val_x_pinned: Optional[np.ndarray] = None
    val_y_pinned: Optional[np.ndarray] = None

    if config.data_mode == "synthetic":
        _require_multiple_of_16(config.image_size)
        rng = np.random.default_rng(config.seed)
        x_all, y_all = generate_synthetic_segmentation_data(
            config.num_samples,
            config.image_size,
            config.image_size,
            config.num_classes,
            rng=rng,
        )
        n_val = max(1, int(0.1 * config.num_samples))
        x_train, y_train = x_all[:-n_val], y_all[:-n_val]
        x_val, y_val = x_all[-n_val:], y_all[-n_val:]
        train_data = (x_train, y_train)
        val_data = (x_val, y_val)
        steps_per_epoch = max(1, len(x_train) // config.batch_size)
        val_x_pinned = x_val[: min(4, len(x_val))]
        val_y_pinned = y_val[: min(4, len(y_val))]
    elif config.data_mode == "oxford_pets":
        if config.num_classes != 1:
            raise ValueError("oxford_pets dataset only supports --num-classes 1")
        train_ds, val_ds, steps_per_epoch, _ = load_oxford_pets_dataset(
            image_size=config.image_size,
            batch_size=config.batch_size,
            max_train=config.max_train_samples,
            max_val=config.max_val_samples,
            seed=config.seed,
        )
        train_data, val_data = train_ds, val_ds
        # Materialize a pinned slice for the visualization callback.
        for batch in val_ds.take(1):
            vx, vy = batch
            val_x_pinned = vx.numpy()[:4]
            val_y_pinned = vy.numpy()[:4]
            break
    else:
        raise ValueError(f"Unknown --data-mode: {config.data_mode!r}")

    # --- Model ---
    model = build_model(config)
    compile_model(model, config, steps_per_epoch)

    # --- Callbacks ---
    model_name = f"accunet_{config.data_mode}_{config.image_size}"
    callbacks, results_dir = create_callbacks(
        model_name=model_name,
        results_dir_prefix=os.path.join(config.output_dir, "accunet"),
        monitor=config.monitor,
        patience=config.early_stopping_patience,
        use_lr_schedule=True,  # we own the LR schedule via optimizer_builder
        include_tensorboard=True,
        include_terminate_on_nan=True,
        include_analyzer=False,
    )
    if val_x_pinned is not None and val_y_pinned is not None:
        viz_dir = os.path.join(results_dir, "viz")
        callbacks.append(
            SegmentationGridCallback(
                val_x=val_x_pinned,
                val_y=val_y_pinned,
                output_dir=viz_dir,
                every_n_epochs=config.grid_every_n_epochs,
            )
        )

    # --- Fit ---
    logger.info(f"Starting fit: epochs={config.epochs}, steps/epoch={steps_per_epoch}")
    if isinstance(train_data, tuple):
        history = model.fit(
            train_data[0], train_data[1],
            validation_data=val_data,
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=callbacks,
            verbose=2,
        )
    else:
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=2,
        )

    # --- Save final + round-trip check ---
    final_path = os.path.join(results_dir, "final_model.keras")
    model.save(final_path)
    logger.info(f"Saved final model -> {final_path}")

    # NB: ``compile=False`` because ``dl_techniques.losses.segmentation_loss``
    # ``WrappedLoss`` (the loss returned by ``create_loss_function``) currently
    # serializes a ``reduction`` kwarg that its ``__init__`` doesn't accept,
    # so a full ``compile_from_config`` round-trip raises ``TypeError`` from
    # inside Keras' deserializer. The trainer-side round-trip check only
    # cares about the model topology + weights, so skip compile.
    loaded = keras.models.load_model(
        final_path,
        custom_objects={"BinaryDiceMetric": BinaryDiceMetric},
        compile=False,
    )
    sample = np.zeros(
        (1, config.image_size, config.image_size, config.input_channels),
        dtype="float32",
    )
    out_orig = np.asarray(model(sample, training=False))
    out_load = np.asarray(loaded(sample, training=False))
    if out_orig.shape != out_load.shape:
        raise RuntimeError(
            f"Round-trip shape mismatch: {out_orig.shape} vs {out_load.shape}"
        )
    max_diff = float(np.max(np.abs(out_orig - out_load)))
    logger.info(
        f"Round-trip OK: shape={out_orig.shape}, max|orig-loaded|={max_diff:.3e}"
    )

    return model, results_dir


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _parse_args() -> AccUNetTrainingConfig:
    p = argparse.ArgumentParser(
        description="Train ACC-UNet on a segmentation dataset (Pattern-4)."
    )
    # Data
    p.add_argument("--data-mode", choices=["synthetic", "oxford_pets"],
                   default="synthetic")
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--input-channels", type=int, default=3)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--num-samples", type=int, default=1000,
                   help="synthetic only: total samples to generate")
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="oxford_pets: cap training set size")
    p.add_argument("--max-val-samples", type=int, default=None,
                   help="oxford_pets: cap validation set size")
    # Model
    p.add_argument("--base-filters", type=int, default=32)
    p.add_argument("--mlfc-iterations", type=int, default=3)
    # Optimization
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=float, default=2.0)
    p.add_argument("--lr-schedule", dest="lr_schedule_type",
                   default="cosine_decay")
    p.add_argument("--gradient-clipping", type=float, default=1.0)
    p.add_argument("--loss", dest="loss_name", default="combo",
                   choices=["combo", "dice", "focal", "cross_entropy",
                            "focal_tversky", "tversky", "lovasz",
                            "boundary", "hausdorff"])
    p.add_argument("--monitor", default="val_loss")
    p.add_argument("--early-stopping-patience", type=int, default=15)
    p.add_argument("--grid-every-n-epochs", type=int, default=5)
    # System
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--init-from", default=None)

    ns = p.parse_args()

    # Multi-class default monitor is val_loss too; keep user's --monitor as-is.
    return AccUNetTrainingConfig(
        data_mode=ns.data_mode,
        num_classes=ns.num_classes,
        input_channels=ns.input_channels,
        image_size=ns.image_size,
        num_samples=ns.num_samples,
        max_train_samples=ns.max_train_samples,
        max_val_samples=ns.max_val_samples,
        base_filters=ns.base_filters,
        mlfc_iterations=ns.mlfc_iterations,
        epochs=ns.epochs,
        batch_size=ns.batch_size,
        lr=ns.lr,
        weight_decay=ns.weight_decay,
        warmup_epochs=ns.warmup_epochs,
        lr_schedule_type=ns.lr_schedule_type,
        gradient_clipping=ns.gradient_clipping,
        loss_name=ns.loss_name,
        monitor=ns.monitor,
        early_stopping_patience=ns.early_stopping_patience,
        grid_every_n_epochs=ns.grid_every_n_epochs,
        gpu=ns.gpu,
        seed=ns.seed,
        output_dir=ns.output_dir,
        init_from=ns.init_from,
    )


def main() -> int:
    config = _parse_args()
    try:
        _, results_dir = run_training(config)
        logger.info(f"Training complete. Results in {results_dir}")
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    except Exception as exc:  # pragma: no cover - top-level reporter
        logger.error(f"Training failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
