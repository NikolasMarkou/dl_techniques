"""
Training script for CliffordNet vision backbone.

Trains CliffordNet on CIFAR-10 or CIFAR-100 following the protocol from
arXiv:2601.06793v2: AdamW optimiser, cosine LR schedule with linear warmup,
random flip/crop, and random erasing (Cutout).
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import keras
from datetime import datetime
from typing import Dict, Any, List, Literal, Tuple, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.cliffordnet import CliffordNet
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)

# ---------------------------------------------------------------------

Dataset = Literal["cifar10", "cifar100"]
Variant = Literal["nano", "lite", "lite_g", "custom"]


# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------


class LRLoggingCallback(keras.callbacks.Callback):
    """Log the current learning rate at the end of each epoch.

    :param log_interval: Log every this many epochs. Defaults to 1.
    """

    def __init__(self, log_interval: int = 1) -> None:
        super().__init__()
        self.log_interval = log_interval

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log LR if the epoch index is a multiple of ``log_interval``.

        :param epoch: Zero-based epoch index.
        :param logs: Metric dict from training (unused).
        """
        if (epoch + 1) % self.log_interval == 0:
            try:
                lr = float(
                    keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)
                )
            except Exception:
                lr = float(self.model.optimizer.learning_rate)
            logger.info(f"Epoch {epoch + 1}: learning_rate={lr:.6e}")


# ---------------------------------------------------------------------
# Custom augmentation layer
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class RandomCutout(keras.layers.Layer):
    """Apply random Cutout / random erasing to a batch of images.

    Erases ``num_patches`` rectangular regions of size ``patch_size x patch_size``
    per image by filling them with ``fill_value``.  Applied only during
    training (``training=True``).

    :param num_patches: Number of erased patches per image. Defaults to 1.
    :param patch_size: Side length of each erased square. Defaults to 8.
    :param fill_value: Scalar fill value for erased pixels. Defaults to 0.0.
    :param kwargs: Passed to :class:`keras.layers.Layer`.
    """

    def __init__(
        self,
        num_patches: int = 1,
        patch_size: int = 8,
        fill_value: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if num_patches < 0:
            raise ValueError(f"num_patches must be >= 0, got {num_patches}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.fill_value = fill_value

    def _erase_single(self, image: tf.Tensor) -> tf.Tensor:
        """Erase ``num_patches`` random regions from a single image.

        :param image: Float tensor ``(H, W, C)``.
        :return: Augmented image ``(H, W, C)``.
        """
        h = tf.shape(image)[0]
        w = tf.shape(image)[1]
        half = self.patch_size // 2
        result = image
        # Python loop unrolls at trace time; num_patches is a fixed int.
        for _ in range(self.num_patches):
            cy = tf.random.uniform((), minval=0, maxval=h, dtype=tf.int32)
            cx = tf.random.uniform((), minval=0, maxval=w, dtype=tf.int32)
            y1 = tf.maximum(0, cy - half)
            y2 = tf.minimum(h, cy + half)
            x1 = tf.maximum(0, cx - half)
            x2 = tf.minimum(w, cx + half)
            y_in = tf.logical_and(tf.range(h) >= y1, tf.range(h) < y2)   # (H,)
            x_in = tf.logical_and(tf.range(w) >= x1, tf.range(w) < x2)   # (W,)
            region = tf.cast(
                tf.logical_and(y_in[:, tf.newaxis], x_in[tf.newaxis, :]),
                result.dtype,
            )[:, :, tf.newaxis]  # (H, W, 1)
            result = result * (1.0 - region) + self.fill_value * region
        return result

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Apply cutout to ``inputs`` when ``training=True``.

        :param inputs: Float batch ``(B, H, W, C)``.
        :param training: Whether in training mode.
        :return: Augmented batch ``(B, H, W, C)``.
        """
        if not training or self.num_patches == 0:
            return inputs
        return tf.map_fn(self._erase_single, inputs)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return unchanged input shape.

        :param input_shape: Input shape tuple.
        :return: Same shape as input.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return serialisable config.

        :return: Config dict.
        """
        config = super().get_config()
        config.update(
            {
                "num_patches": self.num_patches,
                "patch_size": self.patch_size,
                "fill_value": self.fill_value,
            }
        )
        return config


# ---------------------------------------------------------------------
# GPU setup
# ---------------------------------------------------------------------


def setup_gpu() -> None:
    """Configure GPU memory growth to avoid OOM on first allocation."""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as exc:
            logger.error(f"GPU setup error: {exc}")
    else:
        logger.info("No GPUs found, using CPU")


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------


def load_dataset(
    dataset_name: Dataset,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], int]:
    """Load and normalise a CIFAR dataset.

    :param dataset_name: ``"cifar10"`` or ``"cifar100"``.
    :return: ``(x_train, y_train)``, ``(x_test, y_test)``, ``num_classes``.
    :raises ValueError: If ``dataset_name`` is not supported.
    """
    logger.info(f"Loading {dataset_name} dataset...")

    if dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        num_classes = 10
    elif dataset_name == "cifar100":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")

    y_train = y_train.flatten()
    y_test = y_test.flatten()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    logger.info(
        f"Dataset: {dataset_name} | train={x_train.shape[0]}, "
        f"test={x_test.shape[0]}, classes={num_classes}"
    )
    return (x_train, y_train), (x_test, y_test), num_classes


# ---------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------


def build_augmentation_pipeline(
    input_shape: Tuple[int, int, int],
    cutout_patches: int = 1,
    cutout_size: int = 8,
) -> keras.Sequential:
    """Build a training augmentation :class:`keras.Sequential` model.

    Applies random horizontal flip, pad-then-crop, and optional random
    erasing (Cutout).  Matches the augmentation protocol from the paper.

    :param input_shape: ``(H, W, C)`` of the training images.
    :param cutout_patches: Number of random-erase patches. ``0`` disables.
    :param cutout_size: Side length of each erased square patch.
    :return: A :class:`keras.Sequential` augmentation model.
    """
    h, w, _ = input_shape
    pad = 4

    aug_layers: List[keras.layers.Layer] = [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.ZeroPadding2D(padding=pad),
        keras.layers.RandomCrop(h, w),
    ]

    if cutout_patches > 0:
        aug_layers.append(
            RandomCutout(
                num_patches=cutout_patches,
                patch_size=cutout_size,
                name="random_cutout",
            )
        )

    return keras.Sequential(aug_layers, name="augmentation")


# ---------------------------------------------------------------------
# tf.data pipelines
# ---------------------------------------------------------------------


def build_train_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    augmentation: keras.Sequential,
) -> tf.data.Dataset:
    """Build a shuffled, augmented training dataset.

    :param x: Images ``(N, H, W, C)`` float32 in ``[0, 1]``.
    :param y: Integer labels ``(N,)``.
    :param batch_size: Mini-batch size.
    :param augmentation: Keras Sequential augmentation model.
    :return: Prefetched :class:`tf.data.Dataset`.
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(buffer_size=len(x), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(
        lambda imgs, labels: (augmentation(imgs, training=True), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def build_eval_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> tf.data.Dataset:
    """Build a non-shuffled evaluation dataset.

    :param x: Images ``(N, H, W, C)`` float32 in ``[0, 1]``.
    :param y: Integer labels ``(N,)``.
    :param batch_size: Mini-batch size.
    :return: Prefetched :class:`tf.data.Dataset`.
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------


def build_model(
    variant: Variant,
    num_classes: int,
    channels: int,
    depth: int,
    shifts: List[int],
    cli_mode: str,
    ctx_mode: str,
    use_global_context: bool,
    layer_scale_init: float,
    stochastic_depth_rate: float,
    dropout_rate: float,
    weight_decay: float,
) -> CliffordNet:
    """Instantiate a :class:`CliffordNet` model.

    :param variant: Pre-defined variant key or ``"custom"``.
    :param num_classes: Number of output classes.
    :param channels: Feature dimension ``D`` (only used for ``"custom"``).
    :param depth: Number of blocks (only used for ``"custom"``).
    :param shifts: Channel-shift offsets (only used for ``"custom"``).
    :param cli_mode: ``"inner"`` | ``"wedge"`` | ``"full"``.
    :param ctx_mode: ``"diff"`` | ``"abs"``.
    :param use_global_context: Whether to add global context branch.
    :param layer_scale_init: LayerScale initial value.
    :param stochastic_depth_rate: Max DropPath rate.
    :param dropout_rate: Classifier head dropout.
    :param weight_decay: L2 regularisation factor (applied via kernel_regularizer).
    :return: Instantiated :class:`CliffordNet`.
    :raises ValueError: If ``variant`` is not recognised.
    """
    reg = keras.regularizers.L2(weight_decay) if weight_decay > 0.0 else None

    shared_kwargs: Dict[str, Any] = dict(
        cli_mode=cli_mode,
        ctx_mode=ctx_mode,
        use_global_context=use_global_context,
        layer_scale_init=layer_scale_init,
        stochastic_depth_rate=stochastic_depth_rate,
        dropout_rate=dropout_rate,
        kernel_regularizer=reg,
    )

    if variant == "nano":
        model = CliffordNet.nano(num_classes=num_classes, **shared_kwargs)
    elif variant == "lite":
        model = CliffordNet.lite(num_classes=num_classes, **shared_kwargs)
    elif variant == "lite_g":
        model = CliffordNet.lite_g(num_classes=num_classes, **shared_kwargs)
    elif variant == "custom":
        model = CliffordNet(
            num_classes=num_classes,
            channels=channels,
            depth=depth,
            shifts=shifts,
            **shared_kwargs,
        )
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    logger.info(
        f"Created CliffordNet-{variant} | classes={num_classes} | "
        f"cli={cli_mode}, ctx={ctx_mode}, global={use_global_context}"
    )
    return model


# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------


def build_callbacks(
    results_dir: str,
    monitor: str = "val_accuracy",
    patience: int = 20,
    lr_log_interval: int = 1,
) -> List[keras.callbacks.Callback]:
    """Assemble training callbacks.

    :param results_dir: Directory for checkpoints and logs.
    :param monitor: Metric to monitor for early stopping and checkpointing.
    :param patience: Early-stopping patience in epochs.
    :param lr_log_interval: Log learning rate every this many epochs.
    :return: List of :class:`keras.callbacks.Callback` instances.
    """
    return [
        keras.callbacks.TerminateOnNaN(),
        LRLoggingCallback(log_interval=lr_log_interval),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(results_dir, "best_model.keras"),
            monitor=monitor,
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            filename=os.path.join(results_dir, "training_log.csv")
        ),
    ]


# ---------------------------------------------------------------------
# Save-load validation
# ---------------------------------------------------------------------


def validate_model_loading(
    model_path: str,
    test_input: np.ndarray,
    original_output: np.ndarray,
    rtol: float = 0.1,
) -> bool:
    """Verify that a saved CliffordNet reloads with matching predictions.

    :param model_path: Path to the saved ``.keras`` file.
    :param test_input: Small input batch used for comparison.
    :param original_output: Logits produced before saving.
    :param rtol: Relative tolerance threshold. Fails if ``rel_diff > rtol``.
    :return: ``True`` if relative max-difference is within ``rtol``.
    """
    try:
        loaded_model = keras.models.load_model(
            model_path,
            custom_objects={"CliffordNet": CliffordNet},
        )
        loaded_output = loaded_model.predict(test_input, verbose=0)
        max_diff = float(np.max(np.abs(loaded_output - original_output)))
        denom = float(np.max(np.abs(original_output))) + 1e-8
        rel_diff = max_diff / denom
        logger.info(
            f"Load validation ({os.path.basename(model_path)}): "
            f"max_diff={max_diff:.6f}, rel_diff={rel_diff:.4f}"
        )
        if rel_diff > rtol:
            logger.warning(
                f"Large prediction discrepancy after loading ({rel_diff:.4f} > {rtol})"
            )
            return False
        return True
    except Exception as exc:
        logger.error(f"Model loading validation failed: {exc}")
        return False


# ---------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------


def train_model(args: argparse.Namespace) -> None:
    """End-to-end training routine for CliffordNet.

    :param args: Parsed command-line arguments.
    """
    logger.info("Starting CliffordNet training")
    setup_gpu()

    # ---- Data --------------------------------------------------------
    (x_train, y_train), (x_test, y_test), num_classes = load_dataset(args.dataset)
    input_shape: Tuple[int, int, int] = x_train.shape[1:]

    augmentation = build_augmentation_pipeline(
        input_shape,
        cutout_patches=args.cutout_patches,
        cutout_size=args.cutout_size,
    )
    train_ds = build_train_dataset(x_train, y_train, args.batch_size, augmentation)
    val_ds = build_eval_dataset(x_test, y_test, args.batch_size)
    steps_per_epoch = len(x_train) // args.batch_size

    # ---- Shifts parsing (guard against empty string) ----------------
    shifts: List[int]
    if args.shifts:
        try:
            shifts = [int(s.strip()) for s in args.shifts.split(",") if s.strip()]
        except ValueError as exc:
            raise ValueError(
                f"Invalid --shifts value {args.shifts!r}. "
                "Expected comma-separated integers."
            ) from exc
    else:
        shifts = [1, 2]

    # ---- Model -------------------------------------------------------
    model = build_model(
        variant=args.variant,
        num_classes=num_classes,
        channels=args.channels,
        depth=args.depth,
        shifts=shifts,
        cli_mode=args.cli_mode,
        ctx_mode=args.ctx_mode,
        use_global_context=args.use_global_context,
        layer_scale_init=args.layer_scale_init,
        stochastic_depth_rate=args.stochastic_depth_rate,
        dropout_rate=args.dropout_rate,
        weight_decay=args.weight_decay,
    )

    # Warm-up build
    _ = model(np.zeros((1,) + input_shape, dtype="float32"), training=False)
    logger.info(f"Total parameters: {model.count_params():,}")

    # ---- Optimiser & schedule ----------------------------------------
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    # Guard: decay_steps must be at least 1 even if warmup_epochs >= epochs.
    decay_steps = max(1, total_steps - warmup_steps)

    lr_config: Dict[str, Any] = {
        "type": "cosine_decay",
        "learning_rate": args.learning_rate,
        "decay_steps": decay_steps,
        "alpha": 1e-2,
        "warmup_steps": warmup_steps,
        "warmup_start_lr": 1e-8,
    }

    opt_config: Dict[str, Any] = {
        "type": "adamw",
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": args.weight_decay,
    }

    lr_schedule = learning_rate_schedule_builder(lr_config)
    optimizer = optimizer_builder(opt_config, lr_schedule)

    logger.info(
        f"Schedule: cosine_decay | total_steps={total_steps}, "
        f"warmup_steps={warmup_steps}, decay_steps={decay_steps}, "
        f"peak_lr={args.learning_rate}"
    )

    # ---- Compile -----------------------------------------------------
    metrics: List[Any] = ["accuracy"]
    # Top-5 accuracy is only meaningful when num_classes > 10.
    if num_classes > 10:
        metrics.append(
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")
        )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=metrics,
    )

    # ---- Callbacks & output dirs -------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(
        args.output_dir,
        f"cliffordnet_{args.variant}_{args.dataset}_{timestamp}",
    )
    os.makedirs(results_dir, exist_ok=True)

    callbacks = build_callbacks(
        results_dir,
        patience=args.patience,
        lr_log_interval=max(1, args.epochs // 20),
    )

    logger.info(
        f"Training | dataset={args.dataset}, variant={args.variant}, "
        f"epochs={args.epochs}, batch={args.batch_size}, lr={args.learning_rate}"
    )

    # ---- Training ----------------------------------------------------
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    # ---- Save final model & validate load ----------------------------
    test_sample = x_test[:4]
    pre_save_output = model.predict(test_sample, verbose=0)

    final_path = os.path.join(
        results_dir,
        f"cliffordnet_{args.variant}_{args.dataset}_final.keras",
    )
    try:
        model.save(final_path)
        logger.info(f"Final model saved to: {final_path}")
        validate_model_loading(final_path, test_sample, pre_save_output)
    except Exception as exc:
        logger.warning(f"Failed to save / validate final model: {exc}")

    # ---- Load best checkpoint & evaluate ----------------------------
    best_path = os.path.join(results_dir, "best_model.keras")
    best_model = model  # fallback to current weights (already best via EarlyStopping)

    if os.path.exists(best_path):
        try:
            best_model = keras.models.load_model(
                best_path,
                custom_objects={"CliffordNet": CliffordNet},
            )
            logger.info("Loaded best checkpoint for final evaluation")
        except Exception as exc:
            logger.warning(
                f"Could not load best checkpoint ({exc}); "
                "using EarlyStopping-restored weights"
            )

    test_results = best_model.evaluate(val_ds, verbose=1, return_dict=True)
    logger.info(f"Test results: {test_results}")

    # ---- Training summary file ---------------------------------------
    val_acc_history: List[float] = history.history.get("val_accuracy", [])
    best_val_acc = max(val_acc_history) if val_acc_history else float("nan")
    trained_epochs = len(history.history.get("loss", []))
    convergence_epoch = next(
        (
            i
            for i, acc in enumerate(val_acc_history)
            if acc >= best_val_acc * 0.95
        ),
        trained_epochs,
    )

    summary_path = os.path.join(results_dir, "training_summary.txt")
    _write_summary(
        path=summary_path,
        args=args,
        input_shape=input_shape,
        num_classes=num_classes,
        shifts=shifts,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        trained_epochs=trained_epochs,
        test_results=test_results,
        best_val_acc=best_val_acc,
        convergence_epoch=convergence_epoch,
        total_params=model.count_params(),
    )
    logger.info(f"Summary written to: {summary_path}")
    logger.info(
        f"Training complete. Best val_accuracy={best_val_acc:.4f} "
        f"({model.count_params():,} params)"
    )


def _write_summary(
    path: str,
    args: argparse.Namespace,
    input_shape: Tuple[int, ...],
    num_classes: int,
    shifts: List[int],
    warmup_steps: int,
    decay_steps: int,
    trained_epochs: int,
    test_results: Dict[str, float],
    best_val_acc: float,
    convergence_epoch: int,
    total_params: int,
) -> None:
    """Write a plain-text training summary to ``path``.

    :param path: Destination file path.
    :param args: Parsed CLI arguments.
    :param input_shape: Training image shape.
    :param num_classes: Number of output classes.
    :param shifts: Resolved shift list used for the run.
    :param warmup_steps: Computed warmup step count.
    :param decay_steps: Computed decay step count.
    :param trained_epochs: Actual number of trained epochs.
    :param test_results: Dict from ``model.evaluate(..., return_dict=True)``.
    :param best_val_acc: Best validation accuracy observed.
    :param convergence_epoch: First epoch that reached 95 % of best val accuracy.
    :param total_params: Total parameter count.
    """
    lines = [
        "CliffordNet Training Summary",
        "=" * 40,
        "",
        f"Dataset          : {args.dataset}",
        f"Variant          : {args.variant}",
        f"Input shape      : {input_shape}",
        f"Num classes      : {num_classes}",
        "",
        "Model configuration:",
        f"  cli_mode       : {args.cli_mode}",
        f"  ctx_mode       : {args.ctx_mode}",
        f"  global context : {args.use_global_context}",
        f"  shifts         : {shifts}",
        f"  layer_scale    : {args.layer_scale_init}",
        f"  stoch_depth    : {args.stochastic_depth_rate}",
        f"  dropout        : {args.dropout_rate}",
        f"  weight_decay   : {args.weight_decay}",
        f"  parameters     : {total_params:,}",
        "",
        "Training configuration:",
        f"  epochs         : {trained_epochs}",
        f"  batch_size     : {args.batch_size}",
        f"  learning_rate  : {args.learning_rate}",
        f"  warmup_epochs  : {args.warmup_epochs}",
        f"  warmup_steps   : {warmup_steps}",
        f"  decay_steps    : {decay_steps}",
        f"  cutout_patches : {args.cutout_patches}",
        f"  cutout_size    : {args.cutout_size}",
        "",
        "Results:",
    ]
    for key, val in test_results.items():
        lines.append(f"  {key:<20}: {val:.4f}")
    lines += [
        "",
        f"  best_val_acc   : {best_val_acc:.4f}",
        f"  convergence    : epoch {convergence_epoch + 1}/{trained_epochs}",
    ]

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> None:
    """Entry point for the CliffordNet training script."""
    parser = argparse.ArgumentParser(
        description="Train CliffordNet (geometric-algebra vision backbone)"
    )

    # ---- Dataset / model ----
    parser.add_argument(
        "--dataset", type=str, default="cifar100",
        choices=["cifar10", "cifar100"],
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--variant", type=str, default="lite",
        choices=["nano", "lite", "lite_g", "custom"],
        help="Pre-defined variant or 'custom'.",
    )

    # ---- Custom variant overrides (ignored unless --variant custom) ----
    parser.add_argument(
        "--channels", type=int, default=64,
        help="Feature dimension D (custom variant only).",
    )
    parser.add_argument(
        "--depth", type=int, default=12,
        help="Number of blocks L (custom variant only).",
    )
    parser.add_argument(
        "--shifts", type=str, default=None,
        help=(
            "Comma-separated shift offsets, e.g. '1,2,4,8,16' "
            "(custom variant only; defaults to '1,2' if omitted)."
        ),
    )

    # ---- Algebraic / context options ----
    parser.add_argument(
        "--cli-mode", type=str, default="full",
        choices=["inner", "wedge", "full"],
        dest="cli_mode",
        help="Clifford interaction mode.",
    )
    parser.add_argument(
        "--ctx-mode", type=str, default="diff",
        choices=["diff", "abs"],
        dest="ctx_mode",
        help="Context generation mode (differential / absolute).",
    )
    parser.add_argument(
        "--use-global-context", action="store_true", default=False,
        dest="use_global_context",
        help="Add global-avg-pool context branch (gFFN-G).",
    )

    # ---- Regularisation ----
    parser.add_argument(
        "--layer-scale-init", type=float, default=1e-5,
        dest="layer_scale_init",
        help="LayerScale gamma initialisation value.",
    )
    parser.add_argument(
        "--stochastic-depth-rate", type=float, default=0.1,
        dest="stochastic_depth_rate",
        help="Maximum DropPath rate (linearly scheduled across blocks).",
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=0.0,
        dest="dropout_rate",
        help="Classifier head dropout rate.",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.05,
        dest="weight_decay",
        help="AdamW / L2 weight decay.",
    )

    # ---- Augmentation ----
    parser.add_argument(
        "--cutout-patches", type=int, default=1,
        dest="cutout_patches",
        help="Number of random-erase patches per image. 0 disables cutout.",
    )
    parser.add_argument(
        "--cutout-size", type=int, default=8,
        dest="cutout_size",
        help="Side length of each cutout square patch.",
    )

    # ---- Optimiser / schedule ----
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3,
        dest="learning_rate",
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=5,
        dest="warmup_epochs",
        help="Number of linear warm-up epochs (0 disables warmup).",
    )

    # ---- Training ----
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Total training epochs.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        dest="batch_size",
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--patience", type=int, default=30,
        help="Early-stopping patience in epochs.",
    )

    # ---- I/O ----
    parser.add_argument(
        "--output-dir", type=str, default="results",
        dest="output_dir",
        help="Root directory for checkpoints and logs.",
    )

    args = parser.parse_args()

    try:
        train_model(args)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        raise


# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()