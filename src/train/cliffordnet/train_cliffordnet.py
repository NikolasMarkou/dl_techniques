"""
Training script for CliffordNet vision backbone.

Trains CliffordNet on CIFAR-10 or CIFAR-100 following the protocol from
arXiv:2601.06793v2: AdamW optimiser, cosine LR schedule with linear warmup,
random flip/crop, and random erasing (Cutout).
"""

import os
import numpy as np
import tensorflow as tf
import keras
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

from train.common import (
    setup_gpu,
    create_base_argument_parser,
    load_dataset,
    get_class_names,
    create_callbacks,
    validate_model_loading,
    run_model_analysis,
)


# ---------------------------------------------------------------------

Dataset = Literal["cifar10", "cifar100"]
Variant = Literal["nano", "lite", "lite_g", "custom"]


# ---------------------------------------------------------------------
# Custom augmentation layer
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class RandomCutout(keras.layers.Layer):
    """Apply random Cutout / random erasing to a batch of images."""

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
        h = tf.shape(image)[0]
        w = tf.shape(image)[1]
        half = self.patch_size // 2
        result = image
        for _ in range(self.num_patches):
            cy = tf.random.uniform((), minval=0, maxval=h, dtype=tf.int32)
            cx = tf.random.uniform((), minval=0, maxval=w, dtype=tf.int32)
            y1 = tf.maximum(0, cy - half)
            y2 = tf.minimum(h, cy + half)
            x1 = tf.maximum(0, cx - half)
            x2 = tf.minimum(w, cx + half)
            y_in = tf.logical_and(tf.range(h) >= y1, tf.range(h) < y2)
            x_in = tf.logical_and(tf.range(w) >= x1, tf.range(w) < x2)
            region = tf.cast(
                tf.logical_and(y_in[:, tf.newaxis], x_in[tf.newaxis, :]),
                result.dtype,
            )[:, :, tf.newaxis]
            result = result * (1.0 - region) + self.fill_value * region
        return result

    def call(self, inputs, training=None):
        if not training or self.num_patches == 0:
            return inputs
        return tf.map_fn(self._erase_single, inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "patch_size": self.patch_size,
            "fill_value": self.fill_value,
        })
        return config


# ---------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------


def build_augmentation_pipeline(
    input_shape: Tuple[int, int, int],
    cutout_patches: int = 1,
    cutout_size: int = 8,
) -> keras.Sequential:
    """Build training augmentation: random flip, pad-then-crop, cutout."""
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
    x: np.ndarray, y: np.ndarray, batch_size: int,
    augmentation: keras.Sequential,
) -> tf.data.Dataset:
    """Build a shuffled, augmented training dataset."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(buffer_size=len(x), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(
        lambda imgs, labels: (augmentation(imgs, training=True), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def build_eval_dataset(
    x: np.ndarray, y: np.ndarray, batch_size: int,
) -> tf.data.Dataset:
    """Build a non-shuffled evaluation dataset."""
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
    """Instantiate a CliffordNet model."""
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
# Main training routine
# ---------------------------------------------------------------------


def train_model(args) -> None:
    """End-to-end training routine for CliffordNet."""
    logger.info("Starting CliffordNet training")
    setup_gpu(gpu_id=args.gpu)

    # ---- Data --------------------------------------------------------
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = load_dataset(
        args.dataset, batch_size=args.batch_size
    )

    augmentation = build_augmentation_pipeline(
        input_shape,
        cutout_patches=args.cutout_patches,
        cutout_size=args.cutout_size,
    )
    train_ds = build_train_dataset(x_train, y_train, args.batch_size, augmentation)
    val_ds = build_eval_dataset(x_test, y_test, args.batch_size)
    steps_per_epoch = len(x_train) // args.batch_size

    # ---- Shifts parsing ----------------------------------------------
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
    decay_steps = max(1, total_steps - warmup_steps)

    lr_schedule = learning_rate_schedule_builder({
        "type": "cosine_decay",
        "learning_rate": args.learning_rate,
        "decay_steps": decay_steps,
        "alpha": 1e-2,
        "warmup_steps": warmup_steps,
        "warmup_start_lr": 1e-8,
    })

    optimizer = optimizer_builder({
        "type": "adamw",
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": args.weight_decay,
    }, lr_schedule)

    logger.info(
        f"Schedule: cosine_decay | total_steps={total_steps}, "
        f"warmup_steps={warmup_steps}, decay_steps={decay_steps}, "
        f"peak_lr={args.learning_rate}"
    )

    # ---- Compile -----------------------------------------------------
    metrics: List[Any] = ["accuracy"]
    if num_classes > 10:
        metrics.append(
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")
        )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=metrics,
    )

    # ---- Callbacks ---------------------------------------------------
    custom_objects = {"CliffordNet": CliffordNet, "RandomCutout": RandomCutout}
    callbacks, results_dir = create_callbacks(
        model_name=f"{args.variant}_{args.dataset}",
        results_dir_prefix="cliffordnet",
        monitor="val_accuracy",
        patience=args.patience,
        use_lr_schedule=True,
    )
    callbacks.insert(0, keras.callbacks.TerminateOnNaN())

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

    # ---- Save final model & validate ---------------------------------
    test_sample = x_test[:4]
    pre_save_output = model.predict(test_sample, verbose=0)

    final_path = os.path.join(
        results_dir,
        f"cliffordnet_{args.variant}_{args.dataset}_final.keras",
    )
    try:
        model.save(final_path)
        logger.info(f"Final model saved to: {final_path}")
        validate_model_loading(final_path, test_sample, pre_save_output, custom_objects)
    except Exception as exc:
        logger.warning(f"Failed to save / validate final model: {exc}")

    # ---- Load best checkpoint & evaluate -----------------------------
    best_path = os.path.join(results_dir, "best_model.keras")
    best_model = model
    if os.path.exists(best_path):
        try:
            best_model = keras.models.load_model(best_path, custom_objects=custom_objects)
            logger.info("Loaded best checkpoint for final evaluation")
        except Exception as exc:
            logger.warning(f"Could not load best checkpoint ({exc}); using EarlyStopping-restored weights")

    test_results = best_model.evaluate(val_ds, verbose=1, return_dict=True)
    logger.info(f"Test results: {test_results}")

    # ---- Post-training analysis --------------------------------------
    run_model_analysis(
        model=best_model,
        test_data=(x_test, y_test),
        training_history=history,
        model_name=f"cliffordnet_{args.variant}_{args.dataset}",
        results_dir=results_dir,
    )

    # ---- Training summary --------------------------------------------
    val_acc_history: List[float] = history.history.get("val_accuracy", [])
    best_val_acc = max(val_acc_history) if val_acc_history else float("nan")
    trained_epochs = len(history.history.get("loss", []))

    summary_path = os.path.join(results_dir, "training_summary.txt")
    with open(summary_path, "w") as f:
        f.write("CliffordNet Training Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Variant: {args.variant}\n")
        f.write(f"Input shape: {input_shape}\n")
        f.write(f"Num classes: {num_classes}\n")
        f.write(f"Parameters: {model.count_params():,}\n\n")
        f.write(f"Epochs: {trained_epochs}, Batch: {args.batch_size}\n")
        f.write(f"LR: {args.learning_rate}, Weight decay: {args.weight_decay}\n\n")
        f.write("Test Results:\n")
        for key, val in test_results.items():
            f.write(f"  {key}: {val:.4f}\n")
        f.write(f"\nBest val_accuracy: {best_val_acc:.4f}\n")

    logger.info(f"Summary written to: {summary_path}")
    logger.info(f"Training complete. Best val_accuracy={best_val_acc:.4f}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> None:
    """Entry point for the CliffordNet training script."""
    parser = create_base_argument_parser(
        description="Train CliffordNet (geometric-algebra vision backbone)",
        default_dataset="cifar100",
        dataset_choices=["cifar10", "cifar100"],
    )

    # Model variant
    parser.add_argument('--variant', type=str, default='lite',
                        choices=['nano', 'lite', 'lite_g', 'custom'],
                        help='Pre-defined variant or custom.')

    # Custom variant overrides
    parser.add_argument('--channels', type=int, default=64,
                        help='Feature dimension D (custom variant only).')
    parser.add_argument('--depth', type=int, default=12,
                        help='Number of blocks L (custom variant only).')
    parser.add_argument('--shifts', type=str, default=None,
                        help="Comma-separated shift offsets, e.g. '1,2,4,8,16'.")

    # Algebraic / context options
    parser.add_argument('--cli-mode', type=str, default='full',
                        choices=['inner', 'wedge', 'full'], dest='cli_mode')
    parser.add_argument('--ctx-mode', type=str, default='diff',
                        choices=['diff', 'abs'], dest='ctx_mode')
    parser.add_argument('--use-global-context', action='store_true', default=False,
                        dest='use_global_context')

    # Regularisation
    parser.add_argument('--layer-scale-init', type=float, default=1e-5,
                        dest='layer_scale_init')
    parser.add_argument('--stochastic-depth-rate', type=float, default=0.1,
                        dest='stochastic_depth_rate')
    parser.add_argument('--dropout-rate', type=float, default=0.0,
                        dest='dropout_rate')

    # Augmentation
    parser.add_argument('--cutout-patches', type=int, default=1, dest='cutout_patches')
    parser.add_argument('--cutout-size', type=int, default=8, dest='cutout_size')

    # Warmup
    parser.add_argument('--warmup-epochs', type=int, default=5, dest='warmup_epochs')

    # Override defaults from base parser
    parser.set_defaults(weight_decay=0.05, batch_size=128, epochs=200, patience=30)

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
