"""
Training script for CliffordNet vision backbone.

Trains CliffordNet on CIFAR-10 or CIFAR-100 following the protocol from
arXiv:2601.06793v2: AdamW optimiser, cosine LR schedule with linear warmup,
AutoAugment (CIFAR-10 policy), random flip/crop, per-channel normalisation,
and random erasing.
"""

import os
import math
import numpy as np
import tensorflow as tf
import keras
from typing import Dict, Any, List, Literal, Tuple, Optional
from PIL import Image, ImageOps, ImageEnhance

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
# Per-channel normalisation constants (computed from training sets)
# ---------------------------------------------------------------------

_CIFAR_STATS: Dict[str, Dict[str, np.ndarray]] = {
    "cifar10": {
        "mean": np.array([0.4914, 0.4822, 0.4465], dtype=np.float32),
        "std": np.array([0.2470, 0.2435, 0.2616], dtype=np.float32),
    },
    "cifar100": {
        "mean": np.array([0.5071, 0.4867, 0.4408], dtype=np.float32),
        "std": np.array([0.2675, 0.2565, 0.2761], dtype=np.float32),
    },
}


# ---------------------------------------------------------------------
# CIFAR-10 AutoAugment policy (from the original AutoAugment paper)
# Each sub-policy is a list of (op_name, probability, magnitude_index).
# Magnitude indices map to operation-specific ranges (num_bins=31).
# ---------------------------------------------------------------------

_CIFAR10_POLICIES = [
    [("Invert", 0.1, 7), ("Contrast", 0.2, 6)],
    [("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)],
    [("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)],
    [("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)],
    [("AutoContrast", 0.5, 8), ("Equalize", 0.9, 2)],
    [("ShearY", 0.7, 2), ("Invert", 0.4, 7)],
    [("Color", 0.6, 1), ("Equalize", 1.0, 2)],
    [("Invert", 0.6, 4), ("Rotate", 0.6, 4)],
    [("Solarize", 0.3, 5), ("AutoContrast", 0.1, 12)],
    [("Equalize", 0.2, 4), ("Rotate", 0.6, 8)],
    [("Color", 0.1, 0), ("Brightness", 0.7, 2)],
    [("Sharpness", 0.4, 7), ("TranslateX", 0.9, 9)],
    [("ShearX", 0.6, 5), ("Rotate", 0.7, 2)],
    [("AutoContrast", 0.8, 4), ("Solarize", 0.4, 5)],
    [("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)],
    [("AutoContrast", 0.9, 2), ("Solarize", 0.8, 3)],
    [("Equalize", 0.8, 8), ("Invert", 0.1, 3)],
    [("TranslateY", 0.7, 9), ("AutoContrast", 0.9, 1)],
    [("Solarize", 0.4, 5), ("AutoContrast", 0.9, 3)],
    [("TranslateY", 0.7, 9), ("TranslateY", 0.7, 9)],
    [("AutoContrast", 0.9, 0), ("Solarize", 0.4, 3)],
    [("Equalize", 0.7, 5), ("Invert", 0.1, 3)],
    [("Equalize", 0.7, 8), ("TranslateY", 0.3, 9)],
    [("Rotate", 0.8, 7), ("TranslateY", 0.7, 9)],
    [("ShearX", 0.7, 5), ("Equalize", 0.8, 5)],
]

_NUM_MAGNITUDE_BINS = 31


def _apply_autoaugment_op(
    pil_img: Image.Image, op_name: str, magnitude_idx: int, img_size: int,
) -> Image.Image:
    """Apply a single AutoAugment operation to a PIL image.

    Magnitude mapping matches torchvision AutoAugment with num_bins=31.
    """
    m = magnitude_idx
    n = _NUM_MAGNITUDE_BINS - 1  # 30

    # --- Unsigned ops (no magnitude or special handling) ---
    if op_name == "Invert":
        return ImageOps.invert(pil_img)
    if op_name == "AutoContrast":
        return ImageOps.autocontrast(pil_img)
    if op_name == "Equalize":
        return ImageOps.equalize(pil_img)
    if op_name == "Posterize":
        bits = max(1, int(8 - round(m / (n / 4.0))))
        return ImageOps.posterize(pil_img, bits)
    if op_name == "Solarize":
        threshold = int(255 - m / n * 255)
        return ImageOps.solarize(pil_img, threshold)

    # --- Compute base magnitude ---
    if op_name in ("ShearX", "ShearY"):
        magnitude = m / n * 0.3
    elif op_name in ("TranslateX", "TranslateY"):
        magnitude = m / n * (150.0 / 331.0 * img_size)
    elif op_name == "Rotate":
        magnitude = m / n * 30.0
    elif op_name in ("Brightness", "Color", "Contrast", "Sharpness"):
        magnitude = m / n * 0.9
    else:
        return pil_img

    # --- Random sign for signed operations ---
    if np.random.random() > 0.5:
        magnitude = -magnitude

    # --- Enhancement operations ---
    if op_name == "Brightness":
        return ImageEnhance.Brightness(pil_img).enhance(1.0 + magnitude)
    if op_name == "Color":
        return ImageEnhance.Color(pil_img).enhance(1.0 + magnitude)
    if op_name == "Contrast":
        return ImageEnhance.Contrast(pil_img).enhance(1.0 + magnitude)
    if op_name == "Sharpness":
        return ImageEnhance.Sharpness(pil_img).enhance(1.0 + magnitude)

    # --- Geometric operations ---
    fill = (128, 128, 128)

    if op_name == "Rotate":
        return pil_img.rotate(magnitude, fillcolor=fill)
    if op_name == "ShearX":
        return pil_img.transform(
            pil_img.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0),
            fillcolor=fill,
        )
    if op_name == "ShearY":
        return pil_img.transform(
            pil_img.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0),
            fillcolor=fill,
        )
    if op_name == "TranslateX":
        pixels = int(round(magnitude))
        return pil_img.transform(
            pil_img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0),
            fillcolor=fill,
        )
    if op_name == "TranslateY":
        pixels = int(round(magnitude))
        return pil_img.transform(
            pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels),
            fillcolor=fill,
        )

    return pil_img


def _autoaugment_cifar10(image_np: np.ndarray) -> np.ndarray:
    """Apply CIFAR-10 AutoAugment policy to a float32 [0,1] HWC array."""
    pil_img = Image.fromarray(
        np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    )
    img_size = pil_img.size[0]

    idx = np.random.randint(len(_CIFAR10_POLICIES))
    sub_policy = _CIFAR10_POLICIES[idx]

    for op_name, prob, mag_idx in sub_policy:
        if np.random.random() > prob:
            continue
        pil_img = _apply_autoaugment_op(pil_img, op_name, mag_idx, img_size)

    return np.array(pil_img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------
# Random erasing (matches torchvision.transforms.RandomErasing defaults)
# Applied AFTER normalisation so fill_value=0 corresponds to the mean.
# ---------------------------------------------------------------------


def _random_erase(
    image_np: np.ndarray,
    sl: float = 0.02,
    sh: float = 0.33,
    r1: float = 0.3,
    r2: float = 3.3,
) -> np.ndarray:
    """Erase a random rectangle from a normalised image, filling with 0."""
    h, w, _ = image_np.shape
    area = h * w

    for _ in range(100):
        target_area = np.random.uniform(sl, sh) * area
        log_ratio = np.random.uniform(math.log(r1), math.log(r2))
        aspect = math.exp(log_ratio)

        eh = int(round(math.sqrt(target_area * aspect)))
        ew = int(round(math.sqrt(target_area / aspect)))

        if 0 < eh < h and 0 < ew < w:
            top = np.random.randint(0, h - eh)
            left = np.random.randint(0, w - ew)
            image_np = image_np.copy()
            image_np[top : top + eh, left : left + ew, :] = 0.0
            return image_np

    return image_np


# ---------------------------------------------------------------------
# Per-image augmentation (train) — matches the reference PyTorch recipe:
#   RandomCrop(32, pad=4) → RandomHFlip → AutoAugment → Normalize → RandomErasing
# ---------------------------------------------------------------------


def _make_train_augment_fn(
    dataset_name: str, random_erasing_prob: float = 0.25,
):
    """Create a per-image train augmentation function (numpy-level).

    Returned callable takes a float32 [0,1] HWC array and returns
    a normalised float32 HWC array with augmentations applied.
    """
    stats = _CIFAR_STATS[dataset_name]
    mean = stats["mean"]
    std = stats["std"]
    erasing_prob = random_erasing_prob

    def augment(image_np: np.ndarray) -> np.ndarray:
        # 1. Random horizontal flip
        if np.random.random() > 0.5:
            image_np = image_np[:, ::-1, :].copy()

        # 2. Pad-4 + random crop back to 32×32
        padded = np.pad(image_np, ((4, 4), (4, 4), (0, 0)), mode="constant")
        top = np.random.randint(0, 9)  # 0..8 inclusive (40-32=8)
        left = np.random.randint(0, 9)
        image_np = padded[top : top + 32, left : left + 32, :]

        # 3. AutoAugment (CIFAR-10 policy, applied on [0,1] images)
        image_np = _autoaugment_cifar10(image_np)

        # 4. Per-channel normalisation
        image_np = (image_np - mean) / std

        # 5. Random erasing (on normalised images, fill=0 = dataset mean)
        if np.random.random() < erasing_prob:
            image_np = _random_erase(image_np)

        return image_np.astype(np.float32)

    return augment


# ---------------------------------------------------------------------
# tf.data pipelines
# ---------------------------------------------------------------------


def build_train_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    dataset_name: str,
    random_erasing_prob: float = 0.25,
) -> tf.data.Dataset:
    """Build a shuffled, augmented, normalised training dataset."""
    augment_fn = _make_train_augment_fn(dataset_name, random_erasing_prob)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(buffer_size=len(x), reshuffle_each_iteration=True)

    def _map_fn(image, label):
        [image] = tf.numpy_function(augment_fn, [image], [tf.float32])
        image.set_shape([32, 32, 3])
        return image, label

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_eval_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    dataset_name: str,
) -> tf.data.Dataset:
    """Build a normalised (no augmentation) evaluation dataset."""
    stats = _CIFAR_STATS[dataset_name]
    mean = tf.constant(stats["mean"])
    std = tf.constant(stats["std"])

    def _normalize(image, label):
        return (image - mean) / std, label

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
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

    train_ds = build_train_dataset(
        x_train, y_train, args.batch_size, args.dataset,
        random_erasing_prob=args.random_erasing_prob,
    )
    val_ds = build_eval_dataset(x_test, y_test, args.batch_size, args.dataset)
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
    custom_objects = {"CliffordNet": CliffordNet}
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
        f.write(f"LR: {args.learning_rate}, Weight decay: {args.weight_decay}\n")
        f.write(f"Stochastic depth: {args.stochastic_depth_rate}\n\n")
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
    parser.add_argument('--variant', type=str, default='nano',
                        choices=['nano', 'lite', 'lite_g', 'custom'],
                        help='Pre-defined variant or custom.')

    # Custom variant overrides
    parser.add_argument('--channels', type=int, default=128,
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
    parser.add_argument('--stochastic-depth-rate', type=float, default=0.3,
                        dest='stochastic_depth_rate')
    parser.add_argument('--dropout-rate', type=float, default=0.0,
                        dest='dropout_rate')

    # Augmentation
    parser.add_argument('--random-erasing-prob', type=float, default=0.25,
                        dest='random_erasing_prob')

    # Warmup
    parser.add_argument('--warmup-epochs', type=int, default=5, dest='warmup_epochs')

    # Override defaults from base parser to match the reference recipe
    parser.set_defaults(
        weight_decay=0.1,
        batch_size=128,
        epochs=200,
        patience=30,
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
