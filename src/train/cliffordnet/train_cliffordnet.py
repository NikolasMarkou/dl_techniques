"""
Training script for CliffordNet vision backbone.

Trains CliffordNet on CIFAR-10 or CIFAR-100 following the protocol from
arXiv:2601.06793v2 and its authors' train.py (github.com/ParaMind2025/CAN):
200 epochs, batch 128, AdamW (lr 1e-3, weight decay 0.1), cosine decay with NO
warm-up ("without using warm-up training epochs"), DropPath 0.3 (0.4 for cn_64_5),
AutoAugment (CIFAR-10 policy), random flip/crop, per-channel normalisation, and
random erasing 0.25. Remaining differences from the authors' run: the cosine here
decays per step and ends at 1e-5 (alpha 0.01), theirs per epoch to 0.

Label smoothing is opt-in (``--label-smoothing``, off in the paper recipe).

``--dataset imagenet`` trains on the extracted ImageNet-1k (``--imagenet-root``,
``--devkit-dir``) through ``train.common.imagenet``: random-resized-crop and flip,
ImageNet mean/std, no AutoAugment and no random erasing. The end-of-run figures,
model analysis and save/reload check use the first ``--eval-samples`` val images;
training-time validation and the final evaluation use all 50,000.
"""

import os
import math
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from typing import Dict, Any, List, Tuple, Optional, Sequence
from PIL import Image, ImageOps, ImageEnhance

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.datasets.vision.imagenet2012 import IMAGENET_MEAN, IMAGENET_STD
from dl_techniques.models.vision.cliffordnet import CliffordNet, HierarchicalCliffordNet
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)

from train.common import (
    setup_gpu,
    set_seeds,
    create_base_argument_parser,
    load_dataset,
    add_imagenet_arguments,
    eval_head,
    load_imagenet_splits,
    require_imagenet_root,
    create_callbacks,
    validate_model_loading,
    run_model_analysis,
    save_config_json,
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
)
from train.common import run_summary
from train.common.callbacks import LearningRateLogger
from train.common.classification_viz import TrainingDashboardCallback
from train.common.conformal import add_conformal_arguments, run_conformal
from train.common.datasets import get_class_names
from train.common.run_io import save_training_history_json
from train.convnext.common import build_loss


# ---------------------------------------------------------------------

Variant = str  # a key of CliffordNet.MODEL_VARIANTS, or "custom"

# DropPath of the authors' networks.py: 0.4 for cliffordnet_64_5, 0.3 for the rest.
_DEFAULT_STOCHASTIC_DEPTH = 0.3
_STOCHASTIC_DEPTH_BY_VARIANT: Dict[str, float] = {
    "cn_64_5": 0.4,
    # The authors' HierarchicalCliffordNet constructor default.
    "hier_345": 0.1,
    "custom_hier": 0.1,
}
# Variants built by HierarchicalCliffordNet: the presets of its own table plus "custom_hier".
_HIERARCHICAL_VARIANTS = (*HierarchicalCliffordNet.MODEL_VARIANTS, "custom_hier")


# ---------------------------------------------------------------------
# Per-channel normalisation constants (computed from training sets; the
# one home is train.common.datasets, D-112)
# ---------------------------------------------------------------------

_CIFAR_STATS: Dict[str, Dict[str, np.ndarray]] = {
    "cifar10": {
        "mean": np.array(CIFAR10_MEAN, dtype=np.float32),
        "std": np.array(CIFAR10_STD, dtype=np.float32),
    },
    "cifar100": {
        "mean": np.array(CIFAR100_MEAN, dtype=np.float32),
        "std": np.array(CIFAR100_STD, dtype=np.float32),
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


def _normalization_stats(
    dataset_name: str,
    normalization: Optional[Tuple[Sequence[float], Sequence[float]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """``(mean, std)`` as float32 arrays: the caller's, else the dataset's."""
    if normalization is None:
        stats = _CIFAR_STATS[dataset_name]
        return stats["mean"], stats["std"]
    mean, std = normalization
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def _make_train_augment_fn(
    dataset_name: str, random_erasing_prob: float = 0.25,
    normalization: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
):
    """Create a per-image train augmentation function (numpy-level).

    Returned callable takes a float32 [0,1] HWC array and returns
    a normalised float32 HWC array with augmentations applied. The
    normalisation is ``normalization`` (``(mean, std)``) when given, else the
    dataset's own statistics.
    """
    mean, std = _normalization_stats(dataset_name, normalization)
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
    normalization: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
) -> tf.data.Dataset:
    """Build a shuffled, augmented, normalised training dataset.

    ``normalization`` (``(mean, std)``) overrides the dataset statistics; the
    CliffordCLIP vision pretraining passes the contrastive stage's (D-090).
    """
    augment_fn = _make_train_augment_fn(
        dataset_name, random_erasing_prob, normalization)

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
    normalization: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
) -> tf.data.Dataset:
    """Build a normalised (no augmentation) evaluation dataset.

    ``normalization`` (``(mean, std)``) overrides the dataset statistics.
    """
    mean_np, std_np = _normalization_stats(dataset_name, normalization)
    mean = tf.constant(mean_np)
    std = tf.constant(std_np)

    def _normalize(image, label):
        return (image - mean) / std, label

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------


def _parse_int_list(text: Optional[str], flag: str) -> Optional[List[int]]:
    """Parse a comma-separated integer list; ``None`` or empty gives ``None``.

    :param text: The flag's value.
    :param flag: The flag name, for the error message.
    :raises ValueError: If an entry is not an integer.
    """
    if not text:
        return None
    try:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(
            f"Invalid {flag} value {text!r}. Expected comma-separated integers."
        ) from exc


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
    stage_depths: Optional[List[int]] = None,
    stage_dims: Optional[List[int]] = None,
) -> keras.Model:
    """Instantiate a CliffordNet or HierarchicalCliffordNet model.

    ``channels``, ``depth`` and ``use_global_context`` belong to the isotropic
    variants; ``stage_depths`` and ``stage_dims`` to ``custom_hier``. The
    hierarchical model has no global branch, so it refuses
    ``use_global_context=True`` instead of dropping it.

    Note: weight decay is handled solely by the AdamW optimiser
    (decoupled weight decay), matching the reference PyTorch recipe.
    No L2 kernel regularizer is applied — that would double-penalise.
    """
    shared_kwargs: Dict[str, Any] = dict(
        cli_mode=cli_mode,
        ctx_mode=ctx_mode,
        use_global_context=use_global_context,
        layer_scale_init=layer_scale_init,
        stochastic_depth_rate=stochastic_depth_rate,
        dropout_rate=dropout_rate,
    )

    if variant in _HIERARCHICAL_VARIANTS:
        if use_global_context:
            raise ValueError(
                f"{variant!r} is hierarchical and has no global-context branch"
            )
        hier_kwargs = {k: v for k, v in shared_kwargs.items() if k != "use_global_context"}
        if variant == "custom_hier":
            model = HierarchicalCliffordNet(
                num_classes=num_classes,
                stage_depths=stage_depths,
                stage_dims=stage_dims,
                shifts=shifts,
                **hier_kwargs,
            )
        else:
            model = HierarchicalCliffordNet.from_variant(
                variant, num_classes=num_classes, **hier_kwargs
            )
    elif variant in CliffordNet.MODEL_VARIANTS:
        model = CliffordNet.from_variant(
            variant, num_classes=num_classes, **shared_kwargs
        )
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
    set_seeds(args.seed)

    # ---- Data --------------------------------------------------------
    # x_eval / y_eval are the in-memory images the end-of-run figures, the model
    # analysis and the save/reload check read; the datasets feed fit and evaluate.
    figure_kwargs: Dict[str, Any] = {}
    if args.dataset == "imagenet":
        require_imagenet_root(args.imagenet_root)
        train_data, val_data = load_imagenet_splits(
            root=args.imagenet_root, devkit_dir=args.devkit_dir,
            image_size=args.image_size, batch_size=args.batch_size,
            seed=args.seed, augment=True,
        )
        train_ds, val_ds = train_data.dataset, val_data.dataset
        steps_per_epoch = train_data.steps_per_epoch
        num_classes = train_data.num_classes
        input_shape = (args.image_size, args.image_size, 3)
        class_names = train_data.class_names
        x_eval, y_eval = eval_head(val_data, args.eval_samples)
        # The images are standardized, so the figures un-standardize them.
        figure_kwargs = dict(
            mean=np.array(IMAGENET_MEAN, dtype=np.float32),
            std=np.array(IMAGENET_STD, dtype=np.float32),
        )
        # fit and the final evaluate see all of val; the figures see the head.
        figure_ds = tf.data.Dataset.from_tensor_slices(x_eval).batch(args.batch_size)
    else:
        (x_train, y_train), (x_eval, y_eval), input_shape, num_classes = load_dataset(
            args.dataset
        )
        train_ds = build_train_dataset(
            x_train, y_train, args.batch_size, args.dataset,
            random_erasing_prob=args.random_erasing_prob,
        )
        val_ds = build_eval_dataset(x_eval, y_eval, args.batch_size, args.dataset)
        steps_per_epoch = len(x_train) // args.batch_size
        class_names = get_class_names(args.dataset, num_classes)
        figure_ds = val_ds

    # ---- Shifts and stage lists parsing ------------------------------
    shifts: List[int] = _parse_int_list(args.shifts, "--shifts") or [1, 2]
    stage_depths = _parse_int_list(args.stage_depths, "--stage-depths")
    stage_dims = _parse_int_list(args.stage_dims, "--stage-dims")

    # ---- Model -------------------------------------------------------
    model = build_model(
        variant=args.variant,
        num_classes=num_classes,
        channels=args.channels,
        depth=args.depth,
        shifts=shifts,
        stage_depths=stage_depths,
        stage_dims=stage_dims,
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
    model.summary(print_fn=logger.info)

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

    if args.label_smoothing > 0.0:
        logger.info(
            f"Label smoothing {args.label_smoothing:g}: loss and val_loss include "
            "the smoothing and are not comparable with unsmoothed runs"
        )
    model.compile(
        optimizer=optimizer,
        loss=build_loss(args.label_smoothing),
        metrics=metrics,
    )

    # ---- Callbacks ---------------------------------------------------
    custom_objects = {
        "CliffordNet": CliffordNet,
        "HierarchicalCliffordNet": HierarchicalCliffordNet,
    }
    callbacks, results_dir = create_callbacks(
        model_name=f"{args.variant}_{args.dataset}",
        results_dir_prefix="cliffordnet",
        output_root=args.output_dir,
        monitor="val_accuracy",
        patience=args.patience,
        use_lr_schedule=True,
        # DECISION plan-2026-10-23T100412-e7eaaa02/D-002
        # Per-epoch analysis stays opt-in: do NOT make it the default again.
        # The shared default (every epoch) was removed in b6903e109 for cost.
        include_analyzer=args.epoch_analysis_every > 0,
        analyzer_epoch_frequency=max(1, args.epoch_analysis_every),
        # start at N so the snapshots land on epochs N, 2N, ... (default start 1
        # would give 1, N+1, ...).
        analyzer_start_epoch=max(1, args.epoch_analysis_every),
    )
    callbacks.insert(0, keras.callbacks.TerminateOnNaN())
    # Standard classification visualizations (src/train/AGENTS.md): the learning
    # rate reaches the CSV and the dashboard, which is drawn last and redrawn on a
    # cadence into <run>/visualizations/.
    vis_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    callbacks.insert(0, LearningRateLogger(at_epoch_start=True))

    def _baseline(untrained: keras.Model) -> Dict[str, float]:
        measured = untrained.evaluate(val_ds, verbose=0, return_dict=True)
        return {"loss": measured["loss"], "accuracy": measured["accuracy"]}

    callbacks.append(TrainingDashboardCallback(
        out_path=os.path.join(vis_dir, "training_dashboard.png"),
        baseline_fn=_baseline,
        title=f"cliffordnet_{args.variant}_{args.dataset} (seed {args.seed})",
    ))

    # DECISION plan-2026-10-14T062616-81938e81/D-177: the config is written here,
    # before training, and NOT wrapped. A failed write raises before any compute
    # is spent. Do NOT move it back after fit inside try/except-and-warn: the run
    # then exited 0 with config.json missing. See decisions.md D-177.
    save_config_json({
        "dataset": args.dataset, "variant": args.variant,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "stochastic_depth_rate": args.stochastic_depth_rate,
        "dropout_rate": args.dropout_rate,
        "label_smoothing": args.label_smoothing,
        "cli_mode": args.cli_mode, "ctx_mode": args.ctx_mode,
        "use_global_context": args.use_global_context,
        "warmup_epochs": args.warmup_epochs,
        "random_erasing_prob": args.random_erasing_prob,
        "stage_depths": stage_depths, "stage_dims": stage_dims,
        "epoch_analysis_every": args.epoch_analysis_every,
        "input_shape": list(input_shape),
        "num_classes": num_classes,
        **({"imagenet_root": args.imagenet_root, "devkit_dir": args.devkit_dir,
            "eval_samples": args.eval_samples,
            "conformal_alpha": None if args.no_conformal else args.conformal_alpha}
           if args.dataset == "imagenet" else {}),
    }, results_dir)

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
    test_sample = x_eval[:4]
    pre_save_output = model.predict(test_sample, verbose=0)

    final_path = os.path.join(
        results_dir,
        f"cliffordnet_{args.variant}_{args.dataset}_final.keras",
    )
    # A save or reload error surfaces (D-085); a value mismatch is logged by
    # validate_model_loading itself.
    model.save(final_path)
    logger.info(f"Final model saved to: {final_path}")
    # After the save: a failed history write raises (D-174).
    save_training_history_json(history, results_dir)
    validate_model_loading(final_path, test_sample, pre_save_output, custom_objects)

    # ---- Load best checkpoint & evaluate -----------------------------
    best_path = os.path.join(results_dir, "best_model.keras")
    best_model = model
    if os.path.exists(best_path):
        # A best checkpoint that exists but does not load raises (D-183); the old
        # fallback scored the last-epoch weights under the best checkpoint's name.
        best_model = keras.models.load_model(best_path, custom_objects=custom_objects)
        logger.info("Loaded best checkpoint for final evaluation")

    test_results = best_model.evaluate(val_ds, verbose=1, return_dict=True)
    logger.info(f"Test results: {test_results}")

    # ---- Conformal prediction sets (ImageNet), over ALL of val ----------
    # Before the figures and the model analysis, which are the heavy steps.
    conformal: Optional[Dict[str, Any]] = None
    if args.dataset == "imagenet" and not args.no_conformal:
        conformal = run_conformal(
            best_model, val_ds, alpha=args.conformal_alpha, seed=args.seed,
            out_dir=Path(results_dir), input_shape=input_shape,
            model_name=f"cliffordnet_{args.variant}_imagenet",
            weights="best checkpoint, selected on this same validation set's "
                    "val_accuracy" if os.path.exists(best_path) else "last epoch",
            selection_overlaps_calibration=True,
        )

    # ---- Classification figures --------------------------------------
    # The model emits logits; every probability consumer gets a softmax. The eval
    # data is ordered, so the predictions line up with y_eval. For CIFAR x_eval is
    # in [0, 1] (normalisation happens inside the dataset), so no mean/std is
    # given; the ImageNet head is already standardized and comes with its own.
    logits = best_model.predict(figure_ds, verbose=0)
    probs = keras.ops.convert_to_numpy(keras.ops.softmax(logits, axis=-1))
    run_summary.write_classification_figures(
        Path(vis_dir), x_eval, y_eval, probs, class_names, **figure_kwargs,
    )

    # ---- Post-training analysis --------------------------------------
    run_model_analysis(
        model=best_model,
        test_data=(x_eval, y_eval),
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
        if args.dataset == "imagenet":
            f.write(f"ImageNet root: {args.imagenet_root}\n")
            f.write(f"Figures and analysis on the first {len(x_eval)} val images\n")
        f.write(f"Parameters: {model.count_params():,}\n\n")
        f.write(f"Epochs: {trained_epochs}, Batch: {args.batch_size}\n")
        f.write(f"LR: {args.learning_rate}, Weight decay: {args.weight_decay}\n")
        f.write(f"Stochastic depth: {args.stochastic_depth_rate}\n")
        f.write(f"Label smoothing: {args.label_smoothing}\n\n")
        f.write("Test Results:\n")
        for key, val in test_results.items():
            f.write(f"  {key}: {val:.4f}\n")
        if conformal is not None:
            splits = conformal["repeated_splits"]
            f.write(
                f"\nConformal sets (alpha {conformal['alpha']}, tau {conformal['tau']}): "
                f"coverage {splits['coverage_mean']:.4f} on random half splits "
                f"(min {splits['coverage_min']:.4f}), mean set size "
                f"{splits['mean_size']:.2f}\n"
            )
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
        dataset_choices=["cifar10", "cifar100", "imagenet"],
    )
    add_imagenet_arguments(parser, note="Used only when --dataset imagenet: ")
    add_conformal_arguments(parser, note="Used only when --dataset imagenet: ")
    parser.add_argument('--eval-samples', type=int, default=None, dest='eval_samples',
                        help='Only with --dataset imagenet: how many val images the '
                             'end-of-run figures, model analysis and save/reload '
                             'check use (default 2000). Training-time validation '
                             'and the final evaluation use all of val.')

    # Model variant
    parser.add_argument('--variant', type=str, default='nano',
                        choices=[*CliffordNet.MODEL_VARIANTS,
                                 *HierarchicalCliffordNet.MODEL_VARIANTS,
                                 'custom', 'custom_hier'],
                        help='A preset of CliffordNet.MODEL_VARIANTS or of '
                             'HierarchicalCliffordNet.MODEL_VARIANTS, custom '
                             '(isotropic, --channels/--depth/--shifts) or '
                             'custom_hier (--stage-depths/--stage-dims/--shifts).')

    # Custom variant overrides
    # None means "not given"; the isotropic variants resolve them to 128 / 12 in main().
    parser.add_argument('--channels', type=int, default=None,
                        help='Feature dimension D (custom variant only; default 128).')
    parser.add_argument('--depth', type=int, default=None,
                        help='Number of blocks L (custom variant only; default 12).')
    parser.add_argument('--stage-depths', type=str, default=None, dest='stage_depths',
                        help="custom_hier only: blocks per stage, e.g. '3,4,5'.")
    parser.add_argument('--stage-dims', type=str, default=None, dest='stage_dims',
                        help="custom_hier only: width per stage, e.g. '32,64,128' "
                             "(default: 32 doubling per stage).")
    parser.add_argument('--shifts', type=str, default=None,
                        help="Comma-separated shift offsets, e.g. '1,2,4,8,16'.")

    # Algebraic / context options
    # None means "the preset's value" (custom: full / diff / off), resolved in main().
    parser.add_argument('--cli-mode', type=str, default=None,
                        choices=['inner', 'wedge', 'full'], dest='cli_mode')
    parser.add_argument('--ctx-mode', type=str, default=None,
                        choices=['diff', 'abs'], dest='ctx_mode')
    parser.add_argument('--use-global-context', action='store_true', default=None,
                        dest='use_global_context',
                        help='Add the gFFN-G branch (on in lite_g and nano_g).')

    # Regularisation
    parser.add_argument('--layer-scale-init', type=float, default=1e-5,
                        dest='layer_scale_init')
    parser.add_argument('--stochastic-depth-rate', type=float, default=None,
                        dest='stochastic_depth_rate',
                        help='Maximum DropPath rate, linearly ramped over depth. '
                             'Default: 0.4 for cn_64_5, else 0.3.')
    parser.add_argument('--dropout-rate', type=float, default=0.0,
                        dest='dropout_rate')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        dest='label_smoothing',
                        help='Label smoothing in [0, 1) (0 = off, the stock loss). '
                             'Above 0 the loss and val_loss include the smoothing '
                             'and are not comparable with unsmoothed runs.')

    # Augmentation
    parser.add_argument('--random-erasing-prob', type=float, default=None,
                        dest='random_erasing_prob',
                        help='CIFAR only (default 0.25). ImageNet uses '
                             'random-resized-crop and flip, so the flag is refused there.')

    # Analysis
    parser.add_argument('--epoch-analysis-every', type=int, default=0,
                        dest='epoch_analysis_every',
                        help='Run the data-free weight/spectral model analyzer '
                             'every N epochs into <run>/epoch_analysis/ '
                             '(0 = off).')

    # Warmup
    parser.add_argument('--warmup-epochs', type=int, default=0, dest='warmup_epochs',
                        help='Linear warm-up epochs (paper v2: none).')

    # Output
    parser.add_argument('--output-dir', type=str, default='results', dest='output_dir',
                        help='Directory under which the timestamped run directory '
                             'is created (default: results).')

    # Override defaults from base parser to match the reference recipe
    parser.set_defaults(
        weight_decay=0.1,
        batch_size=128,
        epochs=200,
        patience=30,
    )
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for Python, NumPy and Keras (train.common.set_seeds)')

    args = parser.parse_args()
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error(f"--label-smoothing must be in [0, 1), got {args.label_smoothing}")
    # A flag that does not apply to the chosen dataset is refused, never ignored.
    if args.dataset == "imagenet":
        if args.random_erasing_prob is not None:
            parser.error("--random-erasing-prob only applies to the CIFAR datasets")
        args.random_erasing_prob = 0.0  # none is applied; config.json records that
        args.eval_samples = 2000 if args.eval_samples is None else args.eval_samples
        if args.eval_samples <= 0:
            parser.error(f"--eval-samples must be positive, got {args.eval_samples}")
    else:
        if args.eval_samples is not None:
            parser.error("--eval-samples only applies with --dataset imagenet")
        if args.random_erasing_prob is None:
            args.random_erasing_prob = 0.25
    # DECISION plan-2026-10-23T100412-e7eaaa02/D-009
    # Flags the user did not pass take the PRESET's value. Do NOT give these
    # three flags their own defaults again: always passing cli_mode='full' and
    # use_global_context=False as overrides made `--variant lite_g` train without
    # its global branch and would have turned cn_32_5 / cn_64_5 into 'full'.
    hierarchical = args.variant in _HIERARCHICAL_VARIANTS
    # A flag that does not apply to the chosen architecture is refused, never ignored.
    if hierarchical:
        for flag, given in (("--channels", args.channels), ("--depth", args.depth),
                            ("--use-global-context", args.use_global_context)):
            if given is not None:
                parser.error(
                    f"{flag} does not apply to the hierarchical variant "
                    f"{args.variant!r} (it has no such parameter)"
                )
    if args.variant != "custom_hier":
        for flag, given in (("--stage-depths", args.stage_depths),
                            ("--stage-dims", args.stage_dims)):
            if given is not None:
                parser.error(f"{flag} only applies with --variant custom_hier")
    if not hierarchical:
        args.channels = 128 if args.channels is None else args.channels
        args.depth = 12 if args.depth is None else args.depth
    preset = {
        **CliffordNet.MODEL_VARIANTS, **HierarchicalCliffordNet.MODEL_VARIANTS
    }.get(args.variant, {})
    for name, fallback in (("cli_mode", "full"), ("ctx_mode", "diff"),
                           ("use_global_context", False)):
        if getattr(args, name) is None:
            setattr(args, name, preset.get(name, fallback))
    if args.stochastic_depth_rate is None:
        args.stochastic_depth_rate = _STOCHASTIC_DEPTH_BY_VARIANT.get(
            args.variant, _DEFAULT_STOCHASTIC_DEPTH
        )

    try:
        train_model(args)
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        raise


# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
