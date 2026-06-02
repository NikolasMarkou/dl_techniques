"""
Vision Transformer Classification Training (resnet Pattern-4 style)

Training pipeline for ViT on CIFAR-10/CIFAR-100/ImageNet. Mirrors the
structure of ``train_resnet.py`` without the deep-supervision path
(DS is deferred to a follow-up plan, see plan_2026-05-12_f2d29729).

Supports variants: vit_pico / vit_tiny / vit_small / vit_base / vit_large /
vit_huge.

Usage:
    python -m train.vit.train_vit --dataset cifar10 --variant vit_pico \\
        --epochs 50 --batch-size 128 --learning-rate 3e-4 \\
        --optimizer adamw --weight-decay 0.05 --warmup-epochs 5 \\
        --output-dir results/vit_cifar10 --experiment-name iter1_baseline --gpu 0
"""

import os
import sys
import gc
import json
import time
import keras
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any, Union

from train.common import setup_gpu, create_callbacks as create_common_callbacks, save_config_json, convert_keras_history_to_training_history, CIFAR10_MEAN, CIFAR10_STD, make_imagenet_filesystem_dataset, EpochMetricsPlotCallback
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)
from dl_techniques.models.vit import ViT, create_vit
from dl_techniques.visualization import (
    VisualizationManager,
    TrainingHistory,
    PlotConfig,
    PlotStyle,
    ColorScheme,
    TrainingCurvesVisualization,
    ConfusionMatrixVisualization,
    NetworkArchitectureVisualization,
    ModelComparisonBarChart,
    ROCPRCurves,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for ViT classification training.

    Mirrors :class:`train.resnet.train_resnet.TrainingConfig` with ViT-specific
    additions (patch_size, dropout_rate, attention_dropout_rate). Deep-supervision
    fields are intentionally absent -- DS for ViT is deferred to a follow-up plan.
    """

    # Data
    dataset: str = "cifar10"  # cifar10 | cifar100 | imagenet
    train_data_dir: Optional[str] = None  # only used for imagenet
    val_data_dir: Optional[str] = None
    image_size: int = 32  # 32 for CIFAR, 224 for ImageNet
    batch_size: int = 128

    # Model
    model_variant: str = "vit_pico"
    num_classes: int = 10
    pretrained: Union[bool, str] = False
    patch_size: int = 4  # 4 for CIFAR/32x32, 16 for ImageNet/224x224
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    # Training
    epochs: int = 50
    learning_rate: float = 3e-4
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "cosine_decay"
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    gradient_clipping: float = 1.0
    momentum: float = 0.9  # used only when optimizer_type='sgd'

    # Augmentation
    augment_data: bool = True
    label_smoothing: float = 0.1

    # Monitoring
    monitor_every_n_epochs: int = 5
    save_best_only: bool = True
    early_stopping_patience: int = 15
    validation_steps: Optional[int] = None

    # Data pipeline
    cache_dataset: bool = False
    prefetch_buffer: int = tf.data.AUTOTUNE
    num_parallel_calls: int = tf.data.AUTOTUNE

    # Output
    output_dir: str = "results"
    experiment_name: Optional[str] = None
    save_training_samples: bool = True
    save_model_checkpoints: bool = True

    # Convergence guard (DECISION plan_2026-05-12_f2d29729/D-007)
    # If None, threshold auto-derives to min(max(2/num_classes, 0.05), 0.95).
    success_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"vit_{self.dataset}_{self.model_variant}_{timestamp}"

        if self.image_size <= 0:
            raise ValueError("Invalid image_size: must be positive")
        if self.num_classes <= 0:
            raise ValueError("Invalid num_classes: must be positive")
        if self.dataset == "imagenet":
            if not self.train_data_dir or not Path(self.train_data_dir).exists():
                raise ValueError(
                    f"--train-data-dir is required for --dataset imagenet "
                    f"(got: {self.train_data_dir!r})"
                )
            if not self.val_data_dir or not Path(self.val_data_dir).exists():
                raise ValueError(
                    f"--val-data-dir is required for --dataset imagenet "
                    f"(got: {self.val_data_dir!r})"
                )


# =============================================================================
# CIFAR DATA PIPELINE
# =============================================================================

def get_cifar_preprocessing() -> Tuple[List[float], List[float]]:
    """Return CIFAR mean/std for normalization."""
    return CIFAR10_MEAN, CIFAR10_STD


def _cifar_augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Standard CIFAR augmentation on [0,1]-valued float32 images.

    DECISION plan_2026-05-12_f2d29729/D-007: this function operates on the
    [0,1]-valued pre-normalization tensors. Augmentation MUST run BEFORE
    per-channel mean/std normalization (which is applied in
    `_cifar_normalize` via a subsequent `.map`). The previous version
    applied augmentation AFTER numpy-level normalization and then
    `tf.clip_by_value(image, 0.0, 1.0)`, which saturated most pixels to
    {0,1} on normalized data (range ~[-1.99, +2.06]) and produced a
    train/val distribution mismatch (val skipped augment). Do NOT add a
    `clip_by_value` here; brightness ±0.1 and contrast 0.9-1.1 may
    transiently nudge a few pixels slightly outside [0,1] but the
    subsequent normalization is unaffected and the upstream pipelines in
    train_resnet.py follow the same no-clip convention.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(
        tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode="REFLECT"),
        size=tf.shape(image),
    )
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label


def _cifar_normalize(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Per-channel CIFAR mean/std normalization.

    DECISION plan_2026-05-12_f2d29729/D-007: applied as a `.map` AFTER
    `_cifar_augment` so that both train and val pipelines see the same
    normalized distribution. Replaces the previous numpy-level
    normalization that was applied before augmentation (which broke
    train/val parity — see D-006).
    """
    mean_vals, std_vals = get_cifar_preprocessing()
    mean = tf.constant(mean_vals, dtype=tf.float32, shape=(1, 1, 3))
    std = tf.constant(std_vals, dtype=tf.float32, shape=(1, 1, 3))
    image = (image - mean) / std
    return image, label


def _assert_train_val_distribution_match(
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        *,
        mean_tol: float = 0.5,
        std_ratio_tol: float = 0.5,
) -> None:
    """Pre-fit guard: train and val batches must agree on per-channel mean/std.

    DECISION plan_2026-05-12_f2d29729/D-007: lock-in for the iter-1 cliff bug
    (D-006). Pulls ONE batch from each dataset, computes per-channel mean/std,
    logs both, and raises ``RuntimeError`` if either the mean offset or the std
    ratio exceeds tolerance on any channel. Tolerances are loose by design --
    the iter-1 bug produced ``|Δmean| ~ 2.0`` per channel; this catches
    structural divergence, not numerical drift.

    Args:
        train_ds: training dataset (repeating; one batch consumed here is safe
            since Keras builds its own iterator at fit time).
        val_ds: validation dataset.
        mean_tol: maximum allowed ``|mean_train_c - mean_val_c|`` per channel.
        std_ratio_tol: maximum allowed ``|std_train_c / std_val_c - 1|`` per
            channel.

    Raises:
        RuntimeError: when either tolerance is exceeded on any channel.
    """
    x_train_batch, _ = next(iter(train_ds))
    x_val_batch, _ = next(iter(val_ds))

    mean_train = tf.reduce_mean(x_train_batch, axis=[0, 1, 2])
    mean_val = tf.reduce_mean(x_val_batch, axis=[0, 1, 2])
    std_train = tf.math.reduce_std(x_train_batch, axis=[0, 1, 2])
    std_val = tf.math.reduce_std(x_val_batch, axis=[0, 1, 2])

    mean_train_np = mean_train.numpy()
    mean_val_np = mean_val.numpy()
    std_train_np = std_train.numpy()
    std_val_np = std_val.numpy()

    for c, (mt, mv) in enumerate(zip(mean_train_np, mean_val_np)):
        logger.info(
            f"distribution check: channel {c} train mean={float(mt):+.4f} "
            f"val mean={float(mv):+.4f} |Δ|={abs(float(mt) - float(mv)):.4f}"
        )
    for c, (st, sv) in enumerate(zip(std_train_np, std_val_np)):
        ratio = float(st) / float(sv) if float(sv) > 0.0 else float("inf")
        logger.info(
            f"distribution check: channel {c} train std={float(st):.4f} "
            f"val std={float(sv):.4f} ratio={ratio:.4f}"
        )

    mean_diff = tf.reduce_max(tf.abs(mean_train - mean_val))
    std_ratio_dev = tf.reduce_max(tf.abs(std_train / std_val - 1.0))
    mean_diff_f = float(mean_diff.numpy())
    std_ratio_dev_f = float(std_ratio_dev.numpy())

    if mean_diff_f >= mean_tol or std_ratio_dev_f >= std_ratio_tol:
        raise RuntimeError(
            "Train/val distribution mismatch detected before fit. "
            f"max|Δmean|={mean_diff_f:.4f} (tol={mean_tol}); "
            f"max|std_ratio-1|={std_ratio_dev_f:.4f} (tol={std_ratio_tol}). "
            f"Per-channel train mean={mean_train_np.tolist()} "
            f"val mean={mean_val_np.tolist()} "
            f"train std={std_train_np.tolist()} "
            f"val std={std_val_np.tolist()}. "
            "This is the fingerprint of D-006 (augment-then-normalize bug). "
            "Inspect create_cifar_dataset pipeline ordering."
        )


def create_cifar_dataset(
        config: TrainingConfig,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """Build CIFAR-10 or CIFAR-100 in-memory datasets.

    Returns:
        train_ds, val_ds, steps_per_epoch, val_steps
    """
    ds_name = config.dataset.lower()
    if ds_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    elif ds_name == "cifar100":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    else:
        raise ValueError(f"create_cifar_dataset called with dataset={ds_name!r}")

    y_train = y_train.flatten()
    y_test = y_test.flatten()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # DECISION plan_2026-05-12_f2d29729/D-007: do NOT normalize here at the
    # numpy level. Normalization runs as `_cifar_normalize` AFTER augmentation
    # in the tf.data pipeline so that train and val see identical statistics.
    # See D-006 for the iter-1 bug fingerprint.

    logger.info(f"{ds_name.upper()}: {x_train.shape[0]} train, {x_test.shape[0]} test")

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000)
    if config.augment_data:
        train_ds = train_ds.map(_cifar_augment, num_parallel_calls=config.num_parallel_calls)
    train_ds = train_ds.map(_cifar_normalize, num_parallel_calls=config.num_parallel_calls)
    train_ds = (
        train_ds.repeat()
        .batch(config.batch_size, drop_remainder=True)
        .prefetch(config.prefetch_buffer)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .map(_cifar_normalize, num_parallel_calls=config.num_parallel_calls)
        .batch(config.batch_size)
        .prefetch(config.prefetch_buffer)
    )

    steps_per_epoch = len(x_train) // config.batch_size
    val_steps = max(1, len(x_test) // config.batch_size)
    return train_ds, val_ds, steps_per_epoch, val_steps


# =============================================================================
# IMAGENET DATA PIPELINE
# =============================================================================

# The ImageNet class-subdir tf.data pipeline now lives in
# train.common.make_imagenet_filesystem_dataset. ViT uses the default
# augment_color=False (no colour augmentations); IMAGENET_MEAN/STD
# normalization and the unconditional clip_by_value(0,255) are applied inside
# the helper. See plan_2026-06-02_35651564/D-001.


# =============================================================================
# CALLBACKS
# =============================================================================

def create_callbacks(config: TrainingConfig) -> Tuple[List[keras.callbacks.Callback], str]:
    """Common callbacks (early-stop, ckpt, CSV, analyzer) + ViT-specific metrics viz."""
    callbacks, results_dir = create_common_callbacks(
        model_name=config.experiment_name or config.model_variant,
        results_dir_prefix="vit",
        monitor="val_accuracy",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
    )
    viz_dir = Path(config.output_dir) / config.experiment_name / "training_metrics"
    callbacks.append(EpochMetricsPlotCallback(
        str(viz_dir),
        ["accuracy", "top5_accuracy"],
        every_n=config.monitor_every_n_epochs,
    ))
    return callbacks, results_dir


# =============================================================================
# VISUALIZATION
# =============================================================================

def setup_visualization_manager(experiment_name: str, results_dir: str) -> VisualizationManager:
    viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    config = PlotConfig(
        style=PlotStyle.PUBLICATION,
        color_scheme=ColorScheme(
            primary="#2E86AB", secondary="#A23B72",
            success="#06D6A0", warning="#FFD166",
        ),
        title_fontsize=14, label_fontsize=12,
        save_format="png", dpi=300, fig_size=(12, 8),
    )
    viz_manager = VisualizationManager(
        experiment_name=experiment_name, output_dir=viz_dir, config=config
    )
    viz_manager.register_template("training_curves", TrainingCurvesVisualization)
    viz_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)
    viz_manager.register_template("network_architecture", NetworkArchitectureVisualization)
    viz_manager.register_template("model_comparison_bars", ModelComparisonBarChart)
    viz_manager.register_template("roc_pr_curves", ROCPRCurves)
    return viz_manager


# =============================================================================
# MAIN TRAINING
# =============================================================================

def train_vit(
        config: TrainingConfig, gpu_id: Optional[int] = None
) -> Dict[str, Any]:
    """Orchestrate the ViT classification training pipeline.

    Returns:
        Dict with keys:
          - ``model``: the trained ``keras.Model``.
          - ``best_val_acc``: ``float`` peak ``val_accuracy`` observed.
          - ``early_stop_epoch``: ``int`` number of epochs actually run
            (``len(history.history['loss'])``).
          - ``total_epochs``: ``int`` configured ``config.epochs``.
          - ``history``: raw ``keras.callbacks.History`` object.
    """
    setup_gpu(gpu_id)

    logger.info(f"Experiment: {config.experiment_name}, Variant: {config.model_variant}")
    logger.info(f"Dataset: {config.dataset}, Image size: {config.image_size}, "
                f"Patch size: {config.patch_size}")

    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(config, str(output_dir), "config.json")

    # ---- Datasets ----
    if config.dataset in ("cifar10", "cifar100"):
        train_ds, val_ds, steps_per_epoch, val_steps = create_cifar_dataset(config)
        input_shape = (config.image_size, config.image_size, 3)
        # DECISION plan_2026-05-12_f2d29729/D-007: distribution invariant guard.
        # Catches the D-006 fingerprint (train/val pipeline divergence) BEFORE
        # any compute is spent on fit. ImageNet path skipped — different
        # invariants (file-based, no in-memory pre-normalization).
        _assert_train_val_distribution_match(train_ds, val_ds)
    elif config.dataset == "imagenet":
        train_ds = make_imagenet_filesystem_dataset(
            config.train_data_dir,
            config.image_size,
            config.batch_size,
            is_training=True,
            augment=config.augment_data,
            augment_color=False,
            num_parallel_calls=config.num_parallel_calls,
            cache_val=config.cache_dataset,
            prefetch_buffer=config.prefetch_buffer,
        )
        val_ds = make_imagenet_filesystem_dataset(
            config.val_data_dir,
            config.image_size,
            config.batch_size,
            is_training=False,
            augment=config.augment_data,
            augment_color=False,
            num_parallel_calls=config.num_parallel_calls,
            cache_val=config.cache_dataset,
            prefetch_buffer=config.prefetch_buffer,
        )
        train_dir = Path(config.train_data_dir)
        num_train_images = sum(
            len(list((train_dir / cd).glob("*.JPEG")))
            for cd in train_dir.iterdir() if cd.is_dir()
        )
        steps_per_epoch = num_train_images // config.batch_size
        val_dir = Path(config.val_data_dir)
        num_val_images = sum(
            len(list((val_dir / cd).glob("*.JPEG")))
            for cd in val_dir.iterdir() if cd.is_dir()
        )
        val_steps = max(1, num_val_images // config.batch_size)
        input_shape = (config.image_size, config.image_size, 3)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    logger.info(f"Steps per epoch: {steps_per_epoch}, Val steps: {val_steps}")

    # ---- Model ----
    # LESSONS L72 (Double Weight Decay guard): when optimizer is AdamW, pass
    # weight_decay to optimizer_builder ONLY and leave kernel_regularizer=None.
    # When optimizer is SGD, use kernel_regularizer=L2(weight_decay) instead.
    use_adamw = config.optimizer_type.lower() == "adamw"
    kernel_reg = (
        None
        if use_adamw
        else (keras.regularizers.L2(config.weight_decay) if config.weight_decay > 0 else None)
    )

    model = create_vit(
        variant=config.model_variant,
        num_classes=config.num_classes,
        input_shape=input_shape,
        patch_size=config.patch_size,
        pretrained=config.pretrained,
        dropout_rate=config.dropout_rate,
        attention_dropout_rate=config.attention_dropout_rate,
        kernel_regularizer=kernel_reg,
    )
    # Probe build so summary + count_params work.
    model.build((None,) + input_shape)
    model.summary()

    # ---- Optimization ----
    lr_schedule = learning_rate_schedule_builder({
        "type": config.lr_schedule_type,
        "learning_rate": config.learning_rate,
        "decay_steps": steps_per_epoch * config.epochs,
        "warmup_steps": steps_per_epoch * config.warmup_epochs,
        "alpha": 0.01,
    })

    opt_config: Dict[str, Any] = {
        "type": config.optimizer_type,
        "gradient_clipping_by_norm": config.gradient_clipping,
    }
    if use_adamw:
        opt_config["weight_decay"] = config.weight_decay
    elif config.optimizer_type.lower() == "sgd":
        opt_config["momentum"] = config.momentum

    optimizer = optimizer_builder(opt_config, lr_schedule)

    # ---- Loss + metrics ----
    # Keras' SparseCategoricalCrossentropy doesn't support label_smoothing
    # (that's a CategoricalCrossentropy feature). Keep config.label_smoothing
    # for potential future one-hot path -- currently informational only.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    if config.label_smoothing > 0.0:
        logger.warning(
            f"label_smoothing={config.label_smoothing} requested but "
            "SparseCategoricalCrossentropy doesn't support it; ignoring."
        )

    metrics = [
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
    ]

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    # ---- Train ----
    callbacks, _ = create_callbacks(config)

    start_time = time.time()
    history = model.fit(
        train_ds,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed_h = (time.time() - start_time) / 3600.0
    logger.info(f"Training completed in {elapsed_h:.2f} hours")

    # ---- Save history ----
    try:
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history_dict, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save training history: {e}")

    # Convergence accounting (DECISION plan_2026-05-12_f2d29729/D-007)
    val_acc_curve = history.history.get("val_accuracy", [0.0]) or [0.0]
    best_val_acc = float(max(val_acc_curve))
    early_stop_epoch = int(len(history.history.get("loss", [])))

    gc.collect()
    return {
        "model": model,
        "best_val_acc": best_val_acc,
        "early_stop_epoch": early_stop_epoch,
        "total_epochs": int(config.epochs),
        "history": history,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ViT on CIFAR-10 / CIFAR-100 / ImageNet (resnet Pattern-4 style)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "imagenet"])
    parser.add_argument("--train-data-dir", type=str, default=None,
                        help="Required only when --dataset imagenet")
    parser.add_argument("--val-data-dir", type=str, default=None,
                        help="Required only when --dataset imagenet")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Auto: 32 for CIFAR, 224 for ImageNet")

    # Model
    parser.add_argument("--variant", type=str, default="vit_pico",
                        choices=["vit_pico", "vit_tiny", "vit_small",
                                 "vit_base", "vit_large", "vit_huge"])
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Auto: 10 for cifar10, 100 for cifar100, 1000 for imagenet")
    parser.add_argument("--patch-size", type=int, default=None,
                        help="Auto: 4 for CIFAR, 16 for ImageNet")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to local .keras weights file (boolean True is not supported)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention-dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "sgd"])
    parser.add_argument("--lr-schedule", type=str, default="cosine_decay",
                        choices=["cosine_decay", "exponential_decay", "constant"])
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.05)

    # Augmentation
    parser.add_argument("--no-augmentation", dest="augment_data", action="store_false")
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    # Output
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--monitor-every", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    parser.add_argument(
        "--success-threshold", type=float, default=None,
        help="Override the auto-derived val_accuracy convergence threshold "
             "for the SUCCESS log guard. Default: auto = "
             "min(max(2/num_classes, 0.05), 0.95).",
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_arguments()

    # Auto-derive image_size / num_classes / patch_size from dataset.
    ds = args.dataset.lower()
    image_size = args.image_size if args.image_size is not None else (
        32 if ds in ("cifar10", "cifar100") else 224
    )
    num_classes = args.num_classes if args.num_classes is not None else (
        10 if ds == "cifar10" else (100 if ds == "cifar100" else 1000)
    )
    patch_size = args.patch_size if args.patch_size is not None else (
        4 if ds in ("cifar10", "cifar100") else 16
    )

    pretrained_arg: Union[bool, str] = args.pretrained if args.pretrained else False

    config = TrainingConfig(
        dataset=ds,
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        model_variant=args.variant,
        num_classes=num_classes,
        pretrained=pretrained_arg,
        patch_size=patch_size,
        dropout_rate=args.dropout,
        attention_dropout_rate=args.attention_dropout,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer,
        lr_schedule_type=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        augment_data=args.augment_data,
        label_smoothing=args.label_smoothing,
        monitor_every_n_epochs=args.monitor_every,
        early_stopping_patience=args.early_stopping_patience,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        success_threshold=args.success_threshold,
    )

    logger.info(
        f"Config: variant={config.model_variant}, dataset={config.dataset}, "
        f"{config.epochs} epochs, batch={config.batch_size}, lr={config.learning_rate}, "
        f"opt={config.optimizer_type}, wd={config.weight_decay}"
    )

    try:
        result = train_vit(config, gpu_id=args.gpu)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # DECISION plan_2026-05-12_f2d29729/D-007: guard the SUCCESS log on
    # actual convergence + non-trivially-early stop. Iter-1 emitted SUCCESS
    # at best_val_acc=0.23 (random-class chance ~0.10) which masked the
    # data-pipeline bug. Threshold derives from `config.success_threshold`
    # if set, else clamped auto = min(max(2/num_classes, 0.05), 0.95).
    if config.success_threshold is not None:
        threshold = float(config.success_threshold)
    else:
        threshold = min(max(2.0 / config.num_classes, 0.05), 0.95)
    converged = result["best_val_acc"] >= threshold
    stopped_early = result["early_stop_epoch"] < 0.5 * result["total_epochs"]
    if converged and not stopped_early:
        logger.info(
            f"=== TRAINING COMPLETED SUCCESSFULLY "
            f"(best_val_acc={result['best_val_acc']:.4f} >= {threshold:.4f}) ==="
        )
    elif converged and stopped_early:
        logger.warning(
            f"Training converged (best_val_acc={result['best_val_acc']:.4f} >= "
            f"{threshold:.4f}) but early-stopped at epoch "
            f"{result['early_stop_epoch']}/{result['total_epochs']} (<50%). "
            "Inspect curves."
        )
    else:
        logger.error(
            f"=== TRAINING DID NOT CONVERGE "
            f"(best_val_acc={result['best_val_acc']:.4f} < {threshold:.4f}) ==="
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
