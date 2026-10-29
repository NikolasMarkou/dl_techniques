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

Note: ``--output-dir`` (default ``"results"``) is resolved relative to the
current working directory, not to the repo root. Invoke this module from the
repo root (as in the example above, ``python -m train.vit.train_vit ...``)
so runs land in the repo-root ``results/`` directory per the project
convention -- never ``src/results/``. This matches every other Pattern-1
trainer under ``src/train/`` (see plans/plan-2026-09-17T032714-403de954/decisions.md D-002).
"""

import sys
import gc
import time
import keras
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any, Union

from train.common import setup_gpu, set_seeds, create_callbacks as create_common_callbacks, EpochMetricsPlotCallback
from train.common.classification_data import add_selection_arguments, load_classification_data
from train.common.image_augment import build_augment
from train.common.conformal import DEFAULT_ALPHA, add_conformal_arguments, run_conformal
from train.common.imagenet import require_imagenet_root
from train.common.run_io import default_experiment_name, prepare_run_dir, save_training_history_json
from dl_techniques.utils.logger import logger
from dl_techniques.datasets.vision.imagenet2012 import IMAGENET2012_DEFAULT_ROOT
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)
from dl_techniques.models.vision.vit import create_vit


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
    imagenet_root: str = IMAGENET2012_DEFAULT_ROOT  # only used for imagenet
    devkit_dir: Optional[str] = None  # default: <imagenet_root>/devkit (val labels)
    image_size: int = 32  # 32 for CIFAR, 224 for ImageNet
    batch_size: int = 128
    # Selection is a held-out slice of train (validation_split: CIFAR; holdout_per_class:
    # ImageNet); the test split never selects. None means the loader's default.
    validation_split: Optional[float] = None
    holdout_per_class: Optional[int] = None
    max_samples: Optional[int] = None

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
    # DECISION plan-2026-09-17T032714-403de954/D-003: default is 0.0, not the
    # previous 0.1. Keras' SparseCategoricalCrossentropy (used below) has no
    # label_smoothing support, so a non-zero value can never be honored on
    # this trainer's sparse-label pipeline -- see __post_init__'s hard raise.
    # Do not silently re-introduce a non-zero default; see decisions.md D-003.
    label_smoothing: float = 0.0

    # Monitoring
    monitor_every_n_epochs: int = 5
    early_stopping_patience: int = 15

    # Conformal prediction sets fitted on the official val split after training (None = off);
    # ImageNet only
    conformal_alpha: Optional[float] = DEFAULT_ALPHA

    # Output
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    # Convergence guard (DECISION plan_2026-05-12_f2d29729/D-007)
    # If None, threshold auto-derives to min(max(2/num_classes, 0.05), 0.95).
    success_threshold: Optional[float] = None

    seed: int = 42  # train.common.set_seeds: Python, NumPy and Keras

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            self.experiment_name = default_experiment_name("vit", self.dataset, self.model_variant)

        if self.image_size <= 0:
            raise ValueError("Invalid image_size: must be positive")
        if self.label_smoothing != 0.0:
            raise ValueError(
                f"label_smoothing={self.label_smoothing} is not supported: "
                "this trainer uses SparseCategoricalCrossentropy on sparse "
                "integer labels, and Keras' SparseCategoricalCrossentropy "
                "has no label_smoothing parameter (only CategoricalCrossentropy "
                "does, which requires one-hot labels). Pass --label-smoothing "
                "0.0 (the default), or omit the flag."
            )
        if self.num_classes <= 0:
            raise ValueError("Invalid num_classes: must be positive")
        if self.dataset == "imagenet":
            require_imagenet_root(self.imagenet_root)


# =============================================================================
# CIFAR DATA PIPELINE
# =============================================================================

def _augment_hook(config: "TrainingConfig"):
    """The train augmentation hook of ``config``'s dataset, or ``None``.

    CIFAR gets the shared ``cifar_light`` preset (flip, reflect pad-4 crop, brightness 0.1,
    contrast 0.9-1.1) on the [0, 1] image BEFORE standardization, with no clipping
    (DECISION plan_2026-05-12_f2d29729/D-007: the old version augmented the normalized
    image and then clipped it to [0, 1], which saturated most pixels and made train and
    validation disagree; do NOT add a ``clip_by_value``). ImageNet keeps the loader's own
    random-resized-crop and flip: the CIFAR recipe was passed to it once and the loader
    refused it.
    """
    if not config.augment_data or config.dataset not in ("cifar10", "cifar100"):
        return None
    return build_augment("cifar_light")


# =============================================================================
# IMAGENET DATA PIPELINE
# =============================================================================

# The ImageNet tf.data pipeline now lives in
# dl_techniques.datasets.vision.imagenet2012.load_imagenet2012. ViT uses the
# default augment_color=False (no colour augmentations); ImageNet mean/std
# normalization is applied inside the loader.


# =============================================================================
# CALLBACKS
# =============================================================================

def create_callbacks(
        config: TrainingConfig, run_dir: Union[str, Path]
) -> Tuple[List[keras.callbacks.Callback], str]:
    """Common callbacks (early-stop, ckpt, CSV, analyzer) + ViT-specific metrics viz.

    Args:
        config: The training config.
        run_dir: The run directory already created by ``prepare_run_dir(config)``
            in ``train_vit()``. Passed through as ``create_common_callbacks``'s
            ``run_dir=`` so the shared helper writes ``best_model.keras`` /
            ``training_log.csv`` into the SAME directory ``config.json`` already
            lives in, instead of synthesizing a second, independently-timestamped
            directory.
    """
    # DECISION plan-2026-09-17T032714-403de954/D-006: do NOT let
    # create_common_callbacks() derive its own directory here (the
    # run_dir=None default path) -- model_name is config.experiment_name,
    # which ALREADY starts with results_dir_prefix="vit"'s value, so the
    # derived name doubles to vit_vit_..._<ts1>_<ts2> and diverges from
    # prepare_run_dir's directory. See decisions.md D-006.
    callbacks, results_dir = create_common_callbacks(
        model_name=config.experiment_name or config.model_variant,
        results_dir_prefix="vit",
        run_dir=str(run_dir),
        monitor="val_accuracy",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
    )
    viz_dir = Path(results_dir) / "training_metrics"
    callbacks.append(EpochMetricsPlotCallback(
        str(viz_dir),
        ["accuracy", "top5_accuracy"],
        every_n=config.monitor_every_n_epochs,
    ))
    return callbacks, results_dir


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
    set_seeds(config.seed)

    logger.info(f"Experiment: {config.experiment_name}, Variant: {config.model_variant}")
    logger.info(f"Dataset: {config.dataset}, Image size: {config.image_size}, "
                f"Patch size: {config.patch_size}")

    output_dir = prepare_run_dir(config)

    # ---- Data: one loader for every dataset (selection on a held-out slice of train) ----
    data = load_classification_data(
        config.dataset, batch_size=config.batch_size, seed=config.seed,
        normalize="standardize",
        augment=_augment_hook(config),
        imagenet_augment=config.augment_data,
        validation_split=config.validation_split,
        holdout_per_class=config.holdout_per_class,
        max_samples=config.max_samples, eval_samples=0, imagenet_root=config.imagenet_root,
        devkit_dir=config.devkit_dir, image_size=config.image_size,
    )
    if data.num_classes != config.num_classes:
        raise ValueError(
            f"num_classes={config.num_classes} but {config.dataset} has {data.num_classes} classes"
        )
    train_ds, val_ds, steps_per_epoch = data.train_ds, data.select_ds, data.steps_per_epoch
    input_shape = (config.image_size, config.image_size, 3)
    logger.info(
        f"Steps per epoch: {steps_per_epoch}; {data.n_train} train, {data.n_select} "
        f"selection ({data.selection_split}), {data.n_test} test"
    )

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
    # config.label_smoothing is guaranteed 0.0 here: __post_init__ hard-raises
    # on any non-zero value (DECISION plan-2026-09-17T032714-403de954/D-003),
    # since SparseCategoricalCrossentropy has no label_smoothing parameter.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    metrics = [
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
    ]

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    # ---- Train ----
    callbacks, _ = create_callbacks(config, output_dir)

    start_time = time.time()
    history = model.fit(
        train_ds,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed_h = (time.time() - start_time) / 3600.0
    logger.info(f"Training completed in {elapsed_h:.2f} hours")

    # ---- Save history ----
    save_training_history_json(history, output_dir)

    # ---- Conformal prediction sets (ImageNet), over ALL of the official val split ----
    # The model holds the best epoch's weights (EarlyStopping restored them), which were
    # selected on the held-out slice of train, not on this split.
    if config.conformal_alpha is not None and config.dataset == "imagenet":
        run_conformal(
            model, data.test_ds, alpha=config.conformal_alpha, seed=config.seed,
            out_dir=output_dir, input_shape=input_shape,
            model_name=f"vit_{config.model_variant}",
            weights=f"best val_accuracy epoch, selected on {data.selection_split}",
            selection_overlaps_calibration=False,
        )

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
    add_selection_arguments(parser, figures=False)
    add_conformal_arguments(parser, note="Used only when --dataset imagenet: ")
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
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0,
        help="Must be 0.0 (default): SparseCategoricalCrossentropy has no "
             "label_smoothing support; a non-zero value raises at config-"
             "validation time rather than being silently ignored.",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Resolved relative to the current working directory, not the "
             "repo root -- invoke this module from the repo root so runs "
             "land in the repo-root results/ directory.",
    )
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--monitor-every", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    parser.add_argument("--seed", type=int, default=42, help="Seed for Python, NumPy and Keras (train.common.set_seeds)")
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
        imagenet_root=args.imagenet_root,
        devkit_dir=args.devkit_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        holdout_per_class=args.holdout_per_class,
        max_samples=args.max_samples,
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
        conformal_alpha=None if args.no_conformal else args.conformal_alpha,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        success_threshold=args.success_threshold,
        seed=args.seed,
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
