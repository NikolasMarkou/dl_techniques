"""
RBFProtoNet CIFAR-100 Classification Training (Pattern-1 style)

Training pipeline for ``RBFProtoNet`` (CNN backbone + RBF prototype-
classification head, ``output_mode='normalized'``) on CIFAR-100. Modeled on
``train.vit.train_vit`` (the repo's Pattern-1 vision-classification
exemplar, see ``src/train/CLAUDE.md``), with the deviations required by the
RBF head documented inline where they occur (see decisions.md D-003 for this
plan).

This is a single-purpose, CIFAR-100-only trainer (no ``--dataset`` choice,
no ViT-style ``--patch-size``/``--image-size 224`` knobs) -- CIFAR-100 is the
only dataset ``RBFProtoNet`` targets at this stage.

Usage:
    MPLBACKEND=Agg .venv/bin/python -m train.rbf_protonet.train_rbf_protonet \\
        --epochs 20 --batch-size 128 --learning-rate 3e-4 \\
        --output-dir results/rbf_protonet_cifar100_smoke --gpu 1
"""

import os
import sys
import gc
import time
import keras
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any

from train.common import setup_gpu, load_dataset, create_callbacks as create_common_callbacks
from train.common.run_io import default_experiment_name, prepare_run_dir, save_training_history_json
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)
from dl_techniques.models.vision.rbf_protonet import create_rbf_protonet


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for RBFProtoNet CIFAR-100 classification training.

    CIFAR-100-only -- no ``dataset`` field, unlike the multi-dataset
    ``train_vit.TrainingConfig``. Image size is fixed at 32 (CIFAR native
    resolution), not exposed as a CLI override.
    """

    # Data
    batch_size: int = 128

    # Model
    num_classes: int = 100
    feature_dim: int = 128  # per decisions.md D-006/D-007 (Step 1 smoke test)
    repulsion_strength: float = 0.1
    min_distance: float = 1.0

    # Training
    epochs: int = 20
    learning_rate: float = 3e-4
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "cosine_decay"
    warmup_epochs: int = 2
    weight_decay: float = 0.05
    gradient_clipping: float = 1.0
    momentum: float = 0.9  # used only when optimizer_type='sgd'

    # Augmentation
    augment_data: bool = True

    # Monitoring
    monitor_every_n_epochs: int = 5
    early_stopping_patience: int = 10
    validation_steps: Optional[int] = None

    # Data pipeline
    prefetch_buffer: int = tf.data.AUTOTUNE
    num_parallel_calls: int = tf.data.AUTOTUNE

    # Output
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            self.experiment_name = default_experiment_name("rbf_protonet", "cifar100")

        if self.num_classes <= 0:
            raise ValueError("Invalid num_classes: must be positive")
        if self.batch_size <= 0:
            raise ValueError("Invalid batch_size: must be positive")
        if self.epochs <= 0:
            raise ValueError("Invalid epochs: must be positive")


# =============================================================================
# CIFAR-100 DATA PIPELINE
# =============================================================================
# The RBF prototype head expects inputs on the SAME [0,1] scale RBFProtoNet
# itself was trained/tested against (no normalization layer is built into
# the model, per plan.md's invariant #1 and findings/cifar_training_pipeline.md's
# explicit "do not reuse CIFAR10_MEAN/STD for CIFAR-100" constraint). This
# pipeline therefore stays on [0,1] end to end -- no per-channel mean/std
# normalization step, unlike train_vit.py's CIFAR pipeline.

def _cifar_augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Flip + reflect-pad-crop augmentation on [0,1]-valued float32 images."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(
        tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode="REFLECT"),
        size=tf.shape(image),
    )
    return image, label


def _assert_train_val_distribution_match(
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        *,
        mean_tol: float = 0.5,
        std_ratio_tol: float = 0.5,
) -> None:
    """Pre-fit guard: train and val batches must agree on per-channel mean/std.

    Modeled on ``train_vit.py``'s ``_assert_train_val_distribution_match``
    (DECISION plan_2026-05-12_f2d29729/D-007) -- catches a train/val tf.data
    pipeline-ordering divergence (e.g. augmentation applied to one split but
    not the other) before any compute is spent on ``fit``.
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
            "Inspect create_cifar100_dataset pipeline ordering."
        )


def create_cifar100_dataset(
        config: TrainingConfig,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """Build the CIFAR-100 train/val ``tf.data`` pipelines.

    Uses ``train.common.load_dataset('cifar100', ...)`` for the raw,
    already-[0,1]-normalized numpy arrays (per findings/cifar_training_pipeline.md),
    then applies flip/crop-with-reflect-pad augmentation to the training
    split only.

    Returns:
        train_ds, val_ds, steps_per_epoch, val_steps
    """
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = load_dataset(
        "cifar100", batch_size=config.batch_size,
    )
    if num_classes != config.num_classes:
        raise ValueError(
            f"load_dataset('cifar100', ...) returned num_classes={num_classes}, "
            f"expected {config.num_classes}"
        )

    logger.info(f"CIFAR100: {x_train.shape[0]} train, {x_test.shape[0]} test")

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000)
    if config.augment_data:
        train_ds = train_ds.map(_cifar_augment, num_parallel_calls=config.num_parallel_calls)
    train_ds = (
        train_ds.repeat()
        .batch(config.batch_size, drop_remainder=True)
        .prefetch(config.prefetch_buffer)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(config.batch_size)
        .prefetch(config.prefetch_buffer)
    )

    steps_per_epoch = len(x_train) // config.batch_size
    val_steps = max(1, len(x_test) // config.batch_size)
    return train_ds, val_ds, steps_per_epoch, val_steps


# =============================================================================
# CALLBACKS
# =============================================================================

def create_callbacks(config: TrainingConfig) -> Tuple[List[keras.callbacks.Callback], str]:
    """Standard early-stop/checkpoint/CSV bundle (no RBF-specific callback exists yet,
    see findings/cifar_training_pipeline.md section 3)."""
    callbacks, results_dir = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="rbf_protonet",
        monitor="val_accuracy",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
    )
    return callbacks, results_dir


# =============================================================================
# MAIN TRAINING
# =============================================================================

def train_rbf_protonet(
        config: TrainingConfig, gpu_id: Optional[int] = None
) -> Dict[str, Any]:
    """Orchestrate the RBFProtoNet CIFAR-100 classification training pipeline.

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

    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Dataset: cifar100, image size: 32x32x3, feature_dim: {config.feature_dim}")

    output_dir = prepare_run_dir(config)

    # ---- Dataset ----
    input_shape = (32, 32, 3)
    train_ds, val_ds, steps_per_epoch, val_steps = create_cifar100_dataset(config)
    _assert_train_val_distribution_match(train_ds, val_ds)

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

    model = create_rbf_protonet(
        num_classes=config.num_classes,
        input_shape=input_shape,
        feature_dim=config.feature_dim,
        repulsion_strength=config.repulsion_strength,
        min_distance=config.min_distance,
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
    # DECISION plan-2026-09-16-7dfede94/D-003: from_logits=False, deliberately
    # deviating from train_vit.py's from_logits=True. RBFProtoNet's RBF head
    # is constructed with output_mode='normalized' (softmax-over-negative-
    # distance), which IS already a probability vector, not logits. Passing
    # from_logits=True here would silently re-apply softmax to an
    # already-softmax output, corrupting the gradient. Do NOT "fix" this back
    # to from_logits=True without re-reading D-003.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    metrics = [
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
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
    save_training_history_json(history, output_dir)

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
        description="Train RBFProtoNet on CIFAR-100 (Pattern-1 style)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--feature-dim", type=int, default=128,
                        help="Pooled backbone feature dimensionality feeding the RBF head "
                             "(locked to 128 per decisions.md D-006)")
    parser.add_argument("--repulsion-strength", type=float, default=0.1)
    parser.add_argument("--min-distance", type=float, default=1.0)

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "sgd"])
    parser.add_argument("--lr-schedule", type=str, default="cosine_decay",
                        choices=["cosine_decay", "exponential_decay", "constant"])
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.05)

    # Augmentation
    parser.add_argument("--no-augmentation", dest="augment_data", action="store_false")

    # Output
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--monitor-every", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_arguments()

    config = TrainingConfig(
        batch_size=args.batch_size,
        feature_dim=args.feature_dim,
        repulsion_strength=args.repulsion_strength,
        min_distance=args.min_distance,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer,
        lr_schedule_type=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        augment_data=args.augment_data,
        monitor_every_n_epochs=args.monitor_every,
        early_stopping_patience=args.early_stopping_patience,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )

    logger.info(
        f"Config: {config.epochs} epochs, batch={config.batch_size}, "
        f"lr={config.learning_rate}, opt={config.optimizer_type}, wd={config.weight_decay}"
    )

    try:
        result = train_rbf_protonet(config, gpu_id=args.gpu)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

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
