"""
RBFProtoNet CIFAR-100 Classification Training (Pattern-1 style)

Training pipeline for RBF prototype-classification heads
(``output_mode='normalized'``) on CIFAR-100, over two selectable backbone
variants (``--model-variant``):

- ``rbf_protonet`` (default): ``RBFProtoNet``'s CNN backbone. Modeled on
  ``train.vit.train_vit`` (the repo's Pattern-1 vision-classification
  exemplar, see ``src/train/AGENTS.md``), with the deviations required by the
  RBF head documented inline where they occur (see decisions.md D-003 for
  this plan).
- ``clifford_rbf_protonet``: ``CliffordRBFProtoNet``'s isotropic
  Clifford-algebra backbone (nano-equivalent defaults, see decisions.md
  D-003/D-004 for the plan that added it), tunable via ``--clifford-depth``
  and ``--clifford-channels``.

This is a single-purpose, CIFAR-100-only trainer (no ``--dataset`` choice,
no ViT-style ``--patch-size``/``--image-size 224`` knobs) -- CIFAR-100 is the
only dataset either variant targets at this stage.

Usage:
    MPLBACKEND=Agg .venv/bin/python -m train.rbf_protonet.train_rbf_protonet \\
        --epochs 20 --batch-size 128 --learning-rate 3e-4 \\
        --output-dir results/rbf_protonet_cifar100_smoke --gpu 1

    MPLBACKEND=Agg .venv/bin/python -m train.rbf_protonet.train_rbf_protonet \\
        --model-variant clifford_rbf_protonet --clifford-depth 12 \\
        --clifford-channels 128 --gpu 1
"""

import sys
import gc
import time
import keras
import argparse
import tensorflow as tf
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any

from train.common import setup_gpu, set_seeds, create_callbacks as create_common_callbacks
from train.common.classification_data import (
    ClassificationData,
    add_selection_arguments,
    flip_pad_crop,
    load_classification_data,
)
from train.common.run_io import default_experiment_name, prepare_run_dir, save_training_history_json
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)
from dl_techniques.models.vision.rbf_protonet import (
    create_rbf_protonet,
    create_clifford_rbf_protonet,
)


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
    model_variant: str = "rbf_protonet"
    num_classes: int = 100
    feature_dim: int = 128  # per decisions.md D-006/D-007 (Step 1 smoke test)
    repulsion_strength: float = 0.1
    min_distance: float = 1.0
    clifford_depth: int = 12
    clifford_channels: int = 128

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

    # Data: selection is a held-out slice of train; the test split never selects.
    validation_split: Optional[float] = None
    max_samples: Optional[int] = None

    # Monitoring
    monitor_every_n_epochs: int = 5
    early_stopping_patience: int = 10

    # Data pipeline
    prefetch_buffer: int = tf.data.AUTOTUNE
    num_parallel_calls: int = tf.data.AUTOTUNE

    # Output
    output_dir: str = "results"
    experiment_name: Optional[str] = None
    seed: int = 42  # train.common.set_seeds: Python, NumPy and Keras

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
# explicit "do not reuse CIFAR10_MEAN/STD for CIFAR-100" constraint). The shared
# loader is therefore asked for ``normalize="unit"``: [0,1] end to end, with the
# flip + reflect-pad-crop augmentation applied to the training split only.

def create_cifar100_data(config: TrainingConfig) -> ClassificationData:
    """Load CIFAR-100 through the shared loader: selection on a held-out slice of train.

    Raises:
        ValueError: ``config.num_classes`` is not CIFAR-100's 100.
    """
    data = load_classification_data(
        "cifar100", batch_size=config.batch_size, seed=config.seed, normalize="unit",
        validation_split=config.validation_split, max_samples=config.max_samples,
        augment=flip_pad_crop(mode="REFLECT") if config.augment_data else None,
    )
    if data.num_classes != config.num_classes:
        raise ValueError(
            f"CIFAR-100 has num_classes={data.num_classes}, expected {config.num_classes}"
        )
    return data


# =============================================================================
# CALLBACKS
# =============================================================================

def create_callbacks(
        config: TrainingConfig, run_dir: str
) -> Tuple[List[keras.callbacks.Callback], str]:
    """Standard early-stop/checkpoint/CSV bundle (no RBF-specific callback exists yet,
    see findings/cifar_training_pipeline.md section 3)."""
    # DECISION plan-2026-09-16T052349-7dfede94/D-008: run_dir=run_dir is REQUIRED here.
    # Do not drop it: without it, create_common_callbacks derives its OWN
    # "{prefix}_{model_name}_{timestamp}" directory under output_root="results",
    # silently ignoring the run dir prepare_run_dir(config) already created for
    # config.json/training_history.json -- producing two disjoint result trees
    # for one run (measured in Step 7's first attempt). See decisions.md D-008.
    callbacks, results_dir = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="rbf_protonet",
        run_dir=run_dir,
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
    set_seeds(config.seed)

    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Dataset: cifar100, image size: 32x32x3, feature_dim: {config.feature_dim}")

    output_dir = prepare_run_dir(config)

    # ---- Dataset ----
    input_shape = (32, 32, 3)
    data = create_cifar100_data(config)
    steps_per_epoch = data.steps_per_epoch

    logger.info(
        f"CIFAR100: {data.n_train} train, {data.n_select} selection, {data.n_test} test; "
        f"steps per epoch: {steps_per_epoch}"
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

    if config.model_variant == "rbf_protonet":
        model = create_rbf_protonet(
            num_classes=config.num_classes,
            feature_dim=config.feature_dim,
            repulsion_strength=config.repulsion_strength,
            min_distance=config.min_distance,
            kernel_regularizer=kernel_reg,
        )
    elif config.model_variant == "clifford_rbf_protonet":
        model = create_clifford_rbf_protonet(
            num_classes=config.num_classes,
            feature_dim=config.feature_dim,
            repulsion_strength=config.repulsion_strength,
            min_distance=config.min_distance,
            depth=config.clifford_depth,
            channels=config.clifford_channels,
            kernel_regularizer=kernel_reg,
        )
    else:
        raise ValueError(f"Unknown model_variant: {config.model_variant!r}")
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
    # DECISION plan-2026-09-16T052349-7dfede94/D-003: from_logits=False, deliberately
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
    callbacks, _ = create_callbacks(config, str(output_dir))

    start_time = time.time()
    history = model.fit(
        data.train_ds,
        epochs=config.epochs,
        validation_data=data.select_ds,
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
    parser.add_argument("--model-variant", type=str, default="rbf_protonet",
                        choices=["rbf_protonet", "clifford_rbf_protonet"],
                        help="Backbone variant: 'rbf_protonet' (CNN backbone, default) or "
                             "'clifford_rbf_protonet' (isotropic Clifford-algebra backbone)")
    parser.add_argument("--feature-dim", type=int, default=128,
                        help="Pooled backbone feature dimensionality feeding the RBF head "
                             "(locked to 128 per decisions.md D-006)")
    parser.add_argument("--repulsion-strength", type=float, default=0.1)
    parser.add_argument("--min-distance", type=float, default=1.0)
    parser.add_argument("--clifford-depth", type=int, default=12,
                        help="Number of CliffordNetBlocks (only used when "
                             "--model-variant=clifford_rbf_protonet)")
    parser.add_argument("--clifford-channels", type=int, default=128,
                        help="Channel width of the Clifford backbone (only used when "
                             "--model-variant=clifford_rbf_protonet)")

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
    add_selection_arguments(parser, imagenet=False)

    # Output
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--monitor-every", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    parser.add_argument("--seed", type=int, default=42, help="Seed for Python, NumPy and Keras (train.common.set_seeds)")

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_arguments()

    config = TrainingConfig(
        batch_size=args.batch_size,
        model_variant=args.model_variant,
        feature_dim=args.feature_dim,
        repulsion_strength=args.repulsion_strength,
        min_distance=args.min_distance,
        clifford_depth=args.clifford_depth,
        clifford_channels=args.clifford_channels,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer,
        lr_schedule_type=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        augment_data=args.augment_data,
        validation_split=args.validation_split,
        max_samples=args.max_samples,
        monitor_every_n_epochs=args.monitor_every,
        early_stopping_patience=args.early_stopping_patience,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        seed=args.seed,
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
