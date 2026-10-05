"""DaSiamRPN tracking training (Pattern 4: detection-style dense prediction).

Trains :class:`DaSiamRPN` on centered exemplar/search pairs with the
per-anchor classification + box-delta objectives (:class:`DaSiamRPNClsLoss` /
:class:`DaSiamRPNRegLoss`, targets from ``datasets/vision/tracking.py``
matched against the reference anchor grid). Two pair sources behind
``--data-source``:

- ``synthetic`` (default): seeded noise backgrounds with a rectangle target
  plus unannotated distractors -- offline, deterministic, what the smoke
  tests use;
- ``coco``: real pairs from COCO 2017 via TFDS, one uniformly sampled box
  per image.

The long-term redetection schedule and the online distractor-template update
are tracking-loop policies, not network training, and are not implemented
here.

Usage:
    MPLBACKEND=Agg .venv/bin/python -m train.dasiamrpn.train_dasiamrpn \\
        --variant otb --epochs 5 --batch-size 2 --data-source synthetic --gpu 1

    MPLBACKEND=Agg .venv/bin/python -m train.dasiamrpn.train_dasiamrpn \\
        --variant big --data-source coco --steps-per-epoch 500 --epochs 50 --gpu 1
"""

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf

from train.common import setup_gpu, create_callbacks as create_common_callbacks
from train.common.run_io import (
    default_experiment_name,
    prepare_run_dir,
    save_training_history_json,
)
from train.common import set_seeds
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)
from dl_techniques.models.vision.dasiamrpn import (
    DaSiamRPN,
    dasiamrpn_score_size,
    generate_dasiamrpn_anchors,
)
from dl_techniques.losses.siamese_tracking_loss import (
    DaSiamRPNClsLoss,
    DaSiamRPNRegLoss,
)
from dl_techniques.datasets.vision.tracking import (
    synthetic_tracking_generator,
    coco_tracking_generator,
    build_rpn_example,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DaSiamRPNTrainingConfig:
    """Configuration for DaSiamRPN one-shot detection training.

    ``steps_per_epoch`` / ``val_steps`` derive from the synthetic counts and
    need explicit values only for COCO (unbounded cardinality).
    """

    # Data
    data_source: str = "synthetic"
    coco_data_dir: Optional[str] = None
    coco_split: str = "train"
    num_synthetic_train: int = 2048
    num_synthetic_val: int = 256
    steps_per_epoch: Optional[int] = None
    val_steps: Optional[int] = None
    augment: bool = True
    brightness_delta: float = 0.125
    seed: int = 0
    prefetch_buffer: int = tf.data.AUTOTUNE

    # Model / anchors / matching
    variant: str = "otb"
    exemplar_size: int = 127
    search_size: int = 271
    use_batch_norm: bool = True
    pos_iou: float = 0.6
    neg_iou: float = 0.3
    cls_weight: float = 1.0
    reg_weight: float = 1.0
    huber_delta: float = 1.0

    # Training
    batch_size: int = 2
    epochs: int = 5
    learning_rate: float = 1e-3
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "cosine_decay"
    warmup_epochs: int = 1
    weight_decay: float = 5e-4
    gradient_clipping: float = 1.0
    momentum: float = 0.9  # used only when optimizer_type='sgd'

    # Monitoring
    early_stopping_patience: int = 10

    # Output
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            self.experiment_name = default_experiment_name(
                f"dasiamrpn_{self.variant}", self.data_source
            )
        if self.variant not in DaSiamRPN.MODEL_VARIANTS:
            raise ValueError(
                f"variant must be one of {list(DaSiamRPN.MODEL_VARIANTS.keys())}, "
                f"got {self.variant!r}"
            )
        if self.data_source not in ("synthetic", "coco"):
            raise ValueError(f"data_source must be 'synthetic' or 'coco', got {self.data_source!r}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.search_size <= self.exemplar_size:
            raise ValueError("search_size must exceed exemplar_size")
        if not 0.0 <= self.neg_iou < self.pos_iou <= 1.0:
            raise ValueError("need 0 <= neg_iou < pos_iou <= 1")
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if self.brightness_delta < 0:
            raise ValueError("brightness_delta must be non-negative")


# =============================================================================
# DATA PIPELINE
# =============================================================================

def _example_iterator(config: DaSiamRPNTrainingConfig, train: bool):
    """Yield ``((z, x), {"cls":..., "reg":...})`` NumPy examples."""
    split_seed = config.seed + (0 if train else 1_000_000)
    rng = np.random.default_rng(split_seed)
    score_size = dasiamrpn_score_size(config.exemplar_size, config.search_size)
    anchors = generate_dasiamrpn_anchors(score_size)
    anchor_num = anchors.shape[0] // (score_size * score_size)
    if config.data_source == "synthetic":
        count = config.num_synthetic_train if train else config.num_synthetic_val
        base = synthetic_tracking_generator(count, seed=split_seed)
    else:
        split = config.coco_split if train else "validation"
        base = coco_tracking_generator(
            config.coco_data_dir, split=split, seed=split_seed
        )
    do_augment = config.augment and train
    for image, box in base:
        yield build_rpn_example(
            image,
            box,
            anchors,
            anchor_num,
            score_size,
            exemplar_size=config.exemplar_size,
            search_size=config.search_size,
            pos_iou=config.pos_iou,
            neg_iou=config.neg_iou,
            brightness_delta=config.brightness_delta,
            rng=rng,
            augment=do_augment,
        )


def create_pair_dataset(
    config: DaSiamRPNTrainingConfig, train: bool
) -> Tuple[tf.data.Dataset, Optional[int]]:
    """Build the ``tf.data`` pipeline for one split.

    Returns:
        Dataset yielding ``((z, x), {"cls":..., "reg":...})`` batches, plus
        the derived step count (None never happens here: COCO requires
        explicit steps, synthetic derives them).
    """
    score_size = dasiamrpn_score_size(config.exemplar_size, config.search_size)
    anchors = generate_dasiamrpn_anchors(score_size)
    anchor_num = anchors.shape[0] // (score_size * score_size)
    signature = (
        (
            tf.TensorSpec((config.exemplar_size, config.exemplar_size, 3), tf.float32),
            tf.TensorSpec((config.search_size, config.search_size, 3), tf.float32),
        ),
        {
            "cls": tf.TensorSpec((score_size, score_size, anchor_num, 2), tf.float32),
            "reg": tf.TensorSpec((score_size, score_size, anchor_num, 5), tf.float32),
        },
    )
    ds = tf.data.Dataset.from_generator(
        lambda: _example_iterator(config, train), output_signature=signature
    )
    if train:
        ds = ds.shuffle(min(1000, max(64, config.batch_size * 16)), seed=config.seed)
        ds = ds.repeat().batch(config.batch_size, drop_remainder=True)
        if config.data_source == "synthetic" and config.steps_per_epoch is None:
            steps: Optional[int] = max(1, config.num_synthetic_train // config.batch_size)
        else:
            steps = config.steps_per_epoch
            if steps is None:
                raise ValueError(
                    "steps_per_epoch is required for data_source='coco' "
                    "(unbounded cardinality)"
                )
    else:
        ds = ds.batch(config.batch_size)
        if config.data_source == "synthetic" and config.val_steps is None:
            steps = max(1, config.num_synthetic_val // config.batch_size)
        else:
            steps = config.val_steps
            if steps is None:
                raise ValueError("val_steps is required for data_source='coco'")
    return ds.prefetch(config.prefetch_buffer), steps


# =============================================================================
# CALLBACKS
# =============================================================================

def create_callbacks(
    config: DaSiamRPNTrainingConfig, run_dir: str
) -> Tuple[List[keras.callbacks.Callback], str]:
    """Early-stop / checkpoint / CSV bundle monitoring ``val_loss``."""
    # run_dir=run_dir is REQUIRED: without it the common factory derives its
    # own directory and the run splits across two result trees.
    callbacks, results_dir = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="dasiamrpn",
        run_dir=run_dir,
        monitor="val_loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
    )
    return callbacks, results_dir


# =============================================================================
# MAIN TRAINING
# =============================================================================

def train_dasiamrpn(
    config: DaSiamRPNTrainingConfig, gpu_id: Optional[int] = None
) -> Dict[str, Any]:
    """Orchestrate the DaSiamRPN training pipeline.

    Returns:
        Dict with keys ``model``, ``best_val_loss``, ``epochs_run``,
        ``total_epochs`` and ``history``.
    """
    setup_gpu(gpu_id)
    set_seeds(config.seed)

    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(
        f"Data: {config.data_source}, variant={config.variant}, "
        f"exemplar={config.exemplar_size}, search={config.search_size}"
    )

    output_dir = prepare_run_dir(config)

    train_ds, steps_per_epoch = create_pair_dataset(config, train=True)
    val_ds, val_steps = create_pair_dataset(config, train=False)
    logger.info(f"Steps per epoch: {steps_per_epoch}, Val steps: {val_steps}")

    model = DaSiamRPN.from_variant(
        config.variant,
        exemplar_size=config.exemplar_size,
        search_size=config.search_size,
        use_batch_norm=config.use_batch_norm,
    )
    model.build(
        [
            (None, config.exemplar_size, config.exemplar_size, 3),
            (None, config.search_size, config.search_size, 3),
        ]
    )
    model.summary()

    lr_schedule = learning_rate_schedule_builder({
        "type": config.lr_schedule_type,
        "learning_rate": config.learning_rate,
        "decay_steps": steps_per_epoch * config.epochs,
        "warmup_steps": steps_per_epoch * config.warmup_epochs,
        "alpha": 0.01,
    })

    use_adamw = config.optimizer_type.lower() == "adamw"
    opt_config: Dict[str, Any] = {
        "type": config.optimizer_type,
        "gradient_clipping_by_norm": config.gradient_clipping,
    }
    if use_adamw:
        opt_config["weight_decay"] = config.weight_decay
    elif config.optimizer_type.lower() == "sgd":
        opt_config["momentum"] = config.momentum
    optimizer = optimizer_builder(opt_config, lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss={
            "cls": DaSiamRPNClsLoss(),
            "reg": DaSiamRPNRegLoss(huber_delta=config.huber_delta),
        },
        loss_weights={"cls": config.cls_weight, "reg": config.reg_weight},
    )

    callbacks, _ = create_callbacks(config, str(output_dir))

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
    logger.info(f"Training completed in {(time.time() - start_time) / 3600.0:.2f} hours")

    save_training_history_json(history, output_dir)

    val_curve = history.history.get("val_loss", [float("inf")]) or [float("inf")]
    gc.collect()
    return {
        "model": model,
        "best_val_loss": float(min(val_curve)),
        "epochs_run": int(len(history.history.get("loss", []))),
        "total_epochs": int(config.epochs),
        "history": history,
    }


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train DaSiamRPN on centered tracking pairs (synthetic or COCO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--variant", type=str, default="otb",
                        choices=["big", "vot", "otb"])
    parser.add_argument("--data-source", type=str, default="synthetic",
                        choices=["synthetic", "coco"])
    parser.add_argument("--coco-data-dir", type=str, default=None)
    parser.add_argument("--coco-split", type=str, default="train")
    parser.add_argument("--num-synthetic-train", type=int, default=2048)
    parser.add_argument("--num-synthetic-val", type=int, default=256)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--val-steps", type=int, default=None)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--brightness-delta", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exemplar-size", type=int, default=127)
    parser.add_argument("--search-size", type=int, default=271)
    parser.add_argument("--no-batch-norm", dest="use_batch_norm", action="store_false")
    parser.add_argument("--pos-iou", type=float, default=0.6)
    parser.add_argument("--neg-iou", type=float, default=0.3)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--reg-weight", type=float, default=1.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--lr-schedule", type=str, default="cosine_decay",
                        choices=["cosine_decay", "exponential_decay", "constant"])
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gradient-clipping", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    return parser


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments (``argv`` override keeps tests hermetic)."""
    return _build_parser().parse_args(argv)


def config_from_args(args: argparse.Namespace) -> DaSiamRPNTrainingConfig:
    """Build the training config from a parsed namespace (flag liveness point)."""
    return DaSiamRPNTrainingConfig(
        variant=args.variant,
        data_source=args.data_source,
        coco_data_dir=args.coco_data_dir,
        coco_split=args.coco_split,
        num_synthetic_train=args.num_synthetic_train,
        num_synthetic_val=args.num_synthetic_val,
        steps_per_epoch=args.steps_per_epoch,
        val_steps=args.val_steps,
        augment=args.augment,
        brightness_delta=args.brightness_delta,
        seed=args.seed,
        exemplar_size=args.exemplar_size,
        search_size=args.search_size,
        use_batch_norm=args.use_batch_norm,
        pos_iou=args.pos_iou,
        neg_iou=args.neg_iou,
        cls_weight=args.cls_weight,
        reg_weight=args.reg_weight,
        huber_delta=args.huber_delta,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer,
        lr_schedule_type=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        gradient_clipping=args.gradient_clipping,
        momentum=args.momentum,
        early_stopping_patience=args.early_stopping_patience,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point: parse argv first, then train. Returns a process exit code."""
    args = parse_arguments(argv)
    config = config_from_args(args)
    logger.info(
        f"Config: {config.epochs} epochs, batch={config.batch_size}, "
        f"lr={config.learning_rate}, opt={config.optimizer_type}, "
        f"source={config.data_source}, variant={config.variant}"
    )
    try:
        result = train_dasiamrpn(config, gpu_id=args.gpu)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    logger.info(f"Best val_loss={result['best_val_loss']:.4f}")
    if not np.isfinite(result["best_val_loss"]):
        logger.error("Training diverged (non-finite val_loss)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
