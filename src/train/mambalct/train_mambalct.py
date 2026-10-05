"""MambaLCT tracking training (Pattern 4: detection-style dense prediction).

Trains :class:`MambaLCT` on template/search clips with a per-frame presence
score (binary cross-entropy) plus box regression (:class:`MambaLCTBoxLoss`:
``l1_weight * L1 + giou_weight * (1 - GIoU)``). Two clip sources behind
``--data-source``:

- ``synthetic`` (default): seeded noise backgrounds with a rectangle target
  (:func:`synthetic_tracking_generator`) reframed as clips by
  (:func:`build_mambalct_clip_example`) — offline, deterministic, what the
  smoke tests use;
- ``coco``: still-image pseudo-clips from COCO 2017 via TFDS
  (:func:`coco_tracking_generator`) with center jitter standing in for
  motion (no video dataset ships with this repo; see README).

Usage:
    MPLBACKEND=Agg .venv/bin/python -m train.mambalct.train_mambalct \\
        --epochs 10 --batch-size 4 --data-source synthetic --gpu 1
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
from dl_techniques.models.vision.mambalct import create_mambalct
from dl_techniques.losses.mambalct_loss import MambaLCTBoxLoss
from dl_techniques.datasets.vision.tracking import (
    synthetic_tracking_generator,
    coco_tracking_generator,
    build_mambalct_clip_example,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MambaLCTTrainingConfig:
    """Configuration for MambaLCT clip training.

    ``steps_per_epoch`` / ``val_steps`` derive from the synthetic counts and
    need explicit values only for COCO (unbounded cardinality). Encoder
    overrides (``stage_dims`` / ``stage_depths`` / ``num_heads``) are
    smoke-run scaling knobs; ``None`` keeps the model defaults.
    """

    # Data
    data_source: str = "synthetic"
    coco_data_dir: Optional[str] = None
    coco_split: str = "train"
    num_synthetic_train: int = 512
    num_synthetic_val: int = 64
    steps_per_epoch: Optional[int] = None
    val_steps: Optional[int] = None
    clip_length: int = 2
    max_shift_ratio: float = 0.1
    augment: bool = True
    brightness_delta: float = 0.125
    seed: int = 0
    prefetch_buffer: int = tf.data.AUTOTUNE

    # Model
    variant: str = "mambalct-256"
    template_size: int = 128
    search_size: int = 256
    allow_custom_geometry: bool = False
    stage_dims: Optional[List[int]] = None
    stage_depths: Optional[List[int]] = None
    num_heads: Optional[List[int]] = None
    l1_weight: float = 5.0
    giou_weight: float = 2.0

    # Training
    batch_size: int = 4
    epochs: int = 10
    learning_rate: float = 2e-4
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
                "mambalct", self.data_source
            )
        if self.data_source not in ("synthetic", "coco"):
            raise ValueError(
                f"data_source must be 'synthetic' or 'coco', got {self.data_source!r}"
            )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.clip_length <= 0:
            raise ValueError("clip_length must be positive")
        if self.variant not in ("mambalct-256", "mambalct-384"):
            raise ValueError(
                f"variant must be 'mambalct-256' or 'mambalct-384', "
                f"got {self.variant!r}"
            )
        expected = {
            "mambalct-256": (128, 256),
            "mambalct-384": (192, 384),
        }[self.variant]
        if (
            not self.allow_custom_geometry
            and (self.template_size, self.search_size) != expected
        ):
            raise ValueError(
                f"template_size/search_size ({self.template_size}, "
                f"{self.search_size}) disagree with variant '{self.variant}' "
                f"{expected}: pass allow_custom_geometry=True to train a "
                f"mislabeled geometry on purpose (e.g. smoke tests)"
            )
        if self.search_size <= self.template_size:
            raise ValueError("search_size must exceed template_size")
        if self.brightness_delta < 0:
            raise ValueError("brightness_delta must be non-negative")
        if self.l1_weight < 0 or self.giou_weight < 0:
            raise ValueError("loss weights must be non-negative")


# =============================================================================
# DATA PIPELINE
# =============================================================================

def _example_iterator(config: MambaLCTTrainingConfig, train: bool):
    """Yield ``((template, search_clip), {"scores", "boxes"})`` examples."""
    split_seed = config.seed + (0 if train else 1_000_000)
    rng = np.random.default_rng(split_seed)
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
        yield build_mambalct_clip_example(
            image,
            box,
            clip_length=config.clip_length,
            template_size=config.template_size,
            search_size=config.search_size,
            max_shift_ratio=(
                config.max_shift_ratio if do_augment else 0.0
            ),
            brightness_delta=config.brightness_delta,
            rng=rng,
            augment=do_augment,
        )


def create_clip_dataset(
    config: MambaLCTTrainingConfig, train: bool
) -> Tuple[tf.data.Dataset, Optional[int]]:
    """Build the ``tf.data`` pipeline for one split.

    Returns:
        Dataset yielding ``((template, search_clip), labels)`` batches, plus
        the derived step count (None when the caller must supply it).
    """
    signature = (
        (
            tf.TensorSpec(
                (config.template_size, config.template_size, 3), tf.float32
            ),
            tf.TensorSpec(
                (
                    config.clip_length,
                    config.search_size,
                    config.search_size,
                    3,
                ),
                tf.float32,
            ),
        ),
        {
            "scores": tf.TensorSpec((config.clip_length, 1), tf.float32),
            "boxes": tf.TensorSpec((config.clip_length, 4), tf.float32),
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
    config: MambaLCTTrainingConfig, run_dir: str
) -> Tuple[List[keras.callbacks.Callback], str]:
    """Early-stop / checkpoint / CSV bundle monitoring ``val_loss``."""
    # run_dir=run_dir is REQUIRED: without it the common factory derives its
    # own directory and the run splits across two result trees.
    callbacks, results_dir = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="mambalct",
        run_dir=run_dir,
        monitor="val_loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=config.lr_schedule_type != "constant",
    )
    return callbacks, results_dir


# =============================================================================
# MAIN TRAINING
# =============================================================================

def train_mambalct(
    config: MambaLCTTrainingConfig, gpu_id: Optional[int] = None
) -> Dict[str, Any]:
    """Orchestrate the MambaLCT training pipeline.

    Returns:
        Dict with keys ``model``, ``best_val_loss``, ``epochs_run``,
        ``total_epochs`` and ``history``.
    """
    setup_gpu(gpu_id)
    set_seeds(config.seed)

    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(
        f"Data: {config.data_source}, variant={config.variant}, "
        f"clip={config.clip_length}, augment={config.augment}"
    )

    output_dir = prepare_run_dir(config)

    train_ds, steps_per_epoch = create_clip_dataset(config, train=True)
    val_ds, val_steps = create_clip_dataset(config, train=False)
    logger.info(f"Steps per epoch: {steps_per_epoch}, Val steps: {val_steps}")

    model_kwargs: Dict[str, Any] = {
        "template_size": config.template_size,
        "search_size": config.search_size,
    }
    if config.stage_dims is not None:
        model_kwargs["stage_dims"] = config.stage_dims
    if config.stage_depths is not None:
        model_kwargs["stage_depths"] = config.stage_depths
    if config.num_heads is not None:
        model_kwargs["num_heads"] = config.num_heads
    model = create_mambalct(config.variant, **model_kwargs)
    model.build([
        (None, config.template_size, config.template_size, 3),
        (None, config.clip_length, config.search_size, config.search_size, 3),
        (None, model.context_len, model.embed_dim),
    ])
    model.summary()

    # The shared schedule builder knows decay schedules only; "constant"
    # passes the bare float through and the plateau callback (not an
    # external schedule) drives decay.
    if config.lr_schedule_type == "constant":
        lr: Any = config.learning_rate
    else:
        lr = learning_rate_schedule_builder({
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
    optimizer = optimizer_builder(opt_config, lr)

    model.compile(
        optimizer=optimizer,
        loss={
            "scores": keras.losses.BinaryCrossentropy(),
            "boxes": MambaLCTBoxLoss(
                l1_weight=config.l1_weight, giou_weight=config.giou_weight
            ),
        },
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
        description="Train MambaLCT on template/search clips (synthetic or COCO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-source", type=str, default="synthetic",
                        choices=["synthetic", "coco"])
    parser.add_argument("--coco-data-dir", type=str, default=None)
    parser.add_argument("--coco-split", type=str, default="train")
    parser.add_argument("--num-synthetic-train", type=int, default=512)
    parser.add_argument("--num-synthetic-val", type=int, default=64)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--val-steps", type=int, default=None)
    parser.add_argument("--clip-length", type=int, default=2)
    parser.add_argument("--max-shift-ratio", type=float, default=0.1)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--brightness-delta", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", type=str, default="mambalct-256",
                        choices=["mambalct-256", "mambalct-384"])
    parser.add_argument("--template-size", type=int, default=128)
    parser.add_argument("--search-size", type=int, default=256)
    parser.add_argument("--allow-custom-geometry", dest="allow_custom_geometry",
                        action="store_true")
    parser.add_argument("--l1-weight", type=float, default=5.0)
    parser.add_argument("--giou-weight", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
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


def config_from_args(args: argparse.Namespace) -> MambaLCTTrainingConfig:
    """Build the training config from a parsed namespace (flag liveness point)."""
    return MambaLCTTrainingConfig(
        data_source=args.data_source,
        coco_data_dir=args.coco_data_dir,
        coco_split=args.coco_split,
        num_synthetic_train=args.num_synthetic_train,
        num_synthetic_val=args.num_synthetic_val,
        steps_per_epoch=args.steps_per_epoch,
        val_steps=args.val_steps,
        clip_length=args.clip_length,
        max_shift_ratio=args.max_shift_ratio,
        augment=args.augment,
        brightness_delta=args.brightness_delta,
        seed=args.seed,
        variant=args.variant,
        template_size=args.template_size,
        search_size=args.search_size,
        allow_custom_geometry=args.allow_custom_geometry,
        l1_weight=args.l1_weight,
        giou_weight=args.giou_weight,
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
        f"source={config.data_source}"
    )
    try:
        result = train_mambalct(config, gpu_id=args.gpu)
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
