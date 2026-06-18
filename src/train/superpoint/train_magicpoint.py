"""MagicPoint stage-1 trainer: SuperPoint detector head on synthetic shapes.

This is stage 1 of the SuperPoint training recipe (DeTone et al., CVPRW 2018):
the detector head is bootstrapped on synthetic geometric primitives with
exactly-known corner / junction keypoints (the "MagicPoint" pre-training). Only
the detector path is supervised here -- the SuperPoint model still emits its
``{"keypoints", "descriptors"}`` dict, but the loss is keyed ONLY on
``"keypoints"`` so Keras binds the 65-class detector loss to that head and the
descriptor head simply receives no gradient. This keeps ONE model class across
all three training stages (no detector-only subclass to maintain).

Data is an infinite ``tf.data`` stream over
:func:`dl_techniques.datasets.synthetic_shapes.synthetic_shapes_generator`,
yielding ``(image (H, W, 1) float32, {"keypoints": grid_label (H/8, W/8) int32})``.
``H`` and ``W`` are kept divisible by 8 so the detector grid ``H/8 x W/8`` matches
the label map.

Results are written to the repo-root ``results/`` directory.

Usage::

    MPLBACKEND=Agg python -m train.superpoint.train_magicpoint \\
        --variant tiny --input-size 128 --batch-size 16 --epochs 50 --gpu 1

    # Fast smoke (seconds on GPU1):
    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg \\
        python -m train.superpoint.train_magicpoint --smoke --gpu 1
"""

import time
import json
import keras
import argparse
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from train.common import (
    setup_gpu,
    set_seeds,
    create_callbacks as create_common_callbacks,
    save_config_json,
)
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)
from dl_techniques.models.superpoint import create_superpoint
from dl_techniques.losses.superpoint_loss import SuperPointDetectorLoss
from dl_techniques.datasets.synthetic_shapes import (
    synthetic_shapes_generator,
    DEFAULT_CELL,
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


@dataclass
class MagicPointConfig:
    """Configuration for MagicPoint (SuperPoint detector) stage-1 training."""

    # Data
    input_size: int = 128
    channels: int = 1
    cell: int = DEFAULT_CELL  # detector-head cell size (8 -> 65 classes)

    # Model
    variant: str = "tiny"

    # Training
    batch_size: int = 16
    epochs: int = 50
    steps_per_epoch: int = 500

    # Optimization
    learning_rate: float = 1e-3
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "cosine_decay"
    warmup_epochs: int = 5
    gradient_clipping: float = 1.0

    # Monitoring
    early_stopping_patience: int = 15

    # Reproducibility
    seed: int = 42

    # Output
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    # Smoke mode: tiny everything, runs in seconds on GPU1.
    smoke: bool = False

    def __post_init__(self):
        if self.smoke:
            self.input_size = 64
            self.batch_size = 2
            self.epochs = 1
            self.steps_per_epoch = 2
            self.variant = "tiny"
            self.early_stopping_patience = 1

        if self.input_size <= 0 or self.channels <= 0:
            raise ValueError("Invalid input size or channel configuration")
        if self.input_size % self.cell != 0:
            raise ValueError(
                f"input_size ({self.input_size}) must be divisible by cell "
                f"({self.cell}) so the detector grid matches the label map"
            )
        if self.input_size < 64:
            raise ValueError(
                f"input_size ({self.input_size}) must be >= 64 so H/8 >= 8 "
                f"(the 8x8 detector reshape stays meaningful)"
            )
        if self.variant not in ("tiny", "base", "large"):
            raise ValueError(f"Unknown variant: {self.variant}")

        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"magicpoint_{self.variant}_{timestamp}"


# ---------------------------------------------------------------------
# DATASET BUILDER
# ---------------------------------------------------------------------


def create_dataset(config: MagicPointConfig) -> tf.data.Dataset:
    """Build an infinite tf.data stream of synthetic-shapes detector examples.

    Yields ``(image (H, W, 1) float32, {"keypoints": grid_label (Hc, Wc) int32})``
    so Keras binds :class:`SuperPointDetectorLoss` to the detector head.

    Args:
        config: Training configuration.

    Returns:
        A batched, prefetched ``tf.data.Dataset``.
    """
    H = W = config.input_size
    Hc = H // config.cell
    Wc = W // config.cell

    output_signature = (
        tf.TensorSpec(shape=(H, W, config.channels), dtype=tf.float32),
        tf.TensorSpec(shape=(Hc, Wc), dtype=tf.int32),
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: synthetic_shapes_generator(
            H, W, seed=config.seed, cell=config.cell, n_samples=None
        ),
        output_signature=output_signature,
    )

    # Wrap the integer grid label in the detector-head dict key so Keras keys the
    # detector loss to "keypoints" and leaves the descriptor head ungradiented.
    dataset = dataset.map(
        lambda img, label: (img, {"keypoints": label}),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------


def train_magicpoint(config: MagicPointConfig) -> keras.Model:
    """Train the SuperPoint detector head on synthetic shapes (MagicPoint)."""
    logger.info(f"Starting MagicPoint training: {config.experiment_name}")

    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(config, str(output_dir), "config.json")

    # Dataset
    train_dataset = create_dataset(config)

    # Model: full SuperPoint, but only the detector head is supervised below.
    input_shape = (config.input_size, config.input_size, config.channels)
    model = create_superpoint(config.variant, input_shape=input_shape)
    model.build((None, *input_shape))
    model.summary(print_fn=logger.info)

    # Optimizer with LR schedule (mirrors the cliffordnet trainer).
    lr_schedule = learning_rate_schedule_builder(
        {
            "type": config.lr_schedule_type,
            "learning_rate": config.learning_rate,
            "decay_steps": config.steps_per_epoch * config.epochs,
            "warmup_steps": config.steps_per_epoch * config.warmup_epochs,
            "alpha": 0.01,
        }
    )
    optimizer = optimizer_builder(
        {
            "type": config.optimizer_type,
            "gradient_clipping_by_norm": config.gradient_clipping,
        },
        lr_schedule,
    )

    # Detector-only: loss keyed to "keypoints" -> descriptor head gets no grad.
    # DECISION plan_2026-06-18_e1411ebf/D-004: jit_compile=False is REQUIRED.
    # Do NOT enable XLA here. The descriptor head's bicubic upsample
    # (keras.ops.image.resize -> ResizeBicubic) has no XLA_GPU_JIT OpKernel in
    # TF 2.18, so any XLA-compiled fit step raises a tf2xla conversion error even
    # though only the detector head is supervised (the descriptor subgraph is
    # still traced). See decisions.md D-004.
    model.compile(
        optimizer=optimizer,
        loss={"keypoints": SuperPointDetectorLoss()},
        jit_compile=False,
    )
    logger.info(f"Model compiled with {model.count_params():,} parameters")

    # Callbacks: monitor train loss (no validation split in the smoke stream).
    callbacks, _ = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="magicpoint",
        run_dir=str(output_dir),
        monitor="loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_terminate_on_nan=True,
        include_analyzer=False,
    )

    start_time = time.time()
    history = model.fit(
        train_dataset,
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        callbacks=callbacks,
        verbose=1,
    )
    logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")

    # Save final model + history.
    try:
        model_path = output_dir / "final_model.keras"
        model.save(model_path)
        logger.info(f"Final model saved to: {model_path}")
    except Exception as e:
        logger.error(f"Failed to save final model: {e}")

    try:
        history_dict = {
            k: [float(v) for v in vals] for k, vals in history.history.items()
        }
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history_dict, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save training history: {e}")

    return model


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MagicPoint (SuperPoint detector) on synthetic shapes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variant", choices=["tiny", "base", "large"], default="tiny"
    )
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Force tiny config (input 64, batch 2, 1 epoch, 2 steps) for a "
        "fast build+fit smoke check.",
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_gpu(gpu_id=args.gpu)
    set_seeds(args.seed)

    config = MagicPointConfig(
        input_size=args.input_size,
        variant=args.variant,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        smoke=args.smoke,
    )

    logger.info(
        f"Config: variant={config.variant}, input={config.input_size}, "
        f"batch={config.batch_size}, epochs={config.epochs}, "
        f"steps/epoch={config.steps_per_epoch}, smoke={config.smoke}"
    )

    try:
        train_magicpoint(config)
        logger.info("MagicPoint training completed successfully!")
    except Exception as e:
        logger.error(f"MagicPoint training failed: {e}")
        raise


if __name__ == "__main__":
    main()
