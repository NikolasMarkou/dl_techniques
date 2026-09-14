"""RADConvNet training on DTD (Describable Textures Dataset).

Trains a :class:`~dl_techniques.models.vision.rad_convnet.model.RADConvNet`
(built on RAD-Conv, arXiv:2509.15436) for texture classification on DTD's
predefined split 1 (47 classes, 1880 train / 1880 val / 1880 test images).
Stock ``model.fit()`` only -- no custom ``train_step``.

Usage:
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 python -m train.rad_convnet.train_rad_convnet \
        --variant tiny --epochs 30 --image-size 96

    # Fast end-to-end smoke run:
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 python -m train.rad_convnet.train_rad_convnet \
        --debug --epochs 1 --max-samples 64 --image-size 64
"""

import os
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import keras
import tensorflow as tf

from train.common.gpu import setup_gpu
from train.common.callbacks import create_callbacks
from train.common.run_io import prepare_run_dir, save_training_history_json, default_experiment_name
from dl_techniques.utils.logger import logger
from dl_techniques.models.vision.rad_convnet.model import RADConvNet

from train.rad_convnet.data import make_dtd_dataset

# ---------------------------------------------------------------------

DEFAULT_DTD_ROOT = "/media/arxwn/data0_4tb/datasets/dtd"

# ---------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for RADConvNet DTD training."""

    dtd_root: str = DEFAULT_DTD_ROOT
    split_index: int = 1
    image_size: int = 96
    batch_size: int = 32

    model_variant: str = "tiny"

    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    output_dir: str = "results"
    experiment_name: Optional[str] = None

    gpu_id: Optional[int] = None
    max_samples: Optional[int] = None  # debug: truncate each split for a fast smoke run

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            self.experiment_name = default_experiment_name("rad_convnet", self.model_variant)

# ---------------------------------------------------------------------

def build_model(config: TrainingConfig, num_classes: int) -> RADConvNet:
    """Build a RADConvNet for training.

    :param config: Training configuration.
    :type config: TrainingConfig
    :param num_classes: Number of output classes.
    :type num_classes: int
    :return: A configured, uncompiled ``RADConvNet``.
    :rtype: RADConvNet
    """
    model = RADConvNet.from_variant(
        config.model_variant,
        num_classes=num_classes,
        input_shape=(config.image_size, config.image_size, 3),
    )
    return model

# ---------------------------------------------------------------------

def compile_model(model: RADConvNet, config: TrainingConfig, steps_per_epoch: int) -> None:
    """Compile ``model`` with AdamW + cosine-decay LR, sparse categorical loss.

    :param model: The model to compile, in place.
    :type model: RADConvNet
    :param config: Training configuration.
    :type config: TrainingConfig
    :param steps_per_epoch: Number of batches per epoch, used to size the
        cosine-decay schedule.
    :type steps_per_epoch: int
    """
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=max(1, steps_per_epoch * config.epochs),
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=config.weight_decay
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

# ---------------------------------------------------------------------

def main(config: TrainingConfig) -> None:
    """Run the full RADConvNet DTD training pipeline.

    :param config: Training configuration.
    :type config: TrainingConfig
    """
    setup_gpu(config.gpu_id)

    train_ds, class_names, n_train = make_dtd_dataset(
        config.dtd_root,
        split="train",
        image_size=config.image_size,
        batch_size=config.batch_size,
        split_index=config.split_index,
        augment=True,
        max_samples=config.max_samples,
    )
    val_ds, _, _ = make_dtd_dataset(
        config.dtd_root,
        split="val",
        image_size=config.image_size,
        batch_size=config.batch_size,
        split_index=config.split_index,
        augment=False,
        max_samples=config.max_samples,
    )
    num_classes = len(class_names)

    model = build_model(config, num_classes)

    steps_per_epoch = max(1, n_train // config.batch_size)
    compile_model(model, config, steps_per_epoch)

    run_dir = prepare_run_dir(config)
    callbacks, results_dir = create_callbacks(
        model_name=config.model_variant,
        results_dir_prefix="rad_convnet",
        run_dir=str(run_dir),
        monitor="val_accuracy",
        monitor_mode="max",
        use_lr_schedule=True,
        include_analyzer=False,
    )

    logger.info(f"Training RADConvNet-{config.model_variant} on DTD ({num_classes} classes)")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
    )

    save_training_history_json(history, results_dir)

    final_model_path = Path(results_dir) / "final_model.keras"
    model.save(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")

# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Parsed arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Train RADConvNet on DTD")
    parser.add_argument("--dtd-root", type=str, default=DEFAULT_DTD_ROOT)
    parser.add_argument("--split-index", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--variant", type=str, default="tiny", choices=["tiny", "small", "base"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Debug: truncate each split to this many samples for a fast smoke run.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Shorthand for a fast smoke run: --max-samples 64 if not already set.",
    )
    return parser.parse_args()

# ---------------------------------------------------------------------

if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args()
    max_samples = args.max_samples
    if args.debug and max_samples is None:
        max_samples = 64

    cfg = TrainingConfig(
        dtd_root=args.dtd_root,
        split_index=args.split_index,
        image_size=args.image_size,
        batch_size=args.batch_size,
        model_variant=args.variant,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        gpu_id=args.gpu_id,
        max_samples=max_samples,
    )
    main(cfg)
