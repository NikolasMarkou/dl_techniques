"""
AlexNet classification training (Pattern 1: vision classification).

Trains ``dl_techniques.models.vision.alexnet.AlexNet`` on Imagenette (the default),
CIFAR-10, CIFAR-100 or ImageNet. Structure follows ``train/vit/train_vit.py``
(Pattern 1 exemplar) without ViT's deep-supervision or patch-embedding paths, none of
which apply here.

Default recipe: 90 epochs, SGD with momentum 0.9, learning rate 0.1 **divided by 10
every 30 epochs** (``--lr-decay-factor 0.1``), weight decay 5e-4, batch size 256. On
modern hardware the schedule is aggressive and usually wants tuning -- pass
``--lr-schedule`` to change it. The paper (Krizhevsky et al. 2012, section 5) used a
batch of 128 and a learning rate of 0.01, divided by 10 when the validation error
stopped improving (three times in about 90 epochs); the batch and learning-rate
defaults here are not the paper's.

Imagenette (``imagenette/320px-v2`` from TFDS, read from ``$TFDS_DATA_DIR`` and
never downloaded; 9,469 train / 3,925 validation images, 10 classes) is the
default because AlexNet is an ImageNet architecture and Imagenette is a
10-class ImageNet subset that is on disk. It runs at the model's own 227 input
through the shared ``train.common.build_raw_image_dataset`` with
``crop_policy="random_resized_crop"``: a random-resized crop plus flip for
training, a resize to 227 / 0.875 = 259 then centre crop for validation, and
the ImageNet mean/std. CIFAR-10 and CIFAR-100 are each normalised with their own
training-set statistics.

One weights + spectral analysis of the trained model runs after training, into
``model_analysis/`` (``train.common.run_summary.run_data_free_analysis``); a failed
analysis fails the run. ``--epoch-analysis-every N`` adds the same analysis every N
epochs; it is off by default because on this model each one costs about 300 s
against an Imagenette epoch of about 8 s.

Note the input size. AlexNet was defined at 227x227, and the port's default is
227. CIFAR's native 32x32 is upsampled to ``--image-size``; at 32 the feature map
would collapse (the model's minimum spatial extent is 55 and it refuses anything
smaller), so a CIFAR run uses a small but legal size such as 64 or 128, and pays a
proportional input cost. 227 is the faithful choice and the expensive one.

Usage:
    # Default: Imagenette at 227 (TFDS records on disk)
    python -m train.alexnet.train_alexnet --epochs 30 --batch-size 128 \\
        --learning-rate 0.01 --gpu 0

    # ImageNet-scale run from a class-directory tree
    python -m train.alexnet.train_alexnet --dataset imagenet \\
        --train-data-dir /data/imagenet/train --val-data-dir /data/imagenet/val \\
        --image-size 227 --epochs 90 --batch-size 256 \\
        --optimizer sgd --learning-rate 0.1 --lr-schedule exponential_decay \\
        --weight-decay 5e-4 --gpu 0

    # Cheap smoke run on CIFAR-10
    python -m train.alexnet.train_alexnet --dataset cifar10 \\
        --image-size 64 --epochs 5 --batch-size 128 --gpu 0

Note: ``--output-dir`` (default ``"results"``) is resolved relative to the current
working directory, not to the repo root. Invoke this module from the repo root (as in
the examples above) so runs land in the repo-root ``results/`` directory, never
``src/results/``. This matches every other Pattern-1 trainer under ``src/train/``.
"""

import sys
import argparse
import tensorflow as tf
import keras
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any, Union

from train.common import (
    setup_gpu,
    create_callbacks as create_common_callbacks,
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
    IMAGENETTE_NUM_CLASSES,
    build_raw_image_dataset,
    make_imagenet_filesystem_dataset,
    EpochMetricsPlotCallback,
)
from train.common.run_io import (
    default_experiment_name,
    prepare_run_dir,
    save_training_history_json,
)
from train.common.run_summary import run_data_free_analysis
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)
from dl_techniques.models.vision.alexnet import create_alexnet
from dl_techniques.models.vision.alexnet.model import MIN_SPATIAL_EXTENT


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class TrainingConfig:
    """Configuration for AlexNet classification training.

    The data default is Imagenette; the optimisation defaults are the module
    docstring's recipe (not the paper's batch or learning rate). There is no variant field
    because ``models/vision/alexnet`` has no variant table: AlexNet is a single
    architecture, and inventing a variant selector here would have nothing to
    select.
    """

    # Data
    dataset: str = "imagenette"  # imagenette | cifar10 | cifar100 | imagenet
    train_data_dir: Optional[str] = None  # only used for imagenet
    val_data_dir: Optional[str] = None  # only used for imagenet
    image_size: int = 227
    batch_size: int = 256
    augment_data: bool = True

    # Model
    num_classes: int = IMAGENETTE_NUM_CLASSES
    dropout_rate: float = 0.5
    include_top: bool = True
    #: Local-response-normalization hyper-parameters; None keeps the paper's
    #: (n=5, alpha=1e-4, beta=0.75, k=1). Passing --lrn-radius/--lrn-alpha
    #: overrides, and 0 is a legal value that disables the neighbourhood.
    lrn_depth_radius: int = 2
    lrn_alpha: float = 1e-4
    lrn_beta: float = 0.75
    lrn_k: float = 1.0

    # Optimization -- the paper's recipe
    epochs: int = 90
    learning_rate: float = 0.1
    optimizer_type: str = "sgd"  # sgd | adam | adamw
    lr_schedule_type: str = "exponential_decay"  # exponential_decay | cosine_decay | cosine_decay_restarts
    #: `exponential_decay` multiplies the LR by `lr_decay_factor` every
    #: `lr_decay_steps` epochs: "divide by 10 every 30 of 90 epochs", a fixed-step
    #: version of the paper's divide-on-plateau rule. Set `lr_decay_factor=0.5` for the halved schedule some
    #: reproductions use instead.
    lr_decay_steps: int = 30
    lr_decay_factor: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9

    # Callbacks / logging
    monitor_every_n_epochs: int = 5
    #: Run the weights + spectral analysis every N epochs during training; 0
    #: (the default) never does. One analysis of the trained model always runs
    #: at the end. Per epoch it cost 296 to 305 s against 7.5 s of training and
    #: validation on Imagenette at 227 px (RTX 4070; D-114).
    epoch_analysis_every: int = 0
    early_stopping_patience: int = 90  # the paper trains all 90; do not stop early
    output_dir: str = "results"
    experiment_name: Optional[str] = None
    model_variant: str = "alexnet"
    success_threshold: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# DATASETS
# =============================================================================


def create_cifar_dataset(config: TrainingConfig):
    """Build the CIFAR train/validation pipelines.

    :param config: The training configuration.
    :type config: TrainingConfig
    :return: ``(train_ds, val_ds, steps_per_epoch, val_steps)``.
    :rtype: Tuple[Any, Any, int, int]
    :raises ValueError: If ``image_size`` is below the model's minimum.
    """
    _check_image_size(config)
    if config.dataset == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    else:
        # Each CIFAR split with its own training-set statistics (D-103 of
        # plan-2026-10-14T062616-81938e81).
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

    def prepare(images, labels, training):
        # A tf.data map hands this function symbolic tensors, which have no
        # NumPy ``astype``.
        images = tf.cast(images, tf.float32) / 255.0
        images = (images - mean) / std
        # Upsample 32x32 to the model's input. AlexNet was defined at 227; on
        # CIFAR a smaller legal size is a deliberate cost trade, not the paper's
        # configuration.
        images = tf.image.resize(images, (config.image_size, config.image_size))
        if training and config.augment_data:
            images = tf.image.random_flip_left_right(images)
        return images, labels

    x_train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    x_train_ds = x_train_ds.shuffle(len(x_train)).map(
        lambda a, b: prepare(a, b, True),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    x_train_ds = x_train_ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    x_test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    x_test_ds = x_test_ds.map(
        lambda a, b: prepare(a, b, False), num_parallel_calls=tf.data.AUTOTUNE
    )
    x_test_ds = x_test_ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = len(x_train) // config.batch_size
    val_steps = max(1, len(x_test) // config.batch_size)
    return x_train_ds, x_test_ds, steps_per_epoch, val_steps


def _check_image_size(config: TrainingConfig) -> None:
    """Refuse an ``image_size`` below the model's minimum spatial extent.

    :param config: The training configuration.
    :type config: TrainingConfig
    :raises ValueError: If ``image_size`` is below the model's minimum.
    """
    if config.image_size < MIN_SPATIAL_EXTENT:
        raise ValueError(
            f"--image-size {config.image_size} is below AlexNet's minimum spatial "
            f"extent {MIN_SPATIAL_EXTENT}; the final pool is padding='valid' and "
            f"would produce an all-NaN feature map. Use at least "
            f"{MIN_SPATIAL_EXTENT}, and note the cost grows with the square of "
            f"this number."
        )


def create_imagenette_dataset(config: TrainingConfig):
    """Build the Imagenette train/validation pipelines.

    Reads ``imagenette/320px-v2`` from ``$TFDS_DATA_DIR`` (never downloads)
    through the shared ``build_raw_image_dataset`` with
    ``crop_policy="random_resized_crop"``: training images are a random-resized
    crop plus flip of the full-resolution decode, validation images a resize
    to ``image_size / 0.875`` and a centre crop; both are normalised with the
    ImageNet mean/std (Imagenette is an ImageNet subset).

    :param config: The training configuration.
    :type config: TrainingConfig
    :return: ``(train_ds, val_ds, steps_per_epoch, val_steps)``. The training
        pipeline repeats; the validation one is one finite pass, and
        ``val_steps`` covers all of it.
    :rtype: Tuple[Any, Any, int, int]
    :raises ValueError: If ``image_size`` is below the model's minimum.
    """
    _check_image_size(config)
    common = dict(crop_policy="random_resized_crop")
    train_ds, num_train, _ = build_raw_image_dataset(
        "imagenette", config.image_size, config.batch_size,
        is_training=True, augment=config.augment_data, **common,
    )
    val_ds, num_val, _ = build_raw_image_dataset(
        "imagenette", config.image_size, config.batch_size,
        is_training=False, **common,
    )
    steps_per_epoch = num_train // config.batch_size
    val_steps = -(-num_val // config.batch_size)
    return train_ds, val_ds, steps_per_epoch, val_steps


def _count_images(directory: str) -> int:
    """Count images under an ImageNet-style class-directory tree.

    :param directory: Root of a directory-per-class tree.
    :type directory: str
    :return: The number of image files found.
    :rtype: int
    """
    root = Path(directory)
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpeg", ".jpg", ".png"}
    )


# =============================================================================
# CALLBACKS
# =============================================================================


def create_callbacks(
        config: TrainingConfig, run_dir: Union[str, Path]
) -> Tuple[List[keras.callbacks.Callback], str]:
    """Shared callbacks plus a per-epoch metrics figure.

    ``run_dir`` is passed straight through so the checkpoint, CSV and config all land
    in the SAME directory ``prepare_run_dir`` already created, rather than in a
    second independently-timestamped one.

    :param config: The training configuration.
    :type config: TrainingConfig
    :param run_dir: The already-created run directory.
    :type run_dir: Union[str, Path]
    :return: ``(callbacks, results_dir)``.
    :rtype: Tuple[List[keras.callbacks.Callback], str]
    """
    callbacks, results_dir = create_common_callbacks(
        model_name=config.experiment_name or config.model_variant,
        results_dir_prefix="alexnet",
        run_dir=str(run_dir),
        monitor="val_accuracy",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        # DECISION plan-2026-10-14T062616-81938e81/D-114
        # Do NOT restore the shared default (include_analyzer=True, every
        # epoch): on AlexNet it cost ~300 s per 7.5 s epoch. One analysis runs
        # after fit (run_data_free_analysis); per-epoch is opt-in.
        include_analyzer=config.epoch_analysis_every > 0,
        analyzer_epoch_frequency=max(1, config.epoch_analysis_every),
    )
    callbacks.append(EpochMetricsPlotCallback(
        str(Path(results_dir) / "training_metrics"),
        ["accuracy"],
        every_n=config.monitor_every_n_epochs,
    ))
    return callbacks, results_dir


# =============================================================================
# MAIN TRAINING
# =============================================================================


def build_model(config: TrainingConfig) -> keras.Model:
    """Construct the AlexNet model from a configuration.

    :param config: The training configuration.
    :type config: TrainingConfig
    :return: The constructed, unbuilt model.
    :rtype: keras.Model
    """
    return create_alexnet(
        num_classes=config.num_classes,
        input_shape=(config.image_size, config.image_size, 3),
        dropout_rate=config.dropout_rate,
        include_top=config.include_top,
        # The paper's hyper-parameters, always passed explicitly rather than left to
        # the defaults, so a run records what it actually used.
        lrn_hyperparameters={
            "depth_radius": config.lrn_depth_radius,
            "alpha": config.lrn_alpha,
            "beta": config.lrn_beta,
            "k": config.lrn_k,
        },
    )


def train_alexnet(
        config: TrainingConfig, gpu_id: Optional[int] = None
) -> Dict[str, Any]:
    """Orchestrate the AlexNet training pipeline.

    :param config: The training configuration.
    :type config: TrainingConfig
    :param gpu_id: GPU device index, or ``None`` for the default device.
    :type gpu_id: Optional[int]
    :return: Dict with ``model``, ``best_val_acc``, ``early_stop_epoch``,
        ``total_epochs`` and ``history``.
    :rtype: Dict[str, Any]
    """
    setup_gpu(gpu_id)

    if config.experiment_name is None:
        config.experiment_name = default_experiment_name("alexnet")

    logger.info(
        f"Experiment: {config.experiment_name}, dataset: {config.dataset}, "
        f"image_size: {config.image_size}, num_classes: {config.num_classes}"
    )

    output_dir = prepare_run_dir(config)

    # ---- Datasets ----
    if config.dataset == "imagenette":
        train_ds, val_ds, steps_per_epoch, val_steps = create_imagenette_dataset(config)
    elif config.dataset in ("cifar10", "cifar100"):
        train_ds, val_ds, steps_per_epoch, val_steps = create_cifar_dataset(config)
    elif config.dataset == "imagenet":
        if not config.train_data_dir or not config.val_data_dir:
            raise ValueError(
                "--train-data-dir and --val-data-dir are required for "
                "--dataset imagenet"
            )
        train_ds = make_imagenet_filesystem_dataset(
            config.train_data_dir, config.image_size, config.batch_size,
            is_training=True, augment=config.augment_data, augment_color=False,
        )
        val_ds = make_imagenet_filesystem_dataset(
            config.val_data_dir, config.image_size, config.batch_size,
            is_training=False, augment=config.augment_data, augment_color=False,
        )
        steps_per_epoch = _count_images(config.train_data_dir) // config.batch_size
        val_steps = max(1, _count_images(config.val_data_dir) // config.batch_size)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    if steps_per_epoch < 1:
        raise ValueError(
            f"steps_per_epoch is {steps_per_epoch}; the dataset is smaller than "
            f"one batch of {config.batch_size}."
        )

    logger.info(f"Steps per epoch: {steps_per_epoch}, Val steps: {val_steps}")

    # ---- Model ----
    model = build_model(config)
    model.build((None, config.image_size, config.image_size, 3))
    model.summary()
    logger.info(f"Total parameters: {model.count_params():,}")

    # LESSONS L72 (double-weight-decay guard): AdamW applies decoupled weight decay
    # internally, so passing kernel_regularizer as well would penalize the loss AND
    # decay the parameter a second time. Choose one, never both.
    use_adamw = config.optimizer_type.lower() == "adamw"
    kernel_reg = (
        None if use_adamw
        else (keras.regularizers.L2(config.weight_decay) if config.weight_decay > 0 else None)
    )

    # ---- Optimization ----
    # The schedule names are dl_techniques.optimization's, NOT Keras's: the
    # step decay ("divide by 10 every 30 epochs") is `exponential_decay` here, because it is the
    # type that takes a `decay_rate`. There is no `step_decay` key, and passing one
    # raises ValueError naming the three that exist.
    lr_schedule = learning_rate_schedule_builder({
        "type": config.lr_schedule_type,
        "learning_rate": config.learning_rate,
        # Decay every `lr_decay_steps` epochs, i.e. every lr_decay_steps *
        # steps_per_epoch optimizer steps.
        "decay_steps": max(1, steps_per_epoch * config.lr_decay_steps),
        "decay_rate": config.lr_decay_factor,
        "warmup_steps": 0,
    })

    # lr_schedule is a SEPARATE positional parameter of optimizer_builder, not a key
    # inside its config dict: passing "learning_rate" in the dict raises
    # TypeError: optimizer_builder() missing 1 required positional argument.
    optimizer = optimizer_builder(
        {
            "type": config.optimizer_type,
            "weight_decay": config.weight_decay,
            "momentum": config.momentum,
        },
        lr_schedule,
    )

    model.compile(
        optimizer=optimizer,
        # AlexNet returns logits (its fc8 is linear).
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    # `kernel_reg` is reported rather than applied: the AdamW branch above is the
    # correct one, and re-attaching an L2 here would undo it.
    logger.info(f"Kernel regularizer: {kernel_reg}")

    callbacks, results_dir = create_callbacks(config, output_dir)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=2,
    )

    save_training_history_json(history, results_dir)

    # One weights + spectral analysis of the trained model (the best epoch's
    # weights, restored by EarlyStopping), in <run_dir>/model_analysis/. The
    # helper never raises; a run whose analysis did not land fails here
    # (D-102 / D-105 rule).
    sample, labels = next(iter(val_ds))
    analysis = run_data_free_analysis(
        model, np.asarray(sample), np.asarray(labels), history,
        config.experiment_name, Path(results_dir))
    if analysis["status"] != "ok":
        raise RuntimeError(
            f"model analysis {analysis['status']}: {analysis['error']}")

    best_val_acc = float(max(history.history.get("val_accuracy", [0.0])))
    return {
        "model": model,
        "best_val_acc": best_val_acc,
        "early_stop_epoch": len(history.history["loss"]),
        "total_epochs": config.epochs,
        "history": history,
    }


# =============================================================================
# ARGUMENTS
# =============================================================================


#: ``--dataset`` choices and the class count each one implies.
DATASET_CLASSES: Dict[str, int] = {
    "imagenette": IMAGENETTE_NUM_CLASSES,
    "cifar10": 10,
    "cifar100": 100,
    "imagenet": 1000,
}


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Tokens without the program name. ``None`` reads ``sys.argv[1:]``.
    :type argv: Optional[List[str]]
    :return: The parsed arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Train AlexNet (Pattern 1, vision classification)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--dataset", type=str, default="imagenette",
                        choices=list(DATASET_CLASSES))
    parser.add_argument("--train-data-dir", type=str, default=None,
                        help="Required only when --dataset imagenet")
    parser.add_argument("--val-data-dir", type=str, default=None,
                        help="Required only when --dataset imagenet")
    parser.add_argument("--image-size", type=int, default=227,
                        help=f"AlexNet's own extent; must be >= {MIN_SPATIAL_EXTENT}")
    parser.add_argument("--no-augmentation", dest="augment_data",
                        action="store_false")

    # Model
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Auto: 10 / 10 / 100 / 1000 for imagenette / cifar10 / "
                             "cifar100 / imagenet")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lrn-depth-radius", type=int, default=2,
                        help="n = 2*radius+1; the paper uses 2, i.e. n=5")
    parser.add_argument("--lrn-alpha", type=float, default=1e-4)
    parser.add_argument("--lrn-beta", type=float, default=0.75)
    parser.add_argument("--lrn-k", type=float, default=1.0)

    # Training -- the paper's recipe
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "adam", "adamw"])
    parser.add_argument("--lr-schedule", type=str, default="exponential_decay",
                        choices=["exponential_decay", "cosine_decay",
                                 "cosine_decay_restarts"],
                        help="The paper's step decay is 'exponential_decay' here; "
                             "there is no 'step_decay' key in this library")
    parser.add_argument("--lr-decay-steps", type=int, default=30,
                        help="Epochs between LR multiplications")
    parser.add_argument("--lr-decay-factor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Resolved relative to the current working directory, not the repo "
             "root -- invoke this module from the repo root so runs land in the "
             "repo-root results/ directory.",
    )
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--monitor-every", type=int, default=5)
    parser.add_argument("--epoch-analysis-every", type=int, default=0,
                        help="Weights + spectral analysis every N epochs during "
                             "training (0: never; one analysis always runs at "
                             "the end, into model_analysis/)")
    parser.add_argument("--early-stopping-patience", type=int, default=90)
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    parser.add_argument("--success-threshold", type=float, default=None,
                        help="Override the auto-derived val_accuracy threshold")

    return parser.parse_args(argv)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Entry point: parse arguments, train, and report convergence."""
    args = parse_arguments()

    ds = args.dataset.lower()
    num_classes = (
        args.num_classes if args.num_classes is not None else DATASET_CLASSES[ds]
    )

    config = TrainingConfig(
        dataset=ds,
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        augment_data=args.augment_data,
        num_classes=num_classes,
        dropout_rate=args.dropout,
        lrn_depth_radius=args.lrn_depth_radius,
        lrn_alpha=args.lrn_alpha,
        lrn_beta=args.lrn_beta,
        lrn_k=args.lrn_k,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer,
        lr_schedule_type=args.lr_schedule,
        lr_decay_steps=args.lr_decay_steps,
        lr_decay_factor=args.lr_decay_factor,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        monitor_every_n_epochs=args.monitor_every,
        epoch_analysis_every=args.epoch_analysis_every,
        early_stopping_patience=args.early_stopping_patience,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        success_threshold=args.success_threshold,
    )

    logger.info(
        f"Config: dataset={config.dataset}, size={config.image_size}, "
        f"classes={config.num_classes}, epochs={config.epochs}, "
        f"batch={config.batch_size}, lr={config.learning_rate}, "
        f"opt={config.optimizer_type}, wd={config.weight_decay}"
    )

    try:
        result = train_alexnet(config, gpu_id=args.gpu)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    threshold = (
        float(config.success_threshold) if config.success_threshold is not None
        else min(max(2.0 / config.num_classes, 0.05), 0.95)
    )
    converged = result["best_val_acc"] >= threshold
    stopped_early = result["early_stop_epoch"] < 0.5 * result["total_epochs"]
    if converged and not stopped_early:
        logger.info(
            f"=== TRAINING COMPLETED SUCCESSFULLY "
            f"(best_val_acc={result['best_val_acc']:.4f} >= {threshold:.4f}) ==="
        )
    elif converged and stopped_early:
        logger.warning(
            f"Training converged (best_val_acc={result['best_val_acc']:.4f}) but "
            f"early-stopped at epoch {result['early_stop_epoch']}/"
            f"{result['total_epochs']}. Inspect curves."
        )
    else:
        logger.error(
            f"=== TRAINING DID NOT CONVERGE "
            f"(best_val_acc={result['best_val_acc']:.4f} < {threshold:.4f}) ==="
        )
        sys.exit(1)


if __name__ == "__main__":
    main()