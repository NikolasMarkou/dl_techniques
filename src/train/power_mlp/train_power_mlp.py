#!/usr/bin/env python3
"""
PowerMLP classification training script (MNIST / CIFAR-10).

Trains :class:`PowerMLP` (``dl_techniques.models.general_purpose.power_mlp.model``)
on a flattened image-classification task. The model's ``hidden_units`` contract is
``[input_dim, *hidden_widths, num_classes]``: entry 0 is the input width and is
NOT a layer, so the preset table below holds the hidden widths only and
:func:`effective_hidden_units` prepends the input width and appends the class
count. Labels stay integer end to end (``SparseCategoricalCrossentropy``), which
is also what ``ModelAnalyzer`` feeds ``model.evaluate``.

Input normalization: MNIST is one channel (``load_dataset('mnist')`` repeats it
into 3 identical channels, so channel 0 is lossless) standardized with the
canonical 0.1307 / 0.3081; CIFAR-10 is standardized per channel with
``CIFAR10_MEAN`` / ``CIFAR10_STD``; both are then flattened. The validation split
is a seeded shuffle of the training set and is never the test set.

Every run writes to ``<repo>/results/<experiment_name>/`` regardless of the
current working directory (``train.common.resolved_run_dir``).

Usage:
    python -m train.power_mlp.train_power_mlp --help
    python -m train.power_mlp.train_power_mlp --dataset mnist --epochs 3 --seed 0 --gpu 1
    python -m train.power_mlp.train_power_mlp --dataset cifar10 --architecture large \\
        --k 2 --dropout-rate 0.2 --batch-normalization
"""

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np

from dl_techniques.models.general_purpose.power_mlp.model import PowerMLP
from dl_techniques.optimization import optimizer_builder
from dl_techniques.utils.logger import logger

from train.common import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    create_callbacks,
    default_experiment_name,
    load_dataset,
    prepare_run_dir,
    resolved_run_dir,
    save_training_history_json,
    set_seeds,
    setup_gpu,
)
from train.common.callbacks import LearningRateLogger


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# Canonical MNIST statistics for pixels scaled to [0, 1].
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# Per-variable gradient-norm clip: the only guard against ReLU-k blow-up
# (degree k**depth) in the deeper presets. Passed to ``optimizer_builder`` as
# ``gradient_clipping_by_norm_local``, which renames it to Keras' ``clipnorm``.
GRADIENT_CLIP_NORM = 1.0

DATASETS: Tuple[str, ...] = ("mnist", "cifar10")
OPTIMIZERS: Tuple[str, ...] = ("adam", "adamw", "sgd", "rmsprop")

# Hidden widths ONLY. ``PowerMLP`` reads ``hidden_units[0]`` as the input width
# and ``hidden_units[-1]`` as the class count, so neither belongs in this table;
# see ``effective_hidden_units``. Passing ``[256, 128, 64, 10]`` (the old shape)
# silently trained ``[128, 64]`` on a 2352-wide input.
ARCHITECTURE_HIDDEN_WIDTHS: Dict[str, Dict[str, List[int]]] = {
    "small": {"mnist": [128, 64], "cifar10": [256, 128]},
    "default": {"mnist": [256, 128, 64], "cifar10": [512, 256, 128]},
    "large": {"mnist": [512, 256, 128, 64], "cifar10": [1024, 512, 256, 128]},
    "deep": {
        "mnist": [256, 128, 128, 64, 64, 32],
        "cifar10": [512, 256, 256, 128, 128, 64],
    },
}


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

def _validate_validation_split(validation_split: float) -> None:
    """Raise ``ValueError`` unless ``0 < validation_split < 1``.

    Shared by ``TrainingConfig.__post_init__`` and ``prepare_data`` so the rule
    lives once. ``0`` is refused rather than meaning "validate on the test set".
    """
    if not 0.0 < validation_split < 1.0:
        raise ValueError(
            f"validation_split must be strictly inside (0, 1), got {validation_split}. "
            "0 would silently validate on the test set."
        )


@dataclass
class TrainingConfig:
    """Configuration for PowerMLP classification training.

    Every field is read by the trainer (``tests/test_train/
    test_config_fields_are_live.py``). ``experiment_name`` defaults to
    ``powermlp_<dataset>_<architecture>_<timestamp>``.
    """

    # Data
    dataset: str = "mnist"
    validation_split: float = 0.1

    # Model
    architecture: str = "default"
    k: int = 2
    dropout_rate: float = 0.1
    batch_normalization: bool = False

    # Training (defaults are this trainer's intentional historical values)
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 3e-4
    optimizer: str = "adam"
    weight_decay: float = 0.0
    patience: int = 15
    seed: int = 42

    # Monitoring / output
    epoch_analysis: bool = True
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate ranges and derive the experiment name.

        Raises:
            ValueError: If any field is outside its supported range.
        """
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}, got {self.dataset!r}")
        if self.architecture not in ARCHITECTURE_HIDDEN_WIDTHS:
            raise ValueError(
                f"architecture must be one of {tuple(ARCHITECTURE_HIDDEN_WIDTHS)}, "
                f"got {self.architecture!r}"
            )
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        _validate_validation_split(self.validation_split)

        if self.experiment_name is None:
            self.experiment_name = default_experiment_name(
                "powermlp", self.dataset, self.architecture
            )


# ---------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the unparsed PowerMLP trainer parser.

    A purpose-built parser rather than ``create_base_argument_parser()``: that
    parser's ``--image-size`` (ImageNet only), ``--lr-schedule`` (this trainer
    uses ``ReduceLROnPlateau``) and ``--show-plots`` (Agg backend) were dead
    flags here, and its ``--weight-decay`` / ``--optimizer`` never reached the
    optimizer. Defaults mirror :class:`TrainingConfig` so the two cannot drift
    (they are read off a default instance, not retyped). ``--experiment-name``
    defaults to ``None`` so the timestamp is taken when the config is built,
    not when the parser is.

    Returns:
        An ``argparse.ArgumentParser`` with every flag of the trainer.
    """
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(
        description="Train PowerMLP on MNIST or CIFAR-10.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--dataset", type=str, default=defaults.dataset, choices=DATASETS,
                      help="Dataset to train on.")
    data.add_argument("--validation-split", type=float, default=defaults.validation_split,
                      help="Fraction of the training set held out (seeded shuffle), "
                           "strictly inside (0, 1); never the test set.")

    model = parser.add_argument_group("model")
    model.add_argument("--architecture", type=str, default=defaults.architecture,
                       choices=tuple(ARCHITECTURE_HIDDEN_WIDTHS),
                       help="Hidden-width preset.")
    model.add_argument("--k", type=int, default=defaults.k,
                       help="Power of the ReLU-k activation.")
    model.add_argument("--dropout-rate", type=float, default=defaults.dropout_rate,
                       help="Dropout rate after each hidden layer, in [0, 1).")
    model.add_argument("--batch-normalization", action="store_true",
                       default=defaults.batch_normalization,
                       help="Enable batch normalization after each hidden layer.")

    train = parser.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=defaults.epochs,
                       help="Maximum number of training epochs.")
    train.add_argument("--batch-size", type=int, default=defaults.batch_size,
                       help="Training batch size.")
    train.add_argument("--learning-rate", type=float, default=defaults.learning_rate,
                       help="Initial learning rate.")
    train.add_argument("--optimizer", type=str, default=defaults.optimizer, choices=OPTIMIZERS,
                       help="Optimizer (built through optimizer_builder).")
    train.add_argument("--weight-decay", type=float, default=defaults.weight_decay,
                       help="Decoupled optimizer weight decay. Never also applied as a "
                            "kernel regularizer.")
    train.add_argument("--patience", type=int, default=defaults.patience,
                       help="Early-stopping patience in epochs.")
    train.add_argument("--seed", type=int, default=defaults.seed,
                       help="Seed for weights, shuffling and the validation split.")
    train.add_argument("--no-epoch-analysis", dest="epoch_analysis", action="store_false",
                       default=defaults.epoch_analysis,
                       help="Disable the per-epoch ModelAnalyzer callback.")

    out = parser.add_argument_group("output")
    out.add_argument("--output-dir", type=str, default=defaults.output_dir,
                     help="Output root; a relative path is anchored at the repo root.")
    out.add_argument("--experiment-name", type=str, default=None,
                     help="Run directory name (default: powermlp_<dataset>_<architecture>_"
                          "<timestamp>).")
    out.add_argument("--gpu", type=int, default=None,
                     help="GPU device index (default: all visible GPUs).")
    return parser


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line. ``--help`` exits here, before anything expensive.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The parsed ``argparse.Namespace``.
    """
    return _build_parser().parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """Build a :class:`TrainingConfig` from a parsed namespace.

    ``--gpu`` is deliberately not a config field: it is consumed once by
    ``setup_gpu`` in :func:`main`.

    Args:
        args: Namespace returned by :func:`parse_arguments`.

    Returns:
        The validated config.

    Raises:
        ValueError: If a value is outside its supported range.
    """
    return TrainingConfig(
        dataset=args.dataset,
        validation_split=args.validation_split,
        architecture=args.architecture,
        k=args.k,
        dropout_rate=args.dropout_rate,
        batch_normalization=args.batch_normalization,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        epoch_analysis=args.epoch_analysis,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )


# ---------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------

def effective_hidden_units(
        dataset: str,
        architecture: str,
        input_dim: int,
        num_classes: int,
) -> List[int]:
    """Return the ``hidden_units`` list :class:`PowerMLP` is constructed with.

    Args:
        dataset: ``"mnist"`` or ``"cifar10"``.
        architecture: A key of ``ARCHITECTURE_HIDDEN_WIDTHS``.
        input_dim: Flattened input width (entry 0; not a layer).
        num_classes: Number of classes (last entry; the softmax head).

    Returns:
        ``[input_dim, *hidden_widths, num_classes]``.

    Raises:
        ValueError: On an unknown architecture or dataset.
    """
    if architecture not in ARCHITECTURE_HIDDEN_WIDTHS:
        raise ValueError(f"Unknown architecture: {architecture}")
    widths = ARCHITECTURE_HIDDEN_WIDTHS[architecture]
    if dataset not in widths:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return [input_dim, *widths[dataset], num_classes]


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def prepare_data(
        dataset: str,
        validation_split: float,
        seed: int,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Dict[str, Any],
]:
    """Load, standardize, flatten and split a dataset for PowerMLP.

    Args:
        dataset: ``"mnist"`` or ``"cifar10"``.
        validation_split: Fraction of the training set held out, strictly inside
            ``(0, 1)``.
        seed: Seeds the permutation that draws the validation set.

    Returns:
        ``(train, val, test, info)``. ``train`` / ``val`` / ``test`` are
        ``(x, y)`` with ``x`` float32 ``(N, input_dim)`` standardized and ``y``
        int32 ``(N,)``. ``info`` holds ``image_shape`` (H, W, C as displayed:
        MNIST is ``(28, 28, 1)``), ``input_dim``, ``num_classes``, and ``mean`` /
        ``std`` (float32 arrays of shape ``(C,)``) to un-standardize for display.

    Raises:
        ValueError: If ``validation_split`` is not in ``(0, 1)`` or is so small
            that no sample would be held out.
    """
    _validate_validation_split(validation_split)
    (x_train, y_train), (x_test, y_test), _, num_classes = load_dataset(dataset)

    if dataset == "mnist":
        # load_dataset repeats the single MNIST channel 3x; channel 0 is lossless.
        x_train, x_test = x_train[..., :1], x_test[..., :1]
        mean = np.asarray([MNIST_MEAN], dtype=np.float32)
        std = np.asarray([MNIST_STD], dtype=np.float32)
    else:
        mean = np.asarray(CIFAR10_MEAN, dtype=np.float32)
        std = np.asarray(CIFAR10_STD, dtype=np.float32)

    image_shape = tuple(int(d) for d in x_train.shape[1:])
    x_train = ((x_train - mean) / std).astype(np.float32).reshape(len(x_train), -1)
    x_test = ((x_test - mean) / std).astype(np.float32).reshape(len(x_test), -1)
    y_train = np.asarray(y_train).reshape(-1).astype(np.int32)
    y_test = np.asarray(y_test).reshape(-1).astype(np.int32)

    val_size = int(len(x_train) * validation_split)
    if val_size == 0:
        raise ValueError(
            f"validation_split={validation_split} holds out 0 of {len(x_train)} samples"
        )
    order = np.random.default_rng(seed).permutation(len(x_train))
    val_idx, train_idx = order[:val_size], order[val_size:]
    x_val, y_val = x_train[val_idx], y_train[val_idx]
    x_train, y_train = x_train[train_idx], y_train[train_idx]

    info = {
        "image_shape": image_shape,
        "input_dim": int(x_train.shape[1]),
        "num_classes": int(num_classes),
        "mean": mean,
        "std": std,
    }
    logger.info(
        f"PowerMLP data - train {x_train.shape}, val {x_val.shape}, test {x_test.shape}"
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), info


# ---------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------

def create_optimizer(
        name: str,
        learning_rate: float,
        weight_decay: float,
) -> keras.optimizers.Optimizer:
    """Build an optimizer through :func:`optimizer_builder`.

    Args:
        name: One of ``OPTIMIZERS``. Anything else raises: there is no silent
            fall-back to Adam.
        learning_rate: Initial learning rate.
        weight_decay: Decoupled weight decay handed to the optimizer. The model
            never also carries a ``kernel_regularizer`` (that would double it).

    Returns:
        The configured optimizer with ``clipnorm=GRADIENT_CLIP_NORM``.

    Raises:
        ValueError: If ``name`` is not in ``OPTIMIZERS``.
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {name!r}; expected one of {OPTIMIZERS}")
    config: Dict[str, Any] = {
        "type": name,
        "weight_decay": weight_decay,
        "gradient_clipping_by_norm_local": GRADIENT_CLIP_NORM,
    }
    if name == "sgd":
        config["momentum"] = 0.9
    return optimizer_builder(config, learning_rate)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_model(config: TrainingConfig) -> Dict[str, Any]:
    """Train PowerMLP and evaluate it on the test set.

    Args:
        config: A validated :class:`TrainingConfig`.

    Returns:
        A dict with ``run_dir``, ``effective_hidden_units``, ``params`` and the
        final-weights ``test_metrics``.
    """
    logger.info("Starting PowerMLP training")
    set_seeds(config.seed)
    run_dir = prepare_run_dir(config, output_dir=resolved_run_dir(config))
    logger.info(f"Run directory: {run_dir}")

    train, val, test, info = prepare_data(
        config.dataset, config.validation_split, config.seed
    )
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = train, val, test

    hidden_units = effective_hidden_units(
        config.dataset, config.architecture, info["input_dim"], info["num_classes"]
    )
    # ``kernel_regularizer`` is never passed: weight decay lives in the optimizer.
    model = PowerMLP(
        hidden_units=hidden_units,
        k=config.k,
        dropout_rate=config.dropout_rate,
        batch_normalization=config.batch_normalization,
        output_activation="softmax",
        kernel_initializer="glorot_normal",
        bias_initializer="zeros",
    )
    model.build((None, info["input_dim"]))
    model.compile(
        optimizer=create_optimizer(config.optimizer, config.learning_rate, config.weight_decay),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    logger.info(f"  Architecture: {hidden_units}, k={config.k}")
    logger.info(f"  Dropout: {config.dropout_rate}, BatchNorm: {config.batch_normalization}")
    logger.info(
        f"  LR: {config.learning_rate}, Optimizer: {config.optimizer}, "
        f"Weight decay: {config.weight_decay}, Params: {model.count_params():,}"
    )

    callbacks, _ = create_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="powermlp",
        run_dir=str(run_dir),
        monitor="val_loss",
        patience=config.patience,
        use_lr_schedule=False,  # ReduceLROnPlateau, no external schedule
        include_terminate_on_nan=True,
        include_analyzer=config.epoch_analysis,
    )
    # Index 0: `lr` must be in `logs` before CSVLogger reads it.
    callbacks.insert(0, LearningRateLogger())

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    save_training_history_json(history, str(run_dir))

    test_metrics = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    logger.info(f"Test results: {test_metrics}")
    return {
        "run_dir": str(run_dir),
        "effective_hidden_units": hidden_units,
        "params": int(model.count_params()),
        "test_metrics": test_metrics,
    }


# ---------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point. Parses ``argv`` FIRST so ``--help`` allocates nothing."""
    args = parse_arguments(argv)
    config = config_from_args(args)
    setup_gpu(gpu_id=args.gpu)
    try:
        train_model(config)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
