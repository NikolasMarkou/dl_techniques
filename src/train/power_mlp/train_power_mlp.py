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

Input scaling (``--input-scaling``): ``standardize`` subtracts a mean and divides
by a std (MNIST: 0.1307 / 0.3081 on one channel, since ``load_dataset('mnist')``
repeats it into 3 identical channels and channel 0 is lossless; CIFAR-10: per
channel ``CIFAR10_MEAN`` / ``CIFAR10_STD``); ``unit`` keeps the loader's plain
``[0, 1]`` pixels (the default). Both are then flattened. The validation split is
a seeded shuffle of the training set and is never the test set.

Initialization (``--kernel-initializer``): ReLU-k composes to degree ``k**depth``,
so a gain above the fixed point blows the logits up doubly exponentially with
depth. The pre-fit sanity evaluate therefore compares the untrained loss with
``ln(num_classes)`` and logs a WARNING when the ratio exceeds
``INITIAL_LOSS_WARN_FACTOR`` (it does not raise, a probe run may want the number);
``results_summary.json`` records ``initial_loss_ratio``, ``init_scale_warning`` and
``initial_loss_mode``: a batch-normalized model is measured in training mode (its
moving statistics are restored afterwards), any other model in inference mode.
The default is ``lecun_normal`` on ``unit`` inputs (measured initial loss about
``ln(C)``); ``glorot_normal`` on ``standardize`` inputs starts about 95x above it.
Batch normalization (``--batch-normalization`` / ``--no-batch-normalization``) has a
PER-DATASET default, ``BATCH_NORMALIZATION_BY_DATASET``: ON for ``mnist``, OFF for
``cifar10``. Each entry comes from a pre-registered 3-seed x 10-epoch grid on that
dataset (MNIST: plan decision D-021, recorded in D-025, BN +0.0024 test accuracy;
CIFAR-10: D-027, recorded in D-028, BN -0.031). An explicit flag always wins; with
neither, ``TrainingConfig`` resolves ``None`` from the table in ``__post_init__`` so
``config.json`` and the summary record the value actually used. ``lecun_normal`` +
``unit`` inputs are the global defaults.

Per-epoch ``ModelAnalyzer`` is opt-in (``--epoch-analysis``); the final
``run_model_analysis`` always runs.

A run whose ``loss`` or ``val_loss`` turns non-finite is ``status: "diverged"``: no
evaluation, figures, analysis or ``final_model.keras``, a reduced strict-JSON
``results_summary.json`` (non-finite values as ``null``, ``best_epoch: null``) is
written, and only then a ``RuntimeError`` is raised. Every run also writes
``run.log`` (the ``dl`` logger, plus the LR-reduction and early-stop lines that
Keras only prints to stdout).

Every run writes to ``<repo>/results/<experiment_name>/`` regardless of the
current working directory (``train.common.resolved_run_dir``).

Usage:
    python -m train.power_mlp.train_power_mlp --help
    python -m train.power_mlp.train_power_mlp --dataset mnist --epochs 50 --architecture default --k 2
    python -m train.power_mlp.train_power_mlp --dataset cifar10 --architecture large \\
        --k 2 --dropout-rate 0.2
    python -m train.power_mlp.train_power_mlp --dataset mnist --no-batch-normalization
    python -m train.power_mlp.train_power_mlp --dataset cifar10 --batch-normalization

Results land in ``results/<experiment_name>/`` at the repository root (never
under ``src/``): ``config.json``, ``training_log.csv`` (with ``lr``),
``training_history.json``, ``best_model.keras``, ``final_model.keras``,
``results_summary.json``, ``run.log``, ``visualizations/`` and ``model_analysis/`` (plus
``epoch_analysis/`` only with ``--epoch-analysis``). The CSV ``epoch`` column is
0-based; ``best_epoch`` in the summary is 1-based (``best_epoch_csv_index`` is the
0-based twin).
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np

from dl_techniques.models.general_purpose.power_mlp.model import PowerMLP
from dl_techniques.optimization import optimizer_builder
from dl_techniques.utils.logger import LOGGER_FORMAT, logger

from train.common import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    create_callbacks,
    default_experiment_name,
    get_class_names,
    json_numpy_default,
    load_dataset,
    prepare_run_dir,
    resolve_monitor_mode,
    resolved_run_dir,
    run_model_analysis,
    save_training_history_json,
    set_seeds,
    setup_gpu,
    validate_model_loading,
)
from train.common.callbacks import LearningRateLogger, best_checkpoint_path
from train.power_mlp.visualization import (
    TrainingDashboardCallback,
    plot_calibration,
    plot_confident_errors,
    plot_confusion_matrix,
    plot_per_class_metrics,
)


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

# The metric that drives early stopping, the best checkpoint and best_epoch.
MONITOR = "val_loss"

# Samples used by the pre-fit sanity evaluate and by validate_model_loading.
SANITY_SAMPLES = 1024
LOAD_CHECK_SAMPLES = 64
# Best-checkpoint vs in-memory final weights should be identical (D-011).
WEIGHT_MISMATCH_TOLERANCE = 1e-6

DATASETS: Tuple[str, ...] = ("mnist", "cifar10")
OPTIMIZERS: Tuple[str, ...] = ("adam", "adamw", "sgd", "rmsprop")
KERNEL_INITIALIZERS: Tuple[str, ...] = (
    "glorot_normal", "he_normal", "lecun_normal", "glorot_uniform",
)
# ``standardize``: (x - mean) / std. ``unit``: the loader's plain [0, 1] pixels.
INPUT_SCALINGS: Tuple[str, ...] = ("standardize", "unit")

# ``_check_initial_loss`` warns when loss / ln(num_classes) is STRICTLY above this.
INITIAL_LOSS_WARN_FACTOR = 10.0

# The remedy named by the initial-loss WARNING and by both divergence errors (review
# iteration 3, C6: the old hint named only the two knobs already at their shipped
# values in every hazardous default run and omitted the one measured to fix them all).
REMEDY_HINT = (
    "Try --kernel-initializer lecun_normal and --input-scaling unit; if you are already "
    "on lecun_normal + unit, try --batch-normalization (measured: every preset and "
    "k in {2, 3} then starts at about 1.05-1.25x ln(num_classes)), or lower --k."
)

# ``results_summary.json["status"]``: the run finished / stopped on a non-finite loss.
STATUS_OK = "ok"
STATUS_DIVERGED = "diverged"

# The run's own narrative: the ``dl`` logger, tee'd into ``<run_dir>/run.log``.
RUN_LOG_NAME = "run.log"

# Files whose presence means an experiment directory already holds a run:
# ``train_model`` refuses to start there (review iteration 3, C3: a reused
# ``--experiment-name`` used to merge two runs into one directory).
RUN_ARTIFACT_NAMES: Tuple[str, ...] = (
    "results_summary.json", "config.json", "best_model.keras", RUN_LOG_NAME, "training_log.csv",
)

# DECISION plan-2026-09-18T213948-68dcb72c/D-028: BN is ON for MNIST (D-025) and OFF for
# CIFAR-10, each from its own pre-registered 3-seed x 10-epoch grid (CIFAR-10: BN mean
# 0.4791 vs 0.5102 without, BN worse in 3 of 3 seeds). Do NOT collapse this back to
# one global flag "for simplicity": either value is the measured loser on the other
# dataset. Change an entry only through a new multi-seed grid and a decision entry;
# the literal pins are test_batch_normalization_default_is_resolved_per_dataset and
# the two default-parameter pins.
# Resolved by ``TrainingConfig.__post_init__`` when ``batch_normalization`` is None
# (the CLI default). A dataset absent here is a ``KeyError`` at config time, not a
# silent fallback: a new dataset needs its own grid.
BATCH_NORMALIZATION_BY_DATASET: Dict[str, bool] = {"mnist": True, "cifar10": False}

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
    ``powermlp_<dataset>_<architecture>_<timestamp>``. ``batch_normalization=None``
    (the default) is replaced in ``__post_init__`` by
    ``BATCH_NORMALIZATION_BY_DATASET[dataset]``, so after construction it is always a
    ``bool``.
    """

    # Data
    dataset: str = "mnist"
    validation_split: float = 0.1

    # Model
    architecture: str = "default"
    k: int = 2
    dropout_rate: float = 0.1
    # DECISION plan-2026-09-18T213948-68dcb72c/D-025: the defaults below
    # (batch_normalization=True ON MNIST (per-dataset table since D-028),
    # kernel_initializer="lecun_normal", input_scaling
    # "unit") come from the pre-registered 3-seed x 10-epoch MNIST grid of D-021:
    # BN beat the best non-BN arm by +0.00243 mean test accuracy (bar 0.002) with
    # its worst seed above that mean, and lecun vs glorot tied under BN
    # (-0.00010). Do NOT turn BN back off on MNIST or switch to glorot_normal because a
    # single run or a 3-epoch table looks better: one seed and 3 epochs already
    # picked a winner the 5-seed replication refuted (D-019). Change the default
    # only through a new multi-seed grid and a new decision entry; the literal
    # pin is test_training_config_defaults_are_the_d025_outcome.
    # None -> BATCH_NORMALIZATION_BY_DATASET[dataset] in __post_init__ (D-028).
    batch_normalization: Optional[bool] = None
    # Both defaults are explained by the D-025 anchor above; the literal pin is
    # test_training_config_defaults_are_the_d025_outcome.
    kernel_initializer: str = "lecun_normal"
    input_scaling: str = "unit"

    # Training (defaults are this trainer's intentional historical values)
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 3e-4
    optimizer: str = "adam"
    weight_decay: float = 0.0
    patience: int = 15
    seed: int = 42

    # Monitoring / output
    epoch_analysis: bool = False
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
        if self.batch_normalization is None:
            self.batch_normalization = BATCH_NORMALIZATION_BY_DATASET[self.dataset]
        if self.kernel_initializer not in KERNEL_INITIALIZERS:
            raise ValueError(
                f"kernel_initializer must be one of {KERNEL_INITIALIZERS}, "
                f"got {self.kernel_initializer!r}"
            )
        if self.input_scaling not in INPUT_SCALINGS:
            raise ValueError(
                f"input_scaling must be one of {INPUT_SCALINGS}, got {self.input_scaling!r}"
            )
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
    data.add_argument("--input-scaling", type=str, default=defaults.input_scaling,
                      choices=INPUT_SCALINGS,
                      help="'standardize' = (x - mean) / std; 'unit' = plain [0, 1] pixels "
                           "(default: unit).")
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
    # default=None, NOT defaults.batch_normalization: that is the MNIST value (a bare
    # ``TrainingConfig()`` is an MNIST config); the dataset-dependent default is
    # resolved after parsing, from BATCH_NORMALIZATION_BY_DATASET.
    model.add_argument("--batch-normalization", action=argparse.BooleanOptionalAction,
                       default=None,
                       help="Batch normalization after each hidden layer. Default per "
                            "dataset: on for mnist, off for cifar10 (each measured); "
                            "--batch-normalization / --no-batch-normalization override it.")
    model.add_argument("--kernel-initializer", type=str, default=defaults.kernel_initializer,
                       choices=KERNEL_INITIALIZERS,
                       help="Kernel initializer of every layer; it sets the initial logit "
                            "scale (see the initial-loss WARNING).")

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
    train.add_argument("--epoch-analysis", action="store_true",
                       default=defaults.epoch_analysis,
                       help="Run the per-epoch ModelAnalyzer callback (off by default; the "
                            "final analysis always runs).")

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
        kernel_initializer=args.kernel_initializer,
        input_scaling=args.input_scaling,
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
        input_scaling: str,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Dict[str, Any],
]:
    """Load, scale, flatten and split a dataset for PowerMLP.

    Args:
        dataset: ``"mnist"`` or ``"cifar10"``.
        validation_split: Fraction of the training set held out, strictly inside
            ``(0, 1)``.
        seed: Seeds the permutation that draws the validation set.
        input_scaling: One of ``INPUT_SCALINGS``. Required (no default, so a
            caller cannot silently inherit a second copy of the default).

    Returns:
        ``(train, val, test, info)``. ``train`` / ``val`` / ``test`` are
        ``(x, y)`` with ``x`` float32 ``(N, input_dim)`` scaled and ``y``
        int32 ``(N,)``. ``info`` holds ``image_shape`` (H, W, C as displayed:
        MNIST is ``(28, 28, 1)``), ``input_dim``, ``num_classes``,
        ``input_scaling``, and ``mean`` / ``std`` (float32 arrays of shape
        ``(C,)``) to un-scale for display. For ``"unit"`` they are zeros / ones,
        so un-scaling is the identity.

    Raises:
        ValueError: If ``validation_split`` is not in ``(0, 1)``, is so small
            that no sample would be held out, or ``input_scaling`` is unknown.
    """
    _validate_validation_split(validation_split)
    if input_scaling not in INPUT_SCALINGS:
        raise ValueError(
            f"input_scaling must be one of {INPUT_SCALINGS}, got {input_scaling!r}"
        )
    (x_train, y_train), (x_test, y_test), _, num_classes = load_dataset(dataset)

    if dataset == "mnist":
        # load_dataset repeats the single MNIST channel 3x; channel 0 is lossless.
        x_train, x_test = x_train[..., :1], x_test[..., :1]
        mean = np.asarray([MNIST_MEAN], dtype=np.float32)
        std = np.asarray([MNIST_STD], dtype=np.float32)
    else:
        mean = np.asarray(CIFAR10_MEAN, dtype=np.float32)
        std = np.asarray(CIFAR10_STD, dtype=np.float32)
    if input_scaling == "unit":
        mean, std = np.zeros_like(mean), np.ones_like(std)

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
        "input_scaling": input_scaling,
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

# DECISION plan-2026-09-18T213948-68dcb72c/D-029: the epoch-0 numbers (initial-loss guard
# AND the dashboard baseline star) come from THIS one function. Do NOT let the dashboard
# callback call ``model.evaluate`` itself "because it is simpler": under batch
# normalization that is the inference-mode pass (9.7e10 where the guard reads 2.68 on
# the same model) and the star and the summary disagree. Guards:
# test_a_batch_normalized_dashboard_baseline_equals_the_guard_not_the_inference_value.
def _untrained_metrics(
        model: keras.Model, x: np.ndarray, y: np.ndarray, batch_size: int = SANITY_SAMPLES
) -> Tuple[Dict[str, float], str]:
    """``loss`` and ``accuracy`` of the UNTRAINED model on ``(x, y)`` and the mode used.

    The single measurement behind both epoch-0 numbers (the initial-loss guard and
    the dashboard's baseline star), so the two cannot disagree (review iteration 3,
    C2: the dashboard used ``model.evaluate`` and read 9.7e10 where the guard read
    2.68 on the same batch-normalized model).

    A model with batch normalization is measured in TRAINING mode: at init the
    moving statistics are (0, 1), so an inference-mode pass sees un-normalized
    activations (measured 4.2e11 where the first training step, which uses batch
    statistics, sees 2.7). The pass runs in chunks of ``batch_size`` rows (each
    chunk normalized by its own batch statistics, like a training step; ``x`` no
    larger than ``batch_size`` is one pass), assigns the moving statistics, so the
    weights are snapshotted before and restored after; the model is left exactly as
    it was. Dropout stays active (the regime of the first training step). A model
    without batch normalization keeps ``model.evaluate`` (inference mode).

    Args:
        model: Built and compiled model (``batch_normalization`` attribute set).
        x: Inputs ``(N, input_dim)``.
        y: Integer labels ``(N,)``.
        batch_size: Rows per pass.

    Returns:
        ``({"loss": ..., "accuracy": ...}, mode)`` with ``mode`` ``"training"`` or
        ``"inference"``. The values are not checked for finiteness here.
    """
    if not model.batch_normalization:
        metrics = model.evaluate(x, y, batch_size=batch_size, verbose=0, return_dict=True)
        return {k: float(v) for k, v in metrics.items()}, "inference"
    snapshot = model.get_weights()
    loss_sum, correct = 0.0, 0
    try:
        for start in range(0, len(x), batch_size):
            probabilities = model(x[start:start + batch_size], training=True)
            per_sample = keras.ops.convert_to_numpy(
                keras.losses.sparse_categorical_crossentropy(y[start:start + batch_size], probabilities)
            )
            loss_sum += float(np.sum(per_sample, dtype=np.float64))
            predicted = np.argmax(keras.ops.convert_to_numpy(probabilities), axis=-1)
            correct += int(np.sum(predicted == y[start:start + batch_size]))
    finally:
        model.set_weights(snapshot)  # undo the moving-statistics update
    return {"loss": loss_sum / len(x), "accuracy": correct / len(x)}, "training"


def _check_initial_loss(
        model: keras.Model, x: np.ndarray, y: np.ndarray
) -> Tuple[float, float, bool, str]:
    """Evaluate the untrained model on a small slice; refuse NaN, warn on a huge loss.

    The loss is compared with ``ln(num_classes)``, the loss of a uniform
    prediction. ReLU-k composes to degree ``k**depth``, so a large init gain or an
    unscaled input starts the run at ``loss / ln(C)`` in the tens to the millions
    and the first epochs are spent recovering (iteration 1: 218.6 vs 2.30). The
    measurement (and the batch-normalization training-mode rule) is
    :func:`_untrained_metrics`.

    Args:
        model: Built and compiled model.
        x: Inputs ``(N, input_dim)``.
        y: Integer labels ``(N,)``.

    Returns:
        ``(loss, ratio, warned, mode)``: the finite scalar loss on ``x``,
        ``loss / ln(num_classes)``, whether ``ratio > INITIAL_LOSS_WARN_FACTOR``
        (strict) so a WARNING naming the remedies (``--kernel-initializer``,
        ``--input-scaling``, ``--batch-normalization``) was logged, and how the loss
        was measured (``"inference"`` or ``"training"``). It does not raise on a
        large loss.

    Raises:
        RuntimeError: If the loss is NaN or infinite. Raising (not returning)
            is deliberate: a silent return would leave an empty run directory.
    """
    metrics, mode = _untrained_metrics(model, x, y)
    loss = metrics["loss"]
    uniform_loss = float(np.log(model.hidden_units[-1]))
    logger.info(
        f"Sanity evaluate BEFORE fit ({mode} mode): initial loss {loss:.4f} on {len(x)} "
        f"train samples (uniform-prediction loss would be {uniform_loss:.4f})"
    )
    if not np.isfinite(loss):
        raise RuntimeError(
            f"Initial loss is {loss} on {len(x)} samples: the model diverges before "
            f"training. {REMEDY_HINT}"
        )
    ratio = loss / uniform_loss
    warned = ratio > INITIAL_LOSS_WARN_FACTOR
    if warned:
        logger.warning(
            f"Initial loss {loss:.4f} is {ratio:.1f}x ln(num_classes)={uniform_loss:.4f} "
            f"(warn above {INITIAL_LOSS_WARN_FACTOR:g}x): the initial logit scale is far "
            f"too large and early epochs will be spent recovering. {REMEDY_HINT}"
        )
    return loss, ratio, warned, mode


def _evaluate(model: keras.Model, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """``model.evaluate`` as a plain ``{metric: float}`` dict."""
    metrics = model.evaluate(x, y, batch_size=1024, verbose=0, return_dict=True)
    return {k: float(v) for k, v in metrics.items()}


def _best_epoch(history: Dict[str, List[float]]) -> int:
    """1-based epoch that is best under ``MONITOR`` (direction from ``resolve_monitor_mode``).

    Raises:
        ValueError: If any monitored value is NaN or infinite. ``np.argmin`` of a
            list holding a NaN returns the NaN's index, so a diverged run would
            otherwise report its NaN epoch as the best one.
    """
    values = np.asarray(history[MONITOR], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"cannot pick a best epoch: {MONITOR} has non-finite values {values.tolist()}")
    mode = resolve_monitor_mode(MONITOR)
    return int(np.argmin(values) if mode == "min" else np.argmax(values)) + 1


def _non_finite_metrics(history: Dict[str, List[float]]) -> List[str]:
    """Names among ``loss`` and ``MONITOR`` that are empty or hold a NaN / infinity."""
    return [
        key for key in dict.fromkeys(("loss", MONITOR))
        if not history.get(key) or not np.all(np.isfinite(history[key]))
    ]


def _lr_reduction_epochs(lrs: Sequence[float]) -> List[int]:
    """1-based epochs whose learning rate is lower than the previous epoch's.

    ``lrs`` is the ``lr`` history (``LearningRateLogger`` writes the rate used
    DURING each epoch), so an entry ``e`` means ``ReduceLROnPlateau`` acted
    after epoch ``e - 1`` and epoch ``e`` was the first one trained at the lower
    rate. Keras prints that message to stdout, not to the logger, so this is how
    the run's own log and summary learn about it.

    Args:
        lrs: Per-epoch learning rates, epoch 1 first.

    Returns:
        The reduction epochs in ascending order (empty for a constant rate).
    """
    return [i + 1 for i in range(1, len(lrs)) if lrs[i] < lrs[i - 1]]


def _write_summary(run_dir: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Write ``results_summary.json`` as STRICT JSON and return what was written.

    Every summary (normal or diverged) goes through here. The dict is
    round-tripped through ``json`` (numpy values via ``json_numpy_default``),
    every non-finite float becomes ``None`` (``null``), and the dump uses
    ``allow_nan=False`` so a ``NaN`` / ``Infinity`` token can never reach the
    file (jq and most non-Python readers reject them).

    Args:
        run_dir: Existing run directory.
        summary: The summary dict (may hold numpy scalars / arrays).

    Returns:
        The sanitized, pure-JSON dict that was written.
    """
    def clean(value: Any) -> Any:
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    written = clean(json.loads(json.dumps(summary, default=json_numpy_default)))
    with open(run_dir / "results_summary.json", "w") as f:
        json.dump(written, f, indent=2, allow_nan=False)
    logger.info(f"Wrote {run_dir / 'results_summary.json'}")
    return written


def _load_best_metrics(
        run_dir: Path, x: np.ndarray, y: np.ndarray
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """Evaluate the reloaded ``best_model.keras``.

    Returns:
        ``(metrics, None)`` on success, ``(None, error_text)`` if the checkpoint
        is missing or does not load.
    """
    path = best_checkpoint_path(str(run_dir))
    try:
        best_model = keras.models.load_model(path)
        return _evaluate(best_model, x, y), None
    except Exception as e:  # noqa: BLE001 - reported, not fatal
        logger.warning(f"Could not evaluate best checkpoint {path}: {e}")
        return None, f"{type(e).__name__}: {e}"


def _write_visualizations(
        vis_dir: Path,
        x_test: np.ndarray,
        y_test: np.ndarray,
        probs: np.ndarray,
        class_names: List[str],
        info: Dict[str, Any],
) -> Dict[str, Any]:
    """Render the four end-of-run figures and the classification report.

    Each figure is independent: a failure logs a warning and the others still
    render.

    Returns:
        ``{"files": [names written], "ece": float | None, "failed": [names]}``.
    """
    y_pred = np.argmax(probs, axis=-1)
    out: Dict[str, Any] = {"files": [], "ece": None, "failed": []}

    def _attempt(name: str, fn) -> Any:
        try:
            result = fn()
            out["files"].append(name)
            return result
        except Exception as e:  # noqa: BLE001 - a figure must not fail the run
            logger.warning(f"Visualization {name} failed: {e}")
            out["failed"].append(name)
            return None

    _attempt("confusion_matrix.png", lambda: plot_confusion_matrix(
        y_test, y_pred, class_names, vis_dir / "confusion_matrix.png"))
    report = _attempt("per_class_metrics.png", lambda: plot_per_class_metrics(
        y_test, y_pred, class_names, vis_dir / "per_class_metrics.png"))
    out["ece"] = _attempt("confidence_calibration.png", lambda: plot_calibration(
        y_test, probs, vis_dir / "confidence_calibration.png"))
    _attempt("misclassifications.png", lambda: plot_confident_errors(
        x_test, y_test, probs, vis_dir / "misclassifications.png",
        info["image_shape"], info["mean"], info["std"], class_names=class_names))

    if report is not None:
        try:
            with open(vis_dir / "classification_report.json", "w") as f:
                json.dump(report, f, indent=2, default=json_numpy_default)
            out["files"].append("classification_report.json")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"classification_report.json failed: {e}")
            out["failed"].append("classification_report.json")
    return out


def _read_analysis_status(run_dir: Path, model_name: str) -> Dict[str, Any]:
    """Read ``model_analysis/analysis_results.json`` back from disk.

    ``run_model_analysis`` swallows evaluation errors and logs "completed
    successfully" regardless, so the file is the only trustworthy source.

    Returns:
        ``{"status", "loss", "accuracy", "error", "n_samples"}`` for
        ``model_name``; ``{"status": "missing", ...}`` if the file, the key or
        the JSON is unusable.
    """
    path = run_dir / "model_analysis" / "analysis_results.json"
    missing: Dict[str, Any] = {"status": "missing", "loss": None, "accuracy": None,
                               "error": None, "path": str(path)}
    try:
        with open(path) as f:
            metrics = json.load(f)["model_metrics"][model_name]
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not read analyzer status from {path}: {e}")
        missing["error"] = f"{type(e).__name__}: {e}"
        return missing
    return {
        "status": metrics.get("status", "missing"),
        "loss": metrics.get("loss"),
        "accuracy": metrics.get("accuracy"),
        "error": metrics.get("error"),
        "path": str(path),
    }


# DECISION plan-2026-09-18T213948-68dcb72c/D-029: a reused --experiment-name is REFUSED,
# never merged, overwritten, deleted or auto-suffixed (``_r2``): results/ is gitignored,
# so an overwrite is unrecoverable and a silent rename changes the name the user asked
# for. Guard: test_a_reused_experiment_name_is_refused_and_the_first_run_is_byte_identical.
def _refuse_existing_run(run_dir: Path) -> None:
    """Raise ``FileExistsError`` when ``run_dir`` already holds a run's files.

    Nothing is written, overwritten or deleted, ever: results are unrecoverable
    (gitignored). A missing or empty directory, or one holding only unrelated
    files, is fine.

    Raises:
        FileExistsError: Naming ``run_dir`` and the files found, and telling the
            caller to choose a new ``--experiment-name``.
    """
    found = [name for name in RUN_ARTIFACT_NAMES if (run_dir / name).exists()]
    if found:
        raise FileExistsError(
            f"Experiment directory {run_dir} already holds a run ({', '.join(found)}). "
            "Nothing was written. Choose a new --experiment-name (or omit it for a "
            "timestamped name); existing results are never overwritten or deleted."
        )


def train_model(config: TrainingConfig) -> Dict[str, Any]:
    """Train PowerMLP, evaluate it, write every artifact and return the summary.

    The ``dl`` logger is also written to ``<run_dir>/run.log`` for the duration of
    the call (the handler is removed on return AND on an exception). Keras' own
    stdout lines (``ReduceLROnPlateau reducing...``, ``Epoch N: early stopping``)
    are not in it; the trainer derives and logs them after ``fit``.

    Args:
        config: A validated :class:`TrainingConfig`.

    Returns:
        The strict-JSON dict also written to ``<run_dir>/results_summary.json``
        (``status`` is ``"ok"``; see the keys assembled at the bottom).

    Raises:
        FileExistsError: If the experiment directory already holds a run (see
            :func:`_refuse_existing_run`); raised before anything is written.
        RuntimeError: If the initial loss before fitting is NaN or infinite, or
            (after ``results_summary.json`` with ``status: "diverged"`` was
            written) if any epoch ``loss`` / ``val_loss`` is non-finite.
    """
    logger.info("Starting PowerMLP training")
    resolved = Path(resolved_run_dir(config))
    _refuse_existing_run(resolved)
    set_seeds(config.seed)
    run_dir = Path(prepare_run_dir(config, output_dir=resolved)).resolve()
    vis_dir = run_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    run_log = logging.FileHandler(run_dir / RUN_LOG_NAME, mode="w")
    run_log.setFormatter(logging.Formatter(LOGGER_FORMAT))
    logger.addHandler(run_log)
    try:
        logger.info(f"Run directory: {run_dir}")

        train, val, test, info = prepare_data(
            config.dataset, config.validation_split, config.seed, config.input_scaling
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
            kernel_initializer=config.kernel_initializer,
            bias_initializer="zeros",
        )
        model.build((None, info["input_dim"]))
        model.compile(
            optimizer=create_optimizer(config.optimizer, config.learning_rate, config.weight_decay),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        # Read off the BUILT model, not the config: the input width is the shape the
        # model was built at, the rest are the layers' own widths.
        effective_units = [
            info["input_dim"],
            *[int(layer.units) for layer in model.hidden_layers],
            int(model.output_layer.units),
        ]
        params = int(model.count_params())
        logger.info(f"  Architecture: {effective_units}, k={config.k}")
        logger.info(f"  Dropout: {config.dropout_rate}, BatchNorm: {config.batch_normalization}")
        logger.info(
            f"  LR: {config.learning_rate}, Optimizer: {config.optimizer}, "
            f"Weight decay: {config.weight_decay}, Params: {params:,}"
        )

        initial_loss, initial_loss_ratio, init_scale_warning, initial_loss_mode = (
            _check_initial_loss(model, x_train[:SANITY_SAMPLES], y_train[:SANITY_SAMPLES])
        )
        # Keys every summary carries, whether the run finished or diverged.
        summary_head: Dict[str, Any] = {
            "run_dir": str(run_dir),
            "experiment_name": config.experiment_name,
            "dataset": config.dataset,
            "architecture": config.architecture,
            "effective_hidden_units": effective_units,
            "params": params,
            "k": config.k,
            "dropout_rate": config.dropout_rate,
            "batch_normalization": config.batch_normalization,
            "kernel_initializer": config.kernel_initializer,
            "input_scaling": config.input_scaling,
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "input_normalization": {
                "mean": info["mean"], "std": info["std"], "input_dim": info["input_dim"],
            },
            "epochs_requested": config.epochs,
            "monitor": MONITOR,
            "initial_loss_sanity_eval": {
                "loss": initial_loss, "n_samples": int(min(SANITY_SAMPLES, len(x_train))),
                "split": "train", "before_fit": True,
            },
            "initial_loss_ratio": initial_loss_ratio,
            "init_scale_warning": init_scale_warning,
            "initial_loss_mode": initial_loss_mode,
        }

        callbacks, _ = create_callbacks(
            model_name=config.experiment_name,
            results_dir_prefix="powermlp",
            run_dir=str(run_dir),
            monitor=MONITOR,
            patience=config.patience,
            use_lr_schedule=False,  # ReduceLROnPlateau, no external schedule
            include_terminate_on_nan=True,
            include_analyzer=config.epoch_analysis,
        )
        # Index 0: `lr` must be in `logs` before CSVLogger reads it.
        callbacks.insert(0, LearningRateLogger())
        # Last: it reads `logs['lr']`, written by LearningRateLogger above.
        dashboard = TrainingDashboardCallback(
            out_path=vis_dir / "training_dashboard.png",
            baseline_fn=lambda m: _untrained_metrics(m, x_val, y_val)[0],
            baseline_mode=initial_loss_mode,
            title=f"{config.experiment_name} (seed {config.seed})",
        )
        callbacks.append(dashboard)

        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        save_training_history_json(history, str(run_dir))
        hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        epochs_run = len(hist.get(MONITOR, []))
        non_finite = _non_finite_metrics(hist)

        # What Keras prints to stdout (never to the logger), derived from the history.
        lr_reduction_epochs = _lr_reduction_epochs(hist.get("lr", []))
        for epoch in lr_reduction_epochs:
            logger.info(
                f"ReduceLROnPlateau: learning rate {hist['lr'][epoch - 2]:.3g} -> "
                f"{hist['lr'][epoch - 1]:.3g} from epoch {epoch}"
            )

        if non_finite:
            # TerminateOnNaN ended the run: fewer epochs than requested is NOT an early
            # stop and no best weights were restored (review iteration 3, C1), so
            # ``stopped_early`` is unknown and the EarlyStopping line is not logged.
            message = (
                f"Training diverged: {non_finite} hold a non-finite or missing value after "
                f"{epochs_run} epoch(s); no evaluation, figures, analysis or final_model.keras "
                f"were produced. Initial-loss ratio was {initial_loss_ratio:.3g}. {REMEDY_HINT}"
            )
            logger.error(message)
            _write_summary(run_dir, {
                "status": STATUS_DIVERGED,
                **summary_head,
                "epochs_run": epochs_run,
                "stopped_early": None,
                "best_epoch": None,
                "non_finite_metrics": non_finite,
                "lr_reduction_epochs": lr_reduction_epochs,
                "history": hist,
                "epoch_times": list(dashboard.epoch_times),
                "notes": [
                    message,
                    "non-finite values are written as null (strict JSON)",
                    "`stopped_early` is null: the run was ended by a non-finite loss "
                    "(TerminateOnNaN), not by EarlyStopping, and no best weights were restored",
                ],
            })
            raise RuntimeError(message)

        stopped_early = epochs_run < config.epochs
        if stopped_early:
            logger.info(
                f"EarlyStopping: stopped after epoch {epochs_run} of {config.epochs} "
                f"(patience {config.patience} on {MONITOR}); the best weights were restored"
            )

        # Final (in-memory) weights vs the reloaded best checkpoint. Keras 3.8
        # EarlyStopping restores the best weights at every train end, so equality is
        # expected (D-011) and a difference is a bug signal.
        test_metrics_final = _evaluate(model, x_test, y_test)
        logger.info(f"Test results (final weights): {test_metrics_final}")
        test_metrics_best, best_load_error = _load_best_metrics(run_dir, x_test, y_test)
        if test_metrics_best is not None:
            logger.info(f"Test results (best_model.keras): {test_metrics_best}")
            diffs = {k: abs(test_metrics_final[k] - test_metrics_best[k])
                     for k in test_metrics_final if k in test_metrics_best}
            if any(d > WEIGHT_MISMATCH_TOLERANCE for d in diffs.values()):
                logger.warning(
                    f"Final weights and best_model.keras disagree on the test set: {diffs}. "
                    "EarlyStopping should have restored the best weights (D-011)."
                )

        final_path = run_dir / "final_model.keras"
        model.save(final_path)
        load_check: Optional[bool] = None
        try:
            sample = x_val[:LOAD_CHECK_SAMPLES]
            load_check = bool(validate_model_loading(
                str(final_path), sample, model.predict(sample, verbose=0)
            ))
        except Exception as e:  # noqa: BLE001 - log-only
            logger.warning(f"validate_model_loading raised: {e}")

        probs = model.predict(x_test, batch_size=1024, verbose=0)
        visualizations = _write_visualizations(
            vis_dir, x_test, y_test, probs,
            get_class_names(config.dataset, info["num_classes"]), info,
        )

        run_model_analysis(
            model, (x_test, y_test), history, config.experiment_name, str(run_dir)
        )
        analysis = _read_analysis_status(run_dir, config.experiment_name)
        logger.info(f"Analyzer status (read back from disk): {analysis['status']}")

        best_epoch = _best_epoch(hist)
        best_i, final_i = best_epoch - 1, epochs_run - 1
        val_keys = [k for k in hist if k.startswith("val_")]
        widths = ARCHITECTURE_HIDDEN_WIDTHS[config.architecture][config.dataset]
        notes: List[str] = [
            f"expected hidden widths {widths}, built {effective_units[1:-1]}: "
            f"{'match' if widths == effective_units[1:-1] else 'MISMATCH'}",
            f"initial loss {initial_loss:.4f} vs uniform ln(C)={np.log(info['num_classes']):.4f} "
            f"(ratio {initial_loss_ratio:.2f}, warn above {INITIAL_LOSS_WARN_FACTOR:g})",
            f"initial loss mode: {initial_loss_mode} "
            f"({'batch statistics, dropout on, weights restored afterwards' if initial_loss_mode == 'training' else 'model.evaluate, dropout off'})",
            f"the init-scale guard is a floor for hazards (warns above {INITIAL_LOSS_WARN_FACTOR:g}x "
            "ln(C)); a ratio between 3x and 10x raises no warning but is not healthy",
            "CSV `epoch` column is 0-based, `best_epoch` is 1-based "
            "(`best_epoch_csv_index` = `best_epoch` - 1)",
            "analyzer accuracy is on the first 1000 test samples only, not the full test set",
            "top-level `ece` is computed on the FULL test set; the analyzer's calibration "
            "numbers in model_analysis/ use the first 1000 test samples, so the two differ",
            "`test_metrics_final` describes the in-memory weights AFTER EarlyStopping restored "
            "the best-val_loss weights, so it equals `test_metrics_best` by construction; "
            "`final_val_metrics` is the LAST epoch's validation metrics",
            f"final weights vs best_model.keras: "
            f"{'not compared' if test_metrics_best is None else 'compared, see test_metrics_*'}",
        ]
        summary: Dict[str, Any] = {
            "status": STATUS_OK,
            **summary_head,
            "epochs_run": epochs_run,
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_epoch_csv_index": best_i,
            "lr_reduction_epochs": lr_reduction_epochs,
            "best_val_metrics": {k: hist[k][best_i] for k in val_keys},
            "final_val_metrics": {k: hist[k][final_i] for k in val_keys},
            "test_metrics_final": test_metrics_final,
            "test_metrics_best": test_metrics_best,
            "best_checkpoint_load_error": best_load_error,
            "epoch_times": list(dashboard.epoch_times),
            "ece": visualizations["ece"],
            "visualizations": visualizations,
            "model_loading_validated": load_check,
            "analyzer": analysis,
            "notes": notes,
        }
        return _write_summary(run_dir, summary)
    finally:
        logger.removeHandler(run_log)
        run_log.close()


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
