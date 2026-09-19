"""
Shared orchestrator of the ConvNeXt V1 / V2 classification trainers.

``train_convnext_v1.py`` and ``train_convnext_v2.py`` are thin wrappers over this
module: everything except the model family (its factory and its variant table) is
common, so a defect is fixed once. The family is a :class:`TrainingConfig` field
(``model_family``, ``"v1"`` or ``"v2"``) resolved through :data:`MODEL_FAMILIES`;
the model is always built through ``create_convnext_v1`` / ``create_convnext_v2``
(depths and dims come from the variant table, never from kwargs).

Data (mnist / cifar10 / cifar100 through ``train.common.load_dataset``, i.e. the
local Keras cache): a seeded ``validation_split`` fraction of the TRAIN set is held
out for early stopping and checkpoint selection, the TEST set is used only for the
final report, per-channel mean / std are computed on the train split and applied to
all three, and the train pipeline (tf.data) applies a pad-4 random crop (plus a
horizontal flip for the CIFAR datasets, never for MNIST). ``--max-samples`` caps the
train pool and the test set (smoke runs and tests). ``imagenet`` is refused with an
explicit error: ``tfds`` ``imagenet2012`` is not available here and the full-test-set
figures cannot be built for it.

Optimization: ONE optimizer, ``AdamW(learning_rate=<schedule>, weight_decay=wd,
clipnorm=1.0)``, identical for V1 and V2 and never combined with an L2 regularizer
(decoupled decay is applied once). The schedule is a cosine over the WHOLE run
(``steps_per_epoch`` is passed, without it the cosine collapses to its floor within
``epochs`` optimizer steps), optionally preceded by a linear warmup.

Health: the untrained model is evaluated on the validation split before ``fit``
(ConvNeXt is LayerNorm-only, so a plain ``evaluate`` is the true epoch-0 loss); the
ratio to ``ln(num_classes)`` is recorded and a WARNING is logged above
``INITIAL_LOSS_WARN_FACTOR``. A run whose ``loss`` or ``val_loss`` turns non-finite is
``status: "diverged"``: a reduced strict-JSON summary is written, then a
``RuntimeError`` is raised.

Every run writes to ``<repo>/results/<experiment_name>/`` (a relative ``--output-dir``
is anchored at the repo root): ``config.json``, ``training_log.csv`` (with ``lr``),
``training_history.json``, ``best_model.keras``, ``final_model.keras``,
``results_summary.json``, ``run.log``, ``visualizations/`` and ``model_analysis/``
(plus ``epoch_analysis/`` only with ``--epoch-analysis``). The CSV ``epoch`` column
is 0-based; ``best_epoch`` in the summary is 1-based. A reused experiment name is
refused, never merged or overwritten.
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.models.vision.convnext.convnext_v1 import ConvNeXtV1, create_convnext_v1
from dl_techniques.models.vision.convnext.convnext_v2 import ConvNeXtV2, create_convnext_v2
from dl_techniques.utils.logger import logger

from train.common import (
    attach_run_log,
    create_callbacks,
    create_learning_rate_schedule,
    default_experiment_name,
    get_class_names,
    json_numpy_default,
    load_dataset,
    log_gpu_peak_memory,
    prepare_run_dir,
    refuse_existing_run,
    resolved_run_dir,
    run_model_analysis,
    save_training_history_json,
    set_seeds,
    setup_gpu,
    validate_model_loading,
    write_summary_json,
)
from train.common.callbacks import LearningRateLogger, best_checkpoint_path
from train.common.classification_viz import (
    TrainingDashboardCallback,
    plot_calibration,
    plot_confident_errors,
    plot_confusion_matrix,
    plot_per_class_metrics,
)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

DATASETS: Tuple[str, ...] = ("mnist", "cifar10", "cifar100")
LR_SCHEDULES: Tuple[str, ...] = ("cosine", "exponential", "constant")
STOCHASTIC_MODES: Tuple[str, ...] = ("depth", "gradient")

IMAGENET_REFUSAL = (
    "dataset 'imagenet' is not supported by the normalized ConvNeXt trainers: the tfds "
    "'imagenet2012' data is not available here and the full-test-set figures (confusion "
    "matrix, calibration, confident errors) cannot be built for a streaming dataset. "
    f"Choose one of {DATASETS}."
)

# Inherited from the pre-normalization CLI (the old scripts' per-dataset table), NOT
# measured here. Filled into ``TrainingConfig`` fields left at ``None``.
REGULARIZATION_DEFAULTS: Dict[str, Dict[str, float]] = {
    "mnist": {"drop_path_rate": 0.1, "dropout_rate": 0.1},
    "cifar10": {"drop_path_rate": 0.1, "dropout_rate": 0.1},
    "cifar100": {"drop_path_rate": 0.2, "dropout_rate": 0.2},
}

# The datasets whose train pipeline also flips horizontally (a flipped digit is a
# different digit, so MNIST gets the random crop only).
FLIP_DATASETS: Tuple[str, ...] = ("cifar10", "cifar100")
# Random-crop padding in pixels. Zeros are added AFTER standardization, so the border
# is the per-channel mean colour, not black.
AUGMENT_PAD = 4

# Global-norm-free per-variable clip applied by the optimizer (identical in V1 and V2).
GRADIENT_CLIP_NORM = 1.0

# The metric that drives early stopping, the best checkpoint and best_epoch.
MONITOR = "val_loss"

# Rows per pass of every ``evaluate`` / ``predict`` / eval pipeline.
EVAL_BATCH_SIZE = 256
# Rows fed to the round-trip check of ``final_model.keras``.
LOAD_CHECK_SAMPLES = 64
# The reloaded best checkpoint should reproduce the in-memory best weights exactly.
WEIGHT_MISMATCH_TOLERANCE = 1e-6

# ``_check_initial_loss`` warns when loss / ln(num_classes) is STRICTLY above this. A
# hazard floor chosen independently of the PowerMLP trainer's equal value.
INITIAL_LOSS_WARN_FACTOR = 10.0

# ``results_summary.json["status"]``.
STATUS_OK = "ok"
STATUS_DIVERGED = "diverged"

# Classes for which the top-5 metric is meaningful (10-class tasks report top-1 only).
TOP_K_MIN_CLASSES = 11


@dataclass(frozen=True)
class ModelFamily:
    """A ConvNeXt family: how to build it and which variants it offers.

    Attributes:
        label: Human name used in logs and titles.
        factory: ``create_convnext_v1`` / ``create_convnext_v2``.
        variants: The keys of the class's ``MODEL_VARIANTS`` table.
    """

    label: str
    factory: Callable[..., keras.Model]
    variants: Tuple[str, ...]


MODEL_FAMILIES: Dict[str, ModelFamily] = {
    "v1": ModelFamily("ConvNeXt V1", create_convnext_v1, tuple(ConvNeXtV1.MODEL_VARIANTS)),
    "v2": ModelFamily("ConvNeXt V2", create_convnext_v2, tuple(ConvNeXtV2.MODEL_VARIANTS)),
}


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration of one ConvNeXt classification run.

    Every field is read by the trainer (``tests/test_train/
    test_config_fields_are_live.py``). ``drop_path_rate`` and ``dropout_rate`` left at
    ``None`` are replaced in ``__post_init__`` from :data:`REGULARIZATION_DEFAULTS`, so
    after construction they are always floats and ``config.json`` records what was
    used. ``experiment_name`` defaults to
    ``convnext_<family>_<dataset>_<variant>_<timestamp>``.
    """

    # Model
    model_family: str = "v1"
    variant: str = "cifar10"
    kernel_size: int = 7
    strides: int = 4
    drop_path_rate: Optional[float] = None
    stochastic_mode: str = "depth"
    dropout_rate: Optional[float] = None
    use_gamma: bool = True

    # Data
    dataset: str = "cifar10"
    validation_split: float = 0.1
    max_samples: Optional[int] = None

    # Training (defaults are the pre-normalization CLI's values, not measured here)
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_schedule: str = "cosine"
    warmup_epochs: int = 0
    patience: int = 50
    seed: int = 42

    # Monitoring / output
    epoch_analysis: bool = False
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate ranges, fill the per-dataset defaults and derive the name.

        Raises:
            ValueError: If any field is outside its supported range.
        """
        if self.model_family not in MODEL_FAMILIES:
            raise ValueError(
                f"model_family must be one of {tuple(MODEL_FAMILIES)}, got {self.model_family!r}"
            )
        family = MODEL_FAMILIES[self.model_family]
        if self.dataset == "imagenet":
            raise ValueError(IMAGENET_REFUSAL)
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}, got {self.dataset!r}")
        if self.variant not in family.variants:
            raise ValueError(
                f"variant must be one of {family.variants} for {family.label}, got {self.variant!r}"
            )
        if self.stochastic_mode not in STOCHASTIC_MODES:
            raise ValueError(
                f"stochastic_mode must be one of {STOCHASTIC_MODES}, got {self.stochastic_mode!r}"
            )
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}, got {self.lr_schedule!r}")
        defaults = REGULARIZATION_DEFAULTS[self.dataset]
        if self.drop_path_rate is None:
            self.drop_path_rate = defaults["drop_path_rate"]
        if self.dropout_rate is None:
            self.dropout_rate = defaults["dropout_rate"]
        if not 0.0 <= self.drop_path_rate < 1.0:
            raise ValueError(f"drop_path_rate must be in [0, 1), got {self.drop_path_rate}")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")
        if self.kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {self.kernel_size}")
        if self.strides < 1:
            raise ValueError(f"strides must be >= 1, got {self.strides}")
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
        if not 0.0 < self.validation_split < 1.0:
            raise ValueError(
                f"validation_split must be strictly inside (0, 1), got {self.validation_split}. "
                "0 would silently validate on the test set."
            )
        if self.max_samples is not None and self.max_samples < 2:
            raise ValueError(f"max_samples must be >= 2 (or None), got {self.max_samples}")
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {self.warmup_epochs}")
        if self.warmup_epochs > 0:
            # create_learning_rate_schedule builds a warmup ONLY for the cosine schedule;
            # accepting the flag elsewhere would be a knob that does nothing.
            if self.lr_schedule != "cosine":
                raise ValueError(
                    f"warmup_epochs > 0 needs lr_schedule 'cosine', got {self.lr_schedule!r}"
                )
            if self.warmup_epochs >= self.epochs:
                raise ValueError(
                    f"warmup_epochs ({self.warmup_epochs}) must be smaller than epochs ({self.epochs})"
                )
        if self.experiment_name is None:
            self.experiment_name = default_experiment_name(
                f"convnext_{self.model_family}", self.dataset, self.variant
            )


# ---------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------

def _dataset_arg(value: str) -> str:
    """argparse ``type`` for ``--dataset``: turn ``imagenet`` into the explicit refusal."""
    if value == "imagenet":
        raise argparse.ArgumentTypeError(IMAGENET_REFUSAL)
    return value


def _build_parser(model_family: str) -> argparse.ArgumentParser:
    """Build the unparsed trainer parser of one model family.

    Defaults are read off a default :class:`TrainingConfig`, so parser and config
    cannot drift. ``--drop-path-rate`` / ``--dropout-rate`` / ``--experiment-name``
    default to ``None`` (resolved by the config, per dataset / at construction time).
    ``--gpu`` is not a config field.

    Args:
        model_family: A key of :data:`MODEL_FAMILIES`.

    Returns:
        An ``argparse.ArgumentParser`` with every flag of the trainer.

    Raises:
        KeyError: If ``model_family`` is unknown.
    """
    family = MODEL_FAMILIES[model_family]
    defaults = TrainingConfig(model_family=model_family)
    parser = argparse.ArgumentParser(
        description=f"Train {family.label} on MNIST, CIFAR-10 or CIFAR-100.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--dataset", type=_dataset_arg, default=defaults.dataset, choices=DATASETS,
                      help="Dataset to train on ('imagenet' is refused).")
    data.add_argument("--validation-split", type=float, default=defaults.validation_split,
                      help="Fraction of the train set held out (seeded shuffle) for early "
                           "stopping and checkpoint selection, strictly inside (0, 1); the "
                           "test set is only used for the final report.")
    data.add_argument("--max-samples", type=int, default=defaults.max_samples,
                      help="Cap the train pool and the test set at this many samples "
                           "(smoke runs); default: everything.")

    model = parser.add_argument_group("model")
    model.add_argument("--variant", type=str, default=defaults.variant, choices=family.variants,
                       help="Model variant (sets depths and dims).")
    model.add_argument("--kernel-size", type=int, default=defaults.kernel_size,
                       help="Depthwise convolution kernel size.")
    model.add_argument("--strides", type=int, default=defaults.strides,
                       help="Stem patch size and downsampling stride. 4 gives 8,2,1,1 feature "
                            "maps on 32x32 inputs (see stage_feature_map_sizes in the summary).")
    model.add_argument("--drop-path-rate", type=float, default=None,
                       help="Maximum stochastic-depth rate (default per dataset: 0.1 for mnist "
                            "and cifar10, 0.2 for cifar100).")
    model.add_argument("--stochastic-mode", type=str, default=defaults.stochastic_mode,
                       choices=STOCHASTIC_MODES,
                       help="depth = StochasticDepth, gradient = StochasticGradient.")
    model.add_argument("--dropout-rate", type=float, default=None,
                       help="Dropout rate inside each block (default per dataset: 0.1 for mnist "
                            "and cifar10, 0.2 for cifar100).")
    model.add_argument("--use-gamma", action=argparse.BooleanOptionalAction,
                       default=defaults.use_gamma,
                       help="Learnable per-channel layer scale in each block.")

    train = parser.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=defaults.epochs,
                       help="Maximum number of training epochs (the cosine spans this many).")
    train.add_argument("--batch-size", type=int, default=defaults.batch_size,
                       help="Training batch size.")
    train.add_argument("--learning-rate", type=float, default=defaults.learning_rate,
                       help="Peak learning rate.")
    train.add_argument("--weight-decay", type=float, default=defaults.weight_decay,
                       help="Decoupled AdamW weight decay (never also an L2 regularizer).")
    train.add_argument("--lr-schedule", type=str, default=defaults.lr_schedule,
                       choices=LR_SCHEDULES,
                       help="Learning-rate schedule over the whole run; 'constant' adds "
                            "ReduceLROnPlateau.")
    train.add_argument("--warmup-epochs", type=int, default=defaults.warmup_epochs,
                       help="Linear warmup epochs before the cosine (cosine schedule only).")
    train.add_argument("--patience", type=int, default=defaults.patience,
                       help="Early-stopping patience in epochs on val_loss.")
    train.add_argument("--seed", type=int, default=defaults.seed,
                       help="Seed for weights, shuffling, augmentation and the splits.")
    train.add_argument("--epoch-analysis", action="store_true", default=defaults.epoch_analysis,
                       help="Run the per-epoch ModelAnalyzer callback (off by default; the "
                            "final analysis always runs).")

    out = parser.add_argument_group("output")
    out.add_argument("--output-dir", type=str, default=defaults.output_dir,
                     help="Output root; a relative path is anchored at the repo root.")
    out.add_argument("--experiment-name", type=str, default=None,
                     help=f"Run directory name (default: convnext_{model_family}_<dataset>_"
                          "<variant>_<timestamp>). A name that already holds a run is refused.")
    out.add_argument("--gpu", type=int, default=None,
                     help="GPU device index; sets CUDA_VISIBLE_DEVICES before TensorFlow first "
                          "enumerates devices (measured effective here, and it overrides an "
                          "exported CUDA_VISIBLE_DEVICES). Default: use the environment. The "
                          "summary records the device TF actually used (gpu_name).")
    return parser


def parse_arguments(argv: Optional[Sequence[str]], model_family: str) -> argparse.Namespace:
    """Parse the command line. ``--help`` exits here, before anything expensive.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv[1:]``.
        model_family: A key of :data:`MODEL_FAMILIES`.

    Returns:
        The parsed ``argparse.Namespace``.
    """
    return _build_parser(model_family).parse_args(argv)


def config_from_args(args: argparse.Namespace, model_family: str) -> TrainingConfig:
    """Build a :class:`TrainingConfig` from a parsed namespace.

    ``--gpu`` is deliberately not a config field: it is consumed once by
    ``setup_gpu`` in :func:`main`.

    Args:
        args: Namespace returned by :func:`parse_arguments`.
        model_family: The family the parser was built for.

    Returns:
        The validated config.

    Raises:
        ValueError: If a value is outside its supported range.
    """
    # Every config field except the family comes from the namespace of the same name, so a
    # field with no flag raises AttributeError here instead of silently keeping its default.
    values = {f.name: getattr(args, f.name) for f in fields(TrainingConfig) if f.name != "model_family"}
    return TrainingConfig(model_family=model_family, **values)


# ---------------------------------------------------------------------
# Geometry and devices
# ---------------------------------------------------------------------

def stage_feature_map_sizes(
        input_hw: Tuple[int, int], depths: Sequence[int], strides: int
) -> List[Tuple[int, int]]:
    """Spatial size of the feature map each stage runs on.

    The stem is a ``strides x strides`` convolution at stride ``strides`` with
    ``"valid"`` padding (``"same"`` when ``strides == 1``), giving ``n // strides``;
    every downsample between stages is ``"same"``, giving ``ceil(n / strides)``. This
    is why the default ``strides=4`` on 32x32 inputs gives 8, 2, 1, 1 and later stages
    run on a single pixel.

    Args:
        input_hw: ``(height, width)`` of the input image.
        depths: Blocks per stage; only its length (the stage count) is used.
        strides: The model's ``strides`` (stem patch size and downsample stride).

    Returns:
        One ``(height, width)`` per stage, stage 0 first.
    """
    def stem(n: int) -> int:
        return n if strides == 1 else n // strides

    sizes = [(stem(input_hw[0]), stem(input_hw[1]))]
    for _ in range(len(depths) - 1):
        h, w = sizes[-1]
        sizes.append((math.ceil(h / strides), math.ceil(w / strides)))
    return sizes


def describe_devices() -> Dict[str, Any]:
    """Which GPU(s) TensorFlow sees in this process, for the run log and the summary.

    The first call enumerates devices, after which a changed ``CUDA_VISIBLE_DEVICES``
    no longer selects a different one; ``gpu_name`` is what TF actually sees, while
    ``cuda_visible_devices`` is only the environment value at call time.

    Returns:
        ``{"cuda_visible_devices", "tf_visible_devices", "gpu_names", "gpu_name"}``;
        ``gpu_name`` is the first visible GPU's name or ``None`` on a CPU-only process.
    """
    gpus = tf.config.list_physical_devices("GPU")
    names = [
        str(tf.config.experimental.get_device_details(gpu).get("device_name", gpu.name))
        for gpu in gpus
    ]
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf_visible_devices": [gpu.name for gpu in gpus],
        "gpu_names": names,
        "gpu_name": names[0] if names else None,
    }


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SplitData:
    """Standardized train / validation / test arrays plus what figures need.

    Attributes:
        x_train, y_train, x_val, y_val, x_test, y_test: ``x`` is float32 NHWC
            standardized with ``mean`` / ``std``, ``y`` is int32 ``(N,)``.
        mean, std: Per-channel statistics ``(C,)`` of the TRAIN split (float32).
        input_shape: ``(H, W, C)``.
        num_classes: Number of classes.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    input_shape: Tuple[int, int, int]
    num_classes: int


# The validation split is cut from the TRAIN set; the TEST set never influences early
# stopping or the best checkpoint (decisions.md D-004). The old scripts passed the test set
# as ``validation_data`` and so leaked selection into the reported number. Do NOT pass the
# test set to ``fit`` "for a bigger validation set".
def prepare_data(config: TrainingConfig) -> SplitData:
    """Load, split and standardize a dataset from the local Keras cache.

    ``load_dataset`` reads ``~/.keras/datasets`` (a missing file makes Keras try a
    download and fail loudly; nothing here retries). The validation split is a seeded
    permutation of the train set; with ``max_samples`` the train pool and the test set
    are first cut to a seeded subset of that size.

    Args:
        config: A validated :class:`TrainingConfig`.

    Returns:
        The :class:`SplitData`.

    Raises:
        ValueError: If the validation split would hold out zero samples.
    """
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = load_dataset(config.dataset)
    train_order = np.random.default_rng(config.seed).permutation(len(x_train))
    test_order = np.random.default_rng([config.seed, 1]).permutation(len(x_test))
    if config.max_samples is not None:
        train_order = train_order[:config.max_samples]
        test_order = test_order[:config.max_samples]
    val_size = int(len(train_order) * config.validation_split)
    if val_size == 0:
        raise ValueError(
            f"validation_split={config.validation_split} holds out 0 of {len(train_order)} samples"
        )
    val_idx, train_idx = train_order[:val_size], train_order[val_size:]

    x_fit = x_train[train_idx]
    mean = x_fit.mean(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    std = x_fit.std(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)

    def standardize(x: np.ndarray) -> np.ndarray:
        return ((x - mean) / std).astype(np.float32)

    def labels(y: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return np.asarray(y).reshape(-1)[idx].astype(np.int32)

    data = SplitData(
        x_train=standardize(x_fit), y_train=labels(y_train, train_idx),
        x_val=standardize(x_train[val_idx]), y_val=labels(y_train, val_idx),
        x_test=standardize(x_test[test_order]), y_test=labels(y_test, test_order),
        mean=mean, std=std,
        input_shape=tuple(int(d) for d in input_shape), num_classes=int(num_classes),
    )
    logger.info(
        f"ConvNeXt data - train {data.x_train.shape}, val {data.x_val.shape}, "
        f"test {data.x_test.shape}; mean {mean.tolist()}, std {std.tolist()}"
    )
    return data


def make_train_dataset(
        x: np.ndarray, y: np.ndarray, batch_size: int, seed: int, flip: bool
) -> "tf.data.Dataset":
    """The augmented, shuffled, batched train pipeline (tf.data).

    Each epoch reshuffles the whole pool. Every image gets a pad-``AUGMENT_PAD``
    random crop back to its own size, and a random horizontal flip when ``flip``.
    ``from_tensor_slices`` embeds the arrays in the graph, so the pool must stay
    under TensorFlow's 2 GB constant limit (true for MNIST and CIFAR).

    Args:
        x: Standardized NHWC float32 images.
        y: Integer labels.
        batch_size: Batch size; the last batch is kept (no ``drop_remainder``).
        seed: Shuffle seed.
        flip: Whether to flip horizontally.

    Returns:
        A ``tf.data.Dataset`` of ``(images, labels)`` batches.
    """
    height, width, channels = (int(d) for d in x.shape[1:])

    def augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if flip:
            image = tf.image.random_flip_left_right(image)
        image = tf.pad(image, [[AUGMENT_PAD, AUGMENT_PAD], [AUGMENT_PAD, AUGMENT_PAD], [0, 0]])
        return tf.image.random_crop(image, (height, width, channels)), label

    return (
        tf.data.Dataset.from_tensor_slices((x, y))
        .shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def make_eval_dataset(x: np.ndarray, y: np.ndarray) -> "tf.data.Dataset":
    """The un-augmented, un-shuffled, batched pipeline used as ``validation_data``.

    ``fit`` evaluates a numpy ``validation_data`` at Keras' small default batch when the
    train data is a dataset, so the validation split is batched here explicitly.
    """
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(EVAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------

# DECISION plan-2026-09-19T040641-db6932ec/D-013: ``steps_per_epoch`` MUST reach the
# schedule. Without it ``CosineDecay`` counts optimizer STEPS against ``decay_steps =
# epochs`` and reaches its floor (1% of the peak) after ``epochs`` batches: the old
# trainers trained at about 1e-5 from the second epoch on. Warmup is engaged only through
# ``warmup_steps`` (``warmup_epochs`` of the library function is a reserved no-op). Guard:
# the LR test in tests/test_train/test_convnext (step 4).
def build_lr_schedule(config: TrainingConfig, steps_per_epoch: int) -> Any:
    """The learning rate handed to the optimizer: a schedule object or a plain float.

    Args:
        config: A validated :class:`TrainingConfig`.
        steps_per_epoch: Optimizer steps in one epoch of the train pipeline.

    Returns:
        A Keras ``LearningRateSchedule`` spanning ALL ``config.epochs`` epochs
        (``'constant'`` returns the bare float).
    """
    return create_learning_rate_schedule(
        initial_lr=config.learning_rate,
        schedule_type=config.lr_schedule,
        total_epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=config.warmup_epochs * steps_per_epoch,
    )


def build_metrics(num_classes: int) -> List[keras.metrics.Metric]:
    """Accuracy, plus top-5 accuracy for tasks with more than 10 classes.

    The top-5 metric is a metric OBJECT: the string alias ``"top_5_accuracy"`` is not
    resolvable by Keras and crashes ``compile`` on a 100-class task.
    """
    metrics: List[keras.metrics.Metric] = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    if num_classes >= TOP_K_MIN_CLASSES:
        metrics.append(keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy"))
    return metrics


class _LastEpochWeights(keras.callbacks.Callback):
    """Keeps a copy of the weights at the end of the most recent completed epoch.

    ``create_callbacks`` builds ``EarlyStopping(restore_best_weights=True)`` and Keras
    3.8 restores the best weights at EVERY train end, so after ``fit`` the in-memory
    model is the best one and the true last-epoch weights are gone unless captured.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weights: Optional[List[np.ndarray]] = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self.weights = self.model.get_weights()


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------

def _evaluate(model: keras.Model, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """``model.evaluate`` as a plain ``{metric: float}`` dict."""
    metrics = model.evaluate(x, y, batch_size=EVAL_BATCH_SIZE, verbose=0, return_dict=True)
    return {k: float(v) for k, v in metrics.items()}


def _check_initial_loss(model: keras.Model, x: np.ndarray, y: np.ndarray, num_classes: int
                        ) -> Tuple[Dict[str, float], float, bool]:
    """Evaluate the untrained model; refuse a non-finite loss, warn on a huge one.

    The single epoch-0 measurement: both the summary's initial-loss ratio and the
    dashboard's baseline marker come from THIS result, so the two cannot disagree. A
    LayerNorm-only model has no training-mode statistics, so a plain inference-mode
    ``evaluate`` is the true loss of the first training step's starting point.

    Args:
        model: Built and compiled model.
        x: Validation inputs.
        y: Validation labels.
        num_classes: Class count; the uniform-prediction loss is ``ln(num_classes)``.

    Returns:
        ``(metrics, ratio, warned)``: the evaluate dict, ``loss / ln(num_classes)`` and
        whether the ratio is STRICTLY above :data:`INITIAL_LOSS_WARN_FACTOR`.

    Raises:
        RuntimeError: If the loss is NaN or infinite (raising, not returning, so no
            empty run directory is left behind as a "finished" run).
    """
    metrics = _evaluate(model, x, y)
    loss = metrics["loss"]
    uniform_loss = float(np.log(num_classes))
    logger.info(
        f"Untrained evaluate BEFORE fit: val loss {loss:.4f} on {len(x)} samples "
        f"(uniform-prediction loss would be {uniform_loss:.4f})"
    )
    if not np.isfinite(loss):
        raise RuntimeError(
            f"Initial validation loss is {loss}: the model diverges before training. Lower "
            "--drop-path-rate / --dropout-rate or check --strides and --kernel-size."
        )
    ratio = loss / uniform_loss
    warned = ratio > INITIAL_LOSS_WARN_FACTOR
    if warned:
        logger.warning(
            f"Initial loss {loss:.4f} is {ratio:.1f}x ln(num_classes)={uniform_loss:.4f} "
            f"(warn above {INITIAL_LOSS_WARN_FACTOR:g}x): the initial logit scale is far too "
            "large and the early epochs will be spent recovering."
        )
    return metrics, ratio, warned


def _best_epoch(history: Dict[str, List[float]]) -> int:
    """1-based epoch with the lowest ``MONITOR`` value.

    Raises:
        ValueError: If any monitored value is NaN or infinite (``np.argmin`` of a list
            holding a NaN returns the NaN's index).
    """
    values = np.asarray(history[MONITOR], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"cannot pick a best epoch: {MONITOR} has non-finite values {values.tolist()}")
    return int(np.argmin(values)) + 1


def _non_finite_metrics(history: Dict[str, List[float]]) -> List[str]:
    """Names among ``loss`` and ``MONITOR`` that are empty or hold a NaN / infinity."""
    return [
        key for key in dict.fromkeys(("loss", MONITOR))
        if not history.get(key) or not np.all(np.isfinite(history[key]))
    ]


def _load_best_metrics(
        run_dir: Path, x: np.ndarray, y: np.ndarray
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """Evaluate the reloaded ``best_model.keras``.

    Returns:
        ``(metrics, None)`` on success, ``(None, error_text)`` if the checkpoint is
        missing or does not load.
    """
    path = best_checkpoint_path(str(run_dir))
    try:
        return _evaluate(keras.models.load_model(path), x, y), None
    except Exception as e:  # noqa: BLE001 - reported, not fatal
        logger.warning(f"Could not evaluate best checkpoint {path}: {e}")
        return None, f"{type(e).__name__}: {e}"


def _check_best_checkpoint(
        model: keras.Model, run_dir: Path, data: SplitData
) -> Tuple[Optional[Dict[str, float]], Optional[str], Optional[float]]:
    """Test metrics of the reloaded ``best_model.keras`` and its gap to ``model``.

    ``model`` must hold the best weights (true after ``fit``: EarlyStopping restores
    them). The reloaded checkpoint must reproduce them, so a gap above
    :data:`WEIGHT_MISMATCH_TOLERANCE` is logged as a bug signal.

    Returns:
        ``(metrics, load_error, max_abs_diff)``; ``metrics`` and ``max_abs_diff`` are
        ``None`` when the checkpoint is missing or does not load.
    """
    reloaded, load_error = _load_best_metrics(run_dir, data.x_test, data.y_test)
    if reloaded is None:
        return None, load_error, None
    logger.info(f"Test results (best_model.keras): {reloaded}")
    in_memory = _evaluate(model, data.x_test, data.y_test)
    gap = max(abs(in_memory[k] - reloaded[k]) for k in in_memory if k in reloaded)
    if gap > WEIGHT_MISMATCH_TOLERANCE:
        logger.warning(
            f"best_model.keras and the in-memory best weights disagree on the test set by {gap:.3g}"
        )
    return reloaded, None, gap


def _write_visualizations(
        vis_dir: Path, data: SplitData, probs: np.ndarray, class_names: List[str]
) -> Dict[str, Any]:
    """Render the four end-of-run figures and the classification report.

    ``probs`` MUST be probabilities (softmax of the logits): calibration and confident
    errors read them as such. Each figure is independent; a failure logs a warning and
    the others still render.

    Args:
        vis_dir: Existing output directory.
        data: The split data (test images and labels, standardization statistics).
        probs: Class probabilities ``(N_test, C)``.
        class_names: One name per class.

    Returns:
        ``{"files": [names written], "ece": float | None, "failed": [names]}``.
    """
    y_pred = np.argmax(probs, axis=-1)
    out: Dict[str, Any] = {"files": [], "ece": None, "failed": []}

    def _attempt(name: str, fn: Callable[[], Any]) -> Any:
        try:
            result = fn()
            out["files"].append(name)
            return result
        except Exception as e:  # noqa: BLE001 - a figure must not fail the run
            logger.warning(f"Visualization {name} failed: {e}")
            out["failed"].append(name)
            return None

    _attempt("confusion_matrix.png", lambda: plot_confusion_matrix(
        data.y_test, y_pred, class_names, vis_dir / "confusion_matrix.png"))
    report = _attempt("per_class_metrics.png", lambda: plot_per_class_metrics(
        data.y_test, y_pred, class_names, vis_dir / "per_class_metrics.png"))
    out["ece"] = _attempt("confidence_calibration.png", lambda: plot_calibration(
        data.y_test, probs, vis_dir / "confidence_calibration.png"))
    _attempt("misclassifications.png", lambda: plot_confident_errors(
        data.x_test, data.y_test, probs, vis_dir / "misclassifications.png",
        mean=data.mean, std=data.std, class_names=class_names))

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

    ``run_model_analysis`` swallows evaluation errors and logs "completed successfully"
    regardless, so the file is the only trustworthy source.

    Returns:
        ``{"status", "loss", "accuracy", "error", "path"}`` for ``model_name``;
        ``{"status": "missing", ...}`` if the file, the key or the JSON is unusable.
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


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def _summary_head(
        config: TrainingConfig, run_dir: Path, data: SplitData, model: keras.Model,
        geometry: List[Tuple[int, int]], steps_per_epoch: int, params: int,
        baseline: Dict[str, float], initial_loss_ratio: float, init_scale_warning: bool,
        devices: Dict[str, Any],
) -> Dict[str, Any]:
    """Keys every summary carries, whether the run finished or diverged."""
    return {
        "run_dir": str(run_dir),
        "experiment_name": config.experiment_name,
        "model_family": config.model_family,
        "dataset": config.dataset,
        "variant": config.variant,
        "params": params,
        "depths": list(model.depths),
        "dims": list(model.dims),
        "strides": config.strides,
        "kernel_size": config.kernel_size,
        "drop_path_rate": config.drop_path_rate,
        "stochastic_mode": config.stochastic_mode,
        "dropout_rate": config.dropout_rate,
        "use_gamma": config.use_gamma,
        "input_shape": list(data.input_shape),
        "num_classes": data.num_classes,
        "stage_feature_map_sizes": [list(size) for size in geometry],
        "optimizer": "AdamW",
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "learning_rate": config.learning_rate,
        "lr_schedule": config.lr_schedule,
        "warmup_epochs": config.warmup_epochs,
        "steps_per_epoch": steps_per_epoch,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "validation_split": config.validation_split,
        "max_samples": config.max_samples,
        "n_train": int(len(data.x_train)),
        "n_val": int(len(data.x_val)),
        "n_test": int(len(data.x_test)),
        "input_normalization": {"mean": data.mean, "std": data.std},
        "epochs_requested": config.epochs,
        "monitor": MONITOR,
        "initial_loss_sanity_eval": {
            "loss": baseline["loss"], "n_samples": int(len(data.x_val)),
            "split": "val", "before_fit": True,
        },
        "initial_loss_ratio": initial_loss_ratio,
        "init_scale_warning": init_scale_warning,
        "gpu_name": devices["gpu_name"],
        "tf_visible_devices": devices["tf_visible_devices"],
        "cuda_visible_devices": devices["cuda_visible_devices"],
    }


def train(config: TrainingConfig) -> Dict[str, Any]:
    """Train a ConvNeXt, evaluate it, write every artifact and return the summary.

    The ``dl`` logger is also written to ``<run_dir>/run.log`` for the duration of the
    call (the handler is removed on return AND on an exception).

    ``test_metrics_best`` is the reloaded ``best_model.keras`` on the test set (the
    checkpoint round trip), ``test_metrics_final`` the LAST epoch's weights, which is
    what ``final_model.keras`` holds. The figures and ``model_analysis/`` describe the
    best (selected) weights. The two are the same model when the last epoch is the
    best one (``final_is_best``).

    Args:
        config: A validated :class:`TrainingConfig`.

    Returns:
        The strict-JSON dict also written to ``<run_dir>/results_summary.json``
        (``status`` is ``"ok"``).

    Raises:
        FileExistsError: If the experiment directory already holds a run; raised before
            anything is written.
        RuntimeError: If the initial validation loss is non-finite, or (after
            ``results_summary.json`` with ``status: "diverged"`` was written) if any
            epoch ``loss`` / ``val_loss`` is non-finite.
    """
    family = MODEL_FAMILIES[config.model_family]
    logger.info(f"Starting {family.label} training")
    resolved = Path(resolved_run_dir(config))
    refuse_existing_run(resolved)
    set_seeds(config.seed)
    run_dir = Path(prepare_run_dir(config, output_dir=resolved)).resolve()
    vis_dir = run_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    with attach_run_log(run_dir):
        logger.info(f"Run directory: {run_dir}")
        devices = describe_devices()
        logger.info(
            f"Devices: CUDA_VISIBLE_DEVICES={devices['cuda_visible_devices']!r}, "
            f"TensorFlow sees {devices['tf_visible_devices']} ({devices['gpu_names']})"
        )

        data = prepare_data(config)
        steps_per_epoch = math.ceil(len(data.x_train) / config.batch_size)
        train_ds = make_train_dataset(
            data.x_train, data.y_train, config.batch_size, config.seed,
            flip=config.dataset in FLIP_DATASETS,
        )
        val_ds = make_eval_dataset(data.x_val, data.y_val)

        # ``kernel_regularizer`` is never passed: weight decay lives in the optimizer.
        model = family.factory(
            variant=config.variant,
            num_classes=data.num_classes,
            input_shape=data.input_shape,
            strides=config.strides,
            kernel_size=config.kernel_size,
            drop_path_rate=config.drop_path_rate,
            stochastic_mode=config.stochastic_mode,
            dropout_rate=config.dropout_rate,
            use_gamma=config.use_gamma,
        )
        # One optimizer, built once, identical in V1 and V2 (decisions.md D-005): a direct
        # AdamW rather than ``optimizer_builder`` so the clipping is set where Keras
        # requires it and no key-renaming step can drop it silently.
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=build_lr_schedule(config, steps_per_epoch),
                weight_decay=config.weight_decay,
                clipnorm=GRADIENT_CLIP_NORM,
            ),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=build_metrics(data.num_classes),
        )
        model.build((None, *data.input_shape))
        params = int(model.count_params())
        geometry = stage_feature_map_sizes(data.input_shape[:2], model.depths, config.strides)
        logger.info(f"  {family.label} {config.variant}: depths {model.depths}, dims {model.dims}, "
                    f"params {params:,}")
        logger.info(f"  Stage feature maps (H, W): {geometry}")
        logger.info(
            f"  LR: {config.learning_rate} ({config.lr_schedule}, warmup {config.warmup_epochs} "
            f"epochs, {steps_per_epoch} steps/epoch), weight decay {config.weight_decay}, "
            f"clipnorm {GRADIENT_CLIP_NORM}, batch {config.batch_size}"
        )

        # DECISION plan-2026-09-19T040641-db6932ec/D-014: the epoch-0 numbers (the summary's
        # initial-loss ratio AND the dashboard baseline marker) come from THIS one
        # evaluation. Do NOT let the dashboard callback call ``model.evaluate`` itself
        # "because it is simpler": it would measure twice and the two numbers could
        # disagree.
        baseline, initial_loss_ratio, init_scale_warning = _check_initial_loss(
            model, data.x_val, data.y_val, data.num_classes
        )
        summary_head = _summary_head(
            config, run_dir, data, model, geometry, steps_per_epoch, params,
            baseline, initial_loss_ratio, init_scale_warning, devices,
        )

        callbacks, _ = create_callbacks(
            model_name=config.experiment_name,
            results_dir_prefix=f"convnext_{config.model_family}",
            run_dir=str(run_dir),
            monitor=MONITOR,
            patience=config.patience,
            use_lr_schedule=config.lr_schedule != "constant",
            include_terminate_on_nan=True,
            include_analyzer=config.epoch_analysis,
        )
        # Index 0: `lr` must be in `logs` before CSVLogger reads it.
        callbacks.insert(0, LearningRateLogger())
        last_weights = _LastEpochWeights()
        callbacks.append(last_weights)
        # Last: it reads `logs['lr']`, written by LearningRateLogger above.
        dashboard = TrainingDashboardCallback(
            out_path=vis_dir / "training_dashboard.png",
            baseline_fn=lambda _model: dict(baseline),
            title=f"{config.experiment_name} (seed {config.seed})",
        )
        callbacks.append(dashboard)

        fit_started = time.perf_counter()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=1,
        )
        fit_wall_seconds = time.perf_counter() - fit_started
        log_gpu_peak_memory()
        save_training_history_json(history, str(run_dir))
        hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        epochs_run = len(hist.get(MONITOR, []))
        non_finite = _non_finite_metrics(hist)

        if non_finite:
            # TerminateOnNaN ended the run: fewer epochs than requested is NOT an early
            # stop and no best weights were restored, so ``stopped_early`` is unknown.
            message = (
                f"Training diverged: {non_finite} hold a non-finite or missing value after "
                f"{epochs_run} epoch(s); no evaluation, figures, analysis or final_model.keras "
                f"were produced. Initial-loss ratio was {initial_loss_ratio:.3g}."
            )
            logger.error(message)
            write_summary_json(run_dir, {
                "status": STATUS_DIVERGED,
                **summary_head,
                "epochs_run": epochs_run,
                "stopped_early": None,
                "best_epoch": None,
                "non_finite_metrics": non_finite,
                "history": hist,
                "epoch_times": list(dashboard.epoch_times),
                "fit_wall_seconds": fit_wall_seconds,
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
        best_epoch = _best_epoch(hist)
        best_i, final_i = best_epoch - 1, epochs_run - 1

        # The in-memory model holds the BEST weights here (EarlyStopping restored them).
        test_metrics_best, best_load_error, checkpoint_max_diff = _check_best_checkpoint(
            model, run_dir, data
        )

        # Figures and the analyzer describe the best (selected) weights. The model emits
        # LOGITS, so every probability consumer gets a softmax first.
        logits = model.predict(data.x_test, batch_size=EVAL_BATCH_SIZE, verbose=0)
        probs = keras.ops.convert_to_numpy(keras.ops.softmax(logits, axis=-1))
        visualizations = _write_visualizations(
            vis_dir, data, probs, get_class_names(config.dataset, data.num_classes)
        )
        run_model_analysis(
            model, (data.x_test, data.y_test), history, config.experiment_name, str(run_dir)
        )
        analysis = _read_analysis_status(run_dir, config.experiment_name)
        logger.info(f"Analyzer status (read back from disk): {analysis['status']}")

        # DECISION plan-2026-09-19T040641-db6932ec/D-012: the FINAL model is the last
        # epoch's weights, captured by ``_LastEpochWeights``, because Keras 3.8
        # EarlyStopping restores the best weights at every train end. Do NOT save
        # ``model`` as "final" without the ``set_weights`` below: it would silently be
        # the best model and ``test_metrics_final`` would equal ``test_metrics_best`` by
        # construction, hiding any best-versus-final difference.
        model.set_weights(last_weights.weights)
        test_metrics_final = _evaluate(model, data.x_test, data.y_test)
        logger.info(f"Test results (final weights, epoch {epochs_run}): {test_metrics_final}")
        final_path = run_dir / "final_model.keras"
        model.save(final_path)
        load_check: Optional[bool] = None
        try:
            sample = data.x_val[:LOAD_CHECK_SAMPLES]
            load_check = bool(validate_model_loading(
                str(final_path), sample, model.predict(sample, verbose=0)
            ))
        except Exception as e:  # noqa: BLE001 - log-only
            logger.warning(f"validate_model_loading raised: {e}")

        val_keys = [k for k in hist if k.startswith("val_")]
        summary: Dict[str, Any] = {
            "status": STATUS_OK,
            **summary_head,
            "epochs_run": epochs_run,
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_epoch_csv_index": best_i,
            "final_is_best": best_epoch == epochs_run,
            "lr_first_epoch": hist["lr"][0] if hist.get("lr") else None,
            "lr_last_epoch": hist["lr"][-1] if hist.get("lr") else None,
            "best_val_metrics": {k: hist[k][best_i] for k in val_keys},
            "final_val_metrics": {k: hist[k][final_i] for k in val_keys},
            "test_metrics_best": test_metrics_best,
            "test_metrics_final": test_metrics_final,
            "best_checkpoint_load_error": best_load_error,
            "best_checkpoint_max_abs_diff": checkpoint_max_diff,
            "epoch_times": list(dashboard.epoch_times),
            "fit_wall_seconds": fit_wall_seconds,
            "ece": visualizations["ece"],
            "visualizations": visualizations,
            "model_loading_validated": load_check,
            "analyzer": analysis,
            "notes": [
                f"initial loss {baseline['loss']:.4f} vs ln(C)={np.log(data.num_classes):.4f} "
                f"(ratio {initial_loss_ratio:.2f}, warn above {INITIAL_LOSS_WARN_FACTOR:g}), a "
                "plain evaluate on the validation split (LayerNorm model)",
                "CSV `epoch` is 0-based, `best_epoch` is 1-based (`best_epoch_csv_index` = "
                "`best_epoch` - 1)",
                "CSV `lr` is read AFTER the epoch's last step: with a per-step schedule it is "
                "the rate of the NEXT epoch's first step, not the rate the epoch trained at",
                "`test_metrics_best` is the reloaded best_model.keras, `test_metrics_final` the "
                "last epoch's weights (final_model.keras); figures and model_analysis/ use the "
                "best weights; the test set never influenced selection",
                "analyzer accuracy and its calibration numbers use the first 1000 test samples; "
                "top-level `ece` uses the full test set",
                "`fit_wall_seconds` minus the sum of `epoch_times` is time outside the epoch "
                "clock (dashboard redraws, checkpoint saves, train-end restore)",
            ],
        }
        return write_summary_json(run_dir, summary)


# ---------------------------------------------------------------------

def main(model_family: str, argv: Optional[Sequence[str]] = None) -> None:
    """Entry point of both wrappers. Parses ``argv`` FIRST so ``--help`` allocates nothing.

    Args:
        model_family: ``"v1"`` or ``"v2"``.
        argv: Argument vector; ``None`` reads ``sys.argv[1:]``.
    """
    args = parse_arguments(argv, model_family)
    config = config_from_args(args, model_family)
    setup_gpu(gpu_id=args.gpu)
    try:
        train(config)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
