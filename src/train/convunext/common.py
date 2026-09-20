"""
Orchestrator of the ConvUNext segmentation trainer.

This package trains ``create_convunext`` (``use_bias=True``, ``output_channels=3``, linear
head) as a semantic segmenter on Oxford-IIIT Pet and writes the run directory of
``src/train/convnext/`` (``config.json``, ``run.log``, ``training_log.csv``,
``training_history.json``, ``best_model.keras``, ``final_model.keras``,
``results_summary.json``, ``visualizations/`` and ``model_analysis/``). It is a sibling of
``train.convnext`` and imports every generic piece from ``train.common``; the bias-free ConvUNeXt DENOISER stays in ``train.bfunet``.

Data: ``oxford_iiit_pet`` 4.x from the local TFDS cache (``download=False``), decoded once by
:func:`load_oxford_pet` into in-memory uint8 arrays (bilinear images, NEAREST masks,
label = mask - 1, so 0 pet, 1 background, 2 border). A seeded ``validation_split`` slice of
the TRAIN split drives early stopping and the best checkpoint; the TEST split is read only
after ``fit``. ``--max-samples`` caps the TRAIN pool only; the test split is always the full
3669 images, so every run's test numbers are comparable.

Training: ``AdamW(cosine schedule with optional warmup, weight_decay, clipnorm=1.0)``, the
stock ``SparseCategoricalCrossentropy(from_logits=True)`` over sparse ``(B, H, W)`` masks
and a linear head, metrics ``accuracy`` (pixel accuracy, named so the shared dashboard draws
it) and stock ``MeanIoU`` (``miou``). ``EarlyStopping`` is built with
``restore_best_weights=False``: the in-memory model after ``fit`` is the real last epoch and
is what ``final_model.keras`` holds; the best epoch is ``best_model.keras`` (monitor
``val_loss``). The test report (per-class IoU, 3x3 confusion, pixel accuracy, mIoU) is
computed from a confusion matrix in numpy, independent of the Keras metric, next to a
majority-class baseline computed from the test masks.

Figures (``segmentation_viz``): a per-epoch grid of a fixed seeded validation batch
(``epoch_000`` is the untrained model), the shared training dashboard, and at the end of the
run the confusion matrix, per-class scores with ``segmentation_report.json``, best-versus-final
predictions and the mIoU curve, each isolated so a failure is recorded in the summary's
``visualizations.failed`` and never fails the run. ``model_analysis/`` (weights and spectral
analyses only) is written by the shared ``run_model_analysis``, read back from disk into the
summary's ``analyzer`` block, and wrapped so it cannot fail a finished run.

Refusals happen at config time, before any directory is created:

- an unknown ``variant`` (the choices ARE the keys of ``CONVUNEXT_CONFIGS``);
- a numeric field outside its range, and a warmup that is not shorter than the run;
- a fit split (train pool minus the validation slice) smaller than one full batch, because
  the train pipeline drops the incomplete last batch and would yield no step;
- an ``image_size`` below :func:`min_image_size`. MEASURED: ``create_convunext`` BUILDS at
  any size (1 px included) and only its forward pass fails, so a build probe cannot detect
  this. The smallest size that runs is ``2 ** depth`` (the bottleneck becomes 1x1), for
  every size above it too, odd ones included;
- a reused ``--experiment-name`` (:func:`resolve_new_run_dir`), before the GPU is configured.

A missing or incompatible TFDS cache raises from :func:`load_oxford_pet` inside
:func:`train` BEFORE the run directory is created. ``--help`` parses first, so it allocates
no GPU and no directory. ``--gpu`` is not a config field: ``main`` hands it to
``setup_gpu``, which OVERWRITES an exported ``CUDA_VISIBLE_DEVICES``.
"""

import argparse
import inspect
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.models.vision.convunext.model import (
    CONVUNEXT_CONFIGS,
    create_convunext,
    create_convunext_variant,
)
from dl_techniques.utils.logger import logger

from train.common import (
    attach_run_log,
    create_callbacks,
    create_learning_rate_schedule,
    default_experiment_name,
    log_gpu_peak_memory,
    prepare_run_dir,
    refuse_existing_run,
    resolved_run_dir,
    save_training_history_json,
    set_seeds,
    setup_gpu,
    validate_model_loading,
    write_summary_json,
)
from train.common import run_summary
from train.common.callbacks import best_checkpoint_path
from train.common.callbacks import EpochLogLine, LearningRateLogger
from train.common.classification_viz import TrainingDashboardCallback, plot_confusion_counts
# The split arithmetic, the "fit split holds at least one batch" rule and the constants
# below are the generic rules and values the ConvNeXt trainer states once; reused, not
# re-implemented.
from train.convnext.common import (
    GRADIENT_CLIP_NORM,
    LOAD_CHECK_SAMPLES,
    MONITOR,
    STATUS_DIVERGED,
    STATUS_OK,
    split_sizes,
    steps_per_epoch_for,
)
from train.convunext.segmentation_viz import (
    SegmentationGridCallback,
    class_scores,
    plot_best_vs_final_predictions,
    plot_miou_curve,
    plot_per_class_scores,
    write_segmentation_report,
)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# Train-split size of ``oxford_iiit_pet`` 4.0.0 (measured from the TFDS ``dataset_info``:
# train 3680, test 3669). Used ONLY to refuse an impossible configuration before a run
# directory exists; the data loader sizes the split from the array it really loads.
OXFORD_PET_TRAIN_SIZE = 3680

VARIANTS: Tuple[str, ...] = tuple(CONVUNEXT_CONFIGS)

DATASET_NAME = "oxford_iiit_pet"
# ``tfds`` version request; the cache holds 4.0.0 only (``3.*.*`` cannot load offline).
TFDS_NAME = "oxford_iiit_pet:4.*.*"
# TFDS mask values 1 (pet), 2 (background), 3 (border) minus 1.
CLASS_NAMES: Tuple[str, ...] = ("pet", "background", "border")
NUM_CLASSES = len(CLASS_NAMES)
# Images decoded per pass of the loader (bounds its peak memory; no effect on the result).
DECODE_BATCH = 128

# ``_check_initial_loss`` warns when loss / ln(num_classes) is STRICTLY above this.
INITIAL_LOSS_WARN_FACTOR = 10.0
# The reloaded best checkpoint must reproduce the validation metrics of its epoch.
BEST_CHECKPOINT_TOLERANCE = 1e-4

EPOCH_LINE_KEYS = ("loss", "accuracy", "miou", "val_loss", "val_accuracy", "val_miou")

# Largest 32-bit unsigned value: ``numpy.random.seed`` takes 0 .. 2**32 - 1.
MAX_SEED = 2 ** 32 - 1


# DECISION plan-2026-09-19T224205-49c8bf80/D-012: the minimum image size is the ANALYTIC
# ``2 ** depth``, not a build probe: ``create_convunext`` builds at every size and only its
# forward pass fails. Do NOT "measure" it by building a model. Guard:
# ``test_min_image_size_is_the_measured_forward_boundary`` (a real forward at min-1, min, min+1).
def min_image_size(variant: str) -> int:
    """Smallest square input side on which the ``variant`` ConvUNext runs: ``2 ** depth``.

    ``create_convunext`` builds at any size but its forward pass raises
    ``InvalidArgumentError`` below this (measured for depths 2 to 5 on CPU; guard:
    ``test_min_image_size_is_the_measured_forward_boundary``). Do NOT replace this with a
    build probe: the build succeeds at 1 px, so such a probe would accept every size.

    Args:
        variant: A key of ``CONVUNEXT_CONFIGS``.

    Returns:
        The minimum image side in pixels.

    Raises:
        KeyError: If ``variant`` is not a key of ``CONVUNEXT_CONFIGS``.
    """
    return 2 ** CONVUNEXT_CONFIGS[variant]["depth"]


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class SegTrainingConfig:
    """Configuration of one ConvUNext segmentation run.

    Every field is read by the trainer (``tests/test_train/test_config_fields_are_live.py``).
    ``experiment_name`` defaults to ``convunext_seg_<variant>_<timestamp>``. The defaults
    of the training block are the ConvNeXt trainer's, NOT measured on this task.
    """

    # Model
    variant: str = "tiny"
    image_size: int = 128

    # Data
    validation_split: float = 0.1
    max_samples: Optional[int] = None

    # Training
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 0
    patience: int = 10
    seed: int = 42

    # Monitoring / output
    viz_freq: int = 1
    viz_samples: int = 4
    model_analysis: bool = True
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate ranges and derive the experiment name.

        Raises:
            ValueError: If any field is outside its supported range.
        """
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")
        minimum = min_image_size(self.variant)
        if self.image_size < minimum:
            raise ValueError(
                f"image_size must be >= {minimum} for variant {self.variant!r} (depth "
                f"{CONVUNEXT_CONFIGS[self.variant]['depth']}: the forward pass fails below "
                f"2 ** depth), got {self.image_size}"
            )
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
        if not 0 <= self.seed <= MAX_SEED:
            raise ValueError(f"seed must be in [0, {MAX_SEED}], got {self.seed}")
        if not 0.0 < self.validation_split < 1.0:
            raise ValueError(
                f"validation_split must be strictly inside (0, 1), got {self.validation_split}. "
                "0 would silently validate on the test set."
            )
        if self.max_samples is not None and self.max_samples < 2:
            raise ValueError(f"max_samples must be >= 2 (or None), got {self.max_samples}")
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {self.warmup_epochs}")
        if 0 < self.warmup_epochs >= self.epochs:
            raise ValueError(
                f"warmup_epochs ({self.warmup_epochs}) must be smaller than epochs ({self.epochs})"
            )
        if self.viz_freq < 1:
            raise ValueError(f"viz_freq must be >= 1, got {self.viz_freq}")
        if self.viz_samples < 1:
            raise ValueError(f"viz_samples must be >= 1, got {self.viz_samples}")
        # The train pipeline drops the incomplete last batch, so the fit split must hold at
        # least one full batch, and that is refused HERE: after ``prepare_run_dir`` it
        # would leave a run directory behind and burn the experiment name.
        n_fit, _ = split_sizes(OXFORD_PET_TRAIN_SIZE, self.max_samples, self.validation_split)
        steps_per_epoch_for(n_fit, self.batch_size)
        if self.experiment_name is None:
            self.experiment_name = default_experiment_name("convunext_seg", self.variant)


def resolve_new_run_dir(config: SegTrainingConfig) -> Path:
    """Resolve the run directory of ``config`` and refuse it if it already holds a run.

    Writes nothing and creates nothing, so it is safe to call before the GPU is
    configured. A relative ``output_dir`` is anchored at the repo root (never the working
    directory); an absolute one is used as given.

    Args:
        config: A validated :class:`SegTrainingConfig`.

    Returns:
        ``<output_dir>/<experiment_name>``; it may not exist yet.

    Raises:
        FileExistsError: If the directory already holds a run's files (nothing is written).
    """
    run_dir = Path(resolved_run_dir(config))
    refuse_existing_run(run_dir)
    return run_dir


# ---------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the unparsed trainer parser.

    Defaults are read off a default :class:`SegTrainingConfig`, so parser and config
    cannot drift; ``--experiment-name`` defaults to ``None`` (derived by the config).
    ``--gpu`` is not a config field.

    Returns:
        An ``argparse.ArgumentParser`` with every flag of the trainer.
    """
    defaults = SegTrainingConfig()
    parser = argparse.ArgumentParser(
        description="Train ConvUNext as a 3-class segmenter on Oxford-IIIT Pet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    model = parser.add_argument_group("model")
    model.add_argument("--variant", type=str, default=defaults.variant, choices=VARIANTS,
                       help="ConvUNext variant (sets depth, width and blocks).")
    model.add_argument("--image-size", type=int, default=defaults.image_size,
                       help="Square side images and masks are resized to. At least 2 ** depth "
                            "of the variant (tiny 8, base 16, xlarge 32); odd sizes are fine.")

    data = parser.add_argument_group("data")
    data.add_argument("--validation-split", type=float, default=defaults.validation_split,
                      help="Fraction of the train set held out (seeded) for early stopping and "
                           "checkpoint selection, strictly inside (0, 1); the test set is only "
                           "used for the final report.")
    data.add_argument("--max-samples", type=int, default=defaults.max_samples,
                      help="Cap the train pool (fit and validation splits together) at this many "
                           "samples (smoke runs and tests); default: the whole train set. The "
                           "test split is never capped (always all 3669 images).")

    train = parser.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=defaults.epochs,
                       help="Maximum number of training epochs (the cosine spans this many).")
    train.add_argument("--batch-size", type=int, default=defaults.batch_size,
                       help="Training batch size; the fit split must hold at least one batch.")
    train.add_argument("--learning-rate", type=float, default=defaults.learning_rate,
                       help="Peak learning rate.")
    train.add_argument("--weight-decay", type=float, default=defaults.weight_decay,
                       help="Decoupled AdamW weight decay.")
    train.add_argument("--warmup-epochs", type=int, default=defaults.warmup_epochs,
                       help="Linear warmup epochs before the cosine; smaller than --epochs.")
    train.add_argument("--patience", type=int, default=defaults.patience,
                       help="Early-stopping patience in epochs on val_loss.")
    train.add_argument("--seed", type=int, default=defaults.seed,
                       help="Seed for weights, shuffling, augmentation and the splits.")
    train.add_argument("--viz-freq", type=int, default=defaults.viz_freq,
                       help="Write the segmentation grid every this many epochs.")
    train.add_argument("--viz-samples", type=int, default=defaults.viz_samples,
                       help="Validation images shown in each segmentation grid.")

    train.add_argument("--model-analysis", action=argparse.BooleanOptionalAction,
                       default=defaults.model_analysis,
                       help="Run the end-of-run ModelAnalyzer (weights and spectral analyses only) "
                            "into model_analysis/; --no-model-analysis skips it.")

    out = parser.add_argument_group("output")
    out.add_argument("--output-dir", type=str, default=defaults.output_dir,
                     help="Output root; a relative path is anchored at the repo root.")
    out.add_argument("--experiment-name", type=str, default=None,
                     help="Run directory name (default: convunext_seg_<variant>_<timestamp>). "
                          "A name that already holds a run is refused.")
    out.add_argument("--gpu", type=int, default=None,
                     help="GPU device index; sets CUDA_VISIBLE_DEVICES before TensorFlow first "
                          "enumerates devices and OVERRIDES an exported CUDA_VISIBLE_DEVICES. "
                          "Default: use the environment.")
    return parser


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line. ``--help`` exits here, before anything expensive.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The parsed ``argparse.Namespace``.
    """
    return _build_parser().parse_args(argv)


def config_from_args(args: argparse.Namespace) -> SegTrainingConfig:
    """Build a :class:`SegTrainingConfig` from a parsed namespace.

    ``--gpu`` is deliberately not a config field: it is consumed once by ``setup_gpu`` in
    :func:`main`.

    Args:
        args: Namespace returned by :func:`parse_arguments`.

    Returns:
        The validated config.

    Raises:
        ValueError: If a value is outside its supported range.
    """
    # Every config field comes from the namespace attribute of the same name, so a field
    # with no flag raises AttributeError here instead of silently keeping its default.
    return SegTrainingConfig(**{f.name: getattr(args, f.name) for f in fields(SegTrainingConfig)})


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def load_oxford_pet(image_size: int) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Decode ``oxford_iiit_pet`` 4.x from the local TFDS cache into uint8 arrays.

    The ONE function that touches TFDS; tests stub exactly this. Images are resized to
    ``image_size`` x ``image_size`` bilinearly (with antialiasing: the source is about 500
    px), masks with NEAREST so no invented value appears, and the label is the TFDS mask
    minus 1 (TFDS 1 pet, 2 background, 3 border become 0, 1, 2). Order is the TFDS shard
    order (no shuffling), so the arrays are identical from run to run.

    Args:
        image_size: Output side in pixels.

    Returns:
        ``((x_train, y_train), (x_test, y_test))``: ``x`` is uint8 ``(N, S, S, 3)``, ``y`` is
        uint8 ``(N, S, S)`` with values in ``{0, 1, 2}``. 3680 train and 3669 test images.

    Raises:
        RuntimeError: If the dataset cannot be read with ``download=False`` (cache missing,
            wrong version, ``TFDS_DATA_DIR`` unset or wrong); nothing is downloaded.
        ValueError: If a decoded mask holds a value outside ``{1, 2, 3}``.
    """
    import tensorflow_datasets as tfds  # lazy: keeps ``--help`` and the config path light

    size = (int(image_size), int(image_size))

    def resize(example: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.image.resize(example["image"], size, method="bilinear", antialias=True)
        image = tf.cast(tf.clip_by_value(tf.round(image), 0.0, 255.0), tf.uint8)
        mask = tf.image.resize(example["segmentation_mask"], size, method="nearest")
        return image, tf.cast(tf.squeeze(mask, axis=-1), tf.int32) - 1

    def split(name: str) -> Tuple[np.ndarray, np.ndarray]:
        try:
            dataset = tfds.load(TFDS_NAME, split=name, download=False, shuffle_files=False)
        except Exception as e:  # noqa: BLE001 - re-raised with the fix named
            raise RuntimeError(
                f"Could not read {TFDS_NAME!r} split {name!r} from the local TFDS cache with "
                f"download=False: {type(e).__name__}: {e}. Set TFDS_DATA_DIR to the directory "
                "that holds oxford_iiit_pet/4.0.0 (nothing is downloaded by this trainer)."
            ) from e
        batches = tfds.as_numpy(dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE).batch(DECODE_BATCH))
        images, labels = zip(*batches)
        x, y = np.concatenate(images), np.concatenate(labels)
        if y.min() < 0 or y.max() >= NUM_CLASSES:
            raise ValueError(
                f"{TFDS_NAME} split {name!r}: mask - 1 must lie in [0, {NUM_CLASSES - 1}], got "
                f"[{y.min()}, {y.max()}]"
            )
        return x, y.astype(np.uint8)

    return split("train"), split("test")


def prepare_data(config: SegTrainingConfig) -> Dict[str, np.ndarray]:
    """Load the dataset and cut the seeded train / validation split; the test split is kept apart.

    The validation slice is the first ``n_val`` entries of a seeded permutation of the TRAIN
    split (after ``--max-samples`` truncated the permutation), the fit split is the rest, so
    the two are disjoint by construction; the test split is returned untouched and never
    influences the split or the checkpoint.

    Args:
        config: A validated :class:`SegTrainingConfig`.

    Returns:
        ``{"x_train", "y_train", "x_val", "y_val", "x_test", "y_test"}``: uint8 images
        ``(N, S, S, 3)`` and uint8 masks ``(N, S, S)`` with values in ``{0, 1, 2}``.

    Raises:
        RuntimeError: From :func:`load_oxford_pet`, if the dataset cannot be read.
        ValueError: If the validation split would hold out zero samples.
    """
    (x_train, y_train), (x_test, y_test) = load_oxford_pet(config.image_size)
    order = np.random.default_rng(config.seed).permutation(len(x_train))
    if config.max_samples is not None:
        order = order[:config.max_samples]
    _, n_val = split_sizes(len(x_train), config.max_samples, config.validation_split)
    val_idx, fit_idx = order[:n_val], order[n_val:]
    return {
        "x_train": x_train[fit_idx], "y_train": y_train[fit_idx],
        "x_val": x_train[val_idx], "y_val": y_train[val_idx],
        "x_test": x_test, "y_test": y_test,
    }


def _scale(image: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """uint8 image to float32 in [0, 1], uint8 mask to int32 class ids."""
    return tf.cast(image, tf.float32) / 255.0, tf.cast(mask, tf.int32)


def _random_flip(
        seed: int, position: tf.Tensor, image: tf.Tensor, mask: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Flip image ``(H, W, 3)`` and mask ``(H, W)`` left-right TOGETHER with probability 0.5.

    The draw is a pure function of ``(seed, position)`` (``position`` = the element's index
    in this epoch's shuffled order), so it does not depend on which parallel map worker gets
    to run first.
    """
    # DECISION plan-2026-09-19T224205-49c8bf80/D-017: a STATELESS draw keyed by the epoch
    # position. ``tf.random.uniform([])`` inside ``map(num_parallel_calls=AUTOTUNE)`` is a
    # stateful op whose draws go to elements in worker-scheduling order, so two same-seed
    # runs flipped different samples (a candidate cause of the audit's same-seed spread).
    # Do NOT put a stateful random op back into a parallel map, and do NOT drop
    # ``num_parallel_calls`` to hide the race (the map is the input-pipeline cost); the guard
    # is ``test_two_train_pipelines_with_one_seed_yield_identical_batches_over_two_epochs``.
    key = tf.stack([tf.constant(seed, tf.int64), position])
    flip = tf.random.stateless_uniform([], seed=key) < 0.5
    return (tf.where(flip, tf.reverse(image, axis=[1]), image),
            tf.where(flip, tf.reverse(mask, axis=[1]), mask))


# The train pipeline DROPS the incomplete last batch (``steps_per_epoch_for`` and the LR
# schedule count on it; a smaller final batch is a second input shape that recompiles the
# train step). The pool is reshuffled every epoch, so no sample is permanently excluded.
def make_train_dataset(x: np.ndarray, y: np.ndarray, batch_size: int, seed: int) -> "tf.data.Dataset":
    """The shuffled, flip-augmented, batched train pipeline (tf.data).

    Args:
        x: uint8 images ``(N, S, S, 3)``.
        y: uint8 masks ``(N, S, S)``.
        batch_size: Batch size; an epoch yields :func:`steps_per_epoch_for` batches.
        seed: Seed of the shuffle AND of the flips. The flips are stateless, so the batches
            (images and masks) of every epoch are reproducible from it under any parallelism
            (the shuffle also reads the global TensorFlow seed, which the trainer sets).

    Returns:
        A dataset of ``(float32 images in [0, 1], int32 masks)`` batches.
    """
    return (
        tf.data.Dataset.from_tensor_slices((x, y))
        .shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
        .enumerate()
        .map(lambda position, pair: _random_flip(seed, position, *_scale(*pair)),
             num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )


def make_eval_dataset(x: np.ndarray, y: np.ndarray, batch_size: int) -> "tf.data.Dataset":
    """The un-augmented, un-shuffled, batched pipeline (validation and test).

    Args:
        x: uint8 images ``(N, S, S, 3)``.
        y: uint8 masks ``(N, S, S)``.
        batch_size: Rows per batch (the last batch may be smaller).

    Returns:
        A dataset of ``(float32 images in [0, 1], int32 masks)`` batches.
    """
    return (
        tf.data.Dataset.from_tensor_slices((x, y))
        .map(_scale, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


# ---------------------------------------------------------------------
# Metrics from a confusion matrix (numpy, independent of the Keras metric)
# ---------------------------------------------------------------------

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Pixel confusion matrix: entry ``[t, p]`` counts pixels of true class ``t`` predicted ``p``.

    Args:
        y_true: Integer class ids, any shape.
        y_pred: Integer class ids, same number of elements.
        num_classes: Number of classes ``C``.

    Returns:
        ``int64`` ``(C, C)``.

    Raises:
        ValueError: If the two differ in size or a value is outside ``[0, C)`` (a wrong
            value would land in another cell of the flattened bincount instead of failing).
    """
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.int64)
    if y_true.size != y_pred.size:
        raise ValueError(f"y_true has {y_true.size} elements, y_pred {y_pred.size}")
    if y_true.size and (min(y_true.min(), y_pred.min()) < 0 or max(y_true.max(), y_pred.max()) >= num_classes):
        raise ValueError(f"class ids must lie in [0, {num_classes})")
    return np.bincount(y_true * num_classes + y_pred, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def segmentation_scores(confusion: np.ndarray) -> Dict[str, Any]:
    """Per-class IoU, mean IoU and pixel accuracy of a confusion matrix.

    ``IoU_c = TP_c / (true_c + predicted_c - TP_c)``. A class with an empty union (no true
    and no predicted pixel) has no IoU: ``None`` in ``per_class_iou`` and left out of the
    mean, the convention of ``keras.metrics.MeanIoU`` (which the tests cross-check). The
    formula lives ONCE, in ``segmentation_viz.class_scores`` (which also computes Dice,
    precision and recall for the report); this returns its three summary keys.

    Args:
        confusion: ``(C, C)`` counts from :func:`confusion_matrix`.

    Returns:
        ``{"per_class_iou": [float | None] * C, "miou": float | None, "pixel_accuracy":
        float | None}``; ``None`` where the denominator is zero.
    """
    scores = class_scores(confusion)
    return {key: scores[key] for key in ("per_class_iou", "miou", "pixel_accuracy")}


def trivial_baseline(y_true: np.ndarray, num_classes: int = NUM_CLASSES) -> Dict[str, Any]:
    """Scores of predicting the majority class of ``y_true`` for every pixel.

    The floor a trained segmenter must beat: mIoU above it AND more than one predicted class.

    Args:
        y_true: Integer masks (the TEST masks), any shape.
        num_classes: Number of classes.

    Returns:
        :func:`segmentation_scores` of the constant predictor plus ``predicted_class`` (ties
        resolve to the lowest class id) and ``confusion``.
    """
    counts = np.bincount(np.asarray(y_true).reshape(-1).astype(np.int64), minlength=num_classes)
    majority = int(np.argmax(counts))
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    confusion[:, majority] = counts
    return {"predicted_class": majority, **segmentation_scores(confusion), "confusion": confusion.tolist()}


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def build_lr_schedule(config: SegTrainingConfig, steps_per_epoch: int) -> Any:
    """Cosine over ALL ``config.epochs`` epochs, optionally preceded by a linear warmup.

    ``steps_per_epoch`` MUST reach the schedule: without it the cosine counts optimizer
    steps against ``decay_steps = epochs`` and hits its floor after ``epochs`` batches.
    Warmup engages only through ``warmup_steps``.

    Args:
        config: A validated :class:`SegTrainingConfig`.
        steps_per_epoch: Optimizer steps of one epoch of :func:`make_train_dataset`.

    Returns:
        A Keras ``LearningRateSchedule``.
    """
    return create_learning_rate_schedule(
        initial_lr=config.learning_rate,
        schedule_type="cosine",
        total_epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=config.warmup_epochs * steps_per_epoch,
    )


def build_model(config: SegTrainingConfig, schedule: Any) -> keras.Model:
    """Build and compile the segmenter.

    Args:
        config: A validated :class:`SegTrainingConfig`.
        schedule: The learning rate handed to ``AdamW`` (:func:`build_lr_schedule`).

    Returns:
        A compiled functional model, input ``(S, S, 3)`` in [0, 1], output ``(S, S, 3)`` LOGITS.
    """
    # DECISION plan-2026-09-19T224205-49c8bf80/D-004: a LINEAR 3-channel head trained with the
    # stock ``SparseCategoricalCrossentropy(from_logits=True)`` over sparse ``(B, H, W)``
    # masks, and stock ``MeanIoU`` for the mIoU. Do NOT switch to a softmax head "to get
    # probabilities" (then ``from_logits=True`` softmaxes twice) and do NOT wrap the
    # repo's ``SegmentationLosses``: they need one-hot targets of the prediction's shape and
    # probability inputs, i.e. a wrapper class and a second label format. ``use_bias=True``
    # is required for anything but a linear/relu head (the bias-free arm is the denoiser).
    model = create_convunext_variant(
        config.variant, (config.image_size, config.image_size, 3),
        use_bias=True, output_channels=NUM_CLASSES, final_activation="linear",
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=schedule, weight_decay=config.weight_decay, clipnorm=GRADIENT_CLIP_NORM,
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.MeanIoU(num_classes=NUM_CLASSES, sparse_y_true=True, sparse_y_pred=False, name="miou"),
        ],
    )
    return model


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------

def _evaluate_split(model: keras.Model, x: np.ndarray, y: np.ndarray, batch_size: int) -> Dict[str, Any]:
    """Loss, accuracy, mIoU and the confusion-matrix report of ``model`` on one split, in ONE pass.

    Each batch is predicted once; ``loss`` is the compiled loss of those logits (weighted by
    batch size, as ``fit`` and ``evaluate`` average it), and ``accuracy`` / ``miou`` are read
    off the accumulated :func:`confusion_matrix`, so ``accuracy`` equals ``pixel_accuracy`` and
    ``miou`` equals ``miou_from_confusion`` by construction. ``model.evaluate`` gave the same
    numbers from a second pass over the split; the test that compares the two is
    ``test_the_one_pass_evaluation_equals_keras_evaluate``.

    Args:
        model: A compiled model.
        x: uint8 images.
        y: uint8 masks.
        batch_size: Rows per batch.

    Returns:
        A JSON-ready dict.
    """
    # DECISION plan-2026-09-19T224205-49c8bf80/D-026: ONE pass. ``model.evaluate`` plus this
    # loop cost 17.6 s + 15.7 s on the 3669 test images (cold, tiny, 128 px, GPU 1) for
    # numbers that agree to 1e-6. Do NOT put ``evaluate`` back "to have Keras' own numbers":
    # ``test_the_one_pass_evaluation_equals_keras_evaluate`` is the cross-check.
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    loss_sum = 0.0
    for images, masks in make_eval_dataset(x, y, batch_size):
        logits = model.predict_on_batch(images)
        loss_sum += float(model.loss(masks, logits)) * len(masks)
        confusion += confusion_matrix(masks.numpy(), np.argmax(logits, axis=-1), NUM_CLASSES)
    scores = segmentation_scores(confusion)
    return {
        "loss": loss_sum / len(x),
        "accuracy": scores["pixel_accuracy"],
        "miou": scores["miou"],
        "miou_from_confusion": scores["miou"],
        "pixel_accuracy": scores["pixel_accuracy"],
        "per_class_iou": scores["per_class_iou"],
        "confusion": confusion.tolist(),
    }


def _check_initial_loss(
        model: keras.Model, dataset: "tf.data.Dataset", n_samples: int
) -> Tuple[Dict[str, float], float, bool]:
    """Evaluate the untrained model on the validation split; refuse a non-finite loss.

    The single epoch-0 measurement: the summary's initial-loss ratio and the dashboard's
    baseline marker both come from THIS result, so they cannot disagree.

    Args:
        model: Built and compiled model.
        dataset: The validation pipeline.
        n_samples: Validation images (for the log line only).

    Returns:
        ``(metrics, ratio, warned)``: the evaluate dict, ``loss / ln(3)`` and whether the
        ratio is STRICTLY above :data:`INITIAL_LOSS_WARN_FACTOR`.

    Raises:
        RuntimeError: If the loss is NaN or infinite (raised before ``fit``, so a broken
            model does not end as a "finished" run).
    """
    metrics = {k: float(v) for k, v in model.evaluate(dataset, verbose=0, return_dict=True).items()}
    loss = metrics["loss"]
    uniform_loss = float(np.log(NUM_CLASSES))
    logger.info(
        f"Untrained evaluate BEFORE fit: val loss {loss:.4f} on {n_samples} images "
        f"(uniform-prediction loss would be {uniform_loss:.4f})"
    )
    if not np.isfinite(loss):
        raise RuntimeError(f"Initial validation loss is {loss}: the model diverges before training.")
    ratio = loss / uniform_loss
    warned = ratio > INITIAL_LOSS_WARN_FACTOR
    if warned:
        logger.warning(
            f"Initial loss {loss:.4f} is {ratio:.1f}x ln(3)={uniform_loss:.4f} (warn above "
            f"{INITIAL_LOSS_WARN_FACTOR:g}x): the initial logit scale is far too large."
        )
    return metrics, ratio, warned


def _check_best_checkpoint(
        run_dir: Path, data: Dict[str, np.ndarray], batch_size: int, hist: Dict[str, List[float]], best_i: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[float]]:
    """Test metrics of the reloaded ``best_model.keras`` and how well it reproduces its epoch.

    The in-memory model after ``fit`` is the LAST epoch (``restore_best_weights=False``), so
    the checkpoint cannot be compared with in-memory best weights. It is compared with the
    History instead: the reloaded model's validation loss, accuracy and mIoU must equal the
    values ``fit`` recorded for the best epoch.

    Args:
        run_dir: The run directory holding ``best_model.keras``.
        data: The :func:`prepare_data` dict.
        batch_size: Evaluation batch size.
        hist: ``{metric: per-epoch floats}`` of the finished fit.
        best_i: 0-based index of the best epoch.

    Returns:
        ``(test_metrics, load_error, max_abs_diff)``; the first and last are ``None`` when the
        checkpoint is missing or does not load. ``max_abs_diff`` is the largest absolute gap
        between the reloaded and the recorded validation metric.
    """
    def evaluate(model: keras.Model) -> Dict[str, Any]:
        val = model.evaluate(make_eval_dataset(data["x_val"], data["y_val"], batch_size),
                             verbose=0, return_dict=True)
        return {"val": {k: float(v) for k, v in val.items()},
                "test": _evaluate_split(model, data["x_test"], data["y_test"], batch_size)}

    reloaded, load_error = run_summary.load_best_metrics(run_dir, evaluate)
    if reloaded is None:
        return None, load_error, None
    logger.info(f"Test results (best_model.keras): {reloaded['test']}")
    gap = max(abs(reloaded["val"][k] - hist[f"val_{k}"][best_i]) for k in ("loss", "accuracy", "miou"))
    if gap > BEST_CHECKPOINT_TOLERANCE:
        logger.warning(
            f"best_model.keras does not reproduce the best epoch's validation metrics (gap {gap:.3g})"
        )
    return reloaded["test"], None, gap


# ---------------------------------------------------------------------
# End of run: figures and the analyzer (neither may fail a finished run)
# ---------------------------------------------------------------------

def _write_figures(
        run_dir: Path, grid: SegmentationGridCallback, model: keras.Model, hist: Dict[str, List[float]],
        best_epoch: int, test_metrics_best: Optional[Dict[str, Any]], config: SegTrainingConfig,
) -> Dict[str, Any]:
    """The end-of-run figures, each isolated: one that raises is recorded, the rest still draw.

    Args:
        run_dir: The run directory (figures go to ``<run_dir>/visualizations``).
        grid: The fitted grid callback (its fixed samples are reused for best-vs-final).
        model: The in-memory model, the LAST epoch's weights.
        hist: ``{metric: per-epoch floats}`` of the finished fit.
        best_epoch: 1-based best epoch.
        test_metrics_best: The reloaded best checkpoint's test block, or ``None`` if it did not load.
        config: The run config (its ``experiment_name`` titles the figures).

    Returns:
        ``{"files": [names that exist on disk], "failed": [names], "skipped":
        {name: reason}, "seconds"}``: the dashboard and the per-epoch grids come first, then
        the figures of this function. ``seconds`` is the wall time of the per-epoch grids
        (``grid.seconds``) plus the end-of-run figures (the dashboard redraw is not timed).
    """
    vis_dir = run_dir / "visualizations"
    epochs_run = len(hist[MONITOR])
    started = time.perf_counter()
    out: Dict[str, Any] = {"files": [], "failed": list(grid.failed), "skipped": {}}
    out["files"] += [n for n in ["training_dashboard.png", *grid.written] if (vis_dir / n).is_file()]

    def attempt(name: str, draw: Callable[[], Any]) -> None:
        try:
            draw()
            if not (vis_dir / name).is_file():
                raise RuntimeError("the figure function returned without writing the file")
            out["files"].append(name)
        except Exception as e:  # noqa: BLE001 - a figure must not fail the run
            logger.warning(f"Visualization {name} failed: {type(e).__name__}: {e}")
            out["failed"].append(name)

    def best_confusion() -> np.ndarray:
        if test_metrics_best is None:
            raise RuntimeError("best_model.keras did not load, so there is no best-weights confusion matrix")
        return np.asarray(test_metrics_best["confusion"])

    def best_vs_final() -> None:
        best = grid.predict_classes(keras.models.load_model(best_checkpoint_path(str(run_dir))))
        plot_best_vs_final_predictions(
            grid.images, grid.masks, best, grid.predict_classes(model), CLASS_NAMES,
            vis_dir / "best_vs_final_predictions.png",
            best_label=f"best (epoch {best_epoch})", final_label=f"final (epoch {epochs_run})",
            title=config.experiment_name)

    attempt("confusion_matrix.png", lambda: plot_confusion_counts(
        best_confusion(), CLASS_NAMES, vis_dir / "confusion_matrix.png",
        subtitle=f"{config.experiment_name}: test split, best weights, pixel counts"))
    attempt("per_class_metrics.png", lambda: plot_per_class_scores(
        best_confusion(), CLASS_NAMES, vis_dir / "per_class_metrics.png"))
    attempt("segmentation_report.json", lambda: write_segmentation_report(
        best_confusion(), CLASS_NAMES, vis_dir / "segmentation_report.json"))
    if best_epoch == epochs_run:
        # DECISION plan-2026-09-19T224205-49c8bf80/D-029: the best epoch IS the last, so the two
        # columns would be the same weights (two identical predictions, 200-230 KB, 0.3 s in every
        # loop-2 run). Do NOT draw it "for completeness" and do NOT draw it silently missing: the
        # skip is recorded in ``skipped`` and the summary stays truthful.
        out["skipped"]["best_vs_final_predictions.png"] = "best epoch == last epoch: identical columns"
    else:
        attempt("best_vs_final_predictions.png", best_vs_final)
    attempt("miou_curve.png", lambda: plot_miou_curve(
        hist, best_epoch, vis_dir / "miou_curve.png", title=f"{config.experiment_name}: mIoU per epoch"))
    out["seconds"] = grid.seconds + time.perf_counter() - started
    logger.info(
        f"Visualizations: {len(out['files'])} written ({', '.join(out['files'])}), "
        f"{len(out['failed'])} failed{': ' + ', '.join(out['failed']) if out['failed'] else ''}, "
        f"{len(out['skipped'])} skipped{': ' + ', '.join(out['skipped']) if out['skipped'] else ''}, "
        f"grids plus end-of-run figures took {out['seconds']:.1f}s"
    )
    return out


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def _analyzer_notes(ran: bool) -> List[str]:
    """The ``notes`` lines about ``model_analysis/``: what it measured, or that it was skipped."""
    if not ran:
        return ["model_analysis/ was skipped (--no-model-analysis); `analyzer.status` is 'skipped'"]
    return [
        "model_analysis/ holds the WEIGHTS and SPECTRAL analyses only (calibration, information "
        "flow and training dynamics read per-image labels and are off for a dense output) of the "
        "LAST epoch's weights (the in-memory model, final_model.keras); `analyzer.status` is read "
        "back from analysis_results.json (`analyzers` lists which sections have results, "
        "`seconds` is the wall time of the analysis and its figures); its spectral verdicts are "
        "heuristics, unreliable for a short run",
    ]


def architecture_of(variant: str) -> Dict[str, Any]:
    """The architecture keys of the summary, resolved the way :func:`build_model` builds.

    ``create_convunext_variant`` fills ``create_convunext``'s signature defaults with the
    variant's row of ``CONVUNEXT_CONFIGS`` and then applies the three arguments
    :func:`build_model` passes; this repeats exactly that resolution, so no value is typed
    a second time. Names follow the ConvNeXt summary where the model has the same thing:
    ``dims`` is the channel count of the encoder levels and the bottleneck
    (``initial_filters * filter_multiplier ** i``, the ``Filter progression`` the model logs)
    and ``kernel_size`` the block kernel. The model has no ``strides``, ``use_gamma`` or
    ``stochastic_mode`` (ConvNeXt V2 blocks carry a GRN instead of a layer scale), so those
    ConvNeXt keys are absent here on purpose.

    Args:
        variant: A key of ``CONVUNEXT_CONFIGS``.

    Returns:
        A JSON-ready dict.

    Raises:
        KeyError: If ``variant`` is not a key of ``CONVUNEXT_CONFIGS``.
    """
    resolved = {name: p.default for name, p in inspect.signature(create_convunext).parameters.items()}
    resolved.update({k: v for k, v in CONVUNEXT_CONFIGS[variant].items() if k != "description"})
    resolved.update(use_bias=True, output_channels=NUM_CLASSES, final_activation="linear")
    dims = [int(round(resolved["initial_filters"] * resolved["filter_multiplier"] ** i))
            for i in range(resolved["depth"] + 1)]
    return {
        "convnext_version": resolved["convnext_version"],
        "depth": resolved["depth"],
        "blocks_per_level": resolved["blocks_per_level"],
        "dims": dims,
        "kernel_size": resolved["block_kernel_size"],
        "stem_kernel_size": resolved["stem_kernel_size"],
        "block_normalization": resolved["block_normalization"],
        "drop_path_rate": resolved["drop_path_rate"],
        "dropout_rate": resolved["dropout_rate"],
        "use_bias": resolved["use_bias"],
        "final_activation": resolved["final_activation"],
    }


def _summary_head(
        config: SegTrainingConfig, run_dir: Path, data: Dict[str, np.ndarray], params: int,
        steps_per_epoch: int, baseline: Dict[str, float], initial_loss_ratio: float,
        init_scale_warning: bool, devices: Dict[str, Any], data_load_seconds: float,
) -> Dict[str, Any]:
    """Keys every summary carries, whether the run finished or diverged."""
    return {
        **run_summary.summary_head(
            config, run_dir, params=params, steps_per_epoch=steps_per_epoch, devices=devices),
        "model_family": "convunext-seg",
        "dataset": DATASET_NAME,
        **architecture_of(config.variant),
        "input_shape": [config.image_size, config.image_size, 3],
        "num_classes": NUM_CLASSES,
        "class_names": list(CLASS_NAMES),
        "optimizer": "AdamW",
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "lr_schedule": "cosine",
        "validation_split": config.validation_split,
        "max_samples": config.max_samples,
        "n_train": int(len(data["x_train"])),
        "n_val": int(len(data["x_val"])),
        "n_test": int(len(data["x_test"])),
        "data_load_seconds": data_load_seconds,
        "monitor": MONITOR,
        "initial_loss_sanity_eval": {
            "loss": baseline["loss"], "n_samples": int(len(data["x_val"])),
            "split": "val", "before_fit": True,
        },
        "initial_loss_ratio": initial_loss_ratio,
        "init_scale_warning": init_scale_warning,
    }


def train(config: SegTrainingConfig) -> Dict[str, Any]:
    """Train the segmenter, evaluate it on the test split, write every artifact, return the summary.

    Order: refuse a reused name (nothing written), load the data (a missing cache raises
    HERE, before any directory exists), then create the run directory and attach
    ``run.log`` for the rest of the call (detached on return AND on an exception).

    ``test_metrics_best`` is the reloaded ``best_model.keras`` on the test split,
    ``test_metrics_final`` the LAST epoch's weights, which is what ``final_model.keras``
    holds. The test split is evaluated only after ``fit`` and never selects anything.

    Args:
        config: A validated :class:`SegTrainingConfig`.

    Returns:
        The strict-JSON dict also written to ``<run_dir>/results_summary.json``
        (``status`` is ``"ok"``).

    Raises:
        FileExistsError: If the experiment directory already holds a run (nothing written).
        RuntimeError: If the TFDS cache cannot be read (nothing written), if the initial
            validation loss is non-finite, or (after a ``status: "diverged"`` summary was
            written) if an epoch ``loss`` / ``val_loss`` is non-finite.
    """
    resolved = resolve_new_run_dir(config)
    load_started = time.perf_counter()
    data = prepare_data(config)
    data_load_seconds = time.perf_counter() - load_started
    steps_per_epoch = steps_per_epoch_for(len(data["x_train"]), config.batch_size)

    run_dir = Path(prepare_run_dir(config, output_dir=resolved)).resolve()
    (run_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    with attach_run_log(run_dir):
        set_seeds(config.seed)
        logger.info(f"Run directory: {run_dir}")
        devices = run_summary.describe_devices()
        logger.info(
            f"Devices: CUDA_VISIBLE_DEVICES={devices['cuda_visible_devices']!r}, "
            f"TensorFlow sees {devices['tf_visible_devices']} ({devices['gpu_names']})"
        )
        logger.info(
            f"Data ({DATASET_NAME}, {data_load_seconds:.1f}s to load): train {data['x_train'].shape}, "
            f"val {data['x_val'].shape}, test {data['x_test'].shape}; "
            f"train pixel classes {np.bincount(data['y_train'].reshape(-1), minlength=NUM_CLASSES).tolist()}"
        )

        train_ds = make_train_dataset(data["x_train"], data["y_train"], config.batch_size, config.seed)
        val_ds = make_eval_dataset(data["x_val"], data["y_val"], config.batch_size)

        schedule = build_lr_schedule(config, steps_per_epoch)
        model = build_model(config, schedule)
        params = int(model.count_params())
        logger.info(
            f"  ConvUNext {config.variant}: params {params:,}, LR {config.learning_rate} (cosine, "
            f"warmup {config.warmup_epochs} epochs, {steps_per_epoch} steps/epoch), weight decay "
            f"{config.weight_decay}, clipnorm {GRADIENT_CLIP_NORM}, batch {config.batch_size}"
        )

        baseline, initial_loss_ratio, init_scale_warning = _check_initial_loss(
            model, val_ds, len(data["x_val"]))
        summary_head = _summary_head(
            config, run_dir, data, params, steps_per_epoch, baseline, initial_loss_ratio,
            init_scale_warning, devices, data_load_seconds,
        )

        callbacks, _ = create_callbacks(
            model_name=config.experiment_name,
            results_dir_prefix="convunext_seg",
            run_dir=str(run_dir),
            monitor=MONITOR,
            patience=config.patience,
            use_lr_schedule=True,
            include_terminate_on_nan=True,
            include_analyzer=False,
        )
        # DECISION plan-2026-09-19T224205-49c8bf80/D-008: ``create_callbacks`` builds
        # ``EarlyStopping(restore_best_weights=True)`` and Keras 3.8 then restores the best
        # weights at EVERY train end, so ``final_model.keras`` would silently be the best
        # epoch. The best epoch is already on disk (``best_model.keras``), so the restore is
        # switched off and the in-memory model stays the real last epoch. Do NOT set it back
        # to True and do NOT copy ``_LastEpochWeights`` from the ConvNeXt trainer instead;
        # ``test_final_model_holds_the_last_epoch_weights_and_best_holds_the_best_epoch``
        # is the guard.
        for callback in callbacks:
            if isinstance(callback, keras.callbacks.EarlyStopping):
                callback.restore_best_weights = False
        # Index 0: ``lr`` must be in ``logs`` before CSVLogger reads it, and it is the rate
        # at the START of the epoch (the default reads the next epoch's first-step rate).
        callbacks.insert(0, LearningRateLogger(at_epoch_start=True))
        # After every callback that edits ``logs`` (the grid title reads ``val_miou``), before
        # the dashboard redraw.
        callbacks.append(EpochLogLine(EPOCH_LINE_KEYS))
        grid = SegmentationGridCallback(
            data["x_val"], data["y_val"], run_dir / "visualizations", CLASS_NAMES,
            viz_freq=config.viz_freq, viz_samples=config.viz_samples, seed=config.seed,
            title=config.experiment_name,
        )
        callbacks.append(grid)
        dashboard = TrainingDashboardCallback(
            out_path=run_dir / "visualizations" / "training_dashboard.png",
            baseline_fn=lambda _model: dict(baseline),
            title=f"{config.experiment_name} (seed {config.seed})",
            best_key=MONITOR,
        )
        callbacks.append(dashboard)

        fit_started = time.perf_counter()
        history = model.fit(
            train_ds, validation_data=val_ds, epochs=config.epochs, callbacks=callbacks, verbose=1,
        )
        fit_wall_seconds = time.perf_counter() - fit_started
        log_gpu_peak_memory()
        save_training_history_json(history, str(run_dir))
        hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        epochs_run = len(hist.get(MONITOR, []))
        non_finite = run_summary.non_finite_metrics(hist, MONITOR)

        if non_finite:
            # TerminateOnNaN ended the run: fewer epochs than requested is NOT an early stop.
            message = (
                f"Training diverged: {non_finite} hold a non-finite or missing value after "
                f"{epochs_run} epoch(s); no evaluation, figures or final_model.keras were "
                f"produced. Initial-loss ratio was {initial_loss_ratio:.3g}."
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
                    "(TerminateOnNaN), not by EarlyStopping",
                ],
            })
            raise RuntimeError(message)

        stopped_early = epochs_run < config.epochs
        if stopped_early:
            logger.info(
                f"EarlyStopping: stopped after epoch {epochs_run} of {config.epochs} "
                f"(patience {config.patience} on {MONITOR})"
            )
        best_epoch = run_summary.best_epoch(hist, MONITOR)
        best_i, final_i = best_epoch - 1, epochs_run - 1
        batch_size = config.batch_size

        # The test split is read for the first time HERE, after ``fit``.
        test_started = time.perf_counter()
        test_metrics_best, best_load_error, checkpoint_max_diff = _check_best_checkpoint(
            run_dir, data, batch_size, hist, best_i)
        # DECISION plan-2026-09-19T224205-49c8bf80/D-016: when the best epoch IS the last one the
        # reloaded ``best_model.keras`` holds the very weights of the in-memory model, so the
        # test split is scored ONCE and the result serves both keys (a second pass over the
        # 3669 test images cost 25 s of a 205 s audit run and printed identical digits). Do NOT
        # score twice "to be safe" and do NOT drop the ``test_metrics_final`` key: it stays
        # populated, ``final_reused_best`` says it is a copy. If the checkpoint did not load
        # there is no result to reuse, so the in-memory model is scored.
        final_reused_best = best_epoch == epochs_run and test_metrics_best is not None
        if final_reused_best:
            test_metrics_final = test_metrics_best
            logger.info(f"Test results (final weights, epoch {epochs_run}): same weights as best, scored once")
        else:
            test_metrics_final = _evaluate_split(model, data["x_test"], data["y_test"], batch_size)
            logger.info(f"Test results (final weights, epoch {epochs_run}): {test_metrics_final}")
        baseline_test = trivial_baseline(data["y_test"])
        logger.info(f"Majority-class baseline on the test masks: {baseline_test}")
        test_eval_seconds = time.perf_counter() - test_started

        final_path = run_dir / "final_model.keras"
        model.save(final_path)
        load_check: Optional[bool] = None
        try:
            sample = data["x_val"][:LOAD_CHECK_SAMPLES].astype(np.float32) / 255.0
            load_check = bool(validate_model_loading(
                str(final_path), sample, model.predict(sample, verbose=0)))
        except Exception as e:  # noqa: BLE001 - log-only
            logger.warning(f"validate_model_loading raised: {e}")

        # Both are isolated: nothing below can fail a run whose weights are already on disk.
        visualizations = _write_figures(run_dir, grid, model, hist, best_epoch, test_metrics_best, config)
        analyzer = run_summary.run_data_free_analysis(
            model, data["x_val"][:LOAD_CHECK_SAMPLES].astype(np.float32) / 255.0,
            data["y_val"][:LOAD_CHECK_SAMPLES], history, config.experiment_name, run_dir,
            enabled=config.model_analysis,
        )

        val_keys = [k for k in hist if k.startswith("val_")]
        steps_run = epochs_run * steps_per_epoch
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
            "lr_last_step": float(keras.ops.convert_to_numpy(schedule(steps_run - 1))),
            "best_val_metrics": {k: hist[k][best_i] for k in val_keys},
            "final_val_metrics": {k: hist[k][final_i] for k in val_keys},
            "test_metrics_best": test_metrics_best,
            "test_metrics_final": test_metrics_final,
            "final_reused_best": final_reused_best,
            "trivial_baseline": baseline_test,
            "visualizations": visualizations,
            "analyzer": analyzer,
            "test_eval_seconds": test_eval_seconds,
            "best_checkpoint_load_error": best_load_error,
            "best_checkpoint_max_abs_diff": checkpoint_max_diff,
            "epoch_times": list(dashboard.epoch_times),
            "fit_wall_seconds": fit_wall_seconds,
            "model_loading_validated": load_check,
            "notes": [
                f"initial loss {baseline['loss']:.4f} vs ln(3)={np.log(NUM_CLASSES):.4f} (ratio "
                f"{initial_loss_ratio:.2f}, warn above {INITIAL_LOSS_WARN_FACTOR:g}), a plain "
                "evaluate on the validation split (LayerNorm model)",
                "CSV `epoch` is 0-based, `best_epoch` is 1-based (`best_epoch_csv_index` = "
                "`best_epoch` - 1)",
                "CSV `lr` is the rate at the START of the epoch (its first step); `lr_last_step` "
                "is the rate of the final optimizer step",
                "`test_metrics_best` is the reloaded best_model.keras, `test_metrics_final` the "
                "last epoch's weights (final_model.keras); when `final_is_best` the test split "
                "is scored once and `final_reused_best` is true (the two blocks are the same "
                "result); the test split never influenced "
                "selection and is always the full split (--max-samples caps the train pool only)",
                "a test block is ONE pass over the split: `loss` is the compiled loss, `accuracy` "
                "= `pixel_accuracy` and `miou` = `miou_from_confusion` come from the confusion "
                "matrix (null where a class has no true and no predicted pixel), and a test "
                "checks them against keras `evaluate` to 1e-5; the validation metrics are keras' "
                "own; `trivial_baseline` predicts the majority test class everywhere",
                "`best_checkpoint_max_abs_diff` is the largest gap between the reloaded "
                "best_model.keras' validation loss/accuracy/miou and the values fit recorded "
                "for the best epoch",
                "`fit_wall_seconds` minus the sum of `epoch_times` is time outside the epoch "
                "clock (dashboard redraws, checkpoint saves)",
                "`visualizations.files` lists what exists on disk (dashboard, per-epoch grids on "
                "the same fixed validation samples, then the end-of-run figures); `failed` maps a "
                "figure that raised to its error and never fails the run. The confusion matrix, "
                "per-class chart and segmentation_report.json describe the TEST split with the "
                "BEST weights; best_vs_final_predictions.png uses fixed validation samples",
                *_analyzer_notes(config.model_analysis),
            ],
        }
        return write_summary_json(run_dir, summary)


# ---------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point. Parses ``argv`` FIRST so ``--help`` allocates nothing.

    Order (pinned by tests): parse, build and validate the config, refuse a reused
    experiment name, configure the GPU, then train.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv[1:]``.

    Raises:
        ValueError: If a config value is outside its range.
        FileExistsError: If the experiment directory already holds a run.
        RuntimeError: From :func:`train` (unreadable cache, diverged run).
    """
    args = parse_arguments(argv)
    config = config_from_args(args)
    resolve_new_run_dir(config)
    setup_gpu(gpu_id=args.gpu)
    try:
        train(config)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
