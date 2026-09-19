"""
Orchestrator of the ConvUNext segmentation trainer (part A: config, parser, refusals).

This package trains ``create_convunext`` (``use_bias=True``, ``output_channels=3``, linear
head) as a semantic segmenter on Oxford-IIIT Pet and writes the run directory of
``src/train/convnext/`` (``config.json``, ``run.log``, ``training_log.csv``,
``training_history.json``, ``best_model.keras``, ``final_model.keras``,
``results_summary.json``, ``visualizations/``, ``model_analysis/``). It is a sibling of
``train.convnext`` and imports every generic piece from ``train.common``; the bias-free
ConvUNeXt DENOISER stays in ``train.bfunet``. Data, ``train()``, figures and the analyzer
land in later steps; this module holds what those steps read: :class:`SegTrainingConfig`,
the parser, and the refusals that must happen before a run directory or a GPU exists.

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

``--help`` parses first, so it allocates no GPU and no directory. ``--gpu`` is not a config
field: ``main`` hands it to ``setup_gpu``, which OVERWRITES an exported
``CUDA_VISIBLE_DEVICES``.
"""

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Sequence, Tuple

from dl_techniques.models.vision.convunext.model import CONVUNEXT_CONFIGS

from train.common import (
    default_experiment_name,
    refuse_existing_run,
    resolved_run_dir,
    setup_gpu,
)
# The split arithmetic and the "fit split holds at least one batch" rule are the same
# generic rules the ConvNeXt trainer states once; reused, not re-implemented.
from train.convnext.common import split_sizes, steps_per_epoch_for


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# Train-split size of ``oxford_iiit_pet`` 4.0.0 (measured from the TFDS ``dataset_info``:
# train 3680, test 3669). Used ONLY to refuse an impossible configuration before a run
# directory exists; the data loader sizes the split from the array it really loads.
OXFORD_PET_TRAIN_SIZE = 3680

VARIANTS: Tuple[str, ...] = tuple(CONVUNEXT_CONFIGS)

# Largest 32-bit unsigned value: ``numpy.random.seed`` takes 0 .. 2**32 - 1.
MAX_SEED = 2 ** 32 - 1


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
                           "samples (smoke runs and tests); default: the whole train set.")

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


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point. Parses ``argv`` FIRST so ``--help`` allocates nothing.

    Order (pinned by tests): parse, build and validate the config, refuse a reused
    experiment name, configure the GPU, then train. ``train()`` lands in step 3; until then
    the run stops with ``NotImplementedError`` after the GPU is configured and before any
    directory exists.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv[1:]``.

    Raises:
        ValueError: If a config value is outside its range.
        FileExistsError: If the experiment directory already holds a run.
        NotImplementedError: Always, until ``train()`` exists.
    """
    args = parse_arguments(argv)
    config = config_from_args(args)
    resolve_new_run_dir(config)
    setup_gpu(gpu_id=args.gpu)
    raise NotImplementedError("train() lands in step 3")
