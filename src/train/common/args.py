"""Common argument parser utilities for training scripts."""

import sys
import argparse
from typing import Optional, List, Sequence, Set

from dl_techniques.datasets.time_series import TimeSeriesGeneratorConfig


# ---------------------------------------------------------------------

def explicitly_set_flags(
        parser: argparse.ArgumentParser,
        argv: Optional[Sequence[str]] = None,
) -> Set[str]:
    """Record which argparse **dest** names the caller passed explicitly.

    Scans the raw token list against the parser's registered option strings, so
    it reports PROVENANCE ("was this flag typed?") rather than VALUE ("does this
    differ from the default?").

    Why it exists: a ``--smoke``-style preset must lose to any flag the caller
    actually typed, INCLUDING one typed at the flag's own parser default. A
    parsed-value-vs-default comparison structurally cannot express that — an
    explicitly-passed default and an omission produce the identical Namespace.
    Scanning the tokens is the only place the distinction still survives.

    ``argparse.BooleanOptionalAction`` registers BOTH ``--x`` and ``--no-x`` on
    one dest, and this function iterates ``action.option_strings`` (plural), so
    either spelling counts as an explicit mention of that dest. The
    ``--flag=value`` equals form is handled by splitting on the first ``=``.

    Args:
        parser: The fully-populated parser whose actions define the recognised
            option strings. Pass it BEFORE or AFTER ``parse_args``; only the
            registered actions are read, never the parse result.
        argv: Token list to scan, WITHOUT the program name (the same list shape
            ``parse_args`` accepts). ``None`` reads ``sys.argv[1:]``, which is
            what a production entry point wants; tests pass an explicit list.

    Returns:
        The set of argparse ``dest`` names mentioned in ``argv``. Dest names,
        not flag spellings — so a caller can test membership against dataclass
        field names via its own rename map.
    """
    dest_by_opt = {}
    for action in parser._actions:
        for opt in action.option_strings:
            dest_by_opt[opt] = action.dest

    tokens = sys.argv[1:] if argv is None else list(argv)

    explicit: Set[str] = set()
    for token in tokens:
        key = token.split("=", 1)[0]
        if key in dest_by_opt:
            explicit.add(dest_by_opt[key])
    return explicit


# ---------------------------------------------------------------------

def create_base_argument_parser(
        description: str = "Train model",
        default_dataset: str = "cifar10",
        dataset_choices: Optional[List[str]] = None,
) -> argparse.ArgumentParser:
    """Create argument parser with standard training arguments.

    Scripts extend this parser with model-specific arguments:

        parser = create_base_argument_parser("Train MyModel")
        parser.add_argument('--variant', type=str, default='tiny')
        args = parser.parse_args()

    Args:
        description: Help text for the parser.
        default_dataset: Default dataset name.
        dataset_choices: Valid dataset names. Defaults to
            ['mnist', 'cifar10', 'cifar100', 'imagenet'].

    Returns:
        ArgumentParser with common training arguments.
    """
    if dataset_choices is None:
        dataset_choices = ['mnist', 'cifar10', 'cifar100', 'imagenet']

    parser = argparse.ArgumentParser(description=description)

    # Data
    parser.add_argument('--dataset', type=str, default=default_dataset,
                        choices=dataset_choices, help='Dataset to use')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size (for ImageNet, default: 224)')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')

    # Optimization
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--lr-schedule', type=str, default='cosine',
                        choices=['cosine', 'exponential', 'constant'],
                        help='Learning rate schedule')

    # Early stopping
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')

    # GPU
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device index to use (default: all GPUs)')

    # Output
    parser.add_argument('--show-plots', action='store_true', default=False,
                        help='Show plots interactively')

    return parser


# ---------------------------------------------------------------------

def create_ts_argument_parser(description: str) -> argparse.ArgumentParser:
    """Create argument parser with the shared time-series training arguments.

    Consolidates the argparse block duplicated across the synthetic
    time-series trainers (mdn, nbeats, prism, tirex, deepar, xlstm,
    adaptive_ema). Scripts extend the returned parser with
    architecture-specific arguments before parsing:

        parser = create_ts_argument_parser("Train MyTSModel")
        parser.add_argument('--preset', type=str, default='small')
        args = parser.parse_args()

    Flag style is UNDERSCORE (``--batch_size``, ``--steps_per_epoch``) to match
    the scripts' existing CLI. The warmup toggle is ``--no-warmup`` (store_false
    into ``use_warmup``, default True); the deep-analysis toggle is
    ``--no-deep-analysis`` (store_false into ``perform_deep_analysis``, default
    True) — both reproducing the scripts' exact mechanism.

    Defaults are the most-common value across the four scripts. Where they
    differ (``epochs`` 100/200/150/200 → 200; ``batch_size`` 256/128/64/128 →
    128; ``steps_per_epoch`` 1000/500 → 1000; ``learning_rate`` 5e-4/1e-4 →
    1e-4; ``max_patterns_per_category`` 10/100 → 10) a script that needs a
    different default re-adds the argument after this call (argparse keeps the
    last registration's default), or passes the value explicitly.

    Args:
        description: Help text for the parser.

    Returns:
        ArgumentParser with the shared TS training arguments (NOT parsed args),
        so callers can ``.add_argument(...)`` arch-specific flags then
        ``.parse_args()``.
    """
    parser = argparse.ArgumentParser(description=description)

    # Experiment
    parser.add_argument("--experiment_name", type=str, default="timeseries",
                        help="Experiment name / results-dir prefix")
    parser.add_argument("--result_dir", type=str, default="results",
                        help="Root output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Synthetic data generation
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Number of synthetic series samples to generate")
    parser.add_argument("--noise_level", type=float, default=0.1,
                        help="Default noise level for the synthetic generator")

    # Training
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--steps_per_epoch", type=int, default=1000,
                        help="Steps per epoch")

    # Optimization
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0,
                        help="Gradient clipping max-norm")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        help="Optimizer name")

    # Warmup (toggle off with --no-warmup)
    parser.add_argument("--no-warmup", dest="use_warmup", action="store_false",
                        help="Disable learning-rate warmup")
    parser.set_defaults(use_warmup=True)
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps")
    parser.add_argument("--warmup_start_lr", type=float, default=1e-6,
                        help="Warmup starting learning rate")

    # Visualization
    parser.add_argument("--max_patterns_per_category", type=int, default=10,
                        help="Max patterns selected per category")
    parser.add_argument("--visualize_every_n_epochs", type=int, default=5,
                        help="Epoch frequency for per-epoch visualization")
    parser.add_argument("--plot_top_k_patterns", type=int, default=12,
                        help="Number of top patterns to plot")

    # Deep analysis (toggle off with --no-deep-analysis)
    parser.add_argument("--no-deep-analysis", dest="perform_deep_analysis",
                        action="store_false",
                        help="Disable the deep ModelAnalyzer callback")
    parser.set_defaults(perform_deep_analysis=True)
    parser.add_argument("--analysis_frequency", type=int, default=10,
                        help="Epoch frequency for the deep-analysis callback")
    parser.add_argument("--analysis_start_epoch", type=int, default=1,
                        help="Epoch at which the deep-analysis callback starts")

    # GPU
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU device index to use (default: all GPUs)")

    return parser


# ---------------------------------------------------------------------

def build_generator_config(args: argparse.Namespace) -> TimeSeriesGeneratorConfig:
    """Build a ``TimeSeriesGeneratorConfig`` from parsed TS training args.

    Deduplicates the copy-pasted ``TimeSeriesGeneratorConfig(...)`` triple
    previously scattered across the synthetic time-series trainers, wiring the
    shared CLI flags (``--n_samples``, ``--seed``, ``--noise_level``) into the
    generator config's corresponding fields.

    Args:
        args: Parsed namespace from a parser produced by
            ``create_ts_argument_parser`` (must carry ``n_samples``, ``seed``,
            and ``noise_level`` attributes).

    Returns:
        A ``TimeSeriesGeneratorConfig`` populated from ``args``.
    """
    return TimeSeriesGeneratorConfig(
        n_samples=args.n_samples,
        random_seed=args.seed,
        default_noise_level=args.noise_level,
    )
