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

    **What the scan DOES see**: ``--flag value``; ``--flag=value``; both
    ``BooleanOptionalAction`` spellings; and — when the parser allows
    abbreviation, which is argparse's default — an UNAMBIGUOUS long-option
    PREFIX such as ``--ema-warmup-ep`` for ``--ema-warmup-epochs``, resolved
    the way argparse itself resolves it (exact match first, then a prefix
    matching exactly one registered long option), at either the spaced or the
    ``=`` form.

    **What the scan does NOT see**, deliberately:

    * Tokens after a bare ``--`` separator. argparse treats those as
      positionals, so a positional whose text happens to equal an option
      string must NOT be reported as an explicit flag. Scanning stops there.
    * An AMBIGUOUS prefix (matching two or more registered long options).
      argparse rejects such a token with "ambiguous option", so it cannot
      appear in an argv that parsed successfully; it is reported as not-typed
      rather than guessed.
    * Any abbreviation at all when the parser was built with
      ``allow_abbrev=False`` — then a prefix is an argparse ERROR, so counting
      it as typed would be wrong. The flag is READ from the parser.
    * Options that are not long (``--``) spellings — and rather than answer
      wrongly, the scan REFUSES them: a parser registering one raises
      ``ValueError``. argparse resolves single-dash tokens through a
      short-option code path this scan does not reimplement, so an ATTACHED
      value (``-b8``) or a GROUPED run of flags (``-vb 8``) parses fine and is
      reported as not-typed, which is verbatim the regression D-016 exists to
      prevent. Exactly one exemption: ``-h`` bound to argparse's own HELP
      ACTION, which IS reported (as dest ``help``) — harmlessly, because that
      action prints and exits, so ``parse_args`` never returns and no caller
      ever reads the report. The exemption is keyed on the ACTION, not on the
      string: a parser that binds ``-h`` to something else (``add_help=False``
      plus ``-h/--horizon``) is REFUSED like any other short option.
    * A VALUE that happens to spell a registered option (``--name --smoke``).
      Distinguishing that needs the parser's nargs/type machinery, which this
      token scan deliberately does not reimplement. Unreachable in practice:
      argparse rejects that argv before the answer can matter.

    Change a branch here and change the list above in the same edit — this
    helper is shared by three trainers and that list is the contract they
    rely on.

    Args:
        parser: The fully-populated parser whose actions define the recognised
            option strings. Pass it BEFORE or AFTER ``parse_args``; only the
            registered actions and ``parser.allow_abbrev`` are read, never the
            parse result.
        argv: Token list to scan, WITHOUT the program name (the same list shape
            ``parse_args`` accepts). ``None`` reads ``sys.argv[1:]``, which is
            what a production entry point wants; tests pass an explicit list.

    Returns:
        The set of argparse ``dest`` names mentioned in ``argv``. Dest names,
        not flag spellings — so a caller can test membership against dataclass
        field names via its own rename map.

    Raises:
        ValueError: If ``parser`` registers any option string that is not a
            long (``--``) spelling, whose attached and grouped forms this scan
            cannot see. The sole exemption is ``-h`` bound to argparse's HELP
            ACTION — an action-type test, not a string test, because a parser
            may bind the string ``-h`` to an ordinary value-taking option.
    """
    # DECISION plan-2026-08-03T043010-cecf4357/D-016
    # Resolve each token the way argparse resolves it, INCLUDING unambiguous
    # long-option prefixes. Do NOT go back to a plain `token in dest_by_opt`
    # membership test: argparse accepts `--ema-warmup-ep 1.5` as
    # `--ema-warmup-epochs`, so a literal-only scan reports "not typed" for a
    # flag the caller really typed and a `--smoke`-style preset then silently
    # overrides it. Do NOT "fix" it instead by building the parser with
    # allow_abbrev=False — that removes an abbreviation users can already use.
    # `allow_abbrev` is READ, not assumed: with abbreviation off, a prefix is
    # an argparse ERROR, so it must not count as explicit. See decisions.md
    # D-016 for the reproduction of the regression this repairs.
    dest_by_opt = {}
    for action in parser._actions:
        for opt in action.option_strings:
            dest_by_opt[opt] = action.dest

    # DECISION plan-2026-08-03T043010-cecf4357/D-021
    # REFUSE rather than answer wrongly. MEASURED: argparse ACCEPTS `-b8` as
    # {'batch_size': 8} and `-vb 8` as {'verbose': True, 'batch_size': 8}, and
    # the loop below sees NEITHER — it reports set() for a flag the caller
    # really typed, which is exactly the D-016 regression. Do NOT "fix" this by
    # reimplementing argparse's short-option tokenizer (attached values,
    # grouped flags, single-dash prefixes): that is a chunk of real surface
    # with its own bugs, for a shape no current consumer uses.
    #
    # DECISION plan-2026-08-03T043010-cecf4357/D-023
    # The `-h` exemption is keyed on the ACTION TYPE, never on the STRING. Do
    # NOT simplify this back to `o != "-h"`: MEASURED, `ArgumentParser(
    # add_help=False)` + `add_argument("-h", "--horizon", type=int)` parses
    # `["-h8"]` as {'horizon': 8} while a string-keyed refusal accepts the
    # parser and reports set() — the guard FAILING OPEN on the one string it
    # exempts, i.e. the very D-016 regression it exists to stop. What makes
    # `-h` safe is not its spelling but that argparse's help action prints and
    # exits, so no caller ever reads the report; `isinstance` asks exactly
    # that. `parser._actions` is iterated (not `dest_by_opt`) so a
    # `conflict_handler="resolve"` parser, where `-h` moves from the help
    # action to another one, is seen too. See decisions.md D-023.
    unscannable = sorted(
        o for a in parser._actions for o in a.option_strings
        if not o.startswith("--")
        and not (o == "-h" and isinstance(a, argparse._HelpAction)))
    if unscannable:
        raise ValueError(
            f"explicitly_set_flags cannot scan the non-long options "
            f"{unscannable}: this token scan resolves only long (--) "
            f"spellings, and argparse accepts attached (-b8) and grouped "
            f"(-vb 8) single-dash forms it does not see, so it would silently "
            f"report a typed flag as not-typed. Give the flag a long "
            f"spelling, or extend the scan. See decisions.md D-021, D-023."
        )

    tokens = sys.argv[1:] if argv is None else list(argv)
    allow_abbrev = getattr(parser, "allow_abbrev", True)

    explicit: Set[str] = set()
    for token in tokens:
        if token == "--":
            break
        key = token.split("=", 1)[0]
        if key in dest_by_opt:
            explicit.add(dest_by_opt[key])
        elif allow_abbrev and key.startswith("--") and len(key) > 2:
            matches = [opt for opt in dest_by_opt if opt.startswith(key)]
            if len(matches) == 1:
                explicit.add(dest_by_opt[matches[0]])
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
