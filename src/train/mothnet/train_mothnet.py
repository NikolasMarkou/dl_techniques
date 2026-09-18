"""
MothNet Training Script
====================================================================

Trains `MothNet` (`dl_techniques.models.general_purpose.mothnet.model`) on a small
MNIST subsample via its hand-rolled `train_hebbian()` method — never `model.fit()`,
since `HebbianReadoutLayer.readout_weights` is `trainable=False` and is updated only
by a direct Hebbian `.assign()` inside `train_hebbian` (see
`plans/plan-2026-09-18T045308-c89cdf76/decisions.md` D-001/D-002).

This module currently only wires CLI argument parsing (plan Step 1). Data loading,
model construction, the training loop, and visualization land in later plan steps.

Usage:
    python -m train.mothnet.train_mothnet --help
    python -m train.mothnet.train_mothnet --epochs 3 --num-train-samples 2000 \
        --num-val-samples 500
"""

import argparse

import matplotlib
matplotlib.use("Agg")

from dl_techniques.utils.logger import logger


def parse_arguments(argv=None) -> argparse.Namespace:
    """Parse CLI arguments for the MothNet trainer.

    Built as a fresh, purpose-built `argparse.ArgumentParser` rather than
    `create_base_argument_parser()` — MothNet has no optimizer/LR-schedule/patience/
    dataset-choice surface for that shared parser to usefully cover
    (`decisions.md` D-003).

    :param argv: Argument list to parse; ``None`` defers to ``sys.argv[1:]``.
    :return: Parsed arguments namespace.
    """
    # DECISION plan-2026-09-18T045308-c89cdf76/D-003: fresh ArgumentParser, not
    # create_base_argument_parser(). That shared parser's surface (--dataset,
    # --image-size, --weight-decay, --lr-schedule, --patience) is entirely dead for
    # MothNet (no optimizer, no LR schedule, no early stopping, no image resize) —
    # do not "fix" this by reintroducing the shared parser + a pruning helper; see
    # decisions.md D-003 for the full trade-off.
    parser = argparse.ArgumentParser(
        description=(
            "Train MothNet on a small MNIST subsample via its hand-rolled "
            "train_hebbian() method."
        ),
    )
    parser.add_argument(
        "--mb-units", type=int, default=2000,
        help="Number of Mushroom Body units (default: 2000, matches MothNet's own default).",
    )
    parser.add_argument(
        "--mb-sparsity", type=float, default=0.1,
        help="Fraction of MB units allowed to fire per sample (default: 0.1).",
    )
    parser.add_argument(
        "--al-units", type=int, default=None,
        help=(
            "Number of Antennal Lobe units (default: None, meaning MothNet infers it "
            "from the input dimension)."
        ),
    )
    parser.add_argument(
        "--connection-sparsity", type=float, default=0.1,
        help="Mushroom Body projection connection sparsity (default: 0.1).",
    )
    parser.add_argument(
        "--hebbian-learning-rate", type=float, default=0.01,
        help="Learning rate used by the Hebbian readout update (default: 0.01).",
    )
    parser.add_argument(
        "--inhibition-strength", type=float, default=0.5,
        help="Antennal Lobe inhibition strength (default: 0.5).",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help=(
            "Number of training epochs; each epoch is one "
            "train_hebbian(epochs=1) call (default: 3, must be >= 1)."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Mini-batch size passed to train_hebbian (default: 32).",
    )
    parser.add_argument(
        "--viz-freq", type=int, default=2,
        help=(
            "Render a periodic MB-sparsity visualization every N epochs; "
            "<= 0 disables periodic visualization (default: 2)."
        ),
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU index to use (default: 0).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=(
            "Output directory for run artifacts (default: None, meaning "
            "repo-root results/, resolved later)."
        ),
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None,
        help=(
            "Experiment name (default: None, meaning auto-generated via "
            "default_experiment_name)."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for NumPy's global RNG (default: 42).",
    )
    parser.add_argument(
        "--num-train-samples", type=int, default=2000,
        help="Number of MNIST training samples to subsample (default: 2000).",
    )
    parser.add_argument(
        "--num-val-samples", type=int, default=500,
        help="Number of MNIST validation samples to subsample (default: 500).",
    )

    args = parser.parse_args(argv)

    if args.epochs < 1:
        parser.error(f"--epochs must be >= 1, got {args.epochs}")

    return args


def main(argv=None) -> int:
    """Entry point for the MothNet trainer.

    The first statement parses argv, so ``--help`` exits 0 with a ``usage:`` line
    and no GPU/dataset/model work happens before parsing returns.

    :param argv: Argument list to parse; ``None`` defers to ``sys.argv[1:]``.
    :return: Process exit code.
    """
    args = parse_arguments(argv)

    logger.info(f"Parsed args: {args}")
    return 0


if __name__ == "__main__":
    main()
