r"""Train DocScanner's LOCALIZATION module (stage 1) through stock ``fit()``.

Usage::

    # what the CLI offers -- allocates nothing, touches no GPU
    MPLBACKEND=Agg .venv/bin/python -m train.doc_scanner.train_doc_scanner_segmenter --help

    # the synthetic corpus needs no download
    MPLBACKEND=Agg .venv/bin/python -m train.doc_scanner.train_doc_scanner_segmenter \
        --epochs 2 --steps-per-epoch 10 --gpu 1

The paper trains the two modules INDEPENDENTLY (arXiv:2110.14968v2 §4.3), so
this script trains ONLY the U2NET-P segmenter, against the page mask, with the
paper's own recipe: binary cross-entropy on all seven deep-supervision heads,
Adam, batch 32, learning rate 1e-4 dropped by 0.1x after 30 epochs, converging
by 45. Its twin is ``train_doc_scanner_rectifier.py``.

This module is an ENTRY POINT and nothing else. The config, the flag surface,
the data gate, the ``tf.data`` pipeline, the loss, the optimizer, the callbacks
and ``fit()`` all live in :mod:`train.doc_scanner.common`, so there is exactly
one place to read and exactly one place a defect can hide. The two entry points
differ ONLY in the ``stage`` they bind.

What this file owes, and why each piece is where it is
------------------------------------------------------

``main()`` PARSES FIRST.
    ``args, config = parse_arguments(argv)`` is the first statement, so
    ``--help`` prints a ``usage:`` line and exits without claiming a GPU,
    building a model or walking a corpus. ``src/train/CLAUDE.md`` states the
    trap this avoids: **exit 0 is not a passing** ``--help`` -- a script with
    no parser at all ignores ``--help``, runs its whole job and exits 0 anyway,
    so ``tests/test_train/test_doc_scanner/test_cli_contract.py`` asserts the
    ``usage:`` line and asserts, through sentinels, that nothing expensive was
    reached.

NOTHING EAGER AT MODULE SCOPE.
    No ``tf.`` call runs at import time. An eager op here would initialise the
    eager context -- and with it a GPU -- for every importer, ``--help``
    included.

``require_training_data`` RUNS BEFORE ``setup_gpu``.
    The synthetic generator warps REAL page rasters and has no page content of
    its own, and the UVDoc corpus is a 27.5 GB download. Asking for either
    without it staged must produce that reason, not an empty-glob mystery and
    not a GPU claimed for a run that cannot start.

``--stage`` IS NOT A FLAG.
    The stage is fixed by which script you ran. A ``--stage`` flag would let
    this script train the OTHER module under this script's name, this script's
    recipe and this script's run-directory prefix.

``--gpu`` IS NOT A CONFIG FIELD.
    It acts on the process, is consumed by ``setup_gpu`` and never reaches
    :class:`~train.doc_scanner.common.DocScannerTrainingConfig`.
    :data:`NON_CONFIG_DESTS` is the single, checked exemption from the "every
    parser dest is a config field" contract.
"""

from __future__ import annotations

import argparse
from typing import FrozenSet, Optional, Sequence, Tuple

from dl_techniques.utils.logger import logger
from train.common import setup_gpu
from train.doc_scanner.common import (
    STAGE_SEGMENTER,
    DocScannerTrainingConfig,
    add_common_arguments,
    config_from_args,
    require_training_data,
    train,
)

__all__ = [
    "NON_CONFIG_DESTS",
    "PROGRAM_NAME",
    "STAGE",
    "build_parser",
    "config_from_argv",
    "main",
    "parse_arguments",
]


PROGRAM_NAME: str = "train_doc_scanner_segmenter.py"
"""``prog`` for the parser, so ``--help`` names the script rather than
``__main__`` when the module is run with ``python -m``."""

STAGE: str = STAGE_SEGMENTER
"""The one thing this file decides. Everything else is ``common``'s."""

NON_CONFIG_DESTS: FrozenSet[str] = frozenset({"gpu"})
"""Parser destinations that are deliberately NOT config fields.

The ONLY permitted way out of the flag-to-field contract.
``tests/test_train/test_doc_scanner/test_cli_contract.py`` asserts both that
every other dest maps to a :class:`DocScannerTrainingConfig` field and that
every name listed here really is not one -- a stale exemption silently widens
the carve-out."""


def build_parser() -> argparse.ArgumentParser:
    """Build the real parser: the common flags at the segmenter's defaults.

    Interface contract: pure. Constructs and returns a parser; parses nothing,
    reads no environment and touches no filesystem. It is the single parser
    both :func:`parse_arguments` and the CLI contract guard drive, so there is
    no parser a test can pass against while the trainer uses another.

    :return: The parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description=(
            "Train DocScanner's localization module (U2NET-P) with stock "
            "Keras fit(). Stage 1 of two; the rectifier trains separately "
            "with `python -m train.doc_scanner.train_doc_scanner_rectifier`."
        ),
    )
    add_common_arguments(parser, STAGE)
    parser.add_argument(
        "--gpu", type=int, default=None,
        help=(
            "GPU index to use (e.g. 1). Omit to let the process see whatever "
            "CUDA_VISIBLE_DEVICES exposes. Not a config field."
        ),
    )
    return parser


def parse_arguments(
        argv: Optional[Sequence[str]] = None,
) -> Tuple[argparse.Namespace, DocScannerTrainingConfig]:
    """Run the full ``argv -> parse -> config`` path.

    :param argv: Tokens without the program name. ``None`` reads
        ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    :return: ``(namespace, config)``.
    :rtype: Tuple[argparse.Namespace, DocScannerTrainingConfig]
    :raises SystemExit: As argparse does, on ``--help`` or a bad flag.
    :raises ValueError: From ``DocScannerTrainingConfig.__post_init__`` on a
        knob combination the trainer cannot honour.
    """
    args = build_parser().parse_args(argv)
    return args, config_from_args(args, STAGE)


def config_from_argv(
        argv: Optional[Sequence[str]] = None,
) -> DocScannerTrainingConfig:
    """The config half of :func:`parse_arguments`.

    :param argv: Tokens without the program name.
    :type argv: Optional[Sequence[str]]
    :return: The validated config.
    :rtype: DocScannerTrainingConfig
    """
    return parse_arguments(argv)[1]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse the CLI, check the corpus exists, set the process up, and train.

    The statement ORDER is the contract, and it is what the CLI guard's
    sentinels measure:

    1. parse -- so ``--help`` costs nothing;
    2. ``require_training_data`` -- so a missing corpus fails with the real
       reason before a GPU is claimed;
    3. ``setup_gpu`` -- process-level, from the non-config ``--gpu`` dest;
    4. ``train`` -- which seeds, builds the datasets, builds the model and
       calls stock ``fit()``.

    :param argv: Tokens without the program name. ``None`` reads
        ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    :raises MissingTrainingDataError: If the selected corpus is not staged,
        naming why.
    """
    args, config = parse_arguments(argv)

    require_training_data(config)

    setup_gpu(gpu_id=args.gpu)

    logger.info(
        "DocScanner segmenter: training variant %r on the %r corpus",
        config.model_variant, config.data_source,
    )
    _, _, results_dir = train(config)
    logger.info(
        "DocScanner segmenter training complete; artifacts in %s", results_dir
    )


if __name__ == "__main__":
    main()
