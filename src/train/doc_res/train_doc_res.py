r"""Train DocRes on ONE document-restoration task through stock ``fit()``.

Usage::

    # what the CLI offers -- allocates nothing, touches no GPU
    MPLBACKEND=Agg .venv/bin/python -m train.doc_res.train_doc_res --help

    # the only task with a staged corpus today (see ``README.md``)
    MPLBACKEND=Agg .venv/bin/python -m train.doc_res.train_doc_res \
        --task binarization --epochs 2 --steps-per-epoch 50 --gpu 1

This module is an ENTRY POINT and nothing else. Every piece of behaviour --
the config, the flag surface, the path worklist, the tf.data pipeline, the
per-task loss, the optimizer, the callbacks and ``fit()`` itself -- lives in
:mod:`train.doc_res.common`, so there is exactly one place to read and exactly
one place a defect can hide. Anything substantive added here would be a second
home for logic that the pipeline tests already cover through ``common``.

What this file owes, and why each piece is where it is
------------------------------------------------------

``main()`` PARSES FIRST.
    ``args, config = parse_arguments(argv)`` is the first statement, so
    ``--help`` prints a ``usage:`` line and exits without claiming a GPU,
    building a model or walking the staging volume.
    ``src/train/CLAUDE.md`` states the trap this avoids: **exit 0 is not a
    passing** ``--help``. A script with no parser at all ignores ``--help``,
    runs its whole job and exits 0 anyway, so
    ``tests/test_train/test_doc_res/test_cli_contract.py`` asserts the
    ``usage:`` line and asserts, through sentinels, that nothing expensive was
    reached.

NOTHING EAGER AT MODULE SCOPE.
    No ``tf.`` call runs at import time. An eager op here would initialise the
    eager context -- and with it a GPU -- for every importer, ``--help``
    included.

``require_task_data`` RUNS BEFORE ``setup_gpu``.
    Four of the five DocRes tasks have no staged training corpus (a
    registration gate, a dead host, a Drive folder that cannot be paged
    through -- ``prepare_doc_res_data.MANIFEST`` records the real reason for
    each). Asking for one of them must produce that reason, not an empty-glob
    mystery and not a GPU claimed for a run that cannot start. The check is the
    SAME function ``common.collect_task_triplets`` calls -- one implementation,
    called from two places -- and it is idempotent and read-only; calling it
    here only moves the failure ahead of the GPU.

SEEDING IS NOT REPEATED HERE.
    ``common.train`` calls ``set_seeds(config.seed)`` as its first statement
    and nothing in ``main()`` consumes randomness before that, so a second
    ``set_seeds`` call in this file would be a decorative duplicate of the one
    site that matters.

``--gpu`` IS NOT A CONFIG FIELD.
    It acts on the process, is consumed by ``setup_gpu`` and never reaches
    :class:`~train.doc_res.common.DocResTrainingConfig`. It is declared here
    rather than in ``add_common_arguments`` for that reason, and
    :data:`NON_CONFIG_DESTS` is the single, checked exemption from the
    "every parser dest is a config field" contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import FrozenSet, Optional, Sequence, Tuple

from dl_techniques.utils.logger import logger
from train.common import setup_gpu
from train.doc_res.common import (
    DocResTrainingConfig,
    add_common_arguments,
    config_from_args,
    require_task_data,
    train,
)

__all__ = [
    "NON_CONFIG_DESTS",
    "build_parser",
    "config_from_argv",
    "main",
    "parse_arguments",
]


PROGRAM_NAME: str = "train_doc_res.py"
"""``prog`` for the parser, so ``--help`` names the script rather than
``__main__`` when the module is run with ``python -m``."""

NON_CONFIG_DESTS: FrozenSet[str] = frozenset({"gpu"})
"""Parser destinations that are deliberately NOT config fields.

The ONLY permitted way out of the flag-to-field contract.
``tests/test_train/test_doc_res/test_cli_contract.py`` asserts both that every
other dest maps to a :class:`DocResTrainingConfig` field and that every name
listed here really is not one -- a stale exemption silently widens the
carve-out."""


def build_parser() -> argparse.ArgumentParser:
    """Build the real parser: the common flags plus ``--gpu``.

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
            "Train DocRes (DTSPrompt + Restormer) on ONE document-restoration "
            "task with stock Keras fit(). Stage a corpus first with "
            "`python -m train.doc_res.prepare_doc_res_data`."
        ),
    )
    add_common_arguments(parser)
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
) -> Tuple[argparse.Namespace, DocResTrainingConfig]:
    """Run the full ``argv -> parse -> config`` path.

    Interface contract: the single entry point :func:`main` uses AND the single
    entry point the CLI guard drives. The namespace is returned only so
    :func:`main` can read the process-level dests in :data:`NON_CONFIG_DESTS`;
    every other value arrives through the config.

    :param argv: Tokens without the program name. ``None`` reads
        ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    :return: ``(namespace, config)``.
    :rtype: Tuple[argparse.Namespace, DocResTrainingConfig]
    :raises SystemExit: As argparse does, on ``--help`` or a bad flag.
    :raises ValueError: From ``DocResTrainingConfig.__post_init__`` on a knob
        combination the trainer cannot honour.
    """
    args = build_parser().parse_args(argv)
    return args, config_from_args(args)


def config_from_argv(
    argv: Optional[Sequence[str]] = None,
) -> DocResTrainingConfig:
    """The config half of :func:`parse_arguments`.

    :param argv: Tokens without the program name.
    :type argv: Optional[Sequence[str]]
    :return: The validated config.
    :rtype: DocResTrainingConfig
    """
    return parse_arguments(argv)[1]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse the CLI, check the corpus exists, set the process up, and train.

    The statement ORDER is the contract, and it is what the CLI guard's
    sentinels measure:

    1. parse -- so ``--help`` costs nothing;
    2. ``require_task_data`` -- so a task with no corpus fails with the real
       reason before a GPU is claimed;
    3. ``setup_gpu`` -- process-level, from the non-config ``--gpu`` dest;
    4. ``train`` -- which seeds, builds the datasets, builds the model and
       calls stock ``fit()``.

    :param argv: Tokens without the program name. ``None`` reads
        ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    :raises MissingTrainingDataError: If the requested task has no staged
        training corpus, naming why.
    """
    args, config = parse_arguments(argv)

    require_task_data(Path(config.dataset_root), config.task)

    setup_gpu(gpu_id=args.gpu)

    logger.info(
        "DocRes: training task %r (variant %r) from %s",
        config.task, config.model_variant, config.dataset_root,
    )
    _, _, results_dir = train(config)
    logger.info(
        "DocRes %s training complete; artifacts in %s", config.task, results_dir
    )


if __name__ == "__main__":
    main()
