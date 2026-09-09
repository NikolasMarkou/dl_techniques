r"""Pretrain H-Net on raw Wikipedia bytes through stock ``fit()``.

Usage::

    # what the CLI offers -- allocates nothing, touches no GPU
    MPLBACKEND=Agg .venv/bin/python -m train.hnet.train_hnet --help

    # the dev-scale smoke layout on the already-staged Wikipedia dump
    MPLBACKEND=Agg .venv/bin/python -m train.hnet.train_hnet \
        --arch-variant dev --seq-len 512 --batch-size 8 \
        --epochs 1 --steps-per-epoch 200 --gpu 0

This module is an ENTRY POINT and nothing else. The config, the flag surface,
the byte pipeline, the optimizer, the model construction, the callbacks and
``fit()`` itself all live in :mod:`train.hnet.common`, so there is exactly one
place to read and exactly one place a defect can hide. Anything substantive
added here would be a second home for logic the pipeline tests already cover
through ``common``.

What this file owes, and why each piece is where it is
------------------------------------------------------

``main()`` PARSES FIRST.
    ``args, config = parse_arguments(argv)`` is the first statement, so
    ``--help`` prints a ``usage:`` line and exits without claiming a GPU,
    seeding the process, opening the Arrow cache or constructing a model.
    ``src/train/CLAUDE.md`` states the trap this avoids: **exit 0 is not a
    passing** ``--help``. A script with no parser at all ignores ``--help``,
    runs its whole job and exits 0 anyway, so
    ``tests/test_train/test_hnet/test_cli_contract.py`` asserts the ``usage:``
    line and asserts, through sentinels, that nothing expensive was reached.

NOTHING EAGER AT MODULE SCOPE.
    No ``tf.`` call runs at import time -- this module does not import
    ``tensorflow`` at all. An eager op here would initialise the eager context
    and with it a GPU device, for every importer, ``--help`` included
    (``src/train/CLAUDE.md``).

THE FILE IS ``train_hnet.py``, NEVER ``train.py``.
    A module named ``train.py`` inside a package on ``sys.path`` shadows the
    ``train`` package itself and breaks ``from train.common import ...``
    (``src/train/CLAUDE.md`` § File naming).

SEEDING IS NOT REPEATED HERE.
    ``common.train`` calls ``set_seeds(config.seed)`` as its first statement
    and nothing in ``main()`` consumes randomness before that, so a second
    ``set_seeds`` call here would be a decorative duplicate of the one site
    that matters.

``--gpu`` IS NOT A CONFIG FIELD.
    It acts on the process, is consumed by ``setup_gpu`` and never reaches
    :class:`~train.hnet.common.HNetTrainingConfig`. It is declared here rather
    than in ``add_common_arguments`` for that reason, and
    :data:`NON_CONFIG_DESTS` is the single, checked exemption from the "every
    parser dest is a config field" contract.

Scale, honestly
---------------
The measured cost of this architecture on this hardware is in ``README.md``
and is not encouraging: the Mamba-2 mixer runs a sequential scan, and one bare
block at ``L=8192, batch=8`` costs 38.7 s per training step on an RTX 4090
(plan ``decisions.md`` D-009). Byte-level context lengths in the reference's
regime are out of reach here regardless of card. ``--arch-variant dev`` exists
for that reason; the six reference variants are constructible and trainable in
principle, and none of them has ever been trained in this repository.
"""

from __future__ import annotations

import argparse
from typing import FrozenSet, Optional, Sequence, Tuple

from dl_techniques.utils.logger import logger
from train.common import setup_gpu
from train.hnet.common import (
    HNetTrainingConfig,
    add_common_arguments,
    arch_variant_names,
    config_from_args,
    train,
)

__all__ = [
    "NON_CONFIG_DESTS",
    "PROGRAM_NAME",
    "build_parser",
    "config_from_argv",
    "main",
    "parse_arguments",
]


PROGRAM_NAME: str = "train_hnet.py"
"""``prog`` for the parser, so ``--help`` names the script rather than
``__main__`` when the module is run with ``python -m``."""

NON_CONFIG_DESTS: FrozenSet[str] = frozenset({"gpu"})
"""Parser destinations that are deliberately NOT config fields.

The ONLY permitted way out of the flag-to-field contract.
``tests/test_train/test_hnet/test_cli_contract.py`` asserts both that every
other dest maps to an :class:`HNetTrainingConfig` field and that every name
listed here really is not one -- a stale exemption silently widens the
carve-out."""


def build_parser() -> argparse.ArgumentParser:
    """Build the real parser: the common flags plus ``--gpu``.

    Interface contract: pure. Constructs and returns a parser; parses nothing,
    reads no environment, touches no filesystem and allocates no device. It is
    the single parser both :func:`parse_arguments` and the CLI contract guard
    drive, so there is no parser a test can pass against while the trainer
    uses another.

    :return: The parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description=(
            "Pretrain H-Net (hierarchical dynamic chunking) on raw UTF-8 bytes "
            "from the staged Wikipedia dump with stock Keras fit(). There is "
            "no tokenizer: vocab_size is 256. Corpus status and the gated "
            "FineWeb-Edu slot: `python -m train.hnet.prepare_hnet_data "
            "--dry-run`."
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
) -> Tuple[argparse.Namespace, HNetTrainingConfig]:
    """Run the full ``argv -> parse -> config`` path.

    Interface contract: the single entry point :func:`main` uses AND the single
    entry point the CLI guard drives. The namespace is returned only so
    :func:`main` can read the process-level dests in :data:`NON_CONFIG_DESTS`;
    every other value arrives through the config.

    :param argv: Tokens without the program name. ``None`` reads
        ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    :return: ``(namespace, config)``.
    :rtype: Tuple[argparse.Namespace, HNetTrainingConfig]
    :raises SystemExit: As argparse does, on ``--help`` or a bad flag.
    :raises ValueError: From ``HNetTrainingConfig.__post_init__`` on a knob
        combination the trainer cannot honour (an unknown variant, a
        ``max_chunks`` list of the wrong length, a floor above the peak
        learning rate, ...).
    """
    args = build_parser().parse_args(argv)
    return args, config_from_args(args)


def config_from_argv(
    argv: Optional[Sequence[str]] = None,
) -> HNetTrainingConfig:
    """The config half of :func:`parse_arguments`.

    :param argv: Tokens without the program name.
    :type argv: Optional[Sequence[str]]
    :return: The validated config.
    :rtype: HNetTrainingConfig
    """
    return parse_arguments(argv)[1]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse the CLI, set the process up, and train.

    The statement ORDER is the contract, and it is what the CLI guard's
    sentinels measure:

    1. parse -- so ``--help`` costs nothing;
    2. ``setup_gpu`` -- process-level, from the non-config ``--gpu`` dest;
    3. ``train`` -- which seeds, builds the byte pipeline, builds the model
       and calls stock ``fit()``.

    :param argv: Tokens without the program name. ``None`` reads
        ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    """
    args, config = parse_arguments(argv)

    setup_gpu(gpu_id=args.gpu)

    logger.info(
        "H-Net: training variant %r (of %s) on %s bytes at seq_len=%d, "
        "batch=%d",
        config.arch_variant,
        list(arch_variant_names()),
        config.dataset_root,
        config.seq_len,
        config.batch_size,
    )
    _, _, results_dir = train(config)
    logger.info(
        "H-Net %s training complete; artifacts in %s",
        config.arch_variant, results_dir,
    )


if __name__ == "__main__":
    main()
