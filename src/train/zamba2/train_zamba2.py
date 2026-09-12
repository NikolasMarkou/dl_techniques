r"""Pretrain Zamba2 on Wikipedia through stock ``fit()``.

Usage::

    # what the CLI offers -- allocates nothing, touches no GPU
    MPLBACKEND=Agg .venv/bin/python -m train.zamba2.train_zamba2 --help

    # the mini variant smoke layout on the already-staged Wikipedia dump
    MPLBACKEND=Agg .venv/bin/python -m train.zamba2.train_zamba2 \
        --variant zamba2_mini --seq-length 256 --batch-size 4 \
        --epochs 1 --steps-per-epoch 10 --gpu 1

This module is an ENTRY POINT and nothing else. The config, the flag
surface, the packed-CLM pipeline, the optimizer, the model construction, the
callbacks and ``fit()`` itself all live in :mod:`train.zamba2.common`, so
there is exactly one place to read and exactly one place a defect can hide.

``main()`` PARSES FIRST.
    ``args, config = parse_arguments(argv)`` is the first statement, so
    ``--help`` prints a ``usage:`` line and exits without claiming a GPU,
    seeding the process, opening the Wikipedia Arrow cache, instantiating a
    tokenizer, or constructing a model. ``src/train/CLAUDE.md`` states the
    trap this avoids: **exit 0 is not a passing** ``--help``. A script with
    no parser at all ignores ``--help``, runs its whole job and exits 0
    anyway.

NOTHING EAGER AT MODULE SCOPE.
    No ``tf.``/``keras.`` call runs at import time.

THE FILE IS ``train_zamba2.py``, NEVER ``train.py``.
    A module named ``train.py`` inside a package on ``sys.path`` shadows the
    ``train`` package itself and breaks ``from train.common import ...``
    (``src/train/CLAUDE.md`` § File naming).

SEEDING IS NOT REPEATED HERE.
    ``common.train`` calls ``set_seeds(config.seed)`` as its first statement.

``--gpu`` IS NOT A CONFIG FIELD.
    It acts on the process, is consumed by ``setup_gpu`` and never reaches
    :class:`~train.zamba2.common.Zamba2TrainingConfig`. :data:`NON_CONFIG_DESTS`
    is the single, checked exemption from the "every parser dest is a config
    field" contract.
"""

from __future__ import annotations

import argparse
from typing import FrozenSet, Optional, Sequence, Tuple

from dl_techniques.utils.logger import logger
from train.common import setup_gpu
from train.zamba2.common import (
    Zamba2TrainingConfig,
    add_common_arguments,
    config_from_args,
    train,
    variant_names,
)

__all__ = [
    "NON_CONFIG_DESTS",
    "PROGRAM_NAME",
    "build_parser",
    "config_from_argv",
    "main",
    "parse_arguments",
]

PROGRAM_NAME: str = "train_zamba2.py"
"""``prog`` for the parser, so ``--help`` names the script rather than
``__main__`` when the module is run with ``python -m``."""

NON_CONFIG_DESTS: FrozenSet[str] = frozenset({"gpu"})
"""Parser destinations that are deliberately NOT config fields. The ONLY
permitted way out of the flag-to-field contract."""


def build_parser() -> argparse.ArgumentParser:
    """Build the real parser: the common flags plus ``--gpu``.

    Interface contract: pure. Constructs and returns a parser; parses
    nothing, reads no environment, touches no filesystem and allocates no
    device.

    :return: The parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description=(
            "Pretrain Zamba2 (hybrid Mamba2 + shared-attention mem-blocks) "
            "on the staged Wikipedia dump with stock Keras fit(). Tokenizer "
            f"is Tiktoken. Variants: {list(variant_names())}."
        ),
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--gpu", type=int, default=None,
        help=(
            "GPU index to use (e.g. 1). Omit to let the process see "
            "whatever CUDA_VISIBLE_DEVICES exposes. Not a config field."
        ),
    )
    return parser


def parse_arguments(
    argv: Optional[Sequence[str]] = None,
) -> Tuple[argparse.Namespace, Zamba2TrainingConfig]:
    """Run the full ``argv -> parse -> config`` path.

    :param argv: Tokens without the program name. ``None`` reads
        ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    :return: ``(namespace, config)``.
    :rtype: Tuple[argparse.Namespace, Zamba2TrainingConfig]
    :raises SystemExit: As argparse does, on ``--help`` or a bad flag.
    :raises ValueError: From ``Zamba2TrainingConfig.__post_init__`` on a
        knob combination the trainer cannot honour.
    """
    args = build_parser().parse_args(argv)
    return args, config_from_args(args)


def config_from_argv(
    argv: Optional[Sequence[str]] = None,
) -> Zamba2TrainingConfig:
    """The config half of :func:`parse_arguments`.

    :param argv: Tokens without the program name.
    :type argv: Optional[Sequence[str]]
    :return: The validated config.
    :rtype: Zamba2TrainingConfig
    """
    return parse_arguments(argv)[1]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse the CLI, set the process up, and train.

    The statement ORDER is the contract:

    1. parse -- so ``--help`` costs nothing;
    2. ``setup_gpu`` -- process-level, from the non-config ``--gpu`` dest;
    3. ``train`` -- which seeds, builds the packed-CLM pipeline, builds the
       model and calls stock ``fit()``.

    :param argv: Tokens without the program name. ``None`` reads
        ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    """
    args, config = parse_arguments(argv)

    setup_gpu(gpu_id=args.gpu)

    logger.info(
        "Zamba2: training variant %r (of %s) on %s at seq_len=%d, batch=%d",
        config.variant,
        list(variant_names()),
        config.dataset_root,
        config.max_seq_length,
        config.batch_size,
    )
    _, _, results_dir = train(config)
    logger.info(
        "Zamba2 %s training complete; artifacts in %s",
        config.variant, results_dir,
    )


if __name__ == "__main__":
    main()
