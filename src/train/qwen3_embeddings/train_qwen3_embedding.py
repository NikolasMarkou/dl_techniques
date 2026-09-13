r"""Train `Qwen3EmbeddingModel` with in-batch-negative InfoNCE on synthetic data.

See `train.qwen3_embeddings.common`'s module docstring for the full InfoNCE
decision (D-013) and the synthetic-data caveat this shares with the ColBERT
trainers: any checkpoint this script produces is a WIRING result, never a
retrieval-quality claim.

Usage::

    # what the CLI offers -- allocates nothing, touches no GPU
    MPLBACKEND=Agg .venv/bin/python -m train.qwen3_embeddings.train_qwen3_embedding --help

    # tiny end-to-end wiring check, seconds on CPU
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" .venv/bin/python -m \
        train.qwen3_embeddings.train_qwen3_embedding --smoke

``main()`` PARSES FIRST, so `--help` costs nothing (`src/train/CLAUDE.md`).
"""

from __future__ import annotations

import argparse
from typing import Any, Optional, Sequence, Tuple

import keras

from dl_techniques.utils.logger import logger
from train.common import set_seeds, setup_gpu
from train.common.args import config_values_from_args
from train.common.callbacks import best_checkpoint_path, create_callbacks

from train.qwen3_embeddings.common import (
    CLI_TO_CONFIG,
    SMOKE_PRESET,
    TrainingConfig,
    add_common_arguments,
    build_embedding_datasets,
    build_embedding_model,
    build_optimizer,
    embedding_steps_per_epoch,
)

__all__ = ["build_parser", "main", "train_qwen3_embedding"]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the embedding-tower parser.

    :returns: A parser carrying every flag in `CLI_TO_CONFIG` plus `--gpu`.
    """
    parser = argparse.ArgumentParser(
        prog="train_qwen3_embedding.py",
        description=(
            "Train Qwen3EmbeddingModel (bi-encoder) with in-batch-negative "
            "InfoNCE on synthetic query/positive-document pairs. Wiring "
            "only -- no pretrained weights, no IR dataset."
        ),
    )
    return add_common_arguments(parser)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def train_qwen3_embedding(
    config: TrainingConfig,
) -> Tuple[keras.Model, keras.callbacks.History, str]:
    """Run the embedding-tower recipe end to end.

    :param config: Fully resolved configuration.
    :returns: `(model, history, results_dir)`.
    """
    logger.info("=" * 70)
    logger.info("Qwen3EmbeddingModel -- in-batch-negative InfoNCE on SYNTHETIC data")
    logger.info("WIRING RESULT ONLY: no pretrained weights, no IR corpus.")
    logger.info("=" * 70)
    logger.info(f"Config: {config!r}")

    set_seeds(config.seed)

    train_dataset, val_dataset, _query_tok, _doc_tok = build_embedding_datasets(config)
    vocab_size = _query_tok.vocab_size

    steps_per_epoch = embedding_steps_per_epoch(config)
    optimizer = build_optimizer(config, steps_per_epoch)
    model = build_embedding_model(config, vocab_size=vocab_size, optimizer=optimizer)

    callbacks, results_dir = create_callbacks(
        model_name="qwen3_embedding",
        results_dir_prefix=config.results_dir_prefix,
        output_root=config.output_root,
        monitor="val_loss",
        monitor_mode="min",
        patience=config.patience,
        use_lr_schedule=True,
        include_terminate_on_nan=True,
        # The epoch analyzer inspects a single labelled classifier output;
        # this model returns raw embeddings and has no class labels.
        include_analyzer=False,
    )
    logger.info(f"Run directory: {results_dir}")

    history = model.fit(
        train_dataset,
        epochs=config.epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=2,
    )

    logger.info(f"Best checkpoint: {best_checkpoint_path(results_dir)}")
    logger.info(
        "Reminder: this loss is a wiring signal on synthetic data, not a "
        "retrieval metric."
    )
    return model, history, results_dir


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse `argv`, build the config and train.

    :param argv: Tokens without the program name. `None` reads `sys.argv`.
    """
    parser = build_parser()
    args, values = config_values_from_args(parser, argv, CLI_TO_CONFIG, SMOKE_PRESET)
    config = TrainingConfig(**values)

    setup_gpu(gpu_id=args.gpu)
    train_qwen3_embedding(config)


if __name__ == "__main__":
    main()
