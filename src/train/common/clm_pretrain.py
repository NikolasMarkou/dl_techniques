"""The CLM-pretraining WRAPPER layer shared by every causal-LM trainer.

``train.common.nlp`` already owns every LEAF helper these trainers need
(``preprocess_clm_dataset``, ``load_text_dataset``, ``estimate_clm_steps_per_epoch``,
...). What was duplicated four times is the thin WRAPPER layer above them: the
``dataset_source`` dispatch, the dict-output label wrap, the TFDS steps-per-epoch
short-circuit, the loss-type branch and the checkpoint-name step parser.

This module is that layer, and nothing else. It holds no model, no optimizer and no
training loop: every function here takes a trainer ``TrainingConfig``-shaped object,
reads a fixed set of FIELD NAMES off it, and delegates the real work to
``train.common.nlp`` / ``dl_techniques``. That is why ``config`` is typed ``Any`` --
the four trainers each declare their own ``TrainingConfig`` dataclass and this module
must not learn any one of them.

Promoted from four verbatim copies (``train.gpt2.pretrain``,
``train.wave_field.pretrain``, ``train.wave_field.train_memory``,
``train.cliffordnet.train_cliffordnet_nlp``); each of those keeps a module-level alias
under its old private name so every import path live before the move still resolves.
"""

import os
import re
from typing import Any, Optional, Tuple

import keras
import tensorflow as tf

from train.common.nlp import (
    load_text_dataset,
    preprocess_clm_dataset,
    estimate_clm_steps_per_epoch,
)

from dl_techniques.utils.logger import logger
from dl_techniques.datasets.nlp import load_wikipedia_train_val
from dl_techniques.losses import MaskedCausalLMLoss, FocalCausalLMLoss

__all__ = [
    "extract_step_from_checkpoint",
    "create_clm_loss_fn",
    "load_train_val_datasets",
    "load_tfds_clm_datasets",
    "load_hf_clm_datasets",
    "make_clm_steps_per_epoch",
]


# ---------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------


def extract_step_from_checkpoint(path: str) -> int:
    """Extract the training step from a checkpoint filename.

    Handles ``step_0025000.keras`` and ``final.keras`` patterns.
    Returns 0 if the step cannot be determined.

    Args:
        path: Checkpoint path; only its basename is inspected.

    Returns:
        The first ``step_<digits>`` group in the basename as an ``int``, else 0.
    """
    basename = os.path.basename(path)
    match = re.search(r"step_(\d+)", basename)
    if match:
        return int(match.group(1))
    return 0


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------


# DECISION plan-2026-08-12T123743-e798a9e1/D-008
# The two `logger.info` lines below are LOAD-BEARING for this consolidation, not
# decoration: they are the 3-of-4 majority spelling of a function that existed in four
# copies, and `train.wave_field.train_memory`'s copy had silently dropped them.
# WHAT NOT TO DO: do NOT "simplify" this into `return FocalCausalLMLoss(...)` /
# `return MaskedCausalLMLoss(...)` early returns. That is exactly the shape
# train_memory.py had drifted into, and collapsing back to it re-removes the loss
# provenance line from all four CLM trainers' logs -- an observability regression that
# no test can see, because the returned loss object is identical either way.
# WHAT NOT TO DO (2): do NOT restore gpt2's unicode `γ` in the f-string. ASCII `gamma`
# is the 3-of-4 majority and survives a non-UTF-8 log consumer.
# See decisions.md D-008.
def create_clm_loss_fn(config: Any) -> keras.losses.Loss:
    """Create the CLM loss function from configuration.

    Args:
        config: Trainer config; reads ``loss_type``, ``focal_gamma`` and
            ``label_smoothing``.

    Returns:
        ``FocalCausalLMLoss`` when ``config.loss_type == "focal"``, else
        ``MaskedCausalLMLoss``.
    """
    if config.loss_type == "focal":
        loss_fn = FocalCausalLMLoss(
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
        )
        logger.info(f"Loss: FocalCausalLMLoss(gamma={config.focal_gamma})")
    else:
        loss_fn = MaskedCausalLMLoss(
            label_smoothing=config.label_smoothing,
        )
        logger.info("Loss: MaskedCausalLMLoss")
    return loss_fn


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


# DECISION plan-2026-08-12T123743-e798a9e1/D-009
# `data_seed` is REQUIRED and has NO default. Three of the four pre-move copies
# declared it required; `train.cliffordnet.train_cliffordnet_nlp` carried a
# `data_seed: int = 42` default that was DEAD (its single caller passes
# `data_seed=data_seed` explicitly, at train_cliffordnet_nlp.py:443-444).
# WHAT NOT TO DO: do NOT re-add `= 42`. A default here turns the CLM resume-seeding
# contract (`src/train/CLAUDE.md` Pattern 3 / D-006: `data_seed = config.seed +
# initial_step`) from a `TypeError` at the call site into a SILENT wrong article
# ordering on resume -- a resumed run would quietly replay the first N chunks.
# WHAT NOT TO DO (2): do NOT shorten the ValueError message below to just
# `f"Unknown dataset_source: {config.dataset_source!r}"`. That truncated form is
# `train.wave_field.train_memory`'s drifted copy; the 3-of-4 majority names the two
# legal values, which is the only part of the message a user can act on.
# See decisions.md D-009.
def load_train_val_datasets(
    config: Any,
    preprocessor,
    data_seed: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[int]]:
    """Load, preprocess, and wrap train/val datasets for the dict-output model.

    Args:
        config: Trainer config; reads ``dataset_source`` plus whatever the selected
            branch's loader needs.
        preprocessor: The ``TiktokenPreprocessor`` used to tokenize/chunk.
        data_seed: Holdout-split seed. REQUIRED -- see the D-009 anchor above.

    Returns:
        ``(train_ds, val_ds, n_train_articles)``. The article count is the post-filter
        Wikipedia article count (HF path) or ``None`` (TFDS path).

    Raises:
        ValueError: If ``config.dataset_source`` is neither ``"tfds"`` nor
            ``"huggingface"``.
    """
    n_train_articles: Optional[int] = None
    if config.dataset_source == "tfds":
        train_ds, val_ds = load_tfds_clm_datasets(config, preprocessor)
    elif config.dataset_source == "huggingface":
        train_ds, val_ds, n_train_articles = load_hf_clm_datasets(
            config, preprocessor, data_seed,
        )
    else:
        raise ValueError(
            f"Unknown dataset_source: {config.dataset_source!r}. "
            f"Use 'tfds' or 'huggingface'."
        )

    # Wrap labels for dict-output model: (x, y) -> (x, {"logits": y})
    wrap = lambda ds: ds.map(
        lambda x, y: (x, {"logits": y}),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return wrap(train_ds), wrap(val_ds), n_train_articles


def load_tfds_clm_datasets(
    config: Any,
    preprocessor,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load train/val from TFDS (e.g. IMDB).

    Args:
        config: Trainer config; reads ``dataset_name``, ``max_samples``,
            ``max_seq_length``, ``batch_size``.
        preprocessor: The ``TiktokenPreprocessor`` used to tokenize/chunk.

    Returns:
        ``(train_ds, val_ds)`` -- the TFDS ``"train"`` and ``"test"`` splits.
    """
    train = preprocess_clm_dataset(
        load_text_dataset(config.dataset_name, "train", config.max_samples),
        preprocessor, config.max_seq_length, config.batch_size,
    )
    val = preprocess_clm_dataset(
        load_text_dataset(config.dataset_name, "test", config.max_samples),
        preprocessor, config.max_seq_length, config.batch_size,
    )
    return train, val


def load_hf_clm_datasets(
    config: Any,
    preprocessor,
    data_seed: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """Load train/val from Wikipedia with a random holdout split.

    Args:
        config: Trainer config; reads ``hf_cache_dir``, ``hf_wikipedia_config``,
            ``min_article_length``, ``val_fraction``, ``max_train_samples``,
            ``max_val_samples``, ``shuffle_shards``, ``max_seq_length``, ``batch_size``.
        preprocessor: The ``TiktokenPreprocessor`` used to tokenize/chunk.
        data_seed: Holdout-split seed, passed straight through as ``seed=``.

    Returns:
        ``(train_ds, val_ds, n_train_articles)`` with the POST-filter train article
        count, suitable for :func:`make_clm_steps_per_epoch`.
    """
    train_raw, val_raw, n_train, _n_val = load_wikipedia_train_val(
        cache_dir=config.hf_cache_dir,
        config_name=config.hf_wikipedia_config,
        min_article_length=config.min_article_length,
        val_fraction=config.val_fraction,
        max_train_samples=config.max_train_samples,
        max_val_samples=config.max_val_samples,
        seed=data_seed,
        num_shards=config.shuffle_shards,
        return_counts=True,
    )
    train = preprocess_clm_dataset(
        train_raw, preprocessor,
        config.max_seq_length, config.batch_size,
    )
    val = preprocess_clm_dataset(
        val_raw, preprocessor,
        config.max_seq_length, config.batch_size,
    )
    return train, val, n_train


# ---------------------------------------------------------------------
# Steps per epoch
# ---------------------------------------------------------------------


def make_clm_steps_per_epoch(
    config: Any, n_train_articles: Optional[int],
) -> int:
    """Resolve ``steps_per_epoch`` via the canonical helper (D-001).

    The TFDS branch short-circuits: TFDS gives a sample count, not an article count,
    so the chunk-aware estimator does not apply there.

    Args:
        config: Trainer config; reads ``dataset_source``, ``max_samples``,
            ``steps_per_epoch``, ``max_train_samples``, ``max_seq_length``,
            ``batch_size``.
        n_train_articles: Post-filter train article count, or ``None`` (TFDS path).

    Returns:
        The resolved number of optimizer steps per epoch.
    """
    if config.dataset_source == "tfds" and config.max_samples and config.steps_per_epoch is None:
        return max(1, config.max_samples // config.batch_size)
    return estimate_clm_steps_per_epoch(
        num_articles=n_train_articles or config.max_train_samples,
        max_seq_length=config.max_seq_length,
        batch_size=config.batch_size,
        override=config.steps_per_epoch,
    )
