"""Shared building blocks for the Zamba2 Pattern-3 (subword CLM) trainer.

Zamba2 is a standard subword causal LM, not a byte-level one, so this module
follows the Pattern-3 shape (`src/train/CLAUDE.md`, exemplar
``src/train/bert/pretrain.py``) rather than ``train.hnet``'s Pattern-6 byte
pipeline -- even though the STRUCTURE below (config dataclass + argparse +
``config_from_args`` single-wiring-site + ``build_datasets``/
``build_optimizer``/``build_model``/``train``) deliberately mirrors
``train.hnet.common``'s, which is the more current convention for a new
Pattern-3/6 trainer per ``src/train/CLAUDE.md``.

Two things this module deliberately does NOT do:

* **No custom ``train_step`` authored here.** ``build_model`` wraps
  ``Zamba2Model`` in ``CausalLanguageModel`` (``skip_head=True,
  pre_shifted=True, causality_probe_plain_tensor=True``,
  plan-2026-09-13T052422-19022ba2 step 4), which owns the
  ``tf.GradientTape``-based ``train_step``/``test_step`` and the injected
  ``loss_fn=create_clm_loss_fn(config)``; ``compile()`` here receives only
  the optimizer. Zamba2 has no auxiliary loss (unlike H-Net's boundary-ratio
  loss, confirmed by a zero-hit ``self.add_loss`` grep of this package), so
  ``aggregate_backbone_losses`` stays at its default ``False``.
* **No ``ClmPretrainConfig``/``load_train_val_datasets`` reuse.**
  ``train.common.clm_pretrain``'s wrapper layer wraps every label tensor as
  ``{"logits": y}`` for the four DICT-output trainers that share it (GPT-2,
  wave_field, cliffordnet) -- see ``load_train_val_datasets``'s
  ``# Wrap labels for dict-output model`` comment. ``Zamba2Model.call``
  returns a plain ``(batch, seq_len, vocab_size)`` tensor, not a dict, so
  reusing that wrapper would either silently mismatch ``model.fit``'s label
  shape or require wrapping ``Zamba2Model`` in a dict output it does not
  have. This module therefore calls the LEAF helpers directly
  (``load_wikipedia_train_val``, ``preprocess_clm_packed_dataset``,
  ``estimate_clm_steps_per_epoch``) the same way ``train.hnet.common`` calls
  the byte-level leaves, and reuses ``create_clm_loss_fn`` (which reads only
  scalar config fields, not the dict-output assumption) for the loss.
  See ``decisions.md`` D-006.

Public surface:
    * :data:`VARIANT_NAMES` -- the three shipped :data:`MODEL_VARIANTS` keys.
    * :class:`Zamba2TrainingConfig` -- the run knobs. Every field is consumed
      by something other than the config dump.
    * :func:`add_common_arguments` -- the shared CLI flags.
    * :func:`config_from_args` -- namespace -> config, the ONE wiring site.
    * :func:`build_datasets` -- the Wikipedia packed-CLM pipeline.
    * :func:`build_optimizer` / :func:`build_model` -- AdamW through
      ``optimizer_builder``; ``build_model`` wraps the backbone in
      ``CausalLanguageModel``, which owns its own loss/metric tracking.
    * :func:`train` -- stock ``fit()``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import keras
import tensorflow as tf

from dl_techniques.datasets.nlp import (
    DEFAULT_WIKIPEDIA_CACHE_DIR,
    DEFAULT_WIKIPEDIA_CONFIG,
    load_wikipedia_train_val,
)
from dl_techniques.models.language.masked_language_model.clm import CausalLanguageModel
from dl_techniques.models.language.zamba2 import MODEL_VARIANTS, create_zamba2
from dl_techniques.optimization import (
    learning_rate_schedule_builder,
    optimizer_builder,
)
from dl_techniques.utils.logger import logger
from train.common import create_callbacks, set_seeds
from train.common.clm_pretrain import create_clm_loss_fn
from train.common.config_io import save_config_json
from train.common.nlp import (
    create_tokenizer,
    estimate_clm_steps_per_epoch,
    preprocess_clm_packed_dataset,
)
from train.common.run_io import save_training_history_json

__all__ = [
    "DEFAULT_VARIANT",
    "RESULTS_DIR_PREFIX",
    "TRAIN_MONITOR",
    "VARIANT_NAMES",
    "WEIGHT_DECAY_EXCLUDED",
    "Zamba2TrainingConfig",
    "add_common_arguments",
    "build_datasets",
    "build_model",
    "build_optimizer",
    "config_from_args",
    "train",
    "variant_names",
]

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

RESULTS_DIR_PREFIX: str = "zamba2"
"""Prefix of the timestamped run directory under ``--output-dir``."""

TRAIN_MONITOR: str = "val_loss"
"""The monitored metric. Its direction is resolved by
``train.common.callbacks.resolve_monitor_mode``, never hand-written here."""

WEIGHT_DECAY_EXCLUDED: Tuple[str, ...] = ("bias", "gamma", "beta")
"""Variable-name patterns kept out of AdamW's decoupled decay (matches
``train.hnet.common``'s convention: Keras spells LayerNorm/RMSNorm
parameters ``gamma``/``beta``). Weight decay is applied by the optimizer and
by NOTHING else -- no ``kernel_regularizer`` is ever attached."""

DEFAULT_VARIANT: str = "zamba2_mini"
"""Default ``--variant`` -- the smallest shipped size, for fast iteration."""

VARIANT_NAMES: Tuple[str, ...] = tuple(MODEL_VARIANTS)
"""Every ``--variant`` choice: the three keys of
:data:`~dl_techniques.models.language.zamba2.model.MODEL_VARIANTS`, in
declaration order (``zamba2_mini``, ``zamba2_small``, ``zamba2_base``)."""


def variant_names() -> Tuple[str, ...]:
    """:returns: :data:`VARIANT_NAMES`, as the CLI's ``choices``.
    :rtype: Tuple[str, ...]
    """
    return VARIANT_NAMES


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass
class Zamba2TrainingConfig:
    """Knobs for one Zamba2 pretraining run.

    Every annotated field is read by something other than the config dump
    (``save_config_json``/``asdict`` serialize the whole config, which
    RECORDS a field without consuming it -- ``src/train/CLAUDE.md``).

    :param variant: A member of :data:`VARIANT_NAMES`.
    :param dataset_root: Arrow cache directory holding the Wikipedia dump.
    :param wikipedia_config: Wikipedia dump config name, e.g. ``20231101.en``.
    :param encoding_name: Tiktoken encoding name. Determines both the
        tokenizer used to pack the corpus and the model's ``vocab_size``
        (read off the live tokenizer, never hardcoded -- see
        :func:`build_datasets`).
    :param max_seq_length: Tokens the MODEL sees per window. The packer is
        asked for ``max_seq_length + 1`` tokens so the causal shift has a
        token to spend.
    :param batch_size: Windows per optimizer step.
    :param epochs: Training epochs.
    :param steps_per_epoch: Override for the estimated steps per epoch;
        ``None`` derives it from the post-filter article count.
    :param learning_rate: Peak learning rate, reached at the end of warmup.
    :param final_learning_rate: Cosine floor.
    :param weight_decay: Decoupled AdamW weight decay. Applied by the
        optimizer ONLY -- never also as a ``kernel_regularizer``.
    :param warmup_ratio: Fraction of total steps spent warming up.
    :param gradient_clip_norm: Per-variable gradient-norm clip, or ``0`` to
        disable. Passed to ``optimizer_builder`` under its own key name.
    :param loss_type: ``"ce"`` (:class:`MaskedCausalLMLoss`) or ``"focal"``
        (:class:`FocalCausalLMLoss`), read by :func:`create_clm_loss_fn`.
    :param focal_gamma: Focal-loss gamma; only consumed when
        ``loss_type == "focal"``.
    :param label_smoothing: Label smoothing in ``[0, 1)``, forwarded to the
        CLM loss.
    :param min_article_length: Articles shorter than this many CHARACTERS
        are skipped. ``0`` (the packed-CLM default) keeps every token.
    :param max_train_samples: Cap on training articles, ``None`` for all.
    :param max_val_samples: Cap on validation articles.
    :param val_fraction: Fraction of articles held out for validation.
    :param shuffle_shards: Parallel Wikipedia shards; ``>1`` reshuffles the
        article order at every epoch boundary.
    :param shuffle_buffer: tf.data shuffle buffer over packed CLM windows.
    :param seed: Seed for ``set_seeds`` and for the corpus split.
    :param patience: Early-stopping patience.
    :param output_dir: Root under which the timestamped run directory is
        made. Relative paths resolve against the process CWD -- invoke from
        the repo root so this lands at repo-root ``results/``, never
        ``src/results/`` (``plans/SYSTEM.md`` Invariants).
    """

    variant: str = DEFAULT_VARIANT
    dataset_root: str = DEFAULT_WIKIPEDIA_CACHE_DIR
    wikipedia_config: str = DEFAULT_WIKIPEDIA_CONFIG
    encoding_name: str = "cl100k_base"

    max_seq_length: int = 512
    batch_size: int = 8
    epochs: int = 1
    steps_per_epoch: Optional[int] = None

    learning_rate: float = 3e-4
    final_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.02
    gradient_clip_norm: float = 1.0

    loss_type: str = "ce"
    focal_gamma: float = 1.0
    label_smoothing: float = 0.0

    min_article_length: int = 0
    max_train_samples: Optional[int] = None
    max_val_samples: int = 5000
    val_fraction: float = 0.02
    shuffle_shards: int = 4
    shuffle_buffer: int = 4096

    seed: int = 42
    patience: int = 5
    output_dir: str = "results"

    def __post_init__(self) -> None:
        """Validate the knobs at construction time.

        :raises ValueError: on an unknown variant, a non-positive count, or a
            ratio outside its range.
        """
        if self.variant not in MODEL_VARIANTS:
            raise ValueError(
                f"unknown variant {self.variant!r}; available: "
                f"{list(VARIANT_NAMES)}"
            )
        if self.max_seq_length < 2:
            raise ValueError(
                f"max_seq_length must be >= 2 (the causal shift spends one "
                f"token), got {self.max_seq_length}"
            )
        positive = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "shuffle_shards": self.shuffle_shards,
            "shuffle_buffer": self.shuffle_buffer,
            "max_val_samples": self.max_val_samples,
            "patience": self.patience,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.steps_per_epoch is not None and self.steps_per_epoch <= 0:
            raise ValueError(
                f"steps_per_epoch must be positive or None, got "
                f"{self.steps_per_epoch}"
            )
        if self.max_train_samples is not None and self.max_train_samples <= 0:
            raise ValueError(
                f"max_train_samples must be positive or None, got "
                f"{self.max_train_samples}"
            )
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if not 0.0 < self.final_learning_rate <= self.learning_rate:
            raise ValueError(
                "final_learning_rate must be positive and no greater than "
                f"learning_rate, got {self.final_learning_rate} vs "
                f"{self.learning_rate}"
            )
        if self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(
                f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}"
            )
        if self.gradient_clip_norm < 0.0:
            raise ValueError(
                f"gradient_clip_norm must be non-negative (0 disables it), "
                f"got {self.gradient_clip_norm}"
            )
        if self.loss_type not in ("ce", "focal"):
            raise ValueError(
                f"loss_type must be 'ce' or 'focal', got {self.loss_type!r}"
            )
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {self.label_smoothing}"
            )
        if self.min_article_length < 0:
            raise ValueError(
                f"min_article_length must be non-negative, got "
                f"{self.min_article_length}"
            )
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(
                f"val_fraction must be in (0, 1), got {self.val_fraction}"
            )

    @property
    def packed_window(self) -> int:
        """:returns: The window length asked of the packer -- one token more
            than the model's :attr:`max_seq_length`, because
            ``preprocess_clm_packed_dataset`` spends one on the causal shift.
        :rtype: int
        """
        return self.max_seq_length + 1


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the shared Zamba2 training flags.

    ``--gpu`` is deliberately NOT here: it is consumed by ``setup_gpu`` in
    ``main()`` and is not a config field.

    The four flags every CLM script in this tree exposes --
    ``--steps-per-epoch``, ``--seed``, ``--min-article-length``,
    ``--shuffle-shards`` -- are all present and spelled identically
    (``src/train/CLAUDE.md``).

    :param parser: The parser to extend.
    :type parser: argparse.ArgumentParser
    :returns: The same parser, for chaining.
    :rtype: argparse.ArgumentParser
    """
    defaults = Zamba2TrainingConfig()

    parser.add_argument(
        "--variant", type=str, default=defaults.variant,
        choices=list(variant_names()),
        help="Zamba2 model size to train.",
    )
    parser.add_argument(
        "--dataset-root", type=str, default=defaults.dataset_root,
        help="Arrow cache directory holding the Wikipedia dump.",
    )
    parser.add_argument(
        "--wikipedia-config", type=str, default=defaults.wikipedia_config,
        help="Wikipedia dump config name.",
    )
    parser.add_argument(
        "--encoding-name", type=str, default=defaults.encoding_name,
        help="Tiktoken encoding name; also determines the model vocab_size.",
    )
    parser.add_argument(
        "--seq-length", "--max-seq-length", dest="max_seq_length",
        type=int, default=defaults.max_seq_length,
        help="Tokens the model sees per window.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=defaults.batch_size,
        help="Windows per optimizer step.",
    )
    parser.add_argument(
        "--epochs", type=int, default=defaults.epochs, help="Training epochs.",
    )
    parser.add_argument(
        "--steps-per-epoch", type=int, default=defaults.steps_per_epoch,
        help="Override the estimated steps per epoch.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=defaults.learning_rate,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--final-learning-rate", type=float, default=defaults.final_learning_rate,
        help="Cosine floor learning rate.",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=defaults.weight_decay,
        help="Decoupled AdamW weight decay (never also an L2 regularizer).",
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=defaults.warmup_ratio,
        help="Fraction of total steps spent warming up.",
    )
    parser.add_argument(
        "--gradient-clip-norm", type=float, default=defaults.gradient_clip_norm,
        help="Per-variable gradient-norm clip; 0 disables clipping.",
    )
    parser.add_argument(
        "--loss-type", type=str, default=defaults.loss_type,
        choices=["ce", "focal"],
        help="'ce' (MaskedCausalLMLoss) or 'focal' (FocalCausalLMLoss).",
    )
    parser.add_argument(
        "--focal-gamma", type=float, default=defaults.focal_gamma,
        help="Focal loss gamma (only used when --loss-type focal).",
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=defaults.label_smoothing,
        help="Label smoothing in [0, 1).",
    )
    parser.add_argument(
        "--min-article-length", type=int, default=defaults.min_article_length,
        help="Skip articles shorter than this many characters.",
    )
    parser.add_argument(
        "--max-train-samples", type=int, default=defaults.max_train_samples,
        help="Cap on training articles; omit for all of them.",
    )
    parser.add_argument(
        "--max-val-samples", type=int, default=defaults.max_val_samples,
        help="Cap on validation articles.",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=defaults.val_fraction,
        help="Fraction of articles held out for validation.",
    )
    parser.add_argument(
        "--shuffle-shards", type=int, default=defaults.shuffle_shards,
        help="Parallel Wikipedia shards (>1 reshuffles every epoch).",
    )
    parser.add_argument(
        "--shuffle-buffer", type=int, default=defaults.shuffle_buffer,
        help="tf.data shuffle buffer over packed CLM windows.",
    )
    parser.add_argument(
        "--seed", type=int, default=defaults.seed, help="Random seed.",
    )
    parser.add_argument(
        "--patience", type=int, default=defaults.patience,
        help="Early-stopping patience.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=defaults.output_dir,
        help="Root for the timestamped run directory (repo-root 'results' "
             "when invoked from the repo root; never under src/).",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Zamba2TrainingConfig:
    """Build a config from a parsed namespace.

    The ONE wiring site between :func:`add_common_arguments` and
    :class:`Zamba2TrainingConfig`. A flag that does not arrive here silently
    does nothing.

    :param args: A namespace produced by a parser carrying the common flags.
    :type args: argparse.Namespace
    :returns: The config.
    :rtype: Zamba2TrainingConfig
    """
    return Zamba2TrainingConfig(
        variant=args.variant,
        dataset_root=args.dataset_root,
        wikipedia_config=args.wikipedia_config,
        encoding_name=args.encoding_name,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_clip_norm=args.gradient_clip_norm,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        min_article_length=args.min_article_length,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        val_fraction=args.val_fraction,
        shuffle_shards=args.shuffle_shards,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        patience=args.patience,
        output_dir=args.output_dir,
    )


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------


def build_datasets(
        config: Zamba2TrainingConfig,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """Build the Wikipedia packed-CLM pipeline, the step budget, and vocab_size.

    :param config: The run config.
    :type config: Zamba2TrainingConfig
    :returns: ``(train_ds, val_ds, steps_per_epoch, vocab_size)``. ``vocab_size``
        is read off the live tokenizer (:attr:`TiktokenPreprocessor.vocab_size`)
        rather than hardcoded, so the model's embedding/head always match the
        encoding actually used to pack the corpus (see the module docstring's
        vocab-size note).
    :rtype: Tuple[tf.data.Dataset, tf.data.Dataset, int, int]
    """
    preprocessor = create_tokenizer(
        encoding_name=config.encoding_name,
        max_length=config.max_seq_length,
    )
    encoder = preprocessor.tokenizer
    eot_token_id = int(encoder.eot_token)
    encoding_name = getattr(encoder, "name", None) or config.encoding_name

    train_text, val_text, n_train, n_val = load_wikipedia_train_val(
        cache_dir=config.dataset_root,
        config_name=config.wikipedia_config,
        min_article_length=config.min_article_length,
        val_fraction=config.val_fraction,
        max_val_samples=config.max_val_samples,
        max_train_samples=config.max_train_samples,
        seed=config.seed,
        return_counts=True,
        num_shards=config.shuffle_shards,
    )
    steps_per_epoch = estimate_clm_steps_per_epoch(
        num_articles=n_train,
        max_seq_length=config.max_seq_length,
        batch_size=config.batch_size,
        override=config.steps_per_epoch,
    )
    logger.info(
        f"Zamba2 corpus: {n_train} train / {n_val} val articles; "
        f"steps_per_epoch={steps_per_epoch}, vocab_size={preprocessor.vocab_size}"
    )

    train_ds = preprocess_clm_packed_dataset(
        train_text,
        encoding_name=encoding_name,
        chunk_length=config.packed_window,
        batch_size=config.batch_size,
        eot_token_id=eot_token_id,
        shuffle_buffer=config.shuffle_buffer,
        repeat=True,
    )
    val_ds = preprocess_clm_packed_dataset(
        val_text,
        # Validation order is deterministic on purpose: a shuffled val stream
        # makes two evaluations of the same checkpoint disagree. Not
        # repeated: a finite dataset, evaluated to exhaustion once per
        # epoch, needs no explicit `validation_steps`.
        encoding_name=encoding_name,
        chunk_length=config.packed_window,
        batch_size=config.batch_size,
        eot_token_id=eot_token_id,
        shuffle_buffer=1,
        repeat=False,
    )
    return train_ds, val_ds, steps_per_epoch, preprocessor.vocab_size


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


def build_optimizer(
        config: Zamba2TrainingConfig,
        steps_per_epoch: int,
) -> keras.optimizers.Optimizer:
    """AdamW on a warmup + cosine-decay schedule, through ``optimizer_builder``.

    ``optimizer_builder`` RENAMES the clipping keys, so the clip is passed as
    ``gradient_clipping_by_norm_local`` and never as a literal ``"clipnorm"``:
    an unrecognised key is dropped silently, with no error and no warning
    (``src/train/CLAUDE.md``).

    :param config: The run config.
    :type config: Zamba2TrainingConfig
    :param steps_per_epoch: Steps in one epoch, for the decay horizon.
    :type steps_per_epoch: int
    :returns: The optimizer.
    :rtype: keras.optimizers.Optimizer
    """
    total_steps = max(1, int(config.epochs) * int(steps_per_epoch))
    warmup_steps = int(config.warmup_ratio * total_steps)
    schedule = learning_rate_schedule_builder(
        {
            "type": "cosine_decay",
            "learning_rate": config.learning_rate,
            # DECISION plan-2026-09-12T075714-035fd488/D-006: `decay_steps` is
            # `total_steps - warmup_steps`, NOT `total_steps`. Do not
            # "simplify" it. `WarmupSchedule.__call__` hands the primary
            # schedule `step - warmup_steps`, so `decay_steps` is measured
            # from the END of warmup, not from step 0 -- the same trap
            # `train.hnet.common.build_optimizer` documents and a fresh
            # instance of it here, not a new discovery. Passing the full
            # `total_steps` silently stretches the cosine past the end of
            # the run. See decisions.md D-006.
            "decay_steps": max(1, total_steps - warmup_steps),
            "alpha": config.final_learning_rate / config.learning_rate,
            "warmup_steps": warmup_steps,
        }
    )
    optimizer_config: Dict[str, Any] = {
        "type": "adamw",
        "weight_decay": config.weight_decay,
        "exclude_from_weight_decay": list(WEIGHT_DECAY_EXCLUDED),
    }
    if config.gradient_clip_norm > 0.0:
        optimizer_config["gradient_clipping_by_norm_local"] = (
            config.gradient_clip_norm
        )
    return optimizer_builder(optimizer_config, schedule)


def build_model(
        config: Zamba2TrainingConfig,
        steps_per_epoch: int,
        vocab_size: int,
) -> CausalLanguageModel:
    """Create and compile the Zamba2 causal-LM model for one run.

    :param config: The run config.
    :type config: Zamba2TrainingConfig
    :param steps_per_epoch: Steps in one epoch, for the decay horizon.
    :type steps_per_epoch: int
    :param vocab_size: The live tokenizer's vocab size
        (:func:`build_datasets`'s fourth return value) -- passed as an
        explicit override so the model's embedding/head always match the
        encoding actually used to pack the corpus, rather than trusting the
        variant table's own ``DEFAULT_VOCAB_SIZE``.
    :type vocab_size: int
    :returns: The compiled model, a
        :class:`~dl_techniques.models.language.masked_language_model.clm.CausalLanguageModel`
        wrapping a bare :class:`Zamba2Model` backbone. ``skip_head=True``
        since ``Zamba2Model.call()`` already bakes its own tied head and
        returns logits directly as a plain tensor -- no second,
        differently-initialized head is built. ``pre_shifted=True`` matches
        ``preprocess_clm_packed_dataset``'s own pre-shifted
        ``(input_ids, labels)`` tuples. ``causality_probe_plain_tensor=True``
        because ``Zamba2Model.call(self, input_ids, training=None)`` accepts
        only a plain positional tensor, never the
        ``{"input_ids": ..., "attention_mask": ...}`` dict shape the probe
        defaults to (see
        ``CausalLanguageModel``'s ``causality_probe_plain_tensor`` docstring,
        plan-2026-09-13T052422-19022ba2/D-003). ``compile()`` receives only
        the optimizer: the class tracks its own loss/accuracy/perplexity,
        reading ``loss_fn`` internally rather than a compiled ``loss=``.
    :rtype: CausalLanguageModel
    """
    backbone = create_zamba2(
        variant=config.variant,
        vocab_size=vocab_size,
        max_seq_len=config.max_seq_length,
    )
    model = CausalLanguageModel(
        backbone=backbone,
        vocab_size=vocab_size,
        skip_head=True,
        pre_shifted=True,
        loss_fn=create_clm_loss_fn(config),
        causality_probe_plain_tensor=True,
        verify_causality=True,
    )
    model.compile(optimizer=build_optimizer(config, steps_per_epoch))
    # DECISION plan-2026-09-13T052422-19022ba2/D-004 (zamba2 migration): mamba's
    # D-008 eager-dummy-forward pattern, reused here because it is NOT specific
    # to skip_head=False. `skip_head=True` still routes `build()`'s causality
    # probe (`_verify_backbone_causality`) through `_backbone_forward`, whose
    # `ops.convert_to_numpy` call raises `NotImplementedError` on a symbolic
    # tensor if the first build happens inside `fit()`'s traced `train_step`
    # `tf.function` rather than eagerly. Zamba2 had NO dummy-forward call at
    # all before this migration -- unlike Mamba2/gemma/qwen, whose migrations
    # each added one. Do NOT remove this call; see decisions.md D-008 (prior
    # plan-2026-09-12T195532-422091c3) for the measured trace-boundary crash
    # this avoids.
    model(tf.zeros((1, 2), dtype="int32"), training=False)
    return model


# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------


def train(config: Zamba2TrainingConfig) -> Tuple[CausalLanguageModel, Any, str]:
    """Pretrain Zamba2 on Wikipedia with stock ``fit()``.

    :param config: The run config.
    :type config: Zamba2TrainingConfig
    :returns: ``(model, history, results_dir)``.
    :rtype: Tuple[CausalLanguageModel, Any, str]
    """
    set_seeds(config.seed)

    train_ds, val_ds, steps_per_epoch, vocab_size = build_datasets(config)
    model = build_model(config, steps_per_epoch, vocab_size)

    # Calling `create_callbacks` directly rather than `train.common.nlp
    # .create_nlp_callbacks`: the latter has no `output_root` parameter (its
    # signature is `model_name, results_dir_prefix, monitor, patience,
    # include_analyzer, analyzer_epoch_frequency, analyzer_start_epoch` --
    # nothing forwards a custom root), so it could never honour
    # `--output-dir`. `create_callbacks` is what it wraps anyway
    # (`train.common.nlp.create_nlp_callbacks` body), with the same NLP
    # defaults (`use_lr_schedule=True`, `include_tensorboard=True`) spelled
    # out explicitly here instead of through the narrower wrapper.
    callbacks, results_dir = create_callbacks(
        model_name=config.variant,
        results_dir_prefix=RESULTS_DIR_PREFIX,
        output_root=config.output_dir,
        monitor=TRAIN_MONITOR,
        patience=config.patience,
        use_lr_schedule=True,
        include_tensorboard=True,
    )
    save_config_json(config, results_dir, "config.json")

    history = model.fit(
        train_ds,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )
    save_training_history_json(history, results_dir)
    return model, history, results_dir
