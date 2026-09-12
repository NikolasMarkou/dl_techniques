"""Shared building blocks for the Gemma3 Pattern-3 (subword CLM) trainer.

Gemma3 (``dl_techniques.models.language.gemma.gemma3.Gemma3``) is a standard
subword causal LM, so this module follows the Pattern-3 shape
(``src/train/CLAUDE.md``, exemplar ``src/train/bert/pretrain.py``), the same
as ``src/train/zamba2/common.py`` / ``src/train/mamba/common.py`` -- and
deliberately mirrors THAT module's structure (config dataclass + argparse +
``config_from_args`` single-wiring-site + ``build_datasets``/
``build_optimizer``/``build_model``/``train``), which is the more current
convention for a new Pattern-3/6 trainer per ``src/train/CLAUDE.md``.

Unlike ``Mamba2`` (D-004: a pure encoder with no CLM head, requiring a local
functional wrapper), ``Gemma3.call()`` ALREADY bakes its own independent,
untied ``Dense`` LM head and returns a PLAIN TENSOR of logits, shape
``(batch, seq_len, vocab_size)`` (confirmed by direct read,
``gemma3.py:416-441``; also stated in the module docstring: "The LM head is
an independent ``Dense``, not the tied embedding matrix.") -- the same shape
GPT-2/Zamba2/wave_field/hnet already produce. So this module needs no
``build_causal_lm_model``-style wrapper: :func:`build_model` calls
``Gemma3.from_variant`` directly, exactly like ``zamba2/common.py``'s
``build_model`` calls ``create_zamba2`` directly. See
``plans/plan-2026-09-12T173329-e20362c4/decisions.md`` D-005 (queues the
CLM-head consolidation onto ``CausalLanguageModel`` for iter-2, out of scope
here -- Gemma3 keeps its baked-in head unchanged).

Two things this module deliberately does NOT do, for the same reasons
``zamba2/common.py`` gives:

* **No custom ``train_step``.** Gemma3 has no auxiliary loss; the next-token
  cross-entropy reaches the optimizer through stock ``compile()``/``fit()``
  alone.
* **No ``ClmPretrainConfig``/``load_train_val_datasets`` reuse.** That
  wrapper wraps every label tensor as ``{"logits": y}`` because its four
  DICT-output callers (GPT-2, wave_field, cliffordnet) already bake an LM
  head into their own ``call()`` and return ``{"logits": ...}`` (a dict).
  ``Gemma3.call`` returns a PLAIN TENSOR of logits, not a dict, so this
  module calls the leaf helpers directly (``load_wikipedia_train_val``,
  ``preprocess_clm_packed_dataset``, ``estimate_clm_steps_per_epoch``), the
  same as ``zamba2/common.py`` and ``mamba/common.py``.

Public surface:
    * :data:`VARIANT_NAMES` -- the shipped :data:`Gemma3.MODEL_VARIANTS` keys.
    * :class:`Gemma3TrainingConfig` -- the run knobs. Every field is
      consumed by something other than the config dump.
    * :func:`add_common_arguments` -- the shared CLI flags.
    * :func:`config_from_args` -- namespace -> config, the ONE wiring site.
    * :func:`build_datasets` -- the Wikipedia packed-CLM pipeline (identical
      shape to zamba2's/mamba's).
    * :func:`build_optimizer` / :func:`build_model` -- AdamW through
      ``optimizer_builder``, compiled with the shared CLM loss + metrics.
    * :func:`train` -- stock ``fit()``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import keras
import tensorflow as tf

from dl_techniques.datasets.nlp import (
    DEFAULT_WIKIPEDIA_CACHE_DIR,
    DEFAULT_WIKIPEDIA_CONFIG,
    load_wikipedia_train_val,
)
from dl_techniques.models.language.gemma.gemma3 import Gemma3
from dl_techniques.optimization import (
    learning_rate_schedule_builder,
    optimizer_builder,
)
from dl_techniques.utils.logger import logger
from train.common import create_callbacks, set_seeds
from train.common.clm_pretrain import create_clm_loss_fn
from train.common.config_io import save_config_json
from train.common.nlp import (
    build_clm_metrics,
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
    "Gemma3TrainingConfig",
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

RESULTS_DIR_PREFIX: str = "gemma"
"""Prefix of the timestamped run directory under ``--output-dir``."""

TRAIN_MONITOR: str = "val_loss"
"""The monitored metric. Its direction is resolved by
``train.common.callbacks.resolve_monitor_mode``, never hand-written here."""

WEIGHT_DECAY_EXCLUDED: Tuple[str, ...] = ("bias", "gamma", "beta")
"""Variable-name patterns kept out of AdamW's decoupled decay (matches
``train.zamba2.common``'s convention: Keras spells LayerNorm/RMSNorm
parameters ``gamma``/``beta``). Weight decay is applied by the optimizer and
by NOTHING else -- no ``kernel_regularizer`` is ever attached."""

DEFAULT_VARIANT: str = "tiny"
"""Default ``--variant`` -- the smallest shipped size, for fast iteration."""

VARIANT_NAMES: Tuple[str, ...] = tuple(Gemma3.MODEL_VARIANTS)
"""Every ``--variant`` choice: the three :data:`Gemma3.MODEL_VARIANTS` keys
(``270m``, ``small``, ``tiny``), in declaration order. Unlike ``Mamba2``,
``Gemma3`` declares no ``VARIANT_ALIASES``."""


def variant_names() -> Tuple[str, ...]:
    """:returns: :data:`VARIANT_NAMES`, as the CLI's ``choices``.
    :rtype: Tuple[str, ...]
    """
    return VARIANT_NAMES


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass
class Gemma3TrainingConfig:
    """Knobs for one Gemma3 pretraining run.

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
        token to spend. Also forwarded as ``Gemma3``'s ``max_seq_len`` (the
        RoPE table horizon).
    :param batch_size: Windows per optimizer step.
    :param epochs: Training epochs.
    :param steps_per_epoch: Override for the estimated steps per epoch;
        ``None`` derives it from the post-filter article count.
    :param hidden_size: Overrides the variant's hidden width. ``None`` keeps
        the variant's own value; a smoke run passes a tiny value here. Must
        stay divisible by whatever ``num_attention_heads`` resolves to
        (``Gemma3.__init__`` raises ``ValueError`` otherwise) -- this config
        does not re-validate that interaction, since it does not know the
        variant's other knobs; the model constructor is the single source of
        truth for it.
    :param num_layers: Overrides the variant's block count. When set, the
        variant's ``layer_types`` list (whose length must match
        ``num_layers``) is also cleared to ``None`` in
        :attr:`variant_overrides`, so ``Gemma3.__init__`` regenerates a
        same-length default (all ``"full_attention"``) instead of raising on
        a length mismatch against the ORIGINAL variant's list.
    :param num_attention_heads: Overrides the variant's attention head count.
    :param num_key_value_heads: Overrides the variant's GQA key/value head
        count.
    :param ffn_hidden_size: Overrides the variant's FFN intermediate size.
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

    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    ffn_hidden_size: Optional[int] = None

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
        if self.variant not in VARIANT_NAMES:
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
        for name, value in (
            ("hidden_size", self.hidden_size),
            ("num_layers", self.num_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("ffn_hidden_size", self.ffn_hidden_size),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None, got {value}")
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

    @property
    def variant_overrides(self) -> Dict[str, Any]:
        """:returns: The non-``None`` architecture overrides
            (``hidden_size``/``num_layers``/``num_attention_heads``/
            ``num_key_value_heads``/``ffn_hidden_size``) as a kwargs dict for
            :meth:`Gemma3.from_variant`. When ``num_layers`` is overridden,
            ``layer_types`` is ALSO set to ``None`` here so
            ``Gemma3.from_variant``'s ``config.update(kwargs)`` replaces the
            variant's original (now length-mismatched) ``layer_types`` list
            rather than leaving it in place -- ``Gemma3.__init__`` then
            regenerates a same-length default (see the ``num_layers``
            docstring above).
        :rtype: Dict[str, Any]
        """
        overrides: Dict[str, Any] = {}
        if self.hidden_size is not None:
            overrides["hidden_size"] = self.hidden_size
        if self.num_layers is not None:
            overrides["num_layers"] = self.num_layers
            overrides["layer_types"] = None
        if self.num_attention_heads is not None:
            overrides["num_attention_heads"] = self.num_attention_heads
        if self.num_key_value_heads is not None:
            overrides["num_key_value_heads"] = self.num_key_value_heads
        if self.ffn_hidden_size is not None:
            overrides["ffn_hidden_size"] = self.ffn_hidden_size
        return overrides


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the shared Gemma3 training flags.

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
    defaults = Gemma3TrainingConfig()

    parser.add_argument(
        "--variant", type=str, default=defaults.variant,
        choices=list(variant_names()),
        help="Gemma3 model size to train.",
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
        "--hidden-size", type=int, default=defaults.hidden_size,
        help="Override the variant's hidden width (e.g. for a tiny smoke run).",
    )
    parser.add_argument(
        "--num-layers", type=int, default=defaults.num_layers,
        help="Override the variant's block count.",
    )
    parser.add_argument(
        "--num-attention-heads", type=int, default=defaults.num_attention_heads,
        help="Override the variant's attention head count.",
    )
    parser.add_argument(
        "--num-key-value-heads", type=int, default=defaults.num_key_value_heads,
        help="Override the variant's GQA key/value head count.",
    )
    parser.add_argument(
        "--ffn-hidden-size", type=int, default=defaults.ffn_hidden_size,
        help="Override the variant's FFN intermediate size.",
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


def config_from_args(args: argparse.Namespace) -> Gemma3TrainingConfig:
    """Build a config from a parsed namespace.

    The ONE wiring site between :func:`add_common_arguments` and
    :class:`Gemma3TrainingConfig`. A flag that does not arrive here silently
    does nothing.

    :param args: A namespace produced by a parser carrying the common flags.
    :type args: argparse.Namespace
    :returns: The config.
    :rtype: Gemma3TrainingConfig
    """
    return Gemma3TrainingConfig(
        variant=args.variant,
        dataset_root=args.dataset_root,
        wikipedia_config=args.wikipedia_config,
        encoding_name=args.encoding_name,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        ffn_hidden_size=args.ffn_hidden_size,
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
        config: Gemma3TrainingConfig,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """Build the Wikipedia packed-CLM pipeline, the step budget, and vocab_size.

    Identical shape to ``train.zamba2.common.build_datasets`` /
    ``train.mamba.common.build_datasets``.

    :param config: The run config.
    :type config: Gemma3TrainingConfig
    :returns: ``(train_ds, val_ds, steps_per_epoch, vocab_size)``. ``vocab_size``
        is read off the live tokenizer (:attr:`TiktokenPreprocessor.vocab_size`)
        rather than hardcoded, so the model's embedding/head always match the
        encoding actually used to pack the corpus.
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
        f"Gemma3 corpus: {n_train} train / {n_val} val articles; "
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
        config: Gemma3TrainingConfig,
        steps_per_epoch: int,
) -> keras.optimizers.Optimizer:
    """AdamW on a warmup + cosine-decay schedule, through ``optimizer_builder``.

    ``optimizer_builder`` RENAMES the clipping keys, so the clip is passed as
    ``gradient_clipping_by_norm_local`` and never as a literal ``"clipnorm"``:
    an unrecognised key is dropped silently, with no error and no warning
    (``src/train/CLAUDE.md``).

    :param config: The run config.
    :type config: Gemma3TrainingConfig
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
            # `decay_steps` is `total_steps - warmup_steps`, NOT
            # `total_steps` -- see zamba2/common.py's build_optimizer for the
            # measured trap this avoids (WarmupSchedule hands the primary
            # schedule `step - warmup_steps`).
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
        config: Gemma3TrainingConfig,
        steps_per_epoch: int,
        vocab_size: int,
) -> Gemma3:
    """Create and compile the Gemma3 causal-LM model for one run.

    Gemma3 already bakes its own LM head (see the module docstring), so this
    is a direct ``Gemma3.from_variant`` call, exactly like
    ``zamba2/common.py``'s ``build_model`` calling ``create_zamba2``
    directly -- no local head-wrapper is needed (unlike
    ``mamba/common.py``'s ``build_causal_lm_model``, D-004).

    :param config: The run config.
    :type config: Gemma3TrainingConfig
    :param steps_per_epoch: Steps in one epoch, for the decay horizon.
    :type steps_per_epoch: int
    :param vocab_size: The live tokenizer's vocab size
        (:func:`build_datasets`'s fourth return value) -- passed as an
        explicit override so the model's embedding/head always match the
        encoding actually used to pack the corpus, rather than trusting the
        variant table's own ``vocab_size``.
    :type vocab_size: int
    :returns: The compiled model.
    :rtype: Gemma3
    """
    model = Gemma3.from_variant(
        config.variant,
        vocab_size=vocab_size,
        max_seq_len=config.max_seq_length,
        **config.variant_overrides,
    )
    model.compile(
        optimizer=build_optimizer(config, steps_per_epoch),
        loss=create_clm_loss_fn(config),
        metrics=build_clm_metrics(config.encoding_name),
    )
    return model


# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------


def train(config: Gemma3TrainingConfig) -> Tuple[Gemma3, Any, str]:
    """Pretrain Gemma3 on Wikipedia with stock ``fit()``.

    :param config: The run config.
    :type config: Gemma3TrainingConfig
    :returns: ``(model, history, results_dir)``.
    :rtype: Tuple[Gemma3, Any, str]
    """
    set_seeds(config.seed)

    train_ds, val_ds, steps_per_epoch, vocab_size = build_datasets(config)
    model = build_model(config, steps_per_epoch, vocab_size)

    # Calling `create_callbacks` directly rather than `train.common.nlp
    # .create_nlp_callbacks`: the latter has no `output_root` parameter, so
    # it could never honour `--output-dir` -- same reasoning as
    # zamba2/common.py's / mamba/common.py's `train`.
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
