"""Shared building blocks for the H-Net byte-level language-model trainer.

The whole pipeline is bytes end to end. There is no tokenizer anywhere in it:
``dl_techniques.datasets.nlp.load_wikipedia_train_val`` already returns raw
UTF-8 strings, and ``dl_techniques.datasets.byte_lm`` turns those into the
packed causal-LM windows the model consumes. Nothing here re-implements a
packer, an encoder or a steps-per-epoch estimator -- all four live in
``byte_lm`` and are imported.

Two things this module deliberately does NOT do:

* **No custom ``train_step``.** The ratio loss reaches the optimizer through
  ``HNet.call``'s ``add_loss`` (``model.py:461-466``); stock ``fit()`` already
  sums ``model.losses`` into the compiled loss. A hand-written training step
  would also silently skip ``scale_loss`` under ``mixed_float16``.
* **No per-stage learning-rate multipliers.** The reference attaches an
  ``_optim["lr_multiplier"]`` attribute per parameter (``hnet.py:150-159``,
  ``mixer_seq.py:64-76``) which ``hnet/utils/train.py:group_params`` later
  turns into per-multiplier optimizer param-groups. Keras 3's optimizer takes
  ONE learning rate for all variables, so honouring that would mean either a
  custom optimizer or several optimizers stepping disjoint variable sets --
  neither of which stock ``fit()` supports. The knob is therefore **absent
  from the config rather than declared and unread**: a field nothing consumes
  is a knob that silently does nothing (``src/train/CLAUDE.md`` § Config
  fields must be live). The reference's own ``configs/*.json`` carry no
  ``lr_multiplier`` values either, so there is not even a number to port. See
  ``decisions.md`` D-026.

Public surface:
    * :data:`ARCH_VARIANTS` / :func:`get_arch_config` -- the six shipped
      variants plus the tiny ``dev`` layout the smoke run and the suite use.
    * :class:`HNetTrainingConfig` -- the run knobs. Every field is consumed by
      something other than the config dump.
    * :func:`add_common_arguments` -- the shared CLI flags.
    * :func:`config_from_args` -- namespace -> config, the ONE wiring site.
    * :func:`build_datasets` -- the tf.data byte pipeline.
    * :func:`build_optimizer` / :func:`build_model` -- AdamW through
      ``optimizer_builder``, compiled with a from-logits sparse CE.
    * :func:`train` -- stock ``fit()``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import keras
import tensorflow as tf

from dl_techniques.datasets.byte_lm import (
    BYTE_VOCAB_SIZE,
    DEFAULT_DATASET_ROOT,
    DEFAULT_SHUFFLE_BUFFER,
    build_byte_clm_dataset,
    estimate_byte_clm_steps_per_epoch,
)
from dl_techniques.datasets.nlp import (
    DEFAULT_WIKIPEDIA_CONFIG,
    load_wikipedia_train_val,
)
from dl_techniques.models.language.hnet.config import (
    MODEL_VARIANTS,
    AttnSpec,
    HNetArchConfig,
    SSMSpec,
)
from dl_techniques.models.language.hnet.losses import DEFAULT_TARGET_RATIO
from dl_techniques.models.language.hnet.model import RATIO_LOSS_ALPHA, HNet
from dl_techniques.optimization import (
    learning_rate_schedule_builder,
    optimizer_builder,
)
from dl_techniques.utils.logger import logger
from train.common import create_callbacks, set_seeds
from train.common.config_io import save_config_json
from train.common.run_io import save_training_history_json

__all__ = [
    "ARCH_VARIANTS",
    "DEV_ARCH_CONFIG",
    "DEV_VARIANT",
    "RESULTS_DIR_PREFIX",
    "TRAIN_MONITOR",
    "WEIGHT_DECAY_EXCLUDED",
    "HNetTrainingConfig",
    "add_common_arguments",
    "arch_variant_names",
    "build_datasets",
    "build_model",
    "build_optimizer",
    "config_from_args",
    "get_arch_config",
    "train",
]

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

RESULTS_DIR_PREFIX: str = "hnet"
"""Prefix of the timestamped run directory under ``--output-dir``."""

TRAIN_MONITOR: str = "val_loss"
"""The monitored metric. Its DIRECTION is never spelled here: ``create_callbacks``
resolves it through ``train.common.resolve_monitor_mode``, the one producer of a
checkpoint-selection direction (``src/train/CLAUDE.md``). Hand-writing
``mode='min'`` beside a monitor name is how a run ends up restoring its worst
epoch when the monitor later changes."""

WEIGHT_DECAY_EXCLUDED: Tuple[str, ...] = ("bias", "gamma", "beta")
"""Variable-name patterns kept out of AdamW's decoupled decay.

The reference does the same thing by hand -- ``hnet/utils/train.py:67-69``
zeroes ``weight_decay`` for any parameter whose name contains ``.bias`` or
``.norm.``. Keras spells the two normalisation parameters ``gamma`` and
``beta``, which is why the patterns differ from upstream's spelling while the
SET of excluded tensors matches.

Weight decay is applied by the optimizer and by NOTHING else: no
``kernel_regularizer`` is ever attached, because AdamW's decoupled decay plus
an L2 penalty decays the same parameter twice (``src/train/CLAUDE.md``)."""

DEV_VARIANT: str = "dev"
"""Name of the tiny, non-reference architecture below."""

DEV_ARCH_CONFIG: HNetArchConfig = HNetArchConfig(
    arch_layout=["m1", ["T1"], "m1"],
    d_model=(64, 64),
    d_intermediate=(0, 0),
    vocab_size=BYTE_VOCAB_SIZE,
    ssm_cfg=SSMSpec(d_conv=4, expand=2, d_state=16),
    attn_cfg=AttnSpec(
        num_heads=(4, 4), rotary_emb_dim=(8, 8), window_size=(-1, -1)
    ),
    # UNTIED, like all six reference variants. Tying is not free at this
    # scale: the embedding table is initialised at unit standard deviation
    # (``model.py:EMBEDDING_INIT_STDDEV``, upstream's own choice), so a tied
    # head emits logits of that scale and the model STARTS at a measured
    # cross-entropy of 40.5 instead of ln(256) = 5.55. That is a fine place
    # to start a 100B-token run from and a terrible one for a smoke run whose
    # whole claim is a loss trend.
    tie_embeddings=False,
)
"""A dev-scale H-Net: one Mamba-2 encoder layer, one chunking level, one
attention layer inside, one Mamba-2 decoder layer, at ``d_model = 64``.

**It is deliberately NOT a row of**
:data:`~dl_techniques.models.language.hnet.config.MODEL_VARIANTS`. Every entry
of that table is transcribed from a named ``configs/*.json`` in the reference
repository and cited as such; this layout is invented here, for the smoke run
and the test suite. Adding it upstream of the citation boundary would make the
model package's variant table half-cited, which is worse than a trainer-local
constant. The six real variants all start at ``d_model = 1024`` with 22+ inner
layers -- none of them fits a unit test or a minutes-long smoke run."""

ARCH_VARIANTS: Tuple[str, ...] = (DEV_VARIANT,) + tuple(sorted(MODEL_VARIANTS))
"""Every name ``--arch-variant`` accepts, dev first."""


def arch_variant_names() -> Tuple[str, ...]:
    """:returns: :data:`ARCH_VARIANTS`, as the CLI's ``choices``.
    :rtype: Tuple[str, ...]
    """
    return ARCH_VARIANTS


def get_arch_config(variant: str) -> HNetArchConfig:
    """Resolve an architecture name to its config.

    The ONE place a trainer-side name becomes an architecture, so ``dev`` and
    the six reference variants cannot be resolved differently by two call
    sites.

    :param variant: A member of :data:`ARCH_VARIANTS`.
    :type variant: str
    :returns: The architecture description.
    :rtype: HNetArchConfig
    :raises ValueError: if ``variant`` is not a known name; the message lists
        every available name.
    """
    if variant == DEV_VARIANT:
        return DEV_ARCH_CONFIG
    if variant not in MODEL_VARIANTS:
        raise ValueError(
            f"unknown arch_variant {variant!r}; available: {list(ARCH_VARIANTS)}"
        )
    return MODEL_VARIANTS[variant]


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass
class HNetTrainingConfig:
    """Knobs for one H-Net pretraining run.

    Every annotated field is read by something other than the config dump.
    ``asdict`` / ``save_config_json`` / ``prepare_run_dir`` serialize the whole
    config, which RECORDS a field without consuming it; a field that reaches
    nothing else is deleted rather than wired (``src/train/CLAUDE.md``,
    ``tests/test_train/test_config_fields_are_live.py``). That is why there is
    no ``chunk_size`` here -- the Mamba-2 chunk size the reference JSONs carry
    is not a knob this port has (``SSMSpec`` drops it) -- and no
    ``lr_multiplier``; see the module docstring.

    :param arch_variant: A member of :data:`ARCH_VARIANTS`.
    :param dataset_root: Arrow cache directory holding the Wikipedia dump.
    :param wikipedia_config: Wikipedia dump config name, e.g. ``20231101.en``.
    :param seq_len: Byte window the MODEL sees. The packer is asked for
        ``seq_len + 1`` bytes so the causal shift has a byte to spend; see
        :func:`build_datasets`.
    :param batch_size: Windows per optimizer step.
    :param epochs: Training epochs.
    :param steps_per_epoch: Override for the estimated steps per epoch;
        ``None`` derives it from the post-filter article count.
    :param validation_steps: Validation batches per epoch.
    :param learning_rate: Peak learning rate, reached at the end of warmup.
    :param final_learning_rate: Cosine floor.
    :param weight_decay: Decoupled AdamW weight decay. Applied by the optimizer
        ONLY -- never also as a ``kernel_regularizer``.
    :param warmup_ratio: Fraction of total steps spent warming up.
    :param gradient_clip_norm: Per-variable gradient-norm clip, or ``0`` to
        disable. Passed to ``optimizer_builder`` under its OWN key name; see
        :func:`build_optimizer`.
    :param headdim: Mamba-2 head dimension. Must divide ``expand * d_model`` of
        every stage, which is why the dev layout cannot simply take the 64 the
        reference variants use.
    :param ratio_loss_alpha: Weight of the boundary-ratio auxiliary loss.
    :param target_ratio: Target downsampling factor at every chunking level.
    :param max_chunks: One fixed chunk cap per chunking level, outermost first.
        ``None`` lets ``HNet`` derive them from :attr:`seq_len` (D-007).
    :param min_article_length: Articles shorter than this many CHARACTERS are
        skipped. ``0`` (the packed-CLM default) keeps every byte.
    :param max_train_samples: Cap on training articles, ``None`` for all.
    :param max_val_samples: Cap on validation articles.
    :param val_fraction: Fraction of articles held out for validation.
    :param shuffle_shards: Parallel Wikipedia shards; ``>1`` reshuffles the
        article order at every epoch boundary.
    :param shuffle_buffer: tf.data shuffle buffer over packed byte windows.
    :param seed: Seed for ``set_seeds`` and for the corpus split.
    :param patience: Early-stopping patience.
    :param output_dir: Root under which the timestamped run directory is made.
    """

    arch_variant: str = DEV_VARIANT
    dataset_root: str = DEFAULT_DATASET_ROOT
    wikipedia_config: str = DEFAULT_WIKIPEDIA_CONFIG

    seq_len: int = 512
    batch_size: int = 8
    epochs: int = 1
    steps_per_epoch: Optional[int] = None
    validation_steps: int = 20

    learning_rate: float = 3e-4
    final_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.02
    gradient_clip_norm: float = 1.0

    headdim: int = 64
    ratio_loss_alpha: float = RATIO_LOSS_ALPHA
    target_ratio: float = DEFAULT_TARGET_RATIO
    max_chunks: Optional[List[int]] = None

    min_article_length: int = 0
    max_train_samples: Optional[int] = None
    max_val_samples: int = 5000
    val_fraction: float = 0.02
    shuffle_shards: int = 4
    shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER

    seed: int = 42
    patience: int = 5
    output_dir: str = "results"

    def __post_init__(self) -> None:
        """Validate the knobs at construction time.

        :raises ValueError: on an unknown variant, a non-positive count, a
            ratio outside its range, or a ``max_chunks`` list whose length
            disagrees with the architecture's chunking-level count.
        """
        get_arch_config(self.arch_variant)  # raises, listing the legal names

        if self.seq_len < 2:
            raise ValueError(
                f"seq_len must be >= 2 (the causal shift spends one byte), got "
                f"{self.seq_len}"
            )
        positive = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "validation_steps": self.validation_steps,
            "headdim": self.headdim,
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
                f"gradient_clip_norm must be non-negative (0 disables it), got "
                f"{self.gradient_clip_norm}"
            )
        if self.ratio_loss_alpha < 0.0:
            raise ValueError(
                f"ratio_loss_alpha must be non-negative, got "
                f"{self.ratio_loss_alpha}"
            )
        if self.target_ratio <= 1.0:
            raise ValueError(
                f"target_ratio must exceed 1.0 (it is a downsampling factor), "
                f"got {self.target_ratio}"
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
        if self.max_chunks is not None:
            self.max_chunks = [int(value) for value in self.max_chunks]
            expected = self.arch_config.num_stages - 1
            if len(self.max_chunks) != expected:
                raise ValueError(
                    f"max_chunks needs one entry per chunking level: "
                    f"{self.arch_variant!r} has {expected}, got "
                    f"{len(self.max_chunks)} ({self.max_chunks})"
                )
            if any(value < 1 for value in self.max_chunks):
                raise ValueError(
                    f"every max_chunks entry must be >= 1, got {self.max_chunks}"
                )

    @property
    def arch_config(self) -> HNetArchConfig:
        """:returns: The architecture named by :attr:`arch_variant`.
        :rtype: HNetArchConfig
        """
        return get_arch_config(self.arch_variant)

    @property
    def packed_window(self) -> int:
        """:returns: The window length asked of the packer -- one byte more
            than the model's :attr:`seq_len`, because
            ``build_byte_clm_dataset`` spends one on the causal shift.
        :rtype: int
        """
        return self.seq_len + 1


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the shared H-Net training flags.

    ``--gpu`` is deliberately NOT here: it is consumed by ``setup_gpu`` in
    ``main()`` and is not a config field.

    The four flags every CLM script in this tree exposes so users can switch
    scripts without relearning -- ``--steps-per-epoch``, ``--seed``,
    ``--min-article-length``, ``--shuffle-shards`` -- are all present and
    spelled identically (``src/train/CLAUDE.md``).

    :param parser: The parser to extend.
    :type parser: argparse.ArgumentParser
    :returns: The same parser, for chaining.
    :rtype: argparse.ArgumentParser
    """
    defaults = HNetTrainingConfig()

    parser.add_argument(
        "--arch-variant", type=str, default=defaults.arch_variant,
        choices=list(arch_variant_names()),
        help="Architecture to train ('dev' is the tiny smoke-scale layout).",
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
        "--seq-len", type=int, default=defaults.seq_len,
        help="Byte window the model sees.",
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
        "--validation-steps", type=int, default=defaults.validation_steps,
        help="Validation batches per epoch.",
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
        "--headdim", type=int, default=defaults.headdim,
        help="Mamba-2 head dimension.",
    )
    parser.add_argument(
        "--ratio-loss-alpha", type=float, default=defaults.ratio_loss_alpha,
        help="Weight of the boundary-ratio auxiliary loss.",
    )
    parser.add_argument(
        "--target-ratio", type=float, default=defaults.target_ratio,
        help="Target downsampling factor per chunking level.",
    )
    parser.add_argument(
        "--max-chunks", type=int, nargs="+", default=defaults.max_chunks,
        help="Fixed chunk cap per chunking level, outermost first.",
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
        help="tf.data shuffle buffer over packed byte windows.",
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
        help="Root for the timestamped run directory.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> HNetTrainingConfig:
    """Build a config from a parsed namespace.

    The ONE wiring site between :func:`add_common_arguments` and
    :class:`HNetTrainingConfig`. A flag that does not arrive here silently does
    nothing, which ``tests/test_train/test_hnet/test_pipeline.py`` pins field by
    field rather than by spot check.

    :param args: A namespace produced by a parser carrying the common flags.
    :type args: argparse.Namespace
    :returns: The config.
    :rtype: HNetTrainingConfig
    """
    return HNetTrainingConfig(
        arch_variant=args.arch_variant,
        dataset_root=args.dataset_root,
        wikipedia_config=args.wikipedia_config,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_clip_norm=args.gradient_clip_norm,
        headdim=args.headdim,
        ratio_loss_alpha=args.ratio_loss_alpha,
        target_ratio=args.target_ratio,
        max_chunks=args.max_chunks,
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


def build_byte_datasets(
        train_text: tf.data.Dataset,
        val_text: tf.data.Dataset,
        config: HNetTrainingConfig,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Layer the byte packer onto two datasets of raw text.

    Split out from :func:`build_datasets` so the transform can be exercised on
    an in-memory corpus without touching the 20 GB Wikipedia cache -- which is
    exactly what the suite does.

    Both datasets ``repeat()``: ``fit()`` is given an explicit
    ``steps_per_epoch`` and ``validation_steps``, and a finite dataset under a
    fixed step budget ends the epoch early and silently.

    :param train_text: Dataset of raw training text (string tensors).
    :type train_text: tf.data.Dataset
    :param val_text: Dataset of raw validation text (string tensors).
    :type val_text: tf.data.Dataset
    :param config: The run config.
    :type config: HNetTrainingConfig
    :returns: ``(train_ds, val_ds)``, each yielding ``(input_ids, labels)``
        ``int32`` tensors of shape ``(batch_size, seq_len)``.
    :rtype: Tuple[tf.data.Dataset, tf.data.Dataset]
    """
    train_ds = build_byte_clm_dataset(
        train_text,
        seq_len=config.packed_window,
        batch_size=config.batch_size,
        shuffle_buffer=config.shuffle_buffer,
        repeat=True,
    )
    val_ds = build_byte_clm_dataset(
        val_text,
        # Validation order is deterministic on purpose: a shuffled val stream
        # makes two evaluations of the same checkpoint disagree.
        seq_len=config.packed_window,
        batch_size=config.batch_size,
        shuffle_buffer=1,
        repeat=True,
    )
    return train_ds, val_ds


def build_datasets(
        config: HNetTrainingConfig,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """Build the Wikipedia byte pipeline and the step budget.

    :param config: The run config.
    :type config: HNetTrainingConfig
    :returns: ``(train_ds, val_ds, steps_per_epoch)``.
    :rtype: Tuple[tf.data.Dataset, tf.data.Dataset, int]
    """
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
    steps_per_epoch = estimate_byte_clm_steps_per_epoch(
        num_articles=n_train,
        seq_len=config.packed_window,
        batch_size=config.batch_size,
        override=config.steps_per_epoch,
    )
    logger.info(
        f"H-Net corpus: {n_train} train / {n_val} val articles; "
        f"steps_per_epoch={steps_per_epoch}"
    )
    train_ds, val_ds = build_byte_datasets(train_text, val_text, config)
    return train_ds, val_ds, steps_per_epoch


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


def build_optimizer(
        config: HNetTrainingConfig,
        steps_per_epoch: int,
) -> keras.optimizers.Optimizer:
    """AdamW on a warmup + cosine-decay schedule, through ``optimizer_builder``.

    ``optimizer_builder`` RENAMES the clipping keys, so the clip is passed as
    ``gradient_clipping_by_norm_local`` and never as a literal ``"clipnorm"``:
    an unrecognised key is dropped silently, with no error and no warning, and
    gradient clipping would simply vanish (``src/train/CLAUDE.md``). The
    optimizer is never mutated after construction either -- Keras requires
    clipping in the constructor.

    :param config: The run config.
    :type config: HNetTrainingConfig
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
            # `decay_steps` is measured from the END of warmup, not from step
            # 0: `WarmupSchedule.__call__` hands the primary schedule
            # `step - warmup_steps` (``warmup_schedule.py:174-178``). Passing
            # the full `total_steps` here silently stretches the cosine past
            # the end of the run -- MEASURED at a 2% warmup ratio equivalent:
            # the last step landed at 5.64e-4 instead of the 1e-4 floor, a
            # 5.6x overshoot no shape or type check can see.
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
        config: HNetTrainingConfig,
        steps_per_epoch: int,
) -> HNet:
    """Create and compile the H-Net for one run.

    The compiled loss is the next-byte cross-entropy alone. The auxiliary
    boundary-ratio loss is NOT added here: ``HNet.call`` contributes it through
    ``add_loss``, and stock ``fit()`` sums ``model.losses`` into the total. Any
    attempt to add it a second time at compile time would double-count it.

    :param config: The run config.
    :type config: HNetTrainingConfig
    :param steps_per_epoch: Steps in one epoch, for the decay horizon.
    :type steps_per_epoch: int
    :returns: The compiled model.
    :rtype: HNet
    """
    arch = config.arch_config
    n_levels = arch.num_stages - 1
    model = HNet(
        arch_config=arch,
        max_chunks=config.max_chunks,
        max_seq_len=config.seq_len,
        headdim=config.headdim,
        ratio_loss_alpha=config.ratio_loss_alpha,
        target_ratios=(config.target_ratio,) * n_levels,
    )
    model.compile(
        optimizer=build_optimizer(config, steps_per_epoch),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    return model


# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------


def train(config: HNetTrainingConfig) -> Tuple[HNet, Any, str]:
    """Pretrain H-Net on Wikipedia bytes with stock ``fit()``.

    :param config: The run config.
    :type config: HNetTrainingConfig
    :returns: ``(model, history, results_dir)``.
    :rtype: Tuple[HNet, Any, str]
    """
    set_seeds(config.seed)

    train_ds, val_ds, steps_per_epoch = build_datasets(config)
    model = build_model(config, steps_per_epoch)

    callbacks, results_dir = create_callbacks(
        model_name=config.arch_variant,
        results_dir_prefix=RESULTS_DIR_PREFIX,
        output_root=config.output_dir,
        monitor=TRAIN_MONITOR,
        patience=config.patience,
        use_lr_schedule=True,
    )
    save_config_json(config, results_dir, "config.json")

    history = model.fit(
        train_ds,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=config.validation_steps,
        callbacks=callbacks,
        verbose=1,
    )
    save_training_history_json(history, results_dir)
    return model, history, results_dir
