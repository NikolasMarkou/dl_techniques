"""Shared NLP utilities for training scripts.

Provides tokenizer creation, text data loading/preprocessing, warmup LR schedules,
and NLP-specific callback wrappers. Used by BERT, FNet, and other NLP pretrain/finetune
scripts that share the Tiktoken + TFDS pipeline.
"""

import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tiktoken
from typing import List, Optional, Tuple

from train.common import create_callbacks as create_common_callbacks
from train.common.generation_probe import GenerationProbeCallback

from dl_techniques.utils.logger import logger
from dl_techniques.utils.tokenizer import TiktokenPreprocessor
from dl_techniques.optimization.warmup_schedule import WarmupSchedule
from dl_techniques.metrics.perplexity_metric import Perplexity
from dl_techniques.metrics.llm_metrics import (
    BitsPerToken,
    BitsPerCharacter,
    aggregate_probe_metrics as augment_probe_results,
)


# ---------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------

# Default Tiktoken cl100k_base special token IDs
DEFAULT_CLS_TOKEN_ID = 100264
DEFAULT_SEP_TOKEN_ID = 100265
DEFAULT_PAD_TOKEN_ID = 100266
DEFAULT_MASK_TOKEN_ID = 100267


def create_tokenizer(
    encoding_name: str = "cl100k_base",
    max_length: int = 128,
    cls_token_id: int = DEFAULT_CLS_TOKEN_ID,
    sep_token_id: int = DEFAULT_SEP_TOKEN_ID,
    pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
    mask_token_id: int = DEFAULT_MASK_TOKEN_ID,
) -> TiktokenPreprocessor:
    """Create and configure Tiktoken preprocessor for NLP training."""
    preprocessor = TiktokenPreprocessor(
        encoding_name=encoding_name,
        max_length=max_length,
        cls_token_id=cls_token_id,
        sep_token_id=sep_token_id,
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
        truncation=True,
        padding='max_length',
    )
    logger.info(
        f"TiktokenPreprocessor: vocab_size={preprocessor.vocab_size}, "
        f"encoding={encoding_name}"
    )
    return preprocessor


def decode_text(text) -> str:
    """Decode a TF text tensor to a Python string."""
    if isinstance(text, bytes):
        return text.decode('utf-8')
    if hasattr(text, 'numpy'):
        text_np = text.numpy()
        return text_np.decode('utf-8') if isinstance(text_np, bytes) else str(text_np)
    return str(text)


# ---------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------


def load_text_dataset(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    as_supervised: bool = False,
) -> tf.data.Dataset:
    """Load a text dataset from tensorflow-datasets.

    Args:
        dataset_name: TFDS dataset name (e.g., 'imdb_reviews').
        split: Dataset split ('train', 'test', etc.).
        max_samples: Maximum number of training samples (validation gets 1/5).
        as_supervised: If True, returns (text, label) pairs for classification.
    """
    logger.info(f"Loading {dataset_name} ({split})...")
    dataset, _ = tfds.load(
        dataset_name, split=split,
        as_supervised=as_supervised, shuffle_files=True, with_info=True,
    )

    if not as_supervised:
        dataset = dataset.map(lambda x: x["text"], num_parallel_calls=tf.data.AUTOTUNE)

    if max_samples is not None:
        if split == "train":
            dataset = dataset.take(max_samples)
            logger.info(f"Limited training data to {max_samples} samples")
        else:
            limit = max_samples // 5 if max_samples else 2000
            dataset = dataset.take(limit)

    return dataset


def preprocess_mlm_dataset(
    dataset: tf.data.Dataset,
    preprocessor: TiktokenPreprocessor,
    max_seq_length: int,
    batch_size: int,
) -> tf.data.Dataset:
    """Tokenize and batch a text dataset for MLM (masked language modeling) training.

    Expects a dataset of raw text strings (not supervised).
    Returns batched dataset of {input_ids, attention_mask, token_type_ids} dicts.
    """
    def tokenize_fn(text):
        encoded = preprocessor(decode_text(text), return_tensors='np')
        return encoded['input_ids'][0], encoded['attention_mask'][0], encoded['token_type_ids'][0]

    dataset = dataset.map(
        lambda x: tf.py_function(tokenize_fn, [x], [tf.int32, tf.int32, tf.int32]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    seq_len = max_seq_length
    dataset = dataset.map(
        lambda ids, mask, types: {
            'input_ids': tf.ensure_shape(ids, [seq_len]),
            'attention_mask': tf.ensure_shape(mask, [seq_len]),
            'token_type_ids': tf.ensure_shape(types, [seq_len]),
        },
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    # DECISION D-005: do NOT call dataset.cache() here. Caching the tokenized
    # MLM stream OOMs on Wikipedia-scale corpora (~20 GB tokenized in RAM
    # during the first epoch). The trade-off is per-epoch re-tokenization on
    # bounded TFDS sets, which costs <2 min/epoch on a 4090 host CPU.
    dataset = (
        dataset.shuffle(buffer_size=1000)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    logger.info(f"MLM dataset preprocessed: batch_size={batch_size}")
    return dataset


def preprocess_clm_dataset(
    dataset: tf.data.Dataset,
    preprocessor: TiktokenPreprocessor,
    max_seq_length: int,
    batch_size: int,
) -> tf.data.Dataset:
    """Tokenize, pack, and batch a text dataset for CLM pretraining.

    Implements the standard GPT-style *concat-and-chunk* packing
    pipeline:

    1. Each document is encoded with the raw Tiktoken encoder taken
       from ``preprocessor.tokenizer`` — no ``[CLS]``/``[SEP]``/``[PAD]``
       wrapping is applied. The BERT-shaped fields on the preprocessor
       (``cls_token_id``, ``sep_token_id``, ``pad_token_id``,
       ``mask_token_id``) are **ignored** in the CLM path; they exist
       on the shared ``TiktokenPreprocessor`` only to keep the
       MLM/classification preprocessors working.
    2. The encoder's ``<|endoftext|>`` token (``encoder.eot_token``) is
       appended after every document so document boundaries are
       signalled inside the token stream.
    3. The resulting stream is sliced into consecutive
       ``max_seq_length``-long windows. Every source token is trained
       on exactly once per epoch — there is no article truncation and
       no window-to-document alignment.
    4. Each window is turned into an ``(input_ids, labels)`` pair via
       the standard shift: ``input_ids = chunk[:-1]``,
       ``labels = chunk[1:]``. The EOT token is a legitimate training
       target, so no label masking is applied.

    :param dataset: A ``tf.data.Dataset`` yielding raw text strings.
    :param preprocessor: :class:`TiktokenPreprocessor` whose underlying
        ``tokenizer`` is a ``tiktoken.Encoding``. Only the encoder and
        its ``eot_token`` are used.
    :param max_seq_length: Window size **including** the +1 token
        needed for the causal shift. After the shift, the model input
        and label tensors both have length ``max_seq_length - 1``.
    :param batch_size: Output batch size.
    :return: Batched ``tf.data.Dataset`` of ``(input_ids, labels)``
        tuples with shape ``(batch, max_seq_length - 1)``.

    .. note::
        DECISION D-004: the legacy ``streaming`` parameter has been
        removed. The packed pipeline never caches and the flag was a
        no-op kept only for signature compatibility. Callers must drop
        ``streaming=...``.
    """
    encoder = preprocessor.tokenizer
    encoding_name = getattr(encoder, "name", None) or "gpt2"
    eot_token_id = int(encoder.eot_token)
    return preprocess_clm_packed_dataset(
        dataset,
        encoding_name=encoding_name,
        chunk_length=max_seq_length,
        batch_size=batch_size,
        eot_token_id=eot_token_id,
    )


def preprocess_clm_packed_dataset(
    dataset: tf.data.Dataset,
    encoding_name: str,
    chunk_length: int,
    batch_size: int,
    eot_token_id: int,
    shuffle_buffer: int = 4096,
    repeat: bool = False,
) -> tf.data.Dataset:
    """Tokenize, pack, and batch a text dataset for CLM pretraining.

    Unlike :func:`preprocess_clm_dataset`, which maps one raw document
    to one fixed-length window (discarding everything past the first
    ``max_seq_length - 2`` tokens), this preprocessor implements the
    standard GPT-style *concat-and-chunk* pipeline:

    1. Each document is encoded with the raw Tiktoken encoder (no
       ``[CLS]``/``[SEP]`` wrapping).
    2. An end-of-text token ``eot_token_id`` is appended after every
       document.
    3. The resulting token stream is split into consecutive
       ``chunk_length``-token windows. Every token is therefore used
       exactly once per epoch.
    4. Each chunk is turned into an ``(input_ids, labels)`` pair via
       the standard shift: ``input_ids = chunk[:-1]``,
       ``labels = chunk[1:]``. The EOT token is a legitimate training
       target, so no label masking is applied — the loss function's
       ``ignore_index`` is only used for true PAD positions (of which
       there are none in a packed pipeline).

    :param dataset: A ``tf.data.Dataset`` yielding raw text strings.
    :param encoding_name: Tiktoken encoding name (e.g. ``"gpt2"``,
        ``"cl100k_base"``). The encoder is constructed once inside the
        generator thread.
    :param chunk_length: Window size including the +1 token needed for
        the causal shift. After the shift, the model input and label
        tensors both have length ``chunk_length - 1``.
    :param batch_size: Output batch size.
    :param eot_token_id: Token ID appended after every document. For
        GPT-2 / ``"gpt2"`` this is ``50256`` (``<|endoftext|>``).
    :param shuffle_buffer: Size of the tf.data shuffle buffer applied
        to the packed chunks.
    :param repeat: If ``True``, apply ``.repeat()`` to the packed
        dataset so a fixed ``steps_per_epoch`` passed to ``model.fit``
        never hits ``StopIteration`` mid-epoch. Callers that do not
        pass ``steps_per_epoch`` to ``fit`` must leave this at
        ``False`` — otherwise training runs forever.
    :return: Batched dataset of ``(input_ids, labels)`` tuples with
        shape ``(batch, chunk_length - 1)``.
    """
    if chunk_length < 2:
        raise ValueError(
            f"chunk_length must be >= 2, got {chunk_length}"
        )

    def packed_generator():
        encoder = tiktoken.get_encoding(encoding_name)
        buf: List[int] = []
        for text in dataset.as_numpy_iterator():
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")
            else:
                text = str(text)
            buf.extend(encoder.encode(text))
            buf.append(eot_token_id)
            while len(buf) >= chunk_length:
                chunk = buf[:chunk_length]
                buf = buf[chunk_length:]
                chunk_arr = np.asarray(chunk, dtype=np.int32)
                yield chunk_arr[:-1], chunk_arr[1:]

    input_len = chunk_length - 1
    packed = tf.data.Dataset.from_generator(
        packed_generator,
        output_signature=(
            tf.TensorSpec(shape=(input_len,), dtype=tf.int32),
            tf.TensorSpec(shape=(input_len,), dtype=tf.int32),
        ),
    )
    packed = (
        packed.shuffle(buffer_size=shuffle_buffer)
        .batch(batch_size, drop_remainder=True)
    )
    if repeat:
        packed = packed.repeat()
    packed = packed.prefetch(tf.data.AUTOTUNE)
    logger.info(
        f"Packed CLM dataset: encoding={encoding_name}, "
        f"chunk_length={chunk_length}, input_len={input_len}, "
        f"batch_size={batch_size}, eot_id={eot_token_id}, repeat={repeat}"
    )
    return packed


def preprocess_classification_dataset(
    dataset: tf.data.Dataset,
    preprocessor: TiktokenPreprocessor,
    max_seq_length: int,
    batch_size: int,
) -> tf.data.Dataset:
    """Tokenize and batch a supervised text dataset for classification.

    Expects a dataset of (text, label) pairs.
    Returns batched dataset of ({input_ids, attention_mask, token_type_ids}, label).
    """
    def tokenize_fn(text, label):
        encoded = preprocessor(decode_text(text), return_tensors='np')
        return encoded['input_ids'][0], encoded['attention_mask'][0], encoded['token_type_ids'][0], label

    dataset = dataset.map(
        lambda text, label: tf.py_function(
            func=tokenize_fn, inp=[text, label],
            Tout=[tf.int32, tf.int32, tf.int32, tf.int64],
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    seq_len = max_seq_length

    def to_dict_and_label(ids, mask, types, label):
        inputs = {
            'input_ids': tf.ensure_shape(ids, [seq_len]),
            'attention_mask': tf.ensure_shape(mask, [seq_len]),
            'token_type_ids': tf.ensure_shape(types, [seq_len]),
        }
        lbl = tf.cast(label, tf.int32)
        lbl.set_shape(())
        return inputs, lbl

    dataset = dataset.map(to_dict_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = (
        dataset.cache().shuffle(1000)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    logger.info(f"Classification dataset preprocessed: batch_size={batch_size}")
    return dataset


# ---------------------------------------------------------------------
# Step Estimation (packed CLM)
# ---------------------------------------------------------------------


# DECISION D-001: single canonical estimator for packed-CLM steps_per_epoch.
# The packed CLM pipeline (preprocess_clm_packed_dataset) emits chunks, not
# articles, so the historical formula ``num_articles // batch_size`` undercounts
# total optimizer steps by ``avg_tokens_per_article / max_seq_length`` and
# misaligns the warmup+cosine LR schedule. Every CLM training script must call
# this helper instead of rolling its own _estimate_steps_per_epoch.
#
# Default avg_tokens_per_article reflects EN Wikipedia 20231101 with
# ``min_article_length=0`` (~3B tokens / ~6.6M articles ≈ 440 tok/article).
# Callers using ``min_article_length=500`` should pass ``avg_tokens_per_article=600``.

# Total tokens in EN Wikipedia 20231101 with min_article_length=0.
# Used as the fallback when neither override nor num_articles is provided.
_DEFAULT_WIKIPEDIA_TOTAL_TOKENS = 2_900_000_000


def estimate_clm_steps_per_epoch(
    num_articles: Optional[int],
    max_seq_length: int,
    batch_size: int,
    override: Optional[int] = None,
    avg_tokens_per_article: int = 440,
) -> int:
    """Estimate ``steps_per_epoch`` for the packed-CLM pipeline.

    The packed-CLM tokenizer (``preprocess_clm_packed_dataset``) emits
    ``chunks`` of length ``max_seq_length``, not articles. The number of
    optimizer steps per epoch is therefore
    ``(num_articles * avg_tokens_per_article) // max_seq_length // batch_size``,
    not ``num_articles // batch_size``. Getting this wrong miscalibrates the
    warmup + cosine LR schedule; the schedule reaches ``alpha=0`` long before
    training does, then training spends a large fraction of its steps at LR=0.

    :param num_articles: Number of source articles (post-filter). ``None`` →
        fall back to the EN-Wikipedia 20231101 token total
        (``_DEFAULT_WIKIPEDIA_TOTAL_TOKENS``).
    :param max_seq_length: Chunk length used by the packed pipeline.
    :param batch_size: Mini-batch size.
    :param override: If provided, return ``max(1, override)`` and ignore the
        article-based estimate (used for ``--steps-per-epoch`` CLI override).
    :param avg_tokens_per_article: Heuristic average tokens per article
        post-tokenization. Default 440 corresponds to EN Wikipedia 20231101
        with ``min_article_length=0``. Pass 600 for ``min_article_length=500``.
    :return: Estimated steps per epoch (>= 1).
    """
    if override is not None:
        return max(1, int(override))
    if num_articles is None:
        chunks = _DEFAULT_WIKIPEDIA_TOTAL_TOKENS // max(1, max_seq_length)
    else:
        chunks = (int(num_articles) * int(avg_tokens_per_article)) // max(1, max_seq_length)
    return max(1, chunks // max(1, batch_size))


# ---------------------------------------------------------------------
# Learning Rate Schedule
# ---------------------------------------------------------------------


def create_warmup_lr_schedule(
    learning_rate: float,
    num_epochs: int,
    steps_per_epoch: int,
    warmup_ratio: float = 0.1,
) -> WarmupSchedule:
    """Create warmup + cosine decay learning rate schedule for NLP training."""
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(warmup_ratio * total_steps)
    primary = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=total_steps - warmup_steps, alpha=0.0,
    )
    return WarmupSchedule(
        warmup_steps=warmup_steps, primary_schedule=primary, warmup_start_lr=1e-7,
    )


# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------


def create_nlp_callbacks(
    model_name: str,
    results_dir_prefix: str,
    monitor: str = 'val_loss',
    patience: int = 15,
    include_analyzer: bool = True,
    analyzer_epoch_frequency: int = 5,
    analyzer_start_epoch: int = 1,
) -> Tuple[List[keras.callbacks.Callback], str]:
    """Create training callbacks with NLP-specific defaults.

    Wraps common create_callbacks() with NLP defaults: TensorBoard enabled,
    LR schedule managed externally (no ReduceLROnPlateau).
    """
    callbacks, results_dir = create_common_callbacks(
        model_name=model_name,
        results_dir_prefix=results_dir_prefix,
        monitor=monitor,
        patience=patience,
        use_lr_schedule=True,
        include_tensorboard=True,
        include_analyzer=include_analyzer,
        analyzer_epoch_frequency=analyzer_epoch_frequency,
        analyzer_start_epoch=analyzer_start_epoch,
    )
    return callbacks, results_dir


# ---------------------------------------------------------------------
# CLM compile-time metric builder
# ---------------------------------------------------------------------

# Per-encoding default chars-per-token approximations. These are display
# constants used only by ``BitsPerCharacter``; the value is the empirical
# mean characters per BPE token observed on standard English corpora
# (Wikipedia / OpenWebText) for the corresponding tiktoken encoding.
# Override via the ``chars_per_token`` argument of ``build_clm_metrics``
# if your dataset materially deviates (e.g. code, non-English).
_CHARS_PER_TOKEN_DEFAULTS = {
    "gpt2": 4.0,
    "r50k_base": 4.0,
    "p50k_base": 4.0,
    "cl100k_base": 4.0,
    "o200k_base": 4.0,
}


def build_clm_metrics(
        encoding_name: str = "gpt2",
        ignore_index: int = -1,
        chars_per_token: Optional[float] = None,
) -> List[keras.metrics.Metric]:
    """Build the canonical CLM evaluation-metric list.

    Centralizes the metric set that every causal-language-modeling
    trainer in ``src/train/`` uses, so each ``model.compile`` site is a
    single-line call::

        model.compile(
            ...,
            metrics={"logits": build_clm_metrics(config.encoding_name)},
        )

    The returned list is fresh on every call (Keras requires unique
    metric instances per ``compile``).

    Args:
        encoding_name: tiktoken encoding name, used only to look up a
            default ``chars_per_token`` constant. Defaults to ``"gpt2"``.
        ignore_index: Class id to mask out from PPL/BPT/BPC accumulation
            (e.g. ``-1`` for ``MaskedCausalLMLoss`` default, ``-100`` for
            HuggingFace-style label padding). Defaults to ``-1``. Pass
            ``None`` to disable masking. ``SparseCategoricalAccuracy``
            does not support ``ignore_class`` and is therefore unmasked
            -- this is the existing behaviour and matches all 6
            in-scope trainers.
        chars_per_token: Override for the ``BitsPerCharacter`` divisor.
            When ``None``, looked up from ``_CHARS_PER_TOKEN_DEFAULTS``
            using ``encoding_name``; falls back to ``4.0``.

    Returns:
        A list ``[SparseCategoricalAccuracy, Perplexity, BitsPerToken,
        BitsPerCharacter]`` ready to drop into
        ``metrics={"logits": [...]}``.
    """
    if chars_per_token is None:
        chars_per_token = _CHARS_PER_TOKEN_DEFAULTS.get(encoding_name, 4.0)

    return [
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        Perplexity(from_logits=True, ignore_class=ignore_index),
        BitsPerToken(from_logits=True, ignore_class=ignore_index),
        BitsPerCharacter(
            chars_per_token=chars_per_token,
            from_logits=True,
            ignore_class=ignore_index,
        ),
    ]


# ---------------------------------------------------------------------
# Dict-keyed compile shim
# ---------------------------------------------------------------------


# DECISION plan_2026-05-07_824e5687/D-001: subclassed Keras 3 models that
# return ``{"logits": ...}`` from ``call()`` have no ``output_names``
# populated by the framework. Calling ``model.compile(metrics={"logits":
# [...]})`` against such a model causes Keras's ``MetricsList`` to silently
# drop every metric (only ``loss`` survives). Setting ``output_names``
# explicitly on the instance before ``model.compile`` is the minimal,
# library-untouching fix; it is a no-op when Keras already populated the
# attribute (forwards-compat). Re-applied inside every trainer's
# ``compile_model`` so the helper runs on both fresh and resumed
# (``keras.models.load_model``) instances. Hard-codes the single-key case;
# revisit if a CLM model ever needs metrics on multiple output heads.
def prepare_dict_keyed_compile(
    model: keras.Model,
    output_keys: Optional[List[str]] = None,
    output_key: str = "logits",
) -> None:
    """Ensure a dict-output subclassed model has ``output_names`` set.

    Subclassed ``keras.Model`` instances whose ``call`` returns a dict do
    not get ``output_names`` populated by Keras. As a result,
    ``model.compile(metrics={"logits": [...]}, loss={"logits": fn})``
    silently drops the metric list (the loss path uses a different code
    path and works correctly). Setting
    ``model.output_names = [...keys]`` before ``compile`` aligns
    Keras's metric / loss / loss_weights flattening logic with the
    dict-keyed user spec.

    Idempotent: if ``output_names`` is already a non-empty list, this is
    a no-op (forwards-compat with future Keras releases that may
    populate it automatically).

    Args:
        model: The subclassed model whose ``call`` returns a dict.
        output_keys: Ordered list of dict keys produced by ``model.call``
            that the trainer will key losses / metrics / loss_weights
            against. When ``None`` (default), falls back to the single
            key ``[output_key]`` — preserving the legacy single-head
            CLM trainer behaviour. Use this for matryoshka / multi-head
            trainers that emit multiple loss-bearing logits keys (e.g.
            ``["logits", "logits_w64", "logits_w32"]``).
        output_key: Legacy single-key shortcut. Used only when
            ``output_keys`` is ``None``. Defaults to ``"logits"``, which
            matches every single-head CLM trainer in ``src/train/`` today.
    """
    keys: List[str] = list(output_keys) if output_keys else [output_key]
    existing = getattr(model, "output_names", None)
    if not existing:
        model.output_names = keys


__all__ = [
    "create_tokenizer",
    "decode_text",
    "load_text_dataset",
    "preprocess_mlm_dataset",
    "preprocess_clm_dataset",
    "preprocess_clm_packed_dataset",
    "preprocess_classification_dataset",
    "estimate_clm_steps_per_epoch",
    "create_warmup_lr_schedule",
    "create_nlp_callbacks",
    "build_clm_metrics",
    "prepare_dict_keyed_compile",
    "augment_probe_results",
    "GenerationProbeCallback",
]
