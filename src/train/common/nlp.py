"""Shared NLP utilities for training scripts.

Provides tokenizer creation, text data loading/preprocessing, warmup LR schedules,
and NLP-specific callback wrappers. Used by BERT, FNet, and other NLP pretrain/finetune
scripts that share the Tiktoken + TFDS pipeline.
"""

import os
import pickle

import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tiktoken
from typing import Any, Callable, List, Optional, Tuple

from train.common import create_callbacks as create_common_callbacks
from train.common.generation_probe import GenerationProbeCallback

from dl_techniques.analyzer import AnalysisConfig, DataInput, ModelAnalyzer
from dl_techniques.utils.logger import logger
from dl_techniques.utils.tokenizer import TiktokenPreprocessor
# `create_warmup_lr_schedule` now lives in dl_techniques.optimization.schedule
# alongside the other LR-schedule construction. It is re-exported here so the
# ~11 existing `from train.common.nlp import create_warmup_lr_schedule` call
# sites keep working unchanged.
from dl_techniques.optimization.schedule import create_warmup_lr_schedule  # noqa: F401
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
    # Do NOT call dataset.cache() here. Caching the tokenized
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


# Single canonical estimator for packed-CLM steps_per_epoch.
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


# Subclassed Keras 3 models that
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


# ---------------------------------------------------------------------
# Sentiment fine-tuning: post-training analysis
# ---------------------------------------------------------------------


def sentiment_final_model_filename(model_name: str) -> str:
    """Filename the sentiment fine-tuning scripts save their final model under.

    Args:
        model_name: Lower-case script/model name, ``"bert"`` or ``"fnet"``.

    Returns:
        e.g. ``"bert_sentiment_final_best.keras"``.

    Used at BOTH ends of the same contract -- the save site inside each
    script's ``finetune_sentiment_model``, and the read site inside
    :func:`run_finetune_post_training_analysis` -- so the two cannot drift into
    a "saved here, looked for there" mismatch. That failure mode is not
    hypothetical in this file pair: see the ``best_path`` note below.
    """
    return f"{model_name}_sentiment_final_best.keras"


def prepare_data_for_analyzer(val_dataset: tf.data.Dataset, num_samples: int) -> DataInput:
    """Extract samples from validation dataset for ModelAnalyzer.

    Args:
        val_dataset: A BATCHED ``(x_dict, y)`` classification dataset; it is
            unbatched here before taking ``num_samples`` examples.
        num_samples: How many individual examples to take.

    Returns:
        A ``DataInput`` whose ``x_data`` is a dict of stacked numpy arrays
        keyed exactly as the dataset's feature dict.

    Failure mode: an empty ``val_dataset`` raises ``IndexError`` on
    ``x_batches[0]`` -- deliberately loud, since analysing zero samples is
    never the intent.
    """
    logger.info(f"Preparing {num_samples} samples for analysis...")
    val_subset = val_dataset.unbatch().take(num_samples)
    x_batches, y_list = [], []
    for x, y in val_subset:
        x_batches.append(x)
        y_list.append(y.numpy())
    x_data = {key: np.array([d[key].numpy() for d in x_batches]) for key in x_batches[0].keys()}
    return DataInput(x_data=x_data, y_data=np.array(y_list))


# DECISION plan-2026-08-12T123743-e798a9e1/D-017
# `best_path` below points at a file NOTHING IN THIS REPOSITORY EVER WRITES.
# This is a KNOWN LIVE DEFECT (F-23), moved here verbatim from
# bert/finetune.py:241 and fnet/finetune.py:241 so that the consolidation
# commit stays behaviour-preserving -- it is NOT a statement that the path is
# correct. Measured: `best_sentiment_model.keras` has three READ sites and zero
# WRITE sites in all of `src/`; `ModelCheckpoint` writes
# `<results_dir>/best_model.keras` (train/common/callbacks.py:97) -- a
# different DIRECTORY *and* a different FILENAME. Because
# `run_post_training_analysis` defaults to True, both `train.bert.finetune` and
# `train.fnet.finetune` raise `ValueError: File not found` here at the end of
# every default run, AFTER training has completed and the final model has been
# saved (reproduced 2026-08-12, evidence/step2-f19/).
# DO NOT "fix" this by changing `config.save_dir` to the run directory: the
# FILENAME is wrong too, so that alone still misses. The real checkpoint is
# `<results_dir>/best_model.keras`, and `results_dir` is returned by
# `create_nlp_callbacks`, which `post_training_analysis` is not currently given.
# Tracked as plan step 15b; see decisions.md D-017.
def run_finetune_post_training_analysis(
    config: Any,
    model_name: str,
    create_initial_model: Callable[[], keras.Model],
) -> None:
    """Run comprehensive post-training analysis comparing model snapshots.

    One shared implementation of the ~58-line block that
    ``src/train/bert/finetune.py`` and ``src/train/fnet/finetune.py`` carried
    in near-identical copies.

    Args:
        config: The script's ``FinetuneConfig``. Must carry
            ``full_analysis_dir``, ``save_dir``, ``dataset_name``,
            ``max_samples``, ``max_seq_length``, ``batch_size``,
            ``encoding_name``, the four special-token ids, and
            ``analysis_n_samples``.
        model_name: ``"bert"`` or ``"fnet"`` -- selects the final-model
            filename via :func:`sentiment_final_model_filename`.
        create_initial_model: Zero-argument factory returning a FRESH
            (untrained-head) sentiment model, i.e. each script's own
            ``create_sentiment_model``. Called here rather than passed
            pre-built so construction still happens after the analysis
            directory exists, exactly as it did in the per-script copies.

    Failure mode: raises whatever ``keras.models.load_model`` raises when a
    snapshot is missing -- see the D-017 note above, which is exactly that.

    ``custom_objects=`` is deliberately NOT passed: ``BERT``, ``FNet`` and
    ``TreeTransformer`` are all ``@keras.saving.register_keras_serializable()``
    so the registry already resolves them. The two scripts disagreed about this
    (bert omitted it, fnet passed it); verified by execution that both models
    round-trip bit-identically without it.
    """
    logger.info("Running Post-Training Analysis")
    os.makedirs(config.full_analysis_dir, exist_ok=True)

    initial_model = create_initial_model()
    best_path = os.path.join(config.save_dir, "best_sentiment_model.keras")
    final_path = os.path.join(
        config.save_dir, sentiment_final_model_filename(model_name)
    )

    models_to_analyze = {
        "Initial_Model": initial_model,
        "Best_Model(ValAcc)": keras.models.load_model(best_path),
        "Final_Model": keras.models.load_model(final_path),
    }

    history_path = os.path.join(config.save_dir, "training_history.pkl")
    with open(history_path, 'rb') as f:
        history_dict = pickle.load(f)
    training_histories = {name: history_dict for name in models_to_analyze}

    preprocessor = create_tokenizer(
        config.encoding_name, config.max_seq_length,
        config.cls_token_id, config.sep_token_id,
        config.pad_token_id, config.mask_token_id,
    )
    val_dataset = preprocess_classification_dataset(
        load_text_dataset(config.dataset_name, "test", config.max_samples, as_supervised=True),
        preprocessor, config.max_seq_length, config.batch_size,
    )
    analysis_data = prepare_data_for_analyzer(val_dataset, config.analysis_n_samples)

    analyzer = ModelAnalyzer(
        models=models_to_analyze,
        training_history=training_histories,
        config=AnalysisConfig(
            analyze_weights=True, analyze_spectral=True,
            analyze_calibration=True, analyze_training_dynamics=True,
            analyze_information_flow=False, verbose=True,
        ),
        output_dir=config.full_analysis_dir,
    )
    analyzer.analyze(data=analysis_data)
    logger.info(f"Analysis complete. Results: {config.full_analysis_dir}")


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
    "sentiment_final_model_filename",
    "prepare_data_for_analyzer",
    "run_finetune_post_training_analysis",
    "augment_probe_results",
    "GenerationProbeCallback",
]
