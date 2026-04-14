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

from dl_techniques.utils.logger import logger
from dl_techniques.utils.tokenizer import TiktokenPreprocessor
from dl_techniques.optimization.warmup_schedule import WarmupSchedule


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
    dataset = (
        dataset.cache()
        .shuffle(buffer_size=1000)
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
    streaming: bool = False,
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
    :param streaming: Retained for signature compatibility. The packed
        generator never caches, so this flag has no effect — it is
        accepted to avoid breaking callers that pass
        ``streaming=True``. A future release may drop the parameter.
    :return: Batched ``tf.data.Dataset`` of ``(input_ids, labels)``
        tuples with shape ``(batch, max_seq_length - 1)``.
    """
    del streaming  # packed pipeline never caches, parameter kept for API stability.

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
        .prefetch(tf.data.AUTOTUNE)
    )
    logger.info(
        f"Packed CLM dataset: encoding={encoding_name}, "
        f"chunk_length={chunk_length}, input_len={input_len}, "
        f"batch_size={batch_size}, eot_id={eot_token_id}"
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
