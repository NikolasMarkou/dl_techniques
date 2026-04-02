"""Shared NLP utilities for training scripts.

Provides tokenizer creation, text data loading/preprocessing, warmup LR schedules,
and NLP-specific callback wrappers. Used by BERT, FNet, and other NLP pretrain/finetune
scripts that share the Tiktoken + TFDS pipeline.
"""

import keras
import tensorflow as tf
import tensorflow_datasets as tfds
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
) -> tf.data.Dataset:
    """Tokenize and batch a text dataset for CLM (causal language modeling) training.

    Expects a dataset of raw text strings (not supervised).
    Returns batched dataset of (input_ids, labels) tuples where labels are
    input_ids shifted right by one position. The last token of input has no
    corresponding label (truncated).

    :param dataset: A tf.data.Dataset yielding raw text strings.
    :param preprocessor: TiktokenPreprocessor for tokenization.
    :param max_seq_length: Maximum sequence length for tokenization.
    :param batch_size: Batch size for the output dataset.
    :return: Batched tf.data.Dataset of ``(input_ids, labels)`` tuples.
        ``input_ids`` shape: ``(batch, max_seq_length - 1)``,
        ``labels`` shape: ``(batch, max_seq_length - 1)``.
    """
    def tokenize_fn(text):
        encoded = preprocessor(decode_text(text), return_tensors='np')
        ids = encoded['input_ids'][0]
        return ids

    dataset = dataset.map(
        lambda x: tf.py_function(tokenize_fn, [x], tf.int32),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    seq_len = max_seq_length

    def make_clm_pair(ids):
        ids = tf.ensure_shape(ids, [seq_len])
        # Input: tokens [0..n-2], Labels: tokens [1..n-1]
        input_ids = ids[:-1]
        labels = ids[1:]
        return input_ids, labels

    dataset = dataset.map(make_clm_pair, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = (
        dataset.cache()
        .shuffle(buffer_size=1000)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    logger.info(f"CLM dataset preprocessed: batch_size={batch_size}, seq_len={seq_len - 1}")
    return dataset


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
