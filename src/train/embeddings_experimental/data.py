"""Packed, padding-free ASCII datasets for the embeddings study.

Why packing rather than the usual per-document padded batch
-----------------------------------------------------------
One arm of this study mixes tokens with depthwise convolutions rather than
attention, and that block has ``supports_masking = False``: zero padding enters
the convolutional receptive field of real positions, and with the global branch
enabled the pad length shifts every position. The transformer arm honours its
mask; the Clifford arm cannot. A padded-batch comparison would therefore measure
the padding policy as much as the block.

The fix is to remove padding rather than to fake a mask. Stage 1 concatenates
documents into one character stream and cuts it into consecutive windows of
exactly ``seq_len`` characters, so every position of every row is real and the
attention mask is all ones for both arms. This is the standard concat-and-chunk
recipe ``train.common.nlp.preprocess_clm_packed_dataset`` implements for causal
LM; that function is tiktoken-specific and emits shifted CLM pairs, so the ASCII
MLM variant lives here rather than being bolted onto it.

Stage 2 packs too. ``run_contrastive_stage`` calls :func:`build_packed_mlm_dataset`
for its own stream, so SimCSE's two dropout views are of a packed fixed-length
window rather than of a whole sentence. That is a deliberate consequence of the
Clifford arm's ``supports_masking = False`` -- the same reason stage 1 packs --
but it has a consequence worth stating: **no model trained by this package has
ever seen a sequence of any length other than ``max_seq_length``**. Padding, and
the length-bucketing that bounds its effect, return only at EVALUATION
(``evaluate_embeddings.embed_texts``), which is the first time these encoders
meet a short input at all.

An earlier version of this docstring said "Stage 2 cannot pack -- contrastive
learning needs whole sentences". That described an intention, not the code.
"""

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.tokenizers.ascii_char import (
    CLS_ID,
    SEP_ID,
    ASCIICharPreprocessor,
    encode_ascii,
)
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

__all__ = [
    "build_packed_mlm_dataset",
    "packed_character_windows",
]


def packed_character_windows(
    text_iterator: Callable[[], Iterator[Any]],
    seq_len: int,
    lowercase: bool = True,
    normalize_unicode: bool = True,
) -> Iterator[np.ndarray]:
    """Yield consecutive ``seq_len``-character windows over a text stream.

    Documents are encoded to ASCII ids and separated by a single ``[SEP]``, then
    the stream is cut into windows. A window is emitted only when it is
    completely full, so **every row is exactly ``seq_len`` real characters and no
    padding is ever produced** -- which is the property the Clifford arm needs.
    The trailing partial window of the stream is discarded.

    :param text_iterator: Zero-argument callable returning an iterator of
        strings (or utf-8 ``bytes``, or 0-d string tensors).
    :type text_iterator: Callable
    :param seq_len: Window length in characters.
    :type seq_len: int
    :param lowercase: Whether to case-fold. Defaults to ``True``.
    :type lowercase: bool
    :param normalize_unicode: Whether to NFKD-fold accents. Defaults to ``True``.
    :type normalize_unicode: bool
    :return: Iterator of ``(seq_len,)`` int32 arrays.
    :rtype: Iterator[np.ndarray]
    """
    buffer: List[int] = []
    for raw in text_iterator():
        text = raw
        if hasattr(text, "numpy"):
            text = text.numpy()
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")
        if not text:
            continue

        buffer.extend(
            encode_ascii(
                text, lowercase=lowercase, normalize_unicode=normalize_unicode
            )
        )
        buffer.append(SEP_ID)

        while len(buffer) >= seq_len:
            window = buffer[:seq_len]
            del buffer[:seq_len]
            yield np.asarray(window, dtype=np.int32)


def build_packed_mlm_dataset(
    text_dataset: tf.data.Dataset,
    seq_len: int,
    batch_size: int,
    training: bool,
    shuffle_buffer: int = 4096,
    repeat: bool = False,
    lowercase: bool = True,
    normalize_unicode: bool = True,
) -> tf.data.Dataset:
    """Build a padding-free MLM dataset from a stream of raw texts.

    The masking itself is NOT done here: it is dynamic and lives in
    :class:`~dl_techniques.models.language.masked_language_model.mlm.MaskedLanguageModel`,
    which corrupts ``input_ids`` inside its own ``train_step`` so a token is
    masked differently on each epoch.

    :param text_dataset: Dataset yielding raw text scalars.
    :type text_dataset: tf.data.Dataset
    :param seq_len: Packed window length in characters.
    :type seq_len: int
    :param batch_size: Output batch size.
    :type batch_size: int
    :param training: Whether to shuffle.
    :type training: bool
    :param shuffle_buffer: Window shuffle buffer. Windows are consecutive slices
        of one document stream, so without this a batch would be a single
        passage rather than a sample of the corpus.
    :type shuffle_buffer: int
    :param repeat: Whether to repeat indefinitely, for a fixed step budget.
    :type repeat: bool
    :param lowercase: Whether to case-fold.
    :type lowercase: bool
    :param normalize_unicode: Whether to NFKD-fold accents.
    :type normalize_unicode: bool
    :return: Dataset of ``{"input_ids", "attention_mask"}`` int32 batches, both
        of shape ``(batch_size, seq_len)``; the mask is all ones by construction.
    :rtype: tf.data.Dataset
    """

    def generator():
        return packed_character_windows(
            lambda: iter(text_dataset.as_numpy_iterator()),
            seq_len=seq_len,
            lowercase=lowercase,
            normalize_unicode=normalize_unicode,
        )

    windows = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
    )

    if training:
        windows = windows.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    if repeat:
        windows = windows.repeat()

    def to_inputs(ids: tf.Tensor) -> Dict[str, tf.Tensor]:
        return {
            "input_ids": ids,
            # Every position is real: the stream is packed, never padded.
            "attention_mask": tf.ones_like(ids, dtype=tf.int32),
        }

    return (
        windows.map(to_inputs, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
