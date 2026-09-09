"""Byte-level packed-CLM data primitives.

``train.common.nlp`` is tiktoken-only: every causal-LM pipeline in the tree
goes through ``preprocess_clm_packed_dataset``, which builds a
``tiktoken.Encoding`` inside its generator thread. A byte-level model
(``vocab_size = 256``) has no tokenizer object at all, so the packing
transform has to be rebuilt on raw UTF-8 bytes. This module is that rebuild
and nothing more — it holds the four transforms

* :func:`text_to_byte_ids` — ``str`` → ``uint8`` UTF-8 code units,
* :func:`byte_ids_to_text` — the inverse, tolerant of a window boundary
  landing mid-codepoint,
* :func:`pack_byte_windows` — the concat-and-chunk packer,
* :func:`build_byte_clm_dataset` — the two above layered onto a
  ``tf.data.Dataset`` of raw text strings,

plus :func:`estimate_byte_clm_steps_per_epoch`, the byte-unit sibling of
``train.common.nlp.estimate_clm_steps_per_epoch``.

The upstream text source is reused as-is:
:func:`dl_techniques.datasets.nlp.load_wikipedia_train_val` already returns a
memory-mapped ``tf.data.Dataset`` of raw UTF-8 strings, which is exactly the
input this module consumes. Bytes are derived downstream of text in the same
place tokens are, so no new loader is needed.

.. note::
   The corpus location is a **default constant plus a caller-supplied
   argument** (``--dataset-root`` at the CLI), never an environment variable.
   :data:`DEFAULT_DATASET_ROOT` is imported from
   :mod:`dl_techniques.datasets.nlp` rather than retyped, so the two modules
   cannot drift apart.
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, Optional, Sequence, Union

import numpy as np
import tensorflow as tf

from dl_techniques.datasets.nlp import DEFAULT_WIKIPEDIA_CACHE_DIR
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

#: Byte-level vocabulary: one id per possible UTF-8 code unit.
BYTE_VOCAB_SIZE = 256

#: Beginning-of-document id. **A convention, not a mechanism.** 254 and 255
#: are simply the two highest byte values; nothing in the model treats them
#: specially, they are ordinary members of the 256-way vocabulary and are
#: legitimate prediction targets. A document containing the raw byte 0xFE is
#: therefore indistinguishable from one carrying this marker -- that is
#: accepted, exactly as ``<|endoftext|>`` is accepted in the token pipeline.
BOS_ID = 254

#: End-of-document id. Same convention as :data:`BOS_ID`.
EOS_ID = 255

#: Default corpus root. Imported, not retyped -- see the module note.
DEFAULT_DATASET_ROOT = DEFAULT_WIKIPEDIA_CACHE_DIR

#: Default tf.data shuffle buffer over packed windows, matching
#: ``train.common.nlp.preprocess_clm_packed_dataset``'s 4096.
DEFAULT_SHUFFLE_BUFFER = 4096

#: Total UTF-8 text bytes in the already-staged EN Wikipedia 20231101 dump at
#: ``min_article_length=0``. MEASURED (2026-09-09) over all 41 Arrow shards of
#: ``/media/arxwn/data0_4tb/datasets/wikipedia`` by summing
#: ``pyarrow.compute.binary_length`` over the ``text`` column:
#: 6,407,814 articles / 19,567,594,259 bytes. Not an estimate, not a
#: transcription -- the derivation is recorded in the plan's ``decisions.md``.
DEFAULT_WIKIPEDIA_TOTAL_BYTES = 19_567_594_259

#: Average UTF-8 bytes per Wikipedia article, ``19_567_594_259 / 6_407_814``
#: = 3053.71, rounded to the nearest integer. This is the byte-unit analogue
#: of ``train.common.nlp``'s ``avg_tokens_per_article=440``; the two are NOT
#: interchangeable and their ratio (~6.9 bytes/token) is not the ~4.0
#: bytes/token an English cl100k stream actually shows, because the 440
#: constant is itself a rounded heuristic. Do not derive one from the other.
DEFAULT_AVG_BYTES_PER_ARTICLE = 3054


# ---------------------------------------------------------------------
# Byte encoding / decoding
# ---------------------------------------------------------------------


def text_to_byte_ids(
    text: Union[str, bytes],
    add_bos: bool = False,
    add_eos: bool = False,
) -> np.ndarray:
    """Encode text to UTF-8 byte ids.

    The encoding is plain UTF-8, so a non-ASCII character contributes 2-4
    ids. This is deliberately **not** ``ord()`` / latin-1: those agree with
    UTF-8 on ASCII and disagree on everything else, which is precisely the
    shape of defect a corpus of mostly-ASCII text hides.

    :param text: A Python ``str``, or ``bytes`` already holding UTF-8 (the
        form ``tf.data``'s ``as_numpy_iterator`` yields for a string
        tensor). ``bytes`` are passed through unchanged, not re-encoded.
    :param add_bos: Prepend :data:`BOS_ID`.
    :param add_eos: Append :data:`EOS_ID`.
    :return: 1-D ``uint8`` array of length ``len(utf8) + add_bos + add_eos``.
        An empty input yields an empty array, never a raise.
    """
    if isinstance(text, bytes):
        raw = text
    elif isinstance(text, (bytearray, memoryview)):
        raw = bytes(text)
    else:
        raw = str(text).encode("utf-8")

    ids = np.frombuffer(raw, dtype=np.uint8)
    if not (add_bos or add_eos):
        # frombuffer aliases a read-only buffer; copy so callers may mutate.
        return ids.copy()

    parts: List[np.ndarray] = []
    if add_bos:
        parts.append(np.asarray([BOS_ID], dtype=np.uint8))
    parts.append(ids)
    if add_eos:
        parts.append(np.asarray([EOS_ID], dtype=np.uint8))
    return np.concatenate(parts).astype(np.uint8, copy=False)


def byte_ids_to_text(
    ids: Sequence[int],
    errors: str = "replace",
) -> str:
    """Decode byte ids back to text, tolerating a truncated codepoint.

    A packed window boundary lands wherever the byte count says it does, so
    the first or last codepoint of a window is routinely cut in half. The
    reference implementation's ``generate.py`` handles the same hazard
    defensively when streaming model output. This helper therefore defaults
    to ``errors="replace"`` and **never raises** on a mid-codepoint edge;
    pass ``errors="strict"`` when a caller genuinely wants the raise.

    :param ids: Iterable of byte ids in ``[0, 255]``.
    :param errors: Python codec error policy.
    :return: The decoded string.
    """
    return bytes(bytearray(int(i) for i in ids)).decode("utf-8", errors=errors)


# ---------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------


def pack_byte_windows(
    stream: Iterable[Sequence[int]],
    seq_len: int,
    drop_remainder: bool = True,
) -> Iterator[np.ndarray]:
    """Concat-and-chunk a stream of per-document byte ids into windows.

    This is the byte-unit rebuild of the packing loop in
    ``train.common.nlp.preprocess_clm_packed_dataset``: documents are
    concatenated into one buffer and sliced into consecutive
    ``seq_len``-long windows, so every source byte is trained on exactly
    once per epoch. There is no per-document truncation and no
    window-to-document alignment -- a window may span a document boundary,
    and may split a multi-byte codepoint.

    Document separators are **not** inserted here. The caller appends
    :data:`EOS_ID` (see :func:`build_byte_clm_dataset`), which keeps this
    function a pure reshaping transform.

    :param stream: Iterable of per-document byte-id sequences.
    :param seq_len: Window length in bytes. Must be >= 1.
    :param drop_remainder: When ``True`` (the token pipeline's behaviour) a
        trailing buffer shorter than ``seq_len`` is discarded. When
        ``False`` it is yielded as a final SHORT window, so the caller must
        cope with a ragged length.
    :return: Iterator of 1-D ``uint8`` arrays, each of length ``seq_len``
        except possibly the last when ``drop_remainder=False``.
    :raises ValueError: If ``seq_len < 1``.
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")

    buf: List[int] = []
    for document in stream:
        buf.extend(int(b) for b in np.asarray(document, dtype=np.uint8).tolist())
        while len(buf) >= seq_len:
            window = buf[:seq_len]
            buf = buf[seq_len:]
            yield np.asarray(window, dtype=np.uint8)

    if buf and not drop_remainder:
        yield np.asarray(buf, dtype=np.uint8)


def build_byte_clm_dataset(
    text_ds: tf.data.Dataset,
    seq_len: int,
    batch_size: int,
    eos_id: int = EOS_ID,
    shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER,
    repeat: bool = False,
) -> tf.data.Dataset:
    """Build a batched byte-level packed-CLM dataset from raw text.

    Layers :func:`text_to_byte_ids` and :func:`pack_byte_windows` onto the
    raw UTF-8 strings that
    :func:`dl_techniques.datasets.nlp.load_wikipedia_train_val` already
    returns:

    1. Each document is encoded to UTF-8 bytes.
    2. ``eos_id`` is appended after every document, so document boundaries
       are signalled inside the byte stream (the byte-level analogue of the
       token pipeline's ``<|endoftext|>``).
    3. The stream is sliced into consecutive ``seq_len``-byte windows; the
       trailing partial window of the epoch is dropped.
    4. Each window becomes an ``(input_ids, labels)`` pair via the standard
       causal shift ``input_ids = window[:-1]``, ``labels = window[1:]``.
       ``eos_id`` is a legitimate target, so no label masking is applied.

    :param text_ds: ``tf.data.Dataset`` yielding raw text (string) tensors.
    :param seq_len: Window length **including** the +1 byte the causal shift
        consumes. The emitted tensors therefore have length ``seq_len - 1``
        -- the same contract as ``chunk_length`` in
        ``train.common.nlp.preprocess_clm_packed_dataset``, deliberately
        mirrored so a reader of one pipeline is not surprised by the other.
    :param batch_size: Output batch size. Batches use
        ``drop_remainder=True``, so every batch is full.
    :param eos_id: Document-separator id, :data:`EOS_ID` by default.
    :param shuffle_buffer: tf.data shuffle buffer over packed windows. Pass
        ``1`` to preserve corpus order (used by the tests).
    :param repeat: Apply ``.repeat()`` so a fixed ``steps_per_epoch`` never
        hits ``StopIteration`` mid-epoch. Callers that do not pass
        ``steps_per_epoch`` to ``fit`` must leave this ``False``.
    :return: ``tf.data.Dataset`` of ``(input_ids, labels)`` ``int32`` tensors
        of shape ``(batch_size, seq_len - 1)``.
    :raises ValueError: If ``seq_len < 2`` (a window of 1 byte leaves nothing
        after the shift).
    """
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2, got {seq_len}")

    def packed_generator():
        def byte_stream():
            for text in text_ds.as_numpy_iterator():
                ids = text_to_byte_ids(text)
                yield np.concatenate(
                    [ids, np.asarray([eos_id], dtype=np.uint8)]
                )

        for window in pack_byte_windows(
            byte_stream(), seq_len=seq_len, drop_remainder=True
        ):
            chunk = window.astype(np.int32)
            yield chunk[:-1], chunk[1:]

    input_len = seq_len - 1
    packed = tf.data.Dataset.from_generator(
        packed_generator,
        output_signature=(
            tf.TensorSpec(shape=(input_len,), dtype=tf.int32),
            tf.TensorSpec(shape=(input_len,), dtype=tf.int32),
        ),
    )
    packed = packed.shuffle(buffer_size=max(1, shuffle_buffer)).batch(
        batch_size, drop_remainder=True
    )
    if repeat:
        packed = packed.repeat()
    packed = packed.prefetch(tf.data.AUTOTUNE)
    logger.info(
        f"Packed byte-level CLM dataset: seq_len={seq_len}, "
        f"input_len={input_len}, batch_size={batch_size}, "
        f"eos_id={eos_id}, repeat={repeat}"
    )
    return packed


# ---------------------------------------------------------------------
# Step estimation (packed byte CLM)
# ---------------------------------------------------------------------


def estimate_byte_clm_steps_per_epoch(
    num_articles: Optional[int],
    seq_len: int,
    batch_size: int,
    override: Optional[int] = None,
    avg_bytes_per_article: int = DEFAULT_AVG_BYTES_PER_ARTICLE,
) -> int:
    """Estimate ``steps_per_epoch`` for the packed byte-CLM pipeline.

    The byte-unit sibling of
    ``train.common.nlp.estimate_clm_steps_per_epoch``, with the same
    contract, the same parameter order and the same three behaviours:
    ``override`` short-circuits everything, ``num_articles=None`` falls back
    to the whole-corpus total, and the result is clamped to ``>= 1``. Only
    the UNIT differs -- bytes instead of tiktoken tokens -- which is why the
    fallback constant is :data:`DEFAULT_WIKIPEDIA_TOTAL_BYTES` and the
    per-article heuristic is :data:`DEFAULT_AVG_BYTES_PER_ARTICLE`.

    A separate function rather than a parameter on the token estimator is
    forced by the dependency direction: that estimator lives in
    ``train.common.nlp``, which imports ``tiktoken``, TFDS and several model
    packages, and ``dl_techniques`` must not import from ``train``. The
    arithmetic is four lines and is mirrored here verbatim rather than
    re-derived.

    :param num_articles: Number of source articles (post-filter). ``None``
        falls back to :data:`DEFAULT_WIKIPEDIA_TOTAL_BYTES`.
    :param seq_len: Window length used by :func:`build_byte_clm_dataset`.
    :param batch_size: Mini-batch size.
    :param override: If provided, return ``max(1, override)`` and ignore the
        article-based estimate (the ``--steps-per-epoch`` CLI override).
    :param avg_bytes_per_article: Average UTF-8 bytes per source article.
    :return: Estimated steps per epoch (>= 1).
    """
    if override is not None:
        return max(1, int(override))
    if num_articles is None:
        windows = DEFAULT_WIKIPEDIA_TOTAL_BYTES // max(1, seq_len)
    else:
        windows = (int(num_articles) * int(avg_bytes_per_article)) // max(1, seq_len)
    return max(1, windows // max(1, batch_size))
