"""Data for the language tasks: documents, a document-keyed split, windows and a stream that is a pure function of (seed, step).

The legacy language trainers tokenize a corpus inside a generator in dataset order, pack it, and shuffle with an unseeded
buffer: a stream is not a function of its index, a resumed ``fit`` replays document order, and the one place a resume is
handled (``seed + initial_step``) re-splits the corpus and leaks selection documents into train. This module holds the
corpus in memory as int32 arrays and builds three pieces of numpy, so that ``val_loss`` is a function of the weights alone
and a resumed run continues the stream it was in:

* the DOCUMENTS (:func:`synthetic_documents`, :func:`imdb_documents`), tokenized once (:func:`tokenize_documents`);
* the SPLIT of the documents by ``sha256(seed:select:index)`` (:func:`split_documents`), the selection windows and the test
  windows (:func:`lm_fixed_windows`), fixed for ever;
* the TRAIN STREAM (:class:`TextStream`): epoch ``e`` is a permutation of the documents keyed on ``(seed, e)``, concatenated
  with an end-of-text id after each, cut into windows of ``seq_len + 1`` and permuted again; ``batch_at(step)`` is a function of
  ``(seed, step)``, so a resume at ``step`` yields the batch the uninterrupted run yields at ``step``.

The tensors are built last, inside the two dataset builders (:func:`select_dataset`, :func:`train_dataset`). The module scope
imports numpy only: TensorFlow, tiktoken and tensorflow_datasets are imported inside the function that needs one. (``import
train.common.text_data`` still loads TensorFlow, because the package ``__init__`` of ``train.common`` does: what a check made
before the GPU is claimed needs lives in ``train.api.text_tokenizer``, which this module imports from.)

Interface contract
------------------
``open_tokenizer(spec, directory=None)``
    :param spec: a :class:`train.api.text_tokenizer.TokenizerSpec`; ``directory``: where the BPE files are (``None``: the pinned
        directory); ignored for ``ascii95``.
    :return: a :class:`Tokenizer`: ``encode(text)`` the ordinary ids of ``text`` (the literal ``<|endoftext|>`` in a review is
        text, never an end-of-text id) and ``decode(ids)`` the text of ids in ``[0, spec.vocab_size)`` (the four ids above tiktoken's
        vocabulary, which tiktoken's own ``decode`` raises ``KeyError`` on, print as ``<|cls|>``, ``<|sep|>``, ``<|pad|>``,
        ``<|mask|>``). The files are checked against the pinned hashes BEFORE tiktoken's loader runs, ``TIKTOKEN_CACHE_DIR`` is set
        for the call and given back.
    :raises ConfigError: a missing directory or a missing or different file (:func:`train.api.text_tokenizer.check_files`).
    :raises ValueError: an encoding whose ``n_vocab`` or end-of-text id is not the spec's.

``tokenize_documents(texts, tokenizer)``
    :return: one int32 array per text. :raises ValueError: a token id outside ``[0, spec.vocab_size)`` or equal to the
        end-of-text id, naming the document.

``synthetic_documents(seed, n, kind, split="train")``
    :param kind: :data:`LM` (a first-order word chain a unigram cannot match) or :data:`CLASSIFICATION` (class words and neutral
        words, exactly ``round(0.6 n)`` documents of class 0).
    :return: ``(texts, labels)``, ``labels`` an int64 array for a classification bank and ``None`` for a language one. Each text is
        a function of ``(seed, split, index)`` through a ``RandomState`` (a frozen stream), so a bank is the same in any interpreter.

``imdb_documents(split, data_dir=None, max_documents=None, *, loader=None)``
    :return: ``(texts, labels)`` of the TFDS ``imdb_reviews`` split ``train`` or ``test`` in READ ORDER (the document index is
        the read order, which two fresh processes measured equal), the first ``max_documents`` of them when given. ``loader`` is the
        seam: ``loader(split, data_dir, limit)`` yields ``(text, label)``; the default reads TFDS offline (``download=False``,
        ``shuffle_files=False``).

``corpus_documents(dataset, kind, *, max_samples)``
    :return: ``((train_texts, train_labels), (test_texts, test_labels))`` of the corpus ``dataset`` (``synthetic`` or ``imdb_reviews``,
        ``train.api.text_args.DATASETS``): the first ``max_samples`` documents of each split in read order for ``imdb_reviews``; for the
        synthetic bank ``max_samples`` train documents and a quarter of that many test documents (``None``: the sizes
        ``train.api.text_args`` names). The bank is keyed on :data:`BANK_SEED`, never on the run's seed, so a corpus is the same
        documents for every run and the seed only splits and orders them.

``split_documents(n, seed, validation_split)``
    :return: ``(train_index, select_index)``, int64, ascending: document ``i`` is selected when ``sha256(f"{seed}:select:{i}")``,
        read as a unit float, is below ``validation_split``. A function of ``(seed, i)`` only: stable when documents are appended.

``lm_epoch_windows(docs, order, seq_len, eot)`` / ``lm_window_count(docs, seq_len)`` / ``lm_fixed_windows(docs, seq_len, eot)``
    :return: the ``(n, seq_len + 1)`` int32 windows of the documents taken in ``order`` with ``eot`` after each, the tail shorter
        than a window dropped; the count (the same for every order); the ``(x, y)`` int32 pair of the documents in index order,
        ``y`` the inputs shifted by one.

``TextStream``
    ``batch_at(step)`` is a pure function of ``(seed, step)``: ``(x, y)`` int32 ``(batch_size, seq_len)``, fresh arrays;
    ``iterate(start_step=None)`` yields ``batch_at`` from ``start_step`` (default ``self.start_step``) for ever; ``with_start(step)``
    is the same stream starting at ``step``; ``steps_per_epoch = n_rows // batch_size`` is the same for every epoch (the number of
    windows does not depend on the order). :raises ValueError: a corpus of fewer than ``batch_size`` rows, a bad step.

``lm_stream(docs, *, seq_len, batch_size, seed, eot)`` / ``lm_split(train_docs, test_docs, *, spec, seq_len, batch_size, seed,
stream_seed=None, validation_split)``
    :return: the train :class:`TextStream`; a :class:`TextSplit` (the stream over the train documents, the fixed ``(x, y)``
        windows of the selection and of the test documents, the document indices of both parts). ``seed`` keys the split, the
        stream is keyed on ``stream_seed`` (``None`` means ``seed``) and the split never reads it.
    :raises ValueError: a selection or a test part that holds fewer tokens than one window; fewer train windows than a batch.

``encode_rows(docs, spec, seq_len)``
    :return: ``(ids, truncated)``: int32 ``(n, seq_len)`` rows ``[CLS]`` + the head of the document + pad, and the number of
        documents cut. A row is never all pad (the ``[CLS]`` is its first id). :raises ValueError: ``seq_len`` below 2.

``TextBatches`` / ``lm_batches(split, *, spec, seq_len, batch_size, seed, validation_split)``
    The object the language task's ``build_data`` returns and the run sequence reads: the :class:`TextSplit`, the three datasets
    built from it (``select_dataset`` of the selection and of the test windows, ``train_dataset`` of the stream), the counts in
    windows (``n_train``, ``n_select``, ``n_test``), ``steps_per_epoch``, ``val_steps``, ``input_shape`` ``(seq_len,)``, no classes
    and the plain-language provenance of the two fixed splits. ``with_start(step)`` is the same data whose train stream begins at
    ``step`` (the fixed datasets are the same objects).

``select_dataset((x, y), batch_size)`` / ``train_dataset(stream)``
    TensorFlow datasets. The first is finite, in row order, its last batch ragged, its cardinality stated and checked. The second
    is infinite (``Dataset.counter`` mapped through ``stream.batch_at``), begins at ``stream.start_step`` and holds no state but the
    step it is at.
"""

import dataclasses
import hashlib
import itertools
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from dl_techniques.utils.logger import logger
from train.api.text_args import IMDB, SYNTHETIC, SYNTHETIC_TEST_DOCUMENTS, SYNTHETIC_TRAIN_DOCUMENTS
from train.api.text_tokenizer import ASCII95, TOKENIZERS, TokenizerSpec, check_files

#: The environment variable tiktoken reads its cache directory from.
CACHE_ENV = "TIKTOKEN_CACHE_DIR"

#: The tag of the split's hash: ``sha256(f"{seed}:select:{index}")``.
SELECT_TAG = "select"

#: The two kinds of synthetic bank and the two splits of a corpus.
LM, CLASSIFICATION = "language_model", "classification"
KINDS: Tuple[str, ...] = (LM, CLASSIFICATION)
SPLITS: Tuple[str, ...] = ("train", "test")

# DECISION plan-2026-11-24T092348-7806a290/D-026
# The synthetic bank is keyed on this constant, NOT on the run's seed: a corpus is the same documents for every run (as the IMDB
# reviews are), and the seed splits and orders them. A bank that followed the seed would make two seeds two corpora, so "the
# documents a source run trained on" would name other texts under the same index. Do NOT pass `config.seed` to `synthetic_documents`.
BANK_SEED = 0

#: The words of the synthetic language bank: sixteen, in a ring. A word is followed by the next one (0.80), the one three on
#: (0.15) or any (0.05), so a bigram model reaches 0.72 nats a word where a unigram model cannot go below ln 16 = 2.77.
LM_WORDS: Tuple[str, ...] = (
    "the", "cat", "sat", "on", "a", "mat", "and", "dog", "ran", "to", "red", "big", "tree", "by", "it", "is")
LM_STEP_PROBABILITIES = (0.80, 0.95)    # cumulative: next word, the word three on, otherwise any word

#: The words of the synthetic classification bank. Class 1 draws its cue words from the first list, class 0 from the second
#: (the IMDB convention: 0 negative, 1 positive); six words in ten are cue words, the rest neutral.
POSITIVE_WORDS: Tuple[str, ...] = ("good", "great", "fine", "nice", "best", "love")
NEGATIVE_WORDS: Tuple[str, ...] = ("bad", "poor", "awful", "worst", "hate", "dull")
NEUTRAL_WORDS: Tuple[str, ...] = ("the", "a", "movie", "was", "film", "it", "and", "this")
CUE_SHARE = 0.6
#: The share of class 0 in a classification bank: a fixed 60/40 split, so a constant predictor scores 0.6, above chance.
CLASS_ZERO_SHARE = 0.6


# ---------------------------------------------------------------------
# the tokenizer
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Tokenizer:
    """An opened tokenizer: the ids of ordinary text and the text of ids; see the module contract."""

    spec: TokenizerSpec
    encoding: Any = field(default=None, repr=False)     # tiktoken's Encoding; None for the character tokenizer

    def encode(self, text: str) -> List[int]:
        """The ordinary ids of ``text`` (no end-of-text, cls or sep id is ever produced)."""
        if self.encoding is None:
            from dl_techniques.layers.tokenizers.ascii_char import encode_ascii

            return encode_ascii(text)
        return self.encoding.encode_ordinary(text)

    def decode(self, ids: Sequence[int]) -> str:
        """The text of ``ids``; the specials above tiktoken's vocabulary print by name."""
        spec = self.spec
        ids = [int(i) for i in ids]
        for token in ids:
            if not 0 <= token < spec.vocab_size:
                raise ValueError(f"token id {token} is outside [0, {spec.vocab_size}), the vocabulary of {spec.encoding}")
        if self.encoding is None:
            from dl_techniques.layers.tokenizers.ascii_char import decode_ascii

            return decode_ascii(ids, skip_special_tokens=False)
        names = {spec.cls_id: "<|cls|>", spec.sep_id: "<|sep|>", spec.pad_id: "<|pad|>", spec.mask_id: "<|mask|>"}
        parts: List[str] = []
        for named, run in itertools.groupby(ids, key=lambda token: token in names):
            if named:
                parts.extend(names[token] for token in run)
            else:
                parts.append(self.encoding.decode(list(run)))
        return "".join(parts)


def open_tokenizer(spec: TokenizerSpec, directory: Any = None) -> Tokenizer:
    """Open the tokenizer of ``spec``; see the module contract."""
    if spec.encoding not in TOKENIZERS:
        raise ValueError(f"tokenizer {spec.encoding!r} is not offered; the text tasks take {list(TOKENIZERS)}")
    if spec.encoding == ASCII95:
        return Tokenizer(spec)
    # DECISION plan-2026-11-24T092348-7806a290/D-015
    # The files are checked against the pinned hashes BEFORE `tiktoken.get_encoding` runs. Do NOT call get_encoding first and
    # "let tiktoken verify": its loader REMOVES a cache file whose sha256 differs and then tries the network, and a read-only
    # file does not stop that (only the directory's write bit does). Checked here, a copy of another content is refused by
    # name and stays where it is.
    root = check_files(spec.files, directory)
    import tiktoken

    # DECISION plan-2026-11-24T092348-7806a290/D-019
    # The variable is set for the call and given back, also when the call raises. Do NOT leave it set "so the next call finds the
    # files": the legacy trainers call get_encoding too, and a variable left set would point them at this directory without their
    # asking (hiding the loss of their own /tmp copy). tiktoken reads the variable inside the call, and caches the Encoding.
    previous = os.environ.get(CACHE_ENV)
    os.environ[CACHE_ENV] = str(root)
    try:
        encoding = tiktoken.get_encoding(spec.encoding)
    finally:
        if previous is None:
            os.environ.pop(CACHE_ENV, None)
        else:
            os.environ[CACHE_ENV] = previous
    if encoding.n_vocab != spec.n_vocab:
        raise ValueError(f"tiktoken's {spec.encoding} has n_vocab {encoding.n_vocab}, the record says {spec.n_vocab}")
    if encoding.eot_token != spec.eot_id:
        raise ValueError(f"tiktoken's {spec.encoding} has eot id {encoding.eot_token}, the record says {spec.eot_id}")
    return Tokenizer(spec, encoding)


def tokenize_documents(texts: Sequence[str], tokenizer: Tokenizer) -> List[np.ndarray]:
    """One int32 array of ordinary ids per text; see the module contract."""
    spec = tokenizer.spec
    documents: List[np.ndarray] = []
    for index, text in enumerate(texts):
        ids = np.asarray(tokenizer.encode(text), dtype=np.int64)
        if ids.size:
            low, high = int(ids.min()), int(ids.max())
            if low < 0 or high >= spec.vocab_size:
                raise ValueError(
                    f"document {index} has token id {low if low < 0 else high}, outside [0, {spec.vocab_size}), the vocabulary "
                    f"of {spec.encoding}: the tokenizer and the record disagree")
            if (ids == spec.eot_id).any():
                raise ValueError(
                    f"document {index} contains the end-of-text id {spec.eot_id}: the stream puts that id after every "
                    "document itself, so a document must not hold one")
        documents.append(ids.astype(np.int32))
    return documents


# ---------------------------------------------------------------------
# the documents
# ---------------------------------------------------------------------


def _bank_rng(seed: int, split: str, tag: Any) -> np.random.RandomState:
    """A ``RandomState`` (a frozen stream) seeded from a sha256 of ``(seed, split, tag)``: the bank is the same in any interpreter."""
    digest = hashlib.sha256(f"{seed}:{split}:bank:{tag}".encode()).digest()
    return np.random.RandomState(int.from_bytes(digest[:4], "big"))


def _language_document(rs: np.random.RandomState) -> str:
    words = len(LM_WORDS)
    count = rs.randint(12, 40)
    current = rs.randint(words)
    chosen = [LM_WORDS[current]]
    draws, anywhere = rs.random_sample(count - 1), rs.randint(words, size=count - 1)
    for draw, other in zip(draws, anywhere):
        if draw < LM_STEP_PROBABILITIES[0]:
            current = (current + 1) % words
        elif draw < LM_STEP_PROBABILITIES[1]:
            current = (current + 3) % words
        else:
            current = int(other)
        chosen.append(LM_WORDS[current])
    return " ".join(chosen) + "."


def _classification_document(rs: np.random.RandomState, label: int) -> str:
    cues = POSITIVE_WORDS if label == 1 else NEGATIVE_WORDS
    count = rs.randint(8, 20)
    draws, picks = rs.random_sample(count), rs.randint(0, 1 << 30, size=count)
    chosen = [cues[pick % len(cues)] if draw < CUE_SHARE else NEUTRAL_WORDS[pick % len(NEUTRAL_WORDS)]
              for draw, pick in zip(draws, picks)]
    return " ".join(chosen) + "."


def synthetic_documents(seed: int, n: int, kind: str, split: str = "train") -> Tuple[List[str], Optional[np.ndarray]]:
    """``n`` seeded documents of the ``kind`` bank; see the module contract."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {list(KINDS)}, got {kind!r}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {list(SPLITS)}, got {split!r}")
    if kind == LM:
        return [_language_document(_bank_rng(seed, split, index)) for index in range(n)], None
    zeros = round(CLASS_ZERO_SHARE * n)
    labels = np.repeat(np.array([0, 1], np.int64), [zeros, n - zeros])
    _bank_rng(seed, split, "labels").shuffle(labels)
    return [_classification_document(_bank_rng(seed, split, index), int(label)) for index, label in enumerate(labels)], labels


def _tfds_imdb(split: str, data_dir: Optional[str], limit: Optional[int]) -> Iterator[Tuple[Any, Any]]:
    """Read ``imdb_reviews`` from the local TFDS directory, offline, in file order."""
    import tensorflow_datasets as tfds

    # the slice syntax reads only the records it names; a `.take` after the load reads the file through its cache and warns
    window = split if limit is None else f"{split}[:{limit}]"
    ds = tfds.load("imdb_reviews", split=window, data_dir=data_dir, download=False, shuffle_files=False, as_supervised=True)
    yield from tfds.as_numpy(ds)


def imdb_documents(
        split: str, data_dir: Optional[str] = None, max_documents: Optional[int] = None, *,
        loader: Optional[Callable[[str, Optional[str], Optional[int]], Iterator[Tuple[Any, Any]]]] = None,
) -> Tuple[List[str], np.ndarray]:
    """The ``imdb_reviews`` documents of ``split`` in read order, the first ``max_documents`` of them; see the module contract."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {list(SPLITS)} (the unsupervised reviews have no use here), got {split!r}")
    if max_documents is not None and max_documents < 1:
        raise ValueError(f"max_documents must be at least 1, got {max_documents}")
    rows = (_tfds_imdb if loader is None else loader)(split, data_dir, max_documents)
    texts: List[str] = []
    labels: List[int] = []
    for text, label in itertools.islice(rows, max_documents):
        texts.append(text.decode("utf-8", errors="replace") if isinstance(text, bytes) else str(text))
        labels.append(int(label))
    return texts, np.asarray(labels, np.int64)


def corpus_documents(
        dataset: str, kind: str, *, max_samples: Optional[int],
) -> Tuple[Tuple[List[str], Optional[np.ndarray]], Tuple[List[str], Optional[np.ndarray]]]:
    """The train and test documents of ``dataset``: ``((texts, labels), (texts, labels))``; see the module contract."""
    if dataset == SYNTHETIC:
        n_train = SYNTHETIC_TRAIN_DOCUMENTS if max_samples is None else max_samples
        n_test = SYNTHETIC_TEST_DOCUMENTS if max_samples is None else max(1, max_samples // 4)
        return synthetic_documents(BANK_SEED, n_train, kind, "train"), synthetic_documents(BANK_SEED, n_test, kind, "test")
    if dataset == IMDB:
        return imdb_documents("train", None, max_samples), imdb_documents("test", None, max_samples)
    raise ValueError(f"dataset must be {SYNTHETIC!r} or {IMDB!r}, got {dataset!r}")


# ---------------------------------------------------------------------
# the split and the windows
# ---------------------------------------------------------------------


def _hash64(text: str) -> int:
    """The first 8 bytes of ``sha256(text)`` as an unsigned integer: the one keyed hash of the split and of the stream's orders."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def split_documents(n: int, seed: int, validation_split: float) -> Tuple[np.ndarray, np.ndarray]:
    """``(train_index, select_index)`` of ``n`` documents; see the module contract."""
    if not 0 < validation_split < 1:
        raise ValueError(f"validation_split must be strictly between 0 and 1, got {validation_split!r}")
    unit = np.fromiter((_hash64(f"{seed}:{SELECT_TAG}:{i}") / 2 ** 64 for i in range(n)), dtype=np.float64, count=n)
    chosen = unit < validation_split
    return np.flatnonzero(~chosen), np.flatnonzero(chosen)


def lm_window_count(docs: Sequence[np.ndarray], seq_len: int) -> int:
    """The windows of ``docs`` with an eot after each: the same for every order of the documents."""
    return (sum(len(doc) for doc in docs) + len(docs)) // (seq_len + 1)


def lm_epoch_windows(docs: Sequence[np.ndarray], order: Sequence[int], seq_len: int, eot: int) -> np.ndarray:
    """The ``(n, seq_len + 1)`` int32 windows of the documents taken in ``order``; see the module contract."""
    if seq_len < 1:
        raise ValueError(f"seq_len must be at least 1, got {seq_len}")
    end = np.full(1, eot, np.int32)
    pieces: List[np.ndarray] = []
    for index in order:
        pieces.append(docs[index])
        pieces.append(end)
    flat = np.concatenate(pieces).astype(np.int32, copy=False) if pieces else np.zeros(0, np.int32)
    count = len(flat) // (seq_len + 1)
    return flat[:count * (seq_len + 1)].reshape(count, seq_len + 1)


def lm_fixed_windows(docs: Sequence[np.ndarray], seq_len: int, eot: int) -> Tuple[np.ndarray, np.ndarray]:
    """The ``(x, y)`` windows of the documents in index order: ``y`` is ``x`` shifted by one; see the module contract."""
    windows = lm_epoch_windows(docs, np.arange(len(docs)), seq_len, eot)
    return np.ascontiguousarray(windows[:, :-1]), np.ascontiguousarray(windows[:, 1:])


# DECISION plan-2026-11-24T092348-7806a290/D-017
# The permutations are an argsort of sha256 keys, NOT numpy's `Generator.permutation`: the stream has to be the same function
# of (seed, epoch) in the interpreter that resumes a run as in the one that started it, and numpy does not promise the stream of
# a `Generator` across releases. A test holds the rule against a plain-python oracle. Do NOT swap it for `default_rng`.
def _hash_order(seed: int, epoch: int, tag: str, n: int) -> np.ndarray:
    keys = np.fromiter((_hash64(f"{seed}:{epoch}:{tag}:{i}") for i in range(n)), dtype=np.uint64, count=n)
    return np.argsort(keys, kind="stable")


class _EpochCache:
    """The last two epochs' rows, built under a lock: a stream is read by the input thread and by a resume's first batch."""

    KEEP = 2

    def __init__(self) -> None:
        self._rows: "OrderedDict[int, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, epoch: int, build: Callable[[int], Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if epoch in self._rows:
                self._rows.move_to_end(epoch)
            else:
                self._rows[epoch] = build(epoch)
                while len(self._rows) > self.KEEP:
                    self._rows.popitem(last=False)
            return self._rows[epoch]


@dataclass(frozen=True)
class TextStream:
    """The train stream: batch ``step`` is a pure function of ``(seed, step)``; see the module contract.

    :param epoch_rows: ``epoch -> (x_rows, y_rows)``, the ``n_rows`` rows of that epoch in the order they are consumed; a pure
        function of the epoch (and of what the stream was built from).
    :param n_rows: rows in an epoch (the same for every epoch); ``batch_size``: rows in a batch; ``start_step``: where
        :meth:`iterate` and a dataset begin.
    """

    epoch_rows: Callable[[int], Tuple[np.ndarray, np.ndarray]] = field(repr=False)
    n_rows: int
    batch_size: int
    start_step: int = 0
    _cache: _EpochCache = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")
        if self.start_step < 0:
            raise ValueError(f"start_step must be non-negative, got {self.start_step}")
        if self.n_rows < self.batch_size:
            raise ValueError(
                f"the train stream holds {self.n_rows} rows per epoch, fewer than one batch of {self.batch_size}: lower the batch "
                "size, raise data.max_samples, or lower the validation share")
        object.__setattr__(self, "_cache", _EpochCache())

    @property
    def steps_per_epoch(self) -> int:
        return self.n_rows // self.batch_size

    def batch_at(self, step: int) -> Tuple[np.ndarray, np.ndarray]:
        """The ``(x, y)`` batch of train step ``step``: fresh arrays, a function of the stream's seed and ``step`` alone."""
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 0:
            raise ValueError(f"step must be a non-negative int, got {step!r}")
        epoch, position = divmod(int(step), self.steps_per_epoch)
        x, y = self._cache.get(epoch, self.epoch_rows)
        rows = slice(position * self.batch_size, (position + 1) * self.batch_size)
        return x[rows].copy(), y[rows].copy()

    def iterate(self, start_step: Optional[int] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """The batches of steps ``start_step``, ``start_step + 1``, ... for ever (``None``: from ``self.start_step``)."""
        step = self.start_step if start_step is None else start_step
        while True:
            yield self.batch_at(step)
            step += 1

    def with_start(self, step: int) -> "TextStream":
        """The same stream beginning at ``step``."""
        # DECISION plan-2026-11-24T092348-7806a290/D-003
        # A resume CONTINUES the stream it was in: the epoch orders are keyed on (seed, epoch) and a restart only moves where
        # reading begins. Do NOT derive a new stream seed on resume (the pretrain rule for masks keyed on a counter): the first
        # process consumed these documents in THIS order, so another order would make "the resumed run is the uninterrupted run"
        # false and untestable, and it would visit some documents twice in an epoch and others not at all.
        return dataclasses.replace(self, start_step=step)


def lm_stream(docs: Sequence[np.ndarray], *, seq_len: int, batch_size: int, seed: int, eot: int) -> TextStream:
    """The train stream of language-model windows over ``docs``; see the module contract."""
    if seq_len < 1:
        raise ValueError(f"seq_len must be at least 1, got {seq_len}")
    docs = list(docs)
    n_rows = lm_window_count(docs, seq_len)

    def epoch_rows(epoch: int) -> Tuple[np.ndarray, np.ndarray]:
        windows = lm_epoch_windows(docs, _hash_order(seed, epoch, "docs", len(docs)), seq_len, eot)
        if len(windows) != n_rows:
            raise RuntimeError(f"epoch {epoch} has {len(windows)} windows, not the {n_rows} the stream counted")
        windows = windows[_hash_order(seed, epoch, "windows", n_rows)]
        return windows[:, :-1], windows[:, 1:]

    return TextStream(epoch_rows, n_rows, batch_size)


@dataclass(frozen=True, eq=False)
class TextSplit:
    """The numpy half of what a text task consumes.

    :param stream: the train stream over the train documents. :param select, test: the fixed ``(x, y)`` windows of the selection
        documents and of the test documents. :param train_index, select_index: the document numbers of the train and of the
        selection parts, in the corpus given. :param n_test_docs: documents of the test part.
    """

    stream: TextStream
    select: Tuple[np.ndarray, np.ndarray] = field(repr=False)
    test: Tuple[np.ndarray, np.ndarray] = field(repr=False)
    train_index: np.ndarray = field(repr=False)
    select_index: np.ndarray = field(repr=False)
    n_test_docs: int


def lm_split(
        train_docs: Sequence[np.ndarray], test_docs: Sequence[np.ndarray], *, spec: TokenizerSpec, seq_len: int, batch_size: int,
        seed: int, stream_seed: Optional[int] = None, validation_split: float,
) -> TextSplit:
    """Split ``train_docs`` into train and selection documents and build the three language-model parts; see the module contract."""
    train_docs, test_docs = list(train_docs), list(test_docs)
    # DECISION plan-2026-11-24T092348-7806a290/D-003
    # The split is keyed on the SPLIT seed and the document index and on nothing else: `stream_seed` moves the train order
    # only. Do NOT hand the stream seed (or an epoch, or a step) to `split_documents`: a resumed or re-seeded run would then
    # re-split, train on documents it selects on, and score the same weights on other documents.
    train_index, select_index = split_documents(len(train_docs), seed, validation_split)
    select = lm_fixed_windows([train_docs[i] for i in select_index], seq_len, spec.eot_id)
    test = lm_fixed_windows(test_docs, seq_len, spec.eot_id)
    for name, rows, count in (("selection", select[0], len(select_index)), ("test", test[0], len(test_docs))):
        if len(rows) == 0:
            raise ValueError(
                f"the {name} documents ({count} of them) hold fewer tokens than one window of {seq_len + 1}: raise "
                "data.max_samples (or data.validation_split for the selection), or lower the sequence length")
    stream = lm_stream([train_docs[i] for i in train_index], seq_len=seq_len, batch_size=batch_size,
                       seed=seed if stream_seed is None else stream_seed, eot=spec.eot_id)
    logger.info(
        f"language-model split: {len(train_index)} train, {len(select_index)} selection, {len(test_docs)} test documents; "
        f"{stream.n_rows} train, {len(select[0])} selection, {len(test[0])} test windows of {seq_len + 1} tokens")
    return TextSplit(stream, select, test, train_index, select_index, len(test_docs))


# ---------------------------------------------------------------------
# the rows of a classifier
# ---------------------------------------------------------------------


def encode_rows(docs: Sequence[np.ndarray], spec: TokenizerSpec, seq_len: int) -> Tuple[np.ndarray, int]:
    """``(ids, truncated)``: ``[CLS]`` + the first ``seq_len - 1`` ids of each document, right-padded; see the module contract."""
    if seq_len < 2:
        raise ValueError(f"seq_len must be at least 2 (a row is [CLS] and at least one token), got {seq_len}")
    ids = np.full((len(docs), seq_len), spec.pad_id, np.int32)
    ids[:, 0] = spec.cls_id
    truncated = 0
    for row, doc in enumerate(docs):
        take = min(len(doc), seq_len - 1)
        ids[row, 1:1 + take] = doc[:take]
        truncated += len(doc) > seq_len - 1
    return ids, truncated


# ---------------------------------------------------------------------
# the datasets
# ---------------------------------------------------------------------


def select_dataset(rows: Tuple[np.ndarray, np.ndarray], batch_size: int) -> "Any":
    """A finite dataset of the fixed ``(x, y)`` rows in order, the last batch ragged, its cardinality stated and checked."""
    import tensorflow as tf

    x, y = rows
    if len(x) == 0:
        raise ValueError("the dataset has no rows")
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
    expected = -(-len(x) // batch_size)
    if int(dataset.cardinality()) != expected:
        # Keras logs n / (n + 1) of the loss over a dataset of unknown length (D-265 of plan 81938e81)
        raise ValueError(f"the selection dataset states cardinality {int(dataset.cardinality())}, one pass is {expected} batches")
    return dataset.prefetch(tf.data.AUTOTUNE)


def train_dataset(stream: TextStream) -> "Any":
    """The infinite dataset of ``stream``'s batches from ``stream.start_step``: a function of the step and nothing else."""
    import tensorflow as tf

    # DECISION plan-2026-11-24T092348-7806a290/D-018
    # A counter mapped through `batch_at`, NOT `Dataset.from_generator`: a generator is an iterator with a position (a second
    # iterator over it starts again from its own beginning, and a resumed fit would have to be told where to start), and every
    # `from_generator` site must be classified in `test_a_dataset_reaching_fit_states_its_length`. A counter is stateless, infinite
    # by construction, and the dataset of a resumed stream is the dataset of the same counter from another start.
    first_x, first_y = stream.batch_at(stream.start_step)
    types = (tf.as_dtype(first_x.dtype), tf.as_dtype(first_y.dtype))

    def fetch(step: Any) -> Tuple[Any, Any]:
        x, y = tf.numpy_function(stream.batch_at, [step], types)
        x.set_shape(first_x.shape)
        y.set_shape(first_y.shape)
        return x, y

    dataset = tf.data.Dataset.counter(stream.start_step).map(fetch).prefetch(tf.data.AUTOTUNE)
    if int(dataset.cardinality()) != int(tf.data.INFINITE_CARDINALITY):
        raise ValueError(f"the train dataset states cardinality {int(dataset.cardinality())}, not infinite")
    return dataset


# ---------------------------------------------------------------------
# what a language task consumes
# ---------------------------------------------------------------------


# DECISION plan-2026-11-24T092348-7806a290/D-025
# The object the run sequence reads (`build_data`'s return value) is a frozen dataclass of the datasets and the counts, beside the
# numpy `TextSplit` it was built from, as `PretrainBatches` and `ForecastBatches` are: the sequence needs `train_ds`, `select_ds`,
# `test_ds`, `steps_per_epoch`, the three counts, `input_shape` and the provenance strings by those names, and a resume replaces
# only the train dataset (`with_start`). Do NOT fold the datasets into `TextSplit` (it is numpy only, and the stream tests need no
# TensorFlow) and do NOT hand the sequence a bare namespace (`dataclasses.replace` is how a test watches the stream).
@dataclass(frozen=True, eq=False)
class TextBatches:
    """What the language task consumes; the fields the run sequence reads; see the module contract."""

    split: TextSplit
    train_ds: Any = field(repr=False)
    select_ds: Any = field(repr=False)
    test_ds: Any = field(repr=False)
    steps_per_epoch: int
    val_steps: int
    n_train: int
    n_select: int
    n_test: int
    input_shape: Tuple[int]
    num_classes: Optional[int]
    selection_split: str
    test_split: str
    spec: TokenizerSpec
    seed: int
    batch_size: int

    def with_start(self, step: int) -> "TextBatches":
        """The same data whose train stream begins at ``step``: the fixed datasets are the same objects."""
        stream = self.split.stream.with_start(step)
        return dataclasses.replace(self, split=dataclasses.replace(self.split, stream=stream), train_ds=train_dataset(stream))


def lm_batches(
        split: TextSplit, *, spec: TokenizerSpec, seq_len: int, batch_size: int, seed: int, validation_split: float,
) -> TextBatches:
    """The three datasets of a :class:`TextSplit` and their counts; see the module contract."""
    n_select, n_test = len(split.select[0]), len(split.test[0])
    return TextBatches(
        split=split,
        train_ds=train_dataset(split.stream),
        select_ds=select_dataset(split.select, batch_size),
        test_ds=select_dataset(split.test, batch_size),
        steps_per_epoch=split.stream.steps_per_epoch,
        val_steps=-(-n_select // batch_size),
        n_train=split.stream.n_rows,
        n_select=n_select,
        n_test=n_test,
        input_shape=(seq_len,),
        num_classes=None,
        selection_split=(
            f"the documents whose sha256 of '{seed}:{SELECT_TAG}:<index>' is below validation_split={validation_split} of the train "
            "documents, cut into windows"),
        test_split="the test documents of the corpus, cut into windows",
        spec=spec,
        seed=seed,
        batch_size=batch_size,
    )
