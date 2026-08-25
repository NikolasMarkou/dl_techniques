"""
Host-side text preparation for ColBERT's two asymmetric input paths.

ColBERT encodes queries and documents with **one** set of weights. Nothing in
the network distinguishes the two; the entire asymmetry lives here, in how the
token stream is assembled before it ever reaches the encoder. There are exactly
three asymmetries, and every one of them is a documented mechanism from the
paper rather than an implementation convenience.

**1. Marker tokens.** A query is ``[CLS] [Q] <content> [SEP]``; a document is
``[CLS] [D] <content> [SEP]``. The marker sits at position 1, immediately after
``[CLS]``, and is the only signal the shared encoder receives about which side
of the retrieval problem it is looking at. Without it the same sentence encodes
identically as a query and as a document, and the two projections collapse onto
each other.

**2. Query augmentation with ``[MASK]``.** A query is padded to
``query_maxlen`` and then *every* padding slot is overwritten with ``[MASK]``.
This is not padding hygiene -- it is the paper's query-expansion mechanism. Each
``[MASK]`` position is a real position in the transformer: it attends to the
query's actual terms and produces a contextualized embedding of its own, which
then participates in MaxSim exactly like a real query token. The model is free
to learn to place, at those slots, embeddings that match terms the user did not
type -- a learned, differentiable re-weighting and expansion of the query. Short
queries therefore get *more* expansion capacity than long ones, which is the
intended behaviour. Documents are never augmented this way: a document is padded
with ``[PAD]`` and the padding is masked out. Asserting both directions is the
only way to pin this, because a tokenizer that augments both sides looks
perfectly healthy from the query side alone.

**3. Punctuation filtering on documents only.** ColBERT drops punctuation tokens
from the *document* representation, on the grounds that a punctuation mark
carries no retrievable content but can still win a MaxSim max against a query
term and pollute the score. Queries pass an empty skiplist and are never
filtered. This module reports the filter as a separate ``skiplist_mask`` output
rather than folding it into ``attention_mask``, so that each returned array
means exactly one thing: ``attention_mask`` is "this position exists",
``skiplist_mask`` is "this position carries content". The consumer forms the
reference's participation mask as their product::

    participation = attention_mask * skiplist_mask

Named deviations from the reference
-----------------------------------

**No WordPiece.** The reference allocates ``[Q]`` and ``[D]`` to BERT
WordPiece's reserved ``[unused0]`` / ``[unused1]`` slots. This repository has no
WordPiece tokenizer anywhere; its NLP stack is Tiktoken ``cl100k_base``
(``n_vocab == 100277``), whose special tokens are allocated downward from the
top of the vocabulary by :func:`dl_techniques.utils.tokenizer.get_special_token_ids`
(``cls = n_vocab - 20`` through ``unk = n_vocab - 17``). The markers here
therefore take the next two free slots by that same convention,
``n_vocab - 16`` and ``n_vocab - 15``, derived from the live vocabulary size
rather than hardcoded. Id-for-id parity with published ColBERT checkpoints is
permanently forfeited by this choice -- and was already unreachable for an
independent reason, since this repository's BERT runs ``gelu_tanh`` rather than
exact GELU. Both facts are stated rather than implied away. The markers are
control ids that no ``encode`` call can produce and that ``decode`` will drop;
nothing here depends on decoding them.

**Space-prefixed punctuation.** The reference builds its skiplist as
``{encode(symbol)[0] for symbol in string.punctuation}``. That is sufficient for
WordPiece, where ``"cat, sat"`` yields ``","`` as its own token. It is *not*
sufficient for a byte-pair encoding: ``cl100k_base`` tokenizes the same text
with a leading-space token ``" ,"`` (id 1174), which is a different id from bare
``","`` (id 11), so a literal port of the reference rule would build a skiplist
that matches almost nothing in ordinary prose. This module therefore also adds
the first id of ``encode(" " + symbol)`` for each symbol. Measured on
``cl100k_base``: all 32 space-prefixed forms are single tokens, and the bare and
space-prefixed sets are disjoint (32 + 32 = 64 ids). This is a deliberate,
recorded widening of the reference rule to preserve its *intent* under a
different tokenizer, not a transcription of it.

Attention over augmented ``[MASK]`` slots
-----------------------------------------

The reference exposes ``attend_to_mask_tokens`` (``QuerySettings``, default
``False``), which controls whether the transformer's attention mask is 1 at the
augmented ``[MASK]`` positions. It is worth being precise about what that flag
does and does not do in the reference, because the two masks involved are
easily conflated:

* The reference's *participation* mask for queries -- the one multiplied onto
  the projected embeddings before L2-normalization -- is computed as
  ``x != pad_token_id`` **after** the pad ids have already been overwritten with
  ``[MASK]``. It is therefore **all ones**, at every ``query_maxlen`` position,
  regardless of the flag. Augmented ``[MASK]`` embeddings always participate in
  MaxSim. That is the whole point of the mechanism.
* The reference's ``attend_to_mask_tokens`` flag governs only the *transformer*
  attention mask, i.e. whether the ``[MASK]`` positions are visible as keys to
  the other positions.

This class emits a single ``attention_mask`` that a consumer will naturally use
for both purposes. Defaulting it to 0 at the augmented slots would therefore
silently annihilate query augmentation the moment the mask is used as the
projection's participation mask -- the MaxSim contribution of every ``[MASK]``
slot would be exactly zero, the suite would stay green, and the paper's headline
mechanism would be dead. Given the choice between reproducing the reference's
transformer-side default and preserving the reference's participation
semantics, this module preserves the semantics:

    ``attend_to_mask_tokens`` defaults to **True** here. Augmented ``[MASK]``
    positions get ``attention_mask == 1``.

**Consequence.** Relative to the reference's default, the ``[MASK]`` slots are
additionally visible as attention *keys* to the real query terms. Set the flag
to ``False`` to reproduce the reference's transformer-side masking exactly, at
the cost of those positions being excluded from any consumer that derives its
participation mask from ``attention_mask``. Both settings are pinned by test.

References:
    - Khattab and Zaharia, 2020. ColBERT: Efficient and Effective Passage
      Search via Contextualized Late Interaction over BERT.
      (https://arxiv.org/abs/2004.12832)
    - Santhanam et al., 2022. ColBERTv2: Effective and Efficient Retrieval via
      Lightweight Late Interaction. (https://arxiv.org/abs/2112.01488)
    - Reference query tokenizer (pad-then-overwrite-with-MASK):
      https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/modeling/tokenization/query_tokenization.py
    - Reference document tokenizer:
      https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/modeling/tokenization/doc_tokenization.py
    - Reference skiplist construction and its documents-only application:
      https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/modeling/colbert.py
    - Reference defaults (``query_maxlen=32``, ``doc_maxlen=220``,
      ``mask_punctuation=True``, ``attend_to_mask_tokens=False``):
      https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/infra/config/settings.py
"""

import string
import tiktoken
import numpy as np
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.tokenizer import (
    TiktokenPreprocessor,
    get_special_token_ids,
)

# ---------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------

#: Minimum viable query length: ``[CLS] [Q] <one content token> [SEP]``.
MIN_QUERY_MAXLEN: int = 4

#: Minimum viable document length: ``[CLS] [D] [SEP]``. A document of exactly
#: this length carries no content, which is degenerate but well defined;
#: anything shorter cannot even hold the marker prefix and the separator.
MIN_DOC_MAXLEN: int = 3

#: Offsets below ``n_vocab`` for the two marker ids. ``get_special_token_ids``
#: claims ``n_vocab - 20`` (cls) through ``n_vocab - 17`` (unk); these are the
#: next two free slots by that same convention.
_QUERY_MARKER_OFFSET: int = 16
_DOC_MARKER_OFFSET: int = 15

# ---------------------------------------------------------------------


class ColBERTTokenizer:
    """Build ColBERT's asymmetric query and document token streams.

    A plain Python class, deliberately **not** a ``keras.Layer``: everything it
    does is host-side ``str`` -> ``int`` work that runs once, outside the
    compute graph, and it owns a Tiktoken encoding that is not a Keras variable.

    :param encoding_name: Tiktoken encoding to wrap.
    :type encoding_name: str
    :param query_maxlen: Total query length including ``[CLS]``, ``[Q]`` and
        ``[SEP]``. Must be at least :data:`MIN_QUERY_MAXLEN`.
    :type query_maxlen: int
    :param doc_maxlen: Total document length including ``[CLS]``, ``[D]`` and
        ``[SEP]``. Must be at least :data:`MIN_DOC_MAXLEN`. The default of 220
        is the current official ``DocSettings`` default; the 180 sometimes
        quoted for ColBERT was not found in any consulted source.
    :type doc_maxlen: int
    :param mask_punctuation: Whether to build the document punctuation
        skiplist. When ``False``, :attr:`punctuation_token_ids` is empty and
        every ``skiplist_mask`` is all ones.
    :type mask_punctuation: bool
    :param attend_to_mask_tokens: Whether augmented ``[MASK]`` positions in a
        query receive ``attention_mask == 1``. Defaults to ``True`` -- see the
        module docstring for why this diverges from the reference default and
        what it costs.
    :type attend_to_mask_tokens: bool
    :param cls_token_id: Override for the ``[CLS]`` id.
    :type cls_token_id: Optional[int]
    :param sep_token_id: Override for the ``[SEP]`` id.
    :type sep_token_id: Optional[int]
    :param pad_token_id: Override for the ``[PAD]`` id.
    :type pad_token_id: Optional[int]
    :param mask_token_id: Override for the ``[MASK]`` id.
    :type mask_token_id: Optional[int]
    :param unk_token_id: Override for the ``[UNK]`` id. Not emitted by this
        class, but carried so the marker-collision check sees the full set of
        reserved ids and so ``get_config`` round-trips it.
    :type unk_token_id: Optional[int]
    :param query_marker_token_id: Override for the ``[Q]`` id. Defaults to
        ``n_vocab - 16``.
    :type query_marker_token_id: Optional[int]
    :param doc_marker_token_id: Override for the ``[D]`` id. Defaults to
        ``n_vocab - 15``.
    :type doc_marker_token_id: Optional[int]
    :raises ValueError: If either maximum length is below its minimum, if a
        marker id collides with a reserved special id or with the other marker,
        or if a marker id falls outside the vocabulary.

    Example:
        >>> tok = ColBERTTokenizer(query_maxlen=8, doc_maxlen=12)
        >>> q = tok.tokenize_queries(["what is late interaction"])
        >>> q["input_ids"].shape
        (1, 8)
        >>> d = tok.tokenize_documents(["late interaction, explained."])
        >>> sorted(d)
        ['attention_mask', 'input_ids', 'skiplist_mask']
    """

    def __init__(
        self,
        *,
        encoding_name: str = "cl100k_base",
        query_maxlen: int = 32,
        doc_maxlen: int = 220,
        mask_punctuation: bool = True,
        attend_to_mask_tokens: bool = True,
        cls_token_id: Optional[int] = None,
        sep_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        unk_token_id: Optional[int] = None,
        query_marker_token_id: Optional[int] = None,
        doc_marker_token_id: Optional[int] = None,
    ) -> None:
        if query_maxlen < MIN_QUERY_MAXLEN:
            raise ValueError(
                f"query_maxlen must be at least {MIN_QUERY_MAXLEN} to hold "
                f"'[CLS] [Q] <content> [SEP]', got {query_maxlen}"
            )
        if doc_maxlen < MIN_DOC_MAXLEN:
            raise ValueError(
                f"doc_maxlen must be at least {MIN_DOC_MAXLEN} to hold "
                f"'[CLS] [D] [SEP]', got {doc_maxlen}"
            )

        self.encoding_name: str = encoding_name
        self.query_maxlen: int = query_maxlen
        self.doc_maxlen: int = doc_maxlen
        self.mask_punctuation: bool = mask_punctuation
        self.attend_to_mask_tokens: bool = attend_to_mask_tokens

        self._encoding = tiktoken.get_encoding(encoding_name)
        n_vocab: int = self._encoding.n_vocab

        specials = get_special_token_ids(encoding_name)
        self.cls_token_id: int = (
            specials["cls"] if cls_token_id is None else int(cls_token_id)
        )
        self.sep_token_id: int = (
            specials["sep"] if sep_token_id is None else int(sep_token_id)
        )
        self.pad_token_id: int = (
            specials["pad"] if pad_token_id is None else int(pad_token_id)
        )
        self.mask_token_id: int = (
            specials["mask"] if mask_token_id is None else int(mask_token_id)
        )
        self.unk_token_id: int = (
            specials["unk"] if unk_token_id is None else int(unk_token_id)
        )

        # DECISION plan-2026-08-25T121346-c71fc3ad/D-010
        # Markers are DERIVED from the live n_vocab, never hardcoded to
        # 100261 / 100262. A different cl100k_base revision would shift
        # n_vocab, and a hardcoded literal would then silently land on a real
        # content token -- a collision that produces perfectly shaped tensors
        # and wrong retrieval forever. Do NOT "simplify" these to constants.
        # See decisions.md D-008 and D-022.
        self.query_marker_token_id: int = (
            n_vocab - _QUERY_MARKER_OFFSET
            if query_marker_token_id is None
            else int(query_marker_token_id)
        )
        self.doc_marker_token_id: int = (
            n_vocab - _DOC_MARKER_OFFSET
            if doc_marker_token_id is None
            else int(doc_marker_token_id)
        )
        self._validate_markers(n_vocab)

        self._punctuation_token_ids: frozenset = (
            self._build_punctuation_token_ids()
            if mask_punctuation
            else frozenset()
        )

        # Each preprocessor is asked for one slot LESS than the final length:
        # it emits '[CLS] <content> [SEP]', and the marker is spliced in at
        # position 1 afterwards, restoring the full width.
        self._query_preprocessor = self._build_preprocessor(query_maxlen - 1)
        self._doc_preprocessor = self._build_preprocessor(doc_maxlen - 1)

        logger.debug(
            f"ColBERTTokenizer ready: encoding={encoding_name}, "
            f"query_maxlen={query_maxlen}, doc_maxlen={doc_maxlen}, "
            f"[Q]={self.query_marker_token_id}, "
            f"[D]={self.doc_marker_token_id}, "
            f"|skiplist|={len(self._punctuation_token_ids)}, "
            f"attend_to_mask_tokens={attend_to_mask_tokens}"
        )

    # -----------------------------------------------------------------
    # construction helpers
    # -----------------------------------------------------------------

    def _validate_markers(self, n_vocab: int) -> None:
        """Reject a marker id that collides with a reserved id or the vocabulary.

        :param n_vocab: Size of the wrapped Tiktoken vocabulary.
        :type n_vocab: int
        :raises ValueError: On any collision or out-of-range marker.
        """
        reserved: Dict[str, int] = {
            "[CLS]": self.cls_token_id,
            "[SEP]": self.sep_token_id,
            "[PAD]": self.pad_token_id,
            "[MASK]": self.mask_token_id,
            "[UNK]": self.unk_token_id,
        }
        markers: Dict[str, int] = {
            "[Q]": self.query_marker_token_id,
            "[D]": self.doc_marker_token_id,
        }

        for marker_name, marker_id in markers.items():
            if not (0 <= marker_id < n_vocab):
                raise ValueError(
                    f"marker {marker_name} id {marker_id} is outside the "
                    f"vocabulary range [0, {n_vocab})"
                )
            for special_name, special_id in reserved.items():
                if marker_id == special_id:
                    raise ValueError(
                        f"marker {marker_name} id {marker_id} collides with "
                        f"special token {special_name} (id {special_id}); "
                        f"choose a free slot"
                    )

        if self.query_marker_token_id == self.doc_marker_token_id:
            raise ValueError(
                f"marker [Q] and marker [D] share id "
                f"{self.query_marker_token_id}; queries and documents would "
                f"be indistinguishable to the shared encoder"
            )

    def _build_punctuation_token_ids(self) -> frozenset:
        """Build the document skiplist id set.

        Adds, for each symbol in :data:`string.punctuation`, the first id of
        the bare symbol *and* the first id of the space-prefixed symbol -- see
        the module docstring's "Space-prefixed punctuation" deviation.

        :return: Ids whose document positions are filtered from the
            representation.
        :rtype: frozenset
        """
        # DECISION plan-2026-08-25T121346-c71fc3ad/D-009
        # The space-prefixed form is NOT redundant and must not be deleted as
        # "belt and braces". The reference's rule (bare symbol only) is a
        # WordPiece rule; under cl100k_base BPE, ordinary prose emits " ," (id
        # 1174), never bare "," (id 11), so a literal port builds a skiplist
        # that matches essentially nothing and the punctuation filter is dead
        # while every test on the id SET still passes. See decisions.md D-009.
        token_ids = set()
        for symbol in string.punctuation:
            token_ids.add(self._encoding.encode(symbol)[0])
            token_ids.add(self._encoding.encode(f" {symbol}")[0])
        return frozenset(token_ids)

    def _build_preprocessor(self, max_length: int) -> TiktokenPreprocessor:
        """Instantiate the wrapped preprocessor with this class's special ids.

        :param max_length: Width the preprocessor should emit, which is one
            less than the final width because the marker is spliced in later.
        :type max_length: int
        :return: A configured preprocessor.
        :rtype: TiktokenPreprocessor
        """
        return TiktokenPreprocessor(
            encoding_name=self.encoding_name,
            max_length=max_length,
            cls_token_id=self.cls_token_id,
            sep_token_id=self.sep_token_id,
            pad_token_id=self.pad_token_id,
            mask_token_id=self.mask_token_id,
            truncation=True,
            padding="max_length",
        )

    # -----------------------------------------------------------------
    # public surface
    # -----------------------------------------------------------------

    @property
    def punctuation_token_ids(self) -> frozenset:
        """Ids filtered out of document representations.

        Empty when ``mask_punctuation`` is ``False``. Never applied to
        queries.

        :return: The skiplist.
        :rtype: frozenset
        """
        return self._punctuation_token_ids

    def tokenize_queries(self, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        """Encode queries as ``[CLS] [Q] <content> [SEP] [MASK] ... [MASK]``.

        Every slot after ``[SEP]`` is a ``[MASK]``, never a ``[PAD]``. Content
        is truncated so that the ``[SEP]`` always survives.

        :param texts: Query strings. A bare ``str`` is rejected -- pass a list.
        :type texts: Sequence[str]
        :return: ``input_ids`` and ``attention_mask``, both
            ``(batch, query_maxlen)`` ``int32``.
        :rtype: Dict[str, np.ndarray]
        :raises TypeError: If ``texts`` is a bare string or holds a non-string.
        :raises ValueError: If ``texts`` is empty.
        """
        input_ids, attention_mask = self._tensorize(
            texts,
            preprocessor=self._query_preprocessor,
            marker_token_id=self.query_marker_token_id,
        )

        # The reference mechanism verbatim: pad normally, then overwrite every
        # pad slot with [MASK]. Content can never produce the pad id, so this
        # selects exactly the padding region.
        augmented = input_ids == self.pad_token_id
        input_ids = np.where(
            augmented, np.int32(self.mask_token_id), input_ids
        ).astype(np.int32)
        # DECISION plan-2026-08-25T121346-c71fc3ad/D-008
        # Do NOT "restore fidelity" by flipping this default to the
        # reference's False. This class emits ONE attention_mask that a
        # consumer uses both as the transformer mask and as the projection's
        # participation mask; the reference's participation mask over queries
        # is all-ones by construction. A False default therefore zeroes every
        # augmented [MASK] embedding's MaxSim contribution and silently kills
        # query augmentation with a green suite. See decisions.md D-008 and
        # the module docstring's "Attention over augmented [MASK] slots".
        if self.attend_to_mask_tokens:
            attention_mask = np.where(
                augmented, np.int32(1), attention_mask
            ).astype(np.int32)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def tokenize_documents(self, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        """Encode documents as ``[CLS] [D] <content> [SEP] [PAD] ... [PAD]``.

        Documents are never ``[MASK]``-augmented. The returned
        ``skiplist_mask`` is 0 exactly at punctuation positions and 1
        everywhere else, independently of padding; a consumer forms the
        reference's participation mask as
        ``attention_mask * skiplist_mask``.

        :param texts: Document strings. A bare ``str`` is rejected.
        :type texts: Sequence[str]
        :return: ``input_ids``, ``attention_mask`` and ``skiplist_mask``, all
            ``(batch, doc_maxlen)`` ``int32``.
        :rtype: Dict[str, np.ndarray]
        :raises TypeError: If ``texts`` is a bare string or holds a non-string.
        :raises ValueError: If ``texts`` is empty.
        """
        input_ids, attention_mask = self._tensorize(
            texts,
            preprocessor=self._doc_preprocessor,
            marker_token_id=self.doc_marker_token_id,
        )

        skiplist_mask = np.ones_like(input_ids, dtype=np.int32)
        if self._punctuation_token_ids:
            is_punctuation = np.isin(
                input_ids, np.array(sorted(self._punctuation_token_ids), dtype=np.int32)
            )
            skiplist_mask = np.where(
                is_punctuation, np.int32(0), skiplist_mask
            ).astype(np.int32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "skiplist_mask": skiplist_mask,
        }

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument by name.

        :return: A dict accepted verbatim by :meth:`from_config`.
        :rtype: Dict[str, Any]
        """
        return {
            "encoding_name": self.encoding_name,
            "query_maxlen": self.query_maxlen,
            "doc_maxlen": self.doc_maxlen,
            "mask_punctuation": self.mask_punctuation,
            "attend_to_mask_tokens": self.attend_to_mask_tokens,
            "cls_token_id": self.cls_token_id,
            "sep_token_id": self.sep_token_id,
            "pad_token_id": self.pad_token_id,
            "mask_token_id": self.mask_token_id,
            "unk_token_id": self.unk_token_id,
            "query_marker_token_id": self.query_marker_token_id,
            "doc_marker_token_id": self.doc_marker_token_id,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ColBERTTokenizer":
        """Rebuild a tokenizer from :meth:`get_config` output.

        :param config: Constructor arguments.
        :type config: Dict[str, Any]
        :return: An equivalent tokenizer.
        :rtype: ColBERTTokenizer
        """
        return cls(**config)

    def __repr__(self) -> str:
        """:return: A short debugging representation.

        :rtype: str
        """
        return (
            f"ColBERTTokenizer(encoding={self.encoding_name}, "
            f"query_maxlen={self.query_maxlen}, "
            f"doc_maxlen={self.doc_maxlen}, "
            f"mask_punctuation={self.mask_punctuation})"
        )

    # -----------------------------------------------------------------
    # internals
    # -----------------------------------------------------------------

    def _tensorize(
        self,
        texts: Sequence[str],
        preprocessor: TiktokenPreprocessor,
        marker_token_id: int,
    ) -> tuple:
        """Encode a batch and splice the marker in at position 1.

        :param texts: Input strings.
        :type texts: Sequence[str]
        :param preprocessor: The side-specific wrapped preprocessor.
        :type preprocessor: TiktokenPreprocessor
        :param marker_token_id: ``[Q]`` or ``[D]``.
        :type marker_token_id: int
        :return: ``(input_ids, attention_mask)``, both ``int32`` and one column
            wider than what the preprocessor emitted.
        :rtype: tuple
        :raises TypeError: If ``texts`` is a bare string or holds a non-string.
        :raises ValueError: If ``texts`` is empty.
        """
        if isinstance(texts, str):
            raise TypeError(
                "texts must be a sequence of strings, not a bare str; "
                "wrap it in a list to encode a single item"
            )
        text_list: List[str] = list(texts)
        if not text_list:
            raise ValueError("texts must contain at least one string")
        for index, item in enumerate(text_list):
            if not isinstance(item, str):
                raise TypeError(
                    f"texts[{index}] must be a str, got "
                    f"{type(item).__name__}"
                )

        encoded = preprocessor(text_list)
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int32)
        attention_mask = np.asarray(encoded["attention_mask"], dtype=np.int32)

        batch = input_ids.shape[0]
        marker_column = np.full((batch, 1), marker_token_id, dtype=np.int32)
        attended_column = np.ones((batch, 1), dtype=np.int32)

        input_ids = np.concatenate(
            [input_ids[:, :1], marker_column, input_ids[:, 1:]], axis=1
        )
        attention_mask = np.concatenate(
            [attention_mask[:, :1], attended_column, attention_mask[:, 1:]],
            axis=1,
        )
        return input_ids, attention_mask

# ---------------------------------------------------------------------
