"""
Guards for :class:`ColBERTTokenizer`'s two asymmetric input paths.

The whole query/document asymmetry of ColBERT lives in the tokenizer, and every
part of it is the kind of defect that leaves shapes, dtypes and dtype-invariant
smoke tests perfectly green: a query padded with ``[PAD]`` instead of ``[MASK]``
still has the right shape, a document marked ``[Q]`` still encodes, and a
skiplist applied to both sides still returns arrays of ones and zeros. So each
guard below asserts **both directions** of the asymmetry -- what must happen on
one side and what must *not* happen on the other -- rather than only the
positive half.

RED-proof record (2026-08-25, iter-1/step-2). Each injection was applied to
``tokenization.py`` from a ``cp`` backup, the suite run, and the source restored
and verified with ``diff -q``:

* **(a) pad queries with ``[PAD]`` instead of ``[MASK]``** -- removed the
  pad-to-mask overwrite in ``tokenize_queries``.
  RED: ``test_every_query_slot_after_sep_is_mask_and_never_pad``, at the
  assertion ``"query slot ... is [PAD]; augmentation is dead"``.
* **(b) apply the punctuation skiplist to queries too** -- had
  ``tokenize_queries`` compute and return a ``skiplist_mask``.
  RED: ``test_the_punctuation_skiplist_is_documents_only``, at the assertion
  ``"queries must not carry a skiplist_mask"``.
* **(c) use the ``[D]`` marker for queries** -- passed
  ``self.doc_marker_token_id`` as the query marker.
  RED: ``test_every_query_slot_after_sep_is_mask_and_never_pad``, at the
  assertion ``"query position 1 must be the [Q] marker"``.
"""

import string
import pytest
import tiktoken
import numpy as np

from dl_techniques.models.language.colbert.tokenization import (
    ColBERTTokenizer,
    MIN_DOC_MAXLEN,
    MIN_QUERY_MAXLEN,
)

# ---------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------

#: A punctuated text used on BOTH sides, so the documents-only skiplist and the
#: queries-only augmentation can each be checked against the same input.
PUNCTUATED_TEXT = "late interaction , explained ."


@pytest.fixture(scope="module")
def encoding():
    """The raw Tiktoken encoding, used to derive expectations independently."""
    return tiktoken.get_encoding("cl100k_base")


@pytest.fixture
def tokenizer():
    """A small tokenizer: short enough to read a whole row in a failure message."""
    return ColBERTTokenizer(query_maxlen=12, doc_maxlen=16)


# ---------------------------------------------------------------------
# 1. query layout
# ---------------------------------------------------------------------


def test_every_query_slot_after_sep_is_mask_and_never_pad(tokenizer):
    """``[CLS] [Q] <content> [SEP]`` then MASK to the end -- never PAD."""
    out = tokenizer.tokenize_queries(["what is late interaction"])
    ids = out["input_ids"][0]

    assert ids.shape == (tokenizer.query_maxlen,)
    assert out["input_ids"].dtype == np.int32
    assert ids[0] == tokenizer.cls_token_id, (
        f"query position 0 must be [CLS] ({tokenizer.cls_token_id}), "
        f"got {ids[0]}"
    )
    assert ids[1] == tokenizer.query_marker_token_id, (
        f"query position 1 must be the [Q] marker "
        f"({tokenizer.query_marker_token_id}), got {ids[1]}; a query marked "
        f"[D] ({tokenizer.doc_marker_token_id}) is indistinguishable from a "
        f"document to the shared encoder"
    )

    sep_positions = np.flatnonzero(ids == tokenizer.sep_token_id)
    assert sep_positions.size == 1, (
        f"query must contain exactly one [SEP], found {sep_positions.size} "
        f"in {ids.tolist()}"
    )
    sep = int(sep_positions[0])
    assert sep >= 2, "the [SEP] must follow the [CLS] [Q] prefix"

    for position in range(sep + 1, tokenizer.query_maxlen):
        assert ids[position] != tokenizer.pad_token_id, (
            f"query slot {position} is [PAD]; augmentation is dead -- every "
            f"slot after [SEP] must be [MASK] "
            f"({tokenizer.mask_token_id}). Row: {ids.tolist()}"
        )
        assert ids[position] == tokenizer.mask_token_id, (
            f"query slot {position} is {ids[position]}, expected [MASK] "
            f"({tokenizer.mask_token_id}). Row: {ids.tolist()}"
        )

    assert sep + 1 < tokenizer.query_maxlen, (
        "this fixture query must leave at least one augmented slot, "
        "otherwise the loop above is vacuous"
    )


# ---------------------------------------------------------------------
# 2. document layout (the other direction of H-5)
# ---------------------------------------------------------------------


def test_documents_are_pad_padded_and_never_mask_augmented(tokenizer):
    """``[CLS] [D] <content> [SEP]`` then PAD -- documents are never augmented."""
    out = tokenizer.tokenize_documents(["late interaction"])
    ids = out["input_ids"][0]
    attention = out["attention_mask"][0]

    assert ids.shape == (tokenizer.doc_maxlen,)
    assert ids[0] == tokenizer.cls_token_id
    assert ids[1] == tokenizer.doc_marker_token_id, (
        f"document position 1 must be the [D] marker "
        f"({tokenizer.doc_marker_token_id}), got {ids[1]}"
    )

    sep_positions = np.flatnonzero(ids == tokenizer.sep_token_id)
    assert sep_positions.size == 1
    sep = int(sep_positions[0])
    assert sep + 1 < tokenizer.doc_maxlen, (
        "this fixture document must leave at least one padding slot, "
        "otherwise the loop below is vacuous"
    )

    for position in range(sep + 1, tokenizer.doc_maxlen):
        assert ids[position] == tokenizer.pad_token_id, (
            f"document slot {position} is {ids[position]}, expected [PAD] "
            f"({tokenizer.pad_token_id}). Row: {ids.tolist()}"
        )
        assert ids[position] != tokenizer.mask_token_id, (
            f"document slot {position} is [MASK]; query augmentation must "
            f"never be applied to documents. Row: {ids.tolist()}"
        )
        assert attention[position] == 0, (
            f"document padding slot {position} must have attention 0, "
            f"got {attention[position]}"
        )

    assert np.all(attention[: sep + 1] == 1)


# ---------------------------------------------------------------------
# 3. punctuation asymmetry, both directions
# ---------------------------------------------------------------------


def test_the_punctuation_skiplist_is_documents_only(tokenizer, encoding):
    """Documents get zeros at exactly the punctuation ids; queries are untouched."""
    document = tokenizer.tokenize_documents([PUNCTUATED_TEXT])
    ids = document["input_ids"][0]
    skiplist_mask = document["skiplist_mask"][0]

    expected_zero = np.array(
        [tid in tokenizer.punctuation_token_ids for tid in ids.tolist()]
    )
    assert expected_zero.any(), (
        f"fixture text {PUNCTUATED_TEXT!r} produced no punctuation tokens "
        f"({ids.tolist()}); this test would be vacuous"
    )
    np.testing.assert_array_equal(
        skiplist_mask == 0,
        expected_zero,
        err_msg=(
            "skiplist_mask must be 0 at exactly the punctuation positions "
            f"and 1 everywhere else. ids={ids.tolist()} "
            f"mask={skiplist_mask.tolist()}"
        ),
    )

    # The other direction: the SAME punctuation in a query is untouched.
    query = tokenizer.tokenize_queries([PUNCTUATED_TEXT])
    assert "skiplist_mask" not in query, (
        "queries must not carry a skiplist_mask -- the reference passes "
        f"skiplist=[] on the query path. Got keys {sorted(query)}"
    )
    query_ids = query["input_ids"][0]
    punctuated_in_query = [
        int(tid)
        for tid in query_ids.tolist()
        if tid in tokenizer.punctuation_token_ids
    ]
    assert punctuated_in_query, (
        "the query fixture must itself contain punctuation ids, otherwise "
        "the queries-are-unaffected direction is vacuous"
    )
    for position, tid in enumerate(query_ids.tolist()):
        if tid in tokenizer.punctuation_token_ids:
            assert query["attention_mask"][0][position] == 1, (
                f"query punctuation at position {position} was masked out; "
                f"punctuation filtering is documents-only"
            )


def test_disabling_punctuation_masking_empties_the_skiplist():
    """``mask_punctuation=False`` -> empty id set and an all-ones mask."""
    tokenizer = ColBERTTokenizer(
        query_maxlen=12, doc_maxlen=16, mask_punctuation=False
    )
    assert tokenizer.punctuation_token_ids == frozenset()

    out = tokenizer.tokenize_documents([PUNCTUATED_TEXT])
    assert np.all(out["skiplist_mask"] == 1), (
        "with mask_punctuation=False every skiplist_mask entry must be 1, "
        f"got {out['skiplist_mask'][0].tolist()}"
    )


def test_the_skiplist_covers_every_punctuation_symbol_in_both_spacings(
    tokenizer, encoding
):
    """Bare and space-prefixed forms of every symbol are both in the skiplist.

    The space-prefixed half is this repository's named deviation from the
    reference rule (a BPE tokenizer emits ``" ,"`` in prose, not ``","``).
    """
    for symbol in string.punctuation:
        bare = encoding.encode(symbol)[0]
        spaced = encoding.encode(f" {symbol}")[0]
        assert bare in tokenizer.punctuation_token_ids, (
            f"bare punctuation {symbol!r} (id {bare}) missing from skiplist"
        )
        assert spaced in tokenizer.punctuation_token_ids, (
            f"space-prefixed punctuation {' ' + symbol!r} (id {spaced}) "
            f"missing from skiplist; a BPE document would slip past the filter"
        )


# ---------------------------------------------------------------------
# 4. truncation and the exactly-full budget
# ---------------------------------------------------------------------


def test_a_long_document_is_truncated_with_its_sep_surviving(tokenizer):
    """Overflow truncates content, never the ``[SEP]``."""
    out = tokenizer.tokenize_documents([" ".join(["token"] * 500)])
    ids = out["input_ids"][0]

    assert ids.shape == (tokenizer.doc_maxlen,)
    assert ids[-1] == tokenizer.sep_token_id, (
        f"a document that overflows doc_maxlen must end in [SEP] "
        f"({tokenizer.sep_token_id}), got {ids[-1]}. Row: {ids.tolist()}"
    )
    assert ids[0] == tokenizer.cls_token_id
    assert ids[1] == tokenizer.doc_marker_token_id
    assert np.all(out["attention_mask"][0] == 1)
    assert not np.any(ids == tokenizer.pad_token_id)


def test_a_query_that_exactly_fills_its_budget_keeps_sep_and_gets_no_masks(
    encoding,
):
    """Zero augmentation slots is a legal state, not an error."""
    query_maxlen = 12
    content_budget = query_maxlen - 3  # [CLS] [Q] ... [SEP]
    text = " ".join(["token"] * content_budget)
    assert len(encoding.encode(text)) == content_budget, (
        "fixture text must encode to exactly the content budget for this "
        "test to exercise the boundary"
    )

    tokenizer = ColBERTTokenizer(query_maxlen=query_maxlen, doc_maxlen=16)
    ids = tokenizer.tokenize_queries([text])["input_ids"][0]

    assert ids[-1] == tokenizer.sep_token_id, (
        f"an exactly-full query must still end in [SEP], got {ids[-1]}"
    )
    assert not np.any(ids == tokenizer.mask_token_id), (
        f"an exactly-full query has no augmentation slots, found [MASK] in "
        f"{ids.tolist()}"
    )
    assert not np.any(ids == tokenizer.pad_token_id)


# ---------------------------------------------------------------------
# 5. construction-time guards
# ---------------------------------------------------------------------


def test_a_marker_colliding_with_a_special_id_raises():
    """A marker landing on a reserved id must name the collision."""
    probe = ColBERTTokenizer(query_maxlen=8, doc_maxlen=8)

    with pytest.raises(ValueError, match=r"\[MASK\]"):
        ColBERTTokenizer(
            query_maxlen=8,
            doc_maxlen=8,
            query_marker_token_id=probe.mask_token_id,
        )
    with pytest.raises(ValueError, match=r"\[CLS\]"):
        ColBERTTokenizer(
            query_maxlen=8,
            doc_maxlen=8,
            doc_marker_token_id=probe.cls_token_id,
        )


def test_the_two_markers_may_not_share_an_id():
    """``[Q] == [D]`` erases the only signal telling the encoder them apart."""
    with pytest.raises(ValueError, match="indistinguishable"):
        ColBERTTokenizer(
            query_maxlen=8,
            doc_maxlen=8,
            query_marker_token_id=50,
            doc_marker_token_id=50,
        )


@pytest.mark.parametrize(
    "kwargs, minimum",
    [
        ({"query_maxlen": 2, "doc_maxlen": 16}, MIN_QUERY_MAXLEN),
        ({"query_maxlen": 12, "doc_maxlen": 1}, MIN_DOC_MAXLEN),
    ],
)
def test_a_too_small_maxlen_raises_naming_the_minimum(kwargs, minimum):
    """The error must quote the minimum, not just fail somewhere downstream."""
    with pytest.raises(ValueError, match=f"at least {minimum}"):
        ColBERTTokenizer(**kwargs)


def test_markers_are_derived_from_the_live_vocabulary(encoding):
    """The markers follow ``get_special_token_ids``'s own downward convention."""
    tokenizer = ColBERTTokenizer(query_maxlen=8, doc_maxlen=8)
    assert tokenizer.query_marker_token_id == encoding.n_vocab - 16
    assert tokenizer.doc_marker_token_id == encoding.n_vocab - 15
    assert tokenizer.query_marker_token_id not in {
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
        tokenizer.mask_token_id,
        tokenizer.unk_token_id,
    }


# ---------------------------------------------------------------------
# 6. attend_to_mask_tokens policy pin
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "attend_to_mask_tokens, expected_attention",
    [(True, 1), (False, 0)],
)
def test_the_attend_to_mask_tokens_policy_is_pinned(
    attend_to_mask_tokens, expected_attention
):
    """Default ``True``: augmented slots are attended and participate downstream.

    This is a deliberate divergence from the reference's ``QuerySettings``
    default of ``False`` -- see the module docstring of ``tokenization.py``.
    """
    tokenizer = ColBERTTokenizer(
        query_maxlen=12,
        doc_maxlen=16,
        attend_to_mask_tokens=attend_to_mask_tokens,
    )
    out = tokenizer.tokenize_queries(["late interaction"])
    ids = out["input_ids"][0]
    attention = out["attention_mask"][0]

    augmented = ids == tokenizer.mask_token_id
    assert augmented.any(), "fixture must leave augmented slots"
    assert np.all(attention[augmented] == expected_attention), (
        f"with attend_to_mask_tokens={attend_to_mask_tokens} every augmented "
        f"[MASK] slot must have attention {expected_attention}, got "
        f"{attention.tolist()}"
    )
    # Real content is attended either way.
    assert np.all(attention[~augmented] == 1)


def test_the_default_attends_to_mask_tokens():
    """The documented default is ``True``; a silent flip would be invisible."""
    assert ColBERTTokenizer().attend_to_mask_tokens is True


# ---------------------------------------------------------------------
# 7. config round trip and input validation
# ---------------------------------------------------------------------


def test_get_config_round_trip_reconstructs_an_equivalent_tokenizer():
    """Every constructor argument survives ``get_config`` -> ``from_config``."""
    original = ColBERTTokenizer(
        query_maxlen=10,
        doc_maxlen=20,
        mask_punctuation=True,
        attend_to_mask_tokens=False,
    )
    config = original.get_config()
    restored = ColBERTTokenizer.from_config(config)

    assert restored.get_config() == config
    assert restored.query_maxlen == original.query_maxlen
    assert restored.doc_maxlen == original.doc_maxlen
    assert restored.query_marker_token_id == original.query_marker_token_id
    assert restored.doc_marker_token_id == original.doc_marker_token_id
    assert restored.punctuation_token_ids == original.punctuation_token_ids
    assert restored.attend_to_mask_tokens is False

    for method in ("tokenize_queries", "tokenize_documents"):
        before = getattr(original, method)([PUNCTUATED_TEXT])
        after = getattr(restored, method)([PUNCTUATED_TEXT])
        assert sorted(before) == sorted(after)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])


def test_a_bare_string_is_rejected(tokenizer):
    """``"abc"`` is a ``Sequence[str]`` by accident; encoding it per character
    would silently produce a batch of three."""
    with pytest.raises(TypeError, match="bare str"):
        tokenizer.tokenize_queries("late interaction")
    with pytest.raises(TypeError, match="bare str"):
        tokenizer.tokenize_documents("late interaction")


def test_an_empty_batch_is_rejected(tokenizer):
    """An empty list would produce a ``(0, maxlen)`` array downstream."""
    with pytest.raises(ValueError, match="at least one string"):
        tokenizer.tokenize_queries([])


def test_a_batch_encodes_row_wise(tokenizer):
    """Batch shape and per-row independence."""
    texts = ["late interaction", "a much longer query about retrieval systems"]
    out = tokenizer.tokenize_queries(texts)
    assert out["input_ids"].shape == (2, tokenizer.query_maxlen)
    for row, text in enumerate(texts):
        single = tokenizer.tokenize_queries([text])["input_ids"][0]
        np.testing.assert_array_equal(out["input_ids"][row], single)
