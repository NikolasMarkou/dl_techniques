"""Tests for the character-level ASCII tokenizer.

Covers ``ASCIICharTokenizer`` (the serializable Keras-layer artifact) and
``ASCIICharPreprocessor`` (the plain object implementing the ``tf.data``
pipeline contract), plus the module-level pure mapping functions.

Two of these are single-claim guards rather than coverage: the special-id
exclusion (which is what makes a non-pad length count correct by
construction) and the pipeline contract (which is the seam that would
otherwise drift between the two classes).
"""

import itertools

import keras
import numpy as np
import pytest

from dl_techniques.layers.tokenizers.ascii_char import (
    CLS_ID,
    ENCODING_NAME,
    FIRST_PRINTABLE,
    LAST_PRINTABLE,
    MASK_ID,
    NEWLINE_ID,
    NUM_SPECIAL_TOKENS,
    PAD_ID,
    SEP_ID,
    UNK_ID,
    VOCAB_SIZE,
    ASCIICharPreprocessor,
    ASCIICharTokenizer,
    char_to_id,
    decode_ascii,
    encode_ascii,
    id_to_char,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

PRINTABLE = "".join(chr(c) for c in range(FIRST_PRINTABLE, LAST_PRINTABLE + 1))

#: Characters that must never crash the encoder and must never produce a
#: special id: C0/C1 controls, Latin-1, Greek, CJK, emoji, currency.
FUZZ_CODE_POINTS = list(
    itertools.chain(
        range(0x00, 0x100),
        range(0x370, 0x400),
        [0x4E2D, 0x6587, 0x1F600, 0x1F4A9, 0x20AC, 0x00A3, 0x2665, 0x00A9],
    )
)


@pytest.fixture
def tokenizer():
    """A tokenizer with a short sequence budget."""
    return ASCIICharTokenizer(max_length=32)


@pytest.fixture
def preprocessor():
    """A preprocessor with a short sequence budget."""
    return ASCIICharPreprocessor(max_length=16)


# ---------------------------------------------------------------------
# Vocabulary layout
# ---------------------------------------------------------------------

class TestVocabularyLayout:
    """The vocabulary is a fixed, corpus-independent mapping."""

    def test_vocab_size_is_six_specials_plus_ninety_five_printable(self):
        assert NUM_SPECIAL_TOKENS == 6
        assert LAST_PRINTABLE - FIRST_PRINTABLE + 1 == 95
        assert VOCAB_SIZE == 101

    def test_every_printable_character_has_a_distinct_id(self):
        ids = [char_to_id(c) for c in PRINTABLE]
        assert len(set(ids)) == len(PRINTABLE)
        assert min(ids) == NUM_SPECIAL_TOKENS
        assert max(ids) == VOCAB_SIZE - 1

    def test_char_to_id_and_id_to_char_are_inverse_on_printable(self):
        for c in PRINTABLE:
            assert id_to_char(char_to_id(c)) == c

    def test_id_to_char_rejects_out_of_range(self):
        with pytest.raises(ValueError, match=r"\[0, 101\)"):
            id_to_char(VOCAB_SIZE)
        with pytest.raises(ValueError, match=r"\[0, 101\)"):
            id_to_char(-1)

    def test_newline_has_its_own_id_and_survives_a_round_trip(self):
        assert char_to_id("\n") == NEWLINE_ID
        assert decode_ascii(encode_ascii("a\nb")) == "a\nb"

    def test_tab_and_carriage_return_fold_onto_space(self):
        space_id = char_to_id(" ")
        assert char_to_id("\t") == space_id
        assert char_to_id("\r") == space_id


# ---------------------------------------------------------------------
# The special-id guard
# ---------------------------------------------------------------------

class TestEncodeNeverEmitsASpecialId:
    """``encode_ascii`` output can never collide with a framing token.

    This is what makes "sequence length = count of non-``PAD`` positions"
    correct by construction rather than by convention, which every masked
    pooling and last-token gather downstream depends on.
    """

    def test_no_special_id_over_a_wide_fuzz_corpus(self):
        forbidden = {PAD_ID, CLS_ID, SEP_ID, MASK_ID}
        leaks = []
        for code_point in FUZZ_CODE_POINTS:
            for token_id in encode_ascii(chr(code_point)):
                if token_id in forbidden:
                    leaks.append((code_point, token_id))
        assert leaks == []

    def test_ids_are_always_in_range(self):
        for code_point in FUZZ_CODE_POINTS:
            for token_id in encode_ascii(chr(code_point)):
                assert 0 <= token_id < VOCAB_SIZE

    def test_unrepresentable_characters_become_unk_not_dropped(self):
        # Alignment must hold: three characters in, three ids out.
        ids = encode_ascii("a♥b")
        assert len(ids) == 3
        assert ids[1] == UNK_ID

    def test_encoding_never_raises_on_arbitrary_input(self):
        text = "".join(chr(c) for c in FUZZ_CODE_POINTS)
        ids = encode_ascii(text)
        assert all(0 <= i < VOCAB_SIZE for i in ids)

    def test_alignment_is_exact_when_unicode_folding_is_off(self):
        """One character in, one id out -- with folding disabled.

        With folding ON this does not hold, and deliberately so: NFKD
        EXPANDS a few characters. Measured over the fuzz corpus, exactly
        three do -- the vulgar fractions U+00BC/BD/BE, each decomposing to
        three characters (e.g. "1", U+2044, "4"), so 408 characters encode
        to 414 ids. That is a property of NFKD, not a defect, and it is
        pinned here so the expansion is not mistaken for dropped input.
        """
        text = "".join(chr(c) for c in FUZZ_CODE_POINTS)
        assert len(encode_ascii(text, normalize_unicode=False)) == len(text)

        expanding = [c for c in FUZZ_CODE_POINTS if len(encode_ascii(chr(c))) != 1]
        assert expanding == [0xBC, 0xBD, 0xBE]
        assert len(encode_ascii(text)) == len(text) + 6


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

class TestNormalization:
    """Case folding and NFKD accent folding."""

    def test_accents_fold_onto_their_ascii_base(self):
        assert decode_ascii(encode_ascii("café")) == "cafe"
        assert decode_ascii(encode_ascii("naïve résumé")) == "naive resume"

    def test_lowercase_is_applied_by_default(self):
        assert decode_ascii(encode_ascii("ABC")) == "abc"

    def test_lowercase_can_be_disabled(self):
        assert decode_ascii(encode_ascii("ABC", lowercase=False)) == "ABC"

    def test_unicode_normalization_can_be_disabled(self):
        ids = encode_ascii("café", normalize_unicode=False)
        assert ids[-1] == UNK_ID

    def test_round_trip_is_identity_on_lowercase_printable_text(self):
        text = "the quick brown fox jumps over 13 lazy dogs! (x=2*3; y<=4)"
        assert decode_ascii(encode_ascii(text)) == text

    def test_bytes_input_is_decoded_as_utf8(self):
        assert decode_ascii(encode_ascii(b"hello")) == "hello"


# ---------------------------------------------------------------------
# ASCIICharTokenizer
# ---------------------------------------------------------------------

class TestASCIICharTokenizer:
    """The serializable Keras-layer surface."""

    def test_rejects_non_positive_max_length(self):
        with pytest.raises(ValueError, match="must be positive"):
            ASCIICharTokenizer(max_length=0)

    def test_rejects_max_length_too_small_for_framing(self):
        with pytest.raises(ValueError, match="room for"):
            ASCIICharTokenizer(max_length=2)

    def test_rejects_non_integer_max_length(self):
        with pytest.raises(ValueError, match="must be an int"):
            ASCIICharTokenizer(max_length=8.0)

    def test_framing_layout_is_cls_content_sep_then_pad(self, tokenizer):
        ids = tokenizer.encode_with_framing("hi")
        assert len(ids) == tokenizer.max_length
        assert ids[0] == CLS_ID
        assert ids[3] == SEP_ID
        assert set(ids[4:]) == {PAD_ID}

    def test_pad_appears_only_as_a_contiguous_suffix(self, tokenizer):
        ids = tokenizer.encode_with_framing("some short text")
        first_pad = ids.index(PAD_ID)
        assert set(ids[first_pad:]) == {PAD_ID}
        assert PAD_ID not in ids[:first_pad]

    def test_truncation_keeps_sep_at_the_final_slot(self, tokenizer):
        ids = tokenizer.encode_with_framing("x" * 500)
        assert len(ids) == tokenizer.max_length
        assert ids[0] == CLS_ID
        assert ids[-1] == SEP_ID
        assert PAD_ID not in ids

    def test_encode_adds_no_framing(self, tokenizer):
        ids = tokenizer.encode("hi")
        assert len(ids) == 2
        assert CLS_ID not in ids and SEP_ID not in ids

    def test_tokenize_texts_returns_one_fixed_length_row_per_text(self, tokenizer):
        rows = tokenizer.tokenize_texts(["a", "bb", "ccc"])
        assert len(rows) == 3
        assert all(len(r) == tokenizer.max_length for r in rows)

    def test_decode_tokens_matches_decode(self, tokenizer):
        ids = tokenizer.encode("hello")
        assert tokenizer.decode_tokens(ids) == tokenizer.decode(ids)

    def test_decode_can_keep_special_tokens(self, tokenizer):
        ids = tokenizer.encode_with_framing("hi")
        assert tokenizer.decode(ids) == "hi"
        kept = tokenizer.decode(ids, skip_special_tokens=False)
        assert kept.startswith("[CLS]") and "[SEP]" in kept

    def test_call_raises(self, tokenizer):
        with pytest.raises(NotImplementedError, match="backend-agnostic"):
            tokenizer(np.array(["hello"]))

    def test_compute_output_shape(self, tokenizer):
        assert tokenizer.compute_output_shape((4,)) == (4, tokenizer.max_length)

    def test_config_round_trip_preserves_behaviour(self, tokenizer):
        restored = ASCIICharTokenizer.from_config(tokenizer.get_config())
        assert restored.max_length == tokenizer.max_length
        assert restored.lowercase == tokenizer.lowercase
        assert restored.normalize_unicode == tokenizer.normalize_unicode
        text = "Round-Trip Test 42!"
        assert restored.encode(text) == tokenizer.encode(text)

    def test_encoding_name_is_stable_across_instances(self):
        """A Keras auto-name is process-uniquified and must never key a cache."""
        instances = [ASCIICharTokenizer(max_length=8) for _ in range(3)]
        assert {t.encoding_name for t in instances} == {ENCODING_NAME}
        # The Keras name is precisely what this property exists to avoid.
        assert len({t.name for t in instances}) == 3

    def test_token_id_properties_match_module_constants(self, tokenizer):
        assert tokenizer.pad_token_id == PAD_ID
        assert tokenizer.cls_token_id == CLS_ID
        assert tokenizer.sep_token_id == SEP_ID
        assert tokenizer.mask_token_id == MASK_ID
        assert tokenizer.unk_token_id == UNK_ID
        assert tokenizer.vocab_size == VOCAB_SIZE
        assert tokenizer.n_vocab == VOCAB_SIZE

    def test_is_registered_as_serializable(self):
        # A bare register_keras_serializable() registers under the default
        # "Custom" package, the same key shape BPETokenizer gets.
        cls = keras.saving.get_registered_object("Custom>ASCIICharTokenizer")
        assert cls is ASCIICharTokenizer


# ---------------------------------------------------------------------
# ASCIICharPreprocessor -- the tf.data pipeline contract
# ---------------------------------------------------------------------

class TestPreprocessorPipelineContract:
    """The exact contract ``train.common.nlp`` depends on.

    ``preprocess_mlm_dataset`` does::

        encoded = preprocessor(decode_text(text), return_tensors='np')
        return (encoded['input_ids'][0], encoded['attention_mask'][0],
                encoded['token_type_ids'][0])

    and then applies ``tf.ensure_shape(ids, [seq_len])``. So the result must
    be batched, must carry exactly those three keys, and every row must be
    exactly ``max_length`` long.
    """

    def test_returns_exactly_the_three_required_keys(self, preprocessor):
        out = preprocessor("hello", return_tensors="np")
        assert set(out) == {"input_ids", "attention_mask", "token_type_ids"}

    def test_a_single_string_is_batched_to_rank_two(self, preprocessor):
        out = preprocessor("hello", return_tensors="np")
        for key, value in out.items():
            assert value.ndim == 2, key
            assert value.shape == (1, preprocessor.max_length), key

    def test_rows_are_always_exactly_max_length(self, preprocessor):
        for text in ["", "a", "hello world", "x" * 500]:
            out = preprocessor(text, return_tensors="np")
            assert out["input_ids"].shape == (1, preprocessor.max_length)

    def test_output_is_integer_typed(self, preprocessor):
        out = preprocessor("hello", return_tensors="np")
        for key, value in out.items():
            assert np.issubdtype(value.dtype, np.integer), key

    def test_a_list_produces_one_row_per_text(self, preprocessor):
        out = preprocessor(["a", "bb", "ccc"], return_tensors="np")
        assert out["input_ids"].shape == (3, preprocessor.max_length)

    def test_attention_mask_marks_exactly_the_non_pad_positions(self, preprocessor):
        out = preprocessor("hello", return_tensors="np")
        ids = out["input_ids"][0]
        mask = out["attention_mask"][0]
        np.testing.assert_array_equal(mask, (ids != PAD_ID).astype(mask.dtype))

    def test_token_type_ids_are_all_zero(self, preprocessor):
        out = preprocessor("hello", return_tensors="np")
        assert not out["token_type_ids"].any()

    def test_default_return_tensors_is_numpy(self, preprocessor):
        out = preprocessor("hello")
        assert isinstance(out["input_ids"], np.ndarray)

    def test_tf_return_tensors(self, preprocessor):
        tf = pytest.importorskip("tensorflow")
        out = preprocessor("hello", return_tensors="tf")
        assert isinstance(out["input_ids"], tf.Tensor)
        assert tuple(out["input_ids"].shape) == (1, preprocessor.max_length)

    def test_rejects_an_unknown_return_tensors(self, preprocessor):
        with pytest.raises(ValueError, match="return_tensors"):
            preprocessor("hello", return_tensors="pt")

    def test_rejects_a_non_string_input(self, preprocessor):
        with pytest.raises(TypeError, match="string"):
            preprocessor(42)

    def test_exposes_the_attributes_evaluate_mlm_model_reads(self, preprocessor):
        assert preprocessor.vocab_size == VOCAB_SIZE
        assert preprocessor.pad_token_id == PAD_ID
        assert preprocessor.cls_token_id == CLS_ID
        assert preprocessor.sep_token_id == SEP_ID
        assert preprocessor.mask_token_id == MASK_ID
        assert callable(preprocessor.decode)

    def test_decode_accepts_a_numpy_row(self, preprocessor):
        out = preprocessor("hello", return_tensors="np")
        assert preprocessor.decode(out["input_ids"]) == "hello"

    def test_encode_rejects_a_non_string(self, preprocessor):
        with pytest.raises(TypeError, match="expects a string"):
            preprocessor.encode(["a", "b"])

    def test_truncation_false_raises_on_overlong_input(self):
        strict = ASCIICharPreprocessor(max_length=8, truncation=False)
        with pytest.raises(ValueError, match="truncation=False"):
            strict("x" * 100)

    def test_rejects_an_unknown_padding_strategy(self):
        with pytest.raises(ValueError, match="padding must be"):
            ASCIICharPreprocessor(padding="longest")


class TestTheTwoSurfacesAgree:
    """The Layer and the preprocessor must not drift apart.

    They are separate classes so that ``keras.layers.Layer.__call__`` is not
    shadowed; that split is exactly what creates the risk of divergence, so
    it is asserted directly rather than assumed.
    """

    def test_framed_ids_are_identical(self):
        tokenizer = ASCIICharTokenizer(max_length=24)
        preprocessor = ASCIICharPreprocessor(max_length=24)
        for text in ["hello world", "", "x" * 100, "MiXeD CaSe 42!"]:
            expected = tokenizer.encode_with_framing(text)
            actual = preprocessor(text, return_tensors="np")["input_ids"][0]
            np.testing.assert_array_equal(actual, np.asarray(expected))

    def test_decoded_text_is_identical(self):
        tokenizer = ASCIICharTokenizer(max_length=24)
        preprocessor = ASCIICharPreprocessor(max_length=24)
        text = "shared decode path"
        ids = tokenizer.encode_with_framing(text)
        assert tokenizer.decode(ids) == preprocessor.decode(np.asarray(ids))

    def test_both_report_the_same_vocabulary(self):
        assert ASCIICharTokenizer(max_length=8).vocab_size == (
            ASCIICharPreprocessor(max_length=8).vocab_size
        )
