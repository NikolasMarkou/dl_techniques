"""Guards for the byte-level packed-CLM data primitives.

Every expected value in this file is derived BY HAND from the UTF-8 spec or
from the packing arithmetic, never by calling the code under test and pinning
whatever it printed. The four hand-encoded characters are re-derived in the
docstring of the test that uses them so a reader can check the arithmetic
without a terminal.

The module is pure integer plumbing -- there is not a single float
comparison, so every assertion here is exact equality rather than a
tolerance. Each "these agree" assertion has a "these differ" twin, because an
encoder that returned a constant, a packer that emitted nothing, and an
estimator that ignored its inputs would all satisfy the agreement half alone.
"""

import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.datasets.byte_lm import (
    BOS_ID,
    BYTE_VOCAB_SIZE,
    DEFAULT_AVG_BYTES_PER_ARTICLE,
    DEFAULT_DATASET_ROOT,
    DEFAULT_WIKIPEDIA_TOTAL_BYTES,
    EOS_ID,
    build_byte_clm_dataset,
    byte_ids_to_text,
    estimate_byte_clm_steps_per_epoch,
    pack_byte_windows,
    text_to_byte_ids,
)


# ---------------------------------------------------------------------
# text_to_byte_ids
# ---------------------------------------------------------------------


class TestTextToByteIds:
    """UTF-8 encoding, hand-derived."""

    def test_ascii_encodes_to_its_code_points(self):
        """``"Hi!"`` is U+0048 U+0069 U+0021, single-byte each: 72, 105, 33."""
        ids = text_to_byte_ids("Hi!")
        assert ids.dtype == np.uint8
        assert ids.tolist() == [72, 105, 33]

    def test_a_two_byte_character_encodes_to_two_ids(self):
        """U+00E9 (e-acute).

        11 significant bits ``000 1110 1001`` split as ``00011`` / ``101001``
        into the 2-byte form ``110xxxxx 10xxxxxx``:
        ``110 00011`` = 0xC3 = 195, ``10 101001`` = 0xA9 = 169.
        """
        assert text_to_byte_ids("é").tolist() == [195, 169]

    def test_a_three_byte_character_encodes_to_three_ids(self):
        """U+20AC (euro sign).

        16 bits ``0010 0000 1010 1100`` split ``0010`` / ``000010`` /
        ``101100`` into ``1110xxxx 10xxxxxx 10xxxxxx``:
        0xE2 = 226, 0x82 = 130, 0xAC = 172.
        """
        assert text_to_byte_ids("€").tolist() == [226, 130, 172]

    def test_a_four_byte_character_encodes_to_four_ids(self):
        """U+1F600 (grinning face).

        21 bits ``000 011111 011000 000000`` into
        ``11110xxx 10xxxxxx 10xxxxxx 10xxxxxx``:
        0xF0 = 240, 0x9F = 159, 0x98 = 152, 0x80 = 128.
        """
        assert text_to_byte_ids("\U0001f600").tolist() == [240, 159, 152, 128]

    def test_utf8_and_latin1_agree_on_ascii_and_differ_on_a_non_ascii_char(self):
        """The anti-vacuity twin for the encoding CHOICE.

        ``ord()``/latin-1 is correct for every ASCII character, so an ASCII
        corpus cannot tell the two apart. U+00E9 is where they separate:
        latin-1 gives the single id 233, UTF-8 gives 195, 169.
        """
        assert text_to_byte_ids("Hi!").tolist() == list(b"Hi!".decode("latin-1").encode("latin-1"))
        assert text_to_byte_ids("é").tolist() == [195, 169]
        assert text_to_byte_ids("é").tolist() != [233]
        assert len(text_to_byte_ids("é")) == 2 != len("é")

    def test_the_empty_string_yields_an_empty_array_and_does_not_raise(self):
        ids = text_to_byte_ids("")
        assert ids.dtype == np.uint8
        assert ids.shape == (0,)

    def test_utf8_bytes_are_passed_through_unchanged(self):
        """tf.data yields ``bytes`` for a string tensor; re-encoding would
        mojibake them."""
        assert text_to_byte_ids("é".encode("utf-8")).tolist() == [195, 169]

    def test_bos_and_eos_wrap_the_payload_in_that_order(self):
        assert text_to_byte_ids("A", add_bos=True, add_eos=True).tolist() == [
            BOS_ID,
            65,
            EOS_ID,
        ]
        assert text_to_byte_ids("A", add_bos=True).tolist() == [BOS_ID, 65]
        assert text_to_byte_ids("A", add_eos=True).tolist() == [65, EOS_ID]

    def test_the_markers_are_the_two_highest_byte_values_and_are_distinct(self):
        """S4: 254/255 are a CONVENTION -- the two highest byte values -- not
        a mechanism. If they ever swap, every packed stream's document
        separator silently changes identity."""
        assert BOS_ID == BYTE_VOCAB_SIZE - 2 == 254
        assert EOS_ID == BYTE_VOCAB_SIZE - 1 == 255
        assert BOS_ID != EOS_ID
        assert BOS_ID < EOS_ID < BYTE_VOCAB_SIZE

    def test_the_returned_array_is_writable(self):
        ids = text_to_byte_ids("A")
        ids[0] = 7  # must not raise on a read-only frombuffer alias
        assert ids.tolist() == [7]


# ---------------------------------------------------------------------
# byte_ids_to_text / round trip
# ---------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.parametrize(
        "text",
        ["", "Hello, world!", "café", "€ 5", "\U0001f600 ok", "中文"],
        ids=["empty", "ascii", "two-byte", "three-byte", "four-byte", "cjk"],
    )
    def test_text_to_bytes_to_text_is_the_identity(self, text):
        assert byte_ids_to_text(text_to_byte_ids(text)) == text

    def test_two_different_strings_do_not_round_trip_to_each_other(self):
        """The twin: identity would also hold for a decoder that ignored its
        input and returned the constant it was last given."""
        a = byte_ids_to_text(text_to_byte_ids("café"))
        b = byte_ids_to_text(text_to_byte_ids("cafe"))
        assert a != b

    def test_a_truncated_codepoint_decodes_to_a_replacement_rather_than_raising(self):
        """``"aé!"`` is bytes 97, 195, 169, 33. Cutting after the 195
        leaves a dangling UTF-8 lead byte."""
        assert byte_ids_to_text([97, 195]) == "a�"
        assert byte_ids_to_text([169, 33]) == "�!"

    def test_strict_decoding_still_raises_on_the_same_input(self):
        """The twin proving ``errors='replace'`` is doing real work rather
        than the input being decodable anyway."""
        with pytest.raises(UnicodeDecodeError):
            byte_ids_to_text([97, 195], errors="strict")


# ---------------------------------------------------------------------
# pack_byte_windows
# ---------------------------------------------------------------------


class TestPackByteWindows:
    def test_the_concat_and_chunk_shape_is_hand_derived(self):
        """Documents [1,2,3] and [4,5,6,7] at seq_len=3.

        Buffer walk: [1,2,3] -> emit [1,2,3], buffer empty; extend to
        [4,5,6,7] -> emit [4,5,6], buffer [7]. Two full windows, remainder
        [7].
        """
        out = list(pack_byte_windows([[1, 2, 3], [4, 5, 6, 7]], seq_len=3))
        assert [w.tolist() for w in out] == [[1, 2, 3], [4, 5, 6]]
        assert all(w.dtype == np.uint8 for w in out)

    def test_the_short_final_chunk_is_kept_only_when_drop_remainder_is_false(self):
        kept = list(
            pack_byte_windows([[1, 2, 3], [4, 5, 6, 7]], seq_len=3, drop_remainder=False)
        )
        assert [w.tolist() for w in kept] == [[1, 2, 3], [4, 5, 6], [7]]
        assert len(kept[-1]) == 1 < 3

        dropped = list(
            pack_byte_windows([[1, 2, 3], [4, 5, 6, 7]], seq_len=3, drop_remainder=True)
        )
        assert [w.tolist() for w in dropped] == [[1, 2, 3], [4, 5, 6]]

    def test_an_exactly_divisible_stream_emits_no_remainder_under_either_flag(self):
        """The twin for the test above: when nothing is left over, the two
        flags must AGREE, so a packer that always kept or always dropped is
        invisible here and visible there."""
        for drop in (True, False):
            out = list(pack_byte_windows([[1, 2], [3, 4]], seq_len=2, drop_remainder=drop))
            assert [w.tolist() for w in out] == [[1, 2], [3, 4]]

    def test_a_window_spans_a_document_boundary(self):
        """Packing is document-agnostic: [1,2] then [3,4] at seq_len=3 gives
        [1,2,3], which no per-document pipeline could produce."""
        out = list(pack_byte_windows([[1, 2], [3, 4]], seq_len=3, drop_remainder=False))
        assert [w.tolist() for w in out] == [[1, 2, 3], [4]]

    def test_the_stride_is_seq_len_so_no_byte_is_emitted_twice(self):
        """An off-by-one stride re-emits the boundary byte. Concatenating all
        windows must reproduce the input stream exactly, in order."""
        docs = [[1, 2, 3, 4], [5, 6, 7], [8]]
        flat = [b for d in docs for b in d]
        out = list(pack_byte_windows(docs, seq_len=3, drop_remainder=False))
        assert np.concatenate(out).tolist() == flat
        assert sum(len(w) for w in out) == len(flat) == 8

    def test_a_different_seq_len_gives_different_windows(self):
        """The twin: a packer that ignored seq_len would pass every
        assertion above that uses a single seq_len."""
        docs = [[1, 2, 3, 4, 5, 6]]
        two = [w.tolist() for w in pack_byte_windows(docs, seq_len=2)]
        three = [w.tolist() for w in pack_byte_windows(docs, seq_len=3)]
        assert two == [[1, 2], [3, 4], [5, 6]]
        assert three == [[1, 2, 3], [4, 5, 6]]
        assert two != three

    def test_a_multi_byte_character_may_straddle_a_window_boundary(self):
        """``"aé!"`` -> bytes [97, 195, 169, 33]. At seq_len=2 the
        2-byte U+00E9 is split across windows 0 and 1.

        This is CORRECT for a byte model and is exactly where a byte
        pipeline silently corrupts text: neither window decodes to valid
        UTF-8 on its own, but their concatenation does.
        """
        ids = text_to_byte_ids("aé!")
        assert ids.tolist() == [97, 195, 169, 33]

        windows = list(pack_byte_windows([ids], seq_len=2, drop_remainder=False))
        assert [w.tolist() for w in windows] == [[97, 195], [169, 33]]

        assert byte_ids_to_text(windows[0]) == "a�"
        assert byte_ids_to_text(windows[1]) == "�!"
        assert byte_ids_to_text(np.concatenate(windows)) == "aé!"

    def test_an_empty_stream_and_empty_documents_yield_nothing(self):
        assert list(pack_byte_windows([], seq_len=3)) == []
        assert list(pack_byte_windows([[], []], seq_len=3, drop_remainder=False)) == []

    def test_a_non_positive_seq_len_raises(self):
        with pytest.raises(ValueError, match="seq_len must be >= 1"):
            list(pack_byte_windows([[1, 2]], seq_len=0))


# ---------------------------------------------------------------------
# build_byte_clm_dataset
# ---------------------------------------------------------------------


def _text_ds(texts):
    return tf.data.Dataset.from_tensor_slices(tf.constant(texts, dtype=tf.string))


class TestBuildByteClmDataset:
    def test_the_pairs_are_hand_derived_end_to_end(self):
        """Corpus ["ab", "cd"], seq_len=3, batch=2.

        Bytes: 97,98 + EOS(255) + 99,100 + EOS(255) = [97,98,255,99,100,255].
        Windows of 3: [97,98,255] and [99,100,255].
        Causal shift: inputs [97,98] / [99,100], labels [98,255] / [100,255].
        One full batch of 2.
        """
        ds = build_byte_clm_dataset(
            _text_ds(["ab", "cd"]), seq_len=3, batch_size=2, shuffle_buffer=1
        )
        batches = list(ds.as_numpy_iterator())
        assert len(batches) == 1
        inputs, labels = batches[0]
        assert inputs.dtype == np.int32 and labels.dtype == np.int32
        assert inputs.shape == (2, 2) and labels.shape == (2, 2)
        assert inputs.tolist() == [[97, 98], [99, 100]]
        assert labels.tolist() == [[98, 255], [100, 255]]

    def test_labels_are_the_inputs_shifted_left_by_one_not_right(self):
        """The twin that separates the shift from its mirror image: with a
        strictly increasing byte stream, ``labels[i] == inputs[i] + 1``
        holds one way round and fails the other."""
        text = bytes(range(65, 85)).decode("ascii")
        ds = build_byte_clm_dataset(
            _text_ds([text]), seq_len=5, batch_size=1, shuffle_buffer=1
        )
        inputs, labels = next(iter(ds.as_numpy_iterator()))
        assert inputs.tolist() == [[65, 66, 67, 68]]
        assert labels.tolist() == [[66, 67, 68, 69]]
        assert labels.tolist() != inputs.tolist()

    def test_the_eos_separator_is_present_between_documents(self):
        """Without the separator the byte stream of ["ab","cd"] would be
        4 bytes and could not fill two 3-byte windows at all."""
        ds = build_byte_clm_dataset(
            _text_ds(["ab", "cd"]), seq_len=3, batch_size=1, shuffle_buffer=1
        )
        flat = np.concatenate(
            [np.concatenate([i[0], l[0][-1:]]) for i, l in ds.as_numpy_iterator()]
        )
        assert flat.tolist() == [97, 98, 255, 99, 100, 255]
        assert EOS_ID in flat.tolist()

    def test_a_custom_eos_id_is_honoured(self):
        """The twin for the test above: an appender that hard-coded 255
        would pass it and fail this."""
        ds = build_byte_clm_dataset(
            _text_ds(["ab"]), seq_len=3, batch_size=1, shuffle_buffer=1, eos_id=7
        )
        inputs, labels = next(iter(ds.as_numpy_iterator()))
        assert inputs.tolist() == [[97, 98]]
        assert labels.tolist() == [[98, 7]]

    def test_the_trailing_partial_window_is_dropped(self):
        """["abcd"] -> [97,98,99,100,255], 5 bytes. At seq_len=3 exactly one
        window is full; the trailing [100,255] is dropped."""
        ds = build_byte_clm_dataset(
            _text_ds(["abcd"]), seq_len=3, batch_size=1, shuffle_buffer=1
        )
        batches = list(ds.as_numpy_iterator())
        assert len(batches) == 1
        assert batches[0][0].tolist() == [[97, 98]]

    def test_an_incomplete_batch_is_dropped(self):
        """Two windows at batch_size=3 yields zero batches, never a ragged
        one."""
        ds = build_byte_clm_dataset(
            _text_ds(["ab", "cd"]), seq_len=3, batch_size=3, shuffle_buffer=1
        )
        assert list(ds.as_numpy_iterator()) == []

    def test_a_multi_byte_document_produces_byte_ids_not_character_ids(self):
        """"café" is 4 characters but 5 bytes: 99,97,102,195,169."""
        ds = build_byte_clm_dataset(
            _text_ds(["café"]), seq_len=6, batch_size=1, shuffle_buffer=1
        )
        inputs, labels = next(iter(ds.as_numpy_iterator()))
        assert inputs.tolist() == [[99, 97, 102, 195, 169]]
        assert labels.tolist() == [[97, 102, 195, 169, 255]]

    def test_repeat_makes_the_dataset_unbounded(self):
        ds = build_byte_clm_dataset(
            _text_ds(["ab", "cd"]), seq_len=3, batch_size=1, shuffle_buffer=1, repeat=True
        )
        it = ds.as_numpy_iterator()
        taken = [next(it)[0].tolist() for _ in range(5)]
        assert len(taken) == 5
        assert taken[0] == taken[2] == taken[4] == [[97, 98]]

    def test_without_repeat_the_dataset_is_finite(self):
        """The twin for the test above."""
        ds = build_byte_clm_dataset(
            _text_ds(["ab", "cd"]), seq_len=3, batch_size=1, shuffle_buffer=1
        )
        assert len(list(ds.as_numpy_iterator())) == 2

    def test_a_seq_len_below_two_raises(self):
        with pytest.raises(ValueError, match="seq_len must be >= 2"):
            build_byte_clm_dataset(_text_ds(["ab"]), seq_len=1, batch_size=1)


# ---------------------------------------------------------------------
# estimate_byte_clm_steps_per_epoch
# ---------------------------------------------------------------------


class TestEstimateByteClmStepsPerEpoch:
    def test_the_estimate_is_hand_computed(self):
        """10 articles x 100 bytes = 1000 bytes; 1000 // 8 = 125 windows;
        125 // 4 = 31 steps."""
        assert (
            estimate_byte_clm_steps_per_epoch(
                num_articles=10, seq_len=8, batch_size=4, avg_bytes_per_article=100
            )
            == 31
        )

    def test_the_floor_divisions_do_not_round_up(self):
        """7 articles x 10 = 70 bytes; 70 // 8 = 8 windows; 8 // 3 = 2."""
        assert (
            estimate_byte_clm_steps_per_epoch(
                num_articles=7, seq_len=8, batch_size=3, avg_bytes_per_article=10
            )
            == 2
        )

    def test_the_override_short_circuits_the_estimate(self):
        assert (
            estimate_byte_clm_steps_per_epoch(
                num_articles=10,
                seq_len=8,
                batch_size=4,
                override=7,
                avg_bytes_per_article=100,
            )
            == 7
        )

    def test_the_override_differs_from_the_estimate_it_replaces(self):
        """The twin: an override equal to the estimate could not tell an
        honoured override from an ignored one."""
        estimated = estimate_byte_clm_steps_per_epoch(
            num_articles=10, seq_len=8, batch_size=4, avg_bytes_per_article=100
        )
        assert estimated == 31 != 7

    def test_a_zero_or_negative_override_is_clamped_to_one(self):
        assert estimate_byte_clm_steps_per_epoch(10, 8, 4, override=0) == 1
        assert estimate_byte_clm_steps_per_epoch(10, 8, 4, override=-5) == 1

    def test_none_articles_falls_back_to_the_whole_corpus_byte_total(self):
        expected = DEFAULT_WIKIPEDIA_TOTAL_BYTES // 1024 // 8
        assert (
            estimate_byte_clm_steps_per_epoch(None, seq_len=1024, batch_size=8)
            == expected
        )
        assert expected > 1

    def test_the_fallback_differs_from_the_per_article_estimate(self):
        """The twin: a fallback that reused the article path would be
        invisible without a second, differently-valued arm."""
        fallback = estimate_byte_clm_steps_per_epoch(None, seq_len=1024, batch_size=8)
        counted = estimate_byte_clm_steps_per_epoch(
            num_articles=1000, seq_len=1024, batch_size=8
        )
        assert fallback != counted

    def test_the_result_is_never_below_one(self):
        assert (
            estimate_byte_clm_steps_per_epoch(
                num_articles=1, seq_len=1_000_000, batch_size=1024
            )
            == 1
        )

    def test_a_larger_average_yields_more_steps(self):
        small = estimate_byte_clm_steps_per_epoch(
            1000, 512, 8, avg_bytes_per_article=100
        )
        large = estimate_byte_clm_steps_per_epoch(
            1000, 512, 8, avg_bytes_per_article=1000
        )
        assert large > small

    def test_the_default_average_is_the_measured_corpus_ratio(self):
        """3053.71 = 19_567_594_259 / 6_407_814, measured over all 41 staged
        Wikipedia Arrow shards, rounded to the nearest integer."""
        assert DEFAULT_AVG_BYTES_PER_ARTICLE == 3054
        assert round(DEFAULT_WIKIPEDIA_TOTAL_BYTES / 6_407_814) == 3054


# ---------------------------------------------------------------------
# Plumbing conventions
# ---------------------------------------------------------------------


class TestPlumbingConventions:
    def test_the_dataset_root_is_the_shared_constant_not_a_second_copy(self):
        from dl_techniques.datasets.nlp import DEFAULT_WIKIPEDIA_CACHE_DIR

        assert DEFAULT_DATASET_ROOT == DEFAULT_WIKIPEDIA_CACHE_DIR
        assert DEFAULT_DATASET_ROOT.startswith("/media/arxwn/data0_4tb")

    def test_the_module_reads_no_environment_variable(self):
        """S2: the corpus root is a default constant plus a CLI flag, never
        an env var. A source assertion because there is no value instrument
        that can see an env read that never fires in the test process."""
        import inspect

        from dl_techniques.datasets import byte_lm

        source = inspect.getsource(byte_lm)
        assert "os.environ" not in source
        assert "getenv" not in source
