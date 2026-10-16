"""Guards for the fixed byte-patch transforms.

Same contract as ``test_byte_lm.py``: every expected value is derived BY HAND
from the UTF-8 spec or the patching arithmetic, never by pinning whatever the
code printed. Pure integer plumbing, so all assertions are exact equality.
Each "these agree" assertion has a "these differ" twin.

Hand-derived fixtures used throughout (UTF-8, big-endian code units):
``"Hi!"`` = [72, 105, 33]; ``"ABCDEF"`` = [65..70]; U+00E9 (e-acute) =
[195, 169] (0xC3 0xA9); U+0394 (Greek Delta) = [206, 148] (0xCE 0x94).
"""

import numpy as np
import pytest

from dl_techniques.datasets.byte_patch import (
    DEFAULT_PATCH_SIZE,
    PAD_BYTE_ID,
    bytes_to_patches,
    make_lm_labels,
    pack_byte_patch_windows,
    patches_to_bytes,
)


# ---------------------------------------------------------------------
# bytes_to_patches
# ---------------------------------------------------------------------


class TestBytesToPatches:
    """Fixed blocking with 0x00 tail fill, hand-derived."""

    def test_short_ascii_gets_one_padded_row(self):
        """[72,105,33] at k=4 -> [[72,105,33,0]]: 3 content + 1 pad."""
        out = bytes_to_patches([72, 105, 33], patch_size=4)
        assert out.dtype == np.int32
        assert out.tolist() == [[72, 105, 33, 0]]

    def test_exact_multiple_gains_no_pad_row(self):
        """[65..70] (6 bytes) at k=2 -> 3 full rows, no 0x00 anywhere."""
        out = bytes_to_patches([65, 66, 67, 68, 69, 70], patch_size=2)
        assert out.tolist() == [[65, 66], [67, 68], [69, 70]]

    def test_empty_input_yields_empty_patches(self):
        """[] at k=4 -> shape (0, 4), not a crash and not one pad row."""
        out = bytes_to_patches([], patch_size=4)
        assert out.shape == (0, 4)

    def test_patch_size_one_is_identity_column(self):
        """k=1 never pads: [72,105] -> [[72],[105]]."""
        out = bytes_to_patches([72, 105], patch_size=1)
        assert out.tolist() == [[72], [105]]

    def test_multibyte_split_is_preserved_untouched(self):
        """U+00E9 [195,169] at k=4 pads, never reorders: [[195,169,0,0]]."""
        out = bytes_to_patches([195, 169], patch_size=4)
        assert out.tolist() == [[195, 169, 0, 0]]

    def test_content_change_moves_the_row(self):
        """Differ twin: [72,105,33] vs [72,105,63] ('?') differ in row 0."""
        assert (
            bytes_to_patches([72, 105, 33], patch_size=4).tolist()
            != bytes_to_patches([72, 105, 63], patch_size=4).tolist()
        )

    def test_nonpositive_patch_size_raises(self):
        """k=0 and k=-2 both raise ValueError."""
        with pytest.raises(ValueError):
            bytes_to_patches([72], patch_size=0)
        with pytest.raises(ValueError):
            bytes_to_patches([72], patch_size=-2)

    def test_out_of_range_pad_id_raises(self):
        """pad_id=256 and pad_id=-1 both raise ValueError."""
        with pytest.raises(ValueError):
            bytes_to_patches([72], patch_size=4, pad_id=256)
        with pytest.raises(ValueError):
            bytes_to_patches([72], patch_size=4, pad_id=-1)

    def test_defaults_are_documented_constants(self):
        """DEFAULT_PATCH_SIZE=4, PAD_BYTE_ID=0x00."""
        assert DEFAULT_PATCH_SIZE == 4
        assert PAD_BYTE_ID == 0


# ---------------------------------------------------------------------
# patches_to_bytes
# ---------------------------------------------------------------------


class TestPatchesToBytes:
    """Flatten-only inverse."""

    def test_flatten_is_row_major(self):
        """[[65,66],[67,68]] -> [65,66,67,68]."""
        out = patches_to_bytes([[65, 66], [67, 68]])
        assert out.dtype == np.int32
        assert out.tolist() == [65, 66, 67, 68]

    def test_exact_multiple_round_trips_byte_exact(self):
        """No-pad input survives the round trip with zero residue."""
        ids = [65, 66, 67, 68, 69, 70]
        assert patches_to_bytes(bytes_to_patches(ids, patch_size=2)).tolist() == ids

    def test_padded_round_trip_keeps_the_pads(self):
        """[72,105,33] k=4 -> [[72,105,33,0]] -> [72,105,33,0]: pads kept."""
        out = patches_to_bytes(bytes_to_patches([72, 105, 33], patch_size=4))
        assert out.tolist() == [72, 105, 33, 0]

    def test_split_codepoint_round_trips(self):
        """U+0394 [206,148] k=1 splits nothing, k=3 pads: both invert."""
        assert patches_to_bytes(bytes_to_patches([206, 148], patch_size=1)).tolist() == [206, 148]
        assert patches_to_bytes(bytes_to_patches([206, 148], patch_size=3)).tolist() == [206, 148, 0]

    def test_different_rows_flatten_differently(self):
        """Differ twin: row order is significant."""
        assert patches_to_bytes([[65, 66], [67, 68]]).tolist() != patches_to_bytes(
            [[67, 68], [65, 66]]
        ).tolist()


# ---------------------------------------------------------------------
# pack_byte_patch_windows
# ---------------------------------------------------------------------


class TestPackBytePatchWindows:
    """Concat-and-chunk at patch resolution, hand-derived."""

    def test_two_docs_pack_into_patch_windows(self):
        """Docs [65,66] + [67,68,69,70] = 6 bytes; k=2, ppw=2 (4B windows):
        first window [[65,66],[67,68]], tail [69,70] dropped."""
        out = list(
            pack_byte_patch_windows([[65, 66], [67, 68, 69, 70]], patch_size=2, patches_per_window=2)
        )
        assert len(out) == 1
        assert out[0].tolist() == [[65, 66], [67, 68]]

    def test_kept_tail_is_padded_rectangular(self):
        """Same 6 bytes with drop_remainder=False: tail [69,70] -> [[69,70]],
        shape (1, 2) — rectangular, no ragged rows."""
        out = list(
            pack_byte_patch_windows(
                [[65, 66], [67, 68, 69, 70]],
                patch_size=2,
                patches_per_window=2,
                drop_remainder=False,
            )
        )
        assert len(out) == 2
        assert out[1].tolist() == [[69, 70]]
        assert out[1].shape == (1, 2)

    def test_generator_stream_is_consumed_exactly_once(self):
        """One-shot generator, 10 bytes, k=2 ppw=2 (4B windows): windows hold
        4+4+2 bytes = every source byte exactly once, in order. A
        double-iteration implementation would see an empty stream the second
        time and lose the tail."""

        def gen():
            yield [65, 66, 67]
            yield [68, 69, 70, 71]
            yield [72, 73, 74]

        out = list(pack_byte_patch_windows(gen(), patch_size=2, patches_per_window=2, drop_remainder=False))
        assert [w.shape for w in out] == [(2, 2), (2, 2), (1, 2)]
        assert patches_to_bytes([row for w in out for row in w.tolist()]).tolist() == list(range(65, 75))

    def test_short_tail_pads_to_full_patches(self):
        """5 bytes, k=4, ppw=2 (8B window), keep tail: 1 window [[b0..b3],
        [b4,0,0,0]] — the tail pads to full patches, never ragged."""
        out = list(
            pack_byte_patch_windows([[65, 66, 67, 68, 69]], patch_size=4, patches_per_window=2, drop_remainder=False)
        )
        assert len(out) == 1
        assert out[0].tolist() == [[65, 66, 67, 68], [69, 0, 0, 0]]

    def test_byte_order_changes_window_content(self):
        """Differ twin: doc order [AB][CD] vs [CD][AB] give different first windows."""
        first = list(pack_byte_patch_windows([[65, 66], [67, 68]], patch_size=2, patches_per_window=2))
        second = list(pack_byte_patch_windows([[67, 68], [65, 66]], patch_size=2, patches_per_window=2))
        assert first[0].tolist() != second[0].tolist()

    def test_nonpositive_windows_raises(self):
        """ppw=0 raises; patch_size=0 raises via bytes_to_patches path."""
        with pytest.raises(ValueError):
            list(pack_byte_patch_windows([[65]], patch_size=2, patches_per_window=0))
        with pytest.raises(ValueError):
            list(pack_byte_patch_windows([[65]], patch_size=0, patches_per_window=2))


# ---------------------------------------------------------------------
# make_lm_labels
# ---------------------------------------------------------------------


class TestMakeLmLabels:
    """Prompt + pad masking with ignore_index, hand-derived."""

    def test_prompt_and_pads_masked_answer_kept(self):
        """[65,66,67,68,0], prompt=2 -> [-1,-1,67,68,-1]."""
        out = make_lm_labels([65, 66, 67, 68, 0], num_prompt_bytes=2)
        assert out.dtype == np.int32
        assert out.tolist() == [-1, -1, 67, 68, -1]

    def test_zero_prompt_masks_only_pads(self):
        """prompt=0 leaves content: [65,0] -> [65,-1]."""
        assert make_lm_labels([65, 0], num_prompt_bytes=0).tolist() == [65, -1]

    def test_full_prompt_masks_everything(self):
        """prompt=len -> all -1 (a prompt-only batch scores exactly 0)."""
        assert make_lm_labels([65, 66], num_prompt_bytes=2).tolist() == [-1, -1]

    def test_custom_ignore_index(self):
        """ignore_index=-100 lands in prompt and pad slots alike."""
        assert make_lm_labels([65, 0], num_prompt_bytes=1, ignore_index=-100).tolist() == [-100, -100]

    def test_prompt_length_moves_the_mask(self):
        """Differ twin: prompt=1 vs prompt=2 on [65,66,67] differ at index 1."""
        assert make_lm_labels([65, 66, 67], num_prompt_bytes=1).tolist() != make_lm_labels(
            [65, 66, 67], num_prompt_bytes=2
        ).tolist()

    def test_out_of_range_prompt_raises(self):
        """prompt=4 on 3 bytes, and prompt=-1, both raise ValueError."""
        with pytest.raises(ValueError):
            make_lm_labels([65, 66, 67], num_prompt_bytes=4)
        with pytest.raises(ValueError):
            make_lm_labels([65, 66, 67], num_prompt_bytes=-1)

    def test_nonnegative_ignore_index_raises(self):
        """ignore_index=0 would collide with pad content: refused."""
        with pytest.raises(ValueError):
            make_lm_labels([65], num_prompt_bytes=0, ignore_index=0)
