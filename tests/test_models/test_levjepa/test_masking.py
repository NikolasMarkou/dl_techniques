"""Tests for :mod:`dl_techniques.models.vision.levjepa.masking`.

The block-causal mask's CLS row/column semantic is pinned by an exact,
per-query-position delta-impulse probe (not a shape-only check) — this is
the defect class (`plans/LESSONS.md`'s "guard the DIRECTION, not just the
shape") this repo's own memory flags repeatedly.
"""

import numpy as np
import pytest
import keras

from dl_techniques.models.vision.levjepa.masking import (
    build_block_causal_mask,
    random_token_drop,
)


def _to_numpy(t):
    return keras.ops.convert_to_numpy(t)


# ---------------------------------------------------------------------
# build_block_causal_mask
# ---------------------------------------------------------------------


class TestBuildBlockCausalMask:
    def test_exact_visibility_per_query_with_cls(self):
        """num_frames=3, tokens_per_frame=2, num_prefix_tokens=1.

        Grid layout (flat index -> (frame, slot)):
            0: CLS
            1,2: frame 0
            3,4: frame 1
            5,6: frame 2
        """
        mask = build_block_causal_mask(
            num_frames=3, tokens_per_frame=2, num_prefix_tokens=1
        )
        m = _to_numpy(mask)
        assert m.dtype == np.bool_
        assert m.shape == (1, 1, 7, 7)
        m = m[0, 0]

        # Query = CLS (index 0): sees everything.
        np.testing.assert_array_equal(m[0], np.ones(7, dtype=bool))

        # Query = frame-0 patch (index 1): sees only frame-0 patches
        # (indices 1, 2), never CLS, never frame-1/2.
        expected_row1 = np.zeros(7, dtype=bool)
        expected_row1[[1, 2]] = True
        np.testing.assert_array_equal(m[1], expected_row1)

        # Query = the other frame-0 patch (index 2): symmetric to index 1.
        expected_row2 = np.zeros(7, dtype=bool)
        expected_row2[[1, 2]] = True
        np.testing.assert_array_equal(m[2], expected_row2)

        # Query = frame-1 patch (index 3): sees frame-0 and frame-1
        # patches, not CLS, not frame-2.
        expected_row3 = np.zeros(7, dtype=bool)
        expected_row3[[1, 2, 3, 4]] = True
        np.testing.assert_array_equal(m[3], expected_row3)

        # Query = frame-2 patch (index 5): sees frame-0, frame-1, and
        # frame-2 patches, not CLS.
        expected_row5 = np.zeros(7, dtype=bool)
        expected_row5[[1, 2, 3, 4, 5, 6]] = True
        np.testing.assert_array_equal(m[5], expected_row5)
        expected_row6 = expected_row5.copy()
        np.testing.assert_array_equal(m[6], expected_row6)

        # CLS's own column: every patch-query's visibility of the CLS key
        # is entirely False (patches never attend to CLS).
        cls_column_for_patches = m[1:, 0]
        np.testing.assert_array_equal(
            cls_column_for_patches, np.zeros(6, dtype=bool)
        )

    def test_token_ids_path_uses_true_grid_position(self):
        """An out-of-order/dropped `token_ids` array must be resolved by
        its TRUE grid position, not by its position in the (shortened)
        sequence."""
        # True grid indices kept, in a shuffled sequence order:
        # sequence position 0 -> true index 5 (frame 2, tokens_per_frame=2)
        # sequence position 1 -> true index 1 (frame 0)
        # sequence position 2 -> true index 3 (frame 1)
        token_ids = keras.ops.convert_to_tensor([[5, 1, 3]], dtype="int32")
        mask = build_block_causal_mask(
            num_frames=3,
            tokens_per_frame=2,
            token_ids=token_ids,
            num_prefix_tokens=1,
        )
        m = _to_numpy(mask)
        assert m.shape == (1, 1, 4, 4)
        m = m[0, 0]

        # CLS row: all True.
        np.testing.assert_array_equal(m[0], np.ones(4, dtype=bool))
        # CLS column for patch queries: all False.
        np.testing.assert_array_equal(m[1:, 0], np.zeros(3, dtype=bool))

        # Patch sub-block, keyed by TRUE frame ids [2, 0, 1] for sequence
        # positions [0, 1, 2] respectively (frame_ids = token_ids // 2).
        expected_patch = np.array(
            [
                [True, True, True],  # frame2 query sees frame0,1,2 keys
                [False, True, False],  # frame0 query sees only frame0 key
                [False, True, True],  # frame1 query sees frame0,1 keys
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(m[1:, 1:], expected_patch)

    def test_num_prefix_tokens_zero_is_pure_patch_causal_grid(self):
        mask = build_block_causal_mask(
            num_frames=2, tokens_per_frame=3, num_prefix_tokens=0
        )
        m = _to_numpy(mask)
        assert m.shape == (1, 1, 6, 6)
        m = m[0, 0]

        frame_ids = np.array([0, 0, 0, 1, 1, 1])
        expected = frame_ids[:, None] >= frame_ids[None, :]
        np.testing.assert_array_equal(m, expected)

    def test_batch_size_tiles_the_default_grid(self):
        mask = build_block_causal_mask(
            num_frames=2,
            tokens_per_frame=2,
            num_prefix_tokens=1,
            batch_size=3,
        )
        m = _to_numpy(mask)
        assert m.shape == (3, 1, 5, 5)
        # All batch slices identical (no per-sample variation without
        # token_ids).
        np.testing.assert_array_equal(m[0], m[1])
        np.testing.assert_array_equal(m[0], m[2])


# ---------------------------------------------------------------------
# random_token_drop
# ---------------------------------------------------------------------


class TestRandomTokenDrop:
    def test_drop_rate_zero_is_true_identity(self):
        x = keras.random.normal((2, 10, 4), seed=0)
        out, token_ids = random_token_drop(x, drop_rate=0.0, training=True)
        assert out is x
        assert token_ids is None
        np.testing.assert_array_equal(_to_numpy(out), _to_numpy(x))

    def test_training_false_is_true_identity(self):
        x = keras.random.normal((2, 10, 4), seed=0)
        out, token_ids = random_token_drop(x, drop_rate=0.9, training=False)
        assert out is x
        assert token_ids is None

    def test_shape_and_keep_len_math(self):
        x = keras.random.normal((4, 40, 8), seed=1)
        out, token_ids = random_token_drop(
            x, drop_rate=0.95, training=True, seed=42
        )
        expected_keep_len = max(1, round(40 * (1 - 0.95)))
        assert expected_keep_len == 2
        out_np = _to_numpy(out)
        ids_np = _to_numpy(token_ids)
        assert out_np.shape == (4, expected_keep_len, 8)
        assert ids_np.shape == (4, expected_keep_len)
        assert ids_np.min() >= 0
        assert ids_np.max() < 40
        for row in ids_np:
            assert len(np.unique(row)) == len(row)

    def test_integration_with_build_block_causal_mask(self):
        """A kept token's mask visibility must reflect its TRUE (pre-drop)
        frame membership, not its post-drop sequence index."""
        num_frames, tokens_per_frame = 4, 3
        n = num_frames * tokens_per_frame
        x = keras.random.normal((1, n, 5), seed=7)
        dropped_x, token_ids = random_token_drop(
            x, drop_rate=0.5, training=True, seed=123
        )
        mask = build_block_causal_mask(
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            token_ids=token_ids,
            num_prefix_tokens=1,
        )
        m = _to_numpy(mask)[0, 0]
        ids_np = _to_numpy(token_ids)[0]
        true_frame_ids = ids_np // tokens_per_frame

        # Patch sub-block (drop CLS row/col at index 0).
        patch_block = m[1:, 1:]
        expected = true_frame_ids[:, None] >= true_frame_ids[None, :]
        np.testing.assert_array_equal(patch_block, expected)
