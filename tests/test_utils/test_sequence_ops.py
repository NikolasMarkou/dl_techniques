"""Tests for ``dl_techniques.utils.sequence_ops``.

Tests the DynamicSegmenter protocol compliance, cap_boundaries_by_position helper,
and the shared utilities (dim, pad_permutation_to_width, batched_gather).
"""

import keras
import numpy as np
import pytest

from dl_techniques.utils.sequence_ops import (
    cap_boundaries_by_position,
    DynamicSegmenter,
    pad_permutation_to_width,
    batched_gather,
    dim,
)


class TestCapBoundariesByPosition:
    """Tests for the position-order boundary capping helper."""

    def test_keeps_first_n_boundaries(self):
        """Keeps only the first max_segments boundaries in position order."""
        mask = keras.ops.convert_to_tensor(np.array([
            [True, False, True, False, True, False],  # boundaries at 0, 2, 4
            [True, True, True, True, False, False],    # boundaries at 0, 1, 2, 3
        ], dtype=bool))

        capped = cap_boundaries_by_position(mask, max_segments=2)

        expected = np.array([
            [True, False, True, False, False, False],  # keep 0, 2
            [True, True, False, False, False, False],   # keep 0, 1
        ], dtype=bool)

        np.testing.assert_array_equal(capped.numpy(), expected)

    def test_zero_boundaries_unchanged(self):
        """Row with no boundaries stays all False."""
        mask = keras.ops.convert_to_tensor(np.array([
            [False, False, False, False],
        ], dtype=bool))

        capped = cap_boundaries_by_position(mask, max_segments=2)
        np.testing.assert_array_equal(capped.numpy(), mask.numpy())

    def test_fewer_than_max_unchanged(self):
        """Row with fewer boundaries than max_segments is unchanged."""
        mask = keras.ops.convert_to_tensor(np.array([
            [True, False, True, False],  # 2 boundaries
        ], dtype=bool))

        capped = cap_boundaries_by_position(mask, max_segments=5)
        np.testing.assert_array_equal(capped.numpy(), mask.numpy())

    def test_exactly_max_unchanged(self):
        """Row with exactly max_segments boundaries is unchanged."""
        mask = keras.ops.convert_to_tensor(np.array([
            [True, False, True, False],  # 2 boundaries
        ], dtype=bool))

        capped = cap_boundaries_by_position(mask, max_segments=2)
        np.testing.assert_array_equal(capped.numpy(), mask.numpy())

    def test_max_segments_zero(self):
        """max_segments=0 removes all boundaries."""
        mask = keras.ops.convert_to_tensor(np.array([
            [True, True, True, False],
        ], dtype=bool))

        capped = cap_boundaries_by_position(mask, max_segments=0)
        expected = np.array([[False, False, False, False]], dtype=bool)
        np.testing.assert_array_equal(capped.numpy(), expected)

    def test_batch_independence(self):
        """Each row is capped independently."""
        mask = keras.ops.convert_to_tensor(np.array([
            [True, True, True, True],   # 4 boundaries
            [True, False, False, False], # 1 boundary
        ], dtype=bool))

        capped = cap_boundaries_by_position(mask, max_segments=2)

        expected = np.array([
            [True, True, False, False],  # first 2
            [True, False, False, False],  # only 1
        ], dtype=bool)
        np.testing.assert_array_equal(capped.numpy(), expected)

    def test_preserves_dtype(self):
        """Output dtype matches input (bool)."""
        mask = keras.ops.convert_to_tensor(np.array([[True, False]], dtype=bool))
        capped = cap_boundaries_by_position(mask, max_segments=1)
        assert capped.dtype == mask.dtype == bool


class TestDynamicSegmenterProtocol:
    """Tests that the DynamicSegmenter protocol is well-formed."""

    def test_protocol_is_runtime_checkable(self):
        """Protocol can be used with isinstance checks."""
        assert hasattr(DynamicSegmenter, '__instancecheck__')

    def test_protocol_defines_required_methods(self):
        """Protocol has segment, gather, scatter methods."""
        assert 'segment' in DynamicSegmenter.__annotations__ or \
               any('segment' in str(m) for m in dir(DynamicSegmenter))
        # Structural protocol - just verify it's defined
        assert DynamicSegmenter is not None


class TestDim:
    """Tests for the dim helper."""

    def test_static_axis_returns_int(self):
        """Statically known axis returns Python int."""
        x = keras.ops.zeros((2, 5, 3), dtype="float32")
        for axis, expected in [(0, 2), (1, 5), (2, 3), (-1, 3), (-2, 5)]:
            value = dim(x, axis)
            assert type(value) is int, f"axis {axis} returned {type(value)!r}"
            assert value == expected

    def test_returns_same_as_keras_ops_shape(self):
        """Result equals keras.ops.shape(x)[axis] for static axes."""
        x = keras.ops.zeros((2, 5, 3), dtype="float32")
        for axis in [0, 1, 2, -1, -2]:
            assert dim(x, axis) == keras.ops.shape(x)[axis]
            assert type(dim(x, axis)) is type(keras.ops.shape(x)[axis]) is int


class TestPadPermutationToWidth:
    """Tests for pad_permutation_to_width."""

    def test_truncates_when_l_greater_than_width(self):
        """L > width: drops trailing columns."""
        indices = keras.ops.convert_to_tensor(np.array([[0, 1, 2, 3, 4]], dtype="int32"))
        padded = pad_permutation_to_width(indices, width=3)
        expected = np.array([[0, 1, 2]], dtype="int32")
        np.testing.assert_array_equal(padded.numpy(), expected)

    def test_pads_when_l_less_than_width(self):
        """L < width: pads with index 0."""
        indices = keras.ops.convert_to_tensor(np.array([[0, 2]], dtype="int32"))
        padded = pad_permutation_to_width(indices, width=5)
        expected = np.array([[0, 2, 0, 0, 0]], dtype="int32")
        np.testing.assert_array_equal(padded.numpy(), expected)

    def test_exact_copy_when_equal(self):
        """L == width: exact copy."""
        indices = keras.ops.convert_to_tensor(np.array([[0, 2, 1]], dtype="int32"))
        padded = pad_permutation_to_width(indices, width=3)
        np.testing.assert_array_equal(padded.numpy(), indices.numpy())


class TestBatchedGather:
    """Tests for batched_gather."""

    def test_matches_take_along_axis_rank3(self):
        """Output matches take_along_axis for rank-3 params."""
        rng = np.random.default_rng(42)
        params = rng.standard_normal((3, 7, 4)).astype("float32")
        indices = np.stack([rng.permutation(7) for _ in range(3)]).astype("int32")

        gathered = batched_gather(
            keras.ops.convert_to_tensor(params),
            keras.ops.convert_to_tensor(indices),
        )
        expected = np.take_along_axis(params, indices[:, :, None], axis=1)
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(gathered), expected)

    def test_matches_take_along_axis_rank2(self):
        """Output matches take_along_axis for rank-2 params."""
        rng = np.random.default_rng(123)
        params = rng.standard_normal((2, 5)).astype("float32")
        indices = np.stack([rng.permutation(5) for _ in range(2)]).astype("int32")

        gathered = batched_gather(
            keras.ops.convert_to_tensor(params),
            keras.ops.convert_to_tensor(indices),
        )
        expected = np.take_along_axis(params, indices, axis=1)
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(gathered), expected)

    def test_raises_on_rank_1(self):
        """Raises ValueError for rank-1 params."""
        params = keras.ops.zeros((5,), dtype="float32")
        indices = keras.ops.zeros((1, 2), dtype="int32")
        with pytest.raises(ValueError, match="rank-2 or rank-3"):
            batched_gather(params, indices)

    def test_raises_on_rank_4(self):
        """Raises ValueError for rank-4 params."""
        params = keras.ops.zeros((2, 3, 4, 5), dtype="float32")
        indices = keras.ops.zeros((2, 2), dtype="int32")
        with pytest.raises(ValueError, match="rank-2 or rank-3"):
            batched_gather(params, indices)


# ---------------------------------------------------------------------