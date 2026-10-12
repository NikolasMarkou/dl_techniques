"""MPSLayer contracts the last axis at any rank, in the input dtype.

`build()` admitted any rank >= 2 and `compute_output_shape` promised N-D,
but `call()` indexed `inputs[:, i]` and read `shape[0]/shape[1]` -- a rank-3
input died in Einsum. The chain now flattens leading dims through the
contraction and restores them, and the boundary vector takes the input
dtype (it defaulted to float32, returning float32 under mixed_float16).
"""

import numpy as np
import pytest

from dl_techniques.layers.structured_linear.mps_layer import MPSLayer


def test_rank_three_and_rank_four_forward_shapes():
    layer = MPSLayer(output_dim=4)
    layer.build((None, 3, 6))
    assert tuple(layer(np.ones((2, 3, 6), dtype=np.float32)).shape) == (2, 3, 4)
    assert tuple(layer(np.ones((2, 5, 3, 6), dtype=np.float32)).shape) == (2, 5, 3, 4)


def test_leading_dims_are_independent_through_the_chain():
    layer = MPSLayer(output_dim=4, bond_dim=4)
    layer.build((None, 3, 6))
    rng = np.random.RandomState(0)
    x = rng.randn(2, 3, 6).astype(np.float32)
    joint = np.asarray(layer(x))
    for i in range(2):
        for j in range(3):
            single = np.asarray(layer(x[i : i + 1, j : j + 1, :]))
            # Batched GPU reductions reassociate with batch size (measured
            # worst diff 7.4e-06 across draws), so this is tight but not exact.
            np.testing.assert_allclose(joint[i, j], single[0, 0], rtol=0, atol=1e-4)


def test_rank_two_output_is_unchanged_by_the_generalisation():
    layer = MPSLayer(output_dim=4, bond_dim=4)
    x = np.random.RandomState(1).randn(3, 6).astype(np.float32)
    out = np.asarray(layer(x))
    assert out.shape == (3, 4)
    assert np.all(np.isfinite(out))


def test_a_wrong_last_dimension_raises():
    layer = MPSLayer(output_dim=4)
    layer.build((None, 6))
    with pytest.raises(ValueError, match="built for input_dim=6"):
        layer(np.ones((2, 7), dtype=np.float32))


def test_the_output_follows_the_input_dtype_under_mixed_float16(
    mixed_float16_policy,
):
    layer = MPSLayer(output_dim=4)
    layer.build((None, 6))
    out = layer(np.ones((2, 6), dtype=np.float32))
    assert "float16" in str(out.dtype)


def test_compute_output_shape_is_rank_generic():
    layer = MPSLayer(output_dim=4)
    assert layer.compute_output_shape((2, 3, 6)) == (2, 3, 4)
    assert layer.compute_output_shape([[None, 8]]) == (None, 4)
