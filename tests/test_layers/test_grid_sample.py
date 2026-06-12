"""Scoped tests for TF-native grid_sample helpers (plan step 2).

Pins both numerical correctness (vs scipy.ndimage.map_coordinates) AND the
INV-4 differentiability contract that step 9 (Jacobian-TV) depends on.
"""

import numpy as np
import pytest
import tensorflow as tf
from scipy.ndimage import map_coordinates

from dl_techniques.layers.grid_sample import make_grid, interpolate_grid


# ---------------------------------------------------------------------
# 1. make_grid
# ---------------------------------------------------------------------


def test_make_grid_shape_and_corner():
    g = make_grid(4)
    assert g.shape == (4, 4, 2)
    # Corner [0,0] holds the first pixel center on both axes: -0.5 + 1/8.
    np.testing.assert_allclose(g[0, 0, 0], -0.5 + 1.0 / 8, atol=1e-6)
    np.testing.assert_allclose(g[0, 0, 1], -0.5 + 1.0 / 8, atol=1e-6)
    # Opposite corner: 0.5 - 1/8.
    np.testing.assert_allclose(g[-1, -1, 0], 0.5 - 1.0 / 8, atol=1e-6)
    np.testing.assert_allclose(g[-1, -1, 1], 0.5 - 1.0 / 8, atol=1e-6)


def test_make_grid_center_symmetry():
    g = make_grid(6)
    # Grid is point-symmetric about the origin.
    np.testing.assert_allclose(g, -g[::-1, ::-1, :], atol=1e-6)
    # Mean is (approximately) zero.
    np.testing.assert_allclose(g.mean(axis=(0, 1)), [0.0, 0.0], atol=1e-6)


def test_make_grid_hand_computed():
    # n=2 -> centers at -0.25, 0.25; indexing='ij' so axis0=h varies down rows.
    g = make_grid(2)
    expected = np.array(
        [
            [[-0.25, -0.25], [-0.25, 0.25]],
            [[0.25, -0.25], [0.25, 0.25]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(g, expected, atol=1e-6)


def test_make_grid_rectangular():
    g = make_grid((2, 3))
    assert g.shape == (2, 3, 2)
    # h axis: 2 centers; w axis: 3 centers.
    np.testing.assert_allclose(g[0, 0, 0], -0.25, atol=1e-6)
    np.testing.assert_allclose(g[0, 0, 1], -0.5 + 1.0 / 6, atol=1e-6)


def test_make_grid_int_equals_square_tuple():
    np.testing.assert_array_equal(make_grid(5), make_grid((5, 5)))


# ---------------------------------------------------------------------
# helpers for the scipy oracle
# ---------------------------------------------------------------------


def _scipy_sample(grid_2d, coords_bhw2, order):
    """Reference per-channel sampling via scipy.map_coordinates(mode='nearest').

    grid_2d: (H', W') single channel. coords_bhw2: (Hq, Wq, 2) in [-0.5,0.5].
    Returns (Hq, Wq).
    """
    sh, sw = grid_2d.shape
    pix_h = coords_bhw2[..., 0] * sh + (sh - 1) / 2.0
    pix_w = coords_bhw2[..., 1] * sw + (sw - 1) / 2.0
    out = map_coordinates(
        grid_2d,
        [pix_h.ravel(), pix_w.ravel()],
        order=order,
        mode="nearest",
    )
    return out.reshape(pix_h.shape)


# ---------------------------------------------------------------------
# 2. order=0 correctness
# ---------------------------------------------------------------------


def test_order0_recovers_grid_at_own_centers():
    # Ramp grid 4x4, C=1. Sampling at make_grid centers must recover values.
    ramp = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
    coords = make_grid(4)[None]  # (1,4,4,2)
    out = interpolate_grid(coords, ramp, order=0).numpy()
    np.testing.assert_allclose(out, ramp, atol=1e-5)


def test_order0_matches_scipy_random():
    rng = np.random.default_rng(0)
    grid_2d = rng.standard_normal((5, 7)).astype(np.float32)
    coords = (rng.random((3, 4, 2)).astype(np.float32) - 0.5)  # in [-0.5,0.5]
    grid = grid_2d[None, :, :, None]  # (1,5,7,1)
    out = interpolate_grid(coords[None], grid, order=0).numpy()[0, :, :, 0]
    ref = _scipy_sample(grid_2d, coords, order=0)
    np.testing.assert_allclose(out, ref, atol=1e-5)


# ---------------------------------------------------------------------
# 3. order=1 correctness
# ---------------------------------------------------------------------


def test_order1_matches_scipy_random():
    rng = np.random.default_rng(1)
    grid_2d = rng.standard_normal((6, 5)).astype(np.float32)
    coords = (rng.random((4, 4, 2)).astype(np.float32) - 0.5)
    grid = grid_2d[None, :, :, None]
    out = interpolate_grid(coords[None], grid, order=1).numpy()[0, :, :, 0]
    ref = _scipy_sample(grid_2d, coords, order=1)
    np.testing.assert_allclose(out, ref, atol=1e-4)


# ---------------------------------------------------------------------
# 4. order=1 differentiability (INV-4)
# ---------------------------------------------------------------------


def test_order1_grad_wrt_coords_finite_nonzero():
    rng = np.random.default_rng(2)
    grid = rng.standard_normal((1, 5, 5, 3)).astype(np.float32)
    coords = tf.Variable(
        (rng.random((1, 4, 4, 2)).astype(np.float32) - 0.5)
    )
    with tf.GradientTape() as tape:
        out = interpolate_grid(coords, grid, order=1)
        loss = tf.reduce_sum(out)
    grad = tape.gradient(loss, coords)
    assert grad is not None
    grad_np = grad.numpy()
    assert np.all(np.isfinite(grad_np))
    assert np.any(np.abs(grad_np) > 1e-6)


# ---------------------------------------------------------------------
# 5. identity sample (order=1)
# ---------------------------------------------------------------------


def test_order1_identity_at_own_centers():
    rng = np.random.default_rng(3)
    grid = rng.standard_normal((1, 6, 6, 2)).astype(np.float32)
    coords = make_grid(6)[None]
    out = interpolate_grid(coords, grid, order=1).numpy()
    np.testing.assert_allclose(out, grid, atol=1e-4)


# ---------------------------------------------------------------------
# 6. batch correctness
# ---------------------------------------------------------------------


def test_batch_samples_independently():
    rng = np.random.default_rng(4)
    g0 = rng.standard_normal((5, 5)).astype(np.float32)
    g1 = rng.standard_normal((5, 5)).astype(np.float32)
    grid = np.stack([g0, g1], axis=0)[..., None]  # (2,5,5,1)
    coords = (rng.random((2, 3, 3, 2)).astype(np.float32) - 0.5)
    out = interpolate_grid(coords, grid, order=1).numpy()

    ref0 = _scipy_sample(g0, coords[0], order=1)
    ref1 = _scipy_sample(g1, coords[1], order=1)
    np.testing.assert_allclose(out[0, :, :, 0], ref0, atol=1e-4)
    np.testing.assert_allclose(out[1, :, :, 0], ref1, atol=1e-4)


def test_invalid_order_raises():
    grid = np.zeros((1, 3, 3, 1), dtype=np.float32)
    coords = np.zeros((1, 2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        interpolate_grid(coords, grid, order=2)
