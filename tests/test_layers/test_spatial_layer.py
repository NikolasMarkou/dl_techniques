"""Tests for `layers/spatial_layer.py`: `coordinate_grid`, `interpolate_grid`
and the `SpatialLayer` generator.

Merged on 2026-08-31 from the former `tests/test_layers/test_grid_sample.py`
when `layers/grid_sample.py` was folded into `layers/spatial_layer.py`. The
sampler tests are unchanged in intent: numerical correctness against
`scipy.ndimage.map_coordinates`, plus the INV-4 differentiability contract that
the THERA Jacobian-TV loss depends on.

Some tests deliberately reach for `tensorflow` — the gradient and graph-mode
arms need a tape and a `tf.function`, which `keras.ops` does not provide. The
MODULE UNDER TEST imports no tensorflow, and `test_the_module_uses_no_raw_tensorflow`
below is what enforces that.
"""

import ast
import os

import keras
import numpy as np
import pytest
import tensorflow as tf
from scipy.ndimage import map_coordinates

import dl_techniques.layers.spatial_layer as spatial_layer_module
from dl_techniques.layers.spatial_layer import (
    SpatialLayer,
    coordinate_grid,
    interpolate_grid,
)

B, H, W, C = 2, 8, 8, 3


@pytest.fixture
def sample():
    return np.random.default_rng(0).standard_normal((B, H, W, C)).astype("float32")


def _np(x):
    """Backend-agnostic materialization (the module no longer returns tf only)."""
    return np.asarray(keras.ops.convert_to_numpy(x))


# ---------------------------------------------------------------------
# 1. coordinate_grid — the THERA ('centers', 'ij', 'none') corner
# ---------------------------------------------------------------------


def test_coordinate_grid_shape_and_corner():
    g = coordinate_grid(4)
    assert g.shape == (4, 4, 2)
    # Corner [0,0] holds the first pixel center on both axes: -0.5 + 1/8.
    np.testing.assert_allclose(g[0, 0, 0], -0.5 + 1.0 / 8, atol=1e-6, rtol=0)
    np.testing.assert_allclose(g[0, 0, 1], -0.5 + 1.0 / 8, atol=1e-6, rtol=0)
    # Opposite corner: 0.5 - 1/8.
    np.testing.assert_allclose(g[-1, -1, 0], 0.5 - 1.0 / 8, atol=1e-6, rtol=0)
    np.testing.assert_allclose(g[-1, -1, 1], 0.5 - 1.0 / 8, atol=1e-6, rtol=0)


def test_coordinate_grid_center_symmetry():
    g = coordinate_grid(6)
    # Grid is point-symmetric about the origin.
    np.testing.assert_allclose(g, -g[::-1, ::-1, :], atol=1e-6, rtol=0)
    # Mean is (approximately) zero.
    np.testing.assert_allclose(g.mean(axis=(0, 1)), [0.0, 0.0], atol=1e-6, rtol=0)


def test_coordinate_grid_hand_computed():
    # n=2 -> centers at -0.25, 0.25; indexing='ij' so axis0=h varies down rows.
    g = coordinate_grid(2)
    expected = np.array(
        [
            [[-0.25, -0.25], [-0.25, 0.25]],
            [[0.25, -0.25], [0.25, 0.25]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(g, expected, atol=1e-6, rtol=0)


def test_coordinate_grid_rectangular():
    g = coordinate_grid((2, 3))
    assert g.shape == (2, 3, 2)
    # h axis: 2 centers; w axis: 3 centers.
    np.testing.assert_allclose(g[0, 0, 0], -0.25, atol=1e-6, rtol=0)
    np.testing.assert_allclose(g[0, 0, 1], -0.5 + 1.0 / 6, atol=1e-6, rtol=0)


def test_coordinate_grid_int_equals_square_tuple():
    np.testing.assert_array_equal(coordinate_grid(5), coordinate_grid((5, 5)))


# ---------------------------------------------------------------------
# 1b. coordinate_grid — the full 2x2x2 convention matrix, hand-computed
#
# Each knob is a silent transpose / rescale if wrong: every shape assertion in
# this file still passes under a flipped `channel_order` or a swapped
# `alignment`. So the matrix is pinned by VALUE, at n=2 where the four numbers
# are writable by hand.
#
#   centers,   n=2 -> [-0.25, 0.25]
#   endpoints, n=2 -> [-0.5,  0.5 ]
#   zscore of a symmetric 2-point axis -> [-1, +1] under either alignment
#     (z-score is invariant to affine reparameterization, which is exactly the
#     property the module docstring claims)
# ---------------------------------------------------------------------

_H2 = {"centers": -0.25, "endpoints": -0.5}


@pytest.mark.parametrize("alignment", ["centers", "endpoints"])
@pytest.mark.parametrize("channel_order", ["ij", "xy"])
@pytest.mark.parametrize("normalization", ["none", "zscore"])
def test_coordinate_grid_convention_matrix(alignment, channel_order, normalization):
    g = coordinate_grid(
        2,
        alignment=alignment,
        channel_order=channel_order,
        normalization=normalization,
    )
    assert g.shape == (2, 2, 2)

    lo = -1.0 if normalization == "zscore" else _H2[alignment]
    # h varies down rows (axis 0), w varies across columns (axis 1).
    h_plane = np.array([[lo, lo], [-lo, -lo]], dtype=np.float32)
    w_plane = np.array([[lo, -lo], [lo, -lo]], dtype=np.float32)

    if channel_order == "ij":
        expected = np.stack([h_plane, w_plane], axis=-1)
    else:
        expected = np.stack([w_plane, h_plane], axis=-1)

    np.testing.assert_allclose(g, expected, atol=1e-6, rtol=0)


def test_coordinate_grid_channel_order_is_a_real_transpose():
    """Anti-vacuity: the two channel orders must actually differ.

    A rectangular grid is used on purpose — on a square grid with symmetric
    axes the two orders are a transpose of each other, and an implementation
    that ignored the knob entirely could still look plausible.
    """
    ij = coordinate_grid((2, 3), channel_order="ij")
    xy = coordinate_grid((2, 3), channel_order="xy")
    assert np.abs(ij - xy).max() > 0.1
    np.testing.assert_array_equal(ij[..., 0], xy[..., 1])
    np.testing.assert_array_equal(ij[..., 1], xy[..., 0])


def test_coordinate_grid_alignment_gap_shrinks_with_n():
    """`centers` and `endpoints` differ by 1/(2n) at the first sample."""
    for n in (2, 4, 8, 64):
        c = coordinate_grid(n, alignment="centers")
        e = coordinate_grid(n, alignment="endpoints")
        np.testing.assert_allclose(
            c[0, 0, 0] - e[0, 0, 0], 1.0 / (2.0 * n), atol=1e-6, rtol=0
        )


def test_coordinate_grid_single_element_axis_is_finite_under_zscore():
    """n=1 has zero variance — the epsilon is the only thing preventing a NaN."""
    g = coordinate_grid((1, 5), normalization="zscore")
    assert g.shape == (1, 5, 2)
    assert np.isfinite(g).all()
    # The degenerate h axis standardizes to exactly zero.
    np.testing.assert_allclose(g[..., 0], 0.0, atol=1e-6, rtol=0)
    # The w axis still standardizes to unit std.
    np.testing.assert_allclose(g[..., 1].std(), 1.0, atol=1e-5, rtol=0)


@pytest.mark.parametrize("bad", [
    {"size": (4,)},
    {"size": (0, 4)},
    {"size": (4, -1)},
    {"alignment": "bogus"},
    {"channel_order": "bogus"},
    {"normalization": "bogus"},
])
def test_coordinate_grid_invalid_args_raise(bad):
    kwargs = {"size": 4}
    kwargs.update(bad)
    with pytest.raises(ValueError):
        coordinate_grid(**kwargs)


def test_coordinate_grid_dtype_is_honoured():
    assert coordinate_grid(3).dtype == np.float32
    assert coordinate_grid(3, dtype="float64").dtype == np.float64


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
# 2. interpolate_grid, order=0 correctness
# ---------------------------------------------------------------------


def test_order0_recovers_grid_at_own_centers():
    # Ramp grid 4x4, C=1. Sampling at coordinate_grid centers must recover values.
    ramp = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
    coords = coordinate_grid(4)[None]  # (1,4,4,2)
    out = _np(interpolate_grid(coords, ramp, order=0))
    np.testing.assert_allclose(out, ramp, atol=1e-5, rtol=0)


def test_order0_matches_scipy_random():
    rng = np.random.default_rng(0)
    grid_2d = rng.standard_normal((5, 7)).astype(np.float32)
    coords = (rng.random((3, 4, 2)).astype(np.float32) - 0.5)  # in [-0.5,0.5]
    grid = grid_2d[None, :, :, None]  # (1,5,7,1)
    out = _np(interpolate_grid(coords[None], grid, order=0))[0, :, :, 0]
    ref = _scipy_sample(grid_2d, coords, order=0)
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------
# 3. order=1 correctness
# ---------------------------------------------------------------------


def test_order1_matches_scipy_random():
    rng = np.random.default_rng(1)
    grid_2d = rng.standard_normal((6, 5)).astype(np.float32)
    coords = (rng.random((4, 4, 2)).astype(np.float32) - 0.5)
    grid = grid_2d[None, :, :, None]
    out = _np(interpolate_grid(coords[None], grid, order=1))[0, :, :, 0]
    ref = _scipy_sample(grid_2d, coords, order=1)
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=0)


def test_out_of_range_coords_clamp_to_the_edge():
    """Border handling is mode='nearest' — replicate, never wrap or zero-fill.

    Exercised in isolation rather than incidentally: every other numeric test
    samples inside the domain, where a wrap and a clamp agree.
    """
    ramp = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
    far = np.array([[[[-5.0, -5.0], [5.0, 5.0]]]], dtype=np.float32)  # (1,1,2,2)
    for order in (0, 1):
        out = _np(interpolate_grid(far, ramp, order=order))
        # Beyond the low edge -> corner (0,0) = 0; beyond the high edge -> 15.
        np.testing.assert_allclose(out[0, 0, 0, 0], 0.0, atol=1e-5, rtol=0)
        np.testing.assert_allclose(out[0, 0, 1, 0], 15.0, atol=1e-5, rtol=0)


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


def test_order1_grad_wrt_grid_finite_nonzero():
    """The gather must also pass gradient back to the sampled VALUES.

    `keras.ops.take` replaced `tf.gather_nd` here; a take that dropped the
    value-side gradient would leave the coordinate arm above green.
    """
    rng = np.random.default_rng(5)
    coords = (rng.random((1, 4, 4, 2)).astype(np.float32) - 0.5)
    grid = tf.Variable(rng.standard_normal((1, 5, 5, 3)).astype(np.float32))
    for order in (0, 1):
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(interpolate_grid(coords, grid, order=order))
        grad = tape.gradient(loss, grid)
        assert grad is not None, f"no gradient to the grid at order={order}"
        g = grad.numpy()
        assert np.all(np.isfinite(g))
        assert np.any(np.abs(g) > 1e-6)


# ---------------------------------------------------------------------
# 5. identity sample (order=1)
# ---------------------------------------------------------------------


def test_order1_identity_at_own_centers():
    rng = np.random.default_rng(3)
    grid = rng.standard_normal((1, 6, 6, 2)).astype(np.float32)
    coords = coordinate_grid(6)[None]
    out = _np(interpolate_grid(coords, grid, order=1))
    np.testing.assert_allclose(out, grid, atol=1e-4, rtol=0)


# ---------------------------------------------------------------------
# 6. batch correctness
# ---------------------------------------------------------------------


@pytest.mark.parametrize("batch", [2, 3])
def test_batch_samples_independently(batch):
    """Pins the batch offset of the linearized gather index.

    The source grid is deliberately NON-SQUARE. The index is
    ``(b * H' + ih) * W' + iw``, and on a square grid ``H' == W'`` so a wrong
    stride is arithmetically identical — a 5x5 version of this test passed
    against an injected ``b * W'`` and saw nothing.
    """
    rng = np.random.default_rng(4)
    planes = [rng.standard_normal((5, 7)).astype(np.float32) for _ in range(batch)]
    grid = np.stack(planes, axis=0)[..., None]  # (batch,5,7,1)
    coords = (rng.random((batch, 3, 3, 2)).astype(np.float32) - 0.5)
    out = _np(interpolate_grid(coords, grid, order=1))

    for b, plane in enumerate(planes):
        ref = _scipy_sample(plane, coords[b], order=1)
        np.testing.assert_allclose(out[b, :, :, 0], ref, atol=1e-4, rtol=0)


def test_invalid_order_raises():
    grid = np.zeros((1, 3, 3, 1), dtype=np.float32)
    coords = np.zeros((1, 2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        interpolate_grid(coords, grid, order=2)


# ---------------------------------------------------------------------
# 7. graph mode with an UNKNOWN batch dimension
#
# The `keras.ops` port builds a linearized flat index from `ops.shape`, where
# the `tf.gather_nd` original did not. Eager execution hands it concrete ints
# and so cannot see a shape bug that only bites when the batch dim is None —
# which is the regime `model.fit` uses.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("order", [0, 1])
def test_dynamic_batch_under_tf_function(order):
    rng = np.random.default_rng(6)

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, 3, 4, 2], tf.float32),
            tf.TensorSpec([None, 5, 7, 3], tf.float32),
        ]
    )
    def sample(coords, grid):
        return interpolate_grid(coords, grid, order=order)

    for batch in (1, 4):
        coords = (rng.random((batch, 3, 4, 2)) - 0.5).astype(np.float32)
        grid = rng.standard_normal((batch, 5, 7, 3)).astype(np.float32)
        traced = sample(coords, grid).numpy()
        assert traced.shape == (batch, 3, 4, 3)

        # Eager-vs-traced alone is a SELF-REFERENTIAL oracle: both sides run the
        # same code, so a wrong flat index agrees with itself. Measured — an
        # injected wrong batch stride left this test green until the scipy arm
        # below was added. Compare against an INDEPENDENT reference.
        eager = _np(interpolate_grid(coords, grid, order=order))
        np.testing.assert_allclose(traced, eager, atol=0, rtol=0)
        for b in range(batch):
            for c in range(3):
                ref = _scipy_sample(grid[b, :, :, c], coords[b], order=order)
                np.testing.assert_allclose(
                    traced[b, :, :, c], ref, atol=1e-4, rtol=0
                )


# ---------------------------------------------------------------------
# 7b. the D-046 dtype contract, guarded NEXT TO THE CODE
#
# This rule also has an arm in `tests/test_models/test_the_fp16_unreachable_family_runs.py`,
# which is where the original defect was caught. That is a models-level file and
# a scoped run of `tests/test_layers/` never loads it — measured: an injected
# "always return float32" left all 55 tests in this module green. A contract
# owned by this module needs a guard in this module.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("grid_dtype", ["float16", "float32", "float64"])
@pytest.mark.parametrize("coord_dtype", ["float16", "float32"])
def test_sampled_values_carry_the_grid_dtype(order, grid_dtype, coord_dtype):
    """Index math is float32; the RETURN follows the grid, not the coords.

    The coord dtype is varied independently of the grid dtype on purpose: with
    the two tied together, an implementation returning either operand's dtype
    passes.
    """
    rng = np.random.default_rng(11)
    coords = (rng.random((1, 5, 5, 2)) - 0.5).astype(coord_dtype)
    grid = rng.standard_normal((1, 4, 4, 3)).astype(grid_dtype)
    out = interpolate_grid(coords, grid, order=order)
    assert keras.backend.standardize_dtype(out.dtype) == grid_dtype
    assert np.isfinite(_np(keras.ops.cast(out, "float32"))).all()


def test_an_integer_grid_is_promoted_to_float32():
    """Integer feature grids cannot hold a lerp; they are cast, not rejected."""
    grid = np.arange(16, dtype="int32").reshape(1, 4, 4, 1)
    for order in (0, 1):
        out = interpolate_grid(coordinate_grid(4)[None], grid, order=order)
        assert keras.backend.standardize_dtype(out.dtype) == "float32"
        np.testing.assert_allclose(
            _np(out), grid.astype("float32"), atol=1e-5, rtol=0
        )


# ---------------------------------------------------------------------
# 8. the module itself
# ---------------------------------------------------------------------


def test_the_module_uses_no_raw_tensorflow():
    """`spatial_layer.py` must reach the backend only through `keras.ops`.

    `interpolate_grid` used to be raw `tf` (`tf.gather_nd`) and was the reason
    the repo's authoring guide named `grid_sample` as an unmigratable op. The
    port is measured bit-identical; this keeps it from creeping back.
    """
    tree = ast.parse(open(spatial_layer_module.__file__, encoding="utf-8").read())
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders += [
                a.name for a in node.names if a.name.split(".")[0] == "tensorflow"
            ]
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] == "tensorflow":
                offenders.append(node.module)
    assert offenders == [], (
        f"{spatial_layer_module.__file__} imports tensorflow: {offenders}"
    )


# ---------------------------------------------------------------------
# 9. SpatialLayer
# ---------------------------------------------------------------------


class TestSpatialLayer:

    def test_construction(self):
        layer = SpatialLayer(resolution=(4, 4))
        assert layer.resolution == (4, 4)
        # Defaults are the historical CoordConv conventions.
        assert layer.alignment == "endpoints"
        assert layer.channel_order == "xy"
        assert layer.normalization == "zscore"

    @pytest.mark.parametrize("bad", [
        {"resolution": (4,)},
        {"resolution": (0, 4)},
        {"resize_method": "bogus"},
        {"alignment": "bogus"},
        {"channel_order": "bogus"},
        {"normalization": "bogus"},
    ])
    def test_invalid_args_raise(self, bad):
        with pytest.raises(ValueError):
            SpatialLayer(**bad)

    def test_forward_pass(self, sample):
        out = SpatialLayer(resolution=(4, 4))(sample)
        assert tuple(out.shape) == (B, H, W, 2)

    def test_build_wrong_rank_raises(self):
        with pytest.raises(ValueError):
            SpatialLayer().build((B, H))

    def test_compute_output_shape(self):
        assert SpatialLayer().compute_output_shape((B, H, W, C)) == (B, H, W, 2)

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(H, W, C))
        out = SpatialLayer(resolution=(4, 4), resize_method="bilinear", name="spatial")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample)
        path = os.path.join(tmp_path, "spatial.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"SpatialLayer": SpatialLayer}
        )
        y1 = loaded(sample)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )

    def test_config_round_trip_carries_the_conventions(self):
        layer = SpatialLayer(
            resolution=None,
            alignment="centers",
            channel_order="ij",
            normalization="none",
        )
        clone = SpatialLayer.from_config(layer.get_config())
        for knob in ("resolution", "alignment", "channel_order", "normalization",
                     "resize_method"):
            assert getattr(clone, knob) == getattr(layer, knob), knob

    # -- the values, not just the shapes -------------------------------
    #
    # Captured from the pre-merge implementation before `build()` was rewritten
    # to delegate to `coordinate_grid`. Nothing used to test the grid's numbers
    # at all, so this had to be read off the OLD code, not the new.
    #
    #   z-score of linspace(-0.5, 0.5, 4) = [-0.5, -1/6, 1/6, 0.5] / 0.372678
    #                                     = [-1.3416408, -0.4472136, ...]
    #
    # The delta between old and new is <= 1.19e-07 (numpy accumulates the mean
    # and std in float64 where `keras.ops` used float32); the tolerance below is
    # 1e-6, an order above that and five orders below the values themselves.

    _DEFAULT_AXIS_4 = np.array(
        [-1.3416408, -0.4472136, 0.4472136, 1.3416408], dtype=np.float32
    )

    def test_default_prototype_grid_values_are_unchanged(self):
        layer = SpatialLayer()  # resolution=(4, 4), endpoints, xy, zscore
        layer.build((None, H, W, C))
        proto = _np(layer.xy_grid)[0]
        assert proto.shape == (4, 4, 2)
        # channel 0 is x (varies across columns), channel 1 is y (down rows).
        for row in range(4):
            np.testing.assert_allclose(
                proto[row, :, 0], self._DEFAULT_AXIS_4, atol=1e-6, rtol=0
            )
        for col in range(4):
            np.testing.assert_allclose(
                proto[:, col, 1], self._DEFAULT_AXIS_4, atol=1e-6, rtol=0
            )

    def test_default_forward_values_are_unchanged(self, sample):
        """Nearest-resize of the 4x4 prototype onto an 8x8 input duplicates
        each coordinate step — the piecewise-constant ramp the docstring warns
        about, pinned so a resize-path change cannot pass unnoticed."""
        out = _np(SpatialLayer()(sample))
        assert out.shape == (B, H, W, 2)
        expected_x = np.repeat(self._DEFAULT_AXIS_4, 2)
        np.testing.assert_allclose(out[0, 0, :, 0], expected_x, atol=1e-6, rtol=0)
        np.testing.assert_allclose(out[0, :, 0, 1], expected_x, atol=1e-6, rtol=0)
        # Every batch element is the same grid.
        np.testing.assert_array_equal(out[0], out[1])

    # -- the claim that let `grid_sample.py` be deleted -----------------

    @pytest.mark.parametrize("hw", [(8, 8), (5, 7), (31, 37)])
    def test_layer_reproduces_coordinate_grid_exactly(self, hw):
        """`SpatialLayer` spans the convention `make_grid` used to own.

        With `resolution=None` there is no resize, so this is an EXACT equality
        (atol=0), not an approximation — which is what makes the old module's
        "NOT equivalent" claim obsolete rather than merely close.
        """
        h, w = hw
        layer = SpatialLayer(
            resolution=None,
            alignment="centers",
            channel_order="ij",
            normalization="none",
        )
        out = _np(layer(np.zeros((3, h, w, C), dtype="float32")))
        expected = np.broadcast_to(coordinate_grid((h, w))[None], (3, h, w, 2))
        assert out.shape == (3, h, w, 2)
        np.testing.assert_allclose(out, expected, atol=0, rtol=0)

    def test_resolution_none_requires_static_spatial_dims(self):
        layer = SpatialLayer(resolution=None)
        with pytest.raises(ValueError, match="statically known"):
            layer.build((None, None, None, C))

    def test_resolution_none_differs_from_a_coarse_prototype(self):
        """`resolution=None` is a different GRID, not just a cheaper path.

        Deliberately NOT named "skips the resize": measured, forcing the
        `resolution=None` branch back through `keras.ops.image.resize` at the
        same target size is an EXACT identity (the equality test above still
        held at atol=0), so "skips the resize" is a cost claim, not an
        observable one, and a test asserting it would be untestable by
        construction. What IS observable is that a 4x4 prototype resampled up
        to 5x5 is a piecewise-constant approximation of the real 5x5 grid.
        """
        x = np.zeros((1, 5, 5, C), dtype="float32")
        exact = _np(SpatialLayer(resolution=None)(x))
        resized = _np(SpatialLayer(resolution=(4, 4))(x))
        assert np.abs(exact - resized).max() > 0.1

    def test_output_follows_the_compute_dtype(self):
        """The grid is built in float32 for precision but returned in the
        layer's compute dtype, so it can be concatenated onto a float16
        feature map without a mismatch."""
        layer = SpatialLayer(dtype="float16")
        out = layer(np.zeros((1, H, W, C), dtype="float16"))
        assert keras.backend.standardize_dtype(out.dtype) == "float16"
        assert np.isfinite(_np(out)).all()
