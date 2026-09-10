"""Value-level guards for ``doc_scanner``'s pure rectification primitives.

Every assertion here is against a HAND-COMPUTED number or an explicit
orientation, never against a shape. That is the whole point of the file. The
defects this module is pointed at -- a half-pixel sampling offset, an ``(x, y)``
vs ``(h, w)`` channel swap, a double-reversed meshgrid -- share one property:
they are *shape-preserving*. A model built on any of them produces
``(B, 288, 288, 2)`` finite float32 output, saves, reloads, and trains its loss
down. It simply learns a transposed or half-pixel-shifted warp. This repo has a
recorded incident where exactly this defect class (a single sign error in one
``ops.roll``) survived 249 shape/config/serialization tests.

Three design rules follow from that, and are enforced throughout:

1. **Non-square fixtures.** A square fixture is structurally blind to an h/w
   axis swap: both branches produce identical output. Where an axis could be
   confused, the fixture is ``H=4, W=7``.
2. **Hand-computed expectations, ``rtol=0``.** The 5x5 reference values below
   are derived in the docstring that carries them, from the sampler's own
   documented formula -- not read back from the implementation under test.
3. **RED-proven.** Both critical guards were run against a deliberately wrong
   implementation and observed to FAIL before being accepted; the observed
   failures are recorded in this plan's ``decisions.md`` under D-009. A guard
   that has never been seen red is not known to work.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.vision.image_restoration.doc_scanner.components import (
    SPATIAL_DIVISOR,
)
from dl_techniques.models.vision.image_restoration.doc_scanner.warp import (
    coords_grid,
    sample_at_pixel_coords,
)

# The repo convention for a value assertion: an explicit absolute tolerance and
# NO relative component, so a near-zero expected value is not silently excused.
ATOL = 1e-6


def _ramp_5x5() -> np.ndarray:
    """``(1, 5, 5, 2)``: channel 0 is the column index, channel 1 the row index.

    Sampling this map at pixel ``(x, y)`` under a correct ``align_corners=True``
    adapter returns exactly ``(x, y)`` -- bilinear interpolation of a linear
    ramp is exact -- which is what makes the reference hand-computable on BOTH
    axes at once.
    """
    rows, cols = np.meshgrid(np.arange(5.0), np.arange(5.0), indexing="ij")
    return np.stack([cols, rows], axis=-1)[None].astype("float32")


class TestAlignCornersAdapterAgainstAHandComputedReference:
    """The 5x5 reference from plan Assumption A1 / findings F-14.

    ``interpolate_grid`` is a half-pixel sampler: it computes internally
    ``pix = coord * S + (S - 1) / 2``. At ``S = 5`` that is
    ``pix = 5 * coord + 2``. The adapter therefore has to send an absolute
    pixel index ``p`` to ``coord = (p - 2) / 5``, and the four probe points
    below are hand-evaluated from that::

        p = 0    -> coord = -0.4  -> pix = 5 * (-0.4) + 2 = 0.0
        p = 1    -> coord = -0.2  -> pix = 5 * (-0.2) + 2 = 1.0
        p = 2.5  -> coord =  0.1  -> pix = 5 * ( 0.1) + 2 = 2.5
        p = 4    -> coord =  0.4  -> pix = 5 * ( 0.4) + 2 = 4.0

    The contrast that matters: the *natural-looking* half-pixel adapter
    ``coord = p / S - 0.5`` gives ``pix = p - 0.5``, i.e. ``{-0.5, 0.5, 2.0,
    3.5}`` -- clamped and wrong at every probe but the first, and wrong by half
    a pixel everywhere. Nothing but a reference like this one can see that.
    """

    PROBES = [0.0, 1.0, 2.5, 4.0]

    @pytest.mark.parametrize("probe", PROBES)
    def test_align_corners_adapter_reads_the_x_axis_at_the_exact_pixel(self, probe):
        """``p -> p`` along x, for ``p`` in ``{0, 1, 2.5, 4}``."""
        fmap = _ramp_5x5()
        pix = np.array([[[[probe, 0.0]]]], dtype="float32")

        out = keras.ops.convert_to_numpy(sample_at_pixel_coords(fmap, pix))

        np.testing.assert_allclose(out[0, 0, 0, 0], probe, rtol=0, atol=ATOL)

    @pytest.mark.parametrize("probe", PROBES)
    def test_align_corners_adapter_reads_the_y_axis_at_the_exact_pixel(self, probe):
        """``p -> p`` along y as well -- the swap must not lose one axis."""
        fmap = _ramp_5x5()
        pix = np.array([[[[0.0, probe]]]], dtype="float32")

        out = keras.ops.convert_to_numpy(sample_at_pixel_coords(fmap, pix))

        np.testing.assert_allclose(out[0, 0, 0, 1], probe, rtol=0, atol=ATOL)

    def test_align_corners_adapter_interpolates_a_nonlinear_ramp_by_hand(self):
        """A ramp alone pins only an affine adapter; this pins the LERP too.

        Values ``[0, 1, 4, 9, 16]`` along x. At ``p = 2.5`` a 4-corner bilinear
        read returns the midpoint of pixels 2 and 3, ``(4 + 9) / 2 = 6.5`` --
        NOT ``2.5 ** 2 = 6.25``. Asserting 6.5 pins the interpolation kind as
        well as the coordinate transform.
        """
        values = np.arange(5.0) ** 2
        fmap = np.broadcast_to(values[None, None, :, None], (1, 5, 5, 1)).astype(
            "float32"
        )
        pix = np.array([[[[2.5, 0.0]]]], dtype="float32")

        out = keras.ops.convert_to_numpy(sample_at_pixel_coords(fmap, pix))

        np.testing.assert_allclose(out[0, 0, 0, 0], 6.5, rtol=0, atol=ATOL)

    @pytest.mark.parametrize(
        "normalized, expected_pixel", [(-1.0, 0.0), (0.0, 2.0), (1.0, 4.0)]
    )
    def test_align_corners_adapter_composes_with_the_torch_normalization(
        self, normalized, expected_pixel
    ):
        """End-to-end equivalence with upstream ``bilinear_sampler``.

        Upstream (``model.py:12-13``) normalizes a pixel coordinate with
        ``n = 2 * p / (S - 1) - 1`` and passes ``align_corners=True``. Inverting
        it, ``p = (n + 1) / 2 * (S - 1)``, so at ``S = 5`` the torch-normalized
        probes ``{-1, 0, 1}`` denote pixels ``{0, 2, 4}``. This test drives the
        adapter through THAT composition, which is the property the port
        actually needs -- the pixel-domain tests above are its two halves.
        """
        size = 5
        pixel = (normalized + 1.0) / 2.0 * (size - 1)
        assert pixel == expected_pixel

        fmap = _ramp_5x5()
        pix = np.array([[[[pixel, pixel]]]], dtype="float32")

        out = keras.ops.convert_to_numpy(sample_at_pixel_coords(fmap, pix))

        np.testing.assert_allclose(
            out[0, 0, 0], [expected_pixel, expected_pixel], rtol=0, atol=ATOL
        )

    def test_align_corners_adapter_is_edge_clamped_not_zero_padded(self):
        """``F.grid_sample``'s default padding is ``'zeros'``; ours clamps.

        Recorded deliberately rather than asserted as parity: upstream never
        samples out of range in its own forward path (``coords1`` stays inside
        the image for any trained network), and the plan's edge-case note calls
        clamping the graceful degradation for a badly-initialised one. A future
        reader comparing against torch will find this difference here.
        """
        fmap = _ramp_5x5()
        pix = np.array([[[[-3.0, 9.0]]]], dtype="float32")

        out = keras.ops.convert_to_numpy(sample_at_pixel_coords(fmap, pix))

        np.testing.assert_allclose(out[0, 0, 0], [0.0, 4.0], rtol=0, atol=ATOL)


class TestAlignCornersAdapterOnANonSquareMap:
    """``H=4, W=7``. A square fixture cannot see an h/w swap at all."""

    def test_align_corners_adapter_uses_each_axis_own_size(self):
        """Each axis divides by ITS OWN extent, not by a shared one.

        With ``H = 4`` and ``W = 7`` the two half-pixel offsets differ
        (``1.5`` vs ``3.0``), so an implementation that reuses one size for
        both axes -- inert at 288x288 -- reads the wrong pixel here.
        """
        height, width = 4, 7
        rows, cols = np.meshgrid(
            np.arange(float(height)), np.arange(float(width)), indexing="ij"
        )
        fmap = np.stack([cols, rows], axis=-1)[None].astype("float32")

        pix = np.array([[[[6.0, 3.0], [0.0, 3.0], [6.0, 0.0]]]], dtype="float32")

        out = keras.ops.convert_to_numpy(sample_at_pixel_coords(fmap, pix))

        np.testing.assert_allclose(
            out[0, 0], [[6.0, 3.0], [0.0, 3.0], [6.0, 0.0]], rtol=0, atol=ATOL
        )

    def test_align_corners_adapter_rejects_a_transposed_read(self):
        """A swapped-channel adapter would read ``(3, 6)``, out of range on x.

        On a ``4 x 7`` map the coordinate ``(x=6, y=3)`` is valid, but its
        transpose ``(x=3, y=6)`` is not -- ``y=6`` clamps to ``3``. So the
        transposed implementation returns ``(3, 3)`` where the correct one
        returns ``(6, 3)``, and the two are distinguishable. At ``H == W`` they
        would not be.
        """
        height, width = 4, 7
        rows, cols = np.meshgrid(
            np.arange(float(height)), np.arange(float(width)), indexing="ij"
        )
        fmap = np.stack([cols, rows], axis=-1)[None].astype("float32")
        pix = np.array([[[[6.0, 3.0]]]], dtype="float32")

        out = keras.ops.convert_to_numpy(sample_at_pixel_coords(fmap, pix))

        assert out[0, 0, 0, 0] == pytest.approx(6.0, abs=ATOL)
        assert out[0, 0, 0, 0] != pytest.approx(3.0, abs=ATOL)


class TestCoordsGridChannelOrder:
    """``coords_grid`` emits ``(x, y)``. Fixture is NON-SQUARE, deliberately."""

    def test_channel_zero_is_x_and_channel_one_is_y(self):
        """At ``(row=1, col=3)`` the value is ``(3, 1)``, not ``(1, 3)``.

        This is the double-reverse guard (findings G-1). A port that applies
        ``model.py:26-27``'s ``coords[::-1]`` on top of ``keras.ops.meshgrid``'s
        already-``'xy'`` default emits ``(1, 3)`` here.
        """
        grid = keras.ops.convert_to_numpy(coords_grid(1, 4, 7))

        assert grid.shape == (1, 4, 7, 2)
        np.testing.assert_allclose(grid[0, 1, 3], [3.0, 1.0], rtol=0, atol=ATOL)

    def test_the_x_channel_spans_the_width_and_the_y_channel_the_height(self):
        """Ranges alone separate the two orders on a non-square field."""
        grid = keras.ops.convert_to_numpy(coords_grid(1, 4, 7))

        assert grid[..., 0].min() == 0.0 and grid[..., 0].max() == 6.0
        assert grid[..., 1].min() == 0.0 and grid[..., 1].max() == 3.0

    def test_the_field_is_tiled_identically_across_the_batch(self):
        grid = keras.ops.convert_to_numpy(coords_grid(3, 4, 7))

        assert grid.shape == (3, 4, 7, 2)
        np.testing.assert_array_equal(grid[0], grid[1])
        np.testing.assert_array_equal(grid[0], grid[2])

    def test_a_symbolic_batch_size_is_accepted(self):
        """The rectifier passes ``keras.ops.shape(x)[0]``, not a Python int."""
        dummy = keras.ops.zeros((2, 4, 7, 3))
        grid = coords_grid(keras.ops.shape(dummy)[0], 4, 7)

        assert keras.ops.convert_to_numpy(grid).shape == (2, 4, 7, 2)


class TestTheIdentityGridIsTheSamplersIdentity:
    """``sample(fmap, coords_grid(...)) == fmap``, the composition guard.

    This is the one test that would catch a coordinated error in BOTH functions
    -- a channel-order mistake in ``coords_grid`` cancelled by the same mistake
    in the adapter's swap. It is run non-square so the cancellation cannot hide.
    """

    def test_sampling_at_the_identity_grid_returns_the_feature_map(self):
        rng = np.random.default_rng(0)
        fmap = rng.standard_normal((2, 4, 7, 3)).astype("float32")

        out = keras.ops.convert_to_numpy(
            sample_at_pixel_coords(fmap, coords_grid(2, 4, 7))
        )

        np.testing.assert_allclose(out, fmap, rtol=0, atol=ATOL)

    def test_a_pure_integer_shift_of_the_identity_grid_moves_content_one_column(self):
        """Direction, not just magnitude: ``+1`` on x reads the column to the RIGHT.

        A backward map is a *gather*: output pixel ``(x, y)`` takes its value
        from input pixel ``map[y, x]``. So adding ``+1`` to the x channel makes
        every output column show what its right-hand neighbour showed. A sign
        error produces a shift of the same magnitude in the other direction and
        is invisible to any test that only measures ``|shift|``.
        """
        rng = np.random.default_rng(1)
        fmap = rng.standard_normal((1, 4, 7, 1)).astype("float32")

        shift = np.zeros((1, 4, 7, 2), dtype="float32")
        shift[..., 0] = 1.0
        shifted = keras.ops.convert_to_numpy(
            sample_at_pixel_coords(fmap, coords_grid(1, 4, 7) + shift)
        )

        np.testing.assert_allclose(
            shifted[0, :, :6, 0], fmap[0, :, 1:, 0], rtol=0, atol=ATOL
        )
        # The last column has no right-hand neighbour: it clamps to itself.
        np.testing.assert_allclose(
            shifted[0, :, 6, 0], fmap[0, :, 6, 0], rtol=0, atol=ATOL
        )


class TestTheAdapterAtTheRefinementResolution:
    """The stride the rectifier actually runs its loop at.

    ``SPATIAL_DIVISOR`` is imported rather than written as ``8``: the divisor is
    a transcribed upstream constant (``model.py:48-49``) and this file must move
    with it, not pin a second copy of it.
    """

    def test_the_identity_holds_at_one_over_the_spatial_divisor(self):
        """Same code, stride-8 extents -- the functions are stride-agnostic."""
        height, width = 32, 56
        low_h, low_w = height // SPATIAL_DIVISOR, width // SPATIAL_DIVISOR
        assert (low_h, low_w) == (4, 7)

        rng = np.random.default_rng(2)
        fmap = rng.standard_normal((1, low_h, low_w, 5)).astype("float32")

        out = keras.ops.convert_to_numpy(
            sample_at_pixel_coords(fmap, coords_grid(1, low_h, low_w))
        )

        np.testing.assert_allclose(out, fmap, rtol=0, atol=ATOL)

    def test_the_full_resolution_grid_spans_the_divisor_times_the_low_one(self):
        """``coodslar`` and ``coords0`` differ only in extent (``model.py:47-49``)."""
        low = keras.ops.convert_to_numpy(coords_grid(1, 4, 7))
        full = keras.ops.convert_to_numpy(
            coords_grid(1, 4 * SPATIAL_DIVISOR, 7 * SPATIAL_DIVISOR)
        )

        assert full[..., 0].max() == (7 * SPATIAL_DIVISOR) - 1
        assert low[..., 0].max() == 7 - 1


class TestGradientsReachTheCoordinates:
    """The refinement loop trains THROUGH the sampler, not around it.

    ``warpfea = sample_at_pixel_coords(fmap1, coords1)`` (``model.py:92``) is
    inside the unrolled 12-iteration loop, so a sampler that is differentiable
    only w.r.t. its VALUES would leave the coordinate branch of the graph
    untrained -- and would still run, still produce finite output, and still
    lower the loss through the value path. Nothing but a gradient assertion on
    ``coords`` can see that.
    """

    def test_gradients_flow_to_the_pixel_coordinates(self):
        rng = np.random.default_rng(3)
        fmap = tf.constant(rng.standard_normal((1, 4, 7, 3)), dtype=tf.float32)
        pix = tf.Variable(
            keras.ops.convert_to_numpy(coords_grid(1, 4, 7)) + 0.25, dtype=tf.float32
        )

        with tf.GradientTape() as tape:
            out = sample_at_pixel_coords(fmap, pix)
            loss = tf.reduce_sum(out ** 2)

        grad = tape.gradient(loss, pix)

        assert grad is not None
        grad = grad.numpy()
        assert np.all(np.isfinite(grad))
        assert np.abs(grad).max() > 0.0
        # Both channels move: a swap-and-drop bug could leave one axis dead.
        assert np.abs(grad[..., 0]).max() > 0.0
        assert np.abs(grad[..., 1]).max() > 0.0

    def test_gradients_are_finite_at_an_out_of_range_coordinate(self):
        """Clamping must not produce NaN where a badly-initialised net wanders."""
        rng = np.random.default_rng(4)
        fmap = tf.constant(rng.standard_normal((1, 4, 7, 3)), dtype=tf.float32)
        pix = tf.Variable(
            np.full((1, 4, 7, 2), -50.0, dtype="float32"), dtype=tf.float32
        )

        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(sample_at_pixel_coords(fmap, pix) ** 2)

        grad = tape.gradient(loss, pix)

        assert grad is not None
        assert np.all(np.isfinite(grad.numpy()))
