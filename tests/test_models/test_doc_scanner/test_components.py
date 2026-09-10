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
    CONVEX_NEIGHBOURS,
    convex_upsample,
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



# =====================================================================
# convex_upsample
# =====================================================================
#
# The 3x3 neighbour index `k` used throughout this section is the row-major
# `(kh, kw)` index of `F.unfold`/`extract_patches`, so neighbour `k` of source
# pixel `(h, w)` is the input pixel `(h + k // 3 - 1, w + k % 3 - 1)`:
#
#     k = 0 1 2      (dy, dx) = (-1,-1) (-1, 0) (-1,+1)
#         3 4 5                 ( 0,-1) ( 0, 0) ( 0,+1)
#         6 7 8                 (+1,-1) (+1, 0) (+1,+1)
#
# k = 4 is the centre. A guard that only exercises k = 4 cannot distinguish a
# transposed 3x3 from a correct one, which is why every k is exercised below.

_UP_H, _UP_W = 4, 7  # NON-SQUARE, deliberately -- see rule 1 in the module docstring.
_MASK_CHANNELS = CONVEX_NEIGHBOURS * SPATIAL_DIVISOR * SPATIAL_DIVISOR
_VERY_NEGATIVE = -1.0e4  # softmax of this against 0.0 is one-hot to ~1e-4343.


def _neighbour_offset(k: int) -> tuple:
    """``k`` -> ``(dy, dx)``, the row-major 3x3 offset. Written out, not derived
    from the implementation under test."""
    return ((-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 0), (0, 1),
            (1, -1), (1, 0), (1, 1))[k]


def _one_hot_mask(k: int) -> np.ndarray:
    """A ``(1, H, W, 576)`` logit mask that selects neighbour ``k`` everywhere.

    Built in the ``(neighbour, sub_row, sub_col)`` decomposition and then
    flattened, so a wrongly-ordered reshape inside ``convex_upsample`` reads a
    DIFFERENT neighbour per sub-pixel and smears the impulse.
    """
    logits = np.full(
        (1, _UP_H, _UP_W, CONVEX_NEIGHBOURS, SPATIAL_DIVISOR, SPATIAL_DIVISOR),
        _VERY_NEGATIVE,
        dtype="float32",
    )
    logits[:, :, :, k, :, :] = 0.0
    return logits.reshape(1, _UP_H, _UP_W, _MASK_CHANNELS)


def _impulse_flow(row: int, col: int, channel: int = 0) -> np.ndarray:
    flow = np.zeros((1, _UP_H, _UP_W, 2), dtype="float32")
    flow[0, row, col, channel] = 1.0
    return flow


class TestConvexUpsampleNeighbourOrdering:
    """The delta-impulse ordering guard: where does the mass LAND?

    One non-zero neighbour weight at a known 3x3 offset must move the impulse
    to exactly the destination block that offset predicts. This is the only
    instrument that can see the two reshape orderings, both of which are
    shape-preserving.
    """

    @pytest.mark.parametrize("k", list(range(CONVEX_NEIGHBOURS)))
    def test_a_one_hot_neighbour_moves_the_impulse_by_that_offset(self, k):
        src_row, src_col = 2, 3  # interior, so every one of the 9 offsets is in range.
        out = np.array(
            convex_upsample(_impulse_flow(src_row, src_col), _one_hot_mask(k))
        )

        assert out.shape == (
            1,
            _UP_H * SPATIAL_DIVISOR,
            _UP_W * SPATIAL_DIVISOR,
            2,
        )

        dy, dx = _neighbour_offset(k)
        # Neighbour k of source (h, w) is input (h + dy, w + dx); the impulse at
        # (src_row, src_col) is therefore READ by source pixel
        # (src_row - dy, src_col - dx), whose whole 8x8 destination block lights up.
        dst_h, dst_w = src_row - dy, src_col - dx

        expected = np.zeros_like(out)
        expected[
            0,
            dst_h * SPATIAL_DIVISOR : (dst_h + 1) * SPATIAL_DIVISOR,
            dst_w * SPATIAL_DIVISOR : (dst_w + 1) * SPATIAL_DIVISOR,
            0,
        ] = float(SPATIAL_DIVISOR)

        np.testing.assert_allclose(out, expected, rtol=0, atol=1e-3)

    def test_the_centre_neighbour_is_a_plain_nearest_upsample(
            self, golden_reference_device):
        """k = 4 alone: a pure block-replicating 8x upsample of ``8 * flow``.

        Pinned to the golden-reference device. This is the only probe in the
        module whose expectation is an arbitrary float rather than 0 or a small
        integer, and on an RTX 4070 with TF32 matmul enabled it reads
        ``0.273438`` (= 35/128, a 10-bit mantissa) against a true ``0.273542``
        -- a 3.3e-3 error that is precision, not ordering. Measured
        2026-09-10 during step 4; the other arms are exact at any precision and
        do not need the pin.
        """
        rng = np.random.default_rng(11)
        flow = rng.standard_normal((1, _UP_H, _UP_W, 2)).astype("float32")

        with keras.device(golden_reference_device):
            out = np.array(convex_upsample(flow, _one_hot_mask(4)))
        expected = np.repeat(
            np.repeat(flow * SPATIAL_DIVISOR, SPATIAL_DIVISOR, axis=1),
            SPATIAL_DIVISOR,
            axis=2,
        )

        np.testing.assert_allclose(out, expected, rtol=0, atol=1e-3)

    def test_an_offset_neighbour_reads_zero_across_the_border(self):
        """``F.unfold(..., padding=1)`` is ZERO-padded, not edge-clamped.

        Distinct from :func:`sample_at_pixel_coords`, which clamps. Selecting
        the 'up' neighbour on the top row must therefore yield zeros.
        """
        flow = np.ones((1, _UP_H, _UP_W, 2), dtype="float32")
        out = np.array(convex_upsample(flow, _one_hot_mask(1)))  # (dy, dx) = (-1, 0)

        top_block = out[0, :SPATIAL_DIVISOR, :, :]
        np.testing.assert_allclose(top_block, 0.0, rtol=0, atol=1e-3)
        # ...and the row below it reads the (all-ones) row above, scaled by 8.
        np.testing.assert_allclose(
            out[0, SPATIAL_DIVISOR : 2 * SPATIAL_DIVISOR, :, :],
            float(SPATIAL_DIVISOR),
            rtol=0,
            atol=1e-3,
        )


class TestConvexUpsampleSubPixelInterleave:
    """Which of the two 8s is the sub-ROW and which is the sub-COLUMN.

    Swapping them transposes every 8x8 block. Output shape, dtype and
    finiteness are all identical, and at the square 288x288 training resolution
    the swapped variant is not even out of range -- so only an ASYMMETRIC
    sub-pixel probe can see it.
    """

    def test_one_sub_pixel_selecting_a_different_neighbour_lands_asymmetrically(self):
        sub_row, sub_col = 1, 0  # asymmetric: (1, 0) and (0, 1) are distinguishable.

        logits = np.full(
            (1, _UP_H, _UP_W, CONVEX_NEIGHBOURS, SPATIAL_DIVISOR, SPATIAL_DIVISOR),
            _VERY_NEGATIVE,
            dtype="float32",
        )
        logits[:, :, :, 4, :, :] = 0.0  # everything selects the centre...
        logits[:, :, :, 4, sub_row, sub_col] = _VERY_NEGATIVE
        logits[:, :, :, 1, sub_row, sub_col] = 0.0  # ...except this one, which selects 'up'.
        mask = logits.reshape(1, _UP_H, _UP_W, _MASK_CHANNELS)

        src_row, src_col = 2, 3
        out = np.array(convex_upsample(_impulse_flow(src_row, src_col), mask))

        # The centre-selecting sub-pixels of block (2, 3) see the impulse...
        centre_block = out[
            0,
            src_row * SPATIAL_DIVISOR : (src_row + 1) * SPATIAL_DIVISOR,
            src_col * SPATIAL_DIVISOR : (src_col + 1) * SPATIAL_DIVISOR,
            0,
        ]
        assert centre_block[0, 0] == pytest.approx(SPATIAL_DIVISOR, abs=1e-3)
        # ...but the one deviant sub-pixel does not: it reads (src_row - 1, src_col).
        assert centre_block[sub_row, sub_col] == pytest.approx(0.0, abs=1e-3)

        # And the impulse reappears one BLOCK down, at sub-pixel (1, 0) of it --
        # NOT at (0, 1), which is where a swapped interleave would put it.
        below = out[
            0,
            (src_row + 1) * SPATIAL_DIVISOR : (src_row + 2) * SPATIAL_DIVISOR,
            src_col * SPATIAL_DIVISOR : (src_col + 1) * SPATIAL_DIVISOR,
            0,
        ]
        assert below[sub_row, sub_col] == pytest.approx(SPATIAL_DIVISOR, abs=1e-3)
        assert below[sub_col, sub_row] == pytest.approx(0.0, abs=1e-3)


class TestConvexUpsampleUniformMaskControl:
    """With every logit equal, the operator degenerates to a known filter.

    The expected value is computed independently in numpy -- an explicit
    zero-padded 3x3 mean followed by a block-replicating 8x upsample -- never by
    calling :func:`convex_upsample`.
    """

    @staticmethod
    def _reference(flow: np.ndarray) -> np.ndarray:
        scaled = flow * SPATIAL_DIVISOR
        padded = np.pad(scaled, ((0, 0), (1, 1), (1, 1), (0, 0)))
        mean = np.zeros_like(scaled)
        for h in range(scaled.shape[1]):
            for w in range(scaled.shape[2]):
                mean[:, h, w, :] = padded[:, h : h + 3, w : w + 3, :].sum(
                    axis=(1, 2)
                ) / float(CONVEX_NEIGHBOURS)
        return np.repeat(
            np.repeat(mean, SPATIAL_DIVISOR, axis=1), SPATIAL_DIVISOR, axis=2
        )

    def test_a_uniform_mask_is_a_mean_filtered_nearest_upsample(self):
        rng = np.random.default_rng(7)
        flow = rng.standard_normal((2, _UP_H, _UP_W, 2)).astype("float32")
        mask = np.full((2, _UP_H, _UP_W, _MASK_CHANNELS), 0.3, dtype="float32")

        out = np.array(convex_upsample(flow, mask))

        np.testing.assert_allclose(out, self._reference(flow), rtol=0, atol=1e-5)

    def test_the_weights_are_convex_so_a_constant_field_is_attenuated_at_the_border(self):
        """Non-negative and summing to one: a constant interior stays constant.

        The border does NOT, because the padding is zero rather than reflective
        -- so this also pins the padding mode against a uniform mask.
        """
        flow = np.full((1, _UP_H, _UP_W, 2), 0.5, dtype="float32")
        mask = np.zeros((1, _UP_H, _UP_W, _MASK_CHANNELS), dtype="float32")

        out = np.array(convex_upsample(flow, mask))

        interior = out[
            0, SPATIAL_DIVISOR : -SPATIAL_DIVISOR, SPATIAL_DIVISOR : -SPATIAL_DIVISOR, :
        ]
        np.testing.assert_allclose(
            interior, 0.5 * SPATIAL_DIVISOR, rtol=0, atol=1e-5
        )
        # A corner sees only 4 of its 9 neighbours.
        assert out[0, 0, 0, 0] == pytest.approx(
            0.5 * SPATIAL_DIVISOR * 4.0 / CONVEX_NEIGHBOURS, abs=1e-5
        )


class TestConvexUpsampleTakesRawLogits:
    """The ``0.25`` of ``update.py:104`` belongs to the mask head, not here."""

    def test_scaling_the_logits_changes_the_output(self):
        rng = np.random.default_rng(23)
        flow = rng.standard_normal((1, _UP_H, _UP_W, 2)).astype("float32")
        logits = rng.standard_normal(
            (1, _UP_H, _UP_W, _MASK_CHANNELS)
        ).astype("float32") * 4.0

        sharp = np.array(convex_upsample(flow, logits))
        soft = np.array(convex_upsample(flow, logits * 0.25))

        # If this function silently applied (or absorbed) a scaling of its own,
        # one of these would be a no-op. Softmax is not scale-invariant.
        assert np.max(np.abs(sharp - soft)) > 1e-3


class TestConvexUpsampleGradientsAndGraphSafety:

    def test_gradients_reach_both_the_flow_and_the_mask(self):
        rng = np.random.default_rng(5)
        flow = tf.Variable(
            rng.standard_normal((1, _UP_H, _UP_W, 2)).astype("float32")
        )
        mask = tf.Variable(
            rng.standard_normal((1, _UP_H, _UP_W, _MASK_CHANNELS)).astype("float32")
        )

        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(convex_upsample(flow, mask) ** 2)

        g_flow, g_mask = tape.gradient(loss, [flow, mask])

        assert g_flow is not None and g_mask is not None
        assert np.all(np.isfinite(g_flow.numpy()))
        assert np.all(np.isfinite(g_mask.numpy()))
        # A non-trivial mask gradient is the real assertion: a `stop_gradient`
        # or an argmax-style selection would leave this identically zero.
        assert np.max(np.abs(g_mask.numpy())) > 0.0

    def test_it_runs_under_a_traced_tf_function_with_an_unknown_batch(self):
        """No Python branch on a symbolic value; the reshape must stay dynamic."""
        traced = tf.function(
            convex_upsample,
            input_signature=[
                tf.TensorSpec([None, _UP_H, _UP_W, 2], tf.float32),
                tf.TensorSpec([None, _UP_H, _UP_W, _MASK_CHANNELS], tf.float32),
            ],
        )
        out = traced(
            tf.zeros((3, _UP_H, _UP_W, 2)),
            tf.zeros((3, _UP_H, _UP_W, _MASK_CHANNELS)),
        )
        assert tuple(out.shape) == (
            3,
            _UP_H * SPATIAL_DIVISOR,
            _UP_W * SPATIAL_DIVISOR,
            2,
        )


# =====================================================================
# The feature encoder: the 80-channel stem norm, the torch-exact stride-2
# padding, the shortcut algebra, gradients and the round trip.
#
# Everything below is pointed at a defect class that a shape test cannot see:
#
# * `norm1` built at the reference's literal 64 instead of the 80 channels
#   `conv1` actually emits (H-7 / F-02). Upstream this is INERT, because
#   `nn.InstanceNorm2d` defaults to `affine=False` and never uses the argument;
#   here the argument is the GROUP COUNT, so it either raises (64 does not
#   divide 80) or -- for a divisor that does, like 16 -- silently normalizes in
#   5-channel groups while every shape, dtype, gradient and serialization test
#   stays green. Hence the guards below assert the group count AND the
#   per-channel statistics, never just the output shape.
# * a stride-2 convolution padded with Keras "same" instead of torch's
#   symmetric `padding=k//2`. Identical output SHAPE, feature map shifted by one
#   pixel (D-012).
# =====================================================================

from dl_techniques.models.vision.image_restoration.doc_scanner.components import (  # noqa: E402
    INSTANCE_NORM_EPSILON,
    _VARIANT_SPEC,
    DocScannerFeatureEncoder,
    DocScannerResidualBlock,
    _instance_norm,
)
from ..gradient_flow_oracle import (  # noqa: E402
    assert_gradients_reach_every_trainable_weight,
)
from ..roundtrip_instrument_oracle import (  # noqa: E402
    assert_roundtrip_output_values,
    assert_weights_restored_before_first_call,
    measure_roundtrip,
)

_SPEC = _VARIANT_SPEC["docscanner-l"]
_STEM_CHANNELS = _SPEC["encoder_stem_channels"]
_STAGE_CHANNELS = _SPEC["encoder_stage_channels"]
_FNET_OUTPUT_DIM = _SPEC["fnet_output_dim"]

# The reference's literal, kept here ONLY so the guards can say what must not
# be built. `extractor.py:90` / `:93`.
_REFERENCE_NORM1_LITERAL = 64


def _encoder() -> DocScannerFeatureEncoder:
    """The encoder at the one shipped variant's widths. No literal enters here."""
    return DocScannerFeatureEncoder(
        stem_channels=_STEM_CHANNELS,
        stage_channels=_STAGE_CHANNELS,
        output_dim=_FNET_OUTPUT_DIM,
    )


def _built_encoder(height: int = 64, width: int = 48) -> DocScannerFeatureEncoder:
    encoder = _encoder()
    encoder.build((None, height, width, 3))
    return encoder


class TestTheStemNormIsEightyWideNotSixtyFour:
    """H-7. The reference's ``InstanceNorm2d(64)`` is fed 80 channels.

    ``extractor.py:93`` constructs the norm with 64; ``extractor.py:95``
    constructs the convolution that feeds it with 80 outputs. Torch never
    notices because ``affine=False`` allocates nothing. This port must use 80.
    """

    def test_the_group_count_is_the_stem_width_and_not_the_references_literal(self):
        encoder = _built_encoder()
        assert encoder.norm1.groups == _STEM_CHANNELS
        assert encoder.norm1.groups != _REFERENCE_NORM1_LITERAL, (
            "norm1 was built at the reference's 64. That literal is inert in "
            "torch (affine=False allocates no parameters) but is the GROUP "
            "COUNT here. See extractor.py:93 vs :95 and decisions.md D-011.")

    def test_the_group_count_matches_what_the_stem_convolution_actually_emits(self):
        """The two are read from the same place, so they cannot drift apart."""
        encoder = _built_encoder()
        assert encoder.conv1.filters == encoder.norm1.groups

    def test_all_eighty_channels_are_normalized_independently(self):
        """The numerical form of the same claim, for a divisor that DOES divide 80.

        ``groups=16`` divides 80 and would raise nothing; it would normalize in
        5-channel groups. Only a per-channel statistic can see that, so this is
        the guard that survives when the crash-shaped one does not.
        """
        channels = _STEM_CHANNELS
        rng = np.random.default_rng(0)
        # Deliberately different per-channel mean and scale, so a grouped
        # normalization CANNOT accidentally produce per-channel zero mean.
        offsets = np.arange(channels, dtype="float32") * 3.0 - 40.0
        scales = 1.0 + np.arange(channels, dtype="float32") * 0.25
        sample = rng.standard_normal((2, 9, 7, channels)).astype("float32")
        sample = sample * scales + offsets

        norm = _instance_norm(channels, name="probe")
        norm.build((None, 9, 7, channels))
        out = np.asarray(keras.ops.convert_to_numpy(norm(sample)))

        per_channel_mean = out.mean(axis=(1, 2))
        per_channel_var = out.var(axis=(1, 2))
        np.testing.assert_allclose(
            per_channel_mean, np.zeros_like(per_channel_mean),
            atol=1e-4, rtol=0,
            err_msg="a channel was not normalized on its own; the group count "
                    "does not equal the channel count")
        np.testing.assert_allclose(
            per_channel_var, np.ones_like(per_channel_var),
            atol=1e-3, rtol=0,
            err_msg="a channel's variance is not 1; channels are being pooled")

    def test_the_norm_is_not_affine_matching_torchs_default(self):
        """D-011. ``nn.InstanceNorm2d`` defaults to ``affine=False``.

        Zero weights is the whole point: it is *why* the reference's 64 is inert
        upstream. An affine port would be a divergence, so it is asserted, not
        assumed.
        """
        encoder = _built_encoder()
        assert len(encoder.norm1.weights) == 0
        assert encoder.norm1.center is False
        assert encoder.norm1.scale is False

    def test_the_epsilon_is_torchs_and_not_the_keras_default(self):
        """1e-5, not GroupNormalization's own 1e-3 -- a silent 100x."""
        encoder = _built_encoder()
        assert encoder.norm1.epsilon == INSTANCE_NORM_EPSILON
        assert encoder.norm1.epsilon != 1e-3


class TestTheStrideTwoPaddingIsTorchsAndNotKerasSame:
    """D-012. Same output shape, one-pixel-shifted sampling grid.

    A 7x7 stride-2 convolution with torch's ``padding=3`` samples input centres
    at ``2j``. Keras/TF ``"same"`` splits its 5 pad columns as 2/3 and samples
    ``2j + 1``. Both give ``ceil(H / 2)`` outputs, so no shape assertion can
    tell them apart. These two tests are two-sided: an impulse at an even index
    must be SEEN and an impulse at the neighbouring odd index must NOT be.
    """

    @staticmethod
    def _centre_tap_stem(encoder: DocScannerFeatureEncoder) -> None:
        """Make ``conv1`` a single tap at the kernel centre, bias 0."""
        kernel = np.zeros(encoder.conv1.kernel.shape, dtype="float32")
        kernel[3, 3, 0, 0] = 1.0
        encoder.conv1.kernel.assign(kernel)
        encoder.conv1.bias.assign(
            np.zeros(encoder.conv1.bias.shape, dtype="float32"))

    @pytest.mark.parametrize("row,col", [(2, 4), (4, 2), (6, 6)])
    def test_the_stem_reads_an_impulse_at_an_even_index_into_half_that_index(
            self, row, col):
        encoder = _built_encoder(height=16, width=16)
        self._centre_tap_stem(encoder)

        impulse = np.zeros((1, 16, 16, 3), dtype="float32")
        impulse[0, row, col, 0] = 1.0
        out = np.asarray(keras.ops.convert_to_numpy(
            encoder.conv1(encoder.pad1(impulse))))[0, :, :, 0]

        assert out[row // 2, col // 2] == pytest.approx(1.0), (
            "torch's padding=3 puts the kernel centre of output j on input 2j; "
            "Keras 'same' puts it on 2j+1. See decisions.md D-012.")
        assert out.sum() == pytest.approx(1.0)

    @pytest.mark.parametrize("row,col", [(3, 5), (5, 3)])
    def test_the_stem_does_not_see_an_impulse_at_an_odd_index(self, row, col):
        """The other side of the same claim: under Keras 'same' THIS would fire."""
        encoder = _built_encoder(height=16, width=16)
        self._centre_tap_stem(encoder)

        impulse = np.zeros((1, 16, 16, 3), dtype="float32")
        impulse[0, row, col, 0] = 1.0
        out = np.asarray(keras.ops.convert_to_numpy(
            encoder.conv1(encoder.pad1(impulse))))[0, :, :, 0]

        assert out.sum() == pytest.approx(0.0, abs=1e-6), (
            "an odd-indexed impulse reached a stride-2 output centre; the "
            "padding is Keras 'same', not torch's symmetric padding")

    def test_the_residual_blocks_stride_two_conv_has_the_same_alignment(self):
        """The 3x3 stride-2 branch conv, same claim, ``padding=1``."""
        block = DocScannerResidualBlock(filters=4, stride=2)
        block.build((None, 8, 8, 3))
        kernel = np.zeros(block.conv1.kernel.shape, dtype="float32")
        kernel[1, 1, 0, 0] = 1.0
        block.conv1.kernel.assign(kernel)
        block.conv1.bias.assign(np.zeros(block.conv1.bias.shape, dtype="float32"))

        even = np.zeros((1, 8, 8, 3), dtype="float32")
        even[0, 4, 2, 0] = 1.0
        odd = np.zeros((1, 8, 8, 3), dtype="float32")
        odd[0, 5, 3, 0] = 1.0

        even_out = np.asarray(keras.ops.convert_to_numpy(
            block.conv1(block.pad1(even))))[0, :, :, 0]
        odd_out = np.asarray(keras.ops.convert_to_numpy(
            block.conv1(block.pad1(odd))))[0, :, :, 0]

        assert even_out[2, 1] == pytest.approx(1.0)
        assert odd_out.sum() == pytest.approx(0.0, abs=1e-6)

    def test_the_projection_shortcut_subsamples_the_even_indices(self):
        """The 1x1 stride-2 shortcut must read input ``2j``, like torch's."""
        block = DocScannerResidualBlock(filters=3, stride=2)
        block.build((None, 7, 9, 3))
        eye = np.zeros(block.downsample_conv.kernel.shape, dtype="float32")
        for index in range(3):
            eye[0, 0, index, index] = 1.0
        block.downsample_conv.kernel.assign(eye)
        block.downsample_conv.bias.assign(
            np.zeros(block.downsample_conv.bias.shape, dtype="float32"))

        rng = np.random.default_rng(3)
        sample = rng.standard_normal((2, 7, 9, 3)).astype("float32")
        out = np.asarray(keras.ops.convert_to_numpy(
            block.downsample_conv(sample)))
        np.testing.assert_allclose(out, sample[:, ::2, ::2, :], atol=0, rtol=0)


class TestTheResidualBlockShortcut:
    """``extractor.py:26-30, 36-39``: identity at stride 1, projection at stride 2."""

    def test_a_stride_one_block_has_no_projection_at_all(self):
        block = DocScannerResidualBlock(filters=8, stride=1)
        assert block.downsample_conv is None
        assert block.norm3 is None
        assert block.pad1 is None, (
            "stride 1 needs no explicit pad: Keras 'same' IS torch padding=1 "
            "there. A pad layer here would shift the identity path.")

    def test_a_stride_two_block_has_a_projection_and_its_own_norm(self):
        block = DocScannerResidualBlock(filters=8, stride=2)
        assert block.downsample_conv is not None
        assert block.norm3 is not None

    def test_the_stride_one_shortcut_is_the_RAW_input_not_a_projection(self):
        """Zero the branch and the block must reduce to ``relu(x)`` exactly.

        A projection shortcut -- even an accidentally-added one -- would put a
        convolution and a normalization on this path, and the equality below
        would fail by an amount no shape test could report.
        """
        block = DocScannerResidualBlock(filters=5, stride=1)
        block.build((None, 6, 4, 5))
        for conv in (block.conv1, block.conv2):
            conv.kernel.assign(np.zeros(conv.kernel.shape, dtype="float32"))
            conv.bias.assign(np.zeros(conv.bias.shape, dtype="float32"))

        rng = np.random.default_rng(11)
        sample = rng.standard_normal((2, 6, 4, 5)).astype("float32")
        out = np.asarray(keras.ops.convert_to_numpy(block(sample)))
        np.testing.assert_allclose(out, np.maximum(sample, 0.0), atol=0, rtol=0)

    def test_a_stride_one_block_refuses_a_channel_change(self):
        """The identity add is illegal there, and the reference never builds one."""
        block = DocScannerResidualBlock(filters=8, stride=1)
        with pytest.raises(ValueError, match="IDENTITY shortcut"):
            block.build((None, 6, 6, 16))

    def test_the_second_relu_is_inside_the_branch_before_the_add(self):
        """``extractor.py:34,39``: ``relu(norm2(conv2(y)))`` and then ``relu(x + y)``.

        The usual ResNet ordering adds the UN-activated branch. Under that
        ordering the branch can be negative, so a strictly-negative input plus
        a branch tuned to cancel it would leave 0; under the reference's
        ordering the branch is already non-negative, so it cannot cancel and
        the block's output is >= relu(x) everywhere.
        """
        block = DocScannerResidualBlock(filters=6, stride=1)
        block.build((None, 5, 5, 6))
        rng = np.random.default_rng(5)
        sample = rng.standard_normal((3, 5, 5, 6)).astype("float32")
        out = np.asarray(keras.ops.convert_to_numpy(block(sample)))
        assert (out >= np.maximum(sample, 0.0) - 1e-5).all(), (
            "the branch went negative before the add, so the second ReLU is "
            "outside it; the reference applies one on the branch AND one after")


class TestTheEncoderShapeLadder:
    """Stem 2, stages 1/2/2 -- total stride 8, and 320 output channels."""

    def test_a_square_288_input_lands_at_36_by_36_by_320(self):
        encoder = _encoder()
        out = encoder(np.zeros((2, 288, 288, 3), dtype="float32"))
        assert tuple(out.shape) == (2, 36, 36, _FNET_OUTPUT_DIM)
        assert 288 // SPATIAL_DIVISOR == 36

    def test_a_non_square_input_keeps_height_and_width_distinct(self):
        """288x224 -> 36x28. A square fixture cannot see an H/W transpose."""
        encoder = _encoder()
        out = encoder(np.zeros((2, 288, 224, 3), dtype="float32"))
        assert tuple(out.shape) == (2, 36, 28, _FNET_OUTPUT_DIM)

    def test_compute_output_shape_agrees_with_the_real_forward(self):
        encoder = _encoder()
        declared = encoder.compute_output_shape((None, 288, 224, 3))
        out = encoder(np.zeros((2, 288, 224, 3), dtype="float32"))
        assert declared[1:] == tuple(out.shape)[1:]

    def test_the_stage_ladder_is_two_blocks_each_with_the_stride_on_the_first(self):
        """``extractor.py:99-101, 115-118``."""
        encoder = _encoder()
        assert [block.filters for block in encoder.blocks] == [
            _STAGE_CHANNELS[0], _STAGE_CHANNELS[0],
            _STAGE_CHANNELS[1], _STAGE_CHANNELS[1],
            _STAGE_CHANNELS[2], _STAGE_CHANNELS[2],
        ]
        assert [block.stride for block in encoder.blocks] == [1, 1, 2, 1, 2, 1]

    def test_the_output_projection_carries_no_norm_and_no_activation(self):
        """``extractor.py:104``, ``:131``: the 1x1 is bare.

        The rectifier splits these 320 channels into a ``tanh`` hidden state and
        a ``relu`` context; a ReLU here would half-rectify the hidden state
        before ``tanh`` ever saw it. Negative outputs are the observable.
        """
        encoder = _encoder()
        rng = np.random.default_rng(7)
        out = np.asarray(keras.ops.convert_to_numpy(
            encoder(rng.standard_normal((2, 64, 48, 3)).astype("float32"))))
        assert (out < 0.0).any(), "the encoder output is non-negative; something " \
                                  "rectifying was appended to the output 1x1"

    def test_a_stage_count_that_breaks_the_stride_contract_is_refused(self):
        """The rectifier upsamples by exactly SPATIAL_DIVISOR; the two must agree."""
        with pytest.raises(ValueError, match="total stride"):
            DocScannerFeatureEncoder(
                stem_channels=_STEM_CHANNELS,
                stage_channels=_STAGE_CHANNELS[:2],
                output_dim=_FNET_OUTPUT_DIM,
            )


def _encoder_functional_model() -> keras.Model:
    """The encoder wrapped so the shared model oracles can judge it."""
    inputs = keras.Input(shape=(32, 24, 3))
    return keras.Model(inputs, _encoder()(inputs), name="doc_scanner_fnet")


def _encoder_inputs() -> np.ndarray:
    return np.linspace(-1.0, 1.0, 2 * 32 * 24 * 3, dtype="float32").reshape(
        (2, 32, 24, 3))


class TestTheEncoderTrainsAndRoundTrips:
    """The shared oracles, adopted rather than reimplemented."""

    def test_every_trainable_weight_receives_a_live_gradient(self):
        model = _encoder_functional_model()
        assert_gradients_reach_every_trainable_weight(
            model, _encoder_inputs(), training=True)

    def test_the_saved_and_reloaded_encoder_reproduces_its_output_exactly(self):
        report = measure_roundtrip(
            _encoder_functional_model, _encoder_inputs, training=False)
        assert report["self_max_delta"] == 0.0, (
            "the encoder became non-deterministic; the exact bound below would "
            "then be measuring that instead of the round trip")
        assert_roundtrip_output_values(report, atol=0.0)

    def test_the_weights_are_restored_before_the_reloaded_model_is_called(self):
        report = measure_roundtrip(
            _encoder_functional_model, _encoder_inputs, training=False)
        assert report["call_count_before_weight_read"] == 0
        assert_weights_restored_before_first_call(report, atol=0.0)

    def test_the_encoder_config_round_trips_every_constructor_argument(self):
        encoder = _encoder()
        clone = DocScannerFeatureEncoder.from_config(encoder.get_config())
        assert clone.stem_channels == encoder.stem_channels
        assert tuple(clone.stage_channels) == tuple(encoder.stage_channels)
        assert clone.output_dim == encoder.output_dim

    def test_the_block_config_round_trips_every_constructor_argument(self):
        block = DocScannerResidualBlock(filters=12, stride=2)
        clone = DocScannerResidualBlock.from_config(block.get_config())
        assert clone.filters == 12
        assert clone.stride == 2

    def test_it_runs_under_a_traced_tf_function_with_an_unknown_batch(self):
        encoder = _encoder()
        encoder.build((None, 32, 24, 3))
        traced = tf.function(
            lambda batch: encoder(batch),
            input_signature=[tf.TensorSpec([None, 32, 24, 3], tf.float32)],
        )
        assert tuple(traced(tf.zeros((3, 32, 24, 3))).shape) == (
            3, 4, 3, _FNET_OUTPUT_DIM)
