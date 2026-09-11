"""Guards for the synthetic warped-page generator.

The one thing this suite exists to pin is **exactness of the emitted backward
map**. The generator renders the distorted image by gathering the flat page
through the distorted-to-flat direction of an exactly-invertible warp, and
emits the other direction as ``f_gt``. Those two are algebraic inverses, so the
pair ``(image, f_gt)`` is consistent to float round-off rather than to whatever
a scattered-data inversion happened to achieve. ``TestTheBackwardMapIsExact``
measures that, and every arm of it has been proven RED against a sub-pixel
perturbation of the emitted map.

The second thing it pins is the **direction**. ``f_gt`` is indexed by the
rectified grid and holds distorted-image coordinates -- the port's own
convention (``DocScannerRectifier`` emits it, ``DocScannerFlowSequenceLoss``
takes it as the first two channels of ``y_true``). Emitting the render map
instead is a silent inversion with no shape, dtype, range or finiteness
symptom, and ``TestTheMapPointsFromTheRectifiedGridIntoTheDistortedImage`` is
the arm that sees it.

Independence of the oracle: the rectification arms resample with
``scipy.ndimage.map_coordinates``, not with the module's own
``_bilinear_gather``, so a bug in that sampler cannot cancel itself out.
"""

import ast
import importlib
import pathlib

import numpy as np
import pytest
from scipy import ndimage

from dl_techniques.datasets.document_rectification import synthetic_warp as sw
from dl_techniques.datasets.document_rectification.synthetic_warp import (
    DegenerateWarpError,
    InvertibleWarp,
    WarpQuality,
    assess_warp,
    generate_sample,
    hsv_to_rgb,
    is_degenerate,
    jitter_hsv,
    map_jacobian_stats,
    render_sample,
    rgb_to_hsv,
    sample_warp,
)

# The exactness budget, MEASURED at 96x128 on seed 3: the emitted `f_gt`
# composed with the warp's closed-form inverse returns to the flat pixel grid
# with a maximum error of 1.3e-05 px. That residual is float32 STORAGE of the
# emitted map, not construction error -- the same composition in float64 closes
# at 7.8e-16. The bound below leaves ~75x headroom and is not a fudge factor.
ROUNDTRIP_ATOL_PIXELS = 1e-3

# End-to-end rectification budget on a bilinear test page, MEASURED: max
# |error| 2.3e-03, mean 5.1e-05, against 9.9e-01 for the un-rectified image.
# This one is genuinely approximate (it is a bilinear resample of an already
# resampled image), which is exactly why the exact guard above is separate.
RECTIFY_MAX_ERROR = 0.02
RECTIFY_MEAN_ERROR = 1e-3


# ---------------------------------------------------------------------------
# Fixtures. A bilinear page is used wherever a resampling error has to be
# attributable: bilinear interpolation reproduces a bilinear function exactly,
# so the FIRST gather contributes no error and only the second one does.
# ---------------------------------------------------------------------------


@pytest.fixture
def page():
    """(96, 128, 3) float page whose content is exactly bilinear."""
    height, width = 96, 128
    cols, rows = np.meshgrid(
        np.arange(width) / (width - 1), np.arange(height) / (height - 1), indexing="xy"
    )
    plane = 0.1 + 0.3 * cols + 0.4 * rows + 0.2 * cols * rows
    return np.stack([plane, plane * 0.8 + 0.1, 1.0 - plane * 0.5], axis=-1)


@pytest.fixture
def background():
    """A small, noisy, deliberately non-page-like texture."""
    return np.random.default_rng(11).random((37, 53, 3))


def _flat_reference(page_array, height, width):
    """The flat page resampled onto the rectified grid, via an independent sampler."""
    uv = sw._flat_uv_grid(height, width)
    page_height, page_width = page_array.shape[:2]
    return np.stack(
        [
            ndimage.map_coordinates(
                page_array[..., channel],
                [uv[..., 1] * page_height, uv[..., 0] * page_width],
                order=1,
                mode="nearest",
            )
            for channel in range(3)
        ],
        axis=-1,
    )


def _rectify(image, f_gt):
    """Gather ``image`` at ``f_gt`` with scipy, i.e. apply the backward map."""
    x = f_gt[..., 0].astype(np.float64)
    y = f_gt[..., 1].astype(np.float64)
    return np.stack(
        [
            ndimage.map_coordinates(
                image[..., channel].astype(np.float64), [y, x], order=1, mode="nearest"
            )
            for channel in range(3)
        ],
        axis=-1,
    )


def _interior(mask, f_gt):
    """Rectified pixels whose whole 2x2 sampling footprint is inside the page."""
    x = f_gt[..., 0].astype(np.float64)
    y = f_gt[..., 1].astype(np.float64)
    sampled = ndimage.map_coordinates(
        mask[..., 0].astype(np.float64), [y, x], order=1, mode="nearest"
    )
    return sampled >= 1.0


# ---------------------------------------------------------------------------
# 1. THE POINT OF THE STEP: the emitted backward map is exact.
# ---------------------------------------------------------------------------


class TestTheBackwardMapIsExact:
    """``f_gt`` is the algebraic inverse of the map the image was rendered with.

    Every arm here goes RED when the emitted map is perturbed by half a pixel
    after rendering -- which is the whole class of defect a shape/dtype/range
    test cannot see.
    """

    def test_the_emitted_map_inverts_the_render_gather_map(self, page, background):
        """Composing the emitted ``f_gt`` with the render map is the identity."""
        height, width = 96, 128
        warp = sample_warp(np.random.default_rng(3))
        _, f_gt, _ = render_sample(
            page, background, warp, (height, width), np.random.default_rng(4)
        )
        extent = np.array([width, height], dtype=np.float64)
        returned = warp.invert(f_gt.reshape(-1, 2).astype(np.float64) / extent) * extent
        grid = sw._flat_uv_grid(height, width).reshape(-1, 2) * extent
        np.testing.assert_allclose(
            returned, grid, rtol=0.0, atol=ROUNDTRIP_ATOL_PIXELS
        )

    def test_the_composition_closes_at_float64_round_off(self):
        """The construction itself is exact; float32 storage is the only loss."""
        warp = sample_warp(np.random.default_rng(7))
        uv = sw._flat_uv_grid(64, 64).reshape(-1, 2)
        np.testing.assert_allclose(warp.invert(warp.apply(uv)), uv, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(warp.apply(warp.invert(uv)), uv, rtol=0.0, atol=1e-12)

    def test_rectifying_the_emitted_image_with_the_emitted_map_returns_the_page(
        self, page, background
    ):
        """End to end, with an independent (scipy) sampler on both sides."""
        height, width = 96, 128
        image, f_gt, mask = generate_sample(
            np.random.default_rng(5), page, background, (height, width), jitter=False
        )
        interior = _interior(mask, f_gt)
        assert interior.mean() > 0.2, "the page barely landed; fixture is not exercising"
        error = np.abs(_rectify(image, f_gt) - _flat_reference(page, height, width))
        error = error[interior]
        assert error.max() < RECTIFY_MAX_ERROR, f"max |err| {error.max()}"
        assert error.mean() < RECTIFY_MEAN_ERROR, f"mean |err| {error.mean()}"

    def test_the_unrectified_image_is_nowhere_near_the_page(self, page, background):
        """Anti-vacuity for the arm above: the tolerance is not trivially met."""
        height, width = 96, 128
        image, f_gt, mask = generate_sample(
            np.random.default_rng(5), page, background, (height, width), jitter=False
        )
        interior = _interior(mask, f_gt)
        raw = np.abs(
            image.astype(np.float64) - _flat_reference(page, height, width)
        )[interior]
        assert raw.max() > 20.0 * RECTIFY_MAX_ERROR, (
            "the warp is so mild that the un-rectified image already passes; "
            f"max |err| {raw.max()}"
        )


class TestTheMapPointsFromTheRectifiedGridIntoTheDistortedImage:
    """The DIRECTION guard: ``f_gt`` is flat-to-distorted, not the reverse.

    Emitting the render map (distorted-to-flat, the paper's forward map ``g``)
    instead would keep the shape, the dtype, the pixel units, the channel order
    and the value range identical, and would train a model that unwarps
    backwards.
    """

    def test_gathering_at_the_emitted_map_flattens_rather_than_warps(
        self, page, background
    ):
        height, width = 96, 128
        warp = sample_warp(np.random.default_rng(13))
        image, f_gt, mask = render_sample(
            page, background, warp, (height, width), np.random.default_rng(14),
            jitter=False,
        )
        interior = _interior(mask, f_gt)
        reference = _flat_reference(page, height, width)
        with_emitted = np.abs(_rectify(image, f_gt) - reference)[interior].mean()
        with_reversed = np.abs(
            _rectify(image, warp.forward_map(height, width)) - reference
        )[interior].mean()
        assert with_emitted < with_reversed / 10.0, (
            "the emitted map does not flatten the page better than its own "
            f"inverse does: {with_emitted} vs {with_reversed}"
        )

    def test_the_two_directions_are_not_accidentally_equal(self):
        """A near-identity warp would make the direction arm vacuous."""
        warp = sample_warp(np.random.default_rng(13))
        forward = warp.forward_map(64, 64)
        backward = warp.backward_map(64, 64)
        assert np.abs(forward - backward).max() > 2.0


class TestTheUnitsAndChannelOrder:
    """Channel 0 is x (column), channel 1 is y (row), in absolute pixels.

    Pinned on a NON-SQUARE grid; a square fixture cannot see a channel swap at
    all.
    """

    def test_the_identity_warp_reproduces_the_pixel_grid(self):
        height, width = 61, 97
        warp = sample_warp(
            np.random.default_rng(0),
            shear_stages=0,
            shear_amplitude=0.0,
            perspective_amplitude=0.0,
            margin=0.0,
        )
        f_gt = warp.backward_map(height, width)
        cols, rows = np.meshgrid(
            np.arange(width, dtype=np.float64),
            np.arange(height, dtype=np.float64),
            indexing="xy",
        )
        np.testing.assert_allclose(f_gt[..., 0], cols, rtol=0.0, atol=1e-2)
        np.testing.assert_allclose(f_gt[..., 1], rows, rtol=0.0, atol=1e-2)

    def test_channel_zero_spans_the_width_and_channel_one_the_height(self):
        """A swap survives the arm above only if H == W; this one is dimensional."""
        height, width = 61, 97
        warp = sample_warp(
            np.random.default_rng(0),
            shear_stages=0,
            shear_amplitude=0.0,
            perspective_amplitude=0.0,
            margin=0.0,
        )
        f_gt = warp.backward_map(height, width)
        assert f_gt[..., 0].max() > height, "channel 0 must range over the WIDTH"
        assert f_gt[..., 1].max() < height

    def test_the_emitted_dtypes_and_shapes_are_the_documented_contract(
        self, page, background
    ):
        image, f_gt, mask = generate_sample(
            np.random.default_rng(1), page, background, (48, 64)
        )
        assert image.shape == (48, 64, 3) and image.dtype == np.float32
        assert f_gt.shape == (48, 64, 2) and f_gt.dtype == np.float32
        assert mask.shape == (48, 64, 1) and mask.dtype == np.float32
        assert np.isfinite(image).all() and np.isfinite(f_gt).all()
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert set(np.unique(mask)).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# 2. Determinism.
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same generator seed, bit-identical sample; no global numpy state."""

    def test_the_same_seed_reproduces_all_three_arrays_bit_for_bit(
        self, page, background
    ):
        first = generate_sample(np.random.default_rng(42), page, background, (48, 64))
        second = generate_sample(np.random.default_rng(42), page, background, (48, 64))
        for a, b, name in zip(first, second, ("image", "f_gt", "mask")):
            assert np.array_equal(a, b), f"{name} differs across identical seeds"

    def test_a_different_seed_gives_a_different_sample(self, page, background):
        first = generate_sample(np.random.default_rng(42), page, background, (48, 64))
        second = generate_sample(np.random.default_rng(43), page, background, (48, 64))
        assert not np.array_equal(first[0], second[0])
        assert not np.array_equal(first[1], second[1])

    def test_seeding_global_numpy_state_changes_nothing(self, page, background):
        """The generator reads no global state -- and this arm writes none either.

        The legacy global RNG is process-wide for the whole pytest session, and
        at least one unrelated module in ``tests/test_datasets/`` draws from it
        unseeded, so leaving it re-seeded here reddens a test three files
        later. MEASURED, not hypothetical: without the save/restore below,
        ``test_masked_patches.py::TestPatchOrder::
        test_identity_kernel_reproduces_patchify_targets`` fails in a full
        ``tests/test_datasets/`` run and passes in isolation.
        """
        saved = np.random.get_state()
        try:
            np.random.seed(1234)
            first = generate_sample(np.random.default_rng(9), page, background, (48, 64))
            np.random.seed(4321)
            _ = np.random.random(1000)
            second = generate_sample(np.random.default_rng(9), page, background, (48, 64))
        finally:
            np.random.set_state(saved)
        assert np.array_equal(first[0], second[0])


# ---------------------------------------------------------------------------
# 3. The mask.
# ---------------------------------------------------------------------------


class TestTheMaskMatchesTheWarp:
    """The mask is exactly where the page landed -- not an approximation of it."""

    def test_the_mask_is_the_in_range_test_of_the_render_map(self, page, background):
        height, width = 71, 53
        warp = sample_warp(np.random.default_rng(21))
        _, _, mask = render_sample(
            page, background, warp, (height, width), np.random.default_rng(22)
        )
        flat_uv = warp.invert(sw._flat_uv_grid(height, width).reshape(-1, 2))
        expected = np.all((flat_uv >= 0.0) & (flat_uv < 1.0), axis=1)
        assert np.array_equal(mask.reshape(-1) > 0.5, expected)

    def test_outside_the_mask_the_page_content_is_absent(self, background):
        """Swapping the page changes the image ONLY where the mask is 1."""
        height, width = 64, 64
        warp = sample_warp(np.random.default_rng(23))
        page_a = np.zeros((32, 32, 3))
        page_b = np.ones((32, 32, 3))
        image_a, _, mask = render_sample(
            page_a, background, warp, (height, width), np.random.default_rng(24),
            jitter=False,
        )
        image_b, _, _ = render_sample(
            page_b, background, warp, (height, width), np.random.default_rng(24),
            jitter=False,
        )
        outside = mask[..., 0] < 0.5
        assert np.array_equal(image_a[outside], image_b[outside])
        assert not np.array_equal(image_a[~outside], image_b[~outside])

    def test_the_mask_covers_a_usable_fraction_of_the_frame(self, page, background):
        for seed in range(6):
            _, _, mask = generate_sample(
                np.random.default_rng(seed), page, background, (64, 64)
            )
            assert mask.mean() >= sw.MIN_PAGE_COVERAGE * 0.75, (
                f"seed {seed} produced a page covering only {mask.mean():.3f}"
            )


# ---------------------------------------------------------------------------
# 4. Shapes, ranges, odd sizes.
# ---------------------------------------------------------------------------


class TestOddAndNonSquareSizes:
    """Nothing here may assume ``H == W`` or an even extent."""

    @pytest.mark.parametrize("size", [(37, 91), (91, 37), (33, 33), (2, 5), (288, 288)])
    def test_the_sample_is_well_formed(self, page, background, size):
        image, f_gt, mask = generate_sample(
            np.random.default_rng(2), page, background, size
        )
        assert image.shape == (size[0], size[1], 3)
        assert f_gt.shape == (size[0], size[1], 2)
        assert mask.shape == (size[0], size[1], 1)
        assert np.isfinite(f_gt).all()

    @pytest.mark.parametrize("size", [(37, 91), (91, 37), (64, 64)])
    def test_the_backward_map_lands_inside_the_distorted_frame(
        self, page, background, size
    ):
        """Every ``f_gt`` value indexes a real pixel of the emitted image.

        This is a property of the fit affine, which places the warped page
        inside a margin of the frame; it is not clipping after the fact, and a
        warp that escaped the frame would show up here rather than as a silent
        edge-clamp during training.
        """
        height, width = size
        _, f_gt, _ = generate_sample(
            np.random.default_rng(2), page, background, size
        )
        assert f_gt[..., 0].min() >= 0.0
        assert f_gt[..., 1].min() >= 0.0
        assert f_gt[..., 0].max() <= width - 1.0
        assert f_gt[..., 1].max() <= height - 1.0

    def test_a_too_small_size_is_rejected(self, page, background):
        with pytest.raises(ValueError, match="size must be"):
            generate_sample(np.random.default_rng(0), page, background, (1, 8))

    def test_a_bad_page_shape_is_rejected(self, background):
        with pytest.raises(ValueError, match="page must be"):
            generate_sample(
                np.random.default_rng(0), np.zeros((8, 8, 4)), background, (16, 16)
            )


# ---------------------------------------------------------------------------
# 5. The degeneracy policy: reject and resample.
# ---------------------------------------------------------------------------


class TestTheDegeneracyPolicy:
    """Extreme warps are rejected, not clamped, and the detector is not vacuous."""

    def test_the_jacobian_detector_sees_a_folded_map(self):
        """A hand-built folding map -- which this warp family cannot produce.

        Feeding the detector an array rather than a warp object is what makes
        this arm real: the accept path can never exercise a fold, so without
        this the ``min_det`` half of the policy would be untested code.
        """
        height, width = 32, 32
        cols, rows = np.meshgrid(
            np.arange(width, dtype=np.float64),
            np.arange(height, dtype=np.float64),
            indexing="xy",
        )
        folded = np.stack([cols + 4.0 * np.sin(cols * 0.9), rows], axis=-1)
        min_det, max_condition = map_jacobian_stats(folded)
        assert min_det < 0.0, f"fold not detected, min det {min_det}"
        assert max_condition > sw.MAX_JACOBIAN_CONDITION

    def test_the_identity_map_has_unit_jacobian(self):
        cols, rows = np.meshgrid(
            np.arange(40, dtype=np.float64),
            np.arange(40, dtype=np.float64),
            indexing="xy",
        )
        min_det, max_condition = map_jacobian_stats(np.stack([cols, rows], axis=-1))
        assert min_det == pytest.approx(1.0)
        assert max_condition == pytest.approx(1.0)

    def test_a_determinant_test_alone_cannot_see_an_extreme_shear(self):
        """Why the policy is written on the CONDITION NUMBER, not the determinant.

        A shear has determinant exactly 1 at every amplitude. A guard that
        tested only ``det`` would accept an arbitrarily violent warp.
        """
        cols, rows = np.meshgrid(
            np.arange(40, dtype=np.float64),
            np.arange(40, dtype=np.float64),
            indexing="xy",
        )
        sheared = np.stack([cols, rows + 9.0 * cols], axis=-1)
        min_det, max_condition = map_jacobian_stats(sheared)
        assert min_det == pytest.approx(1.0)
        assert max_condition > sw.MAX_JACOBIAN_CONDITION

    def test_an_extreme_amplitude_is_reported_degenerate(self):
        quality = WarpQuality(
            min_jacobian_det=1.0,
            max_jacobian_condition=sw.MAX_JACOBIAN_CONDITION + 1.0,
            page_coverage=0.9,
        )
        assert is_degenerate(quality)

    def test_each_threshold_is_load_bearing_on_its_own(self):
        healthy = WarpQuality(
            min_jacobian_det=1.0, max_jacobian_condition=1.0, page_coverage=0.9
        )
        assert not is_degenerate(healthy)
        assert is_degenerate(
            WarpQuality(
                min_jacobian_det=sw.MIN_JACOBIAN_DET,
                max_jacobian_condition=1.0,
                page_coverage=0.9,
            )
        )
        assert is_degenerate(
            WarpQuality(
                min_jacobian_det=1.0,
                max_jacobian_condition=1.0,
                page_coverage=sw.MIN_PAGE_COVERAGE - 1e-6,
            )
        )

    def test_sample_warp_never_returns_a_degenerate_warp(self):
        for seed in range(25):
            quality = assess_warp(sample_warp(np.random.default_rng(seed)))
            assert not is_degenerate(quality), f"seed {seed} returned {quality}"

    def test_an_unsatisfiable_request_raises_rather_than_clamping(self):
        """The policy is reject-and-resample; giving up is LOUD, never silent."""
        with pytest.raises(DegenerateWarpError, match="too large"):
            sample_warp(np.random.default_rng(0), shear_amplitude=3.0)

    def test_the_argument_validation_is_real(self):
        with pytest.raises(ValueError, match="margin"):
            sample_warp(np.random.default_rng(0), margin=0.7)
        with pytest.raises(ValueError, match="non-negative"):
            sample_warp(np.random.default_rng(0), shear_amplitude=-0.1)

    def test_the_jacobian_stats_reject_an_undifferentiable_array(self):
        with pytest.raises(ValueError, match="at least 2x2"):
            map_jacobian_stats(np.zeros((1, 8, 2)))
        with pytest.raises(ValueError, match=r"\(H, W, 2\)"):
            map_jacobian_stats(np.zeros((8, 8, 3)))


# ---------------------------------------------------------------------------
# 6. The warp primitives.
# ---------------------------------------------------------------------------


class TestTheWarpPrimitivesInvertExactly:
    """Each primitive's inverse is algebraic, so it must close at round-off."""

    def _points(self):
        rng = np.random.default_rng(99)
        return rng.uniform(-0.4, 1.4, size=(500, 2))

    def test_a_shear_inverts_exactly(self):
        profile = sw._ClampedSpline(
            np.linspace(0.0, 1.0, 5), np.array([0.0, 0.3, -0.2, 0.1, 0.0])
        )
        for axis in (0, 1):
            shear = sw._Shear(axis=axis, profile=profile)
            points = self._points()
            np.testing.assert_allclose(
                shear.invert(shear.apply(points)), points, rtol=0.0, atol=1e-14
            )

    def test_a_shear_leaves_its_driving_axis_untouched(self):
        """The property the closed-form inverse depends on."""
        profile = sw._ClampedSpline(np.linspace(0.0, 1.0, 4), np.array([0.1, 0.2, 0.3, 0.4]))
        shear = sw._Shear(axis=1, profile=profile)
        points = self._points()
        np.testing.assert_array_equal(shear.apply(points)[:, 0], points[:, 0])

    def test_the_spline_is_clamped_not_extrapolated(self):
        """Cubic extrapolation outside [0, 1] would blow the mask up."""
        profile = sw._ClampedSpline(
            np.linspace(0.0, 1.0, 4), np.array([0.0, 1.0, -1.0, 0.0])
        )
        far = profile(np.array([-50.0, 50.0]))
        edge = profile(np.array([0.0, 1.0]))
        np.testing.assert_allclose(far, edge, rtol=0.0, atol=0.0)

    def test_a_homography_inverts_exactly(self):
        matrix = sw._homography_from_corners(
            np.array([[0.02, -0.03], [1.05, 0.04], [0.96, 1.02], [-0.04, 0.97]])
        )
        homography = sw._Homography(matrix)
        points = self._points()
        np.testing.assert_allclose(
            homography.invert(homography.apply(points)), points, rtol=0.0, atol=1e-12
        )

    def test_the_homography_solver_hits_its_corner_correspondences(self):
        corners = np.array([[0.02, -0.03], [1.05, 0.04], [0.96, 1.02], [-0.04, 0.97]])
        homography = sw._Homography(sw._homography_from_corners(corners))
        unit_square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        np.testing.assert_allclose(homography.apply(unit_square), corners, atol=1e-12)

    def test_an_affine_inverts_exactly(self):
        affine = sw._Affine(scale=0.37, offset=[0.11, -0.42])
        points = self._points()
        np.testing.assert_allclose(
            affine.invert(affine.apply(points)), points, rtol=0.0, atol=1e-14
        )

    def test_an_empty_composition_is_rejected(self):
        with pytest.raises(ValueError, match="at least one primitive"):
            InvertibleWarp([])

    def test_a_non_positive_scale_is_rejected(self):
        with pytest.raises(ValueError, match="scale must be positive"):
            sw._Affine(scale=0.0, offset=[0.0, 0.0])

    def test_a_bad_shear_axis_is_rejected(self):
        profile = sw._ClampedSpline(np.linspace(0.0, 1.0, 3), np.zeros(3))
        with pytest.raises(ValueError, match="axis must be 0 or 1"):
            sw._Shear(axis=2, profile=profile)


# ---------------------------------------------------------------------------
# 7. The HSV jitter.
# ---------------------------------------------------------------------------


class TestTheHsvJitter:
    """A pointwise photometric transform, written in numpy, that moves no pixel."""

    def test_the_colour_conversion_round_trips(self):
        rgb = np.random.default_rng(5).random((64, 64, 3))
        np.testing.assert_allclose(hsv_to_rgb(rgb_to_hsv(rgb)), rgb, atol=1e-12)

    def test_the_conversion_matches_hand_computed_anchors(self):
        anchors = np.array(
            [
                [1.0, 0.0, 0.0],  # pure red   -> h = 0
                [0.0, 1.0, 0.0],  # pure green -> h = 1/3
                [0.0, 0.0, 1.0],  # pure blue  -> h = 2/3
                [0.5, 0.5, 0.5],  # grey       -> s = 0
                [0.0, 0.0, 0.0],  # black      -> s = v = 0
            ]
        )
        hsv = rgb_to_hsv(anchors)
        np.testing.assert_allclose(hsv[:, 0], [0.0, 1 / 3, 2 / 3, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(hsv[:, 1], [1.0, 1.0, 1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(hsv[:, 2], [1.0, 1.0, 1.0, 0.5, 0.0], atol=1e-12)

    def test_the_jitter_stays_in_range_and_actually_changes_something(self):
        rgb = np.random.default_rng(6).random((32, 32, 3))
        jittered = jitter_hsv(rgb, np.random.default_rng(7))
        assert jittered.min() >= 0.0 and jittered.max() <= 1.0
        assert not np.allclose(jittered, rgb)

    def test_the_jitter_is_geometry_free(self, page, background):
        """``f_gt`` and the mask are bit-identical with and without it."""
        warp = sample_warp(np.random.default_rng(31))
        with_jitter = render_sample(
            page, background, warp, (48, 64), np.random.default_rng(32), jitter=True
        )
        without = render_sample(
            page, background, warp, (48, 64), np.random.default_rng(32), jitter=False
        )
        assert np.array_equal(with_jitter[1], without[1])
        assert np.array_equal(with_jitter[2], without[2])
        assert not np.array_equal(with_jitter[0], without[0])

    def test_a_bad_trailing_axis_is_rejected(self):
        with pytest.raises(ValueError, match="size-3 axis"):
            rgb_to_hsv(np.zeros((4, 4, 2)))
        with pytest.raises(ValueError, match="size-3 axis"):
            hsv_to_rgb(np.zeros((4, 4, 2)))


# ---------------------------------------------------------------------------
# 8. The dependency house rule.
# ---------------------------------------------------------------------------


def _top_level_imports(path: pathlib.Path):
    """Module-scope imports of one file (function-body imports do not count)."""
    tree = ast.parse(path.read_text())
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


class TestTheShippedPackageStaysOnDeclaredDependencies:
    """``cv2``/``skimage`` are undeclared; ``Pillow`` and ``h5py`` are extra-only.

    One forbidden set for the whole package, not one per module: ``uvdoc.py``
    needs ``h5py`` to read UVDoc's MATLAB v7.3 ground truth and ``PIL`` to
    decode its renders, and both are declared only in ``pyproject.toml``'s
    ``data`` extra -- so both are imported inside the function that needs them,
    exactly like ``synthetic_warp.load_rgb``.
    """

    def _sources(self):
        package = importlib.import_module(
            "dl_techniques.datasets.document_rectification"
        )
        return sorted(pathlib.Path(package.__file__).parent.glob("*.py"))

    def test_no_module_imports_an_extra_only_dependency_at_module_scope(self):
        forbidden = {"cv2", "skimage", "PIL", "h5py"}
        offenders = {
            path.name: sorted(_top_level_imports(path) & forbidden)
            for path in self._sources()
            if _top_level_imports(path) & forbidden
        }
        assert not offenders, f"undeclared module-scope imports: {offenders}"

    def test_the_scanner_can_actually_see_an_import(self, tmp_path):
        probe = tmp_path / "probe.py"
        probe.write_text(
            "import numpy\nimport cv2\nfrom skimage.filters import x\n"
            "from PIL import Image\n"
            "def f():\n    import scipy\n"
        )
        found = _top_level_imports(probe)
        assert {"cv2", "skimage", "PIL", "numpy"} <= found
        assert "scipy" not in found, "a function-body import must not count"

    def test_the_replacement_stack_is_present(self):
        imports = _top_level_imports(pathlib.Path(sw.__file__))
        assert "numpy" in imports and "scipy" in imports

    def test_the_flat_uv_grid_is_reused_not_reimplemented(self):
        """The grid comes from ``dtsprompt.base_coordinate_grid``, D-021 and all."""
        imports = _top_level_imports(pathlib.Path(sw.__file__))
        assert "dl_techniques" in imports
        from dl_techniques.datasets.document_restoration import base_coordinate_grid

        np.testing.assert_array_equal(
            sw._flat_uv_grid(7, 11), base_coordinate_grid(7, 11).astype(np.float64)
        )
