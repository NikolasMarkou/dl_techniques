"""Guards for the UVDoc reader, densifier and inverter.

Three things this suite exists to pin.

**1. The orientation.** UVDoc ships MATLAB v7.3 ``.mat`` files, and h5py hands
those back with every axis reversed. ``read_grid2d`` therefore transposes
``(2, cols, rows) -> (rows, cols, 2)``, which is NOT the identity-looking
``transpose(1, 2, 0)``. On the real corpus a wrong transpose raises, because the
lattice is 89x61; on the square 288x288 grid everything downstream lives on, the
same mistake is completely silent and transposes every page. Every fixture here
is deliberately NON-SQUARE for that reason.

**2. The accuracy of the one approximate step in the port.** ``synthetic_warp``
needs no numerical inversion at all (D-036); UVDoc's coarse correspondence
lattice has no analytic inverse, so ``g`` is fitted. The arms below measure
that fit against an EXACT reference rather than against itself: a closed-form
:class:`InvertibleWarp` is sampled on a UVDoc-shaped lattice, densified through
the real code path, and compared to the warp's own ``backward_map`` /
``forward_map``. Every bound carries the number that was measured.

**3. Convention agreement with ``synthetic_warp``.** Asserted through the
shared instrument in ``backward_map_convention.py`` -- one assertion applied to
both producers, not two copies of a comment. That instrument's own RED proofs
live in ``test_backward_map_convention.py``.

Archive gating: the 27.5 GB ``UVDoc_final.zip`` is NOT required. Every arm
except :class:`TestTheStagedArchiveIsShapedTheWayThisModuleAssumes` and
:class:`TestTheInversionAgreesWithUVDocsOwnUvMap` runs against a synthetic
stand-in corpus written into ``tmp_path`` as real MATLAB-v7.3-shaped HDF5 --
same reader, same transpose, same code path, ~3 KB.
"""

import json
import os
import zipfile

import numpy as np
import pytest
from scipy import ndimage

from dl_techniques.datasets.document_rectification import synthetic_warp as sw
from dl_techniques.datasets.document_rectification import uvdoc
from dl_techniques.datasets.document_rectification.uvdoc import (
    MAX_HELD_OUT_DENSIFICATION_PIXELS,
    MAX_INTERIOR_ROUNDTRIP_PIXELS,
    UVDocDensificationError,
    UVDocError,
    UVDocSource,
    assess_densification,
    control_point_uv,
    densify_backward_map,
    held_out_densification_error,
    invert_backward_map,
    is_unusable,
    load_geometry,
    read_grid2d,
    read_segmentation,
    rectified_pixel_grid,
    roundtrip_error,
)

from .backward_map_convention import (
    CONVENTION_HEIGHT,
    CONVENTION_WIDTH,
    assert_is_backward_map,
    assert_is_forward_map,
    assert_same_backward_map_convention,
)

h5py = pytest.importorskip(
    "h5py",
    reason="h5py reads UVDoc's MATLAB v7.3 ground truth; it is in the 'data' extra",
)

# ---------------------------------------------------------------------------
# The staged archive. Present on the machine this port was written on; absent
# in any clean checkout, and NOTHING here may require it.
# ---------------------------------------------------------------------------

UVDOC_ARCHIVE = "/media/arxwn/data0_4tb/datasets/doc_scanner/uvdoc/UVDoc_final.zip"

requires_archive = pytest.mark.skipif(
    not os.path.isfile(UVDOC_ARCHIVE),
    reason=f"the staged UVDoc archive is not present at {UVDOC_ARCHIVE}",
)

# ---------------------------------------------------------------------------
# Measured budgets. Each is the number this suite actually observed, with the
# stated headroom -- not a tolerance chosen until the test passed.
# ---------------------------------------------------------------------------

#: Densifying a UVDoc-shaped 89x61 lattice sampled from an EXACT warp, at
#: 288x288. MEASURED over synthetic_warp seeds 0-5: max 0.26-1.09 px,
#: mean 0.003-0.010 px. The bound is on a MILD warp (see `mild_warp`), whose
#: measured max is 0.031 px.
EXACT_DENSIFY_MAX_PIXELS = 0.25

#: `g` against the warp's own closed-form `forward_map`, on the page interior.
#: MEASURED on the mild warp: max 0.010 px.
EXACT_INVERSION_MAX_PIXELS = 0.10

#: Round trip ||g(f(x)) - x|| on the rectified interior. MEASURED: 0.0035 px on
#: the mild warp, 0.030-0.574 px on violent synthetic warps, 0.030-0.480 px
#: over 40 real UVDoc geometries.
INTERIOR_ROUNDTRIP_MAX_PIXELS = 0.05

#: The control lattice this suite writes into its stand-in corpus. Non-square,
#: and large enough for `held_out_densification_error`'s 8x8 floor.
STANDIN_ROWS = 17
STANDIN_COLS = 11

#: The stand-in corpus's frame. Non-square, and NOT the target grid, so a
#: forgotten `source_size` rescale shows up.
STANDIN_HEIGHT = 71
STANDIN_WIDTH = 53

#: The target grid the stand-in is densified onto. Non-square on purpose.
TARGET_HEIGHT = CONVENTION_HEIGHT
TARGET_WIDTH = CONVENTION_WIDTH


# ---------------------------------------------------------------------------
# Fixtures: a stand-in UVDoc corpus built from an EXACT warp.
# ---------------------------------------------------------------------------


@pytest.fixture
def mild_warp():
    """A gentle, exactly invertible warp -- the reference the fit is scored on.

    Mild on purpose. The densifier's job here is to reproduce a smooth
    deformation from an 89x61-style lattice; scoring it on a violent warp would
    measure the sampling theorem, not the code.
    """
    return sw.sample_warp(
        np.random.default_rng(5),
        shear_stages=2,
        shear_amplitude=0.02,
        perspective_amplitude=0.02,
    )


def _write_mat(path, key, array):
    """Write ``array`` the way MATLAB v7.3 does: HDF5 with every axis reversed."""
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=np.ascontiguousarray(array.T))


def _control_lattice(warp, height, width, rows=STANDIN_ROWS, cols=STANDIN_COLS):
    """Sample ``warp``'s backward map on a UVDoc-style control lattice."""
    uv = control_point_uv(rows, cols)
    pixels = warp.apply(uv.reshape(-1, 2)).reshape(rows, cols, 2)
    return pixels * np.array([width, height], dtype=np.float64)


def _write_standin_corpus(root, lattice, height, width, geometry="geo_0", sample="00000"):
    """Write a minimal but structurally faithful UVDoc tree under ``root``."""
    tree = os.path.join(root, uvdoc.UVDOC_ROOT)
    for directory in (
        uvdoc.GRID2D_DIR,
        uvdoc.SEG_DIR,
        uvdoc.SAMPLE_METADATA_DIR,
        uvdoc.IMAGE_DIR,
    ):
        os.makedirs(os.path.join(tree, directory), exist_ok=True)

    _write_mat(
        os.path.join(tree, uvdoc.GRID2D_DIR, f"{geometry}.mat"),
        uvdoc.GRID2D_KEY,
        lattice,
    )
    segmentation = np.zeros((height, width), dtype=np.uint8)
    segmentation[height // 4 : -height // 4, width // 4 : -width // 4] = 1
    _write_mat(
        os.path.join(tree, uvdoc.SEG_DIR, f"{geometry}.mat"),
        uvdoc.SEG_KEY,
        segmentation,
    )
    with open(
        os.path.join(tree, uvdoc.SAMPLE_METADATA_DIR, f"{sample}.json"), "w"
    ) as handle:
        json.dump({"geom_name": geometry, "sample_id": sample}, handle)
    return tree


@pytest.fixture
def standin_corpus(tmp_path, mild_warp):
    """A 1-geometry UVDoc tree on disk, plus the exact warp behind it."""
    lattice = _control_lattice(mild_warp, STANDIN_HEIGHT, STANDIN_WIDTH)
    _write_standin_corpus(str(tmp_path), lattice, STANDIN_HEIGHT, STANDIN_WIDTH)
    return str(tmp_path), lattice, mild_warp


@pytest.fixture
def standin_source(standin_corpus):
    root, _, _ = standin_corpus
    with UVDocSource(root) as source:
        yield source


# ---------------------------------------------------------------------------
# 1. THE POINT OF THE STEP, part one: the grid is read the right way round.
# ---------------------------------------------------------------------------


class TestTheGridIsReadInTheRightOrientation:
    """h5py reverses MATLAB's axes; ``read_grid2d`` must undo exactly that."""

    def test_the_lattice_comes_back_rows_by_cols_by_two(self, standin_source):
        grid = read_grid2d(standin_source, "geo_0")
        assert grid.shape == (STANDIN_ROWS, STANDIN_COLS, 2), (
            "a wrong transpose of a (2, cols, rows) h5py array yields "
            f"{grid.shape}; the lattice is non-square precisely so this raises"
        )
        assert grid.dtype == np.float64

    def test_the_values_are_the_control_points_that_were_written(
        self, standin_source, standin_corpus
    ):
        """Byte-level identity, not a tolerance: this is a transpose, not a fit."""
        _, lattice, _ = standin_corpus
        np.testing.assert_array_equal(read_grid2d(standin_source, "geo_0"), lattice)

    def test_channel_zero_is_x_and_ranges_over_the_width(self, standin_source):
        grid = read_grid2d(standin_source, "geo_0")
        assert grid[..., 0].max() < STANDIN_WIDTH + 1.0
        assert grid[..., 1].max() > STANDIN_WIDTH, (
            "channel 1 must be y, which on this 71x53 frame exceeds the WIDTH; "
            "if it does not, the channels are swapped"
        )

    def test_the_segmentation_comes_back_height_by_width(self, standin_source):
        mask = read_segmentation(standin_source, "geo_0")
        assert mask.shape == (STANDIN_HEIGHT, STANDIN_WIDTH)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)) <= {0, 1}

    def test_a_lattice_with_the_wrong_rank_is_refused(self, tmp_path):
        tree = os.path.join(str(tmp_path), uvdoc.UVDOC_ROOT, uvdoc.GRID2D_DIR)
        os.makedirs(tree)
        _write_mat(
            os.path.join(tree, "bad.mat"), uvdoc.GRID2D_KEY, np.zeros((5, 5))
        )
        with UVDocSource(str(tmp_path)) as source:
            with pytest.raises(UVDocError, match="expected"):
                read_grid2d(source, "bad")

    def test_a_missing_dataset_key_names_what_was_found(self, tmp_path):
        tree = os.path.join(str(tmp_path), uvdoc.UVDOC_ROOT, uvdoc.GRID2D_DIR)
        os.makedirs(tree)
        _write_mat(
            os.path.join(tree, "bad.mat"), "not_grid2d", np.zeros((17, 11, 2))
        )
        with UVDocSource(str(tmp_path)) as source:
            with pytest.raises(UVDocError, match="not_grid2d"):
                read_grid2d(source, "bad")


# ---------------------------------------------------------------------------
# 2. THE POINT OF THE STEP, part two: the densification is accurate, and the
#    guard that proves it is the one with power.
# ---------------------------------------------------------------------------


class TestTheDensifiedMapMatchesAnExactReference:
    """Scored against a closed-form warp, not against the fit's own inputs."""

    def test_the_densified_map_reproduces_the_exact_backward_map(self, mild_warp):
        lattice = _control_lattice(mild_warp, TARGET_HEIGHT, TARGET_WIDTH)
        dense = densify_backward_map(lattice, TARGET_HEIGHT, TARGET_WIDTH)
        exact = mild_warp.backward_map(TARGET_HEIGHT, TARGET_WIDTH)
        error = np.linalg.norm(dense - exact, axis=-1)
        assert error.max() < EXACT_DENSIFY_MAX_PIXELS, (
            f"densification error {error.max():.4f} px exceeds "
            f"{EXACT_DENSIFY_MAX_PIXELS} px (MEASURED on this fixture: 0.031 px "
            f"max, {error.mean():.5f} px mean this run)"
        )

    def test_the_source_size_rescale_is_applied(self, standin_corpus):
        """A forgotten rescale is invisible when the frame IS the target grid."""
        _, lattice, warp = standin_corpus
        dense = densify_backward_map(
            lattice,
            TARGET_HEIGHT,
            TARGET_WIDTH,
            source_size=(STANDIN_HEIGHT, STANDIN_WIDTH),
        )
        exact = warp.backward_map(TARGET_HEIGHT, TARGET_WIDTH)
        error = np.linalg.norm(dense - exact, axis=-1)
        assert error.max() < 1.0, (
            f"{error.max():.4f} px: without the (W/W_src, H/H_src) rescale this "
            f"is off by the whole ratio, ~{TARGET_WIDTH / STANDIN_WIDTH:.2f}x"
        )

    def test_a_degenerate_target_or_lattice_is_refused(self, standin_corpus):
        _, lattice, _ = standin_corpus
        with pytest.raises(ValueError, match="at least 2x2"):
            densify_backward_map(lattice, 1, 32)
        with pytest.raises(ValueError, match=r"\(rows, cols, 2\)"):
            densify_backward_map(lattice[..., 0], 32, 32)


class TestTheControlPointGuardIsWeakAndTheHeldOutGuardIsNot:
    """The cheap oracle is nearly vacuous, and this suite says so out loud."""

    def test_the_densified_map_reproduces_its_own_control_points(self, mild_warp):
        """Asked for by the step brief -- and it CANNOT fail. See below."""
        rows, cols = STANDIN_ROWS, STANDIN_COLS
        lattice = _control_lattice(mild_warp, TARGET_HEIGHT, TARGET_WIDTH)
        # Densify onto a grid whose pixels land exactly on the control lattice:
        # (rows - 1) x (cols - 1) tiles, so u = c / (cols - 1) is hit exactly.
        dense = densify_backward_map(lattice, (rows - 1) * 4, (cols - 1) * 4)
        sampled = dense[:: 4, :: 4]
        error = np.linalg.norm(sampled - lattice[:-1, :-1], axis=-1)
        assert error.max() < 1e-3, f"{error.max():.3e} px"

    def test_the_control_point_guard_is_vacuous_on_an_interpolating_spline(
        self, mild_warp
    ):
        """The meta-arm: reproducing the fit's own inputs measures nothing.

        A tensor-product spline with ``s=0`` interpolates by construction, so it
        returns its control points at ~1e-13 px whatever the data is -- INCLUDING
        data that no smooth warp produced. This arm proves that by fitting pure
        noise: the control points still come back exactly, while the held-out
        error explodes. Anyone tempted to delete the held-out metric as
        redundant should read this first.
        """
        rng = np.random.default_rng(0)
        noise = rng.uniform(0.0, 64.0, size=(STANDIN_ROWS, STANDIN_COLS, 2))
        dense = densify_backward_map(noise, (STANDIN_ROWS - 1) * 4, (STANDIN_COLS - 1) * 4)
        reproduction = np.linalg.norm(
            dense[::4, ::4] - noise[:-1, :-1], axis=-1
        ).max()
        held_out_max, _ = held_out_densification_error(
            noise, np.random.default_rng(1)
        )
        assert reproduction < 1e-3, (
            f"even on pure noise the fit returns its own control points "
            f"({reproduction:.3e} px -- float32 STORAGE of a ~64 px value, not "
            f"fit error) -- that is what makes it a weak oracle"
        )
        assert held_out_max > 100.0 * max(reproduction, 1e-9), (
            f"held-out error {held_out_max:.3f} px must dwarf the "
            f"reproduction error {reproduction:.3e} px on noise; if it does "
            f"not, the held-out split has lost its power"
        )

    def test_the_held_out_split_never_extrapolates(self, mild_warp):
        """D-042: a parity split hands the lattice's outer ring to the held-out
        set, and predicting it is extrapolation. Measured 7x worse on real data.

        Proven behaviourally: the held-out error on a smooth warp must stay far
        below what a boundary-extrapolating split produces on the same lattice.
        """
        lattice = _control_lattice(
            mild_warp, TARGET_HEIGHT, TARGET_WIDTH, rows=41, cols=27
        )
        interior_errors = [
            held_out_densification_error(lattice, np.random.default_rng(seed))[0]
            for seed in range(8)
        ]
        assert max(interior_errors) < MAX_HELD_OUT_DENSIFICATION_PIXELS, (
            f"the interior held-out split must stay inside the shipped bound "
            f"across seeds; got {max(interior_errors):.4f} px"
        )
        # The rejected alternative, computed here so the comparison is measured
        # rather than asserted: fit on rows 1,3,5,... which drops row 0 and the
        # last row into the held-out set.
        rows, cols = lattice.shape[:2]
        fit_rows = np.arange(1, rows, 2)
        fit_cols = np.arange(1, cols, 2)
        held_rows = np.setdiff1d(np.arange(rows), fit_rows)
        held_cols = np.setdiff1d(np.arange(cols), fit_cols)
        v_axis = np.linspace(0.0, 1.0, rows)
        u_axis = np.linspace(0.0, 1.0, cols)
        splines = uvdoc._spline_pair(
            lattice[np.ix_(fit_rows, fit_cols)], v_axis[fit_rows], u_axis[fit_cols]
        )
        query_u, query_v = np.meshgrid(
            u_axis[held_cols], v_axis[held_rows], indexing="xy"
        )
        predicted = np.stack(
            [spline.ev(query_v, query_u) for spline in splines], axis=-1
        )
        extrapolating = np.linalg.norm(
            predicted - lattice[np.ix_(held_rows, held_cols)], axis=-1
        ).max()
        assert extrapolating > max(interior_errors), (
            f"the boundary-extrapolating split ({extrapolating:.4f} px) must be "
            f"worse than the interior one ({max(interior_errors):.4f} px); if "
            f"it is not, D-042's reason for existing has evaporated"
        )

    def test_a_lattice_too_small_to_cross_validate_is_refused(self):
        with pytest.raises(ValueError, match="8x8"):
            held_out_densification_error(
                np.zeros((5, 5, 2)), np.random.default_rng(0)
            )


# ---------------------------------------------------------------------------
# 3. The inversion -- the one genuinely approximate step in the port.
# ---------------------------------------------------------------------------


class TestTheInversionIsBounded:
    """``g`` is fitted, so its error is measured against an exact reference."""

    def test_g_matches_the_warps_own_closed_form_forward_map(self, mild_warp):
        f_gt = densify_backward_map(
            _control_lattice(mild_warp, TARGET_HEIGHT, TARGET_WIDTH),
            TARGET_HEIGHT,
            TARGET_WIDTH,
        )
        g, outside = invert_backward_map(f_gt)
        exact = mild_warp.forward_map(TARGET_HEIGHT, TARGET_WIDTH)
        on_page = (
            np.all((exact >= 0.0), axis=-1)
            & (exact[..., 0] < TARGET_WIDTH)
            & (exact[..., 1] < TARGET_HEIGHT)
            & (~outside)
        )
        # Eroded by 2 px: the page's own outline is where the linear
        # interpolant meets the hull of the point cloud, and the error there is
        # a boundary artefact of the fill (MEASURED: 1.00 px un-eroded on this
        # fixture, against 0.010 px two pixels in). The composed L_line term
        # reads g where a predicted flow lands, i.e. on the page, not on its
        # outline.
        on_page = ndimage.binary_erosion(on_page, iterations=2)
        error = np.linalg.norm(g - exact, axis=-1)[on_page]
        assert error.max() < EXACT_INVERSION_MAX_PIXELS, (
            f"inversion error on the page {error.max():.4f} px exceeds "
            f"{EXACT_INVERSION_MAX_PIXELS} px (MEASURED on this fixture: "
            f"0.010 px max, {error.mean():.5f} px mean this run)"
        )

    def test_the_round_trip_closes_on_the_rectified_interior(self, mild_warp):
        f_gt = densify_backward_map(
            _control_lattice(mild_warp, TARGET_HEIGHT, TARGET_WIDTH),
            TARGET_HEIGHT,
            TARGET_WIDTH,
        )
        g, _ = invert_backward_map(f_gt)
        error = roundtrip_error(f_gt, g)
        border = uvdoc.ROUNDTRIP_BORDER_PIXELS
        interior = error[border:-border, border:-border].max()
        assert interior < INTERIOR_ROUNDTRIP_MAX_PIXELS, (
            f"||g(f(x)) - x|| on the interior is {interior:.4f} px, over the "
            f"{INTERIOR_ROUNDTRIP_MAX_PIXELS} px bound. MEASURED elsewhere: "
            f"0.0035 px on this fixture, 0.030-0.480 px over 40 real UVDoc "
            f"geometries, worst 0.574 px on a violent synthetic warp -- all of "
            f"which sit under the shipped gate of "
            f"{MAX_INTERIOR_ROUNDTRIP_PIXELS} px"
        )

    def test_the_whole_grid_maximum_is_worse_than_the_interior(self, mild_warp):
        """D-040's premise, measured: the border ring is the hull-edge artefact.

        If this ever inverts, gating on the whole-grid maximum would have been
        the right call after all and D-040 must be revisited.
        """
        f_gt = densify_backward_map(
            _control_lattice(mild_warp, TARGET_HEIGHT, TARGET_WIDTH),
            TARGET_HEIGHT,
            TARGET_WIDTH,
        )
        g, _ = invert_backward_map(f_gt)
        error = roundtrip_error(f_gt, g)
        border = uvdoc.ROUNDTRIP_BORDER_PIXELS
        assert error.max() > error[border:-border, border:-border].max()

    def test_the_hull_miss_is_reported_and_the_fill_is_finite(self, mild_warp):
        f_gt = densify_backward_map(
            _control_lattice(mild_warp, TARGET_HEIGHT, TARGET_WIDTH),
            TARGET_HEIGHT,
            TARGET_WIDTH,
        )
        g, outside = invert_backward_map(f_gt)
        assert outside.shape == (TARGET_HEIGHT, TARGET_WIDTH)
        assert outside.dtype == np.bool_
        assert outside.any(), (
            "a warped page does not fill its frame; some pixel must fall "
            "outside the inverted point cloud's hull"
        )
        assert np.isfinite(g).all(), (
            "an unfilled NaN reaches L_line the first time a predicted flow "
            "lands off-page (D-040)"
        )

    def test_a_non_finite_map_is_refused_rather_than_inverted(self):
        broken = np.zeros((8, 9, 2), dtype=np.float32)
        broken[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            invert_backward_map(broken)

    def test_subsampling_is_faster_and_measurably_worse(self, mild_warp):
        """The knob is documented as a trade; a knob that does nothing is a lie."""
        f_gt = densify_backward_map(
            _control_lattice(mild_warp, TARGET_HEIGHT, TARGET_WIDTH),
            TARGET_HEIGHT,
            TARGET_WIDTH,
        )
        full, _ = invert_backward_map(f_gt, subsample=1)
        sparse, _ = invert_backward_map(f_gt, subsample=3)
        assert not np.array_equal(full, sparse)
        border = uvdoc.ROUNDTRIP_BORDER_PIXELS
        interior = slice(border, -border)
        full_error = roundtrip_error(f_gt, full)[interior, interior].max()
        sparse_error = roundtrip_error(f_gt, sparse)[interior, interior].max()
        assert sparse_error > full_error, (
            f"subsample=3 ({sparse_error:.4f} px) must be worse than "
            f"subsample=1 ({full_error:.4f} px)"
        )

    def test_a_non_positive_subsample_is_refused(self, mild_warp):
        f_gt = densify_backward_map(
            _control_lattice(mild_warp, 32, 41), 32, 41
        )
        with pytest.raises(ValueError, match="subsample"):
            invert_backward_map(f_gt, subsample=0)


# ---------------------------------------------------------------------------
# 4. Convention agreement with synthetic_warp -- ONE shared assertion.
# ---------------------------------------------------------------------------


class TestTheTwoCorporaShareOneConvention:
    """Both producers go through ``backward_map_convention``, not two comments."""

    def test_both_modules_emit_the_same_backward_map_contract(
        self, standin_source, mild_warp
    ):
        from_uvdoc = load_geometry(
            standin_source,
            "geo_0",
            np.random.default_rng(0),
            size=(TARGET_HEIGHT, TARGET_WIDTH),
        ).f_gt
        _, from_synthetic, _ = sw.render_sample(
            np.zeros((8, 8, 3), dtype=np.float32),
            np.zeros((8, 8, 3), dtype=np.float32),
            mild_warp,
            (TARGET_HEIGHT, TARGET_WIDTH),
            np.random.default_rng(0),
            jitter=False,
        )
        assert_same_backward_map_convention(
            from_uvdoc, from_synthetic, "uvdoc.load_geometry", "synthetic_warp"
        )

    def test_the_uvdoc_forward_map_satisfies_the_g_contract(self, standin_source):
        geometry = load_geometry(
            standin_source,
            "geo_0",
            np.random.default_rng(0),
            size=(TARGET_HEIGHT, TARGET_WIDTH),
        )
        assert_is_forward_map(
            geometry.g, TARGET_HEIGHT, TARGET_WIDTH, "uvdoc.load_geometry"
        )
        assert_is_backward_map(
            geometry.f_gt, TARGET_HEIGHT, TARGET_WIDTH, "uvdoc.load_geometry"
        )

    def test_the_two_modules_produce_the_SAME_NUMBERS_from_the_same_warp(
        self, mild_warp
    ):
        """The strongest form: identical values, not merely identical shapes.

        A UVDoc-style lattice sampled from ``mild_warp`` and densified through
        the UVDoc code path must land on ``synthetic_warp``'s own exact backward
        map. Any disagreement about units, channel order, domain or the
        ``[0, 1)`` vs ``[0, 1]`` span shows up here as pixels.
        """
        lattice = _control_lattice(
            mild_warp, TARGET_HEIGHT, TARGET_WIDTH, rows=61, cols=41
        )
        from_uvdoc = densify_backward_map(lattice, TARGET_HEIGHT, TARGET_WIDTH)
        from_synthetic = mild_warp.backward_map(TARGET_HEIGHT, TARGET_WIDTH)
        difference = np.linalg.norm(from_uvdoc - from_synthetic, axis=-1).max()
        assert difference < EXACT_DENSIFY_MAX_PIXELS, (
            f"the two producers disagree by {difference:.4f} px on the same "
            f"warp; a channel swap would show as ~{TARGET_WIDTH} px here"
        )

    def test_the_rectified_grid_is_the_repos_grid_not_a_new_one(self):
        """Reuse, byte for byte: same helper, same D-021 channel-order ruling."""
        from dl_techniques.datasets.document_restoration import base_coordinate_grid

        height, width = 7, 11
        expected = base_coordinate_grid(height, width).astype(np.float64) * np.array(
            [width, height], dtype=np.float64
        )
        np.testing.assert_array_equal(rectified_pixel_grid(height, width), expected)
        np.testing.assert_array_equal(
            rectified_pixel_grid(height, width),
            sw._flat_uv_grid(height, width) * np.array([width, height]),
        )

    def test_the_control_lattice_spans_the_closed_unit_square(self):
        """[0, 1] inclusive for control points, [0, 1) for pixels -- both, on purpose."""
        uv = control_point_uv(5, 3)
        assert uv.shape == (5, 3, 2)
        np.testing.assert_allclose(uv[0, 0], [0.0, 0.0])
        np.testing.assert_allclose(uv[-1, -1], [1.0, 1.0])
        np.testing.assert_allclose(uv[0, 1], [0.5, 0.0])
        with pytest.raises(ValueError, match="at least 2x2"):
            control_point_uv(1, 4)


# ---------------------------------------------------------------------------
# 5. The rejection path.
# ---------------------------------------------------------------------------


class TestTheRejectionPath:
    """A UVDoc geometry cannot be re-drawn, so a bad one is dropped and named."""

    @staticmethod
    def _folded_lattice(warp):
        """A lattice whose map is NOT injective: two rows swapped.

        This is the failure the synthetic generator cannot produce at all (a
        composition of bijections is a bijection, D-037), which is exactly why
        the UVDoc path needs its own gate.
        """
        lattice = _control_lattice(warp, TARGET_HEIGHT, TARGET_WIDTH)
        lattice[[4, 12]] = lattice[[12, 4]]
        return lattice

    def test_a_folded_lattice_is_measured_as_unusable(self, mild_warp):
        lattice = self._folded_lattice(mild_warp)
        f_gt = densify_backward_map(lattice, TARGET_HEIGHT, TARGET_WIDTH)
        g, outside = invert_backward_map(f_gt)
        quality = assess_densification(
            lattice, f_gt, g, outside, np.random.default_rng(0)
        )
        assert is_unusable(quality), (
            f"a non-injective lattice must be rejected; measured "
            f"held-out {quality.held_out_densification_pixels:.3f} px, "
            f"interior round trip {quality.interior_roundtrip_pixels:.3f} px"
        )

    def test_a_healthy_lattice_is_measured_as_usable(self, mild_warp):
        """The other half: a gate that rejects everything is not a gate."""
        lattice = _control_lattice(mild_warp, TARGET_HEIGHT, TARGET_WIDTH)
        f_gt = densify_backward_map(lattice, TARGET_HEIGHT, TARGET_WIDTH)
        g, outside = invert_backward_map(f_gt)
        quality = assess_densification(
            lattice, f_gt, g, outside, np.random.default_rng(0)
        )
        assert not is_unusable(quality), quality

    def test_load_geometry_raises_and_names_the_metric_that_failed(
        self, tmp_path, mild_warp
    ):
        lattice = self._folded_lattice(mild_warp)
        _write_standin_corpus(
            str(tmp_path), lattice, TARGET_HEIGHT, TARGET_WIDTH, geometry="folded"
        )
        with UVDocSource(str(tmp_path)) as source:
            with pytest.raises(UVDocDensificationError) as failure:
                load_geometry(
                    source,
                    "folded",
                    np.random.default_rng(0),
                    size=(TARGET_HEIGHT, TARGET_WIDTH),
                )
        message = str(failure.value)
        assert "folded" in message
        assert "px" in message and ">" in message, (
            f"the rejection must name the measured value and its bound; got: "
            f"{message}"
        )

    def test_strict_false_returns_the_sample_with_its_measurements(
        self, tmp_path, mild_warp
    ):
        lattice = self._folded_lattice(mild_warp)
        _write_standin_corpus(
            str(tmp_path), lattice, TARGET_HEIGHT, TARGET_WIDTH, geometry="folded"
        )
        with UVDocSource(str(tmp_path)) as source:
            geometry = load_geometry(
                source,
                "folded",
                np.random.default_rng(0),
                size=(TARGET_HEIGHT, TARGET_WIDTH),
                strict=False,
            )
        assert is_unusable(geometry.quality)
        assert np.isfinite(geometry.f_gt).all() and np.isfinite(geometry.g).all()

    def test_a_collapsed_map_is_refused_by_the_inverter_itself(self):
        collapsed = np.zeros((16, 21, 2), dtype=np.float32)
        with pytest.raises(UVDocDensificationError, match="degenerate"):
            invert_backward_map(collapsed)


# ---------------------------------------------------------------------------
# 6. The emitted triple, determinism, and the source abstraction.
# ---------------------------------------------------------------------------


class TestTheEmittedTriple:
    """Shape/dtype contract of what the pipeline actually consumes."""

    def test_the_shapes_and_dtypes_are_the_documented_contract(self, standin_source):
        geometry = load_geometry(
            standin_source,
            "geo_0",
            np.random.default_rng(0),
            size=(TARGET_HEIGHT, TARGET_WIDTH),
        )
        assert geometry.f_gt.shape == (TARGET_HEIGHT, TARGET_WIDTH, 2)
        assert geometry.g.shape == (TARGET_HEIGHT, TARGET_WIDTH, 2)
        assert geometry.mask.shape == (TARGET_HEIGHT, TARGET_WIDTH, 1)
        assert geometry.f_gt.dtype == geometry.g.dtype == np.float32
        assert geometry.mask.dtype == np.float32
        assert set(np.unique(geometry.mask)) <= {0.0, 1.0}
        assert geometry.name == "geo_0"

    def test_the_mask_is_the_page_footprint_resampled_not_a_constant(
        self, standin_source
    ):
        geometry = load_geometry(
            standin_source,
            "geo_0",
            np.random.default_rng(0),
            size=(TARGET_HEIGHT, TARGET_WIDTH),
        )
        coverage = float(geometry.mask.mean())
        assert 0.1 < coverage < 0.9, (
            f"the stand-in page covers a quarter-inset rectangle; a mask "
            f"coverage of {coverage:.3f} means the resample collapsed it"
        )


class TestDeterminism:
    """Same generator, same numbers; and no global numpy state is touched."""

    def test_the_same_generator_reproduces_the_quality_metrics(self, standin_source):
        first = load_geometry(
            standin_source, "geo_0", np.random.default_rng(3), size=(48, 61)
        )
        second = load_geometry(
            standin_source, "geo_0", np.random.default_rng(3), size=(48, 61)
        )
        assert first.quality == second.quality
        np.testing.assert_array_equal(first.f_gt, second.f_gt)
        np.testing.assert_array_equal(first.g, second.g)

    def test_different_generators_move_only_the_cross_validation(
        self, standin_source
    ):
        first = load_geometry(
            standin_source, "geo_0", np.random.default_rng(3), size=(48, 61)
        )
        second = load_geometry(
            standin_source, "geo_0", np.random.default_rng(4), size=(48, 61)
        )
        np.testing.assert_array_equal(first.f_gt, second.f_gt)
        np.testing.assert_array_equal(first.g, second.g)

    def test_no_global_numpy_state_is_read_or_written(self, standin_source):
        """The cross-module hazard this test directory has already caused once.

        Seeding or consuming the LEGACY GLOBAL numpy RNG from a dataset module
        reddens unrelated tests in other files. The whole module takes an
        explicit ``Generator``; this arm proves the global stream is untouched.
        """
        before = np.random.get_state()
        try:
            load_geometry(
                standin_source, "geo_0", np.random.default_rng(0), size=(48, 61)
            )
            after = np.random.get_state()
        finally:
            np.random.set_state(before)
        assert before[0] == after[0]
        np.testing.assert_array_equal(before[1], after[1])
        assert before[2:] == after[2:]


class TestTheSourceReadsBothStagedForms:
    """A zip and an extracted tree must behave identically."""

    def test_a_directory_and_a_zip_agree_member_for_member(
        self, tmp_path, standin_corpus
    ):
        root, _, _ = standin_corpus
        archive = str(tmp_path / "packed.zip")
        with zipfile.ZipFile(archive, "w") as handle:
            for folder, _, files in os.walk(os.path.join(root, uvdoc.UVDOC_ROOT)):
                for name in files:
                    full = os.path.join(folder, name)
                    handle.write(full, os.path.relpath(full, root))
            # The resource forks the real archive is half made of.
            handle.writestr(
                f"__MACOSX/{uvdoc.UVDOC_ROOT}/{uvdoc.GRID2D_DIR}/._geo_0.mat",
                b"\x00\x05\x16\x07not a mat file",
            )
        with UVDocSource(root) as directory, UVDocSource(archive) as packed:
            assert directory.geometry_names() == packed.geometry_names() == ["geo_0"]
            assert directory.sample_ids() == packed.sample_ids()
            np.testing.assert_array_equal(
                read_grid2d(directory, "geo_0"), read_grid2d(packed, "geo_0")
            )

    def test_the_macosx_resource_forks_are_not_listed(self, tmp_path, standin_corpus):
        """D-039: half of UVDoc_final.zip's 182,492 members are resource forks.

        Without the filter every listing doubles and a fork is eventually opened
        as a ``.mat``, failing inside h5py far from the listing that caused it.
        """
        root, _, _ = standin_corpus
        archive = str(tmp_path / "forked.zip")
        with zipfile.ZipFile(archive, "w") as handle:
            handle.write(
                os.path.join(root, uvdoc.UVDOC_ROOT, uvdoc.GRID2D_DIR, "geo_0.mat"),
                f"{uvdoc.UVDOC_ROOT}/{uvdoc.GRID2D_DIR}/geo_0.mat",
            )
            handle.writestr(
                f"__MACOSX/{uvdoc.UVDOC_ROOT}/{uvdoc.GRID2D_DIR}/geo_0.mat", b"junk"
            )
        with UVDocSource(archive) as source:
            assert source.geometry_names() == ["geo_0"], (
                "the __MACOSX fork carrying the SAME basename must not appear"
            )

    def test_the_sample_to_geometry_mapping_comes_from_the_metadata(
        self, standin_source
    ):
        assert standin_source.geometry_for_sample("00000") == "geo_0"

    def test_a_missing_member_is_named_in_the_error(self, standin_source):
        with pytest.raises(UVDocError, match="grid2d/nope.mat"):
            read_grid2d(standin_source, "nope")

    def test_a_path_that_is_neither_is_refused(self, tmp_path):
        with pytest.raises(UVDocError, match="no such"):
            UVDocSource(str(tmp_path / "absent"))
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(UVDocError, match="holds no"):
            UVDocSource(str(empty))

    def test_close_is_idempotent(self, standin_corpus):
        root, _, _ = standin_corpus
        source = UVDocSource(root)
        source.close()
        source.close()


# ---------------------------------------------------------------------------
# 7. ARCHIVE-GATED: the real corpus is shaped the way this module assumes.
# ---------------------------------------------------------------------------


@requires_archive
class TestTheStagedArchiveIsShapedTheWayThisModuleAssumes:
    """Every number in ``uvdoc.py``'s docstring, re-derived from the archive.

    SKIPPED without the 27.5 GB ``UVDoc_final.zip``. Nothing else in this suite
    needs it.
    """

    @pytest.fixture(scope="class")
    def archive(self):
        with UVDocSource(UVDOC_ARCHIVE) as source:
            yield source

    def test_the_member_counts_are_what_the_docstring_claims(self, archive):
        assert len(archive.geometry_names()) == 4032
        assert len(archive.sample_ids()) == 20000

    def test_the_mat_files_really_are_matlab_v7_3(self, archive):
        header = archive.read_bytes(
            f"{uvdoc.GRID2D_DIR}/{archive.geometry_names()[0]}.mat"
        )[:16]
        assert header.startswith(b"MATLAB 7.3 MAT-f"), (
            f"scipy.io.loadmat handles <7.3 only; header was {header!r}"
        )

    def test_the_lattice_is_89_by_61_in_absolute_pixels(self, archive):
        grid = read_grid2d(archive, archive.geometry_names()[0])
        assert grid.shape == (uvdoc.UVDOC_GRID_ROWS, uvdoc.UVDOC_GRID_COLS, 2)
        assert grid.dtype == np.float64
        assert 0.0 < grid[..., 0].max() < uvdoc.UVDOC_IMAGE_WIDTH + 32
        assert uvdoc.UVDOC_IMAGE_WIDTH < grid[..., 1].max() < (
            uvdoc.UVDOC_IMAGE_HEIGHT + 32
        )

    def test_the_frame_is_488_by_712(self, archive):
        mask = read_segmentation(archive, archive.geometry_names()[0])
        assert mask.shape == (uvdoc.UVDOC_IMAGE_HEIGHT, uvdoc.UVDOC_IMAGE_WIDTH)

    def test_the_lattice_domain_is_the_rectified_grid_per_uvdocs_own_uvmap(
        self, archive
    ):
        """The direction check, against an artefact this module does not produce.

        ``uvmap`` is UVDoc's own dense ``(u, v)`` of the distorted frame. Read at
        the pixel a control point names, it must return that control point's own
        lattice coordinates -- which is what makes ``grid2d`` the BACKWARD map
        (rectified domain, distorted values) rather than its inverse.
        """
        name = archive.geometry_names()[0]
        grid = read_grid2d(archive, name)
        uv_map = uvdoc.read_uvmap(archive, name)
        rows, cols = grid.shape[:2]
        height, width = uv_map.shape[:2]
        checked = 0
        for row in (rows // 4, rows // 2, 3 * rows // 4):
            for col in (cols // 4, cols // 2, 3 * cols // 4):
                x, y = grid[row, col]
                read = uv_map[
                    int(np.clip(round(y), 0, height - 1)),
                    int(np.clip(round(x), 0, width - 1)),
                ]
                if not np.isfinite(read).all():
                    continue
                np.testing.assert_allclose(
                    read, [col / (cols - 1), row / (rows - 1)], atol=5e-3
                )
                checked += 1
        assert checked >= 6, f"only {checked} interior control points were on-page"


@requires_archive
class TestTheInversionAgreesWithUVDocsOwnUvMap:
    """The independent oracle for ``g``: UVDoc's renderer, not this module.

    SKIPPED without the archive.
    """

    def test_g_reproduces_the_shipped_uvmap_on_the_page(self):
        with UVDocSource(UVDOC_ARCHIVE) as source:
            name = source.geometry_names()[0]
            grid = read_grid2d(source, name)
            segmentation = read_segmentation(source, name)
            uv_map = uvdoc.read_uvmap(source, name)
        height, width = segmentation.shape
        f_gt = densify_backward_map(grid, height, width)
        g, outside = invert_backward_map(f_gt)
        reference = np.stack(
            [uv_map[..., 0] * width, uv_map[..., 1] * height], axis=-1
        )
        on_page = (segmentation > 0) & np.isfinite(uv_map[..., 0]) & (~outside)
        error = np.linalg.norm(g - reference, axis=-1)[on_page]
        assert error.mean() < 1.0 and np.percentile(error, 99) < 2.0, (
            f"the fitted inverse must agree with UVDoc's own dense uv map: "
            f"mean {error.mean():.3f} px, p99 {np.percentile(error, 99):.3f} px "
            f"(MEASURED: 0.52 / 0.69 px over {on_page.sum()} page pixels)"
        )

    def test_a_real_geometry_passes_the_shipped_gate(self):
        with UVDocSource(UVDOC_ARCHIVE) as source:
            geometry = load_geometry(
                source, source.geometry_names()[0], np.random.default_rng(0)
            )
        assert not is_unusable(geometry.quality), geometry.quality
        assert_is_backward_map(geometry.f_gt, 288, 288, "uvdoc/real")
        assert_is_forward_map(geometry.g, 288, 288, "uvdoc/real")
