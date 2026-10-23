"""Tests for :mod:`train.topolm.plotting`.

Plotting is the one part of this trainer whose bugs are INVISIBLE in the JSON
report, so these tests are about the two ways a topographic figure can lie:

- **the shared colour scale.** Per-panel autoscaling renders noise and signal at
  identical contrast, which is the easiest way to manufacture a topographic
  result out of an untrained model. A row that cannot share a scale must be
  visible as such.
- **the significance overlay's direction.** ``contourf`` fills the UNMASKED
  cells, so the obvious way to write "hatch the rejected region" hatches the
  survivors instead. That inversion is invisible in a render and wrong in every
  pixel, so it is pinned by measuring which cells the hatch covers.
"""

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from dl_techniques.metrics.topographic_selectivity import grow_clusters
from train.topolm.plotting import (
    _draw_boundaries,
    _shared_limit,
    plot_cluster_overlay,
    plot_morans_i_profile,
    plot_t_maps,
)

GRID = 8


def _t_maps(n_taps=3, seed=0):
    rng = np.random.default_rng(seed)
    return {
        f"blocks.{i}.ffn/attn_pre": rng.normal(size=(GRID, GRID))
        for i in range(n_taps)
    }


def _capture_axes(*args, **kwargs):
    """Call a plotting function and hand back its ``(fig, axes)``.

    The module closes the figure on every path, so the artists are only
    inspectable if the figure is caught at creation. Intercepting
    ``plt.subplots`` is the seam; ``contourf`` artists are found through
    ``ax.collections``, and a ``QuadContourSet`` carries its hatch on the set
    rather than on a per-level collection, hence ``_hatched``.
    """
    import train.topolm.plotting as module

    holder = {}
    original = module.plt.subplots

    def capturing(*inner_args, **inner_kwargs):
        result = original(*inner_args, **inner_kwargs)
        holder["fig"], holder["axes"] = result
        return result

    module.plt.subplots = capturing
    try:
        fn_name = kwargs.pop("_fn")
        getattr(module, fn_name)(*args, **kwargs)
    finally:
        module.plt.subplots = original
    return holder["fig"], holder["axes"]


def _hatched(axis):
    """Contour overlays on ``axis`` that carry a hatch pattern.

    A ``QuadContourSet`` is one artist whose ``get_hatch()`` returns ``None``
    even when its polygons are hatched -- the pattern lives on the set. Reading
    it from the set is what makes the direction assertion below possible.
    """
    found = []
    for collection in axis.collections:
        hatch = None
        if hasattr(collection, "get_hatch"):
            hatch = collection.get_hatch()
        if not hatch:
            hatch = getattr(collection, "hatches", None)
        if hatch and any(pattern for pattern in hatch):
            found.append(collection)
    return found


class TestTheSharedColourScale:
    def test_every_panel_gets_the_same_limits(self):
        """The property that makes a row comparable, asserted on the artists.

        Reading ``vmin``/``vmax`` off the rendered image is what a reader of the
        figure experiences; anything less direct would pass on a figure that
        rescales per panel.
        """
        maps = _t_maps(n_taps=4)
        # One panel with a large blob. If limits were per-panel, this one would
        # be squashed relative to the rest.
        maps["blocks.0.ffn/attn_pre"][2:5, 2:5] += 50.0

        fig, axes = _capture_axes(maps, _fn="plot_t_maps")
        flat = [ax for row in np.atleast_2d(axes) for ax in row]
        images = [ax.images[0] for ax in flat if len(ax.images)]
        assert len(images) == 4, len(images)

        limits = {(im.get_clim()[0], im.get_clim()[1]) for im in images}
        assert len(limits) == 1, (
            f"panels have {len(limits)} different colour limits {limits}; a row "
            f"whose panels autoscale independently cannot be read as one figure"
        )
        plt.close(fig)

    def test_the_limit_is_symmetric_about_zero(self):
        """A diverging map needs a symmetric range, or zero is not the midpoint.

        An asymmetric limit puts the neutral colour somewhere other than "no
        difference", so a positive and a negative effect of equal size render
        with different apparent weight.
        """
        maps = {"a": np.linspace(-1.0, 3.0, GRID * GRID).reshape(GRID, GRID)}
        low, high = -_shared_limit(maps, 100.0), _shared_limit(maps, 100.0)
        assert low == -high
        assert high == pytest.approx(3.0)

    def test_a_single_outlier_does_not_set_the_limit(self):
        """Percentile, not max: one diverging cell must not flatten the map.

        With a max-based limit every ordinary cell lands in one end of the
        colormap and the figure carries no information about where structure is.
        """
        values = np.zeros((GRID, GRID))
        values[0, 0] = 1000.0
        maps = {"a": values}
        assert _shared_limit(maps, 99.0) < 1000.0

    def test_an_all_zero_map_still_gets_a_positive_limit(self):
        """Zero would make matplotlib pick a degenerate scale or raise."""
        assert _shared_limit({"a": np.zeros((GRID, GRID))}, 99.0) > 0.0

    def test_an_all_nan_map_does_not_propagate_nan(self):
        maps = {"a": np.full((GRID, GRID), np.nan)}
        limit = _shared_limit(maps, 99.0)
        assert np.isfinite(limit) and limit > 0.0


class TestTheSignificanceOverlayDirection:
    def test_the_hatch_covers_rejected_cells_and_spares_surviving_ones(self):
        """The inversion this module exists to prevent, pinned by measurement.

        ``contourf`` fills the UNMASKED cells, so
        ``masked_where(~sig, ones)`` fills the SURVIVORS. Verified here by
        locating the hatched region on the axes and comparing it against the
        mask, rather than by reading the image.
        """
        sig = np.zeros((GRID, GRID), dtype=bool)
        sig[3:5, 3:5] = True

        fig, axes = _capture_axes(
            {"t": np.zeros((GRID, GRID))},
            sig_grids={"t": sig},
            _fn="plot_t_maps",
        )
        axis = np.atleast_2d(axes)[0, 0]
        hatched = _hatched(axis)
        assert len(hatched) == 1, (
            f"expected one hatched region, found {len(hatched)}; the overlay was "
            f"skipped or doubled"
        )

        extent = hatched[0].get_paths()[0].get_extents()
        plt.close(fig)

        # The significant block is rows/cols 3:5, i.e. data coords ~3.0-5.0.
        # A correct rejected-region hatch spans the panel and leaves that hole.
        spans_panel = (
            extent.x0 <= 0.5
            and extent.y0 <= 0.5
            and extent.x1 >= GRID - 1.5
            and extent.y1 >= GRID - 1.5
        )
        assert spans_panel, (
            f"the hatch covers only {extent}, not the rejected region; if this "
            f"fails the mask argument was inverted and the SURVIVORS are hatched"
        )
        assert not (
            extent.x0 > 2.5 and extent.x1 < 5.5 and extent.y0 > 2.5 and extent.y1 < 5.5
        ), "the hatch is confined to the significant block"

    def test_no_hatch_is_drawn_when_everything_is_significant(self):
        """Nothing to reject means no overlay, not a hatch over the whole map."""
        fig = plt.figure()
        plot_t_maps(
            {"t": np.zeros((GRID, GRID))},
            sig_grids={"t": np.ones((GRID, GRID), dtype=bool)},
        )
        plt.close(fig)

    def test_a_missing_sig_grid_is_not_an_error(self):
        assert (
            plot_t_maps({"t": np.zeros((GRID, GRID))}, sig_grids=None) is None
        )

    def test_a_sig_grid_for_an_absent_tap_is_ignored(self):
        assert (
            plot_t_maps(
                {"t": np.zeros((GRID, GRID))},
                sig_grids={"other": np.ones((GRID, GRID), dtype=bool)},
            )
            is None
        )


class TestClusterBoundaries:
    def test_a_cluster_touching_the_edge_is_outlined_on_that_edge(self):
        """The zero frame makes edge-touching clusters close.

        Without it a cluster on the border is drawn as an open shape, which
        reads as a cluster that runs off the grid.

        Rewritten deliberately for D-002. It used to assert the outline at the
        raw integers ``0`` (y and x), the PADDED-grid indices, which is where the
        drawer put a cluster in the top-left corner and is one cell off: under
        ``imshow`` the image's own top and left edges are at ``-0.5``, so that is
        where the outline of a corner cluster must run. The geometry oracle in
        ``TestClusterBoundariesSitOnTheirCells`` compares every segment; this
        test keeps the edge-closure claim in its original small form.
        """
        labels = np.zeros((GRID, GRID), dtype=int)
        labels[0:3, 0:3] = 1

        fig, axis = plt.subplots()
        _draw_boundaries(axis, labels)
        lines = axis.get_lines()
        plt.close(fig)

        assert lines, "no boundaries drawn at all"
        assert any(line.get_ydata()[0] == -0.5 for line in lines), (
            "no line on the top edge of a cluster that touches it"
        )
        assert any(line.get_xdata()[0] == -0.5 for line in lines), (
            "no line on the left edge of a cluster that touches it"
        )

    def test_a_single_cell_cluster_is_outlined_on_all_four_sides(self):
        """Rewritten deliberately for D-002. It used to pin ``{2.0, 3.0}`` for a
        cell at (2, 2), the padded-grid integers, which are one cell to the right
        and below the cell. Cell (2, 2) spans 1.5 to 2.5 on both axes, so its four
        sides are at 1.5 and 2.5."""
        labels = np.zeros((GRID, GRID), dtype=int)
        labels[2, 2] = 1

        fig, axis = plt.subplots()
        _draw_boundaries(axis, labels)
        lines = axis.get_lines()
        plt.close(fig)

        rows = {line.get_ydata()[0] for line in lines}
        cols = {line.get_xdata()[0] for line in lines}
        assert rows == {1.5, 2.5}, rows
        assert cols == {1.5, 2.5}, cols

    def test_an_interior_cluster_gets_only_its_own_border(self):
        """A hollow centre is outlined at the hole, not as a filled blob.

        Segments are drawn per CELL edge, so a 4x4 ring around a 2x2 hole has
        12 horizontal and 12 vertical edges: 4 along the outer top, 4 along the
        outer bottom, and 2 on each side of the hole. Asserting the counts per
        ROW is what makes this fail if the hole is drawn over or skipped -- a
        bare total of 24 would also match a cluster whose border was drawn twice.

        Rewritten deliberately for D-002: the per-row counts are unchanged and
        the row keys moved. They used to be the padded-grid integers 2, 3, 5, 6,
        which sit one cell below the edges they stand for. The edge between
        image rows 1 and 2 is at ``y = 1.5``, so the keys are 1.5, 2.5, 4.5, 5.5.
        """
        labels = np.zeros((GRID, GRID), dtype=int)
        labels[2:6, 2:6] = 1
        labels[3:5, 3:5] = 2

        fig, axis = plt.subplots()
        _draw_boundaries(axis, labels)
        lines = axis.get_lines()
        plt.close(fig)

        horizontal = [
            line for line in lines if line.get_ydata()[0] == line.get_ydata()[1]
        ]
        by_row = {}
        for line in horizontal:
            by_row.setdefault(line.get_ydata()[0], []).append(line)
        assert {row: len(seg) for row, seg in sorted(by_row.items())} == {
            1.5: 4,
            2.5: 2,
            4.5: 2,
            5.5: 4,
        }, {row: len(seg) for row, seg in sorted(by_row.items())}
        assert len(horizontal) == 12, len(horizontal)

    def test_an_empty_label_grid_draws_nothing(self):
        fig, axis = plt.subplots()
        _draw_boundaries(axis, np.zeros((GRID, GRID), dtype=int))
        assert axis.get_lines() == []
        plt.close(fig)


class TestFiguresWriteAndDegradeGracefully:
    def test_a_single_tap_writes_a_file(self, tmp_path):
        path = plot_t_maps(
            _t_maps(n_taps=1), output_path=str(tmp_path / "one.png")
        )
        assert path is not None
        assert (tmp_path / "one.png").stat().st_size > 0

    def test_writing_creates_missing_directories(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "fig.png"
        assert plot_t_maps(_t_maps(1), output_path=str(target)) is not None
        assert target.exists()

    def test_no_output_path_means_no_file_and_no_error(self, tmp_path):
        assert plot_t_maps(_t_maps(1), output_path=None) is None
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize(
        "n_taps", [1, 2, 7, 12, 13, 25]
    )
    def test_any_panel_count_lays_out(self, tmp_path, n_taps):
        """Wrapping must hold for counts either side of a row boundary."""
        assert plot_t_maps(
            _t_maps(n_taps), output_path=str(tmp_path / f"{n_taps}.png")
        ) is not None

    def test_a_zero_cluster_overlay_still_writes(self, tmp_path):
        assert plot_cluster_overlay(
            np.zeros((GRID, GRID), dtype=int),
            output_path=str(tmp_path / "empty.png"),
        ) is not None

    def test_a_single_cluster_overlay_writes(self, tmp_path):
        """One colour plus the unassigned background; matplotlib rejects less."""
        labels = np.zeros((GRID, GRID), dtype=int)
        labels[1:4, 1:4] = 1
        assert plot_cluster_overlay(
            labels, output_path=str(tmp_path / "one.png")
        ) is not None

    def test_a_morans_profile_with_nan_values_writes(self, tmp_path):
        """Moran's I is NaN on a constant map, which is a real outcome."""
        assert plot_morans_i_profile(
            {"a": float("nan"), "b": 0.2, "c": 0.3},
            output_path=str(tmp_path / "moran.png"),
        ) is not None

    def test_the_profile_x_axis_follows_insertion_order(self, tmp_path):
        """Depth is positional, so a renamed path cannot silently reorder it."""
        names = {"deep/nested/name_9": 0.9, "name_0": 0.1, "name_5": 0.5}
        assert plot_morans_i_profile(
            names, output_path=str(tmp_path / "moran.png")
        ) is not None


class TestInputValidation:
    def test_empty_t_maps_is_an_error(self):
        with pytest.raises(ValueError, match="empty"):
            plot_t_maps({})

    def test_a_non_2d_map_is_an_error_naming_the_tap(self):
        with pytest.raises(ValueError, match="t-only"):
            plot_t_maps({"t-only": np.zeros(GRID)})

    def test_maps_that_disagree_in_shape_are_an_error_naming_the_tap(self):
        with pytest.raises(ValueError, match="blocks.1"):
            plot_t_maps(
                {
                    "blocks.0.ffn/attn_pre": np.zeros((GRID, GRID)),
                    "blocks.1.ffn/attn_pre": np.zeros((GRID + 2, GRID)),
                }
            )

    def test_a_non_2d_label_grid_is_an_error(self):
        with pytest.raises(ValueError, match="2-D"):
            plot_cluster_overlay(np.zeros(GRID, dtype=int))


class TestTheFigureAgreesWithTheAnalysis:
    def test_clusters_drawn_are_the_clusters_grown(self, tmp_path):
        """End to end: the pipeline's clusters are the ones on the figure.

        Not a property test -- a wiring check. The figure is drawn from the
        label grid the report counts, so this fails if the two are ever sourced
        from different arrays.
        """
        values = np.random.default_rng(3).normal(size=(GRID, GRID))
        values[2:5, 2:5] += 8.0
        sig = values > 2.5

        clusters, labels = grow_clusters(
            values, sig, sign=1, min_size=2, connectivity="queen"
        )
        assert clusters, "the fixture found no cluster, so this proves nothing"

        fig, axes = _capture_axes(
            {"t": values},
            sig_grids={"t": sig},
            label_grids={"t": labels},
            output_path=str(tmp_path / "fig.png"),
            _fn="plot_t_maps",
        )
        axis = np.atleast_2d(axes)[0, 0]
        # The cluster's own cells must be outlined, so boundary segments exist on
        # the axes and the figure is not just a hatched heatmap.
        assert axis.get_lines(), "no cluster boundaries were drawn"
        assert len(_hatched(axis)) == 1
        plt.close(fig)


# =====================================================================
# plan-2026-10-23T190557-972a9ac5, step 1: pins and RED tests
#
# The old ``TestClusterBoundaries`` above pinned the integers the drawer emitted,
# and the drawer was wrong (D-002): the new class compares the drawn segments with
# a brute-force edge set derived from cell geometry. Step 3 fixed the drawer and
# rewrote the old tests to the corrected coordinates, with the reason in each
# docstring.
# =====================================================================

import hashlib
from collections import Counter

import matplotlib.image as mpimg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure


def _pieces(x0, y0, x1, y1):
    """Split one axis-aligned segment into unit pieces keyed by (kind, fixed, start).

    Keyed this way a builder that merges collinear runs and one that draws one
    segment per cell edge are compared on the same footing, while a missing,
    shifted or doubled edge still shows (the keys are compared as a multiset).
    """
    assert x0 == x1 or y0 == y1, "a boundary segment must be axis-aligned"
    if y0 == y1:
        low, high = sorted((x0, x1))
        return [("h", round(float(y0), 6), round(float(low) + k, 6))
                for k in range(int(round(high - low)))]
    low, high = sorted((y0, y1))
    return [("v", round(float(x0), 6), round(float(low) + k, 6))
            for k in range(int(round(high - low)))]


def _drawn_pieces(axis):
    """Every boundary segment on ``axis`` as a multiset of unit pieces."""
    found = Counter()
    for line in axis.get_lines():
        xs, ys = line.get_xdata(), line.get_ydata()
        for (xa, ya), (xb, yb) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
            found.update(_pieces(xa, ya, xb, yb))
    for collection in axis.collections:
        if isinstance(collection, LineCollection):
            for segment in collection.get_segments():
                for (xa, ya), (xb, yb) in zip(segment[:-1], segment[1:]):
                    found.update(_pieces(xa, ya, xb, yb))
    return found


def _expected_pieces(labels):
    """The cell-edge set, from geometry alone.

    Cell ``(r, c)`` spans ``x`` in ``[c - 0.5, c + 0.5]`` and ``y`` in
    ``[r - 0.5, r + 0.5]`` under ``imshow``. An edge of a cell is a boundary when
    the cell across it carries a different label, the outside of the grid
    counting as label 0 -- so a cluster touching the border is closed there, and
    two zero cells never draw anything between them.
    """
    labels = np.asarray(labels)
    height, width = labels.shape

    def label_at(row, col):
        if 0 <= row < height and 0 <= col < width:
            return labels[row, col]
        return 0

    edges = set()
    for row in range(height):
        for col in range(width):
            here = labels[row, col]
            if label_at(row - 1, col) != here:
                edges.add(("h", row - 0.5, col - 0.5))
            if label_at(row + 1, col) != here:
                edges.add(("h", row + 0.5, col - 0.5))
            if label_at(row, col - 1) != here:
                edges.add(("v", col - 0.5, row - 0.5))
            if label_at(row, col + 1) != here:
                edges.add(("v", col + 0.5, row - 0.5))
    return Counter(
        {(kind, round(fixed, 6), round(start, 6)): 1 for kind, fixed, start in edges}
    )


def _boundary_fixtures():
    single = np.zeros((8, 8), dtype=int)
    single[3, 4] = 1
    border = np.zeros((8, 8), dtype=int)
    border[0:3, 0:3] = 1
    corner = np.zeros((8, 8), dtype=int)
    corner[7, 7] = corner[7, 6] = 1
    ring = np.zeros((8, 8), dtype=int)
    ring[2:6, 2:6] = 1
    ring[3:5, 3:5] = 2
    touching = np.zeros((8, 8), dtype=int)
    touching[1:4, 1:4] = 1
    touching[4:7, 1:4] = 2          # two clusters sharing a whole edge
    touching[2, 5] = 3              # and a lone third one
    return {"single": single, "border": border, "corner": corner,
            "ring": ring, "touching": touching}


class TestClusterBoundariesSitOnTheirCells:
    """D-002. RED on the baseline: the integers the old code draws are shifted off
    the cells (MEASURED: a cell at row 3, column 4 spans x 3.5 to 4.5 and y 2.5 to
    3.5, but the segments sat at x 5 to 6 and 4 to 5)."""

    @pytest.mark.parametrize("name", list(_boundary_fixtures()))
    def test_the_drawn_edges_are_the_cell_edges_of_the_label_grid(self, name):
        labels = _boundary_fixtures()[name]
        fig, axis = plt.subplots()
        axis.imshow(np.zeros(labels.shape))
        _draw_boundaries(axis, labels)
        drawn = _drawn_pieces(axis)
        plt.close(fig)

        expected = _expected_pieces(labels)
        assert drawn == expected, {
            "missing": sorted((expected - drawn).elements())[:6],
            "unexpected": sorted((drawn - expected).elements())[:6],
        }

    def test_one_cell_at_row_3_col_4_is_outlined_on_x_3p5_to_4p5_and_y_2p5_to_3p5(self):
        """The literal case from the measurement, without the brute-force helper."""
        labels = np.zeros((8, 8), dtype=int)
        labels[3, 4] = 1
        fig, axis = plt.subplots()
        axis.imshow(np.zeros((8, 8)))
        _draw_boundaries(axis, labels)
        drawn = _drawn_pieces(axis)
        plt.close(fig)
        assert drawn == Counter({
            ("h", 2.5, 3.5): 1,   # top edge, x from 3.5 to 4.5
            ("h", 3.5, 3.5): 1,   # bottom edge
            ("v", 3.5, 2.5): 1,   # left edge, y from 2.5 to 3.5
            ("v", 4.5, 2.5): 1,   # right edge
        })

    def test_a_cluster_on_the_border_is_closed_on_the_edge_of_the_image(self):
        """The image's own edge is x = -0.5 / y = -0.5, so that is where the
        outline of a cluster in the top-left corner must run."""
        labels = np.zeros((8, 8), dtype=int)
        labels[0:3, 0:3] = 1
        fig, axis = plt.subplots()
        axis.imshow(np.zeros((8, 8)))
        _draw_boundaries(axis, labels)
        drawn = _drawn_pieces(axis)
        plt.close(fig)
        assert {k for k in drawn if k[0] == "h" and k[1] == -0.5} == {
            ("h", -0.5, -0.5), ("h", -0.5, 0.5), ("h", -0.5, 1.5)
        }
        assert {k for k in drawn if k[0] == "v" and k[1] == -0.5} == {
            ("v", -0.5, -0.5), ("v", -0.5, 0.5), ("v", -0.5, 1.5)
        }

    def test_the_t_map_panel_draws_the_outline_where_the_drawer_does(self, tmp_path):
        """Through ``plot_t_maps`` rather than the helper: no caller may add or
        remove an offset between the drawer and the panel."""
        labels = _boundary_fixtures()["touching"]
        fig, axes = _capture_axes(
            {"t": np.zeros((8, 8))},
            label_grids={"t": labels},
            output_path=str(tmp_path / "panel.png"),
            _fn="plot_t_maps",
        )
        drawn = _drawn_pieces(np.atleast_2d(axes)[0, 0])
        plt.close(fig)
        assert drawn == _expected_pieces(labels)

    def test_the_cluster_overlay_draws_the_outline_where_the_drawer_does(self, tmp_path):
        labels = _boundary_fixtures()["ring"]
        fig, axes = _capture_axes(
            labels, output_path=str(tmp_path / "overlay.png"), _fn="plot_cluster_overlay"
        )
        drawn = _drawn_pieces(axes)
        plt.close(fig)
        assert drawn == _expected_pieces(labels)


# ---------------------------------------------------------------------
# 2-D figures WITHOUT clusters must not move (pixel pins)
# ---------------------------------------------------------------------


def _pixel_digest(path):
    """Shape and SHA-256 of the decoded RGBA pixels (same environment pin: the
    hash depends on the matplotlib, freetype and font build, not just on us)."""
    pixels = mpimg.imread(path)
    return pixels.shape, hashlib.sha256(
        np.ascontiguousarray(pixels).tobytes()
    ).hexdigest()[:16]


class TestTwoDimensionalUnchangedPixels:
    """The fix for D-002 and the rank dispatch for 3-D may change a figure with
    cluster outlines; they may not change a pixel of one without. Measured on
    the unmodified code (matplotlib 3.10.0, Agg), identical across processes."""

    @staticmethod
    def _maps():
        rng = np.random.default_rng(31)
        maps = {f"blocks.{i}.ffn/attn_pre": rng.normal(size=(8, 8)) * (1 + i)
                for i in range(5)}
        return maps, {k: np.abs(v) > 1.5 for k, v in maps.items()}

    def test_a_t_map_row_without_overlays(self, tmp_path):
        maps, _ = self._maps()
        path = plot_t_maps(maps, output_path=str(tmp_path / "a.png"))
        assert _pixel_digest(path) == ((547, 1983, 4), "351efc58717dda02")

    def test_a_t_map_row_with_the_hatch_and_the_legend_but_no_clusters(self, tmp_path):
        maps, sig = self._maps()
        path = plot_t_maps(
            maps, sig_grids=sig, arm="raw", condition_a="a", condition_b="b",
            output_path=str(tmp_path / "b.png"),
        )
        assert _pixel_digest(path) == ((547, 1983, 4), "eab7091b764aa543")

    def test_an_all_zero_label_grid_changes_nothing(self, tmp_path):
        """No cluster, no outline: the same pixels as passing no label grids."""
        maps, sig = self._maps()
        empty = {k: np.zeros((8, 8), dtype=int) for k in maps}
        path = plot_t_maps(
            maps, sig_grids=sig, label_grids=empty, arm="raw",
            condition_a="a", condition_b="b", output_path=str(tmp_path / "c.png"),
        )
        assert _pixel_digest(path) == ((547, 1983, 4), "eab7091b764aa543")

    def test_the_moran_profile(self, tmp_path):
        maps, _ = self._maps()
        path = plot_morans_i_profile(
            {k: i / 10 - 0.1 for i, k in enumerate(maps)},
            output_path=str(tmp_path / "d.png"), arm="raw",
        )
        assert _pixel_digest(path) == ((463, 783, 4), "2bf6cdf35fed80ed")


class TestTwoDimensionalUnchangedInputValidation:
    """What the rank dispatch must keep: a rank that is neither 2 nor 3 is an
    error that names the tap (or the label grid) and says ``2-D``."""

    def test_a_4d_map_names_the_tap_and_says_2d(self):
        with pytest.raises(ValueError) as caught:
            plot_t_maps({"tap-4d": np.zeros((2, 2, 2, 2))})
        assert "tap-4d" in str(caught.value) and "2-D" in str(caught.value)

    def test_a_4d_label_grid_says_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            plot_cluster_overlay(np.zeros((2, 2, 2, 2), dtype=int))

    def test_a_2d_map_followed_by_a_3d_map_names_the_second(self):
        with pytest.raises(ValueError, match="tap-3d"):
            plot_t_maps({"tap-2d": np.zeros((4, 4)), "tap-3d": np.zeros((4, 4, 4))})


# ---------------------------------------------------------------------
# 3-D figures (D-004)
# ---------------------------------------------------------------------

_VOLUME = (7, 8, 14)


def _captured_figure(monkeypatch, call):
    """Run ``call()`` and hand back the figure it saved, before it was closed.

    Intercepting ``Figure.savefig`` instead of ``plt.subplots`` keeps the test
    independent of how the 3-D drawer builds its layout.
    """
    holder = {}
    original = Figure.savefig

    def recording(self, *args, **kwargs):
        holder["fig"] = self
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", recording)
    call()
    return holder["fig"]


def _image_axes(fig):
    return [ax for ax in fig.axes if ax.images]


def _volume_fixture():
    """One tap: significant +4 at (2, 3, 4), significant -6 at (5, 1, 9), and a
    larger +9 at (0, 7, 13) that is NOT significant -- so an argmax over t, over
    |t|, or over everything each pick a different voxel."""
    volume = np.random.default_rng(1).normal(scale=0.1, size=_VOLUME)
    volume[2, 3, 4] = 4.0
    volume[5, 1, 9] = -6.0
    volume[0, 7, 13] = 9.0
    significant = np.zeros(_VOLUME, dtype=bool)
    significant[4:7, 0:3, 8:11] = True      # the block around (5, 1, 9)
    significant[2, 3, 4] = True
    return volume, significant


class TestThreeDimensionalFigures:
    """RED on the baseline: ``plot_t_maps`` and ``plot_cluster_overlay`` raise on a
    3-D array (``_panel_shape`` demands 2-D), and the 3-D drawers do not exist."""

    def test_the_drawers_exist(self):
        from train.topolm.plotting import (  # noqa: F401
            plot_cluster_montage_3d,
            plot_t_maps_3d,
        )

    def test_a_3d_t_map_row_is_dispatched_and_written(self, tmp_path):
        volume, significant = _volume_fixture()
        maps = {f"blocks.{i}.ffn/attn_pre": volume * (1 + i) for i in range(3)}
        path = plot_t_maps(
            maps, sig_grids={k: significant for k in maps},
            output_path=str(tmp_path / "t3d.png"),
        )
        assert path is not None and (tmp_path / "t3d.png").stat().st_size > 0

    def test_each_tap_gets_three_panels(self, monkeypatch, tmp_path):
        volume, significant = _volume_fixture()
        maps = {f"blocks.{i}.ffn/attn_pre": volume for i in range(4)}
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(
                maps, sig_grids={k: significant for k in maps},
                output_path=str(tmp_path / "t3d.png"),
            ),
        )
        assert len(_image_axes(fig)) == 3 * len(maps)
        plt.close(fig)

    def test_the_crosshair_is_the_largest_significant_abs_t(self, monkeypatch, tmp_path):
        """(5, 1, 9) is the answer: -6 beats +4 in |t| among the significant
        voxels, and the +9 voxel is ignored because it is not significant. The
        three panels are the planes through it, in axis order 0, 1, 2; the
        (7, 8, 14) shape tells the planes apart, and a plane may be shown
        transposed."""
        volume, significant = _volume_fixture()
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(
                {"blocks.0.ffn/attn_pre": volume},
                sig_grids={"blocks.0.ffn/attn_pre": significant},
                output_path=str(tmp_path / "t3d.png"),
            ),
        )
        planes = [volume[5, :, :], volume[:, 1, :], volume[:, :, 9]]
        shown = [np.asarray(ax.images[0].get_array()) for ax in _image_axes(fig)]
        plt.close(fig)
        assert len(shown) == 3
        for index, (image, plane) in enumerate(zip(shown, planes)):
            assert np.array_equal(image, plane) or np.array_equal(image, plane.T), (
                f"panel {index} is not the plane through (5, 1, 9) on axis {index}"
            )

    def test_all_panels_share_the_pooled_colour_limit(self, monkeypatch, tmp_path):
        volume, significant = _volume_fixture()
        maps = {"a/attn_pre": volume, "b/attn_pre": 3.0 * volume}
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(maps, output_path=str(tmp_path / "t3d.png")),
        )
        limits = {ax.images[0].get_clim() for ax in _image_axes(fig)}
        plt.close(fig)
        pooled = np.concatenate([np.abs(v).ravel() for v in maps.values()])
        expected = float(np.percentile(pooled, 99.0))
        assert limits == {(-expected, expected)}, limits

    def test_the_hatch_covers_the_rejected_cells_of_every_plane(self, monkeypatch, tmp_path):
        """Same direction as in 2-D (``contourf`` fills the unmasked cells): on each
        plane the hatch spans the panel and leaves a hole at the significant block."""
        volume, significant = _volume_fixture()
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(
                {"t": volume}, sig_grids={"t": significant},
                output_path=str(tmp_path / "t3d.png"),
            ),
        )
        for ax in _image_axes(fig):
            rows, cols = np.asarray(ax.images[0].get_array()).shape
            hatched = _hatched(ax)
            assert len(hatched) == 1, "the plane has no hatch, or two"
            extent = hatched[0].get_paths()[0].get_extents()
            assert extent.x0 <= 0.5 and extent.y0 <= 0.5
            assert extent.x1 >= cols - 1.5 and extent.y1 >= rows - 1.5
        plt.close(fig)

    def test_a_3d_cluster_grid_is_a_montage_with_one_panel_per_slice(
        self, monkeypatch, tmp_path
    ):
        labels = np.zeros((5, 6, 6), dtype=int)
        labels[1:4, 1:3, 1:3] = 1
        labels[2:5, 3:6, 3:6] = 2
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_cluster_overlay(labels, output_path=str(tmp_path / "m.png")),
        )
        assert len(_image_axes(fig)) == 5
        plt.close(fig)
        assert (tmp_path / "m.png").stat().st_size > 0

    def test_the_montage_reports_the_clusters_it_could_not_colour(
        self, monkeypatch, tmp_path
    ):
        """Same truncation rule as the 2-D overlay: 14 ids, 12 coloured, the other
        two named in the figure rather than silently dropped."""
        flat = np.zeros(4 * 6 * 6, dtype=int)
        flat[np.arange(14) * 5] = np.arange(1, 15)      # 14 distinct single cells
        labels = flat.reshape(4, 6, 6)
        assert len(np.unique(labels[labels > 0])) == 14
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_cluster_overlay(labels, output_path=str(tmp_path / "m.png")),
        )
        texts = [t.get_text() for t in fig.findobj(matplotlib.text.Text)]
        plt.close(fig)
        assert any("+2 not coloured" in text for text in texts), texts
