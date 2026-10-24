"""Tests for :mod:`train.topolm.plotting`.

Plotting is the one part of this trainer whose bugs are INVISIBLE in the JSON
report, so these tests are about the two ways a topographic figure can lie:

- **the shared colour scale.** Per-panel autoscaling renders noise and signal at
  identical contrast, which is the easiest way to manufacture a topographic
  result out of an untrained model. A row that cannot share a scale must be
  visible as such.
- **the significance overlay's direction and coverage.** Hatching the survivors
  instead of the rejected cells (the first drawer's ``contourf`` filled the
  UNMASKED cells, so the obvious mask argument did exactly that) is invisible in
  a render and wrong in every pixel, and a hatch that skips cells reads as
  "significant" there. Both are pinned by measuring, cell by cell, which cells the
  hatch covers (step 4.6, D-021).
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
    plot_topography_curve,
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


#: Where inside a cell a hatch is sampled: three offsets per axis, all strictly
#: inside the cell ``[-0.5, 0.5]`` with 0.15 to spare, nine points a cell.
_CELL_SAMPLES = (-0.35, 0.0, 0.35)


def _hatch_paths(axis):
    """The outline paths, in data coordinates, of EVERY hatch-bearing artist on
    ``axis``: collections (a ``contourf`` set, a ``PatchCollection``, ...) and loose
    patches. Reads only what is drawn, never how the module built it."""
    paths = []
    for collection in _hatched(axis):
        assert collection.get_transform().contains_branch(axis.transData), (
            "a hatch collection that is not in data coordinates"
        )
        assert not np.any(collection.get_offsets()), "a hatch drawn through offsets"
        paths.extend(collection.get_paths())
    for patch in axis.patches:
        if patch.get_hatch():
            paths.append(patch.get_patch_transform().transform_path(patch.get_path()))
    return paths


def _hatch_coverage(axis, shape):
    """``(rows, cols, 9)`` boolean: which sample points of each cell lie inside a
    hatched region. Cell ``(r, c)`` is centred on ``x = c``, ``y = r`` (``imshow``)."""
    rows, cols = shape
    points = np.array([
        (col + dx, row + dy)
        for row in range(rows) for col in range(cols)
        for dy in _CELL_SAMPLES for dx in _CELL_SAMPLES
    ])
    inside = np.zeros(len(points), dtype=bool)
    for path in _hatch_paths(axis):
        inside |= path.contains_points(points)
    return inside.reshape(rows, cols, len(_CELL_SAMPLES) ** 2)


def _hatch_boxes(axis):
    """Each hatch path's bounding box, rounded: ``(x0, y0, x1, y1)``."""
    boxes = []
    for path in _hatch_paths(axis):
        extent = path.get_extents()
        boxes.append(tuple(round(float(v), 6) for v in
                           (extent.x0, extent.y0, extent.x1, extent.y1)))
    return boxes


def _assert_hatch_is_exactly(axis, significant):
    """The hatch on ``axis`` covers EVERY cell that failed BH-FDR (``~significant``)
    and NO cell that passed: all nine sample points of a rejected cell are hatched,
    none of an accepted cell's."""
    significant = np.asarray(significant, dtype=bool)
    coverage = _hatch_coverage(axis, significant.shape)
    rejected = ~significant
    missed = np.argwhere(rejected & ~coverage.all(axis=2))
    spilled = np.argwhere(~rejected & coverage.any(axis=2))
    assert missed.size == 0, (
        f"{len(missed)} of {int(rejected.sum())} rejected cells are not (fully) "
        f"hatched, first (row, col) {missed[:5].tolist()}"
    )
    assert spilled.size == 0, (
        f"the hatch covers {len(spilled)} accepted cells, first (row, col) "
        f"{spilled[:5].tolist()}"
    )


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

        Hatching the cells that PASSED instead of the cells that FAILED is
        invisible in a render and wrong in every pixel (the first drawer used
        ``contourf``, which fills the UNMASKED cells, so the obvious mask argument
        marked the survivors). Verified here by locating the hatched region on the
        axes and comparing it, cell by cell, against the mask rather than by
        reading the image.
        """
        sig = np.zeros((GRID, GRID), dtype=bool)
        sig[3:5, 3:5] = True

        fig, axes = _capture_axes(
            {"t": np.zeros((GRID, GRID))},
            sig_grids={"t": sig},
            _fn="plot_t_maps",
        )
        axis = np.atleast_2d(axes)[0, 0]
        assert _hatched(axis), "the overlay was skipped"

        _assert_hatch_is_exactly(axis, sig)
        plt.close(fig)
        # The block that PASSED (rows/cols 3:5) is the hole, not the hatch: the
        # four cells of it are not hatched at all, whatever the mask direction.
        assert not _hatch_coverage(axis, sig.shape)[3:5, 3:5].any(), (
            "the hatch covers the significant block: the SURVIVORS are hatched, "
            "the mask argument was inverted"
        )

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
    """The 2-D figures' pixel pins, RE-PINNED ON PURPOSE in step 4.2.

    Until step 4.2 these hashes pinned the old 2-D layout, and the old layout was
    the defect: the colour bar sat on top of the last panel and its title, the
    headline and the line under it collided, and a real tap path overflowed its
    panel. The pixels changed deliberately, for LAYOUT, not for content: the new
    figure is laid out in inches, has the colour bar in its own axes, and titles
    its panels with the shared-prefix-free labels. What the figure shows is pinned
    separately and independently of layout in
    :class:`TestTwoDimensionalContentDoesNotDependOnLayout`; these hashes pin the
    new layout so it cannot drift unnoticed, and the geometric tests in
    :class:`TestTwoDimensionalTitlesAreReadableForRealTapPaths` say what the layout
    must satisfy. Measured on the step-4.2 code (matplotlib 3.10.0, Agg),
    identical across processes."""

    @staticmethod
    def _maps():
        rng = np.random.default_rng(31)
        maps = {f"blocks.{i}.ffn/attn_pre": rng.normal(size=(8, 8)) * (1 + i)
                for i in range(5)}
        return maps, {k: np.abs(v) > 1.5 for k, v in maps.items()}

    def test_a_t_map_row_without_overlays(self, tmp_path):
        """Re-pinned (layout): was ((547, 1983, 4), "351efc58717dda02")."""
        maps, _ = self._maps()
        path = plot_t_maps(maps, output_path=str(tmp_path / "a.png"))
        assert _pixel_digest(path) == ((468, 1993, 4), "445e18d5a116726c")

    def test_a_t_map_row_with_the_hatch_and_the_legend_but_no_clusters(self, tmp_path):
        """Re-pinned three times. Layout (step 4.2): was ((547, 1983, 4),
        "eab7091b764aa543"), then "25ac5e61b97c81a8". Step 4.4 (D-018): the line
        under the headline now says what the hatch means ("hatched = not
        significant after BH-FDR"), which is TEXT only: the figure keeps its size
        (468 x 1993), pin "b0327169643eb99c". Step 4.6 (D-021): re-pinned to
        "dc6f9805b89805b1" because the HATCH ITSELF changed, deliberately: the old
        ``contourf`` hatch left 60 % of the rejected cells unmarked (this pin hashed
        that faulty picture), the new one hatches every rejected cell. Size, colours,
        image arrays, colour limits and outlines are unchanged (pinned
        layout-independently in
        :class:`TestTwoDimensionalContentDoesNotDependOnLayout`); the cell-by-cell
        truth of the hatch is :class:`TestTheHatchMarksEveryRejectedCell`.

        Step 4.7 (D-026): re-pinned to "2af516f6faf8f240" because the hatch is
        DENSER on purpose (``///`` instead of ``//``): at ``//`` the strokes are
        about two cells apart, so 241 of 868 rejected cells of a trained-like 2-D
        figure carried no stroke in their middle half; at ``///`` none does
        (:class:`TestEveryRejectedCellCarriesStrokes`). Nothing else moved: the
        size (468 x 1993), the image arrays, the colour limits and the outlines are
        the same, and the no-overlay figure (``445e18d5a116726c``) and the Moran
        profile (``2a787e0115cb9d6c``) are NOT re-pinned."""
        maps, sig = self._maps()
        path = plot_t_maps(
            maps, sig_grids=sig, arm="raw", condition_a="a", condition_b="b",
            output_path=str(tmp_path / "b.png"),
        )
        assert _pixel_digest(path) == ((468, 1993, 4), "2af516f6faf8f240")

    def test_an_all_zero_label_grid_changes_nothing(self, tmp_path):
        """No cluster, no outline: the same pixels as passing no label grids
        (re-pinned with the figure above, same new subtitle text and, since step
        4.6, the same per-cell hatch: "b0327169643eb99c" -> "dc6f9805b89805b1"; since
        step 4.7 the denser ``///`` hatch: "dc6f9805b89805b1" -> "2af516f6faf8f240",
        the same value as the figure above, which is what "changes nothing" means)."""
        maps, sig = self._maps()
        empty = {k: np.zeros((8, 8), dtype=int) for k in maps}
        path = plot_t_maps(
            maps, sig_grids=sig, label_grids=empty, arm="raw",
            condition_a="a", condition_b="b", output_path=str(tmp_path / "c.png"),
        )
        assert _pixel_digest(path) == ((468, 1993, 4), "2af516f6faf8f240")

    def test_the_moran_profile(self, tmp_path):
        """Re-pinned (labels): the ticks are now the tap path without the shared
        prefix instead of its last component alone, which read ``attention_tap`` at
        every other tick for real paths. Was ((463, 783, 4), "2bf6cdf35fed80ed")."""
        maps, _ = self._maps()
        path = plot_morans_i_profile(
            {k: i / 10 - 0.1 for i, k in enumerate(maps)},
            output_path=str(tmp_path / "d.png"), arm="raw",
        )
        assert _pixel_digest(path) == ((462, 783, 4), "2a787e0115cb9d6c")


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
        import train.topolm.plotting as module

        assert callable(module.plot_t_maps_3d)
        assert callable(module.plot_cluster_montage_3d)

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
        """Same direction as in 2-D: on each plane through the crosshair (5, 1, 9)
        the hatch covers exactly the cells that FAILED, so the significant block
        around the crosshair is the hole. Cell for cell, since the per-cell
        oracle of :class:`TestTheHatchMarksEveryRejectedCell` is the full proof."""
        volume, significant = _volume_fixture()
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(
                {"t": volume}, sig_grids={"t": significant},
                output_path=str(tmp_path / "t3d.png"),
            ),
        )
        planes = _image_axes(fig)
        assert len(planes) == 3
        expected = [significant[5, :, :], significant[:, 1, :], significant[:, :, 9]]
        for ax, plane in zip(planes, expected):
            assert not plane.all() and plane.any(), "the fixture plane must mix both"
            _assert_hatch_is_exactly(ax, plane)
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


# ---------------------------------------------------------------------
# 3-D figures: what the planes and the montage CONTAIN (added with the drawers)
# ---------------------------------------------------------------------
#
# The step-1 tests above count panels and check one crosshair. These pin the parts
# a count cannot see: the fallback and the tie rule of the crosshair, where the
# crosshair is drawn, the titles, and that montage panel k IS slice k of the label
# grid with one colour per cluster id.


def _planes_shown(fig):
    return [np.asarray(ax.images[0].get_array()) for ax in _image_axes(fig)]


def _t_map_figure(monkeypatch, tmp_path, volume, significant, name="blocks.0.ffn/attn_pre"):
    return _captured_figure(
        monkeypatch,
        lambda: plot_t_maps(
            {name: volume},
            sig_grids=None if significant is None else {name: significant},
            output_path=str(tmp_path / "t3d.png"),
        ),
    )


class TestThreeDimensionalCrosshairAndMontageContent:
    def test_with_nothing_significant_the_crosshair_is_the_largest_abs_t_overall(
        self, monkeypatch, tmp_path
    ):
        """The fixture's +9 at (0, 7, 13) is the largest |t| and is not significant:
        it is the fallback answer when nothing is, with or without a mask."""
        volume, _ = _volume_fixture()
        for significant in (np.zeros(_VOLUME, dtype=bool), None):
            fig = _t_map_figure(monkeypatch, tmp_path, volume, significant)
            shown = _planes_shown(fig)
            plt.close(fig)
            expected = [volume[0, :, :], volume[:, 7, :], volume[:, :, 13]]
            for index, (image, plane) in enumerate(zip(shown, expected)):
                assert np.array_equal(image, plane), f"panel {index}"

    def test_ties_go_to_the_lowest_flat_index(self, monkeypatch, tmp_path):
        volume = np.zeros(_VOLUME)
        significant = np.zeros(_VOLUME, dtype=bool)
        for voxel in ((6, 0, 0), (3, 5, 2), (3, 2, 9)):      # equal |t|, +/- mixed
            volume[voxel] = 5.0 if voxel != (3, 5, 2) else -5.0
            significant[voxel] = True
        fig = _t_map_figure(monkeypatch, tmp_path, volume, significant)
        shown = _planes_shown(fig)
        plt.close(fig)
        # flat order is C order, so (3, 2, 9) comes before (3, 5, 2) and (6, 0, 0)
        assert np.array_equal(shown[0], volume[3, :, :])
        assert np.array_equal(shown[1], volume[:, 2, :])
        assert np.array_equal(shown[2], volume[:, :, 9])

    def test_the_crosshair_lines_sit_on_the_in_plane_coordinates(
        self, monkeypatch, tmp_path
    ):
        """At (5, 1, 9): plane axis 0 has rows = axis 1 and columns = axis 2, so a
        horizontal line at y = 1 and a vertical one at x = 9; plane axis 1 has rows
        = axis 0 (y = 5) and columns = axis 2 (x = 9); plane axis 2 has rows =
        axis 0 (y = 5) and columns = axis 1 (x = 1)."""
        volume, significant = _volume_fixture()
        fig = _t_map_figure(monkeypatch, tmp_path, volume, significant)
        expected = [(1, 9), (5, 9), (5, 1)]
        for axis, (row, col) in zip(_image_axes(fig), expected):
            horizontal = [l for l in axis.lines if len(set(l.get_ydata())) == 1
                          and len(set(l.get_xdata())) > 1 and float(l.get_ydata()[0]) == row]
            vertical = [l for l in axis.lines if len(set(l.get_xdata())) == 1
                        and len(set(l.get_ydata())) > 1 and float(l.get_xdata()[0]) == col]
            assert horizontal and vertical, (row, col)
        plt.close(fig)

    def test_every_panel_title_names_the_tap_and_the_crosshair(self, monkeypatch, tmp_path):
        volume, significant = _volume_fixture()
        fig = _t_map_figure(monkeypatch, tmp_path, volume, significant)
        titles = [ax.get_title() for ax in _image_axes(fig)]
        plt.close(fig)
        assert len(titles) == 3
        for plane_axis, title in enumerate(titles):
            assert "blocks.0.ffn/attn_pre" in title, title
            assert "(5, 1, 9)" in title, title
            assert f"axis {plane_axis} = {(5, 1, 9)[plane_axis]}" in title, title

    def test_a_mask_of_the_wrong_shape_is_ignored_not_fatal(self, monkeypatch, tmp_path):
        volume, _ = _volume_fixture()
        fig = _t_map_figure(monkeypatch, tmp_path, volume, np.ones((2, 2, 2), dtype=bool))
        assert len(_image_axes(fig)) == 3
        assert not any(_hatched(ax) for ax in _image_axes(fig))
        plt.close(fig)

    def test_the_3d_drawer_refuses_a_2d_map_naming_the_tap(self):
        from train.topolm.plotting import plot_t_maps_3d

        with pytest.raises(ValueError, match="flat-tap") as caught:
            plot_t_maps_3d({"flat-tap": np.zeros((4, 4))})
        assert "3-D" in str(caught.value)

    def test_a_4d_map_is_refused_naming_both_supported_ranks(self):
        with pytest.raises(ValueError) as caught:
            plot_t_maps({"tap-4d": np.zeros((2, 2, 2, 2))})
        assert "2-D" in str(caught.value) and "3-D" in str(caught.value)

    def test_montage_panel_k_is_slice_k_and_an_id_keeps_its_colour(
        self, monkeypatch, tmp_path
    ):
        """Non-contiguous ids (3, 7, 12) over a (4, 5, 6) grid, with id 7 absent from
        slice 0: each panel must show slice k, coded by the id's RANK among the
        shown ids (3 -> 1, 7 -> 2, 12 -> 3), so the colour of 7 is the same wherever
        it appears."""
        labels = np.zeros((4, 5, 6), dtype=int)
        labels[0, 0:2, 0:2] = 3
        labels[1:4, 2:4, 2:4] = 7
        labels[2:4, 4, 5] = 12
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_cluster_overlay(labels, output_path=str(tmp_path / "m.png")),
        )
        shown = _planes_shown(fig)
        clims = {ax.images[0].get_clim() for ax in _image_axes(fig)}
        plt.close(fig)
        rank = {0: 0, 3: 1, 7: 2, 12: 3}
        expected = np.vectorize(rank.get)(labels)
        assert len(shown) == 4
        for k, image in enumerate(shown):
            assert np.array_equal(image, expected[k]), f"panel {k} is not slice {k}"
        assert clims == {(0, 3)}, "one colour scale for every panel"

    def test_a_montage_of_an_empty_grid_still_writes(self, tmp_path):
        path = plot_cluster_overlay(
            np.zeros((3, 4, 4), dtype=int), output_path=str(tmp_path / "m.png")
        )
        assert path is not None and (tmp_path / "m.png").stat().st_size > 0


# ---------------------------------------------------------------------
# 3-D panel titles with REAL tap paths (completion fix step-4.1)
# ---------------------------------------------------------------------
# The step-4 checks used short synthetic tap names, and a real run's paths
# (``causal_language_model/topolm/block_0/attention_tap``) are wider than the
# plane they sit over, so neighbouring titles overprinted each other. The oracle
# is geometric and independent of how the title is built: the rendered extents
# of the title artists, not a pixel hash and not a character count.

_REAL_PREFIX = "causal_language_model/topolm/"


def _real_tap_names(n_blocks):
    return [
        f"{_REAL_PREFIX}block_{b}/{tap}"
        for b in range(n_blocks)
        for tap in ("attention_tap", "mlp_tap")
    ]


def _title_boxes_and_obstacles(fig):
    """``(title_boxes, other_boxes)`` in display pixels, one title per image axes.

    ``other_boxes`` are what a title must not overprint: every other axes frame and
    the figure-level texts (headline and subtitle).
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    image_axes = _image_axes(fig)
    titles = [(ax, ax.title.get_window_extent(renderer)) for ax in image_axes]
    texts = [t.get_window_extent(renderer) for t in fig.texts if t.get_text()]
    return titles, [ax.get_window_extent(renderer) for ax in image_axes], texts


def _overlap(a, b, tolerance=0.5):
    """Whether two display boxes share more than ``tolerance`` pixels in BOTH axes."""
    return (
        min(a.x1, b.x1) - max(a.x0, b.x0) > tolerance
        and min(a.y1, b.y1) - max(a.y0, b.y0) > tolerance
    )


class TestThreeDimensionalTitlesAreReadableForRealTapPaths:
    @pytest.mark.parametrize("n_blocks", [1, 2, 12], ids=["2taps", "4taps", "24taps"])
    def test_no_title_overprints_another_title_a_panel_or_the_headline(
        self, monkeypatch, tmp_path, n_blocks
    ):
        names = _real_tap_names(n_blocks)
        rng = np.random.default_rng(3)
        maps = {n: rng.normal(size=(7, 8, 7)) for n in names}
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(
                maps, condition_a="a", condition_b="b",
                output_path=str(tmp_path / "real.png"),
            ),
        )
        titles, frames, texts = _title_boxes_and_obstacles(fig)
        bounds = fig.bbox
        plt.close(fig)
        assert len(titles) == 3 * len(names)
        for index, (axis, box) in enumerate(titles):
            assert box.x0 >= bounds.x0 and box.x1 <= bounds.x1, f"title {index} clipped"
            assert box.y0 >= bounds.y0 and box.y1 <= bounds.y1, f"title {index} clipped"
            for other_index, (_, other) in enumerate(titles):
                if other_index > index:
                    assert not _overlap(box, other), (
                        f"titles {index} and {other_index} intersect: {box} {other}"
                    )
            for frame_index, frame in enumerate(frames):
                if frame_index != index:
                    assert not _overlap(box, frame), (
                        f"title {index} overprints panel {frame_index}"
                    )
            for text in texts:
                assert not _overlap(box, text), f"title {index} overprints the headline"

    def test_a_single_tap_with_its_full_real_path_is_also_readable(
        self, monkeypatch, tmp_path
    ):
        name = f"{_REAL_PREFIX}block_0/attention_tap"
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(
                {name: np.random.default_rng(0).normal(size=(7, 8, 7))},
                output_path=str(tmp_path / "one.png"),
            ),
        )
        titles, _, texts = _title_boxes_and_obstacles(fig)
        plt.close(fig)
        boxes = [box for _, box in titles]
        for i, box in enumerate(boxes):
            assert box.x0 >= fig.bbox.x0 and box.x1 <= fig.bbox.x1
            assert not any(_overlap(box, t) for t in texts)
            assert not any(_overlap(box, o) for j, o in enumerate(boxes) if j != i)

    def test_the_full_path_is_recoverable_from_the_figure(self, monkeypatch, tmp_path):
        """Each title carries the tail that tells the taps apart, and the headline
        says what was dropped, so prefix + tail is the original path."""
        names = _real_tap_names(2)
        maps = {n: np.random.default_rng(i).normal(size=(7, 8, 7)) for i, n in enumerate(names)}
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(maps, output_path=str(tmp_path / "rec.png")),
        )
        subtitle = " ".join(t.get_text() for t in fig.texts)
        titles = [ax.get_title() for ax in _image_axes(fig)]
        plt.close(fig)
        assert _REAL_PREFIX in subtitle
        for tap, name in enumerate(names):
            tail = name[len(_REAL_PREFIX):]
            for title in titles[3 * tap:3 * tap + 3]:
                assert "".join(title.split("\n")[:-1]) == tail, title


class TestTapLabelContract:
    def test_a_short_unique_name_is_returned_unchanged(self):
        from train.topolm.plotting import _panel_tap_labels

        labels, prefix, fontsize = _panel_tap_labels(["blocks.0.ffn/attn_pre"])
        assert labels == {"blocks.0.ffn/attn_pre": "blocks.0.ffn/attn_pre"}
        assert prefix == "" and fontsize == 5.0

    def test_shared_leading_components_move_to_the_prefix(self):
        from train.topolm.plotting import _panel_tap_labels

        names = _real_tap_names(2)
        labels, prefix, _ = _panel_tap_labels(names)
        assert prefix == _REAL_PREFIX
        for name, label in labels.items():
            assert prefix + label == name

    def test_at_least_the_last_component_survives(self):
        from train.topolm.plotting import _panel_tap_labels

        labels, prefix, _ = _panel_tap_labels(["a/b/c", "a/b/c"])
        assert labels["a/b/c"] == "c" and prefix == "a/b/"

    def test_wrapping_loses_nothing_and_keeps_lines_short(self):
        from train.topolm.plotting import _TITLE_MAX_CHARS, _panel_tap_labels

        name = f"{_REAL_PREFIX}block_0/attention_tap"
        labels, prefix, fontsize = _panel_tap_labels([name])
        assert prefix == ""
        assert "".join(labels[name].split("\n")) == name
        assert max(len(line) for line in labels[name].split("\n")) <= _TITLE_MAX_CHARS
        assert fontsize == 5.0

    def test_an_unbreakable_long_component_shrinks_the_font(self):
        from train.topolm.plotting import _panel_tap_labels

        _, _, fontsize = _panel_tap_labels(["x" * 60])
        assert 3.5 <= fontsize < 5.0


class TestTheLiveNoteDoesNotCrowdTheTitle:
    """Step 6: the live callback passes a ``note`` (step and stimulus set) that is
    appended to the line under the title. On a one-row 2-D figure the title and
    that line overlapped, so the figure grows when a note is given (and only then:
    the no-note path is pixel-pinned)."""

    def _title_and_subtitle_boxes(self, **kwargs):
        fig, _ = _capture_axes(_t_maps(n_taps=3), arm="raw", _fn="plot_t_maps", **kwargs)
        renderer = fig.canvas.get_renderer()
        title = fig._suptitle.get_window_extent(renderer)
        subtitle = next(t for t in fig.texts if t is not fig._suptitle).get_window_extent(renderer)
        return fig, title, subtitle

    def test_a_noted_one_row_figure_has_a_title_clear_of_its_subtitle(self):
        fig, title, subtitle = self._title_and_subtitle_boxes(
            condition_a="a", condition_b="b", note="step 24 | smoke stimuli, n=5/5 per condition"
        )
        assert not title.overlaps(subtitle)
        assert "step 24" in [t for t in fig.texts if t is not fig._suptitle][0].get_text()

    def test_the_note_is_absent_without_one(self):
        fig, _, _ = self._title_and_subtitle_boxes(condition_a="a", condition_b="b")
        assert "step" not in [t for t in fig.texts if t is not fig._suptitle][0].get_text()


# ---------------------------------------------------------------------
# Step 4.2: the 2-D figures with the names and the note a real run gives them
# ---------------------------------------------------------------------
# The live figure of a real tiny run (6 taps, ``causal_language_model/topolm/...``
# paths, a step-and-stimulus note) had the headline on top of the line under it,
# the colour bar over the last panel and its title, and tap titles touching their
# neighbours. Like the 3-D checks above, the oracle is geometric: the rendered
# extents of the artists, never a pixel hash.

_LIVE_NOTE = "step 24 | smoke stimuli, n=5/5 per condition"
_REAL_NAME_CASES = [
    pytest.param(1, id="2taps"),
    pytest.param(3, id="6taps"),
    pytest.param(12, id="24taps"),
]


def _real_maps(n_blocks, size=8, seed=5):
    rng = np.random.default_rng(seed)
    return {name: rng.normal(size=(size, size)) * 2 for name in _real_tap_names(n_blocks)}


def _t_map_geometry(fig):
    """The rendered extents of a 2-D t-map figure, in display pixels.

    Returns ``(titles, frames, bar, texts)``: the title of each image axes, each
    image axes' frame, the colour bar's axes INCLUDING its tick labels and label
    (``get_tightbbox``), and the figure-level texts (the headline and the line
    under it).
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panels = _image_axes(fig)
    bars = [ax for ax in fig.axes if not ax.images and ax.get_visible()]
    assert len(bars) == 1, f"expected exactly one colour bar axes, found {len(bars)}"
    return (
        [ax.title.get_window_extent(renderer) for ax in panels],
        [ax.get_window_extent(renderer) for ax in panels],
        bars[0].get_tightbbox(renderer),
        [t.get_window_extent(renderer) for t in fig.texts if t.get_text()],
    )


def _inside(box, bounds):
    return (box.x0 >= bounds.x0 - 0.5 and box.x1 <= bounds.x1 + 0.5
            and box.y0 >= bounds.y0 - 0.5 and box.y1 <= bounds.y1 + 0.5)


class TestTwoDimensionalTitlesAreReadableForRealTapPaths:
    @pytest.mark.parametrize("note", [None, _LIVE_NOTE], ids=["no_note", "live_note"])
    @pytest.mark.parametrize("n_blocks", _REAL_NAME_CASES)
    def test_nothing_overprints_anything_and_nothing_leaves_the_figure(
        self, monkeypatch, tmp_path, n_blocks, note
    ):
        maps = _real_maps(n_blocks)
        sig = {k: np.abs(v) > 3 for k, v in maps.items()}
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(
                maps, sig_grids=sig, condition_a="a", condition_b="b", note=note,
                output_path=str(tmp_path / "real.png"),
            ),
        )
        titles, frames, bar, texts = _t_map_geometry(fig)
        bounds = fig.bbox
        plt.close(fig)
        assert len(titles) == len(maps) and len(frames) == len(maps)
        assert len(texts) == 2, "expected the headline and the line under it"

        # Nothing leaves the figure.
        for name, box in [(f"title {i}", b) for i, b in enumerate(titles)] + [
            (f"panel {i}", b) for i, b in enumerate(frames)
        ] + [("colour bar", bar)] + [(f"text {i}", b) for i, b in enumerate(texts)]:
            assert _inside(box, bounds), f"{name} leaves the figure: {box} in {bounds}"

        # No title overprints another title, any panel, the colour bar or a headline.
        for i, box in enumerate(titles):
            for j in range(i + 1, len(titles)):
                assert not _overlap(box, titles[j]), f"titles {i} and {j} intersect"
            for j, frame in enumerate(frames):
                if j != i:
                    assert not _overlap(box, frame), f"title {i} overprints panel {j}"
            assert not _overlap(box, bar), f"title {i} overprints the colour bar"
            for k, text in enumerate(texts):
                assert not _overlap(box, text), f"title {i} overprints text {k}"

        # The headline and the line under it are apart, and clear of every panel
        # and the colour bar.
        assert not _overlap(texts[0], texts[1]), "the headline overprints its subtitle"
        for k, text in enumerate(texts):
            assert not _overlap(text, bar), f"text {k} overprints the colour bar"
            for j, frame in enumerate(frames):
                assert not _overlap(text, frame), f"text {k} overprints panel {j}"

        # The colour bar does not touch any panel (not even its frame).
        for j, frame in enumerate(frames):
            assert not _overlap(bar, frame, tolerance=-0.5), (
                f"the colour bar touches panel {j}: {bar} {frame}"
            )

    def test_a_single_tap_with_its_full_real_path_is_also_readable(
        self, monkeypatch, tmp_path
    ):
        name = f"{_REAL_PREFIX}block_0/attention_tap"
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(
                {name: np.random.default_rng(0).normal(size=(8, 8))}, note=_LIVE_NOTE,
                output_path=str(tmp_path / "one.png"),
            ),
        )
        titles, frames, bar, texts = _t_map_geometry(fig)
        bounds = fig.bbox
        plt.close(fig)
        assert all(_inside(b, bounds) for b in titles + frames + texts + [bar])
        assert not any(_overlap(titles[0], t) for t in texts + [bar])
        assert not _overlap(texts[0], texts[1])
        assert not _overlap(bar, frames[0], tolerance=-0.5)

    def test_the_short_name_case_is_clean_too(self, monkeypatch, tmp_path):
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(_t_maps(n_taps=4), condition_a="a", condition_b="b",
                                output_path=str(tmp_path / "short.png")),
        )
        titles, frames, bar, texts = _t_map_geometry(fig)
        bounds = fig.bbox
        plt.close(fig)
        assert all(_inside(b, bounds) for b in titles + frames + texts + [bar])
        assert not any(_overlap(titles[i], titles[i + 1]) for i in range(3))
        assert not any(_overlap(t, bar) for t in titles + texts)
        assert not _overlap(texts[0], texts[1])
        assert not any(_overlap(bar, f, tolerance=-0.5) for f in frames)

    def test_the_titles_drop_the_shared_prefix_and_the_subtitle_names_it(
        self, monkeypatch, tmp_path
    ):
        names = _real_tap_names(3)
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(_real_maps(3), output_path=str(tmp_path / "p.png")),
        )
        subtitle = " ".join(t.get_text() for t in fig.texts)
        titles = [ax.get_title() for ax in _image_axes(fig)]
        plt.close(fig)
        assert _REAL_PREFIX in " ".join(subtitle.split("\n"))
        for name, title in zip(names, titles):
            assert "".join(title.split("\n")) == name[len(_REAL_PREFIX):]


class TestTwoDimensionalContentDoesNotDependOnLayout:
    """What the 2-D figure SHOWS, asserted on the artists and the inputs, so the
    layout re-pin in step 4.2 cannot hide a change of content: the image data, the
    colour limits, the hatch geometry and the cluster outlines are all in data
    coordinates and independent of where the figure puts its axes."""

    # SHA-256 (first 16 hex digits) of the hatch paths' vertices and codes for the
    # five fixture panels. Same environment pin as the pixel digests. RE-PINNED ON
    # PURPOSE in step 4.6 (D-021): "08d1c8a366184f84" hashed the ``contourf`` polygons,
    # which covered only the cells whose four neighbours were all rejected (it was
    # unchanged by the step-4.2 layout change, which is what it was for). The new
    # value hashes one unit box per rejected cell, in row-major order, over the same
    # five panels; that those boxes are EXACTLY the rejected cells is proved
    # independently of this hash in TestTheHatchMarksEveryRejectedCell.
    # RE-PINNED ON PURPOSE again in step 4.7 (D-026): the digest also covers each
    # collection's hatch PATTERN now. It covered only the geometry before, so a
    # change of the pattern was invisible to it ("3a384d867d13f736" is the same
    # boxes without the pattern; with the old ``//`` it is "ecb45af41c88cf6b").
    # The boxes are unchanged; the pattern is the denser ``///``.
    HATCH_GEOMETRY = "4d27b3e3fa1594d4"

    @staticmethod
    def _panels(fig):
        return [ax for ax in fig.axes if ax.images]

    def _check_content(self, fig, maps, labels=None):
        limit = _shared_limit(maps, 99.0)
        panels = self._panels(fig)
        assert len(panels) == len(maps)
        for axis, values in zip(panels, maps.values()):
            image = axis.images[0]
            np.testing.assert_array_equal(np.asarray(image.get_array()), values)
            assert image.get_clim() == (-limit, limit)
        bar = next(ax for ax in fig.axes if not ax.images and ax.get_visible())
        assert tuple(bar.get_ylim()) == (-limit, limit), "the colour bar has another scale"
        if labels is not None:
            for axis, grid in zip(panels, labels.values()):
                assert _drawn_pieces(axis) == _expected_pieces(grid)

    @pytest.mark.parametrize("note", [None, _LIVE_NOTE], ids=["no_note", "note"])
    def test_short_names_the_pinned_fixture(self, monkeypatch, tmp_path, note):
        maps, sig = TestTwoDimensionalUnchangedPixels._maps()
        labels = {k: _boundary_fixtures()["touching"] for k in maps}
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(maps, sig_grids=sig, label_grids=labels, note=note,
                                output_path=str(tmp_path / "c.png")),
        )
        self._check_content(fig, maps, labels)
        plt.close(fig)

    @pytest.mark.parametrize("n_blocks", _REAL_NAME_CASES)
    def test_real_names_with_a_note(self, monkeypatch, tmp_path, n_blocks):
        maps = _real_maps(n_blocks)
        labels = {k: _boundary_fixtures()["ring"] for k in maps}
        sig = {k: np.abs(v) > 3 for k, v in maps.items()}
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(maps, sig_grids=sig, label_grids=labels,
                                note=_LIVE_NOTE, output_path=str(tmp_path / "r.png")),
        )
        self._check_content(fig, maps, labels)
        plt.close(fig)

    def test_the_hatch_geometry_is_one_box_per_rejected_cell(self):
        import hashlib

        maps, sig = TestTwoDimensionalUnchangedPixels._maps()
        fig, axes = _capture_axes(maps, sig_grids=sig, _fn="plot_t_maps")
        digest = hashlib.sha256()
        for axis in [a for row in np.atleast_2d(axes) for a in row]:
            for collection in axis.collections:
                for path in collection.get_paths():
                    digest.update(np.round(path.vertices, 6).tobytes())
                    if path.codes is not None:
                        digest.update(path.codes.tobytes())
                digest.update(repr(collection.get_hatch()).encode())
        plt.close(fig)
        assert digest.hexdigest()[:16] == self.HATCH_GEOMETRY


class TestTheMoransProfileIsReadableForRealTapPaths:
    @pytest.mark.parametrize("n_blocks", _REAL_NAME_CASES)
    def test_every_tick_names_its_tap_and_nothing_overprints(
        self, monkeypatch, tmp_path, n_blocks
    ):
        from train.topolm.plotting import plot_morans_i_profile as draw

        names = _real_tap_names(n_blocks)
        fig = _captured_figure(
            monkeypatch,
            lambda: draw({n: 0.05 * i for i, n in enumerate(names)}, arm="raw",
                         note=_LIVE_NOTE, output_path=str(tmp_path / "m.png")),
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axis = fig.axes[0]
        ticks = [t for t in axis.get_xticklabels() if t.get_text()]
        boxes = [t.get_window_extent(renderer) for t in ticks]
        other = [axis.title.get_window_extent(renderer),
                 axis.xaxis.label.get_window_extent(renderer)]
        bounds = fig.bbox
        shown = [t.get_text() for t in ticks]
        subtitle = " ".join(axis.xaxis.label.get_text().split("\n"))
        plt.close(fig)

        assert len(set(shown)) == len(shown), f"two ticks read the same: {shown}"
        assert _REAL_PREFIX in subtitle
        # The label names the prefix that was dropped (a one-block figure drops the
        # block as well), so prefix + tick is the tap's full path.
        prefix = subtitle.split("shared prefix ")[1]
        for text in shown:
            assert f"{prefix}{text}" in names
        for i, box in enumerate(boxes + other):
            assert _inside(box, bounds), f"element {i} leaves the figure"
        for i, box in enumerate(boxes):
            assert not any(_overlap(box, o) for o in other), f"tick {i} overprints a label"
            assert not any(_overlap(box, boxes[j]) for j in range(i + 1, len(boxes)))


class TestThreeDimensionalHeadlineFitsWithANote:
    """Step 4.2 found the 3-D line under the headline 12 px wider than a one-tap
    figure once it carried the live note; it is wrapped to the figure now."""

    @pytest.mark.parametrize("n_taps", [1, 2, 6, 24], ids=lambda n: f"{n}taps")
    def test_the_headline_texts_stay_inside_and_clear_of_the_titles(
        self, monkeypatch, tmp_path, n_taps
    ):
        names = _real_tap_names(12)[:n_taps]
        maps = {n: np.random.default_rng(3).normal(size=(7, 8, 7)) for n in names}
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(maps, condition_a="a", condition_b="b",
                                note=_LIVE_NOTE, output_path=str(tmp_path / "n.png")),
        )
        titles, _, texts = _title_boxes_and_obstacles(fig)
        bounds = fig.bbox
        plt.close(fig)
        assert all(_inside(t, bounds) for t in texts)
        assert not _overlap(texts[0], texts[1])
        for _, box in titles:
            assert not any(_overlap(box, t) for t in texts)


# ---------------------------------------------------------------------
# Completion fix step 4.3: what a person watching a real run saw
# ---------------------------------------------------------------------
#
# Three defects read off the real-run figures: the curve's val_loss panel was
# drawn at the draw step (one cadence late, a false plateau), the 2-D cluster title
# read "mlp_tap (raw): 1 clusters" (block dropped, wrong plural), and the 3-D
# crosshair was a thin green dash that vanished on saturated cells.


def _curve_rows(draw_steps, val_losses, val_steps):
    """Live rows as the callback writes them: a ``val_loss`` that repeats between
    epochs, with ``val_loss_step`` naming where it was measured."""
    return [
        {
            "step": step, "val_loss": loss, "val_loss_step": measured,
            "mean_morans_i": 0.1 * index, "total_clusters_a": index,
            "total_clusters_b": index, "mean_num_significant_units": 3.0,
            "per_tap": {"t": {"morans_i": 0.1 * index}},
        }
        for index, (step, loss, measured) in enumerate(
            zip(draw_steps, val_losses, val_steps)
        )
    ]


def _val_loss_line(fig):
    axis = next(ax for ax in fig.axes if ax.get_ylabel().startswith("val_loss"))
    line = axis.lines[0]
    return list(map(float, line.get_xdata())), list(map(float, line.get_ydata()))


class TestTheCurveDrawsValidationAtItsOwnStep:
    """RED on the previous drawer: it plotted ``val_loss`` against ``step``."""

    def test_each_validation_is_drawn_once_at_the_step_it_was_measured(
        self, monkeypatch, tmp_path
    ):
        rows = _curve_rows(
            draw_steps=[0, 50, 100, 150, 200, 250, 300],
            val_losses=[None, None, None, 7.9, 7.9, 7.7, 7.7],
            val_steps=[None, None, None, 100, 100, 200, 200],
        )
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_topography_curve(rows, output_path=str(tmp_path / "c.png")),
        )
        steps, values = _val_loss_line(fig)
        plt.close(fig)
        assert steps == [100.0, 200.0], "drawn at the draw step, or a repeat kept"
        assert values == [7.9, 7.7]

    def test_a_row_without_a_measured_step_falls_back_to_its_draw_step(
        self, monkeypatch, tmp_path
    ):
        """An older ``metrics.jsonl`` has no ``val_loss_step``."""
        rows = _curve_rows([0, 10, 20], [1.0, 0.5, 0.25], [None, None, None])
        for row in rows:
            del row["val_loss_step"]
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_topography_curve(rows, output_path=str(tmp_path / "c.png")),
        )
        steps, values = _val_loss_line(fig)
        plt.close(fig)
        assert (steps, values) == ([0.0, 10.0, 20.0], [1.0, 0.5, 0.25])

    def test_a_curve_with_no_validation_has_no_val_loss_panel(self, monkeypatch, tmp_path):
        rows = _curve_rows([0, 50], [None, None], [None, None])
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_topography_curve(rows, output_path=str(tmp_path / "c.png")),
        )
        labels = [ax.get_ylabel() for ax in fig.axes]
        plt.close(fig)
        assert not any(label.startswith("val_loss") for label in labels), labels


def _cluster_report_and_arrays(names, populated, size=8):
    """The smallest ``(report, arrays)`` pair ``plot_topography`` accepts, with one
    cluster in the tap ``populated``."""
    rng = np.random.default_rng(1)
    labels = {n: np.zeros((size, size), dtype=int) for n in names}
    labels[populated][1:5, 1:4] = 1
    report = {
        "contrast": ["a", "b"],
        "arms": {"raw": {"per_tap": {
            n: {"morans_i": {"standard": 0.3}} for n in names}}},
    }
    arrays = {"arms": {"raw": {
        "t_maps": {n: rng.normal(size=(size, size)) * 2 for n in names},
        "sig_grids": {n: np.zeros((size, size), dtype=bool) for n in names},
        "label_grids": labels,
    }}}
    return report, arrays


def _cluster_figure_title(monkeypatch, tmp_path, names, populated):
    from train.topolm import common

    report, arrays = _cluster_report_and_arrays(names, populated)
    holder = {}
    original = Figure.savefig

    def recording(self, path, *args, **kwargs):
        if "clusters_" in str(path):
            holder["fig"] = self
        return original(self, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", recording)
    common.plot_topography(report, arrays, str(tmp_path))
    return holder["fig"]


class TestTheClusterFigureNamesItsTapAndCountsCorrectly:
    def test_the_title_keeps_the_block_and_drops_only_the_shared_prefix(
        self, monkeypatch, tmp_path
    ):
        names = _real_tap_names(4)
        fig = _cluster_figure_title(monkeypatch, tmp_path, names, names[-1])
        title = fig.axes[0].get_title()
        plt.close(fig)
        # Re-pinned on purpose (step 4.5, D2): the figure shows condition A's
        # clusters only and now says so; it was "block_3/mlp_tap (raw): 1 cluster".
        assert title == "block_3/mlp_tap (raw), condition a: 1 cluster", title

    def test_a_single_tap_keeps_its_whole_path_and_stays_inside_the_figure(
        self, monkeypatch, tmp_path
    ):
        names = _real_tap_names(4)[-1:]
        fig = _cluster_figure_title(monkeypatch, tmp_path, names, names[0])
        title = fig.axes[0].title
        text = title.get_text()
        fig.canvas.draw()
        box = title.get_window_extent(fig.canvas.get_renderer())
        bounds = fig.bbox
        plt.close(fig)
        assert "".join(text.split("\n")).startswith(names[0]), text
        assert _inside(box, bounds), "the wrapped title leaves the figure"

    @pytest.mark.parametrize("count,expected", [(1, "1 cluster"), (2, "2 clusters")])
    def test_the_plural_follows_the_count_in_2d_and_3d(
        self, monkeypatch, tmp_path, count, expected
    ):
        flat = np.zeros((6, 6), dtype=int)
        for index in range(count):
            flat[index * 3, 0:2] = index + 1
        solid = np.stack([flat, flat])
        for grid in (flat, solid):
            fig = _captured_figure(
                monkeypatch,
                lambda: plot_cluster_overlay(grid, output_path=str(tmp_path / "x.png")),
            )
            texts = [t.get_text() for t in fig.findobj(matplotlib.text.Text)]
            plt.close(fig)
            assert any(t == f"clusters: {expected}" for t in texts), (grid.ndim, texts)


class TestTheCrosshairIsVisibleOnSaturatedCells:
    """RED on the previous drawer: a bare 0.5 pt green dash, no outline."""

    def test_every_crosshair_line_has_a_wider_white_stroke_under_it(
        self, monkeypatch, tmp_path
    ):
        import matplotlib.patheffects as path_effects

        volume, significant = _volume_fixture()
        fig = _t_map_figure(monkeypatch, tmp_path, volume, significant)
        checked = 0
        for axis in _image_axes(fig):
            for line in axis.lines:
                effects = line.get_path_effects()
                strokes = [e for e in effects if isinstance(e, path_effects.Stroke)]
                assert strokes, "a crosshair line has no outline"
                halo = strokes[0]
                assert halo._gc["foreground"] == "white"
                assert halo._gc["linewidth"] > line.get_linewidth()
                assert any(isinstance(e, path_effects.Normal) for e in effects), (
                    "the green line itself is no longer drawn"
                )
                checked += 1
        plt.close(fig)
        assert checked == 6, "three planes with two lines each"


# =====================================================================
# Step 4.4 (plan-2026-10-23T190557-972a9ac5, decision D-018): what the outlines and
# the hatch MEAN, both conditions' outlines, and a legend on the live curve.
# Written RED-first against the 02af58996 drawers.
# =====================================================================

#: Outline colour of condition B; condition A keeps the black it always had.
_A_COLOUR, _B_COLOUR = "black", "magenta"


def _pieces_in_colour(axis, colour):
    """The unit pieces of the boundary lines of ONE colour on ``axis`` (the
    multiset the step-3 oracle compares), so A's and B's outlines are told apart
    by how they are drawn and not by trusting the code that drew them."""
    found = Counter()
    for line in axis.get_lines():
        if matplotlib.colors.same_color(line.get_color(), colour):
            xs, ys = line.get_xdata(), line.get_ydata()
            for (xa, ya), (xb, yb) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
                found.update(_pieces(xa, ya, xb, yb))
    return found


def _two_condition_labels():
    """A's clusters and B's, on cells A and B never share (a cell has one sign)."""
    a = np.zeros((8, 8), dtype=int)
    a[0:3, 0:3] = 1                # touches the corner
    a[6, 6] = 2
    b = np.zeros((8, 8), dtype=int)
    b[4:7, 1:4] = 1
    b[2:4, 5] = 2                  # a vertical pair
    b[4, 0] = 3
    assert not np.any((a > 0) & (b > 0))
    return a, b


def _subtitle_of(fig):
    return " ".join(
        t.get_text().replace("\n", " ") for t in fig.texts if t is not fig._suptitle
    )


class TestTheTMapOutlinesCoverBothConditions:
    def test_both_conditions_outlines_equal_the_brute_force_cell_edges(
        self, monkeypatch, tmp_path
    ):
        a, b = _two_condition_labels()
        fig, axes = _capture_axes(
            {"t": np.zeros((8, 8))}, label_grids={"t": a}, label_grids_b={"t": b},
            condition_a="a", condition_b="b",
            output_path=str(tmp_path / "ab.png"), _fn="plot_t_maps",
        )
        axis = np.atleast_2d(axes)[0, 0]
        drawn_a = _pieces_in_colour(axis, _A_COLOUR)
        drawn_b = _pieces_in_colour(axis, _B_COLOUR)
        plt.close(fig)
        assert drawn_a == _expected_pieces(a), "condition A's outline moved"
        assert drawn_b == _expected_pieces(b), {
            "missing": sorted((_expected_pieces(b) - drawn_b).elements())[:6],
            "unexpected": sorted((drawn_b - _expected_pieces(b)).elements())[:6],
        }

    def test_the_two_conditions_are_drawn_in_different_styles(self, tmp_path):
        a, b = _two_condition_labels()
        fig, axes = _capture_axes(
            {"t": np.zeros((8, 8))}, label_grids={"t": a}, label_grids_b={"t": b},
            output_path=str(tmp_path / "s.png"), _fn="plot_t_maps",
        )
        lines = np.atleast_2d(axes)[0, 0].get_lines()
        plt.close(fig)
        styles = {(l.get_color(), l.get_linestyle()) for l in lines}
        assert len(styles) == 2, styles
        assert ("black", "-") in styles

    def test_without_condition_b_grids_nothing_new_is_drawn(self, tmp_path):
        a, _ = _two_condition_labels()
        fig, axes = _capture_axes(
            {"t": np.zeros((8, 8))}, label_grids={"t": a},
            output_path=str(tmp_path / "a.png"), _fn="plot_t_maps",
        )
        axis = np.atleast_2d(axes)[0, 0]
        drawn = _drawn_pieces(axis)
        plt.close(fig)
        assert drawn == _expected_pieces(a)
        assert not _pieces_in_colour(axis, _B_COLOUR)

    def test_the_3d_planes_outline_both_conditions_at_the_crosshair(
        self, monkeypatch, tmp_path
    ):
        from train.topolm.plotting import _crosshair_voxel

        volume, significant = _volume_fixture()
        a = np.zeros(volume.shape, dtype=int)
        b = np.zeros(volume.shape, dtype=int)
        a[2:4, 2:5, 3:6] = 1
        b[4:7, 0:3, 8:11] = 1                      # around the crosshair voxel (5, 1, 9)
        name = "blocks.0.ffn/attn_pre"
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(
                {name: volume}, sig_grids={name: significant},
                label_grids={name: a}, label_grids_b={name: b},
                condition_a="a", condition_b="b",
                output_path=str(tmp_path / "t3d.png"),
            ),
        )
        crosshair = _crosshair_voxel(volume, significant)
        axes = _image_axes(fig)
        checked = 0
        for plane_axis, axis in enumerate(axes):
            index = [slice(None)] * 3
            index[plane_axis] = crosshair[plane_axis]
            index = tuple(index)
            assert _pieces_in_colour(axis, _A_COLOUR) == _expected_pieces(a[index])
            drawn_b = _pieces_in_colour(axis, _B_COLOUR)
            assert drawn_b == _expected_pieces(b[index])
            checked += len(drawn_b)
        plt.close(fig)
        assert checked > 0, "the fixture put no B cluster on the crosshair planes"


class TestTheTMapSaysWhatItShows:
    def _subtitle(self, monkeypatch, tmp_path, **kwargs):
        a, b = _two_condition_labels()
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(
                _t_maps(n_taps=2),
                sig_grids={k: np.abs(v) > 1.5 for k, v in _t_maps(n_taps=2).items()},
                output_path=str(tmp_path / "sub.png"),
                **kwargs,
            ),
        )
        text = _subtitle_of(fig)
        plt.close(fig)
        return text

    def test_the_subtitle_explains_the_hatch_and_both_outlines(self, monkeypatch, tmp_path):
        a, b = _two_condition_labels()
        labels = {k: a for k in _t_maps(n_taps=2)}
        labels_b = {k: b for k in _t_maps(n_taps=2)}
        text = self._subtitle(
            monkeypatch, tmp_path, condition_a="a", condition_b="b",
            label_grids=labels, label_grids_b=labels_b,
        )
        assert "positive = a, negative = b" in text
        assert "hatched = not significant after BH-FDR" in text
        assert "outline = cluster of significant units" in text
        assert "solid black = a" in text and "dashed magenta = b" in text

    def test_when_only_condition_a_is_given_it_says_b_is_not_drawn(
        self, monkeypatch, tmp_path
    ):
        a, _ = _two_condition_labels()
        text = self._subtitle(
            monkeypatch, tmp_path, condition_a="a", condition_b="b",
            label_grids={k: a for k in _t_maps(n_taps=2)},
        )
        assert "solid black = a" in text
        assert "b clusters not drawn" in text
        assert "dashed" not in text

    def test_no_clusters_means_no_outline_sentence(self, monkeypatch, tmp_path):
        empty = {k: np.zeros((8, 8), dtype=int) for k in _t_maps(n_taps=2)}
        text = self._subtitle(
            monkeypatch, tmp_path, condition_a="a", condition_b="b",
            label_grids=empty, label_grids_b=empty,
        )
        assert "outline" not in text and "hatched = not significant" in text

    def test_no_hatch_and_no_clusters_is_not_described(self, monkeypatch, tmp_path):
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(_t_maps(n_taps=2), condition_a="a", condition_b="b",
                                output_path=str(tmp_path / "plain.png")),
        )
        text = _subtitle_of(fig)
        plt.close(fig)
        assert text.strip() == "positive = a, negative = b"

    @pytest.mark.parametrize("n_blocks", _REAL_NAME_CASES)
    def test_the_longer_subtitle_still_clears_everything_in_2d(
        self, monkeypatch, tmp_path, n_blocks
    ):
        maps = _real_maps(n_blocks)
        sig = {k: np.abs(v) > 3 for k, v in maps.items()}
        a, b = _two_condition_labels()
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(
                maps, sig_grids=sig, condition_a="a", condition_b="b",
                label_grids={k: a for k in maps}, label_grids_b={k: b for k in maps},
                note=_LIVE_NOTE, output_path=str(tmp_path / "long.png"),
            ),
        )
        titles, frames, bar, texts = _t_map_geometry(fig)
        bounds = fig.bbox
        plt.close(fig)
        assert len(texts) == 2
        assert all(_inside(box, bounds) for box in titles + frames + texts + [bar])
        assert not _overlap(texts[0], texts[1])
        for text in texts:
            assert not _overlap(text, bar)
            assert not any(_overlap(text, frame) for frame in frames)
        for i, title in enumerate(titles):
            assert not any(_overlap(title, text) for text in texts)
            assert not _overlap(title, bar)
            assert not any(_overlap(title, titles[j]) for j in range(i + 1, len(titles)))

    @pytest.mark.parametrize("n_blocks", [1, 2, 12], ids=["2taps", "4taps", "24taps"])
    def test_the_longer_subtitle_still_clears_everything_in_3d(
        self, monkeypatch, tmp_path, n_blocks
    ):
        names = _real_tap_names(n_blocks)
        rng = np.random.default_rng(3)
        maps = {n: rng.normal(size=(7, 8, 7)) for n in names}
        labels = {n: np.zeros((7, 8, 7), dtype=int) for n in names}
        for grid in labels.values():
            grid[1:3, 1:3, 1:3] = 1
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(
                maps, sig_grids={n: np.abs(m) > 1 for n, m in maps.items()},
                label_grids=labels, label_grids_b=labels,
                condition_a="a", condition_b="b", note=_LIVE_NOTE,
                output_path=str(tmp_path / "long3d.png"),
            ),
        )
        titles, frames, texts = _title_boxes_and_obstacles(fig)
        bounds = fig.bbox
        plt.close(fig)
        assert "dashed magenta = b" in " ".join(
            t.get_text().replace("\n", " ") for t in fig.texts
        )
        for index, (_, box) in enumerate(titles):
            assert _inside(box, bounds), f"title {index} clipped"
            assert not any(_overlap(box, text) for text in texts), index
            assert not any(
                _overlap(box, other) for j, (_, other) in enumerate(titles) if j > index
            )
        for text in texts:
            assert _inside(text, bounds)


def _populated_evaluation(tmp_path):
    """A real ``evaluate_topography`` with every unit significant, so both
    polarities form clusters on an untrained tiny head."""
    from train.topolm import SMOKE_STIMULI, evaluate_topography
    from tests.test_train.test_topolm.test_topolm_trainer import (
        StubTokenizer,
        _config,
    )
    from train.topolm import build_backbone

    backbone = build_backbone(_config())
    return evaluate_topography(
        backbone, SMOKE_STIMULI, ("a", "b"), StubTokenizer(),
        readout_fwhm=None, min_cluster_size=2, fdr_alpha=1.0,
        permutation_p_value=False,
    )


class TestTheCallSitesPassBothConditions:
    def test_the_arrays_carry_condition_b_labels_that_match_the_report(self, tmp_path):
        report, arrays = _populated_evaluation(tmp_path)
        arm = arrays["arms"]["raw"]
        assert set(arm) == {"t_maps", "sig_grids", "label_grids", "label_grids_b"}
        total_b = 0
        for name, entry in report["arms"]["raw"]["per_tap"].items():
            a, b = np.asarray(arm["label_grids"][name]), np.asarray(arm["label_grids_b"][name])
            assert not np.any((a > 0) & (b > 0)), "one cell in two conditions' clusters"
            assert int((a > 0).sum()) == sum(entry["cluster_sizes_a"])
            assert int((b > 0).sum()) == sum(entry["cluster_sizes_b"])
            assert len(np.unique(b[b > 0])) == len(entry["cluster_sizes_b"])
            total_b += len(entry["cluster_sizes_b"])
        assert total_b > 0, "no B cluster formed: the check above was vacuous"

    def test_plot_topography_outlines_both_from_the_evaluation(self, monkeypatch, tmp_path):
        from train.topolm import plot_topography

        report, arrays = _populated_evaluation(tmp_path)
        holder = {}
        original = Figure.savefig

        def recording(self, path, *args, **kwargs):
            if "t_maps_raw" in str(path):
                holder["fig"] = self
            return original(self, path, *args, **kwargs)

        monkeypatch.setattr(Figure, "savefig", recording)
        plot_topography(report, arrays, str(tmp_path))
        fig = holder["fig"]
        raw = arrays["arms"]["raw"]
        checked_b = 0
        for axis, name in zip(_image_axes(fig), raw["t_maps"]):
            assert _pieces_in_colour(axis, _A_COLOUR) == _expected_pieces(raw["label_grids"][name])
            drawn_b = _pieces_in_colour(axis, _B_COLOUR)
            assert drawn_b == _expected_pieces(raw["label_grids_b"][name])
            checked_b += len(drawn_b)
        text = _subtitle_of(fig)
        plt.close(fig)
        assert checked_b > 0
        assert "dashed magenta = b" in text

    def test_arrays_without_condition_b_grids_still_plot(self, tmp_path):
        from train.topolm import plot_topography

        report, arrays = _cluster_report_and_arrays(["t0", "t1"], "t1")
        assert "label_grids_b" not in arrays["arms"]["raw"]
        written = plot_topography(report, arrays, str(tmp_path))
        assert any(p.endswith("t_maps_raw.png") for p in written)


# -- the live curve's legend ---------------------------------------------------


def _legend_rows(names, steps=(0, 25, 50)):
    return [
        {
            "step": step, "val_loss": None, "val_loss_step": None,
            "mean_morans_i": 0.05 * i, "total_clusters_a": i, "total_clusters_b": i,
            "mean_num_significant_units": 3.0,
            "per_tap": {n: {"morans_i": 0.05 * i + 0.01 * k} for k, n in enumerate(names)},
        }
        for i, step in enumerate(steps)
    ]


class TestTheCurveNamesItsTapLines:
    @pytest.mark.parametrize("n_blocks", [1, 3, 12], ids=["2taps", "6taps", "24taps"])
    def test_the_legend_sits_right_of_the_axes_and_inside_the_figure(
        self, monkeypatch, tmp_path, n_blocks
    ):
        names = _real_tap_names(n_blocks)
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_topography_curve(
                _legend_rows(names), output_path=str(tmp_path / "c.png"),
                note="smoke stimuli, n=5/5 per condition",
            ),
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        moran_axis = fig.axes[0]
        legend = moran_axis.get_legend()
        assert legend is not None, "the curve has no legend"
        box = legend.get_window_extent(renderer)
        frames = [ax.get_window_extent(renderer) for ax in fig.axes]
        texts = [t.get_window_extent(renderer) for t in fig.texts if t.get_text()]
        entries = [t.get_text() for t in legend.get_texts()]
        bounds = fig.bbox
        # What ``savefig(bbox_inches="tight")`` keeps: a legend that is not in the
        # layout is rendered on the canvas and CROPPED OUT of the saved PNG.
        saved = fig.get_tightbbox(renderer)
        saved_px = (saved.x0 * fig.dpi, saved.y0 * fig.dpi, saved.x1 * fig.dpi, saved.y1 * fig.dpi)
        plt.close(fig)

        assert saved_px[0] - 0.5 <= box.x0 and box.x1 <= saved_px[2] + 0.5, (
            f"the saved figure's extent {saved_px} cuts the legend {box}"
        )
        assert saved_px[1] - 0.5 <= box.y0 and box.y1 <= saved_px[3] + 0.5, (
            f"the saved figure's extent {saved_px} cuts the legend {box}"
        )
        assert len(entries) == len(names) + 1, entries     # the taps and the mean line
        assert "mean over taps" in entries
        # the shortened labels: the shared prefix is gone, the block and tap stay
        parts = [n.split("/") for n in names]
        shared = 0
        while (
            len(names) > 1 and shared < min(map(len, parts)) - 1
            and len({p[shared] for p in parts}) == 1
        ):
            shared += 1
        for path in parts:
            assert "/".join(path[shared:]) in entries, (path, shared, entries)
        # right of every axes, below nothing it could cover, inside the figure
        for index, frame in enumerate(frames):
            assert box.x0 >= frame.x1 - 0.5 or box.y1 <= frame.y0 or box.y0 >= frame.y1, (
                f"the legend covers axes {index}: {box} {frame}"
            )
        assert box.x0 >= frames[0].x1 - 0.5, "the legend is not to the right of the axes"
        assert _inside(box, bounds), f"the legend leaves the figure: {box} in {bounds}"
        for text in texts:
            assert not _overlap(box, text)

    def test_24_taps_use_two_columns_and_a_smaller_font_than_2(self, monkeypatch, tmp_path):
        def legend_of(n_blocks):
            fig = _captured_figure(
                monkeypatch,
                lambda: plot_topography_curve(
                    _legend_rows(_real_tap_names(n_blocks)),
                    output_path=str(tmp_path / f"c{n_blocks}.png"),
                ),
            )
            legend = fig.axes[0].get_legend()
            info = (legend._ncols, legend.get_texts()[0].get_fontsize())
            plt.close(fig)
            return info

        few, many = legend_of(1), legend_of(12)
        assert few[0] == 1 and many[0] == 2
        assert many[1] < few[1]

    def test_every_tap_line_is_named_by_its_legend_entry(self, monkeypatch, tmp_path):
        names = _real_tap_names(2)
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_topography_curve(
                _legend_rows(names), output_path=str(tmp_path / "c.png")
            ),
        )
        legend = fig.axes[0].get_legend()
        handles = legend.legend_handles
        entries = [t.get_text() for t in legend.get_texts()]
        plt.close(fig)
        assert len(handles) == len(entries) == len(names) + 1


# =====================================================================
# Step 4.5 (plan-2026-10-23T190557-972a9ac5, decision D-019, defects D2, D3, D5,
# D6): what the real GPU run's figures got wrong. Written RED-first against the
# c28fc90b5 drawers.
# =====================================================================

import re


def _cluster_report_3d(names, populated, shape=(4, 5, 5), contrast=("a", "b")):
    """Like :func:`_cluster_report_and_arrays`, with 3-D grids (a montage)."""
    rng = np.random.default_rng(2)
    labels = {n: np.zeros(shape, dtype=int) for n in names}
    labels[populated][1:3, 1:4, 1:3] = 1
    report = {
        "contrast": list(contrast),
        "arms": {"raw": {"per_tap": {
            n: {"morans_i": {"standard": 0.3}} for n in names}}},
    }
    arrays = {"arms": {"raw": {
        "t_maps": {n: rng.normal(size=shape) * 2 for n in names},
        "sig_grids": {n: np.zeros(shape, dtype=bool) for n in names},
        "label_grids": labels,
    }}}
    return report, arrays


def _cluster_figure(monkeypatch, tmp_path, report, arrays):
    from train.topolm import common

    holder = {}
    original = Figure.savefig

    def recording(self, path, *args, **kwargs):
        if "clusters_" in str(path):
            holder["fig"] = self
        return original(self, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", recording)
    common.plot_topography(report, arrays, str(tmp_path))
    return holder["fig"]


class TestTheClusterFigureNamesTheConditionItShows:
    """D2: ``clusters_raw.png`` drew condition A's clusters and said nothing about
    it, while the t-map of the same tap outlines both conditions."""

    def test_the_2d_title_names_condition_a(self, monkeypatch, tmp_path):
        names = _real_tap_names(4)
        report, arrays = _cluster_report_and_arrays(names, names[-1])
        fig = _cluster_figure(monkeypatch, tmp_path, report, arrays)
        title = fig.axes[0].get_title()
        plt.close(fig)
        assert title == "block_3/mlp_tap (raw), condition a: 1 cluster"

    def test_the_name_is_the_contrasts_first_condition_not_a_literal(
        self, monkeypatch, tmp_path
    ):
        names = _real_tap_names(2)
        report, arrays = _cluster_report_and_arrays(names, names[-1])
        report["contrast"] = ["tall", "short"]
        fig = _cluster_figure(monkeypatch, tmp_path, report, arrays)
        title = fig.axes[0].get_title()
        plt.close(fig)
        assert "condition tall" in title and "short" not in title, title

    def test_the_figure_shows_the_grid_the_title_names(self, monkeypatch, tmp_path):
        """The condition in the title is the condition of the drawn labels: the
        figure's image holds A's cluster cells and nothing of B's."""
        names = _real_tap_names(2)
        report, arrays = _cluster_report_and_arrays(names, names[-1])
        arrays["arms"]["raw"]["label_grids_b"] = {
            n: np.zeros((8, 8), dtype=int) for n in names
        }
        arrays["arms"]["raw"]["label_grids_b"][names[-1]][6:8, 5:8] = 1
        fig = _cluster_figure(monkeypatch, tmp_path, report, arrays)
        drawn = np.asarray(fig.axes[0].images[0].get_array())
        plt.close(fig)
        assert np.array_equal(drawn > 0, arrays["arms"]["raw"]["label_grids"][names[-1]] > 0)

    def test_the_3d_montage_title_names_condition_a_too(self, monkeypatch, tmp_path):
        names = _real_tap_names(2)
        report, arrays = _cluster_report_3d(names, names[-1])
        fig = _cluster_figure(monkeypatch, tmp_path, report, arrays)
        title = fig._suptitle.get_text()
        plt.close(fig)
        assert title == "block_1/mlp_tap (raw), condition a: 1 cluster", title

    @pytest.mark.parametrize("three_d", [False, True], ids=["2d", "3d"])
    def test_a_long_title_stays_inside_the_figure(self, monkeypatch, tmp_path, three_d):
        """The longest realistic title (a readout arm, a double-digit block, a
        clipped palette) must not leave the figure."""
        names = _real_tap_names(12)
        build = _cluster_report_3d if three_d else _cluster_report_and_arrays
        report, arrays = build(names, names[-2])
        report["contrast"] = ["sentences_a", "sentences_b"]
        arrays["arms"]["readout"] = arrays["arms"].pop("raw")
        report["arms"]["readout"] = report["arms"].pop("raw")
        fig = _cluster_figure(monkeypatch, tmp_path, report, arrays)
        text = fig._suptitle if three_d else fig.axes[0].title
        fig.canvas.draw()
        box = text.get_window_extent(fig.canvas.get_renderer())
        content = text.get_text()
        bounds = fig.bbox
        plt.close(fig)
        assert "condition sentences_a" in content.replace("\n", " ")
        assert _inside(box, bounds), (content, box, bounds)


def _tap_lines(fig):
    """The per-tap Moran's I lines of a curve figure, in tap order (not the mean,
    not the zero line)."""
    axis = fig.axes[0]
    return [
        line for line in axis.lines
        if line.get_label() and not line.get_label().startswith(("mean", "_"))
    ]


def _rgb_distance(a, b):
    return float(np.linalg.norm(np.asarray(a[:3]) - np.asarray(b[:3])))


def _curve_figure(monkeypatch, tmp_path, n_blocks):
    names = _real_tap_names(n_blocks)
    return _captured_figure(
        monkeypatch,
        lambda: plot_topography_curve(
            _legend_rows(names), output_path=str(tmp_path / f"palette{n_blocks}.png")
        ),
    )


class TestTheTapLinesCanBeToldApart:
    """D3: eight tap lines in near-identical viridis hues could not be matched to
    the legend where they overlapped."""

    @pytest.mark.parametrize("n_blocks", [1, 4, 5], ids=["2taps", "8taps", "10taps"])
    def test_up_to_ten_taps_get_well_separated_colours_and_their_own_marker(
        self, monkeypatch, tmp_path, n_blocks
    ):
        fig = _curve_figure(monkeypatch, tmp_path, n_blocks)
        lines = _tap_lines(fig)
        plt.close(fig)
        assert len(lines) == 2 * n_blocks
        colours = [matplotlib.colors.to_rgba(line.get_color()) for line in lines]
        nearest = min(
            _rgb_distance(a, b)
            for i, a in enumerate(colours) for b in colours[i + 1:]
        )
        assert nearest >= 0.25, f"two tap lines are nearly the same colour ({nearest:.3f})"
        assert len({line.get_marker() for line in lines}) == len(lines), "markers repeat"

    @pytest.mark.parametrize("n_blocks", [1, 5, 8, 12], ids=["2taps", "10taps", "16taps", "24taps"])
    def test_no_two_tap_lines_share_colour_marker_and_style(
        self, monkeypatch, tmp_path, n_blocks
    ):
        fig = _curve_figure(monkeypatch, tmp_path, n_blocks)
        lines = _tap_lines(fig)
        plt.close(fig)
        triples = [
            (matplotlib.colors.to_hex(line.get_color()), line.get_marker(),
             line.get_linestyle())
            for line in lines
        ]
        assert len(set(triples)) == len(triples), triples

    def test_beyond_ten_taps_the_colours_come_from_a_20_colour_categorical_palette(
        self, monkeypatch, tmp_path
    ):
        fig = _curve_figure(monkeypatch, tmp_path, 12)
        lines = _tap_lines(fig)
        plt.close(fig)
        colours = [matplotlib.colors.to_hex(line.get_color()) for line in lines]
        tab20 = {matplotlib.colors.to_hex(c) for c in plt.get_cmap("tab20").colors}
        assert len(lines) == 24
        assert set(colours) <= tab20, "a colour outside the categorical palette"
        assert len(set(colours)) == 20, "the 20 colours are not all used"

    def test_the_mean_line_is_black_and_no_tap_line_is(self, monkeypatch, tmp_path):
        fig = _curve_figure(monkeypatch, tmp_path, 3)
        mean = next(l for l in fig.axes[0].lines if l.get_label() == "mean over taps")
        taps = _tap_lines(fig)
        plt.close(fig)
        assert matplotlib.colors.same_color(mean.get_color(), "black")
        assert not any(matplotlib.colors.same_color(l.get_color(), "black") for l in taps)

    @pytest.mark.parametrize("n_blocks", [1, 4, 12], ids=["2taps", "8taps", "24taps"])
    def test_every_legend_handle_matches_its_line(self, monkeypatch, tmp_path, n_blocks):
        fig = _curve_figure(monkeypatch, tmp_path, n_blocks)
        legend = fig.axes[0].get_legend()
        by_label = {l.get_label(): l for l in fig.axes[0].lines if l.get_label()}
        checked = 0
        for handle, text in zip(legend.legend_handles, legend.get_texts()):
            line = by_label[text.get_text()]
            assert matplotlib.colors.same_color(handle.get_color(), line.get_color())
            assert handle.get_marker() == line.get_marker(), text.get_text()
            assert handle.get_linestyle() == line.get_linestyle(), text.get_text()
            checked += 1
        plt.close(fig)
        assert checked == 2 * n_blocks + 1

    def test_the_lines_are_visible_enough_to_follow(self, monkeypatch, tmp_path):
        fig = _curve_figure(monkeypatch, tmp_path, 4)
        lines = _tap_lines(fig)
        plt.close(fig)
        assert all(l.get_linewidth() >= 0.8 and l.get_alpha() >= 0.75 for l in lines)


class TestTheCrosshairSentenceIsTrueAtEveryStep:
    """D5: the 3-D subtitle said the crosshair sits at the largest SIGNIFICANT
    ``|t|`` at step 0, where nothing is significant."""

    def _text(self, monkeypatch, tmp_path, volume, significant, n_taps=1):
        names = _real_tap_names(n_taps)[:n_taps]
        maps = {n: volume for n in names}
        sig = None if significant is None else {
            n: (significant[i] if isinstance(significant, list) else significant)
            for i, n in enumerate(names)
        }
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(
                maps, sig_grids=sig, condition_a="a", condition_b="b",
                output_path=str(tmp_path / "cross.png"),
            ),
        )
        text = _subtitle_of(fig)
        plt.close(fig)
        return text

    def test_when_something_is_significant_it_says_so_plainly(self, monkeypatch, tmp_path):
        volume, significant = _volume_fixture()
        text = self._text(monkeypatch, tmp_path, volume, significant)
        assert "dashed green = crosshair at the largest significant |t|" in text
        assert "nothing is significant" not in text and "where none" not in text

    def test_when_nothing_is_significant_it_does_not_claim_significance(
        self, monkeypatch, tmp_path
    ):
        volume, _ = _volume_fixture()
        text = self._text(monkeypatch, tmp_path, volume, np.zeros(_VOLUME, dtype=bool))
        assert "dashed green = crosshair at the largest |t| (nothing is significant)" in text
        assert "largest significant" not in text

    def test_without_a_mask_it_names_the_largest_abs_t_and_claims_nothing_more(
        self, monkeypatch, tmp_path
    ):
        volume, _ = _volume_fixture()
        text = self._text(monkeypatch, tmp_path, volume, None)
        assert "dashed green = crosshair at the largest |t|" in text
        assert "largest significant" not in text and "nothing is significant" not in text

    def test_when_only_some_taps_have_significant_cells_it_names_the_fallback(
        self, monkeypatch, tmp_path
    ):
        volume, significant = _volume_fixture()
        none = np.zeros(_VOLUME, dtype=bool)
        text = self._text(monkeypatch, tmp_path, volume, [significant, none], n_taps=2)
        assert (
            "crosshair at the largest significant |t| "
            "(the largest |t| where none is significant)"
        ) in text

    def test_a_mask_of_the_wrong_shape_counts_as_no_mask(self, monkeypatch, tmp_path):
        volume, _ = _volume_fixture()
        text = self._text(monkeypatch, tmp_path, volume, np.ones((2, 2, 2), dtype=bool))
        assert "largest significant" not in text

    def test_the_sentence_and_the_crosshair_use_one_rule(self):
        """The count of taps the sentence calls significant is the count whose
        crosshair is restricted to significant cells (both from the same helper)."""
        from train.topolm.plotting import _crosshair_voxel

        volume, significant = _volume_fixture()
        assert _crosshair_voxel(volume, significant) == (5, 1, 9)
        assert _crosshair_voxel(volume, np.zeros(_VOLUME, dtype=bool)) == (0, 7, 13)


class TestTheSubtitleBreaksAtClauses:
    """D6: the 3-D subtitle wrapped mid-phrase (``dashed`` at the end of one line,
    ``magenta = b`` at the start of the next)."""

    TEXT = (
        "positive = a, negative = b; hatched = not significant after BH-FDR; "
        "outline = cluster of significant units (solid black = a, dashed magenta = b); "
        "dashed green = crosshair at the largest significant |t| | step 400 | "
        "smoke stimuli, n=5/5 per condition"
    )

    def _clauses(self, text):
        # a clause ends at "; " or at the bar of " | " (never inside "|t|")
        return re.split(r"(?<=;) |(?<= \|) ", text)

    @pytest.mark.parametrize("width_in", [6.0, 7.5, 9.6])
    def test_every_break_falls_on_a_clause_boundary_when_each_clause_fits(self, width_in):
        from train.topolm.plotting import _SUBTITLE_CHARS_PER_INCH, _wrap_subtitle

        columns = max(20, int((width_in - 0.4) * _SUBTITLE_CHARS_PER_INCH))
        assert max(map(len, self._clauses(self.TEXT))) <= columns, "fixture too wide"
        wrapped = _wrap_subtitle(self.TEXT, width_in)
        lines = wrapped.split("\n")
        assert len(lines) > 1
        for line in lines[:-1]:
            assert line.endswith((";", " |")), line
        assert all(len(line) <= columns for line in lines)
        assert " ".join(lines) == self.TEXT, "the wrap lost or changed text"

    def test_a_clause_wider_than_the_figure_falls_back_to_spaces(self):
        from train.topolm.plotting import _wrap_subtitle

        wrapped = _wrap_subtitle(self.TEXT, 3.0)
        lines = wrapped.split("\n")
        assert " ".join(lines) == self.TEXT
        assert all(len(line) <= 39 for line in lines), lines

    def test_the_bars_of_abs_t_are_not_a_boundary(self):
        """``|t|`` has a pipe on each side and a space after the second: a break
        there would split ``the largest |t|`` from the words it belongs to. The
        clause fits on a line of its own, so it must move down whole."""
        from train.topolm.plotting import _wrap_subtitle

        first = "A" * 21 + ";"
        second = "the largest |t| " + "B" * 20
        wrapped = _wrap_subtitle(f"{first} {second}", 3.2)       # 42 columns
        assert wrapped.split("\n") == [first, second]

    def test_a_bar_between_spaces_is_a_boundary(self):
        from train.topolm.plotting import _wrap_subtitle

        first = "A" * 20 + " |"
        second = "B" * 30
        assert _wrap_subtitle(f"{first} {second}", 3.2).split("\n") == [first, second]

    def test_text_that_fits_is_unchanged(self):
        from train.topolm.plotting import _wrap_subtitle

        assert _wrap_subtitle("positive = a, negative = b", 6.0) == "positive = a, negative = b"

    def test_the_real_3d_subtitle_never_breaks_inside_a_phrase(self, monkeypatch, tmp_path):
        names = _real_tap_names(2)
        rng = np.random.default_rng(3)
        maps = {n: rng.normal(size=(7, 8, 7)) for n in names}
        labels = {n: np.zeros((7, 8, 7), dtype=int) for n in names}
        for grid in labels.values():
            grid[1:3, 1:3, 1:3] = 1
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(
                maps, sig_grids={n: np.abs(m) > 1 for n, m in maps.items()},
                label_grids=labels, label_grids_b=labels, condition_a="a",
                condition_b="b", note=_LIVE_NOTE,
                output_path=str(tmp_path / "wrap3d.png"),
            ),
        )
        subtitle = next(
            t.get_text() for t in fig.texts if t is not fig._suptitle and t.get_text()
        )
        plt.close(fig)
        lines = subtitle.split("\n")
        assert len(lines) >= 2, subtitle
        for line in lines[:-1]:
            assert line.endswith((";", " |")), f"a break inside a phrase: {line!r}"


# =====================================================================
# Step 4.6 (plan-2026-10-23T190557-972a9ac5, decision D-021): the hatch marks EVERY
# rejected cell, and the count panels of the live curve have integer ticks.
# Written RED-first against the c5cca75c3 drawers: ``contourf`` of a masked array
# fills only the quads whose four corner cells are all rejected, so an isolated
# rejected cell and a one-cell-wide strip carried no hatch (326 of 545 rejected
# cells, 60 %, on the trained 2-D example) while the line under the headline says
# "hatched = not significant after BH-FDR".
# =====================================================================


def _mask_cases():
    """``{name: significant}`` for the 2-D panel; ``~significant`` is what must be
    hatched. Covers the shapes a contour-based hatch gets wrong (isolated cell,
    one-cell strips, checkerboard) and the extremes (none, all)."""
    rng = np.random.default_rng(5)
    isolated = np.ones((8, 8), dtype=bool)
    isolated[3, 4] = False
    vertical = np.ones((8, 8), dtype=bool)
    vertical[:, 2] = False
    horizontal = np.ones((8, 8), dtype=bool)
    horizontal[5, :] = False
    corner = np.ones((8, 8), dtype=bool)
    corner[0, 0] = False
    corner[7, 7] = False
    return {
        "random": rng.random((10, 10)) < 0.6,
        "isolated_cell": isolated,
        "one_cell_vertical_strip": vertical,
        "one_cell_horizontal_strip": horizontal,
        "two_corner_cells": corner,
        "checkerboard": (np.indices((8, 8)).sum(axis=0) % 2).astype(bool),
        "all_rejected": np.zeros((6, 6), dtype=bool),
        "none_rejected": np.ones((6, 6), dtype=bool),
        "non_square": rng.random((5, 9)) < 0.5,
        "single_row": rng.random((1, 7)) < 0.5,
    }


_MASK_CASES = _mask_cases()


class TestTheHatchMarksEveryRejectedCell:
    """RED on c5cca75c3: its hatch exists only where all four neighbouring cells
    are rejected, so every case with an isolated cell or a one-cell strip fails and
    so does the rejected border of a full block. The oracle reads the artists on the
    axes of the PUBLIC figure functions; it never calls ``_hatch_rejected``."""

    @pytest.mark.parametrize("name", list(_MASK_CASES))
    def test_the_2d_panel_hatches_exactly_the_rejected_cells(self, name):
        significant = _MASK_CASES[name]
        fig, axes = _capture_axes(
            {"t": np.zeros(significant.shape)}, sig_grids={"t": significant},
            _fn="plot_t_maps",
        )
        axis = np.atleast_2d(axes)[0, 0]
        try:
            _assert_hatch_is_exactly(axis, significant)
        finally:
            plt.close(fig)

    @pytest.mark.parametrize("name", list(_MASK_CASES))
    def test_the_hatch_is_one_exact_cell_box_per_rejected_cell(self, name):
        """The boxes themselves: cell ``(r, c)`` is ``[c - 0.5, c + 0.5] x
        [r - 0.5, r + 0.5]`` in the coordinates the panel's image uses, each rejected
        cell once, no box for an accepted cell."""
        significant = _MASK_CASES[name]
        fig, axes = _capture_axes(
            {"t": np.zeros(significant.shape)}, sig_grids={"t": significant},
            _fn="plot_t_maps",
        )
        boxes = _hatch_boxes(np.atleast_2d(axes)[0, 0])
        plt.close(fig)
        expected = [
            (col - 0.5, row - 0.5, col + 0.5, row + 0.5)
            for row, col in np.argwhere(~significant)
        ]
        assert sorted(boxes) == sorted(expected)

    def test_nothing_rejected_draws_no_hatch_artist_at_all(self):
        significant = np.ones((6, 6), dtype=bool)
        fig, axes = _capture_axes(
            {"t": np.zeros((6, 6))}, sig_grids={"t": significant}, _fn="plot_t_maps",
        )
        axis = np.atleast_2d(axes)[0, 0]
        found = _hatch_paths(axis)
        plt.close(fig)
        assert found == []

    def test_the_old_geometry_is_what_the_oracle_rejects(self):
        """The oracle can fail: a contour-style hatch (the cells whose four corner
        cells are all rejected, as ``contourf`` drew it) is refused on an isolated
        cell. Without this the passing cases above prove nothing."""
        from matplotlib.patches import Rectangle

        significant = _MASK_CASES["isolated_cell"]
        fig, axis = plt.subplots()
        axis.imshow(np.zeros(significant.shape))
        for row in range(significant.shape[0] - 1):
            for col in range(significant.shape[1] - 1):
                if not significant[row:row + 2, col:col + 2].any():
                    axis.add_patch(Rectangle(
                        (col, row), 1, 1, fill=False, hatch="//", linewidth=0,
                    ))
        try:
            with pytest.raises(AssertionError, match="rejected cells are not"):
                _assert_hatch_is_exactly(axis, significant)
        finally:
            plt.close(fig)

    def test_the_hatch_pattern_and_look_are_the_ones_it_always_had(self):
        """The pattern is ``///`` since step 4.7 (D-026; ``//`` before), no fill, the
        rcParams hatch colour, and no outline of the boxes (line width 0): only the
        hatch strokes show, as before."""
        significant = _MASK_CASES["random"]
        fig, axes = _capture_axes(
            {"t": np.zeros(significant.shape)}, sig_grids={"t": significant},
            _fn="plot_t_maps",
        )
        axis = np.atleast_2d(axes)[0, 0]
        artists = _hatched(axis)
        plt.close(fig)
        assert len(artists) == 1
        artist = artists[0]
        patterns = [artist.get_hatch()] if artist.get_hatch() else list(artist.hatches)
        assert patterns == ["///"]
        assert all(c[3] == 0 for c in np.atleast_2d(artist.get_facecolor()).reshape(-1, 4)), (
            "the cell boxes must not be filled: only the hatch shows"
        )
        edge = np.atleast_2d(artist.get_edgecolor()).reshape(-1, 4)
        if len(edge) and np.all(np.asarray(artist.get_linewidth()) > 0):
            pytest.fail("the cell boxes are outlined: only the hatch must show")
        # The hatch strokes take the EDGE colour; an invisible one draws no hatch.
        assert len(edge) == 1 and edge[0][3] == 1.0
        assert tuple(edge[0]) == matplotlib.colors.to_rgba(matplotlib.rcParams["hatch.color"])

    def test_the_hatch_strokes_are_actually_painted(self, tmp_path):
        """The artists can be exactly right and draw nothing (an invisible edge
        colour gives no hatch strokes): read the PIXELS. On a mid-grey image, an
        all-rejected panel has dark strokes in the panel and a panel with nothing
        rejected has none."""
        def dark_pixels(significant):
            path = plot_t_maps(
                {"t": np.zeros(significant.shape)}, sig_grids={"t": significant},
                output_path=str(tmp_path / f"p{int(significant.sum())}.png"),
            )
            pixels = mpimg.imread(path)[..., :3].mean(axis=2)
            return int((pixels < 0.2).sum())

        none_rejected = dark_pixels(np.ones((6, 6), dtype=bool))
        all_rejected = dark_pixels(np.zeros((6, 6), dtype=bool))
        assert all_rejected > none_rejected + 100, (none_rejected, all_rejected)

    def test_the_hatch_is_above_the_image_and_below_the_cluster_outlines(self):
        significant = _MASK_CASES["random"]
        labels = np.zeros(significant.shape, dtype=int)
        labels[2:5, 2:5] = 1
        fig, axes = _capture_axes(
            {"t": np.zeros(significant.shape)}, sig_grids={"t": significant},
            label_grids={"t": labels}, _fn="plot_t_maps",
        )
        axis = np.atleast_2d(axes)[0, 0]
        hatch_z = [artist.get_zorder() for artist in _hatched(axis)]
        image_z = axis.images[0].get_zorder()
        line_z = [line.get_zorder() for line in axis.get_lines()]
        plt.close(fig)
        assert hatch_z and line_z
        assert min(hatch_z) > image_z, "the hatch is under the image"
        assert max(hatch_z) < min(line_z), "the hatch is over the cluster outlines"

    def test_every_tap_panel_is_hatched_from_its_own_mask(self):
        rng = np.random.default_rng(8)
        masks = {f"blocks.{i}.ffn/attn_pre": rng.random((7, 7)) < 0.5 for i in range(4)}
        fig, axes = _capture_axes(
            {k: np.zeros((7, 7)) for k in masks}, sig_grids=masks, _fn="plot_t_maps",
        )
        panels = [ax for ax in np.atleast_2d(axes).ravel() if ax.images]
        try:
            assert len(panels) == len(masks)
            for axis, mask in zip(panels, masks.values()):
                _assert_hatch_is_exactly(axis, mask)
        finally:
            plt.close(fig)


def _distinct_volume(shape, crosshair):
    """A volume whose every voxel has its own value, with the largest one at
    ``crosshair``, so a drawn plane tells which voxels it shows, and the crosshair
    (the largest ``|t|`` among the significant voxels) is where the test says."""
    volume = np.arange(int(np.prod(shape)), dtype=float).reshape(shape)
    volume[crosshair] = volume.max() + 1.0
    return volume


def _expected_plane_masks(volume, significant, plane_array):
    """The significance of each cell of a drawn plane, found by looking each cell's
    value up in the volume, so the expectation does not depend on how the module
    picks the plane."""
    where = {float(v): idx for idx, v in np.ndenumerate(volume)}
    out = np.zeros(plane_array.shape, dtype=bool)
    for (row, col), value in np.ndenumerate(plane_array):
        out[row, col] = significant[where[float(value)]]
    return out


def _volume_mask_cases():
    cases = {}
    shape, crosshair = (7, 7, 7), (3, 5, 6)

    random = np.random.default_rng(2).random(shape) < 0.55
    random[crosshair] = True
    cases["random"] = (shape, crosshair, random)

    isolated = np.ones(shape, dtype=bool)
    isolated[3, 2, 4] = False                       # on the axis-0 plane through (3, 5, 6)
    cases["isolated_voxel"] = (shape, crosshair, isolated)

    strip = np.ones(shape, dtype=bool)
    strip[3, 0:7, 2] = False                        # a one-cell strip on the axis-0 plane
    cases["one_cell_strip"] = (shape, crosshair, strip)

    cases["all_rejected"] = (shape, crosshair, np.zeros(shape, dtype=bool))
    cases["none_rejected"] = (shape, crosshair, np.ones(shape, dtype=bool))

    odd_shape, odd_crosshair = (4, 6, 5), (1, 3, 2)
    odd = np.random.default_rng(4).random(odd_shape) < 0.5
    odd[odd_crosshair] = True
    cases["non_cubic"] = (odd_shape, odd_crosshair, odd)
    return cases


_VOLUME_MASK_CASES = _volume_mask_cases()


class TestTheHatchMarksEveryRejectedVoxelOfEveryPlane:
    """The same oracle on the 3-D figure: each of the three planes through the
    crosshair hatches exactly the voxels that failed BH-FDR on that plane."""

    @pytest.mark.parametrize("name", list(_VOLUME_MASK_CASES))
    def test_each_plane_hatches_exactly_its_rejected_cells(
        self, monkeypatch, tmp_path, name
    ):
        shape, crosshair, significant = _VOLUME_MASK_CASES[name]
        volume = _distinct_volume(shape, crosshair)
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(
                {"t": volume}, sig_grids={"t": significant},
                output_path=str(tmp_path / "t3d.png"),
            ),
        )
        planes = _image_axes(fig)
        try:
            assert len(planes) == 3
            hatched_cells = 0
            for axis in planes:
                shown = np.asarray(axis.images[0].get_array())
                plane_significant = _expected_plane_masks(volume, significant, shown)
                _assert_hatch_is_exactly(axis, plane_significant)
                hatched_cells += int((~plane_significant).sum())
            if name == "none_rejected":
                assert hatched_cells == 0
            else:
                assert hatched_cells > 0, "the case must put a rejected cell on a plane"
        finally:
            plt.close(fig)

    def test_the_isolated_voxel_is_on_the_plane_the_case_claims(
        self, monkeypatch, tmp_path
    ):
        """Guards the fixture, not the module: the isolated rejected voxel
        (3, 2, 4) really is on one of the drawn planes, so the case above can fail."""
        shape, crosshair, significant = _VOLUME_MASK_CASES["isolated_voxel"]
        volume = _distinct_volume(shape, crosshair)
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_t_maps(
                {"t": volume}, sig_grids={"t": significant},
                output_path=str(tmp_path / "t3d.png"),
            ),
        )
        shown = [np.asarray(ax.images[0].get_array()) for ax in _image_axes(fig)]
        plt.close(fig)
        target = float(volume[3, 2, 4])
        assert any(np.any(plane == target) for plane in shown)


class TestTheCurveCountPanelsHaveIntegerTicks:
    """RED on c5cca75c3: with counts of 0 to 3 the clusters panel was ticked at
    0.5 steps and the significant-units panel at 0.25 steps, for quantities that
    are counted (or, for the units, averaged over taps but read as a count). The
    first frames of a run are the hard case: ``MaxNLocator(integer=True)`` alone
    falls back to ticks like -0.04, 0.04 when the view spans less than one unit."""

    #: ``(clusters_a, clusters_b, mean significant units)`` per draw.
    CASES = {
        "a_few": [(0, 0, 0.0), (1, 1, 0.5), (2, 1, 1.5), (3, 2, 2.0)],
        "first_frame_nothing_yet": [(0, 0, 0.0)],
        "all_zero": [(0, 0, 0.0), (0, 0, 0.0), (0, 0, 0.0)],
        "less_than_one_unit": [(0, 0, 0.0), (0, 0, 0.25), (0, 0, 0.5)],
        "one_cluster": [(0, 0, 0.0), (1, 0, 1.0)],
        "many": [(0, 0, 0.0), (7, 4, 51.5), (14, 9, 190.25)],
    }

    @classmethod
    def _rows(cls, case):
        counts = cls.CASES[case]
        rows = _curve_rows(
            [50 * i for i in range(len(counts))], [None] * len(counts),
            [None] * len(counts),
        )
        for row, (a, b, units) in zip(rows, counts):
            row["total_clusters_a"], row["total_clusters_b"] = a, b
            row["mean_num_significant_units"] = units
        return rows

    @staticmethod
    def _visible_ticks(axis):
        low, high = sorted(axis.get_ylim())
        return [float(t) for t in axis.get_yticks() if low - 1e-9 <= t <= high + 1e-9]

    def _axis(self, monkeypatch, tmp_path, case, label):
        fig = _captured_figure(
            monkeypatch,
            lambda: plot_topography_curve(
                self._rows(case), output_path=str(tmp_path / "c.png")
            ),
        )
        fig.canvas.draw()
        axis = next(ax for ax in fig.axes if ax.get_ylabel().startswith(label))
        return fig, axis

    @pytest.mark.parametrize("case", list(CASES))
    @pytest.mark.parametrize("label", ["clusters", "significant units"])
    def test_the_count_panels_tick_at_whole_numbers(
        self, monkeypatch, tmp_path, label, case
    ):
        fig, axis = self._axis(monkeypatch, tmp_path, case, label)
        ticks = self._visible_ticks(axis)
        plt.close(fig)
        assert len(ticks) >= 2, ticks
        assert all(t == round(t) for t in ticks), ticks

    def test_the_data_stays_inside_the_view_of_a_widened_count_panel(
        self, monkeypatch, tmp_path
    ):
        """Widening the view for the integer ticks must not clip the line."""
        fig, axis = self._axis(monkeypatch, tmp_path, "less_than_one_unit", "significant units")
        low, high = sorted(axis.get_ylim())
        values = list(map(float, axis.lines[0].get_ydata()))
        plt.close(fig)
        assert low <= min(values) and max(values) <= high

    def test_the_other_panels_keep_their_own_ticks(self, monkeypatch, tmp_path):
        """Moran's I is not a count: its ticks must stay free to be fractional
        (guards against the integer locator being applied to every panel)."""
        fig, axis = self._axis(monkeypatch, tmp_path, "a_few", "Moran")
        ticks = self._visible_ticks(axis)
        plt.close(fig)
        assert any(t != round(t) for t in ticks), ticks


# =====================================================================
# Step 4.7 (plan-2026-10-23T190557-972a9ac5, decision D-026): the hatch strokes are
# dense enough that every rejected cell carries some. Written RED-first against the
# ``//`` pattern, whose strokes are about two cells apart: at the figures' saved
# scale 241 of 868 rejected cells of a trained-like 2-D figure (the reviewer
# measured 244 of 1737 on the real example) had no stroke pixel in their middle
# half, although the line under the headline says "hatched = not significant after
# BH-FDR" and the guide says a hatched cell did not pass. The per-cell boxes of step
# 4.6 are exact, and exactness of the BOXES does not make the STROKES reach every
# box: only the pixels can say that.
# =====================================================================

#: The figures are saved at ``dpi=160``; the stroke pitch and the stroke width both
#: scale with the dpi, so the census is taken at the scale the files have.
_SAVED_DPI = 160

#: A rejected cell must hold a stroke pixel inside the MIDDLE HALF of the cell (a
#: quarter of its width and height off on every side); an accepted cell must hold
#: none inside all but a tenth margin (the strokes are clipped to the boxes, so a
#: pixel of antialiasing at an edge is not a stroke inside the cell).
_REJECTED_MARGIN = 0.25
_ACCEPTED_MARGIN = 0.10


def _trained_like_mask(shape, seed, accepted_fraction=0.15):
    """A significance mask shaped like a trained map's: a few smooth islands of
    accepted cells in a field of rejected ones (single cells and thin strips
    included), about 15 % accepted as on the 2-D example (38.9 of 256 per tap)."""
    from scipy.ndimage import gaussian_filter

    field = np.abs(gaussian_filter(
        np.random.default_rng(seed).normal(size=shape), 1.2, mode="wrap"
    ))
    return field > np.quantile(field, 1.0 - accepted_fraction)


def _render_rgba(fig):
    fig.set_dpi(_SAVED_DPI)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).astype(int).copy()


def _painted_hatch_pixels(fig):
    """``(height, width)`` bool: the pixels the hatch paints dark at the saved scale.

    Found as the difference between the render with the hatch collections shown and
    the render with them hidden, so what the map underneath looks like cannot be
    mistaken for a stroke; a pixel counts only if it is dark (the rcParams hatch
    colour is black), which an invisible edge colour would not give.
    """
    shown = _render_rgba(fig)
    hidden = [
        artist for axis in fig.axes for artist in _hatched(axis)
    ]
    assert hidden, "no hatch artist on the figure"
    for artist in hidden:
        artist.set_visible(False)
    try:
        bare = _render_rgba(fig)
    finally:
        for artist in hidden:
            artist.set_visible(True)
    changed = np.abs(shown - bare).max(axis=2) > 0
    return changed & (shown[..., :3].max(axis=2) < 128)


def _stroke_census(fig, panels):
    """Count, over ``[(axis, significant), ...]``, the rejected cells without a stroke
    in their middle half and the accepted cells with a stroke anywhere inside.

    A cell ``(row, col)`` of an ``imshow`` axis is ``[col - 0.5, col + 0.5] x
    [row - 0.5, row + 0.5]`` in data coordinates; its pixel box comes from the axis'
    own transform at the saved dpi, so no layout is assumed.
    """
    painted = _painted_hatch_pixels(fig)
    height = painted.shape[0]
    census = dict(rejected=0, bare_rejected=0, accepted=0, stroked_accepted=0)
    for axis, significant in panels:
        for (row, col), passed in np.ndenumerate(np.asarray(significant, dtype=bool)):
            (x0, y0), (x1, y1) = axis.transData.transform(
                [(col - 0.5, row - 0.5), (col + 0.5, row + 0.5)]
            )
            left, right = sorted((x0, x1))
            top, bottom = sorted((height - y0, height - y1))
            margin = _ACCEPTED_MARGIN if passed else _REJECTED_MARGIN
            inside = painted[
                int(np.ceil(top + margin * (bottom - top))):
                int(np.floor(bottom - margin * (bottom - top))),
                int(np.ceil(left + margin * (right - left))):
                int(np.floor(right - margin * (right - left))),
            ]
            if passed:
                census["accepted"] += 1
                census["stroked_accepted"] += int(inside.any())
            else:
                census["rejected"] += 1
                census["bare_rejected"] += int(not inside.any())
    return census


def _two_dimensional_figure(monkeypatch, tmp_path):
    maps = {
        f"blocks.{i}.ffn/attn_pre": np.random.default_rng(i).normal(size=(16, 16))
        for i in range(4)
    }
    masks = [_trained_like_mask((16, 16), 10 + i) for i in range(4)]
    fig = _captured_figure(
        monkeypatch,
        lambda: plot_t_maps(
            maps, sig_grids=dict(zip(maps, masks)), arm="raw",
            condition_a="a", condition_b="b", output_path=str(tmp_path / "t.png"),
        ),
    )
    return fig, list(zip(_image_axes(fig), masks))


def _three_dimensional_figure(shape):
    def build(monkeypatch, tmp_path):
        return _three_dimensional_figure_of(shape, monkeypatch, tmp_path)

    return build


def _three_dimensional_figure_of(shape, monkeypatch, tmp_path):
    crosshair = (3, 5, 6)
    volume = _distinct_volume(shape, crosshair)
    significant = _trained_like_mask(shape, 4, accepted_fraction=0.25)
    significant[crosshair] = True
    fig = _captured_figure(
        monkeypatch,
        lambda: plot_t_maps(
            {"t": volume}, sig_grids={"t": significant}, arm="raw",
            condition_a="a", condition_b="b", output_path=str(tmp_path / "t3.png"),
        ),
    )
    planes = _image_axes(fig)
    assert len(planes) == 3
    return fig, [
        (axis, _expected_plane_masks(volume, significant, np.asarray(axis.images[0].get_array())))
        for axis in planes
    ]


def _old_pitch_hatch(axis, significant):
    """The step-4.6 drawer verbatim, with the old ``//`` pattern: the negative
    control that proves the census can fail."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    rejected = ~np.asarray(significant, dtype=bool)
    if rejected.any():
        axis.add_collection(
            PatchCollection(
                [Rectangle((c - 0.5, r - 0.5), 1.0, 1.0) for r, c in np.argwhere(rejected)],
                facecolor="none", edgecolor=matplotlib.rcParams["hatch.color"],
                linewidth=0.0, hatch="//",
            ),
            autolim=False,
        )


#: kind -> (figure builder, fewest rejected cells the fixture must contain). The
#: grids are the example runs' (16 x 16 and 7 x 7 x 7) and the 3-D paper grid
#: (10 x 10 x 10, where ``//`` already leaves bare cells).
_FIGURES = {
    "2d_16x16": (_two_dimensional_figure, 800),
    "3d_7x7x7": (_three_dimensional_figure((7, 7, 7)), 90),
    "3d_10x10x10": (_three_dimensional_figure((10, 10, 10)), 250),
}


class TestEveryRejectedCellCarriesStrokes:
    """RED on the ``//`` pattern (D-026): painted stroke pixels per cell, read from
    the render at the saved scale, for a trained-like 2-D figure and a 3-D figure."""

    @pytest.mark.parametrize("kind", list(_FIGURES))
    def test_every_rejected_cell_has_a_stroke_in_its_middle_half(
        self, monkeypatch, tmp_path, kind
    ):
        build, fewest = _FIGURES[kind]
        fig, panels = build(monkeypatch, tmp_path)
        try:
            census = _stroke_census(fig, panels)
        finally:
            plt.close(fig)
        assert census["rejected"] >= fewest, census
        assert census["bare_rejected"] == 0, census

    @pytest.mark.parametrize("kind", list(_FIGURES))
    def test_no_accepted_cell_holds_a_hatch_stroke(self, monkeypatch, tmp_path, kind):
        build, _ = _FIGURES[kind]
        fig, panels = build(monkeypatch, tmp_path)
        try:
            census = _stroke_census(fig, panels)
        finally:
            plt.close(fig)
        assert census["accepted"] >= 30, census
        assert census["stroked_accepted"] == 0, census

    @pytest.mark.parametrize(
        "kind, at_least", [("2d_16x16", 100), ("3d_10x10x10", 30)]
    )
    def test_the_census_can_fail_the_old_pattern_leaves_bare_cells(
        self, monkeypatch, tmp_path, kind, at_least
    ):
        """The same figures drawn with ``//``: many rejected cells hold no stroke
        (241 of 868 in 2-D, 62 of 262 in 3-D). Without this the tests above could
        pass for a census that sees nothing."""
        import train.topolm.plotting as module

        monkeypatch.setattr(module, "_hatch_rejected", _old_pitch_hatch)
        build, _ = _FIGURES[kind]
        fig, panels = build(monkeypatch, tmp_path)
        try:
            census = _stroke_census(fig, panels)
        finally:
            plt.close(fig)
        assert census["bare_rejected"] > at_least, census
        assert census["stroked_accepted"] == 0, census
