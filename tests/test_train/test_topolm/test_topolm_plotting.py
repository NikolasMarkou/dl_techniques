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
        """
        labels = np.zeros((GRID, GRID), dtype=int)
        labels[0:3, 0:3] = 1

        fig, axis = plt.subplots()
        _draw_boundaries(axis, labels)
        lines = axis.get_lines()
        plt.close(fig)

        assert lines, "no boundaries drawn at all"
        assert any(line.get_ydata()[0] == 0 for line in lines), (
            "no line on the top edge of a cluster that touches it"
        )
        assert any(line.get_xdata()[0] == 0 for line in lines), (
            "no line on the left edge of a cluster that touches it"
        )

    def test_a_single_cell_cluster_is_outlined_on_all_four_sides(self):
        labels = np.zeros((GRID, GRID), dtype=int)
        labels[2, 2] = 1

        fig, axis = plt.subplots()
        _draw_boundaries(axis, labels)
        lines = axis.get_lines()
        plt.close(fig)

        rows = {line.get_ydata()[0] for line in lines}
        cols = {line.get_xdata()[0] for line in lines}
        assert rows == {2.0, 3.0}, rows
        assert cols == {2.0, 3.0}, cols

    def test_an_interior_cluster_gets_only_its_own_border(self):
        """A hollow centre is outlined at the hole, not as a filled blob.

        Segments are drawn per CELL edge, so a 4x4 ring around a 2x2 hole has
        12 horizontal and 12 vertical edges: 4 along the outer top, 4 along the
        outer bottom, and 2 on each side of the hole. Asserting the counts per
        ROW is what makes this fail if the hole is drawn over or skipped -- a
        bare total of 24 would also match a cluster whose border was drawn twice.
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
            2.0: 4,
            3.0: 2,
            5.0: 2,
            6.0: 4,
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
