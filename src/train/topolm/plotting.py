"""
Topographic map rendering for :mod:`train.topolm`.

The paper's central figure is a row of 2D t-maps, one per tapped layer, with the
surviving clusters outlined. This module renders that figure from the arrays
:func:`~train.topolm.common.evaluate_topography` already computes.

Why a separate module
---------------------
:mod:`train.topolm.common` is the analysis pipeline and is importable without
matplotlib. Keeping the rendering here means importing the analysis does not
import a plotting stack, and the figure code can be tested on synthetic arrays
without building a model.

What is drawn, and what deliberately is not
-------------------------------------------
Each panel is the raw t-map -- the contrast statistic, unthresholded -- with two
overlays drawn ON TOP rather than baked into the colour:

- cluster boundaries, from the same ``label_grid`` the report counts, so a
  cluster in the figure is provably the cluster in the JSON;
- a hatch over cells that failed BH-FDR, so rejected-but-plausible cells are
  visible instead of silently absent.

The colour scale is SHARED across the whole row and diverging at zero, because
per-panel autoscaling is the single easiest way to make a topographic figure
lie: it renders noise and signal at identical contrast. A row whose panels
required different scales is not shown as one figure at all -- see
:func:`plot_t_maps`.

Per-panel colour limits are therefore checked, and the caller is told, rather
than the figure quietly rescaling.

Three-dimensional grids
-----------------------
A 3-D map cannot be shown whole in a plane, so :func:`plot_t_maps` hands a rank-3
input to :func:`plot_t_maps_3d`: three orthogonal planes per tap through one
crosshair voxel, drawn with the same colour scale, hatch and outlines as the 2-D
panel. A 3-D label grid goes to :func:`plot_cluster_montage_3d`, which shows every
slice of it. The crosshair is a PROPOSAL for where to look (the largest
significant ``|t|``), not a result: a cluster away from it is in the montage.
The 2-D path is untouched; ranks other than 2 and 3 are refused by name.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt  # noqa: E402  (after the backend selection)

from dl_techniques.utils.logger import logger

__all__ = [
    "PlotScaleWarning",
    "plot_t_maps",
    "plot_t_maps_3d",
    "plot_cluster_overlay",
    "plot_cluster_montage_3d",
    "plot_morans_i_profile",
]


class PlotScaleWarning(UserWarning):
    """Raised as a warning when a single row cannot share one colour scale."""


def _panel_shape(t_maps: Dict[str, np.ndarray]) -> Tuple[int, ...]:
    """The grid shape, from the first t-map: ``(height, width)`` or a 3-D shape."""
    if not t_maps:
        raise ValueError("cannot infer a grid shape from zero t-maps")
    first = np.asarray(next(iter(t_maps.values())))
    if first.ndim not in (2, 3):
        raise ValueError(
            f"t-maps must be 2-D (height, width) or 3-D (depth, height, width); "
            f"got shape {first.shape} for tap {next(iter(t_maps))!r}"
        )
    return tuple(int(extent) for extent in first.shape)


def _validated_grid_shape(t_maps: Dict[str, np.ndarray]) -> Tuple[int, ...]:
    """The one grid shape every map shares, or a ``ValueError`` naming the offender.

    :raises ValueError: If ``t_maps`` is empty, the first map is neither 2-D nor
        3-D, or a later map has a different shape.
    """
    if not t_maps:
        raise ValueError("t_maps is empty; there is nothing to plot")
    grid_shape = _panel_shape(t_maps)
    for name, values in t_maps.items():
        if tuple(np.asarray(values).shape) != grid_shape:
            raise ValueError(
                f"t-map for tap {name!r} has shape {np.asarray(values).shape}, "
                f"expected {grid_shape}; all panels must share one grid"
            )
    return grid_shape


def _shared_limit(t_maps: Dict[str, np.ndarray], percentile: float) -> float:
    """A colour limit that covers every panel, so no panel is clipped silently.

    Taken over the pooled absolute values at ``percentile`` rather than the
    maximum, because a single divergent outlier otherwise compresses the whole
    map into one colour.
    """
    pooled = np.concatenate(
        [np.abs(np.asarray(v, dtype="float64")).ravel() for v in t_maps.values()]
    )
    finite = pooled[np.isfinite(pooled)]
    if finite.size == 0:
        return 1.0
    limit = float(np.percentile(finite, percentile))
    return limit if limit > 0.0 else float(finite.max()) or 1.0


def _hatch_rejected(axis: plt.Axes, significant: np.ndarray) -> None:
    """Hatch the cells of ``axis`` that FAILED BH-FDR; draw nothing if none did.

    ``significant`` is a boolean array of the shape of the image on ``axis``.
    Shared by the 2-D panel and every 3-D plane, so "hatched" means the same
    thing in both.

    Hatching the REJECTED region, so "significant" is a visible property of the
    panel rather than an inference from colour.

    No contour LINES here. Contouring the mask at 0.5 traces every surviving
    cell, and at realistic BH-FDR densities that is a dense mesh over the whole
    panel that buries the t-values it was meant to annotate. The hatch alone
    carries the same information and leaves the colour readable. Cluster edges,
    the one place a line is wanted, are drawn from the label grid.

    MEASURED, and the direction is counter-intuitive: contourf FILLS the
    unmasked cells. So to hatch the REJECTED region the MASK argument is
    ``significant`` (the significant cells) -- masking ``~significant`` instead
    fills the survivors, which is the inverse of what the name suggests and
    silently marks the wrong cells.
    """
    if significant.all():
        return
    rejected_fill = np.ma.masked_where(significant, np.ones(significant.shape))
    axis.contourf(
        rejected_fill,
        levels=[0.5, 1.5],
        colors="none",
        hatches=["//"],
        alpha=0.0,
    )


def _t_map_headline(
    arm: str,
    condition_a: Optional[str],
    condition_b: Optional[str],
    suptitle: Optional[str],
) -> Tuple[str, str]:
    """The figure title and the line under it, for the 2-D and 3-D t-map rows."""
    if condition_a and condition_b:
        subtitle = f"positive = {condition_a}, negative = {condition_b}"
    else:
        subtitle = "hatched outline = rejected by BH-FDR"
    return suptitle or f"TopoLM t-maps ({arm})", subtitle


def _save_and_close(fig: plt.Figure, output_path: Optional[str], what: str) -> Optional[str]:
    """Write ``fig`` to ``output_path`` (creating its directory) and close it.

    ``output_path=None`` only closes the figure and returns ``None``.
    """
    if output_path is None:
        plt.close(fig)
        return None
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"{what} written to {output_path}")
    return output_path


def plot_t_maps(
    t_maps: Dict[str, np.ndarray],
    sig_grids: Optional[Dict[str, np.ndarray]] = None,
    label_grids: Optional[Dict[str, np.ndarray]] = None,
    arm: str = "raw",
    condition_a: Optional[str] = None,
    condition_b: Optional[str] = None,
    output_path: Optional[str] = None,
    max_panels_per_row: int = 6,
    limit_percentile: float = 99.0,
    suptitle: Optional[str] = None,
) -> Optional[str]:
    """Render one t-map per tapped layer, with significance and cluster overlays.

    A rank-3 input is drawn by :func:`plot_t_maps_3d` instead (three planes per
    tap); the arguments mean the same thing and the 2-D path is unchanged.

    :param t_maps: ``{tap_path: (height, width) t-values}`` (or 3-D, see above).
    :type t_maps: Mapping[str, numpy.ndarray]
    :param sig_grids: ``{tap_path: (height, width) bool}`` BH-FDR rejections, or
        ``None`` to omit the hatch overlay.
    :type sig_grids: Optional[Mapping[str, numpy.ndarray]]
    :param label_grids: ``{tap_path: (height, width) int}`` cluster labels, ``0``
        unassigned, or ``None`` to omit the boundaries.
    :type label_grids: Optional[Mapping[str, numpy.ndarray]]
    :param arm: ``"raw"`` or ``"readout"``, for the title.
    :type arm: str
    :param condition_a: Positive-polarity condition, named in the legend.
    :type condition_a: Optional[str]
    :param condition_b: Negative-polarity condition, named in the legend.
    :type condition_b: Optional[str]
    :param output_path: Where to write the PNG, or ``None`` to skip writing.
    :type output_path: Optional[str]
    :param max_panels_per_row: Panels per row; the figure wraps after this.
    :type max_panels_per_row: int
    :param limit_percentile: Pooled percentile for the shared colour limit.
    :type limit_percentile: float
    :param suptitle: Overrides the generated title.
    :type suptitle: Optional[str]
    :return: The written path, or ``None`` if ``output_path`` was ``None``.
    :rtype: Optional[str]
    :raises ValueError: If ``t_maps`` is empty, a map is neither 2-D nor 3-D, or
        the maps disagree in shape.
    """
    grid_shape = _validated_grid_shape(t_maps)
    if len(grid_shape) == 3:
        return plot_t_maps_3d(
            t_maps,
            sig_grids=sig_grids,
            label_grids=label_grids,
            arm=arm,
            condition_a=condition_a,
            condition_b=condition_b,
            output_path=output_path,
            max_panels_per_row=max_panels_per_row,
            limit_percentile=limit_percentile,
            suptitle=suptitle,
        )

    ordered: List[Tuple[str, np.ndarray]] = [
        (name, np.asarray(values, dtype="float64")) for name, values in t_maps.items()
    ]
    limit = _shared_limit(t_maps, limit_percentile)

    n_panels = len(ordered)
    per_row = max(1, min(int(max_panels_per_row), n_panels))
    n_rows = int(math.ceil(n_panels / per_row))
    height = 2.6 * n_rows + 1.1
    width = 2.6 * per_row + 0.8

    fig, axes = plt.subplots(
        n_rows, per_row, figsize=(width, height), squeeze=False
    )
    flat_axes = [ax for row in axes for ax in row]
    for unused in flat_axes[n_panels:]:
        unused.set_visible(False)

    image = None
    for axis, (name, values) in zip(flat_axes, ordered):
        image = axis.imshow(
            values,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
            origin="upper",
        )
        if sig_grids is not None and name in sig_grids:
            mask = np.asarray(sig_grids[name], dtype=bool)
            if mask.shape == grid_shape:
                _hatch_rejected(axis, mask)
        if label_grids is not None and name in label_grids:
            _draw_boundaries(axis, np.asarray(label_grids[name]))
        axis.set_title(name, fontsize=7)
        axis.set_xticks([])
        axis.set_yticks([])

    if image is not None:
        bar = fig.colorbar(image, ax=flat_axes[:n_panels], fraction=0.02, pad=0.02)
        bar.set_label("t (A - B)", fontsize=8)
        bar.ax.tick_params(labelsize=7)

    title, subtitle = _t_map_headline(arm, condition_a, condition_b, suptitle)
    fig.suptitle(title, fontsize=10)
    fig.text(0.5, 0.945, subtitle, ha="center", fontsize=7, alpha=0.75)
    # NOT tight_layout: a figure carrying a colorbar has axes that layout cannot
    # reconcile, and matplotlib warns rather than fixing it. savefig's
    # bbox_inches="tight" below does the job without the warning.
    fig.subplots_adjust(
        left=0.04, right=0.9, top=0.90, bottom=0.04, wspace=0.12, hspace=0.28
    )

    return _save_and_close(fig, output_path, "T-maps")


# Widest title line, in characters, that fits above one 3-D plane at the title
# font size: the planes are 1.3 in wide with 0.16 in between, and the 5 pt title
# averages about 2.4 pt a character, so 26 characters leave a margin on both sides
# for the neighbouring title. Measured on the rendered figure, not estimated; the
# geometric test in test_topolm_plotting.py is what holds it.
_TITLE_MAX_CHARS = 26
_TITLE_FONTSIZE = 5.0
_TITLE_MIN_FONTSIZE = 3.5


def _panel_tap_labels(names: List[str]) -> Tuple[Dict[str, str], str, float]:
    """Shorten and wrap the tap paths used as 3-D panel titles.

    Real tap paths (``causal_language_model/topolm/block_0/attention_tap``) are
    far wider than a 1.3 in plane, and the three planes of a tap sit side by side,
    so the unmodified path overprints the neighbouring titles. Two steps, in
    this order:

    1. With two or more taps, the leading ``/``-separated components that EVERY
       tap shares are dropped from the title and returned as ``prefix`` so the
       caller can print them once. At least the last component of each path is
       always kept, and a single tap keeps its whole path (there is nothing to
       tell it apart from).
    2. The remainder is wrapped at ``/`` into lines of at most
       :data:`_TITLE_MAX_CHARS` characters, the ``/`` staying at the end of the
       line it closes. A name that fits is returned unchanged, so a short tap name
       is still one contiguous string in its title. A single component wider than
       the limit cannot be wrapped; the font is shrunk instead (floor
       :data:`_TITLE_MIN_FONTSIZE`).

    Joining a tap's lines (``"".join(label.split("\\n"))``) and prepending
    ``prefix`` gives the original path back, so no information is lost.

    :param names: Tap paths in figure order.
    :type names: List[str]
    :return: ``(labels, prefix, fontsize)``: ``labels`` maps each tap path to its
        possibly multi-line title text, ``prefix`` is the dropped shared leading
        path including its trailing ``/`` (``""`` if nothing was dropped), and
        ``fontsize`` is the title font size for every panel.
    :rtype: Tuple[Dict[str, str], str, float]
    """
    split = {name: name.split("/") for name in names}
    shared = 0
    if len(names) > 1:
        shortest = min(len(parts) for parts in split.values())
        while shared < shortest - 1 and len({p[shared] for p in split.values()}) == 1:
            shared += 1
    prefix = "/".join(next(iter(split.values()))[:shared]) + "/" if shared else ""

    labels: Dict[str, str] = {}
    widest = 0
    for name, parts in split.items():
        lines: List[str] = []
        for index, part in enumerate(parts[shared:]):
            piece = part + ("/" if index < len(parts[shared:]) - 1 else "")
            if lines and len(lines[-1]) + len(piece) <= _TITLE_MAX_CHARS:
                lines[-1] += piece
            else:
                lines.append(piece)
        labels[name] = "\n".join(lines)
        widest = max(widest, max(len(line) for line in lines))
    fontsize = _TITLE_FONTSIZE
    if widest > _TITLE_MAX_CHARS:
        fontsize = max(_TITLE_MIN_FONTSIZE, _TITLE_FONTSIZE * _TITLE_MAX_CHARS / widest)
    return labels, prefix, fontsize


def _crosshair_voxel(t_map: np.ndarray, significant: Optional[np.ndarray]) -> Tuple[int, ...]:
    """The voxel of a 3-D map the three planes pass through.

    The voxel of the largest ``|t|`` among the significant, finite voxels; if
    none is significant, the largest finite ``|t|`` overall; a map with no finite
    value gets the origin. Ties go to the lowest flat (C-order) index, so the
    choice is a function of the arrays alone.

    The ``|t|`` is deliberate: the strongest effect of either sign is the thing
    worth looking at, and an argmax over signed ``t`` would only ever pick the
    positive polarity. The significance restriction is deliberate too: the
    largest ``|t|`` overall is often a lone voxel that failed FDR.
    """
    magnitude = np.abs(np.asarray(t_map, dtype="float64"))
    finite = np.isfinite(magnitude)
    candidates = finite
    if significant is not None and significant.shape == magnitude.shape:
        survivors = finite & significant
        if survivors.any():
            candidates = survivors
    if not candidates.any():
        return (0,) * magnitude.ndim
    score = np.where(candidates, magnitude, -np.inf)
    return tuple(int(i) for i in np.unravel_index(int(np.argmax(score)), score.shape))


def plot_t_maps_3d(
    t_maps: Dict[str, np.ndarray],
    sig_grids: Optional[Dict[str, np.ndarray]] = None,
    label_grids: Optional[Dict[str, np.ndarray]] = None,
    arm: str = "raw",
    condition_a: Optional[str] = None,
    condition_b: Optional[str] = None,
    output_path: Optional[str] = None,
    max_panels_per_row: int = 6,
    limit_percentile: float = 99.0,
    suptitle: Optional[str] = None,
) -> Optional[str]:
    """Render 3-D t-maps as three orthogonal planes per tapped layer.

    Each tap gets three panels in axis order: the plane ``axis0 = i`` of shape
    ``(n1, n2)``, the plane ``axis1 = j`` of shape ``(n0, n2)`` and the plane
    ``axis2 = k`` of shape ``(n0, n1)``, where ``(i, j, k)`` is the crosshair
    voxel (see :func:`_crosshair_voxel`), marked by a dashed line on every plane
    and named in each panel title. The colour scale is the pooled
    :func:`_shared_limit` over the WHOLE volumes, not over the planes shown, so a
    plane through a quiet region looks quiet. The hatch of rejected cells and the
    cluster outlines are the 2-D ones, applied to each plane.

    A row holds ``max_panels_per_row // 3`` taps (at least one), two with the
    default of 6. Only three planes of each volume are visible, so a cluster away
    from the crosshair is NOT in this figure; :func:`plot_cluster_montage_3d`
    shows every slice of one tap's clusters.

    :param t_maps: ``{tap_path: (n0, n1, n2) t-values}``.
    :type t_maps: Mapping[str, numpy.ndarray]
    :param sig_grids: ``{tap_path: (n0, n1, n2) bool}`` BH-FDR rejections, or
        ``None`` to omit the hatch and to place the crosshair on the largest
        ``|t|`` overall.
    :type sig_grids: Optional[Mapping[str, numpy.ndarray]]
    :param label_grids: ``{tap_path: (n0, n1, n2) int}`` cluster labels, ``0``
        unassigned, or ``None`` to omit the outlines.
    :type label_grids: Optional[Mapping[str, numpy.ndarray]]
    :param arm: ``"raw"`` or ``"readout"``, for the title.
    :type arm: str
    :param condition_a: Positive-polarity condition, named in the legend.
    :type condition_a: Optional[str]
    :param condition_b: Negative-polarity condition, named in the legend.
    :type condition_b: Optional[str]
    :param output_path: Where to write the PNG, or ``None`` to skip writing.
    :type output_path: Optional[str]
    :param max_panels_per_row: Panels per row, three per tap.
    :type max_panels_per_row: int
    :param limit_percentile: Pooled percentile for the shared colour limit.
    :type limit_percentile: float
    :param suptitle: Overrides the generated title.
    :type suptitle: Optional[str]
    :return: The written path, or ``None`` if ``output_path`` was ``None``.
    :rtype: Optional[str]
    :raises ValueError: If ``t_maps`` is empty, a map is not 3-D, or the maps
        disagree in shape.
    """
    grid_shape = _validated_grid_shape(t_maps)
    if len(grid_shape) != 3:
        raise ValueError(
            f"plot_t_maps_3d needs 3-D (depth, height, width) t-maps; got shape "
            f"{grid_shape} for tap {next(iter(t_maps))!r}"
        )

    ordered: List[Tuple[str, np.ndarray]] = [
        (name, np.asarray(values, dtype="float64")) for name, values in t_maps.items()
    ]
    limit = _shared_limit(t_maps, limit_percentile)

    n_taps = len(ordered)
    taps_per_row = max(1, min(int(max_panels_per_row) // 3, n_taps))
    n_rows = int(math.ceil(n_taps / taps_per_row))
    # The layout is fixed in inches, because the planes are about 10 voxels across
    # and drawn square: a figure sized by a fraction would leave a gap between rows
    # that grows with the number of rows, and put the headline on the title line.
    tap_labels, shared_prefix, title_fontsize = _panel_tap_labels([n for n, _ in ordered])
    # A wrapped title is taller than the two lines the gaps were sized for: grow
    # the gap between rows and the top margin by the extra lines, so a wrapped
    # title never reaches the row above or the subtitle.
    extra_lines = max(label.count("\n") for label in tap_labels.values())
    extra = 0.08 * extra_lines
    cell, gap_x, gap_y = 1.3, 0.16, 0.5 + extra
    n_cols = 3 * taps_per_row
    margin_left, margin_right, margin_top, margin_bottom = 0.1, 0.9, 0.85 + extra, 0.1
    fig_w = margin_left + n_cols * cell + (n_cols - 1) * gap_x + margin_right
    fig_h = margin_top + n_rows * cell + (n_rows - 1) * gap_y + margin_bottom
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    flat_axes = [ax for row in axes for ax in row]
    for unused in flat_axes[3 * n_taps:]:
        unused.set_visible(False)

    image = None
    for tap_index, (name, values) in enumerate(ordered):
        mask = None
        if sig_grids is not None and name in sig_grids:
            mask = np.asarray(sig_grids[name], dtype=bool)
            if mask.shape != grid_shape:
                mask = None
        labels = None
        if label_grids is not None and name in label_grids:
            labels = np.asarray(label_grids[name])
        crosshair = _crosshair_voxel(values, mask)

        for plane_axis in range(3):
            axis = flat_axes[3 * tap_index + plane_axis]
            index = [slice(None)] * 3
            index[plane_axis] = crosshair[plane_axis]
            index = tuple(index)
            image = axis.imshow(
                values[index],
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
                origin="upper",
            )
            if mask is not None:
                _hatch_rejected(axis, mask[index])
            if labels is not None and labels.shape == grid_shape:
                _draw_boundaries(axis, labels[index])
            # The in-plane coordinates of the crosshair: the two axes that are
            # NOT the plane axis, in order (rows, then columns).
            row_axis, col_axis = [a for a in range(3) if a != plane_axis]
            axis.axhline(crosshair[row_axis], color="limegreen", linewidth=0.5,
                         linestyle="--")
            axis.axvline(crosshair[col_axis], color="limegreen", linewidth=0.5,
                         linestyle="--")
            axis.set_title(
                f"{tap_labels[name]}\naxis {plane_axis} = {crosshair[plane_axis]}  "
                f"@{crosshair}",
                fontsize=title_fontsize,
                pad=2,
            )
            axis.set_xticks([])
            axis.set_yticks([])

    if image is not None:
        bar_height = min(2.0, 0.6 * fig_h) / fig_h
        bar_axis = fig.add_axes(
            [1.0 - 0.62 / fig_w, 0.5 - bar_height / 2, 0.14 / fig_w, bar_height]
        )
        bar = fig.colorbar(image, cax=bar_axis)
        bar.set_label("t (A - B)", fontsize=7)
        bar.ax.tick_params(labelsize=6)

    title, subtitle = _t_map_headline(arm, condition_a, condition_b, suptitle)
    fig.suptitle(title, fontsize=10, y=1.0 - 0.08 / fig_h, va="top")
    note = "; dashed green = crosshair at the largest significant |t|"
    if shared_prefix:
        note += f"; tap paths shown without the shared prefix {shared_prefix}"
    fig.text(
        0.5, 1.0 - 0.38 / fig_h,
        subtitle + note,
        ha="center", va="top", fontsize=7, alpha=0.75,
    )
    # No tight_layout, for the reason given in plot_t_maps.
    fig.subplots_adjust(
        left=margin_left / fig_w,
        right=1.0 - margin_right / fig_w,
        top=1.0 - margin_top / fig_h,
        bottom=margin_bottom / fig_h,
        wspace=gap_x / cell,
        hspace=gap_y / cell,
    )
    return _save_and_close(fig, output_path, "3-D t-maps")


def _draw_boundaries(axis: plt.Axes, labels: np.ndarray) -> None:
    """Outline each connected cluster in a ``label_grid``.

    Boundaries are traced between cells whose labels differ, so a cluster's edge
    is its own -- not the bounding box of the coloured cells underneath.

    The segments are in the coordinates ``imshow`` gives the same axis: cell
    ``(row, col)`` spans ``x`` in ``[col - 0.5, col + 0.5]`` and ``y`` in
    ``[row - 0.5, row + 0.5]``, so the outline lies ON the cells it encloses.
    """
    values = np.asarray(labels)
    if values.ndim != 2 or not np.any(values > 0):
        return
    padded = np.zeros((values.shape[0] + 2, values.shape[1] + 2), dtype=values.dtype)
    padded[1:-1, 1:-1] = values

    # DECISION plan-2026-10-23T190557-972a9ac5/D-002: the zero frame shifts every
    # index by one, and imshow's cell centres sit on integers, so a padded index p
    # is the image coordinate p - 1 and an edge between padded lines p and p + 1 is
    # at p - 0.5. Do NOT draw at the raw padded integers (the previous code did,
    # [col, col + 1] at y = row): that hangs the outline one cell off to the right
    # and the bottom of the cells it encloses, off the panel for a cluster on the
    # last row or column. Guarded by TestClusterBoundariesSitOnTheirCells, which
    # compares the drawn segments with an edge set derived from cell geometry.
    #
    # The zero frame is what makes this correct at the edges: padded[r] is the
    # cell ABOVE the boundary at row r, and the frame row stands in for "outside
    # the grid", so a cluster touching the top edge gets a closed outline.
    for row in range(values.shape[0] + 1):
        above = padded[row]
        below = padded[row + 1]
        for col in np.where(above != below)[0]:
            axis.plot(
                [col - 1.5, col - 0.5], [row - 0.5, row - 0.5],
                color="black", linewidth=1.1,
            )
    for col in range(values.shape[1] + 1):
        left = padded[:, col]
        right = padded[:, col + 1]
        for row in np.where(left != right)[0]:
            axis.plot(
                [col - 0.5, col - 0.5], [row - 1.5, row - 0.5],
                color="black", linewidth=1.1,
            )


def _cluster_palette(
    values: np.ndarray, max_clusters_shown: int
) -> Tuple[np.ndarray, int, List[object]]:
    """The cluster ids to colour, how many are left out, and the colour list.

    The colour list starts with white for "unassigned" and has one entry per
    shown id. Shared by the 2-D overlay and the 3-D montage so both truncate and
    colour the same way.
    """
    present = np.unique(values[values > 0])
    shown = present[:max_clusters_shown]
    hidden = int(present.size - shown.size)
    # A single cluster still needs at least one colour, or ListedColormap raises
    # on an empty list. Its id is arbitrary, so its hue is too.
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, max(1, len(shown))))
    colours = ["white", *np.asarray(palette).tolist()][: len(shown) + 1]
    return shown, hidden, colours


def plot_cluster_overlay(
    label_grid: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "clusters",
    max_clusters_shown: int = 12,
) -> Optional[str]:
    """Render one ``label_grid`` as a categorical cluster map.

    Separate from :func:`plot_t_maps` because the categorical colour scale must
    NOT be shared with a t-map: cluster ids are arbitrary labels, and assigning
    them t-map colours would imply an ordering that does not exist. This function
    colours by id and shows at most ``max_clusters_shown`` ids, then reports how
    many were hidden rather than silently truncating the legend.

    A rank-3 grid is drawn by :func:`plot_cluster_montage_3d` instead.

    :param label_grid: ``(height, width)`` integer cluster labels, ``0``
        unassigned (or a 3-D grid, see above).
    :type label_grid: numpy.ndarray
    :param output_path: Where to write the PNG, or ``None`` to skip writing.
    :type output_path: Optional[str]
    :param title: Figure title.
    :type title: str
    :param max_clusters_shown: Distinct ids to colour before truncating.
    :type max_clusters_shown: int
    :return: The written path, or ``None`` if ``output_path`` was ``None``.
    :rtype: Optional[str]
    :raises ValueError: If ``label_grid`` is neither 2-D nor 3-D.
    """
    values = np.asarray(label_grid)
    if values.ndim == 3:
        return plot_cluster_montage_3d(
            values, output_path=output_path, title=title,
            max_clusters_shown=max_clusters_shown,
        )
    if values.ndim != 2:
        raise ValueError(
            f"label_grid must be 2-D (or 3-D, drawn as a montage); got shape "
            f"{values.shape}"
        )

    shown, hidden, colours = _cluster_palette(values, max_clusters_shown)
    shown_mask = np.isin(values, shown)
    from matplotlib.colors import ListedColormap

    fig, axis = plt.subplots(figsize=(3.4, 3.2))
    axis.imshow(
        np.where(shown_mask, values, 0),
        cmap=ListedColormap(colours),
        interpolation="nearest",
        origin="upper",
    )
    _draw_boundaries(axis, values)
    label = f"{title}: {shown.size} clusters"
    if hidden:
        label += f" (+{hidden} not coloured)"
    axis.set_title(label, fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])
    fig.tight_layout()
    return _save_and_close(fig, output_path, "Cluster overlay")


def plot_cluster_montage_3d(
    label_grid: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "clusters",
    max_clusters_shown: int = 12,
) -> Optional[str]:
    """Render a 3-D ``label_grid`` as one categorical panel per slice along axis 0.

    The 3-D counterpart of :func:`plot_cluster_overlay`, with the same palette,
    the same truncation to ``max_clusters_shown`` ids with the rest NAMED in the
    title, and the same outlines. Every slice is shown, so unlike the t-map planes
    it leaves no cluster out. A cluster id keeps ONE colour in every panel: the
    ids are coded by their rank among the shown ids before drawing, because the
    colour of a raw label would otherwise depend on which labels the slice
    happens to contain.

    :param label_grid: ``(n0, n1, n2)`` integer cluster labels, ``0`` unassigned.
    :type label_grid: numpy.ndarray
    :param output_path: Where to write the PNG, or ``None`` to skip writing.
    :type output_path: Optional[str]
    :param title: Figure title.
    :type title: str
    :param max_clusters_shown: Distinct ids to colour before truncating.
    :type max_clusters_shown: int
    :return: The written path, or ``None`` if ``output_path`` was ``None``.
    :rtype: Optional[str]
    :raises ValueError: If ``label_grid`` is not 3-D.
    """
    values = np.asarray(label_grid)
    if values.ndim != 3:
        raise ValueError(f"label_grid must be 3-D for a montage; got shape {values.shape}")

    shown, hidden, colours = _cluster_palette(values, max_clusters_shown)
    # Rank code: 0 = unassigned or not coloured, k = the k-th shown id.
    coded = np.where(
        np.isin(values, shown), np.searchsorted(shown, values) + 1, 0
    )
    from matplotlib.colors import ListedColormap

    n_slices = values.shape[0]
    n_cols = min(n_slices, 6)
    n_rows = int(math.ceil(n_slices / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.5 * n_cols + 0.4, 1.5 * n_rows + 0.9),
        squeeze=False,
    )
    flat_axes = [ax for row in axes for ax in row]
    for unused in flat_axes[n_slices:]:
        unused.set_visible(False)

    for index, axis in enumerate(flat_axes[:n_slices]):
        axis.imshow(
            coded[index],
            cmap=ListedColormap(colours),
            vmin=0,
            vmax=max(1, len(shown)),
            interpolation="nearest",
            origin="upper",
        )
        _draw_boundaries(axis, values[index])
        axis.set_title(f"axis 0 = {index}", fontsize=6)
        axis.set_xticks([])
        axis.set_yticks([])

    label = f"{title}: {shown.size} clusters"
    if hidden:
        label += f" (+{hidden} not coloured)"
    fig.suptitle(label, fontsize=8)
    fig.subplots_adjust(
        left=0.02, right=0.98, top=0.86 if n_rows > 1 else 0.78, bottom=0.02,
        wspace=0.12, hspace=0.35,
    )
    return _save_and_close(fig, output_path, "Cluster montage")


def plot_morans_i_profile(
    morans_by_tap: Dict[str, float],
    output_path: Optional[str] = None,
    arm: str = "raw",
) -> Optional[str]:
    """Plot standard Moran's I against layer depth.

    Depth is taken from the position in the mapping, not parsed out of the tap
    path, so a renamed or nested path cannot silently reorder the x-axis.

    :param morans_by_tap: ``{tap_path: I}`` in depth order.
    :type morans_by_tap: Mapping[str, float]
    :param output_path: Where to write the PNG, or ``None`` to skip writing.
    :type output_path: Optional[str]
    :param arm: ``"raw"`` or ``"readout"``, for the title.
    :type arm: str
    :return: The written path, or ``None`` if ``output_path`` was ``None``.
    :rtype: Optional[str]
    """
    names = list(morans_by_tap)
    values = np.array(
        [float(morans_by_tap[name]) for name in names], dtype="float64"
    )
    fig, axis = plt.subplots(figsize=(5.0, 3.0))
    depth = np.arange(len(values))
    axis.plot(depth, values, marker="o", linewidth=1.2, markersize=3)
    axis.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    axis.set_xlabel("layer (depth order)")
    axis.set_ylabel("Moran's I (standard)")
    axis.set_title(f"Topographic smoothness across depth ({arm})", fontsize=9)
    step = max(1, len(names) // 12)
    axis.set_xticks(depth[::step])
    axis.set_xticklabels(
        [names[i].split("/")[-1] for i in range(0, len(names), step)],
        rotation=90,
        fontsize=6,
    )
    fig.tight_layout()

    if output_path is None:
        plt.close(fig)
        return None
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Moran's I profile written to {output_path}")
    return output_path
