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
Ranks other than 2 and 3 are refused by name.

Layout
------
Both t-map figures are laid out in inches rather than fractions, so a real tap
path (``causal_language_model/topolm/block_0/attention_tap``) never overprints a
neighbour, the colour bar has an axes of its own clear of the panels, and the
headline and the line under it never collide. Panel titles drop the leading path
components every tap shares (:func:`_panel_tap_labels`) and the subtitle names
them once. The geometric tests in ``test_topolm_plotting.py`` hold all of this.
"""

from __future__ import annotations

import math
import os
import re
import textwrap
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt  # noqa: E402  (after the backend selection)
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from dl_techniques.utils.logger import logger

__all__ = [
    "PlotScaleWarning",
    "plot_t_maps",
    "plot_t_maps_3d",
    "plot_cluster_overlay",
    "plot_cluster_montage_3d",
    "plot_morans_i_profile",
    "plot_topography_curve",
    "last_row_per_step",
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
    """Hatch every cell of ``axis`` that FAILED BH-FDR; draw nothing if none did.

    ``significant`` is a boolean array of the shape of the image on ``axis``.
    Shared by the 2-D panel and every 3-D plane, so "hatched" means the same
    thing in both. The line under the headline says "hatched = not significant",
    so a cell without hatch must be a cell that passed: each rejected cell gets
    its own box, ``[col - 0.5, col + 0.5] x [row - 0.5, row + 0.5]`` in the
    coordinates ``imshow`` gives the same axis, all in ONE collection with
    no fill and a line width of 0, so no box outline is stroked and only the ``///``
    strokes show, and the t-values under them stay readable. The edge colour is
    the hatch colour and must NOT be ``"none"`` (matplotlib then draws no hatch at
    all: the strokes take the edge colour, the box outline takes the line width). Cluster edges, the one place a line is wanted, are
    drawn from the label grid, above this collection.

    The pattern is ``///`` and not ``//`` (D-026): the boxes are exact, but a
    stroke only shows where the pattern puts one, and at ``//`` the strokes are
    about two cells apart, so 241 of 868 rejected cells of a trained-like 2-D
    figure (the reviewer: 244 of 1737 on the real example) had no stroke in their
    middle half at the saved ``dpi=160``. At ``///`` every rejected cell of the
    example grids (16 x 16, 7 x 7 x 7, 10 x 10 x 10) holds one. The pitch is fixed
    in inches, not in cells: on a paper-size 28 x 28 panel (cells about 12 px) even
    ``///`` leaves 855 of 2664 rejected cells (32 %) without a stroke in their
    middle half, so there the box is still exact and the stroke is not a per-cell
    proof.

    Interface: ``axis`` has an image drawn on it; ``significant`` is a bool array
    of that image's shape (True = passed). Returns ``None``; it adds exactly one
    ``PatchCollection`` (zorder 1: above the image, below the outlines) when some
    cell was rejected and nothing otherwise.
    """
    # DECISION plan-2026-10-23T190557-972a9ac5/D-026: the denser ``///`` pattern, NOT
    # ``//``. Do NOT go back to ``//`` for a lighter look: its strokes are about two
    # cells apart, so a quarter of the rejected cells (241 of 868 on a trained-like
    # 16 x 16 figure at the saved dpi) carry no stroke in their middle half and read
    # as cells that passed. See decisions.md D-026; guarded by
    # TestEveryRejectedCellCarriesStrokes (a pixel census at dpi 160) and the
    # re-pinned hatch golden, which now hashes the pattern.
    # DECISION plan-2026-10-23T190557-972a9ac5/D-021: one box per rejected cell, NOT
    # a contour of the mask. The first drawer called ``contourf`` on a masked array
    # of ones, which fills only the quads whose four neighbouring cells are ALL
    # rejected: an isolated rejected cell and a one-cell-wide strip got no hatch,
    # 326 of 545 rejected cells (60 %) on the trained 2-D example, so the figure
    # contradicted its own subtitle. Do NOT go back to contourf (or any contour of
    # the mask) to save patches; a 3-D plane or a paper-size grid is a few hundred
    # boxes. See decisions.md D-021; guarded by TestTheHatchMarksEveryRejectedCell
    # and TestTheHatchMarksEveryRejectedVoxelOfEveryPlane.
    rejected = ~np.asarray(significant, dtype=bool)
    if not rejected.any():
        return
    boxes = [
        Rectangle((col - 0.5, row - 0.5), 1.0, 1.0)
        for row, col in np.argwhere(rejected)
    ]
    axis.add_collection(
        PatchCollection(
            boxes, facecolor="none", edgecolor=matplotlib.rcParams["hatch.color"],
            linewidth=0.0, hatch="///",
        ),
        autolim=False,
    )


#: How condition B's cluster outlines are drawn, next to A's solid black: dashed
#: magenta shows on both the red and the blue half of the colour scale, and a
#: second colour survives a grid so large that a dash pattern cannot (each outline
#: is one short segment per cell edge, and a dash pattern restarts on every one).
_B_OUTLINE = dict(color="magenta", linestyle="--")


def _overlay_flags(
    grid_shape: Tuple[int, ...],
    sig_grids: Optional[Mapping[str, np.ndarray]],
    label_grids: Optional[Mapping[str, np.ndarray]],
    label_grids_b: Optional[Mapping[str, np.ndarray]],
) -> Tuple[bool, bool, bool]:
    """What the panels will actually draw: ``(hatched, outlined_a, outlined_b)``.

    The line under the headline explains only what is on the figure, so it is
    derived from the same inputs the panels use: a hatch is drawn when some
    panel's mask (of the grid's shape) rejects at least one cell, and an outline
    when some label grid of the grid's shape holds a cluster.
    """
    def has_clusters(grids):
        return any(
            np.asarray(grid).shape == grid_shape and np.any(np.asarray(grid) > 0)
            for grid in (grids or {}).values()
        )

    hatched = any(
        np.asarray(mask).shape == grid_shape and not np.asarray(mask, dtype=bool).all()
        for mask in (sig_grids or {}).values()
    )
    return hatched, has_clusters(label_grids), has_clusters(label_grids_b)


def _overlay_legend(
    condition_a: Optional[str],
    condition_b: Optional[str],
    hatched: bool,
    outlined_a: bool,
    outlined_b: bool,
    b_given: bool,
) -> str:
    """The sentence saying what the hatch and the outlines on a t-map mean.

    Empty when the figure carries neither. When condition B's label grids were
    not handed over it says so, instead of leaving a reader to take the absence of
    a dashed outline for the absence of a cluster.
    """
    name_a, name_b = condition_a or "positive", condition_b or "negative"
    parts = []
    if hatched:
        parts.append("hatched = not significant after BH-FDR")
    if outlined_a or outlined_b:
        styles = (
            f"solid black = {name_a}, dashed magenta = {name_b}"
            if b_given
            else f"solid black = {name_a}; {name_b} clusters not drawn"
        )
        parts.append(f"outline = cluster of significant units ({styles})")
    return "; ".join(parts)


def _t_map_headline(
    arm: str,
    condition_a: Optional[str],
    condition_b: Optional[str],
    suptitle: Optional[str],
    note: Optional[str] = None,
    legend: str = "",
) -> Tuple[str, str]:
    """The figure title and the line under it, for the 2-D and 3-D t-map rows.

    ``legend`` (from :func:`_overlay_legend`) says what the hatch and the outlines
    mean and follows the sign convention; ``note`` (the live callback's step and
    stimulus set) is appended after it. With no conditions, no legend and no note
    the line is the one the figure always had.
    """
    if condition_a and condition_b:
        subtitle = f"positive = {condition_a}, negative = {condition_b}"
        if legend:
            subtitle = f"{subtitle}; {legend}"
    else:
        subtitle = legend or "hatched outline = rejected by BH-FDR"
    if note:
        subtitle = f"{subtitle} | {note}"
    return suptitle or f"TopoLM t-maps ({arm})", subtitle


def _prefix_note(shared_prefix: str) -> str:
    """The sentence naming the tap-path prefix the panel titles leave out (or ``""``)."""
    if not shared_prefix:
        return ""
    return f"; tap paths shown without the shared prefix {shared_prefix}"


#: Characters of 7 pt subtitle text per inch of figure width, deliberately a little
#: pessimistic (the measured average is about 17), so a wrapped line stays inside the
#: figure. The geometric tests in test_topolm_plotting.py are what hold it.
_SUBTITLE_CHARS_PER_INCH = 15.0


#: Where a subtitle may be broken first: the space after a ``;`` and the space after
#: the bar of `` | `` (never after the ``|`` of ``|t|``, which has no space before it).
_CLAUSE_BOUNDARY = re.compile(r"(?<=;) |(?<= \|) ")


def _wrap_subtitle(text: str, width_in: float) -> str:
    """Break the line under the headline so it fits a figure ``width_in`` wide.

    A one-row figure of two taps is about 5.6 in wide, and the subtitle (the
    conditions, the live callback's note, the dropped tap prefix) is wider than
    that. Lines break at clause boundaries first, after a ``;`` or the bar of
    `` | ``, so a phrase such as ``dashed magenta = b`` stays on one line; only a
    clause wider than the figure is broken at spaces. Lines are joined with
    ``"\n"``, and joining them with a space gives the text back; text that
    already fits is unchanged.
    """
    columns = max(20, int((width_in - 0.4) * _SUBTITLE_CHARS_PER_INCH))
    lines: List[str] = []
    for clause in _CLAUSE_BOUNDARY.split(text):
        if lines and len(lines[-1]) + 1 + len(clause) <= columns:
            lines[-1] += " " + clause
        elif len(clause) <= columns:
            lines.append(clause)
        else:
            lines.extend(textwrap.wrap(clause, width=columns, break_long_words=False))
    return "\n".join(lines) or text


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
    note: Optional[str] = None,
    label_grids_b: Optional[Dict[str, np.ndarray]] = None,
) -> Optional[str]:
    """Render one t-map per tapped layer, with significance and cluster overlays.

    A rank-3 input is drawn by :func:`plot_t_maps_3d` instead (three planes per
    tap); the arguments mean the same thing.

    :param t_maps: ``{tap_path: (height, width) t-values}`` (or 3-D, see above).
    :type t_maps: Mapping[str, numpy.ndarray]
    :param sig_grids: ``{tap_path: (height, width) bool}`` BH-FDR rejections, or
        ``None`` to omit the hatch overlay.
    :type sig_grids: Optional[Mapping[str, numpy.ndarray]]
    :param label_grids: ``{tap_path: (height, width) int}`` cluster labels of
        condition A (positive t), ``0`` unassigned, or ``None`` to omit the
        boundaries. Outlined solid black.
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
    :param note: Appended to the line under the title (the live callback's step
        and stimulus set); ``None`` adds nothing.
    :type note: Optional[str]
    :param label_grids_b: Same as ``label_grids`` for condition B (negative t),
        outlined dashed magenta. ``None`` draws no B outline and the line under
        the title says so when A's are drawn; the two grids never label the same
        cell, because a cell has one sign.
    :type label_grids_b: Optional[Mapping[str, numpy.ndarray]]
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
            note=note,
            label_grids_b=label_grids_b,
        )

    ordered: List[Tuple[str, np.ndarray]] = [
        (name, np.asarray(values, dtype="float64")) for name, values in t_maps.items()
    ]
    limit = _shared_limit(t_maps, limit_percentile)

    n_panels = len(ordered)
    per_row = max(1, min(int(max_panels_per_row), n_panels))
    n_rows = int(math.ceil(n_panels / per_row))

    # The layout is fixed in inches, like the 3-D one, and for the same reasons: a
    # figure sized by fractions put the colour bar on top of the last panel (and over
    # its title), let the headline and the line under it collide, and left no room
    # for a real tap path (``causal_language_model/topolm/block_0/attention_tap``) as
    # a title. Titles are the shared-prefix-free, wrapped labels the 3-D figure uses,
    # the dropped prefix is named once under the headline, and the colour bar has an
    # axes of its own to the right of the panels, a gap clear of them.
    tap_labels, shared_prefix, title_fontsize = _panel_tap_labels(
        [name for name, _ in ordered],
        max_chars=_T_MAP_TITLE_MAX_CHARS,
        fontsize=_T_MAP_TITLE_FONTSIZE,
    )
    title_lines = max(label.count("\n") + 1 for label in tap_labels.values())
    title_block = 0.12 * title_lines + 0.08
    cell, gap_x, gap_y = 2.1, 0.25, title_block + 0.12
    margin_left, margin_right, margin_bottom = 0.15, 1.05, 0.1
    grid_w = per_row * cell + (per_row - 1) * gap_x
    grid_h = n_rows * cell + (n_rows - 1) * gap_y
    fig_w = margin_left + grid_w + margin_right

    hatched, outlined_a, outlined_b = _overlay_flags(
        grid_shape, sig_grids, label_grids, label_grids_b
    )
    legend = _overlay_legend(
        condition_a, condition_b, hatched, outlined_a, outlined_b,
        b_given=label_grids_b is not None,
    )
    title, subtitle = _t_map_headline(
        arm, condition_a, condition_b, suptitle, note, legend
    )
    subtitle = _wrap_subtitle(subtitle + _prefix_note(shared_prefix), fig_w)
    subtitle_lines = subtitle.count("\n") + 1
    # From the top: a margin, the headline (10 pt), the subtitle (7 pt, any number of
    # lines), a gap, then the first row's titles above its panels.
    margin_top = 0.06 + 0.17 + 0.06 + 0.12 * subtitle_lines + 0.08 + title_block
    fig_h = margin_top + grid_h + margin_bottom

    fig, axes = plt.subplots(
        n_rows, per_row, figsize=(fig_w, fig_h), squeeze=False
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
        if label_grids_b is not None and name in label_grids_b:
            _draw_boundaries(axis, np.asarray(label_grids_b[name]), **_B_OUTLINE)
        axis.set_title(tap_labels[name], fontsize=title_fontsize, pad=3)
        axis.set_xticks([])
        axis.set_yticks([])

    if image is not None:
        bar_h = min(2.4, grid_h)
        bar_axis = fig.add_axes([
            (margin_left + grid_w + 0.22) / fig_w,
            (margin_bottom + (grid_h - bar_h) / 2) / fig_h,
            0.14 / fig_w,
            bar_h / fig_h,
        ])
        bar = fig.colorbar(image, cax=bar_axis)
        bar.set_label("t (A - B)", fontsize=8)
        bar.ax.tick_params(labelsize=7)

    fig.suptitle(title, fontsize=10, y=1.0 - 0.06 / fig_h, va="top")
    fig.text(
        0.5, 1.0 - 0.29 / fig_h, subtitle,
        ha="center", va="top", fontsize=7, alpha=0.75,
    )
    # NOT tight_layout: a figure carrying a colorbar has axes that layout cannot
    # reconcile, and matplotlib warns rather than fixing it. The margins are in
    # inches above, so nothing here needs it.
    fig.subplots_adjust(
        left=margin_left / fig_w,
        right=1.0 - margin_right / fig_w,
        top=1.0 - margin_top / fig_h,
        bottom=margin_bottom / fig_h,
        wspace=gap_x / cell,
        hspace=gap_y / cell,
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
# The 2-D map is 2.1 in wide and its title is 7 pt, about 3.9 pt a character, so 30
# characters (1.6 in) leave a margin for the neighbouring title; measured by the
# geometric test, like the 3-D constants above.
_T_MAP_TITLE_MAX_CHARS = 30
_T_MAP_TITLE_FONTSIZE = 7.0


def _panel_tap_labels(
    names: List[str],
    max_chars: int = _TITLE_MAX_CHARS,
    fontsize: float = _TITLE_FONTSIZE,
) -> Tuple[Dict[str, str], str, float]:
    """Shorten and wrap the tap paths used as panel titles (3-D planes and 2-D maps).

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
       ``max_chars`` characters, the ``/`` staying at the end of the
       line it closes. A name that fits is returned unchanged, so a short tap name
       is still one contiguous string in its title. A single component wider than
       the limit cannot be wrapped; the font is shrunk instead (floor
       :data:`_TITLE_MIN_FONTSIZE`).

    Joining a tap's lines (``"".join(label.split("\\n"))``) and prepending
    ``prefix`` gives the original path back, so no information is lost.

    :param names: Tap paths in figure order.
    :type names: List[str]
    :param max_chars: Widest title line in characters; the default is the 3-D
        plane's, the 2-D map passes its own (:data:`_T_MAP_TITLE_MAX_CHARS`).
    :type max_chars: int
    :param fontsize: The title font size, shrunk only for an unbreakable
        component wider than ``max_chars``.
    :type fontsize: float
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
            if lines and len(lines[-1]) + len(piece) <= max_chars:
                lines[-1] += piece
            else:
                lines.append(piece)
        labels[name] = "\n".join(lines)
        widest = max(widest, max(len(line) for line in lines))
    if widest > max_chars:
        fontsize = max(_TITLE_MIN_FONTSIZE, fontsize * max_chars / widest)
    return labels, prefix, fontsize


# A thin green dashed line vanishes on a saturated cell of either sign, so each
# crosshair line is drawn over a wider white stroke (a halo): the green keeps its
# colour on pale cells and the halo separates it from dark red and dark blue ones.
_CROSSHAIR_STYLE = dict(color="limegreen", linewidth=0.8, linestyle="--")
_CROSSHAIR_HALO_WIDTH = 2.2


def _crosshair_outline() -> list:
    """The path effects of one crosshair line: a white halo, then the line itself."""
    import matplotlib.patheffects as path_effects

    return [
        path_effects.Stroke(linewidth=_CROSSHAIR_HALO_WIDTH, foreground="white"),
        path_effects.Normal(),
    ]


def _tap_mask(
    sig_grids: Optional[Mapping[str, np.ndarray]], name: str, grid_shape: Tuple[int, ...]
) -> Optional[np.ndarray]:
    """The BH-FDR mask of tap ``name`` as booleans, or ``None`` if absent or of another shape."""
    if sig_grids is None or name not in sig_grids:
        return None
    mask = np.asarray(sig_grids[name], dtype=bool)
    return mask if mask.shape == grid_shape else None


def _crosshair_candidates(
    t_map: np.ndarray, significant: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """``(|t|, candidate mask, restricted)``: where a crosshair may sit.

    The candidates are the finite voxels that are significant when the mask has
    the map's shape and any of them survives, else every finite voxel; ``restricted``
    says which of the two it is. The ONE rule behind :func:`_crosshair_voxel` and the
    sentence under the headline (:func:`_crosshair_clause`), so the figure never
    says "significant" about a crosshair that is not on a significant voxel.
    """
    magnitude = np.abs(np.asarray(t_map, dtype="float64"))
    finite = np.isfinite(magnitude)
    if significant is not None and significant.shape == magnitude.shape:
        survivors = finite & significant
        if survivors.any():
            return magnitude, survivors, True
    return magnitude, finite, False


def _crosshair_clause(restricted: Sequence[bool], masks_given: bool) -> str:
    """The sentence naming where the dashed green crosshair sits, true for this figure.

    :param restricted: Per tap, whether its crosshair is restricted to significant
        voxels (:func:`_crosshair_candidates`).
    :param masks_given: Whether any tap had a usable significance mask.
    """
    where = "dashed green = crosshair at the largest"
    if restricted and all(restricted):
        return f"{where} significant |t|"
    if any(restricted):
        return f"{where} significant |t| (the largest |t| where none is significant)"
    if masks_given:
        return f"{where} |t| (nothing is significant)"
    return f"{where} |t|"


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
    magnitude, candidates, _ = _crosshair_candidates(t_map, significant)
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
    note: Optional[str] = None,
    label_grids_b: Optional[Dict[str, np.ndarray]] = None,
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
    :param label_grids: ``{tap_path: (n0, n1, n2) int}`` cluster labels of
        condition A, ``0`` unassigned, or ``None`` to omit the outlines. Outlined
        solid black on each plane.
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
    :param note: Appended to the line under the title (the live callback's step
        and stimulus set); ``None`` adds nothing.
    :type note: Optional[str]
    :param label_grids_b: Same as ``label_grids`` for condition B, outlined dashed
        magenta; see :func:`plot_t_maps`.
    :type label_grids_b: Optional[Mapping[str, numpy.ndarray]]
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
    # The line under the headline can be wider than a one-tap figure (5.2 in): wrap
    # it, and grow the top margin by the extra lines. One line changes nothing.
    hatched, outlined_a, outlined_b = _overlay_flags(
        grid_shape, sig_grids, label_grids, label_grids_b
    )
    legend = _overlay_legend(
        condition_a, condition_b, hatched, outlined_a, outlined_b,
        b_given=label_grids_b is not None,
    )
    title, subtitle = _t_map_headline(
        arm, condition_a, condition_b, suptitle, note, legend
    )
    masks = {
        name: _tap_mask(sig_grids, name, grid_shape) for name, _ in ordered
    }
    restricted = [
        _crosshair_candidates(values, masks[name])[2] for name, values in ordered
    ]
    legend_note = "; " + _crosshair_clause(
        restricted, masks_given=any(mask is not None for mask in masks.values())
    )
    legend_note += _prefix_note(shared_prefix)
    subtitle = _wrap_subtitle(subtitle + legend_note, fig_w)
    margin_top += 0.12 * subtitle.count("\n")
    fig_h = margin_top + n_rows * cell + (n_rows - 1) * gap_y + margin_bottom
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    flat_axes = [ax for row in axes for ax in row]
    for unused in flat_axes[3 * n_taps:]:
        unused.set_visible(False)

    image = None
    for tap_index, (name, values) in enumerate(ordered):
        mask = masks[name]
        labels = None
        if label_grids is not None and name in label_grids:
            labels = np.asarray(label_grids[name])
        labels_b = None
        if label_grids_b is not None and name in label_grids_b:
            labels_b = np.asarray(label_grids_b[name])
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
            if labels_b is not None and labels_b.shape == grid_shape:
                _draw_boundaries(axis, labels_b[index], **_B_OUTLINE)
            # The in-plane coordinates of the crosshair: the two axes that are
            # NOT the plane axis, in order (rows, then columns).
            row_axis, col_axis = [a for a in range(3) if a != plane_axis]
            for line in (
                axis.axhline(crosshair[row_axis], **_CROSSHAIR_STYLE),
                axis.axvline(crosshair[col_axis], **_CROSSHAIR_STYLE),
            ):
                line.set_path_effects(_crosshair_outline())
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

    fig.suptitle(title, fontsize=10, y=1.0 - 0.08 / fig_h, va="top")
    fig.text(
        0.5, 1.0 - 0.38 / fig_h,
        subtitle,
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


def _draw_boundaries(
    axis: plt.Axes,
    labels: np.ndarray,
    color: str = "black",
    linestyle: str = "-",
) -> None:
    """Outline each connected cluster in a ``label_grid``.

    ``color`` and ``linestyle`` tell condition A's outlines (the defaults, solid
    black) from condition B's (:data:`_B_OUTLINE`).

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
                color=color, linewidth=1.1, linestyle=linestyle,
            )
    for col in range(values.shape[1] + 1):
        left = padded[:, col]
        right = padded[:, col + 1]
        for row in np.where(left != right)[0]:
            axis.plot(
                [col - 0.5, col - 0.5], [row - 1.5, row - 0.5],
                color=color, linewidth=1.1, linestyle=linestyle,
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


def _cluster_title(title: str, n_shown: int, hidden: int) -> str:
    """``"<title>: 1 cluster"`` / ``"<title>: 3 clusters (+2 not coloured)"``.

    Shared by the 2-D overlay and the 3-D montage so the two read alike.

    :param title: The caller's title (the tap and the arm).
    :type title: str
    :param n_shown: Cluster ids coloured in the figure.
    :type n_shown: int
    :param hidden: Cluster ids beyond ``max_clusters_shown`` that are not coloured.
    :type hidden: int
    :return: The figure title.
    :rtype: str
    """
    label = f"{title}: {n_shown} cluster{'' if n_shown == 1 else 's'}"
    if hidden:
        label += f" (+{hidden} not coloured)"
    return label


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
    label = _cluster_title(title, shown.size, hidden)
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

    label = _cluster_title(title, shown.size, hidden)
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
    note: Optional[str] = None,
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
    :param note: A second title line (the live callback's step and stimulus
        set); ``None`` leaves the title as it was.
    :type note: Optional[str]
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
    # Real tap paths end in the same two names in every block, so the last component
    # alone (the old tick label) read ``attention_tap`` at every other tick. The
    # tick is the path without the prefix every tap shares, which is named once
    # under the axis; the wrapped title labels are joined back into one line.
    labels, prefix, _ = _panel_tap_labels(
        names, max_chars=_T_MAP_TITLE_MAX_CHARS, fontsize=_T_MAP_TITLE_FONTSIZE
    )
    xlabel = "layer (depth order)"
    if prefix:
        # Wrapped to the 5 in figure: the longest prefix (a one-block run drops the
        # block too) is wider than the figure at 9 pt on one line.
        xlabel += "\n" + _wrap_subtitle(
            f"tap paths shown without the shared prefix {prefix}", 5.0
        )
    axis.set_xlabel(xlabel, fontsize=8 if prefix else None)
    axis.set_ylabel("Moran's I (standard)")
    title = f"Topographic smoothness across depth ({arm})"
    axis.set_title(f"{title}\n{note}" if note else title, fontsize=9)
    step = max(1, len(names) // 12)
    axis.set_xticks(depth[::step])
    axis.set_xticklabels(
        ["".join(labels[names[i]].split("\n")) for i in range(0, len(names), step)],
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


def _series(rows: Sequence[Mapping[str, Any]], getter) -> np.ndarray:
    """One value per row, ``nan`` where the row has none (a ``null`` in the JSONL)."""
    values = []
    for row in rows:
        try:
            value = getter(row)
        except (KeyError, TypeError):
            value = None
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype="float64")


def last_row_per_step(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """``rows`` with one row per ``step``, repeated steps merged.

    The live callback writes a step more than once in two situations. After a
    weight restore it redraws the step (the second row carries
    ``"restored_weights": true`` and describes the model the run ended with). At
    the end of training it closes the file with a metrics-only row
    (``"val_only": true``: ``step``, ``val_loss``, ``val_loss_step``, no map
    statistics) for a validation that Keras ran after the last draw. Both give
    one entry here:

    - a row that carries map statistics (a ``per_tap`` entry) describes a draw
      completely and REPLACES what was kept for its step, nulls included: a
      non-finite statistic of the restored model must not be filled in from the
      pre-restore draw;
    - a row without them (the metrics-only row) is MERGED onto the kept row:
      its non-null fields override, its nulls and its absences leave the earlier
      value alone, so it cannot shadow the map statistics. The merged row is a
      full row, so ``val_only`` is dropped from it; a metrics-only row with
      nothing to merge onto stays as it is.

    Order is by first appearance of each step; a row without a ``step`` is kept
    as it is. The input rows are not modified.

    :param rows: Parsed ``metrics.jsonl`` rows, oldest first.
    :type rows: Sequence[Mapping[str, Any]]
    :return: The rows to plot or read.
    :rtype: List[Mapping[str, Any]]
    """
    position: Dict[Any, int] = {}
    kept: List[Mapping[str, Any]] = []
    for row in rows:
        step = row.get("step") if hasattr(row, "get") else None
        if step is None or step not in position:
            if step is not None:
                position[step] = len(kept)
            kept.append(row)
        elif "per_tap" in row:
            kept[position[step]] = row
        else:
            merged = dict(kept[position[step]])
            merged.update({key: value for key, value in row.items() if value is not None})
            if "per_tap" in merged:
                merged.pop("val_only", None)
            kept[position[step]] = merged
    return kept


def _validation_points(rows: Sequence[Mapping[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """``(steps, values)`` of the validations in ``rows``, each drawn once at its OWN step.

    A row's ``val_loss`` is the last one Keras reported, so consecutive rows
    (drawn every N steps while validation happens once per epoch) repeat the same
    value, and its ``val_loss_step`` is where that value was measured. Drawing it
    at the row's ``step`` instead shifts every point right by up to one cadence
    and shows a plateau that is not there. A row without ``val_loss_step`` (an
    older ``metrics.jsonl``) falls back to its ``step``. Rows without a finite
    ``val_loss`` contribute nothing; a repeated ``(step, value)`` pair is kept
    once, in order.

    :param rows: Parsed ``metrics.jsonl`` rows, oldest first.
    :type rows: Sequence[Mapping[str, Any]]
    :return: Two equal-length arrays, empty when no row carries a validation.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    """
    values = _series(rows, lambda row: row["val_loss"])
    own = _series(rows, lambda row: row["val_loss_step"])
    draw = _series(rows, lambda row: row["step"])
    where = np.where(np.isfinite(own), own, draw)
    seen = set()
    kept = []
    for index in np.flatnonzero(np.isfinite(values)):
        key = (float(where[index]), float(values[index]))
        if key not in seen:
            seen.add(key)
            kept.append(index)
    return where[kept], values[kept]


#: Markers and dash patterns the tap lines of the curve cycle through, so two lines
#: differ in more than a hue where they overlap.
_TAP_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "*")
_TAP_LINESTYLES = ("-", "--", "-.", ":")


def _tap_line_styles(n_taps: int) -> List[Dict[str, Any]]:
    """One ``{color, marker, linestyle}`` per tap line of the curve.

    Colours are CATEGORICAL, not a ramp: ``tab10`` for up to ten taps (nine
    near-neighbours of a sequential map cannot be told apart at a crossing) and
    ``tab20`` beyond. The marker advances with the tap and shifts one place each
    time the palette wraps, so no two of up to 200 taps share colour and marker;
    the dash pattern cycles on top. The legend is built from the same lines, so it
    always matches.

    :param n_taps: How many tap lines the figure has.
    :type n_taps: int
    :return: ``n_taps`` style dictionaries for ``Axes.plot``.
    :rtype: List[Dict[str, Any]]
    """
    palette = plt.get_cmap("tab10" if n_taps <= 10 else "tab20").colors
    return [
        {
            "color": palette[index % len(palette)],
            "marker": _TAP_MARKERS[(index + index // len(palette)) % len(_TAP_MARKERS)],
            "linestyle": _TAP_LINESTYLES[index % len(_TAP_LINESTYLES)],
        }
        for index in range(n_taps)
    ]


#: Taps up to which the curve's legend is one column in the larger font.
_LEGEND_SINGLE_COLUMN_MAX_TAPS = 8


def _curve_legend_width(
    entries: Sequence[str], title: str, columns: int, fontsize: float
) -> float:
    """Inches the curve's outside legend needs: a deliberately generous estimate.

    A glyph of the legend font averages about half its point size; 0.6 leaves a
    margin, and the handle, the gap between columns and the frame are added. The
    geometric test in test_topolm_plotting.py measures the rendered legend, so an
    estimate that is too small fails there instead of clipping a figure.
    """
    char_in = 0.6 * fontsize / 72.0
    column = 0.35 + max((len(entry) for entry in entries), default=0) * char_in
    body = columns * column + (columns - 1) * 0.15
    head = len(title) * char_in
    return max(body, head) + 0.45


def _tick_at_whole_numbers(axis: plt.Axes) -> None:
    """Tick the y axis of a counted quantity at integers only.

    ``MaxNLocator(integer=True)`` alone is not enough: matplotlib gives up the
    integer promise when the view spans less than one unit (no clusters yet, a
    mean of half a unit) and falls back to ticks like -0.04, 0.04, 0.08. So the
    view is widened to span at least 1.1 first, which puts both 0 and 1 in it.
    Call after the data is plotted: the view is read from it.
    """
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    low, high = axis.get_ylim()
    if high - low < 1.1:
        axis.set_ylim(low, low + 1.1)


def plot_topography_curve(
    rows: Sequence[Mapping[str, Any]],
    output_path: Optional[str] = None,
    note: Optional[str] = None,
) -> Optional[str]:
    """Plot how the topography forms: one point per live draw, against the step.

    The figure the live callback rewrites after every draw, from the rows of its
    ``metrics.jsonl`` (``null`` for a value that was not finite). Stacked panels
    share the step axis: Moran's I of every tap (thin, each its own categorical
    colour, marker and dash pattern, named in a legend to the right of the axes
    with the shared path prefix as its title) with the mean over taps in black;
    the clusters found for each polarity; the mean number of BH-FDR-significant
    units; and the validation loss when a row carries one,
    each validation once at the step it was measured (``val_loss_step``, see
    :func:`_validation_points`). A single row draws as markers, so the first
    frame is not an empty axis.

    :param rows: Parsed ``metrics.jsonl`` rows, oldest first; a step written twice
        (the redraw after a weight restore, the closing metrics-only row of the
        last validation) is drawn once, merged by :func:`last_row_per_step`; the
        validations are read from every row. Each has ``step``,
        ``mean_morans_i``, ``total_clusters_a``, ``total_clusters_b``,
        ``mean_num_significant_units``, ``val_loss`` (with ``val_loss_step``)
        and ``per_tap``
        (``{tap_path: {"morans_i": ...}}``); any value may be ``None``.
    :type rows: Sequence[Mapping[str, Any]]
    :param output_path: Where to write the PNG, or ``None`` to skip writing.
    :type output_path: Optional[str]
    :param note: Appended to the title (the stimulus set).
    :type note: Optional[str]
    :return: The written path, or ``None`` if ``output_path`` was ``None``.
    :rtype: Optional[str]
    :raises ValueError: If ``rows`` is empty.
    """
    if not rows:
        raise ValueError("cannot draw a topography curve from zero rows")
    # DECISION plan-2026-10-23T190557-972a9ac5/D-019: the validation points are read
    # from ALL the rows, not from the merged ones. Every validation is a measurement
    # whichever row repeats it, and merging the closing metrics-only row over the full
    # row of its step overwrites that row's older ``val_loss``, which can be the only
    # place a validation (measured one cadence earlier) is recorded. Do NOT move this
    # line below ``last_row_per_step``: the curve would silently lose a point.
    # See decisions.md D-019; guarded by test_the_curve_keeps_every_validation_even_
    # when_a_row_is_shadowed.
    val_steps, val_loss = _validation_points(rows)
    rows = last_row_per_step(rows)

    steps = _series(rows, lambda row: row["step"])
    panels = ["moran", "clusters", "units"] + (["val_loss"] if val_loss.size else [])
    taps = list(rows[0].get("per_tap", {}))
    # One legend entry per tap line, outside the axes on the right so it covers no
    # data, with the shared path prefix named once in its title. Many taps (24 at
    # paper depth) get two columns and a smaller font.
    tap_labels, tap_prefix, _ = _panel_tap_labels(taps, max_chars=10 ** 6)
    legend_columns = 1 if len(taps) <= _LEGEND_SINGLE_COLUMN_MAX_TAPS else 2
    legend_fontsize = 6.5 if legend_columns == 1 else 5.5
    legend_width = _curve_legend_width(
        [tap_labels[t] for t in taps] + ["mean over taps"], tap_prefix,
        legend_columns, legend_fontsize,
    )
    fig_width = 6.6 + legend_width
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(fig_width, 2.1 * len(panels) + 0.7),
        sharex=True, squeeze=False,
    )
    axes = [row[0] for row in axes]

    moran_axis = axes[0]
    tap_lines = []
    for style, tap in zip(_tap_line_styles(len(taps)), taps):
        (line,) = moran_axis.plot(
            steps,
            _series(rows, lambda row, tap=tap: row["per_tap"][tap]["morans_i"]),
            linewidth=1.0, alpha=0.85, markersize=3.2, label=tap_labels[tap],
            **style,
        )
        tap_lines.append(line)
    (mean_line,) = moran_axis.plot(
        steps, _series(rows, lambda row: row["mean_morans_i"]),
        color="black", linewidth=1.8, marker="o", markersize=3.5,
        label="mean over taps",
    )
    moran_axis.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    moran_axis.set_ylabel("Moran's I")
    legend = moran_axis.legend(
        handles=[mean_line, *tap_lines], loc="upper left",
        bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, ncol=legend_columns,
        fontsize=legend_fontsize, title=tap_prefix.rstrip("/") or None,
        title_fontsize=legend_fontsize, handlelength=1.6, columnspacing=1.0,
        labelspacing=0.35,
    )
    # The legend stays IN the layout: tight_layout reserves its measured width, and
    # ``bbox_inches="tight"`` at save time keeps it (a legend taken out of the
    # layout is cropped away from the saved PNG). The figure is already widened by
    # the estimate in ``_curve_legend_width``, so the axes keep about 6.6 in.

    cluster_axis = axes[1]
    cluster_axis.plot(
        steps, _series(rows, lambda row: row["total_clusters_a"]),
        marker="o", markersize=3, linewidth=1.2, label="clusters, condition A",
    )
    cluster_axis.plot(
        steps, _series(rows, lambda row: row["total_clusters_b"]),
        marker="s", markersize=3, linewidth=1.2, linestyle="--",
        label="clusters, condition B",
    )
    cluster_axis.set_ylabel("clusters (all taps)")
    cluster_axis.legend(fontsize=7, loc="best")

    axes[2].plot(
        steps, _series(rows, lambda row: row["mean_num_significant_units"]),
        marker="o", markersize=3, linewidth=1.2, color="tab:red",
    )
    axes[2].set_ylabel("significant units\n(mean per tap)")
    # The two count panels tick at whole numbers: with a handful of clusters or
    # units the default locator picks 2.5, 7.5, 12.5 for a quantity that is
    # counted. Moran's I and the loss are not counts and keep their own ticks.
    for count_axis in (cluster_axis, axes[2]):
        _tick_at_whole_numbers(count_axis)

    if "val_loss" in panels:
        axes[3].plot(val_steps, val_loss, marker="o", markersize=3, linewidth=1.2,
                     color="tab:green")
        axes[3].set_ylabel("val_loss\n(each validation)")

    for axis in axes:
        axis.tick_params(labelsize=7)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("optimizer step")
    title = "TopoLM topography while training"
    fig.suptitle(f"{title}\n{note}" if note else title, fontsize=9)
    fig.tight_layout()

    return _save_and_close(fig, output_path, "Topography curve")
