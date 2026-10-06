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
    "plot_cluster_overlay",
    "plot_morans_i_profile",
]


class PlotScaleWarning(UserWarning):
    """Raised as a warning when a single row cannot share one colour scale."""


def _panel_shape(t_maps: Dict[str, np.ndarray]) -> Tuple[int, int]:
    """The grid shape, from the first t-map."""
    if not t_maps:
        raise ValueError("cannot infer a grid shape from zero t-maps")
    first = np.asarray(next(iter(t_maps.values())))
    if first.ndim != 2:
        raise ValueError(
            f"t-maps must be 2-D (height, width); got shape {first.shape} for "
            f"tap {next(iter(t_maps))!r}"
        )
    return int(first.shape[0]), int(first.shape[1])


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

    :param t_maps: ``{tap_path: (height, width) t-values}``.
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
    :raises ValueError: If ``t_maps`` is empty, a map is not 2-D, or the maps
        disagree in shape.
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
            if mask.shape == grid_shape and not mask.all():
                # Hatch the REJECTED region, so "significant" is a visible
                # property of the panel rather than an inference from colour.
                #
                # No contour LINES here. Contouring the mask at 0.5 traces every
                # surviving cell, and at realistic BH-FDR densities that is a
                # dense mesh over the whole panel that buries the t-values it
                # was meant to annotate. The hatch alone carries the same
                # information and leaves the colour readable. Cluster edges, the
                # one place a line is wanted, are drawn from the label grid.
                # MEASURED, and the direction is counter-intuitive: contourf FILLS the
                # unmasked cells. So to hatch the REJECTED region the MASK
                # argument is `mask` (the significant cells) -- masking
                # `~mask` instead fills the survivors, which is the inverse of
                # what the name suggests and silently marks the wrong cells.
                rejected_fill = np.ma.masked_where(mask, np.ones(grid_shape))
                axis.contourf(
                    rejected_fill,
                    levels=[0.5, 1.5],
                    colors="none",
                    hatches=["//"],
                    alpha=0.0,
                )
        if label_grids is not None and name in label_grids:
            _draw_boundaries(axis, np.asarray(label_grids[name]))
        axis.set_title(name, fontsize=7)
        axis.set_xticks([])
        axis.set_yticks([])

    if image is not None:
        bar = fig.colorbar(image, ax=flat_axes[:n_panels], fraction=0.02, pad=0.02)
        bar.set_label("t (A - B)", fontsize=8)
        bar.ax.tick_params(labelsize=7)

    if condition_a and condition_b:
        subtitle = f"positive = {condition_a}, negative = {condition_b}"
    else:
        subtitle = "hatched outline = rejected by BH-FDR"

    title = suptitle or f"TopoLM t-maps ({arm})"
    fig.suptitle(title, fontsize=10)
    fig.text(0.5, 0.945, subtitle, ha="center", fontsize=7, alpha=0.75)
    # NOT tight_layout: a figure carrying a colorbar has axes that layout cannot
    # reconcile, and matplotlib warns rather than fixing it. savefig's
    # bbox_inches="tight" below does the job without the warning.
    fig.subplots_adjust(
        left=0.04, right=0.9, top=0.90, bottom=0.04, wspace=0.12, hspace=0.28
    )

    if output_path is None:
        plt.close(fig)
        return None

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"T-maps written to {output_path}")
    return output_path


def _draw_boundaries(axis: plt.Axes, labels: np.ndarray) -> None:
    """Outline each connected cluster in a ``label_grid``.

    Boundaries are traced between cells whose labels differ, so a cluster's edge
    is its own -- not the bounding box of the coloured cells underneath.
    """
    values = np.asarray(labels)
    if values.ndim != 2 or not np.any(values > 0):
        return
    padded = np.zeros((values.shape[0] + 2, values.shape[1] + 2), dtype=values.dtype)
    padded[1:-1, 1:-1] = values

    # The zero frame is what makes this correct at the edges: padded[r] is the
    # cell ABOVE the boundary at row r, and the frame row stands in for "outside
    # the grid", so a cluster touching the top edge gets a closed outline.
    for row in range(values.shape[0] + 1):
        above = padded[row]
        below = padded[row + 1]
        for col in np.where(above != below)[0]:
            axis.plot([col, col + 1], [row, row], color="black", linewidth=1.1)
    for col in range(values.shape[1] + 1):
        left = padded[:, col]
        right = padded[:, col + 1]
        for row in np.where(left != right)[0]:
            axis.plot([col, col], [row, row + 1], color="black", linewidth=1.1)


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

    :param label_grid: ``(height, width)`` integer cluster labels, ``0``
        unassigned.
    :type label_grid: numpy.ndarray
    :param output_path: Where to write the PNG, or ``None`` to skip writing.
    :type output_path: Optional[str]
    :param title: Figure title.
    :type title: str
    :param max_clusters_shown: Distinct ids to colour before truncating.
    :type max_clusters_shown: int
    :return: The written path, or ``None`` if ``output_path`` was ``None``.
    :rtype: Optional[str]
    :raises ValueError: If ``label_grid`` is not 2-D.
    """
    values = np.asarray(label_grid)
    if values.ndim != 2:
        raise ValueError(f"label_grid must be 2-D; got shape {values.shape}")

    present = np.unique(values[values > 0])
    shown = present[:max_clusters_shown]
    hidden = int(present.size - shown.size)

    shown_mask = np.isin(values, shown)
    from matplotlib.colors import ListedColormap

    # A single cluster still needs at least one colour, or ListedColormap raises
    # on an empty list. Its id is arbitrary, so its hue is too.
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, max(1, len(shown))))
    colours = ["white", *np.asarray(palette).tolist()][: len(shown) + 1]

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

    if output_path is None:
        plt.close(fig)
        return None
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Cluster overlay written to {output_path}")
    return output_path


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
