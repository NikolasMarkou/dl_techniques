"""
Figures and the per-epoch grid callback of the ConvUNext segmentation trainer.

Pure plotting plus one Keras callback; nothing here imports the trainer, so every function is
testable from hand-built arrays. Figures are drawn on ``matplotlib.figure.Figure`` objects
that own no pyplot state (no global figure registry to leak, no ``matplotlib.use`` at import,
so the module is safe under any ``MPLBACKEND``); ``Figure.savefig`` renders with Agg.

Contents (every ``plot_*`` / ``write_*`` function RAISES on malformed input: the trainer
decides that a missing figure is not fatal, not the function):

- :func:`class_scores` / :func:`segmentation_report` / :func:`write_segmentation_report` --
  per-class IoU, Dice, precision, recall and support from a confusion matrix, strict JSON.
- :func:`extended_palette` -- the colour table for any class count (step 29 of ``train.api``).
- :func:`plot_segmentation_grid` -- image | ground truth | prediction, one row per sample.
- :func:`plot_best_vs_final_predictions` -- image | ground truth | best | final.
- :func:`plot_per_class_scores` -- grouped bars of the four per-class scores.
- :func:`plot_miou_curve` -- train and validation mIoU per epoch with the best epoch marked.
- :class:`SegmentationGridCallback` -- the per-epoch grid on a fixed seeded validation batch.

Many classes and void (``train.api``'s ``image_segmentation`` task, D-079). The drawing functions take two
optional arguments, ``palette`` (default :data:`PALETTE`, the three colours of the Oxford-IIIT Pet trainer) and
``ignore_index`` (default ``None``). With the defaults every function is what it was: three classes at most, no void,
the same pixels. ``extended_palette(C)`` gives ``C`` colours whose first three rows are :data:`PALETTE`; a void pixel
is drawn in :data:`VOID_COLOR`, listed in the legend as :data:`VOID_NAME`, and is counted in no row's pixel accuracy
(a row with no valid pixel has ``nan``, titled ``n/a``). A legend of more than :data:`MAX_LEGEND` classes lists the
first ones that appear in the rows shown and says how many more there are; a per-class chart of more than
:data:`MAX_GROUPED_CLASSES` classes draws the IoU bars only.

The confusion figure is NOT here: ``train.common.classification_viz.plot_confusion_counts``
draws it from a count matrix (the label-based ``plot_confusion_matrix`` would need one label
pair per pixel, tens of millions).
"""

import colorsys
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import keras
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from dl_techniques.utils.logger import logger

PathLike = Union[str, Path]

DPI = 100
# One RGB colour per class id, in the order of the trainer's ``CLASS_NAMES`` (pet, background,
# border). Fixed and shared by the ground-truth and prediction columns so a colour means the
# same class in every panel of every figure.
PALETTE = np.array([[230, 126, 34], [52, 73, 94], [241, 196, 15]], dtype=np.uint8)
#: The colour of a void pixel and its legend name: a light grey, not a row of :func:`extended_palette`.
VOID_COLOR = np.array([190, 190, 190], dtype=np.uint8)
VOID_NAME = "void"
#: A legend lists at most this many classes (then "+N more"), and a per-class chart groups four bars per class
#: only up to this many classes.
MAX_LEGEND = 24
MAX_GROUPED_CLASSES = 12
TRAIN_COLOR = "#d62728"
VAL_COLOR = "#1f77b4"
SCORE_COLORS = {"iou": "#1f77b4", "dice": "#2ca02c", "precision": "#ff7f0e", "recall": "#9467bd"}
# Height in inches of one grid row and width of one panel column.
PANEL_INCHES = 2.4


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def _as_uint8_images(images: np.ndarray) -> np.ndarray:
    """``(N, H, W, 3)`` uint8, from uint8 or float images in ``[0, 1]``.

    Raises:
        ValueError: On a wrong rank or channel count, no image, or a float value outside [0, 1].
    """
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] != 3 or images.shape[0] < 1:
        raise ValueError(f"images must be a non-empty (N, H, W, 3) array, got shape {images.shape}")
    if images.dtype == np.uint8:
        return images
    if not np.issubdtype(images.dtype, np.floating):
        raise ValueError(f"images must be uint8 or float in [0, 1], got dtype {images.dtype}")
    if images.min() < 0.0 or images.max() > 1.0:
        raise ValueError(f"float images must lie in [0, 1], got [{images.min()}, {images.max()}]")
    return np.rint(images * 255.0).astype(np.uint8)


def extended_palette(num_classes: int) -> np.ndarray:
    """``(num_classes, 3)`` uint8 colours, the first three rows being :data:`PALETTE`.

    Deterministic and prefix-stable (``extended_palette(n + 1)[:n]`` is ``extended_palette(n)``): the rows
    after the third step the hue by the golden ratio and cycle saturation and value, and skip a colour already
    taken or equal to :data:`VOID_COLOR`, so no two classes and no class and void share a colour.

    Raises:
        ValueError: For fewer than one class.
    """
    if num_classes < 1:
        raise ValueError(f"need at least one class, got {num_classes}")
    if num_classes <= len(PALETTE):
        return PALETTE[:num_classes].copy()
    rows = [tuple(int(v) for v in row) for row in PALETTE]
    taken = set(rows) | {tuple(int(v) for v in VOID_COLOR)}
    saturations, values, hue, step = (0.85, 0.55, 0.7), (0.95, 0.75, 0.85), 0.1, 0.6180339887498949
    while len(rows) < num_classes:
        index = len(rows) - len(PALETTE)
        hue = (hue + step) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, saturations[index % 3], values[(index // 3) % 3])
        colour = tuple(int(round(255 * channel)) for channel in rgb)
        if colour not in taken:
            taken.add(colour)
            rows.append(colour)
    return np.array(rows, dtype=np.uint8)


def _as_masks(
        masks: np.ndarray, like: np.ndarray, num_classes: int, what: str, ignore_index: Optional[int] = None,
) -> np.ndarray:
    """``(N, H, W)`` int class ids matching ``like``'s ``(N, H, W)`` and lying in ``[0, C)``.

    A pixel equal to ``ignore_index`` (void) is allowed as well.

    Raises:
        ValueError: On a shape mismatch, a non-integer dtype, an id outside ``[0, C)`` that is not void, or an
            ``ignore_index`` that is a class or below zero.
    """
    masks = np.asarray(masks)
    if masks.shape != like.shape[:3]:
        raise ValueError(f"{what} has shape {masks.shape}, the images are {like.shape[:3]}")
    if not np.issubdtype(masks.dtype, np.integer):
        raise ValueError(f"{what} must hold integer class ids, got dtype {masks.dtype}")
    if ignore_index is not None and (ignore_index < 0 or ignore_index < num_classes):
        raise ValueError(f"ignore_index must be a whole number outside [0, {num_classes}), got {ignore_index}")
    shown = masks if ignore_index is None else masks[masks != ignore_index]
    if shown.size and (shown.min() < 0 or shown.max() >= num_classes):
        raise ValueError(f"{what} ids must lie in [0, {num_classes}), got [{shown.min()}, {shown.max()}]")
    return masks


def _colorize(mask: np.ndarray, palette: np.ndarray = PALETTE, ignore_index: Optional[int] = None) -> np.ndarray:
    """``(H, W)`` class ids to ``(H, W, 3)`` uint8 through ``palette``; void pixels get :data:`VOID_COLOR`."""
    if ignore_index is None:
        return palette[mask]
    void = mask == ignore_index
    return np.where(void[..., None], VOID_COLOR, palette[np.where(void, 0, mask)])


def _save(fig: Figure, out_path: PathLike) -> str:
    """Save ``fig`` (parents created) and return the path written as a string."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    return str(out)


def _check_class_names(class_names: Sequence[str], palette: np.ndarray = PALETTE) -> List[str]:
    """The names as a list; at most ``len(palette)`` classes have a colour.

    Raises:
        ValueError: If there are no names or more names than palette colours.
    """
    names = [str(name) for name in class_names]
    if not 1 <= len(names) <= len(palette):
        raise ValueError(f"need between 1 and {len(palette)} class names, got {names}")
    return names


# ---------------------------------------------------------------------
# Scores from a confusion matrix
# ---------------------------------------------------------------------

def class_scores(confusion: np.ndarray) -> Dict[str, Any]:
    """Per-class and overall scores of a pixel confusion matrix (rows true, columns predicted).

    With ``TP_c`` the diagonal, ``true_c`` the row sum and ``pred_c`` the column sum:
    ``IoU = TP / (true + pred - TP)``, ``Dice = 2 TP / (true + pred)``,
    ``precision = TP / pred``, ``recall = TP / true``. A score with a zero denominator is
    ``None`` (never NaN, so the report stays strict JSON) and a ``None`` IoU is left out of
    ``miou``, the convention of ``keras.metrics.MeanIoU``. This is the ONE implementation of
    IoU in the package: the trainer's ``segmentation_scores`` delegates here.

    Args:
        confusion: ``(C, C)`` non-negative integer counts.

    Returns:
        ``{"per_class_iou", "per_class_dice", "per_class_precision", "per_class_recall"}``
        (lists of ``float | None`` of length C), ``"support"`` (true pixels per class, ints),
        ``"miou"`` and ``"mean_dice"`` (``float | None``), ``"pixel_accuracy"``
        (``float | None``).

    Raises:
        ValueError: If ``confusion`` is not a square matrix of non-negative counts.
    """
    confusion = np.asarray(confusion)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1] or confusion.shape[0] < 1:
        raise ValueError(f"confusion must be a square (C, C) matrix, got shape {confusion.shape}")
    if (confusion < 0).any():
        raise ValueError("confusion holds a negative count")
    confusion = confusion.astype(np.int64)
    tp = np.diag(confusion).astype(np.float64)
    true, pred = confusion.sum(axis=1).astype(np.float64), confusion.sum(axis=0).astype(np.float64)

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> List[Optional[float]]:
        return [float(n / d) if d > 0 else None for n, d in zip(numerator, denominator)]

    def mean(values: List[Optional[float]]) -> Optional[float]:
        valid = [v for v in values if v is not None]
        return float(np.mean(valid)) if valid else None

    iou, dice = ratio(tp, true + pred - tp), ratio(2.0 * tp, true + pred)
    total = confusion.sum()
    return {
        "per_class_iou": iou,
        "per_class_dice": dice,
        "per_class_precision": ratio(tp, pred),
        "per_class_recall": ratio(tp, true),
        "support": [int(v) for v in true],
        "miou": mean(iou),
        "mean_dice": mean(dice),
        "pixel_accuracy": float(tp.sum() / total) if total else None,
    }


def segmentation_report(confusion: np.ndarray, class_names: Sequence[str]) -> Dict[str, Any]:
    """:func:`class_scores` keyed by class name, with the counts, as one JSON-ready dict.

    Args:
        confusion: ``(C, C)`` non-negative integer counts.
        class_names: C names.

    Returns:
        ``{"class_names", "per_class": {name: {iou, dice, precision, recall, support}},
        "miou", "mean_dice", "pixel_accuracy", "confusion"}``.

    Raises:
        ValueError: If the matrix is malformed or ``class_names`` is not C names.
    """
    scores = class_scores(confusion)
    names = [str(n) for n in class_names]
    if len(names) != len(scores["support"]):
        raise ValueError(f"{len(names)} class names for a {len(scores['support'])}-class matrix")
    return {
        "class_names": names,
        "per_class": {
            name: {"iou": scores["per_class_iou"][i], "dice": scores["per_class_dice"][i],
                   "precision": scores["per_class_precision"][i],
                   "recall": scores["per_class_recall"][i], "support": scores["support"][i]}
            for i, name in enumerate(names)
        },
        "miou": scores["miou"],
        "mean_dice": scores["mean_dice"],
        "pixel_accuracy": scores["pixel_accuracy"],
        "confusion": np.asarray(confusion).astype(np.int64).tolist(),
    }


def write_segmentation_report(confusion: np.ndarray, class_names: Sequence[str], out_path: PathLike) -> Dict[str, Any]:
    """Write :func:`segmentation_report` as strict JSON (``allow_nan=False``) and return it.

    Args:
        confusion: ``(C, C)`` counts.
        class_names: C names.
        out_path: JSON destination (parents created).

    Returns:
        The report that was written.

    Raises:
        ValueError: If the matrix or the names are malformed (nothing is written).
    """
    report = segmentation_report(confusion, class_names)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def _classes_in(array: np.ndarray, ignore_index: Optional[int]) -> np.ndarray:
    """The class ids of ``array`` as a flat vector, void left out."""
    flat = array.ravel()
    return flat if ignore_index is None else flat[flat != ignore_index]


def _legend_entries(
        names: List[str], palette: np.ndarray, shown: np.ndarray, ignore_index: Optional[int],
) -> List[Patch]:
    """The legend patches: every class when there are at most :data:`MAX_LEGEND`, else the first ones shown.

    ``shown`` holds the class ids that appear in the rows drawn (void excluded); void gets its own entry
    when ``ignore_index`` is set.
    """
    if len(names) <= MAX_LEGEND:
        ids = list(range(len(names)))
        extra = ""
    else:
        ids = sorted(int(k) for k in np.unique(shown))[:MAX_LEGEND]
        left = len(names) - len(ids)
        extra = f"+{left} more"
    patches = [Patch(facecolor=palette[i] / 255.0, label=names[i]) for i in ids]
    if extra:
        patches.append(Patch(facecolor="white", edgecolor="none", label=extra))
    if ignore_index is not None:
        patches.append(Patch(facecolor=VOID_COLOR / 255.0, label=VOID_NAME))
    return patches


def _draw_rows(
        images: np.ndarray, truths: np.ndarray, predictions: Sequence[Tuple[str, np.ndarray]],
        class_names: Sequence[str], out_path: PathLike, title: Optional[str],
        palette: np.ndarray = PALETTE, ignore_index: Optional[int] = None,
) -> List[List[float]]:
    """Shared renderer: one row per sample, columns image | ground truth | each prediction.

    Args:
        images: ``(N, H, W, 3)`` uint8 or float in ``[0, 1]``.
        truths: ``(N, H, W)`` class ids.
        predictions: ``(column label, (N, H, W) class ids)`` pairs, at least one.
        class_names: Names for the legend, one per class.
        out_path: PNG destination.
        title: Figure suptitle, or ``None``.
        palette: ``(K, 3)`` uint8 colours, ``K >= len(class_names)``.
        ignore_index: The void value of ``truths``, or ``None``.

    Returns:
        Per prediction column, the per-row pixel accuracy over the VALID pixels (also drawn in each panel
        title); ``nan`` for a row whose pixels are all void.

    Raises:
        ValueError: On any malformed input.
    """
    names = _check_class_names(class_names, palette)
    images = _as_uint8_images(images)
    truths = _as_masks(truths, images, len(names), "ground truth", ignore_index)
    if not predictions:
        raise ValueError("need at least one prediction column")
    masks = [(label, _as_masks(p, images, len(names), f"prediction {label!r}", ignore_index))
             for label, p in predictions]

    n_rows, n_cols = len(images), 2 + len(masks)
    fig = Figure(figsize=(PANEL_INCHES * n_cols, PANEL_INCHES * n_rows + 1.0), layout="constrained")
    axes = np.atleast_2d(fig.subplots(n_rows, n_cols, squeeze=False))
    accuracies: List[List[float]] = [[] for _ in masks]
    for row in range(n_rows):
        panels = [("image", images[row]), ("ground truth", _colorize(truths[row], palette, ignore_index))]
        valid = np.ones(truths[row].shape, bool) if ignore_index is None else truths[row] != ignore_index
        for k, (label, predicted) in enumerate(masks):
            # DECISION plan-2026-10-31T180033-2faf890a/D-079
            # A void pixel is in NO row's accuracy. Do NOT take the mean over the whole row: a pixel the
            # model cannot be right about would lower every row by the void share, and a row that is all
            # void would read 0 (or, with a void prediction, 1). The mean is over the valid pixels, and a
            # row with none is `nan`, titled n/a.
            accuracy = float(np.mean(predicted[row][valid] == truths[row][valid])) if valid.any() else float("nan")
            accuracies[k].append(accuracy)
            shown_accuracy = "n/a" if math.isnan(accuracy) else f"{accuracy:.3f}"
            panels.append((f"{label}\npixel acc {shown_accuracy}", _colorize(predicted[row], palette, ignore_index)))
        for col, (panel_title, picture) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(picture, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0 or col >= 2:
                ax.set_title(panel_title, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11)
    appearing = np.unique(np.concatenate(
        [_classes_in(truths, ignore_index)] + [_classes_in(p, ignore_index) for _, p in masks]))
    entries = _legend_entries(names, palette, appearing, ignore_index)
    fig.legend(handles=entries, loc="outside lower center", ncol=min(len(entries), 8), fontsize=9, frameon=False)
    _save(fig, out_path)
    return accuracies


def plot_segmentation_grid(
        images: np.ndarray, truths: np.ndarray, predictions: np.ndarray, class_names: Sequence[str],
        out_path: PathLike, title: Optional[str] = None, palette: np.ndarray = PALETTE,
        ignore_index: Optional[int] = None,
) -> List[float]:
    """Save a grid with one row per sample: image | ground truth | prediction.

    Ground truth and prediction share :data:`PALETTE` and a legend names the classes; each
    prediction panel's title carries that sample's pixel accuracy.

    Args:
        images: ``(N, H, W, 3)`` uint8 or float in ``[0, 1]``, ``N >= 1``.
        truths: ``(N, H, W)`` integer class ids in ``[0, C)``.
        predictions: ``(N, H, W)`` integer class ids in ``[0, C)``.
        class_names: C names, ``C <= len(palette)`` (3 with the default palette).
        out_path: PNG destination (parents created).
        title: Optional figure title.
        palette: Colour table; pass ``extended_palette(C)`` for more than three classes.
        ignore_index: The void value of ``truths`` (drawn in :data:`VOID_COLOR`, left out of the
            per-row accuracy), or ``None``.

    Returns:
        The per-row pixel accuracy over the valid pixels (``nan`` for a row that is all void).

    Raises:
        ValueError: On a wrong shape, an empty batch, a non-integer mask or an id outside ``[0, C)``.
    """
    return _draw_rows(images, truths, [("prediction", predictions)], class_names, out_path, title,
                      palette, ignore_index)[0]


def plot_best_vs_final_predictions(
        images: np.ndarray, truths: np.ndarray, best: np.ndarray, final: np.ndarray,
        class_names: Sequence[str], out_path: PathLike, best_label: str = "best",
        final_label: str = "final", title: Optional[str] = None, palette: np.ndarray = PALETTE,
        ignore_index: Optional[int] = None,
) -> Tuple[List[float], List[float]]:
    """Save image | ground truth | best-weights prediction | final-weights prediction per sample.

    Args:
        images: ``(N, H, W, 3)`` uint8 or float in ``[0, 1]``.
        truths: ``(N, H, W)`` class ids.
        best: ``(N, H, W)`` predictions of the best checkpoint.
        final: ``(N, H, W)`` predictions of the last-epoch weights.
        class_names: C names, ``C <= 3``.
        out_path: PNG destination.
        best_label, final_label: Column titles (name the epoch there).
        title: Optional figure title.
        palette, ignore_index: As :func:`plot_segmentation_grid`.

    Returns:
        ``(best per-row pixel accuracy, final per-row pixel accuracy)``.

    Raises:
        ValueError: On any malformed input, as :func:`plot_segmentation_grid`.
    """
    best_acc, final_acc = _draw_rows(
        images, truths, [(best_label, best), (final_label, final)], class_names, out_path, title,
        palette, ignore_index)
    return best_acc, final_acc


def _plot_many_class_scores(report: Dict[str, Any], out_path: PathLike) -> None:
    """One IoU bar per class (a class with no IoU is an empty slot), labels thinned to about 40 ticks."""
    names = report["class_names"]
    n = len(names)
    iou = [report["per_class"][name]["iou"] for name in names]
    fig = Figure(figsize=(max(7.0, 0.16 * n + 2.0), 4.8), layout="constrained")
    ax = fig.subplots()
    ax.bar(np.arange(n), [0.0 if v is None else v for v in iou], 0.8, color=SCORE_COLORS["iou"], label="iou")
    every = max(1, math.ceil(n / 40))
    ax.set_xticks(np.arange(0, n, every))
    ax.set_xticklabels([names[k] for k in range(0, n, every)], rotation=90, fontsize=7)
    ax.set_xlim(-0.8, n - 0.2)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("IoU")
    miou, absent = report["miou"], sum(v is None for v in iou)
    ax.set_title(f"Per-class IoU, {n} classes (mIoU {'n/a' if miou is None else f'{miou:.4f}'}"
                 f"{f'; {absent} without an IoU' if absent else ''})")
    ax.grid(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, out_path)


def plot_per_class_scores(confusion: np.ndarray, class_names: Sequence[str], out_path: PathLike) -> Dict[str, Any]:
    """Save grouped bars of IoU, Dice, precision and recall per class; return the report.

    More than :data:`MAX_GROUPED_CLASSES` classes are too many for four bars each: the chart is then one IoU
    bar per class (the report still holds all four scores).

    A score with no defined value (a class absent from both masks) is drawn as an empty
    slot labelled ``n/a`` rather than as a zero.

    Args:
        confusion: ``(C, C)`` counts.
        class_names: C names.
        out_path: PNG destination.

    Returns:
        :func:`segmentation_report` of the same matrix.

    Raises:
        ValueError: If the matrix or the names are malformed.
    """
    report = segmentation_report(confusion, class_names)
    names = report["class_names"]
    n = len(names)
    if n > MAX_GROUPED_CLASSES:
        _plot_many_class_scores(report, out_path)
        return report
    fig = Figure(figsize=(max(7.0, 2.4 * n + 2.0), 4.8), layout="constrained")
    ax = fig.subplots()
    width = 0.8 / len(SCORE_COLORS)
    for k, (key, color) in enumerate(SCORE_COLORS.items()):
        values = [report["per_class"][name][key] for name in names]
        x = np.arange(n) + (k - (len(SCORE_COLORS) - 1) / 2.0) * width
        ax.bar(x, [0.0 if v is None else v for v in values], width, color=color, label=key)
        for xi, v in zip(x, values):
            ax.annotate("n/a" if v is None else f"{v:.2f}", xy=(xi, 0.0 if v is None else v),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([f"{name}\n(n={report['per_class'][name]['support']:,} px)" for name in names], fontsize=9)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("score")
    miou = report["miou"]
    ax.set_title("Per-class IoU / Dice / precision / recall "
                 f"(mIoU {'n/a' if miou is None else f'{miou:.4f}'})")
    # ``dl_techniques.visualization.core`` sets a whitegrid style at import (axes.grid True
    # process-wide): switch the grid off first so no vertical line runs through the bar groups.
    ax.grid(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    _save(fig, out_path)
    return report


def plot_miou_curve(
        history: Dict[str, Sequence[float]], best_epoch: int, out_path: PathLike, title: Optional[str] = None,
        monitor: str = "val_loss",
) -> None:
    """Save train and validation mIoU per epoch with the best epoch marked.

    ``best_epoch`` is the epoch chosen by the checkpoint ``monitor`` (validation loss by
    default), which need not be the epoch of the highest validation mIoU; the label says so.

    Args:
        history: ``{"miou": [...], "val_miou": [...]}`` per epoch, equal non-zero lengths.
        best_epoch: 1-based epoch to mark, within ``[1, epochs]``.
        out_path: PNG destination.
        title: Optional figure title.
        monitor: Name of the checkpoint metric, printed in the label.

    Raises:
        ValueError: If a series is missing, empty, of unequal length or non-finite, or
            ``best_epoch`` is outside the epoch range.
    """
    for key in ("miou", "val_miou"):
        if not len(history.get(key, ())):
            raise ValueError(f"history has no {key!r} values")
    train, val = np.asarray(history["miou"], dtype=float), np.asarray(history["val_miou"], dtype=float)
    if train.shape != val.shape or not (np.isfinite(train).all() and np.isfinite(val).all()):
        raise ValueError("miou and val_miou must have equal length and finite values")
    n = len(val)
    if not 1 <= best_epoch <= n:
        raise ValueError(f"best_epoch must be in [1, {n}], got {best_epoch}")
    epochs = np.arange(1, n + 1)
    fig = Figure(figsize=(7.0, 4.4), layout="constrained")
    ax = fig.subplots()
    ax.plot(epochs, train, "o-", color=TRAIN_COLOR, label="train (running mean over the epoch)")
    ax.plot(epochs, val, "o-", color=VAL_COLOR, label="validation (end of epoch)")
    ax.axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=1.2,
               label=f"best epoch {best_epoch} (by {monitor})")
    ax.plot([best_epoch], [val[best_epoch - 1]], "*", color="#2ca02c", markersize=14)
    ax.set_xlim(0.5, n + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("epoch")
    ax.set_ylabel("mIoU")
    ax.set_title(title or "mIoU per epoch")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _save(fig, out_path)


# ---------------------------------------------------------------------
# Per-epoch grid
# ---------------------------------------------------------------------

class SegmentationGridCallback(keras.callbacks.Callback):
    """Writes ``epoch_NNN_seg_grid.png`` for one fixed validation batch, on a cadence.

    The ``viz_samples`` validation images are drawn ONCE, at construction, with
    ``numpy.random.RandomState(seed)``, so every epoch of a run (and every run of one seed)
    shows the same images and a change between two grids is a change in the model. Files
    are numbered by completed epochs: ``epoch_000`` is the untrained model
    (``on_train_begin``), ``epoch_001`` the model after the first epoch, then every
    ``viz_freq``-th epoch, and the last completed epoch is always drawn (the planned final
    epoch, or the stop epoch of an early stop, in ``on_train_end``). Predictions use
    ``training=False``.

    A figure failure never stops training: it is logged as a WARNING and recorded in
    :attr:`failed`. Place it AFTER the callbacks that edit ``logs`` (so ``val_miou`` is
    there for the title) and BEFORE the dashboard.

    Args:
        x: Validation images ``(N, H, W, 3)`` uint8 (scaled by 1 / 255 for the model).
        y: Validation masks ``(N, H, W)`` integer class ids.
        out_dir: Directory the grids are written to (created on first write).
        class_names: One name per class.
        viz_freq: Draw every this many completed epochs; ``>= 1``.
        viz_samples: Number of validation images per grid; ``>= 1`` (clamped to ``N``).
        seed: Seed of the sample choice.
        title: Figure title prefix.
        palette: Colour table (default the three-colour :data:`PALETTE`; ``extended_palette(C)`` for more classes).
        ignore_index: The void value of ``y``, or ``None``.
        val_population: What ``val_miou`` in the title is a score OF, in words; ``None`` is the legacy
            ``all N`` (the images drawn are drawn from the validation set it was scored on). A caller that draws
            other images than the ones ``val_miou`` is scored on (``train.api``'s segmentation task draws test
            pairs and scores the selection split) must say so here, or the title would claim otherwise.

    Attributes:
        n_val: Number of validation images (the population of the ``val mIoU`` in the title).
        indices: The chosen rows of ``x``, sorted ascending.
        written: File names written, in order.
        failed: ``{file name: error text}`` of figures that raised.
        seconds: Wall seconds spent drawing (prediction plus figure) over every grid so far.

    Raises:
        ValueError: If ``viz_freq`` or ``viz_samples`` is below 1, or ``x`` and ``y`` differ
            in length or are empty.
    """

    def __init__(
            self, x: np.ndarray, y: np.ndarray, out_dir: PathLike, class_names: Sequence[str],
            viz_freq: int, viz_samples: int, seed: int, title: str = "",
            palette: np.ndarray = PALETTE, ignore_index: Optional[int] = None,
            val_population: Optional[str] = None,
    ) -> None:
        super().__init__()
        if viz_freq < 1 or viz_samples < 1:
            raise ValueError(f"viz_freq and viz_samples must be >= 1, got {viz_freq} and {viz_samples}")
        if len(x) != len(y) or len(x) < 1:
            raise ValueError(f"x has {len(x)} rows and y {len(y)}: need equal, non-zero lengths")
        self.n_val = len(x)
        self.indices = np.sort(np.random.RandomState(seed).choice(
            len(x), size=min(viz_samples, len(x)), replace=False))
        self.images = np.asarray(x)[self.indices]
        self.masks = np.asarray(y)[self.indices]
        self.out_dir = Path(out_dir)
        self.class_names = list(class_names)
        self.viz_freq = viz_freq
        self.title = title
        self.palette = palette
        self.ignore_index = ignore_index
        self.val_population = val_population
        self.written: List[str] = []
        self.failed: Dict[str, str] = {}
        self.seconds = 0.0
        self._completed = 0
        self._drawn_at = -1
        self._last_logs: Dict[str, Any] = {}

    def predict_classes(self, model: Optional[keras.Model] = None) -> np.ndarray:
        """Argmax class ids ``(n, H, W)`` of ``model`` (default: the fitted one) on the fixed batch.

        Args:
            model: A model taking float images in ``[0, 1]`` and returning per-pixel logits.

        Returns:
            ``int64`` array; the model runs with ``training=False``.
        """
        model = model or self.model
        logits = model(self.images.astype(np.float32) / 255.0, training=False)
        return np.argmax(keras.ops.convert_to_numpy(logits), axis=-1)

    def _draw(self, completed: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Write the grid of ``completed`` epochs; never raises."""
        name = f"epoch_{completed:03d}_seg_grid.png"
        suffix = "untrained" if completed == 0 else f"after epoch {completed}"
        val_miou = (logs or {}).get("val_miou")
        if val_miou is not None:
            # The mIoU is of the WHOLE validation set; the sheet shows only ``len(indices)`` of it.
            population = f"all {self.n_val}" if self.val_population is None else self.val_population
            suffix += f", val mIoU {float(val_miou):.3f} ({population})"
        started = time.perf_counter()
        try:
            plot_segmentation_grid(
                self.images, self.masks, self.predict_classes(), self.class_names,
                self.out_dir / name, title=f"{self.title} {suffix}".strip(),
                palette=self.palette, ignore_index=self.ignore_index)
            self.written.append(name)
            self.failed.pop(name, None)
            logger.info(f"Segmentation grid written: {name}")
        except Exception as e:  # noqa: BLE001 - a figure must never stop training
            logger.warning(f"Segmentation grid {name} failed: {type(e).__name__}: {e}")
            self.failed[name] = f"{type(e).__name__}: {e}"
        self.seconds += time.perf_counter() - started
        self._drawn_at = completed

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """The untrained model, as ``epoch_000``."""
        self._completed = 0
        self._draw(0)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Draw after every ``viz_freq``-th epoch and after the planned last one."""
        self._completed = epoch + 1
        self._last_logs = dict(logs or {})
        planned = (self.params or {}).get("epochs")
        if self._completed % self.viz_freq == 0 or self._completed == planned:
            self._draw(self._completed, logs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Draw the last completed epoch if it was not drawn (an early stop off the cadence)."""
        if self._completed and self._drawn_at != self._completed:
            self._draw(self._completed, self._last_logs)
