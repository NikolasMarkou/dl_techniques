"""
Plots shared by the classification trainers (PowerMLP, ConvNeXt).

Matplotlib + numpy + scikit-learn only (no seaborn / pandas / tensorflow), so
importing this module allocates nothing. Every figure is drawn headless (Agg),
saved at ``dpi=120`` with a tight bounding box, and closed on every path.

Contents:

- :func:`render_training_dashboard` / :class:`TrainingDashboardCallback` - one
  overwritten multi-panel PNG of the per-epoch curves, redrawn on a cadence that
  scales with the planned epoch count (see :data:`DASHBOARD_TARGET_DRAWS`), with
  an epoch-0 baseline marker. Only panels that have data are drawn (no blank
  cells).
- :func:`plot_confusion_matrix` - counts and row-normalized side by side. A local
  two-panel figure rather than the library ``confusion_matrix`` plugin: that
  plugin saves under a timestamped subdirectory of its own choosing (so the
  caller cannot fix the output path), imports seaborn and pandas at module
  level, and its grid lines cut through the cell annotations.
- :func:`plot_per_class_metrics`, :func:`plot_calibration`,
  :func:`plot_confident_errors` - post-training evaluation figures.

The ``plot_*`` functions raise on bad input (the caller decides whether a missing
figure is fatal); the dashboard callback never raises, so a plotting bug cannot
abort a training run.
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import: headless-safe

import keras  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # noqa: E402

from dl_techniques.utils.logger import logger  # noqa: E402

PathLike = Union[str, Path]

DPI = 120
TRAIN_COLOR = "#d62728"
VAL_COLOR = "#1f77b4"
ACCENT_COLOR = "#2ca02c"

# DECISION plan-2026-09-18T213948-68dcb72c/D-027: do NOT go back to redrawing the
# dashboard after every epoch "so the PNG is always live": that cost 40% of a
# 100-epoch run (decisions.md D-027, finding F1). Epoch 1 and the final state are
# always drawn; guards: test_dashboard_draws_on_a_cadence_and_always_shows_the_final_state.
# The dashboard is redrawn about this many times per run, not once per epoch: a render
# costs 1.0-1.7 s (CPU, measured) against a 1.8 s MNIST training epoch, so drawing
# every epoch of a 100-epoch run spent about 40% of the wall-clock on plotting
# (run wall 348 s vs 196 s of summed epoch times). The cadence is
# ``max(1, planned_epochs // DASHBOARD_TARGET_DRAWS)`` epochs; epoch 1 and the final
# state are always drawn.
DASHBOARD_TARGET_DRAWS = 20

# Reliability-diagram bins holding fewer samples than this are drawn hatched and
# lighter and annotated with their n: a bin of 3 samples has an accuracy of 0, 1/3,
# 2/3 or 1 and must not read as evidence.
CALIBRATION_MIN_BIN_SAMPLES = 20

# The smoothed-loss panel is only informative once there are enough epochs.
SMOOTHED_PANEL_MIN_EPOCHS = 6

# Dashboard caption: what the two curve families measure (they are NOT the same
# quantity, which makes the epoch-1 generalization gap look negative).
TRAIN_METRIC_CAPTION = "train: running mean over the epoch; val: end of epoch"

# Legend text of the epoch-0 marker for a baseline measured with ``model.evaluate``.
BASELINE_LABEL = "epoch-0 baseline (val)"

# The epoch-0 baseline marker is drawn at most this many times the largest finite
# plotted curve value; a larger baseline (an init loss of 218 over curves near 1)
# is clipped there and annotated with its true value, so it cannot flatten the axis.
# 10.0, not 4.0: the healthy headline run has ln(10) = 2.31 over a curve max of
# about 0.5 (ratio 4.65) and a factor of 4 clipped that correct baseline.
BASELINE_CLIP_FACTOR = 10.0

# The per-epoch-time panel draws a bar above this many times the median of epochs
# >= 2 at that ceiling and annotates it with its true value. Epoch 1 pays the XLA
# warmup (about 22 s against about 2 s) and would otherwise set the whole y-scale.
TIME_CLIP_FACTOR = 3.0

# The row-normalized confusion-matrix panel annotates only cells strictly ABOVE this
# fraction. Strict, because 0.005 itself (5 of 1000) formats as "0%" (round half to
# even), the exact text this floor exists to remove.
CONFUSION_MIN_ANNOTATION = 0.005

# The per-class bar chart starts its y axis at zero, so bar length is proportional to the
# score. Only when EVERY bar is at or above this value is the axis truncated (a 0.98
# MNIST chart would otherwise be a wall of equal bars) and the axis label says so. A
# truncated axis on a weaker run made a 0.52 bar look a seventh as tall as a 0.86 one.
PER_CLASS_TRUNCATE_FROM = 0.8

# More classes than this and the confusion matrix / per-class chart switch to their
# bounded layouts: a 100-class run made a 15585 x 7571 px confusion matrix (about 1 px of
# text) and an 8837 x 524 px per-class strip, neither readable at any size. Up to this
# many classes the original layouts (counts + row-normalized; grouped bars) are kept.
MANY_CLASSES_THRESHOLD = 20

# The per-class chart of a many-class run draws the worst and the best this many classes
# by F1 (so at most ``2 * PER_CLASS_SHOWN_EACH_END`` rows, whatever the class count); the
# full table stays in ``classification_report.json``.
PER_CLASS_SHOWN_EACH_END = 15

# The many-class confusion figure lists this many most-confused (true -> predicted) pairs.
CONFUSION_TOP_PAIRS = 15

# Row-normalized heatmap of a many-class run: tick labels go only to classes in the top
# pairs, and never closer together than ``n / CONFUSION_MAX_TICK_LABELS`` classes so the
# labels cannot overprint one another whatever the class count.
CONFUSION_MAX_TICK_LABELS = 40

# Gamma of the many-class heatmap colour scale: the diagonal (0.4 to 0.9) would otherwise
# leave every off-diagonal cell (0.01 to 0.1) an unreadable near-white.
CONFUSION_HEATMAP_GAMMA = 0.5

# The learning-rate panel shows at most this many decades below the peak. A warmup that
# starts at 1e-8 under a 1e-3 peak spans 10 decades and squeezed the real schedule into
# the top fifth of the axis; points below the floor are drawn ON its bottom edge and
# annotated with their value.
LR_PANEL_MAX_DECADES = 3.0


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _save_and_close(fig: plt.Figure, out_path: PathLike) -> str:
    """Save ``fig`` to ``out_path`` (parents created) and always close it.

    Returns:
        The path written, as a string.
    """
    out = Path(out_path)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
    finally:
        plt.close(fig)
    return str(out)


# Below this ratio (hi / lo) a log axis is "narrow": its ticks come from a nice set of
# mantissas (see ``_nice_log_ticks``) and it has no minor ticks; at or above it the
# ordinary decade ticks carry the labels alone.
LOG_NARROW_MAX_RATIO = 10.0

# Mantissa sets tried in order until between ``LOG_MIN_NICE_TICKS`` and
# ``LOG_MAX_NICE_TICKS`` ticks fall inside the view: coarse first (1, 1.5, 2, 3, 5 read
# cleanly), then finer for a narrow view. The last set (steps of 0.1) is only a fallback
# for a very narrow view: a 2.8 to 4.7 loss range gives 3 and 4 from the second set and
# 3, 3.5, 4, 4.5 from the third, and must never fall through to twenty labels.
_NICE_MANTISSAS: Tuple[Tuple[float, ...], ...] = (
    (1.0, 1.5, 2.0, 3.0, 5.0),
    (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0),
    (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0),
    tuple(round(1.0 + 0.1 * i, 1) for i in range(90)),
)
LOG_MIN_NICE_TICKS = 3
LOG_MAX_NICE_TICKS = 8


def _nice_log_ticks(vmin: float, vmax: float) -> np.ndarray:
    """Ticks in ``[vmin, vmax]`` from the coarsest mantissa set that gives enough, but not too many.

    A 0.8 to 2.4 loss range reads ``1, 1.5, 2`` (matplotlib's own minor ticks read
    ``2, 1, 0.9, 0.8``: crowded at one end, empty at 1.5).

    Args:
        vmin, vmax: The visible range, ``0 < vmin < vmax``.

    Returns:
        Ascending tick values inside the range, at most :data:`LOG_MAX_NICE_TICKS` (fewer
        than :data:`LOG_MIN_NICE_TICKS` if no set fits that many inside the range).
    """
    lo_exp, hi_exp = int(np.floor(np.log10(vmin))), int(np.ceil(np.log10(vmax)))
    best = np.array([])
    for mantissas in _NICE_MANTISSAS:
        candidates = np.array([m * 10.0 ** e for e in range(lo_exp, hi_exp + 1) for m in mantissas])
        ticks = candidates[(candidates >= vmin) & (candidates <= vmax)]
        if LOG_MIN_NICE_TICKS <= len(ticks) <= LOG_MAX_NICE_TICKS:
            return ticks
        if best.size < len(ticks) <= LOG_MAX_NICE_TICKS:
            best = ticks
    return best


class _NarrowAwareLogLocator(LogLocator):
    """A ``LogLocator`` whose ticks depend on how wide the view is.

    Wide (``vmax / vmin`` at or above :data:`LOG_NARROW_MAX_RATIO`): behaves as the base
    class (decades as major ticks, the usual minor ticks). Narrow: the major locator
    returns :func:`_nice_log_ticks` and the minor locator returns nothing, so no tick
    is labelled twice.

    Args:
        minor: ``True`` for the minor-tick instance.
    """

    def __init__(self, minor: bool = False) -> None:
        super().__init__(base=10.0, subs="auto" if minor else (1.0,), numticks="auto")
        self._is_minor = minor

    def tick_values(self, vmin: float, vmax: float) -> np.ndarray:
        if vmin > 0.0 and vmax < vmin * LOG_NARROW_MAX_RATIO:
            return np.array([]) if self._is_minor else _nice_log_ticks(vmin, vmax)
        return super().tick_values(vmin, vmax)


def _plain_log_ticks(ax: plt.Axes) -> None:
    """Label a log-scaled y axis with plain numbers at readable positions.

    The default log formatter prints scientific notation on every minor tick once the
    range is under a decade (a loss curve of 1.1 to 2.5 read ``1.8x10^0``, ``2x10^0``).
    Labels read ``{value:.3g}`` (``1.5``, ``0.001``). A wide range (``hi / lo`` at or above
    :data:`LOG_NARROW_MAX_RATIO`) keeps its decades and no minor labels; a narrow one gets
    ticks from a nice set (:func:`_nice_log_ticks`) and no minor ticks. The range is read
    when the ticks are drawn, so later ``set_ylim`` calls are honoured.

    Args:
        ax: An axes whose y scale is already ``"log"``; nothing is checked or drawn here.
    """
    def label(value: float, _pos: Optional[int]) -> str:
        return f"{value:.3g}"

    ax.yaxis.set_major_locator(_NarrowAwareLogLocator())
    ax.yaxis.set_minor_locator(_NarrowAwareLogLocator(minor=True))
    ax.yaxis.set_major_formatter(FuncFormatter(label))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average, length ``len(values) - window + 1`` ('valid')."""
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(np.asarray(values, dtype=np.float64), kernel, mode="valid")


def _series(history: Dict[str, List[float]], key: str) -> Optional[np.ndarray]:
    """Return ``history[key]`` as a float array, or ``None`` if absent/empty."""
    values = history.get(key)
    if not values:
        return None
    return np.asarray(values, dtype=np.float64)


def _class_labels(n: int, class_names: Optional[Sequence[str]]) -> List[str]:
    """Names for ``n`` classes; falls back to ``"0".."n-1"``."""
    if class_names is not None:
        if len(class_names) != n:
            raise ValueError(f"class_names has {len(class_names)} entries, expected {n}")
        return [str(c) for c in class_names]
    return [str(i) for i in range(n)]


def _align_zero_lines(
        ax_left: plt.Axes, left: Optional[np.ndarray],
        ax_right: plt.Axes, right: Optional[np.ndarray],
) -> None:
    """Give two y axes limits that put ``0`` at the same height on both.

    The generalization-gap panel has one axis per gap (loss, accuracy), each autoscaled
    on its own, so the two zero lines sat at different heights (about 11 px apart in run
    3c) and a curve could seem to cross or coincide with the other for no reason. The
    limits are ``[-f * R, (1 - f) * R]`` on each axis: zero at fraction ``f`` of the
    height, ``R`` the smallest range that holds that axis's data with 5% padding. ``f`` is
    the natural zero fraction of whichever axis wastes the least room when the other adopts
    it (clipped to ``[0.05, 0.95]``, so an all-positive series still gets a margin).

    Args:
        ax_left, ax_right: The two axes; each gets ``set_ylim``.
        left, right: The plotted values of each axis, or ``None`` when it has no data
            (its limits then just follow the other axis's zero fraction).
    """
    def extents(values: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
        if values is None or not np.isfinite(values).any():
            return None
        finite = values[np.isfinite(values)]
        below, above = max(0.0, -float(finite.min())), max(0.0, float(finite.max()))
        pad = 0.05 * max(below + above, 1e-12)
        return below + pad, above + pad

    both = [e for e in (extents(left), extents(right)) if e is not None]
    if not both:
        return

    def limits(e: Optional[Tuple[float, float]], f: float) -> Tuple[float, float]:
        span = 1.0 if e is None else max(e[0] / f, e[1] / (1.0 - f))
        return -f * span, (1.0 - f) * span

    def waste(f: float) -> float:
        return sum(max(b / f, a / (1.0 - f)) / (a + b) for b, a in both)

    fractions = [float(np.clip(b / (a + b), 0.05, 0.95)) for b, a in both]
    f = min(fractions, key=waste)
    ax_left.set_ylim(*limits(extents(left), f))
    ax_right.set_ylim(*limits(extents(right), f))


# ---------------------------------------------------------------------
# Training dashboard
# ---------------------------------------------------------------------

def render_training_dashboard(
        history: Dict[str, List[float]],
        out_path: PathLike,
        title: str = "",
        epoch_times: Optional[Sequence[float]] = None,
        baseline: Optional[Dict[str, float]] = None,
        baseline_label: str = BASELINE_LABEL,
) -> List[str]:
    """Render the per-epoch training dashboard to a single PNG.

    Panels, each drawn only when it has data: ``Loss``, ``Accuracy``,
    ``Learning rate`` (log axis; needs ``lr``), ``Generalization gap``
    (``val_loss - loss`` and ``accuracy - val_accuracy``), ``Per-epoch time``
    (needs ``epoch_times``) and ``Smoothed loss`` (only when at least
    ``SMOOTHED_PANEL_MIN_EPOCHS`` epochs exist). A ``Per-epoch time`` bar above
    ``TIME_CLIP_FACTOR`` times the median of epochs >= 2 (the XLA warmup epoch)
    is drawn at that ceiling, hatched, and annotated with its true value. The grid is built from the panels
    that exist, so a short run has no empty cells; a partly filled last row is
    centred. ``TRAIN_METRIC_CAPTION`` is written under the grid with ``fig.text``
    so it survives every panel layout.

    Args:
        history: Per-epoch lists with any of the keys ``loss``, ``val_loss``,
            ``accuracy``, ``val_accuracy``, ``lr``. Epoch ``i`` is plotted at
            x = ``i + 1``.
        out_path: PNG destination (parent directories are created).
        title: Figure suptitle.
        epoch_times: Wall-clock seconds per epoch.
        baseline: Epoch-0 metrics of the untrained model on the validation data
            (keys ``loss`` and/or ``accuracy``); drawn as a marker at x = 0 on the
            validation curves. A value above ``BASELINE_CLIP_FACTOR`` times the
            largest finite plotted value is drawn AT that ceiling (upward
            triangle) and annotated with its true value; anything at or below
            the ceiling is drawn exactly where it is, and so is a baseline
            when no plotted curve value is finite (there is no ceiling).
        baseline_label: Legend text of the baseline marker (the callback names the
            measurement mode, e.g. ``"epoch-0 baseline (val, training mode)"``).

    Returns:
        The titles of the panels drawn, in order (empty, and nothing written,
        when ``history`` has no epochs).
    """
    loss, val_loss = _series(history, "loss"), _series(history, "val_loss")
    acc, val_acc = _series(history, "accuracy"), _series(history, "val_accuracy")
    top5, val_top5 = _series(history, "top_5_accuracy"), _series(history, "val_top_5_accuracy")
    lr = _series(history, "lr")
    times = None if not epoch_times else np.asarray(epoch_times, dtype=np.float64)
    n_epochs = max((len(s) for s in (loss, val_loss, acc, val_acc) if s is not None), default=0)
    if n_epochs == 0:
        return []
    baseline = baseline or {}

    def _x(values: np.ndarray) -> np.ndarray:
        return np.arange(1, len(values) + 1)

    def _curves(ax, train, val, base_key: str, ylabel: str,
                extra: Optional[Callable[[Any], None]] = None) -> None:
        clipped = False
        if train is not None:
            ax.plot(_x(train), train, color=TRAIN_COLOR, lw=1.6, marker="o", ms=3, label="train")
        if val is not None:
            ax.plot(_x(val), val, color=VAL_COLOR, lw=1.6, marker="o", ms=3, label="val")
            if base_key in baseline and np.isfinite(baseline[base_key]):
                true_value = baseline[base_key]
                curve_values = np.concatenate(
                    [c[np.isfinite(c)] for c in (train, val) if c is not None])
                # No finite curve value (a diverged run) -> no ceiling to clip at.
                ceiling = (BASELINE_CLIP_FACTOR * float(curve_values.max())
                           if curve_values.size else np.inf)
                clipped = true_value > ceiling
                drawn = ceiling if clipped else true_value
                ax.plot([0, 1], [drawn, val[0]], color=VAL_COLOR, lw=1.0,
                        ls=":", alpha=0.7)
                ax.scatter([0], [drawn], marker="^" if clipped else "*",
                           s=70 if clipped else 110, color=VAL_COLOR,
                           edgecolor="black", zorder=5, label=baseline_label)
                if clipped:
                    ax.annotate(f"epoch-0: {true_value:.4g} (clipped)", xy=(0, drawn),
                                xytext=(8, 0), textcoords="offset points", va="center",
                                fontsize=8, color=VAL_COLOR)
        if extra is not None:
            extra(ax)  # before the legend, so its labelled lines join it
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        # 'best' does not see annotations: with a clipped baseline the annotation sits
        # top-left and the curves hug the bottom, so the legend goes to the empty
        # middle of the right edge.
        ax.legend(fontsize=8, loc="center right" if clipped else "best")

    def _loss(ax) -> None:
        _curves(ax, loss, val_loss, "loss", "loss (log scale)")
        ax.set_yscale("log")
        _plain_log_ticks(ax)

    def _accuracy(ax) -> None:
        # Top-5 (recorded by a run that compiles it, e.g. CIFAR-100): dashed, same colours.
        def top5_curves(ax_) -> None:
            for values, color, label in ((top5, TRAIN_COLOR, "train top-5"),
                                         (val_top5, VAL_COLOR, "val top-5")):
                if values is not None:
                    ax_.plot(_x(values), values, color=color, lw=1.4, ls="--", marker="s",
                             ms=2.5, label=label)

        _curves(ax, acc, val_acc, "accuracy", "accuracy", extra=top5_curves)

    def _lr(ax) -> None:
        """Log-scaled rate curve; see :data:`LR_PANEL_MAX_DECADES` and the sparse-tick label."""
        x = _x(lr)
        ax.plot(x, lr, color=ACCENT_COLOR, lw=1.6, marker="o", ms=3)
        ax.set_yscale("log")
        _plain_log_ticks(ax)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate (log scale)")
        positive = lr[np.isfinite(lr) & (lr > 0.0)]
        if positive.size == 0:
            return
        peak = float(positive.max())
        below = np.zeros(len(lr), dtype=bool)
        if peak / float(positive.min()) > 10.0 ** LR_PANEL_MAX_DECADES:
            floor = peak / 10.0 ** LR_PANEL_MAX_DECADES
            ax.set_ylim(floor, peak * 1.5)
            below = np.isfinite(lr) & (lr < floor)
            ax.scatter(x[below], np.full(int(below.sum()), floor), marker="v", s=40,
                       color=ACCENT_COLOR, edgecolor="black", zorder=5, clip_on=False)
            for k, (xi, value) in enumerate(zip(x[below], lr[below])):
                ax.annotate(f"epoch {int(xi)}: {value:.3g}, below axis", xy=(xi, floor),
                            xytext=(8, 6 + 12 * k), textcoords="offset points", fontsize=8,
                            color="#444444", va="bottom")
        lo, hi = ax.get_ylim()
        ticks = ax.yaxis.get_major_locator().tick_values(lo, hi)
        if int(np.count_nonzero((ticks >= lo) & (ticks <= hi))) < 2:
            # Too few labelled ticks to read the ends (a 2.1e-4 to 2e-3 cosine): name them.
            for xi, value, ha in ((x[0], lr[0], "left"), (x[-1], lr[-1], "right")):
                if np.isfinite(value) and value > 0.0 and lo <= value <= hi:
                    ax.annotate(f"{value:.3g}", xy=(xi, value), xytext=(0, 7),
                                textcoords="offset points", fontsize=8, color="#444444",
                                ha=ha, va="bottom")

    def _gap(ax) -> None:
        loss_gap = acc_gap = None
        if loss is not None and val_loss is not None:
            n = min(len(loss), len(val_loss))
            loss_gap = val_loss[:n] - loss[:n]
            ax.plot(np.arange(1, n + 1), loss_gap, color=VAL_COLOR, lw=1.6,
                    marker="o", ms=3, label="val_loss - loss")
        ax.axhline(0.0, color="grey", lw=0.8)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss gap", color=VAL_COLOR)
        if acc is not None and val_acc is not None:
            n = min(len(acc), len(val_acc))
            ax2 = ax.twinx()
            acc_gap = acc[:n] - val_acc[:n]
            ax2.plot(np.arange(1, n + 1), acc_gap, color=TRAIN_COLOR, lw=1.6,
                     marker="s", ms=3, ls="--", label="acc - val_acc")
            _align_zero_lines(ax, loss_gap, ax2, acc_gap)
            ax2.set_ylabel("accuracy gap", color=TRAIN_COLOR)
            ax2.grid(False)  # a global style may grid every axes; keep one grid
            lines = [ln for ln in ax.get_lines() + ax2.get_lines()
                     if not ln.get_label().startswith("_")]
            ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8)
        else:
            ax.legend(fontsize=8)

    def _time(ax) -> None:
        x = _x(times)
        drawn = times
        clipped = np.zeros(len(times), dtype=bool)
        if len(times) > 1:
            reference = float(np.median(times[1:]))  # epoch 1 carries the XLA warmup
            if np.isfinite(reference) and reference > 0.0:
                ceiling = TIME_CLIP_FACTOR * reference
                clipped = times > ceiling
                drawn = np.where(clipped, ceiling, times)
        ax.bar(x, drawn, color="#7f7f7f")
        if clipped.any():
            ax.bar(x[clipped], drawn[clipped], color="#d9d9d9", edgecolor="#7f7f7f", hatch="//")
            for xi, top, true_value in zip(x[clipped], drawn[clipped], times[clipped]):
                ax.annotate(f"{true_value:.4g} s (clipped)", xy=(xi, top), xytext=(0, 3),
                            textcoords="offset points", fontsize=8, color="#444444",
                            ha="left" if xi <= (len(times) + 1) / 2 else "right", va="bottom")
            ax.set_ylim(0.0, float(drawn.max()) * 1.25)  # room for the annotation
        ax.set_xlabel("epoch")
        ax.set_ylabel("seconds")

    def _smoothed(ax) -> None:
        window = max(2, min(5, n_epochs // 3))
        for values, color, label in ((loss, TRAIN_COLOR, "train"), (val_loss, VAL_COLOR, "val")):
            if values is None or len(values) < window:
                continue
            ax.plot(_x(values), values, color=color, lw=0.8, alpha=0.3)
            ax.plot(np.arange(window, len(values) + 1), _moving_average(values, window),
                    color=color, lw=2.0, label=f"{label} (MA-{window})")
        ax.set_yscale("log")
        _plain_log_ticks(ax)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss (log scale)")
        ax.legend(fontsize=8)

    panels: List[Tuple[str, Any]] = []
    if loss is not None or val_loss is not None:
        panels.append(("Loss", _loss))
    if acc is not None or val_acc is not None:
        panels.append(("Accuracy", _accuracy))
    if lr is not None:
        panels.append(("Learning rate", _lr))
    if (loss is not None and val_loss is not None) or (acc is not None and val_acc is not None):
        panels.append(("Generalization gap", _gap))
    if times is not None and len(times):
        panels.append(("Per-epoch time", _time))
    if n_epochs >= SMOOTHED_PANEL_MIN_EPOCHS and (loss is not None or val_loss is not None):
        panels.append(("Smoothed loss", _smoothed))

    n_cols = min(3, len(panels))
    n_rows = -(-len(panels) // n_cols)
    fig = plt.figure(figsize=(5.2 * n_cols, 3.9 * n_rows + 0.5))
    try:
        # Half-column grid: each panel spans 2 slots, so a short last row centres.
        grid = fig.add_gridspec(n_rows, 2 * n_cols, hspace=0.45, wspace=1.9)
        for r in range(n_rows):
            row_panels = panels[r * n_cols:(r + 1) * n_cols]
            start = n_cols - len(row_panels)
            for i, (name, draw) in enumerate(row_panels):
                ax = fig.add_subplot(grid[r, start + 2 * i:start + 2 * i + 2])
                draw(ax)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_title(name, fontsize=11)
                ax.grid(alpha=0.3)
        if title:
            fig.suptitle(title, fontsize=13, y=0.995)
        fig.text(0.5, 0.0, TRAIN_METRIC_CAPTION, ha="center", va="top", fontsize=9,
                 style="italic", color="#444444")
        _save_and_close(fig, out_path)
    finally:
        plt.close(fig)
    return [name for name, _ in panels]


class TrainingDashboardCallback(keras.callbacks.Callback):
    """Redraws the training dashboard on a cadence, overwriting one PNG.

    Keeps its own accumulated history (so it needs no ``History`` callback),
    including the learning rate, and its own per-epoch wall times (accumulated on
    EVERY epoch, drawn or not). If ``baseline_fn`` or ``baseline_data`` is given,
    the untrained model is measured in ``on_train_begin`` and drawn as the epoch-0
    marker: ``baseline_fn(model)`` when given (the trainer passes the SAME
    measurement its initial-loss guard uses, so the two cannot disagree under batch
    normalization), else ``model.evaluate`` on ``baseline_data``.
    Every evaluation and render is wrapped so a plotting failure logs a warning
    and never aborts training. Place it AFTER ``LearningRateLogger`` so
    ``logs['lr']`` exists; otherwise the rate is read off the optimizer.

    Draw schedule (see :data:`DASHBOARD_TARGET_DRAWS`): after epoch 1 (so the
    baseline and the first point exist), then after every
    ``max(1, planned_epochs // 20)``-th epoch, and once more in ``on_train_end`` if
    the last completed epoch was not already drawn (an early stop or a run whose
    length is off the cadence), so the file always shows the final state. The
    planned epoch count comes from ``self.params['epochs']``; if it is missing the
    dashboard is drawn every epoch.

    Args:
        out_path: PNG destination, overwritten on each draw.
        baseline_data: Optional ``(x_val, y_val)`` for the epoch-0 baseline
            (``model.evaluate``; ignored when ``baseline_fn`` is given).
        title: Figure suptitle.
        batch_size: Batch size for the ``baseline_data`` evaluation.
        baseline_fn: Optional ``model -> {"loss": ..., "accuracy": ...}`` returning
            the epoch-0 metrics; it must leave the model unchanged. This module
            imports nothing from the trainer, the trainer injects the function.
        baseline_mode: How ``baseline_fn`` measured (``"training"`` or
            ``"inference"``); a mode other than ``"inference"`` is named in the
            marker's legend text.

    Attributes:
        history: Accumulated per-epoch metrics (``loss``, ``val_loss``, ...,
            ``lr``).
        epoch_times: Wall seconds per completed epoch.
        baseline: Epoch-0 metrics dict, or ``None``.
    """

    def __init__(
            self,
            out_path: PathLike,
            baseline_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            title: str = "",
            batch_size: int = 1024,
            baseline_fn: Optional[Callable[[keras.Model], Dict[str, float]]] = None,
            baseline_mode: str = "inference",
    ) -> None:
        super().__init__()
        self.out_path = Path(out_path)
        self.baseline_data = baseline_data
        self.baseline_fn = baseline_fn
        self.baseline_label = (
            BASELINE_LABEL if baseline_mode == "inference"
            else f"epoch-0 baseline (val, {baseline_mode} mode)"
        )
        self.title = title
        self.batch_size = batch_size
        self.history: Dict[str, List[float]] = {}
        self.epoch_times: List[float] = []
        self.baseline: Optional[Dict[str, float]] = None
        self._epoch_start = 0.0
        # Number of completed epochs the PNG on disk shows (0 = nothing drawn yet).
        self._drawn_epochs = 0

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Measure the untrained model (epoch 0): ``baseline_fn`` or ``baseline_data``."""
        if self.baseline_fn is None and self.baseline_data is None:
            return
        try:
            if self.baseline_fn is not None:
                metrics = self.baseline_fn(self.model)
            else:
                x, y = self.baseline_data
                metrics = self.model.evaluate(
                    x, y, batch_size=self.batch_size, verbose=0, return_dict=True
                )
            self.baseline = {k: float(v) for k, v in metrics.items()}
        except Exception as e:  # noqa: BLE001 - never abort training for a plot
            logger.warning(f"Dashboard epoch-0 baseline failed: {e}")

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Start the epoch wall-clock."""
        self._epoch_start = time.perf_counter()

    def _current_lr(self, logs: Dict[str, Any]) -> Optional[float]:
        """Learning rate from ``logs['lr']`` or, failing that, the optimizer."""
        if logs.get("lr") is not None:
            return float(logs["lr"])
        try:
            return float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        except Exception:  # noqa: BLE001
            return None

    def _draw_interval(self) -> int:
        """Epochs between redraws: ``max(1, planned // 20)``, or 1 if unknown."""
        planned = (self.params or {}).get("epochs")
        try:
            return max(1, int(planned) // DASHBOARD_TARGET_DRAWS)
        except (TypeError, ValueError):
            return 1

    def _draw(self) -> None:
        """Render the dashboard from the accumulated state; never raises."""
        completed = len(self.epoch_times)
        try:
            render_training_dashboard(
                self.history, self.out_path, self.title,
                epoch_times=self.epoch_times, baseline=self.baseline,
                baseline_label=self.baseline_label,
            )
            self._drawn_epochs = completed
        except Exception as e:  # noqa: BLE001 - never abort training for a plot
            logger.warning(f"Dashboard render failed at epoch {completed}: {e}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Accumulate this epoch's logs; redraw if this epoch is on the cadence."""
        logs = logs or {}
        try:
            self.epoch_times.append(time.perf_counter() - self._epoch_start)
            for key in ("loss", "val_loss", "accuracy", "val_accuracy",
                        "top_5_accuracy", "val_top_5_accuracy"):
                if key in logs:
                    self.history.setdefault(key, []).append(float(logs[key]))
            lr = self._current_lr(logs)
            if lr is not None:
                self.history.setdefault("lr", []).append(lr)
        except Exception as e:  # noqa: BLE001 - never abort training for a plot
            logger.warning(f"Dashboard accumulation failed at epoch {epoch + 1}: {e}")
            return
        completed = len(self.epoch_times)
        if completed == 1 or completed % self._draw_interval() == 0:
            self._draw()

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Draw the final state unless the last completed epoch is already on disk."""
        completed = len(self.epoch_times)
        if completed and self._drawn_epochs != completed:
            self._draw()


# ---------------------------------------------------------------------
# Post-training evaluation figures
# ---------------------------------------------------------------------

def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[Sequence[str]],
        out_path: PathLike,
) -> np.ndarray:
    """Save a confusion-matrix figure whose pixel size is bounded for any class count.

    Up to :data:`MANY_CLASSES_THRESHOLD` classes: two panels, raw counts and
    row-normalized recall. More classes: the row-normalized heatmap without cell text
    (tick labels only for the classes in the top confusions) beside a panel of the
    :data:`CONFUSION_TOP_PAIRS` most confused ``true -> predicted`` pairs (count and rate),
    see :func:`_plot_confusion_many_classes`. The two-panel layout scaled as
    ``0.62 * n`` inches per panel: 100 classes made a 15585 x 7571 px image.

    The row-normalized panel leaves cells at or below ``CONFUSION_MIN_ANNOTATION``
    (0.5%) unannotated so it never prints "0%"; the counts panel annotates every
    cell. Grid lines are switched off explicitly on both image axes (measured:
    matplotlib's colorbar axes never show one, so they need no call):
    ``dl_techniques.visualization.core`` sets a whitegrid style at import
    (``axes.grid`` True process-wide), so the look must not depend on which
    module was imported first.

    Args:
        y_true: Integer true labels ``(N,)``.
        y_pred: Integer predicted labels ``(N,)``.
        class_names: One name per class (also fixes the class count), or ``None``
            to infer ``max(label) + 1`` classes named ``"0".."n-1"``.
        out_path: PNG destination.

    Returns:
        The ``(n, n)`` integer count matrix (rows true, columns predicted).
    """
    y_true, y_pred = np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)
    n = len(class_names) if class_names is not None else int(max(y_true.max(), y_pred.max())) + 1
    names = _class_labels(n, class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    acc = float(np.trace(cm)) / max(int(cm.sum()), 1)
    if n > MANY_CLASSES_THRESHOLD:
        _plot_confusion_many_classes(cm, norm, names, acc, out_path)
        return cm

    size = max(6.0, 0.62 * n + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(2 * size + 1.0, size))
    try:
        for ax, mat, label, fmt, floor in (
                (axes[0], cm, "Counts", lambda v: f"{int(v)}", -1.0),
                (axes[1], norm, "Row-normalized (recall per true class)",
                 lambda v: f"{v:.0%}", CONFUSION_MIN_ANNOTATION),
        ):
            im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(float(mat.max()), 1e-9))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            ax.grid(False)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_yticklabels(names)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(label)
            cut = mat.max() / 2.0
            for i in range(n):
                for j in range(n):
                    if mat[i, j] <= floor:
                        continue
                    ax.text(j, i, fmt(mat[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if mat[i, j] > cut else "black")
        fig.suptitle(f"Confusion matrix (accuracy {acc:.4f}, n={int(cm.sum())})", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _save_and_close(fig, out_path)
    finally:
        plt.close(fig)
    return cm


def _top_confused_pairs(cm: np.ndarray, k: int) -> List[Tuple[int, int, int, float]]:
    """The ``k`` most confused off-diagonal cells, largest count first.

    Args:
        cm: ``(n, n)`` count matrix, rows true, columns predicted.
        k: Maximum number of pairs.

    Returns:
        ``(true, predicted, count, rate)`` tuples, ``rate`` being the count as a fraction
        of the true class's samples; ties in count go to the higher rate, then the lower
        indices. Cells with a zero count are never listed (fewer than ``k`` when the
        classifier makes few distinct errors, an empty list for a perfect one).
    """
    n = cm.shape[0]
    rows, cols = np.nonzero(~np.eye(n, dtype=bool) & (cm > 0))
    counts = cm[rows, cols]
    rates = counts / np.maximum(cm.sum(axis=1)[rows], 1)
    order = np.lexsort((cols, rows, -rates, -counts))[:k]
    return [(int(rows[i]), int(cols[i]), int(counts[i]), float(rates[i])) for i in order]


def _plot_confusion_many_classes(
        cm: np.ndarray, norm: np.ndarray, names: List[str], acc: float, out_path: PathLike,
) -> None:
    """The bounded confusion figure for more than :data:`MANY_CLASSES_THRESHOLD` classes.

    Left: the row-normalized heatmap, no cell text, a power-law colour scale
    (:data:`CONFUSION_HEATMAP_GAMMA`) so off-diagonal cells show next to the diagonal, tick
    labels for the classes of the top pairs only (most confused first, never closer than
    ``ceil(n / CONFUSION_MAX_TICK_LABELS)`` classes). Right: horizontal bars of the top
    :data:`CONFUSION_TOP_PAIRS` pairs, each labelled ``true -> predicted`` with its count and
    its rate (share of the true class). The figure size is a constant.

    Args:
        cm: ``(n, n)`` count matrix.
        norm: ``cm`` divided by its row sums.
        names: One name per class.
        acc: Overall accuracy, for the title.
        out_path: PNG destination.
    """
    n = len(names)
    pairs = _top_confused_pairs(cm, CONFUSION_TOP_PAIRS)
    min_gap = -(-n // CONFUSION_MAX_TICK_LABELS)
    ticks: List[int] = []
    for true_i, pred_j, _, _ in pairs:
        for c in (true_i, pred_j):
            if c not in ticks and all(abs(c - t) >= min_gap for t in ticks):
                ticks.append(c)
    ticks.sort()

    fig = plt.figure(figsize=(17.0, 9.0))
    try:
        grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.4)
        ax = fig.add_subplot(grid[0, 0])
        im = ax.imshow(norm, cmap="Blues", interpolation="nearest",
                       norm=PowerNorm(CONFUSION_HEATMAP_GAMMA, vmin=0.0,
                                      vmax=max(float(norm.max()), 1e-9)))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.grid(False)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([names[t] for t in ticks], rotation=90, fontsize=7)
        ax.set_yticklabels([names[t] for t in ticks], fontsize=7)
        ax.set_xlabel("predicted (labelled: classes in the most confused pairs)")
        ax.set_ylabel("true")
        ax.set_title("Row-normalized (recall per true class)")

        bars = fig.add_subplot(grid[0, 1])
        if pairs:
            y = np.arange(len(pairs))[::-1]  # the most confused pair on top
            counts = np.array([c for _, _, c, _ in pairs])
            bars.barh(y, counts, color=TRAIN_COLOR, alpha=0.85)
            bars.set_yticks(y)
            bars.set_yticklabels([f"{names[t]} -> {names[p]}" for t, p, _, _ in pairs], fontsize=8)
            for yi, count, rate in zip(y, counts, [r for _, _, _, r in pairs]):
                bars.annotate(f"{int(count)} ({rate:.1%})", xy=(count, yi), xytext=(4, 0),
                              textcoords="offset points", va="center", fontsize=8)
            bars.set_xlim(0.0, float(counts.max()) * 1.25)
            bars.set_xlabel("samples of the true class predicted as the other class")
            bars.grid(False)
            bars.grid(axis="x", alpha=0.3)
            bars.set_axisbelow(True)
        else:
            bars.text(0.5, 0.5, "no misclassifications", ha="center", va="center",
                      transform=bars.transAxes)
            bars.set_xticks([])
            bars.set_yticks([])
        bars.set_title(f"Most confused pairs (top {len(pairs)}; count and share of the true class)")
        fig.suptitle(f"Confusion matrix (accuracy {acc:.4f}, n={int(cm.sum())}, {n} classes)",
                     fontsize=13)
        _save_and_close(fig, out_path)
    finally:
        plt.close(fig)


def plot_per_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[Sequence[str]],
        out_path: PathLike,
) -> Dict[str, Any]:
    """Save the per-class precision / recall / F1 chart.

    Up to :data:`MANY_CLASSES_THRESHOLD` classes: grouped vertical bars of all three, one
    group per class. More classes: horizontal bars of F1 for the
    :data:`PER_CLASS_SHOWN_EACH_END` best and worst classes (precision and recall as markers
    on the same row), the title naming how many of how many classes are shown; a 100-class
    grouped chart was an 8837 x 524 px strip. Either way the returned report holds EVERY
    class.

    The score axis starts at 0 unless every bar is at or above :data:`PER_CLASS_TRUNCATE_FROM`,
    in which case it is truncated and the axis label says so.

    Args:
        y_true: Integer true labels ``(N,)``.
        y_pred: Integer predicted labels ``(N,)``.
        class_names: One name per class, or ``None`` to infer.
        out_path: PNG destination.

    Returns:
        A JSON-serializable report ``{"per_class": {name: {precision, recall,
        f1, support}}, "macro_f1": float, "accuracy": float}``. Classes with no
        predictions score 0 (``zero_division=0``).
    """
    y_true, y_pred = np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)
    n = len(class_names) if class_names is not None else int(max(y_true.max(), y_pred.max())) + 1
    names = _class_labels(n, class_names)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n)), zero_division=0
    )

    if n > MANY_CLASSES_THRESHOLD:
        _plot_per_class_extremes(prec, rec, f1, support, names, out_path)
        return _per_class_report(names, prec, rec, f1, support, y_true, y_pred)

    fig, ax = plt.subplots(figsize=(max(8.0, 0.9 * n + 2.0), 4.8))
    try:
        x, width = np.arange(n), 0.27
        for offset, values, color, label in (
                (-width, prec, "#1f77b4", "precision"),
                (0.0, rec, "#ff7f0e", "recall"),
                (width, f1, "#2ca02c", "F1"),
        ):
            ax.bar(x + offset, values, width, color=color, label=label)
        lo = float(min(prec.min(), rec.min(), f1.min()))
        truncated = lo >= PER_CLASS_TRUNCATE_FROM
        ax.set_ylim(max(0.0, lo - 0.05) if truncated else 0.0, 1.005)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{nm}\n(n={int(s)})" for nm, s in zip(names, support)], fontsize=8)
        ax.set_ylabel("score (y-axis truncated)" if truncated else "score")
        ax.set_title(f"Per-class precision / recall / F1 (macro F1 {float(f1.mean()):.4f})")
        # ``grid(axis="y")`` alone leaves the x grid at the ambient rcParams value:
        # ``dl_techniques.visualization.core`` sets a whitegrid style at import, so
        # vertical lines ran through the bars. Off explicitly, horizontal grid behind.
        ax.grid(False)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
        _save_and_close(fig, out_path)
    finally:
        plt.close(fig)
    return _per_class_report(names, prec, rec, f1, support, y_true, y_pred)


def _per_class_report(
        names: List[str], prec: np.ndarray, rec: np.ndarray, f1: np.ndarray,
        support: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
) -> Dict[str, Any]:
    """The JSON-serializable per-class report of :func:`plot_per_class_metrics` (all classes)."""
    return {
        "per_class": {
            nm: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for nm, p, r, f, s in zip(names, prec, rec, f1, support)
        },
        "macro_f1": float(f1.mean()),
        "accuracy": float(np.mean(y_true == y_pred)),
    }


def _plot_per_class_extremes(
        prec: np.ndarray, rec: np.ndarray, f1: np.ndarray, support: np.ndarray,
        names: List[str], out_path: PathLike,
) -> None:
    """Horizontal F1 bars of the best and worst classes, for more than 20 classes.

    Rows run best (top) to worst (bottom), with a gap row reading ``... N more classes ...``
    between the two blocks when classes are left out; at most
    ``2 * PER_CLASS_SHOWN_EACH_END`` rows, so the figure height is bounded for any class
    count. Each row is a green F1 bar with precision and recall as markers. The axis starts
    at zero unless every DRAWN value is at or above :data:`PER_CLASS_TRUNCATE_FROM`
    (same rule and label as the grouped chart).

    Args:
        prec, rec, f1, support: Per-class arrays, length n.
        names: One name per class.
        out_path: PNG destination.
    """
    n = len(names)
    k = PER_CLASS_SHOWN_EACH_END
    order = np.argsort(-f1, kind="stable")  # best first; ties keep the lower class index first
    shown = order if n <= 2 * k else np.concatenate([order[:k], order[-k:]])
    gap = n > 2 * k
    rows = np.arange(len(shown), dtype=float)
    if gap:
        rows[k:] += 1.0  # one empty row between the blocks
    y = -rows  # best at the top

    fig, ax = plt.subplots(figsize=(11.0, 0.34 * (len(shown) + int(gap)) + 1.8))
    try:
        ax.barh(y, f1[shown], height=0.62, color="#2ca02c", alpha=0.75, label="F1")
        ax.scatter(prec[shown], y, marker="o", s=26, color="#1f77b4", zorder=4, label="precision")
        ax.scatter(rec[shown], y, marker="D", s=26, color="#ff7f0e", zorder=4, label="recall")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{names[i]} (n={int(support[i])})" for i in shown], fontsize=8)
        if gap:
            ax.text(0.5, -float(k), f"... {n - 2 * k} more classes ...", ha="center",
                    va="center", fontsize=9, color="#444444", style="italic",
                    transform=ax.get_yaxis_transform())
        lo = float(min(prec[shown].min(), rec[shown].min(), f1[shown].min()))
        truncated = lo >= PER_CLASS_TRUNCATE_FROM
        ax.set_xlim(max(0.0, lo - 0.05) if truncated else 0.0, 1.005)
        ax.set_ylim(y.min() - 0.7, y.max() + 0.7)
        ax.set_xlabel("score (axis truncated)" if truncated else "score")
        which = (f"{len(shown)} of {n} classes shown ({k} best and {k} worst by F1)" if gap
                 else f"{len(shown)} of {n} classes shown, best to worst by F1")
        ax.set_title(f"Per-class F1, precision, recall: {which}; macro F1 {float(f1.mean()):.4f}",
                     fontsize=11)
        ax.grid(False)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
        _save_and_close(fig, out_path)
    finally:
        plt.close(fig)


def plot_calibration(
        y_true: np.ndarray,
        probs: np.ndarray,
        out_path: PathLike,
        n_bins: int = 10,
) -> float:
    """Save a reliability diagram (with ECE) and a correct-vs-wrong confidence histogram.

    Confidence is the top-class probability. ECE is the bin-population-weighted
    mean of ``|accuracy - mean confidence|`` over ``n_bins`` equal-width bins.
    Bins holding fewer than :data:`CALIBRATION_MIN_BIN_SAMPLES` samples are drawn
    hatched, lighter and annotated with their ``n`` (their accuracy is one of a
    handful of values, not evidence); they still count in the ECE, weighted by
    their tiny population.

    Args:
        y_true: Integer true labels ``(N,)``.
        probs: Class probabilities ``(N, C)``.
        out_path: PNG destination.
        n_bins: Number of equal-width confidence bins.

    Returns:
        The expected calibration error.
    """
    y_true = np.asarray(y_true).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64)
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)

    bin_acc = np.full(n_bins, np.nan)
    bin_conf = np.full(n_bins, np.nan)
    weight = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            bin_acc[b], bin_conf[b] = correct[mask].mean(), conf[mask].mean()
            weight[b] = mask.mean()
            counts[b] = int(mask.sum())
    populated = weight > 0
    sparse = populated & (counts < CALIBRATION_MIN_BIN_SAMPLES)
    dense = populated & ~sparse
    ece = float(np.sum(weight[populated] * np.abs(bin_acc[populated] - bin_conf[populated])))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    try:
        centers = (edges[:-1] + edges[1:]) / 2
        width = 1.0 / n_bins
        ax1.bar(centers[dense], bin_acc[dense], width * 0.95, color=VAL_COLOR, alpha=0.8,
                edgecolor="black", label="accuracy in bin")
        if sparse.any():
            ax1.bar(centers[sparse], bin_acc[sparse], width * 0.95, color=VAL_COLOR, alpha=0.25,
                    edgecolor="black", hatch="//",
                    label=f"n < {CALIBRATION_MIN_BIN_SAMPLES} (not evidence)")
            for c, a, n_b in zip(centers[sparse], bin_acc[sparse], counts[sparse]):
                if a == 0.0:
                    # A hatched bar of height 0 is invisible; a hairline at the bin accuracy
                    # shows the bin exists (clip_on off so it is not half cut by the axis).
                    ax1.plot([c - width * 0.475, c + width * 0.475], [0.0, 0.0], color="black",
                             lw=1.5, solid_capstyle="butt", clip_on=False, zorder=4)
                high = a > 0.9  # keep the label inside the axes for a full-height bar
                ax1.text(c, a - 0.02 if high else a + 0.02, f"n={int(n_b)}", ha="center",
                         va="top" if high else "bottom", fontsize=8)
        ax1.plot([0, 1], [0, 1], "k--", lw=1.2, label="perfect calibration")
        ax1.text(0.04, 0.93, f"ECE = {ece:.4f}", transform=ax1.transAxes, fontsize=12,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1.02)
        ax1.set_xlabel("confidence (top-class probability)")
        ax1.set_ylabel("accuracy")
        ax1.set_title("Reliability diagram")
        # Upper left under the ECE box: the bars of a trained model crowd the right
        # side and a lower-right legend hid the n labels of the low-confidence bins.
        ax1.legend(loc="upper left", bbox_to_anchor=(0.02, 0.86), fontsize=9)
        ax1.grid(alpha=0.3)

        ax2.hist([conf[correct], conf[~correct]], bins=edges, stacked=False,
                 color=[VAL_COLOR, TRAIN_COLOR], label=[f"correct (n={int(correct.sum())})",
                                                        f"wrong (n={int((~correct).sum())})"])
        ax2.set_yscale("log")
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("confidence (top-class probability)")
        ax2.set_ylabel("samples (log)")
        ax2.set_title("Confidence: correct vs wrong")
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(alpha=0.3)
        _save_and_close(fig, out_path)
    finally:
        plt.close(fig)
    return ece


def plot_confident_errors(
        x: np.ndarray,
        y_true: np.ndarray,
        probs: np.ndarray,
        out_path: PathLike,
        image_shape: Optional[Tuple[int, ...]] = None,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        n: int = 16,
        class_names: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Save a grid of the ``n`` most-confident misclassifications.

    Inputs are un-standardized (``x * std + mean``, clipped to ``[0, 1]``) when
    ``mean`` and ``std`` are given, and shown in gray when the image has one
    channel. Each title reads ``true -> pred (p=<confidence>)``.

    ``x`` is either flat ``(N, H*W*C)`` (then ``image_shape`` is required and each
    row is reshaped to it) or already images ``(N, H, W, C)`` (used as they are;
    ``image_shape`` is not consulted).

    Args:
        x: Standardized inputs, flat ``(N, H*W*C)`` or NHWC ``(N, H, W, C)``.
        y_true: Integer true labels ``(N,)``.
        probs: Class probabilities ``(N, C)``.
        out_path: PNG destination.
        image_shape: ``(H, W, C)`` to reshape each row to; required for flat ``x``.
        mean: Per-channel mean ``(C,)`` used to standardize; ``None`` skips the
            un-standardization (``x`` is then taken to be in ``[0, 1]``).
        std: Per-channel std ``(C,)`` used to standardize; ``None`` skips it.
        n: Maximum number of errors shown.
        class_names: Optional class names for the titles.

    Returns:
        The path written, or ``None`` (nothing written, logged) if the model made
        no errors.

    Raises:
        ValueError: If ``x`` is flat and ``image_shape`` is ``None``.
    """
    x = np.asarray(x)
    if x.ndim < 4 and image_shape is None:
        raise ValueError("plot_confident_errors needs image_shape when x is flat (ndim < 4)")
    y_true = np.asarray(y_true).reshape(-1)
    probs = np.asarray(probs)
    pred = probs.argmax(axis=1)
    wrong = np.flatnonzero(pred != y_true)
    if wrong.size == 0:
        logger.info("No misclassifications; skipping the confident-errors figure.")
        return None
    top = wrong[np.argsort(-probs[wrong, pred[wrong]])[:n]]
    names = _class_labels(probs.shape[1], class_names)

    n_cols = min(8, len(top))
    n_rows = -(-len(top) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.1 * n_cols, 2.5 * n_rows + 0.6),
                             squeeze=False)
    try:
        for ax in axes.ravel():
            ax.axis("off")
        for ax, idx in zip(axes.ravel(), top):
            img = np.asarray(x[idx], dtype=np.float32)
            if x.ndim < 4:
                img = img.reshape(image_shape)
            if mean is not None and std is not None:
                img = img * np.asarray(std) + np.asarray(mean)
            img = np.clip(img, 0.0, 1.0)
            if img.shape[-1] == 1:
                ax.imshow(img[..., 0], cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(img)
            ax.set_title(f"{names[y_true[idx]]} -> {names[pred[idx]]}\n(p={probs[idx, pred[idx]]:.2f})",
                         fontsize=8)
        fig.suptitle(f"Most confident errors ({len(top)} of {wrong.size} misclassified)", fontsize=11)
        return _save_and_close(fig, out_path)
    finally:
        plt.close(fig)
