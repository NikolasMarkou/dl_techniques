"""
Plots for the PowerMLP classification trainer.

Matplotlib + numpy + scikit-learn only (no seaborn / pandas / tensorflow), so
importing this module allocates nothing. Every figure is drawn headless (Agg),
saved at ``dpi=120`` with a tight bounding box, and closed on every path.

Contents:

- :func:`render_training_dashboard` / :class:`TrainingDashboardCallback` - one
  overwritten multi-panel PNG of the per-epoch curves, redrawn every epoch, with
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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import: headless-safe

import keras  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # noqa: E402

from dl_techniques.utils.logger import logger  # noqa: E402

PathLike = Union[str, Path]

DPI = 120
TRAIN_COLOR = "#d62728"
VAL_COLOR = "#1f77b4"
ACCENT_COLOR = "#2ca02c"

# The smoothed-loss panel is only informative once there are enough epochs.
SMOOTHED_PANEL_MIN_EPOCHS = 6

# Dashboard caption: what the two curve families measure (they are NOT the same
# quantity, which makes the epoch-1 generalization gap look negative).
TRAIN_METRIC_CAPTION = "train: running mean over the epoch; val: end of epoch"

# The epoch-0 baseline marker is drawn at most this many times the largest finite
# plotted curve value; a larger baseline (an init loss of 218 over curves near 1)
# is clipped there and annotated with its true value, so it cannot flatten the axis.
BASELINE_CLIP_FACTOR = 4.0

# The row-normalized confusion-matrix panel annotates only cells strictly ABOVE this
# fraction. Strict, because 0.005 itself (5 of 1000) formats as "0%" (round half to
# even), the exact text this floor exists to remove.
CONFUSION_MIN_ANNOTATION = 0.005


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


# ---------------------------------------------------------------------
# Training dashboard
# ---------------------------------------------------------------------

def render_training_dashboard(
        history: Dict[str, List[float]],
        out_path: PathLike,
        title: str = "",
        epoch_times: Optional[Sequence[float]] = None,
        baseline: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Render the per-epoch training dashboard to a single PNG.

    Panels, each drawn only when it has data: ``Loss``, ``Accuracy``,
    ``Learning rate`` (log axis; needs ``lr``), ``Generalization gap``
    (``val_loss - loss`` and ``accuracy - val_accuracy``), ``Per-epoch time``
    (needs ``epoch_times``) and ``Smoothed loss`` (only when at least
    ``SMOOTHED_PANEL_MIN_EPOCHS`` epochs exist). The grid is built from the panels
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
            the ceiling is drawn exactly where it is.

    Returns:
        The titles of the panels drawn, in order (empty, and nothing written,
        when ``history`` has no epochs).
    """
    loss, val_loss = _series(history, "loss"), _series(history, "val_loss")
    acc, val_acc = _series(history, "accuracy"), _series(history, "val_accuracy")
    lr = _series(history, "lr")
    times = None if not epoch_times else np.asarray(epoch_times, dtype=np.float64)
    n_epochs = max((len(s) for s in (loss, val_loss, acc, val_acc) if s is not None), default=0)
    if n_epochs == 0:
        return []
    baseline = baseline or {}

    def _x(values: np.ndarray) -> np.ndarray:
        return np.arange(1, len(values) + 1)

    def _curves(ax, train, val, base_key: str, ylabel: str) -> None:
        clipped = False
        if train is not None:
            ax.plot(_x(train), train, color=TRAIN_COLOR, lw=1.6, marker="o", ms=3, label="train")
        if val is not None:
            ax.plot(_x(val), val, color=VAL_COLOR, lw=1.6, marker="o", ms=3, label="val")
            if base_key in baseline and np.isfinite(baseline[base_key]):
                true_value = baseline[base_key]
                curve_values = np.concatenate(
                    [c[np.isfinite(c)] for c in (train, val) if c is not None])
                ceiling = BASELINE_CLIP_FACTOR * float(curve_values.max())
                clipped = true_value > ceiling
                drawn = ceiling if clipped else true_value
                ax.plot([0, 1], [drawn, val[0]], color=VAL_COLOR, lw=1.0,
                        ls=":", alpha=0.7)
                ax.scatter([0], [drawn], marker="^" if clipped else "*",
                           s=70 if clipped else 110, color=VAL_COLOR,
                           edgecolor="black", zorder=5, label="epoch-0 baseline (val)")
                if clipped:
                    ax.annotate(f"epoch-0: {true_value:.4g} (clipped)", xy=(0, drawn),
                                xytext=(8, 0), textcoords="offset points", va="center",
                                fontsize=8, color=VAL_COLOR)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        # 'best' does not see annotations: with a clipped baseline the annotation sits
        # top-left and the curves hug the bottom, so the legend goes to the empty
        # middle of the right edge.
        ax.legend(fontsize=8, loc="center right" if clipped else "best")

    def _loss(ax) -> None:
        _curves(ax, loss, val_loss, "loss", "loss")
        ax.set_yscale("log")

    def _accuracy(ax) -> None:
        _curves(ax, acc, val_acc, "accuracy", "accuracy")

    def _lr(ax) -> None:
        ax.plot(_x(lr), lr, color=ACCENT_COLOR, lw=1.6, marker="o", ms=3)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")

    def _gap(ax) -> None:
        if loss is not None and val_loss is not None:
            n = min(len(loss), len(val_loss))
            ax.plot(np.arange(1, n + 1), val_loss[:n] - loss[:n], color=VAL_COLOR, lw=1.6,
                    marker="o", ms=3, label="val_loss - loss")
        ax.axhline(0.0, color="grey", lw=0.8)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss gap", color=VAL_COLOR)
        if acc is not None and val_acc is not None:
            n = min(len(acc), len(val_acc))
            ax2 = ax.twinx()
            ax2.plot(np.arange(1, n + 1), acc[:n] - val_acc[:n], color=TRAIN_COLOR, lw=1.6,
                     marker="s", ms=3, ls="--", label="acc - val_acc")
            ax2.set_ylabel("accuracy gap", color=TRAIN_COLOR)
            ax2.grid(False)  # a global style may grid every axes; keep one grid
            lines = [ln for ln in ax.get_lines() + ax2.get_lines()
                     if not ln.get_label().startswith("_")]
            ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8)
        else:
            ax.legend(fontsize=8)

    def _time(ax) -> None:
        ax.bar(_x(times), times, color="#7f7f7f")
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
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
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
    """Redraws the training dashboard after every epoch, overwriting one PNG.

    Keeps its own accumulated history (so it needs no ``History`` callback),
    including the learning rate, and its own per-epoch wall times. If
    ``baseline_data`` is given, the untrained model is evaluated on it in
    ``on_train_begin`` and drawn as the epoch-0 marker. Every evaluation and
    render is wrapped so a plotting failure logs a warning and never aborts
    training. Place it AFTER ``LearningRateLogger`` so ``logs['lr']`` exists;
    otherwise the rate is read off the optimizer.

    Args:
        out_path: PNG destination, overwritten each epoch.
        baseline_data: Optional ``(x_val, y_val)`` for the epoch-0 baseline.
        title: Figure suptitle.
        batch_size: Batch size for the baseline evaluation.

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
    ) -> None:
        super().__init__()
        self.out_path = Path(out_path)
        self.baseline_data = baseline_data
        self.title = title
        self.batch_size = batch_size
        self.history: Dict[str, List[float]] = {}
        self.epoch_times: List[float] = []
        self.baseline: Optional[Dict[str, float]] = None
        self._epoch_start = 0.0

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Evaluate the untrained model on ``baseline_data`` (epoch 0)."""
        if self.baseline_data is None:
            return
        try:
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

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Accumulate this epoch's logs and redraw the dashboard."""
        logs = logs or {}
        try:
            self.epoch_times.append(time.perf_counter() - self._epoch_start)
            for key in ("loss", "val_loss", "accuracy", "val_accuracy"):
                if key in logs:
                    self.history.setdefault(key, []).append(float(logs[key]))
            lr = self._current_lr(logs)
            if lr is not None:
                self.history.setdefault("lr", []).append(lr)
            render_training_dashboard(
                self.history, self.out_path, self.title,
                epoch_times=self.epoch_times, baseline=self.baseline,
            )
        except Exception as e:  # noqa: BLE001 - never abort training for a plot
            logger.warning(f"Dashboard render failed at epoch {epoch + 1}: {e}")


# ---------------------------------------------------------------------
# Post-training evaluation figures
# ---------------------------------------------------------------------

def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[Sequence[str]],
        out_path: PathLike,
) -> np.ndarray:
    """Save a two-panel confusion matrix: raw counts and row-normalized recall.

    The row-normalized panel leaves cells at or below ``CONFUSION_MIN_ANNOTATION``
    (0.5%) unannotated so it never prints "0%"; the counts panel annotates every
    cell.

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
        acc = float(np.trace(cm)) / max(int(cm.sum()), 1)
        fig.suptitle(f"Confusion matrix (accuracy {acc:.4f}, n={int(cm.sum())})", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _save_and_close(fig, out_path)
    finally:
        plt.close(fig)
    return cm


def plot_per_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[Sequence[str]],
        out_path: PathLike,
) -> Dict[str, Any]:
    """Save grouped bars of precision / recall / F1 per class.

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
        ax.set_ylim(max(0.0, lo - 0.05), 1.005)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{nm}\n(n={int(s)})" for nm, s in zip(names, support)], fontsize=8)
        ax.set_ylabel("score (y-axis truncated)")
        ax.set_title(f"Per-class precision / recall / F1 (macro F1 {float(f1.mean()):.4f})")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
        _save_and_close(fig, out_path)
    finally:
        plt.close(fig)
    return {
        "per_class": {
            nm: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for nm, p, r, f, s in zip(names, prec, rec, f1, support)
        },
        "macro_f1": float(f1.mean()),
        "accuracy": float(np.mean(y_true == y_pred)),
    }


def plot_calibration(
        y_true: np.ndarray,
        probs: np.ndarray,
        out_path: PathLike,
        n_bins: int = 10,
) -> float:
    """Save a reliability diagram (with ECE) and a correct-vs-wrong confidence histogram.

    Confidence is the top-class probability. ECE is the bin-population-weighted
    mean of ``|accuracy - mean confidence|`` over ``n_bins`` equal-width bins.

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
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            bin_acc[b], bin_conf[b] = correct[mask].mean(), conf[mask].mean()
            weight[b] = mask.mean()
    populated = weight > 0
    ece = float(np.sum(weight[populated] * np.abs(bin_acc[populated] - bin_conf[populated])))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    try:
        centers = (edges[:-1] + edges[1:]) / 2
        width = 1.0 / n_bins
        ax1.bar(centers[populated], bin_acc[populated], width * 0.95, color=VAL_COLOR, alpha=0.8,
                edgecolor="black", label="accuracy in bin")
        ax1.plot([0, 1], [0, 1], "k--", lw=1.2, label="perfect calibration")
        ax1.text(0.04, 0.93, f"ECE = {ece:.4f}", transform=ax1.transAxes, fontsize=12,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1.02)
        ax1.set_xlabel("confidence (top-class probability)")
        ax1.set_ylabel("accuracy")
        ax1.set_title("Reliability diagram")
        ax1.legend(loc="lower right", fontsize=9)
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
        image_shape: Tuple[int, ...],
        mean: np.ndarray,
        std: np.ndarray,
        n: int = 16,
        class_names: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Save a grid of the ``n`` most-confident misclassifications.

    Inputs are un-standardized (``x * std + mean``, clipped to ``[0, 1]``) and
    shown in gray when the image has one channel. Each title reads
    ``true -> pred (p=<confidence>)``.

    Args:
        x: Standardized flattened inputs ``(N, H*W*C)``.
        y_true: Integer true labels ``(N,)``.
        probs: Class probabilities ``(N, C)``.
        out_path: PNG destination.
        image_shape: ``(H, W, C)`` to reshape each row to.
        mean: Per-channel mean ``(C,)`` used to standardize.
        std: Per-channel std ``(C,)`` used to standardize.
        n: Maximum number of errors shown.
        class_names: Optional class names for the titles.

    Returns:
        The path written, or ``None`` (nothing written, logged) if the model made
        no errors.
    """
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
            img = np.asarray(x[idx], dtype=np.float32).reshape(image_shape)
            img = np.clip(img * np.asarray(std) + np.asarray(mean), 0.0, 1.0)
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
