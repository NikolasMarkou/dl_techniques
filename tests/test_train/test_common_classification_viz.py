"""Guards for the many-class figures and the dashboard fixes of the run-3 audit (H1-H7).

Each guard was proven RED by injecting the defect it names (``findings/red-proofs-iter3.md``,
step 18):

- H1 the per-class chart of a many-class run is bounded in pixels and shows only the worst
  and best classes, saying how many of how many
- H2 the confusion figure of a many-class run is bounded in pixels, has no cell text, and
  lists the most confused pairs with count and rate
- H4 top-5 accuracy joins the Accuracy panel (dashed) when the history has it
- H5 a warmup learning-rate panel is floored at peak / 1e3 with the points below marked and
  annotated, and a sparse tick set gets its two ends labelled
- H6 the generalization-gap panel's two zero lines sit at the same height
- H7 log-scaled axes say so; a hatched calibration bin of accuracy 0 draws a hairline

The figures are read from the live ``Figure`` a spy captures inside ``_save_and_close`` (the
artists survive the close); saved-image sizes are read back from the PNG.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

import train.common.classification_viz as viz  # noqa: E402

# A figure a reader must be able to open: the many-class figures stay under this in both
# directions (the 100-class figures were 15585 x 7571 and 8837 x 524 px).
MAX_PIXELS = 3000


@pytest.fixture
def figures(monkeypatch):
    """Every figure about to be saved, with the canvas drawn (ticks and transforms exist)."""
    seen = []
    real = viz._save_and_close

    def spy(fig, out_path):
        fig.canvas.draw()
        seen.append(fig)
        return real(fig, out_path)

    monkeypatch.setattr(viz, "_save_and_close", spy)
    return seen


def _by_title(fig, prefix):
    """The one axes whose title starts with ``prefix``."""
    (ax,) = [a for a in fig.axes if a.get_title().startswith(prefix)]
    return ax


def _size(path):
    with Image.open(path) as image:
        return image.size


def _predictions(n_classes, n_samples=1500, seed=0):
    """Synthetic labels and predictions; class ``i`` is right with probability rising in ``i``."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n_samples)
    skill = np.linspace(0.15, 0.95, n_classes)
    pred = np.where(rng.random(n_samples) < skill[y], y, rng.integers(0, n_classes, n_samples))
    return y, pred


# ---------------------------------------------------------------------
# H1: per-class chart
# ---------------------------------------------------------------------


@pytest.mark.parametrize("n_classes", [21, 100, 1000])
def test_the_many_class_per_class_chart_is_bounded_in_pixels(tmp_path, n_classes) -> None:
    y, pred = _predictions(n_classes, n_samples=max(1500, 3 * n_classes))
    out = tmp_path / "pc.png"
    report = viz.plot_per_class_metrics(y, pred, [f"c{i}" for i in range(n_classes)], out)
    width, height = _size(out)
    assert width <= MAX_PIXELS and height <= MAX_PIXELS, (width, height)
    assert len(report["per_class"]) == n_classes, "the full table is still returned"


def test_the_many_class_per_class_chart_shows_the_worst_and_best_and_says_how_many(
        tmp_path, figures) -> None:
    n = 100
    # Class i has F1 = i / 99 exactly: i correct of 99 samples with no false positives.
    y = np.concatenate([np.full(99, i) for i in range(n)])
    pred = y.copy()
    for i in range(n):
        wrong = np.flatnonzero(y == i)[: 99 - i]
        pred[wrong] = (i + 1) % n if i + 1 < n else 0  # errors go to another class
    names = [f"c{i:02d}" for i in range(n)]
    viz.plot_per_class_metrics(y, pred, names, tmp_path / "pc.png")
    (fig,) = figures
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    shown = {label.split(" ")[0] for label in labels}
    assert len(shown) == 2 * viz.PER_CLASS_SHOWN_EACH_END == 30, sorted(shown)
    worst = {label.split(" ")[0] for label in labels[-viz.PER_CLASS_SHOWN_EACH_END:]}
    best = {label.split(" ")[0] for label in labels[:viz.PER_CLASS_SHOWN_EACH_END]}
    assert "c50" not in shown, "a middle class is not drawn"
    assert best.isdisjoint(worst)
    # Best first: the top row is the class with the highest F1.
    f1 = {nm: v["f1"] for nm, v in viz.plot_per_class_metrics(
        y, pred, names, tmp_path / "pc2.png")["per_class"].items()}
    assert max(f1[nm] for nm in worst) < min(f1[nm] for nm in best), "worst rows below best rows"
    assert f1[labels[0].split(" ")[0]] == max(f1.values())
    assert f1[labels[-1].split(" ")[0]] == min(f1.values())
    title = ax.get_title()
    assert "30 of 100" in title and "macro F1" in title, title
    assert [t.get_text() for t in ax.texts] == ["... 70 more classes ..."]


def test_between_the_threshold_and_twice_the_end_count_every_class_is_shown(
        tmp_path, figures) -> None:
    n = 25
    y, pred = _predictions(n)
    viz.plot_per_class_metrics(y, pred, [f"c{i}" for i in range(n)], tmp_path / "pc.png")
    (fig,) = figures
    ax = fig.axes[0]
    assert len(ax.get_yticklabels()) == n
    assert "25 of 25" in ax.get_title(), ax.get_title()
    assert not any("more classes" in t.get_text() for t in ax.texts)


@pytest.mark.parametrize("n_classes,grouped", [(20, True), (21, False)])
def test_the_per_class_layout_switches_just_above_the_threshold(
        tmp_path, figures, n_classes, grouped) -> None:
    """20 classes keep the grouped vertical bars (3 per class); 21 switch to the ranked chart."""
    y, pred = _predictions(n_classes)
    viz.plot_per_class_metrics(y, pred, [f"c{i}" for i in range(n_classes)], tmp_path / "pc.png")
    (fig,) = figures
    ax = fig.axes[0]
    assert viz.MANY_CLASSES_THRESHOLD == 20
    if grouped:
        assert ax.get_title().startswith("Per-class precision / recall / F1 (macro F1")
        assert len(ax.patches) == 3 * n_classes
    else:
        assert "classes shown" in ax.get_title()


# ---------------------------------------------------------------------
# H2: confusion figure
# ---------------------------------------------------------------------


@pytest.mark.parametrize("n_classes", [21, 100, 1000])
def test_the_many_class_confusion_figure_is_bounded_in_pixels(tmp_path, n_classes) -> None:
    y, pred = _predictions(n_classes, n_samples=max(1500, 3 * n_classes))
    out = tmp_path / "cm.png"
    cm = viz.plot_confusion_matrix(y, pred, [f"c{i}" for i in range(n_classes)], out)
    width, height = _size(out)
    assert width <= MAX_PIXELS and height <= MAX_PIXELS, (width, height)
    assert cm.shape == (n_classes, n_classes) and int(cm.sum()) == len(y)


def _figure_from_counts(counts):
    y_true, y_pred = [], []
    for i, row in enumerate(counts):
        for j, count in enumerate(row):
            y_true += [i] * int(count)
            y_pred += [j] * int(count)
    return np.array(y_true), np.array(y_pred)


def test_the_many_class_confusion_figure_has_no_cell_text_and_lists_the_top_pairs(
        tmp_path, figures) -> None:
    n = 30
    counts = np.diag(np.full(n, 40))
    counts[3, 7] = 25      # the most confused pair: 25 of 65 samples of class 3
    counts[12, 4] = 10
    counts[20, 21] = 5
    y_true, y_pred = _figure_from_counts(counts)
    names = [f"cls{i}" for i in range(n)]
    viz.plot_confusion_matrix(y_true, y_pred, names, tmp_path / "cm.png")
    (fig,) = figures
    heat = _by_title(fig, "Row-normalized")
    assert list(heat.texts) == [], "no cell text on the heatmap"
    pairs = _by_title(fig, "Most confused pairs")
    labels = [t.get_text() for t in pairs.get_yticklabels()]
    assert labels[0] == "cls3 -> cls7" and labels[1] == "cls12 -> cls4", labels
    assert labels[:3] == ["cls3 -> cls7", "cls12 -> cls4", "cls20 -> cls21"]
    assert len(labels) == 3, "only pairs with errors are listed"
    notes = [t.get_text() for t in pairs.texts]
    assert notes == ["25 (38.5%)", "10 (20.0%)", "5 (11.1%)"], notes
    # The top of the panel is the biggest bar (labels[0] is drawn last, at the top).
    tick_y = list(pairs.get_yticks())
    assert tick_y[0] > tick_y[1] > tick_y[2]
    # Heatmap tick labels: only classes of the listed pairs.
    ticks = {t.get_text() for t in heat.get_xticklabels()}
    assert -(-n // viz.CONFUSION_MAX_TICK_LABELS) == 1, "30 classes: no tick spacing needed"
    assert ticks == {"cls3", "cls7", "cls12", "cls4", "cls20", "cls21"}, ticks
    assert {t.get_text() for t in heat.get_yticklabels()} == ticks


def test_the_many_class_confusion_lists_at_most_the_top_pairs_and_marks_a_perfect_run(
        tmp_path, figures) -> None:
    n = 40
    rng = np.random.default_rng(1)
    y = rng.integers(0, n, 4000)
    pred = np.where(rng.random(4000) < 0.5, y, rng.integers(0, n, 4000))
    viz.plot_confusion_matrix(y, pred, [f"c{i}" for i in range(n)], tmp_path / "cm.png")
    pairs = _by_title(figures[0], "Most confused pairs")
    assert len(pairs.get_yticklabels()) == viz.CONFUSION_TOP_PAIRS == 15
    assert f"top {viz.CONFUSION_TOP_PAIRS}" in pairs.get_title()
    # A perfect classifier: a figure is still written and says so.
    perfect = np.arange(n).repeat(3)
    out = tmp_path / "perfect.png"
    viz.plot_confusion_matrix(perfect, perfect, [f"c{i}" for i in range(n)], out)
    assert out.is_file()
    assert [t.get_text() for t in _by_title(figures[1], "Most confused pairs").texts] == [
        "no misclassifications"]


def test_top_confused_pairs_orders_by_count_then_rate_and_skips_the_diagonal() -> None:
    cm = np.zeros((4, 4), dtype=int)
    cm[0, 0], cm[1, 1] = 100, 100          # diagonals never listed
    cm[0, 1] = 5                           # rate 5 / 105
    cm[2, 3] = 5                           # rate 5 / 5 = 1.0 -> outranks (0, 1) on the tie
    cm[3, 0] = 9
    assert viz._top_confused_pairs(cm, 2) == [(3, 0, 9, 1.0), (2, 3, 5, 1.0)]
    assert [p[:2] for p in viz._top_confused_pairs(cm, 10)] == [(3, 0), (2, 3), (0, 1)]
    assert viz._top_confused_pairs(np.diag([3, 3, 3]), 5) == []


def test_the_confusion_figure_of_ten_classes_keeps_its_two_panel_layout(
        tmp_path, figures) -> None:
    """Unchanged for 20 or fewer classes: counts + row-normalized, text in the cells, and the
    size the layout always had (2208 x 1044 px for 10 classes, within a few percent)."""
    y, pred = _predictions(10)
    out = tmp_path / "cm.png"
    viz.plot_confusion_matrix(y, pred, list("abcdefghij"), out)
    (fig,) = figures
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles == ["Counts", "Row-normalized (recall per true class)"]
    assert len(_by_title(fig, "Counts").texts) == 100
    width, height = _size(out)
    assert 2100 <= width <= 2350 and 950 <= height <= 1150, (width, height)


def test_the_twenty_class_confusion_figure_keeps_the_two_panel_layout_within_bounds(
        tmp_path, figures) -> None:
    y, pred = _predictions(20)
    out = tmp_path / "cm20.png"
    viz.plot_confusion_matrix(y, pred, [f"c{i}" for i in range(20)], out)
    assert [ax.get_title() for ax in figures[0].axes if ax.get_title()] == [
        "Counts", "Row-normalized (recall per true class)"]
    width, height = _size(out)
    assert width <= 4000 and height <= 2000, (width, height)


# ---------------------------------------------------------------------
# H3-adjacent: the figures take real names
# ---------------------------------------------------------------------


def test_the_many_class_figures_carry_the_class_names(tmp_path, figures) -> None:
    names = [f"name_{i}" for i in range(50)]
    y, pred = _predictions(50)
    viz.plot_per_class_metrics(y, pred, names, tmp_path / "pc.png")
    assert all(t.get_text().startswith("name_") for t in figures[0].axes[0].get_yticklabels())


# ---------------------------------------------------------------------
# Dashboard helpers
# ---------------------------------------------------------------------


def _history(n=5, **extra):
    e = np.arange(1, n + 1, dtype=np.float64)
    history = {
        "loss": (3.0 / e ** 0.5).tolist(), "val_loss": (3.2 / e ** 0.5).tolist(),
        "accuracy": (0.1 + 0.08 * e).tolist(), "val_accuracy": (0.09 + 0.06 * e).tolist(),
    }
    history.update(extra)
    return history


def _dashboard(tmp_path, figures, history, **kwargs):
    viz.render_training_dashboard(history, tmp_path / "d.png", "t", **kwargs)
    return figures[-1]


def _panel(fig, title):
    (ax,) = [a for a in fig.axes if a.get_title() == title]
    return ax


# ---- H4


def test_top5_accuracy_is_drawn_dashed_in_the_accuracy_panel_when_recorded(
        tmp_path, figures) -> None:
    history = _history(top_5_accuracy=[0.3, 0.4, 0.5, 0.6, 0.7],
                       val_top_5_accuracy=[0.28, 0.38, 0.48, 0.55, 0.62])
    ax = _panel(_dashboard(tmp_path, figures, history), "Accuracy")
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert {"train", "val", "train top-5", "val top-5"} <= set(legend), legend
    dashed = {ln.get_label(): (ln.get_linestyle(), ln.get_color()) for ln in ax.get_lines()
              if "top-5" in ln.get_label()}
    assert dashed == {"train top-5": ("--", viz.TRAIN_COLOR), "val top-5": ("--", viz.VAL_COLOR)}
    top1 = [ln for ln in ax.get_lines() if ln.get_label() in ("train", "val")]
    assert all(ln.get_linestyle() == "-" for ln in top1)
    assert list(next(ln for ln in ax.get_lines() if ln.get_label() == "val top-5").get_ydata()) == \
        history["val_top_5_accuracy"]


def test_the_accuracy_panel_is_unchanged_without_top5(tmp_path, figures) -> None:
    ax = _panel(_dashboard(tmp_path, figures, _history()), "Accuracy")
    assert not any("top-5" in ln.get_label() for ln in ax.get_lines())
    assert {t.get_text() for t in ax.get_legend().get_texts()} == {"train", "val"}


def test_the_dashboard_callback_accumulates_top5(tmp_path) -> None:
    callback = viz.TrainingDashboardCallback(tmp_path / "d.png")
    callback.on_epoch_begin(0)
    callback.on_epoch_end(0, {"loss": 1.0, "accuracy": 0.5, "top_5_accuracy": 0.8,
                              "val_top_5_accuracy": 0.7, "lr": 1e-3})
    assert callback.history["top_5_accuracy"] == [0.8]
    assert callback.history["val_top_5_accuracy"] == [0.7]


# ---- H5


WARMUP_LR = [1e-8, 1e-3, 8.5e-4, 5e-4, 1.5e-4]


def test_a_warmup_learning_rate_panel_is_floored_and_marks_the_point_below(
        tmp_path, figures) -> None:
    ax = _panel(_dashboard(tmp_path, figures, _history(lr=WARMUP_LR)), "Learning rate")
    lo, hi = ax.get_ylim()
    assert lo == pytest.approx(1e-3 / 10.0 ** viz.LR_PANEL_MAX_DECADES), (lo, hi)
    assert hi > 1e-3
    assert viz.LR_PANEL_MAX_DECADES == 3.0
    texts = [t.get_text() for t in ax.texts]
    assert "epoch 1: 1e-08, below axis" in texts, texts
    marker_xy = np.concatenate([c.get_offsets() for c in ax.collections])
    assert marker_xy.tolist() == [[1.0, pytest.approx(lo)]], marker_xy
    assert ax.get_ylabel() == "learning rate (log scale)"


def test_every_point_below_the_floor_is_annotated_with_its_own_value(
        tmp_path, figures) -> None:
    history = _history(n=6, lr=[1e-9, 1e-7, 1e-3, 9e-4, 6e-4, 2e-4])
    ax = _panel(_dashboard(tmp_path, figures, history), "Learning rate")
    texts = [t.get_text() for t in ax.texts]
    assert "epoch 1: 1e-09, below axis" in texts and "epoch 2: 1e-07, below axis" in texts, texts
    assert len(ax.collections[0].get_offsets()) == 2


def test_a_learning_rate_range_within_three_decades_is_not_floored(tmp_path, figures) -> None:
    ax = _panel(_dashboard(tmp_path, figures, _history(lr=[1e-5, 1e-3, 5e-4, 2e-4, 5e-5])),
                "Learning rate")
    assert ax.get_ylim()[0] < 1e-5 * 1.01, ax.get_ylim()
    assert not any("below axis" in t.get_text() for t in ax.texts)
    assert len(ax.collections) == 0


def _visible_y_labels(ax):
    lo, hi = ax.get_ylim()
    return [t.get_text() for t, loc in zip(ax.yaxis.get_majorticklabels(),
                                            ax.yaxis.get_majorticklocs())
            if lo <= loc <= hi and t.get_text()]


def test_the_run_3d_learning_rate_range_has_its_two_ends_labelled(tmp_path, figures) -> None:
    """3d: a cosine from 2e-3 down to 2.1e-4 had a single tick label (0.001), so neither end
    could be read; the first and last values are now written on the panel."""
    ax = _panel(_dashboard(tmp_path, figures, _history(lr=[2e-3, 1.8e-3, 1.2e-3, 6e-4, 2.1e-4])),
                "Learning rate")
    assert _visible_y_labels(ax) == ["0.001"], _visible_y_labels(ax)
    assert [t.get_text() for t in ax.texts] == ["0.002", "0.00021"]


def test_a_sparse_tick_set_gets_its_first_and_last_value_labelled(tmp_path, figures) -> None:
    """Ratio 12.5: only the 1e-4 decade tick falls in view, so the two ends of the curve are
    named on the panel (first value at its point, last value at its point)."""
    ax = _panel(_dashboard(tmp_path, figures, _history(lr=[5e-4, 3e-4, 1.5e-4, 8e-5, 4e-5])),
                "Learning rate")
    assert len(_visible_y_labels(ax)) < 2, _visible_y_labels(ax)
    texts = [t.get_text() for t in ax.texts]
    assert texts == ["0.0005", "4e-05"], texts


def test_a_dense_tick_set_gets_no_extra_end_labels(tmp_path, figures) -> None:
    ax = _panel(_dashboard(tmp_path, figures, _history(lr=[1e-3, 6e-4, 2e-4, 8e-5, 1e-5])),
                "Learning rate")
    assert list(ax.texts) == [], [t.get_text() for t in ax.texts]


# ---- H6


def test_the_generalization_gap_zero_lines_sit_at_the_same_height(tmp_path, figures) -> None:
    """A loss gap of -0.1 to 2.0 and an accuracy gap of -0.05 to 0.3 have different natural
    zero fractions (5% and 14%); the two axes must still put 0 at one height."""
    history = {"loss": [2.0, 1.5, 1.0, 0.8, 0.6], "val_loss": [1.9, 1.8, 2.6, 2.4, 2.8],
               "accuracy": [0.5, 0.6, 0.7, 0.8, 0.9], "val_accuracy": [0.55, 0.6, 0.6, 0.55, 0.6]}
    fig = _dashboard(tmp_path, figures, history)
    left = _panel(fig, "Generalization gap")
    (right,) = [a for a in fig.axes if a.get_title() == "" and a.bbox.bounds == left.bbox.bounds]
    zero_left = left.transData.transform((0, 0))[1]
    zero_right = right.transData.transform((0, 0))[1]
    assert abs(zero_left - zero_right) < 1.0, (zero_left, zero_right)
    # Both curves are still fully inside their axes.
    for ax in (left, right):
        lo, hi = ax.get_ylim()
        for ln in ax.get_lines():
            y = np.asarray(ln.get_ydata())
            if len(y) > 2:
                assert lo <= y.min() and y.max() <= hi, (lo, hi, y)


def test_the_gap_zero_lines_align_for_an_all_positive_series_too(tmp_path, figures) -> None:
    history = {"loss": [1.0, 0.9, 0.8], "val_loss": [1.2, 1.3, 1.5],
               "accuracy": [0.6, 0.7, 0.8], "val_accuracy": [0.5, 0.55, 0.6]}
    fig = _dashboard(tmp_path, figures, history)
    left = _panel(fig, "Generalization gap")
    (right,) = [a for a in fig.axes if a.get_title() == "" and a.bbox.bounds == left.bbox.bounds]
    assert abs(left.transData.transform((0, 0))[1] - right.transData.transform((0, 0))[1]) < 1.0


# ---- H7


def test_log_scaled_loss_axes_say_so(tmp_path, figures) -> None:
    fig = _dashboard(tmp_path, figures, _history(n=8))
    for title in ("Loss", "Smoothed loss"):
        ax = _panel(fig, title)
        assert ax.get_yscale() == "log"
        assert ax.get_ylabel() == "loss (log scale)", (title, ax.get_ylabel())


def _reliability(groups):
    y, probs = [], []
    for n, conf, n_correct in groups:
        y += [0] * n_correct + [1] * (n - n_correct)
        probs += [[conf, 1.0 - conf]] * n
    return np.array(y), np.array(probs)


def test_a_hatched_calibration_bin_of_accuracy_zero_draws_a_hairline(tmp_path, figures) -> None:
    y, probs = _reliability([(150, 0.97, 147), (3, 0.65, 0), (2, 0.55, 1)])
    viz.plot_calibration(y, probs, tmp_path / "cal.png")
    ax = _by_title(figures[0], "Reliability diagram")
    hairlines = [ln for ln in ax.get_lines()
                 if len(ln.get_ydata()) == 2 and list(ln.get_ydata()) == [0.0, 0.0]]
    assert len(hairlines) == 1, "exactly the 3-sample bin of accuracy 0"
    x = np.asarray(hairlines[0].get_xdata())
    assert x.min() < 0.65 < x.max() and (x.max() - x.min()) > 0.05, x
    assert hairlines[0].get_linewidth() >= 1.0 and hairlines[0].get_visible()


def test_no_hairline_when_no_hatched_bin_has_accuracy_zero(tmp_path, figures) -> None:
    y, probs = _reliability([(150, 0.97, 147), (3, 0.65, 2)])
    viz.plot_calibration(y, probs, tmp_path / "cal.png")
    ax = _by_title(figures[0], "Reliability diagram")
    assert not [ln for ln in ax.get_lines() if list(ln.get_ydata()) == [0.0, 0.0]]


def test_every_function_of_the_shared_viz_module_is_fully_annotated() -> None:
    """The repo convention is type hints on every parameter and return (review-iter-3 C7).

    ``_per_class_report`` was the one new helper of iteration 3 with bare parameters, and
    two axis helpers next to it had bare ``ax`` parameters too.
    """
    import inspect

    bare = []
    for name, fn in inspect.getmembers(viz, inspect.isfunction):
        if fn.__module__ != viz.__name__:
            continue
        signature = inspect.signature(fn)
        bare += [f"{name}({p.name})" for p in signature.parameters.values()
                 if p.annotation is inspect.Parameter.empty]
        if signature.return_annotation is inspect.Signature.empty:
            bare.append(f"{name} -> ?")
    assert bare == []
