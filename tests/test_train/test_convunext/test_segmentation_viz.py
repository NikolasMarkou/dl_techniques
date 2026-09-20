"""Guards for ``train.convunext.segmentation_viz``: the figures and the per-epoch grid callback.

Every plot function is fed a hand-built input whose correct picture is known:

- the palette is shared by the ground-truth and prediction columns and every class has a legend
  entry (a figure whose colours mean different things in two columns misleads silently);
- the per-row pixel accuracy in a title equals a hand count;
- ``segmentation_report.json`` equals a hand-computed IoU / Dice / precision / recall on a 3x3
  confusion, and is STRICT json (a class absent from both masks is ``null``, never ``NaN``);
- the mIoU curve marks the requested epoch and its x ticks are integers;
- a malformed input raises and leaves no file behind (the trainer, not the function, decides
  that a missing figure is not fatal).

The callback tests fit a 8x8 one-layer model on random arrays: the sample choice is seeded,
``epoch_000`` exists BEFORE the first epoch, the cadence and the last-epoch rule hold (also
after an early stop off the cadence), predictions use ``training=False`` and a failing figure
never stops ``fit``.

Everything is written under ``tmp_path``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List

os.environ.setdefault("MPLBACKEND", "Agg")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import train.convunext.common as common  # noqa: E402
import train.convunext.segmentation_viz as viz  # noqa: E402

NAMES = ["pet", "background", "border"]
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
# rows true, columns predicted: true = [10, 10, 10], predicted = [10, 8, 12], TP = [8, 6, 9]
CONFUSION = np.array([[8, 1, 1], [2, 6, 2], [0, 1, 9]])


def _strict(text: str) -> Any:
    def refuse(token: str):
        raise ValueError(f"non-strict JSON constant {token!r}")

    return json.loads(text, parse_constant=refuse)


def _is_png(path: Path) -> bool:
    return path.is_file() and path.read_bytes()[:8] == PNG_MAGIC and path.stat().st_size > 500


@pytest.fixture
def figures(monkeypatch):
    """Every figure about to be saved (the artists survive: ``Figure`` has no pyplot registry)."""
    seen: List[Any] = []
    real = viz._save

    def spy(fig, out_path):
        seen.append(fig)
        return real(fig, out_path)

    monkeypatch.setattr(viz, "_save", spy)
    return seen


def _batch(n: int = 3, size: int = 8):
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(n, size, size, 3)).astype(np.uint8)
    truths = np.broadcast_to((np.arange(size) * 3 // size)[None, :, None], (n, size, size)).astype(np.int64).copy()
    return images, truths


# ---------------------------------------------------------------------
# plot_segmentation_grid
# ---------------------------------------------------------------------

def test_the_grid_shares_one_palette_across_columns_names_every_class_and_reports_row_accuracy(
        tmp_path, figures) -> None:
    images, truths = _batch()
    predictions = truths.copy()
    predictions[1, :4] = (predictions[1, :4] + 1) % 3      # row 1: exactly half the rows wrong
    predictions[2] = 0                                      # row 2: a constant-class prediction
    out = tmp_path / "grid.png"
    accuracy = viz.plot_segmentation_grid(images, truths, predictions, NAMES, out, title="epoch 7")
    assert _is_png(out)
    assert accuracy == pytest.approx([1.0, 0.5, float(np.mean(truths[2] == 0))])

    (fig,) = figures
    axes = [ax for ax in fig.axes if ax.images]
    assert len(axes) == 3 * 3, "3 rows x (image, ground truth, prediction)"
    shown = [np.asarray(ax.images[0].get_array()) for ax in axes]
    for row in range(3):
        np.testing.assert_array_equal(shown[3 * row], images[row])
        np.testing.assert_array_equal(shown[3 * row + 1], viz.PALETTE[truths[row]])
        np.testing.assert_array_equal(shown[3 * row + 2], viz.PALETTE[predictions[row]])
    assert len({tuple(c) for c in viz.PALETTE}) == 3, "three distinguishable class colours"
    (legend,) = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == NAMES
    titles = [ax.get_title() for ax in axes]
    assert "pixel acc 1.000" in titles[2] and "pixel acc 0.500" in titles[5]
    assert fig._suptitle.get_text() == "epoch 7"


def test_the_grid_accepts_float_images_in_the_unit_range(tmp_path, figures) -> None:
    images, truths = _batch(n=1)
    viz.plot_segmentation_grid(images / 255.0, truths, truths, NAMES, tmp_path / "f.png")
    (fig,) = figures
    np.testing.assert_array_equal(np.asarray(fig.axes[0].images[0].get_array()), images[0])


@pytest.mark.parametrize("case", [
    "image_rank", "image_channels", "empty_batch", "float_out_of_range", "mask_shape",
    "mask_dtype", "mask_id_too_large", "mask_negative", "prediction_shape", "too_many_names", "no_names",
])
def test_the_grid_raises_on_malformed_input_and_writes_nothing(tmp_path, case) -> None:
    images, truths = _batch()
    names, predictions = NAMES, truths
    if case == "image_rank":
        images = images[..., 0]
    elif case == "image_channels":
        images = images[..., :2]
    elif case == "empty_batch":
        images, truths, predictions = images[:0], truths[:0], truths[:0]
    elif case == "float_out_of_range":
        images = images.astype(np.float32)          # 0..255 as float: not a unit-range image
    elif case == "mask_shape":
        truths = truths[:, :4]
    elif case == "mask_dtype":
        truths = truths.astype(np.float32)
    elif case == "mask_id_too_large":
        predictions = truths + 3
    elif case == "mask_negative":
        truths = truths - 1
    elif case == "prediction_shape":
        predictions = truths[:2]
    elif case == "too_many_names":
        names = NAMES + ["extra"]
    elif case == "no_names":
        names = []
    out = tmp_path / "grid.png"
    with pytest.raises(ValueError):
        viz.plot_segmentation_grid(images, truths, predictions, names, out)
    assert not out.exists()


# ---------------------------------------------------------------------
# plot_best_vs_final_predictions
# ---------------------------------------------------------------------

def test_best_vs_final_draws_both_predictions_in_their_columns(tmp_path, figures) -> None:
    images, truths = _batch(n=2)
    best, final = truths.copy(), truths.copy()
    final[0, :2] = (final[0, :2] + 1) % 3
    out = tmp_path / "bvf.png"
    best_acc, final_acc = viz.plot_best_vs_final_predictions(
        images, truths, best, final, NAMES, out, best_label="best (epoch 2)",
        final_label="final (epoch 5)", title="run")
    assert _is_png(out)
    assert best_acc == [1.0, 1.0] and final_acc == pytest.approx([0.75, 1.0])
    (fig,) = figures
    axes = [ax for ax in fig.axes if ax.images]
    assert len(axes) == 2 * 4
    np.testing.assert_array_equal(np.asarray(axes[2].images[0].get_array()), viz.PALETTE[best[0]])
    np.testing.assert_array_equal(np.asarray(axes[3].images[0].get_array()), viz.PALETTE[final[0]])
    assert "best (epoch 2)" in axes[2].get_title() and "final (epoch 5)" in axes[3].get_title()


def test_best_vs_final_raises_when_one_prediction_does_not_match(tmp_path) -> None:
    images, truths = _batch(n=2)
    with pytest.raises(ValueError):
        viz.plot_best_vs_final_predictions(images, truths, truths, truths[:1], NAMES, tmp_path / "x.png")
    assert not (tmp_path / "x.png").exists()


# ---------------------------------------------------------------------
# Scores, report, per-class chart
# ---------------------------------------------------------------------

def test_class_scores_equal_the_hand_computed_values_on_a_3x3_confusion() -> None:
    scores = viz.class_scores(CONFUSION)
    assert scores["per_class_iou"] == pytest.approx([8 / 12, 6 / 12, 9 / 13])
    assert scores["per_class_dice"] == pytest.approx([16 / 20, 12 / 18, 18 / 22])
    assert scores["per_class_precision"] == pytest.approx([8 / 10, 6 / 8, 9 / 12])
    assert scores["per_class_recall"] == pytest.approx([8 / 10, 6 / 10, 9 / 10])
    assert scores["support"] == [10, 10, 10]
    assert scores["miou"] == pytest.approx((8 / 12 + 6 / 12 + 9 / 13) / 3)
    assert scores["mean_dice"] == pytest.approx((16 / 20 + 12 / 18 + 18 / 22) / 3)
    assert scores["pixel_accuracy"] == pytest.approx(23 / 30)


def test_the_trainers_segmentation_scores_delegate_to_the_one_implementation() -> None:
    """IoU is written once: the trainer's summary and the report cannot disagree."""
    rng = np.random.default_rng(3)
    confusion = rng.integers(0, 500, size=(3, 3))
    theirs = viz.class_scores(confusion)
    ours = common.segmentation_scores(confusion)
    assert ours == {k: theirs[k] for k in ("per_class_iou", "miou", "pixel_accuracy")}


def test_a_class_with_no_pixels_has_null_scores_and_the_report_stays_strict_json(tmp_path) -> None:
    confusion = np.array([[6, 2, 0], [1, 3, 0], [0, 0, 0]])          # class 2 absent everywhere
    out = tmp_path / "report.json"
    report = viz.write_segmentation_report(confusion, NAMES, out)
    loaded = _strict(out.read_text())                                # NaN would raise here
    assert loaded == report
    border = loaded["per_class"]["border"]
    assert border == {"iou": None, "dice": None, "precision": None, "recall": None, "support": 0}
    assert loaded["miou"] == pytest.approx((6 / 9 + 3 / 6) / 2), "the empty-union class is not in the mean"


def test_the_report_file_carries_the_hand_computed_values_by_class_name(tmp_path) -> None:
    out = tmp_path / "vis" / "segmentation_report.json"
    viz.write_segmentation_report(CONFUSION, NAMES, out)
    report = _strict(out.read_text())
    assert report["class_names"] == NAMES and report["confusion"] == CONFUSION.tolist()
    pet, background, border = (report["per_class"][n] for n in NAMES)
    assert pet["iou"] == pytest.approx(8 / 12) and pet["dice"] == pytest.approx(16 / 20)
    assert background["precision"] == pytest.approx(6 / 8) and background["recall"] == pytest.approx(6 / 10)
    assert border["iou"] == pytest.approx(9 / 13) and border["support"] == 10
    assert report["pixel_accuracy"] == pytest.approx(23 / 30)


@pytest.mark.parametrize("confusion,names", [
    (np.zeros((2, 3)), NAMES),
    (np.zeros(3), NAMES),
    (np.array([[1, -1, 0], [0, 1, 0], [0, 0, 1]]), NAMES),
    (CONFUSION, ["a", "b"]),
])
def test_the_report_and_the_per_class_chart_raise_on_a_malformed_matrix_and_write_nothing(
        tmp_path, confusion, names) -> None:
    with pytest.raises(ValueError):
        viz.write_segmentation_report(confusion, names, tmp_path / "r.json")
    with pytest.raises(ValueError):
        viz.plot_per_class_scores(confusion, names, tmp_path / "p.png")
    assert list(tmp_path.iterdir()) == []


def test_the_per_class_chart_draws_four_bars_per_class_and_marks_an_undefined_score(tmp_path, figures) -> None:
    viz.plot_per_class_scores(CONFUSION, NAMES, tmp_path / "ok.png")
    assert _is_png(tmp_path / "ok.png")
    (fig,) = figures
    (ax,) = fig.axes
    assert len(ax.patches) == 3 * 4 and [t.get_text() for t in ax.get_legend().get_texts()] == [
        "iou", "dice", "precision", "recall"]
    assert "n=10 px" in ax.get_xticklabels()[0].get_text()

    absent = np.array([[6, 2, 0], [1, 3, 0], [0, 0, 0]])
    viz.plot_per_class_scores(absent, NAMES, tmp_path / "absent.png")
    assert "n/a" in [t.get_text() for t in figures[1].axes[0].texts]


# ---------------------------------------------------------------------
# plot_miou_curve
# ---------------------------------------------------------------------

def test_the_miou_curve_marks_the_best_epoch_and_uses_integer_ticks(tmp_path, figures) -> None:
    history = {"miou": [0.2, 0.4, 0.5, 0.55], "val_miou": [0.25, 0.45, 0.4, 0.5]}
    out = tmp_path / "miou.png"
    viz.plot_miou_curve(history, best_epoch=2, out_path=out)
    assert _is_png(out)
    (fig,) = figures
    (ax,) = fig.axes
    fig.canvas.draw()
    lines = {tuple(np.round(line.get_xdata(), 6)): line for line in ax.lines}
    marker = [line for line in ax.lines if line.get_marker() == "*"]
    assert len(marker) == 1 and list(marker[0].get_xdata()) == [2] and list(marker[0].get_ydata()) == [0.45]
    vertical = [line for line in ax.lines if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]]
    assert [line.get_xdata()[0] for line in vertical] == [2]
    ticks = [t for t in ax.get_xticks() if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    assert ticks == [1.0, 2.0, 3.0, 4.0], "integer epochs only, no 1.5"
    assert any("best epoch 2" in t.get_text() for t in ax.get_legend().get_texts())
    assert lines, "the two curves are drawn"


def test_the_miou_curve_of_two_epochs_has_no_fractional_ticks(tmp_path, figures) -> None:
    viz.plot_miou_curve({"miou": [0.2, 0.3], "val_miou": [0.25, 0.35]}, 2, tmp_path / "m.png")
    (ax,) = figures[0].axes
    figures[0].canvas.draw()
    ticks = [t for t in ax.get_xticks() if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    assert ticks == [1.0, 2.0]


@pytest.mark.parametrize("history,best", [
    ({}, 1),
    ({"miou": [0.1, 0.2]}, 1),                                    # no val_miou
    ({"miou": [], "val_miou": []}, 1),
    ({"miou": [0.1, 0.2], "val_miou": [0.1]}, 1),                 # unequal length
    ({"miou": [0.1, float("nan")], "val_miou": [0.1, 0.2]}, 1),
    ({"miou": [0.1, 0.2], "val_miou": [0.1, 0.2]}, 0),            # best epoch out of range
    ({"miou": [0.1, 0.2], "val_miou": [0.1, 0.2]}, 3),
])
def test_the_miou_curve_raises_on_malformed_input_and_writes_nothing(tmp_path, history, best) -> None:
    with pytest.raises(ValueError):
        viz.plot_miou_curve(history, best, tmp_path / "m.png")
    assert not (tmp_path / "m.png").exists()


# ---------------------------------------------------------------------
# SegmentationGridCallback
# ---------------------------------------------------------------------

def _arrays(n: int = 40, size: int = 8):
    rng = np.random.default_rng(1)
    x = rng.integers(0, 256, size=(n, size, size, 3)).astype(np.uint8)
    y = rng.integers(0, 3, size=(n, size, size)).astype(np.uint8)
    return x, y


def _model(size: int = 8, dropout: float = 0.0) -> keras.Model:
    inputs = keras.Input((size, size, 3))
    hidden = keras.layers.Conv2D(8, 1, activation="relu")(inputs)
    if dropout:
        hidden = keras.layers.Dropout(dropout)(hidden)
    model = keras.Model(inputs, keras.layers.Conv2D(3, 1)(hidden))
    model.compile(
        optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.MeanIoU(num_classes=3, sparse_y_true=True, sparse_y_pred=False, name="miou")])
    return model


def _fit(cb, epochs: int, extra=(), n: int = 40):
    x, y = _arrays(n)
    model = _model()
    model.fit(x.astype("float32") / 255.0, y.astype("int32"), validation_split=0.25, epochs=epochs,
              batch_size=10, verbose=0, callbacks=[cb, *extra])
    return model


def _grids(directory: Path) -> List[str]:
    return sorted(p.name for p in directory.glob("epoch_*_seg_grid.png"))


def test_the_same_seed_picks_the_same_samples_and_another_seed_picks_others(tmp_path) -> None:
    x, y = _arrays()
    make = lambda seed: viz.SegmentationGridCallback(x, y, tmp_path, NAMES, 1, 4, seed)  # noqa: E731
    a, b, c = make(5), make(5), make(6)
    np.testing.assert_array_equal(a.indices, b.indices)
    assert not np.array_equal(a.indices, c.indices)
    assert len(a.indices) == 4 == len(set(a.indices.tolist())) and list(a.indices) == sorted(a.indices)
    assert a.indices.max() < len(x)
    np.testing.assert_array_equal(a.images, x[a.indices])
    np.testing.assert_array_equal(a.masks, y[a.indices])
    assert len(viz.SegmentationGridCallback(x, y, tmp_path, NAMES, 1, 500, 0).indices) == len(x), "clamped to N"


@pytest.mark.parametrize("freq,samples", [(0, 4), (1, 0), (-1, 4)])
def test_the_callback_refuses_a_non_positive_cadence_or_sample_count(tmp_path, freq, samples) -> None:
    x, y = _arrays()
    with pytest.raises(ValueError):
        viz.SegmentationGridCallback(x, y, tmp_path, NAMES, freq, samples, 0)


def test_the_callback_refuses_mismatched_or_empty_arrays(tmp_path) -> None:
    x, y = _arrays()
    with pytest.raises(ValueError):
        viz.SegmentationGridCallback(x, y[:5], tmp_path, NAMES, 1, 2, 0)
    with pytest.raises(ValueError):
        viz.SegmentationGridCallback(x[:0], y[:0], tmp_path, NAMES, 1, 2, 0)


def test_epoch_000_exists_before_the_first_epoch_and_one_grid_follows_each_epoch(tmp_path) -> None:
    class Probe(keras.callbacks.Callback):
        seen = None

        def on_epoch_begin(self, epoch, logs=None):
            if epoch == 0:
                Probe.seen = _grids(tmp_path)

    x, y = _arrays()
    cb = viz.SegmentationGridCallback(x, y, tmp_path, NAMES, 1, 3, 0)
    _fit(cb, 3, extra=[Probe()])
    assert Probe.seen == ["epoch_000_seg_grid.png"], "the untrained model is drawn before epoch 1"
    assert _grids(tmp_path) == [f"epoch_{i:03d}_seg_grid.png" for i in range(4)]
    assert cb.written == _grids(tmp_path) and cb.failed == {}
    assert all(_is_png(tmp_path / n) for n in cb.written)


def test_the_cadence_follows_viz_freq_and_always_ends_with_the_last_planned_epoch(tmp_path) -> None:
    x, y = _arrays()
    cb = viz.SegmentationGridCallback(x, y, tmp_path, NAMES, 2, 2, 0)
    _fit(cb, 5)
    assert _grids(tmp_path) == ["epoch_000_seg_grid.png", "epoch_002_seg_grid.png",
                                "epoch_004_seg_grid.png", "epoch_005_seg_grid.png"]


def test_an_early_stop_off_the_cadence_still_draws_the_last_completed_epoch(tmp_path) -> None:
    class StopAfter(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch == 2:
                self.model.stop_training = True

    x, y = _arrays()
    cb = viz.SegmentationGridCallback(x, y, tmp_path, NAMES, 2, 2, 0)
    _fit(cb, 6, extra=[StopAfter()])
    assert _grids(tmp_path) == ["epoch_000_seg_grid.png", "epoch_002_seg_grid.png", "epoch_003_seg_grid.png"]


def test_the_grid_title_carries_the_epoch_and_the_validation_miou(tmp_path, monkeypatch) -> None:
    titles: List[str] = []
    real = viz.plot_segmentation_grid

    def spy(*args, **kwargs):
        titles.append(kwargs["title"])
        return real(*args, **kwargs)

    monkeypatch.setattr(viz, "plot_segmentation_grid", spy)
    x, y = _arrays()
    _fit(viz.SegmentationGridCallback(x, y, tmp_path, NAMES, 1, 2, 0, title="run"), 2)
    assert titles[0] == "run untrained" and titles[1].startswith("run after epoch 1, val mIoU ")
    assert titles[2].startswith("run after epoch 2, val mIoU ")
    # The mIoU is of the WHOLE validation set while the sheet shows 2 images of it: say so.
    assert titles[1].endswith(f"(all {len(x)})") and titles[2].endswith(f"(all {len(x)})")
    assert len(x) > 2


def test_predictions_are_made_with_training_false(tmp_path) -> None:
    """A dropout of 0.9 would make two training-mode passes disagree almost everywhere."""
    x, y = _arrays()
    cb = viz.SegmentationGridCallback(x, y, tmp_path, NAMES, 1, 4, 0)
    model = _model(dropout=0.9)
    first, second = cb.predict_classes(model), cb.predict_classes(model)
    np.testing.assert_array_equal(first, second)
    reference = np.argmax(keras.ops.convert_to_numpy(
        model(cb.images.astype(np.float32) / 255.0, training=False)), axis=-1)
    np.testing.assert_array_equal(first, reference)
    assert first.shape == (4, 8, 8) and set(np.unique(first)) <= {0, 1, 2}


def test_a_failing_figure_is_recorded_and_never_stops_fit(tmp_path, monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("plot exploded")

    monkeypatch.setattr(viz, "plot_segmentation_grid", boom)
    x, y = _arrays()
    cb = viz.SegmentationGridCallback(x, y, tmp_path, NAMES, 1, 2, 0)
    model = _fit(cb, 3)
    assert model.history.epoch == [0, 1, 2], "fit ran every epoch"
    assert cb.written == [] and list(cb.failed) == [f"epoch_{i:03d}_seg_grid.png" for i in range(4)]
    assert all("plot exploded" in error for error in cb.failed.values())


def test_a_figure_that_fails_only_once_is_recorded_once_and_the_rest_are_written(tmp_path, monkeypatch) -> None:
    real, calls = viz.plot_segmentation_grid, []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("disk full")
        return real(*args, **kwargs)

    monkeypatch.setattr(viz, "plot_segmentation_grid", flaky)
    x, y = _arrays()
    cb = viz.SegmentationGridCallback(x, y, tmp_path, NAMES, 1, 2, 0)
    _fit(cb, 2)
    assert list(cb.failed) == ["epoch_001_seg_grid.png"] and "disk full" in cb.failed["epoch_001_seg_grid.png"]
    assert _grids(tmp_path) == ["epoch_000_seg_grid.png", "epoch_002_seg_grid.png"]
