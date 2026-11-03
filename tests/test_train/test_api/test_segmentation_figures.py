"""The figures and the analyzer of a segmentation run (plan 2faf890a step 29, D-079).

Three layers, each observed where a reader of the figure would see it:

* **The drawing functions** (``train.convunext.segmentation_viz``) take a colour table and a void value
  as optional arguments. With neither they are the legacy 3-class functions to the pixel (the legacy
  ``tests/test_train/test_convunext/test_segmentation_viz.py`` runs unchanged, including the
  ``too_many_names`` refusal); with ``extended_palette(C)`` they draw 150 classes, and a void pixel is
  drawn in its own named colour and counted in no row's accuracy (a hand fixture with a void quarter,
  where counting void as wrong gives another number).
* **The confusion** of a run (``segmentation_run.confusion_from_dataset``) equals an independent
  ``numpy`` count and gives the mIoU that Keras' ``MeanIoU`` reported for the same pixels.
* **The task** writes, under ``analysis_args.figures`` and with a test split, the per-epoch grid
  (``epoch_000`` the untrained model), ``predictions_best.png``, ``miou_curve.png``, ``per_class_scores.png``
  and ``segmentation_report.json``, names them in ``visualizations.files`` and ``notes``; with figures off
  it writes none and ``visualizations/`` does not exist; with no test split it writes none and says so.

The tests write only under ``tmp_path``.
"""

import json
from pathlib import Path

import keras
import numpy as np
import pytest

from train.api.config import validate
from train.api.tasks.image_segmentation import ImageSegmentationConfig, ImageSegmentationTrainer
from train.common import segmentation_run
from train.convunext import segmentation_viz as viz

from .test_image_segmentation_task import (
    SIZE,
    VOID,
    data_of,
    folders,  # noqa: F401  (fixture)
    read_json,
    world,  # noqa: F401  (fixture)
)

NAMES = ["pet", "background", "border"]


# ---------------------------------------------------------------------
# The palette
# ---------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 3])
def test_up_to_three_classes_the_palette_is_the_legacy_one(count):
    assert np.array_equal(viz.extended_palette(count), viz.PALETTE[:count])


@pytest.mark.parametrize("count", [4, 21, 150, 255])
def test_a_longer_palette_keeps_the_first_three_rows_and_never_repeats_a_colour_or_the_void_one(count):
    palette = viz.extended_palette(count)
    assert palette.shape == (count, 3) and palette.dtype == np.uint8
    assert np.array_equal(palette[:3], viz.PALETTE), "additive: the 3-class pixels do not move"
    colours = {tuple(row) for row in palette}
    assert len(colours) == count and tuple(viz.VOID_COLOR) not in colours
    assert np.array_equal(palette, viz.extended_palette(count)), "deterministic"
    assert np.array_equal(viz.extended_palette(count + 5)[:count], palette), "prefix-stable"


def test_the_palette_skips_the_void_colour_even_when_a_generated_hue_would_equal_it(monkeypatch):
    generated = viz.extended_palette(5)[3].copy()
    monkeypatch.setattr(viz, "VOID_COLOR", generated)
    palette = viz.extended_palette(8)
    assert tuple(generated) not in {tuple(row) for row in palette}
    assert len({tuple(row) for row in palette}) == 8


def test_no_class_is_no_palette():
    with pytest.raises(ValueError):
        viz.extended_palette(0)


# ---------------------------------------------------------------------
# The grid: many classes and void
# ---------------------------------------------------------------------


@pytest.fixture
def figures(monkeypatch):
    seen = []
    real = viz._save

    def spy(fig, out_path):
        seen.append(fig)
        return real(fig, out_path)

    monkeypatch.setattr(viz, "_save", spy)
    return seen


def test_a_void_pixel_is_drawn_in_the_void_colour_and_counted_in_no_row(tmp_path, figures):
    """Rows of 8 pixels. Row 0 has 2 void pixels and 5 of its 6 valid pixels right (5/6; with the void
    pixels counted as pixels the model got wrong the row would read 5/8), row 1 has none and 7 of 8 right,
    row 2 is all void and has no accuracy."""
    truths = np.array([[[0, 0, 1, 1, 2, 2, VOID, VOID]],
                       [[0, 0, 1, 1, 2, 2, 0, 1]],
                       [[VOID] * 8]], dtype=np.int64)
    predictions = np.array([[[0, 1, 1, 1, 2, 2, VOID % 3, 0]],      # 5 of 6 valid right
                            [[0, 0, 1, 1, 2, 2, 0, 0]],              # 7 of 8 right
                            [[0] * 8]], dtype=np.int64)
    images = np.zeros((3, 1, 8, 3), np.uint8)
    out = tmp_path / "grid.png"
    accuracy = viz.plot_segmentation_grid(
        images, truths, predictions, NAMES, out, palette=viz.extended_palette(3), ignore_index=VOID)
    assert accuracy[0] == pytest.approx(5 / 6) and accuracy[1] == pytest.approx(7 / 8)
    assert np.isnan(accuracy[2]), "a row with no valid pixel has no accuracy, not 0 and not 1"
    # The control: void counted as a pixel the model must predict gives another number for row 0.
    assert float(np.mean(predictions[0] == truths[0])) != accuracy[0]
    (fig,) = figures
    axes = [ax for ax in fig.axes if ax.images]
    shown = [np.asarray(ax.images[0].get_array()) for ax in axes]
    truth_row0 = shown[1]
    assert np.array_equal(truth_row0[0, 6], viz.VOID_COLOR) and np.array_equal(truth_row0[0, 7], viz.VOID_COLOR)
    assert np.array_equal(truth_row0[0, 0], viz.PALETTE[0])
    assert "pixel acc n/a" in axes[8].get_title()
    (legend,) = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == NAMES + [viz.VOID_NAME]


def test_the_grid_draws_150_classes_with_a_capped_legend_that_lists_the_classes_shown(tmp_path, figures):
    classes = 150
    names = [f"class{k}" for k in range(classes)]
    truths = np.arange(classes * 2).reshape(1, 6, 50) % classes
    # Most pixels are class k, a few void.
    truths = truths.astype(np.int64)
    predictions = (truths + 1) % classes
    images = np.zeros((1, 6, 50, 3), np.uint8)
    out = tmp_path / "many.png"
    viz.plot_segmentation_grid(
        images, truths, predictions, names, out, palette=viz.extended_palette(classes))
    assert out.is_file()
    (fig,) = figures
    (legend,) = fig.legends
    texts = [t.get_text() for t in legend.get_texts()]
    assert len(texts) == viz.MAX_LEGEND + 1 and texts[-1] == f"+{classes - viz.MAX_LEGEND} more"
    assert texts[:viz.MAX_LEGEND] == names[:viz.MAX_LEGEND]
    drawn = np.asarray([ax for ax in fig.axes if ax.images][1].images[0].get_array())
    assert np.array_equal(drawn.reshape(-1, 3)[:5], viz.extended_palette(classes)[truths.reshape(-1)[:5]])


def test_without_the_new_arguments_the_legacy_limits_stand(tmp_path):
    images = np.zeros((1, 2, 2, 3), np.uint8)
    truths = np.zeros((1, 2, 2), np.int64)
    with pytest.raises(ValueError, match="between 1 and 3"):
        viz.plot_segmentation_grid(images, truths, truths, NAMES + ["extra"], tmp_path / "g.png")
    with pytest.raises(ValueError, match="ids must lie in"):
        viz.plot_segmentation_grid(images, truths + VOID, truths, NAMES, tmp_path / "g.png")
    assert not (tmp_path / "g.png").exists()


def test_a_void_value_inside_the_classes_or_below_zero_is_refused_as_a_void_value(tmp_path):
    images = np.zeros((1, 2, 2, 3), np.uint8)
    truths = np.zeros((1, 2, 2), np.int64)
    for bad in (1, -1):
        with pytest.raises(ValueError, match="ignore_index"):
            viz.plot_segmentation_grid(images, truths, truths, NAMES, tmp_path / "g.png", ignore_index=bad)


def test_many_classes_are_drawn_as_iou_bars_and_few_keep_the_grouped_chart(tmp_path, figures):
    rng = np.random.default_rng(0)
    for classes, bars in ((4, 4 * 4), (150, 150)):
        confusion = rng.integers(0, 50, size=(classes, classes))
        figures.clear()
        report = viz.plot_per_class_scores(confusion, [f"c{k}" for k in range(classes)], tmp_path / f"p{classes}.png")
        assert (tmp_path / f"p{classes}.png").is_file() and len(report["per_class"]) == classes
        (fig,) = figures
        assert sum(len(c.patches) for c in fig.axes[0].containers) == bars, classes


# ---------------------------------------------------------------------
# The confusion of a run
# ---------------------------------------------------------------------


def test_the_grid_title_names_the_population_val_miou_is_scored_on(tmp_path, figures):
    images = np.zeros((2, 4, 4, 3), np.uint8)
    masks = np.zeros((2, 4, 4), np.int64)
    class Model(keras.Model):
        def call(self, x, training=False):
            return keras.ops.zeros(x.shape[:3] + (3,))
    for population, expected in ((None, "(all 2)"), ("all 9 selection pairs; the pairs shown are test pairs",
                                                      "(all 9 selection pairs; the pairs shown are test pairs)")):
        figures.clear()
        callback = viz.SegmentationGridCallback(images, masks, tmp_path, NAMES, 1, 2, 0,
                                                val_population=population)
        callback.set_model(Model())
        callback.on_epoch_end(0, {"val_miou": 0.5})
        assert expected in figures[-1]._suptitle.get_text()


def test_the_confusion_is_an_independent_count_with_void_left_out_and_its_miou_is_keras():
    rng = np.random.default_rng(3)
    classes = 4
    masks = rng.integers(0, classes, size=(5, 6, 7))
    masks[rng.random(masks.shape) < 0.2] = VOID
    logits = rng.normal(size=masks.shape + (classes,)).astype(np.float32)
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices(
        (logits, masks.astype(np.int32))).batch(2)    # the "image" IS the logits

    class Identity(keras.Model):
        def call(self, x, training=False):
            return x

    confusion = segmentation_run.confusion_from_dataset(Identity(), dataset, classes, VOID)
    predicted = logits.argmax(-1)
    expected = np.zeros((classes, classes), np.int64)
    for truth, guess in zip(masks.reshape(-1), predicted.reshape(-1)):
        if truth != VOID:
            expected[truth, guess] += 1
    assert np.array_equal(confusion, expected)
    metric = keras.metrics.MeanIoU(num_classes=classes, sparse_y_true=True, sparse_y_pred=False, ignore_class=VOID)
    metric.update_state(masks.astype(np.int32), logits)
    assert viz.class_scores(confusion)["miou"] == pytest.approx(float(metric.result()), abs=1e-6)


# ---------------------------------------------------------------------
# The task writes them
# ---------------------------------------------------------------------

GRID = "epoch_{:03d}_seg_grid.png"


def request(out, folders, kind="void", **changes):
    mapping = {
        "out": str(out), "model": "convunext/tiny", "data": data_of(folders, kind), "epochs": 2,
        "batch_size": 8, "transform_args": {"image_size": SIZE},
        "analysis_args": {"model_analysis": False, "figures": True},
    }
    mapping.update(changes)
    return validate(ImageSegmentationConfig, mapping)


@pytest.fixture(scope="module")
def drawn(world, folders, tmp_path_factory):  # noqa: F811
    out = tmp_path_factory.mktemp("figures_run") / "run"
    result = ImageSegmentationTrainer(request(out, folders)).run()
    return out, read_json(result.summary_path)


FINAL = ("predictions_best.png", "miou_curve.png", "per_class_scores.png", "segmentation_report.json")


def test_a_run_with_figures_writes_the_grid_per_epoch_and_the_four_final_files(drawn):
    out, summary = drawn
    vis = out / "visualizations"
    grids = sorted(p.name for p in vis.glob("epoch_*_seg_grid.png"))
    assert grids == [GRID.format(k) for k in (0, 1, 2)], "the untrained model, then each epoch"
    for name in (*FINAL, "training_dashboard.png"):
        assert (vis / name).is_file(), name
    record = summary["visualizations"]
    assert sorted(record["files"]) == sorted(FINAL)
    assert record["failed"] == [] and record["skipped"] is False
    assert "ece" not in record and "ece" not in summary
    assert not any("no segmentation figure" in note for note in summary["notes"])


def test_the_report_is_the_test_confusion_of_the_best_weights_and_agrees_with_keras(drawn):
    out, summary = drawn
    report = json.loads((out / "visualizations" / "segmentation_report.json").read_text())
    assert report["class_names"] == ["0", "1", "2"]
    confusion = np.asarray(report["confusion"])
    # 8 test pairs of 32 x 32 pixels, 4 columns of 32 void in every mask: void is left out.
    assert confusion.sum() == 8 * 32 * 28
    assert report["miou"] == pytest.approx(summary["test_metrics_best"]["miou"], abs=1e-6)
    assert report["pixel_accuracy"] == pytest.approx(summary["test_metrics_best"]["accuracy"], abs=1e-6)
    # The void fixture: 18 columns of class 0, 8 of class 1, 2 of class 2 and 4 void (the class counts are by construction).
    assert confusion.sum(axis=1).tolist() == [8 * 32 * 18, 8 * 32 * 8, 8 * 32 * 2]


def test_the_miou_curve_is_given_the_epoch_and_the_metric_the_run_kept(world, folders, tmp_path, monkeypatch):  # noqa: F811
    calls = []
    real = viz.plot_miou_curve

    def spy(history, best_epoch, out_path, title=None, monitor="val_loss"):
        calls.append((best_epoch, monitor, list(history["val_miou"])))
        return real(history, best_epoch, out_path, title=title, monitor=monitor)

    monkeypatch.setattr(viz, "plot_miou_curve", spy)
    result = ImageSegmentationTrainer(request(tmp_path / "curve", folders, kind="clean")).run()
    summary = read_json(result.summary_path)
    (best_epoch, monitor, val_miou), = calls
    assert monitor == "val_miou" and best_epoch == summary["best_epoch"]
    assert val_miou == pytest.approx(json.loads(Path(result.run_dir / "training_history.json").read_text())["val_miou"])
    assert (result.run_dir / "visualizations" / "miou_curve.png").is_file()


def test_figures_off_writes_none_and_does_not_create_the_directory(world, folders, tmp_path):  # noqa: F811
    config = request(tmp_path / "off", folders, kind="clean",
                     analysis_args={"model_analysis": False, "figures": False})
    result = ImageSegmentationTrainer(config).run()
    assert not (result.run_dir / "visualizations").exists()
    summary = read_json(result.summary_path)
    assert summary["visualizations"] == {"files": [], "failed": [], "skipped": True}
    assert any("analysis_args.figures is false" in note for note in summary["notes"])


def test_no_test_split_means_no_figure_and_the_summary_says_so(world, folders, tmp_path):  # noqa: F811
    data = data_of(folders, "clean")
    for key in ("test_images", "test_masks"):
        data.pop(key)
    config = request(tmp_path / "none", folders, kind="clean", data=data)
    result = ImageSegmentationTrainer(config).run()
    vis = result.run_dir / "visualizations"
    assert not list(vis.glob("*_seg_grid.png")) and not any((vis / n).exists() for n in FINAL)
    summary = read_json(result.summary_path)
    assert summary["visualizations"]["skipped"] is True and summary["test_metrics_best"] is None
    assert any("no test split" in note for note in summary["notes"])


def test_the_figure_hook_is_after_the_log_editors_and_before_the_dashboard(world, folders, tmp_path):  # noqa: F811
    seen = {}

    class Spy(ImageSegmentationTrainer):
        def _callbacks(self, *arguments):
            epoch_line, callbacks = super()._callbacks(*arguments)
            seen["order"] = [type(c).__name__ for c in callbacks]
            return epoch_line, callbacks

    Spy(request(tmp_path / "order", folders, kind="clean", epochs=1)).run()
    order = seen["order"]
    grid, dashboard, line = (order.index(n) for n in
                             ("SegmentationGridCallback", "TrainingDashboardCallback", "EpochLogLine"))
    assert line < grid < dashboard
    assert order.index("LearningRateLogger") == 0 and order.index("LastEpochWeights") < line


def test_the_grid_callback_gets_uint8_images_the_palette_of_the_run_and_its_void_value(world, folders, tmp_path):  # noqa: F811
    trainer = ImageSegmentationTrainer(request(tmp_path / "hook", folders))
    data = type("D", (), {})()
    data.x_fig = np.full((3, 8, 8, 3), 100 / 255.0, np.float32)
    data.y_fig = np.zeros((3, 8, 8), np.int32)
    data.class_names, data.ignore_index, data.n_select = ["a", "b", "c", "d"], VOID, 37
    (callback,) = trainer.figure_callbacks(trainer.config, data, tmp_path)
    assert callback.images.dtype == np.uint8 and int(callback.images.max()) == 100
    assert np.array_equal(callback.palette, viz.extended_palette(4)) and callback.ignore_index == VOID
    assert "all 37 selection pairs" in callback.val_population and "test pairs" in callback.val_population
    # Rounded, not truncated: a value a hair under a byte (arithmetic noise of 1e-6) is that byte.
    data.x_fig = np.full((3, 8, 8, 3), 100 / 255.0 - 1e-6, np.float32)
    (noisy,) = trainer.figure_callbacks(trainer.config, data, tmp_path)
    assert int(noisy.images.min()) == int(noisy.images.max()) == 100
    empty = type("D", (), {"x_fig": np.zeros((0, 8, 8, 3), np.float32)})()
    assert trainer.figure_callbacks(trainer.config, empty, tmp_path) == []
