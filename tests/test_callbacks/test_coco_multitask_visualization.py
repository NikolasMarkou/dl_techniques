"""The COCO multi-task grid callback never kills ``fit`` and draws 80 distinct classes.

Plan plan-2026-11-11T131349-13b28b37 step 16. Four claims, each with the
failure the unmodified callback showed:

1. a plotting or summary error is logged with its traceback and training goes
   on (old: ``try/finally`` with no ``except`` -- ``class_names`` shorter than
   the label indices raised ``IndexError`` out of ``fit``);
2. every one of the 80 COCO classes gets its own colour (old: ``tab20``
   resampled to 80 repeated each of 20 colours four times);
3. drawing the grid emits no warning (old: ``cm.get_cmap`` raised a
   ``MatplotlibDeprecationWarning``, removal scheduled for matplotlib 3.11);
4. a figure is closed even when saving it fails (old: leaked into pyplot's
   global registry).
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import warnings

import keras
import matplotlib.axes
import matplotlib.figure
import numpy as np
import pytest

from dl_techniques.callbacks.coco_multitask_visualization import (
    COCOMultiTaskPredictionGridCallback,
)
from dl_techniques.utils.matplotlib_backend import import_pyplot

_H = _W = 8
_SEG_CLASSES = 81  # background + 80 COCO classes
_CLS_CLASSES = 80


def _val_arrays(n: int = 2):
    rng = np.random.default_rng(0)
    rgb = rng.random((n, _H, _W, 3)).astype("float32")
    seg = rng.integers(0, _SEG_CLASSES, (n, _H, _W)).astype("int32")
    cls = np.zeros((n, _CLS_CLASSES), "float32")
    cls[:, 5] = 1.0  # a label index well beyond a 3-name list
    return rgb, seg, cls


def _callback(tmp_path, **kwargs) -> COCOMultiTaskPredictionGridCallback:
    rgb, seg, cls = _val_arrays()
    return COCOMultiTaskPredictionGridCallback(
        val_rgb=rgb,
        val_seg=seg,
        val_cls=cls,
        output_dir=str(tmp_path / "viz"),
        frequency=1,
        **kwargs,
    )


def _two_head_model() -> keras.Model:
    inputs = keras.Input(shape=(_H, _W, 3))
    seg = keras.layers.Conv2D(_SEG_CLASSES, 1, name="segmentation")(inputs)
    pooled = keras.layers.GlobalAveragePooling2D()(inputs)
    cls = keras.layers.Dense(_CLS_CLASSES, name="classification")(pooled)
    model = keras.Model(inputs, {"segmentation": seg, "classification": cls})
    model.compile(
        optimizer=keras.optimizers.SGD(0.01),
        loss={"segmentation": "mse", "classification": "mse"},
    )
    return model


def _fit(model: keras.Model, callback) -> None:
    rgb, _, _ = _val_arrays(4)
    model.fit(
        rgb,
        {
            "segmentation": np.zeros((4, _H, _W, _SEG_CLASSES), "float32"),
            "classification": np.zeros((4, _CLS_CLASSES), "float32"),
        },
        batch_size=2,
        epochs=1,
        verbose=0,
        callbacks=[callback],
    )


def _pred_seg(n: int) -> np.ndarray:
    return np.zeros((n, _H, _W, _SEG_CLASSES), "float32")


# ---------------------------------------------------------------------------
# 1. A failure inside the epoch body does not escape ``fit``
# ---------------------------------------------------------------------------
def test_a_short_class_names_list_does_not_kill_fit(tmp_path, caplog):
    callback = _callback(tmp_path, class_names=["a"] * 3)
    with caplog.at_level("WARNING", logger="dl"):
        _fit(_two_head_model(), callback)  # old: IndexError out of fit
    failures = [
        r for r in caplog.records
        if r.levelname == "WARNING" and "COCOMultiTaskPredictionGridCallback" in r.getMessage()
    ]
    assert len(failures) == 1, [r.getMessage() for r in caplog.records]
    assert failures[0].levelname == "WARNING"
    assert failures[0].exc_info is not None, "the traceback must travel with the warning"
    assert isinstance(failures[0].exc_info[1], IndexError)


def test_a_failing_epoch_still_runs_the_collector(tmp_path, monkeypatch):
    import gc

    calls = []
    real = gc.collect
    monkeypatch.setattr(gc, "collect", lambda *a, **k: calls.append(1) or real(*a, **k))
    callback = _callback(tmp_path, class_names=["a"] * 3)
    callback.set_model(_two_head_model())
    callback.on_epoch_end(0)
    assert calls, "gc.collect() must stay in the finally block"


# ---------------------------------------------------------------------------
# 2. 80 classes, 80 colours
# ---------------------------------------------------------------------------
def _render_gt_mask(tmp_path, monkeypatch, classes: np.ndarray) -> np.ndarray:
    """Return the RGB image the callback hands to ``imshow`` for the GT column."""
    rgb = np.zeros((1, 1, classes.size, 3), "float32")
    seg = classes.reshape(1, 1, -1).astype("int32")
    cls = np.zeros((1, _CLS_CLASSES), "float32")
    callback = COCOMultiTaskPredictionGridCallback(
        val_rgb=rgb, val_seg=seg, val_cls=cls,
        output_dir=str(tmp_path / "palette"), frequency=1,
    )
    shown = []
    real = matplotlib.axes.Axes.imshow

    def _recording(self, image, *a, **k):
        shown.append(np.asarray(image))
        return real(self, image, *a, **k)

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", _recording)
    callback._save_grid(1, _pred_seg(1)[:, :1, :1], np.zeros((1, _CLS_CLASSES)))
    # column order per sample: RGB, GT seg, predicted seg
    return shown[1]


def test_80_classes_get_80_distinct_colours(tmp_path, monkeypatch):
    gt = _render_gt_mask(tmp_path, monkeypatch, np.arange(1, 81))
    colours = {tuple(np.round(px, 4)) for px in gt.reshape(-1, 3)}
    assert len(colours) == 80, f"only {len(colours)} distinct colours for 80 classes"


def test_background_is_black_and_distinct_from_every_class(tmp_path, monkeypatch):
    gt = _render_gt_mask(tmp_path, monkeypatch, np.arange(0, 81))
    flat = gt.reshape(-1, 3)
    assert np.all(flat[0] == 0.0)
    assert len({tuple(np.round(px, 4)) for px in flat}) == 81


# ---------------------------------------------------------------------------
# 3. No deprecated API on the drawing path
# ---------------------------------------------------------------------------
def test_drawing_the_grid_emits_no_warning(tmp_path):
    callback = _callback(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # old: MatplotlibDeprecationWarning (cm.get_cmap)
        callback._save_grid(1, _pred_seg(2), np.zeros((2, _CLS_CLASSES)))
    assert (callback.output_dir / "epoch_001_seg.png").is_file()


# ---------------------------------------------------------------------------
# 4. Figures are closed even when saving fails
# ---------------------------------------------------------------------------
def _fail_savefig(monkeypatch) -> None:
    def _boom(self, *a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", _boom)


def test_the_seg_grid_closes_its_figure_when_saving_fails(tmp_path, monkeypatch):
    plt = import_pyplot()
    plt.close("all")
    callback = _callback(tmp_path)
    _fail_savefig(monkeypatch)
    with pytest.raises(RuntimeError, match="disk full"):
        callback._save_grid(1, _pred_seg(2), np.zeros((2, _CLS_CLASSES)))
    assert plt.get_fignums() == [], "the figure leaked into pyplot's registry"


def test_the_det_grid_closes_its_figure_when_saving_fails(tmp_path, monkeypatch):
    import dl_techniques.utils.yolo_decode as yolo_decode

    plt = import_pyplot()
    plt.close("all")
    boxes = np.full((2, 3, 5), -1.0, "float32")
    callback = _callback(
        tmp_path,
        val_boxes=boxes,
        detection_reg_max=1,
        detection_num_classes=_CLS_CLASSES,
        detection_strides=(4,),
    )
    empty = (np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), "int64"))
    monkeypatch.setattr(yolo_decode, "decode_predictions", lambda *a, **k: [empty, empty])
    monkeypatch.setattr(yolo_decode, "nms_per_class", lambda *a, **k: empty)
    _fail_savefig(monkeypatch)
    with pytest.raises(RuntimeError, match="disk full"):
        callback._save_det_grid(1, np.zeros((2, 4, 4 + _CLS_CLASSES)))
    assert plt.get_fignums() == [], "the figure leaked into pyplot's registry"
