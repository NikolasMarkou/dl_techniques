"""``TrainingCurvesCallback``: non-scalar logs, figure lifecycle, documented layout.

Plan plan-2026-11-11T131349-13b28b37 step 16 (findings row L1). Old behaviour:

* ``float(v)`` ran before the guarded block, so an array, string or ``None``
  value in ``logs`` raised ``TypeError`` / ``ValueError`` out of ``fit``;
* a figure was closed only on the success path;
* the docstring promised ``{output_dir}/training_curves/`` and omitted the
  ``detection`` group, while the code writes ``{output_dir}/{prefix}{group}.png``.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import keras
import matplotlib.figure
import numpy as np
import pytest

import dl_techniques.callbacks.training_curves as training_curves
from dl_techniques.callbacks.training_curves import (
    TrainingCurvesCallback,
    _group_metrics,
)
from dl_techniques.utils.matplotlib_backend import import_pyplot


def _drive(callback: TrainingCurvesCallback, rows) -> None:
    callback.on_train_begin()
    for epoch, logs in enumerate(rows):
        callback.on_epoch_end(epoch, logs)


# ---------------------------------------------------------------------------
# Non-scalar log values
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    [np.array([1.0, 2.0]), "not a number", None],
    ids=["array", "string", "none"],
)
def test_a_non_scalar_log_value_does_not_raise_out_of_fit(tmp_path, bad):
    callback = TrainingCurvesCallback(str(tmp_path))
    rows = [{"loss": 1.0, "weird": bad}, {"loss": 0.5, "weird": bad}]
    _drive(callback, rows)  # old: TypeError / ValueError
    assert (tmp_path / "loss.png").is_file(), "the scalar keys must still be plotted"
    assert not (tmp_path / "other.png").exists(), "the skipped key must not be plotted"
    assert [row["loss"] for row in callback._history] == [1.0, 0.5]
    assert all("weird" not in row for row in callback._history)
    assert callback._epochs == [1, 2]


def test_a_skipped_key_is_reported_once_per_run(tmp_path, caplog):
    callback = TrainingCurvesCallback(str(tmp_path))
    with caplog.at_level("WARNING", logger="dl"):
        _drive(callback, [{"loss": 1.0, "w": np.ones(3)}] * 3)
    skipped = [r for r in caplog.records if "'w'" in r.getMessage()]
    assert len(skipped) == 1, [r.getMessage() for r in caplog.records]


def test_a_non_scalar_value_survives_a_real_fit(tmp_path):
    class _Poison(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs["poison"] = np.array([1.0, 2.0])

    model = keras.Sequential([keras.Input(shape=(2,)), keras.layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.SGD(0.01), loss="mse")
    model.fit(
        np.ones((4, 2), "float32"),
        np.zeros((4, 1), "float32"),
        epochs=2,
        verbose=0,
        callbacks=[_Poison(), TrainingCurvesCallback(str(tmp_path))],
    )
    assert (tmp_path / "loss.png").is_file()


# ---------------------------------------------------------------------------
# Figure lifecycle
# ---------------------------------------------------------------------------
def test_figures_are_closed_when_saving_fails(tmp_path, monkeypatch):
    plt = import_pyplot()
    plt.close("all")

    def _boom(self, *a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", _boom)
    callback = TrainingCurvesCallback(str(tmp_path))
    _drive(callback, [{"loss": 1.0}])  # failure is swallowed (visualization only)
    assert plt.get_fignums() == [], "the figure leaked into pyplot's registry"


def test_no_figure_is_left_open_after_a_normal_epoch(tmp_path):
    plt = import_pyplot()
    plt.close("all")
    callback = TrainingCurvesCallback(str(tmp_path))
    _drive(callback, [{"loss": 1.0, "val_loss": 1.1}])
    assert plt.get_fignums() == []


# ---------------------------------------------------------------------------
# The documented layout is the real one
# ---------------------------------------------------------------------------
def test_the_files_land_where_the_docstring_says(tmp_path):
    callback = TrainingCurvesCallback(str(tmp_path), file_prefix="run_")
    _drive(callback, [{"loss": 1.0, "map50": 0.1}])
    assert (tmp_path / "run_loss.png").is_file()
    assert (tmp_path / "run_detection.png").is_file()
    assert not (tmp_path / "training_curves").exists()


def test_the_docstrings_name_the_real_path_and_every_group():
    emitted = _group_metrics(
        ["loss", "classification_acc", "segmentation_pix_acc", "map50", "abs_rel", "lr"]
    )
    assert set(emitted) == {
        "loss", "classification", "segmentation", "detection", "depth", "other",
    }
    docs = (training_curves.__doc__ or "") + (TrainingCurvesCallback.__doc__ or "")
    assert "training_curves/" not in (training_curves.__doc__ or ""), (
        "the module docstring still promises a training_curves/ subdirectory"
    )
    assert "{prefix}{group}.png" in (training_curves.__doc__ or "")
    for group in emitted:
        assert f"``{group}.png``" in (TrainingCurvesCallback.__doc__ or ""), group
        assert group in docs
