"""The CliffordNet trainer writes the standard classification visualizations.

Guard for plan-2026-10-23T100412-e7eaaa02 (D-004). Before this, the trainer wrote
only the end-of-run ``model_analysis/`` plots; the ConvNeXt trainers also write a
live training dashboard and the confusion matrix, per-class metrics, calibration
and misclassification figures plus a classification report under
``visualizations/``. This runs the real ``main()`` end to end on a tiny stubbed
dataset and model and reads the run directory it produced.
"""

import sys
from pathlib import Path

import keras
import numpy as np
import pytest

EXPECTED = [
    "training_dashboard.png",
    "confusion_matrix.png",
    "per_class_metrics.png",
    "confidence_calibration.png",
    "misclassifications.png",
    "classification_report.json",
]


def _tiny_cifar(*_args, **_kwargs):
    rng = np.random.default_rng(0)
    x_train = rng.random((64, 32, 32, 3), dtype=np.float32)
    y_train = rng.integers(0, 10, size=(64,))
    x_test = rng.random((32, 32, 32, 3), dtype=np.float32)
    y_test = rng.integers(0, 10, size=(32,))
    return (x_train, y_train), (x_test, y_test), (32, 32, 3), 10


def _tiny_classifier(**_kwargs):
    inputs = keras.Input((32, 32, 3))
    x = keras.layers.Conv2D(4, 3, strides=4)(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    return keras.Model(inputs, keras.layers.Dense(10)(x))


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    from train.cliffordnet import train_cliffordnet as module

    out = tmp_path_factory.mktemp("cliffordnet_run")
    mp = pytest.MonkeyPatch()
    mp.setattr(module, "setup_gpu", lambda *a, **k: None)
    mp.setattr(module, "load_dataset", _tiny_cifar)
    mp.setattr(module, "build_model", _tiny_classifier)
    mp.setattr(module, "run_model_analysis", lambda *a, **k: None)
    mp.setattr(
        sys, "argv",
        ["prog", "--dataset", "cifar10", "--epochs", "2", "--batch-size", "8",
         "--output-dir", str(out)],
    )
    try:
        module.main()
    finally:
        mp.undo()
    (run,) = [p for p in out.iterdir() if p.is_dir()]
    return run


@pytest.mark.parametrize("name", EXPECTED)
def test_the_visualization_is_written(run_dir, name):
    path = Path(run_dir) / "visualizations" / name
    assert path.is_file() and path.stat().st_size > 0


def test_the_learning_rate_reaches_the_csv(run_dir):
    header = (Path(run_dir) / "training_log.csv").read_text().splitlines()[0]
    assert "lr" in header.split(",")
