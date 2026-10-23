"""The CliffordNet trainer's per-epoch model analysis is opt-in and lands on N, 2N, ...

Guard for plan-2026-10-23T100412-e7eaaa02 (D-002). Before the flag, the trainer
passed ``include_analyzer=False`` and analysed once after ``fit``. With
``--epoch-analysis-every N`` the shared ``EpochAnalyzerCallback`` must run at
epochs N, 2N, ... : its default ``start_epoch=1`` would instead give 1, N+1, ...
"""

import sys

import keras
import numpy as np
import pytest

from dl_techniques.callbacks import analyzer_callback
from train.common.callbacks import create_callbacks


class _Stop(Exception):
    """Raised by the stubbed ``create_callbacks`` once it has its arguments."""


def _tiny_cifar(*_args, **_kwargs):
    rng = np.random.default_rng(0)
    x = rng.random((8, 32, 32, 3), dtype=np.float32)
    y = rng.integers(0, 10, size=(8,))
    return (x, y), (x, y), (32, 32, 3), 10


def _tiny_classifier(**_kwargs):
    inputs = keras.Input((32, 32, 3))
    x = keras.layers.GlobalAveragePooling2D()(inputs)
    return keras.Model(inputs, keras.layers.Dense(10)(x))


def _callbacks_kwargs(monkeypatch, tmp_path, extra_argv):
    from train.cliffordnet import train_cliffordnet as module

    seen = {}

    def fake_create_callbacks(**kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(module, "create_callbacks", fake_create_callbacks)
    monkeypatch.setattr(module, "setup_gpu", lambda *a, **k: None)
    monkeypatch.setattr(module, "load_dataset", _tiny_cifar)
    monkeypatch.setattr(module, "build_model", _tiny_classifier)
    monkeypatch.setattr(
        sys, "argv",
        ["prog", "--dataset", "cifar10", "--epochs", "1", "--batch-size", "4",
         "--output-dir", str(tmp_path), *extra_argv],
    )
    with pytest.raises(_Stop):
        module.main()
    return seen


def test_the_analyzer_is_off_by_default(monkeypatch, tmp_path):
    seen = _callbacks_kwargs(monkeypatch, tmp_path, [])
    assert seen["include_analyzer"] is False


def test_the_flag_reaches_create_callbacks(monkeypatch, tmp_path):
    seen = _callbacks_kwargs(
        monkeypatch, tmp_path, ["--epoch-analysis-every", "10"]
    )
    assert seen["include_analyzer"] is True
    assert seen["analyzer_epoch_frequency"] == 10
    assert seen["analyzer_start_epoch"] == 10


def test_the_analyzer_fires_on_multiples_of_the_flag(monkeypatch, tmp_path):
    seen = _callbacks_kwargs(
        monkeypatch, tmp_path, ["--epoch-analysis-every", "10"]
    )
    fired = []

    class _Analyzer:
        def __init__(self, models, config, output_dir):
            fired.append(output_dir.rsplit("epoch_", 1)[1])

        def analyze(self):
            return None

    monkeypatch.setattr(analyzer_callback, "ModelAnalyzer", _Analyzer)
    callbacks, _ = create_callbacks(**{**seen, "output_root": str(tmp_path)})
    analyzer = next(
        c for c in callbacks
        if isinstance(c, analyzer_callback.EpochAnalyzerCallback)
    )
    for epoch in range(25):
        analyzer.on_epoch_end(epoch)
    assert fired == ["010", "020"]
