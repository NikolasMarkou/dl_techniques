"""The standard analysis schedule of ``EpochAnalyzerCallback``.

Guard for plan-2026-10-23T100412-e7eaaa02 (D-003). The standard is one snapshot
before training starts (``epoch_000``), one after epoch 1, then every
``epoch_frequency`` epochs counted from ``start_epoch``. Both extra snapshots can
be switched off, which restores the earlier schedule exactly.
"""

import keras
import pytest

import dl_techniques.callbacks.analyzer_callback as callback_module
from dl_techniques.callbacks.analyzer_callback import EpochAnalyzerCallback


def _tiny_model(build=True):
    inputs = keras.Input((4,))
    return keras.Model(inputs, keras.layers.Dense(2)(inputs))


class _Recorder:
    """Stands in for ``ModelAnalyzer`` and records the directories it is given."""

    dirs = []

    def __init__(self, models, config, output_dir):
        _Recorder.dirs.append(output_dir.rsplit("epoch_", 1)[1])

    def analyze(self):
        return None


@pytest.fixture(autouse=True)
def _recorder(monkeypatch):
    _Recorder.dirs = []
    monkeypatch.setattr(callback_module, "ModelAnalyzer", _Recorder)


def _run(tmp_path, epochs, **kwargs):
    callback = EpochAnalyzerCallback(output_dir=str(tmp_path), **kwargs)
    callback.set_model(_tiny_model())
    callback.on_train_begin()
    for epoch in range(epochs):
        callback.on_epoch_end(epoch)
    return list(_Recorder.dirs)


def test_the_standard_schedule_is_initial_then_epoch_one_then_every_n(tmp_path):
    got = _run(tmp_path, 25, epoch_frequency=10, start_epoch=10)
    assert got == ["000", "001", "010", "020"]


def test_an_every_epoch_run_gains_only_the_initial_snapshot(tmp_path):
    got = _run(tmp_path, 3)
    assert got == ["000", "001", "002", "003"]


def test_switching_both_off_restores_the_earlier_schedule(tmp_path):
    got = _run(
        tmp_path, 25, epoch_frequency=10, start_epoch=10,
        analyze_on_train_begin=False, analyze_first_epoch=False,
    )
    assert got == ["010", "020"]
    _Recorder.dirs = []
    got = _run(
        tmp_path, 25, epoch_frequency=10,
        analyze_on_train_begin=False, analyze_first_epoch=False,
    )
    assert got == ["001", "011", "021"]


def test_the_initial_snapshot_needs_a_built_model(tmp_path, caplog):
    callback = EpochAnalyzerCallback(output_dir=str(tmp_path))
    callback.set_model(keras.Sequential([keras.layers.Dense(2)]))
    callback.on_train_begin()
    assert _Recorder.dirs == []


def test_a_failed_initial_analysis_raises(monkeypatch, tmp_path):
    class Failing:
        def __init__(self, *a, **k):
            pass

        def analyze(self):
            raise RuntimeError("SVD did not converge")

    monkeypatch.setattr(callback_module, "ModelAnalyzer", Failing)
    callback = EpochAnalyzerCallback(output_dir=str(tmp_path))
    callback.set_model(_tiny_model())
    with pytest.raises(RuntimeError, match="SVD did not converge"):
        callback.on_train_begin()
