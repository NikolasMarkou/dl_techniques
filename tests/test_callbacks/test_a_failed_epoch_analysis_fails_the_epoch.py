"""A failed ``EpochAnalyzerCallback`` analysis fails the epoch.

Guard for plan-2026-10-14T062616-81938e81 step 13.7 (b) (D-117). Measured at
b50fd4e80: ``on_epoch_end`` wrapped ``ModelAnalyzer(...).analyze()`` in
``try/except Exception``, logged the error and returned, so ``fit`` went on and
the run exited 0 with the epoch's analysis missing. The one recorded instance in
this plan is the Step 13 DarkIR smoke, whose analysis failed with "SVD did not
converge" on NaN weights at both epochs while the run reported success.

The success case runs the real analyzer on a tiny model, so the stub is not
what makes the failing case raise.
"""

import os

import keras
import numpy as np
import pytest

import dl_techniques.callbacks.analyzer_callback as callback_module
from dl_techniques.callbacks.analyzer_callback import EpochAnalyzerCallback


class _AnalysisFailed(RuntimeError):
    """What the patched analyzer raises."""


def _tiny_model():
    inputs = keras.Input((4,))
    hidden = keras.layers.Dense(8, name="hidden")(inputs)
    return keras.Model(inputs, keras.layers.Dense(2, name="out")(hidden))


def test_a_failed_analysis_raises_out_of_on_epoch_end(monkeypatch, tmp_path):
    class FailingAnalyzer:
        def __init__(self, *args, **kwargs):
            pass

        def analyze(self, *args, **kwargs):
            raise _AnalysisFailed("SVD did not converge")

    monkeypatch.setattr(callback_module, "ModelAnalyzer", FailingAnalyzer)
    callback = EpochAnalyzerCallback(output_dir=str(tmp_path))
    callback.set_model(_tiny_model())
    with pytest.raises(_AnalysisFailed, match="SVD did not converge"):
        callback.on_epoch_end(0, logs={})


def test_a_failed_analysis_stops_fit(monkeypatch, tmp_path):
    """The exception leaves ``fit``, so the run's exit code says so."""
    class FailingAnalyzer:
        def __init__(self, *args, **kwargs):
            pass

        def analyze(self, *args, **kwargs):
            raise _AnalysisFailed("analysis failed")

    monkeypatch.setattr(callback_module, "ModelAnalyzer", FailingAnalyzer)
    model = _tiny_model()
    model.compile(optimizer="sgd", loss="mse")
    x = np.zeros((8, 4), dtype="float32")
    y = np.zeros((8, 2), dtype="float32")
    with pytest.raises(_AnalysisFailed):
        model.fit(x, y, epochs=2, verbose=0,
                  callbacks=[EpochAnalyzerCallback(output_dir=str(tmp_path))])


def test_a_successful_analysis_writes_its_epoch_directory(tmp_path):
    callback = EpochAnalyzerCallback(output_dir=str(tmp_path))
    callback.set_model(_tiny_model())
    callback.on_epoch_end(0, logs={})
    epoch_dir = tmp_path / "epoch_001"
    assert epoch_dir.is_dir()
    assert os.listdir(epoch_dir), "the analysis wrote nothing"
