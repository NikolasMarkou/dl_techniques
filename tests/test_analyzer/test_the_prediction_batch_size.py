"""``AnalysisConfig.prediction_batch_size``: the analyzer's own ``predict`` can be given a batch (D-040).

The analyzer predicted at Keras' default of 32 images a pass. For a 224 px model that is 2.5 GB
(measured on ``cliffordnet/nano``), too much for a run whose recipe trains at batch 8, so the field
lets a caller lower it. ``None`` (the default) passes nothing, so every existing caller predicts
exactly as before, and the published ``config`` block of ``analysis_results.json`` gains the key
ONLY when it is set: an artifact of a run that never set it is the artifact it always was.

Tests write only under ``tmp_path``.
"""

import json

import keras
import numpy as np
import pytest

from dl_techniques.analyzer import AnalysisConfig, DataInput, ModelAnalyzer

N_IMAGES = 40
N_CLASSES = 5


def _model():
    model = keras.Sequential([keras.layers.Input((6,)), keras.layers.Dense(N_CLASSES)])
    model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    return model


def _config(**overrides):
    kwargs = dict(
        analyze_weights=False, analyze_calibration=True, analyze_information_flow=False,
        analyze_training_dynamics=False, analyze_spectral=False, verbose=False,
        random_state=0, n_samples=N_IMAGES, save_plots=False,
    )
    kwargs.update(overrides)
    return AnalysisConfig(**kwargs)


def _data():
    rng = np.random.default_rng(0)
    return DataInput(
        x_data=rng.random((N_IMAGES, 6), dtype=np.float32),
        y_data=rng.integers(0, N_CLASSES, N_IMAGES),
    )


def _analyze(tmp_path, monkeypatch, **overrides):
    """Run the calibration analysis; return ``(batch_size kwargs of every predict, config block)``."""
    calls = []
    original = keras.Model.predict

    def spy(model, x, *args, **kwargs):
        calls.append(kwargs.get("batch_size", "unset"))
        return original(model, x, *args, **kwargs)

    monkeypatch.setattr(keras.Model, "predict", spy)
    out = tmp_path / "analysis"
    ModelAnalyzer({"m": _model()}, _config(**overrides), output_dir=str(out)).analyze(data=_data())
    monkeypatch.setattr(keras.Model, "predict", original)
    block = json.loads((out / "analysis_results.json").read_text())["config"]
    return calls, block


def test_the_default_passes_no_batch_and_the_artifact_has_no_new_key(tmp_path, monkeypatch):
    """Mutant ``the field is always published``: every artifact would gain a key."""
    calls, block = _analyze(tmp_path, monkeypatch)
    assert calls == [None]  # one cache pass, at Keras' own default
    assert "prediction_batch_size" not in block
    assert "calibration_bins" in block  # liveness: the block is the real one


def test_a_set_value_is_the_batch_of_the_cached_predictions_and_is_published(tmp_path, monkeypatch):
    """Mutant ``the field is ignored``: the analyzer would predict at 32 whatever was asked."""
    calls, block = _analyze(tmp_path, monkeypatch, prediction_batch_size=4)
    assert calls == [4]
    assert block["prediction_batch_size"] == 4


def test_the_batch_changes_no_number_it_only_bounds_memory(tmp_path):
    """A smaller batch is the same function of the same images: the calibration is equal."""
    def run(sub, **overrides):
        out = tmp_path / sub
        analyzer = ModelAnalyzer({"m": _model_fixed()}, _config(**overrides), output_dir=str(out))
        return analyzer.analyze(data=_data()).calibration_metrics["m"]

    base = run("a")
    small = run("b", prediction_batch_size=3)
    for key in ("ece", "brier_score"):
        assert small[key] == pytest.approx(base[key], abs=1e-6)


_FIXED = {}


def _model_fixed():
    """One model for the equality test, so two analyzers see the same weights."""
    if "m" not in _FIXED:
        _FIXED["m"] = _model()
    return _FIXED["m"]
