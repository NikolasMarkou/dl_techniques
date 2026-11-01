"""An attempted analysis that left no result is recorded with its reason, and survives the rest.

Plan 2faf890a step 14.3, review-iter-1 concern 2, D-027. Before this change:

* an exception inside the information-flow analyzer (swallowed per model) or a spectral analysis
  that found no layer left no trace in the result object or the file;
* a weights or training-dynamics analyzer that raised, or a figure that raised, escaped
  ``analyze()`` before ``analysis_results.json`` was written, so every computed result was lost
  and the caller could only say ``FileNotFoundError ... analysis_results.json``.

Now ``AnalysisResults.analysis_failures`` maps analysis -> model -> reason, the analyzers fill it
where they swallow, ``ModelAnalyzer`` fills it for what produced nothing and for what raised, the
file is written BEFORE any figure, and the first exception is still re-raised so a direct caller
of ``analyze()`` is not told that a bug in an analyzer is fine.

Every test runs the REAL analyzers on a tiny model and injects the failure at the place named.
The file of a clean run carries no ``analysis_failures`` key, which is what keeps the legacy
artifacts byte-equal.
"""

import json

import keras
import numpy as np
import pytest

from dl_techniques.analyzer import AnalysisConfig, DataInput, ModelAnalyzer
from dl_techniques.analyzer.analyzers.information_flow_analyzer import InformationFlowAnalyzer
from dl_techniques.analyzer.analyzers.spectral_analyzer import SpectralAnalyzer
from dl_techniques.analyzer.analyzers.training_dynamics_analyzer import TrainingDynamicsAnalyzer
from dl_techniques.analyzer.analyzers.weight_analyzer import WeightAnalyzer
from dl_techniques.analyzer.visualizers.calibration_visualizer import CalibrationVisualizer
from dl_techniques.analyzer.visualizers.summary_visualizer import SummaryVisualizer

HISTORY = {"loss": [1.0, 0.8, 0.7], "val_loss": [1.1, 0.9, 0.85],
           "accuracy": [0.3, 0.5, 0.6], "val_accuracy": [0.3, 0.45, 0.5]}


def _model(width=4):
    keras.utils.set_random_seed(0)
    inputs = keras.Input((8, 8, 3))
    h = keras.layers.Conv2D(width, 3, activation="relu")(inputs)
    h = keras.layers.GlobalAveragePooling2D()(h)
    model = keras.Model(inputs, keras.layers.Dense(3)(h), name="m")
    model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    return model


def _data(n=24):
    rng = np.random.default_rng(0)
    return DataInput(x_data=rng.random((n, 8, 8, 3), dtype=np.float32),
                     y_data=rng.integers(0, 3, n).astype(np.int32))


def _config(**changes):
    kwargs = dict(analyze_weights=True, analyze_calibration=True, analyze_information_flow=True,
                  analyze_training_dynamics=True, analyze_spectral=False, verbose=False,
                  random_state=0, analyze_fisher=False)
    kwargs.update(changes)
    return AnalysisConfig(**kwargs)


def _analyzer(tmp_path, model=None, config=None, history=HISTORY):
    return ModelAnalyzer({"m": model or _model()}, config or _config(), output_dir=str(tmp_path / "out"),
                         training_history={"m": history} if history is not None else None)


def _boom(*args, **kwargs):
    raise RuntimeError("INJECTED")


def _file(analyzer):
    return json.loads((analyzer.output_dir / "analysis_results.json").read_text())


# ---------------------------------------------------------------------
# a clean run records nothing and its file has no new key
# ---------------------------------------------------------------------

def test_a_clean_run_records_no_failure_and_its_file_has_no_failure_key(tmp_path):
    analyzer = _analyzer(tmp_path)
    results = analyzer.analyze(_data())
    assert results.analysis_failures == {}
    assert "analysis_failures" not in _file(analyzer)


# ---------------------------------------------------------------------
# an analyzer that swallows: the reason is the exception text
# ---------------------------------------------------------------------

def test_a_swallowed_information_flow_error_is_recorded_with_its_text(tmp_path, monkeypatch):
    monkeypatch.setattr(InformationFlowAnalyzer, "_analyze_layer_information", _boom)
    analyzer = _analyzer(tmp_path)
    results = analyzer.analyze(_data())  # swallowed per model, as before: no exception
    assert results.analysis_failures == {"information_flow": {"m": "RuntimeError: INJECTED"}}
    assert "m" not in results.information_flow
    assert _file(analyzer)["analysis_failures"] == {"information_flow": {"m": "RuntimeError: INJECTED"}}


def test_a_calibration_that_could_not_cache_its_predictions_names_the_real_cause(tmp_path):
    """Void labels with no ``ignore_label``: the cause is the label range, not 'cache has logits'."""
    keras.utils.set_random_seed(0)
    inputs = keras.Input((4, 4, 3))
    model = keras.Model(inputs, keras.layers.Conv2D(3, 1)(inputs), name="m")
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, (6, 4, 4)).astype(np.int32)
    y[:, 0, 0] = 255
    analyzer = ModelAnalyzer({"m": model}, _config(analyze_weights=False, analyze_information_flow=False,
                                                   analyze_training_dynamics=False),
                             output_dir=str(tmp_path / "out"))
    results = analyzer.analyze(DataInput(x_data=rng.random((6, 4, 4, 3), dtype=np.float32), y_data=y))
    reason = results.analysis_failures["calibration"]["m"]
    assert reason.startswith("ValueError:") and "255" in reason
    assert "logits" not in reason


def test_a_prediction_cache_that_could_not_be_built_names_its_own_error(tmp_path):
    """One-hot masks fail while the cache is built; the reason is that error, not 'cache has logits'."""
    keras.utils.set_random_seed(0)
    inputs = keras.Input((4, 4, 3))
    model = keras.Model(inputs, keras.layers.Conv2D(3, 1)(inputs), name="m")
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    rng = np.random.default_rng(0)
    one_hot = np.eye(3, dtype=np.float32)[rng.integers(0, 3, (6, 4, 4))]
    config = _config(analyze_weights=False, analyze_information_flow=False, analyze_training_dynamics=False)
    analyzer = ModelAnalyzer({"m": model}, config, output_dir=str(tmp_path / "out"))
    results = analyzer.analyze(DataInput(x_data=rng.random((6, 4, 4, 3), dtype=np.float32), y_data=one_hot))
    reason = results.analysis_failures["calibration"]["m"]
    assert reason.startswith("ValueError: per-pixel labels of shape (6, 4, 4, 3)")
    assert "logits" not in reason


def test_spectral_that_finds_no_layer_says_why(tmp_path):
    """A conv with 4 filters has 4 eigenvalues, below ``spectral_min_evals`` (10): no ESD tail to fit."""
    analyzer = _analyzer(tmp_path, config=_config(analyze_spectral=True))
    results = analyzer.analyze(_data())
    reason = results.analysis_failures["spectral"]["m"]
    assert "spectral_min_evals=10" in reason and "no layer qualifies" in reason
    assert "m" not in results.spectral_summary_per_model


def test_spectral_finds_layers_on_the_segmentation_family():
    """The review saw 'No layers found' on a stand-in mask net whose convolutions have 8, 8 and 3
    filters. The real family is not in that position: ConvUNext tiny has 35 qualifying layers."""
    from dl_techniques.models.vision.convunext.model import create_convunext_variant

    model = create_convunext_variant("tiny", (16, 16, 3), use_bias=True, output_channels=3,
                                     final_activation="linear")
    details, layers = SpectralAnalyzer({"u": model}, _config(analyze_spectral=True))._describe_model(model)
    assert len(layers) > 100 and len(details) >= 30


def test_training_dynamics_without_a_history_is_recorded(tmp_path):
    analyzer = _analyzer(tmp_path, history=None)
    results = analyzer.analyze(_data())
    assert results.analysis_failures["training_dynamics"] == {"*": "no training history was given"}


def test_a_weights_analysis_that_produces_nothing_is_recorded(tmp_path):
    """A model without a weight tensor: the analyzer runs and ``weight_stats['m']`` stays empty."""
    inputs = keras.Input((8, 8, 3))
    model = keras.Model(inputs, keras.layers.Flatten()(inputs), name="m")
    config = _config(analyze_calibration=False, analyze_information_flow=False,
                     analyze_training_dynamics=False)
    results = _analyzer(tmp_path, model=model, config=config).analyze(_data())
    assert results.analysis_failures["weights"]["m"] == "the analysis ran and produced no result for this model"


# ---------------------------------------------------------------------
# an analyzer that raises: the other results survive, the file is written, the error is re-raised
# ---------------------------------------------------------------------

@pytest.mark.parametrize("analysis, target", [
    ("weights", (WeightAnalyzer, "analyze")),
    ("training_dynamics", (TrainingDynamicsAnalyzer, "analyze")),
])
def test_an_analyzer_that_raises_loses_nothing_else_and_still_raises(tmp_path, monkeypatch, analysis, target):
    monkeypatch.setattr(*target, _boom)
    analyzer = _analyzer(tmp_path)
    with pytest.raises(RuntimeError, match="INJECTED"):
        analyzer.analyze(_data())
    payload = _file(analyzer)
    assert payload["analysis_failures"][analysis] == {"*": "RuntimeError: INJECTED"}
    assert "m" in payload["calibration_metrics"] and "m" in payload["information_flow"]
    assert (analyzer.output_dir / "confidence_calibration_analysis.png").is_file()


def test_the_file_exists_before_the_first_figure_is_drawn(tmp_path, monkeypatch):
    seen = {}

    def figure(self):
        seen["file_was_there"] = (self.output_dir / "analysis_results.json").is_file()
        seen["failures_key"] = "analysis_failures" in json.loads(
            (self.output_dir / "analysis_results.json").read_text())

    monkeypatch.setattr(CalibrationVisualizer, "create_visualizations", figure)
    _analyzer(tmp_path).analyze(_data())
    assert seen == {"file_was_there": True, "failures_key": False}


@pytest.mark.parametrize("visualizer, name", [
    (CalibrationVisualizer, "visualizer:calibration"),
    (SummaryVisualizer, "visualizer:summary"),
])
def test_a_figure_that_raises_loses_no_number_and_is_named(tmp_path, monkeypatch, visualizer, name):
    monkeypatch.setattr(visualizer, "create_visualizations", _boom)
    analyzer = _analyzer(tmp_path)
    results = analyzer.analyze(_data())  # a figure never fails the analysis
    assert results.analysis_failures[name] == {"*": "RuntimeError: INJECTED"}
    payload = _file(analyzer)
    assert payload["analysis_failures"][name] == {"*": "RuntimeError: INJECTED"}
    assert "m" in payload["calibration_metrics"] and "m" in payload["weight_stats"]


def test_an_error_before_any_analyzer_ran_is_written_down_then_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(ModelAnalyzer, "_cache_predictions", _boom)
    analyzer = _analyzer(tmp_path)
    with pytest.raises(RuntimeError, match="INJECTED"):
        analyzer.analyze(_data())
    assert _file(analyzer)["analysis_failures"] == {"analysis": {"*": "RuntimeError: INJECTED"}}
