"""Tests for :class:`~dl_techniques.analyzer.model_analyzer.ModelAnalyzer`.

``model_analyzer.py`` is the public entry point the package README documents, and it had
zero direct test coverage. Every test in this module is a guard for a defect that was
measured against the shipped code before its fix landed; each class docstring names the
wrong value that was actually observed.
"""

import json

import keras
import numpy as np
import pytest

from dl_techniques.analyzer.calibration_metrics import compute_prediction_entropy_stats
from dl_techniques.analyzer.config import AnalysisConfig
from dl_techniques.analyzer.data_types import DataInput
from dl_techniques.analyzer.model_analyzer import ModelAnalyzer

N_SAMPLES = 40
N_FEATURES = 6
N_CLASSES = 3


def _quiet_config(**overrides) -> AnalysisConfig:
    """Build an AnalysisConfig with every analysis off and plotting disabled.

    Individual tests switch back on only the analysis they need, which keeps each
    guard pointed at one code path and keeps the module fast.

    Args:
        **overrides: Field values to override on the returned config.

    Returns:
        AnalysisConfig: A configuration with all five analysis toggles disabled.
    """
    defaults = dict(
        analyze_weights=False,
        analyze_calibration=False,
        analyze_information_flow=False,
        analyze_training_dynamics=False,
        analyze_spectral=False,
        n_samples=N_SAMPLES,
        save_plots=False,
        verbose=False,
    )
    defaults.update(overrides)
    return AnalysisConfig(**defaults)


def _build_classifier(name: str, seed: int) -> keras.Model:
    """Build a tiny softmax classifier with deterministic weights."""
    keras.utils.set_random_seed(seed)
    inputs = keras.Input(shape=(N_FEATURES,), name=f"{name}_in")
    x = keras.layers.Dense(8, activation="relu", name=f"{name}_d1")(inputs)
    outputs = keras.layers.Dense(N_CLASSES, activation="softmax", name=f"{name}_out")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name=name)


@pytest.fixture()
def probe_data() -> DataInput:
    """Deterministic (x, y) with integer labels over ``N_CLASSES``."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N_SAMPLES, N_FEATURES)).astype("float32")
    y = rng.integers(0, N_CLASSES, size=N_SAMPLES)
    return DataInput(x_data=x, y_data=y)


# ---------------------------------------------------------------------
# C1 -- entropy_std
# ---------------------------------------------------------------------

class TestEntropyStdKey:
    """``get_summary_statistics`` must report the producer's entropy spread.

    Defect guarded (C1): ``model_analyzer.py:782`` read ``metrics.get('entropy_std', ...)``
    while the producer ``calibration_metrics.compute_prediction_entropy_stats`` writes
    ``std_entropy``. The key never resolved, so the reported value was a hard
    ``DEFAULT_METRIC_VALUE`` of 0.0 sitting beside a live ``mean_entropy``.
    """

    def test_entropy_std_equals_the_producers_std_entropy(self, tmp_path, probe_data):
        model = _build_classifier("c1_model", seed=11)
        analyzer = ModelAnalyzer(
            models={"c1_model": model},
            config=_quiet_config(analyze_calibration=True),
            output_dir=str(tmp_path / "c1"),
        )
        analyzer.analyze(probe_data, analysis_types={"calibration"})
        summary = analyzer.get_summary_statistics()

        # Re-derive the expected spread from the model's own probabilities rather
        # than from the results dict, so the guard does not compare a value to itself.
        probabilities = model.predict(probe_data.x_data, verbose=0)
        expected = float(compute_prediction_entropy_stats(probabilities)["std_entropy"])

        confidence = summary["confidence_summary"]["c1_model"]

        # The producer's own spread must be non-degenerate, otherwise this guard
        # would pass against the broken 0.0 default for the wrong reason.
        assert expected > 1e-6, (
            "anti-vacuity: the probe's entropy spread is ~0, so a hard 0.0 would "
            f"pass either way (std_entropy={expected})"
        )
        assert confidence["entropy_std"] == pytest.approx(expected, rel=1e-9), (
            "entropy_std is not the producer's std_entropy: "
            f"reported={confidence['entropy_std']} expected={expected}"
        )


# ---------------------------------------------------------------------
# C8 -- the multi-input warning
# ---------------------------------------------------------------------

def _build_multi_input_model(name: str = "mi_model") -> keras.Model:
    """A two-input functional model, so ``_identify_multi_input_models`` flags it."""
    keras.utils.set_random_seed(7)
    left = keras.Input(shape=(N_FEATURES,), name="left")
    right = keras.Input(shape=(N_FEATURES,), name="right")
    merged = keras.layers.Concatenate(name=f"{name}_cat")([left, right])
    outputs = keras.layers.Dense(N_CLASSES, activation="softmax", name=f"{name}_out")(merged)
    return keras.Model(inputs=[left, right], outputs=outputs, name=name)


class TestMultiInputWarning:
    """The limited-analysis warning for multi-input models must actually fire.

    Defect guarded (C8): ``analyze()`` computed
    ``affected_models = analysis_types & self._multi_input_models``. ``analysis_types``
    holds ANALYSIS-TYPE names ('weights', 'calibration', ...) while
    ``_multi_input_models`` holds MODEL names, so the intersection was empty unless a
    model happened to be named after an analysis type. ``if affected_models:`` was
    therefore always false and the warning was unreachable.
    """

    def test_warning_names_the_multi_input_model(self, tmp_path, caplog):
        model = _build_multi_input_model()
        rng = np.random.default_rng(3)
        x = {
            "left": rng.standard_normal((N_SAMPLES, N_FEATURES)).astype("float32"),
            "right": rng.standard_normal((N_SAMPLES, N_FEATURES)).astype("float32"),
        }
        y = rng.integers(0, N_CLASSES, size=N_SAMPLES)

        analyzer = ModelAnalyzer(
            models={"mi_model": model},
            config=_quiet_config(analyze_information_flow=True),
            output_dir=str(tmp_path / "c8"),
        )
        assert analyzer._multi_input_models == {"mi_model"}, (
            "anti-vacuity: the probe was not detected as multi-input, so the warning "
            "would be correctly absent for a reason unrelated to the defect"
        )

        with caplog.at_level("WARNING"):
            analyzer.analyze(
                DataInput(x_data=x, y_data=y), analysis_types={"information_flow"}
            )

        # Match the WARNING specifically. The constructor also logs an INFO record
        # containing 'multi-input' ("Detected multi-input models: ..."), and a filter
        # loose enough to catch that one passes against the unfixed code.
        limited = [
            record.getMessage()
            for record in caplog.records
            if record.levelname == "WARNING"
            and "limited" in record.getMessage().lower()
            and "multi-input" in record.getMessage().lower()
        ]
        assert limited, (
            "no limited-analysis warning was emitted for a multi-input model; "
            "records seen: "
            f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
        )
        assert any("mi_model" in message for message in limited), (
            f"the warning fired but does not name the affected model: {limited}"
        )


# ---------------------------------------------------------------------
# C11 -- DataInput.from_object attribute spellings
# ---------------------------------------------------------------------

class _ReadmeShapedData:
    """The spelling the README and ``ModelAnalyzer.analyze``'s docstring promise."""

    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y


class _DocstringShapedData:
    """The spelling ``from_object``'s own docstring promised."""

    def __init__(self, x, y):
        self.x_test = x
        self.y_test = y


class TestDataInputFromObject:
    """``from_object`` must accept both documented attribute spellings.

    Defect guarded (C11): the body was ``cls(x_data=data.x_test, y_data=data.y_test)``,
    but ``README.md`` and ``model_analyzer.py``'s ``analyze()`` docstring both promise
    "any object with x_data/y_data attributes". An object shaped the way the README
    documents raised ``AttributeError``. Both spellings are documented, so the fix keeps
    both working rather than editing one of the docs.
    """

    @pytest.mark.parametrize(
        "carrier", [_ReadmeShapedData, _DocstringShapedData], ids=["x_data", "x_test"]
    )
    def test_both_documented_spellings_resolve(self, carrier):
        x = np.arange(12, dtype="float32").reshape(4, 3)
        y = np.arange(4)

        resolved = DataInput.from_object(carrier(x, y))

        np.testing.assert_array_equal(resolved.x_data, x)
        np.testing.assert_array_equal(resolved.y_data, y)

    def test_x_data_wins_when_an_object_carries_both(self):
        """``x_data``/``y_data`` is the preferred spelling, so it must take priority."""
        preferred_x = np.zeros((2, 2), dtype="float32")
        fallback_x = np.ones((2, 2), dtype="float32")

        carrier = _ReadmeShapedData(preferred_x, np.zeros(2))
        carrier.x_test = fallback_x
        carrier.y_test = np.ones(2)

        resolved = DataInput.from_object(carrier)
        np.testing.assert_array_equal(resolved.x_data, preferred_x)

    def test_an_object_with_neither_spelling_raises_a_named_error(self):
        class _Empty:
            pass

        with pytest.raises(AttributeError, match="x_data.*x_test|x_test.*x_data"):
            DataInput.from_object(_Empty())


# ---------------------------------------------------------------------
# C7 -- create_smoothed_model on recurrent layers
# ---------------------------------------------------------------------

RECURRENT_TIMESTEPS = 5
RECURRENT_FEATURES = 20
RECURRENT_UNITS = 16


def _build_recurrent_model(name: str = "rnn_model") -> keras.Model:
    """A model containing an LSTM, which holds THREE weight tensors."""
    keras.utils.set_random_seed(5)
    inputs = keras.Input(
        shape=(RECURRENT_TIMESTEPS, RECURRENT_FEATURES), name=f"{name}_in"
    )
    x = keras.layers.LSTM(RECURRENT_UNITS, name=f"{name}_lstm")(inputs)
    outputs = keras.layers.Dense(N_CLASSES, activation="softmax", name=f"{name}_out")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name=name)


class TestSmoothedModelRecurrent:
    """``create_smoothed_model`` must not abort on an LSTM/GRU layer.

    Defect guarded (C7): the weight write was
    ``layer.set_weights([new_weights, old_bias] if has_bias else [new_weights])``.
    ``get_layer_weights_and_bias`` reports ``has_bias = False`` for LSTM/GRU (the
    bias branch is reachable only for Dense/Conv/Embedding), so a one-element list
    was handed to a layer holding three weight tensors and Keras raised
    ``ValueError``. Nothing caught it, so the whole call aborted.
    """

    def _analyzer(self, tmp_path):
        model = _build_recurrent_model()
        analyzer = ModelAnalyzer(
            models={"rnn_model": model},
            config=_quiet_config(analyze_spectral=True),
            output_dir=str(tmp_path / "c7"),
        )
        analyzer.analyze(analysis_types={"spectral"})
        return model, analyzer

    def test_create_smoothed_model_handles_lstm(self, tmp_path):
        model, analyzer = self._analyzer(tmp_path)

        # Anti-vacuity: the LSTM must actually be one of the analyzed layers,
        # otherwise the smoothing loop would never reach the recurrent branch.
        analyzed = set(analyzer.results.spectral_analysis["name"])
        assert "rnn_model_lstm" in analyzed, (
            f"the LSTM was not admitted to spectral analysis: {sorted(analyzed)}"
        )

        smoothed = analyzer.create_smoothed_model("rnn_model", method="detX")

        assert smoothed is not None
        lstm = smoothed.get_layer("rnn_model_lstm")
        assert len(lstm.get_weights()) == 3, (
            "the LSTM lost weight tensors during smoothing: "
            f"{len(lstm.get_weights())} instead of 3"
        )

    def test_smoothing_preserves_the_untouched_recurrent_tensors(self, tmp_path):
        """Only the input kernel is smoothed; the other two tensors carry over."""
        model, analyzer = self._analyzer(tmp_path)
        original = model.get_layer("rnn_model_lstm").get_weights()

        smoothed = analyzer.create_smoothed_model("rnn_model", method="detX")
        produced = smoothed.get_layer("rnn_model_lstm").get_weights()

        np.testing.assert_array_equal(
            produced[1], original[1], err_msg="the recurrent kernel was disturbed"
        )
        np.testing.assert_array_equal(
            produced[2], original[2], err_msg="the LSTM bias was disturbed"
        )


# ---------------------------------------------------------------------
# C9 -- the JSON artifact must be parseable
# ---------------------------------------------------------------------

SPECTRAL_FEATURES = 20


def _reject_json_constant(token: str):
    """``parse_constant`` hook: strict JSON has no NaN/Infinity/-Infinity."""
    raise ValueError(f"analysis_results.json contains the non-JSON constant {token!r}")


def _build_spectral_probe(name: str, zeroed: bool) -> keras.Model:
    """A model wide enough for spectral analysis, optionally degenerate.

    Args:
        name: Model name.
        zeroed: When True the hidden kernel is set to all zeros, which makes the
            power-law fit fail and leaves NaN in the erg metric columns.

    Returns:
        keras.Model: The probe.
    """
    keras.utils.set_random_seed(13)
    # Both kernel dimensions must clear `spectral_min_evals` (10): the analyzer gates on
    # M = min(N, M), so the default 6-feature input of the other probes is skipped
    # outright and would make this guard vacuous.
    inputs = keras.Input(shape=(SPECTRAL_FEATURES,), name=f"{name}_in")
    hidden = keras.layers.Dense(16, activation="relu", name=f"{name}_h")
    x = hidden(inputs)
    outputs = keras.layers.Dense(N_CLASSES, activation="softmax", name=f"{name}_out")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    if zeroed:
        weights = hidden.get_weights()
        weights[0] = np.zeros_like(weights[0])
        hidden.set_weights(weights)
    return model


class TestJsonArtifactParseability:
    """``analysis_results.json`` must be strict, parseable JSON.

    Defect guarded (C9): ``json.dump`` ran with ``allow_nan`` at its default True and
    ``convert_numpy`` passed NaN straight through ``float()``, so a failed spectral fit
    wrote a bare ``NaN`` token. The file then could not be read by any strict parser.
    The ``try/except`` around the dump downgrades a write failure to a log line, so this
    guard asserts on the FILE, never on the absence of an exception.
    """

    def _run(self, tmp_path):
        models = {
            "good": _build_spectral_probe("good", zeroed=False),
            "degenerate": _build_spectral_probe("degenerate", zeroed=True),
        }
        analyzer = ModelAnalyzer(
            models=models,
            config=_quiet_config(analyze_spectral=True),
            output_dir=str(tmp_path / "c9"),
        )
        analyzer.analyze(analysis_types={"spectral"})

        # Anti-vacuity: the spectral frame must exist and must contain a FAILED fit,
        # otherwise no NaN is generated and the parse below succeeds trivially.
        frame = analyzer.results.spectral_analysis
        assert frame is not None and not frame.empty, (
            "no spectral analysis was produced, so no NaN can reach the artifact"
        )
        assert frame.isna().to_numpy().sum() > 0, (
            "the probe produced no NaN cells; this guard would pass either way. "
            f"statuses: {frame['status'].tolist() if 'status' in frame else 'n/a'}"
        )
        return analyzer.output_dir / "analysis_results.json"

    def test_the_artifact_parses_under_a_strict_parser(self, tmp_path):
        artifact = self._run(tmp_path)
        assert artifact.exists(), "save_results wrote no artifact at all"

        text = artifact.read_text()
        json.loads(text, parse_constant=_reject_json_constant)

    def test_the_artifact_carries_no_bare_non_finite_tokens(self, tmp_path):
        """A literal-token check, independent of the parser's own leniency."""
        artifact = self._run(tmp_path)
        text = artifact.read_text()

        offenders = [
            token for token in ("NaN", "Infinity", "-Infinity") if token in text
        ]
        assert not offenders, (
            f"analysis_results.json contains bare non-JSON tokens: {offenders}"
        )


# =====================================================================
# P6 - the model walk must read attributes, not evaluate properties
# =====================================================================

_PROPERTY_HITS = []


class PropertyProbeBlock(keras.layers.Layer):
    """Declares `zeta, alpha, mid` and carries two hostile properties.

    `recursively_get_layers` used to sweep `dir(current_layer)` and `getattr` every
    public name inside a blanket `except Exception: continue`. A subclassed Layer
    exposes 52 such names — including `weights`, `variables`, `trainable_weights`,
    `losses`, `input` and `output`, all of which are properties — so the walk paid a
    property evaluation per name and swallowed whatever raised.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.zeta = keras.layers.Dense(5, name="p6_zeta")
        self.alpha = keras.layers.Dense(7, name="p6_alpha")
        self.mid = keras.layers.Dense(6, name="p6_mid")

    @property
    def expensive(self):
        _PROPERTY_HITS.append("expensive")
        return 42

    @property
    def exploding(self):
        _PROPERTY_HITS.append("exploding")
        raise RuntimeError("this property must never be evaluated by the walk")

    def call(self, inputs):
        return self.zeta(self.mid(self.alpha(inputs)))


# Declaration order, read off `PropertyProbeBlock.__init__` above.
_DECLARATION_ORDER = ["p6_zeta", "p6_alpha", "p6_mid"]


def _build_property_probe_model(name: str = "p6_model") -> keras.Model:
    inputs = keras.Input(shape=(N_FEATURES,), name=f"{name}_in")
    return keras.Model(inputs, PropertyProbeBlock(name="p6_blk")(inputs), name=name)


class TestModelWalkDoesNotEvaluateProperties:
    """Guards for `dl_techniques.analyzer.utils.recursively_get_layers`.

    They live in this module because `ModelAnalyzer.create_smoothed_model` indexes
    `all_layers[layer_id]` on a CLONED model, which is what makes the walk's order
    load-bearing.
    """

    def setup_method(self):
        _PROPERTY_HITS.clear()

    def test_no_property_is_evaluated(self):
        from dl_techniques.analyzer.utils import recursively_get_layers

        model = _build_property_probe_model()
        _PROPERTY_HITS.clear()
        recursively_get_layers(model)

        assert _PROPERTY_HITS == [], (
            f"the walk evaluated {len(_PROPERTY_HITS)} properties: {_PROPERTY_HITS}"
        )

    def test_the_hostile_properties_are_reachable(self):
        """Anti-vacuity: the probe's properties really do fire when touched."""
        model = _build_property_probe_model()
        block = [l for l in model.layers if l.name == "p6_blk"][0]
        _PROPERTY_HITS.clear()

        assert block.expensive == 42
        with pytest.raises(RuntimeError):
            _ = block.exploding
        assert _PROPERTY_HITS == ["expensive", "exploding"]

    def test_sublayers_come_back_in_declaration_order(self):
        from dl_techniques.analyzer.utils import recursively_get_layers

        model = _build_property_probe_model()
        names = [l.name for l in recursively_get_layers(model)]
        got = [n for n in names if n in _DECLARATION_ORDER]

        # Anti-vacuity: the probe is non-degenerate — reverse-alphabetical (what the
        # `dir()` sweep produced) is a DIFFERENT order from declaration order.
        assert sorted(_DECLARATION_ORDER, reverse=True) != _DECLARATION_ORDER
        assert got == _DECLARATION_ORDER, (
            f"walk order for the sublayers is {got}, declaration order is "
            f"{_DECLARATION_ORDER}"
        )

    def test_the_walk_order_is_stable_across_clone_model(self):
        """MEASURED, not assumed: `create_smoothed_model` indexes a clone by position.

        This is a PIN, not RED evidence — it passed before the fix too. It exists
        because `model_analyzer.py:1004-1005` does `all_layers[layer_id]` on a model
        produced by `keras.models.clone_model`, so a walk whose order differed
        between original and clone would rewrite the wrong layer's weights.
        """
        from dl_techniques.analyzer.utils import recursively_get_layers

        model = _build_property_probe_model()
        clone = keras.models.clone_model(model)

        original = [l.name for l in recursively_get_layers(model)
                    if l.name in _DECLARATION_ORDER]
        cloned = [l.name for l in recursively_get_layers(clone)
                  if l.name in _DECLARATION_ORDER]

        assert original, "the probe produced no walkable sublayers"
        assert cloned == original, (
            f"clone walk order {cloned} differs from original {original}; "
            f"create_smoothed_model would write the wrong layer's weights"
        )
