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

# The rcParams `restore_plotting_style` deliberately does NOT put back. Shared by
# every rcParams guard in this module so the exclusion cannot drift between them.
#
# `setup_plotting_style` calls `matplotlib.use('Agg')`, which is a repo-wide
# HEADLESS requirement (pinned by
# `tests/test_callbacks/test_the_matplotlib_backend_is_headless.py`), not part of
# this configuration's styling: restoring an interactive backend such as `tkagg`
# inside a test session would re-introduce the X11 crash that pin exists to
# prevent. `matplotlib.use` sets `backend`, and additionally clears
# `backend_fallback` and `interactive`, so all three are excluded. Asserting on
# them made `test_the_saved_rcparams_can_be_restored` ORDER-DEPENDENT: it passed
# in the full suite only because an earlier test had already forced `Agg`, and
# failed standalone with
# `{'backend': ('tkagg', 'Agg'), 'backend_fallback': (True, False)}`.
_RCPARAMS_NOT_RESTORED = {"backend", "backend_fallback", "interactive"}


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

    def test_no_keras_owned_property_is_evaluated(self):
        """The P6 contract, narrowed by D-038.

        `dir()` made the walk evaluate all 52 public names of a subclassed Layer,
        including Keras' own `weights` / `variables` / `input` / `output`. That is
        still forbidden. USER-declared properties ARE now evaluated, because a
        sublayer held behind one is otherwise invisible (D-038) — so this test
        checks the exclusion rule directly rather than counting all hits.
        """
        from dl_techniques.analyzer.utils import (
            recursively_get_layers, _user_property_names)

        model = _build_property_probe_model()
        block = [l for l in model.layers if l.name == "p6_blk"][0]
        user_properties = _user_property_names(type(block))

        for keras_property in ("weights", "variables", "trainable_weights",
                               "non_trainable_weights", "losses", "input",
                               "output", "metrics", "dtype"):
            assert keras_property not in user_properties, (
                f"the walk would evaluate Keras' own '{keras_property}'")

        # Anti-vacuity: the rule is a filter, not a blanket refusal.
        assert "expensive" in user_properties

        # ...and the walk itself completes on a built model without raising.
        recursively_get_layers(model)

    def test_a_keras_property_override_is_still_skipped(self):
        """The exclusion is by NAME, discovered from the Keras bases.

        A subclass that overrides `weights` is still Keras' API surface and
        still expensive, so it must not be evaluated either.
        """
        from dl_techniques.analyzer.utils import _user_property_names

        class OverridingBlock(keras.layers.Layer):
            @property
            def weights(self):  # pragma: no cover - must never be called
                raise AssertionError("the walk evaluated an overridden `weights`")

            @property
            def blocks(self):
                return []

        names = _user_property_names(OverridingBlock)
        assert "weights" not in names
        assert "blocks" in names

    def test_a_property_that_raises_does_not_break_the_walk(self):
        """`exploding` IS evaluated now; the walk must survive it."""
        from dl_techniques.analyzer.utils import recursively_get_layers

        model = _build_property_probe_model()
        _PROPERTY_HITS.clear()
        names = [l.name for l in recursively_get_layers(model)]

        assert "exploding" in _PROPERTY_HITS, (
            "the probe's raising property was never reached; this test is vacuous")
        assert set(_DECLARATION_ORDER).issubset(set(names))

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


# =====================================================================
# A6 + C10 - the config surface must be live and must match the README
# =====================================================================

DEAD_CONFIG_FIELDS = (
    "n_samples_per_digit",
    "sample_digits",
    "show_statistical_tests",
    "show_confidence_intervals",
    "catch_specific_exceptions",
    "enable_parallel_analysis",
    "activation_layer_name",
    "activation_layer_index",
)


def _config_field_names() -> set:
    """Public (non-underscore) field names declared on ``AnalysisConfig``."""
    import dataclasses

    return {
        f.name for f in dataclasses.fields(AnalysisConfig)
        if not f.name.startswith("_")
    }


def _readme_field_names() -> set:
    """Field names named in the ``AnalysisConfig`` table of the package README."""
    import re
    from pathlib import Path

    import dl_techniques.analyzer as analyzer_pkg

    readme = Path(analyzer_pkg.__file__).parent / "README.md"
    lines = readme.read_text().splitlines()

    start = lines.index("## `AnalysisConfig`")
    names = set()
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        if not line.startswith("|") or line.startswith("|---"):
            continue
        first_cell = line.split("|")[1]
        names.update(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", first_cell))
    names.discard("Field")
    return names


class TestTheConfigSurfaceIsLive:
    """Every `AnalysisConfig` field must have a reader, and the README must agree.

    Defect guarded (A6 + C10): eight fields had ZERO references anywhere outside
    `config.py` (re-grepped over the whole repo in-step). `catch_specific_exceptions`
    is the dangerous one - it reads as a live error-handling switch and controls
    nothing. Separately `json_include_raw_esds` was serialized INTO the artifact and
    documented in the README as controlling raw eigenvalue arrays, while
    `save_results` never mentioned `spectral_esds` at all.
    """

    @pytest.mark.parametrize("name", DEAD_CONFIG_FIELDS)
    def test_a_dead_field_is_not_advertised(self, name):
        assert name not in _config_field_names(), (
            f"AnalysisConfig still declares {name!r}, which has no reader anywhere "
            "in the repository"
        )

    def test_the_readme_table_and_the_config_agree(self):
        config_names = _config_field_names()
        readme_names = _readme_field_names()

        # Anti-vacuity: the parser really did find the table.
        assert len(readme_names) > 20, (
            f"the README AnalysisConfig table parsed to {len(readme_names)} names; "
            "the parser, not the docs, is what failed"
        )
        assert readme_names - config_names == set(), (
            f"README documents fields AnalysisConfig does not have: "
            f"{sorted(readme_names - config_names)}"
        )
        assert config_names - readme_names == set(), (
            f"AnalysisConfig declares fields the README does not document: "
            f"{sorted(config_names - readme_names)}"
        )


class TestJsonIncludeRawEsdsIsHonoured:
    """`json_include_raw_esds` was serialized but never read.

    MEASURED on unfixed HEAD: with the flag True the artifact carried
    `config.json_include_raw_esds == True` and NO `spectral_esds` key, so the output
    advertised a knob that did nothing while `README.md` documented it as controlling
    raw eigenvalue arrays. The data was already on `AnalysisResults.spectral_esds`.
    """

    def _artifact(self, tmp_path, include: bool):
        models = {"good": _build_spectral_probe("good", zeroed=False)}
        analyzer = ModelAnalyzer(
            models=models,
            config=_quiet_config(
                analyze_spectral=True, json_include_raw_esds=include),
            output_dir=str(tmp_path / f"esd_{include}"),
        )
        analyzer.analyze(analysis_types={"spectral"})

        # Anti-vacuity: there IS a spectrum to include, so an empty artifact key
        # cannot pass by the analysis having produced nothing.
        assert analyzer.results.spectral_esds, (
            "the spectral analysis produced no ESDs; this guard would be vacuous"
        )
        return json.loads(
            (analyzer.output_dir / "analysis_results.json").read_text())

    def test_the_raw_esds_reach_the_artifact_when_the_flag_is_on(self, tmp_path):
        payload = self._artifact(tmp_path, include=True)

        assert "spectral_esds" in payload, (
            "json_include_raw_esds=True but the artifact has no 'spectral_esds' "
            f"key; top-level keys: {sorted(payload)}"
        )
        esds = payload["spectral_esds"]
        assert esds and any(esds.values()), f"'spectral_esds' is empty: {esds}"
        first_layer = next(iter(next(iter(esds.values())).values()))
        assert isinstance(first_layer, list) and len(first_layer) > 0

    def test_the_flag_still_excludes_them_when_off(self, tmp_path):
        """Anti-vacuity: the key must be ABSENT when the flag is off."""
        payload = self._artifact(tmp_path, include=False)
        assert "spectral_esds" not in payload
        assert "spectral_rand_esds" not in payload


# =====================================================================
# A7 - the config's private matplotlib state must be declared, not filtered
# =====================================================================

class TestTheConfigsPrivateStateIsDeclared:
    """`_original_rcParams` was an undeclared attribute filtered by string name.

    `setup_plotting_style` assigned `self._original_rcParams = plt.rcParams.copy()`
    on a dataclass that never declared it, so it was absent from `fields()` and
    `asdict()`, nothing ever read it back, and `save_results` had to exclude it
    with a literal `if k != '_original_rcParams'` over `self.config.__dict__`. That
    filter is name-specific: ANY other private attribute reaches the artifact.

    `matplotlib.use('Agg')` at `config.py` is deliberately NOT in scope - it is a
    repo-wide headless requirement pinned by
    `tests/test_callbacks/test_the_matplotlib_backend_is_headless.py`.
    """

    def test_the_saved_rcparams_are_a_declared_field(self):
        import dataclasses

        names = {f.name for f in dataclasses.fields(AnalysisConfig)}
        assert "_original_rcParams" in names, (
            "_original_rcParams is assigned by setup_plotting_style but is not a "
            f"declared field; fields() reports {sorted(names)}"
        )

    def test_the_saved_rcparams_can_be_restored(self):
        import matplotlib.pyplot as plt

        config = _quiet_config(plot_style="presentation", dpi=137)

        # Force a known NON-analyzer state first: `setup_plotting_style` is
        # idempotent, so if a previous test left the style applied the
        # anti-vacuity check below would be the thing that fails.
        plt.rcParams["axes.grid"] = False
        plt.rcParams["savefig.dpi"] = 71

        before = dict(plt.rcParams)

        config.setup_plotting_style()
        moved = [
            k for k, v in before.items()
            if k not in _RCPARAMS_NOT_RESTORED and plt.rcParams[k] != v
        ]
        # Anti-vacuity: the style really did mutate process-global state, so the
        # restore below has something to undo. `font.size` is NOT usable here -
        # `sns.set_theme` resets it to the seaborn default at the end of setup.
        assert moved, "setup_plotting_style changed no restorable rcParam at all"

        config.restore_plotting_style()
        still_wrong = {
            k: (before[k], plt.rcParams[k]) for k in moved
            if plt.rcParams[k] != before[k]
        }
        assert not still_wrong, (
            f"restore_plotting_style left {len(still_wrong)} rcParams changed: "
            f"{dict(list(still_wrong.items())[:5])}"
        )

    def test_the_backend_is_deliberately_not_restored(self):
        """The exclusion above is a CONTRACT, not a convenience.

        The mechanism is the ORDER inside ``setup_plotting_style``: it calls
        ``matplotlib.use('Agg')`` BEFORE snapshotting ``plt.rcParams``, so the
        snapshot already carries ``Agg`` and the restore cannot put an
        interactive backend back. Reordering those two lines - the obvious
        "snapshot the true pre-state" edit - would silently un-headless the
        session, so it is pinned here.
        """
        import matplotlib.pyplot as plt

        # A pre-existing interactive backend, set WITHOUT switching to it (the
        # value is what `restore` would put back; no GUI toolkit is loaded).
        plt.rcParams["backend"] = "tkagg"
        assert plt.rcParams["backend"] == "tkagg", (
            "anti-vacuity: the pre-state is not interactive, so 'Agg' afterwards "
            "would prove nothing"
        )

        config = _quiet_config()
        config.setup_plotting_style()
        config.restore_plotting_style()

        assert plt.rcParams["backend"] == "Agg", (
            "restore_plotting_style put the interactive backend back; the headless "
            f"requirement is broken (backend={plt.rcParams['backend']!r})"
        )

    def test_an_arbitrary_private_attribute_does_not_reach_the_artifact(
            self, tmp_path, probe_data):
        """The string-name filter is what this proves wrong."""
        config = _quiet_config()
        analyzer = ModelAnalyzer(
            models={"m": _build_classifier("m", seed=1)},
            config=config,
            output_dir=str(tmp_path / "a7"),
        )
        analyzer.config._probe_private_state = {"leaked": True}
        analyzer.analyze(data=probe_data, analysis_types=set())

        payload = json.loads(
            (analyzer.output_dir / "analysis_results.json").read_text())
        private = [k for k in payload["config"] if k.startswith("_")]
        assert not private, (
            f"the serialized config carries private keys: {private}"
        )

    def test_the_serialized_config_is_exactly_the_declared_public_fields(
            self, tmp_path, probe_data):
        """PIN: the artifact's config block must track `fields()`, not `__dict__`."""
        analyzer = ModelAnalyzer(
            models={"m": _build_classifier("m", seed=2)},
            config=_quiet_config(),
            output_dir=str(tmp_path / "a7b"),
        )
        analyzer.analyze(data=probe_data, analysis_types=set())

        payload = json.loads(
            (analyzer.output_dir / "analysis_results.json").read_text())
        assert set(payload["config"]) == _config_field_names()


# =====================================================================
# S11 - a sigmoid multi-label head must not be softmaxed
# =====================================================================

def _build_sigmoid_multilabel(name: str) -> keras.Model:
    """A multi-label head: values in [0, 1] whose rows do NOT sum to 1."""
    keras.utils.set_random_seed(11)
    inputs = keras.Input(shape=(N_FEATURES,), name=f"{name}_in")
    x = keras.layers.Dense(8, activation="relu", name=f"{name}_d1")(inputs)
    outputs = keras.layers.Dense(
        N_CLASSES, activation="sigmoid", name=f"{name}_out")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name=name)


class TestASigmoidHeadIsNotSoftmaxed:
    """`is_logits` flagged a sigmoid multi-label head as logits and softmaxed it.

    `model_analyzer.py:472-483` computed
    `is_logits = np.any(predictions < 0) or not np.allclose(row_sums, 1.0)`. A
    sigmoid head emits values in [0, 1] whose rows do not sum to 1, so the second
    clause is True, `keras.ops.softmax` was applied to ALREADY-normalized
    probabilities, and every downstream ECE / Brier number was computed on the
    wrong array. Nothing was logged on that branch.
    """

    def _cached(self, tmp_path, probe_data, tag, **config_overrides):
        model = _build_sigmoid_multilabel("ml")
        analyzer = ModelAnalyzer(
            models={"ml": model},
            config=_quiet_config(**config_overrides),
            output_dir=str(tmp_path / tag),
        )
        raw = model.predict(probe_data.x_data, verbose=0)
        analyzer._cache_predictions(probe_data)
        cached = analyzer._prediction_cache["ml"]["predictions"]
        return raw, cached, analyzer

    def test_the_probe_really_is_a_sigmoid_head(self, tmp_path, probe_data):
        """Anti-vacuity: the probe must trip the old heuristic's second clause."""
        raw, _, _ = self._cached(tmp_path, probe_data, "s0")
        assert raw.min() >= 0.0 and raw.max() <= 1.0
        row_sums = raw.sum(axis=-1)
        assert not np.allclose(row_sums, 1.0, atol=1e-3), (
            f"the probe's rows sum to ~1 ({row_sums[:3]}); it would not have been "
            "misclassified either way"
        )

    def test_the_probabilities_survive_the_heuristic(self, tmp_path, probe_data):
        raw, cached, _ = self._cached(tmp_path, probe_data, "s1")
        np.testing.assert_allclose(cached, raw, rtol=0, atol=0)

    def test_the_inference_is_logged(self, tmp_path, probe_data, caplog):
        import logging

        with caplog.at_level(logging.INFO):
            self._cached(tmp_path, probe_data, "s2")
        assert any("sigmoid" in record.message.lower()
                   for record in caplog.records), (
            "no INFO line reports the inferred output activation; the branch was "
            f"silent. messages: {[r.message for r in caplog.records][:10]}"
        )

    def test_an_explicit_output_activation_is_obeyed(self, tmp_path, probe_data):
        """`output_activation='logits'` must override the heuristic."""
        raw, cached, _ = self._cached(
            tmp_path, probe_data, "s3", output_activation="logits")
        expected = np.array(keras.ops.softmax(raw, axis=-1))
        np.testing.assert_allclose(cached, expected, rtol=1e-6, atol=1e-7)
        assert not np.allclose(cached, raw), (
            "output_activation='logits' left the predictions untouched"
        )

    def test_a_real_logit_head_is_still_softmaxed(self, tmp_path, probe_data):
        """Anti-vacuity: the fix must not disable logit detection."""
        keras.utils.set_random_seed(12)
        inputs = keras.Input(shape=(N_FEATURES,), name="lg_in")
        outputs = keras.layers.Dense(N_CLASSES, name="lg_out")(inputs)
        model = keras.Model(inputs, outputs, name="lg")

        analyzer = ModelAnalyzer(
            models={"lg": model},
            config=_quiet_config(),
            output_dir=str(tmp_path / "s4"),
        )
        raw = model.predict(probe_data.x_data, verbose=0)
        assert raw.min() < 0.0, "the probe emitted no negative value; not logits"

        analyzer._cache_predictions(probe_data)
        cached = analyzer._prediction_cache["lg"]["predictions"]
        expected = np.array(keras.ops.softmax(raw, axis=-1))
        np.testing.assert_allclose(cached, expected, rtol=1e-6, atol=1e-7)

    def test_a_softmax_head_is_still_left_alone(self, tmp_path, probe_data):
        """Anti-vacuity: the ordinary single-label path is unchanged."""
        model = _build_classifier("sm", seed=13)
        analyzer = ModelAnalyzer(
            models={"sm": model},
            config=_quiet_config(),
            output_dir=str(tmp_path / "s5"),
        )
        raw = model.predict(probe_data.x_data, verbose=0)
        analyzer._cache_predictions(probe_data)
        np.testing.assert_allclose(
            analyzer._prediction_cache["sm"]["predictions"], raw, rtol=0, atol=0)


# ---------------------------------------------------------------------
# C-2 (review iteration 1) -- reported accuracy was 0.0 at status='success'
# ---------------------------------------------------------------------

def _build_compiled_classifier(name: str, seed: int, metrics=("accuracy",)) -> keras.Model:
    """A softmax classifier compiled the ordinary way.

    Args:
        name: Model name; also prefixes the layer names.
        seed: Seed for deterministic weights.
        metrics: Passed straight to ``compile(metrics=...)``. Pass ``()`` for a
            model with no metric at all.

    Returns:
        keras.Model: A compiled, deterministic classifier.
    """
    model = _build_classifier(name, seed=seed)
    model.compile(loss="sparse_categorical_crossentropy", metrics=list(metrics))
    return model


def _performance(tmp_path, probe_data, models: dict) -> dict:
    """Run the analyzer over ``models`` and return its performance summary."""
    analyzer = ModelAnalyzer(
        models=models,
        config=_quiet_config(analyze_calibration=True),
        output_dir=str(tmp_path),
    )
    analyzer.analyze(probe_data, analysis_types={"calibration"})
    return analyzer.get_summary_statistics()["model_performance"], analyzer


class TestReportedAccuracyIsTheRealAccuracy:
    """`summary['model_performance'][m]['accuracy']` was 0.0 at status='success'.

    MEASURED against the shipped code on two `Dense(3, softmax)` models compiled
    with `metrics=["accuracy"]`: `results.model_metrics` carried
    `{'status':'success','loss':1.6421177,'compile_metrics':0.2750000}` while the
    summary reported `{'accuracy': 0.0, ...}`. The alias was a substring match of
    `'acc'` against `model.metrics_names`, and Keras 3 names the aggregated
    compiled metric `compile_metrics` — which contains no `'acc'`.
    """

    def test_keras_3_really_does_hide_the_metric_name(self):
        """Anti-vacuity: pin the Keras behaviour the fix exists for.

        If a future Keras restores per-metric names in `metrics_names`, this
        reddens and the resolution logic should be re-read, not silently kept.
        """
        model = _build_compiled_classifier("kn", seed=31)
        x = np.zeros((4, N_FEATURES), dtype="float32")
        model.evaluate(x, np.zeros(4, dtype="int64"), verbose=0)
        assert model.metrics_names == ["loss", "compile_metrics"]
        assert not any("acc" in n for n in model.metrics_names)
        # ...while the name the caller compiled with survives here:
        assert "accuracy" in model.get_metrics_result()

    def test_a_compiled_model_reports_its_real_accuracy(self, tmp_path, probe_data):
        models = {"a": _build_compiled_classifier("a", seed=41)}
        performance, analyzer = _performance(tmp_path, probe_data, models)

        raw = analyzer.results.model_metrics["a"]
        assert raw["status"] == "success"
        truth = float(raw["compile_metrics"])

        assert performance["a"]["accuracy"] is not None
        assert performance["a"]["accuracy"] == pytest.approx(truth, rel=1e-12), (
            f"summary accuracy {performance['a']['accuracy']!r} against the "
            f"model's own compiled metric {truth!r}"
        )
        # Anti-vacuity: a model that is merely reported as 0.0 would pass an
        # `is not None` check, so pin that the value is a live, non-sentinel one.
        assert truth > 0.0, "the probe model scored exactly 0.0; pick another seed"
        assert performance["a"]["accuracy"] != 0.0

    def test_accuracy_is_none_when_no_accuracy_metric_was_compiled(
            self, tmp_path, probe_data):
        """No accuracy metric must read as UNKNOWN, never as 'scored zero'."""
        models = {"n": _build_compiled_classifier("n", seed=42, metrics=())}
        performance, analyzer = _performance(tmp_path, probe_data, models)

        assert analyzer.results.model_metrics["n"]["status"] == "success"
        assert performance["n"]["accuracy"] is None
        assert performance["n"]["loss"] > 0.0

    def test_accuracy_is_resolved_by_metric_class_not_by_its_name(
            self, tmp_path, probe_data):
        """A user-renamed accuracy metric is still found.

        Resolution is by metric CLASS first, so `name='hit_rate'` — which
        matches no `ACC_PATTERNS` entry — is still reported as the accuracy.
        """
        model = _build_classifier("r", seed=43)
        model.compile(
            loss="sparse_categorical_crossentropy",
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="hit_rate")],
        )
        performance, analyzer = _performance(tmp_path, probe_data, {"r": model})

        raw = analyzer.results.model_metrics["r"]
        assert "hit_rate" in raw, f"the renamed metric never landed: {raw!r}"
        assert performance["r"]["accuracy"] == pytest.approx(
            float(raw["hit_rate"]), rel=1e-12)

    def test_a_failed_evaluation_reports_none_rather_than_zero(
            self, tmp_path, probe_data):
        """An UNCOMPILED model cannot be evaluated; that is not 'accuracy 0.0'."""
        models = {"u": _build_classifier("u", seed=44)}
        performance, analyzer = _performance(tmp_path, probe_data, models)

        raw = analyzer.results.model_metrics["u"]
        assert raw["status"] in ("evaluation_failed", "error"), raw
        assert raw["accuracy"] is None
        assert performance["u"]["accuracy"] is None

    def test_two_models_are_ranked_by_accuracy_not_flattened_to_zero(
            self, tmp_path, probe_data):
        """The cross-model comparison the summary exists for.

        Two DIFFERENT models must produce two distinguishable accuracies; under
        the shipped defect both read 0.0 and every ranking was a tie.
        """
        models = {
            "m1": _build_compiled_classifier("m1", seed=51),
            "m2": _build_compiled_classifier("m2", seed=52),
        }
        performance, analyzer = _performance(tmp_path, probe_data, models)
        values = [performance[m]["accuracy"] for m in ("m1", "m2")]
        assert all(v is not None for v in values), performance
        for model_name, value in zip(("m1", "m2"), values):
            assert value == pytest.approx(
                float(analyzer.results.model_metrics[model_name]["compile_metrics"]),
                rel=1e-12)


# ---------------------------------------------------------------------
# W-4 (review iteration 1) -- sublayers behind a @property
# ---------------------------------------------------------------------

class PropertyExposedBlock(keras.layers.Layer):
    """Holds its sublayers privately and exposes them through a property.

    A common Keras idiom, and invisible to a `vars()`-only walk: the attribute
    that holds them is `_inner`, which the walk skips as private, and `blocks`
    lives on the CLASS, not in the instance `__dict__`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._inner = [
            keras.layers.Dense(4, name="prop_d1"),
            keras.layers.Dense(4, name="prop_d2"),
        ]

    @property
    def blocks(self):
        return self._inner

    def call(self, inputs):
        x = inputs
        for block in self._inner:
            x = block(x)
        return x


class AttributeExposedBlock(keras.layers.Layer):
    """The control: the same two sublayers on a PUBLIC attribute."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inner = [
            keras.layers.Dense(4, name="attr_d1"),
            keras.layers.Dense(4, name="attr_d2"),
        ]

    def call(self, inputs):
        x = inputs
        for block in self.inner:
            x = block(x)
        return x


def _walk_names(block_cls) -> list:
    from dl_techniques.analyzer.utils import recursively_get_layers

    inputs = keras.Input(shape=(3,), name="w4_in")
    block = block_cls(name="w4_blk")
    model = keras.Model(inputs, block(inputs), name="w4_model")
    return [layer.name for layer in recursively_get_layers(model)]


class TestSublayersBehindAPropertyAreFound:
    """`vars()` alone dropped them, silently shrinking analysis coverage.

    MEASURED against the shipped `vars()` walk: a block holding
    `self._inner = [Dense, Dense]` behind `@property def blocks` walked to
    `['input_layer', 'blk']` — the two Dense layers were absent from weight,
    spectral AND information-flow analysis, with no warning.
    """

    def test_property_exposed_sublayers_are_walked(self):
        names = _walk_names(PropertyExposedBlock)
        assert "prop_d1" in names and "prop_d2" in names, (
            f"the walk returned {names}; the two property-exposed Dense layers "
            f"are missing"
        )

    def test_the_attribute_control_is_unchanged(self):
        """Anti-vacuity: the public-attribute route must still work identically."""
        names = _walk_names(AttributeExposedBlock)
        assert "attr_d1" in names and "attr_d2" in names

    def test_the_two_routes_agree(self):
        """The whole point: how a sublayer is EXPOSED must not change coverage."""
        prop = [n for n in _walk_names(PropertyExposedBlock) if n.startswith("prop_")]
        attr = [n for n in _walk_names(AttributeExposedBlock) if n.startswith("attr_")]
        assert len(prop) == len(attr) == 2


# ---------------------------------------------------------------------
# W-6 (review iteration 1) -- restore_plotting_style had no production caller
# ---------------------------------------------------------------------

class TestTheAnalyzerCanUndoItsGlobalStyleMutation:
    """`restore_plotting_style` existed with NO caller outside tests.

    D-029 introduced it as the answer to "`setup_plotting_style` mutates
    process-global matplotlib state, which outlives the `ModelAnalyzer`", but
    `grep -rn restore_plotting_style src/ tests/` returned only its definition
    and one test, so the leak was unsolved on every shipped path.
    """

    def test_the_context_manager_restores_the_rcparams(self, tmp_path):
        import matplotlib.pyplot as plt

        # The backend keys are deliberately NOT restored.
        skip = _RCPARAMS_NOT_RESTORED

        # Force a known NON-analyzer state first: `setup_plotting_style` is
        # idempotent, so if a previous test left the style applied this guard
        # would see nothing move and its anti-vacuity check would be the thing
        # that fails.
        plt.rcParams["axes.grid"] = False
        plt.rcParams["savefig.dpi"] = 71

        before = plt.rcParams.copy()
        with ModelAnalyzer(
                models={"style": _build_classifier("style", seed=61)},
                config=_quiet_config(),
                output_dir=str(tmp_path / "style"),
        ) as analyzer:
            inside = plt.rcParams.copy()
            moved = [k for k in before if k not in skip and inside[k] != before[k]]
            # Anti-vacuity: the analyzer really did mutate the global state.
            assert moved, "constructing the analyzer changed no rcParam at all"
            assert analyzer is not None

        after = plt.rcParams.copy()
        still_wrong = [k for k in before if k not in skip and after[k] != before[k]]
        assert not still_wrong, (
            f"leaving the context left {len(still_wrong)} rcParams changed: "
            f"{still_wrong[:8]}"
        )

    def test_the_backend_is_deliberately_not_restored(self, tmp_path):
        """`Agg` is a repo-wide headless requirement, not part of the styling."""
        import matplotlib

        with ModelAnalyzer(
                models={"bk": _build_classifier("bk", seed=62)},
                config=_quiet_config(),
                output_dir=str(tmp_path / "bk"),
        ):
            pass
        assert matplotlib.get_backend().lower() == "agg"

    def test_the_context_manager_does_not_swallow_exceptions(self, tmp_path):
        with pytest.raises(RuntimeError, match="propagate me"):
            with ModelAnalyzer(
                    models={"ex": _build_classifier("ex", seed=63)},
                    config=_quiet_config(),
                    output_dir=str(tmp_path / "ex"),
            ):
                raise RuntimeError("propagate me")

    def test_without_the_context_manager_the_style_still_leaks(self, tmp_path):
        """PIN, not RED evidence: the leak outside `with` is DELIBERATE.

        Scoping the style with `rc_context` would strip `savefig.dpi` from the
        `Figure` `create_pareto_analysis` returns to the caller (D-029/D-039).
        This test records that choice so a future "fix" of it is a decision, not
        an accident.
        """
        import matplotlib.pyplot as plt

        before = plt.rcParams.copy()
        try:
            # Force a known NON-analyzer state so this guard does not depend on
            # what the previous test left in the process-global rcParams.
            plt.rcParams["axes.grid"] = False
            plt.rcParams["savefig.dpi"] = 71
            ModelAnalyzer(
                models={"lk": _build_classifier("lk", seed=64)},
                config=_quiet_config(),
                output_dir=str(tmp_path / "lk"),
            )
            assert plt.rcParams["axes.grid"] is True, (
                "the style no longer leaks without `with`; D-039 needs updating")
            assert plt.rcParams["savefig.dpi"] != 71, (
                "the style no longer leaks without `with`; D-039 needs updating")
        finally:
            plt.rcParams.update(before)


# ---------------------------------------------------------------------
# W-7 (review iteration 1) -- artifacts must be self-identifying
# ---------------------------------------------------------------------

class TestSavedResultsCarryASchemaStamp:
    """`per_class_ece` changed MEANING while keeping its key name.

    It went from the masked top-1 ECE (now published as
    `per_class_conditional_top1_ece`) to Kull classwise ECE (D-015) — measured
    moving `per_class_ece[2]` from `0.0` to `0.30000` on a class the model never
    predicts. A stored `analysis_results.json` from before that change compares
    silently against a different quantity unless the artifact says which version
    wrote it.
    """

    def _saved(self, tmp_path, probe_data) -> dict:
        analyzer = ModelAnalyzer(
            models={"sv": _build_compiled_classifier("sv", seed=71)},
            config=_quiet_config(analyze_calibration=True),
            output_dir=str(tmp_path),
        )
        analyzer.analyze(probe_data, analysis_types={"calibration"})
        analyzer.save_results("stamped.json")
        with open(tmp_path / "stamped.json") as fh:
            return json.load(fh)

    def test_the_artifact_names_its_schema_and_the_analyzer_version(
            self, tmp_path, probe_data):
        from dl_techniques.analyzer import __version__ as package_version
        from dl_techniques.analyzer.model_analyzer import RESULTS_SCHEMA_VERSION

        saved = self._saved(tmp_path, probe_data)
        assert saved["schema_version"] == RESULTS_SCHEMA_VERSION
        assert saved["analyzer_version"] == package_version
        assert saved["analyzer_version"] != "unknown"

    def test_the_stamp_covers_the_key_that_changed_meaning(
            self, tmp_path, probe_data):
        """Anti-vacuity: the artifact really does carry the redefined key.

        A schema stamp on an artifact that does not contain `per_class_ece`
        would identify nothing.
        """
        saved = self._saved(tmp_path, probe_data)
        calibration = saved["calibration_metrics"]["sv"]
        assert "per_class_ece" in calibration
        # ...and the legacy quantity ships beside it under an honest name, so a
        # reader of an old artifact can reconstruct what they used to have.
        assert "per_class_conditional_top1_ece" in calibration
        assert len(calibration["per_class_ece"]) == N_CLASSES


# ---------------------------------------------------------------------
# A failed evaluation must not report `loss: 0.0` -- a perfect model
# ---------------------------------------------------------------------

class TestReportedLossIsNotASentinel:
    """`loss = 0.0` for a model that was never evaluated reads as PERFECT.

    This is the same sentinel class D-036 removed for `accuracy`, left in place
    for `loss`: `model_analyzer.py` wrote `DEFAULT_METRIC_VALUE` (0.0) on both
    failure branches. MEASURED at HEAD with a model whose compiled metric
    promotes a Python `str` (`keras.ops.divide(y_pred, "mean")`, which raises
    `dtype='string' is not a valid dtype for Keras type promotion`):
    `{'loss': 0.0, 'accuracy': None, 'status': 'evaluation_failed'}` -- so the
    same record reported the accuracy honestly and the loss as a flawless score.
    """

    @staticmethod
    def _string_promoting_model(name: str) -> keras.Model:
        """A model whose METRIC (not its data) reaches a promoting op with a str.

        Data-borne string routes are all pre-empted by Keras 3.8 with a different
        error, because the array adapter standardizes dtypes and `Loss.__call__`
        casts before any metric op runs. The string can only arrive from
        model-side code that runs under `evaluate` and not under `predict`.
        """

        class _StrPromoMetric(keras.metrics.Metric):
            def __init__(self, **kwargs):
                super().__init__(name="str_promo", **kwargs)
                self.mode = "mean"  # a bare Python str
                self.total = self.add_weight(name="total", initializer="zeros")

            def update_state(self, y_true, y_pred, sample_weight=None):
                self.total.assign_add(
                    keras.ops.mean(keras.ops.divide(y_pred, self.mode)))

            def result(self):
                return self.total

        model = _build_classifier(name, seed=61)
        model.compile(loss="sparse_categorical_crossentropy",
                      metrics=[_StrPromoMetric()])
        return model

    def test_the_probe_really_hits_the_string_promotion_branch(
            self, tmp_path, probe_data):
        """Anti-vacuity: pin the mechanism, and that `predict` still succeeds."""
        models = {"s": self._string_promoting_model("s")}
        _, analyzer = _performance(tmp_path, probe_data, models)

        raw = analyzer.results.model_metrics["s"]
        assert raw["status"] == "evaluation_failed", raw
        assert "dtype='string'" in raw["error"], raw["error"]
        # The diagnostic signature: predict works, evaluate does not.
        assert analyzer._prediction_cache["s"]["predictions"] is not None

    def test_a_failed_evaluation_reports_loss_none_not_zero(
            self, tmp_path, probe_data):
        models = {"s": self._string_promoting_model("s")}
        performance, analyzer = _performance(tmp_path, probe_data, models)

        raw = analyzer.results.model_metrics["s"]
        assert raw["loss"] is None, (
            f"loss reported as {raw['loss']!r} for a model that never evaluated; "
            "0.0 is a perfect score, not an absence"
        )
        assert performance["s"]["loss"] is None, performance["s"]

    def test_an_uncompiled_model_also_reports_loss_none(self, tmp_path, probe_data):
        """The OUTER failure branch, and the invariant that `analyze()` survives."""
        models = {"u": _build_classifier("u2", seed=62),
                  "ok": _build_compiled_classifier("ok", seed=63)}
        performance, analyzer = _performance(tmp_path, probe_data, models)

        assert analyzer.results.model_metrics["u"]["loss"] is None
        assert performance["u"]["loss"] is None
        # Invariant: one model failing must not cost the other its results.
        assert performance["ok"]["loss"] is not None
        assert performance["ok"]["loss"] > 0.0

    def test_a_successful_model_still_reports_a_real_loss(
            self, tmp_path, probe_data):
        """Anti-vacuity: the fix must not turn every loss into None."""
        models = {"g": _build_compiled_classifier("g", seed=64)}
        performance, analyzer = _performance(tmp_path, probe_data, models)
        assert analyzer.results.model_metrics["g"]["status"] == "success"
        assert performance["g"]["loss"] > 0.0

    def test_the_schema_version_was_bumped_for_the_loss_change(self):
        """A published VALUE changed meaning, so the stamp must move with it."""
        from dl_techniques.analyzer.model_analyzer import RESULTS_SCHEMA_VERSION

        assert RESULTS_SCHEMA_VERSION >= 4, (
            "`loss` changed from 0.0 to null for an unevaluated model — an "
            "artifact written before and after this compares silently"
        )
