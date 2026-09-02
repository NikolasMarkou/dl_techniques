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
        before = dict(plt.rcParams)

        config.setup_plotting_style()
        moved = [k for k, v in before.items() if plt.rcParams[k] != v]
        # Anti-vacuity: the style really did mutate process-global state, so the
        # restore below has something to undo. `font.size` is NOT usable here -
        # `sns.set_theme` resets it to the seaborn default at the end of setup.
        assert moved, "setup_plotting_style changed no rcParam at all"

        config.restore_plotting_style()
        still_wrong = {
            k: (before[k], plt.rcParams[k]) for k in moved
            if plt.rcParams[k] != before[k]
        }
        assert not still_wrong, (
            f"restore_plotting_style left {len(still_wrong)} rcParams changed: "
            f"{dict(list(still_wrong.items())[:5])}"
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
