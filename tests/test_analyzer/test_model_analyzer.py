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
