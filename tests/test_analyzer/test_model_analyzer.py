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
