"""Guards for `analyzers/training_dynamics_analyzer.py`.

S9 - the training-dynamics metrics are in RAW LOSS UNITS. The review's strongest
sub-claim is REFUTED and is deliberately not fixed here: `find_pareto_front` and
`normalize_metric` are monotone / min-max, so within a sweep sharing ONE loss
function the Pareto front and the ranking are unaffected. What IS confirmed is that
`overfitting_index` (`val_final - train_final`) and `training_stability_score`
(`std(val_loss[-10:])`) are not comparable across models with DIFFERENT loss
functions - exactly the multi-architecture comparison `ModelAnalyzer` advertises.
"""

import numpy as np
import pytest

from dl_techniques.analyzer.config import AnalysisConfig
from dl_techniques.analyzer.data_types import AnalysisResults
from dl_techniques.analyzer.analyzers.training_dynamics_analyzer import (
    TrainingDynamicsAnalyzer,
)


N_EPOCHS = 30


def _history(scale: float, gap_ratio: float = 0.2, noise: float = 0.05):
    """A history whose loss units are `scale` and whose RELATIVE gap is fixed.

    Args:
        scale: Multiplier on every loss value - the "different loss function".
        gap_ratio: `(val - train) / train` in the final third, scale-free.
        noise: Amplitude of the deterministic val-loss oscillation, relative to
            `scale`, so the coefficient of variation is scale-free too.

    Returns:
        A Keras-shaped history dict with `loss`, `val_loss` and `val_accuracy`.
    """
    epochs = np.arange(N_EPOCHS, dtype=float)
    train = scale * (1.0 + np.exp(-epochs / 6.0))
    wobble = noise * np.sin(epochs)
    val = train * (1.0 + gap_ratio) * (1.0 + wobble)
    acc = 0.5 + 0.4 * (1.0 - np.exp(-epochs / 5.0))
    return {
        "loss": train.tolist(),
        "val_loss": val.tolist(),
        "val_accuracy": acc.tolist(),
    }


def _run(histories):
    """Run the analyzer over `{model_name: history}` and return its metrics."""
    results = AnalysisResults()
    results.training_history = histories
    analyzer = TrainingDynamicsAnalyzer(
        models={name: None for name in histories},
        config=AnalysisConfig(smooth_training_curves=False, verbose=False),
    )
    analyzer.analyze(results)
    return results.training_metrics


class TestTheRelativeMetricsAreScaleFree:
    """Two models with identical dynamics in units 1x and 100x apart.

    MEASURED on unfixed HEAD: `overfitting_index` reads 0.21109788665355111 for the
    1x model and 21.10978866535511 for the 100x model - a 100x spread produced
    entirely by the loss units - and `training_stability_score` 0.04704031345071739
    vs 4.704031345071731. There is no relative counterpart on `TrainingMetrics`.
    """

    @pytest.fixture(scope="class")
    def metrics(self):
        return _run({"unit": _history(1.0), "hundred": _history(100.0)})

    def test_the_raw_overfitting_index_is_not_comparable(self, metrics):
        """Anti-vacuity: the raw metric really does disagree across loss scales."""
        raw = metrics.overfitting_index
        assert raw["hundred"] == pytest.approx(100.0 * raw["unit"], rel=1e-6), (
            f"the probe did not produce a 100x unit difference: {raw}"
        )
        assert abs(raw["hundred"] - raw["unit"]) > 1.0

    def test_the_relative_overfitting_index_agrees(self, metrics):
        rel = metrics.relative_overfitting_index
        assert rel["unit"] == pytest.approx(rel["hundred"], rel=1e-9), (
            f"relative_overfitting_index disagrees across loss scales: {rel}"
        )
        # It is the actual relative gap, not merely some scale-free number.
        assert rel["unit"] == pytest.approx(0.2, abs=0.02), rel

    def test_the_stability_cv_agrees(self, metrics):
        raw = metrics.training_stability_score
        cv = metrics.stability_cv

        # Anti-vacuity: the raw score is scale-dependent.
        assert raw["hundred"] == pytest.approx(100.0 * raw["unit"], rel=1e-6), raw
        assert cv["unit"] == pytest.approx(cv["hundred"], rel=1e-9), (
            f"stability_cv disagrees across loss scales: {cv}"
        )
        assert cv["unit"] > 0.0

    def test_the_raw_keys_are_still_published(self, metrics):
        """The fix is ADDITIVE: nothing that was published may disappear."""
        for name in ("unit", "hundred"):
            assert name in metrics.overfitting_index
            assert name in metrics.training_stability_score
            assert name in metrics.final_gap
            assert name in metrics.epochs_to_convergence


class TestTheRelativeMetricsDegradeSafely:
    """A zero or absent denominator must not produce inf/nan silently."""

    def test_a_zero_train_loss_yields_no_relative_index(self):
        history = _history(1.0)
        history["loss"] = [0.0] * N_EPOCHS
        metrics = _run({"zero": history})

        assert "zero" in metrics.overfitting_index
        assert "zero" not in metrics.relative_overfitting_index, (
            "a zero final train loss produced "
            f"{metrics.relative_overfitting_index.get('zero')}"
        )

    def test_a_run_too_short_for_the_stability_window_has_no_cv(self):
        short = {k: v[:3] for k, v in _history(1.0).items()}
        metrics = _run({"short": short})

        assert "short" not in metrics.training_stability_score
        assert "short" not in metrics.stability_cv


class TestTheParetoAxisCarriesTheUnitsCaveat:
    """The Pareto plot puts RAW `overfitting_index` on a shared axis.

    The ranking itself is unaffected within a single-loss sweep (`find_pareto_front`
    and `normalize_metric` are monotone / min-max), so the plot is not restructured -
    only labelled honestly.
    """

    def test_the_axis_label_names_the_units(self):
        import inspect
        import re

        from dl_techniques.analyzer.model_analyzer import ModelAnalyzer

        source = inspect.getsource(ModelAnalyzer.create_pareto_analysis)
        # Read the call's ARGUMENTS, not the source line: the label is a multi-line
        # implicit concatenation, so a line scan would miss it and a bare
        # `"loss units" in source` would match any comment nearby.
        calls = re.findall(r"set_xlabel\((.*?)\)", source, flags=re.DOTALL)
        assert calls, "create_pareto_analysis sets no x label"
        assert any("loss units" in call for call in calls), (
            f"the Pareto x axis does not name its units: {calls}"
        )


class TestEpochsToConvergenceDocumentsItsOwnWeakness:
    """`epochs_to_convergence` is relative to EACH model's own peak accuracy.

    Two models that converge at the same epoch to very different accuracies score
    identically, so the metric measures speed-to-own-plateau, not quality. The
    definition is deliberately NOT changed; the docstring must say so.
    """

    def test_a_worse_model_can_score_the_same(self):
        good = _history(1.0)
        bad = _history(1.0)
        bad["val_accuracy"] = [0.5 * a for a in good["val_accuracy"]]

        metrics = _run({"good": good, "bad": bad})
        assert (metrics.epochs_to_convergence["good"]
                == metrics.epochs_to_convergence["bad"]), (
            "the probe failed to produce the tie this metric's weakness predicts: "
            f"{metrics.epochs_to_convergence}"
        )

    def test_the_weakness_is_documented(self):
        import inspect

        source = inspect.getsource(
            TrainingDynamicsAnalyzer._compute_training_metrics)
        assert "own peak" in source, (
            "_compute_training_metrics does not document that the convergence "
            "threshold is relative to each model's own peak accuracy"
        )
