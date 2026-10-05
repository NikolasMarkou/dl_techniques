"""A sigmoid multi-label head must be scored with multi-label metrics (F-080, F-081).

The defect. `ModelAnalyzer._infer_output_activation` goes to real trouble to distinguish
a sigmoid multi-label head from a softmax one -- it checks the row sum, logs its reasoning,
and then **throws the answer away**: the activation name is never put in the prediction
cache, so `CalibrationAnalyzer` cannot see it. Every metric was therefore computed with
single-label assumptions regardless of the head:

| metric | why the single-label form is wrong for independent sigmoids |
|---|---|
| `brier_score` | the multiclass form SUMS over classes; for K independent labels that charges one sample K times, inflating the score by exactly K (3x on a 3-label task) |
| `ece` (top-1) | "was the single top class right" is not a calibration question when there is no single correct class |
| `entropy` | `-Σ p log p` is the Shannon entropy of a *distribution*; over independent sigmoids it is not an entropy and its maximum is not `log K` |
| `margin` | top-1 minus top-2 is not a margin over alternatives |
| `gini_coefficient` | `1 - Σp²` is the Simpson index of a distribution; there is none |

What is still correct for a sigmoid head, and is kept: `per_class_ece` (each column is an
independent probability against its own 0/1 indicator — D-015's classwise form is exactly
right here), and the Brier score under the per-label convention.

The Brier factor is the part that can be pinned numerically: MEASURED 3.0x on a 3-label
probe, which is the signature of the wrong convention rather than of a bad model.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from dl_techniques.analyzer.analyzers.calibration_analyzer import CalibrationAnalyzer
from dl_techniques.analyzer.calibration_metrics import (
    compute_brier_score,
    is_multilabel_output,
    compute_confidence_statistics,
)
from dl_techniques.analyzer.config import AnalysisConfig
from dl_techniques.analyzer.data_types import AnalysisResults, DataInput

#: A confident, genuinely multi-label head: rows do NOT sum to 1.
SIGMOID_PROBS = np.array([
    [0.90, 0.20, 0.05],
    [0.10, 0.85, 0.30],
    [0.60, 0.40, 0.10],
    [0.20, 0.15, 0.90],
])

#: A softmax head: every row sums to 1.
SOFTMAX_PROBS = np.array([
    [0.90, 0.08, 0.02],
    [0.10, 0.80, 0.10],
    [0.60, 0.30, 0.10],
    [0.20, 0.15, 0.65],
])

SIGMOID_LABELS = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
])

SOFTMAX_LABELS = np.array([0, 1, 0, 2])


class TestTheHeadRouterAgreesWithTheModelsOwnInference:
    """`is_multilabel_output` must not disagree with `_infer_output_activation`."""

    def test_sigmoid_rows_are_multilabel(self):
        assert is_multilabel_output(SIGMOID_PROBS) is True

    def test_softmax_rows_are_not_multilabel(self):
        assert is_multilabel_output(SOFTMAX_PROBS) is False

    def test_a_1_d_output_is_never_multilabel(self):
        assert is_multilabel_output(np.array([0.3, 0.7, 0.5])) is False

    def test_an_empty_array_is_not_multilabel(self):
        assert is_multilabel_output(np.zeros((0, 3))) is False

    def test_values_outside_the_unit_range_are_not_multilabel(self):
        """Logits are not sigmoid output; treating them as multi-label would be wrong."""
        assert is_multilabel_output(np.array([[2.0, -1.0, 0.5]])) is False

    def test_it_matches_the_models_own_inference_on_both_probes(self):
        """The strongest form: the two implementations must classify identically.

        `ModelAnalyzer._infer_output_activation` is a method on the orchestrator, so this
        reaches for the same row-sum test it uses and compares. If either changes its
        tolerance, this fails.
        """
        from dl_techniques.analyzer.model_analyzer import ModelAnalyzer

        analyzer = ModelAnalyzer.__new__(ModelAnalyzer)
        analyzer.config = AnalysisConfig()

        for probs, expected_activation in (
            (SOFTMAX_PROBS, 'softmax'),
            (SIGMOID_PROBS, 'sigmoid'),
        ):
            activation = analyzer._infer_output_activation('m', probs)
            assert activation == expected_activation
            expected_multilabel = (expected_activation == 'sigmoid')
            assert is_multilabel_output(probs) is expected_multilabel, (
                f"_infer_output_activation said {activation!r} but "
                f"is_multilabel_output said {is_multilabel_output(probs)!r}; the two "
                f"cannot be allowed to disagree about the convention"
            )


class TestTheBrierConventionsDifferByExactlyK:
    """F-080: the two conventions, pinned numerically."""

    def test_multilabel_divides_by_the_class_count(self):
        multiclass = compute_brier_score(SIGMOID_LABELS.astype(float), SIGMOID_PROBS)
        multilabel = compute_brier_score(SIGMOID_LABELS.astype(float), SIGMOID_PROBS,
                                         multilabel=True)
        assert multiclass == pytest.approx(multilabel * 3.0, rel=1e-12), (
            f"multiclass {multiclass!r} vs multilabel {multilabel!r}: the ratio is the "
            f"class count, which is the signature of the wrong convention"
        )

    def test_the_multilabel_score_stays_inside_the_unit_interval(self):
        score = compute_brier_score(SIGMOID_LABELS.astype(float), SIGMOID_PROBS,
                                    multilabel=True)
        assert 0.0 <= score <= 1.0, (
            f"{score!r} is outside [0, 1]; the per-label Brier score cannot exceed 1"
        )

    def test_the_multiclass_score_is_on_a_different_scale_than_the_multilabel_one(self):
        """The two conventions are not two estimates of one quantity.

        Multiclass Brier sums over classes and so lives in `[0, 2]`; the per-label form
        averages and lives in `[0, 1]`. A reader comparing a number against a published
        threshold needs to know which scale it is on, which is why the scale difference —
        not only the factor-of-K difference — is pinned.
        """
        # Uniform predictions against strictly ONE-HOT targets, so the arithmetic is
        # exact: every row scores (1/3-1)^2 + 2*(1/3-0)^2 = 2/3 under the multiclass
        # convention and (2/3)/3 = 2/9 under the per-label one.
        one_hot_targets = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]],
                                   dtype=float)
        uniform = np.full((4, 3), 1.0 / 3.0)
        multiclass = compute_brier_score(one_hot_targets, uniform)
        per_label = compute_brier_score(one_hot_targets, uniform, multilabel=True)
        assert multiclass == pytest.approx(2.0 / 3.0, rel=1e-12)
        assert per_label == pytest.approx(2.0 / 9.0, rel=1e-12)
        assert multiclass == pytest.approx(per_label * 3.0, rel=1e-12)

    def test_a_perfect_multilabel_forecast_scores_zero(self):
        targets = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
        assert compute_brier_score(targets, targets, multilabel=True) == pytest.approx(0.0)


class TestConfidenceStatisticsRespectTheHead:
    """F-081: single-label quantities are NaN, not wrong, for a sigmoid head."""

    def test_a_softmax_head_gets_the_ranking_quantities(self):
        stats = compute_confidence_statistics(SOFTMAX_PROBS)
        assert np.isfinite(stats['max_probability']).all()
        assert np.isfinite(stats['margin']).all()
        assert np.isfinite(stats['simpson_index']).all()
        # margin is the top-1 minus top-2 gap
        assert np.allclose(stats['margin'],
                           np.sort(SOFTMAX_PROBS, axis=1)[:, -1]
                           - np.sort(SOFTMAX_PROBS, axis=1)[:, -2])

    def test_a_sigmoid_head_gets_the_multilabel_quantities(self):
        stats = compute_confidence_statistics(SIGMOID_PROBS, multilabel=True)
        assert np.isfinite(stats['mean_predicted_probability']).all()
        assert np.isfinite(stats['predicted_positive_rate']).all()
        for key in ('max_probability', 'margin', 'gini_coefficient'):
            assert np.isnan(stats[key]).all(), (
                f"{key} is not NaN for a multi-label head: a plausible number where the "
                f"quantity is undefined is exactly the failure F-081 removes"
            )

    def test_the_simpson_index_is_named_both_ways(self):
        """F-082: `gini_coefficient` is the Simpson index, and the old key is kept.

        Renaming the published key silently would break every stored artifact, so the
        accurate name is added alongside rather than substituted.
        """
        stats = compute_confidence_statistics(SOFTMAX_PROBS)
        assert np.allclose(stats['gini_coefficient'], stats['simpson_index'])
        expected = 1.0 - np.sum(SOFTMAX_PROBS ** 2, axis=1)
        assert np.allclose(stats['simpson_index'], expected)

    def test_a_single_class_output_has_a_nan_margin_not_a_zero(self):
        """The old code returned `gini = 0.0` unconditionally for a single class.

        `margin` (top-1 minus top-2) needs two classes, so NaN is right. `1 - Σp²` for a
        single class is `1 - p²`, which is well defined — 0 for a certain forecast and 1
        for an impossible one — so it is NOT NaN-ed, and this asserts that too.
        """
        stats = compute_confidence_statistics(np.array([[1.0], [0.0]]))
        assert np.isnan(stats['margin']).all(), (
            "a top-1 minus top-2 margin needs at least two classes"
        )
        assert np.allclose(stats['simpson_index'], [0.0, 1.0]), (
            "for one class `1 - Σp²` is `1 - p²`, which is well defined"
        )


def _analyze(probs, labels, config=None):
    # `CalibrationAnalyzer.analyze` iterates `self.models`, so the analyzer must actually
    # BE given the model key it is later asked about.
    analyzer = CalibrationAnalyzer({'m': None}, config or AnalysisConfig())
    results = AnalysisResults()
    cache = {'m': {'predictions': probs, 'y_data': labels,
                   'x_data': np.zeros(len(labels)), 'status': 'success'}}
    analyzer.analyze(results, None, cache)
    return results


class TestTheAnalyzerRoutesAMultiLabelHead:
    def test_the_brier_score_uses_the_per_label_convention(self):
        results = _analyze(SIGMOID_PROBS, SIGMOID_LABELS)
        published = results.calibration_metrics['m']['brier_score']
        expected = compute_brier_score(SIGMOID_LABELS.astype(float), SIGMOID_PROBS,
                                       multilabel=True)
        assert published == pytest.approx(expected), (
            f"published {published!r}, per-label convention gives {expected!r}"
        )
        multiclass = compute_brier_score(SIGMOID_LABELS.astype(float), SIGMOID_PROBS)
        assert published != pytest.approx(multiclass), (
            "the multiclass (sum-over-classes) convention was published for a sigmoid "
            "head; that inflates the score by exactly K"
        )

    def test_the_top_1_ece_is_not_published(self):
        results = _analyze(SIGMOID_PROBS, SIGMOID_LABELS)
        assert results.calibration_metrics['m']['ece'] is None, (
            "a top-1 ECE was published for a multi-label head, where 'the single top "
            "class' does not exist"
        )

    def test_there_is_no_reliability_diagram_for_a_multi_label_head(self):
        results = _analyze(SIGMOID_PROBS, SIGMOID_LABELS)
        assert 'm' not in results.reliability_data, (
            "a reliability diagram was stored for a head with no top-1 correctness "
            "forecast to bin"
        )

    def test_per_class_ece_is_still_computed(self):
        """The classwise form is exactly right for a sigmoid head, so it is KEPT."""
        results = _analyze(SIGMOID_PROBS, SIGMOID_LABELS)
        per_class = results.calibration_metrics['m']['per_class_ece']
        assert len(per_class) == 3
        assert all(isinstance(v, float) and np.isfinite(v) for v in per_class)

    def test_the_conditional_top_1_ece_is_not_fabricated(self):
        results = _analyze(SIGMOID_PROBS, SIGMOID_LABELS)
        conditional = results.calibration_metrics['m']['per_class_conditional_top1_ece']
        assert all(v is None for v in conditional), (
            f"got {conditional!r}; these must be None, not a 0.0 standing in for an "
            f"undefined top-1 quantity"
        )

    def test_the_head_type_is_recorded(self):
        results = _analyze(SIGMOID_PROBS, SIGMOID_LABELS)
        assert results.calibration_metrics['m']['multilabel'] is True
        soft = _analyze(SOFTMAX_PROBS, SOFTMAX_LABELS)
        assert soft.calibration_metrics['m']['multilabel'] is False

    def test_entropy_is_absent_for_a_multi_label_head(self):
        results = _analyze(SIGMOID_PROBS, SIGMOID_LABELS)
        confidence = results.confidence_metrics['m']
        assert 'mean_entropy' not in confidence, (
            "a Shannon entropy was published for independent sigmoids, where it is not "
            "an entropy and its maximum is not log K"
        )

    def test_a_non_binary_indicator_is_rejected(self):
        """A multi-hot block must be 0/1, not a probability."""
        bad_labels = np.array([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]])
        results = _analyze(SIGMOID_PROBS, bad_labels)
        assert 'm' not in results.calibration_metrics, (
            "a non-0/1 multi-label target was accepted; it is a probability target, "
            "which this analyzer does not support and must not silently score"
        )


class TestASoftmaxHeadIsUnchanged:
    """The single-label path must be untouched by any of this."""

    def test_ece_and_reliability_are_still_published(self):
        results = _analyze(SOFTMAX_PROBS, SOFTMAX_LABELS)
        metrics = results.calibration_metrics['m']
        assert metrics['ece'] is not None and metrics['ece'] >= 0.0
        assert 'm' in results.reliability_data

    def test_the_brier_score_keeps_the_multiclass_convention(self):
        results = _analyze(SOFTMAX_PROBS, SOFTMAX_LABELS)
        published = results.calibration_metrics['m']['brier_score']
        expected = compute_brier_score(
            np.eye(3)[SOFTMAX_LABELS], SOFTMAX_PROBS)
        assert published == pytest.approx(expected)

    def test_entropy_is_still_published(self):
        results = _analyze(SOFTMAX_PROBS, SOFTMAX_LABELS)
        assert 'mean_entropy' in results.confidence_metrics['m']
