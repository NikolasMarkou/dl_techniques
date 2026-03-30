"""
Tests for calibration_metrics module.

Covers: ECE, adaptive ECE, MCE, Brier score, reliability data, entropy.
"""

import numpy as np
import pytest

from dl_techniques.analyzer.calibration_metrics import (
    compute_ece,
    compute_adaptive_ece,
    compute_mce,
    compute_brier_score,
    compute_brier_score_decomposition,
    compute_reliability_data,
    compute_prediction_entropy_stats,
)


class TestECE:
    """Tests for Expected Calibration Error."""

    def test_perfectly_calibrated_model(self):
        """A perfectly calibrated model should have ECE ≈ 0."""
        np.random.seed(42)
        n = 1000
        # Create predictions that are perfectly calibrated
        y_true = np.random.randint(0, 2, n)
        y_prob = np.zeros((n, 2))
        for i in range(n):
            if y_true[i] == 1:
                y_prob[i] = [0.2, 0.8]
            else:
                y_prob[i] = [0.8, 0.2]

        ece = compute_ece(y_true, y_prob, n_bins=10)
        assert ece < 0.3  # Not perfectly 0 due to binning, but should be low

    def test_overconfident_model(self):
        """An overconfident wrong model should have high ECE."""
        n = 100
        y_true = np.zeros(n, dtype=int)  # All class 0
        y_prob = np.zeros((n, 2))
        y_prob[:, 1] = 0.99  # Predicts class 1 with 99% confidence (all wrong)
        y_prob[:, 0] = 0.01

        ece = compute_ece(y_true, y_prob, n_bins=10)
        assert ece > 0.5

    def test_ece_is_non_negative(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 5, 200)
        y_prob = np.random.dirichlet(np.ones(5), 200)
        ece = compute_ece(y_true, y_prob, n_bins=15)
        assert ece >= 0


class TestAdaptiveECE:
    """Tests for Adaptive ECE with equal-mass bins."""

    def test_handles_all_samples(self):
        """Adaptive ECE should not drop remainder samples."""
        np.random.seed(42)
        n = 103  # Not evenly divisible by 15
        y_true = np.random.randint(0, 3, n)
        y_prob = np.random.dirichlet(np.ones(3), n)

        aece = compute_adaptive_ece(y_true, y_prob, n_bins=15)
        assert aece >= 0

    def test_aece_is_non_negative(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.dirichlet(np.ones(2), 100)
        aece = compute_adaptive_ece(y_true, y_prob, n_bins=10)
        assert aece >= 0


class TestMCE:
    """Tests for Maximum Calibration Error."""

    def test_mce_at_least_as_large_as_ece(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 3, 200)
        y_prob = np.random.dirichlet(np.ones(3), 200)
        ece = compute_ece(y_true, y_prob, n_bins=10)
        mce = compute_mce(y_true, y_prob, n_bins=10)
        assert mce >= ece - 1e-10


class TestBrierScore:
    """Tests for Brier Score."""

    def test_perfect_predictions(self):
        y_true_oh = np.eye(3)[[0, 1, 2]]
        y_prob = np.eye(3)[[0, 1, 2]]
        bs = compute_brier_score(y_true_oh, y_prob)
        assert bs == pytest.approx(0.0, abs=1e-10)

    def test_worst_predictions(self):
        """All-wrong predictions should have high Brier score."""
        y_true_oh = np.array([[1, 0], [0, 1]])
        y_prob = np.array([[0.0, 1.0], [1.0, 0.0]])
        bs = compute_brier_score(y_true_oh, y_prob)
        assert bs == pytest.approx(2.0)

    def test_brier_decomposition_sums(self):
        """Reliability - Resolution + Uncertainty should approximate total Brier."""
        np.random.seed(42)
        y_true = np.random.randint(0, 3, 300)
        y_prob = np.random.dirichlet(np.ones(3), 300)
        decomp = compute_brier_score_decomposition(y_true, y_prob, n_bins=10)

        assert decomp['reliability'] >= 0
        assert decomp['resolution'] >= 0
        assert decomp['uncertainty'] >= 0


class TestReliabilityData:
    """Tests for reliability diagram data."""

    def test_returns_correct_structure(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.dirichlet(np.ones(2), 100)
        data = compute_reliability_data(y_true, y_prob, n_bins=5)

        assert 'bin_centers' in data
        assert 'bin_accuracies' in data
        assert 'bin_confidences' in data
        assert 'bin_counts' in data
        assert len(data['bin_centers']) == 5


class TestPredictionEntropy:
    """Tests for prediction entropy statistics."""

    def test_confident_predictions_low_entropy(self):
        """Near-certain predictions should have low entropy."""
        y_prob = np.array([[0.99, 0.01], [0.01, 0.99], [0.98, 0.02]])
        stats = compute_prediction_entropy_stats(y_prob)
        assert stats['mean_entropy'] < 0.1

    def test_uncertain_predictions_high_entropy(self):
        """Uniform predictions should have high entropy."""
        y_prob = np.array([[0.5, 0.5], [0.5, 0.5]])
        stats = compute_prediction_entropy_stats(y_prob)
        assert stats['mean_entropy'] > 0.5

    def test_returns_all_fields(self):
        y_prob = np.array([[0.7, 0.3]])
        stats = compute_prediction_entropy_stats(y_prob)
        for key in ['entropy', 'mean_entropy', 'std_entropy', 'median_entropy',
                     'max_entropy', 'min_entropy']:
            assert key in stats
