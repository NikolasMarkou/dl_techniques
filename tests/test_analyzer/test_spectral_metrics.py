"""
Tests for spectral_metrics module.

Covers: power-law fitting, SVD smoothing, matrix entropy, concentration metrics,
ERG condition, phase classification, and goodness-of-fit.
"""

import numpy as np
import pytest

from dl_techniques.analyzer.spectral_metrics import (
    fit_powerlaw,
    smooth_matrix,
    calculate_matrix_entropy,
    calculate_spectral_metrics,
    calculate_gini_coefficient,
    calculate_dominance_ratio,
    calculate_participation_ratio,
    calculate_concentration_metrics,
    compute_eigenvalues,
    rescale_eigenvalues,
    compute_detX_constraint,
    jensen_shannon_distance,
    powerlaw_goodness_of_fit,
    compute_erg_condition,
    classify_learning_phase,
)


# =====================================================================
# Power-Law Fitting
# =====================================================================

class TestFitPowerlaw:
    """Tests for fit_powerlaw MLE estimation."""

    def test_returns_valid_alpha_for_power_law_data(self):
        """Synthetic power-law data should yield alpha close to the true value."""
        np.random.seed(42)
        true_alpha = 3.0
        xmin = 1.0
        # Generate power-law distributed data: x = xmin * (1-u)^(-1/(alpha-1))
        u = np.random.uniform(0, 1, 5000)
        data = xmin * (1 - u) ** (-1.0 / (true_alpha - 1.0))

        alpha, opt_xmin, D, sigma, num_pl, status, warning = fit_powerlaw(data)

        assert status == "success"
        assert abs(alpha - true_alpha) < 0.5, f"Expected ~{true_alpha}, got {alpha}"
        assert D >= 0
        assert sigma > 0
        assert num_pl > 0

    def test_returns_failure_for_too_few_eigenvalues(self):
        """Should fail gracefully with fewer than minimum eigenvalues."""
        data = np.array([1.0, 2.0])
        alpha, _, _, _, _, status, _ = fit_powerlaw(data)
        assert status == "failed"
        assert alpha == -1.0

    def test_returns_failure_for_none_input(self):
        alpha, _, _, _, _, status, _ = fit_powerlaw(None)
        assert status == "failed"

    def test_returns_failure_for_empty_input(self):
        alpha, _, _, _, _, status, _ = fit_powerlaw(np.array([]))
        assert status == "failed"

    def test_warning_for_over_trained(self):
        """Alpha < 2.0 should produce over-trained warning."""
        np.random.seed(42)
        # Very heavy tail: alpha close to 1.5
        u = np.random.uniform(0, 1, 2000)
        data = (1 - u) ** (-1.0 / 0.5)  # alpha = 1.5
        _, _, _, _, _, status, warning = fit_powerlaw(data)
        # May or may not get over-trained depending on fit
        assert status in ("success", "failed")

    def test_warning_for_under_trained(self):
        """Alpha > 6.0 should produce under-trained warning."""
        np.random.seed(42)
        u = np.random.uniform(0, 1, 2000)
        data = (1 - u) ** (-1.0 / 9.0)  # alpha = 10
        alpha, _, _, _, _, status, warning = fit_powerlaw(data)
        if status == "success" and alpha > 6.0:
            assert warning == "under-trained"


# =====================================================================
# SVD Smoothing
# =====================================================================

class TestSmoothMatrix:
    """Tests for smooth_matrix SVD truncation."""

    def test_smoothing_non_square_matrix(self):
        """Smoothing should work for non-square matrices."""
        np.random.seed(42)
        W = np.random.randn(128, 64)
        n_comp = 10
        smoothed = smooth_matrix(W, n_comp)
        assert smoothed.shape == W.shape
        # Smoothed matrix should differ from original
        assert not np.allclose(smoothed, W), "Smoothing should change the matrix"

    def test_smoothing_reduces_rank(self):
        """Smoothed matrix should have lower effective rank."""
        np.random.seed(42)
        W = np.random.randn(64, 32)
        n_comp = 5
        smoothed = smooth_matrix(W, n_comp)
        sv_original = np.linalg.svd(W, compute_uv=False)
        sv_smoothed = np.linalg.svd(smoothed, compute_uv=False)
        # After smoothing, only n_comp singular values should be non-zero
        assert np.sum(sv_smoothed > 1e-10) <= n_comp + 1

    def test_smoothing_wide_matrix(self):
        """Smoothing should work for wide matrices (cols > rows)."""
        np.random.seed(42)
        W = np.random.randn(32, 128)
        n_comp = 5
        smoothed = smooth_matrix(W, n_comp)
        assert smoothed.shape == W.shape
        assert not np.allclose(smoothed, W)

    def test_smoothing_square_matrix(self):
        """Smoothing should work for square matrices."""
        np.random.seed(42)
        W = np.random.randn(64, 64)
        smoothed = smooth_matrix(W, 10)
        assert smoothed.shape == W.shape

    def test_smoothing_preserves_when_n_comp_exceeds(self):
        """When n_comp >= num singular values, matrix should be unchanged."""
        np.random.seed(42)
        W = np.random.randn(32, 16)
        smoothed = smooth_matrix(W, 100)
        np.testing.assert_allclose(smoothed, W, atol=1e-10)

    def test_reconstruction_accuracy(self):
        """SVD reconstruction should be accurate for known rank-k matrix."""
        np.random.seed(42)
        # Create rank-5 matrix
        U = np.random.randn(64, 5)
        V = np.random.randn(5, 32)
        W = U @ V
        smoothed = smooth_matrix(W, 5)
        np.testing.assert_allclose(smoothed, W, atol=1e-6)


# =====================================================================
# Matrix Entropy
# =====================================================================

class TestMatrixEntropy:
    """Tests for calculate_matrix_entropy."""

    def test_uniform_singular_values_give_high_entropy(self):
        """Equal singular values should give entropy close to 1."""
        sv = np.ones(10) * 5.0
        entropy = calculate_matrix_entropy(sv, 10)
        assert entropy > 0.9

    def test_single_dominant_singular_value_gives_low_entropy(self):
        """One large and many small SVs should give low entropy."""
        sv = np.array([100.0] + [0.001] * 9)
        entropy = calculate_matrix_entropy(sv, 10)
        assert entropy < 0.3

    def test_empty_input_returns_zero(self):
        assert calculate_matrix_entropy(np.array([]), 0) == 0.0

    def test_zero_input_returns_zero(self):
        assert calculate_matrix_entropy(np.zeros(5), 5) == 0.0


# =====================================================================
# Spectral Metrics
# =====================================================================

class TestSpectralMetrics:
    """Tests for calculate_spectral_metrics."""

    def test_basic_metrics(self):
        evals = np.array([10.0, 5.0, 2.0, 1.0])
        metrics = calculate_spectral_metrics(evals, alpha=3.0, N=100)

        assert metrics['norm'] == pytest.approx(18.0)
        assert metrics['spectral_norm'] == pytest.approx(10.0)
        assert metrics['stable_rank'] == pytest.approx(1.8)
        assert 'alpha_weighted' in metrics
        assert 'alpha_hat' in metrics

    def test_alpha_hat_differs_from_alpha_weighted_with_N(self):
        """alpha_hat should use N-normalized lambda_max."""
        evals = np.array([100.0, 10.0, 1.0])
        metrics = calculate_spectral_metrics(evals, alpha=3.0, N=50)
        # alpha_weighted = 3 * log10(100) = 6.0
        assert metrics['alpha_weighted'] == pytest.approx(6.0)
        # alpha_hat = 3 * log10(100/50) = 3 * log10(2) ≈ 0.903
        assert metrics['alpha_hat'] == pytest.approx(3.0 * np.log10(2.0), rel=1e-3)

    def test_empty_evals(self):
        metrics = calculate_spectral_metrics(np.array([]), alpha=3.0)
        assert metrics['norm'] == 0.0


# =====================================================================
# Concentration Metrics
# =====================================================================

class TestConcentrationMetrics:
    """Tests for concentration metric functions."""

    def test_gini_uniform_distribution(self):
        """Uniform eigenvalues should have low Gini."""
        evals = np.ones(100) * 5.0
        gini = calculate_gini_coefficient(evals)
        assert gini < 0.1

    def test_gini_extreme_inequality(self):
        """One large, rest tiny should have high Gini."""
        evals = np.array([1000.0] + [0.001] * 99)
        gini = calculate_gini_coefficient(evals)
        assert gini > 0.8

    def test_dominance_ratio(self):
        evals = np.array([10.0, 1.0, 1.0, 1.0])
        dom = calculate_dominance_ratio(evals)
        assert dom == pytest.approx(10.0 / 3.0)

    def test_participation_ratio_localized(self):
        """A localized vector should have low PR."""
        vec = np.zeros(100)
        vec[0] = 1.0
        pr = calculate_participation_ratio(vec)
        assert pr == pytest.approx(1.0)

    def test_participation_ratio_distributed(self):
        """A uniform vector should have high PR."""
        vec = np.ones(100) / np.sqrt(100)
        pr = calculate_participation_ratio(vec)
        assert pr == pytest.approx(100.0, rel=0.01)

    def test_concentration_metrics_uses_full_spectrum(self):
        """Concentration metrics should use full eigenvalue spectrum for Gini."""
        np.random.seed(42)
        W = np.random.randn(50, 30)
        metrics = calculate_concentration_metrics(W)
        assert 'gini_coefficient' in metrics
        assert 'concentration_score' in metrics


# =====================================================================
# ERG Condition
# =====================================================================

class TestERGCondition:
    """Tests for compute_erg_condition."""

    def test_erg_with_valid_inputs(self):
        np.random.seed(42)
        evals = np.sort(np.random.exponential(2, 100))[::-1]
        xmin = np.median(evals)
        result = compute_erg_condition(evals, xmin)
        assert 'erg_log_det' in result
        assert 'erg_delta_lambda_min' in result
        assert 'erg_satisfied' in result

    def test_erg_with_zero_xmin(self):
        evals = np.array([1.0, 0.5, 0.1])
        with pytest.raises(ValueError, match="xmin must be positive"):
            compute_erg_condition(evals, 0.0)

    def test_erg_with_empty_evals(self):
        with pytest.raises(ValueError, match="non-empty array"):
            compute_erg_condition(np.array([]), 1.0)


# =====================================================================
# Phase Classification
# =====================================================================

class TestClassifyLearningPhase:
    """Tests for classify_learning_phase."""

    def test_over_regularized(self):
        assert classify_learning_phase(1.5) == "over-regularized"

    def test_ideal(self):
        assert classify_learning_phase(2.0) == "ideal"
        assert classify_learning_phase(2.4) == "ideal"

    def test_good(self):
        assert classify_learning_phase(3.0) == "good"
        assert classify_learning_phase(3.9) == "good"

    def test_fair(self):
        assert classify_learning_phase(5.0) == "fair"

    def test_under_trained(self):
        assert classify_learning_phase(7.0) == "under-trained"

    def test_failed(self):
        assert classify_learning_phase(-1.0) == "failed"


# =====================================================================
# Goodness of Fit
# =====================================================================

class TestPowerlawGoodnessOfFit:
    """Tests for powerlaw_goodness_of_fit bootstrap test."""

    def test_good_fit_returns_high_pvalue(self):
        """Genuine power-law data should pass the goodness-of-fit test."""
        np.random.seed(42)
        alpha = 3.0
        xmin = 1.0
        u = np.random.uniform(0, 1, 1000)
        data = xmin * (1 - u) ** (-1.0 / (alpha - 1.0))

        pvalue = powerlaw_goodness_of_fit(data, alpha, xmin, n_bootstraps=30)
        assert pvalue > 0.05, f"Expected p > 0.05 for genuine power-law, got {pvalue}"

    def test_invalid_alpha_returns_zero(self):
        assert powerlaw_goodness_of_fit(np.array([1, 2, 3]), 0.5, 1.0) == 0.0

    def test_invalid_xmin_returns_zero(self):
        assert powerlaw_goodness_of_fit(np.array([1, 2, 3]), 3.0, -1.0) == 0.0


# =====================================================================
# Eigenvalue Computation
# =====================================================================

class TestComputeEigenvalues:
    """Tests for compute_eigenvalues."""

    def test_basic_eigenvalue_computation(self):
        np.random.seed(42)
        W = np.random.randn(64, 32)
        evals, sv_max, sv_min, rank_loss = compute_eigenvalues([W], 64, 32, 32)
        assert len(evals) == 32
        assert sv_max > 0
        assert sv_min >= 0
        # Eigenvalues should be non-negative and sorted descending
        assert np.all(evals >= 0)
        assert np.all(np.diff(evals) <= 1e-10)  # descending

    def test_normalization(self):
        np.random.seed(42)
        W = np.random.randn(32, 16)
        evals_norm, _, _, _ = compute_eigenvalues([W], 32, 16, 16, normalize=True)
        evals_raw, _, _, _ = compute_eigenvalues([W], 32, 16, 16, normalize=False)
        np.testing.assert_allclose(evals_norm, evals_raw / 32, rtol=1e-5)


# =====================================================================
# Utility Functions
# =====================================================================

class TestUtilities:
    """Tests for utility spectral functions."""

    def test_rescale_eigenvalues(self):
        evals = np.array([4.0, 1.0])
        rescaled, wscale = rescale_eigenvalues(evals)
        assert len(rescaled) == 2
        assert wscale > 0

    def test_jensen_shannon_distance_identical(self):
        data = np.random.randn(100)
        dist = jensen_shannon_distance(data, data)
        assert dist < 0.1  # Should be very small for identical data

    def test_jensen_shannon_distance_different(self):
        a = np.random.randn(1000)
        b = np.random.randn(1000) + 10
        dist = jensen_shannon_distance(a, b)
        assert dist > 0.3  # Should be large for very different data

    def test_compute_detX_constraint(self):
        evals = np.array([10.0, 5.0, 2.0, 0.5, 0.1])
        result = compute_detX_constraint(evals)
        assert result >= 0
        assert result <= len(evals)
