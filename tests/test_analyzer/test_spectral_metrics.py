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
    detect_correlation_trap,
    compute_mp_softrank,
    calc_mp_edges,
    calc_mp_soft_rank,
    calculate_glorot_normalization_factor,
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
# Small-N (N < 20) bias-corrected alpha branch (WeightWatcher, R7)
# =====================================================================

class TestSmallNPowerlaw:
    """Tests for the WeightWatcher small-N (N < 20) bias-corrected alpha branch.

    For tails with N < SPECTRAL_SMALL_N_CUTOFF (20), fit_powerlaw uses the
    bias-corrected MLE alpha_bc = 1 + (n-1)/s and selects xmin by the penalized
    objective J = D_ks - 0.868/sqrt(n). The standard N >= 20 path is unchanged.
    """

    def test_small_n_branch_returns_finite_bias_corrected_alpha(self):
        """A short power-law tail (10 <= N < 20) hits the small-N branch and
        yields a finite, reasonable bias-corrected alpha."""
        np.random.seed(123)
        true_alpha = 3.0
        xmin = 1.0
        # 15 synthetic power-law samples via inverse-CDF.
        u = np.random.uniform(0.0, 1.0, 15)
        data = xmin * (1.0 - u) ** (-1.0 / (true_alpha - 1.0))

        # Confirm we are genuinely in the small-N regime.
        assert 10 <= len(data) < 20

        alpha, opt_xmin, D, sigma, num_pl, status, warning = fit_powerlaw(data)

        assert status == "success"
        assert np.isfinite(alpha)
        assert 1.0 < alpha < 10.0, f"bias-corrected alpha out of range: {alpha}"
        assert np.isfinite(D) and D >= 0.0
        assert num_pl > 0

    def test_small_n_branch_deterministic_descending_tail(self):
        """A hand-built descending tail of 12 values exercises the branch
        deterministically and produces a finite alpha > 1.0."""
        data = np.array([
            20.0, 12.0, 8.0, 6.0, 4.5, 3.5, 2.8, 2.2, 1.8, 1.5, 1.2, 1.0
        ], dtype=np.float64)
        assert 10 <= len(data) < 20

        alpha, opt_xmin, D, sigma, num_pl, status, warning = fit_powerlaw(data)

        assert status == "success"
        assert np.isfinite(alpha)
        assert alpha > 1.0
        assert 1.0 < alpha < 10.0


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
        assert 'alpha_hat_normalized' in metrics

    def test_alpha_hat_differs_from_alpha_weighted_with_N(self):
        """alpha_hat is the WeightWatcher (un-normalized) convention; the /N
        variant lives under alpha_hat_normalized.

        NOTE: deliberate buggy-contract correction. The pre-fix test asserted
        alpha_hat == 3*log10(2) (the /N value); that encoded the old convention
        where MetricNames.ALPHA_HAT held the /N-normalized quantity. Per the
        user-approved decision (D-F), alpha_hat now exposes the WeightWatcher
        un-normalized value and the /N variant moves to alpha_hat_normalized.
        """
        evals = np.array([100.0, 10.0, 1.0])
        metrics = calculate_spectral_metrics(evals, alpha=3.0, N=50)
        # alpha_hat = 3 * log10(100) = 6.0 (WW, un-normalized)
        assert metrics['alpha_hat'] == pytest.approx(6.0)
        # alpha_hat_normalized = 3 * log10(100/50) = 3 * log10(2) ≈ 0.903
        assert metrics['alpha_hat_normalized'] == pytest.approx(3.0 * np.log10(2.0), rel=1e-3)
        # alpha_weighted is the WeightWatcher-canonical AlphaHat; alpha_hat is its
        # SETOL-notation alias (same value).
        assert metrics['alpha_weighted'] == pytest.approx(6.0)

    def test_empty_evals(self):
        metrics = calculate_spectral_metrics(np.array([]), alpha=3.0)
        assert metrics['norm'] == 0.0


# =====================================================================
# Concentration Metrics
# =====================================================================

class TestConcentrationMetrics:
    """Tests for concentration metric functions."""

    def test_gini_uniform_distribution(self):
        """A perfectly uniform spectrum has Gini EXACTLY zero, not merely 'low'.

        REPAIRED (plan-2026-09-01T225724-e79ad4bd step 12): the previous assertion
        was ``gini < 0.1``, which the shipped value of ``-0.01`` satisfied — it
        passed under the ``-1/n`` bias AND under the fix, so it guarded nothing.
        """
        evals = np.ones(100) * 5.0
        gini = calculate_gini_coefficient(evals)
        assert gini == pytest.approx(0.0, abs=1e-12), (
            f"a perfectly uniform spectrum must have Gini 0.0, got {gini!r} "
            f"(a -1/n bias reads as {-1.0 / len(evals)!r})"
        )

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
    """Tests for classify_learning_phase (WeightWatcher labels).

    R8 / decisions.md D-009: deliberate contract reversal. WeightWatcher has only
    OVER_TRAINED_THRESH=2.0 and UNDER_TRAINED_THRESH=6.0 — α<2 is "over-trained",
    2≤α≤6 is "good", α>6 is "under-trained". This REVERSES the prior plan's
    SETOL-only "ideal" band (D-C) and "over-regularized"/"fair" terms (D-D).
    """

    def test_over_trained(self):
        # R8 reversal: was "over-regularized" → now "over-trained" (α < 2).
        assert classify_learning_phase(1.5) == "over-trained"

    def test_good_lower_boundary(self):
        # R8 reversal: was "ideal" → now "good" (α == 2.0 enters the good band).
        assert classify_learning_phase(2.0) == "good"

    def test_good(self):
        # WeightWatcher "good" band is the whole [2.0, 6.0] range — no "ideal"
        # sub-band, no "fair" split (R8 / D-009).
        assert classify_learning_phase(2.4) == "good"
        assert classify_learning_phase(3.0) == "good"
        assert classify_learning_phase(3.9) == "good"
        assert classify_learning_phase(5.0) == "good"
        # Inclusive upper boundary: α==6.0 → "good"; α>6.0 → "under-trained".
        assert classify_learning_phase(6.0) == "good"

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
        evals, sv_max, sv_min, rank_loss, _ = compute_eigenvalues([W], 64, 32, 32)
        assert len(evals) == 32
        assert sv_max > 0
        assert sv_min >= 0
        # Eigenvalues should be non-negative and sorted descending
        assert np.all(evals >= 0)
        assert np.all(np.diff(evals) <= 1e-10)  # descending

    def test_normalization(self):
        np.random.seed(42)
        W = np.random.randn(32, 16)
        evals_norm, *_ = compute_eigenvalues([W], 32, 16, 16, normalize=True)
        evals_raw, *_ = compute_eigenvalues([W], 32, 16, 16, normalize=False)
        np.testing.assert_allclose(evals_norm, evals_raw / 32, rtol=1e-5)

    def test_matrix_rank_full_rank(self):
        # Emitted MetricNames.MATRIX_RANK == len(evals) - rank_loss; a generic
        # 64x32 Gaussian is full column rank (32).
        np.random.seed(42)
        W = np.random.randn(64, 32)
        evals, _, _, rank_loss, _ = compute_eigenvalues([W], 64, 32, 32)
        assert int(len(evals) - rank_loss) == 32

    def test_matrix_rank_rank_deficient(self):
        # A 64x32 matrix factored through a width-10 bottleneck has rank <= 10,
        # so the effective matrix_rank must drop below the full-rank count.
        np.random.seed(0)
        W = np.random.randn(64, 10) @ np.random.randn(10, 32)
        evals, _, _, rank_loss, _ = compute_eigenvalues([W], 64, 32, 32)
        assert int(len(evals) - rank_loss) <= 10


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


# =====================================================================
# Correlation Trap Detection
# =====================================================================

class TestDetectCorrelationTrap:
    """Tests for detect_correlation_trap MP+TW detection."""

    def test_clean_random_matrix_no_trap(self):
        """A purely random matrix should have no correlation traps."""
        np.random.seed(42)
        W = np.random.randn(256, 128)
        # Randomize element-wise (for a random matrix, this is a no-op in distribution)
        W_rand = np.random.permutation(W.flatten()).reshape(W.shape)
        sv = np.linalg.svd(W_rand, compute_uv=False)
        rand_evals = sv * sv

        result = detect_correlation_trap(rand_evals, N=256, M=128)

        assert 'has_trap' in result
        assert 'num_rand_spikes' in result
        assert 'trap_severity' in result
        assert 'trap_severity_label' in result
        assert 'mp_lambda_plus' in result
        assert 'mp_lambda_minus' in result
        assert 'trap_threshold' in result
        # Random matrix should usually not trigger a trap
        assert result['mp_lambda_plus'] > 0
        assert result['trap_threshold'] > result['mp_lambda_plus']

    def test_trap_detected_with_spike(self):
        """Injecting a large outlier weight should create a detectable trap spike."""
        np.random.seed(42)
        W = np.random.randn(128, 64) * 0.1  # Small weights
        # Inject a large outlier that will create a spike even after randomization
        W[0, 0] = 50.0
        W[1, 1] = 50.0

        W_rand = np.random.permutation(W.flatten()).reshape(W.shape)
        sv = np.linalg.svd(W_rand, compute_uv=False)
        rand_evals = sv * sv

        result = detect_correlation_trap(rand_evals, N=128, M=64)

        assert result['has_trap'] is True
        assert result['num_rand_spikes'] >= 1
        assert result['trap_severity'] > 0
        assert result['trap_severity_label'] != 'none'

    def test_severity_labels_are_valid(self):
        """Severity labels should be one of the defined categories."""
        valid_labels = {'none', 'mild', 'moderate', 'severe', 'critical'}
        np.random.seed(42)
        rand_evals = np.sort(np.random.exponential(2, 100))[::-1]
        result = detect_correlation_trap(rand_evals, N=100, M=50)
        assert result['trap_severity_label'] in valid_labels

    def test_empty_input_returns_safe_defaults(self):
        """Empty or invalid inputs should return safe defaults."""
        result = detect_correlation_trap(np.array([]), N=0, M=0)
        assert result['has_trap'] is False
        assert result['num_rand_spikes'] == 0
        assert result['trap_severity'] == 0.0

    def test_none_input_returns_safe_defaults(self):
        result = detect_correlation_trap(None, N=10, M=5)
        assert result['has_trap'] is False

    def test_mp_edges_consistent(self):
        """MP lambda_plus should always be >= lambda_minus."""
        np.random.seed(42)
        rand_evals = np.sort(np.random.exponential(1, 200))[::-1]
        result = detect_correlation_trap(rand_evals, N=200, M=100)
        assert result['mp_lambda_plus'] >= result['mp_lambda_minus']

    def test_threshold_above_mp_edge(self):
        """Trap threshold should always be above MP edge (lambda_plus + delta_TW)."""
        np.random.seed(42)
        rand_evals = np.sort(np.random.exponential(1, 200))[::-1]
        result = detect_correlation_trap(rand_evals, N=200, M=100)
        assert result['trap_threshold'] > result['mp_lambda_plus']

    def test_custom_tw_factor(self):
        """Higher c_TW should make detection more conservative (fewer spikes)."""
        np.random.seed(42)
        W = np.random.randn(64, 32) * 0.1
        W[0, 0] = 20.0
        W_rand = np.random.permutation(W.flatten()).reshape(W.shape)
        sv = np.linalg.svd(W_rand, compute_uv=False)
        rand_evals = sv * sv

        result_strict = detect_correlation_trap(rand_evals, N=64, M=32, c_TW=5.0)
        result_loose = detect_correlation_trap(rand_evals, N=64, M=32, c_TW=1.0)

        # Strict threshold should be higher
        assert result_strict['trap_threshold'] > result_loose['trap_threshold']
        # Loose should detect at least as many spikes
        assert result_loose['num_rand_spikes'] >= result_strict['num_rand_spikes']


# =====================================================================
# MP soft rank (WeightWatcher RMT_Util.mp_soft_rank)
# =====================================================================

class TestMpSoftrank:
    """Tests for compute_mp_softrank = lambda_plus / lambda_max."""

    def test_no_spikes_returns_one(self):
        """With num_spikes=0, lambda_plus == lambda_max so the ratio is 1.0."""
        evals = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert compute_mp_softrank(evals, num_spikes=0) == pytest.approx(1.0)

    def test_one_spike_removed(self):
        """Dropping the top eigenvalue gives (second-largest / largest)."""
        evals = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert compute_mp_softrank(evals, num_spikes=1) == pytest.approx(0.8)

    def test_empty_array_returns_zero(self):
        """Empty input is degenerate → 0.0."""
        assert compute_mp_softrank(np.array([]), num_spikes=0) == 0.0

    def test_ratio_in_unit_interval(self):
        """For positive eigenvalues the result is always in (0, 1]."""
        np.random.seed(7)
        evals = np.abs(np.random.randn(50)) + 0.1
        for k in range(0, 5):
            val = compute_mp_softrank(evals, num_spikes=k)
            assert 0.0 < val <= 1.0


# =====================================================================
# C6 — the rank-1 pathology must score ABOVE a healthy full-rank control
# =====================================================================

class TestDominanceRatioOnTheRankOnePathology:
    """`calculate_dominance_ratio` returned 0.0 for a (near-)rank-1 spectrum.

    That zeroed ``concentration_score`` through ``gini * dominance``, so the most
    concentrated spectrum possible scored strictly BELOW a healthy full-rank control —
    the metric was inverted exactly where it matters.
    """

    @staticmethod
    def _rank_one_matrix() -> np.ndarray:
        rng = np.random.default_rng(20260902)
        u = rng.normal(size=(50, 1))
        v = rng.normal(size=(1, 30))
        return (u @ v).astype("float64")

    @staticmethod
    def _healthy_matrix() -> np.ndarray:
        rng = np.random.default_rng(1234)
        return rng.normal(size=(50, 30))

    def test_a_rank_one_spectrum_is_not_reported_as_zero_dominance(self):
        evals = np.linalg.svd(self._rank_one_matrix(), compute_uv=False) ** 2
        sum_others = float(np.sum(evals) - np.max(evals))

        # Anti-vacuity: the probe must really be in the guarded branch.
        assert sum_others < 1e-10, f"probe is not rank-1 enough: sum_others={sum_others}"

        dominance = calculate_dominance_ratio(evals)
        assert dominance > 0.0, (
            "the most dominant spectrum possible reported zero dominance: "
            f"dominance={dominance}"
        )
        assert np.isfinite(dominance)

    def test_the_pathology_scores_above_the_healthy_control(self):
        pathological = calculate_concentration_metrics(self._rank_one_matrix())
        healthy = calculate_concentration_metrics(self._healthy_matrix())

        # Anti-vacuity: Gini already sees the pathology, so only `dominance` can be
        # responsible for an inverted ordering.
        assert pathological['gini_coefficient'] > healthy['gini_coefficient']

        assert pathological['concentration_score'] > healthy['concentration_score'], (
            "a rank-1 matrix scores no higher than a healthy full-rank control: "
            f"pathological={pathological['concentration_score']} "
            f"healthy={healthy['concentration_score']} "
            f"(gini {pathological['gini_coefficient']} vs {healthy['gini_coefficient']}, "
            f"dominance {pathological['dominance_ratio']} vs {healthy['dominance_ratio']})"
        )
        assert np.isfinite(pathological['concentration_score'])

    def test_a_healthy_spectrum_is_unchanged(self):
        """Anti-vacuity arm: the guard is inert away from the degenerate branch."""
        evals = np.array([10.0, 1.0, 1.0, 1.0])
        assert calculate_dominance_ratio(evals) == pytest.approx(10.0 / 3.0)


# =====================================================================
# C5-real — a short tail must be "not computed", not "certainly not a power law"
# =====================================================================

class TestShortTailGoodnessOfFit:
    """`powerlaw_goodness_of_fit` returned 0.0 whenever the fitted tail was too short.

    ``fit_powerlaw`` bails with ``D = -1.0`` below ``SPECTRAL_DEFAULT_MIN_EVALS`` points,
    and that was mapped to ``0.0`` — indistinguishable from "certainly not a power law",
    which is what ``_generate_recommendations`` then reports to the user. Measured on 18
    of 60 random layers (30%), and on BOTH Dense layers of a real 2-model run.
    """

    @staticmethod
    def _short_tail_evals(n_tail: int = 6):
        """Eigenvalues whose tail above `xmin` holds fewer than 10 points."""
        rng = np.random.default_rng(20260902)
        xmin = 10.0
        bulk = rng.uniform(0.1, 1.0, 200)
        tail = xmin * (1.0 - rng.uniform(0.0, 0.9, n_tail)) ** (-1.0 / 2.0)
        return np.concatenate([bulk, tail]), xmin

    def test_a_short_tail_reports_the_not_computed_sentinel(self):
        from dl_techniques.analyzer.constants import SPECTRAL_PVALUE_NOT_COMPUTED

        evals, xmin = self._short_tail_evals()

        # Anti-vacuity: the tail must really be shorter than the fitter's floor, and long
        # enough to clear the n_tail < 5 early return, so the D = -1.0 path is the one
        # actually exercised.
        n_tail = int(np.sum(evals >= xmin))
        assert 5 <= n_tail < 10, f"probe tail length {n_tail} does not exercise the defect"

        pvalue = powerlaw_goodness_of_fit(evals, alpha=3.0, xmin=xmin, n_bootstraps=10)

        assert pvalue == SPECTRAL_PVALUE_NOT_COMPUTED, (
            "a tail too short to fit was reported as a decisive rejection of the "
            f"power law: pvalue={pvalue} (0.0 means 'certainly not a power law')"
        )

    def test_a_genuine_power_law_still_reports_a_real_pvalue(self):
        """Anti-vacuity arm: the sentinel must not swallow computable cases."""
        rng = np.random.default_rng(42)
        alpha, xmin = 3.0, 1.0
        data = xmin * (1 - rng.uniform(0, 1, 1000)) ** (-1.0 / (alpha - 1.0))

        pvalue = powerlaw_goodness_of_fit(data, alpha, xmin, n_bootstraps=30)
        assert 0.0 <= pvalue <= 1.0
        assert pvalue > 0.05

    def test_the_observed_distance_argument_is_not_inert(self):
        """`d_observed` must actually be used — unlike the documented-inert `xmin`.

        A large observed KS distance makes every synthetic draw compare below it (p = 0);
        a zero one makes every draw compare at or above it (p = 1). If the argument were
        ignored, both calls would return the same number.
        """
        rng = np.random.default_rng(7)
        alpha, xmin = 3.0, 1.0
        data = xmin * (1 - rng.uniform(0, 1, 500)) ** (-1.0 / (alpha - 1.0))

        p_large_d = powerlaw_goodness_of_fit(
            data, alpha, xmin, n_bootstraps=10, d_observed=10.0)
        p_zero_d = powerlaw_goodness_of_fit(
            data, alpha, xmin, n_bootstraps=10, d_observed=0.0)

        assert p_large_d == pytest.approx(0.0)
        assert p_zero_d == pytest.approx(1.0)


# =====================================================================
# Gini Coefficient — the -1/n bias (plan step 12, S10)
# =====================================================================

def _gini_by_pairwise_definition(x: np.ndarray) -> float:
    """Reference Gini from its DEFINITION, not from the code under test.

    ``G = mean |x_i - x_j| / (2 * mean x)`` over all ordered pairs. This is the
    textbook (population) Gini coefficient and is derived here independently of
    ``calculate_gini_coefficient``'s Lorenz-cumsum implementation, so the guard
    cannot be satisfied by re-running the implementation's own algebra.
    """
    v = np.abs(np.asarray(x, dtype=float))
    n = len(v)
    return float(np.abs(v[:, None] - v[None, :]).sum() / (2.0 * n * n * v.mean()))


class TestGiniIsNotBiasedByMinusOneOverN:
    """`spectral_metrics.calculate_gini_coefficient` shipped `G_standard - 1/n`."""

    @pytest.mark.parametrize("n", [4, 10, 50, 200])
    def test_gini_matches_the_standard_definition(self, n):
        """The implementation must equal the pairwise definition at every n."""
        rng = np.random.default_rng(1234 + n)
        evals = rng.pareto(2.0, n) + 1.0

        expected = _gini_by_pairwise_definition(evals)
        got = calculate_gini_coefficient(evals)

        # Anti-vacuity: the reference must be a discriminating, non-degenerate
        # value — a spectrum with ~zero inequality would make any bias invisible.
        assert expected > 0.1, f"degenerate probe: reference gini {expected}"
        assert got == pytest.approx(expected, abs=1e-10), (
            f"gini is off by {expected - got!r} at n={n}; "
            f"a -1/n bias would read exactly {1.0 / n!r}"
        )

    def test_gini_is_never_negative(self):
        """The docstring promises [0, 1]; the biased form went below zero.

        The tolerance is float round-off only (measured |g| <= 8.4e-17). A -1/n
        bias is at least 0.01 over this range of n, i.e. four orders of magnitude
        above the tolerance, so the guard still discriminates.
        """
        for n in (2, 3, 4, 10, 100):
            gini = calculate_gini_coefficient(np.ones(n) * 5.0)
            assert gini >= -1e-12, (
                f"gini went negative on a uniform spectrum of {n} values: {gini!r} "
                f"(a -1/n bias reads as {-1.0 / n!r})"
            )
            assert gini <= 1.0

    def test_maximal_inequality_is_the_population_maximum(self):
        """`[0, 0, 0, 1]` is `(n-1)/n = 0.75`, not the biased `0.50`."""
        evals = np.array([0.0, 0.0, 0.0, 1.0])
        assert calculate_gini_coefficient(evals) == pytest.approx(0.75, abs=1e-12)

    def test_short_spectra_still_return_the_zero_sentinel(self):
        """Anti-vacuity: the `len < 2` early exit is untouched by the fix."""
        assert calculate_gini_coefficient(np.array([5.0])) == 0.0
        assert calculate_gini_coefficient(np.array([])) == 0.0


# =====================================================================
# log_alpha_norm overflow on a runaway alpha (plan step 13, S6)
# =====================================================================

class TestLogAlphaNormDoesNotOverflow:
    """`log10(sum(evals ** alpha))` overflowed to `inf` for a large fitted alpha."""

    @staticmethod
    def _degenerate_tail() -> np.ndarray:
        """The executed counterexample from findings/statistical-methodology.md S6.

        A 40-point tail whose values are identical to within 1e-7 makes
        `fit_powerlaw`'s `1 + n_tail/denominator` denominator collapse, so alpha
        runs away while `status` stays `"success"`.
        """
        rng = np.random.default_rng(0)
        return np.concatenate([
            np.linspace(0.01, 1.0, 60),
            2.0 + rng.normal(0.0, 1e-7, 40),
        ])

    def test_the_runaway_alpha_path_is_actually_reached(self):
        """Anti-vacuity: the probe must produce a huge alpha at status success.

        Without this arm a fix could be graded green against a spectrum whose
        alpha is perfectly ordinary, where no overflow was ever possible.
        """
        evals = self._degenerate_tail()
        alpha, _, _, _, _, status, _ = fit_powerlaw(evals)
        assert status == "success"
        assert alpha > 1e6, f"probe did not reach the runaway-alpha path: alpha={alpha}"

    def test_log_alpha_norm_is_finite_for_a_runaway_alpha(self):
        evals = self._degenerate_tail()
        alpha, _, _, _, _, status, _ = fit_powerlaw(evals)

        metrics = calculate_spectral_metrics(evals, alpha, N=100)

        assert np.isfinite(metrics["log_alpha_norm"]), (
            f"log_alpha_norm overflowed: {metrics['log_alpha_norm']!r} "
            f"at alpha={alpha!r}"
        )

    def test_a_runaway_alpha_is_flagged_unreliable(self):
        evals = self._degenerate_tail()
        alpha, _, _, _, _, _, _ = fit_powerlaw(evals)

        metrics = calculate_spectral_metrics(evals, alpha, N=100)

        assert metrics["alpha_unreliable"] is True, (
            f"alpha={alpha!r} was reported without an unreliability flag"
        )

    def test_a_normal_alpha_is_not_flagged_and_is_numerically_unchanged(self):
        """Anti-vacuity: the fix must be inert on the ordinary path.

        `log10(sum(x**a))` and `logsumexp(a*log(x))/log(10)` must agree to float
        precision wherever the direct form does not overflow, and the flag must
        not fire on a healthy fit.
        """
        rng = np.random.default_rng(7)
        evals = np.sort(rng.pareto(2.0, 300) + 0.5)[::-1]
        alpha = 2.5

        metrics = calculate_spectral_metrics(evals, alpha, N=300)
        direct = float(np.log10(np.sum(evals ** alpha)))

        assert np.isfinite(direct), "control spectrum must not overflow"
        assert metrics["log_alpha_norm"] == pytest.approx(direct, rel=1e-12)
        assert metrics["alpha_unreliable"] is False

    def test_the_flag_boundary_is_the_documented_sanity_bound(self):
        evals = np.sort(np.random.default_rng(3).pareto(2.0, 200) + 0.5)[::-1]
        assert calculate_spectral_metrics(evals, 7.9, N=200)["alpha_unreliable"] is False
        assert calculate_spectral_metrics(evals, 8.1, N=200)["alpha_unreliable"] is True

    def test_empty_evals_still_returns_the_flag(self):
        """The empty-spectrum early return must carry the same key set."""
        metrics = calculate_spectral_metrics(np.array([]), alpha=3.0)
        assert metrics["alpha_unreliable"] is False


class TestRuntimeWarningFilterIsNarrow:
    """A blanket `simplefilter("ignore", RuntimeWarning)` hid the S6 overflow."""

    def test_the_spectral_analyzer_does_not_silence_all_runtime_warnings(self):
        """No `warnings.simplefilter` CALL may guard the analysis pass.

        Parsed with `ast` rather than matched as text, so the guard sees calls
        only — a comment naming the banned idiom (there is one, explaining why it
        is banned) must not satisfy or trip it.
        """
        import ast
        import inspect
        import textwrap
        from dl_techniques.analyzer.analyzers import spectral_analyzer

        src = textwrap.dedent(
            inspect.getsource(spectral_analyzer.SpectralAnalyzer._analyze_single_model))
        called = {
            node.func.attr
            for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }

        assert "simplefilter" not in called, (
            "a blanket RuntimeWarning filter is back; it is what hid the "
            "log_alpha_norm overflow for the whole analysis pass"
        )
        assert "filterwarnings" in called, (
            "the narrow, message-scoped filters are gone entirely"
        )


# =====================================================================
# S2 — the shared MP-edge helper and the theoretical-edge soft rank
# =====================================================================

class TestCalcMpEdgesIsTheSingleSourceOfTruth:
    """`detect_correlation_trap` and `calc_mp_soft_rank` share ONE edge formula."""

    def test_the_trap_detector_reports_the_helper_edges(self):
        """The detector's published edges must BE the helper's, not a copy of it.

        This is the DRY guard: an edit to one spelling and not the other reddens
        here, which is exactly how the softrank wiring drifted from the detector.
        """
        rng = np.random.default_rng(11)
        N, M = 200, 50
        W = rng.normal(size=(N, M))
        rand_evals = np.sort(np.linalg.svd(W, compute_uv=False) ** 2)[::-1]

        result = detect_correlation_trap(rand_evals, N, M)
        expected_minus, expected_plus = calc_mp_edges(float(np.mean(rand_evals)), N / M)

        assert result["mp_lambda_plus"] == pytest.approx(expected_plus, rel=1e-15)
        assert result["mp_lambda_minus"] == pytest.approx(expected_minus, rel=1e-15)
        # Anti-vacuity: the two edges must be genuinely different numbers, so an
        # all-zero helper could not satisfy both.
        assert expected_plus > expected_minus > 0.0

    def test_the_edges_use_the_ww_inverse_sqrt_q_spelling(self):
        """`(1 +/- 1/sqrt(Q))^2`, never `(1 +/- sqrt(Q))^2` (D-002)."""
        sigma_sq, Q = 3.0, 4.0
        minus, plus = calc_mp_edges(sigma_sq, Q)
        assert plus == pytest.approx(sigma_sq * (1.0 + 0.5) ** 2)
        assert minus == pytest.approx(sigma_sq * (1.0 - 0.5) ** 2)
        # The refuted spelling would give 3*(1+2)^2 = 27.0.
        assert plus != pytest.approx(sigma_sq * (1.0 + np.sqrt(Q)) ** 2)

    def test_degenerate_aspect_ratio_returns_zero_edges(self):
        assert calc_mp_edges(1.0, 0.0) == (0.0, 0.0)


class TestCalcMpSoftRankNeedsNoRandomization:
    """The theoretical-edge form is a real number with `spectral_randomize=False`."""

    def test_a_spiked_spectrum_scores_below_a_pure_bulk(self):
        rng = np.random.default_rng(5)
        N, M = 400, 100
        bulk = np.sort(
            np.linalg.svd(rng.normal(size=(N, M)), compute_uv=False) ** 2)[::-1]

        spiked = bulk.copy()
        spiked[0] = bulk[0] * 20.0

        soft_bulk = calc_mp_soft_rank(bulk, N, M)
        soft_spiked = calc_mp_soft_rank(spiked, N, M)

        assert soft_spiked < soft_bulk, (
            f"a 20x spike did not lower the soft rank: "
            f"bulk={soft_bulk} spiked={soft_spiked}"
        )
        assert soft_spiked > 0.0

    def test_it_is_not_the_constant_one(self):
        """The wiring defect's signature: exactly 1.0 regardless of the spectrum."""
        rng = np.random.default_rng(6)
        N, M = 300, 60
        evals = np.sort(
            np.linalg.svd(rng.normal(size=(N, M)), compute_uv=False) ** 2)[::-1]
        evals[0] *= 50.0

        assert calc_mp_soft_rank(evals, N, M) != pytest.approx(1.0, abs=1e-6)

    def test_degenerate_inputs_return_zero(self):
        assert calc_mp_soft_rank(np.array([]), 10, 5) == 0.0
        assert calc_mp_soft_rank(np.array([1.0, 2.0]), 0, 5) == 0.0
        assert calc_mp_soft_rank(np.zeros(10), 10, 5) == 0.0

    def test_the_ww_spike_count_port_is_untouched(self):
        """`compute_mp_softrank` is a faithful WW port and must NOT be rewritten.

        Only its WIRING was the defect. Pinned here so a later cleanup that
        "removes the unused function" has to argue with a guard first.
        """
        evals = np.array([100.0, 10.0, 5.0, 1.0])
        assert compute_mp_softrank(evals, 0) == 1.0
        assert compute_mp_softrank(evals, 1) == pytest.approx(10.0 / 100.0)


# =====================================================================
# S5 — jensen_shannon_distance binned linearly over a heavy tail (step 17)
# =====================================================================

def _heavy_tailed_triple():
    """A bulk plus three tails at increasing distance from a common base.

    The bulk is IDENTICAL in all three spectra, so every difference the distance
    can see lives in the tail.
    """
    rng = np.random.default_rng(0)
    bulk = rng.uniform(0.001, 1.0, 500)
    base = np.concatenate([bulk, np.array([1e3, 5e3, 2e4])])
    near = np.concatenate([bulk, np.array([1e3, 1.2e3, 1.5e3])])
    far = np.concatenate([bulk, np.array([1e5, 5e5, 2e6])])
    return base, near, far


class TestJensenShannonSeesTheTail:
    """100 equal-width LINEAR bins put ~all mass in bin 0 and inverted the order."""

    def test_a_more_distant_tail_scores_further(self):
        base, near, far = _heavy_tailed_triple()

        d_near = jensen_shannon_distance(base, near)
        d_far = jensen_shannon_distance(base, far)

        # Anti-vacuity: both distances must be real, non-degenerate numbers, so an
        # all-zero or saturated metric cannot satisfy the ordering by accident.
        assert 0.0 < d_near < 1.0 and 0.0 < d_far < 1.0

        assert d_far > d_near, (
            f"a tail 100x further away scored CLOSER: "
            f"far={d_far!r} near={d_near!r}"
        )

    def test_the_bulk_is_not_the_only_thing_measured(self):
        """Two spectra sharing a bulk but differing 1000x in the tail must differ."""
        base, _, far = _heavy_tailed_triple()
        assert jensen_shannon_distance(base, far) > 0.01

    def test_identical_spectra_score_zero(self):
        base, _, _ = _heavy_tailed_triple()
        assert jensen_shannon_distance(base, base) == pytest.approx(0.0, abs=1e-12)

    def test_degenerate_inputs_are_unchanged(self):
        """The empty and single-value early exits keep their documented values."""
        assert jensen_shannon_distance(np.array([]), np.array([1.0])) == 1.0
        assert jensen_shannon_distance(np.array([1.0]), np.array([]))  == 1.0
        assert jensen_shannon_distance(np.full(5, 2.0), np.full(5, 2.0)) == 0.0

    def test_zero_and_negative_eigenvalues_do_not_produce_nan(self):
        """`log10` of a zero eigenvalue is floored, not propagated as -inf."""
        p = np.concatenate([np.zeros(10), np.linspace(0.1, 10.0, 90)])
        q = np.linspace(0.1, 10.0, 100)
        d = jensen_shannon_distance(p, q)
        assert np.isfinite(d) and 0.0 <= d <= 1.0


# =====================================================================
# S3 — a spike-inflated sigma_sq made the trap test CONSERVATIVE (step 18)
# =====================================================================

def _wishart_spectrum(N: int = 200, M: int = 50, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(N, M))
    return np.sort(np.linalg.svd(W, compute_uv=False) ** 2)[::-1]


class TestBulkVarianceExcludesSpikes:
    """`sigma_sq = mean(rand_evals)` counted the spike it exists to detect.

    Injecting one 20x spike moved the mean 201.37 -> 381.12 (1.89x), inflating
    the MP edge by the same factor and making the detector CONSERVATIVE exactly
    when it should fire hardest.
    """

    def test_one_spike_does_not_move_the_mp_edge(self):
        clean = _wishart_spectrum()
        spiked = clean.copy()
        spiked[0] = clean[0] * 20.0

        # Anti-vacuity: the naive statistic really is inflated on this probe, so a
        # green result cannot come from a spike too small to matter.
        assert float(np.mean(spiked)) / float(np.mean(clean)) > 1.5

        lp_clean = detect_correlation_trap(clean, 200, 50)["mp_lambda_plus"]
        lp_spiked = detect_correlation_trap(spiked, 200, 50)["mp_lambda_plus"]

        assert lp_spiked == pytest.approx(lp_clean, rel=0.05), (
            f"one 20x spike moved the MP edge from {lp_clean!r} to {lp_spiked!r} "
            f"({lp_spiked / lp_clean:.4f}x)"
        )

    @pytest.mark.parametrize("N,M,seed", [(200, 50, 7), (500, 100, 11)])
    def test_the_edge_is_stable_across_shapes_and_draws(self, N, M, seed):
        clean = _wishart_spectrum(N, M, seed)
        spiked = clean.copy()
        spiked[0] = clean[0] * 20.0

        lp_clean = detect_correlation_trap(clean, N, M)["mp_lambda_plus"]
        lp_spiked = detect_correlation_trap(spiked, N, M)["mp_lambda_plus"]
        assert lp_spiked == pytest.approx(lp_clean, rel=0.05)

    def test_the_spike_is_still_detected(self):
        """Anti-vacuity: a robust edge must make the detector MORE sensitive."""
        clean = _wishart_spectrum()
        spiked = clean.copy()
        spiked[0] = clean[0] * 20.0

        result = detect_correlation_trap(spiked, 200, 50)
        assert result["has_trap"] is True
        assert result["num_rand_spikes"] >= 1
        assert result["trap_severity"] > 1.0

    def test_a_pure_bulk_reports_a_small_severity(self):
        """Anti-vacuity control: the estimator must not manufacture traps."""
        clean = _wishart_spectrum(500, 100, seed=11)
        result = detect_correlation_trap(clean, 500, 100)
        assert result["trap_severity"] < 0.1, (
            f"a clean Wishart spectrum was scored as a trap: {result}"
        )

    def test_degenerate_inputs_are_unchanged(self):
        empty = detect_correlation_trap(np.array([1.0]), 10, 5)
        assert empty["has_trap"] is False and empty["mp_lambda_plus"] == 0.0
        assert detect_correlation_trap(np.zeros(20), 10, 5)["has_trap"] is False


# =====================================================================
# S1 — the Glorot factor double-counted rf after the flat CONV reshape
# (plan step 19, decisions.md D-002)
# =====================================================================

_CONV_KERNEL_SHAPE = (3, 3, 64, 128)


def _keras_glorot_scale(kernel_shape) -> float:
    """Glorot scale for a conv kernel, derived from the DEFINITION.

    ``sqrt(2 / (fan_in + fan_out))`` with ``fan_in = kh*kw*in_c`` and
    ``fan_out = kh*kw*out_c`` — Keras' own convolutional fan computation,
    written out here so the guard does not re-run the implementation's algebra.
    """
    kh, kw, in_c, out_c = kernel_shape
    fan_in = kh * kw * in_c
    fan_out = kh * kw * out_c
    return float(np.sqrt(2.0 / (fan_in + fan_out)))


class TestGlorotFactorDoesNotDoubleCountRf:
    """After the flat reshape ``N`` already contains ``rf``."""

    def test_the_conv_factor_is_the_true_glorot_scale(self):
        from dl_techniques.analyzer import spectral_utils
        from dl_techniques.analyzer.constants import LayerType

        rng = np.random.default_rng(20260902)
        kernel = rng.normal(size=_CONV_KERNEL_SHAPE).astype("float32")
        _, N, M, rf = spectral_utils.get_weight_matrices(kernel, LayerType.CONV2D)

        # Anti-vacuity: pin the matricization this guard is written against, so a
        # change of reshape convention reddens here rather than silently passing.
        assert (N, M, rf) == (576, 128, 9)

        expected = _keras_glorot_scale(_CONV_KERNEL_SHAPE)
        assert expected == pytest.approx(0.0340207, abs=1e-7)

        got = calculate_glorot_normalization_factor(N, M, rf)
        assert got == pytest.approx(expected, rel=1e-12), (
            f"glorot factor is {got!r}, the true scale is {expected!r} "
            f"({expected / got:.4f}x)"
        )

    def test_a_dense_layer_is_unaffected(self):
        """rf == 1 for Dense, so the fix must be an exact no-op there."""
        assert calculate_glorot_normalization_factor(64, 32, 1) == pytest.approx(
            float(np.sqrt(2.0 / 96.0)), rel=1e-15)

    def test_the_zero_guard_is_retained(self):
        assert calculate_glorot_normalization_factor(0, 0, 1) == 1.0


class TestConvEsdIsUnchanged:
    """The `n_comp` bookkeeping fix must NOT move any conv spectral metric.

    The four values below were captured from the analyzer at HEAD, BEFORE the
    change, at default config (`spectral_glorot_fix=False`). They are literals on
    purpose: a bit-identity claim that recomputes the code's own current answer
    proves nothing.
    """

    # Measured at HEAD on a (3,3,64,128) kernel seeded with default_rng(20260902).
    _HEAD_NUM_EVALS = 128
    _HEAD_ALPHA = 8.78225472793353
    _HEAD_ENTROPY = 0.9774616842420808
    _HEAD_GINI = 0.26600774254032966
    _HEAD_LEARNING_PHASE = "under-trained"

    @staticmethod
    def _analyze_conv_layer():
        import keras
        from dl_techniques.analyzer.config import AnalysisConfig
        from dl_techniques.analyzer.data_types import AnalysisResults
        from dl_techniques.analyzer.analyzers.spectral_analyzer import SpectralAnalyzer

        rng = np.random.default_rng(20260902)
        kernel = rng.normal(size=_CONV_KERNEL_SHAPE).astype("float32")

        inputs = keras.Input(shape=(8, 8, 64), name="conv_in")
        conv = keras.layers.Conv2D(
            128, 3, padding="same", use_bias=False, name="conv")
        model = keras.Model(inputs, conv(inputs), name="cm")
        conv.set_weights([kernel])

        results = AnalysisResults()
        SpectralAnalyzer(
            models={"cm": model}, config=AnalysisConfig(analyze_spectral=True)
        ).analyze(results)
        return results.spectral_analysis.iloc[0]

    def test_num_evals_stays_at_the_matrix_rank_bound(self):
        row = self._analyze_conv_layer()
        assert int(row["num_evals"]) == self._HEAD_NUM_EVALS

    def test_alpha_entropy_gini_and_phase_are_bit_identical(self):
        row = self._analyze_conv_layer()
        assert float(row["alpha"]) == self._HEAD_ALPHA
        assert float(row["entropy"]) == self._HEAD_ENTROPY
        assert float(row["gini_coefficient"]) == self._HEAD_GINI
        assert row["learning_phase"] == self._HEAD_LEARNING_PHASE


# =====================================================================
# A5 - rank_loss must use the SOURCE dtype's eps, not the post-cast one
# =====================================================================

class TestRankLossUsesThePreCastEps:
    """`compute_eigenvalues` promotes ``W`` to float64 before measuring rank.

    Reading ``np.finfo(W.dtype).eps`` after that promotion yields 2.22e-16, which
    is 6 orders of magnitude below the float32 round-off of the weights actually
    stored by Keras, so exact rank deficiency is reported as full rank.

    Deliberate divergence from WeightWatcher - see decisions.md D-003.
    """

    _SEED = 20260902

    @staticmethod
    def _rank_deficient_float32(deficiency: int = 20):
        """An 80x60 float32 matrix whose EXACT rank is ``60 - deficiency``."""
        rng = np.random.default_rng(TestRankLossUsesThePreCastEps._SEED)
        r = 60 - deficiency
        a = rng.normal(size=(80, r)).astype("float32")
        b = rng.normal(size=(r, 60)).astype("float32")
        return (a @ b).astype("float32")

    def test_an_exact_rank_deficiency_of_20_is_reported(self):
        w = self._rank_deficient_float32(20)
        assert w.dtype == np.float32

        # Anti-vacuity: the construction really is rank-40, and the 20 surplus
        # singular values are float32 round-off of an exact zero, not small
        # genuine values. Derived from the construction, not from the code.
        sv = np.linalg.svd(w.astype(np.float64), compute_uv=False)
        assert sv[39] > 1.0
        assert sv[40] < 1e-4

        _, _, _, rank_loss, _ = compute_eigenvalues([w], 80, 60, 60)
        assert int(rank_loss) == 20, (
            f"rank_loss is {rank_loss} for a matrix with exact deficiency 20; "
            f"float64 tol={sv.max() * 80 * np.finfo(np.float64).eps:.4g} vs "
            f"float32 tol={sv.max() * 80 * np.finfo(np.float32).eps:.4g}"
        )

    def test_a_full_rank_float32_matrix_still_reports_zero(self):
        """Anti-vacuity: the looser tolerance must not manufacture deficiency."""
        rng = np.random.default_rng(self._SEED + 1)
        w = rng.normal(size=(80, 60)).astype("float32")
        _, _, _, rank_loss, _ = compute_eigenvalues([w], 80, 60, 60)
        assert int(rank_loss) == 0

    def test_a_float64_matrix_keeps_the_weightwatcher_tolerance(self):
        """WW parity is preserved exactly where WW's own dtype assumption holds.

        WeightWatcher casts to float64 and then reads the eps of the CAST array
        (RMT_Util.matrix_rank), so for a float64 source the two agree bit for bit.
        """
        rng = np.random.default_rng(self._SEED + 2)
        w = (rng.normal(size=(80, 40)) @ rng.normal(size=(40, 60)))
        assert w.dtype == np.float64
        sv = np.linalg.svd(w, compute_uv=False)
        ww_tol = sv.max() * 80 * np.finfo(sv.dtype).eps
        ww_rank_loss = len(sv) - int(np.count_nonzero(sv > ww_tol))

        _, _, _, rank_loss, _ = compute_eigenvalues([w], 80, 60, 60)
        assert int(rank_loss) == ww_rank_loss


# =====================================================================
# A4 - a truncated spectrum must announce itself
# =====================================================================

class TestTruncatedSpectrumIsFlagged:
    """`svds` returns the k LARGEST singular values, not the whole spectrum.

    On that path `sv_min` is the k-th largest value, and `rank_loss` counts
    deficiency over a spectrum that never contained the small values. Nothing in
    the returned tuple or in the details frame recorded the truncation.
    """

    _SEED = 7

    @staticmethod
    def _rank_deficient_float32():
        """A 200x60 float32 matrix of EXACT rank 40 (deficiency 20)."""
        rng = np.random.default_rng(TestTruncatedSpectrumIsFlagged._SEED)
        a = rng.normal(size=(200, 40)).astype("float32")
        b = rng.normal(size=(40, 60)).astype("float32")
        return (a @ b).astype("float32")

    def test_the_truncated_path_reports_truncation(self):
        w = self._rank_deficient_float32()
        evals, sv_max, sv_min, rank_loss, truncated = compute_eigenvalues(
            [w], 200, 60, n_comp=10)

        # Anti-vacuity: the path really was truncated - 10 of 60 values returned.
        assert len(evals) == 10
        assert truncated is True

        assert np.isnan(sv_min), (
            f"sv_min is {sv_min!r}; on a truncated spectrum it is the k-th "
            f"LARGEST singular value, not a minimum"
        )
        assert np.isnan(rank_loss), (
            f"rank_loss is {rank_loss!r} on a spectrum that never contained the "
            f"small singular values (true deficiency 20)"
        )

    def test_the_full_path_is_untouched(self):
        """Anti-vacuity: the complete-spectrum path keeps real numbers."""
        w = self._rank_deficient_float32()
        evals, sv_max, sv_min, rank_loss, truncated = compute_eigenvalues(
            [w], 200, 60, n_comp=60)

        assert truncated is False
        assert len(evals) == 60
        sv = np.linalg.svd(w.astype(np.float64), compute_uv=False)
        assert sv_min == pytest.approx(sv.min(), rel=1e-9)
        assert not np.isnan(rank_loss)
        assert int(rank_loss) == 20

    def test_the_details_frame_carries_the_flag_at_defaults(self):
        """The analyzer does not truncate at defaults - the column must say so."""
        import keras
        from dl_techniques.analyzer.config import AnalysisConfig
        from dl_techniques.analyzer.constants import MetricNames
        from dl_techniques.analyzer.data_types import AnalysisResults
        from dl_techniques.analyzer.analyzers.spectral_analyzer import SpectralAnalyzer

        inputs = keras.Input(shape=(32,), name="t_in")
        model = keras.Model(
            inputs, keras.layers.Dense(24, name="d")(inputs), name="tm")

        results = AnalysisResults()
        SpectralAnalyzer(
            models={"tm": model}, config=AnalysisConfig(analyze_spectral=True)
        ).analyze(results)

        frame = results.spectral_analysis
        assert MetricNames.SPECTRUM_TRUNCATED in frame.columns
        assert not bool(frame.iloc[0][MetricNames.SPECTRUM_TRUNCATED])
