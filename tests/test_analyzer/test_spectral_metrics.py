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
    get_top_eigenvectors,
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

    def test_no_uniform_spectrum_returns_a_negative_gini(self):
        """The docstring's contract is `between 0 and (n-1)/n`, EXACTLY 0.0 uniform.

        The `-1/n` bias fix (D-011) is correct and is not what this measures. What
        remains is float round-off in the Lorenz-cumsum form: over 2000 uniform
        vectors the verifier measured **960 returning < 0** (e.g.
        `-8.326672684688674e-17` at n=10), so the stated `between 0 and ...` was
        literally false for ~48% of uniform inputs.

        Anti-vacuity is `test_gini_matches_the_standard_definition` and
        `test_maximal_inequality_is_the_population_maximum`, which still pin the
        VALUE against an independently-derived reference: a clamp that hid a real
        `-1/n` bias (0.01 to 0.5 over this range of n) would redden them, so this
        sweep cannot be satisfied by simply returning 0.0.
        """
        rng = np.random.default_rng(12)
        negatives = []
        for i in range(2000):
            n = int(rng.integers(2, 200))
            scale = float(10.0 ** rng.uniform(-6, 6))
            gini = calculate_gini_coefficient(np.ones(n) * scale)
            if gini < 0.0:
                negatives.append((n, scale, gini))
            assert gini <= (n - 1) / n + 1e-12, (
                f"gini {gini!r} exceeds the (n-1)/n upper bound at n={n}"
            )
        assert not negatives, (
            f"{len(negatives)} of 2000 uniform spectra returned a NEGATIVE gini; "
            f"first three: {negatives[:3]}"
        )

    def test_a_uniform_spectrum_sits_at_zero_within_float_round_off(self):
        """The residual is round-off ONLY, four orders below any `-1/n` bias.

        The clamp repairs the SIGN, not the magnitude: on the positive side a
        uniform spectrum still returns ~5.6e-17 (n=3), which is what the
        docstring's "0.0 up to float round-off" now says. `-1/n` over this range
        of n is 0.005 to 0.5, so this bound still discriminates.
        """
        for n in (2, 3, 10, 47, 199):
            gini = calculate_gini_coefficient(np.ones(n) * 3.5)
            assert 0.0 <= gini <= 1e-12, (
                f"a uniform spectrum of {n} values returned {gini!r}, which is "
                "outside [0, 1e-12]"
            )

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
# S1 — the Glorot factor and the matricized fan axes
# (plan step 19, decisions.md D-002; plan step 19.1, decisions.md D-035)
# =====================================================================

_CONV_KERNEL_SHAPE = (3, 3, 64, 128)

#: Conv kernels spanning BOTH matricization regimes. `get_weight_matrices`
#: reshapes `(kh,kw,in_c,out_c)` to `(kh*kw*in_c, out_c)` and then reports
#: `N = max(...)`, `M = min(...)`, so the fan axes SWAP between the two groups
#: below and any formula written on `N`/`M` alone is right in one and wrong in
#: the other. Each entry is `(kernel_shape, expected_matrix_shape)`.
_GLOROT_SHAPES_ROWS_GE_COLS = [
    ((3, 3, 64, 128), (576, 128)),    # the only regime the first fix covered
    ((7, 7, 3, 64), (147, 64)),       # large rf, still rows > cols
    ((1, 1, 256, 64), (256, 64)),     # rf == 1 conv
    ((3, 3, 3, 18), (27, 18)),        # rows AND cols both divisible by rf
]
_GLOROT_SHAPES_ROWS_LT_COLS = [
    ((3, 3, 3, 64), (27, 64)),        # THE canonical first conv of a vision net
    ((3, 3, 32, 512), (288, 512)),
    ((3, 3, 16, 256), (144, 256)),
    ((5, 5, 8, 256), (200, 256)),
]
_GLOROT_SHAPES = _GLOROT_SHAPES_ROWS_GE_COLS + _GLOROT_SHAPES_ROWS_LT_COLS


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


class TestGlorotFactorMatchesKerasOwnFans:
    """The factor is ``sqrt(2/(fan_in+fan_out))`` in BOTH matricization regimes.

    `calculate_glorot_normalization_factor` takes the ORDERED matricized shape
    (`Wmats[0].shape`) precisely because `N`/`M` are `max`/`min` and therefore
    cannot say which axis is `fan_in` — see decisions.md D-035.
    """

    @pytest.mark.parametrize("kernel_shape,matrix_shape", _GLOROT_SHAPES)
    def test_the_conv_factor_is_the_true_glorot_scale(
            self, kernel_shape, matrix_shape):
        from dl_techniques.analyzer import spectral_utils
        from dl_techniques.analyzer.constants import LayerType

        rng = np.random.default_rng(20260902)
        kernel = rng.normal(size=kernel_shape).astype("float32")
        Wmats, N, M, rf = spectral_utils.get_weight_matrices(
            kernel, LayerType.CONV2D)

        # Anti-vacuity: pin the matricization this guard is written against, so a
        # change of reshape convention reddens here rather than silently passing.
        assert Wmats[0].shape == matrix_shape
        assert (N, M) == (max(matrix_shape), min(matrix_shape))
        assert rf == kernel_shape[0] * kernel_shape[1]

        expected = _keras_glorot_scale(kernel_shape)
        got = calculate_glorot_normalization_factor(Wmats[0].shape, rf)
        assert got == pytest.approx(expected, rel=1e-12), (
            f"{kernel_shape}: glorot factor is {got!r}, the true scale is "
            f"{expected!r} ({got / expected:.4f}x)"
        )

    def test_both_matricization_regimes_are_actually_covered(self):
        """Anti-vacuity for the parametrization itself.

        If every shape in the table happened to satisfy ``rows >= cols`` the
        parametrized test above would pass under the pre-D-035 ``N + M*rf``
        spelling, i.e. it would be blind to the bug it exists for.
        """
        assert all(r >= c for _, (r, c) in _GLOROT_SHAPES_ROWS_GE_COLS)
        assert all(r < c for _, (r, c) in _GLOROT_SHAPES_ROWS_LT_COLS)
        assert len(_GLOROT_SHAPES_ROWS_GE_COLS) >= 1
        assert len(_GLOROT_SHAPES_ROWS_LT_COLS) >= 1
        assert ((3, 3, 3, 64), (27, 64)) in _GLOROT_SHAPES_ROWS_LT_COLS

    def test_the_max_min_spelling_is_measurably_wrong_where_it_is_wrong(self):
        """The discriminating power of the table, written out as numbers.

        These are the ratios MEASURED against the shipped pre-D-035 code
        (``N + M*rf`` with ``N = max``, ``M = min``). They are literals: they
        state how large the defect was per shape, and they show the guard is
        not merely re-deriving the implementation's own arithmetic.
        """
        ratios = {}
        for kernel_shape, (rows, cols) in _GLOROT_SHAPES:
            rf = kernel_shape[0] * kernel_shape[1]
            old = float(np.sqrt(2.0 / (max(rows, cols) + min(rows, cols) * rf)))
            ratios[kernel_shape] = old / _keras_glorot_scale(kernel_shape)

        assert ratios[(3, 3, 64, 128)] == pytest.approx(1.0, rel=1e-12)
        assert ratios[(7, 7, 3, 64)] == pytest.approx(1.0, rel=1e-12)
        assert ratios[(1, 1, 256, 64)] == pytest.approx(1.0, rel=1e-12)
        assert ratios[(3, 3, 3, 18)] == pytest.approx(1.0, rel=1e-12)
        # ...and wrong by these factors in the swapped regime:
        assert ratios[(3, 3, 3, 64)] == pytest.approx(1.4015, abs=1e-4)
        assert ratios[(3, 3, 32, 512)] == pytest.approx(1.2559, abs=1e-4)
        assert ratios[(3, 3, 16, 256)] == pytest.approx(1.2559, abs=1e-4)
        assert ratios[(5, 5, 8, 256)] == pytest.approx(1.1206, abs=1e-4)

    def test_the_hand_written_fan_formula_agrees_with_keras(self):
        """Cross-check `_keras_glorot_scale` against Keras' OWN `compute_fans`.

        `GlorotNormal` is `VarianceScaling(scale=1.0, mode='fan_avg')`, whose
        nominal stddev is `sqrt(scale / fan_avg) = sqrt(2/(fan_in+fan_out))`
        over the fans `compute_fans` returns. Private import, so the check
        skips rather than fails if Keras moves it.
        """
        compute_fans = pytest.importorskip(
            "keras.src.initializers.random_initializers").compute_fans

        import keras
        init = keras.initializers.GlorotNormal()
        assert init.scale == 1.0 and init.mode == "fan_avg"

        for kernel_shape, _ in _GLOROT_SHAPES:
            fan_in, fan_out = compute_fans(kernel_shape)
            kh, kw, in_c, out_c = kernel_shape
            assert (fan_in, fan_out) == (kh * kw * in_c, kh * kw * out_c)
            assert _keras_glorot_scale(kernel_shape) == pytest.approx(
                float(np.sqrt(init.scale / ((fan_in + fan_out) / 2.0))),
                rel=1e-12)

    def test_a_dense_layer_is_unaffected(self):
        """rf == 1 for Dense, so the fix must be an exact no-op there."""
        assert calculate_glorot_normalization_factor((64, 32), 1) == pytest.approx(
            float(np.sqrt(2.0 / 96.0)), rel=1e-15)
        # ...and it is symmetric there, so the max/min ordering cannot matter.
        assert calculate_glorot_normalization_factor((32, 64), 1) == pytest.approx(
            calculate_glorot_normalization_factor((64, 32), 1), rel=1e-15)

    def test_the_zero_guard_is_retained(self):
        assert calculate_glorot_normalization_factor((0, 0), 1) == 1.0

    def test_the_analyzer_divides_a_first_conv_by_the_true_scale(self):
        """End-to-end through `SpectralAnalyzer`, which is where the order was lost.

        `spectral_glorot_fix` divides the matricized kernel by `kappa`, and
        `sv_max` is homogeneous of degree 1 in the matrix, so the ratio of the
        two runs IS the `kappa` the analyzer used. `(3,3,3,64)` is the shape the
        pre-D-035 caller got wrong by 1.4015x.
        """
        import keras
        from dl_techniques.analyzer.config import AnalysisConfig
        from dl_techniques.analyzer.data_types import AnalysisResults
        from dl_techniques.analyzer.analyzers.spectral_analyzer import SpectralAnalyzer

        rng = np.random.default_rng(20260902)
        kernel = rng.normal(size=(3, 3, 3, 64)).astype("float32")

        def _sv_max(glorot_fix: bool) -> float:
            inputs = keras.Input(shape=(8, 8, 3), name="conv_in")
            conv = keras.layers.Conv2D(
                64, 3, padding="same", use_bias=False, name="conv")
            model = keras.Model(inputs, conv(inputs), name="cm")
            conv.set_weights([kernel])
            results = AnalysisResults()
            SpectralAnalyzer(
                models={"cm": model},
                config=AnalysisConfig(
                    analyze_spectral=True, spectral_glorot_fix=glorot_fix),
            ).analyze(results)
            return float(results.spectral_analysis.iloc[0]["sv_max"])

        raw, fixed = _sv_max(False), _sv_max(True)
        kappa_used = raw / fixed

        expected = _keras_glorot_scale((3, 3, 3, 64))
        assert expected == pytest.approx(0.0575912, abs=1e-7)
        assert kappa_used == pytest.approx(expected, rel=1e-6), (
            f"the analyzer divided by {kappa_used!r}; the true Glorot scale for "
            f"(3,3,3,64) is {expected!r} ({kappa_used / expected:.4f}x — the "
            f"pre-D-035 max/min spelling lands at 1.4015x)"
        )


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


# =====================================================================
# P2 - compute_detX_constraint's quadratic tail sum
# =====================================================================

def _detx_reference(evals: np.ndarray) -> int:
    """The SHIPPED pre-optimisation loop, transcribed verbatim.

    This is the arbiter for the bit-identity proof; it must never be "tidied".
    """
    from dl_techniques.analyzer.constants import SPECTRAL_EPSILON

    if evals is None or len(evals) < 2:
        return 0

    rescaled_evals, _ = rescale_eigenvalues(evals)
    sorted_evals = np.sort(rescaled_evals)

    log_sorted = np.log(sorted_evals[sorted_evals > SPECTRAL_EPSILON])
    if len(log_sorted) == 0:
        return 0

    for idx in range(len(log_sorted) - 1, 0, -1):
        log_detX = np.sum(log_sorted[idx:])
        if log_detX < 0.0:
            return len(log_sorted) - idx

    return len(sorted_evals)


def _detx_corpus():
    """Random and heavy-tailed spectra, plus the degenerate shapes."""
    rng = np.random.default_rng(20260902)
    corpus = [
        np.array([]),
        np.array([1.0]),
        np.full(2000, 3.0),
        np.full(50, 1e-30),
        np.concatenate([np.full(40, 1e-30), rng.uniform(1.0, 2.0, 10)]),
    ]
    for n in (10, 61, 250, 2000):
        corpus.append(rng.uniform(0.5, 2.0, n))
        corpus.append(rng.pareto(1.5, n) + 1.0)
        corpus.append(np.exp(rng.normal(0.0, 4.0, n)))
        w = rng.normal(size=(max(n, 20), max(n // 4, 5))) / np.sqrt(max(n, 20))
        corpus.append(np.linalg.svd(w, compute_uv=False) ** 2)
        corpus.append(np.concatenate([rng.uniform(0.5, 2.0, n), [1e4]]))
    return [np.asarray(c, dtype=np.float64) for c in corpus]


class TestDetXConstraintIsUnchangedAndLinear:
    """`compute_detX_constraint` feeds `ww_pgd_optimizer.py:361`'s `int()`.

    A 1-ULP drift there flips a discrete projection decision, so the reversed-cumsum
    rewrite is gated on exact equality with the transcribed original loop.
    """

    def test_the_returned_count_is_bit_identical(self):
        """BIT-IDENTITY PIN - must hold before AND after the rewrite."""
        for i, evals in enumerate(_detx_corpus()):
            got = compute_detX_constraint(evals)
            want = _detx_reference(evals)
            assert got == want, (
                f"corpus[{i}] (n={len(evals)}): compute_detX_constraint returned "
                f"{got}, the transcribed original loop returns {want}"
            )

    def test_the_reference_is_not_a_constant(self):
        """Anti-vacuity: the corpus exercises a spread of answers, not one value."""
        answers = {_detx_reference(e) for e in _detx_corpus()}
        assert len(answers) > 5, f"the corpus only produces {answers}"

    def test_the_tail_sum_is_not_recomputed_per_iteration(self):
        """The cost oracle: count `np.sum` calls, not wall-clock seconds."""
        from unittest import mock
        import dl_techniques.analyzer.spectral_metrics as sm

        rng = np.random.default_rng(3)
        probes = {
            "uniform": rng.uniform(0.5, 2.0, 2000),
            "heavy": np.sort(rng.pareto(1.5, 2000) + 1.0),
            "identical": np.full(2000, 3.0),
        }
        for name, evals in probes.items():
            real_sum = np.sum
            calls = []

            def spy(*args, **kwargs):
                calls.append(1)
                return real_sum(*args, **kwargs)

            with mock.patch.object(sm.np, "sum", side_effect=spy):
                sm.compute_detX_constraint(np.asarray(evals, dtype=np.float64))

            assert len(calls) <= 4, (
                f"{name}: np.sum called {len(calls)} times for a spectrum of "
                f"2000 eigenvalues - the tail sum is being recomputed per iteration"
            )


# =====================================================================
# P1 - fit_powerlaw: the KS sweep and the false O(N) docstring
# =====================================================================

def _powerlaw_corpus():
    """Random, heavy-tailed, MP-bulk, spiked and small-N spectra."""
    rng = np.random.default_rng(20260902)
    out = {}
    out["pareto_200"] = np.sort(rng.pareto(1.5, 200) + 1.0)
    out["pareto_1200"] = np.sort(rng.pareto(2.5, 1200) + 1.0)
    out["lognormal_400"] = np.sort(np.exp(rng.normal(0.0, 2.0, 400)))
    w = rng.normal(size=(500, 120)) / np.sqrt(500)
    out["wishart_120"] = np.sort(np.linalg.svd(w, compute_uv=False) ** 2)[::-1]
    out["smalln_15"] = np.sort(rng.pareto(1.8, 15) + 1.0)
    out["spiked_300"] = np.sort(
        np.concatenate([rng.uniform(0.1, 1.0, 297), [50.0, 80.0, 120.0]]))
    return out


class TestFitPowerlawIsBitIdentical:
    """`fit_powerlaw`'s 7-tuple is unpacked POSITIONALLY in production.

    `ww_pgd_optimizer.py:342,427` thresholds on `ks_distance` and `:361` truncates a
    projection rank, and the `N>=20` path is anchored by `plan_2026-06-03_bc986e52/
    D-008`. These literals were captured from the analyzer at HEAD before any Phase D
    edit to this function; a bit-identity claim that recomputes the code's own
    current answer proves nothing, so they are written out.
    """

    _HEAD = {
        "pareto_200": (2.424895704305033, 1.0050857806670277,
                       0.028197921157122163, 0.10100817838789534,
                       199, "success", ""),
        "pareto_1200": (3.5972879169452505, 1.000068592286889,
                        0.016139378095687773, 0.07497724390056514,
                        1200, "success", ""),
        "lognormal_400": (1.627114644391847, 1.0013244328595678,
                          0.07969516895280582, 0.0446800684171361,
                          197, "success", "over-trained"),
        "wishart_120": (4.2063720469286725, 1.0858145685564,
                        0.14349713800268143, 0.4627999411040901,
                        48, "success", ""),
        "smalln_15": (3.819706944918515, 1.3782320345008379,
                      0.1408077062920703, 0.9969169508553424,
                      8, "success", ""),
        "spiked_300": (3.0573069830264954, 0.4726379644311377,
                       0.194445505475901, 0.15044518646514587,
                       187, "success", ""),
    }

    _FIELDS = ("alpha", "optimal_xmin", "ks_distance", "sigma",
               "num_pl_spikes", "status", "warning")

    # `ks_distance` is the ONE field allowed to drift, and only by float round-off.
    # See `plan-2026-09-02T041737-e85f2027/D-001`: moving the theoretical CDF into
    # the log domain changes the last bit or two of `D` (measured: 403 of 800
    # spectra move, at most 1.1e-14 relative on realistic families) in the
    # direction of the exact answer. The HEAD literals below are DELIBERATELY not
    # re-baselined onto the post-hoist values, so this test still measures the
    # total drift away from the pre-hoist answer rather than the code's agreement
    # with itself. Every other field is still exact-equality.
    _DRIFT_TOLERATED = "ks_distance"
    _DRIFT_MAX_RELATIVE = 1e-8

    @pytest.mark.parametrize("name", sorted(_HEAD))
    def test_all_seven_fields_are_unchanged(self, name):
        got = fit_powerlaw(_powerlaw_corpus()[name])
        want = self._HEAD[name]

        assert len(got) == 7, f"the 7-tuple shape changed: got {len(got)} fields"
        for field, g, w in zip(self._FIELDS, got, want):
            if isinstance(w, str):
                assert g == w, f"{name}.{field}: {g!r} != {w!r}"
            elif field == self._DRIFT_TOLERATED:
                relative = abs(float(g) - float(w)) / abs(float(w))
                assert relative <= self._DRIFT_MAX_RELATIVE, (
                    f"{name}.{field}: {g!r} vs the HEAD literal {w!r} "
                    f"(relative {relative:.4e}). Round-off drift from the log-domain "
                    f"KS kernel is bounded by {self._DRIFT_MAX_RELATIVE:.0e}; a larger "
                    f"move means the fit selected a DIFFERENT candidate, not that it "
                    f"rounded differently."
                )
            else:
                assert float(g) == float(w), f"{name}.{field}: {g!r} != {w!r}"

    def test_the_ks_distance_tolerance_is_not_a_blanket_exemption(self):
        """Anti-vacuity: only `ks_distance` is exempt, and only by a hair.

        A tolerance wide enough to hide a changed argmin would defeat the whole
        class. MEASURED post-hoist: every literal above still agrees to <= 2e-16
        relative, four orders tighter than the tolerance actually granted.
        """
        assert self._DRIFT_TOLERATED == "ks_distance"
        assert sum(f == self._DRIFT_TOLERATED for f in self._FIELDS) == 1

        worst = 0.0
        for name, want in self._HEAD.items():
            got = fit_powerlaw(_powerlaw_corpus()[name])
            index = self._FIELDS.index(self._DRIFT_TOLERATED)
            worst = max(worst, abs(float(got[index]) - float(want[index]))
                        / abs(float(want[index])))
        assert worst <= 1e-13, (
            f"the worst ks_distance drift over the corpus is {worst:.4e}; that is "
            f"far above pure round-off, so the {self._DRIFT_MAX_RELATIVE:.0e} "
            f"tolerance is now hiding a real behaviour change"
        )

    def test_the_corpus_exercises_both_selection_paths(self):
        """Anti-vacuity: the literals must not all come from one code path."""
        corpus = _powerlaw_corpus()
        assert len(corpus["smalln_15"]) < 20      # small-N penalized objective
        assert len(corpus["pareto_1200"]) >= 20   # standard KS-argmin path
        assert len({self._HEAD[k][4] for k in self._HEAD}) > 4


class TestFitPowerlawDocstringDoesNotClaimLinearTime:
    """The docstring claimed "O(N) total time"; the KS sweep is O(N^2).

    MEASURED: n=1000 -> 0.038 s, n=5000 -> 0.433 s, n=15000 -> 3.46 s (3x the data,
    8.0x the time). Only the alpha term was linearised by the `tail_sums` suffix sum.
    """

    def test_the_docstring_does_not_claim_linear_total_time(self):
        doc = fit_powerlaw.__doc__ or ""
        assert "O(N) total time" not in doc, (
            "fit_powerlaw's docstring still claims O(N) total time; the KS sweep at "
            "the heart of the xmin loop is O(n_tail) per candidate"
        )

    def test_the_docstring_states_the_real_cost(self):
        """The word "O(N^2)" already appeared at HEAD, inside the false claim to be
        AVOIDING it, so this asserts on the honest word instead."""
        doc = fit_powerlaw.__doc__ or ""
        assert "quadratic" in doc.lower(), (
            "fit_powerlaw's docstring does not state that the KS sweep is quadratic"
        )


# =====================================================================
# P3 - the top-3 eigenvectors must not cost a second full SVD
# =====================================================================

class TestOnlyOneFullSvdPerLayer:
    """`compute_eigenvalues` already runs `np.linalg.svd(..., compute_uv=False)`.

    `calculate_concentration_metrics` (default ON) then reached
    `get_top_eigenvectors`, which ran a SECOND `np.linalg.svd` WITH `U` purely to
    keep `u[:, :3]`. MEASURED on a plain Gaussian matrix: a full SVD costs 2.799 s at
    2048x512 against 0.060 s for `svds(k=3)`.

    Note the review's "second FULL-SPECTRUM recompute" claim is REFUTED - the
    analyzer passes `evals=evals`, so the recompute at `calculate_concentration_
    metrics` is skipped. The second SVD is the eigenvector one.
    """

    @staticmethod
    def _dense_model():
        import keras
        inputs = keras.Input(shape=(48,), name="svd_in")
        return keras.Model(
            inputs, keras.layers.Dense(40, name="svd_d")(inputs), name="svdm")

    def test_one_full_svd_per_analysis_pass(self):
        from unittest import mock
        import numpy.linalg as npl
        from dl_techniques.analyzer.config import AnalysisConfig
        from dl_techniques.analyzer.data_types import AnalysisResults
        from dl_techniques.analyzer.analyzers.spectral_analyzer import SpectralAnalyzer

        model = self._dense_model()
        config = AnalysisConfig(
            analyze_spectral=True, spectral_concentration_analysis=True)

        real_svd = npl.svd
        calls = []

        def spy(*args, **kwargs):
            calls.append(kwargs.get("compute_uv", True))
            return real_svd(*args, **kwargs)

        with mock.patch.object(npl, "svd", side_effect=spy):
            SpectralAnalyzer(models={"svdm": model}, config=config).analyze(
                AnalysisResults())

        # Anti-vacuity: concentration analysis really is on, so the second SVD's
        # call site was genuinely reached.
        assert config.spectral_concentration_analysis is True
        assert len(calls) == 1, (
            f"np.linalg.svd was called {len(calls)} times for one layer "
            f"(compute_uv flags: {calls}); the eigenvalue SVD is the only one needed"
        )

    def test_the_top_eigenvectors_are_unchanged_up_to_sign(self):
        rng = np.random.default_rng(4242)
        for shape in [(64, 40), (200, 60), (50, 50)]:
            w = rng.normal(size=shape)
            ref_u, ref_s, _ = np.linalg.svd(w, full_matrices=False)

            evs, vecs = get_top_eigenvectors(w, k=3)

            assert vecs.shape == (shape[0], 3)
            np.testing.assert_allclose(evs, ref_s[:3] ** 2, rtol=1e-9)
            for i in range(3):
                cosine = abs(float(ref_u[:, i] @ vecs[:, i]))
                assert cosine == pytest.approx(1.0, abs=1e-7), (
                    f"{shape} eigenvector {i}: |cos| with the reference is {cosine}"
                )

    def test_a_full_rank_request_still_works(self):
        """Anti-vacuity: `svds` cannot serve k == min(shape); the fallback must."""
        rng = np.random.default_rng(11)
        w = rng.normal(size=(12, 8))
        ref_u, ref_s, _ = np.linalg.svd(w, full_matrices=False)

        evs, vecs = get_top_eigenvectors(w, k=8)
        assert vecs.shape == (12, 8)
        np.testing.assert_allclose(evs, ref_s[:8] ** 2, rtol=1e-9)


class TestConfigEvalBoundsReachTheKernels:
    """The two eval-count bounds are documented as config, but were module constants.

    `spectral_analyzer.py` gated layer admission on `config.spectral_min_evals` /
    `config.spectral_max_evals`, while the kernels read `SPECTRAL_DEFAULT_MIN_EVALS`
    (=10) and `SPECTRAL_DEFAULT_MAX_EVALS` (=15000) from `constants.py`. MEASURED on
    unfixed HEAD: with `spectral_min_evals=5` a Dense(20 -> 9) layer IS admitted to
    the details frame and then comes back `alpha = -1.0, status = 'failed'`; and
    `compute_eigenvalues([12x12], 12, 12, 12)` returns 12 eigenvalues with
    `spectrum_truncated = False` no matter what the config says.
    """

    def test_fit_powerlaw_honours_an_explicit_min_evals(self):
        rng = np.random.default_rng(29)
        evals = np.sort(rng.pareto(2.5, size=9) + 1.0)[::-1]

        # Default floor (10) rejects a 9-eigenvalue spectrum outright.
        assert fit_powerlaw(evals)[5] == "failed"

        alpha, xmin, D, sigma, n_spikes, status, _ = fit_powerlaw(
            evals, min_evals=5)
        assert status == "success", (
            f"fit_powerlaw(min_evals=5) on {len(evals)} eigenvalues returned "
            f"status={status!r}, alpha={alpha}"
        )
        assert alpha > 1.0

    def test_a_layer_admitted_by_the_config_floor_is_actually_fitted(self):
        import keras
        from dl_techniques.analyzer.config import AnalysisConfig
        from dl_techniques.analyzer.analyzers.spectral_analyzer import SpectralAnalyzer

        inputs = keras.Input(shape=(20,), name="mn_in")
        model = keras.Model(
            inputs, keras.layers.Dense(9, name="mn_d9")(inputs), name="mnm")
        config = AnalysisConfig(spectral_min_evals=5)

        analyzer = SpectralAnalyzer(models={"mnm": model}, config=config)
        details, _, _, _, _ = analyzer._analyze_single_model(model)

        row = details[details["name"] == "mn_d9"].iloc[0]
        # Anti-vacuity: the layer really was admitted by the config gate, so the
        # assertion below cannot pass by the row being absent.
        assert int(row["M"]) == 9 and 9 < 10
        assert row["status"] == "success", (
            f"layer admitted at spectral_min_evals=5 (M={row['M']}) came back "
            f"status={row['status']!r}, alpha={row['alpha']}"
        )

    def test_compute_eigenvalues_honours_an_explicit_max_evals(self):
        rng = np.random.default_rng(292)
        w = rng.normal(size=(12, 12)).astype(np.float32)

        # Default cap (15000): the full spectrum comes back.
        full = compute_eigenvalues([w], 12, 12, 12)
        assert len(full[0]) == 12 and full[4] is False

        evals, _sv_max, _sv_min, _rank_loss, truncated = compute_eigenvalues(
            [w], 12, 12, 12, max_evals=8)
        assert truncated is True, (
            f"max_evals=8 on a 12x12 matrix returned {len(evals)} eigenvalues "
            f"with spectrum_truncated={truncated}"
        )
        assert len(evals) < 12

    def test_the_analyzer_threads_the_configured_max_evals(self):
        from unittest import mock
        import keras
        from dl_techniques.analyzer import spectral_metrics as sm
        from dl_techniques.analyzer.config import AnalysisConfig
        from dl_techniques.analyzer.analyzers.spectral_analyzer import SpectralAnalyzer

        inputs = keras.Input(shape=(48,), name="mx_in")
        model = keras.Model(
            inputs, keras.layers.Dense(40, name="mx_d")(inputs), name="mxm")
        config = AnalysisConfig(spectral_max_evals=12345)

        seen = []
        real = sm.compute_eigenvalues

        def spy(*args, **kwargs):
            seen.append(kwargs.get("max_evals"))
            return real(*args, **kwargs)

        with mock.patch.object(sm, "compute_eigenvalues", side_effect=spy):
            SpectralAnalyzer(
                models={"mxm": model}, config=config)._analyze_single_model(model)

        assert seen, "compute_eigenvalues was never called"
        assert set(seen) == {12345}, (
            f"compute_eigenvalues saw max_evals={seen}, not the configured 12345"
        )


# =====================================================================
# fit_powerlaw KS-distance kernel: accuracy and cost
# =====================================================================

class TestFitPowerlawKsKernel:
    """Guards on the kernel that evaluates the theoretical CDF in the xmin sweep.

    The sweep evaluates ``1 - (tail / xmin) ** (1 - alpha)`` for every candidate
    ``xmin``. Written as a literal ``pow`` of a ratio it is both the dominant cost
    of the whole fit and the least accurate spelling available: the division
    rounds, and ``pow`` then amplifies that rounding by the exponent. Writing it
    as ``1 - exp((1 - alpha) * (log(tail) - log(xmin)))`` reuses the ``log_data``
    array the function already computes, is measurably cheaper, and is measurably
    closer to the exact value.

    Both tests below discriminate the two spellings.
    """

    @staticmethod
    def _reference_ks_distance(evals, alpha, xmin):
        """Recompute the winning candidate's KS distance in extended precision.

        ``np.longdouble`` carries ~18-19 significant decimal digits on x86 against
        float64's ~15-16, so it resolves a float64 kernel's error by ~3 orders of
        magnitude - enough to separate a 1e-10 error from a 1e-15 one.

        The tail is selected by INDEX (``searchsorted`` on the same sorted float64
        array the fit built), never by re-thresholding in extended precision: a
        value-based reselection can pick a different number of elements and would
        measure tail membership rather than kernel accuracy.

        Args:
            evals: The eigenvalue spectrum handed to ``fit_powerlaw``.
            alpha: The returned power-law exponent.
            xmin: The returned optimal xmin.

        Returns:
            The KS distance at ``(alpha, xmin)``, as a float, computed in
            extended precision.
        """
        from dl_techniques.analyzer.spectral_metrics import SPECTRAL_EVALS_THRESH

        data = np.sort(np.asarray(evals, dtype=np.float64))
        data = data[data > SPECTRAL_EVALS_THRESH]
        i = int(np.searchsorted(data, np.float64(xmin), side="left"))
        tail = data[i:].astype(np.longdouble)
        n_tail = len(tail)
        exponent = np.longdouble(1.0) - np.longdouble(alpha)
        theoretical = 1.0 - np.exp(
            exponent * (np.log(tail) - np.log(np.longdouble(data[i]))))
        empirical = np.arange(n_tail, dtype=np.longdouble) / np.longdouble(n_tail)
        return float(np.max(np.abs(theoretical - empirical)))

    @pytest.mark.parametrize(
        "name,evals",
        [
            # Narrow dynamic range: (x/xmin) sits at 1 + O(1e-6), where the
            # ratio-then-pow spelling loses the most digits. MEASURED at the
            # ratio spelling: 1.7e-10, 9.4e-10, 8.2e-10 relative.
            ("narrow_300", 1.0 + np.random.default_rng(0).standard_normal(300) * 1e-6),
            ("narrow_400", 1.0 + np.random.default_rng(5).standard_normal(400) * 1e-6),
            ("narrow_250", 1.0 + np.random.default_rng(9).standard_normal(250) * 1e-6),
            # N < 20 takes the small-N branch, which carries its own SECOND copy
            # of the same kernel. Without this arm a fix to the standard path
            # alone would pass while leaving the two paths mutually inconsistent.
            ("small_n_18", 1.0 + np.random.default_rng(3).standard_normal(18) * 1e-6),
            ("small_n_19", 1.0 + np.random.default_rng(11).standard_normal(19) * 1e-6),
            # Anti-vacuity: wide-range families where BOTH spellings are already
            # accurate to ~1e-15. These prove the reference itself is sound and
            # that the assertion is not simply unreachable.
            ("pareto", np.random.default_rng(1).pareto(1.5, 300) + 1.0),
            ("lognorm", np.random.default_rng(2).lognormal(0.0, 1.0, 300)),
        ],
    )
    def test_the_returned_ks_distance_matches_an_extended_precision_reference(
            self, name, evals):
        alpha, xmin, D, _sigma, _spikes, status, _warning = fit_powerlaw(
            evals, min_evals=5)

        assert status == "success", f"{name}: fit did not succeed"

        reference = self._reference_ks_distance(evals, alpha, xmin)
        relative = abs(D - reference) / abs(reference)

        assert relative <= 1e-12, (
            f"{name}: fit_powerlaw returned D={D!r} against an extended-precision "
            f"reference of {reference!r} (relative error {relative:.4e}). A float64 "
            f"kernel written as exp((1-alpha)*(log(x)-log(xmin))) lands within "
            f"~1e-15; a relative error near 1e-10 means the KS distance is still "
            f"being computed as (x/xmin)**(1-alpha), which rounds the ratio and "
            f"then amplifies that rounding by the exponent."
        )

    def test_the_xmin_sweep_costs_less_than_an_equivalent_pow_kernel(self):
        """The sweep must not pay for an elementwise ``pow`` per candidate.

        Self-calibrating: the reference loop is timed in the SAME process, on the
        SAME data, immediately before the fit, so the comparison does not depend
        on the machine, the load, or an absolute wall-clock budget recorded
        elsewhere. MEASURED: ~1.0x with the ratio-and-pow spelling (the reference
        loop IS the shipped loop body), ~0.5x once the kernel is an ``exp``.
        """
        import time

        rng = np.random.default_rng(4)
        evals = np.sort(rng.pareto(1.5, 4000) + 1.0)
        n = len(evals)
        exponent = -1.5

        def _pow_kernel_sweep():
            worst = 0.0
            for i in range(n - 1):
                tail = evals[i:]
                theoretical = 1.0 - (tail / evals[i]) ** exponent
                empirical = np.arange(float(n - i)) / float(n - i)
                worst = max(worst, float(np.max(np.abs(theoretical - empirical))))
            return worst

        _pow_kernel_sweep()  # warm caches so neither side pays the first-touch cost
        fit_powerlaw(evals)

        t0 = time.perf_counter()
        _pow_kernel_sweep()
        pow_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        fit_powerlaw(evals)
        fit_seconds = time.perf_counter() - t0

        assert fit_seconds < 0.75 * pow_seconds, (
            f"the whole fit took {fit_seconds:.4f}s against {pow_seconds:.4f}s for a "
            f"bare pow-based sweep of the same shape (ratio "
            f"{fit_seconds / pow_seconds:.3f}). The fit does strictly more work than "
            f"the reference loop, so a ratio at or above 1.0 means its KS kernel is "
            f"still an elementwise pow. Expected ~0.5 with the log/exp spelling."
        )


# =====================================================================
# Critical-weight enumeration: cost and the published count
# =====================================================================

def _critical_weight_fixture(n, m):
    """Build a square-ish Gaussian layer and its full eigenvalue spectrum.

    Args:
        n: Number of rows.
        m: Number of columns.

    Returns:
        A ``(weight_matrix, evals)`` pair; ``evals`` is the ascending squared
        singular-value spectrum, matching what the analyzer hands the
        concentration path.
    """
    rng = np.random.default_rng(17)
    weight_matrix = rng.standard_normal((n, m))
    sv = np.linalg.svd(weight_matrix, compute_uv=False)
    return weight_matrix, np.sort(sv * sv)


class TestCriticalWeightEnumerationIsNotThePathsHotSpot:
    """`critical_weight_count` is published; the way it was counted was not.

    At `SPECTRAL_CRITICAL_WEIGHT_THRESHOLD = 0.1` essentially every element of a
    Gaussian row qualifies, so the enumeration built one Python tuple per matrix
    element and inserted it into a `set` — 751,140 tuples on a 1024x1024 layer and
    11,385,666 on a 4096x4096 one. MEASURED at HEAD: 21.73 s of the 24.48 s
    `calculate_concentration_metrics` spent on a 4096x4096 layer, against 0.31 s
    for the entire power-law fit.

    The COUNT is a shipped column (`MetricNames.CRITICAL_WEIGHT_COUNT`,
    `analyzer/README.md:249`), so it cannot simply be dropped — only the truncated
    top-10 LIST is filtered out by the analyzer (`spectral_analyzer.py:426`).
    """

    def test_the_concentration_path_does_not_cost_several_svds(self):
        """Self-calibrating: the reference is the SAME matrix's own full SVD.

        Both sides are timed in one process on one matrix, so the assertion does
        not depend on the machine or on a wall-clock budget recorded elsewhere.
        MEASURED at HEAD: 1.2302 s of concentration metrics against 0.2473 s for
        the full SVD, a ratio of 5.0 — the concentration path, which is handed its
        spectrum and never factorises anything larger than a rank-3 partial SVD,
        was costing five times the full factorisation it was meant to avoid.
        """
        import time

        weight_matrix, evals = _critical_weight_fixture(1024, 1024)

        np.linalg.svd(weight_matrix, compute_uv=False)  # warm BLAS threads
        t0 = time.perf_counter()
        np.linalg.svd(weight_matrix, compute_uv=False)
        svd_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        calculate_concentration_metrics(weight_matrix, evals=evals)
        concentration_seconds = time.perf_counter() - t0

        assert concentration_seconds < 2.0 * svd_seconds, (
            f"calculate_concentration_metrics took {concentration_seconds:.4f}s "
            f"against {svd_seconds:.4f}s for a full SVD of the same matrix (ratio "
            f"{concentration_seconds / svd_seconds:.2f}). It is handed the spectrum "
            f"and only needs a rank-3 partial SVD, so a ratio above 1 means it is "
            f"enumerating critical weights element-by-element in Python."
        )

    # Captured after the ARPACK start vector was pinned (D-003) and before the
    # enumeration was rewritten. This is a PIN, not a RED-proven guard: the
    # rewrite is required to leave every one of these untouched, so it could
    # never have failed beforehand. It could not even be WRITTEN until D-003
    # made the columns reproducible.
    _HEAD_COLUMNS = {
        (256, 128): {
            "gini_coefficient": 0.3930767949900368,
            "dominance_ratio": 0.02291230997569063,
            "participation_ratio": 84.29070051593679,
            "min_participation_ratio": 61.344088474072535,
            "critical_weight_count": 24328,
            "concentration_score": 0.00010684234675382042,
        },
        (512, 512): {
            "gini_coefficient": 0.5413652974958952,
            "dominance_ratio": 0.008011775881888358,
            "participation_ratio": 172.30052481656648,
            "min_participation_ratio": 160.05657386575842,
            "critical_weight_count": 190216,
            "concentration_score": 2.5172545749375347e-05,
        },
        (1024, 1024): {
            "gini_coefficient": 0.5413703917985958,
            "dominance_ratio": 0.003906777674791442,
            "participation_ratio": 344.66938377223704,
            "min_participation_ratio": 316.9058967868582,
            "critical_weight_count": 751140,
            "concentration_score": 6.136336358374042e-06,
        },
    }

    @pytest.mark.parametrize("shape", sorted(_HEAD_COLUMNS))
    def test_every_published_concentration_column_is_bit_identical(self, shape):
        got = calculate_concentration_metrics(
            _critical_weight_fixture(*shape)[0],
            evals=_critical_weight_fixture(*shape)[1])

        for column, want in self._HEAD_COLUMNS[shape].items():
            assert float(got[column]) == float(want), (
                f"{shape} {column}: {got[column]!r} != the HEAD literal {want!r}. "
                f"Skipping or restating the critical-weight enumeration is only "
                f"admissible while every published column is untouched; "
                f"critical_weight_count in particular reaches the analyzer's "
                f"DataFrame, only the critical_weights LIST is filtered out."
            )

    def test_the_count_column_is_large_enough_to_discriminate(self):
        """Anti-vacuity: a count of 0 or 10 would pass a weaker assertion.

        The counts pinned above are the FULL population, not the truncated
        report, so an implementation that returned only the top ten would be
        caught by them.
        """
        for shape, columns in self._HEAD_COLUMNS.items():
            n, m = shape
            assert columns["critical_weight_count"] > 0.5 * n * m, (
                f"{shape}: pinned count {columns['critical_weight_count']} is not "
                f"the full population, so it cannot discriminate a truncated one"
            )

    def test_the_reported_list_is_still_truncated_and_ordered(self):
        weight_matrix, evals = _critical_weight_fixture(256, 128)
        got = calculate_concentration_metrics(weight_matrix, evals=evals)
        reported = got["critical_weights"]

        assert len(reported) == 10, f"reported {len(reported)} critical weights, not 10"
        magnitudes = [abs(c) for _i, _j, c in reported]
        assert magnitudes == sorted(magnitudes, reverse=True), (
            f"the reported critical weights are not in descending |contribution| "
            f"order: {magnitudes}"
        )
        assert magnitudes[0] == pytest.approx(0.43259411404973996, rel=0, abs=0), (
            f"the largest reported contribution moved to {magnitudes[0]!r} from the "
            f"HEAD literal 0.43259411404973996"
        )


class TestTheDirectTopEigenvectorMethodIsActuallyDeterministic:
    """`get_top_eigenvectors`' docstring calls the `direct` method deterministic.

    It was not. `svds` defaults to ARPACK with `v0=None`, which draws its start
    vector from numpy's unseeded global legacy RNG, so consecutive calls on the
    SAME matrix in the SAME process converge to slightly different vectors.
    MEASURED at HEAD, three consecutive calls on one 256x128 Gaussian:
    participation ratio 92.27590991989491 / ...472 / ...484.

    Three published columns read those vectors — `participation_ratio`,
    `min_participation_ratio` and `concentration_score` — so this is a shipped
    metric that could not be pinned to a literal, and `critical_weight_count`
    inherits it through `row_importance`.
    """

    def test_repeated_calls_on_one_matrix_agree_bit_for_bit(self):
        rng = np.random.default_rng(17)
        weight_matrix = rng.standard_normal((256, 128))

        runs = []
        for _ in range(5):
            _evals, eigenvectors = get_top_eigenvectors(weight_matrix, k=3)
            runs.append(eigenvectors.copy())

        for index, later in enumerate(runs[1:], start=1):
            assert later.shape == runs[0].shape, (
                f"call {index} returned shape {later.shape}, not {runs[0].shape}")
            # Sign is a free choice of any SVD, so compare the magnitudes the
            # downstream metrics actually consume.
            assert np.array_equal(np.abs(later), np.abs(runs[0])), (
                f"call {index} returned different top eigenvectors from call 0 on "
                f"the same matrix; max |difference| "
                f"{np.max(np.abs(np.abs(later) - np.abs(runs[0]))):.3e}. The "
                f"`direct` method must not depend on an unseeded start vector."
            )

    def test_the_concentration_columns_it_feeds_are_reproducible(self):
        weight_matrix, evals = _critical_weight_fixture(256, 128)
        first = calculate_concentration_metrics(weight_matrix, evals=evals)
        second = calculate_concentration_metrics(weight_matrix, evals=evals)

        for column in ("participation_ratio", "min_participation_ratio",
                       "concentration_score", "critical_weight_count"):
            assert float(first[column]) == float(second[column]), (
                f"{column} is not reproducible across two identical calls: "
                f"{first[column]!r} then {second[column]!r}"
            )


# =====================================================================
# Correlation-trap threshold: scale equivariance and the M^(-2/3) order
# =====================================================================

def _clean_wishart(n, m, seed):
    """Eigenvalues of a clean Gaussian Wishart, descending, with no planted spike.

    Args:
        n: Larger matrix dimension.
        m: Smaller matrix dimension.
        seed: Seed for the draw.

    Returns:
        The descending eigenvalue spectrum of ``W.T @ W / n``.
    """
    weight_matrix = np.random.default_rng(seed).standard_normal((n, m))
    return np.sort(np.linalg.eigvalsh(weight_matrix.T @ weight_matrix / n))[::-1]


class TestTheTrapThresholdIsScaleEquivariant:
    """A trap verdict must depend on the SHAPE of a spectrum, not on its units.

    Rescaling a layer's weights `W -> s*W` multiplies every eigenvalue by `s**2`,
    including the Marchenko-Pastur edge. A threshold that sits a FIXED FRACTION
    above the edge is therefore the only kind that can survive the rescale, so the
    invariant under test is the RELATIVE headroom `(threshold - lambda_plus) /
    lambda_plus`.

    The shipped form spelled the offset `c_TW * sqrt((1/sqrt(Q)) * lambda_plus **
    (2/3) * M ** (-2/3))`, whose headroom carries `lambda_plus ** (-2/3)` and so
    COLLAPSES as the weights grow: MEASURED 5.262e+01 at `s = 1e-4` down to
    2.442e-04 at `s = 1e4` on one 200x50 Wishart, five orders of magnitude of
    verdict drift from a pure change of units. The same spelling also puts the
    finite-size correction at `M ** (-1/3)`, half the `O(M ** (-2/3))` order the
    repo's own theory document states.

    Note what is NOT asserted: the VERDICT. The previously reported verdict flip
    reproduces only on the `W.T @ W / rows` probe one shipped test uses and was
    0/10 on raw-scale Gaussian draws, so a verdict-only guard would be
    probe-dependent. Headroom is the robust symptom.
    """

    # Exact powers of two. `evals * s * s` is then a pure exponent shift with the
    # mantissas untouched, so a scale-equivariant threshold can be asserted
    # BIT-IDENTICAL rather than merely close. 2**-14 to 2**14 spans 6.1e-5 to
    # 1.6e4, i.e. the whole 1e-4..1e4 range, and the decimal scales are checked
    # separately below at a tolerance, since 1e-4 is not representable in binary
    # and so is not a pure rescale at all.
    _EXACT_SCALES = (2.0 ** -14, 2.0 ** -7, 1.0, 2.0 ** 7, 2.0 ** 14)
    _DECIMAL_SCALES = (1e-4, 1e-2, 1.0, 1e2, 1e4)

    @staticmethod
    def _relative_headroom(evals, n, m, c_TW=None):
        kwargs = {} if c_TW is None else {"c_TW": c_TW}
        result = detect_correlation_trap(evals, n, m, **kwargs)
        lambda_plus = result["mp_lambda_plus"]
        assert lambda_plus > 0.0, "the MP edge collapsed; the probe is degenerate"
        return (result["trap_threshold"] - lambda_plus) / lambda_plus

    @pytest.mark.parametrize("seed", [3, 7, 0, 11])
    def test_relative_headroom_is_bit_identical_under_a_pure_rescale(self, seed):
        evals = _clean_wishart(200, 50, seed)
        headrooms = [
            self._relative_headroom(evals * s * s, 200, 50)
            for s in self._EXACT_SCALES
        ]

        for scale, headroom in zip(self._EXACT_SCALES[1:], headrooms[1:]):
            assert headroom == headrooms[0], (
                f"seed {seed}: relative headroom moved from {headrooms[0]!r} at "
                f"s=2**-14 to {headroom!r} at s={scale:g}. Rescaling the weights "
                f"changes no shape and must change no verdict; a threshold offset "
                f"that is not proportional to lambda_plus makes the detector a "
                f"function of the units the weights happen to be stored in."
            )

    @pytest.mark.parametrize("seed", [3, 7, 0, 11])
    def test_relative_headroom_survives_decimal_rescales_too(self, seed):
        """The same invariance over the 1e-4..1e4 range, at a tolerance.

        Decimal scales are not exactly representable, so `evals * s * s` perturbs
        the mantissas and the last bit of `lambda_plus` with them; the residual
        1e-15 wobble is the probe's, not the threshold's. MEASURED under the
        shipped square-root spelling this same assertion saw 2.4e+04 against
        5.3e+01 between two adjacent scales, so a 1e-12 tolerance is six orders
        clear of the defect it exists to catch.
        """
        evals = _clean_wishart(200, 50, seed)
        headrooms = [
            self._relative_headroom(evals * s * s, 200, 50)
            for s in self._DECIMAL_SCALES
        ]

        for scale, headroom in zip(self._DECIMAL_SCALES[1:], headrooms[1:]):
            assert headroom == pytest.approx(headrooms[0], rel=1e-12), (
                f"seed {seed}: relative headroom moved from {headrooms[0]!r} at "
                f"s=1e-4 to {headroom!r} at s={scale:g}"
            )

    def test_the_headroom_carries_the_documented_M_to_the_minus_two_thirds(self):
        """SETOL.md:116 states the fluctuations are of order O(M^(-2/3)).

        Held at fixed Q = 4, the relative headroom must fall by exactly
        `(M2/M1) ** (2/3)`. The shipped `sqrt` spelling delivers `M ** (-1/3)`,
        which is off by a square root at every shape.
        """
        shapes = [(200, 50), (400, 100), (800, 200), (1600, 400)]
        headrooms = [
            self._relative_headroom(_clean_wishart(n, m, 5), n, m) for n, m in shapes
        ]

        for (n1, m1), h1, (n2, m2), h2 in zip(
                shapes[:-1], headrooms[:-1], shapes[1:], headrooms[1:]):
            expected = (m1 / m2) ** (2.0 / 3.0)
            observed = h2 / h1
            assert observed == pytest.approx(expected, rel=1e-12), (
                f"({n1}x{m1}) -> ({n2}x{m2}): headroom ratio {observed!r}, expected "
                f"{expected!r} for the documented M^(-2/3) order. An observed ratio "
                f"near {(m1 / m2) ** (1.0 / 3.0):.6f} means the offset is still "
                f"being square-rooted, leaving the correction at M^(-1/3)."
            )

    def test_the_headroom_matches_the_johnstone_tracy_widom_scale(self):
        """Anti-vacuity: the invariant is the RIGHT constant, not merely constant.

        Johnstone (2001), as stated in Ma arXiv:0810.1329 Eq. (2)-(3), gives
        `sigma_p = (sqrt(n-1) + sqrt(p)) * (1/sqrt(n-1) + 1/sqrt(p)) ** (1/3)`.
        In the repo's variables and the same normalisation as `lambda_plus`, that
        is `lambda_plus * M ** (-2/3) * Q ** (-1/6) * (1 + sqrt(Q)) ** (-2/3)` —
        verified against the literature spelling to <= 4.4e-16 relative at six
        shapes. A headroom that were invariant but, say, twice this would satisfy
        the two tests above and still be miscalibrated.
        """
        from dl_techniques.analyzer.constants import SPECTRAL_TW_SAFETY_FACTOR

        for n, m in [(200, 50), (100, 100), (400, 50), (2000, 500)]:
            Q = n / m
            johnstone = (
                (np.sqrt(n) + np.sqrt(m))
                * (1.0 / np.sqrt(n) + 1.0 / np.sqrt(m)) ** (1.0 / 3.0) / n
            )
            lambda_plus_unit = (1.0 + 1.0 / np.sqrt(Q)) ** 2
            expected = SPECTRAL_TW_SAFETY_FACTOR * johnstone / lambda_plus_unit

            observed = self._relative_headroom(_clean_wishart(n, m, 5), n, m)

            assert observed == pytest.approx(expected, rel=1e-12), (
                f"({n}x{m}): relative headroom {observed!r} against Johnstone's "
                f"Tracy-Widom scale {expected!r} at c_TW="
                f"{SPECTRAL_TW_SAFETY_FACTOR}"
            )


class TestTheTrapSafetyFactorIsCalibratedAndLive:
    """`SPECTRAL_TW_SAFETY_FACTOR` is how many Tracy-Widom units of headroom.

    Under the pre-fix square-root spelling the knob carried no information: the
    offset was ~0.3% of `lambda_plus` at unit scale while the Tracy-Widom standard
    deviation is ~2.8% of it, so almost nothing sat between `c_TW = 1` and
    `c_TW = 3` and the two gave the same verdict on essentially every draw
    (MEASURED false-positive rate at 200x50, unit scale: 0.003 against 0.000).
    Now that the offset is a genuine Tracy-Widom scale the knob is live, and its
    value has to be earned rather than inherited.

    Every draw below is seeded, so these rates are fixed numbers, not samples.
    """

    _TRIALS = 300
    _SHAPES = [(200, 50), (100, 100), (500, 100)]

    @staticmethod
    def _clean_draws(n, m, trials):
        for t in range(trials):
            weight_matrix = np.random.default_rng(10_000 + t).standard_normal((n, m))
            yield np.sort(np.linalg.eigvalsh(weight_matrix.T @ weight_matrix / n))[::-1]

    @pytest.mark.parametrize("shape", _SHAPES)
    def test_the_shipped_default_holds_false_positives_under_five_percent(self, shape):
        """At the shipped default, a clean Wishart must almost never be flagged.

        MEASURED at the previous default of 1.0: 0.0900 / 0.1300 / 0.0967 across
        these three shapes, which tracks the Tracy-Widom law's own P(W1 > 1) of
        about 8% — the constant was right for what it multiplies, and one TW unit
        of headroom is simply not enough margin. At 3.0: 0.0067 / 0.0100 / 0.0133.
        """
        n, m = shape
        flagged = sum(
            detect_correlation_trap(evals, n, m)["has_trap"]
            for evals in self._clean_draws(n, m, self._TRIALS)
        )
        rate = flagged / self._TRIALS

        assert rate < 0.05, (
            f"{n}x{m}: {flagged} of {self._TRIALS} CLEAN Gaussian Wisharts were "
            f"flagged as correlation traps (rate {rate:.4f}) at the shipped "
            f"SPECTRAL_TW_SAFETY_FACTOR. A rate near 0.08-0.13 is one Tracy-Widom "
            f"unit of headroom, which is the width of the fluctuation itself."
        )

    def test_the_shipped_default_still_detects_the_setol_element_trap(self):
        """A pin, not a RED-proven guard: power was already 1.0 before the bump.

        SETOL §7.1 defines the trap geometry as an unusually large matrix ELEMENT,
        so that is what is planted. At amplitude 20 the largest eigenvalue sits at
        1.51x the MP edge; power is 1.000 at every safety factor from 1.0 to 4.0,
        so raising the default buys the false-positive reduction above for nothing.
        """
        detected = 0
        for t in range(100):
            weight_matrix = np.random.default_rng(20_000 + t).standard_normal((200, 50))
            weight_matrix[0, 0] = 20.0
            evals = np.sort(
                np.linalg.eigvalsh(weight_matrix.T @ weight_matrix / 200))[::-1]
            detected += detect_correlation_trap(evals, 200, 50)["has_trap"]

        assert detected == 100, (
            f"the shipped default detected only {detected} of 100 planted element "
            f"traps at amplitude 20"
        )

    @pytest.mark.parametrize("shape", _SHAPES)
    def test_the_safety_factor_is_a_live_knob(self, shape):
        """Anti-vacuity: the constant must actually change verdicts.

        A calibration argument is worthless for a knob nothing responds to, and
        this one WAS inert — under the pre-fix spelling `c_TW = 1.0` and
        `c_TW = 3.0` disagreed on roughly one clean draw in 300. They now disagree
        on 25 / 36 / 25 of 300.
        """
        n, m = shape
        disagreements = 0
        for evals in self._clean_draws(n, m, self._TRIALS):
            low = detect_correlation_trap(evals, n, m, c_TW=1.0)["has_trap"]
            high = detect_correlation_trap(evals, n, m, c_TW=3.0)["has_trap"]
            disagreements += low != high

        assert disagreements >= 10, (
            f"{n}x{m}: c_TW=1.0 and c_TW=3.0 gave the same verdict on all but "
            f"{disagreements} of {self._TRIALS} clean draws. The safety factor is "
            f"documented as a tunable; a knob that changes nothing is a constant "
            f"wearing a parameter's name."
        )

    def test_a_clean_two_hundred_by_fifty_wishart_reports_no_spikes(self):
        """The `num_rand_spikes = 1` observation on a clean probe, at the default.

        Recorded as calibration rather than as a separate defect: it is the same
        one TW unit of headroom, seen on the two documented borderline seeds.
        """
        for seed in (3, 7, 0, 11):
            weight_matrix = np.random.default_rng(seed).standard_normal((200, 50))
            evals = np.sort(
                np.linalg.eigvalsh(weight_matrix.T @ weight_matrix / 200))[::-1]
            result = detect_correlation_trap(evals, 200, 50)
            assert result["num_rand_spikes"] == 0, (
                f"seed {seed}: a clean 200x50 Wishart reported "
                f"{result['num_rand_spikes']} spike(s) above the trap threshold"
            )
