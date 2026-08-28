"""Unit tests for ``train.common.stats``.

Previously ``tests/test_train/test_logic/test_multiseed_stats.py``; the module
under test moved to ``train.common.stats`` when the ``train.logic`` and
``train.rms_variants_train`` forks were consolidated.

Pinned contracts:
- mean_std uses sample std (ddof=1) and is NaN-tolerant.
- bootstrap_ci is deterministic given a fixed RNG.
- bootstrap_ci on all-identical input returns zero-width CI.
- paired_permutation_test on all-zero diffs returns p=1.0.
- paired_permutation_test is symmetric in its two-sided p-value.

Plan: ``plans/plan_2026-05-14_9c6387a3``  (D-002, D-004).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from train.common.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    format_mean_std,
    holm_bonferroni,
    mean_std,
    min_pairs_for_significance,
    min_reachable_p_signflip,
    paired_permutation_test,
)


# ---------------------------------------------------------------------------
# mean_std
# ---------------------------------------------------------------------------

class TestMeanStd:
    def test_known_values_ddof_1(self):
        mean, std = mean_std([1.0, 2.0, 3.0, 4.0, 5.0])
        assert mean == pytest.approx(3.0)
        # sample std of 1..5 with ddof=1 == sqrt(2.5)
        assert std == pytest.approx(math.sqrt(2.5))

    def test_all_same(self):
        mean, std = mean_std([7.0, 7.0, 7.0, 7.0, 7.0])
        assert mean == pytest.approx(7.0)
        assert std == 0.0

    def test_with_nan(self):
        # [1, nan, 3] -> mean=2.0, std (ddof=1) = sqrt(2)
        mean, std = mean_std([1.0, float("nan"), 3.0])
        assert mean == pytest.approx(2.0)
        assert std == pytest.approx(math.sqrt(2.0))

    def test_all_nan(self):
        mean, std = mean_std([float("nan"), float("nan")])
        assert math.isnan(mean)
        assert math.isnan(std)

    def test_empty(self):
        mean, std = mean_std([])
        assert math.isnan(mean)
        assert math.isnan(std)

    def test_single_finite_value(self):
        mean, std = mean_std([42.0])
        assert mean == 42.0
        assert std == 0.0  # not NaN — formatting depends on this contract

    def test_ddof_zero(self):
        # population std of 1..5 with ddof=0 == sqrt(2.0)
        _, std = mean_std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=0)
        assert std == pytest.approx(math.sqrt(2.0))


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_all_same_zero_width(self):
        rng = np.random.default_rng(0)
        lo, hi = bootstrap_ci([5.0, 5.0, 5.0, 5.0, 5.0], rng=rng)
        assert lo == 5.0 and hi == 5.0

    def test_single_value_zero_width(self):
        rng = np.random.default_rng(0)
        lo, hi = bootstrap_ci([3.14], rng=rng)
        assert lo == 3.14 and hi == 3.14

    def test_deterministic_given_rng(self):
        data = [0.1, 0.2, 0.3, 0.4, 0.5]
        a = bootstrap_ci(data, rng=np.random.default_rng(42), n_boot=500)
        b = bootstrap_ci(data, rng=np.random.default_rng(42), n_boot=500)
        assert a == b

    def test_different_seeds_differ(self):
        data = [0.1, 0.2, 0.3, 0.4, 0.5]
        a = bootstrap_ci(data, rng=np.random.default_rng(0), n_boot=500)
        b = bootstrap_ci(data, rng=np.random.default_rng(1), n_boot=500)
        assert a != b

    def test_ci_brackets_mean_on_gaussian(self):
        # 1000 samples from N(0, 1): 95% CI of the mean should bracket 0.
        rng = np.random.default_rng(123)
        data = rng.normal(0.0, 1.0, size=1000)
        lo, hi = bootstrap_ci(data, rng=np.random.default_rng(456), n_boot=2000)
        assert lo < 0.0 < hi

    def test_nan_dropped(self):
        rng = np.random.default_rng(0)
        # NaN-bearing input must not poison the CI; remaining values are
        # identical → zero-width CI.
        lo, hi = bootstrap_ci([5.0, float("nan"), 5.0, 5.0], rng=rng)
        assert lo == 5.0 and hi == 5.0

    def test_empty_input_nan(self):
        rng = np.random.default_rng(0)
        lo, hi = bootstrap_ci([], rng=rng)
        assert math.isnan(lo) and math.isnan(hi)

    def test_invalid_confidence_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            bootstrap_ci([1.0, 2.0], confidence=1.5, rng=rng)
        with pytest.raises(ValueError):
            bootstrap_ci([1.0, 2.0], confidence=0.0, rng=rng)

    def test_invalid_n_boot_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            bootstrap_ci([1.0, 2.0], n_boot=0, rng=rng)


# ---------------------------------------------------------------------------
# paired_permutation_test
# ---------------------------------------------------------------------------

class TestPairedPermutationTest:
    def test_all_zero_diffs_p_equals_one(self):
        rng = np.random.default_rng(0)
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 3.0]
        obs, p = paired_permutation_test(a, b, rng=rng)
        assert obs == 0.0
        assert p == 1.0

    def test_identical_inputs(self):
        rng = np.random.default_rng(0)
        x = [0.5, 0.7, 0.9, 0.1, 0.3]
        obs, p = paired_permutation_test(x, x, rng=rng)
        assert obs == 0.0
        assert p == 1.0

    def test_large_effect_small_p(self):
        # all-positive diff of 1.0 on n=5 → only 2^5 = 32 sign patterns; the
        # observed pattern is one of two extremes → p ~ 2/32 ≈ 0.0625.
        rng = np.random.default_rng(0)
        a = [1.0, 1.0, 1.0, 1.0, 1.0]
        b = [0.0, 0.0, 0.0, 0.0, 0.0]
        obs, p = paired_permutation_test(a, b, n_perm=10000, rng=rng)
        assert obs == pytest.approx(1.0)
        assert p < 0.1  # exact lower bound for n=5 paired permutation

    def test_deterministic_given_rng(self):
        a = [0.1, 0.2, 0.3, 0.4, 0.5]
        b = [0.0, 0.1, 0.2, 0.3, 0.4]
        r1 = paired_permutation_test(a, b, rng=np.random.default_rng(42), n_perm=5000)
        r2 = paired_permutation_test(a, b, rng=np.random.default_rng(42), n_perm=5000)
        assert r1 == r2

    def test_two_sided_symmetric_p(self):
        # Swapping (a, b) flips observed_diff sign but two-sided p-value
        # uses |diff| and so must match given the same RNG.
        a = [0.5, 0.6, 0.7, 0.8, 0.9]
        b = [0.1, 0.2, 0.3, 0.4, 0.5]
        obs_ab, p_ab = paired_permutation_test(
            a, b, rng=np.random.default_rng(7), n_perm=5000
        )
        obs_ba, p_ba = paired_permutation_test(
            b, a, rng=np.random.default_rng(7), n_perm=5000
        )
        assert obs_ab == pytest.approx(-obs_ba)
        assert p_ab == pytest.approx(p_ba)

    def test_nan_pairs_dropped(self):
        # NaN in either side drops the pair entirely.
        rng = np.random.default_rng(0)
        a = [1.0, float("nan"), 1.0]
        b = [0.0, 0.0, 0.0]
        obs, p = paired_permutation_test(a, b, rng=rng, n_perm=1000)
        assert obs == pytest.approx(1.0)
        # n=2 retained → both diffs positive → only 2/4 sign patterns hit
        # |mean| >= 1.0; with add-one correction p = (n_extreme+1)/(n_perm+1).
        assert 0.0 < p <= 1.0

    def test_shape_mismatch_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            paired_permutation_test([1, 2, 3], [1, 2], rng=rng)

    def test_invalid_n_perm_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            paired_permutation_test([1.0], [0.0], n_perm=0, rng=rng)

    def test_empty_after_nan_drop_returns_nan(self):
        rng = np.random.default_rng(0)
        a = [float("nan"), float("nan")]
        b = [0.0, 0.0]
        obs, p = paired_permutation_test(a, b, rng=rng)
        assert math.isnan(obs) and math.isnan(p)


# ---------------------------------------------------------------------------
# format_mean_std
# ---------------------------------------------------------------------------

class TestFormatMeanStd:
    def test_basic_4_decimals(self):
        assert format_mean_std(0.7006, 0.0123) == "0.7006 ± 0.0123"

    def test_custom_decimals(self):
        assert format_mean_std(0.7006, 0.0123, decimals=2) == "0.70 ± 0.01"

    def test_zero_std(self):
        assert format_mean_std(1.0, 0.0) == "1.0000 ± 0.0000"

    def test_nan_inputs(self):
        assert format_mean_std(float("nan"), 0.1) == "nan ± nan"
        assert format_mean_std(0.1, float("nan")) == "nan ± nan"

    def test_negative_decimals_raises(self):
        with pytest.raises(ValueError):
            format_mean_std(1.0, 0.1, decimals=-1)


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------

class TestHolmBonferroni:
    """Step-down FWER correction, checked against hand-computed values."""

    def test_a_hand_computed_family(self):
        # m=3. Sorted p = [0.01, 0.02, 0.03].
        #   raw*(m-i): 3*0.01=0.03, 2*0.02=0.04, 1*0.03=0.03
        #   running max:      0.03,       0.04,       0.04
        rejected, adjusted = holm_bonferroni([0.01, 0.02, 0.03], alpha=0.05)
        np.testing.assert_allclose(adjusted, [0.03, 0.04, 0.04], atol=1e-12)
        assert rejected.tolist() == [True, True, True]

    def test_identity_at_family_size_one(self):
        _, adjusted = holm_bonferroni([0.04])
        np.testing.assert_allclose(adjusted, [0.04], atol=1e-12)

    def test_output_is_in_input_order(self):
        _, adjusted = holm_bonferroni([0.03, 0.01, 0.02])
        # the 0.01 entry is the most significant wherever it sits
        assert adjusted[1] == min(adjusted)

    def test_adjusted_never_below_raw(self):
        raw = [0.001, 0.01, 0.04, 0.2]
        _, adjusted = holm_bonferroni(raw)
        assert np.all(adjusted >= np.asarray(raw) - 1e-12)

    def test_adjusted_is_clipped_to_one(self):
        _, adjusted = holm_bonferroni([0.9, 0.95, 0.99])
        assert np.all(adjusted <= 1.0)

    def test_nan_entries_do_not_inflate_the_family(self):
        """A comparison that could not be run must not penalise the others."""
        _, alone = holm_bonferroni([0.01])
        _, with_nans = holm_bonferroni([0.01, np.nan, np.nan])
        np.testing.assert_allclose(with_nans[0], alone[0], atol=1e-12)
        assert np.isnan(with_nans[1]) and np.isnan(with_nans[2])

    def test_rejects_a_bad_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            holm_bonferroni([0.01], alpha=0.0)

    def test_rejects_an_out_of_range_p_value(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            holm_bonferroni([0.5, 1.5])


class TestBenjaminiHochberg:
    """Step-up FDR correction."""

    def test_the_textbook_boundary_case(self):
        # p = k*alpha/m for every k, so every entry sits exactly on the line.
        rejected, adjusted = benjamini_hochberg(
            [0.01, 0.02, 0.03, 0.04, 0.05], alpha=0.05
        )
        np.testing.assert_allclose(adjusted, [0.05] * 5, atol=1e-12)
        assert rejected.tolist() == [True] * 5

    def test_identity_at_family_size_one(self):
        _, adjusted = benjamini_hochberg([0.04])
        np.testing.assert_allclose(adjusted, [0.04], atol=1e-12)

    def test_is_never_more_conservative_than_holm(self):
        """BH trades FWER for power; it must never adjust higher than Holm."""
        rng = np.random.default_rng(0)
        for _ in range(2000):
            p = rng.random(rng.integers(2, 12))
            _, bh = benjamini_hochberg(p)
            _, holm = holm_bonferroni(p)
            assert np.all(bh <= holm + 1e-12)

    def test_adjusted_is_monotone_in_the_raw_ordering(self):
        rng = np.random.default_rng(1)
        for _ in range(500):
            p = np.sort(rng.random(8))
            _, adjusted = benjamini_hochberg(p)
            assert np.all(np.diff(adjusted) >= -1e-12)

    def test_nan_entries_do_not_inflate_the_family(self):
        _, alone = benjamini_hochberg([0.02])
        _, with_nans = benjamini_hochberg([0.02, np.nan])
        np.testing.assert_allclose(with_nans[0], alone[0], atol=1e-12)


class TestTheSeedFloorIsAPropertyOfTheTest:
    """How many seeds a corrected sign-flip test needs before it can reject.

    A paired sign-flip test over n pairs enumerates 2**n sign vectors, so the
    smallest two-sided p it can produce is 2/2**n -- regardless of effect size.
    Any correction tightens the bar to alpha/m, so correcting a bigger family
    costs seeds. These are the numbers that set the study's GPU budget.
    """

    def test_the_analytic_bound(self):
        assert min_reachable_p_signflip(6) == pytest.approx(2 ** -5)
        assert min_reachable_p_signflip(1) == 1.0
        assert min_reachable_p_signflip(0) == 1.0

    @pytest.mark.parametrize("n", [6, 8, 10])
    def test_the_bound_matches_the_monte_carlo_estimator(self, n):
        """Pins the analytic bound to this module's own implementation.

        ``paired_permutation_test`` SAMPLES sign vectors with replacement
        rather than enumerating them, so its p-value is a noisy estimate of
        the exact ``2/2**n`` and fluctuates on BOTH sides of it -- measured at
        n=6, 20000 draws, seed 7: 0.029099 against an exact 0.031250, which is
        1.8 binomial standard errors low. An assertion that Monte Carlo
        approaches the bound from above is therefore wrong; the tolerance here
        is derived from the estimator's own standard error, not pasted.
        """
        rng = np.random.default_rng(7)
        n_perm = 20000
        a = [10.0 + i * 0.01 for i in range(n)]
        b = [1.0 + i * 0.01 for i in range(n)]
        _, p = paired_permutation_test(a, b, n_perm=n_perm, rng=rng)

        bound = min_reachable_p_signflip(n)
        std_err = math.sqrt(bound * (1.0 - bound) / n_perm)
        assert abs(p - bound) <= 5.0 * std_err + 1.0 / (n_perm + 1)

    @pytest.mark.parametrize(
        "family_size,expected", [(1, 6), (3, 7), (18, 10), (21, 10), (63, 12)]
    )
    def test_the_seed_floor_table(self, family_size, expected):
        assert min_pairs_for_significance(family_size) == expected

    @pytest.mark.parametrize(
        "family_size,expected", [(1, 6), (3, 7), (18, 10), (63, 12)]
    )
    def test_the_floor_is_reachable_and_the_step_below_is_not(
        self, family_size, expected
    ):
        """Measured, not assumed: run the real test at n and at n-1."""
        alpha = 0.05
        rng = np.random.default_rng(11)

        def smallest_p(n):
            a = [10.0 + i * 0.01 for i in range(n)]
            b = [1.0 + i * 0.01 for i in range(n)]
            return paired_permutation_test(a, b, n_perm=20000, rng=rng)[1]

        assert smallest_p(expected) <= alpha / family_size
        assert smallest_p(expected - 1) > alpha / family_size

    def test_rejects_a_bad_family_size(self):
        with pytest.raises(ValueError, match="family_size"):
            min_pairs_for_significance(0)
