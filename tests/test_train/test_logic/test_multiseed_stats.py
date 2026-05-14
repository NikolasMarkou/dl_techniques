"""Unit tests for ``train.logic.multiseed_stats``.

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

from train.logic.multiseed_stats import (
    bootstrap_ci,
    format_mean_std,
    mean_std,
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
