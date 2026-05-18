"""Unit tests for ``train.rms_variants_train.stats`` (Refinement E, step 1).

Covers oracle cases, NaN tolerance, zero-variance degenerate paths, two-sided
symmetry of the permutation test, determinism under fixed rng, parameter
validation, and equivalence with the ``train.logic.multiseed_stats`` precedent.

Plan: ``plans/plan_2026-05-18_6776f8ba`` (step 1 gate: 30+ PASS).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from train.rms_variants_train import stats as st


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------

class TestSurface:
    def test_public_callables(self) -> None:
        assert callable(st.mean_std)
        assert callable(st.bootstrap_ci)
        assert callable(st.paired_permutation_test)
        assert callable(st.format_mean_std)

    def test_all_exports_present(self) -> None:
        expected = {"mean_std", "bootstrap_ci", "paired_permutation_test", "format_mean_std"}
        assert expected.issubset(set(st.__all__))


# ---------------------------------------------------------------------------
# mean_std
# ---------------------------------------------------------------------------

class TestMeanStd:
    def test_oracle_values(self) -> None:
        m, s = st.mean_std([1.0, 2.0, 3.0, 4.0, 5.0])
        assert m == pytest.approx(3.0)
        assert s == pytest.approx(math.sqrt(2.5), rel=1e-9)

    def test_ddof_zero(self) -> None:
        # Population std of [1,2,3,4,5] is sqrt(2.0)
        m, s = st.mean_std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=0)
        assert m == pytest.approx(3.0)
        assert s == pytest.approx(math.sqrt(2.0), rel=1e-9)

    def test_empty_input(self) -> None:
        m, s = st.mean_std([])
        assert math.isnan(m)
        assert math.isnan(s)

    def test_all_nan(self) -> None:
        m, s = st.mean_std([float("nan"), float("nan")])
        assert math.isnan(m)
        assert math.isnan(s)

    def test_single_finite_value(self) -> None:
        m, s = st.mean_std([7.5])
        assert m == pytest.approx(7.5)
        assert s == 0.0

    def test_nan_skipping(self) -> None:
        m, s = st.mean_std([1.0, 2.0, float("nan"), 3.0])
        assert m == pytest.approx(2.0)
        assert s == pytest.approx(1.0, rel=1e-9)  # ddof=1 over [1,2,3]

    def test_accepts_numpy_array(self) -> None:
        m, s = st.mean_std(np.array([10.0, 20.0, 30.0]))
        assert m == pytest.approx(20.0)
        assert s == pytest.approx(10.0, rel=1e-9)

    def test_accepts_tuple(self) -> None:
        m, s = st.mean_std((1.0, 1.0, 1.0))
        assert m == pytest.approx(1.0)
        assert s == 0.0


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_deterministic_same_seed(self) -> None:
        rng1 = np.random.default_rng(20260518)
        rng2 = np.random.default_rng(20260518)
        v = [0.70, 0.71, 0.72, 0.73, 0.74]
        lo1, hi1 = st.bootstrap_ci(v, rng=rng1, n_boot=500)
        lo2, hi2 = st.bootstrap_ci(v, rng=rng2, n_boot=500)
        assert lo1 == lo2
        assert hi1 == hi2

    def test_ci_brackets_data(self) -> None:
        rng = np.random.default_rng(0)
        lo, hi = st.bootstrap_ci([0.7, 0.71, 0.72, 0.73, 0.74], rng=rng, n_boot=2000)
        assert 0.70 <= lo <= hi <= 0.74

    def test_single_value(self) -> None:
        rng = np.random.default_rng(0)
        lo, hi = st.bootstrap_ci([0.5], rng=rng, n_boot=10)
        assert lo == pytest.approx(0.5)
        assert hi == pytest.approx(0.5)

    def test_all_identical(self) -> None:
        rng = np.random.default_rng(0)
        lo, hi = st.bootstrap_ci([0.4, 0.4, 0.4, 0.4], rng=rng, n_boot=10)
        assert lo == pytest.approx(0.4)
        assert hi == pytest.approx(0.4)

    def test_all_nan(self) -> None:
        rng = np.random.default_rng(0)
        lo, hi = st.bootstrap_ci([float("nan"), float("nan")], rng=rng, n_boot=10)
        assert math.isnan(lo) and math.isnan(hi)

    def test_empty(self) -> None:
        rng = np.random.default_rng(0)
        lo, hi = st.bootstrap_ci([], rng=rng, n_boot=10)
        assert math.isnan(lo) and math.isnan(hi)

    def test_nan_dropped_before_resample(self) -> None:
        rng = np.random.default_rng(0)
        # All finite values identical; NaN must not contribute to width
        lo, hi = st.bootstrap_ci([0.5, float("nan"), 0.5, 0.5], rng=rng, n_boot=50)
        assert lo == pytest.approx(0.5)
        assert hi == pytest.approx(0.5)

    def test_confidence_out_of_range_low(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            st.bootstrap_ci([0.0, 1.0], confidence=0.0, rng=rng, n_boot=10)

    def test_confidence_out_of_range_high(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            st.bootstrap_ci([0.0, 1.0], confidence=1.0, rng=rng, n_boot=10)

    def test_n_boot_zero_raises(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            st.bootstrap_ci([0.0, 1.0], rng=rng, n_boot=0)

    def test_wider_ci_with_lower_confidence(self) -> None:
        # 99% CI strictly contains 50% CI for non-degenerate data.
        v = np.random.default_rng(1).normal(0.0, 1.0, size=50).tolist()
        rng99 = np.random.default_rng(7)
        rng50 = np.random.default_rng(7)
        lo99, hi99 = st.bootstrap_ci(v, confidence=0.99, n_boot=1000, rng=rng99)
        lo50, hi50 = st.bootstrap_ci(v, confidence=0.50, n_boot=1000, rng=rng50)
        assert lo99 <= lo50
        assert hi99 >= hi50


# ---------------------------------------------------------------------------
# paired_permutation_test
# ---------------------------------------------------------------------------

class TestPairedPermutation:
    def test_oracle_positive_effect(self) -> None:
        rng = np.random.default_rng(0)
        a = np.array([0.80, 0.81, 0.82, 0.83, 0.84])
        b = np.array([0.70, 0.71, 0.72, 0.73, 0.74])
        diff, p = st.paired_permutation_test(a, b, rng=rng, n_perm=2000)
        assert diff == pytest.approx(0.10, rel=1e-9)
        assert 0.0 <= p <= 1.0
        # With +0.1 mean diff and matched scale, p should be small.
        assert p < 0.2

    def test_zero_difference_returns_p_one(self) -> None:
        rng = np.random.default_rng(0)
        a = np.array([0.5, 0.6, 0.7])
        b = np.array([0.5, 0.6, 0.7])
        diff, p = st.paired_permutation_test(a, b, rng=rng, n_perm=500)
        assert diff == 0.0
        assert p == 1.0

    def test_no_finite_pairs(self) -> None:
        rng = np.random.default_rng(0)
        a = np.array([float("nan"), float("nan")])
        b = np.array([1.0, 2.0])
        diff, p = st.paired_permutation_test(a, b, rng=rng, n_perm=10)
        assert math.isnan(diff) and math.isnan(p)

    def test_partial_nan_dropped(self) -> None:
        rng = np.random.default_rng(0)
        a = np.array([1.0, 2.0, float("nan"), 4.0])
        b = np.array([0.5, 1.5, 2.5, 3.5])
        diff, _ = st.paired_permutation_test(a, b, rng=rng, n_perm=100)
        # Retained pairs: (1,0.5),(2,1.5),(4,3.5) -> diffs 0.5,0.5,0.5 -> mean 0.5
        assert diff == pytest.approx(0.5, rel=1e-9)

    def test_two_sided_symmetry(self) -> None:
        # Swapping a and b must flip sign of diff and leave p unchanged.
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        diff_ab, p_ab = st.paired_permutation_test(a, b, rng=rng1, n_perm=500)
        diff_ba, p_ba = st.paired_permutation_test(b, a, rng=rng2, n_perm=500)
        assert diff_ab == pytest.approx(-diff_ba, rel=1e-12, abs=1e-12)
        assert p_ab == pytest.approx(p_ba, rel=1e-12, abs=1e-12)

    def test_p_value_lower_bound_phipson_smyth(self) -> None:
        # With add-one correction p >= 1 / (n_perm + 1)
        rng = np.random.default_rng(0)
        a = np.arange(20, dtype=float) + 100.0
        b = np.arange(20, dtype=float)  # huge separation
        _, p = st.paired_permutation_test(a, b, rng=rng, n_perm=1000)
        assert p >= 1.0 / (1000 + 1) - 1e-12
        assert p <= 1.0

    def test_p_value_in_unit_interval(self) -> None:
        rng = np.random.default_rng(0)
        a = np.random.default_rng(1).normal(0, 1, 30)
        b = np.random.default_rng(2).normal(0, 1, 30)
        _, p = st.paired_permutation_test(a, b, rng=rng, n_perm=500)
        assert 0.0 <= p <= 1.0

    def test_shape_mismatch_raises(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            st.paired_permutation_test([1.0, 2.0], [1.0, 2.0, 3.0], rng=rng, n_perm=10)

    def test_n_perm_zero_raises(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            st.paired_permutation_test([1.0], [2.0], rng=rng, n_perm=0)

    def test_deterministic_under_seed(self) -> None:
        a = [1.0, 2.0, 3.0, 4.0]
        b = [1.5, 1.8, 3.2, 3.9]
        diff1, p1 = st.paired_permutation_test(
            a, b, rng=np.random.default_rng(99), n_perm=400
        )
        diff2, p2 = st.paired_permutation_test(
            a, b, rng=np.random.default_rng(99), n_perm=400
        )
        assert diff1 == diff2
        assert p1 == p2


# ---------------------------------------------------------------------------
# format_mean_std
# ---------------------------------------------------------------------------

class TestFormatMeanStd:
    def test_basic(self) -> None:
        s = st.format_mean_std(0.7006, 0.0123)
        assert s == "0.7006 ± 0.0123"

    def test_decimals_kwarg(self) -> None:
        s = st.format_mean_std(0.7006, 0.0123, decimals=2)
        assert s == "0.70 ± 0.01"

    def test_zero_decimals(self) -> None:
        s = st.format_mean_std(3.0, 1.0, decimals=0)
        assert s == "3 ± 1"

    def test_negative_decimals_raises(self) -> None:
        with pytest.raises(ValueError):
            st.format_mean_std(1.0, 0.5, decimals=-1)

    def test_nan_mean(self) -> None:
        s = st.format_mean_std(float("nan"), 0.5)
        assert s == "nan ± nan"

    def test_nan_std(self) -> None:
        s = st.format_mean_std(1.0, float("nan"))
        assert s == "nan ± nan"


# ---------------------------------------------------------------------------
# Equivalence with multiseed_stats precedent
# ---------------------------------------------------------------------------

class TestEquivalenceWithPrecedent:
    """The new stats module is API- and behaviour-equivalent to the precedent.

    Plan step 1 explicitly references ``train.logic.multiseed_stats`` as the
    behavioural reference. These tests pin equivalence so future divergence
    is caught immediately.
    """

    def test_mean_std_matches_precedent(self) -> None:
        from train.logic import multiseed_stats as ref
        v = [0.1, 0.2, 0.3, float("nan"), 0.5]
        assert st.mean_std(v) == ref.mean_std(v)

    def test_bootstrap_ci_matches_precedent(self) -> None:
        from train.logic import multiseed_stats as ref
        v = [0.7, 0.71, 0.72, 0.73, 0.74]
        rng1 = np.random.default_rng(2024)
        rng2 = np.random.default_rng(2024)
        a = st.bootstrap_ci(v, rng=rng1, n_boot=300)
        b = ref.bootstrap_ci(v, rng=rng2, n_boot=300)
        assert a == b

    def test_paired_permutation_matches_precedent(self) -> None:
        from train.logic import multiseed_stats as ref
        a = [1.0, 2.0, 3.0, 4.0]
        b = [1.1, 1.9, 3.2, 3.8]
        x = st.paired_permutation_test(a, b, rng=np.random.default_rng(7), n_perm=400)
        y = ref.paired_permutation_test(a, b, rng=np.random.default_rng(7), n_perm=400)
        assert x == y

    def test_format_mean_std_matches_precedent(self) -> None:
        from train.logic import multiseed_stats as ref
        assert st.format_mean_std(0.5, 0.1) == ref.format_mean_std(0.5, 0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-vvv"])
