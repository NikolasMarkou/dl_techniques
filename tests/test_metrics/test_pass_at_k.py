"""Tests for ``dl_techniques.metrics.pass_at_k``.

The module's whole claim is numerical, so the central guard is an ORACLE: every
estimator value is compared against an exact Python-``math.comb`` evaluation over
randomly generated outcome matrices, for every valid ``k``. The ``gammaln`` path is a
numerical optimisation of a closed form, and a closed form that is merely *nearly*
right is the kind of thing that passes a hand-picked example and fails on real data.

Guards, and what each one is for:

* T1 -- the unbiased estimator equals the exact combinatorial oracle (the anti-vacuity
  arm is T2, which is the same computation by a different route).
* T2 -- the two degenerate regions are EXACT: fully-solved rows give 1.0, never-solved
  rows give 0.0. This is where an off-by-one in the ``C(n - c, k) == 0`` test lived.
* T3 -- ``pass_at_k_curve`` agrees with a loop of ``pass_at_k`` (the vectorisation is
  an optimisation, not a second definition).
* T4 -- monotonicity in ``k`` and the ``unbiased <= plug_in`` ordering.
* T5 -- default ``k`` ladder shape, including the non-power-of-two ``n_samples``.
* T6 -- the input contract: shape, k range, estimator name, finite scores.
* T7 -- partial credit: a fractional reward counts as solved, and thresholding
  upstream reproduces the binarised result exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dl_techniques.metrics.pass_at_k import (
    PASS_AT_K_ESTIMATORS,
    pass_at_k,
    pass_at_k_curve,
    per_problem_pass_at_k,
    solved_counts,
)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _exact_unbiased(n: int, c: int, k: int) -> float:
    """``1 - C(n - c, k) / C(n, k)`` in exact Python integer arithmetic.

    The oracle. Returns exactly 1.0 when ``n - c < k`` (every ``k``-subset must
    contain a correct sample) and otherwise the exact rational as a float.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _random_outcomes(rng: np.random.Generator, n_samples: int, n_problems: int):
    """A random outcome matrix with a random per-problem solve probability."""
    rates = rng.random(n_problems)
    return (rng.random((n_problems, n_samples)) < rates[:, None]).astype(float)


# ---------------------------------------------------------------------
# T1 -- the unbiased estimator is EXACT
# ---------------------------------------------------------------------


class TestTheUnbiasedEstimatorIsExact:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_the_combinatorial_oracle(self, seed: int) -> None:
        """Every (n, c, k) combination matches ``math.comb`` to float64 round-off."""
        rng = np.random.default_rng(seed)
        worst = 0.0
        for _ in range(60):
            n = int(rng.integers(1, 14))
            n_problems = int(rng.integers(1, 9))
            outcomes = _random_outcomes(rng, n, n_problems)
            counts = solved_counts(outcomes)
            for k in range(1, n + 1):
                got = per_problem_pass_at_k(outcomes, k)
                expected = np.array(
                    [_exact_unbiased(n, int(c), k) for c in counts]
                )
                worst = max(worst, float(np.max(np.abs(got - expected))))
        # Float64 round-off on a ratio of gamma functions, not a modelling gap.
        assert worst < 1e-12, f"worst deviation from the exact oracle: {worst}"

    def test_k_equals_1_reproduces_the_mean_success_rate(self) -> None:
        """At ``k = 1`` the unbiased estimator degenerates to the plain mean rate."""
        rng = np.random.default_rng(7)
        outcomes = _random_outcomes(rng, 12, 40)
        assert pass_at_k(outcomes, k=1) == pytest.approx(outcomes.mean(), abs=1e-12)

    def test_a_single_solved_sample_is_not_credited_to_every_k(self) -> None:
        """The headline reason the unbiased estimator is the default.

        One correct sample out of eight is ``pass@1 = 0.125`` but ``pass@8 = 1.0``.
        A pipeline that scored a checkpoint pool by the plug-in number would score
        this row as full coverage; the unbiased number still separates it from a row
        that was genuinely solved eight different ways.
        """
        outcomes = np.zeros((1, 8))
        outcomes[0, 0] = 1.0
        assert pass_at_k(outcomes, k=1) == pytest.approx(0.125)
        assert pass_at_k(outcomes, k=8) == pytest.approx(1.0)
        assert pass_at_k(outcomes, k=8, estimator="plug_in") == pytest.approx(1.0)
        # At k=4 the two estimators genuinely differ: the plug-in says 1.0, the
        # unbiased says 1 - C(7,4)/C(8,4) = 1 - 35/70 = 0.5.
        assert pass_at_k(outcomes, k=4) == pytest.approx(0.5)
        assert pass_at_k(outcomes, k=4, estimator="plug_in") == pytest.approx(1.0)


# ---------------------------------------------------------------------
# T2 -- the degenerate regions are EXACT
# ---------------------------------------------------------------------


class TestTheDegenerateRegionsAreExact:
    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16])
    def test_fully_solved_is_exactly_one(self, n: int) -> None:
        """``c == n`` gives 1.0 at every ``k``, with no NaN from ``gammaln``.

        This is the row that dominates a real eval matrix, and the boundary it
        guards is ``n - c < k`` and NOT ``n - c <= k``: at ``n - c == k`` exactly,
        ``C(n - c, k) = 1`` and the ratio is ``1 / C(n, k) > 0``.
        """
        outcomes = np.ones((3, n))
        for k in range(1, n + 1):
            values = per_problem_pass_at_k(outcomes, k)
            assert np.all(np.isfinite(values)), f"NaN at n={n}, k={k}"
            assert np.allclose(values, 1.0), f"not 1.0 at n={n}, k={k}"

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16])
    def test_never_solved_is_exactly_zero(self, n: int) -> None:
        """``c == 0`` gives 0.0 at every ``k``, including ``k == n``.

        ``k == n`` is the off-by-one: with the boundary at ``<= 0`` the never-solved
        row was reported as ``1.0``, i.e. a model that solved nothing scored as
        perfect coverage.
        """
        outcomes = np.zeros((3, n))
        for k in range(1, n + 1):
            values = per_problem_pass_at_k(outcomes, k)
            assert np.allclose(values, 0.0), f"not 0.0 at n={n}, k={k}"

    def test_the_n_minus_c_equals_k_boundary(self) -> None:
        """``n - c == k`` is NOT degenerate: every k-subset is all-wrong.

        One solved sample among five, drawing four: ``1 - C(4,4)/C(5,4) = 1 - 1/5``.
        """
        outcomes = np.zeros((1, 5))
        outcomes[0, 0] = 1.0
        assert per_problem_pass_at_k(outcomes, 4)[0] == pytest.approx(0.8)

    def test_solved_counts_are_integers(self) -> None:
        counts = solved_counts(np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))
        assert counts.dtype == np.int64
        assert counts.tolist() == [2, 0]


# ---------------------------------------------------------------------
# T3 -- the vectorised curve is the same function
# ---------------------------------------------------------------------


class TestTheCurveAgreesWithTheLoop:
    @pytest.mark.parametrize("seed", [11, 12])
    def test_curve_equals_repeated_calls(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        for _ in range(25):
            n = int(rng.integers(1, 20))
            outcomes = _random_outcomes(rng, n, int(rng.integers(1, 10)))
            ks = list(range(1, n + 1))
            curve = pass_at_k_curve(outcomes, ks)
            assert sorted(curve) == ks
            for k in ks:
                assert curve[k] == pytest.approx(pass_at_k(outcomes, k), abs=1e-12)

    def test_curve_agrees_for_the_plug_in_estimator_too(self) -> None:
        rng = np.random.default_rng(3)
        outcomes = _random_outcomes(rng, 10, 12)
        curve = pass_at_k_curve(outcomes, estimator="plug_in")
        for k in curve:
            assert curve[k] == pytest.approx(
                pass_at_k(outcomes, k, estimator="plug_in"), abs=1e-12
            )

    def test_duplicate_and_unsorted_ks_collapse(self) -> None:
        outcomes = np.array([[1, 0, 1, 0]], dtype=float)
        curve = pass_at_k_curve(outcomes, ks=[3, 1, 3, 2, 1])
        assert list(curve) == [1, 2, 3]


# ---------------------------------------------------------------------
# T4 -- the orderings that make the numbers readable
# ---------------------------------------------------------------------


class TestTheOrderings:
    @pytest.mark.parametrize("seed", range(6))
    def test_monotone_nondecreasing_in_k(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n = int(rng.integers(2, 25))
        outcomes = _random_outcomes(rng, n, 30)
        curve = pass_at_k_curve(outcomes)
        values = [curve[k] for k in sorted(curve)]
        assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))

    @pytest.mark.parametrize("seed", range(6))
    def test_unbiased_never_exceeds_the_plug_in(self, seed: int) -> None:
        """The plug-in is an upper bound for ``k < n``; this is why quoting both is
        a false comparison rather than a belt-and-braces one."""
        rng = np.random.default_rng(seed)
        n = int(rng.integers(2, 16))
        outcomes = _random_outcomes(rng, n, 25)
        for k in range(1, n + 1):
            assert pass_at_k(outcomes, k) <= pass_at_k(
                outcomes, k, estimator="plug_in"
            ) + 1e-12

    def test_values_stay_in_the_unit_interval(self) -> None:
        rng = np.random.default_rng(99)
        outcomes = _random_outcomes(rng, 9, 50)
        for estimator in PASS_AT_K_ESTIMATORS:
            for k in range(1, 10):
                assert 0.0 <= pass_at_k(outcomes, k, estimator=estimator) <= 1.0

    def test_all_solved_saturates_at_one(self) -> None:
        outcomes = np.ones((7, 6))
        curve = pass_at_k_curve(outcomes)
        assert all(value == pytest.approx(1.0) for value in curve.values())


# ---------------------------------------------------------------------
# T5 -- the default ladder
# ---------------------------------------------------------------------


class TestTheDefaultLadder:
    @pytest.mark.parametrize(
        "n_samples,expected",
        [
            (1, [1]),
            (2, [1, 2]),
            (3, [1, 2, 3]),
            (4, [1, 2, 4]),
            (5, [1, 2, 4, 5]),
            (8, [1, 2, 4, 8]),
            (16, [1, 2, 4, 8, 16]),
            (100, [1, 2, 4, 8, 16, 32, 64, 100]),
        ],
    )
    def test_powers_of_two_plus_n_samples(self, n_samples, expected) -> None:
        outcomes = np.ones((2, n_samples))
        assert list(pass_at_k_curve(outcomes)) == expected

    def test_n_samples_is_always_present_even_when_not_a_power_of_two(self) -> None:
        """The headline breadth number needs ``k = n``; a powers-of-two ladder alone
        would stop at 64 for a 100-sample pool and silently report the wrong
        ``pass@k``."""
        curve = pass_at_k_curve(np.ones((2, 100)))
        assert max(curve) == 100


# ---------------------------------------------------------------------
# T6 -- the input contract
# ---------------------------------------------------------------------


class TestTheInputContract:
    def test_one_dimensional_input_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be 2-D"):
            pass_at_k(np.array([1.0, 0.0, 1.0]), k=1)

    def test_three_dimensional_input_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be 2-D"):
            pass_at_k(np.zeros((2, 3, 4)), k=1)

    @pytest.mark.parametrize("shape", [(0, 4), (3, 0), (0, 0)])
    def test_empty_axes_are_refused(self, shape) -> None:
        """An empty ``n_samples`` makes every pass@k undefined, not zero."""
        with pytest.raises(ValueError, match="non-zero extent"):
            pass_at_k(np.zeros(shape), k=1)

    @pytest.mark.parametrize("k", [0, -1, 5, 100])
    def test_out_of_range_k_is_refused(self, k: int) -> None:
        with pytest.raises(ValueError, match="k must be"):
            pass_at_k(np.ones((2, 4)), k=k)

    def test_non_integer_k_is_refused(self) -> None:
        with pytest.raises(ValueError, match="k must be an integer"):
            pass_at_k(np.ones((2, 4)), k=2.0)

    def test_bool_k_is_refused(self) -> None:
        """``True`` is an ``int`` in Python; accepting it would silently mean k=1."""
        with pytest.raises(ValueError, match="k must be an integer"):
            pass_at_k(np.ones((2, 4)), k=True)

    def test_unknown_estimator_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown estimator"):
            pass_at_k(np.ones((2, 4)), k=1, estimator="biased")

    def test_estimator_name_is_case_and_whitespace_tolerant(self) -> None:
        outcomes = np.ones((2, 4))
        assert pass_at_k(outcomes, k=1, estimator="  UNBIASED ") == pytest.approx(1.0)

    def test_empty_ks_is_refused(self) -> None:
        with pytest.raises(ValueError, match="ks must not be empty"):
            pass_at_k_curve(np.ones((2, 4)), ks=[])

    def test_k_above_n_samples_in_the_ladder_is_refused_not_clamped(self) -> None:
        with pytest.raises(ValueError, match="k must be <= n_samples"):
            pass_at_k_curve(np.ones((2, 4)), ks=[1, 8])


# ---------------------------------------------------------------------
# T7 -- partial credit
# ---------------------------------------------------------------------


class TestPartialCredit:
    def test_a_fractional_reward_counts_as_solved(self) -> None:
        """``> 0``, not ``>= 1``: pass@k asks whether a usable answer EXISTS, and
        whether ``0.4`` is usable is the caller's call, made by thresholding
        upstream."""
        assert solved_counts(np.array([[0.4]])).tolist() == [1]

    def test_thresholding_upstream_reproduces_the_binarised_result(self) -> None:
        rng = np.random.default_rng(5)
        rewards = rng.random((20, 6))
        assert np.array_equal(
            solved_counts(rewards > 0.5), solved_counts((rewards > 0.5).astype(float))
        )

    def test_binarising_is_necessary_to_exclude_partial_credit(self) -> None:
        """The rule is blunt on purpose, and this is what being blunt looks like.

        Both ``0.9`` and ``0.1`` are ``> 0``, so a rubric score of 0.1 counts as a
        SOLVED sample. A pipeline that means "at least half credit" must threshold
        before calling; one that means "any credit at all" must not.
        """
        rewards = np.array([[0.9, 0.1, 0.0, 0.0]])
        assert solved_counts(rewards).tolist() == [2]
        assert solved_counts(rewards > 0.5).tolist() == [1]
        assert solved_counts(rewards > 0.95).tolist() == [0]

    def test_out_of_range_rewards_are_counted_by_the_sign_rule(self) -> None:
        """Rewards above 1 (an unbounded reward scale) still count as solved; the
        unit-interval check lives in the Signal half, not here."""
        assert solved_counts(np.array([[2.5, -1.0]])).tolist() == [1]
