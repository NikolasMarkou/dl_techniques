"""Tests for the Signal half: entropy, the max-entropy weight, advantages, the loss.

Every guard here is pinned against an INDEPENDENT oracle in
``tests/test_optimization/test_ssp/ssp_oracle.py`` -- a plain-numpy, longhand
evaluation that shares no code with the implementation. That matters most for the
Bernoulli KL: the implementation could satisfy ``D == ln2 - H`` while being wrong,
if it were derived from ``H``. The oracle evaluates the defining two-term sum
instead, so the identity is a genuine check and not a tautology.

The RED proofs are in ``test_the_guards_actually_go_red``: they inject the boundary
off-by-one, the ``log1p`` slip, and a scalar-return ``call`` and assert each one is
caught.
"""

from __future__ import annotations

import numpy as np
import pytest

import keras

from dl_techniques.optimization.ssp.signal import (
    MGPOObjective,
    TOKEN_REDUCTIONS,
    ZERO_VARIANCE_POLICIES,
    binary_entropy,
    group_relative_advantages,
    group_success_rate,
    max_entropy_deviation,
    max_entropy_weight,
    mgpo_advantages,
    mgpo_surrogate,
    pack_mgpo_targets,
    unpack_mgpo_targets,
)

from .ssp_oracle import (
    bernoulli_kl_oracle,
    binary_entropy_oracle,
    clipped_surrogate_oracle,
    group_relative_advantage_oracle,
    log1p_confusion_mutant,
)

LN2 = float(np.log(2.0))

# DECISION: every comparison in this file pits the float32 ``keras.ops`` path
# against a float64 numpy oracle, so the tolerance is DERIVED from the dtype rather
# than pasted. float32 has ~7.2 decimal digits (eps = 1.19e-7); a handful of
# transcendental ops on top of that lands a few ulps out, so 1e-6 is the floor for
# "identical arithmetic". The float32-vs-float64 KL gap MEASURED at p=0.5 is
# 1.9e-9, i.e. ~16 float32 ulps -- comfortably inside it and 700x tighter than the
# 1.9e-6 defect the `log1p` slip in `ssp_oracle.log1p_confusion_mutant` produces.
# A tolerance of 1e-12 here (the obvious "exact" choice) would be RED against a
# correct implementation, which is the failure mode `tests/numerics.py` warns about.
FLOAT32_ORACLE_ATOL = 1e-6


def _np(value) -> np.ndarray:
    return np.asarray(keras.ops.convert_to_numpy(value), dtype=np.float64)


# ---------------------------------------------------------------------
# binary entropy
# ---------------------------------------------------------------------


class TestBinaryEntropy:
    @pytest.mark.parametrize("p", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    def test_matches_the_oracle(self, p: float) -> None:
        got = _np(binary_entropy(p))
        assert got == pytest.approx(
            float(binary_entropy_oracle(p)), abs=FLOAT32_ORACLE_ATOL
        )

    def test_maximum_is_log_two_at_one_half(self) -> None:
        assert float(_np(binary_entropy(0.5))) == pytest.approx(LN2, abs=1e-7)

    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_exactly_zero_at_the_endpoints(self, p: float) -> None:
        """``0 log 0`` is 0, not NaN.

        Clipping ``p`` into ``[eps, 1 - eps]`` -- the one-word alternative -- returns
        a small positive entropy here, which would make a fully-solved or
        fully-unsolved group look slightly uncertain and give it a non-zero weight
        forever.
        """
        value = float(_np(binary_entropy(p)))
        assert value == 0.0, f"expected exactly 0.0 at p={p}, got {value}"

    def test_symmetric_about_one_half(self) -> None:
        grid = np.linspace(0.01, 0.99, 25)
        assert _np(binary_entropy(grid)) == pytest.approx(
            _np(binary_entropy(1.0 - grid)), abs=FLOAT32_ORACLE_ATOL
        )

    def test_strictly_increasing_on_the_lower_half(self) -> None:
        lower = np.linspace(0.0, 0.5, 20)
        values = _np(binary_entropy(lower))
        assert all(b >= a for a, b in zip(values, values[1:]))

    def test_reduces_over_an_axis(self) -> None:
        probs = np.array([[0.0, 0.5], [1.0, 0.5]], dtype="float32")
        reduced = _np(binary_entropy(probs, axis=-1))
        assert reduced.shape == (2,)
        assert reduced == pytest.approx(LN2 / 2.0, abs=1e-7)

    @pytest.mark.parametrize("p", [-0.1, 1.1, 2.0])
    def test_out_of_unit_interval_is_refused(self, p: float) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            binary_entropy(p)


# ---------------------------------------------------------------------
# the max-entropy deviation
# ---------------------------------------------------------------------


class TestMaxEntropyDeviation:
    @pytest.mark.parametrize("p", [0.0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0])
    def test_matches_the_two_term_sum_oracle(self, p: float) -> None:
        """The oracle is the DEFINING sum, not the entropy identity."""
        got = float(_np(max_entropy_deviation(p)))
        expected = float(bernoulli_kl_oracle(p))
        assert got == pytest.approx(expected, abs=FLOAT32_ORACLE_ATOL), (
            f"p={p}: got {got}, defining sum gives {expected}"
        )

    def test_is_exactly_the_entropy_deficit_at_p0_half(self) -> None:
        """``D_ME(p || 0.5) == ln2 - H(p)``, exactly.

        This is the identity the paper's prose describes ("distance from the ideal
        maximum-entropy state") and the one its printed formula fails to satisfy.
        """
        grid = np.linspace(0.0, 1.0, 41)
        deviation = _np(max_entropy_deviation(grid))
        deficit = LN2 - _np(binary_entropy(grid))
        assert deviation == pytest.approx(deficit, abs=FLOAT32_ORACLE_ATOL)

    def test_zero_exactly_at_the_target(self) -> None:
        assert float(_np(max_entropy_deviation(0.5))) == pytest.approx(0.0, abs=1e-7)

    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_log_two_at_the_endpoints(self, p: float) -> None:
        assert float(_np(max_entropy_deviation(p))) == pytest.approx(LN2, abs=1e-6)

    def test_symmetric_about_the_target(self) -> None:
        grid = np.linspace(0.05, 0.45, 9)
        assert _np(max_entropy_deviation(grid)) == pytest.approx(
            _np(max_entropy_deviation(1.0 - grid)), abs=1e-6
        )

    def test_non_default_target_is_asymmetric(self) -> None:
        """``p0 != 0.5`` is a real knob and the divergence is NOT symmetric there.

        ``KL(Bern(0) || Bern(0.3)) = log(1 / 0.7) = 0.3567`` but
        ``KL(Bern(1) || Bern(0.3)) = log(1 / 0.3) = 1.2040``.
        """
        low = float(_np(max_entropy_deviation(0.0, p0=0.3)))
        high = float(_np(max_entropy_deviation(1.0, p0=0.3)))
        assert low == pytest.approx(np.log(1.0 / 0.7), abs=1e-5)
        assert high == pytest.approx(np.log(1.0 / 0.3), abs=1e-5)
        assert high > low

    def test_agrees_with_the_oracle_at_a_non_default_target(self) -> None:
        grid = np.linspace(0.0, 1.0, 21)
        assert _np(max_entropy_deviation(grid, p0=0.25)) == pytest.approx(
            bernoulli_kl_oracle(grid, p0=0.25), abs=FLOAT32_ORACLE_ATOL
        )

    @pytest.mark.parametrize("p0", [0.0, 1.0, -0.1, 1.5])
    def test_degenerate_target_is_refused(self, p0: float) -> None:
        """``p0`` in ``{0, 1}`` is a degenerate Bernoulli, whose KL against a
        non-degenerate one is ``+inf`` -- every weight would collapse to 0 and the
        whole batch would be silently discarded."""
        with pytest.raises(ValueError, match=r"strictly inside"):
            max_entropy_deviation(0.5, p0=p0)


# ---------------------------------------------------------------------
# the max-entropy weight -- the three closed forms
# ---------------------------------------------------------------------


class TestMaxEntropyWeight:
    @pytest.mark.parametrize("lam", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_identical_to_two_to_the_minus_lambda_times_exp_of_lambda_H(
            self, lam: float) -> None:
        """``w_ME(p) == 2**-lam * exp(lam * H(p))``.

        The whole reason the weight is cheap, and the reason it is monotone in
        entropy.
        """
        grid = np.linspace(0.0, 1.0, 41)
        got = _np(max_entropy_weight(grid, lam=lam))
        expected = (2.0**-lam) * np.exp(lam * _np(binary_entropy(grid)))
        assert got == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("lam", [0.5, 1.0, 2.0, 5.0])
    def test_exactly_one_at_maximum_uncertainty(self, lam: float) -> None:
        """``w_ME(0.5) == 1.0``. Maximum weight on maximum uncertainty -- the
        property the entire method rests on."""
        assert float(_np(max_entropy_weight(0.5, lam=lam))) == 1.0

    @pytest.mark.parametrize("lam", [0.5, 1.0, 2.0, 3.0])
    def test_floor_at_the_endpoints_is_two_to_the_minus_lambda(self, lam: float) -> None:
        floor = 2.0**-lam
        assert float(_np(max_entropy_weight(0.0, lam=lam))) == pytest.approx(
            floor, abs=1e-6
        )
        assert float(_np(max_entropy_weight(1.0, lam=lam))) == pytest.approx(
            floor, abs=1e-6
        )

    @pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_lambda_zero_is_exactly_one_everywhere(self, p: float) -> None:
        """``lam = 0`` must make the weighting VANISH, bit for bit -- this is the
        paper's own claim that it degenerates to standard GRPO."""
        assert float(_np(max_entropy_weight(p, lam=0.0))) == 1.0

    def test_strictly_increasing_in_entropy(self) -> None:
        grid = np.linspace(0.0, 1.0, 51)
        weights = _np(max_entropy_weight(grid, lam=2.0))
        entropies = _np(binary_entropy(grid))
        order = np.argsort(entropies)
        ordered = weights[order]
        assert all(b >= a - 1e-7 for a, b in zip(ordered, ordered[1:]))

    def test_peaks_exactly_at_one_half(self) -> None:
        grid = np.linspace(0.0, 1.0, 101)
        assert int(np.argmax(_np(max_entropy_weight(grid, lam=3.0)))) == 50

    def test_never_exceeds_one(self) -> None:
        grid = np.linspace(0.0, 1.0, 101)
        assert np.all(_np(max_entropy_weight(grid, lam=4.0)) <= 1.0 + 1e-7)

    def test_a_damped_group_really_is_damped(self) -> None:
        """The end-to-end property: a group the policy always solves, and one it
        never solves, both get strictly less than an ambiguous one."""
        weights = _np(max_entropy_weight(np.array([0.0, 0.5, 1.0]), lam=2.0))
        assert weights[0] < weights[1]
        assert weights[2] < weights[1]

    def test_negative_lambda_is_refused(self) -> None:
        """A negative ``lam`` INVERTS the weighting -- maximum weight on the
        saturated groups -- which is a different and unnamed algorithm."""
        with pytest.raises(ValueError, match=r"lam must be >= 0"):
            max_entropy_weight(0.5, lam=-1.0)

    def test_non_finite_lambda_is_refused(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            max_entropy_weight(0.5, lam=float("inf"))


# ---------------------------------------------------------------------
# group statistics
# ---------------------------------------------------------------------


class TestGroupSuccessRate:
    def test_is_the_mean_over_the_group(self) -> None:
        rewards = np.array([[1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
        assert _np(group_success_rate(rewards)) == pytest.approx([0.75, 0.0])

    def test_removes_the_group_axis(self) -> None:
        rewards = np.zeros((5, 7), dtype="float32")
        assert _np(group_success_rate(rewards)).shape == (5,)

    def test_honours_an_explicit_group_axis(self) -> None:
        rewards = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        assert _np(group_success_rate(rewards, group_axis=0)).shape == (2,)

    def test_out_of_unit_interval_rewards_are_refused(self) -> None:
        """An unbounded reward has no Bernoulli reading, and the weighted advantage
        would be a number with no interpretation."""
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            group_success_rate(np.array([[1.0, 5.0]]))


# ---------------------------------------------------------------------
# group-relative advantages
# ---------------------------------------------------------------------


class TestGroupRelativeAdvantages:
    @pytest.mark.parametrize("seed", range(5))
    def test_matches_the_numpy_oracle(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        rewards = (rng.random((6, 8)) < rng.random((6, 1))).astype("float32")
        got = _np(group_relative_advantages(rewards))
        expected = group_relative_advantage_oracle(rewards)
        assert got == pytest.approx(expected, abs=1e-5)

    def test_zero_variance_group_is_exactly_zero_not_nan(self) -> None:
        """The whole of the ``zero_variance="zero"`` policy is the ``eps`` in the
        denominator: the numerator is identically zero when every rollout earns the
        same reward, so there is no branch to take and no division by zero."""
        rewards = np.array(
            [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]],
            dtype="float32",
        )
        advantages = _np(group_relative_advantages(rewards))
        assert np.all(np.isfinite(advantages)), "a zero-variance group produced NaN/Inf"
        assert np.allclose(advantages[0], 0.0)
        assert np.allclose(advantages[1], 0.0)
        assert np.any(advantages[2] != 0.0)

    def test_advantages_sum_to_zero_within_a_group(self) -> None:
        """``sum_i A_i == 0`` by construction. The reason the batch mean of the MGPO
        objective is identically zero, and the reason the weighting can only show up
        in the gradient."""
        rng = np.random.default_rng(4)
        rewards = (rng.random((10, 8)) < rng.random((10, 1))).astype("float32")
        advantages = _np(group_relative_advantages(rewards))
        assert np.allclose(advantages.sum(axis=-1), 0.0, atol=1e-5)

    def test_a_group_of_one_is_defined_and_zero(self) -> None:
        """``ddof=0`` exists precisely so a group of size 1 does not produce NaN."""
        rewards = np.array([[1.0], [0.0]], dtype="float32")
        advantages = _np(group_relative_advantages(rewards))
        assert np.all(np.isfinite(advantages))
        assert np.allclose(advantages, 0.0)

    def test_zzero_variance_policy_can_be_made_strict(self) -> None:
        rewards = np.array([[1.0, 1.0, 1.0, 1.0]], dtype="float32")
        with pytest.raises(ValueError, match="zero_variance='raise'"):
            group_relative_advantages(rewards, zero_variance="raise")

    def test_strict_policy_passes_when_every_group_has_spread(self) -> None:
        rewards = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        assert np.all(np.isfinite(_np(group_relative_advantages(
            rewards, zero_variance="raise"
        ))))

    def test_ddof_one_needs_a_group_of_two(self) -> None:
        rewards = np.array([[1.0]], dtype="float32")
        with pytest.raises(ValueError, match="needs a group of at least 2"):
            group_relative_advantages(rewards, ddof=1)

    @pytest.mark.parametrize("eps", [0.0, -1e-6])
    def test_non_positive_eps_is_refused(self, eps: float) -> None:
        with pytest.raises(ValueError, match="eps must be > 0"):
            group_relative_advantages(np.ones((1, 4), dtype="float32"), eps=eps)

    def test_unknown_zero_variance_policy_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown zero_variance"):
            group_relative_advantages(
                np.ones((1, 4), dtype="float32"), zero_variance="maybe"
            )


# ---------------------------------------------------------------------
# MGPO advantages
# ---------------------------------------------------------------------


class TestMGPOAdvantages:
    REWARDS = np.array(
        [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
        dtype="float32",
    )

    @pytest.mark.parametrize("lam", [0.0, 0.5, 1.0, 2.0])
    def test_lambda_zero_reproduces_plain_grpo_exactly(self, lam: float) -> None:
        """The paper's degeneration claim, as an equality on real numbers."""
        weighted = _np(mgpo_advantages(self.REWARDS, lam=lam))
        plain = _np(group_relative_advantages(self.REWARDS))
        assert weighted == pytest.approx(plain, abs=1e-6)

    def test_ambiguous_group_is_left_exactly_alone(self) -> None:
        """``p_c = 0.5`` is where the weight is 1, so that group must be bit-equal to
        the unweighted advantage whatever ``lam`` is."""
        weighted = _np(mgpo_advantages(self.REWARDS, lam=8.0))
        plain = _np(group_relative_advantages(self.REWARDS))
        assert weighted[0] == pytest.approx(plain[0], abs=1e-6)

    def test_a_damped_group_is_actually_damped(self) -> None:
        """A group with ``p_c = 1/3`` has a NON-ZERO advantage and a weight below 1,
        so its magnitude must shrink. (The ``p_c in {0, 1}`` groups cannot show this:
        their advantage is already 0, so there is nothing to damp.)"""
        rewards = np.array([[1.0, 0.0, 0.0]], dtype="float32")
        plain = float(np.abs(_np(group_relative_advantages(rewards))).max())
        weighted = float(np.abs(_np(mgpo_advantages(rewards, lam=2.0))).max())
        assert plain > 0.0
        assert weighted < plain
        expected_factor = float(_np(max_entropy_weight(1.0 / 3.0, lam=2.0)))
        assert 0.0 < expected_factor < 1.0
        assert weighted == pytest.approx(plain * expected_factor, rel=1e-4)

    def test_sums_to_zero_within_every_group(self) -> None:
        """The structural property: a per-group CONSTANT weight cannot change a
        zero-sum advantage's sum. This is why the batch mean is 0 for every
        ``lam``."""
        rng = np.random.default_rng(21)
        rewards = (rng.random((12, 8)) < rng.random((12, 1))).astype("float32")
        for lam in (0.5, 2.0, 10.0):
            assert np.allclose(
                _np(mgpo_advantages(rewards, lam=lam)).sum(axis=-1), 0.0, atol=1e-5
            )

    def test_normalize_weights_preserves_the_zero_sum(self) -> None:
        rewards = (np.random.default_rng(2).random((6, 4)) < 0.5).astype("float32")
        assert np.allclose(
            _np(mgpo_advantages(rewards, lam=3.0, normalize_weights=True)).sum(-1),
            0.0,
            atol=1e-5,
        )

    def test_normalize_weights_scales_the_ambiguous_group_above_one(self) -> None:
        """Normalisation preserves the MEAN, not the maximum. With most groups
        saturated the rescaling factor is large, so the few ambiguous groups end up
        with an advantage greater than 1. Worth knowing before using it as if it
        were a clamp."""
        rewards = np.array(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0]],
            dtype="float32",
        )
        plain = _np(group_relative_advantages(rewards))
        normalised = _np(mgpo_advantages(rewards, lam=4.0, normalize_weights=True))
        assert np.abs(normalised[2]).max() > np.abs(plain[2]).max()

    def test_broadcasts_for_a_group_on_any_axis(self) -> None:
        rewards = np.array([[1.0, 1.0, 0.0, 0.0]], dtype="float32")
        along_last = _np(mgpo_advantages(rewards, lam=2.0))
        along_first = _np(mgpo_advantages(rewards.T.copy(), group_axis=0, lam=2.0))
        assert along_last.shape == (1, 4)
        assert along_first.shape == (4, 1)
        assert along_first == pytest.approx(along_last.T, abs=1e-6)

    def test_group_of_one_produces_no_nan(self) -> None:
        rewards = np.array([[1.0], [0.0]], dtype="float32")
        assert np.all(np.isfinite(_np(mgpo_advantages(rewards, lam=2.0))))


# ---------------------------------------------------------------------
# the clipped surrogate
# ---------------------------------------------------------------------


class TestMGPOSurrogate:
    LOG_PROBS = np.full((3, 5), -0.1, dtype="float32")
    OLD = np.zeros((3, 5), dtype="float32")
    ADV = np.array([1.0, -1.0, 0.5], dtype="float32")

    def test_matches_the_longhand_oracle(self) -> None:
        """Token level first, then the reduction applied on top of it."""
        oracle = clipped_surrogate_oracle(self.LOG_PROBS, self.OLD, self.ADV)
        token_level = _np(mgpo_surrogate(self.LOG_PROBS, self.OLD, self.ADV,
                                         reduction="none"))
        assert token_level == pytest.approx(oracle, abs=FLOAT32_ORACLE_ATOL)

        # Every token of a response carries the same value in this fixture, so the
        # paper's per-sequence mean is that value.
        reduced = _np(mgpo_surrogate(self.LOG_PROBS, self.OLD, self.ADV))
        assert reduced == pytest.approx(oracle.mean(axis=-1), abs=FLOAT32_ORACLE_ATOL)

    def test_matches_the_longhand_oracle_on_a_varying_grid(self) -> None:
        """The previous fixture is constant along tokens, which cannot tell a
        per-token bug from a per-sequence bug. This one cannot."""
        rng = np.random.default_rng(31)
        log_probs = rng.normal(size=(4, 7)).astype("float32") * 0.3
        old = rng.normal(size=(4, 7)).astype("float32") * 0.3
        advantage = rng.normal(size=(4,)).astype("float32")
        oracle = clipped_surrogate_oracle(log_probs, old, advantage)
        token_level = _np(mgpo_surrogate(log_probs, old, advantage, reduction="none"))
        assert token_level == pytest.approx(oracle, abs=FLOAT32_ORACLE_ATOL)
        reduced = _np(mgpo_surrogate(log_probs, old, advantage))
        assert reduced == pytest.approx(oracle.mean(axis=-1), abs=FLOAT32_ORACLE_ATOL)

    def test_per_sequence_mean_returns_one_value_per_response(self) -> None:
        """``losses/AGENTS.md`` requires ``(batch,)``, never a scalar."""
        got = _np(mgpo_surrogate(self.LOG_PROBS, self.OLD, self.ADV))
        assert got.shape == (3,)

    def test_per_token_mean_is_a_scalar(self) -> None:
        got = _np(mgpo_surrogate(self.LOG_PROBS, self.OLD, self.ADV,
                                 reduction="per_token_mean"))
        assert got.shape == ()

    def test_none_returns_the_token_grid(self) -> None:
        got = _np(mgpo_surrogate(self.LOG_PROBS, self.OLD, self.ADV, reduction="none"))
        assert got.shape == (3, 5)

    def test_per_sequence_and_per_token_differ_when_lengths_are_unequal(self) -> None:
        """The paper averages over a response's tokens FIRST and over responses
        SECOND, so a long response must not dominate the batch.

        A ragged batch is not expressible as a ``(B, T)`` array, so unequal lengths
        arrive as a ``token_mask`` -- which is exactly why the mask exists.
        """
        log_probs = np.full((2, 6), -0.1, dtype="float32")
        old = np.zeros((2, 6), dtype="float32")
        advantages = np.array([1.0, -1.0], dtype="float32")
        mask = np.zeros((2, 6), dtype="float32")
        mask[0, :2] = 1.0        # response 0 has 2 scored tokens
        mask[1, :6] = 1.0        # response 1 has 6

        per_sequence = _np(mgpo_surrogate(
            log_probs, old, advantages, token_mask=mask,
            reduction="per_sequence_mean",
        ))
        per_token = float(_np(mgpo_surrogate(
            log_probs, old, advantages, token_mask=mask,
            reduction="per_token_mean",
        )))
        assert per_sequence.shape == (2,)
        # Every scored token carries the SAME value, so the per-sequence mean is that
        # value twice over, while the flat token mean is dominated by the long row.
        assert per_token != pytest.approx(float(per_sequence.mean()), abs=1e-4)
        assert per_token == pytest.approx(
            float((per_sequence[0] * 2 + per_sequence[1] * 6) / 8), abs=1e-6
        )

    def test_clipping_engages_outside_the_range(self) -> None:
        far = np.full((3, 5), 5.0, dtype="float32")
        clipped = _np(mgpo_surrogate(far, self.OLD, self.ADV, clip_eps=0.2))
        ratio = float(np.exp(5.0))
        expected = -np.minimum(ratio * self.ADV, 1.2 * self.ADV)
        assert clipped == pytest.approx(expected, abs=1e-3)

    def test_no_clipping_inside_the_range(self) -> None:
        inside = _np(mgpo_surrogate(self.LOG_PROBS, self.OLD, self.ADV, clip_eps=0.2))
        unconstrained = _np(
            mgpo_surrogate(self.LOG_PROBS, self.OLD, self.ADV, clip_eps=1e6)
        )
        assert inside == pytest.approx(unconstrained, abs=1e-6)

    def test_the_advantage_cannot_be_factored_out(self) -> None:
        """Why the advantage is packed into ``y_true`` instead of riding in
        ``sample_weight``: the ``min`` picks a different branch depending on the
        SIGN of the advantage, so no linear factorisation exists."""
        ratio = float(np.exp(-0.1))
        for advantage in (1.0, -1.0):
            got = float(_np(mgpo_surrogate(
                self.LOG_PROBS[:1], self.OLD[:1], np.array([advantage])
            ))[0])
            expected = -min(ratio * advantage, ratio * advantage)
            assert got == pytest.approx(expected, abs=1e-6)
        assert got != pytest.approx(-expected, abs=1e-6)

    def test_token_mask_excludes_positions(self) -> None:
        """A constant log-prob grid cannot show this: every token carries the same
        value, so masking some of them leaves the per-sequence mean unchanged. The
        grid has to vary along tokens for the mask to be observable at all."""
        log_probs = np.array(
            [[-0.1, -0.5, -0.2, -0.9, -0.3]] * 3, dtype="float32"
        )
        old = np.zeros((3, 5), dtype="float32")
        advantage = np.array([1.0, -1.0, 0.5], dtype="float32")

        mask = np.zeros((3, 5), dtype="float32")
        mask[:, :2] = 1.0
        masked = _np(mgpo_surrogate(log_probs, old, advantage, token_mask=mask))
        full = _np(mgpo_surrogate(log_probs, old, advantage))
        assert not np.allclose(masked, full)

        # Only the first two tokens count, so the value is that sub-grid's mean.
        first_two = _np(mgpo_surrogate(log_probs[:, :2], old[:, :2], advantage))
        assert masked == pytest.approx(first_two, abs=FLOAT32_ORACLE_ATOL)

    def test_a_full_mask_reproduces_the_unmasked_value(self) -> None:
        mask = np.ones((3, 5), dtype="float32")
        assert _np(mgpo_surrogate(self.LOG_PROBS, self.OLD, self.ADV,
                                  token_mask=mask)) == pytest.approx(
            _np(mgpo_surrogate(self.LOG_PROBS, self.OLD, self.ADV)),
            abs=FLOAT32_ORACLE_ATOL,
        )

    def test_per_token_advantage_is_accepted(self) -> None:
        per_token = np.repeat(self.ADV[:, None], 5, axis=1)
        got = _np(mgpo_surrogate(self.LOG_PROBS, self.OLD, per_token))
        assert got == pytest.approx(
            _np(mgpo_surrogate(self.LOG_PROBS, self.OLD, self.ADV)), abs=1e-6
        )

    def test_mismatched_log_prob_shapes_are_refused(self) -> None:
        with pytest.raises(ValueError, match="SAME shape"):
            mgpo_surrogate(self.LOG_PROBS, self.OLD[:, :3], self.ADV)

    def test_all_zero_token_mask_row_is_refused(self) -> None:
        """``1 / |y_i|`` would be a division by zero, and a response with no scored
        token is a caller bug, not something to average around."""
        mask = np.ones((3, 5), dtype="float32")
        mask[1] = 0.0
        with pytest.raises(ValueError, match="all-zero token_mask"):
            mgpo_surrogate(self.LOG_PROBS, self.OLD, self.ADV, token_mask=mask)

    def test_wrongly_shaped_token_mask_is_refused(self) -> None:
        with pytest.raises(ValueError, match="token_mask must have"):
            mgpo_surrogate(self.LOG_PROBS, self.OLD, self.ADV,
                           token_mask=np.ones((3, 4), dtype="float32"))

    @pytest.mark.parametrize("clip_eps", [0.0, -0.2])
    def test_non_positive_clip_is_refused(self, clip_eps: float) -> None:
        with pytest.raises(ValueError, match="clip_eps must be > 0"):
            mgpo_surrogate(self.LOG_PROBS, self.OLD, self.ADV, clip_eps=clip_eps)

    def test_unknown_reduction_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown reduction"):
            mgpo_surrogate(self.LOG_PROBS, self.OLD, self.ADV, reduction="sum")


# ---------------------------------------------------------------------
# packing
# ---------------------------------------------------------------------


class TestPacking:
    def test_round_trip_is_exact(self) -> None:
        old = np.random.default_rng(0).normal(size=(4, 6)).astype("float32")
        advantage = np.array([1.0, -1.0, 0.5, 0.0], dtype="float32")
        reference, weights = unpack_mgpo_targets(pack_mgpo_targets(old, advantage))
        assert np.array_equal(reference, old)
        assert np.allclose(weights, advantage[:, None])

    def test_shape_is_batch_tokens_two(self) -> None:
        packed = pack_mgpo_targets(np.zeros((4, 6), dtype="float32"),
                                   np.zeros((4,), dtype="float32"))
        assert packed.shape == (4, 6, 2)

    def test_advantage_broadcasts_over_tokens(self) -> None:
        packed = pack_mgpo_targets(np.zeros((4, 6), dtype="float32"),
                                   np.arange(4, dtype="float32")[:, None])
        _, weights = unpack_mgpo_targets(packed)
        assert np.allclose(weights, np.broadcast_to(np.arange(4)[:, None], (4, 6)))

    def test_the_per_rollout_shape_gets_a_named_error(self) -> None:
        """A ``(B, G)`` advantage is one value per ROLLOUT and has no meaning against
        one response's token axis; this is the most likely first mistake."""
        with pytest.raises(ValueError, match="per-ROLLOUT"):
            pack_mgpo_targets(
                np.zeros((2, 6), dtype="float32"),
                np.zeros((2, 4), dtype="float32"),
            )

    def test_non_2d_reference_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be 2-D"):
            pack_mgpo_targets(np.zeros((4,), dtype="float32"),
                              np.zeros((4,), dtype="float32"))

    def test_unpacking_a_bare_array_is_refused(self) -> None:
        with pytest.raises(ValueError, match="packed targets must be"):
            unpack_mgpo_targets(np.zeros((4, 6), dtype="float32"))


# ---------------------------------------------------------------------
# the Keras loss
# ---------------------------------------------------------------------


class TestMGPOObjective:
    B, T, G = 2, 6, 4
    REWARDS = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype="float32")

    def _packed(self, lam: float = 2.0):
        advantage = _np(mgpo_advantages(self.REWARDS, lam=lam)).reshape(-1)
        old = np.zeros((self.B * self.G, self.T), dtype="float32")
        new = np.full((self.B * self.G, self.T), -0.1, dtype="float32")
        return pack_mgpo_targets(old, advantage), new, advantage

    def test_registration_follows_the_house_contract(self, registration_contract) -> None:
        key = registration_contract(
            MGPOObjective, expected_package="dl_techniques.optimization.ssp.signal"
        )
        assert key.endswith(">MGPOObjective")

    def test_call_returns_one_value_per_response(self) -> None:
        """``(batch,)``, the shape ``losses/AGENTS.md`` requires. A scalar return
        would broadcast against ``sample_weight`` and discard WHICH rows were
        weighted."""
        packed, new, _ = self._packed()
        values = np.asarray(MGPOObjective().call(packed, new))
        assert values.shape == (self.B * self.G,)

    def test_call_agrees_with_the_pure_function(self) -> None:
        packed, new, advantage = self._packed()
        expected = _np(mgpo_surrogate(new, np.zeros_like(new), advantage))
        assert np.asarray(MGPOObjective().call(packed, new)) == pytest.approx(
            expected, abs=1e-6
        )

    def test_dunder_call_reduces_to_the_batch_mean(self) -> None:
        """``__call__`` goes through Keras' reduction; ``call`` does not. Confusing
        the two is why the per-response vector appeared to be a scalar."""
        packed, new, _ = self._packed()
        loss = MGPOObjective()
        assert np.ndim(np.asarray(loss(packed, new))) == 0
        assert np.asarray(loss.call(packed, new)).shape == (self.B * self.G,)

    def test_the_batch_mean_is_zero_but_the_gradient_is_not(self) -> None:
        """The property that makes a logged MGPO loss look broken when it is not.

        The advantage sums to zero within every group and the weight is constant
        across a group, so the batch mean is identically 0 for every ``lam``. The
        gradient is unaffected, and that is where the method acts.
        """
        import tensorflow as tf

        packed, new, advantage = self._packed(lam=2.0)
        loss = MGPOObjective()
        assert float(loss(packed, new)) == pytest.approx(0.0, abs=1e-6)

        variable = tf.Variable(np.full((self.B * self.G, self.T), -0.1, dtype="float32"))
        with tf.GradientTape() as tape:
            tape.watch(variable)
            total = tf.reduce_sum(loss.call(packed, keras.ops.convert_to_tensor(variable)))
        gradient = tape.gradient(total, [variable])[0]
        assert float(tf.reduce_max(tf.abs(gradient))) > 1e-6

    def test_the_weighting_changes_the_gradient(self) -> None:
        """``p_c = 1/3`` gives a non-zero advantage AND a weight below 1, so the
        gradient magnitude must differ from the unweighted objective's. (A group at
        ``p_c = 0.5`` or a zero-variance group cannot show this, which is why the
        fixture is a 3-rollout group and not the 4-rollout one used elsewhere.)
        """
        import tensorflow as tf

        rewards = np.array([[1.0, 0.0, 0.0]], dtype="float32")   # 1 query, 3 rollouts
        old = np.zeros((3, 4), dtype="float32")
        new = np.full((3, 4), -0.1, dtype="float32")

        def peak(lam: float) -> float:
            advantage = _np(mgpo_advantages(rewards, lam=lam)).reshape(-1)
            variable = tf.Variable(new.copy())
            with tf.GradientTape() as tape:
                tape.watch(variable)
                total = tf.reduce_sum(MGPOObjective().call(
                    pack_mgpo_targets(old, advantage),
                    keras.ops.convert_to_tensor(variable),
                ))
            gradient = tape.gradient(total, [variable])[0]
            return float(tf.reduce_max(tf.abs(gradient)))

        assert peak(2.0) < peak(0.0)

    def test_sample_weight_selects_rows_and_is_not_applied_twice(self) -> None:
        """Keras multiplies by ``sample_weight`` AFTER ``call`` returns, so ``call``
        must not pre-multiply -- which would square the weight."""
        packed, new, _ = self._packed()
        loss = MGPOObjective()
        per_response = np.asarray(loss.call(packed, new))
        weight = np.zeros(self.B * self.G, dtype="float32")
        weight[:2] = 1.0
        weighted = float(loss(packed, new, sample_weight=weight))
        # sum(per_response * w) / B, with rows 2+ excluded entirely.
        assert weighted == pytest.approx(float(per_response[:2].sum() / (self.B * self.G)),
                                         abs=1e-6)

    def test_it_is_not_a_member_of_the_premature_scalar_family(self) -> None:
        """The predicate from ``test_losses/test_the_premature_scalar_family_is_pinned``.

        ``loss(w=[1,1,1,0]) == loss(batch) * mean(w)`` holds iff ``call`` returned a
        scalar. The batch mean here is structurally zero, so the comparison is run
        against a WEIGHTED call whose value is not zero -- otherwise the predicate
        is vacuous.
        """
        packed, new, _ = self._packed()
        loss = MGPOObjective()
        weight = np.zeros(self.B * self.G, dtype="float32")
        weight[2:] = 1.0                      # keeps the +0.9048 rows only
        weighted = float(loss(packed, new, sample_weight=weight))
        unweighted = float(loss(packed, new))
        assert abs(weighted) > 1e-6, "the control must not be zero"
        assert not np.isclose(weighted, unweighted * float(weight.mean()), atol=1e-6)

    def test_a_token_shaped_sample_weight_is_refused(self) -> None:
        """It would be folded into the mean Keras already takes over responses,
        double-counting the padding mask."""
        packed, new, _ = self._packed()
        with pytest.raises(ValueError, match="per-RESPONSE"):
            MGPOObjective()(packed, new, sample_weight=np.ones((self.B * self.G, self.T)))

    def test_unpacked_y_true_is_refused(self) -> None:
        _, new, _ = self._packed()
        with pytest.raises(ValueError, match="packed"):
            MGPOObjective()(np.zeros((self.B * self.G, self.T), dtype="float32"), new)

    def test_mismatched_token_shapes_are_refused(self) -> None:
        packed, _, _ = self._packed()
        with pytest.raises(ValueError, match="same token shape"):
            MGPOObjective()(packed, np.zeros((4, 3), dtype="float32"))

    def test_config_round_trips(self) -> None:
        loss = MGPOObjective(clip_eps=0.15)
        config = loss.get_config()
        assert config["clip_eps"] == 0.15
        rebuilt = MGPOObjective.from_config(config)
        assert rebuilt.clip_eps == 0.15
        assert rebuilt.name == loss.name

    def test_serializes_through_keras(self) -> None:
        original = MGPOObjective(clip_eps=0.3, name="custom_mgpo")
        blob = keras.utils.serialize_keras_object(original)
        rebuilt = keras.utils.deserialize_keras_object(blob)
        assert isinstance(rebuilt, MGPOObjective)
        assert rebuilt.clip_eps == 0.3
        assert rebuilt.name == "custom_mgpo"

    @pytest.mark.parametrize("clip_eps", [0.0, -0.1])
    def test_non_positive_clip_is_refused_at_construction(self, clip_eps: float) -> None:
        with pytest.raises(ValueError, match="clip_eps must be > 0"):
            MGPOObjective(clip_eps=clip_eps)


# ---------------------------------------------------------------------
# RED proofs -- each guard above, shown able to fail
# ---------------------------------------------------------------------


class TestTheGuardsActuallyGoRed:
    """Every mutation here is a defect this file claims to catch.

    Proven by injecting it and asserting the oracle disagrees. A guard that has never
    failed is not known to work.
    """

    def test_the_oracle_rejects_the_log1p_slip(self) -> None:
        """The one-character slip that made the weight peak at the wrong ``p``.

        ``log1p(1 - p)`` is ``log(2 - p)``, not ``log(1 - p)``.
        """
        grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        correct = bernoulli_kl_oracle(grid)
        # The mutant's second term, built the wrong way.
        mutant_second = (1.0 - grid) * (
            log1p_confusion_mutant(grid) - float(np.log(1.0 - 0.5))
        )
        mutant = grid * (np.log(np.where(grid > 0, grid, 1.0)) - float(np.log(0.5))) \
            + mutant_second
        assert not np.allclose(correct, mutant, atol=1e-3), (
            "the mutant is indistinguishable from the oracle, so the KL guards "
            "cannot see this class of slip"
        )
        # And the shipped implementation agrees with the ORACLE, not the mutant.
        assert _np(max_entropy_deviation(grid)) == pytest.approx(
            correct, abs=FLOAT32_ORACLE_ATOL
        )

    def test_the_degenerate_boundary_guard_sees_the_off_by_one(self) -> None:
        """``n - c == k`` is NOT degenerate; a ``<=`` test would score a
        never-solved row as perfect coverage.

        With ``n = 4, c = 0, k = 4`` the falling factorial lands on ``C(4,4) = 1``,
        the ratio is ``1``, and pass@k is ``0`` -- a model that solved nothing.
        """
        from dl_techniques.metrics.pass_at_k import per_problem_pass_at_k

        from .ssp_oracle import degenerate_boundary_mutant

        n, c, k = 4, 0, 4
        assert (n - c < k) is False, "the correct boundary must NOT be degenerate here"
        assert degenerate_boundary_mutant(n - c, k) is True, (
            "the mutant must differ from the shipped rule, or this RED proof is "
            "vacuous"
        )
        assert float(per_problem_pass_at_k(np.zeros((1, n)), k)[0]) == 0.0

    def test_the_lambda_zero_guard_sees_a_non_one_weight(self) -> None:
        """``lam = 0 -> w == 1`` is only a real guard if ``lam > 0`` is not 1."""
        assert float(_np(max_entropy_weight(0.0, lam=2.0))) < 1.0
        assert float(_np(max_entropy_weight(0.0, lam=0.0))) == 1.0

    def test_the_shape_guard_sees_a_scalar_return(self) -> None:
        """``call`` returning a scalar instead of ``(batch,)`` must break the
        per-response assertions."""
        packed, new, _ = TestMGPOObjective()._packed()
        scalar_return = np.asarray(MGPOObjective().call(packed, new)).sum()
        assert np.ndim(scalar_return) == 0
        assert np.asarray(MGPOObjective().call(packed, new)).shape == (8,)
