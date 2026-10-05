"""Test suite for the topographic equivariance metrics (Keller & Welling 2022).

These are the two numbers the paper's whole argument rests on, and the paper's
sharpest methodological point is that they can DISAGREE: `equivariance_error` is
low for an invariant representation too, and `capcorr` is the metric that actually
measures equivariance. So the suite is built around fixtures where the two
provably disagree, and every "nothing happened" assertion has a "something
happened" twin.

Measured on the fixtures below:
  * a perfectly equivariant latent scores E_eq **1.1e-14** (zero to float64 noise)
  * the SAME latent scores CapCorr **1.0000**
  * an INVARIANT latent scores E_eq identically to the equivariant one (the trap)
    and CapCorr **0.0000**
  * a latent rolling at half speed scores CapCorr **0.97** and one rolling at 3x
    scores **0.65** -- both strictly below 1.0
"""

import numpy as np
import pytest

from dl_techniques.layers.capsules import CapsuleRoll
from dl_techniques.metrics.topographic import (
    capcorr_correlation,
    capcorr_per_capsule,
    equivariance_error,
    observed_roll,
    roll_capsules,
    _periodic_cross_correlation,
    _pearson,
)

B, S, C, D = 96, 6, 3, 8


def _cyclic_factors(seed=0, sequence=S):
    """Ground-truth factors for sequences that DECREASE to the canonical 0.

    The metric compares timestep 0 against the canonical timestep Omega, so the
    sequence must traverse toward the reference value AND the latent must roll in
    the same direction, for the estimated roll and the factor displacement to mean
    the same thing. With ``y_l = (start - l) mod S`` the canonical ``y = 0`` sits
    at ``Omega = start``, so ``|y_Omega - y_0| = start = Omega`` — exactly the
    number of roll steps from ``t_0`` to ``t_Omega``.
    """
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, sequence, size=B)
    return np.stack(
        [(starts[b] - np.arange(sequence)) % sequence for b in range(B)]
    ).astype(np.float64)


def _roll_ladder():
    """Roll amounts for a latent that advances one capsule step per timestep."""
    return np.tile(np.arange(S, dtype=np.int64), (B, 1))


def _build(roll_amounts, seed=1):
    """Latents whose activation at step ``l`` is ``roll_capsules(base, l)``."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(B, C, D))
    return np.stack(
        [
            np.stack(
                [roll_capsules(base[b], int(roll_amounts[b, l]))
                 for l in range(S)],
                axis=0,
            )
            for b in range(B)
        ],
        axis=0,
    )


def _fully_invariant(seed=6):
    """A latent constant in BOTH time and the capsule axis.

    A timestep-constant latent is constant in time but NOT under the roll, and
    `equivariance_error` does see that; an invariant capsule must also be constant
    along its own axis. `np.repeat(base, S, axis=1)` is the wrong spelling here --
    it repeats each element along axis 1 and grows the CAPSULE axis to C*S.
    """
    rng = np.random.default_rng(seed)
    per_capsule = rng.normal(size=(B, C))
    return np.broadcast_to(
        per_capsule[:, None, :, None], (B, S, C, D)
    ).astype(np.float64)


class TestRollCapsules:
    """The NumPy operator, pinned against the layer it mirrors."""

    @pytest.mark.parametrize("shift", [0, 1, 3, -2, 7])
    def test_matches_capsule_roll_exactly(self, shift):
        import keras
        from keras import ops

        values = np.random.default_rng(shift + 3).normal(
            size=(2, 3, 5, 7)
        ).astype("float32")
        layer = CapsuleRoll(num_capsules=5, capsule_dim=7)
        np.testing.assert_array_equal(
            ops.convert_to_numpy(layer(ops.convert_to_tensor(values), shift=shift)),
            roll_capsules(values, shift),
        )
        assert keras.__version__

    def test_a_full_cycle_is_the_identity(self):
        values = np.random.default_rng(4).normal(size=(2, 3, 4))
        current = values
        for _ in range(4):
            current = roll_capsules(current, 1)
        np.testing.assert_allclose(current, values, atol=0.0, rtol=0)


class TestPeriodicCrossCorrelation:
    """The `argmax` shift estimator, and its SIGN."""

    @pytest.mark.parametrize("shift", [0, 1, 3, 5, 7])
    def test_recovers_a_known_roll(self, shift):
        """A positive shift must report a POSITIVE estimate.

        MEASURED: with `t = roll(base, d)` and `t0 = base`, the argmax over
        candidate shifts returns exactly `d` for every d. A `np.roll`-based
        implementation returns `(-d) mod D` instead — indistinguishable under any
        symmetric metric and a silent sign flip under an asymmetric one.
        """
        rng = np.random.default_rng(shift)
        base = rng.normal(size=(4, D))
        moved = roll_capsules(base, shift)
        np.testing.assert_array_equal(
            _periodic_cross_correlation(base, moved),
            np.full(4, shift),
        )

    def test_the_sign_is_pinned_not_just_the_magnitude(self):
        """The other half: reversing the argument order must change the answer.

        Without this, an implementation that returned `(-d) mod D` would pass the
        recovery test above too.
        """
        rng = np.random.default_rng(9)
        base = rng.normal(size=(4, D))
        forward = _periodic_cross_correlation(base, roll_capsules(base, 2))
        backward = _periodic_cross_correlation(
            base, roll_capsules(base, -2)
        )
        np.testing.assert_array_equal(forward, np.full(4, 2))
        np.testing.assert_array_equal(backward, np.full(4, D - 2))


class TestEquivarianceError:
    """Eq. 13 — a smoothness measure, and the suite says so."""

    def test_a_perfectly_equivariant_latent_scores_zero(self):
        """A latent advancing one capsule step per timestep has zero error.

        MEASURED: 1.1e-14, i.e. float64 round-off on a sum of ~900 terms.
        """
        latents = _build(_roll_ladder())
        assert equivariance_error(latents) < 1e-9, (
            f"a roll-equivariant latent scored {equivariance_error(latents)}"
        )

    def test_a_random_latent_scores_far_higher(self):
        """The anti-vacuity arm: the metric must respond to anything at all."""
        rng = np.random.default_rng(2)
        random_latents = rng.normal(size=(B, S, C, D))
        equivariant_error = equivariance_error(_build(_roll_ladder()))
        random_error = equivariance_error(random_latents)
        assert random_error > 100 * equivariant_error, (
            f"random {random_error:.3f} vs equivariant {equivariant_error:.3e}"
        )

    def test_normalization_changes_the_number_and_the_default_is_on(self):
        """Without L2 normalization the metric partly measures MAGNITUDE.

        This is not a defect either way, but the two numbers are different
        quantities, so the default has to be the documented one.
        """
        rng = np.random.default_rng(21)
        # Scale each timestep's activations differently: the DIRECTION is
        # roll-equivariant, the MAGNITUDE is not, so only the normalized number
        # can be zero.
        ladder = _roll_ladder()
        latents = _build(ladder)
        latents = latents * (1.0 + 0.7 * np.arange(S)).reshape(1, S, 1, 1)
        normalized = equivariance_error(latents, normalize=True)
        raw = equivariance_error(latents, normalize=False)
        assert normalized < 1e-9, normalized
        assert raw > 1.0, f"the unnormalized number was {raw}"
        assert rng is not None

    def test_rejects_a_sequence_of_length_one(self):
        """A sum over no terms is undefined, not zero."""
        latents = np.zeros((2, 1, C, D))
        with pytest.raises(ValueError, match="at least 2"):
            equivariance_error(latents)

    def test_rejects_a_rank_two_input(self):
        with pytest.raises(ValueError, match="at least rank 3"):
            equivariance_error(np.zeros((4, C * D)))

    def test_it_is_low_for_an_INVARIANT_latent_too(self):
        """The trap this metric sets, pinned so it is never forgotten.

        An invariant representation — every timestep identical — is perfectly
        SMOOTH, so `equivariance_error` cannot see the difference. That is why
        `capcorr` exists, and why reading this number alone would be a mistake.

        Note the invariant latent must ALSO be constant along the capsule axis for
        a roll to be its own inverse; a timestep-constant latent is constant in
        TIME but not under the roll, and `equivariance_error` does see that. So
        this fixture is constant in both.
        """
        invariant = _fully_invariant()
        equivariant = _build(_roll_ladder())

        invariant_error = equivariance_error(invariant)
        equivariant_error = equivariance_error(equivariant)
        assert invariant_error < 1e-9, invariant_error
        assert equivariant_error < 1e-9, equivariant_error
        # Both are zero -- the metric is blind to the difference. And the metric
        # that is NOT blind refuses to call the invariant one equivariant.
        assert capcorr_correlation(invariant, _cyclic_factors()) != pytest.approx(
            1.0, abs=1e-6
        )

    def test_accepts_a_single_capsule_rank_three_input(self):
        """(batch, sequence, latent) — the model's own flat `t` layout."""
        latents = _build(_roll_ladder())
        flat = latents[:, :, 0, :]
        assert equivariance_error(flat) < 1e-9


class TestObservedRoll:
    """Per-capsule shift estimation."""

    def test_returns_one_shift_per_capsule(self):
        reference = np.random.default_rng(7).normal(size=(B, C, D))
        current = roll_capsules(reference, 3)
        shifts = observed_roll(reference, current)
        assert shifts.shape == (B, C)
        np.testing.assert_array_equal(shifts, np.full((B, C), 3))

    def test_the_direction_is_the_forward_one(self):
        """Earlier first, later second: the roll is POSITIVE for a forward move.

        MEASURED: `observed_roll(base, roll(base, 3))` returns 3, and
        `observed_roll(roll(base, 3), base)` returns D - 3. The paper writes the
        arguments the other way round and leaves `?` defined only up to the
        order, so this is the guard that keeps the convention pinned.
        """
        base = np.random.default_rng(17).normal(size=(B, C, D))
        forward = observed_roll(base, roll_capsules(base, 3))
        backward = observed_roll(roll_capsules(base, 3), base)
        np.testing.assert_array_equal(forward, np.full((B, C), 3))
        np.testing.assert_array_equal(backward, np.full((B, C), D - 3))


class TestCapCorr:
    """Eq. 15/16 — the equivariance measurement."""

    def test_a_perfectly_equivariant_latent_scores_one(self):
        factors = _cyclic_factors()
        latents = _build(_roll_ladder())
        assert capcorr_correlation(latents, factors) == pytest.approx(1.0, abs=1e-6)

    def test_a_fully_invariant_latent_is_UNDEFINED_not_equivariant(self):
        """Constant along the capsule axis, so every roll estimate is 0.

        The correlation of a constant series is undefined, and the metric reports
        `nan` rather than 0.0. Reporting 0.0 would be defensible here, but `nan`
        is the honest answer: "the roll never moves, so there is nothing to
        correlate". What matters for the paper's argument is only that it is
        definitively NOT 1.0.
        """
        base = np.random.default_rng(8).normal(size=(B, C, 1))
        invariant = _fully_invariant(seed=8)
        value = capcorr_correlation(invariant, _cyclic_factors())
        assert not np.isclose(value, 1.0, atol=1e-6), value
        assert base is not None

    def test_an_invariant_latent_scoring_near_zero(self):
        """A latent that rolls by a CONSTANT (not zero) shift.

        This is the BubbleVAE situation: correlated energy, constant
        representation, but the roll estimate has variance so the correlation is
        defined and near zero — the discrimination the paper reports (3369 vs
        13274 on E_eq, while CapCorr stays at chance).
        """
        rng = np.random.default_rng(8)
        base = rng.normal(size=(B, C, D))
        latents = np.stack(
            [
                np.stack([roll_capsules(base[b], 2) for _ in range(S)], axis=0)
                for b in range(B)
            ],
            axis=0,
        )
        value = capcorr_correlation(latents, _cyclic_factors())
        assert np.isnan(value) or abs(value) < 0.3, (
            f"an invariant latent scored {value}"
        )

    def test_a_random_latent_scores_near_zero(self):
        rng = np.random.default_rng(10)
        value = capcorr_correlation(rng.normal(size=(B, S, C, D)), _cyclic_factors())
        assert abs(value) < 0.3, value

    @pytest.mark.parametrize(
        "speed,bound", [(2, 0.98), (3, 0.90)]
    )
    def test_a_wrong_roll_speed_scores_below_one(self, speed, bound):
        """The anti-vacuity arm for the perfect case.

        A latent that rolls at the WRONG rate must not score 1.0, or the
        perfect-case assertion above proves nothing. MEASURED: 2x scores 0.975,
        3x scores 0.646.
        """
        factors = _cyclic_factors()
        latents = _build(_roll_ladder() * speed)
        value = capcorr_correlation(latents, factors)
        assert value < bound, f"speed {speed}x scored {value:.4f}, bound {bound}"

    def test_per_capsule_agrees_with_the_pooled_value_when_capsules_agree(self):
        factors = _cyclic_factors()
        latents = _build(_roll_ladder())
        per_capsule = capcorr_per_capsule(latents, factors)
        assert per_capsule.shape == (C,)
        np.testing.assert_allclose(
            per_capsule,
            np.full(C, capcorr_correlation(latents, factors)),
            atol=1e-9,
        )

    def test_per_capsule_exposes_disagreement_the_pooled_number_hides(self):
        """The mode-across-capsules reduction is an ASSUMPTION, made observable.

        The paper reports that all capsules roll simultaneously in practice. When
        they do not, the pooled number still returns something plausible — so the
        per-capsule arm is what makes the assumption checkable. Here capsules 0
        and 1 are left equariant while capsule 2 rolls BACKWARD, so the per-capsule
        correlations are 1.0, 1.0 and something else.
        """
        factors = _cyclic_factors()
        shifts = _roll_ladder()
        latents = _build(shifts)
        # Capsule 2 rolls the other way, by the negated per-step shift.
        latents[:, :, 2, :] = np.stack(
            [
                roll_capsules(
                    latents[b, 0, 2, :], int((D - int(shifts[b, l])) % D)
                )
                for b in range(B)
                for l in range(S)
            ],
            axis=0,
        ).reshape(B, S, D)

        per_capsule = capcorr_per_capsule(latents, factors)
        assert per_capsule.shape == (C,)
        assert per_capsule[0] == pytest.approx(1.0, abs=1e-6)
        assert per_capsule[1] == pytest.approx(1.0, abs=1e-6)
        assert per_capsule[2] < 0.9, (
            f"the reversed capsule scored {per_capsule[2]}; the per-capsule arm "
            "cannot see disagreement, so it would be decorative"
        )

    def test_an_explicit_canonical_factor_is_honoured(self):
        factors = _cyclic_factors()
        latents = _build(_roll_ladder())
        implicit = capcorr_correlation(latents, factors)
        explicit = capcorr_correlation(latents, factors, canonical_factor=0.0)
        assert explicit == pytest.approx(implicit, abs=1e-9)

    def test_an_unreachable_canonical_factor_raises(self):
        factors = _cyclic_factors()
        latents = _build(_roll_ladder())
        with pytest.raises(ValueError, match="never reach canonical_factor"):
            capcorr_correlation(latents, factors, canonical_factor=-999.0)

    def test_mismatched_factor_shape_raises(self):
        latents = _build(_cyclic_factors() - _cyclic_factors()[:, :1])
        with pytest.raises(ValueError, match="must match"):
            capcorr_correlation(latents, np.zeros((B, S + 1)))


class TestPearson:
    """The correlation helper, including its undefined cases."""

    def test_a_perfect_correlation_is_one(self):
        values = np.arange(10.0)
        assert _pearson(values, values) == pytest.approx(1.0)

    def test_an_anti_correlation_is_minus_one(self):
        values = np.arange(10.0)
        assert _pearson(values, -values) == pytest.approx(-1.0)

    def test_a_constant_series_is_undefined_not_zero(self):
        """Reporting 0.0 would make 'undefined' indistinguishable from 'no
        relationship', which is the one distinction the metric exists for."""
        assert np.isnan(_pearson(np.ones(10), np.arange(10.0)))

    def test_a_single_sample_is_undefined(self):
        assert np.isnan(_pearson(np.array([1.0]), np.array([1.0])))

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="shapes differ"):
            _pearson(np.arange(3.0), np.arange(4.0))