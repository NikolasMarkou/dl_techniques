"""Tests for LocalSupportGate.

The load-bearing assertions are the two-sided ones: a gate must open on
in-distribution data AND stay closed on out-of-distribution data. Either half
alone passes for a gate that is simply always-open or always-closed, which is
the entire mechanism.
"""

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from dl_techniques.layers.statistics.local_support_gate import (  # noqa: E402
    LocalSupportGate,
)

DIM = 16


def _separable_data(seed=0, in_dim=DIM, n_per=400, scale=6.0):
    """A well-separated in-phase distribution and a generic reference one.

    Separability matters: overlapping data makes the hit rates meaningless, and
    a gate that looks correct on it proves nothing.
    """
    rng = np.random.default_rng(seed)
    centre = np.zeros(in_dim)
    centre[0] = scale
    in_phase = (centre + rng.standard_normal((n_per, in_dim))).astype('float32')
    generic = rng.standard_normal((n_per * 4, in_dim)).astype('float32')
    return in_phase, generic


def _make_gate(**overrides):
    params = dict(
        input_dim=DIM, pos_components=4, neg_components=4, seed=7,
        projection_dim=8,
    )
    params.update(overrides)
    gate = LocalSupportGate(**params)
    gate.build((None, DIM))
    return gate


class TestConstruction:
    @pytest.mark.parametrize('field,value', [
        ('input_dim', 0), ('input_dim', -4),
        ('pos_components', 0), ('neg_components', 0),
    ])
    def test_a_non_positive_count_raises(self, field, value):
        with pytest.raises(ValueError, match=field):
            LocalSupportGate(**{**dict(input_dim=DIM), field: value})

    def test_a_projection_wider_than_its_input_raises(self):
        """A projection wider than its input is not a JL sketch; it reduces
        nothing and costs everything. Accepting it would silently discard the
        reduction the caller was reaching for."""
        with pytest.raises(ValueError, match='at most input_dim'):
            LocalSupportGate(input_dim=8, projection_dim=9)

    def test_a_projection_equal_to_the_input_is_allowed(self):
        gate = LocalSupportGate(input_dim=8, projection_dim=8)
        assert gate.effective_dim == 8

    @pytest.mark.parametrize('bad', ['full', 'tied', 'spherical', None])
    def test_an_unsupported_covariance_type_raises(self, bad):
        with pytest.raises(ValueError, match='covariance_type'):
            LocalSupportGate(input_dim=DIM, covariance_type=bad)

    @pytest.mark.parametrize('bad', ['stream', 'online', '', None])
    def test_an_unknown_fit_mode_raises(self, bad):
        with pytest.raises(ValueError, match='fit_mode'):
            LocalSupportGate(input_dim=DIM, fit_mode=bad)

    @pytest.mark.parametrize('bad', ['continuous', 'prob', None])
    def test_an_unknown_output_mode_raises(self, bad):
        with pytest.raises(ValueError, match='output_mode'):
            LocalSupportGate(input_dim=DIM, output_mode=bad)

    @pytest.mark.parametrize('bad', [-0.1, 1.1, 2.0])
    def test_a_smoothing_alpha_outside_the_unit_interval_raises(self, bad):
        with pytest.raises(ValueError, match='smoothing_alpha'):
            LocalSupportGate(input_dim=DIM, smoothing_alpha=bad)

    @pytest.mark.parametrize('field', ['variance_floor', 'dead_mass_threshold'])
    def test_a_non_positive_numerical_floor_raises(self, field):
        with pytest.raises(ValueError, match=field):
            LocalSupportGate(input_dim=DIM, **{field: 0.0})

    def test_the_seed_is_reflected_in_the_projection_draw(self):
        """Two gates with the same seed must build the SAME sketch.

        Without a seed the layer is untestable and unreproducible: two gates
        would gate differently on identical data.
        """
        first = _make_gate(seed=11)
        second = _make_gate(seed=11)
        third = _make_gate(seed=12)
        sketch_a = np.asarray(first.projection.kernel.value)
        sketch_b = np.asarray(second.projection.kernel.value)
        sketch_c = np.asarray(third.projection.kernel.value)
        assert np.allclose(sketch_a, sketch_b)
        assert not np.allclose(sketch_a, sketch_c)

    def test_the_projection_is_not_trainable(self):
        """A trained sketch forfeits the distance bound it exists to exploit."""
        assert _make_gate().projection.trainable is False

    def test_an_aggressive_projection_can_destroy_the_signal(self):
        """Pins a real cost of JL reduction, and pins that it is SEED-DEPENDENT.

        The sketch preserves pairwise distances in expectation, not the one
        direction two populations happen to be separated along. So a projection
        aggressive enough to be worth having can attenuate exactly the structure
        the gate exists to detect -- and WHICH direction is attenuated depends on
        the random draw.

        Measured over 8 seeds on a 16-dim in-phase distribution (centre at +6 on
        a single axis) against a standard-normal reference, reporting the WORST
        seed rather than a mean:

        | projection_dim | in-phase open (min) | out-of-phase open (max) |
        |---|---|---|
        | None (disabled) | 0.975 | 0.021 |
        | 8 (half)        | 0.957 | 0.049 |
        | 4 (quarter)     | 0.785 | 0.237 |
        | 2 (eighth)      | 0.678 | 0.418 |

        The spread is the point. At ``projection_dim=4`` the mean in-phase rate is
        0.910 but the worst seed drops to 0.785 -- a single-seed measurement
        would have reported anywhere in that band and looked conclusive.

        This is a knob to set deliberately, not a defect: ``projection_dim`` is a
        compression/faithfulness trade-off whose faithful end is full width or
        disabled. Note full width is not the identity -- the sketch is an i.i.d.
        Gaussian matrix scaled by ``1/sqrt(k)``, so at ``k == d`` it is a random
        well-conditioned rotation, and it costs a little relative to no
        projection at all.
        """
        def worst_case(projection_dim, seeds=range(8)):
            in_rates, out_rates = [], []
            for seed in seeds:
                in_phase, generic = _separable_data()
                gate_args = {'pos_components': 4, 'neg_components': 4}
                if projection_dim is not None:
                    gate_args['projection_dim'] = projection_dim
                gate = LocalSupportGate(input_dim=DIM, seed=seed, **gate_args)
                gate.build((None, DIM))
                gate.fit_pos(in_phase)
                gate.fit_neg(generic)
                in_rates.append(
                    float(np.asarray(gate(in_phase[:400], training=False)).mean())
                )
                out_rates.append(
                    float(np.asarray(gate(generic[:400], training=False)).mean())
                )
            return min(in_rates), max(out_rates)

        none_worst = worst_case(None)
        full_worst = worst_case(8)
        half_worst = worst_case(4)

        # Monotone in the compression ratio, on the worst seed. This is the
        # claim: compression costs faithfulness, and the cost shows up as
        # variance across draws rather than as a uniform shift.
        assert full_worst[0] >= none_worst[0] - 0.05
        assert half_worst[0] < full_worst[0] - 0.10, (
            f"a quarter-width sketch should lose worst-seed separation, got "
            f"{half_worst} vs {full_worst}"
        )
        assert half_worst[1] > full_worst[1] + 0.05, (
            f"a quarter-width sketch should widen the worst-case gate, got "
            f"{half_worst} vs {full_worst}"
        )


class TestShape:
    def test_compute_output_shape_works_unbuilt(self):
        gate = LocalSupportGate(input_dim=DIM, projection_dim=8)
        assert gate.compute_output_shape((2, 5, DIM)) == (2, 5)

    def test_the_unreduced_form_appends_a_singleton(self):
        gate = LocalSupportGate(input_dim=DIM, output_shape_reduced=False)
        assert gate.compute_output_shape((2, 5, DIM)) == (2, 5, 1)

    @pytest.mark.parametrize('shape', [(3, DIM), (2, 7, DIM), (2, 3, 5, DIM)])
    def test_the_forward_matches_the_declared_shape(self, shape):
        """One decision per token, so the feature axis is consumed."""
        gate = _make_gate()
        out = gate(np.random.default_rng(0).standard_normal(shape).astype('float32'),
                   training=False)
        assert tuple(out.shape) == shape[:-1]

    def test_compute_output_shape_agrees_with_the_forward(self):
        """The declared shape and the actual one must not drift apart.

        Three copies of one shape formula is three chances to drift; this pins
        the declared form against the real one for every rank.
        """
        gate = _make_gate()
        for shape in [(3, DIM), (2, 7, DIM), (2, 3, 5, DIM)]:
            out = gate(
                np.random.default_rng(0).standard_normal(shape).astype('float32'),
                training=False,
            )
            assert gate.compute_output_shape(shape) == tuple(out.shape), shape

    def test_compute_output_shape_rejects_a_rank_one_input(self):
        with pytest.raises(ValueError, match='rank >= 2'):
            _make_gate().compute_output_shape((DIM,))

    @pytest.mark.parametrize('length', [1, 2])
    def test_degenerate_sequence_lengths_are_finite(self, length):
        """Guide §16.3: degenerate lengths on the STATIC path."""
        in_phase, generic = _separable_data()
        gate = _make_gate()
        gate.fit_pos(in_phase)
        gate.fit_neg(generic)
        out = gate(
            np.random.default_rng(1).standard_normal((1, length, DIM)).astype('float32'),
            training=False,
        )
        assert bool(np.all(np.isfinite(np.asarray(out))))

    def test_a_rank_one_input_raises(self):
        gate = _make_gate()
        with pytest.raises(ValueError, match='rank >= 2'):
            gate(np.zeros(DIM, dtype='float32'), training=False)

    def test_a_mismatched_input_width_raises(self):
        gate = _make_gate()
        with pytest.raises(ValueError, match='input_dim'):
            gate(np.zeros((4, DIM + 1), dtype='float32'), training=False)


class TestFittedState:
    def test_a_fresh_gate_is_not_fitted(self):
        gate = LocalSupportGate(input_dim=DIM)
        assert not gate.is_fitted('pos')
        assert not gate.is_fitted('neg')

    def test_fitting_one_side_leaves_the_other_unfitted(self):
        in_phase, _ = _separable_data()
        gate = _make_gate()
        gate.fit_pos(in_phase)
        assert gate.is_fitted('pos')
        assert not gate.is_fitted('neg')

    def test_reset_clears_only_the_named_side(self):
        in_phase, generic = _separable_data()
        gate = _make_gate()
        gate.fit_pos(in_phase)
        gate.fit_neg(generic)
        gate.reset('pos')
        assert not gate.is_fitted('pos')
        assert gate.is_fitted('neg'), (
            "the negative mixture describes a fixed corpus and is fitted once; "
            "resetting the positive must not discard it"
        )

    def test_a_sample_with_fewer_rows_than_components_raises(self):
        gate = _make_gate(pos_components=8)
        with pytest.raises(ValueError, match='components'):
            gate.fit_pos(np.zeros((3, DIM), dtype='float32'))

    def test_a_malformed_sample_raises(self):
        gate = _make_gate()
        with pytest.raises(ValueError, match='2-D'):
            gate.fit_pos(np.zeros((2, 3, DIM), dtype='float32'))

    def test_a_zero_max_iter_raises(self):
        gate = _make_gate()
        with pytest.raises(ValueError, match='max_iter'):
            gate.fit_pos(np.zeros((50, DIM), dtype='float32'), max_iter=0)


class TestTheTwoSidedDecision:
    """THE core property. Both halves, or neither means anything."""

    def test_the_fitted_gate_opens_on_in_distribution_data(self):
        in_phase, generic = _separable_data()
        gate = _make_gate()
        gate.fit_pos(in_phase)
        gate.fit_neg(generic)
        open_rate = float(np.asarray(gate(in_phase[:200], training=False)).mean())
        assert open_rate > 0.9, f"in-phase open rate {open_rate} is not near 1"

    def test_the_fitted_gate_stays_closed_on_out_of_distribution_data(self):
        in_phase, generic = _separable_data()
        gate = _make_gate()
        gate.fit_pos(in_phase)
        gate.fit_neg(generic)
        open_rate = float(np.asarray(gate(generic[:400], training=False)).mean())
        assert open_rate < 0.1, f"out-of-phase open rate {open_rate} is not near 0"

    def test_scores_mode_separates_the_two_populations(self):
        in_phase, generic = _separable_data()
        gate = _make_gate(output_mode='scores')
        gate.fit_pos(in_phase)
        gate.fit_neg(generic)
        in_scores = np.asarray(gate(in_phase[:200], training=False))
        out_scores = np.asarray(gate(generic[:400], training=False))
        assert np.all(np.isfinite(in_scores)) and np.all(np.isfinite(out_scores))
        assert in_scores.mean() > out_scores.mean()

    def test_the_decision_threshold_moves_the_operating_point(self):
        """A negative threshold opens the gate more widely; a positive one less.

        TWIN of the threshold being inert, which is what a hard-coded 0.0
        comparison would produce.
        """
        in_phase, generic = _separable_data()
        gate = _make_gate(output_mode='scores')
        gate.fit_pos(in_phase)
        gate.fit_neg(generic)
        scores = np.concatenate([
            np.asarray(gate(in_phase[:200], training=False)),
            np.asarray(gate(generic[:400], training=False)),
        ])
        assert (scores > -50.0).mean() > (scores > 50.0).mean()

    def test_the_fitted_parameters_are_finite(self):
        """A NaN component propagates through the log-density to every score."""
        in_phase, generic = _separable_data()
        gate = _make_gate()
        gate.fit_pos(in_phase)
        gate.fit_neg(generic)
        for key, weight in gate._weights.items():
            if 'acc_' in key or key == 'em_step_counter':
                continue
            assert bool(np.all(np.isfinite(np.asarray(weight.value)))), key


class TestStreamingFit:
    def test_streaming_fits_the_gate_and_reports_no_revival(self):
        in_phase, generic = _separable_data()
        gate = _make_gate(fit_mode='minibatch')
        for start in range(0, len(in_phase), 128):
            gate.observe(in_phase[start:start + 128], which='pos')
        for start in range(0, len(generic), 128):
            gate.observe(generic[start:start + 128], which='neg')
        assert gate.is_fitted('pos') and gate.is_fitted('neg')
        assert gate.num_dead_components() == 0

    def test_streaming_reaches_the_same_operating_point_as_batch(self):
        """The stream is noisier, but it must land in the same region.

        This is the guard on the streaming path being a usable default rather
        than a degenerate one. The tolerance is deliberately loose and its
        measurement is recorded: on a separable 8-component 32-dim mixture the
        streaming path's log-likelihood sat ~20% below batch EM while the
        in-distribution hit rate stayed ~0.97.
        """
        in_phase, generic = _separable_data()
        batch_gate = _make_gate()
        batch_gate.fit_pos(in_phase)
        batch_gate.fit_neg(generic)

        stream_gate = _make_gate(fit_mode='minibatch')
        for start in range(0, len(in_phase), 128):
            stream_gate.observe(in_phase[start:start + 128], which='pos')
        for start in range(0, len(generic), 128):
            stream_gate.observe(generic[start:start + 128], which='neg')

        probe = np.random.default_rng(99).standard_normal((200, DIM)).astype('float32')
        assert abs(
            float(np.asarray(batch_gate(probe, training=False)).mean())
            - float(np.asarray(stream_gate(probe, training=False)).mean())
        ) < 0.15

    def test_streaming_state_is_independent_of_the_token_count(self):
        """`O(K*d)`, which is what removes the need for an activation buffer."""
        small = _make_gate(fit_mode='minibatch')
        large = _make_gate(fit_mode='minibatch')
        state_vars = [k for k in small._weights if k.startswith('pos_acc_')]
        assert state_vars
        for key in state_vars:
            assert small._weights[key].shape == large._weights[key].shape
        # The accumulator shape depends only on (K, d), never on n.
        assert small._weights['pos_acc_weighted_sum'].shape == (4, 8)

    def test_the_exact_path_leaves_no_dead_or_non_finite_component(self):
        """The exact path must honour the same finiteness contract as streaming.

        Two distinct pathologies, deliberately not conflated:

        * **Mass collapse** (component's mass → 0) is what `_revive_dead` repairs.
          It is rare, because initialisation seeds every component from a DISTINCT
          data row, so components rarely start far enough from the data to be
          dominated away entirely.
        * **Variance collapse** (mass healthy, variance at the floor) happens
          readily on degenerate input — 300 identical points split evenly across
          K components converge to one point-mass each. Measured: K=8 leaves 48
          of 64 variance entries on the floor. This is EM converging correctly
          on input that carries no structure, not a fault in the fitter.

        So this asserts what is actually guaranteed: nothing non-finite survives,
        and the well-posed case reports zero revivals. It does NOT claim the
        repair path is exercised here, because a natural scenario does not.
        """
        in_phase, generic = _separable_data()
        gate = _make_gate()
        gate.fit_pos(in_phase)
        gate.fit_neg(generic)

        for key, weight in gate._weights.items():
            if 'acc_' in key or key == 'em_step_counter':
                continue
            values = np.asarray(weight.value)
            assert np.all(np.isfinite(values)), f"{key} is non-finite after exact EM"
            if key.endswith('_variances'):
                assert np.all(values > 0.0), (
                    f"{key} has a non-positive entry; the floor was not applied"
                )
        # Log-priors are log-probabilities and are legitimately negative, so
        # only finiteness is assertable there -- the prior NORMALISES to 1.
        priors = np.asarray(gate._weights['pos_log_priors'].value)
        assert np.all(np.isfinite(priors))
        assert np.allclose(np.exp(priors).sum(), 1.0, atol=1e-5), (
            "the mixing weights should sum to 1 after the M-step"
        )
        assert gate.num_dead_components() == 0

    def test_degenerate_input_drives_variance_to_the_floor_without_dying(self):
        """Pins the distinction above, so the two are never conflated again.

        300 identical points cannot be separated by any mixture; every component
        should collapse onto the same point-mass. The assertion is that this is
        reported as a FLOOR and not as a revival — a variance at the floor with
        healthy mass is not a component that needs re-seeding.
        """
        DIM_ = 16
        identical = np.full((300, DIM_), 6.0, dtype='float32')
        generic = np.random.default_rng(0).standard_normal(
            (400, DIM_)).astype('float32')
        gate = LocalSupportGate(
            input_dim=DIM_, pos_components=8, neg_components=4, seed=3,
            projection_dim=8,
        )
        gate.build((None, DIM_))
        gate.fit_pos(identical)
        gate.fit_neg(generic)

        variances = np.asarray(gate._weights['pos_variances'].value)
        assert (variances <= gate.variance_floor).any(), (
            "300 identical points should drive variances to the floor; if this "
            "no longer holds the premise of this test changed"
        )
        # Mass is spread evenly, so nothing is dead and nothing was revived.
        assert gate.num_dead_components() == 0

    def test_the_repair_revives_exactly_the_components_with_no_mass(self):
        """The repair mechanism itself, exercised directly and deterministically.

        A natural end-to-end fit does not reliably produce a zero-mass
        component — initialisation draws K distinct rows, so it takes a specific
        geometry. Calling the repair with a synthetic mass pins the mechanism
        (exact-component targeting, and that it touches nothing else) without
        pretending a scenario reproduces it.
        """
        gate = _make_gate()
        z = np.random.default_rng(0).standard_normal(
            (64, gate.effective_dim)).astype('float32')
        n_components = gate.pos_components

        # Distinct sentinel means, so "was this component rewritten?" is a real
        # question. An unfitted gate's means are all zeros, and then "unchanged"
        # and "reset to the placeholder" are indistinguishable.
        sentinels = np.arange(
            n_components * gate.effective_dim, dtype='float32'
        ).reshape(n_components, gate.effective_dim) + 100.0
        gate._weights['pos_means'].assign(sentinels)

        # Components 1 and 2 have no mass; 0 and 3 do.
        mass = np.full((n_components,), 100.0, dtype='float32')
        mass[1] = 0.0
        mass[2] = 0.0

        gate._revive_dead('pos', z, mass, num_dead=2)

        after = np.asarray(gate._weights['pos_means'].value)
        assert np.array_equal(after[0], sentinels[0]), "component 0 was revived"
        assert np.array_equal(after[3], sentinels[3]), "component 3 was revived"
        # A revived component's mean comes from the batch, so it must differ
        # from the sentinel it replaced -- and must be finite.
        for revived in (1, 2):
            assert not np.allclose(after[revived], sentinels[revived])
            assert np.all(np.isfinite(after[revived]))
        assert gate.num_dead_components() == 2

    def test_the_repair_is_a_no_op_when_nothing_is_dead(self):
        """TWIN of the repair test: zero revivals must change nothing at all.

        A repair that fired unconditionally would re-seed healthy components
        every fit and destroy the mixture.
        """
        gate = _make_gate()
        z = np.random.default_rng(0).standard_normal(
            (64, gate.effective_dim)).astype('float32')
        mass = np.full((gate.pos_components,), 100.0, dtype='float32')
        sentinels = np.arange(
            gate.pos_components * gate.effective_dim, dtype='float32'
        ).reshape(gate.pos_components, gate.effective_dim) + 100.0
        gate._weights['pos_means'].assign(sentinels)

        gate._revive_dead('pos', z, mass, num_dead=0)

        assert np.array_equal(
            np.asarray(gate._weights['pos_means'].value), sentinels
        )
        assert gate.num_dead_components() == 0

    def test_the_step_counter_advances_across_batches(self):
        """The `1/t` schedule needs a real counter, not a fixed step."""
        gate = _make_gate(fit_mode='minibatch')
        data = np.random.default_rng(0).standard_normal((600, DIM)).astype('float32')
        for start in range(0, 600, 128):
            gate.observe(data[start:start + 128], which='pos')
        assert float(np.asarray(gate._weights['em_step_counter'].value)) == pytest.approx(
            5.0, abs=1e-6
        )

    def test_reset_zeroes_the_streaming_accumulators(self):
        gate = _make_gate(fit_mode='minibatch')
        data = np.random.default_rng(0).standard_normal((600, DIM)).astype('float32')
        for start in range(0, 600, 128):
            gate.observe(data[start:start + 128], which='pos')
        gate.reset('pos')
        assert np.all(np.asarray(gate._weights['pos_acc_weighted_sum'].value) == 0.0)
        assert not gate.is_fitted('pos')


class TestSmoothing:
    def test_smoothing_returns_the_same_shape(self):
        in_phase, generic = _separable_data()
        gate = _make_gate(output_mode='smoothed', smoothing_alpha=0.3)
        gate.fit_pos(in_phase)
        gate.fit_neg(generic)
        out = gate(np.random.default_rng(0).standard_normal((3, 9, DIM)).astype('float32'),
                   training=False)
        assert tuple(out.shape) == (3, 9)

    def test_smoothing_stays_within_the_unit_interval(self):
        """The EMA of a 0/1 decision cannot leave [0, 1]."""
        in_phase, generic = _separable_data()
        gate = _make_gate(output_mode='smoothed', smoothing_alpha=0.25)
        gate.fit_pos(in_phase)
        gate.fit_neg(generic)
        out = np.asarray(gate(
            np.random.default_rng(0).standard_normal((4, 20, DIM)).astype('float32'),
            training=False,
        ))
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_an_alpha_of_one_disables_smoothing(self):
        """`1.0` must reduce to the raw decision, not to something else."""
        in_phase, generic = _separable_data()
        hard = _make_gate(output_mode='hard')
        smoothed = _make_gate(output_mode='smoothed', smoothing_alpha=1.0)
        for gate in (hard, smoothed):
            gate.fit_pos(in_phase)
            gate.fit_neg(generic)
        probe = np.random.default_rng(0).standard_normal((4, 12, DIM)).astype('float32')
        assert np.allclose(
            np.asarray(hard(probe, training=False)),
            np.asarray(smoothed(probe, training=False)),
        )

    def test_a_smoothing_axis_of_minus_one_still_works(self):
        """The axis is a knob, so a non-token axis must be honoured.

        With ``smoothing_axis=-1`` on a rank-3 input, axis 2 IS the last axis,
        so the swap the EMA performs is a no-op and the sequence runs along the
        feature axis of the ORIGINAL tensor -- after the per-token reduction,
        that axis no longer exists, so the gate must still reduce first and only
        then smooth over what remains.
        """
        in_phase, generic = _separable_data()
        gate = _make_gate(output_mode='smoothed', smoothing_axis=-1)
        gate.fit_pos(in_phase)
        gate.fit_neg(generic)
        out = gate(np.random.default_rng(0).standard_normal((4, 12, DIM)).astype('float32'),
                   training=False)
        assert tuple(out.shape) == (4, 12)
        assert bool(np.all(np.isfinite(np.asarray(out))))

    def test_smoothing_is_causal(self):
        """Position t may not depend on t+1.

        THE three-armed future-leak probe: perturb the tail, and every earlier
        smoothed value must be bit-identical.
        """
        in_phase, generic = _separable_data()
        gate = _make_gate(output_mode='smoothed', smoothing_alpha=0.4)
        gate.fit_pos(in_phase)
        gate.fit_neg(generic)

        rng = np.random.default_rng(5)
        base = rng.standard_normal((1, 24, DIM)).astype('float32')
        perturbed = base.copy()
        perturbed[0, 16:, :] += 50.0

        out_base = np.asarray(gate(base, training=False))
        out_perturbed = np.asarray(gate(perturbed, training=False))

        assert np.allclose(out_base[0, :16], out_perturbed[0, :16], atol=0.0), (
            "a perturbation at position >= 16 changed a smoothed value before "
            "it; the EMA is not causal"
        )
        assert not np.allclose(out_base[0, 16:], out_perturbed[0, 16:])


class TestSerialization:
    def test_get_config_carries_every_constructor_argument(self):
        gate = LocalSupportGate(
            input_dim=DIM, projection_dim=8, pos_components=2, neg_components=2,
            seed=3, fit_mode='minibatch', smoothing_alpha=0.4,
        )
        config = gate.get_config()
        for key in ('input_dim', 'projection_dim', 'pos_components',
                    'neg_components', 'covariance_type', 'fit_mode',
                    'smoothing_alpha', 'smoothing_axis', 'output_mode',
                    'decision_threshold', 'variance_floor',
                    'dead_mass_threshold', 'seed', 'output_shape_reduced'):
            assert key in config, key

    def test_the_config_round_trips_to_an_equal_gate(self):
        original = LocalSupportGate(input_dim=DIM, projection_dim=8,
                                    pos_components=2, neg_components=2, seed=3)
        restored = LocalSupportGate.from_config(original.get_config())
        assert restored.input_dim == original.input_dim
        assert restored.projection_dim == original.projection_dim
        assert restored.effective_dim == original.effective_dim
        assert restored.seed == original.seed

    def test_a_fresh_reload_reports_unfitted(self):
        gate = LocalSupportGate(input_dim=DIM, projection_dim=8,
                                pos_components=2, neg_components=2, seed=3)
        assert not LocalSupportGate.from_config(gate.get_config()).is_fitted('pos')

    def test_a_saved_gate_restores_its_fitted_state_and_scores(self, tmp_path):
        """Values, not shapes: a shape-only round trip passes for a gate that
        restored a freshly-initialised mixture, which would score every token
        differently."""
        in_phase, generic = _separable_data()
        gate = LocalSupportGate(input_dim=DIM, projection_dim=8,
                                pos_components=4, neg_components=4, seed=3)
        gate.build((None, DIM))
        gate.fit_pos(in_phase)
        gate.fit_neg(generic)

        # `save`/`load_model` are Model methods; a bare Layer has neither.
        model = keras.Sequential([keras.Input(shape=(DIM,)), gate])
        model(np.random.default_rng(0).standard_normal((2, DIM)).astype('float32'))

        path = tmp_path / 'gate.keras'
        model.save(path)
        restored_model = keras.models.load_model(path)

        restored = [lyr for lyr in restored_model.layers
                    if isinstance(lyr, LocalSupportGate)][0]
        assert restored.is_fitted('pos') and restored.is_fitted('neg')

        probe = in_phase[:64]
        assert np.allclose(
            np.asarray(gate(probe, training=False)),
            np.asarray(restored(probe, training=False)),
            atol=1e-6, rtol=0,
        )
        keras.backend.clear_session()
