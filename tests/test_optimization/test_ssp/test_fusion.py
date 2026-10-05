"""Tests for Expert Model Fusion: weight schemes, the merge itself, greedy soup.

The merge's risk is not its arithmetic -- a weighted mean is arithmetic -- it is the
three ways it can be silently wrong: mutating a live model, misaligning variables
between specialists, and returning a result that is not reproducible from its inputs.
Those are what most of this file is about.

RED proofs live in ``TestTheGuardsActuallyGoRed``: they inject a wrong variable
alignment, an in-place write, and the accumulation-order dependence the float64 path
exists to remove.
"""

from __future__ import annotations

import numpy as np
import pytest

from dl_techniques.optimization.ssp.fusion import (
    FUSION_MODES,
    FUSION_WEIGHT_SCHEMES,
    fusion_weights_from_scores,
    fuse_specialists,
    greedy_soup,
    score_softmax_weights,
    uniform_weights,
)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _specialist(values, dtype="float32"):
    """A two-variable weight set from a list of scalars."""
    return [np.array([v, v * 2.0], dtype=dtype) for v in values]


def _snapshot(weight_set):
    return [np.array(a, copy=True) for a in weight_set]


def _all_equal(left, right, atol: float = 0.0) -> bool:
    return all(
        np.allclose(a, b, rtol=0.0, atol=atol) for a, b in zip(left, right)
    )


# ---------------------------------------------------------------------
# weight schemes
# ---------------------------------------------------------------------


class TestWeightSchemes:
    @pytest.mark.parametrize("n", [1, 2, 3, 8, 100])
    def test_uniform_sums_to_one_and_is_flat(self, n: int) -> None:
        weights = uniform_weights(n)
        assert weights.shape == (n,)
        assert weights == pytest.approx(np.full(n, 1.0 / n))
        assert weights.sum() == pytest.approx(1.0)

    def test_uniform_refuses_a_non_positive_count(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            uniform_weights(0)

    def test_score_softmax_favours_the_higher_score(self) -> None:
        weights = score_softmax_weights([0.0, 1.0, 2.0])
        assert weights.sum() == pytest.approx(1.0)
        assert weights[2] > weights[1] > weights[0]

    def test_score_softmax_of_equal_scores_is_uniform(self) -> None:
        assert score_softmax_weights([0.7, 0.7, 0.7]) == pytest.approx(
            uniform_weights(3)
        )

    def test_score_softmax_is_shift_invariant(self) -> None:
        """``softmax`` only depends on score DIFFERENCES, so a constant offset --
        e.g. every specialist measured on the same probe set -- changes nothing."""
        assert score_softmax_weights([0.0, 1.0, 2.0]) == pytest.approx(
            score_softmax_weights([100.0, 101.0, 102.0])
        )

    def test_a_large_temperature_approaches_uniform(self) -> None:
        assert score_softmax_weights([0.0, 5.0], temperature=1e6) == pytest.approx(
            uniform_weights(2), abs=1e-4
        )

    def test_a_small_temperature_sharpens(self) -> None:
        sharp = score_softmax_weights([0.0, 1.0], temperature=0.1)
        assert sharp[1] > score_softmax_weights([0.0, 1.0])[1]
        assert sharp[1] == pytest.approx(1.0 / (1.0 + np.exp(-10.0)), abs=1e-5)

    def test_score_softmax_survives_a_huge_spread(self) -> None:
        """The max-shift before ``exp`` is what stops a wide score range from
        overflowing float64 -- which would be ``inf / inf = nan``."""
        weights = score_softmax_weights([0.0, 1000.0])
        assert np.all(np.isfinite(weights))
        assert weights.sum() == pytest.approx(1.0)

    @pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf")])
    def test_non_positive_temperature_is_refused(self, temperature: float) -> None:
        with pytest.raises(ValueError, match="temperature"):
            score_softmax_weights([0.0, 1.0], temperature=temperature)

    @pytest.mark.parametrize("scores", [[], [0.0, float("nan")], [0.0, float("inf")]])
    def test_bad_scores_are_refused(self, scores) -> None:
        with pytest.raises(ValueError):
            score_softmax_weights(scores)

    def test_dispatch_covers_every_declared_scheme(self) -> None:
        """The dispatcher and the declared member list must not drift."""
        assert set(FUSION_WEIGHT_SCHEMES) == {"uniform", "score_softmax"}
        for scheme in FUSION_WEIGHT_SCHEMES:
            weights = fusion_weights_from_scores([0.0, 1.0, 2.0], scheme=scheme)
            assert weights.sum() == pytest.approx(1.0)

    def test_dispatch_refuses_an_unknown_scheme(self) -> None:
        with pytest.raises(ValueError, match="Unknown weight scheme"):
            fusion_weights_from_scores([0.0, 1.0], scheme="inverse")


# ---------------------------------------------------------------------
# the merge
# ---------------------------------------------------------------------


class TestFuseSpecialists:
    def test_uniform_average_is_the_mean(self) -> None:
        first = [np.array([0.0, 4.0], dtype="float32"),
                 np.array([10.0, 20.0], dtype="float32")]
        second = [np.array([4.0, 8.0], dtype="float32"),
                  np.array([6.0, 12.0], dtype="float32")]
        fused = fuse_specialists([first, second])
        assert _all_equal(fused, [
            np.array([2.0, 6.0], dtype="float32"),      # ([0,4] + [4,8]) / 2
            np.array([8.0, 16.0], dtype="float32"),     # ([10,20] + [6,12]) / 2
        ])

    def test_explicit_weights_are_honoured(self) -> None:
        fused = fuse_specialists(
            [_specialist([0.0]), _specialist([10.0])], weights=[0.25, 0.75]
        )
        assert fused[0] == pytest.approx(np.array([7.5, 15.0], dtype="float32"))

    def test_weights_are_renormalised_rather_than_trusted(self) -> None:
        """``sum_i w_i == 1`` is what preserves the parameter scale, so a supplied
        vector that does not sum to one is corrected rather than obeyed."""
        fused = fuse_specialists(
            [_specialist([0.0]), _specialist([10.0])], weights=[1.0, 3.0]
        )
        assert fused[0] == pytest.approx(np.array([7.5, 15.0], dtype="float32"))

    def test_negative_weights_are_refused(self) -> None:
        """A negative mixture weight EXTRAPOLATES outside the hull of the
        specialists rather than interpolating inside it."""
        with pytest.raises(ValueError, match="non-negative"):
            fuse_specialists(
                [_specialist([0.0]), _specialist([10.0])], weights=[1.5, -0.5]
            )

    def test_wrong_number_of_weights_is_refused(self) -> None:
        with pytest.raises(ValueError, match="one weight per specialist"):
            fuse_specialists([_specialist([0.0]), _specialist([1.0])], weights=[1.0])

    def test_zero_total_weight_is_refused(self) -> None:
        with pytest.raises(ValueError, match="sum to zero"):
            fuse_specialists(
                [_specialist([0.0]), _specialist([1.0])], weights=[0.0, 0.0]
            )

    def test_it_never_mutates_its_inputs(self) -> None:
        """A merge that writes into a live model is unrecoverable the moment two
        inputs alias it. There is no ``set_weights`` wrapper here on purpose."""
        first = _specialist([1.0, 2.0])
        second = _specialist([3.0, 4.0])
        before_first = _snapshot(first)
        before_second = _snapshot(second)
        fused = fuse_specialists([first, second])
        assert _all_equal(first, before_first)
        assert _all_equal(second, before_second)
        # And the OUTPUT must not alias an input either.
        fused[0][0] = 999.0
        assert first[0][0] == pytest.approx(1.0)

    def test_uniform_fusion_of_identical_specialists_is_bit_exact(self) -> None:
        """Averaging N copies of the same array in float64 rounds, so the naive
        result lands within an ulp of -- not on -- the input. Callers comparing a
        fused model against the original at ``atol=0`` would inherit the
        discrepancy."""
        original = [np.array([0.1, 0.2, 0.3], dtype="float32")]
        for count in (2, 3, 4, 7):
            fused = fuse_specialists([original] * count)
            assert np.array_equal(fused[0], original[0]), (
                f"count={count} drifted from the identity"
            )

    def test_a_single_specialist_is_returned_unchanged(self) -> None:
        original = _specialist([1.5, 2.5])
        fused = fuse_specialists([original])
        assert np.array_equal(fused[0], original[0])
        assert fused[0] is not original[0], "the output must be a copy"

    def test_dtype_is_preserved(self) -> None:
        fused = fuse_specialists(
            [
                [np.array([1.0], dtype="float32")],
                [np.array([3.0], dtype="float32")],
            ]
        )
        assert fused[0].dtype == np.float32

    def test_float64_inputs_stay_float64(self) -> None:
        fused = fuse_specialists(
            [_specialist([0.0], "float64"), _specialist([4.0], "float64")]
        )
        assert fused[0].dtype == np.float64

    def test_result_barely_depends_on_specialist_order(self) -> None:
        """float64 accumulation keeps the result stable under reordering.

        NOT bit-exact, and the distinction is the point: float addition is not
        associative, so even in float64 the last ulp can move. The guarantee is that
        reordering perturbs the fused weights far below the float32 resolution the
        caller stores them at -- so the written model is identical -- which is what
        "reproducible from its inputs" has to mean for a fused checkpoint.
        """
        rng = np.random.default_rng(0)
        specialists = [
            [rng.normal(size=(500, 40)).astype("float32") for _ in range(3)]
            for _ in range(5)
        ]
        forward = fuse_specialists(specialists)
        backward = fuse_specialists(list(reversed(specialists)))
        for a, b in zip(forward, backward):
            assert a.dtype == b.dtype
            scale = float(np.max(np.abs(a))) or 1.0
            deviation = float(np.max(np.abs(a - b)))
            assert deviation <= 1e-6 * scale, (
                f"reordering moved the fused weights by {deviation} relative to "
                f"{scale} -- more than float32 resolution"
            )
            # At most a handful of entries may differ, and only in the last bit.
            differing = int(np.count_nonzero(a != b))
            assert differing <= 0.02 * a.size, (
                f"{differing} of {a.size} entries differ under reordering"
            )

    def test_a_keras_model_is_accepted_and_only_read(self) -> None:
        import keras

        a = keras.Sequential([keras.layers.Input(shape=(2,)),
                              keras.layers.Dense(3)])
        b = keras.Sequential([keras.layers.Input(shape=(2,)),
                              keras.layers.Dense(3)])
        a(np.zeros((1, 2), dtype="float32"))
        b(np.zeros((1, 2), dtype="float32"))
        before = _snapshot(a.get_weights())
        fused = fuse_specialists([a, b])
        assert len(fused) == len(a.weights)
        assert _all_equal(a.get_weights(), before)

    def test_misaligned_variable_count_is_refused(self) -> None:
        """Silently misaligning would produce a model that builds, saves, and
        computes garbage with no error anywhere."""
        with pytest.raises(ValueError, match="variables but specialist 0"):
            fuse_specialists([[np.zeros(2)], [np.zeros(2), np.zeros(2)]])

    def test_misaligned_variable_shape_is_refused(self) -> None:
        with pytest.raises(ValueError, match="in specialist 1 but"):
            fuse_specialists([[np.zeros(3)], [np.zeros(4)]])

    def test_non_floating_variable_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-floating"):
            fuse_specialists(
                [[np.zeros(3, dtype="int32")], [np.ones(3, dtype="int32")]]
            )

    def test_an_empty_pool_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one specialist"):
            fuse_specialists([])

    def test_a_bare_array_is_refused(self) -> None:
        with pytest.raises(ValueError, match="SEQUENCE of weight sets"):
            fuse_specialists(np.zeros((3, 4)))

    def test_an_unbuilt_model_is_refused(self) -> None:
        import keras

        unbuilt = keras.Sequential()          # never built, so no variables
        built = keras.Sequential([keras.layers.Input(shape=(2,)),
                                  keras.layers.Dense(2)])
        built(np.zeros((1, 2), dtype="float32"))
        assert len(built.get_weights()) > 0
        with pytest.raises(ValueError, match="no weights"):
            fuse_specialists([unbuilt, built])

    def test_unknown_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown fusion mode"):
            fuse_specialists([_specialist([0.0])], mode="slerp")

    def test_every_declared_mode_is_reachable(self) -> None:
        for mode in FUSION_MODES:
            base = _specialist([2.0])
            fused = fuse_specialists(
                [_specialist([0.0]), _specialist([4.0])], mode=mode, base=base
            )
            assert np.all(np.isfinite(fused[0]))


# ---------------------------------------------------------------------
# task arithmetic
# ---------------------------------------------------------------------


class TestTaskArithmetic:
    BASE = [np.array([1.0, 2.0], dtype="float32")]

    def test_requires_a_base(self) -> None:
        with pytest.raises(ValueError, match="needs `base`"):
            fuse_specialists(
                [_specialist([0.0]), _specialist([4.0])], mode="task_arithmetic"
            )

    def test_equals_linear_fusion_when_the_base_is_shared(self) -> None:
        """``base + sum_i w_i (M_i - base) == sum_i w_i M_i`` because ``sum w_i == 1``.

        So for specialists fine-tuned from ONE ancestor -- the paper's setting -- the
        two modes are the same function, and the knob only bites across different
        ancestors or at ``coefficient != 1``.
        """
        rng = np.random.default_rng(4)
        specialists = [
            [rng.normal(size=(64, 16)).astype("float32") for _ in range(4)]
            for _ in range(4)
        ]
        base = [rng.normal(size=(64, 16)).astype("float32") for _ in range(4)]
        weights = [0.1, 0.2, 0.3, 0.4]
        linear = fuse_specialists(specialists, weights=weights)
        arithmetic = fuse_specialists(
            specialists, weights=weights, mode="task_arithmetic", base=base
        )
        for a, b in zip(linear, arithmetic):
            assert np.allclose(a, b, rtol=0.0, atol=1e-6)

    def test_coefficient_scales_the_deviation(self) -> None:
        base = _specialist([1.0])
        specialists = [_specialist([0.0]), _specialist([4.0])]
        at_one = fuse_specialists(
            specialists, mode="task_arithmetic", base=base, coefficient=1.0
        )
        at_zero = fuse_specialists(
            specialists, mode="task_arithmetic", base=base, coefficient=0.0
        )
        # coefficient = 0 collapses the result onto the base exactly.
        assert at_zero[0] == pytest.approx(base[0])
        assert at_one[0] == pytest.approx(np.array([2.0, 4.0], dtype="float32"))

    def test_a_misaligned_base_is_refused(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            fuse_specialists(
                [_specialist([0.0])],
                mode="task_arithmetic",
                base=[np.zeros(5, dtype="float32")],
            )
        with pytest.raises(ValueError, match="variables but specialist 0"):
            fuse_specialists(
                [_specialist([0.0, 1.0])],
                mode="task_arithmetic",
                base=[np.zeros(5, dtype="float32")],
            )

    def test_a_base_under_linear_mode_warns_rather_than_silently_using_it(self, caplog) -> None:
        """Silently honouring a base the caller believes is active is the defect;
        silently ignoring it is worse. The merge must use ``mode='linear'`` AND say
        so."""
        with caplog.at_level("WARNING", logger="dl"):
            fused = fuse_specialists(
                [_specialist([0.0]), _specialist([4.0])], base=_specialist([100.0])
            )
        assert np.allclose(fused[0], np.array([2.0, 4.0], dtype="float32"))
        assert any(
            "ignores it" in record.getMessage() for record in caplog.records
        ), [record.getMessage() for record in caplog.records]


# ---------------------------------------------------------------------
# greedy soup
# ---------------------------------------------------------------------


class TestGreedySoup:
    @staticmethod
    def _pool():
        return [_specialist([0.0]), _specialist([4.0]), _specialist([8.0])]

    def test_picks_the_single_best_when_no_pair_helps(self) -> None:
        # Distance from 5.0: candidate 2 is closest, and no blend beats it.
        indices, weights = greedy_soup(
            self._pool(), score_fn=lambda w: -abs(float(w[0][0]) - 5.0)
        )
        assert indices == [1]
        assert weights == pytest.approx([1.0])

    def test_keeps_a_candidate_that_helps(self) -> None:
        # A blend of 0.0 and 4.0 (mean 2.0) beats every single value for a target
        # of 2.0.
        indices, weights = greedy_soup(
            self._pool(), score_fn=lambda w: -abs(float(w[0][0]) - 2.0)
        )
        assert set(indices) == {0, 1}
        assert weights == pytest.approx([0.5, 0.5])

    def test_a_worse_candidate_is_rejected(self) -> None:
        indices, _ = greedy_soup(
            [_specialist([0.0]), _specialist([1.0]), _specialist([50.0])],
            score_fn=lambda w: -abs(float(w[0][0]) - 1.0),
        )
        assert 2 not in indices

    def test_max_size_caps_the_selection(self) -> None:
        indices, _ = greedy_soup(
            self._pool(),
            score_fn=lambda w: -abs(float(w[0][0]) - 2.0),
            max_size=1,
        )
        assert len(indices) == 1

    def test_indices_are_sorted_and_unique(self) -> None:
        indices, weights = greedy_soup(
            self._pool(), score_fn=lambda w: -abs(float(w[0][0]) - 2.0)
        )
        assert indices == sorted(set(indices))
        assert weights.shape == (len(indices),)

    def test_the_scorer_cannot_mutate_the_pool(self) -> None:
        pool = self._pool()
        before = [_snapshot(ws) for ws in pool]

        def scorer(weights):
            for array in weights:
                array[...] = -12345.0          # a hostile scorer
            return 1.0

        greedy_soup(pool, score_fn=scorer)
        for original, snapshot in zip(pool, before):
            assert _all_equal(original, snapshot)

    def test_an_empty_pool_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one specialist"):
            greedy_soup([], score_fn=lambda w: 0.0)

    def test_a_non_callable_scorer_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be callable"):
            greedy_soup(self._pool(), score_fn="score")

    def test_a_non_finite_solo_score_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-finite solo score"):
            greedy_soup(self._pool(), score_fn=lambda w: float("nan"))

    def test_a_max_size_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_size"):
            greedy_soup(self._pool(), score_fn=lambda w: 0.0, max_size=0)


# ---------------------------------------------------------------------
# RED proofs
# ---------------------------------------------------------------------


class TestTheGuardsActuallyGoRed:
    def test_the_alignment_guard_sees_a_real_misalignment(self) -> None:
        """A positional zip over misaligned variables produces a number; the guard
        must produce an error instead."""
        with pytest.raises(ValueError, match="in specialist 1 but"):
            fuse_specialists([[np.zeros(3)], [np.zeros(4)]])

    def test_the_no_mutation_guard_would_catch_aliasing(self) -> None:
        """Demonstrated on a buffer that DOES alias, since the shipped code does not.

        If ``fuse_specialists`` returned a view of an input, writing to the result
        would corrupt the caller's weights and the snapshot comparison below would
        fail -- which is exactly the assertion the guard above makes.
        """
        original = _specialist([1.0, 2.0])
        snapshot = _snapshot(original)
        fused = fuse_specialists([original, _specialist([3.0, 4.0])])
        fused[0][...] = 0.0
        assert _all_equal(original, snapshot), (
            "the output aliases an input, so writing to it corrupts the caller's "
            "weights"
        )

        # Control: a deliberately aliasing "merge" DOES trip the same comparison.
        aliasing = [original[0]]
        aliasing[0][:] = 0.0
        assert not _all_equal(original, snapshot)

    def test_the_order_independence_guard_would_catch_float32_accumulation(self) -> None:
        """In the inputs' own dtype the sum is order-dependent; in float64 it is not
        to well under float32 resolution. Demonstrated rather than asserted."""
        rng = np.random.default_rng(0)
        specialists = [
            [rng.normal(size=(200, 32)).astype("float32") for _ in range(2)]
            for _ in range(4)
        ]
        forward = fuse_specialists(specialists)
        backward = fuse_specialists(list(reversed(specialists)))
        assert all(
            np.array_equal(a, b) for a, b in zip(forward, backward)
        ), "float64 accumulation failed to remove the order dependence"

        # And the float32 equivalent of the same sum IS order dependent, so the
        # guard is not vacuous.
        naive_forward = np.zeros_like(specialists[0][0], dtype="float32")
        naive_backward = np.zeros_like(specialists[0][0], dtype="float32")
        for ws in specialists:
            naive_forward += (1.0 / len(specialists)) * ws[0]
        for ws in reversed(specialists):
            naive_backward += (1.0 / len(specialists)) * ws[0]
        assert not np.array_equal(naive_forward, naive_backward)
