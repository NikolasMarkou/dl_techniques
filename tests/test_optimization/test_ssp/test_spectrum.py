"""Tests for the Spectrum phase: pool profiling, specialist selection, sampling.

Two things this file is really about.

**The selection has to beat the naive rule, and say so when it does not.**
``spectrum_gain`` is the amount by which per-subdomain argmax beats the best single
checkpoint; it is ``0.0`` exactly when the extra machinery bought nothing. A guard
that only ever sees a positive gain would never notice a selection function that
returns the global argmax every time.

**The sampling weights must reuse the max-entropy criterion, not reimplement it.**
``spectrum_sampling_weights(mode="max_entropy")`` calls
``signal.max_entropy_weight`` directly, so the data-side and checkpoint-side criteria
cannot drift. The guard pins that by value: the sampler and the Signal half must agree
to float32 precision on the same inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from dl_techniques.metrics.pass_at_k import pass_at_k, pass_at_k_curve
from dl_techniques.optimization.ssp.signal import binary_entropy, max_entropy_weight
from dl_techniques.optimization.ssp.spectrum import (
    SAMPLING_MODES,
    select_specialists,
    spectrum_profile,
    spectrum_sampling_weights,
)

FLOAT32_ATOL = 1e-6


def _np(value) -> np.ndarray:
    import keras

    return np.asarray(keras.ops.convert_to_numpy(value), dtype=np.float64)


# ---------------------------------------------------------------------
# profiling
# ---------------------------------------------------------------------


class TestSpectrumProfile:
    OUTCOMES = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )

    def test_reports_pass_at_1_and_pass_at_k_together(self) -> None:
        """SSP's central claim is that breadth and single-shot accuracy are jointly
        maximisable. It is only checkable if both numbers are present, which is why
        reporting one is how the claim gets mistaken for an established result."""
        profile = spectrum_profile(self.OUTCOMES)
        assert profile["pass_at_1"] == pytest.approx(5.0 / 24.0)
        assert profile["pass_at_k"] == pytest.approx(2.0 / 3.0)

    def test_the_headline_k_is_the_widest_one_asked_for(self) -> None:
        profile = spectrum_profile(self.OUTCOMES)
        assert profile["pass_at_k"] == profile["pass_at_k_curve"][max(
            profile["pass_at_k_curve"]
        )]

    def test_a_narrower_ladder_narrows_the_headline(self) -> None:
        wide = spectrum_profile(self.OUTCOMES)
        narrow = spectrum_profile(self.OUTCOMES, ks=[1, 2])
        assert narrow["pass_at_k"] == pytest.approx(
            narrow["pass_at_k_curve"][2]
        )
        assert narrow["pass_at_k"] <= wide["pass_at_k"]

    def test_the_curve_is_monotone(self) -> None:
        curve = spectrum_profile(self.OUTCOMES)["pass_at_k_curve"]
        values = [curve[k] for k in sorted(curve)]
        assert all(b >= a - FLOAT32_ATOL for a, b in zip(values, values[1:]))

    def test_reports_the_shape_and_breadth_diagnostics(self) -> None:
        profile = spectrum_profile(self.OUTCOMES)
        assert profile["n_problems"] == 3
        assert profile["n_samples"] == 8
        assert profile["solved_fraction"] == pytest.approx(2.0 / 3.0)
        assert profile["mean_solved_count"] == pytest.approx(5.0 / 3.0)

    def test_outcome_entropy_separates_a_uniform_pool_from_a_polarised_one(self) -> None:
        """The diagnostic pass@k cannot supply. Every problem split evenly has high
        outcome entropy; a pool where each problem is all-or-nothing has none, even
        at the same pass@k."""
        uniform = np.tile(np.array([[1, 0, 1, 0]], dtype=float), (8, 1))
        polarised = np.vstack([
            np.ones((4, 4)), np.zeros((4, 4))
        ]).astype(float)
        assert spectrum_profile(uniform)["outcome_entropy"] > \
            spectrum_profile(polarised)["outcome_entropy"] + 1e-3

    def test_outcome_entropy_is_the_mean_binary_entropy(self) -> None:
        outcomes = np.array([[1, 0, 1, 0], [0, 0, 0, 0]], dtype=float)
        profile = spectrum_profile(outcomes)
        rates = np.array([0.5, 0.0])
        assert profile["outcome_entropy"] == pytest.approx(
            float(np.mean(_np(binary_entropy(rates)))), abs=FLOAT32_ATOL
        )

    def test_per_problem_is_opt_in(self) -> None:
        assert "per_problem" not in spectrum_profile(self.OUTCOMES)
        profile = spectrum_profile(self.OUTCOMES, include_per_problem=True)
        assert profile["per_problem"].shape == (3,)

    def test_agrees_with_the_canonical_metric(self) -> None:
        profile = spectrum_profile(self.OUTCOMES)
        assert profile["pass_at_k"] == pytest.approx(
            pass_at_k(self.OUTCOMES, k=8), abs=FLOAT32_ATOL
        )

    def test_plug_in_estimator_flows_through(self) -> None:
        profile = spectrum_profile(self.OUTCOMES, estimator="plug_in")
        assert profile["pass_at_k"] == pytest.approx(2.0 / 3.0)

    def test_a_non_2d_matrix_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be 2-D"):
            spectrum_profile(np.zeros(4))

    def test_an_unknown_estimator_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown estimator"):
            spectrum_profile(self.OUTCOMES, estimator="biased")


# ---------------------------------------------------------------------
# specialist selection
# ---------------------------------------------------------------------


class TestSelectSpecialists:
    # One checkpoint is best overall but NOT best in any single subdomain, which is
    # the case a global argmax gets wrong.
    SCORES = np.array(
        [
            [0.10, 0.90, 0.50],
            [0.80, 0.20, 0.40],
            [0.30, 0.40, 0.95],
        ]
    )

    def test_selects_the_argmax_per_subdomain(self) -> None:
        selection = select_specialists(self.SCORES)
        assert selection["checkpoint_index"].tolist() == [1, 0, 2]

    def test_beats_the_global_argmax(self) -> None:
        selection = select_specialists(self.SCORES)
        per_subdomain_mean = float(np.mean(self.SCORES[[1, 0, 2], np.arange(3)]))
        assert selection["spectrum_gain"] == pytest.approx(
            per_subdomain_mean - selection["best_overall_score"]
        )
        assert selection["spectrum_gain"] > 0.0

    def test_the_gain_is_zero_when_one_checkpoint_dominates(self) -> None:
        """The honest measurement: if the naive rule was already optimal, the
        diversity-aware machinery bought nothing and ``spectrum_gain`` says so."""
        dominated = np.array([[0.9, 0.8, 0.7], [0.5, 0.5, 0.5], [0.1, 0.2, 0.3]])
        selection = select_specialists(dominated)
        assert selection["spectrum_gain"] == pytest.approx(0.0, abs=1e-12)
        assert selection["checkpoint_index"].tolist() == [0, 0, 0]
        assert selection["best_overall_index"] == 0

    def test_the_gain_is_never_negative(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(40):
            scores = rng.random((5, 4))
            assert select_specialists(scores)["spectrum_gain"] >= -1e-12

    def test_selected_scores_are_the_column_maxima(self) -> None:
        selection = select_specialists(self.SCORES)
        for column, value in enumerate(self.SCORES.T):
            assert selection["selected_scores"][column] == pytest.approx(
                value.max()
            )

    def test_labels_are_keyed_by_subdomain(self) -> None:
        selection = select_specialists(self.SCORES, subdomains=["alg", "geo", "calc"])
        assert selection["checkpoint_by_subdomain"] == {
            "alg": 1, "geo": 0, "calc": 2
        }
        assert selection["subdomains"] == ("alg", "geo", "calc")

    def test_default_labels_are_generated(self) -> None:
        selection = select_specialists(self.SCORES)
        assert selection["subdomains"] == ("domain_0", "domain_1", "domain_2")

    def test_ties_go_to_the_lowest_index(self) -> None:
        """Two checkpoints measured identically. Selection must not depend on dict
        ordering or on which was scanned first, or the choice is not re-derivable
        from a saved score matrix."""
        tied = np.array([[0.5, 0.9], [0.5, 0.4], [0.2, 0.8]])
        selection = select_specialists(tied)
        # Column 0: rows 0 and 1 tie at 0.5 -> index 0. Column 1: 0.9 is unique.
        assert selection["checkpoint_index"].tolist() == [0, 0]
        # And the tie is stable across repeated evaluation, not a scan-order
        # artefact.
        assert select_specialists(tied)["checkpoint_index"].tolist() == [0, 0]

    def test_a_single_subdomain_is_the_degenerate_case(self) -> None:
        """A ``(n_checkpoints,)`` vector promotes to ``(n_checkpoints, 1)``, so the
        one-subdomain case is a spelling rather than a special case -- and with one
        column there is nothing for per-subdomain selection to beat."""
        selection = select_specialists(np.array([0.1, 0.9, 0.5]))
        assert selection["checkpoint_index"].tolist() == [1]
        assert selection["spectrum_gain"] == pytest.approx(0.0)

    def test_a_one_dimensional_matrix_is_promoted(self) -> None:
        assert select_specialists(np.array([0.1, 0.9]))["checkpoint_index"].tolist() == [1]

    def test_a_wrong_label_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="one label per column"):
            select_specialists(self.SCORES, subdomains=["only_one"])

    def test_a_non_finite_score_is_refused(self) -> None:
        """A NaN score would silently lose the argmax rather than raise."""
        scores = self.SCORES.copy()
        scores[1, 0] = np.nan
        with pytest.raises(ValueError, match="must be finite"):
            select_specialists(scores)

    @pytest.mark.parametrize("shape", [(0, 3), (3, 0)])
    def test_empty_axes_are_refused(self, shape) -> None:
        with pytest.raises(ValueError, match="non-zero extent"):
            select_specialists(np.zeros(shape))

    def test_a_non_2d_matrix_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be 2-D"):
            select_specialists(np.zeros((2, 2, 2)))


# ---------------------------------------------------------------------
# sampling weights
# ---------------------------------------------------------------------


class TestSpectrumSamplingWeights:
    COVERAGE = np.array([0.0, 0.5, 1.0, 0.5])

    def test_every_mode_returns_a_distribution(self) -> None:
        for mode in SAMPLING_MODES:
            weights = spectrum_sampling_weights(self.COVERAGE, mode=mode, lam=2.0)
            assert weights.shape == (4,)
            assert np.all(weights >= 0.0)
            assert weights.sum() == pytest.approx(1.0)

    def test_max_entropy_peaks_in_the_middle(self) -> None:
        """The whole point. ``proportional`` up-weights what the model already solves
        best; ``inverse`` points at what it cannot reach. Only the middle peaks."""
        weights = spectrum_sampling_weights(self.COVERAGE, mode="max_entropy", lam=2.0)
        assert weights[1] == weights[3]
        assert weights[1] > weights[0]
        assert weights[1] > weights[2]

    def test_proportional_is_the_documented_anti_criterion(self) -> None:
        """Pinned because it is the intuitive choice and it is wrong: it up-weights
        coverage the model has already maxed out."""
        weights = spectrum_sampling_weights(self.COVERAGE, mode="proportional")
        assert weights[2] == pytest.approx(0.5)          # coverage 1.0 gets half
        assert weights[0] == pytest.approx(0.0)          # coverage 0.0 gets none

    def test_inverse_points_at_the_unreachable(self) -> None:
        weights = spectrum_sampling_weights(self.COVERAGE, mode="inverse")
        assert weights[0] == pytest.approx(0.5)
        assert weights[2] == pytest.approx(0.0)

    def test_uniform_ignores_coverage(self) -> None:
        assert spectrum_sampling_weights(self.COVERAGE, mode="uniform") == pytest.approx(
            np.full(4, 0.25)
        )

    def test_max_entropy_matches_the_signal_half_exactly(self) -> None:
        """The criterion is SHARED, not reimplemented. If this drifts, the checkpoint
        selection and the training-set sampler quietly disagree about what "broad"
        means."""
        coverage = np.linspace(0.0, 1.0, 11)
        for lam in (0.5, 2.0, 5.0):
            sampled = spectrum_sampling_weights(coverage, mode="max_entropy", lam=lam)
            shared = _np(max_entropy_weight(coverage, lam=lam))
            expected = shared / shared.sum()
            assert sampled == pytest.approx(expected, abs=FLOAT32_ATOL)

    def test_a_larger_lambda_sharpens(self) -> None:
        soft = spectrum_sampling_weights(self.COVERAGE, mode="max_entropy", lam=0.5)
        sharp = spectrum_sampling_weights(self.COVERAGE, mode="max_entropy", lam=8.0)
        assert sharp[1] > soft[1]
        assert sharp[0] < soft[0]

    def test_lambda_zero_flattens_to_uniform(self) -> None:
        assert spectrum_sampling_weights(
            self.COVERAGE, mode="max_entropy", lam=0.0
        ) == pytest.approx(np.full(4, 0.25))

    def test_p0_moves_the_peak(self) -> None:
        coverage = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        weights = spectrum_sampling_weights(coverage, mode="max_entropy", p0=0.75)
        assert int(np.argmax(weights)) == 3

    def test_a_degenerate_p0_is_refused(self) -> None:
        with pytest.raises(ValueError, match="strictly inside"):
            spectrum_sampling_weights(
                self.COVERAGE, mode="max_entropy", p0=0.0
            )

    def test_all_zero_coverage_cannot_be_normalised(self) -> None:
        """A real signal, not a degenerate input: nothing has been solved, so
        coverage carries no ordering to express."""
        with pytest.raises(ValueError, match="no distribution to normalise"):
            spectrum_sampling_weights(np.zeros(4), mode="proportional")

    def test_an_unknown_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown sampling mode"):
            spectrum_sampling_weights(self.COVERAGE, mode="argmax")

    def test_a_non_1d_coverage_vector_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be 1-D"):
            spectrum_sampling_weights(np.zeros((2, 2)))

    def test_empty_coverage_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            spectrum_sampling_weights(np.array([]))

    @pytest.mark.parametrize("bad", [[-0.1, 0.5], [0.5, 1.1]])
    def test_out_of_unit_interval_coverage_is_refused(self, bad) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            spectrum_sampling_weights(np.array(bad))

    def test_non_finite_coverage_is_refused(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            spectrum_sampling_weights(np.array([0.0, np.nan]))


# ---------------------------------------------------------------------
# RED proofs
# ---------------------------------------------------------------------


class TestTheGuardsActuallyGoRed:
    def test_the_gain_metric_alone_cannot_detect_a_degenerate_selection(self) -> None:
        """``spectrum_gain`` is a WEAK discriminator, and this is the demonstration.

        A ``select_specialists`` that returned the best-overall checkpoint for every
        subdomain yields a gain of exactly ``0.0`` -- identical to a legitimate
        "one checkpoint dominates" result, because the best-overall row's mean IS
        the mean of its own per-column values. So a guard on the gain alone would
        pass an implementation that ignores subdomains entirely.

        The statistic that does discriminate is the COUNT of subdomains where the
        chosen checkpoint is a true column argmax: 1 of 3 for the degenerate answer,
        3 of 3 for the real one. That is why this file pins the indices as well as
        the gain.
        """
        scores = TestSelectSpecialists.SCORES
        best_overall = int(np.argmax(scores.mean(axis=1)))
        degenerate = [best_overall] * scores.shape[1]
        correct = [1, 0, 2]
        assert degenerate != correct

        def gain_of(indices) -> float:
            return float(np.mean(scores[indices, np.arange(3)])) - \
                float(scores.mean(axis=1).max())

        # Both are 0.0 here, so the gain cannot tell them apart.
        assert gain_of(degenerate) == pytest.approx(0.0, abs=1e-12)
        assert gain_of(correct) > 0.0

        def hits(indices) -> int:
            columns = np.arange(scores.shape[1])
            return int(sum(
                scores[index, column] == pytest.approx(scores[:, column].max())
                for index, column in zip(indices, columns)
            ))

        assert hits(degenerate) == 1
        assert hits(correct) == 3

    def test_the_shared_criterion_guard_sees_a_drifted_criterion(self) -> None:
        """If the sampler carried its own copy of the weighting, a perturbation
        would separate the two."""
        coverage = np.linspace(0.0, 1.0, 11)
        shared = spectrum_sampling_weights(coverage, mode="max_entropy", lam=2.0)
        drifted = np.abs(coverage - 0.5)                    # a plausible wrong rule
        drifted = drifted / drifted.sum()
        assert not np.allclose(shared, drifted, atol=1e-3)

    def test_the_peak_guard_sees_a_proportional_implementation(self) -> None:
        coverage = TestSpectrumSamplingWeights.COVERAGE
        proportional = spectrum_sampling_weights(coverage, mode="proportional")
        max_entropy = spectrum_sampling_weights(coverage, mode="max_entropy", lam=2.0)
        assert int(np.argmax(proportional)) == 2, "proportional peaks at coverage 1.0"
        assert int(np.argmax(max_entropy)) in (1, 3), (
            "max_entropy must peak at coverage 0.5, so the two must disagree"
        )
