"""Tests for ``dl_techniques.utils.conformal_classification`` (APS prediction sets)."""

import numpy as np
import pytest

from dl_techniques.utils.conformal_classification import (
    aps_scores,
    calibrate_aps,
    evaluate_sets,
    label_scores,
    predict_sets,
    repeated_split_coverage,
    softmax,
)


def _synthetic(n=4000, c=10, seed=0):
    """Exchangeable samples from an overconfident classifier."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, c)) * 3.0
    true_p = softmax(logits / 2.0)  # the world is less sharp than the model says
    labels = np.array([rng.choice(c, p=p) for p in true_p])
    return logits, labels


class TestScore:
    def test_the_research_note_example(self):
        # pi = [0.1, 0.6, 0.3], y = 3 (index 2) has rank 2: 0.6 + 0.5 * 0.3 = 0.75
        logits = np.log(np.array([[0.1, 0.6, 0.3]]))
        score = label_scores(logits, np.array([0.5]))[0, 2]
        assert score == pytest.approx(0.75)

    def test_every_label_gets_its_own_score_in_class_order(self):
        logits = np.log(np.array([[0.1, 0.6, 0.3]]))
        got = label_scores(logits, np.array([1.0]))[0]
        assert got == pytest.approx([1.0, 0.6, 0.9])

    def test_ties_are_ordered_by_class_index(self):
        got = label_scores(np.zeros((1, 3)), np.array([1.0]))[0]
        assert got == pytest.approx([1 / 3, 2 / 3, 1.0])

    def test_many_ties_are_still_ordered_by_class_index(self):
        n = 400
        got = label_scores(np.zeros((1, n)), np.array([1.0]))[0]
        assert got == pytest.approx((np.arange(n) + 1) / n)

    def test_aps_scores_picks_the_true_label(self):
        logits = np.log(np.array([[0.1, 0.6, 0.3], [0.1, 0.6, 0.3]]))
        got = aps_scores(logits, np.array([1, 0]), randomized=False)
        assert got == pytest.approx([0.6, 1.0])

    def test_softmax_is_stable_for_huge_logits(self):
        p = softmax(np.array([[1e4, 0.0, -1e4]]))
        assert np.isfinite(p).all() and p[0, 0] == pytest.approx(1.0)


class TestSets:
    def test_sets_grow_with_tau(self):
        logits, _ = _synthetic(n=200)
        small = predict_sets(logits, 0.5, randomized=False)
        big = predict_sets(logits, 0.9, randomized=False)
        assert (small <= big).all() and small.sum() < big.sum()

    def test_a_label_whose_score_equals_tau_is_in_the_set(self):
        # [0.5, 0.5] at u = 1: scores are [0.5, 1.0]; tau = 0.5 keeps exactly class 0.
        got = predict_sets(np.zeros((1, 2)), 0.5, randomized=False)
        assert got.tolist() == [[True, False]]

    def test_a_tau_above_the_total_mass_gives_every_label(self):
        logits, _ = _synthetic(n=50)
        assert predict_sets(logits, 1.0 + 1e-9, randomized=False).all()

    def test_for_one_tau_the_deterministic_sets_sit_inside_the_randomized_ones(self):
        # u <= 1 only lowers a label's score, so more labels pass at the same tau.
        logits, _ = _synthetic(n=300)
        rand = predict_sets(logits, 0.8, randomized=True, seed=3)
        det = predict_sets(logits, 0.8, randomized=False)
        assert (det <= rand).all() and det.sum() < rand.sum()

    def test_deterministic_ignores_the_seed_and_random_follows_it(self):
        logits, _ = _synthetic(n=300)
        assert np.array_equal(predict_sets(logits, 0.8, False, seed=1),
                              predict_sets(logits, 0.8, False, seed=2))
        a = predict_sets(logits, 0.8, True, seed=1)
        assert np.array_equal(a, predict_sets(logits, 0.8, True, seed=1))
        assert not np.array_equal(a, predict_sets(logits, 0.8, True, seed=2))


class TestCalibration:
    def test_too_few_samples_give_an_infinite_tau_and_full_sets(self):
        logits, labels = _synthetic(n=5, c=4)
        tau = calibrate_aps(logits, labels, alpha=0.1, randomized=False)
        assert tau == np.inf
        assert predict_sets(logits, tau, randomized=False).all()

    @pytest.mark.parametrize("randomized", [True, False])
    def test_calibrated_coverage_meets_the_target(self, randomized):
        logits, labels = _synthetic()
        out = repeated_split_coverage(logits, labels, alpha=0.1, repeats=20,
                                      seed=0, randomized=randomized)
        assert out["coverage_mean"] >= 0.9 - 0.01
        assert out["n_calibration"] + out["n_evaluation"] == len(labels)

    def test_a_shrunk_tau_misses_the_target(self):
        # RED proof of the check above: halving tau must fall well below 90 %.
        logits, labels = _synthetic()
        half = len(labels) // 2
        tau = calibrate_aps(logits[:half], labels[:half], 0.1, randomized=False)
        sets = predict_sets(logits[half:], 0.5 * tau, randomized=False)
        assert evaluate_sets(logits[half:], labels[half:], sets)["coverage"] < 0.85

    def test_the_finite_sample_correction_is_applied(self):
        # n = 9, alpha = 0.1: k = ceil(10 * 0.9) = 9, the largest score, not the 8th.
        logits = np.log(np.tile([[0.7, 0.2, 0.1]], (9, 1)))
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 1, 2])
        scores = np.sort(aps_scores(logits, labels, randomized=False))
        assert calibrate_aps(logits, labels, 0.1, randomized=False) == scores[-1]


class TestEvaluate:
    def test_hand_case(self):
        logits = np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7], [0.5, 0.4, 0.1]]))
        labels = np.array([0, 1, 2])
        sets = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]], dtype=bool)
        got = evaluate_sets(logits, labels, sets)
        assert got["coverage"] == pytest.approx(2 / 3)
        assert got["mean_size"] == pytest.approx(5 / 3)
        assert got["median_size"] == 2.0
        assert got["empty_rate"] == 0.0
        assert got["size_given_covered"] == pytest.approx(1.5)
        assert got["class_coverage_min"] == 0.0
        assert got["accuracy"] == pytest.approx(1 / 3)

    def test_empty_sets_are_counted(self):
        logits, labels = _synthetic(n=20, c=3)
        sets = np.zeros((20, 3), dtype=bool)
        got = evaluate_sets(logits, labels, sets)
        assert got["empty_rate"] == 1.0 and got["coverage"] == 0.0
        assert np.isnan(got["size_given_covered"])


class TestRefusals:
    def test_a_label_out_of_range(self):
        with pytest.raises(ValueError, match="labels must be in"):
            aps_scores(np.zeros((2, 3)), np.array([0, 3]))

    def test_logits_must_be_2d(self):
        with pytest.raises(ValueError, match="logits must be"):
            predict_sets(np.zeros((2, 3, 1)), 0.5)

    def test_label_shape(self):
        with pytest.raises(ValueError, match="labels must be"):
            aps_scores(np.zeros((2, 3)), np.array([0]))

    def test_repeats_and_size(self):
        logits, labels = _synthetic(n=10, c=3)
        with pytest.raises(ValueError, match="repeats"):
            repeated_split_coverage(logits, labels, 0.1, repeats=0)
        with pytest.raises(ValueError, match="at least 2"):
            repeated_split_coverage(logits[:1], labels[:1], 0.1)

    def test_sets_shape_in_evaluate(self):
        with pytest.raises(ValueError, match="sets must be"):
            evaluate_sets(np.zeros((2, 3)), np.array([0, 1]), np.zeros((2, 2), bool))
