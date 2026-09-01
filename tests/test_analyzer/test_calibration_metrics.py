"""
Tests for calibration_metrics module.

Covers: ECE, adaptive ECE, MCE, Brier score, reliability data, entropy.
"""

import numpy as np
import pytest

from dl_techniques.analyzer.calibration_metrics import (
    compute_ece,
    compute_adaptive_ece,
    compute_mce,
    compute_brier_score,
    compute_brier_score_decomposition,
    compute_reliability_data,
    compute_prediction_entropy_stats,
)


class TestECE:
    """Tests for Expected Calibration Error."""

    def test_perfectly_calibrated_model(self):
        """A perfectly calibrated model should have ECE ≈ 0."""
        np.random.seed(42)
        n = 1000
        # Create predictions that are perfectly calibrated
        y_true = np.random.randint(0, 2, n)
        y_prob = np.zeros((n, 2))
        for i in range(n):
            if y_true[i] == 1:
                y_prob[i] = [0.2, 0.8]
            else:
                y_prob[i] = [0.8, 0.2]

        ece = compute_ece(y_true, y_prob, n_bins=10)
        assert ece < 0.3  # Not perfectly 0 due to binning, but should be low

    def test_overconfident_model(self):
        """An overconfident wrong model should have high ECE."""
        n = 100
        y_true = np.zeros(n, dtype=int)  # All class 0
        y_prob = np.zeros((n, 2))
        y_prob[:, 1] = 0.99  # Predicts class 1 with 99% confidence (all wrong)
        y_prob[:, 0] = 0.01

        ece = compute_ece(y_true, y_prob, n_bins=10)
        assert ece > 0.5

    def test_ece_is_non_negative(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 5, 200)
        y_prob = np.random.dirichlet(np.ones(5), 200)
        ece = compute_ece(y_true, y_prob, n_bins=15)
        assert ece >= 0


class TestAdaptiveECE:
    """Tests for Adaptive ECE with equal-mass bins."""

    def test_handles_all_samples(self):
        """Adaptive ECE should not drop remainder samples."""
        np.random.seed(42)
        n = 103  # Not evenly divisible by 15
        y_true = np.random.randint(0, 3, n)
        y_prob = np.random.dirichlet(np.ones(3), n)

        aece = compute_adaptive_ece(y_true, y_prob, n_bins=15)
        assert aece >= 0

    def test_aece_is_non_negative(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.dirichlet(np.ones(2), 100)
        aece = compute_adaptive_ece(y_true, y_prob, n_bins=10)
        assert aece >= 0


class TestMCE:
    """Tests for Maximum Calibration Error."""

    def test_mce_at_least_as_large_as_ece(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 3, 200)
        y_prob = np.random.dirichlet(np.ones(3), 200)
        ece = compute_ece(y_true, y_prob, n_bins=10)
        mce = compute_mce(y_true, y_prob, n_bins=10)
        assert mce >= ece - 1e-10


class TestBrierScore:
    """Tests for Brier Score."""

    def test_perfect_predictions(self):
        y_true_oh = np.eye(3)[[0, 1, 2]]
        y_prob = np.eye(3)[[0, 1, 2]]
        bs = compute_brier_score(y_true_oh, y_prob)
        assert bs == pytest.approx(0.0, abs=1e-10)

    def test_worst_predictions(self):
        """All-wrong predictions should have high Brier score."""
        y_true_oh = np.array([[1, 0], [0, 1]])
        y_prob = np.array([[0.0, 1.0], [1.0, 0.0]])
        bs = compute_brier_score(y_true_oh, y_prob)
        assert bs == pytest.approx(2.0)

    def test_brier_decomposition_sums(self):
        """The identity the function is NAMED for: BS = Rel - Res + Unc.

        REPAIRED (plan-2026-09-01T225724-e79ad4bd step 15): this test was named
        for the identity and asserted only `>= 0` on each of the three terms — it
        never formed the sum, so it passed while the identity failed by 30-65%.
        """
        np.random.seed(42)
        y_true = np.random.randint(0, 3, 300)
        y_prob = np.random.dirichlet(np.ones(3), 300)
        decomp = compute_brier_score_decomposition(y_true, y_prob, n_bins=10)

        assert decomp['reliability'] >= 0
        assert decomp['resolution'] >= 0
        assert decomp['uncertainty'] >= 0

        recombined = (
            decomp['reliability'] - decomp['resolution'] + decomp['uncertainty'])
        assert recombined == pytest.approx(decomp['brier_score'], abs=1e-12), (
            f"the decomposition does not decompose: "
            f"Rel - Res + Unc = {recombined!r} vs BS = {decomp['brier_score']!r}"
        )


class TestReliabilityData:
    """Tests for reliability diagram data."""

    def test_returns_correct_structure(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.dirichlet(np.ones(2), 100)
        data = compute_reliability_data(y_true, y_prob, n_bins=5)

        assert 'bin_centers' in data
        assert 'bin_accuracies' in data
        assert 'bin_confidences' in data
        assert 'bin_counts' in data
        assert len(data['bin_centers']) == 5


class TestPredictionEntropy:
    """Tests for prediction entropy statistics."""

    def test_confident_predictions_low_entropy(self):
        """Near-certain predictions should have low entropy."""
        y_prob = np.array([[0.99, 0.01], [0.01, 0.99], [0.98, 0.02]])
        stats = compute_prediction_entropy_stats(y_prob)
        assert stats['mean_entropy'] < 0.1

    def test_uncertain_predictions_high_entropy(self):
        """Uniform predictions should have high entropy."""
        y_prob = np.array([[0.5, 0.5], [0.5, 0.5]])
        stats = compute_prediction_entropy_stats(y_prob)
        assert stats['mean_entropy'] > 0.5

    def test_returns_all_fields(self):
        y_prob = np.array([[0.7, 0.3]])
        stats = compute_prediction_entropy_stats(y_prob)
        for key in ['entropy', 'mean_entropy', 'std_entropy', 'median_entropy',
                     'max_entropy', 'min_entropy']:
            assert key in stats


# =====================================================================
# S7 — the Brier decomposition must actually decompose (plan step 15)
# =====================================================================

def _top1_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Top-1 Brier score, derived here independently of the module under test.

    The binary outcome is "was the top-1 prediction correct" and the forecast is
    the top-1 confidence. This is the scalar the decomposition must reproduce.
    """
    scores = np.max(y_prob, axis=1)
    outcomes = (np.argmax(y_prob, axis=1) == y_true).astype(float)
    return float(np.mean((scores - outcomes) ** 2))


class TestBrierDecompositionIdentity:
    """`Rel - Res + Unc` was formed from TWO different outcome spaces.

    Reliability and resolution came from a top-1 BINARY reduction while
    uncertainty came from MULTICLASS one-hot base rates, so the identity failed
    by 30-65% at every K.
    """

    @staticmethod
    def _probe(n_classes: int, n: int = 2000, seed: int = 0):
        rng = np.random.default_rng(seed)
        y_prob = rng.dirichlet(np.ones(n_classes), n)
        y_true = np.array([rng.choice(n_classes, p=row) for row in y_prob])
        return y_true, y_prob

    @pytest.mark.parametrize("n_classes", [2, 3, 10])
    def test_the_identity_holds(self, n_classes):
        y_true, y_prob = self._probe(n_classes)
        decomp = compute_brier_score_decomposition(y_true, y_prob, n_bins=15)

        # Anti-vacuity: all three terms zero would satisfy any identity.
        assert decomp['uncertainty'] > 1e-3, "degenerate probe: zero uncertainty"
        assert decomp['brier_score'] > 1e-3, "degenerate probe: zero Brier score"

        recombined = (
            decomp['reliability'] - decomp['resolution'] + decomp['uncertainty'])
        assert recombined == pytest.approx(decomp['brier_score'], abs=1e-12), (
            f"K={n_classes}: Rel - Res + Unc = {recombined!r} vs "
            f"BS = {decomp['brier_score']!r}"
        )

    @pytest.mark.parametrize("n_classes", [2, 3, 10])
    def test_the_reported_brier_score_is_the_top1_one(self, n_classes):
        """The decomposed scalar must be the top-1 Brier score, not a stand-in.

        `brier_score + binning_residual` reconstructs the RAW top-1 Brier score
        computed independently in this file, so the decomposition is anchored to
        a real quantity rather than to an internally-consistent invention.
        """
        y_true, y_prob = self._probe(n_classes)
        decomp = compute_brier_score_decomposition(y_true, y_prob, n_bins=15)

        raw = _top1_brier_score(y_true, y_prob)
        assert raw > 1e-3, "degenerate probe"
        assert decomp['brier_score'] + decomp['binning_residual'] == pytest.approx(
            raw, abs=1e-12), (
            f"K={n_classes}: binned BS {decomp['brier_score']!r} + residual "
            f"{decomp['binning_residual']!r} != raw top-1 BS {raw!r}"
        )

    def test_uncertainty_is_the_top1_correctness_variance(self):
        """`Unc = acc * (1 - acc)` on the top-1 correctness variable.

        The shipped value was `sum(p_c * (1 - p_c))` over MULTICLASS base rates,
        which is a different quantity in a different outcome space.
        """
        y_true, y_prob = self._probe(3)
        decomp = compute_brier_score_decomposition(y_true, y_prob, n_bins=15)

        acc = float(np.mean(np.argmax(y_prob, axis=1) == y_true))
        multiclass_form = float(np.sum(
            np.mean(np.eye(3)[y_true], axis=0) * (1 - np.mean(np.eye(3)[y_true], axis=0))))

        # Anti-vacuity: the two candidate formulas must actually differ here.
        assert abs(acc * (1 - acc) - multiclass_form) > 1e-3

        assert decomp['uncertainty'] == pytest.approx(acc * (1 - acc), abs=1e-12)

    def test_a_perfectly_confident_and_correct_model_decomposes_to_zero(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_prob = np.eye(3)[y_true]
        decomp = compute_brier_score_decomposition(y_true, y_prob, n_bins=10)

        assert decomp['brier_score'] == pytest.approx(0.0, abs=1e-12)
        assert decomp['reliability'] == pytest.approx(0.0, abs=1e-12)
        assert decomp['resolution'] == pytest.approx(0.0, abs=1e-12)
        assert decomp['uncertainty'] == pytest.approx(0.0, abs=1e-12)
