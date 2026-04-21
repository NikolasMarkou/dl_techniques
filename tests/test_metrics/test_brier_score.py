"""Tests for dl_techniques.metrics.brier_score.

Covers:
- BrierScore (multi-label): happy path, from_logits vs probability input,
  sample_weight handling, correctness against scikit-learn / hand-computed
  values, state reset, serialization round-trip.
- CategoricalBrierScore (multi-class): dense one-hot path, sparse integer
  path, equivalence of the two, sample_weight, serialization round-trip.
"""

import keras
import numpy as np
import pytest

from dl_techniques.metrics.brier_score import (
    BrierScore,
    CategoricalBrierScore,
)


# ---------------------------------------------------------------------------
# BrierScore — binary / multi-label
# ---------------------------------------------------------------------------


class TestBrierScoreBinary:
    def test_perfect_prediction_is_zero(self):
        m = BrierScore(from_logits=False)
        y_true = np.array([[1.0], [0.0], [1.0], [0.0]])
        y_pred = np.array([[1.0], [0.0], [1.0], [0.0]])
        m.update_state(y_true, y_pred)
        assert float(m.result()) == pytest.approx(0.0, abs=1e-6)

    def test_worst_prediction(self):
        m = BrierScore(from_logits=False)
        y_true = np.array([[1.0], [0.0]])
        y_pred = np.array([[0.0], [1.0]])
        m.update_state(y_true, y_pred)
        assert float(m.result()) == pytest.approx(1.0, abs=1e-6)

    def test_uniform_0_5_gives_0_25(self):
        m = BrierScore(from_logits=False)
        y_true = np.array([[1.0], [0.0], [1.0], [0.0]])
        y_pred = np.full_like(y_true, 0.5)
        m.update_state(y_true, y_pred)
        assert float(m.result()) == pytest.approx(0.25, abs=1e-6)

    def test_hand_computed_example(self):
        """Worked example from brier_score.md (binary, 3 emails)."""
        m = BrierScore(from_logits=False)
        y_true = np.array([[1.0], [0.0], [0.0]])
        y_pred = np.array([[0.9], [0.2], [0.8]])
        m.update_state(y_true, y_pred)
        # (0.01 + 0.04 + 0.64) / 3 = 0.23
        assert float(m.result()) == pytest.approx(0.23, abs=1e-6)

    def test_from_logits_matches_sigmoid_then_bs(self):
        logits = np.array([[-2.0], [1.0], [0.0], [3.0]])
        probs = 1.0 / (1.0 + np.exp(-logits))
        y_true = np.array([[0.0], [1.0], [1.0], [0.0]])

        m_logit = BrierScore(from_logits=True)
        m_logit.update_state(y_true, logits)

        m_prob = BrierScore(from_logits=False)
        m_prob.update_state(y_true, probs)

        assert float(m_logit.result()) == pytest.approx(
            float(m_prob.result()), abs=1e-6
        )


class TestBrierScoreMultiLabel:
    def test_multi_label_mean_over_all_elements(self):
        m = BrierScore(from_logits=False)
        y_true = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.2]], dtype=np.float32)
        # SE: [(0.3^2, 0.2^2, 0.1^2), (0.1^2, 0.2^2, 0.2^2)]
        #   = [0.09, 0.04, 0.01, 0.01, 0.04, 0.04]
        # mean over 6 elements = 0.23 / 6 ≈ 0.03833
        m.update_state(y_true, y_pred)
        expected = (0.09 + 0.04 + 0.01 + 0.01 + 0.04 + 0.04) / 6.0
        assert float(m.result()) == pytest.approx(expected, abs=1e-6)

    def test_sample_weight_per_sample_broadcasts(self):
        m = BrierScore(from_logits=False)
        y_true = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.2]], dtype=np.float32)
        sw = np.array([2.0, 1.0], dtype=np.float32)
        m.update_state(y_true, y_pred, sample_weight=sw)
        # Weighted numerator:
        #   2 * (0.09 + 0.04 + 0.01) + 1 * (0.01 + 0.04 + 0.04) = 2*0.14 + 0.09 = 0.37
        # Weighted denominator: 2 * 3 + 1 * 3 = 9
        expected = 0.37 / 9.0
        assert float(m.result()) == pytest.approx(expected, abs=1e-6)


class TestBrierScoreLifecycle:
    def test_reset_state(self):
        m = BrierScore(from_logits=False)
        m.update_state(np.array([[1.0]]), np.array([[0.5]]))
        assert float(m.result()) > 0
        m.reset_state()
        assert float(m.result()) == pytest.approx(0.0, abs=1e-6)

    def test_serialization_round_trip(self):
        m = BrierScore(from_logits=True, name="brier_custom")
        config = m.get_config()
        m2 = BrierScore.from_config(config)
        assert m2.name == "brier_custom"
        assert m2.from_logits is True

    def test_accumulates_over_batches(self):
        m = BrierScore(from_logits=False)
        # Two batches of the 3-email example split into (1, 2)
        m.update_state(np.array([[1.0]]), np.array([[0.9]]))
        m.update_state(np.array([[0.0], [0.0]]), np.array([[0.2], [0.8]]))
        assert float(m.result()) == pytest.approx(0.23, abs=1e-6)


# ---------------------------------------------------------------------------
# CategoricalBrierScore — multi-class
# ---------------------------------------------------------------------------


class TestCategoricalBrierScoreDense:
    def test_perfect_one_hot(self):
        m = CategoricalBrierScore(from_logits=False, sparse_labels=False)
        y_true = np.eye(3)[np.array([0, 1, 2])].astype(np.float32)
        y_pred = y_true.copy()
        m.update_state(y_true, y_pred)
        assert float(m.result()) == pytest.approx(0.0, abs=1e-6)

    def test_uniform_1_over_K(self):
        # Predicting 1/K for every class → per-sample BS = sum_k (1/K - o_k)^2
        # = (K-1) * (1/K)^2 + (1 - 1/K)^2 = (K-1)/K^2 + (K-1)^2 / K^2 = (K-1)/K.
        K = 5
        m = CategoricalBrierScore(from_logits=False, sparse_labels=False)
        y_true = np.eye(K)[np.array([0, 1, 2])].astype(np.float32)
        y_pred = np.full_like(y_true, 1.0 / K)
        m.update_state(y_true, y_pred)
        assert float(m.result()) == pytest.approx((K - 1) / K, abs=1e-6)

    def test_worst_prediction_range_is_2(self):
        # Put 100% on the wrong class → per-sample BS = (1 - 0)^2 + (0 - 1)^2 = 2.
        m = CategoricalBrierScore(from_logits=False, sparse_labels=False)
        y_true = np.array([[1.0, 0.0, 0.0]])
        y_pred = np.array([[0.0, 1.0, 0.0]])
        m.update_state(y_true, y_pred)
        assert float(m.result()) == pytest.approx(2.0, abs=1e-6)

    def test_from_logits_matches_softmax(self):
        logits = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 5.0]])
        probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        y_true = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        m_logit = CategoricalBrierScore(from_logits=True, sparse_labels=False)
        m_logit.update_state(y_true, logits)
        m_prob = CategoricalBrierScore(from_logits=False, sparse_labels=False)
        m_prob.update_state(y_true, probs)
        assert float(m_logit.result()) == pytest.approx(
            float(m_prob.result()), abs=1e-6
        )


class TestCategoricalBrierScoreSparse:
    def test_sparse_matches_dense_on_random(self):
        rng = np.random.default_rng(42)
        K = 7
        N = 16
        probs_raw = rng.random((N, K)).astype(np.float32)
        probs = probs_raw / probs_raw.sum(axis=-1, keepdims=True)
        labels = rng.integers(0, K, size=(N,))
        onehot = np.eye(K)[labels].astype(np.float32)

        m_sparse = CategoricalBrierScore(from_logits=False, sparse_labels=True)
        m_sparse.update_state(labels, probs)

        m_dense = CategoricalBrierScore(from_logits=False, sparse_labels=False)
        m_dense.update_state(onehot, probs)

        assert float(m_sparse.result()) == pytest.approx(
            float(m_dense.result()), abs=1e-6
        )

    def test_sparse_segmentation_shape(self):
        """Sparse path over (B, H, W) labels with (B, H, W, K) probs."""
        rng = np.random.default_rng(0)
        B, H, W, K = 2, 4, 4, 6
        probs_raw = rng.random((B, H, W, K)).astype(np.float32)
        probs = probs_raw / probs_raw.sum(axis=-1, keepdims=True)
        labels = rng.integers(0, K, size=(B, H, W))

        m = CategoricalBrierScore(from_logits=False, sparse_labels=True)
        m.update_state(labels, probs)
        value = float(m.result())
        assert 0.0 <= value <= 2.0


class TestCategoricalBrierScoreLifecycle:
    def test_sample_weight_on_sparse_segmentation(self):
        rng = np.random.default_rng(0)
        B, H, W, K = 2, 3, 3, 4
        probs = rng.random((B, H, W, K)).astype(np.float32)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        labels = rng.integers(0, K, size=(B, H, W))
        # Weight: ignore half the pixels in batch 1.
        sw = np.ones((B, H, W), dtype=np.float32)
        sw[1, :, :W // 2] = 0.0

        m = CategoricalBrierScore(from_logits=False, sparse_labels=True)
        m.update_state(labels, probs, sample_weight=sw)
        value = float(m.result())
        assert 0.0 <= value <= 2.0

    def test_serialization_round_trip(self):
        m = CategoricalBrierScore(
            from_logits=True, sparse_labels=True, name="cat_brier", axis=-1,
        )
        config = m.get_config()
        m2 = CategoricalBrierScore.from_config(config)
        assert m2.name == "cat_brier"
        assert m2.from_logits is True
        assert m2.sparse_labels is True
        assert m2.axis == -1

    def test_reset_state(self):
        m = CategoricalBrierScore(from_logits=False, sparse_labels=False)
        m.update_state(
            np.array([[1.0, 0.0]]), np.array([[0.3, 0.7]])
        )
        assert float(m.result()) > 0.0
        m.reset_state()
        assert float(m.result()) == pytest.approx(0.0, abs=1e-6)
