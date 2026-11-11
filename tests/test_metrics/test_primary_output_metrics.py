"""Tests for PrimaryOutputAccuracy and PrimaryOutputTopKAccuracy."""

import keras
from keras import ops
import pytest
import numpy as np

from dl_techniques.metrics.primary_output_metrics import (
    PrimaryOutputAccuracy,
    PrimaryOutputTopKAccuracy,
)


class TestPrimaryOutputAccuracy:
    """Tests for PrimaryOutputAccuracy."""

    def test_init_defaults(self):
        metric = PrimaryOutputAccuracy()
        assert metric.name == "primary_accuracy"

    def test_single_output_one_hot(self):
        """Should work with single-output one-hot labels."""
        metric = PrimaryOutputAccuracy()
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
        y_pred = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.3, 0.6, 0.1],  # wrong prediction
        ], dtype=np.float32)
        metric.update_state(y_true, y_pred)
        result = float(metric.result())
        np.testing.assert_allclose(result, 0.75, atol=1e-5)

    def test_multi_output_list(self):
        """Should extract first output from list."""
        metric = PrimaryOutputAccuracy()
        primary_true = np.array([[1, 0], [0, 1]], dtype=np.float32)
        aux_true = np.random.rand(2, 4).astype(np.float32)
        primary_pred = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
        aux_pred = np.random.rand(2, 4).astype(np.float32)

        metric.update_state([primary_true, aux_true], [primary_pred, aux_pred])
        result = float(metric.result())
        np.testing.assert_allclose(result, 1.0, atol=1e-5)

    def test_perfect_accuracy(self):
        metric = PrimaryOutputAccuracy()
        y_true = np.eye(5, dtype=np.float32)
        y_pred = np.eye(5, dtype=np.float32) * 10  # strong logits
        metric.update_state(y_true, y_pred)
        np.testing.assert_allclose(float(metric.result()), 1.0, atol=1e-5)

    def test_reset_state(self):
        metric = PrimaryOutputAccuracy()
        y_true = np.array([[1, 0]], dtype=np.float32)
        y_pred = np.array([[0.9, 0.1]], dtype=np.float32)
        metric.update_state(y_true, y_pred)
        metric.reset_state()
        assert float(metric.total) == 0.0
        assert float(metric.count) == 0.0

    def test_get_config_roundtrip(self):
        metric = PrimaryOutputAccuracy(name="test_acc")
        config = metric.get_config()
        restored = PrimaryOutputAccuracy.from_config(config)
        assert restored.name == "test_acc"

    def test_serialization(self):
        metric = PrimaryOutputAccuracy()
        config = keras.saving.serialize_keras_object(metric)
        restored = keras.saving.deserialize_keras_object(config)
        assert isinstance(restored, PrimaryOutputAccuracy)


class TestPrimaryOutputTopKAccuracy:
    """Tests for PrimaryOutputTopKAccuracy."""

    def test_init_defaults(self):
        metric = PrimaryOutputTopKAccuracy()
        assert metric.k == 5
        assert metric.name == "primary_top5_accuracy"

    def test_init_custom_k(self):
        metric = PrimaryOutputTopKAccuracy(k=3, name="top3")
        assert metric.k == 3
        assert metric.name == "top3"

    def test_top5_accuracy(self):
        """True class in top-5 should count as correct."""
        metric = PrimaryOutputTopKAccuracy(k=5)
        # 10 classes, true class is 7
        y_true = np.zeros((1, 10), dtype=np.float32)
        y_true[0, 7] = 1.0
        # Predictions: class 7 is 6th highest — not in top 5
        y_pred = np.array([[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]], dtype=np.float32)
        metric.update_state(y_true, y_pred)
        result = float(metric.result())
        np.testing.assert_allclose(result, 0.0, atol=1e-5)

        metric.reset_state()
        # Now class 7 is in top 5 (5th highest)
        y_pred = np.array([[1, 2, 3, 4, 5, 6, 7, 10, 8, 9]], dtype=np.float32)
        metric.update_state(y_true, y_pred)
        result = float(metric.result())
        np.testing.assert_allclose(result, 1.0, atol=1e-5)

    def test_multi_output(self):
        """Should extract first output from list."""
        metric = PrimaryOutputTopKAccuracy(k=2)
        primary_true = np.array([[0, 0, 1]], dtype=np.float32)
        primary_pred = np.array([[0.1, 0.8, 0.7]], dtype=np.float32)  # class 2 is in top 2
        metric.update_state(
            [primary_true, np.zeros((1, 5), dtype=np.float32)],
            [primary_pred, np.zeros((1, 5), dtype=np.float32)],
        )
        np.testing.assert_allclose(float(metric.result()), 1.0, atol=1e-5)

    def test_reset_state(self):
        metric = PrimaryOutputTopKAccuracy()
        y_true = np.eye(10, dtype=np.float32)[:2]
        y_pred = np.random.rand(2, 10).astype(np.float32)
        metric.update_state(y_true, y_pred)
        metric.reset_state()
        assert float(metric.total) == 0.0
        assert float(metric.count) == 0.0

    def test_get_config_roundtrip(self):
        metric = PrimaryOutputTopKAccuracy(k=3, name="top3_test")
        config = metric.get_config()
        restored = PrimaryOutputTopKAccuracy.from_config(config)
        assert restored.k == 3
        assert restored.name == "top3_test"

    def test_serialization(self):
        metric = PrimaryOutputTopKAccuracy(k=10)
        config = keras.saving.serialize_keras_object(metric)
        restored = keras.saving.deserialize_keras_object(config)
        assert isinstance(restored, PrimaryOutputTopKAccuracy)
        assert restored.k == 10


# ---------------------------------------------------------------------
# Symbolic label width: the one-hot-vs-sparse decision must come from the
# STATIC shape. `ops.shape(y)[-1] > 1` is a symbolic tf.Tensor used as a
# Python bool when the label width is unknown, which raises
# OperatorNotAllowedInGraphError under tf.function(autograph=False)
# (the mode Keras trainers use). Rule: rank > 1 and (width is None or
# width > 1) means one-hot (decision D-024).
# ---------------------------------------------------------------------

import tensorflow as tf

_ONE_HOT_TRUE = np.eye(4, dtype=np.float32)[[0, 1, 2, 3, 0]]
_PRED = np.array(
    [
        [0.9, 0.05, 0.03, 0.02],  # right
        [0.1, 0.8, 0.05, 0.05],  # right
        [0.4, 0.3, 0.2, 0.1],  # wrong
        [0.1, 0.1, 0.1, 0.7],  # right
        [0.2, 0.3, 0.4, 0.1],  # wrong
    ],
    dtype=np.float32,
)


def _run_graph(metric, true_spec, pred_spec, y_true, y_pred, jit_compile):
    """Run update_state + result inside a graph with the given signatures."""

    @tf.function(
        input_signature=[true_spec, pred_spec],
        autograph=False,
        jit_compile=jit_compile,
    )
    def step(t, p):
        metric.update_state(t, p)
        return metric.result()

    return float(step(tf.constant(y_true), tf.constant(y_pred)))


class TestUnknownLabelWidthInGraphMode:
    """Labels whose last dim is statically unknown must not raise."""

    @pytest.mark.parametrize("jit_compile", [False, True])
    def test_accuracy_with_unknown_label_width(self, jit_compile):
        value = _run_graph(
            PrimaryOutputAccuracy(),
            tf.TensorSpec((None, None), tf.float32),
            tf.TensorSpec((None, 4), tf.float32),
            _ONE_HOT_TRUE,
            _PRED,
            jit_compile,
        )
        assert value == pytest.approx(3.0 / 5.0)

    @pytest.mark.parametrize("jit_compile", [False, True])
    def test_top_k_with_unknown_label_width(self, jit_compile):
        value = _run_graph(
            PrimaryOutputTopKAccuracy(k=3),
            tf.TensorSpec((None, None), tf.float32),
            tf.TensorSpec((None, 4), tf.float32),
            _ONE_HOT_TRUE,
            _PRED,
            jit_compile,
        )
        # Rows 2 and 4 are wrong at top-1 but their true class is rank 3.
        assert value == pytest.approx(1.0)

    @pytest.mark.parametrize("jit_compile", [False, True])
    def test_static_width_one_labels_stay_sparse(self, jit_compile):
        """A statically-1 last dim is a sparse (B, 1) label, not one-hot."""
        sparse = np.array([[0], [1], [2], [3], [0]], dtype=np.float32)
        value = _run_graph(
            PrimaryOutputAccuracy(),
            tf.TensorSpec((None, 1), tf.float32),
            tf.TensorSpec((None, 4), tf.float32),
            sparse,
            _PRED,
            jit_compile,
        )
        assert value == pytest.approx(3.0 / 5.0)

    @pytest.mark.parametrize("jit_compile", [False, True])
    def test_rank_one_sparse_labels_in_graph_mode(self, jit_compile):
        sparse = np.array([0, 1, 2, 3, 0], dtype=np.float32)
        value = _run_graph(
            PrimaryOutputTopKAccuracy(k=1),
            tf.TensorSpec((None,), tf.float32),
            tf.TensorSpec((None, 4), tf.float32),
            sparse,
            _PRED,
            jit_compile,
        )
        assert value == pytest.approx(3.0 / 5.0)


class TestTopKIndexing:
    def test_top_k_is_indexed_not_attribute_accessed(self, monkeypatch):
        """`ops.top_k` returns a plain list on JAX (`.indices` is TF-only);
        the metric must index `[1]`. Cosmetic on TF: a list-returning
        top_k is the only way to see the difference here."""
        real_top_k = ops.top_k

        def list_top_k(x, k=1, sorted=True):
            values, indices = real_top_k(x, k=k, sorted=sorted)
            return [values, indices]

        monkeypatch.setattr(ops, "top_k", list_top_k)
        metric = PrimaryOutputTopKAccuracy(k=3)
        metric.update_state(_ONE_HOT_TRUE, _PRED)
        assert float(metric.result()) == pytest.approx(1.0)
