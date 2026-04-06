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
