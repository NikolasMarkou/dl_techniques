"""Tests for SequenceAccuracy and BitErrorRate."""

import keras
from keras import ops
import pytest
import numpy as np

from dl_techniques.metrics.sequence_metrics import SequenceAccuracy, BitErrorRate


class TestSequenceAccuracy:
    """Tests for SequenceAccuracy."""

    def test_init_defaults(self):
        metric = SequenceAccuracy()
        assert metric.name == "sequence_accuracy"
        assert metric.threshold == 0.5

    def test_init_custom(self):
        metric = SequenceAccuracy(threshold=0.3, name="seq_acc")
        assert metric.threshold == 0.3
        assert metric.name == "seq_acc"

    def test_perfect_sequences(self):
        """All sequences match perfectly."""
        metric = SequenceAccuracy()
        y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        y_pred = np.array([[0.9, 0.1, 0.8], [0.1, 0.9, 0.2]], dtype=np.float32)
        metric.update_state(y_true, y_pred)
        np.testing.assert_allclose(float(metric.result()), 1.0, atol=1e-5)

    def test_partial_match(self):
        """One correct sequence, one wrong."""
        metric = SequenceAccuracy()
        y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        y_pred = np.array([
            [0.9, 0.1, 0.8],  # correct
            [0.1, 0.9, 0.9],  # wrong (last element)
        ], dtype=np.float32)
        metric.update_state(y_true, y_pred)
        np.testing.assert_allclose(float(metric.result()), 0.5, atol=1e-5)

    def test_no_match(self):
        """No sequences match."""
        metric = SequenceAccuracy()
        y_true = np.array([[1, 1, 1]], dtype=np.float32)
        y_pred = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
        metric.update_state(y_true, y_pred)
        np.testing.assert_allclose(float(metric.result()), 0.0, atol=1e-5)

    def test_3d_sequences(self):
        """Should work with (batch, time, features) shape."""
        metric = SequenceAccuracy()
        y_true = np.array([
            [[1, 0], [0, 1]],
            [[1, 1], [0, 0]],
        ], dtype=np.float32)
        y_pred = np.array([
            [[0.9, 0.1], [0.1, 0.9]],  # correct
            [[0.9, 0.9], [0.1, 0.9]],  # wrong (last feature of second step)
        ], dtype=np.float32)
        metric.update_state(y_true, y_pred)
        np.testing.assert_allclose(float(metric.result()), 0.5, atol=1e-5)

    def test_reset_state(self):
        metric = SequenceAccuracy()
        y_true = np.ones((2, 5), dtype=np.float32)
        y_pred = np.ones((2, 5), dtype=np.float32)
        metric.update_state(y_true, y_pred)
        metric.reset_state()
        assert float(metric.correct) == 0.0
        assert float(metric.total) == 0.0

    def test_get_config_roundtrip(self):
        metric = SequenceAccuracy(threshold=0.3, name="test")
        config = metric.get_config()
        restored = SequenceAccuracy.from_config(config)
        assert restored.threshold == 0.3
        assert restored.name == "test"

    def test_serialization(self):
        metric = SequenceAccuracy(threshold=0.7)
        config = keras.saving.serialize_keras_object(metric)
        restored = keras.saving.deserialize_keras_object(config)
        assert isinstance(restored, SequenceAccuracy)
        assert restored.threshold == 0.7


class TestBitErrorRate:
    """Tests for BitErrorRate."""

    def test_init_defaults(self):
        metric = BitErrorRate()
        assert metric.name == "bit_error_rate"
        assert metric.threshold == 0.5

    def test_zero_errors(self):
        """Perfect predictions should give BER = 0."""
        metric = BitErrorRate()
        y_true = np.array([[1, 0, 1, 0]], dtype=np.float32)
        y_pred = np.array([[0.9, 0.1, 0.8, 0.2]], dtype=np.float32)
        metric.update_state(y_true, y_pred)
        np.testing.assert_allclose(float(metric.result()), 0.0, atol=1e-5)

    def test_all_errors(self):
        """All bits wrong should give BER = 1."""
        metric = BitErrorRate()
        y_true = np.array([[1, 1, 1, 1]], dtype=np.float32)
        y_pred = np.array([[0.1, 0.1, 0.1, 0.1]], dtype=np.float32)
        metric.update_state(y_true, y_pred)
        np.testing.assert_allclose(float(metric.result()), 1.0, atol=1e-5)

    def test_partial_errors(self):
        """Half bits wrong should give BER = 0.5."""
        metric = BitErrorRate()
        y_true = np.array([[1, 1, 0, 0]], dtype=np.float32)
        y_pred = np.array([[0.9, 0.1, 0.9, 0.1]], dtype=np.float32)
        metric.update_state(y_true, y_pred)
        np.testing.assert_allclose(float(metric.result()), 0.5, atol=1e-5)

    def test_multiple_batches(self):
        """BER should accumulate correctly across batches."""
        metric = BitErrorRate()
        # Batch 1: 0 errors in 4 bits
        metric.update_state(
            np.array([[1, 0]], dtype=np.float32),
            np.array([[0.9, 0.1]], dtype=np.float32),
        )
        # Batch 2: 2 errors in 2 bits
        metric.update_state(
            np.array([[1, 0]], dtype=np.float32),
            np.array([[0.1, 0.9]], dtype=np.float32),
        )
        # Total: 2 errors in 4 bits = 0.5
        np.testing.assert_allclose(float(metric.result()), 0.5, atol=1e-5)

    def test_reset_state(self):
        metric = BitErrorRate()
        metric.update_state(
            np.ones((2, 4), dtype=np.float32),
            np.zeros((2, 4), dtype=np.float32),
        )
        metric.reset_state()
        assert float(metric.errors) == 0.0
        assert float(metric.total) == 0.0

    def test_get_config_roundtrip(self):
        metric = BitErrorRate(threshold=0.3, name="ber_test")
        config = metric.get_config()
        restored = BitErrorRate.from_config(config)
        assert restored.threshold == 0.3
        assert restored.name == "ber_test"

    def test_serialization(self):
        metric = BitErrorRate(threshold=0.8)
        config = keras.saving.serialize_keras_object(metric)
        restored = keras.saving.deserialize_keras_object(config)
        assert isinstance(restored, BitErrorRate)
        assert restored.threshold == 0.8
