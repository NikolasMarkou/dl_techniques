"""Tests for ScaledMseLoss."""

import keras
from keras import ops
import pytest
import numpy as np

from dl_techniques.losses.scaled_mse_loss import ScaledMseLoss


class TestScaledMseLoss:
    """Tests for ScaledMseLoss."""

    def test_init_defaults(self):
        loss = ScaledMseLoss()
        assert loss.name == "scaled_mse_loss"
        assert loss.interpolation == "bilinear"

    def test_init_custom(self):
        loss = ScaledMseLoss(interpolation="nearest", name="custom_loss")
        assert loss.name == "custom_loss"
        assert loss.interpolation == "nearest"

    def test_same_size(self):
        """When y_true and y_pred have the same spatial dims, should equal MSE."""
        loss_fn = ScaledMseLoss()
        y_true = np.random.rand(2, 32, 32, 3).astype(np.float32)
        y_pred = np.random.rand(2, 32, 32, 3).astype(np.float32)
        loss = loss_fn(y_true, y_pred)
        expected = np.mean((y_true - y_pred) ** 2)
        np.testing.assert_allclose(float(loss), expected, rtol=1e-5)

    def test_different_size(self):
        """Loss should resize y_true to match y_pred spatial dims."""
        loss_fn = ScaledMseLoss()
        y_true = np.random.rand(2, 64, 64, 1).astype(np.float32)
        y_pred = np.random.rand(2, 32, 32, 1).astype(np.float32)
        loss = loss_fn(y_true, y_pred)
        assert float(loss) > 0.0
        assert np.isfinite(float(loss))

    def test_zero_loss(self):
        """Identical inputs after resize should give zero loss."""
        loss_fn = ScaledMseLoss()
        y_true = np.ones((2, 32, 32, 1), dtype=np.float32) * 0.5
        y_pred = np.ones((2, 32, 32, 1), dtype=np.float32) * 0.5
        loss = loss_fn(y_true, y_pred)
        np.testing.assert_allclose(float(loss), 0.0, atol=1e-7)

    def test_gradient_flow(self):
        """Gradients should flow through the loss."""
        import tensorflow as tf
        loss_fn = ScaledMseLoss()
        y_true = tf.constant(np.random.rand(2, 64, 64, 1).astype(np.float32))
        y_pred = tf.Variable(np.random.rand(2, 32, 32, 1).astype(np.float32))
        with tf.GradientTape() as tape:
            loss = loss_fn(y_true, y_pred)
        grads = tape.gradient(loss, y_pred)
        assert grads is not None
        assert tuple(grads.shape) == tuple(y_pred.shape)

    def test_get_config_roundtrip(self):
        loss = ScaledMseLoss(interpolation="nearest", name="test")
        config = loss.get_config()
        restored = ScaledMseLoss.from_config(config)
        assert restored.interpolation == "nearest"
        assert restored.name == "test"

    def test_serialization(self):
        loss = ScaledMseLoss(interpolation="bicubic")
        config = keras.saving.serialize_keras_object(loss)
        restored = keras.saving.deserialize_keras_object(config)
        assert isinstance(restored, ScaledMseLoss)
        assert restored.interpolation == "bicubic"

    def test_batch_sizes(self):
        """Should work with different batch sizes."""
        loss_fn = ScaledMseLoss()
        for batch_size in [1, 4, 16]:
            y_true = np.random.rand(batch_size, 64, 64, 3).astype(np.float32)
            y_pred = np.random.rand(batch_size, 16, 16, 3).astype(np.float32)
            loss = loss_fn(y_true, y_pred)
            assert np.isfinite(float(loss))
