"""Tests for MultiTaskLoss."""

import keras
from keras import ops
import pytest
import numpy as np

from dl_techniques.losses.multi_task_loss import MultiTaskLoss


class TestMultiTaskLoss:
    """Tests for MultiTaskLoss."""

    def test_init_defaults(self):
        loss = MultiTaskLoss()
        assert loss.name == "multi_task_loss"
        assert loss.loss_weights == {}

    def test_init_custom_weights(self):
        weights = {"pressure": 2.0, "velocity": 0.5}
        loss = MultiTaskLoss(loss_weights=weights)
        assert loss.loss_weights == weights

    def test_single_task(self):
        loss_fn = MultiTaskLoss()
        y_true = {"task_a": np.random.rand(4, 10).astype(np.float32)}
        y_pred = {"task_a": np.random.rand(4, 10).astype(np.float32)}
        loss = loss_fn(y_true, y_pred)
        expected = np.mean((y_true["task_a"] - y_pred["task_a"]) ** 2)
        np.testing.assert_allclose(float(loss), expected, rtol=1e-5)

    def test_multi_task_equal_weights(self):
        loss_fn = MultiTaskLoss()
        y_true = {
            "a": np.ones((4, 5), dtype=np.float32),
            "b": np.zeros((4, 5), dtype=np.float32),
        }
        y_pred = {
            "a": np.zeros((4, 5), dtype=np.float32),
            "b": np.ones((4, 5), dtype=np.float32),
        }
        loss = loss_fn(y_true, y_pred)
        # Both tasks have MSE=1.0, total = 2.0
        np.testing.assert_allclose(float(loss), 2.0, rtol=1e-5)

    def test_multi_task_custom_weights(self):
        loss_fn = MultiTaskLoss(loss_weights={"a": 2.0, "b": 0.5})
        y_true = {
            "a": np.ones((4, 5), dtype=np.float32),
            "b": np.zeros((4, 5), dtype=np.float32),
        }
        y_pred = {
            "a": np.zeros((4, 5), dtype=np.float32),
            "b": np.ones((4, 5), dtype=np.float32),
        }
        loss = loss_fn(y_true, y_pred)
        # task_a: MSE=1.0 * 2.0 = 2.0, task_b: MSE=1.0 * 0.5 = 0.5
        np.testing.assert_allclose(float(loss), 2.5, rtol=1e-5)

    def test_missing_key_in_pred(self):
        """Tasks in y_true but not y_pred should be skipped."""
        loss_fn = MultiTaskLoss()
        y_true = {
            "a": np.ones((4, 5), dtype=np.float32),
            "b": np.zeros((4, 5), dtype=np.float32),
        }
        y_pred = {"a": np.zeros((4, 5), dtype=np.float32)}
        loss = loss_fn(y_true, y_pred)
        np.testing.assert_allclose(float(loss), 1.0, rtol=1e-5)

    def test_zero_loss(self):
        loss_fn = MultiTaskLoss()
        data = {"x": np.random.rand(4, 10).astype(np.float32)}
        loss = loss_fn(data, data)
        np.testing.assert_allclose(float(loss), 0.0, atol=1e-7)

    def test_get_config_roundtrip(self):
        weights = {"pressure": 1.5, "velocity": 2.0}
        loss = MultiTaskLoss(loss_weights=weights, name="cfd_loss")
        config = loss.get_config()
        restored = MultiTaskLoss.from_config(config)
        assert restored.loss_weights == weights
        assert restored.name == "cfd_loss"

    def test_serialization(self):
        loss = MultiTaskLoss(loss_weights={"a": 1.0, "b": 2.0})
        config = keras.saving.serialize_keras_object(loss)
        restored = keras.saving.deserialize_keras_object(config)
        assert isinstance(restored, MultiTaskLoss)
        assert restored.loss_weights == {"a": 1.0, "b": 2.0}
