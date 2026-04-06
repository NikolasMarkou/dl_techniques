import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.metrics.capsule_accuracy import CapsuleAccuracy


class TestCapsuleAccuracy:
    """Tests for CapsuleAccuracy metric."""

    def test_init_default(self):
        metric = CapsuleAccuracy()
        assert metric.name == "capsule_accuracy"

    def test_perfect_accuracy_with_tensor(self):
        metric = CapsuleAccuracy()
        # 4 samples, 3 classes — predictions match labels perfectly
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype="float32")
        y_pred = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9], [0.7, 0.2, 0.1]], dtype="float32")

        metric.update_state(y_true, y_pred)
        result = float(metric.result())
        assert abs(result - 1.0) < 1e-6

    def test_perfect_accuracy_with_dict(self):
        metric = CapsuleAccuracy()
        y_true = np.array([[1, 0, 0], [0, 1, 0]], dtype="float32")
        lengths = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1]], dtype="float32")

        metric.update_state(y_true, {"length": lengths})
        result = float(metric.result())
        assert abs(result - 1.0) < 1e-6

    def test_zero_accuracy(self):
        metric = CapsuleAccuracy()
        # All predictions are wrong
        y_true = np.array([[1, 0, 0], [0, 1, 0]], dtype="float32")
        y_pred = np.array([[0.0, 0.9, 0.1], [0.9, 0.0, 0.1]], dtype="float32")

        metric.update_state(y_true, y_pred)
        result = float(metric.result())
        assert abs(result - 0.0) < 1e-6

    def test_partial_accuracy(self):
        metric = CapsuleAccuracy()
        # 1 out of 2 correct
        y_true = np.array([[1, 0, 0], [0, 1, 0]], dtype="float32")
        y_pred = np.array([[0.9, 0.1, 0.0], [0.9, 0.0, 0.1]], dtype="float32")

        metric.update_state(y_true, y_pred)
        result = float(metric.result())
        assert abs(result - 0.5) < 1e-6

    def test_sample_weight(self):
        metric = CapsuleAccuracy()
        y_true = np.array([[1, 0, 0], [0, 1, 0]], dtype="float32")
        y_pred = np.array([[0.9, 0.1, 0.0], [0.9, 0.0, 0.1]], dtype="float32")
        # First is correct (weight=2), second is wrong (weight=1)
        weights = np.array([2.0, 1.0], dtype="float32")

        metric.update_state(y_true, y_pred, sample_weight=weights)
        result = float(metric.result())
        # total = 2*1 + 1*0 = 2, count = 2
        assert abs(result - 1.0) < 1e-6

    def test_reset_state(self):
        metric = CapsuleAccuracy()
        y_true = np.array([[1, 0, 0]], dtype="float32")
        y_pred = np.array([[0.9, 0.1, 0.0]], dtype="float32")

        metric.update_state(y_true, y_pred)
        assert float(metric.result()) > 0.0

        metric.reset_state()
        # After reset, result should be 0 (divide_no_nan returns 0 for 0/0)
        assert float(metric.result()) == 0.0

    def test_accumulation(self):
        metric = CapsuleAccuracy()
        y_true = np.array([[1, 0, 0]], dtype="float32")
        y_pred_correct = np.array([[0.9, 0.1, 0.0]], dtype="float32")
        y_pred_wrong = np.array([[0.0, 0.9, 0.1]], dtype="float32")

        metric.update_state(y_true, y_pred_correct)
        metric.update_state(y_true, y_pred_wrong)
        result = float(metric.result())
        assert abs(result - 0.5) < 1e-6

    def test_get_config_and_from_config(self):
        metric = CapsuleAccuracy(name="my_capsule_acc")
        config = metric.get_config()
        assert config["name"] == "my_capsule_acc"

        restored = CapsuleAccuracy.from_config(config)
        assert restored.name == "my_capsule_acc"

    def test_serialization_round_trip(self):
        metric = CapsuleAccuracy()
        config = metric.get_config()
        restored = CapsuleAccuracy.from_config(config)

        # Both should produce same results
        y_true = np.array([[1, 0, 0], [0, 1, 0]], dtype="float32")
        y_pred = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1]], dtype="float32")

        metric.update_state(y_true, y_pred)
        restored.update_state(y_true, y_pred)

        assert abs(float(metric.result()) - float(restored.result())) < 1e-6
