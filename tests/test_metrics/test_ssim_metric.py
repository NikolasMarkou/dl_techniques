"""Tests for SsimMetric."""

import keras
from keras import ops
import pytest
import numpy as np

from dl_techniques.metrics.ssim_metric import SsimMetric


class TestSsimMetric:
    """Tests for SsimMetric."""

    def test_init_defaults(self):
        metric = SsimMetric()
        assert metric.name == "ssim"
        assert metric.max_val == 1.0

    def test_init_custom(self):
        metric = SsimMetric(max_val=255.0, name="custom_ssim")
        assert metric.max_val == 255.0
        assert metric.name == "custom_ssim"

    def test_identical_images(self):
        """Identical images should have SSIM = 1.0."""
        metric = SsimMetric()
        images = np.random.rand(4, 32, 32, 3).astype(np.float32)
        metric.update_state(images, images)
        result = float(metric.result())
        np.testing.assert_allclose(result, 1.0, atol=1e-5)

    def test_different_images(self):
        """Different images should have SSIM < 1.0."""
        metric = SsimMetric()
        y_true = np.random.rand(4, 32, 32, 3).astype(np.float32)
        y_pred = np.random.rand(4, 32, 32, 3).astype(np.float32)
        metric.update_state(y_true, y_pred)
        result = float(metric.result())
        assert result < 1.0
        assert result > -1.0  # SSIM range is [-1, 1]

    def test_reset_state(self):
        metric = SsimMetric()
        images = np.random.rand(4, 32, 32, 3).astype(np.float32)
        metric.update_state(images, images)
        metric.reset_state()
        # After reset, result should be 0 (no data)
        assert float(metric.ssim_sum) == 0.0
        assert float(metric.count) == 0.0

    def test_multiple_updates(self):
        """Multiple update_state calls should accumulate correctly."""
        metric = SsimMetric()
        images = np.random.rand(2, 32, 32, 3).astype(np.float32)
        metric.update_state(images, images)
        metric.update_state(images, images)
        result = float(metric.result())
        np.testing.assert_allclose(result, 1.0, atol=1e-5)

    def test_get_config_roundtrip(self):
        metric = SsimMetric(max_val=2.0, name="test_ssim")
        config = metric.get_config()
        restored = SsimMetric.from_config(config)
        assert restored.max_val == 2.0
        assert restored.name == "test_ssim"

    def test_serialization(self):
        metric = SsimMetric(max_val=255.0)
        config = keras.saving.serialize_keras_object(metric)
        restored = keras.saving.deserialize_keras_object(config)
        assert isinstance(restored, SsimMetric)
        assert restored.max_val == 255.0

    def test_max_val_255(self):
        """Should work with [0, 255] range images."""
        metric = SsimMetric(max_val=255.0)
        images = (np.random.rand(2, 32, 32, 3) * 255).astype(np.float32)
        metric.update_state(images, images)
        result = float(metric.result())
        np.testing.assert_allclose(result, 1.0, atol=1e-5)
