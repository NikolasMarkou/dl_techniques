import numpy as np
import pytest
import keras

from dl_techniques.metrics.psnr_metric import PsnrMetric


class TestPsnrMetric:
    """Tests for PsnrMetric."""

    def test_init_default(self):
        metric = PsnrMetric()
        assert metric.name == "primary_psnr"
        assert metric.max_val == 1.0

    def test_init_custom(self):
        metric = PsnrMetric(max_val=255.0, name="custom_psnr")
        assert metric.name == "custom_psnr"
        assert metric.max_val == 255.0

    def test_identical_images(self):
        """Identical images should produce very high PSNR."""
        metric = PsnrMetric(max_val=1.0)
        images = np.random.rand(4, 8, 8, 3).astype("float32")
        metric.update_state(images, images)
        result = float(metric.result())
        # PSNR should be very high (limited by epsilon clamping ~70 dB)
        assert result > 50.0

    def test_known_psnr_value(self):
        """Test PSNR against a known MSE value."""
        metric = PsnrMetric(max_val=1.0)
        # Create images with known MSE = 0.01
        y_true = np.zeros((1, 10, 10, 1), dtype="float32")
        y_pred = np.full((1, 10, 10, 1), 0.1, dtype="float32")
        # MSE = 0.01, PSNR = 10 * log10(1 / 0.01) = 10 * 2 = 20 dB

        metric.update_state(y_true, y_pred)
        result = float(metric.result())
        assert abs(result - 20.0) < 0.1

    def test_known_psnr_max_val_255(self):
        """Test PSNR with max_val=255."""
        metric = PsnrMetric(max_val=255.0)
        # Create images with MSE = 100
        y_true = np.zeros((1, 10, 10, 1), dtype="float32")
        y_pred = np.full((1, 10, 10, 1), 10.0, dtype="float32")
        # MSE = 100, PSNR = 10 * log10(255^2 / 100) = 10 * log10(650.25) ≈ 28.13 dB

        metric.update_state(y_true, y_pred)
        result = float(metric.result())
        expected = 10.0 * np.log10(255.0 ** 2 / 100.0)
        assert abs(result - expected) < 0.1

    def test_multi_output_list(self):
        """Test with list inputs (multi-output model)."""
        metric = PsnrMetric(max_val=1.0)

        primary = np.random.rand(2, 8, 8, 3).astype("float32")
        secondary = np.random.rand(2, 4, 4, 3).astype("float32")

        metric.update_state([primary, secondary], [primary, secondary])
        result = float(metric.result())
        # Primary output is identical → very high PSNR
        assert result > 50.0

    def test_reset_state(self):
        metric = PsnrMetric()
        images = np.random.rand(2, 8, 8, 3).astype("float32")
        metric.update_state(images, images)
        assert float(metric.result()) > 0.0

        metric.reset_state()
        assert float(metric.result()) == 0.0

    def test_accumulation(self):
        metric = PsnrMetric(max_val=1.0)
        batch1_true = np.random.rand(2, 8, 8, 3).astype("float32")
        batch1_pred = batch1_true + np.random.normal(0, 0.01, batch1_true.shape).astype("float32")
        batch2_true = np.random.rand(3, 8, 8, 3).astype("float32")
        batch2_pred = batch2_true + np.random.normal(0, 0.01, batch2_true.shape).astype("float32")

        metric.update_state(batch1_true, batch1_pred)
        metric.update_state(batch2_true, batch2_pred)
        result = float(metric.result())
        # With small noise, PSNR should be reasonably high
        assert result > 30.0

    def test_get_config_and_from_config(self):
        metric = PsnrMetric(max_val=255.0, name="test_psnr")
        config = metric.get_config()
        assert config["max_val"] == 255.0
        assert config["name"] == "test_psnr"

        restored = PsnrMetric.from_config(config)
        assert restored.max_val == 255.0
        assert restored.name == "test_psnr"

    def test_serialization_round_trip(self):
        metric = PsnrMetric(max_val=1.0)
        config = metric.get_config()
        restored = PsnrMetric.from_config(config)

        y_true = np.random.rand(2, 8, 8, 3).astype("float32")
        y_pred = y_true + 0.1

        metric.update_state(y_true, y_pred)
        restored.update_state(y_true, y_pred)
        assert abs(float(metric.result()) - float(restored.result())) < 1e-4
