import numpy as np
import pytest
import keras

from dl_techniques.metrics.time_series_metrics import SMAPE, calculate_comprehensive_metrics


class TestSMAPE:
    """Tests for SMAPE metric."""

    def test_init_default(self):
        metric = SMAPE()
        assert metric.name == "smape"

    def test_perfect_predictions(self):
        metric = SMAPE()
        y = np.array([1.0, 2.0, 3.0], dtype="float32")
        metric.update_state(y, y)
        result = float(metric.result())
        assert abs(result) < 1e-3

    def test_known_smape(self):
        """sMAPE of y_true=1, y_pred=2 should be 200 * |1-2| / (|1|+|2|) ≈ 66.67."""
        metric = SMAPE()
        y_true = np.array([1.0], dtype="float32")
        y_pred = np.array([2.0], dtype="float32")
        metric.update_state(y_true, y_pred)
        result = float(metric.result())
        expected = 200.0 * 1.0 / 3.0  # ≈ 66.67
        assert abs(result - expected) < 0.1

    def test_symmetry(self):
        """sMAPE should be symmetric: sMAPE(a,b) == sMAPE(b,a)."""
        m1 = SMAPE()
        m2 = SMAPE()
        a = np.array([1.0, 2.0, 3.0], dtype="float32")
        b = np.array([2.0, 1.0, 4.0], dtype="float32")

        m1.update_state(a, b)
        m2.update_state(b, a)
        assert abs(float(m1.result()) - float(m2.result())) < 1e-5

    def test_bounded(self):
        """sMAPE should be between 0 and 200."""
        metric = SMAPE()
        y_true = np.array([0.0, 1.0, -1.0, 5.0], dtype="float32")
        y_pred = np.array([10.0, -5.0, 3.0, -2.0], dtype="float32")
        metric.update_state(y_true, y_pred)
        result = float(metric.result())
        assert 0.0 <= result <= 200.0 + 1e-3

    def test_reset_state(self):
        metric = SMAPE()
        metric.update_state(
            np.array([1.0], dtype="float32"),
            np.array([2.0], dtype="float32"),
        )
        assert float(metric.result()) > 0.0
        metric.reset_state()
        assert float(metric.result()) == 0.0

    def test_accumulation(self):
        metric = SMAPE()
        y1 = np.array([1.0, 2.0], dtype="float32")
        p1 = np.array([1.0, 2.0], dtype="float32")  # perfect
        y2 = np.array([1.0], dtype="float32")
        p2 = np.array([2.0], dtype="float32")  # imperfect

        metric.update_state(y1, p1)
        metric.update_state(y2, p2)
        result = float(metric.result())
        # 2 perfect + 1 imperfect, average should be positive but moderate
        assert 0.0 < result < 100.0

    def test_get_config_and_from_config(self):
        metric = SMAPE(name="my_smape")
        config = metric.get_config()
        assert config["name"] == "my_smape"

        restored = SMAPE.from_config(config)
        assert restored.name == "my_smape"

    def test_division_by_zero_safety(self):
        """result() should not crash when no data has been added."""
        metric = SMAPE()
        result = float(metric.result())
        assert result == 0.0  # divide_no_nan returns 0 for 0/0


class TestCalculateComprehensiveMetrics:
    """Tests for calculate_comprehensive_metrics function."""

    def test_perfect_prediction(self):
        batch, forecast, features = 4, 5, 2
        y = np.random.rand(batch, forecast, features)
        backcast = np.random.rand(batch, 10, features)
        result = calculate_comprehensive_metrics(y, y, backcast)

        assert abs(result["MAE"]) < 1e-7
        assert abs(result["RMSE"]) < 1e-7
        assert abs(result["sMAPE"]) < 1e-3
        assert "rMAE" in result
        assert "MASE" in result

    def test_output_keys(self):
        y = np.random.rand(2, 3, 1)
        backcast = np.random.rand(2, 5, 1)
        result = calculate_comprehensive_metrics(y, y + 0.1, backcast)
        expected_keys = {"MAE", "RMSE", "sMAPE", "rMAE", "MASE"}
        assert set(result.keys()) == expected_keys

    def test_mae_correctness(self):
        y_true = np.array([[[1.0], [2.0], [3.0]]])
        y_pred = np.array([[[1.5], [2.5], [3.5]]])
        backcast = np.array([[[0.0], [0.5], [1.0], [1.5], [2.0]]])
        result = calculate_comprehensive_metrics(y_true, y_pred, backcast)
        assert abs(result["MAE"] - 0.5) < 1e-6

    def test_rmse_correctness(self):
        y_true = np.array([[[0.0], [0.0]]])
        y_pred = np.array([[[1.0], [1.0]]])
        backcast = np.array([[[0.0], [0.0], [0.0]]])
        result = calculate_comprehensive_metrics(y_true, y_pred, backcast)
        assert abs(result["RMSE"] - 1.0) < 1e-6

    def test_smape_bounded(self):
        y_true = np.random.rand(4, 5, 2)
        y_pred = np.random.rand(4, 5, 2)
        backcast = np.random.rand(4, 10, 2)
        result = calculate_comprehensive_metrics(y_true, y_pred, backcast)
        assert 0.0 <= result["sMAPE"] <= 200.0 + 1e-3
