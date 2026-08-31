import numpy as np
import pytest
import keras

from dl_techniques.metrics.time_series_metrics import (
    SMAPE,
    PerHorizonError,
    calculate_comprehensive_metrics,
    horizon_profile,
    per_horizon_metrics,
)


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


class TestSMAPESampleWeight:
    """``count`` must be weighted like the ``total`` it normalises.

    It was not: ``total`` accumulated the WEIGHTED sum while ``count`` took
    ``ops.size(y_true)``, so a down-weighted row still counted as a full
    observation in the denominator and pulled the reported sMAPE toward zero.
    ``CoverageMetric`` / ``SharpnessMetric`` in
    ``probabilistic_forecast_metrics.py`` always did this correctly.

    The oracle is a SUBSET: weighting rows 0-2 with 1 and rows 3-5 with 0 must
    give exactly the sMAPE of rows 0-2 alone. Measured 2026-08-31 on a
    ``(6, 4, 2)`` pair: weighted 146.82056, subset 146.82056, unweighted
    143.92577 -- i.e. the pre-fix code would have reported a diluted value.
    """

    @staticmethod
    def _pair():
        return (
            np.random.default_rng(0).normal(size=(6, 4, 2)).astype("float32"),
            np.random.default_rng(1).normal(size=(6, 4, 2)).astype("float32"),
        )

    def test_zero_weights_are_equivalent_to_dropping_the_rows(self):
        y_true, y_pred = self._pair()
        weights = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype="float32")

        weighted = SMAPE()
        weighted.update_state(y_true, y_pred, sample_weight=weights[:, None, None])

        subset = SMAPE()
        subset.update_state(y_true[:3], y_pred[:3])

        np.testing.assert_allclose(
            float(weighted.result()), float(subset.result()), rtol=0, atol=1e-4
        )

    def test_weighting_actually_changes_the_result(self):
        """Anti-vacuity: the subset is not accidentally equal to the whole."""
        y_true, y_pred = self._pair()
        weights = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype="float32")

        weighted = SMAPE()
        weighted.update_state(y_true, y_pred, sample_weight=weights[:, None, None])

        unweighted = SMAPE()
        unweighted.update_state(y_true, y_pred)

        assert abs(float(weighted.result()) - float(unweighted.result())) > 1.0

    def test_unweighted_behaviour_is_unchanged(self):
        """A uniform weight of 1 must be a no-op, not a rescale."""
        y_true, y_pred = self._pair()

        plain = SMAPE()
        plain.update_state(y_true, y_pred)

        ones = SMAPE()
        ones.update_state(y_true, y_pred, sample_weight=np.ones((6, 1, 1), dtype="float32"))

        np.testing.assert_allclose(
            float(plain.result()), float(ones.result()), rtol=0, atol=1e-5
        )


class TestPerHorizonError:
    """A per-step profile is the diagnostic a multistep loss needs.

    Every other metric in the module reduces the horizon away, so ``MSEh``
    improving step 12 while destroying steps 1-11 reports the same aggregate MAE
    as a uniform improvement.
    """

    @staticmethod
    def _pair(shape=(6, 4, 2)):
        return (
            np.random.default_rng(0).normal(size=shape).astype("float32"),
            np.random.default_rng(1).normal(size=shape).astype("float32"),
        )

    def test_agrees_with_the_numpy_profile(self):
        y_true, y_pred = self._pair()
        metrics = per_horizon_metrics(4)
        for metric in metrics:
            metric.update_state(y_true, y_pred)

        np.testing.assert_allclose(
            np.array([float(m.result()) for m in metrics]),
            horizon_profile(y_true, y_pred),
            rtol=0, atol=1e-6,
        )

    def test_each_step_scores_only_its_own_slice(self):
        """Corrupt step 2 alone; only step 2's metric may move."""
        y_true, y_pred = self._pair((6, 4))
        before = horizon_profile(y_true, y_pred)

        corrupted = y_pred.copy()
        corrupted[:, 1] += 10.0
        after = horizon_profile(y_true, corrupted)

        np.testing.assert_allclose(
            np.delete(after, 1), np.delete(before, 1), rtol=0, atol=1e-6
        )
        assert after[1] > before[1] + 5.0

    def test_mse_mode(self):
        y_true, y_pred = self._pair((6, 4))
        metric = PerHorizonError(step=3, error="mse")
        metric.update_state(y_true, y_pred)
        np.testing.assert_allclose(
            float(metric.result()),
            float(((y_true[:, 2] - y_pred[:, 2]) ** 2).mean()),
            rtol=0, atol=1e-6,
        )

    def test_accumulates_across_batches(self):
        y_true, y_pred = self._pair((6, 4))
        streaming = PerHorizonError(step=1)
        streaming.update_state(y_true[:3], y_pred[:3])
        streaming.update_state(y_true[3:], y_pred[3:])

        whole = PerHorizonError(step=1)
        whole.update_state(y_true, y_pred)

        np.testing.assert_allclose(
            float(streaming.result()), float(whole.result()), rtol=0, atol=1e-6
        )

    def test_reset_state(self):
        y_true, y_pred = self._pair((6, 4))
        metric = PerHorizonError(step=1)
        metric.update_state(y_true, y_pred)
        metric.reset_state()
        assert float(metric.result()) == 0.0

    def test_default_name_identifies_the_step(self):
        assert PerHorizonError(step=7).name == "mae_h7"
        assert PerHorizonError(step=2, error="mse").name == "mse_h2"

    def test_get_config_round_trip(self):
        metric = PerHorizonError(step=5, error="mse")
        restored = PerHorizonError.from_config(metric.get_config())
        assert restored.step == 5 and restored.error == "mse"
        assert restored.name == metric.name

    @pytest.mark.parametrize(
        "kwargs", [{"step": 0}, {"step": -1}, {"step": 1.5}, {"step": 1, "error": "rmse"}]
    )
    def test_rejects_bad_configuration(self, kwargs):
        with pytest.raises(ValueError):
            PerHorizonError(**kwargs)

    def test_per_horizon_metrics_rejects_a_bad_horizon(self):
        with pytest.raises(ValueError):
            per_horizon_metrics(0)

    def test_horizon_profile_rejects_a_bad_error_name(self):
        y_true, y_pred = self._pair((6, 4))
        with pytest.raises(ValueError):
            horizon_profile(y_true, y_pred, error="rmse")
