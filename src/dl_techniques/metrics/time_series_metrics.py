import keras
from keras import ops
import numpy as np
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SMAPE(keras.metrics.Metric):
    """Symmetric Mean Absolute Percentage Error metric.

    Computes sMAPE as: 200 * mean(|y_true - y_pred| / (|y_true| + |y_pred| + epsilon))

    This metric is bounded between 0 and 200, and is symmetric with respect to
    over-predictions and under-predictions.

    Args:
        name: Name of the metric instance.
        **kwargs: Additional keyword arguments passed to the parent Metric class.
    """

    def __init__(self, name: str = "smape", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        sample_weight: Optional[keras.KerasTensor] = None,
    ) -> None:
        """Update the metric state with new predictions.

        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            sample_weight: Optional weighting of each sample.
        """
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")

        numerator = ops.abs(y_true - y_pred)
        denominator = ops.abs(y_true) + ops.abs(y_pred) + 1e-7

        values = 200.0 * (numerator / denominator)

        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, "float32")
            values = values * sample_weight

        self.total.assign_add(ops.sum(values))
        self.count.assign_add(ops.cast(ops.size(y_true), "float32"))

    def result(self) -> keras.KerasTensor:
        """Compute the current metric value.

        Returns:
            The sMAPE value.
        """
        return ops.divide_no_nan(self.total, self.count)

    def reset_state(self) -> None:
        """Reset the metric state."""
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization."""
        return super().get_config()


# ---------------------------------------------------------------------
# Metric Calculation Utilities
# ---------------------------------------------------------------------


def calculate_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    backcast: np.ndarray,
) -> Dict[str, float]:
    """Calculate comprehensive time series forecasting metrics.

    Computes MAE, RMSE, sMAPE, rMAE, and MASE for forecast evaluation.
    The rMAE and MASE metrics require historical data (backcast) to establish
    baseline comparisons.

    Args:
        y_true: Ground truth values with shape (batch, forecast_len, features).
        y_pred: Predicted values with shape (batch, forecast_len, features).
        backcast: Historical values with shape (batch, backcast_len, features).
            Used to compute naive baseline and scaling factor.

    Returns:
        Dictionary containing:
        - MAE: Mean Absolute Error
        - RMSE: Root Mean Squared Error
        - sMAPE: Symmetric Mean Absolute Percentage Error (0-200 scale)
        - rMAE: Relative MAE compared to naive last-value baseline
        - MASE: Mean Absolute Scaled Error using random walk scaling
    """
    epsilon: float = 1e-7

    # MAE: Mean Absolute Error
    mae: float = np.mean(np.abs(y_true - y_pred))

    # RMSE: Root Mean Squared Error
    rmse: float = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # sMAPE: Symmetric Mean Absolute Percentage Error
    denominator: np.ndarray = np.abs(y_true) + np.abs(y_pred) + epsilon
    smape: float = 200.0 * np.mean(np.abs(y_true - y_pred) / denominator)

    # Naive baseline: repeat last backcast value for all forecast steps
    last_val: np.ndarray = backcast[:, -1:, :]  # Shape: (batch, 1, features)
    naive_forecast: np.ndarray = np.tile(last_val, (1, y_true.shape[1], 1))
    mae_naive: float = np.mean(np.abs(y_true - naive_forecast))

    # rMAE: Relative MAE (normalized by naive forecast)
    rmae: float = mae / (mae_naive + epsilon)

    # MASE: Mean Absolute Scaled Error
    # Scale is the MAE of first-order differences in the backcast (random walk baseline)
    backcast_diff: np.ndarray = np.abs(backcast[:, 1:, :] - backcast[:, :-1, :])
    scale: float = np.mean(backcast_diff) + epsilon
    mase: float = mae / scale

    return {
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE": smape,
        "rMAE": rmae,
        "MASE": mase,
    }
