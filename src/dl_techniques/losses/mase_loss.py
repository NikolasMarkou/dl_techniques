"""
Mean Absolute Scaled Error (MASE), a scale-free loss metric.

This loss function evaluates forecast accuracy by comparing the model's
performance against a simple, naive baseline forecast (e.g., predicting
the previous value). This relative comparison makes the metric scale-
independent, allowing for meaningful performance evaluation across different
time series, regardless of their native units or magnitudes.

Architecturally, MASE provides a more informative training objective than
standard error metrics like MAE or MSE for many forecasting tasks. A MASE
value is directly interpretable: a value less than 1.0 indicates that the
model is outperforming the naive benchmark, while a value greater than 1.0
indicates it is performing worse. This provides a clear, standardized
measure of forecasting skill. This specific implementation calculates the
naive forecast error on a per-batch basis, making it a self-contained and
practical loss function for mini-batch training, although this is an
approximation of the canonical MASE which uses a global scaling factor
derived from the entire training set.

Foundational Mathematics
------------------------
MASE is defined as the ratio of the Mean Absolute Error (MAE) of the
forecast to the MAE of a naive, in-sample benchmark forecast.

    MASE = MAE_forecast / MAE_naive

1.  **Forecast Error (Numerator)**: This is the standard MAE of the
    model's predictions over the forecast horizon.
    `MAE_forecast = mean(|y_true - y_pred|)`

2.  **Naive Scaling Factor (Denominator)**: This is the MAE of a simple,
    non-seasonal (`m=1`) or seasonal (`m > 1`) naive forecast, where the
    forecast for time `t` is the value from time `t-m`. For a time
    series `y` of length `T`, it is calculated over the training data as:
    `MAE_naive = (1 / (T - m)) * Σ |y_t - y_{t-m}|` for `t` from `m+1`
    to `T`.

This denominator acts as a robust scaling factor, normalizing the forecast
error by a measure of the inherent variability and one-step
predictability of the time series itself.

References
----------
The MASE metric was proposed by:
-   Hyndman, R. J., & Koehler, A. B. (2006). "Another look at measures
    of forecast accuracy". *International Journal of Forecasting*.
"""

import keras
from keras import ops

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MASELoss(keras.losses.Loss):
    """Mean Absolute Scaled Error (MASE) loss.

    MASE is a scale-independent error metric that compares the forecast accuracy
    to a naive forecast. It's particularly useful for time series with different
    scales. This implementation computes the scaling factor (the MAE of the
    naive forecast) on a per-batch basis.

    The formula is:
    MASE = MAE / MAE_naive

    Where MAE_naive is the mean absolute error of a naive forecast.

    Args:
        seasonal_periods: The number of periods in a season for the seasonal
                          naive forecast. Defaults to 1 for a simple one-step
                          naive forecast.
        epsilon: A small float value to avoid division by zero.
        name: Name for the loss function.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        seasonal_periods: int = 1,
        epsilon: float = 1e-8,
        name: str = "mase_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.seasonal_periods = seasonal_periods
        self.epsilon = epsilon

    def call(self,
             y_true: keras.KerasTensor,
             y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute MASE loss.

        Args:
            y_true: Ground truth values.
            y_pred: The predicted values.

        Returns:
            MASE loss value.
        """
        y_true = ops.cast(y_true, y_pred.dtype)

        # Calculate MAE of the forecast
        mae_forecast = ops.mean(ops.abs(y_true - y_pred))

        # Calculate MAE of the naive forecast
        # Note: This is a batch-wise approximation. For a canonical MASE, the
        # scaling factor should be computed on the training set.
        if len(y_true.shape) > 1 and y_true.shape[1] > self.seasonal_periods:
            naive_true = y_true[:, self.seasonal_periods:]
            naive_forecast = y_true[:, :-self.seasonal_periods]
            mae_naive = ops.mean(ops.abs(naive_true - naive_forecast))
        else:
            # Fallback for short sequences where seasonal naive is not possible.
            # Uses a simple naive-1 forecast error.
            if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                mae_naive = ops.mean(ops.abs(y_true[:, 1:] - y_true[:, :-1]))
            else:
                # Heuristic for very short or 1D sequences. This is not a
                # true naive forecast but provides a stable scaling factor.
                mae_naive = ops.mean(ops.abs(y_true))


        # Add epsilon to avoid division by zero
        mae_naive = ops.maximum(mae_naive, self.epsilon)

        # Calculate MASE
        mase = mae_forecast / mae_naive

        return mase

    def get_config(self) -> dict:
        """Get loss configuration."""
        config = super().get_config()
        config.update({
            'seasonal_periods': self.seasonal_periods,
            'epsilon': self.epsilon,
        })
        return config

def mase_metric(seasonal_periods: int = 1):
    """Factory function for a MASE metric for use with `model.compile()`.

    This allows configuring the seasonal period for the MASE calculation.
    Example usage: `model.compile(metrics=[mase_metric(seasonal_periods=7)])`

    Args:
        seasonal_periods: The number of periods in a season. Defaults to 1.

    Returns:
        A callable metric function.
    """
    def metric(y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """
        Computes the Mean Absolute Scaled Error metric.
        """
        y_true = ops.cast(y_true, y_pred.dtype)
        epsilon = 1e-8

        mae_forecast = ops.mean(ops.abs(y_true - y_pred))

        if len(y_true.shape) > 1 and y_true.shape[1] > seasonal_periods:
            naive_true = y_true[:, seasonal_periods:]
            naive_forecast = y_true[:, :-seasonal_periods]
            mae_naive = ops.mean(ops.abs(naive_true - naive_forecast))
        else:
            if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                mae_naive = ops.mean(ops.abs(y_true[:, 1:] - y_true[:, :-1]))
            else:
                mae_naive = ops.mean(ops.abs(y_true))


        return mae_forecast / ops.maximum(mae_naive, epsilon)

    metric.__name__ = f'mase_metric_sp{seasonal_periods}'
    return metric

# ---------------------------------------------------------------------
