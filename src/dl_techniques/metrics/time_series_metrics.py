import keras
import numpy as np
import tensorflow as tf
from typing import Dict, Optional


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SMAPE(keras.metrics.Metric):
    """Symmetric Mean Absolute Percentage Error metric.

    Computes sMAPE as: 200 * mean(|y_true - y_pred| / (|y_true| + |y_pred| + epsilon))

    This metric is bounded between 0 and 200, and is symmetric with respect to
    over-predictions and under-predictions.

    Parameters
    ----------
    name : str, optional
        Name of the metric instance.
    **kwargs : dict
        Additional keyword arguments passed to the parent Metric class.
    """

    def __init__(self, name: str = 'smape', **kwargs) -> None:
        super(SMAPE, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
            sample_weight: Optional[tf.Tensor] = None
    ) -> None:
        """Update the metric state with new predictions.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth values.
        y_pred : tf.Tensor
            Predicted values.
        sample_weight : tf.Tensor, optional
            Optional weighting of each sample.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        numerator = tf.abs(y_true - y_pred)
        epsilon = tf.constant(1e-7, dtype=tf.float32)
        denominator = tf.abs(y_true) + tf.abs(y_pred) + epsilon

        values = 200.0 * (numerator / denominator)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            values = values * sample_weight

        self.total.assign_add(tf.reduce_sum(values))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self) -> tf.Tensor:
        """Compute the current metric value.

        Returns
        -------
        tf.Tensor
            The sMAPE value.
        """
        return self.total / self.count

    def reset_state(self) -> None:
        """Reset the metric state."""
        self.total.assign(0.0)
        self.count.assign(0.0)


# ---------------------------------------------------------------------
# Metric Calculation Utilities
# ---------------------------------------------------------------------


def calculate_comprehensive_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        backcast: np.ndarray
) -> Dict[str, float]:
    """Calculate comprehensive time series forecasting metrics.

    Computes MAE, RMSE, sMAPE, rMAE, and MASE for forecast evaluation.
    The rMAE and MASE metrics require historical data (backcast) to establish
    baseline comparisons.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values with shape (batch, forecast_len, features).
    y_pred : np.ndarray
        Predicted values with shape (batch, forecast_len, features).
    backcast : np.ndarray
        Historical values with shape (batch, backcast_len, features).
        Used to compute naive baseline and scaling factor.

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - MAE: Mean Absolute Error
        - RMSE: Root Mean Squared Error
        - sMAPE: Symmetric Mean Absolute Percentage Error (0-200 scale)
        - rMAE: Relative MAE compared to naive last-value baseline
        - MASE: Mean Absolute Scaled Error using random walk scaling

    Notes
    -----
    - sMAPE: Symmetric percentage error, bounded [0, 200]
    - rMAE: MAE normalized by naive forecast MAE (values <1 indicate better than naive)
    - MASE: MAE scaled by historical first-difference MAE (values <1 indicate better than random walk)
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
        "MASE": mase
    }