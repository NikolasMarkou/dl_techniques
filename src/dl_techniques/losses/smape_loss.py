"""
SMAPE Loss Function for Time Series Forecasting.

This module implements the Symmetric Mean Absolute Percentage Error (SMAPE) loss
function as used in the N-BEATS paper.
"""

import keras
from keras import ops
from typing import Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SMAPELoss(keras.losses.Loss):
    """Symmetric Mean Absolute Percentage Error (SMAPE) loss.

    SMAPE is a commonly used loss function for time series forecasting.
    It is defined as:

    SMAPE = 200 * mean(|y_true - y_pred| / (|y_true| + |y_pred| + epsilon))

    The loss handles zero values by masking them out and using a small epsilon
    value to avoid division by zero.

    Args:
        epsilon: Float, small value to avoid division by zero (default: 1e-8).
        name: String, name of the loss function.
        **kwargs: Additional keyword arguments for the Loss parent class.
    """

    def __init__(
            self,
            epsilon: float = 1e-8,
            name: str = 'smape_loss',
            **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.epsilon = epsilon

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute SMAPE loss.

        Args:
            y_true: Ground truth tensor of shape (batch_size, ...).
            y_pred: Predicted tensor of shape (batch_size, ...).

        Returns:
            SMAPE loss tensor.
        """
        # Cast to float32 to ensure proper computation
        y_true = ops.cast(y_true, 'float32')
        y_pred = ops.cast(y_pred, 'float32')

        # Create mask for non-zero values
        mask = ops.cast(ops.not_equal(y_true, 0.0), 'float32')

        # Compute absolute values
        abs_true = ops.abs(y_true)
        abs_pred = ops.abs(y_pred)
        abs_diff = ops.abs(y_pred - y_true)

        # Compute symmetric sum
        sym_sum = abs_true + abs_pred

        # Create condition for valid denominators
        condition = ops.greater(sym_sum, self.epsilon)

        # Compute weights (inverse of symmetric sum)
        weights = ops.where(
            condition,
            1.0 / (sym_sum + self.epsilon),
            0.0
        )

        # Compute SMAPE components
        smape_components = abs_diff * weights * mask

        # Return 200 * mean(smape_components)
        return 200.0 * ops.mean(smape_components)

    def get_config(self) -> dict:
        """Get loss configuration."""
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'SMAPELoss':
        """Create loss from configuration."""
        return cls(**config)

# ---------------------------------------------------------------------

def smape_loss(
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        epsilon: float = 1e-8
) -> keras.KerasTensor:
    """Functional interface for SMAPE loss.

    Args:
        y_true: Ground truth tensor.
        y_pred: Predicted tensor.
        epsilon: Small value to avoid division by zero.

    Returns:
        SMAPE loss tensor.
    """
    loss_fn = SMAPELoss(epsilon=epsilon)
    return loss_fn(y_true, y_pred)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MASELoss(keras.losses.Loss):
    """Mean Absolute Scaled Error (MASE) loss.

    MASE is another common loss function for time series forecasting that
    scales the absolute error by the mean absolute error of a naive seasonal forecast.

    Args:
        seasonal_period: Integer, seasonality period for the naive forecast.
        epsilon: Float, small value to avoid division by zero.
        name: String, name of the loss function.
        **kwargs: Additional keyword arguments for the Loss parent class.
    """

    def __init__(
            self,
            seasonal_period: int = 1,
            epsilon: float = 1e-8,
            name: str = 'mase_loss',
            **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.seasonal_period = seasonal_period
        self.epsilon = epsilon

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute MASE loss.

        Args:
            y_true: Ground truth tensor of shape (batch_size, sequence_length, ...).
            y_pred: Predicted tensor of shape (batch_size, sequence_length, ...).

        Returns:
            MASE loss tensor.
        """
        # Cast to float32
        y_true = ops.cast(y_true, 'float32')
        y_pred = ops.cast(y_pred, 'float32')

        # Compute forecast error
        forecast_error = ops.abs(y_true - y_pred)

        # Compute naive seasonal forecast error
        if self.seasonal_period > 1:
            # Seasonal naive forecast
            naive_forecast = y_true[..., :-self.seasonal_period, :]
            naive_actual = y_true[..., self.seasonal_period:, :]
            naive_error = ops.abs(naive_actual - naive_forecast)
        else:
            # Simple naive forecast (previous value)
            naive_forecast = y_true[..., :-1, :]
            naive_actual = y_true[..., 1:, :]
            naive_error = ops.abs(naive_actual - naive_forecast)

        # Compute mean naive error
        mean_naive_error = ops.mean(naive_error) + self.epsilon

        # Compute MASE
        mase = forecast_error / mean_naive_error

        return ops.mean(mase)

    def get_config(self) -> dict:
        """Get loss configuration."""
        config = super().get_config()
        config.update({
            'seasonal_period': self.seasonal_period,
            'epsilon': self.epsilon,
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'MASELoss':
        """Create loss from configuration."""
        return cls(**config)

# ---------------------------------------------------------------------

def mase_loss(
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        seasonal_period: int = 1,
        epsilon: float = 1e-8
) -> keras.KerasTensor:
    """Functional interface for MASE loss.

    Args:
        y_true: Ground truth tensor.
        y_pred: Predicted tensor.
        seasonal_period: Seasonality period for naive forecast.
        epsilon: Small value to avoid division by zero.

    Returns:
        MASE loss tensor.
    """
    loss_fn = MASELoss(seasonal_period=seasonal_period, epsilon=epsilon)
    return loss_fn(y_true, y_pred)

# ---------------------------------------------------------------------
