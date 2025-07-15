"""
Loss functions for time series forecasting.

This module implements specialized loss functions commonly used in time series
forecasting, including SMAPE (Symmetric Mean Absolute Percentage Error) and
MASE (Mean Absolute Scaled Error).
"""

import keras
from keras import ops
from typing import Optional, Union


@keras.saving.register_keras_serializable()
class SMAPELoss(keras.losses.Loss):
    """Symmetric Mean Absolute Percentage Error (SMAPE) loss.

    SMAPE is a percentage-based error metric that is symmetric and bounded.
    It's commonly used in time series forecasting competitions.

    The formula is:
    SMAPE = (100 / n) * Σ(|y_true - y_pred| / ((|y_true| + |y_pred|) / 2))

    Args:
        epsilon: Small value to avoid division by zero.
        name: Name of the loss function.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        epsilon: float = 1e-8,
        name: str = "smape_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.epsilon = epsilon

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute SMAPE loss.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            SMAPE loss value.
        """
        y_true = ops.cast(y_true, y_pred.dtype)

        # Calculate absolute differences
        abs_diff = ops.abs(y_true - y_pred)

        # Calculate denominator (average of absolute true and predicted values)
        denominator = (ops.abs(y_true) + ops.abs(y_pred)) / 2.0

        # Add epsilon to avoid division by zero
        denominator = ops.maximum(denominator, self.epsilon)

        # Calculate SMAPE - use 100 instead of 200 for better numerical stability
        smape = ops.mean(abs_diff / denominator) * 100.0

        return smape

    def get_config(self) -> dict:
        """Get loss configuration."""
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
        })
        return config


@keras.saving.register_keras_serializable()
class MASELoss(keras.losses.Loss):
    """Mean Absolute Scaled Error (MASE) loss.

    MASE is a scale-independent error metric that compares the forecast accuracy
    to a naive forecast. It's particularly useful for time series with different
    scales or when comparing forecasts across different series.

    The formula is:
    MASE = MAE / MAE_naive

    Where MAE_naive is the mean absolute error of a naive forecast (e.g., last value).

    Args:
        seasonal_periods: Number of periods in a season (for seasonal naive forecast).
        epsilon: Small value to avoid division by zero.
        name: Name of the loss function.
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

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute MASE loss.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            MASE loss value.
        """
        y_true = ops.cast(y_true, y_pred.dtype)

        # Calculate MAE of the forecast
        mae_forecast = ops.mean(ops.abs(y_true - y_pred))

        # Calculate naive forecast error (seasonal naive)
        if len(y_true.shape) > 1 and y_true.shape[1] > self.seasonal_periods:
            # For multi-step forecasts, use seasonal naive within the sequence
            naive_forecast = y_true[:, :-self.seasonal_periods]
            naive_true = y_true[:, self.seasonal_periods:]
            mae_naive = ops.mean(ops.abs(naive_true - naive_forecast))
        else:
            # For single-step or short sequences, use simple differences
            if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                # Multi-step case
                naive_errors = ops.abs(y_true[:, 1:] - y_true[:, :-1])
                mae_naive = ops.mean(naive_errors)
            else:
                # Single-step case - use a simple heuristic
                mae_naive = ops.mean(ops.abs(y_true))

        # Add epsilon to avoid division by zero
        mae_naive = mae_naive + self.epsilon

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


@keras.saving.register_keras_serializable()
class MQLoss(keras.losses.Loss):
    """Mean Quantile Loss for probabilistic forecasting.

    This loss function is used for quantile regression in time series forecasting,
    enabling the model to produce prediction intervals.

    Args:
        quantile: Quantile to predict (between 0 and 1).
        name: Name of the loss function.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        quantile: float = 0.5,
        name: str = "mq_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.quantile = quantile

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute quantile loss.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            Quantile loss value.
        """
        y_true = ops.cast(y_true, y_pred.dtype)

        # Calculate error
        error = y_true - y_pred

        # Calculate quantile loss
        loss = ops.maximum(self.quantile * error, (self.quantile - 1) * error)

        return ops.mean(loss)

    def get_config(self) -> dict:
        """Get loss configuration."""
        config = super().get_config()
        config.update({
            'quantile': self.quantile,
        })
        return config


@keras.saving.register_keras_serializable()
class HuberLoss(keras.losses.Loss):
    """Huber loss for robust time series forecasting.

    Huber loss is less sensitive to outliers than squared error loss.
    It's quadratic for small errors and linear for large errors.

    Args:
        delta: Threshold for switching from quadratic to linear loss.
        name: Name of the loss function.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        delta: float = 1.0,
        name: str = "huber_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.delta = delta

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute Huber loss.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            Huber loss value.
        """
        y_true = ops.cast(y_true, y_pred.dtype)

        # Calculate absolute error
        abs_error = ops.abs(y_true - y_pred)

        # Calculate Huber loss
        quadratic = 0.5 * ops.square(abs_error)
        linear = self.delta * abs_error - 0.5 * self.delta * self.delta

        # Use quadratic for small errors, linear for large errors
        loss = ops.where(abs_error <= self.delta, quadratic, linear)

        return ops.mean(loss)

    def get_config(self) -> dict:
        """Get loss configuration."""
        config = super().get_config()
        config.update({
            'delta': self.delta,
        })
        return config


def smape_metric(y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
    """SMAPE metric function for use with model.compile().

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        SMAPE metric value.
    """
    y_true = ops.cast(y_true, y_pred.dtype)

    abs_diff = ops.abs(y_true - y_pred)
    denominator = (ops.abs(y_true) + ops.abs(y_pred)) / 2.0 + 1e-8

    return ops.mean(abs_diff / denominator) * 100.0


def mase_metric(y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
    """MASE metric function for use with model.compile().

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        MASE metric value.
    """
    y_true = ops.cast(y_true, y_pred.dtype)

    mae_forecast = ops.mean(ops.abs(y_true - y_pred))

    # Simple naive forecast error calculation
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        naive_errors = ops.abs(y_true[:, 1:] - y_true[:, :-1])
        mae_naive = ops.mean(naive_errors)
    else:
        mae_naive = ops.mean(ops.abs(y_true))

    mae_naive = mae_naive + 1e-8

    return mae_forecast / mae_naive


def mape_metric(y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
    """MAPE (Mean Absolute Percentage Error) metric function.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        MAPE metric value.
    """
    y_true = ops.cast(y_true, y_pred.dtype)

    # Avoid division by zero
    mask = ops.abs(y_true) > 1e-8
    y_true_masked = ops.where(mask, y_true, ops.ones_like(y_true))
    y_pred_masked = ops.where(mask, y_pred, ops.zeros_like(y_pred))

    ape = ops.abs((y_true_masked - y_pred_masked) / y_true_masked)

    return ops.mean(ape) * 100.0


# Alias for backward compatibility
sMAPE = SMAPELoss
MASE = MASELoss