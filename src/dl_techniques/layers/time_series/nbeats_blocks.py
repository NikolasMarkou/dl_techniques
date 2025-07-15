"""
N-BEATS Block Layers for Time Series Forecasting.

This module implements the building blocks for the N-BEATS architecture,
including generic, trend, and seasonality blocks.
"""

import keras
import numpy as np
from keras import ops
from typing import Optional, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NBeatsBlock(keras.layers.Layer):
    """Base N-BEATS block layer.

    This is the fundamental building block of the N-BEATS architecture.
    It processes input time series and produces both backcast and forecast outputs.

    Args:
        units: Integer, number of hidden units in the fully connected layers.
        thetas_dim: Integer, dimensionality of the theta parameters.
        backcast_length: Integer, length of the input time series.
        forecast_length: Integer, length of the forecast horizon.
        share_weights: Boolean, whether to share weights across blocks in the same stack.
        **kwargs: Additional keyword arguments for the Layer parent class.
    """

    def __init__(
            self,
            units: int,
            thetas_dim: int,
            backcast_length: int,
            forecast_length: int,
            share_weights: bool = False,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_weights = share_weights

        # Will be initialized in build()
        self.dense1 = None
        self.dense2 = None
        self.dense3 = None
        self.dense4 = None
        self.theta_backcast = None
        self.theta_forecast = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer."""
        self._build_input_shape = input_shape

        # Four fully connected layers
        self.dense1 = keras.layers.Dense(
            self.units,
            activation='relu',
            name='dense1'
        )
        self.dense2 = keras.layers.Dense(
            self.units,
            activation='relu',
            name='dense2'
        )
        self.dense3 = keras.layers.Dense(
            self.units,
            activation='relu',
            name='dense3'
        )
        self.dense4 = keras.layers.Dense(
            self.units,
            activation='relu',
            name='dense4'
        )

        # Theta layers (to be overridden by subclasses)
        self.theta_backcast = keras.layers.Dense(
            self.thetas_dim,
            activation='linear',
            use_bias=False,
            name='theta_backcast'
        )
        self.theta_forecast = keras.layers.Dense(
            self.thetas_dim,
            activation='linear',
            use_bias=False,
            name='theta_forecast'
        )

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> Tuple[
        keras.KerasTensor, keras.KerasTensor]:
        """Forward pass of the N-BEATS block.

        Args:
            inputs: Input tensor of shape (batch_size, backcast_length).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Tuple of (backcast, forecast) tensors.
        """
        # Pass through four fully connected layers
        x = self.dense1(inputs, training=training)
        x = self.dense2(x, training=training)
        x = self.dense3(x, training=training)
        x = self.dense4(x, training=training)

        # Generate theta parameters
        theta_b = self.theta_backcast(x, training=training)
        theta_f = self.theta_forecast(x, training=training)

        # Generate backcast and forecast (to be implemented by subclasses)
        backcast = self._generate_backcast(theta_b)
        forecast = self._generate_forecast(theta_f)

        return backcast, forecast

    def _generate_backcast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate backcast from theta parameters.

        Args:
            theta: Theta parameters tensor.

        Returns:
            Backcast tensor.
        """
        raise NotImplementedError("Subclasses must implement _generate_backcast")

    def _generate_forecast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate forecast from theta parameters.

        Args:
            theta: Theta parameters tensor.

        Returns:
            Forecast tensor.
        """
        raise NotImplementedError("Subclasses must implement _generate_forecast")

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[
        Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Tuple of (backcast_shape, forecast_shape).
        """
        batch_size = input_shape[0]
        backcast_shape = (batch_size, self.backcast_length)
        forecast_shape = (batch_size, self.forecast_length)
        return backcast_shape, forecast_shape

    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'thetas_dim': self.thetas_dim,
            'backcast_length': self.backcast_length,
            'forecast_length': self.forecast_length,
            'share_weights': self.share_weights,
        })
        return config

    def get_build_config(self) -> dict:
        """Get build configuration."""
        return {'input_shape': self._build_input_shape}

    def build_from_config(self, config: dict) -> None:
        """Build from configuration."""
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])


@keras.saving.register_keras_serializable()
class GenericBlock(NBeatsBlock):
    """Generic N-BEATS block.

    This block uses generic basis functions and learns the backcast and forecast
    directly without any specific structure.

    Args:
        units: Integer, number of hidden units.
        thetas_dim: Integer, dimensionality of theta parameters.
        backcast_length: Integer, length of the input time series.
        forecast_length: Integer, length of the forecast horizon.
        share_weights: Boolean, whether to share weights across blocks.
        **kwargs: Additional keyword arguments.
    """

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the generic block."""
        super().build(input_shape)

        # Override with separate theta layers for backcast and forecast
        self.theta_backcast = keras.layers.Dense(
            self.thetas_dim,
            activation='linear',
            use_bias=False,
            name='theta_backcast'
        )
        self.theta_forecast = keras.layers.Dense(
            self.thetas_dim,
            activation='linear',
            use_bias=False,
            name='theta_forecast'
        )

        # Generic basis functions
        self.backcast_basis = keras.layers.Dense(
            self.backcast_length,
            activation='linear',
            name='backcast_basis'
        )
        self.forecast_basis = keras.layers.Dense(
            self.forecast_length,
            activation='linear',
            name='forecast_basis'
        )

    def _generate_backcast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate backcast using generic basis functions."""
        return self.backcast_basis(theta)

    def _generate_forecast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate forecast using generic basis functions."""
        return self.forecast_basis(theta)


@keras.saving.register_keras_serializable()
class TrendBlock(NBeatsBlock):
    """Trend N-BEATS block.

    This block uses polynomial basis functions to model trend components
    in the time series.

    Args:
        units: Integer, number of hidden units.
        thetas_dim: Integer, polynomial degree (number of polynomial coefficients).
        backcast_length: Integer, length of the input time series.
        forecast_length: Integer, length of the forecast horizon.
        share_weights: Boolean, whether to share weights across blocks.
        **kwargs: Additional keyword arguments.
    """

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the trend block."""
        super().build(input_shape)

        # Trend blocks share the same theta for backcast and forecast
        self.theta_trend = keras.layers.Dense(
            self.thetas_dim,
            activation='linear',
            use_bias=False,
            name='theta_trend'
        )

        # Precompute polynomial basis matrices
        self._setup_polynomial_basis()

    def _setup_polynomial_basis(self) -> None:
        """Setup polynomial basis matrices."""
        # Backcast time grid (normalized to [0, 1])
        backcast_time = ops.cast(ops.arange(0, self.backcast_length), 'float32') / float(self.backcast_length)

        # Forecast time grid (normalized to [0, 1])
        forecast_time = ops.cast(ops.arange(0, self.forecast_length), 'float32') / float(self.forecast_length)

        # Create polynomial basis matrices
        backcast_basis = []
        forecast_basis = []

        for i in range(self.thetas_dim):
            if i == 0:
                # Constant term
                backcast_basis.append(ops.ones_like(backcast_time))
                forecast_basis.append(ops.ones_like(forecast_time))
            else:
                backcast_basis.append(ops.power(backcast_time, float(i)))
                forecast_basis.append(ops.power(forecast_time, float(i)))

        self.backcast_basis_matrix = ops.stack(backcast_basis, axis=0)  # (thetas_dim, backcast_length)
        self.forecast_basis_matrix = ops.stack(forecast_basis, axis=0)  # (thetas_dim, forecast_length)

    def _generate_backcast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate backcast using polynomial basis functions."""
        theta_trend = self.theta_trend(theta)
        return ops.matmul(theta_trend, self.backcast_basis_matrix)

    def _generate_forecast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate forecast using polynomial basis functions."""
        theta_trend = self.theta_trend(theta)
        return ops.matmul(theta_trend, self.forecast_basis_matrix)


@keras.saving.register_keras_serializable()
class SeasonalityBlock(NBeatsBlock):
    """Seasonality N-BEATS block.

    This block uses Fourier basis functions to model seasonal components
    in the time series.

    Args:
        units: Integer, number of hidden units.
        thetas_dim: Integer, number of Fourier harmonics.
        backcast_length: Integer, length of the input time series.
        forecast_length: Integer, length of the forecast horizon.
        share_weights: Boolean, whether to share weights across blocks.
        **kwargs: Additional keyword arguments.
    """

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the seasonality block."""
        super().build(input_shape)

        # Seasonality blocks have separate theta layers
        self.theta_backcast = keras.layers.Dense(
            self.thetas_dim,
            activation='linear',
            use_bias=False,
            name='theta_backcast'
        )
        self.theta_forecast = keras.layers.Dense(
            self.thetas_dim,
            activation='linear',
            use_bias=False,
            name='theta_forecast'
        )

        # Setup Fourier basis matrices
        self._setup_fourier_basis()

    def _setup_fourier_basis(self) -> None:
        """Setup Fourier basis matrices."""
        # Number of cos/sin pairs
        p1 = self.thetas_dim // 2
        p2 = self.thetas_dim - p1

        # Backcast time grid
        backcast_time = ops.arange(0, self.backcast_length, dtype='float32') / self.forecast_length

        # Forecast time grid
        forecast_time = ops.arange(0, self.forecast_length, dtype='float32') / self.forecast_length

        # Create Fourier basis matrices
        backcast_basis = []
        forecast_basis = []

        # Cosine terms
        for i in range(p1):
            backcast_basis.append(ops.cos(2 * np.pi * i * backcast_time))
            forecast_basis.append(ops.cos(2 * np.pi * i * forecast_time))

        # Sine terms
        for i in range(p2):
            backcast_basis.append(ops.sin(2 * np.pi * i * backcast_time))
            forecast_basis.append(ops.sin(2 * np.pi * i * forecast_time))

        self.backcast_basis_matrix = ops.stack(backcast_basis, axis=0)  # (thetas_dim, backcast_length)
        self.forecast_basis_matrix = ops.stack(forecast_basis, axis=0)  # (thetas_dim, forecast_length)

    def _generate_backcast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate backcast using Fourier basis functions."""
        return ops.matmul(theta, self.backcast_basis_matrix)

    def _generate_forecast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate forecast using Fourier basis functions."""
        return ops.matmul(theta, self.forecast_basis_matrix)