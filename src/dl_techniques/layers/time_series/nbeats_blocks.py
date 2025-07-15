"""
Fixed N-BEATS Block Layers for Time Series Forecasting.

This module implements corrected building blocks for the N-BEATS architecture.
"""

import keras
import numpy as np
from keras import ops
from typing import Optional, Any, Tuple

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class NBeatsBlock(keras.layers.Layer):
    """Base N-BEATS block layer.

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

        # Theta layers
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
        """Forward pass of the N-BEATS block."""
        # Pass through four fully connected layers
        x = self.dense1(inputs, training=training)
        x = self.dense2(x, training=training)
        x = self.dense3(x, training=training)
        x = self.dense4(x, training=training)

        # Generate theta parameters
        theta_b = self.theta_backcast(x, training=training)
        theta_f = self.theta_forecast(x, training=training)

        # Generate backcast and forecast using theta directly
        backcast = self._generate_backcast(theta_b)
        forecast = self._generate_forecast(theta_f)

        return backcast, forecast

    def _generate_backcast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate backcast from theta parameters."""
        raise NotImplementedError("Subclasses must implement _generate_backcast")

    def _generate_forecast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate forecast from theta parameters."""
        raise NotImplementedError("Subclasses must implement _generate_forecast")

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[
        Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Compute the output shape of the layer."""
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
    """Generic N-BEATS block with learnable basis functions."""

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the generic block."""
        super().build(input_shape)

        # Learnable basis functions that map theta to backcast/forecast
        self.backcast_basis = keras.layers.Dense(
            self.backcast_length,
            activation='linear',
            use_bias=False,
            name='backcast_basis'
        )
        self.forecast_basis = keras.layers.Dense(
            self.forecast_length,
            activation='linear',
            use_bias=False,
            name='forecast_basis'
        )

    def _generate_backcast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate backcast using learnable basis functions."""
        return self.backcast_basis(theta)

    def _generate_forecast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate forecast using learnable basis functions."""
        return self.forecast_basis(theta)


@keras.saving.register_keras_serializable()
class TrendBlock(NBeatsBlock):
    """Trend N-BEATS block with polynomial basis functions."""

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the trend block."""
        super().build(input_shape)

        # Setup polynomial basis matrices
        self._setup_polynomial_basis()

    def _setup_polynomial_basis(self) -> None:
        """Setup polynomial basis matrices as layer weights."""
        # Create basis matrices
        backcast_basis_np = np.zeros((self.thetas_dim, self.backcast_length), dtype=np.float32)
        forecast_basis_np = np.zeros((self.thetas_dim, self.forecast_length), dtype=np.float32)

        # CORRECTED: Proper time grids
        # Backcast time grid (normalized to [-1, 0])
        backcast_time = (np.arange(self.backcast_length, dtype=np.float32) - self.backcast_length) / self.backcast_length

        # Forecast time grid (normalized to [0, 1])
        forecast_time = (np.arange(self.forecast_length, dtype=np.float32) + 1) / self.forecast_length

        # Create polynomial basis matrices
        for i in range(self.thetas_dim):
            backcast_basis_np[i, :] = np.power(backcast_time, i)
            forecast_basis_np[i, :] = np.power(forecast_time, i)

        # Add as non-trainable weights
        self.backcast_basis_matrix = self.add_weight(
            name='backcast_basis_matrix',
            shape=(self.thetas_dim, self.backcast_length),
            initializer='zeros',
            trainable=False
        )
        self.forecast_basis_matrix = self.add_weight(
            name='forecast_basis_matrix',
            shape=(self.thetas_dim, self.forecast_length),
            initializer='zeros',
            trainable=False
        )

        # Set the values
        self.backcast_basis_matrix.assign(backcast_basis_np)
        self.forecast_basis_matrix.assign(forecast_basis_np)

    def _generate_backcast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate backcast using polynomial basis functions."""
        # CORRECTED: Use theta directly with basis matrix
        return ops.matmul(theta, self.backcast_basis_matrix)

    def _generate_forecast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate forecast using polynomial basis functions."""
        # CORRECTED: Use theta directly with basis matrix
        return ops.matmul(theta, self.forecast_basis_matrix)


@keras.saving.register_keras_serializable()
class SeasonalityBlock(NBeatsBlock):
    """Seasonality N-BEATS block with Fourier basis functions."""

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the seasonality block."""
        super().build(input_shape)

        # Setup Fourier basis matrices
        self._setup_fourier_basis()

    def _setup_fourier_basis(self) -> None:
        """Setup Fourier basis matrices as layer weights."""
        # Number of cos/sin pairs
        half_thetas = self.thetas_dim // 2

        # Create basis matrices
        backcast_basis_np = np.zeros((self.thetas_dim, self.backcast_length), dtype=np.float32)
        forecast_basis_np = np.zeros((self.thetas_dim, self.forecast_length), dtype=np.float32)

        # CORRECTED: Proper time grids
        # Backcast time grid (normalized to [0, 1])
        backcast_time = np.arange(self.backcast_length, dtype=np.float32) / self.backcast_length

        # Forecast time grid (continuing from backcast, normalized)
        forecast_time = (np.arange(self.forecast_length, dtype=np.float32) + self.backcast_length) / (self.backcast_length + self.forecast_length)

        # Create Fourier basis matrices
        for i in range(half_thetas):
            # Cosine terms
            backcast_basis_np[2*i, :] = np.cos(2 * np.pi * (i + 1) * backcast_time)
            forecast_basis_np[2*i, :] = np.cos(2 * np.pi * (i + 1) * forecast_time)

            # Sine terms
            if 2*i + 1 < self.thetas_dim:
                backcast_basis_np[2*i + 1, :] = np.sin(2 * np.pi * (i + 1) * backcast_time)
                forecast_basis_np[2*i + 1, :] = np.sin(2 * np.pi * (i + 1) * forecast_time)

        # Add as non-trainable weights
        self.backcast_basis_matrix = self.add_weight(
            name='backcast_basis_matrix',
            shape=(self.thetas_dim, self.backcast_length),
            initializer='zeros',
            trainable=False
        )
        self.forecast_basis_matrix = self.add_weight(
            name='forecast_basis_matrix',
            shape=(self.thetas_dim, self.forecast_length),
            initializer='zeros',
            trainable=False
        )

        # Set the values
        self.backcast_basis_matrix.assign(backcast_basis_np)
        self.forecast_basis_matrix.assign(forecast_basis_np)

    def _generate_backcast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate backcast using Fourier basis functions."""
        # CORRECTED: Use theta directly with basis matrix
        return ops.matmul(theta, self.backcast_basis_matrix)

    def _generate_forecast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate forecast using Fourier basis functions."""
        # CORRECTED: Use theta directly with basis matrix
        return ops.matmul(theta, self.forecast_basis_matrix)