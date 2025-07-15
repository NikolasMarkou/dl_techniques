"""
N-BEATS Block Layers for Time Series Forecasting.

This module implements the building blocks for the N-BEATS architecture
with proper gradient flow and shape handling.
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
    """Base N-BEATS block layer with proper gradient flow.

    This is the base class for all N-BEATS blocks. It implements the common
    architecture of four fully connected layers followed by theta parameter
    generation. Subclasses must implement the basis functions for generating
    backcast and forecast from theta parameters.

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
        """Build the layer weights."""
        self._build_input_shape = input_shape

        # Four fully connected layers with proper initialization
        self.dense1 = keras.layers.Dense(
            self.units,
            activation='relu',
            kernel_initializer='he_normal',
            name='dense1'
        )
        self.dense2 = keras.layers.Dense(
            self.units,
            activation='relu',
            kernel_initializer='he_normal',
            name='dense2'
        )
        self.dense3 = keras.layers.Dense(
            self.units,
            activation='relu',
            kernel_initializer='he_normal',
            name='dense3'
        )
        self.dense4 = keras.layers.Dense(
            self.units,
            activation='relu',
            kernel_initializer='he_normal',
            name='dense4'
        )

        # Theta layers with proper initialization
        self.theta_backcast = keras.layers.Dense(
            self.thetas_dim,
            activation='linear',
            use_bias=False,
            kernel_initializer='glorot_uniform',
            name='theta_backcast'
        )
        self.theta_forecast = keras.layers.Dense(
            self.thetas_dim,
            activation='linear',
            use_bias=False,
            kernel_initializer='glorot_uniform',
            name='theta_forecast'
        )

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> Tuple[
        keras.KerasTensor, keras.KerasTensor]:
        """Forward pass of the N-BEATS block.

        Args:
            inputs: Input tensor of shape (batch_size, backcast_length).
            training: Boolean indicating training mode.

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

        # Generate backcast and forecast
        backcast = self._generate_backcast(theta_b)
        forecast = self._generate_forecast(theta_f)

        return backcast, forecast

    def _generate_backcast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate backcast from theta parameters.

        Args:
            theta: Theta parameters tensor.

        Returns:
            Backcast tensor.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement _generate_backcast")

    def _generate_forecast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate forecast from theta parameters.

        Args:
            theta: Theta parameters tensor.

        Returns:
            Forecast tensor.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement _generate_forecast")

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[
        Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Tuple containing (backcast_shape, forecast_shape).
        """
        batch_size = input_shape[0]
        backcast_shape = (batch_size, self.backcast_length)
        forecast_shape = (batch_size, self.forecast_length)
        return backcast_shape, forecast_shape

    def get_config(self) -> dict:
        """Get layer configuration for serialization.

        Returns:
            Configuration dictionary.
        """
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
        """Get build configuration for serialization.

        Returns:
            Build configuration dictionary.
        """
        return {'input_shape': self._build_input_shape}

    def build_from_config(self, config: dict) -> None:
        """Build layer from configuration.

        Args:
            config: Build configuration dictionary.
        """
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class GenericBlock(NBeatsBlock):
    """Generic N-BEATS block with learnable linear transformations.

    This block uses learnable linear transformations to map theta parameters
    to backcast and forecast sequences. It's the most flexible block type
    but requires more parameters to learn.
    """

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the generic block."""
        super().build(input_shape)

        # Learnable transformations from theta to backcast/forecast
        self.backcast_basis = keras.layers.Dense(
            self.backcast_length,
            activation='linear',
            use_bias=False,
            kernel_initializer='glorot_uniform',
            name='backcast_basis'
        )
        self.forecast_basis = keras.layers.Dense(
            self.forecast_length,
            activation='linear',
            use_bias=False,
            kernel_initializer='glorot_uniform',
            name='forecast_basis'
        )

    def _generate_backcast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate backcast using learnable basis functions.

        Args:
            theta: Theta parameters tensor of shape (batch_size, thetas_dim).

        Returns:
            Backcast tensor of shape (batch_size, backcast_length).
        """
        return self.backcast_basis(theta)

    def _generate_forecast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate forecast using learnable basis functions.

        Args:
            theta: Theta parameters tensor of shape (batch_size, thetas_dim).

        Returns:
            Forecast tensor of shape (batch_size, forecast_length).
        """
        return self.forecast_basis(theta)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TrendBlock(NBeatsBlock):
    """Trend N-BEATS block with polynomial basis functions.

    This block uses polynomial basis functions to capture trend patterns
    in the time series. It's particularly effective for data with clear
    polynomial trends.
    """

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the trend block."""
        super().build(input_shape)

        # Create polynomial basis matrices
        self._create_polynomial_basis()

    def _create_polynomial_basis(self) -> None:
        """Create polynomial basis matrices for trend modeling."""
        # Create time grids - backcast uses past (negative) time, forecast uses future (positive) time
        backcast_grid = np.linspace(-1, 0, self.backcast_length, dtype=np.float32)
        forecast_grid = np.linspace(0, 1, self.forecast_length, dtype=np.float32)

        # Create polynomial basis matrices
        backcast_basis = np.zeros((self.thetas_dim, self.backcast_length), dtype=np.float32)
        forecast_basis = np.zeros((self.thetas_dim, self.forecast_length), dtype=np.float32)

        # Generate polynomial terms (1, t, t^2, t^3, ...)
        for i in range(self.thetas_dim):
            backcast_basis[i] = np.power(backcast_grid, i)
            forecast_basis[i] = np.power(forecast_grid, i)

        # Store as non-trainable weights and set values immediately
        self.backcast_basis_matrix = self.add_weight(
            name='backcast_basis_matrix',
            shape=(self.thetas_dim, self.backcast_length),
            initializer=lambda shape, dtype: backcast_basis,
            trainable=False
        )
        self.forecast_basis_matrix = self.add_weight(
            name='forecast_basis_matrix',
            shape=(self.thetas_dim, self.forecast_length),
            initializer=lambda shape, dtype: forecast_basis,
            trainable=False
        )

    def _generate_backcast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate backcast using polynomial basis functions.

        Args:
            theta: Theta parameters tensor of shape (batch_size, thetas_dim).

        Returns:
            Backcast tensor of shape (batch_size, backcast_length).
        """
        return ops.matmul(theta, self.backcast_basis_matrix)

    def _generate_forecast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate forecast using polynomial basis functions.

        Args:
            theta: Theta parameters tensor of shape (batch_size, thetas_dim).

        Returns:
            Forecast tensor of shape (batch_size, forecast_length).
        """
        return ops.matmul(theta, self.forecast_basis_matrix)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SeasonalityBlock(NBeatsBlock):
    """Seasonality N-BEATS block with Fourier basis functions.

    This block uses Fourier (sine and cosine) basis functions to capture
    seasonal patterns in the time series. It's particularly effective for
    data with periodic behavior.
    """

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the seasonality block."""
        super().build(input_shape)

        # Create Fourier basis matrices
        self._create_fourier_basis()

    def _create_fourier_basis(self) -> None:
        """Create Fourier basis matrices for seasonality modeling."""
        # Create a continuous set of integer time steps for both backcast and forecast.
        # This ensures that the frequency of the seasonal patterns is consistent
        # when extrapolated from the backcast to the forecast period.
        time_steps = np.arange(
            self.backcast_length + self.forecast_length, dtype=np.float32
        )

        # Split the time steps into separate grids for backcast and forecast
        backcast_grid = time_steps[:self.backcast_length]
        forecast_grid = time_steps[self.backcast_length:]

        # Create empty placeholder matrices for the basis functions
        backcast_basis = np.zeros((self.thetas_dim, self.backcast_length), dtype=np.float32)
        forecast_basis = np.zeros((self.thetas_dim, self.forecast_length), dtype=np.float32)

        # Each harmonic requires a sine and cosine component, so we have half as many
        # harmonics as the dimension of the theta parameters.
        num_harmonics = self.thetas_dim // 2

        # Define the fundamental period for the Fourier series. We assume the
        # most dominant seasonal patterns are related to the length of the lookback window.
        period = self.backcast_length

        for i in range(num_harmonics):
            # The harmonic number (e.g., 1, 2, 3, ...)
            harmonic = i + 1

            # Calculate the frequency for this harmonic
            frequency = 2 * np.pi * harmonic / period

            # Cosine terms
            cos_idx = 2 * i
            if cos_idx < self.thetas_dim:
                backcast_basis[cos_idx] = np.cos(frequency * backcast_grid)
                forecast_basis[cos_idx] = np.cos(frequency * forecast_grid)

            # Sine terms
            sin_idx = 2 * i + 1
            if sin_idx < self.thetas_dim:
                backcast_basis[sin_idx] = np.sin(frequency * backcast_grid)
                forecast_basis[sin_idx] = np.sin(frequency * forecast_grid)

        # If thetas_dim is an odd number, we use the last available dimension
        # to model a constant DC component (a simple offset).
        if self.thetas_dim % 2 == 1:
            backcast_basis[-1] = 1.0
            forecast_basis[-1] = 1.0

        # Store the generated basis matrices as non-trainable weights within the layer
        self.backcast_basis_matrix = self.add_weight(
            name='backcast_basis_matrix',
            shape=(self.thetas_dim, self.backcast_length),
            initializer=lambda shape, dtype: backcast_basis,
            trainable=False
        )
        self.forecast_basis_matrix = self.add_weight(
            name='forecast_basis_matrix',
            shape=(self.thetas_dim, self.forecast_length),
            initializer=lambda shape, dtype: forecast_basis,
            trainable=False
        )

    def _generate_backcast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate backcast using Fourier basis functions.

        Args:
            theta: Theta parameters tensor of shape (batch_size, thetas_dim).

        Returns:
            Backcast tensor of shape (batch_size, backcast_length).
        """
        return ops.matmul(theta, self.backcast_basis_matrix)

    def _generate_forecast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """Generate forecast using Fourier basis functions.

        Args:
            theta: Theta parameters tensor of shape (batch_size, thetas_dim).

        Returns:
            Forecast tensor of shape (batch_size, forecast_length).
        """
        return ops.matmul(theta, self.forecast_basis_matrix)

# ---------------------------------------------------------------------
