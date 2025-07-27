"""
N-BEATS Block Layers for Time Series Forecasting.

This module implements the building blocks for the N-BEATS (Neural Basis Expansion
Analysis for Time Series) architecture, a deep neural network designed for
univariate time series forecasting that achieves state-of-the-art performance
while maintaining interpretability.

Architecture Overview
====================

N-BEATS is based on a hierarchical doubly residual architecture with the following
key components:

1. **Stacks**: Collections of blocks that focus on different aspects of the forecast
   - Trend Stack: Captures long-term trends using polynomial basis functions
   - Seasonality Stack: Captures periodic patterns using Fourier basis functions
   - Generic Stack: Learns flexible representations using linear transformations

2. **Blocks**: Individual processing units within stacks that generate:
   - Backcast: Reconstruction of the input sequence (for residual connections)
   - Forecast: Contribution to the final prediction

3. **Basis Functions**: Mathematical functions that provide interpretable decomposition:
   - Polynomial: For trend modeling (t, t², t³, ...)
   - Fourier: For seasonality modeling (sin, cos at various frequencies)
   - Learnable: For flexible pattern capture

Key Features
============

- **Interpretability**: Each block's contribution can be analyzed separately
- **Doubly Residual**: Both forecast and backcast have residual connections
- **Stack Specialization**: Different stacks capture different temporal patterns
- **Parameter Sharing**: Optional weight sharing within stacks for regularization
- **Pure Deep Learning**: No feature engineering or domain knowledge required

Block Types
===========

1. **NBeatsBlock (Base Class)**
   - Common architecture with 4 fully connected layers
   - Generates theta parameters for basis function coefficients
   - Abstract methods for basis function implementation

2. **GenericBlock**
   - Uses learnable linear transformations as basis functions
   - Most flexible but requires more parameters
   - Good for capturing complex, non-standard patterns
   - Best for: Initial exploration, complex datasets

3. **TrendBlock**
   - Uses polynomial basis functions (1, t, t², t³, ...)
   - Captures long-term trends and growth patterns
   - Interpretable coefficients show trend components
   - Best for: Data with clear trends, economic forecasting

4. **SeasonalityBlock**
   - Uses Fourier basis functions (sin/cos at various frequencies)
   - Captures periodic and seasonal patterns
   - Interpretable coefficients show frequency components
   - Best for: Data with seasonality, demand forecasting

Mathematical Foundation
=======================

Each block follows this computation:

1. **Feature Extraction**: x → FC₁ → FC₂ → FC₃ → FC₄ → h
2. **Parameter Generation**:
   - θᵦ = Linear(h)  # Backcast coefficients
   - θf = Linear(h)  # Forecast coefficients
3. **Basis Expansion**:
   - Backcast = Σᵢ θᵦᵢ × Bᵢ(t_past)
   - Forecast = Σᵢ θfᵢ × Bᵢ(t_future)

Where B(t) are the basis functions:
- Generic: Bᵢ(t) = learnable linear transformation
- Trend: Bᵢ(t) = tᵢ (polynomial terms)
- Seasonal: Bᵢ(t) = [sin(2πft), cos(2πft)] (Fourier terms)
"""

import keras
import numpy as np
from keras import ops
from typing import Optional, Any, Tuple, Union

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
            Must be positive.
        thetas_dim: Integer, dimensionality of the theta parameters.
            Must be positive.
        backcast_length: Integer, length of the input time series.
            Must be positive.
        forecast_length: Integer, length of the forecast horizon.
            Must be positive.
        share_weights: Boolean, whether to share weights across blocks in the same stack.
        activation: String or callable, activation function for hidden layers.
        kernel_initializer: String or Initializer, initializer for FC layers.
        theta_initializer: String or Initializer, initializer for theta layers.
        kernel_regularizer: Optional regularizer for FC layer weights.
        theta_regularizer: Optional regularizer for theta layer weights.
        **kwargs: Additional keyword arguments for the Layer parent class.

    Raises:
        ValueError: If any dimension parameter is not positive.
    """

    def __init__(
            self,
            units: int,
            thetas_dim: int,
            backcast_length: int,
            forecast_length: int,
            share_weights: bool = False,
            activation: Union[str, callable] = 'relu',
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
            theta_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            theta_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        if thetas_dim <= 0:
            raise ValueError(f"thetas_dim must be positive, got {thetas_dim}")
        if backcast_length <= 0:
            raise ValueError(f"backcast_length must be positive, got {backcast_length}")
        if forecast_length <= 0:
            raise ValueError(f"forecast_length must be positive, got {forecast_length}")

        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_weights = share_weights
        self.activation = activation
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.theta_initializer = keras.initializers.get(theta_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.theta_regularizer = keras.regularizers.get(theta_regularizer)

        # Will be initialized in build()
        self.dense1 = None
        self.dense2 = None
        self.dense3 = None
        self.dense4 = None
        self.theta_backcast = None
        self.theta_forecast = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer weights.

        Args:
            input_shape: Shape of the input tensor.
        """
        self._build_input_shape = input_shape

        # Validate input shape
        if len(input_shape) != 2:
            raise ValueError(f"Expected 2D input shape, got {len(input_shape)}D: {input_shape}")

        # Four fully connected layers with proper initialization
        self.dense1 = keras.layers.Dense(
            self.units,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='dense1'
        )
        self.dense2 = keras.layers.Dense(
            self.units,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='dense2'
        )
        self.dense3 = keras.layers.Dense(
            self.units,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='dense3'
        )
        self.dense4 = keras.layers.Dense(
            self.units,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='dense4'
        )

        # Theta layers with proper initialization
        self.theta_backcast = keras.layers.Dense(
            self.thetas_dim,
            activation='linear',
            use_bias=False,
            kernel_initializer=self.theta_initializer,
            kernel_regularizer=self.theta_regularizer,
            name='theta_backcast'
        )
        self.theta_forecast = keras.layers.Dense(
            self.thetas_dim,
            activation='linear',
            use_bias=False,
            kernel_initializer=self.theta_initializer,
            kernel_regularizer=self.theta_regularizer,
            name='theta_forecast'
        )

        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None) -> Tuple[
        keras.KerasTensor, keras.KerasTensor]:
        """Forward pass of the N-BEATS block.

        Args:
            inputs: Input tensor of shape (batch_size, backcast_length).
            training: Boolean indicating training mode.

        Returns:
            Tuple of (backcast, forecast) tensors.

        Raises:
            ValueError: If input shape is incompatible.
        """
        # Validate input shape at runtime
        input_shape = ops.shape(inputs)
        if len(input_shape) != 2:
            raise ValueError(f"Expected 2D input, got shape: {input_shape}")

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

    def _generate_backcast(self, theta) -> keras.KerasTensor:
        """Generate backcast from theta parameters.

        Args:
            theta: Theta parameters tensor.

        Returns:
            Backcast tensor.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement _generate_backcast")

    def _generate_forecast(self, theta) -> keras.KerasTensor:
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
        if len(input_shape) != 2:
            raise ValueError(f"Expected 2D input shape, got {len(input_shape)}D: {input_shape}")

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
            'activation': self.activation,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'theta_initializer': keras.initializers.serialize(self.theta_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'theta_regularizer': keras.regularizers.serialize(self.theta_regularizer),
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

    Args:
        basis_initializer: String or Initializer, initializer for basis matrices.
        basis_regularizer: Optional regularizer for basis matrices.
        **kwargs: Arguments passed to parent NBeatsBlock.
    """

    def __init__(
        self,
        basis_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        basis_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.basis_initializer = keras.initializers.get(basis_initializer)
        self.basis_regularizer = keras.regularizers.get(basis_regularizer)

        # Will be initialized in build()
        self.backcast_basis = None
        self.forecast_basis = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the generic block.

        Args:
            input_shape: Shape of the input tensor.
        """
        super().build(input_shape)

        # Learnable transformations from theta to backcast/forecast
        self.backcast_basis = keras.layers.Dense(
            self.backcast_length,
            activation='linear',
            use_bias=False,
            kernel_initializer=self.basis_initializer,
            kernel_regularizer=self.basis_regularizer,
            name='backcast_basis'
        )
        self.forecast_basis = keras.layers.Dense(
            self.forecast_length,
            activation='linear',
            use_bias=False,
            kernel_initializer=self.basis_initializer,
            kernel_regularizer=self.basis_regularizer,
            name='forecast_basis'
        )

    def _generate_backcast(self, theta) -> keras.KerasTensor:
        """Generate backcast using learnable basis functions.

        Args:
            theta: Theta parameters tensor of shape (batch_size, thetas_dim).

        Returns:
            Backcast tensor of shape (batch_size, backcast_length).
        """
        return self.backcast_basis(theta)

    def _generate_forecast(self, theta) -> keras.KerasTensor:
        """Generate forecast using learnable basis functions.

        Args:
            theta: Theta parameters tensor of shape (batch_size, thetas_dim).

        Returns:
            Forecast tensor of shape (batch_size, forecast_length).
        """
        return self.forecast_basis(theta)

    def get_config(self) -> dict:
        """Get layer configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'basis_initializer': keras.initializers.serialize(self.basis_initializer),
            'basis_regularizer': keras.regularizers.serialize(self.basis_regularizer),
        })
        return config

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TrendBlock(NBeatsBlock):
    """Trend N-BEATS block with polynomial basis functions.

    This block uses polynomial basis functions to capture trend patterns
    in the time series. It's particularly effective for data with clear
    polynomial trends.

    Args:
        normalize_basis: Boolean, whether to normalize polynomial basis functions.
        **kwargs: Arguments passed to parent NBeatsBlock.
    """

    def __init__(
        self,
        normalize_basis: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.normalize_basis = normalize_basis

        # Will be initialized in build()
        self.backcast_basis_matrix = None
        self.forecast_basis_matrix = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the trend block.

        Args:
            input_shape: Shape of the input tensor.
        """
        super().build(input_shape)

        # Create polynomial basis matrices
        self._create_polynomial_basis()

    def _create_polynomial_basis(self) -> None:
        """Create polynomial basis matrices for trend modeling."""
        # Create time grids - backcast uses normalized past time, forecast uses normalized future time
        backcast_grid = np.linspace(-1, 0, self.backcast_length, dtype=np.float32)
        forecast_grid = np.linspace(0, 1, self.forecast_length, dtype=np.float32)

        # Create polynomial basis matrices
        backcast_basis = np.zeros((self.thetas_dim, self.backcast_length), dtype=np.float32)
        forecast_basis = np.zeros((self.thetas_dim, self.forecast_length), dtype=np.float32)

        # Generate polynomial terms (1, t, t^2, t^3, ...)
        for i in range(self.thetas_dim):
            backcast_basis[i] = np.power(backcast_grid, i)
            forecast_basis[i] = np.power(forecast_grid, i)

            # Optional normalization to prevent numerical issues
            if self.normalize_basis and i > 0:
                backcast_norm = np.linalg.norm(backcast_basis[i])
                forecast_norm = np.linalg.norm(forecast_basis[i])
                if backcast_norm > 1e-8:
                    backcast_basis[i] /= backcast_norm
                if forecast_norm > 1e-8:
                    forecast_basis[i] /= forecast_norm

        # Store as non-trainable weights
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
        self.backcast_basis_matrix.assign(backcast_basis)
        self.forecast_basis_matrix.assign(forecast_basis)

    def _generate_backcast(self, theta) -> keras.KerasTensor:
        """Generate backcast using polynomial basis functions.

        Args:
            theta: Theta parameters tensor of shape (batch_size, thetas_dim).

        Returns:
            Backcast tensor of shape (batch_size, backcast_length).
        """
        return ops.matmul(theta, self.backcast_basis_matrix)

    def _generate_forecast(self, theta) -> keras.KerasTensor:
        """Generate forecast using polynomial basis functions.

        Args:
            theta: Theta parameters tensor of shape (batch_size, thetas_dim).

        Returns:
            Forecast tensor of shape (batch_size, forecast_length).
        """
        return ops.matmul(theta, self.forecast_basis_matrix)

    def get_config(self) -> dict:
        """Get layer configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'normalize_basis': self.normalize_basis,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SeasonalityBlock(NBeatsBlock):
    """Seasonality N-BEATS block with Fourier basis functions.

    This block uses Fourier (sine and cosine) basis functions to capture
    seasonal patterns in the time series. It's particularly effective for
    data with periodic behavior.

    Args:
        normalize_basis: Boolean, whether to normalize Fourier basis functions.
        **kwargs: Arguments passed to parent NBeatsBlock.
    """

    def __init__(
        self,
        normalize_basis: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.normalize_basis = normalize_basis

        # Will be initialized in build()
        self.backcast_basis_matrix = None
        self.forecast_basis_matrix = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the seasonality block.

        Args:
            input_shape: Shape of the input tensor.
        """
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
        period = max(self.backcast_length, 12)  # Ensure minimum period for seasonality

        for i in range(num_harmonics):
            # The harmonic number (e.g., 1, 2, 3, ...)
            harmonic = i + 1

            # Calculate the frequency for this harmonic
            frequency = 2 * np.pi * harmonic / period

            # Cosine terms
            cos_idx = 2 * i
            if cos_idx < self.thetas_dim:
                cos_backcast = np.cos(frequency * backcast_grid)
                cos_forecast = np.cos(frequency * forecast_grid)

                # Optional normalization
                if self.normalize_basis:
                    cos_backcast_norm = np.linalg.norm(cos_backcast)
                    cos_forecast_norm = np.linalg.norm(cos_forecast)
                    if cos_backcast_norm > 1e-8:
                        cos_backcast /= cos_backcast_norm
                    if cos_forecast_norm > 1e-8:
                        cos_forecast /= cos_forecast_norm

                backcast_basis[cos_idx] = cos_backcast
                forecast_basis[cos_idx] = cos_forecast

            # Sine terms
            sin_idx = 2 * i + 1
            if sin_idx < self.thetas_dim:
                sin_backcast = np.sin(frequency * backcast_grid)
                sin_forecast = np.sin(frequency * forecast_grid)

                # Optional normalization
                if self.normalize_basis:
                    sin_backcast_norm = np.linalg.norm(sin_backcast)
                    sin_forecast_norm = np.linalg.norm(sin_forecast)
                    if sin_backcast_norm > 1e-8:
                        sin_backcast /= sin_backcast_norm
                    if sin_forecast_norm > 1e-8:
                        sin_forecast /= sin_forecast_norm

                backcast_basis[sin_idx] = sin_backcast
                forecast_basis[sin_idx] = sin_forecast

        # If thetas_dim is an odd number, we use the last available dimension
        # to model a constant DC component (a simple offset).
        if self.thetas_dim % 2 == 1:
            backcast_basis[-1] = 1.0
            forecast_basis[-1] = 1.0

        # Store the generated basis matrices as non-trainable weights within the layer
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
        self.backcast_basis_matrix.assign(backcast_basis)
        self.forecast_basis_matrix.assign(forecast_basis)

    def _generate_backcast(self, theta) -> keras.KerasTensor:
        """Generate backcast using Fourier basis functions.

        Args:
            theta: Theta parameters tensor of shape (batch_size, thetas_dim).

        Returns:
            Backcast tensor of shape (batch_size, backcast_length).
        """
        return ops.matmul(theta, self.backcast_basis_matrix)

    def _generate_forecast(self, theta) -> keras.KerasTensor:
        """Generate forecast using Fourier basis functions.

        Args:
            theta: Theta parameters tensor of shape (batch_size, thetas_dim).

        Returns:
            Forecast tensor of shape (batch_size, forecast_length).
        """
        return ops.matmul(theta, self.forecast_basis_matrix)

    def get_config(self) -> dict:
        """Get layer configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'normalize_basis': self.normalize_basis,
        })
        return config

# ---------------------------------------------------------------------