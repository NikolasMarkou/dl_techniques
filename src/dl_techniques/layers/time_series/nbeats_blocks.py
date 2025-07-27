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

Usage Examples
==============

Basic Usage:
```python
# Create individual blocks
trend_block = TrendBlock(
    units=128,
    thetas_dim=4,  # Polynomial degree up to t³
    backcast_length=96,  # 4 days of hourly data
    forecast_length=24   # 1 day forecast
)

seasonal_block = SeasonalityBlock(
    units=128,
    thetas_dim=20,  # 10 harmonics (sin/cos pairs)
    backcast_length=96,
    forecast_length=24
)

generic_block = GenericBlock(
    units=256,
    thetas_dim=32,
    backcast_length=96,
    forecast_length=24
)

# Use in a model
inputs = keras.Input(shape=(96,))
x = inputs

# Trend stack
for _ in range(3):
    backcast, forecast = trend_block(x)
    x = x - backcast  # Residual connection

# Seasonality stack
for _ in range(3):
    backcast, forecast = seasonal_block(x)
    x = x - backcast

# Final prediction combines all forecasts
```

Advanced Configuration:
```python
# With regularization and custom initialization
block = TrendBlock(
    units=64,
    thetas_dim=6,
    backcast_length=168,  # 1 week hourly
    forecast_length=24,   # 1 day
    kernel_regularizer=keras.regularizers.L2(1e-4),
    theta_regularizer=keras.regularizers.L1(1e-5),
    kernel_initializer='he_normal',
    normalize_basis=True  # Improve numerical stability
)
```

Performance Tips
================

1. **Stack Order**: Typically trend → seasonality → generic for best results
2. **Block Depth**: 3-4 blocks per stack is usually sufficient
3. **Theta Dimension**:
   - Trend: 3-8 (polynomial degree)
   - Seasonal: 10-20 (number of harmonics × 2)
   - Generic: 16-64 (model capacity)
4. **Regularization**: Use L2 on kernels, L1 on theta for sparsity
5. **Normalization**: Enable for numerical stability with high-degree polynomials

Implementation Details
======================

This Keras 3.x implementation includes:

- **Proper Serialization**: Full save/load support with get_config()
- **Backend Agnostic**: Uses keras.ops for TensorFlow/JAX/PyTorch compatibility
- **Input Validation**: Comprehensive error checking and helpful messages
- **Numerical Stability**: Optional basis function normalization
- **Memory Efficient**: Basis matrices stored as non-trainable weights
- **Type Safety**: Full type hints for better IDE support

References
==========

- Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2019).
  "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting."
  International Conference on Learning Representations (ICLR).

- Challu, C., Olivares, K. G., Oreshkin, B. N., Ramirez, F. G., Canseco, M. M., & Dubrawski, A. (2022).
  "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting."
  AAAI Conference on Artificial Intelligence.

See Also
=========

- dl_techniques.models.nbeats: Complete N-BEATS model implementation
- dl_techniques.models.nbeats_probabilistic: Probabilistic variant with uncertainty
- dl_techniques.utils.datasets.nbeats: Data utilities for N-BEATS training
"""

import keras
import numpy as np
from keras import ops
from typing import Optional, Any, Tuple, Union, List

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

        Raises:
            ValueError: If input shape is invalid.
        """
        self._build_input_shape = input_shape

        # Validate input shape
        if len(input_shape) != 2:
            raise ValueError(f"Expected 2D input shape, got {len(input_shape)}D: {input_shape}")

        if input_shape[-1] is not None and input_shape[-1] != self.backcast_length:
            logger.warning(
                f"Input sequence length {input_shape[-1]} does not match "
                f"expected backcast_length {self.backcast_length}"
            )

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

        Raises:
            ValueError: If input shape is invalid.
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
        random_seed: Optional integer, seed for reproducible basis generation.
        **kwargs: Arguments passed to parent NBeatsBlock.
    """

    def __init__(
        self,
        normalize_basis: bool = True,
        random_seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.normalize_basis = normalize_basis
        self.random_seed = random_seed

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

        # Add small deterministic offset for block uniqueness
        if self.random_seed is not None:
            rng = np.random.RandomState(self.random_seed)
            block_offset = rng.uniform(0, 0.01)
        else:
            # Use a deterministic offset based on layer parameters
            offset_seed = hash((self.units, self.thetas_dim, self.backcast_length)) % 2**31
            rng = np.random.RandomState(offset_seed)
            block_offset = rng.uniform(0, 0.01)

        backcast_grid += block_offset
        forecast_grid += block_offset

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

        # Validate basis matrices for NaN/Inf values
        if np.any(np.isnan(backcast_basis)) or np.any(np.isinf(backcast_basis)):
            raise ValueError("Polynomial backcast basis contains NaN or Inf values")
        if np.any(np.isnan(forecast_basis)) or np.any(np.isinf(forecast_basis)):
            raise ValueError("Polynomial forecast basis contains NaN or Inf values")

        # Store as non-trainable weights using proper serializable initialization
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

        # Assign the computed basis values
        self.backcast_basis_matrix.assign(backcast_basis)
        self.forecast_basis_matrix.assign(forecast_basis)

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

    def get_config(self) -> dict:
        """Get layer configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'normalize_basis': self.normalize_basis,
            'random_seed': self.random_seed,
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
        seasonal_period: Optional integer, dominant seasonal period for frequency calculation.
                       If None, uses max(backcast_length, 12).
        **kwargs: Arguments passed to parent NBeatsBlock.
    """

    def __init__(
        self,
        normalize_basis: bool = True,
        seasonal_period: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.normalize_basis = normalize_basis
        self.seasonal_period = seasonal_period

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
        """Create Fourier basis matrices for seasonality modeling.

        This implementation creates Fourier basis functions that:
        1. Maintain temporal continuity between backcast and forecast
        2. Use proper frequency spacing to avoid aliasing
        3. Include both fundamental and harmonic frequencies
        4. Provide orthogonal basis functions for better decomposition
        """
        # Determine the fundamental seasonal period
        if self.seasonal_period is not None:
            base_period = self.seasonal_period
        else:
            # Adaptive period selection based on sequence length
            base_period = max(self.backcast_length // 2, 12)

        # Store for debugging
        self._computed_base_period = base_period

        # Create continuous normalized time grids for better numerical properties
        # Backcast covers [-1, 0] and forecast covers [0, forecast_length/backcast_length]
        backcast_grid = np.linspace(-1.0, 0.0, self.backcast_length, dtype=np.float32)

        # Forecast grid continues seamlessly from backcast
        forecast_ratio = self.forecast_length / self.backcast_length
        forecast_grid = np.linspace(0.0, forecast_ratio, self.forecast_length, dtype=np.float32)

        # Scale time grids by the seasonal period for proper frequency spacing
        backcast_time = backcast_grid * base_period
        forecast_time = forecast_grid * base_period

        # Initialize basis matrices
        backcast_basis = np.zeros((self.thetas_dim, self.backcast_length), dtype=np.float32)
        forecast_basis = np.zeros((self.thetas_dim, self.forecast_length), dtype=np.float32)

        # Reserve first dimension for DC component (constant term)
        if self.thetas_dim >= 1:
            backcast_basis[0, :] = 1.0
            forecast_basis[0, :] = 1.0
            remaining_dims = self.thetas_dim - 1
        else:
            remaining_dims = self.thetas_dim

        # Calculate number of harmonics (each harmonic needs sin and cos)
        num_harmonics = remaining_dims // 2

        # Generate Fourier harmonics with proper frequency spacing
        for harmonic in range(1, num_harmonics + 1):
            # Fundamental frequency and its harmonics
            # Use 2π/period normalization for proper seasonal modeling
            frequency = 2 * np.pi * harmonic / base_period

            # Calculate indices for this harmonic (skip DC component at index 0)
            cos_idx = 1 + (harmonic - 1) * 2
            sin_idx = 1 + (harmonic - 1) * 2 + 1

            # Generate cosine components
            if cos_idx < self.thetas_dim:
                cos_backcast = np.cos(frequency * backcast_time)
                cos_forecast = np.cos(frequency * forecast_time)

                # Apply normalization for numerical stability
                if self.normalize_basis:
                    cos_backcast = self._normalize_basis_function(cos_backcast)
                    cos_forecast = self._normalize_basis_function(cos_forecast)

                backcast_basis[cos_idx] = cos_backcast
                forecast_basis[cos_idx] = cos_forecast

            # Generate sine components
            if sin_idx < self.thetas_dim:
                sin_backcast = np.sin(frequency * backcast_time)
                sin_forecast = np.sin(frequency * forecast_time)

                # Apply normalization for numerical stability
                if self.normalize_basis:
                    sin_backcast = self._normalize_basis_function(sin_backcast)
                    sin_forecast = self._normalize_basis_function(sin_forecast)

                backcast_basis[sin_idx] = sin_backcast
                forecast_basis[sin_idx] = sin_forecast

        # Handle case where thetas_dim is not perfectly divisible
        # Add additional low-frequency components if needed
        remaining_indices = remaining_dims - (num_harmonics * 2)
        if remaining_indices > 0:
            # Add fractional harmonics for finer frequency resolution
            for i in range(remaining_indices):
                idx = 1 + num_harmonics * 2 + i
                if idx < self.thetas_dim:
                    # Use fractional harmonics (0.5, 1.5, 2.5, etc.)
                    fractional_harmonic = 0.5 + i * 0.5
                    frequency = 2 * np.pi * fractional_harmonic / base_period

                    # Alternate between cosine and sine for fractional harmonics
                    if i % 2 == 0:
                        func_backcast = np.cos(frequency * backcast_time)
                        func_forecast = np.cos(frequency * forecast_time)
                    else:
                        func_backcast = np.sin(frequency * backcast_time)
                        func_forecast = np.sin(frequency * forecast_time)

                    if self.normalize_basis:
                        func_backcast = self._normalize_basis_function(func_backcast)
                        func_forecast = self._normalize_basis_function(func_forecast)

                    backcast_basis[idx] = func_backcast
                    forecast_basis[idx] = func_forecast

        # Validate basis matrices for numerical issues
        self._validate_basis_matrices(backcast_basis, forecast_basis)

        # Store the generated basis matrices as non-trainable weights
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

        # Assign the computed basis values
        self.backcast_basis_matrix.assign(backcast_basis)
        self.forecast_basis_matrix.assign(forecast_basis)

    def _normalize_basis_function(self, basis_func: np.ndarray) -> np.ndarray:
        """Normalize a basis function for numerical stability.

        Args:
            basis_func: Basis function array to normalize.

        Returns:
            Normalized basis function.
        """
        norm = np.linalg.norm(basis_func)
        if norm > 1e-8:
            return basis_func / norm
        else:
            logger.warning("Basis function has very small norm, skipping normalization")
            return basis_func

    def _validate_basis_matrices(self, backcast_basis: np.ndarray, forecast_basis: np.ndarray) -> None:
        """Validate basis matrices for numerical issues.

        Args:
            backcast_basis: Backcast basis matrix.
            forecast_basis: Forecast basis matrix.

        Raises:
            ValueError: If basis matrices contain invalid values.
        """
        # Check for NaN/Inf values
        if np.any(np.isnan(backcast_basis)) or np.any(np.isinf(backcast_basis)):
            raise ValueError("Fourier backcast basis contains NaN or Inf values")
        if np.any(np.isnan(forecast_basis)) or np.any(np.isinf(forecast_basis)):
            raise ValueError("Fourier forecast basis contains NaN or Inf values")

        # Check for proper range (Fourier functions should be in [-1, 1])
        if np.any(np.abs(backcast_basis) > 2.0):  # Allow some tolerance for normalization
            logger.warning("Some backcast basis values are outside expected range [-2, 2]")
        if np.any(np.abs(forecast_basis) > 2.0):
            logger.warning("Some forecast basis values are outside expected range [-2, 2]")

        # Check that DC component (if present) is non-zero
        if self.thetas_dim >= 1:
            if np.all(backcast_basis[0] == 0) or np.all(forecast_basis[0] == 0):
                logger.warning("DC component appears to be zero, check basis generation")

        # Log basis characteristics for debugging
        logger.debug(f"Fourier basis created: thetas_dim={self.thetas_dim}, "
                    f"base_period={self._computed_base_period}, "
                    f"backcast_range=[{np.min(backcast_basis):.3f}, {np.max(backcast_basis):.3f}], "
                    f"forecast_range=[{np.min(forecast_basis):.3f}, {np.max(forecast_basis):.3f}]")

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

    def get_config(self) -> dict:
        """Get layer configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'normalize_basis': self.normalize_basis,
            'seasonal_period': self.seasonal_period,
        })
        return config

# ---------------------------------------------------------------------