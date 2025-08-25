import keras
import numpy as np
from keras import ops
from abc import abstractmethod
from typing import Optional, Any, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class NBeatsBlock(keras.layers.Layer):
    """
    Enhanced N-BEATS block layer with performance optimizations and modern Keras 3 compliance.

    This is the base class for all N-BEATS blocks implementing the fundamental N-BEATS
    architecture with proper sub-layer management, improved initialization strategies,
    and numerical stability improvements. The block consists of a 4-layer fully connected
    stack followed by specialized basis function generation for backcast and forecast.

    **Intent**: Provide the foundational building block for N-BEATS time series forecasting
    models, ensuring proper serialization, gradient flow, and extensibility for different
    basis function types (generic, trend, seasonality).

    **Architecture**:
    ```
    Input(shape=[batch, backcast_length])
           ↓
    Dense₁(units, activation)
           ↓
    Dense₂(units, activation)
           ↓
    Dense₃(units, activation)
           ↓
    Dense₄(units, activation)
           ↓
    ┌─────────────────┬─────────────────┐
    ↓                 ↓                 ↓
    ThetaBackcast    ThetaForecast
    (thetas_dim)     (thetas_dim)
    ↓                 ↓
    BasisBackcast    BasisForecast    ← (implemented by subclasses)
    ↓                 ↓
    Output([batch, backcast_length], [batch, forecast_length])
    ```

    **Mathematical Operations**:
    1. **Feature Extraction**: x₄ = Dense₄(Dense₃(Dense₂(Dense₁(input))))
    2. **Parameter Generation**: θ_b = ThetaBackcast(x₄), θ_f = ThetaForecast(x₄)
    3. **Basis Expansion**: backcast = Basis_b(θ_b), forecast = Basis_f(θ_f)

    The basis functions are implemented by concrete subclasses (Generic, Trend, Seasonality).

    Args:
        units: Integer, number of hidden units in the fully connected layers.
            Must be positive. Typical values: 256, 512, 1024.
        thetas_dim: Integer, dimensionality of the theta parameters passed to basis functions.
            Must be positive. Should match the complexity of the targeted pattern.
        backcast_length: Integer, length of the input time series (lookback window).
            Must be positive. Recommended: 3-5 times forecast_length.
        forecast_length: Integer, length of the forecast horizon.
            Must be positive. The number of future time steps to predict.
        share_weights: Boolean, whether to share weights across blocks in the same stack.
            Currently stored but not implemented. Defaults to False.
        activation: String or callable, activation function for hidden layers.
            Defaults to 'silu' for better gradient flow than ReLU.
        use_bias: Boolean, whether to add bias to the dense layers. Defaults to True.
        kernel_initializer: String or Initializer, initializer for FC layer weights.
            Defaults to 'he_normal' for better gradient flow with SiLU.
        theta_initializer: String or Initializer, initializer for theta layers.
            Defaults to 'glorot_uniform' for balanced parameter initialization.
        kernel_regularizer: Optional regularizer for FC layer weights.
        theta_regularizer: Optional regularizer for theta layer weights.
        **kwargs: Additional keyword arguments for the Layer parent class.

    Input shape:
        2D tensor with shape: `(batch_size, backcast_length)`.

    Output shape:
        Tuple of 2D tensors:
        - Backcast: `(batch_size, backcast_length)`
        - Forecast: `(batch_size, forecast_length)`

    Attributes:
        dense1, dense2, dense3, dense4: Fully connected feature extraction layers.
        theta_backcast, theta_forecast: Parameter generation layers.

    Example:
        ```python
        # Define concrete subclass (e.g., GenericBlock, TrendBlock, SeasonalityBlock)
        block = GenericBlock(
            units=512,
            thetas_dim=32,
            backcast_length=168,  # 1 week hourly
            forecast_length=24    # 1 day hourly
        )

        inputs = keras.Input(shape=(168,))
        backcast, forecast = block(inputs)
        ```

    Note:
        This is an abstract base class. Use concrete implementations:
        - GenericBlock: Learnable linear basis functions
        - TrendBlock: Polynomial basis functions for trends
        - SeasonalityBlock: Fourier basis functions for seasonality

    Raises:
        ValueError: If any dimension parameter is non-positive.
        ValueError: If input shape is not 2D during forward pass.
    """

    def __init__(
            self,
            units: int,
            thetas_dim: int,
            backcast_length: int,
            forecast_length: int,
            share_weights: bool = False,
            activation: Union[str, callable] = 'silu',
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
            theta_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            theta_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs with enhanced checks
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        if thetas_dim <= 0:
            raise ValueError(f"thetas_dim must be positive, got {thetas_dim}")
        if backcast_length <= 0:
            raise ValueError(f"backcast_length must be positive, got {backcast_length}")
        if forecast_length <= 0:
            raise ValueError(f"forecast_length must be positive, got {forecast_length}")

        # Warn if backcast_length might be too short
        if backcast_length < 2 * forecast_length:
            logger.warning(
                f"backcast_length ({backcast_length}) < 2 * forecast_length ({forecast_length}). "
                f"Consider using backcast_length >= 3-5 * forecast_length for better performance."
            )

        # Store configuration
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_weights = share_weights
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.theta_initializer = keras.initializers.get(theta_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.theta_regularizer = keras.regularizers.get(theta_regularizer)

        # CREATE all sub-layers in __init__ (modern Keras 3 pattern)
        self.dense1 = keras.layers.Dense(
            self.units,
            use_bias=self.use_bias,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='dense1'
        )
        self.dense2 = keras.layers.Dense(
            self.units,
            use_bias=self.use_bias,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='dense2'
        )
        self.dense3 = keras.layers.Dense(
            self.units,
            use_bias=self.use_bias,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='dense3'
        )
        self.dense4 = keras.layers.Dense(
            self.units,
            use_bias=self.use_bias,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='dense4'
        )
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

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer weights and explicitly build all sub-layers."""
        # Validate input shape
        if len(input_shape) != 2:
            raise ValueError(f"Expected 2D input shape, got {len(input_shape)}D: {input_shape}")

        # BUILD all sub-layers explicitly for proper serialization
        self.dense1.build(input_shape)

        # Subsequent layers take the output of the previous dense layer
        dense_output_shape = (input_shape[0], self.units)
        self.dense2.build(dense_output_shape)
        self.dense3.build(dense_output_shape)
        self.dense4.build(dense_output_shape)
        self.theta_backcast.build(dense_output_shape)
        self.theta_forecast.build(dense_output_shape)

        # Call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Forward pass with performance optimizations.

        Args:
            inputs: Input tensor of shape (batch_size, backcast_length).
            training: Boolean indicating training mode.

        Returns:
            Tuple of (backcast, forecast) tensors.
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

        # Generate backcast and forecast using basis functions (implemented by subclasses)
        backcast = self._generate_backcast(theta_b)
        forecast = self._generate_forecast(theta_f)

        return backcast, forecast

    @abstractmethod
    def _generate_backcast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """
        Generate backcast from theta parameters using basis functions.

        Args:
            theta: Theta parameters for backcast generation.

        Returns:
            Backcast tensor.
        """
        pass

    @abstractmethod
    def _generate_forecast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """
        Generate forecast from theta parameters using basis functions.

        Args:
            theta: Theta parameters for forecast generation.

        Returns:
            Forecast tensor.
        """
        pass

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Tuple of (backcast_shape, forecast_shape).
        """
        if len(input_shape) != 2:
            raise ValueError(f"Expected 2D input shape, got {len(input_shape)}D: {input_shape}")

        batch_size = input_shape[0]
        backcast_shape = (batch_size, self.backcast_length)
        forecast_shape = (batch_size, self.forecast_length)
        return backcast_shape, forecast_shape

    def get_config(self) -> dict:
        """
        Get layer configuration for serialization.

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
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'theta_initializer': keras.initializers.serialize(self.theta_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'theta_regularizer': keras.regularizers.serialize(self.theta_regularizer),
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class GenericBlock(NBeatsBlock):
    """
    Generic N-BEATS block with learnable linear transformations for flexible pattern modeling.

    This block uses trainable Dense layers as basis functions, allowing the model to learn
    arbitrary linear transformations from theta parameters to backcast/forecast outputs.
    It provides maximum flexibility by not constraining the basis functions to specific
    mathematical forms (unlike Trend or Seasonality blocks).

    **Intent**: Provide a flexible N-BEATS block that can learn any linear patterns
    without mathematical constraints, suitable for complex time series that don't
    follow clear trend or seasonal patterns.

    **Architecture**:
    ```
    Input → NBeatsBlock (4 Dense + 2 Theta) → [theta_b, theta_f]
                                               ↓         ↓
                                         BasisBackcast  BasisForecast
                                         (Dense Linear) (Dense Linear)
                                               ↓         ↓
                                           backcast   forecast
    ```

    **Basis Function Details**:
    - **Backcast Basis**: Dense(thetas_dim → backcast_length, linear activation)
    - **Forecast Basis**: Dense(thetas_dim → forecast_length, linear activation)
    - **Initialization**: Orthogonal with small gain (0.1) for stable training

    The orthogonal initialization with small gain helps prevent exploding gradients
    while maintaining expressiveness for learning diverse patterns.

    Args:
        basis_initializer: String or Initializer, initializer for basis matrices.
            Defaults to 'glorot_uniform'. Consider 'orthogonal' for better stability.
        basis_regularizer: Optional regularizer for basis matrices.
            Can help prevent overfitting in the learned basis functions.
        **kwargs: Arguments passed to parent NBeatsBlock.

    Input shape:
        2D tensor with shape: `(batch_size, backcast_length)`.

    Output shape:
        Tuple of 2D tensors:
        - Backcast: `(batch_size, backcast_length)`
        - Forecast: `(batch_size, forecast_length)`

    Attributes:
        backcast_basis: Dense layer for backcast basis function.
        forecast_basis: Dense layer for forecast basis function.

    Example:
        ```python
        # Standard generic block
        block = GenericBlock(
            units=512,
            thetas_dim=64,  # Higher for more expressiveness
            backcast_length=168,
            forecast_length=24
        )

        # With regularization for complex datasets
        block = GenericBlock(
            units=256,
            thetas_dim=32,
            backcast_length=96,
            forecast_length=12,
            basis_regularizer=keras.regularizers.L2(0.001)
        )

        inputs = keras.Input(shape=(96,))
        backcast, forecast = block(inputs)
        ```

    Note:
        Generic blocks are most effective when you don't have strong prior knowledge
        about the time series patterns. They can learn complex relationships but may
        require more data and careful regularization to avoid overfitting.
    """

    def __init__(
            self,
            basis_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            basis_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration
        self.basis_initializer = keras.initializers.get(basis_initializer)
        self.basis_regularizer = keras.regularizers.get(basis_regularizer)

        # CREATE sub-layers specific to GenericBlock in __init__
        # Use orthogonal initialization with small gain for stability
        orthogonal_init = keras.initializers.Orthogonal(gain=0.1)

        self.backcast_basis = keras.layers.Dense(
            self.backcast_length,
            activation='linear',
            use_bias=False,
            kernel_initializer=orthogonal_init,
            kernel_regularizer=self.basis_regularizer,
            name='backcast_basis'
        )
        self.forecast_basis = keras.layers.Dense(
            self.forecast_length,
            activation='linear',
            use_bias=False,
            kernel_initializer=orthogonal_init,
            kernel_regularizer=self.basis_regularizer,
            name='forecast_basis'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the generic block and its sub-layers."""
        # BUILD the GenericBlock-specific sub-layers
        # Their input is theta, which has shape (batch_size, thetas_dim)
        theta_shape = (input_shape[0], self.thetas_dim)
        self.backcast_basis.build(theta_shape)
        self.forecast_basis.build(theta_shape)

        # Call parent build method (which builds the Dense stack)
        super().build(input_shape)

    def _generate_backcast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """
        Generate backcast using learnable basis functions.

        Args:
            theta: Theta parameters for backcast generation.

        Returns:
            Backcast tensor.
        """
        return self.backcast_basis(theta)

    def _generate_forecast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """
        Generate forecast using learnable basis functions.

        Args:
            theta: Theta parameters for forecast generation.

        Returns:
            Forecast tensor.
        """
        return self.forecast_basis(theta)

    def get_config(self) -> dict:
        """
        Get layer configuration for serialization.

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
    """
    Trend N-BEATS block with polynomial basis functions for modeling trending behavior.

    This block uses mathematically-defined polynomial basis functions to explicitly
    model trend patterns in time series data. The polynomial basis functions ensure
    smooth continuity between backcast and forecast, making them ideal for capturing
    linear trends, growth patterns, and other monotonic behaviors.

    **Intent**: Provide a specialized N-BEATS block for explicitly modeling trend
    components in time series, using polynomial basis functions that guarantee
    mathematical continuity and smooth extrapolation.

    **Architecture & Mathematical Foundation**:
    ```
    Input → NBeatsBlock (4 Dense + 2 Theta) → [theta_b, theta_f]
                                               ↓         ↓
                                         PolyBasis_b  PolyBasis_f
                                         (degree 0,1,2,...)
                                               ↓         ↓
                                           backcast   forecast

    Polynomial Basis Functions:
    - Degree 0: f₀(t) = 1              (constant/level)
    - Degree 1: f₁(t) = t              (linear trend)
    - Degree 2: f₂(t) = t²             (quadratic trend)
    - ...
    - Degree n-1: fₙ₋₁(t) = t^(n-1)    (higher-order trends)

    Time normalization: t ∈ [-1, 1] centered at backcast-forecast transition
    ```

    **Continuity Guarantee**: The polynomial basis ensures that the trend at the
    end of the backcast period exactly matches the trend at the start of the
    forecast period, providing smooth mathematical continuity.

    Args:
        normalize_basis: Boolean, whether to normalize polynomial basis functions
            for better numerical conditioning. Defaults to True.
        **kwargs: Arguments passed to parent NBeatsBlock.

    Input shape:
        2D tensor with shape: `(batch_size, backcast_length)`.

    Output shape:
        Tuple of 2D tensors:
        - Backcast: `(batch_size, backcast_length)`
        - Forecast: `(batch_size, forecast_length)`

    Attributes:
        backcast_basis_matrix: Non-trainable weight matrix for backcast polynomial basis.
        forecast_basis_matrix: Non-trainable weight matrix for forecast polynomial basis.

    Example:
        ```python
        # Linear trend modeling (degree 1)
        block = TrendBlock(
            units=256,
            thetas_dim=2,  # constant + linear
            backcast_length=96,
            forecast_length=24
        )

        # Complex trend modeling (up to degree 3)
        block = TrendBlock(
            units=512,
            thetas_dim=4,  # constant + linear + quadratic + cubic
            backcast_length=168,
            forecast_length=24,
            normalize_basis=True  # Better numerical stability
        )

        inputs = keras.Input(shape=(96,))
        backcast, forecast = block(inputs)
        ```

    Note:
        - thetas_dim controls polynomial degree: thetas_dim=3 → degree 2 polynomial
        - Higher degrees can model complex trends but may overfit
        - Normalization improves numerical stability for higher-degree polynomials
        - Best for time series with clear trending behavior
    """

    def __init__(
            self,
            normalize_basis: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration
        self.normalize_basis = normalize_basis

        # Weights created in build() - these are not sub-layers
        self.backcast_basis_matrix = None
        self.forecast_basis_matrix = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the trend block with corrected basis functions."""
        if self.thetas_dim < 1:
            raise ValueError(f"thetas_dim must be at least 1 for TrendBlock, got {self.thetas_dim}")

        # Create polynomial basis matrices (these are weights, not sub-layers)
        self._create_polynomial_basis()

        # Call parent's build method to handle Dense layers
        super().build(input_shape)

    def _create_polynomial_basis(self) -> None:
        """Create mathematically correct polynomial basis matrices with continuity."""
        # Create continuous time vector for proper polynomial extrapolation
        total_length = self.backcast_length + self.forecast_length

        # Create continuous time indices
        time_indices = np.arange(total_length, dtype=np.float32)

        # Normalize to [-1, 1] range for better numerical stability
        # This centers the polynomial around the transition point
        time_normalized = 2.0 * (time_indices - self.backcast_length) / total_length

        # Split into backcast and forecast portions
        backcast_time = time_normalized[:self.backcast_length]
        forecast_time = time_normalized[self.backcast_length:]

        # Initialize basis matrices
        backcast_basis = np.zeros((self.thetas_dim, self.backcast_length), dtype=np.float32)
        forecast_basis = np.zeros((self.thetas_dim, self.forecast_length), dtype=np.float32)

        # Generate polynomial terms with improved numerical stability
        for degree in range(self.thetas_dim):
            if degree == 0:
                # Constant term
                backcast_basis[degree] = np.ones_like(backcast_time)
                forecast_basis[degree] = np.ones_like(forecast_time)
            else:
                # Polynomial terms: t^degree
                backcast_basis[degree] = np.power(backcast_time, degree)
                forecast_basis[degree] = np.power(forecast_time, degree)

            # Optional normalization for better conditioning
            if self.normalize_basis and degree > 0:
                # Normalize based on the expected range of values
                scale_factor = np.sqrt(degree + 1)  # Simple scaling based on degree
                backcast_basis[degree] /= scale_factor
                forecast_basis[degree] /= scale_factor

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

        logger.info(f"TrendBlock: Created continuous polynomial basis with degree {self.thetas_dim - 1}")

    def _generate_backcast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """
        Generate backcast using polynomial basis functions.

        Args:
            theta: Theta parameters for backcast generation.

        Returns:
            Backcast tensor.
        """
        return ops.matmul(theta, self.backcast_basis_matrix)

    def _generate_forecast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """
        Generate forecast using polynomial basis functions.

        Args:
            theta: Theta parameters for forecast generation.

        Returns:
            Forecast tensor.
        """
        return ops.matmul(theta, self.forecast_basis_matrix)

    def get_config(self) -> dict:
        """
        Get layer configuration for serialization.

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
    """
    Seasonality N-BEATS block with corrected Fourier basis functions for periodic patterns.

    This block uses mathematically-defined Fourier (sine/cosine) basis functions to
    explicitly model seasonal and periodic patterns in time series data. The Fourier
    basis functions ensure proper frequency relationships and mathematical continuity
    between backcast and forecast, making them ideal for capturing daily, weekly,
    monthly, and other cyclical behaviors.

    **Intent**: Provide a specialized N-BEATS block for explicitly modeling seasonal
    and periodic components in time series, using Fourier basis functions that
    guarantee mathematical continuity and proper frequency relationships.

    **Architecture & Mathematical Foundation**:
    ```
    Input → NBeatsBlock (4 Dense + 2 Theta) → [theta_b, theta_f]
                                               ↓         ↓
                                         FourierBasis_b  FourierBasis_f
                                         (harmonics 1,2,3,...)
                                               ↓         ↓
                                           backcast   forecast

    Fourier Basis Functions (for harmonic k):
    - Cosine: cos(2πk·t / T)  where T = backcast_length + forecast_length
    - Sine:   sin(2πk·t / T)

    Harmonic Sequence: k = 1, 2, 3, ..., num_harmonics
    ```

    **Frequency Relationship**: The fundamental period T spans the entire sequence
    (backcast + forecast), ensuring that seasonal patterns align correctly between
    the two segments and providing smooth mathematical continuity.

    **Theta Dimension Mapping**:
    - thetas_dim = 2n: Creates n harmonics (n cos/sin pairs)
    - thetas_dim = 2n+1: Creates n harmonics plus DC component

    Args:
        normalize_basis: Boolean, whether to normalize Fourier basis functions
            based on their full-sequence energy for better numerical stability.
            Defaults to True.
        **kwargs: Arguments passed to parent NBeatsBlock.

    Input shape:
        2D tensor with shape: `(batch_size, backcast_length)`.

    Output shape:
        Tuple of 2D tensors:
        - Backcast: `(batch_size, backcast_length)`
        - Forecast: `(batch_size, forecast_length)`

    Attributes:
        backcast_basis_matrix: Non-trainable weight matrix for backcast Fourier basis.
        forecast_basis_matrix: Non-trainable weight matrix for forecast Fourier basis.

    Example:
        ```python
        # Daily seasonality (12 harmonics for detailed patterns)
        block = SeasonalityBlock(
            units=256,
            thetas_dim=24,  # 12 harmonics (cos/sin pairs)
            backcast_length=168,  # 1 week hourly
            forecast_length=24    # 1 day hourly
        )

        # Simple seasonality (4 harmonics for basic patterns)
        block = SeasonalityBlock(
            units=128,
            thetas_dim=8,   # 4 harmonics
            backcast_length=96,
            forecast_length=12,
            normalize_basis=True  # Better numerical stability
        )

        inputs = keras.Input(shape=(96,))
        backcast, forecast = block(inputs)
        ```

    Note:
        - More harmonics (higher thetas_dim) capture finer seasonal details
        - Normalization improves numerical stability and continuity
        - Best for time series with clear periodic/seasonal patterns
        - Consider data frequency when choosing backcast/forecast lengths
    """

    def __init__(
            self,
            normalize_basis: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration
        self.normalize_basis = normalize_basis

        # Weights created in build() - these are not sub-layers
        self.backcast_basis_matrix = None
        self.forecast_basis_matrix = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the seasonality block with corrected basis functions."""
        if self.thetas_dim < 2:
            logger.warning(f"thetas_dim ({self.thetas_dim}) < 2 for SeasonalityBlock. Consider using even numbers.")

        # Create Fourier basis matrices (these are weights, not sub-layers)
        self._create_fourier_basis()

        # Call parent's build method to handle Dense layers
        super().build(input_shape)

    def _create_fourier_basis(self) -> None:
        """Create mathematically correct Fourier basis matrices with continuity."""
        # Number of harmonics (sin/cos pairs)
        num_harmonics = self.thetas_dim // 2

        # Create continuous time indices for proper frequency relationship
        backcast_indices = np.arange(self.backcast_length, dtype=np.float32)
        forecast_indices = np.arange(
            self.backcast_length,
            self.backcast_length + self.forecast_length,
            dtype=np.float32
        )

        # Initialize basis matrices
        backcast_basis = np.zeros((self.thetas_dim, self.backcast_length), dtype=np.float32)
        forecast_basis = np.zeros((self.thetas_dim, self.forecast_length), dtype=np.float32)

        # Use total sequence length for period calculation
        total_length = self.backcast_length + self.forecast_length

        # Generate Fourier terms with correct frequencies
        basis_idx = 0

        for harmonic in range(1, num_harmonics + 1):
            if basis_idx >= self.thetas_dim:
                break

            # Frequency for this harmonic
            frequency = 2.0 * np.pi * harmonic / total_length

            # Cosine component with continuous normalization
            if basis_idx < self.thetas_dim:
                cos_backcast = np.cos(frequency * backcast_indices)
                cos_forecast = np.cos(frequency * forecast_indices)

                # Normalize based on COMBINED sequence for continuity
                if self.normalize_basis:
                    # Create full continuous cosine for norm calculation
                    full_indices = np.arange(total_length, dtype=np.float32)
                    full_cosine = np.cos(frequency * full_indices)
                    full_norm = np.linalg.norm(full_cosine)

                    if full_norm > 1e-8:
                        cos_backcast /= full_norm
                        cos_forecast /= full_norm

                backcast_basis[basis_idx] = cos_backcast
                forecast_basis[basis_idx] = cos_forecast
                basis_idx += 1

            # Sine component with continuous normalization
            if basis_idx < self.thetas_dim:
                sin_backcast = np.sin(frequency * backcast_indices)
                sin_forecast = np.sin(frequency * forecast_indices)

                # Normalize based on COMBINED sequence for continuity
                if self.normalize_basis:
                    # Create full continuous sine for norm calculation
                    full_indices = np.arange(total_length, dtype=np.float32)
                    full_sine = np.sin(frequency * full_indices)
                    full_norm = np.linalg.norm(full_sine)

                    if full_norm > 1e-8:
                        sin_backcast /= full_norm
                        sin_forecast /= full_norm

                backcast_basis[basis_idx] = sin_backcast
                forecast_basis[basis_idx] = sin_forecast
                basis_idx += 1

        # Handle odd theta_dim: add DC component
        if self.thetas_dim % 2 == 1 and basis_idx < self.thetas_dim:
            backcast_basis[basis_idx] = 1.0  # DC component
            forecast_basis[basis_idx] = 1.0
            basis_idx += 1

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

        logger.info(f"SeasonalityBlock: Created continuous Fourier basis with {num_harmonics} harmonics")

    def _generate_backcast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """
        Generate backcast using Fourier basis functions.

        Args:
            theta: Theta parameters for backcast generation.

        Returns:
            Backcast tensor.
        """
        return ops.matmul(theta, self.backcast_basis_matrix)

    def _generate_forecast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """
        Generate forecast using Fourier basis functions.

        Args:
            theta: Theta parameters for forecast generation.

        Returns:
            Forecast tensor.
        """
        return ops.matmul(theta, self.forecast_basis_matrix)

    def get_config(self) -> dict:
        """
        Get layer configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'normalize_basis': self.normalize_basis,
        })
        return config

# ---------------------------------------------------------------------
