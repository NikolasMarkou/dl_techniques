"""
Foundational blocks of the N-BEATS architecture.

The N-BEATS block is the core computational unit of the Neural Basis Expansion
Analysis for Time Series (N-BEATS) model. It performs a functional
decomposition of the input time series by learning coefficients for a set of
basis functions.

The block learns to represent the input time series in a compact, latent
parameter space (theta). These parameters are then used as coefficients to
generate both a forecast for the future and a backcast that reconstructs the
input:
    theta = f_theta(x)
    y = G(theta) = sum_i(theta_i * v_i)

The architecture consists of two main parts:
1.  A deep, fully-connected stack (MLP) that processes the input time series
    and extracts a high-level feature representation.
2.  Two linear projection heads that map this representation to separate
    theta vectors for the backcast and forecast.

Concrete subclasses (GenericBlock, TrendBlock, SeasonalityBlock) define
the specific mathematical form of the basis functions (learnable linear
transforms, polynomials, Fourier series). A key concept is the block's dual
output of backcast and forecast. In the full N-BEATS model, the backcast is
subtracted from the input, and the residual is passed to the next block
("doubly residual stacking"), enabling decomposition of the time series into
successive components.

References:
    - Oreshkin et al. (2020). N-BEATS: Neural Basis Expansion Analysis for
      interpretable Time Series forecasting. In ICLR.
      https://arxiv.org/abs/1905.10437
"""

import keras
import numpy as np
from keras import ops
from abc import abstractmethod
from typing import Optional, Any, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms.rms_norm import RMSNorm

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class NBeatsBlock(keras.layers.Layer):
    """
    Base N-BEATS block with a 4-layer dense stack and dual theta projection.

    This abstract base class implements the fundamental N-BEATS architecture:
    a 4-layer fully connected stack followed by two linear heads that produce
    theta coefficients for backcast and forecast generation. Concrete subclasses
    (GenericBlock, TrendBlock, SeasonalityBlock) define the basis functions that
    expand theta into time-domain signals.

    The block approximates a function that maps an input time series x of
    length H to a forecast y of length T in two stages:
        theta = f_theta(x)        (MLP learns coefficients)
        y = G(theta)              (basis expansion generates output)

    For multivariate inputs, theta coefficients are generated per-feature.

    **Architecture Overview:**

    .. code-block:: text

        Input: (batch, backcast_length * input_dim)
                        |
                        v
               +------------------+
               | Dense1 -> [Norm] |
               | -> [Dropout]     |
               +--------+---------+
                        |
                        v
               +------------------+
               | Dense2 -> [Norm] |
               | -> [Dropout]     |
               +--------+---------+
                        |
                        v
               +------------------+
               | Dense3 -> [Norm] |
               | -> [Dropout]     |
               +--------+---------+
                        |
                        v
               +------------------+
               | Dense4 -> [Norm] |
               | -> [Dropout]     |
               +--------+---------+
                        |
              +---------+---------+
              |                   |
              v                   v
        +------------+     +------------+
        | Theta_back |     | Theta_fore |
        +------+-----+     +------+-----+
               |                   |
               v                   v
        _generate_backcast   _generate_forecast
        (subclass basis)     (subclass basis)
               |                   |
               v                   v
        Backcast output     Forecast output

    :param units: Number of hidden units in the fully connected layers.
        Must be positive.
    :type units: int
    :param thetas_dim: Dimensionality of the theta parameters passed to
        basis functions. Must be positive.
    :type thetas_dim: int
    :param backcast_length: Length of the input time series (lookback window).
        Must be positive.
    :type backcast_length: int
    :param forecast_length: Length of the forecast horizon. Must be positive.
    :type forecast_length: int
    :param input_dim: Number of input features (channels).
    :type input_dim: int
    :param output_dim: Number of output features (channels).
    :type output_dim: int
    :param share_weights: Whether to share weights across blocks in the same
        stack. Currently stored but not implemented.
    :type share_weights: bool
    :param dropout_rate: Dropout rate (0 to 1) applied after each dense layer.
    :type dropout_rate: float
    :param activation: Activation function for hidden layers.
    :type activation: str or callable
    :param use_bias: Whether to add bias to the dense layers.
    :type use_bias: bool
    :param use_normalization: Whether to apply RMSNorm after each dense layer.
    :type use_normalization: bool
    :param kernel_initializer: Initializer for FC layer weights.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param theta_initializer: Initializer for theta layers.
    :type theta_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for FC layer weights.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param theta_regularizer: Optional regularizer for theta layer weights.
    :type theta_regularizer: keras.regularizers.Regularizer or None
    :param kwargs: Additional keyword arguments for the Layer parent class.

    :raises ValueError: If units, thetas_dim, backcast_length, forecast_length,
        input_dim, or output_dim are not positive, or if dropout_rate is out
        of range [0, 1).
    """

    def __init__(
            self,
            units: int,
            thetas_dim: int,
            backcast_length: int,
            forecast_length: int,
            input_dim: int = 1,
            output_dim: int = 1,
            share_weights: bool = False,
            dropout_rate: float = 0.0,
            activation: Union[str, callable] = 'relu',
            use_bias: bool = False,
            use_normalization: bool = False,
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
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

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
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.share_weights = share_weights
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_bias = use_bias
        self.use_normalization = use_normalization
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.theta_initializer = keras.initializers.get(theta_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.theta_regularizer = keras.regularizers.get(theta_regularizer)

        # Conditionally create normalization layers
        if self.use_normalization:
            self.norm1 = RMSNorm(axis=-1, use_scale=False)
            self.norm2 = RMSNorm(axis=-1, use_scale=False)
            self.norm3 = RMSNorm(axis=-1, use_scale=False)
            self.norm4 = RMSNorm(axis=-1, use_scale=False)
        else:
            self.norm1 = self.norm2 = self.norm3 = self.norm4 = None

        # Conditionally create dropout layers
        if self.dropout_rate > 0:
            self.dropout1 = keras.layers.Dropout(self.dropout_rate)
            self.dropout2 = keras.layers.Dropout(self.dropout_rate)
            self.dropout3 = keras.layers.Dropout(self.dropout_rate)
            self.dropout4 = keras.layers.Dropout(self.dropout_rate)
        else:
            self.dropout1 = self.dropout2 = self.dropout3 = self.dropout4 = None

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

        # Theta projection layers
        # For multivariate, we generate unique thetas for each feature
        # Shape: thetas_dim * input_dim (or output_dim)
        self.theta_backcast = keras.layers.Dense(
            self.thetas_dim * self.input_dim,
            activation='linear',
            use_bias=False,
            kernel_initializer=self.theta_initializer,
            kernel_regularizer=self.theta_regularizer,
            name='theta_backcast'
        )
        self.theta_forecast = keras.layers.Dense(
            self.thetas_dim * self.output_dim,
            activation='linear',
            use_bias=False,
            kernel_initializer=self.theta_initializer,
            kernel_regularizer=self.theta_regularizer,
            name='theta_forecast'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer weights and explicitly build all sub-layers.

        :param input_shape: Shape of the input tensor, expected 2D.
        :type input_shape: tuple

        :raises ValueError: If input shape is not 2D.
        """
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
        Forward pass through the dense stack and theta projection.

        :param inputs: Input tensor of shape (batch_size, backcast_length * input_dim).
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: bool or None
        :return: Tuple of (backcast, forecast) tensors.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]

        :raises ValueError: If the input is not 2D.
        """
        # Validate input shape at runtime
        input_shape = ops.shape(inputs)
        if len(input_shape) != 2:
            raise ValueError(f"Expected 2D input, got shape: {input_shape}")

        # Pass through four fully connected layers
        # Stack 1
        x = self.dense1(inputs, training=training)
        if self.use_normalization:
            x = self.norm1(x)
        if self.dropout_rate > 0:
            x = self.dropout1(x, training=training)

        # Stack 2
        x = self.dense2(x, training=training)
        if self.use_normalization:
            x = self.norm2(x)
        if self.dropout_rate > 0:
            x = self.dropout2(x, training=training)

        # Stack 3
        x = self.dense3(x, training=training)
        if self.use_normalization:
            x = self.norm3(x)
        if self.dropout_rate > 0:
            x = self.dropout3(x, training=training)

        # Stack 4
        x = self.dense4(x, training=training)
        if self.use_normalization:
            x = self.norm4(x)
        if self.dropout_rate > 0:
            x = self.dropout4(x, training=training)

        # Generate theta parameters
        # Shape: (batch_size, thetas_dim * dim)
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

        :param theta: Theta parameters for backcast generation.
        :type theta: keras.KerasTensor
        :return: Backcast tensor of shape (batch, backcast_length * input_dim).
        :rtype: keras.KerasTensor
        """
        pass

    @abstractmethod
    def _generate_forecast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """
        Generate forecast from theta parameters using basis functions.

        :param theta: Theta parameters for forecast generation.
        :type theta: keras.KerasTensor
        :return: Forecast tensor of shape (batch, forecast_length * output_dim).
        :rtype: keras.KerasTensor
        """
        pass

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple
        :return: Tuple of (backcast_shape, forecast_shape).
        :rtype: tuple[tuple, tuple]

        :raises ValueError: If input shape is not 2D.
        """
        if len(input_shape) != 2:
            raise ValueError(f"Expected 2D input shape, got {len(input_shape)}D: {input_shape}")

        batch_size = input_shape[0]
        backcast_shape = (batch_size, self.backcast_length * self.input_dim)
        forecast_shape = (batch_size, self.forecast_length * self.output_dim)
        return backcast_shape, forecast_shape

    def get_config(self) -> dict:
        """
        Return configuration dictionary for serialization.

        :return: Configuration dictionary.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            'units': self.units,
            'thetas_dim': self.thetas_dim,
            'backcast_length': self.backcast_length,
            'forecast_length': self.forecast_length,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'share_weights': self.share_weights,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'use_normalization': self.use_normalization,
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
    Generic N-BEATS block with learnable linear basis functions.

    This block uses trainable Dense layers as basis functions, allowing the model
    to learn arbitrary linear transformations from theta parameters to
    backcast/forecast outputs. It provides maximum flexibility by not constraining
    the basis functions to specific mathematical forms (unlike TrendBlock or
    SeasonalityBlock).

    The backcast basis is Dense(thetas_dim * input_dim -> backcast_len * input_dim)
    and the forecast basis is Dense(thetas_dim * output_dim -> forecast_len * output_dim),
    both with orthogonal initialization (gain=0.1) for stable training.

    **Architecture Overview:**

    .. code-block:: text

        Input: (batch, backcast_length * input_dim)
                        |
                        v
               +------------------+
               |  NBeatsBlock     |
               |  (4 Dense +      |
               |   2 Theta heads) |
               +--------+---------+
                        |
              +---------+---------+
              |                   |
              v                   v
        theta_backcast      theta_forecast
              |                   |
              v                   v
        +------------+     +------------+
        | Dense      |     | Dense      |
        | (linear)   |     | (linear)   |
        +------+-----+     +------+-----+
               |                   |
               v                   v
        Backcast             Forecast
        (batch, B*in_dim)    (batch, F*out_dim)

    :param basis_initializer: Initializer for basis matrices.
    :type basis_initializer: str or keras.initializers.Initializer
    :param basis_regularizer: Optional regularizer for basis matrices.
    :type basis_regularizer: keras.regularizers.Regularizer or None
    :param kwargs: Arguments passed to parent NBeatsBlock.
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

        # For Generic Block, the projection is just a large matrix multiplication
        # We project from flattened theta to flattened time series directly.
        self.backcast_basis = keras.layers.Dense(
            self.backcast_length * self.input_dim,
            activation='linear',
            use_bias=False,
            kernel_initializer=orthogonal_init,
            kernel_regularizer=self.basis_regularizer,
            name='backcast_basis'
        )
        self.forecast_basis = keras.layers.Dense(
            self.forecast_length * self.output_dim,
            activation='linear',
            use_bias=False,
            kernel_initializer=orthogonal_init,
            kernel_regularizer=self.basis_regularizer,
            name='forecast_basis'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the generic block and its basis sub-layers.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple
        """
        # BUILD the GenericBlock-specific sub-layers
        # Their input is theta, which has shape (batch_size, thetas_dim * dim)
        theta_backcast_shape = (input_shape[0], self.thetas_dim * self.input_dim)
        theta_forecast_shape = (input_shape[0], self.thetas_dim * self.output_dim)

        self.backcast_basis.build(theta_backcast_shape)
        self.forecast_basis.build(theta_forecast_shape)

        # Call parent build method (which builds the Dense stack)
        super().build(input_shape)

    def _generate_backcast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """
        Generate backcast using learnable linear basis functions.

        :param theta: Theta parameters of shape (batch, thetas_dim * input_dim).
        :type theta: keras.KerasTensor
        :return: Backcast tensor of shape (batch, backcast_length * input_dim).
        :rtype: keras.KerasTensor
        """
        return self.backcast_basis(theta)

    def _generate_forecast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """
        Generate forecast using learnable linear basis functions.

        :param theta: Theta parameters of shape (batch, thetas_dim * output_dim).
        :type theta: keras.KerasTensor
        :return: Forecast tensor of shape (batch, forecast_length * output_dim).
        :rtype: keras.KerasTensor
        """
        return self.forecast_basis(theta)

    def get_config(self) -> dict:
        """
        Return configuration dictionary for serialization.

        :return: Configuration dictionary.
        :rtype: dict
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
    Trend N-BEATS block with polynomial basis functions for modeling trends.

    This block uses mathematically-defined polynomial basis functions to explicitly
    model trend patterns in time series data. The polynomial degree is determined
    by thetas_dim, and each basis function corresponds to a power of time:
        f_0(t) = 1              (constant/level)
        f_1(t) = t              (linear trend)
        f_2(t) = t^2            (quadratic trend)
        f_{n-1}(t) = t^{n-1}   (higher-order trends)

    Time is normalized to t in [-1, 1] centered at the backcast-forecast
    transition for numerical stability. In multivariate settings, theta is
    reshaped to (Batch, FeatureDim, Degree) and the basis matrix is
    (Degree, Time), producing per-feature trends.

    **Architecture Overview:**

    .. code-block:: text

        Input: (batch, backcast_length * input_dim)
                        |
                        v
               +------------------+
               |  NBeatsBlock     |
               |  (4 Dense +      |
               |   2 Theta heads) |
               +--------+---------+
                        |
              +---------+---------+
              |                   |
              v                   v
        theta_backcast      theta_forecast
              |                   |
              v                   v
        Reshape to              Reshape to
        (B, in_dim, degree)     (B, out_dim, degree)
              |                   |
              v                   v
        matmul with             matmul with
        PolyBasis_backcast      PolyBasis_forecast
        (degree, backcast_len)  (degree, forecast_len)
              |                   |
              v                   v
        Transpose + Flatten     Transpose + Flatten
              |                   |
              v                   v
        Backcast                Forecast

    :param normalize_basis: Whether to normalize polynomial basis functions
        for better numerical conditioning.
    :type normalize_basis: bool
    :param kwargs: Arguments passed to parent NBeatsBlock.
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
        """
        Build the trend block with polynomial basis matrices.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple

        :raises ValueError: If thetas_dim is less than 1.
        """
        if self.thetas_dim < 1:
            raise ValueError(f"thetas_dim must be at least 1 for TrendBlock, got {self.thetas_dim}")

        # Create polynomial basis matrices (these are weights, not sub-layers)
        self._create_polynomial_basis()

        # Call parent's build method to handle Dense layers
        super().build(input_shape)

    def _create_polynomial_basis(self) -> None:
        """
        Create polynomial basis matrices with continuous time normalization.

        Generates non-trainable weight matrices where each row corresponds to
        a polynomial degree evaluated over the normalized time range [-1, 1].
        """
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
        # Shape: (thetas_dim, time)
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

    def _generate_backcast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """
        Generate backcast using polynomial basis functions.

        Reshapes theta to (Batch, InputDim, PolyDegree), multiplies by the
        polynomial basis matrix (PolyDegree, Time), transposes to
        (Batch, Time, InputDim), and flattens.

        :param theta: Theta parameters of shape (batch, thetas_dim * input_dim).
        :type theta: keras.KerasTensor
        :return: Backcast tensor of shape (batch, backcast_length * input_dim).
        :rtype: keras.KerasTensor
        """
        # 1. Reshape to separate features and polynomial degrees
        theta_reshaped = ops.reshape(theta, (-1, self.input_dim, self.thetas_dim))

        # 2. Apply basis function (broadcasts over batch and input_dim)
        result = ops.matmul(theta_reshaped, self.backcast_basis_matrix)

        # 3. Transpose to (Batch, Time, InputDim) to match flattened order
        result = ops.transpose(result, (0, 2, 1))

        # 4. Flatten back to residual stream format
        return ops.reshape(result, (-1, self.backcast_length * self.input_dim))

    def _generate_forecast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """
        Generate forecast using polynomial basis functions.

        :param theta: Theta parameters of shape (batch, thetas_dim * output_dim).
        :type theta: keras.KerasTensor
        :return: Forecast tensor of shape (batch, forecast_length * output_dim).
        :rtype: keras.KerasTensor
        """
        # 1. Reshape to separate features and polynomial degrees
        theta_reshaped = ops.reshape(theta, (-1, self.output_dim, self.thetas_dim))

        # 2. Apply basis function
        result = ops.matmul(theta_reshaped, self.forecast_basis_matrix)

        # 3. Transpose to (Batch, Time, OutputDim)
        result = ops.transpose(result, (0, 2, 1))

        # 4. Flatten
        return ops.reshape(result, (-1, self.forecast_length * self.output_dim))

    def get_config(self) -> dict:
        """
        Return configuration dictionary for serialization.

        :return: Configuration dictionary.
        :rtype: dict
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
    Seasonality N-BEATS block with Fourier basis functions for periodic patterns.

    This block uses mathematically-defined Fourier (sine/cosine) basis functions to
    explicitly model seasonal and periodic patterns. The number of harmonics is
    thetas_dim // 2, and each harmonic k generates a cosine and sine pair:
        cos(2 * pi * k * t / T)
        sin(2 * pi * k * t / T)

    If thetas_dim is odd, a DC component (constant 1) is appended. Basis functions
    are normalized based on the full (backcast + forecast) sequence energy for
    continuity. In multivariate settings, theta is reshaped to
    (Batch, FeatureDim, Harmonics) before applying the basis matrix.

    **Architecture Overview:**

    .. code-block:: text

        Input: (batch, backcast_length * input_dim)
                        |
                        v
               +------------------+
               |  NBeatsBlock     |
               |  (4 Dense +      |
               |   2 Theta heads) |
               +--------+---------+
                        |
              +---------+---------+
              |                   |
              v                   v
        theta_backcast      theta_forecast
              |                   |
              v                   v
        Reshape to              Reshape to
        (B, in_dim, harmonics)  (B, out_dim, harmonics)
              |                   |
              v                   v
        matmul with             matmul with
        FourierBasis_backcast   FourierBasis_forecast
        (harmonics, B_len)      (harmonics, F_len)
              |                   |
              v                   v
        Transpose + Flatten     Transpose + Flatten
              |                   |
              v                   v
        Backcast                Forecast

    :param normalize_basis: Whether to normalize Fourier basis functions based
        on their full-sequence energy for better numerical stability.
    :type normalize_basis: bool
    :param kwargs: Arguments passed to parent NBeatsBlock.
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
        """
        Build the seasonality block with Fourier basis matrices.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple
        """
        if self.thetas_dim < 2:
            logger.warning(f"thetas_dim ({self.thetas_dim}) < 2 for SeasonalityBlock. Consider using even numbers.")

        # Create Fourier basis matrices (these are weights, not sub-layers)
        self._create_fourier_basis()

        # Call parent's build method to handle Dense layers
        super().build(input_shape)

    def _create_fourier_basis(self) -> None:
        """
        Create Fourier basis matrices with continuous time indices.

        Generates non-trainable weight matrices containing cosine and sine
        pairs for each harmonic, normalized by full-sequence energy when
        normalize_basis is True.
        """
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

    def _generate_backcast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """
        Generate backcast using Fourier basis functions.

        Reshapes theta to (Batch, InputDim, Harmonics), multiplies by the
        Fourier basis matrix (Harmonics, Time), transposes to
        (Batch, Time, InputDim), and flattens.

        :param theta: Theta parameters of shape (batch, thetas_dim * input_dim).
        :type theta: keras.KerasTensor
        :return: Backcast tensor of shape (batch, backcast_length * input_dim).
        :rtype: keras.KerasTensor
        """
        # 1. Reshape
        theta_reshaped = ops.reshape(theta, (-1, self.input_dim, self.thetas_dim))

        # 2. Apply basis function
        result = ops.matmul(theta_reshaped, self.backcast_basis_matrix)

        # 3. Transpose
        result = ops.transpose(result, (0, 2, 1))

        # 4. Flatten
        return ops.reshape(result, (-1, self.backcast_length * self.input_dim))

    def _generate_forecast(self, theta: keras.KerasTensor) -> keras.KerasTensor:
        """
        Generate forecast using Fourier basis functions.

        :param theta: Theta parameters of shape (batch, thetas_dim * output_dim).
        :type theta: keras.KerasTensor
        :return: Forecast tensor of shape (batch, forecast_length * output_dim).
        :rtype: keras.KerasTensor
        """
        # 1. Reshape
        theta_reshaped = ops.reshape(theta, (-1, self.output_dim, self.thetas_dim))

        # 2. Apply basis function
        result = ops.matmul(theta_reshaped, self.forecast_basis_matrix)

        # 3. Transpose
        result = ops.transpose(result, (0, 2, 1))

        # 4. Flatten
        return ops.reshape(result, (-1, self.forecast_length * self.output_dim))

    def get_config(self) -> dict:
        """
        Return configuration dictionary for serialization.

        :return: Configuration dictionary.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            'normalize_basis': self.normalize_basis,
        })
        return config

# ---------------------------------------------------------------------
