"""
Foundational blocks of the N-BEATS architecture.

An N-BEATS block is the core computational unit of the Neural Basis Expansion
Analysis for Time Series (N-BEATS) model. It decomposes an input window by
learning the coefficients of a set of basis functions.

The block first maps the input window to a small latent vector, theta. Theta is
then used as the coefficient vector of a basis expansion, which produces two
outputs: a forecast for the future window, and a backcast that reconstructs the
input window.

    theta = f_theta(x)
    y = G(theta) = sum_i(theta_i * v_i)

Every block has two parts:

1.  A 4-layer fully connected stack that reads the input window and produces a
    hidden representation.
2.  Two linear heads that project that representation to a backcast theta and a
    forecast theta.

The subclasses choose the basis. GenericBlock learns it as a Dense matrix.
TrendBlock uses polynomials. SeasonalityBlock uses a Fourier series.

Both outputs are used. In the full N-BEATS model the backcast is subtracted from
the input and the residual goes to the next block. This is "doubly residual
stacking", and it is what lets a stack decompose a series into successive
components.

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
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.time_series.nbeats_blocks")
class NBeatsBlock(keras.layers.Layer):
    """
    Base N-BEATS block with a 4-layer dense stack and dual theta projection.

    This abstract base class holds the part of N-BEATS that every block shares:
    four fully connected layers, then two linear heads that emit the theta
    coefficients. Subclasses supply the basis that turns theta back into a time
    series. GenericBlock learns the basis, TrendBlock uses polynomials, and
    SeasonalityBlock uses a Fourier series.

    The block maps an input window x of length ``backcast_length`` to a forecast
    of length ``forecast_length`` in two stages:

        theta = f_theta(x)        (the dense stack learns the coefficients)
        y = G(theta)              (the basis expands them back to time)

    Input and output are flat 2D tensors. A multivariate window is laid out as
    ``backcast_length * input_dim`` values per row, and theta is wide enough to
    carry one coefficient set per feature.

    **Architecture Overview:**

    .. code-block:: text

        Input: (batch, backcast_length * input_dim)
                          │
                          ▼
             ┌────────────────────────┐
             │ Dense1 (units)         │
             │ RMSNorm, Dropout (opt) │
             └───────────┬────────────┘
                         ▼
             ┌────────────────────────┐
             │ Dense2 (units)         │
             │ RMSNorm, Dropout (opt) │
             └───────────┬────────────┘
                         ▼
             ┌────────────────────────┐
             │ Dense3 (units)         │
             │ RMSNorm, Dropout (opt) │
             └───────────┬────────────┘
                         ▼
             ┌────────────────────────┐
             │ Dense4 (units)         │
             │ RMSNorm, Dropout (opt) │
             └───────────┬────────────┘
                         │ (batch, units)
             ┌───────────┴────────────┐
             ▼                        ▼
      ┌──────────────┐         ┌──────────────┐
      │theta_backcast│         │theta_forecast│
      │Dense, linear │         │Dense, linear │
      └──────┬───────┘         └──────┬───────┘
             │ (B, td*in_dim)         │ (B, td*out_dim)
             ▼                        ▼
      ┌──────────────┐         ┌──────────────┐
      │_generate_    │         │_generate_    │
      │  backcast    │         │  forecast    │
      │ (subclass)   │         │ (subclass)   │
      └──────┬───────┘         └──────┬───────┘
             ▼                        ▼
        backcast                 forecast
        (B, B_len*in_dim)        (B, F_len*out_dim)

    Both branches are always returned, as a tuple. RMSNorm runs only when
    ``use_normalization`` is True and Dropout only when ``dropout_rate > 0``.
    Here td = thetas_dim, B_len = backcast_length, F_len = forecast_length.

    :param units: Number of hidden units in the fully connected layers.
        Must be positive.
    :type units: int
    :param thetas_dim: Number of theta coefficients per feature. The basis
        functions expand exactly this many coefficients. Must be positive.
    :type thetas_dim: int
    :param backcast_length: Length of the input time series (lookback window).
        Must be positive.
    :type backcast_length: int
    :param forecast_length: Length of the forecast horizon. Must be positive.
    :type forecast_length: int
    :param input_dim: Number of input features (channels). Must be positive.
    :type input_dim: int
    :param output_dim: Number of output features (channels). Must be positive.
    :type output_dim: int
    :param dropout_rate: Dropout rate applied after each dense layer. Must be in
        [0, 1). At 0 no Dropout layer is created.
    :type dropout_rate: float
    :param activation: Activation for the four dense layers.
    :type activation: str or callable
    :param use_bias: Whether the four dense layers carry a bias.
    :type use_bias: bool
    :param use_normalization: Whether to apply RMSNorm after each dense layer.
        At False no RMSNorm layer is created.
    :type use_normalization: bool
    :param kernel_initializer: Initializer for the four dense layers.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param theta_initializer: Initializer for the two theta heads.
    :type theta_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for the four dense layers.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param theta_regularizer: Optional regularizer for the two theta heads.
    :type theta_regularizer: keras.regularizers.Regularizer or None
    :param kwargs: Additional keyword arguments for the Layer parent class.

    :raises ValueError: If units, thetas_dim, backcast_length, forecast_length,
        input_dim, or output_dim are not positive, or if dropout_rate is out
        of range [0, 1).

    Input shape:
        2D tensor of shape ``(batch, backcast_length * input_dim)``.

    Output shape:
        A tuple of two 2D tensors: ``(batch, backcast_length * input_dim)`` and
        ``(batch, forecast_length * output_dim)``.
    """

    def __init__(
            self,
            units: int,
            thetas_dim: int,
            backcast_length: int,
            forecast_length: int,
            input_dim: int = 1,
            output_dim: int = 1,
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
        """
        Validate the configuration and create the dense stack and theta heads.

        See the class docstring for the full parameter list.

        :raises ValueError: If units, thetas_dim, backcast_length,
            forecast_length, input_dim or output_dim is not positive, or if
            dropout_rate is outside [0, 1).
        """
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
        self.dropout_rate = dropout_rate
        self.activation = deserialize_activation(activation)
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
            'dropout_rate': self.dropout_rate,
            'activation': serialize_activation(self.activation),
            'use_bias': self.use_bias,
            'use_normalization': self.use_normalization,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'theta_initializer': keras.initializers.serialize(self.theta_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'theta_regularizer': keras.regularizers.serialize(self.theta_regularizer),
        })
        return config


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.time_series.nbeats_blocks")
class GenericBlock(NBeatsBlock):
    """
    Generic N-BEATS block with learnable linear basis functions.

    This block uses two trainable Dense layers as its basis. The model learns the
    map from theta to the output window instead of being given one. That makes it
    the most flexible of the three blocks, and the least interpretable: unlike
    TrendBlock and SeasonalityBlock, its basis has no fixed mathematical form.

    The backcast basis is a Dense layer from ``thetas_dim * input_dim`` to
    ``backcast_length * input_dim``. The forecast basis is a Dense layer from
    ``thetas_dim * output_dim`` to ``forecast_length * output_dim``. Neither uses
    a bias. Both default to orthogonal initialization at gain 0.1, which keeps the
    block's contribution to the residual stack small at the start of training.

    **Architecture Overview:**

    .. code-block:: text

        Input: (batch, backcast_length * input_dim)
                          │
                          ▼
             ┌─────────────────────────┐
             │ NBeatsBlock base        │
             │ 4 Dense + 2 theta heads │
             └───────────┬─────────────┘
             ┌───────────┴────────────┐
             ▼                        ▼
        theta_backcast           theta_forecast
        (B, td*in_dim)           (B, td*out_dim)
             │                        │
             ▼                        ▼
      ┌──────────────┐         ┌──────────────┐
      │backcast_basis│         │forecast_basis│
      │Dense, linear │         │Dense, linear │
      │no bias       │         │no bias       │
      └──────┬───────┘         └──────┬───────┘
             ▼                        ▼
        backcast                 forecast
        (B, B_len*in_dim)        (B, F_len*out_dim)

    Both basis layers are trainable weights. Both are created from the same
    ``basis_initializer``, so the two paths share a setting but not a weight.

    :param basis_initializer: Initializer for the two basis Dense layers. Pass a
        Keras initializer or its string name. The default ``None`` resolves to
        ``Orthogonal(gain=0.1)``.
    :type basis_initializer: str or keras.initializers.Initializer or None
    :param basis_regularizer: Optional regularizer for the two basis Dense layers.
    :type basis_regularizer: keras.regularizers.Regularizer or None
    :param kwargs: Arguments passed to parent NBeatsBlock.

    Input shape:
        2D tensor of shape ``(batch, backcast_length * input_dim)``.

    Output shape:
        A tuple of ``(batch, backcast_length * input_dim)`` and
        ``(batch, forecast_length * output_dim)``.
    """

    def __init__(
            self,
            basis_initializer: Union[str, keras.initializers.Initializer, None] = None,
            basis_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        """
        Resolve the basis initializer and create the two basis Dense layers.

        See the class docstring for the full parameter list.
        """
        super().__init__(**kwargs)

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-072: the default None resolves
        # to Orthogonal(gain=0.1), and basis_initializer now actually reaches both
        # basis Dense layers instead of being stored, serialized and ignored.
        # Do NOT default it to 'glorot_uniform': that replaces the small-gain
        # orthogonal map every existing GenericBlock was tuned around. See D-072.
        if basis_initializer is None:
            basis_initializer = keras.initializers.Orthogonal(gain=0.1)

        # Store configuration
        self.basis_initializer = keras.initializers.get(basis_initializer)
        self.basis_regularizer = keras.regularizers.get(basis_regularizer)

        # For Generic Block, the projection is just a large matrix multiplication
        # We project from flattened theta to flattened time series directly.
        self.backcast_basis = keras.layers.Dense(
            self.backcast_length * self.input_dim,
            activation='linear',
            use_bias=False,
            kernel_initializer=self.basis_initializer,
            kernel_regularizer=self.basis_regularizer,
            name='backcast_basis'
        )
        self.forecast_basis = keras.layers.Dense(
            self.forecast_length * self.output_dim,
            activation='linear',
            use_bias=False,
            kernel_initializer=self.basis_initializer,
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


@register_dl_technique("dl_techniques.layers.time_series.nbeats_blocks")
class TrendBlock(NBeatsBlock):
    """
    Trend N-BEATS block with polynomial basis functions for modeling trends.

    This block fixes the basis to powers of time, so it can only express a
    polynomial trend. ``thetas_dim`` sets how many powers are available, one per
    row of the basis matrix:

        f_0(t) = 1              (constant, the level)
        f_1(t) = t              (linear trend)
        f_2(t) = t^2            (quadratic trend)
        f_{n-1}(t) = t^{n-1}   (higher-order trends)

    Time is normalized to t in [-1, 1] and centered on the boundary between the
    backcast and forecast windows. That keeps the higher powers small enough to
    stay numerically well behaved and makes extrapolation continuous across the
    boundary. With ``normalize_basis`` on, every row ``d > 0`` is further divided
    by ``sqrt(d + 1)``; the constant row is left alone.

    Both basis matrices are non-trainable. Only theta learns.

    **Architecture Overview:**

    .. code-block:: text

        Input: (batch, backcast_length * input_dim)
                          │
                          ▼
             ┌─────────────────────────┐
             │ NBeatsBlock base        │
             │ 4 Dense + 2 theta heads │
             └───────────┬─────────────┘
             ┌───────────┴────────────┐
             ▼                        ▼
        theta_backcast           theta_forecast
             │                        │
             ▼                        ▼
        reshape                  reshape
        (B, in_dim, td)          (B, out_dim, td)
             │                        │
             ▼                        ▼
      ┌──────────────┐         ┌──────────────┐
      │ matmul with  │         │ matmul with  │
      │ poly basis   │         │ poly basis   │
      │ (td, B_len)  │         │ (td, F_len)  │
      │ fixed weight │         │ fixed weight │
      └──────┬───────┘         └──────┬───────┘
             │ (B, in_dim, B_len)     │ (B, out_dim, F_len)
             ▼                        ▼
        transpose (0,2,1)        transpose (0,2,1)
        then reshape             then reshape
             │                        │
             ▼                        ▼
        backcast                 forecast
        (B, B_len*in_dim)        (B, F_len*out_dim)

    The two basis matrices are separate non-trainable weights, built once from
    one shared time vector. Here td = thetas_dim, B_len = backcast_length,
    F_len = forecast_length.

    :param normalize_basis: Whether to divide each polynomial row ``d > 0`` by
        ``sqrt(d + 1)``, which keeps the higher-degree rows on a similar scale.
    :type normalize_basis: bool
    :param kwargs: Arguments passed to parent NBeatsBlock.

    Input shape:
        2D tensor of shape ``(batch, backcast_length * input_dim)``.

    Output shape:
        A tuple of ``(batch, backcast_length * input_dim)`` and
        ``(batch, forecast_length * output_dim)``.
    """

    def __init__(
            self,
            normalize_basis: bool = True,
            **kwargs: Any
    ) -> None:
        """
        Record the normalization flag and reserve the two basis matrix slots.

        See the class docstring for the full parameter list.
        """
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
        Create the two polynomial basis matrices as non-trainable weights.

        Row ``d`` of each matrix is ``t ** d``, evaluated over a single time
        vector that is normalized to [-1, 1] and centered on the boundary between
        the backcast and forecast windows. The vector is then split, so the two
        matrices are continuous across that boundary. With ``normalize_basis``
        set, every row ``d > 0`` is divided by ``sqrt(d + 1)``.

        :return: Nothing. Sets ``backcast_basis_matrix`` of shape
            ``(thetas_dim, backcast_length)`` and ``forecast_basis_matrix`` of
            shape ``(thetas_dim, forecast_length)``.
        :rtype: None
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
                # Simple scaling based on degree, to keep the rows on one scale
                scale_factor = np.sqrt(degree + 1)
                backcast_basis[degree] /= scale_factor
                forecast_basis[degree] /= scale_factor

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-028: both basis matrices are
        # materialized by an initializer closing over the NumPy arrays above.
        # Do NOT use add_weight(initializer='zeros') + .assign(): Keras 3 builds
        # this block inside a StatelessScope that records the assign then discards
        # it, and a trend-only NBeatsNet.predict() then returned exactly 0.0.
        self.backcast_basis_matrix = self.add_weight(
            name='backcast_basis_matrix',
            shape=(self.thetas_dim, self.backcast_length),
            initializer=lambda shape, dtype=None: ops.cast(
                ops.convert_to_tensor(backcast_basis), dtype or self.variable_dtype
            ),
            trainable=False
        )
        self.forecast_basis_matrix = self.add_weight(
            name='forecast_basis_matrix',
            shape=(self.thetas_dim, self.forecast_length),
            initializer=lambda shape, dtype=None: ops.cast(
                ops.convert_to_tensor(forecast_basis), dtype or self.variable_dtype
            ),
            trainable=False
        )

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


@register_dl_technique("dl_techniques.layers.time_series.nbeats_blocks")
class SeasonalityBlock(NBeatsBlock):
    """
    Seasonality N-BEATS block with Fourier basis functions for periodic patterns.

    This block fixes the basis to sines and cosines, so it can only express a
    periodic pattern. Each basis matrix has ``thetas_dim`` rows. They are filled
    by ``thetas_dim // 2`` harmonics, each contributing a cosine row and a sine
    row:

        cos(2 * pi * k * t / T)
        sin(2 * pi * k * t / T)

    Here T is ``backcast_length + forecast_length``, so a harmonic keeps one
    period across both windows. If ``thetas_dim`` is odd, the leftover row is
    filled with a constant 1 (the DC term). With ``normalize_basis`` on, each row
    is divided by the norm of that harmonic over the full T-length sequence, not
    over its own window. Using one shared norm is what keeps the backcast and
    forecast on the same scale.

    Both basis matrices are non-trainable. Only theta learns.

    **Architecture Overview:**

    .. code-block:: text

        Input: (batch, backcast_length * input_dim)
                          │
                          ▼
             ┌─────────────────────────┐
             │ NBeatsBlock base        │
             │ 4 Dense + 2 theta heads │
             └───────────┬─────────────┘
             ┌───────────┴────────────┐
             ▼                        ▼
        theta_backcast           theta_forecast
             │                        │
             ▼                        ▼
        reshape                  reshape
        (B, in_dim, td)          (B, out_dim, td)
             │                        │
             ▼                        ▼
      ┌──────────────┐         ┌──────────────┐
      │ matmul with  │         │ matmul with  │
      │Fourier basis │         │Fourier basis │
      │ (td, B_len)  │         │ (td, F_len)  │
      │ fixed weight │         │ fixed weight │
      └──────┬───────┘         └──────┬───────┘
             │ (B, in_dim, B_len)     │ (B, out_dim, F_len)
             ▼                        ▼
        transpose (0,2,1)        transpose (0,2,1)
        then reshape             then reshape
             │                        │
             ▼                        ▼
        backcast                 forecast
        (B, B_len*in_dim)        (B, F_len*out_dim)

    The reshape and the matmul both use td = thetas_dim, the full row count, not
    the harmonic count ``thetas_dim // 2``. B_len = backcast_length and
    F_len = forecast_length.

    :param normalize_basis: Whether to divide each Fourier row by its norm over
        the full ``backcast_length + forecast_length`` sequence. A norm at or
        below 1e-8 leaves the row unscaled.
    :type normalize_basis: bool
    :param kwargs: Arguments passed to parent NBeatsBlock.

    Input shape:
        2D tensor of shape ``(batch, backcast_length * input_dim)``.

    Output shape:
        A tuple of ``(batch, backcast_length * input_dim)`` and
        ``(batch, forecast_length * output_dim)``.
    """

    def __init__(
            self,
            normalize_basis: bool = True,
            **kwargs: Any
    ) -> None:
        """
        Record the normalization flag and reserve the two basis matrix slots.

        See the class docstring for the full parameter list.
        """
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
        Create the two Fourier basis matrices as non-trainable weights.

        Rows are filled harmonic by harmonic, a cosine row then a sine row, until
        ``thetas_dim`` rows are used. Time indices run continuously across both
        windows, so the forecast rows continue the backcast rows rather than
        restarting at 0. With ``normalize_basis`` set, both rows of a harmonic are
        divided by that harmonic's norm over the full sequence. If ``thetas_dim``
        is odd, the last row is a constant 1.

        :return: Nothing. Sets ``backcast_basis_matrix`` of shape
            ``(thetas_dim, backcast_length)`` and ``forecast_basis_matrix`` of
            shape ``(thetas_dim, forecast_length)``.
        :rtype: None
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
            # DC component
            backcast_basis[basis_idx] = 1.0
            forecast_basis[basis_idx] = 1.0
            basis_idx += 1

        # Materialized by an initializer for the same reason as
        # TrendBlock._create_polynomial_basis: an .assign() issued during Keras 3's
        # symbolic build pass is discarded, leaving an all-zero basis and a block
        # that emits exactly zero. See the anchor there, and decisions.md D-028.
        self.backcast_basis_matrix = self.add_weight(
            name='backcast_basis_matrix',
            shape=(self.thetas_dim, self.backcast_length),
            initializer=lambda shape, dtype=None: ops.cast(
                ops.convert_to_tensor(backcast_basis), dtype or self.variable_dtype
            ),
            trainable=False
        )
        self.forecast_basis_matrix = self.add_weight(
            name='forecast_basis_matrix',
            shape=(self.thetas_dim, self.forecast_length),
            initializer=lambda shape, dtype=None: ops.cast(
                ops.convert_to_tensor(forecast_basis), dtype or self.variable_dtype
            ),
            trainable=False
        )

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
