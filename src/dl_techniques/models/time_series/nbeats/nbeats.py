"""
NBeatsNet and its factory create_nbeats_model, an N-BEATS forecaster built from
stacks of fully connected blocks that predict through doubly residual basis
expansion.

Instead of emitting the horizon directly, each block emits a short coefficient
vector theta, and the horizon is synthesized from a fixed or learned set of
basis functions, y_hat = sum_i theta_i * g_i(t). A block whose basis is a
polynomial can only produce a trend; one whose basis is sine/cosine pairs can
only produce a periodic signal, so the per-stack forecast is a real
decomposition. Every block also emits a backcast over the lookback window,
which is subtracted from its input before the residual reaches the next block
(residual_l = residual_{l-1} - backcast_l), so each block sees only what its
predecessors failed to explain, while forecasts across all blocks are summed.
The model flattens (batch, time, features) to (batch, time * features) before
each block, and applies reversible per-instance normalization (mean/std over
the time axis) by default. call() returns (forecast, final_residual): the
residual exposes what the whole stack failed to explain, so a training loop
can penalize it directly through a second loss term; predict() returns only
the forecast. share_weights_in_stack reuses one block object at every stack
position (real weight sharing, not tied copies) and is off by default.

References:
    - Oreshkin et al., 2019. N-BEATS: Neural basis expansion analysis for
      interpretable time series forecasting. ICLR 2020.
      (https://arxiv.org/abs/1905.10437)
    - Kim et al., 2022. Reversible Instance Normalization for Accurate Time-Series
      Forecasting against Distribution Shift. ICLR 2022.
"""

import keras
import numpy as np
from keras import ops, initializers, regularizers
from typing import List, Tuple, Optional, Union, Any, Dict, Callable, Sequence

from dl_techniques.utils.logger import logger
from dl_techniques.models.time_series.forecast import Forecast, ForecastMixin
from dl_techniques.layers.time_series.nbeats_blocks import (
    GenericBlock, TrendBlock, SeasonalityBlock
)
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

@register_dl_technique("dl_techniques.models.nbeats.nbeats")
class NBeatsNet(keras.Model, ForecastMixin):
    """Neural Basis Expansion Analysis for Time Series (N-BEATS) forecasting model.

    Processes input through stacks of blocks. Each block produces a backcast
    (explanation of its input) and a forecast; the backcast is subtracted from
    the input to form the residual for the next block, and all forecasts are
    summed for the final prediction. ``stack_types`` selects each stack's basis:
    ``'generic'`` (fully learned), ``'trend'`` (polynomial) or ``'seasonality'``
    (Fourier).

    Architecture:

    .. code-block:: text

        input [B, T, D]
             |
             v
        ┌──────────────┐
        │ normalize      │  (optional, reversible)
        └──────────────┘
             |
             v
        residual [B, T*D]
             |
             v
        ┌──────────────┐
        │ stack 1        │──> forecast_1
        │ (blocks)       │
        └──────────────┘
             |  residual
             v
        ┌──────────────┐
        │ stack 2 ...    │──> forecast_2 ...
        └──────────────┘
             |
             v
        final residual        sum(forecast_i) --> forecast [B, H, D]

    :param backcast_length: Length of the input time series window. Recommended
        to be 3-5x ``forecast_length``.
    :type backcast_length: int
    :param forecast_length: Length of the forecast horizon.
    :type forecast_length: int
    :param stack_types: Stack types to use, from ``'generic'``, ``'trend'``,
        ``'seasonality'``. Order matters for interpretation: trend before
        seasonality lets trend absorb the low-frequency mass first.
    :type stack_types: Sequence[str]
    :param nb_blocks_per_stack: Number of blocks per stack.
    :type nb_blocks_per_stack: int
    :param thetas_dim: Theta-vector width per stack; must match the length of
        ``stack_types``. Auto-calculated by the factory if not given directly.
    :type thetas_dim: Sequence[int]
    :param hidden_layer_units: Hidden units in each block's MLP.
    :type hidden_layer_units: int
    :param share_weights_in_stack: If ``True``, every block in a stack is the
        same object, so the stack holds one set of weights applied
        ``nb_blocks_per_stack`` times.
    :type share_weights_in_stack: bool
    :param use_normalization: Whether to apply reversible per-instance
        normalization to the input.
    :type use_normalization: bool
    :param kernel_regularizer: Regularizer for block weights.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param theta_regularizer: Regularizer for theta parameters.
    :type theta_regularizer: Optional[regularizers.Regularizer]
    :param dropout_rate: Dropout rate between blocks, in ``[0, 1)``.
    :type dropout_rate: float
    :param activation: Activation function for hidden layers.
    :type activation: Union[str, Callable]
    :param kernel_initializer: Initializer for layer weights.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param input_dim: Dimensionality of input features; greater than 1 for
        multivariate series.
    :type input_dim: int
    :param output_dim: Dimensionality of output features.
    :type output_dim: int
    :param use_bias: Whether linear layers use a bias term.
    :type use_bias: bool
    :param kwargs: Additional arguments for the ``Model`` base class.

    Input shape:
        - 3D tensor: ``(batch_size, backcast_length, input_dim)``
        - 2D tensor for univariate: ``(batch_size, backcast_length)`` - automatically
          expanded to 3D

    Output shape:
        When calling the model (e.g., in ``train_step`` or ``test_step``):
            Tuple of two tensors:

            - Forecast: ``(batch_size, forecast_length, output_dim)``
            - Final Residual: ``(batch_size, backcast_length * input_dim)``

        When calling ``model.predict()``:
            Single tensor (Keras automatically returns first output):

            - Forecast: ``(batch_size, forecast_length, output_dim)``

    Example:
        >>> # Create multivariate model
        >>> model = NBeatsNet(
        ...     backcast_length=96,
        ...     forecast_length=24,
        ...     input_dim=5,
        ...     output_dim=5,
        ...     stack_types=['trend', 'seasonality'],
        ...     nb_blocks_per_stack=3,
        ...     hidden_layer_units=256
        ... )
        >>>
        >>> # Compile and train
        >>> model.compile(optimizer='adam', loss='mse')
        >>> model.fit(train_data, epochs=100)
        >>>
        >>> # Predict
        >>> predictions = model.predict(test_data)

    Note:
        - For optimal performance, use backcast_length >= 3 × forecast_length
        - The model returns (forecast, residual) during training but only forecast
          during inference via predict()
        - Stack order matters: trend → seasonality → generic is recommended
    """

    # Valid stack type constants
    GENERIC_BLOCK: str = 'generic'
    TREND_BLOCK: str = 'trend'
    SEASONALITY_BLOCK: str = 'seasonality'
    VALID_STACK_TYPES: set = {GENERIC_BLOCK, TREND_BLOCK, SEASONALITY_BLOCK}

    # Numeric floor for the RevIN std divisor (prevents division by zero).
    NORM_EPSILON: float = 1e-7

    def __init__(
            self,
            backcast_length: int,
            forecast_length: int,
            stack_types: Sequence[str] = ('trend', 'seasonality', 'generic'),
            nb_blocks_per_stack: int = 3,
            thetas_dim: Sequence[int] = (4, 8, 16),
            hidden_layer_units: int = 256,
            share_weights_in_stack: bool = False,
            use_normalization: bool = True,
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            theta_regularizer: Optional[regularizers.Regularizer] = None,
            dropout_rate: float = 0.0,
            activation: Union[str, Callable] = 'relu',
            kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
            input_dim: int = 1,
            output_dim: int = 1,
            use_bias: bool = True,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and create every sub-layer.

        Sub-layers are created here; ``build()`` builds them explicitly so
        that saved weights restore correctly.
        """
        super().__init__(**kwargs)

        self._validate_configuration(
            backcast_length, forecast_length, stack_types, nb_blocks_per_stack,
            thetas_dim, hidden_layer_units, dropout_rate, input_dim, output_dim
        )

        # Every constructor parameter is stored, for get_config()/from_config().
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.stack_types = list(stack_types)
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = list(thetas_dim)
        self.hidden_layer_units = hidden_layer_units
        self.share_weights_in_stack = share_weights_in_stack
        self.use_normalization = use_normalization
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.theta_regularizer = regularizers.get(theta_regularizer)
        self.dropout_rate = dropout_rate
        self.activation = deserialize_activation(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        self.blocks: List[List[Union[GenericBlock, TrendBlock, SeasonalityBlock]]] = []
        self._create_block_stacks()

    def _validate_configuration(
            self,
            backcast_length: int,
            forecast_length: int,
            stack_types: List[str],
            nb_blocks_per_stack: int,
            thetas_dim: List[int],
            hidden_layer_units: int,
            dropout_rate: float,
            input_dim: int,
            output_dim: int
    ) -> None:
        """Validate model configuration parameters.

        :raises ValueError: If any parameter is invalid or inconsistent.
        """
        if backcast_length <= 0:
            raise ValueError(f"backcast_length must be positive, got {backcast_length}")
        if forecast_length <= 0:
            raise ValueError(f"forecast_length must be positive, got {forecast_length}")
        if nb_blocks_per_stack <= 0:
            raise ValueError(
                f"nb_blocks_per_stack must be positive, got {nb_blocks_per_stack}"
            )
        if hidden_layer_units <= 0:
            raise ValueError(
                f"hidden_layer_units must be positive, got {hidden_layer_units}"
            )
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")

        if len(stack_types) != len(thetas_dim):
            raise ValueError(
                f"Length of stack_types ({len(stack_types)}) must match "
                f"length of thetas_dim ({len(thetas_dim)})"
            )

        for i, (stack_type, theta_dim) in enumerate(zip(stack_types, thetas_dim)):
            if stack_type not in self.VALID_STACK_TYPES:
                raise ValueError(
                    f"Invalid stack type at index {i}: '{stack_type}'. "
                    f"Must be one of: {self.VALID_STACK_TYPES}"
                )
            if theta_dim <= 0:
                raise ValueError(f"thetas_dim[{i}] must be positive, got {theta_dim}")

        # Warn if backcast/forecast ratio is suboptimal
        ratio = backcast_length / forecast_length
        if ratio < 3.0:
            logger.warning(
                f"backcast_length ({backcast_length}) / forecast_length "
                f"({forecast_length}) = {ratio:.1f}. For optimal performance, "
                f"use ratio >= 3.0"
            )

    def _create_block_stacks(self) -> None:
        """Create every block for every stack. build() builds them later."""

        for stack_id, (stack_type, theta_dim) in enumerate(
                zip(self.stack_types, self.thetas_dim)
        ):
            stack_blocks = []

            for block_id in range(self.nb_blocks_per_stack):
                if self.share_weights_in_stack and stack_blocks:
                    # Sharing is realized by reusing the *same layer object* at
                    # every position in the stack, which is what makes the
                    # weights literally one set rather than tied copies.
                    stack_blocks.append(stack_blocks[0])
                    continue

                block_name = (
                    f"stack_{stack_id}_shared_{stack_type}"
                    if self.share_weights_in_stack
                    else f"stack_{stack_id}_block_{block_id}_{stack_type}"
                )

                # Common block configuration
                block_kwargs = {
                    'units': self.hidden_layer_units,
                    'thetas_dim': theta_dim,
                    'backcast_length': self.backcast_length,
                    'forecast_length': self.forecast_length,
                    'input_dim': self.input_dim,
                    'output_dim': self.output_dim,
                    'activation': self.activation,
                    'use_bias': self.use_bias,
                    'kernel_initializer': self.kernel_initializer,
                    'kernel_regularizer': self.kernel_regularizer,
                    'theta_regularizer': self.theta_regularizer,
                    'dropout_rate': self.dropout_rate,
                    'name': block_name
                }

                # Create appropriate block type
                if stack_type == self.GENERIC_BLOCK:
                    block = GenericBlock(**block_kwargs)
                elif stack_type == self.TREND_BLOCK:
                    block = TrendBlock(**block_kwargs)
                else:  # SEASONALITY_BLOCK
                    block = SeasonalityBlock(**block_kwargs)

                stack_blocks.append(block)

            self.blocks.append(stack_blocks)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every block explicitly, so weights exist before a load restores them.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Expand 2D to 3D if needed to get consistent shape
        if len(input_shape) == 2:
            batch_size, seq_len = input_shape
            build_shape = (batch_size, seq_len, self.input_dim)
        else:
            build_shape = input_shape

        # Compute the shape that blocks will receive (flattened feature dim)
        batch_size = build_shape[0]
        block_input_shape = (batch_size, self.backcast_length * self.input_dim)

        for stack_blocks in self.blocks:
            for block in stack_blocks:
                # Under share_weights_in_stack the same object appears at every
                # position; building it twice would add a second set of weights.
                if not block.built:
                    block.build(block_input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Run the stacks, accumulating forecasts and passing residuals forward.

        :param inputs: Input tensor, ``(batch_size, backcast_length)`` or
            ``(batch_size, backcast_length, input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether dropout runs in training mode.
        :type training: Optional[bool]
        :return: ``(forecast, final_residual)`` — forecast of shape
            ``(batch_size, forecast_length, output_dim)``, residual of shape
            ``(batch_size, backcast_length * input_dim)``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]

        Note:
            ``model.predict()`` returns only the forecast (Keras keeps the
            first output). Training and validation see both outputs.
        """
        # Expand 2D inputs to 3D for consistent processing
        if len(inputs.shape) == 2:
            inputs_3d = ops.expand_dims(inputs, axis=-1)
        else:
            inputs_3d = inputs

        batch_size = ops.shape(inputs_3d)[0]

        # Apply instance normalization if enabled
        if self.use_normalization:
            mean = ops.mean(inputs_3d, axis=1, keepdims=True)
            std = ops.std(inputs_3d, axis=1, keepdims=True)
            # Prevent division by zero
            std = ops.maximum(std, self.NORM_EPSILON)
            normalized_input = (inputs_3d - mean) / std
        else:
            normalized_input = inputs_3d
            mean = None
            std = None

        # N-BEATS is a dense architecture: flatten (batch, time, feat) to (batch, time*feat).
        processed_input = ops.reshape(
            normalized_input,
            (batch_size, self.backcast_length * self.input_dim)
        )

        # Initialize residual and forecast accumulator
        residual = processed_input
        forecast_sum = ops.zeros(
            (batch_size, self.forecast_length * self.output_dim),
            dtype=self.compute_dtype
        )

        for stack_blocks in self.blocks:
            for block in stack_blocks:
                backcast, forecast = block(residual, training=training)
                residual = residual - backcast
                forecast_sum = forecast_sum + forecast

        forecast_3d = ops.reshape(
            forecast_sum,
            (batch_size, self.forecast_length, self.output_dim)
        )

        if self.use_normalization:
            if self.input_dim == self.output_dim:
                forecast_3d = (forecast_3d * std) + mean

                residual_3d = ops.reshape(
                    residual,
                    (batch_size, self.backcast_length, self.input_dim)
                )
                residual_3d = residual_3d * std

                residual = ops.reshape(
                    residual_3d,
                    (batch_size, self.backcast_length * self.input_dim)
                )
            else:
                # input_dim != output_dim: assume the first output_dim input
                # features are the targets and slice their statistics.
                std_out = std[:, :, :self.output_dim]
                mean_out = mean[:, :, :self.output_dim]
                forecast_3d = (forecast_3d * std_out) + mean_out

                residual_3d = ops.reshape(
                    residual,
                    (batch_size, self.backcast_length, self.input_dim)
                )
                residual_3d = residual_3d * std
                residual = ops.reshape(
                    residual_3d,
                    (batch_size, self.backcast_length * self.input_dim)
                )

        return forecast_3d, residual

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Return the shapes of the forecast and residual outputs.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(forecast_shape, residual_shape)`` —
            ``(batch_size, forecast_length, output_dim)`` and
            ``(batch_size, backcast_length * input_dim)``.
        :rtype: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]
        """
        batch_size = input_shape[0]
        forecast_shape = (batch_size, self.forecast_length, self.output_dim)
        residual_shape = (batch_size, self.backcast_length * self.input_dim)
        return forecast_shape, residual_shape

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor parameter, for ``from_config()``.

        :return: The full configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'backcast_length': self.backcast_length,
            'forecast_length': self.forecast_length,
            'stack_types': self.stack_types,
            'nb_blocks_per_stack': self.nb_blocks_per_stack,
            'thetas_dim': self.thetas_dim,
            'hidden_layer_units': self.hidden_layer_units,
            'share_weights_in_stack': self.share_weights_in_stack,
            'use_normalization': self.use_normalization,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'theta_regularizer': regularizers.serialize(self.theta_regularizer),
            'dropout_rate': self.dropout_rate,
            'activation': serialize_activation(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'use_bias': self.use_bias,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NBeatsNet':
        """Reconstruct a model from its configuration dictionary.

        :param config: Dictionary from :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new instance with the same configuration.
        :rtype: NBeatsNet
        """
        if config.get('kernel_regularizer') is not None:
            config['kernel_regularizer'] = regularizers.deserialize(
                config['kernel_regularizer']
            )
        if config.get('theta_regularizer') is not None:
            config['theta_regularizer'] = regularizers.deserialize(
                config['theta_regularizer']
            )
        if config.get('kernel_initializer') is not None:
            config['kernel_initializer'] = initializers.deserialize(
                config['kernel_initializer']
            )

        return cls(**config)

    def _forecast(self, x: Any, **kwargs: Any) -> Forecast:
        """Produce a point-only :class:`Forecast` (``ForecastMixin`` hook).

        N-BEATS is a point forecaster: it emits no predictive intervals, so
        this never fabricates quantile bands. ``predict`` returns the forecast
        tensor (the first element of the ``(forecast, residual)`` tuple from
        :meth:`call`).

        :param x: Context window, ``[B, backcast_length, input_dim]`` (or the
            2D univariate form ``[B, backcast_length]``).
        :type x: Any
        :param kwargs: Forwarded to ``keras.Model.predict`` (e.g. ``batch_size``).
        :return: A :class:`Forecast` with ``point`` of shape
            ``[B, forecast_length, output_dim]`` and ``quantiles=None``.
        :rtype: Forecast
        """
        kwargs.setdefault("verbose", 0)
        preds = self.predict(x, **kwargs)
        # call() returns a (forecast, residual) tuple; predict mirrors that, so
        # take the first element (the forecast [B, H, output_dim]) when present.
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        return Forecast(
            point=np.asarray(preds),
            quantiles=None,
            quantile_levels=None,
        )


def create_nbeats_model(
        backcast_length: int = 96,
        forecast_length: int = 24,
        stack_types: Sequence[str] = ('trend', 'seasonality', 'generic'),
        nb_blocks_per_stack: int = 3,
        thetas_dim: Optional[List[int]] = None,
        hidden_layer_units: int = 256,
        activation: str = "relu",
        use_normalization: bool = True,
        dropout_rate: float = 0.0,
        input_dim: int = 1,
        output_dim: int = 1,
        **kwargs: Any
) -> NBeatsNet:
    """Create an N-BEATS model with sensible defaults and auto-calculated theta dimensions.

    :param backcast_length: Length of the input sequence.
    :type backcast_length: int
    :param forecast_length: Length of the forecast sequence.
    :type forecast_length: int
    :param stack_types: Stack types to use.
    :type stack_types: Sequence[str]
    :param nb_blocks_per_stack: Number of blocks per stack.
    :type nb_blocks_per_stack: int
    :param thetas_dim: Theta dimensions per stack. If ``None``, calculated as
        4 (trend, cubic polynomial), ``2 * min(forecast_length // 2, 16)``
        (seasonality, Fourier harmonics), or ``max(16, forecast_length * 2)``
        (generic).
    :type thetas_dim: Optional[List[int]]
    :param hidden_layer_units: Hidden units in each layer.
    :type hidden_layer_units: int
    :param activation: Activation function for hidden layers.
    :type activation: str
    :param use_normalization: Whether to apply instance normalization.
    :type use_normalization: bool
    :param dropout_rate: Dropout rate for regularization.
    :type dropout_rate: float
    :param input_dim: Dimensionality of input features.
    :type input_dim: int
    :param output_dim: Dimensionality of output features.
    :type output_dim: int
    :param kwargs: Additional arguments passed to the ``NBeatsNet`` constructor.
    :return: An un-compiled N-BEATS model.
    :rtype: NBeatsNet

    Example::

        model = create_nbeats_model(backcast_length=96, forecast_length=24)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(train_data, epochs=100)

        model = create_nbeats_model(
            backcast_length=168,
            forecast_length=24,
            input_dim=10,
            output_dim=10,
            stack_types=['trend', 'seasonality', 'generic'],
            hidden_layer_units=512
        )
    """
    if thetas_dim is None:
        thetas_dim = []
        for stack_type in stack_types:
            if stack_type == 'trend':
                thetas_dim.append(4)
            elif stack_type == 'seasonality':
                harmonics = min(forecast_length // 2, 16)
                thetas_dim.append(harmonics * 2)
            else:  # 'generic'
                thetas_dim.append(max(16, forecast_length * 2))

    model = NBeatsNet(
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        stack_types=stack_types,
        nb_blocks_per_stack=nb_blocks_per_stack,
        thetas_dim=thetas_dim,
        hidden_layer_units=hidden_layer_units,
        activation=activation,
        use_normalization=use_normalization,
        dropout_rate=dropout_rate,
        input_dim=input_dim,
        output_dim=output_dim,
        **kwargs
    )

    # Log configuration
    ratio = backcast_length / forecast_length
    logger.info("Created N-BEATS model with configuration:")
    logger.info(
        f"  - Architecture: {len(stack_types)} stacks, "
        f"{nb_blocks_per_stack} blocks each"
    )
    logger.info(
        f"  - Sequence: {backcast_length} → {forecast_length} "
        f"(ratio: {ratio:.1f})"
    )
    logger.info(f"  - Dimensions: Input {input_dim} → Output {output_dim}")
    logger.info(f"  - Stack types: {stack_types}")
    logger.info(f"  - Theta dimensions: {thetas_dim}")
    logger.info(f"  - Hidden units: {hidden_layer_units}")

    if dropout_rate > 0.0:
        logger.info(f"  - Dropout: {dropout_rate}")

    if ratio < 3.0:
        logger.warning(
            f"  ⚠ Backcast/forecast ratio is {ratio:.1f}. "
            f"Consider increasing to >= 3.0 for better performance."
        )

    return model

