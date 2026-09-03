"""
NBeatsXNet and its factory create_nbeatsx_model, the doubly residual N-BEATS
stack extended with an exogenous-covariate block type, alongside trend /
seasonality / generic blocks.

Plain N-BEATS forecasts a series from its own past alone, so it has no port
for covariates known in advance (calendar effects, scheduled prices, weather
forecasts). Concatenating covariates onto the target would work but would mix
target and covariate into every block, breaking the readable decomposition.
Instead, N-BEATSx keeps the same residual stacking law and adds a block type
whose basis is a function of the covariates: history and future covariates are
concatenated along time, a dilated causal temporal convolutional network
encodes them to a fixed channel width, and the result splits at the boundary
into a backcast basis and a forecast basis. The block's dense trunk, which
sees only the target residual, emits one weight per basis channel rather than
per timestep. Setting use_tcn=False selects an interpretable variant where the
raw covariate tensor is the basis and each theta component is the signed
contribution of one named covariate.

The model takes a dictionary input (target_history, exog_history,
exog_forecast) rather than a single tensor, since these are three tensors of
two different time lengths. Endogenous blocks always use input_dim=1 and
output_dim=1: the residual stream is the univariate target, and covariate
width lives entirely in exogenous_dim. Reversible instance normalization
covers the target only; covariates pass through unscaled. call() returns the
forecast tensor alone, with no reconstruction output.

References:
    - Olivares et al., 2023. Neural basis expansion analysis with exogenous
      variables: Forecasting electricity prices with NBEATSx. International
      Journal of Forecasting. (https://arxiv.org/abs/2104.05522)
    - Oreshkin et al., 2019. N-BEATS: Neural basis expansion analysis for
      interpretable time series forecasting. ICLR 2020.
      (https://arxiv.org/abs/1905.10437)
    - Bai et al., 2018. An Empirical Evaluation of Generic Convolutional and
      Recurrent Networks for Sequence Modeling. (https://arxiv.org/abs/1803.01271)
    - Kim et al., 2022. Reversible Instance Normalization for Accurate Time-Series
      Forecasting against Distribution Shift. ICLR 2022.
"""

import keras
from typing import Any, Callable, Dict, List, Optional, Union, Sequence
from keras import ops, layers, initializers, regularizers

from dl_techniques.utils.logger import logger
from dl_techniques.layers.time_series.nbeatsx_blocks import ExogenousBlock
from dl_techniques.layers.time_series.nbeats_blocks import GenericBlock, TrendBlock, SeasonalityBlock
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.models.nbeats.nbeatsx")
class NBeatsXNet(keras.Model):
    """N-BEATSx: Neural Basis Expansion Analysis with Exogenous Variables.

    Extends the doubly residual N-BEATS topology with an exogenous-variable
    pathway, so side information in the history window and the forecast
    horizon (calendar features, prices, weather) can be folded into the
    forecast without breaking the residual stacking discipline. Endogenous
    blocks operate on the normalized target residual; exogenous blocks
    (:class:`ExogenousBlock`) consume the history and future exogenous
    tensors through an optional TCN encoder and emit their own backcast and
    forecast contributions into the same residual stream.

    Architecture:

    .. code-block:: text

        target_history [B, backcast, 1]
             |
             v
        ┌──────────────┐
        │ revin          │  (optional)
        └──────────────┘
             |
             v
        residual [B, backcast]
             |
             v
        ┌──────────────┐   exog_history, exog_forecast
        │ stack 0        │◄──────────────────────────
        │ trend/season/  │
        │ generic/       │
        │ exogenous(tcn) │
        └──────────────┘
             |  residual'          forecast_sum
             v                          |
            ...                         v
                              ┌──────────────┐
                              │ denorm         │  (optional)
                              └──────────────┘
                                          |
                                          v
                                   forecast [B, forecast, 1]

    Exogenous and endogenous blocks alternate freely within ``stack_types``;
    each subtracts its backcast from the running residual and adds its
    forecast to the accumulator.

    Input handling requires a dictionary during ``fit``/``predict``::

        inputs = {
            "target_history": (batch, backcast_len, 1),
            "exog_history":   (batch, backcast_len, exog_dim),
            "exog_forecast":  (batch, forecast_len, exog_dim),
        }

    Stack types:
        - ``'trend'`` / ``'seasonality'`` / ``'generic'``: standard N-BEATS
          blocks (endogenous target residual only).
        - ``'exogenous'``: block using a TCN over exogenous variables.
        - ``'exogenous_interpretable'``: block using raw (un-encoded)
          exogenous variables.

    :param backcast_length: Length of the input (history) window.
    :type backcast_length: int
    :param forecast_length: Length of the forecast horizon.
    :type forecast_length: int
    :param exogenous_dim: Number of exogenous features per timestep.
    :type exogenous_dim: int
    :param stack_types: Stack-type strings (see Stack types above).
    :type stack_types: Sequence[str]
    :param nb_blocks_per_stack: Blocks per stack.
    :type nb_blocks_per_stack: int
    :param thetas_dim: Basis-expansion width, one entry per stack.
    :type thetas_dim: Sequence[int]
    :param hidden_layer_units: Hidden width of each block's dense trunk.
    :type hidden_layer_units: int
    :param share_weights_in_stack: If ``True``, every block in a stack is the
        same object, dividing the stack's parameter count by
        ``nb_blocks_per_stack``.
    :type share_weights_in_stack: bool
    :param use_normalization: Whether to apply reversible instance
        normalization to the target series (RevIN at the model boundary).
    :type use_normalization: bool
    :param use_block_normalization: Whether to enable the RMSNorm layers
        inside every block's dense trunk. ``None`` follows
        ``use_normalization``. Pass ``False`` to keep the target RevIN while
        matching ``NBeatsNet``, whose blocks never receive this flag.
    :type use_block_normalization: Optional[bool]
    :param dropout_rate: Residual-stream dropout probability, in ``[0, 1)``.
    :type dropout_rate: float
    :param activation: Activation for block hidden layers.
    :type activation: Union[str, Callable]
    :param use_bias: Whether block dense layers use a bias term.
    :type use_bias: bool
    :param kernel_initializer: Initializer for block dense kernels.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Regularizer for block dense kernels.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param theta_regularizer: Regularizer for block theta projections.
    :type theta_regularizer: Optional[regularizers.Regularizer]
    :param tcn_filters: Channels for the exogenous TCN encoder.
    :type tcn_filters: int
    :param tcn_kernel_size: Kernel size for the exogenous TCN.
    :type tcn_kernel_size: int
    :param tcn_dropout_rate: Dropout inside the exogenous TCN.
    :type tcn_dropout_rate: float
    :param kwargs: Forwarded to ``keras.Model``.

    Example:
        >>> import keras
        >>> model = NBeatsXNet(
        ...     backcast_length=48, forecast_length=12, exogenous_dim=3,
        ...     stack_types=['trend', 'exogenous'], thetas_dim=[4, 16],
        ...     kernel_regularizer=keras.regularizers.L2(1e-4),
        ... )
        >>> inputs = {
        ...     'target_history': keras.random.normal((8, 48, 1)),
        ...     'exog_history':   keras.random.normal((8, 48, 3)),
        ...     'exog_forecast':  keras.random.normal((8, 12, 3)),
        ... }
        >>> y_hat = model(inputs)
        >>> y_hat.shape
        (8, 12, 1)
    """

    EXOGENOUS_BLOCK: str = 'exogenous'
    EXOGENOUS_INTERP_BLOCK: str = 'exogenous_interpretable'

    def __init__(
            self,
            backcast_length: int,
            forecast_length: int,
            exogenous_dim: int,
            stack_types: Sequence[str] = ('trend', 'seasonality', 'exogenous'),
            nb_blocks_per_stack: int = 3,
            thetas_dim: Sequence[int] = (4, 8, 16),
            hidden_layer_units: int = 256,
            share_weights_in_stack: bool = False,
            use_normalization: bool = True,
            use_block_normalization: Optional[bool] = None,
            dropout_rate: float = 0.0,
            activation: Union[str, Callable] = 'relu',
            use_bias: bool = False,
            kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            theta_regularizer: Optional[regularizers.Regularizer] = None,
            tcn_filters: int = 16,
            tcn_kernel_size: int = 3,
            tcn_dropout_rate: float = 0.0,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Validate configuration before storing any state.
        self._validate_configuration(
            backcast_length, forecast_length, exogenous_dim,
            nb_blocks_per_stack, hidden_layer_units, dropout_rate,
            stack_types, thetas_dim,
        )

        # Configuration
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.exogenous_dim = exogenous_dim
        self.stack_types = list(stack_types)
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = list(thetas_dim)
        self.hidden_layer_units = hidden_layer_units
        self.share_weights_in_stack = share_weights_in_stack
        self.use_normalization = use_normalization
        # DECISION plan-2026-08-19T163559-499b6f0e/D-116: keep `use_block_normalization`
        # additive; do not rename `use_normalization` to `use_target_normalization`.
        # It stays the historical coupling (target RevIN + in-block RMSNorm) or every stored config breaks. See decisions.md.
        self.use_block_normalization = use_block_normalization
        self.dropout_rate = dropout_rate
        self.activation = deserialize_activation(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.theta_regularizer = regularizers.get(theta_regularizer)

        # TCN Config
        self.tcn_filters = tcn_filters
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_dropout_rate = tcn_dropout_rate

        self.blocks = []
        self.dropout_layers = []
        self._create_block_stacks()

    def _validate_configuration(
            self,
            backcast_length: int,
            forecast_length: int,
            exogenous_dim: int,
            nb_blocks_per_stack: int,
            hidden_layer_units: int,
            dropout_rate: float,
            stack_types: List[str],
            thetas_dim: List[int],
    ) -> None:
        """Validate constructor arguments, raising ValueError on bad input."""
        if backcast_length <= 0:
            raise ValueError(f"backcast_length must be positive, got {backcast_length}")
        if forecast_length <= 0:
            raise ValueError(f"forecast_length must be positive, got {forecast_length}")
        if exogenous_dim <= 0:
            raise ValueError(f"exogenous_dim must be positive, got {exogenous_dim}")
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
        if len(stack_types) != len(thetas_dim):
            raise ValueError(
                f"Length of stack_types ({len(stack_types)}) must match "
                f"length of thetas_dim ({len(thetas_dim)})"
            )

    def _create_block_stacks(self) -> None:
        dropout_counter = 0

        for stack_id, (stack_type, theta_dim) in enumerate(zip(self.stack_types, self.thetas_dim)):
            stack_blocks = []

            for block_id in range(self.nb_blocks_per_stack):
                if self.share_weights_in_stack and stack_blocks:
                    # Sharing is realized by reusing the *same layer object* at
                    # every position in the stack, which is what makes the
                    # weights literally one set rather than tied copies.
                    stack_blocks.append(stack_blocks[0])
                    if self.dropout_rate > 0.0:
                        self.dropout_layers.append(
                            layers.Dropout(
                                self.dropout_rate, name=f"dropout_{dropout_counter}"
                            )
                        )
                        dropout_counter += 1
                    continue

                block_name = (
                    f"stack_{stack_id}_shared_{stack_type}"
                    if self.share_weights_in_stack
                    else f"stack_{stack_id}_block_{block_id}_{stack_type}"
                )

                # Base args for standard N-BEATS blocks
                base_kwargs = {
                    'units': self.hidden_layer_units,
                    'thetas_dim': theta_dim,
                    'backcast_length': self.backcast_length,
                    'forecast_length': self.forecast_length,
                    'input_dim': 1,  # Endogenous target is usually univariate
                    'output_dim': 1,
                    'use_normalization': self._block_normalization,
                    'activation': self.activation,
                    'use_bias': self.use_bias,
                    'kernel_initializer': self.kernel_initializer,
                    'kernel_regularizer': self.kernel_regularizer,
                    'theta_regularizer': self.theta_regularizer,
                    'name': block_name
                }

                if stack_type == 'trend':
                    block = TrendBlock(**base_kwargs)
                elif stack_type == 'seasonality':
                    block = SeasonalityBlock(**base_kwargs)
                elif stack_type == 'generic':
                    block = GenericBlock(**base_kwargs)
                elif stack_type in [self.EXOGENOUS_BLOCK, self.EXOGENOUS_INTERP_BLOCK]:
                    # NBEATSx Exogenous Block
                    use_tcn = (stack_type == self.EXOGENOUS_BLOCK)
                    block = ExogenousBlock(
                        exogenous_dim=self.exogenous_dim,
                        tcn_filters=self.tcn_filters,
                        tcn_kernel_size=self.tcn_kernel_size,
                        tcn_dropout_rate=self.tcn_dropout_rate,
                        use_tcn=use_tcn,
                        **base_kwargs
                    )
                else:
                    raise ValueError(f"Unknown stack type: {stack_type}")

                stack_blocks.append(block)

                if self.dropout_rate > 0.0:
                    self.dropout_layers.append(
                        layers.Dropout(self.dropout_rate, name=f"dropout_{dropout_counter}")
                    )
                    dropout_counter += 1

            self.blocks.append(stack_blocks)

    def build(self, input_shape: Optional[Any] = None) -> None:
        """Build every block against the flattened univariate residual shape."""
        # Univariate target residual shape.
        dummy_resid_shape = (None, self.backcast_length * 1)

        for stack in self.blocks:
            for block in stack:
                # Under share_weights_in_stack the same object appears at every
                # position; building it twice would add a second set of weights.
                if not block.built:
                    block.build(dummy_resid_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Normalize the target, run the stacks, and denormalize the forecast.

        :param inputs: Dictionary with ``'target_history'`` ``(B, backcast, 1)``,
            ``'exog_history'`` ``(B, backcast, exog_dim)`` and
            ``'exog_forecast'`` ``(B, forecast, exog_dim)``.
        :type inputs: Dict[str, keras.KerasTensor]
        :param training: Whether dropout runs in training mode.
        :type training: Optional[bool]
        :return: Forecast, ``(B, forecast_length, 1)``.
        :rtype: keras.KerasTensor
        """
        y_hist = inputs['target_history']
        x_hist = inputs['exog_history']
        x_fore = inputs['exog_forecast']

        batch_size = ops.shape(y_hist)[0]

        if self.use_normalization:
            y_mean = ops.mean(y_hist, axis=1, keepdims=True)
            y_std = ops.std(y_hist, axis=1, keepdims=True)
            y_std = ops.maximum(y_std, 1e-7)
            residual = (y_hist - y_mean) / y_std
        else:
            residual = y_hist
            y_mean = None
            y_std = None

        # Dense stacks need a flat (B, Time * 1) residual.
        residual = ops.reshape(residual, (batch_size, self.backcast_length))

        forecast_sum = ops.zeros((batch_size, self.forecast_length))

        dropout_idx = 0

        for stack in self.blocks:
            for block in stack:
                if isinstance(block, ExogenousBlock):
                    backcast, forecast = block(
                        residual,
                        training=training,
                        exogenous_inputs=(x_hist, x_fore)
                    )
                else:
                    backcast, forecast = block(residual, training=training)

                residual = residual - backcast
                forecast_sum = forecast_sum + forecast

                if self.dropout_rate > 0.0 and dropout_idx < len(self.dropout_layers):
                    residual = self.dropout_layers[dropout_idx](residual, training=training)
                    dropout_idx += 1

        forecast_3d = ops.reshape(forecast_sum, (batch_size, self.forecast_length, 1))

        if self.use_normalization:
            forecast_3d = (forecast_3d * y_std) + y_mean

        # predict() returns only the forecast — there is no reconstruction output.
        return forecast_3d

    @property
    def _block_normalization(self) -> bool:
        """Resolve the in-block RMSNorm switch.

        ``use_block_normalization is None`` means "follow ``use_normalization``",
        which is what this class did before the two effects were separable. See
        the D-116 anchor in ``__init__``.
        """
        if self.use_block_normalization is None:
            return self.use_normalization
        return self.use_block_normalization

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'backcast_length': self.backcast_length,
            'forecast_length': self.forecast_length,
            'exogenous_dim': self.exogenous_dim,
            'stack_types': self.stack_types,
            'nb_blocks_per_stack': self.nb_blocks_per_stack,
            'thetas_dim': self.thetas_dim,
            'hidden_layer_units': self.hidden_layer_units,
            'share_weights_in_stack': self.share_weights_in_stack,
            'use_normalization': self.use_normalization,
            'use_block_normalization': self.use_block_normalization,
            'dropout_rate': self.dropout_rate,
            'activation': serialize_activation(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'theta_regularizer': regularizers.serialize(self.theta_regularizer),
            'tcn_filters': self.tcn_filters,
            'tcn_kernel_size': self.tcn_kernel_size,
            'tcn_dropout_rate': self.tcn_dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NBeatsXNet':
        """Reconstruct from config, deserializing initializer/regularizers."""
        if config.get('kernel_initializer') is not None:
            config['kernel_initializer'] = initializers.deserialize(
                config['kernel_initializer']
            )
        if config.get('kernel_regularizer') is not None:
            config['kernel_regularizer'] = regularizers.deserialize(
                config['kernel_regularizer']
            )
        if config.get('theta_regularizer') is not None:
            config['theta_regularizer'] = regularizers.deserialize(
                config['theta_regularizer']
            )
        return cls(**config)


def create_nbeatsx_model(
        backcast_length: int = 168,
        forecast_length: int = 24,
        exogenous_dim: int = 2,
        stack_types: Sequence[str] = ('trend', 'seasonality', 'exogenous'),
        **kwargs
) -> NBeatsXNet:
    """Create an :class:`NBeatsXNet` with auto-calculated theta dimensions if not given.

    :param backcast_length: Length of the input (history) window.
    :type backcast_length: int
    :param forecast_length: Length of the forecast horizon.
    :type forecast_length: int
    :param exogenous_dim: Number of exogenous features per timestep.
    :type exogenous_dim: int
    :param stack_types: Stack types to use.
    :type stack_types: Sequence[str]
    :param kwargs: Forwarded to :class:`NBeatsXNet`.
    :return: The configured model.
    :rtype: NBeatsXNet
    """
    if 'thetas_dim' not in kwargs:
        thetas_dim = []
        for s in stack_types:
            if s == 'trend':
                thetas_dim.append(4)
            elif s == 'seasonality':
                thetas_dim.append(8)
            elif s == 'exogenous':
                # Matches tcn_filters.
                thetas_dim.append(16)
            else:
                thetas_dim.append(16)
        kwargs['thetas_dim'] = thetas_dim

    model = NBeatsXNet(
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        exogenous_dim=exogenous_dim,
        stack_types=stack_types,
        **kwargs
    )

    logger.info(f"Created NBEATSx with stacks: {stack_types}")
    return model
