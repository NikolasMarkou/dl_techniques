import keras
from typing import Any, Callable, Dict, List, Optional, Union
from keras import ops, layers, initializers, regularizers

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.time_series.nbeatsx_blocks import ExogenousBlock
from dl_techniques.layers.time_series.nbeats_blocks import GenericBlock, TrendBlock, SeasonalityBlock

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NBeatsXNet(keras.Model):
    """N-BEATSx: Neural Basis Expansion Analysis with Exogenous Variables.

    **Intent**: Extend the interpretable doubly-residual N-BEATS topology
    (Trend / Seasonality decomposition over a univariate target) with a
    dedicated exogenous-variable pathway, so that side information available
    both in the history window AND the forecast horizon (calendar features,
    prices, weather, etc.) can be folded into the forecast without breaking
    the residual stacking discipline. Endogenous blocks operate on the
    normalized target residual; exogenous blocks (``ExogenousBlock``) consume
    the history+future exogenous tensors via an optional TCN encoder and emit
    their own backcast/forecast contributions into the same residual stream.

    Architecture::

        target_history ──RevIN──> residual ─┐
                                            v
          ┌───────────────── stack 0 ─────────────────┐
          │  block: Trend / Seasonality / Generic      │  (endogenous)
          │  block: Exogenous(TCN(exog_hist, exog_fut)) │  (exogenous)
          └───────────────────────────────────────────┘
                 │ backcast (subtracted)  │ forecast (summed)
                 v                         v
              residual'               forecast_sum ──denorm──> y_hat

    Each block subtracts its backcast from the running residual and adds its
    forecast to the global accumulator; the summed forecast is de-normalized
    with the target's own statistics (reversible instance norm).

    **Input Handling**: history and future exogenous variables require a
    dictionary input during ``fit`` / ``predict``::

        inputs = {
            "target_history": (batch, backcast_len, 1),
            "exog_history":   (batch, backcast_len, exog_dim),
            "exog_forecast":  (batch, forecast_len, exog_dim),
        }

    **Stack Types**:
        - ``'trend'`` / ``'seasonality'`` / ``'generic'``: standard N-BEATS
          blocks (endogenous target residual only).
        - ``'exogenous'``: NBEATSx block using a TCN over exogenous variables.
        - ``'exogenous_interpretable'``: NBEATSx block using raw (un-encoded)
          exogenous variables.

    Args:
        backcast_length: Integer, length of the input (history) window.
        forecast_length: Integer, length of the forecast horizon.
        exogenous_dim: Integer, number of exogenous features per timestep.
        stack_types: List of stack-type strings (see **Stack Types**).
        nb_blocks_per_stack: Integer, blocks per stack.
        thetas_dim: List of basis-expansion dims, one per stack.
        hidden_layer_units: Integer, hidden width of each block's FC trunk.
        share_weights_in_stack: Boolean, share FC weights across blocks in a
            stack (threaded to each block as ``share_weights``).
        use_normalization: Boolean, apply reversible instance norm to target.
        dropout_rate: Float in [0, 1), residual-stream dropout probability.
        activation: Activation for block hidden layers.
        use_bias: Boolean, bias on block FC layers.
        kernel_initializer: Initializer for block FC kernels.
        kernel_regularizer: Optional regularizer for block FC kernels.
        theta_regularizer: Optional regularizer for block theta projections.
        tcn_filters: Integer, channels for the exogenous TCN encoder.
        tcn_kernel_size: Integer, kernel size for the exogenous TCN.
        tcn_dropout: Float, dropout inside the exogenous TCN.
        **kwargs: Forwarded to ``keras.Model``.

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
            stack_types: List[str] = ['trend', 'seasonality', 'exogenous'],
            nb_blocks_per_stack: int = 3,
            thetas_dim: List[int] = [4, 8, 16],
            hidden_layer_units: int = 256,
            share_weights_in_stack: bool = False,
            use_normalization: bool = True,
            dropout_rate: float = 0.0,
            activation: Union[str, Callable] = 'relu',
            use_bias: bool = False,
            kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            theta_regularizer: Optional[regularizers.Regularizer] = None,
            tcn_filters: int = 16,
            tcn_kernel_size: int = 3,
            tcn_dropout: float = 0.0,
            **kwargs
    ):
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
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.theta_regularizer = regularizers.get(theta_regularizer)

        # TCN Config
        self.tcn_filters = tcn_filters
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_dropout = tcn_dropout

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

    def _create_block_stacks(self):
        dropout_counter = 0

        for stack_id, (stack_type, theta_dim) in enumerate(zip(self.stack_types, self.thetas_dim)):
            stack_blocks = []

            for block_id in range(self.nb_blocks_per_stack):
                block_name = f"stack_{stack_id}_block_{block_id}_{stack_type}"

                # Base args for standard N-BEATS blocks
                base_kwargs = {
                    'units': self.hidden_layer_units,
                    'thetas_dim': theta_dim,
                    'backcast_length': self.backcast_length,
                    'forecast_length': self.forecast_length,
                    'input_dim': 1,  # Endogenous target is usually univariate
                    'output_dim': 1,
                    'share_weights': self.share_weights_in_stack,
                    'use_normalization': self.use_normalization,
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
                        tcn_dropout=self.tcn_dropout,
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

    def build(self, input_shape=None):
        # We manually trigger builds for sub-blocks to ensure variables exist
        # Assuming input is dict, we define standard shapes
        dummy_resid_shape = (None, self.backcast_length * 1)  # Univariate target

        # NOTE: D-003 (plan_2026-06-11_fe7401f4) resolved by plan_2026-06-11_5f49f080 —
        # TemporalConvNet now has a real build(); ExogenousBlock.build()->encoder.build()
        # materializes all TCN Conv1D children, so no eager dummy forward is needed.
        for stack in self.blocks:
            for block in stack:
                block.build(dummy_resid_shape)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Args:
            inputs: Dictionary containing:
                - 'target_history': (B, Backcast, 1)
                - 'exog_history': (B, Backcast, ExogDim)
                - 'exog_forecast': (B, Forecast, ExogDim)
        """
        # Unpack Inputs
        y_hist = inputs['target_history']
        x_hist = inputs['exog_history']
        x_fore = inputs['exog_forecast']

        batch_size = ops.shape(y_hist)[0]

        # 1. Normalize Target History (Endogenous)
        if self.use_normalization:
            y_mean = ops.mean(y_hist, axis=1, keepdims=True)
            y_std = ops.std(y_hist, axis=1, keepdims=True)
            y_std = ops.maximum(y_std, 1e-7)
            residual = (y_hist - y_mean) / y_std
        else:
            residual = y_hist
            y_mean = None
            y_std = None

        # Flatten Residual for Dense Stacks: (B, Time * 1)
        residual = ops.reshape(residual, (batch_size, self.backcast_length))

        # Accumulator for forecasts
        forecast_sum = ops.zeros((batch_size, self.forecast_length))

        dropout_idx = 0

        for stack in self.blocks:
            for block in stack:
                if isinstance(block, ExogenousBlock):
                    # Pass exogenous data specifically
                    backcast, forecast = block(
                        residual,
                        training=training,
                        exogenous_inputs=(x_hist, x_fore)
                    )
                else:
                    # Standard N-BEATS block (Trend/Seasonality/Generic)
                    backcast, forecast = block(residual, training=training)

                # Update Residual (subtract backcast explanation)
                residual = residual - backcast

                # Accumulate Forecast
                forecast_sum = forecast_sum + forecast

                # Dropout
                if self.dropout_rate > 0.0 and dropout_idx < len(self.dropout_layers):
                    residual = self.dropout_layers[dropout_idx](residual, training=training)
                    dropout_idx += 1

        # Reshape forecast output (B, Time, 1)
        forecast_3d = ops.reshape(forecast_sum, (batch_size, self.forecast_length, 1))

        # Denormalize
        if self.use_normalization:
            forecast_3d = (forecast_3d * y_std) + y_mean

        # Return only forecast for predict() consistency
        return forecast_3d

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
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'theta_regularizer': regularizers.serialize(self.theta_regularizer),
            'tcn_filters': self.tcn_filters,
            'tcn_kernel_size': self.tcn_kernel_size,
            'tcn_dropout': self.tcn_dropout,
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

# ---------------------------------------------------------------------

def create_nbeatsx_model(
        backcast_length: int = 168,
        forecast_length: int = 24,
        exogenous_dim: int = 2,
        stack_types: List[str] = ['trend', 'seasonality', 'exogenous'],
        **kwargs
) -> NBeatsXNet:
    """Factory for NBEATSx Model."""

    # Auto-calculate thetas if not provided
    if 'thetas_dim' not in kwargs:
        thetas_dim = []
        for s in stack_types:
            if s == 'trend':
                thetas_dim.append(4)
            elif s == 'seasonality':
                thetas_dim.append(8)
            elif s == 'exogenous':
                thetas_dim.append(16)  # Matches tcn_filters
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

# ---------------------------------------------------------------------
