import keras
from typing import List
from keras import ops, layers

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.time_series.nbeatsx_blocks import ExogenousBlock
from dl_techniques.layers.time_series.nbeats_blocks import GenericBlock, TrendBlock, SeasonalityBlock

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NBeatsXNet(keras.Model):
    """
    N-BEATSx: Neural Basis Expansion Analysis with Exogenous Variables.

    Extends the N-BEATS architecture to include exogenous variables via a specialized
    block (ExogenousBlock) and TCN encoder.

    **Input Handling**:
    To handle history and future exogenous variables, this model expects a dictionary
    input during `fit` and `predict`:
    ```python
    inputs = {
        "target_history": (batch, backcast_len, 1),
        "exog_history":   (batch, backcast_len, exog_dim),
        "exog_forecast":  (batch, forecast_len, exog_dim)
    }
    ```

    **Stack Types**:
    - 'trend', 'seasonality': Standard N-BEATS blocks (endogenous only).
    - 'exogenous': NBEATSx block using TCN on exogenous variables.
    - 'exogenous_interpretable': NBEATSx block using raw exogenous variables.

    Args:
        exogenous_dim: Integer, number of exogenous features.
        tcn_filters: Integer, channels for TCN encoder.
        tcn_kernel_size: Integer, kernel size for TCN.
        tcn_dropout: Float, dropout for TCN.
        (All other args same as NBeatsNet)
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
            tcn_filters: int = 16,
            tcn_kernel_size: int = 3,
            tcn_dropout: float = 0.0,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Configuration
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.exogenous_dim = exogenous_dim
        self.stack_types = stack_types
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = thetas_dim
        self.hidden_layer_units = hidden_layer_units
        self.share_weights_in_stack = share_weights_in_stack
        self.use_normalization = use_normalization
        self.dropout_rate = dropout_rate

        # TCN Config
        self.tcn_filters = tcn_filters
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_dropout = tcn_dropout

        self.blocks = []
        self.dropout_layers = []
        self._create_block_stacks()

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
                    'use_normalization': self.use_normalization,
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

    def get_config(self):
        config = super().get_config()
        config.update({
            'backcast_length': self.backcast_length,
            'forecast_length': self.forecast_length,
            'exogenous_dim': self.exogenous_dim,
            'stack_types': self.stack_types,
            'nb_blocks_per_stack': self.nb_blocks_per_stack,
            'thetas_dim': self.thetas_dim,
            'hidden_layer_units': self.hidden_layer_units,
            'tcn_filters': self.tcn_filters,
            'tcn_kernel_size': self.tcn_kernel_size,
            'tcn_dropout': self.tcn_dropout,
            'use_normalization': self.use_normalization,
            'dropout_rate': self.dropout_rate
        })
        return config

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
