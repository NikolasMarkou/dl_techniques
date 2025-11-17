import keras
from keras import ops, layers, initializers, regularizers
from typing import List, Tuple, Optional, Union, Any, Dict, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.time_series.nbeats_blocks import (
    GenericBlock, TrendBlock, SeasonalityBlock
)


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class NBeatsNet(keras.Model):
    """
    Neural Basis Expansion Analysis for Time Series (N-BEATS) forecasting model.

    This implementation follows modern Keras 3 patterns and includes proper
    normalization, serialization, and an optional reconstruction loss for
    regularization, forcing the model to fully explain the input signal.

    **Intent**: Provide a production-ready N-BEATS implementation for time series
    forecasting with proper residual connections, normalization, and serialization
    support following modern Keras 3 best practices.

    **Architecture**:
    The model processes the input through a series of stacks. Each block in a
    stack produces a backcast and a forecast. The backcast is subtracted from
    the input to form a residual, which is passed to the next block. Forecasts
    from all blocks are summed to produce the final prediction.

    Args:
        backcast_length: Integer, length of the input time series window.
        forecast_length: Integer, length of the forecast horizon.
        stack_types: List of strings, types of stacks to use ('generic', 'trend',
            'seasonality'). Default: ['trend', 'seasonality'].
        nb_blocks_per_stack: Integer, number of blocks per stack. Defaults to 3.
        thetas_dim: List of integers, dimensionality of theta for each stack.
        hidden_layer_units: Integer, number of hidden units in each block.
        share_weights_in_stack: Boolean, whether to share weights within stacks.
        use_normalization: Boolean, whether to use instance normalization.
        kernel_regularizer: Optional regularizer for block weights.
        theta_regularizer: Optional regularizer for theta parameters.
        dropout_rate: Float, dropout rate. Must be in [0, 1).
        activation: String or callable, activation function for hidden layers.
        kernel_initializer: String or Initializer, initializer for layer weights.
        input_dim: Integer, dimensionality of input features. Defaults to 1.
        output_dim: Integer, dimensionality of output features. Defaults to 1.
        use_bias: Boolean, whether to use bias terms in linear layers.
        reconstruction_weight: Float, weight for the reconstruction loss. If > 0,
            a loss term is added to penalize the final residual, forcing the
            model to explain the entire input signal. Defaults to 0.0 for
            backward compatibility.
        **kwargs: Additional keyword arguments for the Model base class.

    Input shape:
        3D tensor: `(batch_size, backcast_length, input_dim)`.
        2D tensor for univariate: `(batch_size, backcast_length)`.

    Output shape:
        When calling the model (e.g., in `train_step` or `test_step`):
            A tuple of two tensors:
            - Forecast: `(batch_size, forecast_length, output_dim)`
            - Final Residual: `(batch_size, backcast_length * input_dim)`
        When calling `model.predict()`:
            A single tensor (Keras automatically returns the first output):
            - Forecast: `(batch_size, forecast_length, output_dim)`
    """

    # Valid stack type constants
    GENERIC_BLOCK = 'generic'
    TREND_BLOCK = 'trend'
    SEASONALITY_BLOCK = 'seasonality'
    VALID_STACK_TYPES = {GENERIC_BLOCK, TREND_BLOCK, SEASONALITY_BLOCK}

    def __init__(
            self,
            backcast_length: int,
            forecast_length: int,
            stack_types: List[str] = ['trend', 'seasonality', 'generic'],
            nb_blocks_per_stack: int = 3,
            thetas_dim: List[int] = [4, 8],
            hidden_layer_units: int = 256,
            share_weights_in_stack: bool = False,
            use_normalization: bool = True,
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            theta_regularizer: Optional[regularizers.Regularizer] = None,
            dropout_rate: float = 0.0,
            activation: Union[str, Callable] = 'relu',
            kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
            use_bias: bool = True,
            reconstruction_weight: float = 0.0,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate configuration
        self._validate_configuration(
            backcast_length, forecast_length, stack_types, nb_blocks_per_stack,
            thetas_dim, hidden_layer_units, dropout_rate
        )

        # Store ALL configuration parameters for serialization
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
        self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.input_dim = 1
        self.output_dim = 1
        self.use_bias = use_bias
        self.reconstruction_weight = reconstruction_weight

        self.output_projection = None

        self.blocks: List[List[Union[GenericBlock, TrendBlock, SeasonalityBlock]]] = []
        self.dropout_layers: List[layers.Dropout] = []
        self._create_block_stacks()

    def _validate_configuration(
            self, backcast_length: int, forecast_length: int, stack_types: List[str],
            nb_blocks_per_stack: int, thetas_dim: List[int],
            hidden_layer_units: int, dropout_rate: float
    ) -> None:
        """Validate model configuration parameters."""
        if backcast_length <= 0:
            raise ValueError(f"backcast_length must be positive, got {backcast_length}")
        if forecast_length <= 0:
            raise ValueError(f"forecast_length must be positive, got {forecast_length}")
        if nb_blocks_per_stack <= 0:
            raise ValueError(
                f"nb_blocks_per_stack must be positive, got {nb_blocks_per_stack}")
        if hidden_layer_units <= 0:
            raise ValueError(
                f"hidden_layer_units must be positive, got {hidden_layer_units}")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

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

        ratio = backcast_length / forecast_length
        if ratio < 3.0:
            logger.warning(
                f"backcast_length ({backcast_length}) / forecast_length ({forecast_length}) "
                f"= {ratio:.1f}. For optimal performance, use ratio >= 3.0"
            )

    def _create_block_stacks(self) -> None:
        """Create all N-BEATS block stacks."""
        dropout_counter = 0
        block_backcast_len = self.backcast_length * self.input_dim
        block_forecast_len = self.forecast_length * self.input_dim

        for stack_id, (stack_type, theta_dim) in enumerate(
                zip(self.stack_types, self.thetas_dim)):
            stack_blocks = []
            for block_id in range(self.nb_blocks_per_stack):
                block_name = f"stack_{stack_id}_block_{block_id}_{stack_type}"
                block_kwargs = {
                    'units': self.hidden_layer_units,
                    'thetas_dim': theta_dim,
                    'backcast_length': block_backcast_len,
                    'forecast_length': block_forecast_len,
                    'share_weights': self.share_weights_in_stack,
                    'activation': self.activation,
                    'use_bias': self.use_bias,
                    'kernel_initializer': self.kernel_initializer,
                    'kernel_regularizer': self.kernel_regularizer,
                    'theta_regularizer': self.theta_regularizer,
                    'name': block_name
                }

                if stack_type == self.GENERIC_BLOCK:
                    block = GenericBlock(**block_kwargs)
                elif stack_type == self.TREND_BLOCK:
                    block = TrendBlock(**block_kwargs)
                else:  # Seasonality
                    block = SeasonalityBlock(**block_kwargs)
                stack_blocks.append(block)

                if self.dropout_rate > 0.0:
                    self.dropout_layers.append(
                        layers.Dropout(self.dropout_rate, name=f"dropout_{dropout_counter}")
                    )
                    dropout_counter += 1
            self.blocks.append(stack_blocks)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the model and all its sub-layers."""
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Forward pass through the N-BEATS network.

        Always returns a tuple of (forecast, final_residual). During inference,
        `model.predict()` will automatically return only the first element.
        """
        if len(inputs.shape) == 2:
            inputs_3d = ops.expand_dims(inputs, axis=-1)
        else:
            inputs_3d = inputs

        batch_size = ops.shape(inputs_3d)[0]

        if self.use_normalization:
            mean = keras.ops.mean(inputs_3d, axis=1, keepdims=True)
            std = keras.ops.maximum(keras.ops.std(inputs_3d + 1e-5, axis=1, keepdims=True), 1e-7)
            normalized_input = (inputs_3d - mean) / std
        else:
            normalized_input = inputs_3d

        processed_input = ops.reshape(
            normalized_input,
            (batch_size, self.backcast_length * self.input_dim)
        )

        residual = processed_input
        forecast_sum = ops.zeros(
            (batch_size, self.forecast_length * self.input_dim),
            dtype=self.compute_dtype
        )

        for stack_blocks in self.blocks:
            for block in stack_blocks:
                backcast, forecast = block(residual, training=training)
                residual = residual - backcast
                forecast_sum = forecast_sum + forecast

        forecast_3d = ops.reshape(
            forecast_sum,
            (batch_size, self.forecast_length, self.input_dim)
        )

        if self.use_normalization:
            forecast_3d = (forecast_3d * std) + mean

        final_residual = residual
        return forecast_3d, final_residual

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Compute output shape of the model (forecast, residual)."""
        batch_size = input_shape[0]
        forecast_shape = (batch_size, self.forecast_length, self.output_dim)
        residual_shape = (batch_size, self.backcast_length * self.input_dim)
        return forecast_shape, residual_shape

    def get_config(self) -> Dict[str, Any]:
        """Return complete configuration for serialization."""
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
            'activation': self.activation,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'use_bias': self.use_bias,
            'reconstruction_weight': self.reconstruction_weight,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NBeatsNet':
        """Create model instance from configuration."""
        # Deserialize complex objects
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


# ---------------------------------------------------------------------
# factory method
# ---------------------------------------------------------------------

def create_nbeats_model(
        backcast_length: int = 96,
        forecast_length: int = 24,
        stack_types: List[str] = ['trend', 'seasonality'],
        nb_blocks_per_stack: int = 3,
        thetas_dim: Optional[List[int]] = None,
        hidden_layer_units: int = 256,
        activation: str = "relu",
        use_normalization: bool = True,
        reconstruction_weight: float = 0.0,
        **kwargs: Any
) -> NBeatsNet:
    """
    Create an N-BEATS model instance with optimal defaults.

    This factory simplifies model creation with sensible defaults and automatic
    theta dimension calculation. It returns an un-compiled model instance.

    Args:
        backcast_length: Length of input sequence.
        forecast_length: Length of forecast sequence.
        stack_types: Types of stacks to use.
        nb_blocks_per_stack: Number of blocks per stack.
        thetas_dim: Theta dimensions for each stack. Auto-calculated if None.
        hidden_layer_units: Hidden units in each layer.
        activation: Activation function for hidden layers.
        use_normalization: Whether to use instance normalization.
        reconstruction_weight: Weight for reconstruction loss.
        **kwargs: Additional arguments passed to NBeatsNet constructor.

    Returns:
        An un-compiled N-BEATS model instance.
    """
    if thetas_dim is None:
        thetas_dim = []
        for stack_type in stack_types:
            if stack_type == 'trend':
                thetas_dim.append(4)  # 3rd order polynomial
            elif stack_type == 'seasonality':
                harmonics = min(forecast_length // 2, 16)
                thetas_dim.append(harmonics * 2)
            else:  # 'generic'
                thetas_dim.append(max(16, forecast_length * 2))

    model = NBeatsNet(
        activation=activation,
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        stack_types=stack_types,
        nb_blocks_per_stack=nb_blocks_per_stack,
        thetas_dim=thetas_dim,
        hidden_layer_units=hidden_layer_units,
        use_normalization=use_normalization,
        reconstruction_weight=reconstruction_weight,
        **kwargs
    )

    ratio = backcast_length / forecast_length
    logger.info("Created N-BEATS model with configuration:")
    logger.info(
        f"  - Architecture: {len(stack_types)} stacks, {nb_blocks_per_stack} blocks each"
    )
    logger.info(f"  - Sequence: {backcast_length} -> {forecast_length} (ratio: {ratio:.1f})")
    logger.info(f"  - Theta dimensions: {thetas_dim}")
    if reconstruction_weight > 0.0:
        logger.info(f"  - Reconstruction Loss Weight: {reconstruction_weight}")

    return model