"""
Fixed N-BEATS Model for Time Series Forecasting.

This module implements a corrected N-BEATS architecture that addresses
shape mismatches and gradient flow issues.
"""

import keras
from keras import ops
from typing import List, Tuple, Optional, Union, Any, Dict

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class NBeatsNet(keras.Model):
    """Fixed N-BEATS neural network for time series forecasting.

    Args:
        input_dim: Integer, dimensionality of input time series (default: 1).
        output_dim: Integer, dimensionality of output time series (default: 1).
        backcast_length: Integer, length of the input time series.
        forecast_length: Integer, length of the forecast horizon.
        stack_types: List of strings, types of stacks to use.
        nb_blocks_per_stack: Integer, number of blocks per stack.
        thetas_dim: List of integers, dimensionality of theta parameters for each stack.
        share_weights_in_stack: Boolean, whether to share weights within each stack.
        hidden_layer_units: Integer, number of hidden units in each layer.
        **kwargs: Additional keyword arguments for the Model parent class.
    """

    GENERIC_BLOCK = 'generic'
    TREND_BLOCK = 'trend'
    SEASONALITY_BLOCK = 'seasonality'

    def __init__(
            self,
            input_dim: int = 1,
            output_dim: int = 1,
            backcast_length: int = 10,
            forecast_length: int = 1,
            stack_types: List[str] = ['trend', 'seasonality'],
            nb_blocks_per_stack: int = 3,
            thetas_dim: List[int] = [4, 8],
            share_weights_in_stack: bool = False,
            hidden_layer_units: int = 256,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.stack_types = stack_types
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = thetas_dim
        self.share_weights_in_stack = share_weights_in_stack
        self.hidden_layer_units = hidden_layer_units

        # Validate inputs
        if len(self.stack_types) != len(self.thetas_dim):
            raise ValueError(
                f"Length of stack_types ({len(self.stack_types)}) must match "
                f"length of thetas_dim ({len(self.thetas_dim)})"
            )

        # Initialize blocks storage
        self.blocks: List[List[Any]] = []
        self.output_projection = None

        # Build the network
        self._build_network()

    def _build_network(self) -> None:
        """Build the N-BEATS network architecture."""
        from dl_techniques.layers.time_series.nbeats_blocks import (
            GenericBlock, TrendBlock, SeasonalityBlock
        )

        logger.info(f"Building N-BEATS network with {len(self.stack_types)} stacks")

        # Create blocks for each stack
        for stack_id, (stack_type, theta_dim) in enumerate(zip(self.stack_types, self.thetas_dim)):
            stack_blocks = []

            for block_id in range(self.nb_blocks_per_stack):
                block_name = f"stack_{stack_id}_block_{block_id}_{stack_type}"

                # Create appropriate block type
                if stack_type == self.GENERIC_BLOCK:
                    block = GenericBlock(
                        units=self.hidden_layer_units,
                        thetas_dim=theta_dim,
                        backcast_length=self.backcast_length,
                        forecast_length=self.forecast_length,
                        share_weights=self.share_weights_in_stack,
                        name=block_name
                    )
                elif stack_type == self.TREND_BLOCK:
                    block = TrendBlock(
                        units=self.hidden_layer_units,
                        thetas_dim=theta_dim,
                        backcast_length=self.backcast_length,
                        forecast_length=self.forecast_length,
                        share_weights=self.share_weights_in_stack,
                        name=block_name
                    )
                elif stack_type == self.SEASONALITY_BLOCK:
                    block = SeasonalityBlock(
                        units=self.hidden_layer_units,
                        thetas_dim=theta_dim,
                        backcast_length=self.backcast_length,
                        forecast_length=self.forecast_length,
                        share_weights=self.share_weights_in_stack,
                        name=block_name
                    )
                else:
                    raise ValueError(f"Unknown stack type: {stack_type}")

                stack_blocks.append(block)

            self.blocks.append(stack_blocks)

        # Output projection layer if input/output dimensions differ
        if self.input_dim != self.output_dim:
            self.output_projection = keras.layers.Dense(
                self.output_dim,
                activation='linear',
                name='output_projection'
            )

        logger.info(f"N-BEATS network built with {sum(len(stack) for stack in self.blocks)} total blocks")

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the N-BEATS model.

        Args:
            inputs: Input tensor of shape (batch_size, backcast_length) or (batch_size, backcast_length, 1).
            training: Boolean indicating training mode.

        Returns:
            Forecast tensor of shape (batch_size, forecast_length).
        """
        # Handle input shapes - ensure 2D for processing
        if len(inputs.shape) == 3:
            # Remove last dimension if it's 1
            if inputs.shape[-1] == 1:
                inputs = ops.squeeze(inputs, axis=-1)
            else:
                # Take first channel if multivariate
                inputs = inputs[..., 0]
        elif len(inputs.shape) == 1:
            # Add batch dimension if missing
            inputs = ops.expand_dims(inputs, axis=0)

        batch_size = ops.shape(inputs)[0]

        # Initialize residual input and forecast accumulator
        residual = inputs  # Shape: (batch_size, backcast_length)
        forecast_sum = ops.zeros((batch_size, self.forecast_length))

        # Process each stack
        for stack_id, stack_blocks in enumerate(self.blocks):
            for block_id, block in enumerate(stack_blocks):
                # Get backcast and forecast from the block
                backcast, forecast = block(residual, training=training)

                # Subtract backcast from residual (residual connection)
                residual = residual - backcast

                # Add forecast to accumulator
                forecast_sum = forecast_sum + forecast

        # Apply output projection if needed
        if self.output_projection is not None:
            # Add channel dimension for projection
            forecast_sum = ops.expand_dims(forecast_sum, axis=-1)
            forecast_sum = self.output_projection(forecast_sum, training=training)
            # Remove channel dimension
            forecast_sum = ops.squeeze(forecast_sum, axis=-1)

        return forecast_sum

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the model."""
        batch_size = input_shape[0]
        return (batch_size, self.forecast_length)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'backcast_length': self.backcast_length,
            'forecast_length': self.forecast_length,
            'stack_types': self.stack_types,
            'nb_blocks_per_stack': self.nb_blocks_per_stack,
            'thetas_dim': self.thetas_dim,
            'share_weights_in_stack': self.share_weights_in_stack,
            'hidden_layer_units': self.hidden_layer_units,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NBeatsNet':
        """Create model from configuration."""
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print model summary."""
        logger.info("N-BEATS Model Summary:")
        logger.info(f"  Input dimension: {self.input_dim}")
        logger.info(f"  Output dimension: {self.output_dim}")
        logger.info(f"  Backcast length: {self.backcast_length}")
        logger.info(f"  Forecast length: {self.forecast_length}")
        logger.info(f"  Stack types: {self.stack_types}")
        logger.info(f"  Blocks per stack: {self.nb_blocks_per_stack}")
        logger.info(f"  Theta dimensions: {self.thetas_dim}")
        logger.info(f"  Hidden units: {self.hidden_layer_units}")
        logger.info(f"  Share weights: {self.share_weights_in_stack}")

        total_blocks = sum(len(stack) for stack in self.blocks)
        logger.info(f"  Total blocks: {total_blocks}")

        # Call parent summary if needed
        super().summary(**kwargs)


def create_nbeats_model(
        config: Optional[Dict[str, Any]] = None,
        optimizer: Union[str, keras.optimizers.Optimizer] = 'adam',
        loss: Union[str, keras.losses.Loss] = 'mae',
        metrics: Optional[List[Union[str, keras.metrics.Metric]]] = None,
        learning_rate: float = 0.001,
        **kwargs: Any
) -> NBeatsNet:
    """Create a compiled N-BEATS model with sensible defaults.

    Args:
        config: Configuration dictionary for the model.
        optimizer: Optimizer to use for training.
        loss: Loss function to use.
        metrics: List of metrics to track during training.
        learning_rate: Learning rate for the optimizer.
        **kwargs: Additional arguments passed to NBeatsNet.

    Returns:
        Compiled N-BEATS model ready for training.
    """
    # Default config
    default_config = {
        'backcast_length': 48,
        'forecast_length': 12,
        'stack_types': ['trend', 'seasonality'],
        'nb_blocks_per_stack': 2,
        'thetas_dim': [3, 6],
        'hidden_layer_units': 128,
        'share_weights_in_stack': False,
    }

    if config is not None:
        default_config.update(config)

    # Override with kwargs
    default_config.update(kwargs)

    # Create model
    model = NBeatsNet(**default_config)

    # Setup default metrics if not provided
    if metrics is None:
        metrics = ['mae', 'mse']

    # Setup optimizer with learning rate
    if isinstance(optimizer, str):
        if optimizer.lower() == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'adamw':
            optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
        else:
            optimizer = keras.optimizers.get(optimizer)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    logger.info(f"Created and compiled N-BEATS model with {optimizer.__class__.__name__} optimizer")

    return model