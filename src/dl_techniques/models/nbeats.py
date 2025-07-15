"""
N-BEATS Model for Time Series Forecasting.

This module implements the N-BEATS (Neural Basis Expansion Analysis for Time Series)
architecture as described in the paper.
"""

import keras
from keras import ops
from typing import List, Tuple, Optional, Union, Any, Dict

from dl_techniques.utils.logger import logger
from dl_techniques.layers.time_series.nbeats_blocks import (
    NBeatsBlock, GenericBlock, TrendBlock, SeasonalityBlock
)
from dl_techniques.losses.smape_loss import SMAPELoss, MASELoss

@keras.saving.register_keras_serializable()
class NBeatsNet(keras.Model):
    """N-BEATS neural network for time series forecasting.

    N-BEATS is a deep neural architecture based on backward and forward residual
    links and a very deep stack of fully-connected layers. It uses basis expansion
    to interpret the outputs of the network.

    Args:
        input_dim: Integer, dimensionality of input time series (default: 1).
        output_dim: Integer, dimensionality of output time series (default: 1).
        backcast_length: Integer, length of the input time series.
        forecast_length: Integer, length of the forecast horizon.
        stack_types: List of strings, types of stacks to use. Options are:
            'generic', 'trend', 'seasonality'.
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
        self.blocks: List[List[NBeatsBlock]] = []
        self.output_projection = None

        # Build the network
        self._build_network()

    def _build_network(self) -> None:
        """Build the N-BEATS network architecture."""
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
            inputs: Input tensor of shape (batch_size, backcast_length, input_dim).
            training: Boolean indicating whether the model should behave in
                training mode or inference mode.

        Returns:
            Forecast tensor of shape (batch_size, forecast_length, output_dim).
        """
        # Handle input shapes
        if len(inputs.shape) == 2:
            # Add feature dimension if not present
            inputs = ops.expand_dims(inputs, axis=-1)

        batch_size = ops.shape(inputs)[0]

        # Split input into individual time series if multivariate
        if self.input_dim > 1:
            x_series = [inputs[..., i] for i in range(self.input_dim)]
        else:
            x_series = [ops.squeeze(inputs, axis=-1)]

        # Initialize forecast accumulator
        forecast_sum = [ops.zeros((batch_size, self.forecast_length)) for _ in range(self.input_dim)]

        # Process each stack
        for stack_id, stack_blocks in enumerate(self.blocks):
            for block_id, block in enumerate(stack_blocks):
                # Process each input dimension
                for dim_id in range(self.input_dim):
                    # Get backcast and forecast from the block
                    backcast, forecast = block(x_series[dim_id], training=training)

                    # Subtract backcast from input (residual connection)
                    x_series[dim_id] = x_series[dim_id] - backcast

                    # Add forecast to accumulator
                    forecast_sum[dim_id] = forecast_sum[dim_id] + forecast

        # Combine multivariate forecasts
        if self.input_dim > 1:
            # Stack forecasts along the feature dimension
            forecast_output = ops.stack(forecast_sum, axis=-1)
        else:
            # Single dimension output
            forecast_output = ops.expand_dims(forecast_sum[0], axis=-1)

        # Apply output projection if needed
        if self.output_projection is not None:
            forecast_output = self.output_projection(forecast_output, training=training)

        return forecast_output

    def predict_with_backcast(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Predict both forecast and final backcast residual.

        Args:
            inputs: Input tensor of shape (batch_size, backcast_length, input_dim).
            training: Boolean indicating training mode.

        Returns:
            Tuple of (forecast, backcast_residual) tensors.
        """
        # Handle input shapes
        if len(inputs.shape) == 2:
            inputs = ops.expand_dims(inputs, axis=-1)

        batch_size = ops.shape(inputs)[0]

        # Split input into individual time series
        if self.input_dim > 1:
            x_series = [inputs[..., i] for i in range(self.input_dim)]
        else:
            x_series = [ops.squeeze(inputs, axis=-1)]

        # Initialize forecast accumulator
        forecast_sum = [ops.zeros((batch_size, self.forecast_length)) for _ in range(self.input_dim)]

        # Process each stack
        for stack_id, stack_blocks in enumerate(self.blocks):
            for block_id, block in enumerate(stack_blocks):
                for dim_id in range(self.input_dim):
                    backcast, forecast = block(x_series[dim_id], training=training)
                    x_series[dim_id] = x_series[dim_id] - backcast
                    forecast_sum[dim_id] = forecast_sum[dim_id] + forecast

        # Prepare outputs
        if self.input_dim > 1:
            forecast_output = ops.stack(forecast_sum, axis=-1)
            backcast_residual = ops.stack(x_series, axis=-1)
        else:
            forecast_output = ops.expand_dims(forecast_sum[0], axis=-1)
            backcast_residual = ops.expand_dims(x_series[0], axis=-1)

        # Apply output projection if needed
        if self.output_projection is not None:
            forecast_output = self.output_projection(forecast_output, training=training)
            backcast_residual = self.output_projection(backcast_residual, training=training)

        return forecast_output, backcast_residual

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the model.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape.
        """
        batch_size = input_shape[0]
        return (batch_size, self.forecast_length, self.output_dim)

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


class NBeatsModelBuilder:
    """Builder class for creating N-BEATS models with fluent API.

    This builder provides a fluent interface for configuring N-BEATS models
    with various architectural choices and hyperparameters.

    Example:
        >>> builder = NBeatsModelBuilder()
        >>> model = (builder
        ...     .set_sequence_lengths(backcast=24, forecast=12)
        ...     .add_trend_stack(degree=3, blocks=3)
        ...     .add_seasonality_stack(harmonics=8, blocks=3)
        ...     .set_hidden_units(512)
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        """Initialize the builder with default values."""
        self._input_dim = 1
        self._output_dim = 1
        self._backcast_length = 24
        self._forecast_length = 12
        self._stack_types = []
        self._thetas_dim = []
        self._nb_blocks_per_stack = 3
        self._hidden_layer_units = 512
        self._share_weights_in_stack = False
        self._model_name = 'nbeats_model'

    def set_sequence_lengths(
            self,
            backcast: int,
            forecast: int
    ) -> 'NBeatsModelBuilder':
        """Set the backcast and forecast sequence lengths.

        Args:
            backcast: Length of the input time series.
            forecast: Length of the forecast horizon.

        Returns:
            Self for method chaining.
        """
        self._backcast_length = backcast
        self._forecast_length = forecast
        return self

    def set_dimensions(
            self,
            input_dim: int = 1,
            output_dim: int = 1
    ) -> 'NBeatsModelBuilder':
        """Set the input and output dimensions.

        Args:
            input_dim: Input dimensionality.
            output_dim: Output dimensionality.

        Returns:
            Self for method chaining.
        """
        self._input_dim = input_dim
        self._output_dim = output_dim
        return self

    def add_generic_stack(
            self,
            thetas_dim: int = 64,
            blocks: int = 3
    ) -> 'NBeatsModelBuilder':
        """Add a generic stack to the model.

        Args:
            thetas_dim: Dimensionality of theta parameters.
            blocks: Number of blocks in the stack.

        Returns:
            Self for method chaining.
        """
        self._stack_types.append('generic')
        self._thetas_dim.append(thetas_dim)
        self._nb_blocks_per_stack = blocks
        return self

    def add_trend_stack(
            self,
            degree: int = 3,
            blocks: int = 3
    ) -> 'NBeatsModelBuilder':
        """Add a trend stack to the model.

        Args:
            degree: Polynomial degree for trend modeling.
            blocks: Number of blocks in the stack.

        Returns:
            Self for method chaining.
        """
        self._stack_types.append('trend')
        self._thetas_dim.append(degree)
        self._nb_blocks_per_stack = blocks
        return self

    def add_seasonality_stack(
            self,
            harmonics: int = 8,
            blocks: int = 3
    ) -> 'NBeatsModelBuilder':
        """Add a seasonality stack to the model.

        Args:
            harmonics: Number of Fourier harmonics.
            blocks: Number of blocks in the stack.

        Returns:
            Self for method chaining.
        """
        self._stack_types.append('seasonality')
        self._thetas_dim.append(harmonics)
        self._nb_blocks_per_stack = blocks
        return self

    def set_hidden_units(self, units: int) -> 'NBeatsModelBuilder':
        """Set the number of hidden units in each layer.

        Args:
            units: Number of hidden units.

        Returns:
            Self for method chaining.
        """
        self._hidden_layer_units = units
        return self

    def enable_weight_sharing(self) -> 'NBeatsModelBuilder':
        """Enable weight sharing within stacks.

        Returns:
            Self for method chaining.
        """
        self._share_weights_in_stack = True
        return self

    def disable_weight_sharing(self) -> 'NBeatsModelBuilder':
        """Disable weight sharing within stacks.

        Returns:
            Self for method chaining.
        """
        self._share_weights_in_stack = False
        return self

    def set_model_name(self, name: str) -> 'NBeatsModelBuilder':
        """Set the model name.

        Args:
            name: Name for the model.

        Returns:
            Self for method chaining.
        """
        self._model_name = name
        return self

    def build(self) -> NBeatsNet:
        """Build the N-BEATS model with current configuration.

        Returns:
            Configured N-BEATS model.

        Raises:
            ValueError: If no stacks have been added.
        """
        if not self._stack_types:
            raise ValueError("At least one stack must be added to the model")

        logger.info(f"Building N-BEATS model '{self._model_name}' with configuration:")
        logger.info(f"  Backcast length: {self._backcast_length}")
        logger.info(f"  Forecast length: {self._forecast_length}")
        logger.info(f"  Input/Output dims: {self._input_dim}/{self._output_dim}")
        logger.info(f"  Stack types: {self._stack_types}")
        logger.info(f"  Theta dimensions: {self._thetas_dim}")
        logger.info(f"  Hidden units: {self._hidden_layer_units}")
        logger.info(f"  Weight sharing: {self._share_weights_in_stack}")

        return NBeatsNet(
            input_dim=self._input_dim,
            output_dim=self._output_dim,
            backcast_length=self._backcast_length,
            forecast_length=self._forecast_length,
            stack_types=self._stack_types,
            nb_blocks_per_stack=self._nb_blocks_per_stack,
            thetas_dim=self._thetas_dim,
            share_weights_in_stack=self._share_weights_in_stack,
            hidden_layer_units=self._hidden_layer_units,
            name=self._model_name
        )

    def reset(self) -> 'NBeatsModelBuilder':
        """Reset the builder to default values.

        Returns:
            Self for method chaining.
        """
        self.__init__()
        return self


class NBeatsConfig:
    """Configuration class for N-BEATS models.

    This class provides predefined configurations for common N-BEATS model
    architectures and use cases.
    """

    @staticmethod
    def small_model() -> Dict[str, Any]:
        """Configuration for a small N-BEATS model.

        Returns:
            Configuration dictionary for a small model.
        """
        return {
            'backcast_length': 24,
            'forecast_length': 6,
            'stack_types': ['trend', 'seasonality'],
            'nb_blocks_per_stack': 2,
            'thetas_dim': [3, 6],
            'hidden_layer_units': 128,
            'share_weights_in_stack': False,
        }

    @staticmethod
    def medium_model() -> Dict[str, Any]:
        """Configuration for a medium N-BEATS model.

        Returns:
            Configuration dictionary for a medium model.
        """
        return {
            'backcast_length': 48,
            'forecast_length': 12,
            'stack_types': ['trend', 'seasonality'],
            'nb_blocks_per_stack': 3,
            'thetas_dim': [4, 8],
            'hidden_layer_units': 256,
            'share_weights_in_stack': False,
        }

    @staticmethod
    def large_model() -> Dict[str, Any]:
        """Configuration for a large N-BEATS model.

        Returns:
            Configuration dictionary for a large model.
        """
        return {
            'backcast_length': 96,
            'forecast_length': 24,
            'stack_types': ['trend', 'seasonality'],
            'nb_blocks_per_stack': 4,
            'thetas_dim': [6, 12],
            'hidden_layer_units': 512,
            'share_weights_in_stack': True,
        }

    @staticmethod
    def generic_model() -> Dict[str, Any]:
        """Configuration for a generic N-BEATS model.

        Returns:
            Configuration dictionary for a generic model.
        """
        return {
            'backcast_length': 48,
            'forecast_length': 12,
            'stack_types': ['generic', 'generic'],
            'nb_blocks_per_stack': 3,
            'thetas_dim': [64, 64],
            'hidden_layer_units': 512,
            'share_weights_in_stack': False,
        }

    @staticmethod
    def interpretable_model() -> Dict[str, Any]:
        """Configuration for an interpretable N-BEATS model.

        Returns:
            Configuration dictionary for an interpretable model.
        """
        return {
            'backcast_length': 48,
            'forecast_length': 12,
            'stack_types': ['trend', 'seasonality'],
            'nb_blocks_per_stack': 3,
            'thetas_dim': [4, 8],
            'hidden_layer_units': 256,
            'share_weights_in_stack': False,
        }


def create_compiled_nbeats_model(
        config: Optional[Dict[str, Any]] = None,
        optimizer: Union[str, keras.optimizers.Optimizer] = 'adam',
        loss: Union[str, keras.losses.Loss] = None,
        metrics: Optional[List[Union[str, keras.metrics.Metric]]] = None,
        learning_rate: float = 0.001,
        **kwargs: Any
) -> NBeatsNet:
    """Create a compiled N-BEATS model with sensible defaults.

    Args:
        config: Configuration dictionary for the model. If None, uses medium config.
        optimizer: Optimizer to use for training.
        loss: Loss function to use. If None, uses SMAPE loss.
        metrics: List of metrics to track during training.
        learning_rate: Learning rate for the optimizer.
        **kwargs: Additional arguments passed to NBeatsNet.

    Returns:
        Compiled N-BEATS model ready for training.
    """
    # Use medium config as default
    if config is None:
        config = NBeatsConfig.medium_model()

    # Override config with any provided kwargs
    config.update(kwargs)

    # Create model
    model = NBeatsNet(**config)

    # Setup default loss if not provided
    if loss is None:
        loss = SMAPELoss()

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