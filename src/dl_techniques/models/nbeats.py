"""
N-BEATS Model for Time Series Forecasting.

This module implements the complete N-BEATS architecture with all bug fixes
applied for proper shape handling and gradient flow.
"""

import keras
from keras import ops
from typing import List, Tuple, Optional, Union, Any, Dict

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
    """N-BEATS neural network for time series forecasting.

    N-BEATS (Neural Basis Expansion Analysis for Time Series) is a deep neural
    architecture designed for time series forecasting. It uses a hierarchical
    doubly residual architecture with forward and backward residual links.

    The model consists of several stacks, each containing multiple blocks.
    Each block produces a backcast (reconstruction of the input) and a forecast
    (prediction of future values). The final forecast is the sum of forecasts
    from all blocks.

    Args:
        input_dim: Integer, dimensionality of input time series (default: 1).
        output_dim: Integer, dimensionality of output time series (default: 1).
        backcast_length: Integer, length of the input time series window.
        forecast_length: Integer, length of the forecast horizon.
        stack_types: List of strings, types of stacks to use. Options are
            'generic', 'trend', and 'seasonality'.
        nb_blocks_per_stack: Integer, number of blocks per stack.
        thetas_dim: List of integers, dimensionality of theta parameters for each stack.
        share_weights_in_stack: Boolean, whether to share weights within each stack.
        hidden_layer_units: Integer, number of hidden units in each fully connected layer.
        **kwargs: Additional keyword arguments for the Model parent class.

    Example:
        >>> model = NBeatsNet(
        ...     backcast_length=48,
        ...     forecast_length=12,
        ...     stack_types=['trend', 'seasonality'],
        ...     nb_blocks_per_stack=2,
        ...     thetas_dim=[3, 6]
        ... )
        >>> model.compile(optimizer='adam', loss='mae')
        >>> # Train with data of shape (batch_size, backcast_length, 1)
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

        # Store configuration
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

        # Validate stack types
        valid_stack_types = {self.GENERIC_BLOCK, self.TREND_BLOCK, self.SEASONALITY_BLOCK}
        for stack_type in self.stack_types:
            if stack_type not in valid_stack_types:
                raise ValueError(f"Invalid stack type: {stack_type}. "
                               f"Must be one of: {valid_stack_types}")

        # Initialize components
        self.blocks: List[List[Any]] = []
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

        total_blocks = sum(len(stack) for stack in self.blocks)
        logger.info(f"N-BEATS network built with {total_blocks} total blocks")

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the N-BEATS model.

        Args:
            inputs: Input tensor of shape (batch_size, backcast_length) or
                   (batch_size, backcast_length, 1).
            training: Boolean indicating training mode.

        Returns:
            Forecast tensor of shape (batch_size, forecast_length, output_dim).
        """
        # Handle input shapes - ensure 2D for processing
        if len(inputs.shape) == 3:
            # Remove last dimension if it's 1 (univariate case)
            if inputs.shape[-1] == 1:
                inputs = ops.squeeze(inputs, axis=-1)
            else:
                # For multivariate, take first channel or sum across channels
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

                # Always subtract backcast from residual (standard N-BEATS behavior)
                residual = residual - backcast

                # Add forecast to accumulator
                forecast_sum = forecast_sum + forecast

        # BUG FIX 1: Reshape to 3D to match expected output shape
        forecast = ops.expand_dims(forecast_sum, axis=-1)

        # Apply output projection if needed
        if self.output_projection is not None:
            forecast = self.output_projection(forecast, training=training)

        return forecast

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the model.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Shape of the output tensor.
        """
        batch_size = input_shape[0]
        # BUG FIX 2: Return correct 3D shape including output_dim
        return (batch_size, self.forecast_length, self.output_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Returns:
            Configuration dictionary containing all model parameters.
        """
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
        """Create model from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            Instantiated NBeatsNet model.
        """
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print detailed model summary."""
        logger.info("=" * 60)
        logger.info("N-BEATS Model Summary")
        logger.info("=" * 60)
        logger.info(f"Input dimension: {self.input_dim}")
        logger.info(f"Output dimension: {self.output_dim}")
        logger.info(f"Backcast length: {self.backcast_length}")
        logger.info(f"Forecast length: {self.forecast_length}")
        logger.info(f"Stack types: {self.stack_types}")
        logger.info(f"Blocks per stack: {self.nb_blocks_per_stack}")
        logger.info(f"Theta dimensions: {self.thetas_dim}")
        logger.info(f"Hidden units: {self.hidden_layer_units}")
        logger.info(f"Share weights: {self.share_weights_in_stack}")

        total_blocks = sum(len(stack) for stack in self.blocks)
        logger.info(f"Total blocks: {total_blocks}")

        # Stack breakdown
        for i, (stack_type, theta_dim) in enumerate(zip(self.stack_types, self.thetas_dim)):
            logger.info(f"Stack {i}: {stack_type} ({self.nb_blocks_per_stack} blocks, "
                       f"theta_dim={theta_dim})")

        logger.info("=" * 60)

        # Call parent summary if needed
        super().summary(**kwargs)

# ---------------------------------------------------------------------

def create_nbeats_model(
        config: Optional[Dict[str, Any]] = None,
        optimizer: Union[str, keras.optimizers.Optimizer] = 'adam',
        loss: Union[str, keras.losses.Loss] = 'mae',
        metrics: Optional[List[Union[str, keras.metrics.Metric]]] = None,
        learning_rate: float = 0.001,
        **kwargs: Any
) -> NBeatsNet:
    """Create a compiled N-BEATS model with sensible defaults.

    This utility function creates and compiles an N-BEATS model with commonly
    used configurations for time series forecasting tasks.

    Args:
        config: Configuration dictionary for the model. If None, uses defaults.
        optimizer: Optimizer to use for training. Can be string or optimizer instance.
        loss: Loss function to use. Can be string or loss instance.
        metrics: List of metrics to track during training.
        learning_rate: Learning rate for the optimizer (if optimizer is a string).
        **kwargs: Additional arguments passed to NBeatsNet constructor.

    Returns:
        Compiled N-BEATS model ready for training.

    Example:
        >>> model = create_nbeats_model(
        ...     config={'backcast_length': 96, 'forecast_length': 24},
        ...     optimizer='adamw',
        ...     learning_rate=0.001
        ... )
        >>> model.fit(train_data, validation_data=val_data, epochs=100)
    """
    # Default configuration optimized for common time series tasks
    default_config = {
        'backcast_length': 48,        # Look back 48 time steps
        'forecast_length': 12,        # Forecast 12 time steps ahead
        'stack_types': ['trend', 'seasonality'],  # Capture trend and seasonal patterns
        'nb_blocks_per_stack': 2,     # 2 blocks per stack for efficiency
        'thetas_dim': [3, 6],         # Polynomial degree 3 for trend, 6 harmonics for seasonality
        'hidden_layer_units': 128,    # Moderate capacity
        'share_weights_in_stack': False,  # Independent blocks for flexibility
        'input_dim': 1,               # Univariate by default
        'output_dim': 1,              # Univariate output
    }

    # Update with user config
    if config is not None:
        default_config.update(config)

    # Override with any additional kwargs
    default_config.update(kwargs)

    # Create model
    model = NBeatsNet(**default_config)

    # Setup default metrics if not provided
    if metrics is None:
        metrics = ['mae', 'mse']

    # Setup optimizer with learning rate
    if isinstance(optimizer, str):
        optimizer_name = optimizer.lower()
        if optimizer_name == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'adamw':
            optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            # Fallback to keras.optimizers.get for other optimizers
            optimizer = keras.optimizers.get(optimizer)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    logger.info(f"Created and compiled N-BEATS model:")
    logger.info(f"  - Optimizer: {optimizer.__class__.__name__}")
    logger.info(f"  - Loss: {loss}")
    logger.info(f"  - Learning rate: {learning_rate}")
    logger.info(f"  - Metrics: {metrics}")

    return model


def create_interpretable_nbeats_model(
        backcast_length: int = 48,
        forecast_length: int = 12,
        trend_polynomial_degree: int = 3,
        seasonality_harmonics: int = 6,
        hidden_units: int = 128,
        **kwargs: Any
) -> NBeatsNet:
    """Create an interpretable N-BEATS model with trend and seasonality stacks.

    This is a convenience function for creating N-BEATS models focused on
    interpretability, using only trend and seasonality blocks with
    interpretable basis functions.

    Args:
        backcast_length: Length of input sequence.
        forecast_length: Length of forecast sequence.
        trend_polynomial_degree: Degree of polynomial for trend modeling.
        seasonality_harmonics: Number of Fourier harmonics for seasonality.
        hidden_units: Number of hidden units in each layer.
        **kwargs: Additional arguments for model creation.

    Returns:
        Compiled interpretable N-BEATS model.

    Example:
        >>> model = create_interpretable_nbeats_model(
        ...     backcast_length=96,
        ...     forecast_length=24,
        ...     trend_polynomial_degree=4,
        ...     seasonality_harmonics=8
        ... )
    """
    config = {
        'backcast_length': backcast_length,
        'forecast_length': forecast_length,
        'stack_types': ['trend', 'seasonality'],
        'nb_blocks_per_stack': 2,
        'thetas_dim': [trend_polynomial_degree + 1, seasonality_harmonics * 2],
        'hidden_layer_units': hidden_units,
        'share_weights_in_stack': False,
        'input_dim': 1,
        'output_dim': 1,
    }

    return create_nbeats_model(config=config, **kwargs)

# ---------------------------------------------------------------------

