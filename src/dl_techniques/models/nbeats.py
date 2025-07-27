"""
N-BEATS Model for Time Series Forecasting.

This module implements the complete N-BEATS (Neural Basis Expansion Analysis for
Time Series) architecture, a state-of-the-art deep learning model for univariate
time series forecasting that achieves excellent performance while maintaining
interpretability through its hierarchical doubly residual architecture.

Key Features
============

- **Hierarchical Architecture**: Multiple stacks capturing different temporal patterns
- **Doubly Residual Connections**: Both forecast and backcast residual links
- **Interpretable Basis Functions**: Polynomial (trend) and Fourier (seasonality)
- **Flexible Configuration**: Support for custom stack compositions
- **Production Ready**: Comprehensive validation, serialization, and error handling

Architecture Overview
====================

The N-BEATS model consists of:

1. **Input Processing**: Handles various input shapes and validates compatibility
2. **Stack Hierarchy**: Sequential processing through specialized stacks:
   - Trend Stack: Captures long-term growth/decline patterns
   - Seasonality Stack: Captures periodic/cyclical patterns
   - Generic Stack: Captures complex non-linear patterns
3. **Residual Processing**: Each block produces backcast (removed from input) and forecast (accumulated)
4. **Output Projection**: Optional transformation for multi-dimensional outputs

Mathematical Foundation
======================

For each block b in stack s:
- Input: x_b (residual from previous block)
- Processing: h = FC_stack(x_b)  # 4 fully connected layers
- Parameters: θ_backcast, θ_forecast = Linear(h)
- Basis expansion:
  - backcast_b = Σ θ_backcast_i × B_s_i(t_past)
  - forecast_b = Σ θ_forecast_i × B_s_i(t_future)
- Residual update: x_{b+1} = x_b - backcast_b
- Forecast accumulation: y += forecast_b

Final prediction: y = Σ_{all blocks} forecast_b

Performance Characteristics
==========================

- **Memory**: O(B × L × H) where B=blocks, L=sequence_length, H=hidden_units
- **Computation**: O(B × L × H²) per forward pass
- **Training**: Typically 50-200 epochs for convergence
- **Inference**: Fast, single forward pass

Best Practices
==============

1. **Data Preprocessing**:
   - Normalize/standardize inputs
   - Handle missing values before training
   - Consider differencing for non-stationary series

2. **Model Configuration**:
   - Start with interpretable stacks (trend + seasonality)
   - Add generic stacks for complex patterns
   - Use 2-4 blocks per stack typically

3. **Training**:
   - Use early stopping on validation loss
   - Learning rate scheduling often helps
   - Gradient clipping for stability

4. **Hyperparameter Tuning**:
   - Backcast length: 2-10x forecast length
   - Hidden units: 64-512 depending on data complexity
   - Stack composition based on data characteristics

Usage Examples
==============

Basic Usage:
```python
# Simple trend + seasonality model
model = create_nbeats_model(
    backcast_length=168,  # 1 week of hourly data
    forecast_length=24,   # 1 day forecast
    stack_types=['trend', 'seasonality'],
    thetas_dim=[4, 10]    # 4th order polynomial, 5 harmonics
)

# Train the model
model.fit(train_data, validation_data=val_data, epochs=100)

# Generate forecasts
forecasts = model.predict(test_data)
```

Advanced Configuration:
```python
# High-capacity model with regularization
model = NBeatsNet(
    backcast_length=336,  # 2 weeks
    forecast_length=48,   # 2 days
    stack_types=['trend', 'seasonality', 'generic'],
    nb_blocks_per_stack=3,
    thetas_dim=[6, 20, 32],
    hidden_layer_units=256,
    kernel_regularizer=keras.regularizers.L2(1e-4),
    dropout_rate=0.1,
    use_batch_norm=True
)
```

Interpretability:
```python
# Extract interpretable components
model = create_interpretable_nbeats_model()
trend_components, seasonal_components = model.decompose_forecast(data)
```
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
    architecture designed for univariate time series forecasting that achieves
    state-of-the-art performance while maintaining interpretability through
    hierarchical doubly residual architecture.

    The model consists of several stacks, each containing multiple blocks.
    Each block produces a backcast (reconstruction of the input) and a forecast
    (prediction of future values). The final forecast is the sum of forecasts
    from all blocks, while backcasts are subtracted residually from the input.

    Args:
        backcast_length: Integer, length of the input time series window.
            Must be positive.
        forecast_length: Integer, length of the forecast horizon.
            Must be positive.
        stack_types: List of strings, types of stacks to use. Options are
            'generic', 'trend', and 'seasonality'. Default: ['trend', 'seasonality'].
        nb_blocks_per_stack: Integer, number of blocks per stack.
            Must be positive. Default: 3.
        thetas_dim: List of integers, dimensionality of theta parameters for each stack.
            Length must match stack_types. Default: [4, 8].
        hidden_layer_units: Integer, number of hidden units in each fully connected layer.
            Must be positive. Default: 256.
        share_weights_in_stack: Boolean, whether to share weights within each stack.
            Default: False.
        kernel_regularizer: Optional regularizer for block weights.
        theta_regularizer: Optional regularizer for theta parameters.
        dropout_rate: Float, dropout rate for regularization. Default: 0.0.
        use_batch_norm: Boolean, whether to use batch normalization. Default: False.
        activation: String or callable, activation function for hidden layers. Default: 'relu'.
        kernel_initializer: String or Initializer, initializer for weights. Default: 'he_normal'.
        input_dim: Integer, dimensionality of input features. Default: 1.
        output_dim: Integer, dimensionality of output features. Default: 1.
        **kwargs: Additional keyword arguments for the Model parent class.

    Raises:
        ValueError: If configuration parameters are invalid.

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
    VALID_STACK_TYPES = {GENERIC_BLOCK, TREND_BLOCK, SEASONALITY_BLOCK}

    def __init__(
            self,
            backcast_length: int,
            forecast_length: int,
            stack_types: List[str] = ['trend', 'seasonality'],
            nb_blocks_per_stack: int = 3,
            thetas_dim: List[int] = [4, 8],
            hidden_layer_units: int = 256,
            share_weights_in_stack: bool = False,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            theta_regularizer: Optional[keras.regularizers.Regularizer] = None,
            dropout_rate: float = 0.0,
            use_batch_norm: bool = False,
            activation: Union[str, callable] = 'relu',
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
            input_dim: int = 1,
            output_dim: int = 1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        self._validate_configuration(
            backcast_length, forecast_length, stack_types,
            nb_blocks_per_stack, thetas_dim, hidden_layer_units,
            dropout_rate, input_dim, output_dim
        )

        # Store configuration
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.stack_types = stack_types.copy()
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = thetas_dim.copy()
        self.hidden_layer_units = hidden_layer_units
        self.share_weights_in_stack = share_weights_in_stack
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.theta_regularizer = keras.regularizers.get(theta_regularizer)
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Components to be built
        self.input_layer = None
        self.blocks: List[List[Union[GenericBlock, TrendBlock, SeasonalityBlock]]] = []
        self.output_projection = None
        self.dropout_layers: List[keras.layers.Dropout] = []

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

        Args:
            backcast_length: Input sequence length.
            forecast_length: Output sequence length.
            stack_types: List of stack type names.
            nb_blocks_per_stack: Number of blocks per stack.
            thetas_dim: Theta dimensions for each stack.
            hidden_layer_units: Hidden layer size.
            dropout_rate: Dropout probability.
            input_dim: Input feature dimension.
            output_dim: Output feature dimension.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if backcast_length <= 0:
            raise ValueError(f"backcast_length must be positive, got {backcast_length}")
        if forecast_length <= 0:
            raise ValueError(f"forecast_length must be positive, got {forecast_length}")
        if nb_blocks_per_stack <= 0:
            raise ValueError(f"nb_blocks_per_stack must be positive, got {nb_blocks_per_stack}")
        if hidden_layer_units <= 0:
            raise ValueError(f"hidden_layer_units must be positive, got {hidden_layer_units}")
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

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the N-BEATS network components.

        Args:
            input_shape: Shape of input tensor.
        """
        logger.info(f"Building N-BEATS network with input shape: {input_shape}")

        # Validate and normalize input shape
        if len(input_shape) == 2:
            # (batch_size, backcast_length) -> add feature dimension
            batch_size, seq_len = input_shape
            if seq_len != self.backcast_length:
                raise ValueError(
                    f"Input sequence length {seq_len} doesn't match "
                    f"configured backcast_length {self.backcast_length}"
                )
            normalized_shape = (batch_size, seq_len, self.input_dim)
        elif len(input_shape) == 3:
            # (batch_size, backcast_length, features)
            batch_size, seq_len, features = input_shape
            if seq_len != self.backcast_length:
                raise ValueError(
                    f"Input sequence length {seq_len} doesn't match "
                    f"configured backcast_length {self.backcast_length}"
                )
            if features != self.input_dim:
                raise ValueError(
                    f"Input feature dimension {features} doesn't match "
                    f"configured input_dim {self.input_dim}"
                )
            normalized_shape = input_shape
        else:
            raise ValueError(
                f"Input must be 2D (batch, sequence) or 3D (batch, sequence, features), "
                f"got {len(input_shape)}D: {input_shape}"
            )

        # Create input processing layer
        self.input_layer = keras.layers.Flatten(name='input_flatten')

        # Build blocks for each stack
        self._build_stacks()

        # Output projection if needed
        if self.input_dim != self.output_dim:
            self.output_projection = keras.layers.Dense(
                self.output_dim,
                activation='linear',
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='output_projection'
            )

        super().build(input_shape)

    def _build_stacks(self) -> None:
        """Build all stacks and their constituent blocks."""
        logger.info(f"Building {len(self.stack_types)} stacks")

        for stack_id, (stack_type, theta_dim) in enumerate(zip(self.stack_types, self.thetas_dim)):
            stack_blocks = []

            for block_id in range(self.nb_blocks_per_stack):
                block_name = f"stack_{stack_id}_block_{block_id}_{stack_type}"

                # Common block parameters
                block_kwargs = {
                    'units': self.hidden_layer_units,
                    'thetas_dim': theta_dim,
                    'backcast_length': self.backcast_length,
                    'forecast_length': self.forecast_length,
                    'share_weights': self.share_weights_in_stack,
                    'activation': self.activation,
                    'kernel_initializer': self.kernel_initializer,
                    'kernel_regularizer': self.kernel_regularizer,
                    'theta_regularizer': self.theta_regularizer,
                    'name': block_name
                }

                # Create appropriate block type
                if stack_type == self.GENERIC_BLOCK:
                    block = GenericBlock(**block_kwargs)
                elif stack_type == self.TREND_BLOCK:
                    block = TrendBlock(**block_kwargs)
                elif stack_type == self.SEASONALITY_BLOCK:
                    block = SeasonalityBlock(**block_kwargs)
                else:
                    raise ValueError(f"Unknown stack type: {stack_type}")

                stack_blocks.append(block)

                # Add dropout layer if specified
                if self.dropout_rate > 0.0:
                    dropout_layer = keras.layers.Dropout(
                        self.dropout_rate,
                        name=f"dropout_{block_name}"
                    )
                    self.dropout_layers.append(dropout_layer)

            self.blocks.append(stack_blocks)

        total_blocks = sum(len(stack) for stack in self.blocks)
        logger.info(f"Built {total_blocks} total blocks across {len(self.blocks)} stacks")

    def call(self, inputs, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the N-BEATS model.

        Args:
            inputs: Input tensor of shape (batch_size, backcast_length) or
                   (batch_size, backcast_length, input_dim).
            training: Boolean indicating training mode.

        Returns:
            Forecast tensor of shape (batch_size, forecast_length, output_dim).
        """
        # Normalize input shape to 2D for block processing
        if len(inputs.shape) == 3:
            # For multivariate inputs, take appropriate channel(s)
            if inputs.shape[-1] == self.input_dim:
                if self.input_dim == 1:
                    # Squeeze out feature dimension for univariate case
                    processed_input = ops.squeeze(inputs, axis=-1)
                else:
                    # Take first channel or apply reduction for multivariate
                    processed_input = inputs[..., 0]
            else:
                raise ValueError(
                    f"Input feature dimension {inputs.shape[-1]} doesn't match "
                    f"expected input_dim {self.input_dim}"
                )
        elif len(inputs.shape) == 2:
            processed_input = inputs
        else:
            raise ValueError(f"Invalid input shape: {inputs.shape}")

        batch_size = ops.shape(processed_input)[0]

        # Initialize residual and forecast accumulator
        residual = processed_input  # Shape: (batch_size, backcast_length)
        forecast_sum = ops.zeros((batch_size, self.forecast_length))

        # Process through all stacks and blocks
        dropout_idx = 0
        for stack_id, stack_blocks in enumerate(self.blocks):
            for block_id, block in enumerate(stack_blocks):
                # Forward pass through block
                backcast, forecast = block(residual, training=training)

                # Apply dropout if configured
                if self.dropout_rate > 0.0 and dropout_idx < len(self.dropout_layers):
                    forecast = self.dropout_layers[dropout_idx](forecast, training=training)
                    dropout_idx += 1

                # Update residual (subtract backcast)
                residual = residual - backcast

                # Accumulate forecast
                forecast_sum = forecast_sum + forecast

        # Reshape to 3D output format: (batch_size, forecast_length, output_dim)
        forecast_output = ops.expand_dims(forecast_sum, axis=-1)

        # Apply output projection if needed
        if self.output_projection is not None:
            forecast_output = self.output_projection(forecast_output, training=training)

        return forecast_output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the model.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Shape of the output tensor.
        """
        batch_size = input_shape[0]
        return (batch_size, self.forecast_length, self.output_dim)

    def decompose_forecast(
        self,
        inputs,
        training: Optional[bool] = None
    ) -> Dict[str, List[keras.KerasTensor]]:
        """Decompose forecast into interpretable components from each stack.

        This method provides interpretability by returning the individual
        contributions from each stack type (trend, seasonality, generic).

        Args:
            inputs: Input tensor for forecasting.
            training: Boolean indicating training mode.

        Returns:
            Dictionary mapping stack types to lists of forecast contributions.

        Example:
            >>> components = model.decompose_forecast(test_data)
            >>> trend_components = components['trend']
            >>> seasonal_components = components['seasonality']
        """
        # Normalize input
        if len(inputs.shape) == 3 and inputs.shape[-1] == 1:
            processed_input = ops.squeeze(inputs, axis=-1)
        elif len(inputs.shape) == 2:
            processed_input = inputs
        else:
            raise ValueError(f"Invalid input shape for decomposition: {inputs.shape}")

        batch_size = ops.shape(processed_input)[0]
        residual = processed_input

        decomposition = {stack_type: [] for stack_type in set(self.stack_types)}

        # Process through stacks, collecting forecasts by type
        for stack_id, (stack_type, stack_blocks) in enumerate(zip(self.stack_types, self.blocks)):
            stack_forecasts = []

            for block in stack_blocks:
                backcast, forecast = block(residual, training=training)
                residual = residual - backcast
                stack_forecasts.append(forecast)

            decomposition[stack_type].extend(stack_forecasts)

        return decomposition

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Returns:
            Configuration dictionary containing all model parameters.
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
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'theta_regularizer': keras.regularizers.serialize(self.theta_regularizer),
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'activation': self.activation,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
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
        """Print detailed model summary with architecture breakdown."""
        logger.info("=" * 60)
        logger.info("N-BEATS Model Summary")
        logger.info("=" * 60)
        logger.info(f"Input dimension: {self.input_dim}")
        logger.info(f"Output dimension: {self.output_dim}")
        logger.info(f"Backcast length: {self.backcast_length}")
        logger.info(f"Forecast length: {self.forecast_length}")
        logger.info(f"Hidden units: {self.hidden_layer_units}")
        logger.info(f"Dropout rate: {self.dropout_rate}")
        logger.info(f"Share weights: {self.share_weights_in_stack}")

        total_blocks = sum(len(stack) for stack in self.blocks)
        logger.info(f"Total blocks: {total_blocks}")

        # Stack-by-stack breakdown
        for i, (stack_type, theta_dim) in enumerate(zip(self.stack_types, self.thetas_dim)):
            logger.info(f"Stack {i}: {stack_type} ({self.nb_blocks_per_stack} blocks, "
                       f"theta_dim={theta_dim})")

        # Regularization info
        if self.kernel_regularizer is not None:
            logger.info(f"Kernel regularizer: {self.kernel_regularizer.__class__.__name__}")
        if self.theta_regularizer is not None:
            logger.info(f"Theta regularizer: {self.theta_regularizer.__class__.__name__}")

        logger.info("=" * 60)

        # Call parent summary
        super().summary(**kwargs)

# ---------------------------------------------------------------------

def create_nbeats_model(
        backcast_length: int = 48,
        forecast_length: int = 12,
        stack_types: List[str] = ['trend', 'seasonality'],
        nb_blocks_per_stack: int = 2,
        thetas_dim: Optional[List[int]] = None,
        hidden_layer_units: int = 128,
        optimizer: Union[str, keras.optimizers.Optimizer] = 'adam',
        loss: Union[str, keras.losses.Loss] = 'mae',
        metrics: Optional[List[Union[str, keras.metrics.Metric]]] = None,
        learning_rate: float = 0.001,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        dropout_rate: float = 0.0,
        **kwargs: Any
) -> NBeatsNet:
    """Create a compiled N-BEATS model with sensible defaults.

    This utility function creates and compiles an N-BEATS model with commonly
    used configurations for time series forecasting tasks.

    Args:
        backcast_length: Length of input sequence. Default: 48.
        forecast_length: Length of forecast sequence. Default: 12.
        stack_types: Types of stacks to use. Default: ['trend', 'seasonality'].
        nb_blocks_per_stack: Number of blocks per stack. Default: 2.
        thetas_dim: Theta dimensions for each stack. If None, uses sensible defaults.
        hidden_layer_units: Hidden units in each layer. Default: 128.
        optimizer: Optimizer for training. Default: 'adam'.
        loss: Loss function. Default: 'mae'.
        metrics: List of metrics to track. Default: ['mae', 'mse'].
        learning_rate: Learning rate for optimizer. Default: 0.001.
        kernel_regularizer: Weight regularizer. Default: None.
        dropout_rate: Dropout rate for regularization. Default: 0.0.
        **kwargs: Additional arguments for NBeatsNet constructor.

    Returns:
        Compiled N-BEATS model ready for training.

    Example:
        >>> model = create_nbeats_model(
        ...     backcast_length=96,
        ...     forecast_length=24,
        ...     stack_types=['trend', 'seasonality', 'generic'],
        ...     dropout_rate=0.1,
        ...     learning_rate=0.001
        ... )
        >>> model.fit(train_data, validation_data=val_data, epochs=100)
    """
    # Set sensible defaults for theta dimensions based on stack types
    if thetas_dim is None:
        thetas_dim = []
        for stack_type in stack_types:
            if stack_type == 'trend':
                thetas_dim.append(4)  # 4th order polynomial
            elif stack_type == 'seasonality':
                thetas_dim.append(8)  # 4 harmonics (8 sin/cos terms)
            elif stack_type == 'generic':
                thetas_dim.append(16)  # Flexible representation
            else:
                thetas_dim.append(8)  # Default fallback

    # Create model
    model = NBeatsNet(
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        stack_types=stack_types,
        nb_blocks_per_stack=nb_blocks_per_stack,
        thetas_dim=thetas_dim,
        hidden_layer_units=hidden_layer_units,
        kernel_regularizer=kernel_regularizer,
        dropout_rate=dropout_rate,
        **kwargs
    )

    # Setup default metrics
    if metrics is None:
        metrics = ['mae', 'mse']

    # Setup optimizer with learning rate
    if isinstance(optimizer, str):
        optimizer_map = {
            'adam': keras.optimizers.Adam,
            'adamw': keras.optimizers.AdamW,
            'rmsprop': keras.optimizers.RMSprop,
            'sgd': keras.optimizers.SGD,
        }
        optimizer_cls = optimizer_map.get(optimizer.lower())
        if optimizer_cls:
            optimizer = optimizer_cls(learning_rate=learning_rate)
        else:
            optimizer = keras.optimizers.get(optimizer)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    logger.info(f"Created N-BEATS model:")
    logger.info(f"  - Architecture: {len(stack_types)} stacks, {nb_blocks_per_stack} blocks each")
    logger.info(f"  - Sequence: {backcast_length} → {forecast_length}")
    logger.info(f"  - Optimizer: {optimizer.__class__.__name__} (lr={learning_rate})")
    logger.info(f"  - Loss: {loss}")
    logger.info(f"  - Regularization: dropout={dropout_rate}")

    return model


def create_interpretable_nbeats_model(
        backcast_length: int = 48,
        forecast_length: int = 12,
        trend_polynomial_degree: int = 3,
        seasonality_harmonics: int = 4,
        hidden_units: int = 128,
        **kwargs: Any
) -> NBeatsNet:
    """Create an interpretable N-BEATS model with trend and seasonality stacks.

    This convenience function creates N-BEATS models focused on interpretability,
    using only trend and seasonality blocks with interpretable basis functions.
    The resulting model can decompose forecasts into trend and seasonal components.

    Args:
        backcast_length: Length of input sequence. Default: 48.
        forecast_length: Length of forecast sequence. Default: 12.
        trend_polynomial_degree: Degree of polynomial for trend modeling. Default: 3.
        seasonality_harmonics: Number of Fourier harmonics for seasonality. Default: 4.
        hidden_units: Number of hidden units in each layer. Default: 128.
        **kwargs: Additional arguments for model creation.

    Returns:
        Compiled interpretable N-BEATS model.

    Example:
        >>> model = create_interpretable_nbeats_model(
        ...     backcast_length=168,  # 1 week hourly
        ...     forecast_length=24,   # 1 day
        ...     trend_polynomial_degree=4,
        ...     seasonality_harmonics=6
        ... )
        >>>
        >>> # Train model
        >>> model.fit(train_data, validation_data=val_data)
        >>>
        >>> # Decompose forecasts for interpretability
        >>> components = model.decompose_forecast(test_data)
        >>> trend_forecasts = components['trend']
        >>> seasonal_forecasts = components['seasonality']
    """
    return create_nbeats_model(
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        stack_types=['trend', 'seasonality'],
        nb_blocks_per_stack=2,  # Fewer blocks for interpretability
        thetas_dim=[trend_polynomial_degree + 1, seasonality_harmonics * 2],
        hidden_layer_units=hidden_units,
        **kwargs
    )

# ---------------------------------------------------------------------