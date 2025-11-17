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
from dl_techniques.layers.time_series.revin import RevIN

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NBeatsNet(keras.Model):
    """
    Neural Basis Expansion Analysis for Time Series (N-BEATS) forecasting model.

    This implementation follows modern Keras 3 patterns and includes proper RevIN
    normalization that stores statistics internally for improved forecasting performance.
    The model uses specialized blocks (Generic, Trend, Seasonality) that implement
    the mathematical formulations specific to the N-BEATS architecture.

    **Intent**: Provide a production-ready N-BEATS implementation for time series
    forecasting with proper residual connections, normalization, and serialization
    support following modern Keras 3 best practices.

    **Architecture**:
    ```
    Input(shape=[batch, backcast_length, features])
           ↓
    RevIN Normalization (optional)
           ↓
    Stack 1: [Block₁, Block₂, ..., Blockₙ] → (backcast₁, forecast₁)
           ↓ (residual = input - backcast₁)
    Stack 2: [Block₁, Block₂, ..., Blockₙ] → (backcast₂, forecast₂)
           ↓ (residual = residual - backcast₂)
           ...
           ↓
    Forecast Sum = Σ(forecasts)
           ↓
    RevIN Denormalization (optional)
           ↓
    Output Projection (if input_dim != output_dim)
           ↓
    Output(shape=[batch, forecast_length, output_features])
    ```

    **Mathematical Foundation**:
    The model learns to decompose the time series into interpretable components:
    - Trend: Polynomial basis functions for long-term patterns
    - Seasonality: Fourier basis functions for periodic patterns
    - Generic: Learnable basis functions for complex patterns

    Each block produces a backcast (reconstruction) and forecast, with residual
    connections ensuring proper signal decomposition across the stack hierarchy.

    Args:
        backcast_length: Integer, length of the input time series window.
            This determines how much historical context the model uses.
            Should typically be 3-7x the forecast_length for optimal performance.
        forecast_length: Integer, length of the forecast horizon.
            The number of time steps to predict into the future.
        stack_types: List of strings, types of stacks to use in the model.
            Each string must be one of: 'generic', 'trend', 'seasonality'.
            Default: ['trend', 'seasonality'] for interpretable forecasting.
        nb_blocks_per_stack: Integer, number of blocks per stack.
            More blocks can capture more complex patterns but may overfit.
            Defaults to 3 blocks per stack.
        thetas_dim: List of integers, dimensionality of theta parameters for each stack.
            Must have same length as stack_types. Each value determines the complexity
            of the corresponding stack's basis functions. Defaults to [4, 8].
        hidden_layer_units: Integer, number of hidden units in each block's layers.
            Controls model capacity and computational cost. Defaults to 256.
        share_weights_in_stack: Boolean, whether to share weights within each stack.
            Weight sharing reduces parameters but may limit expressiveness.
            Defaults to False.
        use_revin: Boolean, whether to use RevIN normalization.
            RevIN typically improves performance by 10-20% on real datasets.
            Defaults to True.
        normalization_type: String, type of normalization for internal layers.
            Uses the normalization factory for consistency. Defaults to 'layer_norm'.
        kernel_regularizer: Optional regularizer for block weights.
            Helps prevent overfitting in complex models.
        theta_regularizer: Optional regularizer for theta parameters.
            Regularizes the basis function coefficients.
        dropout_rate: Float, dropout rate for regularization.
            Applied to forecasts between blocks. Must be in [0, 1).
        activation: String or callable, activation function for hidden layers.
            Defaults to 'silu' for smooth gradients.
        kernel_initializer: String or Initializer, initializer for layer weights.
            Defaults to 'he_normal' for ReLU-family activations.
        input_dim: Integer, dimensionality of input features.
            Number of variables in multivariate time series. Defaults to 1.
        output_dim: Integer, dimensionality of output features.
            Number of variables to forecast. Defaults to 1.
        use_bias: Boolean, whether to use bias terms in linear layers.
            Defaults to True for better fitting capacity.
        **kwargs: Additional keyword arguments for the Model base class.

    Input shape:
        3D tensor with shape: `(batch_size, backcast_length, input_dim)`.
        Can also accept 2D tensor `(batch_size, backcast_length)` for univariate.

    Output shape:
        3D tensor with shape: `(batch_size, forecast_length, output_dim)`.

    Attributes:
        All constructor parameters are stored as instance attributes.
        blocks: List of lists containing the created N-BEATS blocks.
        global_revin: RevIN normalization layer (if use_revin=True).
        output_projection: Dense layer for dimension projection (if needed).
        dropout_layers: List of dropout layers for regularization.

    Example:
        ```python
        # Univariate forecasting model
        model = NBeatsNet(
            backcast_length=96,  # 4 days of hourly data
            forecast_length=24,  # 1 day forecast
            stack_types=['trend', 'seasonality'],
            nb_blocks_per_stack=3,
            thetas_dim=[4, 8],  # 3rd order polynomial, 4 harmonics
            use_revin=True
        )

        # Multivariate model with custom configuration
        model = NBeatsNet(
            backcast_length=168,
            forecast_length=24,
            stack_types=['trend', 'seasonality', 'generic'],
            thetas_dim=[6, 12, 16],
            input_dim=5,    # 5 input variables
            output_dim=3,   # Forecast 3 variables
            hidden_layer_units=512,
            dropout_rate=0.1,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )
        ```

    References:
        - Oreshkin et al. "N-BEATS: Neural basis expansion analysis for interpretable
          time series forecasting" ICLR 2020.
        - Kim et al. "Reversible Instance Normalization for Accurate Time-Series
          Forecasting against Distribution Shift" ICLR 2022.

    Note:
        This implementation uses stateful RevIN that stores normalization statistics
        internally, ensuring consistent normalization across training and inference
        without requiring external state management.
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
            stack_types: List[str] = ['trend', 'seasonality'],
            nb_blocks_per_stack: int = 3,
            thetas_dim: List[int] = [4, 8],
            hidden_layer_units: int = 256,
            share_weights_in_stack: bool = False,
            use_revin: bool = True,
            normalization_type: str = 'layer_norm',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            theta_regularizer: Optional[regularizers.Regularizer] = None,
            dropout_rate: float = 0.0,
            activation: Union[str, Callable] = 'silu',
            kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
            input_dim: int = 1,
            output_dim: int = 1,
            use_bias: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate configuration
        self._validate_configuration(
            backcast_length, forecast_length, stack_types, nb_blocks_per_stack,
            thetas_dim, hidden_layer_units, dropout_rate, input_dim, output_dim
        )

        # Store ALL configuration parameters for serialization
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.stack_types = list(stack_types)  # Create copy
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = list(thetas_dim)  # Create copy
        self.hidden_layer_units = hidden_layer_units
        self.share_weights_in_stack = share_weights_in_stack
        self.use_revin = use_revin
        self.normalization_type = normalization_type
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.theta_regularizer = regularizers.get(theta_regularizer)
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        # CREATE sub-layers in __init__ (unbuilt)
        # RevIN layer for normalization
        if self.use_revin:
            self.global_revin = RevIN(
                num_features=self.input_dim,
                affine=False,
                name='global_revin'
            )
        else:
            self.global_revin = None

        # Output projection layer if input/output dims differ
        if self.input_dim != self.output_dim:
            self.output_projection = layers.Dense(
                self.output_dim,
                activation='linear',
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='output_projection'
            )
        else:
            self.output_projection = None

        # Create block stacks
        self.blocks: List[List[Union[GenericBlock, TrendBlock, SeasonalityBlock]]] = []
        self.dropout_layers: List[layers.Dropout] = []

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
        """Validate model configuration parameters."""
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

        # Performance warning
        ratio = backcast_length / forecast_length
        if ratio < 3.0:
            logger.warning(
                f"backcast_length ({backcast_length}) / forecast_length ({forecast_length}) "
                f"= {ratio:.1f}. For optimal performance, use ratio >= 3.0"
            )

    def _create_block_stacks(self) -> None:
        """Create all N-BEATS block stacks."""
        dropout_counter = 0

        # The core logic for multivariate is to have all blocks operate on the flattened vector.
        # This means they all need the flattened lengths.
        block_backcast_len = self.backcast_length * self.input_dim
        # Forecast is generated in input_dim space to allow for RevIN denormalization,
        # before being projected to output_dim.
        block_forecast_len = self.forecast_length * self.input_dim

        for stack_id, (stack_type, theta_dim) in enumerate(zip(self.stack_types, self.thetas_dim)):
            stack_blocks = []

            for block_id in range(self.nb_blocks_per_stack):
                block_name = f"stack_{stack_id}_block_{block_id}_{stack_type}"

                # Common block parameters
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

                # Create appropriate block type with consistent arguments
                if stack_type == self.GENERIC_BLOCK:
                    block = GenericBlock(**block_kwargs)
                elif stack_type == self.TREND_BLOCK:
                    block = TrendBlock(**block_kwargs)
                elif stack_type == self.SEASONALITY_BLOCK:
                    block = SeasonalityBlock(**block_kwargs)
                else:
                    raise ValueError(f"Unknown stack type: {stack_type}")

                stack_blocks.append(block)

                # Add dropout layer if configured
                if self.dropout_rate > 0.0:
                    dropout_layer = layers.Dropout(
                        self.dropout_rate,
                        name=f"dropout_{dropout_counter}"
                    )
                    self.dropout_layers.append(dropout_layer)
                    dropout_counter += 1

            self.blocks.append(stack_blocks)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the model and all its sub-layers."""
        logger.info(f"Building N-BEATS network with input shape: {input_shape}")

        # Validate input shape
        if len(input_shape) == 2:
            batch_size, seq_len = input_shape
            if seq_len != self.backcast_length:
                raise ValueError(
                    f"Input sequence length {seq_len} doesn't match "
                    f"backcast_length {self.backcast_length}"
                )
        elif len(input_shape) == 3:
            batch_size, seq_len, features = input_shape
            if seq_len != self.backcast_length:
                raise ValueError(
                    f"Input sequence length {seq_len} doesn't match "
                    f"backcast_length {self.backcast_length}"
                )
            if features != self.input_dim:
                raise ValueError(
                    f"Input feature dimension {features} doesn't match "
                    f"input_dim {self.input_dim}"
                )
        else:
            raise ValueError(
                f"Input must be 2D or 3D, got {len(input_shape)}D: {input_shape}"
            )

        # Build RevIN layer if used
        if self.global_revin is not None:
            # RevIN expects 3D input
            if len(input_shape) == 2:
                revin_input_shape = (input_shape[0], input_shape[1], 1)
            else:
                revin_input_shape = input_shape
            self.global_revin.build(revin_input_shape)

        # Determine shape for block processing
        block_input_shape = (input_shape[0], self.backcast_length * self.input_dim)

        # Build all blocks
        for stack_blocks in self.blocks:
            for block in stack_blocks:
                block.build(block_input_shape)

        # Build dropout layers
        # Dropout is applied on the flattened forecast vector in input_dim space
        forecast_shape = (input_shape[0], self.forecast_length * self.input_dim)
        for dropout_layer in self.dropout_layers:
            dropout_layer.build(forecast_shape)

        # Build output projection if needed
        if self.output_projection is not None:
            # Projection is applied on the 3D forecast in input_dim space
            projection_input_shape = (input_shape[0], self.forecast_length, self.input_dim)
            self.output_projection.build(projection_input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the N-BEATS network."""

        # Ensure input is 3D for RevIN processing
        if len(inputs.shape) == 2:
            inputs_3d = ops.expand_dims(inputs, axis=-1)
        else:
            inputs_3d = inputs

        batch_size = ops.shape(inputs_3d)[0]

        # Apply RevIN normalization
        if self.global_revin is not None:
            normalized_input = self.global_revin(inputs_3d, training=training)
        else:
            normalized_input = inputs_3d

        # Convert to flattened format expected by blocks
        processed_input = ops.reshape(
            normalized_input,
            (batch_size, self.backcast_length * self.input_dim)
        )

        # Initialize residual and forecast accumulator
        residual = processed_input
        # Forecast sum is accumulated in the flattened input_dim space
        forecast_sum = ops.zeros((batch_size, self.forecast_length * self.input_dim))

        # Process through all blocks with proper residual connections
        dropout_idx = 0
        for stack_blocks in self.blocks:
            for block in stack_blocks:
                # Forward pass through block
                backcast, forecast = block(residual, training=training)

                # Apply dropout to forecast if configured
                if (self.dropout_rate > 0.0 and
                        dropout_idx < len(self.dropout_layers)):
                    forecast = self.dropout_layers[dropout_idx](
                        forecast, training=training
                    )
                    dropout_idx += 1

                # Update residual (subtract backcast) and accumulate forecast
                residual = residual - backcast
                forecast_sum = forecast_sum + forecast

        # Convert forecast back to 3D in input_dim space for denormalization
        forecast_3d = ops.reshape(
            forecast_sum,
            (batch_size, self.forecast_length, self.input_dim)
        )

        # Apply RevIN denormalization
        if self.global_revin is not None:
            denormalized_forecast = self.global_revin.denormalize(forecast_3d)
        else:
            denormalized_forecast = forecast_3d

        # Apply output projection if needed (maps from input_dim to output_dim)
        if self.output_projection is not None:
            final_forecast = self.output_projection(
                denormalized_forecast, training=training
            )
        else:
            final_forecast = denormalized_forecast

        return final_forecast

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape of the model."""
        batch_size = input_shape[0]
        return (batch_size, self.forecast_length, self.output_dim)

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
            'use_revin': self.use_revin,
            'normalization_type': self.normalization_type,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'theta_regularizer': regularizers.serialize(self.theta_regularizer),
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'use_bias': self.use_bias,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NBeatsNet':
        """Create model instance from configuration."""
        # Deserialize regularizers and initializers
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
        use_revin: bool = True,
        optimizer: Union[str, keras.optimizers.Optimizer] = 'adam',
        loss: Union[str, keras.losses.Loss] = 'mae',
        metrics: Optional[List[Union[str, keras.metrics.Metric]]] = None,
        learning_rate: float = 1e-4,
        gradient_clip_norm: float = 1.0,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        dropout_rate: float = 0.0,
        **kwargs: Any
) -> NBeatsNet:
    """
    Create and compile an N-BEATS model with optimal defaults and configuration.

    This factory function provides a convenient way to create N-BEATS models with
    sensible defaults, automatic theta dimension calculation, and proper compilation
    with gradient clipping for training stability.

    **Intent**: Simplify N-BEATS model creation while ensuring optimal hyperparameter
    defaults, proper compilation, and performance optimizations based on research
    best practices.

    **Key Features**:
    - Automatic theta dimension calculation based on stack types
    - Gradient clipping for training stability (essential for N-BEATS)
    - Performance warnings and recommendations
    - Sensible defaults optimized for time series forecasting
    - RevIN normalization enabled by default for improved performance

    Args:
        backcast_length: Length of input sequence. Default: 96 (4x forecast).
            Should typically be 3-7x forecast_length for optimal performance.
        forecast_length: Length of forecast sequence. Default: 24.
        stack_types: Types of stacks to use. Default: ['trend', 'seasonality'].
            Each must be 'generic', 'trend', or 'seasonality'.
        nb_blocks_per_stack: Number of blocks per stack. Default: 3.
            More blocks increase capacity but may cause overfitting.
        thetas_dim: Theta dimensions for each stack. Auto-calculated if None.
            When None, uses optimal defaults:
            - Trend: 4 (3rd order polynomial)
            - Seasonality: Based on forecast_length, typically 8-16
            - Generic: Moderate complexity based on forecast_length
        hidden_layer_units: Hidden units in each layer. Default: 256.
            Controls model capacity and computational cost.
        use_revin: Whether to use RevIN normalization. Default: True.
            RevIN typically improves performance by 10-20% on real datasets.
        optimizer: Optimizer for training. Default: 'adam'.
            Can be string name or optimizer instance. Gradient clipping will be added.
        loss: Loss function. Default: 'mae'.
            MAE often works better than MSE for N-BEATS forecasting.
        metrics: List of metrics to track. Default: ['mae', 'mse'] if None.
        learning_rate: Learning rate for optimizer. Default: 1e-4.
            Conservative default for stable N-BEATS training.
        gradient_clip_norm: Gradient clipping norm. Default: 1.0.
            Essential for N-BEATS training stability. Set to None to disable.
        kernel_regularizer: Weight regularizer. Default: None.
            Consider L2(1e-4) for complex models to prevent overfitting.
        dropout_rate: Dropout rate for regularization. Default: 0.0.
            Applied to forecasts between blocks.
        **kwargs: Additional arguments passed to NBeatsNet constructor.

    Returns:
        Compiled N-BEATS model ready for training.

    Raises:
        ValueError: If stack_types contains invalid values or configuration is invalid.

    Example:
        ```python
        # Simple univariate forecasting model
        model = create_nbeats_model(
            backcast_length=96,   # 4 days hourly
            forecast_length=24,   # 1 day forecast
            stack_types=['trend', 'seasonality']
        )

        # Complex model with custom configuration
        model = create_nbeats_model(
            backcast_length=168,  # 1 week hourly
            forecast_length=24,   # 1 day forecast
            stack_types=['trend', 'seasonality', 'generic'],
            hidden_layer_units=512,
            dropout_rate=0.1,
            kernel_regularizer=keras.regularizers.L2(1e-4),
            learning_rate=5e-5,
            gradient_clip_norm=0.5
        )

        # Custom optimizer with existing gradient clipping
        optimizer = keras.optimizers.AdamW(
            learning_rate=1e-4,
            clipnorm=1.0  # Will be preserved
        )
        model = create_nbeats_model(
            backcast_length=96,
            forecast_length=24,
            optimizer=optimizer
        )
        ```

    Note:
        The function automatically calculates optimal theta dimensions if not provided,
        validates the backcast/forecast ratio for performance, and ensures proper
        gradient clipping for training stability.
    """

    # Auto-calculate theta dimensions if not provided
    if thetas_dim is None:
        thetas_dim = []
        for stack_type in stack_types:
            if stack_type == 'trend':
                # 3rd order polynomial (4 theta dimensions)
                thetas_dim.append(4)
            elif stack_type == 'seasonality':
                # A reasonable number of harmonics can be half the forecast length, but capped for stability.
                harmonics = min(forecast_length // 2, max(4, forecast_length // 3))
                thetas_dim.append(harmonics * 2)
            elif stack_type == 'generic':
                theta_size = max(16, forecast_length * 2)
                thetas_dim.append(theta_size)
            else:
                # Fallback for any other types
                thetas_dim.append(8)

    # Validate backcast/forecast ratio and provide recommendations
    ratio = backcast_length / forecast_length
    if ratio < 3.0:
        logger.warning(
            f"backcast_length/forecast_length = {ratio:.1f} < 3.0. "
            f"Consider increasing backcast_length to {forecast_length * 4} "
            f"for better N-BEATS performance."
        )
    elif ratio >= 4.0:
        logger.info(f"Good backcast/forecast ratio: {ratio:.1f}")

    # Create model with configuration
    model = NBeatsNet(
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        stack_types=stack_types,
        nb_blocks_per_stack=nb_blocks_per_stack,
        thetas_dim=thetas_dim,
        hidden_layer_units=hidden_layer_units,
        use_revin=use_revin,
        kernel_regularizer=kernel_regularizer,
        dropout_rate=dropout_rate,
        **kwargs
    )

    # Setup default metrics if not provided
    if metrics is None:
        metrics = ['mae', 'mse', "mape"]

    # Setup optimizer with gradient clipping for training stability
    if isinstance(optimizer, str):
        optimizer_map = {
            'adam': keras.optimizers.Adam,
            'adamw': keras.optimizers.AdamW,
            'rmsprop': keras.optimizers.RMSprop,
            'sgd': keras.optimizers.SGD,
        }
        optimizer_cls = optimizer_map.get(optimizer.lower())
        if optimizer_cls is not None:
            # Create optimizer with gradient clipping if specified
            optimizer_kwargs = {'learning_rate': learning_rate}
            if gradient_clip_norm is not None:
                optimizer_kwargs['clipnorm'] = gradient_clip_norm
            optimizer = optimizer_cls(**optimizer_kwargs)
        else:
            # Fallback to keras.optimizers.get
            optimizer = keras.optimizers.get(optimizer)
            if (gradient_clip_norm is not None and
                    hasattr(optimizer, 'clipnorm') and
                    optimizer.clipnorm is None):
                optimizer.clipnorm = gradient_clip_norm
    elif isinstance(optimizer, keras.optimizers.Optimizer):
        # Add gradient clipping to existing optimizer if not already set
        if (gradient_clip_norm is not None and
                hasattr(optimizer, 'clipnorm') and
                optimizer.clipnorm is None):
            logger.info(f"Adding gradient clipping (norm={gradient_clip_norm}) to existing optimizer")
            optimizer.clipnorm = gradient_clip_norm

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    # Log model configuration
    logger.info("Created N-BEATS model with configuration:")
    logger.info(f"  - Architecture: {len(stack_types)} stacks, {nb_blocks_per_stack} blocks each")
    logger.info(f"  - Sequence: {backcast_length} → {forecast_length} (ratio: {ratio:.1f})")
    logger.info(f"  - Theta dimensions: {thetas_dim}")
    logger.info(f"  - RevIN normalization: {'✓' if use_revin else '✗'}")
    logger.info(f"  - Hidden units: {hidden_layer_units}")
    if dropout_rate > 0.0:
        logger.info(f"  - Dropout: {dropout_rate}")
    if gradient_clip_norm is not None:
        logger.info(f"  - Gradient clipping: {gradient_clip_norm}")
    logger.info(f"  - Optimizer: {optimizer.__class__.__name__}")
    logger.info(f"  - Loss: {loss}")

    return model

# ---------------------------------------------------------------------