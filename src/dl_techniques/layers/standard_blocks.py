"""
Configurable Building Blocks for Deep Learning Architectures.

This module provides a comprehensive collection of reusable, configurable building
blocks for constructing modern deep learning architectures. Each block is designed
with flexibility, composability, and best practices in mind, supporting various
normalization techniques, activation functions, and architectural patterns.

**Design Philosophy**:
    - **Composability**: Blocks can be easily combined to build complex architectures
    - **Configurability**: Factory patterns enable flexible selection of components
    - **Robustness**: Full Keras 3 serialization support with explicit build patterns
    - **Best Practices**: Type hints, comprehensive validation, and detailed documentation
    - **Performance**: Efficient implementations following modern deep learning patterns

**Available Blocks**:

1. **ConvBlock**: Flexible 2D convolutional block with configurable normalization,
   activation, dropout, and optional pooling. Ideal for CNN architectures.

2. **DenseBlock**: Fully connected block with optional normalization, activation,
   and dropout. Perfect for MLP components and classifier heads.

3. **ResidualDenseBlock**: Dense block with skip connection for deep networks,
   enabling gradient flow and improved training dynamics.

4. **BasicBlock**: Two-layer 3x3 convolutional block with residual connection,
   used in ResNet-18 and ResNet-34 architectures.

5. **BottleneckBlock**: Three-layer bottleneck block (1x1→3x3→1x1) with residual
   connection, used in ResNet-50, ResNet-101, and ResNet-152 architectures.

**Key Features**:

- **Factory Pattern Integration**: Uses `create_normalization_layer()` and
  `create_activation_layer()` factories for consistent component selection

- **Regularization Support**: All blocks support kernel regularizers, dropout,
  and various other regularization techniques

- **Flexible Initialization**: Customizable weight initializers for all layers

- **Complete Serialization**: All blocks are properly registered with Keras and
  support full save/load cycles

- **Explicit Building**: Follows modern Keras 3 patterns with explicit sub-layer
  building for robust serialization

**Usage Examples**:

    Basic convolutional block for image classification:

    >>> from dl_techniques.layers.standard_blocks import BasicBlock, BottleneckBlock, ConvBlock
    >>> 
    >>> # Create a standard conv block
    >>> conv = ConvBlock(
    ...     filters=64,
    ...     kernel_size=3,
    ...     normalization_type='batch_norm',
    ...     activation_type='relu',
    ...     use_pooling=True
    ... )
    >>> 
    >>> # Use in a model
    >>> inputs = keras.Input(shape=(224, 224, 3))
    >>> x = conv(inputs)
    >>> model = keras.Model(inputs, x)

    Building a ResNet-style architecture:

    >>> from dl_techniques.layers.standard_blocks import BasicBlock, BottleneckBlock
    >>> 
    >>> # ResNet-18 style block
    >>> inputs = keras.Input(shape=(56, 56, 64))
    >>> x = BasicBlock(64, stride=1)(inputs)
    >>> x = BasicBlock(64, stride=1)(x)
    >>> 
    >>> # ResNet-50 style block  
    >>> x = BottleneckBlock(64, stride=2, use_projection=True)(x)
    >>> x = BottleneckBlock(64, stride=1)(x)

    Creating a deep MLP with residual connections:

    >>> from dl_techniques.layers.standard_blocks import BasicBlock, BottleneckBlock
    >>> 
    >>> # Build a deep feedforward network
    >>> inputs = keras.Input(shape=(784,))
    >>> x = DenseBlock(512, normalization_type='layer_norm', activation_type='gelu')(inputs)
    >>> x = ResidualDenseBlock(normalization_type='rms_norm', dropout_rate=0.1)(x)
    >>> x = ResidualDenseBlock(normalization_type='rms_norm', dropout_rate=0.1)(x)
    >>> x = DenseBlock(10, normalization_type=None, activation_type='softmax')(x)
    >>> model = keras.Model(inputs, x)

**Normalization Options**:
    All blocks support flexible normalization through factory functions:
    - 'batch_norm': Batch Normalization
    - 'layer_norm': Layer Normalization
    - 'rms_norm': RMS Normalization
    - 'zero_centered_rms_norm': Zero-Centered RMS Normalization
    - Additional custom normalizations can be added via factory

**Activation Options**:
    All blocks support flexible activations through factory functions:
    - 'relu': Rectified Linear Unit
    - 'gelu': Gaussian Error Linear Unit
    - 'mish': Mish activation
    - 'swish' / 'silu': Swish/SiLU activation
    - 'hard_swish': Hard Swish activation
    - Additional custom activations can be added via factory

**Serialization Support**:
    All blocks are properly registered with Keras and support full serialization:

    >>> # Save model with custom blocks
    >>> model.save('my_model.keras')
    >>> 
    >>> # Load model (blocks automatically registered)
    >>> loaded_model = keras.models.load_model('my_model.keras')

**Architecture Patterns**:

1. **CNN Backbone**:
   ```
   Input → ConvBlock → ConvBlock → ... → GlobalPooling → DenseBlock → Output
   ```

2. **ResNet-Style Network**:
   ```
   Input → Conv → [BasicBlock/BottleneckBlock] × N → GlobalPooling → Dense → Output
   ```

3. **Deep MLP with Residuals**:
   ```
   Input → DenseBlock → [ResidualDenseBlock] × N → DenseBlock → Output
   ```

4. **Hybrid Architecture**:
   ```
   Input → [ConvBlock] × N → Flatten → [ResidualDenseBlock] × M → Output
   ```
"""

import keras
from typing import Optional, Union, Tuple, Dict, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .norms import create_normalization_layer
from .activations import create_activation_layer


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ConvBlock(keras.layers.Layer):
    """
    Configurable convolutional block with normalization, activation, and optional pooling.

    This layer implements a flexible convolutional building block consisting of:
    Conv2D → Normalization → Activation → Optional Dropout → Optional Pooling

    **Intent**: Provide a highly configurable convolutional block that can adapt
    to different architectural requirements through factory-based selection of
    normalization and activation layers.

    **Architecture**:
    ```
    Input(shape=[batch, height, width, channels])
           ↓
    Conv2D(filters, kernel_size, strides, padding)
           ↓
    Normalization (configurable type: layer_norm, batch_norm, rms_norm, etc.)
           ↓
    Activation (configurable type: relu, gelu, mish, etc.)
           ↓
    Optional Dropout(rate)
           ↓
    Optional Pooling (max or average)
           ↓
    Output(shape=[batch, new_height, new_width, filters])
    ```

    Args:
        filters: Number of convolutional filters.
        kernel_size: Size of convolutional kernel.
        strides: Convolution strides.
        padding: Padding mode ('same' or 'valid').
        normalization_type: Type of normalization ('layer_norm', 'batch_norm', 'rms_norm', etc.).
        activation_type: Type of activation ('relu', 'gelu', 'mish', etc.).
        dropout_rate: Dropout rate (0.0 to disable).
        use_pooling: Whether to apply pooling layer.
        pool_size: Size of pooling window.
        pool_type: Type of pooling ('max' or 'avg').
        kernel_regularizer: Regularizer for convolution kernel.
        kernel_initializer: Initializer for convolution kernel.
        normalization_kwargs: Additional arguments for normalization layer.
        activation_kwargs: Additional arguments for activation layer.
        **kwargs: Additional arguments for Layer base class.

    Example:
        ```python
        # Standard conv block with batch normalization and ReLU
        block1 = ConvBlock(
            filters=64,
            kernel_size=3,
            normalization_type='batch_norm',
            activation_type='relu'
        )

        # Advanced conv block with RMS normalization and Mish activation
        block2 = ConvBlock(
            filters=128,
            kernel_size=3,
            normalization_type='rms_norm',
            activation_type='mish',
            use_pooling=True,
            pool_type='max',
            normalization_kwargs={'use_scale': True, 'epsilon': 1e-5}
        )

        # Transformer-style conv block with layer normalization
        block3 = ConvBlock(
            filters=256,
            kernel_size=1,
            normalization_type='layer_norm',
            activation_type='gelu',
            normalization_kwargs={'axis': -1}
        )
        ```
    """

    def __init__(
            self,
            filters: int,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            strides: Union[int, Tuple[int, int]] = 1,
            padding: str = "same",
            normalization_type: str = "batch_norm",
            activation_type: str = "relu",
            dropout_rate: float = 0.0,
            use_pooling: bool = False,
            pool_size: Union[int, Tuple[int, int]] = 2,
            pool_type: Literal["max", "avg"] = "max",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            activation_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        """Initialize ConvBlock with specified parameters."""
        super().__init__(**kwargs)

        # Validate inputs
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if pool_type not in ["max", "avg"]:
            raise ValueError(f"pool_type must be 'max' or 'avg', got {pool_type}")
        if padding not in ["same", "valid"]:
            raise ValueError(f"padding must be 'same' or 'valid', got {padding}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0,1], got {dropout_rate}")

        # Store configuration
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.normalization_type = normalization_type
        self.activation_type = activation_type
        self.dropout_rate = dropout_rate
        self.use_pooling = use_pooling
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.normalization_kwargs = normalization_kwargs or {}
        self.activation_kwargs = activation_kwargs or {}

        # Create sub-layers in __init__
        self.conv = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
            name=f"{self.name}_conv"
        )

        # Create normalization layer using factory
        self.norm = create_normalization_layer(
            normalization_type,
            name=f"{self.name}_norm",
            **self.normalization_kwargs
        )

        # Create activation layer using factory
        self.activation = create_activation_layer(
            activation_type,
            name=f"{self.name}_activation",
            **self.activation_kwargs
        )

        # Create dropout layer if requested
        if dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(
                rate=dropout_rate, name=f"{self.name}_dropout"
            )
        else:
            self.dropout = None

        # Create pooling layer if requested
        if use_pooling:
            if pool_type == "max":
                self.pool = keras.layers.MaxPooling2D(
                    pool_size=pool_size, name=f"{self.name}_pool"
                )
            else:  # avg
                self.pool = keras.layers.AveragePooling2D(
                    pool_size=pool_size, name=f"{self.name}_pool"
                )
        else:
            self.pool = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build sub-layers explicitly for proper serialization.

        CRITICAL: Explicitly build each sub-layer to ensure weight variables
        exist before weight restoration during model loading.
        """
        # Build sub-layers in computational order
        self.conv.build(input_shape)

        conv_output_shape = self.conv.compute_output_shape(input_shape)
        self.norm.build(conv_output_shape)
        self.activation.build(conv_output_shape)

        if self.dropout is not None:
            self.dropout.build(conv_output_shape)

        if self.pool is not None:
            self.pool.build(conv_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the convolutional block."""
        x = self.conv(inputs)
        x = self.norm(x, training=training)
        x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        if self.pool is not None:
            x = self.pool(x)

        return x

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        shape = self.conv.compute_output_shape(input_shape)
        if self.pool is not None:
            shape = self.pool.compute_output_shape(shape)
        return shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'normalization_type': self.normalization_type,
            'activation_type': self.activation_type,
            'dropout_rate': self.dropout_rate,
            'use_pooling': self.use_pooling,
            'pool_size': self.pool_size,
            'pool_type': self.pool_type,
            'kernel_regularizer': keras.regularizers.serialize(
                self.kernel_regularizer) if self.kernel_regularizer else None,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'normalization_kwargs': self.normalization_kwargs,
            'activation_kwargs': self.activation_kwargs,
        })
        return config


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DenseBlock(keras.layers.Layer):
    """
    Configurable dense block with normalization, activation, and optional dropout.

    This layer implements a flexible dense building block consisting of:
    Dense → Optional Normalization → Activation → Optional Dropout

    **Intent**: Provide a highly configurable dense block that can adapt to
    different architectural requirements through factory-based selection of
    normalization and activation layers.

    **Architecture**:
    ```
    Input(shape=[batch, input_features])
           ↓
    Dense(units, kernel_regularizer, kernel_initializer)
           ↓
    Optional Normalization (configurable type: layer_norm, rms_norm, etc.)
           ↓
    Activation (configurable type: relu, gelu, mish, etc.)
           ↓
    Optional Dropout(rate)
           ↓
    Output(shape=[batch, units])
    ```

    Args:
        units: Number of dense units.
        normalization_type: Type of normalization ('layer_norm', 'rms_norm', etc.).
            Set to None to disable normalization.
        activation_type: Type of activation ('relu', 'gelu', 'mish', etc.).
        dropout_rate: Dropout rate (0.0 to disable).
        kernel_regularizer: Regularizer for dense kernel.
        bias_regularizer: Regularizer for dense bias.
        activity_regularizer: Regularizer for dense layer activity.
        kernel_initializer: Initializer for dense kernel.
        bias_initializer: Initializer for dense bias.
        use_bias: Whether to use bias in dense layer.
        normalization_kwargs: Additional arguments for normalization layer.
        activation_kwargs: Additional arguments for activation layer.
        **kwargs: Additional arguments for Layer base class.

    Example:
        ```python
        # Standard dense block with layer normalization and ReLU
        block1 = DenseBlock(
            units=512,
            normalization_type='layer_norm',
            activation_type='relu',
            dropout_rate=0.1
        )

        # Transformer-style dense block with RMS normalization and GELU
        block2 = DenseBlock(
            units=2048,
            normalization_type='rms_norm',
            activation_type='gelu',
            normalization_kwargs={'use_scale': True}
        )

        # Simple dense block without normalization
        block3 = DenseBlock(
            units=256,
            normalization_type=None,  # No normalization
            activation_type='mish',
            dropout_rate=0.2
        )

        # Advanced block with custom regularization
        block4 = DenseBlock(
            units=1024,
            normalization_type='zero_centered_rms_norm',
            activation_type='hard_swish',
            kernel_regularizer=keras.regularizers.L2(1e-4),
            normalization_kwargs={'epsilon': 1e-5},
            dropout_rate=0.15
        )
        ```
    """

    def __init__(
            self,
            units: int,
            normalization_type: Optional[str] = "layer_norm",
            activation_type: str = "relu",
            dropout_rate: float = 0.0,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_constraint: Optional[keras.constraints.Constraint] = None,
            bias_constraint: Optional[keras.constraints.Constraint] = None,
            use_bias: bool = True,
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            activation_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        """Initialize DenseBlock with specified parameters."""
        super().__init__(**kwargs)

        # Validate inputs
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0,1], got {dropout_rate}")

        # Store configuration
        self.units = units
        self.normalization_type = normalization_type
        self.activation_type = activation_type
        self.dropout_rate = dropout_rate
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.use_bias = use_bias
        self.normalization_kwargs = normalization_kwargs or {}
        self.activation_kwargs = activation_kwargs or {}

        # Create sub-layers in __init__
        self.dense = keras.layers.Dense(
            units=units,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=f"{self.name}_dense"
        )

        # Create normalization layer using factory (optional)
        if normalization_type is not None:
            self.norm = create_normalization_layer(
                normalization_type,
                name=f"{self.name}_norm",
                **self.normalization_kwargs
            )
        else:
            self.norm = None

        # Create activation layer using factory
        self.activation = create_activation_layer(
            activation_type,
            name=f"{self.name}_activation",
            **self.activation_kwargs
        )

        # Create dropout layer if requested
        if dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(
                rate=dropout_rate, name=f"{self.name}_dropout"
            )
        else:
            self.dropout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build sub-layers explicitly for proper serialization.

        CRITICAL: Explicitly build each sub-layer to ensure weight variables
        exist before weight restoration during model loading.
        """
        # Build sub-layers in computational order
        self.dense.build(input_shape)

        dense_output_shape = self.dense.compute_output_shape(input_shape)

        if self.norm is not None:
            self.norm.build(dense_output_shape)

        self.activation.build(dense_output_shape)

        if self.dropout is not None:
            self.dropout.build(dense_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the dense block."""
        x = self.dense(inputs)

        if self.norm is not None:
            x = self.norm(x, training=training)

        x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        return x

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return self.dense.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'normalization_type': self.normalization_type,
            'activation_type': self.activation_type,
            'dropout_rate': self.dropout_rate,
            'kernel_regularizer': keras.regularizers.serialize(
                self.kernel_regularizer) if self.kernel_regularizer else None,
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer) if self.bias_regularizer else None,
            'activity_regularizer': keras.regularizers.serialize(
                self.activity_regularizer) if self.activity_regularizer else None,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_constraint': keras.constraints.serialize(
                self.kernel_constraint) if self.kernel_constraint else None,
            'bias_constraint': keras.constraints.serialize(self.bias_constraint) if self.bias_constraint else None,
            'use_bias': self.use_bias,
            'normalization_kwargs': self.normalization_kwargs,
            'activation_kwargs': self.activation_kwargs,
        })
        return config


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ResidualDenseBlock(keras.layers.Layer):
    """
    Dense block with residual connection and configurable normalization/activation.

    This layer implements a residual dense block consisting of:
    Input → Dense → Optional Normalization → Activation → Optional Dropout → Add(Input) → Output

    **Intent**: Provide a residual dense block that enables deeper networks with
    skip connections while maintaining configurability through factory patterns.

    **Architecture**:
    ```
    Input(shape=[batch, features])
           ↓                    ↘ (skip connection)
    Dense(units=features)       ↓
           ↓                    ↓
    Optional Normalization      ↓
           ↓                    ↓
    Activation                  ↓
           ↓                    ↓
    Optional Dropout            ↓
           ↓                    ↓
    Add ←──────────────────────↙
           ↓
    Output(shape=[batch, features])
    ```

    Args:
        units: Number of dense units. If None, will match input dimension (determined in build()).
        normalization_type: Type of normalization ('layer_norm', 'rms_norm', etc.).
            Set to None to disable normalization.
        activation_type: Type of activation ('relu', 'gelu', 'mish', etc.).
        dropout_rate: Dropout rate (0.0 to disable).
        kernel_regularizer: Regularizer for dense kernel.
        kernel_initializer: Initializer for dense kernel.
        use_bias: Whether to use bias in dense layer.
        normalization_kwargs: Additional arguments for normalization layer.
        activation_kwargs: Additional arguments for activation layer.
        **kwargs: Additional arguments for Layer base class.

    Note:
        If units is None, the dense layer will have the same number of units as
        the input features to enable the residual connection. The layer automatically
        determines the correct number of units from the input shape during build.
        If units is specified, it must match the input dimension for the residual
        connection to work.
    """

    def __init__(
            self,
            units: Optional[int] = None,
            normalization_type: Optional[str] = "layer_norm",
            activation_type: str = "relu",
            dropout_rate: float = 0.0,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            use_bias: bool = True,
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            activation_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        """Initialize ResidualDenseBlock with specified parameters."""
        super().__init__(**kwargs)

        # Validate inputs
        if units is not None and units <= 0:
            raise ValueError(f"units must be positive or None, got {units}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0,1], got {dropout_rate}")

        # Store configuration
        self.units = units
        self.normalization_type = normalization_type
        self.activation_type = activation_type
        self.dropout_rate = dropout_rate
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.normalization_kwargs = normalization_kwargs or {}
        self.activation_kwargs = activation_kwargs or {}

        # Create sub-layers in __init__ (except dense which requires input shape)
        # Dense layer will be created in build() since units may depend on input shape
        self.dense = None
        self.norm = None
        self.activation = None
        self.dropout = None
        self.add = keras.layers.Add(name=f"{self.name}_add")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build sub-layers with input-dependent configuration.

        CRITICAL: Create and build the dense layer here because it depends on
        the input shape to determine the number of units for the residual connection.
        """
        if len(input_shape) < 2:
            raise ValueError(f"Input must be at least 2D, got shape {input_shape}")

        # Get the number of input features for residual connection
        input_features = input_shape[-1]
        if input_features is None:
            raise ValueError("Input shape must have a defined last dimension")

        # Determine units: use specified units or match input features
        units = self.units if self.units is not None else input_features

        # Validate that units matches input features for residual connection
        if units != input_features:
            raise ValueError(
                f"For residual connection, units ({units}) must match input features "
                f"({input_features}). Either set units={input_features} or leave units=None."
            )

        # Create dense layer with same units as input features
        self.dense = keras.layers.Dense(
            units=units,
            use_bias=self.use_bias,
            kernel_regularizer=self.kernel_regularizer,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_dense"
        )
        self.dense.build(input_shape)

        # Create and build normalization layer using factory (optional)
        dense_output_shape = self.dense.compute_output_shape(input_shape)
        if self.normalization_type is not None:
            self.norm = create_normalization_layer(
                self.normalization_type,
                name=f"{self.name}_norm",
                **self.normalization_kwargs
            )
            self.norm.build(dense_output_shape)

        # Create and build activation layer using factory
        self.activation = create_activation_layer(
            self.activation_type,
            name=f"{self.name}_activation",
            **self.activation_kwargs
        )
        self.activation.build(dense_output_shape)

        # Create and build dropout layer if requested
        if self.dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(
                rate=self.dropout_rate, name=f"{self.name}_dropout"
            )
            self.dropout.build(dense_output_shape)

        # Build add layer
        self.add.build([input_shape, dense_output_shape])

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the residual dense block."""
        # Forward pass through transformation
        x = self.dense(inputs)

        if self.norm is not None:
            x = self.norm(x, training=training)

        x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        # Add residual connection
        return self.add([inputs, x])

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape (same as input for residual connection)."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'normalization_type': self.normalization_type,
            'activation_type': self.activation_type,
            'dropout_rate': self.dropout_rate,
            'kernel_regularizer': keras.regularizers.serialize(
                self.kernel_regularizer) if self.kernel_regularizer else None,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'use_bias': self.use_bias,
            'normalization_kwargs': self.normalization_kwargs,
            'activation_kwargs': self.activation_kwargs,
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BasicBlock(keras.layers.Layer):
    """
    Basic ResNet block with two 3x3 convolutions.

    Used in ResNet-18 and ResNet-34. The block consists of:
    - Conv 3x3, filters
    - BatchNorm
    - ReLU
    - Conv 3x3, filters
    - BatchNorm
    - Add shortcut
    - ReLU

    **Intent**: Provide the basic building block for ResNet architectures,
    enabling deep residual learning with two-layer transformations and
    skip connections.

    **Architecture**:
    ```
    Input(shape=[batch, height, width, channels])
           ↓                               ↘ (shortcut)
    Conv2D(3x3, filters, stride)            ↓
           ↓                                ↓
    Normalization                           ↓
           ↓                                ↓
    Activation                              ↓
           ↓                                ↓
    Conv2D(3x3, filters)                    ↓
           ↓                                ↓
    Normalization                           ↓
           ↓                                ↓
    Add ←──────────────────────────────────↙
           ↓
    Activation
           ↓
    Output(shape=[batch, new_height, new_width, filters])
    ```

    Args:
        filters: Number of output filters.
        stride: Stride for the first convolution. Default is 1.
        use_projection: Whether to use a 1x1 projection for the shortcut.
        kernel_regularizer: Regularizer for convolution kernels.
        normalization_type: Type of normalization layer. Default is 'batch_norm'.
        activation_type: Type of activation function. Default is 'relu'.
        **kwargs: Additional keyword arguments for Layer.

    Example:
        ```python
        # Basic block with stride 1 (no downsampling)
        block1 = BasicBlock(64, stride=1, use_projection=False)

        # Basic block with stride 2 (downsampling, requires projection)
        block2 = BasicBlock(128, stride=2, use_projection=True)

        # Basic block with custom normalization and activation
        block3 = BasicBlock(
            filters=256,
            normalization_type='layer_norm',
            activation_type='gelu'
        )
        ```
    """

    def __init__(
            self,
            filters: int,
            stride: int = 1,
            use_projection: bool = False,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            normalization_type: str = "batch_norm",
            activation_type: str = "relu",
            **kwargs: Any
    ) -> None:
        """Initialize BasicBlock with specified parameters."""
        super().__init__(**kwargs)

        # Validate inputs
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        # Store configuration
        self.filters = filters
        self.stride = stride
        self.use_projection = use_projection
        self.kernel_regularizer = kernel_regularizer
        self.normalization_type = normalization_type
        self.activation_type = activation_type

        # Create sub-layers in __init__
        # First convolution
        self.conv1 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=kernel_regularizer,
            name=f"{self.name}_conv1"
        )
        self.bn1 = create_normalization_layer(
            normalization_type,
            name=f"{self.name}_bn1"
        )
        self.act1 = create_activation_layer(
            activation_type,
            name=f"{self.name}_act1"
        )

        # Second convolution
        self.conv2 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=kernel_regularizer,
            name=f"{self.name}_conv2"
        )
        self.bn2 = create_normalization_layer(
            normalization_type,
            name=f"{self.name}_bn2"
        )

        # Shortcut projection if needed
        if use_projection:
            self.shortcut_conv = keras.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=stride,
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
                kernel_regularizer=kernel_regularizer,
                name=f"{self.name}_shortcut_conv"
            )
            self.shortcut_bn = create_normalization_layer(
                normalization_type,
                name=f"{self.name}_shortcut_bn"
            )
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None

        self.add = keras.layers.Add(name=f"{self.name}_add")
        self.act_final = create_activation_layer(
            activation_type,
            name=f"{self.name}_act_final"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build sub-layers explicitly for proper serialization.

        CRITICAL: Explicitly build each sub-layer to ensure weight variables
        exist before weight restoration during model loading.
        """
        # Build main path
        self.conv1.build(input_shape)
        conv1_output_shape = self.conv1.compute_output_shape(input_shape)

        self.bn1.build(conv1_output_shape)
        self.act1.build(conv1_output_shape)

        self.conv2.build(conv1_output_shape)
        conv2_output_shape = self.conv2.compute_output_shape(conv1_output_shape)

        self.bn2.build(conv2_output_shape)

        # Build shortcut path
        if self.use_projection:
            self.shortcut_conv.build(input_shape)
            shortcut_output_shape = self.shortcut_conv.compute_output_shape(input_shape)
            self.shortcut_bn.build(shortcut_output_shape)

        # Build add layer
        shortcut_shape = shortcut_output_shape if self.use_projection else input_shape
        self.add.build([conv2_output_shape, shortcut_shape])

        # Build final activation
        add_output_shape = conv2_output_shape  # Add preserves shape
        self.act_final.build(add_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the basic block."""
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        # Shortcut path
        if self.use_projection:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = inputs

        # Add and activate
        x = self.add([x, shortcut])
        x = self.act_final(x)

        return x

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return self.conv2.compute_output_shape(
            self.conv1.compute_output_shape(input_shape)
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "stride": self.stride,
            "use_projection": self.use_projection,
            "kernel_regularizer": keras.regularizers.serialize(
                self.kernel_regularizer) if self.kernel_regularizer else None,
            "normalization_type": self.normalization_type,
            "activation_type": self.activation_type,
        })
        return config


@keras.saving.register_keras_serializable()
class BottleneckBlock(keras.layers.Layer):
    """
    Bottleneck ResNet block with 1x1 → 3x3 → 1x1 convolutions.

    Used in ResNet-50, ResNet-101, and ResNet-152. The block consists of:
    - Conv 1x1, filters (dimensionality reduction)
    - BatchNorm
    - ReLU
    - Conv 3x3, filters (bottleneck)
    - BatchNorm
    - ReLU
    - Conv 1x1, filters * 4 (dimensionality expansion)
    - BatchNorm
    - Add shortcut
    - ReLU

    **Intent**: Provide an efficient building block for deep ResNet architectures,
    using bottleneck design to reduce computational cost while maintaining
    representational capacity through dimensionality reduction and expansion.

    **Architecture**:
    ```
    Input(shape=[batch, height, width, in_channels])
           ↓                                  ↘ (shortcut)
    Conv2D(1x1, filters)                      ↓
           ↓                                  ↓
    Normalization                             ↓
           ↓                                  ↓
    Activation                                ↓
           ↓                                  ↓
    Conv2D(3x3, filters, stride)              ↓
           ↓                                  ↓
    Normalization                             ↓
           ↓                                  ↓
    Activation                                ↓
           ↓                                  ↓
    Conv2D(1x1, filters * 4)                  ↓
           ↓                                  ↓
    Normalization                             ↓
           ↓                                  ↓
    Add ←─────────────────────────────────────↙
           ↓
    Activation
           ↓
    Output(shape=[batch, new_height, new_width, filters * 4])
    ```

    Args:
        filters: Number of filters in the bottleneck (middle layer).
            Output will be filters * 4.
        stride: Stride for the 3x3 convolution. Default is 1.
        use_projection: Whether to use a 1x1 projection for the shortcut.
        kernel_regularizer: Regularizer for convolution kernels.
        normalization_type: Type of normalization layer. Default is 'batch_norm'.
        activation_type: Type of activation function. Default is 'relu'.
        **kwargs: Additional keyword arguments for Layer.

    Example:
        ```python
        # Bottleneck block with stride 1 (no downsampling)
        block1 = BottleneckBlock(64, stride=1, use_projection=True)

        # Bottleneck block with stride 2 (downsampling, requires projection)
        block2 = BottleneckBlock(128, stride=2, use_projection=True)

        # Bottleneck block with custom normalization and activation
        block3 = BottleneckBlock(
            filters=256,
            normalization_type='layer_norm',
            activation_type='gelu'
        )
        ```
    """

    def __init__(
            self,
            filters: int,
            stride: int = 1,
            use_projection: bool = False,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            normalization_type: str = "batch_norm",
            activation_type: str = "relu",
            **kwargs: Any
    ) -> None:
        """Initialize BottleneckBlock with specified parameters."""
        super().__init__(**kwargs)

        # Validate inputs
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        # Store configuration
        self.filters = filters
        self.stride = stride
        self.use_projection = use_projection
        self.kernel_regularizer = kernel_regularizer
        self.normalization_type = normalization_type
        self.activation_type = activation_type
        self.expansion = 4  # Bottleneck expansion factor

        # Create sub-layers in __init__
        # First 1x1 convolution (dimensionality reduction)
        self.conv1 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=kernel_regularizer,
            name=f"{self.name}_conv1"
        )
        self.bn1 = create_normalization_layer(
            normalization_type,
            name=f"{self.name}_bn1"
        )
        self.act1 = create_activation_layer(
            activation_type,
            name=f"{self.name}_act1"
        )

        # Second 3x3 convolution (bottleneck)
        self.conv2 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=kernel_regularizer,
            name=f"{self.name}_conv2"
        )
        self.bn2 = create_normalization_layer(
            normalization_type,
            name=f"{self.name}_bn2"
        )
        self.act2 = create_activation_layer(
            activation_type,
            name=f"{self.name}_act2"
        )

        # Third 1x1 convolution (dimensionality expansion)
        self.conv3 = keras.layers.Conv2D(
            filters=filters * self.expansion,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=kernel_regularizer,
            name=f"{self.name}_conv3"
        )
        self.bn3 = create_normalization_layer(
            normalization_type,
            name=f"{self.name}_bn3"
        )

        # Shortcut projection if needed
        if use_projection:
            self.shortcut_conv = keras.layers.Conv2D(
                filters=filters * self.expansion,
                kernel_size=1,
                strides=stride,
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
                kernel_regularizer=kernel_regularizer,
                name=f"{self.name}_shortcut_conv"
            )
            self.shortcut_bn = create_normalization_layer(
                normalization_type,
                name=f"{self.name}_shortcut_bn"
            )
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None

        self.add = keras.layers.Add(name=f"{self.name}_add")
        self.act_final = create_activation_layer(
            activation_type,
            name=f"{self.name}_act_final"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build sub-layers explicitly for proper serialization.

        CRITICAL: Explicitly build each sub-layer to ensure weight variables
        exist before weight restoration during model loading.
        """
        # Build main path - first conv
        self.conv1.build(input_shape)
        conv1_output_shape = self.conv1.compute_output_shape(input_shape)

        self.bn1.build(conv1_output_shape)
        self.act1.build(conv1_output_shape)

        # Build main path - second conv
        self.conv2.build(conv1_output_shape)
        conv2_output_shape = self.conv2.compute_output_shape(conv1_output_shape)

        self.bn2.build(conv2_output_shape)
        self.act2.build(conv2_output_shape)

        # Build main path - third conv
        self.conv3.build(conv2_output_shape)
        conv3_output_shape = self.conv3.compute_output_shape(conv2_output_shape)

        self.bn3.build(conv3_output_shape)

        # Build shortcut path
        if self.use_projection:
            self.shortcut_conv.build(input_shape)
            shortcut_output_shape = self.shortcut_conv.compute_output_shape(input_shape)
            self.shortcut_bn.build(shortcut_output_shape)

        # Build add layer
        shortcut_shape = shortcut_output_shape if self.use_projection else input_shape
        self.add.build([conv3_output_shape, shortcut_shape])

        # Build final activation
        add_output_shape = conv3_output_shape  # Add preserves shape
        self.act_final.build(add_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the bottleneck block."""
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        # Shortcut path
        if self.use_projection:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = inputs

        # Add and activate
        x = self.add([x, shortcut])
        x = self.act_final(x)

        return x

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        shape = self.conv1.compute_output_shape(input_shape)
        shape = self.conv2.compute_output_shape(shape)
        shape = self.conv3.compute_output_shape(shape)
        return shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "stride": self.stride,
            "use_projection": self.use_projection,
            "kernel_regularizer": keras.regularizers.serialize(
                self.kernel_regularizer) if self.kernel_regularizer else None,
            "normalization_type": self.normalization_type,
            "activation_type": self.activation_type,
        })
        return config

# ---------------------------------------------------------------------