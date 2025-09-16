import keras
from keras import layers
from typing import Tuple, Optional, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class InvertedResidualBlock(keras.layers.Layer):
    """Inverted residual block, the core building block for MobileNetV2.

    This block implements the inverted residual structure with a linear bottleneck,
    which is a key innovation of the MobileNetV2 architecture. It consists of
    three main stages: expansion, depthwise convolution, and projection.

    **Intent**: To provide a faithful and robust implementation of the MobileNetV2
    building block, following modern Keras 3 best practices for custom composite
    layers, ensuring correct serialization and weight restoration.

    **Architecture**:
    ```
    Input(shape=[H, W, C_in])
           │
           ├─(Residual Connection, if stride=1 and C_in=C_out)
           ↓
    1. Expansion: 1x1 Conv2D -> BN -> ReLU6 (channels: C_in -> C_expanded)
           ↓
    2. Depthwise: 3x3 DepthwiseConv2D -> BN -> ReLU6 (channels: C_expanded)
           ↓
    3. Projection: 1x1 Conv2D -> BN (channels: C_expanded -> C_out)
           │                        (LINEAR activation - the "bottleneck")
           ↓
    Output(shape=[H', W', C_out]) + Residual
    ```

    **Mathematical Operation**:
    1. **Expansion**: `x_expanded = ReLU6(BN(Conv_1x1(inputs)))`
    2. **Depthwise**: `x_depthwise = ReLU6(BN(DepthwiseConv_3x3(x_expanded)))`
    3. **Projection**: `x_projected = BN(Conv_1x1(x_depthwise))`
    4. **Residual**: `output = inputs + x_projected` (if applicable)

    The key innovation is the linear projection layer, which prevents non-linearities
    from destroying information in the low-dimensional bottleneck.

    Args:
        filters: Integer, the number of output filters (channels).
        expansion_factor: Integer, the expansion ratio for the first layer.
            This determines the intermediate channel dimension.
        stride: Integer, the stride for the depthwise convolution (1 or 2).
        block_id: Integer, a unique identifier for the block for naming purposes.
        skip_connection: Boolean, whether to use a residual connection.
            A residual connection is only added if `stride=1` and input/output
            channels match.
        kernel_initializer: String or Initializer for weight initialization.
        kernel_regularizer: Optional regularizer for kernel weights.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.

    Output shape:
        4D tensor with shape: `(batch_size, new_height, new_width, filters)`.
        `new_height` and `new_width` will be smaller if `stride > 1`.

    Attributes:
        expand_conv, expand_bn, expand_relu: Layers for the expansion phase.
        depthwise_conv, depthwise_bn, depthwise_relu: Layers for the depthwise phase.
        project_conv, project_bn: Layers for the projection phase.
        add: Add layer for the residual connection (if used).

    Example:
        ```python
        # A block that downsamples spatially (stride=2)
        block = InvertedResidualBlock(filters=32, expansion_factor=6, stride=2, block_id=1)
        inputs = keras.Input(shape=(56, 56, 24))
        outputs = block(inputs) # Output shape: (None, 28, 28, 32)

        # A block with a residual connection
        block = InvertedResidualBlock(filters=24, expansion_factor=6, stride=1, block_id=2)
        inputs = keras.Input(shape=(56, 56, 24))
        outputs = block(inputs) # Output shape: (None, 56, 56, 24)
        ```

    References:
        - MobileNetV2: Inverted Residuals and Linear Bottlenecks: https://arxiv.org/abs/1801.04381
    """

    def __init__(
            self,
            filters: int,
            expansion_factor: int = 6,
            stride: int = 1,
            block_id: int = 0,
            skip_connection: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store ALL configuration
        self.filters = filters
        self.expansion_factor = expansion_factor
        self.stride = stride
        self.block_id = block_id
        self.skip_connection = skip_connection
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

        # Initialize attributes for sub-layers (created in build)
        self.in_channels = None
        self.expanded_channels = None
        self.use_residual = False

        # --- CREATE shape-independent sub-layers in __init__ ---
        # ReLU layers are shape-independent and can be created here.
        self.expand_relu = layers.ReLU(max_value=6, name=f'expand_relu6_{self.block_id}')
        self.depthwise_relu = layers.ReLU(max_value=6, name=f'depthwise_relu6_{self.block_id}')
        self.add = layers.Add(name=f'add_{self.block_id}')

        # Shape-dependent layers will be created and built in the build method.
        self.expand_conv = None
        self.expand_bn = None
        self.depthwise_conv = None
        self.depthwise_bn = None
        self.project_conv = None
        self.project_bn = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create and build the block's weights and sub-layers."""
        self.in_channels = input_shape[-1]
        if self.in_channels is None:
            raise ValueError("The channel dimension of the inputs must be defined. Found `None`.")
        self.expanded_channels = self.in_channels * self.expansion_factor

        # Determine if a residual connection should be used
        self.use_residual = (self.skip_connection and self.stride == 1 and self.in_channels == self.filters)

        # --- CREATE shape-dependent sub-layers in build ---
        # Layer creation is deferred to `build` because parameters like the number of
        # filters depend on the input shape.

        # 1. Expansion phase (only if expansion factor > 1)
        if self.expansion_factor != 1:
            self.expand_conv = layers.Conv2D(
                filters=self.expanded_channels,
                kernel_size=1,
                padding='same',
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'expand_{self.block_id}'
            )
            self.expand_bn = layers.BatchNormalization(name=f'expand_bn_{self.block_id}')

        # 2. Depthwise convolution phase
        self.depthwise_conv = layers.DepthwiseConv2D(
            kernel_size=3,
            strides=self.stride,
            padding='same',
            use_bias=False,
            depthwise_initializer=self.kernel_initializer,
            depthwise_regularizer=self.kernel_regularizer,
            name=f'depthwise_{self.block_id}'
        )
        self.depthwise_bn = layers.BatchNormalization(name=f'depthwise_bn_{self.block_id}')

        # 3. Projection phase (with linear bottleneck)
        self.project_conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=f'project_{self.block_id}'
        )
        self.project_bn = layers.BatchNormalization(name=f'project_bn_{self.block_id}')

        # --- BUILD all sub-layers explicitly for robust serialization ---
        # This follows the guide's pattern for composite layers.
        current_shape = input_shape
        if self.expansion_factor != 1:
            self.expand_conv.build(current_shape)
            current_shape = self.expand_conv.compute_output_shape(current_shape)
            self.expand_bn.build(current_shape)

        self.depthwise_conv.build(current_shape)
        current_shape = self.depthwise_conv.compute_output_shape(current_shape)
        self.depthwise_bn.build(current_shape)

        self.project_conv.build(current_shape)
        current_shape = self.project_conv.compute_output_shape(current_shape)
        self.project_bn.build(current_shape)

        # Always call parent build last
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass through the inverted residual block."""
        x = inputs

        # 1. Expansion phase
        if self.expansion_factor != 1:
            x = self.expand_conv(x)
            x = self.expand_bn(x, training=training)
            x = self.expand_relu(x)

        # 2. Depthwise convolution phase
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x, training=training)
        x = self.depthwise_relu(x)

        # 3. Projection phase (linear bottleneck)
        x = self.project_conv(x)
        x = self.project_bn(x, training=training)

        # 4. Residual connection
        if self.use_residual:
            x = self.add([inputs, x])

        return x

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'expansion_factor': self.expansion_factor,
            'stride': self.stride,
            'block_id': self.block_id,
            'skip_connection': self.skip_connection,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------
