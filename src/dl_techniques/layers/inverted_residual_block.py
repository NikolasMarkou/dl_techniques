import keras
from typing import  Optional, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .universal_inverted_bottleneck import UniversalInvertedBottleneck

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class InvertedResidualBlock(UniversalInvertedBottleneck):
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
            kernel_initializer: Union[
                str, keras.initializers.Initializer
            ] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any,
    ) -> None:
        # Store original arguments for get_config serialization.
        # These are the public API parameters for this specialized class.
        self._block_id = block_id
        self._skip_connection_arg = skip_connection
        self._kernel_initializer_arg = kernel_initializer
        self._kernel_regularizer_arg = kernel_regularizer

        # Call the parent UniversalInvertedBottleneck's __init__ with the
        # specific configuration for a MobileNetV2 block.
        super().__init__(
            filters=filters,
            expansion_factor=expansion_factor,
            stride=stride,
            kernel_size=3,
            use_dw1=True,
            use_dw2=False,
            activation_type="relu",
            activation_args={"max_value": 6},  # This creates ReLU6
            normalization_type="batch_norm",
            dropout_rate=0.0,
            use_squeeze_excitation=False,
            kernel_initializer=kernel_initializer,
            depthwise_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            depthwise_regularizer=kernel_regularizer,
            name=f"inverted_residual_block_{block_id}",
            **kwargs,
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        This method ensures the layer is saved using its own simplified
        constructor arguments, not the full UIB configuration.
        """
        # Start with the full configuration from the UIB parent.
        config = super().get_config()

        # Define the set of UIB-specific parameters that are hard-coded
        # in this class and should not be part of the saved config.
        params_to_remove = [
            "expanded_channels",
            "kernel_size",
            "use_dw1",
            "use_dw2",
            "activation_type",
            "activation_args",
            "normalization_type",
            "normalization_args",
            "dropout_rate",
            "use_squeeze_excitation",
            "se_ratio",
            "se_activation",
            "use_bias",
            "padding",
            "block_type",
            "depthwise_initializer",
            "depthwise_regularizer",
        ]
        for param in params_to_remove:
            config.pop(param, None)

        # Update the config with the original arguments of this class,
        # ensuring they are properly serialized.
        config.update(
            {
                "block_id": self._block_id,
                "skip_connection": self._skip_connection_arg,
                "kernel_initializer": keras.initializers.serialize(
                    self._kernel_initializer_arg
                ),
                "kernel_regularizer": keras.regularizers.serialize(
                    self._kernel_regularizer_arg
                ),
            }
        )

        return config

# ---------------------------------------------------------------------