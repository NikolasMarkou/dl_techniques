import keras
from keras import ops
from typing import Tuple, Optional, Any, Dict, Union

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class UIB(keras.layers.Layer):
    """Universal Inverted Bottleneck (UIB) block.

    This block unifies and extends various efficient building blocks:
    Inverted Bottleneck (IB), ConvNext, Feed-Forward Network (FFN), and Extra Depthwise (ExtraDW).

    Args:
        filters: Number of output filters
        expansion_factor: Expansion factor for the block
        stride: Stride for the depthwise convolutions
        kernel_size: Kernel size for depthwise convolutions
        use_dw1: Whether to use the first depthwise convolution
        use_dw2: Whether to use the second depthwise convolution
        block_type: Type of the block ('IB', 'ConvNext', 'ExtraDW', or 'FFN')
        kernel_initializer: Initializer for the convolution kernels
        kernel_regularizer: Regularizer for the convolution kernels
    """

    def __init__(
            self,
            filters: int,
            expansion_factor: int = 4,
            stride: int = 1,
            kernel_size: int = 3,
            use_dw1: bool = False,
            use_dw2: bool = True,
            block_type: str = 'IB',
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.expansion_factor = expansion_factor
        self.stride = stride
        self.kernel_size = kernel_size
        self.use_dw1 = use_dw1
        self.use_dw2 = use_dw2
        self.block_type = block_type
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.expanded_filters = filters * expansion_factor

        # Initialize layer attributes to None - will be built in build()
        self.expand_conv = None
        self.bn1 = None
        self.activation = None
        self.dw1 = None
        self.bn_dw1 = None
        self.dw2 = None
        self.bn_dw2 = None
        self.project_conv = None
        self.bn2 = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the layer weights and sublayers.

        Args:
            input_shape: Shape of the input tensor
        """
        # Store for serialization
        self._build_input_shape = input_shape

        # Layer configurations
        conv_config = {
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "padding": "same",
            "use_bias": False
        }

        # Build sublayers following proper normalization order: Conv/Linear -> Norm -> Activation
        self.expand_conv = keras.layers.Conv2D(self.expanded_filters, 1, **conv_config)
        self.bn1 = keras.layers.BatchNormalization()
        self.activation = keras.layers.ReLU()

        if self.use_dw1:
            self.dw1 = keras.layers.DepthwiseConv2D(
                self.kernel_size,
                self.stride,
                **conv_config
            )
            self.bn_dw1 = keras.layers.BatchNormalization()

        if self.use_dw2:
            self.dw2 = keras.layers.DepthwiseConv2D(
                self.kernel_size,
                1,
                **conv_config
            )
            self.bn_dw2 = keras.layers.BatchNormalization()

        self.project_conv = keras.layers.Conv2D(self.filters, 1, **conv_config)
        self.bn2 = keras.layers.BatchNormalization()

        # Build sublayers explicitly
        self.expand_conv.build(input_shape)
        self.bn1.build(input_shape[:-1] + (self.expanded_filters,))

        if self.use_dw1:
            dw1_input_shape = input_shape[:-1] + (self.expanded_filters,)
            self.dw1.build(dw1_input_shape)
            dw1_output_shape = self._compute_conv_output_shape(dw1_input_shape, self.stride)
            self.bn_dw1.build(dw1_output_shape)

        if self.use_dw2:
            dw2_input_shape = input_shape[:-1] + (self.expanded_filters,)
            if self.use_dw1:
                dw2_input_shape = dw1_output_shape
            self.dw2.build(dw2_input_shape)
            self.bn_dw2.build(dw2_input_shape)

        project_input_shape = input_shape[:-1] + (self.expanded_filters,)
        self.project_conv.build(project_input_shape)
        self.bn2.build(input_shape[:-1] + (self.filters,))

        super().build(input_shape)

    def _compute_conv_output_shape(self, input_shape: Tuple[int, ...], stride: int) -> Tuple[int, ...]:
        """Compute output shape after convolution with given stride.

        Args:
            input_shape: Input shape
            stride: Convolution stride

        Returns:
            Output shape
        """
        if input_shape[1] is None or input_shape[2] is None:
            return input_shape

        height = input_shape[1] // stride if input_shape[1] is not None else None
        width = input_shape[2] // stride if input_shape[2] is not None else None
        return input_shape[:1] + (height, width) + input_shape[3:]

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the UIB block.

        Args:
            inputs: Input tensor
            training: Whether in training mode

        Returns:
            Output tensor
        """
        x = self.expand_conv(inputs)
        x = self.bn1(x, training=training)
        x = self.activation(x)

        if self.use_dw1:
            x = self.dw1(x)
            x = self.bn_dw1(x, training=training)
            x = self.activation(x)

        if self.use_dw2:
            x = self.dw2(x)
            x = self.bn_dw2(x, training=training)
            x = self.activation(x)

        x = self.project_conv(x)
        x = self.bn2(x, training=training)

        # Residual connection
        if self.stride == 1 and ops.shape(inputs)[-1] == self.filters:
            return inputs + x
        return x

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input

        Returns:
            Output shape
        """
        input_shape_list = list(input_shape)

        # Apply stride to spatial dimensions
        if self.stride > 1:
            if input_shape_list[1] is not None:
                input_shape_list[1] = input_shape_list[1] // self.stride
            if input_shape_list[2] is not None:
                input_shape_list[2] = input_shape_list[2] // self.stride

        # Update channel dimension
        input_shape_list[-1] = self.filters

        return tuple(input_shape_list)

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration
        """
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "expansion_factor": self.expansion_factor,
            "stride": self.stride,
            "kernel_size": self.kernel_size,
            "use_dw1": self.use_dw1,
            "use_dw2": self.use_dw2,
            "block_type": self.block_type,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        Returns:
            Dictionary containing the build configuration
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------