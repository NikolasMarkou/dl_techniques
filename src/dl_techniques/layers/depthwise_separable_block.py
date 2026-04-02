"""
Depthwise separable convolution block, a core building block of MobileNet.

A standard convolution operation performs spatial filtering and channel
mixing in a single, monolithic step. A depthwise separable convolution
decomposes this into two sequential steps: depthwise convolution (spatial
filtering per channel) followed by pointwise convolution (channel mixing
via 1x1 convolution).

The efficiency gain comes from the dramatic reduction in parameters. For a
standard 3x3 convolution with C_in input channels and C_out output channels,
the number of parameters is 3 * 3 * C_in * C_out. In contrast, a depthwise
separable convolution has K * K * C_in + C_in * C_out parameters, resulting
in an approximately 8-9x reduction for typical configurations.

References:
    - Howard, A. G., et al. (2017). "MobileNets: Efficient Convolutional
      Neural Networks for Mobile Vision Applications."
    - Chollet, F. (2017). "Xception: Deep Learning with Depthwise Separable
      Convolutions."
"""

import keras
from typing import Optional, Union, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .norms import create_normalization_layer
from .activations import create_activation_layer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DepthwiseSeparableBlock(keras.layers.Layer):
    """
    Configurable depthwise separable convolution block.

    This block implements the core building block of MobileNetV1, decomposing
    standard convolution into depthwise convolution (spatial filtering per channel)
    followed by pointwise convolution (1x1 channel mixing). This reduces parameters
    by approximately 8-9x compared to standard convolution while maintaining
    comparable accuracy. The total parameter count is
    channels * (K * K + filters) versus channels * filters * K * K for standard
    convolution.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────────────────────┐
        │  Input [batch, height, width, channels]       │
        └──────────────────┬────────────────────────────┘
                           ▼
        ┌───────────────────────────────────────────────┐
        │  DepthwiseConv2D (K x K, stride)              │
        │  Spatial filtering per channel independently  │
        └──────────────────┬────────────────────────────┘
                           ▼
        ┌───────────────────────────────────────────────┐
        │  Normalization ──▶ Activation                 │
        └──────────────────┬────────────────────────────┘
                           ▼
        ┌───────────────────────────────────────────────┐
        │  Conv2D (1 x 1) ── Pointwise / channel mixing │
        └──────────────────┬────────────────────────────┘
                           ▼
        ┌───────────────────────────────────────────────┐
        │  Normalization ──▶ Activation                 │
        └──────────────────┬────────────────────────────┘
                           ▼
        ┌────────────────────────────────────────────────┐
        │  Output [batch, new_height, new_width, filters]│
        └────────────────────────────────────────────────┘

    :param filters: Number of output filters (channels). Must be positive.
    :type filters: int
    :param depthwise_kernel_size: Kernel size for the depthwise convolution.
        Defaults to 3.
    :type depthwise_kernel_size: Union[int, Tuple[int, int]]
    :param stride: Stride for the depthwise convolution. Defaults to 1.
    :type stride: int
    :param block_id: Unique identifier for the block used in layer naming.
        Defaults to 0.
    :type block_id: int
    :param kernel_initializer: Initializer for weight initialization. Defaults to
        'he_normal'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for weight regularization.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param normalization_type: Type of normalization layer. Defaults to 'batch_norm'.
    :type normalization_type: str
    :param activation_type: Type of activation function. Defaults to 'relu'.
    :type activation_type: str
    :param normalization_kwargs: Optional arguments for the normalization layer factory.
    :type normalization_kwargs: Optional[Dict[str, Any]]
    :param activation_kwargs: Optional arguments for the activation layer factory.
    :type activation_kwargs: Optional[Dict[str, Any]]
    :param kwargs: Additional arguments for Layer base class.
    """

    def __init__(
            self,
            filters: int,
            depthwise_kernel_size: Union[int, Tuple[int, int]] = 3,
            stride: int = 1,
            block_id: int = 0,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            normalization_type: str = "batch_norm",
            activation_type: str = "relu",
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            activation_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if block_id < 0:
            raise ValueError(f"block_id must be non-negative, got {block_id}")

        # Store ALL configuration for get_config()
        self.filters = filters
        self.depthwise_kernel_size = depthwise_kernel_size
        self.stride = stride
        self.block_id = block_id
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.normalization_type = normalization_type
        self.activation_type = activation_type
        self.normalization_kwargs = normalization_kwargs or {}
        self.activation_kwargs = activation_kwargs or {}

        # CREATE all sub-layers in __init__ (they are unbuilt)
        # Depthwise convolution pathway
        self.depthwise_conv = keras.layers.DepthwiseConv2D(
            kernel_size=depthwise_kernel_size,
            strides=stride,
            padding='same',
            use_bias=False,  # Normalization makes bias redundant
            depthwise_initializer=self.kernel_initializer,
            depthwise_regularizer=self.kernel_regularizer,
            name=f'conv_dw_{block_id}'
        )

        # Use factory for configurable normalization
        self.depthwise_norm = create_normalization_layer(
            normalization_type=self.normalization_type,
            name=f'conv_dw_{block_id}_norm',
            **self.normalization_kwargs
        )

        # Use factory for configurable activation
        self.depthwise_activation = create_activation_layer(
            activation_type=self.activation_type,
            name=f'conv_dw_{block_id}_act',
            **self.activation_kwargs
        )

        # Pointwise convolution pathway
        self.pointwise_conv = keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,  # Normalization makes bias redundant
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=f'conv_pw_{block_id}'
        )

        # Use factory for configurable normalization
        self.pointwise_norm = create_normalization_layer(
            normalization_type=self.normalization_type,
            name=f'conv_pw_{block_id}_norm',
            **self.normalization_kwargs
        )

        # Use factory for configurable activation
        self.pointwise_activation = create_activation_layer(
            activation_type=self.activation_type,
            name=f'conv_pw_{block_id}_act',
            **self.activation_kwargs
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all its sub-layers.

        Explicitly builds each sub-layer to ensure proper weight creation
        before serialization.

        :param input_shape: Shape tuple (including batch dimension).
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input (batch, height, width, channels), "
                f"got shape with {len(input_shape)} dimensions"
            )

        # Build sub-layers in computational order
        # 1. Depthwise convolution pathway
        self.depthwise_conv.build(input_shape)

        # Compute shape after depthwise conv
        depthwise_output_shape = self.depthwise_conv.compute_output_shape(input_shape)

        self.depthwise_norm.build(depthwise_output_shape)
        # Activation layers don't typically need explicit building, but for consistency
        self.depthwise_activation.build(depthwise_output_shape)

        # Activation preserves shape
        activation_output_shape = depthwise_output_shape

        # 2. Pointwise convolution pathway
        self.pointwise_conv.build(activation_output_shape)

        # Compute shape after pointwise conv
        pointwise_output_shape = self.pointwise_conv.compute_output_shape(activation_output_shape)

        self.pointwise_norm.build(pointwise_output_shape)
        self.pointwise_activation.build(pointwise_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

        logger.debug(
            f"Built DepthwiseSeparableBlock_{self.block_id}: "
            f"input_shape={input_shape} -> output_shape={pointwise_output_shape}, "
            f"kernel_size={self.depthwise_kernel_size}, norm={self.normalization_type}, "
            f"act={self.activation_type}"
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the depthwise separable block.

        :param inputs: Input tensor of shape (batch, height, width, channels).
        :type inputs: keras.KerasTensor
        :param training: Boolean or None indicating training mode.
        :type training: Optional[bool]
        :return: Output tensor of shape (batch, new_height, new_width, filters).
        :rtype: keras.KerasTensor
        """
        # Depthwise convolution pathway
        x = self.depthwise_conv(inputs, training=training)
        x = self.depthwise_norm(x, training=training)
        x = self.depthwise_activation(x)

        # Pointwise convolution pathway
        x = self.pointwise_conv(x, training=training)
        x = self.pointwise_norm(x, training=training)
        x = self.pointwise_activation(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple (including batch dimension).
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple (batch_size, new_height, new_width, filters).
        :rtype: Tuple[Optional[int], ...]
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape, got {len(input_shape)}D"
            )

        batch_size, height, width, _ = input_shape

        # Calculate spatial dimensions after strided convolution
        # With 'same' padding: output_size = ceil(input_size / stride)
        if height is not None:
            new_height = (height + self.stride - 1) // self.stride
        else:
            new_height = None

        if width is not None:
            new_width = (width + self.stride - 1) // self.stride
        else:
            new_width = None

        return (batch_size, new_height, new_width, self.filters)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'depthwise_kernel_size': self.depthwise_kernel_size,
            'stride': self.stride,
            'block_id': self.block_id,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer)
            if self.kernel_regularizer else None,
            'normalization_type': self.normalization_type,
            'activation_type': self.activation_type,
            'normalization_kwargs': self.normalization_kwargs,
            'activation_kwargs': self.activation_kwargs,
        })
        return config

# ---------------------------------------------------------------------
