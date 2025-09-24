"""Implements the depthwise separable convolution block, a core of MobileNet.

Architecture and Core Concepts:

A standard convolution operation performs spatial filtering and channel
mixing in a single, monolithic step. A depthwise separable convolution
decomposes this into two sequential steps:

1.  **Depthwise Convolution (Spatial Filtering):** In the first step, a
    single spatial filter (e.g., 3x3) is applied independently to *each*
    input channel. This step learns spatial patterns and features, such as
    edges or textures, within each channel individually. It does not combine
    or mix information across different channels.

2.  **Pointwise Convolution (Channel Mixing):** The second step uses a 1x1
    convolution to project the features from the depthwise step onto a new
    channel space. This operation is purely a linear combination of the
    channels at each spatial location. Its function is to mix the
    spatially-filtered information from the previous step to generate new,
    rich features.

By separating these two functions, the model can learn representations far
more efficiently. The intuition is that spatial correlations and
cross-channel correlations are sufficiently independent that they do not need
to be learned simultaneously in a large, multidimensional kernel.

Mathematical Foundation:

The efficiency gain comes from the dramatic reduction in parameters. For a
standard 3x3 convolution with `C_in` input channels and `C_out` output
channels, the number of parameters is `3 * 3 * C_in * C_out`.

In contrast, a depthwise separable convolution has:
-   `K * K * C_in` parameters for the depthwise step.
-   `1 * 1 * C_in * C_out` parameters for the pointwise step.

The ratio of reduction is approximately `(K*K + C_out) / (K*K * C_out)`, which
for a reasonable number of output channels, results in an ~8-9x reduction in
both parameters and computational cost, with only a small, often negligible,
loss in accuracy.

References:

While the concept of separable convolutions has existed for some time, its
application as a core component of modern, efficient deep neural networks
was popularized by the following seminal works:

-   Howard, A. G., et al. (2017). "MobileNets: Efficient Convolutional
    Neural Networks for Mobile Vision Applications." This paper introduced
    the MobileNetV1 architecture, which demonstrated the remarkable
    effectiveness of depthwise separable convolutions for creating small,
    fast, and accurate models for mobile devices.
-   Chollet, F. (2017). "Xception: Deep Learning with Depthwise Separable
    Convolutions." This work further explored the idea, proposing that a
    stack of depthwise separable convolution blocks could outperform even
    large-scale architectures like Inception V3 by making more efficient
    use of model parameters.

"""

import keras
from typing import Optional, Union, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.activations import create_activation_layer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DepthwiseSeparableBlock(keras.layers.Layer):
    """
    Configurable depthwise separable convolution block.

    This block implements the core building block of MobileNetV1, which decomposes
    standard convolution into two separate layers for computational efficiency:
    depthwise convolution (spatial filtering) followed by pointwise convolution
    (channel mixing). This drastically reduces the number of parameters and
    computational cost compared to standard convolutions.

    **Intent**: Provide an efficient convolutional building block for mobile and
    edge device deployment, reducing parameters by ~8-9x compared to standard
    convolution while maintaining comparable accuracy. Now with configurable
    activation and normalization layers for maximum flexibility.

    **Architecture**:
    ```
    Input(shape=[batch, height, width, channels])
           ↓
    DepthwiseConv2D(K×K, stride) - Spatial filtering per channel
           ↓
    Normalization → Activation
           ↓
    Conv2D(1×1) - Pointwise/channel mixing
           ↓
    Normalization → Activation
           ↓
    Output(shape=[batch, new_height, new_width, filters])
    ```

    **Mathematical Operations**:
    1. **Depthwise**: Each input channel convolved with its own K×K kernel
       - Parameters: channels × K × K
    2. **Pointwise**: 1×1 convolution to mix channels
       - Parameters: channels × filters

    Total parameters: channels × (K × K + filters) vs standard: channels × filters × K × K

    Args:
        filters: Integer, number of output filters (channels). Must be positive.
            This determines the output channel dimension.
        depthwise_kernel_size: Integer or tuple, kernel size for the depthwise convolution.
            Controls the spatial filtering window size. Defaults to 3 (3x3 convolution).
        stride: Integer, stride for the depthwise convolution. Controls spatial
            downsampling. Common values: 1 (no downsampling) or 2 (2x downsampling).
            Defaults to 1.
        block_id: Integer, unique identifier for the block used in layer naming.
            Helps identify layers in large models. Defaults to 0.
        kernel_initializer: String name or Initializer instance for weight initialization.
            Applies to both depthwise and pointwise convolution kernels.
            Defaults to 'he_normal'.
        kernel_regularizer: Optional Regularizer instance for weight regularization.
            Applies to both depthwise and pointwise convolution kernels.
            Defaults to None.
        normalization_type: String specifying the normalization layer type.
            Supported types: 'batch_norm', 'layer_norm', 'rms_norm', 'zero_centered_rms_norm',
            'band_rms', 'global_response_norm', etc. Defaults to 'batch_norm'.
        activation_type: String specifying the activation function type.
            Supported types: 'relu', 'gelu', 'mish', 'hard_swish', 'silu', etc.
            Defaults to 'relu'.
        normalization_kwargs: Optional dictionary of arguments to pass to the
            normalization layer factory. Defaults to {}.
        activation_kwargs: Optional dictionary of arguments to pass to the
            activation layer factory. Defaults to {}.
        **kwargs: Additional arguments for Layer base class (name, trainable, etc.).

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        4D tensor with shape: `(batch_size, new_height, new_width, filters)`
        Where new_height = height // stride, new_width = width // stride
        (with padding adjustments)

    Attributes:
        depthwise_conv: DepthwiseConv2D layer for spatial filtering.
        depthwise_norm: Normalization layer after depthwise convolution.
        depthwise_activation: Activation layer after depthwise normalization.
        pointwise_conv: Conv2D layer for channel mixing.
        pointwise_norm: Normalization layer after pointwise convolution.
        pointwise_activation: Activation layer after pointwise normalization.

    Example:
        ```python
        # Basic depthwise separable block (backwards compatible)
        block = DepthwiseSeparableBlock(filters=64, stride=1, block_id=1)
        inputs = keras.Input(shape=(224, 224, 32))
        outputs = block(inputs)  # Shape: (batch, 224, 224, 64)

        # Configurable block with modern components
        modern_block = DepthwiseSeparableBlock(
            filters=128,
            depthwise_kernel_size=5,  # 5x5 depthwise filters
            stride=2,  # Spatial downsampling
            normalization_type='layer_norm',
            activation_type='gelu',
            block_id=2
        )

        # With custom layer parameters
        advanced_block = DepthwiseSeparableBlock(
            filters=256,
            normalization_type='rms_norm',
            activation_type='mish',
            normalization_kwargs={'epsilon': 1e-5, 'use_scale': True},
            kernel_regularizer=keras.regularizers.L2(1e-4),
            block_id=3
        )
        ```

    Note:
        This implementation follows the CRITICAL pattern from the Modern Keras 3 guide:
        sub-layers are created in __init__() but explicitly built in build() method
        for robust serialization. Without explicit building, model loading will fail.

        The block is backwards compatible - existing code using the old interface
        will continue to work exactly as before, while new code can take advantage
        of the configurable normalization and activation layers.
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
        """
        Build the layer and all its sub-layers.

        CRITICAL: This method explicitly builds each sub-layer to ensure proper
        weight creation before serialization. Without this, model loading will fail
        with "Layer was never built" errors.

        Args:
            input_shape: Shape tuple (including batch dimension).
                Expected shape: (batch_size, height, width, channels)
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
        """
        Forward pass through the depthwise separable block.

        Args:
            inputs: Input tensor of shape (batch, height, width, channels).
            training: Boolean or None. If True, layers like BatchNorm use
                training mode. If None, uses keras.backend.learning_phase().

        Returns:
            Output tensor of shape (batch, new_height, new_width, filters).
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
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple (including batch dimension).

        Returns:
            Output shape tuple (batch_size, new_height, new_width, filters).
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
        """
        Return configuration for serialization.

        Returns:
            Dictionary containing all constructor parameters needed
            to recreate this layer.
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