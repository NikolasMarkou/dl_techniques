"""Implements the depthwise separable convolution block, a core of MobileNet.

This layer provides a highly efficient alternative to the standard 2D
convolution layer by factorizing the operation into two distinct, simpler
steps. The design is motivated by the hypothesis that spatial and
cross-channel correlations in convolutional network feature maps can be
decoupled and learned separately. This decomposition leads to a drastic
reduction in computational cost and model parameters, making it a
cornerstone of modern, efficient architectures designed for mobile and
edge devices.

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
-   `3 * 3 * C_in` parameters for the depthwise step.
-   `1 * 1 * C_in * C_out` parameters for the pointwise step.

The ratio of reduction is approximately `(3*3 + C_out) / (3*3 * C_out)`, which
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
from dl_techniques.utils.logger import logger
from typing import Optional, Union, Dict, Any, Tuple

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DepthwiseSeparableBlock(keras.layers.Layer):
    """
    Depthwise separable convolution block.

    This block implements the core building block of MobileNetV1, which decomposes
    standard convolution into two separate layers for computational efficiency:
    depthwise convolution (spatial filtering) followed by pointwise convolution
    (channel mixing). This drastically reduces the number of parameters and
    computational cost compared to standard convolutions.

    **Intent**: Provide an efficient convolutional building block for mobile and
    edge device deployment, reducing parameters by ~8-9x compared to standard
    convolution while maintaining comparable accuracy.

    **Architecture**:
    ```
    Input(shape=[batch, height, width, channels])
           ↓
    DepthwiseConv2D(3×3, stride) - Spatial filtering per channel
           ↓
    BatchNormalization → ReLU
           ↓
    Conv2D(1×1) - Pointwise/channel mixing
           ↓
    BatchNormalization → ReLU
           ↓
    Output(shape=[batch, new_height, new_width, filters])
    ```

    **Mathematical Operations**:
    1. **Depthwise**: Each input channel convolved with its own 3×3 kernel
       - Parameters: channels × 3 × 3
    2. **Pointwise**: 1×1 convolution to mix channels
       - Parameters: channels × filters

    Total parameters: channels × (3 × 3 + filters) vs standard: channels × filters × 3 × 3

    Args:
        filters: Integer, number of output filters (channels). Must be positive.
            This determines the output channel dimension.
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
        **kwargs: Additional arguments for Layer base class (name, trainable, etc.).

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        4D tensor with shape: `(batch_size, new_height, new_width, filters)`
        Where new_height = height // stride, new_width = width // stride
        (with padding adjustments)

    Attributes:
        depthwise_conv: DepthwiseConv2D layer for spatial filtering.
        depthwise_bn: BatchNormalization after depthwise convolution.
        depthwise_relu: ReLU activation after depthwise batch norm.
        pointwise_conv: Conv2D layer for channel mixing.
        pointwise_bn: BatchNormalization after pointwise convolution.
        pointwise_relu: ReLU activation after pointwise batch norm.

    Example:
        ```python
        # Basic depthwise separable block
        block = DepthwiseSeparableBlock(filters=64, stride=1, block_id=1)
        inputs = keras.Input(shape=(224, 224, 32))
        outputs = block(inputs)  # Shape: (batch, 224, 224, 64)

        # Downsampling block (stride=2)
        downsample_block = DepthwiseSeparableBlock(
            filters=128,
            stride=2,  # Spatial downsampling
            block_id=2
        )
        outputs = downsample_block(inputs)  # Shape: (batch, 112, 112, 128)

        # With regularization
        regularized_block = DepthwiseSeparableBlock(
            filters=256,
            stride=1,
            kernel_regularizer=keras.regularizers.L2(1e-4),
            block_id=3
        )
        ```

    Note:
        This implementation follows the CRITICAL pattern from the Modern Keras 3 guide:
        sub-layers are created in __init__() but explicitly built in build() method
        for robust serialization. Without explicit building, model loading will fail.
    """

    def __init__(
            self,
            filters: int,
            stride: int = 1,
            block_id: int = 0,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
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
        self.stride = stride
        self.block_id = block_id
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

        # CREATE all sub-layers in __init__ (they are unbuilt)
        # Depthwise convolution pathway
        self.depthwise_conv = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=stride,
            padding='same',
            use_bias=False,  # Batch norm makes bias redundant
            depthwise_initializer=self.kernel_initializer,
            depthwise_regularizer=self.kernel_regularizer,
            name=f'conv_dw_{block_id}'
        )
        self.depthwise_bn = keras.layers.BatchNormalization(
            name=f'conv_dw_{block_id}_bn'
        )
        self.depthwise_relu = keras.layers.ReLU(
            name=f'conv_dw_{block_id}_relu'
        )

        # Pointwise convolution pathway
        self.pointwise_conv = keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,  # Batch norm makes bias redundant
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=f'conv_pw_{block_id}'
        )
        self.pointwise_bn = keras.layers.BatchNormalization(
            name=f'conv_pw_{block_id}_bn'
        )
        self.pointwise_relu = keras.layers.ReLU(
            name=f'conv_pw_{block_id}_relu'
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

        self.depthwise_bn.build(depthwise_output_shape)
        # ReLU doesn't change shape, and doesn't need explicit building
        # But we compute the shape for clarity
        relu_output_shape = depthwise_output_shape  # ReLU preserves shape

        # 2. Pointwise convolution pathway
        self.pointwise_conv.build(relu_output_shape)

        # Compute shape after pointwise conv
        pointwise_output_shape = self.pointwise_conv.compute_output_shape(relu_output_shape)

        self.pointwise_bn.build(pointwise_output_shape)
        # Final ReLU doesn't need building

        # Always call parent build at the end
        super().build(input_shape)

        logger.debug(
            f"Built DepthwiseSeparableBlock_{self.block_id}: "
            f"input_shape={input_shape} -> output_shape={pointwise_output_shape}"
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
        x = self.depthwise_bn(x, training=training)
        x = self.depthwise_relu(x)

        # Pointwise convolution pathway
        x = self.pointwise_conv(x, training=training)
        x = self.pointwise_bn(x, training=training)
        x = self.pointwise_relu(x)

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
            'stride': self.stride,
            'block_id': self.block_id,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer)
            if self.kernel_regularizer else None,
        })
        return config

# ---------------------------------------------------------------------
