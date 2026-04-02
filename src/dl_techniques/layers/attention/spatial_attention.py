"""
A spatial attention map for convolutional feature maps.

This module implements the spatial attention mechanism from the Convolutional
Block Attention Module (CBAM). It is designed to identify the most
information-rich spatial regions within a feature map. By learning "where"
to focus, it complements channel attention, which learns "what" to focus on.

Architecture:
    The core idea is to first aggregate the rich information spread across
    all channels into a compact and effective spatial descriptor, and then
    use this descriptor to generate a 2D attention map. The architecture
    achieves this in two main steps:

    1.  **Channel Information Aggregation:** The module compresses the
        channel-wise information for each spatial position into two distinct
        2D feature maps. This is done by applying two pooling operations
        along the channel axis:
        -   **Average Pooling:** Creates a map summarizing the average
            features for each spatial location across all channels,
            capturing global context.
        -   **Max Pooling:** Creates a map highlighting the most salient
            feature response for each spatial location, capturing peak
            activation information.

    2.  **Spatial Map Generation:** The two resulting 2D feature maps are
        concatenated along their channel dimension, forming a refined
        spatial descriptor of shape `(H, W, 2)`. A single convolutional
        layer (typically with a large 7x7 kernel) is then applied to this
        concatenated map. This convolution learns to identify important
        spatial regions based on the aggregated channel information. The
        final output is passed through a sigmoid function to produce a
        normalized attention map.

Foundational Mathematics:
    The spatial attention map `M_s` for an input feature map `F` is computed
    using the following formula:

        M_s(F) = σ( f^k ([AvgPool(F); MaxPool(F)]) )

    -   `AvgPool(F)` and `MaxPool(F)` represent average and max pooling
        operations performed along the channel axis, reducing a tensor of
        shape `(H, W, C)` to `(H, W, 1)`.
    -   `[;]` denotes the concatenation of these two maps along the channel
        axis, resulting in a tensor of shape `(H, W, 2)`.
    -   `f^k` represents a 2D convolution with a single filter of kernel
        size `k x k` (e.g., 7x7). This operation effectively acts as a
        spatial feature detector.
    -   `σ` is the sigmoid activation function, which scales the output to
        the range `[0, 1]`, making it suitable for use as a multiplicative
        attention mask.

References:
    - The foundational paper for this module:
      Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). "CBAM:
      Convolutional Block Attention Module". European Conference on
      Computer Vision (ECCV).
"""

import keras
from typing import Optional, Union, Dict, Any, Tuple

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SpatialAttention(keras.layers.Layer):
    """
    Spatial attention module from CBAM that identifies informative spatial regions.

    Implements the spatial attention mechanism from the Convolutional Block
    Attention Module (CBAM). The module compresses channel information via
    parallel average and max pooling along the channel axis, concatenates the
    resulting 2D maps, and applies a convolution with sigmoid activation to
    produce a spatial attention mask via
    ``M_s(F) = sigma(f^{k x k}([AvgPool(F); MaxPool(F)]))``.

    **Architecture Overview:**

    .. code-block:: text

        Input [B, H, W, C]
              │
              ├────────────────────┐
              ▼                    ▼
        ┌──────────────┐   ┌──────────────┐
        │ AvgPool      │   │ MaxPool      │
        │ axis=C       │   │ axis=C       │
        │ → [B,H,W,1]  │   │ → [B,H,W,1]  │
        └──────┬───────┘   └───────┬──────┘
               │                   │
               └──── Concat ───────┘
                       ▼
                 [B, H, W, 2]
                       ▼
               ┌───────────────┐
               │ Conv2D(k×k,1) │
               │ + Sigmoid     │
               └───────┬───────┘
                       ▼
                 [B, H, W, 1]

    :param kernel_size: Size of the convolution kernel. Must be odd and
        positive. Defaults to 7 following the original CBAM paper.
    :type kernel_size: int
    :param kernel_initializer: Initializer for the convolution kernel
        weights. Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for the convolution
        kernel weights. Defaults to ``None``.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param use_bias: Whether to include bias in the convolution layer.
        Defaults to ``True``.
    :type use_bias: bool
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``kernel_size`` is not positive or not odd.
    """

    def __init__(
            self,
            kernel_size: int = 7,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            use_bias: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd for 'same' padding, got {kernel_size}")

        # Store configuration
        self.kernel_size = kernel_size
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias

        # CREATE sub-layer in __init__ following modern Keras 3 pattern
        self.conv = keras.layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding='same',
            activation='sigmoid',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=self.use_bias,
            name='spatial_attention_conv'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and its sub-layers.

        Creates weight variables for the convolution layer based on the
        expected input shape after channel-wise pooling and concatenation.

        :param input_shape: Shape tuple of the input tensor. Expected to be
            ``(batch_size, height, width, channels)``.
        :type input_shape: tuple
        """
        # Build the convolution layer with concatenated pooling features (2 channels)
        # After avg_pool and max_pool concatenation, we have 2 channels
        conv_input_shape = list(input_shape)
        conv_input_shape[-1] = 2  # avg_pool + max_pool channels
        self.conv.build(tuple(conv_input_shape))

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply spatial attention to the input tensor.

        :param inputs: Input tensor of shape
            ``(batch_size, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional attention mask tensor.
        :type attention_mask: keras.KerasTensor or None
        :param training: Whether the layer should behave in training mode
            or inference mode.
        :type training: bool or None
        :return: Spatial attention map of shape
            ``(batch_size, height, width, 1)`` with values in ``[0, 1]``.
        :rtype: keras.KerasTensor
        """
        # Apply channel-wise pooling to compress channel information
        avg_pool = keras.ops.mean(inputs, axis=-1, keepdims=True)  # (B, H, W, 1)
        max_pool = keras.ops.max(inputs, axis=-1, keepdims=True)  # (B, H, W, 1)

        # Concatenate pooled features along channel dimension
        concat = keras.ops.concatenate([avg_pool, max_pool], axis=-1)  # (B, H, W, 2)

        # Apply convolution with sigmoid activation to generate attention map
        attention_map = self.conv(concat, training=training)  # (B, H, W, 1)

        return attention_map

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: Output shape tuple ``(batch_size, height, width, 1)``.
        :rtype: tuple
        """
        # Output has same spatial dimensions but single channel
        output_shape = list(input_shape)
        output_shape[-1] = 1  # Single attention channel
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
        })
        return config

# ---------------------------------------------------------------------
