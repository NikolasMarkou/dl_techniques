"""
Convolutional Block Attention Module (CBAM) Implementation.

This module implements the CBAM attention mechanism as described in:
'CBAM: Convolutional Block Attention Module' (Woo et al., 2018)
https://arxiv.org/abs/1807.06521v2

The implementation consists of three main components:
    - ChannelAttention: Implements channel-wise attention mechanism
    - SpatialAttention: Implements spatial attention mechanism
    - CBAM: Combines both attention mechanisms into a single module

Example:
    >>> cbam = CBAM(channels=64, ratio=8)
    >>> refined_features = cbam(input_features)
"""

import keras
import tensorflow as tf
from typing import Optional, Union
from keras.api import layers, regularizers, initializers

# ---------------------------------------------------------------------


class ChannelAttention(keras.layers.Layer):
    """Channel attention module of CBAM.

    This module applies channel-wise attention by using both max-pooling
    and average-pooling features, followed by a shared MLP network.

    Args:
        channels: Number of input channels.
        ratio: Reduction ratio for the shared MLP.
        kernel_initializer: Initializer for the dense layer kernels.
        kernel_regularizer: Regularizer function for the dense layer kernels.
        use_bias: Whether to include bias in dense layers.
    """

    def __init__(
            self,
            channels: int,
            ratio: int = 8,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            use_bias: bool = False
    ) -> None:
        super().__init__()
        self.channels = channels
        self.ratio = ratio

        self.shared_mlp = keras.Sequential([
            layers.Dense(
                channels // ratio,
                activation='relu',
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer
            ),
            layers.Dense(
                channels,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer
            )
        ])

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply channel attention to input tensor.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).

        Returns:
            Channel attention map of shape (batch_size, 1, 1, channels).
        """
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)

        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)

        return tf.sigmoid(avg_out + max_out)

# ---------------------------------------------------------------------


class SpatialAttention(keras.layers.Layer):
    """Spatial attention module of CBAM.

    This module applies spatial attention using channel-wise pooling
    followed by a convolution operation.

    Args:
        kernel_size: Size of the convolution kernel.
        kernel_initializer: Initializer for the convolution kernels.
        kernel_regularizer: Regularizer function for the convolution kernels.
        use_bias: Whether to include bias in convolution layer.
    """

    def __init__(
            self,
            kernel_size: int = 7,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            use_bias: bool = True
    ) -> None:
        super().__init__()
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            padding='same',
            activation='sigmoid',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            use_bias=use_bias
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply spatial attention to input tensor.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).

        Returns:
            Spatial attention map of shape (batch_size, height, width, 1).
        """
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)

        return self.conv(concat)

# ---------------------------------------------------------------------


class CBAM(keras.layers.Layer):
    """Convolutional Block Attention Module.

    This module combines channel and spatial attention mechanisms to
    refine feature maps in a sequential manner.

    Args:
        channels: Number of input channels.
        ratio: Reduction ratio for the channel attention module.
        kernel_size: Kernel size for the spatial attention module.
        channel_kernel_initializer: Initializer for channel attention kernels.
        spatial_kernel_initializer: Initializer for spatial attention kernels.
        channel_kernel_regularizer: Regularizer for channel attention kernels.
        spatial_kernel_regularizer: Regularizer for spatial attention kernels.
    """

    def __init__(
            self,
            channels: int,
            ratio: int = 8,
            kernel_size: int = 7,
            channel_kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            spatial_kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            channel_kernel_regularizer: Optional[regularizers.Regularizer] = None,
            spatial_kernel_regularizer: Optional[regularizers.Regularizer] = None
    ) -> None:
        super().__init__()

        self.channel_attention = ChannelAttention(
            channels=channels,
            ratio=ratio,
            kernel_initializer=channel_kernel_initializer,
            kernel_regularizer=channel_kernel_regularizer
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=kernel_size,
            kernel_initializer=spatial_kernel_initializer,
            kernel_regularizer=spatial_kernel_regularizer
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply CBAM attention to input tensor.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).

        Returns:
            Refined feature map of shape (batch_size, height, width, channels).
        """
        channel_refined = inputs * self.channel_attention(inputs)
        refined = channel_refined * self.spatial_attention(channel_refined)
        return refined

# ---------------------------------------------------------------------
