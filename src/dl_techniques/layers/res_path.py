"""
ResPath Implementation for ACC-UNet Skip Connections.

This module implements the ResPath layer, which provides modified skip connections
with residual blocks to reduce semantic gap between encoder and decoder features.
"""

import keras
from typing import Optional, Union, Tuple, Any

from .squeeze_excitation import SqueezeExcitation


class ResPath(keras.layers.Layer):
    """
    Residual Path layer for improved skip connections.

    This layer implements a series of residual blocks along the skip connection
    path to reduce the semantic gap between encoder and decoder features.
    Each residual block consists of:
    1. 3x3 convolution
    2. Batch normalization
    3. Squeeze-Excitation
    4. Residual connection

    The number of residual blocks is typically set based on the level difference
    between encoder and decoder (deeper levels use more blocks).

    Args:
        channels: Number of channels (kept constant throughout).
        num_blocks: Number of residual blocks to apply.
        kernel_initializer: Initializer for convolution kernels.
        bias_initializer: Initializer for bias vectors.
        kernel_regularizer: Regularizer for convolution kernels.
        bias_regularizer: Regularizer for bias vectors.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        4D tensor with shape (batch_size, height, width, channels).

    Output shape:
        4D tensor with shape (batch_size, height, width, channels).

    Example:
        ```python
        # Basic usage - 4 blocks for deepest skip connection
        res_path = ResPath(channels=32, num_blocks=4)

        # Fewer blocks for shallower connections
        res_path = ResPath(channels=64, num_blocks=2)

        # Custom initialization
        res_path = ResPath(
            channels=128,
            num_blocks=3,
            kernel_initializer='he_normal'
        )
        ```

    Note:
        The channel count remains constant throughout the ResPath.
        This layer is typically used in U-Net skip connections to
        process encoder features before concatenating with decoder features.
    """

    def __init__(
            self,
            channels: int,
            num_blocks: int,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.channels = channels
        self.num_blocks = num_blocks
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Validate parameters
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        # Will be initialized in build()
        self.conv_blocks = []
        self.bn_blocks = []
        self.se_blocks = []
        self.final_bn = None
        self.activation = None
        self.final_se = None

        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the residual blocks."""
        self._build_input_shape = input_shape

        # Create residual blocks
        self.conv_blocks = []
        self.bn_blocks = []
        self.se_blocks = []

        for i in range(self.num_blocks):
            # 3x3 Convolution
            conv = keras.layers.Conv2D(
                filters=self.channels,
                kernel_size=3,
                padding='same',
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'conv_block_{i}'
            )
            self.conv_blocks.append(conv)

            # Batch Normalization
            bn = keras.layers.BatchNormalization(name=f'bn_block_{i}')
            self.bn_blocks.append(bn)

            # Squeeze-Excitation
            se = SqueezeExcitation(
                reduction_ratio=0.25,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'se_block_{i}'
            )
            self.se_blocks.append(se)

        # Final batch normalization
        self.final_bn = keras.layers.BatchNormalization(name='final_bn')

        # Activation
        self.activation = keras.layers.LeakyReLU(negative_slope=0.01, name='activation')

        # Final squeeze-excitation
        self.final_se = SqueezeExcitation(
            reduction_ratio=0.25,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='final_se'
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass computation."""
        x = inputs

        # Apply residual blocks
        for i in range(self.num_blocks):
            # Residual block: conv -> bn -> se -> activation + residual
            residual = x

            x = self.conv_blocks[i](x)
            x = self.bn_blocks[i](x, training=training)
            x = self.se_blocks[i](x)
            x = self.activation(x)

            # Add residual connection
            x = x + residual

        # Final processing
        x = self.final_se(x)
        x = self.activation(x)
        x = self.final_bn(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return input_shape  # Shape remains unchanged

    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'num_blocks': self.num_blocks,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    def get_build_config(self) -> dict:
        """Get build configuration."""
        return {
            'input_shape': self._build_input_shape,
        }

    def build_from_config(self, config: dict) -> None:
        """Build from configuration."""
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])