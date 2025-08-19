"""
Residual Path (ResPath) Layer for Enhanced U-Net Skip Connections.

This layer implements a sophisticated skip connection enhancement mechanism,
inspired by ACC-UNet, to address the semantic gap between encoder and
decoder features in U-Net architectures. By processing encoder features
through a series of residual blocks, it progressively refines them to better
match the semantic level of the decoder.

This implementation follows modern Keras 3 best practices for creating robust,
serializable composite layers.
"""

import keras
from typing import Optional, Union, Tuple, Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .squeeze_excitation import SqueezeExcitation

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ResPath(keras.layers.Layer):
    """Residual Path layer for improving skip connections in U-Net architectures.

    This layer applies a series of residual blocks to features from the encoder
    before they are passed to the decoder via skip connections. This process
    helps bridge the "semantic gap" by refining the encoder features.

    Each residual block consists of a 3x3 convolution, batch normalization,
    a Squeeze-and-Excitation block, an activation, and a residual connection.

    Args:
        channels (int): The number of channels for the input and output features.
            This value is kept constant throughout the ResPath. Must be positive.
        num_blocks (int): The number of residual blocks to apply. Must be positive.
        kernel_initializer (Union[str, keras.initializers.Initializer], optional):
            Initializer for the convolution kernel weights. Defaults to 'glorot_uniform'.
        kernel_regularizer (Optional[keras.regularizers.Regularizer], optional):
            Regularizer function applied to the kernel weights. Defaults to None.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        A 4D tensor with shape `(batch_size, height, width, channels)`.

    Output shape:
        A 4D tensor with the same shape as the input: `(batch_size, height, width, channels)`.

    Attributes:
        conv_blocks (List[keras.layers.Conv2D]): List of convolution layers for each residual block.
        bn_blocks (List[keras.layers.BatchNormalization]): List of batch normalization layers.
        se_blocks (List[SqueezeExcitation]): List of Squeeze-and-Excitation layers.
        final_se (SqueezeExcitation): The final Squeeze-and-Excitation block.
        final_bn (keras.layers.BatchNormalization): The final batch normalization layer.

    Example:
        ```python
        # For a deep-level skip connection with 32 channels and 4 blocks
        input_features = keras.Input(shape=(64, 64, 32))
        respath_layer = ResPath(channels=32, num_blocks=4)
        refined_features = respath_layer(input_features)
        # refined_features.shape will be (None, 64, 64, 32)
        ```

    Raises:
        ValueError: If `channels` or `num_blocks` are not positive integers.
    """

    def __init__(
            self,
            channels: int,
            num_blocks: int,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # === Parameter Validation ===
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        # === Store Configuration ===
        self.channels = channels
        self.num_blocks = num_blocks
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # === CREATE all sub-layers (unbuilt) ===
        # This follows the "Create vs. Build" golden rule.
        # All layers are instantiated here and will be built in the build() method.
        self.conv_blocks: List[keras.layers.Conv2D] = []
        self.bn_blocks: List[keras.layers.BatchNormalization] = []
        self.se_blocks: List[SqueezeExcitation] = []

        for i in range(self.num_blocks):
            self.conv_blocks.append(keras.layers.Conv2D(
                filters=self.channels,
                kernel_size=3,
                padding='same',
                use_bias=False, # Standard practice in blocks with BN
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'conv_block_{i}'
            ))
            self.bn_blocks.append(keras.layers.BatchNormalization(name=f'bn_block_{i}'))
            self.se_blocks.append(SqueezeExcitation(
                reduction_ratio=0.25,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'se_block_{i}'
            ))

        self.activation = keras.layers.LeakyReLU(negative_slope=0.01, name='activation')
        self.final_se = SqueezeExcitation(
            reduction_ratio=0.25,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='final_se'
        )
        self.final_bn = keras.layers.BatchNormalization(name='final_bn')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the residual blocks and all sub-layers.

        This method explicitly builds each sub-layer created in `__init__`,
        which is critical for robust serialization and weight restoration.
        """
        # Since all internal layers maintain the same shape, we can use
        # the initial input_shape to build all of them.
        for conv, bn, se in zip(self.conv_blocks, self.bn_blocks, self.se_blocks):
            conv.build(input_shape)
            bn.build(input_shape)
            se.build(input_shape)

        self.final_se.build(input_shape)
        self.final_bn.build(input_shape)

        # Always call the parent's build() method at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the series of residual blocks."""
        x = inputs

        # Apply residual blocks sequentially
        for i in range(self.num_blocks):
            residual = x
            x = self.conv_blocks[i](x)
            x = self.bn_blocks[i](x, training=training)
            x = self.se_blocks[i](x)
            x = self.activation(x)
            x = keras.layers.add([x, residual])

        # Apply final processing steps
        x = self.final_se(x)
        x = self.activation(x)
        x = self.final_bn(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        return input_shape  # Shape remains unchanged throughout the path

    def get_config(self) -> dict:
        """Returns the layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'num_blocks': self.num_blocks,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------
