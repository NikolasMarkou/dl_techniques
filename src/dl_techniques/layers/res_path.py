"""
``ResPath`` refines U-Net encoder features before they reach the decoder, to
narrow the semantic gap in a skip connection.

Early encoder features carry fine spatial detail but weak semantics; the
matching decoder features carry strong semantics but coarse spatial detail.
Concatenating them directly mismatches the two. ``ResPath`` closes that gap by
running the skip connection through ``num_blocks`` residual blocks -- each a
3x3 convolution, batch norm, Squeeze-and-Excitation channel recalibration, and
a LeakyReLU activation, with an identity shortcut around the block
(``y = F(x) + x``) -- before the features reach the decoder. Connections from
earlier encoder stages, where the semantic gap is larger, typically use more
blocks.

References:
    - Oktay et al., 2018. Attention U-Net: Learning Where to Look for the
      Pancreas.
    - Ronneberger et al., 2015. U-Net: Convolutional Networks for Biomedical
      Image Segmentation.
    - He et al., 2016. Deep Residual Learning for Image Recognition.
    - Hu et al., 2018. Squeeze-and-Excitation Networks.
"""

import keras
from typing import Optional, Union, Tuple, Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .squeeze_excitation import SqueezeExcitation
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.res_path")
class ResPath(keras.layers.Layer):
    """Residual path for bridging the semantic gap in U-Net skip connections.

    This layer refines encoder features before they reach the decoder by
    passing them through ``num_blocks`` sequential residual blocks. Each
    block applies a ``3x3`` convolution, batch normalisation,
    Squeeze-and-Excitation channel recalibration, and a LeakyReLU
    activation with an identity shortcut:
    ``y = F(x) + x`` where ``F`` is the conv-BN-SE-act pipeline.
    Stacking these blocks progressively enriches the semantic content of
    low-level encoder features while preserving spatial resolution,
    effectively narrowing the representation gap before concatenation
    with decoder features.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input [B, H, W, C]              │
        └──────────────┬───────────────────┘
                       │
                       ▼  (repeat num_blocks times)
        ┌──────────────────────────────────┐
        │  Conv2D 3x3 (same, no bias)      │
        │  BatchNorm → SE → LeakyReLU      │
        │  + residual shortcut             │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Final SE → LeakyReLU → BN       │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Output [B, H, W, C]             │
        └──────────────────────────────────┘

    :param channels: Number of channels (kept constant throughout).
        Must be positive.
    :type channels: int
    :param num_blocks: Number of residual blocks. Must be positive.
    :type num_blocks: int
    :param kernel_initializer: Initializer for convolution kernels.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any"""

    def __init__(
            self,
            channels: int,
            num_blocks: int,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        self.channels = channels
        self.num_blocks = num_blocks
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.conv_blocks: List[keras.layers.Conv2D] = []
        self.bn_blocks: List[keras.layers.BatchNormalization] = []
        self.se_blocks: List[SqueezeExcitation] = []

        for i in range(self.num_blocks):
            self.conv_blocks.append(keras.layers.Conv2D(
                filters=self.channels,
                kernel_size=3,
                padding='same',
                use_bias=False,
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
        """Build all residual sub-layers.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]"""
        # All sub-layers preserve the input shape, so the same shape builds each.
        for conv, bn, se in zip(self.conv_blocks, self.bn_blocks, self.se_blocks):
            conv.build(input_shape)
            bn.build(input_shape)
            se.build(input_shape)

        self.final_se.build(input_shape)
        self.final_bn.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the series of residual blocks.

        :param inputs: Input tensor ``(batch, H, W, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Refined tensor with same shape as input.
        :rtype: keras.KerasTensor"""
        x = inputs

        for i in range(self.num_blocks):
            residual = x
            x = self.conv_blocks[i](x)
            x = self.bn_blocks[i](x, training=training)
            x = self.se_blocks[i](x)
            x = self.activation(x)
            x = keras.layers.add([x, residual])

        x = self.final_se(x)
        x = self.activation(x)
        x = self.final_bn(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape (same as input).

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]"""
        return input_shape

    def get_config(self) -> dict:
        """Return layer configuration for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: dict"""
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'num_blocks': self.num_blocks,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------
