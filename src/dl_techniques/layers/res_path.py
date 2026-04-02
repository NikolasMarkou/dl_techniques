"""
A residual path to bridge the semantic gap in U-Net skip connections.

This layer provides a sophisticated feature refinement mechanism for the skip
connections in U-Net-like architectures. It addresses a common challenge in
encoder-decoder models known as the "semantic gap." This gap arises because
features from early encoder layers are rich in low-level spatial detail but
semantically weak, while features in the corresponding decoder layer are
semantically strong but spatially coarse. Simply concatenating these disparate
feature maps can lead to a semantic mismatch and hinder model performance.

Architecture and Core Concepts:

The ResPath layer is inserted into the skip connection path. Instead of passing
encoder features directly to the decoder, it first processes them through a
series of `num_blocks` identical residual blocks. The number of blocks is
typically chosen based on the depth of the skip connection; connections from
earlier in the encoder (with a larger semantic gap) receive more residual
blocks.

Each residual block in the path is designed to progressively enrich the
semantic content of the feature maps while preserving their high-resolution
spatial information. A block consists of:

1.  **Convolutional Layer:** A standard 3x3 convolution extracts local
    features.
2.  **Squeeze-and-Excitation (SE) Block:** This is a crucial component for
    adaptive feature recalibration. The SE block learns to model
    interdependencies between channels, selectively amplifying informative
    feature channels while suppressing less useful ones for the given input.
3.  **Residual Connection:** An identity shortcut adds the input of the block
    to its output. This allows for the stable training of a deep stack of
    blocks, ensuring that gradients can flow easily and preventing the
    degradation of feature quality.

By stacking these blocks, the ResPath effectively creates a small, dedicated
convolutional network within the skip connection itself. This internal network
learns to transform the low-level encoder features into a more abstract,
semantically-aligned representation that can be more effectively fused with
the decoder's feature maps.

Mathematical Foundation:

The core of each block is the residual learning principle. If `x` is the
input to a block and `F(x)` is the transformation applied by the convolutional
and SE layers, the output `y` is computed as:
`y = F(x) + x`

This formulation allows the block to easily learn an identity mapping (`F(x)=0`)
if no further refinement is needed, which simplifies the optimization landscape
for very deep models. The Squeeze-and-Excitation block performs a
channel-wise recalibration by computing a scaling factor for each channel based
on global information, making the feature maps more discriminative.

References:

The architectural pattern of using stacked residual blocks to refine skip
connections was notably proposed in:
-   Oktay, O., et al. (2018). "Attention U-Net: Learning Where to Look for
    the Pancreas." While this paper focused on attention gates, the idea of
    processing skip connections gained traction.
-   The specific "ResPath" with Squeeze-and-Excitation is inspired by models
    like ACC-UNet, which integrate advanced attention and context-aware
    modules into the U-Net framework.

The foundational concepts upon which this layer is built are:
-   Ronneberger, O., et al. (2015). "U-Net: Convolutional Networks for
    Biomedical Image Segmentation."
-   He, K., et al. (2016). "Deep Residual Learning for Image Recognition."
-   Hu, J., et al. (2018). "Squeeze-and-Excitation Networks."

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

    **Architecture Overview:**

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
        """Build all residual sub-layers.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]"""
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
        """Forward pass through the series of residual blocks.

        :param inputs: Input tensor ``(batch, H, W, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Refined tensor with same shape as input.
        :rtype: keras.KerasTensor"""
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
        """Compute the output shape (same as input).

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]"""
        return input_shape  # Shape remains unchanged throughout the path

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
