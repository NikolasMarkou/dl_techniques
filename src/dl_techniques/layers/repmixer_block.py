"""
A RepMixer block, an efficient feature-mixing architecture.

This layer serves as a highly efficient, convolution-based alternative to the
self-attention mechanism found in Vision Transformers. It is designed to model
the two fundamental types of interactions in feature maps—spatial and
channel-wise—using separate, specialized components. The core design principle
is to decouple the mixing of information across spatial locations ("token
mixing") from the mixing of information across feature channels ("channel
mixing"), achieving strong performance with a fraction of the computational
cost of standard self-attention.

Architecture and Core Concepts:

The RepMixer block is a residual block composed of two main sub-layers, each
preceded by a normalization layer (typically LayerNorm):

1.  **Token Mixing (Spatial Interaction):** This component is responsible for
    propagating information across the spatial dimensions (height and width) of
    the feature map. Instead of the computationally expensive all-to-all
    comparison of self-attention, it employs a simple yet effective sequence
    of depthwise convolutions. A 3x3 depthwise convolution captures local
    spatial context, followed by a 1x1 depthwise convolution (equivalent to a
    per-channel linear layer). This design allows the block to efficiently
    model spatial relationships within each feature channel independently.

2.  **Channel Mixing (Feature Interaction):** This component operates on a
    per-pixel basis, mixing information across the feature channels. It uses
    an MLP structure, implemented with 1x1 convolutions, which is a standard
    and effective technique for learning complex, non-linear interactions
    between different feature maps. Typically, this MLP follows an inverted
    bottleneck design, expanding the channel dimension before projecting it
    back down.

The separation of these two mixing operations is a key architectural choice.
It is based on the hypothesis that spatial and feature-wise correlations can
be learned effectively in a factorized manner, avoiding the quadratic
complexity of self-attention while retaining much of its representational
power.

Mathematical Foundation:

The efficiency of the RepMixer block stems from its reliance on
computationally cheap operations. The token mixer's use of depthwise
convolutions is particularly important. A depthwise convolution has a
computational cost that is linear with respect to the number of channels, as
opposed to a standard convolution whose cost is quadratic.

The overall structure is a direct application of the "mixer" paradigm, where
the model alternates between operations on spatial dimensions and channel
dimensions:
-   `Y = X + TokenMixer(Norm(X))`
-   `Z = Y + ChannelMixer(Norm(Y))`

This design avoids the `O(N^2 * C)` complexity of self-attention (where `N` is
the number of tokens/pixels) and replaces it with operations that are linear in
`N`, making it highly scalable to high-resolution images.

References:

The design of this block is primarily based on:
-   "RepMixer: Representation Mixing for Efficient Vision Transformers" by
    Che et al.

It belongs to a broader family of "mixer" architectures that have explored
alternatives to self-attention, including:
-   Tolstikhin, I. O., et al. (2021). "MLP-Mixer: An all-MLP Architecture for
    Vision." This paper popularized the concept of explicitly separating token
    and channel mixing.
-   Trockman, A., & Kolter, J. Z. (2022). "Patches Are All You Need? (ConvMixer)"
    which proposed a similar, purely convolutional approach.

The channel mixer's inverted bottleneck structure is a well-established pattern
from efficient CNNs:
-   Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear
    Bottlenecks."

"""

import keras
from typing import Optional, Union, Tuple, Dict, Any
from keras import layers, initializers, regularizers, activations

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from .mobile_one_block import MobileOneBlock

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class RepMixerBlock(keras.layers.Layer):
    """RepMixer block for efficient spatial and channel feature mixing.

    This residual block decouples spatial token mixing from channel mixing,
    replacing the quadratic self-attention mechanism with linear-cost
    depthwise convolutions. Token mixing uses a ``3x3`` depthwise
    convolution followed by a ``1x1`` depthwise convolution with batch
    normalisation, while channel mixing employs an inverted-bottleneck MLP
    implemented with ``1x1`` pointwise convolutions. Each sub-block is
    preceded by normalisation and connected through a residual shortcut:
    ``Y = X + TokenMixer(Norm(X))``,
    ``Z = Y + ChannelMixer(Norm(Y))``.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input [B, H, W, C]             │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Norm1 → Token Mixer            │
        │  DWConv3x3 → BN → Act →        │
        │  DWConv1x1 → BN                 │
        └──────────────┬───────────────────┘
                       │ + residual
                       ▼
        ┌──────────────────────────────────┐
        │  Norm2 → Channel Mixer          │
        │  Conv1x1(expand) → Act →        │
        │  Conv1x1(project)               │
        └──────────────┬───────────────────┘
                       │ + residual
                       ▼
        ┌──────────────────────────────────┐
        │  Output [B, H, W, C]            │
        └──────────────────────────────────┘

    :param dim: Input and output feature dimension. Must be positive.
    :type dim: int
    :param kernel_size: Kernel size for depthwise convolutions. Must be
        positive and odd.
    :type kernel_size: int
    :param expansion_ratio: Channel expansion ratio for channel mixer.
    :type expansion_ratio: float
    :param dropout_rate: Dropout rate for both mixers.
    :type dropout_rate: float
    :param activation: Activation function name or callable.
    :type activation: Union[str, callable]
    :param use_layer_norm: If ``True`` use LayerNorm, else BatchNorm.
    :type use_layer_norm: bool
    :param kernel_initializer: Initializer for convolution kernels.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Initializer for bias terms.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernels.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for biases.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any"""

    def __init__(
            self,
            dim: int,
            kernel_size: int = 3,
            expansion_ratio: float = 4.0,
            dropout_rate: float = 0.0,
            activation: Union[str, callable] = 'gelu',
            use_layer_norm: bool = True,
            kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be positive and odd, got {kernel_size}")
        if expansion_ratio <= 0:
            raise ValueError(f"expansion_ratio must be positive, got {expansion_ratio}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store configuration
        self.dim = dim
        self.kernel_size = kernel_size
        self.expansion_ratio = expansion_ratio
        self.dropout_rate = dropout_rate
        self.activation = activations.get(activation)
        self.use_layer_norm = use_layer_norm
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # Calculate expanded dimension for channel mixing
        self.expanded_dim = int(dim * expansion_ratio)

        # CREATE normalization layers
        if use_layer_norm:
            self.norm1 = layers.LayerNormalization(name='token_norm')
            self.norm2 = layers.LayerNormalization(name='channel_norm')
        else:
            self.norm1 = layers.BatchNormalization(name='token_norm')
            self.norm2 = layers.BatchNormalization(name='channel_norm')

        # CREATE token mixer (spatial mixing with depthwise convolutions)
        self.token_mixer = keras.Sequential([
            layers.DepthwiseConv2D(
                kernel_size=kernel_size,
                padding='same',
                use_bias=False,
                depthwise_initializer=self.kernel_initializer,
                depthwise_regularizer=self.kernel_regularizer,
                name='token_dw_conv1'
            ),
            layers.BatchNormalization(name='token_bn1'),
            layers.Activation(self.activation, name='token_act1'),
            layers.DepthwiseConv2D(
                kernel_size=1,
                padding='same',
                use_bias=True,
                depthwise_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                depthwise_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='token_dw_conv2'
            ),
            layers.BatchNormalization(name='token_bn2'),
        ], name='token_mixer')

        # CREATE channel mixer (feature mixing with pointwise convolutions)
        channel_layers = [
            layers.Conv2D(
                filters=self.expanded_dim,
                kernel_size=1,
                padding='same',
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='channel_expand'
            ),
            layers.Activation(self.activation, name='channel_act'),
        ]

        # Add dropout if specified
        if dropout_rate > 0.0:
            channel_layers.append(layers.Dropout(dropout_rate, name='channel_dropout'))

        channel_layers.append(
            layers.Conv2D(
                filters=dim,
                kernel_size=1,
                padding='same',
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='channel_project'
            )
        )

        self.channel_mixer = keras.Sequential(channel_layers, name='channel_mixer')

        # Dropout for token mixer
        if dropout_rate > 0.0:
            self.token_dropout = layers.Dropout(dropout_rate, name='token_dropout')
            self.channel_dropout_final = layers.Dropout(dropout_rate, name='channel_dropout_final')
        else:
            self.token_dropout = None
            self.channel_dropout_final = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all sub-layers.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]"""
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input, got {len(input_shape)}D")

        if input_shape[-1] != self.dim:
            raise ValueError(
                f"Input channels ({input_shape[-1]}) must match dim ({self.dim})"
            )

        # Build normalization layers
        self.norm1.build(input_shape)
        self.norm2.build(input_shape)

        # Build token mixer
        self.token_mixer.build(input_shape)

        # Build channel mixer
        self.channel_mixer.build(input_shape)

        # Build dropout layers if present
        if self.token_dropout is not None:
            self.token_dropout.build(input_shape)
        if self.channel_dropout_final is not None:
            self.channel_dropout_final.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the RepMixer block.

        :param inputs: Input tensor ``(batch, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Output tensor with same shape as input.
        :rtype: keras.KerasTensor"""
        # Token mixing with residual connection
        x_norm1 = self.norm1(inputs, training=training)
        token_mixed = self.token_mixer(x_norm1, training=training)

        if self.token_dropout is not None:
            token_mixed = self.token_dropout(token_mixed, training=training)

        x = inputs + token_mixed

        # Channel mixing with residual connection
        x_norm2 = self.norm2(x, training=training)
        channel_mixed = self.channel_mixer(x_norm2, training=training)

        if self.channel_dropout_final is not None:
            channel_mixed = self.channel_dropout_final(channel_mixed, training=training)

        x = x + channel_mixed

        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape (identical to input).

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]"""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]"""
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'kernel_size': self.kernel_size,
            'expansion_ratio': self.expansion_ratio,
            'dropout_rate': self.dropout_rate,
            'activation': activations.serialize(self.activation),
            'use_layer_norm': self.use_layer_norm,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ConvolutionalStem(keras.layers.Layer):
    """Convolutional stem for FastVLM using MobileOne blocks.

    This layer forms the initial feature extraction stage by applying three
    successive MobileOne blocks that progressively downsample the spatial
    dimensions by a factor of 4 while mapping to a consistent channel
    depth. The first two blocks use ``3x3`` kernels with stride 2 for
    spatial reduction; the third uses a ``1x1`` kernel at stride 1 for
    channel refinement. All blocks support structural reparameterisation
    for efficient inference.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input [H, W, 3]                │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  MobileOneBlock(k=3, s=2)       │
        │  → [H/2, W/2, out_channels]     │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  MobileOneBlock(k=3, s=2, dw)   │
        │  → [H/4, W/4, out_channels]     │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  MobileOneBlock(k=1, s=1)       │
        │  → [H/4, W/4, out_channels]     │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Output [H/4, W/4, out_channels]│
        └──────────────────────────────────┘

    :param out_channels: Number of output channels for all blocks.
    :type out_channels: int
    :param use_se: Whether to use Squeeze-and-Excitation.
    :type use_se: bool
    :param activation: Activation function for all blocks.
    :type activation: Union[str, callable]
    :param kernel_initializer: Initializer for convolution kernels.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernels.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any"""

    def __init__(
            self,
            out_channels: int,
            use_se: bool = False,
            activation: Union[str, callable] = 'gelu',
            kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")

        # Store configuration
        self.out_channels = out_channels
        self.use_se = use_se
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

        # CREATE stem blocks
        self.blocks = [
            MobileOneBlock(
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                use_se=use_se,
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='stem_block_1'
            ),
            MobileOneBlock(
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                use_se=use_se,
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='stem_block_2_dw'
            ),
            MobileOneBlock(
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                use_se=use_se,
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='stem_block_3_pw'
            )
        ]

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all stem blocks.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]"""
        current_shape = input_shape

        for block in self.blocks:
            block.build(current_shape)
            current_shape = block.compute_output_shape(current_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the stem blocks.

        :param inputs: Input image tensor.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Feature tensor after 4x spatial downsampling.
        :rtype: keras.KerasTensor"""
        x = inputs
        for block in self.blocks:
            x = block(x, training=training)
        return x

    def reparameterize(self) -> None:
        """Reparameterize all blocks in the stem for efficient inference.

        Fuses multi-branch training topology into single-branch for
        deployment."""
        logger.info(f"Reparameterizing ConvolutionalStem with {len(self.blocks)} blocks")
        for i, block in enumerate(self.blocks):
            try:
                block.reparameterize()
                logger.info(f"Successfully reparameterized stem block {i}")
            except Exception as e:
                logger.warning(f"Failed to reparameterize stem block {i}: {e}")

    def reset_reparameterization(self) -> None:
        """Reset all blocks to training mode."""
        logger.info("Resetting ConvolutionalStem reparameterization")
        for block in self.blocks:
            block.reset_reparameterization()

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape through all blocks.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]"""
        current_shape = input_shape
        for block in self.blocks:
            current_shape = block.compute_output_shape(current_shape)
        return current_shape

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]"""
        config = super().get_config()
        config.update({
            'out_channels': self.out_channels,
            'use_se': self.use_se,
            'activation': self.activation,
            'kernel_initializer': initializers.serialize(
                initializers.get(self.kernel_initializer)
            ),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------
