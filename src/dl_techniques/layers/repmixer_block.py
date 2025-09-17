"""Implement the RepMixer block, an efficient feature-mixing architecture.

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
    """
    RepMixer block for efficient feature mixing in vision models.

    This layer implements the RepMixer architecture that efficiently combines
    token mixing (spatial interaction) and channel mixing (feature transformation)
    operations. It uses depthwise convolutions for token mixing and pointwise
    convolutions for channel mixing, with residual connections and normalization.

    **Intent**: Provide an efficient alternative to self-attention for vision tasks
    that maintains strong representational capacity while reducing computational cost,
    especially suitable for mobile and edge deployment scenarios.

    **Architecture**:
    ```
    Input(shape=[..., H, W, C])
           ↓
    LayerNorm → RepMixer Token Mixing
           ↓ (residual)
    LayerNorm → RepMixer Channel Mixing
           ↓ (residual)
    Output(shape=[..., H, W, C])
    ```

    **Token Mixing (Spatial)**:
    ```
    x → DWConv(3x3) → BN → GELU → DWConv(1x1) → BN → x
    ```

    **Channel Mixing (Feature)**:
    ```
    x → Conv1x1(expand) → GELU → Conv1x1(project) → x
    ```

    **Mathematical Operations**:
    - **Token Mixing**: Captures spatial relationships using depthwise convolutions
    - **Channel Mixing**: Captures feature relationships using pointwise convolutions
    - **Residual Connections**: Enable gradient flow and feature reuse

    Args:
        dim: Integer, input and output feature dimension. Must be positive.
        kernel_size: Integer, kernel size for depthwise convolutions in token mixing.
            Must be positive and odd. Defaults to 3.
        expansion_ratio: Float, expansion ratio for channel mixing MLP.
            Must be positive. Defaults to 4.0.
        dropout_rate: Float, dropout rate applied in both mixers.
            Must be between 0 and 1. Defaults to 0.0.
        activation: String or callable, activation function to use.
            Defaults to 'gelu'.
        use_layer_norm: Boolean, whether to use LayerNormalization.
            If False, uses BatchNormalization. Defaults to True.
        kernel_initializer: String or initializer, initializer for conv kernels.
            Defaults to 'he_normal'.
        bias_initializer: String or initializer, initializer for bias terms.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for conv kernels.
        bias_regularizer: Optional regularizer for bias terms.
        **kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        4D tensor with same shape as input: `(batch_size, height, width, channels)`

    Attributes:
        token_mixer: Sequential block for spatial token mixing.
        channel_mixer: Sequential block for channel mixing.
        norm1: First normalization layer.
        norm2: Second normalization layer.
        dropout: Dropout layer for regularization.

    Example:
        ```python
        # Basic RepMixer block
        mixer = RepMixerBlock(dim=256)
        inputs = keras.Input(shape=(56, 56, 256))
        outputs = mixer(inputs)  # Shape: (None, 56, 56, 256)

        # With custom parameters
        mixer = RepMixerBlock(
            dim=512,
            kernel_size=5,
            expansion_ratio=6.0,
            dropout_rate=0.1,
            activation='swish'
        )

        # With batch normalization instead of layer norm
        mixer = RepMixerBlock(
            dim=128,
            use_layer_norm=False,
            dropout_rate=0.05
        )

        # In a sequential model
        model = keras.Sequential([
            layers.Conv2D(64, 7, strides=2, padding='same'),
            RepMixerBlock(dim=64),
            RepMixerBlock(dim=64),
            layers.GlobalAveragePooling2D(),
            layers.Dense(1000, activation='softmax')
        ])
        ```

    Note:
        This implementation is optimized for efficiency while maintaining the core
        RepMixer functionality. The token mixing uses depthwise convolutions which
        are hardware-friendly and achieve good spatial modeling capabilities.

    References:
        RepMixer: Representation Mixing for Efficient Vision Transformers
        FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization
    """

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
        """Build the layer and all sub-layers."""
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
        """Forward pass through RepMixer block."""
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
        """Output shape is identical to input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
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
    """
    Convolutional stem for FastVLM using MobileOne blocks.

    This layer creates the initial feature extraction stage of a vision model
    using a sequence of MobileOne blocks with progressively reduced spatial
    dimensions and increased feature depth. It's designed to efficiently
    capture low-level visual features before passing to the main architecture.

    **Intent**: Provide an efficient and effective initial processing stage
    that converts raw images to meaningful feature representations while
    maintaining computational efficiency through structural reparameterization.

    **Architecture**:
    ```
    Input(shape=[H, W, 3])
           ↓
    MobileOneBlock(out_channels, k=3, s=2) → [H/2, W/2, out_channels]
           ↓
    MobileOneBlock(out_channels, k=3, s=2, dw) → [H/4, W/4, out_channels]
           ↓
    MobileOneBlock(out_channels, k=1, s=1) → [H/4, W/4, out_channels]
           ↓
    Output(shape=[H/4, W/4, out_channels])
    ```

    **Design Principles**:
    - Progressive spatial downsampling (4x reduction)
    - Consistent channel dimension throughout
    - Mix of different kernel sizes (3x3, 1x1)
    - Efficient depthwise operations where appropriate

    Args:
        out_channels: Integer, number of output channels for all blocks.
            Must be positive. This determines the feature dimension.
        use_se: Boolean, whether to use Squeeze-and-Excitation in blocks.
            Defaults to False for efficiency.
        activation: String or callable, activation function for all blocks.
            Defaults to 'gelu'.
        kernel_initializer: String or initializer, initializer for conv kernels.
            Defaults to 'he_normal'.
        kernel_regularizer: Optional regularizer for conv kernels.
        **kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, 3)`
        Typically expects RGB images.

    Output shape:
        4D tensor with shape: `(batch_size, height/4, width/4, out_channels)`

    Attributes:
        blocks: List of MobileOne blocks comprising the stem.

    Example:
        ```python
        # Basic usage for 224x224 input
        stem = ConvolutionalStem(out_channels=64)
        inputs = keras.Input(shape=(224, 224, 3))
        features = stem(inputs)  # Shape: (None, 56, 56, 64)

        # With Squeeze-and-Excitation
        stem = ConvolutionalStem(out_channels=96, use_se=True)

        # Custom activation
        stem = ConvolutionalStem(
            out_channels=128,
            activation='swish',
            kernel_initializer='glorot_uniform'
        )

        # In a complete model
        inputs = keras.Input(shape=(224, 224, 3))
        features = ConvolutionalStem(64)(inputs)
        # ... rest of model architecture
        ```

    Note:
        The 4x spatial downsampling is designed to balance between capturing
        fine-grained details and computational efficiency. The stem can be
        reparameterized for inference by calling reparameterize() on each block.
    """

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
        """Build all stem blocks."""
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
        """Forward pass through stem blocks."""
        x = inputs
        for block in self.blocks:
            x = block(x, training=training)
        return x

    def reparameterize(self) -> None:
        """Reparameterize all blocks in the stem for efficient inference."""
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
        """Compute output shape through all blocks."""
        current_shape = input_shape
        for block in self.blocks:
            current_shape = block.compute_output_shape(current_shape)
        return current_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
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
