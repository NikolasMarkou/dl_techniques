"""
Convolutional Block Attention Module (CBAM), a lightweight and
effective attention mechanism for CNNs. CBAM operates on the
principle of inferring attention maps along two separate dimensions, channel
and spatial, and then sequentially applying them to the input feature map for
adaptive feature refinement. The key architectural choice is this sequential
arrangement: channel attention is applied first, followed by spatial attention.
This allows the spatial attention mechanism to operate on features that have
already been recalibrated for channel-wise importance.

The foundational mathematics of CBAM is divided into its two sub-modules:

1.  **Channel Attention (`Mc`):** This module aims to answer "what" is
    meaningful in the input feature map. It aggregates spatial information by
    applying both average-pooling and max-pooling operations across the
    spatial dimensions (H x W), producing two distinct context descriptors.
    These descriptors capture both the average and the most salient features
    across the spatial grid for each channel. Both descriptors are then
    processed by a shared Multi-Layer Perceptron (MLP) with a bottleneck
    structure to efficiently compute the channel attention weights. The outputs
    are merged via element-wise summation and passed through a sigmoid
    function to generate the final channel attention map, which encodes the
    inter-channel relationship of features.

2.  **Spatial Attention (`Ms`):** Following channel refinement, this module
    aims to answer "where" is the most informative region. It first aggregates
    the channel information at each spatial location by applying average-
    pooling and max-pooling along the channel axis. This generates two 2D maps
    that effectively summarize the features across all channels for each
    pixel, highlighting regions with high average and high peak activations.
    These two maps are concatenated and then passed through a standard
    convolutional layer to produce a single 2D spatial attention map. After a
    final sigmoid activation, this map highlights the most salient spatial
-   regions to focus on.

The complete CBAM operation is a sequential multiplication: the input feature
map `F` is first multiplied by the channel attention map `Mc(F)`, and the
result is then multiplied by the spatial attention map `Ms(F')`. This
factorization of attention into two sequential, decoupled modules makes CBAM
lightweight and easily integrable into existing CNN architectures, enhancing
their representational power by allowing the network to learn to selectively
focus on informative features and suppress irrelevant ones.

References:
    - Woo et al., 2018. CBAM: Convolutional Block Attention Module.
      (https://arxiv.org/abs/1807.06521)

"""

import keras
from typing import Optional, Union, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .channel_attention import ChannelAttention
from .spatial_attention import SpatialAttention

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CBAM(keras.layers.Layer):
    """
    Convolutional Block Attention Module for feature refinement.

    This layer implements the CBAM attention mechanism that sequentially applies
    channel and spatial attention to input feature maps. CBAM is designed to
    improve representation power of CNNs by focusing on important features
    while suppressing unnecessary ones.

    The module operates in two sequential stages:
    1. Channel Attention: Computes channel-wise attention weights using global
       average and max pooling followed by a shared MLP
    2. Spatial Attention: Computes spatial attention weights using channel-wise
       pooling followed by a convolutional layer

    Mathematical formulation:
        F' = Ms(F) ⊗ (Mc(F) ⊗ F)

    Where F is input feature, Mc is channel attention, Ms is spatial attention,
    and ⊗ denotes element-wise multiplication.

    Args:
        channels: Integer, number of input channels. Must be positive.
        ratio: Integer, reduction ratio for channel attention MLP.
            Controls the bottleneck size in the shared MLP. Higher values
            reduce parameters but may limit representation capacity.
            Must be positive. Defaults to 8.
        kernel_size: Integer, kernel size for spatial attention convolution.
            Typically uses 7x7 convolution as in the original paper.
            Must be positive and odd. Defaults to 7.
        channel_kernel_initializer: String or Initializer, kernel initializer
            for channel attention layers. Defaults to 'glorot_uniform'.
        spatial_kernel_initializer: String or Initializer, kernel initializer
            for spatial attention layers. Defaults to 'glorot_uniform'.
        channel_kernel_regularizer: Optional Regularizer, kernel regularizer
            for channel attention layers. Defaults to None.
        spatial_kernel_regularizer: Optional Regularizer, kernel regularizer
            for spatial attention layers. Defaults to None.
        channel_use_bias: Boolean, whether to use bias in channel attention
            layers. Channel attention typically doesn't use bias in MLP.
            Defaults to False.
        spatial_use_bias: Boolean, whether to use bias in spatial attention
            convolution. Defaults to True.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor with shape (batch_size, height, width, channels) for
        channels-last data format, or (batch_size, channels, height, width)
        for channels-first data format.

    Output shape:
        Same as input shape. CBAM preserves spatial and channel dimensions
        while refining the feature representations.

    Attributes:
        channel_attention: ChannelAttention sub-module for channel refinement.
        spatial_attention: SpatialAttention sub-module for spatial refinement.

    Example:
        ```python
        # Basic usage for ResNet-style features
        inputs = keras.Input(shape=(56, 56, 256))
        cbam_layer = CBAM(channels=256)
        refined_features = cbam_layer(inputs)

        # Advanced configuration with regularization
        cbam_layer = CBAM(
            channels=512,
            ratio=16,  # Stronger bottleneck
            kernel_size=5,  # Smaller spatial kernel
            channel_kernel_regularizer=keras.regularizers.L2(1e-4),
            spatial_kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # Integration in CNN architecture
        inputs = keras.Input(shape=(224, 224, 3))
        x = keras.layers.Conv2D(64, 3, activation='relu')(inputs)
        x = keras.layers.Conv2D(128, 3, activation='relu')(x)

        # Apply CBAM attention
        x = CBAM(channels=128)(x)

        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        ```

    References:
        - CBAM: Convolutional Block Attention Module, Woo et al., 2018
        - https://arxiv.org/abs/1807.06521v2

    Raises:
        ValueError: If channels is not positive.
        ValueError: If ratio is not positive.
        ValueError: If kernel_size is not positive.

    Note:
        This implementation follows the modern Keras 3 pattern where sub-layers
        are created in __init__ and built explicitly in build() for robust
        serialization support.
    """

    def __init__(
        self,
        channels: int,
        ratio: int = 8,
        kernel_size: int = 7,
        channel_kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        spatial_kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        channel_kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        spatial_kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        channel_use_bias: bool = False,
        spatial_use_bias: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if ratio <= 0:
            raise ValueError(f"ratio must be positive, got {ratio}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")

        # Store ALL configuration parameters
        self.channels = channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.channel_kernel_initializer = keras.initializers.get(channel_kernel_initializer)
        self.spatial_kernel_initializer = keras.initializers.get(spatial_kernel_initializer)
        self.channel_kernel_regularizer = keras.regularizers.get(channel_kernel_regularizer)
        self.spatial_kernel_regularizer = keras.regularizers.get(spatial_kernel_regularizer)
        self.channel_use_bias = channel_use_bias
        self.spatial_use_bias = spatial_use_bias

        # CREATE sub-layers in __init__ (following modern Keras 3 pattern)
        # These will be unbuilt until build() is called
        self.channel_attention = ChannelAttention(
            channels=self.channels,
            ratio=self.ratio,
            kernel_initializer=self.channel_kernel_initializer,
            kernel_regularizer=self.channel_kernel_regularizer,
            use_bias=self.channel_use_bias,
            name='channel_attention'
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=self.kernel_size,
            kernel_initializer=self.spatial_kernel_initializer,
            kernel_regularizer=self.spatial_kernel_regularizer,
            use_bias=self.spatial_use_bias,
            name='spatial_attention'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        This ensures that when a model is saved and loaded, all weight variables
        exist before weight restoration occurs.

        Args:
            input_shape: Shape tuple indicating the input shape of the layer.
        """
        # BUILD sub-layers explicitly for serialization robustness
        self.channel_attention.build(input_shape)
        self.spatial_attention.build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply CBAM attention to input tensor.

        Implements the sequential attention mechanism:
        1. Channel attention: F' = Mc(F) ⊗ F
        2. Spatial attention: F'' = Ms(F') ⊗ F'

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Refined feature map of shape (batch_size, height, width, channels).
        """
        # Step 1: Apply channel attention
        # Generate channel attention map (batch, 1, 1, channels)
        channel_attention_map = self.channel_attention(inputs, training=training)

        # Refine features using channel attention
        channel_refined = inputs * channel_attention_map

        # Step 2: Apply spatial attention to channel-refined features
        # Generate spatial attention map (batch, height, width, 1)
        spatial_attention_map = self.spatial_attention(channel_refined, training=training)

        # Final refinement using spatial attention
        refined_features = channel_refined * spatial_attention_map

        return refined_features

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple (same as input shape for CBAM).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns ALL parameters needed to reconstruct the layer during
        model loading. This must include every parameter from __init__.

        Returns:
            Dictionary containing the complete layer configuration.
        """
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "ratio": self.ratio,
            "kernel_size": self.kernel_size,
            "channel_kernel_initializer": keras.initializers.serialize(self.channel_kernel_initializer),
            "spatial_kernel_initializer": keras.initializers.serialize(self.spatial_kernel_initializer),
            "channel_kernel_regularizer": keras.regularizers.serialize(self.channel_kernel_regularizer),
            "spatial_kernel_regularizer": keras.regularizers.serialize(self.spatial_kernel_regularizer),
            "channel_use_bias": self.channel_use_bias,
            "spatial_use_bias": self.spatial_use_bias,
        })
        return config

# ---------------------------------------------------------------------