"""
Keras Implementation of the Convolutional Block Attention Module (CBAM).

This file provides a Keras-native implementation of the CBAM attention mechanism,
as detailed in the paper: 'CBAM: Convolutional Block Attention Module' by
Woo et al. (2018). Link: https://arxiv.org/abs/1807.06521v2

The primary export of this file is the `CBAM` layer, which serves as a
composite layer that sequentially applies channel and spatial attention to refine
input feature maps.

Implementation Details:
-----------------------

1.  **Modular and Composite Structure:**
    The implementation is highly modular. The `CBAM` class acts as a container
    and orchestrator for two specialized, independent sub-modules:
    - `ChannelAttention`: Handles the channel-wise attention mechanism.
    - `SpatialAttention`: Handles the spatial-wise attention mechanism.
    These sub-modules are imported from `.channel_attention` and
    `.spatial_attention` respectively, promoting code organization and reusability.

2.  **Sequential Attention Flow:**
    The core logic resides in the `call` method, which strictly follows the
    sequential attention process described in the paper:
    - **Step 1 (Channel Attention):** The input tensor is first passed to the
      `ChannelAttention` sub-module to generate a 1D channel attention map
      (shape: `(batch, 1, 1, channels)`).
    - **Step 2 (Channel Refinement):** This attention map is broadcasted across the
      spatial dimensions and multiplied element-wise with the original input tensor
      to produce a channel-refined feature map.
    - **Step 3 (Spatial Attention):** The channel-refined feature map is then passed
      to the `SpatialAttention` sub-module. This module generates a 2D spatial
      attention map (shape: `(batch, height, width, 1)`).
    - **Step 4 (Final Refinement):** The spatial attention map is broadcasted across
      the channel dimension and multiplied element-wise with the channel-refined
      feature map to produce the final output.
"""

import keras
from typing import Optional, Union, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .channel_attention import ChannelAttention
from .spatial_attention import SpatialAttention

# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
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
        channel_use_bias: Whether to use bias in channel attention layers.
        spatial_use_bias: Whether to use bias in spatial attention layers.
        **kwargs: Additional keyword arguments for the Layer base class.
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
        self.channels = channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.channel_kernel_initializer = keras.initializers.get(channel_kernel_initializer)
        self.spatial_kernel_initializer = keras.initializers.get(spatial_kernel_initializer)
        self.channel_kernel_regularizer = keras.regularizers.get(channel_kernel_regularizer)
        self.spatial_kernel_regularizer = keras.regularizers.get(spatial_kernel_regularizer)
        self.channel_use_bias = channel_use_bias
        self.spatial_use_bias = spatial_use_bias

        # Sublayers will be initialized in build()
        self.channel_attention = None
        self.spatial_attention = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's weights based on input shape.

        Args:
            input_shape: Shape tuple indicating the input shape of the layer.
        """
        self._build_input_shape = input_shape

        # Create channel attention module
        self.channel_attention = ChannelAttention(
            channels=self.channels,
            ratio=self.ratio,
            kernel_initializer=self.channel_kernel_initializer,
            kernel_regularizer=self.channel_kernel_regularizer,
            use_bias=self.channel_use_bias,
            name='channel_attention'
        )
        self.channel_attention.build(input_shape)

        # Create spatial attention module
        self.spatial_attention = SpatialAttention(
            kernel_size=self.kernel_size,
            kernel_initializer=self.spatial_kernel_initializer,
            kernel_regularizer=self.spatial_kernel_regularizer,
            use_bias=self.spatial_use_bias,
            name='spatial_attention'
        )
        self.spatial_attention.build(input_shape)

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Apply CBAM attention to input tensor.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Refined feature map of shape (batch_size, height, width, channels).
        """
        # Apply channel attention
        channel_attention_map = self.channel_attention(inputs, training=training)
        channel_refined = inputs * channel_attention_map

        # Apply spatial attention to channel-refined features
        spatial_attention_map = self.spatial_attention(channel_refined, training=training)
        refined = channel_refined * spatial_attention_map

        return refined

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape (same as input shape).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
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

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
