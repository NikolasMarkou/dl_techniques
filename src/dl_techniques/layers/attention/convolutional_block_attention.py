"""
The Convolutional Block Attention Module (CBAM), a lightweight attention block for CNNs.

CBAM infers attention maps along two separate dimensions — channel and spatial —
and applies them sequentially to the input feature map for adaptive feature
refinement. The key architectural choice is the ordering: channel attention runs
first, so the spatial stage operates on features that have *already* been
recalibrated for channel-wise importance.

This module is deliberately a **composition, not a re-implementation**. It owns no
attention math of its own: the two stages are the package's existing
:class:`~dl_techniques.layers.attention.channel_attention.ChannelAttention` and
:class:`~dl_techniques.layers.attention.spatial_attention.SpatialAttention` layers,
instantiated as sub-layers. See the ``[REUSE]`` note on :class:`CBAM` below.

Architecture:
    The block factorizes 3D attention over ``(H, W, C)`` into two cheap 1D/2D
    problems applied back to back:

    1.  **Channel Attention (`Mc`) — "what" matters.** Spatial information is
        aggregated by average-pooling and max-pooling across the spatial
        dimensions ``(H x W)``, producing two context descriptors per channel.
        Both are processed by a *shared* bottleneck MLP, merged by element-wise
        summation, and passed through a sigmoid. The result encodes the
        inter-channel relationship of features.

    2.  **Spatial Attention (`Ms`) — "where" matters.** Operating on the
        channel-refined features, this stage aggregates channel information at
        each spatial location via average- and max-pooling along the channel
        axis. The two resulting 2D maps are concatenated and passed through a
        single convolution, then a sigmoid, highlighting the most salient
        spatial regions.

Foundational Mathematics:
    The complete CBAM operation is a sequential (not parallel) multiplication::

        F'  = M_c(F)  ⊗ F
        F'' = M_s(F') ⊗ F'

    where ``⊗`` is element-wise multiplication with broadcasting: ``M_c`` has
    shape ``(B, 1, 1, C)`` and broadcasts over space, while ``M_s`` has shape
    ``(B, H, W, 1)`` and broadcasts over channels. Factorizing attention into two
    sequential, decoupled modules is what makes CBAM cheap: it costs
    ``O(C^2/r + k^2)`` parameters rather than the ``O(H*W*C)`` of a dense 3D map.

References:
    - Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). "CBAM: Convolutional
      Block Attention Module". ECCV. (https://arxiv.org/abs/1807.06521)
"""

# ---------------------------------------------------------------------

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
    Convolutional Block Attention Module for sequential channel-spatial feature refinement.

    Implements the CBAM attention mechanism that sequentially applies channel
    and spatial attention to input feature maps. Channel attention first
    recalibrates "what" features matter, then spatial attention refines "where"
    to focus. The complete operation is
    ``F'' = M_s(M_c(F) * F) * (M_c(F) * F)``, where ``M_c`` and ``M_s``
    are channel and spatial attention maps respectively.

    **[REUSE]** This layer contains **no attention math of its own**. Both stages
    are the package's existing standalone layers, held as sub-layers:
    :class:`~dl_techniques.layers.attention.channel_attention.ChannelAttention`
    and
    :class:`~dl_techniques.layers.attention.spatial_attention.SpatialAttention`.
    ``call()`` is only the two multiplications that wire them together. Consequences
    a maintainer must respect:

    * A fix to either stage's math belongs in that stage's module and is inherited
      here for free. Do **not** inline or fork the pooling/MLP/conv logic into this
      file — that would create the exact drift the composition exists to prevent.
    * The constructor's ``channel_*`` / ``spatial_*`` parameter pairs exist because
      each sub-layer is independently configurable; they are forwarded verbatim and
      are not re-validated here beyond the three cheap positivity checks.
    * ``channels`` is forwarded to the channel stage only —
      :class:`SpatialAttention` is channel-count agnostic by construction.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────────────────────┐
        │                          CBAM                           │
        │                                                         │
        │   Input F [B, H, W, C]                                  │
        │          │                                              │
        │          ▼                                              │
        │   ┌─────────────────────────────────────────────┐       │
        │   │      Channel Attention   M_c: [B,1,1,C]     │       │
        │   └──────────────────────┬──────────────────────┘       │
        │                          ▼                              │
        │             F' = M_c(F) ⊗ F                             │
        │                          │                              │
        │                          ▼                              │
        │   ┌─────────────────────────────────────────────┐       │
        │   │      Spatial Attention   M_s: [B,H,W,1]     │       │
        │   └──────────────────────┬──────────────────────┘       │
        │                          ▼                              │
        │            F'' = M_s(F') ⊗ F'                           │
        │                          │                              │
        │                          ▼                              │
        │   Output [B, H, W, C]                                   │
        └─────────────────────────────────────────────────────────┘

    :param channels: Number of input channels. Must be positive.
    :type channels: int
    :param ratio: Reduction ratio for channel attention MLP bottleneck.
        Higher values reduce parameters but may limit representation
        capacity. Must be positive. Defaults to 8.
    :type ratio: int
    :param kernel_size: Kernel size for spatial attention convolution.
        Must be positive and odd. Defaults to 7.
    :type kernel_size: int
    :param channel_kernel_initializer: Kernel initializer for channel
        attention layers. Defaults to ``'glorot_uniform'``.
    :type channel_kernel_initializer: str or keras.initializers.Initializer
    :param spatial_kernel_initializer: Kernel initializer for spatial
        attention layers. Defaults to ``'glorot_uniform'``.
    :type spatial_kernel_initializer: str or keras.initializers.Initializer
    :param channel_kernel_regularizer: Optional kernel regularizer for
        channel attention layers. Defaults to ``None``.
    :type channel_kernel_regularizer: keras.regularizers.Regularizer or None
    :param spatial_kernel_regularizer: Optional kernel regularizer for
        spatial attention layers. Defaults to ``None``.
    :type spatial_kernel_regularizer: keras.regularizers.Regularizer or None
    :param channel_use_bias: Whether to use bias in channel attention
        dense layers. Defaults to ``False``.
    :type channel_use_bias: bool
    :param spatial_use_bias: Whether to use bias in spatial attention
        convolution. Defaults to ``True``.
    :type spatial_use_bias: bool
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``channels`` is not positive.
    :raises ValueError: If ``ratio`` is not positive.
    :raises ValueError: If ``kernel_size`` is not positive.
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
        #
        # `channels` is a real CNN channel count, not a "model dimension"; the
        # `GUIDE.md` naming table's `channels` -> `dim` migration line does NOT apply
        # to the CNN family (documented carve-out, `README.md:17-18,90`). Renaming it
        # would break the frozen public API and every serialized `get_config()`.
        #
        # Note the division of labour: `channels % ratio` is NOT checked here. That
        # check lives in `ChannelAttention.__init__` and fires when the sub-layer is
        # constructed three statements below, so a bad (channels, ratio) pair still
        # raises from this constructor, carrying the sub-layer's message.
        # Duplicating the check here would give two spellings of one rule.
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

        Explicitly builds each sub-layer for robust serialization, ensuring
        all weight variables exist before weight restoration during model
        loading.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        """
        if self.built:
            return

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
        Apply sequential channel-then-spatial CBAM attention to the input tensor.

        :param inputs: Input tensor of shape
            ``(batch_size, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer should behave in training mode
            or inference mode.
        :type training: bool or None
        :return: Refined feature map of shape
            ``(batch_size, height, width, channels)``.
        :rtype: keras.KerasTensor
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

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: Output shape tuple (same as input shape).
        :rtype: tuple
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Dictionary containing the complete layer configuration.
        :rtype: dict
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
