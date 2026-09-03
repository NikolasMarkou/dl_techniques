"""The Convolutional Block Attention Module, the :class:`CBAM` layer: channel attention then spatial attention, in sequence.

A convolutional feature map has three axes worth attending over. A dense
attention map across all of them would cost `O(H*W*C)` parameters, more
than the convolution it refines. CBAM factorizes the problem into a
per-channel vector and a per-location map, dropping the cost to
`O(C^2/r + k^2)`, cheap enough to insert after every block of a
backbone. The two stages run in sequence, not in parallel: channel
attention runs first, so the spatial stage sees features already
recalibrated for channel importance rather than the raw input, which the
paper measures as the better arrangement.

The layer is a composition, not a re-implementation. It owns no
attention math of its own; the two stages are the package's existing
:class:`ChannelAttention` and :class:`SpatialAttention` layers held as
sub-layers, and `call` is the two broadcasting multiplies that wire them
together. `channels` is forwarded to the channel stage only —
`SpatialAttention` fully reduces the channel axis before its
convolution, so it works for any channel count.

Foundational mathematics::

    F'  = M_c(F)  (x) F
    F'' = M_s(F') (x) F'

``(x)`` is element-wise multiplication with broadcasting. ``M_c`` is
``(B, 1, 1, C)`` and broadcasts over space. ``M_s`` is ``(B, H, W, 1)`` and
broadcasts over channels. Both gates are sigmoid-bounded, so each stage can
only attenuate.

References:
    - Woo et al., 2018. CBAM: Convolutional Block Attention Module. ECCV.
      (https://arxiv.org/abs/1807.06521)
    - Hu et al., 2018. Squeeze-and-Excitation Networks. (the channel-attention
      ancestor CBAM's first stage extends with max pooling)
      (https://arxiv.org/abs/1709.01507)
    - Park et al., 2018. BAM: Bottleneck Attention Module. (the same
      authors' parallel-branch variant, which CBAM's sequential ordering
      is measured against) (https://arxiv.org/abs/1807.06514)
"""

import keras
from typing import Optional, Union, Dict, Any, Tuple

from .channel_attention import ChannelAttention
from .spatial_attention import SpatialAttention
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.attention.convolutional_block_attention")
class CBAM(keras.layers.Layer):
    """
    CBAM: sequential channel-then-spatial feature refinement.

    Applies channel attention and then spatial attention to a convolutional
    feature map. Channel attention recalibrates which features matter. Spatial
    attention then refines where to focus, operating on the already
    recalibrated features. The full operation is
    ``F'' = M_s(M_c(F) * F) * (M_c(F) * F)``, where ``M_c`` and ``M_s`` are the
    channel and spatial attention maps. The layer returns the refined features,
    not the maps.

    This layer contains no attention math of its own. Both stages are the
    package's existing standalone layers, held as sub-layers:
    :class:`~dl_techniques.layers.attention.channel_attention.ChannelAttention`
    and
    :class:`~dl_techniques.layers.attention.spatial_attention.SpatialAttention`.
    ``call()`` is only the two multiplications that wire them together. A fix
    to either stage's math belongs in that stage's module rather than being
    inlined or forked here, which would recreate the drift the composition
    exists to prevent. The constructor's ``channel_*`` and ``spatial_*``
    parameter pairs are forwarded verbatim and are not re-validated here
    beyond three cheap positivity checks. ``channels`` is forwarded to the
    channel stage only: :class:`SpatialAttention` reduces the channel axis
    away before its convolution, so it works for any channel count.

    Architecture:

    .. code-block:: text

        inputs F [B, H, W, C]
                │
        ┌───────┴──────────┐
        ▼                    │
        channel_attention(F) │
        -> M_c [B, 1, 1, C]  │
        │                    │
        ▼                    │
        (x) <────────────────┘  broadcast over H, W
        │
        ▼
        F' [B, H, W, C]  channel-refined
                │
        ┌───────┴──────────┐
        ▼                    │
        spatial_attention(F')│
        -> M_s [B, H, W, 1]  │
        │                    │
        ▼                    │
        (x) <────────────────┘  broadcast over C
        │
        ▼
        output F'' [B, H, W, C]

    ``M_s`` is computed from ``F'``, never from ``F`` — that conditioning
    is why the two stages run in sequence rather than in parallel.

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

    :ivar channels: The configured channel count.
    :vartype channels: int
    :ivar ratio: The configured MLP bottleneck reduction ratio.
    :vartype ratio: int
    :ivar kernel_size: The configured spatial convolution kernel size.
    :vartype kernel_size: int
    :ivar channel_attention: The channel stage sub-layer.
    :vartype channel_attention: ChannelAttention
    :ivar spatial_attention: The spatial stage sub-layer.
    :vartype spatial_attention: SpatialAttention

    :raises ValueError: If ``channels`` is not positive.
    :raises ValueError: If ``ratio`` is not positive.
    :raises ValueError: If ``kernel_size`` is not positive.
    :raises ValueError: From the sub-layers, if ``channels`` is not divisible by
        ``ratio`` (raised by :class:`ChannelAttention` during construction) or if
        ``kernel_size`` is even (raised by :class:`SpatialAttention`).

    Input shape:
        4D tensor with shape ``(batch_size, height, width, channels)``, where the
        channel count must equal ``channels``.

    Output shape:
        4D tensor with shape ``(batch_size, height, width, channels)``, identical
        to the input. This layer returns the refined features, not the attention
        maps.

    Example:
        >>> # Drop-in refinement after a conv block
        >>> cbam = CBAM(channels=256)
        >>> feats = keras.random.normal((4, 32, 32, 256))
        >>> refined = cbam(feats)                     # (4, 32, 32, 256)
        >>>
        >>> # Cheaper channel stage, smaller spatial kernel
        >>> cbam = CBAM(channels=256, ratio=16, kernel_size=3)
        >>>
        >>> # Per-stage regularization
        >>> cbam = CBAM(channels=256,
        ...             channel_kernel_regularizer=keras.regularizers.L2(1e-4))
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
        """Validate three cheap invariants and create the two attention stages.

        Everything else is delegated. The ``channel_*`` and ``spatial_*``
        parameters are forwarded verbatim to the sub-layers, which own their own
        validation.

        :param channels: Number of input channels. Must be positive.
        :type channels: int
        :param ratio: Reduction ratio for the channel MLP bottleneck.
        :type ratio: int
        :param kernel_size: Kernel size for the spatial convolution.
        :type kernel_size: int
        :param channel_kernel_initializer: Kernel initializer for the channel
            stage.
        :type channel_kernel_initializer: str or keras.initializers.Initializer
        :param spatial_kernel_initializer: Kernel initializer for the spatial
            stage.
        :type spatial_kernel_initializer: str or keras.initializers.Initializer
        :param channel_kernel_regularizer: Optional regularizer for the channel
            stage.
        :type channel_kernel_regularizer: keras.regularizers.Regularizer or None
        :param spatial_kernel_regularizer: Optional regularizer for the spatial
            stage.
        :type spatial_kernel_regularizer: keras.regularizers.Regularizer or None
        :param channel_use_bias: Whether the channel dense layers carry a bias.
        :type channel_use_bias: bool
        :param spatial_use_bias: Whether the spatial convolution carries a bias.
        :type spatial_use_bias: bool
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any

        :raises ValueError: If ``channels``, ``ratio`` or ``kernel_size`` is not
            positive, or if a sub-layer rejects the pair it is given.
        """
        super().__init__(**kwargs)

        # `channels % ratio` is not checked here; ChannelAttention.__init__
        # raises it when constructed below, so duplicating it here would give
        # two spellings of one rule.
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if ratio <= 0:
            raise ValueError(f"ratio must be positive, got {ratio}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")

        self.channels = channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.channel_kernel_initializer = keras.initializers.get(channel_kernel_initializer)
        self.spatial_kernel_initializer = keras.initializers.get(spatial_kernel_initializer)
        self.channel_kernel_regularizer = keras.regularizers.get(channel_kernel_regularizer)
        self.spatial_kernel_regularizer = keras.regularizers.get(spatial_kernel_regularizer)
        self.channel_use_bias = channel_use_bias
        self.spatial_use_bias = spatial_use_bias

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

    def build(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> None:
        """Build both attention stages explicitly.

        Both stages take the same ``input_shape``. The channel stage's map
        broadcasts rather than reshapes, so the spatial stage still sees a
        ``(B, H, W, C)`` tensor. Building by hand guarantees every weight
        variable exists before Keras restores a checkpoint into it.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple

        :raises ValueError: From ``SpatialAttention.build``, if ``input_shape``
            is not rank 4.
        """
        if self.built:
            return

        self.channel_attention.build(input_shape)
        self.spatial_attention.build(input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply sequential channel-then-spatial CBAM attention to the input.

        The spatial map is computed from ``channel_refined``, not from
        ``inputs``. That conditioning is why the two stages run in sequence
        rather than in parallel.

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
        # Step 1: channel attention. The map is (batch, 1, 1, channels).
        channel_attention_map = self.channel_attention(inputs, training=training)

        # Refine features using channel attention
        channel_refined = inputs * channel_attention_map

        # Step 2: spatial attention over the channel-refined features.
        # The map is (batch, height, width, 1).
        spatial_attention_map = self.spatial_attention(channel_refined, training=training)

        # Final refinement using spatial attention
        refined_features = channel_refined * spatial_attention_map

        return refined_features

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Return the output shape, which equals the input shape.

        Both stages are multiplicative gates, so neither changes the shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: Output shape tuple (same as input shape).
        :rtype: tuple
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        Every constructor argument is included, so both stages are reconstructed
        from config rather than serialized as sub-layers.

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
