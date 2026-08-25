"""
The Convolutional Block Attention Module (CBAM): channel attention then spatial
attention, applied in sequence.

A convolutional feature map has three axes worth attending over, and a dense
attention map across all of them would cost ``O(H*W*C)`` parameters — more than the
convolution it is meant to refine. CBAM's premise is that the two questions being
asked are separable: WHICH feature detectors matter here, and WHERE in the image
they matter. Factorizing the 3D problem into a per-channel vector and a per-location
map drops the cost to ``O(C^2/r + k^2)``, small enough to insert after every block
of a backbone, while still expressing both kinds of selectivity.

The ordering is the substantive design choice, and it is sequential rather than
parallel. Channel attention runs first, so the spatial stage sees features that have
already been recalibrated for channel-wise importance — it decides where to look in
a map whose channels have been reweighted, not in the raw one. Running them in
parallel and summing would ask both questions of the same unrefined input, which
loses that conditioning; the paper measures the sequential order, channel-first, as
the better arrangement.

Each stage aggregates with BOTH average and max pooling, over its own axes. The
average carries global context; the max reports the strongest single response, which
survives when a signal is concentrated rather than spread. Channel attention pools
over ``(H, W)`` and feeds both descriptors through a SHARED bottleneck MLP before
summing — shared, so the two descriptors are scored by the same function, and
bottlenecked at ratio ``r`` so the module stays cheap. Spatial attention pools over
the channel axis, concatenates the two resulting 2D maps, and convolves them with a
single large kernel, because saliency is a neighbourhood property.

This module is deliberately a **composition, not a re-implementation**. It owns no
attention math of its own: the two stages are the package's existing
`ChannelAttention` and `SpatialAttention` layers, held as sub-layers, and `call` is
just the two broadcasting multiplies that wire them together. `channels` is
forwarded to the channel stage only, because `SpatialAttention` fully reduces the
channel axis before its convolution and is therefore channel-count agnostic.

Foundational mathematics::

    F'  = M_c(F)  (x) F
    F'' = M_s(F') (x) F'

where ``(x)`` is element-wise multiplication with broadcasting: ``M_c`` is
``(B, 1, 1, C)`` and broadcasts over space, ``M_s`` is ``(B, H, W, 1)`` and
broadcasts over channels. Both gates are sigmoid-bounded, so each stage can only
attenuate.

References:
    - Woo et al., 2018. CBAM: Convolutional Block Attention Module. ECCV.
      (https://arxiv.org/abs/1807.06521)
    - Hu et al., 2018. Squeeze-and-Excitation Networks. (the channel-attention
      ancestor CBAM's first stage extends with max pooling)
      (https://arxiv.org/abs/1709.01507)
    - Park et al., 2018. BAM: Bottleneck Attention Module. (the same authors'
      PARALLEL-branch variant, which CBAM's sequential ordering is measured
      against) (https://arxiv.org/abs/1807.06514)
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
    CBAM: sequential channel-then-spatial feature refinement.

    Applies channel attention and then spatial attention to a convolutional
    feature map. Channel attention first recalibrates "what" features matter, then
    spatial attention refines "where" to focus — in that order, so the spatial
    stage operates on already-recalibrated features. The complete operation is
    ``F'' = M_s(M_c(F) * F) * (M_c(F) * F)``, where ``M_c`` and ``M_s`` are the
    channel and spatial attention maps.

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

        ┌──────────────────────────────────────────────────────────────┐
        │  Input F [B, H, W, C]                                        │
        └───────────────┬──────────────────────────────────────────────┘
                        ├──────────────────────────────┐
                        ▼                              │
        ┌──────────────────────────────────────────┐   │
        │  channel_attention → M_c [B, 1, 1, C]    │   │
        │    pool over (H, W): avg AND max         │   │
        │    → SHARED bottleneck MLP (ratio r)     │   │
        │    → sum → sigmoid                       │   │
        │    "WHAT matters"                        │   │
        └──────────────────┬───────────────────────┘   │
                           ▼                           │
                          (×)◄─────────────────────────┘
                           │  broadcast over space
                           ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  F' [B, H, W, C]  — channel-refined                          │
        └───────────────┬──────────────────────────────┬───────────────┘
                        ▼                              │
        ┌──────────────────────────────────────────┐   │
        │  spatial_attention → M_s [B, H, W, 1]    │   │
        │    pool over C: avg AND max → concat     │   │
        │    → Conv2D(1 filter, k×k) → sigmoid     │   │
        │    "WHERE it matters" — computed from F',│   │
        │    NOT from F: that conditioning is why  │   │
        │    the order is sequential, not parallel │   │
        └──────────────────┬───────────────────────┘   │
                           ▼                           │
                          (×)◄─────────────────────────┘
                           │  broadcast over channels
                           ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  Output F'' [B, H, W, C] — shape unchanged; both gates are   │
        │  sigmoid-bounded, so each stage can only attenuate           │
        └──────────────────────────────────────────────────────────────┘

    **Cost:**

    .. code-block:: text

        channel stage   O(C² / r)   the shared MLP bottleneck
        spatial stage   O(k²)       one k×k filter over a 2-channel map
        dense 3D map    O(H·W·C)    what the factorization avoids

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
    :raises ValueError: From the sub-layers, if ``channels`` is not divisible by
        ``ratio`` (raised by :class:`ChannelAttention` during construction) or if
        ``kernel_size`` is even (raised by :class:`SpatialAttention`).

    Input shape:
        4D tensor with shape ``(batch_size, height, width, channels)``, where the
        channel count must equal ``channels``.

    Output shape:
        4D tensor with shape ``(batch_size, height, width, channels)`` —
        identical to the input. This layer returns the REFINED features, not the
        attention maps.

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
        """Validate the three cheap invariants and create the two attention stages.

        Everything else is delegated: the ``channel_*`` and ``spatial_*``
        parameters are forwarded verbatim to the sub-layers, which own their own
        validation. See the class docstring for the parameter reference.
        """
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

    def build(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> None:
        """Build both attention stages explicitly.

        Both stages take the SAME ``input_shape``: the channel stage's map
        broadcasts rather than reshapes, so the spatial stage still sees a
        ``(B, H, W, C)`` tensor. Building by hand is what guarantees every weight
        variable exists before Keras restores a checkpoint into it.

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

        Note that the spatial map is computed from ``channel_refined``, not from
        ``inputs`` — that conditioning is the whole reason the two stages are
        sequential rather than parallel.

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

# ---------------------------------------------------------------------
