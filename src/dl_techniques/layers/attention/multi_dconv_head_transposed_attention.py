"""Channel-wise ("transposed") self-attention with a depthwise-convolutional
qkv projection, the :class:`MultiDconvHeadTransposedAttention` layer.

Standard self-attention on an image builds an ``(HW, HW)`` affinity matrix,
which is quadratic in the pixel count and unusable at document resolution.
Restormer's Multi-Dconv Head Transposed Attention (MDTA) transposes the
problem: it contracts over the *spatial* axis instead of the channel axis, so
the affinity matrix is ``(C/heads, C/heads)`` per head and the cost is
**linear** in ``H*W``. The queries and keys are L2-normalised along the
flattened spatial axis before the product, which makes each entry a cosine
similarity between two channel maps, and a per-head learnable ``temperature``
rescales the logits before the softmax.

The projection producing q, k and v is a ``1x1`` convolution followed by a
fully depthwise ``3x3`` convolution. The depthwise stage is what gives the
attention a local spatial prior: without it every channel map would be mixed
globally and the layer would carry no notion of neighbourhood at all.

References:
    - Zamir et al., 2022. Restormer: Efficient Transformer for High-Resolution
      Image Restoration. CVPR 2022. (https://arxiv.org/abs/2111.09881)
    - Zhang et al., 2024. DocRes: A Generalist Model Toward Unifying Document
      Image Restoration Tasks. CVPR 2024. (https://arxiv.org/abs/2405.04408)
      DocRes is the consumer in this repository; its backbone is an unmodified
      Restormer, so this layer is shared rather than buried in that model.
"""

import keras
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.dtype_policy import stability_floor
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique(
    "dl_techniques.layers.attention.multi_dconv_head_transposed_attention"
)
class MultiDconvHeadTransposedAttention(keras.layers.Layer):
    """
    Restormer's Multi-Dconv Head Transposed Attention (MDTA), in NHWC.

    Attention is computed **across channels**, not across pixels: the query and
    key tensors are contracted over the flattened spatial axis, giving one
    ``(C/heads, C/heads)`` affinity matrix per head. The layer is therefore
    linear in ``H*W`` and can be applied at full document resolution.

    Architecture:

    .. code-block:: text

        inputs [B, H, W, C]
                │
                ▼
        Conv2D(3C, 1x1, use_bias)          # qkv projection
                │  [B, H, W, 3C]
                ▼
        DepthwiseConv2D(3x3, same)         # fully depthwise over 3C channels
                │  [B, H, W, 3C]
                ▼
        split(3, axis=-1) -> q, k, v       # order is Q, K, V
                │  each [B, H, W, C]
                ▼
        reshape/transpose to [B, heads, C/heads, H*W]
                │                          # head is the OUTER channel factor
                ▼
        q <- q / max(||q||_2 over H*W, floor)
        k <- k / max(||k||_2 over H*W, floor)
                │
                ▼
        attn = (q @ k^T) * temperature     # [B, heads, C/heads, C/heads]
        attn = softmax(attn, axis=-1)
        out  = attn @ v                    # [B, heads, C/heads, H*W]
                │
                ▼
        transpose/reshape back to [B, H, W, C]
                │
                ▼
        Conv2D(C, 1x1, use_bias)           # output projection
                │
                ▼
        output [B, H, W, C]

    Three orderings in that diagram are load-bearing and invisible to a shape
    test, so they are named here as well as guarded in
    ``tests/test_layers/test_attention/test_mdta.py``:

    1. the split order is ``[q, k, v]``;
    2. ``head`` is the **outer** factor of the channel split, i.e. channels
       ``[0, C/heads)`` belong to head 0 (einops ``'b (head c) h w'``);
    3. the L2 normalisation runs over the flattened **spatial** axis, never
       over channels.

    :param dim: Number of input and output channels. Must be positive and
        divisible by ``num_heads``.
    :type dim: int
    :param num_heads: Number of attention heads. Must be positive. In
        Restormer/DocRes the per-head channel width ``dim // num_heads`` is 48
        at every level.
    :type num_heads: int
    :param use_bias: Whether the three convolutions carry a bias. Defaults to
        ``False``, which is the value every Restormer and DocRes call site
        uses.
    :type use_bias: bool
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar dim: The configured channel count.
    :vartype dim: int
    :ivar num_heads: The configured head count.
    :vartype num_heads: int
    :ivar head_dim: ``dim // num_heads``, the per-head channel width.
    :vartype head_dim: int
    :ivar qkv: The ``1x1`` projection to ``3 * dim`` channels.
    :vartype qkv: keras.layers.Conv2D
    :ivar qkv_dwconv: The fully depthwise ``3x3`` convolution on ``3 * dim``
        channels.
    :vartype qkv_dwconv: keras.layers.DepthwiseConv2D
    :ivar project_out: The ``1x1`` output projection back to ``dim`` channels.
    :vartype project_out: keras.layers.Conv2D
    :ivar temperature: Learnable per-head logit scale, shape
        ``(num_heads, 1, 1)``, initialised to ones.
    :vartype temperature: keras.Variable

    :raises ValueError: If ``dim`` or ``num_heads`` is not positive, or if
        ``dim`` is not divisible by ``num_heads``.
    :raises ValueError: From ``build()``, if the input is not rank 4 or if its
        trailing dimension does not equal ``dim``.

    Input shape:
        4D tensor with shape ``(batch_size, height, width, dim)``. ``height``
        and ``width`` may be ``None``; the layer reads them at runtime.

    Output shape:
        4D tensor with shape ``(batch_size, height, width, dim)``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.attention.multi_dconv_head_transposed_attention \
            import MultiDconvHeadTransposedAttention

        x = keras.random.normal((2, 64, 64, 96))
        y = MultiDconvHeadTransposedAttention(dim=96, num_heads=2)(x)
        # y.shape == (2, 64, 64, 96)
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            use_bias: bool = False,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and create the three sub-layers.

        :param dim: Number of input and output channels.
        :type dim: int
        :param num_heads: Number of attention heads.
        :type num_heads: int
        :param use_bias: Whether the convolutions carry a bias.
        :type use_bias: bool
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any

        :raises ValueError: If ``dim`` or ``num_heads`` is not positive, or if
            ``dim`` is not divisible by ``num_heads``.
        """
        super().__init__(**kwargs)

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )

        self.dim = dim
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.head_dim = dim // num_heads

        self.qkv = keras.layers.Conv2D(
            filters=3 * dim,
            kernel_size=1,
            use_bias=use_bias,
            name="qkv",
        )

        # A Keras `DepthwiseConv2D` with the default `depth_multiplier=1` on
        # `3 * dim` input channels is exactly the upstream
        # `Conv2d(3*dim, 3*dim, 3, groups=3*dim)`: its kernel is
        # `(3, 3, 3*dim, 1)`, so each output channel sees only its own input
        # channel. `test_mdta.py::test_the_qkv_dwconv_is_fully_depthwise`
        # asserts both the weight count and that per-channel independence.
        self.qkv_dwconv = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=use_bias,
            name="qkv_dwconv",
        )

        self.project_out = keras.layers.Conv2D(
            filters=dim,
            kernel_size=1,
            use_bias=use_bias,
            name="project_out",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the three convolutions and create the ``temperature`` weight.

        Every sub-layer that ``call()`` runs is built explicitly here, so all
        weight variables exist before weight restoration during model loading.

        :param input_shape: Shape tuple of the input tensor, expected to be
            ``(batch_size, height, width, dim)``.
        :type input_shape: tuple

        :raises ValueError: If ``input_shape`` is not rank 4, or if its
            trailing dimension does not equal ``dim``.
        """
        if self.built:
            return

        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch, height, width, channels), "
                f"got {len(input_shape)}D: {input_shape}"
            )
        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(
                f"Expected input channels ({input_shape[-1]}) to match "
                f"layer dim ({self.dim})"
            )

        self.qkv.build(input_shape)
        qkv_shape = (input_shape[0], input_shape[1], input_shape[2], 3 * self.dim)
        self.qkv_dwconv.build(qkv_shape)
        self.project_out.build(
            (input_shape[0], input_shape[1], input_shape[2], self.dim)
        )

        # Created here via `add_weight` and NEVER `.assign()`d: an `.assign()`
        # inside `build()` is discarded under `StatelessScope`, so the
        # initializer is the only reliable way to set the value.
        self.temperature = self.add_weight(
            name="temperature",
            shape=(self.num_heads, 1, 1),
            initializer="ones",
            trainable=True,
        )

        super().build(input_shape)

    def _to_heads(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Reshape ``(B, H, W, C)`` to ``(B, heads, C/heads, H*W)``.

        The upstream einops pattern is ``'b (head c) h w -> b head c (h w)'``
        on an NCHW tensor: ``head`` is the OUTER factor of the channel split,
        so channels ``[0, C/heads)`` belong to head 0. Reproducing that in
        NHWC takes two moves, and each one is here for a reason:

        1. ``reshape (B, H, W, C) -> (B, H*W, heads, head_dim)`` flattens the
           two spatial axes into one (that axis is what q and k are later
           contracted over) and splits the trailing channel axis into
           ``(heads, head_dim)``. Because the channel axis is the *last* one
           and reshape is row-major, this split is precisely the outer/inner
           factorisation above -- channel index ``head * head_dim + c``.
        2. ``transpose to (0, 2, 3, 1)`` moves ``heads`` to axis 1 and
           ``head_dim`` to axis 2, leaving the flattened spatial axis LAST so
           that the L2 normalisation and the ``q @ k^T`` contraction both act
           on it, exactly as ``F.normalize(..., dim=-1)`` does upstream.

        :param x: Tensor of shape ``(B, H, W, C)``.
        :type x: keras.KerasTensor
        :return: Tensor of shape ``(B, heads, C/heads, H*W)``.
        :rtype: keras.KerasTensor
        """
        # Spatial extents are read at RUNTIME: DocRes runs at arbitrary sizes
        # and the layer must build with `H`/`W` unknown.
        shape = keras.ops.shape(x)
        batch, height, width = shape[0], shape[1], shape[2]
        x = keras.ops.reshape(
            x, (batch, height * width, self.num_heads, self.head_dim)
        )
        return keras.ops.transpose(x, (0, 2, 3, 1))

    def _from_heads(
            self,
            x: keras.KerasTensor,
            height: Any,
            width: Any,
    ) -> keras.KerasTensor:
        """Invert :meth:`_to_heads`: ``(B, heads, C/heads, H*W)`` -> ``(B, H, W, C)``.

        :param x: Tensor of shape ``(B, heads, C/heads, H*W)``.
        :type x: keras.KerasTensor
        :param height: Runtime height, as returned by ``keras.ops.shape``.
        :param width: Runtime width, as returned by ``keras.ops.shape``.
        :return: Tensor of shape ``(B, H, W, C)``.
        :rtype: keras.KerasTensor
        """
        batch = keras.ops.shape(x)[0]
        # Undo the transpose first, restoring (B, H*W, heads, head_dim); the
        # channel axis then re-merges as `head * head_dim + c`, the same
        # ordering `_to_heads` split.
        x = keras.ops.transpose(x, (0, 3, 1, 2))
        return keras.ops.reshape(x, (batch, height, width, self.dim))

    def _l2_normalize_spatial(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """L2-normalise over the LAST axis, which is the flattened spatial axis.

        This mirrors upstream's ``F.normalize(q, dim=-1)`` applied *after* the
        head rearrange, i.e. one unit vector per ``(batch, head, channel)``
        row across all spatial positions. Normalising over channels instead is
        a different layer; ``test_mdta.py`` guards that.

        The denominator floor comes from
        :func:`~dl_techniques.utils.dtype_policy.stability_floor` rather than a
        bare ``1e-12``: that literal rounds to ``0.0`` in float16, which
        reintroduces the division by zero the floor exists to prevent. The
        floor is inert here (``ops.maximum`` shape, and ``||q||`` over dozens
        of unit-scale activations is O(1)).

        :param x: Tensor whose last axis is the flattened spatial axis.
        :type x: keras.KerasTensor
        :return: The same tensor, L2-normalised along its last axis.
        :rtype: keras.KerasTensor
        """
        # DECISION plan-2026-09-08T111844-de235227/D-009: the floor comes from
        # `stability_floor`, NOT from a bare `1e-12` literal and NOT from
        # `keras.ops.cast(1e-12, x.dtype)`. Both of those round the literal to
        # exactly `0.0` in float16 -- the cast that looks protective is itself
        # the mechanism of the defect (`layers/time_series/ema_layer.py:198`).
        # `stability_floor` is in scope here precisely because this is the
        # `ops.maximum(value, floor)` shape, where the floor is inert against a
        # healthy `||q||`; do not reuse it for an ADDED epsilon. See D-009.
        floor = stability_floor(self.compute_dtype, 1e-12)
        norm = keras.ops.sqrt(
            keras.ops.sum(keras.ops.square(x), axis=-1, keepdims=True)
        )
        return x / keras.ops.maximum(norm, keras.ops.cast(floor, x.dtype))

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply channel-wise attention to a feature map.

        :param inputs: Input tensor of shape ``(batch, height, width, dim)``.
        :type inputs: keras.KerasTensor
        :param training: Standard Keras training flag. Unused -- the layer has
            no dropout and no normalisation statistics.
        :type training: Optional[bool]
        :return: Tensor of shape ``(batch, height, width, dim)``.
        :rtype: keras.KerasTensor
        """
        spatial = keras.ops.shape(inputs)
        height, width = spatial[1], spatial[2]

        qkv = self.qkv_dwconv(self.qkv(inputs))

        # Split order is Q, K, V -- `torch.chunk(qkv, 3, dim=1)` upstream.
        q, k, v = keras.ops.split(qkv, 3, axis=-1)

        q = self._to_heads(q)
        k = self._to_heads(k)
        v = self._to_heads(v)

        q = self._l2_normalize_spatial(q)
        k = self._l2_normalize_spatial(k)

        # (B, heads, C/heads, H*W) @ (B, heads, H*W, C/heads)
        #   -> (B, heads, C/heads, C/heads): attention over CHANNELS.
        attn = keras.ops.matmul(q, keras.ops.transpose(k, (0, 1, 3, 2)))
        attn = attn * keras.ops.cast(self.temperature, attn.dtype)
        attn = keras.ops.softmax(attn, axis=-1)

        out = keras.ops.matmul(attn, v)
        out = self._from_heads(out, height, width)

        return self.project_out(out)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Return the output shape, which matches the input with ``dim`` channels.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: ``(batch, height, width, dim)``.
        :rtype: tuple
        """
        return tuple(input_shape[:-1]) + (self.dim,)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments needed to recreate this layer.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "use_bias": self.use_bias,
        })
        return config

# ---------------------------------------------------------------------
