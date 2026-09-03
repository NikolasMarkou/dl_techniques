"""``LeVJEPABlock``: pre-norm self-attention + MLP transformer block for LeVJEPA.

Ports the LeVJEPA PyTorch reference's ``Block`` class:
``x = x + Attn(LN(x)); x = x + MLP(LN(x))``, with
``Attn(y) = softmax(Q K^T / sqrt(d_head) + mask) V``. No ``LayerScale`` on
either residual branch, matching the reference (see the ``DECISION`` note
below for why this diverges from an earlier draft of this file's spec).

The attention is hand-rolled (QKV, reshape, optional RoPE, scaled dot
product, projection) rather than built on
``layers/attention/multi_head_attention.py``, because :class:`VideoRoPE3D`
must rotate ``q``/``k`` after the head split and before the softmax, a hook
the generic attention layer does not expose.

References:
    - LeVJEPA PyTorch reference, ``module.py::Block`` / ``Attention`` /
      ``RoPEAttention`` (pasted transcript; no public arXiv id in this plan's
      context).
    - Vaswani et al. (2017). "Attention Is All You Need". arXiv:1706.03762.
"""

import keras
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.initializers import clone_initializer
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.layers.embedding.video_rope import VideoRoPE3D
from dl_techniques.layers.attention.common import (
    apply_attention_mask,
    compute_attention_scale,
    validate_head_divisibility,
)

# ---------------------------------------------------------------------

# DECISION plan-2026-09-03T113223-2a714a91/D-011: no LayerScale sub-layers here.
# An earlier draft of this file's spec described gamma_a/gamma_m LayerScale gates,
# but the actual pasted PyTorch reference uses plain residual addition. See decisions.md.

REFERENCE_INIT_STD = 0.02


@register_dl_technique("dl_techniques.models.levjepa.blocks")
class LeVJEPABlock(keras.layers.Layer):
    """Pre-norm self-attention + MLP block, with an optional 3-axis video RoPE.

    Ports the LeVJEPA reference's ``Block`` verbatim: ``x = x + Attn(LN(x));
    x = x + MLP(LN(x))``, with no ``LayerScale`` on either residual (see the
    module-level ``DECISION`` note).

    The attention is a bespoke QKV projection and scaled-dot-product, not a
    call into ``layers/attention/multi_head_attention.py``: ``VideoRoPE3D``
    (when ``use_rope=True``) must rotate ``q``/``k`` after the head split and
    before the softmax, a hook the generic attention layer does not expose.
    A block-causal (or any other) attention mask is accepted as a pre-built
    boolean keep predicate (``True`` = attend), applied via
    ``layers/attention/common.py::apply_attention_mask``; this layer infers
    no polarity of its own.

    Architecture:

    .. code-block:: text

        x [B, N, D]
            |
            +---------------------------------+  (residual)
        LayerNorm(eps=1e-6)                    |
            |                                  |
        Dense(3D) -> reshape [B, N, 3, H, d]    |
            |                                  |
        split q, k, v  [B, H, N, d] each        |
            |                                  |
        (use_rope) VideoRoPE3D rotates          |
        q[:, :, prefix:, :], k[:, :, prefix:, :] |
            |                                  |
        logits = (q k^T) * scale                |
        + block-causal mask (optional)          |
        softmax -> attn_drop                    |
            |                                  |
        out = attn @ v -> reshape [B, N, D]     |
            |                                  |
        Dense(D) -> proj_drop                   |
            |                                  |
           (+) <-------------------------------+
            |
            +---------------------------------+  (residual)
        LayerNorm(eps=1e-6)                    |
            |                                  |
        Dense(hidden) -> GELU -> drop           |
        Dense(D) -> drop                        |
            |                                  |
           (+) <-------------------------------+
            |
        x' [B, N, D]

    :param dim: Model / embedding dimension. Must be positive and divisible
        by ``num_heads``.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param mlp_ratio: MLP hidden-dimension multiplier. Must be positive.
        Defaults to ``4.0``.
    :type mlp_ratio: float
    :param qkv_bias: Whether the fused QKV projection has a bias. Defaults to
        ``True``, matching the reference.
    :type qkv_bias: bool
    :param use_rope: Whether to rotate ``q``/``k`` with :class:`VideoRoPE3D`
        before the softmax. When ``True``, ``call()`` requires ``num_frames``,
        ``height_patches`` and ``width_patches``. Defaults to ``False``.
    :type use_rope: bool
    :param rope_theta: Rotary base frequency, forwarded to ``VideoRoPE3D``
        when ``use_rope=True``. Defaults to ``10000.0``.
    :type rope_theta: float
    :param num_prefix_tokens: Number of leading tokens (the CLS token(s))
        excluded from RoPE rotation -- they pass through unrotated, matching
        the reference's ``q[:, :, num_prefix:, :]`` slicing. Defaults to
        ``1``.
    :type num_prefix_tokens: int
    :param dropout_rate: Dropout rate applied after the output projection and
        inside the MLP. Must be in ``[0, 1]``. Defaults to ``0.0``.
    :type dropout_rate: float
    :param attention_dropout_rate: Dropout rate applied to the post-softmax
        attention weights. Must be in ``[0, 1]``. Defaults to ``0.0``.
    :type attention_dropout_rate: float
    :param layer_id: 1-indexed block position within the encoder stack. Used
        only to compute the block-depth-rescaled initializer std for
        ``proj`` and ``fc2`` (``init_std / sqrt(2 * layer_id)``), matching
        the reference's ``_rescale_blocks`` post-hoc weight division. ``None``
        (default) disables rescaling -- both kernels use ``init_std``
        directly, i.e. ``layer_id`` behaves as if it were ``inf``. See the
        ``_rescale_blocks`` note below.
    :type layer_id: Optional[int]
    :param init_std: Base truncated-normal std for every kernel in this block.
        Defaults to ``0.02``, the reference's ``init_std``.
    :type init_std: float
    :param bias_initializer: Bias initializer for every Dense sub-layer.
        Defaults to ``'zeros'``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for every kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for every bias.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar norm1: Pre-attention ``LayerNormalization(epsilon=1e-6)``.
    :ivar norm2: Pre-MLP ``LayerNormalization(epsilon=1e-6)``.
    :ivar qkv: Fused ``Dense(3 * dim)`` projection.
    :ivar proj: Attention output projection, ``Dense(dim)``.
    :ivar rope: :class:`VideoRoPE3D` instance, or ``None`` when
        ``use_rope=False``.
    :ivar fc1: MLP expansion ``Dense(hidden_dim)``.
    :ivar fc2: MLP contraction ``Dense(dim)``.

    Input shape:
        ``x``: ``(batch, num_tokens, dim)``.

    Output shape:
        ``(batch, num_tokens, dim)``, unchanged.

    :raises ValueError: If ``dim`` is not divisible by ``num_heads``, if
        ``dim``, ``num_heads`` or ``mlp_ratio`` is not positive, or if
        ``dropout_rate``/``attention_dropout_rate`` leaves ``[0, 1]``. Raised
        from ``__init__``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.models.vision.levjepa.blocks import LeVJEPABlock

        block = LeVJEPABlock(dim=192, num_heads=3, use_rope=True, layer_id=1)
        x = keras.random.normal((2, 1 + 18, 192))
        block(x, num_frames=2, height_patches=3, width_patches=3).shape
        # (2, 19, 192)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        num_prefix_tokens: int = 1,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        layer_id: Optional[int] = None,
        init_std: float = REFERENCE_INIT_STD,
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Validate the configuration and create every sub-layer.

        :param dim: Model / embedding dimension.
        :type dim: int
        :param num_heads: Number of attention heads.
        :type num_heads: int
        :param mlp_ratio: MLP hidden-dimension multiplier.
        :type mlp_ratio: float
        :param qkv_bias: Whether the QKV projection has a bias.
        :type qkv_bias: bool
        :param use_rope: Whether to rotate q/k with VideoRoPE3D.
        :type use_rope: bool
        :param rope_theta: Rotary base frequency.
        :type rope_theta: float
        :param num_prefix_tokens: Number of leading unrotated prefix tokens.
        :type num_prefix_tokens: int
        :param dropout_rate: Output/MLP dropout rate.
        :type dropout_rate: float
        :param attention_dropout_rate: Post-softmax attention dropout rate.
        :type attention_dropout_rate: float
        :param layer_id: 1-indexed block position for depth rescaling.
        :type layer_id: Optional[int]
        :param init_std: Base truncated-normal std.
        :type init_std: float
        :param bias_initializer: Bias initializer.
        :type bias_initializer: Union[str, keras.initializers.Initializer]
        :param kernel_regularizer: Optional kernel regularizer.
        :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
        :param bias_regularizer: Optional bias regularizer.
        :type bias_regularizer: Optional[keras.regularizers.Regularizer]
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any
        :raises ValueError: If the configuration is invalid.
        """
        super().__init__(**kwargs)

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        validate_head_divisibility(dim, num_heads, dim_name="dim")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(
                f"attention_dropout_rate must be in [0, 1], got {attention_dropout_rate}"
            )
        if num_prefix_tokens < 0:
            raise ValueError(
                f"num_prefix_tokens must be non-negative, got {num_prefix_tokens}"
            )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.mlp_ratio = float(mlp_ratio)
        self.hidden_dim = int(dim * mlp_ratio)
        self.qkv_bias = bool(qkv_bias)
        self.use_rope = bool(use_rope)
        self.rope_theta = float(rope_theta)
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.dropout_rate = float(dropout_rate)
        self.attention_dropout_rate = float(attention_dropout_rate)
        self.layer_id = layer_id
        self.init_std = float(init_std)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        self._scale = compute_attention_scale(self.head_dim)

        # DECISION plan-2026-09-03T113223-2a714a91/D-012: block-depth rescaling is
        # a pre-scaled initializer std, not a post-build `.assign()` (which
        # StatelessScope discards). Dividing a TruncatedNormal sample by a
        # constant is distributionally identical to scaling its std. See decisions.md.
        rescale_std = self.init_std
        if self.layer_id is not None:
            rescale_std = self.init_std / ((2.0 * float(self.layer_id)) ** 0.5)

        base_kernel_init = {
            "class_name": "TruncatedNormal",
            "config": {"stddev": self.init_std},
        }
        rescaled_kernel_init = {
            "class_name": "TruncatedNormal",
            "config": {"stddev": rescale_std},
        }

        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6, name="norm2")

        self.qkv = keras.layers.Dense(
            self.dim * 3,
            use_bias=self.qkv_bias,
            kernel_initializer=clone_initializer(keras.initializers.get(base_kernel_init)),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="qkv",
        )
        self.proj = keras.layers.Dense(
            self.dim,
            use_bias=True,
            kernel_initializer=clone_initializer(keras.initializers.get(rescaled_kernel_init)),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="proj",
        )
        self.attn_drop = (
            keras.layers.Dropout(self.attention_dropout_rate, name="attn_drop")
            if self.attention_dropout_rate > 0.0
            else None
        )
        self.proj_drop = (
            keras.layers.Dropout(self.dropout_rate, name="proj_drop")
            if self.dropout_rate > 0.0
            else None
        )

        self.rope = (
            VideoRoPE3D(head_dim=self.head_dim, rope_theta=self.rope_theta, name="rope")
            if self.use_rope
            else None
        )

        self.fc1 = keras.layers.Dense(
            self.hidden_dim,
            use_bias=True,
            kernel_initializer=clone_initializer(keras.initializers.get(base_kernel_init)),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="mlp_fc1",
        )
        self.act = keras.layers.Activation("gelu", name="mlp_act")
        self.drop1 = (
            keras.layers.Dropout(self.dropout_rate, name="mlp_drop1")
            if self.dropout_rate > 0.0
            else None
        )
        self.fc2 = keras.layers.Dense(
            self.dim,
            use_bias=True,
            kernel_initializer=clone_initializer(keras.initializers.get(rescaled_kernel_init)),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="mlp_fc2",
        )
        self.drop2 = (
            keras.layers.Dropout(self.dropout_rate, name="mlp_drop2")
            if self.dropout_rate > 0.0
            else None
        )

        logger.info(
            f"Initialized LeVJEPABlock with dim={self.dim}, num_heads={self.num_heads}, "
            f"use_rope={self.use_rope}, layer_id={self.layer_id}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every sub-layer in computational order.

        :param input_shape: Shape of ``x``, ``(batch, num_tokens, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not rank-3 or its last
            dimension does not equal ``dim``.
        """
        if self.built:
            return

        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input (batch, tokens, dim), got {input_shape}")
        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(
                f"Input last dimension ({input_shape[-1]}) must equal dim ({self.dim})"
            )

        self.norm1.build(input_shape)
        self.qkv.build(input_shape)
        self.proj.build(input_shape)
        if self.attn_drop is not None:
            self.attn_drop.build(input_shape)
        if self.proj_drop is not None:
            self.proj_drop.build(input_shape)

        self.norm2.build(input_shape)
        self.fc1.build(input_shape)
        hidden_shape = input_shape[:-1] + (self.hidden_dim,)
        if self.drop1 is not None:
            self.drop1.build(hidden_shape)
        self.fc2.build(hidden_shape)
        if self.drop2 is not None:
            self.drop2.build(input_shape)

        # VideoRoPE3D owns no weights and no build(); nothing to do here.

        super().build(input_shape)

    def call(
        self,
        inputs: Any,
        num_frames: Optional[int] = None,
        height_patches: Optional[int] = None,
        width_patches: Optional[int] = None,
        token_ids: Optional[Any] = None,
        attn_mask: Optional[Any] = None,
        training: Optional[bool] = None,
    ) -> Any:
        """Apply the pre-norm attention + MLP block.

        :param inputs: Token sequence, ``(batch, num_tokens, dim)``.
        :type inputs: keras.KerasTensor
        :param num_frames: Number of frame positions in the video grid.
            Required when ``use_rope=True`` and ``token_ids`` is not given
            (default identity grid).
        :type num_frames: Optional[int]
        :param height_patches: Number of patches along the height axis.
            Required when ``use_rope=True``.
        :type height_patches: Optional[int]
        :param width_patches: Number of patches along the width axis.
            Required when ``use_rope=True``.
        :type width_patches: Optional[int]
        :param token_ids: Optional true flat grid index per PATCH token
            (excluding the prefix tokens), shape ``(num_patches,)`` or
            ``(batch, num_patches)`` -- forwarded straight to
            :class:`VideoRoPE3D`. ``None`` defaults to the no-dropping
            identity grid.
        :type token_ids: Optional[Any]
        :param attn_mask: Optional pre-built boolean KEEP predicate (``True``
            = attend), broadcastable against the ``(batch, num_heads,
            num_tokens, num_tokens)`` attention logits -- typically
            :func:`~dl_techniques.models.vision.levjepa.masking.build_block_causal_mask`'s
            output. ``None`` means full (unmasked) attention.
        :type attn_mask: Optional[Any]
        :param training: Standard Keras training flag.
        :type training: Optional[bool]
        :return: Output sequence, same shape as ``inputs``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``use_rope=True`` and ``height_patches`` or
            ``width_patches`` is not given.
        """
        if self.use_rope and (height_patches is None or width_patches is None):
            raise ValueError(
                "LeVJEPABlock(use_rope=True) requires height_patches and "
                "width_patches at call time."
            )

        residual = inputs
        y = self.norm1(inputs, training=training)

        batch_size = keras.ops.shape(y)[0]
        num_tokens = keras.ops.shape(y)[1]

        qkv = self.qkv(y, training=training)
        qkv = keras.ops.reshape(qkv, (batch_size, num_tokens, 3, self.num_heads, self.head_dim))
        # (3, B, H, N, d)
        qkv = keras.ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rope:
            p = self.num_prefix_tokens
            if p > 0:
                q_prefix, q_body = q[:, :, :p, :], q[:, :, p:, :]
                k_prefix, k_body = k[:, :, :p, :], k[:, :, p:, :]
            else:
                q_prefix = k_prefix = None
                q_body, k_body = q, k

            q_body, k_body = self.rope(
                q_body,
                k_body,
                num_frames=num_frames,
                height_patches=height_patches,
                width_patches=width_patches,
                token_ids=token_ids,
                training=training,
            )

            if p > 0:
                q = keras.ops.concatenate([q_prefix, q_body], axis=2)
                k = keras.ops.concatenate([k_prefix, k_body], axis=2)
            else:
                q, k = q_body, k_body

        logits = keras.ops.matmul(q, keras.ops.moveaxis(k, -1, -2)) * self._scale

        if attn_mask is not None:
            logits = apply_attention_mask(logits, attn_mask, rescue_axis=-1)

        attn = keras.ops.softmax(keras.ops.cast(logits, "float32"), axis=-1)
        attn = keras.ops.cast(attn, v.dtype)

        if self.attn_drop is not None:
            attn = self.attn_drop(attn, training=training)

        out = keras.ops.matmul(attn, v)  # (B, H, N, d)
        out = keras.ops.transpose(out, (0, 2, 1, 3))  # (B, N, H, d)
        out = keras.ops.reshape(out, (batch_size, num_tokens, self.dim))

        out = self.proj(out, training=training)
        if self.proj_drop is not None:
            out = self.proj_drop(out, training=training)

        x = residual + out

        residual2 = x
        h = self.norm2(x, training=training)
        h = self.fc1(h, training=training)
        h = self.act(h)
        if self.drop1 is not None:
            h = self.drop1(h, training=training)
        h = self.fc2(h, training=training)
        if self.drop2 is not None:
            h = self.drop2(h, training=training)

        return residual2 + h

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return ``input_shape`` unchanged -- the block preserves shape.

        :param input_shape: Shape of ``x``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``input_shape``, unchanged.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the layer for serialization.

        :return: Dictionary holding every ``__init__`` parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "qkv_bias": self.qkv_bias,
                "use_rope": self.use_rope,
                "rope_theta": self.rope_theta,
                "num_prefix_tokens": self.num_prefix_tokens,
                "dropout_rate": self.dropout_rate,
                "attention_dropout_rate": self.attention_dropout_rate,
                "layer_id": self.layer_id,
                "init_std": self.init_std,
                "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            }
        )
        return config


# ---------------------------------------------------------------------
