"""Pre-norm self-attention and MLP transformer block for LeVJEPA.

``LeVJEPABlock`` ports the LeVJEPA reference's ``Block``:

    x = x + Attn(LN(x)); x = x + MLP(LN(x))

with plain residual addition on both branches and no ``LayerScale``. The
attention is written out here (fused QKV, head split, optional rotary
embedding, scaled dot product, output projection) instead of delegating to
``layers/attention/multi_head_attention.py``, because :class:`VideoRoPE3D` has
to rotate ``q`` and ``k`` after the head split and before the softmax, which the
shared attention layer gives no hook for. Block-depth rescaling of the ``proj``
and ``fc2`` kernels is expressed as a pre-scaled initializer standard deviation
rather than a post-build weight division.

An attention mask arrives as a pre-built boolean keep predicate, where ``True``
means attend; the layer infers no polarity of its own. ``use_rope=True``
requires ``height_patches`` and ``width_patches`` at call time. ``layer_id``
defaults to ``None``, which disables depth rescaling.

References:
    - LeVJEPA PyTorch reference, ``module.py::Block`` / ``Attention`` /
      ``RoPEAttention``.
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
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

# DECISION D-011: no LayerScale sub-layers; the reference uses plain residual
# addition on both branches. See decisions.md.

REFERENCE_INIT_STD = 0.02

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.levjepa.blocks")
class LeVJEPABlock(keras.layers.Layer):
    """Apply pre-norm self-attention and an MLP, each inside a residual branch.

    The block normalizes before each sub-layer and adds the sub-layer output
    back to its input. Attention runs through a fused QKV projection, an
    optional 3-axis video rotary embedding on ``q`` and ``k``, and an output
    projection. The first ``num_prefix_tokens`` tokens pass through the rotation
    unrotated.

    Architecture:

    .. code-block:: text

                           x [B, N, D]
                                │
            ┌───────────────────┤
            │                   ▼
            │         ┌───────────────────┐
            │         │ LayerNorm eps 1e-6│
            │         └─────────┬─────────┘
            │                   ▼
            │         ┌───────────────────┐
            │         │ self-attention    │
            │         │  (see below)      │
            │         └─────────┬─────────┘
            │                   ▼
            │         ┌───────────────────┐
            └────────►│ add               │
                      └─────────┬─────────┘
                                ▼
            ┌───────────────────┤
            │                   ▼
            │         ┌───────────────────┐
            │         │ LayerNorm eps 1e-6│
            │         └─────────┬─────────┘
            │                   ▼
            │         ┌───────────────────┐
            │         │ MLP               │
            │         │  (see below)      │
            │         └─────────┬─────────┘
            │                   ▼
            │         ┌───────────────────┐
            └────────►│ add               │
                      └─────────┬─────────┘
                                ▼
                          x' [B, N, D]

    Self-attention:

    .. code-block:: text

               y [B, N, D] after LayerNorm
                                ▼
                ┌───────────────────────────────┐
                │ Dense(3D), qkv_bias           │
                └───────────────┬───────────────┘
                           [B, N, 3D]
                                ▼
                ┌───────────────────────────────┐
                │ reshape to [B, N, 3, H, d]    │
                │  transpose to [3, B, H, N, d] │
                └───────────────┬───────────────┘
                   q, k, v  [B, H, N, d] each
                                ▼
                ┌───────────────────────────────┐
                │ VideoRoPE3D rotates q, k      │
                │  tokens from num_prefix_tokens│
                │  ('use_rope' only)            │
                └───────────────┬───────────────┘
                                ▼
                ┌───────────────────────────────┐
                │ logits = q k^T * scale        │
                └───────────────┬───────────────┘
                          [B, H, N, N]
                                ▼
                ┌───────────────────────────────┐
                │ apply_attention_mask          │
                │  (attn_mask, True = attend)   │
                └───────────────┬───────────────┘
                                ▼
                ┌───────────────────────────────┐
                │ softmax in float32, cast back │
                │  attn_drop (optional)         │
                └───────────────┬───────────────┘
                                ▼
                ┌───────────────────────────────┐
                │ attn @ v, merge heads         │
                └───────────────┬───────────────┘
                            [B, N, D]
                                ▼
                ┌───────────────────────────────┐
                │ Dense(D), proj_drop optional  │
                └───────────────┬───────────────┘
                                ▼
                   attention output [B, N, D]

    MLP:

    .. code-block:: text

               h [B, N, D] after LayerNorm
                                ▼
                ┌───────────────────────────────┐
                │ Dense(hidden_dim)             │
                └───────────────┬───────────────┘
                      [B, N, D * mlp_ratio]
                                ▼
                ┌───────────────────────────────┐
                │ gelu                          │
                │  drop1 (optional)             │
                └───────────────┬───────────────┘
                                ▼
                ┌───────────────────────────────┐
                │ Dense(D)                      │
                │  drop2 (optional)             │
                └───────────────┬───────────────┘
                                ▼
                      mlp output [B, N, D]

    Kernel initializer std:

    .. code-block:: text

        kernel      std
        ──────      ────────────────────────────────
        qkv, fc1    init_std
        proj, fc2   init_std / sqrt(2 * layer_id)
                    init_std, when layer_id is None

    :param dim: Model dimension. Must be positive and divisible by
        ``num_heads``.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param mlp_ratio: MLP hidden-dimension multiplier. Must be positive.
        Defaults to ``4.0``.
    :type mlp_ratio: float
    :param qkv_bias: Give the fused QKV projection a bias. Defaults to ``True``,
        matching the reference.
    :type qkv_bias: bool
    :param use_rope: Rotate ``q`` and ``k`` with :class:`VideoRoPE3D` before the
        softmax. When ``True``, ``call()`` requires ``num_frames``,
        ``height_patches`` and ``width_patches``. Defaults to ``False``.
    :type use_rope: bool
    :param rope_theta: Rotary base frequency, forwarded to ``VideoRoPE3D`` when
        ``use_rope=True``. Defaults to ``10000.0``.
    :type rope_theta: float
    :param num_prefix_tokens: Number of leading tokens, the CLS tokens, that
        pass through the rotation unrotated. Defaults to ``1``.
    :type num_prefix_tokens: int
    :param dropout_rate: Dropout applied after the output projection and inside
        the MLP. Must be in ``[0, 1]``. Defaults to ``0.0``.
    :type dropout_rate: float
    :param attention_dropout_rate: Dropout applied to the post-softmax attention
        weights. Must be in ``[0, 1]``. Defaults to ``0.0``.
    :type attention_dropout_rate: float
    :param layer_id: 1-indexed block position within the encoder stack. Sets the
        initializer std of ``proj`` and ``fc2`` to
        ``init_std / sqrt(2 * layer_id)``, matching the reference's
        ``_rescale_blocks`` weight division. ``None`` (default) leaves both at
        ``init_std``.
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
    :param kwargs: Additional ``Layer`` base-class arguments.

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

    :raises ValueError: From ``__init__``, if ``dim`` is not divisible by
        ``num_heads``, if ``dim``, ``num_heads`` or ``mlp_ratio`` is not
        positive, if ``num_prefix_tokens`` is negative, or if ``dropout_rate``
        or ``attention_dropout_rate`` falls outside ``[0, 1]``.

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

        Arguments are documented on the class.

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

        # DECISION D-012: depth rescaling is a pre-scaled initializer std, never
        # a post-build .assign(), which StatelessScope discards. decisions.md.
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
        """Apply the pre-norm attention and MLP block.

        :param inputs: Token sequence, ``(batch, num_tokens, dim)``.
        :type inputs: keras.KerasTensor
        :param num_frames: Number of frame positions in the video grid.
            Required when ``use_rope=True`` and ``token_ids`` is not given.
        :type num_frames: Optional[int]
        :param height_patches: Number of patches along the height axis.
            Required when ``use_rope=True``.
        :type height_patches: Optional[int]
        :param width_patches: Number of patches along the width axis.
            Required when ``use_rope=True``.
        :type width_patches: Optional[int]
        :param token_ids: Optional flat grid index per patch token, excluding
            the prefix tokens, of shape ``(num_patches,)`` or
            ``(batch, num_patches)``, forwarded to :class:`VideoRoPE3D`.
            ``None`` uses the identity grid.
        :type token_ids: Optional[Any]
        :param attn_mask: Optional pre-built boolean keep predicate, where
            ``True`` means attend, broadcastable against the
            ``(batch, num_heads, num_tokens, num_tokens)`` logits. Typically the
            output of
            :func:`~dl_techniques.models.vision.levjepa.masking.build_block_causal_mask`.
            ``None`` means unmasked attention.
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

        # The softmax runs in float32 so a half-precision block keeps a stable
        # normalization, then casts back to the value dtype.
        attn = keras.ops.softmax(keras.ops.cast(logits, "float32"), axis=-1)
        attn = keras.ops.cast(attn, v.dtype)

        if self.attn_drop is not None:
            attn = self.attn_drop(attn, training=training)

        out = keras.ops.matmul(attn, v)
        out = keras.ops.transpose(out, (0, 2, 1, 3))
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
        """Return ``input_shape`` unchanged; the block preserves shape.

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