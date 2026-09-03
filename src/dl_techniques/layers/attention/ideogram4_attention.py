"""Ideogram4Attention, fused-QKV multi-head self-attention over a packed
token sequence, with per-head RMS QK-norm, mRoPE, and a segment mask.

One tensor packs several independent sequences end to end; a per-token
segment id says which sequence a token belongs to. Attention is
restricted to be block-diagonal in that id, added to the pre-softmax
scores rather than applied as a boolean keep-mask, since there is no
backend-agnostic fused SDPA op to hand a boolean mask to. Query and key
are RMS-normalized per head before the rotary injection, which bounds
the score magnitude independently of activation scale.

The layer takes ``cos``/``sin`` rotary tables and ``segment_ids`` as call
arguments; it does not own or build the rotary embedding. There is no
separate padding mask: a padding token needs its own segment id, or it
is attended to as an ordinary token.

References:
    - Vaswani et al., 2017. Attention Is All You Need. (https://arxiv.org/abs/1706.03762)
    - Zhang & Sennrich, 2019. Root Mean Square Layer Normalization. (https://arxiv.org/abs/1910.07467)
    - Su et al., 2021. RoFormer: Enhanced Transformer with Rotary Position Embedding. (https://arxiv.org/abs/2104.09864)
"""

import keras
from typing import Any, Dict, Optional, Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.embedding.multi_axis_rope import apply_rotary_pos_emb

from .common import (
    compute_attention_scale,
    validate_head_divisibility
)
from dl_techniques.utils.keras_registration import register_dl_technique

# Not the same object as common.MASK_BIAS_VALUE: this layer casts straight
# to scores.dtype instead of going through common.mask_dtype(...). Stays
# finite under softmax because the diagonal is always same-segment.
_MASK_NEG = -1e9


@register_dl_technique("dl_techniques.layers.attention.ideogram4_attention")
class Ideogram4Attention(keras.layers.Layer):
    """Compute packed self-attention with per-head QK-norm, mRoPE, and a
    segment mask.

    Computes multi-head self-attention over a packed token sequence where the
    rotary position embedding (``cos`` / ``sin``) and the per-token
    ``segment_ids`` are supplied by the caller. Attention is restricted to be
    block-diagonal in ``segment_ids`` (tokens attend only within their own
    segment).

    The ``hidden_size % num_heads`` check and the ``1 / sqrt(head_dim)``
    temperature come from :mod:`~dl_techniques.layers.attention.common`.
    The QK-norms are the repository
    :class:`~dl_techniques.layers.norms.rms_norm.RMSNorm`, and the rotary
    injection is the shared
    :func:`~dl_techniques.layers.embedding.multi_axis_rope.apply_rotary_pos_emb`.

    Shapes below use ``B`` = batch, ``L`` = packed sequence length,
    ``H`` = num_heads, ``D`` = head_dim = ``hidden_size / num_heads``.

    Architecture:

    .. code-block:: text

              x [B, L, hidden]     segment_ids [B, L]
                     │              cos, sin [B, L, D]
                     │                     │
                     ▼                     │
          ┌────────────────────────┐       │  segment_ids and the mRoPE
          │ qkv                    │       │  tables are call arguments.
          │ Dense(3 * hidden),     │       │  This layer owns no rope
          │ no bias                │       │  sub-layer; the transformer
          └───────────┬────────────┘       │  computes them once and
                      ▼                    │  passes them in.
          reshape [B, L, 3, H, D]          │
          slice into q, k, v               │
                      │                    │
                      ▼                    │
          ┌────────────────────────┐       │
          │ norm_q on q            │       │
          │ norm_k on k            │       │
          │ RMSNorm, per head,     │       │
          │ over D. v is not       │       │
          │ normalized.            │       │
          └───────────┬────────────┘       │
                      ▼                    │
          transpose to [B, H, L, D]        │
                      │                    │
                      ▼                    ▼
          ┌─────────────────────────────────────────┐
          │ apply_rotary_pos_emb(q, k, cos, sin)    │
          │ mRoPE goes into q and k only            │
          └───────────────────┬─────────────────────┘
                              ▼
                 S = q~ . k~^T * (1/sqrt(D))
                              │  [B, H, L, L]
                              ▼
          ┌─────────────────────────────────────────┐
          │ block-diagonal segment mask, built here │
          │ rather than by the common mask helper   │
          │   M = where(seg_i == seg_j, 0.0, -1e9)  │
          │       [B, 1, L, L], broadcast over H    │
          │   S = S + M                             │
          │ added, so softmax stays a proper        │
          │ distribution. The diagonal is always    │
          │ same-segment, so no row is entirely     │
          │ -1e9 even when fp16 turns it into -inf. │
          └───────────────────┬─────────────────────┘
                              ▼
                     softmax(S, axis=-1)
                              ▼
                   multiply by v, merge heads
                              │  [B, L, hidden]
                              ▼
          ┌─────────────────────────────────────────┐
          │ o : Dense(hidden), no bias              │
          └───────────────────┬─────────────────────┘
                              ▼
                   output  [B, L, hidden]

        There is no attention_mask parameter. Padding must be given its
        own segment id, or it is attended to as an ordinary token.

    :param hidden_size: Model / embedding dimensionality. Must be divisible by
        ``num_heads``.
    :type hidden_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param eps: Epsilon for the per-head RMS QK-norm. Defaults to ``1e-5``
        (matching the PyTorch reference).
    :type eps: float
    :param kwargs: Additional ``keras.layers.Layer`` arguments.
    :type kwargs: Any

    :raises ValueError: If ``hidden_size`` is not a positive integer.
    :raises ValueError: If ``num_heads`` is not a positive integer.
    :raises ValueError: If ``hidden_size`` is not divisible by ``num_heads``.
    :raises ValueError: If ``eps`` is not positive.
    :raises ValueError: From ``build()``, if the input is not 3D or its last
        dimension is not ``hidden_size``.

    .. warning::
        There is no ``attention_mask`` parameter. Masking is driven entirely
        by ``segment_ids``; a pad token needs its own segment id, otherwise it
        is attended to as an ordinary token of whichever segment it was
        assigned.

    Input/Output:
        ``call(x, segment_ids, cos, sin)`` with
        ``x: (B, L, hidden_size)``, ``segment_ids: (B, L)``,
        ``cos / sin: (B, L, head_dim)`` returns ``(B, L, hidden_size)``.

    Example:
        >>> attn = Ideogram4Attention(hidden_size=256, num_heads=4)
        >>> x = keras.random.normal((2, 8, 256))
        >>> seg = keras.ops.zeros((2, 8), dtype="int32")
        >>> cos = keras.ops.ones((2, 8, 64))
        >>> sin = keras.ops.zeros((2, 8, 64))
        >>> y = attn(x, seg, cos, sin)
        >>> y.shape
        (2, 8, 256)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        eps: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        """Validate the configuration and create the four sub-layers.

        :param hidden_size: Model dimensionality, divisible by ``num_heads``.
        :type hidden_size: int
        :param num_heads: Number of attention heads.
        :type num_heads: int
        :param eps: Epsilon for the per-head RMS QK-norm.
        :type eps: float
        :param kwargs: Additional ``keras.layers.Layer`` arguments.
        :type kwargs: Any

        :raises ValueError: If ``hidden_size`` or ``num_heads`` is not a
            positive integer, if ``hidden_size`` is not divisible by
            ``num_heads``, or if ``eps`` is not positive.
        """
        super().__init__(**kwargs)

        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be a positive integer, got {hidden_size}"
            )
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(
                f"num_heads must be a positive integer, got {num_heads}"
            )
        # `dim_name` keeps the message naming this layer's own constructor
        # argument, matching the message that stood here before the swap.
        validate_head_divisibility(
            hidden_size, num_heads, dim_name="hidden_size"
        )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.eps = float(eps)
        # Computed here as a Python float, not in call() with keras.ops.sqrt:
        # a backend tensor made in __init__ can leak out of a symbolic graph.
        # head_dim ** -0.5 is not bit-identical to this form (differs in the
        # last ULP for 16 of 27 tested head dims).
        self._inv_sqrt_dim = compute_attention_scale(self.head_dim)

        self.qkv = keras.layers.Dense(
            3 * hidden_size, use_bias=False, name="qkv"
        )
        # Each carries a learnable scale of shape (head_dim,), which is why
        # build() gives them a 4D shape.
        self.norm_q = RMSNorm(axis=-1, epsilon=self.eps, name="norm_q")
        self.norm_k = RMSNorm(axis=-1, epsilon=self.eps, name="norm_k")
        self.o = keras.layers.Dense(
            hidden_size, use_bias=False, name="o"
        )

        logger.debug(
            f"Initialized Ideogram4Attention(hidden_size={hidden_size}, "
            f"num_heads={num_heads}, head_dim={self.head_dim}, eps={self.eps})"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the fused-QKV, QK-norm and output sub-layers.

        The QK-norms are built at a 4D shape whose last axis is ``head_dim``,
        so each scale parameter comes out ``(head_dim,)`` rather than
        ``(hidden_size,)``.

        :param input_shape: Shape of ``x``, expected ``(B, L, hidden_size)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not rank 3, or if its last
            dimension is not ``hidden_size``.
        """
        if self.built:
            return

        if len(input_shape) != 3 or input_shape[-1] != self.hidden_size:
            raise ValueError(
                f"Ideogram4Attention expects x of shape (B, L, hidden_size="
                f"{self.hidden_size}), got input_shape {input_shape}"
            )

        # Fused QKV consumes (B, L, hidden_size).
        self.qkv.build(input_shape)

        # QK-norm normalizes over the per-head dim; build with a shape whose
        # last axis is head_dim so the scale parameter is (head_dim,).
        qk_norm_shape = (None, None, None, self.head_dim)
        self.norm_q.build(qk_norm_shape)
        self.norm_k.build(qk_norm_shape)

        # Output projection consumes the re-merged (B, L, hidden_size).
        self.o.build((input_shape[0], input_shape[1], self.hidden_size))

        super().build(input_shape)

    def call(
        self,
        x: keras.KerasTensor,
        segment_ids: keras.KerasTensor,
        cos: keras.KerasTensor,
        sin: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run packed self-attention over one batch of packed sequences.

        Project, QK-norm, inject mRoPE, score, add the segment mask, softmax,
        weight v, merge heads, project out.

        :param x: Token features of shape ``(B, L, hidden_size)``.
        :type x: keras.KerasTensor
        :param segment_ids: Integer segment id per token, shape ``(B, L)``.
            Tokens attend only to others carrying the same id. Padding needs
            its own id; there is no separate padding mask.
        :type segment_ids: keras.KerasTensor
        :param cos: mRoPE cosine table, shape ``(B, L, head_dim)``.
        :type cos: keras.KerasTensor
        :param sin: mRoPE sine table, shape ``(B, L, head_dim)``.
        :type sin: keras.KerasTensor
        :param training: Forwarded to sub-layers (unused by RMSNorm/Dense here).
        :type training: Optional[bool]
        :return: Attention output of shape ``(B, L, hidden_size)``.
        :rtype: keras.KerasTensor
        """
        input_shape = keras.ops.shape(x)
        batch = input_shape[0]
        length = input_shape[1]

        # Fused QKV -> (B, L, 3, num_heads, head_dim).
        qkv = self.qkv(x)
        qkv = keras.ops.reshape(
            qkv, (batch, length, 3, self.num_heads, self.head_dim)
        )
        # Each slice is (B, L, num_heads, head_dim).
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        # Per-head RMS QK-norm over the head_dim axis. v is left alone.
        q = self.norm_q(q, training=training)
        k = self.norm_k(k, training=training)

        # (B, L, num_heads, head_dim) -> (B, num_heads, L, head_dim).
        q = keras.ops.transpose(q, (0, 2, 1, 3))
        k = keras.ops.transpose(k, (0, 2, 1, 3))
        v = keras.ops.transpose(v, (0, 2, 1, 3))

        # Inject mRoPE into q / k (cos/sin broadcast over heads internally).
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product scores: (B, num_heads, L, L).
        scores = keras.ops.matmul(
            q, keras.ops.transpose(k, (0, 1, 3, 2))
        )
        scores = scores * self._inv_sqrt_dim

        # DECISION plan_2026-06-12_59a18a10/D-004: mask is added to the logits, not a
        # boolean keep-mask -- keras.ops has no fused SDPA op to hand one to. -1e9 is -inf under fp16, but the diagonal is always same-segment. See decisions.md.
        # (B, L, 1) and (B, 1, L) broadcast to a (B, L, L) bool.
        seg_i = keras.ops.expand_dims(segment_ids, axis=2)
        seg_j = keras.ops.expand_dims(segment_ids, axis=1)
        same_segment = keras.ops.equal(seg_i, seg_j)
        additive_mask = keras.ops.where(
            same_segment,
            keras.ops.zeros_like(same_segment, dtype=scores.dtype),
            keras.ops.cast(_MASK_NEG, scores.dtype),
        )
        # (B, L, L) -> (B, 1, L, L) to broadcast over heads.
        additive_mask = keras.ops.expand_dims(additive_mask, axis=1)
        scores = scores + additive_mask

        attn = keras.ops.softmax(scores, axis=-1)
        # -> (B, num_heads, L, head_dim)
        out = keras.ops.matmul(attn, v)

        # (B, num_heads, L, head_dim) -> (B, L, hidden_size).
        out = keras.ops.transpose(out, (0, 2, 1, 3))
        out = keras.ops.reshape(out, (batch, length, self.hidden_size))

        return self.o(out)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape (identical to ``x``'s shape).

        :param input_shape: Shape of ``x`` ``(B, L, hidden_size)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(B, L, hidden_size)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the serialization config.

        :return: Dictionary with all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "eps": self.eps,
            }
        )
        return config
