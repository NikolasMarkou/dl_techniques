"""
Ideogram4 self-attention: fused QKV + per-head RMS QK-norm + mRoPE + segment mask.

This layer ports the Ideogram4 ``Ideogram4Attention`` module to Keras 3. It is a
multi-head self-attention block specialized for the Ideogram4 packed-sequence DiT:

- **Fused QKV**: a single bias-free ``Dense(3 * hidden_size)`` projection, reshaped
  to ``(B, L, 3, num_heads, head_dim)`` and split into q / k / v.
- **Per-head RMS QK-norm**: q and k are RMS-normalized over the ``head_dim`` axis
  (``axis=-1``) by two independent :class:`RMSNorm` sub-layers (scale shape
  ``(head_dim,)``), stabilizing attention logits. This reuses the repository
  ``RMSNorm`` (no re-implementation).
- **mRoPE injection**: precomputed ``cos`` / ``sin`` tables (shape
  ``(B, L, head_dim)``, produced upstream by ``Ideogram4MRoPE``) are applied to
  q and k via the shared :func:`apply_rotary_pos_emb` helper. This layer does
  NOT own / instantiate the mRoPE layer — the transformer passes cos/sin in.
- **Block-diagonal segment mask**: tokens attend only to tokens sharing the same
  ``segment_ids`` value. An ADDITIVE mask (``0`` same-segment, large-negative
  cross-segment) is built and added to the pre-softmax scores. See the
  ``# DECISION`` anchor in :meth:`call` for why the additive form is used instead
  of the PyTorch boolean keep-mask.
- **SDPA**: scaled dot-product attention is computed manually with ``keras.ops``
  (matmul, scale by ``1/sqrt(head_dim)``, add mask, softmax, matmul v), then the
  output is projected by a bias-free ``Dense(hidden_size)``.

PyTorch reference (faithfully ported)::

    qkv = self.qkv(x).view(B, L, 3, num_heads, head_dim)
    q, k, v = qkv.unbind(dim=2)
    q = self.norm_q(q); k = self.norm_k(k)              # RMS over head_dim
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))    # (B, num_heads, L, head_dim)
    q, k = _apply_rotary_pos_emb(q, k, cos, sin)
    attn_mask = (segment_ids[:, :, None] == segment_ids[:, None, :])[:, None]  # bool keep
    out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    out = out.transpose(1, 2).reshape(B, L, hidden_size)
    return self.o(out)
"""

import keras
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.embedding.multi_axis_rope import apply_rotary_pos_emb

# ---------------------------------------------------------------------

# Large-negative fill for masked-out (cross-segment) attention logits.
_MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="dl_techniques.layers")
class Ideogram4Attention(keras.layers.Layer):
    """Ideogram4 packed self-attention with QK-norm, mRoPE, and a segment mask.

    Computes multi-head self-attention over a packed token sequence where the
    rotary position embedding (``cos`` / ``sin``) and the per-token
    ``segment_ids`` are supplied by the caller. Attention is restricted to be
    block-diagonal in ``segment_ids`` (tokens attend only within their own
    segment).

    :param hidden_size: Model / embedding dimensionality. Must be divisible by
        ``num_heads``.
    :type hidden_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param eps: Epsilon for the per-head RMS QK-norm. Defaults to ``1e-5``
        (matching the PyTorch reference).
    :type eps: float
    :param kwargs: Additional ``keras.layers.Layer`` arguments.

    :raises ValueError: If ``hidden_size`` is not divisible by ``num_heads``,
        or if either is not a positive integer.

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
        super().__init__(**kwargs)

        # --- validation -------------------------------------------------
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be a positive integer, got {hidden_size}"
            )
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(
                f"num_heads must be a positive integer, got {num_heads}"
            )
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        # --- store config ----------------------------------------------
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.eps = float(eps)
        self._inv_sqrt_dim = 1.0 / (self.head_dim ** 0.5)

        # --- sub-layers (created here, built in build()) ---------------
        self.qkv = keras.layers.Dense(
            3 * hidden_size, use_bias=False, name="qkv"
        )
        # Per-head RMS norm over head_dim; learnable scale of shape (head_dim,).
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
        """Build the fused-QKV, QK-norm, and output sub-layers.

        :param input_shape: Shape of ``x``, expected ``(B, L, hidden_size)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the last input dimension is not ``hidden_size``.
        """
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
        """Run packed self-attention.

        :param x: Token features of shape ``(B, L, hidden_size)``.
        :type x: keras.KerasTensor
        :param segment_ids: Integer segment id per token, shape ``(B, L)``.
            Tokens attend only to others with the same id.
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
        q = qkv[:, :, 0]  # (B, L, num_heads, head_dim)
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        # Per-head RMS QK-norm over the head_dim axis (axis=-1).
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

        # DECISION plan_2026-06-12_59a18a10/D-004: build an ADDITIVE block-diagonal
        # segment mask (0.0 same-segment, -1e9 cross-segment) and add it to the
        # pre-softmax scores, rather than porting PyTorch's BOOLEAN keep-mask fed
        # to F.scaled_dot_product_attention. Reason: SDPA is implemented manually
        # in keras.ops here (there is no backend-agnostic boolean-mask SDPA op),
        # so the mask must be folded into the softmax logits numerically. Do NOT
        # replace with `keras.ops.where(keep, scores, -inf)` using a boolean keep
        # mask + raw -inf: -inf rows (a token with an empty segment) produce NaN
        # after softmax; the finite -1e9 additive form keeps softmax well-defined.
        # See decisions.md D-004.
        seg_i = keras.ops.expand_dims(segment_ids, axis=2)  # (B, L, 1)
        seg_j = keras.ops.expand_dims(segment_ids, axis=1)  # (B, 1, L)
        same_segment = keras.ops.equal(seg_i, seg_j)  # (B, L, L) bool
        additive_mask = keras.ops.where(
            same_segment,
            keras.ops.zeros_like(same_segment, dtype=scores.dtype),
            keras.ops.cast(_MASK_NEG, scores.dtype),
        )
        # (B, L, L) -> (B, 1, L, L) to broadcast over heads.
        additive_mask = keras.ops.expand_dims(additive_mask, axis=1)
        scores = scores + additive_mask

        attn = keras.ops.softmax(scores, axis=-1)
        out = keras.ops.matmul(attn, v)  # (B, num_heads, L, head_dim)

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

# ---------------------------------------------------------------------
