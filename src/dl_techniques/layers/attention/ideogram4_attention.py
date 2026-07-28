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

Architecture:
    A single fused ``Dense(3 * hidden_size, use_bias=False)`` produces q / k / v in
    one matmul, which is reshaped to ``(B, L, 3, H, D)`` and sliced. q and k pass
    through two independent :class:`RMSNorm` instances over ``head_dim``, then
    receive mRoPE from caller-supplied ``cos`` / ``sin`` tables. Scores are formed
    manually, biased by an additive block-diagonal segment mask, normalized with
    ``keras.ops.softmax``, applied to v, and projected back by a bias-free
    ``Dense(hidden_size)``.

    Two structural properties are load-bearing:

    -   The layer does **not** own the mRoPE layer. ``cos`` / ``sin`` are call
        arguments produced upstream by ``Ideogram4MRoPE``, so one rope table is
        computed once per forward pass and shared by every block.
    -   Masking is ADDITIVE, not a boolean keep-mask handed to a fused SDPA op.
        See the ``plan_2026-06-12_59a18a10/D-004`` anchor in :meth:`call`.

Foundational Mathematics:
    For head ``h`` with per-head dimension ``D = hidden_size / H``::

        q, k, v   = split( W_qkv x ,  3 )              # fused projection
        q_hat     = RMSNorm(q) ,   k_hat = RMSNorm(k)  # per-head, over D
        q~, k~    = mRoPE(q_hat, k_hat ; cos, sin)     # rotary injection
        S_ij      = (q~_i . k~_j) / sqrt(D)  +  M_ij
        M_ij      = 0                if segment_id_i == segment_id_j
                    -1e9             otherwise
        out_i     = sum_j softmax_j(S_ij) v_j

    RMS QK-norm rescales each query/key to a fixed L2 radius before the dot
    product, which bounds ``|S_ij|`` independently of the activation scale and is
    what keeps the logits stable without a learned temperature. The mask is added
    in the *logit* domain rather than multiplied in the probability domain, so
    ``softmax`` remains a proper distribution over each token's own segment.

References:
    - Vaswani et al. (2017). "Attention Is All You Need". NeurIPS.
      (https://arxiv.org/abs/1706.03762)
    - Zhang & Sennrich (2019). "Root Mean Square Layer Normalization". NeurIPS.
      (https://arxiv.org/abs/1910.07467)
    - Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position
      Embedding". (https://arxiv.org/abs/2104.09864)
"""

# ---------------------------------------------------------------------

import keras
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .common import compute_attention_scale, validate_head_divisibility
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.embedding.multi_axis_rope import apply_rotary_pos_emb

# ---------------------------------------------------------------------

# Large-negative fill for masked-out (cross-segment) attention logits.
#
# R13 cross-reference: numerically the same magnitude as `common.MASK_BIAS_VALUE`,
# and deliberately NOT replaced by it. The shared constant carries a contract —
# materialize the bias in `common.mask_dtype(...)`, never in the layer's compute
# dtype — that this layer does not follow: it casts straight to `scores.dtype`.
# Importing the name without porting the dtype discipline would advertise a safety
# guarantee the code does not implement.
#
# That said, this layer is NOT exposed to the systemic fp16 mask-NaN hazard
# catalogued elsewhere in this package, and the reason is structural rather than
# lucky. `call()` uses the `ops.where(same_segment, 0.0, -1e9)` form, not the
# arithmetic `scores + (1 - mask) * -1e9` form. Under `mixed_float16` the cast makes
# this constant `-inf`, but `where` never multiplies it by zero, so `0 * -inf = NaN`
# cannot arise. The softmax is then well-defined because the DIAGONAL is always
# same-segment, so every row keeps at least one finite (0.0) entry and the
# max-subtraction inside softmax stays finite. Verified empirically in fp16, not
# reasoned about on paper: bias `[0, 0, -inf, -inf]` -> softmax
# `[0.269, 0.731, 0, 0]`. Do NOT "modernize" this into the arithmetic form.
_MASK_NEG = -1e9


@keras.saving.register_keras_serializable()
class Ideogram4Attention(keras.layers.Layer):
    """Ideogram4 packed self-attention with QK-norm, mRoPE, and a segment mask.

    Computes multi-head self-attention over a packed token sequence where the
    rotary position embedding (``cos`` / ``sin``) and the per-token
    ``segment_ids`` are supplied by the caller. Attention is restricted to be
    block-diagonal in ``segment_ids`` (tokens attend only within their own
    segment).

    **[REUSE]** The ``hidden_size % num_heads`` check and the ``1 / sqrt(head_dim)``
    temperature come from :mod:`~dl_techniques.layers.attention.common`; the QK-norms
    are the repository :class:`~dl_techniques.layers.norms.rms_norm.RMSNorm` and the
    rotary injection is the shared
    :func:`~dl_techniques.layers.embedding.multi_axis_rope.apply_rotary_pos_emb`.
    Nothing here re-implements a primitive that already exists in the package.

    Shapes below use ``B`` = batch, ``L`` = packed sequence length,
    ``H`` = num_heads, ``D`` = head_dim = ``hidden_size / num_heads``.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────────────────────────────┐
        │   Ideogram4Attention — packed self-attention, segment-masked    │
        │                                                                 │
        │  x [B, L, hidden]    segment_ids [B, L]    cos/sin [B, L, D]    │
        │  — segment_ids and the mRoPE tables are CALL ARGUMENTS: this    │
        │  layer owns no rope sub-layer, the transformer passes them in.  │
        │                             ▼                                   │
        │  Dense(3*hidden, no bias) ► reshape [B,L,3,H,D] ► q, k, v       │
        │                             ▼                                   │
        │  RMSNorm(norm_q) on q,  RMSNorm(norm_k) on k   — per head,      │
        │  over D.  v is NOT normalized.                                  │
        │                             ▼                                   │
        │  transpose q, k, v ► [B, H, L, D]                               │
        │                             ▼                                   │
        │  apply_rotary_pos_emb(q, k, cos, sin)   ◄── mRoPE injected      │
        │                             ▼                                   │
        │  S = q~ · k~ᵀ * (1/sqrt(D))                     [B, H, L, L]    │
        │                             ▼                                   │
        │  block-diagonal segment mask, built INLINE (deliberately not    │
        │  the common mask helper):                                       │
        │    M = where(seg_i == seg_j, 0.0, -1e9)       [B, 1, L, L]      │
        │    S = S + M   ► ADDITIVE, so softmax stays a distribution      │
        │                ► the diagonal is always same-segment, so no     │
        │                  row is all -1e9 even when fp16 makes it -inf   │
        │                             ▼                                   │
        │  softmax(S, axis=-1) ► · v ► merge heads [B, L, hidden]         │
        │                             ▼                                   │
        │  Dense(hidden, no bias) ► Output [B, L, hidden]                 │
        │                                                                 │
        │  There is NO attention_mask parameter.  Padding must be given   │
        │  its own segment id or it is attended to as a normal token.     │
        └─────────────────────────────────────────────────────────────────┘


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
        **No padding-mask argument.** Masking is driven entirely by
        ``segment_ids``; there is no ``attention_mask`` parameter. Padding must be
        expressed by giving pad tokens their own segment id, otherwise they are
        attended to as ordinary tokens of whichever segment they were assigned.

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
        # R13/A4: adopts the shared validator. Its message is character-identical
        # to what stood here apart from the trailing full stop, and
        # `test_ideogram4_attention.py` pins no `pytest.raises(match=...)` on it
        # (checked before the swap), so no diagnostic regresses. `dim_name` keeps
        # this layer naming its OWN constructor argument.
        validate_head_divisibility(
            hidden_size, num_heads, dim_name="hidden_size"
        )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        # --- store config ----------------------------------------------
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.eps = float(eps)
        # R13: was `1.0 / (self.head_dim ** 0.5)`. Verified rather than assumed
        # before swapping — `.hex()` compared against the helper across 27 realistic
        # head dims (1..512), 0 mismatches, so this is a bit-identical rename, not a
        # numerics change. (Note the neighbouring form `head_dim ** -0.5` is NOT
        # bit-identical: it differs in the last ULP for 16 of those 27 dims. The two
        # spellings are not interchangeable.) Still a Python float computed in
        # `__init__`, never in `call()`, per `plan_2026-06-14_33b77a7a/D-002`.
        self._inv_sqrt_dim = compute_attention_scale(self.head_dim)

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
