"""
Ideogram4 self-attention: fused QKV, per-head RMS QK-norm, mRoPE, segment mask.

This is the Keras 3 port of Ideogram4's ``Ideogram4Attention``. It is multi-head
self-attention specialized for the Ideogram4 packed-sequence DiT, where one
tensor holds several independent sequences end to end and a per-token segment id
says which is which.

Five parts:

- **Fused QKV.** One bias-free ``Dense(3 * hidden_size)``, reshaped to
  ``(B, L, 3, num_heads, head_dim)`` and sliced into q, k and v.
- **Per-head RMS QK-norm.** q and k are RMS-normalized over the ``head_dim``
  axis by two independent :class:`RMSNorm` sub-layers with scale shape
  ``(head_dim,)``. v is not normalized. This reuses the repository ``RMSNorm``.
- **mRoPE injection.** Precomputed ``cos`` and ``sin`` tables of shape
  ``(B, L, head_dim)`` are applied to q and k through the shared
  :func:`apply_rotary_pos_emb`. This layer does NOT own or build the mRoPE
  layer - the transformer computes the tables once and passes them in.
- **Block-diagonal segment mask.** Tokens attend only to tokens sharing their
  ``segment_ids`` value. The mask is ADDITIVE: 0.0 for same-segment, a large
  negative value for cross-segment, added to the pre-softmax scores. The
  ``DECISION`` anchor in :meth:`call` says why, and why a boolean keep-mask
  fed to a fused SDPA op is not the shape to use here.
- **SDPA by hand.** Scores, scale by ``1/sqrt(head_dim)``, add the mask,
  softmax, multiply by v, then a bias-free ``Dense(hidden_size)`` output
  projection. All in ``keras.ops``; there is no fused SDPA op to call.

Architecture:
    Two structural properties matter to callers:

    -   The layer does not own the mRoPE layer. ``cos`` and ``sin`` are call
        arguments, so one rope table is computed per forward pass and shared by
        every block.
    -   Masking is additive, in the logit domain, not a boolean keep-mask handed
        to a fused op. See the ``DECISION`` anchor in :meth:`call`.

Foundational Mathematics:
    For head ``h`` with per-head dimension ``D = hidden_size / H``::

        q, k, v   = split( W_qkv x ,  3 )              # fused projection
        q_hat     = RMSNorm(q) ,   k_hat = RMSNorm(k)  # per-head, over D
        q~, k~    = mRoPE(q_hat, k_hat ; cos, sin)     # rotary injection
        S_ij      = (q~_i . k~_j) / sqrt(D)  +  M_ij
        M_ij      = 0                if segment_id_i == segment_id_j
                    -1e9             otherwise
        out_i     = sum_j softmax_j(S_ij) v_j

    RMS QK-norm rescales each query and key to a fixed L2 radius before the dot
    product. That bounds ``|S_ij|`` independently of the activation scale, which
    is what keeps the logits stable without a learned temperature.

    The mask is added in the LOGIT domain rather than multiplied in the
    probability domain, so softmax stays a proper distribution over each token's
    own segment.

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
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.embedding.multi_axis_rope import apply_rotary_pos_emb

from .common import (
    compute_attention_scale,
    validate_head_divisibility
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

# Large-negative fill for masked-out (cross-segment) attention logits.
#
# Same magnitude as `common.MASK_BIAS_VALUE`, and not replaced by it. The shared
# constant carries a contract this layer does not follow: materialize the bias in
# `common.mask_dtype(...)`, never in the layer's compute dtype. This layer casts
# straight to `scores.dtype`. Importing the name without porting the dtype
# discipline would advertise a safety guarantee the code does not implement.
#
# This layer is nonetheless not exposed to the fp16 mask-NaN hazard catalogued
# elsewhere in this package, for a structural reason. `call()` uses the
# `ops.where(same_segment, 0.0, -1e9)` form, not the arithmetic
# `scores + (1 - mask) * -1e9` form. Under `mixed_float16` the cast does make
# this constant `-inf`, but `where` never multiplies it by zero, so
# `0 * -inf = NaN` cannot arise.
#
# Softmax stays well-defined because the DIAGONAL is always same-segment, so
# every row keeps at least one finite 0.0 entry and the max-subtraction inside
# softmax stays finite. Measured in fp16 rather than argued on paper: scores
# `[0, 1, 2, 3]` plus the bias `[0, 0, -inf, -inf]` gives softmax
# `[0.269, 0.731, 0, 0]`. Don't rewrite this into the arithmetic form.
_MASK_NEG = -1e9


@register_dl_technique("dl_techniques.layers.attention.ideogram4_attention")
class Ideogram4Attention(keras.layers.Layer):
    """Ideogram4 packed self-attention with QK-norm, mRoPE, and a segment mask.

    Computes multi-head self-attention over a packed token sequence where the
    rotary position embedding (``cos`` / ``sin``) and the per-token
    ``segment_ids`` are supplied by the caller. Attention is restricted to be
    block-diagonal in ``segment_ids`` (tokens attend only within their own
    segment).

    **[REUSE]** The ``hidden_size % num_heads`` check and the
    ``1 / sqrt(head_dim)`` temperature come from
    :mod:`~dl_techniques.layers.attention.common`. The QK-norms are the
    repository :class:`~dl_techniques.layers.norms.rms_norm.RMSNorm`, and the
    rotary injection is the shared
    :func:`~dl_techniques.layers.embedding.multi_axis_rope.apply_rotary_pos_emb`.
    Nothing here re-implements a primitive the package already has.

    Shapes below use ``B`` = batch, ``L`` = packed sequence length,
    ``H`` = num_heads, ``D`` = head_dim = ``hidden_size / num_heads``.

    **Architecture Overview:**

    .. code-block:: text

              x [B, L, hidden]     segment_ids [B, L]
                     │              cos, sin [B, L, D]
                     │                     │
                     ▼                     │
          ┌────────────────────────┐       │  segment_ids and the mRoPE
          │ qkv                    │       │  tables are CALL ARGUMENTS.
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
          │ over D. v is NOT       │       │
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
          │ ADDITIVE, so softmax stays a proper     │
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

        There is NO attention_mask parameter. Padding must be given its
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

        # --- validation -------------------------------------------------
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be a positive integer, got {hidden_size}"
            )
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(
                f"num_heads must be a positive integer, got {num_heads}"
            )
        # Adopts the shared validator. Its message is character-identical to what
        # stood here apart from the trailing full stop, and
        # `test_ideogram4_attention.py::test_ctor_raises_on_indivisible` uses a
        # bare `pytest.raises(ValueError)` with no `match=`, so no diagnostic
        # regressed. `dim_name` keeps the message naming this layer's OWN
        # constructor argument.
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
        # This was `1.0 / (self.head_dim ** 0.5)`. The swap was measured, not
        # assumed: `.hex()` against the helper over 27 realistic head dims
        # (1 to 512) gave 0 mismatches, so it is a bit-identical rename rather
        # than a numerics change.
        #
        # Don't reach for the neighbouring spelling `head_dim ** -0.5`. It is NOT
        # bit-identical - it differs in the last ULP for 16 of those 27 dims.
        #
        # Keep the scale a Python float computed here in `__init__`. Don't move
        # it into `call()` and don't build it with `keras.ops.sqrt`: a backend
        # tensor made in `__init__` can leak out of a symbolic scratch graph.
        self._inv_sqrt_dim = compute_attention_scale(self.head_dim)

        # --- sub-layers (created here, built in build()) ---------------
        self.qkv = keras.layers.Dense(
            3 * hidden_size, use_bias=False, name="qkv"
        )
        # Per-head RMS norm over head_dim. Each carries a learnable scale of
        # shape (head_dim,), which is why build() gives them a 4D shape.
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

        # DECISION plan_2026-06-12_59a18a10/D-004
        # The originating plan directory is gone, so this comment is the record.
        # Build an ADDITIVE block-diagonal segment mask, 0.0 same-segment and
        # -1e9 cross-segment, and add it to the pre-softmax scores. The PyTorch
        # source hands a BOOLEAN keep-mask to F.scaled_dot_product_attention;
        # that is not portable here, because SDPA is written out by hand in
        # keras.ops and there is no backend-agnostic boolean-mask SDPA op, so
        # the mask has to be folded into the logits numerically.
        #
        # Don't replace this with `keras.ops.where(keep, scores, -inf)` over a
        # boolean keep-mask and raw -inf. What keeps every row well-defined is
        # the DIAGONAL, not the magnitude: `same_segment` is true at i == j for
        # every token, so no row is ever entirely suppressed and softmax never
        # sees 0/0. Do NOT read -1e9 as "the finite alternative to -inf" --
        # `np.float16(-1e9)` IS -inf, and the cast below is to `scores.dtype`,
        # the COMPUTE dtype, so under mixed_float16 this bias is exactly -inf.
        # common.py's rule (bias only inside a `mask_dtype(...)` chain, never in
        # the compute dtype) is not followed here; the diagonal is what makes
        # that survivable.
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

# ---------------------------------------------------------------------
