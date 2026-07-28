"""
Exact attention for long sequences via blockwise processing.

This layer implements Ring Attention, a memory-efficient algorithm that
makes it possible to apply transformer attention to sequences of nearly
unlimited length. It overcomes the quadratic memory complexity of standard
attention by computing the attention matrix in smaller, fixed-size blocks,
reducing memory from ``O(N^2)`` to ``O(block_size^2)`` while maintaining
exact mathematical equivalence through online softmax computation.

Architecture:
    Q, K and V come from three separate ``Dense`` projections and are reshaped to
    ``(batch, num_heads, seq_len, head_dim)``. The sequence is then cut into
    ``ceil(seq_len / block_size)`` blocks and a doubly-nested Python loop walks
    every (query block, key/value block) pair, maintaining per-query running
    ``max`` / ``sum`` / output accumulators. Only one ``block_size x block_size``
    score tile exists at a time.

    Two structural properties are load-bearing and must survive any future
    style pass:

    -   The sequence length is read **statically** as a Python ``int``, not via
        ``ops.shape``, because ``range(num_blocks)`` needs a concrete count under
        ``@tf.function``/jit. See the ``plan_2026-06-14_ab855e7e/D-002`` anchor in
        ``_blockwise_attention``.
    -   No ``probability_type`` hook is exposed. The online softmax is tied to the
        exponential normalizer; see the ``[NO PROBABILITY HOOK]`` note on the
        class below.

Foundational Mathematics:
    Ring/online attention is the streaming form of the softmax. For a fixed query
    block, after absorbing key/value block ``j`` the running state ``(m, l, O)``
    satisfies::

        m_new = max(m, max_j S_ij)
        l_new = l * exp(m - m_new) + sum_j exp(S_ij - m_new)
        O_new = O * exp(m - m_new) + exp(S_ij - m_new) V_j

    The common rescaling factor ``exp(m - m_new)`` retroactively re-bases every
    previously accumulated term onto the new maximum, so the final ``O / l`` is
    algebraically **identical** to a single global ``softmax(S) V`` — this is an
    exact reformulation, not an approximation. The max-subtraction is what keeps
    ``exp`` from overflowing, exactly as in the standard stable softmax.

References:
    - Liu, H., et al. (2023). "Ring Attention with Blockwise Transformers for
      Near-Infinite Context."
    - Milakov, M. & Gimelshein, N. (2018). "Online normalizer calculation
      for softmax."
    - Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact
      Attention with IO-Awareness."
"""

# ---------------------------------------------------------------------

import keras
from keras import ops
from typing import Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .common import (
    apply_attention_mask,
    compute_attention_scale,
    mask_dtype,
    validate_head_divisibility,
)
from ..norms.factory import create_normalization_layer

# ---------------------------------------------------------------------

#: Diagnostic for an ``attention_mask`` whose rank this layer cannot dispatch on.
#: Held as a module constant because it is raised from TWO places — once up front
#: in :meth:`RingAttention._blockwise_attention` (fail fast, before any block work)
#: and once as the ``else`` of the in-loop rank dispatch, which is what structurally
#: guarantees ``mask_slice`` can never be read unbound again. Keep the two in sync
#: by keeping them one string.
_UNSUPPORTED_MASK_RANK = (
    "RingAttention supports attention_mask of rank 2 "
    "(batch, seq_len) key-padding, rank 3 (batch, seq_len, seq_len) or rank 4 "
    "(batch, num_heads, seq_len, seq_len); got rank {rank}."
)


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class RingAttention(keras.layers.Layer):
    """Ring Attention with blockwise processing for extremely long sequences.

    Implements the Ring Attention algorithm that partitions the sequence into
    fixed-size blocks and computes attention incrementally using online softmax.
    For each query block ``Q_i``, the algorithm iterates through all key-value
    blocks ``(K_j, V_j)``, computing partial scores
    ``S_ij = Q_i K_j^T / sqrt(d_k)`` and maintaining running statistics:
    ``m_new = max(m_prev, max(S_ij))``,
    ``O_new = O_prev * exp(m_prev - m_new) + exp(S_ij - m_new) V_j``,
    ``l_new = l_prev * exp(m_prev - m_new) + sum(exp(S_ij - m_new))``.
    The final output is ``O / l``, which is mathematically identical to standard
    attention without ever materializing the full ``N x N`` attention matrix.

    **[NO PROBABILITY HOOK — intentional constraint, not an omission]** Unlike most
    siblings in this package, this layer exposes no ``probability_type`` /
    ``probability_config`` pair and does not use the shared
    :class:`~dl_techniques.layers.activations.ProbabilityOutput`. That is a
    mathematical constraint, not an oversight: the blockwise loop never
    materializes a full score row, so it can only use a normalizer that admits a
    *streaming* form. The exponential normalizer does — its running ``(max, sum)``
    recurrence is exactly what makes this layer exact — whereas sparsemax,
    threshmax and adaptive-softmax need the whole row at once and have no
    equivalent online update. Adding the hook would either break exactness or
    silently force full materialization, defeating the layer's entire purpose. The
    same statement is repeated in ``_blockwise_attention`` next to the loop it
    constrains; keep both.

    **[STATIC SEQUENCE LENGTH]** ``call()`` requires a statically-known sequence
    dimension and raises ``ValueError`` when it is ``None``. The block count feeds
    a Python ``range()``, which cannot consume a traced tensor. The batch dimension
    stays fully dynamic.

    **[REUSE]** The ``dim % num_heads`` check and the ``1 / sqrt(head_dim)``
    temperature come from :mod:`~dl_techniques.layers.attention.common`; the
    optional Q/K norms come from
    :func:`~dl_techniques.layers.norms.factory.create_normalization_layer`.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────────────────────┐
        │   RingAttention — blockwise online-softmax attention    │
        │                                                         │
        │   Flash/Ring-style blockwise attention: the (N x N)     │
        │   score matrix is NEVER materialized. Needs a static    │
        │   seq_len (the block count is a Python int).            │
        │                                                         │
        │                   Input  [B, S, dim]                    │
        │                            ▼                            │
        │        W_q / W_k / W_v  ►  heads  [B, H, S, d_h]        │
        │                            ▼                            │
        │    optional q_norm / k_norm  (once, before blocking)    │
        │                            ▼                            │
        │                      q = q * scale                      │
        │                            ▼                            │
        │   mask (if given): validate rank 2/3/4, then rescue a   │
        │   row that keeps NOTHING — once, OVER THE FULL KEY      │
        │   AXIS, before the loop. Never per tile: a per-tile     │
        │   rescue un-masks the future under a causal mask.       │
        │                            ▼                            │
        │  ┌── for q_block in range(num_blocks) ───────────────┐  │
        │  │  m = -inf,   l = 0,   O = 0                       │  │
        │  │  for kv_block in range(num_blocks):               │  │
        │  │    S  = q_blk @ k_blkᵀ            (one tile only) │  │
        │  │    S  = S + mask bias  (rescue_axis=None)         │  │
        │  │    m' = max(m, rowmax(S))                         │  │
        │  │    O  = O*exp(m-m') + exp(S-m') @ v_blk           │  │
        │  │    l  = l*exp(m-m') + rowsum(exp(S-m'))           │  │
        │  │    m  = m'                                        │  │
        │  │  block_out = O / l                                │  │
        │  └───────────────────────────────────────────────────┘  │
        │                            ▼                            │
        │   concatenate the block outputs on the sequence axis    │
        │   (Python list + ops.concatenate, NOT slice_update)     │
        │                            ▼                            │
        │       merge heads  ►  W_o  ►  Output  [B, S, dim]       │
        │                                                         │
        │   return_attention_weights=True gives (output, None):   │
        │   the weights are never materialized, so there is none. │
        └─────────────────────────────────────────────────────────┘

    :param dim: Input/output dimension (embedding size). Must be positive and
        divisible by num_heads.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param block_size: Sequence block size for processing. Larger blocks use
        more memory but may be more efficient.
    :type block_size: int
    :param dropout_rate: Dropout rate for attention weights, between 0.0 and 1.0.
    :type dropout_rate: float
    :param use_bias: Whether to use bias in linear projections.
    :type use_bias: bool
    :param kernel_initializer: Initializer for kernel weights.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias weights.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param qk_norm_type: Optional normalization type applied per-head to Q and K
        once, before the blockwise loop, forwarded to
        :func:`~dl_techniques.layers.norms.factory.create_normalization_layer`.
        ``None`` disables QK-norm. Defaults to ``None``.
    :type qk_norm_type: Optional[str]
    :param qk_norm_kwargs: Optional keyword arguments forwarded to
        :func:`~dl_techniques.layers.norms.factory.create_normalization_layer` for
        both the Q and K norms. Defaults to ``None``.
    :type qk_norm_kwargs: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the Layer parent class.
    :type kwargs: Any

    :raises ValueError: If dim is not positive or not divisible by num_heads.
    :raises ValueError: If num_heads or block_size are not positive.
    :raises ValueError: If dropout_rate is not between 0.0 and 1.0.
    :raises ValueError: From ``call()``, if the input's sequence dimension is not
        statically known (see the ``[STATIC SEQUENCE LENGTH]`` note above).
    :raises ValueError: From ``call()``, if ``attention_mask`` has a rank other
        than 2, 3 or 4.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            block_size: int = 512,
            dropout_rate: float = 0.0,
            use_bias: bool = False,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            qk_norm_type: Optional[str] = None,
            qk_norm_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        self._validate_inputs(dim, num_heads, block_size, dropout_rate)

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs

        # Derived parameters
        self.head_dim = self.dim // self.num_heads
        # R13: was `1.0 / math.sqrt(self.head_dim)`, which is exactly the shared
        # helper's body (`math.sqrt(int)` and `math.sqrt(float(int))` are the same
        # double). Verified rather than assumed: `.hex()` compared across 27
        # realistic head dims (1..512), all equal. Still a Python float computed in
        # `__init__` and never in `call()`, per `plan_2026-06-14_33b77a7a/D-002`.
        self.scale = compute_attention_scale(self.head_dim)

        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.w_q = keras.layers.Dense(
            self.num_heads * self.head_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_q'
        )

        self.w_k = keras.layers.Dense(
            self.num_heads * self.head_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_k'
        )

        self.w_v = keras.layers.Dense(
            self.num_heads * self.head_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_v'
        )

        self.w_o = keras.layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_o'
        )

        # Attention dropout layer
        self.dropout = keras.layers.Dropout(self.dropout_rate, name='attention_dropout')

        # Optional QK-normalization sub-layers
        if self.qk_norm_type is not None:
            self.q_norm = create_normalization_layer(
                self.qk_norm_type, name="q_norm", **(self.qk_norm_kwargs or {})
            )
            self.k_norm = create_normalization_layer(
                self.qk_norm_type, name="k_norm", **(self.qk_norm_kwargs or {})
            )
        else:
            self.q_norm = None
            self.k_norm = None

        logger.info(f"RingAttention initialized: dim={dim}, "
                    f"num_heads={num_heads}, block_size={block_size}")

    def _validate_inputs(
            self,
            dim: int,
            num_heads: int,
            block_size: int,
            dropout_rate: float
    ) -> None:
        """Validate initialization parameters.

        :param dim: Model dimension to validate.
        :type dim: int
        :param num_heads: Number of attention heads to validate.
        :type num_heads: int
        :param block_size: Block size for sequence processing to validate.
        :type block_size: int
        :param dropout_rate: Dropout rate to validate.
        :type dropout_rate: float

        :raises ValueError: If any parameter is invalid.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        # R13: adopts the shared validator. Its message is character-for-character
        # what stood here, so the regex pinned at test_ring_attention.py:82 still
        # matches and the diagnostic is unchanged.
        validate_head_divisibility(dim, num_heads)
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all sub-layers for robust serialization.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        # Build sub-layers in computational order
        self.w_q.build(input_shape)
        self.w_k.build(input_shape)
        self.w_v.build(input_shape)

        # Output projection uses the same input shape as it processes the
        # reshaped attention output which has the same dim dimension
        self.w_o.build(input_shape)

        # Dropout doesn't need explicit building as it has no weights
        self.dropout.build(input_shape)

        # Build QK-norm sub-layers on the per-head Q/K shape:
        # (batch, num_heads, seq_len, head_dim).
        if self.q_norm is not None:
            qk_shape = (input_shape[0], self.num_heads, input_shape[1], self.head_dim)
            self.q_norm.build(qk_shape)
            self.k_norm.build(qk_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
            attention_mask: Optional[keras.KerasTensor] = None,
            return_attention_weights: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, None]]:
        """Apply ring attention with blockwise processing.

        :param inputs: Input tensor of shape ``(batch_size, seq_len, dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode. Affects dropout behavior.
        :type training: Optional[bool]
        :param attention_mask: Optional attention mask, ``1 = keep``. Rank 2
            ``(batch_size, seq_len)`` is a key-padding mask shared by every query
            row; rank 3 ``(batch_size, seq_len, seq_len)`` and rank 4
            ``(batch_size, num_heads, seq_len, seq_len)`` are per-query-position.
            Any other rank raises ``ValueError``.
        :type attention_mask: Optional[keras.KerasTensor]
        :param return_attention_weights: Whether to return attention weights.
            Always returns ``None`` due to blockwise processing.
        :type return_attention_weights: bool

        :return: Output tensor of shape ``(batch_size, seq_len, dim)``.
            If return_attention_weights is ``True``, returns ``(output, None)``.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, None]]
        """
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # Project to Q, K, V
        q = self.w_q(inputs)  # (batch, seq_len, num_heads * head_dim)
        k = self.w_k(inputs)  # (batch, seq_len, num_heads * head_dim)
        v = self.w_v(inputs)  # (batch, seq_len, num_heads * head_dim)

        # Reshape for multi-head attention
        q = ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ops.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = ops.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))

        # Transpose to (batch, num_heads, seq_len, head_dim) for attention computation
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        # Optional QK-normalization (applied once before the blockwise loop;
        # subsequent block slices reuse the normalized Q/K).
        if self.q_norm is not None:
            q = self.q_norm(q, training=training)
            k = self.k_norm(k, training=training)

        # Apply scaling
        q = q * self.scale

        # Compute blockwise attention
        attention_output = self._blockwise_attention(
            q, k, v, attention_mask=attention_mask, training=training
        )

        # Transpose back and reshape to original format
        attention_output = ops.transpose(attention_output, (0, 2, 1, 3))  # (batch, seq_len, num_heads, head_dim)
        attention_output = ops.reshape(attention_output, (batch_size, seq_len, self.dim))

        # Final output projection
        output = self.w_o(attention_output, training=training)

        if return_attention_weights:
            # Return None for attention weights as they're not materialized in Ring Attention
            return output, None
        return output

    def _blockwise_attention(
            self,
            queries: keras.KerasTensor,
            keys: keras.KerasTensor,
            values: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Compute blockwise attention with online softmax.

        :param queries: Query tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        :type queries: keras.KerasTensor
        :param keys: Key tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        :type keys: keras.KerasTensor
        :param values: Value tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        :type values: keras.KerasTensor
        :param attention_mask: Optional mask for attention computation.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]

        :return: Attention output of shape ``(batch, num_heads, seq_len, head_dim)``.
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(queries)[0]
        num_heads = self.num_heads
        # DECISION plan_2026-06-14_ab855e7e/D-002: the block-wise loop count and
        # `range(num_blocks)` below require a Python int. A dynamic
        # ops.shape(queries)[2] makes range() crash under @tf.function/jit (the
        # static-shape defect class already fixed in capsule/PFA). Read the seq
        # dim statically and fail loud on None; batch stays dynamic. Do NOT revert
        # to ops.shape for the sequence dim here.
        seq_len = queries.shape[2]
        if seq_len is None:
            raise ValueError(
                "RingAttention requires a statically-known sequence length "
                "(the block-wise loop needs a Python-int block count); got None. "
                "Provide inputs with a fixed sequence dimension."
            )
        head_dim = self.head_dim

        # Calculate number of blocks
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        # NOTE: A `probability_type` customization hook is intentionally NOT
        # exposed on this layer. The online/blockwise softmax below maintains
        # running max/sum statistics that are mathematically tied to the
        # exponential normalizer; alternatives like sparsemax or threshmax do
        # not admit an equivalent streaming form and would break exactness.
        #
        # This is a CONSTRAINT, not a missing feature. Do NOT "restore parity"
        # with the rest of the package by wiring in `ProbabilityOutput` here: any
        # non-exponential normalizer needs the whole score row at once, which
        # would force full `N x N` materialization and destroy the only reason
        # this layer exists. The same statement is on the class docstring under
        # `[NO PROBABILITY HOOK]`; keep both in sync.

        # Accumulate per-block outputs in a Python list, concatenated after the
        # loop. NOTE: do NOT reassemble via ops.slice_update — on the TF backend
        # it lowers to XlaDynamicUpdateSlice, which has no registered eager
        # gradient (LookupError on backprop). list + ops.concatenate is fully
        # differentiable and forward-identical (blocks appended in q_block_idx
        # order map 1:1 to the original q_start offsets along axis=2).
        block_outputs = []

        # DECISION plan-2026-07-27T183600-b4ef45f0/D-011
        # TWO things happen here that do NOT happen at the other nine mask sites,
        # and both are forced by the ONLINE (blockwise) softmax below.
        #
        # (1) THE ACCUMULATION RUNS IN `mask_dtype(...)` WHENEVER A MASK IS
        #     SUPPLIED. `running_max` starts at `-inf`, so a `(q_block, kv_block)`
        #     tile whose rows are ALL masked gives `block_max = -inf`, and the very
        #     next line evaluates `running_max - new_max` = `-inf - -inf` = **NaN**,
        #     which then multiplies into `running_sum` and `accumulated_output` and
        #     destroys the row. With a FINITE `MASK_BIAS_VALUE` the same algebra is
        #     exactly right (the tile contributes `exp(-1e9 - new_max) == 0`, and a
        #     later tile's larger max renormalizes any transient to zero) — which is
        #     why float32 has always been correct here and fp16 has always been
        #     broken. An entirely-masked tile is NOT exotic: a left-padded batch
        #     blanks the leading kv blocks outright, and a causal mask blanks every
        #     strictly-upper tile.
        #     WHAT NOT TO DO: do NOT pass `out_dtype=<compute dtype>` here "for
        #     consistency with the other nine sites". Those sites hand their biased
        #     logits straight to a softmax, which tolerates `-inf`; this one feeds a
        #     running max/sum recurrence, which does not. Do NOT instead special-case
        #     `-inf` with `ops.where(...)` after the fact: the unselected branch still
        #     contributes `0 * NaN` in the BACKWARD pass (the same trap D-003/D-006
        #     rejected). Do NOT seed `running_max` with a per-dtype finite floor
        #     either — `common.py`'s D-002 anchor rules out per-dtype magic constants.
        #     ACCEPTED COST: under `mixed_float16` a MASKED forward accumulates in
        #     float32, so it costs more memory/bandwidth than the unmasked one. That
        #     is what every production flash-attention kernel does anyway. The
        #     no-mask path keeps the compute dtype and traces the same graph as
        #     before (every `ops.cast` below is an identity in that case).
        #
        # (2) THE DEGENERATE-ROW RESCUE IS DONE ONCE, HERE, OVER THE FULL KEY AXIS —
        #     and the per-block call below therefore passes `rescue_axis=None`.
        #     `apply_attention_mask`'s default `rescue_axis=-1` means "a row that
        #     keeps nothing keeps everything", but inside this loop a "row" is one
        #     TILE of the key axis, not the whole key axis. Applied per tile it would
        #     read every entirely-masked tile as degenerate and UN-MASK it — under a
        #     causal mask that lets a query attend to the FUTURE, silently, with a
        #     perfectly finite output. MEASURED with that injection: a future token
        #     moved the earliest query rows by 24.14 in float32, mixed_float16 AND
        #     float64 (0.0 for the code as shipped) while every finiteness test in
        #     the module still passed. Pinned by
        #     `TestRingAttentionCausalitySurvivesTheRescue`.
        #     WHAT NOT TO DO: do NOT "simplify" this by deleting the pre-loop rescue
        #     and letting the helper's default do the work per block. Do NOT move the
        #     rescue after the loop either: by then the fp16 NaN has already been
        #     formed in `running_max`.
        # See decisions.md D-011 (plan-2026-07-27T183600-b4ef45f0).
        state_dtype = keras.backend.standardize_dtype(queries.dtype)
        if attention_mask is not None:
            # DECISION plan-2026-07-27T183600-b4ef45f0/D-013
            # The rank is validated ONCE, HERE, before any block work and before
            # the rescue below — not only inside the loop. A rank this layer
            # cannot dispatch on used to fall through both branches of the in-loop
            # `if/elif` and surface as `UnboundLocalError: cannot access local
            # variable 'mask_slice'` on the FIRST (q_block 0, kv_block 0)
            # iteration. Validating up front means the caller gets a named
            # `ValueError` naming the ranks that ARE supported, before a single
            # matmul runs.
            # WHAT NOT TO DO: do NOT delete the `else: raise` at the in-loop
            # dispatch on the grounds that this check makes it unreachable. That
            # `else` is the STRUCTURAL guarantee that `mask_slice` cannot be read
            # unbound; a future rank added here but not there would re-create
            # exactly the defect this fixes, silently and only on the masked path.
            # See decisions.md D-013 (plan-2026-07-27T183600-b4ef45f0).
            mask_rank = len(ops.shape(attention_mask))
            if mask_rank not in (2, 3, 4):
                raise ValueError(_UNSUPPORTED_MASK_RANK.format(rank=mask_rank))

            state_dtype = mask_dtype(state_dtype)
            # THIS SITE'S MASK POLARITY, passed through verbatim: `1 = keep`. The
            # rescue is spelled on the PREDICATE, exactly as `apply_attention_mask`
            # spells it internally, so that no all-`MASK_BIAS_VALUE` row is ever
            # FORMED (and no NaN gradient with it). It is duplicated here rather
            # than extracted because a rescue-only helper in `common.py` is a named
            # STOP tripwire for this plan, and the parameter cannot express
            # "reduce over an axis the caller has split into tiles".
            # This runs AFTER the rank check above and BEFORE the loop, so every
            # supported rank gets the IDENTICAL rescue: `axis=-1` is the key axis
            # for rank 2 `(batch, seq_len)`, rank 3 `(batch, seq_q, seq_k)` and
            # rank 4 `(batch, heads, seq_q, seq_k)` alike. For rank 2 the reduction
            # is per batch item rather than per query row, which is the same thing
            # — every query row shares one key mask.
            _keep = ops.cast(attention_mask, state_dtype) > 0.0
            _keep = ops.logical_or(
                _keep,
                ops.logical_not(ops.any(_keep, axis=-1, keepdims=True)),
            )
            attention_mask = ops.cast(_keep, state_dtype)

        # Process each query block
        for q_block_idx in range(num_blocks):
            # Get query block bounds
            q_start = q_block_idx * self.block_size
            q_end = min(q_start + self.block_size, seq_len)  # Python int (static seq_len)
            q_block = queries[:, :, q_start:q_end, :]  # (batch, num_heads, q_block_size, head_dim)

            q_block_size = q_end - q_start

            # Initialize running statistics for online softmax
            # (batch, num_heads, q_block_size)
            # `state_dtype` is the compute dtype when no mask is supplied and
            # `mask_dtype(...)` when one is — see the D-011 anchor above.
            running_max = ops.full(
                (batch_size, num_heads, q_block_size),
                -float('inf'),
                dtype=state_dtype
            )
            running_sum = ops.zeros(
                (batch_size, num_heads, q_block_size),
                dtype=state_dtype
            )
            # (batch, num_heads, q_block_size, head_dim)
            accumulated_output = ops.cast(ops.zeros_like(q_block), state_dtype)

            # Process each key/value block for this query block
            for kv_block_idx in range(num_blocks):
                # Get key/value block bounds
                kv_start = kv_block_idx * self.block_size
                kv_end = min(kv_start + self.block_size, seq_len)  # Python int (static seq_len)
                k_block = keys[:, :, kv_start:kv_end, :]  # (batch, num_heads, kv_block_size, head_dim)
                v_block = values[:, :, kv_start:kv_end, :]

                # Compute attention scores for this block pair
                # (batch, num_heads, q_block_size, head_dim) @ (batch, num_heads, head_dim, kv_block_size)
                # -> (batch, num_heads, q_block_size, kv_block_size)
                scores = ops.matmul(q_block, ops.transpose(k_block, (0, 1, 3, 2)))

                # Apply attention mask if provided
                #
                # DEFECT #1 — FIXED (plan-2026-07-27T183600-b4ef45f0, step 7).
                # The dispatch below used to handle rank 3 and rank 4 only, with no
                # `else`. A rank-2 `(batch, seq_len)` key-padding mask — the shape
                # most Keras callers produce, and the shape `rpc_attention.py`
                # explicitly supports — fell through BOTH branches, leaving
                # `mask_slice` unbound; MEASURED, that raised `UnboundLocalError:
                # cannot access local variable 'mask_slice'` on the FIRST
                # (q_block 0, kv_block 0) iteration. It now has its own branch, and
                # the `else` below makes an unbound read impossible by construction.
                # See the D-013 anchor above the pre-loop rank check.
                #
                # DEFECT #2 — FIXED (plan-2026-07-27T183600-b4ef45f0, step 5). The
                # bias used to be the arithmetic form `(1.0 - mask_slice) * -1e9`
                # evaluated in `scores.dtype`; under `mixed_float16` that made
                # `-1e9` into `-inf` and every UNMASKED position `0 * -inf = NaN`.
                # MEASURED at (B=2, N=64, dim=64, num_heads=4, block_size=16):
                # 8192/8192 NaN for an all-ones mask, right padding, left padding
                # AND a causal mask, with float32 and float64 at 0/8192 and an fp16
                # no-mask forward also clean. It now delegates to
                # `common.apply_attention_mask`; see the D-011 anchor above the loop
                # for why this site's `out_dtype` and `rescue_axis` differ from every
                # other adoption in the package.
                if attention_mask is not None:
                    # Extract relevant portion of mask for these blocks
                    if len(ops.shape(attention_mask)) == 2:
                        # (batch, seq_len) key-padding -> slice the KEY axis only
                        # and expand to (batch, 1, 1, kv_block_size). The two size-1
                        # axes broadcast over heads and over the query block inside
                        # `apply_attention_mask`, which is what keeps this layer's
                        # mask memory O(N) instead of the O(N^2) a caller would pay
                        # by pre-expanding to rank 3 — the whole reason RingAttention
                        # exists. Numerically it is EXACTLY the rank-3/rank-4 result
                        # (pinned by `TestRingAttentionRank2MaskDispatch`).
                        mask_slice = attention_mask[:, kv_start:kv_end]
                        mask_slice = ops.expand_dims(mask_slice, axis=1)
                        mask_slice = ops.expand_dims(mask_slice, axis=1)
                    elif len(ops.shape(attention_mask)) == 3:
                        # (batch, seq_len, seq_len) -> extract block region
                        mask_slice = attention_mask[:, q_start:q_end, kv_start:kv_end]
                        # Expand for heads: (batch, 1, q_block_size, kv_block_size)
                        mask_slice = ops.expand_dims(mask_slice, axis=1)
                        mask_slice = ops.repeat(mask_slice, num_heads, axis=1)
                    elif len(ops.shape(attention_mask)) == 4:
                        # (batch, num_heads, seq_len, seq_len) -> extract block region
                        mask_slice = attention_mask[:, :, q_start:q_end, kv_start:kv_end]
                    else:
                        # Unreachable via `call()` — the pre-loop check above already
                        # rejected every other rank. Kept anyway: this `else` is what
                        # makes an unbound `mask_slice` read structurally impossible,
                        # which is the actual defect being fixed. Do NOT delete it.
                        raise ValueError(
                            _UNSUPPORTED_MASK_RANK.format(
                                rank=len(ops.shape(attention_mask))
                            )
                        )

                    # Convert mask to additive form: 0 -> MASK_BIAS_VALUE, 1 -> 0.
                    # `mask_slice` is this site's `1 = keep` predicate, passed
                    # through verbatim (no `> 0` comparison invented, no inversion —
                    # the helper performs no polarity inference by design).
                    mask_slice = ops.cast(mask_slice, scores.dtype)
                    scores = apply_attention_mask(
                        scores,
                        mask_slice,
                        out_dtype=state_dtype,
                        rescue_axis=None,   # D-011: rescued once, before the loop
                    )

                # Compute new maximum for safe softmax
                # (batch, num_heads, q_block_size)
                block_max = ops.max(scores, axis=-1)
                new_max = ops.maximum(running_max, block_max)

                # Renormalize previous results
                max_diff = running_max - new_max
                renorm_factor = ops.exp(max_diff)

                # Update running statistics
                running_sum = running_sum * renorm_factor
                accumulated_output = accumulated_output * ops.expand_dims(renorm_factor, axis=-1)

                # Compute new contributions
                # (batch, num_heads, q_block_size, kv_block_size)
                new_scores = ops.exp(scores - ops.expand_dims(new_max, axis=-1))

                # Apply dropout to attention weights. The cast back to `state_dtype`
                # is required, not cosmetic: `self.dropout` is a Keras layer with
                # autocasting ON, so under `mixed_float16` it returns fp16 even when
                # handed the float32 accumulation, and the matmul below would then
                # see two different dtypes. It is an identity on the no-mask path.
                if training and self.dropout_rate > 0:
                    new_scores = ops.cast(
                        self.dropout(new_scores, training=training), state_dtype
                    )

                # Accumulate new attention output
                # (batch, num_heads, q_block_size, kv_block_size) @ (batch, num_heads, kv_block_size, head_dim)
                # -> (batch, num_heads, q_block_size, head_dim)
                # `ops.cast(..., state_dtype)` is an identity on the no-mask path and
                # promotes the values to the float32 accumulation on the masked one.
                new_output = ops.matmul(new_scores, ops.cast(v_block, state_dtype))
                accumulated_output = accumulated_output + new_output

                # Update running sum
                running_sum = running_sum + ops.sum(new_scores, axis=-1)
                running_max = new_max

            # Normalize final output for this query block
            # (batch, num_heads, q_block_size, head_dim)
            running_sum_expanded = ops.expand_dims(running_sum, axis=-1)
            block_output = accumulated_output / running_sum_expanded

            # Collect this block's output; concatenated in loop order below.
            block_outputs.append(block_output)

        # Reassemble along the sequence axis. Loop order (q_block_idx 0,1,2,...)
        # places each block at its original q_start offset, so concatenation is
        # numerically identical to the prior slice_update assembly, and the
        # last (possibly partial) block carries its own size via the q-slice.
        outputs = ops.concatenate(block_outputs, axis=2)
        # Back to the layer's compute dtype. An identity unless a mask promoted the
        # accumulation to `mask_dtype(...)` — see the D-011 anchor above.
        return ops.cast(outputs, keras.backend.standardize_dtype(queries.dtype))

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: Output shape tuple, same as input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Dictionary containing all parameters required to recreate this layer.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'block_size': self.block_size,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'qk_norm_type': self.qk_norm_type,
            'qk_norm_kwargs': self.qk_norm_kwargs,
        })
        return config

# ---------------------------------------------------------------------
