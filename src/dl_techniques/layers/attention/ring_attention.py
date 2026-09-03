"""
Exact attention for long sequences, computed blockwise with an online softmax.

Standard attention materializes an ``N x N`` score matrix, so memory grows
with the square of the sequence length. Softmax admits a streaming form, so
this layer walks the score matrix in fixed-size tiles, keeping a running
``(m, l, O)`` state per query row -- the largest score seen so far, the
accumulated normalizer, and the accumulated weighted values -- instead::

    m_new = max(m, max_j S_ij)
    l_new = l * exp(m - m_new) + sum_j exp(S_ij - m_new)
    O_new = O * exp(m - m_new) + exp(S_ij - m_new) V_j

``O / l`` at the end is an exact reformulation of a single global
``softmax(S) V``, not an approximation. Peak memory then depends on
``block_size``, not on ``N``, but only for the forward pass: this layer does
no gradient checkpointing, so a backward pass still retains every block's
intermediates (measured: 12 MB at N=256 rising to 547 MB at N=2048, against
a flat ~1-2 MB at inference).

Because the recurrence needs the exponential normalizer's algebra, this
layer exposes no ``probability_type`` hook; sparsemax and similar strategies
need a whole score row at once and have no streaming form. The sequence
length must be a static Python ``int``, since the block loop is
``range(num_blocks)``. Masking differs from other layers in this package in
two ways: the accumulation runs in a wider dtype whenever a mask is given,
and the degenerate-row rescue applies once over the full key axis before the
loop, not per tile -- a per-tile rescue would un-mask the future under a
causal mask. ``return_attention_weights=True`` yields ``(output, None)``:
there is no ``N x N`` matrix to hand back.

References:
    - Liu et al., 2023. Ring Attention with Blockwise Transformers for
      Near-Infinite Context. (https://arxiv.org/abs/2310.01889)
    - Milakov and Gimelshein, 2018. Online normalizer calculation for softmax.
      (the ``(m, l)`` recurrence this layer is built on)
      (https://arxiv.org/abs/1805.02867)
    - Dao et al., 2022. FlashAttention: Fast and Memory-Efficient Exact Attention
      with IO-Awareness. (https://arxiv.org/abs/2205.14135)
    - Rabe and Staats, 2021. Self-attention Does Not Need O(n^2) Memory.
      (https://arxiv.org/abs/2112.05682)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
"""

import keras
from typing import Optional, Union, Any, Dict, Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.initializers.clone import clone_initializer
from dl_techniques.layers.norms.factory import create_normalization_layer
from .common import (
    apply_attention_mask,
    compute_attention_scale,
    mask_dtype,
    validate_head_divisibility,
)
from dl_techniques.utils.keras_registration import register_dl_technique

#: Raised from two places -- the pre-loop rank check and the in-loop
#: dispatch's else -- so mask_slice can never be read unbound.
_UNSUPPORTED_MASK_RANK = (
    "RingAttention supports attention_mask of rank 2 "
    "(batch, seq_len) key-padding, rank 3 (batch, seq_len, seq_len) or rank 4 "
    "(batch, num_heads, seq_len, seq_len); got rank {rank}."
)


@register_dl_technique("dl_techniques.layers.attention.ring_attention")
class RingAttention(keras.layers.Layer):
    """
    Ring Attention: exact attention computed blockwise with an online softmax.

    Partitions the sequence into fixed-size blocks and accumulates attention
    incrementally, so the ``N x N`` score matrix is never materialized. For each
    query block ``Q_i`` the algorithm walks every key/value block ``(K_j, V_j)``,
    forms one score tile ``S_ij = Q_i K_j^T / sqrt(d_k)``, and updates the running
    statistics ``m_new = max(m, max(S_ij))``,
    ``O_new = O * exp(m - m_new) + exp(S_ij - m_new) V_j``,
    ``l_new = l * exp(m - m_new) + sum(exp(S_ij - m_new))``. The block output
    ``O / l`` is mathematically identical to standard attention; memory is
    ``O(block_size^2)`` rather than ``O(N^2)`` for the forward pass. There is no
    gradient checkpointing here, so a backward pass retains per-block
    intermediates and training-time peak memory still grows with ``N``; see the
    measured figures in the module docstring.

    This layer exposes no ``probability_type`` / ``probability_config`` pair
    and does not use the shared
    :class:`~dl_techniques.layers.activations.ProbabilityOutput`, unlike most
    siblings in this package. The blockwise loop never materializes a full
    score row, so it can only use a normalizer with a streaming form; the
    exponential normalizer has one and sparsemax, threshmax and
    adaptive-softmax do not. Adding the hook would either break exactness or
    force full materialization. The same constraint is stated again in
    ``_blockwise_attention`` next to the loop it governs.

    ``call()`` requires a statically-known sequence dimension and raises
    ``ValueError`` when it is ``None``: the block count feeds a Python
    ``range()``, which cannot consume a traced tensor. The batch dimension
    stays fully dynamic.

    The ``dim % num_heads`` check and the ``1 / sqrt(head_dim)`` temperature
    come from :mod:`~dl_techniques.layers.attention.common`; the optional Q/K
    norms come from
    :func:`~dl_techniques.layers.norms.factory.create_normalization_layer`.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input [B, S, dim]                   │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  w_q / w_k / w_v  →  per-head  [B, H, S, d_h]                │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  optional q_norm / k_norm — once, before blocking            │
        │  q = q · scale                                               │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  mask, if given: validate rank 2/3/4, then rescue a row that │
        │  keeps nothing — once, over the full key axis, before the    │
        │  loop. Never per tile: a per-tile rescue reads a fully       │
        │  masked tile as degenerate and un-masks the future under a   │
        │  causal mask.                                                │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  for q_block in range(num_blocks):        ← Python int loop  │
        │    m = −inf,  l = 0,  O = 0                                  │
        │                                                              │
        │    for kv_block in range(num_blocks):                        │
        │      S  = q_blk @ k_blkᵀ            one tile in memory       │
        │      S  = S + mask bias             (rescue_axis=None)       │
        │      m' = max(m, rowmax(S))                                  │
        │      P  = exp(S−m')                                          │
        │      P  = dropout(P)                (training, rate > 0)     │
        │      O  = O·exp(m−m') + P @ v_blk                            │
        │      l  = l·exp(m−m') + rowsum(P)                            │
        │      m  = m'                                                 │
        │                                                              │
        │    block_out = O / l                                         │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  concatenate block outputs on the sequence axis              │
        │  (Python list + ops.concatenate, never slice_update, which   │
        │   has no registered eager gradient on the TF backend)        │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  merge heads → w_o                   │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  Output [B, S, dim]                                          │
        │  return_attention_weights=True → (output, None): the weights │
        │  are never materialized, so there is none to return.         │
        └──────────────────────────────────────────────────────────────┘

    Mask handling — how this site differs from other layers in the package:

    .. code-block:: text

        rank 2  [B, S]       key-padding. Sliced on the key axis only,
                             broadcast over heads and the query block.
                             Mask memory stays O(N), not O(N²).
        rank 3  [B, S, S]    per-query. Block region extracted, then
                             expanded to heads.
        rank 4  [B, H, S, S] per-query per-head; region extracted.
        other                ValueError, before any block work.

        dtype   the accumulation runs in mask_dtype(compute) whenever a
                mask is given. An all-masked tile would otherwise give
                −inf − −inf = NaN in the running max.
        rescue  applied once over the full key axis, before the loop.
                The per-block call passes rescue_axis=None. Per tile it
                would un-mask the future under a causal mask.

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
    :raises ValueError: From ``call()``, if the input's sequence dimension is
        not statically known.
    :raises ValueError: From ``call()``, if ``attention_mask`` has a rank other
        than 2, 3 or 4.

    Input shape:
        3D tensor with shape ``(batch_size, seq_len, dim)``. ``seq_len`` must be
        statically known; ``batch_size`` may be dynamic. The optional
        ``attention_mask`` is ``1 = keep`` and may be rank 2, 3 or 4 as tabulated
        above.

    Output shape:
        3D tensor with shape ``(batch_size, seq_len, dim)``. With
        ``return_attention_weights=True`` the return is the tuple
        ``(output, None)`` — the second entry is always ``None``.

    Example:
        >>> attn = RingAttention(dim=512, num_heads=8, block_size=512)
        >>> x = keras.random.normal((2, 8192, 512))       # static seq_len
        >>> y = attn(x, training=False)                   # (2, 8192, 512)
        >>>
        >>> # Key-padding mask: rank 2 keeps mask memory O(N)
        >>> pad = keras.ops.ones((2, 8192))
        >>> y = attn(x, attention_mask=pad, training=False)
        >>>
        >>> # There are no attention weights to hand back
        >>> y, w = attn(x, return_attention_weights=True)   # w is None

    Note:
        Blockwise processing is exact, not approximate: the output equals what a
        single global softmax would produce, up to floating-point associativity.
        What it trades is time for memory — the doubly-nested loop performs the
        same number of FLOPs as dense attention while holding only one score tile.

    :ivar w_q: Query projection; gets its own cloned initializer.
    :vartype w_q: keras.layers.Dense
    :ivar w_k: Key projection; gets its own cloned initializer.
    :vartype w_k: keras.layers.Dense
    :ivar w_v: Value projection; gets its own cloned initializer.
    :vartype w_v: keras.layers.Dense
    :ivar w_o: Output projection back to ``dim``.
    :vartype w_o: keras.layers.Dense
    :ivar dropout: Attention-weight dropout, applied per tile.
    :vartype dropout: keras.layers.Dropout
    :ivar q_norm: Optional per-head Q-norm, or ``None``.
    :vartype q_norm: keras.layers.Layer or None
    :ivar k_norm: Optional per-head K-norm, or ``None``.
    :vartype k_norm: keras.layers.Layer or None
    :ivar head_dim: ``dim // num_heads``.
    :vartype head_dim: int
    :ivar scale: The ``1 / sqrt(head_dim)`` temperature, a Python float.
    :vartype scale: float
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
        """Validate the configuration and create the projections, dropout and norms.

        This layer owns no weights of its own; :meth:`build` only materializes the
        sub-layers. See the class docstring for the parameter reference.
        """
        super().__init__(**kwargs)

        self._validate_inputs(dim, num_heads, block_size, dropout_rate)

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

        self.head_dim = self.dim // self.num_heads
        # A Python float computed here, never in call(); math.sqrt(int) and
        # math.sqrt(float(int)) give the same double, checked across 27 realistic head dims.
        self.scale = compute_attention_scale(self.head_dim)

        # DECISION plan-2026-08-22T035419-a11304c8/D-200: clone the initializer
        # per projection, never pass self.kernel_initializer directly -- a shared instance gives bit-identical Q/K/V/w_o kernels. See decisions.md.
        self.w_q = keras.layers.Dense(
            self.num_heads * self.head_dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_q'
        )

        self.w_k = keras.layers.Dense(
            self.num_heads * self.head_dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_k'
        )

        self.w_v = keras.layers.Dense(
            self.num_heads * self.head_dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_v'
        )

        self.w_o = keras.layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
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
        validate_head_divisibility(dim, num_heads)
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all sub-layers for robust serialization.

        Sub-layers are built in computational order. The QK-norms are built at
        the per-head shape ``(batch, num_heads, seq_len, head_dim)``, since
        that is what they see once the projections have been reshaped.

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

        Project, reshape to heads, apply the optional QK-norms and the
        temperature once, run the blockwise loop, merge heads and project out.

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
        batch_size = keras.ops.shape(inputs)[0]
        seq_len = keras.ops.shape(inputs)[1]

        # Project to Q, K, V
        # Each is (batch, seq_len, num_heads * head_dim).
        q = self.w_q(inputs)
        k = self.w_k(inputs)
        v = self.w_v(inputs)

        # Reshape for multi-head attention
        q = keras.ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = keras.ops.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = keras.ops.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))

        # Transpose to (batch, num_heads, seq_len, head_dim) for attention computation
        q = keras.ops.transpose(q, (0, 2, 1, 3))
        k = keras.ops.transpose(k, (0, 2, 1, 3))
        v = keras.ops.transpose(v, (0, 2, 1, 3))

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
        # (batch, seq_len, num_heads, head_dim)
        attention_output = keras.ops.transpose(attention_output, (0, 2, 1, 3))
        attention_output = keras.ops.reshape(attention_output, (batch_size, seq_len, self.dim))

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

        The doubly-nested loop over (query block, key/value block) pairs, holding
        one score tile at a time and carrying the ``(m, l, O)`` running state per
        query row.

        :param queries: Query tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        :type queries: keras.KerasTensor
        :param keys: Key tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        :type keys: keras.KerasTensor
        :param values: Value tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        :type values: keras.KerasTensor
        :param attention_mask: Optional mask for attention computation, ``1 = keep``,
            of rank 2, 3 or 4.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]

        :return: Attention output of shape ``(batch, num_heads, seq_len, head_dim)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If the sequence dimension is not statically known, or
            if ``attention_mask`` has a rank other than 2, 3 or 4.
        """
        batch_size = keras.ops.shape(queries)[0]
        num_heads = self.num_heads
        # DECISION plan_2026-06-14_ab855e7e/D-002: read seq_len statically here,
        # never via ops.shape -- range(num_blocks) needs a Python int and crashes on a traced tensor. See decisions.md.
        seq_len = queries.shape[2]
        if seq_len is None:
            raise ValueError(
                "RingAttention requires a statically-known sequence length "
                "(the block-wise loop needs a Python-int block count); got None. "
                "Provide inputs with a fixed sequence dimension."
            )
        head_dim = self.head_dim

        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        # Per-block outputs collect in a Python list; ops.slice_update has no
        # registered eager gradient on the TF backend, so concatenation runs after the loop instead.
        block_outputs = []

        # getattr(..., "name", None) avoids a banned Keras-2 dtype-standardize
        # call; see common.py, D-007.
        state_dtype = getattr(queries.dtype, "name", None) or str(queries.dtype)
        if attention_mask is not None:
            # DECISION plan-2026-07-27T183600-b4ef45f0/D-013: validate the mask
            # rank once here, before the loop -- an unsupported rank used to surface as an UnboundLocalError on the first iteration. See decisions.md.
            mask_rank = len(keras.ops.shape(attention_mask))
            if mask_rank not in (2, 3, 4):
                raise ValueError(_UNSUPPORTED_MASK_RANK.format(rank=mask_rank))

            state_dtype = mask_dtype(state_dtype)
            # A row that keeps nothing is rescued to keep everything, over the
            # full key axis, before the loop -- axis=-1 is the key axis for every supported rank.
            _keep = keras.ops.cast(attention_mask, state_dtype) > 0.0
            _keep = keras.ops.logical_or(
                _keep,
                keras.ops.logical_not(keras.ops.any(_keep, axis=-1, keepdims=True)),
            )
            attention_mask = keras.ops.cast(_keep, state_dtype)

        # Process each query block
        for q_block_idx in range(num_blocks):
            # Get query block bounds
            q_start = q_block_idx * self.block_size
            # `q_end` is a Python int because `seq_len` is static.
            # `q_block` is (batch, num_heads, q_block_size, head_dim).
            q_end = min(q_start + self.block_size, seq_len)
            q_block = queries[:, :, q_start:q_end, :]

            q_block_size = q_end - q_start

            # Initialize running statistics for online softmax
            # (batch, num_heads, q_block_size)
            # `state_dtype` is the compute dtype when no mask is supplied and
            # `mask_dtype(...)` when one is; widened where the mask is validated
            # above.
            running_max = keras.ops.full(
                (batch_size, num_heads, q_block_size),
                -float('inf'),
                dtype=state_dtype
            )
            running_sum = keras.ops.zeros(
                (batch_size, num_heads, q_block_size),
                dtype=state_dtype
            )
            # (batch, num_heads, q_block_size, head_dim)
            accumulated_output = keras.ops.cast(keras.ops.zeros_like(q_block), state_dtype)

            # Process each key/value block for this query block
            for kv_block_idx in range(num_blocks):
                # Get key/value block bounds
                kv_start = kv_block_idx * self.block_size
                # `kv_end` is a Python int because `seq_len` is static.
                # `k_block` is (batch, num_heads, kv_block_size, head_dim).
                kv_end = min(kv_start + self.block_size, seq_len)
                k_block = keys[:, :, kv_start:kv_end, :]
                v_block = values[:, :, kv_start:kv_end, :]

                # scores: (B, H, q_block, d_h) x (B, H, d_h, kv_block) -> (B, H, q_block, kv_block)
                scores = keras.ops.matmul(q_block, keras.ops.transpose(k_block, (0, 1, 3, 2)))

                if attention_mask is not None:
                    # Unreachable in practice (the pre-loop check already rejects
                    # every other rank); this else keeps mask_slice from ever being read unbound. Do not delete it.
                    if len(keras.ops.shape(attention_mask)) == 2:
                        # Key-padding: slice the key axis only, O(N) mask memory
                        # instead of pre-expanding to rank 3.
                        mask_slice = attention_mask[:, kv_start:kv_end]
                        mask_slice = keras.ops.expand_dims(mask_slice, axis=1)
                        mask_slice = keras.ops.expand_dims(mask_slice, axis=1)
                    elif len(keras.ops.shape(attention_mask)) == 3:
                        mask_slice = attention_mask[:, q_start:q_end, kv_start:kv_end]
                        mask_slice = keras.ops.expand_dims(mask_slice, axis=1)
                        mask_slice = keras.ops.repeat(mask_slice, num_heads, axis=1)
                    elif len(keras.ops.shape(attention_mask)) == 4:
                        mask_slice = attention_mask[:, :, q_start:q_end, kv_start:kv_end]
                    else:
                        raise ValueError(
                            _UNSUPPORTED_MASK_RANK.format(
                                rank=len(keras.ops.shape(attention_mask))
                            )
                        )

                    # rescue_axis=None: the fully-masked-row rescue already ran
                    # once over the full key axis before the loop.
                    mask_slice = keras.ops.cast(mask_slice, scores.dtype)
                    scores = apply_attention_mask(
                        scores,
                        mask_slice,
                        out_dtype=state_dtype,
                        rescue_axis=None,
                    )

                # Compute new maximum for safe softmax
                # (batch, num_heads, q_block_size)
                block_max = keras.ops.max(scores, axis=-1)
                new_max = keras.ops.maximum(running_max, block_max)

                # Renormalize previous results
                max_diff = running_max - new_max
                renorm_factor = keras.ops.exp(max_diff)

                # Update running statistics
                running_sum = running_sum * renorm_factor
                accumulated_output = accumulated_output * keras.ops.expand_dims(renorm_factor, axis=-1)

                new_scores = keras.ops.exp(scores - keras.ops.expand_dims(new_max, axis=-1))

                # The cast back to state_dtype matters: self.dropout autocasts
                # to fp16 under mixed_float16 even when handed the float32 accumulation, which would mismatch the matmul below.
                if training and self.dropout_rate > 0:
                    new_scores = keras.ops.cast(
                        self.dropout(new_scores, training=training), state_dtype
                    )

                # out: (B, H, q_block, kv_block) x (B, H, kv_block, d_h) -> (B, H, q_block, d_h)
                new_output = keras.ops.matmul(new_scores, keras.ops.cast(v_block, state_dtype))
                accumulated_output = accumulated_output + new_output

                running_sum = running_sum + keras.ops.sum(new_scores, axis=-1)
                running_max = new_max

            running_sum_expanded = keras.ops.expand_dims(running_sum, axis=-1)
            block_output = accumulated_output / running_sum_expanded

            block_outputs.append(block_output)

        # Loop order places each block at its original q_start offset, so
        # concatenation reassembles the sequence axis correctly.
        outputs = keras.ops.concatenate(block_outputs, axis=2)
        # getattr(..., "name", None) avoids a banned Keras-2 dtype-standardize
        # call; see common.py, D-007.
        return keras.ops.cast(
            outputs, getattr(queries.dtype, "name", None) or str(queries.dtype)
        )

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        The output projection maps back to ``dim``, so the layer is
        shape-preserving.

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
