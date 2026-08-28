"""
Grouped Query Attention: many query heads sharing few key/value heads, with
rotary position embeddings.

Autoregressive decoding is bound by memory bandwidth, not arithmetic. Every
generated token re-reads the entire KV cache, and that cache scales with the
number of key/value heads, so multi-head attention spends most of a decode step
moving keys and values rather than computing with them. Multi-query attention
takes the extreme fix — one K,V head for all queries — and pays for it in
quality: the heads lose their ability to look for different things, because they
can no longer disagree about what a key IS, only about how much to weight it.

Grouped query attention is the interpolation. ``num_heads`` query projections are
kept, but only ``num_kv_heads`` key/value projections, with
``group_size = num_heads // num_kv_heads`` query heads sharing each K,V pair. The
cache shrinks by exactly ``num_heads / num_kv_heads`` while each group retains its
own query subspace, and the published result is that a grouped model interpolates
smoothly between MHA quality and MQA speed rather than falling off a cliff.

The sharing is a repeat, not a projection. K and V are computed in their native
``num_kv_heads`` shape and expanded along the head axis at score time, so what is
STORED is ``num_kv_heads`` heads and what is COMPUTED against is ``num_heads`` — the
expansion is never materialized in the cache, which is the whole point. The
optional QK-norm follows the same discipline: K is normalized in its native
``num_kv_heads`` shape, before grouping, so a group's members see identical
normalized keys.

Three responsibilities are delegated rather than reimplemented — rotary position
embeddings, score normalization (so `probability_type` selects softmax, sparsemax
or adaptive with no branching in `call`), and the optional QK-norms. Mask handling
is NOT shared: this is the third of three broadcasting variants in the
package and the most different, because 4D vision inputs arrive with a flattened
``H*W`` sequence axis and the head axis is materialized by an explicit repeat
rather than broadcast. The reasons are anchored at `_apply_mask`.

Both 3D sequence inputs and 4D vision inputs are accepted; for 4D the spatial axes
are flattened before attention and restored afterwards. `mobile_mqa.MobileMQA`
subclasses this layer and overrides only `call`, so the attribute names it reads
are part of the contract.

Foundational mathematics, with ``d_k = dim // num_heads``::

    Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_k) ) @ V

with RoPE optionally applied to Q and K before scoring, and K, V repeated from
``num_kv_heads`` to ``num_heads`` immediately before the score matmul.

References:
    - Ainslie et al., 2023. GQA: Training Generalized Multi-Query Transformer
      Models from Multi-Head Checkpoints. (https://arxiv.org/abs/2305.13245)
    - Shazeer, 2019. Fast Transformer Decoding: One Write-Head is All You Need.
      (the multi-query predecessor this interpolates towards)
      (https://arxiv.org/abs/1911.02150)
    - Su et al., 2021. RoFormer: Enhanced Transformer with Rotary Position
      Embedding. (https://arxiv.org/abs/2104.09864)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
    - Henry et al., 2020. Query-Key Normalization for Transformers. (the optional
      QK-norm) (https://arxiv.org/abs/2010.04245)
"""

# ---------------------------------------------------------------------

import keras
from typing import Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.initializers import clone_initializer
from dl_techniques.layers.activations import ProbabilityOutput
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.embedding.rotary_position_embedding import RotaryPositionEmbedding

from .common import (
    apply_attention_mask, 
    compute_attention_scale, 
    validate_head_divisibility
)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GroupedQueryAttention(keras.layers.Layer):
    """
    Grouped query attention with optional rotary position embeddings.

    ``num_heads`` query projections share ``num_kv_heads`` key/value projections,
    with ``group_size = num_heads // num_kv_heads`` queries per K,V pair. This cuts
    KV-cache memory by ``num_heads / num_kv_heads`` in autoregressive decoding
    while keeping most of full multi-head attention's representational power —
    every query head keeps its own subspace, only the keys and values are shared.
    ``num_heads % num_kv_heads == 0`` is required, and the repeat that expands K,V
    to ``num_heads`` happens at score time, never in the cache.

    Both 3D sequence inputs ``(B, S, D)`` and 4D vision inputs ``(B, H, W, D)`` are
    accepted; 4D inputs are flattened to ``S = H*W`` for attention and restored
    afterwards.

    **[REUSE]** Three responsibilities are delegated rather than reimplemented:

    -   Rotary position embeddings come from the shared
        :class:`~dl_techniques.layers.embedding.rotary_position_embedding.RotaryPositionEmbedding`.
    -   Score normalization goes through the shared
        :class:`~dl_techniques.layers.activations.ProbabilityOutput` layer, so
        ``probability_type`` selects softmax / sparsemax / adaptive without any
        branching in ``call()``.
    -   Optional QK-norm layers come from
        :func:`~dl_techniques.layers.norms.factory.create_normalization_layer`.

    ``mobile_mqa.MobileMQA`` subclasses this layer. It reuses ``w_q``/``w_k``/
    ``w_v``/``w_o``, ``self.scale``, ``self.attn_prob``, ``self.dropout`` and this
    class's ``compute_output_shape()``, and overrides only ``call()``. Changing
    any of those attribute NAMES is a breaking change for the subclass.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────────────────────────┐
        │  Input [B, S, dim]  or  [B, H, W, dim]                       │
        └───────┬─────────────────────┬──────────────────────┬─────────┘
                ▼                     ▼                      ▼
        ┌───────────────┐   ┌───────────────────┐  ┌───────────────────┐
        │ w_q           │   │ w_k               │  │ w_v               │
        │ num_heads·d_h │   │ num_kv_heads·d_h  │  │ num_kv_heads·d_h  │
        └───────┬───────┘   └─────────┬─────────┘  └─────────┬─────────┘
                ▼                     ▼                      │
        ┌─────────────────────────────────────────────────┐  │
        │  [4D only] flatten H·W → S, then reshape        │  │
        │  to per-head and transpose                      │  │
        │    Q [B, num_heads,    S, d_h]                  │  │
        │    K [B, num_kv_heads, S, d_h]                  │  │
        └───────┬─────────────────────┬───────────────────┘  │
                │                     │                      │
        ┌─────────────────────────────────────────────────┐  │
        │  optional RoPE(Q), RoPE(K)                      │  │
        │  optional q_norm(Q), k_norm(K)                  │  │
        │    K is normalized in its NATIVE                │  │
        │    num_kv_heads shape, BEFORE grouping          │  │
        └───────┬─────────────────────┬───────────────────┘  │
                │                     ▼                      ▼
                │        ┌─────────────────────────────────────────────┐
                │        │ keras.ops.repeat(K, num_groups); V likewise │
                │        │   num_kv_heads → num_heads, at SCORE time.  │
                │        │   Never stored in the KV cache. That is     │
                │        │   where the memory saving lives.            │
                │        └──────────────────────┬──────────────────────┘
                ▼                               ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  scores = Q @ Kᵀ · scale          [B, num_heads, S, S]       │
        │    scale is a Python float from __init__ (D-001)             │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  [+ attention_mask]  rank 2/3/4, keep-predicate; the head    │
        │  axis is REPEATED, not broadcast (see _apply_mask)           │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  attn_prob → dropout → weights @ V   │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  merge heads → w_o                   │
        │  [4D only] restore H, W              │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  Output [B, S, dim]  or  [B, H, W, dim]                      │
        │  return_attention_weights=True → (output, weights)           │
        └──────────────────────────────────────────────────────────────┘

    **Grouping:**

    .. code-block:: text

        num_heads  num_kv_heads  num_groups  cache size   equivalent to
            8            8            1         1.00×     full MHA
            8            4            2         0.50×     GQA-4
            8            2            4         0.25×     GQA-2
            8            1            8         0.125×    MQA

    :param dim: Integer, input/output dimension (embedding size). Must be positive and
        divisible by num_heads.
    :type dim: int
    :param num_heads: Integer, number of attention heads for queries. Must be positive.
    :type num_heads: int
    :param num_kv_heads: Integer, number of key-value heads. Must be positive and divide
        num_heads evenly for grouping.
    :type num_kv_heads: int
    :param max_seq_len: Integer, maximum sequence length for positional embeddings.
        Must be positive. Defaults to 2048.
    :type max_seq_len: int
    :param dropout_rate: Float, dropout rate applied to attention weights.
        Must be between 0.0 and 1.0. Defaults to 0.0.
    :type dropout_rate: float
    :param rope_percentage: Float, fraction of head dimensions to apply rotary
        embeddings to. If 0.0, RoPE is disabled. Defaults to 1.0.
    :type rope_percentage: float
    :param rope_theta: Float, base frequency for rotary position embeddings.
        Must be positive. Defaults to 10000.0.
    :type rope_theta: float
    :param use_bias: Boolean, whether to use bias in linear projections.
        Defaults to False.
    :type use_bias: bool
    :param kernel_initializer: String or initializer instance, initializer for
        kernel weights. Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: String or initializer instance, initializer for
        bias weights. Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param probability_type: String identifier for the attention-score
        normalization strategy, forwarded to :class:`ProbabilityOutput` as its
        ``probability_type``. One of ``"softmax"``, ``"sparsemax"``,
        ``"threshmax"``, ``"adaptive"`` and their aliases. Defaults to
        ``"softmax"``. Routing/hierarchical variants are rejected: they consume
        features rather than logits.
    :type probability_type: str
    :param probability_config: Optional dictionary forwarded to
        :class:`ProbabilityOutput` as ``type_config``. Also supplies the mask
        rescue axis; see :meth:`_apply_mask`. Defaults to ``None``.
    :type probability_config: Optional[Dict[str, Any]]
    :param qk_norm_type: Optional normalization type applied per-head to Q and K
        before scoring, forwarded to :func:`create_normalization_layer`. K is
        normalized in its native ``num_kv_heads`` shape, before grouping. ``None``
        disables QK-norm. Defaults to ``None``.
    :type qk_norm_type: Optional[str]
    :param qk_norm_kwargs: Optional keyword arguments forwarded to
        :func:`create_normalization_layer` for both Q and K norms.
        Defaults to ``None``.
    :type qk_norm_kwargs: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the Layer parent class.

    :raises ValueError: If ``dim``, ``num_heads``, ``num_kv_heads``,
        ``max_seq_len`` or ``rope_theta`` is not positive.
    :raises ValueError: If ``dim`` is not divisible by ``num_heads``, or if
        ``num_heads`` is not divisible by ``num_kv_heads``.
    :raises ValueError: If ``dropout_rate`` or ``rope_percentage`` is outside
        ``[0, 1]``.
    :raises ValueError: If ``probability_type`` is a routing/hierarchical variant.

    Input shape:
        3D tensor ``(batch_size, seq_len, dim)`` or 4D tensor
        ``(batch_size, height, width, dim)``. The optional ``attention_mask`` is a
        ``1 = keep`` predicate of rank 2 ``(B, S)``, rank 3 ``(B, S, S)`` or
        rank 4 ``(B, heads, S, S)``.

    Output shape:
        Same shape as the input — 3D in, 3D out; 4D in, 4D out. With
        ``return_attention_weights=True`` the return is
        ``(output, weights)`` where weights is
        ``(batch_size, num_heads, seq_len, seq_len)``.

    Example:
        >>> # GQA-2: eight query heads over two K,V heads, quarter-size cache
        >>> attn = GroupedQueryAttention(dim=512, num_heads=8, num_kv_heads=2)
        >>> x = keras.random.normal((2, 128, 512))
        >>> y = attn(x, training=False)                    # (2, 128, 512)
        >>>
        >>> # Vision input: spatial axes flattened internally, restored on output
        >>> img = keras.random.normal((2, 16, 16, 512))
        >>> y = attn(img, training=False)                  # (2, 16, 16, 512)
        >>>
        >>> # No RoPE, sparse scores, QK-norm on
        >>> attn = GroupedQueryAttention(
        ...     dim=512, num_heads=8, num_kv_heads=2, rope_percentage=0.0,
        ...     probability_type="sparsemax", qk_norm_type="rms_norm",
        ... )

    Note:
        The KV-cache saving comes from what is STORED, not from what is computed:
        K and V exist in ``num_kv_heads`` shape everywhere except the score matmul,
        where ``ops.repeat`` expands them. Materializing the repeat earlier — in a
        cache, or before the QK-norm — gives up the entire benefit.

    Attributes:
        w_q, w_k, w_v: Query and (narrower) key/value projections; each gets a
            CLONED initializer, so the four roles do not start identical.
        w_o: Output projection back to ``dim``.
        dropout: Attention-weight dropout.
        rope: Shared ``RotaryPositionEmbedding``, or ``None``.
        attn_prob: Shared ``ProbabilityOutput`` score normalizer.
        q_norm, k_norm: Optional QK-norms, or ``None``.
        head_dim: ``dim // num_heads``.
        num_groups: ``num_heads // num_kv_heads``.
        scale: The ``1 / sqrt(head_dim)`` temperature, a Python float. Read by
            ``MobileMQA``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int = 2048,
        dropout_rate: float = 0.0,
        rope_percentage: float = 1.0,
        rope_theta: float = 10000.0,
        use_bias: bool = False,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        probability_type: str = "softmax",
        probability_config: Optional[Dict[str, Any]] = None,
        qk_norm_type: Optional[str] = None,
        qk_norm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Validate the configuration and create every sub-layer.

        This layer owns no weights of its own; :meth:`build` only materializes the
        sub-layers. See the class docstring for the parameter reference.
        """
        super().__init__(**kwargs)

        # Validate inputs
        self._validate_inputs(dim, num_heads, num_kv_heads, max_seq_len,
                            dropout_rate, rope_percentage, rope_theta)

        # Validate probability_type — GQA expects logits, not features.
        _invalid_prob_types = (
            "routing",
            "deterministic_routing",
            "hierarchical",
            "hierarchical_routing",
        )
        if probability_type in _invalid_prob_types:
            raise ValueError(
                f"probability_type='{probability_type}' is not supported by "
                f"GroupedQueryAttention; these expect features not logits. "
                f"Use one of: 'softmax', 'sparsemax', 'adaptive', etc."
            )

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.rope_percentage = rope_percentage
        self.rope_theta = rope_theta
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs

        # Derived parameters
        self.head_dim = self.dim // self.num_heads
        self.num_groups = self.num_heads // self.num_kv_heads

        # DECISION plan_2026-06-14_ab855e7e/D-001
        # The originating plan directory is gone, so this comment is the record.
        # Keep the attention scale a static Python float, computed here in
        # `__init__` and never in `call()`. Don't revert to `keras.ops.sqrt` on a
        # cast scalar. `MobileMQA` inherits this attribute and reads it in its own
        # `call()`. The expression now lives in `common.compute_attention_scale`,
        # whose body is `1.0 / math.sqrt(float(head_dim))` — repr-identical for
        # every realistic head_dim, so `self.scale` is bit-identical to what it
        # replaced and the subclass is unaffected.
        self.scale = compute_attention_scale(self.head_dim)

        # Create all sub-layers here, in __init__.
        # DECISION plan-2026-08-19T163559-499b6f0e/D-068 — give each projection its
        # OWN initializer via `clone_initializer`. Don't hand one `Initializer`
        # INSTANCE to several `Dense` layers: a seedless instance replays its draw,
        # measured in `FastVLM` as `w_q == w_k == w_v == w_o` bit-for-bit in all 6
        # `stage3` attention blocks. See decisions.md D-068.
        self.w_q = keras.layers.Dense(
            self.num_heads * self.head_dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_q'
        )

        self.w_k = keras.layers.Dense(
            self.num_kv_heads * self.head_dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_k'
        )

        self.w_v = keras.layers.Dense(
            self.num_kv_heads * self.head_dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_v'
        )

        self.w_o = keras.layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_o'
        )

        # Attention dropout layer
        self.dropout = keras.layers.Dropout(self.dropout_rate, name='attention_dropout')

        # Rotary position embeddings (only if percentage > 0)
        if self.rope_percentage > 0.0:
            self.rope = RotaryPositionEmbedding(
                head_dim=self.head_dim,
                max_seq_len=self.max_seq_len,
                rope_theta=self.rope_theta,
                rope_percentage=self.rope_percentage,
                name='rope'
            )
        else:
            self.rope = None

        # Probability activation for attention scores
        self.attn_prob = ProbabilityOutput(
            probability_type=self.probability_type,
            type_config=self.probability_config,
            name="attn_prob",
        )

        # Optional QK normalization layers
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

        logger.info(f"GroupedQueryAttention initialized: dim={dim}, "
                   f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, groups={self.num_groups}")

    def _validate_inputs(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        dropout_rate: float,
        rope_percentage: float,
        rope_theta: float
    ) -> None:
        """Validate initialization parameters.

        Two divisibility invariants are checked, and they are NOT the same: the
        head split (``dim`` by ``num_heads``) is delegated to the shared validator,
        while the grouping (``num_heads`` by ``num_kv_heads``) stays local for the
        reason anchored inline.

        :param dim: Model dimension.
        :type dim: int
        :param num_heads: Number of query heads.
        :type num_heads: int
        :param num_kv_heads: Number of key-value heads.
        :type num_kv_heads: int
        :param max_seq_len: Maximum sequence length.
        :type max_seq_len: int
        :param dropout_rate: Dropout rate.
        :type dropout_rate: float
        :param rope_percentage: RoPE percentage.
        :type rope_percentage: float
        :param rope_theta: RoPE base frequency.
        :type rope_theta: float
        :raises ValueError: If any parameter is invalid.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {rope_theta}")
        # R13: adopts the shared validator. Its message is character-for-character
        # what stood here, so the regex pinned at test_group_query_attention.py:113
        # (and, via inheritance, test_mobile_mqa.py:119) still matches.
        validate_head_divisibility(dim, num_heads)
        # The check below is a DIFFERENT invariant and stays local on purpose: it is
        # about how many query heads share one K,V head (the "grouping" in GQA), not
        # about splitting a model dimension into heads. `common.
        # validate_head_divisibility` documents a head-SPLIT precondition
        # (`(..., dim) -> (..., num_heads, dim // num_heads)`); routing this through
        # it would make that documented contract untrue for a saved line, and its
        # `*_name` kwargs would produce the same text anyway. Pinned at
        # test_group_query_attention.py:117.
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
        if not 0.0 <= rope_percentage <= 1.0:
            raise ValueError(f"rope_percentage must be in [0, 1], got {rope_percentage}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and every sub-layer explicitly.

        The projections are rank-agnostic, but RoPE, the score normalizer and the
        QK-norms are not: their build shapes are derived from the input rank, with
        4D inputs contributing ``seq_len = H * W``. Note that Q's norm is built at
        ``num_heads`` and K's at ``num_kv_heads`` — the grouping repeat has not
        happened yet at that point in ``call``.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        # Detect if we need to flatten for 4D inputs during build logic if needed,
        # but Dense layers are generally agnostic to outer dimensions.
        self.w_q.build(input_shape)
        self.w_k.build(input_shape)
        self.w_v.build(input_shape)
        self.w_o.build(input_shape)
        self.dropout.build(input_shape)

        if self.rope is not None:
            # RoPE expects (batch, num_heads, seq_len, head_dim)
            # We estimate shapes based on input rank
            batch_size = input_shape[0] if input_shape[0] is not None else 1

            if len(input_shape) == 4:
                # 4D Input: (B, H, W, C) -> seq_len = H*W
                h = input_shape[1] if input_shape[1] is not None else self.max_seq_len
                w = input_shape[2] if input_shape[2] is not None else 1
                seq_len = h * w
            else:
                # 3D Input: (B, S, C)
                seq_len = input_shape[1] if input_shape[1] is not None else self.max_seq_len

            rope_input_shape = (batch_size, self.num_heads, seq_len, self.head_dim)
            self.rope.build(rope_input_shape)

        # Estimate seq_len for sub-layer build shapes
        batch_size = input_shape[0] if input_shape[0] is not None else 1
        if len(input_shape) == 4:
            h = input_shape[1] if input_shape[1] is not None else self.max_seq_len
            w = input_shape[2] if input_shape[2] is not None else 1
            seq_len = h * w
        else:
            seq_len = input_shape[1] if input_shape[1] is not None else self.max_seq_len

        # Build attention probability layer with score shape (B, num_heads, seq, seq)
        score_shape = (batch_size, self.num_heads, seq_len, seq_len)
        self.attn_prob.build(score_shape)

        # Build QK normalization layers if present.
        # Q has num_heads, K has num_kv_heads (group-query attention).
        if self.q_norm is not None:
            q_norm_shape = (batch_size, self.num_heads, seq_len, self.head_dim)
            k_norm_shape = (batch_size, self.num_kv_heads, seq_len, self.head_dim)
            self.q_norm.build(q_norm_shape)
            self.k_norm.build(k_norm_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        attention_mask: Optional[keras.KerasTensor] = None,
        return_attention_weights: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """Apply grouped query attention.

        Supports 3D ``(B, S, D)`` and 4D ``(B, H, W, D)`` inputs. For 4D inputs,
        spatial dimensions are flattened before attention and restored afterward.
        RoPE and the QK-norms run BEFORE the grouping repeat, so K is normalized in
        its native ``num_kv_heads`` shape.

        :param inputs: Input tensor of shape ``(B, S, D)`` or ``(B, H, W, D)``.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode for dropout.
        :type training: Optional[bool]
        :param attention_mask: Optional attention mask tensor, ``1 = keep``, of
            rank 2, 3 or 4.
        :type attention_mask: Optional[keras.KerasTensor]
        :param return_attention_weights: If True, returns attention weights alongside output.
        :type return_attention_weights: bool
        :return: Output tensor with same shape as input, or tuple of
            (output, attention_weights) if ``return_attention_weights=True``.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]
        """
        input_shape = keras.ops.shape(inputs)
        rank = len(inputs.shape)
        batch_size = input_shape[0]

        # 1. Project to Q, K, V
        # Dense layers broadcast over spatial dims, so this works for 3D and 4D
        q = self.w_q(inputs, training=training)
        k = self.w_k(inputs, training=training)
        v = self.w_v(inputs, training=training)

        # 2. Flatten spatial dimensions if 4D
        if rank == 4:
            height, width = input_shape[1], input_shape[2]
            seq_len = height * width
            q = keras.ops.reshape(q, (batch_size, seq_len, -1))
            k = keras.ops.reshape(k, (batch_size, seq_len, -1))
            v = keras.ops.reshape(v, (batch_size, seq_len, -1))
        else:
            seq_len = input_shape[1]

        # 3. Reshape for Multi-Head Attention
        # Q: (B, S, H, D_h)
        q = keras.ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = keras.ops.reshape(k, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        v = keras.ops.reshape(v, (batch_size, seq_len, self.num_kv_heads, self.head_dim))

        # Transpose to (B, H, S, D_h) for efficient attention
        q = keras.ops.transpose(q, (0, 2, 1, 3))
        k = keras.ops.transpose(k, (0, 2, 1, 3))
        v = keras.ops.transpose(v, (0, 2, 1, 3))

        # 4. Apply RoPE (Optional)
        if self.rope is not None:
            q = self.rope(q, training=training)
            k = self.rope(k, training=training)

        # 4b. Optional QK normalization (applied per-head before scoring).
        # K is normalized in its native num_kv_heads shape, prior to grouping.
        if self.q_norm is not None:
            q = self.q_norm(q, training=training)
            k = self.k_norm(k, training=training)

        # 5. Grouping: Repeat K, V to match Q head count
        if self.num_groups > 1:
            k = keras.ops.repeat(k, self.num_groups, axis=1)
            v = keras.ops.repeat(v, self.num_groups, axis=1)

        # 6. Scaled Dot-Product Attention
        # (B, H, S, D_h) @ (B, H, D_h, S) -> (B, H, S, S)
        scores = keras.ops.matmul(q, keras.ops.transpose(k, (0, 1, 3, 2)))
        # `self.scale` is the Python float precomputed in `__init__` (D-001).
        scores = scores * keras.ops.cast(self.scale, scores.dtype)

        if attention_mask is not None:
            scores = self._apply_mask(scores, attention_mask)

        attention_weights = self.attn_prob(scores)
        attention_weights = self.dropout(attention_weights, training=training)

        # 7. Apply weights to Values
        # (B, H, S, S) @ (B, H, S, D_h) -> (B, H, S, D_h)
        out = keras.ops.matmul(attention_weights, v)

        # 8. Restore Output Shape
        # (B, H, S, D_h) -> (B, S, H, D_h) -> (B, S, D).
        out = keras.ops.transpose(out, (0, 2, 1, 3))
        out = keras.ops.reshape(out, (batch_size, seq_len, self.dim))

        # Final projection
        output = self.w_o(out, training=training)

        # 9. Reshape back to 4D if input was 4D
        if rank == 4:
            output = keras.ops.reshape(output, (batch_size, height, width, self.dim))

        if return_attention_weights:
            return output, attention_weights
        return output

    def _apply_mask(
            self,
            scores: keras.KerasTensor,
            mask: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Broadcast and apply the attention mask to the scores.

        Rank 2 is reshaped (not expanded) to ``(B, 1, 1, S)`` and rank 3 gains a
        head axis; a size-1 head axis is then MATERIALIZED to ``num_heads`` by an
        explicit repeat rather than left to broadcasting. Masking itself is
        delegated to the shared helper, which also performs the fully-masked-row
        rescue at the axis this layer's own ``probability_config`` declares.

        :param scores: Attention scores of shape ``(B, H, S, S)``.
        :type scores: keras.KerasTensor
        :param mask: Attention mask tensor, ``1 = keep``, of rank 2, 3 or 4.
        :type mask: keras.KerasTensor
        :return: Masked scores tensor, in the scores' own dtype.
        :rtype: keras.KerasTensor
        """
        # This helper is NOT shared, on purpose.
        #
        # It is the THIRD variant of mask broadcasting in this package and the
        # most different of the three:
        #   * `multi_head_cross_attention.py::_apply_attention_mask` — casts first,
        #     then a double `ops.expand_dims` for the 2D case, and relies on
        #     broadcasting over the head axis;
        #   * `multi_head_latent_attention.py::_apply_attention_mask` — expands
        #     first, casts after, probes rank with `len(ops.shape(mask))`;
        #   * HERE — `ops.reshape` (not `expand_dims`) for the 2D case, and then an
        #     explicit `ops.repeat` that MATERIALIZES the head axis to `num_heads`
        #     instead of broadcasting it. That repeat is a real extra op with a real
        #     memory cost; it exists because 4D inputs arrive with a flattened
        #     `H*W` sequence axis and broadcast alone proved fragile there.
        #
        # Don't unify the three. Any single body has to pick one cast order, one
        # rank probe, and either broadcast or repeat. That silently rewrites the
        # traced graph of the two layers it did not come from.
        #
        mask_shape = keras.ops.shape(mask)

        # Handle 2D padding mask (B, S)
        if len(mask_shape) == 2:
            mask = keras.ops.reshape(mask, (mask_shape[0], 1, 1, mask_shape[1]))
        # Handle 3D causal/combined mask (B, S, S)
        elif len(mask_shape) == 3:
            mask = keras.ops.expand_dims(mask, axis=1)

        # Broadcast head dim if necessary
        if len(keras.ops.shape(mask)) == 4 and keras.ops.shape(mask)[1] == 1:
            mask = keras.ops.repeat(mask, self.num_heads, axis=1)

        # This site's mask polarity is `1 = keep`, passed through verbatim, so
        # `mask` already IS the keep predicate the shared helper wants. Don't
        # "normalize" it with a `> 0` comparison and don't invert it. The helper
        # infers no polarity, so an inversion raises nothing, changes no shape and
        # stays finite. The layer would simply attend to the padding.
        # `TestGroupedQueryAttentionMaskPolarity` is the only guard that sees it.
        #
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-007 — pin `out_dtype` to the
        # SCORES' own dtype. The form this replaces, `scores + (1.0 - mask) * -1e9`,
        # is `0 * -inf = NaN` at every UNMASKED position in float16: measured on
        # unfixed HEAD (B=2, N=64, D=64, H=4, kv=2), an ALL-ONES mask gave 8192/8192
        # NaN. Don't use `out_dtype=None`; `attn_prob` autocasts a float32 input
        # back to float16. See decisions.md D-007.
        #
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-009 — the fully-masked-row
        # rescue IS applied, via the helper's DEFAULT rescue axis, so the all-`-inf`
        # row is never formed. Don't pass `rescue_axis=None` to get the loud NaN
        # back: that also restores the NaN GRADIENT. Don't move the rescue after the
        # softmax either — `ops.where(row_keeps, w, 0)` still contributes `0 * NaN`
        # backward. See decisions.md D-009 and D-008.
        #
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-017 — DERIVE the rescue axis
        # from this layer's own `probability_config`, not the helper's `-1` default;
        # a caller can move the reduction axis. Measured at the sibling
        # `gated_attention` under `mixed_float16` with `{"axis": -2}` and a dead KEY
        # COLUMN: 8192/8192 non-finite. Don't restore a bare `-1`, and don't read
        # this as the rank/shape INFERENCE `common.py` forbids. See decisions.md D-017.
        scores_dtype = keras.backend.standardize_dtype(scores.dtype)
        return apply_attention_mask(
            scores,
            mask,
            out_dtype=scores_dtype,
            rescue_axis=(self.probability_config or {}).get("axis", -1),
        )

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        Holds for both ranks: the output projection maps back to ``dim``, and 4D
        spatial axes are restored after attention. Inherited by ``MobileMQA``.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        ``kernel_initializer`` is reported as the caller passed it — the per-site
        clones in ``__init__`` do not replace it (D-068).

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'num_kv_heads': self.num_kv_heads,
            'max_seq_len': self.max_seq_len,
            'dropout_rate': self.dropout_rate,
            'rope_percentage': self.rope_percentage,
            'rope_theta': self.rope_theta,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'probability_type': self.probability_type,
            'probability_config': self.probability_config,
            'qk_norm_type': self.qk_norm_type,
            'qk_norm_kwargs': self.qk_norm_kwargs,
        })
        return config

# ---------------------------------------------------------------------