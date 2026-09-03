"""
One multi-head attention engine serving both cross-attention and
self-attention, with pluggable score normalization and optional QK-norm.

Cross- and self-attention differ only in whether queries and keys come from
the same tensor, so this layer parameterizes that difference rather than
duplicating the reshape/score/normalize/merge pipeline. A distinct
``kv_input`` gives the asymmetric case (encoder-decoder cross-attention,
Perceiver-style latent bottlenecks); omitting it gives self-attention.
Self-attention can additionally fuse Q/K/V into one ``Dense(3 * dim)`` via
``shared_qk_projections``, which needs a single source tensor and so is
rejected together with ``kv_input``. Score normalization (softmax,
sparsemax, threshmax, adaptive-temperature) and QK-norm are both delegated
to shared factories rather than reimplemented here.

This class is the shared engine behind ``MultiHeadAttention``
(self-attention preset) and ``PerceiverAttention`` (cross-attention
preset); a change to ``call()``'s semantics changes all three.

References:
    - Vaswani et al., 2017. Attention Is All You Need. (https://arxiv.org/abs/1706.03762)
    - Hinton et al., 2015. Distilling the Knowledge in a Neural Network. (https://arxiv.org/abs/1503.02531)
    - Jaegle et al., 2021. Perceiver: General Perception with Iterative
      Attention. (https://arxiv.org/abs/2103.03206)
    - Martins and Astudillo, 2016. From Softmax to Sparsemax: A Sparse
      Model of Attention and Multi-Label Classification. (https://arxiv.org/abs/1602.02068)
    - Henry et al., 2020. Query-Key Normalization for Transformers. (https://arxiv.org/abs/2010.04245)
"""

import keras
from typing import Optional, Any, Dict, Tuple, Union, List

from dl_techniques.layers.activations import ProbabilityOutput
from dl_techniques.layers.norms import create_normalization_layer
from .common import (
    apply_attention_mask,
    compute_attention_scale,
    validate_head_divisibility
)
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.attention.multi_head_cross_attention")
class MultiHeadCrossAttention(keras.layers.Layer):
    """
    Unified multi-head attention: cross- or self-attention, pluggable normalization.

    Supports both attention modes with flexible projection strategies,
    comprehensive masking, and any of the shared probability strategies including
    adaptive-temperature softmax.

    In cross-attention mode, separate projection matrices generate Q from the query
    input and K, V from a distinct key-value input. In self-attention mode with
    shared projections, a single dense layer generates Q, K and V from the same
    input. The core computation is
    ``Attention(Q, K, V) = Normalize(Q @ K^T / sqrt(d_k)) @ V``, where Normalize is
    standard softmax, adaptive-temperature softmax, or another registered strategy.

    Two responsibilities are delegated rather than reimplemented:

    -   Score normalization goes through the shared
        :class:`~dl_techniques.layers.activations.ProbabilityOutput` layer, so
        ``softmax`` / ``sparsemax`` / ``threshmax`` / ``adaptive`` are one
        constructor string apart and their implementations are tested once, in
        one place. Do not add an inline ``ops.softmax`` fast path: it would
        bypass ``ProbabilityOutput``'s build/serialization contract and silently
        ignore ``probability_config``.
    -   Optional QK-norm layers come from
        :func:`~dl_techniques.layers.norms.factory.create_normalization_layer`,
        giving this layer all 18 registered norm types for free.

    This class is in turn the shared engine for two thin facades —
    ``multi_head_attention.MultiHeadAttention`` (self-attention preset) and
    ``perceiver_attention.PerceiverAttention`` (cross-attention preset). Any
    change to ``call()``'s semantics is a change to all three.

    Cross-attention architecture (separate projections):

    .. code-block:: text

        ┌────────────────────────────┐  ┌────────────────────────────┐
        │ query_input [B, Q_seq, D]  │  │ kv_input [B, KV_seq, D]    │
        └──────────────┬─────────────┘  └──────────────┬─────────────┘
                       ▼                               ▼
        ┌────────────────────────┐  ┌─────────────────────────────────┐
        │ q: Dense(D) → heads    │  │ kv: Dense(2·D) → split → heads  │
        │   Q [B, H, Q_seq, D_h] │  │   K, V [B, H, KV_seq, D_h]      │
        └──────────────┬─────────┘  └─────────────────┬───────────────┘
                       ▼                              │
        ┌──────────────────────────┐                  │
        │ optional q_norm / k_norm │◄─────────────────┤
        └──────────────┬───────────┘                  │
                       ▼                              │
        ┌──────────────────────────────────────────┐  │
        │ scores = Q @ Kᵀ · scale                  │  │
        │   [B, H, Q_seq, KV_seq]                  │  │
        └──────────────┬───────────────────────────┘  │
                       ▼                              ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ [+ attention_mask]  rank 2 (B, KV_seq) or rank 3             │
        │ (B, Q_seq, KV_seq), keep-predicate                           │
        │   the bias comes from keras.ops.where, not the arithmetic    │
        │   (1 − keep) · MASK_BIAS_VALUE, which gives 0 · −inf = NaN   │
        │   in fp16 at every unmasked position                         │
        │   a fully-masked row is rescued to keep everything, so an    │
        │   all-−inf row is never formed                               │
        └──────────────┬───────────────────────────────────────────────┘
                       ▼                              │
        ┌──────────────────────────────────────────┐  │
        │ attn_prob (softmax / sparsemax /         │  │
        │ threshmax / adaptive) → dropout          │  │
        └──────────────┬───────────────────────────┘  │
                       └────────── @V ◄───────────────┘
                                    ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ merge heads → proj: Dense(D)                                 │
        └──────────────┬───────────────────────────────────────────────┘
                       ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ Output [B, Q_seq, D] — the query's length, not the KV's      │
        └──────────────────────────────────────────────────────────────┘

    Self-attention (shared projections):

    .. code-block:: text

        ┌──────────────────────────────────────────────────────────────┐
        │ input [B, seq, D]                                            │
        └──────────────┬───────────────────────────────────────────────┘
                       ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ qkv: one Dense(3·D) → split 3 → heads                        │
        │   Q, K, V [B, H, seq, D_h]                                   │
        │   fewer, larger matmuls and one weight — possible because    │
        │   all three read the same tensor, which is why               │
        │   shared_qk_projections=True + kv_input is rejected          │
        └──────────────┬───────────────────────────────────────────────┘
                       ▼
              … identical from here on: QK-norm, scores, mask,
                attn_prob, dropout, @V, merge, proj …
                       ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ Output [B, seq, D]                                           │
        └──────────────────────────────────────────────────────────────┘

    Mode selection:

    .. code-block:: text

        kv_input   shared_qk_projections   ->  mode / projections

        None       False   ->  self-attention   q + kv (2 Dense)
        None       True    ->  self-attention   qkv    (1 Dense)
        given      False   ->  cross-attention  q + kv (2 Dense)
        given      True    ->  ValueError       —

    :param dim: Integer, input/output dimension. Must be positive and divisible
        by num_heads.
    :type dim: int
    :param num_heads: Integer, number of attention heads. Must be positive.
        Defaults to 8.
    :type num_heads: int
    :param dropout_rate: Float, dropout rate for attention weights. Must be between
        0.0 and 1.0. Defaults to 0.0.
    :type dropout_rate: float
    :param shared_qk_projections: Boolean, if True, uses a single dense layer for
        Q, K, and V. Only valid for self-attention. Defaults to False.
    :type shared_qk_projections: bool
    :param use_bias: Boolean, whether to use bias in linear projections.
        Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: String or Initializer for kernel weights.
        Defaults to "glorot_uniform".
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param output_kernel_initializer: Optional initializer for the output
        projection (``proj``) alone. ``None`` (the default) leaves ``proj`` on
        ``kernel_initializer``, i.e. the historical behaviour. Supply it only
        when the residual-path projection needs a different scale from Q/K/V --
        GPT-2's ``1/sqrt(2 * n_layer)`` rule is the motivating case.
    :type output_kernel_initializer: Optional[Union[str, keras.initializers.Initializer]]
    :param bias_initializer: String or Initializer for bias vectors.
        Defaults to "zeros".
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param probability_type: String identifier for the attention-score normalization
        strategy. Forwarded to :class:`ProbabilityOutput`. One of ``"softmax"``,
        ``"sparsemax"``, ``"threshmax"``, ``"adaptive"`` (and their aliases).
        Defaults to ``"softmax"`` (standard scaled dot-product attention).
        ``"routing"`` and ``"hierarchical"`` are rejected with a ``ValueError``
        because they require a fixed ``output_dim`` and consume features rather
        than logits.
    :type probability_type: str
    :param probability_config: Optional dictionary of arguments forwarded to the
        underlying :class:`ProbabilityOutput` strategy. For ``"adaptive"`` accepts
        keys such as ``min_temp``, ``max_temp``, ``entropy_threshold``,
        ``polynomial_coeffs``. Its ``axis`` entry, if present, also supplies the
        mask rescue axis; see :meth:`_apply_attention_mask`.
    :type probability_config: Optional[Dict[str, Any]]
    :param qk_norm_type: Optional normalization type applied to Q and K projections
        before computing attention scores (QK-norm). Forwarded to
        :func:`create_normalization_layer`. ``None`` disables QK-norm.
    :type qk_norm_type: Optional[str]
    :param qk_norm_kwargs: Optional keyword arguments forwarded to
        :func:`create_normalization_layer` when constructing Q/K norms.
    :type qk_norm_kwargs: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If ``dim`` is not divisible by ``num_heads``, or if
        parameters are invalid.
    :raises ValueError: If ``shared_qk_projections=True`` is used with ``kv_input``.
    :raises ValueError: If ``probability_type`` is a routing/hierarchical variant.
    :raises ValueError: From ``build()``, if a cross-attention ``input_shape`` is
        not a pair, if the query shape is not rank 3, or if its last dimension is
        known and does not equal ``dim``.

    Input shape:
        - Self-attention: a 3D tensor ``(batch, seq_len, dim)``.
        - Cross-attention: ``query_input`` ``(batch, query_seq_len, dim)`` plus
          ``kv_input`` ``(batch, kv_seq_len, dim)``; ``build`` accepts the pair as a
          list of two shapes.

        The optional ``attention_mask`` is a ``1 = keep`` predicate of shape
        ``(batch, kv_seq_len)``, ``(batch, query_seq_len, kv_seq_len)``, or any
        broadcastable shape.

    Output shape:
        3D tensor ``(batch, query_seq_len, dim)`` — the query's sequence length
        in both modes, which is what makes the asymmetric case useful. One
        output mode only; attention weights are never returned.

    Example:
        >>> # Cross-attention: 64 latents read a 4096-token input
        >>> attn = MultiHeadCrossAttention(dim=512, num_heads=8)
        >>> latents = keras.random.normal((2, 64, 512))
        >>> inputs = keras.random.normal((2, 4096, 512))
        >>> out = attn(latents, kv_input=inputs)          # (2, 64, 512)
        >>>
        >>> # Self-attention with the fused projection
        >>> attn = MultiHeadCrossAttention(dim=512, num_heads=8,
        ...                                shared_qk_projections=True)
        >>> x = keras.random.normal((2, 128, 512))
        >>> out = attn(x)                                  # (2, 128, 512)
        >>>
        >>> # Adaptive temperature plus QK-norm
        >>> attn = MultiHeadCrossAttention(
        ...     dim=512, num_heads=8, probability_type="adaptive",
        ...     probability_config={"min_temp": 0.5, "max_temp": 2.0},
        ...     qk_norm_type="rms_norm",
        ... )

    Note:
        A fully-masked query row does not produce ``NaN``: it is rescued to
        attend uniformly instead, which is finite garbage rather than a loud
        failure. The ``-inf`` entries handed to the probability sub-layer are
        a contract that sub-layer must honor; softmax meets it, and not every
        strategy does.

    :ivar qkv_dense: Fused Q/K/V projection, or ``None`` when not shared.
    :vartype qkv_dense: keras.layers.Dense or None
    :ivar q_dense: Separate query projection, or ``None`` when shared.
    :vartype q_dense: keras.layers.Dense or None
    :ivar kv_dense: Separate key/value projection, or ``None`` when shared.
    :vartype kv_dense: keras.layers.Dense or None
    :ivar proj_dense: Output projection; the one Dense that may take
        ``output_kernel_initializer``.
    :vartype proj_dense: keras.layers.Dense
    :ivar dropout_layer: Attention-weight dropout, or ``None`` at rate 0.
    :vartype dropout_layer: keras.layers.Dropout or None
    :ivar attn_prob: The shared ``ProbabilityOutput`` normalizer.
    :vartype attn_prob: ProbabilityOutput
    :ivar q_norm: Optional Q-norm, or ``None``.
    :vartype q_norm: keras.layers.Layer or None
    :ivar k_norm: Optional K-norm, or ``None``.
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
            dropout_rate: float = 0.0,
            shared_qk_projections: bool = False,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            output_kernel_initializer: Optional[Union[str, keras.initializers.Initializer]] = None,
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            probability_type: str = "softmax",
            probability_config: Optional[Dict[str, Any]] = None,
            qk_norm_type: Optional[str] = None,
            qk_norm_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and create the projections, normalizer and norms.

        Which projections are created depends on ``shared_qk_projections``; the
        unused attributes are set to ``None`` so both paths have the same attribute
        surface. See the class docstring for the parameter reference.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        validate_head_divisibility(dim, num_heads)
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = self.dim // num_heads
        self.dropout_rate = dropout_rate
        self.shared_qk_projections = shared_qk_projections
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.output_kernel_initializer = (
            keras.initializers.get(output_kernel_initializer)
            if output_kernel_initializer is not None else None
        )
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs

        # DECISION plan_2026-06-14_a5ed2c2a/D-002: scale stays a stdlib
        # math.sqrt Python float, never keras.ops.sqrt -- the ops version returns a backend tensor that leaks out of a symbolic scratch graph. See decisions.md.
        self.scale = compute_attention_scale(self.head_dim)

        # Sub-layers depend on the projection strategy.
        dense_kwargs = {
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer
        }

        if self.shared_qk_projections:
            self.qkv_dense = keras.layers.Dense(
                self.dim * 3, name="qkv", **dense_kwargs
            )
            self.q_dense, self.kv_dense = None, None
        else:
            self.q_dense = keras.layers.Dense(self.dim, name="q", **dense_kwargs)
            self.kv_dense = keras.layers.Dense(self.dim * 2, name="kv", **dense_kwargs)
            self.qkv_dense = None

        # DECISION plan-2026-08-22T035419-a11304c8/D-160: proj is the residual-
        # path projection (GPT-2's attn.c_proj) -- output_kernel_initializer=None keeps proj on the same shared initializer instance as the rest. See decisions.md.
        proj_kwargs = dict(dense_kwargs)
        if self.output_kernel_initializer is not None:
            proj_kwargs["kernel_initializer"] = self.output_kernel_initializer
        self.proj_dense = keras.layers.Dense(self.dim, name="proj", **proj_kwargs)
        self.dropout_layer = keras.layers.Dropout(
            self.dropout_rate, name="dropout"
        ) if self.dropout_rate > 0.0 else None

        # Routing/hierarchical types need a fixed output_dim and project
        # features, incompatible with scores whose last dimension is the dynamic kv sequence length.
        _ptype_lower = self.probability_type.lower()
        if _ptype_lower in (
                "routing",
                "deterministic_routing",
                "hierarchical",
                "hierarchical_routing",
        ):
            raise ValueError(
                f"probability_type='{self.probability_type}' is not supported "
                "in MultiHeadCrossAttention: routing/hierarchical strategies "
                "require a fixed output_dim and consume features rather than "
                "score logits. Use one of: 'softmax', 'sparsemax', 'threshmax', "
                "'adaptive'."
            )

        self.attn_prob = ProbabilityOutput(
            probability_type=self.probability_type,
            type_config=self.probability_config,
            name="attn_prob",
        )

        if self.qk_norm_type is not None:
            self.q_norm = create_normalization_layer(
                self.qk_norm_type,
                name="q_norm",
                **(self.qk_norm_kwargs or {}),
            )
            self.k_norm = create_normalization_layer(
                self.qk_norm_type,
                name="k_norm",
                **(self.qk_norm_kwargs or {}),
            )
        else:
            self.q_norm = None
            self.k_norm = None

    def build(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> None:
        """Build every sub-layer, at the shapes each will actually see.

        A single shape means self-attention; a list of two means cross-attention,
        and the query and key/value lengths are then tracked separately — which
        matters, because the score, dropout and probability layers are built at
        ``(B, H, Q_seq, KV_seq)`` while the Q and K norms are built at their own
        per-head sequence lengths.

        :param input_shape: Shape tuple of the input tensor(s). A single tuple for
            self-attention or a list of two tuples for cross-attention.
        :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        :raises ValueError: If a list is given with a length other than 2, if the
            query shape is not rank 3, or if its last dimension is known and does
            not equal ``dim``.
        """
        if self.built:
            return

        # A list of shapes means cross-attention, a single shape means
        # self-attention. Two sibling files spell this predicate differently for their own edge cases; do not unify them.
        is_list_of_shapes = (
            isinstance(input_shape, list) and
            len(input_shape) > 0 and
            not isinstance(input_shape[0], (int, type(None)))
        )

        if is_list_of_shapes:
            if len(input_shape) != 2:
                raise ValueError(f"Expected 2 inputs for cross-attention, got {len(input_shape)}")
            query_shape, kv_shape = input_shape
        else:
            query_shape = kv_shape = input_shape

        # Validate input shapes
        if len(query_shape) != 3:
            raise ValueError(f"Query input must be 3D, got shape {query_shape}")
        if query_shape[-1] is not None and query_shape[-1] != self.dim:
            raise ValueError(f"Query last dimension ({query_shape[-1]}) must match dim ({self.dim})")

        # Build projection layers explicitly for serialization
        if self.shared_qk_projections:
            self.qkv_dense.build(query_shape)
        else:
            self.q_dense.build(query_shape)
            self.kv_dense.build(kv_shape)

        # Build output projection layer
        proj_input_shape = (query_shape[0], query_shape[1], self.dim)
        self.proj_dense.build(proj_input_shape)

        # Build dropout layer if exists
        if self.dropout_layer is not None:
            # Dropout doesn't change shape, use attention weight shape for building
            attn_shape = (query_shape[0], self.num_heads, query_shape[1], kv_shape[1])
            self.dropout_layer.build(attn_shape)

        # Build the unified probability output layer with the attention score
        # shape (B, H, Q_seq, KV_seq). Note: for routing/hierarchical strategies
        # the last-dim may vary at runtime; rebuild semantics follow
        # ProbabilityOutput's behavior.
        attn_shape = (query_shape[0], self.num_heads, query_shape[1], kv_shape[1])
        self.attn_prob.build(attn_shape)

        # Build QK-norm layers (operate on per-head Q/K projections of shape
        # (B, H, seq, D_h)).
        if self.q_norm is not None:
            q_norm_shape = (query_shape[0], self.num_heads, query_shape[1], self.head_dim)
            self.q_norm.build(q_norm_shape)
        if self.k_norm is not None:
            k_norm_shape = (query_shape[0], self.num_heads, kv_shape[1], self.head_dim)
            self.k_norm.build(k_norm_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def _apply_attention_mask(
            self,
            scores: keras.KerasTensor,
            attention_mask: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Broadcast the mask to rank 4 and apply it to the scores.

        ``attention_mask`` is a keep mask: ``1`` means attend to that position,
        ``0`` means mask it out. It is neither an additive ``-inf`` bias nor a
        drop mask. Every attention layer in this package uses this convention.

        A rank-2 mask is a key-padding mask shared by every query row; a rank-3
        mask is per-query. Both are expanded on the head axis and then handed to
        the shared helper, which owns the fp16-safe bias construction and the
        fully-masked-row rescue.

        :param scores: Attention scores of shape ``(batch, num_heads, query_seq, kv_seq)``.
        :type scores: keras.KerasTensor
        :param attention_mask: Attention mask with supported shapes:
            ``(batch, kv_seq)`` for padding mask, ``(batch, query_seq, kv_seq)``
            for full attention mask, or other broadcastable shapes.
        :type attention_mask: keras.KerasTensor
        :return: Masked scores tensor with same shape as input scores.
        :rtype: keras.KerasTensor
        """
        # Two sibling files (MultiHeadLatentAttention, GroupedQueryAttention)
        # implement the same broadcast with a different cast order and rank probe; do not merge them into one shared helper.
        attention_mask = keras.ops.cast(attention_mask, scores.dtype)

        # Expand the mask to the scores' rank, (batch, num_heads, query_seq,
        # kv_seq). Rank 2 is a key-padding mask, (batch, kv_seq), which becomes
        # (batch, 1, 1, kv_seq). Rank 3 is a full per-query mask,
        # (batch, query_seq, kv_seq), which becomes (batch, 1, query_seq, kv_seq).
        if len(attention_mask.shape) == 2:
            attention_mask = keras.ops.expand_dims(keras.ops.expand_dims(attention_mask, 1), 1)
        elif len(attention_mask.shape) == 3:
            attention_mask = keras.ops.expand_dims(attention_mask, 1)

        # The 1=keep mask passes through as-is to the shared bias helper; do not
        # normalize or invert it, since the helper infers no polarity and an inversion would fail silently.
        # getattr(..., "name", None) avoids a banned Keras-2 dtype-standardize
        # call; see common.py, D-007.
        scores_dtype = getattr(scores.dtype, "name", None) or str(scores.dtype)
        return apply_attention_mask(
            scores,
            attention_mask,
            out_dtype=scores_dtype,
            rescue_axis=(self.probability_config or {}).get("axis", -1),
        )

    def call(
            self,
            query_input: keras.KerasTensor,
            kv_input: Optional[keras.KerasTensor] = None,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass; cross-attention when ``kv_input`` is given, else self-attention.

        The projection branch is chosen by ``shared_qk_projections`` and everything
        after it is common: score, mask, normalize, drop out, attend, merge,
        project. The output always has the query's sequence length.

        :param query_input: Query tensor of shape ``(batch, query_seq_len, dim)``.
        :type query_input: keras.KerasTensor
        :param kv_input: Optional Key-Value tensor of shape ``(batch, kv_seq_len, dim)``.
            If ``None``, self-attention is performed on ``query_input``.
        :type kv_input: Optional[keras.KerasTensor]
        :param attention_mask: Optional mask to prevent attention to certain positions.
            Supports shapes: ``(batch, kv_seq_len)`` for padding mask,
            ``(batch, query_seq_len, kv_seq_len)`` for full mask, or broadcastable shapes.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Boolean indicating training or inference mode.
        :type training: Optional[bool]
        :return: Output tensor with shape ``(batch_size, query_seq_len, dim)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``kv_input`` is supplied while
            ``shared_qk_projections=True`` — the fused projection has no second
            tensor to read from.
        """
        batch_size = keras.ops.shape(query_input)[0]
        query_seq_len = keras.ops.shape(query_input)[1]

        if self.shared_qk_projections:
            if kv_input is not None:
                raise ValueError(
                    "When `shared_qk_projections=True`, `kv_input` must be None "
                    "(self-attention mode only)."
                )

            # qkv: (B, Q_seq, D) -> (B, Q_seq, 3*D) -> (B, Q_seq, 3, H, D_h)
            qkv = self.qkv_dense(query_input)
            qkv = keras.ops.reshape(qkv, (batch_size, query_seq_len, 3, self.num_heads, self.head_dim))
            qkv = keras.ops.transpose(qkv, (2, 0, 3, 1, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]

            if self.q_norm is not None:
                q = self.q_norm(q, training=training)
            if self.k_norm is not None:
                k = self.k_norm(k, training=training)

        else:
            kv_source = kv_input if kv_input is not None else query_input

            # q: (B, Q_seq, D) -> (B, Q_seq, H, D_h) -> (B, H, Q_seq, D_h)
            q = self.q_dense(query_input)
            q = keras.ops.reshape(q, (batch_size, query_seq_len, self.num_heads, self.head_dim))
            q = keras.ops.transpose(q, (0, 2, 1, 3))

            # kv: (B, KV_seq, D) -> (B, KV_seq, 2*D) -> (B, KV_seq, 2, H, D_h)
            kv_seq_len = keras.ops.shape(kv_source)[1]
            kv = self.kv_dense(kv_source)
            kv = keras.ops.reshape(kv, (batch_size, kv_seq_len, 2, self.num_heads, self.head_dim))
            kv = keras.ops.transpose(kv, (2, 0, 3, 1, 4))
            k, v = kv[0], kv[1]

            if self.q_norm is not None:
                q = self.q_norm(q, training=training)
            if self.k_norm is not None:
                k = self.k_norm(k, training=training)

        # scores: (B, H, Q_seq, D_h) x (B, H, D_h, KV_seq) -> (B, H, Q_seq, KV_seq)
        scores = keras.ops.matmul(q, keras.ops.transpose(k, (0, 1, 3, 2)))
        scores = scores * keras.ops.cast(self.scale, q.dtype)

        if attention_mask is not None:
            scores = self._apply_attention_mask(scores, attention_mask)

        attn_weights = self.attn_prob(scores, training=training)

        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights, training=training)

        # out: (B, H, Q_seq, KV_seq) x (B, H, KV_seq, D_h) -> (B, H, Q_seq, D_h)
        out = keras.ops.matmul(attn_weights, v)

        # merge heads: (B, H, Q_seq, D_h) -> (B, Q_seq, H, D_h) -> (B, Q_seq, D)
        out = keras.ops.transpose(out, (0, 2, 1, 3))
        out = keras.ops.reshape(out, (batch_size, query_seq_len, self.dim))

        return self.proj_dense(out)

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]:
        """Return the query's shape, which is the output's in both modes.

        For cross-attention that is the first of the two input shapes — the key/value
        length does not appear in the output.

        :param input_shape: Shape tuple or list of shape tuples.
        :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        :return: Output shape tuple.
        :rtype: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        """
        is_list_of_shapes = (
            isinstance(input_shape, list) and
            len(input_shape) > 0 and
            not isinstance(input_shape[0], (int, type(None)))
        )

        if is_list_of_shapes:
            return input_shape[0]
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization, includes all constructor parameters.

        ``output_kernel_initializer`` is emitted as ``None`` when it was never
        supplied, so a round trip does not silently pin ``proj`` to a separate
        initializer.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "shared_qk_projections": self.shared_qk_projections,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "output_kernel_initializer": (
                keras.initializers.serialize(self.output_kernel_initializer)
                if self.output_kernel_initializer is not None else None
            ),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "probability_type": self.probability_type,
            "probability_config": self.probability_config,
            "qk_norm_type": self.qk_norm_type,
            "qk_norm_kwargs": self.qk_norm_kwargs,
        })
        return config