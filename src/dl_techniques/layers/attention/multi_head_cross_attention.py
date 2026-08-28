"""
One multi-head attention engine serving both cross-attention and self-attention,
with pluggable score normalization and optional QK-norm.

Cross-attention and self-attention are the same computation under different
symmetry assumptions. Both score queries against keys and average values by the
result; they differ only in whether the two sequences are the same object. Writing
them as separate layers duplicates the reshape/score/normalize/merge pipeline —
along with every masking subtlety in it — so this layer parameterizes the
difference instead. Supplying a distinct ``kv_input`` gives the asymmetric case,
where a possibly short query sequence reads from a possibly long key/value one, as
in encoder-decoder cross-attention and Perceiver-style latent bottlenecks. Omitting
it gives the symmetric case.

The projection strategy follows that asymmetry rather than being an independent
knob. Cross-attention needs Q computed from one tensor and K, V from another, so
two `Dense` layers are the natural shape. Self-attention reads all three from one
tensor, which permits a single fused `Dense(3 * dim)` — fewer, larger matmuls and
one weight to load. That is why `shared_qk_projections=True` is rejected alongside
a `kv_input`: the fused projection has no second tensor to read from, so the
combination is not a configuration but a contradiction.

Score normalization is not implemented here. It is delegated to the shared
`ProbabilityOutput`, so softmax, sparsemax, threshmax and the adaptive-temperature
variant are one constructor string apart and are each tested in one place. The
adaptive variant is the interesting option: it reads the entropy of a query's
pre-softmax scores and adjusts that row's temperature, sharpening a diffuse
distribution and softening an over-peaked one, which is a per-query calibration
that a fixed `1/sqrt(d_k)` cannot express. Routing and hierarchical strategies are
rejected — they require a fixed `output_dim` and consume features, whereas a score
tensor's last axis is the key sequence length. QK-norm is likewise delegated to the
norm factory, which makes all registered norm types available for free.

Masking carries the most accumulated knowledge in this file, and it is
site-specific by decision rather than by accident. The bias is BUILT with
`keras.ops.where` rather than the arithmetic `(1 - keep) * MASK_BIAS_VALUE` form,
because the latter produces `0 * -inf = NaN` at every UNMASKED position in fp16; a
query row that keeps nothing is rescued to keep everything, so an all-`-inf` row is
never formed and no NaN gradient with it; and the rescue axis is read from this
layer's own `probability_config` rather than hard-coded, since a caller may move
the reduction axis. Three near-identical mask helpers exist across the package and
are not unified, on purpose — the differences in cast order, rank probing and
head-axis handling each matter at their own site.

This class is the shared engine behind two thin facades,
`multi_head_attention.MultiHeadAttention` (self-attention preset) and
`perceiver_attention.PerceiverAttention` (cross-attention preset), so any change to
`call`'s semantics is a change to all three.

Foundational mathematics::

    Attention(Q, K, V) = P( (Q K^T) / (sqrt(d_k) * T) ) V

where ``P`` is the configured normalization and ``T`` is either 1 or, for the
adaptive strategy, a per-query function of the pre-softmax score entropy.

References:
    - Vaswani et al., 2017. Attention Is All You Need. (scaled dot-product
      attention, and the encoder-decoder cross-attention this generalizes)
      (https://arxiv.org/abs/1706.03762)
    - Hinton et al., 2015. Distilling the Knowledge in a Neural Network. (softmax
      temperature as a sharpness control) (https://arxiv.org/abs/1503.02531)
    - Jaegle et al., 2021. Perceiver: General Perception with Iterative Attention.
      (latent queries attending to a large input set — the asymmetric case this
      layer's cross mode serves) (https://arxiv.org/abs/2103.03206)
    - Martins and Astudillo, 2016. From Softmax to Sparsemax: A Sparse Model of
      Attention and Multi-Label Classification. (https://arxiv.org/abs/1602.02068)
    - Henry et al., 2020. Query-Key Normalization for Transformers. (the optional
      QK-norm) (https://arxiv.org/abs/2010.04245)
"""

# ---------------------------------------------------------------------

import keras
from typing import Optional, Any, Dict, Tuple, Union, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------


from dl_techniques.layers.activations import ProbabilityOutput
from dl_techniques.layers.norms import create_normalization_layer
from .common import (
    apply_attention_mask,
    compute_attention_scale,
    validate_head_divisibility
)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
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

    **[REUSE]** Two responsibilities are delegated rather than reimplemented:

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

    **Architecture Overview — cross-attention (separate projections):**

    .. code-block:: text

        ┌────────────────────────────┐  ┌────────────────────────────┐
        │ query_input [B, Q_seq, D]  │  │ kv_input [B, KV_seq, D]    │
        └──────────────┬─────────────┘  └──────────────┬─────────────┘
                       ▼                               ▼
        ┌────────────────────────┐  ┌─────────────────────────────────┐
        │ q: Dense(D) → heads    │  │ kv: Dense(2·D) → split → heads  │
        │   Q [B, H, Q_seq, D_h] │  │   K, V [B, H, KV_seq, D_h]      │
        └──────────────┬─────────┘  └──────────────┬──────────────────┘
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
        │   the bias is BUILT by keras.ops.where, NOT by               │
        │   (1 − keep) · MASK_BIAS_VALUE — that form gives 0 · −inf =  │
        │   NaN in fp16 at every UNMASKED position                     │
        │   a fully-masked row is RESCUED to keep everything, so an    │
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
        │ Output [B, Q_seq, D] — the QUERY's length, not the KV's      │
        └──────────────────────────────────────────────────────────────┘

    **Self-attention (shared projections):**

    .. code-block:: text

        ┌──────────────────────────────────────────────────────────────┐
        │ input [B, seq, D]                                            │
        └──────────────┬───────────────────────────────────────────────┘
                       ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ qkv: ONE Dense(3·D) → split 3 → heads                        │
        │   Q, K, V [B, H, seq, D_h]                                   │
        │   fewer, larger matmuls and one weight — only possible       │
        │   because all three read the SAME tensor, which is why       │
        │   shared_qk_projections=True + kv_input is rejected          │
        └──────────────┬───────────────────────────────────────────────┘
                       ▼
              … identical from here on: QK-norm, scores, mask,
                attn_prob, dropout, @V, merge, proj …
                       ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ Output [B, seq, D]                                           │
        └──────────────────────────────────────────────────────────────┘

    **Mode selection:**

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
    :param output_kernel_initializer: Optional initializer for the OUTPUT
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
        3D tensor ``(batch, query_seq_len, dim)`` — the QUERY's sequence length in
        both modes, which is what makes the asymmetric case useful. One output mode
        only; attention weights are never returned.

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
        A fully-masked query row does not produce ``NaN``: it is rescued to attend
        uniformly instead. That is finite garbage rather than a loud failure, and it
        is a deliberate package-wide ruling — see the anchors in
        :meth:`_apply_attention_mask`. Note also that the ``-inf`` entries handed to
        the probability sub-layer are a CONTRACT ON THAT SUB-LAYER; softmax meets
        it, and not every strategy does.

    Attributes:
        qkv_dense: Fused Q/K/V projection, or ``None`` when not shared.
        q_dense, kv_dense: Separate projections, or ``None`` when shared.
        proj_dense: Output projection; the one Dense that may take
            ``output_kernel_initializer``.
        dropout_layer: Attention-weight dropout, or ``None`` at rate 0.
        attn_prob: The shared ``ProbabilityOutput`` normalizer.
        q_norm, k_norm: Optional QK-norms, or ``None``.
        head_dim: ``dim // num_heads``.
        scale: The ``1 / sqrt(head_dim)`` temperature, a Python float.
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
        # R13: adopts the shared validator. Its message is character-for-character
        # what stood here, so the regex pinned at
        # test_multi_head_cross_attention.py:587 still matches and no diagnostic
        # detail is lost. Position in the validation sequence is unchanged.
        validate_head_divisibility(dim, num_heads)
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store ALL configuration parameters
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

        # Scale factor for attention scores.
        # DECISION plan_2026-06-14_a5ed2c2a/D-002: this must stay a stdlib
        # `math.sqrt` Python float, NOT `keras.ops.sqrt`. The ops version returns
        # a backend tensor even for a static int head_dim, and when `__init__`
        # runs inside a symbolic scratch graph (a lazy build under a functional
        # trace) that tensor leaks out of scope: "<tf.Tensor '.../truediv:0'> is
        # out of scope". head_dim is a static int, so a plain float is correct and
        # bit-identical at the call site. The shared helper used below IS
        # `1.0 / math.sqrt(float(head_dim))`, so the stored value is unchanged,
        # and it is still called from `__init__`, never from `call()`.
        # The originating plan directory is gone, so this comment is the record.
        self.scale = compute_attention_scale(self.head_dim)

        # CREATE sub-layers based on projection strategy
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

        # DECISION plan-2026-08-22T035419-a11304c8/D-160
        # `proj` is the attention block's RESIDUAL-path projection (GPT-2's
        # `attn.c_proj`), so it is the one Dense here that may want a different
        # initializer scale from Q/K/V. Two things must not be "cleaned up":
        # (1) when `output_kernel_initializer` is None the kwargs dict is passed
        # UNCHANGED, so `proj` keeps sharing the SAME initializer INSTANCE as
        # `qkv`/`q`/`kv` — measured: a shared instance replays its draw, so
        # max|delta| is 0.0 at equal shape;
        # (2) the override REPLACES rather than merges, and reaches `proj` only.
        # Letting it reach Q/K/V would shrink those too, which the GPT-2
        # reference does not do. See decisions.md D-160.
        proj_kwargs = dict(dense_kwargs)
        if self.output_kernel_initializer is not None:
            proj_kwargs["kernel_initializer"] = self.output_kernel_initializer
        self.proj_dense = keras.layers.Dense(self.dim, name="proj", **proj_kwargs)
        self.dropout_layer = keras.layers.Dropout(
            self.dropout_rate, name="dropout"
        ) if self.dropout_rate > 0.0 else None

        # Reject routing/hierarchical probability types: they require an
        # ``output_dim`` (and perform their own projection on FEATURES), which
        # is incompatible with operating on attention scores whose last
        # dimension is the dynamic kv sequence length.
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

        # CREATE unified probability output layer for attention-score normalization.
        # ProbabilityOutput validates probability_type internally.
        self.attn_prob = ProbabilityOutput(
            probability_type=self.probability_type,
            type_config=self.probability_config,
            name="attn_prob",
        )

        # CREATE optional QK-norm layers (applied to Q and K projections).
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

        # Robustly determine if input_shape is a list of shapes (cross-attention)
        # or a single shape (self-attention). This works across backends.
        #
        # This three-line predicate is duplicated, on purpose, in THREE subtly
        # different spellings across the package. They are NOT interchangeable and
        # must not be unified into one helper:
        #   * here, and in `compute_output_shape` below: rejects an element that is
        #     `int`/`None` — an "is this NOT a serialized scalar shape" test;
        #   * `MultiHeadLatentAttention`: requires the element to be a
        #     `list`/`tuple`, the complementary positive test, which classifies a
        #     numpy shape element differently;
        #   * `perceiver_attention._is_list_of_shapes`: the positive test, and it
        #     also accepts a `tuple` CONTAINER, because a `.keras` round-trip
        #     hands shapes back as tuples and a list-only check misreads them.
        # Collapsing them would swap one file's classification of an edge-case
        # input for another's — a behaviour change disguised as de-duplication.
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

        **Mask convention: ``attention_mask`` is a KEEP mask.** A value of ``1``
        means attend to that position and ``0`` means mask it out. It is NOT an
        additive ``-inf`` bias and it is NOT a drop mask. Every attention layer in
        this package uses this convention, and several state it by pointing here,
        so flipping it would silently invert every caller's mask.

        A rank-2 mask is a key-padding mask shared by every query row; a rank-3
        mask is per-query. Both are expanded on the head axis and then handed to
        the shared helper, which owns the fp16-safe bias construction and the
        fully-masked-row rescue. Why this site's arguments differ from the other
        near-twins is explained in the comments below.

        :param scores: Attention scores of shape ``(batch, num_heads, query_seq, kv_seq)``.
        :type scores: keras.KerasTensor
        :param attention_mask: Attention mask with supported shapes:
            ``(batch, kv_seq)`` for padding mask, ``(batch, query_seq, kv_seq)``
            for full attention mask, or other broadcastable shapes.
        :type attention_mask: keras.KerasTensor
        :return: Masked scores tensor with same shape as input scores.
        :rtype: keras.KerasTensor
        """
        # This helper is NOT shared, and that is a choice rather than an oversight.
        #
        # Three near-twins exist in this package and none is textually equivalent
        # to another, so unifying them would change op order or dtype handling:
        #
        #   * `MultiHeadLatentAttention._apply_attention_mask` — same broadcast
        #     RESULT, but it casts AFTER expanding (this one casts FIRST) and
        #     probes rank with `len(ops.shape(mask))` rather than `len(mask.shape)`.
        #   * `GroupedQueryAttention._apply_mask` — reshapes rather than expanding
        #     twice for the 2D case, then materializes the head axis with an
        #     explicit repeat to `num_heads` instead of relying on broadcast.
        #
        # WHAT NOT TO DO: do not merge these into one shared helper. A single
        # implementation has to pick one cast order, one rank probe and one
        # head-axis strategy, silently changing the other two call sites.
        #
        attention_mask = keras.ops.cast(attention_mask, scores.dtype)

        # Expand the mask to the scores' rank, (batch, num_heads, query_seq,
        # kv_seq). Rank 2 is a key-padding mask, (batch, kv_seq), which becomes
        # (batch, 1, 1, kv_seq). Rank 3 is a full per-query mask,
        # (batch, query_seq, kv_seq), which becomes (batch, 1, query_seq, kv_seq).
        if len(attention_mask.shape) == 2:
            attention_mask = keras.ops.expand_dims(keras.ops.expand_dims(attention_mask, 1), 1)
        elif len(attention_mask.shape) == 3:
            attention_mask = keras.ops.expand_dims(attention_mask, 1)

        # THIS SITE'S MASK POLARITY, passed through as-is: `attention_mask` is a
        # `1 = keep` predicate, already cast to the scores dtype above on its own
        # line, which is exactly what the shared bias helper wants. Do NOT
        # "normalize" it into a `> 0` comparison and do NOT invert it. The helper
        # infers no polarity, so an inversion raises nothing, changes no shape and
        # stays finite — the layer would simply attend to the padding instead.
        scores_dtype = keras.backend.standardize_dtype(scores.dtype)
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
        project. The output always has the QUERY's sequence length.

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
        # --- 1. Initial Setup and Shape Extraction ---
        # We extract the batch size and query sequence length from the query input.
        # These values will be used repeatedly for reshaping tensors throughout the process.
        # query_input shape: (B, Q_seq, D)
        batch_size = keras.ops.shape(query_input)[0]
        query_seq_len = keras.ops.shape(query_input)[1]

        # --- 2. Project Inputs to Query, Key, and Value Tensors ---
        # This is the core projection step. Depending on the `shared_qk_projections`
        # flag, we use either a single large dense layer for self-attention or
        # separate dense layers for query and key-value pairs.

        if self.shared_qk_projections:
            # --- 2a. Shared Projection (Self-Attention Only) ---
            # This mode is parameter-efficient and only applicable for self-attention,
            # where query, key, and value all originate from the same input tensor.
            if kv_input is not None:
                raise ValueError(
                    "When `shared_qk_projections=True`, `kv_input` must be None "
                    "(self-attention mode only)."
                )

            # Project the single input into a combined Q, K, V tensor.
            # input shape: (B, Q_seq, D)
            # qkv_dense projects to 3 * D to hold Q, K, and V data.
            # qkv shape: (B, Q_seq, 3 * D)
            qkv = self.qkv_dense(query_input)

            # Reshape to separate Q, K, V and split the model dimension into heads.
            # Shape: (B, Q_seq, 3, H, D_h)
            qkv = keras.ops.reshape(qkv, (batch_size, query_seq_len, 3, self.num_heads, self.head_dim))

            # Transpose to bring the head dimension forward, which is the standard
            # format for multi-head attention computation: (3, B, H, Q_seq, D_h)
            qkv = keras.ops.transpose(qkv, (2, 0, 3, 1, 4))

            # Unpack the first dimension to get separate Q, K, V tensors.
            # Each tensor shape: (B, H, Q_seq, D_h)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Optional QK-norm on per-head Q and K.
            if self.q_norm is not None:
                q = self.q_norm(q, training=training)
            if self.k_norm is not None:
                k = self.k_norm(k, training=training)

        else:
            # --- 2b. Separate Projections (Cross-Attention or Self-Attention) ---
            # This is the more general case. If `kv_input` is provided, we perform
            # cross-attention. Otherwise, we perform self-attention on `query_input`.
            kv_source = kv_input if kv_input is not None else query_input

            # --- Project Query ---
            # q_dense projects query_input to the model dimension.
            # query_input shape: (B, Q_seq, D)
            # q shape (after dense): (B, Q_seq, D)
            q = self.q_dense(query_input)
            # Reshape and transpose to multi-head format.
            # Shape (after reshape): (B, Q_seq, H, D_h)
            q = keras.ops.reshape(q, (batch_size, query_seq_len, self.num_heads, self.head_dim))
            # Shape (after transpose): (B, H, Q_seq, D_h)
            q = keras.ops.transpose(q, (0, 2, 1, 3))

            # --- Project Key and Value ---
            # kv_source shape: (B, KV_seq, D)
            kv_seq_len = keras.ops.shape(kv_source)[1]
            # kv_dense projects to 2 * D to hold both K and V data.
            # kv shape (after dense): (B, KV_seq, 2 * D)
            kv = self.kv_dense(kv_source)
            # Reshape to separate K, V and split into heads.
            # Shape (after reshape): (B, KV_seq, 2, H, D_h)
            kv = keras.ops.reshape(kv, (batch_size, kv_seq_len, 2, self.num_heads, self.head_dim))
            # Transpose to standard multi-head format.
            # Shape (after transpose): (2, B, H, KV_seq, D_h)
            kv = keras.ops.transpose(kv, (2, 0, 3, 1, 4))
            # Unpack to get separate K, V tensors.
            # Each tensor shape: (B, H, KV_seq, D_h)
            k, v = kv[0], kv[1]

            # Optional QK-norm on per-head Q and K.
            if self.q_norm is not None:
                q = self.q_norm(q, training=training)
            if self.k_norm is not None:
                k = self.k_norm(k, training=training)

        # --- 3. Scaled Dot-Product Attention ---
        # Now that we have Q, K, and V, we compute the attention scores.
        # This involves a matrix multiplication between Q and K^T, followed by scaling.
        # q shape:      (B, H, Q_seq, D_h)
        # k shape:      (B, H, KV_seq, D_h)
        # k transposed: (B, H, D_h, KV_seq)
        # scores shape: (B, H, Q_seq, KV_seq)
        scores = keras.ops.matmul(q, keras.ops.transpose(k, (0, 1, 3, 2)))

        # Scale scores by the inverse square root of head dimension to prevent gradients
        # from becoming too small. The cast ensures type compatibility.
        scores = scores * keras.ops.cast(self.scale, q.dtype)

        # --- 4. Apply Attention Mask (Optional) ---
        # If a mask is provided, we apply it to the scores. This sets the scores
        # for masked positions to a very large negative number, so they become
        # zero after the softmax normalization.
        if attention_mask is not None:
            # _apply_attention_mask handles broadcasting the mask to the scores' shape.
            # scores shape remains: (B, H, Q_seq, KV_seq)
            scores = self._apply_attention_mask(scores, attention_mask)

        # --- 5. Normalize Scores to get Attention Weights ---
        # Delegate to the unified ProbabilityOutput layer (softmax / sparsemax /
        # threshmax / adaptive / routing / hierarchical).
        # attn_weights shape: (B, H, Q_seq, KV_seq)
        attn_weights = self.attn_prob(scores, training=training)

        # --- 6. Apply Dropout to Attention Weights (Optional) ---
        # During training, dropout is applied to the attention weights to prevent
        # the model from becoming over-reliant on a few key-value pairs.
        if self.dropout_layer is not None:
            # attn_weights shape remains: (B, H, Q_seq, KV_seq)
            attn_weights = self.dropout_layer(attn_weights, training=training)

        # --- 7. Compute Output by Attending to Values ---
        # The attention weights are used to compute a weighted sum of the value vectors.
        # attn_weights shape: (B, H, Q_seq, KV_seq)
        # v shape:            (B, H, KV_seq, D_h)
        # out shape (context vectors): (B, H, Q_seq, D_h)
        out = keras.ops.matmul(attn_weights, v)

        # --- 8. Reshape and Project Final Output ---
        # The outputs from all heads are concatenated and passed through a final
        # linear projection layer.

        # First, transpose to bring the sequence and head dimensions together.
        # Shape (after transpose): (B, Q_seq, H, D_h)
        out = keras.ops.transpose(out, (0, 2, 1, 3))

        # Reshape to concatenate the head outputs, effectively merging the heads.
        # Shape (after reshape): (B, Q_seq, H * D_h) -> (B, Q_seq, D)
        out = keras.ops.reshape(out, (batch_size, query_seq_len, self.dim))

        # Apply the final linear projection. This allows the model to mix information
        # learned from the different attention heads.
        # out shape remains: (B, Q_seq, D)
        return self.proj_dense(out)

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]:
        """Return the QUERY's shape, which is the output's in both modes.

        For cross-attention that is the FIRST of the two input shapes — the key/value
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

# ---------------------------------------------------------------------