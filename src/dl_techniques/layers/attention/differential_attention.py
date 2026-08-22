"""
Differential Multi-Head Attention Implementation.

This module implements Differential Multi-Head Attention as described in the paper
"DIFFERENTIAL TRANSFORMER: Amplifying attention to the relevant context while canceling noise".

The Differential Attention mechanism employs two parallel scaled dot-product attention
streams and computes a weighted difference between them:
``Attention_diff = SDPA1(x) - lambda * SDPA2(x)``.
This design amplifies relevant context signals while attenuating noise, resulting
in more focused attention patterns.

The adaptive lambda parameter is computed as:
``lambda(l) = (0.8 - 0.6 * exp(-0.3 * (l - 1))) * lambda_learned``
and bounded to ``[0.1, 0.9]`` for training stability.

This implementation uses **manual scaled dot-product attention** rather than two
``keras.layers.MultiHeadAttention`` instances. This makes the per-stream attention
probability normalization customizable via :class:`ProbabilityOutput` and exposes an
optional QK-norm hook applied independently to each stream's Q/K projections.

Architecture:
    Five separate ``Dense`` projections produce ``Q1, K1, Q2, K2`` and a single
    **shared** ``V``. Each ``(Q, K)`` pair drives an independent SDPA stream with its
    own optional QK-norm, its own :class:`ProbabilityOutput` instance and its own
    attention dropout, but both streams read the same ``V``. The two contexts are
    combined by ``out1 - lambda * out2``, merged across heads, projected and dropped
    out.

    Two structural properties are load-bearing:

    -   ``V`` is shared, not duplicated. The two streams differ only in *where* they
        look, never in *what* they read, which is what makes their difference a
        noise cancellation rather than an arbitrary linear combination of two
        unrelated attentions.
    -   ``layer_idx`` is a **call** argument, not constructor state, so it is absent
        from ``get_config()``. It only feeds the lambda schedule.

Foundational Mathematics:
    With ``s = 1 / sqrt(d_head)`` and a shared value matrix ``V``::

        A1 = P(Q1 K1^T s) ,   A2 = P(Q2 K2^T s)
        out = (A1 - lambda * A2) V

    Both streams multiply the same ``V``, so the operator applied to the values is
    the single matrix ``A1 - lambda * A2``. Its rows sum to ``1 - lambda`` rather
    than ``1``: the mechanism is deliberately **not** a convex combination, and that
    is the point. Any key ``j`` that both streams attend to equally contributes
    ``(1 - lambda) * A1_ij``, i.e. is attenuated, while a key that only the first
    stream selects keeps its full weight. Common-mode attention — the diffuse,
    context-independent mass that ordinary softmax attention spreads over irrelevant
    tokens — is therefore subtracted out, in exact analogy to a differential
    amplifier rejecting common-mode voltage.

    The mixing coefficient is scheduled by depth::

        lambda(l) = clip( (0.8 - 0.6 * exp(-0.3 * max(l - 1, 0))) * lambda_learned,
                          0.1, 0.9 )

    which starts shallow layers near 0.2 and saturates deep layers near 0.8, so
    cancellation strengthens with depth. The clip is a training-stability guard: at
    ``lambda -> 1`` the operator's rows sum to zero and the residual stream loses its
    DC component.

References:
    Ye, T., Dong, L., Xia, Y., Sun, Y., Zhu, Y., Huang, G., & Wei, F.
    "DIFFERENTIAL TRANSFORMER: Amplifying attention to the relevant context
    while canceling noise". (https://arxiv.org/abs/2410.05258)

    Vaswani et al. (2017). "Attention Is All You Need". NeurIPS.
    (https://arxiv.org/abs/1706.03762)
"""

# ---------------------------------------------------------------------

import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .common import apply_attention_mask, compute_attention_scale
from ..activations import ProbabilityOutput
from ..norms.factory import create_normalization_layer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DifferentialMultiHeadAttention(keras.layers.Layer):
    """
    Differential multi-head attention mechanism with customizable probability
    normalization and optional QK-norm.

    This layer implements differential attention using two parallel manual
    scaled-dot-product-attention (SDPA) streams. It computes their weighted
    difference to amplify relevant context while canceling noise. The key
    innovation is the learnable lambda parameter that balances the contribution
    of the two attention mechanisms.

    The differential attention is computed as:
    ``Attention_diff = SDPA1(x) - lambda * SDPA2(x)``
    where SDPA1 captures primary patterns, SDPA2 identifies noise, and lambda
    controls the noise cancellation strength.

    Each stream's softmax is replaced by an instance of :class:`ProbabilityOutput`
    (``self.attn_prob_1`` / ``self.attn_prob_2``), enabling arbitrary probability
    types (softmax / sparsemax / threshmax / adaptive). Two separate instances are
    used so that per-site debugging and weight inspection remains
    straightforward.

    **[EXTRA CALL ARGUMENT — intentional, frozen]** ``call()`` takes an extra
    positional ``layer_idx: int = 0`` between ``attention_mask`` and ``training``:
    ``call(inputs, attention_mask=None, layer_idx=0, training=None)``. It selects the
    depth-dependent lambda schedule and is a **call** argument rather than
    constructor state, so it is deliberately absent from ``get_config()`` — a
    reloaded layer defaults to ``layer_idx=0`` unless the caller passes it again.
    Do NOT move it into ``__init__`` or drop it to match the package's standard
    signature: a stack of these layers must pass its own depth, and the signature is
    part of the public contract.

    **[REUSE]** The ``1 / sqrt(head_dim)`` temperature comes from
    :mod:`~dl_techniques.layers.attention.common`; score normalization is the shared
    :class:`~dl_techniques.layers.activations.ProbabilityOutput` and the optional
    QK-norms come from
    :func:`~dl_techniques.layers.norms.factory.create_normalization_layer`. Note that
    this layer has **no** ``dim % num_heads`` check to share: ``head_dim`` is an
    explicit required constructor argument here, so ``dim`` and
    ``num_heads * head_dim`` are independent and the output projection reconciles
    them.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────────────────────┐
        │              DifferentialMultiHeadAttention             │
        │                                                         │
        │  Input [B, L, D]                                        │
        │         │                                               │
        │         ▼                                               │
        │   5 separate Dense -> Q1, K1, Q2, K2, V  (V is SHARED   │
        │   by both streams; there is NO fused QKV matmul)        │
        │         │                                               │
        │   ┌─────┴───────────────┬──────────────────┐            │
        │   ▼                     ▼                  ▼            │
        │  Q1,K1               Q2,K2                 V            │
        │   │                     │                  │            │
        │   ▼                     ▼                  │            │
        │ optional q/k_norm_1  optional q/k_norm_2   │            │
        │   │                     │                  │            │
        │   ▼                     ▼                  │            │
        │ scores1 = Q1@K1^T/√d   scores2 = Q2@K2^T/√d│            │
        │   │                     │                  │            │
        │   ▼                     ▼                  │            │
        │ [+ attention_mask]   [+ attention_mask]    │            │
        │  (one mask, both streams; keep-predicate)  │            │
        │   │                     │                  │            │
        │   ▼                     ▼                  │            │
        │ attn_prob_1          attn_prob_2           │            │
        │   │                     │                  │            │
        │   ▼                     ▼                  │            │
        │ attn_dropout_1       attn_dropout_2 (opt)  │            │
        │   │                     │                  │            │
        │   └──── @V ──┐   ┌── @V ┘                  │            │
        │              ▼   ▼                                      │
        │           out1   out2                                   │
        │              │   │                                      │
        │              └── - lambda * ──┐                         │
        │                               │                         │
        │                               ▼                         │
        │                     Differential Output                 │
        │                               │                         │
        │                               ▼                         │
        │                        Output Projection                │
        │                               │                         │
        │                               ▼                         │
        │                            Dropout                      │
        │                               │                         │
        │                               ▼                         │
        │                       Output [B, L, D]                  │
        └─────────────────────────────────────────────────────────┘

    :param dim: Integer, input and output dimension. Must be positive and should be
        divisible by num_heads for optimal performance.
    :type dim: int
    :param num_heads: Integer, number of attention heads for both attention streams.
        Must be positive.
    :type num_heads: int
    :param head_dim: Integer, dimension of each attention head. Must be positive.
    :type head_dim: int
    :param dropout_rate: Float, output dropout rate applied after projection.
        Must be between 0 and 1. Defaults to 0.0.
    :type dropout_rate: float
    :param attention_dropout_rate: Float, dropout rate applied to attention weights in
        both streams. Must be between 0 and 1. Defaults to 0.0.
    :type attention_dropout_rate: float
    :param lambda_init: Float, initial value for the lambda parameter controlling the
        balance between attention streams. Should be between 0 and 1.
        Defaults to 0.8.
    :type lambda_init: float
    :param probability_type: String identifier for the per-stream probability
        normalization strategy. Forwarded to :class:`ProbabilityOutput`. Both streams
        share the same type. Defaults to ``"softmax"``.
    :type probability_type: str
    :param probability_config: Optional dict of strategy-specific arguments forwarded
        to :class:`ProbabilityOutput`. Both streams share the same config.
    :type probability_config: Optional[Dict[str, Any]]
    :param qk_norm_type: Optional normalization type applied to each stream's
        per-head Q and K projections before computing attention scores (QK-norm).
        Forwarded to :func:`create_normalization_layer`. ``None`` disables QK-norm.
    :type qk_norm_type: Optional[str]
    :param qk_norm_kwargs: Optional keyword arguments forwarded to
        :func:`create_normalization_layer` when constructing per-stream Q/K norms.
    :type qk_norm_kwargs: Optional[Dict[str, Any]]
    :param kernel_initializer: String or Initializer, initializer for kernel weights.
        Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional Regularizer, regularizer applied to kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_initializer: String or Initializer, initializer for bias weights.
        Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param bias_regularizer: Optional Regularizer, regularizer applied to bias weights.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param activity_regularizer: Optional Regularizer, regularizer applied to layer output.
    :type activity_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments passed to Layer base class.

    :raises ValueError: If dim is not positive.
    :raises ValueError: If num_heads is not positive.
    :raises ValueError: If head_dim is not positive.
    :raises ValueError: If dropout rates are not between 0 and 1.
    :raises ValueError: If lambda_init is not between 0 and 1.
    :raises ValueError: If ``probability_type`` is a routing / hierarchical variant
        (those consume features and require a fixed ``output_dim``, which is
        incompatible with score logits whose last axis is the kv sequence length).
    :raises ValueError: If sub-layer construction fails for any reason — the
        underlying exception is logged and re-raised as a ``ValueError``.
    :raises ValueError: From ``build()``, if the input is not 3D or its last
        dimension does not match ``dim``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        lambda_init: float = 0.8,
        probability_type: str = "softmax",
        probability_config: Optional[Dict[str, Any]] = None,
        qk_norm_type: Optional[str] = None,
        qk_norm_kwargs: Optional[Dict[str, Any]] = None,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the differential multi-head attention layer."""
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout_rate}")
        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(
                f"attention_dropout_rate must be between 0 and 1, got {attention_dropout_rate}"
            )
        if not (0.0 <= lambda_init <= 1.0):
            raise ValueError(f"lambda_init must be between 0 and 1, got {lambda_init}")

        # Reject routing/hierarchical probability types: they require an
        # output_dim and consume features rather than score logits, which is
        # incompatible with attention scores whose last dimension is the
        # dynamic kv sequence length.
        _ptype_lower = probability_type.lower()
        if _ptype_lower in (
            "routing",
            "deterministic_routing",
            "hierarchical",
            "hierarchical_routing",
        ):
            raise ValueError(
                f"probability_type='{probability_type}' is not supported in "
                "DifferentialMultiHeadAttention: routing/hierarchical strategies "
                "require a fixed output_dim and consume features rather than "
                "score logits. Use one of: 'softmax', 'sparsemax', 'threshmax', "
                "'adaptive'."
            )

        # Store configuration - ALL __init__ parameters must be stored
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.lambda_init = lambda_init
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs

        # Store serialized initializers and regularizers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Per-head projection width (each stream has num_heads * head_dim
        # features for Q and K; V is shared between streams).
        self._proj_dim = self.num_heads * self.head_dim

        # Scale factor for scaled dot-product attention.
        # stdlib math.sqrt (Python float), not ops.sqrt; see D-002 in
        # multi_head_cross_attention.py (symbolic scratch-graph tensor leak).
        #
        # R13: was written out as `1.0 / math.sqrt(float(self.head_dim))`, which is
        # the shared helper's body character-for-character — the swap is a rename,
        # not a numerics change. Re-confirmed with `.hex()` across 27 realistic head
        # dims (1..512) rather than trusted by inspection. Still a Python float
        # computed in `__init__`, never in `call()`, per
        # `plan_2026-06-14_33b77a7a/D-002`.
        self.scale = compute_attention_scale(self.head_dim)

        # CREATE all sub-layers in __init__ following modern Keras 3 pattern
        try:
            dense_kwargs = {
                "kernel_initializer": self.kernel_initializer,
                "bias_initializer": self.bias_initializer,
                "kernel_regularizer": self.kernel_regularizer,
                "bias_regularizer": self.bias_regularizer,
            }

            # Five separate projection Dense layers (one fused per stream's
            # Q/K is also possible but five separate layers keeps debugging
            # trivial and matches the per-site pattern documented above).
            self.q1_dense = keras.layers.Dense(self._proj_dim, name="q1", **dense_kwargs)
            self.k1_dense = keras.layers.Dense(self._proj_dim, name="k1", **dense_kwargs)
            self.q2_dense = keras.layers.Dense(self._proj_dim, name="q2", **dense_kwargs)
            self.k2_dense = keras.layers.Dense(self._proj_dim, name="k2", **dense_kwargs)
            self.v_dense = keras.layers.Dense(self._proj_dim, name="v", **dense_kwargs)

            # Output projection layer
            self.proj = keras.layers.Dense(
                self.dim,
                name='proj',
                **dense_kwargs,
            )

            # Output dropout layer
            self.dropout_layer = keras.layers.Dropout(self.dropout_rate, name='dropout')

            # Per-stream attention-weight dropout (matches the
            # ``attention_dropout_rate`` of the original MHA-based version).
            if self.attention_dropout_rate > 0.0:
                self.attn_dropout_1 = keras.layers.Dropout(
                    self.attention_dropout_rate, name="attn_dropout_1"
                )
                self.attn_dropout_2 = keras.layers.Dropout(
                    self.attention_dropout_rate, name="attn_dropout_2"
                )
            else:
                self.attn_dropout_1 = None
                self.attn_dropout_2 = None

            # Per-stream probability normalization layers (two instances,
            # sharing the same probability_type / probability_config).
            self.attn_prob_1 = ProbabilityOutput(
                probability_type=self.probability_type,
                type_config=self.probability_config,
                name="attn_prob_1",
            )
            self.attn_prob_2 = ProbabilityOutput(
                probability_type=self.probability_type,
                type_config=self.probability_config,
                name="attn_prob_2",
            )

            # Optional per-stream QK-norm. Each stream gets its own pair of
            # Q/K normalization layers so they remain independent.
            if self.qk_norm_type is not None:
                _qk_kwargs = self.qk_norm_kwargs or {}
                self.q_norm_1 = create_normalization_layer(
                    self.qk_norm_type, name="q_norm_1", **_qk_kwargs
                )
                self.k_norm_1 = create_normalization_layer(
                    self.qk_norm_type, name="k_norm_1", **_qk_kwargs
                )
                self.q_norm_2 = create_normalization_layer(
                    self.qk_norm_type, name="q_norm_2", **_qk_kwargs
                )
                self.k_norm_2 = create_normalization_layer(
                    self.qk_norm_type, name="k_norm_2", **_qk_kwargs
                )
            else:
                self.q_norm_1 = None
                self.k_norm_1 = None
                self.q_norm_2 = None
                self.k_norm_2 = None

        except Exception as e:
            logger.error(f"Failed to create DifferentialMultiHeadAttention sub-layers: {e}")
            raise ValueError(
                f"Failed to create DifferentialMultiHeadAttention sub-layers. "
                f"This might be due to invalid configuration parameters. "
                f"Original error: {e}"
            )

        # Weight attributes - created in build()
        self.lambda_param = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and create the lambda parameter weight.

        Creates the learnable lambda parameter and explicitly builds all sub-layers
        for robust serialization following modern Keras 3 patterns.

        :param input_shape: Shape tuple of the input tensor, expected as
            ``(batch_size, seq_len, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected 3D input (batch_size, seq_len, dim), got shape: {input_shape}"
            )

        input_dim = input_shape[-1]
        if input_dim != self.dim:
            raise ValueError(
                f"Input dimension {input_dim} doesn't match expected dimension {self.dim}"
            )

        # Create the layer's own weights - lambda parameter.
        # The lambda-init schedule (preserved exactly from the previous
        # implementation) is: lambda = clip(layer_dep_init * lambda_param, 0.1, 0.9)
        # where layer_dep_init = 0.8 - 0.6 * exp(-0.3 * max(layer_idx - 1, 0)).
        self.lambda_param = self.add_weight(
            name="lambda_param",
            shape=(1,),
            initializer=keras.initializers.Constant(self.lambda_init),
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        # Build projection layers explicitly for serialization.
        self.q1_dense.build(input_shape)
        self.k1_dense.build(input_shape)
        self.q2_dense.build(input_shape)
        self.k2_dense.build(input_shape)
        self.v_dense.build(input_shape)

        # Output projection consumes (B, L, num_heads*head_dim) and produces (B, L, dim).
        proj_input_shape = (input_shape[0], input_shape[1], self._proj_dim)
        self.proj.build(proj_input_shape)
        self.dropout_layer.build(input_shape)

        # Build per-stream probability layers with the attention-score shape.
        attn_shape = (input_shape[0], self.num_heads, input_shape[1], input_shape[1])
        self.attn_prob_1.build(attn_shape)
        self.attn_prob_2.build(attn_shape)

        if self.attn_dropout_1 is not None:
            self.attn_dropout_1.build(attn_shape)
            self.attn_dropout_2.build(attn_shape)

        # Build per-stream QK-norm layers with the per-head Q/K shape.
        if self.q_norm_1 is not None:
            qk_shape = (input_shape[0], self.num_heads, input_shape[1], self.head_dim)
            self.q_norm_1.build(qk_shape)
            self.k_norm_1.build(qk_shape)
            self.q_norm_2.build(qk_shape)
            self.k_norm_2.build(qk_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def get_lambda(self, layer_idx: int = 0) -> keras.KerasTensor:
        """
        Compute the lambda value with layer-dependent adaptation.

        The lambda parameter is adapted based on layer depth following the paper's
        initialization strategy: ``lambda = 0.8 - 0.6 * exp(-0.3 * (layer_idx - 1))``.
        The learned ``lambda_param`` is then applied as a multiplicative factor.

        :param layer_idx: Integer, index of the layer in the network stack (0-based).
            Used to compute layer-dependent lambda initialization.
        :type layer_idx: int
        :return: Tensor containing the computed lambda value, bounded between 0.1
            and 0.9, in this layer's ``variable_dtype`` (i.e. the dtype of
            ``lambda_param`` itself). :meth:`call` casts it to the dtype of the
            attention streams before combining them.
        :rtype: keras.KerasTensor

        .. note::
            **FIXED (plan-2026-07-27T183600-b4ef45f0, step 5b).** The schedule used
            to be hard-coded to ``float32`` (``ops.cast(layer_idx, "float32")``),
            which made this layer unable to run under ANY non-``float32`` policy,
            with or without a mask:

            * under ``float64`` the multiply on the next line raised
              ``InvalidArgumentError: cannot compute Mul as input #1(zero-based) was
              expected to be a float tensor but is a double tensor``, because
              ``lambda_param`` is created in ``variable_dtype`` (``float64``);
            * under ``mixed_float16`` it survived here but then raised at
              ``out1 - lambda_val * out2`` in :meth:`call`, with ``... but is a half
              tensor``.

            The schedule is now evaluated in ``variable_dtype`` — the dtype
            ``lambda_param`` actually has, so it is ``float32`` under both
            ``float32`` and ``mixed_float16`` (bit-identical to the old behavior)
            and ``float64`` under a ``float64`` policy (full precision, rather than
            a ``float32`` schedule silently truncating a ``float64`` parameter).
            Do NOT "simplify" this to ``self.compute_dtype``: under
            ``mixed_float16`` that would evaluate the schedule in ``float16`` and
            round the clip bounds, changing the layer's numerics for no benefit.
        """
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-016
        # Evaluate the whole schedule in the dtype `lambda_param` actually has.
        #
        # WHAT NOT TO DO:
        #   * Do NOT restore the hard-coded `"float32"`. It made this layer unable
        #     to run under `mixed_float16` OR `float64` at all, with or without a
        #     mask (see the note in this method's docstring for both verbatim
        #     raises).
        #   * Do NOT use `self.compute_dtype` instead. Under `mixed_float16` that
        #     evaluates the schedule in float16 and rounds the 0.1 / 0.9 clip
        #     bounds, changing the number this layer computes for no benefit. The
        #     narrowing to the compute dtype belongs at the ONE place the value is
        #     consumed, in `call()`.
        # See decisions.md D-016.
        dtype = self.variable_dtype

        # Layer-dependent initialization following the paper
        # lambda_init = 0.8 - 0.6*exp(-0.3*(layer_idx - 1))
        layer_factor = ops.cast(layer_idx, dtype=dtype)
        exp_term = ops.exp(-0.3 * ops.maximum(layer_factor - 1.0, 0.0))
        layer_dependent_init = 0.8 - 0.6 * exp_term

        # Apply learned lambda parameter as multiplicative factor
        # Clip to ensure training stability
        lambda_val = ops.clip(
            layer_dependent_init * ops.cast(self.lambda_param[0], dtype),
            0.1,
            0.9,
        )

        return lambda_val

    def _apply_attention_mask(
        self,
        scores: keras.KerasTensor,
        attention_mask: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """
        Apply attention mask to scores tensor.

        :param scores: Attention scores of shape ``(batch, num_heads, q_seq, kv_seq)``.
        :type scores: keras.KerasTensor
        :param attention_mask: Attention mask. Supported shapes: ``(batch, kv_seq)``
            (padding mask), ``(batch, q_seq, kv_seq)`` (full mask), or
            ``(batch, num_heads, q_seq, kv_seq)``.
        :type attention_mask: keras.KerasTensor
        :return: Masked scores tensor with same shape as input scores.
        :rtype: keras.KerasTensor

        .. note::
            **FIXED (plan-2026-07-27T183600-b4ef45f0, step 5).** This used to be the
            eighth catalogued instance of the systemic fp16 mask-NaN bug in
            ``layers/attention``: the arithmetic form
            ``scores + (1.0 - attention_mask) * -1e9`` evaluated in ``scores.dtype``,
            which under ``mixed_precision.set_global_policy('mixed_float16')`` is
            ``0 * -inf = NaN`` at every **unmasked** position. It now delegates the
            bias to :func:`~dl_techniques.layers.attention.common.apply_attention_mask`,
            which builds it with ``ops.where`` inside
            :func:`~dl_techniques.layers.attention.common.mask_dtype`, so the product
            cannot be formed at all.

        .. note::
            **ALSO FIXED (same plan, step 5b).** A separate dtype defect used to keep
            this layer from running under ``mixed_float16`` or ``float64`` AT ALL,
            with or without a mask: :meth:`get_lambda` hard-coded its schedule to
            ``float32``. That is now evaluated in ``variable_dtype`` and cast to the
            streams' dtype in :meth:`call`; see the notes at those two sites. All
            three policies are exercised end-to-end by the mask tests in
            ``tests/test_layers/test_attention/test_differential_attention.py``.
        """
        attention_mask = ops.cast(attention_mask, scores.dtype)
        if len(attention_mask.shape) == 2:
            attention_mask = ops.expand_dims(ops.expand_dims(attention_mask, 1), 1)
        elif len(attention_mask.shape) == 3:
            attention_mask = ops.expand_dims(attention_mask, 1)
        # R13: NOT unified with the sibling `_apply_attention_mask` bodies in
        # `multi_head_cross_attention.py` / `multi_head_latent_attention.py` /
        # `group_query_attention.py`. Those were diffed line-by-line in step 4 of the
        # normalization plan and found to differ in cast order and rank-probe form,
        # so a shared helper would change op ordering somewhere. Left local; only the
        # BIAS is shared now, and the cast/expand lines above are byte-identical.
        #
        # THIS SITE'S MASK POLARITY, passed through verbatim: `attention_mask` is a
        # `1 = keep` predicate (already cast to the scores dtype on its own untouched
        # line above), so it IS the keep predicate `apply_attention_mask` wants. Do
        # NOT "normalize" it into a `> 0` comparison or invert it — the helper
        # performs no polarity inference by design, so an inversion here raises
        # nothing, changes no shape and stays finite; the layer would just attend to
        # the padding. `TestDifferentialAttentionMaskBiasExpression::
        # test_masked_positions_receive_no_probability_mass` is the only guard that
        # can see it.
        #
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-007
        # `out_dtype` is pinned to the SCORES' own dtype, so the biased scores stay
        # in the compute dtype (fp16 under `mixed_float16`), where `MASK_BIAS_VALUE`
        # is `-inf` again. That is deliberate and is NOT the bug being fixed: the bug
        # was the `0 * -inf` PRODUCT of the arithmetic form, which `ops.where` inside
        # `mask_dtype(...)` cannot form at all, and a row keeping >= 1 key softmaxes
        # correctly with `-inf` entries. MEASURED on this method directly at
        # (B=2, H=4, N=64) under `mixed_float16`: the pre-fix expression gave
        # 32768/32768 non-finite attention weights for an ALL-ONES mask (one that
        # masks nothing), for padding and for causal; float32 and float64 gave 0.
        # Do NOT "improve" this to `out_dtype=None`: the next consumer is
        # `attn_prob`, a Keras layer with autocasting ON, which drags a float32
        # tensor straight back to the compute dtype.
        # See decisions.md D-007 (plan-2026-07-27T183600-b4ef45f0).
        #
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-009
        # The fully-masked-row rescue arrives via `apply_attention_mask`'s DEFAULT
        # `rescue_axis=-1`: a query row that keeps NOTHING is treated as keeping
        # EVERYTHING, so the all-`-inf` row is never FORMED and no NaN gradient is
        # created either.
        #
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-017
        # The axis is DERIVED from this layer's own `probability_config` rather than
        # left to the helper's `-1` default. BOTH streams' softmaxes (`attn_prob_1` /
        # `attn_prob_2`) are built from that one config, so one derivation covers both.
        # `ProbabilityOutput` reads its softmax `axis` from `type_config`
        # (`activations/probability_output.py:180`) and this layer forwards
        # `probability_config` VERBATIM, so a caller can move the reduction axis and the
        # pre-step-10 "checked, not assumed" claim held only for the DEFAULT config.
        # MEASURED at the sibling `gated_attention` under `mixed_float16` with
        # `probability_config={"axis": -2}` and a dead KEY COLUMN: 8192/8192 non-finite.
        # WHAT NOT TO DO: do NOT restore a bare `-1` (correct only while the caller
        # leaves the config alone) and do NOT read this as the rank/shape INFERENCE the
        # D-009 anchor in `common.py` forbids — this reads the site's own declared
        # config. The full argument lives at the D-017 anchors in `common.py` and
        # `gated_attention.py`. See decisions.md D-017 (plan-2026-07-27T183600-b4ef45f0).
        #
        # WHAT NOT TO DO: do NOT pass `rescue_axis=None` to "get the loud NaN back" —
        # the user ruled the finite-garbage semantics package-wide on 2026-07-28, and
        # opting out also restores the NaN GRADIENT on that row. The full argument
        # lives at the D-009 / D-008 anchors in `common.py`.
        # See decisions.md D-009 and D-008 (plan-2026-07-27T183600-b4ef45f0).
        return apply_attention_mask(
            scores,
            attention_mask,
            out_dtype=keras.backend.standardize_dtype(scores.dtype),
            rescue_axis=(self.probability_config or {}).get("axis", -1),
        )

    def _project_to_heads(
        self,
        x: keras.KerasTensor,
        batch_size: keras.KerasTensor,
        seq_len: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Reshape a projected tensor ``(B, L, H*D_h)`` to ``(B, H, L, D_h)``."""
        x = ops.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def _stream(
        self,
        q: keras.KerasTensor,
        k: keras.KerasTensor,
        v: keras.KerasTensor,
        q_norm: Optional[keras.layers.Layer],
        k_norm: Optional[keras.layers.Layer],
        attn_prob: ProbabilityOutput,
        attn_dropout_layer: Optional[keras.layers.Dropout],
        attention_mask: Optional[keras.KerasTensor],
        training: Optional[bool],
    ) -> keras.KerasTensor:
        """
        Run a single SDPA stream and return ``(B, H, L, D_h)`` context.

        Applies optional QK-norm, computes scaled dot-product scores, optional
        mask, calls the supplied ``ProbabilityOutput`` to normalize the scores,
        applies optional attention-weight dropout, and returns ``attn @ v``.
        """
        if q_norm is not None:
            q = q_norm(q, training=training)
        if k_norm is not None:
            k = k_norm(k, training=training)

        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        scores = scores * ops.cast(self.scale, q.dtype)

        if attention_mask is not None:
            scores = self._apply_attention_mask(scores, attention_mask)

        attn = attn_prob(scores, training=training)

        if attn_dropout_layer is not None:
            attn = attn_dropout_layer(attn, training=training)

        return ops.matmul(attn, v)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        layer_idx: int = 0,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """
        Apply differential attention mechanism.

        Computes the differential attention as:
        ``Attention_diff = SDPA1(x) - lambda * SDPA2(x)``
        where SDPA1 captures primary attention patterns, SDPA2 identifies noise
        patterns, and lambda controls the balance between them.

        :param inputs: Input tensor of shape ``(batch_size, sequence_length, dim)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional attention mask tensor. Can be 2D, 3D, or 4D
            for different masking strategies.
        :type attention_mask: Optional[keras.KerasTensor]
        :param layer_idx: Integer, index of the layer in the network stack (0-based).
            Used for layer-dependent lambda computation. Defaults to 0.
        :type layer_idx: int
        :param training: Optional boolean indicating whether in training mode.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch_size, sequence_length, dim)`` after
            applying differential attention and output projection.
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # Project to Q1, K1, Q2, K2, V and reshape to per-head format.
        q1 = self._project_to_heads(self.q1_dense(inputs), batch_size, seq_len)
        k1 = self._project_to_heads(self.k1_dense(inputs), batch_size, seq_len)
        q2 = self._project_to_heads(self.q2_dense(inputs), batch_size, seq_len)
        k2 = self._project_to_heads(self.k2_dense(inputs), batch_size, seq_len)
        v = self._project_to_heads(self.v_dense(inputs), batch_size, seq_len)

        # Two parallel SDPA streams (V is shared, lambda-init schedule
        # combines them post-hoc).
        out1 = self._stream(
            q1, k1, v,
            self.q_norm_1, self.k_norm_1,
            self.attn_prob_1, self.attn_dropout_1,
            attention_mask, training,
        )
        out2 = self._stream(
            q2, k2, v,
            self.q_norm_2, self.k_norm_2,
            self.attn_prob_2, self.attn_dropout_2,
            attention_mask, training,
        )

        # Compute layer-dependent lambda value (same schedule as original).
        #
        # FIXED (plan-2026-07-27T183600-b4ef45f0, step 5b): `get_lambda` returns
        # the schedule in `variable_dtype`, which under `mixed_float16` is
        # `float32` while `out2` is `float16`. Combining them without this cast
        # raised `InvalidArgumentError: cannot compute Mul as input #1(zero-based)
        # was expected to be a float tensor but is a half tensor` — with NO mask
        # supplied, so the layer could not run under mixed precision at all. The
        # cast is an identity under `float32` and `float64`.
        lambda_val = ops.cast(self.get_lambda(layer_idx), out2.dtype)

        # Differential attention: SDPA1 - lambda*SDPA2
        diff = out1 - lambda_val * out2

        # Merge heads: (B, H, L, D_h) -> (B, L, H*D_h)
        diff = ops.transpose(diff, (0, 2, 1, 3))
        diff = ops.reshape(diff, (batch_size, seq_len, self._proj_dim))

        # Apply output projection and dropout
        output = self.proj(diff, training=training)
        output = self.dropout_layer(output, training=training)

        return output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape, same as input shape for attention layers.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization, includes all constructor parameters.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'lambda_init': self.lambda_init,
            'probability_type': self.probability_type,
            'probability_config': self.probability_config,
            'qk_norm_type': self.qk_norm_type,
            'qk_norm_kwargs': self.qk_norm_kwargs,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
