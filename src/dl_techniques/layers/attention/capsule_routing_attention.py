"""
Capsule-based dynamic routing mechanism for attention.

This layer replaces the plain softmax normalization in multi-head attention
with an iterative agreement process taken from Capsule Networks. Plain
attention normalizes each score row on its own. This layer instead lets the
attention components influence one another first, and only then normalizes.
The consensus step is "routing by agreement".

Architecture:
    The layer first computes the usual scaled dot-product attention scores.
    Those are the initial "votes". Two routing mechanisms then refine them:

    1.  **Vertical routing (head-wise).** For one query token, the attention
        distributions from all ``H`` heads become low-level capsules. Dynamic
        routing runs across those capsules, so the view each head captures can
        influence the others and reach a consensus.

    2.  **Horizontal routing (token-wise).** For one query token, the attention
        scores from all source tokens become input capsules. Routing lets those
        source-token views agree on a final attention distribution.

    Both refinements are ADDITIVE on the logits. The layer computes
    ``logits + vertical(logits) + horizontal(logits)``, and only then applies
    the mask and the final probability function. Routing biases the attention
    distribution; it never replaces it.

Foundational Mathematics:
    Dynamic routing refines coupling coefficients ``c = softmax(b)`` between
    low-level capsules (the initial scores) and high-level capsules (the
    refined scores). Each iteration does three things: compute the weighted
    sum ``s``; squash it, ``v = squash(s) = ||s||^2 / (1 + ||s||^2) * s /
    ||s||``; update the log-priors ``b`` by agreement, the dot product of ``v``
    with the votes.

    **Deviation from the cited paper: the coupling axis.** Sabour et al.
    normalize ``c_ij = softmax_j(b_ij)`` over the OUTPUT capsules, so
    ``sum_j c_ij == 1`` for each input capsule ``i``. Every input capsule
    distributes one unit of itself across the outputs, and that competition is
    what makes a vote concentrate where it agrees. This implementation
    normalizes over the INPUT capsule axis (``axis=-2``) instead, so
    ``sum_i c_ij == 1`` for each output capsule. That is the transpose of the
    paper's convention. Measured at ``num_heads=4``, ``key_dim=8``, sequence
    length 8::

        routing_weights shape        : (2, 8, 4, 4)
        sum over axis -2 (input)     : [1.0, 1.0, 1.0, 1.0]
        sum over axis -1 (output)    : [0.5405, 0.5022, 0.2400, 2.7173]

    The sibling ``attention_routing_capsule.py`` makes the same choice explicit
    and caller-selectable, with ``softmax_axis="output"`` by default, matching
    the paper. This class hard-codes the opposite one. Read the citation as the
    source of the iterative scheme, not as a claim that the axis matches.

    Don't "fix" it by flipping ``_site_config(-2)`` to ``(-1)``. That was tried
    and rejected by measurement (decisions.md D-008). ``_horizontal_routing``'s
    positional branch calls ``_dynamic_routing`` with
    ``num_output_capsules = 1``. At ``axis=-1`` that softmax is a size-1 no-op,
    which produces a reproducible NaN under ``mixed_float16``:
    ``TestCapsuleRoutingMaskPolarity::
    test_a_masked_token_barely_influences_the_default_routing_config``
    goes red under the ``mixed_float16`` policy. Making the axis paper-exact
    AND fp16-safe means reworking that degenerate branch. That is a redesign,
    not a repair. The axis is pinned by
    ``test_the_capsule_coupling_axis_is_pinned.py`` on a NON-SQUARE capsule
    configuration, so a transpose cannot satisfy it.

    The squash non-linearity is norm-only. It rescales a vector without
    rotating it, mapping ``||s|| -> ||s||^2 / (1 + ||s||^2)`` into ``[0, 1)``.
    That is what lets a capsule length be read as a probability.

References:
    - Sabour, Frosst, & Hinton, 2017. Dynamic Routing Between Capsules.
      (https://arxiv.org/abs/1710.09829)
    - Duan, et al., 2019. Capsule-Transformer for Neural Machine Translation.
      (https://arxiv.org/abs/1909.04321)

"""

# ---------------------------------------------------------------------

import keras
from typing import Optional, Union, Tuple, Dict, Any
from keras import ops, layers, initializers, regularizers

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers.clone import clone_initializer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.activations.probability_output import ProbabilityOutput

from .common import (
    apply_attention_mask,
    compute_attention_scale
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

# Probability types that cannot be used as drop-in replacements for the
# attention/coupling softmaxes in this layer (they consume features, not
# logits, and reshape the output).
_DISALLOWED_PROB_TYPES: Tuple[str, ...] = (
    "routing",
    "deterministic_routing",
    "hierarchical",
    "hierarchical_routing",
)

# Probability types whose underlying implementation does not honor a
# user-supplied ``axis`` argument. For these we have to fall back to
# axis=-1 routing semantics when the routing math requires a different axis.
_AXIS_AGNOSTIC_PROB_TYPES: Tuple[str, ...] = (
    "adaptive",
    "adaptive_softmax",
)

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.attention.capsule_routing_attention")
class CapsuleRoutingSelfAttention(keras.layers.Layer):
    """
    Capsule Routing Self-Attention mechanism from Capsule-Transformer.

    Extends multi-head self-attention by treating attention weights as capsules
    and running dynamic routing over them. Both vertical (head-wise) and
    horizontal (token-wise) routing are available, and horizontal routing has an
    optional positional constraint.

    Plain attention computes ``A = softmax(QK^T / sqrt(d)) @ V``. This layer adds
    the routing output to the logits before the final softmax:
    ``logits + vertical(logits) + horizontal(logits)``. The squash
    non-linearity from capsule networks is used:
    ``squash(s) = ||s||^2 / (1 + ||s||^2) * s / ||s||``.

    **Architecture Overview:**

    .. code-block:: text

        Input  [B, seq, embed_dim]
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
        ┌───────┐   ┌───────┐   ┌───────┐
        │ query │   │  key  │   │ value │  Dense projections
        └───┬───┘   └───┬───┘   └───┬───┘
            ▼           ▼           │
        ┌───────┐   ┌───────┐       │
        │q_norm │   │k_norm │       │  optional (qk_norm_type)
        └───┬───┘   └───┬───┘       │
            └─────┬─────┘           │
                  ▼                 │
        Q @ K^T * 1/sqrt(key_dim)   │  [B, H, seq, seq]
                  │                 │
          ┌───────┴───────┐         │
          ▼               ▼         │
        ┌──────────┐  ┌──────────┐  │
        │ vertical │  │horizontal│  │  each one optional
        │ routing  │  │ routing  │  │
        │head-wise │  │token-wise│  │
        └────┬─────┘  └────┬─────┘  │
             │        ┌────┴────┐   │   horizontal FORKS on
             │        ▼         ▼   │   use_positional_routing
             │     positional  vec- │   True = a `for l in
             │     unrolled    tor- │   range(N)` unroll, and
             │     O(N)        ised │   a STATIC N is REQUIRED
             │        │         │   │   or it raises ValueError
             │        └────┬────┘   │   False = any N, no loop
             └──────┬──────┘        │
                    ▼               │
        logits + vertical + horizontal   ADDITIVE
                    │               │
                    ▼               │
        keep-mask, optional; a row that keeps
        nothing is rescued to keep everything
                    │               │
                    ▼               │
        attn_prob_attention, over keys, axis=-1
                    │               │
                    ▼               │
        dropout                     │
                    │               │
                    └───────┬───────┘
                            ▼
                      weights @ V   [B, H, seq, value_dim]
                            │
                            ▼
        transpose, reshape to [B, seq, H*value_dim]
                            │
                            ▼
        ┌────────────────────────────────┐
        │  output  (Dense)               │
        └───────────────┬────────────────┘
                        ▼
        Output  [B, seq, embed_dim]

    :param num_heads: Integer, number of attention heads. Must be positive and should
        divide embed_dim evenly for optimal performance.
    :type num_heads: int
    :param key_dim: Optional integer, size of each attention head for query and key.
        If None, defaults to ``embed_dim // num_heads``. Must be positive.
    :type key_dim: Optional[int]
    :param value_dim: Optional integer, size of each attention head for value.
        If None, defaults to key_dim. Must be positive.
    :type value_dim: Optional[int]
    :param dropout_rate: Float, dropout rate applied to attention weights. Must be
        in range [0, 1]. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Boolean, whether to use bias in linear projections. Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: String or Initializer instance for kernel weights.
        Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: String or Initializer instance for bias weights.
        Defaults to 'zeros'.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights. Defaults to None.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias weights. Defaults to None.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param activity_regularizer: Optional regularizer for layer output. Defaults to None.
    :type activity_regularizer: Optional[regularizers.Regularizer]
    :param routing_iterations: Integer, number of dynamic routing iterations. Must be
        positive. Higher values allow more sophisticated routing but increase
        computational cost. Defaults to 3.
    :type routing_iterations: int
    :param use_vertical_routing: Boolean, whether to apply vertical (head-wise) capsule
        routing. Enables information aggregation across attention heads.
        Defaults to True.
    :type use_vertical_routing: bool
    :param use_horizontal_routing: Boolean, whether to apply horizontal (token-wise)
        capsule routing. Enables information aggregation across sequence positions.
        Defaults to True.
    :type use_horizontal_routing: bool
    :param use_positional_routing: Boolean, whether to use positional routing constraints
        for horizontal capsules. When True, tokens can only route information from
        previous positions, preserving sequential order. Defaults to True.
    :type use_positional_routing: bool
    :param epsilon: Float, small constant for numerical stability in norm calculations.
        Must be positive. Defaults to 1e-8.
    :type epsilon: float
    :param probability_type: String naming the probability function used at all
        three softmax sites (final attention weights, routing coupling
        coefficients, vertical-aggregation importance). Forwarded to
        ``ProbabilityOutput``. Defaults to ``"softmax"``. The
        ``"routing"``/``"deterministic_routing"``/``"hierarchical"``/
        ``"hierarchical_routing"`` family is rejected in ``__init__``: those types
        consume features rather than logits and reshape their output, so they
        cannot stand in for a logits-to-coefficients softmax.
    :type probability_type: str
    :param probability_config: Optional configuration dictionary forwarded to
        ``ProbabilityOutput`` as ``type_config``. Any ``"axis"`` key it contains is
        OVERRIDDEN per site (``-1`` / ``-2`` / ``-1``) because the routing maths
        depends on the axis; for the axis-agnostic ``"adaptive"`` family the key is
        dropped instead. Defaults to ``None``.
    :type probability_config: Optional[Dict[str, Any]]
    :param qk_norm_type: Optional normalization type applied to the per-head query
        and key tensors before the dot product. Forwarded to
        ``create_normalization_layer``; separate ``q_norm``/``k_norm`` instances are
        created in ``build()``. ``None`` (default) disables QK normalization.
    :type qk_norm_type: Optional[str]
    :param qk_norm_kwargs: Optional keyword arguments forwarded to
        ``create_normalization_layer`` when building those two norm layers.
        Defaults to ``None``.
    :type qk_norm_kwargs: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any

    :raises ValueError: If num_heads, key_dim, or value_dim is not positive.
    :raises ValueError: If dropout_rate is not in range [0, 1].
    :raises ValueError: If routing_iterations is not positive.
    :raises ValueError: If epsilon is not positive.
    :raises ValueError: If probability_type names a routing/hierarchical variant.
    :raises ValueError: From ``build()``, if the input is not 3D or its last
        dimension is undefined.
    :raises ValueError: If embed_dim is not divisible by num_heads (when key_dim is None).
    :raises ValueError: From ``call()``, if POSITIONAL horizontal routing runs
        (``use_horizontal_routing=True`` and ``use_positional_routing=True``) and
        the sequence length is not statically known. The non-positional horizontal
        path does not need a static length and accepts a dynamic one.
    """

    def __init__(
        self,
        num_heads: int,
        key_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        activity_regularizer: Optional[regularizers.Regularizer] = None,
        routing_iterations: int = 3,
        use_vertical_routing: bool = True,
        use_horizontal_routing: bool = True,
        use_positional_routing: bool = True,
        epsilon: float = 1e-8,
        probability_type: str = "softmax",
        probability_config: Optional[Dict[str, Any]] = None,
        qk_norm_type: Optional[str] = None,
        qk_norm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Validate the configuration and create the axis-fixed sub-layers.

        The four Dense projections are NOT created here. Their widths depend on
        ``embed_dim``, which is only known from ``input_shape``, so they are
        created in :meth:`build`. The three ``ProbabilityOutput`` sub-layers are
        created here, because each one's axis is fixed by the routing maths and
        does not depend on the input shape. See the class docstring for the
        parameter reference.

        :raises ValueError: For any invalid argument; see the class docstring's
            ``:raises:`` list.
        """
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)

        # Validate inputs
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if key_dim is not None and key_dim <= 0:
            raise ValueError(f"key_dim must be positive, got {key_dim}")
        if value_dim is not None and value_dim <= 0:
            raise ValueError(f"value_dim must be positive, got {value_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if routing_iterations <= 0:
            raise ValueError(f"routing_iterations must be positive, got {routing_iterations}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # Validate probability_type early. The "routing"/"hierarchical" families
        # consume features and emit a class distribution; they are incompatible
        # with the logit-to-coupling-coefficient role required by the three
        # softmax sites in this layer.
        if probability_type.lower() in _DISALLOWED_PROB_TYPES:
            raise ValueError(
                f"probability_type='{probability_type}' is not supported by "
                f"CapsuleRoutingSelfAttention. The routing/hierarchical types "
                f"consume features (not logits) and replace a Dense layer; "
                f"they cannot stand in for the attention/coupling softmaxes."
            )

        # Store ALL configuration parameters for complete serialization
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.routing_iterations = routing_iterations
        self.use_vertical_routing = use_vertical_routing
        self.use_horizontal_routing = use_horizontal_routing
        self.use_positional_routing = use_positional_routing
        self.epsilon = epsilon
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs

        # Resolve per-site axis-config compositions. Each of the three
        # probability sites operates on a fixed routing axis; the math
        # depends on this axis, so the routing axis is always enforced and
        # any user-supplied "axis" in probability_config is overridden.
        def _site_config(axis: int) -> Dict[str, Any]:
            """Compose the type_config for one ProbabilityOutput site.

            :param axis: The routing axis this site normalizes over. It is
                written into the returned config for every probability type
                that honors an axis, overriding any ``axis`` the caller put in
                ``probability_config``.
            :type axis: int
            :return: A copy of ``probability_config`` with ``axis`` set to the
                site's routing axis, or with ``axis`` removed when the
                probability type ignores it.
            :rtype: Dict[str, Any]
            """
            cfg = dict(self.probability_config or {})
            if self.probability_type.lower() in _AXIS_AGNOSTIC_PROB_TYPES:
                # Adaptive softmax does not honor a custom axis, so we
                # silently drop it here. Callers needing non-default axes
                # must use softmax/sparsemax/threshmax.
                cfg.pop("axis", None)
            else:
                cfg["axis"] = axis
            return cfg

        # Site 1: final attention weights (axis=-1, normalize over keys).
        self.attn_prob_attention = ProbabilityOutput(
            probability_type=self.probability_type,
            type_config=_site_config(-1),
            name="attn_prob_attention",
        )
        # Site 2: dynamic-routing coupling coefficients (axis=-2, normalize
        # over input capsules). Shared between _vertical_routing and
        # _horizontal_routing calls into _dynamic_routing.
        #
        # DECISION plan-2026-08-27T040114-580f8b63/D-008 — keep axis=-2, the
        # TRANSPOSE of Sabour et al. 2017 that the class docstring cites. Flipping
        # it to -1 makes `_horizontal_routing`'s `num_output_capsules = 1` branch a
        # size-1 softmax no-op, which produced NaN under mixed_float16 when tried.
        # See decisions.md D-008 (plan-2026-08-27T040114-580f8b63).
        self.attn_prob_routing = ProbabilityOutput(
            probability_type=self.probability_type,
            type_config=_site_config(-2),
            name="attn_prob_routing",
        )
        # Site 3: vertical-aggregation importance weights (axis=-1).
        self.attn_prob_aggregation = ProbabilityOutput(
            probability_type=self.probability_type,
            type_config=_site_config(-1),
            name="attn_prob_aggregation",
        )

        # QK-norm sub-layers (optional). Built lazily in build().
        self.q_norm: Optional[keras.layers.Layer] = None
        self.k_norm: Optional[keras.layers.Layer] = None

        # These will be set in build() based on input shape
        self.embed_dim = None
        self.actual_key_dim = None
        self.actual_value_dim = None

        # Create ALL sub-layers in __init__ (modern Keras 3 pattern)
        # Note: Dense layers will be properly configured in build() when we know embed_dim
        self.query_dense = None
        self.key_dense = None
        self.value_dense = None
        self.output_dense = None
        self.dropout_layer = layers.Dropout(self.dropout_rate, name="dropout")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and create all sub-components.

        Creates weight variables for both the layer and its sub-layers, ensuring
        proper serialization compatibility by explicitly building each sub-layer
        in computational order.

        :param input_shape: Shape tuple of the input tensor, expected as
            ``(batch_size, seq_len, embed_dim)``.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: If input is not 3D or dimensions are incompatible.
        """
        # DECISION plan_2026-06-14_7734bacd/D-002
        # The four Dense projections are created here, in build(), not in
        # __init__. Their widths depend on embed_dim = input_shape[-1]:
        # actual_key_dim defaults to embed_dim // num_heads, and output_dense
        # uses embed_dim directly. Do NOT "fix" this by moving them to
        # __init__. The guard below makes build() idempotent, so a second
        # build() (functional reuse, or from_config) cannot re-create and
        # discard Dense weights that were already built and restored.
        # The originating plan directory is gone, so this comment is the record.
        if self.built:
            return

        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input, got shape {input_shape}")

        self.embed_dim = input_shape[-1]
        if self.embed_dim is None:
            raise ValueError("Last dimension of input must be defined")

        # Set actual dimensions based on configuration
        self.actual_key_dim = self.key_dim if self.key_dim is not None else self.embed_dim // self.num_heads
        self.actual_value_dim = self.value_dim if self.value_dim is not None else self.actual_key_dim
        # DECISION plan_2026-06-14_33b77a7a/D-004
        # Precompute 1/sqrt(actual_key_dim) here in build(), where the key dim is
        # finally resolved from input_shape; it is None in __init__. Same pattern
        # as D-002 above. This replaces a per-call ops.sqrt.
        # R13: the value now comes from `common.compute_attention_scale`, whose
        # body is `1.0 / math.sqrt(float(head_dim))`. That is character-identical
        # to the expression it replaced, and hex-probed identical across 27 head
        # dims, 0/27 mismatches. Still a Python float, still computed in build().
        # The originating plan directory is gone, so this comment is the record.
        self._inv_sqrt_key_dim = compute_attention_scale(self.actual_key_dim)

        # Validate dimension compatibility.
        #
        # R13/A4: this check does NOT adopt `common.validate_head_divisibility`,
        # and should not. The helper's message would still satisfy the regex
        # pinned in `test_capsule_routing_attention.py` (`embed_dim \(127\) must
        # be divisible by num_heads \(8\)`, reproducible with
        # dim_name="embed_dim"). What it cannot carry is the trailing "when
        # key_dim is None" clause, and that clause is the point: the constraint
        # is CONDITIONAL. A caller who passed an explicit key_dim never hits it.
        # Dropping the clause would degrade the diagnostic, which A4 forbids.
        # `linear_attention.py`'s conditional check is kept for the same reason.
        #
        # Naming note: `embed_dim` is a local attribute derived from
        # `input_shape[-1]` in build(), not a constructor keyword argument, so no
        # part of the public API depends on the name. It is kept anyway, because
        # it names the same quantity the error message and the tests use.
        if self.key_dim is None and self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads}) "
                f"when key_dim is None"
            )

        # Create projection layers now that we know dimensions.
        # DECISION plan-2026-08-22T035419-a11304c8/D-200 — clone the initializer
        # per projection. Don't simplify back to a bare
        # `kernel_initializer=self.kernel_initializer`: one Initializer INSTANCE
        # reused across same-shape weights measured max|delta| = 0.0 across Q, K,
        # V and output. `seed=` is not the discriminator. See decisions.md D-200.
        self.query_dense = layers.Dense(
            self.num_heads * self.actual_key_dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="query"
        )

        self.key_dense = layers.Dense(
            self.num_heads * self.actual_key_dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="key"
        )

        self.value_dense = layers.Dense(
            self.num_heads * self.actual_value_dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="value"
        )

        self.output_dense = layers.Dense(
            self.embed_dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="output"
        )

        # Build sub-layers explicitly in computational order for robust serialization
        batch_size, seq_len, _ = input_shape

        # Build projection layers
        self.query_dense.build(input_shape)
        self.key_dense.build(input_shape)
        self.value_dense.build(input_shape)

        # Output dense receives concatenated multi-head values
        output_input_shape = (batch_size, seq_len, self.num_heads * self.actual_value_dim)
        self.output_dense.build(output_input_shape)

        # Dropout operates on attention weights: (batch, num_heads, seq_len, seq_len)
        dropout_input_shape = (batch_size, self.num_heads, seq_len, seq_len)
        self.dropout_layer.build(dropout_input_shape)

        # Build the three probability sub-layers with their characteristic
        # tensor shapes.
        # Site 1: final attention logits -> weights, axis=-1.
        attn_shape = (batch_size, self.num_heads, seq_len, seq_len)
        self.attn_prob_attention.build(attn_shape)
        # Site 2: routing coupling tensor inside _dynamic_routing.
        # Vertical routing produces logits of shape
        # (batch, seq_len_q, num_heads_in, num_heads_out); axis=-2.
        # The same sub-layer is reused for horizontal routing where the
        # exact rank can differ, but Softmax/Sparsemax/ThreshMax are
        # shape-agnostic at the layer level (they only care about the
        # ``axis`` argument), so a representative shape here is sufficient
        # for serialization purposes.
        routing_shape = (batch_size, seq_len, self.num_heads, self.num_heads)
        self.attn_prob_routing.build(routing_shape)
        # Site 3: vertical-aggregation importance, applied after a transpose
        # to shape (batch, seq_len_q, seq_len_k, num_heads); axis=-1.
        aggregation_shape = (batch_size, seq_len, seq_len, self.num_heads)
        self.attn_prob_aggregation.build(aggregation_shape)

        # Optional QK normalization layers applied to per-head Q and K
        # tensors of shape (batch, num_heads, seq_len, key_dim).
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
            qk_norm_shape = (batch_size, self.num_heads, seq_len, self.actual_key_dim)
            self.q_norm.build(qk_norm_shape)
            self.k_norm.build(qk_norm_shape)

        # Create vertical routing parameters if enabled
        if self.use_vertical_routing:
            self.vertical_aggregation_weights = self.add_weight(
                name="vertical_aggregation_weights",
                shape=(self.num_heads, self.num_heads),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )

            if self.use_bias:
                self.vertical_aggregation_bias = self.add_weight(
                    name="vertical_aggregation_bias",
                    shape=(self.num_heads,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    trainable=True
                )
            else:
                self.vertical_aggregation_bias = None
        else:
            self.vertical_aggregation_weights = None
            self.vertical_aggregation_bias = None

        # Always call parent build at the END
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of capsule routing self-attention.

        :param inputs: Input tensor of shape ``(batch_size, seq_len, embed_dim)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional attention mask tensor. Can be
            ``(batch_size, seq_len)`` for padding mask or
            ``(batch_size, seq_len, seq_len)`` for causal/custom mask. It is a KEEP
            predicate (``True`` / nonzero = attend). A query row that keeps nothing
            is rescued and treated as keeping everything, in every dtype — see the
            D-006 anchor in :meth:`_apply_attention_mask`.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Boolean indicating training mode for dropout.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch_size, seq_len, embed_dim)`` with
            contextualized representations enhanced by capsule routing.
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # Linear projections for Q, K, V.
        # query and key come out as (batch, seq_len, num_heads * key_dim);
        # value as (batch, seq_len, num_heads * value_dim).
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Reshape to multi-head format
        query = ops.reshape(query, (batch_size, seq_len, self.num_heads, self.actual_key_dim))
        key = ops.reshape(key, (batch_size, seq_len, self.num_heads, self.actual_key_dim))
        value = ops.reshape(value, (batch_size, seq_len, self.num_heads, self.actual_value_dim))

        # Transpose to (batch, num_heads, seq_len, head_dim)
        query = ops.transpose(query, [0, 2, 1, 3])
        key = ops.transpose(key, [0, 2, 1, 3])
        value = ops.transpose(value, [0, 2, 1, 3])

        # Optional QK normalization (applied per-head before the dot product).
        if self.q_norm is not None:
            query = self.q_norm(query, training=training)
        if self.k_norm is not None:
            key = self.k_norm(key, training=training)

        # Compute scaled dot-product attention logits
        attention_logits = ops.matmul(query, ops.transpose(key, [0, 1, 3, 2]))
        attention_logits = attention_logits * self._inv_sqrt_key_dim

        # Apply capsule routing enhancements
        routing_output = attention_logits

        if self.use_vertical_routing:
            vertical_output = self._vertical_routing(attention_logits)
            routing_output = routing_output + vertical_output

        if self.use_horizontal_routing:
            horizontal_output = self._horizontal_routing(attention_logits)
            routing_output = routing_output + horizontal_output

        # Apply attention mask if provided
        if attention_mask is not None:
            routing_output = self._apply_attention_mask(routing_output, attention_mask)

        # Convert to attention weights and apply dropout
        attention_weights = self.attn_prob_attention(routing_output, training=training)
        attention_weights = self.dropout_layer(attention_weights, training=training)

        # Apply attention to values
        attended_values = ops.matmul(attention_weights, value)
        # Shape: (batch, num_heads, seq_len, value_dim)

        # Transpose and reshape to concatenate heads
        attended_values = ops.transpose(attended_values, [0, 2, 1, 3])
        # Shape: (batch, seq_len, num_heads, value_dim)

        concatenated = ops.reshape(
            attended_values, (batch_size, seq_len, self.num_heads * self.actual_value_dim)
        )

        # Final linear projection
        output = self.output_dense(concatenated)
        return output

    def _apply_attention_mask(
        self,
        attention_logits: keras.KerasTensor,
        attention_mask: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Apply attention mask to logits.

        :param attention_logits: Attention logits of shape
            ``(batch, num_heads, seq_len, seq_len)``.
        :type attention_logits: keras.KerasTensor
        :param attention_mask: Attention mask tensor. Interpreted as a KEEP
            predicate: ``True`` / nonzero means "attend to this position". A query
            row that keeps NOTHING is rescued and treated as keeping everything —
            see the D-006 anchor below.
        :type attention_mask: keras.KerasTensor
        :return: Masked attention logits, in the same dtype as ``attention_logits``.
        :rtype: keras.KerasTensor
        """
        # Expand mask to match attention shape
        if len(attention_mask.shape) == 2:
            # (batch, seq_len) -> (batch, 1, 1, seq_len)
            attention_mask = ops.expand_dims(ops.expand_dims(attention_mask, 1), 1)
        elif len(attention_mask.shape) == 3:
            # (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
            attention_mask = ops.expand_dims(attention_mask, 1)

        # The dtype the logits arrive in, captured before the helper's internal
        # promotion to `mask_dtype(...)`, so the return dtype of this method is
        # exactly what it always was.
        # `getattr(d, "name", None) or str(d)`, not `keras.backend.standardize_dtype`:
        # a Keras-2 residue banned across `src/`, and `str` alone mis-renders a
        # `tf.DType`. Full note and the measured equivalence at `common.py`; D-007.
        logits_dtype = getattr(attention_logits.dtype, "name", None) or str(attention_logits.dtype)

        # THIS SITE'S MASK POLARITY, passed through verbatim. The eight
        # multiply-form siblings take `1 = keep` floats, and `rpc_attention.py`
        # spells masking `mask == 0`. Here the mask is used directly as a boolean
        # keep predicate, with no comparison at all. `ops.cast(..., 'bool')` is
        # the identity for a bool mask and maps nonzero to True otherwise, which
        # is the rule `ops.where` applied before. Don't "normalize" this into a
        # `> 0` comparison. The shared helper infers no polarity, so an inversion
        # here raises nothing, changes no shape and stays finite: the layer would
        # just attend to the padding. `TestCapsuleRoutingMaskPolarity` is the only
        # guard that can see that.
        keep = ops.cast(attention_mask, "bool")

        # DECISION plan-2026-07-27T183600-b4ef45f0/D-006
        # The degenerate-row rescue: a query row that keeps NOTHING is treated as
        # keeping EVERYTHING, and it must stay in the PREDICATE. An all-False
        # mask row drives every logit in that row to MASK_BIAS_VALUE, which is
        # -inf in float16, and softmax over an all -inf row is 0/0 = NaN.
        # Measured on the unfixed code at (B=2, N=32, D=64, H=4, key_dim=16) with
        # one fully-masked query row: 128/4096 NaN under mixed_float16, against
        # 0/4096 in float32. Superseded in FORM only by D-008 and D-009: the
        # rescue used to be a local `logical_or` here and now lives in the shared
        # helper, whose `rescue_axis` DEFAULTS to -1, so the call below asks for
        # nothing. Passing `rescue_axis=None` opts out and brings the NaN back.
        #
        # WHAT NOT TO DO:
        #   * Don't drop the rescue and rely on dtype alone, the way
        #     `rpc_attention.py` does. The softmax here is
        #     `self.attn_prob_attention`, a Keras layer with autocasting on. A
        #     float32 tensor handed to it is seen inside its own call() as
        #     float16, and a fully-masked float32 -1e9 row still returns 8/8 NaN.
        #     Pinned by `TestCapsuleRoutingMaskHazardIsReal::
        #     test_the_probability_sublayer_autocasts_a_float32_input`.
        #   * Don't mask the NaN after the softmax. The forward pass looks clean
        #     while the unselected branch contributes 0 * NaN in the backward
        #     pass. Rescuing in the predicate never forms the NaN at all.
        #   * Don't reach for a per-dtype sentinel (-6e4 in fp16, as
        #     `lighthouse_attention.py` does). `common.py`'s docstring rules it
        #     out, and a row of equal finite sentinels is still uniform garbage.
        #
        # ACCEPTED SEMANTIC CHANGE, in every dtype: a fully-masked row used to
        # give a uniform distribution over all keys (float32) or NaN (fp16). It
        # now gives softmax over the unmasked logits. Rows that keep at least one
        # key are untouched, verified bit-identical in float32 for a padding and
        # a causal mask.
        #
        # The rescued axis is -1, the KEY axis of the already-broadcast
        # (B, H, Q, K) mask, which is the axis this site's softmax reduces over.
        # This is the one `probability_config`-carrying site in the package that
        # does not have to DERIVE that axis from the config (D-017). `__init__`'s
        # `_site_config` OVERRIDES any caller-supplied "axis" per probability
        # site and pins attn_prob_attention to -1, because the routing math needs
        # fixed axes. If that override is ever relaxed, this call MUST start
        # deriving its axis the way its seven siblings do.
        # See decisions.md D-006, D-008 and D-009.
        return apply_attention_mask(
            attention_logits, keep, out_dtype=logits_dtype
        )

    def _squash(self, vectors: keras.KerasTensor) -> keras.KerasTensor:
        """
        Squashing function from capsule networks.

        Applies the non-linearity: ``v = ||s||^2 / (1 + ||s||^2) * s / ||s||``.

        :param vectors: Input vectors to squash.
        :type vectors: keras.KerasTensor
        :return: Squashed vectors with same shape as input.
        :rtype: keras.KerasTensor
        """
        # Calculate squared norm along last axis
        squared_norm = ops.sum(ops.square(vectors), axis=-1, keepdims=True)
        norm = ops.sqrt(squared_norm + self.epsilon)

        # Apply squashing transformation
        scale = squared_norm / (1 + squared_norm)
        return scale * vectors / norm

    def _dynamic_routing(
        self,
        vote_vectors: keras.KerasTensor,
        num_output_capsules: int
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Apply dynamic routing algorithm between capsules.

        Implements the iterative routing-by-agreement algorithm that computes
        coupling coefficients based on agreement between prediction vectors
        and output capsules.

        :param vote_vectors: Vote vectors of shape
            ``(..., num_input, num_output, capsule_dim)``.
        :type vote_vectors: keras.KerasTensor
        :param num_output_capsules: Number of output capsules.
        :type num_output_capsules: int
        :return: Tuple of ``(output_capsules, routing_weights)`` where
            output_capsules are final capsule outputs after routing and
            routing_weights are final routing coefficients.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Get input dimensions for routing logits initialization
        input_shape = ops.shape(vote_vectors)
        batch_dims = input_shape[:-3]
        num_input_capsules = input_shape[-3]

        # Initialize routing logits to zero (uniform initial routing)
        # FIX: The original code used ops.concatenate, which fails if ops.shape
        # returns a Python tuple (e.g., in eager execution or with static shapes).
        # Using standard tuple concatenation is robust for shape construction.
        routing_logits_shape = batch_dims + (num_input_capsules, num_output_capsules)
        routing_logits = ops.zeros(shape=routing_logits_shape, dtype=vote_vectors.dtype)

        # Iterative routing algorithm
        for iteration in range(self.routing_iterations):
            # Compute coupling coefficients via the configured probability
            # function over input capsules (axis=-2).
            routing_weights = self.attn_prob_routing(routing_logits)

            # Expand routing weights for broadcasting with vote vectors
            routing_weights_expanded = ops.expand_dims(routing_weights, axis=-1)

            # Compute weighted sum of vote vectors (s_j = sum_i c_ij * u_j|i)
            weighted_votes = routing_weights_expanded * vote_vectors
            output_capsules = ops.sum(weighted_votes, axis=-3)

            # Apply squashing function to get final capsule outputs
            output_capsules = self._squash(output_capsules)

            # Update routing logits based on agreement (except on last iteration)
            if iteration < self.routing_iterations - 1:
                # Expand output capsules for agreement calculation
                output_expanded = ops.expand_dims(output_capsules, axis=-3)

                # Calculate agreement: dot product between votes and outputs
                agreement = ops.sum(vote_vectors * output_expanded, axis=-1)
                routing_logits = routing_logits + agreement

        return output_capsules, routing_weights

    def _vertical_routing(self, attention_weights: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply vertical (head-wise) capsule routing.

        Treats attention heads as capsules and applies dynamic routing to aggregate
        information across different attention perspectives for each query position.

        :param attention_weights: Attention weights of shape
            ``(batch, num_heads, seq_len, seq_len)``.
        :type attention_weights: keras.KerasTensor
        :return: Vertical routing output of same shape as input.
        :rtype: keras.KerasTensor
        """
        # Reshape for routing: treat each query position independently
        # (batch, num_heads, seq_len_q, seq_len_k) -> (batch, seq_len_q, num_heads, seq_len_k)
        attention_reshaped = ops.transpose(attention_weights, [0, 2, 1, 3])

        # Create vote vectors: each input head votes for each output head
        # Shape: (batch, seq_len_q, num_heads_in, num_heads_out, seq_len_k)
        vote_vectors = ops.expand_dims(attention_reshaped, axis=3)
        vote_vectors = ops.repeat(vote_vectors, self.num_heads, axis=3)

        # Apply dynamic routing over heads
        output_capsules, _ = self._dynamic_routing(vote_vectors, self.num_heads)
        # output_capsules shape: (batch, seq_len_q, num_heads_out, seq_len_k)

        # Apply learned aggregation weights if available
        if self.vertical_aggregation_weights is not None:
            # Transpose for matrix multiplication: (batch, seq_len_q, seq_len_k, num_heads)
            output_transposed = ops.transpose(output_capsules, [0, 1, 3, 2])

            # Apply linear transformation: (..., num_heads) @ (num_heads, num_heads)
            aggregated = ops.matmul(output_transposed, self.vertical_aggregation_weights)

            if self.vertical_aggregation_bias is not None:
                aggregated = aggregated + self.vertical_aggregation_bias

            # Apply the configured probability function to get importance
            # weights (axis=-1 over heads).
            importance_weights = self.attn_prob_aggregation(aggregated)

            # Weight the output capsules and transpose back
            weighted_output = importance_weights * output_transposed
            vertical_output = ops.transpose(weighted_output, [0, 3, 1, 2])
        else:
            # Reshape back to original attention format
            vertical_output = ops.transpose(output_capsules, [0, 2, 1, 3])

        return vertical_output

    def _horizontal_routing(self, attention_weights: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply horizontal (token-wise) capsule routing with optional positional constraints.

        Treats sequence tokens as capsules and applies dynamic routing to aggregate
        information across token positions, with optional causal masking to preserve
        sequential information flow.

        :param attention_weights: Attention weights of shape
            ``(batch, num_heads, seq_len, seq_len)``.
        :type attention_weights: keras.KerasTensor
        :return: Horizontal routing output of same shape as input.
        :rtype: keras.KerasTensor
        """
        # Use the STATIC sequence-length so `range(...)` unrolls at trace time
        # (graph-safe). `ops.shape(...)[2]` returns a symbolic tensor under
        # tf.function, and `range(symbolic_tensor)` raises TypeError.
        seq_len = attention_weights.shape[2]

        if self.use_positional_routing:
            # DECISION plan-2026-07-27T183600-b4ef45f0/D-014 — this guard belongs
            # INSIDE the positional branch. Don't hoist it out "so the failure is
            # earlier": only the `for l in range(seq_len)` unroll below needs a
            # static length. Accepted cost: a dynamic length with positional
            # routing off used to raise and now runs. See decisions.md D-014.
            if seq_len is None:
                raise ValueError(
                    "CapsuleRoutingSelfAttention positional routing "
                    "(use_horizontal_routing=True and "
                    "use_positional_routing=True) requires a statically-known "
                    "sequence length; got None. Build with a concrete seq_len, "
                    "or set use_positional_routing=False — the non-positional "
                    "horizontal path does not need a static length."
                )

            # Apply positional constraints: each position can only route from previous positions
            routed_rows = []

            for l in range(seq_len):
                if l == 0:
                    # First position: no routing needed
                    routed_row = attention_weights[:, :, :1, :]
                else:
                    # Extract attention for positions up to l (including l)
                    pos_attention = attention_weights[:, :, :l + 1, :]
                    # Shape: (batch, num_heads, l + 1, seq_len)

                    # Create vote vectors: each token's attention is a vote
                    vote_vectors = ops.expand_dims(pos_attention, axis=-2)
                    # Shape: (batch, num_heads, l + 1, 1, seq_len)

                    # Apply routing to aggregate information from tokens <= l
                    output_capsules, _ = self._dynamic_routing(vote_vectors, 1)
                    routed_row = output_capsules

                routed_rows.append(routed_row)

            # Reconstruct full attention matrix
            horizontal_output = ops.concatenate(routed_rows, axis=2)
        else:
            # Standard horizontal routing without positional constraints
            # Reshape: (batch, seq_len, num_heads, seq_len)
            attention_reshaped = ops.transpose(attention_weights, [0, 2, 1, 3])

            # Create vote vectors: (batch, seq_len, num_heads, num_heads, seq_len)
            vote_vectors = ops.expand_dims(attention_reshaped, axis=-2)
            vote_vectors = ops.repeat(vote_vectors, self.num_heads, axis=-2)

            # Apply routing
            output_capsules, _ = self._dynamic_routing(vote_vectors, self.num_heads)

            # Reshape back: (batch, num_heads, seq_len, seq_len)
            horizontal_output = ops.transpose(output_capsules, [0, 2, 1, 3])

        return horizontal_output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape, identical to input shape.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all initialization parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'value_dim': self.value_dim,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'routing_iterations': self.routing_iterations,
            'use_vertical_routing': self.use_vertical_routing,
            'use_horizontal_routing': self.use_horizontal_routing,
            'use_positional_routing': self.use_positional_routing,
            'epsilon': self.epsilon,
            'probability_type': self.probability_type,
            'probability_config': self.probability_config,
            'qk_norm_type': self.qk_norm_type,
            'qk_norm_kwargs': self.qk_norm_kwargs,
        })
        return config

# ---------------------------------------------------------------------
