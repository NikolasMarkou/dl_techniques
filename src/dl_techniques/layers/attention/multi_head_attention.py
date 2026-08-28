"""
Multi-head self-attention: pairwise relationships between elements of one sequence.

The mechanism answers a question a convolution or recurrence cannot ask directly:
for this element, which other elements matter, and how much? Each position emits
three projections — a Query stating what it is looking for, a Key advertising what
it offers, and a Value carrying what it would contribute. Compatibility is the dot
product of a Query with every Key, normalized into a distribution, and the output
is the correspondingly weighted sum of Values. Nothing in that computation refers
to distance, which is why a dependency between positions 1 and 1000 costs the same
as one between 1 and 2.

The multi-head part is what keeps that flexibility from collapsing. A single
attention distribution per position forces one notion of relevance, and the softmax
makes it competitive: attending to a syntactic governor means not attending to a
semantically similar word. Splitting the model dimension into ``h`` narrower
subspaces and attending independently in each lets a position hold several
relevance relations at once, and because the heads share the parameter budget
rather than adding to it, the cost is the concatenation and one output projection
that mixes what the heads found.

The ``1/sqrt(d_k)`` factor is the detail that makes the whole thing trainable. A dot
product of two ``d_k``-dimensional vectors has variance proportional to ``d_k``, so
without rescaling the logits grow with head width, the softmax saturates, and its
gradient vanishes exactly where the model is largest. Dividing by ``sqrt(d_k)``
holds logit variance constant, making head width a free architectural choice rather
than a stability constraint.

This module owns no attention math of its own. It is a thin, self-attention-shaped
facade over `MultiHeadCrossAttention`, invoked with `kv_input=None` and
`shared_qk_projections=True` so a single fused `Dense(3 * dim)` produces Q, K and V
— possible precisely because all three read the same tensor. `build` validates the
rank and forwards, `call` is one delegating expression, and
`compute_output_shape` is the identity. The sibling facade over the same engine is
`perceiver_attention.PerceiverAttention`, which presets the asymmetric
cross-attention configuration instead.

Foundational mathematics, with ``d_k = dim // num_heads``::

    Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) ) V

References:
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
    - Bahdanau et al., 2014. Neural Machine Translation by Jointly Learning to
      Align and Translate. (the additive attention this replaced)
      (https://arxiv.org/abs/1409.0473)
    - Michel et al., 2019. Are Sixteen Heads Really Better than One?. (what the
      individual heads turn out to contribute) (https://arxiv.org/abs/1905.10650)
"""

# ---------------------------------------------------------------------

import keras
from typing import Optional, Tuple, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .common import validate_head_divisibility
from .multi_head_cross_attention import MultiHeadCrossAttention

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MultiHeadAttention(keras.layers.Layer):
    """
    Multi-head self-attention, as a facade over the shared cross-attention engine.

    Provides a clean self-attention interface by wrapping the more general
    ``MultiHeadCrossAttention``, demonstrating the wrapper pattern for specialized
    interfaces while keeping robust serialization and one well-tested
    implementation. The computation is
    ``Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k)) V`` with Q, K and V all
    derived from the same input; ``num_heads`` parallel subspaces are attended
    independently, concatenated, and mixed by a final projection.

    ``shared_qk_projections=True`` is pinned rather than exposed: all three
    projections read the same tensor, so one fused ``Dense(3 * dim)`` is both
    correct and cheaper than three.

    **[REUSE] This class contains no attention arithmetic.** Every projection,
    score, mask application, probability normalization and output projection
    happens in the single ``MultiHeadCrossAttention`` sub-layer created in
    ``__init__``, invoked with ``kv_input=None`` (self-attention) and
    ``shared_qk_projections=True`` (one fused QKV ``Dense`` instead of three).
    ``build()`` only validates the input rank and forwards; ``call()`` is a
    one-expression delegation; ``compute_output_shape()`` is the identity.

    WHAT NOT TO DO: don't inline a copy of the QKV/score/softmax pipeline here
    for clarity, or to remove a layer of indirection. Two copies of scaled
    dot-product attention drift apart - the sibling already carries QK-norm,
    ``ProbabilityOutput`` strategies and a mask-broadcast helper. And every
    ``.keras`` checkpoint of this layer stores the nested ``cross_attention``
    sub-layer's weights under that name, so flattening the wrapper is a silent
    checkpoint break, not a refactor.

    **Architecture Overview:**

    .. code-block:: text

                    inputs  [B, seq, dim]
                             │
                             ▼
          ┌───────────────────────────────────────┐
          │ cross_attention                       │
          │   MultiHeadCrossAttention, built in   │
          │   __init__ with                       │
          │     shared_qk_projections=True        │
          │   called with kv_input=None           │
          │                                       │
          │   ONE fused Dense(3 * dim) -> Q, K, V │
          │   split into num_heads of width d_k   │
          │   optional QK-norm                    │
          │   scores * 1/sqrt(d_k)                │
          │   attention_mask, probability, dropout│
          │   weighted sum of V, merge heads      │
          │   output projection back to dim       │
          └──────────────────┬────────────────────┘
                             ▼
                    output  [B, seq, dim]

        Every stage in that box belongs to MultiHeadCrossAttention. Read
        its diagram for the internals; this class adds no arithmetic.
        Don't flatten the sub-layer away: the name cross_attention is
        baked into every saved .keras checkpoint of this class.

    **Why the heads are split rather than stacked:**

    .. code-block:: text

        one head, width dim     one notion of relevance. The softmax
                                makes it competitive, so a position must
                                choose.

        h heads, width dim/h    h relations held at once, same parameter
                                budget, mixed by the output projection.

        d_k = dim // num_heads. The 1/sqrt(d_k) factor holds logit
        variance constant, so head width does not saturate the softmax.

    :param dim: Integer, dimension of input embeddings. Must be positive
        and divisible by num_heads.
    :type dim: int
    :param num_heads: Integer, number of attention heads. Must be positive.
        Defaults to 8.
    :type num_heads: int
    :param dropout_rate: Float, dropout rate for attention weights. Must be between
        0.0 and 1.0. Defaults to 0.0.
    :type dropout_rate: float
    :param output_kernel_initializer: Optional initializer for the output
        projection (``cross_attention/proj``) alone; ``None`` (the default)
        leaves it on ``kernel_initializer``. See
        :class:`MultiHeadCrossAttention`.
    :type output_kernel_initializer: Optional[Union[str, keras.initializers.Initializer]]
    :param kernel_initializer: String or Initializer for weight matrices.
        Defaults to "he_normal".
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for weight matrices.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param use_bias: Boolean, whether to use bias in dense layers.
        Defaults to False.
    :type use_bias: bool
    :param probability_type: String identifier for the attention-score
        normalization strategy, forwarded unchanged to the wrapped
        ``MultiHeadCrossAttention`` (and from there to
        :class:`ProbabilityOutput`). One of ``"softmax"``, ``"sparsemax"``,
        ``"threshmax"``, ``"adaptive"`` and their aliases. Defaults to
        ``"softmax"``. ``"routing"``/``"hierarchical"`` are rejected by the
        wrapped layer, so they raise from this constructor too.
    :type probability_type: str
    :param probability_config: Optional dictionary forwarded to the
        :class:`ProbabilityOutput` strategy as ``type_config`` (e.g.
        ``min_temp``/``max_temp`` for ``"adaptive"``). Defaults to ``None``.
    :type probability_config: Optional[Dict[str, Any]]
    :param qk_norm_type: Optional normalization type applied to the Q and K
        projections before scoring (QK-norm), forwarded to
        :func:`create_normalization_layer` by the wrapped layer. ``None``
        disables QK-norm. Defaults to ``None``.
    :type qk_norm_type: Optional[str]
    :param qk_norm_kwargs: Optional keyword arguments forwarded to
        :func:`create_normalization_layer` when the Q/K norms are constructed.
        Defaults to ``None``.
    :type qk_norm_kwargs: Optional[Dict[str, Any]]
    :param kwargs: Additional layer arguments.

    :raises ValueError: If dim is not divisible by num_heads.
    :raises ValueError: If parameters are invalid (negative values, etc.).
    :raises ValueError: From ``build()``, if the input is not 3D or its trailing
        dimension does not equal ``dim``.

    Input shape:
        3D tensor with shape ``(batch_size, seq_len, dim)``. The optional
        ``attention_mask`` is a ``1 = keep`` predicate of shape
        ``(batch_size, seq_len)``, ``(batch_size, seq_len, seq_len)`` or
        ``(batch_size, num_heads, seq_len, seq_len)``.

    Output shape:
        3D tensor with shape ``(batch_size, seq_len, dim)`` — unchanged from the
        input. One output mode only; attention weights are never returned.

    Example:
        >>> attn = MultiHeadAttention(dim=512, num_heads=8)
        >>> x = keras.random.normal((2, 128, 512))
        >>> y = attn(x, training=False)                   # (2, 128, 512)
        >>>
        >>> # Causal or padding mask, keep-predicate semantics
        >>> mask = keras.ops.ones((2, 128, 128))
        >>> y = attn(x, attention_mask=mask, training=False)
        >>>
        >>> # GPT-2's residual-path init rule on the output projection only
        >>> attn = MultiHeadAttention(
        ...     dim=768, num_heads=12,
        ...     output_kernel_initializer=keras.initializers.RandomNormal(
        ...         stddev=0.02 / (2 * 12) ** 0.5),
        ... )

    Note:
        This layer is the self-attention preset of ``MultiHeadCrossAttention``, so
        every masking, dtype and normalization subtlety documented there applies
        here unchanged — including the fully-masked-row rescue. Read that class's
        anchors, not this file, when reasoning about the numerics.

    Attributes:
        cross_attention: The single ``MultiHeadCrossAttention`` sub-layer, named
            ``cross_attention``. That name is part of the checkpoint format.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout_rate: float = 0.0,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        output_kernel_initializer: Optional[Union[str, keras.initializers.Initializer]] = None,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = False,
        probability_type: str = "softmax",
        probability_config: Optional[Dict[str, Any]] = None,
        qk_norm_type: Optional[str] = None,
        qk_norm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Validate the cheap invariants and create the wrapped attention engine.

        Every argument is stored and forwarded verbatim. The only values this
        class supplies itself are ``shared_qk_projections=True`` and
        ``bias_initializer="zeros"``. The check ORDER below is load-carrying and
        is explained in the comment beside it. See the class docstring for the
        parameter reference.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        # Adopts the shared validator. Its message is character-for-character
        # what stood here - `dim (63) must be divisible by num_heads (8)` - so the
        # regex in
        # `test_multi_head_attention.py::test_invalid_dim_not_divisible` still
        # matches and no diagnostic detail is lost.
        #
        # Don't move this below the `num_heads <= 0` guard. It has always run
        # first, Python's `%` on a negative modulus does not raise, and callers
        # have seen this ordering.
        validate_head_divisibility(dim, num_heads)
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.output_kernel_initializer = (
            keras.initializers.get(output_kernel_initializer)
            if output_kernel_initializer is not None else None
        )
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs

        # The engine. Its name is part of the checkpoint format.
        self.cross_attention = MultiHeadCrossAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            # One fused Dense(3 * dim): all three projections read the same
            # tensor, so sharing is both correct and cheaper.
            shared_qk_projections=True,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            output_kernel_initializer=self.output_kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer="zeros",
            probability_type=self.probability_type,
            probability_config=self.probability_config,
            qk_norm_type=self.qk_norm_type,
            qk_norm_kwargs=self.qk_norm_kwargs,
            name="cross_attention"
        )

    def build(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> None:
        """Validate the input rank and build the wrapped engine.

        A single shape is forwarded as-is: the engine reads it as self-attention
        and uses it for both the query and key/value roles. Building explicitly is
        what guarantees every weight variable exists before weight restoration.

        :param input_shape: Shape tuple of the input tensor, expected as
            ``(batch_size, seq_len, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not rank 3, or if its last
            dimension does not equal ``dim``.
        """
        if self.built:
            return

        # Validate input shape
        if isinstance(input_shape, list):
            input_shape = tuple(input_shape)

        if len(input_shape) != 3:
            raise ValueError(f"Input must be 3D (batch, seq_len, dim), got shape {input_shape}")
        if input_shape[-1] != self.dim:
            raise ValueError(f"Input last dimension ({input_shape[-1]}) must match dim ({self.dim})")

        # Build the wrapped cross-attention layer explicitly for serialization
        self.cross_attention.build(tuple(input_shape))

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass by delegating to the wrapped engine in self-attention mode.

        One expression, no arithmetic: ``kv_input=None`` selects the engine's
        self-attention path, where the fused projection supplies all three of Q, K
        and V.

        :param inputs: Input tensor of shape ``(batch_size, seq_len, dim)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional attention mask tensor. Supported shapes:
            ``(batch_size, seq_len)``, ``(batch_size, seq_len, seq_len)``, or
            ``(batch_size, num_heads, seq_len, seq_len)``. Values of 1 indicate
            positions to attend to, 0 for masked positions.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Boolean indicating whether in training mode.
        :type training: Optional[bool]
        :return: Attention output tensor of shape ``(batch_size, seq_len, dim)``.
        :rtype: keras.KerasTensor
        """
        # Shape: (B, seq, dim) -> (B, seq, dim).
        # Pure delegation. No arithmetic happens in this frame. See the [REUSE]
        # note on the class docstring.
        #
        # kv_input=None is what selects the engine's self-attention path.
        return self.cross_attention(
            query_input=inputs,
            kv_input=None,
            attention_mask=attention_mask,
            training=training
        )

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        For self-attention query and key/value lengths coincide, and the output
        projection maps back to ``dim``, so this is the identity.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization, includes all constructor parameters.

        The wrapped engine is reconstructed from these values in ``__init__``
        rather than serialized as a nested layer, which is what keeps the
        ``cross_attention`` sub-layer name stable across round trips.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "output_kernel_initializer": (
                keras.initializers.serialize(self.output_kernel_initializer)
                if self.output_kernel_initializer is not None else None
            ),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
            "probability_type": self.probability_type,
            "probability_config": self.probability_config,
            "qk_norm_type": self.qk_norm_type,
            "qk_norm_kwargs": self.qk_norm_kwargs,
        })
        return config

# ---------------------------------------------------------------------