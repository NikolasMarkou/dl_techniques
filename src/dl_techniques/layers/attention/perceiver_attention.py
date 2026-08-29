"""
Asymmetric cross-attention: the Perceiver's information bottleneck.

A transformer's cost is quadratic in its input length, which means the input
itself decides how deep a model can afford to be. For an image or an audio
waveform — tens of thousands of positions — that ceiling arrives immediately, and
the usual response is to shrink the input first with a modality-specific frontend:
patches for images, filterbanks for audio, tokenizers for text.

The Perceiver's answer is to decouple the two quantities instead. A small,
fixed-size latent array supplies the queries; the large data array supplies the
keys and values. The attention matrix is then ``(N, M)`` rather than ``(M, M)``,
costing ``O(N * M)`` with ``N << M``, and — the part that matters — every
self-attention block stacked afterwards operates on the latents alone, at
``O(N^2)``, no matter how large ``M`` was. Depth becomes free of input size, so the
same architecture accepts any modality without a bespoke frontend: the bottleneck
does the reduction, and it learns to.

The asymmetry has a consequence for the weights. Queries and keys/values read
DIFFERENT tensors — often of different modalities — so their projections cannot be
fused or shared, which is why this layer pins `shared_qk_projections=False` in the
wrapped engine rather than exposing it.

This module owns no attention arithmetic. It is a thin, cross-attention-shaped
facade over `MultiHeadCrossAttention`: every projection, score, mask application,
normalization and output projection happens in that one sub-layer. `build` only
disambiguates and validates shapes, and `call` is a single delegating expression.
The sibling facade over the same engine is `multi_head_attention.MultiHeadAttention`,
which presets the self-attention configuration instead.

Foundational mathematics, with a latent array ``X_lat`` of ``N`` rows and a data
array ``X_data`` of ``M`` rows::

    Q = X_lat  W_q,   K = X_data W_k,   V = X_data W_v
    Output    = softmax( Q K^T / sqrt(d_k) ) V

References:
    - Jaegle et al., 2021. Perceiver: General Perception with Iterative Attention.
      ICML. (https://arxiv.org/abs/2103.03206)
    - Jaegle et al., 2021. Perceiver IO: A General Architecture for Structured
      Inputs & Outputs. ICLR 2022. (https://arxiv.org/abs/2107.14795)
    - Vaswani et al., 2017. Attention Is All You Need. (the cross-attention this
      specializes) (https://arxiv.org/abs/1706.03762)
    - Lee et al., 2019. Set Transformer: A Framework for Attention-based
      Permutation-Invariant Neural Networks. (inducing points — the same
      fixed-size-bottleneck idea) (https://arxiv.org/abs/1810.00825)
"""

# ---------------------------------------------------------------------

import keras
from typing import Optional, Any, Dict, Tuple, Union, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .common import validate_head_divisibility
from .multi_head_cross_attention import MultiHeadCrossAttention
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


def _is_list_of_shapes(s: Any) -> bool:
    """Return whether ``s`` is a container of SHAPES rather than a single shape.

    This module's one shape-classification predicate. It sits at module level
    rather than inside ``build`` so that :meth:`PerceiverAttention.build` and
    :meth:`PerceiverAttention.compute_output_shape` cannot classify the same
    argument differently - both call this function and nothing else. Two
    separate copies of the rule disagreeing is the defect it exists to
    prevent, so don't inline it into either caller.

    Note what it accepts: any list OR tuple whose first element is itself a
    list or tuple. A tuple of two shape tuples is a shape PAIR here, not a
    single shape.

    :param s: A shape, or a container of shapes.
    :type s: Any

    :return: ``True`` if ``s``'s first element is itself a shape (list/tuple),
        i.e. ``s`` describes MULTIPLE inputs; ``False`` if ``s`` is one shape.
    :rtype: bool
    """
    return (
        isinstance(s, (list, tuple))
        and len(s) > 0
        and isinstance(s[0], (list, tuple))
    )


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.attention.perceiver_attention")
class PerceiverAttention(keras.layers.Layer):
    """
    Perceiver-style asymmetric cross-attention, as a facade over the shared engine.

    Cross-attention where queries and key-value pairs come from different sources.
    A small, fixed-size latent array forms queries that attend to a large data
    array providing keys and values: ``Q = X_lat W_q``, ``K = X_data W_k``,
    ``V = X_data W_v``, ``Output = softmax(Q K^T / sqrt(d_k)) V``. Cost is
    ``O(N * M)`` rather than ``O(M^2)``, and because ``N`` is fixed by the
    architecture, any depth stacked after the bottleneck is independent of ``M``.

    ``shared_qk_projections=False`` is pinned rather than exposed: the query path
    and the key/value path read different tensors, so their projections must be
    independent.

    **[REUSE] This class contains no attention arithmetic.** Every projection,
    score, mask application, probability normalization and output projection is
    performed by the single ``MultiHeadCrossAttention`` sub-layer created in
    ``__init__``, with ``shared_qk_projections=False`` so the query path and the
    key/value path get independent weights (required — they see different
    tensors). ``build()`` only disambiguates and validates shapes; ``call()`` is a
    one-expression delegation.

    Its sibling facade over the same engine is
    ``multi_head_attention.MultiHeadAttention``, which presets the
    self-attention configuration instead.

    WHAT NOT TO DO: do not inline the QKV/score/softmax pipeline here. The
    wrapper's nested ``cross_attention`` sub-layer name is baked into every
    saved ``.keras`` checkpoint of this layer, so flattening it is a silent
    checkpoint break rather than a refactor.

    **Architecture Overview:**

    .. code-block:: text

          ┌─────────────────────┐   ┌─────────────────────┐
          │ query_input         │   │ kv_input            │
          │   the latent array  │   │   the data array    │
          │   [B, N, D]         │   │   [B, M, D]         │
          │   N is FIXED by     │   │   M varies with     │
          │   the architecture  │   │   the input         │
          └──────────┬──────────┘   └──────────┬──────────┘
                     └────────────┬────────────┘
                                  ▼
          ┌───────────────────────────────────────┐
          │ cross_attention                       │
          │   MultiHeadCrossAttention, built in   │
          │   __init__ with                       │
          │     shared_qk_projections=False       │
          │   so the query path and the key/value │
          │   path get INDEPENDENT weights - they │
          │   read different tensors              │
          │                                       │
          │   Q from query_input, K and V from    │
          │   kv_input, QK-norm, scaled scores,   │
          │   mask, probability, dropout,         │
          │   weighted sum of V, merge heads,     │
          │   output projection                   │
          └──────────────────┬────────────────────┘
                             ▼
                   output  [B, N, D]
                   QUERY-shaped, not data-shaped

        kv_input=None falls through to self-attention in the engine.
        build() only disambiguates and validates shapes. Don't flatten
        the sub-layer away: the name cross_attention is baked into every
        saved .keras checkpoint of this class.

    **Why the asymmetry pays:**

    .. code-block:: text

                                attention cost   depth per block
        self-attention on data  O(M^2)           O(M^2)
        Perceiver bottleneck    O(N * M)         O(N^2)  <- no M

        With N << M the data array is read ONCE, and every self-
        attention block after the bottleneck is priced on N alone. That
        is what lets one architecture take images, audio or point
        clouds with no modality-specific frontend.

    :param dim: Input/output dimension. Must be positive and divisible by num_heads.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param dropout_rate: Dropout rate for attention weights, between 0.0 and 1.0.
    :type dropout_rate: float
    :param use_bias: Whether to use bias in linear projections.
    :type use_bias: bool
    :param kernel_initializer: Initializer for kernel weights.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias vectors.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param probability_type: String identifier for the attention-score
        normalization strategy, forwarded unchanged to the wrapped
        ``MultiHeadCrossAttention``. One of ``"softmax"``, ``"sparsemax"``,
        ``"threshmax"``, ``"adaptive"`` and their aliases. Defaults to
        ``"softmax"``. Routing/hierarchical variants are rejected by the wrapped
        layer and therefore raise from this constructor too.
    :type probability_type: str
    :param probability_config: Optional dictionary forwarded to the
        :class:`ProbabilityOutput` strategy as ``type_config``.
        Defaults to ``None``.
    :type probability_config: Optional[Dict[str, Any]]
    :param qk_norm_type: Optional normalization type applied to Q and K before
        scoring (QK-norm), forwarded to :func:`create_normalization_layer` by the
        wrapped layer. ``None`` disables QK-norm. Defaults to ``None``.
    :type qk_norm_type: Optional[str]
    :param qk_norm_kwargs: Optional keyword arguments forwarded to
        :func:`create_normalization_layer` for the Q/K norms. Defaults to ``None``.
    :type qk_norm_kwargs: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any

    :raises ValueError: If dim is not divisible by num_heads.
    :raises ValueError: If input shapes are invalid.
    :raises ValueError: If parameters are out of valid ranges.

    Input shape:
        - Cross-attention: ``query_input`` ``(batch, N, dim)`` and ``kv_input``
          ``(batch, M, dim)``; ``build`` accepts the pair as a list of two shapes.
          Unlike the wrapped engine, BOTH shapes are validated here — rank 3 and a
          trailing dimension equal to ``dim``.
        - Self-attention: a single 3D shape, used for both roles.

    Output shape:
        3D tensor ``(batch, N, dim)`` — the QUERY's sequence length. The data
        array's length ``M`` never appears in the output; that is the bottleneck.

    Example:
        >>> # 256 latents read a 50k-position input
        >>> attn = PerceiverAttention(dim=512, num_heads=8)
        >>> latents = keras.random.normal((2, 256, 512))
        >>> data = keras.random.normal((2, 50176, 512))
        >>> out = attn(latents, kv_input=data)            # (2, 256, 512)
        >>>
        >>> # Latent self-attention: the same layer, no kv_input
        >>> out = attn(latents)                           # (2, 256, 512)
        >>>
        >>> # Build explicitly with the shape pair
        >>> attn.build([(None, 256, 512), (None, 50176, 512)])

    Note:
        This layer is the cross-attention preset of ``MultiHeadCrossAttention``,
        so every masking, dtype and normalization subtlety documented there
        applies here unchanged — including the fully-masked-row rescue. Read that
        class's anchors, not this file, when reasoning about the numerics.

    Attributes:
        cross_attention: The single ``MultiHeadCrossAttention`` sub-layer, named
            ``cross_attention``. That name is part of the checkpoint format.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            dropout_rate: float = 0.0,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            probability_type: str = "softmax",
            probability_config: Optional[Dict[str, Any]] = None,
            qk_norm_type: Optional[str] = None,
            qk_norm_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        """Validate the cheap invariants and create the wrapped attention engine.

        Every argument is stored and then forwarded verbatim; the only value this
        class supplies itself is ``shared_qk_projections=False``. See the class
        docstring for the parameter reference.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        # Adopts the shared validator. Its message is character-for-character
        # what stood here, so the `match="must be divisible"` regex in
        # `test_perceiver_attention.py::test_invalid_divisibility` still matches
        # and no diagnostic detail is lost. Its position in the validation
        # sequence is unchanged.
        validate_head_divisibility(dim, num_heads)
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs

        # The engine. Its name is part of the checkpoint format.
        self.cross_attention = MultiHeadCrossAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            # Independent Q and K/V projections. They read different tensors,
            # often of different modalities, so they cannot be fused.
            shared_qk_projections=False,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            probability_type=self.probability_type,
            probability_config=self.probability_config,
            qk_norm_type=self.qk_norm_type,
            qk_norm_kwargs=self.qk_norm_kwargs,
            name="cross_attention"
        )

    def build(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> None:
        """Disambiguate the shape argument, validate BOTH shapes, build the sub-layer.

        Classification goes through the module-level :func:`_is_list_of_shapes` so
        this method and :meth:`compute_output_shape` cannot disagree — the two
        disagreeing is the defect that predicate exists to prevent. Both the query
        and key/value shapes are checked here, which is stricter than the wrapped
        engine's own ``build``.

        :param input_shape: Shape of input tensor(s). Can be a single shape tuple
            or a list of two shape tuples for query and kv inputs.
        :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        :raises ValueError: If a shape container has a length other than 2, if
            either shape is not rank 3, or if either trailing dimension does not
            equal ``dim``.
        """
        if self.built:
            return

        # One shape, or a pair? The module-level predicate decides, and
        # compute_output_shape asks the same question of the same function.
        if _is_list_of_shapes(input_shape):
            # A pair: cross-attention.
            if len(input_shape) != 2:
                raise ValueError(f"Expected 2 inputs for cross-attention, got {len(input_shape)}")
            query_shape, kv_shape = input_shape
        else:
            # One shape: self-attention, used for both roles.
            query_shape = kv_shape = input_shape

        # Both shapes are checked here, which is stricter than the engine's own
        # build().
        if len(query_shape) != 3:
            raise ValueError(f"Query input must be 3D, got shape {query_shape}")
        if len(kv_shape) != 3:
            raise ValueError(f"KV input must be 3D, got shape {kv_shape}")
        if query_shape[-1] != self.dim:
            raise ValueError(f"Query last dimension ({query_shape[-1]}) must match dim ({self.dim})")
        if kv_shape[-1] != self.dim:
            raise ValueError(f"KV last dimension ({kv_shape[-1]}) must match dim ({self.dim})")

        # Build the engine explicitly so every weight variable exists before
        # weight restoration.
        self.cross_attention.build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            query_input: keras.KerasTensor,
            kv_input: Optional[keras.KerasTensor] = None,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply Perceiver cross-attention by delegating to the wrapped engine.

        One expression, no arithmetic. ``kv_input=None`` falls through to the
        engine's self-attention path.

        :param query_input: Query tensor of shape ``(batch_size, query_seq_len, dim)``.
        :type query_input: keras.KerasTensor
        :param kv_input: Key-Value tensor of shape ``(batch_size, kv_seq_len, dim)``.
            If ``None``, uses query_input for self-attention mode.
        :type kv_input: Optional[keras.KerasTensor]
        :param attention_mask: Optional attention mask of shape
            ``(batch_size, seq_len, seq_len)`` or ``(batch_size, 1, seq_len, seq_len)``.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]

        :return: Output tensor with same shape as query_input.
        :rtype: keras.KerasTensor
        """
        return self.cross_attention(
            query_input=query_input,
            # None here selects the engine's self-attention path.
            kv_input=kv_input,
            attention_mask=attention_mask,
            training=training
        )

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> Tuple[Optional[int], ...]:
        """Return the QUERY's shape, which is the output's.

        For a shape pair that is the FIRST entry: the data array's length is
        consumed by the bottleneck and never reaches the output.

        :param input_shape: Input shape(s), either single or list of two shapes.
        :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]

        :return: Output shape tuple, same as query input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        if _is_list_of_shapes(input_shape):
            # Entry 0 is the query shape, which is the output shape.
            return tuple(input_shape[0])
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        The wrapped engine is reconstructed from these values in ``__init__``
        rather than serialized as a nested layer, which is what keeps the
        ``cross_attention`` sub-layer name stable across round trips.

        :return: Dictionary containing all layer configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
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
