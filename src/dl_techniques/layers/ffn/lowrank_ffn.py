"""
A low-rank factorized Feed-Forward Network.

This layer has the same shape as the standard Transformer MLP
("expand-then-contract"), but each of its two dense projections is factorized.
Instead of one full-rank weight matrix `W` of shape `(d_in, d_out)`, the layer
learns `U` of shape `(d_in, rank)` and `V` of shape `(rank, d_out)` and uses
their product `U @ V` in place of `W`. When `rank << min(d_in, d_out)` this
cuts both the parameter count and the compute of the projection.

**Architecture Overview:**
The expand/contract shape is unchanged. Each projection becomes a bottleneck:

1.  Expansion (factorized). The input `(..., input_dim)` reaches `hidden_dim`
    through a bottleneck of width `rank`:
    `U1: Dense(rank, no bias) -> V1: Dense(hidden_dim)`.

2.  Non-linear activation. A configurable activation (default GELU) is applied
    element-wise, followed by dropout.

3.  Contraction (factorized). The activated tensor returns to `output_dim`
    through a second bottleneck of the same width `rank`:
    `U2: Dense(rank, no bias) -> V2: Dense(output_dim)`.

The `U` projections carry no bias. A bias there is redundant: the next linear
map `V` consumes it immediately, so it can be folded into `V`'s own bias. Only
the `V` projections carry the optional bias.

**Mathematics:**
For an input vector `x` at a single position the layer computes:

`FFN(x) = V_2(U_2(activation(V_1(U_1(x)))))`

(with biases on the `V` maps when `use_bias=True`). A dense FFN costs
`input_dim*hidden_dim + hidden_dim*output_dim` kernel parameters. The low-rank
form costs `rank*(input_dim + hidden_dim) + rank*(hidden_dim + output_dim)`.
The expansion is cheaper whenever
`rank < (input_dim*hidden_dim)/(input_dim + hidden_dim)`; the contraction
follows the same rule with `hidden_dim` and `output_dim` in place of
`input_dim` and `hidden_dim`.

References:
-   Vaswani, A., et al. (2017). Attention Is All You Need. NIPS. (the base
    FFN structure this layer specializes).
-   Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language
    Models. arXiv:2106.09685. (the low-rank factorization principle, used
    here as the layer's core structure rather than as an adapter).

"""

import keras
from typing import Optional, Union, Any, Dict, Tuple, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class LowRankFFN(keras.layers.Layer):
    """
    Low-rank factorized feed-forward network.

    An expand/contract MLP in which each dense projection is replaced by a
    low-rank product ``Dense(rank, use_bias=False) -> Dense(out)``. The
    computation is ``FFN(x) = v2(u2(activation(v1(u1(x)))))``, applied to every
    token position with the same weights. When ``rank`` is small relative to
    the layer widths, this costs far fewer kernel parameters than a dense MLP
    with the same hidden and output widths.

    **Architecture Overview:**

    .. code-block:: text

        Input  [..., input_dim]
                      │
                      ▼
        ┌───────────────────────────┐
        │ u1: Dense(rank, no bias)  │
        └─────────────┬─────────────┘
                      ▼  [..., rank]
        ┌───────────────────────────┐
        │ v1: Dense(hidden_dim)     │
        └─────────────┬─────────────┘
                      ▼  [..., hidden_dim]
        ┌───────────────────────────┐
        │ activation  (default GELU)│
        └─────────────┬─────────────┘
                      ▼
        ┌───────────────────────────┐
        │ dropout  (no-op at rate 0)│
        └─────────────┬─────────────┘
                      ▼
        ┌───────────────────────────┐
        │ u2: Dense(rank, no bias)  │
        └─────────────┬─────────────┘
                      ▼  [..., rank]
        ┌───────────────────────────┐
        │ v2: Dense(output_dim)     │
        └─────────────┬─────────────┘
                      ▼
        Output [..., output_dim]

        Dropout is always in the graph. At dropout_rate=0.0
        it is a no-op, not absent.

    **One factorized projection (the rank bottleneck):**

    .. code-block:: text

           x  [..., d_in]
                 │
                 ▼
           U: Dense(rank, use_bias=False)
                 │        kernel (d_in, rank)
                 ▼  [..., rank]
           V: Dense(d_out, use_bias=use_bias)
                 │        kernel (rank, d_out)
                 ▼
           y  [..., d_out]

        U never carries a bias. A bias on the bottleneck would
        be multiplied by V and added to V's own bias, so it
        would add parameters without adding any function the
        pair cannot already express.

        Kernel parameters for one projection:

          dense     d_in * d_out
          low-rank  rank * (d_in + d_out)

        The layer runs this twice with one shared rank:
        input_dim -> hidden_dim, then hidden_dim -> output_dim.
        rank defaults to max(1, hidden_dim // 4).

    :param hidden_dim: Hidden (expansion) width. Must be positive.
    :type hidden_dim: int
    :param output_dim: Output (projection) width. Must be positive.
    :type output_dim: int
    :param rank: Bottleneck width shared by both factorized projections. When
        ``None`` it resolves at construction time to
        ``max(1, hidden_dim // 4)``. When given it must be positive. The
        as-passed value (possibly ``None``) is what ``get_config()`` stores, so
        a round trip re-runs the same resolution.
    :type rank: Optional[int]
    :param activation: Activation applied after the expansion. Accepts a name
        ('gelu', 'relu', 'swish') or a callable. Defaults to 'gelu'.
    :type activation: Union[str, Callable]
    :param dropout_rate: Dropout rate applied after the activation. Must be in
        ``[0.0, 1.0)``. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether the ``v1`` / ``v2`` projections carry a bias. The
        ``u1`` / ``u2`` bottlenecks never do. Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for all four Dense kernels. One
        resolved instance is shared by all four layers.
        Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the ``v1`` / ``v2`` biases.
        Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for all four Dense kernels.
        Defaults to None.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Regularizer for the ``v1`` / ``v2`` biases.
        Defaults to None.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Extra arguments for ``keras.layers.Layer`` (``name``,
        ``dtype``, and so on).
    :type kwargs: Any

    :ivar hidden_dim: The stored hidden width.
    :vartype hidden_dim: int
    :ivar output_dim: The stored output width.
    :vartype output_dim: int
    :ivar rank: The RESOLVED bottleneck width, always a positive int.
    :vartype rank: int
    :ivar _rank_arg: The rank as REQUESTED, possibly ``None``. This is what
        ``get_config()`` stores.
    :vartype _rank_arg: Optional[int]
    :ivar activation_name: The activation argument exactly as passed. Kept for
        inspection only; ``get_config()`` serializes ``activation_fn`` instead.
    :vartype activation_name: Union[str, Callable]
    :ivar activation_fn: The resolved activation callable.
    :vartype activation_fn: Callable
    :ivar dropout_rate: The stored dropout rate.
    :vartype dropout_rate: float
    :ivar use_bias: Whether ``v1`` and ``v2`` carry a bias.
    :vartype use_bias: bool
    :ivar kernel_initializer: The resolved kernel initializer.
    :vartype kernel_initializer: keras.initializers.Initializer
    :ivar bias_initializer: The resolved bias initializer.
    :vartype bias_initializer: keras.initializers.Initializer
    :ivar kernel_regularizer: The resolved kernel regularizer, or ``None``.
    :vartype kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar bias_regularizer: The resolved bias regularizer, or ``None``.
    :vartype bias_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar u1: ``Dense(rank, use_bias=False)``, the expansion bottleneck.
    :vartype u1: keras.layers.Dense
    :ivar v1: ``Dense(hidden_dim)``, the expansion output.
    :vartype v1: keras.layers.Dense
    :ivar u2: ``Dense(rank, use_bias=False)``, the contraction bottleneck.
    :vartype u2: keras.layers.Dense
    :ivar v2: ``Dense(output_dim)``, the contraction output.
    :vartype v2: keras.layers.Dense
    :ivar dropout: ``Dropout(dropout_rate)``. Always present, even at rate 0.0.
    :vartype dropout: keras.layers.Dropout

    :raises ValueError: If ``hidden_dim`` or ``output_dim`` is not positive.
    :raises ValueError: If ``rank`` is given and is not positive.
    :raises ValueError: If ``dropout_rate`` is outside ``[0.0, 1.0)``.

    Input shape:
        Tensor of rank >= 2, shape ``(..., input_dim)``. The input width is
        free; it only has to stay the same across calls once built.

    Output shape:
        Same rank and leading axes as the input, with the last axis set to
        ``output_dim``.

    Example:
        .. code-block:: python

            ffn = LowRankFFN(hidden_dim=1024, output_dim=256)
            ffn.rank                # 256, from max(1, 1024 // 4)
            y = ffn(keras.random.normal((2, 10, 512)))
            y.shape                 # (2, 10, 256)

    Note:
        ``self.rank`` holds the resolved integer used to size the bottlenecks.
        ``get_config()`` emits the original ``rank`` argument, which may be
        ``None``, so deserialization repeats the same resolution instead of
        pinning a width the caller never asked for.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        rank: Optional[int] = None,
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = "gelu",
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        """Validate the configuration and create the four Dense projections.

        Every argument is documented on the class. Validation runs before any
        attribute is stored, so a rejected configuration leaves no half-built
        layer behind. ``rank`` is resolved here, not in ``build()``, because
        the two bottleneck Dense layers need its value at creation time.

        :raises ValueError: If ``hidden_dim`` or ``output_dim`` is not
            positive, if ``rank`` is given and not positive, or if
            ``dropout_rate`` is outside ``[0.0, 1.0)``.
        """
        super().__init__(**kwargs)

        # Reject bad configuration before storing anything.
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if rank is not None and rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")

        # Store every constructor argument; get_config() returns all of them.
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Preserve the as-passed rank argument (possibly None) for round-trip
        # serialization; resolve a concrete int for sizing the bottleneck.
        self._rank_arg = rank
        self.rank = rank if rank is not None else max(1, hidden_dim // 4)
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Resolve activation once.
        self.activation_fn = keras.activations.get(activation)

        # Sub-layers are created here, per the Keras 3 pattern. The u1/u2
        # bottlenecks are always bias-free; only v1/v2 carry the optional bias.
        self.u1 = keras.layers.Dense(
            units=self.rank,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="u1"
        )
        self.v1 = keras.layers.Dense(
            units=self.hidden_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="v1"
        )
        self.u2 = keras.layers.Dense(
            units=self.rank,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="u2"
        )
        self.v2 = keras.layers.Dense(
            units=self.output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="v2"
        )

        self.dropout = keras.layers.Dropout(
            rate=self.dropout_rate,
            name="dropout"
        )

        logger.info(
            f"Initialized LowRankFFN with hidden_dim={hidden_dim}, "
            f"output_dim={output_dim}, rank={self.rank} (arg={rank}), "
            f"activation={activation}, dropout_rate={dropout_rate}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers in computational order.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        # Expansion bottleneck: U1 (input_dim -> rank) then V1 (rank -> hidden_dim).
        self.u1.build(input_shape)
        u1_output_shape = self.u1.compute_output_shape(input_shape)
        self.v1.build(u1_output_shape)
        v1_output_shape = self.v1.compute_output_shape(u1_output_shape)

        # Activation has no parameters; dropout preserves shape.
        self.dropout.build(v1_output_shape)

        # Contraction bottleneck: U2 (hidden_dim -> rank) then V2 (rank -> output_dim).
        self.u2.build(v1_output_shape)
        u2_output_shape = self.u2.compute_output_shape(v1_output_shape)
        self.v2.build(u2_output_shape)

        # Always call parent build at the end.
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply the low-rank FFN to input tensors.

        :param inputs: Input tensor of shape (..., input_dim).
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode (affects dropout).
        :type training: Optional[bool]
        :return: Output tensor of shape (..., output_dim).
        :rtype: keras.KerasTensor
        """
        # Factorized expansion.
        h = self.v1(self.u1(inputs))

        # Activation + dropout.
        h = self.activation_fn(h)
        h = self.dropout(h, training=training)

        # Factorized contraction.
        return self.v2(self.u2(h))

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple. All dimensions preserved except the last,
            which changes to output_dim.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns ALL parameters passed to __init__ for complete reconstruction.
        The original ``rank`` argument (possibly ``None``) is emitted so the
        identical default-resolution runs on deserialization.

        :return: Dictionary containing the complete layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "rank": self._rank_arg,
            "activation": keras.activations.serialize(self.activation_fn),
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
