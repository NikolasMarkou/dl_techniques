"""
A low-rank factorized Feed-Forward Network.

This layer is structurally identical to the standard Transformer MLP
("expand-then-contract", `MLPBlock`), but each of its two dense projections is
replaced by a *low-rank factorization*. Instead of a single full-rank weight
matrix `W` of shape `(d_in, d_out)`, the layer learns two smaller matrices
`U` of shape `(d_in, rank)` and `V` of shape `(rank, d_out)` whose product
`U @ V` approximates `W`. When `rank << min(d_in, d_out)` this drastically
reduces the parameter count and compute of the projection.

Architectural Overview:
The network keeps the classic expand/contract shape, but each projection is a
bottleneck:

1.  **Expansion (factorized)**: The input `(..., input_dim)` is projected to
    `hidden_dim` through a bottleneck of width `rank`:
    `U1: Dense(rank, no bias) -> V1: Dense(hidden_dim)`.

2.  **Non-linear Activation**: A configurable activation (default GELU) is
    applied element-wise to the expanded representation, followed by optional
    dropout.

3.  **Contraction (factorized)**: The activated representation is projected
    back down to `output_dim` through a second bottleneck of width `rank`:
    `U2: Dense(rank, no bias) -> V2: Dense(output_dim)`.

The intermediate `U` projections are deliberately bias-free: a bias on the
bottleneck is redundant since it is immediately consumed by the following
linear `V` projection (it can be folded into `V`'s bias). Only the `V`
projections carry the (optional) bias.

Foundational Mathematics:
For an input vector `x` at a single position the layer computes:

`FFN(x) = V_2 @ (U_2 @ activation(V_1 @ (U_1 @ x)))`

(with biases on the `V` maps when `use_bias=True`). A full dense FFN costs
`input_dim*hidden_dim + hidden_dim*output_dim` parameters; the low-rank form
costs `rank*(input_dim + hidden_dim) + rank*(hidden_dim + output_dim)`, which
is strictly smaller whenever
`rank < (input_dim*hidden_dim)/(input_dim + hidden_dim)` (and analogously for
the contraction). For `rank << dims` the savings are substantial.

References:
-   Vaswani, A., et al. (2017). Attention Is All You Need. NIPS. (the base
    FFN structure this layer specializes).
-   Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language
    Models. arXiv:2106.09685. (the low-rank factorization principle reused
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

    This block implements a standard expand/contract MLP in which each dense
    projection is replaced by a low-rank product
    ``Dense(rank, use_bias=False) -> Dense(out)``. The computation is
    ``FFN(x) = V2(U2(activation(V1(U1(x)))))`` applied identically to each
    token position with shared weights. When ``rank`` is small relative to the
    layer dimensions this yields a sub-quadratic parameter count compared to a
    dense MLP of the same hidden/output dimensions.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────┐
        │   Input (..., input_dim)│
        └────────────┬────────────┘
                     ▼
        ┌─────────────────────────┐
        │  U1: Dense(rank, no bias)│
        └────────────┬────────────┘
                     ▼
        ┌─────────────────────────┐
        │  V1: Dense(hidden_dim)  │
        └────────────┬────────────┘
                     ▼
        ┌─────────────────────────┐
        │   Activation (e.g. GELU)│
        └────────────┬────────────┘
                     ▼
        ┌─────────────────────────┐
        │   Dropout (optional)    │
        └────────────┬────────────┘
                     ▼
        ┌─────────────────────────┐
        │  U2: Dense(rank, no bias)│
        └────────────┬────────────┘
                     ▼
        ┌─────────────────────────┐
        │  V2: Dense(output_dim)  │
        └────────────┬────────────┘
                     ▼
        ┌──────────────────────────┐
        │  Output (..., output_dim)│
        └──────────────────────────┘

    :param hidden_dim: Integer, hidden (expansion) dimension. Must be positive.
    :type hidden_dim: int
    :param output_dim: Integer, output (projection) dimension. Must be positive.
    :type output_dim: int
    :param rank: Optional integer bottleneck width shared by both factorized
        projections. If ``None``, it is resolved at construction time to
        ``max(1, hidden_dim // 4)``. If provided it must be positive. The
        original (possibly ``None``) value is preserved for serialization so
        round-trips reconstruct identically.
    :type rank: Optional[int]
    :param activation: Activation function name or callable applied after the
        expansion. Accepts string names ('gelu', 'relu', 'swish') or callables.
        Defaults to 'gelu'.
    :type activation: Union[str, Callable]
    :param dropout_rate: Dropout rate applied after the activation. Must be in
        range [0.0, 1.0). Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether the ``V`` projections use bias vectors. The ``U``
        (bottleneck) projections are always bias-free. Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for the dense layer kernels.
        Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the bias vectors. Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for the dense layer kernels.
        Defaults to None.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Optional regularizer for the dense layer biases.
        Defaults to None.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If hidden_dim or output_dim is not positive.
    :raises ValueError: If rank is provided and is not positive.
    :raises ValueError: If dropout_rate is not in range [0.0, 1.0).

    Note:
        The ``rank`` default resolution (``max(1, hidden_dim // 4)`` when
        ``None``) is a construction-time decision: the resolved integer is
        stored in ``self.rank`` and used to size the bottleneck Dense layers.
        ``get_config`` emits the *original* ``rank`` argument (possibly
        ``None``) so deserialization re-runs the identical resolution.
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
        super().__init__(**kwargs)

        # Validate inputs immediately
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if rank is not None and rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")

        # Store ALL configuration parameters.
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

        # CREATE all sub-layers in __init__ (modern Keras 3 pattern). The U
        # (bottleneck) projections are always bias-free; only the V projections
        # carry the optional bias.
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
