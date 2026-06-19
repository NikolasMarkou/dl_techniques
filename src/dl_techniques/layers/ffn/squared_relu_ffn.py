"""
The Squared-ReLU Feed-Forward Network from the Primer architecture.

This layer is a position-wise feed-forward network identical in structure to
the standard Transformer MLP, but with one deliberate simplification: the
intermediate non-linearity is *fixed* to the squared ReLU, ``relu(x)**2``,
rather than being a configurable activation. This is the single change that
the Primer architecture search (So et al. 2021) identified as one of its most
effective and transferable modifications to the vanilla Transformer FFN.

Architectural Overview:
The network keeps the classic "expand-then-contract" shape:

1.  **Expansion**: An initial linear layer (`fc1`) projects the input from its
    original dimension (`input_dim`) up to a wider intermediate dimension
    (`hidden_dim`), typically a 4x expansion.

2.  **Squared-ReLU Non-linearity**: The expanded representation is passed
    through ``relu(x)**2`` — the ReLU clamps the negative half-plane to zero
    and the square sharpens the positive response. Unlike GELU or plain ReLU
    this is a strictly fixed non-linearity (it is the whole point of this
    layer), so there is no `activation` constructor parameter.

3.  **Contraction**: A second linear layer (`fc2`) projects the activated
    representation back down to `output_dim`, usually equal to `input_dim` so
    the block can sit inside a residual connection.

Foundational Mathematics:
For an input vector `x` at a single sequence position the layer computes:

`FFN(x) = W_2 @ relu(W_1 @ x + b_1)**2 + b_2`

where:
- `W_1`, `b_1` project `x` from `input_dim` to `hidden_dim`.
- `relu(z)**2 = max(z, 0)**2` is applied element-wise (the squared ReLU).
- `W_2`, `b_2` project the result from `hidden_dim` to `output_dim`.

The squared ReLU produces a smoother, faster-growing response on the positive
half-line than ReLU while preserving exact sparsity on the negative half-line,
which the Primer search found to improve training efficiency.

References:
-   So, D. R., Mańke, W., Liu, H., Dai, Z., Shazeer, N., & Le, Q. V. (2021).
    Primer: Searching for Efficient Transformers for Language Modeling.
    arXiv preprint arXiv:2109.08668.
-   Vaswani, A., et al. (2017). Attention Is All You Need. NIPS. (the base
    FFN structure this layer specializes).

"""

import keras
from typing import Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SquaredReLUFFN(keras.layers.Layer):
    """
    Squared-ReLU (Primer) feed-forward network.

    This block implements the Primer feed-forward network: two dense layers
    with a *fixed* squared-ReLU non-linearity and optional dropout in between.
    The computation is ``FFN(x) = W_2 @ relu(W_1 @ x + b_1)**2 + b_2``,
    applied identically to each token position with shared weights. The only
    structural difference from a standard MLP block is the squared-ReLU
    non-linearity, which is intentionally not configurable.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────┐
        │   Input (..., input_dim)│
        └────────────┬────────────┘
                     ▼
        ┌─────────────────────────┐
        │  fc1: Dense(hidden_dim) │
        └────────────┬────────────┘
                     ▼
        ┌─────────────────────────┐
        │   relu(x) ** 2          │
        └────────────┬────────────┘
                     ▼
        ┌─────────────────────────┐
        │   Dropout (optional)    │
        └────────────┬────────────┘
                     ▼
        ┌─────────────────────────┐
        │  fc2: Dense(output_dim) │
        └────────────┬────────────┘
                     ▼
        ┌──────────────────────────┐
        │  Output (..., output_dim)│
        └──────────────────────────┘

    :param hidden_dim: Integer, hidden dimension for the first dense layer
        (expansion). Must be positive.
    :type hidden_dim: int
    :param output_dim: Integer, output dimension for the second dense layer
        (projection). Must be positive.
    :type output_dim: int
    :param dropout_rate: Dropout rate applied after the squared-ReLU
        non-linearity. Must be in range [0.0, 1.0). Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether the dense layers use bias vectors. Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for the dense layer kernels.
        Accepts string names ('glorot_uniform', 'he_normal') or Initializer
        instances. Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the bias vectors. Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for the dense layer kernels.
        Can be string name ('l2') or Regularizer instance. Defaults to None.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Optional regularizer for the dense layer biases.
        Can be string name ('l1') or Regularizer instance. Defaults to None.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If hidden_dim or output_dim is not positive.
    :raises ValueError: If dropout_rate is not in range [0.0, 1.0).

    Note:
        The non-linearity is hard-wired to ``relu(x)**2``; there is no
        ``activation`` parameter by design. Use ``MLPBlock`` if a configurable
        activation is required.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
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
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")

        # Store ALL configuration parameters
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # CREATE all sub-layers in __init__ (modern Keras 3 pattern).
        # They remain unbuilt until build() is called.
        self.fc1 = keras.layers.Dense(
            units=self.hidden_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="fc1"
        )

        self.fc2 = keras.layers.Dense(
            units=self.output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="fc2"
        )

        self.dropout = keras.layers.Dropout(
            rate=self.dropout_rate,
            name="dropout"
        )

        logger.info(
            f"Initialized SquaredReLUFFN with hidden_dim={hidden_dim}, "
            f"output_dim={output_dim}, dropout_rate={dropout_rate}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        # Build sub-layers in computational order for robust serialization.
        self.fc1.build(input_shape)

        # Compute intermediate shape after the first dense layer.
        intermediate_shape = list(input_shape)
        intermediate_shape[-1] = self.hidden_dim
        intermediate_shape_tuple = tuple(intermediate_shape)

        # Dropout does not change shape.
        self.dropout.build(intermediate_shape_tuple)

        # Build second dense layer on the intermediate shape.
        self.fc2.build(intermediate_shape_tuple)

        # Always call parent build at the end.
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply the squared-ReLU FFN to input tensors.

        :param inputs: Input tensor of shape (..., input_dim).
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating whether the layer should behave in
            training mode. Affects dropout behavior.
        :type training: Optional[bool]
        :return: Output tensor of shape (..., output_dim).
        :rtype: keras.KerasTensor
        """
        # First dense layer (expansion).
        x = self.fc1(inputs)

        # Fixed squared-ReLU non-linearity: relu(x) ** 2.
        x = keras.ops.square(keras.ops.relu(x))

        # Dropout after the non-linearity.
        x = self.dropout(x, training=training)

        # Second dense layer (projection).
        x = self.fc2(x)

        return x

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

        :return: Dictionary containing the complete layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config
