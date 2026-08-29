"""
A feed-forward network built from soft logic gates.

This layer replaces the non-linearity of a standard FFN with three
differentiable logic operations. It gives the model a bias towards
symbolic-style reasoning: instead of one activation function, it learns how
much of AND, OR and XOR to apply at each position.

**Architecture Overview:**
The layer works like a small logic circuit:

1.  **Projection into operands**. A Dense layer maps the input to
    ``2 * logic_dim`` features, which are split in half into the two
    operands ``a`` and ``b``.

2.  **Soft bits**. Both operands go through a sigmoid, which puts them in
    ``(0, 1)``. Read a value as the probability that a feature is true.

3.  **Three operations in parallel**. AND, OR and XOR are computed on the
    soft bits, using continuous analogues from probability theory. There is
    no NOT gate; the layer has exactly three operations.

4.  **Dynamic gating**. A second Dense layer maps the SAME input to three
    logits. A temperature-scaled softmax turns them into weights that sum to
    one. This decides, per position, how much each operation counts.

5.  **Weighted combination and output**. The three results are combined by
    the weighted sum, and a final Dense layer maps the combination to
    ``output_dim``.

**Mathematics:**
Let ``x`` be the input vector.

1.  The operands are produced and squashed into soft bits:
    ``[p_a, p_b] = W_logic @ x + b_logic``
    ``a = sigmoid(p_a)``, ``b = sigmoid(p_b)``

2.  The soft logic operations follow probabilistic rules:
    - **Soft AND**: ``y_and = a * b``
      (product rule for independent events: P(A and B) = P(A)P(B))
    - **Soft OR**: ``y_or = a + b - a * b``
      (inclusion-exclusion: P(A or B) = P(A) + P(B) - P(A and B))
    - **Soft XOR**: ``y_xor = (a - b)^2``
      (squared difference: large when ``a`` and ``b`` disagree, near zero
      when they agree, which is the XOR truth table for soft values)

3.  The gates come from the input:
    ``logits = W_gate @ x + b_gate``
    ``g = softmax(logits / temperature)``
    with ``g = [g_and, g_or, g_xor]``.

4.  The combination is the weighted sum:
    ``h = g_and * y_and + g_or * y_or + g_xor * y_xor``

5.  The output is a linear projection: ``y_out = W_out @ h + b_out``

References:
This architecture comes from neuro-symbolic AI, which tries to combine deep
learning with symbolic reasoning. The mechanisms are closest to:

-   Dong, H., et al. (2019). Neural Logic Machines. ICLR.
-   Probabilistic logic and fuzzy logic, which extend Boolean logic to
    uncertainty and to continuous values.

"""

import keras
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class LogicFFN(keras.layers.Layer):
    """
    Logic-based feed-forward network with learnable soft logic gates.

    The input is projected to two operands, squashed to soft bits by a
    sigmoid, and fed to three logic operations: Soft AND (``a * b``), Soft OR
    (``a + b - a*b``) and Soft XOR (``(a - b)^2``). A separate projection of
    the same input produces a temperature-scaled softmax over the three
    operations. The weighted combination is projected to ``output_dim``.

    There are three operations and no more. ``num_logic_ops`` is set to 3 in
    ``__init__`` and is not a constructor argument. There is no NOT gate.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │ Input  [B, T, input_dim]             │
        └──────────────────┬───────────────────┘
                           │
                     ┌─────┴─────┐
                     ▼           ▼
        ┌────────────────┐ ┌────────────────────┐
        │logic_projection│ │  gate_projection   │
        │ Dense(2 * L)   │ │  Dense(3)          │
        └───────┬────────┘ └───────┬────────────┘
                ▼                  ▼
        ┌────────────────┐ ┌──────────────────────┐
        │ split -> a, b  │ │ softmax(g / temp)    │
        │ sigmoid each   │ │ -> [g_and,g_or,g_xor]│
        └───────┬────────┘ └───────┬──────────────┘
                ▼                  │
        ┌────────────────┐         │
        │ AND: a*b       │         │
        │ OR : a+b-a*b   │         │
        │ XOR: (a-b)^2   │         │
        └───────┬────────┘         │
                │  [B, T, 3, L]    │  [B, T, 3, 1]
                └────────┬─────────┘
                         ▼
        ┌──────────────────────────────────────┐
        │  weighted sum over the 3 operations  │
        │  [B, T, L]                           │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │ output_projection: Dense(output_dim) │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │ Output  [B, T, output_dim]           │
        └──────────────────────────────────────┘

        L = logic_dim, T = sequence length.

    **The soft logic gate (block internals):**

    .. code-block:: text

                x  [B, T, input_dim]
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
     logic_projection            gate_projection
     Dense(2 * L)                Dense(3)
             │                         │
             ▼                         ▼
     split on axis -1          softmax(g / temperature)
     a [B,T,L]  b [B,T,L]      [B, T, 3]
             │                         │
             ▼                         ▼
     sigmoid(a), sigmoid(b)    expand_dims -> [B,T,3,1]
             │                         │
             ▼                         │
     AND: a * b                        │
     OR : a + b - a*b                  │
     XOR: (a - b)^2                    │
             │                         │
             ▼                         │
     stack on axis -2  [B,T,3,L]       │
             │                         │
             └────────────┬────────────┘
                          ▼
              multiply, then sum on axis -2
                       [B, T, L]

        Both projections read the SAME input x, in parallel. The
        gate does not see the logic results.

        temperature divides the gate LOGITS before the softmax. A
        large temperature flattens the mix towards 1/3 each; a
        small one makes the layer pick one operation. It is a
        plain float, not a weight, so it never trains.

    :param output_dim: Width of the output. Must be positive.
    :type output_dim: int
    :param logic_dim: Width of each operand, and of the combined logic
        result. ``logic_projection`` emits twice this. Must be positive.
    :type logic_dim: int
    :param use_bias: Whether the three Dense layers carry a bias. Defaults to
        True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for the kernels. The same instance
        goes to all three Dense layers. Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the biases. Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for the kernels. Defaults to None.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Regularizer for the biases. Defaults to None.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param temperature: Divides the gate logits before the softmax. Higher
        values make the mix more uniform. Must be positive. Defaults to 1.0.
    :type temperature: float
    :param kwargs: Extra arguments for ``keras.layers.Layer`` (``name``,
        ``dtype``, and so on).
    :type kwargs: Any

    :ivar output_dim: The stored output width.
    :vartype output_dim: int
    :ivar logic_dim: The stored operand width.
    :vartype logic_dim: int
    :ivar use_bias: Whether the Dense layers carry a bias.
    :vartype use_bias: bool
    :ivar kernel_initializer: The resolved kernel initializer.
    :vartype kernel_initializer: keras.initializers.Initializer
    :ivar bias_initializer: The resolved bias initializer.
    :vartype bias_initializer: keras.initializers.Initializer
    :ivar kernel_regularizer: The resolved kernel regularizer, or ``None``.
    :vartype kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar bias_regularizer: The resolved bias regularizer, or ``None``.
    :vartype bias_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar temperature: The stored softmax temperature.
    :vartype temperature: float
    :ivar num_logic_ops: Fixed at 3, one per operation. Not a constructor
        argument and not serialized.
    :vartype num_logic_ops: int
    :ivar logic_projection: ``Dense(2 * logic_dim)`` producing the operands.
    :vartype logic_projection: keras.layers.Dense
    :ivar gate_projection: ``Dense(3)`` producing the gate logits.
    :vartype gate_projection: keras.layers.Dense
    :ivar output_projection: ``Dense(output_dim)``, the final projection.
    :vartype output_projection: keras.layers.Dense

    :raises ValueError: If ``output_dim``, ``logic_dim`` or ``temperature`` is
        not positive.
    :raises ValueError: From ``build()``, if the input has rank < 2 or if its
        last axis is ``None``.

    Input shape:
        Tensor of rank >= 2, shape ``(..., input_dim)``. The last axis must be
        known at build time.

    Output shape:
        Same rank and leading axes as the input, last axis ``output_dim``.

    Example:
        .. code-block:: python

            ffn = LogicFFN(output_dim=64, logic_dim=32)
            y = ffn(keras.random.normal((2, 10, 48)))
            y.shape                 # (2, 10, 64)

    Note:
        Higher ``logic_dim`` allows more logical patterns at more compute.
        ``logic_projection`` is the layer's widest matrix, at
        ``input_dim x 2 * logic_dim``.
    """

    def __init__(
            self,
            output_dim: int,
            logic_dim: int,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            temperature: float = 1.0,
            **kwargs: Any
    ) -> None:
        """
        Validate the configuration and create the three Dense layers.

        Every argument is documented on the class. ``num_logic_ops`` is set
        here to 3 and is not configurable.

        :raises ValueError: If ``output_dim``, ``logic_dim`` or
            ``temperature`` is not positive.
        """
        super().__init__(**kwargs)

        # Validate input parameters
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if logic_dim <= 0:
            raise ValueError(f"logic_dim must be positive, got {logic_dim}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        # Store ALL configuration parameters
        self.output_dim = output_dim
        self.logic_dim = logic_dim
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.temperature = temperature

        # Number of logic operations: AND, OR, XOR
        self.num_logic_ops = 3

        # CREATE all sub-layers in __init__ - Modern Keras 3 pattern
        self.logic_projection = keras.layers.Dense(
            # Two operands, a and b, are split out of this one projection.
            units=self.logic_dim * 2,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='logic_projection'
        )

        self.gate_projection = keras.layers.Dense(
            units=self.num_logic_ops,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='gate_projection'
        )

        self.output_projection = keras.layers.Dense(
            units=self.output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='output_projection'
        )

        logger.info(
            f"Created LogicFFN: output_dim={self.output_dim}, "
            f"logic_dim={self.logic_dim}, temperature={self.temperature}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        Explicitly builds each sub-layer for robust serialization following
        the modern Keras 3 pattern.

        :param input_shape: Shape tuple of input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If input shape is invalid.
        """
        if self.built:
            return

        # Ensure input_shape is a tuple for consistent handling
        input_shape = tuple(input_shape)

        # Validate input shape
        if len(input_shape) < 2:
            raise ValueError(
                f"Input must be at least 2D, got {len(input_shape)}D: {input_shape}"
            )

        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Input feature dimension must be specified")

        # Explicitly build sub-layers in computational order
        self.logic_projection.build(input_shape)
        self.gate_projection.build(input_shape)

        # Output projection takes logic_dim as input
        # Ensure consistent tuple creation
        logic_output_shape = tuple(list(input_shape[:-1]) + [self.logic_dim])
        self.output_projection.build(logic_output_shape)

        logger.info(
            f"Built LogicFFN: input_dim={input_dim}, "
            f"logic_dim={self.logic_dim}, output_dim={self.output_dim}"
        )

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the logic FFN.

        :param inputs: Input tensor of shape (batch_size, sequence_length, input_dim).
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :return: Output tensor of shape (batch_size, sequence_length, output_dim).
        :rtype: keras.KerasTensor
        """
        # Step 1: Project to logic space and split into two operands
        projected = self.logic_projection(inputs, training=training)
        operand_a, operand_b = keras.ops.split(projected, 2, axis=-1)

        # Step 2: Convert to soft-bits using sigmoid activation
        # This creates continuous approximations of binary values
        soft_a = keras.ops.sigmoid(operand_a)
        soft_b = keras.ops.sigmoid(operand_b)

        # Step 3: Perform logic operations using soft logic
        # AND operation: element-wise multiplication
        logic_and = soft_a * soft_b

        # OR operation: a + b - a*b (inclusion-exclusion, not De Morgan)
        logic_or = soft_a + soft_b - (soft_a * soft_b)

        # XOR operation: (a - b)^2 gives high values when a and b differ
        logic_xor = keras.ops.square(soft_a - soft_b)

        # Step 4: Stack logic operation results
        # Shape: (batch_size, sequence_length, num_logic_ops, logic_dim)
        logic_results = keras.ops.stack([logic_and, logic_or, logic_xor], axis=-2)

        # Step 5: Learn dynamic gates to weight logic operations
        gate_weights = self.gate_projection(inputs, training=training)
        # Apply temperature scaling and softmax for smooth gating
        gate_weights = keras.ops.softmax(gate_weights / self.temperature, axis=-1)

        # Step 6: Apply gates to combine logic operations
        # Expand dimensions for broadcasting: (batch, seq, num_ops, 1)
        expanded_gates = keras.ops.expand_dims(gate_weights, axis=-1)

        # Weighted combination of logic operations
        # Shape: (batch_size, sequence_length, logic_dim)
        combined_logic = keras.ops.sum(logic_results * expanded_gates, axis=-2)

        # Step 7: Project back to output dimension
        output = self.output_projection(combined_logic, training=training)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape given input shape.

        :param input_shape: Shape tuple of input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple with last dimension changed to output_dim.
        :rtype: Tuple[Optional[int], ...]
        """
        # Replace last dimension with output_dim
        output_shape_list = list(input_shape)
        output_shape_list[-1] = self.output_dim
        return tuple(output_shape_list)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns ALL initialization parameters to ensure proper reconstruction.

        :return: Dictionary containing complete layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'output_dim': self.output_dim,
            'logic_dim': self.logic_dim,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'temperature': self.temperature,
        })
        return config


# ---------------------------------------------------------------------
# Factory functions for common configurations
# ---------------------------------------------------------------------

def create_logic_ffn_standard(output_dim: int, logic_dim: int) -> LogicFFN:
    """
    Build a LogicFFN with no regularization.

    This is the plain preset. It pins ``temperature`` to 1.0 and leaves every
    other LogicFFN argument at its default, so the returned layer has no
    kernel or bias regularizer.

    **Presets:**

    .. code-block:: text

        builder                       temp  kernel_reg  bias_reg
        ----------------------------  ----  ----------  ----------
        create_logic_ffn_standard     1.0   None        None
        create_logic_ffn_regularized  1.0   L2(l2_reg)  L2(l2_reg)

        temp is the temperature argument. l2_reg defaults to 1e-4.
        Neither preset exposes use_bias, the initializers or the
        regularizers as arguments, so both keep the LogicFFN
        defaults for them: use_bias=True,
        kernel_initializer='glorot_uniform' and
        bias_initializer='zeros'.

        temperature=1.0 is also the LogicFFN default, so passing
        it here changes nothing; both presets state it anyway.

    :param output_dim: Width of the output, passed straight through.
    :type output_dim: int
    :param logic_dim: Width of each logic operand, passed straight through.
    :type logic_dim: int
    :return: A new ``LogicFFN``, unbuilt.
    :rtype: LogicFFN
    :raises ValueError: If ``output_dim`` or ``logic_dim`` is not positive.
        The check lives in ``LogicFFN.__init__``.
    """
    return LogicFFN(
        output_dim=output_dim,
        logic_dim=logic_dim,
        temperature=1.0
    )


def create_logic_ffn_regularized(
    output_dim: int,
    logic_dim: int,
    l2_reg: float = 1e-4
) -> LogicFFN:
    """
    Build a LogicFFN with L2 on both the kernels and the biases.

    Same as ``create_logic_ffn_standard`` except that a fresh
    ``keras.regularizers.L2(l2_reg)`` is installed on the kernels and a
    second one on the biases. The two-row preset table comparing this builder
    with ``create_logic_ffn_standard`` is in that function's docstring.

    This regularizes the BIASES too, which the standard Keras default does
    not. If you only want the kernels regularized, construct ``LogicFFN``
    directly.

    :param output_dim: Width of the output, passed straight through.
    :type output_dim: int
    :param logic_dim: Width of each logic operand, passed straight through.
    :type logic_dim: int
    :param l2_reg: L2 strength used for both regularizers. Defaults to 1e-4.
    :type l2_reg: float
    :return: A new ``LogicFFN`` carrying two L2 regularizers, unbuilt.
    :rtype: LogicFFN
    :raises ValueError: If ``output_dim`` or ``logic_dim`` is not positive.
        The check lives in ``LogicFFN.__init__``.
    """
    return LogicFFN(
        output_dim=output_dim,
        logic_dim=logic_dim,
        kernel_regularizer=keras.regularizers.L2(l2_reg),
        bias_regularizer=keras.regularizers.L2(l2_reg),
        temperature=1.0
    )
