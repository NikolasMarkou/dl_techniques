"""
This module provides the implementation of `LogicFFN`, a custom Keras 3 layer
designed to integrate principles of soft logical reasoning into deep learning
models. It is intended to be used as a powerful, drop-in replacement for standard
Feed-Forward Network (FFN) or Multi-Layer Perceptron (MLP) blocks, especially
within architectures like the Transformer.

The core idea is to move beyond simple non-linear transformations (like ReLU or GELU)
and instead force the network to perform explicit, differentiable logical operations
on its input features. This provides a strong inductive bias for tasks that may
benefit from structured, logical reasoning.

Core Concepts:
-------------
The `LogicFFN` layer operates through a sequence of carefully designed steps:

1.  **Input Projection & Soft-Bits:**
    Inputs are first linearly projected into a higher-dimensional "logic space"
    with two distinct operands (`a` and `b`). A sigmoid activation function is
    then applied to transform these operands into "soft-bits"—continuous values
    between 0 and 1 that approximate binary logic.

2.  **Soft Logic Operations:**
    Three fundamental logic operations are computed in parallel using their
    probabilistic (or "soft") counterparts:
    - **AND:** `a * b`
    - **OR:** `a + b - a * b`
    - **XOR:** `(a - b)^2`

3.  **Dynamic Gating Mechanism:**
    A key innovation of this layer is its dynamic gating mechanism. A separate
    projection from the original input learns a set of weights for the three
    logic operations. A temperature-scaled softmax is applied to these weights,
    creating a dynamic probability distribution that determines the importance
    of each logic operation for every token in the sequence. This allows the
    model to adapt its reasoning strategy based on the input.

4.  **Weighted Combination and Output:**
    The results of the three logic operations are combined via a weighted sum
    using the learned gates. This aggregated tensor, which now contains a
    logically-processed representation of the input, is then projected back to
    the desired output dimension.
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
    Logic-based Feed-Forward Network using learnable soft logic operations.

    This layer transforms input into soft-bit representations and applies
    fundamental logic operations (AND, OR, XOR) with learnable gating mechanisms
    to combine the results. The architecture is inspired by Neural Logic Machines
    and provides a novel approach to information processing in neural networks.

    The layer works by:
    1. Projecting input to logic space with two operands
    2. Converting to soft-bits using sigmoid activation
    3. Performing logic operations (AND, OR, XOR)
    4. Learning dynamic gates to weight operation importance
    5. Combining results and projecting back to output dimension

    Args:
        output_dim: Integer, the final output dimension of the layer. Must be positive.
        logic_dim: Integer, the intermediate dimension for logic operations.
            Controls the complexity of logical reasoning. Must be positive.
        use_bias: Boolean, whether to use bias terms in dense layers.
            Defaults to True.
        kernel_initializer: String or initializer instance for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or initializer instance for bias weights.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        temperature: Float, temperature parameter for softmax gating.
            Higher values make gating more uniform. Must be positive. Defaults to 1.0.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, input_dim)`

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`

    Attributes:
        logic_projection: Dense layer projecting input to logic space with 2 operands.
        gate_projection: Dense layer learning weights for logic operation gating.
        output_projection: Dense layer projecting combined logic results to output.
        num_logic_ops: Number of logic operations (always 3: AND, OR, XOR).

    Example:
        ```python
        # Basic usage
        layer = LogicFFN(output_dim=768, logic_dim=256)

        inputs = keras.Input(shape=(128, 768))
        outputs = layer(inputs)
        print(outputs.shape)  # (None, 128, 768)

        # Advanced configuration with regularization
        layer = LogicFFN(
            output_dim=768,
            logic_dim=256,
            use_bias=False,
            kernel_regularizer=keras.regularizers.L2(1e-4),
            temperature=1.5
        )

        # In a transformer model
        inputs = keras.Input(shape=(128, 768))
        x = LogicFFN(output_dim=768, logic_dim=256)(inputs)
        model = keras.Model(inputs, x)
        ```

    Raises:
        ValueError: If output_dim, logic_dim, or temperature are not positive.

    Note:
        This layer is particularly effective for tasks requiring explicit
        logical reasoning over input features. The logic_dim parameter
        controls the complexity of logical operations that can be learned.
        Higher logic_dim allows for more complex logical patterns but increases
        computational cost.
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
            units=self.logic_dim * 2,  # Two operands for logic operations
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

        For robust serialization, explicitly build each sub-layer
        following the Modern Keras 3 pattern.

        Args:
            input_shape: Shape tuple of input tensor.

        Raises:
            ValueError: If input shape is invalid.
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

        # Always call parent build at the end
        super().build(input_shape)

        logger.info(
            f"Built LogicFFN: input_dim={input_dim}, "
            f"logic_dim={self.logic_dim}, output_dim={self.output_dim}"
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the logic FFN.

        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, input_dim).
            training: Boolean indicating training mode.

        Returns:
            Output tensor of shape (batch_size, sequence_length, output_dim).
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

        # OR operation: a + b - a*b (De Morgan's law)
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

        Args:
            input_shape: Shape tuple of input tensor.

        Returns:
            Output shape tuple with last dimension changed to output_dim.
        """
        # Replace last dimension with output_dim
        output_shape_list = list(input_shape)
        output_shape_list[-1] = self.output_dim
        return tuple(output_shape_list)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns ALL initialization parameters to ensure proper reconstruction.

        Returns:
            Dictionary containing complete layer configuration.
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
    Create a standard LogicFFN layer with recommended settings.

    Args:
        output_dim: Output dimension.
        logic_dim: Logic operation dimension.

    Returns:
        Configured LogicFFN layer.
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
    Create a LogicFFN layer with L2 regularization.

    Args:
        output_dim: Output dimension.
        logic_dim: Logic operation dimension.
        l2_reg: L2 regularization strength.

    Returns:
        Configured LogicFFN layer with regularization.
    """
    return LogicFFN(
        output_dim=output_dim,
        logic_dim=logic_dim,
        kernel_regularizer=keras.regularizers.L2(l2_reg),
        bias_regularizer=keras.regularizers.L2(l2_reg),
        temperature=1.0
    )