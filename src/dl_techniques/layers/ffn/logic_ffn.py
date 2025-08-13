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
from keras import ops, initializers, regularizers

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
            Higher values make gating more uniform, must be positive. Defaults to 1.0.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, input_dim)`

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`

    Example:
        ```python
        # Basic usage
        layer = LogicFFN(output_dim=768, logic_dim=256)

        # Advanced configuration
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

    Note:
        This layer is particularly effective for tasks requiring explicit
        logical reasoning over input features. The logic_dim parameter
        controls the complexity of logical operations that can be learned.

        This implementation follows the modern Keras 3 pattern where sub-layers
        are created in __init__ and Keras handles the build lifecycle automatically.
        This ensures proper serialization and avoids common build errors.
    """

    def __init__(
            self,
            output_dim: int,
            logic_dim: int,
            use_bias: bool = True,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
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

        # Store ALL configuration parameters as instance attributes
        self.output_dim = output_dim
        self.logic_dim = logic_dim
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.temperature = temperature

        # Number of logic operations: AND, OR, XOR
        self.num_logic_ops = 3

        # CREATE all sub-layers here in __init__ (MODERN KERAS 3 PATTERN)
        logger.info(
            f"Creating LogicFFN sublayers: logic_dim={self.logic_dim}, "
            f"output_dim={self.output_dim}, num_ops={self.num_logic_ops}"
        )

        # Logic projection layer: projects input to logic space with two operands
        self.logic_projection = keras.layers.Dense(
            units=self.logic_dim * 2,  # Two operands for logic operations
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='logic_projection'
        )

        # Gate projection layer: learns weights for logic operations
        self.gate_projection = keras.layers.Dense(
            units=self.num_logic_ops,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='gate_projection'
        )

        # Output projection layer: projects back to desired output dimension
        self.output_projection = keras.layers.Dense(
            units=self.output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='output_projection'
        )

        # No custom weights to create, so no build() method is needed
        # Keras will automatically handle building of sub-layers

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
        operand_a, operand_b = ops.split(projected, 2, axis=-1)

        # Step 2: Convert to soft-bits using sigmoid activation
        # This creates continuous approximations of binary values
        soft_a = ops.sigmoid(operand_a)
        soft_b = ops.sigmoid(operand_b)

        # Step 3: Perform logic operations using soft logic
        # AND operation: element-wise multiplication
        logic_and = soft_a * soft_b

        # OR operation: a + b - a*b (De Morgan's law)
        logic_or = soft_a + soft_b - (soft_a * soft_b)

        # XOR operation: (a - b)^2 gives high values when a and b differ
        logic_xor = ops.square(soft_a - soft_b)

        # Step 4: Stack logic operation results
        # Shape: (batch_size, sequence_length, num_logic_ops, logic_dim)
        logic_results = ops.stack([logic_and, logic_or, logic_xor], axis=-2)

        # Step 5: Learn dynamic gates to weight logic operations
        gate_weights = self.gate_projection(inputs, training=training)
        # Apply temperature scaling and softmax for smooth gating
        gate_weights = ops.softmax(gate_weights / self.temperature, axis=-1)

        # Step 6: Apply gates to combine logic operations
        # Expand dimensions for broadcasting: (batch, seq, num_ops, 1)
        expanded_gates = ops.expand_dims(gate_weights, axis=-1)

        # Weighted combination of logic operations
        # Shape: (batch_size, sequence_length, logic_dim)
        combined_logic = ops.sum(logic_results * expanded_gates, axis=-2)

        # Step 7: Project back to output dimension
        output = self.output_projection(combined_logic, training=training)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape given input shape.

        Args:
            input_shape: Shape tuple of input tensor.

        Returns:
            Output shape tuple.
        """
        # Convert to list for manipulation, then back to tuple
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        This method must return ALL arguments needed to recreate the layer
        via __init__. Uses keras serializers for complex objects.

        Returns:
            Dictionary containing layer configuration.
        """
        config = super().get_config()
        config.update({
            'output_dim': self.output_dim,
            'logic_dim': self.logic_dim,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'temperature': self.temperature,
        })
        return config

    # DELETED: get_build_config() and build_from_config() methods
    # These are deprecated in Keras 3 and cause serialization issues
    # Keras handles the build lifecycle automatically with the modern pattern

# ---------------------------------------------------------------------