import keras
from typing import Optional, Union, Tuple, Dict, Any
from keras import layers, ops, initializers, regularizers

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class LogicFFN(layers.Layer):
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
        output_dim: Integer, the final output dimension of the layer.
        logic_dim: Integer, the intermediate dimension for logic operations.
            Controls the complexity of logical reasoning.
        use_bias: Boolean, whether to use bias terms in dense layers.
            Defaults to True.
        kernel_initializer: String or initializer instance for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or initializer instance for bias weights.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        temperature: Float, temperature parameter for softmax gating.
            Higher values make gating more uniform. Defaults to 1.0.
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
            kernel_regularizer='l2',
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

        # Store configuration
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

        # Initialize sublayers to None - will be created in build()
        self.logic_projection = None
        self.gate_projection = None
        self.output_projection = None

        # Store build input shape for serialization
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the logic FFN sublayers.

        Args:
            input_shape: Shape tuple of input tensor.
        """
        if self.built:
            return

        # Store for serialization
        self._build_input_shape = input_shape

        # Validate input shape
        if len(input_shape) < 2:
            raise ValueError(
                f"Input must be at least 2D, got {len(input_shape)}D: {input_shape}"
            )

        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Input feature dimension must be specified")

        logger.info(
            f"Building LogicFFN: input_dim={input_dim}, "
            f"logic_dim={self.logic_dim}, output_dim={self.output_dim}"
        )

        # Create projection layers
        self.logic_projection = layers.Dense(
            units=self.logic_dim * 2,  # Two operands for logic operations
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='logic_projection'
        )

        self.gate_projection = layers.Dense(
            units=self.num_logic_ops,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='gate_projection'
        )

        self.output_projection = layers.Dense(
            units=self.output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='output_projection'
        )

        # Build sublayers
        self.logic_projection.build(input_shape)
        self.gate_projection.build(input_shape)
        self.output_projection.build(input_shape[:-1] + (self.logic_dim,))

        super().build(input_shape)

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
        # Convert to list for manipulation
        input_shape_list = list(input_shape)

        # Replace last dimension with output_dim
        output_shape_list = input_shape_list[:-1] + [self.output_dim]

        return tuple(output_shape_list)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

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

    def get_build_config(self) -> Dict[str, Any]:
        """
        Get build configuration for serialization.

        Returns:
            Dictionary containing build configuration.
        """
        return {
            'input_shape': self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """
        Build layer from configuration.

        Args:
            config: Dictionary containing build configuration.
        """
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])

# ---------------------------------------------------------------------
