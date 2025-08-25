import keras
from keras import ops
from typing import List, Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .logic_operators import LearnableLogicOperator
from .arithmetic_operators import LearnableArithmeticOperator

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CircuitDepthLayer(keras.layers.Layer):
    """A single depth layer of the neural circuit.

    This layer implements a single depth level with parallel logic and arithmetic
    operators, featuring learnable routing and combination mechanisms. The layer
    processes inputs through multiple parallel operators and combines their outputs
    using learnable weights.

    The layer creates multiple logic and arithmetic operators that run in parallel,
    each receiving a weighted portion of the input. The outputs are then combined
    using another set of learnable weights, with an optional residual connection.

    Args:
        num_logic_ops: Integer, number of logic operators to run in parallel.
            Must be positive.
        num_arithmetic_ops: Integer, number of arithmetic operators to run in parallel.
            Must be positive.
        use_residual: Boolean, whether to use residual connections to add the
            input to the output.
        logic_op_types: Optional list of logic operation types to use in the
            logic operators. If None, uses all available logic operations.
        arithmetic_op_types: Optional list of arithmetic operation types to use
            in the arithmetic operators. If None, uses all available arithmetic operations.
        routing_initializer: Initializer for the routing weights that distribute
            input to different operators.
        combination_initializer: Initializer for the combination weights that
            combine operator outputs.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Returns:
        A 4D tensor with the same shape as the input, containing the processed
        features from the parallel operators.

    Raises:
        ValueError: If num_logic_ops or num_arithmetic_ops is not positive.
        ValueError: If input is not a 4D tensor.

    Example:
        >>> x = np.random.rand(4, 32, 32, 64)
        >>> depth_layer = CircuitDepthLayer(
        ...     num_logic_ops=2,
        ...     num_arithmetic_ops=2,
        ...     use_residual=True
        ... )
        >>> y = depth_layer(x)
        >>> print(y.shape)
        (4, 32, 32, 64)
    """

    def __init__(
            self,
            num_logic_ops: int = 2,
            num_arithmetic_ops: int = 2,
            use_residual: bool = True,
            logic_op_types: Optional[List[str]] = None,
            arithmetic_op_types: Optional[List[str]] = None,
            routing_initializer: Union[str, keras.initializers.Initializer] = "random_uniform",
            combination_initializer: Union[str, keras.initializers.Initializer] = "random_uniform",
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate parameters
        if num_logic_ops <= 0:
            raise ValueError("num_logic_ops must be positive.")
        if num_arithmetic_ops <= 0:
            raise ValueError("num_arithmetic_ops must be positive.")

        # Store ALL configuration parameters
        self.num_logic_ops = num_logic_ops
        self.num_arithmetic_ops = num_arithmetic_ops
        self.use_residual = use_residual
        self.logic_op_types = logic_op_types
        self.arithmetic_op_types = arithmetic_op_types
        self.routing_initializer = keras.initializers.get(routing_initializer)
        self.combination_initializer = keras.initializers.get(combination_initializer)

        # Sub-layers and weights will be initialized in build()
        self.logic_operators = []
        self.arithmetic_operators = []
        self.routing_weights = None
        self.combination_weights = None

        logger.info(
            f"CircuitDepthLayer initialized with {num_logic_ops} logic ops, "
            f"{num_arithmetic_ops} arithmetic ops, use_residual: {use_residual}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer components.

        Args:
            input_shape: Shape of the input tensor. Must be a 4D shape.
        """
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(
                f"CircuitDepthLayer expects 4D input (batch, height, width, channels), "
                f"got shape with {len(input_shape)} dimensions: {input_shape}"
            )

        # Create logic operators and explicitly build them
        for i in range(self.num_logic_ops):
            logic_op = LearnableLogicOperator(
                operation_types=self.logic_op_types,
                name=f"logic_op_{i}"
            )
            logic_op.build(input_shape)
            self.logic_operators.append(logic_op)

        # Create arithmetic operators and explicitly build them
        for i in range(self.num_arithmetic_ops):
            arithmetic_op = LearnableArithmeticOperator(
                operation_types=self.arithmetic_op_types,
                name=f"arithmetic_op_{i}"
            )
            arithmetic_op.build(input_shape)
            self.arithmetic_operators.append(arithmetic_op)

        total_operators = self.num_logic_ops + self.num_arithmetic_ops

        # Create routing weights for input distribution
        self.routing_weights = self.add_weight(
            name="routing_weights",
            shape=(total_operators,),
            initializer=self.routing_initializer,
            trainable=True,
        )

        # Create combination weights for output fusion
        self.combination_weights = self.add_weight(
            name="combination_weights",
            shape=(total_operators,),
            initializer=self.combination_initializer,
            trainable=True,
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the circuit depth layer.

        Args:
            inputs: Input tensor of shape [batch, height, width, features]
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor after processing through parallel operators.
        """
        # Normalize routing and combination weights
        routing_probs = ops.softmax(self.routing_weights)
        combination_probs = ops.softmax(self.combination_weights)

        # Apply logic operators
        logic_outputs = []
        for i, logic_op in enumerate(self.logic_operators):
            # Weight the input for this operator
            weight = ops.expand_dims(routing_probs[i], axis=0)
            # Expand weight to match tensor dimensions
            for _ in range(len(ops.shape(inputs)) - 1):
                weight = ops.expand_dims(weight, axis=-1)
            weighted_input = ops.multiply(inputs, weight)
            output = logic_op(weighted_input, training=training)
            logic_outputs.append(output)

        # Apply arithmetic operators
        arithmetic_outputs = []
        for i, arithmetic_op in enumerate(self.arithmetic_operators):
            # Weight the input for this operator
            weight = ops.expand_dims(routing_probs[self.num_logic_ops + i], axis=0)
            # Expand weight to match tensor dimensions
            for _ in range(len(ops.shape(inputs)) - 1):
                weight = ops.expand_dims(weight, axis=-1)
            weighted_input = ops.multiply(inputs, weight)
            output = arithmetic_op(weighted_input, training=training)
            arithmetic_outputs.append(output)

        # Combine all outputs
        all_outputs = logic_outputs + arithmetic_outputs

        # Weighted combination
        combined_output = ops.zeros_like(inputs)
        for i, output in enumerate(all_outputs):
            weight = ops.expand_dims(combination_probs[i], axis=0)
            # Expand weight to match tensor dimensions
            for _ in range(len(ops.shape(inputs)) - 1):
                weight = ops.expand_dims(weight, axis=-1)
            combined_output = ops.add(combined_output, ops.multiply(weight, output))

        # Apply residual connection if enabled
        if self.use_residual:
            combined_output = ops.add(combined_output, inputs)

        return combined_output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape tuple (same as input shape).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "num_logic_ops": self.num_logic_ops,
            "num_arithmetic_ops": self.num_arithmetic_ops,
            "use_residual": self.use_residual,
            "logic_op_types": self.logic_op_types,
            "arithmetic_op_types": self.arithmetic_op_types,
            "routing_initializer": keras.initializers.serialize(self.routing_initializer),
            "combination_initializer": keras.initializers.serialize(self.combination_initializer),
        })
        return config


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class LearnableNeuralCircuit(keras.layers.Layer):
    """A learnable neural circuit with configurable depth and parallel operators.

    This layer implements a neural circuit that processes 4D tensors through
    multiple depth levels, each containing parallel logic and arithmetic operators.
    The circuit features learnable routing, operation selection, and combination
    mechanisms at each depth level.

    Each depth level contains multiple parallel operators that process the input
    independently, with learnable weights controlling how the input is distributed
    to each operator and how their outputs are combined. Optional layer normalization
    can be applied after each depth level for better training stability.

    Args:
        circuit_depth: Integer, number of depth levels in the circuit. Must be positive.
        num_logic_ops_per_depth: Integer, number of logic operators per depth level.
            Must be positive.
        num_arithmetic_ops_per_depth: Integer, number of arithmetic operators per
            depth level. Must be positive.
        use_residual: Boolean, whether to use residual connections in each depth layer.
        use_layer_norm: Boolean, whether to apply layer normalization after each
            depth level.
        logic_op_types: Optional list of logic operation types to use in the
            logic operators. If None, uses all available logic operations.
        arithmetic_op_types: Optional list of arithmetic operation types to use
            in the arithmetic operators. If None, uses all available arithmetic operations.
        routing_initializer: Initializer for the routing weights in each depth layer.
        combination_initializer: Initializer for the combination weights in each depth layer.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Returns:
        A 4D tensor with the same shape as the input, containing the features
        processed through the full neural circuit.

    Raises:
        ValueError: If circuit_depth, num_logic_ops_per_depth, or
            num_arithmetic_ops_per_depth is not positive.
        ValueError: If input is not a 4D tensor.

    Example:
        >>> circuit = LearnableNeuralCircuit(
        ...     circuit_depth=3,
        ...     num_logic_ops_per_depth=2,
        ...     num_arithmetic_ops_per_depth=2
        ... )
        >>> x = np.random.rand(4, 32, 32, 64)
        >>> output = circuit(x)
        >>> print(output.shape)
        (4, 32, 32, 64)
    """

    def __init__(
            self,
            circuit_depth: int = 3,
            num_logic_ops_per_depth: int = 2,
            num_arithmetic_ops_per_depth: int = 2,
            use_residual: bool = False,
            use_layer_norm: bool = False,
            logic_op_types: Optional[List[str]] = None,
            arithmetic_op_types: Optional[List[str]] = None,
            routing_initializer: Union[str, keras.initializers.Initializer] = "random_uniform",
            combination_initializer: Union[str, keras.initializers.Initializer] = "random_uniform",
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate parameters
        if circuit_depth <= 0:
            raise ValueError("circuit_depth must be positive.")
        if num_logic_ops_per_depth <= 0:
            raise ValueError("num_logic_ops_per_depth must be positive.")
        if num_arithmetic_ops_per_depth <= 0:
            raise ValueError("num_arithmetic_ops_per_depth must be positive.")

        # Store ALL configuration parameters
        self.circuit_depth = circuit_depth
        self.num_logic_ops_per_depth = num_logic_ops_per_depth
        self.num_arithmetic_ops_per_depth = num_arithmetic_ops_per_depth
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.logic_op_types = logic_op_types
        self.arithmetic_op_types = arithmetic_op_types
        self.routing_initializer = keras.initializers.get(routing_initializer)
        self.combination_initializer = keras.initializers.get(combination_initializer)

        # Circuit layers will be initialized in build()
        self.circuit_layers = []
        self.layer_norms = []

        logger.info(
            f"LearnableNeuralCircuit initialized with depth {circuit_depth}, "
            f"{num_logic_ops_per_depth} logic ops per depth, "
            f"{num_arithmetic_ops_per_depth} arithmetic ops per depth, "
            f"use_residual: {use_residual}, use_layer_norm: {use_layer_norm}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the neural circuit layers.

        Args:
            input_shape: Shape of the input tensor. Must be a 4D shape.
        """
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(
                f"LearnableNeuralCircuit expects 4D input (batch, height, width, channels), "
                f"got shape with {len(input_shape)} dimensions: {input_shape}"
            )

        logger.info(f"Building LearnableNeuralCircuit with depth {self.circuit_depth}")

        # Create circuit depth layers and explicitly build them
        for depth in range(self.circuit_depth):
            circuit_layer = CircuitDepthLayer(
                num_logic_ops=self.num_logic_ops_per_depth,
                num_arithmetic_ops=self.num_arithmetic_ops_per_depth,
                use_residual=self.use_residual,
                logic_op_types=self.logic_op_types,
                arithmetic_op_types=self.arithmetic_op_types,
                routing_initializer=self.routing_initializer,
                combination_initializer=self.combination_initializer,
                name=f"circuit_depth_{depth}"
            )
            circuit_layer.build(input_shape)
            self.circuit_layers.append(circuit_layer)

            # Add layer normalization if enabled
            if self.use_layer_norm:
                layer_norm = keras.layers.LayerNormalization(
                    name=f"layer_norm_{depth}"
                )
                layer_norm.build(input_shape)
                self.layer_norms.append(layer_norm)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the neural circuit.

        Args:
            inputs: Input tensor of shape [batch, height, width, features]
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor after processing through the full circuit.
        """
        x = inputs

        # Process through each depth level
        for depth in range(self.circuit_depth):
            # Apply circuit layer
            x = self.circuit_layers[depth](x, training=training)

            # Apply layer normalization if enabled
            if self.use_layer_norm:
                x = self.layer_norms[depth](x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape tuple (same as input shape).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "circuit_depth": self.circuit_depth,
            "num_logic_ops_per_depth": self.num_logic_ops_per_depth,
            "num_arithmetic_ops_per_depth": self.num_arithmetic_ops_per_depth,
            "use_residual": self.use_residual,
            "use_layer_norm": self.use_layer_norm,
            "logic_op_types": self.logic_op_types,
            "arithmetic_op_types": self.arithmetic_op_types,
            "routing_initializer": keras.initializers.serialize(self.routing_initializer),
            "combination_initializer": keras.initializers.serialize(self.combination_initializer),
        })
        return config

# ---------------------------------------------------------------------