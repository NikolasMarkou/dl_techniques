import keras
from keras import ops
from typing import List, Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class LearnableLogicOperator(keras.layers.Layer):
    """A learnable logic operator that can perform various logical operations.

    This layer implements differentiable logic operations with learnable parameters
    to control the operation type and behavior. The layer learns to combine different
    logical operations (AND, OR, XOR, NOT, NAND, NOR) based on the input data and
    training objectives.

    The layer normalizes inputs to [0, 1] range using sigmoid activation and applies
    soft differentiable versions of logical operations. The final output is a weighted
    combination of all selected operations, where the weights are learned during training.

    Args:
        operation_types: List of operation types to choose from. Available operations:
            ['and', 'or', 'xor', 'not', 'nand', 'nor']. If None, all operations
            are included.
        use_temperature: Boolean, whether to use temperature scaling for soft operation
            selection. Temperature scaling helps control the sharpness of operation
            selection during training.
        temperature_init: Float, initial temperature value. Higher values lead to
            more uniform operation selection, lower values lead to sharper selection.
            Must be positive.
        operation_initializer: Initializer for the operation weights. If None,
            uses 'random_uniform'.
        temperature_initializer: Initializer for the temperature parameter. If None,
            uses 'constant' with the temperature_init value.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        - Single tensor: A tensor of any shape `(batch_size, ...)`
        - List of two tensors: Two tensors of the same shape `[(batch_size, ...), (batch_size, ...)]`

    Output shape:
        Same as input shape. If input is a list, output shape matches the first tensor.

    Returns:
        A tensor of the same shape as the input containing the result of the
        learnable logic operations.

    Raises:
        ValueError: If operation_types contains invalid operation names.
        ValueError: If temperature_init is not positive.
        ValueError: If input tensors have different shapes when using binary operations.

    Example:
        Single input (unary operations):
        >>> x = np.random.rand(4, 10, 10, 16)
        >>> logic_op = LearnableLogicOperator(operation_types=['not', 'and'])
        >>> y = logic_op(x)
        >>> print(y.shape)
        (4, 10, 10, 16)

        Two inputs (binary operations):
        >>> x1 = np.random.rand(4, 10, 10, 16)
        >>> x2 = np.random.rand(4, 10, 10, 16)
        >>> logic_op = LearnableLogicOperator(operation_types=['and', 'or', 'xor'])
        >>> y = logic_op([x1, x2])
        >>> print(y.shape)
        (4, 10, 10, 16)
    """

    def __init__(
            self,
            operation_types: Optional[List[str]] = None,
            use_temperature: bool = True,
            temperature_init: float = 1.0,
            operation_initializer: Union[str, keras.initializers.Initializer] = "random_uniform",
            temperature_initializer: Optional[Union[str, keras.initializers.Initializer]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate and set operation types
        if operation_types is None:
            operation_types = ['and', 'or', 'xor', 'not', 'nand', 'nor']

        valid_operations = {'and', 'or', 'xor', 'not', 'nand', 'nor'}
        invalid_ops = set(operation_types) - valid_operations
        if invalid_ops:
            raise ValueError(
                f"Invalid operation types: {invalid_ops}. "
                f"Valid operations are: {valid_operations}"
            )

        # Validate temperature initialization
        if temperature_init <= 0:
            raise ValueError("temperature_init must be positive.")

        # Store ALL configuration parameters
        self.operation_types = operation_types
        self.use_temperature = use_temperature
        self.temperature_init = temperature_init
        self.num_operations = len(operation_types)
        self.operation_initializer = keras.initializers.get(operation_initializer)

        # Set default initializer if not provided or if 'constant' is specified
        if temperature_initializer is None or temperature_initializer == "constant":
            self.temperature_initializer = keras.initializers.Constant(temperature_init)
        else:
            self.temperature_initializer = keras.initializers.get(temperature_initializer)

        # Initialize weight attributes - these will be created in build()
        self.operation_weights = None
        self.temperature = None

        logger.info(
            f"LearnableLogicOperator initialized with operations: {operation_types}, "
            f"use_temperature: {use_temperature}, temperature_init: {temperature_init}"
        )

    def build(self, input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]) -> None:
        """Build the layer weights.

        Args:
            input_shape: Shape of the input tensor(s). Can be a single shape tuple
                or a list of shape tuples for multiple inputs.
        """
        # A single shape can be a list (e.g., from serialization), but a list
        # of shapes will be a list of lists/tuples/TensorShapes.
        # We differentiate by checking if the list's elements are dimensions (int/None).
        is_list_of_shapes = (
            isinstance(input_shape, list)
            and input_shape
            and not isinstance(input_shape[0], (int, type(None)))
        )

        # Validate input shapes for binary operations
        if is_list_of_shapes:
            if len(input_shape) == 2:
                if input_shape[0] != input_shape[1]:
                    raise ValueError(
                        f"Input tensors must have the same shape for binary operations. "
                        f"Got shapes: {input_shape[0]} and {input_shape[1]}"
                    )
            elif len(input_shape) > 2:
                raise ValueError(
                    f"Expected 1 or 2 inputs, got {len(input_shape)}"
                )

        # Create learnable operation selection weights
        self.operation_weights = self.add_weight(
            name="operation_weights",
            shape=(self.num_operations,),
            initializer=self.operation_initializer,
            trainable=True,
        )

        # Create temperature parameter if enabled
        if self.use_temperature:
            self.temperature = self.add_weight(
                name="temperature",
                shape=(),
                initializer=self.temperature_initializer,
                trainable=True,
            )

        super().build(input_shape)

    def _soft_logic_and(self, x1: keras.KerasTensor, x2: keras.KerasTensor) -> keras.KerasTensor:
        """Soft differentiable AND operation.

        Args:
            x1: First input tensor
            x2: Second input tensor

        Returns:
            Result of soft AND operation
        """
        return ops.multiply(x1, x2)

    def _soft_logic_or(self, x1: keras.KerasTensor, x2: keras.KerasTensor) -> keras.KerasTensor:
        """Soft differentiable OR operation.

        Args:
            x1: First input tensor
            x2: Second input tensor

        Returns:
            Result of soft OR operation
        """
        return ops.add(ops.add(x1, x2), ops.negative(ops.multiply(x1, x2)))

    def _soft_logic_xor(self, x1: keras.KerasTensor, x2: keras.KerasTensor) -> keras.KerasTensor:
        """Soft differentiable XOR operation.

        Args:
            x1: First input tensor
            x2: Second input tensor

        Returns:
            Result of soft XOR operation
        """
        return ops.subtract(ops.add(x1, x2), ops.multiply(2.0, ops.multiply(x1, x2)))

    def _soft_logic_not(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Soft differentiable NOT operation.

        Args:
            x: Input tensor

        Returns:
            Result of soft NOT operation
        """
        return ops.subtract(1.0, x)

    def _soft_logic_nand(self, x1: keras.KerasTensor, x2: keras.KerasTensor) -> keras.KerasTensor:
        """Soft differentiable NAND operation.

        Args:
            x1: First input tensor
            x2: Second input tensor

        Returns:
            Result of soft NAND operation
        """
        return ops.subtract(1.0, ops.multiply(x1, x2))

    def _soft_logic_nor(self, x1: keras.KerasTensor, x2: keras.KerasTensor) -> keras.KerasTensor:
        """Soft differentiable NOR operation.

        Args:
            x1: First input tensor
            x2: Second input tensor

        Returns:
            Result of soft NOR operation
        """
        or_result = ops.add(ops.add(x1, x2), ops.negative(ops.multiply(x1, x2)))
        return ops.subtract(1.0, or_result)

    def call(
            self,
            inputs: Union[keras.KerasTensor, List[keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the logic operator.

        Args:
            inputs: Input tensor(s). Can be single tensor or list of two tensors.
                For single tensor input, the same tensor is used for both operands
                in binary operations.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor after applying learnable logic operations.
        """
        # Handle input parsing
        if isinstance(inputs, list):
            if len(inputs) == 2:
                x1, x2 = inputs
            elif len(inputs) == 1:
                x1 = inputs[0]
                x2 = inputs[0]  # Use same input for unary operations
            else:
                raise ValueError(f"Expected 1 or 2 inputs, got {len(inputs)}")
        else:
            x1 = inputs
            x2 = inputs

        # Normalize inputs to [0, 1] range using sigmoid
        x1 = ops.sigmoid(x1)
        x2 = ops.sigmoid(x2)

        # Compute operation selection probabilities
        if self.use_temperature:
            # Clamp temperature to prevent division by zero
            temp = ops.maximum(self.temperature, 1e-7)
            operation_probs = ops.softmax(ops.divide(self.operation_weights, temp))
        else:
            operation_probs = ops.softmax(self.operation_weights)

        # Compute all operations
        operations = []

        for i, op_type in enumerate(self.operation_types):
            if op_type == 'and':
                result = self._soft_logic_and(x1, x2)
            elif op_type == 'or':
                result = self._soft_logic_or(x1, x2)
            elif op_type == 'xor':
                result = self._soft_logic_xor(x1, x2)
            elif op_type == 'not':
                result = self._soft_logic_not(x1)
            elif op_type == 'nand':
                result = self._soft_logic_nand(x1, x2)
            elif op_type == 'nor':
                result = self._soft_logic_nor(x1, x2)
            else:
                # This should not happen due to validation in __init__
                logger.warning(f"Unknown operation type: {op_type}, using identity")
                result = x1

            operations.append(result)

        # Weighted combination of operations
        output = ops.zeros_like(x1)
        for i, op_result in enumerate(operations):
            weight = ops.expand_dims(operation_probs[i], axis=0)
            # Expand weight to match tensor dimensions
            for _ in range(len(ops.shape(x1)) - 1):
                weight = ops.expand_dims(weight, axis=-1)
            output = ops.add(output, ops.multiply(weight, op_result))

        return output

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        Args:
            input_shape: Shape of the input(s).

        Returns:
            Output shape tuple.
        """
        is_list_of_shapes = (
            isinstance(input_shape, list)
            and input_shape
            and not isinstance(input_shape[0], (int, type(None)))
        )
        if is_list_of_shapes:
            return input_shape[0]
        else:
            return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "operation_types": self.operation_types,
            "use_temperature": self.use_temperature,
            "temperature_init": self.temperature_init,
            "operation_initializer": keras.initializers.serialize(self.operation_initializer),
            "temperature_initializer": keras.initializers.serialize(self.temperature_initializer),
        })
        return config