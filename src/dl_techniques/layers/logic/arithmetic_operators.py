import keras
from keras import ops
from typing import List, Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class LearnableArithmeticOperator(keras.layers.Layer):
    """A learnable arithmetic operator that can perform various arithmetic operations.

    This layer implements differentiable arithmetic operations with learnable parameters
    to control the operation type and behavior. The layer learns to combine different
    arithmetic operations (add, multiply, subtract, divide, power, max, min) based on
    the input data and training objectives.

    The layer applies a weighted combination of all selected operations, where the weights
    are learned during training. A scaling factor is also learned to control the magnitude
    of the output, which helps with numerical stability and gradient flow.

    Args:
        operation_types: List of operation types to choose from. Available operations:
            ['add', 'multiply', 'subtract', 'divide', 'power', 'max', 'min']. If None,
            all operations are included.
        use_temperature: Boolean, whether to use temperature scaling for soft operation
            selection. Temperature scaling helps control the sharpness of operation
            selection during training.
        temperature_init: Float, initial temperature value. Higher values lead to
            more uniform operation selection, lower values lead to sharper selection.
            Must be positive.
        use_scaling: Boolean, whether to use a learnable scaling factor for the output.
            The scaling factor helps with numerical stability and gradient flow.
        scaling_init: Float, initial scaling factor value. Should be positive.
        operation_initializer: Initializer for the operation weights. If None,
            uses 'random_uniform'.
        temperature_initializer: Initializer for the temperature parameter. If None,
            uses 'constant' with the temperature_init value.
        scaling_initializer: Initializer for the scaling factor. If None,
            uses 'constant' with the scaling_init value.
        epsilon: Float, small constant for numerical stability in division operations.
        power_clip_range: Tuple of two floats, (min_base, max_base) for clipping the
            base in power operations to ensure numerical stability.
        exponent_clip_range: Tuple of two floats, (min_exp, max_exp) for clipping the
            exponent in power operations.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        - Single tensor: A tensor of any shape `(batch_size, ...)`
        - List of two tensors: Two tensors of the same shape `[(batch_size, ...), (batch_size, ...)]`

    Output shape:
        Same as input shape. If input is a list, output shape matches the first tensor.

    Returns:
        A tensor of the same shape as the input containing the result of the
        learnable arithmetic operations.

    Raises:
        ValueError: If operation_types contains invalid operation names.
        ValueError: If temperature_init is not positive.
        ValueError: If scaling_init is not positive.
        ValueError: If epsilon is not positive.
        ValueError: If input tensors have different shapes when using binary operations.

    Example:
        Single input:
        >>> x = np.random.rand(4, 10, 10, 16)
        >>> arith_op = LearnableArithmeticOperator(operation_types=['add', 'multiply'])
        >>> y = arith_op(x)
        >>> print(y.shape)
        (4, 10, 10, 16)

        Two inputs:
        >>> x1 = np.random.rand(4, 10, 10, 16)
        >>> x2 = np.random.rand(4, 10, 10, 16)
        >>> arith_op = LearnableArithmeticOperator(operation_types=['add', 'multiply', 'max'])
        >>> y = arith_op([x1, x2])
        >>> print(y.shape)
        (4, 10, 10, 16)
    """

    def __init__(
            self,
            operation_types: Optional[List[str]] = None,
            use_temperature: bool = True,
            temperature_init: float = 1.0,
            use_scaling: bool = True,
            scaling_init: float = 1.0,
            operation_initializer: Union[str, keras.initializers.Initializer] = "random_uniform",
            temperature_initializer: Union[str, keras.initializers.Initializer] = keras.initializers.Constant(1.0),
            scaling_initializer: Union[str, keras.initializers.Initializer] = keras.initializers.Constant(1.0),
            epsilon: float = 1e-7,
            power_clip_range: Tuple[float, float] = (1e-7, 10.0),
            exponent_clip_range: Tuple[float, float] = (-2.0, 2.0),
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate and set operation types
        if operation_types is None:
            operation_types = ['add', 'multiply', 'subtract', 'divide', 'power', 'max', 'min']

        valid_operations = {'add', 'multiply', 'subtract', 'divide', 'power', 'max', 'min'}
        invalid_ops = set(operation_types) - valid_operations
        if invalid_ops:
            raise ValueError(
                f"Invalid operation types: {invalid_ops}. "
                f"Valid operations are: {valid_operations}"
            )

        # Validate parameters
        if temperature_init <= 0:
            raise ValueError("temperature_init must be positive.")

        if scaling_init <= 0:
            raise ValueError("scaling_init must be positive.")

        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        if power_clip_range[0] <= 0 or power_clip_range[1] <= power_clip_range[0]:
            raise ValueError("power_clip_range must be (min, max) with 0 < min < max.")

        if exponent_clip_range[1] <= exponent_clip_range[0]:
            raise ValueError("exponent_clip_range must be (min, max) with min < max.")

        self.operation_types = operation_types
        self.use_temperature = use_temperature
        self.temperature_init = temperature_init
        self.use_scaling = use_scaling
        self.scaling_init = scaling_init
        self.num_operations = len(operation_types)
        self.operation_initializer = keras.initializers.get(operation_initializer)
        self.temperature_initializer = keras.initializers.get(temperature_initializer)
        self.scaling_initializer = keras.initializers.get(scaling_initializer)
        self.epsilon = epsilon
        self.power_clip_range = power_clip_range
        self.exponent_clip_range = exponent_clip_range

        # Initialize weight attributes
        self.operation_weights = None
        self.temperature = None
        self.scaling_factor = None
        self._build_input_shape = None

        logger.info(
            f"LearnableArithmeticOperator initialized with operations: {operation_types}, "
            f"use_temperature: {use_temperature}, temperature_init: {temperature_init}, "
            f"use_scaling: {use_scaling}, scaling_init: {scaling_init}"
        )

    def build(self, input_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]]) -> None:
        """Build the layer weights.

        Args:
            input_shape: Shape of the input tensor(s). Can be a single shape tuple
                or a list of shape tuples for multiple inputs.
        """
        self._build_input_shape = input_shape

        # Validate input shapes for binary operations
        if isinstance(input_shape, list):
            if len(input_shape) == 2:
                if input_shape[0] != input_shape[1]:
                    raise ValueError(
                        f"Input tensors must have the same shape for binary operations. "
                        f"Got shapes: {input_shape[0]} and {input_shape[1]}"
                    )
            elif len(input_shape) > 2:
                raise ValueError(
                    f"LearnableArithmeticOperator supports at most 2 inputs, got {len(input_shape)}"
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
            temp_initializer = keras.initializers.Constant(self.temperature_init)
            self.temperature = self.add_weight(
                name="temperature",
                shape=(),
                initializer=temp_initializer,
                trainable=True,
            )

        # Create scaling factor if enabled
        if self.use_scaling:
            scaling_initializer = keras.initializers.Constant(self.scaling_init)
            self.scaling_factor = self.add_weight(
                name="scaling_factor",
                shape=(),
                initializer=scaling_initializer,
                trainable=True,
            )

        super().build(input_shape)

    def _safe_divide(self, x1: keras.KerasTensor, x2: keras.KerasTensor) -> keras.KerasTensor:
        """
        An improved epsilon-based safe division.

        This method modifies the denominator to avoid zero by ensuring its
        magnitude is at least epsilon.

        Args:
            x1: Numerator tensor.
            x2: Denominator tensor.
            epsilon: A small float to add to the denominator's magnitude.

        Returns:
            Result of the division.
        """
        # Get the sign of the denominator, treating 0 as positive.
        sign_x2 = ops.sign(x2)
        # If x2 is 0, ops.sign(x2) is 0. We want the sign to be 1 in this case
        # so that we add epsilon.
        sign_x2 = ops.where(ops.equal(sign_x2, 0.), ops.ones_like(sign_x2), sign_x2)

        # New denominator has its magnitude clamped to be at least epsilon.
        safe_x2 = sign_x2 * ops.maximum(ops.abs(x2), 1-6)

        return ops.divide(x1, safe_x2)

    def _safe_power(self, x1: keras.KerasTensor, x2: keras.KerasTensor) -> keras.KerasTensor:
        """Safe power operation with clipping for numerical stability.

        Args:
            x1: Base tensor
            x2: Exponent tensor

        Returns:
            Result of safe power operation
        """
        # Clip base to prevent numerical issues
        x1_safe = ops.clip(ops.abs(x1), self.power_clip_range[0], self.power_clip_range[1])
        # Clip exponent to prevent overflow
        x2_safe = ops.clip(x2, self.exponent_clip_range[0], self.exponent_clip_range[1])
        return ops.power(x1_safe, x2_safe)

    def _soft_max(self, x1: keras.KerasTensor, x2: keras.KerasTensor) -> keras.KerasTensor:
        """Element-wise maximum operation.

        Args:
            x1: First input tensor
            x2: Second input tensor

        Returns:
            Element-wise maximum of the inputs
        """
        return ops.maximum(x1, x2)

    def _soft_min(self, x1: keras.KerasTensor, x2: keras.KerasTensor) -> keras.KerasTensor:
        """Element-wise minimum operation.

        Args:
            x1: First input tensor
            x2: Second input tensor

        Returns:
            Element-wise minimum of the inputs
        """
        return ops.minimum(x1, x2)

    def call(
            self,
            inputs: Union[keras.KerasTensor, List[keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the arithmetic operator.

        Args:
            inputs: Input tensor(s). Can be single tensor or list of two tensors.
                For single tensor input, the same tensor is used for both operands
                in binary operations.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor after applying learnable arithmetic operations.
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

        # Compute operation selection probabilities
        if self.use_temperature:
            # Clamp temperature to prevent division by zero
            temp = ops.maximum(self.temperature, 0.1)
            operation_probs = ops.softmax(ops.divide(self.operation_weights, temp))
        else:
            operation_probs = ops.softmax(self.operation_weights)

        # Compute all operations
        operations = []

        for i, op_type in enumerate(self.operation_types):
            if op_type == 'add':
                result = ops.add(x1, x2)
            elif op_type == 'multiply':
                result = ops.multiply(x1, x2)
            elif op_type == 'subtract':
                result = ops.subtract(x1, x2)
            elif op_type == 'divide':
                result = self._safe_divide(x1, x2)
            elif op_type == 'power':
                result = self._safe_power(x1, x2)
            elif op_type == 'max':
                result = self._soft_max(x1, x2)
            elif op_type == 'min':
                result = self._soft_min(x1, x2)
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

        # Apply scaling factor if enabled
        if self.use_scaling:
            # Clamp scaling factor to prevent numerical issues
            scale = ops.maximum(ops.abs(self.scaling_factor), 1e-7)
            output = ops.multiply(output, scale)

        return output

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]]
    ) -> Tuple[int, ...]:
        """Compute output shape.

        Args:
            input_shape: Shape of the input(s).

        Returns:
            Output shape tuple.
        """
        if isinstance(input_shape, list):
            # Convert to list for manipulation, then back to tuple
            shape_list = list(input_shape[0])
            return tuple(shape_list)
        else:
            # Convert to list for manipulation, then back to tuple
            shape_list = list(input_shape)
            return tuple(shape_list)

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
            "use_scaling": self.use_scaling,
            "scaling_init": self.scaling_init,
            "operation_initializer": keras.initializers.serialize(self.operation_initializer),
            "temperature_initializer": keras.initializers.serialize(self.temperature_initializer),
            "scaling_initializer": keras.initializers.serialize(self.scaling_initializer),
            "epsilon": self.epsilon,
            "power_clip_range": self.power_clip_range,
            "exponent_clip_range": self.exponent_clip_range,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a configuration.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
