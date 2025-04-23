"""
Fuzzy logic operations module implementing various fuzzy logic systems for Keras 3.x.

This module provides implementations of logical operations (AND, OR, NOT, etc.)
for different fuzzy logic systems that can be used in neural networks with Keras 3.x.
It includes both functional operations and a custom Keras layer implementation.
"""

import keras
from keras import ops
from enum import Enum
from typing import Union, Any

# ---------------------------------------------------------------------


class LogicSystem(str, Enum):
    """Supported logical systems with different characteristics."""
    LUKASIEWICZ = "lukasiewicz"  # Łukasiewicz logic
    GODEL = "godel"  # Gödel logic
    PRODUCT = "product"  # Product logic
    BOOLEAN = "boolean"  # Classical Boolean logic (using sigmoid approximation)


class LogicalOperations:
    """
    Provides implementations of logical operations as differentiable functions.

    Each logical operation is implemented for different logic systems
    (Łukasiewicz, Gödel, Product, Boolean)
    with appropriate smooth approximations for gradient-based learning.
    """

    @staticmethod
    def logical_and(
            x: Any,
            y: Any,
            logic_system: Union[LogicSystem, str] = LogicSystem.LUKASIEWICZ,
            temperature: float = 1.0
    ) -> Any:
        """
        Implements logical AND operation.

        Args:
            x: First input tensor with values in [0, 1]
            y: Second input tensor with values in [0, 1]
            logic_system: Type of logic system to use
            temperature: Temperature parameter for controlling output sharpness

        Returns:
            Tensor representing the truth value of x AND y
        """
        if logic_system == LogicSystem.LUKASIEWICZ:
            # Łukasiewicz t-norm: max(0, x + y - 1)
            return ops.maximum(0.0, x + y - 1.0)
        elif logic_system == LogicSystem.GODEL:
            # Gödel t-norm: min(x, y)
            return ops.minimum(x, y)
        elif logic_system == LogicSystem.PRODUCT:
            # Product t-norm: x * y
            return x * y
        elif logic_system == LogicSystem.BOOLEAN:
            # Boolean approximation with sigmoid and temperature
            # As temperature approaches 0, this becomes a hard AND
            x_scaled = keras.activations.sigmoid((x - 0.5) / temperature)
            y_scaled = keras.activations.sigmoid((y - 0.5) / temperature)
            return keras.activations.sigmoid((x_scaled * y_scaled - 0.5) / temperature)
        else:
            raise ValueError(f"Unsupported logic system: {logic_system}")

    @staticmethod
    def logical_or(
            x: Any,
            y: Any,
            logic_system: Union[LogicSystem, str] = LogicSystem.LUKASIEWICZ,
            temperature: float = 1.0
    ) -> Any:
        """
        Implements logical OR operation.

        Args:
            x: First input tensor with values in [0, 1]
            y: Second input tensor with values in [0, 1]
            logic_system: Type of logic system to use
            temperature: Temperature parameter for controlling output sharpness

        Returns:
            Tensor representing the truth value of x OR y
        """
        if logic_system == LogicSystem.LUKASIEWICZ:
            # Łukasiewicz t-conorm: min(1, x + y)
            return ops.minimum(1.0, x + y)
        elif logic_system == LogicSystem.GODEL:
            # Gödel t-conorm: max(x, y)
            return ops.maximum(x, y)
        elif logic_system == LogicSystem.PRODUCT:
            # Product t-conorm: x + y - x*y
            return x + y - x * y
        elif logic_system == LogicSystem.BOOLEAN:
            # Boolean approximation with sigmoid and temperature
            # As temperature approaches 0, this becomes a hard OR
            x_scaled = keras.activations.sigmoid((x - 0.5) / temperature)
            y_scaled = keras.activations.sigmoid((y - 0.5) / temperature)
            or_value = x_scaled + y_scaled - x_scaled * y_scaled
            return keras.activations.sigmoid((or_value - 0.5) / temperature)
        else:
            raise ValueError(f"Unsupported logic system: {logic_system}")

    @staticmethod
    def logical_not(
            x: Any,
            logic_system: Union[LogicSystem, str] = LogicSystem.LUKASIEWICZ,
            temperature: float = 1.0
    ) -> Any:
        """
        Implements logical NOT operation.

        Args:
            x: Input tensor with values in [0, 1]
            logic_system: Type of logic system to use
            temperature: Temperature parameter for controlling output sharpness

        Returns:
            Tensor representing the truth value of NOT x
        """
        # NOT operation is the same for Łukasiewicz, Gödel, and Product logics
        not_x = 1.0 - x

        if logic_system == LogicSystem.BOOLEAN:
            # Apply temperature scaling for sharper transitions
            return keras.activations.sigmoid((not_x - 0.5) / temperature)
        return not_x

    @staticmethod
    def logical_implies(
            x: Any,
            y: Any,
            logic_system: Union[LogicSystem, str] = LogicSystem.LUKASIEWICZ,
            temperature: float = 1.0
    ) -> Any:
        """
        Implements logical IMPLIES operation.

        Args:
            x: First input tensor with values in [0, 1]
            y: Second input tensor with values in [0, 1]
            logic_system: Type of logic system to use
            temperature: Temperature parameter for controlling output sharpness

        Returns:
            Tensor representing the truth value of x IMPLIES y
        """
        if logic_system == LogicSystem.LUKASIEWICZ:
            # Łukasiewicz implication: min(1, 1 - x + y)
            return ops.minimum(1.0, 1.0 - x + y)
        elif logic_system == LogicSystem.GODEL:
            # Gödel implication: 1 if x ≤ y, y otherwise
            return ops.where(ops.less_equal(x, y), ops.ones_like(x), y)
        elif logic_system == LogicSystem.PRODUCT:
            # Product implication: min(1, y/x) with special handling for x=0
            safe_x = ops.maximum(x, 1e-7)  # Avoid division by zero
            implied = ops.minimum(1.0, y / safe_x)
            # When x is 0, implication is 1 regardless of y
            return ops.where(ops.equal(x, 0.0), ops.ones_like(x), implied)
        elif logic_system == LogicSystem.BOOLEAN:
            # Boolean implication: NOT x OR y
            not_x = LogicalOperations.logical_not(x, logic_system, temperature)
            return LogicalOperations.logical_or(not_x, y, logic_system, temperature)
        else:
            raise ValueError(f"Unsupported logic system: {logic_system}")

    @staticmethod
    def logical_equiv(
            x: Any,
            y: Any,
            logic_system: Union[LogicSystem, str] = LogicSystem.LUKASIEWICZ,
            temperature: float = 1.0
    ) -> Any:
        """
        Implements logical EQUIVALENCE operation (bidirectional implication).

        Args:
            x: First input tensor with values in [0, 1]
            y: Second input tensor with values in [0, 1]
            logic_system: Type of logic system to use
            temperature: Temperature parameter for controlling output sharpness

        Returns:
            Tensor representing the truth value of x EQUIV y
        """
        implies_xy = LogicalOperations.logical_implies(x, y, logic_system, temperature)
        implies_yx = LogicalOperations.logical_implies(y, x, logic_system, temperature)
        return LogicalOperations.logical_and(implies_xy, implies_yx, logic_system, temperature)

    @staticmethod
    def logical_xor(
            x: Any,
            y: Any,
            logic_system: Union[LogicSystem, str] = LogicSystem.LUKASIEWICZ,
            temperature: float = 1.0
    ) -> Any:
        """
        Implements logical XOR operation.

        Args:
            x: First input tensor with values in [0, 1]
            y: Second input tensor with values in [0, 1]
            logic_system: Type of logic system to use
            temperature: Temperature parameter for controlling output sharpness

        Returns:
            Tensor representing the truth value of x XOR y
        """
        # XOR is defined as (x OR y) AND NOT (x AND y)
        or_result = LogicalOperations.logical_or(x, y, logic_system, temperature)
        and_result = LogicalOperations.logical_and(x, y, logic_system, temperature)
        not_and = LogicalOperations.logical_not(and_result, logic_system, temperature)
        return LogicalOperations.logical_and(or_result, not_and, logic_system, temperature)

    @staticmethod
    def logical_nand(
            x: Any,
            y: Any,
            logic_system: Union[LogicSystem, str] = LogicSystem.LUKASIEWICZ,
            temperature: float = 1.0
    ) -> Any:
        """
        Implements logical NAND operation.

        Args:
            x: First input tensor with values in [0, 1]
            y: Second input tensor with values in [0, 1]
            logic_system: Type of logic system to use
            temperature: Temperature parameter for controlling output sharpness

        Returns:
            Tensor representing the truth value of x NAND y
        """
        # NAND is NOT (x AND y)
        and_result = LogicalOperations.logical_and(x, y, logic_system, temperature)
        return LogicalOperations.logical_not(and_result, logic_system, temperature)

    @staticmethod
    def logical_nor(
            x: Any,
            y: Any,
            logic_system: Union[LogicSystem, str] = LogicSystem.LUKASIEWICZ,
            temperature: float = 1.0
    ) -> Any:
        """
        Implements logical NOR operation.

        Args:
            x: First input tensor with values in [0, 1]
            y: Second input tensor with values in [0, 1]
            logic_system: Type of logic system to use
            temperature: Temperature parameter for controlling output sharpness

        Returns:
            Tensor representing the truth value of x NOR y
        """
        # NOR is NOT (x OR y)
        or_result = LogicalOperations.logical_or(x, y, logic_system, temperature)
        return LogicalOperations.logical_not(or_result, logic_system, temperature)


class FuzzyLogicLayer(keras.layers.Layer):
    """
    Custom Keras layer implementing fuzzy logic operations.

    This layer applies fuzzy logic operations to input tensors
    based on the specified operation type and logic system.

    Args:
        operation: String specifying the logical operation
            ('and', 'or', 'not', 'implies', 'equiv', 'xor', 'nand', 'nor')
        logic_system: String or LogicSystem enum specifying the fuzzy logic system
            ('lukasiewicz', 'godel', 'product', 'boolean')
        temperature: Float controlling the sharpness of transitions for boolean approximations
        **kwargs: Additional keyword arguments to pass to the Layer base class

    Input shape:
        For binary operations ('and', 'or', 'implies', 'equiv', 'xor', 'nand', 'nor'):
            List of two tensors [x, y], each with shape (batch_size, ...) containing
            values in range [0, 1]
        For unary operations ('not'):
            A tensor with shape (batch_size, ...) containing values in range [0, 1]

    Output shape:
        Tensor with the same shape as the input tensor(s), containing
        values in range [0, 1]

    Examples:
        >>> # Binary operation (AND)
        >>> x1 = keras.random.uniform((4, 5), 0, 1)
        >>> x2 = keras.random.uniform((4, 5), 0, 1)
        >>> fuzzy_and = FuzzyLogicLayer('and', 'product')
        >>> result = fuzzy_and([x1, x2])
        >>> print(result.shape)
        (4, 5)

        >>> # Unary operation (NOT)
        >>> x = keras.random.uniform((4, 5), 0, 1)
        >>> fuzzy_not = FuzzyLogicLayer('not', 'lukasiewicz')
        >>> result = fuzzy_not(x)
        >>> print(result.shape)
        (4, 5)
    """

    def __init__(
        self,
        operation: str = 'and',
        logic_system: Union[LogicSystem, str] = LogicSystem.LUKASIEWICZ,
        temperature: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.operation = operation.lower()
        self.logic_system = logic_system
        self.temperature = temperature

        # Validate operation
        valid_operations = ['and', 'or', 'not', 'implies', 'equiv', 'xor', 'nand', 'nor']
        if self.operation not in valid_operations:
            raise ValueError(f"Invalid operation: {operation}. Must be one of {valid_operations}")

        # Convert string to enum if needed
        if isinstance(logic_system, str):
            try:
                self.logic_system = LogicSystem(logic_system.lower())
            except ValueError:
                valid_systems = [system.value for system in LogicSystem]
                raise ValueError(
                    f"Invalid logic system: {logic_system}. Must be one of {valid_systems}"
                )

    def call(self, inputs, training=None):
        """
        Forward pass logic.

        Args:
            inputs: Input tensor or list of input tensors. For binary operations,
                this should be a list of two tensors [x, y]. For unary operations ('not'),
                this should be a single tensor.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode (not used in this layer).

        Returns:
            Tensor with the result of applying the fuzzy logic operation
        """
        # Create an instance of LogicalOperations
        logic_ops = LogicalOperations()

        if self.operation == 'not':
            # NOT is a unary operation
            if isinstance(inputs, list) and len(inputs) > 1:
                raise ValueError("'not' operation requires exactly one input")
            x = inputs[0] if isinstance(inputs, list) else inputs
            return logic_ops.logical_not(x, self.logic_system, self.temperature)

        # All other operations are binary
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError(f"'{self.operation}' operation requires exactly two inputs")
        x, y = inputs

        if self.operation == 'and':
            return logic_ops.logical_and(x, y, self.logic_system, self.temperature)
        elif self.operation == 'or':
            return logic_ops.logical_or(x, y, self.logic_system, self.temperature)
        elif self.operation == 'implies':
            return logic_ops.logical_implies(x, y, self.logic_system, self.temperature)
        elif self.operation == 'equiv':
            return logic_ops.logical_equiv(x, y, self.logic_system, self.temperature)
        elif self.operation == 'xor':
            return logic_ops.logical_xor(x, y, self.logic_system, self.temperature)
        elif self.operation == 'nand':
            return logic_ops.logical_nand(x, y, self.logic_system, self.temperature)
        elif self.operation == 'nor':
            return logic_ops.logical_nor(x, y, self.logic_system, self.temperature)
        return x, y

    def get_config(self):
        """Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "operation": self.operation,
            "logic_system": self.logic_system.value if isinstance(self.logic_system, LogicSystem) else self.logic_system,
            "temperature": self.temperature
        })
        return config