"""
Logical operations module implementing various fuzzy logic systems.

This module provides implementations of logical operations (AND, OR, NOT, etc.)
for different fuzzy logic systems that can be used in neural networks.
"""

from enum import Enum
import tensorflow as tf
from typing import Union

# ---------------------------------------------------------------------


class LogicSystem(str, Enum):
    """Supported logical systems with different characteristics."""
    LUKASIEWICZ = "lukasiewicz"  # Łukasiewicz logic
    GODEL = "godel"  # Gödel logic
    PRODUCT = "product"  # Product logic
    BOOLEAN = "boolean"  # Classical Boolean logic (using sigmoid approximation)

# ---------------------------------------------------------------------


class LogicalOperations:
    """
    Provides implementations of logical operations as differentiable functions.

    Each logical operation is implemented for different logic systems
    (Łukasiewicz, Gödel, Product, Boolean)
    with appropriate smooth approximations for gradient-based learning.
    """

    @staticmethod
    def logical_and(
            x: tf.Tensor,
            y: tf.Tensor,
            logic_system: Union[LogicSystem, str] = LogicSystem.LUKASIEWICZ,
            temperature: float = 1.0
    ) -> tf.Tensor:
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
            return tf.maximum(0.0, x + y - 1.0)
        elif logic_system == LogicSystem.GODEL:
            # Gödel t-norm: min(x, y)
            return tf.minimum(x, y)
        elif logic_system == LogicSystem.PRODUCT:
            # Product t-norm: x * y
            return x * y
        elif logic_system == LogicSystem.BOOLEAN:
            # Boolean approximation with sigmoid and temperature
            # As temperature approaches 0, this becomes a hard AND
            x_scaled = tf.sigmoid((x - 0.5) / temperature)
            y_scaled = tf.sigmoid((y - 0.5) / temperature)
            return tf.sigmoid((x_scaled * y_scaled - 0.5) / temperature)
        else:
            raise ValueError(f"Unsupported logic system: {logic_system}")

    @staticmethod
    def logical_or(
            x: tf.Tensor,
            y: tf.Tensor,
            logic_system: Union[LogicSystem, str] = LogicSystem.LUKASIEWICZ,
            temperature: float = 1.0
    ) -> tf.Tensor:
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
            return tf.minimum(1.0, x + y)
        elif logic_system == LogicSystem.GODEL:
            # Gödel t-conorm: max(x, y)
            return tf.maximum(x, y)
        elif logic_system == LogicSystem.PRODUCT:
            # Product t-conorm: x + y - x*y
            return x + y - x * y
        elif logic_system == LogicSystem.BOOLEAN:
            # Boolean approximation with sigmoid and temperature
            # As temperature approaches 0, this becomes a hard OR
            x_scaled = tf.sigmoid((x - 0.5) / temperature)
            y_scaled = tf.sigmoid((y - 0.5) / temperature)
            or_value = x_scaled + y_scaled - x_scaled * y_scaled
            return tf.sigmoid((or_value - 0.5) / temperature)
        else:
            raise ValueError(f"Unsupported logic system: {logic_system}")

    @staticmethod
    def logical_not(
            x: tf.Tensor,
            logic_system: Union[LogicSystem, str] = LogicSystem.LUKASIEWICZ,
            temperature: float = 1.0
    ) -> tf.Tensor:
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
            return tf.sigmoid((not_x - 0.5) / temperature)
        return not_x

    @staticmethod
    def logical_implies(
            x: tf.Tensor,
            y: tf.Tensor,
            logic_system: Union[LogicSystem, str] = LogicSystem.LUKASIEWICZ,
            temperature: float = 1.0
    ) -> tf.Tensor:
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
            return tf.minimum(1.0, 1.0 - x + y)
        elif logic_system == LogicSystem.GODEL:
            # Gödel implication: 1 if x ≤ y, y otherwise
            return tf.where(tf.less_equal(x, y), tf.ones_like(x), y)
        elif logic_system == LogicSystem.PRODUCT:
            # Product implication: min(1, y/x) with special handling for x=0
            safe_x = tf.maximum(x, 1e-7)  # Avoid division by zero
            implied = tf.minimum(1.0, y / safe_x)
            # When x is 0, implication is 1 regardless of y
            return tf.where(tf.equal(x, 0.0), tf.ones_like(x), implied)
        elif logic_system == LogicSystem.BOOLEAN:
            # Boolean implication: NOT x OR y
            not_x = LogicalOperations.logical_not(x, logic_system, temperature)
            return LogicalOperations.logical_or(not_x, y, logic_system, temperature)
        else:
            raise ValueError(f"Unsupported logic system: {logic_system}")

    @staticmethod
    def logical_equiv(
            x: tf.Tensor,
            y: tf.Tensor,
            logic_system: Union[LogicSystem, str] = LogicSystem.LUKASIEWICZ,
            temperature: float = 1.0
    ) -> tf.Tensor:
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
            x: tf.Tensor,
            y: tf.Tensor,
            logic_system: Union[LogicSystem, str] = LogicSystem.LUKASIEWICZ,
            temperature: float = 1.0
    ) -> tf.Tensor:
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
            x: tf.Tensor,
            y: tf.Tensor,
            logic_system: Union[LogicSystem, str] = LogicSystem.LUKASIEWICZ,
            temperature: float = 1.0
    ) -> tf.Tensor:
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
            x: tf.Tensor,
            y: tf.Tensor,
            logic_system: Union[LogicSystem, str] = LogicSystem.LUKASIEWICZ,
            temperature: float = 1.0
    ) -> tf.Tensor:
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

# ---------------------------------------------------------------------
