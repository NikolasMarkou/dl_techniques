"""
Implementation of advanced logic gates based on fuzzy logic systems.

This module provides implementations of logical gates (AND, OR, NOT, etc.)
as Keras layers that support different fuzzy logic systems, truth bounds,
and bidirectional reasoning.
"""

import keras
from typing import Optional, Union

from .advanced_logic_gate import AdvancedLogicGateLayer
from .logical_operations import LogicSystem, LogicalOperations


class FuzzyANDGateLayer(AdvancedLogicGateLayer):
    """
    AND gate layer supporting various fuzzy logic systems.

    The AND operation returns true only when all inputs are true.

    Args:
        logic_system: Logic system to use (Łukasiewicz, Gödel, Product, Boolean)
        use_bounds: Whether to use upper and lower bounds for truth values
        trainable_weights: Whether input weights should be trainable
        initial_weight: Initial value for trainable weights
        temperature: Parameter controlling sharpness of logical transitions
        use_bias: Whether to use a bias vector
        kernel_initializer: Initializer for kernel weights
        bias_initializer: Initializer for bias vector
        kernel_regularizer: Regularizer for kernel weights
        bias_regularizer: Regularizer for bias vector
        activity_regularizer: Regularizer for layer output
        kernel_constraint: Constraint for kernel weights
        bias_constraint: Constraint for bias vector
    """

    def __init__(
            self,
            logic_system: Union[LogicSystem, str] = LogicSystem.BOOLEAN,
            use_bounds: bool = False,
            trainable_weights: bool = False,
            initial_weight: float = 1.0,
            temperature: float = 0.1,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_constraint: Optional[keras.constraints.Constraint] = None,
            bias_constraint: Optional[keras.constraints.Constraint] = None,
            **kwargs
    ):
        """Initialize the fuzzy AND gate layer."""
        super().__init__(
            operation=LogicalOperations.logical_and,
            logic_system=logic_system,
            use_bounds=use_bounds,
            trainable_weights=trainable_weights,
            initial_weight=initial_weight,
            temperature=temperature,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )


class FuzzyORGateLayer(AdvancedLogicGateLayer):
    """
    OR gate layer supporting various fuzzy logic systems.

    The OR operation returns true when at least one input is true.

    Args:
        logic_system: Logic system to use (Łukasiewicz, Gödel, Product, Boolean)
        use_bounds: Whether to use upper and lower bounds for truth values
        trainable_weights: Whether input weights should be trainable
        initial_weight: Initial value for trainable weights
        temperature: Parameter controlling sharpness of logical transitions
        use_bias: Whether to use a bias vector
        kernel_initializer: Initializer for kernel weights
        bias_initializer: Initializer for bias vector
        kernel_regularizer: Regularizer for kernel weights
        bias_regularizer: Regularizer for bias vector
        activity_regularizer: Regularizer for layer output
        kernel_constraint: Constraint for kernel weights
        bias_constraint: Constraint for bias vector
    """

    def __init__(
            self,
            logic_system: Union[LogicSystem, str] = LogicSystem.BOOLEAN,
            use_bounds: bool = False,
            trainable_weights: bool = False,
            initial_weight: float = 1.0,
            temperature: float = 0.1,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_constraint: Optional[keras.constraints.Constraint] = None,
            bias_constraint: Optional[keras.constraints.Constraint] = None,
            **kwargs
    ):
        """Initialize the fuzzy OR gate layer."""
        super().__init__(
            operation=LogicalOperations.logical_or,
            logic_system=logic_system,
            use_bounds=use_bounds,
            trainable_weights=trainable_weights,
            initial_weight=initial_weight,
            temperature=temperature,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )


class FuzzyNOTGateLayer(AdvancedLogicGateLayer):
    """
    NOT gate layer supporting various fuzzy logic systems.

    The NOT operation negates the input truth value.

    Args:
        logic_system: Logic system to use (Łukasiewicz, Gödel, Product, Boolean)
        use_bounds: Whether to use upper and lower bounds for truth values
        temperature: Parameter controlling sharpness of logical transitions
        use_bias: Whether to use a bias vector
        kernel_initializer: Initializer for kernel weights
        bias_initializer: Initializer for bias vector
        kernel_regularizer: Regularizer for kernel weights
        bias_regularizer: Regularizer for bias vector
        activity_regularizer: Regularizer for layer output
        kernel_constraint: Constraint for kernel weights
        bias_constraint: Constraint for bias vector
    """

    def __init__(
            self,
            logic_system: Union[LogicSystem, str] = LogicSystem.BOOLEAN,
            use_bounds: bool = False,
            temperature: float = 0.1,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_constraint: Optional[keras.constraints.Constraint] = None,
            bias_constraint: Optional[keras.constraints.Constraint] = None,
            **kwargs
    ):
        """Initialize the fuzzy NOT gate layer."""
        super().__init__(
            operation=LogicalOperations.logical_not,
            logic_system=logic_system,
            use_bounds=use_bounds,
            trainable_weights=False,  # NOT gate doesn't need trainable weights
            initial_weight=1.0,
            temperature=temperature,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

    def build(self, input_shape):
        """
        Build the layer based on input shape.

        Args:
            input_shape: Shape of the input tensor
        """
        # Validate that we have a single input
        if isinstance(input_shape, list):
            if len(input_shape) > 1:
                raise ValueError("NOTGateLayer accepts only one input.")
            input_shape = input_shape[0]

        # Ensure we're operating on a single feature dimension
        if input_shape[-1] != 1:
            raise ValueError(
                f"Expected input with 1 feature, got {input_shape[-1]}. "
                "Consider slicing your input tensor."
            )

        super().build(input_shape)


class FuzzyXORGateLayer(AdvancedLogicGateLayer):
    """
    XOR gate layer supporting various fuzzy logic systems.

    The XOR operation returns true when an odd number of inputs are true.

    Args:
        logic_system: Logic system to use (Łukasiewicz, Gödel, Product, Boolean)
        use_bounds: Whether to use upper and lower bounds for truth values
        trainable_weights: Whether input weights should be trainable
        initial_weight: Initial value for trainable weights
        temperature: Parameter controlling sharpness of logical transitions
        use_bias: Whether to use a bias vector
        kernel_initializer: Initializer for kernel weights
        bias_initializer: Initializer for bias vector
        kernel_regularizer: Regularizer for kernel weights
        bias_regularizer: Regularizer for bias vector
        activity_regularizer: Regularizer for layer output
        kernel_constraint: Constraint for kernel weights
        bias_constraint: Constraint for bias vector
    """

    def __init__(
            self,
            logic_system: Union[LogicSystem, str] = LogicSystem.BOOLEAN,
            use_bounds: bool = False,
            trainable_weights: bool = False,
            initial_weight: float = 1.0,
            temperature: float = 0.1,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_constraint: Optional[keras.constraints.Constraint] = None,
            bias_constraint: Optional[keras.constraints.Constraint] = None,
            **kwargs
    ):
        """Initialize the fuzzy XOR gate layer."""
        super().__init__(
            operation=LogicalOperations.logical_xor,
            logic_system=logic_system,
            use_bounds=use_bounds,
            trainable_weights=trainable_weights,
            initial_weight=initial_weight,
            temperature=temperature,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

    def build(self, input_shape):
        """
        Build the layer based on input shape.

        Args:
            input_shape: Shape of the input tensor or list of tensors
        """
        # Ensure we have exactly 2 inputs for XOR
        if isinstance(input_shape, list):
            if len(input_shape) != 2:
                raise ValueError("XORGateLayer supports exactly 2 inputs.")
        elif input_shape[-1] != 2:
            raise ValueError(
                f"Expected input with 2 features, got {input_shape[-1]}. "
                "Consider reshaping your input."
            )

        super().build(input_shape)


class FuzzyNANDGateLayer(AdvancedLogicGateLayer):
    """
    NAND gate layer supporting various fuzzy logic systems.

    The NAND operation is the negation of the AND operation.

    Args:
        logic_system: Logic system to use (Łukasiewicz, Gödel, Product, Boolean)
        use_bounds: Whether to use upper and lower bounds for truth values
        trainable_weights: Whether input weights should be trainable
        initial_weight: Initial value for trainable weights
        temperature: Parameter controlling sharpness of logical transitions
        use_bias: Whether to use a bias vector
        kernel_initializer: Initializer for kernel weights
        bias_initializer: Initializer for bias vector
        kernel_regularizer: Regularizer for kernel weights
        bias_regularizer: Regularizer for bias vector
        activity_regularizer: Regularizer for layer output
        kernel_constraint: Constraint for kernel weights
        bias_constraint: Constraint for bias vector
    """

    def __init__(
            self,
            logic_system: Union[LogicSystem, str] = LogicSystem.BOOLEAN,
            use_bounds: bool = False,
            trainable_weights: bool = False,
            initial_weight: float = 1.0,
            temperature: float = 0.1,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_constraint: Optional[keras.constraints.Constraint] = None,
            bias_constraint: Optional[keras.constraints.Constraint] = None,
            **kwargs
    ):
        """Initialize the fuzzy NAND gate layer."""
        super().__init__(
            operation=LogicalOperations.logical_nand,
            logic_system=logic_system,
            use_bounds=use_bounds,
            trainable_weights=trainable_weights,
            initial_weight=initial_weight,
            temperature=temperature,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )


class FuzzyNORGateLayer(AdvancedLogicGateLayer):
    """
    NOR gate layer supporting various fuzzy logic systems.

    The NOR operation is the negation of the OR operation.

    Args:
        logic_system: Logic system to use (Łukasiewicz, Gödel, Product, Boolean)
        use_bounds: Whether to use upper and lower bounds for truth values
        trainable_weights: Whether input weights should be trainable
        initial_weight: Initial value for trainable weights
        temperature: Parameter controlling sharpness of logical transitions
        use_bias: Whether to use a bias vector
        kernel_initializer: Initializer for kernel weights
        bias_initializer: Initializer for bias vector
        kernel_regularizer: Regularizer for kernel weights
        bias_regularizer: Regularizer for bias vector
        activity_regularizer: Regularizer for layer output
        kernel_constraint: Constraint for kernel weights
        bias_constraint: Constraint for bias vector
    """

    def __init__(
            self,
            logic_system: Union[LogicSystem, str] = LogicSystem.BOOLEAN,
            use_bounds: bool = False,
            trainable_weights: bool = False,
            initial_weight: float = 1.0,
            temperature: float = 0.1,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_constraint: Optional[keras.constraints.Constraint] = None,
            bias_constraint: Optional[keras.constraints.Constraint] = None,
            **kwargs
    ):
        """Initialize the fuzzy NOR gate layer."""
        super().__init__(
            operation=LogicalOperations.logical_nor,
            logic_system=logic_system,
            use_bounds=use_bounds,
            trainable_weights=trainable_weights,
            initial_weight=initial_weight,
            temperature=temperature,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )


class FuzzyImpliesGateLayer(AdvancedLogicGateLayer):
    """
    IMPLIES gate layer supporting various fuzzy logic systems.

    The IMPLIES operation (→) models logical implication.

    Args:
        logic_system: Logic system to use (Łukasiewicz, Gödel, Product, Boolean)
        use_bounds: Whether to use upper and lower bounds for truth values
        trainable_weights: Whether input weights should be trainable
        initial_weight: Initial value for trainable weights
        temperature: Parameter controlling sharpness of logical transitions
        use_bias: Whether to use a bias vector
        kernel_initializer: Initializer for kernel weights
        bias_initializer: Initializer for bias vector
        kernel_regularizer: Regularizer for kernel weights
        bias_regularizer: Regularizer for bias vector
        activity_regularizer: Regularizer for layer output
        kernel_constraint: Constraint for kernel weights
        bias_constraint: Constraint for bias vector
    """

    def __init__(
            self,
            logic_system: Union[LogicSystem, str] = LogicSystem.BOOLEAN,
            use_bounds: bool = False,
            trainable_weights: bool = False,
            initial_weight: float = 1.0,
            temperature: float = 0.1,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_constraint: Optional[keras.constraints.Constraint] = None,
            bias_constraint: Optional[keras.constraints.Constraint] = None,
            **kwargs
    ):
        """Initialize the fuzzy IMPLIES gate layer."""
        super().__init__(
            operation=LogicalOperations.logical_implies,
            logic_system=logic_system,
            use_bounds=use_bounds,
            trainable_weights=trainable_weights,
            initial_weight=initial_weight,
            temperature=temperature,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

    def build(self, input_shape):
        """
        Build the layer based on input shape.

        Args:
            input_shape: Shape of the input tensor or list of tensors
        """
        # Ensure we have exactly 2 inputs for IMPLIES
        if isinstance(input_shape, list):
            if len(input_shape) != 2:
                raise ValueError("ImpliesGateLayer supports exactly 2 inputs.")
        elif input_shape[-1] != 2:
            raise ValueError(
                f"Expected input with 2 features, got {input_shape[-1]}. "
                "Consider reshaping your input."
            )

        super().build(input_shape)


class FuzzyEquivGateLayer(AdvancedLogicGateLayer):
    """
    EQUIVALENCE gate layer supporting various fuzzy logic systems.

    The EQUIVALENCE operation (↔) is true when both inputs have the same truth value.

    Args:
        logic_system: Logic system to use (Łukasiewicz, Gödel, Product, Boolean)
        use_bounds: Whether to use upper and lower bounds for truth values
        trainable_weights: Whether input weights should be trainable
        initial_weight: Initial value for trainable weights
        temperature: Parameter controlling sharpness of logical transitions
        use_bias: Whether to use a bias vector
        kernel_initializer: Initializer for kernel weights
        bias_initializer: Initializer for bias vector
        kernel_regularizer: Regularizer for kernel weights
        bias_regularizer: Regularizer for bias vector
        activity_regularizer: Regularizer for layer output
        kernel_constraint: Constraint for kernel weights
        bias_constraint: Constraint for bias vector
    """

    def __init__(
            self,
            logic_system: Union[LogicSystem, str] = LogicSystem.BOOLEAN,
            use_bounds: bool = False,
            trainable_weights: bool = False,
            initial_weight: float = 1.0,
            temperature: float = 0.1,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_constraint: Optional[keras.constraints.Constraint] = None,
            bias_constraint: Optional[keras.constraints.Constraint] = None,
            **kwargs
    ):
        """Initialize the fuzzy EQUIVALENCE gate layer."""
        super().__init__(
            operation=LogicalOperations.logical_equiv,
            logic_system=logic_system,
            use_bounds=use_bounds,
            trainable_weights=trainable_weights,
            initial_weight=initial_weight,
            temperature=temperature,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

    def build(self, input_shape):
        """
        Build the layer based on input shape.

        Args:
            input_shape: Shape of the input tensor or list of tensors
        """
        # Ensure we have exactly 2 inputs for EQUIVALENCE
        if isinstance(input_shape, list):
            if len(input_shape) != 2:
                raise ValueError("EquivGateLayer supports exactly 2 inputs.")
        elif input_shape[-1] != 2:
            raise ValueError(
                f"Expected input with 2 features, got {input_shape[-1]}. "
                "Consider reshaping your input."
            )

        super().build(input_shape)
