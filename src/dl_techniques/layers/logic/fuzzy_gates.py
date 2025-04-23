"""
Implementation of advanced logic gates based on fuzzy logic systems for Keras 3.x.

This module provides implementations of logical gates (AND, OR, NOT, etc.)
as Keras layers that support different fuzzy logic systems, truth bounds,
and bidirectional reasoning.
"""

import keras
from keras import ops
from typing import Optional, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .logic_gates import AdvancedLogicGateLayer
from .logic_operations import LogicSystem, LogicalOperations

# ---------------------------------------------------------------------


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
        **kwargs: Additional keyword arguments for the base Layer

    Input shape:
        List of tensors [x1, x2, ...], each with shape (batch_size, ...) containing
        values in range [0, 1], or a single tensor with shape (batch_size, n_features)

    Output shape:
        Tensor with shape (batch_size, ...) or (batch_size, 1) containing
        values in range [0, 1]

    Example:
        >>> # Using list of inputs
        >>> x1 = keras.random.uniform((4, 5), 0, 1)
        >>> x2 = keras.random.uniform((4, 5), 0, 1)
        >>> fuzzy_and = FuzzyANDGateLayer(logic_system='product', temperature=0.5)
        >>> result = fuzzy_and([x1, x2])
        >>> print(result.shape)
        (4, 5)

        >>> # Using single tensor with multiple features
        >>> x = keras.random.uniform((4, 3), 0, 1)  # 3 features
        >>> fuzzy_and = FuzzyANDGateLayer(logic_system='godel')
        >>> result = fuzzy_and(x)
        >>> print(result.shape)
        (4, 1)
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

    @classmethod
    def from_config(cls, config: Dict[str, Any], custom_objects: Optional[Dict[str, Any]] = None) -> 'FuzzyANDGateLayer':
        """
        Creates a FuzzyANDGateLayer from its config.

        Args:
            config: Layer configuration dictionary
            custom_objects: Dictionary mapping names to custom objects

        Returns:
            A new instance of FuzzyANDGateLayer
        """
        # Add the operation to the config
        if 'operation' not in config:
            config['operation'] = LogicalOperations.logical_and

        return super(FuzzyANDGateLayer, cls).from_config(config, custom_objects)


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
        **kwargs: Additional keyword arguments for the base Layer

    Input shape:
        List of tensors [x1, x2, ...], each with shape (batch_size, ...) containing
        values in range [0, 1], or a single tensor with shape (batch_size, n_features)

    Output shape:
        Tensor with shape (batch_size, ...) or (batch_size, 1) containing
        values in range [0, 1]

    Example:
        >>> # Using list of inputs
        >>> x1 = keras.random.uniform((4, 5), 0, 1)
        >>> x2 = keras.random.uniform((4, 5), 0, 1)
        >>> fuzzy_or = FuzzyORGateLayer(logic_system='lukasiewicz')
        >>> result = fuzzy_or([x1, x2])
        >>> print(result.shape)
        (4, 5)
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

    @classmethod
    def from_config(cls, config: Dict[str, Any], custom_objects: Optional[Dict[str, Any]] = None) -> 'FuzzyORGateLayer':
        """
        Creates a FuzzyORGateLayer from its config.

        Args:
            config: Layer configuration dictionary
            custom_objects: Dictionary mapping names to custom objects

        Returns:
            A new instance of FuzzyORGateLayer
        """
        # Add the operation to the config
        if 'operation' not in config:
            config['operation'] = LogicalOperations.logical_or

        return super(FuzzyORGateLayer, cls).from_config(config, custom_objects)


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
        **kwargs: Additional keyword arguments for the base Layer

    Input shape:
        Tensor with shape (batch_size, ...) containing values in range [0, 1]

    Output shape:
        Tensor with the same shape as input containing values in range [0, 1]

    Example:
        >>> x = keras.random.uniform((4, 5), 0, 1)
        >>> fuzzy_not = FuzzyNOTGateLayer(logic_system='product')
        >>> result = fuzzy_not(x)
        >>> print(result.shape)
        (4, 5)
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

    def build(self, input_shape: Any) -> None:
        """
        Build the layer based on input shape.

        Args:
            input_shape: Shape of the input tensor
        """
        # Validate that we have a single input using backend-agnostic approach
        if isinstance(input_shape, list):
            if len(input_shape) > 1:
                raise ValueError("NOTGateLayer accepts only one input.")
            input_shape = input_shape[0]

        # We're not requiring a single feature dimension anymore
        # This allows more flexibility in the layer's use

        super().build(input_shape)

    @classmethod
    def from_config(cls, config: Dict[str, Any], custom_objects: Optional[Dict[str, Any]] = None) -> 'FuzzyNOTGateLayer':
        """
        Creates a FuzzyNOTGateLayer from its config.

        Args:
            config: Layer configuration dictionary
            custom_objects: Dictionary mapping names to custom objects

        Returns:
            A new instance of FuzzyNOTGateLayer
        """
        # Add the operation to the config
        if 'operation' not in config:
            config['operation'] = LogicalOperations.logical_not

        return super(FuzzyNOTGateLayer, cls).from_config(config, custom_objects)


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
        **kwargs: Additional keyword arguments for the base Layer

    Input shape:
        List of exactly 2 tensors [x1, x2], each with shape (batch_size, ...) containing
        values in range [0, 1], or a single tensor with shape (batch_size, 2)

    Output shape:
        Tensor with shape (batch_size, ...) or (batch_size, 1) containing
        values in range [0, 1]

    Example:
        >>> x1 = keras.random.uniform((4, 5), 0, 1)
        >>> x2 = keras.random.uniform((4, 5), 0, 1)
        >>> fuzzy_xor = FuzzyXORGateLayer(logic_system='boolean', temperature=0.1)
        >>> result = fuzzy_xor([x1, x2])
        >>> print(result.shape)
        (4, 5)
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

    def build(self, input_shape: Any) -> None:
        """
        Build the layer based on input shape.

        Args:
            input_shape: Shape of the input tensor or list of tensors
        """
        # Ensure we have exactly 2 inputs for XOR using backend-agnostic approach
        if isinstance(input_shape, list):
            if len(input_shape) != 2:
                raise ValueError("XORGateLayer supports exactly 2 inputs.")
        elif len(ops.shape(input_shape)) > 1 and input_shape[-1] != 2:
            raise ValueError(
                f"Expected input with 2 features, got {input_shape[-1]}. "
                "Consider reshaping your input."
            )

        super().build(input_shape)

    @classmethod
    def from_config(cls, config: Dict[str, Any], custom_objects: Optional[Dict[str, Any]] = None) -> 'FuzzyXORGateLayer':
        """
        Creates a FuzzyXORGateLayer from its config.

        Args:
            config: Layer configuration dictionary
            custom_objects: Dictionary mapping names to custom objects

        Returns:
            A new instance of FuzzyXORGateLayer
        """
        # Add the operation to the config
        if 'operation' not in config:
            config['operation'] = LogicalOperations.logical_xor

        return super(FuzzyXORGateLayer, cls).from_config(config, custom_objects)


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
        **kwargs: Additional keyword arguments for the base Layer

    Input shape:
        List of tensors [x1, x2, ...], each with shape (batch_size, ...) containing
        values in range [0, 1], or a single tensor with shape (batch_size, n_features)

    Output shape:
        Tensor with shape (batch_size, ...) or (batch_size, 1) containing
        values in range [0, 1]

    Example:
        >>> x1 = keras.random.uniform((4, 5), 0, 1)
        >>> x2 = keras.random.uniform((4, 5), 0, 1)
        >>> x3 = keras.random.uniform((4, 5), 0, 1)
        >>> fuzzy_nand = FuzzyNANDGateLayer(logic_system='product')
        >>> result = fuzzy_nand([x1, x2, x3])
        >>> print(result.shape)
        (4, 5)
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

    @classmethod
    def from_config(cls, config: Dict[str, Any], custom_objects: Optional[Dict[str, Any]] = None) -> 'FuzzyNANDGateLayer':
        """
        Creates a FuzzyNANDGateLayer from its config.

        Args:
            config: Layer configuration dictionary
            custom_objects: Dictionary mapping names to custom objects

        Returns:
            A new instance of FuzzyNANDGateLayer
        """
        # Add the operation to the config
        if 'operation' not in config:
            config['operation'] = LogicalOperations.logical_nand

        return super(FuzzyNANDGateLayer, cls).from_config(config, custom_objects)


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
        **kwargs: Additional keyword arguments for the base Layer

    Input shape:
        List of tensors [x1, x2, ...], each with shape (batch_size, ...) containing
        values in range [0, 1], or a single tensor with shape (batch_size, n_features)

    Output shape:
        Tensor with shape (batch_size, ...) or (batch_size, 1) containing
        values in range [0, 1]

    Example:
        >>> x1 = keras.random.uniform((4, 5), 0, 1)
        >>> x2 = keras.random.uniform((4, 5), 0, 1)
        >>> fuzzy_nor = FuzzyNORGateLayer(logic_system='lukasiewicz')
        >>> result = fuzzy_nor([x1, x2])
        >>> print(result.shape)
        (4, 5)
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

    @classmethod
    def from_config(cls, config: Dict[str, Any], custom_objects: Optional[Dict[str, Any]] = None) -> 'FuzzyNORGateLayer':
        """
        Creates a FuzzyNORGateLayer from its config.

        Args:
            config: Layer configuration dictionary
            custom_objects: Dictionary mapping names to custom objects

        Returns:
            A new instance of FuzzyNORGateLayer
        """
        # Add the operation to the config
        if 'operation' not in config:
            config['operation'] = LogicalOperations.logical_nor

        return super(FuzzyNORGateLayer, cls).from_config(config, custom_objects)


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
        **kwargs: Additional keyword arguments for the base Layer

    Input shape:
        List of exactly 2 tensors [x1, x2], each with shape (batch_size, ...) containing
        values in range [0, 1], or a single tensor with shape (batch_size, 2)

    Output shape:
        Tensor with shape (batch_size, ...) or (batch_size, 1) containing
        values in range [0, 1]

    Example:
        >>> x1 = keras.random.uniform((4, 5), 0, 1)  # premise
        >>> x2 = keras.random.uniform((4, 5), 0, 1)  # conclusion
        >>> fuzzy_implies = FuzzyImpliesGateLayer(logic_system='godel')
        >>> result = fuzzy_implies([x1, x2])  # x1 → x2
        >>> print(result.shape)
        (4, 5)
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

    def build(self, input_shape: Any) -> None:
        """
        Build the layer based on input shape.

        Args:
            input_shape: Shape of the input tensor or list of tensors
        """
        # Ensure we have exactly 2 inputs for IMPLIES using backend-agnostic approach
        if isinstance(input_shape, list):
            if len(input_shape) != 2:
                raise ValueError("ImpliesGateLayer supports exactly 2 inputs.")
        elif len(ops.shape(input_shape)) > 1 and input_shape[-1] != 2:
            raise ValueError(
                f"Expected input with 2 features, got {input_shape[-1]}. "
                "Consider reshaping your input."
            )

        super().build(input_shape)

    @classmethod
    def from_config(cls, config: Dict[str, Any], custom_objects: Optional[Dict[str, Any]] = None) -> 'FuzzyImpliesGateLayer':
        """
        Creates a FuzzyImpliesGateLayer from its config.

        Args:
            config: Layer configuration dictionary
            custom_objects: Dictionary mapping names to custom objects

        Returns:
            A new instance of FuzzyImpliesGateLayer
        """
        # Add the operation to the config
        if 'operation' not in config:
            config['operation'] = LogicalOperations.logical_implies

        return super(FuzzyImpliesGateLayer, cls).from_config(config, custom_objects)


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
        **kwargs: Additional keyword arguments for the base Layer

    Input shape:
        List of exactly 2 tensors [x1, x2], each with shape (batch_size, ...) containing
        values in range [0, 1], or a single tensor with shape (batch_size, 2)

    Output shape:
        Tensor with shape (batch_size, ...) or (batch_size, 1) containing
        values in range [0, 1]

    Example:
        >>> x1 = keras.random.uniform((4, 5), 0, 1)
        >>> x2 = keras.random.uniform((4, 5), 0, 1)
        >>> fuzzy_equiv = FuzzyEquivGateLayer(logic_system='product')
        >>> result = fuzzy_equiv([x1, x2])  # x1 ↔ x2
        >>> print(result.shape)
        (4, 5)
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

    def build(self, input_shape: Any) -> None:
        """
        Build the layer based on input shape.

        Args:
            input_shape: Shape of the input tensor or list of tensors
        """
        # Ensure we have exactly 2 inputs for EQUIVALENCE using backend-agnostic approach
        if isinstance(input_shape, list):
            if len(input_shape) != 2:
                raise ValueError("EquivGateLayer supports exactly 2 inputs.")
        elif len(ops.shape(input_shape)) > 1 and input_shape[-1] != 2:
            raise ValueError(
                f"Expected input with 2 features, got {input_shape[-1]}. "
                "Consider reshaping your input."
            )

        super().build(input_shape)

    @classmethod
    def from_config(cls, config: Dict[str, Any], custom_objects: Optional[Dict[str, Any]] = None) -> 'FuzzyEquivGateLayer':
        """
        Creates a FuzzyEquivGateLayer from its config.

        Args:
            config: Layer configuration dictionary
            custom_objects: Dictionary mapping names to custom objects

        Returns:
            A new instance of FuzzyEquivGateLayer
        """
        # Add the operation to the config
        if 'operation' not in config:
            config['operation'] = LogicalOperations.logical_equiv

        return super(FuzzyEquivGateLayer, cls).from_config(config, custom_objects)