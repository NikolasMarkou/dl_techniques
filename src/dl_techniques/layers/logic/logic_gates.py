"""
Advanced logic gate base layer with support for fuzzy logic systems,
truth bounds, and bidirectional reasoning for Keras 3.x.
"""

import keras
from keras import ops
from typing import Optional, Union, Callable, Any, Tuple, List, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .logic_operations import LogicSystem

# ---------------------------------------------------------------------


class AdvancedLogicGateLayer(keras.layers.Layer):
    """
    Advanced logic gate layer that supports various fuzzy logic systems,
    truth bounds, and bidirectional reasoning.

    This layer provides the foundation for creating logical neural networks
    that combine the interpretability of symbolic logic with the learning
    capabilities of neural networks.

    Args:
        operation: Logical operation function to use
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
        Single tensor or list of tensors with shape (batch_size, ...), or
        tuples of (lower_bound, upper_bound) tensors if using bounds.

    Output shape:
        Tensor with shape matching input shape, or tuple of
        (lower_bound, upper_bound) tensors if using bounds.
    """

    def __init__(
            self,
            operation: Callable,
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
        """Initialize the advanced logic gate layer."""
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.operation = operation

        # Convert string to enum if needed
        if isinstance(logic_system, str):
            try:
                self.logic_system = LogicSystem(logic_system.lower())
            except ValueError:
                valid_systems = [system.value for system in LogicSystem]
                raise ValueError(
                    f"Invalid logic system: {logic_system}. Must be one of {valid_systems}"
                )
        else:
            self.logic_system = logic_system

        self.use_bounds = use_bounds
        self.trainable_weights_flag = trainable_weights
        self.initial_weight = initial_weight
        self.temperature = temperature
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        # Initialize variables for truth bounds
        self.lower_bound = None
        self.upper_bound = None

        # Initialize to None, will be set in build()
        self.kernel = None
        self.bias = None
        self.weights_var = None

    def build(self, input_shape: Union[List, Tuple, Any]) -> None:
        """
        Build the layer based on input shape.

        Args:
            input_shape: Shape of the input tensor or list of tensors
        """
        # Determine number of inputs
        if isinstance(input_shape, list):
            num_inputs = len(input_shape)
        else:
            num_inputs = input_shape[-1]

        # Initialize weights if trainable
        if self.trainable_weights_flag:
            self.weights_var = self.add_weight(
                name='weights',
                shape=(num_inputs,),
                initializer=keras.initializers.Constant(self.initial_weight),
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
            )

        # Initialize variables for truth bounds if used
        if self.use_bounds:
            self.lower_bound = self.add_weight(
                name='lower_bound',
                shape=(1,),
                initializer=keras.initializers.Constant(0.0),
                trainable=False,
            )
            self.upper_bound = self.add_weight(
                name='upper_bound',
                shape=(1,),
                initializer=keras.initializers.Constant(1.0),
                trainable=False,
            )

        # Add bias if used
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(1,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )

        self.built = True

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """
        Forward pass for the logic gate.

        Args:
            inputs: Input tensor or list of input tensors
            training: Boolean indicating whether in training mode

        Returns:
            Output tensor or tuple of (lower_bound, upper_bound) if using bounds
        """
        # Process inputs
        x = self._process_inputs(inputs)

        # Apply the logical operation
        result = self._apply_operation(x)

        # Update truth bounds if using them
        if self.use_bounds and training:
            if isinstance(result, tuple):
                self._update_bounds(result[0], result[1])
            else:
                self._update_bounds(result, result)

        return result

    def _process_inputs(self, inputs: Any) -> Any:
        """
        Process inputs, handling lists and applying weights if needed.

        Args:
            inputs: Input tensor or list of input tensors

        Returns:
            Processed input tensor(s)
        """
        # Convert to list for consistent handling
        if not isinstance(inputs, list):
            if self.use_bounds:
                # If using bounds, extract both bounds
                if isinstance(inputs, tuple) and len(inputs) == 2:
                    return inputs
                else:
                    # If single tensor, use it for both bounds
                    return (inputs, inputs)
            else:
                return [inputs]

        # Apply weights if trainable
        if self.trainable_weights_flag and self.weights_var is not None:
            weighted_inputs = []
            for i, inp in enumerate(inputs):
                weight = self.weights_var[i]
                # Handle bounds
                if isinstance(inp, tuple) and len(inp) == 2 and self.use_bounds:
                    lower, upper = inp
                    weighted_inputs.append((lower * weight, upper * weight))
                else:
                    weighted_inputs.append(inp * weight)

            return weighted_inputs

        return inputs

    def _apply_operation(self, x: Any) -> Any:
        """
        Apply the logical operation to the inputs.

        Args:
            x: Processed input tensor(s)

        Returns:
            Result of the logical operation
        """
        # Handle different types of operations based on input shape
        if isinstance(x, tuple) and len(x) == 2 and self.use_bounds:
            # If using bounds, apply operation to both bounds
            lower, upper = x
            result_lower = self.operation(lower, logic_system=self.logic_system, temperature=self.temperature)
            result_upper = self.operation(upper, logic_system=self.logic_system, temperature=self.temperature)
            return (result_lower, result_upper)

        elif isinstance(x, list):
            # Check if we have a unary operation
            if len(x) == 1:
                result = self.operation(x[0], logic_system=self.logic_system, temperature=self.temperature)
            # Apply binary operation to first two elements
            elif len(x) == 2:
                result = self.operation(x[0], x[1], logic_system=self.logic_system, temperature=self.temperature)
            # For more than two inputs, apply operation sequentially
            else:
                result = x[0]
                for i in range(1, len(x)):
                    result = self.operation(result, x[i], logic_system=self.logic_system, temperature=self.temperature)

            # Apply bias if used
            if self.use_bias and self.bias is not None:
                result = result + self.bias

            return result
        else:
            # Single tensor input (for unary operations like NOT)
            result = self.operation(x, logic_system=self.logic_system, temperature=self.temperature)

            # Apply bias if used
            if self.use_bias and self.bias is not None:
                result = result + self.bias

            return result

    def _update_bounds(self, lower: Any, upper: Any) -> None:
        """
        Update the truth bounds based on new values.

        Args:
            lower: Lower bound tensor
            upper: Upper bound tensor
        """
        if self.lower_bound is not None and self.upper_bound is not None:
            # Ensure bounds are updated correctly with backend-agnostic operations
            new_lower = ops.minimum(self.lower_bound, ops.reduce_min(lower))
            new_upper = ops.maximum(self.upper_bound, ops.reduce_max(upper))

            # Update the weights
            self.lower_bound.assign(new_lower)
            self.upper_bound.assign(new_upper)

    def get_bounds(self) -> Optional[Tuple[Any, Any]]:
        """
        Get the current truth bounds of this gate.

        Returns:
            Tuple of (lower_bound, upper_bound) or None if not using bounds
        """
        if self.use_bounds and self.lower_bound is not None and self.upper_bound is not None:
            return self.lower_bound.numpy(), self.upper_bound.numpy()
        return None

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the layer.

        Returns:
            Dictionary containing the configuration
        """
        config = super().get_config()
        config.update({
            'logic_system': self.logic_system.value if isinstance(self.logic_system, LogicSystem) else self.logic_system,
            'use_bounds': self.use_bounds,
            'trainable_weights_flag': self.trainable_weights_flag,
            'initial_weight': self.initial_weight,
            'temperature': self.temperature,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
        })
        # Note: We don't serialize 'operation' because functions aren't directly serializable
        # It will need to be provided when loading the model
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any], custom_objects: Optional[Dict[str, Any]] = None) -> 'AdvancedLogicGateLayer':
        """
        Create a layer from its config.

        Args:
            config: Layer configuration dictionary
            custom_objects: Dictionary mapping names to custom objects

        Returns:
            A new instance of the layer

        Note:
            This implementation requires manually providing the 'operation' when loading
            since functions cannot be directly serialized.
        """
        # The operation needs to be provided manually when loading
        if 'operation' not in config and custom_objects and 'operation' in custom_objects:
            config['operation'] = custom_objects['operation']
        else:
            raise ValueError("The 'operation' function must be provided in custom_objects when loading the model")

        return cls(**config)


class BoundsLayer(keras.layers.Layer):
    """
    Layer that handles logical truth bounds propagation.

    This layer helps maintain and propagate upper and lower bounds
    on truth values throughout the logical network.

    Args:
        initial_lower: Initial lower bound value
        initial_upper: Initial upper bound value
        trainable: Whether bounds are trainable
        **kwargs: Additional keyword arguments for the base Layer

    Input shape:
        Tensor of shape (batch_size, ...)

    Output shape:
        Tuple of tensors (lower_bound, upper_bound) each with shape (batch_size, ...)

    Example:
        >>> bounds_layer = BoundsLayer(initial_lower=0.1, initial_upper=0.9)
        >>> x = keras.random.uniform((4, 5), 0, 1)
        >>> lower, upper = bounds_layer(x)
        >>> print(lower.shape, upper.shape)
        (4, 5) (4, 5)
    """

    def __init__(
            self,
            initial_lower: float = 0.0,
            initial_upper: float = 1.0,
            trainable: bool = False,
            **kwargs
    ):
        """Initialize the bounds layer."""
        super().__init__(**kwargs)
        self.initial_lower = initial_lower
        self.initial_upper = initial_upper
        self.trainable_bounds = trainable  # Renamed to avoid conflict with Layer.trainable

    def build(self, input_shape: Any) -> None:
        """
        Build the layer.

        Args:
            input_shape: Shape of the input tensor
        """
        self.lower_bound = self.add_weight(
            name='lower_bound',
            shape=(1,),
            initializer=keras.initializers.Constant(self.initial_lower),
            trainable=self.trainable_bounds,
        )

        self.upper_bound = self.add_weight(
            name='upper_bound',
            shape=(1,),
            initializer=keras.initializers.Constant(self.initial_upper),
            trainable=self.trainable_bounds,
        )

        self.built = True

    def call(self, inputs: Any, training: Optional[bool] = None) -> Tuple[Any, Any]:
        """
        Forward pass of the bounds layer.

        Args:
            inputs: Input tensor representing a truth value
            training: Whether in training mode

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Update bounds based on input
        if training:
            # In training, update bounds based on input using backend-agnostic operations
            new_lower = ops.minimum(self.lower_bound, ops.reduce_min(inputs))
            new_upper = ops.maximum(self.upper_bound, ops.reduce_max(inputs))

            # Update the weights
            self.lower_bound.assign(new_lower)
            self.upper_bound.assign(new_upper)

        # Broadcast bounds to match input shape
        broadcast_lower = ops.broadcast_to(self.lower_bound, ops.shape(inputs))
        broadcast_upper = ops.broadcast_to(self.upper_bound, ops.shape(inputs))

        return (broadcast_lower, broadcast_upper)

    def get_bounds(self) -> Tuple[Any, Any]:
        """
        Get the current truth bounds.

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        return self.lower_bound.numpy(), self.upper_bound.numpy()

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the layer.

        Returns:
            Dictionary containing the configuration
        """
        config = super().get_config()
        config.update({
            'initial_lower': self.initial_lower,
            'initial_upper': self.initial_upper,
            'trainable': self.trainable_bounds,
        })
        return config