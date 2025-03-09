"""
Advanced logic gate base layer with support for fuzzy logic systems,
truth bounds, and bidirectional reasoning.
"""

import keras
import tensorflow as tf
from typing import Optional, Union, Callable

from .logical_operations import LogicSystem


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

    def build(self, input_shape):
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

        self.built = True

    def call(self, inputs, training=None):
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
                self.lower_bound.assign(result[0])
                self.upper_bound.assign(result[1])
            else:
                self.lower_bound.assign(result)
                self.upper_bound.assign(result)

        return result

    def _process_inputs(self, inputs):
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

    def _apply_operation(self, x):
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
                return self.operation(x[0], logic_system=self.logic_system, temperature=self.temperature)

            # Apply binary operation to first two elements
            elif len(x) == 2:
                return self.operation(x[0], x[1], logic_system=self.logic_system, temperature=self.temperature)

            # For more than two inputs, apply operation sequentially
            else:
                result = x[0]
                for i in range(1, len(x)):
                    result = self.operation(result, x[i], logic_system=self.logic_system, temperature=self.temperature)
                return result

        else:
            # Single tensor input (for unary operations like NOT)
            return self.operation(x, logic_system=self.logic_system, temperature=self.temperature)

    def get_bounds(self):
        """
        Get the current truth bounds of this gate.

        Returns:
            Tuple of (lower_bound, upper_bound) or None if not using bounds
        """
        if self.use_bounds and self.lower_bound is not None and self.upper_bound is not None:
            return self.lower_bound.numpy(), self.upper_bound.numpy()
        return None

    def get_config(self):
        """
        Get the configuration of the layer.

        Returns:
            Dictionary containing the configuration
        """
        config = super().get_config()
        config.update({
            'logic_system': self.logic_system,
            'use_bounds': self.use_bounds,
            'trainable_weights_flag': self.trainable_weights_flag,
            'initial_weight': self.initial_weight,
            'temperature': self.temperature,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
        })
        # Note: We don't serialize 'operation' because functions aren't directly serializable
        # It will need to be provided when loading the model
        return config


class BoundsLayer(keras.layers.Layer):
    """
    Layer that handles logical truth bounds propagation.

    This layer helps maintain and propagate upper and lower bounds
    on truth values throughout the logical network.

    Args:
        initial_lower: Initial lower bound value
        initial_upper: Initial upper bound value
        trainable: Whether bounds are trainable
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
        self.trainable = trainable

    def build(self, input_shape):
        """
        Build the layer.

        Args:
            input_shape: Shape of the input tensor
        """
        self.lower_bound = self.add_weight(
            name='lower_bound',
            shape=(1,),
            initializer=keras.initializers.Constant(self.initial_lower),
            trainable=self.trainable,
        )

        self.upper_bound = self.add_weight(
            name='upper_bound',
            shape=(1,),
            initializer=keras.initializers.Constant(self.initial_upper),
            trainable=self.trainable,
        )

        self.built = True

    def call(self, inputs, training=None):
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
            # In training, update bounds based on input
            self.lower_bound.assign(tf.minimum(self.lower_bound, inputs))
            self.upper_bound.assign(tf.maximum(self.upper_bound, inputs))

        return (self.lower_bound, self.upper_bound)

    def get_bounds(self):
        """
        Get the current truth bounds.

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        return self.lower_bound.numpy(), self.upper_bound.numpy()

    def get_config(self):
        """
        Get the configuration of the layer.

        Returns:
            Dictionary containing the configuration
        """
        config = super().get_config()
        config.update({
            'initial_lower': self.initial_lower,
            'initial_upper': self.initial_upper,
            'trainable': self.trainable,
        })
        return config