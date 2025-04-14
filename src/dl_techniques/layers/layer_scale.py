"""
Custom Keras Layers for Scaling and Multiplication Operations
============================================================

This module provides specialized Keras layers for learnable scaling operations:

- LearnableMultiplier: Creates trainable multipliers (global or per-channel) with
  configurable constraints
"""

import keras
from enum import Enum
import tensorflow as tf
from typing import Dict, Any, Optional, Tuple, Union

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint
from dl_techniques.regularizers.binary_preference import BinaryPreferenceRegularizer

# ---------------------------------------------------------------------

class MultiplierType(Enum):
    GLOBAL = 0

    CHANNEL = 1

    @staticmethod
    def from_string(type_str: Union[str, "MultiplierType"]) -> "MultiplierType":
        # --- argument checking
        if type_str is None:
            raise ValueError("type_str must not be null")
        if isinstance(type_str, MultiplierType):
            return type_str
        if not isinstance(type_str, str):
            raise ValueError("type_str must be string")
        # --- clean string and get
        type_str = type_str.strip().upper()
        if len(type_str) <= 0:
            raise ValueError("stripped type_str must not be empty")
        return MultiplierType[type_str]

    def to_string(self) -> str:
        return self.name


# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class LearnableMultiplier(keras.layers.Layer):
    """
    Layer implementing learnable multipliers.

    The multipliers can be either global (single value) or per-channel.

    Args:
        multiplier_type: Type of multiplier ('GLOBAL' or 'CHANNEL').
        initializer: Weight initializer (default: constant 1).
        regularizer: Weight regularizer (default: None).
        constraint: Constraint of the weights
        **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            multiplier_type: Union[MultiplierType, str] = MultiplierType.CHANNEL,
            initializer: Optional[keras.initializers.Initializer] = keras.initializers.Constant(1.0),
            regularizer: Optional[keras.regularizers.Regularizer] = None,
            constraint: Optional[keras.constraints.Constraint] = ValueRangeConstraint(min_value=0.0, max_value=1.0),
            **kwargs: Any
    ) -> None:
        """Initialize the LearnableMultiplier layer."""
        super().__init__(**kwargs)

        self.initializer = initializer
        self.regularizer = regularizer
        self.constraint = constraint
        self.multiplier_type = MultiplierType.from_string(multiplier_type)
        self.gamma = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer by creating the multiplier weights.

        Args:
            input_shape: Shape of input tensor.
        """
        # Create shape with ones except for channel dimension if needed
        if self.multiplier_type == MultiplierType.GLOBAL:
            weight_shape = [1] * len(input_shape)
        elif self.multiplier_type == MultiplierType.CHANNEL:
            weight_shape = [1] * (len(input_shape) - 1) + [input_shape[-1]]
        else:
            raise ValueError(f"invalid multiplier_type: [{self.multiplier_type}")

        self.gamma = self.add_weight(
            name="gamma",
            shape=weight_shape,
            initializer=self.initializer,
            regularizer=self.regularizer,
            constraint=self.constraint,
            trainable=True,
            dtype=self.dtype
        )

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None,
            **kwargs: Any
    ) -> tf.Tensor:
        """
        Apply the learnable multipliers to inputs.

        The multipliers are transformed using tanh to constrain their values.
        When capped=True, values are constrained to [0, 1].
        When capped=False, values can exceed 1.0.

        Args:
            inputs: Input tensor.
            training: Whether in training mode (unused).
            **kwargs: Additional call arguments.

        Returns:
            Tensor with multipliers applied.
        """
        # Compute base multiplier using tanh transformation
        return tf.multiply(self.gamma, inputs)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "multiplier_type": self.multiplier_type.to_string(),
            "regularizer": keras.regularizers.serialize(self.regularizer),
            "initializer": keras.initializers.serialize(self.initializer),
            "constraint": keras.constraints.serialize(self.constraint),
        })
        return config

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute output shape (same as input shape)."""
        return input_shape

# ---------------------------------------------------------------------
