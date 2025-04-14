import keras
import tensorflow as tf
from typing import Dict, Union, Optional

# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class ValueRangeConstraint(keras.constraints.Constraint):
    """Constrains weights to be within specified minimum and maximum values.

    This constraint ensures that all weights in a layer stay within a specified range.
    It can be used to enforce bounds on weights, which can be helpful for:
    - Preventing vanishing/exploding gradients
    - Implementing specific architecture requirements
    - Ensuring numerical stability

    Example:
        ```python
        # Constrain weights between 0.01 and 1.0
        constraint = ValueRangeConstraint(min_value=0.01, max_value=1.0)
        layer = keras.layers.Dense(
            units=64,
            kernel_constraint=constraint
        )
        ```
    """

    def __init__(
            self,
            min_value: float,
            max_value: Optional[float] = None,
            clip_gradients: bool = True
    ) -> None:
        """Initialize the constraint with minimum and optional maximum values.

        Args:
            min_value: Minimum allowed value for weights.
            max_value: Maximum allowed value for weights. If None, only minimum constraint
                is applied. Defaults to None.
            clip_gradients: Whether to clip gradients during backpropagation to prevent
                numerical instability. Defaults to True.

        Raises:
            ValueError: If min_value is greater than max_value when max_value is provided.
        """
        if max_value is not None and min_value > max_value:
            raise ValueError(
                f"min_value ({min_value}) cannot be greater than max_value ({max_value})"
            )

        self.min_value = float(min_value)
        self.max_value = float(max_value) if max_value is not None else None
        self.clip_gradients = clip_gradients

    def __call__(self, weights: tf.Tensor) -> tf.Tensor:
        """Apply the constraint to weights.

        Args:
            weights: Input tensor of weights to be constrained.

        Returns:
            tf.Tensor: Tensor with constrained weights.
        """
        constrained = tf.maximum(weights, self.min_value)

        if self.max_value is not None:
            constrained = tf.minimum(constrained, self.max_value)

        if self.clip_gradients:
            # Clip gradients to prevent numerical instability
            constrained = tf.clip_by_value(
                constrained,
                clip_value_min=self.min_value,
                clip_value_max=self.max_value if self.max_value is not None else tf.float32.max
            )

        return constrained

    def get_config(self) -> Dict[str, Union[float, None, bool]]:
        """Return the configuration of the constraint.

        Returns:
            Dict containing the configuration parameters.
        """
        return {
            'min_value': self.min_value,
            'max_value': self.max_value,
            'clip_gradients': self.clip_gradients
        }

    @classmethod
    def from_config(cls, config: Dict[str, Union[float, None, bool]]) -> 'ValueRangeConstraint':
        """Creates a constraint from its config.

        Args:
            config: Dictionary containing configuration parameters.

        Returns:
            A new instance of ValueRangeConstraint initialized with the config.
        """
        return cls(**config)