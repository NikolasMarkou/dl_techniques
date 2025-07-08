import keras
from keras import ops
from typing import Dict, Union, Optional, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ValueRangeConstraint(keras.constraints.Constraint):
    """Constrains weights to be within specified minimum and maximum values.

    This constraint ensures that all weights in a layer stay within a specified range
    by clipping values that fall outside the bounds. It can be used to enforce bounds
    on weights, which can be helpful for:

    * Preventing vanishing/exploding gradients
    * Implementing specific architecture requirements
    * Ensuring numerical stability
    * Enforcing non-negative weights (e.g., for NMF-like decompositions)

    Args:
        min_value (float): Minimum allowed value for weights.
        max_value (Optional[float]): Maximum allowed value for weights. If None,
            only minimum constraint is applied. Defaults to None.

    Raises:
        ValueError: If min_value is greater than max_value when max_value is provided.

    Example:
        >>> # Constrain weights between 0.01 and 1.0
        >>> constraint = ValueRangeConstraint(min_value=0.01, max_value=1.0)
        >>> layer = keras.layers.Dense(
        ...     units=64,
        ...     kernel_constraint=constraint
        ... )

        >>> # Enforce non-negative weights only with custom gradient clipping
        >>> constraint = ValueRangeConstraint(min_value=0.0, clip_gradients=False)
        >>> layer = keras.layers.Dense(
        ...     units=32,
        ...     kernel_constraint=constraint
        ... )
    """

    def __init__(
            self,
            min_value: float,
            max_value: Optional[float] = None,
            clip_gradients: bool = True,
            **kwargs: Any
    ) -> None:
        """Initialize the constraint with minimum and optional maximum values.

        Args:
            min_value (float): Minimum allowed value for weights.
            max_value (Optional[float]): Maximum allowed value for weights. If None,
                only minimum constraint is applied. Defaults to None.
            clip_gradients (bool): Whether to clip gradients during backpropagation to prevent
                numerical instability. Defaults to True. Note: This parameter is kept for
                API compatibility but doesn't affect the constraint behavior as clipping
                is inherent to the constraint operation.
            **kwargs: Additional keyword arguments passed to parent class.

        Raises:
            ValueError: If min_value is greater than max_value when max_value is provided.
        """
        super().__init__(**kwargs)

        if max_value is not None and min_value > max_value:
            raise ValueError(
                f"min_value ({min_value}) cannot be greater than max_value ({max_value})"
            )

        self.min_value = float(min_value)
        self.max_value = float(max_value) if max_value is not None else None
        self.clip_gradients = clip_gradients

        logger.debug(
            f"Initialized ValueRangeConstraint with min_value={self.min_value}, "
            f"max_value={self.max_value}, clip_gradients={self.clip_gradients}"
        )

    def __call__(self, weights: keras.KerasTensor) -> keras.KerasTensor:
        """Apply the constraint to weights by clipping values to the specified range.

        Args:
            weights (keras.KerasTensor): Input tensor of weights to be constrained.

        Returns:
            keras.KerasTensor: Tensor with constrained weights clipped to the valid range.
        """
        # Apply minimum constraint
        constrained = ops.maximum(weights, self.min_value)

        # Apply maximum constraint if specified
        if self.max_value is not None:
            constrained = ops.minimum(constrained, self.max_value)

        return constrained

    def get_config(self) -> Dict[str, Union[float, None, bool]]:
        """Return the configuration of the constraint for serialization.

        Returns:
            Dict[str, Union[float, None, bool]]: Dictionary containing the configuration
                parameters needed to recreate this constraint.
        """
        config = super().get_config()
        config.update({
            'min_value': self.min_value,
            'max_value': self.max_value,
            'clip_gradients': self.clip_gradients,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ValueRangeConstraint':
        """Creates a constraint from its configuration dictionary.

        Args:
            config (Dict[str, Any]): Dictionary containing configuration parameters.

        Returns:
            ValueRangeConstraint: A new instance of ValueRangeConstraint initialized
                with the provided configuration.
        """
        return cls(**config)

    def __repr__(self) -> str:
        """Return string representation of the constraint.

        Returns:
            str: String representation showing the constraint parameters.
        """
        if self.max_value is not None:
            return (f"ValueRangeConstraint(min_value={self.min_value}, "
                    f"max_value={self.max_value}, clip_gradients={self.clip_gradients})")
        else:
            return (f"ValueRangeConstraint(min_value={self.min_value}, "
                    f"clip_gradients={self.clip_gradients})")

# ---------------------------------------------------------------------
