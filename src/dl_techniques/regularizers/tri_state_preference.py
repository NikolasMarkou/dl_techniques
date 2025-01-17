"""
Tri-State Preference Regularizer
==============================

A sophisticated regularizer that encourages neural network weights to converge
towards three preferred states: -1, 0, or 1. This is achieved through a carefully
designed 6th order polynomial:

    f(x) = (32/4.5)x²(x+1)²(x-1)²

Key Properties:
-------------
* Three stable points at x = -1, 0, and 1 where the cost is zero
* Local maxima of exactly 1.0 at x = -0.5 and x = 0.5
* Strictly positive cost between stable points
* Rapidly increasing cost for |x| > 1
* Perfectly symmetric around x = 0

Applications:
-----------
1. Training networks with quantized-like weights (-1, 0, 1)
2. Creating sparse, interpretable representations with clear ternary patterns
3. Reducing model complexity by pushing weights to three distinct values
4. Problems where ternary decisions are naturally beneficial
5. Hardware-efficient neural networks that prefer simple weight values

The regularizer provides a smooth, differentiable cost function that guides weights
towards these three preferred states while maintaining trainability through
gradient descent.
"""

import keras
import tensorflow as tf
from typing import Union, Optional

# ---------------------------------------------------------------------


class TriStatePreferenceRegularizer(keras.regularizers.Regularizer):
    """
    A regularizer that encourages weights to converge to -1, 0, or 1.

    The regularizer implements a 6th order polynomial cost function that creates
    stable points at -1, 0, and 1, with smooth transitions and strong penalties
    for weights outside [-1, 1].

    Attributes:
        scale (float): Scaling factor for the regularization term.
                      Higher values create stronger quantization pressure.
    """

    def __init__(self, scale: float = 1.0) -> None:
        """
        Initialize the tri-state preference regularizer.

        Args:
            scale (float): Scaling factor for the regularization term.
                          Higher values create stronger pressure towards -1, 0, 1.
                          Defaults to 1.0.
        """
        self.scale = scale
        self.base_coefficient = 32.0 / 4.5  # Coefficient for exact maxima of 1

    def __call__(self, weights: tf.Tensor) -> tf.Tensor:
        """
        Calculate the regularization cost for given weights.

        The cost is zero at -1, 0, and 1, reaches maxima of 1.0 at -0.5 and 0.5,
        and increases rapidly outside [-1, 1].

        Args:
            weights (tf.Tensor): Input tensor containing the weights
                               to be regularized.

        Returns:
            tf.Tensor: The calculated regularization cost.
        """
        # Calculate the polynomial terms: x²(x+1)²(x-1)²
        x = weights
        x_squared = tf.square(x)
        plus_one_squared = tf.square(x + 1)
        minus_one_squared = tf.square(x - 1)

        # Combine terms and apply base coefficient
        cost = self.base_coefficient * x_squared * plus_one_squared * minus_one_squared

        # Apply scaling and return mean cost
        return self.scale * tf.reduce_mean(cost)

    def get_config(self) -> dict:
        """
        Return the configuration of the regularizer.

        Returns:
            dict: Configuration dictionary containing the scale parameter.
        """
        return {'scale': self.scale}

    @classmethod
    def from_config(cls, config: dict) -> 'TriStatePreferenceRegularizer':
        """
        Create a regularizer instance from configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing the scale parameter.

        Returns:
            TriStatePreferenceRegularizer: A new instance of the regularizer.
        """
        return cls(**config)


def get_tri_state_regularizer(
        scale: Optional[float] = 1.0
) -> TriStatePreferenceRegularizer:
    """
    Factory function to create a tri-state preference regularizer instance.

    Args:
        scale (Optional[float]): Scaling factor for the regularization term.
                                Higher values create stronger quantization.
                                Defaults to 1.0.

    Returns:
        TriStatePreferenceRegularizer: An instance of the tri-state regularizer.

    Example:
        >>> # Create regularizer with default scaling
        >>> regularizer = get_tri_state_regularizer()
        >>> model.add(Dense(64, kernel_regularizer=regularizer))
        >>>
        >>> # Create regularizer with stronger quantization
        >>> strong_regularizer = get_tri_state_regularizer(scale=2.0)
        >>> model.add(Dense(32, kernel_regularizer=strong_regularizer))
    """
    return TriStatePreferenceRegularizer(scale=scale)

# ---------------------------------------------------------------------
