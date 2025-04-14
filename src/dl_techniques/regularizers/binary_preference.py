"""
Binary Preference Regularizer
============================

This regularizer implements an unconventional cost function that encourages neural
network weights to adopt binary (0 or 1) values through the following equation:

    y = (1 - ((x-0.5)^2)/0.25)^2

Key Properties:
-------------
* Creates two stable points at x=0 and x=1 where the cost is zero
* Maximum penalty occurs at x=0.5 with cost=1.0
* Symmetric parabolic-like curve around x=0.5
* Smooth and differentiable across its domain

Unlike traditional L1/L2 regularizers that push weights toward zero, this regularizer
creates a "binarizing pressure" on the weights. This can be particularly useful for:

1. Training networks with interpretable, nearly-binary weights
2. Creating sparse representations with clear on/off patterns
3. Reducing effective model complexity by simplifying weight distributions
4. Problems where binary-like decisions are inherently beneficial

The regularizer can be scaled through a 'scale' parameter to control the strength
of the binarizing effect during training.

Example Usage:
------------
    >>> regularizer = get_binary_regularizer(scale=1.0)
    >>> model.add(Dense(64, kernel_regularizer=regularizer))

Note: The convergence behavior and final weight distributions may be significantly
      different from traditional regularizers. Consider starting with a small
      scale value and gradually increasing it if needed.
"""

import keras
import tensorflow as tf
from typing import Union, Optional


# ---------------------------------------------------------------------


class BinaryPreferenceRegularizer(keras.regularizers.Regularizer):
    """
    A regularizer that encourages weights to move towards binary values (0 or 1).

    The regularizer implements the cost function:
    y = (1 - ((x-0.5)^2)/0.25)^2

    Key properties:
    - Cost is zero when weights are 0 or 1
    - Maximum cost (1.0) occurs at weight = 0.5
    - Symmetric around 0.5
    - Creates a binarizing effect by penalizing non-binary weights

    This regularizer can be useful for:
    - Training networks with binary-like weights
    - Encouraging sparse, interpretable representations
    - Reducing model complexity by pushing weights to extreme values

    Attributes:
        multiplier (float): Scaling factor for the regularization term
    """

    def __init__(self, multiplier: float = 1.0, scale: float = 1.0) -> None:
        """
        Initialize the binary preference regularizer.

        Args:
            multiplier (float): multiplier factor for the regularization term.
                          Higher values create stronger binarization pressure.
                          Defaults to 1.0.
            scale (float): Squeezes or expands the signature
              Defaults to 1.0.
        """
        if scale <= 0:
            raise ValueError("scale must be > 0")
        self.multiplier = multiplier
        self.scale = scale

    def __call__(self, weights: tf.Tensor) -> tf.Tensor:
        """
        Calculate the regularization cost for given weights.

        The cost is zero for weights at 0 or 1, and maximum (1.0 * scale)
        for weights at 0.5.

        Args:
            weights (tf.Tensor): Input tensor containing the weights
                               to be regularized.

        Returns:
            tf.Tensor: The calculated regularization cost.
        """
        # Calculate (x - 0.5)^2 / 0.25
        normalized = tf.square(self.scale * weights - 0.5) / 0.25

        # Calculate (1 - normalized)^2
        cost = tf.square(1.0 - normalized)

        # Apply scaling and return mean cost
        return self.multiplier * tf.reduce_mean(cost)

    def get_config(self) -> dict:
        """
        Return the configuration of the regularizer.

        Returns:
            dict: Configuration dictionary containing the scale parameter.
        """
        return {
            'multiplier': self.multiplier,
            'scale': self.scale
        }

    @classmethod
    def from_config(cls, config: dict) -> 'BinaryPreferenceRegularizer':
        """
        Create a regularizer instance from configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing the scale parameter.

        Returns:
            BinaryPreferenceRegularizer: A new instance of the regularizer.
        """
        return cls(**config)

# ---------------------------------------------------------------------
