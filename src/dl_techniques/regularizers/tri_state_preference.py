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
from keras import ops
from typing import Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TriStatePreferenceRegularizer(keras.regularizers.Regularizer):
    """A regularizer that encourages weights to converge to -1, 0, or 1.

    The regularizer implements a 6th order polynomial cost function that creates
    stable points at -1, 0, and 1, with smooth transitions and strong penalties
    for weights outside [-1, 1].

    Args:
        multiplier: Multiplier factor for the regularization term.
            Higher values create stronger pressure towards -1, 0, 1.
        scale: Scaling factor applied to weights before regularization.
            Controls the width of the regularization effect.

    Attributes:
        multiplier (float): The regularization strength multiplier.
        scale (float): The weight scaling factor.
        base_coefficient (float): Base coefficient for exact maxima of 1.0.
    """

    def __init__(self, multiplier: float = 1.0, scale: float = 1.0) -> None:
        """Initialize the tri-state preference regularizer.

        Args:
            multiplier: Multiplier factor for the regularization term.
                Higher values create stronger pressure towards -1, 0, 1.
                Defaults to 1.0.
            scale: Scaling factor applied to weights before regularization.
                Defaults to 1.0.

        Raises:
            ValueError: If multiplier or scale are not positive numbers.
        """
        if multiplier <= 0:
            raise ValueError(f"multiplier must be positive, got {multiplier}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")

        self.multiplier = float(multiplier)
        self.scale = float(scale)
        self.base_coefficient = 32.0 / 4.5  # Coefficient for exact maxima of 1

        logger.info(
            f"Initialized TriStatePreferenceRegularizer with "
            f"multiplier={self.multiplier}, scale={self.scale}"
        )

    def __call__(self, weights) -> keras.KerasTensor:
        """Calculate the regularization cost for given weights.

        The cost is zero at -1, 0, and 1, reaches maxima of 1.0 at -0.5 and 0.5,
        and increases rapidly outside [-1, 1].

        Args:
            weights: Input tensor containing the weights to be regularized.

        Returns:
            The calculated regularization cost as a scalar tensor.
        """
        # Scale the weights
        x = self.scale * weights

        # Calculate the polynomial terms: x²(x+1)²(x-1)²
        x_squared = ops.square(x)
        plus_one_squared = ops.square(x + 1.0)
        minus_one_squared = ops.square(x - 1.0)

        # Combine terms and apply base coefficient
        cost = (
            self.base_coefficient *
            x_squared *
            plus_one_squared *
            minus_one_squared
        )

        # Apply multiplier and return mean cost
        regularization_loss = self.multiplier * ops.mean(cost)

        return regularization_loss

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the regularizer.

        Returns:
            Configuration dictionary containing the regularizer parameters.
        """
        return {
            'multiplier': self.multiplier,
            'scale': self.scale
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TriStatePreferenceRegularizer':
        """Create a regularizer instance from configuration dictionary.

        Args:
            config: Configuration dictionary containing the regularizer parameters.

        Returns:
            A new instance of the TriStatePreferenceRegularizer.
        """
        return cls(**config)

    def __repr__(self) -> str:
        """Return string representation of the regularizer.

        Returns:
            String representation including key parameters.
        """
        return (
            f"TriStatePreferenceRegularizer("
            f"multiplier={self.multiplier}, "
            f"scale={self.scale})"
        )

# ---------------------------------------------------------------------
