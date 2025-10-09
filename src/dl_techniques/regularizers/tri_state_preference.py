"""
Encourage network weights to adopt ternary values (-1, 0, or 1).

This regularizer modifies the optimization landscape to guide network
weights towards one of three preferred states: -1, 0, or 1. It serves
as an alternative to standard L1/L2 regularization, which solely
penalizes weight magnitude. By creating three distinct low-cost "valleys"
in the loss surface, this regularizer is particularly useful for
training networks that are intrinsically sparse and quantized, which can
lead to significant model compression and hardware efficiency.

Architecturally, the regularizer implements a "triple-well potential"
function. This design creates three stable equilibrium points where the
regularization cost and its gradient are both zero. Weights that fall
into these wells are no longer pushed by the regularizer, allowing the
optimizer to focus on the main task loss. Conversely, weights lying
between these stable points incur a penalty, creating a continuous
pressure to move them towards the nearest preferred state.

Foundational Mathematics
------------------------
The core of the regularizer is a custom-designed, 6th-order polynomial
that is smooth and differentiable, making it suitable for gradient-based
optimization. For a given weight `x`, the penalty `L(x)` is:

    L(x) = (32/4.5) * x² * (x - 1)² * (x + 1)²

The structure of this polynomial is deliberate and provides key
properties:
1.  **Root Factors**: The terms `x`, `(x - 1)`, and `(x + 1)` ensure that
    the function has roots precisely at the target values of 0, 1, and
    -1.
2.  **Squared Terms**: Squaring each factor, i.e., `x²`, `(x - 1)²`, and
    `(x + 1)²`, is critical. This ensures that not only is the function
    value zero at the roots, but its first derivative is also zero. This
    mathematical property creates the stable, "flat-bottomed" wells in
    the loss landscape, preventing the regularizer from applying any
    force once a weight has reached a target state.
3.  **Normalization Constant**: The leading coefficient, `(32/4.5)`, is a
    normalization factor. It is specifically chosen to scale the penalty
    function such that the local maxima between the wells (at `x ≈ ±0.5`)
    have a value of exactly 1.0. This makes the `multiplier`
    hyperparameter more interpretable as a direct scaling of this
    normalized penalty.

References
----------
This regularizer is not based on a specific academic paper but is
derived from first principles to construct a potential function with the
desired properties. The concept is conceptually related to ideas in:
-   Network quantization and binarization, such as Ternary Connect
    networks, which explore training with discrete weights.
-   Energy-based models, where the goal is to shape an energy function
    (analogous to the loss landscape) to have minima at desired states.
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
