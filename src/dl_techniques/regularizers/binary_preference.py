"""
Encourage network weights to adopt binary values (0 or 1).

This regularizer introduces a penalty to the network's loss function,
designed to guide weights towards either 0 or 1. Unlike traditional L1
or L2 regularization that penalizes the magnitude of weights (based on
vector norms), this regularizer modifies the optimization landscape to
create two stable "valleys" or fixed points at 0 and 1. Any weight
value falling between these two points incurs a penalty, effectively
creating a "binarizing pressure" during training.

Architecturally, this component acts as a non-standard potential
function. It imposes a cost based on a weight's value, not its
magnitude relative to the origin. This is particularly useful for
building intrinsically interpretable or compressed models where weights
represent binary-like states, such as feature selection or network
pruning, without resorting to non-differentiable operations.

Foundational Mathematics
------------------------
The core of the regularizer is a custom-designed, differentiable
cost function that exhibits a "double-well potential" shape. For a
given weight `x`, the penalty `L(x)` is calculated as:

    L(x) = (1 - 4 * (x - 0.5)^2)^2

The intuition behind this equation's structure is as follows:
1.  `(x - 0.5)^2`: This term creates a standard parabola with a
    minimum at `x = 0.5`, the midpoint between the desired binary
    values.
2.  `4 * (...)`: The term is normalized such that its value is 1 at the
    endpoints `x = 0` and `x = 1`. This step is equivalent to dividing
    by the maximum value of the parabola in the [0, 1] range, which is
    0.25.
3.  `1 - (...)`: The normalized parabola is inverted. This creates a
    function with a maximum value of 1 at the midpoint (`x = 0.5`) and
    minimums of 0 at the endpoints (`x = 0` and `x = 1`).
4.  `(...)**2`: Squaring the entire expression is a critical final step.
    It ensures that the function is not only zero at `x = 0` and
    `x = 1`, but also that its first derivative is zero at these points.
    This creates truly stable fixed points in the optimization
    landscape, meaning the regularizer exerts no force on weights that
    have already converged to a binary value.

This function provides a smooth, continuous, and differentiable penalty
that is easily integrated into gradient-based optimization frameworks.

References
----------
This regularizer is not based on a specific academic paper but is derived
from first principles to create a desired penalty shape. The concept is
conceptually related to ideas in energy-based models and the design of
potential functions in physics, where a system is encouraged to settle
into low-energy states. It stands in contrast to standard regularization
methods which are typically based on Lp-norms.
"""

import keras
from keras import ops
from typing import Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

DEFAULT_BINARY_MULTIPLIER: float = 1.0
DEFAULT_BINARY_SCALE: float = 1.0
BINARY_CENTER_POINT: float = 0.5
BINARY_NORMALIZATION: float = 0.25

# String constants for serialization
STR_MULTIPLIER: str = "multiplier"
STR_SCALE: str = "scale"


# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class BinaryPreferenceRegularizer(keras.regularizers.Regularizer):
    """A regularizer that encourages weights to move towards binary values (0 or 1).

    The regularizer implements the cost function:
    y = (1 - ((scale*x - 0.5)^2)/0.25)^2

    This creates a unique regularization profile where:
    - Cost is zero when weights are at binary values (0 or 1 when scale=1)
    - Maximum cost (1.0) occurs at the midpoint (0.5 when scale=1)
    - Symmetric penalty around the midpoint
    - Creates a binarizing effect by penalizing intermediate weight values

    Parameters
    ----------
    multiplier : float, optional
        Scaling factor for the regularization term. Higher values create
        stronger binarization pressure, by default DEFAULT_BINARY_MULTIPLIER
    scale : float, optional
        Scales the effective range of binary targets. When scale=1, targets
        are 0 and 1. Different scales shift the binary target range,
        by default DEFAULT_BINARY_SCALE
    **kwargs : Any
        Additional arguments passed to parent regularizer

    Raises
    ------
    ValueError
        If multiplier is negative or scale is non-positive

    Notes
    -----
    This regularizer can be useful for:
    - Training networks with binary-like weights
    - Encouraging sparse, interpretable representations
    - Reducing model complexity by pushing weights to extreme values
    - Creating networks with clear on/off decision patterns

    The mathematical foundation creates a smooth, differentiable function
    that naturally guides weights toward binary values without hard constraints.

    Examples
    --------
    >>> # Basic binary regularization
    >>> regularizer = BinaryPreferenceRegularizer(multiplier=1.0)
    >>> layer = keras.layers.Dense(64, kernel_regularizer=regularizer)

    >>> # Stronger binarization pressure
    >>> regularizer = BinaryPreferenceRegularizer(multiplier=2.0)
    >>> layer = keras.layers.Conv2D(32, 3, kernel_regularizer=regularizer)
    """

    def __init__(
        self,
        multiplier: float = DEFAULT_BINARY_MULTIPLIER,
        scale: float = DEFAULT_BINARY_SCALE,
        **kwargs: Any
    ) -> None:
        """Initialize the binary preference regularizer.

        Parameters
        ----------
        multiplier : float, optional
            Multiplier factor for the regularization term. Higher values create
            stronger binarization pressure, by default DEFAULT_BINARY_MULTIPLIER
        scale : float, optional
            Scales the effective range of binary targets, by default DEFAULT_BINARY_SCALE
        **kwargs : Any
            Additional arguments passed to parent regularizer

        Raises
        ------
        ValueError
            If multiplier is negative or scale is non-positive
        """
        super().__init__(**kwargs)

        # Validate input parameters
        if multiplier < 0.0:
            raise ValueError(f"multiplier must be non-negative, got {multiplier}")
        if scale <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}")

        self.multiplier = multiplier
        self.scale = scale

        logger.debug(
            f"Initialized BinaryPreferenceRegularizer with "
            f"multiplier={multiplier}, scale={scale}"
        )

    def __call__(self, weights: Union[keras.KerasTensor, Any]) -> Union[keras.KerasTensor, Any]:
        """Calculate the regularization cost for given weights.

        The cost function creates a binarizing pressure by penalizing weights
        that are far from the binary target values. The cost is zero for weights
        at the binary targets and maximum at the midpoint.

        Parameters
        ----------
        weights : Union[keras.KerasTensor, Any]
            Input tensor containing the weights to be regularized

        Returns
        -------
        Union[keras.KerasTensor, Any]
            The calculated regularization cost (scalar tensor)

        Notes
        -----
        The implementation follows these steps:
        1. Scale weights and center around 0.5: (scale * weights - 0.5)
        2. Compute normalized squared deviation: ((scaled_weights)^2) / 0.25
        3. Apply the binary preference function: (1 - normalized)^2
        4. Take mean across all weights and apply multiplier
        """
        # Step 1: Scale weights and center around the binary midpoint
        scaled_weights = ops.subtract(
            ops.multiply(ops.cast(self.scale, dtype=weights.dtype), weights),
            ops.cast(BINARY_CENTER_POINT, dtype=weights.dtype)
        )

        # Step 2: Calculate normalized squared deviation from center
        # This gives us the ((x-0.5)^2)/0.25 term
        squared_deviation = ops.square(scaled_weights)
        normalized_deviation = ops.divide(
            squared_deviation,
            ops.cast(BINARY_NORMALIZATION, dtype=weights.dtype)
        )

        # Step 3: Apply the binary preference function: (1 - normalized)^2
        # This creates the characteristic shape with zeros at binary values
        one_tensor = ops.cast(1.0, dtype=weights.dtype)
        preference_term = ops.subtract(one_tensor, normalized_deviation)
        cost_per_weight = ops.square(preference_term)

        # Step 4: Compute mean cost and apply multiplier
        mean_cost = ops.mean(cost_per_weight)
        multiplier_tensor = ops.cast(self.multiplier, dtype=weights.dtype)

        return ops.multiply(multiplier_tensor, mean_cost)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the regularizer for serialization.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary containing regularizer parameters

        Notes
        -----
        This method is required for proper serialization and deserialization
        of models containing this regularizer.
        """
        config = {}
        config.update({
            STR_MULTIPLIER: self.multiplier,
            STR_SCALE: self.scale
        })
        return config


# ---------------------------------------------------------------------

def create_binary_preference_regularizer(
    multiplier: float = DEFAULT_BINARY_MULTIPLIER,
    scale: float = DEFAULT_BINARY_SCALE,
    **kwargs: Any
) -> BinaryPreferenceRegularizer:
    """Factory function to create binary preference regularizers.

    This convenience function provides a simple interface for creating
    binary preference regularizers with validation and logging.

    Parameters
    ----------
    multiplier : float, optional
        Multiplier factor for the regularization term, by default DEFAULT_BINARY_MULTIPLIER
    scale : float, optional
        Scales the effective range of binary targets, by default DEFAULT_BINARY_SCALE
    **kwargs : Any
        Additional arguments passed to BinaryPreferenceRegularizer

    Returns
    -------
    BinaryPreferenceRegularizer
        Configured BinaryPreferenceRegularizer instance

    Raises
    ------
    ValueError
        If multiplier is negative or scale is non-positive

    Examples
    --------
    >>> # Create a standard binary preference regularizer
    >>> reg = create_binary_preference_regularizer(multiplier=1.0)

    >>> # Create with stronger binarization pressure
    >>> reg = create_binary_preference_regularizer(multiplier=2.0, scale=1.5)

    Notes
    -----
    This factory function provides the same functionality as directly
    instantiating BinaryPreferenceRegularizer but with additional
    logging and a consistent interface pattern.
    """
    # Validate parameters
    if multiplier < 0.0:
        raise ValueError(f"multiplier must be non-negative, got {multiplier}")
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")

    logger.debug(
        f"Creating BinaryPreferenceRegularizer with "
        f"multiplier={multiplier}, scale={scale}"
    )

    return BinaryPreferenceRegularizer(
        multiplier=multiplier,
        scale=scale,
        **kwargs
    )


# ---------------------------------------------------------------------