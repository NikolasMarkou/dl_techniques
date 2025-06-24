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

The regularizer can be scaled through a 'multiplier' parameter to control the strength
of the binarizing effect during training, and a 'scale' parameter to adjust the
effective binary target range.

Example Usage:
------------
    >>> regularizer = get_binary_preference_regularizer(multiplier=1.0)
    >>> model.add(keras.layers.Dense(64, kernel_regularizer=regularizer))

Note: The convergence behavior and final weight distributions may be significantly
      different from traditional regularizers. Consider starting with a small
      multiplier value and gradually increasing it if needed.
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

def get_binary_preference_regularizer(
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
    >>> reg = get_binary_preference_regularizer(multiplier=1.0)

    >>> # Create with stronger binarization pressure
    >>> reg = get_binary_preference_regularizer(multiplier=2.0, scale=1.5)

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