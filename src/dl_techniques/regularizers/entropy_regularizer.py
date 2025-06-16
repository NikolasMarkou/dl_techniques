"""
Entropy-Based Neural Network Regularization

Overview
--------
This module implements a custom regularization technique based on entropy principles from
information theory. The entropy regularizer shapes how information is distributed in neural
network weight matrices, which can influence the network's generalization capabilities and
representational efficiency.

Theoretical Background
---------------------
Entropy regularization draws from information theory principles and statistical mechanics.
Shannon entropy measures the uncertainty or randomness in a probability distribution. In the
context of neural networks, weight matrices with different entropy profiles exhibit different
computational and representational properties:

- **Low entropy weights**: Concentrate information in a few dominant connections, similar to
  sparse coding or feature selection
- **High entropy weights**: Distribute information across many connections, similar to
  distributed representations

Mathematical Foundation
----------------------
The entropy of weight matrices is calculated using Shannon's entropy formula:
    H(W) = -∑(p_i * log(p_i))
where p_i are the normalized weight values (using softmax). The entropy is then
normalized by dividing by the maximum possible entropy log(n) to get a value between 0 and 1.

Different parts of the network are encouraged to have specific entropy targets by
penalizing the squared difference between the actual and target entropy.

Practical Applications
---------------------
Entropy regularization can be useful in:
1. Preventing over-concentration of information in a few weights
2. Creating more robust distributed representations
3. Controlling information flow between layers
4. Improving generalization by enforcing specific information distribution patterns

References
----------
- Shannon, C. E. (1948). "A Mathematical Theory of Communication". Bell System Technical Journal.
- Tishby, N., & Zaslavsky, N. (2015). "Deep learning and the information bottleneck principle".
  IEEE Information Theory Workshop (ITW).
- Yang, G., & Schoenholz, S. (2017). "Mean Field Residual Networks: On the Edge of Chaos".
  Advances in Neural Information Processing Systems (NeurIPS).
- Lin, H. W., Tegmark, M., & Rolnick, D. (2017). "Why does deep and cheap learning work so well?".
  Journal of Statistical Physics.
"""

import keras
from keras import ops
from typing import Dict, Any, Optional, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

DEFAULT_ENTROPY_STRENGTH: float = 0.01
DEFAULT_TARGET_ENTROPY: float = 0.7
DEFAULT_ENTROPY_AXIS: int = -1
DEFAULT_ENTROPY_EPSILON: float = 1e-10

# String constants for serialization
STR_STRENGTH: str = "strength"
STR_TARGET_ENTROPY: str = "target_entropy"
STR_AXIS: str = "axis"
STR_EPSILON: str = "epsilon"

# Predefined entropy targets
ENTROPY_LOW: float = 0.2
ENTROPY_MEDIUM: float = 0.5
ENTROPY_HIGH: float = 0.8

# Mode constants
MODE_LOW: str = "low"
MODE_MEDIUM: str = "medium"
MODE_HIGH: str = "high"


# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class EntropyRegularizer(keras.regularizers.Regularizer):
    """Custom regularizer that promotes entropy-based structure in neural network weights.

    This regularizer calculates the Shannon entropy of weight matrices and penalizes
    deviations from a target entropy value. By enforcing specific entropy profiles,
    we can control how information is distributed within the network, potentially
    improving generalization capabilities.

    The regularizer creates a "thermodynamic"-like control over the network, allowing
    different parts to operate at different "temperatures" (entropy levels), which
    influences how they process and transform information.

    Parameters
    ----------
    strength : float, optional
        Scaling factor for the regularization penalty. Higher values enforce
        stronger adherence to the target entropy profile, by default DEFAULT_ENTROPY_STRENGTH
    target_entropy : float, optional
        Normalized target entropy value between 0 and 1 to encourage.
        Values closer to 0 lead to more concentrated weights, while values closer
        to 1 encourage more uniformly distributed weights, by default DEFAULT_TARGET_ENTROPY
    axis : int, optional
        The axis along which to compute entropy, by default DEFAULT_ENTROPY_AXIS
    epsilon : float, optional
        Small constant added for numerical stability when taking logarithms,
        by default DEFAULT_ENTROPY_EPSILON

    Raises
    ------
    ValueError
        If strength is negative, target_entropy is not in [0, 1], or epsilon is non-positive

    Notes
    -----
    The regularization encourages different information distribution patterns:
    - Low target entropy (≈0.2): Encourages concentrated, sparse-like weights
    - Medium target entropy (≈0.5): Balanced information distribution
    - High target entropy (≈0.8): Encourages broadly distributed weights

    Examples
    --------
    >>> # Apply medium entropy regularization to a layer
    >>> regularizer = EntropyRegularizer(strength=0.01, target_entropy=0.5)
    >>> layer = keras.layers.Dense(64, kernel_regularizer=regularizer)

    >>> # Apply low entropy regularization (more concentrated weights)
    >>> regularizer = EntropyRegularizer(strength=0.02, target_entropy=0.2)
    >>> layer = keras.layers.Dense(64, kernel_regularizer=regularizer)
    """

    def __init__(
        self,
        strength: float = DEFAULT_ENTROPY_STRENGTH,
        target_entropy: float = DEFAULT_TARGET_ENTROPY,
        axis: int = DEFAULT_ENTROPY_AXIS,
        epsilon: float = DEFAULT_ENTROPY_EPSILON,
        **kwargs: Any
    ) -> None:
        """Initialize the entropy regularizer.

        Parameters
        ----------
        strength : float, optional
            Scaling factor for the regularization penalty, by default DEFAULT_ENTROPY_STRENGTH
        target_entropy : float, optional
            Target normalized entropy value (between 0 and 1), by default DEFAULT_TARGET_ENTROPY
        axis : int, optional
            The axis along which to compute entropy, by default DEFAULT_ENTROPY_AXIS
        epsilon : float, optional
            Small constant for numerical stability, by default DEFAULT_ENTROPY_EPSILON
        **kwargs : Any
            Additional arguments passed to parent regularizer

        Raises
        ------
        ValueError
            If parameters are outside valid ranges
        """
        super().__init__(**kwargs)

        # Validate input parameters
        if strength < 0.0:
            raise ValueError(f"strength must be non-negative, got {strength}")
        if not (0.0 <= target_entropy <= 1.0):
            raise ValueError(f"target_entropy must be in [0, 1], got {target_entropy}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.strength = strength
        self.target_entropy = target_entropy
        self.axis = axis
        self.epsilon = epsilon

        logger.debug(
            f"Initialized EntropyRegularizer with strength={strength}, "
            f"target_entropy={target_entropy}, axis={axis}, epsilon={epsilon}"
        )

    def __call__(self, weights: Union[keras.KerasTensor, Any]) -> Union[keras.KerasTensor, Any]:
        """Apply the entropy regularization to weights.

        This method implements the entropy regularization in several steps:
        1. Normalize weights using softmax to create a probability distribution
        2. Calculate Shannon entropy: H = -∑(p_i * log(p_i))
        3. Normalize entropy by dividing by maximum possible entropy (log(n))
        4. Compute penalty as squared difference from target entropy

        Parameters
        ----------
        weights : Union[keras.KerasTensor, Any]
            Weight tensor to apply regularization to

        Returns
        -------
        Union[keras.KerasTensor, Any]
            Regularization loss value (scalar tensor)

        Notes
        -----
        The regularization encourages different information distribution patterns:
        - Low target entropy (≈0.2): Encourages concentrated, sparse-like weights
        - Medium target entropy (≈0.5): Balanced information distribution
        - High target entropy (≈0.8): Encourages broadly distributed weights
        """
        # Step 1: Convert weights to a probability distribution via softmax
        # We use absolute values to make the distribution independent of weight signs
        weights_abs = ops.abs(weights)
        weights_normalized = ops.softmax(weights_abs, axis=self.axis)

        # Step 2: Calculate Shannon entropy: H = -∑(p_i * log(p_i))
        # Add epsilon for numerical stability when taking logarithms
        epsilon_tensor = ops.cast(self.epsilon, dtype=weights.dtype)
        safe_weights = ops.maximum(weights_normalized, epsilon_tensor)

        # Compute entropy using Shannon's formula
        log_weights = ops.log(safe_weights)
        entropy_terms = ops.multiply(weights_normalized, log_weights)
        entropy = ops.negative(ops.sum(entropy_terms, axis=self.axis))

        # Step 3: Normalize by maximum possible entropy (log(n)) to get value between 0 and 1
        # Maximum entropy occurs when all weights have equal probability (1/n)
        n_weights = ops.cast(ops.shape(weights)[self.axis], dtype=weights.dtype)
        max_entropy = ops.log(n_weights)
        normalized_entropy = ops.divide(entropy, max_entropy)

        # Step 4: Penalize deviation from target entropy using squared error loss
        # This creates a penalty that grows quadratically with distance from target
        target_tensor = ops.cast(self.target_entropy, dtype=weights.dtype)
        deviation = ops.subtract(normalized_entropy, target_tensor)
        squared_deviation = ops.square(deviation)
        penalty = ops.mean(squared_deviation)

        # Scale penalty by strength hyperparameter and return
        strength_tensor = ops.cast(self.strength, dtype=weights.dtype)
        return ops.multiply(strength_tensor, penalty)

    def get_config(self) -> Dict[str, Any]:
        """Get the regularizer configuration for serialization.

        This method enables the regularizer to be serialized and deserialized,
        which is essential for saving and loading models.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the regularizer configuration parameters

        Notes
        -----
        This method is required for proper serialization and deserialization
        of models containing this regularizer.
        """
        config = {}
        config.update({
            STR_STRENGTH: self.strength,
            STR_TARGET_ENTROPY: self.target_entropy,
            STR_AXIS: self.axis,
            STR_EPSILON: self.epsilon
        })
        return config


# ---------------------------------------------------------------------

def get_entropy_regularizer(
    strength: float = DEFAULT_ENTROPY_STRENGTH,
    target_entropy: Optional[float] = None,
    mode: Optional[str] = None,
    **kwargs: Any
) -> EntropyRegularizer:
    """Factory function to create entropy regularizers with common configurations.

    This convenience function provides predefined entropy targets based on
    common use cases, accessible through the 'mode' parameter.

    Parameters
    ----------
    strength : float, optional
        Regularization strength factor, by default DEFAULT_ENTROPY_STRENGTH
    target_entropy : Optional[float], optional
        Specific target entropy value (overrides mode if provided), by default None
    mode : Optional[str], optional
        Predefined entropy target mode, by default None
        Available modes:
        - 'low': Low entropy (0.2) for sparse, concentrated weights
        - 'medium': Medium entropy (0.5) for balanced distribution
        - 'high': High entropy (0.8) for widely distributed weights
        - None: Uses default target_entropy (0.7)
    **kwargs : Any
        Additional arguments passed to EntropyRegularizer

    Returns
    -------
    EntropyRegularizer
        Configured EntropyRegularizer instance

    Raises
    ------
    ValueError
        If an invalid mode is specified

    Examples
    --------
    >>> # Create a low-entropy regularizer for concentrated weights
    >>> reg = get_entropy_regularizer(strength=0.01, mode='low')

    >>> # Create a high-entropy regularizer for distributed weights
    >>> reg = get_entropy_regularizer(strength=0.02, mode='high')

    >>> # Create with specific target entropy
    >>> reg = get_entropy_regularizer(strength=0.01, target_entropy=0.6)
    """
    # Validate strength parameter
    if strength < 0.0:
        raise ValueError(f"strength must be non-negative, got {strength}")

    # If specific target_entropy provided, use it directly
    if target_entropy is not None:
        if not (0.0 <= target_entropy <= 1.0):
            raise ValueError(f"target_entropy must be in [0, 1], got {target_entropy}")
        logger.debug(f"Creating EntropyRegularizer with custom target_entropy={target_entropy}")
        return EntropyRegularizer(
            strength=strength,
            target_entropy=target_entropy,
            **kwargs
        )

    # Otherwise select based on mode
    mode_targets = {
        MODE_LOW: ENTROPY_LOW,
        MODE_MEDIUM: ENTROPY_MEDIUM,
        MODE_HIGH: ENTROPY_HIGH,
        None: DEFAULT_TARGET_ENTROPY
    }

    if mode not in mode_targets:
        valid_modes = [k for k in mode_targets.keys() if k is not None]
        raise ValueError(f"Invalid mode '{mode}'. Valid modes are: {valid_modes}")

    selected_target = mode_targets[mode]
    logger.debug(f"Creating EntropyRegularizer with mode='{mode}', target_entropy={selected_target}")

    return EntropyRegularizer(
        strength=strength,
        target_entropy=selected_target,
        **kwargs
    )


# ---------------------------------------------------------------------