"""
# Entropy-Based Neural Network Regularization

## Overview
This module implements a custom regularization technique based on entropy principles from
information theory. The entropy regularizer shapes how information is distributed in neural
network weight matrices, which can influence the network's generalization capabilities and
representational efficiency.

## Theoretical Background
Entropy regularization draws from information theory principles and statistical mechanics.
Shannon entropy measures the uncertainty or randomness in a probability distribution. In the
context of neural networks, weight matrices with different entropy profiles exhibit different
computational and representational properties:

- **Low entropy weights**: Concentrate information in a few dominant connections, similar to
  sparse coding or feature selection
- **High entropy weights**: Distribute information across many connections, similar to
  distributed representations

## Mathematical Foundation
The entropy of weight matrices is calculated using Shannon's entropy formula:
    H(W) = -∑(p_i * log(p_i))
where p_i are the normalized weight values (using softmax). The entropy is then
normalized by dividing by the maximum possible entropy log(n) to get a value between 0 and 1.

Different parts of the network are encouraged to have specific entropy targets by
penalizing the squared difference between the actual and target entropy.

## Practical Applications
Entropy regularization can be useful in:
1. Preventing over-concentration of information in a few weights
2. Creating more robust distributed representations
3. Controlling information flow between layers
4. Improving generalization by enforcing specific information distribution patterns

## References
- Shannon, C. E. (1948). "A Mathematical Theory of Communication". Bell System Technical Journal.
- Tishby, N., & Zaslavsky, N. (2015). "Deep learning and the information bottleneck principle".
  IEEE Information Theory Workshop (ITW).
- Yang, G., & Schoenholz, S. (2017). "Mean Field Residual Networks: On the Edge of Chaos".
  Advances in Neural Information Processing Systems (NeurIPS).
- Lin, H. W., Tegmark, M., & Rolnick, D. (2017). "Why does deep and cheap learning work so well?".
  Journal of Statistical Physics.
"""

import keras
import tensorflow as tf
from typing import Dict, Any, Optional, Union


class EntropyRegularizer(keras.regularizers.Regularizer):
    """
    Custom regularizer that promotes entropy-based structure in neural network weights.

    This regularizer calculates the Shannon entropy of weight matrices and penalizes
    deviations from a target entropy value. By enforcing specific entropy profiles,
    we can control how information is distributed within the network, potentially
    improving generalization capabilities.

    The regularizer creates a "thermodynamic"-like control over the network, allowing
    different parts to operate at different "temperatures" (entropy levels), which
    influences how they process and transform information.

    Args:
        strength: Scaling factor for the regularization penalty. Higher values enforce
            stronger adherence to the target entropy profile.
        target_entropy: Normalized target entropy value between 0 and 1 to encourage.
            Values closer to 0 lead to more concentrated weights, while values closer
            to 1 encourage more uniformly distributed weights.
        axis: The axis along which to compute entropy. Defaults to -1 (last dimension).
        epsilon: Small constant added for numerical stability when taking logarithms.

    Examples:
        >>> # Apply medium entropy regularization to a layer
        >>> layer = keras.layers.Dense(
        ...     64,
        ...     kernel_regularizer=EntropyRegularizer(strength=0.01, target_entropy=0.5)
        ... )

        >>> # Apply low entropy regularization (more concentrated weights)
        >>> layer = keras.layers.Dense(
        ...     64,
        ...     kernel_regularizer=EntropyRegularizer(strength=0.02, target_entropy=0.2)
        ... )
    """

    def __init__(
        self,
        strength: float = 0.01,
        target_entropy: float = 0.7,
        axis: int = -1,
        epsilon: float = 1e-10
    ) -> None:
        """Initialize the entropy regularizer.

        Args:
            strength: Scaling factor for the regularization penalty. Higher values
                enforce stronger adherence to the target entropy.
            target_entropy: Target normalized entropy value (between 0 and 1) to
                encourage in the weight distribution.
            axis: The axis along which to compute entropy. Defaults to -1.
            epsilon: Small constant added for numerical stability when taking logarithms.
        """
        self.strength = strength
        self.target_entropy = target_entropy
        self.axis = axis
        self.epsilon = epsilon

    def __call__(self, weights: tf.Tensor) -> tf.Tensor:
        """Apply the entropy regularization to weights.

        This method implements the entropy regularization in several steps:
        1. Normalize weights using softmax to create a probability distribution
        2. Calculate Shannon entropy: H = -∑(p_i * log(p_i))
        3. Normalize entropy by dividing by maximum possible entropy (log(n))
        4. Compute penalty as squared difference from target entropy

        The regularization encourages different information distribution patterns:
        - Low target entropy (≈0.2): Encourages concentrated, sparse-like weights
        - Medium target entropy (≈0.5): Balanced information distribution
        - High target entropy (≈0.8): Encourages broadly distributed weights

        Args:
            weights: Weight tensor to apply regularization to.

        Returns:
            Regularization loss value (scalar tensor).
        """
        # Step 1: Convert weights to a probability distribution via softmax
        # We use absolute values to make the distribution independent of weight signs
        weights_normalized = tf.nn.softmax(tf.abs(weights), axis=self.axis)

        # Step 2: Calculate Shannon entropy: H = -∑(p_i * log(p_i))
        # Add epsilon for numerical stability when taking logarithms
        entropy = -tf.reduce_sum(
            weights_normalized * tf.math.log(tf.maximum(weights_normalized, self.epsilon)),
            axis=self.axis
        )

        # Step 3: Normalize by maximum possible entropy (log(n)) to get value between 0 and 1
        # Maximum entropy occurs when all weights have equal probability (1/n)
        max_entropy = tf.math.log(tf.cast(tf.shape(weights)[self.axis], tf.float32))
        normalized_entropy = entropy / max_entropy

        # Step 4: Penalize deviation from target entropy using squared error loss
        # This creates a penalty that grows quadratically with distance from target
        penalty = tf.reduce_mean(tf.square(normalized_entropy - self.target_entropy))

        # Scale penalty by strength hyperparameter and return
        return self.strength * penalty

    def get_config(self) -> Dict[str, Any]:
        """Get the regularizer configuration for serialization.

        This method enables the regularizer to be serialized and deserialized,
        which is essential for saving and loading models.

        Returns:
            Dictionary containing the regularizer configuration parameters.
        """
        return {
            'strength': self.strength,
            'target_entropy': self.target_entropy,
            'axis': self.axis,
            'epsilon': self.epsilon
        }


def get_entropy_regularizer(
    strength: float = 0.01,
    target_entropy: Optional[float] = None,
    mode: Optional[str] = None
) -> EntropyRegularizer:
    """Factory function to create entropy regularizers with common configurations.

    This convenience function provides predefined entropy targets based on
    common use cases, accessible through the 'mode' parameter.

    Args:
        strength: Regularization strength factor.
        target_entropy: Specific target entropy value (overrides mode if provided).
        mode: Predefined entropy target mode:
            - 'low': Low entropy (0.2) for sparse, concentrated weights
            - 'medium': Medium entropy (0.5) for balanced distribution
            - 'high': High entropy (0.8) for widely distributed weights
            - None: Uses default target_entropy (0.7)

    Returns:
        Configured EntropyRegularizer instance.

    Examples:
        >>> # Create a low-entropy regularizer for concentrated weights
        >>> reg = get_entropy_regularizer(strength=0.01, mode='low')

        >>> # Create a high-entropy regularizer for distributed weights
        >>> reg = get_entropy_regularizer(strength=0.02, mode='high')
    """
    # If specific target_entropy provided, use it directly
    if target_entropy is not None:
        return EntropyRegularizer(strength=strength, target_entropy=target_entropy)

    # Otherwise select based on mode
    if mode == 'low':
        return EntropyRegularizer(strength=strength, target_entropy=0.2)
    elif mode == 'medium':
        return EntropyRegularizer(strength=strength, target_entropy=0.5)
    elif mode == 'high':
        return EntropyRegularizer(strength=strength, target_entropy=0.8)
    else:
        return EntropyRegularizer(strength=strength, target_entropy=0.7)


# Register the regularizer with Keras
keras.utils.get_custom_objects()['EntropyRegularizer'] = EntropyRegularizer