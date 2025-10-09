"""
FocalUncertaintyLoss: Focal Loss with Uncertainty Regularization
================================================================

This module implements a focused information-theoretic loss function that combines
Focal Loss with uncertainty regularization to improve model robustness and calibration.

This approach addresses two key challenges in classification:
1. **Class Imbalance**: Handled by Focal Loss's focusing mechanism
2. **Overconfidence**: Addressed by uncertainty regularization via conditional entropy

The loss function is designed to be more practical and theoretically sound than
composite losses with multiple conflicting objectives.

Components
----------
1. **Focal Loss**: Addresses class imbalance by focusing on hard examples and
   down-weighting easy examples. The focusing parameter γ reduces the loss
   contribution from well-classified examples.

2. **Uncertainty Regularization**: Maximizes the entropy of individual predictions
   H(p(Ŷ|X)) to prevent overconfident predictions and improve calibration.

Mathematical Foundation
-----------------------
The total loss combines focal loss with uncertainty regularization:

.. math::
    L_{total} = L_{focal} - \\gamma H(p(\\hat{Y}|X))

Where:
- :math:`L_{focal} = -\\alpha(1-p_t)^{\\gamma}\\log(p_t)` is the focal loss
- :math:`H(p(\\hat{Y}|X))` is the conditional entropy (mean entropy of per-sample predictions)
- :math:`\\gamma` is the uncertainty regularization weight

Benefits
--------
- **Focused Learning**: Focal loss emphasizes hard examples and rare classes
- **Calibration**: Uncertainty regularization prevents overconfidence
- **Simplicity**: Single regularization parameter to tune
- **Stability**: No batch-dependent computations

Practical Considerations
------------------------
- **Focal Loss Parameters**:
  - `alpha`: Class weighting (0.25 works well for most cases)
  - `gamma`: Focusing parameter (2.0 is the standard choice)
- **Uncertainty Weight**: Start with values in [0.1, 0.5] range
- **Monitoring**: Track both accuracy and calibration metrics (ECE)

Example:
    >>> import keras
    >>> import numpy as np
    >>> from dl_techniques.utils.logger import logger
    >>>
    >>> # Create loss function
    >>> loss_fn = FocalUncertaintyLoss(
    ...     uncertainty_weight=0.2,
    ...     alpha=0.25,
    ...     gamma=2.0
    ... )
    >>>
    >>> # Synthetic data
    >>> y_true = keras.utils.to_categorical(np.random.randint(0, 10, 32), 10)
    >>> y_pred = np.random.normal(0, 1, (32, 10))
    >>>
    >>> # Compute loss
    >>> loss = loss_fn(y_true, y_pred)
    >>> logger.info(f"Total loss: {loss}")
    >>>
    >>> # Analyze components
    >>> components = analyze_focal_uncertainty_loss(loss_fn, y_true, y_pred)
    >>> logger.info(f"Loss components: {components}")
"""

import keras
from keras import ops
from typing import Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="dl_techniques.losses")
class FocalUncertaintyLoss(keras.losses.Loss):
    """Focal Loss with uncertainty regularization for robust classification.

    This loss function combines Focal Loss (for class imbalance) with uncertainty
    regularization (for calibration) to create a robust classification loss.

    The total loss is calculated as:
    `L = FocalLoss(y, ŷ) - uncertainty_weight * H(ŷ|x)`

    Args:
        uncertainty_weight: Weight (γ) for the conditional entropy term H(p(Y|X)).
            Controls how much the model is penalized for overconfident predictions.
            Higher values promote calibration. Suggested range: [0.1, 0.5].
        alpha: Weighting factor for rare class (alpha=0.25 works well for most cases).
            Can be a scalar or array of shape (num_classes,) for per-class weights.
        gamma: Focusing parameter for focal loss. Higher gamma puts more focus on
            hard examples. Standard value is 2.0.
        label_smoothing: Factor for label smoothing applied to the focal loss
            component. Must be in [0, 1). Defaults to 0.0.
        from_logits: Whether `y_pred` is a tensor of logits or probabilities.
            Defaults to ``True``.
        epsilon: A small constant for numerical stability in entropy calculations.
        name: String name for the loss function.
        reduction: Type of reduction to apply to loss.

    Raises:
        ValueError: If any parameter is outside its valid range.

    Example:
        >>> loss_fn = FocalUncertaintyLoss(
        ...     uncertainty_weight=0.2,
        ...     alpha=0.25,
        ...     gamma=2.0
        ... )
        >>> y_true = keras.utils.to_categorical([0, 1, 2], 3)
        >>> y_pred = [[2.0, 1.0, 0.1], [0.5, 2.1, 0.3], [0.2, 0.8, 1.9]]
        >>> loss = loss_fn(y_true, y_pred)
    """

    def __init__(
        self,
        uncertainty_weight: float = 0.2,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        from_logits: bool = True,
        epsilon: float = 1e-8,
        name: str = "focal_uncertainty_loss",
        reduction: str = "sum_over_batch_size",
        **kwargs: Any,
    ) -> None:
        """Initialize the FocalUncertaintyLoss."""
        super().__init__(name=name, reduction=reduction, **kwargs)

        # Parameter validation with detailed error messages
        if uncertainty_weight < 0:
            raise ValueError(
                f"uncertainty_weight must be non-negative, got {uncertainty_weight}"
            )
        if alpha < 0 or alpha > 1:
            raise ValueError(
                f"alpha must be in [0, 1], got {alpha}"
            )
        if gamma < 0:
            raise ValueError(
                f"gamma must be non-negative, got {gamma}"
            )
        if not (0 <= label_smoothing < 1):
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {label_smoothing}"
            )
        if not (0 < epsilon < 0.1):
            raise ValueError(
                f"epsilon must be a small positive number in (0, 0.1), got {epsilon}"
            )

        # Store configuration parameters
        self.uncertainty_weight = float(uncertainty_weight)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.from_logits = bool(from_logits)
        self.epsilon = float(epsilon)

        # Log configuration for debugging
        logger.debug(
            f"FocalUncertaintyLoss initialized with "
            f"uncertainty_weight={self.uncertainty_weight}, "
            f"alpha={self.alpha}, gamma={self.gamma}, "
            f"label_smoothing={self.label_smoothing}, "
            f"from_logits={self.from_logits}"
        )

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute the focal loss with uncertainty regularization.

        Args:
            y_true: Ground truth labels with shape (batch_size, num_classes).
            y_pred: Predicted logits or probabilities with shape (batch_size, num_classes).

        Returns:
            Computed loss tensor with shape matching the reduction setting.
        """
        # Ensure consistent dtype for numerical stability
        y_true = ops.cast(y_true, dtype=y_pred.dtype)

        # ============================================================================
        # COMPONENT 1: FOCAL LOSS COMPUTATION
        # ============================================================================
        # Focal loss addresses class imbalance by focusing on hard examples
        # FL(p_t) = -α(1-p_t)^γ log(p_t)
        # where p_t is the model's estimated probability for the ground truth class
        #
        # Benefits:
        # - Down-weights easy examples (high p_t)
        # - Focuses learning on hard examples (low p_t)
        # - Handles class imbalance through α parameter
        # - Reduces gradient contribution from well-classified examples
        focal_loss = keras.losses.categorical_focal_crossentropy(
            y_true,
            y_pred,
            alpha=self.alpha,
            gamma=self.gamma,
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing,
        )

        # ============================================================================
        # COMPONENT 2: UNCERTAINTY REGULARIZATION COMPUTATION
        # ============================================================================
        # Skip uncertainty regularization if weight is zero (optimization)
        if self.uncertainty_weight == 0:
            return focal_loss

        # Convert to probabilities for entropy calculation
        if self.from_logits:
            # Apply softmax to convert logits to probabilities
            # softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
            probs = ops.softmax(y_pred, axis=-1)
        else:
            # Input is already probabilities, but clip for numerical stability
            # Prevent log(0) by ensuring all probabilities are in [epsilon, 1-epsilon]
            probs = ops.clip(y_pred, self.epsilon, 1.0 - self.epsilon)

        # ============================================================================
        # CONDITIONAL ENTROPY H(Y|X) COMPUTATION
        # ============================================================================
        # Conditional entropy measures the average uncertainty in individual predictions
        # H(Y|X) = -∑_{i} p(y_i|x) * log(p(y_i|x)) averaged over all samples
        #
        # Purpose: Prevents overconfident predictions by maximizing prediction entropy
        # Higher conditional entropy indicates less overconfident predictions
        # This acts as a regularization term that improves model calibration

        # Compute per-sample entropy: -∑ p_i * log(p_i) for each sample
        # Shape: (batch_size, num_classes) -> (batch_size,)
        conditional_entropy_per_sample = -ops.sum(
            probs * ops.log(probs + self.epsilon), axis=-1
        )

        # Average across the batch to get mean conditional entropy
        # Shape: (batch_size,) -> scalar
        h_conditional = ops.mean(conditional_entropy_per_sample)

        # ============================================================================
        # LOSS COMBINATION: MINIMIZE FOCAL LOSS, MAXIMIZE CONDITIONAL ENTROPY
        # ============================================================================
        # The objective is to minimize focal loss while maximizing conditional entropy
        # This is achieved by subtracting the weighted entropy term from focal loss
        #
        # Mathematical formulation:
        # L_total = L_focal - γ * H(Y|X)
        #
        # Intuition:
        # - Focal loss handles class imbalance and focuses on hard examples
        # - Uncertainty regularization prevents overconfident predictions
        # - The combination improves both accuracy and calibration

        uncertainty_regularization = self.uncertainty_weight * h_conditional

        # Combine focal loss with uncertainty regularization
        # Subtract regularization term to maximize entropy (minimize negative entropy)
        total_loss = focal_loss - uncertainty_regularization

        return total_loss

    def compute_output_shape(self, input_shape) -> tuple:
        """Compute the output shape of the loss.

        Args:
            input_shape: Shape of the input tensors.

        Returns:
            Output shape tuple.
        """
        # Loss function returns a scalar or tensor based on reduction
        return ()

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration dictionary for serialization.

        This method enables the loss function to be saved and loaded with models.
        It serializes all the constructor parameters needed to recreate the loss
        function with the same configuration.

        Returns:
            Dictionary containing the loss function configuration with all
            parameters needed for reconstruction.

        Example:
            >>> loss_fn = FocalUncertaintyLoss(uncertainty_weight=0.2, alpha=0.25)
            >>> config = loss_fn.get_config()
            >>> print(config)
            {
                'name': 'focal_uncertainty_loss',
                'reduction': 'sum_over_batch_size',
                'uncertainty_weight': 0.2,
                'alpha': 0.25,
                'gamma': 2.0,
                'label_smoothing': 0.0,
                'from_logits': True,
                'epsilon': 1e-08
            }
            >>> # Recreate the loss function from config
            >>> new_loss_fn = FocalUncertaintyLoss.from_config(config)
        """
        # Get base configuration from parent class (includes name, reduction, etc.)
        config = super().get_config()

        # Add all custom parameters for this loss function
        config.update(
            {
                "uncertainty_weight": self.uncertainty_weight,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "label_smoothing": self.label_smoothing,
                "from_logits": self.from_logits,
                "epsilon": self.epsilon,
            }
        )

        logger.debug(f"Serializing FocalUncertaintyLoss config: {config}")
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FocalUncertaintyLoss":
        """Create a FocalUncertaintyLoss instance from a configuration dictionary.

        This class method enables reconstruction of the loss function from its
        serialized configuration, which is essential for model saving/loading.

        Args:
            config: Configuration dictionary containing all parameters needed
                to reconstruct the loss function.

        Returns:
            A new FocalUncertaintyLoss instance configured with the provided parameters.

        Example:
            >>> config = {
            ...     'uncertainty_weight': 0.2,
            ...     'alpha': 0.25,
            ...     'gamma': 2.0,
            ...     'from_logits': True
            ... }
            >>> loss_fn = FocalUncertaintyLoss.from_config(config)
        """
        logger.debug(f"Deserializing FocalUncertaintyLoss from config: {config}")
        return cls(**config)

# ---------------------------------------------------------------------

def analyze_focal_uncertainty_loss(
    loss_fn: FocalUncertaintyLoss,
    y_true: keras.KerasTensor,
    y_pred: keras.KerasTensor,
) -> Dict[str, float]:
    """Analyze individual components of the FocalUncertaintyLoss for debugging.

    This function breaks down the loss into its constituent components to help
    understand how each part contributes to the total loss. This is particularly
    useful for hyperparameter tuning and understanding training dynamics.

    Args:
        loss_fn: An instance of the FocalUncertaintyLoss function.
        y_true: Ground truth labels with shape (batch_size, num_classes).
        y_pred: Predicted logits or probabilities with shape (batch_size, num_classes).

    Returns:
        A dictionary containing the values of each loss component for analysis.

    Example:
        >>> loss_fn = FocalUncertaintyLoss(uncertainty_weight=0.2, alpha=0.25, gamma=2.0)
        >>> y_true = keras.utils.to_categorical([0, 1, 2], 3)
        >>> y_pred = [[2.0, 1.0, 0.1], [0.5, 2.1, 0.3], [0.2, 0.8, 1.9]]
        >>> analysis = analyze_focal_uncertainty_loss(loss_fn, y_true, y_pred)
        >>> print(f"Focal loss: {analysis['focal_loss']:.4f}")
        >>> print(f"Conditional entropy: {analysis['h_conditional_unweighted']:.4f}")
    """
    # Ensure consistent dtype
    y_true = ops.cast(y_true, dtype=y_pred.dtype)
    epsilon = loss_fn.epsilon

    # ============================================================================
    # FOCAL LOSS COMPONENT ANALYSIS
    # ============================================================================
    # Compute the focal loss component without uncertainty regularization
    focal_loss_val = keras.losses.categorical_focal_crossentropy(
        y_true,
        y_pred,
        alpha=loss_fn.alpha,
        gamma=loss_fn.gamma,
        from_logits=loss_fn.from_logits,
        label_smoothing=loss_fn.label_smoothing,
    )

    # ============================================================================
    # UNCERTAINTY REGULARIZATION ANALYSIS
    # ============================================================================
    # Only compute if uncertainty weight is non-zero
    if loss_fn.uncertainty_weight > 0:
        # Convert to probabilities for entropy calculation
        if loss_fn.from_logits:
            # Apply softmax transformation to logits
            probs = ops.softmax(y_pred, axis=-1)
        else:
            # Clip probabilities for numerical stability
            probs = ops.clip(y_pred, epsilon, 1.0 - epsilon)

        # Compute the unweighted conditional entropy (uncertainty measure)
        # This represents the average entropy of individual predictions
        conditional_entropy_per_sample = -ops.sum(
            probs * ops.log(probs + epsilon), axis=-1
        )
        h_conditional_unweighted = ops.mean(conditional_entropy_per_sample)

        # Calculate the weighted contribution to the loss
        # Note: This is negative because we subtract it from the total loss
        uncertainty_term_weighted = -loss_fn.uncertainty_weight * h_conditional_unweighted
    else:
        h_conditional_unweighted = 0.0
        uncertainty_term_weighted = 0.0

    # ============================================================================
    # TOTAL LOSS COMPUTATION
    # ============================================================================
    # Combine focal loss with uncertainty regularization
    total_loss = ops.mean(focal_loss_val + uncertainty_term_weighted)

    # ============================================================================
    # CONVERT TO PYTHON SCALARS FOR ANALYSIS
    # ============================================================================
    # Convert all tensor values to Python floats for easy inspection and logging
    results = {
        "total_loss": float(ops.convert_to_numpy(total_loss)),
        "focal_loss": float(ops.convert_to_numpy(ops.mean(focal_loss_val))),
        "h_conditional_unweighted": float(ops.convert_to_numpy(h_conditional_unweighted)),
        "uncertainty_term_weighted": float(ops.convert_to_numpy(uncertainty_term_weighted)),
        "uncertainty_weight": loss_fn.uncertainty_weight,
        "focal_alpha": loss_fn.alpha,
        "focal_gamma": loss_fn.gamma,
    }

    # Log the analysis results for debugging
    logger.debug(f"Focal uncertainty loss analysis: {results}")

    return results

# ---------------------------------------------------------------------
