"""
DecoupledInformationLoss: An Information-Theoretic Loss for Robust Classification
===================================================================================

This module implements a refined information-theoretic loss function designed to
improve model robustness and calibration by regularizing two orthogonal concepts:
**prediction uncertainty** and **prediction diversity**.

This loss function is a more intuitive and tunable alternative to composite losses
where the regularization effects are coupled and difficult to reason about.

.. caution::
    This is an advanced loss function whose effectiveness is **highly sensitive
    to its hyperparameters and the specific task**. It is not a guaranteed
    drop-in replacement for Cross-Entropy. It is recommended to perform a grid
    search on its weights for any new application.

Decoupled Components
--------------------
1.  **Cross-Entropy (CE)**: The primary task loss that drives model accuracy.

2.  **Uncertainty Regularization (Conditional Entropy)**: Maximizes the entropy
    of each individual prediction, :math:`H(p(\\hat{Y}|X))`. This penalizes
    overconfident (low-entropy) predictions, which can improve model calibration
    and reduce overfitting to noisy labels.
    - **Mechanism**: Controlled by ``uncertainty_weight`` (:math:`\\gamma`). It acts
      as a "pressure valve" against overconfidence. (Inspired by Pereyra et al., 2017).

3.  **Diversity Regularization (Marginal Entropy)**: Maximizes the entropy of the
    *batch-averaged* predictive distribution, :math:`H(p(\\hat{Y}))`. This
    encourages the model to utilize all its output classes over a batch,
    preventing "mode collapse" where the model confidently predicts only a few
    classes, regardless of the input.
    - **Mechanism**: Controlled by ``diversity_weight`` (:math:`\\delta`). It acts
      as a "diversity driver," forcing the model to explore its full output space.

Mathematical Foundation
-----------------------
The total loss is a weighted combination of these three decoupled components, where
the goal is to minimize CE while maximizing the two entropy terms:

.. math::
    L_{total} = L_{CE} - \\gamma H(p(\\hat{Y}|X)) - \\delta H(p(\\hat{Y}))

Where:
- :math:`L_{CE}` is the categorical cross-entropy.
- :math:`H(p(\\hat{Y}|X))` is the conditional entropy (mean entropy of per-sample predictions).
- :math:`H(p(\\hat{Y}))` is the marginal entropy (entropy of the batch-averaged prediction).
- :math:`\\gamma` and :math:`\\delta` are the decoupled regularization weights.

Practical Considerations & Tuning Guide
---------------------------------------
- **Tuning Strategy**: The decoupled parameters are easier to reason about.
  1. **Tune Uncertainty (:math:`\\gamma`)**: If your model is overconfident (low
     mean entropy, high ECE), increase ``uncertainty_weight``. Good starting
     values are in the range `[0.1, 0.5]`.
  2. **Tune Diversity (:math:`\\delta`)**: If your model's predictions over the test
     set are heavily skewed to a few classes, increase ``diversity_weight``.
     This is a finer-grained control, so start with smaller values like `[0.001, 0.1]`.
- **Grid Search**: For best results, perform a 2D grid search over a range of
  :math:`\\gamma` and :math:`\\delta` values.
- **Monitor Components**: Use the `analyze_loss_components` function to track the
  unweighted values of `H(p(Y|X))` and `H(p(Y))` to ensure they behave as expected.

Example:
    >>> import keras
    >>> import numpy as np
    >>> from dl_techniques.utils.logger import logger
    >>>
    >>> # Create loss function
    >>> loss_fn = DecoupledInformationLoss(
    ...     uncertainty_weight=0.2,
    ...     diversity_weight=0.01,
    ...     label_smoothing=0.1
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
    >>> components = analyze_decoupled_information_loss(loss_fn, y_true, y_pred)
    >>> logger.info(f"Loss components: {components}")
"""

import keras
from keras import ops
from typing import Dict, Any
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="dl_techniques.losses")
class DecoupledInformationLoss(keras.losses.Loss):
    """A decoupled information-theoretic loss combining cross-entropy with regularization.

    This loss function augments standard cross-entropy with two orthogonal
    information-theoretic regularizers: one for prediction uncertainty and one
    for prediction diversity.

    The total loss is calculated as:
    `L = CE(y, ŷ) - uncertainty_weight * H(ŷ|x) - diversity_weight * H(ŷ)`

    Args:
        uncertainty_weight: Weight (γ) for the conditional entropy term H(p(Y|X)).
            Controls how much the model is penalized for overconfident predictions.
            Higher values promote calibration. Suggested range: [0.1, 1.0].
        diversity_weight: Weight (δ) for the marginal entropy term H(p(Y)).
            Controls how much the model is encouraged to produce diverse predictions
            across a batch, preventing mode collapse. Suggested range: [0.001, 0.1].
        label_smoothing: Factor for label smoothing applied to the cross-entropy
            component. Must be in [0, 1). Defaults to 0.0.
        from_logits: Whether `y_pred` is a tensor of logits or probabilities.
            Defaults to ``True``.
        epsilon: A small constant for numerical stability in log operations.
        name: String name for the loss function.
        reduction: Type of reduction to apply to loss.

    Raises:
        ValueError: If any parameter is outside its valid range.

    Example:
        >>> loss_fn = DecoupledInformationLoss(
        ...     uncertainty_weight=0.2,
        ...     diversity_weight=0.01
        ... )
        >>> y_true = keras.utils.to_categorical([0, 1, 2], 3)
        >>> y_pred = [[2.0, 1.0, 0.1], [0.5, 2.1, 0.3], [0.2, 0.8, 1.9]]
        >>> loss = loss_fn(y_true, y_pred)
    """

    def __init__(
            self,
            uncertainty_weight: float = 0.2,
            diversity_weight: float = 0.01,
            label_smoothing: float = 0.0,
            from_logits: bool = True,
            epsilon: float = 1e-8,
            name: str = "decoupled_information_loss",
            reduction: str = "sum_over_batch_size",
            **kwargs: Any,
    ) -> None:
        """Initialize the DecoupledInformationLoss."""
        super().__init__(name=name, reduction=reduction, **kwargs)

        # Parameter validation with detailed error messages
        if uncertainty_weight < 0:
            raise ValueError(
                f"uncertainty_weight must be non-negative, got {uncertainty_weight}"
            )
        if diversity_weight < 0:
            raise ValueError(
                f"diversity_weight must be non-negative, got {diversity_weight}"
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
        self.diversity_weight = float(diversity_weight)
        self.label_smoothing = float(label_smoothing)
        self.from_logits = bool(from_logits)
        self.epsilon = float(epsilon)

        # Log configuration for debugging
        logger.debug(
            f"DecoupledInformationLoss initialized with "
            f"uncertainty_weight={self.uncertainty_weight}, "
            f"diversity_weight={self.diversity_weight}, "
            f"label_smoothing={self.label_smoothing}, "
            f"from_logits={self.from_logits}"
        )

    def call(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute the decoupled information-theoretic loss for a batch.

        Args:
            y_true: Ground truth labels with shape (batch_size, num_classes).
            y_pred: Predicted logits or probabilities with shape (batch_size, num_classes).

        Returns:
            Computed loss tensor with shape matching the reduction setting.
        """
        # Ensure consistent dtype for numerical stability
        y_true = ops.cast(y_true, dtype=y_pred.dtype)

        # ============================================================================
        # COMPONENT 1: CROSS-ENTROPY LOSS COMPUTATION
        # ============================================================================
        # Standard categorical cross-entropy serves as the primary task loss
        # This drives the model toward correct classifications
        ce_loss = keras.losses.categorical_crossentropy(
            y_true,
            y_pred,
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing,
        )

        # ============================================================================
        # PROBABILITY CONVERSION FOR ENTROPY COMPUTATIONS
        # ============================================================================
        # For entropy calculations, we need probability distributions
        # Convert logits to probabilities using softmax if necessary
        if self.from_logits:
            # Apply softmax to convert logits to probabilities
            # softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
            probs = ops.softmax(y_pred, axis=-1)
        else:
            # Input is already probabilities, but clip for numerical stability
            # Prevent log(0) by ensuring all probabilities are in [epsilon, 1-epsilon]
            probs = ops.clip(y_pred, self.epsilon, 1.0 - self.epsilon)

        # ============================================================================
        # COMPONENT 2: CONDITIONAL ENTROPY H(Y|X) COMPUTATION
        # ============================================================================
        # Conditional entropy measures the average uncertainty in individual predictions
        # H(Y|X) = -∑_{i} p(y_i|x) * log(p(y_i|x)) averaged over all samples
        # Higher conditional entropy indicates less overconfident predictions

        # Compute per-sample entropy: -∑ p_i * log(p_i) for each sample
        # Shape: (batch_size, num_classes) -> (batch_size,)
        conditional_entropy_per_sample = -ops.sum(
            probs * ops.log(probs + self.epsilon), axis=-1
        )

        # Average across the batch to get mean conditional entropy
        # Shape: (batch_size,) -> scalar
        h_conditional = ops.mean(conditional_entropy_per_sample)

        # ============================================================================
        # COMPONENT 3: MARGINAL ENTROPY H(Y) COMPUTATION
        # ============================================================================
        # Marginal entropy measures the diversity of predictions across the batch
        # H(Y) = -∑_{i} p̄(y_i) * log(p̄(y_i)) where p̄(y_i) is the average probability
        # Higher marginal entropy indicates more diverse class predictions

        # Compute batch-averaged probability distribution
        # Shape: (batch_size, num_classes) -> (num_classes,)
        mean_probs = ops.mean(probs, axis=0)

        # Compute entropy of the averaged distribution
        # Shape: (num_classes,) -> scalar
        h_marginal = -ops.sum(mean_probs * ops.log(mean_probs + self.epsilon))

        # ============================================================================
        # LOSS COMBINATION: MINIMIZE CE, MAXIMIZE ENTROPIES
        # ============================================================================
        # The objective is to minimize cross-entropy while maximizing both entropy terms
        # This is achieved by subtracting the weighted entropy terms from CE loss
        #
        # Mathematical formulation:
        # L_total = L_CE - γ * H(Y|X) - δ * H(Y)
        #
        # Intuition:
        # - Minimizing CE ensures correct predictions
        # - Maximizing H(Y|X) prevents overconfidence (uncertainty regularization)
        # - Maximizing H(Y) encourages diverse predictions (diversity regularization)

        uncertainty_regularization = self.uncertainty_weight * h_conditional
        diversity_regularization = self.diversity_weight * h_marginal

        # Combine all components (subtract regularization terms to maximize entropy)
        total_loss = ce_loss - uncertainty_regularization - diversity_regularization

        return total_loss

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration dictionary for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "uncertainty_weight": self.uncertainty_weight,
                "diversity_weight": self.diversity_weight,
                "label_smoothing": self.label_smoothing,
                "from_logits": self.from_logits,
                "epsilon": self.epsilon,
            }
        )
        return config

# ---------------------------------------------------------------------

def analyze_decoupled_information_loss(
        loss_fn: DecoupledInformationLoss,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
) -> Dict[str, float]:
    """Analyze individual components of the DecoupledInformationLoss for debugging.

    This function breaks down the loss into its constituent components to help
    understand how each part contributes to the total loss. This is particularly
    useful for hyperparameter tuning and debugging.

    Args:
        loss_fn: An instance of the DecoupledInformationLoss function.
        y_true: Ground truth labels with shape (batch_size, num_classes).
        y_pred: Predicted logits or probabilities with shape (batch_size, num_classes).

    Returns:
        A dictionary containing the unweighted and weighted values of each
        loss component for analysis.

    Example:
        >>> loss_fn = DecoupledInformationLoss(uncertainty_weight=0.2, diversity_weight=0.01)
        >>> y_true = keras.utils.to_categorical([0, 1, 2], 3)
        >>> y_pred = [[2.0, 1.0, 0.1], [0.5, 2.1, 0.3], [0.2, 0.8, 1.9]]
        >>> analysis = analyze_decoupled_information_loss(loss_fn, y_true, y_pred)
        >>> print(f"Cross-entropy: {analysis['cross_entropy']:.4f}")
        >>> print(f"Conditional entropy: {analysis['h_conditional_unweighted']:.4f}")
    """
    # Ensure consistent dtype
    y_true = ops.cast(y_true, dtype=y_pred.dtype)
    epsilon = loss_fn.epsilon

    # ============================================================================
    # CROSS-ENTROPY COMPONENT ANALYSIS
    # ============================================================================
    # Compute the standard cross-entropy loss without regularization
    ce_loss_val = keras.losses.categorical_crossentropy(
        y_true,
        y_pred,
        from_logits=loss_fn.from_logits,
        label_smoothing=loss_fn.label_smoothing,
    )

    # ============================================================================
    # PROBABILITY CONVERSION FOR ANALYSIS
    # ============================================================================
    # Convert predictions to probabilities for entropy calculations
    if loss_fn.from_logits:
        # Apply softmax transformation to logits
        probs = ops.softmax(y_pred, axis=-1)
    else:
        # Clip probabilities for numerical stability
        probs = ops.clip(y_pred, epsilon, 1.0 - epsilon)

    # ============================================================================
    # CONDITIONAL ENTROPY ANALYSIS H(Y|X)
    # ============================================================================
    # Compute the unweighted conditional entropy (uncertainty measure)
    # This represents the average entropy of individual predictions
    conditional_entropy_per_sample = -ops.sum(
        probs * ops.log(probs + epsilon), axis=-1
    )
    h_conditional_unweighted = ops.mean(conditional_entropy_per_sample)

    # Calculate the weighted contribution to the loss
    # Note: This is negative because we subtract it from the total loss
    uncertainty_term_weighted = -loss_fn.uncertainty_weight * h_conditional_unweighted

    # ============================================================================
    # MARGINAL ENTROPY ANALYSIS H(Y)
    # ============================================================================
    # Compute the unweighted marginal entropy (diversity measure)
    # This represents the entropy of the batch-averaged prediction distribution
    mean_probs = ops.mean(probs, axis=0)
    h_marginal_unweighted = -ops.sum(mean_probs * ops.log(mean_probs + epsilon))

    # Calculate the weighted contribution to the loss
    # Note: This is negative because we subtract it from the total loss
    diversity_term_weighted = -loss_fn.diversity_weight * h_marginal_unweighted

    # ============================================================================
    # TOTAL LOSS COMPUTATION
    # ============================================================================
    # Combine all components to get the total loss
    total_loss = ops.mean(
        ce_loss_val + uncertainty_term_weighted + diversity_term_weighted
    )

    # ============================================================================
    # CONVERT TO PYTHON SCALARS FOR ANALYSIS
    # ============================================================================
    # Convert all tensor values to Python floats for easy inspection and logging
    results = {
        "total_loss": float(ops.convert_to_numpy(total_loss)),
        "cross_entropy": float(ops.convert_to_numpy(ops.mean(ce_loss_val))),
        "h_conditional_unweighted": float(ops.convert_to_numpy(h_conditional_unweighted)),
        "h_marginal_unweighted": float(ops.convert_to_numpy(h_marginal_unweighted)),
        "uncertainty_term_weighted": float(ops.convert_to_numpy(uncertainty_term_weighted)),
        "diversity_term_weighted": float(ops.convert_to_numpy(diversity_term_weighted)),
        "uncertainty_weight": loss_fn.uncertainty_weight,
        "diversity_weight": loss_fn.diversity_weight,
    }

    # Log the analysis results for debugging
    logger.debug(f"Loss analysis: {results}")

    return results

# ---------------------------------------------------------------------
