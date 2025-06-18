"""
GoodhartAwareLoss: Information-Theoretic Loss Function for Robust Classification
================================================================================

This module implements a loss function that addresses Goodhart's Law
("When a measure becomes a target, it ceases to be a good measure") in machine
learning classification tasks through information-theoretic principles.

Goodhart's Law manifests in ML when models optimize for proxy metrics (like cross-entropy)
in ways that don't improve the true underlying objective. This can lead to overconfident
predictions, exploitation of spurious correlations, and poor generalization.

The GoodhartAwareLoss combines a standard cross-entropy loss with two information-theoretic
regularizers to encourage more robust learning:

1. **Standard Cross-Entropy**: The primary task loss that drives the model to be accurate.
   This implementation correctly operates on raw logits for numerical stability.

2. **Entropy Regularization** (λ = 0.01-0.2, default: 0.1): Explicitly encourages
   prediction uncertainty by maximizing the entropy H(p) = -Σ p_i log p_i of the
   output distribution. This prevents the model from collapsing to overly confident solutions.

   **Rationale:** Acts as a "pressure valve" against over-optimization. When the model
   tries to become too confident to minimize cross-entropy, the entropy term pushes back.

3. **Mutual Information Regularization** (β = 0.001-0.05, default: 0.01): Constrains
   the mutual information I(X;Y) between inputs and predictions to prevent memorization
   of spurious patterns, based on the Information Bottleneck principle.

   **Rationale:** Creates a "compression bottleneck" that forces the model to discard
   irrelevant information while preserving task-relevant patterns.

Mathematical Foundation:
- Total Loss: L_total = L_ce + λ * L_entropy + β * L_mi
- Cross-Entropy: L_ce = -Σ y_true * log(softmax(logits))
- Entropy Regularization: L_entropy = -H(p) = Σ p * log(p)
- Mutual Information Regularization: L_mi ≈ H(Y) - H(Y|X)

Usage:
    # The loss function operates on raw logits from the model
    loss_fn = GoodhartAwareLoss(entropy_weight=0.1, mi_weight=0.01)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

References:
- Goodhart's Law: https://en.wikipedia.org/wiki/Goodhart's_law
- Information Bottleneck: Tishby et al. (2000)
- On Calibration of Modern Neural Networks (discusses temperature scaling for POST-HOC calibration): Guo et al. (2017)
"""

import keras
import warnings
from keras import ops
from typing import Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GoodhartAwareLoss(keras.losses.Loss):
    """
    Information-theoretic loss function combining cross-entropy with regularization.

    This loss function combines standard cross-entropy with entropy and mutual
    information regularization to encourage robust, well-calibrated models.

    The total loss is:
    L_total = CrossEntropy(y, z) + λ_entropy * L_entropy(z) + λ_mi * L_mi(z)

    Where `z` are the raw logits from the model.

    Args:
        entropy_weight (float): Weight for the entropy regularization term.
            Controls how much the model is encouraged to maintain uncertainty.
            Typical range: [0.001, 0.5]. Default: 0.1
        mi_weight (float): Weight for the mutual information regularization term.
            Controls the compression of input-output information flow.
            Typical range: [0.001, 0.1]. Default: 0.01
        epsilon (float): Small constant for numerical stability in log operations.
            Should be much smaller than 1/num_classes. Default: 1e-8
        name (str): String name for the loss function. Default: 'goodhart_aware_loss'
        reduction (str): Type of reduction to apply to loss. Default: 'sum_over_batch_size'

    Raises:
        ValueError: If any parameter is outside its valid range.

    Example:
        >>> # Basic usage with a model that outputs logits
        >>> loss_fn = GoodhartAwareLoss(entropy_weight=0.05, mi_weight=0.005)
        >>> model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
        >>>
        >>> # Manual computation
        >>> y_true = ops.one_hot(ops.array([0, 1]), 3)  # shape: (2, 3)
        >>> y_pred_logits = ops.array([[2.0, 1.0, 0.5], [1.5, 3.0, 0.8]])
        >>> loss = loss_fn(y_true, y_pred_logits)
    """

    def __init__(
            self,
            entropy_weight: float = 0.1,
            mi_weight: float = 0.01,
            epsilon: float = 1e-8,
            name: str = 'goodhart_aware_loss',
            reduction: str = 'sum_over_batch_size'
    ) -> None:
        super().__init__(name=name, reduction=reduction)

        # Parameter validation
        if not isinstance(entropy_weight, (int, float)) or entropy_weight < 0:
            raise ValueError(
                f"Entropy weight must be a non-negative number, got {entropy_weight}. "
                f"Typical range: [0.001, 0.5]"
            )
        if not isinstance(mi_weight, (int, float)) or mi_weight < 0:
            raise ValueError(
                f"MI weight must be a non-negative number, got {mi_weight}. "
                f"Typical range: [0.001, 0.1]"
            )
        if not isinstance(epsilon, (int, float)) or epsilon <= 0 or epsilon >= 0.1:
            raise ValueError(
                f"Epsilon must be a small positive number (0, 0.1), got {epsilon}."
            )

        if entropy_weight > 1.0:
            warnings.warn(
                f"Very high entropy weight ({entropy_weight}) may dominate training. "
                f"Consider values in [0.001, 0.5]",
                UserWarning
            )

        self.entropy_weight = float(entropy_weight)
        self.mi_weight = float(mi_weight)
        self.epsilon = float(epsilon)

    def call(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute the Goodhart-aware loss. Expects logits

        Args:
            y_true (KerasTensor): Ground truth labels, shape (batch_size, num_classes).
            y_pred (KerasTensor): Predicted logits, shape (batch_size, num_classes).

        Returns:
            KerasTensor: Scalar tensor representing the total loss for the batch.
        """
        y_true = ops.cast(y_true, dtype=y_pred.dtype)

        # Component 1: Standard cross-entropy loss from logits
        # This is the primary task-oriented loss.
        ce_loss = keras.losses.categorical_crossentropy(
            y_true,
            y_pred,
            from_logits=True
        )

        # Component 2: Entropy regularization loss (only if weight > 0)
        entropy_loss = self._entropy_regularization(y_pred) if self.entropy_weight > 0 else 0.0

        # Component 3: Mutual information regularization loss (only if weight > 0)
        mi_loss = self._mutual_information_regularization(y_pred) if self.mi_weight > 0 else 0.0

        # Combine all components
        total_loss = (ce_loss +
                      self.entropy_weight * entropy_loss +
                      self.mi_weight * mi_loss)

        return total_loss

    def _entropy_regularization(self, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute entropy regularization loss from logits.
        The goal is to maximize entropy, so we minimize its negative.
        L_entropy = -H(p)
        """
        probs = ops.softmax(y_pred, axis=-1)
        probs = ops.clip(probs, self.epsilon, 1.0 - self.epsilon)
        entropy_per_sample = -ops.sum(probs * ops.log(probs), axis=-1)
        # Return negative entropy; minimizing this maximizes entropy.
        return -ops.mean(entropy_per_sample)

    def _mutual_information_regularization(
            self,
            y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Approximate mutual information I(X;Y) to regularize the model.
        I(X;Y) ≈ H(Y) - H(Y|X)
        """
        probs = ops.softmax(y_pred, axis=-1)
        probs = ops.clip(probs, self.epsilon, 1.0 - self.epsilon)

        # H(Y|X): Conditional entropy (average uncertainty of individual predictions)
        conditional_entropy = -ops.sum(probs * ops.log(probs), axis=-1)
        h_y_given_x = ops.mean(conditional_entropy)

        # H(Y): Marginal entropy (entropy of the average prediction across the batch)
        mean_probs = ops.mean(probs, axis=0)
        mean_probs = ops.clip(mean_probs, self.epsilon, 1.0 - self.epsilon)
        h_y = -ops.sum(mean_probs * ops.log(mean_probs))

        # Approximate mutual information I(X;Y)
        mutual_information = h_y - h_y_given_x

        # Only penalize positive MI to prevent the model from becoming too deterministic.
        return ops.maximum(0.0, mutual_information)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization."""
        config = super().get_config()
        config.update({
            'entropy_weight': self.entropy_weight,
            'mi_weight': self.mi_weight,
            'epsilon': self.epsilon
        })
        return config

# ---------------------------------------------------------------------

def analyze_loss_components(
        loss_fn: GoodhartAwareLoss,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor
) -> Dict[str, float]:
    """
    Analyze individual components of the GoodhartAwareLoss for debugging.

    Args:
        loss_fn (GoodhartAwareLoss): The loss function instance.
        y_true (KerasTensor): True labels.
        y_pred (KerasTensor): Predicted logits.

    Returns:
        Dict[str, float]: A dictionary with individual loss components.
    """
    # Compute individual components
    ce_loss = keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
    entropy_loss = loss_fn._entropy_regularization(y_pred)
    mi_loss = loss_fn._mutual_information_regularization(y_pred)

    # Compute weighted contributions
    weighted_entropy = loss_fn.entropy_weight * entropy_loss
    weighted_mi = loss_fn.mi_weight * mi_loss
    total_loss = ce_loss + weighted_entropy + weighted_mi

    # Ensure total_loss is not zero to avoid division by zero
    safe_total_loss = ops.where(ops.equal(total_loss, 0), 1.0, total_loss)

    # Convert tensors to Python floats for easy inspection
    results = {
        'total_loss': float(ops.convert_to_numpy(total_loss)),
        'cross_entropy': float(ops.convert_to_numpy(ce_loss)),
        'entropy_loss': float(ops.convert_to_numpy(entropy_loss)),
        'mi_loss': float(ops.convert_to_numpy(mi_loss)),
        'weighted_entropy': float(ops.convert_to_numpy(weighted_entropy)),
        'weighted_mi': float(ops.convert_to_numpy(weighted_mi)),
        'entropy_weight': loss_fn.entropy_weight,
        'mi_weight': loss_fn.mi_weight
    }
    results.update({
        'ce_contribution_pct': (results['cross_entropy'] / results['total_loss']) * 100 if results['total_loss'] != 0 else 0,
        'entropy_contribution_pct': (results['weighted_entropy'] / results['total_loss']) * 100 if results['total_loss'] != 0 else 0,
        'mi_contribution_pct': (results['weighted_mi'] / results['total_loss']) * 100 if results['total_loss'] != 0 else 0
    })
    return results

# ---------------------------------------------------------------------

def suggest_hyperparameters(
        num_classes: int,
        dataset_size: int,
        model_complexity: str = 'medium'
) -> Dict[str, float]:
    """
    Suggest initial hyperparameters based on problem characteristics.
    These are starting points and may require further tuning.

    Args:
        num_classes (int): Number of output classes.
        dataset_size (int): Size of the training dataset.
        model_complexity (str): 'simple', 'medium', or 'complex'.

    Returns:
        Dict[str, float]: A dictionary with suggested hyperparameters.
    """
    params = {'entropy_weight': 0.1, 'mi_weight': 0.01}
    # Adjust for dataset size
    if dataset_size < 10000:
        params['entropy_weight'] *= 1.5 # More regularization for smaller datasets
        params['mi_weight'] *= 1.5
    elif dataset_size > 200000:
        params['entropy_weight'] *= 0.5 # Less regularization for larger datasets
        params['mi_weight'] *= 0.5
    # Adjust for model complexity
    complexity_map = {'simple': 0.75, 'medium': 1.0, 'complex': 1.25}
    if model_complexity in complexity_map:
        multiplier = complexity_map[model_complexity]
        params['entropy_weight'] *= multiplier
        params['mi_weight'] *= multiplier
    return params

# ---------------------------------------------------------------------
