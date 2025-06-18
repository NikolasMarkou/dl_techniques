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
regularizers to encourage more robust learning. This approach synthesizes established
regularization techniques into a single, unified loss function designed to provide a more
holistic training objective than cross-entropy alone.

1. **Standard Cross-Entropy**: The primary task loss that drives the model to be accurate.
   This implementation correctly operates on raw logits for numerical stability.

2. **Entropy Regularization** (λ = 0.01-0.2, default: 0.1): Explicitly encourages
   prediction uncertainty by maximizing the entropy H(p) = -Σ p_i log p_i of the
   output distribution. This prevents the model from collapsing to overly confident,
   brittle solutions. This component is inspired by techniques used to improve model
   calibration and exploration (e.g., Pereyra et al., 2017).

   **Rationale:** Acts as a "pressure valve" against over-optimization. When the model
   tries to become too certain to minimize cross-entropy, the entropy term pushes back,
   improving calibration and robustness to noisy labels.

3. **Mutual Information Regularization** (β = 0.001-0.05, default: 0.01): Constrains
   the mutual information I(X;Ŷ) between the inputs (X) and the model's predictions (Ŷ),
   based on the Information Bottleneck principle (Tishby et al., 2000). By penalizing
   this mutual information, the model is forced to compress the input, retaining only
   the most essential information required for the task.

   **Rationale:** Creates a "compression bottleneck" that forces the model to discard
   irrelevant information and spurious correlations while preserving task-relevant
   patterns, leading to better generalization.

Mathematical Foundation:
- Total Loss: L_total = L_ce - λ * H(p(Ŷ|X)) + β * I(X; Ŷ)
- Cross-Entropy: L_ce = -Σ y_true * log(softmax(logits))
- Entropy Regularization: -H(p(Ŷ|X)) = Σ p * log(p) (minimizing this maximizes entropy)
- Mutual Information: I(X; Ŷ) ≈ H(Ŷ) - H(Ŷ|X), where H(Ŷ) is the entropy of the batch-averaged prediction.

Usage:
    # Default usage with a model that outputs logits
    loss_fn = GoodhartAwareLoss(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    
    # Usage with a model that has a final softmax activation
    loss_fn = GoodhartAwareLoss(from_logits=False)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

References:
- Goodhart's Law: https://en.wikipedia.org/wiki/Goodhart's_law
- Information Bottleneck: Tishby, N., Pereira, F. C., & Bialek, W. (2000). The information bottleneck method.
- Regularizing by Penalizing Confident Outputs: Pereyra, G., Tucker, G., Chorowski, J., Kaiser, Ł., & Hinton, G. (2017). Regularizing neural networks by penalizing confident output distributions.
- Deep Variational Information Bottleneck: Alemi, A. A., Fischer, I., Dillon, J. V., & Murphy, K. (2017). Deep variational information bottleneck.
- On Calibration of Modern Neural Networks: Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks.
"""

import keras
import warnings
from keras import ops
from typing import Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GoodhartAwareLoss(keras.losses.Loss):
    """
    Information-theoretic loss function combining cross-entropy with regularization.

    This loss function combines standard cross-entropy with entropy and mutual
    information regularization to encourage robust, well-calibrated models.

    The total loss is:
    L_total = CrossEntropy(y, y_pred) - λ_entropy * H(p(y_pred)) + λ_mi * I(X; y_pred)

    Args:
        entropy_weight (float): Weight for the entropy regularization term.
            Controls how much the model is encouraged to maintain uncertainty in its
            per-sample predictions. Higher values combat overconfidence more strongly.
            Typical range: [0.001, 0.5]. Default: 0.1
        mi_weight (float): Weight for the mutual information regularization term.
            Controls the compression of input-output information flow. Higher values
            force the model to be more "forgetful" of spurious details.
            Typical range: [0.001, 0.1]. Default: 0.01
            **Note**: The MI approximation is batch-dependent. Its effect may change
            with batch size. You may need to scale this weight inversely with
            batch size if you change it significantly.
        from_logits (bool): Whether `y_pred` is a tensor of logits or probabilities.
            Set to `True` (default) if your model does not have a final softmax
            activation. Set to `False` if it does.
        epsilon (float): Small constant for numerical stability in log operations.
            Should be much smaller than 1/num_classes. Default: 1e-8
        name (str): String name for the loss function. Default: 'goodhart_aware_loss'
        reduction (str): Type of reduction to apply to loss. Default: 'sum_over_batch_size'

    Raises:
        ValueError: If any parameter is outside its valid range.

    Practical Considerations:
        - **Interaction**: The two regularizers have a complementary tension. Entropy
          regularization encourages high uncertainty in *every* prediction. MI
          regularization encourages low uncertainty in individual predictions
          (`H(Y|X)`) but high uncertainty in the *average* prediction (`H(Y)`).
          This balance helps the model become certain only when truly warranted.
        - **When to Use**: This loss is most effective in scenarios prone to
          overfitting on spurious features, such as datasets with known biases,
          noisy labels, or when out-of-distribution (OOD) robustness is a primary
          goal.
        - **Hyperparameter Tuning**: Start with the defaults. If the model remains
          overconfident (poorly calibrated), increase `entropy_weight`. If the model
          seems to be overfitting to dataset-specific quirks (poor generalization),
          increase `mi_weight`.
    """

    def __init__(
            self,
            entropy_weight: float = 0.1,
            mi_weight: float = 0.01,
            from_logits: bool = True,
            epsilon: float = 1e-8,
            name: str = 'goodhart_aware_loss',
            reduction: str = 'sum_over_batch_size'
    ) -> None:
        super().__init__(name=name, reduction=reduction)

        # Parameter validation
        if not isinstance(entropy_weight, (int, float)) or entropy_weight < 0:
            raise ValueError(f"Entropy weight must be a non-negative number, got {entropy_weight}.")
        if not isinstance(mi_weight, (int, float)) or mi_weight < 0:
            raise ValueError(f"MI weight must be a non-negative number, got {mi_weight}.")
        if not isinstance(epsilon, (int, float)) or epsilon <= 0 or epsilon >= 0.1:
            raise ValueError(f"Epsilon must be a small positive number (0, 0.1), got {epsilon}.")

        if entropy_weight > 1.0:
            warnings.warn(
                f"Very high entropy weight ({entropy_weight}) may dominate training. "
                f"Consider values in [0.001, 0.5]", UserWarning
            )

        self.entropy_weight = float(entropy_weight)
        self.mi_weight = float(mi_weight)
        self.from_logits = bool(from_logits)
        self.epsilon = float(epsilon)

    def call(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute the Goodhart-aware loss.

        Args:
            y_true (KerasTensor): Ground truth labels, shape (batch_size, num_classes).
            y_pred (KerasTensor): Predicted logits or probabilities, shape (batch_size, num_classes).

        Returns:
            KerasTensor: Scalar tensor representing the total loss for the batch.
        """
        y_true = ops.cast(y_true, dtype=y_pred.dtype)

        # Component 1: Standard cross-entropy loss
        # Handles both logits and probabilities based on the init flag.
        ce_loss = keras.losses.categorical_crossentropy(
            y_true,
            y_pred,
            from_logits=self.from_logits
        )

        # For regularization, we always need probabilities.
        if self.from_logits:
            probs = ops.softmax(y_pred, axis=-1)
        else:
            # If input is already probabilities, use it directly.
            # Clipping ensures we don't pass exact 0s or 1s to log, even if the user provides them.
            probs = ops.clip(y_pred, self.epsilon, 1.0 - self.epsilon)

        # Component 2: Entropy regularization loss (only if weight > 0)
        entropy_loss = self._entropy_regularization(probs) if self.entropy_weight > 0 else 0.0

        # Component 3: Mutual information regularization loss (only if weight > 0)
        mi_loss = self._mutual_information_regularization(probs) if self.mi_weight > 0 else 0.0

        # Combine all components.
        total_loss = (ce_loss +
                      self.entropy_weight * entropy_loss +
                      self.mi_weight * mi_loss)

        return total_loss

    def _entropy_regularization(
            self,
            probs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute entropy regularization loss from probabilities.
        The goal is to maximize the conditional entropy H(Ŷ|X) for each sample.
        We achieve this by minimizing its negative, -H(Ŷ|X).
        """
        probs = ops.clip(probs, self.epsilon, 1.0 - self.epsilon)
        conditional_entropy_per_sample = -ops.sum(probs * ops.log(probs), axis=-1)
        return -ops.mean(conditional_entropy_per_sample)

    def _mutual_information_regularization(
            self,
            probs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Approximate mutual information I(X;Ŷ) from probabilities.
        I(X;Ŷ) ≈ H(Ŷ) - H(Ŷ|X)
        """
        probs = ops.clip(probs, self.epsilon, 1.0 - self.epsilon)

        # H(Ŷ|X): Expected conditional entropy.
        h_y_given_x = -ops.mean(ops.sum(probs * ops.log(probs), axis=-1))

        # H(Ŷ): Marginal entropy of the batch-averaged prediction.
        mean_probs = ops.mean(probs, axis=0)
        mean_probs = ops.clip(mean_probs, self.epsilon, 1.0 - self.epsilon)
        h_y = -ops.sum(mean_probs * ops.log(mean_probs))

        # Approximate mutual information I(X;Ŷ)
        mutual_information = h_y - h_y_given_x

        # Clip at zero, as negative MI is not physically meaningful.
        return ops.maximum(0.0, mutual_information)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization."""
        config = super().get_config()
        config.update({
            'entropy_weight': self.entropy_weight,
            'mi_weight': self.mi_weight,
            'from_logits': self.from_logits,
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
        y_pred (KerasTensor): Predicted logits or probabilities.

    Returns:
        Dict[str, float]: A dictionary with individual loss components.
    """
    # Compute individual components
    ce_loss = keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=loss_fn.from_logits)

    if loss_fn.from_logits:
        probs = ops.softmax(y_pred, axis=-1)
    else:
        probs = y_pred

    entropy_loss = loss_fn._entropy_regularization(probs)
    mi_loss = loss_fn._mutual_information_regularization(probs)

    # Compute weighted contributions
    weighted_entropy = loss_fn.entropy_weight * entropy_loss
    weighted_mi = loss_fn.mi_weight * mi_loss
    total_loss = ops.mean(ce_loss + weighted_entropy + weighted_mi)

    # Convert tensors to Python floats for easy inspection
    results = {
        'total_loss': float(ops.convert_to_numpy(total_loss)),
        'cross_entropy': float(ops.convert_to_numpy(ops.mean(ce_loss))),
        'entropy_loss': float(ops.convert_to_numpy(entropy_loss)),
        'mi_loss': float(ops.convert_to_numpy(mi_loss)),
        'weighted_entropy': float(ops.convert_to_numpy(weighted_entropy)),
        'weighted_mi': float(ops.convert_to_numpy(weighted_mi)),
        'entropy_weight': loss_fn.entropy_weight,
        'mi_weight': loss_fn.mi_weight
    }
    total_loss_val = results['total_loss']
    if total_loss_val != 0:
        results.update({
            'ce_contribution_pct': (results['cross_entropy'] / total_loss_val) * 100,
            'entropy_contribution_pct': (results['weighted_entropy'] / total_loss_val) * 100,
            'mi_contribution_pct': (results['weighted_mi'] / total_loss_val) * 100
        })
    else:
        results.update({
            'ce_contribution_pct': 0,
            'entropy_contribution_pct': 0,
            'mi_contribution_pct': 0
        })
    return results

# ---------------------------------------------------------------------
