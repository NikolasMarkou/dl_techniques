# filename: decoupled_information_loss.py

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

"""

import keras
from keras import ops
from typing import Dict, Any


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="custom_losses")
class DecoupledInformationLoss(keras.losses.Loss):
    """
    A decoupled info-theoretic loss combining cross-entropy with regularization.

    This loss function augments standard cross-entropy with two orthogonal
    information-theoretic regularizers: one for prediction uncertainty and one
    for prediction diversity.

    The total loss is calculated as:
    `L = CE(y, ŷ) - uncertainty_weight * H(ŷ|x) - diversity_weight * H(ŷ)`
    """

    def __init__(
            self,
            uncertainty_weight: float = 0.2,
            diversity_weight: float = 0.01,
            label_smoothing: float = 0.0,
            from_logits: bool = True,
            epsilon: float = 1e-8,
            name: str = 'decoupled_information_loss',
            reduction: str = 'sum_over_batch_size'
    ) -> None:
        """
        Initializes the DecoupledInformationLoss.

        :param uncertainty_weight: Weight (γ) for the conditional entropy term
            H(p(Y|X)). Controls how much the model is penalized for overconfident
            predictions. Higher values promote calibration. Suggested range: [0.1, 1.0].
        :type uncertainty_weight: float
        :param diversity_weight: Weight (δ) for the marginal entropy term H(p(Y)).
            Controls how much the model is encouraged to produce diverse predictions
            across a batch, preventing mode collapse. Suggested range: [0.001, 0.1].
        :type diversity_weight: float
        :param label_smoothing: Factor for label smoothing applied to the
            cross-entropy component. Must be in [0, 1). Defaults to 0.0.
        :type label_smoothing: float
        :param from_logits: Whether `y_pred` is a tensor of logits or
            probabilities. Defaults to ``True``.
        :type from_logits: bool
        :param epsilon: A small constant for numerical stability in log operations.
        :type epsilon: float
        :param name: String name for the loss function.
        :type name: str
        :param reduction: Type of reduction to apply to loss.
        :type reduction: str
        :raises ValueError: If any parameter is outside its valid range.
        """
        super().__init__(name=name, reduction=reduction)

        # --- Parameter Validation ---
        if not (uncertainty_weight >= 0 and diversity_weight >= 0):
            raise ValueError("Regularization weights must be non-negative.")
        if not (0 <= label_smoothing < 1):
            raise ValueError(f"label_smoothing must be in [0, 1), but got {label_smoothing}.")
        if not (0 < epsilon < 0.1):
            raise ValueError(f"epsilon must be a small positive number, but got {epsilon}.")

        self.uncertainty_weight = float(uncertainty_weight)
        self.diversity_weight = float(diversity_weight)
        self.label_smoothing = float(label_smoothing)
        self.from_logits = bool(from_logits)
        self.epsilon = float(epsilon)

    def call(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Computes the decoupled information-theoretic loss for a batch."""
        y_true = ops.cast(y_true, dtype=y_pred.dtype)

        # --- Component 1: Standard Cross-Entropy Loss ---
        ce_loss = keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits, label_smoothing=self.label_smoothing
        )

        # For regularization, we always need probabilities.
        if self.from_logits:
            probs = ops.softmax(y_pred, axis=-1)
        else:
            probs = ops.clip(y_pred, self.epsilon, 1.0 - self.epsilon)

        # --- Calculate Entropy Terms (to be maximized) ---
        # H(Y|X): Conditional Entropy - average entropy of each prediction
        conditional_entropy_per_sample = -ops.sum(probs * ops.log(probs + self.epsilon), axis=-1)
        h_conditional = ops.mean(conditional_entropy_per_sample)

        # H(Y): Marginal Entropy - entropy of the average prediction
        mean_probs = ops.mean(probs, axis=0)
        h_marginal = -ops.sum(mean_probs * ops.log(mean_probs + self.epsilon))

        # --- Combine all components ---
        # The objective is to MINIMIZE CE loss while MAXIMIZING the two entropy terms.
        # This is achieved by subtracting the weighted entropies from the CE loss.
        total_loss = ce_loss - (self.uncertainty_weight * h_conditional) - (self.diversity_weight * h_marginal)

        return total_loss

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration dictionary for serialization."""
        config = super().get_config()
        config.update({
            'uncertainty_weight': self.uncertainty_weight,
            'diversity_weight': self.diversity_weight,
            'label_smoothing': self.label_smoothing,
            'from_logits': self.from_logits,
            'epsilon': self.epsilon,
        })
        return config


def analyze_decoupled_information_loss(
        loss_fn: DecoupledInformationLoss,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor
) -> Dict[str, float]:
    """
    Analyzes individual components of the DecoupledInformationLoss for debugging.

    :param loss_fn: An instance of the DecoupledInformationLoss function.
    :type loss_fn: DecoupledInformationLoss
    :param y_true: Ground truth labels.
    :type y_true: keras.KerasTensor
    :param y_pred: Predicted logits or probabilities.
    :type y_pred: keras.KerasTensor
    :return: A dictionary containing the unweighted and weighted values of each
        loss component.
    :rtype: Dict[str, float]
    """
    y_true = ops.cast(y_true, dtype=y_pred.dtype)
    epsilon = loss_fn.epsilon

    # --- Component 1: Cross-Entropy ---
    ce_loss_val = keras.losses.categorical_crossentropy(
        y_true, y_pred, from_logits=loss_fn.from_logits, label_smoothing=loss_fn.label_smoothing
    )

    # --- Get Probabilities ---
    if loss_fn.from_logits:
        probs = ops.softmax(y_pred, axis=-1)
    else:
        probs = ops.clip(y_pred, epsilon, 1.0 - epsilon)

    # --- Component 2: Conditional Entropy ---
    h_conditional_unweighted = ops.mean(-ops.sum(probs * ops.log(probs + epsilon), axis=-1))
    uncertainty_term_weighted = -loss_fn.uncertainty_weight * h_conditional_unweighted

    # --- Component 3: Marginal Entropy ---
    mean_probs = ops.mean(probs, axis=0)
    h_marginal_unweighted = -ops.sum(mean_probs * ops.log(mean_probs + epsilon))
    diversity_term_weighted = -loss_fn.diversity_weight * h_marginal_unweighted

    # --- Total Loss ---
    total_loss = ops.mean(ce_loss_val + uncertainty_term_weighted + diversity_term_weighted)

    # Convert tensors to Python floats for easy inspection
    results = {
        'total_loss': float(ops.convert_to_numpy(total_loss)),
        'cross_entropy': float(ops.convert_to_numpy(ops.mean(ce_loss_val))),
        'h_conditional_unweighted': float(ops.convert_to_numpy(h_conditional_unweighted)),
        'h_marginal_unweighted': float(ops.convert_to_numpy(h_marginal_unweighted)),
        'uncertainty_term_weighted': float(ops.convert_to_numpy(uncertainty_term_weighted)),
        'diversity_term_weighted': float(ops.convert_to_numpy(diversity_term_weighted)),
        'uncertainty_weight': loss_fn.uncertainty_weight,
        'diversity_weight': loss_fn.diversity_weight,
    }
    return results