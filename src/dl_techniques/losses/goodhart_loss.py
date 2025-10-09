"""
An information-theoretic loss to promote robust generalization.

This loss function is designed to address the ML equivalent of Goodhart's Law,
where a model, in optimizing a simple metric like cross-entropy, learns to
exploit statistical shortcuts or "spurious correlations" in the training
data rather than learning the underlying causal features. The objective is to
create a more holistic training signal that encourages robustness and better
generalization by combining the standard task loss with information-theoretic
regularizers.

Conceptual Overview:
    The core idea is that a robust model should not only be accurate but also
    well-calibrated (not overconfident) and information-efficient (it should
    compress the input, retaining only the information essential for the task).
    By explicitly penalizing overconfidence and information redundancy, this
    loss function guides the model away from brittle, shortcut-based solutions
    towards more generalizable representations.

Architectural Design:
    The total loss is a weighted composite of three distinct components:
    1.  Categorical Cross-Entropy: The standard supervised loss that drives
        the model to make accurate predictions. This is the primary "task"
        component.
    2.  Entropy Regularization: This term penalizes overconfident predictions
        by maximizing the Shannon entropy of the model's output distribution for
        each sample. A higher entropy corresponds to a less confident, more
        uniform prediction. This discourages the model from collapsing its
        predictions based on flimsy evidence from a single spurious feature.
    3.  Mutual Information Regularization: Based on the Information Bottleneck
        principle, this term penalizes the mutual information between the raw
        input `X` and the model's prediction `Ŷ`. It encourages the model to
        learn a compressed internal representation, forcing it to "forget"
        information from the input that is not strictly necessary for the
        prediction task. This compression is hypothesized to discard noisy or
        spurious features, retaining only the robust, generalizable ones.

Mathematical Formulation:
    The total loss is a linear combination of the three components:

    L = L_CE - λ * H(p(Ŷ|X)) + β * I(X; Ŷ)

    Where:
    -   `L_CE` is the standard categorical cross-entropy loss.
    -   `H(p(Ŷ|X))` is the conditional entropy of the prediction `Ŷ` given the
        input `X`, averaged over the batch. The loss *maximizes* this entropy
        (by minimizing its negative) to penalize confidence. `λ` is its weight.
    -   `I(X; Ŷ)` is the mutual information between the input and the prediction.
        The loss *minimizes* this term to encourage compression. `β` is its
        weight. The mutual information is practically approximated over a
        batch using the identity `I(X; Ŷ) = H(Ŷ) - H(Ŷ|X)`, where `H(Ŷ)` is the
        entropy of the marginal prediction distribution (the average prediction
        across the batch).

References:
    -   Pereyra, G., et al. (2017). "Regularizing Neural Networks by Penalizing
        Confident Output Distributions." (For the entropy regularization term).
    -   Tishby, N., Pereira, F. C., & Bialek, W. (2000). "The Information
        Bottleneck Method." (For the mutual information term).
"""

import keras
import warnings
from keras import ops
from typing import Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GoodhartAwareLoss(keras.losses.Loss):
    """
    An information-theoretic loss combining cross-entropy with regularization.

    This loss function augments standard cross-entropy with entropy and mutual
    information regularization to encourage robust, well-calibrated models.

    The total loss is calculated as:
    ``L = CE(y, y_pred) - entropy_weight * H(y_pred) + mi_weight * I(X; y_pred)``
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        entropy_weight: float = 0.1,
        mi_weight: float = 0.01,
        from_logits: bool = True,
        epsilon: float = 1e-8,
        name: str = 'goodhart_aware_loss',
        reduction: str = 'sum_over_batch_size'
    ) -> None:
        """
        Initializes the GoodhartAwareLoss.

        :param label_smoothing: Factor for label smoothing, a regularization
            technique applied to the cross-entropy component. Must be in the
            range [0, 1). Defaults to 0.0 (no smoothing).
        :type label_smoothing: float
        :param entropy_weight: Weight (:math:`\\lambda`) for the entropy
            regularization term. Controls how much the model is encouraged to
            maintain uncertainty. Higher values combat overconfidence more
            strongly. Typical range: [0.001, 0.5]. Defaults to 0.1.
        :type entropy_weight: float
        :param mi_weight: Weight (:math:`\\beta`) for the mutual information
            regularization term. Controls the compression of input-output
            information flow. Higher values force the model to be more
            "forgetful" of spurious details. Typical range: [0.001, 0.1].
            Defaults to 0.01.
        :type mi_weight: float
        :param from_logits: Whether `y_pred` is a tensor of logits or
            probabilities. Set to ``True`` (default) if your model does not have
            a final softmax activation.
        :type from_logits: bool
        :param epsilon: A small constant for numerical stability in log
            operations. Should be much smaller than 1/num_classes.
            Defaults to 1e-8.
        :type epsilon: float
        :param name: String name for the loss function.
        :type name: str
        :param reduction: Type of reduction to apply to loss.
        :type reduction: str
        :raises ValueError: If any parameter is outside its valid range.
        """
        super().__init__(name=name, reduction=reduction)

        # --- Parameter Validation ---
        if not (0 <= label_smoothing < 1):
            raise ValueError(
                f"label_smoothing must be in the range [0, 1), "
                f"but got {label_smoothing}."
            )
        if not (isinstance(entropy_weight, (int, float)) and entropy_weight >= 0):
            raise ValueError(
                f"Entropy weight must be a non-negative number, "
                f"but got {entropy_weight}."
            )
        if not (isinstance(mi_weight, (int, float)) and mi_weight >= 0):
            raise ValueError(
                f"MI weight must be a non-negative number, "
                f"but got {mi_weight}."
            )
        if not (0 < epsilon < 0.1):
            raise ValueError(
                f"Epsilon must be a small positive number in (0, 0.1), "
                f"but got {epsilon}."
            )

        if entropy_weight > 1.0:
            warnings.warn(
                f"High entropy_weight ({entropy_weight}) may dominate training. "
                "Consider values in [0.001, 0.5].", UserWarning
            )
        if mi_weight > 0.1:
            warnings.warn(
                f"High mi_weight ({mi_weight}) may dominate training. "
                "Consider values in [0.001, 0.1].", UserWarning
            )

        self.label_smoothing = float(label_smoothing)
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
        Computes the Goodhart-aware loss for a batch.

        :param y_true: Ground truth labels, shape (batch_size, num_classes).
        :type y_true: keras.KerasTensor
        :param y_pred: Predicted logits or probabilities, shape
            (batch_size, num_classes).
        :type y_pred: keras.KerasTensor
        :return: A scalar tensor representing the total loss for the batch.
        :rtype: keras.KerasTensor
        """
        y_true = ops.cast(y_true, dtype=y_pred.dtype)

        # --- Component 1: Standard Cross-Entropy Loss ---
        # This component drives task accuracy and incorporates label smoothing.
        ce_loss = keras.losses.categorical_crossentropy(
            y_true=y_true,
            y_pred=y_pred,
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing
        )

        # For regularization, we always need probabilities.
        if self.from_logits:
            probs = ops.softmax(y_pred, axis=-1)
        else:
            # Clip user-provided probabilities to avoid log(0) issues.
            probs = ops.clip(y_pred, self.epsilon, 1.0 - self.epsilon)

        # --- Component 2: Entropy Regularization ---
        # This term is added to penalize overconfident predictions.
        entropy_term = 0.0
        if self.entropy_weight > 0:
            entropy_term = self.entropy_weight * self._entropy_regularization(probs)

        # --- Component 3: Mutual Information Regularization ---
        # This term is added to penalize reliance on spurious features.
        mi_term = 0.0
        if self.mi_weight > 0:
            mi_term = self.mi_weight * self._mutual_information_regularization(probs)

        # --- Combine all components ---
        total_loss = ce_loss + entropy_term + mi_term

        return total_loss

    def _entropy_regularization(
        self,
        probs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Calculates the negative conditional entropy term to be minimized.

        The goal is to maximize the conditional entropy :math:`H(\\hat{Y}|X)`.
        We achieve this by minimizing its negative, :math:`-H(\\hat{Y}|X)`.

        :param probs: The model's output probabilities.
        :type probs: keras.KerasTensor
        :return: The negative of the mean conditional entropy.
        :rtype: keras.KerasTensor
        """
        probs = ops.clip(probs, self.epsilon, 1.0 - self.epsilon)
        # H(Y|X) per sample = - sum(p * log(p))
        conditional_entropy_per_sample = -ops.sum(probs * ops.log(probs), axis=-1)
        # We want to minimize -mean(H(Y|X)) to maximize entropy.
        return -ops.mean(conditional_entropy_per_sample)

    def _mutual_information_regularization(
        self,
        probs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Approximates the mutual information :math:`I(X; \\hat{Y})` to be minimized.

        The approximation is :math:`I(X; \\hat{Y}) \\approx H(\\hat{Y}) - H(\\hat{Y}|X)`.

        :param probs: The model's output probabilities.
        :type probs: keras.KerasTensor
        :return: The approximated mutual information.
        :rtype: keras.KerasTensor
        """
        probs = ops.clip(probs, self.epsilon, 1.0 - self.epsilon)

        # H(Y|X): Mean conditional entropy across the batch.
        h_y_given_x = -ops.mean(ops.sum(probs * ops.log(probs), axis=-1))

        # H(Y): Entropy of the batch-averaged prediction distribution.
        mean_probs = ops.mean(probs, axis=0)
        mean_probs = ops.clip(mean_probs, self.epsilon, 1.0 - self.epsilon)
        h_y = -ops.sum(mean_probs * ops.log(mean_probs))

        # Approximate mutual information I(X;Y) = H(Y) - H(Y|X)
        mutual_information = h_y - h_y_given_x

        # MI cannot be negative, so clip at zero.
        return ops.maximum(0.0, mutual_information)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration dictionary for serialization.

        :return: A dictionary of the loss function's configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'label_smoothing': self.label_smoothing,
            'entropy_weight': self.entropy_weight,
            'mi_weight': self.mi_weight,
            'from_logits': self.from_logits,
            'epsilon': self.epsilon,
        })
        return config


def analyze_loss_components(
    loss_fn: GoodhartAwareLoss,
    y_true: keras.KerasTensor,
    y_pred: keras.KerasTensor
) -> Dict[str, float]:
    """
    Analyzes individual components of the GoodhartAwareLoss for debugging.

    :param loss_fn: An instance of the GoodhartAwareLoss function.
    :type loss_fn: GoodhartAwareLoss
    :param y_true: Ground truth labels.
    :type y_true: keras.KerasTensor
    :param y_pred: Predicted logits or probabilities.
    :type y_pred: keras.KerasTensor
    :return: A dictionary containing the unweighted and weighted values of each
        loss component, as well as their percentage contribution.
    :rtype: Dict[str, float]
    """
    y_true = ops.cast(y_true, dtype=y_pred.dtype)

    # Calculate individual components using the loss function's settings
    ce_loss = keras.losses.categorical_crossentropy(
        y_true=y_true,
        y_pred=y_pred,
        from_logits=loss_fn.from_logits,
        label_smoothing=loss_fn.label_smoothing
    )

    if loss_fn.from_logits:
        probs = ops.softmax(y_pred, axis=-1)
    else:
        probs = y_pred

    entropy_term_unweighted = loss_fn._entropy_regularization(probs)
    mi_term_unweighted = loss_fn._mutual_information_regularization(probs)

    # Compute weighted contributions
    entropy_term_weighted = loss_fn.entropy_weight * entropy_term_unweighted
    mi_term_weighted = loss_fn.mi_weight * mi_term_unweighted
    total_loss = ops.mean(ce_loss + entropy_term_weighted + mi_term_weighted)

    # Convert tensors to Python floats for easy inspection
    results = {
        'total_loss': float(ops.convert_to_numpy(total_loss)),
        'cross_entropy': float(ops.convert_to_numpy(ops.mean(ce_loss))),
        'entropy_term_unweighted': float(ops.convert_to_numpy(entropy_term_unweighted)),
        'mi_term_unweighted': float(ops.convert_to_numpy(mi_term_unweighted)),
        'entropy_term_weighted': float(ops.convert_to_numpy(entropy_term_weighted)),
        'mi_term_weighted': float(ops.convert_to_numpy(mi_term_weighted)),
        'label_smoothing': loss_fn.label_smoothing,
        'entropy_weight': loss_fn.entropy_weight,
        'mi_weight': loss_fn.mi_weight
    }

    # Calculate percentage contributions
    total_loss_val = results['total_loss']
    if abs(total_loss_val) > 1e-9:  # Avoid division by zero
        results.update({
            'ce_contrib_pct': (results['cross_entropy'] / total_loss_val) * 100,
            'entropy_contrib_pct': (results['entropy_term_weighted'] / total_loss_val) * 100,
            'mi_contrib_pct': (results['mi_term_weighted'] / total_loss_val) * 100
        })
    else:
        results.update({
            'ce_contrib_pct': 0.0,
            'entropy_contrib_pct': 0.0,
            'mi_contrib_pct': 0.0
        })
    return results

# ---------------------------------------------------------------------

