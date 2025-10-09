"""
Margin loss for training Capsule Networks.

This loss function is the objective proposed in the paper "Dynamic Routing
Between Capsules" for training Capsule Networks (CapsNets). Unlike
traditional cross-entropy loss which operates on probabilities, this loss is
designed to work directly with the lengths (L2 norm) of the output capsule
vectors, where a longer vector signifies a higher probability of an entity's
presence.

Conceptual Overview:
    The core idea is to enforce a margin-based separation between the capsule
    representing the correct class and the capsules for incorrect classes.
    For a given input, the model is trained to produce a long output vector
    (length > `m^+`) for the true class and short vectors (length < `m^-`)
    for all other classes. This encourages the network to be decisive and
    produces a more robust representation of classification confidence.

Architectural Design:
    The loss is computed as a sum over two distinct components for each
    class `k`:
    1.  A "positive loss" that penalizes the model only if the correct
        class's capsule vector length is smaller than a high margin `m^+`.
        This term is zero if the length is already greater than `m^+`.
    2.  A "negative loss" that penalizes the model only if an incorrect
        class's capsule vector length is larger than a low margin `m^-`.
        This term is zero if the length is already smaller than `m^-`.

    The negative loss component is typically down-weighted by a factor `λ`
    to prevent the numerous negative classes from overpowering the gradient
    signal from the single positive class, which helps to stabilize training.

Mathematical Formulation:
    For each output capsule `k`, with a vector `v_k`, the loss `L_k` is:

    L_k = T_k * max(0, m⁺ - ||v_k||)² + λ * (1 - T_k) * max(0, ||v_k|| - m⁻)²

    Where:
    -   `T_k` is the ground truth label (1 if class `k` is present, 0 otherwise).
    -   `||v_k||` is the L2 norm (length) of the capsule vector for class `k`.
    -   `m⁺` is the positive margin (e.g., 0.9), the desired lower bound for
        the correct capsule's length.
    -   `m⁻` is the negative margin (e.g., 0.1), the desired upper bound for
        incorrect capsules' lengths.
    -   `λ` is a down-weighting scalar (e.g., 0.5) for the negative loss.

    The total loss for a sample is the sum of `L_k` over all classes `k`.

References:
    - Sabour, S., Frosst, N., & Hinton, G. E. (2017). "Dynamic Routing
      Between Capsules." Advances in Neural Information Processing Systems 30.
      https://papers.nips.cc/paper/2017/file/2cad8fa43b24f4120ef25e47fa44b34c-Paper.pdf
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

def capsule_margin_loss(y_pred, y_true, downweight, positive_margin, negative_margin):
    # Ensure compatible dtypes
    y_true = ops.cast(y_true, dtype=y_pred.dtype)

    # Validate shapes
    if len(ops.shape(y_true)) != len(ops.shape(y_pred)):
        raise ValueError(
            f"y_true and y_pred must have the same number of dimensions. "
            f"Got {len(ops.shape(y_true))} and {len(ops.shape(y_pred))}."
        )

    # Calculate positive class loss: T_k * max(0, m^+ - ||v_k||)^2
    positive_loss = y_true * ops.square(
        ops.maximum(0.0, positive_margin - y_pred)
    )

    # Calculate negative class loss: λ * (1 - T_k) * max(0, ||v_k|| - m^-)^2
    negative_loss = downweight * (1.0 - y_true) * ops.square(
        ops.maximum(0.0, y_pred - negative_margin)
    )

    # Combine losses
    total_loss = positive_loss + negative_loss

    # Sum over classes for each sample, then reduce according to reduction strategy
    return ops.sum(total_loss, axis=-1)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CapsuleMarginLoss(keras.losses.Loss):
    """
    Margin loss function for Capsule Networks.

    This loss function is specifically designed for capsule networks where
    predictions represent capsule vector lengths. It uses separate margins
    for positive and negative classes to encourage long vectors for present
    entities and short vectors for absent entities.

    The loss combines two terms:
    1. Positive loss: Penalizes capsule lengths below the positive margin
    2. Negative loss: Penalizes capsule lengths above the negative margin
    """

    def __init__(
            self,
            positive_margin: float = 0.9,
            negative_margin: float = 0.1,
            downweight: float = 0.5,
            name: str = 'capsule_margin_loss',
            reduction: str = 'sum_over_batch_size'
    ) -> None:
        """
        Initializes the CapsuleMarginLoss.

        :param positive_margin: Margin for positive classes (m^+). Capsule lengths
            below this margin for present classes will be penalized. Should be
            greater than negative_margin. Typical range: [0.7, 0.95].
            Defaults to 0.9.
        :type positive_margin: float
        :param negative_margin: Margin for negative classes (m^-). Capsule lengths
            above this margin for absent classes will be penalized. Should be
            less than positive_margin. Typical range: [0.05, 0.2].
            Defaults to 0.1.
        :type negative_margin: float
        :param downweight: Downweighting factor (λ) for negative class terms.
            Controls the relative importance of negative vs positive losses.
            Lower values make the loss focus more on positive examples.
            Typical range: [0.1, 0.8]. Defaults to 0.5.
        :type downweight: float
        :param name: String name for the loss function.
        :type name: str
        :param reduction: Type of reduction to apply to loss.
        :type reduction: str
        :raises ValueError: If parameters are outside valid ranges or
            positive_margin <= negative_margin.
        """
        super().__init__(name=name, reduction=reduction)

        # Parameter validation
        if not (0 < positive_margin < 1):
            raise ValueError(
                f"positive_margin must be in range (0, 1), "
                f"but got {positive_margin}."
            )
        if not (0 < negative_margin < 1):
            raise ValueError(
                f"negative_margin must be in range (0, 1), "
                f"but got {negative_margin}."
            )
        if positive_margin <= negative_margin:
            raise ValueError(
                f"positive_margin ({positive_margin}) must be greater than "
                f"negative_margin ({negative_margin})."
            )
        if not (0 < downweight <= 1):
            raise ValueError(
                f"downweight must be in range (0, 1], "
                f"but got {downweight}."
            )

        # Warning for potentially problematic values
        if positive_margin - negative_margin < 0.3:
            warnings.warn(
                f"Small margin gap ({positive_margin - negative_margin:.3f}) "
                "may lead to unstable training. Consider increasing the gap.",
                UserWarning
            )
        if downweight < 0.1:
            warnings.warn(
                f"Very low downweight ({downweight}) may cause the model to "
                "ignore negative examples. Consider values >= 0.1.",
                UserWarning
            )

        self.positive_margin = float(positive_margin)
        self.negative_margin = float(negative_margin)
        self.downweight = float(downweight)

    def call(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Computes the margin loss for capsule networks.

        :param y_true: Ground truth labels in one-hot encoded format,
            shape (batch_size, num_classes). Each row should sum to 1.
        :type y_true: keras.KerasTensor
        :param y_pred: Predicted capsule lengths, shape (batch_size, num_classes).
            Values should typically be in range [0, 1] representing the
            magnitude/length of capsule vectors.
        :type y_pred: keras.KerasTensor
        :return: Scalar tensor representing the margin loss.
        :rtype: keras.KerasTensor
        :raises ValueError: If tensor shapes are incompatible.
        """
        return capsule_margin_loss(
            y_pred=y_pred,
            y_true=y_true,
            positive_margin=self.positive_margin,
            negative_margin=self.negative_margin,
            downweight=self.downweight)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration dictionary for serialization.

        :return: A dictionary containing the loss function's configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'positive_margin': self.positive_margin,
            'negative_margin': self.negative_margin,
            'downweight': self.downweight,
        })
        return config


def analyze_margin_loss_components(
        loss_fn: CapsuleMarginLoss,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor
) -> Dict[str, float]:
    """
    Analyzes individual components of the CapsuleMarginLoss for debugging.

    This function breaks down the margin loss into its positive and negative
    components, providing insights into how each part contributes to the
    total loss. This is useful for understanding model behavior and tuning
    hyperparameters.

    :param loss_fn: An instance of the CapsuleMarginLoss function.
    :type loss_fn: CapsuleMarginLoss
    :param y_true: Ground truth labels in one-hot format.
    :type y_true: keras.KerasTensor
    :param y_pred: Predicted capsule lengths.
    :type y_pred: keras.KerasTensor
    :return: A dictionary containing detailed analysis of loss components,
        including total loss, positive/negative contributions, and statistics
        about capsule length distributions.
    :rtype: Dict[str, float]
    """
    y_true = ops.cast(y_true, dtype=y_pred.dtype)

    # Calculate individual components
    positive_loss_per_class = y_true * ops.square(
        ops.maximum(0.0, loss_fn.positive_margin - y_pred)
    )
    negative_loss_per_class = loss_fn.downweight * (1.0 - y_true) * ops.square(
        ops.maximum(0.0, y_pred - loss_fn.negative_margin)
    )

    # Aggregate losses
    positive_loss_total = ops.sum(positive_loss_per_class)
    negative_loss_total = ops.sum(negative_loss_per_class)
    total_loss = positive_loss_total + negative_loss_total

    # Calculate statistics about capsule lengths
    present_classes_mask = ops.cast(y_true > 0.5, dtype=y_pred.dtype)
    absent_classes_mask = 1.0 - present_classes_mask

    # Average capsule lengths for present/absent classes
    present_lengths = ops.sum(y_pred * present_classes_mask) / (
            ops.sum(present_classes_mask) + 1e-8
    )
    absent_lengths = ops.sum(y_pred * absent_classes_mask) / (
            ops.sum(absent_classes_mask) + 1e-8
    )

    # Convert to Python floats for easy inspection
    results = {
        'total_loss': float(ops.convert_to_numpy(total_loss)),
        'positive_loss': float(ops.convert_to_numpy(positive_loss_total)),
        'negative_loss': float(ops.convert_to_numpy(negative_loss_total)),
        'avg_present_length': float(ops.convert_to_numpy(present_lengths)),
        'avg_absent_length': float(ops.convert_to_numpy(absent_lengths)),
        'positive_margin': loss_fn.positive_margin,
        'negative_margin': loss_fn.negative_margin,
        'downweight': loss_fn.downweight,
    }

    # Calculate percentage contributions
    total_loss_val = results['total_loss']
    if abs(total_loss_val) > 1e-9:
        results.update({
            'positive_contrib_pct': (results['positive_loss'] / total_loss_val) * 100,
            'negative_contrib_pct': (results['negative_loss'] / total_loss_val) * 100,
        })
    else:
        results.update({
            'positive_contrib_pct': 0.0,
            'negative_contrib_pct': 0.0,
        })

    # Add margin violations for debugging
    results.update({
        'margin_gap': loss_fn.positive_margin - loss_fn.negative_margin,
        'present_below_margin': float(ops.convert_to_numpy(
            ops.sum(ops.cast(
                (present_classes_mask > 0.5) & (y_pred < loss_fn.positive_margin),
                dtype='float32'
            ))
        )),
        'absent_above_margin': float(ops.convert_to_numpy(
            ops.sum(ops.cast(
                (absent_classes_mask > 0.5) & (y_pred > loss_fn.negative_margin),
                dtype='float32'
            ))
        )),
    })

    return results

# ---------------------------------------------------------------------
