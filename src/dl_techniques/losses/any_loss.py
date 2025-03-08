"""
AnyLoss: Transforming Classification Metrics into Loss Functions

This module implements the AnyLoss framework from the paper:
"AnyLoss: Transforming Classification Metrics into Loss Functions" by Doheon Han,
Nuno Moniz, and Nitesh V Chawla (2024).

The AnyLoss framework provides a general-purpose method for converting any confusion
matrix-based evaluation metric into a differentiable loss function that can be used
directly for training neural networks.

Core Components:
---------------
1. ApproximationFunction - A layer that transforms sigmoid outputs (probabilities)
   into near-binary values, allowing the construction of a differentiable confusion matrix.
   The function is defined as: A(p_i) = 1 / (1 + e^(-L(p_i - 0.5)))
   where L is an amplifying scale (recommended value: 73).

2. AnyLoss - Base class for all confusion matrix-based losses, which handles computing
   the differentiable confusion matrix with entries:
   - True Positive (TP): sum(y_true * y_approx)
   - False Negative (FN): sum(y_true * (1 - y_approx))
   - False Positive (FP): sum((1 - y_true) * y_approx)
   - True Negative (TN): sum((1 - y_true) * (1 - y_approx))

3. Specific Loss Functions:
   - AccuracyLoss: Optimizes accuracy (TP + TN) / (TP + TN + FP + FN)
   - F1Loss: Optimizes F1 score (2*TP) / (2*TP + FP + FN)
   - FBetaLoss: Optimizes F-beta score with configurable beta parameter
   - GeometricMeanLoss: Optimizes G-Mean sqrt(sensitivity * specificity)
   - BalancedAccuracyLoss: Optimizes balanced accuracy (sensitivity + specificity) / 2

Key Benefits:
------------
1. Direct optimization of the evaluation metric of interest
2. Superior performance on imbalanced datasets
3. Universal applicability to any confusion matrix-based metric
4. Competitive learning speed compared to standard loss functions

Usage:
-----
Simply use any of the provided loss functions when compiling your Keras model:

```python
model = keras.Sequential([...])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=F1Loss(),  # Or any other AnyLoss implementation
    metrics=['accuracy']
)
```

For more details, refer to the original paper.
"""

import keras
import numpy as np
import tensorflow as tf
from typing import Callable, Dict, Optional, Tuple, Union


class ApproximationFunction(keras.layers.Layer):
    """Approximation function for transforming sigmoid outputs to near-binary values.

    This layer implements the approximation function A(p_i) from the AnyLoss paper:
    A(p_i) = 1 / (1 + e^(-L(p_i - 0.5)))

    Where L is an amplifying scale (recommended value is 73 based on paper analysis).

    Args:
        amplifying_scale: The L parameter that controls the steepness of the approximation.
            Default is 73.0 as recommended in the paper.
    """

    def __init__(self, amplifying_scale: float = 73.0, **kwargs):
        """Initialize the approximation function layer.

        Args:
            amplifying_scale: The L parameter that controls how close the output values
                are to 0 or 1. Default is 73.0 as recommended in the paper.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.amplifying_scale = amplifying_scale

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply the approximation function to transform probabilities to near-binary values.

        Args:
            inputs: Tensor of sigmoid outputs (class probabilities).

        Returns:
            Tensor of amplified values approximating binary labels.
        """
        return 1.0 / (1.0 + tf.exp(-self.amplifying_scale * (inputs - 0.5)))

    def get_config(self) -> Dict:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({"amplifying_scale": self.amplifying_scale})
        return config


class AnyLoss(keras.losses.Loss):
    """Base class for all confusion matrix-based losses in the AnyLoss framework.

    This abstract base class handles computing the differentiable confusion matrix
    and provides the infrastructure for computing specific metric-based losses.

    Args:
        amplifying_scale: The scale parameter for the approximation function.
            Default is 73.0 as recommended in the paper.
        from_logits: Whether the predictions are logits (not passed through
            a sigmoid). Default is False.
        reduction: Type of reduction to apply to the loss. Default is 'auto'.
    """

    def __init__(
            self,
            amplifying_scale: float = 73.0,
            from_logits: bool = False,
            reduction: str = 'auto',
            name: Optional[str] = None,
            **kwargs
    ):
        """Initialize the AnyLoss base class.

        Args:
            amplifying_scale: The L parameter for the approximation function.
            from_logits: Whether model outputs raw logits without sigmoid.
            reduction: Type of reduction to apply to the loss.
            name: Optional name for the loss.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.amplifying_scale = amplifying_scale
        self.from_logits = from_logits

    def compute_confusion_matrix(
            self, y_true: tf.Tensor, y_pred: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute differentiable confusion matrix entries.

        Args:
            y_true: Ground truth labels (0 or 1).
            y_pred: Predicted probabilities (from sigmoid) or logits.

        Returns:
            Tuple containing (TN, FN, FP, TP) confusion matrix entries.
        """
        # Apply sigmoid if predictions are logits
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)

        # Apply approximation function
        y_approx = 1.0 / (1.0 + tf.exp(-self.amplifying_scale * (y_pred - 0.5)))

        # Ensure y_true is of correct type
        y_true = tf.cast(y_true, dtype=y_pred.dtype)

        # Calculate confusion matrix entries
        true_positive = tf.reduce_sum(y_true * y_approx)
        false_negative = tf.reduce_sum(y_true * (1.0 - y_approx))
        false_positive = tf.reduce_sum((1.0 - y_true) * y_approx)
        true_negative = tf.reduce_sum((1.0 - y_true) * (1.0 - y_approx))

        # Add small epsilon to avoid division by zero
        epsilon = tf.keras.backend.epsilon()
        return (
            true_negative + epsilon,
            false_negative + epsilon,
            false_positive + epsilon,
            true_positive + epsilon
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Abstract method that should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_config(self) -> Dict:
        """Get loss configuration for serialization."""
        config = super().get_config()
        config.update({
            "amplifying_scale": self.amplifying_scale,
            "from_logits": self.from_logits
        })
        return config


class AccuracyLoss(AnyLoss):
    """Loss function that optimizes accuracy.

    The accuracy is calculated as (TP + TN) / (TP + TN + FP + FN).
    This loss returns 1 - accuracy to minimize during training.
    """

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the accuracy loss.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted probabilities or logits.

        Returns:
            Loss value (1 - accuracy).
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return 1.0 - accuracy


class F1Loss(AnyLoss):
    """Loss function that optimizes F1 score.

    The F1 score is the harmonic mean of precision and recall:
    F1 = (2 * TP) / (2 * TP + FP + FN)
    This loss returns 1 - F1 to minimize during training.
    """

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the F1 loss.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted probabilities or logits.

        Returns:
            Loss value (1 - F1 score).
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)
        f1_score = (2.0 * tp) / (2.0 * tp + fp + fn)
        return 1.0 - f1_score


class FBetaLoss(AnyLoss):
    """Loss function that optimizes F-beta score.

    The F-beta score is a weighted harmonic mean of precision and recall:
    F_beta = ((1 + beta^2) * TP) / ((1 + beta^2) * TP + beta^2 * FN + FP)
    This loss returns 1 - F_beta to minimize during training.

    Args:
        beta: The beta parameter that determines the weight of recall relative to precision.
            beta > 1 gives more weight to recall, beta < 1 gives more weight to precision.
            Default is 1.0 (F1 score).
        amplifying_scale: The scale parameter for the approximation function.
            Default is 73.0 as recommended in the paper.
        from_logits: Whether the predictions are logits (not passed through
            a sigmoid). Default is False.
        reduction: Type of reduction to apply to the loss. Default is 'auto'.
    """

    def __init__(
            self,
            beta: float = 1.0,
            amplifying_scale: float = 73.0,
            from_logits: bool = False,
            reduction: str = 'auto',
            name: Optional[str] = None,
            **kwargs
    ):
        """Initialize the FBetaLoss.

        Args:
            beta: Weight parameter for recall vs precision.
            amplifying_scale: The L parameter for the approximation function.
            from_logits: Whether model outputs raw logits without sigmoid.
            reduction: Type of reduction to apply to the loss.
            name: Optional name for the loss.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            amplifying_scale=amplifying_scale,
            from_logits=from_logits,
            reduction=reduction,
            name=name,
            **kwargs
        )
        self.beta = beta
        self.beta_squared = beta ** 2

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the F-beta loss.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted probabilities or logits.

        Returns:
            Loss value (1 - F_beta score).
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)
        numerator = (1.0 + self.beta_squared) * tp
        denominator = numerator + self.beta_squared * fn + fp
        f_beta = numerator / denominator
        return 1.0 - f_beta

    def get_config(self) -> Dict:
        """Get loss configuration for serialization."""
        config = super().get_config()
        config.update({"beta": self.beta})
        return config


class GeometricMeanLoss(AnyLoss):
    """Loss function that optimizes the geometric mean of sensitivity and specificity.

    G-Mean = sqrt(sensitivity * specificity)
            = sqrt((TP / (TP + FN)) * (TN / (TN + FP)))
    This loss returns 1 - G-Mean to minimize during training.
    """

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the geometric mean loss.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted probabilities or logits.

        Returns:
            Loss value (1 - G-Mean).
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        g_mean = tf.sqrt(sensitivity * specificity)
        return 1.0 - g_mean


class BalancedAccuracyLoss(AnyLoss):
    """Loss function that optimizes balanced accuracy.

    Balanced accuracy is the arithmetic mean of sensitivity and specificity:
    B-Acc = (sensitivity + specificity) / 2
          = ((TP / (TP + FN)) + (TN / (TN + FP))) / 2
    This loss returns 1 - B-Acc to minimize during training.
    """

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the balanced accuracy loss.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted probabilities or logits.

        Returns:
            Loss value (1 - balanced accuracy).
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        balanced_accuracy = (sensitivity + specificity) / 2.0
        return 1.0 - balanced_accuracy