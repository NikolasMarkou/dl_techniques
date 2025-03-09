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
import tensorflow as tf
from typing import Dict, Optional, Tuple, Any


class ApproximationFunction(keras.layers.Layer):
    """Approximation function for transforming sigmoid outputs to near-binary values.

    This layer implements the approximation function A(p_i) from the AnyLoss paper:
    A(p_i) = 1 / (1 + e^(-L(p_i - 0.5)))

    Where L is an amplifying scale (recommended value is 73 based on paper analysis).

    Args:
        amplifying_scale: The L parameter that controls the steepness of the approximation.
            Default is 73.0 as recommended in the paper.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(self, amplifying_scale: float = 73.0, **kwargs: Any) -> None:
        """Initialize the approximation function layer.

        Args:
            amplifying_scale: The L parameter that controls how close the output values
                are to 0 or 1. Default is 73.0 as recommended in the paper.
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            ValueError: If amplifying_scale is not positive.
        """
        super().__init__(**kwargs)
        if amplifying_scale <= 0:
            raise ValueError(f"amplifying_scale must be positive, got {amplifying_scale}")
        self.amplifying_scale = amplifying_scale

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply the approximation function to transform probabilities to near-binary values.

        Args:
            inputs: Tensor of sigmoid outputs (class probabilities).

        Returns:
            Tensor of amplified values approximating binary labels.
        """
        return tf.math.reciprocal_no_nan(1.0 + tf.exp(-self.amplifying_scale * (inputs - 0.5)))

    def get_config(self) -> Dict[str, Any]:
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
        name: Optional name for the loss function.
        **kwargs: Additional keyword arguments passed to the parent class.

    Example:
        To implement a custom confusion matrix-based loss, subclass AnyLoss and
        override the call method:

        >>> class CustomLoss(AnyLoss):
        ...     def call(self, y_true, y_pred):
        ...         tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)
        ...         # Compute custom metric
        ...         metric = ...
        ...         # Return 1 - metric as the loss value
        ...         return 1.0 - metric
    """

    def __init__(
            self,
            amplifying_scale: float = 73.0,
            from_logits: bool = False,
            reduction: str = 'auto',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the AnyLoss base class.

        Args:
            amplifying_scale: The L parameter for the approximation function.
            from_logits: Whether model outputs raw logits without sigmoid.
            reduction: Type of reduction to apply to the loss.
            name: Optional name for the loss.
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            ValueError: If amplifying_scale is not positive.
        """
        if amplifying_scale <= 0:
            raise ValueError(f"amplifying_scale must be positive, got {amplifying_scale}")

        super().__init__(reduction=reduction, name=name, **kwargs)
        self.amplifying_scale = amplifying_scale
        self.from_logits = from_logits
        self.approximation = ApproximationFunction(amplifying_scale=amplifying_scale)

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
        # Handle edge case with empty input
        if tf.equal(tf.size(y_true), 0):
            epsilon = tf.keras.backend.epsilon()
            return (epsilon, epsilon, epsilon, epsilon)

        # Apply sigmoid if predictions are logits
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)

        # Apply approximation function
        y_approx = self.approximation(y_pred)

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
        """Abstract method that should be implemented by subclasses.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted probabilities or logits.

        Returns:
            Loss value.

        Raises:
            NotImplementedError: When this method is not overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration for serialization.

        Returns:
            Dictionary containing the loss configuration.
        """
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

    Example:
        >>> model = keras.Sequential([
        ...     keras.layers.Dense(64, activation="relu"),
        ...     keras.layers.Dense(1, activation="sigmoid")
        ... ])
        >>> model.compile(
        ...     optimizer=keras.optimizers.Adam(learning_rate=0.001),
        ...     loss=AccuracyLoss(),
        ...     metrics=["accuracy"]
        ... )
    """

    def __init__(
            self,
            amplifying_scale: float = 73.0,
            from_logits: bool = False,
            reduction: str = 'auto',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the AccuracyLoss.

        Args:
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
            name=name or "accuracy_loss",
            **kwargs
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the accuracy loss.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted probabilities or logits.

        Returns:
            Loss value (1 - accuracy).
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total
        return 1.0 - accuracy


class F1Loss(AnyLoss):
    """Loss function that optimizes F1 score.

    The F1 score is the harmonic mean of precision and recall:
    F1 = (2 * TP) / (2 * TP + FP + FN)
    This loss returns 1 - F1 to minimize during training.

    Example:
        >>> model = keras.Sequential([
        ...     keras.layers.Dense(64, activation="relu"),
        ...     keras.layers.Dense(1, activation="sigmoid")
        ... ])
        >>> model.compile(
        ...     optimizer=keras.optimizers.Adam(learning_rate=0.001),
        ...     loss=F1Loss(),
        ...     metrics=["accuracy"]
        ... )
    """

    def __init__(
            self,
            amplifying_scale: float = 73.0,
            from_logits: bool = False,
            reduction: str = 'auto',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the F1Loss.

        Args:
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
            name=name or "f1_loss",
            **kwargs
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the F1 loss.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted probabilities or logits.

        Returns:
            Loss value (1 - F1 score).
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)
        numerator = 2.0 * tp
        denominator = numerator + fp + fn
        f1_score = numerator / denominator
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
        name: Optional name for the loss function.
        **kwargs: Additional keyword arguments passed to the parent class.

    Example:
        >>> # F2 score (more weight to recall)
        >>> model = keras.Sequential([
        ...     keras.layers.Dense(64, activation="relu"),
        ...     keras.layers.Dense(1, activation="sigmoid")
        ... ])
        >>> model.compile(
        ...     optimizer=keras.optimizers.Adam(learning_rate=0.001),
        ...     loss=FBetaLoss(beta=2.0),
        ...     metrics=["accuracy"]
        ... )
    """

    def __init__(
            self,
            beta: float = 1.0,
            amplifying_scale: float = 73.0,
            from_logits: bool = False,
            reduction: str = 'auto',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the FBetaLoss.

        Args:
            beta: Weight parameter for recall vs precision.
            amplifying_scale: The L parameter for the approximation function.
            from_logits: Whether model outputs raw logits without sigmoid.
            reduction: Type of reduction to apply to the loss.
            name: Optional name for the loss.
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            ValueError: If beta is not positive.
        """
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")

        super().__init__(
            amplifying_scale=amplifying_scale,
            from_logits=from_logits,
            reduction=reduction,
            name=name or "fbeta_loss",
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
        beta_squared = tf.constant(self.beta_squared, dtype=tp.dtype)

        numerator = (1.0 + beta_squared) * tp
        denominator = numerator + beta_squared * fn + fp
        f_beta = numerator / denominator

        return 1.0 - f_beta

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration for serialization.

        Returns:
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({"beta": self.beta})
        return config


class GeometricMeanLoss(AnyLoss):
    """Loss function that optimizes the geometric mean of sensitivity and specificity.

    G-Mean = sqrt(sensitivity * specificity)
            = sqrt((TP / (TP + FN)) * (TN / (TN + FP)))
    This loss returns 1 - G-Mean to minimize during training.

    Example:
        >>> model = keras.Sequential([
        ...     keras.layers.Dense(64, activation="relu"),
        ...     keras.layers.Dense(1, activation="sigmoid")
        ... ])
        >>> model.compile(
        ...     optimizer=keras.optimizers.Adam(learning_rate=0.001),
        ...     loss=GeometricMeanLoss(),
        ...     metrics=["accuracy"]
        ... )
    """

    def __init__(
            self,
            amplifying_scale: float = 73.0,
            from_logits: bool = False,
            reduction: str = 'auto',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the GeometricMeanLoss.

        Args:
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
            name=name or "gmean_loss",
            **kwargs
        )

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

    This is particularly useful for imbalanced datasets where standard accuracy
    might be misleading.

    Example:
        >>> model = keras.Sequential([
        ...     keras.layers.Dense(64, activation="relu"),
        ...     keras.layers.Dense(1, activation="sigmoid")
        ... ])
        >>> model.compile(
        ...     optimizer=keras.optimizers.Adam(learning_rate=0.001),
        ...     loss=BalancedAccuracyLoss(),
        ...     metrics=["accuracy", "AUC"]
        ... )
    """

    def __init__(
            self,
            amplifying_scale: float = 73.0,
            from_logits: bool = False,
            reduction: str = 'auto',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the BalancedAccuracyLoss.

        Args:
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
            name=name or "balanced_accuracy_loss",
            **kwargs
        )

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


class WeightedCrossEntropyWithAnyLoss(AnyLoss):
    """Combines weighted binary cross-entropy with any AnyLoss-based metric.

    This loss function combines traditional binary cross-entropy with a metric-based
    loss from the AnyLoss framework, allowing for a balance between the two.

    Args:
        anyloss: An instance of an AnyLoss subclass.
        alpha: Weight for the AnyLoss component. The binary cross-entropy component
            has a weight of (1-alpha). Default is 0.5.
        amplifying_scale: The scale parameter for the approximation function.
            Default is 73.0 as recommended in the paper.
        from_logits: Whether the predictions are logits (not passed through
            a sigmoid). Default is False.
        reduction: Type of reduction to apply to the loss. Default is 'auto'.
        name: Optional name for the loss function.
        **kwargs: Additional keyword arguments passed to the parent class.

    Example:
        >>> # Combine F1Loss with binary cross-entropy
        >>> model = keras.Sequential([
        ...     keras.layers.Dense(64, activation="relu"),
        ...     keras.layers.Dense(1) # No activation for logits
        ... ])
        >>> combined_loss = WeightedCrossEntropyWithAnyLoss(
        ...     anyloss=F1Loss(),
        ...     alpha=0.7,
        ...     from_logits=True
        ... )
        >>> model.compile(
        ...     optimizer=keras.optimizers.Adam(learning_rate=0.001),
        ...     loss=combined_loss,
        ...     metrics=["accuracy"]
        ... )
    """

    def __init__(
            self,
            anyloss: AnyLoss,
            alpha: float = 0.5,
            amplifying_scale: float = 73.0,
            from_logits: bool = False,
            reduction: str = 'auto',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the WeightedCrossEntropyWithAnyLoss.

        Args:
            anyloss: An instance of an AnyLoss subclass.
            alpha: Weight for the AnyLoss component. The binary cross-entropy component
                has a weight of (1-alpha).
            amplifying_scale: The L parameter for the approximation function.
            from_logits: Whether model outputs raw logits without sigmoid.
            reduction: Type of reduction to apply to the loss.
            name: Optional name for the loss.
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            ValueError: If alpha is not in the range [0, 1].
            TypeError: If anyloss is not an instance of AnyLoss.
        """
        if not isinstance(anyloss, AnyLoss):
            raise TypeError(f"anyloss must be an instance of AnyLoss, got {type(anyloss)}")

        if alpha < 0 or alpha > 1:
            raise ValueError(f"alpha must be in the range [0, 1], got {alpha}")

        super().__init__(
            amplifying_scale=amplifying_scale,
            from_logits=from_logits,
            reduction=reduction,
            name=name or f"weighted_{anyloss.__class__.__name__.lower()}",
            **kwargs
        )
        self.anyloss = anyloss
        self.alpha = alpha
        self.bce = keras.losses.BinaryCrossentropy(from_logits=from_logits)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the combined loss.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted probabilities or logits.

        Returns:
            Weighted combination of binary cross-entropy and AnyLoss.
        """
        anyloss_value = self.anyloss(y_true, y_pred)
        bce_value = self.bce(y_true, y_pred)

        return self.alpha * anyloss_value + (1 - self.alpha) * bce_value

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration for serialization.

        Returns:
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({
            "anyloss": keras.saving.serialize_keras_object(self.anyloss),
            "alpha": self.alpha
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WeightedCrossEntropyWithAnyLoss":
        """Create an instance from config dictionary.

        Args:
            config: Dictionary containing the loss configuration.

        Returns:
            A new instance of WeightedCrossEntropyWithAnyLoss.
        """
        anyloss_config = config.pop("anyloss")
        anyloss = keras.saving.deserialize_keras_object(anyloss_config)
        return cls(anyloss=anyloss, **config)
