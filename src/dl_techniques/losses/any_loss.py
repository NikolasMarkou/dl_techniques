"""
AnyLoss: Differentiable Confusion Matrix Metrics as Loss Functions
==================================================================

Implementation of the AnyLoss framework from:
"AnyLoss: Transforming Classification Metrics into Loss Functions"
by Doheon Han, Nuno Moniz, and Nitesh V Chawla (2024).

This module provides a unified approach to convert any confusion matrix-based
evaluation metric into a differentiable loss function for direct optimization
during neural network training.

Theory
------
The core insight is using an approximation function A(p) = 1 / (1 + e^(-L(p - 0.5)))
to transform soft predictions into near-binary values, enabling construction of a
differentiable confusion matrix. The recommended amplifying scale L=73 provides
optimal gradient flow while maintaining metric fidelity.

Available Losses
----------------
**Basic Metrics:**
    - ``AccuracyLoss``: (TP+TN) / (TP+TN+FP+FN)
    - ``PrecisionLoss``: TP / (TP+FP) - minimize false positives
    - ``RecallLoss``: TP / (TP+FN) - minimize false negatives
    - ``SpecificityLoss``: TN / (TN+FP) - true negative rate

**F-Score Family:**
    - ``F1Loss``: Harmonic mean of precision and recall
    - ``FBetaLoss``: Weighted F-score with configurable beta

**Balanced Metrics:**
    - ``BalancedAccuracyLoss``: (Sensitivity + Specificity) / 2
    - ``GeometricMeanLoss``: sqrt(Sensitivity * Specificity)
    - ``YoudenJLoss``: Sensitivity + Specificity - 1 (ROC optimization)

**Correlation Metrics:**
    - ``MCCLoss``: Matthews Correlation Coefficient - best single binary metric
    - ``CohenKappaLoss``: Agreement beyond chance

**Segmentation Metrics:**
    - ``IoULoss``: Intersection over Union (Jaccard Index)
    - ``DiceLoss``: Dice coefficient (equivalent to F1)
    - ``TverskyLoss``: Generalized Dice with alpha/beta weights
    - ``FocalTverskyLoss``: Focal variant for hard example mining

**Composite:**
    - ``WeightedCrossEntropyWithAnyLoss``: alpha * AnyLoss + (1-alpha) * BCE

Usage
-----
Direct instantiation::

    from dl_techniques.losses.anyloss import F1Loss, MCCLoss

    model.compile(optimizer='adam', loss=F1Loss())

Factory function::

    from dl_techniques.losses.anyloss import get_loss

    loss = get_loss('mcc', from_logits=True)
    loss = get_loss('tversky', alpha=0.3, beta=0.7)

Combined with BCE::

    from dl_techniques.losses.anyloss import WeightedCrossEntropyWithAnyLoss, F1Loss

    loss = WeightedCrossEntropyWithAnyLoss(anyloss=F1Loss(), alpha=0.7)

List available losses::

    from dl_techniques.losses.anyloss import print_available_losses
    print_available_losses()

Selection Guide
---------------
================= ============================================================
Use Case          Recommended Loss
================= ============================================================
Balanced data     ``AccuracyLoss``, ``F1Loss``
Imbalanced data   ``MCCLoss``, ``BalancedAccuracyLoss``, ``GeometricMeanLoss``
Costly FP         ``PrecisionLoss``, ``TverskyLoss(alpha>beta)``
Costly FN         ``RecallLoss``, ``TverskyLoss(alpha<beta)``
Segmentation      ``DiceLoss``, ``IoULoss``, ``FocalTverskyLoss``
ROC optimization  ``YoudenJLoss``
Inter-rater       ``CohenKappaLoss``
================= ============================================================

Notes
-----
- All losses return values in [0, 1] range (normalized where necessary)
- Supports both probability outputs and logits (``from_logits=True``)
- Epsilon is added to confusion matrix entries for numerical stability
- All classes are serializable via ``@keras.saving.register_keras_serializable()``

References
----------
.. [1] Han, D., Moniz, N., & Chawla, N. V. (2024). AnyLoss: Transforming
       Classification Metrics into Loss Functions.
"""

import keras
from keras import ops
from typing import Dict, Optional, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------
# Core Components
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ApproximationFunction(keras.layers.Layer):
    """Approximation function for transforming sigmoid outputs to near-binary values.

    This layer implements the approximation function A(p_i) from the AnyLoss paper:
    A(p_i) = 1 / (1 + e^(-L(p_i - 0.5)))

    Where L is an amplifying scale (recommended value is 73 based on paper analysis).

    Parameters
    ----------
    amplifying_scale : float, default=73.0
        The L parameter that controls the steepness of the approximation.
        Recommended value is 73.0 as per the paper.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Raises
    ------
    ValueError
        If amplifying_scale is not positive.

    Examples
    --------
    >>> approx_fn = ApproximationFunction(amplifying_scale=73.0)
    >>> probabilities = keras.random.uniform([10, 1])
    >>> near_binary = approx_fn(probabilities)
    """

    def __init__(self, amplifying_scale: float = 73.0, **kwargs: Any) -> None:
        """Initialize the approximation function layer.

        Parameters
        ----------
        amplifying_scale : float, default=73.0
            The L parameter that controls how close the output values
            are to 0 or 1. Default is 73.0 as recommended in the paper.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.

        Raises
        ------
        ValueError
            If amplifying_scale is not positive.
        """
        super().__init__(**kwargs)
        if amplifying_scale <= 0:
            raise ValueError(f"amplifying_scale must be positive, got {amplifying_scale}")

        self.amplifying_scale = amplifying_scale
        logger.debug(f"Initialized ApproximationFunction with amplifying_scale={amplifying_scale}")

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Apply the approximation function to transform probabilities to near-binary values.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Tensor of sigmoid outputs (class probabilities).
        training : bool, optional
            Whether the layer should behave in training mode or inference mode.

        Returns
        -------
        keras.KerasTensor
            Tensor of amplified values approximating binary labels.
        """
        shifted = ops.subtract(inputs, 0.5)
        scaled = ops.multiply(shifted, self.amplifying_scale)
        return ops.sigmoid(scaled)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.

        Returns
        -------
        tuple
            Output shape (same as input shape).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns
        -------
        dict
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({"amplifying_scale": self.amplifying_scale})
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class AnyLoss(keras.losses.Loss):
    """Base class for all confusion matrix-based losses in the AnyLoss framework.

    This abstract base class handles computing the differentiable confusion matrix
    and provides the infrastructure for computing specific metric-based losses.

    Parameters
    ----------
    amplifying_scale : float, default=73.0
        The scale parameter for the approximation function.
        Default is 73.0 as recommended in the paper.
    from_logits : bool, default=False
        Whether the predictions are logits (not passed through a sigmoid).
    reduction : str, default='sum_over_batch_size'
        Type of reduction to apply to the loss.
    name : str, optional
        Optional name for the loss function.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Raises
    ------
    ValueError
        If amplifying_scale is not positive.

    Examples
    --------
    To implement a custom confusion matrix-based loss, subclass AnyLoss and
    override the call method:

    >>> class CustomLoss(AnyLoss):
    ...     def call(self, y_true, y_pred):
    ...         tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)
    ...         metric = tp / (tp + fp)  # precision
    ...         return 1.0 - metric
    """

    def __init__(
            self,
            amplifying_scale: float = 73.0,
            from_logits: bool = False,
            reduction: str = 'sum_over_batch_size',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the AnyLoss base class.

        Parameters
        ----------
        amplifying_scale : float, default=73.0
            The L parameter for the approximation function.
        from_logits : bool, default=False
            Whether model outputs raw logits without sigmoid.
        reduction : str, default='sum_over_batch_size'
            Type of reduction to apply to the loss.
        name : str, optional
            Optional name for the loss.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.

        Raises
        ------
        ValueError
            If amplifying_scale is not positive.
        """
        if amplifying_scale <= 0:
            raise ValueError(f"amplifying_scale must be positive, got {amplifying_scale}")

        super().__init__(reduction=reduction, name=name, **kwargs)
        self.amplifying_scale = amplifying_scale
        self.from_logits = from_logits
        self.approximation = ApproximationFunction(amplifying_scale=amplifying_scale)

        logger.debug(
            f"Initialized {self.__class__.__name__} with amplifying_scale={amplifying_scale}, "
            f"from_logits={from_logits}"
        )

    def compute_confusion_matrix(
            self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Compute differentiable confusion matrix entries.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth labels (0 or 1).
        y_pred : keras.KerasTensor
            Predicted probabilities (from sigmoid) or logits.

        Returns
        -------
        tuple
            Tuple containing (TN, FN, FP, TP) confusion matrix entries,
            each with epsilon added for numerical stability.
        """
        epsilon = keras.backend.epsilon()

        if self.from_logits:
            y_pred_sigmoid = keras.activations.sigmoid(y_pred)
        else:
            y_pred_sigmoid = y_pred

        y_approx = self.approximation(y_pred_sigmoid)
        y_true_float = ops.cast(y_true, dtype=y_pred_sigmoid.dtype)

        true_positive = ops.sum(ops.multiply(y_true_float, y_approx))
        false_negative = ops.sum(ops.multiply(y_true_float, ops.subtract(1.0, y_approx)))
        false_positive = ops.sum(ops.multiply(ops.subtract(1.0, y_true_float), y_approx))
        true_negative = ops.sum(
            ops.multiply(ops.subtract(1.0, y_true_float), ops.subtract(1.0, y_approx))
        )

        return (
            ops.add(true_negative, epsilon),
            ops.add(false_negative, epsilon),
            ops.add(false_positive, epsilon),
            ops.add(true_positive, epsilon)
        )

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Abstract method that should be implemented by subclasses.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth binary labels.
        y_pred : keras.KerasTensor
            Predicted probabilities or logits.

        Returns
        -------
        keras.KerasTensor
            Loss value.

        Raises
        ------
        NotImplementedError
            When this method is not overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration for serialization.

        Returns
        -------
        dict
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({
            "amplifying_scale": self.amplifying_scale,
            "from_logits": self.from_logits
        })
        return config


# ---------------------------------------------------------------------
# Basic Metric Losses
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class AccuracyLoss(AnyLoss):
    """Loss function that optimizes accuracy.

    The accuracy is calculated as (TP + TN) / (TP + TN + FP + FN).
    This loss returns 1 - accuracy to minimize during training.

    Parameters
    ----------
    amplifying_scale : float, default=73.0
        The L parameter for the approximation function.
    from_logits : bool, default=False
        Whether model outputs raw logits without sigmoid.
    reduction : str, default='sum_over_batch_size'
        Type of reduction to apply to the loss.
    name : str, optional
        Optional name for the loss.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Examples
    --------
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
            reduction: str = 'sum_over_batch_size',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the AccuracyLoss.

        Parameters
        ----------
        amplifying_scale : float, default=73.0
            The L parameter for the approximation function.
        from_logits : bool, default=False
            Whether model outputs raw logits without sigmoid.
        reduction : str, default='sum_over_batch_size'
            Type of reduction to apply to the loss.
        name : str, optional
            Optional name for the loss.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            amplifying_scale=amplifying_scale,
            from_logits=from_logits,
            reduction=reduction,
            name=name or "accuracy_loss",
            **kwargs
        )

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the accuracy loss.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth binary labels.
        y_pred : keras.KerasTensor
            Predicted probabilities or logits.

        Returns
        -------
        keras.KerasTensor
            Loss value (1 - accuracy).
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)
        total = ops.add(ops.add(ops.add(tp, tn), fp), fn)
        accuracy = ops.divide(ops.add(tp, tn), total)
        return ops.subtract(1.0, accuracy)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PrecisionLoss(AnyLoss):
    """Loss function that optimizes precision.

    Precision = TP / (TP + FP)
    This loss returns 1 - precision to minimize during training.

    Use when false positives are costly (e.g., spam detection, fraud alerts).

    Parameters
    ----------
    amplifying_scale : float, default=73.0
        The L parameter for the approximation function.
    from_logits : bool, default=False
        Whether model outputs raw logits without sigmoid.
    reduction : str, default='sum_over_batch_size'
        Type of reduction to apply to the loss.
    name : str, optional
        Optional name for the loss.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Examples
    --------
    >>> model.compile(
    ...     optimizer=keras.optimizers.Adam(learning_rate=0.001),
    ...     loss=PrecisionLoss(),
    ...     metrics=["precision"]
    ... )
    """

    def __init__(
            self,
            amplifying_scale: float = 73.0,
            from_logits: bool = False,
            reduction: str = 'sum_over_batch_size',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the PrecisionLoss.

        Parameters
        ----------
        amplifying_scale : float, default=73.0
            The L parameter for the approximation function.
        from_logits : bool, default=False
            Whether model outputs raw logits without sigmoid.
        reduction : str, default='sum_over_batch_size'
            Type of reduction to apply to the loss.
        name : str, optional
            Optional name for the loss.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            amplifying_scale=amplifying_scale,
            from_logits=from_logits,
            reduction=reduction,
            name=name or "precision_loss",
            **kwargs
        )

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the precision loss.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth binary labels.
        y_pred : keras.KerasTensor
            Predicted probabilities or logits.

        Returns
        -------
        keras.KerasTensor
            Loss value (1 - precision).
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)
        precision = ops.divide(tp, ops.add(tp, fp))
        return ops.subtract(1.0, precision)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class RecallLoss(AnyLoss):
    """Loss function that optimizes recall (sensitivity).

    Recall = TP / (TP + FN)
    This loss returns 1 - recall to minimize during training.

    Use when false negatives are costly (e.g., disease detection, safety systems).

    Parameters
    ----------
    amplifying_scale : float, default=73.0
        The L parameter for the approximation function.
    from_logits : bool, default=False
        Whether model outputs raw logits without sigmoid.
    reduction : str, default='sum_over_batch_size'
        Type of reduction to apply to the loss.
    name : str, optional
        Optional name for the loss.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Examples
    --------
    >>> model.compile(
    ...     optimizer=keras.optimizers.Adam(learning_rate=0.001),
    ...     loss=RecallLoss(),
    ...     metrics=["recall"]
    ... )
    """

    def __init__(
            self,
            amplifying_scale: float = 73.0,
            from_logits: bool = False,
            reduction: str = 'sum_over_batch_size',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the RecallLoss.

        Parameters
        ----------
        amplifying_scale : float, default=73.0
            The L parameter for the approximation function.
        from_logits : bool, default=False
            Whether model outputs raw logits without sigmoid.
        reduction : str, default='sum_over_batch_size'
            Type of reduction to apply to the loss.
        name : str, optional
            Optional name for the loss.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            amplifying_scale=amplifying_scale,
            from_logits=from_logits,
            reduction=reduction,
            name=name or "recall_loss",
            **kwargs
        )

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the recall loss.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth binary labels.
        y_pred : keras.KerasTensor
            Predicted probabilities or logits.

        Returns
        -------
        keras.KerasTensor
            Loss value (1 - recall).
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)
        recall = ops.divide(tp, ops.add(tp, fn))
        return ops.subtract(1.0, recall)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SpecificityLoss(AnyLoss):
    """Loss function that optimizes specificity (true negative rate).

    Specificity = TN / (TN + FP)
    This loss returns 1 - specificity to minimize during training.

    Use when correctly identifying negatives is important.

    Parameters
    ----------
    amplifying_scale : float, default=73.0
        The L parameter for the approximation function.
    from_logits : bool, default=False
        Whether model outputs raw logits without sigmoid.
    reduction : str, default='sum_over_batch_size'
        Type of reduction to apply to the loss.
    name : str, optional
        Optional name for the loss.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Examples
    --------
    >>> model.compile(
    ...     optimizer=keras.optimizers.Adam(learning_rate=0.001),
    ...     loss=SpecificityLoss(),
    ...     metrics=["specificity"]
    ... )
    """

    def __init__(
            self,
            amplifying_scale: float = 73.0,
            from_logits: bool = False,
            reduction: str = 'sum_over_batch_size',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the SpecificityLoss.

        Parameters
        ----------
        amplifying_scale : float, default=73.0
            The L parameter for the approximation function.
        from_logits : bool, default=False
            Whether model outputs raw logits without sigmoid.
        reduction : str, default='sum_over_batch_size'
            Type of reduction to apply to the loss.
        name : str, optional
            Optional name for the loss.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            amplifying_scale=amplifying_scale,
            from_logits=from_logits,
            reduction=reduction,
            name=name or "specificity_loss",
            **kwargs
        )

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the specificity loss.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth binary labels.
        y_pred : keras.KerasTensor
            Predicted probabilities or logits.

        Returns
        -------
        keras.KerasTensor
            Loss value (1 - specificity).
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)
        specificity = ops.divide(tn, ops.add(tn, fp))
        return ops.subtract(1.0, specificity)


# ---------------------------------------------------------------------
# F-Score Family Losses
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class F1Loss(AnyLoss):
    """Loss function that optimizes F1 score.

    The F1 score is the harmonic mean of precision and recall:
    F1 = (2 * TP) / (2 * TP + FP + FN)
    This loss returns 1 - F1 to minimize during training.

    Parameters
    ----------
    amplifying_scale : float, default=73.0
        The L parameter for the approximation function.
    from_logits : bool, default=False
        Whether model outputs raw logits without sigmoid.
    reduction : str, default='sum_over_batch_size'
        Type of reduction to apply to the loss.
    name : str, optional
        Optional name for the loss.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Examples
    --------
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
            reduction: str = 'sum_over_batch_size',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the F1Loss.

        Parameters
        ----------
        amplifying_scale : float, default=73.0
            The L parameter for the approximation function.
        from_logits : bool, default=False
            Whether model outputs raw logits without sigmoid.
        reduction : str, default='sum_over_batch_size'
            Type of reduction to apply to the loss.
        name : str, optional
            Optional name for the loss.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            amplifying_scale=amplifying_scale,
            from_logits=from_logits,
            reduction=reduction,
            name=name or "f1_loss",
            **kwargs
        )

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the F1 loss.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth binary labels.
        y_pred : keras.KerasTensor
            Predicted probabilities or logits.

        Returns
        -------
        keras.KerasTensor
            Loss value (1 - F1 score).
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)
        numerator = ops.multiply(2.0, tp)
        denominator = ops.add(ops.add(numerator, fp), fn)
        f1_score = ops.divide(numerator, denominator)
        return ops.subtract(1.0, f1_score)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FBetaLoss(AnyLoss):
    """Loss function that optimizes F-beta score.

    The F-beta score is a weighted harmonic mean of precision and recall:
    F_beta = ((1 + beta^2) * TP) / ((1 + beta^2) * TP + beta^2 * FN + FP)
    This loss returns 1 - F_beta to minimize during training.

    Parameters
    ----------
    beta : float, default=1.0
        The beta parameter that determines the weight of recall relative to precision.
        beta > 1 gives more weight to recall, beta < 1 gives more weight to precision.
    amplifying_scale : float, default=73.0
        The scale parameter for the approximation function.
    from_logits : bool, default=False
        Whether the predictions are logits (not passed through a sigmoid).
    reduction : str, default='sum_over_batch_size'
        Type of reduction to apply to the loss.
    name : str, optional
        Optional name for the loss function.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Raises
    ------
    ValueError
        If beta is not positive.

    Examples
    --------
    >>> # F2 score (more weight to recall)
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
            reduction: str = 'sum_over_batch_size',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the FBetaLoss.

        Parameters
        ----------
        beta : float, default=1.0
            Weight parameter for recall vs precision.
        amplifying_scale : float, default=73.0
            The L parameter for the approximation function.
        from_logits : bool, default=False
            Whether model outputs raw logits without sigmoid.
        reduction : str, default='sum_over_batch_size'
            Type of reduction to apply to the loss.
        name : str, optional
            Optional name for the loss.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.

        Raises
        ------
        ValueError
            If beta is not positive.
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

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the F-beta loss.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth binary labels.
        y_pred : keras.KerasTensor
            Predicted probabilities or logits.

        Returns
        -------
        keras.KerasTensor
            Loss value (1 - F_beta score).
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)
        beta_squared = ops.convert_to_tensor(self.beta_squared, dtype=tp.dtype)

        numerator = ops.multiply(ops.add(1.0, beta_squared), tp)
        denominator = ops.add(ops.add(numerator, ops.multiply(beta_squared, fn)), fp)
        f_beta = ops.divide(numerator, denominator)

        return ops.subtract(1.0, f_beta)

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration for serialization.

        Returns
        -------
        dict
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({"beta": self.beta})
        return config


# ---------------------------------------------------------------------
# Balanced Metric Losses
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BalancedAccuracyLoss(AnyLoss):
    """Loss function that optimizes balanced accuracy.

    Balanced accuracy is the arithmetic mean of sensitivity and specificity:
    B-Acc = (sensitivity + specificity) / 2
          = ((TP / (TP + FN)) + (TN / (TN + FP))) / 2
    This loss returns 1 - B-Acc to minimize during training.

    This is particularly useful for imbalanced datasets where standard accuracy
    might be misleading.

    Parameters
    ----------
    amplifying_scale : float, default=73.0
        The L parameter for the approximation function.
    from_logits : bool, default=False
        Whether model outputs raw logits without sigmoid.
    reduction : str, default='sum_over_batch_size'
        Type of reduction to apply to the loss.
    name : str, optional
        Optional name for the loss.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Examples
    --------
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
            reduction: str = 'sum_over_batch_size',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the BalancedAccuracyLoss.

        Parameters
        ----------
        amplifying_scale : float, default=73.0
            The L parameter for the approximation function.
        from_logits : bool, default=False
            Whether model outputs raw logits without sigmoid.
        reduction : str, default='sum_over_batch_size'
            Type of reduction to apply to the loss.
        name : str, optional
            Optional name for the loss.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            amplifying_scale=amplifying_scale,
            from_logits=from_logits,
            reduction=reduction,
            name=name or "balanced_accuracy_loss",
            **kwargs
        )

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the balanced accuracy loss.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth binary labels.
        y_pred : keras.KerasTensor
            Predicted probabilities or logits.

        Returns
        -------
        keras.KerasTensor
            Loss value (1 - balanced accuracy).
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)

        sensitivity = ops.divide(tp, ops.add(tp, fn))
        specificity = ops.divide(tn, ops.add(tn, fp))

        balanced_accuracy = ops.divide(ops.add(sensitivity, specificity), 2.0)
        return ops.subtract(1.0, balanced_accuracy)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GeometricMeanLoss(AnyLoss):
    """Loss function that optimizes the geometric mean of sensitivity and specificity.

    G-Mean = sqrt(sensitivity * specificity)
           = sqrt((TP / (TP + FN)) * (TN / (TN + FP)))
    This loss returns 1 - G-Mean to minimize during training.

    Parameters
    ----------
    amplifying_scale : float, default=73.0
        The L parameter for the approximation function.
    from_logits : bool, default=False
        Whether model outputs raw logits without sigmoid.
    reduction : str, default='sum_over_batch_size'
        Type of reduction to apply to the loss.
    name : str, optional
        Optional name for the loss.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Examples
    --------
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
            reduction: str = 'sum_over_batch_size',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the GeometricMeanLoss.

        Parameters
        ----------
        amplifying_scale : float, default=73.0
            The L parameter for the approximation function.
        from_logits : bool, default=False
            Whether model outputs raw logits without sigmoid.
        reduction : str, default='sum_over_batch_size'
            Type of reduction to apply to the loss.
        name : str, optional
            Optional name for the loss.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            amplifying_scale=amplifying_scale,
            from_logits=from_logits,
            reduction=reduction,
            name=name or "gmean_loss",
            **kwargs
        )

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the geometric mean loss.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth binary labels.
        y_pred : keras.KerasTensor
            Predicted probabilities or logits.

        Returns
        -------
        keras.KerasTensor
            Loss value (1 - G-Mean).
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)

        sensitivity = ops.divide(tp, ops.add(tp, fn))
        specificity = ops.divide(tn, ops.add(tn, fp))

        g_mean = ops.sqrt(ops.multiply(sensitivity, specificity))
        return ops.subtract(1.0, g_mean)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class YoudenJLoss(AnyLoss):
    """Loss function that optimizes Youden's J statistic.

    J = Sensitivity + Specificity - 1 = TPR - FPR
    Range: [-1, 1] where 1 = perfect, 0 = useless classifier

    This metric maximizes the vertical distance from the ROC curve to the diagonal.
    The loss is normalized to [0, 1] range: Loss = 1 - (J + 1) / 2

    Parameters
    ----------
    amplifying_scale : float, default=73.0
        The L parameter for the approximation function.
    from_logits : bool, default=False
        Whether model outputs raw logits without sigmoid.
    reduction : str, default='sum_over_batch_size'
        Type of reduction to apply to the loss.
    name : str, optional
        Optional name for the loss.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Examples
    --------
    >>> model.compile(
    ...     optimizer=keras.optimizers.Adam(learning_rate=0.001),
    ...     loss=YoudenJLoss(),
    ...     metrics=["AUC"]
    ... )
    """

    def __init__(
            self,
            amplifying_scale: float = 73.0,
            from_logits: bool = False,
            reduction: str = 'sum_over_batch_size',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the YoudenJLoss.

        Parameters
        ----------
        amplifying_scale : float, default=73.0
            The L parameter for the approximation function.
        from_logits : bool, default=False
            Whether model outputs raw logits without sigmoid.
        reduction : str, default='sum_over_batch_size'
            Type of reduction to apply to the loss.
        name : str, optional
            Optional name for the loss.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            amplifying_scale=amplifying_scale,
            from_logits=from_logits,
            reduction=reduction,
            name=name or "youden_j_loss",
            **kwargs
        )

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the Youden's J loss.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth binary labels.
        y_pred : keras.KerasTensor
            Predicted probabilities or logits.

        Returns
        -------
        keras.KerasTensor
            Loss value (1 - normalized J statistic).
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)

        sensitivity = ops.divide(tp, ops.add(tp, fn))
        specificity = ops.divide(tn, ops.add(tn, fp))

        # J = sensitivity + specificity - 1, range [-1, 1]
        j = ops.subtract(ops.add(sensitivity, specificity), 1.0)

        # Normalize to [0, 1]: (J + 1) / 2
        j_normalized = ops.divide(ops.add(j, 1.0), 2.0)

        return ops.subtract(1.0, j_normalized)


# ---------------------------------------------------------------------
# Correlation Metric Losses
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MCCLoss(AnyLoss):
    """Loss function that optimizes Matthews Correlation Coefficient.

    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    Range: [-1, 1] where 1 = perfect, 0 = random, -1 = inverse

    The loss is normalized to [0, 1] range: Loss = 1 - (MCC + 1) / 2

    MCC is considered the best single metric for imbalanced binary classification
    as it takes into account all four confusion matrix entries.

    Parameters
    ----------
    amplifying_scale : float, default=73.0
        The L parameter for the approximation function.
    from_logits : bool, default=False
        Whether model outputs raw logits without sigmoid.
    reduction : str, default='sum_over_batch_size'
        Type of reduction to apply to the loss.
    name : str, optional
        Optional name for the loss.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Examples
    --------
    >>> model.compile(
    ...     optimizer=keras.optimizers.Adam(learning_rate=0.001),
    ...     loss=MCCLoss(),
    ...     metrics=["accuracy"]
    ... )
    """

    def __init__(
            self,
            amplifying_scale: float = 73.0,
            from_logits: bool = False,
            reduction: str = 'sum_over_batch_size',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the MCCLoss.

        Parameters
        ----------
        amplifying_scale : float, default=73.0
            The L parameter for the approximation function.
        from_logits : bool, default=False
            Whether model outputs raw logits without sigmoid.
        reduction : str, default='sum_over_batch_size'
            Type of reduction to apply to the loss.
        name : str, optional
            Optional name for the loss.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            amplifying_scale=amplifying_scale,
            from_logits=from_logits,
            reduction=reduction,
            name=name or "mcc_loss",
            **kwargs
        )

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the MCC loss.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth binary labels.
        y_pred : keras.KerasTensor
            Predicted probabilities or logits.

        Returns
        -------
        keras.KerasTensor
            Loss value (1 - normalized MCC).
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)

        # MCC numerator: TP*TN - FP*FN
        numerator = ops.subtract(ops.multiply(tp, tn), ops.multiply(fp, fn))

        # MCC denominator: sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        tp_fp = ops.add(tp, fp)
        tp_fn = ops.add(tp, fn)
        tn_fp = ops.add(tn, fp)
        tn_fn = ops.add(tn, fn)

        denominator = ops.sqrt(
            ops.multiply(ops.multiply(tp_fp, tp_fn), ops.multiply(tn_fp, tn_fn))
        )

        mcc = ops.divide(numerator, denominator)

        # Normalize MCC from [-1, 1] to [0, 1]: (MCC + 1) / 2
        mcc_normalized = ops.divide(ops.add(mcc, 1.0), 2.0)

        return ops.subtract(1.0, mcc_normalized)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class CohenKappaLoss(AnyLoss):
    """Loss function that optimizes Cohen's Kappa statistic.

    Kappa = (p_o - p_e) / (1 - p_e)
    Where:
    - p_o = observed agreement = (TP + TN) / N
    - p_e = expected agreement by chance

    Range: [-1, 1] where 1 = perfect agreement, 0 = chance agreement

    Cohen's Kappa accounts for agreement occurring by chance.
    The loss is normalized to [0, 1] range.

    Parameters
    ----------
    amplifying_scale : float, default=73.0
        The L parameter for the approximation function.
    from_logits : bool, default=False
        Whether model outputs raw logits without sigmoid.
    reduction : str, default='sum_over_batch_size'
        Type of reduction to apply to the loss.
    name : str, optional
        Optional name for the loss.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Examples
    --------
    >>> model.compile(
    ...     optimizer=keras.optimizers.Adam(learning_rate=0.001),
    ...     loss=CohenKappaLoss(),
    ...     metrics=["accuracy"]
    ... )
    """

    def __init__(
            self,
            amplifying_scale: float = 73.0,
            from_logits: bool = False,
            reduction: str = 'sum_over_batch_size',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the CohenKappaLoss.

        Parameters
        ----------
        amplifying_scale : float, default=73.0
            The L parameter for the approximation function.
        from_logits : bool, default=False
            Whether model outputs raw logits without sigmoid.
        reduction : str, default='sum_over_batch_size'
            Type of reduction to apply to the loss.
        name : str, optional
            Optional name for the loss.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            amplifying_scale=amplifying_scale,
            from_logits=from_logits,
            reduction=reduction,
            name=name or "kappa_loss",
            **kwargs
        )

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the Cohen's Kappa loss.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth binary labels.
        y_pred : keras.KerasTensor
            Predicted probabilities or logits.

        Returns
        -------
        keras.KerasTensor
            Loss value (1 - normalized Kappa).
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)

        n = ops.add(ops.add(ops.add(tp, tn), fp), fn)

        # Observed agreement
        p_o = ops.divide(ops.add(tp, tn), n)

        # Expected agreement
        # p_e = ((TP+FP)/N * (TP+FN)/N) + ((TN+FN)/N * (TN+FP)/N)
        pred_pos = ops.divide(ops.add(tp, fp), n)
        pred_neg = ops.divide(ops.add(tn, fn), n)
        actual_pos = ops.divide(ops.add(tp, fn), n)
        actual_neg = ops.divide(ops.add(tn, fp), n)

        p_e = ops.add(
            ops.multiply(pred_pos, actual_pos),
            ops.multiply(pred_neg, actual_neg)
        )

        # Kappa = (p_o - p_e) / (1 - p_e)
        kappa = ops.divide(ops.subtract(p_o, p_e), ops.subtract(1.0, p_e))

        # Normalize from [-1, 1] to [0, 1]
        kappa_normalized = ops.divide(ops.add(kappa, 1.0), 2.0)

        return ops.subtract(1.0, kappa_normalized)


# ---------------------------------------------------------------------
# Segmentation Metric Losses
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class IoULoss(AnyLoss):
    """Loss function that optimizes IoU (Intersection over Union / Jaccard Index).

    IoU = TP / (TP + FP + FN)

    Common in segmentation tasks; equivalent to Tversky with alpha=beta=1.

    Parameters
    ----------
    amplifying_scale : float, default=73.0
        The L parameter for the approximation function.
    from_logits : bool, default=False
        Whether model outputs raw logits without sigmoid.
    reduction : str, default='sum_over_batch_size'
        Type of reduction to apply to the loss.
    name : str, optional
        Optional name for the loss.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Examples
    --------
    >>> model.compile(
    ...     optimizer=keras.optimizers.Adam(learning_rate=0.001),
    ...     loss=IoULoss(),
    ...     metrics=["accuracy"]
    ... )
    """

    def __init__(
            self,
            amplifying_scale: float = 73.0,
            from_logits: bool = False,
            reduction: str = 'sum_over_batch_size',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the IoULoss.

        Parameters
        ----------
        amplifying_scale : float, default=73.0
            The L parameter for the approximation function.
        from_logits : bool, default=False
            Whether model outputs raw logits without sigmoid.
        reduction : str, default='sum_over_batch_size'
            Type of reduction to apply to the loss.
        name : str, optional
            Optional name for the loss.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            amplifying_scale=amplifying_scale,
            from_logits=from_logits,
            reduction=reduction,
            name=name or "iou_loss",
            **kwargs
        )

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the IoU loss.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth binary labels.
        y_pred : keras.KerasTensor
            Predicted probabilities or logits.

        Returns
        -------
        keras.KerasTensor
            Loss value (1 - IoU).
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)

        iou = ops.divide(tp, ops.add(ops.add(tp, fp), fn))
        return ops.subtract(1.0, iou)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DiceLoss(AnyLoss):
    """Loss function that optimizes Dice coefficient.

    Dice = 2*TP / (2*TP + FP + FN)

    Equivalent to F1 score; common in medical image segmentation.

    Parameters
    ----------
    amplifying_scale : float, default=73.0
        The L parameter for the approximation function.
    from_logits : bool, default=False
        Whether model outputs raw logits without sigmoid.
    reduction : str, default='sum_over_batch_size'
        Type of reduction to apply to the loss.
    name : str, optional
        Optional name for the loss.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Examples
    --------
    >>> model.compile(
    ...     optimizer=keras.optimizers.Adam(learning_rate=0.001),
    ...     loss=DiceLoss(),
    ...     metrics=["accuracy"]
    ... )
    """

    def __init__(
            self,
            amplifying_scale: float = 73.0,
            from_logits: bool = False,
            reduction: str = 'sum_over_batch_size',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the DiceLoss.

        Parameters
        ----------
        amplifying_scale : float, default=73.0
            The L parameter for the approximation function.
        from_logits : bool, default=False
            Whether model outputs raw logits without sigmoid.
        reduction : str, default='sum_over_batch_size'
            Type of reduction to apply to the loss.
        name : str, optional
            Optional name for the loss.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            amplifying_scale=amplifying_scale,
            from_logits=from_logits,
            reduction=reduction,
            name=name or "dice_loss",
            **kwargs
        )

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the Dice loss.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth binary labels.
        y_pred : keras.KerasTensor
            Predicted probabilities or logits.

        Returns
        -------
        keras.KerasTensor
            Loss value (1 - Dice coefficient).
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)

        numerator = ops.multiply(2.0, tp)
        denominator = ops.add(ops.add(numerator, fp), fn)
        dice = ops.divide(numerator, denominator)

        return ops.subtract(1.0, dice)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TverskyLoss(AnyLoss):
    """Loss function that optimizes Tversky Index.

    Tversky = TP / (TP + alpha*FP + beta*FN)

    Generalizes Dice/F1 (alpha=beta=0.5) and Jaccard/IoU (alpha=beta=1).

    Parameters
    ----------
    alpha : float, default=0.5
        Weight for false positives.
    beta : float, default=0.5
        Weight for false negatives.
    amplifying_scale : float, default=73.0
        The L parameter for the approximation function.
    from_logits : bool, default=False
        Whether model outputs raw logits without sigmoid.
    reduction : str, default='sum_over_batch_size'
        Type of reduction to apply to the loss.
    name : str, optional
        Optional name for the loss.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Raises
    ------
    ValueError
        If alpha or beta is negative.

    Notes
    -----
    Use alpha < beta to penalize FN more (recall-focused).
    Use alpha > beta to penalize FP more (precision-focused).

    Examples
    --------
    >>> # Recall-focused (penalize FN more)
    >>> model.compile(
    ...     optimizer=keras.optimizers.Adam(learning_rate=0.001),
    ...     loss=TverskyLoss(alpha=0.3, beta=0.7),
    ...     metrics=["accuracy"]
    ... )
    """

    def __init__(
            self,
            alpha: float = 0.5,
            beta: float = 0.5,
            amplifying_scale: float = 73.0,
            from_logits: bool = False,
            reduction: str = 'sum_over_batch_size',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the TverskyLoss.

        Parameters
        ----------
        alpha : float, default=0.5
            Weight for false positives.
        beta : float, default=0.5
            Weight for false negatives.
        amplifying_scale : float, default=73.0
            The L parameter for the approximation function.
        from_logits : bool, default=False
            Whether model outputs raw logits without sigmoid.
        reduction : str, default='sum_over_batch_size'
            Type of reduction to apply to the loss.
        name : str, optional
            Optional name for the loss.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.

        Raises
        ------
        ValueError
            If alpha or beta is negative.
        """
        if alpha < 0 or beta < 0:
            raise ValueError(f"alpha and beta must be non-negative, got alpha={alpha}, beta={beta}")

        super().__init__(
            amplifying_scale=amplifying_scale,
            from_logits=from_logits,
            reduction=reduction,
            name=name or "tversky_loss",
            **kwargs
        )
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the Tversky loss.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth binary labels.
        y_pred : keras.KerasTensor
            Predicted probabilities or logits.

        Returns
        -------
        keras.KerasTensor
            Loss value (1 - Tversky index).
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)

        alpha = ops.convert_to_tensor(self.alpha, dtype=tp.dtype)
        beta = ops.convert_to_tensor(self.beta, dtype=tp.dtype)

        denominator = ops.add(
            ops.add(tp, ops.multiply(alpha, fp)),
            ops.multiply(beta, fn)
        )
        tversky = ops.divide(tp, denominator)

        return ops.subtract(1.0, tversky)

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration for serialization.

        Returns
        -------
        dict
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({"alpha": self.alpha, "beta": self.beta})
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FocalTverskyLoss(AnyLoss):
    """Focal Tversky Loss for hard example mining.

    Loss = (1 - Tversky)^gamma

    Adds focal parameter to focus on hard examples.

    Parameters
    ----------
    alpha : float, default=0.7
        Weight for false positives.
    beta : float, default=0.3
        Weight for false negatives.
    gamma : float, default=0.75
        Focal parameter. Higher values focus more on hard examples.
    amplifying_scale : float, default=73.0
        The L parameter for the approximation function.
    from_logits : bool, default=False
        Whether model outputs raw logits without sigmoid.
    reduction : str, default='sum_over_batch_size'
        Type of reduction to apply to the loss.
    name : str, optional
        Optional name for the loss.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Raises
    ------
    ValueError
        If alpha or beta is negative, or gamma is not positive.

    Examples
    --------
    >>> model.compile(
    ...     optimizer=keras.optimizers.Adam(learning_rate=0.001),
    ...     loss=FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75),
    ...     metrics=["accuracy"]
    ... )
    """

    def __init__(
            self,
            alpha: float = 0.7,
            beta: float = 0.3,
            gamma: float = 0.75,
            amplifying_scale: float = 73.0,
            from_logits: bool = False,
            reduction: str = 'sum_over_batch_size',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the FocalTverskyLoss.

        Parameters
        ----------
        alpha : float, default=0.7
            Weight for false positives.
        beta : float, default=0.3
            Weight for false negatives.
        gamma : float, default=0.75
            Focal parameter. Higher values focus more on hard examples.
        amplifying_scale : float, default=73.0
            The L parameter for the approximation function.
        from_logits : bool, default=False
            Whether model outputs raw logits without sigmoid.
        reduction : str, default='sum_over_batch_size'
            Type of reduction to apply to the loss.
        name : str, optional
            Optional name for the loss.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.

        Raises
        ------
        ValueError
            If alpha or beta is negative, or gamma is not positive.
        """
        if alpha < 0 or beta < 0:
            raise ValueError(f"alpha and beta must be non-negative, got alpha={alpha}, beta={beta}")
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")

        super().__init__(
            amplifying_scale=amplifying_scale,
            from_logits=from_logits,
            reduction=reduction,
            name=name or "focal_tversky_loss",
            **kwargs
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the Focal Tversky loss.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth binary labels.
        y_pred : keras.KerasTensor
            Predicted probabilities or logits.

        Returns
        -------
        keras.KerasTensor
            Focal Tversky loss value.
        """
        tn, fn, fp, tp = self.compute_confusion_matrix(y_true, y_pred)

        alpha = ops.convert_to_tensor(self.alpha, dtype=tp.dtype)
        beta = ops.convert_to_tensor(self.beta, dtype=tp.dtype)
        gamma = ops.convert_to_tensor(self.gamma, dtype=tp.dtype)

        denominator = ops.add(
            ops.add(tp, ops.multiply(alpha, fp)),
            ops.multiply(beta, fn)
        )
        tversky = ops.divide(tp, denominator)
        tversky_loss = ops.subtract(1.0, tversky)

        return ops.power(tversky_loss, gamma)

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration for serialization.

        Returns
        -------
        dict
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma
        })
        return config


# ---------------------------------------------------------------------
# Composite Losses
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class WeightedCrossEntropyWithAnyLoss(AnyLoss):
    """Combines weighted binary cross-entropy with any AnyLoss-based metric.

    This loss function combines traditional binary cross-entropy with a metric-based
    loss from the AnyLoss framework, allowing for a balance between the two.

    Total Loss = alpha * AnyLoss + (1 - alpha) * BCE

    Parameters
    ----------
    anyloss : AnyLoss
        An instance of an AnyLoss subclass.
    alpha : float, default=0.5
        Weight for the AnyLoss component. The binary cross-entropy component
        has a weight of (1-alpha).
    amplifying_scale : float, default=73.0
        The scale parameter for the approximation function.
    from_logits : bool, default=False
        Whether the predictions are logits (not passed through a sigmoid).
    reduction : str, default='sum_over_batch_size'
        Type of reduction to apply to the loss.
    name : str, optional
        Optional name for the loss function.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Raises
    ------
    ValueError
        If alpha is not in the range [0, 1].
    TypeError
        If anyloss is not an instance of AnyLoss.

    Examples
    --------
    >>> # Combine F1Loss with binary cross-entropy
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
            reduction: str = 'sum_over_batch_size',
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the WeightedCrossEntropyWithAnyLoss.

        Parameters
        ----------
        anyloss : AnyLoss
            An instance of an AnyLoss subclass.
        alpha : float, default=0.5
            Weight for the AnyLoss component. The binary cross-entropy component
            has a weight of (1-alpha).
        amplifying_scale : float, default=73.0
            The L parameter for the approximation function.
        from_logits : bool, default=False
            Whether model outputs raw logits without sigmoid.
        reduction : str, default='sum_over_batch_size'
            Type of reduction to apply to the loss.
        name : str, optional
            Optional name for the loss.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.

        Raises
        ------
        ValueError
            If alpha is not in the range [0, 1].
        TypeError
            If anyloss is not an instance of AnyLoss.
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

        logger.info(
            f"Created WeightedCrossEntropyWithAnyLoss with alpha={alpha}, "
            f"combining {anyloss.__class__.__name__} and BinaryCrossentropy"
        )

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the combined loss.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth binary labels.
        y_pred : keras.KerasTensor
            Predicted probabilities or logits.

        Returns
        -------
        keras.KerasTensor
            Weighted combination of binary cross-entropy and AnyLoss.
        """
        anyloss_value = self.anyloss(y_true, y_pred)
        bce_value = self.bce(y_true, y_pred)

        alpha_tensor = ops.convert_to_tensor(self.alpha, dtype=anyloss_value.dtype)
        one_minus_alpha = ops.subtract(1.0, alpha_tensor)

        weighted_anyloss = ops.multiply(alpha_tensor, anyloss_value)
        weighted_bce = ops.multiply(one_minus_alpha, bce_value)

        return ops.add(weighted_anyloss, weighted_bce)

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration for serialization.

        Returns
        -------
        dict
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

        Parameters
        ----------
        config : dict
            Dictionary containing the loss configuration.

        Returns
        -------
        WeightedCrossEntropyWithAnyLoss
            A new instance of WeightedCrossEntropyWithAnyLoss.
        """
        anyloss_config = config.pop("anyloss")
        anyloss = keras.saving.deserialize_keras_object(anyloss_config)
        return cls(anyloss=anyloss, **config)


# ---------------------------------------------------------------------
# Summary and Utilities
# ---------------------------------------------------------------------


ANYLOSS_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Basic Metrics
    "accuracy": {
        "class": AccuracyLoss,
        "formula": "(TP+TN)/(TP+TN+FP+FN)",
        "use_case": "Balanced datasets",
        "category": "basic"
    },
    "precision": {
        "class": PrecisionLoss,
        "formula": "TP/(TP+FP)",
        "use_case": "Minimize false positives",
        "category": "basic"
    },
    "recall": {
        "class": RecallLoss,
        "formula": "TP/(TP+FN)",
        "use_case": "Minimize false negatives",
        "category": "basic"
    },
    "specificity": {
        "class": SpecificityLoss,
        "formula": "TN/(TN+FP)",
        "use_case": "Identify negatives",
        "category": "basic"
    },

    # F-Score Family
    "f1": {
        "class": F1Loss,
        "formula": "2TP/(2TP+FP+FN)",
        "use_case": "Balance precision/recall",
        "category": "f_score"
    },
    "f_beta": {
        "class": FBetaLoss,
        "formula": "((1+beta^2)TP)/((1+beta^2)TP+beta^2*FN+FP)",
        "use_case": "Weighted precision/recall",
        "category": "f_score"
    },

    # Balanced Metrics
    "balanced_accuracy": {
        "class": BalancedAccuracyLoss,
        "formula": "(Sensitivity+Specificity)/2",
        "use_case": "Imbalanced datasets",
        "category": "balanced"
    },
    "g_mean": {
        "class": GeometricMeanLoss,
        "formula": "sqrt(Sensitivity*Specificity)",
        "use_case": "Imbalanced datasets",
        "category": "balanced"
    },
    "youden_j": {
        "class": YoudenJLoss,
        "formula": "Sensitivity+Specificity-1",
        "use_case": "ROC optimization",
        "category": "balanced"
    },

    # Correlation Metrics
    "mcc": {
        "class": MCCLoss,
        "formula": "(TP*TN-FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))",
        "use_case": "Best overall binary metric",
        "category": "correlation"
    },
    "cohen_kappa": {
        "class": CohenKappaLoss,
        "formula": "(p_o-p_e)/(1-p_e)",
        "use_case": "Agreement beyond chance",
        "category": "correlation"
    },

    # Segmentation Metrics
    "iou": {
        "class": IoULoss,
        "formula": "TP/(TP+FP+FN)",
        "use_case": "Segmentation",
        "category": "segmentation"
    },
    "dice": {
        "class": DiceLoss,
        "formula": "2TP/(2TP+FP+FN)",
        "use_case": "Medical imaging",
        "category": "segmentation"
    },
    "tversky": {
        "class": TverskyLoss,
        "formula": "TP/(TP+alpha*FP+beta*FN)",
        "use_case": "Configurable precision/recall",
        "category": "segmentation"
    },
    "focal_tversky": {
        "class": FocalTverskyLoss,
        "formula": "(1-Tversky)^gamma",
        "use_case": "Hard example mining",
        "category": "segmentation"
    },

    # Composite
    "weighted_bce": {
        "class": WeightedCrossEntropyWithAnyLoss,
        "formula": "alpha*AnyLoss + (1-alpha)*BCE",
        "use_case": "Combined optimization",
        "category": "composite"
    },
}


def get_loss(name: str, **kwargs: Any) -> AnyLoss:
    """Factory function to get a loss by name.

    Parameters
    ----------
    name : str
        Name of the loss function.
    **kwargs : dict
        Additional keyword arguments passed to the loss constructor.

    Returns
    -------
    AnyLoss
        Instance of the requested loss function.

    Raises
    ------
    ValueError
        If the loss name is not recognized.

    Examples
    --------
    >>> loss = get_loss("f1", from_logits=True)
    >>> loss = get_loss("tversky", alpha=0.3, beta=0.7)
    """
    name_lower = name.lower()
    if name_lower not in ANYLOSS_REGISTRY:
        available = ", ".join(sorted(ANYLOSS_REGISTRY.keys()))
        raise ValueError(f"Unknown loss '{name}'. Available losses: {available}")

    return ANYLOSS_REGISTRY[name_lower]["class"](**kwargs)
