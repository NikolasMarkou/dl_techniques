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
"""

import keras
from keras import ops
from typing import Dict, Optional, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

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
        # Use keras.ops for backend compatibility
        shifted = ops.subtract(inputs, 0.5)
        scaled = ops.multiply(shifted, self.amplifying_scale)
        exp_neg_scaled = ops.exp(ops.negative(scaled))
        one_plus_exp = ops.add(1.0, exp_neg_scaled)
        return ops.divide(1.0, one_plus_exp)

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
    ...         # Compute custom metric
    ...         metric = ...
    ...         # Return 1 - metric as the loss value
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

        logger.debug(f"Initialized {self.__class__.__name__} with amplifying_scale={amplifying_scale}, "
                    f"from_logits={from_logits}")

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
            Tuple containing (TN, FN, FP, TP) confusion matrix entries.
        """
        epsilon = keras.backend.epsilon()

        # Apply sigmoid if predictions are logits
        if self.from_logits:
            y_pred_sigmoid = keras.activations.sigmoid(y_pred)
        else:
            y_pred_sigmoid = y_pred

        # Apply approximation function
        y_approx = self.approximation(y_pred_sigmoid)

        # Ensure y_true is of correct type
        y_true_float = ops.cast(y_true, dtype=y_pred_sigmoid.dtype)

        # Calculate confusion matrix entries using keras.ops
        true_positive = ops.sum(ops.multiply(y_true_float, y_approx))
        false_negative = ops.sum(ops.multiply(y_true_float, ops.subtract(1.0, y_approx)))
        false_positive = ops.sum(ops.multiply(ops.subtract(1.0, y_true_float), y_approx))
        true_negative = ops.sum(ops.multiply(ops.subtract(1.0, y_true_float), ops.subtract(1.0, y_approx)))

        # Add small epsilon to avoid division by zero
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
class WeightedCrossEntropyWithAnyLoss(AnyLoss):
    """Combines weighted binary cross-entropy with any AnyLoss-based metric.

    This loss function combines traditional binary cross-entropy with a metric-based
    loss from the AnyLoss framework, allowing for a balance between the two.

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

        logger.info(f"Created WeightedCrossEntropyWithAnyLoss with alpha={alpha}, "
                   f"combining {anyloss.__class__.__name__} and BinaryCrossentropy")

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

