"""
A loss function based on Spiegelhalter's Z-test.

This loss function provides a differentiable objective for directly optimizing
a model's calibration during training. Calibration refers to the statistical
consistency between a model's predicted probabilities and the actual
long-run frequencies of outcomes. A well-calibrated model that predicts a
40% probability for an event should be correct 40% of the time over many
such predictions. This loss specifically targets and minimizes systematic
bias (i.e., consistent over- or under-prediction).

Conceptual Overview:
    The core idea is to reframe a statistical test for calibration as a
    differentiable loss function. Spiegelhalter's Z-test is a statistical
    tool that measures the standardized difference between observed outcomes
    and predicted probabilities. A Z-statistic close to zero indicates good
    calibration, while large positive or negative values suggest systematic
    under-prediction or over-prediction, respectively. By minimizing the
    magnitude (or square) of this Z-statistic, the model is explicitly
    trained to reduce this bias.

Architectural Design:
    The loss function computes the Spiegelhalter Z-statistic over a batch of
    predictions and their corresponding true outcomes. The objective is to
    drive this statistic towards zero. The loss is therefore formulated as
    the square of the Z-statistic (`Z²`), which creates a smooth, convex
    objective with a global minimum at `Z=0`. This approach turns the
    problem of improving calibration into a standard gradient-based
    optimization problem.

Mathematical Formulation:
    Spiegelhalter's Z-statistic is defined as the sum of residuals
    (observed - predicted) divided by the standard deviation of that sum,
    under the null hypothesis that the model is well-calibrated.

    For a batch of `N` samples, let `oᵢ` be the observed outcome (0 or 1) and
    `pᵢ` be the predicted probability for the i-th sample.

    The Z-statistic is calculated as:

        Z = Σᵢ (oᵢ - pᵢ) / sqrt( Σᵢ pᵢ(1 - pᵢ) )

    -   Numerator `Σᵢ (oᵢ - pᵢ)`: This is the sum of the residuals. It
        represents the net difference between the number of observed positive
        outcomes and the number of expected positive outcomes. For a
        well-calibrated model, this sum should be close to zero.
    -   Denominator `sqrt( Σᵢ pᵢ(1 - pᵢ) )`: This is the standard deviation
        of the sum of residuals. It is derived from the property that each
        prediction `pᵢ` can be seen as a parameter of an independent
        Bernoulli trial, whose variance is `pᵢ(1 - pᵢ)`.

    The loss function is then the square of this statistic:

        Loss = Z²

References:
    - Spiegelhalter, D. J. (1986). "Probabilistic prediction in patient
      management and clinical trials." Statistics in Medicine, 5(5), 421-433.
"""

import keras
from keras import ops
from typing import Dict, Optional, Any

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BrierScoreLoss(keras.losses.Loss):
    """Differentiable Brier Score loss function for model calibration.

    The Brier Score measures the mean squared difference between predicted
    probabilities and actual outcomes: B = (1/N) * Σ(pᵢ - oᵢ)²

    A lower Brier Score indicates better calibration and accuracy.
    This loss function directly minimizes the Brier Score.

    Args:
        from_logits: Whether the predictions are logits (not passed through
            a sigmoid). Default is False.
        reduction: Type of reduction to apply to the loss. Default is 'sum_over_batch_size'.
        name: Optional name for the loss function.
        **kwargs: Additional keyword arguments passed to the parent class.

    Example:
        >>> model = keras.Sequential([
        ...     keras.layers.Dense(64, activation="relu"),
        ...     keras.layers.Dense(1, activation="sigmoid")
        ... ])
        >>> model.compile(
        ...     optimizer=keras.optimizers.Adam(learning_rate=0.001),
        ...     loss=BrierScoreLoss(),
        ...     metrics=["accuracy"]
        ... )
    """

    def __init__(
        self,
        from_logits: bool = False,
        reduction: str = 'sum_over_batch_size',
        name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the BrierScoreLoss.

        Args:
            from_logits: Whether model outputs raw logits without sigmoid.
            reduction: Type of reduction to apply to the loss.
            name: Optional name for the loss.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            reduction=reduction,
            name=name or "brier_score_loss",
            **kwargs
        )
        self.from_logits = from_logits

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the Brier Score loss.

        Args:
            y_true: Ground truth binary labels (0 or 1).
            y_pred: Predicted probabilities or logits.

        Returns:
            Brier Score loss value.
        """
        # Apply sigmoid if predictions are logits
        if self.from_logits:
            y_pred = ops.sigmoid(y_pred)

        # Ensure y_true is of correct type
        y_true = ops.cast(y_true, dtype=y_pred.dtype)

        # Compute Brier Score: mean squared difference
        squared_diff = ops.square(y_pred - y_true)
        brier_score = ops.mean(squared_diff)

        return brier_score

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration for serialization.

        Returns:
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({"from_logits": self.from_logits})
        return config

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SpiegelhalterZLoss(keras.losses.Loss):
    """Differentiable Spiegelhalter's Z-test loss function for model calibration.

    Spiegelhalter's Z-test measures systematic bias in probability predictions:
    Z = Σ(oᵢ - pᵢ) / √[Σ pᵢ(1 - pᵢ)]

    For well-calibrated models, Z should be close to 0. This loss function
    minimizes |Z| or Z² to achieve good calibration.

    Args:
        use_squared: Whether to use Z² instead of |Z|. Default is True.
        from_logits: Whether the predictions are logits (not passed through
            a sigmoid). Default is False.
        reduction: Type of reduction to apply to the loss. Default is 'sum_over_batch_size'.
        name: Optional name for the loss function.
        **kwargs: Additional keyword arguments passed to the parent class.

    Example:
        >>> model = keras.Sequential([
        ...     keras.layers.Dense(64, activation="relu"),
        ...     keras.layers.Dense(1, activation="sigmoid")
        ... ])
        >>> model.compile(
        ...     optimizer=keras.optimizers.Adam(learning_rate=0.001),
        ...     loss=SpiegelhalterZLoss(use_squared=True),
        ...     metrics=["accuracy"]
        ... )
    """

    def __init__(
        self,
        use_squared: bool = True,
        from_logits: bool = False,
        reduction: str = 'sum_over_batch_size',
        name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the SpiegelhalterZLoss.

        Args:
            use_squared: Whether to use Z² instead of |Z| for smoother gradients.
            from_logits: Whether model outputs raw logits without sigmoid.
            reduction: Type of reduction to apply to the loss.
            name: Optional name for the loss.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            reduction=reduction,
            name=name or "spiegelhalter_z_loss",
            **kwargs
        )
        self.use_squared = use_squared
        self.from_logits = from_logits

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the Spiegelhalter Z-test loss.

        Args:
            y_true: Ground truth binary labels (0 or 1).
            y_pred: Predicted probabilities or logits.

        Returns:
            Spiegelhalter Z-test loss value.
        """
        # Apply sigmoid if predictions are logits
        if self.from_logits:
            y_pred = ops.sigmoid(y_pred)

        # Ensure y_true is of correct type
        y_true = ops.cast(y_true, dtype=y_pred.dtype)

        # Compute residuals: observed - predicted
        residuals = y_true - y_pred

        # Compute variance of residuals under null hypothesis of good calibration
        # Each prediction follows Bernoulli with variance p(1-p)
        variances = y_pred * (1.0 - y_pred)

        # Sum of residuals (numerator of Z-statistic)
        residual_sum = ops.sum(residuals)

        # Standard deviation of residual sum (denominator of Z-statistic)
        # Add small epsilon for numerical stability
        epsilon = keras.backend.epsilon()
        variance_sum = ops.sum(variances) + epsilon
        residual_std = ops.sqrt(variance_sum)

        # Compute Z-statistic
        z_stat = residual_sum / residual_std

        # Return Z² or |Z| depending on configuration
        if self.use_squared:
            return ops.square(z_stat)
        else:
            return ops.abs(z_stat)

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration for serialization.

        Returns:
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({
            "use_squared": self.use_squared,
            "from_logits": self.from_logits
        })
        return config

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CombinedCalibrationLoss(keras.losses.Loss):
    """Combined loss function using both Brier Score and Spiegelhalter's Z-test.

    This loss function combines both calibration metrics to optimize for both
    overall accuracy (Brier Score) and systematic bias (Spiegelhalter's Z-test).

    Loss = α * BrierScore + (1-α) * SpiegelhalterZ

    Args:
        alpha: Weight for the Brier Score component. The Spiegelhalter Z component
            has a weight of (1-alpha). Default is 0.5.
        use_squared_z: Whether to use Z² instead of |Z| for the Z-test component.
            Default is True.
        from_logits: Whether the predictions are logits (not passed through
            a sigmoid). Default is False.
        reduction: Type of reduction to apply to the loss. Default is 'sum_over_batch_size'.
        name: Optional name for the loss function.
        **kwargs: Additional keyword arguments passed to the parent class.

    Example:
        >>> model = keras.Sequential([
        ...     keras.layers.Dense(64, activation="relu"),
        ...     keras.layers.Dense(1, activation="sigmoid")
        ... ])
        >>> model.compile(
        ...     optimizer=keras.optimizers.Adam(learning_rate=0.001),
        ...     loss=CombinedCalibrationLoss(alpha=0.7),
        ...     metrics=["accuracy"]
        ... )
    """

    def __init__(
        self,
        alpha: float = 0.5,
        use_squared_z: bool = True,
        from_logits: bool = False,
        reduction: str = 'sum_over_batch_size',
        name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the CombinedCalibrationLoss.

        Args:
            alpha: Weight for the Brier Score component.
            use_squared_z: Whether to use Z² instead of |Z| for the Z-test.
            from_logits: Whether model outputs raw logits without sigmoid.
            reduction: Type of reduction to apply to the loss.
            name: Optional name for the loss.
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            ValueError: If alpha is not in the range [0, 1].
        """
        if alpha < 0 or alpha > 1:
            raise ValueError(f"alpha must be in the range [0, 1], got {alpha}")

        super().__init__(
            reduction=reduction,
            name=name or "combined_calibration_loss",
            **kwargs
        )
        self.alpha = alpha
        self.use_squared_z = use_squared_z
        self.from_logits = from_logits

        # Initialize component losses
        self.brier_loss = BrierScoreLoss(from_logits=from_logits, reduction='none')
        self.z_loss = SpiegelhalterZLoss(
            use_squared=use_squared_z,
            from_logits=from_logits,
            reduction='none'
        )

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the combined calibration loss.

        Args:
            y_true: Ground truth binary labels (0 or 1).
            y_pred: Predicted probabilities or logits.

        Returns:
            Combined calibration loss value.
        """
        brier_component = self.brier_loss(y_true, y_pred)
        z_component = self.z_loss(y_true, y_pred)

        return self.alpha * brier_component + (1 - self.alpha) * z_component

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration for serialization.

        Returns:
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "use_squared_z": self.use_squared_z,
            "from_logits": self.from_logits
        })
        return config

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BrierScoreMetric(keras.metrics.Metric):
    """Brier Score metric for monitoring calibration during training.

    The Brier Score measures the mean squared difference between predicted
    probabilities and actual outcomes. Lower values indicate better calibration.

    Args:
        name: String name of the metric instance.
        from_logits: Whether predictions are logits (not passed through sigmoid).
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        name: str = 'brier_score',
        from_logits: bool = False,
        **kwargs: Any
    ) -> None:
        """Initialize the BrierScoreMetric.

        Args:
            name: Name of the metric.
            from_logits: Whether predictions are logits.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(name=name, **kwargs)
        self.from_logits = from_logits
        self.total_score = self.add_weight(name='total_score', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        sample_weight: Optional[keras.KerasTensor] = None
    ) -> None:
        """Update the metric state.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted probabilities or logits.
            sample_weight: Optional sample weights.
        """
        if self.from_logits:
            y_pred = ops.sigmoid(y_pred)

        y_true = ops.cast(y_true, y_pred.dtype)
        brier_scores = ops.square(y_pred - y_true)

        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, self.dtype)
            brier_scores = ops.multiply(brier_scores, sample_weight)

        self.total_score.assign_add(ops.sum(brier_scores))
        self.count.assign_add(ops.cast(ops.size(y_true), self.dtype))

    def result(self) -> keras.KerasTensor:
        """Compute the final metric result.

        Returns:
            The Brier Score value.
        """
        # Avoid division by zero
        return ops.where(
            ops.equal(self.count, 0.0),
            0.0,
            self.total_score / self.count
        )

    def reset_state(self) -> None:
        """Reset the metric state."""
        self.total_score.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        """Get metric configuration for serialization.

        Returns:
            Dictionary containing the metric configuration.
        """
        config = super().get_config()
        config.update({"from_logits": self.from_logits})
        return config

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SpiegelhalterZMetric(keras.metrics.Metric):
    """Spiegelhalter Z-statistic metric for monitoring calibration during training.

    The Z-statistic measures systematic bias in probability predictions.
    Values close to 0 indicate good calibration.

    Args:
        name: String name of the metric instance.
        from_logits: Whether predictions are logits (not passed through sigmoid).
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        name: str = 'spiegelhalter_z',
        from_logits: bool = False,
        **kwargs: Any
    ) -> None:
        """Initialize the SpiegelhalterZMetric.

        Args:
            name: Name of the metric.
            from_logits: Whether predictions are logits.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(name=name, **kwargs)
        self.from_logits = from_logits
        self.residual_sum = self.add_weight(name='residual_sum', initializer='zeros')
        self.variance_sum = self.add_weight(name='variance_sum', initializer='zeros')

    def update_state(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        sample_weight: Optional[keras.KerasTensor] = None
    ) -> None:
        """Update the metric state.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted probabilities or logits.
            sample_weight: Optional sample weights.
        """
        if self.from_logits:
            y_pred = ops.sigmoid(y_pred)

        y_true = ops.cast(y_true, y_pred.dtype)
        residuals = y_true - y_pred
        variances = y_pred * (1.0 - y_pred)

        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, self.dtype)
            residuals = ops.multiply(residuals, sample_weight)
            variances = ops.multiply(variances, sample_weight)

        self.residual_sum.assign_add(ops.sum(residuals))
        self.variance_sum.assign_add(ops.sum(variances))

    def result(self) -> keras.KerasTensor:
        """Compute the final metric result.

        Returns:
            The Spiegelhalter Z-statistic value.
        """
        epsilon = keras.backend.epsilon()
        variance_sum_safe = self.variance_sum + epsilon

        # Avoid division by zero
        return ops.where(
            ops.equal(variance_sum_safe, epsilon),
            0.0,
            self.residual_sum / ops.sqrt(variance_sum_safe)
        )

    def reset_state(self) -> None:
        """Reset the metric state."""
        self.residual_sum.assign(0.0)
        self.variance_sum.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        """Get metric configuration for serialization.

        Returns:
            Dictionary containing the metric configuration.
        """
        config = super().get_config()
        config.update({"from_logits": self.from_logits})
        return config

# ---------------------------------------------------------------------

