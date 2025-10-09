"""
Symmetric Mean Absolute Percentage Error (SMAPE).

This loss function provides a relative error metric that is designed to
overcome some of the limitations of the more common Mean Absolute
Percentage Error (MAPE). While MAPE normalizes the error by the ground
truth value, which leads to undefined results for zero-valued targets and
asymmetric penalties, SMAPE normalizes by the average magnitude of both
the true and predicted values.

Architecturally, this symmetric normalization makes the loss function
less biased towards penalizing over-prediction versus under-prediction
and provides a well-defined error value when the true value is zero. The
resulting error is bounded (typically between 0% and 200%), making it a
stable and interpretable objective for training forecasting models,
especially when comparing performance across multiple time series with
different scales.

Foundational Mathematics
------------------------
SMAPE is defined based on the absolute difference between the true value
`y_true` and the predicted value `y_pred`, scaled by the average of their
absolute values. The formula for a single data point is:

    SMAPE_i = |y_pred_i - y_true_i| / ((|y_true_i| + |y_pred_i|) / 2)

The final loss is the mean of these values, typically multiplied by 100
to be expressed as a percentage.

The key component is the denominator: `(|y_true| + |y_pred|) / 2`. This
term treats the ground truth and the prediction symmetrically, ensuring
that the scaling factor is non-zero as long as at least one of the values
is non-zero. This symmetry mitigates the issue in MAPE where a small
over-prediction on a near-zero true value can lead to an extremely large
percentage error. The loss is zero for a perfect forecast and approaches
a maximum of 200% as one value approaches zero while the other remains
large.

References
----------
While several variants exist, this definition is a widely used form,
popularized in forecasting competitions like the M-competitions. Its
properties are discussed in:
-   Makridakis, S. (1993). "Accuracy measures: theoretical and practical
    concerns". *International Journal of Forecasting*.
-   Chen, C., & Yang, Y. (2004). "Assessing forecast accuracy measures".
    *Working paper, Department of Statistics, Texas A&M University*.
"""

import keras
from keras import ops

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SMAPELoss(keras.losses.Loss):
    """Symmetric Mean Absolute Percentage Error (SMAPE) loss.

    This implementation of SMAPE is a common variant that is bounded between 0 and 100.
    It's commonly used in time series forecasting competitions.

    The formula used is:
    SMAPE = 100 * mean(|y_true - y_pred| / ((|y_true| + |y_pred|) / 2))

    Args:
        epsilon: A small float value to avoid division by zero.
        name: Name for the loss function.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        epsilon: float = 1e-8,
        name: str = "smape_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.epsilon = epsilon

    def call(self,
             y_true: keras.KerasTensor,
             y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute SMAPE loss.

        Args:
            y_true: Ground truth values.
            y_pred: The predicted values.

        Returns:
            SMAPE loss value.
        """
        y_true = ops.cast(y_true, y_pred.dtype)

        # Calculate absolute differences
        abs_diff = ops.abs(y_true - y_pred)

        # Calculate denominator
        denominator = (ops.abs(y_true) + ops.abs(y_pred)) / 2.0

        # Add epsilon to avoid division by zero
        denominator = ops.maximum(denominator, self.epsilon)

        # Calculate SMAPE
        smape = ops.mean(abs_diff / denominator) * 100.0

        return smape

    def get_config(self) -> dict:
        """Get loss configuration."""
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
        })
        return config

# ---------------------------------------------------------------------

def smape_metric(y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
    """SMAPE metric function for use with `model.compile()`.

    This metric is a standalone function providing a common variant of SMAPE.
    It uses a small epsilon to avoid division by zero.

    Args:
        y_true: Ground truth values.
        y_pred: The predicted values.

    Returns:
        SMAPE metric value.
    """
    y_true = ops.cast(y_true, y_pred.dtype)
    epsilon = 1e-8

    abs_diff = ops.abs(y_true - y_pred)
    denominator = (ops.abs(y_true) + ops.abs(y_pred)) / 2.0

    return ops.mean(abs_diff / ops.maximum(denominator, epsilon)) * 100.0

# ---------------------------------------------------------------------
