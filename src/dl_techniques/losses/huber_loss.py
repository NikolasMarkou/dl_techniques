"""
Huber loss, a robust loss function for regression tasks.

This loss function provides a compromise between the Mean Absolute Error
(MAE) and the Mean Squared Error (MSE), combining the strengths of both.
It is designed to be less sensitive to outliers in data than MSE, which
can be dominated by large errors, while still providing a smooth,
strongly convex penalty for small errors, which is beneficial for stable
optimization near the minimum.

Architecturally, Huber loss acts as a hybrid objective that dynamically
switches its behavior based on the magnitude of the prediction error. This
makes it particularly suitable for training models on datasets where some
data points may be corrupted or are otherwise unrepresentative outliers
that should not disproportionately influence the final model.

Foundational Mathematics
------------------------
The Huber loss is defined piecewise based on a threshold parameter, `δ`
(delta). For an error term `e = y_true - y_pred`, the loss `L_δ(e)` is:

         | 0.5 * e²                      if |e| <= δ
L_δ(e) = |
         | δ * (|e| - 0.5 * δ)          if |e| > δ

The intuition behind this formulation is as follows:
1.  **Quadratic Behavior (like MSE)**: When the absolute error is smaller
    than `δ`, the loss is quadratic. This ensures a unique minimum and
    provides smoothly decreasing gradients as the prediction approaches
    the true value, which is ideal for fine-tuning when the model is
    close to the correct answer.
2.  **Linear Behavior (like MAE)**: When the absolute error exceeds `δ`,
    the loss becomes linear. This is the key to its robustness. Unlike
    MSE, where the gradient increases linearly with the error, the Huber
    loss has a constant gradient for large errors. This prevents single
    outlier data points from generating excessively large gradients that
    could destabilize the training process.

The parameter `δ` acts as a tunable threshold that defines which data
points are considered outliers. The function is specifically constructed
to be continuously differentiable at the transition points `|e| = δ`,
ensuring smooth optimization.

References
----------
The Huber loss was introduced in the context of robust statistics:
-   Huber, P. J. (1964). "Robust Estimation of a Location Parameter".
    *The Annals of Mathematical Statistics*.
"""

import keras
from keras import ops

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class HuberLoss(keras.losses.Loss):
    """Huber loss for robust time series forecasting.

    Huber loss is less sensitive to outliers than squared error loss. It behaves
    as a quadratic function for small errors and as a linear function for
    large errors.

    Args:
        delta: The threshold at which to switch from quadratic to linear loss.
        name: Name for the loss function.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        delta: float = 1.0,
        name: str = "huber_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.delta = delta

    def call(self,
             y_true: keras.KerasTensor,
             y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute Huber loss.

        Args:
            y_true: Ground truth values.
            y_pred: The predicted values.

        Returns:
            Huber loss value.
        """
        y_true = ops.cast(y_true, y_pred.dtype)

        # Calculate absolute error
        abs_error = ops.abs(y_true - y_pred)

        # Define quadratic and linear parts of the loss
        quadratic = 0.5 * ops.square(abs_error)
        linear = self.delta * (abs_error - 0.5 * self.delta)

        # Combine them using a threshold
        loss = ops.where(abs_error <= self.delta, quadratic, linear)

        return ops.mean(loss)

    def get_config(self) -> dict:
        """Get loss configuration."""
        config = super().get_config()
        config.update({
            'delta': self.delta,
        })
        return config