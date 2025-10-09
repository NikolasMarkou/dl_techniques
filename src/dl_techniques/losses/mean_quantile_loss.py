"""
Quantile loss, also known as the pinball loss.

This loss function facilitates quantile regression, a technique that
shifts the modeling objective from predicting a single point estimate (like
the conditional mean in MSE-based regression) to predicting a specific
conditional quantile of the target distribution. This is the cornerstone
of probabilistic forecasting, as it allows a model to quantify uncertainty.
By training a model on multiple quantiles (e.g., 0.1, 0.5, 0.9), one can
construct prediction intervals that provide a range of likely outcomes.

Architecturally, quantile loss enables a standard neural network to
output a distribution's characteristics rather than just its central
tendency. It achieves this by introducing an asymmetric penalty that is
minimized, in expectation, when the model's output corresponds to the
desired quantile of the true conditional distribution.

Foundational Mathematics
------------------------
The quantile loss is an asymmetric function that penalizes over- and
under-predictions differently, with the asymmetry controlled by the target
quantile `q`. For a given prediction error `e = y_true - y_pred`, the loss
`L_q(e)` is defined as:

         | q * e                if e >= 0  (underprediction)
L_q(e) = |
         | (1 - q) * (-e)       if e < 0   (overprediction)

The intuition behind this "pinball loss" is as follows:
-   If the target quantile `q` is 0.9, the loss for underpredicting (`e > 0`)
    is weighted by `0.9`, while the loss for overpredicting (`e < 0`) is
    weighted by `1 - 0.9 = 0.1`. The model is therefore penalized much more
    heavily for being too low than for being too high. To minimize its
    expected loss, the network learns to output a value that is greater
    than the true value 90% of the time, which is the definition of the
    90th percentile.
-   If `q = 0.5`, the weights for under- and over-prediction are both 0.5,
    making the loss symmetric: `L_0.5(e) = 0.5 * |e|`. Minimizing this is
    equivalent to minimizing the Mean Absolute Error (MAE), which is known
    to yield the conditional median.

This mechanism provides a robust and elegant way to train a deterministic
model to produce probabilistic forecasts.

References
----------
The foundational work on quantile regression was introduced by:
-   Koenker, R., & Bassett, G. (1978). "Regression Quantiles".
    *Econometrica*.
"""

import keras
from keras import ops

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MQLoss(keras.losses.Loss):
    """Mean Quantile Loss for probabilistic forecasting.

    This loss function is used for quantile regression, allowing a model to
    predict conditional quantiles and produce prediction intervals.

    Args:
        quantile: The quantile to be predicted, a float between 0 and 1.
        name: Name for the loss function.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        quantile: float = 0.5,
        name: str = "mq_loss",
        **kwargs
    ):
        if not 0 < quantile < 1:
            raise ValueError("The quantile must be strictly between 0 and 1.")
        super().__init__(name=name, **kwargs)
        self.quantile = quantile

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute quantile loss.

        Args:
            y_true: Ground truth values.
            y_pred: The predicted values.

        Returns:
            Quantile loss value.
        """
        y_true = ops.cast(y_true, y_pred.dtype)

        # Calculate the error
        error = y_true - y_pred

        # Calculate the quantile loss
        loss = ops.maximum(self.quantile * error, (self.quantile - 1) * error)

        return ops.mean(loss)

    def get_config(self) -> dict:
        """Get loss configuration."""
        config = super().get_config()
        config.update({
            'quantile': self.quantile,
        })
        return config

# ---------------------------------------------------------------------
