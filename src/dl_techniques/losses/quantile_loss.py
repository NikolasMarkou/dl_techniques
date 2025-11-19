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
from typing import List

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
        y_true = keras.ops.cast(y_true, y_pred.dtype)

        # Calculate the error
        error = y_true - y_pred

        # Calculate the quantile loss
        loss = keras.ops.maximum(self.quantile * error, (self.quantile - 1) * error)

        return keras.ops.mean(loss)

    def get_config(self) -> dict:
        """Get loss configuration."""
        config = super().get_config()
        config.update({
            'quantile': self.quantile,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class QuantileLoss(keras.losses.Loss):
    """
    Vectorized Mean Quantile Loss for probabilistic forecasting.

    This loss function correctly handles multi-quantile predictions. It computes
    the pinball loss for each predicted quantile in a broadcasted, vectorized
    manner, which is highly efficient.

    Args:
        quantiles: A list of floats, the quantiles to be predicted.
        name: Name for the loss function.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, quantiles: List[float], name: str = "quantile_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        if not all(0 < q < 1 for q in quantiles):
            raise ValueError("All quantiles must be strictly between 0 and 1.")
        
        # Store quantiles as a tensor shaped for broadcasting against (batch, time, quantiles)
        # Shape: (1, 1, num_quantiles)
        self.quantiles_tensor = keras.ops.convert_to_tensor(quantiles, dtype=keras.backend.floatx())
        self.quantiles_tensor = keras.ops.reshape(self.quantiles_tensor, (1, 1, len(quantiles)))

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute vectorized quantile loss.

        Args:
            y_true: Ground truth values. Shape: (batch_size, horizon).
            y_pred: The predicted quantile values. Shape: (batch_size, horizon, num_quantiles).

        Returns:
            A single scalar loss value, the mean of the pinball loss across all
            quantiles, time steps, and batch items.
        """
        y_true = keras.ops.cast(y_true, y_pred.dtype)

        # Expand y_true to be broadcastable with y_pred's shape (batch, horizon, quantiles)
        # y_true is (batch, horizon) -> y_true_expanded becomes (batch, horizon, 1)
        y_true_expanded = keras.ops.expand_dims(y_true, axis=-1)

        # Calculate error. Broadcasting handles the shape difference.
        # (batch, horizon, 1) - (batch, horizon, num_quantiles)
        # error shape: (batch, horizon, num_quantiles)
        error = y_true_expanded - y_pred

        # Calculate pinball loss using vectorized operations.
        # Broadcasting self.quantiles_tensor against error.
        # self.quantiles_tensor shape: (1, 1, num_quantiles)
        loss = keras.ops.maximum(
            self.quantiles_tensor * error,
            (self.quantiles_tensor - 1) * error
        )

        # Return the mean over all dimensions to get a single scalar loss
        return keras.ops.mean(loss)

    def get_config(self) -> dict:
        """Get loss configuration for serialization."""
        config = super().get_config()
        # Convert tensor back to a standard list for JSON serialization
        config.update({
            'quantiles': keras.ops.convert_to_numpy(self.quantiles_tensor).flatten().tolist()
        })
        return config

# ---------------------------------------------------------------------