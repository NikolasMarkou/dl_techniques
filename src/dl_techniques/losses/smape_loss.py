import keras
from keras import ops


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

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
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