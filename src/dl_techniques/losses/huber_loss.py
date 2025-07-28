import keras
from keras import ops

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

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
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