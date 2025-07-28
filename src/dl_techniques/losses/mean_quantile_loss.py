import keras
from keras import ops

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