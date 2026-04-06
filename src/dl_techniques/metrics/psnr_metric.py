import keras
from keras import ops
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class PsnrMetric(keras.metrics.Metric):
    """PSNR metric that evaluates only the primary output for multi-output models.

    Computes Peak Signal-to-Noise Ratio: PSNR = 10 * log10(max_val^2 / MSE).

    This metric is designed for deep supervision scenarios where the model
    produces multiple outputs but we want to track the quality of only the
    main (typically highest resolution) output during training.

    Args:
        max_val: Maximum possible pixel value of the images.
        name: Metric name for logging and visualization.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        max_val: float = 1.0,
        name: str = "primary_psnr",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.max_val = max_val
        self.psnr_sum = self.add_weight(name="psnr_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: Union[keras.KerasTensor, List[keras.KerasTensor]],
        y_pred: Union[keras.KerasTensor, List[keras.KerasTensor]],
        sample_weight: Optional[keras.KerasTensor] = None,
    ) -> None:
        """Update PSNR state using only the primary output.

        Args:
            y_true: Ground truth tensor(s), matching structure of y_pred.
            y_pred: Prediction tensor(s), either single tensor or list for multi-output.
            sample_weight: Optional sample weighting (currently unused).
        """
        # Extract primary output from potentially multi-output structure
        if isinstance(y_pred, list):
            primary_pred = y_pred[0]
            primary_true = y_true[0]
        else:
            primary_pred = y_pred
            primary_true = y_true

        primary_pred = ops.cast(primary_pred, "float32")
        primary_true = ops.cast(primary_true, "float32")

        # Compute per-image MSE: mean over all axes except batch
        # Shape: (batch,) after reducing spatial + channel dims
        diff = primary_true - primary_pred
        # Flatten spatial dims: (batch, -1) then mean over axis=1
        batch_size = ops.shape(diff)[0]
        diff_flat = ops.reshape(diff, (batch_size, -1))
        mse_per_image = ops.mean(diff_flat ** 2, axis=-1)

        # PSNR = 10 * log10(max_val^2 / MSE)
        # Use clamp to avoid log(0) when MSE is zero
        mse_clamped = ops.maximum(mse_per_image, keras.backend.epsilon())
        psnr_per_image = 10.0 * ops.log10(self.max_val ** 2 / mse_clamped)

        self.psnr_sum.assign_add(ops.sum(psnr_per_image))
        self.count.assign_add(ops.cast(batch_size, "float32"))

    def result(self) -> keras.KerasTensor:
        """Compute the mean PSNR across all processed samples."""
        return ops.divide_no_nan(self.psnr_sum, self.count)

    def reset_state(self) -> None:
        """Reset metric state for new epoch or evaluation period."""
        self.psnr_sum.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization."""
        config = super().get_config()
        config.update({
            "max_val": self.max_val,
        })
        return config


# ---------------------------------------------------------------------
