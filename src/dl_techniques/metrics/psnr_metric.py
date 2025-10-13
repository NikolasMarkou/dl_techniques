import keras
import tensorflow as tf
from typing import List, Optional,Union

# ---------------------------------------------------------------------

class PsnrMetric(keras.metrics.Metric):
    """
    PSNR metric that evaluates only the primary output for multi-output models.

    This metric is designed for deep supervision scenarios where the model
    produces multiple outputs but we want to track the quality of only the
    main (typically highest resolution) output during training.
    """

    def __init__(self, name: str = 'primary_psnr', **kwargs) -> None:
        """
        Initialize the primary output PSNR metric.

        Args:
            name: Metric name for logging and visualization
            **kwargs: Additional keyword arguments
        """
        super().__init__(name=name, **kwargs)
        self.psnr_sum = self.add_weight(name='psnr_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(
        self,
        y_true: Union[tf.Tensor, List[tf.Tensor]],
        y_pred: Union[tf.Tensor, List[tf.Tensor]],
        sample_weight: Optional[tf.Tensor] = None
    ) -> None:
        """
        Update PSNR state using only the primary output.

        Args:
            y_true: Ground truth tensor(s), matching structure of y_pred
            y_pred: Prediction tensor(s), either single tensor or list for multi-output
            sample_weight: Optional sample weighting (currently unused)

        Note:
            For multi-output models, both y_true and y_pred are lists with
            matching structures. We extract the first element (primary output).
        """
        # Extract primary output from potentially multi-output structure
        if isinstance(y_pred, list):
            primary_pred = y_pred[0]
            primary_true = y_true[0]
        else:
            primary_pred = y_pred
            primary_true = y_true

        # Compute PSNR for the batch and accumulate
        psnr_batch = tf.image.psnr(primary_pred, primary_true, max_val=1.0)
        self.psnr_sum.assign_add(tf.reduce_sum(psnr_batch))
        self.count.assign_add(tf.cast(tf.size(psnr_batch), tf.float32))

    def result(self) -> tf.Tensor:
        """Compute the mean PSNR across all processed samples."""
        return tf.math.divide_no_nan(self.psnr_sum, self.count)

    def reset_state(self) -> None:
        """Reset metric state for new epoch or evaluation period."""
        self.psnr_sum.assign(0.0)
        self.count.assign(0.0)


# ---------------------------------------------------------------------
