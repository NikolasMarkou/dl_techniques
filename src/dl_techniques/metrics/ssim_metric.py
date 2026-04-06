"""
SSIM Metric
===========

Structural Similarity Index Measure (SSIM) metric for image quality
assessment. SSIM is more perceptually meaningful than PSNR as it accounts
for luminance, contrast, and structural information.

.. note::

    This implementation uses ``tf.image.ssim`` under the hood because
    Keras 3 ``keras.ops`` does not yet provide a backend-agnostic SSIM
    function. It therefore requires the TensorFlow backend.

Example::

    from dl_techniques.metrics.ssim_metric import SsimMetric

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=[SsimMetric(max_val=1.0)],
    )
"""

import keras
from keras import ops
import tensorflow as tf
from typing import Any, Dict, Optional


@keras.saving.register_keras_serializable(package="dl_techniques.metrics")
class SsimMetric(keras.metrics.Metric):
    """Structural Similarity Index Measure (SSIM) metric.

    Wraps ``tf.image.ssim`` and tracks a running mean across batches.

    Args:
        max_val: Maximum possible pixel value of the images
            (e.g., 1.0 for [0, 1] range, 255 for [0, 255]).
        name: Metric name for logging.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        max_val: float = 1.0,
        name: str = "ssim",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.max_val = max_val
        self.ssim_sum = self.add_weight(name="ssim_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        sample_weight: Optional[keras.KerasTensor] = None,
    ) -> None:
        """Update SSIM state with a batch of images.

        Args:
            y_true: Ground truth images of shape ``(B, H, W, C)``.
            y_pred: Predicted images of shape ``(B, H, W, C)``.
            sample_weight: Optional per-sample weighting (unused).
        """
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")

        ssim_values = tf.image.ssim(y_true, y_pred, max_val=self.max_val)

        self.ssim_sum.assign_add(ops.sum(ssim_values))
        self.count.assign_add(ops.cast(ops.shape(y_true)[0], "float32"))

    def result(self) -> keras.KerasTensor:
        """Compute the mean SSIM across all processed samples."""
        return ops.divide_no_nan(self.ssim_sum, self.count)

    def reset_state(self) -> None:
        """Reset metric state for new epoch."""
        self.ssim_sum.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "max_val": self.max_val,
        })
        return config
