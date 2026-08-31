"""
Scaled MSE Loss
===============

Multi-scale Mean Squared Error loss with automatic target resizing for
deep supervision architectures.

When a model produces predictions at multiple spatial resolutions (e.g.,
a U-Net with intermediate outputs), this loss automatically resizes the
ground truth target to match each prediction's spatial dimensions before
computing MSE. This eliminates the need for manual resizing in the
training loop.

Example::

    from dl_techniques.losses import ScaledMseLoss

    # Deep supervision with multiple output scales
    model.compile(
        optimizer='adam',
        loss=[ScaledMseLoss() for _ in range(num_outputs)],
    )
"""

import keras
from keras import ops
from typing import Any, Dict
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.losses.scaled_mse_loss")
class ScaledMseLoss(keras.losses.Loss):
    """MSE loss with automatic target resizing for multi-scale supervision.

    Resizes ``y_true`` to match the spatial dimensions of ``y_pred`` before
    computing the mean squared error. This is useful for deep supervision
    where intermediate outputs have different resolutions than the target.

    Args:
        interpolation: Interpolation method for resizing
            (``'bilinear'``, ``'nearest'``, ``'bicubic'``, ``'lanczos3'``,
            ``'lanczos5'``). Defaults to ``'bilinear'``.
        name: Loss name for logging.
        reduction: Reduction method (``'sum_over_batch_size'``,
            ``'sum'``, ``'none'``).
        **kwargs: Additional arguments passed to ``keras.losses.Loss``.
    """

    def __init__(
        self,
        interpolation: str = "bilinear",
        name: str = "scaled_mse_loss",
        reduction: str = "sum_over_batch_size",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.interpolation = interpolation

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute MSE after resizing y_true to match y_pred spatial dims.

        Args:
            y_true: Ground truth tensor of shape ``(B, H, W, C)``.
            y_pred: Prediction tensor of shape ``(B, H', W', C)``.

        Returns:
            Per-sample MSE of shape ``(B,)``. Keras' reduction recovers the scalar
            this used to return; keeping the batch axis is what lets
            ``sample_weight`` and the ``reduction`` argument select rows.
        """
        y_true = ops.cast(y_true, y_pred.dtype)
        pred_shape = ops.shape(y_pred)
        y_true_resized = ops.image.resize(
            y_true,
            size=(pred_shape[1], pred_shape[2]),
            interpolation=self.interpolation,
        )
        # Mean over the spatial and channel axes ONLY, keeping the batch axis.
        # A scalar here would be multiplied by `sample_weight` before Keras
        # reduces, charging every row the batch aggregate. Each row holds the same
        # H'*W'*C element count, so the mean of this vector equals the old
        # all-axes mean exactly.
        return ops.mean(ops.square(y_pred - y_true_resized), axis=(1, 2, 3))

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "interpolation": self.interpolation,
        })
        return config
