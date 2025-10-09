"""
A loss function invariant to scale and shift transformations.

This loss is designed for tasks where the absolute scale and shift of the
output are ambiguous or irrelevant, most notably in self-supervised or
multi-dataset monocular depth estimation. When training on diverse datasets
with different units (e.g., meters, feet) or unknown scaling factors, a
standard regression loss like L1 or L2 would incorrectly penalize a
prediction that is structurally correct but globally scaled or shifted.

Conceptual Overview:
    The core principle is to normalize both the predicted and ground truth
    depth maps to a canonical representation before computing their
    difference. This normalization removes any global affine transformation
    (scale and shift), making the loss focus solely on the relative,
    structural correctness of the prediction. For example, a predicted depth
    map that is twice the scale of the ground truth but otherwise perfectly
    proportional will incur zero loss.

Architectural Design:
    To achieve invariance, the loss function performs a per-sample
    normalization. For each depth map in a batch (both prediction and
    target), it calculates two robust statistical measures:
    1.  A shift parameter `t`, estimated using the median of all depth values.
    2.  A scale parameter `s`, estimated using the mean absolute deviation
        (MAD) from the median.

    The median and MAD are chosen over the mean and standard deviation for
    their robustness to outliers, which are common in depth data (e.g.,
    invalid "sky" pixels or sensor noise). After normalization, a standard
    L1 (Mean Absolute Error) loss is computed on the transformed maps.

Mathematical Formulation:
    For a given depth map `d` (either predicted `d_pred` or ground truth
    `d_true`), the normalization is performed as follows:

    1.  Compute the shift: `t = median(d)`
    2.  Compute the scale: `s = mean(|d - t|)`
    3.  Normalize the depth map: `d_norm = (d - t) / (s + ε)`

    where `ε` is a small constant for numerical stability. The final loss
    is the L1 distance between the normalized prediction and ground truth:

    Loss = || d_pred_norm - d_true_norm ||₁

References:
    The concept of a scale-invariant loss was popularized in early deep
    learning approaches to monocular depth estimation.
    -   Eigen, D., Puhrsch, C., & Fergus, R. (2014). "Depth Map Prediction
        from a Single Image using a Multi-Scale Deep Network."
        https://arxiv.org/abs/1406.2283
"""

import keras
from keras import ops
from typing import Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class AffineInvariantLoss(keras.losses.Loss):
    """Affine-invariant loss for scale-invariant depth prediction.

    Implements a scale and shift invariant loss function suitable
    for multi-dataset training with different depth scales. The loss
    normalizes both ground truth and predicted depth maps by their
    median and mean absolute deviation before computing the L1 loss.

    Args:
        epsilon: Float, small constant for numerical stability when
            computing scale normalization. Defaults to 1e-6.
        name: String, name of the loss function. Defaults to 'affine_invariant_loss'.
        **kwargs: Additional keyword arguments passed to the parent Loss class.
    """

    def __init__(
            self,
            epsilon: float = 1e-6,
            name: str = 'affine_invariant_loss',
            **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")

        self.epsilon = epsilon
        logger.info(f"Initialized AffineInvariantLoss with epsilon={epsilon}")

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute affine-invariant depth loss.

        Args:
            y_true: Ground truth depth maps. Shape: (batch_size, height, width, channels)
            y_pred: Predicted depth maps. Shape: (batch_size, height, width, channels)

        Returns:
            Computed loss value as a scalar tensor.
        """
        # Flatten spatial dimensions for statistics computation
        batch_size = ops.shape(y_true)[0]
        y_true_flat = ops.reshape(y_true, (batch_size, -1))
        y_pred_flat = ops.reshape(y_pred, (batch_size, -1))

        # Apply scale and shift normalization
        y_true_scaled = self._scale_and_shift(y_true_flat)
        y_pred_scaled = self._scale_and_shift(y_pred_flat)

        # Compute L1 loss between normalized depth maps
        loss = ops.mean(ops.abs(ops.subtract(y_true_scaled, y_pred_scaled)))

        return loss

    def _scale_and_shift(self, d: keras.KerasTensor) -> keras.KerasTensor:
        """Apply scale and shift normalization to depth values.

        Args:
            d: Depth tensor with shape (batch_size, num_pixels)

        Returns:
            Normalized depth tensor with same shape as input.
        """
        # Compute median (approximate using sorted values)
        d_sorted = ops.sort(d, axis=-1)
        num_pixels = ops.shape(d)[-1]
        median_idx = ops.cast(num_pixels // 2, dtype="int32")

        # Handle even/odd number of pixels for median
        if_even = ops.equal(num_pixels % 2, 0)
        median_even = ops.mean(
            ops.stack([
                d_sorted[:, median_idx - 1],
                d_sorted[:, median_idx]
            ], axis=-1),
            axis=-1
        )
        median_odd = d_sorted[:, median_idx]

        t = ops.where(if_even, median_even, median_odd)
        t = ops.expand_dims(t, axis=-1)  # Add back last dimension

        # Compute scale (mean absolute deviation from median)
        s = ops.mean(ops.abs(ops.subtract(d, t)), axis=-1, keepdims=True)

        # Apply normalization with epsilon for numerical stability
        s_stable = ops.maximum(s, ops.cast(self.epsilon, s.dtype))

        return ops.divide(ops.subtract(d, t), s_stable)

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration of the loss function.

        Returns:
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({
            "epsilon": self.epsilon,
        })
        return config

# ---------------------------------------------------------------------
