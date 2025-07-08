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

    Example:
        >>> import keras
        >>> import tensorflow as tf
        >>> loss_fn = AffineInvariantLoss()
        >>> y_true = tf.random.uniform([8, 64, 64, 1], 0, 10)
        >>> y_pred = tf.random.uniform([8, 64, 64, 1], 0, 10)
        >>> loss = loss_fn(y_true, y_pred)
        >>> print(loss.shape)
        ()
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
