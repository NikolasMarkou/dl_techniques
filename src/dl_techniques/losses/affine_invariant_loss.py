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

Masked targets:
    With ``masked=True`` the last channel of ``y_true`` is a validity mask
    (1 = valid) and ``y_true[..., :-1]`` the target. The median, the mean
    absolute deviation and the L1 then run over the valid pixels of each map
    only, which is Depth Anything V1's labeled loss (arXiv 2401.10891, section
    3.1, eq. 1-3) on sparse ground truth such as MegaDepth's.

References:
    -   Yang, L., et al. (2024). "Depth Anything: Unleashing the Power of
        Large-Scale Unlabeled Data." arXiv 2401.10891, eq. 1-3.
    The concept of a scale-invariant loss was popularized in early deep
    learning approaches to monocular depth estimation.
    -   Eigen, D., Puhrsch, C., & Fergus, R. (2014). "Depth Map Prediction
        from a Single Image using a Multi-Scale Deep Network."
        https://arxiv.org/abs/1406.2283
"""

import keras
from typing import Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


def _masked_median(values: keras.KerasTensor, mask: keras.KerasTensor) -> keras.KerasTensor:
    """Per-row median of ``values`` over ``mask``; ``(B, N)`` inputs, ``(B,)`` output.

    A row with no valid entry gets 0. The two middle order statistics are read
    with one flat ``take`` (row offset plus rank), not ``take_along_axis``,
    whose XLA lowering on GPU fails to retrace at a second batch size.
    """
    count = keras.ops.sum(mask, axis=1)
    n_pixels = keras.ops.shape(values)[1]
    # Invalid entries sort after every valid one.
    above = keras.ops.stop_gradient(keras.ops.max(values)) + 1.0
    ordered = keras.ops.sort(keras.ops.where(mask > 0, values, above), axis=1)
    rank = keras.ops.maximum(count - 1.0, 0.0) / 2.0
    lo = keras.ops.cast(keras.ops.floor(rank), "int32")
    hi = keras.ops.cast(keras.ops.ceil(rank), "int32")
    rows = keras.ops.arange(keras.ops.shape(values)[0], dtype="int32") * n_pixels
    flat = keras.ops.reshape(ordered, (-1,))
    median = 0.5 * (keras.ops.take(flat, rows + lo) + keras.ops.take(flat, rows + hi))
    return keras.ops.where(count > 0, median, keras.ops.zeros_like(median))


@register_dl_technique("dl_techniques.losses.affine_invariant_loss")
class AffineInvariantLoss(keras.losses.Loss):
    """Affine-invariant loss for scale-invariant depth prediction.

    Implements a scale and shift invariant loss function suitable
    for multi-dataset training with different depth scales. The loss
    normalizes both ground truth and predicted depth maps by their
    median and mean absolute deviation before computing the L1 loss.

    Args:
        epsilon: Float, small constant for numerical stability when
            computing scale normalization. Defaults to 1e-6.
        masked: Bool. ``True`` reads a validity mask from the last channel of
            ``y_true`` and computes every statistic over the valid pixels only
            (see the module docstring). Defaults to ``False``, the unmasked
            loss over every pixel.
        name: String, name of the loss function. Defaults to 'affine_invariant_loss'.
        **kwargs: Additional keyword arguments passed to the parent Loss class.
    """

    def __init__(
            self,
            epsilon: float = 1e-6,
            masked: bool = False,
            name: str = 'affine_invariant_loss',
            **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")

        self.epsilon = epsilon
        self.masked = bool(masked)
        logger.info(f"Initialized AffineInvariantLoss with epsilon={epsilon}")

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute affine-invariant depth loss.

        Args:
            y_true: Ground truth depth maps. Shape: (batch_size, height, width, channels)
            y_pred: Predicted depth maps. Shape: (batch_size, height, width, channels)

        Returns:
            Per-sample loss with shape (batch_size,) -- the normalized L1 error of
            each depth map. Keras' reduction recovers the scalar this used to
            return; keeping the batch axis is what lets ``sample_weight`` and
            ``reduction=`` select rows.
        """
        if self.masked:
            return self._masked_loss(y_true, y_pred)

        # Flatten spatial dimensions for statistics computation
        batch_size = keras.ops.shape(y_true)[0]
        y_true_flat = keras.ops.reshape(y_true, (batch_size, -1))
        y_pred_flat = keras.ops.reshape(y_pred, (batch_size, -1))

        # Apply scale and shift normalization
        y_true_scaled = self._scale_and_shift(y_true_flat)
        y_pred_scaled = self._scale_and_shift(y_pred_flat)

        # Compute L1 loss between normalized depth maps, per sample. Reducing over
        # the batch axis here would make `sample_weight` multiply a scalar,
        # charging every row the batch aggregate. The scale/shift normalization is
        # already per-sample, and every row holds the same pixel count, so the mean
        # of this vector equals the old all-axes mean exactly.
        loss = keras.ops.mean(keras.ops.abs(keras.ops.subtract(y_true_scaled, y_pred_scaled)), axis=-1)

        return loss

    def _masked_loss(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Per-sample masked loss; ``y_true[..., -1]`` is the validity mask."""
        batch_size = keras.ops.shape(y_pred)[0]
        pred = keras.ops.reshape(y_pred, (batch_size, -1))
        target = keras.ops.cast(
            keras.ops.reshape(y_true[..., :-1], (batch_size, -1)), pred.dtype)
        mask = keras.ops.cast(
            keras.ops.reshape(y_true[..., -1:], (batch_size, -1)), pred.dtype)
        count = keras.ops.maximum(keras.ops.sum(mask, axis=1), 1.0)

        def normalize(d):
            t = keras.ops.expand_dims(_masked_median(d, mask), axis=-1)
            s = keras.ops.sum(keras.ops.abs(d - t) * mask, axis=1, keepdims=True)
            s = s / keras.ops.expand_dims(count, axis=-1)
            return (d - t) / keras.ops.maximum(s, keras.ops.cast(self.epsilon, s.dtype))

        error = keras.ops.abs(normalize(pred) - normalize(target)) * mask
        return keras.ops.sum(error, axis=1) / count

    def _scale_and_shift(self, d: keras.KerasTensor) -> keras.KerasTensor:
        """Apply scale and shift normalization to depth values.

        Args:
            d: Depth tensor with shape (batch_size, num_pixels)

        Returns:
            Normalized depth tensor with same shape as input.
        """
        # Compute median (approximate using sorted values)
        d_sorted = keras.ops.sort(d, axis=-1)
        num_pixels = keras.ops.shape(d)[-1]
        median_idx = keras.ops.cast(num_pixels // 2, dtype="int32")

        # Handle even/odd number of pixels for median
        if_even = keras.ops.equal(num_pixels % 2, 0)
        median_even = keras.ops.mean(
            keras.ops.stack([
                d_sorted[:, median_idx - 1],
                d_sorted[:, median_idx]
            ], axis=-1),
            axis=-1
        )
        median_odd = d_sorted[:, median_idx]

        t = keras.ops.where(if_even, median_even, median_odd)
        t = keras.ops.expand_dims(t, axis=-1)  # Add back last dimension

        # Compute scale (mean absolute deviation from median)
        s = keras.ops.mean(keras.ops.abs(keras.ops.subtract(d, t)), axis=-1, keepdims=True)

        # Apply normalization with epsilon for numerical stability
        s_stable = keras.ops.maximum(s, keras.ops.cast(self.epsilon, s.dtype))

        return keras.ops.divide(keras.ops.subtract(d, t), s_stable)

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration of the loss function.

        Returns:
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({
            "epsilon": self.epsilon,
            "masked": self.masked,
        })
        return config

# ---------------------------------------------------------------------
