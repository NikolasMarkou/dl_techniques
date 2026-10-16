"""Monocular depth estimation evaluation metrics.

Standard depth estimation metrics from the Eigen et al. (2014) benchmark
protocol, adapted for masked (sparse) ground truth.  All metrics expect
a concatenated ``y_true`` tensor where the last channel dimension is
split as ``[depth, validity_mask]``:

- ``y_true[..., :1]``  — ground truth depth in ``[-1, +1]``
- ``y_true[..., 1:]``  — binary validity mask (1 = valid, 0 = invalid)

Internally, depth values are shifted to ``[0, 1]`` for ratio-based
computations.  Invalid pixels (mask == 0) are excluded from all
aggregations.

Metrics
-------
- :class:`AbsRelMetric` — Absolute Relative Error (AbsRel)
- :class:`DeltaThresholdMetric` — Threshold accuracy (delta < t)
- :class:`SqRelMetric` — Squared Relative Error (SqRel)
- :class:`RMSEMetric` — Root Mean Squared Error (RMSE)
- :class:`RMSELogMetric` — Root Mean Squared Log Error (RMSE log)

References
----------
.. [1] Eigen, D., Puhrsch, C., & Fergus, R. (2014). Depth Map
   Prediction from a Single Image using a Multi-Scale Deep Network.
   *NeurIPS*.
"""

from typing import Any, Dict, Optional

import keras
from dl_techniques.utils.keras_registration import register_dl_technique


# =====================================================================
# Helpers
# =====================================================================


def _unpack_depth_and_mask(
    y_true_and_mask: keras.KerasTensor,
) -> tuple:
    """Split concatenated ``[depth, mask]`` tensor.

    Args:
        y_true_and_mask: Tensor with shape ``(..., 2)`` where the last
            channel holds ``[depth, mask]``.

    Returns:
        Tuple of ``(depth, mask)`` each with shape ``(..., 1)``.
    """
    depth = y_true_and_mask[..., :1]
    mask = y_true_and_mask[..., 1:]
    return depth, mask


def align_scale_and_shift(
    y_pred: keras.KerasTensor,
    depth_true: keras.KerasTensor,
    mask: keras.KerasTensor,
) -> keras.KerasTensor:
    """Least-squares scale-and-shift alignment of ``y_pred`` to ``depth_true``.

    Per image, solves ``min_{s, t} sum_p mask_p (s * y_pred_p + t - depth_true_p)^2``
    in closed form and returns ``s * y_pred + t``: the alignment the MiDaS
    authors' reference loss (``compute_scale_and_shift``) and the relative-depth
    evaluation protocol apply to a prediction known only up to scale and shift.

    Interface contract: all three tensors are ``(B, H, W, 1)``; ``mask`` is 1
    on valid pixels. An image whose system is singular (fewer than two
    distinct valid predictions, or no valid pixel) gets ``s = t = 0``, as in the
    reference. Never raises; differentiable in ``y_pred``.

    Args:
        y_pred: Prediction, any scale and shift.
        depth_true: Target in the space the metrics compare in.
        mask: Validity mask.

    Returns:
        The aligned prediction, shape of ``y_pred``.
    """
    axes = (1, 2, 3)
    a_00 = keras.ops.sum(mask * y_pred * y_pred, axis=axes)
    a_01 = keras.ops.sum(mask * y_pred, axis=axes)
    a_11 = keras.ops.sum(mask, axis=axes)
    b_0 = keras.ops.sum(mask * y_pred * depth_true, axis=axes)
    b_1 = keras.ops.sum(mask * depth_true, axis=axes)
    det = a_00 * a_11 - a_01 * a_01
    valid = det > 0
    safe_det = keras.ops.where(valid, det, keras.ops.ones_like(det))
    scale = keras.ops.where(valid, (a_11 * b_0 - a_01 * b_1) / safe_det, 0.0)
    shift = keras.ops.where(valid, (a_00 * b_1 - a_01 * b_0) / safe_det, 0.0)
    scale = keras.ops.reshape(scale, (-1, 1, 1, 1))
    shift = keras.ops.reshape(shift, (-1, 1, 1, 1))
    return scale * y_pred + shift


def _to_positive_range(
    depth: keras.KerasTensor,
) -> keras.KerasTensor:
    """Shift depth from ``[-1, +1]`` to ``[0, 1]``."""
    return (depth + 1.0) / 2.0


# =====================================================================
# AbsRelMetric
# =====================================================================


@register_dl_technique("dl_techniques.metrics.depth_metrics")
class AbsRelMetric(keras.metrics.Metric):
    r"""Absolute Relative Error on valid (masked) pixels.

    .. math::

        \text{AbsRel} = \frac{1}{|V|} \sum_{p \in V}
        \frac{|d_p^{pred} - d_p^{gt}|}{d_p^{gt}}

    where *V* is the set of valid pixels.  Depth values are shifted from
    ``[-1, 1]`` to ``[0, 1]`` before computing ratios.  Pixels with
    ground-truth depth below ``min_depth`` (after shifting) are excluded
    to avoid division by near-zero values.

    Args:
        min_depth: Minimum valid depth in ``[0, 1]`` range (after shift).
            Pixels below this are excluded.  Defaults to ``0.05``.
        name: Metric name.
        **kwargs: Passed to :class:`keras.metrics.Metric`.
    """

    def __init__(
        self,
        min_depth: float = 0.05,
        name: str = "abs_rel",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.min_depth = min_depth
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true_and_mask: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        sample_weight: Optional[keras.KerasTensor] = None,
    ) -> None:
        """Accumulate absolute relative error over valid pixels.

        Args:
            y_true_and_mask: Concatenated ``[depth, mask]`` with shape
                ``(B, H, W, 2)``.
            y_pred: Predicted depth ``(B, H, W, 1)`` in ``[-1, 1]``.
            sample_weight: Unused (masking is via the validity channel).
        """
        depth_true, mask = _unpack_depth_and_mask(y_true_and_mask)

        dt = _to_positive_range(depth_true)
        dp = _to_positive_range(y_pred)

        # Exclude near-zero ground truth to avoid division instability
        depth_valid = keras.ops.cast(dt > self.min_depth, mask.dtype) * mask
        dt_safe = keras.ops.maximum(dt, keras.ops.cast(self.min_depth, dt.dtype))

        rel_err = keras.ops.abs(dp - dt) / dt_safe * depth_valid
        self.total.assign_add(keras.ops.sum(rel_err))
        self.count.assign_add(keras.ops.sum(depth_valid))

    def result(self) -> keras.KerasTensor:
        """Return mean absolute relative error."""
        return self.total / keras.ops.maximum(self.count, 1.0)

    def reset_state(self) -> None:
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["min_depth"] = self.min_depth
        return config


# =====================================================================
# DeltaThresholdMetric
# =====================================================================


@register_dl_technique("dl_techniques.metrics.depth_metrics")
class DeltaThresholdMetric(keras.metrics.Metric):
    r"""Threshold accuracy for depth estimation.

    Fraction of valid pixels where the depth ratio is within a threshold:

    .. math::

        \delta_t = \frac{1}{|V|} \sum_{p \in V}
        \mathbb{1}\!\left[
            \max\!\left(\frac{d_p^{pred}}{d_p^{gt}},\;
                        \frac{d_p^{gt}}{d_p^{pred}}\right) < t
        \right]

    Common thresholds from Eigen et al.:
    :math:`\delta_1 = 1.25`,
    :math:`\delta_2 = 1.25^2`,
    :math:`\delta_3 = 1.25^3`.

    Args:
        threshold: Ratio threshold *t*.  Defaults to ``1.25``.
        name: Metric name.  Auto-generated from threshold if ``None``.
        **kwargs: Passed to :class:`keras.metrics.Metric`.
    """

    def __init__(
        self,
        threshold: float = 1.25,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        name = name or f"delta_{threshold:.2f}"
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true_and_mask: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        sample_weight: Optional[keras.KerasTensor] = None,
    ) -> None:
        """Accumulate threshold accuracy over valid pixels.

        Args:
            y_true_and_mask: Concatenated ``[depth, mask]`` with shape
                ``(B, H, W, 2)``.
            y_pred: Predicted depth ``(B, H, W, 1)`` in ``[-1, 1]``.
            sample_weight: Unused (masking is via the validity channel).
        """
        depth_true, mask = _unpack_depth_and_mask(y_true_and_mask)

        dt = _to_positive_range(depth_true)
        dp = _to_positive_range(y_pred)

        dt_safe = keras.ops.maximum(dt, keras.ops.cast(1e-6, dt.dtype))
        dp_safe = keras.ops.maximum(dp, keras.ops.cast(1e-6, dp.dtype))

        ratio = keras.ops.maximum(dp_safe / dt_safe, dt_safe / dp_safe)
        within = keras.ops.cast(ratio < self.threshold, mask.dtype) * mask

        self.correct.assign_add(keras.ops.sum(within))
        self.count.assign_add(keras.ops.sum(mask))

    def result(self) -> keras.KerasTensor:
        """Return fraction of pixels within the threshold."""
        return self.correct / keras.ops.maximum(self.count, 1.0)

    def reset_state(self) -> None:
        self.correct.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["threshold"] = self.threshold
        return config


# =====================================================================
# SqRelMetric
# =====================================================================


@register_dl_technique("dl_techniques.metrics.depth_metrics")
class SqRelMetric(keras.metrics.Metric):
    r"""Squared Relative Error on valid (masked) pixels.

    .. math::

        \text{SqRel} = \frac{1}{|V|} \sum_{p \in V}
        \frac{(d_p^{pred} - d_p^{gt})^2}{d_p^{gt}}

    Args:
        min_depth: Minimum valid depth in ``[0, 1]`` range (after shift).
        name: Metric name.
        **kwargs: Passed to :class:`keras.metrics.Metric`.
    """

    def __init__(
        self,
        min_depth: float = 0.05,
        name: str = "sq_rel",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.min_depth = min_depth
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true_and_mask: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        sample_weight: Optional[keras.KerasTensor] = None,
    ) -> None:
        """Accumulate squared relative error over valid pixels.

        Args:
            y_true_and_mask: Concatenated ``[depth, mask]`` ``(B, H, W, 2)``.
            y_pred: Predicted depth ``(B, H, W, 1)`` in ``[-1, 1]``.
            sample_weight: Unused.
        """
        depth_true, mask = _unpack_depth_and_mask(y_true_and_mask)

        dt = _to_positive_range(depth_true)
        dp = _to_positive_range(y_pred)

        depth_valid = keras.ops.cast(dt > self.min_depth, mask.dtype) * mask
        dt_safe = keras.ops.maximum(dt, keras.ops.cast(self.min_depth, dt.dtype))

        sq_rel = keras.ops.square(dp - dt) / dt_safe * depth_valid
        self.total.assign_add(keras.ops.sum(sq_rel))
        self.count.assign_add(keras.ops.sum(depth_valid))

    def result(self) -> keras.KerasTensor:
        return self.total / keras.ops.maximum(self.count, 1.0)

    def reset_state(self) -> None:
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["min_depth"] = self.min_depth
        return config


# =====================================================================
# RMSEMetric
# =====================================================================


@register_dl_technique("dl_techniques.metrics.depth_metrics")
class RMSEMetric(keras.metrics.Metric):
    r"""Root Mean Squared Error on valid (masked) pixels.

    .. math::

        \text{RMSE} = \sqrt{\frac{1}{|V|} \sum_{p \in V}
        (d_p^{pred} - d_p^{gt})^2}

    Args:
        name: Metric name.
        **kwargs: Passed to :class:`keras.metrics.Metric`.
    """

    def __init__(
        self,
        name: str = "rmse",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true_and_mask: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        sample_weight: Optional[keras.KerasTensor] = None,
    ) -> None:
        """Accumulate squared error over valid pixels.

        Args:
            y_true_and_mask: Concatenated ``[depth, mask]`` ``(B, H, W, 2)``.
            y_pred: Predicted depth ``(B, H, W, 1)`` in ``[-1, 1]``.
            sample_weight: Unused.
        """
        depth_true, mask = _unpack_depth_and_mask(y_true_and_mask)

        dt = _to_positive_range(depth_true)
        dp = _to_positive_range(y_pred)

        sq_err = keras.ops.square(dp - dt) * mask
        self.total.assign_add(keras.ops.sum(sq_err))
        self.count.assign_add(keras.ops.sum(mask))

    def result(self) -> keras.KerasTensor:
        return keras.ops.sqrt(self.total / keras.ops.maximum(self.count, 1.0))

    def reset_state(self) -> None:
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        return super().get_config()


# =====================================================================
# RMSELogMetric
# =====================================================================


@register_dl_technique("dl_techniques.metrics.depth_metrics")
class RMSELogMetric(keras.metrics.Metric):
    r"""Root Mean Squared Logarithmic Error on valid (masked) pixels.

    .. math::

        \text{RMSE}_{log} = \sqrt{\frac{1}{|V|} \sum_{p \in V}
        (\log d_p^{pred} - \log d_p^{gt})^2}

    Args:
        min_depth: Minimum depth clamp before log (avoids log(0)).
        name: Metric name.
        **kwargs: Passed to :class:`keras.metrics.Metric`.
    """

    def __init__(
        self,
        min_depth: float = 1e-6,
        name: str = "rmse_log",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.min_depth = min_depth
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true_and_mask: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        sample_weight: Optional[keras.KerasTensor] = None,
    ) -> None:
        """Accumulate squared log error over valid pixels.

        Args:
            y_true_and_mask: Concatenated ``[depth, mask]`` ``(B, H, W, 2)``.
            y_pred: Predicted depth ``(B, H, W, 1)`` in ``[-1, 1]``.
            sample_weight: Unused.
        """
        depth_true, mask = _unpack_depth_and_mask(y_true_and_mask)

        dt = _to_positive_range(depth_true)
        dp = _to_positive_range(y_pred)

        dt_safe = keras.ops.maximum(dt, keras.ops.cast(self.min_depth, dt.dtype))
        dp_safe = keras.ops.maximum(dp, keras.ops.cast(self.min_depth, dp.dtype))

        log_diff = keras.ops.log(dp_safe) - keras.ops.log(dt_safe)
        sq_log_err = keras.ops.square(log_diff) * mask

        self.total.assign_add(keras.ops.sum(sq_log_err))
        self.count.assign_add(keras.ops.sum(mask))

    def result(self) -> keras.KerasTensor:
        return keras.ops.sqrt(self.total / keras.ops.maximum(self.count, 1.0))

    def reset_state(self) -> None:
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["min_depth"] = self.min_depth
        return config
