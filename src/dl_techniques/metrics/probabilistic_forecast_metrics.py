"""Probabilistic forecast metrics: interval coverage and sharpness.

This module provides two Keras metrics for evaluating probabilistic
(quantile) forecasts:

- :class:`CoverageMetric` — empirical interval coverage, i.e. the fraction of
  realized targets that fall within the central prediction interval
  ``[lower, upper]``.
- :class:`SharpnessMetric` — mean interval width, i.e. the average of
  ``(upper - lower)`` across all predictions.

Both metrics consume quantile predictions. The canonical model output for a
quantile forecaster is ``[B, H, F, Q]`` (batch, horizon, feature, quantile)
or ``[B, H, Q]`` when there is no explicit feature axis. A Keras
``Metric.update_state(y_true, y_pred, sample_weight=None)`` receives the full
quantile tensor as ``y_pred``; the metric is configured at construction time
with ``low_index`` / ``high_index`` selecting the two quantile levels along the
LAST axis that bound the central interval. For example, with quantile levels
``[0.1, 0.25, 0.5, 0.75, 0.9]`` a 80% central interval uses
``low_index=0, high_index=4``.

These metrics are also used post-hoc (by passing realized targets and quantile
arrays directly), so ``update_state`` / ``result`` are kept simple, correct,
and fully serializable. They mirror the ``SMAPE`` conventions in
``time_series_metrics.py`` (``add_weight`` totals/count, ``keras.ops``,
``ops.divide_no_nan``, bare ``@keras.saving.register_keras_serializable()``).
"""

import keras
from keras import ops
from typing import Any, Dict, Optional

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Probabilistic forecast metrics
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CoverageMetric(keras.metrics.Metric):
    """Empirical interval coverage for quantile forecasts.

    Computes the fraction of realized targets that fall within the central
    prediction interval ``[lower, upper]``, where ``lower`` and ``upper`` are
    quantile slices taken from the last axis of the prediction tensor at the
    configured ``low_index`` and ``high_index``.

    For a well-calibrated model, the empirical coverage should approximate the
    nominal interval level (e.g. ~0.8 for an 80% interval spanning the 0.1 and
    0.9 quantiles).

    Args:
        low_index: Index along the last axis of ``y_pred`` selecting the lower
            interval bound (the low quantile).
        high_index: Index along the last axis of ``y_pred`` selecting the upper
            interval bound (the high quantile).
        name: Name of the metric instance.
        **kwargs: Additional keyword arguments passed to the parent Metric class.
    """

    def __init__(
        self,
        low_index: int,
        high_index: int,
        name: str = "coverage",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.low_index = low_index
        self.high_index = high_index
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        sample_weight: Optional[keras.KerasTensor] = None,
    ) -> None:
        """Update the metric state with new predictions.

        Args:
            y_true: Realized targets, shape ``[B, H, F]`` or ``[B, H]``.
            y_pred: Quantile predictions, shape ``[B, H, F, Q]`` or ``[B, H, Q]``.
            sample_weight: Optional per-element weighting, broadcastable to the
                ``[B, H, F]`` / ``[B, H]`` shape of the comparison.
        """
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")

        lower = y_pred[..., self.low_index]
        upper = y_pred[..., self.high_index]

        inside = ops.logical_and(
            ops.greater_equal(y_true, lower),
            ops.less_equal(y_true, upper),
        )
        values = ops.cast(inside, "float32")

        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, "float32")
            values = values * sample_weight
            self.total.assign_add(ops.sum(values))
            self.count.assign_add(ops.sum(sample_weight))
        else:
            self.total.assign_add(ops.sum(values))
            self.count.assign_add(ops.cast(ops.size(values), "float32"))

    def result(self) -> keras.KerasTensor:
        """Compute the current coverage value.

        Returns:
            The empirical coverage fraction in ``[0, 1]``.
        """
        return ops.divide_no_nan(self.total, self.count)

    def reset_state(self) -> None:
        """Reset the metric state."""
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization."""
        config = super().get_config()
        config.update(
            {
                "low_index": self.low_index,
                "high_index": self.high_index,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class SharpnessMetric(keras.metrics.Metric):
    """Mean interval width (sharpness) for quantile forecasts.

    Computes the average of ``(upper - lower)`` across all predictions, where
    ``lower`` and ``upper`` are quantile slices taken from the last axis of the
    prediction tensor at the configured ``low_index`` and ``high_index``.

    Sharpness measures how tight the prediction intervals are. It is only
    meaningful jointly with coverage: a sharp (narrow) interval is desirable
    only if coverage remains close to the nominal level.

    Args:
        low_index: Index along the last axis of ``y_pred`` selecting the lower
            interval bound (the low quantile).
        high_index: Index along the last axis of ``y_pred`` selecting the upper
            interval bound (the high quantile).
        name: Name of the metric instance.
        **kwargs: Additional keyword arguments passed to the parent Metric class.
    """

    def __init__(
        self,
        low_index: int,
        high_index: int,
        name: str = "sharpness",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.low_index = low_index
        self.high_index = high_index
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        sample_weight: Optional[keras.KerasTensor] = None,
    ) -> None:
        """Update the metric state with new predictions.

        Args:
            y_true: Realized targets, shape ``[B, H, F]`` or ``[B, H]``. Unused
                for the width computation but accepted to satisfy the standard
                Keras ``Metric`` signature.
            y_pred: Quantile predictions, shape ``[B, H, F, Q]`` or ``[B, H, Q]``.
            sample_weight: Optional per-element weighting, broadcastable to the
                ``[B, H, F]`` / ``[B, H]`` shape of the interval widths.
        """
        del y_true  # Not needed for interval width; signature kept for Keras.
        y_pred = ops.cast(y_pred, "float32")

        lower = y_pred[..., self.low_index]
        upper = y_pred[..., self.high_index]
        values = upper - lower

        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, "float32")
            values = values * sample_weight
            self.total.assign_add(ops.sum(values))
            self.count.assign_add(ops.sum(sample_weight))
        else:
            self.total.assign_add(ops.sum(values))
            self.count.assign_add(ops.cast(ops.size(values), "float32"))

    def result(self) -> keras.KerasTensor:
        """Compute the current sharpness value.

        Returns:
            The mean interval width.
        """
        return ops.divide_no_nan(self.total, self.count)

    def reset_state(self) -> None:
        """Reset the metric state."""
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization."""
        config = super().get_config()
        config.update(
            {
                "low_index": self.low_index,
                "high_index": self.high_index,
            }
        )
        return config
