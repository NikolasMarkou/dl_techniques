import keras
from keras import ops
import numpy as np
from typing import Any, Dict, List, Optional
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.metrics.time_series_metrics")
class SMAPE(keras.metrics.Metric):
    """Symmetric Mean Absolute Percentage Error metric.

    Computes sMAPE as: 200 * mean(|y_true - y_pred| / (|y_true| + |y_pred| + epsilon))

    This metric is bounded between 0 and 200, and is symmetric with respect to
    over-predictions and under-predictions.

    Args:
        name: Name of the metric instance.
        **kwargs: Additional keyword arguments passed to the parent Metric class.
    """

    def __init__(self, name: str = "smape", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
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
            y_true: Ground truth values.
            y_pred: Predicted values.
            sample_weight: Optional weighting of each sample.
        """
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")

        numerator = ops.abs(y_true - y_pred)
        denominator = ops.abs(y_true) + ops.abs(y_pred) + 1e-7

        values = 200.0 * (numerator / denominator)

        if sample_weight is not None:
            # The count must be weighted like the values it normalises. It was
            # not: `count` took `ops.size(y_true)` while `total` took the
            # WEIGHTED sum, so any weight below 1 pulled the reported sMAPE
            # toward zero -- a down-weighted row still counted as a full
            # observation in the denominator. `CoverageMetric` and
            # `SharpnessMetric` in `probabilistic_forecast_metrics.py` always
            # did this correctly; this metric now matches them. Unweighted
            # behaviour (the common case) is unchanged: the broadcast weight
            # sums to exactly `size(y_true)`.
            sample_weight = ops.cast(sample_weight, "float32")
            sample_weight = ops.broadcast_to(sample_weight, ops.shape(values))
            values = values * sample_weight
            self.count.assign_add(ops.sum(sample_weight))
        else:
            self.count.assign_add(ops.cast(ops.size(y_true), "float32"))

        self.total.assign_add(ops.sum(values))

    def result(self) -> keras.KerasTensor:
        """Compute the current metric value.

        Returns:
            The sMAPE value.
        """
        return ops.divide_no_nan(self.total, self.count)

    def reset_state(self) -> None:
        """Reset the metric state."""
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization."""
        return super().get_config()


@register_dl_technique("dl_techniques.metrics.time_series_metrics")
class PerHorizonError(keras.metrics.Metric):
    """Forecast error at ONE horizon step, for building an h = 1..H profile.

    Every other metric in this module reduces the horizon away, which is exactly
    the axis a multistep loss reweights. Without a per-step breakdown there is no
    way to see what changing the loss did: `MSEh` improving step 12 while
    destroying steps 1-11 reports the same aggregate MAE as a uniform
    improvement.

    This is a SCALAR metric for a single step rather than a vector-valued metric
    over all steps, because Keras' training loop logs metric results as scalars.
    Use :func:`per_horizon_metrics` to build the whole profile at once, or
    :func:`horizon_profile` for pool-level numpy evaluation.

    Args:
        step: 1-indexed horizon step to score. ``step=1`` is the one-step-ahead
            error; ``step=H`` is the far end of the horizon.
        error: ``"mae"`` (default) or ``"mse"``.
        name: Metric name. Defaults to ``"{error}_h{step}"``.
        **kwargs: Passed to ``keras.metrics.Metric``.

    Raises:
        ValueError: If ``step`` is not a positive integer or ``error`` is not
            one of the two supported names.
    """

    def __init__(
        self,
        step: int,
        error: str = "mae",
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(step, int) or isinstance(step, bool) or step < 1:
            raise ValueError(f"step must be a positive (1-indexed) integer, got {step!r}.")
        if error not in ("mae", "mse"):
            raise ValueError(f"error must be 'mae' or 'mse', got {error!r}.")

        super().__init__(name=name or f"{error}_h{step}", **kwargs)
        self.step = step
        self.error = error
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        sample_weight: Optional[keras.KerasTensor] = None,
    ) -> None:
        """Accumulate the error at this metric's horizon step.

        Args:
            y_true: Ground truth, ``(batch, horizon)`` or
                ``(batch, horizon, features)``. Axis 1 is the horizon.
            y_pred: Predictions, same shape.
            sample_weight: Optional per-sample weighting.

        Raises:
            ValueError: If the inputs have no horizon axis.
        """
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")

        if len(y_pred.shape) < 2:
            raise ValueError(
                f"PerHorizonError needs a horizon axis; got shape {y_pred.shape}."
            )

        difference = y_true[:, self.step - 1] - y_pred[:, self.step - 1]
        values = ops.abs(difference) if self.error == "mae" else ops.square(difference)

        if sample_weight is not None:
            weight = ops.cast(sample_weight, "float32")
            weight = ops.broadcast_to(ops.reshape(weight, (-1,) + (1,) * (len(values.shape) - 1)), ops.shape(values))
            values = values * weight
            self.count.assign_add(ops.sum(weight))
        else:
            self.count.assign_add(ops.cast(ops.size(values), "float32"))

        self.total.assign_add(ops.sum(values))

    def result(self) -> keras.KerasTensor:
        """Return the accumulated error at this horizon step."""
        return ops.divide_no_nan(self.total, self.count)

    def reset_state(self) -> None:
        """Reset the accumulators."""
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization."""
        config = super().get_config()
        config.update({"step": self.step, "error": self.error})
        return config


def per_horizon_metrics(horizon: int, error: str = "mae") -> List[PerHorizonError]:
    """Build one :class:`PerHorizonError` per step, for a whole h = 1..H profile.

    Args:
        horizon: Number of steps, ``H``.
        error: ``"mae"`` or ``"mse"``.

    Returns:
        A list of ``horizon`` metrics named ``{error}_h1 .. {error}_hH``, ready
        to hand to ``model.compile(metrics=...)``.

    Raises:
        ValueError: If ``horizon`` is not a positive integer.

    Example:
        >>> model.compile(optimizer="adam", loss="mse",
        ...               metrics=per_horizon_metrics(12))
    """
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}.")
    return [PerHorizonError(step=h, error=error) for h in range(1, horizon + 1)]


def horizon_profile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    error: str = "mae",
) -> np.ndarray:
    """Pool-level per-step error profile, in numpy.

    The companion to :func:`calculate_comprehensive_metrics`, which reduces the
    horizon away. This keeps it.

    Args:
        y_true: Ground truth, ``(batch, horizon)`` or
            ``(batch, horizon, features)``.
        y_pred: Predictions, same shape.
        error: ``"mae"`` or ``"mse"``.

    Returns:
        Array of shape ``(horizon,)``: the mean error at each step.

    Raises:
        ValueError: If ``error`` is unknown or the inputs have no horizon axis.
    """
    if error not in ("mae", "mse"):
        raise ValueError(f"error must be 'mae' or 'mse', got {error!r}.")
    if y_true.ndim < 2:
        raise ValueError(f"horizon_profile needs a horizon axis; got {y_true.shape}.")

    difference = np.asarray(y_true) - np.asarray(y_pred)
    values = np.abs(difference) if error == "mae" else difference ** 2
    axes = (0,) + tuple(range(2, values.ndim))
    return values.mean(axis=axes)


# ---------------------------------------------------------------------
# Metric Calculation Utilities
# ---------------------------------------------------------------------


def calculate_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    backcast: np.ndarray,
) -> Dict[str, float]:
    """Calculate comprehensive time series forecasting metrics.

    Computes MAE, RMSE, sMAPE, rMAE, and MASE for forecast evaluation.
    The rMAE and MASE metrics require historical data (backcast) to establish
    baseline comparisons.

    Args:
        y_true: Ground truth values with shape (batch, forecast_len, features).
        y_pred: Predicted values with shape (batch, forecast_len, features).
        backcast: Historical values with shape (batch, backcast_len, features).
            Used to compute naive baseline and scaling factor.

    Returns:
        Dictionary containing:
        - MAE: Mean Absolute Error
        - RMSE: Root Mean Squared Error
        - sMAPE: Symmetric Mean Absolute Percentage Error (0-200 scale)
        - rMAE: Relative MAE compared to naive last-value baseline
        - MASE: Mean Absolute Scaled Error using random walk scaling
    """
    epsilon: float = 1e-7

    # MAE: Mean Absolute Error
    mae: float = np.mean(np.abs(y_true - y_pred))

    # RMSE: Root Mean Squared Error
    rmse: float = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # sMAPE: Symmetric Mean Absolute Percentage Error
    denominator: np.ndarray = np.abs(y_true) + np.abs(y_pred) + epsilon
    smape: float = 200.0 * np.mean(np.abs(y_true - y_pred) / denominator)

    # Naive baseline: repeat last backcast value for all forecast steps
    last_val: np.ndarray = backcast[:, -1:, :]  # Shape: (batch, 1, features)
    naive_forecast: np.ndarray = np.tile(last_val, (1, y_true.shape[1], 1))
    mae_naive: float = np.mean(np.abs(y_true - naive_forecast))

    # rMAE: Relative MAE (normalized by naive forecast)
    rmae: float = mae / (mae_naive + epsilon)

    # MASE: Mean Absolute Scaled Error
    # Scale is the MAE of first-order differences in the backcast (random walk baseline)
    backcast_diff: np.ndarray = np.abs(backcast[:, 1:, :] - backcast[:, :-1, :])
    scale: float = np.mean(backcast_diff) + epsilon
    mase: float = mae / scale

    return {
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE": smape,
        "rMAE": rmae,
        "MASE": mase,
    }
