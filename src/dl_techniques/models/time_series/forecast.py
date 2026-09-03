"""Unified forecast contract shared by every time-series model in this package.

This module holds :class:`Forecast`, a plain dataclass carrying the
materialized numpy arrays a single forecast call produces, and
:class:`ForecastMixin`, which gives a model a uniform
``predict_forecast(x) -> Forecast`` entry point over a model-specific
``_forecast`` hook. A point model returns ``quantiles=None`` rather than
fabricate an interval it never estimated; a probabilistic model populates
``quantiles`` of shape ``[B, H, F, Q]`` with matching ``quantile_levels``.
Callers branch on ``has_quantiles()`` instead of the concrete model type.

``Forecast`` is inert data, not a Keras layer: it is never serialized into a
``.keras`` file and never registered with Keras. ``Forecast.interval`` looks
up an exact quantile level; a caller wanting the closest available level
should use the model's own ``predict_quantiles`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------------


@dataclass
class Forecast:
    """Concrete predicted arrays from a single forecast call.

    Shapes use `B` batch size, `H` forecast horizon, `F` feature count,
    `Q` quantile-level count, `S` Monte-Carlo sample count.

    :ivar point: Point or median forecast, shape `[B, H, F]`. Required.
    :vartype point: np.ndarray
    :ivar quantiles: Quantile forecasts, shape `[B, H, F, Q]`, or `None` for point-only models. The last axis matches `quantile_levels`.
    :vartype quantiles: np.ndarray, optional
    :ivar quantile_levels: The `Q` quantile levels, e.g. `[0.1, 0.5, 0.9]`, matching the last axis of `quantiles`; `None` when there are no quantiles.
    :vartype quantile_levels: list[float], optional
    :ivar samples: Optional Monte-Carlo samples, shape `[S, B, H, F]`; `None` unless a model provides them.
    :vartype samples: np.ndarray, optional
    """

    point: np.ndarray
    quantiles: Optional[np.ndarray] = None
    quantile_levels: Optional[list[float]] = None
    samples: Optional[np.ndarray] = None

    def has_quantiles(self) -> bool:
        """Return whether this forecast carries quantile predictions.

        :return: True if both `quantiles` and `quantile_levels` are present.
        :rtype: bool
        """
        return self.quantiles is not None and self.quantile_levels is not None

    def interval(self, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
        """Extract a prediction interval from the stored quantiles.

        Looks up `low` and `high` exactly in `quantile_levels` and slices the
        matching planes out of `quantiles`. For the closest-available level
        instead of an exact match, use the model's `predict_quantiles`.

        :param low: The lower quantile level, must be present in `quantile_levels`.
        :type low: float
        :param high: The upper quantile level, must be present in `quantile_levels`.
        :type high: float
        :return: A `(lower, upper)` tuple of numpy arrays, each shaped `[B, H, F]`.
        :rtype: tuple[np.ndarray, np.ndarray]
        :raises ValueError: If this forecast has no quantiles, or if `low`/`high` are not present in `quantile_levels`.
        """
        if self.quantiles is None or self.quantile_levels is None:
            raise ValueError(
                "Forecast has no quantiles; cannot extract interval"
            )

        levels = self.quantile_levels
        missing = [lvl for lvl in (low, high) if lvl not in levels]
        if missing:
            raise ValueError(
                f"Requested quantile level(s) {missing} not in available "
                f"levels {levels}. interval() requires exact levels; use the "
                f"model's predict_quantiles for closest-quantile mapping."
            )

        low_idx = levels.index(low)
        high_idx = levels.index(high)

        lower = self.quantiles[..., low_idx]
        upper = self.quantiles[..., high_idx]
        return lower, upper

    def __repr__(self) -> str:
        """Concise representation summarizing array shapes (no array dumps)."""

        def _shape(arr: Optional[np.ndarray]) -> str:
            return "None" if arr is None else f"{tuple(arr.shape)}"

        return (
            f"Forecast(point={_shape(self.point)}, "
            f"quantiles={_shape(self.quantiles)}, "
            f"quantile_levels={self.quantile_levels}, "
            f"samples={_shape(self.samples)})"
        )


class ForecastMixin:
    """Mixin granting a model a uniform ``predict_forecast`` entry point.

    A plain mixin, not a Keras layer and not registered for serialization. It
    adds no instance state, so mixing it into a serializable model does not
    affect `get_config`/round-trip behavior.

    A subclass must implement :meth:`_forecast`, returning a :class:`Forecast`.
    The public :meth:`predict_forecast` is a thin validating wrapper around it.
    """

    def _forecast(self, x, **kwargs) -> Forecast:
        """Model-specific forecast hook; a subclass must implement this.

        :param x: Model input, context window or batch; type is model-specific.
        :param kwargs: Model-specific forecast options.
        :return: A :class:`Forecast` for `x`.
        :raises NotImplementedError: Always, in this base implementation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} mixes in ForecastMixin but does not "
            f"implement _forecast(self, x, **kwargs) -> Forecast. Implement it "
            f"to produce a Forecast (reuse the model's existing prediction path)."
        )

    def predict_forecast(self, x, **kwargs) -> Forecast:
        """Produce a validated :class:`Forecast` for `x`.

        Delegates to :meth:`_forecast` and checks the result is a `Forecast`
        with a non-`None` `point`. Does no batching or chunking of its own.

        :param x: Model input, context window or batch; type is model-specific.
        :param kwargs: Forwarded to :meth:`_forecast`.
        :return: The :class:`Forecast` returned by :meth:`_forecast`.
        :raises TypeError: If `_forecast` does not return a `Forecast`.
        :raises ValueError: If the returned forecast has a `None` `point`.
        """
        forecast = self._forecast(x, **kwargs)

        if not isinstance(forecast, Forecast):
            raise TypeError(
                f"{type(self).__name__}._forecast must return a Forecast, got "
                f"{type(forecast).__name__}."
            )
        if forecast.point is None:
            raise ValueError(
                f"{type(self).__name__}._forecast returned a Forecast with "
                f"point=None; the point/median forecast is required."
            )

        logger.debug("predict_forecast produced %r", forecast)
        return forecast
