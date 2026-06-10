"""Unified forecast contract for time-series models.

This module defines the shared inference contract that lets any time-series
forecaster be consumed uniformly regardless of its internal output paradigm
(point, quantile-tensor, parametric, or mixture). It is the foundation layer
of the contract decided in plan ``plan_2026-06-10_39646d39`` (decision D-001).

The contract has two pieces:

``Forecast``
    A plain ``@dataclass`` holding the *concrete* predicted arrays of a single
    forecast call. It is inert data, NOT a Keras layer: it carries numpy arrays
    (already-materialized predictions, not symbolic tensors) and is therefore
    never serialized into a ``.keras`` file and never registered with Keras.

``ForecastMixin``
    A light mixin that gives a model a uniform ``predict_forecast(x) -> Forecast``
    entry point delegating to a model-specific ``_forecast`` hook.

**Point vs probabilistic rationale.** Point models (e.g. plain regressors)
expose ``quantiles=None`` and MUST NOT fabricate intervals; probabilistic
models (e.g. tirex, prism-quantile) populate ``quantiles`` of shape
``[B, H, F, Q]`` alongside the matching ``quantile_levels``. Callers test
``has_quantiles()`` (or catch the ``ValueError`` from ``interval``) instead of
branching on the concrete model type. This keeps a single calling convention
across all four output paradigms while never inventing uncertainty a point
model did not produce.

**Scope note.** ``Forecast.interval`` performs an EXACT level lookup — it is for
callers who know the trained ``quantile_levels``. The closest-quantile fuzzy
mapping deliberately lives in each model's ``predict_quantiles`` and is NOT
duplicated here (D-001 "thin contract").
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

    A ``Forecast`` is inert data (a plain dataclass, not a Keras layer). It holds
    already-materialized numpy predictions and is never serialized into a
    ``.keras`` model file.

    Shape conventions:
        - ``B`` batch size, ``H`` forecast horizon, ``F`` feature/channel count,
          ``Q`` number of quantile levels, ``S`` Monte-Carlo sample count.

    Attributes:
        point: Point/median forecast, shape ``[B, H, F]``. Required.
        quantiles: Quantile forecasts, shape ``[B, H, F, Q]``, or ``None`` for
            point-only models. The last axis is ordered to match
            ``quantile_levels``.
        quantile_levels: The ``Q`` quantile levels (e.g. ``[0.1, 0.5, 0.9]``)
            matching the last axis of ``quantiles``; ``None`` when there are no
            quantiles.
        samples: Optional Monte-Carlo samples, shape ``[S, B, H, F]``; ``None``
            unless a model explicitly provides them.
    """

    point: np.ndarray
    quantiles: Optional[np.ndarray] = None
    quantile_levels: Optional[list[float]] = None
    samples: Optional[np.ndarray] = None

    def has_quantiles(self) -> bool:
        """Return whether this forecast carries quantile predictions.

        Returns:
            ``True`` iff both ``quantiles`` and ``quantile_levels`` are present.
        """
        return self.quantiles is not None and self.quantile_levels is not None

    def interval(self, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
        """Extract a prediction interval from the stored quantiles.

        Performs an EXACT lookup of ``low`` and ``high`` in ``quantile_levels``
        and slices the corresponding planes out of ``quantiles``. This method is
        for callers who know the trained levels; the closest-quantile fuzzy
        mapping lives in each model's ``predict_quantiles`` and is intentionally
        NOT duplicated here.

        Args:
            low: The lower quantile level (must be present in ``quantile_levels``).
            high: The upper quantile level (must be present in ``quantile_levels``).

        Returns:
            A ``(lower, upper)`` tuple of numpy arrays, each of shape ``[B, H, F]``.

        Raises:
            ValueError: If this forecast has no quantiles, or if ``low``/``high``
                are not present in ``quantile_levels``.
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

    This is a plain mixin (NOT a Keras layer and NOT registered for
    serialization). It adds no instance state, so mixing it into a serializable
    model does not affect ``get_config``/round-trip behavior.

    Subclasses MUST implement :meth:`_forecast`, returning a :class:`Forecast`.
    The public :meth:`predict_forecast` is a thin validating wrapper around it.
    """

    def _forecast(self, x, **kwargs) -> Forecast:
        """Model-specific forecast hook (MUST be implemented by subclasses).

        Args:
            x: Model input (context window / batch); type is model-specific.
            **kwargs: Model-specific forecast options.

        Returns:
            A :class:`Forecast` for ``x``.

        Raises:
            NotImplementedError: Always, in this base implementation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} mixes in ForecastMixin but does not "
            f"implement _forecast(self, x, **kwargs) -> Forecast. Implement it "
            f"to produce a Forecast (reuse the model's existing prediction path)."
        )

    def predict_forecast(self, x, **kwargs) -> Forecast:
        """Produce a validated :class:`Forecast` for ``x``.

        Delegates to :meth:`_forecast` and validates the result is a
        ``Forecast`` with a non-``None`` ``point``. Intentionally thin: no
        batching/chunking logic in this iteration.

        Args:
            x: Model input (context window / batch); type is model-specific.
            **kwargs: Forwarded to :meth:`_forecast`.

        Returns:
            The :class:`Forecast` returned by :meth:`_forecast`.

        Raises:
            TypeError: If ``_forecast`` does not return a :class:`Forecast`.
            ValueError: If the returned forecast has a ``None`` ``point``.
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
