"""Unit tests for the unified ``Forecast`` contract and ``ForecastMixin``.

Covers the inference-contract foundation from plan
``plan_2026-06-10_39646d39`` (D-001): the ``Forecast`` dataclass
(point/quantile shapes, ``has_quantiles``, ``interval`` happy path and error
paths, concise ``__repr__``) and the ``ForecastMixin.predict_forecast``
validating wrapper (delegation, missing-hook error, and result validation).

All tests are deterministic (fixed constants, no randomness).
"""

import re

import numpy as np
import pytest

from dl_techniques.models.time_series.forecast import Forecast, ForecastMixin


# ---------------------------------------------------------------------------
# Shape constants for synthetic forecasts (B, H, F, Q).
# ---------------------------------------------------------------------------

B, H, F, Q = 2, 4, 3, 3
QUANTILE_LEVELS = [0.1, 0.5, 0.9]


def _make_point() -> np.ndarray:
    """Return a deterministic point forecast of shape ``[B, H, F]``."""
    return np.full((B, H, F), 5.0, dtype=np.float32)


def _make_quantiles() -> np.ndarray:
    """Return quantiles ``[B, H, F, Q]`` where plane ``q`` is the constant ``q``.

    The last axis is filled so that ``quantiles[..., k] == QUANTILE_LEVELS[k]``
    everywhere, making interval slicing trivially checkable by value.
    """
    quantiles = np.empty((B, H, F, Q), dtype=np.float32)
    for k, level in enumerate(QUANTILE_LEVELS):
        quantiles[..., k] = level
    return quantiles


# ---------------------------------------------------------------------------
# Forecast: point-only (no quantiles).
# ---------------------------------------------------------------------------

class TestForecastPointOnly:
    """A point-only ``Forecast`` (``quantiles=None``)."""

    def test_has_quantiles_is_false(self):
        fc = Forecast(point=_make_point())
        assert fc.has_quantiles() is False

    def test_interval_raises_value_error(self):
        fc = Forecast(point=_make_point())
        with pytest.raises(ValueError, match="no quantiles"):
            fc.interval(0.1, 0.9)


# ---------------------------------------------------------------------------
# Forecast: with quantiles.
# ---------------------------------------------------------------------------

class TestForecastWithQuantiles:
    """A probabilistic ``Forecast`` carrying ``[B, H, F, Q]`` quantiles."""

    def _forecast(self) -> Forecast:
        return Forecast(
            point=_make_point(),
            quantiles=_make_quantiles(),
            quantile_levels=list(QUANTILE_LEVELS),
        )

    def test_has_quantiles_is_true(self):
        assert self._forecast().has_quantiles() is True

    def test_interval_returns_correct_shapes(self):
        lower, upper = self._forecast().interval(0.1, 0.9)
        assert lower.shape == (B, H, F)
        assert upper.shape == (B, H, F)

    def test_interval_returns_correct_values(self):
        # quantiles[..., k] == QUANTILE_LEVELS[k], so the 0.1 plane is all 0.1
        # and the 0.9 plane is all 0.9.
        lower, upper = self._forecast().interval(0.1, 0.9)
        np.testing.assert_allclose(lower, 0.1, atol=1e-7)
        np.testing.assert_allclose(upper, 0.9, atol=1e-7)

    def test_interval_picks_the_requested_levels(self):
        # Requesting the median for both bounds must return the 0.5 plane twice.
        lower, upper = self._forecast().interval(0.5, 0.5)
        np.testing.assert_allclose(lower, 0.5, atol=1e-7)
        np.testing.assert_allclose(upper, 0.5, atol=1e-7)

    def test_interval_unknown_level_raises_listing_available(self):
        fc = self._forecast()
        with pytest.raises(ValueError) as exc_info:
            fc.interval(0.05, 0.9)
        msg = str(exc_info.value)
        assert "0.05" in msg
        # Error must surface the available levels to the caller.
        assert "0.1" in msg and "0.9" in msg


# ---------------------------------------------------------------------------
# Forecast: __repr__ is concise (no full-array dumps).
# ---------------------------------------------------------------------------

class TestForecastRepr:
    """``__repr__`` summarizes shapes without dumping array contents."""

    def test_repr_mentions_class_and_shapes(self):
        fc = Forecast(
            point=_make_point(),
            quantiles=_make_quantiles(),
            quantile_levels=list(QUANTILE_LEVELS),
        )
        rep = repr(fc)
        assert "Forecast" in rep
        # Shapes are present...
        assert str((B, H, F)) in rep
        assert str((B, H, F, Q)) in rep

    def test_repr_is_short_and_omits_array_values(self):
        # Build an array large enough that a full dump would be very long.
        big_point = np.arange(B * H * F, dtype=np.float32).reshape(B, H, F)
        fc = Forecast(point=big_point)
        rep = repr(fc)
        # A real array dump would contain many newlines / element separators;
        # the concise repr fits on a single short line.
        assert "\n" not in rep
        assert len(rep) < 200
        # No interior element value such as the unique "23.0" should leak.
        assert not re.search(r"\b23\.?0*\b", rep)


# ---------------------------------------------------------------------------
# ForecastMixin stubs.
# ---------------------------------------------------------------------------

class _GoodStub(ForecastMixin):
    """Stub implementing ``_forecast`` to return a fixed ``Forecast``."""

    def __init__(self, forecast: Forecast):
        self._fixed = forecast

    def _forecast(self, x, **kwargs) -> Forecast:
        return self._fixed


class _MissingHookStub(ForecastMixin):
    """Stub that does NOT override ``_forecast`` (uses the base raiser)."""


class _NonForecastStub(ForecastMixin):
    """Stub whose ``_forecast`` returns the wrong type."""

    def _forecast(self, x, **kwargs):
        return {"point": _make_point()}


class _NonePointStub(ForecastMixin):
    """Stub whose ``_forecast`` returns a ``Forecast`` with ``point=None``."""

    def _forecast(self, x, **kwargs) -> Forecast:
        return Forecast(point=None)


class TestForecastMixin:
    """``ForecastMixin.predict_forecast`` delegation and validation."""

    def test_predict_forecast_returns_hook_result(self):
        expected = Forecast(
            point=_make_point(),
            quantiles=_make_quantiles(),
            quantile_levels=list(QUANTILE_LEVELS),
        )
        stub = _GoodStub(expected)
        result = stub.predict_forecast(np.zeros((B, H, F), dtype=np.float32))
        assert result is expected
        assert isinstance(result, Forecast)
        assert result.has_quantiles() is True

    def test_missing_hook_raises_not_implemented(self):
        stub = _MissingHookStub()
        with pytest.raises(NotImplementedError, match="_forecast"):
            stub.predict_forecast(np.zeros((B, H, F), dtype=np.float32))

    def test_non_forecast_return_raises_type_error(self):
        stub = _NonForecastStub()
        with pytest.raises(TypeError, match="must return a Forecast"):
            stub.predict_forecast(np.zeros((B, H, F), dtype=np.float32))

    def test_none_point_raises_value_error(self):
        stub = _NonePointStub()
        with pytest.raises(ValueError, match="point=None"):
            stub.predict_forecast(np.zeros((B, H, F), dtype=np.float32))
