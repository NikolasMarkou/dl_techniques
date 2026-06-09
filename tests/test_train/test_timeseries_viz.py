"""Regression tests for the TS viz-data path (plan_2026-06-09_49c73926, Step 1).

Covers the previously-uncovered code that crashed nbeats on its default config:

1. ``_prepare_viz_data_from_processor`` must not raise on *nested* (ragged tuple)
   targets — it keeps the primary (forecast) element so ``np.array`` never stacks
   a ragged tuple. It must remain a strict no-op for single-array targets, and
   return two empty arrays for an empty generator.
2. ``TimeSeriesPerformanceCallback.__init__`` must NEVER let a viz-prep failure
   propagate out — a raising ``_prepare_viz_data`` degrades to empty viz data so
   ``model.fit`` is never aborted by a visualization helper.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from types import SimpleNamespace

import numpy as np
import pytest

from train.common.timeseries import (
    _prepare_viz_data_from_processor,
    TimeSeriesPerformanceCallback,
)

FORECAST_LEN = 24
BACKCAST_LEN = 168
CONTEXT_LEN = 168


class _FakeProcessor:
    """Minimal stand-in exposing only ``_test_generator_raw()``.

    ``mode`` selects the target shape the generator yields:
    - ``"tuple"``   -> ragged reconstruction target ``(forecast[F,1], backcast_flat[B])``
                       (this is exactly what crashed nbeats at ``np.array(viz_y)``).
    - ``"single"``  -> single forecast array ``[F, 1]`` (tirex/prism-style).
    - ``"empty"``   -> yields nothing.
    """

    def __init__(self, mode: str, n: int = 5):
        self.mode = mode
        self.n = n

    def _test_generator_raw(self):
        if self.mode == "empty":
            return
        for i in range(self.n):
            x = np.full((CONTEXT_LEN, 1), float(i), dtype=np.float32)
            if self.mode == "tuple":
                forecast = np.full((FORECAST_LEN, 1), float(i), dtype=np.float32)
                backcast_flat = np.full((BACKCAST_LEN,), float(i), dtype=np.float32)
                yield x, (forecast, backcast_flat)
            else:  # single
                yield x, np.full((FORECAST_LEN, 1), float(i), dtype=np.float32)


def test_ragged_tuple_targets_do_not_crash_and_keep_forecast():
    """Reconstruction-mode ragged tuples must stack cleanly as forecast-only."""
    k = 3
    viz_x, viz_y = _prepare_viz_data_from_processor(_FakeProcessor("tuple", n=5), k)
    # No exception, exactly k windows collected, homogeneous arrays.
    assert viz_x.shape == (k, CONTEXT_LEN, 1)
    assert viz_y.shape == (k, FORECAST_LEN, 1)          # forecast only (backcast dropped)
    assert viz_y.dtype != object                         # must NOT be an object array
    # Consumer contract: test_y[i].flatten() must work (the original crash's symptom).
    assert viz_y[0].flatten().shape == (FORECAST_LEN,)


def test_single_array_targets_are_pass_through_identity():
    """Single-array targets (tirex/prism) must be unchanged by the guard."""
    k = 4
    proc = _FakeProcessor("single", n=10)
    viz_x, viz_y = _prepare_viz_data_from_processor(proc, k)
    assert viz_x.shape == (k, CONTEXT_LEN, 1)
    assert viz_y.shape == (k, FORECAST_LEN, 1)
    # Identity check vs a hand-stacked reference of the same first k samples.
    ref = np.stack([np.full((FORECAST_LEN, 1), float(i), dtype=np.float32) for i in range(k)])
    np.testing.assert_array_equal(viz_y, ref)


def test_empty_generator_returns_two_empty_arrays():
    viz_x, viz_y = _prepare_viz_data_from_processor(_FakeProcessor("empty"), 3)
    assert viz_x.size == 0 and viz_y.size == 0


def test_k_caps_the_number_of_windows():
    viz_x, viz_y = _prepare_viz_data_from_processor(_FakeProcessor("single", n=10), 2)
    assert viz_x.shape[0] == 2 and viz_y.shape[0] == 2


class _RaisingCallback(TimeSeriesPerformanceCallback):
    """A callback whose viz-prep blows up — must be swallowed by __init__."""

    def _prepare_viz_data(self):
        raise ValueError("synthetic viz failure (e.g. ragged shape)")

    def _plot_predictions(self, epoch: int) -> None:  # pragma: no cover - never called here
        raise NotImplementedError


def test_viz_prep_failure_is_non_fatal(tmp_path):
    """A raising _prepare_viz_data must NOT escape callback construction."""
    config = SimpleNamespace(visualize_every_n_epochs=1)
    # Must not raise:
    cb = _RaisingCallback(config, str(tmp_path / "viz"), model_name="raises")
    # Degraded to empty viz data; training would simply skip prediction plots.
    vx, vy = cb.viz_test_data
    assert isinstance(vx, np.ndarray) and isinstance(vy, np.ndarray)
    assert vx.size == 0 and vy.size == 0


def test_healthy_viz_prep_is_preserved(tmp_path):
    """When _prepare_viz_data succeeds, its result is stored unchanged."""

    class _OkCallback(TimeSeriesPerformanceCallback):
        def _prepare_viz_data(self):
            return np.zeros((2, CONTEXT_LEN, 1), np.float32), np.zeros((2, FORECAST_LEN, 1), np.float32)

        def _plot_predictions(self, epoch: int) -> None:  # pragma: no cover
            raise NotImplementedError

    config = SimpleNamespace(visualize_every_n_epochs=1)
    cb = _OkCallback(config, str(tmp_path / "viz"), model_name="ok")
    vx, vy = cb.viz_test_data
    assert vx.shape == (2, CONTEXT_LEN, 1) and vy.shape == (2, FORECAST_LEN, 1)
