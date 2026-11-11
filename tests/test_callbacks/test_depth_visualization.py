"""Depth visualization callbacks: the grid forward pass is capped, the curve keeps its epochs.

Plan plan-2026-11-11T131349-13b28b37 step 17.

* ``DepthPredictionGridCallback`` slices every cached array to ``max_samples``, so
  the model forward pass, the scale-and-shift alignment and the logged MSE run on
  the plotted samples and not on the whole validation set.
* ``DepthMetricsCurveCallback`` plots a metric at the epochs it was logged at (a
  validation metric logged every second epoch used to be drawn at x = 1, 2, 3),
  and starts each ``fit`` with an empty history.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from dl_techniques.callbacks.depth_visualization import (
    DepthMetricsCurveCallback,
    DepthPredictionGridCallback,
)


class _CountingModel:
    """Stands in for a depth model: records the batch size of every forward pass."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, x, training=False):
        self.batch_sizes.append(int(x.shape[0]))
        return np.asarray(x)[..., :1] * 0.5


def _grid(n: int, max_samples: int, out, display_rgb: bool = False):
    rng = np.random.default_rng(0)
    rgb = rng.uniform(-1, 1, (n, 8, 8, 3)).astype("float32")
    depth = rng.uniform(1, 5, (n, 8, 8, 1)).astype("float32")
    mask = np.ones((n, 8, 8, 1), dtype="float32")
    return DepthPredictionGridCallback(
        val_rgb=rgb, val_depth=depth, val_mask=mask, output_dir=str(out),
        frequency=1, max_samples=max_samples,
        display_rgb=rgb.copy() if display_rgb else None,
    )


class TestGridForwardPassIsCapped:

    # DECISION plan-2026-11-11T131349-13b28b37/D-020 guard: the forward pass sees max_samples
    def test_the_model_forward_batch_is_max_samples(self, tmp_path):
        cb = _grid(n=64, max_samples=2, out=tmp_path)
        model = _CountingModel()
        cb.set_model(model)
        cb.on_epoch_end(0, {})
        assert model.batch_sizes == [2], (
            f"the forward pass ran on batches {model.batch_sizes}, expected [2]"
        )
        assert (tmp_path / "epoch_001_depth.png").exists()

    def test_display_rgb_is_capped_too(self, tmp_path):
        cb = _grid(n=64, max_samples=3, out=tmp_path, display_rgb=True)
        assert cb.val_rgb.shape[0] == cb.val_depth.shape[0] == cb.val_mask.shape[0] == 3
        assert cb.display_rgb.shape[0] == 3

    def test_fewer_samples_than_max_samples_run_whole(self, tmp_path):
        cb = _grid(n=3, max_samples=8, out=tmp_path)
        model = _CountingModel()
        cb.set_model(model)
        cb.on_epoch_end(0, {})
        assert model.batch_sizes == [3]
        assert (tmp_path / "epoch_001_depth.png").exists()


@pytest.fixture
def plotted(monkeypatch):
    """Record ``(label, x, y)`` of every ``Axes.plot`` call, in call order."""
    from matplotlib.axes import Axes

    calls: list[tuple[str, list, list]] = []
    original = Axes.plot

    def recording(self, *args, **kwargs):
        calls.append((kwargs.get("label"), list(args[0]), list(args[1])))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", recording)
    return calls


def _last(calls, label):
    return [c for c in calls if c[0] == label][-1]


class TestCurveEpochs:

    # DECISION plan-2026-11-11T131349-13b28b37/D-020 guard: x is the epoch the value was logged at
    def test_a_metric_logged_every_second_epoch_plots_at_those_epochs(
        self, tmp_path, plotted
    ):
        cb = DepthMetricsCurveCallback(
            output_dir=str(tmp_path), train_metrics=["loss"], frequency=1
        )
        for epoch in range(6):
            logs = {"loss": 1.0 / (epoch + 1)}
            if epoch % 2 == 1:
                logs["val_loss"] = 2.0 / (epoch + 1)
            cb.on_epoch_end(epoch, logs)
        assert _last(plotted, "train")[1] == [1, 2, 3, 4, 5, 6]
        assert _last(plotted, "val")[1] == [2, 4, 6]
        assert _last(plotted, "val")[2] == pytest.approx([1.0, 0.5, 1 / 3])

    def test_a_second_run_starts_from_an_empty_history(self, tmp_path, plotted):
        cb = DepthMetricsCurveCallback(
            output_dir=str(tmp_path), train_metrics=["loss"], frequency=1
        )
        cb.on_train_begin()
        for epoch in range(3):
            cb.on_epoch_end(epoch, {"loss": 1.0, "val_loss": 1.0})
        cb.on_train_begin()
        plotted.clear()
        for epoch in range(2):
            cb.on_epoch_end(epoch, {"loss": 2.0, "val_loss": 2.0})
        assert _last(plotted, "train")[1:] == ([1, 2], [2.0, 2.0])
        assert _last(plotted, "val")[1:] == ([1, 2], [2.0, 2.0])
