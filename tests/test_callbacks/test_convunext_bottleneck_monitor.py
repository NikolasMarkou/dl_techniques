"""Unit tests for :class:`ConvUnextBottleneckMonitorCallback`.

Covers:
- Two-grid contract (``_first`` + ``_energy`` PNGs) on a monitored epoch.
- FIXED absolute color scale: every ``imshow`` call receives the configured
  ``vmin``/``vmax`` (the fix for "colors shift between iterations").
- Channel selection: first grid = indices ``0..k-1``; energy grid = top-k by
  mean-square energy (recomputed each epoch).
- Cumulative ``bottleneck_health.png`` + history accumulation, with the
  ``'final'`` re-emit de-duplicated (monotone epoch x-axis).
- Frequency gating: nothing written on a non-monitored epoch.
- A failing forward pass is swallowed and never propagates into training.

No real ConvUNeXt is built: the callback only needs ``full_model(batch,
training=False)`` to return a list/tuple whose LAST element is the bottleneck
latent ``(B, h, w, C)``. A tiny callable stub keeps these tests CPU-only and
fast.
"""

from __future__ import annotations

import os

# Ensure matplotlib uses a non-interactive backend before any import.
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import numpy as np
import pytest

from dl_techniques.callbacks.convunext_bottleneck_monitor import (
    ConvUnextBottleneckMonitorCallback,
)


# ---------------------------------------------------------------------
# Fixtures / stubs
# ---------------------------------------------------------------------
class _FakeModel:
    """Callable stub returning ``[denoised, bottleneck]``.

    The bottleneck has energy increasing with channel index, so the top-energy
    channel set is the REVERSED tail (distinct from the first-k indices).
    """

    def __init__(self, b=2, h=4, w=4, c=128):
        # Channel i is the constant field ``i + 1`` so its mean-square energy is
        # exactly ``(i + 1)^2`` -> strictly increasing with index, giving an
        # unambiguous top-k ranking (no random ties to perturb the order).
        scale = (np.arange(c, dtype="float32") + 1.0)
        self._bottleneck = np.broadcast_to(
            scale[None, None, None, :], (b, h, w, c)
        ).astype("float32")
        self._denoised = np.zeros((b, h, w, 3), dtype="float32")

    def __call__(self, batch, training=False):
        return [self._denoised, self._bottleneck]


class _RaisingModel:
    def __call__(self, batch, training=False):
        raise RuntimeError("forward boom")


@pytest.fixture
def val_batch():
    return np.zeros((2, 4, 4, 3), dtype="float32")


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
class TestConvUnextBottleneckMonitor:
    def test_two_grids_and_health_written(self, tmp_path: Path, val_batch):
        cb = ConvUnextBottleneckMonitorCallback(
            full_model=_FakeModel(c=128),
            val_batch=val_batch,
            output_dir=tmp_path,
            monitor_freq=5,
        )
        cb.on_epoch_end(4)  # epoch+1 == 5 -> monitored

        assert (tmp_path / "epoch_0005_bottleneck_first.png").exists()
        assert (tmp_path / "epoch_0005_bottleneck_energy.png").exists()
        assert (tmp_path / "bottleneck_health.png").exists()

    def test_fixed_color_scale_passed_to_imshow(self, tmp_path: Path, val_batch, monkeypatch):
        from matplotlib.axes import Axes

        calls = []
        orig = Axes.imshow

        def spy(self, *args, **kwargs):
            calls.append(kwargs)
            return orig(self, *args, **kwargs)

        monkeypatch.setattr(Axes, "imshow", spy)

        cb = ConvUnextBottleneckMonitorCallback(
            full_model=_FakeModel(c=128),
            val_batch=val_batch,
            output_dir=tmp_path,
            monitor_freq=5,
            featuremap_vmin=-2.5,
            featuremap_vmax=2.5,
        )
        cb.on_epoch_end(4)

        assert len(calls) > 0, "imshow was never called"
        # Every tile must use the SAME fixed absolute scale (no autoscale).
        for k in calls:
            assert k.get("vmin") == -2.5
            assert k.get("vmax") == 2.5

    def test_channel_selection_first_and_energy(self, tmp_path: Path, val_batch, monkeypatch):
        captured = {}

        def fake_save_grid(self, sample, indices, tag, suffix, title):
            captured[suffix] = list(int(i) for i in indices)

        monkeypatch.setattr(
            ConvUnextBottleneckMonitorCallback, "_save_grid", fake_save_grid
        )

        model = _FakeModel(c=128)
        cb = ConvUnextBottleneckMonitorCallback(
            full_model=model,
            val_batch=val_batch,
            output_dir=tmp_path,
            monitor_freq=5,
            max_featuremap_channels=64,
        )
        cb.on_epoch_end(4)

        # First grid: fixed indices 0..63.
        assert captured["first"] == list(range(64))
        # Energy grid: top-64 by mean-square energy. _FakeModel energy grows with
        # index, so the top-64 are the highest indices in descending order.
        expected_energy = list(range(127, 63, -1))
        assert captured["energy"] == expected_energy

    def test_health_history_dedups_final(self, tmp_path: Path, val_batch):
        cb = ConvUnextBottleneckMonitorCallback(
            full_model=_FakeModel(c=128),
            val_batch=val_batch,
            output_dir=tmp_path,
            monitor_freq=5,
        )
        cb.on_epoch_end(4)   # -> epoch 5
        cb.on_epoch_end(9)   # -> epoch 10
        cb.on_train_end()    # -> 'final', must NOT add a third point

        assert cb.epochs_seen == [5, 10]
        for name, vals in cb.history.items():
            assert len(vals) == 2, f"{name} has {len(vals)} points, expected 2"

    def test_no_emit_on_unmonitored_epoch(self, tmp_path: Path, val_batch):
        cb = ConvUnextBottleneckMonitorCallback(
            full_model=_FakeModel(c=128),
            val_batch=val_batch,
            output_dir=tmp_path,
            monitor_freq=5,
        )
        cb.on_epoch_end(0)  # epoch+1 == 1 -> not monitored

        assert list(tmp_path.glob("*.png")) == []
        assert cb.epochs_seen == []

    def test_forward_failure_is_swallowed(self, tmp_path: Path, val_batch):
        cb = ConvUnextBottleneckMonitorCallback(
            full_model=_RaisingModel(),
            val_batch=val_batch,
            output_dir=tmp_path,
            monitor_freq=5,
        )
        # Must not raise into the training loop.
        cb.on_epoch_end(4)
        cb.on_train_end()

        assert list(tmp_path.glob("*.png")) == []

    def test_small_channel_count_guarded(self, tmp_path: Path, val_batch):
        # C < max_featuremap_channels -> show min(C, cap) without index overflow.
        cb = ConvUnextBottleneckMonitorCallback(
            full_model=_FakeModel(c=16),
            val_batch=val_batch,
            output_dir=tmp_path,
            monitor_freq=5,
            max_featuremap_channels=64,
        )
        cb.on_epoch_end(4)

        assert (tmp_path / "epoch_0005_bottleneck_first.png").exists()
        assert (tmp_path / "epoch_0005_bottleneck_energy.png").exists()
