"""Unit tests for :class:`CliffordBottleneckMonitorCallback`.

Covers:
- The four-PNG contract (health, featuremap, pca, histogram) on a monitored epoch.
- No-op (zero PNGs) when ``val_batch is None``.
- Frequency gating: no epoch-tagged PNGs on a non-monitored epoch.
- A failing ``encode`` is swallowed and never propagates into training.

Models are kept tiny (input 64x64x3, batch 6) for speed. ``small`` has
num_levels=3 so the bottleneck divisor is ``2**(3-1) == 4``; 64 is divisible
by 4. ``keras.callbacks.Callback.model`` is a read-only property, so the model
is attached via ``cb.set_model(m)`` (never ``cb.model = m``).
"""

from __future__ import annotations

import os

# Ensure matplotlib uses a non-interactive backend before any import.
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import numpy as np

from dl_techniques.callbacks.cliffordnet_bottleneck_monitor import (
    CliffordBottleneckMonitorCallback,
)
from dl_techniques.models.cliffordnet.autoencoder import create_clifford_laplacian_unet


def _build_model_and_batch():
    """Construct a tiny ``small`` model and a built (6, 64, 64, 3) batch."""
    m = create_clifford_laplacian_unet("small")
    x = np.random.rand(6, 64, 64, 3).astype("float32")
    m(x)  # build the model
    return m, x


class TestCliffordBottleneckMonitor:
    def test_writes_four_pngs(self, tmp_path):
        m, x = _build_model_and_batch()
        cb = CliffordBottleneckMonitorCallback(
            val_batch=x, output_dir=str(tmp_path), monitor_freq=5
        )
        cb.set_model(m)
        cb.on_epoch_end(4, {"loss": 0.1})  # (4 + 1) % 5 == 0 -> fires

        pngs = list(Path(tmp_path).rglob("*.png"))
        assert len(pngs) >= 4, f"expected >=4 PNGs, got {len(pngs)}: {pngs}"

        names = " ".join(p.name for p in pngs)
        for category in ("health", "featuremap", "pca", "histogram"):
            assert category in names, f"missing '{category}' PNG; got: {names}"

    def test_no_op_when_batch_none(self, tmp_path):
        m, _ = _build_model_and_batch()
        cb = CliffordBottleneckMonitorCallback(
            val_batch=None, output_dir=str(tmp_path), monitor_freq=5
        )
        cb.set_model(m)
        cb.on_epoch_end(4, {})  # must not raise

        pngs = list(Path(tmp_path).rglob("*.png"))
        assert pngs == [], f"expected 0 PNGs with val_batch=None, got: {pngs}"

    def test_frequency_gate(self, tmp_path):
        m, x = _build_model_and_batch()
        cb = CliffordBottleneckMonitorCallback(
            val_batch=x, output_dir=str(tmp_path), monitor_freq=5
        )
        cb.set_model(m)
        cb.on_epoch_end(3, {})  # (3 + 1) % 5 != 0 -> gated out

        pngs = list(Path(tmp_path).rglob("*.png"))
        # Only assert the epoch-tagged plots are absent; do not over-constrain
        # the cumulative health.png (impl may or may not write it).
        epoch_tagged = [
            p
            for p in pngs
            if any(cat in p.name for cat in ("featuremap", "pca", "histogram"))
        ]
        assert epoch_tagged == [], f"expected no epoch-tagged PNGs, got: {epoch_tagged}"

    def test_never_raises_into_training(self, tmp_path, monkeypatch):
        m, x = _build_model_and_batch()
        cb = CliffordBottleneckMonitorCallback(
            val_batch=x, output_dir=str(tmp_path), monitor_freq=5
        )
        cb.set_model(m)

        def _boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(m, "encode", _boom)

        # The try/except inside on_epoch_end must swallow the failure.
        cb.on_epoch_end(4, {})  # must not raise
