"""Smoke tests for :class:`BurstDPVisualizationCallback`.

Builds a tiny BurstDP, feeds a fake val "dataset" (an object indexable
[0] -> (inputs, labels)), wires the callback, and asserts:
    - step trigger fires after ``every_steps`` batch-ends
    - epoch trigger fires on ``on_epoch_end``
    - no exception is raised
    - PNGs land in ``out_dir/viz/``
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from dl_techniques.callbacks.burst_dp_visualization import BurstDPVisualizationCallback
from dl_techniques.models.burst_dp import BurstDP, BurstDPConfig


# Skip the whole module if matplotlib is unavailable.
matplotlib = pytest.importorskip("matplotlib")


def _fake_val_dataset(num_classes: int = 8, b: int = 2, n_max: int = 2, size: int = 32):
    inputs = {
        "ref": np.random.rand(b, size, size, 3).astype("float32"),
        "aux": np.random.rand(b, n_max, size, size, 3).astype("float32"),
        "aux_mask": np.ones((b, n_max), dtype="float32"),
    }
    labels = {
        "recon": np.random.rand(b, size, size, 3).astype("float32"),
        "segmentation": np.random.randint(0, num_classes, size=(b, size, size)).astype("int32"),
    }

    class _Ds:
        def __getitem__(self, idx: int):
            return inputs, labels

        def __len__(self) -> int:
            return 1

    return _Ds()


@pytest.fixture()
def tiny_model():
    cfg = BurstDPConfig(
        image_size=32,
        patch_size=8,
        n_max=2,
        encoder_scale="pico",
        fusion_blocks=1,
        fusion_heads=3,
        fusion_mlp_ratio=2.0,
        num_seg_classes=8,
        decoder_dims=(24, 16, 12),
    )
    return BurstDP(config=cfg)


class TestBurstDPVisualizationCallback:

    def test_step_trigger_fires(self, tiny_model, tmp_path):
        ds = _fake_val_dataset(num_classes=8, b=2, n_max=2, size=32)
        cb = BurstDPVisualizationCallback(
            val_dataset=ds,
            output_dir=str(tmp_path),
            every_steps=2,
            every_epochs=0,
            num_samples=2,
        )
        cb.set_model(tiny_model)
        # 3 batches => one step PNG at step=2.
        for batch in range(3):
            cb.on_train_batch_end(batch, logs={})
        pngs = sorted((tmp_path / "viz").glob("step_*.png"))
        assert len(pngs) == 1, f"expected 1 step PNG, got {[p.name for p in pngs]}"

    def test_epoch_trigger_fires(self, tiny_model, tmp_path):
        ds = _fake_val_dataset(num_classes=8, b=2, n_max=2, size=32)
        cb = BurstDPVisualizationCallback(
            val_dataset=ds,
            output_dir=str(tmp_path),
            every_steps=0,
            every_epochs=1,
            num_samples=2,
        )
        cb.set_model(tiny_model)
        cb.on_epoch_end(0, logs={})
        pngs = sorted((tmp_path / "viz").glob("epoch_*.png"))
        assert pngs == [tmp_path / "viz" / "epoch_0001.png"]

    def test_both_triggers_disabled_is_noop(self, tiny_model, tmp_path):
        ds = _fake_val_dataset(num_classes=8, b=2, n_max=2, size=32)
        cb = BurstDPVisualizationCallback(
            val_dataset=ds,
            output_dir=str(tmp_path),
            every_steps=0,
            every_epochs=0,
            num_samples=2,
        )
        cb.set_model(tiny_model)
        cb.on_train_batch_end(0, logs={})
        cb.on_epoch_end(0, logs={})
        # viz dir still created at __init__, but should be empty.
        assert (tmp_path / "viz").exists()
        assert list((tmp_path / "viz").glob("*.png")) == []

    def test_num_samples_caps_to_batch(self, tiny_model, tmp_path):
        ds = _fake_val_dataset(num_classes=8, b=2, n_max=2, size=32)
        cb = BurstDPVisualizationCallback(
            val_dataset=ds,
            output_dir=str(tmp_path),
            every_steps=0,
            every_epochs=1,
            num_samples=99,  # batch only has 2
        )
        assert cb.num_samples == 2

    def test_render_failure_does_not_crash(self, tiny_model, tmp_path, monkeypatch):
        ds = _fake_val_dataset(num_classes=8, b=2, n_max=2, size=32)
        cb = BurstDPVisualizationCallback(
            val_dataset=ds,
            output_dir=str(tmp_path),
            every_steps=0,
            every_epochs=1,
            num_samples=2,
        )
        cb.set_model(tiny_model)
        # Force render to blow up — callback must swallow and log.
        monkeypatch.setattr(cb, "_render", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        cb.on_epoch_end(0, logs={})  # must not raise
