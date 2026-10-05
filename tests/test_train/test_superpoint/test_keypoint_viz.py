"""Tests for the SuperPoint viz repeatability aggregation.

Drives :class:`SuperPointVizCallback` with a stub backbone emitting canned
logits, so the decode -> reproject -> match -> JSONL chain is pinned without a
model: identity homographies with one dustbin-defying cell must score
repeatability 1.0 on every pair; all-dustbin logits must score 0.0.
"""

import json
import numpy as np

from train.common.keypoint_viz import SuperPointVizCallback


H = W = 16
HC = H // 8  # 2
D = 4


class _StubBackbone:
    """Callable returning canned ``(logits, descriptors)`` for any batch."""

    def __init__(self, logits_cell):
        # logits_cell: (2, 2, 65) base tiled over the batch.
        self._logits_cell = np.asarray(logits_cell, dtype=np.float32)

    def __call__(self, batch, training=False):
        b = int(np.shape(batch)[0])
        logits = np.tile(self._logits_cell[None], (b, 1, 1, 1))
        desc = np.zeros((b, H, W, D), dtype=np.float32)
        return {"keypoints": logits, "descriptors": desc}


def _dustbin_base():
    base = np.zeros((HC, HC, 65), dtype=np.float32)
    base[..., 64] = 10.0  # dustbin wins everywhere by default
    return base


def _run_callback(tmp_path, logits_cell, n_pairs=2):
    viz_images = np.zeros((n_pairs, H, W, 1), dtype=np.float32)
    cb = SuperPointVizCallback(
        viz_dir=str(tmp_path / "viz"),
        viz_images=viz_images,
        viz_warped=viz_images.copy(),
        viz_homographies=np.tile(np.eye(3, dtype=np.float32), (n_pairs, 1, 1)),
        every_n=1,
        border=0,
    )
    cb.set_model(_StubBackbone(logits_cell))
    cb.on_epoch_end(0)
    cb.on_epoch_end(1)
    return tmp_path / "viz"


class TestRepeatabilityAggregation:

    def test_perfect_pair_scores_one(self, tmp_path):
        base = _dustbin_base()
        base[0, 0, 0] = 10.0  # one detection at pixel (0, 0)
        base[0, 0, 64] = 0.0
        viz = _run_callback(tmp_path, base, n_pairs=2)

        assert (viz / "epoch_001_correspondence.png").is_file()
        lines = (viz / "repeatability.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["epoch"] == 1
        assert first["per_pair"] == [1.0, 1.0]
        assert first["mean_repeatability"] == 1.0
        assert first["reprojectable"] == [1, 1]

    def test_empty_scores_zero(self, tmp_path):
        viz = _run_callback(tmp_path, _dustbin_base(), n_pairs=3)

        lines = (viz / "repeatability.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["mean_repeatability"] == 0.0
        assert first["per_pair"] == [0.0, 0.0, 0.0]
