"""Tests for COCOMAPCallback — COCO mAP eval at epoch end.

Three test levels:
1. Scoring path on hand-crafted detection dicts (no model, no GPU).
2. Empty / graceful cases.
3. Full integration: tiny model + real val subset → finite mAP, no crash.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List
from unittest.mock import MagicMock

import keras
import numpy as np
import pytest


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# _score_results — the pycocotools-dependent path, tested in isolation
# ---------------------------------------------------------------------------


class TestScoreResults:
    def _make_fake_loader(self, tmp_path):
        """Build a minimal pycocotools COCO with 2 images + 3 annotations."""
        import json
        from pycocotools.coco import COCO

        gt = {
            "images": [
                {"id": 1, "height": 100, "width": 100, "file_name": "img1.jpg"},
                {"id": 2, "height": 100, "width": 100, "file_name": "img2.jpg"},
            ],
            "annotations": [
                {
                    "id": 10, "image_id": 1, "category_id": 1,
                    "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0,
                },
                {
                    "id": 11, "image_id": 1, "category_id": 2,
                    "bbox": [50, 50, 20, 20], "area": 400, "iscrowd": 0,
                },
                {
                    "id": 12, "image_id": 2, "category_id": 1,
                    "bbox": [30, 30, 40, 40], "area": 1600, "iscrowd": 0,
                },
            ],
            "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
        }
        path = os.path.join(str(tmp_path), "gt.json")
        with open(path, "w") as f:
            json.dump(gt, f)
        coco = COCO(path)

        fake = MagicMock()
        fake.coco = coco
        fake.image_ids = [1, 2]
        fake.idx_to_cat_id = {0: 1, 1: 2}  # model class 0 → COCO cat 1, etc.
        fake.config = MagicMock()
        fake.config.batch_size = 2
        return fake

    def _build_callback(self, fake_loader):
        from dl_techniques.callbacks.coco_map_callback import COCOMAPCallback
        return COCOMAPCallback(
            val_loader=fake_loader,
            image_size=100,  # match GT image size for 1:1 coord scaling
            num_classes=2,
            reg_max=4,
            strides=(10, 20, 50),
            suppress_stdout=True,
        )

    def test_empty_results_returns_zero(self, tmp_path):
        fake = self._make_fake_loader(tmp_path)
        cb = self._build_callback(fake)
        ap50, ap5095 = cb._score_results([], [])
        assert ap50 == 0.0
        assert ap5095 == 0.0

    def test_perfect_predictions_high_map(self, tmp_path):
        """Feed pycocotools the GT boxes as predictions with high score → mAP ~1.0."""
        fake = self._make_fake_loader(tmp_path)
        cb = self._build_callback(fake)
        results: List[Dict[str, Any]] = [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.99},
            {"image_id": 1, "category_id": 2, "bbox": [50, 50, 20, 20], "score": 0.99},
            {"image_id": 2, "category_id": 1, "bbox": [30, 30, 40, 40], "score": 0.99},
        ]
        ap50, ap5095 = cb._score_results(results, [1, 2])
        assert ap50 >= 0.99, f"expected near-perfect mAP@50, got {ap50}"
        assert ap5095 >= 0.99, f"expected near-perfect mAP@50:95, got {ap5095}"

    def test_wrong_class_low_map(self, tmp_path):
        """Correct boxes but wrong class → mAP ≈ 0."""
        fake = self._make_fake_loader(tmp_path)
        cb = self._build_callback(fake)
        # Every detection is miss-classified (swap 1 ↔ 2).
        results: List[Dict[str, Any]] = [
            {"image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 20], "score": 0.99},
            {"image_id": 1, "category_id": 1, "bbox": [50, 50, 20, 20], "score": 0.99},
            {"image_id": 2, "category_id": 2, "bbox": [30, 30, 40, 40], "score": 0.99},
        ]
        ap50, _ = cb._score_results(results, [1, 2])
        assert ap50 <= 0.05, f"wrong-class predictions should score near 0, got {ap50}"


# ---------------------------------------------------------------------------
# on_epoch_end robustness
# ---------------------------------------------------------------------------


class TestOnEpochEnd:
    def _setup(self, tmp_path, skip=False):
        from dl_techniques.callbacks.coco_map_callback import COCOMAPCallback
        import json
        from pycocotools.coco import COCO

        gt = {
            "images": [{"id": 1, "height": 32, "width": 32, "file_name": "x.jpg"}],
            "annotations": [{
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [4, 4, 8, 8], "area": 64, "iscrowd": 0,
            }],
            "categories": [{"id": 1, "name": "a"}],
        }
        path = os.path.join(str(tmp_path), "gt.json")
        with open(path, "w") as f:
            json.dump(gt, f)

        coco = COCO(path)
        fake = MagicMock()
        fake.coco = coco
        fake.image_ids = [1]
        fake.idx_to_cat_id = {0: 1}
        fake.config = MagicMock()
        fake.config.batch_size = 1
        fake.__len__ = lambda self=None: 1
        # Model output: all logits zero → low scores → no detections pass threshold.
        A = 0
        for s in (4, 8, 16):
            A += (32 // s) * (32 // s)
        y_pred_zero = np.zeros((1, A, 4 * 4 + 1), dtype=np.float32)
        fake.__getitem__ = lambda self=None, i=0: (
            np.zeros((1, 32, 32, 3), dtype=np.float32),
            {"detection": np.zeros((1, 50, 5), dtype=np.float32)},
        )

        cb = COCOMAPCallback(
            val_loader=fake,
            image_size=32,
            num_classes=1,
            reg_max=4,
            strides=(4, 8, 16),
            frequency=1,
            score_threshold=0.99,  # force zero detections with default logits
            suppress_stdout=True,
        )
        # Fake model that returns the zeros detection tensor.
        model = MagicMock()
        model.return_value = {"detection": keras.ops.convert_to_tensor(y_pred_zero)}
        cb.set_model(model)
        return cb

    def test_frequency_skip(self, tmp_path):
        cb = self._setup(tmp_path)
        cb.frequency = 2
        logs: Dict[str, Any] = {}
        cb.on_epoch_end(0, logs)  # epoch 0 + 1 = 1, not divisible by 2
        assert "val_map50" not in logs
        cb.on_epoch_end(1, logs)  # epoch 1 + 1 = 2, divisible by 2
        assert "val_map50" in logs

    def test_empty_predictions_returns_zero_without_crash(self, tmp_path):
        cb = self._setup(tmp_path)
        logs: Dict[str, Any] = {}
        cb.on_epoch_end(0, logs)
        assert logs["val_map50"] == pytest.approx(0.0)
        assert logs["val_map5095"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_missing_attrs_raises(self):
        from dl_techniques.callbacks.coco_map_callback import COCOMAPCallback
        loader = MagicMock(spec=[])  # no attrs
        with pytest.raises(AttributeError, match="coco"):
            COCOMAPCallback(
                val_loader=loader, image_size=32, num_classes=1,
                reg_max=4, strides=(4, 8, 16),
            )
