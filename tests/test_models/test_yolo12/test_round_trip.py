"""M2 .keras round-trip + validation tests for the YOLOv12 models.

The yolo12 dir previously had only a build/forward smoke test. This adds real
coverage: construction, a ValueError input-validation path (H4, multitask), a
forward pass, and a full save -> load -> identical-output round-trip (atol 1e-5)
for both the feature extractor and the multi-task model.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops, tree

from dl_techniques.models.yolo12.feature_extractor import (
    YOLOv12FeatureExtractor,
    create_yolov12_feature_extractor,
)
from dl_techniques.models.yolo12.multitask import YOLOv12MultiTask


def _images(b=2, s=64, c=3):
    return np.random.rand(b, s, s, c).astype("float32")


def _assert_tree_close(a, b, atol=1e-5):
    a_flat = tree.flatten(a)
    b_flat = tree.flatten(b)
    assert len(a_flat) == len(b_flat)
    for x, y in zip(a_flat, b_flat):
        np.testing.assert_allclose(
            ops.convert_to_numpy(x), ops.convert_to_numpy(y),
            rtol=1e-5, atol=atol,
        )


class TestYOLOv12MultiTaskValidation:

    def test_invalid_scale(self):
        with pytest.raises(ValueError, match="scale must be one of"):
            YOLOv12MultiTask(scale="zzz", input_shape=(64, 64, 3))

    def test_invalid_reg_max(self):
        with pytest.raises(ValueError, match="reg_max must be positive"):
            YOLOv12MultiTask(scale="n", input_shape=(64, 64, 3), reg_max=0)


class TestYOLOv12FeatureExtractorRoundTrip:

    def test_forward(self):
        model = create_yolov12_feature_extractor(
            input_shape=(64, 64, 3), scale="n"
        )
        out = model(_images(), training=False)
        feats = list(out) if isinstance(out, (list, tuple)) else [out]
        assert len(feats) >= 1

    def test_keras_round_trip(self):
        model = create_yolov12_feature_extractor(
            input_shape=(64, 64, 3), scale="n"
        )
        x = _images()
        y0 = model(x, training=False)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "yolo_fe.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            y1 = reloaded(x, training=False)

        _assert_tree_close(y0, y1)


class TestYOLOv12MultiTaskRoundTrip:

    def test_forward_and_round_trip(self):
        model = YOLOv12MultiTask(
            num_classes=4, scale="n", input_shape=(64, 64, 3)
        )
        x = _images()
        y0 = model(x, training=False)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "yolo_mt.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            y1 = reloaded(x, training=False)

        _assert_tree_close(y0, y1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
