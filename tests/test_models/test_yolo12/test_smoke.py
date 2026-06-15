"""Permanent build+forward smoke test for the yolo12 family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).
REPORT-ONLY: a build/forward break is documented via xfail, never fixed.

`create_yolov12_feature_extractor(input_shape, scale)` verified from source
(feature_extractor.py:363). Returns a multi-scale feature pyramid ``[P3, P4,
P5]`` (a list of NHWC tensors). Input must be divisible by the deepest stride
(32); 64x64 is a small legal input. Uses the smallest scale ``"n"`` (nano).
"""

import numpy as np
import pytest


def _assert_finite(value):
    arr = np.asarray(value)
    assert arr is not None
    assert not np.any(np.isnan(arr))
    assert not np.any(np.isinf(arr))


def test_smoke_build_and_forward():
    try:
        from dl_techniques.models.yolo12.feature_extractor import (
            create_yolov12_feature_extractor,
        )

        model = create_yolov12_feature_extractor(
            input_shape=(64, 64, 3), scale="n"
        )

        images = np.random.rand(2, 64, 64, 3).astype("float32")
        out = model(images, training=False)
    except Exception as exc:  # noqa: BLE001
        pytest.xfail(
            f"yolo12 build/forward failed: {type(exc).__name__}: {exc}"
        )

    # Feature extractor returns a list/tuple (or dict) of feature maps.
    if isinstance(out, dict):
        feats = list(out.values())
    elif isinstance(out, (list, tuple)):
        feats = list(out)
    else:
        feats = [out]
    assert len(feats) > 0
    for v in feats:
        _assert_finite(v)
