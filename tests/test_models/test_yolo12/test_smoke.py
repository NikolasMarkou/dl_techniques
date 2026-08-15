"""Permanent build+forward smoke test for the yolo12 family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).

**No longer REPORT-ONLY.** The original version wrapped construction AND the
forward pass in `except Exception: pytest.xfail(...)`, so a total build break
reported as `xfail` — the package's headline smoke test could not fail, which
is not an instrument. It now fails, and asserts the feature pyramid's shape
rather than only its finiteness.

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
    from dl_techniques.models.yolo12.feature_extractor import (
        create_yolov12_feature_extractor,
    )

    model = create_yolov12_feature_extractor(
        input_shape=(64, 64, 3), scale="n"
    )

    images = np.random.rand(2, 64, 64, 3).astype("float32")
    out = model(images, training=False)

    # Feature extractor returns a list/tuple (or dict) of feature maps.
    if isinstance(out, dict):
        feats = list(out.values())
    elif isinstance(out, (list, tuple)):
        feats = list(out)
    else:
        feats = [out]

    # P3/P4/P5 at strides 8/16/32 of a 64x64 input. Asserting the shapes, not
    # just `len(feats) > 0`: the previous version passed for any non-empty
    # output, including a single wrongly-strided map.
    assert len(feats) == 3, [tuple(f.shape) for f in feats]
    for feat, stride in zip(feats, (8, 16, 32)):
        assert tuple(feat.shape[:3]) == (2, 64 // stride, 64 // stride), (
            f"stride-{stride} map has shape {tuple(feat.shape)}"
        )
        _assert_finite(feat)


def test_the_smoke_test_fails_on_a_build_break():
    """RED-proof, in-suite: a broken build must FAIL, not xfail.

    The original wrapper turned any exception into `pytest.xfail`, so this
    scenario reported green. Simulated here by calling the same factory with an
    input size the architecture cannot accept.
    """
    from dl_techniques.models.yolo12.feature_extractor import (
        create_yolov12_feature_extractor,
    )

    with pytest.raises(Exception):
        model = create_yolov12_feature_extractor(
            input_shape=(64, 64, 3), scale="not_a_scale"
        )
        model(np.zeros((1, 64, 64, 3), dtype="float32"), training=False)
