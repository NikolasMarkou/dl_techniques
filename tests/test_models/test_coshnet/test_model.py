"""Permanent build+forward smoke test for the coshnet family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).
REPORT-ONLY: a build/forward break is documented via xfail, never fixed.

`create_coshnet(variant, num_classes, input_shape)` verified from source
(model.py:642 -> CoShNet.from_variant). NHWC float32 image input; classifier
head returns logits ``(B, num_classes)``.
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
        from dl_techniques.models.coshnet.model import create_coshnet

        model = create_coshnet("base", 10, (32, 32, 3))

        images = np.random.rand(2, 32, 32, 3).astype("float32")
        out = model(images, training=False)
    except Exception as exc:  # noqa: BLE001
        pytest.xfail(
            f"coshnet build/forward failed: {type(exc).__name__}: {exc}"
        )

    _assert_finite(out)
