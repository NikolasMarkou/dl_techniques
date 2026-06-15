"""Permanent build+forward smoke test for the swin_transformer family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).
REPORT-ONLY: a build/forward break is documented via xfail, never fixed.

`create_swin_transformer(variant, num_classes, input_shape)` verified from
source (model.py:627 -> SwinTransformer.from_variant). The tiny variant uses
``patch_size=4`` and ``window_size=8``; input H/W must be divisible by
``patch_size * 8 = 32`` (model.py:292-299), so the smallest legal square input
is 32x32 (matches the docstring example at model.py:30). Returns logits
``(B, num_classes)``.
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
        from dl_techniques.models.swin_transformer.model import (
            create_swin_transformer,
        )

        # 32x32 is the smallest legal input (divisible by patch_size*8 == 32).
        model = create_swin_transformer("tiny", 10, input_shape=(32, 32, 3))

        images = np.random.rand(2, 32, 32, 3).astype("float32")
        out = model(images, training=False)
    except Exception as exc:  # noqa: BLE001
        pytest.xfail(
            f"swin_transformer build/forward failed: "
            f"{type(exc).__name__}: {exc}"
        )

    _assert_finite(out)
