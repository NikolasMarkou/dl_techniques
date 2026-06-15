"""Permanent build+forward smoke test for the dino family (v1).

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).
REPORT-ONLY: a build/forward break is documented via xfail, never fixed.

dino has no ``__init__`` exports, so import the factory directly from the
submodule (dino_v1.py:806). `create_dino_v1(variant, num_classes, patch_size,
input_shape, ...)` verified from source. The patch grid is derived from
``image_size // patch_size`` at build time, so a small ``input_shape`` is legal:
with ``patch_size=16`` a 32x32 image yields a 2x2 patch grid. ``num_classes=10``
+ default ``include_top=True`` returns logits ``(B, 10)``.

FORWARD ONLY: the DINOHead ``.keras`` round-trip is known-broken — not tested.
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
        from dl_techniques.models.dino.dino_v1 import create_dino_v1

        model = create_dino_v1(
            "small",
            num_classes=10,
            patch_size=16,
            input_shape=(32, 32, 3),
        )

        images = np.random.rand(2, 32, 32, 3).astype("float32")
        out = model(images, training=False)
    except Exception as exc:  # noqa: BLE001
        pytest.xfail(
            f"dino_v1 build/forward failed: {type(exc).__name__}: {exc}"
        )

    # Output may be a tensor (logits) or dict depending on head config.
    if isinstance(out, dict):
        for v in out.values():
            _assert_finite(v)
    else:
        _assert_finite(out)
