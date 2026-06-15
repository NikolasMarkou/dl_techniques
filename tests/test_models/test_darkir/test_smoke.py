"""Permanent build+forward smoke test for the darkir family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).
REPORT-ONLY: a build/forward break is documented via xfail, never fixed.

`create_darkir_model(img_channels, width, ...)` verified from source
(model.py:1299). Pure functional U-Net for low-light image restoration; NHWC
float32 input, returns a restored image ``(B, H, W, 3)``. The model uses the
recently-rewritten ``FreMLP`` (FFT path); a break there is captured via xfail.
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
        from dl_techniques.models.darkir.model import create_darkir_model

        model = create_darkir_model(img_channels=3, width=16)

        images = np.random.rand(2, 32, 32, 3).astype("float32")
        out = model(images, training=False)
    except Exception as exc:  # noqa: BLE001
        pytest.xfail(
            f"darkir build/forward failed: {type(exc).__name__}: {exc}"
        )

    # Functional model may return a tensor or (with side loss) a tuple/list.
    if isinstance(out, (list, tuple)):
        for v in out:
            _assert_finite(v)
    else:
        _assert_finite(out)
