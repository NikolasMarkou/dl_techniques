"""Permanent build+forward smoke test for the pft_sr family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).
REPORT-ONLY: a build/forward break is documented via xfail, never fixed.

`create_pft_sr(scale, variant)` verified from source (model.py:376). Permuted
self-attention super-resolution model with ``window_size=8`` (model.py:438), so
input H/W are kept divisible by 8 (32x32). NHWC float32 input ``(B, H, W, 3)``;
at ``scale=2`` the output is the upsampled image ``(B, 2H, 2W, 3)``.

The pft_sr import was recently fixed (was a ModuleNotFoundError) and has NEVER
been exercised end-to-end — a build/forward break is captured via xfail.
"""

import numpy as np


def _assert_finite(value):
    arr = np.asarray(value)
    assert arr is not None
    assert not np.any(np.isnan(arr))
    assert not np.any(np.isinf(arr))


def test_smoke_build_and_forward():
    from dl_techniques.models.pft_sr.model import create_pft_sr

    model = create_pft_sr(scale=2, variant="light")

    images = np.random.rand(2, 32, 32, 3).astype("float32")
    out = model(images, training=False)

    if isinstance(out, (list, tuple)):
        for v in out:
            _assert_finite(v)
    else:
        _assert_finite(out)
