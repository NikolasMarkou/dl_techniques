"""Permanent build+forward smoke test for the fftnet family (VISION path only).

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).
REPORT-ONLY: a build/forward break is documented via xfail, never fixed.

STALE-ENTRYPOINT NOTE: entrypoints.md row 20 lists fftnet as a token-id LM
(``int32 (B,T)``, ``create_fftnet(vocab_size=256)``). The actual source
(model.py:342 class, model.py:814 factory) is a VISION foundation model:
``create_fftnet(variant, image_size, patch_size)`` taking ``(B, H, W, 3)``
float32 images and returning a dict ``{last_hidden_state, cls_token,
patch_features}``. There is no ``vocab_size`` argument. We test the vision
path only and do NOT touch the (triple-dead) SpectreHead stack — out of scope.
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
        from dl_techniques.models.fftnet.model import create_fftnet

        # Smallest variant; tiny image so num_patches stays small.
        model = create_fftnet(variant="tiny", image_size=32, patch_size=16)

        images = np.random.rand(2, 32, 32, 3).astype("float32")
        out = model(images, training=False)
    except Exception as exc:  # noqa: BLE001
        pytest.xfail(
            f"fftnet (vision path) build/forward failed: "
            f"{type(exc).__name__}: {exc}"
        )

    # Output is a dict of tensors.
    assert isinstance(out, dict)
    for v in out.values():
        _assert_finite(v)
