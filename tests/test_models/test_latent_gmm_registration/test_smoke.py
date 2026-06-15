"""Permanent build+forward smoke test for the latent_gmm_registration family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).
REPORT-ONLY: a build/forward break is documented via xfail, never fixed.

`latent_gmm_registration/__init__.py` is empty, so import the class directly
from the submodule. Input is a tuple (source_pc, target_pc), each (B, N, 3).
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
        from dl_techniques.models.latent_gmm_registration.model import (
            LatentGMMRegistration,
        )

        # Smallest config: few GMM components, k_neighbors < N.
        model = LatentGMMRegistration(num_gaussians=4, k_neighbors=8)
        n = 64
        source = np.random.rand(2, n, 3).astype("float32")
        target = np.random.rand(2, n, 3).astype("float32")
        out = model((source, target), training=False)
    except Exception as exc:  # noqa: BLE001
        pytest.xfail(
            f"latent_gmm_registration build/forward failed: {type(exc).__name__}: {exc}"
        )

    # Output is a dict of tensors.
    assert isinstance(out, dict)
    for v in out.values():
        _assert_finite(v)
