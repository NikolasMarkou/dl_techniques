"""Permanent build+forward smoke test for the shgcn family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).
REPORT-ONLY: a build/forward break is documented via xfail, never fixed.

`shgcn/__init__.py` is empty, so import the class directly from the submodule.
Input is an UNBATCHED list [features (N, F), adjacency (N, N)] per call().
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
        from dl_techniques.models.shgcn.model import SHGCNNodeClassifier

        model = SHGCNNodeClassifier(num_classes=3, hidden_dims=[16, 16])
        n, f = 16, 8
        features = np.random.rand(n, f).astype("float32")
        adjacency = np.random.rand(n, n).astype("float32")
        out = model([features, adjacency], training=False)
    except Exception as exc:  # noqa: BLE001
        pytest.xfail(f"shgcn build/forward failed: {type(exc).__name__}: {exc}")

    _assert_finite(out)
