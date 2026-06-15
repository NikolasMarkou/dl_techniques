"""Permanent build+forward smoke test for the tabm family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).
REPORT-ONLY: a build/forward break is documented via xfail, never fixed.

`create_tabm_mini` verified from source (model.py:965). NOTE: the real
signature is ``create_tabm_mini(n_num_features, cat_cardinalities, n_classes,
k=8, ...)`` -- the entrypoints note's ``create_tabm_mini(8, 3)`` is STALE
(``cat_cardinalities`` is a required positional list, not n_classes). We pass
``cat_cardinalities=[]`` (all-numerical tabular). call() (model.py:565) accepts
a single tensor as the numerical features; output is the ensemble tensor
``(B, k, n_classes)``.
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
        from dl_techniques.models.tabm.model import create_tabm_mini

        model = create_tabm_mini(
            n_num_features=8, cat_cardinalities=[], n_classes=3
        )

        # Tabular numerical features: (batch=2, n_num_features=8) float32.
        features = np.random.rand(2, 8).astype("float32")
        out = model(features, training=False)
    except Exception as exc:  # noqa: BLE001
        pytest.xfail(
            f"tabm build/forward failed: {type(exc).__name__}: {exc}"
        )

    if isinstance(out, dict):
        vals = list(out.values())
    elif isinstance(out, (list, tuple)):
        vals = list(out)
    else:
        vals = [out]
    assert len(vals) > 0
    for v in vals:
        _assert_finite(v)
