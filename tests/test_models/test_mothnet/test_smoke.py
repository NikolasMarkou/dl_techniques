"""Permanent build+forward smoke test for the mothnet family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).
REPORT-ONLY: a build/forward break is documented via xfail, never fixed.

`MothNet(num_classes=10)` verified from source (model.py:94). Bio-mimetic
feature generator; ``al_units`` is inferred from the input dimension at build
time, so no factory is needed. Input is a 2D tabular tensor ``(B, F)`` float32
(call() at model.py forwards a single tensor through AL -> MB -> Hebbian
readout); output is class logits ``(B, num_classes)``.
"""

import numpy as np


def _assert_finite(value):
    arr = np.asarray(value)
    assert arr is not None
    assert not np.any(np.isnan(arr))
    assert not np.any(np.isinf(arr))


def test_smoke_build_and_forward():
    from dl_techniques.models.mothnet.model import MothNet

    model = MothNet(num_classes=10)

    # Tabular: (batch=2, num_features=64) float32.
    features = np.random.rand(2, 64).astype("float32")
    out = model(features, training=False)

    if isinstance(out, dict):
        vals = list(out.values())
    elif isinstance(out, (list, tuple)):
        vals = list(out)
    else:
        vals = [out]
    assert len(vals) > 0
    for v in vals:
        _assert_finite(v)
