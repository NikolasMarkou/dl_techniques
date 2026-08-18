"""Permanent build+forward smoke test for the mothnet family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).

The `except Exception: pytest.xfail(...)` wrapper was removed from this file in
an earlier pass, but its "REPORT-ONLY ... via xfail" docstring outlived it and
the body still asserted only `len(vals) > 0` plus finiteness -- so a forward
returning the scalar `0.0` passed. It now asserts the logits' shape.

`MothNet(num_classes=10)` verified from source (model.py:94). Bio-mimetic
feature generator; ``al_units`` is inferred from the input dimension at build
time, so no factory is needed. Input is a 2D tabular tensor ``(B, F)`` float32
(call() forwards a single tensor through AL -> MB -> Hebbian readout).

MEASURED: output is class logits ``(B, num_classes)``. Because ``al_units``
is input-inferred, the output width is the one dimension that does NOT follow
the input -- worth asserting explicitly.
"""

import numpy as np

from ..smoke_contract_oracle import assert_finite

BATCH, NUM_FEATURES, NUM_CLASSES = 2, 64, 10


def _build():
    from dl_techniques.models.mothnet.model import MothNet

    return MothNet(num_classes=NUM_CLASSES)


def _inputs():
    # Tabular: (batch=2, num_features=64) float32.
    return np.random.rand(BATCH, NUM_FEATURES).astype("float32")


def _assert_contract(out):
    """The smoke assertion. Shared with the meta-test so it is proven falsifiable."""
    assert not isinstance(out, (dict, list, tuple)), (
        f"mothnet should return a single logits tensor, got {type(out)}"
    )
    assert tuple(out.shape) == (BATCH, NUM_CLASSES), tuple(out.shape)
    assert_finite(out)


def test_smoke_build_and_forward():
    _assert_contract(_build()(_inputs(), training=False))
