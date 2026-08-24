"""Permanent build+forward smoke test for the tabm family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).

**No longer REPORT-ONLY.** The original version wrapped construction AND the
forward pass in `except Exception: pytest.xfail(...)`, so a total build break
reported as `xfail` -- the package's headline smoke test could not fail, which
is not an instrument. It now fails, and asserts the ensemble tensor's shape
rather than "some non-empty structure of finite values".

`create_tabm_mini` verified from source (model.py:965). NOTE: the real
signature is ``create_tabm_mini(n_num_features, cat_cardinalities, n_classes,
k=8, ...)`` -- the entrypoints note's ``create_tabm_mini(8, 3)`` is STALE
(``cat_cardinalities`` is a required positional list, not n_classes). We pass
``cat_cardinalities=[]`` (all-numerical tabular). call() (model.py:565) accepts
a single tensor as the numerical features.

MEASURED: the output is the ensemble tensor ``(B, k, n_classes)``, with the
default ``k=8``. Asserting that middle axis is the point of this contract -- an
implementation that collapsed the ensemble to ``(B, n_classes)`` would have been
invisible to the previous version.
"""

import numpy as np

from ..smoke_contract_oracle import assert_finite

BATCH, N_NUM_FEATURES, N_CLASSES = 2, 8, 3
K = 8  # create_tabm_mini's default ensemble size


def _build():
    from dl_techniques.models.tabm.model import create_tabm_mini

    return create_tabm_mini(
        n_num_features=N_NUM_FEATURES, cat_cardinalities=[], n_classes=N_CLASSES
    )


def _inputs():
    # Tabular numerical features: (batch=2, n_num_features=8) float32.
    return np.random.rand(BATCH, N_NUM_FEATURES).astype("float32")


def _assert_contract(out):
    """The smoke assertion. Shared with the meta-test so it is proven falsifiable."""
    assert not isinstance(out, (dict, list, tuple)), (
        f"tabm should return a single ensemble tensor, got {type(out)}"
    )
    assert tuple(out.shape) == (BATCH, K, N_CLASSES), tuple(out.shape)
    assert_finite(out)


def test_smoke_build_and_forward():
    _assert_contract(_build()(_inputs(), training=False))
