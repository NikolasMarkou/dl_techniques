"""Permanent build+forward smoke test for the shgcn family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).

**No longer REPORT-ONLY.** The original version wrapped construction AND the
forward pass in `except Exception: pytest.xfail(...)`, so a total build break
reported as `xfail` -- the package's headline smoke test could not fail, which
is not an instrument. It now fails, and asserts the per-node logits' shape
rather than only their finiteness.

`shgcn/__init__.py` is empty, so import the class directly from the submodule.
Input is an UNBATCHED list [features (N, F), adjacency (N, N)] per call().

MEASURED: output is ``(N, num_classes)`` -- one logit row per NODE, with no
batch axis. The missing batch axis is precisely the kind of contract a
finiteness-only assertion cannot state, and it is the shape a caller has to get
right.
"""

import numpy as np

from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
)

NUM_NODES, FEATURE_DIM, NUM_CLASSES = 16, 8, 3


def _build():
    from dl_techniques.models.shgcn.model import SHGCNNodeClassifier

    return SHGCNNodeClassifier(num_classes=NUM_CLASSES, hidden_dims=[16, 16])


def _inputs():
    return [
        np.random.rand(NUM_NODES, FEATURE_DIM).astype("float32"),
        np.random.rand(NUM_NODES, NUM_NODES).astype("float32"),
    ]


def _assert_contract(out):
    """The smoke assertion. Shared with the meta-test so it is proven falsifiable."""
    assert not isinstance(out, (dict, list, tuple)), (
        f"shgcn should return a single per-node logits tensor, got {type(out)}"
    )
    assert tuple(out.shape) == (NUM_NODES, NUM_CLASSES), tuple(out.shape)
    assert_finite(out)


def test_smoke_build_and_forward():
    _assert_contract(_build()(_inputs(), training=False))


def test_the_smoke_contract_rejects_a_broken_forward():
    """RED-proof, in-suite: the assertion above can actually fail.

    Breaks the MODEL -- the built model's forward output is replaced by a
    degenerate one -- rather than the factory's argument validation. See
    ``smoke_contract_oracle`` for why the argument-validation form of this
    meta-test (which yolo12 shipped) proved nothing.

    This package carries the meta-test because its output is UNBATCHED, the one
    shape class where ``slice_leading_axis`` cuts the node axis rather than a
    batch axis -- the injection has to still register as a break.
    """
    assert_contract_rejects_a_broken_forward(_build(), _inputs(), _assert_contract)
