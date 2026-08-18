"""Permanent build+forward smoke test for the relgt family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).

**No longer REPORT-ONLY.** The original version wrapped construction AND the
forward pass in `except Exception: pytest.xfail(...)`, so a total build break
reported as `xfail` -- the package's headline smoke test could not fail, which
is not an instrument. It now fails, and asserts the output's shape rather than
only its finiteness.

`relgt/__init__.py` is empty, so import the factory directly from the
submodule. The dict input keys are sourced from RELGTTokenEncoder.call():
``node_features``, ``node_types``, ``hop_distances``, ``relative_times``,
``subgraph_adjacency`` (NOT the looser names in entrypoints.md).

Output shape MEASURED at this config, not read from a docstring: the 'small'
preset with ``output_dim=2`` returns a single ``(batch, 2)`` prediction tensor.
"""

import numpy as np

from ..smoke_contract_oracle import assert_finite

BATCH, NUM_NODES, FEATURE_DIM = 2, 16, 8
OUTPUT_DIM = 2


def _build():
    from dl_techniques.models.relgt.model import create_relgt_model

    # 'small' preset; defaults: num_node_types=10, max_hops=2, feature_dim free.
    return create_relgt_model(output_dim=OUTPUT_DIM, model_size="small")


def _inputs():
    return {
        "node_features": np.random.rand(BATCH, NUM_NODES, FEATURE_DIM).astype("float32"),
        "node_types": np.random.randint(0, 10, (BATCH, NUM_NODES)).astype("int32"),
        "hop_distances": np.random.randint(0, 3, (BATCH, NUM_NODES)).astype("int32"),
        "relative_times": np.random.rand(BATCH, NUM_NODES, 1).astype("float32"),
        "subgraph_adjacency": np.random.rand(BATCH, NUM_NODES, NUM_NODES).astype("float32"),
    }


def _assert_contract(out):
    """The smoke assertion. Shared with the meta-test so it is proven falsifiable."""
    assert not isinstance(out, (dict, list, tuple)), (
        f"relgt should return a single prediction tensor, got {type(out)}"
    )
    assert tuple(out.shape) == (BATCH, OUTPUT_DIM), tuple(out.shape)
    assert_finite(out)


def test_smoke_build_and_forward():
    _assert_contract(_build()(_inputs(), training=False))
