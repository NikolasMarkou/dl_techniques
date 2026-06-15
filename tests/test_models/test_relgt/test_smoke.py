"""Permanent build+forward smoke test for the relgt family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).
REPORT-ONLY: a build/forward break is documented via xfail, never fixed.

`relgt/__init__.py` is empty, so import the factory directly from the
submodule. The dict input keys are sourced from RELGTTokenEncoder.call():
``node_features``, ``node_types``, ``hop_distances``, ``relative_times``,
``subgraph_adjacency`` (NOT the looser names in entrypoints.md).
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
        from dl_techniques.models.relgt.model import create_relgt_model

        # 'small' preset; defaults: num_node_types=10, max_hops=2, feature_dim free.
        model = create_relgt_model(output_dim=2, model_size="small")

        b, n, f = 2, 16, 8
        inputs = {
            "node_features": np.random.rand(b, n, f).astype("float32"),
            "node_types": np.random.randint(0, 10, (b, n)).astype("int32"),
            "hop_distances": np.random.randint(0, 3, (b, n)).astype("int32"),
            "relative_times": np.random.rand(b, n, 1).astype("float32"),
            "subgraph_adjacency": np.random.rand(b, n, n).astype("float32"),
        }
        out = model(inputs, training=False)
    except Exception as exc:  # noqa: BLE001
        pytest.xfail(f"relgt build/forward failed: {type(exc).__name__}: {exc}")

    _assert_finite(out)
