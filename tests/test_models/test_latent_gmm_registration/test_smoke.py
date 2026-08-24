"""Permanent build+forward smoke test for the latent_gmm_registration family.

Part of the 2026-06-15 model build/forward sweep. The build/forward break
(`keras.ops.get_graph_feature` never existed) was fixed in plan_2026-06-15_00924f53
by adding the local `_get_graph_feature` DGCNN kNN helper, and the xfail wrapper
went with it. What remained was a contract of `isinstance(out, dict)` + non-empty
+ per-value finiteness, which ANY non-empty dict of finite values satisfies. It
now asserts the exact key set and each value's shape.

`latent_gmm_registration/__init__.py` is empty, so import the class directly
from the submodule. Input is a tuple (source_pc, target_pc), each (B, N, 3).

Keys and shapes MEASURED at ``num_gaussians=4, k_neighbors=8``: two
reconstructions ``(B, N, 3)`` and the rigid transform the model exists to
estimate -- a ``(B, 3, 3)`` rotation and a ``(B, 3)`` translation. Pinning
``estimated_r``/``estimated_t`` is the point: they are the model's actual output,
and a version that dropped them kept a non-empty dict of finite values.
"""

import numpy as np

from ..smoke_contract_oracle import assert_finite

BATCH, NUM_POINTS = 2, 64

EXPECTED_SHAPES = {
    "reconstruction_x": (BATCH, NUM_POINTS, 3),
    "reconstruction_y": (BATCH, NUM_POINTS, 3),
    "estimated_r": (BATCH, 3, 3),
    "estimated_t": (BATCH, 3),
}


def _build():
    from dl_techniques.models.latent_gmm_registration.model import (
        LatentGMMRegistration,
    )

    # Smallest config: few GMM components, k_neighbors < N.
    return LatentGMMRegistration(num_gaussians=4, k_neighbors=8)


def _inputs():
    return (
        np.random.rand(BATCH, NUM_POINTS, 3).astype("float32"),
        np.random.rand(BATCH, NUM_POINTS, 3).astype("float32"),
    )


def _assert_contract(out):
    """The smoke assertion. Shared with the meta-test so it is proven falsifiable."""
    assert isinstance(out, dict), f"expected a dict of tensors, got {type(out)}"
    assert set(out) == set(EXPECTED_SHAPES), sorted(out)
    for key, expected in EXPECTED_SHAPES.items():
        assert tuple(out[key].shape) == expected, f"{key}: {tuple(out[key].shape)}"
    assert_finite(out)


def test_smoke_build_and_forward():
    _assert_contract(_build()(_inputs(), training=False))
