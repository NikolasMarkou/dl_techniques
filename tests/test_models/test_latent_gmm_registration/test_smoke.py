"""Permanent build+forward smoke test for the latent_gmm_registration family.

Part of the 2026-06-15 model build/forward sweep. The build/forward break
(`keras.ops.get_graph_feature` never existed) was fixed in plan_2026-06-15_00924f53
by adding the local `_get_graph_feature` DGCNN kNN helper; this test is now a real
forward-pass + finiteness assertion (no longer xfail).

`latent_gmm_registration/__init__.py` is empty, so import the class directly
from the submodule. Input is a tuple (source_pc, target_pc), each (B, N, 3).
"""

import numpy as np


def _assert_finite(value):
    arr = np.asarray(value)
    assert arr is not None
    assert not np.any(np.isnan(arr))
    assert not np.any(np.isinf(arr))


def test_smoke_build_and_forward():
    from dl_techniques.models.latent_gmm_registration.model import (
        LatentGMMRegistration,
    )

    # Smallest config: few GMM components, k_neighbors < N.
    model = LatentGMMRegistration(num_gaussians=4, k_neighbors=8)
    n = 64
    source = np.random.rand(2, n, 3).astype("float32")
    target = np.random.rand(2, n, 3).astype("float32")
    out = model((source, target), training=False)

    # Output is a dict of tensors; every value must be finite.
    assert isinstance(out, dict)
    assert out, "forward returned an empty dict"
    for v in out.values():
        _assert_finite(v)
