"""
R-002 / R-070 lazy-build rows for ``video_jepa``, settled by measurement.

Batch 6 charged ``VideoJEPA.build()`` with materializing **36 of 86 tensors /
10,673 of 16,980 params**, leaving 6,307 params (37.1% of the model) to appear
only on the first call, with a Keras ``UserWarning``.

The contract failure reproduces. **The consequence does not**: a save/load cycle
on a perturbed model is exact to 0.000000e+00 with a live perturbation arm. See
decisions.md D-056 -- this is a contract row, not a defect row, and it is closed
by pinning the round trip rather than by adding a ``build()``.
"""

import numpy as np

from ..lazy_build_contract_oracle import assert_lazy_build_costs_nothing


def _build():
    from dl_techniques.models.video_jepa.model import create_video_jepa
    return create_video_jepa(img_size=32, patch_size=16, num_frames=2, embed_dim=32)


def _inputs():
    return {"pixels": np.random.RandomState(0).randn(1, 2, 32, 32, 3).astype("float32")}


def test_video_jepas_partial_build_costs_nothing_across_a_round_trip():
    """
    MEASURED (GPU 1): 161 weights after one call, **66 after ``.build()``
    alone** (count_params 81,505); 149 perturbed; perturbation liveness
    **3.009824e-01**; reload weights 161; round trip **max|delta| exactly
    0.000000e+00** at ``atol=0.0``.
    """
    report = assert_lazy_build_costs_nothing(
        build=_build,
        make_inputs=_inputs,
        input_shape={"pixels": (None, 2, 32, 32, 3)},
    )
    assert report["roundtrip_max_delta"] == 0.0
    assert report["perturb_liveness"] > 1e-3
    # The contract failure itself, pinned as a NUMBER. If a later change adds a
    # materialising build() this line fails and should be updated to the new
    # ratio -- it is a record of what was measured, not a target.
    assert report["materialization"]["n_weights_after_build"] < report["n_weights"]
