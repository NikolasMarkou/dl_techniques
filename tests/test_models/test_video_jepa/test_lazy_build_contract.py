"""
R-002 / R-070 lazy-build rows for ``video_jepa`` -- the partial build is CLOSED.

Batch 6 charged ``VideoJEPA.build()`` with materializing **36 of 86 tensors /
10,673 of 16,980 params**, leaving 6,307 params (37.1% of the model) to appear
only on the first call, with a Keras ``UserWarning``. D-056 closed that as a
contract row rather than a defect row, by pinning the round trip instead of
adding a ``build()``, and this module's own assertion invited the update:
"if a later change adds a materialising build() this line fails and should be
updated to the new ratio".

That change landed (plan ``plan-2026-08-23T091307-9a110062``, D-425). The model
now hand-walks its weight-bearing sub-layers in ``build()``, so the ratio is
**1.0** and the ``UserWarning`` is gone. The round-trip arm is UNCHANGED and
still measures exactly 0.000000e+00, which is the point: the build was never
what made the round trip safe, and the round trip is still what proves the
build did not break anything.
"""

import numpy as np

from ..lazy_build_contract_oracle import assert_lazy_build_costs_nothing


def _build():
    from dl_techniques.models.vision.video_jepa.model import create_video_jepa
    return create_video_jepa(img_size=32, patch_size=16, num_frames=2, embed_dim=32)


def _inputs():
    return {"pixels": np.random.RandomState(0).randn(1, 2, 32, 32, 3).astype("float32")}


def test_video_jepas_build_is_now_total_and_still_costs_nothing():
    """
    MEASURED 2026-08-23 (CPU): 161 weights after one call and **161 after
    ``.build()`` alone**, ratio 1.0; 149 perturbed; round trip **max|delta|
    exactly 0.000000e+00** at ``atol=0.0``.

    The previous reading on this line was "**66 after ``.build()`` alone**"
    (GPU 1, perturbation liveness 3.009824e-01). It is recorded here rather than
    deleted because it did not reproduce even before the fix: the same
    partial-build side effect measured **158** of 161 on 2026-08-23. Whatever
    the 66 was, it is not what this instrument reads today, and the number that
    replaces it is a TOTAL, which is the only value that cannot drift.
    """
    report = assert_lazy_build_costs_nothing(
        build=_build,
        make_inputs=_inputs,
        input_shape={"pixels": (None, 2, 32, 32, 3)},
    )
    assert report["roundtrip_max_delta"] == 0.0
    assert report["perturb_liveness"] > 1e-3
    # The inverse of the old assertion: `build()` must now materialize the WHOLE
    # model. `<` would pass on a build that lost a sub-layer; `==` cannot.
    assert report["materialization"]["n_weights_after_build"] == report["n_weights"]
    assert report["materialization"]["ratio"] == 1.0
