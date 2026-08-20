"""
R-002 / R-070 lazy-build rows for ``depth_anything``, settled by measurement.

Batch 5 charged ``DepthAnything.build()`` with materializing **300 of 322
tensors / 43,193,088 of 44,467,105 params**, the missing 22 being the
``decoder/`` group -- 1,274,017 params, 2.87% of the model -- and noted the
gap was "masked by ``load_own_variables``' force-build".

**The mask is real**: on the placeholder-encoder configuration the round trip is
exact to 0.000000e+00 against a live perturbation. See decisions.md D-056.

``input_shape`` is deliberately NOT passed to the oracle here: this package's
factory is documented as "Create and build", so the model is already built when
it is returned and a second ``.build()`` raises ``ValueError: You cannot add new
elements of state``. That reading is recorded in D-056 rather than asserted as a
contract, because it is a property of the FACTORY, not of the class.
"""

import numpy as np

from ..lazy_build_contract_oracle import assert_lazy_build_costs_nothing


def _build():
    from dl_techniques.models.depth_anything.model import create_depth_anything
    return create_depth_anything(
        encoder_type='vit_s', image_shape=(64, 64, 3),
        encoder_kind='placeholder', decoder_dims=[16, 16, 16, 16],
    )


def _inputs():
    return np.random.RandomState(0).rand(1, 64, 64, 3).astype("float32")


def test_depth_anythings_partial_build_costs_nothing_across_a_round_trip():
    """
    MEASURED (GPU 1, placeholder encoder): 114 weights after one call, 70
    perturbed (BatchNorm moving statistics excluded); perturbation liveness
    **2.169124e+00**; reload weights 114; round trip **max|delta| exactly
    0.000000e+00** at ``atol=0.0``.
    """
    report = assert_lazy_build_costs_nothing(
        build=_build,
        make_inputs=_inputs,
    )
    assert report["roundtrip_max_delta"] == 0.0
    assert report["perturb_liveness"] > 1e-3
    assert report["n_weights_reloaded"] == report["n_weights"]
