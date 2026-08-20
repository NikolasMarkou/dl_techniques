"""
R-002 / R-070 lazy-build rows for ``cliffordnet``, settled by measurement.

Batch 6 charged ``CliffordNetLM.build()`` with materializing **1 of 19 tensors /
64 of 1,288 params** -- 1,224 params, **95.0% of the model**, the worst ratio in
the plan -- while ``CliffordNet`` itself is clean at 25/25.

The contract failure reproduces exactly. **The consequence does not**: the round
trip is 0.000000e+00 against a live perturbation. See decisions.md D-056.
"""

import numpy as np

from ..lazy_build_contract_oracle import assert_lazy_build_costs_nothing


def _build():
    from dl_techniques.models.cliffordnet.lm import CliffordNetLM
    return CliffordNetLM(vocab_size=16, channels=8, depth=1,
                         max_seq_length=8, shifts=(1,))


def _inputs():
    return np.random.RandomState(0).randint(0, 16, (2, 8)).astype("int32")


def test_cliffordnet_lms_95_percent_lazy_build_costs_nothing_across_a_round_trip():
    """
    MEASURED (GPU 1): 19 weights after one call, **1 after ``.build()`` alone**
    (count_params 16); 19 perturbed; perturbation liveness **6.973980e-02**;
    reload weights 19; round trip **max|delta| exactly 0.000000e+00** at
    ``atol=0.0``.
    """
    report = assert_lazy_build_costs_nothing(
        build=_build,
        make_inputs=_inputs,
        input_shape=(None, 8),
    )
    assert report["roundtrip_max_delta"] == 0.0
    assert report["perturb_liveness"] > 1e-3
    assert report["materialization"]["n_weights_after_build"] == 1
    assert report["n_weights"] == 19
