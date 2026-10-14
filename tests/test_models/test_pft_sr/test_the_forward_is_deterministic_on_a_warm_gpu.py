"""The PFT-SR forward is bit-deterministic on a GPU whose cuDNN autotuner is warm.

Why this file exists
--------------------

``test_roundtrip_instrument_family.py::test_the_roundtrip_compares_output_values[pft_sr]``
failed intermittently with "pft_sr is not deterministic (self spread
1.192093e-07)": one float32 ulp at ``max|y| = 1.416``, between two calls of the
SAME model on the SAME input. Root cause, measured at
plan-2026-10-14T062616-81938e81 step 6.1 (D-024):

* The red needed only TWO tests in one process: any SCUNet test that builds and
  calls the model (``test_the_explicit_build_materializes_the_model.py[scunet]``
  was enough), then the pft_sr round trip. PFT-SR alone: green.
* The non-deterministic layer was the 2026-10-10 ``EnhancementHead``'s
  ``Conv2DTranspose`` (the only transposed conv PFT-SR had). On GPU its forward
  runs cuDNN backward-data (``Conv2DBackpropInput [1,64,64,3] filter
  [3,3,3,52]``), whose algorithm TF picks by TIMING each candidate once.
  ``TF_CPP_VMODULE=gpu_utils=2`` logged the pick: ``ConvBwdData_eng65`` or
  ``eng58`` in a fresh process (4 of 4 processes, spread 0.0), and
  ``ConvBwdData_eng25_k2=2`` after SCUNet's own transposed convs had run (4 of 4,
  spread 1.19e-07). Engine 25 measured 327.7 us cold (first launch) against
  24.6 us warm, so a preceding transposed-conv autotune is what lets it win.
* ``TF_CUDNN_DETERMINISTIC=1`` filters engine 25 out (the pick became
  ``eng58``, spread 0.0 in 3 of 3): cuDNN flags it non-deterministic.
* Toggling op determinism INSIDE the process does not help: the autotune cache
  keeps the earlier pick (measured: spread unchanged after
  ``enable_op_determinism()``, for the same model and for a fresh one), so a
  scoped fixture around the round trip cannot be the fix.

The fix is in the model: PFT-SR's tail is now the official one (pixel shuffle
after forward convolutions, no transposed conv), so no cuDNN backward-data
kernel runs in its forward at all.

This guard reproduces the triggering state on purpose: it warms the autotuner
with the SCUNet round-trip subject first, then asserts the PFT-SR subject's
call-to-call spread is EXACTLY 0.0, on the subject and perturbation
``measure_roundtrip`` uses. RED-proven by restoring a ``Conv2DTranspose`` tail
in place (``changelog.md`` step 6.1). On a host without a GPU the mechanism
does not exist, so the test skips rather than passing vacuously.
"""

import numpy as np
import pytest
import tensorflow as tf
import keras

from ..precision_arm_subjects import roundtrip_subject
from ..roundtrip_instrument_oracle import flatten_tensors, perturb_weights

#: Calls of the same model on the same input. The measured red needed 2.
N_CALLS = 8


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"),
                    reason="cuDNN autotuning only exists on a GPU")
def test_pft_sr_is_deterministic_after_a_transposed_conv_warmed_the_autotuner():
    warm_build, warm_inputs, _ = roundtrip_subject("scunet")
    keras.utils.set_random_seed(0)
    warm_build()(warm_inputs(), training=False)

    build, make_inputs, _ = roundtrip_subject("pft_sr")
    keras.utils.set_random_seed(0)
    model = build()
    x = make_inputs()
    model(x, training=False)
    perturb_weights(model, seed=0)

    def outputs():
        return [np.asarray(t) for t in flatten_tensors(model(x, training=False))]

    reference = outputs()
    assert max(float(np.abs(r).max()) for r in reference) > 1e-3
    spread = 0.0
    for _ in range(N_CALLS):
        spread = max(spread, max(
            float(np.abs(a.astype(np.float64) - r).max())
            for a, r in zip(outputs(), reference)))
    assert spread == 0.0, (
        f"pft_sr is not deterministic on a warm GPU: self spread {spread:.6e}. "
        "A layer in its forward reaches a cuDNN kernel the autotuner can pick "
        "non-deterministically (a transposed conv runs backward-data); see "
        "this module's docstring."
    )
