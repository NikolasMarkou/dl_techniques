"""The VQ-VAE round-trip subjects are exact on a warm GPU (D-083).

``test_roundtrip_instrument_family.py::test_the_roundtrip_compares_output_values[vq_vae_rotation]``
failed intermittently (2 of about 6 full-file runs, never alone), and its
sibling ``vq_vae`` carried a fixed 1e-8 bound for a measured 9.31e-10
self spread. Root cause, measured at step 12 of
``plan-2026-10-14T062616-81938e81`` with
``TF_CPP_VMODULE=gpu_utils=2,conv_grad_input_ops=2``:

* both subjects share ``precision_arm_subjects._tiny_conv_codec``, whose
  decoder was two ``Conv2DTranspose`` layers; on a GPU each runs as
  ``Conv2DBackpropInput``, autotuned by timing;
* in fresh processes the last deconv (``input [1,32,32,1] filter [3,3,1,16]``)
  picked ``ConvBwdData_eng0``/``eng2``/``eng65`` and the self spread was 0.0
  (6 of 6, with and without the sibling first);
* after the SCUNet subject's transposed convs had run (it precedes ``vq_*`` in
  the family's order), one process of four picked ``ConvBwdData_eng25`` and
  ``vq_vae_rotation``'s self spread was 1.86e-09 with the encoder exact. This
  is the engine ``test_pft_sr/test_the_forward_is_deterministic_on_a_warm_gpu.py``
  documents: flagged non-deterministic by cuDNN, chosen only by timing.

The order dependence is therefore real but probabilistic (which engine wins a
timing race), which is why the flake passed alone and on re-run. The fix is in
the harness, the only place that chose a transposed conv: the codec upsamples
with ``UpSampling2D`` + ``Conv2D``. After it, six SCUNet-warmed processes
launched no backward-data kernel for either subject and both spreads were 0.0.

Two checks. The structural one is the RED-provable guard (the old codec fails
it deterministically). The warm-GPU one reproduces the triggering state as the
PFT-SR guard does; because the bad engine wins only some timing races, it
cannot be RED-proven on demand and is kept as a regression check.
"""

import numpy as np
import pytest
import tensorflow as tf
import keras

from .precision_arm_subjects import roundtrip_subject
from .roundtrip_instrument_oracle import flatten_tensors, perturb_weights

SUBJECTS = ("vq_vae", "vq_vae_rotation")

#: Calls of the same model on the same input.
N_CALLS = 8


def _all_layers(model):
    return list(model._flatten_layers(include_self=False, recursive=True))


@pytest.mark.parametrize("name", SUBJECTS)
def test_the_subject_runs_no_transposed_conv(name):
    build, _, _ = roundtrip_subject(name)
    keras.utils.set_random_seed(0)
    model = build()
    layers = _all_layers(model)
    assert any(isinstance(layer, keras.layers.Conv2D) for layer in layers), (
        "the walk found no Conv2D at all; it is not reaching the codec")
    transposed = [layer.name for layer in layers
                  if isinstance(layer, keras.layers.Conv2DTranspose)]
    assert not transposed, (
        f"{name} runs Conv2DTranspose {transposed}: on a GPU that is cuDNN "
        "backward-data, whose autotuner can pick a non-deterministic engine "
        "and break the round trip's exact comparison (D-083)")


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"),
                    reason="cuDNN autotuning only exists on a GPU")
@pytest.mark.parametrize("name", SUBJECTS)
def test_the_subject_is_deterministic_after_a_transposed_conv_warmed_the_autotuner(name):
    warm_build, warm_inputs, _ = roundtrip_subject("scunet")
    keras.utils.set_random_seed(0)
    warm_build()(warm_inputs(), training=False)

    build, make_inputs, _ = roundtrip_subject(name)
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
    assert spread == 0.0, f"{name} is not deterministic on a warm GPU: self spread {spread:.6e}"
