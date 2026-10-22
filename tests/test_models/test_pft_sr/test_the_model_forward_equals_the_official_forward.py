"""The whole PFTSR forward equals the official ``PFT.forward``, by value.

``pft_reference_numpy.model_forward`` transcribes ``PFT.forward`` (:1102):
mirror pad to a multiple of ``window_size``, subtract the RGB mean, scale by
``img_range``, ``conv_first``, the patch-embed LayerNorm, the PFTBs (each a
chain of layers, then its own 3x3 conv plus the stage residual) with one
inherited attention chain per shift parity running ACROSS stages, the final
LayerNorm, ``conv_after_body`` plus the shallow features, the tail, divide by
``img_range``, add the mean back, crop.

The input is ``7 x 10``, not a multiple of the window (padded to ``8 x 12``),
and the ``num_topk`` schedule shrinks the kept keys in both chains, inside a
stage and across a stage boundary. float64 on the CPU, so the bound is float64
reassociation. The port calls the repo's exact GELU and LeakyReLU (D-277) and
the oracle uses the true constants, so no TensorFlow float32 rounding is
emulated any more; the bound is ten times the measured worst residual (D-277).
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.super_resolution.pft_sr import PFTSR

from . import pft_reference_numpy as ref
from ..precision_arm_oracle import float64_scope

#: Ten times the measured worst residual (D-277): 2.665e-15 over these 6 cases, CPU
#: and GPU 0, with the port's exact GELU / LeakyReLU and the oracle's true constants.
ATOL = 3e-14

#: 6 layers in stages [2, 4]; window 4 so 16 keys. Chain 0 (stage-local even
#: layers) is global layers 0, 2, 4; chain 1 is 1, 3, 5.
NUM_TOPK = [16, 12, 10, 6, 7, 3]


def _model(**kw):
    cfg = dict(scale=2, embed_dim=16, num_blocks=[2, 4], num_heads=2, window_size=4,
               mlp_ratio=2.0, convffn_kernel_size=3, num_topk=NUM_TOPK)
    cfg.update(kw)
    keras.utils.set_random_seed(0)
    model = PFTSR(**cfg)
    model.build((None, None, None, cfg.get("in_channels", 3)))
    rng = np.random.default_rng(1)
    for v in model.weights:
        if v.path.endswith(("bias", "beta", "gamma", "relative_position_bias_table")):
            v.assign(rng.normal(0.0, 0.2, v.shape) + (1.0 if v.path.endswith("gamma") else 0.0))
    return model


@pytest.mark.parametrize("kw", [
    dict(),
    dict(upsampler="pixelshuffledirect", scale=3),
    dict(include_top=False),
    dict(img_range=255.0),
    dict(in_channels=1),
    dict(num_topk=None),
], ids=["pixelshuffle_x2", "pixelshuffledirect_x3", "headless", "img_range_255",
        "one_channel_no_mean", "dense_schedule"])
def test_the_forward_equals_the_official_forward(kw, golden_reference_device):
    channels = kw.get("in_channels", 3)
    x = np.random.default_rng(2).random((2, 7, 10, channels))
    with keras.device(golden_reference_device), float64_scope():
        model = _model(**kw)
        y = np.asarray(model(x, training=False))
        expect = ref.model_forward(model, x, model.num_topk)
    assert y.shape == expect.shape
    assert float(np.abs(expect).max()) > 1e-2
    np.testing.assert_allclose(y, expect, rtol=0, atol=ATOL)
