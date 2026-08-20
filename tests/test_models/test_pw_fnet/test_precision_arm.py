"""
R-088 / R-141 precision arm for ``pw_fnet``.

Before this file existed the directory carried zero ``mixed_float16``
occurrences and ``FFTLayer.call`` RAISED ``TypeError: the `real` and `imag`
components have incorrect types: float16 float16 ... must be one of
[tf.float32, tf.float64]`` at ``layers/fft_layers.py:98``.

Batch 8 routed this as step 5.8 "arm (b)" -- an accepted architectural
limitation, remedy = a documented float32-only declaration. **That routing is
overturned by measurement**: a float32 cast island around ``fft2``/``ifft2``
makes the model fully usable under mixed precision, at the cost of two casts,
and leaves the float32 arm bit-identical. See decisions.md D-054.
"""

import numpy as np

from ..precision_arm_oracle import assert_precision_arm


def _build():
    from dl_techniques.models.pw_fnet.model import create_pw_fnet
    return create_pw_fnet(width=8, middle_blk_num=1,
                          enc_blk_nums=[1, 1], dec_blk_nums=[1, 1])


def _inputs():
    return np.random.RandomState(0).randn(1, 32, 32, 3).astype("float32")


def test_pw_fnet_runs_under_mixed_float16():
    """
    MEASURED at the fix (GPU 1): ``mixed_float16`` RAISE -> three multi-scale
    outputs of 3072 / 768 / 192 elements, nan 0, dtype float16, ``absmax``
    9.734375 / 6.519531 / 3.289062 against a float32 control of
    9.728796 / 6.520255 / 3.289530 (CPU, bit-identical before and after the
    fix); fp16 backward loss 4.283896, 100 vars, 0 ``None``, 0 non-finite.
    """
    reports = assert_precision_arm(
        build=_build,
        make_inputs=_inputs,
        rtol_against_float32=2e-2,
    )
    assert reports["mixed_float16"]["n_tensors"] == 3
    assert reports["backward_mixed_float16"]["n_vars"] == 100
