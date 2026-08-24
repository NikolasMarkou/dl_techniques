"""
R-088 / R-141 precision arm for ``fftnet``.

Before this file existed the directory carried zero ``mixed_float16``
occurrences and ``FFTMixer.call`` RAISED ``InvalidArgumentError: cannot compute
AddV2`` at ``model.py:246``: ``magnitude`` is float32 (it is ``abs()`` of a
complex64 tensor, and TensorFlow has no half-precision complex kernel) while
``modrelu_bias`` is an ordinary autocast weight and arrived as float16.
See decisions.md D-054.
"""

import numpy as np

from ..precision_arm_oracle import assert_precision_arm


def _build():
    from dl_techniques.models.fftnet.model import create_fftnet
    return create_fftnet("tiny", image_size=32, patch_size=16)


def _inputs():
    return np.random.RandomState(0).randn(1, 32, 32, 3).astype("float32")


def test_fftnet_runs_under_mixed_float16():
    """
    MEASURED at the fix (GPU 1): ``mixed_float16`` RAISE -> three outputs of
    1920 / 384 / 1536 elements, nan 0, dtype float16; ``float32`` control
    ``absmax`` 4.940991e+00 / 3.601978e+00 / 4.940991e+00, CPU digest
    bit-identical before and after; fp16 backward loss 3.000007, 62 vars,
    0 ``None``, 0 non-finite.
    """
    reports = assert_precision_arm(
        build=_build,
        make_inputs=_inputs,
        rtol_against_float32=2e-2,
    )
    assert reports["mixed_float16"]["n_tensors"] == 3
    assert reports["backward_mixed_float16"]["n_vars"] == 62
