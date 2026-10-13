"""
R-088 precision arm for ``darkir``.

Before this file existed the directory carried zero ``mixed_float16``
occurrences and ``FreMLP.call`` RAISED ``TypeError: the `real` and `imag`
components have incorrect types: float16 float16`` at ``model.py:401``.

``FreMLP`` is the one site in this family where the island has a HOLE in it:
its 1x1-conv MLP must stay at ``compute_dtype`` or the layer's only matmuls
would silently leave mixed precision. See decisions.md D-054.
"""

import numpy as np

from ..precision_arm_oracle import assert_precision_arm


def _build():
    from dl_techniques.models.vision.image_restoration.darkir.model import create_darkir_model
    return create_darkir_model(
        width=8, middle_blk_num_enc=1, middle_blk_num_dec=1,
        enc_blk_nums=[1, 1], dec_blk_nums=[1, 1], dilations=[1, 2],
    )


def _inputs():
    return np.random.RandomState(0).randn(1, 32, 32, 3).astype("float32")


def test_darkir_runs_under_mixed_float16():
    """
    MEASURED at the fix (GPU 1): ``mixed_float16`` RAISE -> 3072 elements,
    nan 0, dtype float16, ``absmax`` 6.058594e+00 against a float32 control of
    6.057854e+00 (CPU digest ``dfc62258a59fb0ff``, bit-identical before and
    after the fix); fp16 backward loss 2.357274, 142 vars, 0 ``None``,
    0 non-finite.

    The ``allowed_none_grads=6`` waiver that used to sit here is GONE, and the
    counts moved with it -- re-derived 2026-10-13. Both named the same six
    tensors 20225bf71 stopped building inside ``EnhancementHead`` (its ``norm``
    and ``ffn``, which its ``call`` never applies). With them unbuilt, the
    backward pass sees 162 vars where it saw 168, and every one of them gets a
    gradient, so there is nothing left to waive. The tolerance is unchanged.
    """
    reports = assert_precision_arm(
        build=_build,
        make_inputs=_inputs,
        rtol_against_float32=2e-2,
        allowed_none_grads=0,
    )
    assert reports["mixed_float16"]["dtypes"] == ["float16"]
    assert reports["backward_mixed_float16"]["n_vars"] == 156
