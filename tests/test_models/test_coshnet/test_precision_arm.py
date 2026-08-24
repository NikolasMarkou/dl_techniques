"""
R-088 / R-141 precision arm for ``coshnet``.

Before this file existed, ``tests/test_models/test_coshnet`` carried zero
``mixed_float16`` occurrences, and the package RAISED
``InvalidArgumentError: cannot compute Mul as input #1(zero-based) was expected
to be a float tensor but is a half tensor`` from
``layers/shearlet_transform.py:367`` on any mixed-precision forward. The root
cause was in ``layers/``, so every shearlet consumer carried it -- see
decisions.md D-054.

The four parts of the arm and why each is not optional are documented once, in
``tests/test_models/precision_arm_oracle.py``; this file only supplies the
subject.
"""

import numpy as np

from ..precision_arm_oracle import assert_precision_arm


def _build():
    from dl_techniques.models.coshnet.model import create_coshnet
    return create_coshnet("tiny", num_classes=4, input_shape=(32, 32, 3))


def _inputs():
    return np.random.RandomState(0).randn(1, 32, 32, 3).astype("float32")


def test_coshnet_runs_under_mixed_float16():
    """
    MEASURED at the fix (GPU 1):

    ==================  ====================================  ====================
    arm                 before                                after
    ==================  ====================================  ====================
    ``mixed_float16``   RAISE ``InvalidArgumentError``        4 el., nan 0, float16
    ``float32``         OK ``absmax`` 2.797625e-01            unchanged, bit-identical on CPU
    fp16 backward       (unreachable)                         loss 0.063551, 10 vars, 0 None
    ==================  ====================================  ====================
    """
    reports = assert_precision_arm(
        build=_build,
        make_inputs=_inputs,
        rtol_against_float32=2e-2,
    )
    assert reports["mixed_float16"]["dtypes"] == ["float16"]
    assert reports["backward_mixed_float16"]["n_vars"] == 10
    # CoShNet is the model that FORCED the D-055 build-spread control into the
    # oracle. Three consecutive float32 builds in one process, each preceded by
    # ``keras.utils.set_random_seed(0)``, measured absmax 0.2797602 / 0.3045650
    # / 0.2811624 -- so a flat cross-arm rtol reported a "precision finding"
    # that was really a non-seedable initializer. The spread is asserted
    # NON-ZERO here on purpose: if a later change makes CoShNet reproducible,
    # this line fails and the widened tolerance above should be tightened
    # rather than left silently loose.
    assert reports["float32_build_spread"][0] > 0.0
