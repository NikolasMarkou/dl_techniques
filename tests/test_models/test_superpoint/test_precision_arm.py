"""
R-088 / R-141 regression pin for ``superpoint``: the float16 bicubic gradient.

The four-part arm itself runs in
``tests/test_models/test_precision_arm_family.py`` (subject ``superpoint``).
This file pins the DEFECT that writing the arm found, which no forward
assertion could see.

MEASURED at HEAD, on the family's own subject (GPU 1):

======================  ==============================  ==================
arm                     before                          after
======================  ==============================  ==================
fp16 ``None`` gradients **2** of 183                    **0** of 183
                        (``descriptor_head/kernel``,
                        ``descriptor_head/bias``)
float32 ``None`` grads  0 of 183                        0 of 183
fp16 ``grad_norm_sum``  1.078201e+02                    unchanged in scale
======================  ==============================  ==================

The whole descriptor head -- the half of SuperPoint that produces descriptors
-- received NO gradient under ``mixed_float16``, silently, because TensorFlow
registers no ``ResizeBicubic`` gradient for float16 and returns ``None``
rather than raising. See decisions.md D-060.
"""

import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from ..precision_arm_oracle import precision_policy, run_backward
from ..precision_arm_subjects import SUBJECTS


@pytest.mark.parametrize("interpolation,expect_none", [
    ("bicubic", True),
    ("bilinear", False),
    ("nearest", False),
])
def test_the_framework_trap_this_fix_works_around_still_exists(
        interpolation, expect_none):
    """The RED half: the pre-fix expression, run directly.

    ``ops.image.resize`` on a float16 tensor is EXACTLY what the model did
    before D-060. If TensorFlow ever registers the missing gradient this test
    fails, and the float32 round-trip in ``model.py`` should then be removed
    rather than left as unexplained ballast.
    """
    x = tf.Variable(np.random.RandomState(0).randn(1, 4, 4, 3).astype("float16"))
    with tf.GradientTape() as tape:
        y = ops.image.resize(x, size=(8, 8), interpolation=interpolation)
        loss = ops.sum(ops.cast(y, "float32"))
    grad = tape.gradient(loss, x)
    assert (grad is None) is expect_none, (
        f"float16 {interpolation} resize gradient is "
        f"{'None' if grad is None else 'present'}, expected the opposite"
    )
    # The float32 control: every interpolation has a gradient there, so the
    # reading above is about the DTYPE and not about the interpolation.
    x32 = tf.Variable(np.random.RandomState(0).randn(1, 4, 4, 3).astype("float32"))
    with tf.GradientTape() as tape:
        loss = ops.sum(ops.image.resize(x32, size=(8, 8),
                                        interpolation=interpolation))
    assert tape.gradient(loss, x32) is not None


def test_the_descriptor_head_receives_a_gradient_under_mixed_float16():
    """The GREEN half: the two named variables are no longer disconnected."""
    build, make_inputs, _kwargs = SUBJECTS["superpoint"]
    named = {}
    for policy in ("mixed_float16", "float32"):
        with precision_policy(policy):
            import keras
            keras.utils.set_random_seed(0)
            model = build()
            with tf.GradientTape() as tape:
                out = model(make_inputs(), training=True)
                loss = sum(ops.mean(ops.square(ops.cast(t, "float32")))
                           for t in out.values())
            grads = tape.gradient(loss, model.trainable_variables)
            named[policy] = {
                v.path for v, g in zip(model.trainable_variables, grads)
                if g is None
            }
    descriptor = {p for p in named["mixed_float16"] if "descriptor_head" in p}
    assert not descriptor, (
        f"descriptor head still disconnected under mixed_float16: {descriptor}")
    assert named["mixed_float16"] == named["float32"], (
        "the two arms disagree on WHICH variables are unreachable: "
        f"fp16 {named['mixed_float16']} vs float32 {named['float32']}"
    )


def test_the_backward_none_count_matches_the_float32_control():
    """The number the family's arm asserts, stated here as a measurement."""
    build, make_inputs, _kwargs = SUBJECTS["superpoint"]
    fp16 = run_backward(build, make_inputs, "mixed_float16")
    f32 = run_backward(build, make_inputs, "float32")
    assert fp16["n_none"] == f32["n_none"] == 0
    assert fp16["n_vars"] == f32["n_vars"]
    assert fp16["grad_norm_sum"] > 0.0
