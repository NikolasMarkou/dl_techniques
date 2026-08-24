"""
R-088 / R-141 regression pin for ``shgcn``: the Poincare double-``where``.

The four-part arm itself runs in
``tests/test_models/test_precision_arm_family.py`` (subject ``shgcn``). This
file pins the DEFECT that writing the arm found.

MEASURED at HEAD on the family's own subject (GPU 1):

=========================  =====================  ===================
arm                        before                 after
=========================  =====================  ===================
fp16 non-finite grad norms **2** of 8             **0** of 8
                           (``curvature_theta``
                           in both sHGCN layers)
fp16 ``grad_norm_sum``     ``nan``                finite
float32 control            0 of 8, 5.572658e-02   unchanged
=========================  =====================  ===================

Root cause (decisions.md D-061): ``ops.where`` differentiates BOTH branches,
so the near-origin branch's ``tanh(u)/u`` -- whose derivative carries
``1/u**2`` at the floored ``u = eps = 1e-5``, SUBNORMAL in float16 -- poisoned
the selected branch through ``0 * nan``.
"""

import numpy as np
import tensorflow as tf
from keras import ops

from dl_techniques.utils.geometry.poincare_math import PoincareMath
from ..precision_arm_oracle import run_backward
from ..precision_arm_subjects import SUBJECTS


def _grad_wrt_curvature(dtype: str, single_where: bool) -> float:
    """``d/d_theta`` of ``sum(exp_map_0(0, softplus(theta)))``.

    ``single_where=True`` re-creates the PRE-FIX expression verbatim, so this
    helper is the RED proof: the same input, the same dtype, one ``where``
    instead of two.
    """
    pm = PoincareMath()
    v = tf.Variable(np.zeros((8,), dtype=dtype))
    theta = tf.Variable(np.array(0.54, dtype=dtype))
    with tf.GradientTape() as tape:
        c = ops.softplus(theta)
        if single_where:
            sqrt_c = ops.sqrt(pm._as_compute_dtype(c, v))
            raw, floored = pm.norm_and_floored_norm(v, axis=-1, keepdims=True)
            scale = ops.tanh(sqrt_c * floored) / (sqrt_c * floored)
            y = ops.where(raw < pm.eps, v, v * scale)
        else:
            y = pm.exp_map_0(v, c)
        loss = ops.sum(ops.cast(y, "float32"))
    grad = tape.gradient(loss, theta)
    return float(ops.convert_to_numpy(ops.cast(grad, "float32")))


def test_the_pre_fix_single_where_is_nan_in_float16_and_clean_in_float32():
    """RED: the defect reproduces on demand, and ONLY in half precision."""
    assert np.isnan(_grad_wrt_curvature("float16", single_where=True)), (
        "the pre-fix expression no longer produces a NaN gradient -- this "
        "control can no longer prove the fix is load-bearing"
    )
    assert _grad_wrt_curvature("float32", single_where=True) == 0.0


def test_the_shipped_double_where_is_finite_in_both_dtypes():
    """GREEN: and it agrees EXACTLY with the float32 answer."""
    assert _grad_wrt_curvature("float16", single_where=False) == 0.0
    assert _grad_wrt_curvature("float32", single_where=False) == 0.0


def test_the_forward_is_unchanged_by_the_fix():
    """The double-``where`` must be a BACKWARD-only change.

    When ``raw >= eps`` the inner ``where`` selects the floored norm itself,
    so the forward is bit-identical; that is asserted rather than assumed.
    """
    pm = PoincareMath()
    for dtype in ("float32", "float16"):
        v = ops.convert_to_tensor(
            np.random.RandomState(0).randn(6, 8).astype(dtype))
        c = ops.convert_to_tensor(np.array(1.0, dtype=dtype))
        sqrt_c = ops.sqrt(pm._as_compute_dtype(c, v))
        raw, floored = pm.norm_and_floored_norm(v, axis=-1, keepdims=True)
        legacy = ops.where(
            raw < pm.eps, v,
            v * (ops.tanh(sqrt_c * floored) / (sqrt_c * floored)))
        shipped = pm.exp_map_0(v, c)
        delta = float(np.abs(
            np.asarray(ops.convert_to_numpy(ops.cast(shipped, "float32")))
            - np.asarray(ops.convert_to_numpy(ops.cast(legacy, "float32")))
        ).max())
        assert delta == 0.0, f"{dtype}: forward moved by {delta:.6e}"


def test_the_model_backward_is_finite_under_mixed_float16():
    """The package-level statement the fix exists for."""
    build, make_inputs, _kwargs = SUBJECTS["shgcn"]
    fp16 = run_backward(build, make_inputs, "mixed_float16")
    f32 = run_backward(build, make_inputs, "float32")
    assert fp16["n_nonfinite"] == 0, fp16
    assert f32["n_nonfinite"] == 0, f32
    assert fp16["n_vars"] == f32["n_vars"] == 8
    assert fp16["grad_norm_sum"] > 0.0
