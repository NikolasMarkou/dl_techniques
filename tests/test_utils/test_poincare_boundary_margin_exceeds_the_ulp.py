"""The Poincare boundary margin must be larger than one ULP of the compute dtype.

Rationale
---------
`PoincareMath.log_map_0` clamps ``||y||`` to ``(1/sqrt(c)) - boundary_eps`` so
``arctanh`` never sees exactly 1.0. That guard was a fixed ``1e-4``. float16's
ULP at 1.0 is **9.765625e-04, 9.77x LARGER**, so the subtraction rounded back to
``1/sqrt(c)``, the clamp became an identity and ``arctanh(1.0) = inf``. MEASURED
before the fix: all three shipped `models/shgcn` classes returned 100% NaN under
``mixed_float16`` (96/96, 36/36 and 5/5 elements) with no warning, while float32
was green.

The instrument here is the RATIO of margin to ULP, not the margin's absolute
size -- a numeric guard is void exactly when it is smaller than the spacing of
the numbers it guards, and "make the constant bigger" is not a fix, it is the
same defect one precision change later.

See decisions.md D-027 (plan-2026-08-19T163559-499b6f0e).
"""

import keras
import numpy as np
import pytest

from dl_techniques.utils.geometry.poincare_math import PoincareMath

# ---------------------------------------------------------------------

DTYPES = ["float16", "float32", "float64"]
CURVATURES = [0.25, 1.0, 4.0]


@pytest.fixture(scope="module")
def math_util() -> PoincareMath:
    return PoincareMath(eps=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("c", CURVATURES)
def test_the_margin_spans_at_least_one_ulp(math_util, dtype, c):
    """A margin below one ULP is arithmetically a no-op."""
    radius = 1.0 / np.sqrt(c)
    margin = float(np.array(math_util.boundary_margin(dtype, radius)))
    ulp = float(np.finfo(dtype).eps) * max(abs(radius), 1.0)
    assert margin > ulp, (
        f"boundary margin {margin:.6e} for dtype={dtype}, c={c} is not larger "
        f"than one ULP ({ulp:.6e}); `radius - margin` rounds back to `radius` "
        f"and the arctanh clamp becomes an identity"
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("c", CURVATURES)
def test_the_clamped_radius_is_strictly_inside_the_ball(math_util, dtype, c):
    """The arithmetic that the ratio test predicts, done in the dtype itself."""
    radius = np.array(1.0 / np.sqrt(c), dtype=dtype)
    margin = np.array(
        float(np.array(math_util.boundary_margin(dtype, float(radius)))),
        dtype=dtype,
    )
    clamped = np.array(radius - margin, dtype=dtype)
    assert clamped < radius, (
        f"(1/sqrt(c)) - margin == 1/sqrt(c) exactly in {dtype} at c={c}: "
        f"{clamped!r} vs {radius!r}"
    )


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_log_map_0_is_finite_at_the_ball_boundary(math_util, dtype):
    """The end-to-end consequence: arctanh must never be handed exactly 1.0."""
    y = keras.ops.convert_to_tensor(
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=dtype), dtype=dtype)
    out = np.array(math_util.log_map_0(y, c=1.0))
    assert np.all(np.isfinite(out)), (
        f"log_map_0 at ||y|| = 1/sqrt(c) is not finite in {dtype}: {out!r}. "
        f"Pre-fix float16 read inf here while float32 read 4.9521313."
    )


def test_the_float32_path_is_unchanged(math_util):
    """Anti-regression: the margin must still be `boundary_eps` in float32.

    In float32 the fixed 1e-4 is ~839x larger than 4 ULP, so the dtype-aware
    max() must select it and float32 numerics must not move at all. This is the
    arm that fails if someone "simplifies" the margin to a pure ULP multiple.
    """
    margin = float(np.array(math_util.boundary_margin("float32", 1.0)))
    assert margin == pytest.approx(math_util.boundary_eps, rel=1e-6), (
        f"float32 margin is {margin:.6e}, not boundary_eps "
        f"({math_util.boundary_eps:.6e}) to within float32 rounding; the float32 "
        f"numerics have moved"
    )
    y = keras.ops.convert_to_tensor(
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"), dtype="float32")
    value = float(np.array(math_util.log_map_0(y, c=1.0)).ravel()[0])
    assert value == pytest.approx(4.9521313, rel=1e-6), (
        f"float32 log_map_0 at the boundary reads {value!r}, not the measured "
        f"pre-fix-and-post-fix 4.9521313"
    )


# ---------------------------------------------------------------------
# Injected-defect / fixed-twin pair.
#
# Reverting the source proves these guards RED, but it proves it with an
# `AttributeError` (the method did not exist before the fix), which is a
# STRUCTURAL red, not a behavioural one. The pair below re-injects the pre-fix
# BEHAVIOUR -- a fixed, dtype-blind margin -- into a subclass, so the predicate
# is shown to fire on the defect itself.
# ---------------------------------------------------------------------


class _DtypeBlindPoincareMath(PoincareMath):
    """The pre-fix behaviour: the margin is `boundary_eps`, whatever the dtype."""

    def boundary_margin(self, dtype, radius):  # noqa: D102 - injected defect
        return self.boundary_eps


def test_the_predicate_fires_on_an_injected_dtype_blind_margin():
    """The injected defect must be REJECTED, by the ULP-ratio assertion."""
    injected = _DtypeBlindPoincareMath(eps=1e-5)
    margin = float(np.array(injected.boundary_margin("float16", 1.0)))
    ulp = float(np.finfo("float16").eps)
    assert margin < ulp, (
        "the injected defect is not actually a defect -- 1e-4 must be smaller "
        f"than float16's ULP {ulp:.6e}"
    )
    with pytest.raises(AssertionError, match="is not larger than one ULP"):
        test_the_margin_spans_at_least_one_ulp(injected, "float16", 1.0)


def test_the_predicate_is_silent_on_the_fixed_twin(math_util):
    """The real implementation must pass the same call, unchanged."""
    test_the_margin_spans_at_least_one_ulp(math_util, "float16", 1.0)
    test_the_margin_spans_at_least_one_ulp(math_util, "float32", 1.0)
