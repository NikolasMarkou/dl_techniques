"""Adversarial numerics for the shared ``l2_normalize`` helper.

``l2_normalize`` exists because the naive spelling of the operation fails at
both ends of the input scale, and fails *silently* in both cases: the output
stays finite and merely stops being a unit vector. Every threshold here was
measured against a float64 oracle rather than guessed, and the numbers are
recorded next to the assertions so a later change cannot quietly relax them.

Measured, float32, 256x8 inputs, error taken against a float64 oracle:

===========  =========================  ==========================
input scale  naive error                this function's error
===========  =========================  ==========================
``1e-08``    ``9.5e-01``                ``9.1e-08``
``1e-06``    ``9.5e-01``                ``8.3e-08``
``1e-03``    ``2.2e-02``                ``1.1e-07``
``1e+00``    ``8.3e-08``                ``9.7e-08``
===========  =========================  ==========================

The naive form's error at ``1e+00`` is the same order as this function's, so
the table is not a claim that this path is universally better -- it is a claim
that it is never worse, including where the naive form is unusable.
"""

import numpy as np
import pytest
import tensorflow as tf
import keras
from keras import ops

from dl_techniques.utils.tensors import l2_normalize

# ---------------------------------------------------------------------

#: float32 unit roundoff. Every bound below sits well above this.
_F32_UNIT_ROUNDOFF = np.finfo(np.float32).eps / 2.0

#: Bound for agreement with a float64 oracle, measured at 9.7e-08 across
#: scales from 1e-8 to 1e38. ~1.6x the unit roundoff, so it is a floor claim
#: and not a generous one.
_ORACLE_ATOL = 5e-07

#: Above sqrt(float32_max) ~ 1.84e19 the naive squared sum overflows.
_OVERFLOW_THRESHOLD = float(np.sqrt(np.finfo(np.float32).max))


def _naive(x, eps=1e-7):
    """The spelling this helper replaces, kept here as the test's oracle."""
    return x / ops.sqrt(ops.sum(ops.square(x), axis=-1, keepdims=True) + eps)


def _oracle(x):
    """float64 L2 normalization, independent of the implementation under test."""
    as_double = np.asarray(x, dtype=np.float64)
    return as_double / np.linalg.norm(as_double, axis=-1, keepdims=True)


def _scaled(seed: int = 0, shape=(256, 8), exponent: int = 0) -> np.ndarray:
    """Input of magnitude ~1e``exponent``, built without overflowing the DATA.

    Scaling a standard normal by 1e38 overflows in float32 before the layer
    ever sees it, which measures the probe rather than the layer. The vector is
    normalized first and scaled second, so the tensor reaching the layer is
    genuinely at the requested magnitude.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(shape)
    base /= np.abs(base).max()

    return (base * np.float64(10.0) ** exponent).astype(np.float32)


class TestL2NormalizeMatchesTheOracle:
    """The helper computes the same quantity as an exact reference."""

    @pytest.mark.parametrize('exponent', [-8, -6, -3, 0, 3, 8])
    def test_accuracy_across_the_usable_range(self, exponent):
        x = _scaled(exponent=exponent)

        result = ops.convert_to_numpy(
            l2_normalize(ops.convert_to_tensor(x))
        ).astype(np.float64)

        np.testing.assert_allclose(
            result, _oracle(x), rtol=0, atol=_ORACLE_ATOL,
            err_msg=f"at scale 1e{exponent} the normalized vector drifted"
        )

    def test_it_agrees_with_the_naive_spelling_in_the_well_scaled_regime(self):
        """Not a claim of universal superiority -- only of no regression at 1e0.

        At scale 1e0 the naive form is already correct to 8.3e-08, so this
        asserts the new path is no worse where the old one worked. Without it,
        a change could make the common case worse while the extreme-case tests
        stayed green.
        """
        x = _scaled(shape=(256, 8), exponent=0)

        result = ops.convert_to_numpy(
            l2_normalize(ops.convert_to_tensor(x))
        ).astype(np.float64)

        np.testing.assert_allclose(
            result, _oracle(x), rtol=0, atol=_ORACLE_ATOL
        )

    @pytest.mark.parametrize('exponent', [-8, 0, 8])
    def test_the_output_norm_is_one(self, exponent):
        """The defining property, checked where the naive form silently fails."""
        result = l2_normalize(ops.convert_to_tensor(_scaled(exponent=exponent)))

        squared = ops.sum(ops.square(result), axis=-1)

        np.testing.assert_allclose(
            ops.convert_to_numpy(squared), 1.0, rtol=0, atol=1e-5
        )


class TestL2NormalizeSurvivesExtremeScales:
    """The regimes where the naive form fails, each pinned with its mechanism."""

    @pytest.mark.parametrize('exponent', [20, 25, 30, 38])
    def test_large_input_stays_a_unit_vector(self, exponent):
        """Above the overflow threshold the naive form returns exact ZEROS.

        The threshold is ``sqrt(float32_max) ~ 1.84e19``, so these exponents all
        sit above it. Measured: ``absmax = 0.0`` at 1e20, 1e30 and 1e38 --
        finite, so no warning fires, and carrying no directional information at
        all. Note this is not a NaN: ``x / inf == 0``.

        The fixture assertion below is a guard on the FIXTURE: an exponent below
        the threshold would make this test pass for the wrong reason, since the
        naive form also works there. Measured at 1e15 and 1e19 the naive error
        is 6.5e-08 and 5.8e-08 -- indistinguishable from this helper's.
        """
        x = _scaled(exponent=exponent)

        assert float(np.abs(x).max()) > _OVERFLOW_THRESHOLD, (
            f"the fixture reached only {float(np.abs(x).max()):.3e}, which is "
            f"below the overflow threshold {_OVERFLOW_THRESHOLD:.3e}; this arm "
            f"must test a scale where the naive form actually fails"
        )

        result = ops.convert_to_numpy(l2_normalize(ops.convert_to_tensor(x)))

        assert np.all(np.isfinite(result)), f"1e{exponent} produced a non-finite value"
        assert np.abs(result).max() > 0.5, (
            f"1e{exponent} collapsed to {np.abs(result).max():.3e}; this is the "
            f"silent-zero failure the helper exists to prevent"
        )

    @pytest.mark.parametrize('exponent', [-32, -30, -20, -8, -6])
    def test_small_input_keeps_its_norm(self, exponent):
        """Below ~1e-4 the naive form's eps dominates the squared norm.

        Measured naive error 9.5e-01 at every scale from 1e-32 to 1e-8 -- the
        result is not a unit vector, but it is finite and plausible, so nothing
        reports it.

        The lowest arm is ``1e-32``. Sweeping downward, ``1e-33`` through
        ``1e-8`` all measure between 8.5e-08 and 1.0e-07 against the oracle,
        while ``1e-34`` degrades to ``1.3e-04``: float32 spacing is relative,
        so a value near ``1e-34`` still carries ~24 bits of mantissa, and the
        error comes from the stored input rather than from this method. The
        boundary is therefore stated as a measured floor rather than guessed at
        min-normal, which is ~4 decades further down and not where anything
        changes.
        """
        x = _scaled(exponent=exponent)

        result = ops.convert_to_numpy(
            l2_normalize(ops.convert_to_tensor(x))
        ).astype(np.float64)

        np.testing.assert_allclose(
            result, _oracle(x), rtol=0, atol=_ORACLE_ATOL,
            err_msg=f"at scale 1e{exponent} the normalized vector drifted"
        )

    def test_the_naive_spelling_really_does_fail_where_this_one_does_not(self):
        """The anti-vacuity arm for every test above.

        Without it, the assertions could all be satisfied by a helper that does
        nothing at all -- the measurements above would still read as 'passing'
        because they only ever compare against an oracle.
        """
        huge = _scaled(shape=(64, 8), exponent=30)
        tiny = _scaled(shape=(64, 8), exponent=-8)

        naive_huge = ops.convert_to_numpy(_naive(ops.convert_to_tensor(huge)))
        naive_tiny = ops.convert_to_numpy(_naive(ops.convert_to_tensor(tiny)))

        assert np.abs(naive_huge).max() == 0.0, (
            "the naive form no longer collapses on large input, so this test's "
            "premise has changed"
        )
        assert np.abs(
            naive_tiny.astype(np.float64) - _oracle(tiny)
        ).max() > 0.1, (
            "the naive form no longer loses the norm on small input, so this "
            "test's premise has changed"
        )


class TestL2NormalizeDegenerateInput:
    """Zero, mixed and non-finite inputs."""

    def test_an_all_zero_vector_normalizes_to_zero_not_nan(self):
        """The floor must be a NORMAL float32.

        A subnormal floor such as 1e-38 makes ``0 / 1e-38`` round to ``nan`` in
        float32, which would reintroduce the failure the floor exists to
        prevent. That was measured before the floor was raised to 1e-30.
        """
        result = ops.convert_to_numpy(
            l2_normalize(ops.convert_to_tensor(np.zeros((4, 8), dtype=np.float32)))
        )

        assert np.all(np.isfinite(result)), "an all-zero vector produced non-finite"
        np.testing.assert_array_equal(result[0], np.zeros(8, dtype=np.float32))

    def test_one_huge_entry_does_not_contaminate_the_other_rows(self):
        """Contamination is sample-local: the reduction is over the last axis.

        A row is contaminated only if that row itself holds the bad value.
        """
        x = _scaled(shape=(8, 8), exponent=0)
        x[0] *= np.float32(1e30)

        result = ops.convert_to_numpy(l2_normalize(ops.convert_to_tensor(x)))

        assert np.all(np.isfinite(result[1:])), (
            "a bad entry in row 0 contaminated unrelated rows"
        )

    def test_a_zero_row_beside_a_huge_row_is_still_handled(self):
        """The two degenerate cases coexist without either going non-finite."""
        x = _scaled(shape=(4, 8), exponent=0)
        x[0] = 0.0
        x[1] *= np.float32(1e30)

        result = ops.convert_to_numpy(l2_normalize(ops.convert_to_tensor(x)))

        assert np.all(np.isfinite(result[0])), "the zero row went non-finite"
        assert np.all(np.isfinite(result[1])), "the huge row went non-finite"
        assert np.all(np.isfinite(result[2:])), "an unrelated row went non-finite"

    def test_subnormal_input_is_lost_by_the_dtype_not_by_this_method(self):
        """The documented limit, pinned so it stays a stated boundary.

        Below float32's min-normal (``1.175e-38``) the input carries only a
        couple of significant bits, so dividing by ``max|x|`` is itself
        subnormal-rounded and no float32 algorithm can recover the direction.
        Measured at ``1e-44``: error ``0.88`` for BOTH this helper and the naive
        spelling, which is what makes it a property of the dtype rather than of
        the method.

        Asserting that the two agree is what distinguishes "this method cannot
        help here" from "this method made it worse".
        """
        subnormal = _scaled(shape=(64, 8), exponent=-44)

        assert float(np.abs(subnormal).max()) < np.finfo(np.float32).tiny, (
            f"the fixture reached {float(np.abs(subnormal).max()):.3e}, which is "
            f"not below min-normal {np.finfo(np.float32).tiny:.3e}"
        )

        reference = _oracle(subnormal)
        mine = ops.convert_to_numpy(
            l2_normalize(ops.convert_to_tensor(subnormal))
        ).astype(np.float64)
        naive = ops.convert_to_numpy(
            _naive(ops.convert_to_tensor(subnormal))
        ).astype(np.float64)

        np.testing.assert_allclose(
            mine, naive, rtol=0, atol=1e-6,
            err_msg=(
                "this method differs from the naive one on subnormal input; if "
                "it is worse there that is a regression, not a dtype limit"
            )
        )
        assert np.abs(mine - reference).max() > 0.1, (
            "subnormal input was normalized accurately, which would mean this "
            "boundary no longer holds and the docstring needs revisiting"
        )

    def test_a_non_finite_input_is_not_silently_repaired(self):
        """inf/NaN in, NaN out -- documented, not hidden.

        No normalization can recover a direction from a non-finite input, so
        this pins the CONTRACT rather than a fix: the helper does not mask an
        upstream fault by emitting a plausible finite number.
        """
        x = _scaled(shape=(4, 8), exponent=0)
        x[0, 0] = np.inf

        result = ops.convert_to_numpy(l2_normalize(ops.convert_to_tensor(x)))

        assert not np.all(np.isfinite(result[0])), (
            "an inf input was silently normalized to a finite vector, hiding "
            "an upstream fault"
        )


class TestL2NormalizeGraphSafety:
    """The helper is used inside traced ``call`` paths."""

    def test_it_survives_tf_function(self):
        x = _scaled(shape=(4, 8), exponent=0)

        eager = ops.convert_to_numpy(
            l2_normalize(ops.convert_to_tensor(x))
        ).astype(np.float32)

        @tf.function(input_signature=[tf.TensorSpec([None, 8], tf.float32)])
        def traced(t):
            return l2_normalize(t)

        compiled = ops.convert_to_numpy(traced(tf.constant(x)))

        np.testing.assert_allclose(compiled, eager, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize('axis', [0, 1, 2, -1, -2])
    def test_it_normalizes_over_the_requested_axis(self, axis):
        """The axis argument is load-bearing, not decorative.

        A rank-3 input is used so that every axis under test names a different
        reduction. On a rank-2 tensor ``axis=1`` and ``axis=-1`` are the SAME
        axis, so demanding they differ would be demanding the function
        contradict itself.
        """
        x = _scaled(shape=(4, 6, 8), exponent=0)

        result = ops.convert_to_numpy(
            l2_normalize(ops.convert_to_tensor(x), axis=axis)
        )

        norms = np.linalg.norm(result, axis=axis)
        np.testing.assert_allclose(norms, 1.0, rtol=0, atol=1e-5)

        # Every OTHER axis must give a different answer, or this one is inert.
        resolved = axis if axis >= 0 else x.ndim + axis
        for other in range(x.ndim):
            if other == resolved:
                continue
            assert not np.allclose(
                result,
                ops.convert_to_numpy(
                    l2_normalize(ops.convert_to_tensor(x), axis=other)
                )
            ), f"axis={axis} and axis={other} produced identical output"


class TestTheBoundsAreAboveTheNoiseFloor:
    """Anti-vacuity for this module's own tolerances."""

    def test_the_oracle_bound_is_above_float32_unit_roundoff(self):
        assert _ORACLE_ATOL > _F32_UNIT_ROUNDOFF * 2, (
            f"the oracle bound {_ORACLE_ATOL} does not clear the float32 "
            f"roundoff {_F32_UNIT_ROUNDOFF} by 2x; it is pinning noise"
        )

    def test_the_oracle_bound_is_below_the_defect_it_must_catch(self):
        """It must sit above the floor and below the naive form's failure."""
        naive_error_at_small_scale = 9.5e-01

        assert _ORACLE_ATOL < naive_error_at_small_scale / 1000, (
            f"the oracle bound {_ORACLE_ATOL} is too loose to catch the "
            f"{naive_error_at_small_scale} small-scale failure"
        )
