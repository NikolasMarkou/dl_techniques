"""Tests for the ``soft_value_range`` map.

Regime honesty is the whole point of this file. Two properties the soft map has as a
REAL-VALUED function are false in float32 and are therefore asserted only inside a
derived regime, never globally:

* **strict interiority** (``lo < y < hi``): the gap ``exp(-beta*d)/beta`` rounds to
  exactly zero once it drops below the local float spacing, so ``y == lo`` EXACTLY for
  ``d >= 0.8`` at ``beta = 25``. What holds unconditionally is *feasibility*
  (``lo <= y <= hi``), including at ``+-1e6``.
* **nonzero gradient outside the interval**: it underflows to exactly ``0.0`` once
  ``d > ~88/beta``. The defining-difference-vs-clipping test therefore pins a
  ``(beta, d)`` pair inside the representable regime and says so.

Measured numbers quoted in the assertions come from the plan's
``findings/measured-numerics.md`` (keras 3.8.0 / TF 2.18, float32 CPU).
"""

import math

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.activations.soft_value_range import soft_value_range

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _np(x) -> np.ndarray:
    """Materialise a backend tensor as a NumPy array."""
    return np.asarray(keras.ops.convert_to_numpy(x))


def _beta(min_value: float, max_value, sharpness: float, relative: bool) -> float:
    """Re-derive the effective ``beta`` the way the implementation does.

    Kept here as an INDEPENDENT restatement of the rule (relative sharpness divides by
    the interval width; it is ignored in one-sided mode) so a test that needs `beta` is
    not simply reading it back out of the code under test.
    """
    if max_value is None or not relative:
        return float(sharpness)
    return float(sharpness) / (float(max_value) - float(min_value))


# ---------------------------------------------------------------------
# Feasibility -- the one unconditional guarantee
# ---------------------------------------------------------------------


class TestFeasibilityIsUnconditional:
    """``y <= hi`` always; ``y >= lo`` up to a QUANTIFIED low-sharpness undershoot.

    ``y >= lo`` is structural in one-sided mode. In two-sided mode the upper branch
    reads the already-lifted value and pulls it back down by at most
    ``log(1 + exp(-beta*(hi-lo))) / beta`` -- exactly ``0.0`` in float32 from
    relative sharpness 20 upward (so at the default 50), but ``0.627`` at sharpness
    1. Asserting a plain ``y >= lo`` here would be asserting something measurably
    false; asserting only the derived bound would not notice a regression at the
    default. Both are asserted.
    """

    @pytest.mark.parametrize("sharpness", [1.0, 50.0, 1000.0])
    def test_two_sided_upper_bound_is_never_crossed(self, sharpness: float) -> None:
        lo, hi = -1.0, 1.0
        x = np.linspace(-50.0, 50.0, 4001).astype("float32")
        y = _np(soft_value_range(x, lo, hi, sharpness=sharpness))
        assert np.all(np.isfinite(y))
        assert np.all(y <= hi), f"max {y.max()!r} crossed hi={hi}"

    @pytest.mark.parametrize("sharpness", [1.0, 50.0, 1000.0])
    def test_two_sided_lower_undershoot_respects_the_derived_bound(
            self, sharpness: float
    ) -> None:
        lo, hi = -1.0, 1.0
        beta = _beta(lo, hi, sharpness, True)
        bound = math.log1p(math.exp(-beta * (hi - lo))) / beta
        x = np.linspace(-50.0, 50.0, 4001).astype("float32")
        y = _np(soft_value_range(x, lo, hi, sharpness=sharpness)).astype(np.float64)
        undershoot = float(max(0.0, lo - y.min()))
        assert undershoot <= bound + 1e-9, (
            f"undershoot {undershoot!r} exceeds log(1+exp(-beta*W))/beta = {bound!r}"
        )

    @pytest.mark.parametrize("sharpness", [20.0, 50.0, 1000.0])
    def test_lower_bound_is_exact_at_usable_sharpness(self, sharpness: float) -> None:
        """From relative sharpness 20 upward the undershoot is exactly 0.0, measured.

        This is the arm that would catch a regression at the DEFAULT setting, which
        the derived-bound test above cannot: at ``sharpness=50`` its bound is
        ``7.7e-24``, i.e. it permits nothing anyway.
        """
        lo, hi = -1.0, 1.0
        x = np.linspace(-50.0, 50.0, 4001).astype("float32")
        y = _np(soft_value_range(x, lo, hi, sharpness=sharpness))
        assert np.all(y >= lo), f"min {y.min()!r} fell below lo={lo}"

    def test_feasible_at_plus_minus_1e6(self) -> None:
        """The magnitude the brief asked about. Feasible -- but NOT strictly interior."""
        lo, hi = -1.0, 1.0
        x = np.array([-1e6, 1e6], dtype="float32")
        y = _np(soft_value_range(x, lo, hi, sharpness=50.0))
        assert np.all(np.isfinite(y))
        assert np.all(y >= lo) and np.all(y <= hi)
        # Documented, not lamented: at this magnitude the map saturates ON the bound.
        assert y[0] == np.float32(lo)
        assert y[1] == np.float32(hi)

    def test_one_sided_lower_bound_is_feasible_everywhere(self) -> None:
        lo = 0.0
        x = np.linspace(-1e6, 1e6, 2001).astype("float32")
        y = _np(soft_value_range(x, lo, None, sharpness=5.0))
        assert np.all(np.isfinite(y))
        assert np.all(y >= lo), f"min {y.min()!r} fell below lo={lo}"


# ---------------------------------------------------------------------
# Strict interiority -- IN-REGIME ONLY, with a DERIVED probe distance
# ---------------------------------------------------------------------


class TestStrictInteriorityInRegime:
    """``lo < y < hi`` holds only while the gap is representable.

    The probe distance is derived from ``beta`` and the dtype's local spacing rather
    than hardcoded, so the test states the regime instead of memorising a magic number.
    """

    def test_strictly_interior_where_the_gap_clears_float_resolution(self) -> None:
        lo, hi, sharpness = -1.0, 1.0, 50.0
        beta = _beta(lo, hi, sharpness, True)  # 25.0

        spacing = float(np.spacing(np.float32(hi)))
        # Gap model (findings/measured-numerics.md): y - lo ~= exp(-beta*d)/beta.
        # Require a 100x margin over the local spacing so the sum lo + gap cannot
        # round back onto lo:  exp(-beta*d)/beta >= 100 * spacing
        #   =>  d <= -log(100 * spacing * beta) / beta
        d_max = -math.log(100.0 * spacing * beta) / beta
        assert d_max > 0.0, "no representable strict-interiority regime exists at all"
        d = 0.5 * d_max  # comfortably inside, not on the edge

        predicted_gap = math.exp(-beta * d) / beta
        assert predicted_gap >= 100.0 * spacing, (
            f"derivation is inconsistent: predicted gap {predicted_gap:.3e} vs "
            f"100*spacing {100.0 * spacing:.3e}"
        )

        x = np.array([lo - d, hi + d], dtype="float32")
        y = _np(soft_value_range(x, lo, hi, sharpness=sharpness))
        assert y[0] > lo, f"lower probe at d={d:.4f} collapsed onto lo exactly"
        assert y[1] < hi, f"upper probe at d={d:.4f} collapsed onto hi exactly"

    def test_far_outside_the_regime_the_output_lands_exactly_on_the_bound(self) -> None:
        """The counterpart claim, pinned so the regime boundary is documented by a test.

        Measured at ``beta=25``: ``y == lo`` exactly for ``d >= 0.8``. This is not a
        defect; it is why the interiority claim above is scoped.
        """
        lo, hi = -1.0, 1.0
        y = _np(soft_value_range(np.array([lo - 0.8], dtype="float32"), lo, hi,
                                 sharpness=50.0))
        assert y[0] == np.float32(lo)


# ---------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------


class TestMonotonicity:

    def test_non_decreasing_on_a_sorted_vector(self) -> None:
        rng = np.random.default_rng(0)
        x = np.sort(rng.standard_normal(10000).astype("float32") * 3.0)
        y = _np(soft_value_range(x, -1.0, 1.0, sharpness=50.0))
        diffs = np.diff(y)
        assert np.all(diffs >= 0.0), f"most negative step {diffs.min()!r}"

    def test_strictly_increasing_in_the_interior_where_resolution_permits(self) -> None:
        """Strictness is only claimed where the step exceeds the local float spacing."""
        x = np.linspace(-0.9, 0.9, 1001).astype("float32")
        y = _np(soft_value_range(x, -1.0, 1.0, sharpness=50.0))
        diffs = np.diff(y)
        spacing = np.spacing(np.abs(y[:-1]).astype(np.float32))
        resolvable = diffs > spacing
        assert resolvable.mean() > 0.99, (
            f"only {resolvable.mean():.3f} of interior steps were resolvable"
        )
        assert np.all(diffs[resolvable] > 0.0)


# ---------------------------------------------------------------------
# Interior bias -- both directions (SC10)
# ---------------------------------------------------------------------


class TestInteriorBiasIsBoundedAndTight:
    """``max|y - x| <= log(2)/beta``, and the bound is APPROACHED, not merely respected.

    The anti-vacuity twin matters: an accidental identity map (``return x``) satisfies
    the upper bound trivially. Only the lower assertion rejects it.
    """

    def test_bias_respects_the_log2_over_beta_bound_in_float64(self) -> None:
        """The bound is a real-valued statement, so it is checked where float noise
        does not dominate. MEASURED here: ``0.027725887222397994`` against
        ``log(2)/25 = 0.027725887222397813`` -- agreement to ~1.8e-16."""
        lo, hi, sharpness = -1.0, 1.0, 50.0
        beta = _beta(lo, hi, sharpness, True)  # 25.0
        bound = math.log(2.0) / beta
        x = np.linspace(lo, hi, 2001).astype("float64")
        measured = float(np.max(np.abs(_np(soft_value_range(x, lo, hi,
                                                            sharpness=sharpness)) - x)))
        assert measured <= bound + 1e-9, (
            f"interior bias {measured!r} exceeded log(2)/beta = {bound!r}"
        )

    def test_bias_respects_the_bound_in_float32_within_one_ulp_of_the_result(self) -> None:
        """Same claim in float32, where the max is attained at ``x = -1`` and the
        RESULT's own rounding adds up to one local ULP.

        MEASURED: ``0.027725934982299805``, i.e. the bound plus ``4.78e-08``, against
        a local spacing of ``1.19e-07`` at ``|y| ~ 1``. The allowance is derived from
        that spacing, not tuned: a wider excess would be a real bias regression.
        """
        lo, hi, sharpness = -1.0, 1.0, 50.0
        beta = _beta(lo, hi, sharpness, True)
        bound = math.log(2.0) / beta
        allowance = float(np.spacing(np.float32(max(abs(lo), abs(hi)))))
        x = np.linspace(lo, hi, 2001).astype("float32")
        measured = float(np.max(np.abs(_np(soft_value_range(x, lo, hi,
                                                            sharpness=sharpness)) - x)))
        assert measured <= bound + allowance, (
            f"interior bias {measured!r} exceeded log(2)/beta = {bound!r} by more "
            f"than one float32 ULP ({allowance!r})"
        )

    def test_bias_actually_approaches_the_bound_so_identity_cannot_pass(self) -> None:
        lo, hi, sharpness = -1.0, 1.0, 50.0
        beta = _beta(lo, hi, sharpness, True)
        bound = math.log(2.0) / beta
        x = np.linspace(lo, hi, 2001).astype("float32")
        measured = float(np.max(np.abs(_np(soft_value_range(x, lo, hi,
                                                            sharpness=sharpness)) - x)))
        # Measured on this stack: 0.02772588722239777 vs bound 0.027725887222397813.
        assert measured >= 0.9 * bound, (
            f"interior bias {measured!r} is far below the tight bound {bound!r}; the "
            f"map is behaving like the identity, so the upper-bound test above is vacuous"
        )


# ---------------------------------------------------------------------
# Convergence to hard clipping
# ---------------------------------------------------------------------


class TestConvergenceToHardClip:

    def test_deviation_from_clip_decreases_monotonically_with_sharpness(self) -> None:
        lo, hi = -1.0, 1.0
        x = np.linspace(-3.0, 3.0, 2001).astype("float32")
        clipped = _np(keras.ops.clip(keras.ops.convert_to_tensor(x), lo, hi))
        deviations = []
        for sharpness in (1.0, 10.0, 100.0, 1000.0):
            y = _np(soft_value_range(x, lo, hi, sharpness=sharpness))
            deviations.append(float(np.max(np.abs(y - clipped))))
        assert all(a > b for a, b in zip(deviations, deviations[1:])), (
            f"deviations not strictly decreasing across sharpness decades: {deviations}"
        )
        # log(2)/beta at sharpness=1000, relative, width 2 => beta=500 => 1.386e-03.
        assert deviations[-1] < 2e-3, f"top-of-range deviation {deviations[-1]!r}"


# ---------------------------------------------------------------------
# One-sided mode
# ---------------------------------------------------------------------


class TestOneSidedMode:

    def test_output_is_above_the_floor_in_the_representable_regime(self) -> None:
        lo, sharpness = 0.0, 5.0
        x = np.linspace(-1.0, 5.0, 601).astype("float32")
        y = _np(soft_value_range(x, lo, None, sharpness=sharpness))
        assert np.all(y >= lo)
        assert y[np.argmin(np.abs(x))] > lo  # near x = 0 the knee is resolvable

    def test_converges_to_the_identity_for_large_positive_inputs(self) -> None:
        lo, sharpness = 0.0, 5.0
        x = np.array([10.0, 100.0, 1000.0], dtype="float32")
        y = _np(soft_value_range(x, lo, None, sharpness=sharpness))
        np.testing.assert_allclose(y, x, rtol=0.0, atol=1e-3)

    def test_relative_sharpness_is_ignored_and_does_not_raise(self) -> None:
        """No width to divide by, so the flag must be inert rather than an error."""
        x = np.linspace(-2.0, 2.0, 101).astype("float32")
        a = _np(soft_value_range(x, 0.0, None, sharpness=5.0, relative_sharpness=True))
        b = _np(soft_value_range(x, 0.0, None, sharpness=5.0, relative_sharpness=False))
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------
# Degenerate interval
# ---------------------------------------------------------------------


class TestDegenerateInterval:

    def test_lo_equals_hi_gives_a_constant_and_warns(self, caplog) -> None:
        x = np.linspace(-5.0, 5.0, 101).astype("float32")
        with caplog.at_level("WARNING", logger="dl"):
            y = _np(soft_value_range(x, 2.0, 2.0, sharpness=50.0))
        assert np.all(np.isfinite(y))
        np.testing.assert_allclose(y, np.full_like(x, 2.0), rtol=0.0, atol=0.0)
        assert any(record.levelname == "WARNING" for record in caplog.records), (
            "the degenerate hi == lo case must log a warning"
        )


# ---------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------


class TestGradients:

    def test_gradient_outside_the_interval_is_nonzero_where_hard_clip_is_zero(self) -> None:
        """The defining difference vs ``keras.ops.clip`` -- pinned INSIDE the regime.

        ``(beta, d) = (25, 1.0)`` is chosen deliberately, not for convenience. The
        derivative decays like ``exp(-beta*d)``, so in float32 it underflows to exactly
        ``0.0`` once ``d > ~88/beta``. Measured on this stack:
        ``beta=25, d=1.0 -> 1.39e-11`` (nonzero), while ``beta=100, d=1.0`` and
        ``beta=25, d=1e6`` are both exactly ``0.0``. A larger ``d`` or a larger
        ``beta`` would make this test assert something FALSE about float32 -- the
        smoothness advantage is real but bounded by the dtype, and that boundary is
        what this pair documents.
        """
        lo, hi, sharpness = -1.0, 1.0, 50.0
        beta = _beta(lo, hi, sharpness, True)
        assert beta == 25.0
        d = 1.0
        assert d < 88.0 / beta, "probe distance must sit inside the representable regime"

        x = tf.constant([lo - d], dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x)
            y = soft_value_range(x, lo, hi, sharpness=sharpness)
        soft_grad = float(_np(tape.gradient(y, x))[0])

        with tf.GradientTape() as tape:
            tape.watch(x)
            y_clip = keras.ops.clip(x, lo, hi)
        clip_grad = float(_np(tape.gradient(y_clip, x))[0])

        assert soft_grad != 0.0, f"soft gradient underflowed to {soft_grad!r}"
        assert soft_grad > 0.0
        assert clip_grad == 0.0, f"hard clip gradient was {clip_grad!r}, expected 0.0"

    def test_gradients_at_1e6_are_finite_but_not_asserted_nonzero(self) -> None:
        """At this magnitude the derivative underflows to exactly 0.0 in float32.

        Asserting nonzero here would assert something measurably false. What IS
        required is that nothing becomes NaN or infinite.
        """
        lo, hi = -1.0, 1.0
        x = tf.constant([-1e6, 1e6], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = soft_value_range(x, lo, hi, sharpness=50.0)
        grads = _np(tape.gradient(y, x))
        assert np.all(np.isfinite(grads)), f"non-finite gradient {grads!r}"
        assert not np.any(np.isnan(grads))

    def test_gradients_at_top_of_range_sharpness_are_finite(self) -> None:
        lo, hi = -1.0, 1.0
        x = tf.constant(np.linspace(-3.0, 3.0, 101), dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = soft_value_range(x, lo, hi, sharpness=1000.0)
        grads = _np(tape.gradient(y, x))
        assert np.all(np.isfinite(grads))
        assert not np.any(np.isnan(grads))

    def test_interior_gradient_is_close_to_one(self) -> None:
        lo, hi = -1.0, 1.0
        x = tf.constant([-0.5, 0.0, 0.5], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = soft_value_range(x, lo, hi, sharpness=50.0)
        grads = _np(tape.gradient(y, x))
        np.testing.assert_allclose(grads, np.ones(3), rtol=0.0, atol=1e-4)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


class TestValidation:

    def test_max_below_min_raises_naming_both_values(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            soft_value_range(np.zeros(3, dtype="float32"), 1.0, -2.0)
        message = str(excinfo.value)
        assert "1.0" in message and "-2.0" in message, message

    def test_zero_sharpness_raises_naming_the_value(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            soft_value_range(np.zeros(3, dtype="float32"), -1.0, 1.0, sharpness=0.0)
        assert "0.0" in str(excinfo.value), str(excinfo.value)

    def test_negative_sharpness_raises_naming_the_value(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            soft_value_range(np.zeros(3, dtype="float32"), -1.0, 1.0, sharpness=-7.5)
        assert "-7.5" in str(excinfo.value), str(excinfo.value)

    def test_equal_bounds_do_not_raise(self) -> None:
        """``hi == lo`` is degenerate, not invalid: it warns and returns the constant."""
        y = _np(soft_value_range(np.zeros(3, dtype="float32"), 1.0, 1.0))
        np.testing.assert_allclose(y, np.full(3, 1.0), rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------
# Dtypes -- the fp16 upcast is the regression guard
# ---------------------------------------------------------------------


class TestDtypes:

    def test_float32_control(self) -> None:
        x = keras.ops.convert_to_tensor(np.linspace(-3.0, 3.0, 51).astype("float32"))
        assert keras.backend.standardize_dtype(x.dtype) == "float32"
        y = soft_value_range(x, -1.0, 1.0, sharpness=50.0)
        assert keras.backend.standardize_dtype(y.dtype) == "float32"
        assert np.all(np.isfinite(_np(y)))

    def test_float64_input_is_not_silently_narrowed(self, float64_policy) -> None:
        """The realised INPUT dtype is asserted, per the fixture's own docstring."""
        x = keras.ops.convert_to_tensor(np.linspace(-3.0, 3.0, 51).astype("float64"))
        assert keras.backend.standardize_dtype(x.dtype) == "float64", (
            "the float64 arm never realised float64; the guard below cannot fail"
        )
        y = soft_value_range(x, -1.0, 1.0, sharpness=50.0)
        assert keras.backend.standardize_dtype(y.dtype) == "float64", (
            "a hardcoded float32 upcast silently narrowed a float64 caller (D-007)"
        )
        assert np.all(np.isfinite(_np(y)))

    def test_mixed_float16_at_extreme_beta_stays_finite(self, mixed_float16_policy) -> None:
        """The fp16 overflow guard: ``beta = 200 / 0.02 = 10000`` and ``beta * x``
        overflows float16 without the widen-when-narrower upcast."""
        x = keras.ops.cast(
            keras.ops.convert_to_tensor(np.linspace(-1.0, 1.0, 201).astype("float32")),
            "float16",
        )
        assert keras.backend.standardize_dtype(x.dtype) == "float16"
        y = soft_value_range(x, -0.01, 0.01, sharpness=200.0, relative_sharpness=True)
        assert keras.backend.standardize_dtype(y.dtype) == "float16"
        y_np = _np(y).astype(np.float64)
        assert np.all(np.isfinite(y_np)), "fp16 overflow: the upcast is missing"
        # The bound is compared CAST TO THE OUTPUT DTYPE: 0.01 is not representable in
        # float16 and rounds to 0.01000213623046875, so a saturated output reads
        # `> 0.01` in float64 while being exactly the float16 image of the bound.
        hi16 = float(np.float16(0.01))
        lo16 = float(np.float16(-0.01))
        assert np.all(y_np >= lo16) and np.all(y_np <= hi16), (
            f"range [{y_np.min()!r}, {y_np.max()!r}] escaped the float16 image of "
            f"the bounds [{lo16!r}, {hi16!r}]"
        )
