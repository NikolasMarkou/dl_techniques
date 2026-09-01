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

import functools
import math

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.activations.soft_value_range import (
    SoftValueRange,
    soft_value_range,
)

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


# =====================================================================
# The `SoftValueRange` Layer
# =====================================================================
#
# The Layer is a thin adapter over the function tested above: `call` delegates,
# `__init__` shares the function's ONE validator. So these classes deliberately do
# NOT re-test the numerics -- they test the Layer contract (guide v2 § 16.3): config
# round trip, `.keras` round trip on values, unbuilt `compute_output_shape`,
# degenerate shapes, the three dtype arms, XLA-vs-eager, knob pinning, and that the
# thing actually trains.
#
# NOTE ON WEIGHTS: `SoftValueRange` is genuinely stateless -- zero weights, zero
# sub-layers. Guide v2's per-variable gradient-flow rule and its
# `len(trainable_variables) > 0` anti-vacuity assertion are therefore N/A and are
# NOT written here (writing them would produce guards that cannot fail). The one
# weight-related assertion the checklist mandates -- weight values equal at
# `atol=0.0` before the loaded model's first call -- IS written, and its docstring
# records that it is vacuous for this layer rather than omitting it silently.


class TestLayerConstructionAndKnobs:
    """Every constructor knob pinned: its default, and that changing it is observable.

    A default that silently changed would alter the numerics of every existing call
    site. Both halves are needed: the value assertions catch a changed default, the
    sensitivity assertions catch a knob that stopped being wired to anything.
    """

    def test_defaults_are_pinned(self) -> None:
        layer = SoftValueRange(min_value=-1.0)
        assert layer.min_value == -1.0
        assert layer.max_value is None
        assert layer.sharpness == 50.0
        assert layer.relative_sharpness is True

    def test_supports_masking_is_enabled(self) -> None:
        """Elementwise and shape-preserving, so a mask must pass through unchanged."""
        assert SoftValueRange(min_value=-1.0, max_value=1.0).supports_masking is True

    def test_the_layer_owns_no_weights(self) -> None:
        layer = SoftValueRange(min_value=-1.0, max_value=1.0)
        layer.build((None, 8))
        assert layer.built is True
        assert len(layer.weights) == 0
        assert len(layer.trainable_variables) == 0

    def test_min_value_changes_the_output(self) -> None:
        x = np.linspace(-3.0, 3.0, 51).astype("float32")[None, :]
        a = _np(SoftValueRange(min_value=-1.0, max_value=1.0)(x))
        b = _np(SoftValueRange(min_value=-0.5, max_value=1.0)(x))
        assert not np.allclose(a, b), "min_value is not wired to the output"

    def test_max_value_changes_the_output(self) -> None:
        x = np.linspace(-3.0, 3.0, 51).astype("float32")[None, :]
        a = _np(SoftValueRange(min_value=-1.0, max_value=1.0)(x))
        b = _np(SoftValueRange(min_value=-1.0, max_value=0.5)(x))
        assert not np.allclose(a, b), "max_value is not wired to the output"

    def test_max_value_none_selects_one_sided_mode(self) -> None:
        x = np.linspace(-3.0, 3.0, 51).astype("float32")[None, :]
        y = _np(SoftValueRange(min_value=-1.0)(x))
        assert np.all(y >= -1.0)
        assert y.max() > 1.0, "one-sided mode must have no ceiling"

    def test_sharpness_changes_the_output(self) -> None:
        x = np.linspace(-3.0, 3.0, 51).astype("float32")[None, :]
        a = _np(SoftValueRange(min_value=-1.0, max_value=1.0, sharpness=5.0)(x))
        b = _np(SoftValueRange(min_value=-1.0, max_value=1.0, sharpness=50.0)(x))
        assert not np.allclose(a, b), "sharpness is not wired to the output"

    def test_relative_sharpness_changes_the_output_in_two_sided_mode(self) -> None:
        """Width 2, so relative gives ``beta = 25`` and absolute gives ``beta = 50``."""
        x = np.linspace(-3.0, 3.0, 51).astype("float32")[None, :]
        a = _np(SoftValueRange(min_value=-1.0, max_value=1.0, sharpness=50.0,
                               relative_sharpness=True)(x))
        b = _np(SoftValueRange(min_value=-1.0, max_value=1.0, sharpness=50.0,
                               relative_sharpness=False)(x))
        assert not np.allclose(a, b), "relative_sharpness is not wired to the output"

    def test_validation_is_shared_with_the_function(self) -> None:
        """The Layer must reject exactly what the function rejects, and say so.

        The checks live in ONE private validator called by both; a second copy could
        drift into accepting something the function refuses while both suites stayed
        green.
        """
        with pytest.raises(ValueError) as excinfo:
            SoftValueRange(min_value=-1.0, max_value=1.0, sharpness=0.0)
        assert "0.0" in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            SoftValueRange(min_value=-1.0, max_value=1.0, sharpness=-7.5)
        assert "-7.5" in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            SoftValueRange(min_value=1.0, max_value=-2.0)
        message = str(excinfo.value)
        assert "1.0" in message and "-2.0" in message, message

    def test_repr_names_the_interval(self) -> None:
        text = repr(SoftValueRange(min_value=-1.0, max_value=1.0, sharpness=10.0))
        assert "SoftValueRange" in text
        assert "-1.0" in text and "1.0" in text and "10.0" in text


class TestLayerForwardPass:

    def test_forward_is_finite_and_shape_preserving(self) -> None:
        x = np.random.default_rng(0).standard_normal((4, 6, 8)).astype("float32") * 5.0
        layer = SoftValueRange(min_value=-1.0, max_value=1.0)
        y = layer(x)
        assert tuple(y.shape) == (4, 6, 8)
        y_np = _np(y)
        assert np.all(keras.ops.convert_to_numpy(keras.ops.all(keras.ops.isfinite(y))))
        assert np.all(y_np >= -1.0) and np.all(y_np <= 1.0)

    def test_the_layer_agrees_with_the_function_exactly(self) -> None:
        """`call` delegates, so this must be bit-identical, not merely close."""
        x = np.linspace(-5.0, 5.0, 101).astype("float32")[None, :]
        layer = SoftValueRange(min_value=-2.0, max_value=3.0, sharpness=12.0,
                               relative_sharpness=False)
        direct = _np(soft_value_range(x, -2.0, 3.0, sharpness=12.0,
                                      relative_sharpness=False))
        np.testing.assert_array_equal(_np(layer(x)), direct)

    def test_training_flag_does_not_change_the_output(self) -> None:
        x = np.linspace(-3.0, 3.0, 51).astype("float32")[None, :]
        layer = SoftValueRange(min_value=-1.0, max_value=1.0)
        np.testing.assert_array_equal(
            _np(layer(x, training=True)), _np(layer(x, training=False))
        )

    def test_a_mask_passes_straight_through(self) -> None:
        inputs = keras.Input(shape=(5,))
        masked = keras.layers.Masking(mask_value=0.0)(
            keras.layers.Reshape((5, 1))(inputs)
        )
        out = SoftValueRange(min_value=-1.0, max_value=1.0)(masked)
        assert out._keras_mask is not None, "the mask was dropped by the layer"


class TestComputeOutputShape:

    def test_unbuilt_instance_from_stored_config(self) -> None:
        """The mandated case: it must answer from configuration alone, unbuilt."""
        layer = SoftValueRange.from_config(
            SoftValueRange(min_value=-1.0, max_value=1.0).get_config()
        )
        assert layer.built is False
        assert layer.compute_output_shape((None, 7)) == (None, 7)
        assert layer.compute_output_shape((3, 4, 5)) == (3, 4, 5)
        assert layer.built is False, "compute_output_shape must not build the layer"

    def test_matches_the_realised_forward_shape(self) -> None:
        layer = SoftValueRange(min_value=-1.0, max_value=1.0)
        x = np.zeros((2, 3, 4), dtype="float32")
        assert tuple(layer.compute_output_shape(x.shape)) == tuple(layer(x).shape)


class TestSerialization:

    def test_get_config_from_config_reproduces_every_parameter(self) -> None:
        original = SoftValueRange(min_value=-2.5, max_value=4.0, sharpness=17.0,
                                  relative_sharpness=False, name="svr")
        config = original.get_config()
        for key in ("min_value", "max_value", "sharpness", "relative_sharpness"):
            assert key in config, f"{key} missing from get_config()"
        restored = SoftValueRange.from_config(config)
        assert restored.min_value == original.min_value
        assert restored.max_value == original.max_value
        assert restored.sharpness == original.sharpness
        assert restored.relative_sharpness == original.relative_sharpness
        assert restored.name == original.name

    def test_get_config_round_trips_the_one_sided_none(self) -> None:
        restored = SoftValueRange.from_config(
            SoftValueRange(min_value=0.0).get_config()
        )
        assert restored.max_value is None

    def test_keras_round_trip_matches_on_values(self, tmp_path) -> None:
        """`.keras` round trip compared on VALUES at ``rtol=0``, ``training=False``.

        Also performs the mandated weight-value comparison at ``atol=0.0`` BEFORE the
        loaded model's first call. **That comparison is VACUOUS here**: the layer is
        genuinely stateless, so both sides carry zero weights and the assertion can
        never fail. It is written rather than omitted so the absence is a recorded
        fact, not a silent gap -- see this section's header comment.
        """
        inputs = keras.Input(shape=(8,))
        hidden = keras.layers.Dense(4, name="dense")(inputs)
        outputs = SoftValueRange(min_value=-1.0, max_value=1.0, sharpness=13.0,
                                 relative_sharpness=False, name="svr")(hidden)
        model = keras.Model(inputs, outputs)

        x = np.random.default_rng(1).standard_normal((6, 8)).astype("float32") * 3.0
        expected = _np(model(x, training=False))

        path = tmp_path / "svr.keras"
        model.save(path)
        loaded = keras.models.load_model(path)

        svr_before = loaded.get_layer("svr")
        assert len(svr_before.weights) == 0
        np.testing.assert_allclose(
            np.concatenate([_np(w).ravel() for w in svr_before.weights] or [np.zeros(0)]),
            np.concatenate([_np(w).ravel() for w in
                            model.get_layer("svr").weights] or [np.zeros(0)]),
            rtol=0.0, atol=0.0,
        )

        actual = _np(loaded(x, training=False))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

        # A round trip compares the model with ITSELF, so it is satisfied by any
        # call -- including an identity one. This pins that the reloaded model is
        # still doing the bounding: the Dense output escapes [-1, 1], the model
        # output does not.
        raw = _np(model.get_layer("dense")(keras.ops.convert_to_tensor(x)))
        assert raw.min() < -1.0 or raw.max() > 1.0, (
            "the probe input never left the interval, so the bound assertion below "
            "is vacuous"
        )
        assert np.all(actual >= -1.0) and np.all(actual <= 1.0)

        restored = loaded.get_layer("svr")
        assert restored.min_value == -1.0
        assert restored.max_value == 1.0
        assert restored.sharpness == 13.0
        assert restored.relative_sharpness is False

    def test_the_layer_is_serializable_in_an_activation_slot(self, tmp_path) -> None:
        """An ``activation=`` slot must survive a save/load -- hence a LAYER, not a
        ``functools.partial``.

        ``soft_value_range`` takes required parameters beyond ``x``, so binding them
        would normally reach for ``functools.partial``; Keras cannot serialize one,
        and the model would reload with a broken activation reference. The registered
        Layer is callable AND carries its own config, so it round trips.
        """
        inputs = keras.Input(shape=(8,))
        outputs = keras.layers.Dense(
            4,
            activation=SoftValueRange(min_value=-1.0, max_value=1.0, sharpness=20.0),
            name="dense",
        )(inputs)
        model = keras.Model(inputs, outputs)

        x = np.random.default_rng(2).standard_normal((5, 8)).astype("float32") * 4.0
        expected = _np(model(x, training=False))

        path = tmp_path / "svr_activation.keras"
        model.save(path)
        loaded = keras.models.load_model(path)

        activation = loaded.get_layer("dense").activation
        assert isinstance(activation, SoftValueRange), (
            f"the activation reloaded as {type(activation)!r}, not a SoftValueRange"
        )
        assert activation.sharpness == 20.0
        reloaded = _np(loaded(x, training=False))
        np.testing.assert_allclose(reloaded, expected, rtol=0.0, atol=0.0)
        # Same anti-vacuity concern as the round trip above: prove the reloaded
        # activation still bounds, rather than merely agreeing with itself.
        assert np.all(reloaded >= -1.0) and np.all(reloaded <= 1.0)
        no_activation = keras.layers.Dense(4)
        no_activation.build((None, 8))
        no_activation.set_weights(loaded.get_layer("dense").get_weights())
        raw = _np(no_activation(x))
        assert raw.min() < -1.0 or raw.max() > 1.0, (
            "the pre-activation never left the interval; the bound assertion is vacuous"
        )

    def test_a_bare_partial_in_the_same_slot_is_not_serializable(self) -> None:
        """The counterpart pin for the docstring's warning, so the claim is tested.

        This asserts the LIMITATION, not a feature: if a future Keras learns to
        serialize a bare ``functools.partial``, this test XFAILs loudly and the
        module docstring's advice can be revisited.
        """
        activation = functools.partial(soft_value_range, min_value=-1.0, max_value=1.0)
        # `TypeError` specifically, not bare `Exception`: a serializer that crashed
        # for some other reason would prove nothing about `partial`.
        with pytest.raises(TypeError) as excinfo:
            keras.activations.serialize(activation)
        assert "functools.partial" in str(excinfo.value), str(excinfo.value)


class TestDegenerateShapes:

    @pytest.mark.parametrize("shape", [(0, 4), (1, 4), (2, 0), (2, 1)])
    def test_degenerate_static_lengths(self, shape) -> None:
        layer = SoftValueRange(min_value=-1.0, max_value=1.0)
        x = np.zeros(shape, dtype="float32")
        y = layer(x)
        assert tuple(y.shape) == shape
        y_np = _np(y)
        assert np.all(np.isfinite(y_np)) if y_np.size else True

    def test_symbolic_trace_with_an_unknown_batch(self) -> None:
        """A `TensorSpec([None, ...])` trace: nothing in `call` may read a shape."""
        layer = SoftValueRange(min_value=-1.0, max_value=1.0)

        @tf.function(input_signature=[tf.TensorSpec([None, 6], tf.float32)])
        def _traced(t):
            return layer(t)

        for n in (0, 1, 7):
            x = np.random.default_rng(3).standard_normal((n, 6)).astype("float32")
            y = _np(_traced(tf.constant(x)))
            assert y.shape == (n, 6)
            assert np.all(np.isfinite(y)) if y.size else True

    def test_functional_model_with_an_unknown_batch(self) -> None:
        inputs = keras.Input(shape=(6,))
        outputs = SoftValueRange(min_value=-1.0, max_value=1.0)(inputs)
        model = keras.Model(inputs, outputs)
        assert tuple(model.output_shape) == (None, 6)


class TestLayerDtypes:
    """The three mandated arms. `float32` is the control, not a formality: it is what
    proves the other two are measuring a dtype effect and not a broken layer."""

    def test_float32_control(self) -> None:
        x = keras.ops.convert_to_tensor(
            np.linspace(-3.0, 3.0, 51).astype("float32")[None, :]
        )
        assert keras.backend.standardize_dtype(x.dtype) == "float32"
        layer = SoftValueRange(min_value=-1.0, max_value=1.0)
        y = layer(x)
        assert keras.backend.standardize_dtype(y.dtype) == "float32"
        assert np.all(keras.ops.convert_to_numpy(keras.ops.all(keras.ops.isfinite(y))))

    def test_float64_is_not_silently_narrowed(self, float64_policy) -> None:
        """Built INSIDE the test, and the REALISED input dtype is asserted first."""
        x = keras.ops.convert_to_tensor(
            np.linspace(-3.0, 3.0, 51).astype("float64")[None, :]
        )
        assert keras.backend.standardize_dtype(x.dtype) == "float64", (
            "the float64 arm never realised float64; the guard below cannot fail"
        )
        layer = SoftValueRange(min_value=-1.0, max_value=1.0)
        y = layer(x)
        assert keras.backend.standardize_dtype(y.dtype) == "float64"
        y_np = _np(y)
        assert np.all(np.isfinite(y_np))
        assert np.all(y_np >= -1.0) and np.all(y_np <= 1.0)

    def test_mixed_float16_at_extreme_beta_stays_finite(
            self, mixed_float16_policy
    ) -> None:
        """The fp16 regression arm: ``beta = 200 / 0.02 = 10000``, which overflows
        float16 in ``beta * x`` without the function's widen-when-narrower upcast.

        The bounds are compared CAST TO THE OUTPUT DTYPE: ``0.01`` is not
        representable in float16 and rounds UP to ``0.010002136230468750``, so a
        saturated output reads ``> 0.01`` in float64 while being exactly the float16
        image of the bound.
        """
        layer = SoftValueRange(min_value=-0.01, max_value=0.01, sharpness=200.0)
        x = keras.ops.cast(
            keras.ops.convert_to_tensor(
                np.linspace(-1.0, 1.0, 201).astype("float32")[None, :]
            ),
            "float16",
        )
        y = layer(x)
        assert keras.backend.standardize_dtype(y.dtype) == "float16"
        y_np = _np(y).astype(np.float64)
        assert np.all(np.isfinite(y_np)), "fp16 overflow: the upcast is missing"
        hi16, lo16 = float(np.float16(0.01)), float(np.float16(-0.01))
        assert np.all(y_np >= lo16) and np.all(y_np <= hi16), (
            f"range [{y_np.min()!r}, {y_np.max()!r}] escaped the float16 image of "
            f"the bounds [{lo16!r}, {hi16!r}]"
        )

    def test_an_explicit_float32_layer_dtype_keeps_the_output_wide(
            self, mixed_float16_policy
    ) -> None:
        """Pins the docstring's mixed-precision advice: `dtype="float32"` is what
        keeps the OUTPUT at full precision. The internal upcast is a numerical guard,
        not a dtype promise, so under the global policy the output is float16."""
        x = keras.ops.convert_to_tensor(
            np.linspace(-3.0, 3.0, 51).astype("float32")[None, :]
        )
        assert keras.backend.standardize_dtype(
            SoftValueRange(min_value=-1.0, max_value=1.0)(x).dtype) == "float16"
        assert keras.backend.standardize_dtype(
            SoftValueRange(min_value=-1.0, max_value=1.0, dtype="float32")(x).dtype
        ) == "float32"


class TestXlaMatchesEager:

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_value": -1.0, "max_value": 1.0},
            {"min_value": -1.0, "max_value": 1.0, "sharpness": 5.0},
            {"min_value": 0.0},
        ],
    )
    def test_jit_compiled_call_agrees_with_eager(
            self, assert_xla_matches_eager, kwargs
    ) -> None:
        """`fit()` runs a traced `jit_compile` graph, not eager; eager-only is not a fix.

        atol is 1e-6: the map is a chain of elementwise ops on values of order 1, so
        the only legitimate disagreement is float32 rounding at ~1.19e-07 per op.
        Measured deviation on this stack is 0.0 for all three arms.
        """
        layer = SoftValueRange(**kwargs)
        x = np.linspace(-5.0, 5.0, 128).astype("float32").reshape(8, 16)
        assert_xla_matches_eager(layer, x, 1e-6, f"SoftValueRange({kwargs})")


class TestTrainingSmoke:

    def test_a_soft_value_range_head_trains(self) -> None:
        """Regression smoke: a bounded head must actually reduce the loss.

        The targets sit strictly INSIDE ``[-1, 1]``, so the model can reach them: a
        hard `clip` head could not, because samples driven outside the box would stop
        receiving gradient. Sharpness 50 (relative, width 2 -> ``beta = 25``) keeps
        the interior bias at ``log(2)/25 = 0.0277``, well below the signal.
        """
        keras.utils.set_random_seed(17)
        rng = np.random.default_rng(17)
        x = rng.standard_normal((256, 8)).astype("float32")
        w = rng.standard_normal((8, 1)).astype("float32")
        y = np.tanh(x @ w).astype("float32") * 0.5  # inside [-1, 1] by construction

        model = keras.Sequential([
            keras.Input(shape=(8,)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
            SoftValueRange(min_value=-1.0, max_value=1.0, sharpness=50.0),
        ])
        model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")
        history = model.fit(x, y, epochs=8, batch_size=32, verbose=0)

        losses = history.history["loss"]
        assert all(np.isfinite(losses)), f"non-finite loss: {losses}"
        assert losses[-1] < losses[0], (
            f"loss did not decrease: first {losses[0]!r}, last {losses[-1]!r}"
        )
        predictions = _np(model(x, training=False))
        assert np.all(np.isfinite(predictions))
        assert np.all(predictions >= -1.0) and np.all(predictions <= 1.0)
