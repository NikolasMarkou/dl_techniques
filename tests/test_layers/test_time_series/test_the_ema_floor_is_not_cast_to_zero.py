"""`ExponentialMovingAverage`'s adjust divisor floor must not be cast to zero.

Guard for `plan-2026-08-31T134711-6271592d` step 8. This site is the clearest
demonstration in the plan that **a same-line cast can be the mechanism of the
defect rather than protection against it**. The code read

    weights_1d = ops.maximum(weights_1d, ops.cast(1e-10, x.dtype))

and `weights_1d` is the divisor of `ema_current / w_t` inside the scan. The cast
casts the LITERAL, not the tensor: under `mixed_float16`, `x.dtype` is `float16`
and `float16(1e-10)` is exactly `0.0`, so the guard degenerates to
`ops.maximum(w, 0.0)` -- it still clips negatives, and it provides exactly zero
protection against the division by zero it was written to prevent. A reviewer
grepping for "is there a cast on this line?" scores it SAFE. It is the opposite.

`w_t = 1 - (1 - alpha)^(t+1)` is exactly zero in float16 whenever
`1 - alpha` rounds to exactly `1.0`, i.e. whenever `alpha < 2.44e-04`, i.e.
whenever `period > 8191` (`alpha = 2 / (period + 1)`). `period` has no upper
bound: `__init__` validates only `period >= 1`. MEASURED at HEAD under
`mixed_float16` with `period=20000`: `[1. inf inf inf inf]`.

The float16 answer is coarse either way -- `1 / 6.10e-05` is 16384, close to
float16's 65504 ceiling -- so the `adjust=True` arm below is deliberately a
SINGLE adjusted step. The point of the fix is that a floored divisor yields a
finite number and a finite gradient; it is not that float16 can represent an EMA
whose adjust weight underflowed.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.time_series.ema_layer import ExponentialMovingAverage
from dl_techniques.utils.dtype_policy import stability_floor

# alpha = 2 / (period + 1) = 1e-04 < 2.44e-04, so float16(1 - alpha) == 1.0.
UNDERFLOWING_PERIOD = 20000
BATCH, FEATURES = 2, 3


class TestTheProtectiveCastIsTheDefect:
    """Anti-vacuity: the cast really does produce zero, in float16 only."""

    def test_casting_the_literal_to_float16_yields_exactly_zero(self):
        assert np.float16(1e-10) == np.float16(0.0)
        # The same cast in float32 -- the dtype the suite runs in -- is benign.
        assert np.float32(1e-10) > np.float32(0.0)

    def test_the_adjust_weight_really_underflows_at_this_period(self):
        alpha = np.float16(2.0 / (UNDERFLOWING_PERIOD + 1.0))
        one_minus_alpha = np.float16(1.0) - alpha
        assert one_minus_alpha == np.float16(1.0)
        assert np.float16(1.0) - one_minus_alpha ** np.float16(2.0) == np.float16(0.0)

    def test_the_policy_floor_is_strictly_positive_in_float16(self):
        assert np.float16(stability_floor("float16", 1e-10)) > np.float16(0.0)
        assert stability_floor("float32", 1e-10) == 1e-10


class TestTheAdjustDivisorIsFloored:

    def test_one_adjusted_step_is_finite(self, dtype_policy):
        layer = ExponentialMovingAverage(period=UNDERFLOWING_PERIOD, adjust=True)
        x = keras.ops.full((BATCH, 2, FEATURES), 0.01, dtype=layer.compute_dtype)

        out = layer(x)

        assert keras.backend.standardize_dtype(out.dtype) == layer.compute_dtype
        if dtype_policy == "mixed_float16":
            assert layer.compute_dtype == "float16"
        values = keras.ops.convert_to_numpy(out)
        assert np.all(np.isfinite(values)), (
            f"ema_current / w_t went non-finite under {dtype_policy}: {values}"
        )

    def test_the_unselected_adjust_branch_does_not_poison_the_gradient(
        self, dtype_policy
    ):
        """`ops.where` evaluates BOTH branches.

        With `adjust=False` the divided branch is discarded, so the forward pass
        is finite even with a zero divisor -- but its `inf` still flows back
        through `where` and turns the gradient into `NaN`. Only a gradient
        exposes this arm.
        """
        layer = ExponentialMovingAverage(period=UNDERFLOWING_PERIOD, adjust=False)
        source = tf.Variable(np.full((BATCH, 5, FEATURES), 0.01, np.float32))

        with tf.GradientTape() as tape:
            tape.watch(source)
            out = layer(keras.ops.cast(source, layer.compute_dtype))
            loss = keras.ops.sum(keras.ops.cast(out, "float32"))
        grad = tape.gradient(loss, source)

        assert keras.backend.standardize_dtype(out.dtype) == layer.compute_dtype
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))
        assert grad is not None
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(grad))), (
            f"d(ema)/d(inputs) went non-finite under {dtype_policy}"
        )

    def test_an_ordinary_period_is_untouched_by_the_floor(self, dtype_policy):
        """The floor must not move the answer where it was never binding.

        At `period=25` (the class default) every adjust weight is far above any
        dtype's floor, so the output must match the float32 reference closely in
        every policy. Without this arm the guard could pass on a layer that had
        started returning a constant.
        """
        layer = ExponentialMovingAverage(period=25, adjust=True)
        rng = np.random.default_rng(0)
        raw = rng.normal(size=(BATCH, 6, FEATURES)).astype(np.float32)

        out = keras.ops.convert_to_numpy(
            layer(keras.ops.cast(raw, layer.compute_dtype))
        ).astype(np.float32)

        alpha = 2.0 / 26.0
        reference = np.empty_like(raw)
        reference[:, 0, :] = raw[:, 0, :]
        for t in range(1, raw.shape[1]):
            reference[:, t, :] = (
                alpha * raw[:, t, :] + (1.0 - alpha) * reference[:, t - 1, :]
            ) / (1.0 - (1.0 - alpha) ** (t + 1))
        np.testing.assert_allclose(out, reference, rtol=2e-2, atol=2e-2)
