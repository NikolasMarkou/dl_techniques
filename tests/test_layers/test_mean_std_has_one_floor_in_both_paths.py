"""``mean_std`` gives the same standard deviation with and without a mask.

Guard for plan-2026-10-14T062616-81938e81 step 5.4 (D-203). Measured at
69b68fe5a: a constant sequence gave std 0.0 unmasked and 0.001 masked
(``sqrt(0 + 1e-6)``; 0.001007 in float16, where the literal 1e-6 is
subnormal), and the unmasked path's gradient was NaN there. Both paths now use
``sqrt(maximum(variance, floor))`` with population variance (ddof 0), computed
in the accumulation dtype.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.sequence_pooling import SequencePooling

B, S, D = 2, 6, 4


def _std(layer, x, mask=None):
    return np.asarray(keras.ops.convert_to_numpy(layer(x, mask=mask)))[:, D:]


def test_masked_all_keep_equals_unmasked_and_is_ddof_zero():
    x = np.random.default_rng(0).normal(size=(B, S, D)).astype("float32")
    layer = SequencePooling(strategy="mean_std")
    unmasked = _std(layer, x)
    masked = _std(layer, x, np.ones((B, S), "float32"))
    np.testing.assert_allclose(masked, unmasked, rtol=1e-6, atol=0)
    np.testing.assert_allclose(unmasked, x.std(axis=1, ddof=0), rtol=1e-6, atol=0)


def test_a_masked_row_is_the_std_of_its_kept_positions():
    x = np.random.default_rng(1).normal(size=(B, S, D)).astype("float32")
    mask = np.ones((B, S), "float32")
    mask[0, 4:] = 0
    std = _std(SequencePooling(strategy="mean_std"), x, mask)
    np.testing.assert_allclose(std[0], x[0, :4].std(axis=0), rtol=1e-5, atol=0)
    np.testing.assert_allclose(std[1], x[1].std(axis=0), rtol=1e-5, atol=0)


@pytest.mark.parametrize("policy", ["float32", "mixed_float16"])
def test_a_constant_sequence_gives_the_same_floor_masked_and_unmasked(policy):
    previous = keras.config.dtype_policy()
    keras.config.set_dtype_policy(policy)
    try:
        layer = SequencePooling(strategy="mean_std")
        x = np.full((B, S, D), 3.0, "float32")
        unmasked = _std(layer, x)
        masked = _std(layer, x, np.ones((B, S), "float32"))
    finally:
        keras.config.set_dtype_policy(previous)
    # sqrt(1e-6), the float32 floor, in both paths and both policies.
    np.testing.assert_allclose(unmasked.astype("float32"), 1e-3, rtol=1e-3, atol=0)
    np.testing.assert_array_equal(unmasked, masked)


@pytest.mark.parametrize("masked", [False, True])
def test_the_gradient_at_a_constant_sequence_is_finite(masked):
    layer = SequencePooling(strategy="mean_std")
    x = tf.constant(np.full((B, S, D), 3.0, "float32"))
    mask = np.ones((B, S), "float32") if masked else None
    with tf.GradientTape() as tape:
        tape.watch(x)
        total = tf.reduce_sum(layer(x, mask=mask))
    assert np.isfinite(tape.gradient(total, x).numpy()).all()
