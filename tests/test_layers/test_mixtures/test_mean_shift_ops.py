"""Tests for ``layers/mixtures/mean_shift_ops.py`` against a float64 NumPy oracle.

The oracle (``mean_shift_oracle.py``) computes the textbook kernel-weighted sums
key by key; the implementation uses the softmax identity with a key-norm bias and
the expanded-square distance. Tight tolerances run on float64 inputs (the ops keep
float64, and the realised dtype is asserted); float32 and float16 arms state their
own tolerances. The three identity checks (one step, query-norm invariance, the
gradient identity) are factored into helper functions so a RED proof can inject a
defect into ``kernel_logits`` and assert that the very same body fails.

The ops are stateless and read the dtype from their inputs, so the float16 arms
feed float16 tensors directly instead of using the ``mixed_float16_policy``
fixture: that fixture changes the global policy, which no function here reads.
"""

import re

import numpy as np
import pytest
import tensorflow as tf
import keras

from dl_techniques.layers.mixtures import mean_shift_ops as ops

from .mean_shift_oracle import (
    oracle_iterate,
    oracle_kde_log_density,
    oracle_median_bandwidth,
    oracle_pairwise_sq_dist,
    oracle_step,
)

pytestmark = pytest.mark.usefixtures("tf32_disabled")


def _dtype(tensor):
    return keras.backend.standardize_dtype(tensor.dtype)


def _np(tensor):
    return keras.ops.convert_to_numpy(tensor)


def _points(batch, count, dim, seed, dtype="float64", scale=1.0):
    rng = np.random.default_rng(seed)
    return (scale * rng.normal(size=(batch, count, dim))).astype(dtype)


def _varied_norm_points(batch, count, dim, seed):
    """Points whose norms differ a lot, so a key-norm bias is not negligible."""
    rng = np.random.default_rng(seed)
    scales = rng.uniform(0.3, 3.0, size=(batch, count, 1))
    return rng.normal(size=(batch, count, dim)) * scales


def _softmax_np(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    expd = np.exp(shifted)
    return expd / expd.sum(axis=-1, keepdims=True)


def _keep_with_empty_rows(batch, count, seed, empty_rows=()):
    rng = np.random.default_rng(seed)
    keep = rng.random((batch, count)) > 0.4
    for row in range(batch):
        keep[row, row % count] = True
    for row in empty_rows:
        keep[row] = False
    return keep


# ---------------------------------------------------------------------
# Identity checks, factored so the RED proofs can run the same bodies
# ---------------------------------------------------------------------


def _check_one_step_matches_oracle():
    q = _varied_norm_points(2, 7, 3, seed=1)
    k = _varied_norm_points(2, 9, 3, seed=2)
    h = 0.9
    out, weights = ops.mean_shift_step(
        keras.ops.convert_to_tensor(q), keras.ops.convert_to_tensor(k), None, h
    )
    assert _dtype(out) == "float64" and _dtype(weights) == "float64"
    expected, expected_weights = oracle_step(q, k, None, h)
    np.testing.assert_allclose(_np(weights), expected_weights, rtol=0, atol=1e-5)
    np.testing.assert_allclose(_np(out), expected, rtol=0, atol=1e-5)


def _check_query_norm_is_softmax_invariant():
    q = _varied_norm_points(2, 6, 3, seed=3)
    k = _varied_norm_points(2, 8, 3, seed=4)
    h = 0.8
    qt, kt = keras.ops.convert_to_tensor(q), keras.ops.convert_to_tensor(k)
    dropped = _np(ops.kernel_logits(qt, kt, h, drop_query_norm=True))
    full = _np(ops.kernel_logits(qt, kt, h, drop_query_norm=False))
    # The logits are defined up to a per-query constant: the dropped term is the
    # query norm measured from the key mean, which centres both operands.
    centre = k.mean(axis=-2, keepdims=True)
    per_row = 0.5 * np.sum((q - centre) ** 2, axis=-1, keepdims=True) / h**2
    assert per_row.max() > 1.0
    np.testing.assert_allclose(
        dropped - full, np.broadcast_to(per_row, dropped.shape), rtol=0, atol=1e-8
    )
    _, oracle_weights = oracle_step(q, k, None, h)
    np.testing.assert_allclose(_softmax_np(dropped), oracle_weights, rtol=0, atol=1e-5)
    np.testing.assert_allclose(_softmax_np(full), oracle_weights, rtol=0, atol=1e-5)


def _check_shift_is_h2_times_density_gradient():
    q = _varied_norm_points(2, 5, 3, seed=5)
    k = _varied_norm_points(2, 11, 3, seed=6)
    h = 0.7
    qt, kt = keras.ops.convert_to_tensor(q), keras.ops.convert_to_tensor(k)
    with tf.GradientTape() as tape:
        tape.watch(qt)
        total = keras.ops.sum(ops.kde_log_density(qt, kt, h))
    grad = _np(tape.gradient(total, qt))
    moved, _ = ops.mean_shift_step(qt, kt, kt, h)
    assert _dtype(moved) == "float64"
    shift = _np(moved) - q
    assert np.abs(shift).max() > 1e-2
    np.testing.assert_allclose(shift, h**2 * grad, rtol=1e-4, atol=1e-4)
    oracle_shift = oracle_step(q, k, None, h)[0] - q
    np.testing.assert_allclose(shift, oracle_shift, rtol=0, atol=1e-5)


CHECKS = {
    "step": _check_one_step_matches_oracle,
    "query_norm": _check_query_norm_is_softmax_invariant,
    "gradient": _check_shift_is_h2_times_density_gradient,
}

# Multiplier of ``||k||^2 / h^2`` added to the correct logits. The correct bias is
# ``-||k||^2 / (2 h^2)``; adding +0.5 removes it, +1.0 flips its sign, -0.5 doubles it.
BIAS_DEFECTS = {"dropped": 0.5, "flipped_sign": 1.0, "doubled": -0.5}


def _with_bias_defect(real, coefficient):
    def patched(q, k, bandwidth, kernel="gaussian", drop_query_norm=True, key_mask=None):
        logits = real(q, k, bandwidth, kernel, drop_query_norm, key_mask)
        k_cast = keras.ops.cast(k, logits.dtype)
        h = keras.ops.convert_to_tensor(bandwidth, dtype=logits.dtype)
        k_sq = keras.ops.expand_dims(
            keras.ops.sum(keras.ops.square(k_cast), axis=-1), axis=-2
        )
        return logits + coefficient * k_sq / (h * h)

    return patched


def _with_key_dependent_query_term(real):
    def patched(q, k, bandwidth, kernel="gaussian", drop_query_norm=True, key_mask=None):
        logits = real(q, k, bandwidth, kernel, drop_query_norm, key_mask)
        if drop_query_norm:
            return logits
        k_cast = keras.ops.cast(k, logits.dtype)
        h = keras.ops.convert_to_tensor(bandwidth, dtype=logits.dtype)
        k_sq = keras.ops.expand_dims(
            keras.ops.sum(keras.ops.square(k_cast), axis=-1), axis=-2
        )
        return logits - 0.5 * k_sq / (h * h)

    return patched


RED_CASES = [
    (check, f"bias_{defect}") for check in CHECKS for defect in BIAS_DEFECTS
] + [("query_norm", "key_dependent_query_term")]


def _patched_logits(defect):
    real = ops.kernel_logits
    if defect == "key_dependent_query_term":
        return _with_key_dependent_query_term(real)
    return _with_bias_defect(real, BIAS_DEFECTS[defect.removeprefix("bias_")])


def test_the_red_population_is_not_empty():
    assert len(RED_CASES) == 10
    assert {check for check, _ in RED_CASES} == set(CHECKS)


@pytest.mark.parametrize("check", sorted(CHECKS))
def test_identity_check_passes_on_the_real_ops(check):
    CHECKS[check]()


@pytest.mark.parametrize("check,defect", RED_CASES)
def test_identity_check_fails_on_an_injected_defect(check, defect, monkeypatch):
    CHECKS[check]()
    monkeypatch.setattr(ops, "kernel_logits", _patched_logits(defect))
    with pytest.raises(AssertionError):
        CHECKS[check]()


# ---------------------------------------------------------------------
# Spec tests 1 to 10
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape_a,shape_b",
    [((2, 6, 4), (2, 9, 4)), ((6, 4), (9, 4)), ((2, 3, 6, 4), (2, 3, 9, 4))],
)
def test_1_pairwise_sq_dist_matches_oracle_is_nonnegative(shape_a, shape_b):
    rng = np.random.default_rng(10)
    a = rng.normal(size=shape_a)
    b = rng.normal(size=shape_b)
    got = ops.pairwise_sq_dist(keras.ops.convert_to_tensor(a), keras.ops.convert_to_tensor(b))
    assert _dtype(got) == "float64"
    np.testing.assert_allclose(_np(got), oracle_pairwise_sq_dist(a, b), rtol=0, atol=1e-10)
    assert _np(got).min() >= 0.0


@pytest.mark.parametrize("dtype,atol", [("float64", 1e-10), ("float32", 1e-4)])
def test_1_pairwise_sq_dist_has_a_zero_diagonal(dtype, atol):
    a = _points(2, 12, 5, seed=11, dtype=dtype, scale=2.0)
    got = _np(ops.pairwise_sq_dist(keras.ops.convert_to_tensor(a), keras.ops.convert_to_tensor(a)))
    assert _dtype(keras.ops.convert_to_tensor(a)) == dtype
    assert got.min() >= 0.0
    assert np.abs(np.diagonal(got, axis1=-2, axis2=-1)).max() < atol


def test_1_pairwise_sq_dist_clamps_cancellation_at_zero():
    base = np.full((1, 4, 3), 1.0e4, dtype=np.float32)
    got = _np(ops.pairwise_sq_dist(keras.ops.convert_to_tensor(base), keras.ops.convert_to_tensor(base)))
    assert got.min() >= 0.0


@pytest.mark.parametrize("dtype,atol", [("float64", 1e-10), ("float32", 1e-5)])
def test_2_one_step_is_softmax_with_key_bias_times_x(dtype, atol):
    q = _varied_norm_points(2, 7, 3, seed=20).astype(dtype)
    k = _varied_norm_points(2, 9, 3, seed=21).astype(dtype)
    h = 0.9
    qt, kt = keras.ops.convert_to_tensor(q), keras.ops.convert_to_tensor(k)
    assert _dtype(qt) == dtype
    out, weights = ops.mean_shift_step(qt, kt, None, h)
    assert _dtype(out) == dtype
    expected, expected_weights = oracle_step(q, k, None, h)
    np.testing.assert_allclose(_np(weights), expected_weights, rtol=0, atol=atol)
    np.testing.assert_allclose(_np(out), expected, rtol=0, atol=atol)
    np.testing.assert_allclose(_np(weights).sum(axis=-1), 1.0, rtol=0, atol=atol)


def test_3_dropping_the_query_norm_leaves_weights_unchanged():
    q = _varied_norm_points(2, 6, 3, seed=30)
    k = _varied_norm_points(2, 8, 3, seed=31)
    qt, kt = keras.ops.convert_to_tensor(q), keras.ops.convert_to_tensor(k)
    dropped = ops.kernel_logits(qt, kt, 0.8, drop_query_norm=True)
    full = ops.kernel_logits(qt, kt, 0.8, drop_query_norm=False)
    assert not np.allclose(_np(dropped), _np(full))
    np.testing.assert_allclose(
        _np(keras.ops.softmax(dropped, axis=-1)),
        _np(keras.ops.softmax(full, axis=-1)),
        rtol=0,
        atol=1e-12,
    )


def test_4_shift_equals_h_squared_times_density_gradient():
    _check_shift_is_h2_times_density_gradient()


def test_4_density_value_matches_the_oracle():
    q = _varied_norm_points(2, 5, 3, seed=40)
    k = _varied_norm_points(2, 11, 3, seed=41)
    got = ops.kde_log_density(keras.ops.convert_to_tensor(q), keras.ops.convert_to_tensor(k), 0.7)
    assert _dtype(got) == "float64"
    np.testing.assert_allclose(_np(got), oracle_kde_log_density(q, k, 0.7), rtol=0, atol=1e-9)


def _two_blobs(seed=50, per_blob=30):
    rng = np.random.default_rng(seed)
    left = rng.normal(size=(per_blob, 2)) * 0.3 + np.array([-4.0, 1.0])
    right = rng.normal(size=(per_blob, 2)) * 0.3 + np.array([4.0, -1.0])
    labels = np.array([0] * per_blob + [1] * per_blob)
    return np.concatenate([left, right])[None], labels


def test_5_non_blurring_iteration_reaches_the_blob_modes():
    x, labels = _two_blobs()
    h = 1.0
    modes = []
    for blob in (0, 1):
        z = x[:, labels == blob].mean(axis=1, keepdims=True)
        for _ in range(200):
            z = oracle_step(z, x, x, h)[0]
        modes.append(z[0, 0])
    assert np.linalg.norm(modes[0] - modes[1]) > 6.0
    out = _np(
        ops.mean_shift_iterate(
            keras.ops.convert_to_tensor(x), h, num_steps=50, blurring=False
        )
    )
    assert _dtype(keras.ops.convert_to_tensor(x)) == "float64"
    for blob in (0, 1):
        spread = np.abs(out[0, labels == blob] - modes[blob]).max()
        assert spread < 1e-3, (blob, spread)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_6_huge_bandwidth_returns_the_masked_mean(dtype):
    x = _points(2, 8, 3, seed=60, dtype=dtype)
    keep = _keep_with_empty_rows(2, 8, seed=61)
    out, _ = ops.mean_shift_step(
        keras.ops.convert_to_tensor(x), keras.ops.convert_to_tensor(x), None, 1e3,
        mask=keras.ops.convert_to_tensor(keep),
    )
    assert _dtype(out) == dtype
    for row in range(2):
        expected = x[row][keep[row]].astype(np.float64).mean(axis=0)
        np.testing.assert_allclose(_np(out)[row], np.broadcast_to(expected, (8, 3)), rtol=0, atol=1e-4)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_6_tiny_bandwidth_returns_the_input_points(dtype):
    x = _points(2, 8, 3, seed=62, dtype=dtype)
    out = ops.mean_shift_iterate(
        keras.ops.convert_to_tensor(x), 1e-3, num_steps=1, blurring=False
    )
    assert _dtype(out) == dtype
    np.testing.assert_allclose(_np(out), x, rtol=0, atol=1e-4)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_7_masked_keys_have_exactly_zero_weight(dtype):
    x = _points(3, 9, 4, seed=70, dtype=dtype)
    keep = _keep_with_empty_rows(3, 9, seed=71)
    assert (~keep).sum() > 0
    xt = keras.ops.convert_to_tensor(x)
    _, weights = ops.mean_shift_step(xt, xt, None, 0.8, mask=keras.ops.convert_to_tensor(keep))
    weights = _np(weights)
    dropped = np.broadcast_to(~keep[:, None, :], weights.shape)
    assert (weights[dropped] == 0.0).all()
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, rtol=0, atol=1e-5)


def test_7_fully_masked_row_is_finite_and_the_plain_mean():
    x = _points(3, 9, 4, seed=72)
    v = _points(3, 9, 5, seed=73)
    keep = _keep_with_empty_rows(3, 9, seed=74, empty_rows=(1,))
    assert not keep[1].any() and keep[0].any() and keep[2].any()
    # A small bandwidth makes a kernel-weighted row differ clearly from the mean.
    h = 0.4
    out, weights = ops.mean_shift_step(
        keras.ops.convert_to_tensor(x), keras.ops.convert_to_tensor(x),
        keras.ops.convert_to_tensor(v), h, mask=keras.ops.convert_to_tensor(keep),
    )
    out, weights = _np(out), _np(weights)
    assert np.isfinite(out).all() and np.isfinite(weights).all()
    np.testing.assert_allclose(out[1], np.broadcast_to(v[1].mean(axis=0), (9, 5)), rtol=0, atol=1e-10)
    np.testing.assert_allclose(weights[1], 1.0 / 9.0, rtol=0, atol=1e-12)
    kernel_weighted = oracle_step(x[1:2], x[1:2], v[1:2], h)[0][0]
    assert np.abs(kernel_weighted - v[1].mean(axis=0)).max() > 0.05
    expected, _ = oracle_step(x, x, v, h, keep)
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-9)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_8_blurring_iteration_is_permutation_equivariant(dtype):
    x = _points(2, 10, 3, seed=80, dtype=dtype)
    keep = _keep_with_empty_rows(2, 10, seed=81)
    perm = np.random.default_rng(82).permutation(10)
    assert not (perm == np.arange(10)).all()
    plain = _np(ops.mean_shift_iterate(
        keras.ops.convert_to_tensor(x), 0.9, 4, mask=keras.ops.convert_to_tensor(keep)
    ))
    shuffled = _np(ops.mean_shift_iterate(
        keras.ops.convert_to_tensor(x[:, perm]), 0.9, 4,
        mask=keras.ops.convert_to_tensor(keep[:, perm]),
    ))
    atol = 1e-12 if dtype == "float64" else 1e-5
    np.testing.assert_allclose(shuffled, plain[:, perm], rtol=0, atol=atol)


def test_9_float16_inputs_are_finite_and_close_to_float32():
    x16 = (_points(2, 12, 3, seed=90, dtype="float32") + np.array([6.0, -6.0, 3.0])).astype("float16")
    x32 = x16.astype("float32")
    keep = _keep_with_empty_rows(2, 12, seed=91)
    for steps in (1, 3):
        low = ops.mean_shift_iterate(
            keras.ops.convert_to_tensor(x16), 0.7, steps, mask=keras.ops.convert_to_tensor(keep)
        )
        ref = ops.mean_shift_iterate(
            keras.ops.convert_to_tensor(x32), 0.7, steps, mask=keras.ops.convert_to_tensor(keep)
        )
        assert _dtype(low) == "float16" and _dtype(ref) == "float32"
        assert np.isfinite(_np(low)).all()
        np.testing.assert_allclose(_np(low).astype("float32"), _np(ref), rtol=0, atol=1e-2)


def test_9_float16_with_squares_beyond_the_float16_range_stays_finite():
    rng = np.random.default_rng(92)
    centers = np.array([[300.0, 0.0], [-300.0, 0.0]])
    pts = np.concatenate([c + rng.normal(size=(6, 2)) for c in centers])[None]
    x16 = pts.astype("float16")
    assert np.square(x16.astype("float32")).max() > np.finfo("float16").max
    low = ops.mean_shift_iterate(keras.ops.convert_to_tensor(x16), 1.0, 3, blurring=False)
    ref = ops.mean_shift_iterate(
        keras.ops.convert_to_tensor(x16.astype("float32")), 1.0, 3, blurring=False
    )
    assert np.isfinite(_np(low)).all()
    # One float16 ulp at |x| in [256, 512) is 0.25; three rounded steps stay within it.
    np.testing.assert_allclose(_np(low).astype("float32"), _np(ref), rtol=0, atol=0.25)


def test_10_graph_mode_with_unknown_batch_and_length_matches_eager():
    dim = 3
    x = _points(2, 9, dim, seed=100, dtype="float32")
    keep = _keep_with_empty_rows(2, 9, seed=101)

    @tf.function(input_signature=[
        tf.TensorSpec([None, None, dim], tf.float32), tf.TensorSpec([None, None], tf.bool)
    ])
    def graph(points, mask):
        shifted = ops.mean_shift_iterate(points, 0.9, 3, mask=mask)
        density = ops.kde_log_density(points, points, 0.9, mask=mask)
        bandwidth = ops.median_heuristic_bandwidth(points, mask)
        return shifted, density, bandwidth

    shifted, density, bandwidth = graph(tf.constant(x), tf.constant(keep))
    xt, kt = keras.ops.convert_to_tensor(x), keras.ops.convert_to_tensor(keep)
    np.testing.assert_allclose(_np(shifted), _np(ops.mean_shift_iterate(xt, 0.9, 3, mask=kt)), atol=1e-5)
    np.testing.assert_allclose(_np(density), _np(ops.kde_log_density(xt, xt, 0.9, mask=kt)), atol=1e-4)
    np.testing.assert_allclose(_np(bandwidth), _np(ops.median_heuristic_bandwidth(xt, kt)), atol=1e-6)
    other = graph(tf.constant(_points(3, 5, dim, seed=102, dtype="float32")), tf.constant(np.ones((3, 5), bool)))
    assert tuple(other[0].shape) == (3, 5, dim)


# ---------------------------------------------------------------------
# Liveness and contract arms
# ---------------------------------------------------------------------


def test_more_steps_move_the_points_further():
    x = _points(1, 20, 2, seed=110) * 2.0
    xt = keras.ops.convert_to_tensor(x)
    one = _np(ops.mean_shift_iterate(xt, 1.0, 1))
    three = _np(ops.mean_shift_iterate(xt, 1.0, 3))
    assert np.abs(three - one).max() > 1e-2
    np.testing.assert_allclose(one, oracle_iterate(x, 1.0, 1), rtol=0, atol=1e-9)
    np.testing.assert_allclose(three, oracle_iterate(x, 1.0, 3), rtol=0, atol=1e-9)


@pytest.mark.parametrize("blurring", [True, False])
@pytest.mark.parametrize("step_size", [1.0, 0.5, 1.5])
def test_iteration_matches_the_oracle(blurring, step_size):
    x = _points(2, 9, 3, seed=111)
    keep = _keep_with_empty_rows(2, 9, seed=112)
    got = _np(ops.mean_shift_iterate(
        keras.ops.convert_to_tensor(x), 0.8, 4, step_size=step_size, blurring=blurring,
        mask=keras.ops.convert_to_tensor(keep),
    ))
    expected = oracle_iterate(x, 0.8, 4, step_size, blurring, keep)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9)


def test_one_step_blurring_and_non_blurring_agree():
    x = keras.ops.convert_to_tensor(_points(2, 7, 3, seed=113))
    np.testing.assert_allclose(
        _np(ops.mean_shift_iterate(x, 0.8, 1, blurring=True)),
        _np(ops.mean_shift_iterate(x, 0.8, 1, blurring=False)),
        rtol=0, atol=1e-12,
    )


def test_zero_steps_return_the_input_and_the_last_weights_are_available():
    x = keras.ops.convert_to_tensor(_points(2, 6, 3, seed=114))
    np.testing.assert_array_equal(_np(ops.mean_shift_iterate(x, 1.0, 0)), _np(x))
    z, weights = ops.mean_shift_iterate(x, 0.8, 2, return_weights=True)
    assert tuple(weights.shape) == (2, 6, 6)
    _, expected = ops.mean_shift_step(
        ops.mean_shift_iterate(x, 0.8, 1), ops.mean_shift_iterate(x, 0.8, 1), None, 0.8
    )
    np.testing.assert_allclose(_np(weights), _np(expected), rtol=0, atol=1e-12)
    assert tuple(z.shape) == (2, 6, 3)


@pytest.mark.parametrize(
    "bandwidth_shape,query_shape",
    [
        ((), (2, 6, 3)),
        ((2, 1, 1), (2, 6, 3)),
        ((2, 4, 1, 1), (2, 4, 6, 3)),
        ((2, 6, 1), (2, 6, 3)),
        ((2, 4, 6, 1), (2, 4, 6, 3)),
    ],
)
def test_every_bandwidth_broadcast_shape_is_accepted_and_correct(bandwidth_shape, query_shape):
    rng = np.random.default_rng(120)
    q = rng.normal(size=query_shape)
    k = rng.normal(size=query_shape[:-2] + (8, 3))
    h = rng.uniform(0.5, 1.5, size=bandwidth_shape) if bandwidth_shape else np.float64(0.8)
    out, weights = ops.mean_shift_step(
        keras.ops.convert_to_tensor(q), keras.ops.convert_to_tensor(k), None,
        keras.ops.convert_to_tensor(h) if bandwidth_shape else 0.8,
    )
    assert tuple(out.shape) == query_shape
    assert tuple(weights.shape) == query_shape[:-1] + (8,)
    expected, _ = oracle_step(q, k, None, h)
    np.testing.assert_allclose(_np(out), expected, rtol=0, atol=1e-9)
    density = ops.kde_log_density(
        keras.ops.convert_to_tensor(q), keras.ops.convert_to_tensor(k),
        keras.ops.convert_to_tensor(h) if bandwidth_shape else 0.8,
    )
    assert tuple(density.shape) == query_shape[:-1]
    np.testing.assert_allclose(_np(density), oracle_kde_log_density(q, k, h), rtol=0, atol=1e-9)


@pytest.mark.parametrize("mask_shape,query_shape", [((2, 8), (2, 4, 6, 3)), ((2, 4, 8), (2, 4, 6, 3)), ((2, 6, 8), (2, 6, 3))])
def test_mask_ranks_are_aligned_before_the_key_axis(mask_shape, query_shape):
    rng = np.random.default_rng(121)
    q = rng.normal(size=query_shape)
    k = rng.normal(size=query_shape[:-2] + (8, 3))
    keep = rng.random(mask_shape) > 0.4
    keep[..., 0] = True
    out, weights = ops.mean_shift_step(
        keras.ops.convert_to_tensor(q), keras.ops.convert_to_tensor(k), None, 0.9,
        mask=keras.ops.convert_to_tensor(keep),
    )
    lead = len(query_shape) - len(mask_shape)
    full_keep = keep.reshape(keep.shape[:-1] + (1,) * lead + (8,)) if lead > 0 else keep
    full_keep = np.broadcast_to(full_keep, tuple(weights.shape))
    assert (~full_keep).sum() > 0
    assert (_np(weights)[~full_keep] == 0.0).all()
    np.testing.assert_allclose(_np(weights).sum(axis=-1), 1.0, rtol=0, atol=1e-12)
    if lead > 0:
        oracle_keep = np.broadcast_to(full_keep[..., 0, :], query_shape[:-2] + (8,))
        expected, _ = oracle_step(q, k, None, 0.9, oracle_keep)
        np.testing.assert_allclose(_np(out), expected, rtol=0, atol=1e-9)
    with pytest.raises(ValueError, match="rank"):
        ops.mean_shift_step(
            keras.ops.convert_to_tensor(q), keras.ops.convert_to_tensor(k), None, 0.9,
            mask=keras.ops.convert_to_tensor(np.ones((1,) * (len(query_shape) + 1), bool)),
        )


def test_kde_masked_and_fully_masked_rows_are_finite_and_match_the_oracle():
    q = _points(3, 5, 3, seed=122)
    k = _points(3, 9, 3, seed=123)
    keep = _keep_with_empty_rows(3, 9, seed=124, empty_rows=(2,))
    got = _np(ops.kde_log_density(
        keras.ops.convert_to_tensor(q), keras.ops.convert_to_tensor(k), 0.8,
        mask=keras.ops.convert_to_tensor(keep),
    ))
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, oracle_kde_log_density(q, k, 0.8, keep), rtol=0, atol=1e-9)


@pytest.mark.parametrize("name", ["kernel_logits", "step", "iterate", "non_string_kernel"])
def test_a_bad_kernel_name_raises_naming_the_value(name):
    x = keras.ops.convert_to_tensor(_points(1, 4, 2, seed=130))
    calls = {
        "kernel_logits": lambda: ops.kernel_logits(x, x, 1.0, kernel="triangle"),
        "step": lambda: ops.mean_shift_step(x, x, None, 1.0, kernel="triangle"),
        "iterate": lambda: ops.mean_shift_iterate(x, 1.0, 1, kernel="triangle"),
        "non_string_kernel": lambda: ops.mean_shift_step(x, x, None, 1.0, kernel=3),
    }
    with pytest.raises(ValueError, match="triangle|3"):
        calls[name]()


@pytest.mark.parametrize("bad", [-1, 2.5, "3", True, None])
def test_a_bad_num_steps_raises_naming_the_value(bad):
    x = keras.ops.convert_to_tensor(_points(1, 4, 2, seed=131))
    with pytest.raises(ValueError, match=r"num_steps.*" + re.escape(repr(bad))):
        ops.mean_shift_iterate(x, 1.0, bad)


@pytest.mark.parametrize("bad", [0.0, 2.0, -0.5, 2.5, "1"])
def test_a_bad_step_size_raises_naming_the_value(bad):
    x = keras.ops.convert_to_tensor(_points(1, 4, 2, seed=132))
    with pytest.raises(ValueError, match="step_size"):
        ops.mean_shift_iterate(x, 1.0, 1, step_size=bad)


def test_return_weights_without_a_step_raises():
    x = keras.ops.convert_to_tensor(_points(1, 4, 2, seed=133))
    with pytest.raises(ValueError, match="num_steps"):
        ops.mean_shift_iterate(x, 1.0, 0, return_weights=True)


def test_vmf_one_step_is_softmax_of_the_dot_product_over_h_squared():
    rng = np.random.default_rng(140)
    q = rng.normal(size=(2, 5, 4))
    k = rng.normal(size=(2, 8, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    h = 0.6
    out, weights = ops.mean_shift_step(
        keras.ops.convert_to_tensor(q), keras.ops.convert_to_tensor(k), None, h, kernel="vmf"
    )
    expected = _softmax_np(np.einsum("bnd,bmd->bnm", q, k) / h**2)
    np.testing.assert_allclose(_np(weights), expected, rtol=0, atol=1e-9)
    np.testing.assert_allclose(_np(out), expected @ k, rtol=0, atol=1e-9)
    oracle_out, _ = oracle_step(q, k, None, h, kernel="vmf")
    np.testing.assert_allclose(_np(out), oracle_out, rtol=0, atol=1e-9)


def test_vmf_normalises_unnormalised_inputs_and_survives_zero_rows():
    rng = np.random.default_rng(141)
    q = rng.normal(size=(1, 4, 3)) * 5.0
    k = rng.normal(size=(1, 6, 3)) * 0.01
    k[0, 2] = 0.0
    logits = _np(ops.kernel_logits(
        keras.ops.convert_to_tensor(q), keras.ops.convert_to_tensor(k), 0.5, kernel="vmf"
    ))
    assert np.isfinite(logits).all()
    assert (logits[0, :, 2] == 0.0).all()
    unit = lambda a: a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-300)
    expected = np.einsum("bnd,bmd->bnm", unit(q), unit(k)) / 0.25
    expected[0, :, 2] = 0.0
    np.testing.assert_allclose(logits, expected, rtol=0, atol=1e-8)


def test_vmf_iteration_stays_on_the_sphere_and_matches_the_oracle():
    x = _points(2, 9, 3, seed=142)
    got = _np(ops.mean_shift_iterate(keras.ops.convert_to_tensor(x), 0.7, 3, kernel="vmf"))
    np.testing.assert_allclose(np.linalg.norm(got, axis=-1), 1.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(got, oracle_iterate(x, 0.7, 3, kernel="vmf"), rtol=0, atol=1e-8)


@pytest.mark.parametrize("masked", [False, True])
def test_median_heuristic_matches_the_oracle(masked):
    x = _points(3, 9, 4, seed=150)
    keep = _keep_with_empty_rows(3, 9, seed=151) if masked else None
    got = ops.median_heuristic_bandwidth(
        keras.ops.convert_to_tensor(x), None if keep is None else keras.ops.convert_to_tensor(keep)
    )
    assert tuple(got.shape) == (3, 1, 1) and _dtype(got) == "float64"
    np.testing.assert_allclose(_np(got), oracle_median_bandwidth(x, keep), rtol=0, atol=1e-9)


def test_median_heuristic_fallbacks_and_floor():
    x = _points(4, 6, 3, seed=152)
    x[3] = 1.0
    keep = np.ones((4, 6), bool)
    keep[0, 1:] = False
    keep[1] = False
    got = _np(ops.median_heuristic_bandwidth(
        keras.ops.convert_to_tensor(x), keras.ops.convert_to_tensor(keep)
    ))[:, 0, 0]
    assert got[0] == 1.0 and got[1] == 1.0
    assert got[2] > 0.0 and got[3] > 0.0 and got[3] < 1e-3


def test_median_heuristic_has_no_gradient():
    x = keras.ops.convert_to_tensor(_points(2, 6, 3, seed=153))
    with tf.GradientTape() as tape:
        tape.watch(x)
        total = keras.ops.sum(ops.median_heuristic_bandwidth(x))
    grad = tape.gradient(total, x)
    assert grad is None or float(np.abs(_np(grad)).max()) == 0.0


def test_median_heuristic_rejects_a_wrong_rank():
    with pytest.raises(ValueError, match="rank"):
        ops.median_heuristic_bandwidth(keras.ops.convert_to_tensor(np.ones((4, 3))))


# ---------------------------------------------------------------------
# Masking without a sentinel: kept logits far below any fixed constant
# ---------------------------------------------------------------------

TINY_BANDWIDTH = 1e-6


def _far_kept_keys():
    """Query at the origin; keys 0, 1 and 3 kept, key 2 (nearest) masked."""
    q = np.zeros((1, 1, 2), dtype="float32")
    k = np.array([[[1.0, 0.0], [3.0, 0.5], [0.0, 0.0], [-2.0, 1.0]]], dtype="float32")
    keep = np.array([[True, True, False, True]])
    return q, k, keep


def test_w1_a_tiny_bandwidth_keeps_the_nearest_kept_key_and_the_masked_weight_is_zero():
    q, k, keep = _far_kept_keys()
    out, weights = ops.mean_shift_step(
        keras.ops.convert_to_tensor(q), keras.ops.convert_to_tensor(k), None,
        TINY_BANDWIDTH, mask=keras.ops.convert_to_tensor(keep),
    )
    logit = -np.sum(k[0, keep[0]] ** 2, axis=-1) / (2 * TINY_BANDWIDTH**2)
    assert logit.max() < -1e9
    weights = _np(weights)[0, 0]
    assert weights[2] == 0.0
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(weights, [1.0, 0.0, 0.0, 0.0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(_np(out)[0, 0], k[0, 0], rtol=0, atol=1e-6)


def test_w1_a_masked_step_equals_the_step_over_the_kept_subset_at_a_tiny_bandwidth():
    q, k, keep = _far_kept_keys()
    qt, kt = keras.ops.convert_to_tensor(q), keras.ops.convert_to_tensor(k)
    out, weights = ops.mean_shift_step(qt, kt, None, 1e-3, mask=keras.ops.convert_to_tensor(keep))
    kept = keras.ops.convert_to_tensor(k[:, keep[0]])
    sub_out, sub_weights = ops.mean_shift_step(qt, kept, None, 1e-3)
    np.testing.assert_allclose(_np(out), _np(sub_out), rtol=0, atol=1e-6)
    np.testing.assert_allclose(_np(weights)[..., keep[0]], _np(sub_weights), rtol=0, atol=1e-6)
    assert (_np(weights)[..., ~keep[0]] == 0.0).all()


def test_w1_the_kde_density_at_a_tiny_bandwidth_is_finite_and_equals_the_subset_density():
    q, k, keep = _far_kept_keys()
    qt, kt = keras.ops.convert_to_tensor(q), keras.ops.convert_to_tensor(k)
    got = _np(ops.kde_log_density(qt, kt, TINY_BANDWIDTH, mask=keras.ops.convert_to_tensor(keep)))
    assert np.isfinite(got).all()
    sub = _np(ops.kde_log_density(qt, keras.ops.convert_to_tensor(k[:, keep[0]]), TINY_BANDWIDTH))
    np.testing.assert_allclose(got, sub, rtol=1e-6, atol=0)
    expected = oracle_kde_log_density(q, k, TINY_BANDWIDTH, keep)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=0)


def test_w1_a_fully_masked_row_stays_uniform_at_a_tiny_bandwidth():
    q, k, _ = _far_kept_keys()
    keep = np.zeros((1, 4), dtype=bool)
    qt, kt = keras.ops.convert_to_tensor(q), keras.ops.convert_to_tensor(k)
    out, weights = ops.mean_shift_step(qt, kt, None, TINY_BANDWIDTH, mask=keras.ops.convert_to_tensor(keep))
    np.testing.assert_allclose(_np(weights), 0.25, rtol=0, atol=0)
    np.testing.assert_allclose(_np(out)[0, 0], k[0].mean(axis=0), rtol=0, atol=1e-6)
    density = _np(ops.kde_log_density(qt, kt, TINY_BANDWIDTH, mask=keras.ops.convert_to_tensor(keep)))
    assert np.isfinite(density).all()


def test_w1_float16_inputs_at_a_tiny_bandwidth_are_finite_with_exact_zero_masked_weights():
    q, k, keep = _far_kept_keys()
    out, weights = ops.mean_shift_step(
        keras.ops.convert_to_tensor(q.astype("float16")),
        keras.ops.convert_to_tensor(k.astype("float16")), None,
        TINY_BANDWIDTH, mask=keras.ops.convert_to_tensor(keep),
    )
    assert np.isfinite(_np(out)).all() and np.isfinite(_np(weights)).all()
    assert (_np(weights)[..., ~keep[0]] == 0.0).all()
    np.testing.assert_allclose(_np(weights).sum(), 1.0, rtol=0, atol=1e-6)


def test_w1_gradients_through_a_masked_step_are_finite():
    q, k, keep = _far_kept_keys()
    qt, kt = tf.constant(q), tf.constant(k)
    with tf.GradientTape() as tape:
        tape.watch([qt, kt])
        out, _ = ops.mean_shift_step(qt, kt, None, 1e-2, mask=tf.constant(keep))
        total = keras.ops.sum(out * out)
    grads = tape.gradient(total, [qt, kt])
    assert all(np.isfinite(_np(g)).all() for g in grads)
    assert float(np.abs(_np(grads[1])[0, 2]).max()) == 0.0


# ---------------------------------------------------------------------
# Translation safety in float32
# ---------------------------------------------------------------------

OFFSETS = (0.0, 10.0, 100.0, 1000.0)
# Offsets up to 1000 on a 2^-10 grid need 20 mantissa bits, so the shifted float32
# data is exactly the shifted data and only the algorithm's own error is measured.
GRID = 1024.0


def _grid_points(batch, count, dim, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    pts = np.round(scale * rng.normal(size=(batch, count, dim)) * GRID) / GRID
    return pts.astype("float32")


def _shifted(x, offset):
    shifted = (x.astype("float64") + offset).astype("float32")
    assert (shifted.astype("float64") - offset == x).all()
    return shifted


def _translation_cases():
    x = _grid_points(2, 24, 4, seed=300, scale=0.5)
    keep = _keep_with_empty_rows(2, 24, seed=301)
    return x, keep


def _translation_errors(offset):
    """Largest deviation from the offset-0 result of every function, per function."""
    x, keep = _translation_cases()
    kt = keras.ops.convert_to_tensor(keep)
    base_x = keras.ops.convert_to_tensor(x)
    moved_x = keras.ops.convert_to_tensor(_shifted(x, offset))
    h = 0.3
    errors = {}

    base = _np(ops.pairwise_sq_dist(base_x, base_x))
    moved = _np(ops.pairwise_sq_dist(moved_x, moved_x))
    errors["pairwise_sq_dist"] = np.abs(moved - base).max() / base.max()

    base_out, base_w = ops.mean_shift_step(base_x, base_x, None, h, mask=kt)
    out, w = ops.mean_shift_step(moved_x, moved_x, None, h, mask=kt)
    errors["step_weights"] = np.abs(_np(w) - _np(base_w)).max()
    errors["step_points"] = np.abs(_np(out).astype("float64") - offset - _np(base_out)).max()

    base_d = _np(ops.kde_log_density(base_x, base_x, h, mask=kt))
    moved_d = _np(ops.kde_log_density(moved_x, moved_x, h, mask=kt))
    errors["kde_log_density"] = np.abs(moved_d - base_d).max() / np.abs(base_d).max()

    base_z = _np(ops.mean_shift_iterate(base_x, h, 4, mask=kt))
    moved_z = _np(ops.mean_shift_iterate(moved_x, h, 4, mask=kt))
    errors["iterate"] = np.abs(moved_z.astype("float64") - offset - base_z).max()

    base_h = _np(ops.median_heuristic_bandwidth(base_x, kt))
    moved_h = _np(ops.median_heuristic_bandwidth(moved_x, kt))
    errors["median_heuristic"] = np.abs(moved_h - base_h).max() / base_h.max()
    return errors


# Measured with the centred ops at offsets 0, 10, 100 and 1000: the distance-like
# outputs deviate by at most 4e-7 relative (float32 epsilon is 1.2e-7); the point
# outputs by at most 1 ulp of the shifted float32 value per step (a result near 1000
# can only be stored to 6.1e-5) and 3.7 ulp after the four iterations. The bounds
# below are 1e-6 relative and 2 ulp per step, 8 ulp for the iteration, a few times
# the measurement. Without the centring the same quantities deviate by 0.07 to 1.2.
RELATIVE_BOUND = 1e-6
STEP_ULPS = {"step_points": 2, "iterate": 8}


def _check_translation_equivariance(offset):
    errors = _translation_errors(offset)
    ulp = float(np.spacing(np.float32(offset + 4.0)))
    bad = {}
    for name, err in errors.items():
        bound = STEP_ULPS[name] * ulp if name in STEP_ULPS else RELATIVE_BOUND
        if not err <= bound:
            bad[name] = (float(err), bound)
    assert not bad, f"offset {offset}: {bad}"


@pytest.mark.parametrize("offset", OFFSETS)
def test_w2_every_op_is_translation_equivariant_in_float32(offset):
    _check_translation_equivariance(offset)


def test_w2_red_proof_without_the_centring_the_translation_check_fails_at_1000(monkeypatch):
    _check_translation_equivariance(1000.0)

    def no_centre(keys, key_mask=None):
        return keras.ops.zeros_like(keras.ops.mean(keys, axis=-2, keepdims=True))

    monkeypatch.setattr(ops, "_key_centre", no_centre)
    with pytest.raises(AssertionError):
        _check_translation_equivariance(1000.0)


def test_w2_pairwise_sq_dist_resolves_a_near_coincident_pair_far_from_the_origin():
    a = np.array([[[1000.0, 1000.0, 1000.0]]], dtype="float32")
    b = a + np.float32(2.0**-6)
    got = float(_np(ops.pairwise_sq_dist(keras.ops.convert_to_tensor(a), keras.ops.convert_to_tensor(b)))[0, 0, 0])
    np.testing.assert_allclose(got, 3 * 2.0**-12, rtol=1e-5)


def test_w2_a_junk_value_in_a_masked_key_does_not_set_the_centre():
    x = _grid_points(1, 12, 3, seed=310)
    keep = np.ones((1, 12), dtype=bool)
    keep[0, 5] = False
    junk = x.copy()
    junk[0, 5] = 1.0e4
    clean = x.copy()
    clean[0, 5] = x[0, 0]
    kt = keras.ops.convert_to_tensor(keep)
    a = _np(ops.mean_shift_step(keras.ops.convert_to_tensor(x), keras.ops.convert_to_tensor(junk), None, 0.4, mask=kt)[1])
    b = _np(ops.mean_shift_step(keras.ops.convert_to_tensor(x), keras.ops.convert_to_tensor(clean), None, 0.4, mask=kt)[1])
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
    da = _np(ops.kde_log_density(keras.ops.convert_to_tensor(x), keras.ops.convert_to_tensor(junk), 0.4, mask=kt))
    db = _np(ops.kde_log_density(keras.ops.convert_to_tensor(x), keras.ops.convert_to_tensor(clean), 0.4, mask=kt))
    np.testing.assert_allclose(da, db, rtol=0, atol=1e-4)


def test_w2_the_logits_are_defined_up_to_a_per_query_constant_and_the_weights_are_not_affected():
    q = _points(2, 5, 3, seed=320)
    k = _points(2, 7, 3, seed=321)
    h = 0.8
    logits = _np(ops.kernel_logits(keras.ops.convert_to_tensor(q), keras.ops.convert_to_tensor(k), h))
    raw = np.einsum("bnd,bmd->bnm", q, k) / h**2 - 0.5 * np.sum(k * k, axis=-1)[:, None, :] / h**2
    centred_logits = logits - logits.max(axis=-1, keepdims=True)
    centred_raw = raw - raw.max(axis=-1, keepdims=True)
    np.testing.assert_allclose(centred_logits, centred_raw, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------
# Tensor-like operands: every tensor argument may be a Variable or an array
# ---------------------------------------------------------------------


def _operand_forms(array):
    return {
        "numpy": array,
        "variable": tf.Variable(array),
        "tensor": keras.ops.convert_to_tensor(array),
    }


@pytest.mark.parametrize("form", ["numpy", "variable"])
def test_n12_every_second_operand_may_be_an_array_or_a_variable(form):
    q = _grid_points(2, 5, 3, seed=330)
    k = _grid_points(2, 7, 3, seed=331)
    v = _grid_points(2, 7, 2, seed=332)
    keep = _keep_with_empty_rows(2, 7, seed=333)
    qt, kt, vt = (keras.ops.convert_to_tensor(a) for a in (q, k, v))
    reference = {
        "sq": _np(ops.pairwise_sq_dist(qt, kt)),
        "logits": _np(ops.kernel_logits(qt, kt, 0.7)),
        "step": _np(ops.mean_shift_step(qt, kt, vt, 0.7, mask=keep)[0]),
        "kde": _np(ops.kde_log_density(qt, kt, 0.7, mask=keep)),
        "iterate": _np(ops.mean_shift_iterate(kt, 0.7, 2)),
        "median": _np(ops.median_heuristic_bandwidth(kt)),
    }
    kf, vf = _operand_forms(k)[form], _operand_forms(v)[form]
    qf = _operand_forms(q)[form]
    got = {
        "sq": _np(ops.pairwise_sq_dist(qf, kf)),
        "logits": _np(ops.kernel_logits(qf, kf, 0.7)),
        "step": _np(ops.mean_shift_step(qf, kf, vf, 0.7, mask=keep)[0]),
        "kde": _np(ops.kde_log_density(qf, kf, 0.7, mask=keep)),
        "iterate": _np(ops.mean_shift_iterate(kf, 0.7, 2)),
        "median": _np(ops.median_heuristic_bandwidth(kf)),
    }
    for name, expected in reference.items():
        np.testing.assert_allclose(got[name], expected, rtol=0, atol=1e-6, err_msg=name)
