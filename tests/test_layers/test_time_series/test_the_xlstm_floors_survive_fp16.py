"""The four `xlstm_blocks.py` stability floors must survive a float16 compute dtype.

Guard for `plan-2026-08-31T134711-6271592d` step 7, defect class I-3: an epsilon
that guards a divide or a ``log`` must be strictly greater than ``0.0`` once it is
materialized in the dtype the site actually computes in. The four sites are

- ``sLSTMCell.call``  ``log(f_activation(f_proj) + eps)``      (the log floor)
- ``sLSTMCell.call``  ``c_t / (n_t + eps)``                    (the divide floor)
- ``mLSTMCell.call``  ``log(sigmoid(f_proj) + eps)``           (the log floor)
- ``mLSTMCell.call``  ``max(|nq|, exp(-m_t)) + eps``           (the divide floor)

each of which was a bare literal ``1e-8``. ``float16(1e-8)`` is exactly ``0.0``,
so under ``mixed_float16`` the guard was a no-op and the op it guarded produced
``NaN`` (the two divides) or an infinite gradient (the two logs). Neither shows
up in float32, which is why 465 green tests in this directory never saw it, and
`models/time_series/xlstm/README.md` §9.5 advertises `mixed_float16` as supported.

Every subject below drives the gates into the exact saturation the floor exists
to cover, by setting the cell's weights directly:

- a very negative input-gate pre-activation makes ``i_t = exp(i_proj - m_t)``
  underflow to ``0.0``, so ``n_t`` is exactly zero and ``c_t / n_t`` is ``0/0``;
- a very negative forget-gate pre-activation makes ``sigmoid(f_proj)`` underflow
  to ``0.0``, so ``log(0)`` is ``-inf`` and its gradient is ``inf``;
- a large positive input-gate pre-activation makes ``exp(-m_t)`` -- the mLSTM's
  own divisor floor -- underflow to ``0.0``.

RED PROOF (recorded in the plan's progress notes): restoring the bare ``1e-8``
into the REAL source file turns all four tests below RED under ``mixed_float16``
ONLY -- the ``float32`` and ``float64`` arms of the shared ``dtype_policy``
fixture pass either way, which is the point. A guard that fails at every policy
would not be testing the dtype at all.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.time_series.xlstm_blocks import mLSTMCell, sLSTMCell
from dl_techniques.utils.dtype_policy import stability_floor

INPUT_DIM = 3
BATCH = 2
UNITS = 4
HEADS = 2


class TestTheFloorHazardIsReal:
    """Anti-vacuity: prove the hazard exists before asserting any defense."""

    def test_the_original_literal_is_exactly_zero_in_float16(self):
        assert np.float16(1e-8) == np.float16(0.0)
        # ...and it is NOT zero in float32, which is why the suite was green.
        assert np.float32(1e-8) > np.float32(0.0)

    def test_the_policy_floor_is_strictly_positive_in_float16(self):
        assert stability_floor("float16", 1e-8) > 0.0
        assert np.float16(stability_floor("float16", 1e-8)) > np.float16(0.0)
        # A no-op in the wide dtypes: no float32 number may move.
        assert stability_floor("float32", 1e-8) == 1e-8
        assert stability_floor("float64", 1e-8) == 1e-8


def _saturated_slstm(i_bias: float, f_bias: float) -> sLSTMCell:
    """Build an sLSTMCell whose gate pre-activations are pinned by its bias."""
    cell = sLSTMCell(units=UNITS)
    x = keras.ops.zeros((BATCH, INPUT_DIM), dtype=cell.compute_dtype)
    cell(x, cell.get_initial_state(batch_size=BATCH))

    bias = np.zeros((4 * UNITS,), np.float32)
    bias[0:UNITS] = i_bias           # gate order is [i, f, o, z]
    bias[UNITS:2 * UNITS] = f_bias
    cell.set_weights([
        np.zeros((INPUT_DIM, 4 * UNITS), np.float32),
        np.zeros((UNITS, 4 * UNITS), np.float32),
        bias,
    ])
    return cell


def _saturated_mlstm(i_bias: float, f_bias: float, kernel_fill: float) -> mLSTMCell:
    """Build an mLSTMCell whose gate pre-activations are pinned by its bias."""
    cell = mLSTMCell(units=UNITS, num_heads=HEADS)
    x = keras.ops.ones((BATCH, INPUT_DIM), dtype=cell.compute_dtype)
    cell(x, cell.get_initial_state(batch_size=BATCH))

    head_dim = UNITS // HEADS
    qkv = 3 * HEADS * head_dim  # projection order is [q, k, v, i, f, o]
    total = qkv + 2 * HEADS + UNITS

    bias = np.zeros((total,), np.float32)
    bias[qkv:qkv + HEADS] = i_bias
    bias[qkv + HEADS:qkv + 2 * HEADS] = f_bias
    kernel = np.zeros((INPUT_DIM, total), np.float32)
    kernel[:, :qkv] = kernel_fill
    cell.set_weights([kernel, np.zeros((UNITS, total), np.float32), bias])
    return cell


def _assert_compute_dtype_is_the_policy(tensor, cell, policy_name: str) -> None:
    """Anti-vacuity: a subject that silently ran in float32 proves nothing."""
    assert keras.backend.standardize_dtype(tensor.dtype) == cell.compute_dtype
    if policy_name == "mixed_float16":
        assert cell.compute_dtype == "float16"


class TestTheSLSTMFloorsSurviveTheComputeDtype:
    """`sLSTMCell` -- the `log` floor and the `c_t / n_t` floor."""

    def test_the_divide_floor_keeps_a_dead_input_gate_finite(self, dtype_policy):
        # i_proj = -30 => i_t underflows, so n_t and c_t are both exactly 0.
        cell = _saturated_slstm(i_bias=-30.0, f_bias=10.0)
        x = keras.ops.zeros((BATCH, INPUT_DIM), dtype=cell.compute_dtype)

        h, _ = cell(x, cell.get_initial_state(batch_size=BATCH))

        _assert_compute_dtype_is_the_policy(h, cell, dtype_policy)
        values = keras.ops.convert_to_numpy(h)
        assert np.all(np.isfinite(values)), (
            f"c_t / (n_t + eps) went non-finite under {dtype_policy}: {values}"
        )

    def test_the_log_floor_keeps_a_saturated_forget_gate_differentiable(
        self, dtype_policy
    ):
        # f_proj = -30 => sigmoid underflows, so log(0) = -inf and d/dx is inf.
        # The forward pass alone is FINITE here; only the gradient exposes it.
        cell = _saturated_slstm(i_bias=0.0, f_bias=-30.0)
        x = keras.ops.ones((BATCH, INPUT_DIM), dtype=cell.compute_dtype)

        with tf.GradientTape() as tape:
            h, _ = cell(x, cell.get_initial_state(batch_size=BATCH))
            loss = keras.ops.sum(keras.ops.cast(h, "float32"))
        grads = tape.gradient(loss, cell.trainable_weights)

        _assert_compute_dtype_is_the_policy(h, cell, dtype_policy)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(h)))
        for weight, grad in zip(cell.trainable_weights, grads):
            assert grad is not None, weight.path
            assert np.all(np.isfinite(keras.ops.convert_to_numpy(grad))), (
                f"d(loss)/d({weight.path}) went non-finite under {dtype_policy}"
            )


class TestTheMLSTMFloorsSurviveTheComputeDtype:
    """`mLSTMCell` -- the `log` floor and the `max(|nq|, exp(-m_t))` floor."""

    def test_the_normalizer_floor_survives_an_underflowed_exp_neg_m(
        self, dtype_policy
    ):
        # i_proj = +30 => exp(-m_t) underflows in float16; k_proj = 0 => nq = 0,
        # so the mLSTM's own divisor floor is exactly zero and only eps is left.
        cell = _saturated_mlstm(i_bias=30.0, f_bias=0.0, kernel_fill=0.0)
        x = keras.ops.ones((BATCH, INPUT_DIM), dtype=cell.compute_dtype)

        h, _ = cell(x, cell.get_initial_state(batch_size=BATCH))

        _assert_compute_dtype_is_the_policy(h, cell, dtype_policy)
        values = keras.ops.convert_to_numpy(h)
        assert np.all(np.isfinite(values)), (
            f"the mLSTM retrieval divide went non-finite under {dtype_policy}: "
            f"{values}"
        )

    def test_the_log_floor_keeps_a_saturated_forget_gate_differentiable(
        self, dtype_policy
    ):
        cell = _saturated_mlstm(i_bias=0.0, f_bias=-30.0, kernel_fill=0.3)
        x = keras.ops.ones((BATCH, INPUT_DIM), dtype=cell.compute_dtype)

        with tf.GradientTape() as tape:
            h, _ = cell(x, cell.get_initial_state(batch_size=BATCH))
            loss = keras.ops.sum(keras.ops.cast(h, "float32"))
        grads = tape.gradient(loss, cell.trainable_weights)

        _assert_compute_dtype_is_the_policy(h, cell, dtype_policy)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(h)))
        for weight, grad in zip(cell.trainable_weights, grads):
            assert grad is not None, weight.path
            assert np.all(np.isfinite(keras.ops.convert_to_numpy(grad))), (
                f"d(loss)/d({weight.path}) went non-finite under {dtype_policy}"
            )
