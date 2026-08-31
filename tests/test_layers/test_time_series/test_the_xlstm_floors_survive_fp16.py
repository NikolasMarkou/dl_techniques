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

**Finiteness is only half of it, and it was the wrong half to stop at.** The
first version of this file asserted ``isfinite`` and nothing else, and it
certified as "fixed" a repair that bought finiteness by raising the added
epsilon to float16's smallest NORMAL magnitude, ``6.10e-05`` -- 1000x the
smallest subnormal and ~6000x the ``1e-8`` it replaced. MEASURED against the
identical inputs in float32, that floor read the log-forget gate as ``-9.61``
where the answer is ``-12.00`` (``+2.39`` nats) and ``-9.70`` against ``-15.91``
(``+6.21`` nats), suppressed the sLSTM output divide by ``90.9%`` and the mLSTM
retrieval by ``37.3%`` -- in regimes that were EXACT before the fix, because a
bare ``1e-8`` is ``0.0`` in float16 and adding zero changes nothing. The sites
now promote to ``accumulation_dtype(compute_dtype)`` and keep ``1e-8``, and
:class:`TestTheFloorsCostNoAccuracy` is the arm that can see the difference:
each of the four numbers above falls below ``1%`` after the repair.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.time_series.xlstm_blocks import mLSTMCell, sLSTMCell
from dl_techniques.utils.dtype_policy import accumulation_dtype, stability_floor

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


def _saturated_slstm(
    i_bias: float,
    f_bias: float,
    z_bias: float = 0.0,
    dtype=None,
) -> sLSTMCell:
    """Build an sLSTMCell whose gate pre-activations are pinned by its bias."""
    cell = sLSTMCell(units=UNITS, dtype=dtype)
    x = keras.ops.zeros((BATCH, INPUT_DIM), dtype=cell.compute_dtype)
    cell(x, cell.get_initial_state(batch_size=BATCH))

    bias = np.zeros((4 * UNITS,), np.float32)
    bias[0:UNITS] = i_bias           # gate order is [i, f, o, z]
    bias[UNITS:2 * UNITS] = f_bias
    bias[3 * UNITS:4 * UNITS] = z_bias
    cell.set_weights([
        np.zeros((INPUT_DIM, 4 * UNITS), np.float32),
        np.zeros((UNITS, 4 * UNITS), np.float32),
        bias,
    ])
    return cell


def _saturated_mlstm(
    i_bias: float,
    f_bias: float,
    kernel_fill: float,
    dtype=None,
) -> mLSTMCell:
    """Build an mLSTMCell whose gate pre-activations are pinned by its bias."""
    cell = mLSTMCell(units=UNITS, num_heads=HEADS, dtype=dtype)
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


# ---------------------------------------------------------------------
# Accuracy, not merely finiteness
# ---------------------------------------------------------------------

#: Absolute tolerance, in nats, on a log-forget gate read under a half policy
#: against the float32 reading of the identical subject. MEASURED after the
#: promotion at f_proj in {-8, -12, -16}: 0.0003, 0.0016, 0.0008. Before it, on
#: the same three inputs: 0.168, 2.389, 6.212. Any value between roughly 0.01
#: and 0.15 separates the two; 0.01 is chosen as the tightest bound the measured
#: residual clears by an order of magnitude.
LOG_TOLERANCE_NATS = 0.01

#: Relative tolerance on a divide whose numerator and denominator share a scale.
#: MEASURED after the promotion: 0.59% (sLSTM output) and 0.77% (mLSTM
#: retrieval). Before it: 90.9% and 37.3%.
DIVIDE_TOLERANCE = 0.02


def _both_dtypes(build, read):
    """Run the SAME subject at ``mixed_float16`` and at ``float32``.

    The dtype is passed to the layer constructor rather than set globally, so
    this comparison mutates no process-global state and cannot leak a policy
    into the next test (the reason `tests/test_layers/conftest.py` owns the only
    `set_global_policy` in this tree).

    :param build: callable taking a ``dtype`` keyword and returning a cell.
    :param read: callable taking a cell and returning a float.
    :return: ``(half_precision_value, float32_reference_value)``.
    """
    half = build(dtype="mixed_float16")
    assert half.compute_dtype == "float16", (
        "the fp16 arm did not actually run in float16; it would agree with the "
        "float32 reference for the wrong reason"
    )
    reference = build(dtype="float32")
    assert reference.compute_dtype == "float32"
    return read(half), read(reference)


def _slstm_log_gate(f_bias: float):
    """Expose `log_f_t` as an output: with i_proj very negative, m_t == log_f_t."""
    def build(dtype):
        return _saturated_slstm(i_bias=-100.0, f_bias=f_bias, dtype=dtype)

    def read(cell):
        x = keras.ops.zeros((BATCH, INPUT_DIM), dtype=cell.compute_dtype)
        _, states = cell(x, cell.get_initial_state(batch_size=BATCH))
        return float(keras.ops.convert_to_numpy(states[3])[0, 0])

    return _both_dtypes(build, read)


def _mlstm_log_gate(f_bias: float):
    """Same trick for `mLSTMCell`, whose fourth returned state is also `m_t`."""
    def build(dtype):
        return _saturated_mlstm(
            i_bias=-100.0, f_bias=f_bias, kernel_fill=0.0, dtype=dtype
        )

    def read(cell):
        x = keras.ops.ones((BATCH, INPUT_DIM), dtype=cell.compute_dtype)
        _, states = cell(x, cell.get_initial_state(batch_size=BATCH))
        return float(keras.ops.convert_to_numpy(states[3])[0, 0])

    return _both_dtypes(build, read)


class TestTheFloorsCostNoAccuracy:
    """The fp16 answer must MATCH the float32 answer, not merely be finite.

    This is the arm the first version of this file was missing. Each subject
    below sits in a regime the added epsilon dominates -- a subnormal sigmoid, a
    subnormal normalizer -- so a coarse float16 floor is directly readable in
    the output, while a finiteness assertion cannot see it at all.
    """

    @pytest.mark.parametrize("f_proj", [-8.0, -12.0, -16.0])
    def test_the_slstm_log_forget_gate_agrees_with_float32(self, f_proj):
        half, reference = _slstm_log_gate(f_proj)
        assert abs(half - reference) <= LOG_TOLERANCE_NATS, (
            f"log(sigmoid({f_proj}) + eps) read {half:.4f} in float16 against "
            f"{reference:.4f} in float32 -- an error of {half - reference:+.4f} "
            f"nats. A coarse added epsilon reads exactly like this."
        )

    @pytest.mark.parametrize("f_proj", [-8.0, -12.0, -16.0])
    def test_the_mlstm_log_forget_gate_agrees_with_float32(self, f_proj):
        half, reference = _mlstm_log_gate(f_proj)
        assert abs(half - reference) <= LOG_TOLERANCE_NATS, (
            f"mLSTM log-forget gate read {half:.4f} in float16 against "
            f"{reference:.4f} in float32 ({half - reference:+.4f} nats)"
        )

    def test_the_slstm_output_divide_agrees_with_float32(self):
        # m_tm1 = 12 makes i_t = exp(-12) subnormal-but-exact in float16, so
        # n_t is ~6.1e-06 -- an order of magnitude BELOW float16's smallest
        # normal. c_t and n_t share that factor, so c_t / n_t is z_t either
        # way; only an epsilon comparable to n_t can move the answer.
        def build(dtype):
            return _saturated_slstm(
                i_bias=0.0, f_bias=10.0, z_bias=1.0, dtype=dtype
            )

        def read(cell):
            cd = cell.compute_dtype
            states = [
                keras.ops.zeros((BATCH, UNITS), dtype=cd),
                keras.ops.zeros((BATCH, UNITS), dtype=cd),
                keras.ops.zeros((BATCH, UNITS), dtype=cd),
                keras.ops.full((BATCH, UNITS), 12.0, dtype=cd),
            ]
            x = keras.ops.zeros((BATCH, INPUT_DIM), dtype=cd)
            h, _ = cell(x, states)
            return float(keras.ops.convert_to_numpy(h)[0, 0])

        half, reference = _both_dtypes(build, read)
        assert reference != 0.0, "the float32 reference is degenerate"
        relative = abs(half - reference) / abs(reference)
        assert relative <= DIVIDE_TOLERANCE, (
            f"h_t = o_t * c_t / (n_t + eps) read {half:.6e} in float16 against "
            f"{reference:.6e} in float32 -- {relative:.2%} off. A floor of "
            f"6.10e-05 against an n_t of ~6e-06 costs 90.9% here."
        )

    def test_the_mlstm_retrieval_divide_agrees_with_float32(self):
        # i_proj = +30 sends exp(-m_t) below float16's range, so the divisor is
        # max(|nq|, 0) + eps; the kernel fill puts |nq| at ~1e-04, i.e. within
        # a factor of two of float16's smallest normal.
        def build(dtype):
            return _saturated_mlstm(
                i_bias=30.0, f_bias=0.0, kernel_fill=0.00236, dtype=dtype
            )

        def read(cell):
            x = keras.ops.ones((BATCH, INPUT_DIM), dtype=cell.compute_dtype)
            h, _ = cell(x, cell.get_initial_state(batch_size=BATCH))
            return float(keras.ops.convert_to_numpy(h)[0, 0])

        half, reference = _both_dtypes(build, read)
        assert reference != 0.0, "the float32 reference is degenerate"
        relative = abs(half - reference) / abs(reference)
        assert relative <= DIVIDE_TOLERANCE, (
            f"the mLSTM retrieval read {half:.6e} in float16 against "
            f"{reference:.6e} in float32 -- {relative:.2%} off. A floor of "
            f"6.10e-05 added to an |nq| of 1e-04 costs 37.3% here."
        )


class TestThePromotionIsInertOutsideHalfPrecision:
    """`accumulation_dtype` must never NARROW, or float64 silently loses bits."""

    def test_only_the_reduced_precision_dtypes_are_promoted(self):
        assert accumulation_dtype("float16") == "float32"
        assert accumulation_dtype("bfloat16") == "float32"
        assert accumulation_dtype("mixed_float16") == "float32"
        # Never narrower than the input: promoting float64 DOWN to float32
        # would be the same defect pointing the other way.
        assert accumulation_dtype("float32") == "float32"
        assert accumulation_dtype("float64") == "float64"

    def test_the_floor_is_the_identity_in_every_promoted_dtype(self):
        # This is what makes the four converted sites bit-neutral at float32
        # and float64: both casts are identities and the epsilon is unchanged.
        for name in ("float16", "bfloat16", "float32", "float64"):
            accum = accumulation_dtype(name)
            assert stability_floor(accum, 1e-8) == 1e-8, name
