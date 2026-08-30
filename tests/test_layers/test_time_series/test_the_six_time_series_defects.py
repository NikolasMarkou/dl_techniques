"""
Six guards for the six code defects in ``dl_techniques.layers.time_series``.

Every test in this module asserts the CORRECTED contract, never the value the
defect currently produces. Each was proven RED against the BASE commit
``cedcaff7`` in a detached ``git worktree`` before any source line was edited;
see ``plans/plan-2026-08-30T020716-ebbaf641/verification.md`` for the six
verbatim failure lines.

The six claims, one test function each:

D-1 ``ForecastabilityGate`` declares the forecast shape, not the backcast one.
D-2 ``QuantileHead`` refuses a 3D input when ``flatten_input=False``.
D-3 ``ExogenousBlock``'s ``dropout_rate`` actually drops.
D-4 ``ExogenousBlock`` honours the inherited ``input_dim`` / ``output_dim``.
D-5 ``TemporalFusionLayer`` charges its ``activity_regularizer`` once.
D-6 ``AdaptiveLagAttentionLayer`` charges its ``activity_regularizer`` once.

**Anti-vacuity control.** This module is meant to be copied into a detached
worktree and run there. A detached worktree silently imports the MAIN repo's
``src/`` via the ambient path unless its own path is forced first, and a
measurement taken that way compares the worktree's tests against the CURRENT
code -- meaningless as a BASE control, and perfect-looking. So the resolved
``dl_techniques.__file__`` is printed at import time (run with ``-s``), and,
when ``DL_TECHNIQUES_REQUIRE_ROOT`` is set, a mismatch raises at import so the
run fails at COLLECTION rather than producing six plausible-looking failures.
"""

import os
import sys

import keras
import numpy as np
import pytest

import dl_techniques

from dl_techniques.layers.time_series.adaptive_lag_attention import AdaptiveLagAttentionLayer
from dl_techniques.layers.time_series.forecasting_layers import (
    ForecastabilityGate,
    create_manokhin_compliant_model,
)
from dl_techniques.layers.time_series.nbeats_blocks import GenericBlock
from dl_techniques.layers.time_series.nbeatsx_blocks import ExogenousBlock
from dl_techniques.layers.time_series.quantile_head_fixed_io import QuantileHead
from dl_techniques.layers.time_series.temporal_fusion import TemporalFusionLayer

# --- Anti-vacuity control: which tree did we actually import? -------------
DL_TECHNIQUES_PATH = os.path.realpath(dl_techniques.__file__)
print(f"\n[provenance] dl_techniques.__file__ = {DL_TECHNIQUES_PATH}", file=sys.stderr)

_REQUIRED_ROOT = os.environ.get("DL_TECHNIQUES_REQUIRE_ROOT")
if _REQUIRED_ROOT:
    _root = os.path.realpath(_REQUIRED_ROOT)
    if not DL_TECHNIQUES_PATH.startswith(_root + os.sep):
        raise RuntimeError(
            "Provenance control FAILED: dl_techniques resolved to "
            f"{DL_TECHNIQUES_PATH!r}, which is not under the required root "
            f"{_root!r}. Any pass/fail measured here would describe the wrong "
            "tree. Force the intended sources with PYTHONPATH=<root>/src."
        )


def _f32(*shape, seed=0):
    rng = np.random.default_rng(seed)
    return keras.ops.convert_to_tensor(rng.standard_normal(shape).astype("float32"))


# =========================================================================
# D-1
# =========================================================================
def test_the_gate_declares_the_forecast_shape():
    """
    ``ForecastabilityGate``'s declared output shape must be the shape the
    functional graph actually produces.

    ``compute_output_shape`` sees only the backcast shape, because
    ``deep_forecast`` / ``naive_forecast`` arrive as extra ``call`` arguments
    that Keras never shape-propagates to ``build``. The corrected contract is
    that the declared shape equals the runtime one; the discriminating
    quantity is the declared shape itself, cross-checked against ``predict``
    so the test cannot be satisfied by a graph and a runtime that agree on
    being wrong together.
    """
    model = create_manokhin_compliant_model(input_shape=(24, 3), forecast_length=8)

    declared = tuple(model.outputs[0].shape)
    runtime = model.predict(np.zeros((5, 24, 3), dtype="float32"), verbose=0)[0].shape

    assert declared == (None, 8, 3), (
        f"functional graph declares {declared}, expected (None, 8, 3)"
    )
    assert runtime == (5, 8, 3), f"predict returned {runtime}, expected (5, 8, 3)"
    assert declared[1:] == runtime[1:], (
        f"declared non-batch dims {declared[1:]} != runtime {runtime[1:]}"
    )


def test_the_gate_still_loads_a_config_written_before_forecast_length_existed():
    """
    Backward-compat control for the D-1 fix.

    ``forecast_length`` is a NEW ``get_config`` key, so every config written
    before this plan lacks it. ``from_config`` must still succeed there and
    leave the attribute ``None`` — otherwise the fix would break reloading of
    any archive containing this layer.
    """
    legacy = ForecastabilityGate(hidden_units=8).get_config()
    legacy.pop("forecast_length", None)
    assert "forecast_length" not in legacy

    restored = ForecastabilityGate.from_config(legacy)
    assert restored.forecast_length is None
    assert restored.hidden_units == 8

    # And the round trip WITH the key preserves it.
    kept = ForecastabilityGate.from_config(
        ForecastabilityGate(hidden_units=8, forecast_length=8).get_config()
    )
    assert kept.forecast_length == 8


# =========================================================================
# D-2
# =========================================================================
def test_the_quantile_head_refuses_a_3d_input_when_not_flattening():
    """
    ``flatten_input=False`` means "the caller has already flattened to 2D".

    A 3D input there is caller error: the head folds the sequence axis into
    batch, contradicting its own ``compute_output_shape``. The corrected
    contract is a loud ``ValueError`` naming ``flatten_input``. The 2D case,
    which ``prism`` depends on, must keep working -- asserted here as a
    control so the guard cannot be satisfied by a blanket raise.
    """
    with pytest.raises(ValueError, match="flatten_input"):
        head = QuantileHead(num_quantiles=3, output_length=5, flatten_input=False)
        head(_f32(4, 7, 16))

    # Control: the 2D path is the supported one and must be untouched.
    head_2d = QuantileHead(num_quantiles=3, output_length=5, flatten_input=False)
    assert tuple(head_2d(_f32(4, 64)).shape) == (4, 5, 3)


# =========================================================================
# D-3
# =========================================================================
def _exog_block(**kw):
    defaults = dict(
        exogenous_dim=2,
        units=16,
        thetas_dim=4,
        backcast_length=10,
        forecast_length=4,
        tcn_dropout_rate=0.0,
    )
    defaults.update(kw)
    return ExogenousBlock(**defaults)


def test_the_exogenous_dropout_actually_drops():
    """
    ``ExogenousBlock`` re-implements the parent's dense spine and must apply
    ``dropout1..dropout4`` exactly as the parent does.

    The discriminating quantity is a STRICT ``max|delta| > 0`` between two
    ``training=True`` calls on identical input at ``dropout_rate=0.5``. Two
    no-op controls bracket it: at ``dropout_rate=0.0`` and at
    ``training=False`` the same comparison must be exactly ``0.0``, so the
    guard cannot be satisfied by making the layer nondeterministic in general.
    """
    y = _f32(4, 10, seed=1)
    exog = (_f32(4, 10, 2, seed=2), _f32(4, 4, 2, seed=3))

    def _two_calls(block, training):
        b1, f1 = block(y, training=training, exogenous_inputs=exog)
        b2, f2 = block(y, training=training, exogenous_inputs=exog)
        return (
            float(np.max(np.abs(keras.ops.convert_to_numpy(b1) - keras.ops.convert_to_numpy(b2)))),
            float(np.max(np.abs(keras.ops.convert_to_numpy(f1) - keras.ops.convert_to_numpy(f2)))),
        )

    live = _exog_block(dropout_rate=0.5)
    d_back, d_fore = _two_calls(live, training=True)
    assert d_back > 0.0, (
        "dropout_rate=0.5 with training=True left the backcast bit-identical "
        f"across two calls (max|delta| == {d_back}); the Dropout layers are inert"
    )
    assert d_fore > 0.0, (
        "dropout_rate=0.5 with training=True left the forecast bit-identical "
        f"across two calls (max|delta| == {d_fore}); the Dropout layers are inert"
    )

    # Control 1: at training=False, Dropout is a no-op.
    assert _two_calls(live, training=False) == (0.0, 0.0)

    # Control 2: at dropout_rate=0.0, there is nothing to drop.
    off = _exog_block(dropout_rate=0.0)
    assert _two_calls(off, training=True) == (0.0, 0.0)


# =========================================================================
# D-4
# =========================================================================
def test_the_exogenous_block_honours_input_and_output_dim():
    """
    ``ExogenousBlock`` must honour the inherited ``input_dim`` / ``output_dim``
    like every sibling block does.

    ``NBeatsBlock.compute_output_shape`` declares
    ``(batch, backcast_length * input_dim)`` / ``(batch, forecast_length *
    output_dim)``, and ``GenericBlock`` implements it. The discriminating
    quantities are the two full shapes AND their equality to ``GenericBlock``'s
    at the same dims -- a rank- or contract-only assertion is satisfied by the
    defect, which returns a well-formed ``(4, 10)`` / ``(4, 4)``. Checked at
    ``use_tcn=True`` and ``use_tcn=False``, whose basis channel counts differ.
    """
    back_len, fore_len, in_dim, out_dim = 10, 4, 3, 2
    common = dict(
        units=16, thetas_dim=4, backcast_length=back_len,
        forecast_length=fore_len, input_dim=in_dim, output_dim=out_dim,
    )

    y = _f32(4, back_len * in_dim, seed=1)
    exog = (_f32(4, back_len, 2, seed=2), _f32(4, fore_len, 2, seed=3))

    generic = GenericBlock(**common)
    g_back, g_fore = generic(y)
    reference = (tuple(g_back.shape), tuple(g_fore.shape))
    assert reference == ((4, back_len * in_dim), (4, fore_len * out_dim)), (
        f"GenericBlock reference shapes moved: {reference}"
    )

    for use_tcn in (True, False):
        block = _exog_block(use_tcn=use_tcn, **common)
        back, fore = block(y, exogenous_inputs=exog)
        got = (tuple(back.shape), tuple(fore.shape))
        assert got == ((4, back_len * in_dim), (4, fore_len * out_dim)), (
            f"use_tcn={use_tcn}: ExogenousBlock returned {got}, expected "
            f"((4, {back_len * in_dim}), (4, {fore_len * out_dim}))"
        )
        assert got == reference, (
            f"use_tcn={use_tcn}: ExogenousBlock {got} != GenericBlock {reference} "
            "at identical input_dim/output_dim"
        )
        assert tuple(block.compute_output_shape(y.shape)[0]) == (4, back_len * in_dim)
        assert tuple(block.compute_output_shape(y.shape)[1]) == (4, fore_len * out_dim)


class _WrongLayoutBlock(ExogenousBlock):
    """
    The ONE plausible wrong fix for D-4, as a subclass — the control for the
    layout guard below.

    It reshapes theta CHANNEL-major, ``(batch, basis_channels, dim)``, contracts
    ``einsum('btc,bci->bti')`` and transposes to ``(batch, dim, time)`` before
    the flatten, so flat entry ``i*T + t`` is step ``t`` of channel ``i`` — the
    dim-major stream the anchor at ``nbeatsx_blocks.py`` says not to produce. It
    passes every assertion in the shape guard above at both ``use_tcn``
    settings, which is why the layout guard exists. Only the dense spine is
    trimmed (dropout and normalization are off in this module's fixtures); the
    projection is a faithful transcription of the wrong ordering.
    """

    def call(self, inputs, training=None, exogenous_inputs=None, **kwargs):
        x_back, x_fore = exogenous_inputs
        x = self.dense1(inputs, training=training)
        x = self.dense2(x, training=training)
        x = self.dense3(x, training=training)
        x = self.dense4(x, training=training)
        theta_b = self.theta_backcast(x, training=training)
        theta_f = self.theta_forecast(x, training=training)

        full = keras.ops.concatenate([x_back, x_fore], axis=1)
        basis = self.encoder(full, training=training) if self.use_tcn else full
        basis_b = basis[:, :self.backcast_length, :]
        basis_f = basis[:, self.backcast_length:, :]

        theta_b_r = keras.ops.reshape(theta_b, (-1, self.basis_channels, self.input_dim))
        theta_f_r = keras.ops.reshape(theta_f, (-1, self.basis_channels, self.output_dim))
        back = keras.ops.einsum('btc,bci->bti', basis_b, theta_b_r)
        fore = keras.ops.einsum('btc,bco->bto', basis_f, theta_f_r)
        back = keras.ops.transpose(back, (0, 2, 1))
        fore = keras.ops.transpose(fore, (0, 2, 1))
        back = keras.ops.reshape(back, (-1, self.backcast_length * self.input_dim))
        fore = keras.ops.reshape(fore, (-1, self.forecast_length * self.output_dim))
        return back, fore


def _flat_backcast_channel_columns(block_cls, use_tcn, theta_slot):
    """
    Drive exactly one theta slot and report which FLAT backcast columns move.

    ``theta_backcast`` is a bias-free ``Dense``, so zeroing every column of its
    kernel except ``theta_slot`` makes ``theta_b`` a one-hot vector: at most one
    output channel of the projection can be non-zero, whatever ordering the
    block uses internally. The returned column indices are therefore a direct
    readout of the flat layout, obtained WITHOUT recomputing the block's own
    einsum — an oracle rebuilt from the code under test would inherit the code's
    ordering and could not tell the two layouts apart.

    :return: ``(sorted non-zero column indices, flat width, basis_channels)``.
    """
    back_len, fore_len, in_dim, out_dim, exog_dim = 10, 4, 3, 2, 2
    block = block_cls(
        exogenous_dim=exog_dim, units=16, thetas_dim=4,
        backcast_length=back_len, forecast_length=fore_len,
        input_dim=in_dim, output_dim=out_dim,
        dropout_rate=0.0, tcn_dropout_rate=0.0, use_tcn=use_tcn,
    )
    y = _f32(4, back_len * in_dim, seed=1)
    exog = (_f32(4, back_len, exog_dim, seed=2), _f32(4, fore_len, exog_dim, seed=3))
    block(y, exogenous_inputs=exog)  # build the weights

    kernel = np.zeros(block.theta_backcast.kernel.shape, dtype="float32")
    # A varied column, not a constant one: a constant column would read
    # sum(x), which can cancel to zero and make the whole measurement vacuous.
    kernel[:, theta_slot] = np.linspace(1.0, 2.0, kernel.shape[0], dtype="float32")
    block.theta_backcast.set_weights([kernel])

    back, _ = block(y, exogenous_inputs=exog)
    flat = np.asarray(keras.ops.convert_to_numpy(back))
    columns = sorted({int(col) for col in np.nonzero(flat)[1]})
    return columns, flat.shape[1], block.basis_channels


def test_the_exogenous_block_interleaves_the_flat_stream_time_major():
    """
    ``ExogenousBlock``'s flat output must be TIME-major: entry ``t*dim + i`` is
    step ``t`` of channel ``i``, matching ``NBeatsNet``'s flatten/reshape and
    both interpretable siblings.

    The shape guard above cannot see this — the dim-major stream ``i*T + t`` has
    the identical shape, equals ``GenericBlock``'s, and satisfies both
    ``compute_output_shape`` assertions. So this guard reads the layout off a
    one-hot theta instead: driving a single theta slot leaves exactly one output
    channel live, and the discriminating quantity is WHERE its ``T`` non-zero
    entries land. Time-major puts them on one residue class mod ``dim``, spread
    with stride ``dim`` across the full width; dim-major puts them in a
    contiguous run of length ``T``. ``_WrongLayoutBlock`` is the control: it
    implements the dim-major ordering and this test rejects it (measured
    ``[20..29]`` where the shipped block gives ``[1, 4, ..., 28]``).
    """
    back_len, in_dim, theta_slot = 10, 3, 2

    for use_tcn in (True, False):
        columns, width, basis_channels = _flat_backcast_channel_columns(
            ExogenousBlock, use_tcn, theta_slot
        )
        assert width == back_len * in_dim
        # Anti-vacuity: a channel that never moves would pass any layout claim.
        assert len(columns) == back_len, (
            f"use_tcn={use_tcn}: driving one theta slot moved {len(columns)} "
            f"flat columns, expected {back_len} (one per time step); the probe "
            "measured nothing"
        )

        residues = {col % in_dim for col in columns}
        assert len(residues) == 1, (
            f"use_tcn={use_tcn}: the live channel's entries fall on residues "
            f"{sorted(residues)} mod {in_dim}, so the flat stream is not "
            f"time-major; columns={columns}"
        )

        # Under the dim-major theta reshape the block documents, slot
        # `theta_slot` belongs to channel `theta_slot // basis_channels`.
        channel = theta_slot // basis_channels
        expected = [t * in_dim + channel for t in range(back_len)]
        assert columns == expected, (
            f"use_tcn={use_tcn}: flat backcast moves at columns {columns}, "
            f"expected {expected} (t*{in_dim}+{channel}); a contiguous run "
            f"would mean the dim-major layout i*T+t"
        )


def test_the_layout_guard_rejects_the_dim_major_layout():
    """
    The RED proof for the guard above: the dim-major subclass must FAIL it.

    A layout assertion that has only ever been run against the shipped code is
    not known to discriminate. ``_WrongLayoutBlock`` produces a contiguous run
    of ``T`` columns rather than one residue class mod ``dim``, so both the
    residue assertion and the exact-column assertion must reject it.
    """
    back_len, in_dim, theta_slot = 10, 3, 2

    for use_tcn in (True, False):
        columns, _, basis_channels = _flat_backcast_channel_columns(
            _WrongLayoutBlock, use_tcn, theta_slot
        )
        assert len(columns) == back_len, "the control probe measured nothing"
        expected = [t * in_dim + theta_slot // basis_channels for t in range(back_len)]
        assert columns != expected, (
            "the dim-major control produced the time-major columns "
            f"{columns}; the layout guard cannot discriminate"
        )
        assert columns == list(range(min(columns), min(columns) + back_len)), (
            f"the dim-major control gave {columns}, expected a contiguous run"
        )


class _NotOne(int):
    """
    An ``int`` worth ``1`` that compares unequal to ``1``.

    ``ExogenousBlock.call`` selects its fast path with
    ``if self.input_dim == 1 and self.output_dim == 1``. Assigning this to
    ``input_dim`` after ``build`` falsifies that one expression while every
    arithmetic use of ``input_dim`` (the theta reshape, the flatten width) still
    reads ``1`` — so the general branch runs on unchanged weights at ``dim == 1``
    without transcribing the block's projection into the test, which is what a
    hand-written subclass would have required.
    """

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True

    def __hash__(self):
        return hash(int(self))


def test_the_two_exogenous_projection_branches_agree_at_dim_one():
    """
    The ``dim == 1`` fast path and the general ``else`` path compute the same
    quantity two ways, so they must agree.

    They are algebraically identical but not bit-identical: D-010 measured the
    float32 reassociation delta between the two contraction paths at
    ``4.768e-07`` on CPU (5 of 45 trials non-zero) and exactly ``0.0`` on GPU,
    which is why the tolerance is ``1e-5`` and not ``0.0``. That rarity is the
    hazard — an edit to one branch alone would show up almost nowhere — so this
    guard runs both branches on ONE built block with identical weights and
    compares. The branch predicate itself is asserted, so the test cannot pass
    by silently taking the fast path twice.
    """
    for use_tcn in (True, False):
        block = _exog_block(use_tcn=use_tcn, input_dim=1, output_dim=1)
        y = _f32(4, 10, seed=1)
        exog = (_f32(4, 10, 2, seed=2), _f32(4, 4, 2, seed=3))

        assert block.input_dim == 1 and block.output_dim == 1
        fast_back, fast_fore = block(y, exogenous_inputs=exog)

        block.input_dim = _NotOne(1)
        # Control: this is the exact expression `call` branches on.
        assert not (block.input_dim == 1 and block.output_dim == 1), (
            "the fast-path predicate is still true, so the general branch "
            "never ran and this comparison is vacuous"
        )
        assert int(block.input_dim) == 1 and block.backcast_length * block.input_dim == 10
        general_back, general_fore = block(y, exogenous_inputs=exog)

        for name, fast, general in (
            ("backcast", fast_back, general_back),
            ("forecast", fast_fore, general_fore),
        ):
            fast_np = np.asarray(keras.ops.convert_to_numpy(fast))
            general_np = np.asarray(keras.ops.convert_to_numpy(general))
            assert fast_np.shape == general_np.shape, (
                f"use_tcn={use_tcn}: {name} shapes diverged, "
                f"{fast_np.shape} vs {general_np.shape}"
            )
            np.testing.assert_allclose(
                fast_np, general_np, atol=1e-5, rtol=0,
                err_msg=(
                    f"use_tcn={use_tcn}: the dim==1 fast path and the general "
                    f"path disagree on the {name} beyond float32 reassociation"
                ),
            )

# =========================================================================
# D-5 / D-6
# =========================================================================
def _assert_charged_once(layer, output):
    """
    The single discriminating check for D-5 and D-6.

    ``len(losses) >= 1`` and ``losses[0] == regularizer(output)`` are BOTH
    satisfied by the defect, which produces two identical entries. Only the
    exact count ``== 1`` separates the two states, so the count is asserted
    first and the value second.
    """
    losses = layer.losses
    expected = float(keras.ops.convert_to_numpy(layer.activity_regularizer(output)))
    assert len(losses) == 1, (
        f"len(layer.losses) == {len(losses)}, expected 1; entries="
        f"{[float(keras.ops.convert_to_numpy(entry)) for entry in losses]}, "
        f"activity_regularizer(output) == {expected}"
    )
    np.testing.assert_allclose(
        float(keras.ops.convert_to_numpy(losses[0])), expected, atol=1e-6, rtol=0
    )


def test_the_temporal_fusion_regularizer_is_charged_once():
    """
    Keras 3's ``Layer.__call__`` already applies ``self.activity_regularizer``
    to the layer's output, so ``call()`` must not apply it a second time.
    """
    layer = TemporalFusionLayer(
        output_dim=3, num_lags=4, activity_regularizer=keras.regularizers.L1(1.0)
    )
    output = layer([_f32(2, 5, seed=4), _f32(2, 4, seed=5)])
    _assert_charged_once(layer, output)


def test_the_adaptive_lag_regularizer_is_charged_once():
    """
    Same contract as ``TemporalFusionLayer``: exactly one charge, applied by
    Keras, never re-applied inside ``call()``.
    """
    layer = AdaptiveLagAttentionLayer(
        num_lags=4, activity_regularizer=keras.regularizers.L1(1.0)
    )
    output = layer([_f32(2, 5, seed=4), _f32(2, 4, seed=5)])
    _assert_charged_once(layer, output)
