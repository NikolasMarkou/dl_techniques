"""Test Suite for RingAttention Layer.

First-ever regression coverage for RingAttention (blockwise online-softmax
attention). Ring attention is normally a distributed/multi-device algorithm;
this implementation runs single-process with internal nested blockwise loops
(``range(num_blocks)`` over query and key/value blocks). Tests keep ``seq_len``
SMALL to bound the unrolled graph and exercise both the single-block
(``block_size >= seq_len``) and multi-block paths.

Coverage:
1. Initialization & Configuration
2. Input Validation
3. Forward Pass (single-block + multi-block, minimal single-process config)
4. Determinism (no dropout -> deterministic; exact softmax equivalence)
5. Serialization (get_config / from_config + full .keras model round-trip)
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

from dl_techniques.layers.attention.ring_attention import RingAttention


# ==============================================================================
# 1. Initialization & Configuration
# ==============================================================================

class TestInitialization:
    """Tests for layer initialization and parameter storage."""

    def test_defaults(self):
        layer = RingAttention(dim=16, num_heads=2)
        assert layer.dim == 16
        assert layer.num_heads == 2
        assert layer.head_dim == 8
        assert layer.block_size == 512
        assert layer.dropout_rate == 0.0
        assert layer.use_bias is False
        expected_scale = 1.0 / np.sqrt(8)
        np.testing.assert_allclose(layer.scale, expected_scale)

    def test_custom_config(self):
        layer = RingAttention(
            dim=32,
            num_heads=4,
            block_size=4,
            dropout_rate=0.1,
            use_bias=True,
        )
        assert layer.dim == 32
        assert layer.num_heads == 4
        assert layer.head_dim == 8
        assert layer.block_size == 4
        assert layer.dropout_rate == 0.1
        assert layer.use_bias is True


# ==============================================================================
# 2. Input Validation
# ==============================================================================

class TestValidation:
    """Tests for __init__ validation."""

    def test_invalid_dim_negative(self):
        with pytest.raises(ValueError, match="dim must be positive"):
            RingAttention(dim=-8, num_heads=2)

    def test_invalid_heads(self):
        with pytest.raises(ValueError, match="num_heads must be positive"):
            RingAttention(dim=16, num_heads=0)

    def test_invalid_block_size(self):
        with pytest.raises(ValueError, match="block_size must be positive"):
            RingAttention(dim=16, num_heads=2, block_size=0)

    def test_invalid_divisibility(self):
        with pytest.raises(ValueError, match="must be divisible"):
            RingAttention(dim=10, num_heads=3)

    def test_invalid_dropout(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            RingAttention(dim=16, num_heads=2, dropout_rate=1.5)


# ==============================================================================
# 3. Forward Pass (single-process minimal config)
# ==============================================================================

class TestForward:
    """Forward-pass tests at minimal single-process configs."""

    def test_output_shape_single_block(self):
        """block_size >= seq_len -> num_blocks == 1 (single blockwise iteration)."""
        x = keras.random.normal((2, 6, 16))
        layer = RingAttention(dim=16, num_heads=2, block_size=8)
        out = layer(x)
        # Output shape must equal input shape.
        assert out.shape == (2, 6, 16)
        assert not np.any(np.isnan(np.array(out)))

    def test_output_shape_multi_block(self):
        """block_size < seq_len -> multiple blocks (nested O(num_blocks^2) loop)."""
        x = keras.random.normal((2, 6, 16))
        layer = RingAttention(dim=16, num_heads=2, block_size=3)
        out = layer(x)
        assert out.shape == (2, 6, 16)
        assert not np.any(np.isnan(np.array(out)))

    def test_return_attention_weights_is_none(self):
        """Blockwise processing never materializes the full attention matrix."""
        x = keras.random.normal((1, 4, 16))
        layer = RingAttention(dim=16, num_heads=2, block_size=4)
        out, weights = layer(x, return_attention_weights=True)
        assert out.shape == (1, 4, 16)
        assert weights is None

    def test_variable_batch(self):
        layer = RingAttention(dim=16, num_heads=2, block_size=4)
        out1 = layer(keras.random.normal((1, 4, 16)))
        out2 = layer(keras.random.normal((3, 4, 16)))
        assert out1.shape[0] == 1
        assert out2.shape[0] == 3


# ==============================================================================
# 4. Determinism (no dropout -> exact, no randomness in forward)
# ==============================================================================

class TestDeterminism:
    """Ring attention has no per-forward randomness (deterministic given weights)."""

    def test_deterministic_inference(self):
        x = keras.random.normal((1, 6, 16))
        layer = RingAttention(dim=16, num_heads=2, block_size=4)
        out1 = layer(x, training=False)
        out2 = layer(x, training=False)
        np.testing.assert_allclose(np.array(out1), np.array(out2), atol=1e-6)

    def test_blockwise_matches_singleblock(self):
        """Online-softmax exactness: 1-block and multi-block configs of the SAME
        layer weights must produce identical output (the whole point of Ring)."""
        x = keras.random.normal((2, 6, 16))
        layer = RingAttention(dim=16, num_heads=2, block_size=8)
        out_single = layer(x, training=False)

        # Reuse identical weights with a smaller block_size by cloning config+weights.
        layer_multi = RingAttention(dim=16, num_heads=2, block_size=3)
        layer_multi.build(x.shape)
        layer_multi.set_weights(layer.get_weights())
        out_multi = layer_multi(x, training=False)

        np.testing.assert_allclose(
            np.array(out_single), np.array(out_multi), atol=1e-5
        )


# ==============================================================================
# 5. Serialization
# ==============================================================================

class TestSerialization:

    def test_get_config(self):
        layer = RingAttention(dim=32, num_heads=4, block_size=8, dropout_rate=0.2)
        config = layer.get_config()
        assert config["dim"] == 32
        assert config["num_heads"] == 4
        assert config["block_size"] == 8
        assert config["dropout_rate"] == 0.2

    def test_from_config(self):
        layer = RingAttention(dim=16, num_heads=2, block_size=4)
        config = layer.get_config()
        rebuilt = RingAttention.from_config(config)
        assert rebuilt.dim == 16
        assert rebuilt.num_heads == 2
        assert rebuilt.block_size == 4

    def test_model_save_load_loop(self):
        """Full .keras save/load round-trip; deterministic -> assert exact output."""
        inputs = keras.Input(shape=(6, 16))
        x = RingAttention(dim=16, num_heads=2, block_size=4)(inputs)
        model = keras.Model(inputs, x)

        x_in = np.random.normal(size=(2, 6, 16)).astype("float32")
        pred_orig = model.predict(x_in, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ring_attention.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            pred_load = loaded.predict(x_in, verbose=0)

        assert pred_orig.shape == pred_load.shape
        np.testing.assert_allclose(pred_orig, pred_load, atol=1e-6)


# ==============================================================================
# 6. Edge Cases
# ==============================================================================

class TestEdgeCases:

    def test_compute_output_shape(self):
        layer = RingAttention(dim=16, num_heads=2)
        assert layer.compute_output_shape((2, 6, 16)) == (2, 6, 16)

    def test_kwargs_passthrough(self):
        layer = RingAttention(dim=16, num_heads=2, name="ring_special")
        assert layer.name == "ring_special"

    def test_gradient_flow(self):
        layer = RingAttention(dim=16, num_heads=2, block_size=4)
        x = keras.random.normal((1, 6, 16))
        with tf.GradientTape() as tape:
            out = layer(x)
            loss = tf.reduce_mean(out)
        grads = tape.gradient(loss, layer.w_q.trainable_variables)
        assert all(g is not None and tf.reduce_any(g != 0) for g in grads)


class TestGraphSafetyRegression:
    """plan_2026-06-14_ab855e7e F3/D-002: static-seq fail-loud guard.

    The block-wise loop needs a Python-int block count; a dynamic (None)
    sequence dim must raise a clear ValueError, not crash cryptically under
    @tf.function. Static-shape forward must keep working.
    """

    def test_dynamic_seq_raises_valueerror(self):
        # call() executes under @tf.function tracing (not the functional/
        # compute_output_shape path); a None seq dim must fail loud.
        layer = RingAttention(dim=16, num_heads=2, block_size=4)
        _ = layer(tf.random.normal((2, 12, 16)))  # build first

        @tf.function
        def traced(x):
            return layer(x)

        spec = tf.TensorSpec(shape=(None, None, 16), dtype=tf.float32)
        with pytest.raises(ValueError, match="statically-known sequence length"):
            traced.get_concrete_function(spec)

    def test_static_seq_still_works(self):
        layer = RingAttention(dim=16, num_heads=2, block_size=4)
        x = tf.random.normal((2, 12, 16))
        out = layer(x)
        assert tuple(out.shape) == (2, 12, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------
# Mixed-precision mask tests (plan-2026-07-27-b4ef45f0, step 5)
# ---------------------------------------------------------------------
#
# WHAT IS BEING GUARDED HERE.
#
# `RingAttention._blockwise_attention` used the ARITHMETIC mask form
#
#     mask_slice = ops.cast(mask_slice, scores.dtype)
#     mask_slice = (1.0 - mask_slice) * -1e9
#     scores = scores + mask_slice
#
# which `common.MASK_BIAS_VALUE`'s own docstring rules out. Under
# `mixed_precision.set_global_policy('mixed_float16')` the literal `-1e9` is
# materialized in float16, where it is `-inf`; at every UNMASKED position
# `(1.0 - mask) == 0`, so the product is `0 * -inf = NaN`.
#
# MEASURED on unfixed HEAD (B=2, N=64, dim=64, num_heads=4, block_size=16), GPU 1 /
# TF 2.18, non-finite entries in the layer OUTPUT:
#
#     policy          no mask   all-ones   right-pad   left-pad   causal
#     float32          0/8192     0/8192      0/8192     0/8192   0/8192
#     mixed_float16    0/8192  8192/8192   8192/8192  8192/8192 8192/8192
#     float64          0/8192     0/8192      0/8192     0/8192   0/8192
#
# THIS SITE IS STRUCTURALLY DIFFERENT FROM THE OTHER NINE, and the difference
# drives two decisions that these tests pin (decisions.md D-011):
#
#  1. THE PER-BLOCK RESCUE WOULD BREAK CAUSALITY. The softmax here is ONLINE: the
#     mask is applied to one `(q_block, kv_block)` tile at a time. Under a causal
#     mask every strictly-upper tile is ENTIRELY masked, so
#     `apply_attention_mask(..., rescue_axis=-1)` — "a row that keeps nothing keeps
#     everything" — would resurrect exactly those tiles and let a query attend to
#     the future, with no exception and a perfectly finite output. This site
#     therefore passes `rescue_axis=None` and performs ONE rescue over the FULL key
#     axis before the loop, which is what the parameter means when the key axis is
#     not split. `TestRingAttentionCausalitySurvivesTheRescue` is the guard;
#     injecting the per-block rescue makes it fail while every finiteness test still
#     passes.
#
#  2. THE ONLINE-SOFTMAX STATE CANNOT HOLD `-inf`. `running_max` starts at `-inf`,
#     so a tile whose rows are ALL masked gives `block_max = -inf`, and
#     `running_max - new_max` is then `-inf - -inf = NaN` — the whole row is lost.
#     With a FINITE `MASK_BIAS_VALUE` the same algebra is exactly correct (the tile
#     contributes `exp(-1e9 - new_max) == 0`, and a later tile's larger max
#     renormalizes any transient away), which is why float32 is fine today and fp16
#     is not. The accumulation therefore runs in `common.mask_dtype(...)` WHENEVER A
#     MASK IS SUPPLIED; the no-mask path keeps the compute dtype and traces the same
#     graph as before. The `left_padding` mask below is the case that makes this
#     non-optional: it blanks the first two kv blocks outright.
#
# ANTI-VACUITY. The `N = 7`-hides-an-fp16-`-inf`-at-`N >= 512` trap does not
# transfer for the `0 * -inf` product (it is per-element), but the BLOCK structure
# does matter here, so the tests assert that the chosen `block_size` really splits
# the sequence into several blocks and that `left_padding` really blanks whole
# tiles — see `TestRingAttentionMaskHazardIsReal`.

from dl_techniques.layers.attention.common import MASK_BIAS_VALUE

_MP_B, _MP_N, _MP_DIM, _MP_H, _MP_BLOCK = 2, 64, 64, 4, 16
_MP_KEEP = _MP_N // 2
_MP_DEG_ROW = 5
_MP_SEED = 1234

# Absolute tolerance for "this policy's masked forward agrees with the float32 one".
#
# PRE-REGISTERED, not tuned after the fact: sized from this layer's own NO-MASK
# dtype error measured on unfixed HEAD, max |policy - float32| = 0.0108
# (mixed_float16) and 0.0054 (float64) against an output absmax of 5.94. The
# entries below carry ~5x headroom on that. Both are UPPER bounds, so a
# TF32-disabled session (which only shrinks the float64 figure) is covered too.
# That regime is no longer something this file can INHERIT: the four import-time
# process-global `enable_tensor_float_32_execution(False)` calls (including
# `test_linear_attention.py`'s) are now scoped to their own modules by the
# `tf32_disabled` fixture in `tests/test_layers/conftest.py`, so this file runs at
# the ambient default in both file-scoped and directory-scoped runs. The bounds are
# unchanged because they were already sized for the WORSE (TF32-on) regime.
_MP_ATOL = {"float32": 1e-6, "mixed_float16": 0.05, "float64": 0.05}


def _mp_input():
    """Deterministic ``(B, N, dim)`` float32 input, shared by every test below."""
    return np.random.default_rng(7).standard_normal(
        (_MP_B, _MP_N, _MP_DIM)
    ).astype("float32")


def _mp_mask(kind):
    """One of the masks these tests need, as a float32 rank-3 ``1 = keep`` array.

    ``'left_padding'`` is the ring-specific case: it blanks the first half of the
    KEYS, i.e. the first two kv BLOCKS entirely, which is what puts an all-masked
    tile at the very start of the online-softmax accumulation.
    """
    if kind == "all_ones":
        return np.ones((_MP_B, _MP_N, _MP_N), dtype="float32")
    if kind == "padding":
        m = np.ones((_MP_B, _MP_N, _MP_N), dtype="float32")
        m[:, :, _MP_KEEP:] = 0.0
        return m
    if kind == "left_padding":
        m = np.ones((_MP_B, _MP_N, _MP_N), dtype="float32")
        m[:, :, :_MP_KEEP] = 0.0
        return m
    if kind == "causal":
        return np.broadcast_to(
            np.tril(np.ones((_MP_N, _MP_N), dtype="float32")),
            (_MP_B, _MP_N, _MP_N),
        ).copy()
    if kind == "degenerate":
        m = np.ones((_MP_B, _MP_N, _MP_N), dtype="float32")
        m[:, _MP_DEG_ROW, :] = 0.0
        return m
    raise ValueError(f"unknown mask kind {kind!r}")


def _mp_layer(**kwargs):
    """A built layer whose TRAINABLE weights are identical under every dtype policy."""
    layer = RingAttention(
        dim=_MP_DIM, num_heads=_MP_H, block_size=_MP_BLOCK, **kwargs
    )
    layer.build((_MP_B, _MP_N, _MP_DIM))
    rng = np.random.default_rng(_MP_SEED)
    for weight in layer.trainable_weights:
        shape = tuple(weight.shape)
        if len(shape) == 1 and ("bias" in weight.name or "beta" in weight.name):
            value = np.zeros(shape)
        elif len(shape) == 1:
            value = 1.0 + 0.1 * rng.standard_normal(shape)
        else:
            value = 0.2 * rng.standard_normal(shape)
        weight.assign(keras.ops.cast(
            keras.ops.convert_to_tensor(value.astype("float32")), weight.dtype
        ))
    return layer


def _mp_forward(layer, array, mask):
    """One masked forward pass, returned as float64 numpy."""
    out = layer(
        keras.ops.convert_to_tensor(array),
        attention_mask=(None if mask is None else keras.ops.convert_to_tensor(mask)),
    )
    return keras.ops.convert_to_numpy(out).astype("float64")


_F32_REFERENCE = {}


def _float32_reference(kind):
    """Masked float32 output for ``kind``, memoized, under an explicit policy."""
    if kind not in _F32_REFERENCE:
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("float32")
        try:
            _F32_REFERENCE[kind] = _mp_forward(
                _mp_layer(), _mp_input(), _mp_mask(kind)
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)
    return _F32_REFERENCE[kind]


class TestRingAttentionMaskHazardIsReal:
    """Anti-vacuity. If these stop holding, every fp16 test below is worthless."""

    def test_policy_really_selects_float16_compute(self, dtype_policy):
        expected = {
            "float32": "float32",
            "mixed_float16": "float16",
            "float64": "float64",
        }[dtype_policy]
        assert keras.mixed_precision.global_policy().compute_dtype == expected

    def test_the_arithmetic_form_really_is_nan_in_the_compute_dtype(self):
        with np.errstate(over="ignore", invalid="ignore"):
            bias = np.float16(MASK_BIAS_VALUE)
            assert np.isneginf(bias), (
                "anti-vacuity FAILED: float16(MASK_BIAS_VALUE) is not -inf"
            )
            assert np.isnan(np.float16(0.0) * bias), (
                "anti-vacuity FAILED: float16(0) * float16(MASK_BIAS_VALUE) is not NaN"
            )
        assert np.isfinite(np.float32(MASK_BIAS_VALUE))
        with np.errstate(invalid="ignore"):
            assert np.isnan(np.float32(-np.inf) - np.float32(-np.inf)), (
                "anti-vacuity FAILED: `-inf - -inf` is not NaN, so the online-softmax "
                "`running_max - new_max` hazard this site's dtype choice guards "
                "against is not reproducible"
            )

    def test_the_sequence_really_splits_into_several_blocks(self):
        assert _MP_N // _MP_BLOCK >= 4, (
            f"seq_len {_MP_N} / block_size {_MP_BLOCK} gives fewer than 4 blocks; "
            "the blockwise online-softmax hazards below are not exercised"
        )

    def test_left_padding_really_blanks_whole_kv_blocks(self):
        mask = _mp_mask("left_padding")
        first_tile = mask[:, :_MP_BLOCK, :_MP_BLOCK]
        assert int((first_tile != 0).sum()) == 0, (
            "the 'left_padding' mask does not blank the FIRST (q_block 0, kv_block "
            "0) tile, so it cannot reach the `-inf - -inf = NaN` state of the "
            "online softmax"
        )

    def test_the_causal_mask_really_blanks_whole_off_diagonal_tiles(self):
        mask = _mp_mask("causal")
        upper_tile = mask[:, :_MP_BLOCK, _MP_BLOCK:2 * _MP_BLOCK]
        assert int((upper_tile != 0).sum()) == 0, (
            "the causal mask does not fully blank the (q_block 0, kv_block 1) tile, "
            "so `TestRingAttentionCausalitySurvivesTheRescue` cannot detect a "
            "per-block rescue"
        )

    def test_the_all_ones_mask_really_masks_nothing(self):
        assert int((_mp_mask("all_ones") == 0).sum()) == 0

    def test_the_degenerate_mask_really_has_exactly_one_fully_masked_row(self):
        empty = (_mp_mask("degenerate") == 0).all(axis=-1)
        assert int(empty.sum()) == _MP_B and bool(empty[:, _MP_DEG_ROW].all())


class TestRingAttentionMixedPrecisionMask:
    """SC1 + SC2: finite AND agreeing with float32, for every legal mask."""

    @pytest.mark.parametrize(
        "kind", ["all_ones", "padding", "left_padding", "causal"]
    )
    def test_masked_forward_is_finite_and_matches_float32(self, dtype_policy, kind):
        out = _mp_forward(_mp_layer(), _mp_input(), _mp_mask(kind))

        n_bad = int((~np.isfinite(out)).sum())
        assert n_bad == 0, (
            f"{n_bad}/{out.size} non-finite output entries for a {kind!r} mask "
            f"under policy {dtype_policy!r}"
        )

        reference = _float32_reference(kind)
        atol = _MP_ATOL[dtype_policy]
        max_dev = float(np.abs(out - reference).max())
        assert max_dev <= atol, (
            f"{kind!r}-masked forward under {dtype_policy!r} deviates from the "
            f"float32 control by {max_dev:.4g} > {atol:.4g}"
        )
        assert float(np.abs(out).max()) > 0.5 * float(np.abs(reference).max())


class TestRingAttentionFullyMaskedRow:
    """A fully-masked query row is RESCUED — over the FULL key axis, not per block."""

    def test_a_fully_masked_row_is_finite_and_matches_float32(self, dtype_policy):
        out = _mp_forward(_mp_layer(), _mp_input(), _mp_mask("degenerate"))
        n_bad = int((~np.isfinite(out)).sum())
        assert n_bad == 0, (
            f"{n_bad}/{out.size} non-finite output entries for a mask with a "
            f"FULLY-MASKED query row, under policy {dtype_policy!r}"
        )
        reference = _float32_reference("degenerate")
        atol = _MP_ATOL[dtype_policy]
        max_dev = float(np.abs(out - reference).max())
        assert max_dev <= atol, (
            f"the degenerate-masked forward under {dtype_policy!r} deviates from "
            f"the float32 control by {max_dev:.4g} > {atol:.4g}"
        )

    def test_the_rescued_row_behaves_as_if_it_kept_everything(self, dtype_policy):
        """The rescue's SEMANTICS, matching the other five sites in the package.

        ANTI-VACUITY: on the pre-fix code this fails in float32 too — the online
        softmax gave that row a UNIFORM average over all keys (every tile
        contributing `exp(-1e9 - -1e9) == 1`), which is a different answer from
        `softmax(unmasked logits)`.
        """
        degenerate = _mp_forward(_mp_layer(), _mp_input(), _mp_mask("degenerate"))
        all_ones = _mp_forward(_mp_layer(), _mp_input(), _mp_mask("all_ones"))
        max_dev = float(np.abs(degenerate - all_ones).max())
        atol = _MP_ATOL[dtype_policy]
        assert max_dev <= atol, (
            f"under {dtype_policy!r} the 'degenerate' mask does not behave like the "
            f"'all_ones' mask: max deviation {max_dev:.4g} > {atol:.4g}"
        )
        assert float(np.abs(all_ones).max()) > 0.0


class TestRingAttentionCausalitySurvivesTheRescue:
    """decisions.md D-011: the degenerate-row rescue must NOT be applied per block.

    Under a causal mask every strictly-upper `(q_block, kv_block)` tile is entirely
    masked. A `rescue_axis=-1` passed to `apply_attention_mask` inside the block
    loop would read those tiles as "this row keeps nothing" and un-mask them, so a
    query would attend to the FUTURE — finite, silent, and catastrophic.

    INJECTION-PROVEN: with the per-block rescue spliced in (`rescue_axis=-1` on the
    in-loop call), this test fails in ALL THREE policies — measured leak 24.1418
    (float32) / 24.1410 (mixed_float16) / 24.1426 (float64), against 0.0 for the
    shipped code — while every finiteness test in this module still passes. That is the "inject a dead
    component, not just the bug" discipline — a finiteness-only suite cannot see it.
    """

    def test_a_future_token_cannot_reach_an_earlier_query_row(self, dtype_policy):
        layer = _mp_layer()
        mask = _mp_mask("causal")
        base_input = _mp_input()

        future = 2 * _MP_BLOCK + 3          # lives in kv_block 2
        early_rows = slice(0, _MP_BLOCK)    # query rows in q_block 0

        perturbed_future = base_input.copy()
        perturbed_future[:, future, :] += 5.0
        perturbed_past = base_input.copy()
        perturbed_past[:, 1, :] += 5.0      # a token those rows MAY see

        base = _mp_forward(layer, base_input, mask)
        assert np.isfinite(base[:, early_rows]).all(), (
            "the early query rows are not finite; the comparison below would be "
            "meaningless"
        )
        leak = float(np.abs(
            _mp_forward(layer, perturbed_future, mask)[:, early_rows]
            - base[:, early_rows]).max())
        signal = float(np.abs(
            _mp_forward(layer, perturbed_past, mask)[:, early_rows]
            - base[:, early_rows]).max())

        assert signal > 0.5, (
            f"perturbing a VISIBLE past token moved the early rows by only "
            f"{signal:.6g}; the test is vacuous"
        )
        assert leak <= 1e-3, (
            f"perturbing a FUTURE token (index {future}, kv_block "
            f"{future // _MP_BLOCK}) moved query rows 0..{_MP_BLOCK - 1} by "
            f"{leak:.6g} under {dtype_policy!r}. Causality is broken — an entirely "
            f"masked block was resurrected by a per-block degenerate-row rescue "
            f"(measured 24.14 in every policy with that injection)."
        )


class TestRingAttentionMaskPolarity:
    """SC6: the mask must suppress the MASKED positions, not the kept ones."""

    @staticmethod
    def _influence(layer, mask):
        base_input = _mp_input()
        perturbed_masked = base_input.copy()
        perturbed_masked[:, _MP_KEEP + 3, :] += 5.0      # a MASKED token
        perturbed_kept = base_input.copy()
        perturbed_kept[:, 3, :] += 5.0                   # a KEPT token

        rows = slice(0, _MP_KEEP)
        base = _mp_forward(layer, base_input, mask)
        assert np.isfinite(base[:, rows]).all(), (
            "the kept query rows are not finite; the comparison below would be "
            "meaningless"
        )
        delta_masked = float(np.abs(
            _mp_forward(layer, perturbed_masked, mask)[:, rows] - base[:, rows]
        ).max())
        delta_kept = float(np.abs(
            _mp_forward(layer, perturbed_kept, mask)[:, rows] - base[:, rows]
        ).max())
        return delta_masked, delta_kept

    def test_a_masked_token_has_no_influence_on_the_kept_rows(self, dtype_policy):
        mask = _mp_mask("padding")
        delta_masked, delta_kept = self._influence(_mp_layer(), mask)
        inverted, _ = self._influence(_mp_layer(), 1.0 - mask)

        assert delta_kept > 0.5, (
            f"perturbing a KEPT token changed the output by only {delta_kept:.6g}; "
            "the test is vacuous"
        )
        assert inverted > 0.5, (
            f"the INVERTED-polarity control moved the output by only {inverted:.6g}"
        )
        assert delta_masked <= 1e-3, (
            f"perturbing a MASKED token changed the kept query rows by "
            f"{delta_masked:.6g} under policy {dtype_policy!r} — the mask polarity "
            f"is INVERTED (control: {inverted:.6g})"
        )


class TestRingAttentionMaskedGraphSafety:
    """The masked path must still trace and jit-compile.

    The blockwise loop is already fully unrolled at trace time; the mask fix adds a
    `mask_dtype` accumulation and one `ops.any`/`ops.logical_or` pair, none of which
    may introduce a data-dependent Python branch.
    """

    def test_masked_forward_is_jit_compilable_and_matches_eager(self, dtype_policy):
        layer = _mp_layer()
        x = keras.ops.convert_to_tensor(_mp_input())
        m = keras.ops.convert_to_tensor(_mp_mask("causal"))

        eager = np.asarray(
            keras.ops.convert_to_numpy(layer(x, attention_mask=m))
        ).astype("float64")

        @tf.function(jit_compile=True)
        def compiled(inputs, mask):
            return layer(inputs, attention_mask=mask)

        jitted = np.asarray(
            keras.ops.convert_to_numpy(compiled(x, m))
        ).astype("float64")

        assert np.isfinite(jitted).all(), (
            f"the jit-compiled masked forward is non-finite under {dtype_policy!r}"
        )
        # A FLAT tolerance, deliberately not `_MP_ATOL`: this is a jit-vs-eager
        # comparison inside ONE dtype, not a cross-dtype one. XLA reassociates the
        # blockwise accumulation and may pick TF32 matmuls where eager does not, so
        # even float32 disagrees — MEASURED 0.0151 on unfixed HEAD, against an
        # output absmax of ~5.9 (0.25% relative). 0.05 keeps ~3x headroom on that
        # while still failing loudly on a NaN or a collapsed output.
        max_dev = float(np.abs(jitted - eager).max())
        assert max_dev <= 0.05, (
            f"jit-compiled output deviates from eager by {max_dev:.4g} under "
            f"{dtype_policy!r}"
        )
        assert float(np.abs(jitted).max()) > 0.5 * float(np.abs(eager).max()), (
            "the jit-compiled output collapsed relative to eager"
        )


def _mp_rank2_mask():
    """A rank-2 ``(batch, seq_len)`` KEY-padding mask, ``1 = keep``.

    Blanks the SECOND half of the keys, so it masks a lot and is not vacuous.
    """
    m = np.ones((_MP_B, _MP_N), dtype="float32")
    m[:, _MP_KEEP:] = 0.0
    return m


def _rank2_as_rank3(mask2):
    """The rank-3 mask a caller would have had to write by hand instead.

    ``m3[b, q, k] = m2[b, k]`` for every query row ``q`` — the definition of a
    key-padding mask.
    """
    return np.broadcast_to(
        mask2[:, None, :], (mask2.shape[0], _MP_N, mask2.shape[1])
    ).copy()


def _rank2_as_rank4(mask2):
    """Same mask again, at rank 4 ``(batch, num_heads, seq_q, seq_k)``."""
    return np.broadcast_to(
        mask2[:, None, None, :], (mask2.shape[0], _MP_H, _MP_N, mask2.shape[1])
    ).copy()


class TestRingAttentionRank2MaskDispatch:
    """Step 7 (a). The FLIPPED form of the old
    `TestRingAttentionRank2MaskIsStillAKnownDefect`.

    Until step 7 this module PINNED the defect: a rank-2 `(batch, seq_len)`
    key-padding mask — the shape most Keras callers produce — fell through both
    branches of the in-loop rank dispatch, leaving `mask_slice` unbound.

    PROVEN RED on the unfixed code (`04caa0e7`): every equivalence test below
    ended in
    `UnboundLocalError: cannot access local variable 'mask_slice' where it is not
    associated with a value`, raised from `ring_attention.py` on the FIRST
    (q_block 0, kv_block 0) iteration — and `test_an_unsupported_rank_raises...`
    ended in `Failed: DID NOT RAISE <class 'ValueError'>` because a rank-5 mask hit
    the same `UnboundLocalError` instead of a named diagnostic.

    The load-bearing assertion here is NOT "it no longer crashes" — that is
    satisfied by any expansion, including a wrong one. It is that the rank-2 result
    is NUMERICALLY IDENTICAL to the same mask pre-expanded to rank 3 and to rank 4,
    which is the only thing that distinguishes a correct broadcast from one that
    silently masks the wrong axis.
    """

    def test_a_rank_2_mask_no_longer_raises_unbound_local_error(self):
        """The literal defect, kept as its own named assertion."""
        layer = _mp_layer()
        out = _mp_forward(layer, _mp_input(), _mp_rank2_mask())
        assert out.shape == (_MP_B, _MP_N, _MP_DIM)
        assert np.isfinite(out).all()

    def test_the_rank_2_mask_really_masks_something(self):
        """ANTI-VACUITY. If the rank-2 mask changed nothing, every equivalence
        assertion below would hold for a mask that was silently ignored."""
        layer = _mp_layer()
        masked = _mp_forward(layer, _mp_input(), _mp_rank2_mask())
        unmasked = _mp_forward(layer, _mp_input(), None)
        assert float(np.abs(masked - unmasked).max()) > 0.1, (
            "the rank-2 mask does not change the output at all — it is being "
            "ignored, and the equivalence tests below are vacuous"
        )

    @pytest.mark.parametrize("expand", ["rank3", "rank4"])
    def test_a_rank_2_mask_equals_the_same_mask_pre_expanded(
        self, dtype_policy, expand
    ):
        """THE assertion: rank 2 is the same computation, not merely a running one.

        A rank-2 mask is expanded to `(batch, 1, 1, kv_block)` per tile rather than
        materialized at `(batch, seq, seq)`, which keeps this layer's mask memory
        O(N). That is only legitimate if the numbers come out identical.
        """
        mask2 = _mp_rank2_mask()
        expanded = _rank2_as_rank3(mask2) if expand == "rank3" else _rank2_as_rank4(mask2)

        got = _mp_forward(_mp_layer(), _mp_input(), mask2)
        want = _mp_forward(_mp_layer(), _mp_input(), expanded)

        # A WITHIN-policy comparison of two spellings of one mask: the tolerance is
        # not `_MP_ATOL` (which budgets a CROSS-dtype comparison against float32).
        # The two paths differ only in broadcasting, so they should agree to the
        # dtype's own noise floor.
        tol = {"float32": 1e-6, "mixed_float16": 1e-3, "float64": 1e-12}[dtype_policy]
        max_dev = float(np.abs(got - want).max())
        assert max_dev <= tol, (
            f"a rank-2 (batch, seq_len) mask disagrees with the SAME mask "
            f"pre-expanded to {expand} by {max_dev:.4g} > {tol:.4g} under "
            f"{dtype_policy!r} — the rank-2 branch broadcasts over the wrong axis"
        )
        assert np.isfinite(got).all()

    def test_a_rank_2_mask_matches_the_pre_expanded_form_under_jit(self):
        """The masked ring path is jit-compiled in production; the new branch adds
        two `expand_dims` inside the already-unrolled loop and must not break it."""
        layer = _mp_layer()
        x = keras.ops.convert_to_tensor(_mp_input())
        mask2 = _mp_rank2_mask()

        @tf.function(jit_compile=True)
        def compiled(inputs, mask):
            return layer(inputs, attention_mask=mask)

        jitted = keras.ops.convert_to_numpy(
            compiled(x, keras.ops.convert_to_tensor(mask2))
        ).astype("float64")
        eager_rank3 = _mp_forward(layer, _mp_input(), _rank2_as_rank3(mask2))

        assert np.isfinite(jitted).all()
        # Flat tolerance for the same reason as
        # `TestRingAttentionMaskedGraphSafety`: XLA reassociates the blockwise
        # accumulation and may pick TF32 matmuls where eager does not (MEASURED
        # 0.0151 eager-vs-jit on this layer at float32).
        max_dev = float(np.abs(jitted - eager_rank3).max())
        assert max_dev <= 0.05, (
            f"the jit-compiled rank-2 masked forward deviates from the eager "
            f"pre-expanded rank-3 forward by {max_dev:.4g}"
        )

    def test_a_rank_2_mask_that_keeps_nothing_is_rescued(self):
        """The D-011 pre-loop rescue must cover rank 2 too.

        For a rank-2 mask `axis=-1` is the KEY axis, so "keeps nothing" is
        per-batch-item rather than per-query-row — the same thing, since every
        query row shares one key mask. The rescued result must equal an all-ones
        mask, exactly as at the other five rescue sites.
        """
        layer = _mp_layer()
        dead = np.zeros((_MP_B, _MP_N), dtype="float32")
        rescued = _mp_forward(layer, _mp_input(), dead)
        all_ones = _mp_forward(layer, _mp_input(), np.ones((_MP_B, _MP_N), "float32"))
        assert np.isfinite(rescued).all()
        assert float(np.abs(rescued - all_ones).max()) <= 1e-6, (
            "an all-zero rank-2 mask does not behave like an all-ones one; the "
            "pre-loop rescue does not reach the rank-2 path"
        )

    def test_a_rank_2_mask_preserves_polarity(self):
        """SC6 for the new branch: the MASKED keys must be the suppressed ones.

        The comparison is restricted to the query rows `0 .. _MP_KEEP` exactly as
        `TestRingAttentionMaskPolarity` does. A rank-2 mask masks KEYS, not
        queries, so the masked token's OWN query row legitimately moves when that
        token is perturbed — measured 7.75 — and reading the whole output would
        make this test fail on correct code.
        """
        layer = _mp_layer()
        mask = _mp_rank2_mask()
        base_input = _mp_input()
        rows = slice(0, _MP_KEEP)

        perturbed_masked = base_input.copy()
        perturbed_masked[:, _MP_KEEP + 3, :] += 5.0
        perturbed_kept = base_input.copy()
        perturbed_kept[:, 3, :] += 5.0

        base = _mp_forward(layer, base_input, mask)[:, rows]
        delta_masked = float(np.abs(
            _mp_forward(layer, perturbed_masked, mask)[:, rows] - base).max())
        delta_kept = float(np.abs(
            _mp_forward(layer, perturbed_kept, mask)[:, rows] - base).max())
        inverted_base = _mp_forward(layer, base_input, 1.0 - mask)[:, rows]
        inverted = float(np.abs(
            _mp_forward(layer, perturbed_masked, 1.0 - mask)[:, rows]
            - inverted_base).max())

        assert delta_kept > 0.5, f"vacuous: a KEPT token moved it by {delta_kept:.6g}"
        assert inverted > 0.5, (
            f"the INVERTED-polarity control moved the output by only {inverted:.6g}"
        )
        assert delta_masked <= 1e-3, (
            f"perturbing a MASKED key moved the output by {delta_masked:.6g} — the "
            f"rank-2 mask polarity is inverted (control: {inverted:.6g})"
        )

    @pytest.mark.parametrize("rank", [1, 5])
    def test_an_unsupported_rank_raises_a_named_value_error(self, rank):
        """No rank may reach the in-loop dispatch unhandled ever again.

        Pre-fix a rank-5 mask produced `UnboundLocalError` from deep inside the
        block loop; rank 1 produced the same. Both must now be a named
        `ValueError` naming the ranks that ARE supported.
        """
        layer = _mp_layer()
        shape = (_MP_N,) if rank == 1 else (_MP_B, 1, _MP_H, _MP_N, _MP_N)
        mask = np.ones(shape, dtype="float32")
        with pytest.raises(ValueError, match=r"attention_mask of rank 2"):
            layer(
                keras.ops.convert_to_tensor(_mp_input()),
                attention_mask=keras.ops.convert_to_tensor(mask),
            )

    def test_the_rank_dispatch_has_no_unbound_fallthrough(self):
        """Structural guard, not a behavioral one.

        The defect was an `if/elif` with no `else`. A future rank added to the
        pre-loop validation but not to the in-loop dispatch would re-create it
        exactly. Assert the `else: raise` is still there in the source.
        """
        import inspect
        from dl_techniques.layers.attention import ring_attention

        source = inspect.getsource(RingAttention._blockwise_attention)
        assert "_UNSUPPORTED_MASK_RANK" in source, (
            "the in-loop rank dispatch no longer raises on an unhandled rank; "
            "`mask_slice` can be read unbound again"
        )
        assert source.count("_UNSUPPORTED_MASK_RANK") >= 2, (
            "the rank diagnostic is raised from fewer than two places — either "
            "the pre-loop fail-fast check or the in-loop `else` was deleted"
        )
        assert ring_attention._UNSUPPORTED_MASK_RANK.format(rank=7).endswith(
            "got rank 7."
        )
