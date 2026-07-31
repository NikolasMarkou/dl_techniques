"""Test Suite for SingleWindowAttention Layer.

First-ever STANDALONE regression coverage for SingleWindowAttention. Prior to
this file the layer was only ``isinstance``-checked as a sub-layer inside
``test_window_attention.py``. This suite exercises it directly.

SingleWindowAttention is multi-head self-attention restricted to a single square
window of ``window_size ** 2`` tokens. Input is 3D ``[B, N, dim]`` (per-window
tokens); the layer pads internally up to ``window_size ** 2`` and strips the
padding off the output, so output shape == input shape ``[B, N, dim]``. It
supports an optional ``attention_mask`` of shape ``[B, N]`` (1 = valid, 0 =
pad) combined with the internal padding mask.

Coverage:
1. Initialization & Configuration
2. Input Validation
3. Forward Pass (output shape == input shape)
4. Masked Forward (attention_mask is honored)
5. Serialization (get_config / from_config + full .keras model round-trip)
"""

import pytest
import numpy as np
import keras
import tempfile
import os

from dl_techniques.layers.attention.single_window_attention import (
    SingleWindowAttention,
)


# ==============================================================================
# 1. Initialization & Configuration
# ==============================================================================

class TestInitialization:
    """Tests for layer initialization and parameter storage."""

    def test_defaults(self):
        layer = SingleWindowAttention(dim=16, window_size=4, num_heads=2)
        assert layer.dim == 16
        assert layer.window_size == 4
        assert layer.num_heads == 2
        assert layer.head_dim == 8
        assert layer.attention_mode == "linear"
        assert layer.probability_type == "softmax"
        expected_scale = 8 ** -0.5
        np.testing.assert_allclose(layer.scale, expected_scale)
        # QKV sub-layer created in __init__ for linear mode.
        assert layer.qkv is not None

    def test_kan_key_mode(self):
        layer = SingleWindowAttention(
            dim=16, window_size=3, num_heads=2, attention_mode="kan_key"
        )
        assert layer.attention_mode == "kan_key"
        assert layer.query is not None
        assert layer.key is not None
        assert layer.value is not None

    def test_custom_scale(self):
        layer = SingleWindowAttention(
            dim=16, window_size=4, num_heads=2, qk_scale=0.25
        )
        assert layer.scale == 0.25
        assert layer.qk_scale == 0.25


# ==============================================================================
# 2. Input Validation
# ==============================================================================

class TestValidation:
    """Tests for __init__ validation."""

    def test_invalid_attention_mode(self):
        with pytest.raises(ValueError, match="Invalid attention_mode"):
            SingleWindowAttention(
                dim=16, window_size=4, num_heads=2, attention_mode="bogus"
            )

    @pytest.mark.parametrize(
        "bad_prob",
        ["routing", "deterministic_routing", "hierarchical", "hierarchical_routing"],
    )
    def test_invalid_probability_type(self, bad_prob):
        with pytest.raises(ValueError, match="Invalid probability_type"):
            SingleWindowAttention(
                dim=16, window_size=4, num_heads=2, probability_type=bad_prob
            )


# ==============================================================================
# 3. Forward Pass (output shape == input shape)
# ==============================================================================

class TestForward:
    """Forward-pass tests at minimal configs."""

    def test_output_shape_full_window(self):
        """N == window_size**2 -> no internal padding."""
        window_size = 4
        n = window_size * window_size  # 16
        x = keras.random.normal((2, n, 16))
        layer = SingleWindowAttention(dim=16, window_size=window_size, num_heads=2)
        out = layer(x)
        assert out.shape == (2, n, 16)
        assert not np.any(np.isnan(np.array(out)))

    def test_output_shape_partial_window(self):
        """N < window_size**2 -> layer pads internally then strips it back off."""
        x = keras.random.normal((2, 5, 16))  # 5 < 16
        layer = SingleWindowAttention(dim=16, window_size=4, num_heads=2)
        out = layer(x)
        # Output must equal the ACTUAL (unpadded) input shape.
        assert out.shape == (2, 5, 16)
        assert not np.any(np.isnan(np.array(out)))

    def test_output_shape_no_relative_bias(self):
        x = keras.random.normal((2, 9, 16))
        layer = SingleWindowAttention(
            dim=16,
            window_size=3,
            num_heads=2,
            use_relative_position_bias=False,
        )
        out = layer(x)
        assert out.shape == (2, 9, 16)
        assert not np.any(np.isnan(np.array(out)))

    def test_kan_key_forward(self):
        x = keras.random.normal((2, 9, 16))
        layer = SingleWindowAttention(
            dim=16, window_size=3, num_heads=2, attention_mode="kan_key"
        )
        out = layer(x)
        assert out.shape == (2, 9, 16)
        assert not np.any(np.isnan(np.array(out)))

    def test_deterministic_inference(self):
        x = keras.random.normal((2, 9, 16))
        layer = SingleWindowAttention(dim=16, window_size=3, num_heads=2)
        out1 = layer(x, training=False)
        out2 = layer(x, training=False)
        np.testing.assert_allclose(np.array(out1), np.array(out2), atol=1e-6)


# ==============================================================================
# 4. Masked Forward (attention_mask is accepted AND honored)
# ==============================================================================

class TestMaskedForward:
    """The layer accepts a [B, N] attention_mask combined with internal padding."""

    def test_masked_forward_shape(self):
        x = keras.random.normal((2, 9, 16))
        # Mask out the last 3 tokens of every sample.
        mask = np.ones((2, 9), dtype="float32")
        mask[:, -3:] = 0.0
        mask = keras.ops.convert_to_tensor(mask)
        layer = SingleWindowAttention(dim=16, window_size=3, num_heads=2)
        out = layer(x, attention_mask=mask)
        assert out.shape == (2, 9, 16)
        assert not np.any(np.isnan(np.array(out)))

    def test_mask_changes_output(self):
        """A non-trivial mask must actually change the attended output vs no mask
        (proves the mask is honored, unlike SpatialAttention's ignored mask)."""
        x = keras.random.normal((2, 9, 16))
        layer = SingleWindowAttention(dim=16, window_size=3, num_heads=2)

        out_unmasked = np.array(layer(x, training=False))

        mask = np.ones((2, 9), dtype="float32")
        mask[:, -4:] = 0.0
        mask = keras.ops.convert_to_tensor(mask)
        out_masked = np.array(layer(x, attention_mask=mask, training=False))

        assert not np.allclose(out_unmasked, out_masked, atol=1e-4), (
            "attention_mask had no effect on the output"
        )


# ==============================================================================
# 5. Serialization
# ==============================================================================

class TestSerialization:

    def test_get_config(self):
        layer = SingleWindowAttention(
            dim=32,
            window_size=4,
            num_heads=4,
            attention_mode="linear",
            dropout_rate=0.1,
            use_relative_position_bias=False,
        )
        config = layer.get_config()
        assert config["dim"] == 32
        assert config["window_size"] == 4
        assert config["num_heads"] == 4
        assert config["attention_mode"] == "linear"
        assert config["dropout_rate"] == 0.1
        assert config["use_relative_position_bias"] is False

    def test_from_config(self):
        layer = SingleWindowAttention(dim=16, window_size=3, num_heads=2)
        config = layer.get_config()
        rebuilt = SingleWindowAttention.from_config(config)
        assert rebuilt.dim == 16
        assert rebuilt.window_size == 3
        assert rebuilt.num_heads == 2
        assert rebuilt.attention_mode == "linear"

    def test_model_save_load_loop(self):
        """Full .keras save/load round-trip; deterministic -> assert exact output."""
        inputs = keras.Input(shape=(9, 16))
        x = SingleWindowAttention(dim=16, window_size=3, num_heads=2)(inputs)
        model = keras.Model(inputs, x)

        x_in = np.random.normal(size=(2, 9, 16)).astype("float32")
        pred_orig = model.predict(x_in, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "single_window_attention.keras")
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
        layer = SingleWindowAttention(dim=16, window_size=4, num_heads=2)
        # Output shape is identical to the input shape.
        assert layer.compute_output_shape((2, 9, 16)) == (2, 9, 16)

    def test_kwargs_passthrough(self):
        layer = SingleWindowAttention(
            dim=16, window_size=4, num_heads=2, name="swa_special"
        )
        assert layer.name == "swa_special"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# ---------------------------------------------------------------------
# Mixed-precision mask tests (plan-2026-07-27-b4ef45f0, step 5)
# ---------------------------------------------------------------------
#
# WHAT IS BEING GUARDED HERE.
#
# `SingleWindowAttention.call` used the ARITHMETIC mask form
#
#     inf_value = ops.convert_to_tensor(-1e9, dtype=attn.dtype)
#     additive_mask = (1.0 - ops.cast(broadcast_mask, attn.dtype)) * inf_value
#     attn = attn + additive_mask
#
# which `common.MASK_BIAS_VALUE`'s own docstring rules out. Under
# `mixed_precision.set_global_policy('mixed_float16')` the literal `-1e9` is
# materialized in float16, where it is `-inf` (np.float16(-1e9) == -inf). At every
# UNMASKED position `(1.0 - mask) == 0`, so the product is `0 * -inf = NaN`.
#
# THIS SITE IS THE ONLY ONE IN THE PACKAGE WITH NO SAFE PATH AT ALL. The mask is
# not optional: `call()` always builds an internal padding mask and always applies
# the bias, so `attention_mask=None` goes down exactly the same line. MEASURED on
# unfixed HEAD (B=2, N=64, dim=64, window_size=8, num_heads=4), GPU 1 / TF 2.18,
# non-finite entries in the layer OUTPUT:
#
#     policy          mask=None   all-ones   right-pad   left-pad
#     float32            0/8192     0/8192      0/8192     0/8192
#     mixed_float16   8192/8192  8192/8192   8192/8192  8192/8192
#     float64            0/8192     0/8192      0/8192     0/8192
#
# i.e. this layer produced NOTHING BUT NaN under `mixed_float16`, with or without a
# caller-supplied mask.
#
# THE FIX is `common.apply_attention_mask`, which builds the bias with `ops.where`
# inside `common.mask_dtype(...)` (>= float32), so `0 * -inf` cannot be formed at
# all. The site keeps its own `ops.reshape` broadcast and its own `1 = keep`
# polarity spelling verbatim.
#
# THE `clip(attn, -30, 30)` MOVED (decisions.md D-010). It used to run AFTER the
# mask bias, which floored a masked logit at `-30` instead of `MASK_BIAS_VALUE` —
# a soft mask, not a hard one. MEASURED: with the relative-position-bias table
# driven to a uniform `-50` (so every logit sits below the clip floor), perturbing
# a MASKED token moved the kept query rows by 0.439 against a kept-token signal of
# 32.6; at `-20` the leak was 2.7e-04, and at the default (~0) it was exactly 0.0.
# The clip now runs on the RAW scores, before the bias, so it keeps doing the job
# its own comment claims (bounding the logits) while masking is exact in every
# regime. See `TestSingleWindowAttentionClipDoesNotFloorTheMask`.
#
# ANTI-VACUITY. The `N = 7`-hides-an-fp16-`-inf`-at-`N >= 512` trap does not
# transfer: this hazard is a per-ELEMENT dtype overflow of a constant multiplied by
# an exact zero, not a long reduction. It is nevertheless asserted reachable rather
# than assumed — see `TestSingleWindowAttentionMaskHazardIsReal`. N = 64 keys
# (window_size 8) is this layer's natural full-window size, not a toy one.
#
# A CAUSAL MASK IS NOT REPRESENTABLE AT THIS SITE and is deliberately absent from
# the mask table below: `call()` accepts a rank-2 `(B, N)` key mask only and
# reshapes it to `(B, 1, 1, N)`, so every query row sees the same keys. The
# left-padding mask takes its place as the third structurally-different mask.

from dl_techniques.layers.attention.common import MASK_BIAS_VALUE

_MP_B, _MP_N, _MP_DIM, _MP_H, _MP_WS = 2, 64, 64, 4, 8
_MP_KEEP = _MP_N // 2
_MP_SEED = 1234

# Absolute tolerance for "this policy's masked forward agrees with the float32 one".
#
# PRE-REGISTERED, not tuned after the fact: sized from this layer's own dtype error
# with the mask bias NEUTERED to 0.0 (the only forward that survives fp16 on unfixed
# HEAD, since the bias is applied unconditionally here), measured max |policy -
# float32| = 0.0081 for mixed_float16 and 0.0083 for float64 against an output
# absmax of 6.45. The entries below carry ~6x headroom on that. The float64 figure
# is the larger one because it is measured with TF32 matmuls ENABLED (the GPU
# default when this file runs alone); a TF32-disabled session only shrinks it, and
# every tolerance here is an UPPER bound, so both regimes are covered.
_MP_ATOL = {"float32": 1e-6, "mixed_float16": 0.05, "float64": 0.05}


def _mp_input():
    """Deterministic ``(B, N, dim)`` float32 input, shared by every test below."""
    return np.random.default_rng(7).standard_normal(
        (_MP_B, _MP_N, _MP_DIM)
    ).astype("float32")


def _mp_mask(kind):
    """One of the masks these tests need, as a float32 rank-2 ``1 = keep`` array.

    ``'all_ones'`` masks NOTHING and is the catastrophic case for the arithmetic
    form. ``'padding'`` masks the second half of the keys, ``'left_padding'`` the
    first half (the shape a left-padded generation batch produces), and
    ``'degenerate'`` blanks batch element 0 entirely — the fully-masked case for a
    site whose mask has no query axis.
    """
    if kind == "all_ones":
        return np.ones((_MP_B, _MP_N), dtype="float32")
    if kind == "padding":
        m = np.ones((_MP_B, _MP_N), dtype="float32")
        m[:, _MP_KEEP:] = 0.0
        return m
    if kind == "left_padding":
        m = np.ones((_MP_B, _MP_N), dtype="float32")
        m[:, :_MP_KEEP] = 0.0
        return m
    if kind == "degenerate":
        m = np.ones((_MP_B, _MP_N), dtype="float32")
        m[0, :] = 0.0
        return m
    raise ValueError(f"unknown mask kind {kind!r}")


def _mp_layer(**kwargs):
    """A built layer whose TRAINABLE weights are identical under every dtype policy.

    Seeding the initializers is NOT sufficient: a ``glorot_uniform`` draw under a
    ``float64`` policy differs from the same-seed draw under ``float32`` (the
    initializer samples in the VARIABLE dtype), so a cross-policy comparison on
    seeded-but-not-assigned weights measures the initializer, not the code under
    test. Explicit values are assigned instead.
    """
    layer = SingleWindowAttention(
        dim=_MP_DIM, window_size=_MP_WS, num_heads=_MP_H, **kwargs
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
    """Masked float32 output for ``kind``, memoized, under an explicit policy.

    This is the CONTROL every mixed-precision assertion compares against. It sets
    and restores the policy itself, so it is valid whichever parametrization of
    ``dtype_policy`` happens to reach it first.
    """
    if kind not in _F32_REFERENCE:
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("float32")
        try:
            layer = _mp_layer()
            _F32_REFERENCE[kind] = _mp_forward(
                layer, _mp_input(), None if kind == "none" else _mp_mask(kind)
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)
    return _F32_REFERENCE[kind]


class TestSingleWindowAttentionMaskHazardIsReal:
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
                "anti-vacuity FAILED: float16(MASK_BIAS_VALUE) is not -inf, so the "
                "`0 * -inf` hazard this module guards is not reproducible here."
            )
            assert np.isnan(np.float16(0.0) * bias), (
                "anti-vacuity FAILED: float16(0) * float16(MASK_BIAS_VALUE) is not "
                "NaN — the arithmetic mask form would be harmless."
            )
        assert np.isfinite(np.float32(MASK_BIAS_VALUE)), (
            "anti-vacuity FAILED: MASK_BIAS_VALUE is not finite in float32, so "
            "`mask_dtype(...)` would not be a fix."
        )

    def test_the_all_ones_mask_really_masks_nothing(self):
        assert int((_mp_mask("all_ones") == 0).sum()) == 0, (
            "the 'all_ones' mask masks something; it no longer reproduces the "
            "signature catastrophe (a vacuous mask destroying the batch)"
        )

    def test_the_partial_masks_really_mask_something(self):
        for kind in ("padding", "left_padding"):
            mask = _mp_mask(kind)
            assert int((mask == 0).sum()) > 0, (
                f"the {kind!r} mask masks nothing; it cannot detect a regression"
            )
            assert not (mask == 0).all(axis=-1).any(), (
                f"the {kind!r} mask blanks a whole batch element, so it no longer "
                "isolates the covered case from the degenerate probe"
            )

    def test_the_degenerate_mask_really_blanks_exactly_one_batch_element(self):
        mask = _mp_mask("degenerate")
        empty = (mask == 0).all(axis=-1)
        assert int(empty.sum()) == 1 and bool(empty[0])

    def test_the_mask_is_not_optional_at_this_site(self):
        """``attention_mask=None`` still goes through the bias line.

        This is what makes the site unique: ``call()`` always builds an internal
        padding mask, so there is no "no-mask path" to fall back on. Measured on
        unfixed HEAD: 8192/8192 NaN under `mixed_float16` with ``mask=None``.
        """
        import inspect
        source = inspect.getsource(SingleWindowAttention.call)
        assert "final_attention_mask = internal_padding_mask" in source, (
            "the internal padding mask is no longer applied unconditionally; the "
            "'mask=None is also broken' claim in these tests is stale"
        )

    def test_the_probability_sublayer_autocasts_a_float32_input(self, dtype_policy):
        """Why ``out_dtype`` cannot rescue a fully-masked row at this site."""
        layer = _mp_layer()
        prob = layer.attn_prob
        assert getattr(prob, "autocast", False) is True

        seen = {}
        original = prob.call

        def spy(x, *args, **kwargs):
            seen["dtype"] = keras.backend.standardize_dtype(x.dtype)
            return original(x, *args, **kwargs)

        prob.call = spy
        try:
            prob(keras.ops.convert_to_tensor(
                np.zeros((1, _MP_H, 4, 4), dtype="float32")
            ))
        finally:
            prob.call = original

        expected = keras.mixed_precision.global_policy().compute_dtype
        assert seen["dtype"] == expected, (
            f"a float32 tensor entering `attn_prob` was seen inside its call() as "
            f"{seen['dtype']!r}, not the compute dtype {expected!r}"
        )


class TestSingleWindowAttentionMixedPrecisionMask:
    """SC1 + SC2: finite AND agreeing with float32, for every legal mask.

    ``'none'`` is included because at this site it is NOT a bypass — see the module
    note above.
    """

    @pytest.mark.parametrize(
        "kind", ["none", "all_ones", "padding", "left_padding"]
    )
    def test_masked_forward_is_finite_and_matches_float32(self, dtype_policy, kind):
        layer = _mp_layer()
        out = _mp_forward(
            layer, _mp_input(), None if kind == "none" else _mp_mask(kind)
        )

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
        assert float(np.abs(out).max()) > 0.5 * float(np.abs(reference).max()), (
            f"output absmax {np.abs(out).max():.4g} collapsed relative to the "
            f"float32 control {np.abs(reference).max():.4g}"
        )


class TestSingleWindowAttentionFullyMaskedRow:
    """The degenerate case: a batch element that keeps NOTHING.

    This site's mask has no query axis (rank 2, reshaped to ``(B, 1, 1, N)``), so
    "a row that keeps nothing" means "a batch element whose whole key mask is
    zero". `common.apply_attention_mask`'s default ``rescue_axis=-1`` treats it as
    keeping EVERYTHING, so the all-`MASK_BIAS_VALUE` row is never FORMED.
    """

    def test_a_fully_masked_element_is_finite_and_matches_float32(self, dtype_policy):
        layer = _mp_layer()
        out = _mp_forward(layer, _mp_input(), _mp_mask("degenerate"))

        n_bad = int((~np.isfinite(out)).sum())
        assert n_bad == 0, (
            f"{n_bad}/{out.size} non-finite output entries for a mask that blanks a "
            f"whole batch element, under policy {dtype_policy!r}"
        )
        reference = _float32_reference("degenerate")
        atol = _MP_ATOL[dtype_policy]
        max_dev = float(np.abs(out - reference).max())
        assert max_dev <= atol, (
            f"the degenerate-masked forward under {dtype_policy!r} deviates from "
            f"the float32 control by {max_dev:.4g} > {atol:.4g}"
        )

    def test_the_rescued_element_behaves_as_if_it_kept_everything(self, dtype_policy):
        """The rescue's SEMANTICS, not merely its finiteness.

        The ``'degenerate'`` mask is all-ones except for batch element 0, which is
        blanked. "Keeps nothing => keeps everything" therefore makes it EQUIVALENT
        to the ``'all_ones'`` mask. That equivalence is the convention this package
        chose (decisions.md D-008 / D-009), so it is asserted directly.
        """
        degenerate = _mp_forward(_mp_layer(), _mp_input(), _mp_mask("degenerate"))
        all_ones = _mp_forward(_mp_layer(), _mp_input(), _mp_mask("all_ones"))

        atol = _MP_ATOL[dtype_policy]
        max_dev = float(np.abs(degenerate - all_ones).max())
        assert max_dev <= atol, (
            f"under {dtype_policy!r} the 'degenerate' mask does not behave like the "
            f"'all_ones' mask: max deviation {max_dev:.4g} > {atol:.4g}"
        )
        assert float(np.abs(all_ones).max()) > 0.0, (
            "anti-vacuity FAILED: the all-ones-masked output is identically zero"
        )


class TestSingleWindowAttentionMaskPolarity:
    """SC6: the mask must suppress the MASKED positions, not the kept ones.

    A polarity inversion here raises nothing, changes no shape and leaves the
    output perfectly finite; only an influence test can see it. The INVERTED-mask
    control is measured in the same call so the tolerance cannot be quietly widened
    into vacuity (`plans/LESSONS.md`: assert a RATIO where an exact zero is not
    available).
    """

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
        layer = _mp_layer()
        mask = _mp_mask("padding")
        delta_masked, delta_kept = self._influence(layer, mask)
        inverted, _ = self._influence(_mp_layer(), 1.0 - mask)

        assert delta_kept > 0.5, (
            f"perturbing a KEPT token changed the output by only {delta_kept:.6g}; "
            "the test is vacuous — the layer is ignoring its input"
        )
        assert inverted > 0.5, (
            f"the INVERTED-polarity control moved the output by only "
            f"{inverted:.6g}; it cannot detect an inversion, so the assertion "
            "below would be vacuous"
        )
        assert delta_masked <= 1e-3, (
            f"perturbing a MASKED token changed the kept query rows by "
            f"{delta_masked:.6g} under policy {dtype_policy!r} — this must be "
            f"exact, so the mask polarity is INVERTED (the inverted-polarity "
            f"control measures {inverted:.6g})"
        )


class TestSingleWindowAttentionClipDoesNotFloorTheMask:
    """D-010: ``clip(attn, -30, 30)`` must not turn a hard mask into a soft one.

    On unfixed HEAD the clip ran AFTER the mask bias, so a masked logit was floored
    at ``-30`` rather than ``MASK_BIAS_VALUE``. When every logit already sits below
    that floor the mask stops masking. MEASURED on HEAD in float32 with the
    relative-position-bias table driven to a uniform value:

        rel_pos_bias      0.0     -20.0     -50.0
        delta_masked      0.0   2.73e-04     0.439
        delta_kept       29.0      31.6      32.6

    So the defect is real and dtype-independent, but only visible once the logits
    are pushed under the clip floor — which is exactly why it survived unnoticed
    and why this test drives the bias table explicitly instead of hoping a random
    input reaches the regime.
    """

    @staticmethod
    def _leak(bias_value):
        layer = _mp_layer()
        layer.relative_position_bias_table.assign(keras.ops.cast(
            keras.ops.convert_to_tensor(np.full(
                tuple(layer.relative_position_bias_table.shape),
                bias_value, dtype="float32",
            )),
            layer.relative_position_bias_table.dtype,
        ))
        mask = _mp_mask("padding")
        base_input = _mp_input()
        perturbed = base_input.copy()
        perturbed[:, _MP_KEEP + 3, :] += 5.0
        rows = slice(0, _MP_KEEP)
        base = _mp_forward(layer, base_input, mask)
        return float(np.abs(
            _mp_forward(layer, perturbed, mask)[:, rows] - base[:, rows]
        ).max())

    def test_the_hazard_regime_is_reachable(self):
        """Anti-vacuity: the -50 bias really does push every logit under the clip."""
        assert self._leak(0.0) <= 1e-6, (
            "even at the default bias the mask leaks; the two regimes below are "
            "not distinguishable and this test proves nothing"
        )

    @pytest.mark.parametrize("bias_value", [-20.0, -50.0])
    def test_a_masked_token_stays_powerless_below_the_clip_floor(self, bias_value):
        leak = self._leak(bias_value)
        assert leak <= 1e-6, (
            f"with the relative-position bias driven to {bias_value}, perturbing a "
            f"MASKED token moved the kept rows by {leak:.6g}. The mask bias is "
            f"being floored by `clip(attn, -30, 30)` instead of suppressing the "
            f"position (measured 0.439 at -50 on unfixed HEAD)."
        )


# ---------------------------------------------------------------------
# Pairwise (rank-3) keep-mask contract (plan-2026-07-31-ddc92265, step 2)
# ---------------------------------------------------------------------
#
# WHAT IS BEING GUARDED HERE.
#
# Before this step `call()` accepted a rank-2 `(B, N)` KEY-ONLY predicate and
# reshaped it to `(B, 1, 1, N)`, so every query row saw the same keys. That
# contract is structurally incapable of expressing shifted-window self-attention
# (SW-MSA): after Swin's cyclic roll, one physical window holds tokens from up to
# four different pre-roll regions, and which keys a query may see depends on the
# QUERY's region. No key-only mask can say that.
#
# The layer now ALSO accepts a rank-3 `(B, N, N)` pairwise keep-mask
# (`mask[b, q, k] == 1` means "query q may attend to key k"), combined with the
# internal padding mask broadcast over the query axis and handed to
# `apply_attention_mask` as a `(B, 1, N, N)` predicate. The rank-2 branch is
# untouched (I2) and `attention_mask=None` is untouched (I1).
#
# RED BEFORE THE CHANGE (measured on ff4f2aa6, CPU, float32): every test in this
# section died at
#
#     ValueError: Cannot reshape a tensor with 512 elements to shape (2, 1, 1, 16)
#     (32 elements) for '{{node Reshape_3}} = Reshape[...]'
#
# raised from `reshape(final_attention_mask, (B, 1, 1, N))` — i.e. the rank-3 mask
# could not even reach the softmax, which is the point of A1 in the plan.

_PW_B, _PW_WS, _PW_DIM, _PW_HEADS = 2, 4, 16, 2
_PW_N = _PW_WS * _PW_WS          # 16 tokens: a FULL window, no internal padding
_PW_SEED = 20260731


def _pw_regions():
    """Two disjoint token regions, SW-MSA-shaped: rows 0-1 vs rows 2-3 of the 4x4 window.

    :return: ``(N,)`` int array of region ids in ``{0, 1}``.
    :rtype: np.ndarray
    """
    rows = np.arange(_PW_N) // _PW_WS
    return (rows >= 2).astype("int32")


def _pw_pairwise_mask():
    """``(B, N, N)`` block-diagonal keep-mask: a query may attend only within its region."""
    r = _pw_regions()
    keep = (r[:, None] == r[None, :]).astype("float32")
    return np.broadcast_to(keep, (_PW_B, _PW_N, _PW_N)).copy()


def _pw_input():
    return np.random.default_rng(_PW_SEED).standard_normal(
        (_PW_B, _PW_N, _PW_DIM)
    ).astype("float32")


def _pw_layer():
    """A built layer with deterministic weights, independent of construction order."""
    layer = SingleWindowAttention(
        dim=_PW_DIM, window_size=_PW_WS, num_heads=_PW_HEADS,
    )
    layer.build((_PW_B, _PW_N, _PW_DIM))
    rng = np.random.default_rng(_PW_SEED + 1)
    for w in layer.weights:
        w.assign(keras.ops.cast(
            keras.ops.convert_to_tensor(
                rng.standard_normal(tuple(w.shape)).astype("float32") * 0.2
            ),
            w.dtype,
        ))
    return layer


def _pw_forward(layer, array, mask):
    out = layer(
        keras.ops.convert_to_tensor(array),
        attention_mask=(
            None if mask is None else keras.ops.convert_to_tensor(mask)
        ),
        training=False,
    )
    return np.asarray(keras.ops.convert_to_numpy(out))


class TestSingleWindowAttentionPairwiseMask:
    """A rank-3 ``(B, N, N)`` keep-mask must make masked (query, key) pairs powerless.

    Polarity is the silent failure mode: an inverted pairwise mask raises nothing,
    changes no shape and stays finite — the layer simply attends to exactly the
    pairs it was told to forbid. Only a perturbation-isolation test can see it, so
    the inverted-mask and all-ones DEAD-COMPONENT controls are measured in the same
    test rather than assumed.
    """

    @staticmethod
    def _cross_region_influence(mask):
        """max |delta| at the region-0 query rows when a region-1 KEY is perturbed."""
        layer = _pw_layer()
        base_input = _pw_input()
        perturbed = base_input.copy()
        perturbed[:, 12, :] += 5.0        # token 12 is in region 1
        rows = _pw_regions() == 0
        base = _pw_forward(layer, base_input, mask)
        moved = _pw_forward(layer, perturbed, mask)
        return float(np.abs(moved[:, rows] - base[:, rows]).max())

    def test_a_region_1_key_cannot_influence_a_region_0_query(self):
        keep = _pw_pairwise_mask()
        leak = self._cross_region_influence(keep)

        # DEAD-COMPONENT probe, run live: with the mask made a no-op (all ones) the
        # very same perturbation MUST move the very same rows. A guard that survives
        # its own component being dead is worthless.
        dead = self._cross_region_influence(np.ones_like(keep))
        assert dead > 1e-2, (
            f"the all-ones (dead-mask) control moved the region-0 rows by only "
            f"{dead:.6g}; this test cannot detect a mask that does nothing and is "
            "therefore vacuous"
        )
        # POLARITY probe: an inverted pairwise mask must also move them.
        inverted = self._cross_region_influence(1.0 - keep)
        assert inverted > 1e-2, (
            f"the INVERTED-polarity control moved the region-0 rows by only "
            f"{inverted:.6g}; it cannot detect an inversion, so the assertion "
            "below would be vacuous"
        )

        assert leak == 0.0, (
            f"perturbing a region-1 token moved the region-0 query rows by "
            f"{leak:.6g}. A correct pairwise mask drives the forbidden key's "
            f"softmax weight to exactly 0.0, so this must be BIT-identical "
            f"(dead-mask control {dead:.6g}, inverted-polarity control "
            f"{inverted:.6g})."
        )

    def test_a_same_region_key_still_influences_its_query(self):
        """Anti-vacuity: the mask must not simply switch the layer off."""
        keep = _pw_pairwise_mask()
        layer = _pw_layer()
        base_input = _pw_input()
        perturbed = base_input.copy()
        perturbed[:, 3, :] += 5.0         # token 3 is in region 0
        rows = _pw_regions() == 0
        base = _pw_forward(layer, base_input, keep)
        moved = _pw_forward(layer, perturbed, keep)
        delta = float(np.abs(moved[:, rows] - base[:, rows]).max())
        assert delta > 1e-2, (
            f"perturbing a SAME-region token moved the region-0 rows by only "
            f"{delta:.6g}; the pairwise mask is suppressing everything, not just "
            "the cross-region pairs"
        )

    def test_a_query_broadcast_rank_3_mask_equals_the_rank_2_mask(self):
        """I2, expressed executably: the pairwise branch is a strict GENERALIZATION.

        A rank-3 mask that is constant over the query axis carries exactly the
        information of the rank-2 key mask it was built from, so the two branches
        must agree BIT-for-BIT. This is what pins the new branch as additive rather
        than as a second, subtly different mask implementation.
        """
        rng = np.random.default_rng(_PW_SEED + 2)
        key_mask = (rng.random((_PW_B, _PW_N)) > 0.4).astype("float32")
        key_mask[:, 0] = 1.0              # never fully mask a row
        rank3 = np.broadcast_to(
            key_mask[:, None, :], (_PW_B, _PW_N, _PW_N)
        ).copy()

        layer = _pw_layer()
        x = _pw_input()
        out2 = _pw_forward(layer, x, key_mask)
        out3 = _pw_forward(layer, x, rank3)
        np.testing.assert_array_equal(
            out2, out3,
            err_msg=(
                "a query-broadcast rank-3 mask disagrees with the equivalent "
                "rank-2 mask; the rank-3 branch is not a generalization of the "
                "rank-2 one"
            ),
        )

    def test_the_fully_masked_row_rescue_never_fires_on_an_sw_msa_mask(self):
        """The `apply_attention_mask` rescue must stay dormant for SW-MSA geometry.

        The rescue's convention is "a query row that keeps NOTHING keeps
        EVERYTHING" — silent un-masking if it ever fires on a real mask. In SW-MSA
        no row can keep nothing, because a token always shares a region with
        itself. That is asserted here two ways:

        1. structurally, on the predicate the layer will build (every row keeps
           >= 1 key, and there is no internal padding to zero a row out); and
        2. by its OBSERVABLE CONSEQUENCE — a mask whose rows genuinely keep
           nothing reproduces the ALL-ONES output exactly (the rescue firing),
           while the SW-MSA mask provably does not.
        """
        keep = _pw_pairwise_mask()
        assert keep.shape[1] == _PW_N == _PW_WS ** 2, (
            "this config has internal padding, so a padded query row would zero "
            "out and the rescue could fire for a reason unrelated to SW-MSA"
        )
        assert (keep.sum(axis=-1) > 0).all(), (
            "the SW-MSA mask has a query row that keeps nothing"
        )

        layer = _pw_layer()
        x = _pw_input()
        all_ones = _pw_forward(layer, x, np.ones_like(keep))
        dead_rows = _pw_forward(layer, x, np.zeros_like(keep))
        np.testing.assert_array_equal(
            dead_rows, all_ones,
            err_msg=(
                "an all-ZERO pairwise mask did not reproduce the all-ONES output, "
                "so the rescue is not observable this way and the assertion below "
                "proves nothing"
            ),
        )
        sw_msa = _pw_forward(layer, x, keep)
        assert np.abs(sw_msa - all_ones).max() > 1e-2, (
            "the SW-MSA-masked output is identical to the unmasked one — the "
            "fully-masked-row rescue fired (or the mask is being ignored)"
        )

    def test_the_pairwise_mask_survives_a_config_round_trip(self):
        """I3: the new branch adds no state, so `from_config` must reproduce it."""
        layer = _pw_layer()
        rebuilt = SingleWindowAttention.from_config(layer.get_config())
        rebuilt.build((_PW_B, _PW_N, _PW_DIM))
        for src, dst in zip(layer.weights, rebuilt.weights):
            dst.assign(src)
        keep = _pw_pairwise_mask()
        x = _pw_input()
        np.testing.assert_array_equal(
            _pw_forward(layer, x, keep), _pw_forward(rebuilt, x, keep),
            err_msg="the rebuilt layer disagrees with the original on the rank-3 path",
        )


class TestWindowAttentionPairwiseMaskPassThrough:
    """`WindowAttention._call_grid` must forward a rank-3 mask verbatim.

    A rank-3 mask is expressed in ALREADY-PARTITIONED window coordinates, so it is
    only meaningful when the internal grid is degenerate (one window per batch
    element, `N == window_size ** 2`) — which is exactly how `SwinTransformerBlock`
    calls this layer. Any other `N` is a caller error and must be LOUD, never a
    silently re-partitioned wrong-geometry mask.
    """

    @staticmethod
    def _layer(window_size=_PW_WS):
        from dl_techniques.layers.attention.window_attention import (
            WindowAttention,
        )
        return WindowAttention(
            dim=_PW_DIM, window_size=window_size, num_heads=_PW_HEADS,
            partition_mode="grid",
        )

    def test_a_rank_3_mask_reaches_the_inner_attention(self):
        layer = self._layer()
        x = keras.ops.convert_to_tensor(_pw_input())
        keep = _pw_pairwise_mask()
        base = np.asarray(keras.ops.convert_to_numpy(
            layer(x, attention_mask=keras.ops.convert_to_tensor(keep))
        ))
        unmasked = np.asarray(keras.ops.convert_to_numpy(
            layer(x, attention_mask=keras.ops.convert_to_tensor(
                np.ones_like(keep)
            ))
        ))
        assert base.shape == (_PW_B, _PW_N, _PW_DIM)
        assert np.isfinite(base).all()
        assert np.abs(base - unmasked).max() > 1e-4, (
            "the rank-3 mask made no difference — it is being dropped somewhere "
            "between `WindowAttention` and `SingleWindowAttention`"
        )

    def test_a_rank_3_mask_on_a_non_degenerate_grid_raises(self):
        layer = self._layer(window_size=2)      # N=16 -> a 4x4 grid of 2x2 windows
        x = keras.ops.convert_to_tensor(_pw_input())
        keep = keras.ops.convert_to_tensor(_pw_pairwise_mask())
        with pytest.raises(ValueError, match="rank-3"):
            layer(x, attention_mask=keep)

    def test_the_rank_3_path_survives_a_keras_round_trip_by_value(self):
        """I3/SC-8: a ``.keras`` round-trip on the NEW path, compared by VALUE.

        The module's pre-existing ``test_model_save_load`` round-trips
        ``WindowAttention`` on its DEFAULT (mask-free) config, which cannot
        observe the rank-3 branch at all — the same class of gap step 6 found
        for ``use_free_transformer`` (a round-trip test that exists but
        exercises the default configuration). This one feeds the pairwise mask
        as a second model input so the reloaded graph must reconstruct the
        rank-3 call path, not merely the weights.
        """
        from dl_techniques.layers.attention.window_attention import (
            WindowAttention,
        )

        x_in = keras.Input(shape=(_PW_N, _PW_DIM), name="tokens")
        m_in = keras.Input(shape=(_PW_N, _PW_N), name="pairwise_mask")
        out = WindowAttention(
            dim=_PW_DIM, window_size=_PW_WS, num_heads=_PW_HEADS,
            partition_mode="grid", name="pairwise_window_attention",
        )(x_in, attention_mask=m_in)
        model = keras.Model([x_in, m_in], out)

        x = _pw_input()
        keep = _pw_pairwise_mask()
        before = np.asarray(keras.ops.convert_to_numpy(
            model([x, keep], training=False)
        ))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "window_attention_pairwise.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            after = np.asarray(keras.ops.convert_to_numpy(
                reloaded([x, keep], training=False)
            ))
            assert isinstance(
                reloaded.get_layer("pairwise_window_attention"), WindowAttention
            )

        assert float(np.abs(before).max()) > 1e-4, (
            "round-trip compared all-zero values"
        )
        np.testing.assert_allclose(before, after, rtol=0, atol=1e-6)

        # Non-vacuity: the RELOADED model must still honour the mask, i.e. the
        # rank-3 branch survived deserialization rather than being ignored.
        after_all_ones = np.asarray(keras.ops.convert_to_numpy(
            reloaded([x, np.ones_like(keep)], training=False)
        ))
        assert float(np.abs(after_all_ones - after).max()) > 1e-4, (
            "the reloaded model ignores the pairwise mask"
        )
