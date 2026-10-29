"""Tests for ``MeanShiftAttention``: attention as mean shift on tied tokens.

The layer projects the tokens once and runs ``num_steps`` mean-shift steps per head.
Its defining identities are checked against references that do not share its code:
the softmax-attention form for the vMF kernel (plain NumPy), one step of the shared
``mean_shift_step`` op and the float64 oracle for the Gaussian kernel, and a NumPy
softmax of ``q.k / sqrt(d) - ||k||^2 / (2 sqrt(d))`` for the default bandwidth. Every
layer gets its kernels set explicitly after ``build``, so results are comparable
across dtype policies. The three identity checks are factored into functions so a RED
proof can inject a defect into the layer's ``kernel_logits`` and assert that the very
same body fails.

The last two classes cover the registration: the attention factory (strict kwargs,
registry defaults equal to the constructor's) and ``TransformerLayer`` (forward, padding
mask, ``.keras`` round trip).
"""

import inspect
import json
import math
import re
import warnings

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.attention import mean_shift_attention
from dl_techniques.layers.attention.factory import (
    ATTENTION_REGISTRY,
    STRICT_DROPPED_KEY_MARKER,
    create_attention_layer,
    get_attention_info,
    list_attention_types,
)
from dl_techniques.layers.attention.mean_shift_attention import (
    MeanShiftAttention,
    softplus_inverse,
)
from dl_techniques.layers.attention.multi_head_attention import MultiHeadAttention
from dl_techniques.layers.mixtures import mean_shift_ops as ops
from dl_techniques.layers.transformers.transformer import TransformerLayer
from tests.test_layers.test_mixtures.mean_shift_oracle import oracle_step

pytestmark = pytest.mark.usefixtures("tf32_disabled")

DIM = 16
HEADS = 2
HEAD_DIM = DIM // HEADS


def _np(tensor):
    return keras.ops.convert_to_numpy(tensor)


def _make(**kwargs):
    kwargs.setdefault("dim", DIM)
    kwargs.setdefault("num_heads", HEADS)
    layer = MeanShiftAttention(**kwargs)
    layer.build((None, None, kwargs["dim"]))
    return layer


def _set_kernels(layer, seed=0, scale=0.25):
    """Give both projections explicit random kernels (comparable across policies)."""
    rng = np.random.default_rng(seed)
    for sub in (layer.projection, layer.output_projection):
        sub.kernel.assign(rng.normal(size=sub.kernel.shape).astype("float32") * scale)
    return layer


def _set_identity_kernels(layer):
    eye = np.eye(DIM, dtype="float32")
    layer.projection.kernel.assign(eye)
    layer.output_projection.kernel.assign(eye)
    return layer


def _tokens(batch, count, seed, scale=1.0, dim=DIM):
    rng = np.random.default_rng(seed)
    return (scale * rng.normal(size=(batch, count, dim))).astype("float32")


def _kernel(sub):
    return _np(sub.kernel).astype("float64")


def _heads(layer, x):
    """Projected tokens as float64 ``(B, H, N, d)`` computed in NumPy."""
    proj = x.astype("float64") @ _kernel(layer.projection)
    batch, count = proj.shape[:2]
    return proj.reshape(batch, count, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)


def _merge(z):
    batch, heads, count, width = z.shape
    return z.transpose(0, 2, 1, 3).reshape(batch, count, heads * width)


def _softmax_np(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    expd = np.exp(shifted)
    return expd / expd.sum(axis=-1, keepdims=True)


def _unit_np(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _set_bandwidth(layer, value):
    """Set the effective bandwidth of a global or per-head layer to ``value``."""
    raw = softplus_inverse(value - layer.min_bandwidth)
    layer.raw_bandwidth.assign(np.full(layer.raw_bandwidth.shape, raw, "float32"))


def _copy_weights(src, dst):
    """Copy every weight of ``dst`` that ``src`` also has, matched by path suffix."""
    def key(layer, variable):
        return variable.path.split(f"{layer.name}/")[-1]

    values = {key(src, v): _np(v) for v in src.weights}
    for variable in dst.weights:
        name = key(dst, variable)
        if name in values:
            variable.assign(values[name])


# ---------------------------------------------------------------------
# Identity checks, factored so the RED proofs can run the same bodies
# ---------------------------------------------------------------------

def _check_vmf_matches_softmax_attention():
    h = 0.9
    layer = _set_identity_kernels(_make(kernel="vmf", bandwidth_init=h))
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 6, HEADS, HEAD_DIM))
    x = _unit_np(x).reshape(2, 6, DIM).astype("float32")
    out, weights = layer(x, return_attention_scores=True)

    q = x.astype("float64").reshape(2, 6, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    ref_weights = _softmax_np(q @ q.transpose(0, 1, 3, 2) / h ** 2)
    ref = _merge(_unit_np(ref_weights @ q))
    np.testing.assert_allclose(_np(weights), ref_weights, atol=1e-5, rtol=0)
    np.testing.assert_allclose(_np(out), ref, atol=1e-5, rtol=0)


def _check_gaussian_matches_mean_shift_step():
    h = 1.3
    layer = _set_kernels(_make(kernel="gaussian", bandwidth_init=h))
    x = _tokens(2, 6, 4)
    out = _np(layer(x))

    q = _heads(layer, x).astype("float32")
    shifted, _ = ops.mean_shift_step(q, q, q, h)
    ref = _merge(_np(shifted).astype("float64")) @ _kernel(layer.output_projection)
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=0)

    shifted64, _ = oracle_step(q.astype("float64"), q.astype("float64"),
                               q.astype("float64"), h)
    ref64 = _merge(shifted64) @ _kernel(layer.output_projection)
    np.testing.assert_allclose(out, ref64, atol=1e-5, rtol=0)


def _check_default_bandwidth_is_scaled_dot_product():
    layer = _set_kernels(_make())
    assert layer.bandwidth_init == pytest.approx(HEAD_DIM ** 0.25)
    x = _tokens(2, 6, 5)
    _, weights = layer(x, return_attention_scores=True)

    q = _heads(layer, x)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    key_sq = np.sum(q * q, axis=-1)[:, :, None, :]
    ref = _softmax_np(q @ q.transpose(0, 1, 3, 2) * scale - 0.5 * key_sq * scale)
    np.testing.assert_allclose(_np(weights), ref, atol=1e-5, rtol=0)

    # With equal-norm tokens the key-norm bias is constant: plain scaled dot product.
    layer = _set_identity_kernels(_make())
    rng = np.random.default_rng(6)
    equal = 2.0 * _unit_np(rng.normal(size=(2, 6, HEADS, HEAD_DIM)))
    equal = equal.reshape(2, 6, DIM).astype("float32")
    _, weights = layer(equal, return_attention_scores=True)
    qe = equal.astype("float64").reshape(2, 6, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    ref = _softmax_np(qe @ qe.transpose(0, 1, 3, 2) * scale)
    np.testing.assert_allclose(_np(weights), ref, atol=1e-5, rtol=0)


CHECKS = {
    "vmf": _check_vmf_matches_softmax_attention,
    "step": _check_gaussian_matches_mean_shift_step,
    "default": _check_default_bandwidth_is_scaled_dot_product,
}


def _patched_logits(defect):
    real = mean_shift_attention.kernel_logits

    def patched(q, k, bandwidth, kernel="gaussian", drop_query_norm=True, key_mask=None):
        if defect == "temperature":
            return real(q, k, bandwidth * 1.05, kernel, drop_query_norm, key_mask)
        base = real(q, k, bandwidth, kernel, drop_query_norm, key_mask)
        if kernel == "gaussian":
            k_sq = keras.ops.expand_dims(
                keras.ops.sum(keras.ops.square(keras.ops.cast(k, base.dtype)), axis=-1),
                axis=-2,
            )
            return base + k_sq / (bandwidth * bandwidth)
        return base

    return patched


RED_CASES = [
    ("vmf", "temperature"),
    ("step", "bias_sign"),
    ("step", "temperature"),
    ("default", "bias_sign"),
    ("default", "temperature"),
]


class TestIdentities:
    @pytest.mark.parametrize("check", sorted(CHECKS))
    def test_the_identity_holds(self, check):
        CHECKS[check]()

    def test_the_red_cases_cover_every_check(self):
        assert {check for check, _ in RED_CASES} == set(CHECKS)

    @pytest.mark.parametrize("check,defect", RED_CASES)
    def test_the_identity_fails_on_an_injected_defect(self, check, defect, monkeypatch):
        monkeypatch.setattr(mean_shift_attention, "kernel_logits", _patched_logits(defect))
        with pytest.raises(AssertionError):
            CHECKS[check]()

    def test_a_fixed_step_size_damps_the_shift(self):
        h, eta = 1.3, 0.5
        layer = _set_kernels(_make(bandwidth_init=h, step_size=eta))
        x = _tokens(2, 6, 7)
        out = _np(layer(x))
        q = _heads(layer, x).astype("float32")
        shifted, _ = ops.mean_shift_step(q, q, q, h)
        z = (1 - eta) * q.astype("float64") + eta * _np(shifted).astype("float64")
        ref = _merge(z) @ _kernel(layer.output_projection)
        np.testing.assert_allclose(out, ref, atol=1e-5, rtol=0)

        full = _set_kernels(_make(bandwidth_init=h, step_size=1.0))
        assert np.abs(_np(full(x)) - out).max() > 1e-3


# ---------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------

def _check_steps_iterate(two_step, one_step, x):
    assert np.abs(_np(two_step(x)) - _np(one_step(x))).max() > 1e-3


class TestIteration:
    @pytest.mark.parametrize("blurring", [True, False])
    def test_a_second_step_changes_the_output(self, blurring):
        x = _tokens(2, 6, 8)
        one = _set_kernels(_make(num_steps=1, blurring=blurring))
        two = _set_kernels(_make(num_steps=2, blurring=blurring))
        _check_steps_iterate(two, one, x)

    def test_the_liveness_check_fails_when_the_steps_do_not_iterate(self):
        x = _tokens(2, 6, 8)
        one = _set_kernels(_make(num_steps=1))
        stuck = _set_kernels(_make(num_steps=2))
        stuck.num_steps = 1
        with pytest.raises(AssertionError):
            _check_steps_iterate(stuck, one, x)

    def test_blurring_and_non_blurring_agree_at_one_step(self):
        x = _tokens(2, 6, 9)
        blur = _set_kernels(_make(num_steps=1, blurring=True))
        fixed = _set_kernels(_make(num_steps=1, blurring=False))
        np.testing.assert_allclose(_np(blur(x)), _np(fixed(x)), atol=1e-6, rtol=0)

    def test_blurring_and_non_blurring_differ_from_two_steps(self):
        x = _tokens(2, 6, 9)
        blur = _set_kernels(_make(num_steps=2, blurring=True))
        fixed = _set_kernels(_make(num_steps=2, blurring=False))
        assert np.abs(_np(blur(x)) - _np(fixed(x))).max() > 1e-3

    @pytest.mark.parametrize("blurring", [True, False])
    def test_non_blurring_keys_stay_at_the_projected_tokens(self, blurring):
        h = 1.3
        x = _tokens(2, 5, 10)
        layer = _set_kernels(_make(num_steps=2, blurring=blurring, bandwidth_init=h))
        out = _np(layer(x))
        q = _heads(layer, x)
        z = q
        for _ in range(2):
            keys = z if blurring else q
            z, _ = oracle_step(z, keys, keys, h)
        ref = _merge(z) @ _kernel(layer.output_projection)
        np.testing.assert_allclose(out, ref, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------
# Symmetry and masks
# ---------------------------------------------------------------------

class TestMasks:
    @pytest.mark.parametrize("blurring", [True, False])
    def test_permuting_the_tokens_permutes_the_output(self, blurring):
        x = _tokens(2, 6, 11)
        perm = np.random.default_rng(1).permutation(6)
        layer = _set_kernels(_make(num_steps=2, blurring=blurring))
        out = _np(layer(x))
        permuted = _np(layer(x[:, perm]))
        np.testing.assert_allclose(permuted, out[:, perm], atol=1e-5, rtol=0)

    @pytest.mark.parametrize("blurring", [True, False])
    def test_padded_keys_do_not_reach_the_unpadded_queries(self, blurring):
        mask = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype="float32")
        x = _tokens(2, 5, 12)
        other = x.copy()
        other[0, 3:] = _tokens(1, 2, 99, scale=5.0)[0]
        layer = _set_kernels(_make(num_steps=3, blurring=blurring))
        out, weights = layer(x, attention_mask=mask, return_attention_scores=True)
        out_other = layer(other, attention_mask=mask)
        np.testing.assert_allclose(
            _np(out)[0, :3], _np(out_other)[0, :3], atol=1e-6, rtol=0
        )
        np.testing.assert_array_equal(_np(out)[1], _np(out_other)[1])
        assert np.all(_np(weights)[0, :, :, 3:] == 0.0)

    @pytest.mark.parametrize("blurring", [True, False])
    def test_a_masked_run_equals_the_run_on_the_kept_tokens(self, blurring):
        mask = np.array([[1, 1, 1, 0, 0]], dtype="float32")
        x = _tokens(1, 5, 13)
        layer = _set_kernels(_make(num_steps=3, blurring=blurring))
        masked = _np(layer(x, attention_mask=mask))[0, :3]
        short = _np(layer(x[:, :3]))[0]
        np.testing.assert_allclose(masked, short, atol=1e-5, rtol=0)

    def test_the_mask_ranks_agree(self):
        layer = _set_kernels(_make(num_steps=2))
        x = _tokens(2, 5, 14)
        rank2 = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype="float32")
        rank3 = np.broadcast_to(rank2[:, None, :], (2, 5, 5)).copy()
        rank4 = rank3[:, None]
        base = _np(layer(x, attention_mask=rank2))
        np.testing.assert_allclose(_np(layer(x, attention_mask=rank3)), base, atol=1e-6)
        np.testing.assert_allclose(_np(layer(x, attention_mask=rank4)), base, atol=1e-6)
        np.testing.assert_allclose(
            _np(layer(x, attention_mask=rank2.astype(bool))), base, atol=1e-6
        )
        with pytest.raises(ValueError, match="rank 2, 3 or 4"):
            layer(x, attention_mask=np.ones((2,), dtype="float32"))

    def test_a_fully_masked_key_set_gives_zeros_and_stays_finite(self):
        mask = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype="float32")
        x = _tokens(2, 5, 15)
        layer = _set_kernels(_make(num_steps=2))
        out, weights = layer(x, attention_mask=mask, return_attention_scores=True)
        out, weights = _np(out), _np(weights)
        assert np.isfinite(out).all() and np.isfinite(weights).all()
        assert np.all(out[1] == 0.0) and np.all(weights[1] == 0.0)
        alone = _np(layer(x[:1]))
        np.testing.assert_allclose(out[:1], alone, atol=1e-6, rtol=0)

    def test_a_query_row_with_no_kept_key_is_zeroed(self):
        mask = np.ones((2, 5, 5), dtype="float32")
        mask[0, 2, :] = 0.0
        x = _tokens(2, 5, 16)
        layer = _set_kernels(_make(num_steps=1))
        out, weights = layer(x, attention_mask=mask, return_attention_scores=True)
        free = _np(layer(x))
        out, weights = _np(out), _np(weights)
        assert np.isfinite(out).all()
        assert np.all(out[0, 2] == 0.0) and np.all(weights[0, :, 2, :] == 0.0)
        keep = np.ones((2, 5), dtype=bool)
        keep[0, 2] = False
        np.testing.assert_allclose(out[keep], free[keep], atol=1e-6, rtol=0)

    def test_a_padded_query_under_a_key_padding_mask_is_not_zeroed(self):
        mask = np.array([[1, 1, 1, 0, 0]], dtype="float32")
        layer = _set_kernels(_make())
        out = _np(layer(_tokens(1, 5, 17), attention_mask=mask))
        assert np.abs(out[0, 3:]).max() > 1e-3

    def test_the_returned_weights_are_rows_of_a_distribution(self):
        mask = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype="float32")
        layer = _set_kernels(_make(num_steps=2))
        out, weights = layer(_tokens(2, 5, 18), attention_mask=mask,
                             return_attention_scores=True)
        weights = _np(weights)
        assert weights.shape == (2, HEADS, 5, 5)
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-5)
        assert tuple(out.shape) == (2, 5, DIM)

    def test_a_single_token_runs_without_a_warning(self):
        layer = _set_kernels(_make(num_steps=2))
        x = _tokens(2, 1, 19)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = _np(layer(x))
        assert np.isfinite(out).all()


# ---------------------------------------------------------------------
# A forbidden key never reaches another row, at any step
# ---------------------------------------------------------------------

LEAK_N = 6
FORBIDDEN_KEY = 5
NO_KEPT_KEY_ROW = 0


def _pairwise_mask(rank, forbid_row_key=False):
    """Pairwise keep mask: key 5 forbidden to every query, query 0 keeps nothing.

    With ``forbid_row_key`` the key 0 is forbidden to every query as well.
    """
    mask = np.ones((1, LEAK_N, LEAK_N), dtype="float32")
    mask[0, :, FORBIDDEN_KEY] = 0.0
    mask[0, NO_KEPT_KEY_ROW, :] = 0.0
    if forbid_row_key:
        mask[0, :, NO_KEPT_KEY_ROW] = 0.0
    return mask if rank == 3 else mask[:, None]


def _check_a_forbidden_key_does_not_leak(rank, num_steps, blurring, kernel):
    layer = _set_kernels(_make(
        num_steps=num_steps, blurring=blurring, kernel=kernel,
        bandwidth_init=1.0 if kernel == "gaussian" else 0.5,
    ))
    x = _tokens(1, LEAK_N, 71)
    shifted = x.copy()
    shifted[0, FORBIDDEN_KEY] += 10.0
    mask = _pairwise_mask(rank)
    out = _np(layer(x, attention_mask=mask))
    out_shifted = _np(layer(shifted, attention_mask=mask))
    others = [i for i in range(LEAK_N) if i not in (NO_KEPT_KEY_ROW, FORBIDDEN_KEY)]
    np.testing.assert_allclose(out_shifted[0, others], out[0, others], atol=1e-5, rtol=0)
    assert np.all(out[0, NO_KEPT_KEY_ROW] == 0.0)


LEAK_CASES = [
    (rank, steps, blurring, kernel)
    for rank in (3, 4)
    for steps in (1, 2, 3)
    for blurring in (True, False)
    for kernel in ("gaussian", "vmf")
]


class TestAForbiddenKeyDoesNotLeak:
    @pytest.mark.parametrize("rank,num_steps,blurring,kernel", LEAK_CASES)
    def test_the_other_rows_do_not_read_the_forbidden_key(
            self, rank, num_steps, blurring, kernel
    ):
        _check_a_forbidden_key_does_not_leak(rank, num_steps, blurring, kernel)

    @pytest.mark.parametrize("kernel", ["gaussian", "vmf"])
    def test_the_check_fails_when_a_row_with_no_kept_key_moves(self, kernel, monkeypatch):
        def every_row_lives(keep):
            return keras.ops.logical_or(keras.ops.any(keep, axis=-1, keepdims=True), True)

        monkeypatch.setattr(MeanShiftAttention, "_rows_with_a_kept_key",
                            staticmethod(every_row_lives))
        with pytest.raises(AssertionError):
            _check_a_forbidden_key_does_not_leak(3, 2, True, kernel)

    @pytest.mark.parametrize("blurring", [True, False])
    @pytest.mark.parametrize("num_steps", [1, 3])
    def test_the_other_rows_equal_the_run_without_the_dead_row_and_the_forbidden_key(
            self, blurring, num_steps
    ):
        layer = _set_kernels(_make(num_steps=num_steps, blurring=blurring))
        x = _tokens(1, LEAK_N, 72)
        mask = _pairwise_mask(3, forbid_row_key=True)
        out = _np(layer(x, attention_mask=mask))
        inner = list(range(1, FORBIDDEN_KEY))
        alone = _np(layer(x[:, inner]))
        np.testing.assert_allclose(out[0, inner], alone[0], atol=1e-5, rtol=0)

    @pytest.mark.parametrize("num_steps", [1, 2, 3])
    def test_a_row_with_no_kept_key_has_zero_output_and_zero_weights(self, num_steps):
        layer = _set_kernels(_make(num_steps=num_steps))
        out, weights = layer(_tokens(1, LEAK_N, 73), attention_mask=_pairwise_mask(3),
                             return_attention_scores=True)
        assert np.all(_np(out)[0, NO_KEPT_KEY_ROW] == 0.0)
        assert np.all(_np(weights)[0, :, NO_KEPT_KEY_ROW, :] == 0.0)
        assert np.all(_np(weights)[0, :, :, FORBIDDEN_KEY] == 0.0)

    def test_a_row_with_no_kept_key_receives_no_gradient_and_nothing_goes_nan(self):
        layer = _set_kernels(_make(num_steps=3))
        x = tf.constant(_tokens(1, LEAK_N, 74))
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = tf.reduce_sum(layer(x, attention_mask=_pairwise_mask(3)) ** 2)
        grad = _np(tape.gradient(loss, x))
        assert np.isfinite(grad).all()


# ---------------------------------------------------------------------
# Padded keys do not set the Gaussian centre
# ---------------------------------------------------------------------

class TestPaddedKeysDoNotSetTheCentre:
    @pytest.mark.parametrize("blurring", [True, False])
    def test_junk_in_the_padded_keys_changes_no_unpadded_output(self, blurring):
        mask = np.array([[1, 1, 1, 1, 0, 0]], dtype="float32")
        clean = _tokens(1, 6, 75)
        clean[0, 4:] = 0.0
        junk = clean.copy()
        junk[0, 4:] = 1e4
        layer = _set_kernels(_make(num_steps=2, blurring=blurring, bandwidth_init=1.0))
        out_clean = _np(layer(clean, attention_mask=mask))
        out_junk = _np(layer(junk, attention_mask=mask))
        assert np.isfinite(out_junk).all()
        np.testing.assert_allclose(out_junk[0, :4], out_clean[0, :4], atol=1e-5, rtol=0)

    def test_the_check_fails_when_the_centre_ignores_the_padding(self, monkeypatch):
        real = mean_shift_attention.kernel_logits

        def plain_mean(q, k, bandwidth, kernel="gaussian", drop_query_norm=True,
                       key_mask=None):
            return real(q, k, bandwidth, kernel, drop_query_norm)

        monkeypatch.setattr(mean_shift_attention, "kernel_logits", plain_mean)
        mask = np.array([[1, 1, 1, 1, 0, 0]], dtype="float32")
        clean = _tokens(1, 6, 75)
        clean[0, 4:] = 0.0
        junk = clean.copy()
        junk[0, 4:] = 1e4
        layer = _set_kernels(_make(num_steps=1, blurring=False, bandwidth_init=1.0))
        gap = np.abs(_np(layer(junk, attention_mask=mask))[0, :4]
                     - _np(layer(clean, attention_mask=mask))[0, :4]).max()
        assert gap > 1e-4


# ---------------------------------------------------------------------
# A masked key has exactly zero weight at any bandwidth
# ---------------------------------------------------------------------

SPREAD_N = 6
SPREAD_SCALE = 500.0
BANDWIDTH_INITS = (0.011, 0.05, None)


def _spread_mask(rank):
    """Keep mask over six widely spread tokens, ``(2, ...)`` batch.

    Rank 2 pads key 5 of batch 0. Rank 3 and 4 forbid every key's own position, so
    the nearest key of each query is the forbidden one; batch 1 (rank 3) and head 1
    (rank 4) also leave one query with no kept key.
    """
    if rank == 2:
        mask = np.ones((2, SPREAD_N), dtype="float32")
        mask[0, 5] = 0.0
        return mask
    diagonal = 1.0 - np.eye(SPREAD_N, dtype="float32")
    if rank == 3:
        mask = np.broadcast_to(diagonal, (2, SPREAD_N, SPREAD_N)).copy()
        mask[1, 3, :] = 0.0
        return mask
    mask = np.broadcast_to(diagonal, (2, HEADS, SPREAD_N, SPREAD_N)).copy()
    mask[:, 1, 2, :] = 0.0
    mask[:, 1, :, 5] = 0.0
    return mask


def _keep_rank_four(mask):
    keep = mask > 0
    if keep.ndim == 2:
        return keep[:, None, None, :]
    return keep[:, None] if keep.ndim == 3 else keep


def _check_a_masked_key_has_exactly_zero_weight(
        rank, kernel, num_steps, blurring, bandwidth_init
):
    layer = _set_identity_kernels(_make(
        num_steps=num_steps, blurring=blurring, kernel=kernel,
        bandwidth_init=bandwidth_init,
    ))
    mask = _spread_mask(rank)
    out, weights = layer(_tokens(2, SPREAD_N, 81, scale=SPREAD_SCALE),
                         attention_mask=mask, return_attention_scores=True)
    out, weights = _np(out), _np(weights)
    keep = np.broadcast_to(_keep_rank_four(mask), weights.shape)
    assert np.isfinite(out).all()
    assert np.all(weights[~keep] == 0.0)
    live = keep.any(axis=-1)
    sums = weights.sum(axis=-1)
    np.testing.assert_allclose(sums[live], 1.0, atol=1e-5, rtol=0)
    assert np.all(sums[~live] == 0.0)


def _check_the_cross_form_padded_key_has_zero_weight():
    layer = MeanShiftAttention(
        dim=4, num_heads=1, num_steps=1, blurring=False, bandwidth_mode="global",
        bandwidth_init=0.011, use_bias=False,
    )
    layer.build([(None, None, 4), (None, None, 4)])
    layer.projection.kernel.assign(np.eye(4, dtype="float32"))
    query = np.zeros((1, 1, 4), dtype="float32")
    keys = np.zeros((1, 3, 4), dtype="float32")
    keys[0, 0, 0], keys[0, 1, 0] = 1000.0, 1001.0
    _, weights = layer([query, keys], attention_mask=np.array([[1, 1, 0]], "float32"),
                       return_attention_scores=True)
    weights = _np(weights).ravel()
    assert weights[2] == 0.0
    np.testing.assert_allclose(weights[:2].sum(), 1.0, atol=1e-6, rtol=0)


def _check_a_forbidden_diagonal_has_zero_weight():
    layer = MeanShiftAttention(
        dim=4, num_heads=1, num_steps=1, blurring=True, bandwidth_mode="global",
        bandwidth_init=0.011, use_bias=False,
    )
    layer.build((None, None, 4))
    layer.projection.kernel.assign(np.eye(4, dtype="float32"))
    tokens = np.zeros((1, 3, 4), dtype="float32")
    tokens[0, 1, 0], tokens[0, 2, 0] = 500.0, -500.0
    mask = (1.0 - np.eye(3, dtype="float32"))[None]
    _, weights = layer(tokens, attention_mask=mask, return_attention_scores=True)
    weights = _np(weights)[0, 0]
    assert np.all(np.diagonal(weights) == 0.0)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-6, rtol=0)


def _without_the_shift(monkeypatch):
    monkeypatch.setattr(MeanShiftAttention, "_shift_to_kept_maximum",
                        staticmethod(lambda logits, keep: logits))


GRID_CASES = [
    (rank, kernel, steps, blurring, bandwidth)
    for rank in (2, 3, 4)
    for kernel in ("gaussian", "vmf")
    for steps in (1, 2, 3)
    for blurring in (True, False)
    for bandwidth in BANDWIDTH_INITS
]


class TestAMaskedKeyHasZeroWeightAtAnyBandwidth:
    @pytest.mark.parametrize("rank,kernel,num_steps,blurring,bandwidth_init", GRID_CASES)
    def test_masked_weights_are_exactly_zero_and_kept_weights_sum_to_one(
            self, rank, kernel, num_steps, blurring, bandwidth_init
    ):
        _check_a_masked_key_has_exactly_zero_weight(
            rank, kernel, num_steps, blurring, bandwidth_init
        )

    def test_a_padded_key_at_the_query_gets_no_weight_in_the_cross_form(self):
        _check_the_cross_form_padded_key_has_zero_weight()

    def test_a_forbidden_diagonal_gets_no_weight_between_distant_tokens(self):
        _check_a_forbidden_diagonal_has_zero_weight()

    @pytest.mark.parametrize("rank", [2, 3, 4])
    def test_the_check_fails_without_the_shift(self, rank, monkeypatch):
        _without_the_shift(monkeypatch)
        with pytest.raises(AssertionError):
            _check_a_masked_key_has_exactly_zero_weight(rank, "gaussian", 1, False, 0.011)

    def test_the_review_cases_fail_without_the_shift(self, monkeypatch):
        _without_the_shift(monkeypatch)
        with pytest.raises(AssertionError):
            _check_the_cross_form_padded_key_has_zero_weight()
        with pytest.raises(AssertionError):
            _check_a_forbidden_diagonal_has_zero_weight()

    @pytest.mark.parametrize("rank", [2, 3, 4])
    @pytest.mark.parametrize("bandwidth_init", [0.011, None])
    def test_every_gradient_is_finite_with_a_learnable_bandwidth(self, rank, bandwidth_init):
        layer = _set_kernels(_make(num_steps=2, bandwidth_init=bandwidth_init,
                                   learnable_step_size=True))
        x = tf.constant(_tokens(2, SPREAD_N, 82, scale=SPREAD_SCALE))
        mask = _spread_mask(rank)
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = tf.reduce_sum(layer(x, attention_mask=mask) ** 2)
        grads = tape.gradient(loss, [x] + layer.trainable_variables)
        assert all(grad is not None and np.isfinite(_np(grad)).all() for grad in grads)

    def test_mixed_float16_keeps_masked_weights_at_zero_and_the_output_finite(self):
        mask = _spread_mask(3)
        x = _tokens(2, SPREAD_N, 83, scale=SPREAD_SCALE)
        layer, _ = _run("mixed_float16", x, mask, num_steps=2, bandwidth_init=0.011)
        out, weights = layer(x, attention_mask=mask, return_attention_scores=True)
        weights = _np(weights).astype("float32")
        keep = np.broadcast_to(_keep_rank_four(mask), weights.shape)
        assert np.isfinite(_np(out).astype("float32")).all()
        assert np.all(weights[~keep] == 0.0)
        live = keep.any(axis=-1)
        np.testing.assert_allclose(weights.sum(axis=-1)[live], 1.0, atol=2e-3, rtol=0)


# ---------------------------------------------------------------------
# A key that no row keeps does not set the Gaussian centre
# ---------------------------------------------------------------------

def _forbidden_to_every_row_mask(rank):
    """Key 5 of six is forbidden to every query, in the given mask rank."""
    if rank == 2:
        mask = np.ones((1, 6), dtype="float32")
        mask[0, 5] = 0.0
        return mask
    mask = np.ones((1, 6, 6), dtype="float32")
    mask[0, :, 5] = 0.0
    return mask if rank == 3 else mask[:, None]


def _check_a_key_no_row_keeps_does_not_set_the_centre(rank):
    layer = _set_kernels(_make(num_steps=2))
    clean = _tokens(1, 6, 85)
    junk = clean.copy()
    junk[0, 5] = 1e4
    mask = _forbidden_to_every_row_mask(rank)
    out_clean = _np(layer(clean, attention_mask=mask))
    out_junk = _np(layer(junk, attention_mask=mask))
    gap = np.abs(out_junk[0, :5] - out_clean[0, :5]).max()
    assert gap < 1e-4 * np.abs(out_clean).max()


def _check_the_centre_of_a_head_ignores_the_keys_that_head_forbids():
    layer = _set_identity_kernels(_make(num_steps=2))
    clean = _tokens(1, 6, 86)
    junk = clean.copy()
    junk[0, 5, :HEAD_DIM] = 1e4
    mask = np.ones((1, HEADS, 6, 6), dtype="float32")
    mask[0, 0, :, 5] = 0.0
    out_clean = _np(layer(clean, attention_mask=mask))
    out_junk = _np(layer(junk, attention_mask=mask))
    gap = np.abs(out_junk[0, :5, :HEAD_DIM] - out_clean[0, :5, :HEAD_DIM]).max()
    assert gap < 1e-4 * np.abs(out_clean).max()


class TestAKeyNoRowKeepsDoesNotSetTheCentre:
    @pytest.mark.parametrize("rank", [2, 3, 4])
    def test_junk_in_a_key_forbidden_to_every_row_changes_no_kept_row(self, rank):
        _check_a_key_no_row_keeps_does_not_set_the_centre(rank)

    def test_the_centre_of_each_head_follows_that_head_s_mask(self):
        _check_the_centre_of_a_head_ignores_the_keys_that_head_forbids()

    @pytest.mark.parametrize("rank", [3, 4])
    def test_the_check_fails_when_the_centre_ignores_the_mask(self, rank, monkeypatch):
        real = mean_shift_attention.kernel_logits

        def plain_mean(q, k, bandwidth, kernel="gaussian", drop_query_norm=True,
                       key_mask=None):
            return real(q, k, bandwidth, kernel, drop_query_norm)

        monkeypatch.setattr(mean_shift_attention, "kernel_logits", plain_mean)
        with pytest.raises(AssertionError):
            _check_a_key_no_row_keeps_does_not_set_the_centre(rank)
        with pytest.raises(AssertionError):
            _check_the_centre_of_a_head_ignores_the_keys_that_head_forbids()


# ---------------------------------------------------------------------
# A row with no kept key holds still, so the rows that read its column see its token
# ---------------------------------------------------------------------

def _reference_with_a_held_row(layer, x, mask, num_steps):
    """Float64 blurring Gaussian mean shift where a row without a kept key stays put.

    Built from the layer's own projection weights and bandwidth, with its own
    NumPy softmax, so it shares no code with the layer's step.
    """
    z = _heads(layer, x)
    h = _np(layer._bandwidth(layer.projection(x), "float32")).astype("float64")
    batch, count = x.shape[0], x.shape[1]
    keep = np.broadcast_to((mask > 0)[:, None], (batch, HEADS, count, count))
    live = keep.any(axis=-1, keepdims=True)
    for _ in range(num_steps):
        logits = (z @ z.transpose(0, 1, 3, 2) / h ** 2
                  - 0.5 * (z ** 2).sum(-1)[:, :, None, :] / h ** 2)
        logits = np.where(keep, logits, -np.inf)
        row_max = np.where(live, logits.max(axis=-1, keepdims=True), 0.0)
        expd = np.where(keep, np.exp(logits - row_max), 0.0)
        weights = expd / np.where(live, expd.sum(axis=-1, keepdims=True), 1.0)
        z = np.where(live, weights @ z, z)
    return _merge(z * live) @ _kernel(layer.output_projection)


def _check_a_row_with_no_kept_key_holds_still(num_steps):
    layer = _set_kernels(_make(num_steps=num_steps, blurring=True))
    x = _tokens(1, 6, 87)
    mask = np.ones((1, 6, 6), dtype="float32")
    mask[0, 0, :] = 0.0
    out = _np(layer(x, attention_mask=mask))
    reference = _reference_with_a_held_row(layer, x, mask, num_steps)
    assert np.all(out[0, 0] == 0.0)
    np.testing.assert_allclose(out[0, 1:], reference[0, 1:], atol=1e-4, rtol=0)


class TestARowWithNoKeptKeyHoldsStill:
    @pytest.mark.parametrize("num_steps", [1, 2, 3])
    def test_the_other_rows_read_the_held_token_of_that_row(self, num_steps):
        _check_a_row_with_no_kept_key_holds_still(num_steps)

    @pytest.mark.parametrize("num_steps", [2, 3])
    def test_the_check_fails_when_the_row_is_not_held(self, num_steps, monkeypatch):
        monkeypatch.setattr(MeanShiftAttention, "_hold_still",
                            staticmethod(lambda live, moved, z: moved))
        with pytest.raises(AssertionError):
            _check_a_row_with_no_kept_key_holds_still(num_steps)


# ---------------------------------------------------------------------
# Cross form and input contract
# ---------------------------------------------------------------------

class TestInputs:
    def test_rank_four_input_raises_in_build(self):
        layer = MeanShiftAttention(dim=DIM, num_heads=HEADS)
        with pytest.raises(ValueError, match="rank 4"):
            layer(np.zeros((1, 2, 2, DIM), dtype="float32"))
        with pytest.raises(ValueError, match="rank 4"):
            MeanShiftAttention(dim=DIM, num_heads=HEADS).build((None, 2, 2, DIM))

    def test_a_wrong_last_axis_raises(self):
        with pytest.raises(ValueError, match=f"last axis {DIM}"):
            MeanShiftAttention(dim=DIM, num_heads=HEADS)(np.zeros((1, 3, DIM + 1), "float32"))

    def test_the_cross_form_attends_from_the_queries_to_the_key_set(self):
        h = 1.2
        layer = _set_kernels(_make(blurring=False, bandwidth_init=h))
        xq, xkv = _tokens(2, 4, 20), _tokens(2, 7, 21)
        mask = np.array([[1] * 7, [1, 1, 1, 1, 1, 0, 0]], dtype="float32")
        out, weights = layer([xq, xkv], attention_mask=mask, return_attention_scores=True)
        assert tuple(out.shape) == (2, 4, DIM)
        assert tuple(weights.shape) == (2, HEADS, 4, 7)

        q, k = _heads(layer, xq), _heads(layer, xkv)
        ref, _ = oracle_step(q, k, k, h, keep=np.broadcast_to(mask[:, None, :] > 0, (2, HEADS, 7)))
        ref = _merge(ref) @ _kernel(layer.output_projection)
        np.testing.assert_allclose(_np(out), ref, atol=1e-5, rtol=0)
        assert np.all(_np(weights)[1, :, :, 5:] == 0.0)

    def test_the_cross_form_with_blurring_raises(self):
        layer = MeanShiftAttention(dim=DIM, num_heads=HEADS, blurring=True)
        with pytest.raises(ValueError, match="blurring=False"):
            layer([_tokens(1, 3, 1), _tokens(1, 4, 2)])

    def test_a_bad_list_length_raises(self):
        layer = MeanShiftAttention(dim=DIM, num_heads=HEADS, blurring=False)
        with pytest.raises(ValueError, match="got 3 entries"):
            layer([_tokens(1, 3, 1)] * 3)
        built = _make(blurring=False)
        with pytest.raises(ValueError, match="got 1 entries"):
            built([_tokens(1, 3, 1)])

    def test_the_output_shape_matches_the_call(self):
        layer = _make(blurring=False)
        assert layer.compute_output_shape((2, 5, DIM)) == (2, 5, DIM)
        assert layer.compute_output_shape([(2, 5, DIM), (2, 9, DIM)]) == (2, 5, DIM)
        assert tuple(layer([_tokens(2, 5, 1), _tokens(2, 9, 2)]).shape) == (2, 5, DIM)


# ---------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------

BAD_ARGUMENTS = [
    ({"dim": 0}, "dim", "0"),
    ({"num_heads": 0}, "num_heads", "0"),
    ({"dim": 30, "num_heads": 4}, "dim", "30"),
    ({"head_dim": 0}, "head_dim", "0"),
    ({"num_steps": 0}, "num_steps", "0"),
    ({"num_steps": 2.0}, "num_steps", "2.0"),
    ({"kernel": "laplace"}, "kernel", "'laplace'"),
    ({"bandwidth_mode": "per_row"}, "bandwidth_mode", "'per_row'"),
    ({"min_bandwidth": 5.0, "max_bandwidth": 5.0}, "min_bandwidth", "5.0"),
    ({"min_bandwidth": 6.0, "max_bandwidth": 5.0}, "max_bandwidth", "5.0"),
    ({"bandwidth_init": 1e3}, "bandwidth_init", "1000.0"),
    ({"bandwidth_init": 1e-2}, "bandwidth_init", "0.01"),
    ({"step_size": 0.0}, "step_size", "0.0"),
    ({"step_size": 2.0}, "step_size", "2.0"),
    ({"dropout_rate": 1.0}, "dropout_rate", "1.0"),
    ({"blurring": 1}, "blurring", "1"),
]


class TestConstructor:
    @pytest.mark.parametrize("kwargs,name,value", BAD_ARGUMENTS)
    def test_a_bad_argument_raises_naming_the_value(self, kwargs, name, value):
        kwargs = {"dim": DIM, "num_heads": HEADS, **kwargs}
        with pytest.raises(ValueError) as caught:
            MeanShiftAttention(**kwargs)
        assert name in str(caught.value) and value in str(caught.value)

    def test_the_defaults(self):
        layer = MeanShiftAttention(dim=32, num_heads=4)
        assert layer.head_dim == 8 and layer.bandwidth_init == pytest.approx(8 ** 0.25)
        assert layer.num_steps == 1 and layer.blurring and layer.kernel == "gaussian"
        assert layer.bandwidth_mode == "per_head" and layer.step_size == 1.0

    def test_a_head_dim_independent_of_the_dim_is_allowed(self):
        layer = _make(dim=30, num_heads=4, head_dim=5)
        assert tuple(layer(_tokens(2, 3, 1, dim=30)).shape) == (2, 3, 30)
        assert tuple(layer.projection.kernel.shape) == (30, 20)

    def test_the_weights_per_bandwidth_mode(self):
        names = {}
        for mode in ("global", "per_head", "per_token"):
            layer = _make(bandwidth_mode=mode, learnable_step_size=True)
            names[mode] = sorted(v.path.split(f"{layer.name}/")[-1] for v in layer.weights)
        assert names["global"] == ["output_projection/kernel", "projection/kernel",
                                   "raw_bandwidth", "raw_step"]
        assert _make(bandwidth_mode="global").raw_bandwidth.shape == ()
        assert _make(bandwidth_mode="per_head").raw_bandwidth.shape == (HEADS,)
        assert "bandwidth_projection/bias" in names["per_token"]
        assert not any("raw_bandwidth" in n for n in names["per_token"])


# ---------------------------------------------------------------------
# Bandwidth
# ---------------------------------------------------------------------

class TestBandwidth:
    @pytest.mark.parametrize("y", [1e-6, 1e-3, 1.0, 10.0, 100.0])
    def test_the_softplus_inverse_round_trips(self, y):
        x = softplus_inverse(y)
        assert math.log1p(math.exp(x)) == pytest.approx(y, rel=1e-9)

    def test_the_softplus_inverse_refuses_a_non_positive_argument(self):
        for bad in (0.0, -1.0, float("inf"), float("nan")):
            with pytest.raises(ValueError):
                softplus_inverse(bad)

    def test_a_tiny_gap_above_the_minimum_does_not_collapse(self):
        draws = np.random.default_rng(0).uniform(1e-7, 1e-4, size=40)
        for gap in draws:
            layer = _make(bandwidth_init=0.01 + float(gap), num_heads=1, dim=8)
            tokens = np.zeros((1, 2, 8), dtype="float32")
            h = float(_np(layer._bandwidth(tokens, "float32")).reshape(-1)[0])
            assert h - 0.01 == pytest.approx(gap, rel=2e-2)

    @pytest.mark.parametrize("mode", ["global", "per_head", "per_token"])
    def test_the_effective_bandwidth_starts_at_the_default(self, mode):
        layer = _make(bandwidth_mode=mode)
        projected = layer.projection(_tokens(2, 4, 1))
        h = _np(layer._bandwidth(projected, "float32"))
        np.testing.assert_allclose(h, HEAD_DIM ** 0.25, rtol=1e-6)

    def test_a_huge_bandwidth_gives_the_mean_and_a_tiny_one_the_nearest_point(self):
        x = _tokens(2, 6, 22)
        huge = _set_kernels(_make())
        huge.raw_bandwidth.assign(np.full((HEADS,), 1e3, "float32"))
        _, weights = huge(x, return_attention_scores=True)
        np.testing.assert_allclose(_np(weights), 1.0 / 6, atol=5e-3)

        tiny = _set_kernels(_make())
        tiny.raw_bandwidth.assign(np.full((HEADS,), -30.0, "float32"))
        out = _np(tiny(x))
        own = _merge(_heads(tiny, x)) @ _kernel(tiny.output_projection)
        np.testing.assert_allclose(out, own, atol=1e-3, rtol=0)

    def test_the_per_token_bandwidth_varies_and_stays_in_range(self):
        x = _tokens(2, 6, 23)
        layer = _set_kernels(_make(bandwidth_mode="per_token"))
        zero_out = _np(layer(x))
        rng = np.random.default_rng(2)
        layer.bandwidth_projection.kernel.assign(
            rng.normal(size=layer.bandwidth_projection.kernel.shape).astype("float32")
        )
        projected = layer.projection(x)
        h = _np(layer._bandwidth(projected, "float32"))
        assert h.shape == (2, HEADS, 6, 1)
        assert h.std() > 1e-2
        assert h.min() >= layer.min_bandwidth and h.max() <= layer.max_bandwidth
        assert np.abs(_np(layer(x)) - zero_out).max() > 1e-3

        layer.bandwidth_projection.kernel.assign(
            500.0 * rng.normal(size=layer.bandwidth_projection.kernel.shape).astype("float32")
        )
        h = _np(layer._bandwidth(layer.projection(x), "float32"))
        assert h.min() >= layer.min_bandwidth and h.max() <= layer.max_bandwidth
        assert h.max() == layer.max_bandwidth

    def test_the_per_token_bandwidth_is_per_query(self):
        h = 1.3
        x = _tokens(1, 5, 24)
        layer = _set_kernels(_make(bandwidth_mode="per_token", bandwidth_init=h))
        rng = np.random.default_rng(3)
        layer.bandwidth_projection.kernel.assign(
            rng.normal(size=layer.bandwidth_projection.kernel.shape).astype("float32") * 0.3
        )
        out, weights = layer(x, return_attention_scores=True)
        hq = _np(layer._bandwidth(layer.projection(x), "float32")).astype("float64")
        q = _heads(layer, x)
        key_sq = np.sum(q * q, axis=-1)[:, :, None, :]
        logits = (q @ q.transpose(0, 1, 3, 2) - 0.5 * key_sq) / hq ** 2
        np.testing.assert_allclose(_np(weights), _softmax_np(logits), atol=1e-5, rtol=0)

    def test_the_step_size_is_one_at_init_when_learnable(self):
        x = _tokens(2, 6, 25)
        fixed = _set_kernels(_make(num_steps=2, step_size=1.0))
        learnable = _make(num_steps=2, learnable_step_size=True)
        _copy_weights(fixed, learnable)
        assert np.all(_np(learnable.raw_step) == 0.0)
        np.testing.assert_allclose(_np(learnable(x)), _np(fixed(x)), atol=1e-6, rtol=0)

    def test_a_learnable_step_size_starts_at_the_requested_value(self):
        layer = _make(learnable_step_size=True, step_size=0.5)
        eta = _np(layer._step_size("float32"))
        np.testing.assert_allclose(eta, 0.5, rtol=1e-6)
        assert eta.shape == (1, HEADS, 1, 1)


# ---------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------

def _gradients(layer, x):
    x = tf.constant(x)
    with tf.GradientTape() as tape:
        tape.watch(x)
        out = layer(x)
        loss = tf.reduce_sum(out * out)
    variables = layer.trainable_weights
    grads = tape.gradient(loss, variables + [x])
    named = {
        v.path.split(f"{layer.name}/")[-1]: _np(g) for v, g in zip(variables, grads[:-1])
    }
    named["input"] = _np(grads[-1])
    return named


class TestGradients:
    @pytest.mark.parametrize("kernel", ["gaussian", "vmf"])
    @pytest.mark.parametrize("mode", ["global", "per_head", "per_token"])
    @pytest.mark.parametrize("arm", ["default", "min_bandwidth"])
    @pytest.mark.parametrize("steps", [1, 3, 8])
    def test_every_gradient_is_finite_and_the_weights_are_live(
            self, steps, arm, mode, kernel
    ):
        layer = _set_kernels(_make(num_steps=steps, bandwidth_mode=mode, kernel=kernel))
        if arm == "min_bandwidth":
            if mode == "per_token":
                layer.bandwidth_projection.bias.assign(np.full((HEADS,), -30.0, "float32"))
            else:
                layer.raw_bandwidth.assign(np.full(layer.raw_bandwidth.shape, -30.0, "float32"))
            h = _np(layer._bandwidth(layer.projection(_tokens(2, 6, 26)), "float32"))
            assert h.min() == h.max() == np.float32(layer.min_bandwidth)
        grads = _gradients(layer, _tokens(3, 6, 26))
        for name, grad in grads.items():
            assert np.isfinite(grad).all(), f"{name} gradient not finite"
        if steps <= 3:
            bandwidth_weight = (
                "bandwidth_projection/bias" if mode == "per_token" else "raw_bandwidth"
            )
            for name in ("projection/kernel", "output_projection/kernel", bandwidth_weight):
                assert np.abs(grads[name]).max() > 0.0, f"{name} gradient is zero"

    def test_a_learnable_step_size_receives_a_gradient(self):
        layer = _set_kernels(_make(num_steps=2, learnable_step_size=True))
        grads = _gradients(layer, _tokens(3, 6, 27))
        assert np.isfinite(grads["raw_step"]).all()
        assert np.abs(grads["raw_step"]).max() > 0.0


# ---------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------

class TestDropout:
    def test_dropout_acts_only_in_training(self):
        x = _tokens(2, 6, 28)
        layer = _set_kernels(_make(dropout_rate=0.5))
        plain = _set_kernels(_make(dropout_rate=0.0))
        np.testing.assert_allclose(
            _np(layer(x, training=False)), _np(plain(x)), atol=1e-6, rtol=0
        )
        np.testing.assert_array_equal(_np(layer(x)), _np(layer(x, training=False)))
        assert np.abs(_np(layer(x, training=True)) - _np(plain(x))).max() > 1e-3


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------

CONFIGS = [
    {},
    {"head_dim": 4, "bandwidth_init": 2.0, "num_steps": 3, "blurring": False},
    {"bandwidth_mode": "per_token", "kernel": "vmf", "learnable_step_size": True,
     "step_size": 0.7, "kernel_regularizer": "l2", "use_bias": True, "dropout_rate": 0.1},
]


class TestSerialization:
    @pytest.mark.parametrize("extra", CONFIGS)
    def test_the_config_round_trips_on_values(self, extra):
        layer = _set_kernels(_make(**extra))
        x = _tokens(2, 5, 29)
        config = layer.get_config()
        json.dumps(config)
        clone = MeanShiftAttention.from_config(config)
        clone.build((None, None, DIM))
        clone.set_weights(layer.get_weights())
        assert clone.get_config() == config
        np.testing.assert_allclose(
            _np(clone(x, training=False)), _np(layer(x, training=False)), atol=1e-6, rtol=0
        )

    def test_none_arguments_stay_none_in_the_config(self):
        config = _make().get_config()
        assert config["head_dim"] is None and config["bandwidth_init"] is None
        assert _make(head_dim=4, bandwidth_init=2.0).get_config()["head_dim"] == 4

    def test_the_layer_saves_and_loads_as_a_keras_file(self, tmp_path):
        inputs = keras.Input((None, DIM))
        layer = MeanShiftAttention(
            dim=DIM, num_heads=HEADS, num_steps=2, bandwidth_mode="per_token",
            learnable_step_size=True,
        )
        model = keras.Model(inputs, layer(inputs))
        x = _tokens(2, 5, 30)
        before = _np(model(x))
        path = tmp_path / "mean_shift.keras"
        model.save(path)
        loaded = keras.models.load_model(path)
        np.testing.assert_allclose(_np(loaded(x)), before, atol=1e-6, rtol=0)
        assert len(loaded.weights) == len(model.weights)

    def test_the_regularizer_reaches_both_projections(self):
        layer = _make(kernel_regularizer="l2")
        assert len(layer.losses) == 2

    def test_fewer_parameters_than_the_multi_head_attention_of_the_same_width(self):
        ours = _make(dim=64, num_heads=8)
        mha = MultiHeadAttention(dim=64, num_heads=8, use_bias=False)
        mha.build((None, None, 64))
        assert ours.count_params() < mha.count_params(), (
            f"MeanShiftAttention {ours.count_params()} vs "
            f"MultiHeadAttention {mha.count_params()}"
        )


# ---------------------------------------------------------------------
# Precision and graph mode
# ---------------------------------------------------------------------

def _run(policy, x, mask=None, **kwargs):
    """Build a layer under ``policy``, give it fixed kernels, run it eagerly."""
    previous = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy(policy)
    try:
        layer = _set_kernels(_make(**kwargs))
        out = layer(x, attention_mask=mask)
        return layer, out
    finally:
        keras.mixed_precision.set_global_policy(previous)


class TestPrecision:
    @pytest.mark.parametrize("kernel", ["gaussian", "vmf"])
    def test_mixed_float16_stays_finite_and_close_to_float32(self, kernel):
        x = _tokens(2, 6, 31)
        mask = np.array([[1, 1, 1, 1, 0, 0], [1] * 6], dtype="float32")
        _, control = _run("float32", x, mask, kernel=kernel, num_steps=3)
        _, mixed = _run("mixed_float16", x, mask, kernel=kernel, num_steps=3)
        assert keras.backend.standardize_dtype(mixed.dtype) == "float16"
        assert np.isfinite(_np(mixed)).all()
        # Measured max difference: 1.5e-3 (gaussian) and 4.4e-4 (vmf) on an output of
        # magnitude about 1.5; the bound is 0.5 percent of the output scale.
        scale = np.abs(_np(control)).max()
        np.testing.assert_allclose(
            _np(mixed).astype("float32"), _np(control), atol=5e-3 * scale, rtol=0
        )

    def test_mixed_float16_keeps_the_attention_weights_a_distribution(self):
        x = _tokens(2, 6, 32)
        layer, _ = _run("mixed_float16", x)
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            _, weights = layer(x, return_attention_scores=True)
        finally:
            keras.mixed_precision.set_global_policy(previous)
        np.testing.assert_allclose(
            _np(weights).astype("float32").sum(axis=-1), 1.0, atol=5e-3
        )

    def test_mixed_float16_dropout_and_per_token_bandwidth_run(self):
        x = _tokens(2, 6, 33)
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            layer = _set_kernels(_make(bandwidth_mode="per_token", dropout_rate=0.2))
            out = layer(x, training=True)
        finally:
            keras.mixed_precision.set_global_policy(previous)
        assert np.isfinite(_np(out).astype("float32")).all()

    def test_float64_runs_in_float64(self, float64_policy):
        x = _tokens(2, 6, 34).astype("float64")
        layer = _set_kernels(_make(num_steps=2))
        out = layer(x)
        assert keras.backend.standardize_dtype(out.dtype) == "float64"
        assert keras.backend.standardize_dtype(layer.raw_bandwidth.dtype) == "float64"
        control = _set_kernels(_make(num_steps=2))
        np.testing.assert_allclose(_np(control(x)), _np(out), atol=1e-12)

    @pytest.mark.parametrize("policy", ["float32", "mixed_float16"])
    def test_one_concrete_function_serves_two_lengths(self, policy):
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy(policy)
        try:
            layer = _set_kernels(_make(num_steps=2))

            @tf.function
            def run(x, mask):
                return layer(x, attention_mask=mask)

            concrete = run.get_concrete_function(
                tf.TensorSpec([None, None, DIM], tf.float32),
                tf.TensorSpec([None, None], tf.float32),
            )
            atol = 1e-5 if policy == "float32" else 1e-3
            for count in (5, 9):
                x = _tokens(2, count, 35)
                mask = np.ones((2, count), dtype="float32")
                mask[0, count - 2:] = 0.0
                graph = _np(concrete(tf.constant(x), tf.constant(mask)))
                eager = _np(layer(x, attention_mask=mask))
                assert graph.shape == (2, count, DIM)
                np.testing.assert_allclose(
                    graph.astype("float32"), eager.astype("float32"), atol=atol, rtol=0
                )
        finally:
            keras.mixed_precision.set_global_policy(previous)


# ---------------------------------------------------------------------
# Attention factory
# ---------------------------------------------------------------------

FULL_KWARGS = {
    "dim": DIM, "num_heads": HEADS, "head_dim": 4, "num_steps": 3, "blurring": False,
    "kernel": "vmf", "bandwidth_mode": "per_token", "bandwidth_init": 2.0,
    "min_bandwidth": 0.05, "max_bandwidth": 50.0, "step_size": 0.8,
    "learnable_step_size": True, "dropout_rate": 0.1, "use_bias": True,
    "kernel_initializer": "he_normal", "kernel_regularizer": "l2",
}


class TestFactory:
    def test_the_key_is_registered_and_listed(self):
        assert "mean_shift" in ATTENTION_REGISTRY
        assert "mean_shift" in list_attention_types()
        assert ATTENTION_REGISTRY["mean_shift"]["class"] is MeanShiftAttention

    def test_dim_alone_builds_the_layer_with_the_constructor_defaults(self):
        layer = create_attention_layer("mean_shift", name="ms", dim=DIM)
        assert type(layer) is MeanShiftAttention
        direct = MeanShiftAttention(dim=DIM, name="ms")
        assert layer.get_config() == direct.get_config()

    def test_a_full_kwargs_dict_reaches_the_constructor(self):
        layer = create_attention_layer("mean_shift", name="ms", **FULL_KWARGS)
        config = layer.get_config()
        assert layer.name == "ms"
        for key in ("dim", "num_heads", "head_dim", "num_steps", "blurring", "kernel",
                    "bandwidth_mode", "bandwidth_init", "min_bandwidth", "max_bandwidth",
                    "step_size", "learnable_step_size", "dropout_rate", "use_bias"):
            assert config[key] == FULL_KWARGS[key], key

    def test_an_unknown_keyword_raises_the_strict_error(self):
        with pytest.raises(ValueError, match=re.escape(STRICT_DROPPED_KEY_MARKER)):
            create_attention_layer("mean_shift", dim=DIM, bandwidth_scale=2.0)

    def test_a_keyword_of_another_type_raises_the_strict_error(self):
        with pytest.raises(ValueError, match=re.escape(STRICT_DROPPED_KEY_MARKER)):
            create_attention_layer("mean_shift", dim=DIM, key_dim=8)

    def test_an_explicit_none_head_dim_is_refused_by_the_factory_but_not_by_the_class(self):
        """The factory validates a caller-passed ``head_dim`` as a positive int; omission is fine."""
        assert MeanShiftAttention(dim=DIM, head_dim=None).head_dim == DIM // 8
        with pytest.raises(ValueError):
            create_attention_layer("mean_shift", dim=DIM, head_dim=None)

    def test_the_registry_defaults_are_the_constructor_defaults(self):
        info = get_attention_info()["mean_shift"]
        assert info["required_params"] == ["dim"]
        parameters = inspect.signature(MeanShiftAttention.__init__).parameters
        defaults = {
            name: p.default for name, p in parameters.items()
            if name not in ("self", "kwargs") and p.default is not inspect.Parameter.empty
        }
        assert info["optional_params"] == defaults
        required = [name for name, p in parameters.items()
                    if name not in ("self", "kwargs") and p.default is inspect.Parameter.empty]
        assert info["required_params"] == required

    def test_a_factory_layer_runs_on_a_padded_batch(self):
        layer = create_attention_layer("mean_shift", dim=DIM, num_heads=HEADS, num_steps=2)
        x = _tokens(2, 6, 41)
        mask = np.ones((2, 6), dtype="float32")
        mask[1, 4:] = 0.0
        out = _np(layer(x, attention_mask=mask, training=False))
        assert out.shape == x.shape and np.isfinite(out).all()

    def test_a_rank_4_input_is_refused_at_build(self):
        with pytest.raises(ValueError):
            create_attention_layer("mean_shift", dim=DIM).build((None, 4, 4, DIM))


# ---------------------------------------------------------------------
# TransformerLayer
# ---------------------------------------------------------------------

def _transformer(**kwargs):
    kwargs.setdefault("attention_dropout_rate", 0.0)
    return TransformerLayer(
        hidden_size=DIM, num_heads=HEADS, intermediate_size=2 * DIM,
        attention_type="mean_shift", dropout_rate=0.0, **kwargs,
    )


class TestTransformerLayer:
    @pytest.mark.parametrize("position", ["pre", "post"])
    @pytest.mark.parametrize("args", [{}, {"num_steps": 3, "kernel": "vmf", "blurring": False}])
    def test_the_block_builds_and_runs(self, position, args):
        layer = _transformer(attention_args=args, normalization_position=position)
        x = _tokens(2, 7, 42)
        out = _np(layer(x, training=False))
        assert out.shape == x.shape and np.isfinite(out).all()
        assert type(layer.attention) is MeanShiftAttention

    def test_the_block_forwards_the_dropout_and_bias_arguments(self):
        layer = TransformerLayer(
            hidden_size=DIM, num_heads=HEADS, intermediate_size=2 * DIM,
            attention_type="mean_shift", attention_dropout_rate=0.25, use_bias=True,
        )
        assert layer.attention.dropout_rate == 0.25 and layer.attention.use_bias is True

    def test_a_padding_mask_changes_nothing_for_the_unpadded_rows(self):
        layer = _transformer(attention_args={"num_steps": 2})
        x = _tokens(2, 6, 43)
        mask = np.ones((2, 6), dtype="float32")
        mask[1, 3:] = 0.0
        masked = _np(layer(x, attention_mask=mask, training=False))
        perturbed = x.copy()
        perturbed[1, 3:] += 5.0
        again = _np(layer(perturbed, attention_mask=mask, training=False))
        np.testing.assert_allclose(masked[1, :3], again[1, :3], atol=1e-5, rtol=0)
        np.testing.assert_allclose(masked[0], again[0], atol=0, rtol=0)
        unmasked = _np(layer(x, training=False))
        assert np.abs(unmasked[1, :3] - masked[1, :3]).max() > 1e-4, "the mask never reached the layer"

    def test_a_model_containing_the_block_round_trips_through_a_keras_file(self, tmp_path):
        inputs = keras.Input((None, DIM))
        outputs = _transformer(attention_args={"num_steps": 2, "bandwidth_mode": "per_token"})(inputs)
        model = keras.Model(inputs, outputs)
        x = _tokens(2, 5, 44)
        before = _np(model(x, training=False))
        path = tmp_path / "block.keras"
        model.save(path)
        loaded = keras.models.load_model(path)
        np.testing.assert_allclose(_np(loaded(x, training=False)), before, atol=1e-6, rtol=0)
        assert len(loaded.weights) == len(model.weights)

    def test_every_weight_of_the_block_receives_a_gradient(self):
        layer = _transformer(attention_args={"num_steps": 2})
        x = tf.Variable(_tokens(2, 6, 45))
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(layer(x, training=True)))
        grads = tape.gradient(loss, layer.trainable_variables)
        missing = [v.path for v, g in zip(layer.trainable_variables, grads) if g is None]
        assert not missing, missing
