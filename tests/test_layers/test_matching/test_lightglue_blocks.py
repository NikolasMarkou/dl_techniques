"""Tests for LightGlueSelfBlock and LightGlueCrossBlock.

The convention-bearing choices (the ``(H, Dh, 3)`` qkv layout, LayerNorm epsilon 1e-5,
exact-erf GELU, the cross-attention double softmax) are invisible to shape checks. Each is
therefore a VALUE test against the explicit-loop numpy oracle in
``tests/lightglue_reference_numpy.py`` with random torch-layout weights loaded through a
test-only transpose mapping, and each has a mutation test that injects the wrong
convention and asserts the parity check FAILS (so the guard is shown able to fail).
"""

import math

import keras
import numpy as np
import pytest

from dl_techniques.layers.matching import lightglue_blocks
from dl_techniques.layers.matching.learned_fourier_rotary import LearnedFourierRotaryEncoding
from dl_techniques.layers.matching.lightglue_blocks import LightGlueCrossBlock, LightGlueSelfBlock
from tests import lightglue_reference_numpy as ref
from tests.test_models.test_sam.dead_component_oracle import (
    NO_GRADIENTS_MESSAGE,
    component_response,
    fit_one_step_moved_variables,
    no_op_kill,
    outputs_stop_gradient,
    zeroed_variables,
)

_EPS32 = float(np.finfo(np.float32).eps)
_EPS16 = float(np.finfo(np.float16).eps)
_NP = {"float16": np.float16, "bfloat16": np.float32, "float32": np.float32, "float64": np.float64}

D, H = 16, 2
DH = D // H


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _torch_weights(seed, prefix_self="s", prefix_cross="c"):
    """Torch-named random weights for one self block and one cross block.

    Reuses the oracle's own generator (layer 0), renaming ``transformers.0.self_attn`` and
    ``transformers.0.cross_attn`` to short prefixes.
    """
    rng = np.random.RandomState(seed)
    full = ref.random_weights(rng, D, H, 1)
    out = {}
    for key, value in full.items():
        if key.startswith("transformers.0.self_attn."):
            out[prefix_self + key[len("transformers.0.self_attn"):]] = value
        elif key.startswith("transformers.0.cross_attn."):
            out[prefix_cross + key[len("transformers.0.cross_attn"):]] = value
    return out


def _round(array, cdt):
    """Round a float64 array through the compute dtype (what Keras' autocast does)."""
    return array.astype(_NP[cdt]).astype(np.float64)


def _round_weights(weights, cdt):
    return {k: _round(v, cdt) for k, v in weights.items()}


def _assign(variable, value):
    variable.assign(np.asarray(value, dtype=variable.dtype))


def _load_ffn(block, w, prefix):
    """Test-only torch -> Keras mapping: Linear weight transposed, LayerNorm weight/bias."""
    _assign(block.ffn_0.kernel, w[prefix + ".ffn.0.weight"].T)
    _assign(block.ffn_0.bias, w[prefix + ".ffn.0.bias"])
    _assign(block.ffn_1.gamma, w[prefix + ".ffn.1.weight"])
    _assign(block.ffn_1.beta, w[prefix + ".ffn.1.bias"])
    _assign(block.ffn_3.kernel, w[prefix + ".ffn.3.weight"].T)
    _assign(block.ffn_3.bias, w[prefix + ".ffn.3.bias"])


def _load_self(block, w, prefix="s"):
    _assign(block.Wqkv.kernel, w[prefix + ".Wqkv.weight"].T)
    _assign(block.Wqkv.bias, w[prefix + ".Wqkv.bias"])
    _assign(block.out_proj.kernel, w[prefix + ".out_proj.weight"].T)
    _assign(block.out_proj.bias, w[prefix + ".out_proj.bias"])
    _load_ffn(block, w, prefix)


def _load_cross(block, w, prefix="c"):
    for name in ("to_qk", "to_v", "to_out"):
        layer = getattr(block, name)
        _assign(layer.kernel, w[f"{prefix}.{name}.weight"].T)
        _assign(layer.bias, w[f"{prefix}.{name}.bias"])
    _load_ffn(block, w, prefix)


def _tol(cdt, expected):
    """Parity tolerance, derived from the arithmetic, not tuned to a run.

    float64: pure rounding of a few hundred ops, 1e-10. float32: eps32 per op over about 8
    chained stages (Wqkv, rotary, scores, softmax, value matmul, out_proj, ffn_0, LN,
    ffn_3), each stage at most doubling the relative error through LayerNorm and fan-in of
    16 to 32 terms: ``64 * eps32 * (1 + max|out|)``. float16: the same stage count with
    eps16 per stage, the Dense and rotary arithmetic and the value matmul run in float16
    (softmax and scores run in float32), ``8 * 4 * eps16 * (1 + max|out|)``; this ceiling
    scales with the output magnitude because the residual stream carries the input
    through a float16 add.
    """
    scale = 1.0 + float(np.abs(expected).max())
    if cdt == "float64":
        return 1e-10 * scale
    if cdt == "float16":
        return 32 * _EPS16 * scale
    return 64 * _EPS32 * scale


def _table(w_posenc, kpts, cdt):
    """Rotary table (2, B, 1, N, Dh) built by the ORACLE (float64), cast like the layer's."""
    enc = np.stack([ref.positional_encoding(w_posenc, k) for k in kpts], axis=1)  # (2,B,N,Dh)
    work = np.float64 if cdt == "float64" else np.float32
    return enc[:, :, None].astype(work)


def _setup(cdt, seed=0, b=2, n=6):
    rng = np.random.RandomState(seed + 100)
    weights = _round_weights(_torch_weights(seed), cdt)
    kpts = rng.uniform(-1, 1, size=(b, n, 2))
    wr = rng.normal(0.0, 1.0, size=(DH // 2, 2))
    x = _round(rng.normal(size=(b, n, D)), cdt)
    return rng, weights, kpts, wr, x


def _self_block(weights, cdt, **kwargs):
    block = LightGlueSelfBlock(dim=D, num_heads=H, **kwargs)
    block.build((None, None, D))
    _load_self(block, weights)
    return block


def _cross_block(weights, cdt, **kwargs):
    block = LightGlueCrossBlock(dim=D, num_heads=H, **kwargs)
    block.build((None, None, D))
    _load_cross(block, weights)
    return block


def _np(t):
    return keras.ops.convert_to_numpy(t).astype(np.float64)


def _tensor(a, cdt):
    return keras.ops.convert_to_tensor(a.astype(_NP[cdt]))


def _self_parity_error(block, weights, x, table, keep=None, mask=None, cdt="float32"):
    """Max abs error of the block against the oracle, over REAL rows only (all if no mask)."""
    got = _np(block(_tensor(x, cdt), table, None if mask is None else _tensor(mask, cdt)))
    worst, scale = 0.0, 0.0
    for b in range(x.shape[0]):
        real = np.ones(x.shape[1], bool) if mask is None else mask[b] > 0
        enc = np.asarray(table[:, b, 0], dtype=np.float64)
        keep_b = None if keep is None else keep[b]
        expected = ref.self_block(x[b], enc, weights, "s", H, keep_b)
        worst = max(worst, float(np.abs(got[b][real] - expected[real]).max()))
        scale = max(scale, float(np.abs(expected[real]).max()))
    return worst, scale


def _cross_parity_error(block, weights, x0, x1, mask0=None, mask1=None, keep=None, cdt="float32"):
    g0, g1 = block(
        _tensor(x0, cdt), _tensor(x1, cdt),
        None if mask0 is None else _tensor(mask0, cdt),
        None if mask1 is None else _tensor(mask1, cdt),
    )
    g0, g1 = _np(g0), _np(g1)
    worst, scale = 0.0, 0.0
    for b in range(x0.shape[0]):
        r0 = np.ones(x0.shape[1], bool) if mask0 is None else mask0[b] > 0
        r1 = np.ones(x1.shape[1], bool) if mask1 is None else mask1[b] > 0
        e0, e1 = ref.cross_block(x0[b], x1[b], weights, "c", H, None if keep is None else keep[b])
        worst = max(worst, float(np.abs(g0[b][r0] - e0[r0]).max()), float(np.abs(g1[b][r1] - e1[r1]).max()))
        scale = max(scale, float(np.abs(e0[r0]).max()), float(np.abs(e1[r1]).max()))
    return worst, scale


def _random_masks(rng, b, n, min_real=2):
    """Per-sample random padding at the tail AND scattered, at least ``min_real`` real."""
    mask = np.ones((b, n))
    for i in range(b):
        n_real = rng.randint(min_real, n)
        perm = rng.permutation(n)
        mask[i] = 0.0
        mask[i, perm[:n_real]] = 1.0
    return mask


# ---------------------------------------------------------------------
# oracle parity
# ---------------------------------------------------------------------


class TestSelfBlockParity:
    def test_unmasked_matches_oracle(self, dtype_policy):
        cdt = keras.mixed_precision.global_policy().compute_dtype
        _, weights, kpts, wr, x = _setup(cdt)
        table = _table(wr, kpts, cdt)
        err, scale = _self_parity_error(_self_block(weights, cdt), weights, x, table, cdt=cdt)
        assert err <= _tol(cdt, np.array([scale])), f"{cdt}: err {err} tol {_tol(cdt, np.array([scale]))}"

    def test_masked_matches_oracle_on_real_rows(self, dtype_policy):
        cdt = keras.mixed_precision.global_policy().compute_dtype
        rng, weights, kpts, wr, x = _setup(cdt, seed=1, b=3, n=7)
        mask = _random_masks(rng, 3, 7)
        keep = np.stack([np.outer(m, m) > 0 for m in mask])
        table = _table(wr, kpts, cdt)
        err, scale = _self_parity_error(
            _self_block(weights, cdt), weights, x, table, keep=keep, mask=mask, cdt=cdt
        )
        assert err <= _tol(cdt, np.array([scale])), f"{cdt}: err {err}"

    def test_bool_mask_equals_float_mask(self):
        _, weights, kpts, wr, x = _setup("float32", seed=2)
        rng = np.random.RandomState(0)
        mask = _random_masks(rng, 2, 6)
        table = _table(wr, kpts, "float32")
        block = _self_block(weights, "float32")
        a = _np(block(_tensor(x, "float32"), table, mask.astype("float32")))
        b = _np(block(_tensor(x, "float32"), table, mask > 0))
        np.testing.assert_array_equal(a, b)


class TestCrossBlockParity:
    def test_unmasked_matches_oracle(self, dtype_policy):
        cdt = keras.mixed_precision.global_policy().compute_dtype
        rng, weights, _, _, x0 = _setup(cdt, seed=3, n=5)
        x1 = _round(rng.normal(size=(2, 8, D)), cdt)       # M != N catches a transposed softmax
        err, scale = _cross_parity_error(_cross_block(weights, cdt), weights, x0, x1, cdt=cdt)
        assert err <= _tol(cdt, np.array([scale])), f"{cdt}: err {err}"

    def test_masked_matches_oracle_on_real_rows(self, dtype_policy):
        cdt = keras.mixed_precision.global_policy().compute_dtype
        rng, weights, _, _, x0 = _setup(cdt, seed=4, b=3, n=5)
        x1 = _round(rng.normal(size=(3, 8, D)), cdt)
        m0, m1 = _random_masks(rng, 3, 5), _random_masks(rng, 3, 8)
        keep = np.stack([np.outer(a, b) > 0 for a, b in zip(m0, m1)])
        err, scale = _cross_parity_error(
            _cross_block(weights, cdt), weights, x0, x1, m0, m1, keep=keep, cdt=cdt
        )
        assert err <= _tol(cdt, np.array([scale])), f"{cdt}: err {err}"

    def test_one_sided_masks_match_oracle(self):
        rng, weights, _, _, x0 = _setup("float32", seed=5, b=2, n=5)
        x1 = rng.normal(size=(2, 8, D))
        block = _cross_block(weights, "float32")
        m1 = _random_masks(rng, 2, 8)
        keep = np.stack([np.ones((5, 8), bool) & (m > 0)[None, :] for m in m1])
        err, scale = _cross_parity_error(block, weights, x0, x1, None, m1, keep=keep)
        assert err <= _tol("float32", np.array([scale]))
        m0 = _random_masks(rng, 2, 5)
        keep = np.stack([(m > 0)[:, None] & np.ones((5, 8), bool) for m in m0])
        err, scale = _cross_parity_error(block, weights, x0, x1, m0, None, keep=keep)
        assert err <= _tol("float32", np.array([scale]))

    def test_both_directions_are_used(self):
        """M != N and a symmetric-input check: the 1->0 output differs from the 0->1 output's role."""
        rng, weights, _, _, x0 = _setup("float32", seed=6, n=5)
        x1 = rng.normal(size=(2, 8, D))
        y0, y1 = _cross_block(weights, "float32")(_tensor(x0, "float32"), _tensor(x1, "float32"))
        assert tuple(y0.shape) == (2, 5, D) and tuple(y1.shape) == (2, 8, D)


# ---------------------------------------------------------------------
# padding invariance, fully masked rows
# ---------------------------------------------------------------------


class TestPaddingInvariance:
    def test_self_block_real_rows_unchanged_by_appended_padding(self, dtype_policy):
        cdt = keras.mixed_precision.global_policy().compute_dtype
        rng, weights, kpts, wr, x = _setup(cdt, seed=7, b=2, n=5)
        block = _self_block(weights, cdt)
        table = _table(wr, kpts, cdt)
        base = _np(block(_tensor(x, cdt), table, np.ones((2, 5), cdt)))

        pad = 4
        junk_x = _round(rng.normal(size=(2, pad, D)) * 5.0, cdt)
        junk_k = rng.uniform(-1, 1, size=(2, pad, 2))
        x_pad = np.concatenate([x, junk_x], axis=1)
        table_pad = _table(wr, np.concatenate([kpts, junk_k], axis=1), cdt)
        mask = np.concatenate([np.ones((2, 5)), np.zeros((2, pad))], axis=1)
        padded = _np(block(_tensor(x_pad, cdt), table_pad, _tensor(mask, cdt)))
        np.testing.assert_allclose(padded[:, :5], base, rtol=0, atol=_tol(cdt, base))

    def test_cross_block_real_rows_unchanged_by_appended_padding(self, dtype_policy):
        cdt = keras.mixed_precision.global_policy().compute_dtype
        rng, weights, _, _, x0 = _setup(cdt, seed=8, b=2, n=5)
        x1 = _round(rng.normal(size=(2, 6, D)), cdt)
        block = _cross_block(weights, cdt)
        b0, b1 = block(_tensor(x0, cdt), _tensor(x1, cdt), np.ones((2, 5), cdt), np.ones((2, 6), cdt))
        p0, p1 = 3, 5
        x0p = np.concatenate([x0, _round(rng.normal(size=(2, p0, D)) * 5.0, cdt)], axis=1)
        x1p = np.concatenate([x1, _round(rng.normal(size=(2, p1, D)) * 5.0, cdt)], axis=1)
        m0 = np.concatenate([np.ones((2, 5)), np.zeros((2, p0))], axis=1).astype(_NP[cdt])
        m1 = np.concatenate([np.ones((2, 6)), np.zeros((2, p1))], axis=1).astype(_NP[cdt])
        q0, q1 = block(_tensor(x0p, cdt), _tensor(x1p, cdt), m0, m1)
        np.testing.assert_allclose(_np(q0)[:, :5], _np(b0), rtol=0, atol=_tol(cdt, _np(b0)))
        np.testing.assert_allclose(_np(q1)[:, :6], _np(b1), rtol=0, atol=_tol(cdt, _np(b1)))

    def test_padding_actually_matters_without_the_mask(self):
        """Negative control: with no mask the junk rows DO change the real rows."""
        rng, weights, kpts, wr, x = _setup("float32", seed=7, b=2, n=5)
        block = _self_block(weights, "float32")
        table = _table(wr, kpts, "float32")
        base = _np(block(_tensor(x, "float32"), table))
        x_pad = np.concatenate([x, rng.normal(size=(2, 4, D)) * 5.0], axis=1)
        kp = np.concatenate([kpts, rng.uniform(-1, 1, size=(2, 4, 2))], axis=1)
        out = _np(block(_tensor(x_pad, "float32"), _table(wr, kp, "float32")))
        assert np.abs(out[:, :5] - base).max() > 1e-2


class TestFullyMaskedRows:
    def test_self_block_all_masked_sample_is_finite(self, dtype_policy):
        cdt = keras.mixed_precision.global_policy().compute_dtype
        _, weights, kpts, wr, x = _setup(cdt, seed=9)
        mask = np.ones((2, 6))
        mask[1] = 0.0                                   # sample 1 has no real keypoint
        out = _np(_self_block(weights, cdt)(_tensor(x, cdt), _table(wr, kpts, cdt), _tensor(mask, cdt)))
        assert np.all(np.isfinite(out))

    def test_cross_block_empty_image_is_finite(self, dtype_policy):
        cdt = keras.mixed_precision.global_policy().compute_dtype
        rng, weights, _, _, x0 = _setup(cdt, seed=10, n=5)
        x1 = _round(rng.normal(size=(2, 6, D)), cdt)
        m0 = np.ones((2, 5))
        m1 = np.ones((2, 6))
        m1[0] = 0.0                                     # image 1 empty in sample 0
        m0[1] = 0.0                                     # image 0 empty in sample 1
        y0, y1 = _cross_block(weights, cdt)(
            _tensor(x0, cdt), _tensor(x1, cdt), _tensor(m0, cdt), _tensor(m1, cdt)
        )
        assert np.all(np.isfinite(_np(y0))) and np.all(np.isfinite(_np(y1)))


# ---------------------------------------------------------------------
# mutation guards: each wrong convention must be CAUGHT by the parity check
# ---------------------------------------------------------------------


def _wrong_layout_split(qkv, num_heads, head_dim):
    """MUTANT: the (3, H, Dh) column layout instead of the reference (H, Dh, 3)."""
    shape = keras.ops.shape(qkv)
    grouped = keras.ops.reshape(qkv, (shape[0], shape[1], 3, num_heads, head_dim))
    grouped = keras.ops.transpose(grouped, (0, 3, 1, 4, 2))      # (B, H, L, Dh, 3)
    return grouped[..., 0], grouped[..., 1], grouped[..., 2]


def _tanh_gelu(x):
    """MUTANT: the tanh-approximated GELU."""
    return keras.activations.gelu(x, approximate=True)


class TestMutationGuards:
    """Each mutant breaks the same parity assertion that the real code passes."""

    def _self_err(self, block_kwargs=None):
        _, weights, kpts, wr, x = _setup("float32", seed=11)
        table = _table(wr, kpts, "float32")
        block = _self_block(weights, "float32", **(block_kwargs or {}))
        return _self_parity_error(block, weights, x, table)

    def test_baseline_passes(self):
        err, scale = self._self_err()
        assert err <= _tol("float32", np.array([scale]))

    def test_eps_1e_minus_3_is_caught(self):
        err, scale = self._self_err({"layer_norm_epsilon": 1e-3})
        assert err > 10 * _tol("float32", np.array([scale])), f"eps mutant not caught: err {err}"

    def test_tanh_gelu_is_caught(self, monkeypatch):
        monkeypatch.setattr(lightglue_blocks, "_ffn_activation", _tanh_gelu)
        err, scale = self._self_err()
        assert err > 10 * _tol("float32", np.array([scale])), f"tanh mutant not caught: err {err}"

    def test_wrong_qkv_layout_is_caught(self, monkeypatch):
        monkeypatch.setattr(lightglue_blocks, "_split_qkv", _wrong_layout_split)
        err, scale = self._self_err()
        assert err > 10 * _tol("float32", np.array([scale])), f"layout mutant not caught: err {err}"

    def test_cross_eps_and_gelu_are_caught(self, monkeypatch):
        rng, weights, _, _, x0 = _setup("float32", seed=12, n=5)
        x1 = rng.normal(size=(2, 8, D))
        err, scale = _cross_parity_error(
            _cross_block(weights, "float32", layer_norm_epsilon=1e-3), weights, x0, x1
        )
        assert err > 10 * _tol("float32", np.array([scale]))
        monkeypatch.setattr(lightglue_blocks, "_ffn_activation", _tanh_gelu)
        err, scale = _cross_parity_error(_cross_block(weights, "float32"), weights, x0, x1)
        assert err > 10 * _tol("float32", np.array([scale]))


# ---------------------------------------------------------------------
# construction, config, serialization
# ---------------------------------------------------------------------


class TestConstruction:
    @pytest.mark.parametrize("cls", [LightGlueSelfBlock, LightGlueCrossBlock])
    @pytest.mark.parametrize(
        "kwargs",
        [{"dim": 15, "num_heads": 4}, {"dim": 12, "num_heads": 4},      # indivisible; odd head_dim 3
         {"dim": 0, "num_heads": 2}, {"dim": 16, "num_heads": 0},
         {"dim": 16, "num_heads": 2, "layer_norm_epsilon": 0.0}],
    )
    def test_bad_arguments_raise(self, cls, kwargs):
        with pytest.raises(ValueError):
            cls(**kwargs)

    @pytest.mark.parametrize("cls", [LightGlueSelfBlock, LightGlueCrossBlock])
    def test_bad_build_shape_raises(self, cls):
        with pytest.raises(ValueError):
            cls(dim=16, num_heads=2).build((None, 5, 17))
        with pytest.raises(ValueError):
            cls(dim=16, num_heads=2).build((5, 16))

    def test_sublayer_names_follow_the_torch_state_dict(self):
        s = LightGlueSelfBlock(dim=16, num_heads=2)
        s.build((None, None, 16))
        assert tuple(s.Wqkv.kernel.shape) == (16, 48)       # torch Wqkv.weight (48, 16), transposed
        assert tuple(s.out_proj.kernel.shape) == (16, 16)
        assert tuple(s.ffn_0.kernel.shape) == (32, 32)
        assert tuple(s.ffn_3.kernel.shape) == (32, 16)
        assert s.ffn_1.epsilon == 1e-5
        c = LightGlueCrossBlock(dim=16, num_heads=2)
        c.build((None, None, 16))
        assert {w.path.split("/")[-2] for w in c.weights} >= {"to_qk", "to_v", "to_out", "ffn_0", "ffn_1", "ffn_3"}

    def test_cross_block_has_one_shared_projection_set(self):
        c = LightGlueCrossBlock(dim=16, num_heads=2)
        c.build((None, None, 16))
        assert len(c.trainable_variables) == 3 * 2 + 2 + 2 + 2   # to_qk, to_v, to_out, ffn_0, ffn_1, ffn_3

    @pytest.mark.parametrize("cls", [LightGlueSelfBlock, LightGlueCrossBlock])
    def test_get_config_round_trip(self, cls):
        layer = cls(dim=32, num_heads=4, layer_norm_epsilon=2e-5, name="blk")
        config = layer.get_config()
        assert config["dim"] == 32 and config["num_heads"] == 4 and config["layer_norm_epsilon"] == 2e-5
        clone = cls.from_config(config)
        assert clone.get_config() == config

    def test_compute_output_shape(self):
        assert LightGlueSelfBlock(dim=16, num_heads=2).compute_output_shape((None, 9, 16)) == (None, 9, 16)
        assert LightGlueCrossBlock(dim=16, num_heads=2).compute_output_shape(
            (None, 9, 16), (None, 7, 16)
        ) == ((None, 9, 16), (None, 7, 16))

    def test_symbolic_call_shapes(self):
        enc = LearnedFourierRotaryEncoding(head_dim=DH)
        k0, x0 = keras.Input((9, 2)), keras.Input((9, D))
        k1, x1 = keras.Input((7, 2)), keras.Input((7, D))
        m0, m1 = keras.Input((9,)), keras.Input((7,))
        s = LightGlueSelfBlock(dim=D, num_heads=H)(x0, enc(k0), m0)
        y0, y1 = LightGlueCrossBlock(dim=D, num_heads=H)(s, x1, m0, m1)
        assert tuple(s.shape) == (None, 9, D)
        assert tuple(y0.shape) == (None, 9, D) and tuple(y1.shape) == (None, 7, D)


def _wrapper_model(n0=6, n1=5):
    enc = LearnedFourierRotaryEncoding(head_dim=DH)
    self_block = LightGlueSelfBlock(dim=D, num_heads=H)
    cross_block = LightGlueCrossBlock(dim=D, num_heads=H)
    k0, x0, m0 = keras.Input((n0, 2)), keras.Input((n0, D)), keras.Input((n0,))
    k1, x1, m1 = keras.Input((n1, 2)), keras.Input((n1, D)), keras.Input((n1,))
    s0 = self_block(x0, enc(k0), m0)
    s1 = self_block(x1, enc(k1), m1)
    y0, y1 = cross_block(s0, s1, m0, m1)
    return keras.Model([k0, x0, m0, k1, x1, m1], [y0, y1])


def _wrapper_data(rng, b=4, n0=6, n1=5):
    m0, m1 = _random_masks(rng, b, n0), _random_masks(rng, b, n1)
    return [
        rng.uniform(-1, 1, (b, n0, 2)).astype("float32"), rng.normal(size=(b, n0, D)).astype("float32"),
        m0.astype("float32"),
        rng.uniform(-1, 1, (b, n1, 2)).astype("float32"), rng.normal(size=(b, n1, D)).astype("float32"),
        m1.astype("float32"),
    ]


class TestSerialization:
    def test_save_load_round_trip_through_a_model(self, tmp_path):
        rng = np.random.RandomState(13)
        model = _wrapper_model()
        data = _wrapper_data(rng)
        expected = model.predict(data, verbose=0)
        path = str(tmp_path / "blocks.keras")
        model.save(path)
        reloaded = keras.models.load_model(path)
        for a, b in zip(reloaded.predict(data, verbose=0), expected):
            np.testing.assert_array_equal(a, b)

    def test_shared_blocks_are_stored_once(self):
        model = _wrapper_model()
        # the self block is applied to both images but owns one weight set
        assert len([layer for layer in model.layers if isinstance(layer, LightGlueSelfBlock)]) == 1


# ---------------------------------------------------------------------
# gradient flow (dead-component instrument)
# ---------------------------------------------------------------------


class TestGradientFlow:
    def _compiled(self):
        model = _wrapper_model()
        model.compile(optimizer=keras.optimizers.Adam(1e-2), loss=["mse", "mse"])
        return model

    def _targets(self, rng, data):
        return [rng.normal(size=(4, 6, D)).astype("float32"), rng.normal(size=(4, 5, D)).astype("float32")]

    def test_every_trainable_variable_moves(self):
        """Liveness arm: ALL variables move after one step, pinned by name, none excused."""
        rng = np.random.RandomState(14)
        model = self._compiled()
        data = _wrapper_data(rng)
        report = fit_one_step_moved_variables(model, data, self._targets(rng, data))
        assert report.total > 0
        assert report.unmoved == (), f"non-movers: {report.unmoved}"
        assert any("Wqkv" in label for label in report.moved)
        assert any("to_qk" in label for label in report.moved)
        assert any("learned_fourier" in label or "kernel" in label for label in report.moved)

    def test_stop_gradient_injection_makes_the_step_raise(self):
        """Dead arm: the instrument must be able to fail, with the exact Keras message."""
        rng = np.random.RandomState(15)
        model = self._compiled()
        data = _wrapper_data(rng)
        targets = self._targets(rng, data)
        model.predict(data, verbose=0)
        with outputs_stop_gradient(model):
            with pytest.raises(ValueError, match=NO_GRADIENTS_MESSAGE):
                model.fit(data, targets, epochs=1, verbose=0)

    def test_each_block_responds_when_its_weights_are_zeroed(self):
        """Component arms: zeroing a block's weights changes the output; a no-op kill does not."""
        rng = np.random.RandomState(16)
        model = _wrapper_model()
        data = _wrapper_data(rng)

        def metric():
            y0, y1 = model.predict(data, verbose=0)
            return float(np.abs(y0).sum() + np.abs(y1).sum())

        control = component_response(metric, no_op_kill, name="no-op control")
        assert not control.moved and control.delta == 0.0, control.summary()
        for cls in (LightGlueSelfBlock, LightGlueCrossBlock):
            block = next(layer for layer in model.layers if isinstance(layer, cls))
            result = component_response(
                metric, lambda b=block: zeroed_variables(b.weights), name=cls.__name__, atol=1e-4
            )
            assert result.moved, result.summary()

    def test_mask_actually_gates_the_gradient_to_padded_inputs(self):
        """A padded row's input must receive no gradient through a REAL row's output."""
        import tensorflow as tf
        _, weights, kpts, wr, x = _setup("float32", seed=17, b=1, n=5)
        block = _self_block(weights, "float32")
        table = _table(wr, kpts, "float32")
        mask = np.array([[1, 1, 1, 0, 0]], "float32")
        xt = tf.constant(x.astype("float32"))
        with tf.GradientTape() as tape:
            tape.watch(xt)
            out = block(xt, table, mask)
            loss = tf.reduce_sum(out[:, :3] ** 2)
        grad = tape.gradient(loss, xt).numpy()
        assert np.abs(grad[0, 3:]).max() == 0.0
        assert np.abs(grad[0, :3]).max() > 0.0
