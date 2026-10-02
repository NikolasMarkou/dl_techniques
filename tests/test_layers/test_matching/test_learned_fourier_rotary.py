"""Tests for LearnedFourierRotaryEncoding (LightGlue learned-Fourier rotary table).

The convention-bearing guards (interleaved pairs, ``repeat_interleave`` not ``tile``) are
invisible to a shape check, so each one is a VALUE test against the explicit-loop numpy
oracle in ``tests/lightglue_reference_numpy.py`` or against a hand-computed case.
"""

import math

import keras
import numpy as np
import pytest

from dl_techniques.layers.matching.learned_fourier_rotary import (
    LearnedFourierRotaryEncoding,
    apply_rotary_interleaved,
    rotate_half_interleaved,
)
from tests import lightglue_reference_numpy as ref

_EPS32 = float(np.finfo(np.float32).eps)


def _kpts(rng, b=2, n=7, m=2):
    return rng.uniform(-1.0, 1.0, size=(b, n, m))


def _layer_with_kernel(head_dim, kernel, **kwargs):
    """Layer built on ``(1, 1, M)`` with the Keras kernel (M, head_dim // 2) assigned."""
    layer = LearnedFourierRotaryEncoding(
        head_dim=head_dim, num_input_features=kernel.shape[0], **kwargs
    )
    layer.build((None, None, kernel.shape[0]))
    layer.kernel.assign(np.asarray(kernel, dtype=layer.kernel.dtype))
    return layer


def _impulse_out(table, channel_zero_phase_index=0):
    """Rotate a one-hot on channel 0 of token 0 with ``table`` (numpy, (2, B, 1, N, Dh))."""
    dh = table.shape[-1]
    t = np.zeros((1, 1, 1, dh))
    t[..., 0] = 1.0
    t = np.broadcast_to(t, (1, 1, table.shape[3], dh))
    rotated = apply_rotary_interleaved(table[:, :1], keras.ops.convert_to_tensor(t, "float64"))
    return keras.ops.convert_to_numpy(rotated)[0, 0, 0]


def _tile_table(kernel, kpts):
    """MUTANT: the same table built with ``tile`` instead of ``repeat_interleave``."""
    phases = kpts @ kernel
    cos = np.tile(np.cos(phases), (1, 1, 2))
    sin = np.tile(np.sin(phases), (1, 1, 2))
    return np.stack([cos, sin])[:, :, None]


class TestImpulse:
    """Invariant 1: a one-hot on channel 0 leaks into channel 1 only, with (cos, sin) of phase 0."""

    def _check(self, out, phase0):
        # Channel 0 and 1 carry the rotated pair, with the SAME phase; the rest is untouched.
        np.testing.assert_allclose(out[0], math.cos(phase0), rtol=0, atol=1e-6)
        np.testing.assert_allclose(out[1], math.sin(phase0), rtol=0, atol=1e-6)
        np.testing.assert_array_equal(out[2:], 0.0)
        np.testing.assert_allclose(out[0] ** 2 + out[1] ** 2, 1.0, rtol=0, atol=1e-6)

    def test_real_layer_satisfies_the_impulse_contract(self):
        rng = np.random.RandomState(0)
        kernel = rng.normal(0.0, 1.0, size=(2, 4))   # 4 phases, head_dim 8
        layer = _layer_with_kernel(8, kernel)
        kpts = _kpts(rng, b=1, n=3)
        table = keras.ops.convert_to_numpy(layer(kpts.astype("float32")))
        out = _impulse_out(table)
        # token 0, phase column 0
        self._check(out, float(kpts[0, 0] @ kernel[:, 0]))

    def test_tile_mutant_is_caught_by_the_same_check(self):
        """The guard must be able to fail: a tile-built table violates the contract.

        With ``tile`` the table is ``[c0, c1, c2, c3, c0, c1, ...]``, so channel 1 holds
        ``sin(phase 1)`` where the reference needs ``sin(phase 0)``.
        """
        rng = np.random.RandomState(0)
        kernel = rng.normal(0.0, 1.0, size=(2, 4))
        kpts = _kpts(rng, b=1, n=3)
        mutant = _tile_table(kernel, kpts)
        out = _impulse_out(mutant)
        with pytest.raises(AssertionError):
            self._check(out, float(kpts[0, 0] @ kernel[:, 0]))

    def test_rotate_half_is_adjacent_not_split_half(self):
        x = keras.ops.convert_to_tensor([[1.0, 2.0, 3.0, 4.0]])
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(rotate_half_interleaved(x)), [[-2.0, 1.0, -4.0, 3.0]]
        )


class TestOracleParity:
    @pytest.mark.parametrize("m", [2, 4])
    def test_table_matches_loop_oracle(self, dtype_policy, m):
        rng = np.random.RandomState(1)
        head_dim = 8
        kernel = rng.normal(0.0, 1.0, size=(m, head_dim // 2))
        layer = _layer_with_kernel(head_dim, kernel)
        kpts = _kpts(rng, b=2, n=5, m=m)

        # What the layer actually receives under the policy: Keras autocasts the input and
        # the kernel to the compute dtype BEFORE call(), outside this layer's control.
        # Feed the oracle the same rounded values, so the comparison isolates the layer's
        # own arithmetic (it must not narrow further).
        cdt = layer.compute_dtype
        np_cdt = {"float16": np.float16, "bfloat16": np.float32, "float32": np.float32,
                  "float64": np.float64}[cdt]
        kernel_seen = np.asarray(layer.kernel.value, dtype=np.float64)
        kernel_used = kernel_seen.astype(np_cdt).astype(np.float64)
        kpts_used = kpts.astype(np_cdt).astype(np.float64)
        if cdt == "float16":
            kernel_used = kernel.astype(np.float16).astype(np.float64)

        out = layer(kpts_used.astype(np_cdt))
        expected_dtype = "float64" if cdt == "float64" else "float32"
        assert keras.backend.standardize_dtype(out.dtype) == expected_dtype
        out = keras.ops.convert_to_numpy(out).astype(np.float64)
        assert out.shape == (2, 2, 1, 5, head_dim)
        assert np.all(np.isfinite(out))

        wr_torch = kernel_used.T                      # torch Wr.weight (out, in)
        tol = 1e-12 if cdt == "float64" else 16 * _EPS32 * (1.0 + float(np.abs(kernel_used).max() * m))
        for b in range(2):
            expected = ref.positional_encoding(wr_torch, kpts_used[b])   # (2, N, Dh)
            np.testing.assert_allclose(out[:, b, 0], expected, rtol=0, atol=tol)

    def test_apply_rotary_matches_oracle(self):
        rng = np.random.RandomState(2)
        head_dim, n, h = 8, 5, 3
        kernel = rng.normal(0.0, 1.0, size=(2, head_dim // 2))
        layer = _layer_with_kernel(head_dim, kernel)
        kpts = _kpts(rng, b=1, n=n)
        table = layer(kpts.astype("float32"))
        t = rng.normal(size=(1, h, n, head_dim)).astype("float32")
        got = keras.ops.convert_to_numpy(LearnedFourierRotaryEncoding.apply_rotary(table, t))
        enc = ref.positional_encoding(kernel.T, kpts[0])
        for head in range(h):
            np.testing.assert_allclose(
                got[0, head], ref.apply_rotary(enc, t[0, head].astype(np.float64)),
                rtol=0, atol=64 * _EPS32 * 4.0,
            )


class TestHandComputed:
    def test_four_channel_case(self):
        """head_dim 4, M 2, kernel chosen so the phases are exactly (pi/2, -pi/2).

        kpts = (0.5, -0.25); torch Wr = [[pi, 0], [0, 2 pi]] gives phases
        (pi/2, -pi/2): cos = (0, 0), sin = (1, -1). Interleaved table:
        cos = [0, 0, 0, 0], sin = [1, 1, -1, -1]. For t = [1, 2, 3, 4],
        rotate_half = [-2, 1, -4, 3] and out = t*cos + rot*sin = [-2, 1, 4, -3].
        """
        wr_torch = np.array([[math.pi, 0.0], [0.0, 2.0 * math.pi]])
        layer = _layer_with_kernel(4, wr_torch.T)
        table = keras.ops.convert_to_numpy(layer(np.array([[[0.5, -0.25]]], "float32")))
        np.testing.assert_allclose(table[0, 0, 0, 0], [0, 0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(table[1, 0, 0, 0], [1, 1, -1, -1], atol=1e-6)
        t = np.array([[[[1.0, 2.0, 3.0, 4.0]]]], "float32")
        out = keras.ops.convert_to_numpy(LearnedFourierRotaryEncoding.apply_rotary(table, t))
        np.testing.assert_allclose(out[0, 0, 0], [-2.0, 1.0, 4.0, -3.0], atol=1e-6)


class TestInit:
    def test_kernel_std_is_gamma_to_the_minus_two(self):
        layer = LearnedFourierRotaryEncoding(head_dim=4096, gamma=2.0)
        layer.build((None, None, 2))
        std = float(np.std(keras.ops.convert_to_numpy(layer.kernel)))
        assert abs(std - 2.0 ** -2) < 0.01

    def test_kernel_layout_is_torch_transposed(self):
        layer = LearnedFourierRotaryEncoding(head_dim=8, num_input_features=4)
        layer.build((None, None, 4))
        assert tuple(layer.kernel.shape) == (4, 4)   # (M, head_dim // 2)
        layer = LearnedFourierRotaryEncoding(head_dim=16, num_input_features=2)
        layer.build((None, None, 2))
        assert tuple(layer.kernel.shape) == (2, 8)

    @pytest.mark.parametrize(
        "kwargs", [{"head_dim": 7}, {"head_dim": 0}, {"head_dim": 8, "num_input_features": 0},
                   {"head_dim": 8, "gamma": 0.0}]
    )
    def test_bad_arguments_raise(self, kwargs):
        with pytest.raises(ValueError):
            LearnedFourierRotaryEncoding(**kwargs)

    def test_bad_input_shape_raises(self):
        layer = LearnedFourierRotaryEncoding(head_dim=8)
        with pytest.raises(ValueError):
            layer.build((None, 5, 3))
        with pytest.raises(ValueError):
            LearnedFourierRotaryEncoding(head_dim=8).build((5, 2))


class TestSerialization:
    def test_get_config_round_trip(self):
        layer = LearnedFourierRotaryEncoding(head_dim=16, num_input_features=4, gamma=1.5, name="posenc")
        config = layer.get_config()
        assert config["head_dim"] == 16 and config["num_input_features"] == 4
        assert config["gamma"] == 1.5
        clone = LearnedFourierRotaryEncoding.from_config(config)
        assert clone.get_config() == config

    def test_weights_round_trip_through_a_model(self, tmp_path):
        rng = np.random.RandomState(3)
        inp = keras.Input((6, 2))
        model = keras.Model(inp, LearnedFourierRotaryEncoding(head_dim=8)(inp))
        x = _kpts(rng, b=2, n=6).astype("float32")
        expected = model.predict(x, verbose=0)
        path = str(tmp_path / "enc.keras")
        model.save(path)
        reloaded = keras.models.load_model(path)
        np.testing.assert_array_equal(reloaded.predict(x, verbose=0), expected)

    def test_compute_output_shape(self):
        layer = LearnedFourierRotaryEncoding(head_dim=64)
        assert layer.compute_output_shape((None, 100, 2)) == (2, None, 1, 100, 64)
        out = LearnedFourierRotaryEncoding(head_dim=64)(keras.Input((100, 2)))
        assert tuple(out.shape) == (2, None, 1, 100, 64)

    def test_registered_name_is_unique(self):
        registered = keras.saving.get_registered_object(
            "dl_techniques.layers.matching>LearnedFourierRotaryEncoding"
        ) or keras.saving.get_registered_object(
            "dl_techniques>LearnedFourierRotaryEncoding"
        )
        assert registered is None or registered is LearnedFourierRotaryEncoding


class TestGradients:
    def test_kernel_receives_gradient(self):
        import tensorflow as tf
        layer = LearnedFourierRotaryEncoding(head_dim=8)
        x = tf.constant(_kpts(np.random.RandomState(4), b=1, n=4).astype("float32"))
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(layer(x)[1] * tf.cast(tf.range(8), "float32"))
        grad = tape.gradient(loss, layer.kernel)
        assert grad is not None and float(tf.reduce_max(tf.abs(grad))) > 0.0


class TestOracleSelfChecks:
    """The shared oracle is consumed by later steps; pin it against hand-computed cases here."""

    def test_filter_matches_hand_case(self):
        probs = np.array([[0.6, 0.1, 0.05], [0.2, 0.5, 0.4], [0.1, 0.45, 0.3]])
        scores = np.zeros((4, 4))
        scores[:3, :3] = np.log(probs)
        # Row argmax (0, 1, 1), column argmax (0, 1, 1): rows 0 and 1 are mutual, row 2 is not.
        m0, m1, ms0, ms1 = ref.filter_matches(scores, 0.1)
        np.testing.assert_array_equal(m0, [0, 1, -1])
        np.testing.assert_array_equal(m1, [0, 1, -1])
        np.testing.assert_allclose(ms0, [0.6, 0.5, 0.0])
        # A threshold of 0.55 removes the 0.5 match; the mutual-miss row stays unmatched.
        m0, m1, _, _ = ref.filter_matches(scores, 0.55)
        np.testing.assert_array_equal(m0, [0, -1, -1])
        np.testing.assert_array_equal(m1, [0, -1, -1])

    def test_rotation_preserves_pair_norms(self):
        rng = np.random.RandomState(5)
        enc = ref.positional_encoding(rng.normal(size=(4, 2)), rng.uniform(-1, 1, size=(6, 2)))
        t = rng.normal(size=(6, 8))
        out = ref.apply_rotary(enc, t)
        np.testing.assert_allclose(
            out.reshape(6, 4, 2).__pow__(2).sum(-1), t.reshape(6, 4, 2).__pow__(2).sum(-1), atol=1e-12
        )

    def test_full_forward_runs_and_is_a_log_distribution_bound(self):
        rng = np.random.RandomState(6)
        d, heads, layers, m, n = 8, 2, 2, 4, 5
        weights = ref.random_weights(rng, d, heads, layers)
        out = ref.forward_single(
            weights, rng.uniform(0, 64, (m, 2)), rng.normal(size=(m, d)), np.array([64.0, 48.0]),
            rng.uniform(0, 64, (n, 2)), rng.normal(size=(n, d)), np.array([64.0, 48.0]),
            layers, heads,
        )
        assert len(out["log_assignments"]) == layers and len(out["confidences0"]) == layers - 1
        scores = out["log_assignments"][-1]
        assert scores.shape == (m + 1, n + 1) and np.all(np.isfinite(scores))
        # Every real row's mass over the real columns plus the dustbin is at most 1 per factor.
        assert np.all(np.exp(scores[:m, :n]).sum(1) <= 1.0 + 1e-9)
