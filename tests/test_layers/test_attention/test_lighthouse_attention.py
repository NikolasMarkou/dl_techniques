"""Test suite for LighthouseAttention layer.

Covers: initialization, validation, build divisibility, forward shapes,
full-attention bypass, runtime toggle, sanity invariant (L=1, K=N ≡ full),
causality, gradient flow, get_config round-trip, and .keras save/load.
"""

import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf
import keras

from dl_techniques.layers.attention.lighthouse_attention import (
    LighthouseAttention,
)


class TestLighthouseAttention:
    """Test suite for LighthouseAttention."""

    # ==================== Fixtures ====================

    @pytest.fixture
    def default_input(self) -> tf.Tensor:
        return tf.random.normal([2, 64, 128])

    @pytest.fixture
    def default_layer(self) -> LighthouseAttention:
        return LighthouseAttention(
            dim=128,
            num_heads=4,
            num_levels=3,
            pooling_factor=4,
            top_k=20,
        )

    # ==================== Initialization ====================

    def test_initialization_defaults(self):
        layer = LighthouseAttention(dim=128, num_heads=4)
        assert layer.dim == 128
        assert layer.num_heads == 4
        assert layer.head_dim == 32
        assert layer.num_levels == 3
        assert layer.pooling_factor == 4
        assert layer.top_k == 1536
        assert layer.scorer == "norm"
        assert layer.full_attention is False
        assert layer.normalization_type == "rms_norm"

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            LighthouseAttention(dim=0, num_heads=4)
        with pytest.raises(ValueError):
            LighthouseAttention(dim=128, num_heads=0)
        with pytest.raises(ValueError):
            LighthouseAttention(dim=128, num_heads=5)  # 128 % 5 != 0
        with pytest.raises(ValueError):
            LighthouseAttention(dim=128, num_heads=4, pooling_factor=1)
        with pytest.raises(ValueError):
            LighthouseAttention(dim=128, num_heads=4, num_levels=0)
        with pytest.raises(ValueError):
            LighthouseAttention(dim=128, num_heads=4, scorer="dilated")
        with pytest.raises(ValueError):
            LighthouseAttention(dim=128, num_heads=4, dropout_rate=-0.1)
        with pytest.raises(ValueError):
            LighthouseAttention(dim=128, num_heads=4, top_k=0)

    # ==================== Build ====================

    def test_build_n_divisibility(self):
        # p^(L-1) = 4^2 = 16; N=63 is not divisible.
        layer = LighthouseAttention(dim=128, num_heads=4, num_levels=3, pooling_factor=4)
        with pytest.raises(ValueError):
            layer.build((None, 63, 128))
        # N=64 OK.
        layer_ok = LighthouseAttention(dim=128, num_heads=4, num_levels=3, pooling_factor=4)
        layer_ok.build((None, 64, 128))
        assert layer_ok._S_pyr == 64 + 16 + 4

    # ==================== Forward pass ====================

    def test_forward_pass_shape(self, default_layer, default_input):
        out = default_layer(default_input)
        assert tuple(out.shape) == (2, 64, 128)
        assert not bool(tf.reduce_any(tf.math.is_nan(out)).numpy())
        assert not bool(tf.reduce_any(tf.math.is_inf(out)).numpy())

    def test_full_attention_bypass_shape(self, default_input):
        layer = LighthouseAttention(
            dim=128,
            num_heads=4,
            num_levels=3,
            pooling_factor=4,
            top_k=20,
            full_attention=True,
        )
        out = layer(default_input)
        assert tuple(out.shape) == (2, 64, 128)
        assert not bool(tf.reduce_any(tf.math.is_nan(out)).numpy())

    def test_set_full_attention_toggle(self, default_input):
        layer = LighthouseAttention(
            dim=128, num_heads=4, num_levels=3, pooling_factor=4, top_k=20
        )
        y_lh = layer(default_input).numpy()
        layer.set_full_attention(True)
        assert layer.full_attention is True
        y_full = layer(default_input).numpy()
        # Outputs should differ (lighthouse vs full).
        assert not np.allclose(y_lh, y_full, atol=1e-6)

    # ==================== Sanity invariant ====================

    def test_sanity_invariant_L1_topk_eq_N(self):
        """L=1, top_k=N must equal full_attention=True to FP tolerance."""
        keras.utils.set_random_seed(123)
        N, dim, H = 32, 64, 4
        l_lh = LighthouseAttention(
            dim=dim, num_heads=H, num_levels=1, pooling_factor=2, top_k=N
        )
        l_full = LighthouseAttention(
            dim=dim,
            num_heads=H,
            num_levels=1,
            pooling_factor=2,
            top_k=N,
            full_attention=True,
        )
        inp = keras.Input(shape=(N, dim))
        m_lh = keras.Model(inp, l_lh(inp))
        inp2 = keras.Input(shape=(N, dim))
        m_full = keras.Model(inp2, l_full(inp2))
        x = tf.random.normal([2, N, dim])
        _ = m_lh(x)
        _ = m_full(x)
        m_full.set_weights(m_lh.get_weights())
        y_lh = m_lh(x).numpy()
        y_full = m_full(x).numpy()
        np.testing.assert_allclose(y_lh, y_full, atol=1e-4, rtol=1e-4)

    # ==================== Causality ====================

    def test_causality(self):
        """Perturbing input at j=N-1 must not change output at i<N/2."""
        keras.utils.set_random_seed(7)
        N, dim, H = 32, 64, 4
        layer = LighthouseAttention(
            dim=dim,
            num_heads=H,
            num_levels=2,
            pooling_factor=2,
            top_k=16,
        )
        inp = keras.Input(shape=(N, dim))
        m = keras.Model(inp, layer(inp))
        x = tf.random.normal([1, N, dim], seed=99)
        y0 = m(x).numpy()

        # Perturb only the last position.
        x_perturbed_np = x.numpy().copy()
        x_perturbed_np[:, -1, :] += 100.0
        x_perturbed = tf.constant(x_perturbed_np)
        y1 = m(x_perturbed).numpy()

        # Output positions 0..N//2 must be unchanged.
        np.testing.assert_allclose(
            y0[:, : N // 2, :], y1[:, : N // 2, :], atol=1e-5
        )

    # ==================== Gradient flow ====================

    def test_gradient_flow(self, default_layer, default_input):
        with tf.GradientTape() as tape:
            tape.watch(default_input)
            out = default_layer(default_input)
            loss = tf.reduce_sum(out)
        grads = tape.gradient(
            loss, default_layer.trainable_variables
        )
        # All weights should receive gradient.
        var_names = [v.name for v in default_layer.trainable_variables]
        assert len(grads) == len(default_layer.trainable_variables)
        for g, name in zip(grads, var_names):
            assert g is not None, f"None grad for {name}"
            # Some grads may be ~0 in pathological cases but for random
            # init + random input they should be non-zero.
            assert float(tf.reduce_sum(tf.abs(g)).numpy()) > 0.0, f"Zero grad for {name}"

    # ==================== Serialization ====================

    def test_get_config_roundtrip(self):
        layer = LighthouseAttention(
            dim=128,
            num_heads=4,
            num_levels=3,
            pooling_factor=4,
            top_k=20,
            full_attention=False,
            normalization_type="rms_norm",
            dropout_rate=0.0,
        )
        config = layer.get_config()
        restored = LighthouseAttention.from_config(config)
        cfg2 = restored.get_config()
        # Initializer/regularizer serialize as dicts — compare keys we care about.
        for k in (
            "dim",
            "num_heads",
            "head_dim",
            "num_levels",
            "pooling_factor",
            "top_k",
            "scorer",
            "full_attention",
            "normalization_type",
            "use_bias",
            "dropout_rate",
        ):
            assert config[k] == cfg2[k], f"config mismatch at {k}"

    def test_save_load_keras_roundtrip(self):
        layer = LighthouseAttention(
            dim=64,
            num_heads=4,
            num_levels=2,
            pooling_factor=2,
            top_k=10,
        )
        inp = keras.Input(shape=(16, 64))
        model = keras.Model(inp, layer(inp))
        x = tf.random.normal([2, 16, 64])
        y_pre = model(x).numpy()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lh.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y_post = loaded(x).numpy()
        np.testing.assert_allclose(y_pre, y_post, atol=1e-5)
