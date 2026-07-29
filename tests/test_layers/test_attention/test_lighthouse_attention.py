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
        # None, not "rms_norm". The class documents this as deliberate
        # (D-004(a)): the scorer ranks ||Q|| and ||K||, and RMSNorm makes both
        # near-constant across positions, erasing the selection signal. The
        # factory registry used to declare "rms_norm" here and silently applied
        # it to every factory-built layer; that disagreement is now fixed in
        # `attention/factory.py`, in the constructor's favour.
        assert layer.qk_norm_type is None
        assert layer.probability_type == "softmax"

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
            qk_norm_type="rms_norm",
            probability_type="softmax",
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
            "qk_norm_type",
            "probability_type",
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

    # ==================== Causality: the exact guarantee ====================

    @staticmethod
    def _perturb(perturb_at, N=32, dim=64, H=4, L=2, p=2, k=16):
        """Return (block_span, positions whose output moved) for a one-token hit."""
        keras.utils.set_random_seed(7)
        layer = LighthouseAttention(
            dim=dim, num_heads=H, num_levels=L, pooling_factor=p, top_k=k
        )
        inp = keras.Input(shape=(N, dim))
        m = keras.Model(inp, layer(inp))
        x = tf.random.normal([1, N, dim], seed=99)
        y0 = m(x).numpy()
        xp = x.numpy().copy()
        xp[:, perturb_at, :] += 100.0
        y1 = m(tf.constant(xp)).numpy()
        d = np.abs(y0 - y1).max(axis=-1)[0]
        return layer._sel_block_span, np.nonzero(d > 1e-5)[0], d

    def test_causality_no_cross_block_leakage(self):
        """The guarantee D-023 actually buys: no leak into an EARLIER block.

        `test_causality` checks one perturbation (the last token) against one
        prefix (`i < N/2`). That single point passed even on some broken
        variants during development, so this sweeps EVERY perturbation position
        and asserts the general property: a perturbation at token T may move
        outputs in T's own causal block or later, never in a block before it.

        This is the assertion that goes red if anyone reinstates a global
        `ops.top_k` over all candidates — measured on the pre-fix bytes,
        perturbing token 31 evicted pyramid entry 15 and moved output 15 by
        2.585, eight blocks earlier.
        """
        leaks = []
        for t in range(4, 32):
            span, changed, d = self._perturb(t)
            cross = [int(i) for i in changed if i // span < t // span]
            if cross:
                leaks.append((t, cross, float(d[cross].max())))
        assert not leaks, (
            "a perturbation leaked into an earlier causal block: "
            + "; ".join(
                f"token {t} (block {t // span}) moved positions {c} by up to "
                f"{m:.4f}"
                for t, c, m in leaks
            )
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "OPEN: causality is BLOCK-granular, not per-position. A "
            "perturbation at token T can still change the selection among "
            "entries whose causal_pos lies in T's own block (span p^(L-1)), so "
            "positions earlier in that same block can move — measured: perturb "
            "token 5 (block 2, span 2) moves position 4 by 2.5558. Per-position "
            "causality needs per-query selection + block-wise SDPA, a different "
            "layer shape (D-023). AN XPASS MEANS SOMEONE BUILT IT: remove this "
            "marker, rename the test, and update D-023 and the class docstring, "
            "which currently promise only the block-granular guarantee."
        ),
    )
    def test_causality_is_per_position(self):
        """Full per-position causality — the property this layer does NOT have.

        Asserts the CORRECT behaviour (nothing before T may move), so it xfails
        today and XPASS-fails the moment the residual is closed. Pinning the
        residual as an assertion instead would go red on the fix, which is
        backwards.
        """
        offenders = []
        for t in range(4, 32):
            _, changed, d = self._perturb(t)
            before = [int(i) for i in changed if i < t]
            if before:
                offenders.append((t, before, float(d[before].max())))
        assert not offenders, (
            "positions before the perturbed token moved: "
            + "; ".join(
                f"token {t} -> {b} by up to {m:.4f}" for t, b, m in offenders
            )
        )
