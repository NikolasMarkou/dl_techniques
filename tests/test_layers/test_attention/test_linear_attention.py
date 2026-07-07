"""Test Suite for LinearAttention Layer.

Regression coverage for ``LinearAttention`` (bias-free, degree-1-homogeneous,
O(N) linear / kernel attention — Miyasawa-compliant by construction).

DETERMINISM contract (contrast PerformerAttention): LinearAttention has NO
per-forward RNG. Its feature map is deterministic and Dropout is inert at
``training=False``. Therefore, unlike the Performer suite, this suite asserts
EXACT VALUE equality where relevant:
  - two ``training=False`` forwards on the same input are bitwise-identical;
  - the ``.keras`` save/load round-trip reproduces outputs within ``atol=1e-6``.

Coverage:
1. Initialization & Configuration + Input Validation
2. Forward Pass (shape, finiteness, determinism)
3. Bias-free static check (Miyasawa property 1)
4. Degree-1 homogeneity numeric probe (Miyasawa property 2 — the core gate)
5. Associativity correctness (O(N) associative == naive O(N^2) reference)
6. Serialization (get_config / from_config + .keras value round-trip)
7. Edge cases (N=1, all-zero token, gradient flow)
"""

import os
import tempfile

import pytest
import numpy as np
import tensorflow as tf
import keras

from dl_techniques.layers.attention.linear_attention import LinearAttention

# Disable TF32 tensor-core matmul so GPU einsum uses true fp32. Without this the
# associativity test compares a TF32 GPU contraction (~1e-3 rel precision) against
# a full-precision numpy reference and spuriously fails at atol=1e-5. TF32 off makes
# the two contraction orders agree to genuine fp32 tolerance.
tf.config.experimental.enable_tensor_float_32_execution(False)


# ==============================================================================
# Test helpers
# ==============================================================================

def _feature_map_np(x: np.ndarray, feature_map: str) -> np.ndarray:
    """NumPy replica of ``LinearAttention._feature_map`` (for the naive reference)."""
    if feature_map == 'relu':
        return np.maximum(x, 0.0)
    if feature_map == 'relu_squared':
        return np.square(np.maximum(x, 0.0))
    # 'abs'
    return np.abs(x)


def _to_heads_np(t: np.ndarray, num_heads: int, head_dim: int) -> np.ndarray:
    """(B, N, inner) -> (B, H, N, head_dim), mirroring the layer's reshape/transpose."""
    b, n, _ = t.shape
    return t.reshape(b, n, num_heads, head_dim).transpose(0, 2, 1, 3)


# ==============================================================================
# 1. Initialization & Configuration
# ==============================================================================

class TestInitialization:

    def test_defaults(self):
        layer = LinearAttention(dim=16, num_heads=2)
        assert layer.dim == 16
        assert layer.num_heads == 2
        assert layer.head_dim == 8
        assert layer.inner_dim == 16
        assert layer.use_bias is False
        assert layer.feature_map == 'relu'
        assert layer.epsilon == 1e-6
        assert layer.dropout_rate == 0.0

    def test_custom_config(self):
        layer = LinearAttention(
            dim=32,
            num_heads=4,
            head_dim=10,
            feature_map='relu_squared',
            epsilon=1e-4,
            dropout_rate=0.1,
            use_bias=True,
        )
        assert layer.num_heads == 4
        assert layer.head_dim == 10
        assert layer.inner_dim == 40  # num_heads * head_dim when head_dim given
        assert layer.feature_map == 'relu_squared'
        assert layer.epsilon == 1e-4
        assert layer.dropout_rate == 0.1
        assert layer.use_bias is True


# ==============================================================================
# 2. Input Validation
# ==============================================================================

class TestValidation:

    def test_invalid_dim(self):
        with pytest.raises(ValueError, match="dim must be positive"):
            LinearAttention(dim=-8, num_heads=2)

    def test_invalid_dim_zero(self):
        with pytest.raises(ValueError, match="dim must be positive"):
            LinearAttention(dim=0, num_heads=2)

    def test_invalid_heads(self):
        with pytest.raises(ValueError, match="num_heads must be positive"):
            LinearAttention(dim=16, num_heads=0)

    def test_invalid_divisibility(self):
        with pytest.raises(ValueError, match="must be divisible"):
            LinearAttention(dim=10, num_heads=3)

    def test_invalid_dropout(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            LinearAttention(dim=16, num_heads=2, dropout_rate=2.0)

    def test_invalid_dropout_negative(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            LinearAttention(dim=16, num_heads=2, dropout_rate=-0.1)

    @pytest.mark.parametrize("bad_map", ['softmax', 'elu_plus_one', 'exp'])
    def test_forbidden_feature_map(self, bad_map):
        # Non-homogeneous maps must be rejected (they break degree-1).
        with pytest.raises(ValueError, match="FORBIDDEN"):
            LinearAttention(dim=16, num_heads=2, feature_map=bad_map)

    def test_unknown_feature_map(self):
        with pytest.raises(ValueError, match="feature_map must be one of"):
            LinearAttention(dim=16, num_heads=2, feature_map='gelu')

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError, match="epsilon must be"):
            LinearAttention(dim=16, num_heads=2, epsilon=-1.0)

    def test_build_wrong_rank(self):
        layer = LinearAttention(dim=16, num_heads=2)
        with pytest.raises(ValueError, match="Expected 3D input"):
            layer.build((None, 16))

    def test_build_mismatched_dim(self):
        layer = LinearAttention(dim=16, num_heads=2)
        with pytest.raises(ValueError, match="must match dim"):
            layer.build((None, 6, 32))


# ==============================================================================
# 3. Forward Pass (shape, finiteness, determinism)
# ==============================================================================

class TestForward:

    @pytest.mark.parametrize("b,n,dim,heads", [
        (2, 6, 16, 2),
        (3, 8, 32, 4),
        (1, 5, 24, 3),
    ])
    def test_output_shape_and_finite(self, b, n, dim, heads):
        x = keras.random.normal((b, n, dim))
        layer = LinearAttention(dim=dim, num_heads=heads)
        out = layer(x, training=False)
        assert out.shape == (b, n, dim)
        assert not np.any(np.isnan(np.array(out)))
        assert np.all(np.isfinite(np.array(out)))

    def test_deterministic_across_calls(self):
        """LinearAttention has NO per-forward RNG: two training=False forwards on
        the same input must be bitwise-identical (contrast Performer's FAVOR+)."""
        x = keras.random.normal((2, 6, 16))
        layer = LinearAttention(dim=16, num_heads=2)
        out1 = np.array(layer(x, training=False))
        out2 = np.array(layer(x, training=False))
        np.testing.assert_array_equal(
            out1, out2,
            err_msg="LinearAttention must be deterministic at training=False "
                    "(no per-forward random projection).",
        )


# ==============================================================================
# 4. Bias-free static check (Miyasawa property 1)
# ==============================================================================

class TestBiasFree:

    def test_no_bias_variables_when_use_bias_false(self):
        layer = LinearAttention(dim=16, num_heads=2, use_bias=False)
        layer(keras.random.normal((2, 6, 16)))  # build
        bias_count = sum('bias' in w.name for w in layer.weights)
        assert bias_count == 0, (
            f"bias-free layer must have zero bias variables, found {bias_count}: "
            f"{[w.name for w in layer.weights if 'bias' in w.name]}"
        )
        assert not any('bias' in w.name for w in layer.weights)

    def test_bias_present_when_use_bias_true(self):
        """Proves the bias-free check is real: use_bias=True DOES create bias vars."""
        layer = LinearAttention(dim=16, num_heads=2, use_bias=True)
        layer(keras.random.normal((2, 6, 16)))  # build
        bias_count = sum('bias' in w.name for w in layer.weights)
        assert bias_count > 0, "use_bias=True should create bias variables."


# ==============================================================================
# 5. Degree-1 homogeneity numeric probe (Miyasawa property 2 — the core gate)
# ==============================================================================

class TestHomogeneity:
    """f(alpha * x) == alpha * f(x) for alpha > 0 (degree-1 positive homogeneity).

    Standalone seq-shaped probe (NOT common.py's 4D image probe). Near-exact
    (target rel-err < 1e-4) because there are NO batch/EMA stats here — pure fp.
    The input-scaled epsilon (D-001) is what preserves this exactly. If this fails
    in (1e-4, 1e-2] the eps floor is the likely cause — DO NOT loosen; report.
    """

    @pytest.mark.parametrize("feature_map", ['relu', 'relu_squared', 'abs'])
    @pytest.mark.parametrize("alpha", [0.5, 2.0])
    def test_degree_one_homogeneity(self, feature_map, alpha):
        keras.utils.set_random_seed(42)
        dim = 16
        layer = LinearAttention(dim=dim, num_heads=2, feature_map=feature_map)

        x = np.random.uniform(-0.5, 0.5, size=(2, 8, dim)).astype('float32')
        f = np.array(layer(x, training=False))
        f_scaled = np.array(layer(alpha * x, training=False))

        target = alpha * f
        rel = np.max(np.abs(f_scaled - target)) / max(np.max(np.abs(target)), 1e-8)

        assert rel < 1e-4, (
            f"degree-1 homogeneity violated for feature_map={feature_map!r}, "
            f"alpha={alpha}: rel-err={rel:.3e} >= 1e-4. This is the Miyasawa "
            f"degree-1 gate (f(alpha*x) == alpha*f(x)); a non-trivial deviation "
            f"means the input-scaled eps (D-001) or an additive/normalizing op "
            f"broke degree-1. Do NOT loosen the tolerance — investigate."
        )


# ==============================================================================
# 6. Associativity correctness (O(N) associative == naive O(N^2) reference)
# ==============================================================================

class TestAssociativity:
    """Assert the layer's O(N) associative output equals a naive O(N^2) reference
    computed independently here. We reconstruct the FULL pipeline (including the
    layer's own projection weights) with an N x N attention matrix and compare to
    the layer's real output — an end-to-end check that the associative contraction
    order + normalizer genuinely compute normalized linear attention.
    """

    @pytest.mark.parametrize("feature_map", ['relu', 'relu_squared', 'abs'])
    def test_associative_equals_naive(self, feature_map):
        keras.utils.set_random_seed(7)
        b, n, dim, heads = 2, 7, 16, 2
        layer = LinearAttention(dim=dim, num_heads=heads, feature_map=feature_map)

        x = np.random.uniform(-1.0, 1.0, size=(b, n, dim)).astype('float32')
        out_layer = np.array(layer(x, training=False))  # builds the layer

        head_dim = layer.head_dim
        inner = layer.inner_dim
        eps = layer.epsilon

        # Pull the SAME projected q/k/v the layer uses (deterministic Dense).
        q = np.array(layer.query_proj(x))
        k = np.array(layer.key_proj(x))
        v = np.array(layer.value_proj(x))

        q = _to_heads_np(q, heads, head_dim)  # (B, H, N, d)
        k = _to_heads_np(k, heads, head_dim)
        v = _to_heads_np(v, heads, head_dim)

        phi_q = _feature_map_np(q, feature_map)
        phi_k = _feature_map_np(k, feature_map)

        # Naive O(N^2): materialize the (B, H, N, N) attention matrix.
        A = np.einsum('bhnd,bhmd->bhnm', phi_q, phi_k)   # phi_q @ phi_k^T
        num_ref = np.einsum('bhnm,bhmd->bhnd', A, v)      # (phi_q phi_k^T) V
        z_ref = A.sum(axis=-1)                            # (phi_q phi_k^T) . 1 -> (B,H,N)

        # Replicate the layer's input-scaled eps (D-001) exactly.
        z_mean = z_ref.mean(axis=-1, keepdims=True)       # (B,H,1)
        eps_eff = eps * z_mean
        denom = np.maximum(z_ref + eps_eff, 1e-20)
        core = num_ref / denom[..., None]                 # (B,H,N,d)

        # Merge heads and apply the layer's own output projection.
        core = core.transpose(0, 2, 1, 3).reshape(b, n, inner)
        out_ref = np.array(layer.output_proj(core))

        np.testing.assert_allclose(
            out_layer, out_ref, atol=1e-5,
            err_msg=f"O(N) associative output != naive O(N^2) reference for "
                    f"feature_map={feature_map!r}; the contraction order or the "
                    f"normalizer is wrong.",
        )


# ==============================================================================
# 7. Serialization (get_config / from_config + .keras value round-trip)
# ==============================================================================

class TestSerialization:

    def test_get_config_from_config(self):
        layer = LinearAttention(
            dim=32, num_heads=4, feature_map='relu_squared',
            epsilon=1e-4, dropout_rate=0.2, use_bias=True,
        )
        config = layer.get_config()
        assert config['dim'] == 32
        assert config['num_heads'] == 4
        assert config['feature_map'] == 'relu_squared'
        assert config['epsilon'] == 1e-4
        assert config['dropout_rate'] == 0.2
        assert config['use_bias'] is True

        rebuilt = LinearAttention.from_config(config)
        assert rebuilt.dim == 32
        assert rebuilt.num_heads == 4
        assert rebuilt.feature_map == 'relu_squared'
        assert rebuilt.epsilon == 1e-4
        assert rebuilt.use_bias is True

    def test_keras_save_load_value_roundtrip(self):
        """Full .keras save/load. LinearAttention is deterministic at
        training=False, so a VALUE assertion is valid (unlike Performer)."""
        inputs = keras.Input(shape=(6, 16))
        x = LinearAttention(dim=16, num_heads=2, feature_map='relu')(inputs)
        model = keras.Model(inputs, x)

        x_in = np.random.normal(size=(2, 6, 16)).astype("float32")
        pred_orig = model.predict(x_in, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "linear_attention.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            pred_load = loaded.predict(x_in, verbose=0)

        assert pred_orig.shape == pred_load.shape == (2, 6, 16)
        np.testing.assert_allclose(
            pred_orig, pred_load, atol=1e-6,
            err_msg="deterministic .keras round-trip must reproduce outputs.",
        )
        reloaded_layer = [
            l for l in loaded.layers if isinstance(l, LinearAttention)
        ][0]
        assert reloaded_layer.dim == 16
        assert reloaded_layer.num_heads == 2
        assert reloaded_layer.feature_map == 'relu'


# ==============================================================================
# 8. Edge Cases
# ==============================================================================

class TestEdgeCases:

    def test_single_token(self):
        """N=1: associativity and the normalizer must stay well-defined."""
        x = keras.random.normal((2, 1, 16))
        layer = LinearAttention(dim=16, num_heads=2)
        out = layer(x, training=False)
        assert out.shape == (2, 1, 16)
        assert np.all(np.isfinite(np.array(out)))

    def test_all_zero_input_is_finite(self):
        """All-zero token -> phi==0 -> 0/0 without the eps_eff / 1e-20 guard.
        Output must be finite (no NaN)."""
        x = np.zeros((2, 6, 16), dtype='float32')
        layer = LinearAttention(dim=16, num_heads=2)
        out = np.array(layer(x, training=False))
        assert not np.any(np.isnan(out))
        assert np.all(np.isfinite(out))

    def test_partial_dead_token(self):
        """One all-zero token mixed with real tokens must stay finite."""
        x = np.random.uniform(-0.5, 0.5, size=(2, 6, 16)).astype('float32')
        x[:, 0, :] = 0.0
        layer = LinearAttention(dim=16, num_heads=2, feature_map='relu')
        out = np.array(layer(x, training=False))
        assert np.all(np.isfinite(out))

    def test_compute_output_shape(self):
        layer = LinearAttention(dim=16, num_heads=2)
        assert layer.compute_output_shape((2, 6, 16)) == (2, 6, 16)

    def test_gradient_flow(self):
        layer = LinearAttention(dim=16, num_heads=2)
        x = keras.random.normal((1, 6, 16))
        with tf.GradientTape() as tape:
            out = layer(x)
            loss = tf.reduce_mean(out)
        grads = tape.gradient(loss, layer.query_proj.trainable_variables)
        assert len(grads) > 0
        assert all(g is not None for g in grads)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
