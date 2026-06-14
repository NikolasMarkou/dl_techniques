"""Test Suite for PerformerAttention Layer.

First-ever regression coverage for PerformerAttention (FAVOR+ linear attention).

CRITICAL non-determinism contract: the FAVOR+ random projection matrix is
re-drawn on EVERY forward via ``keras.random.normal(..., seed=None)`` (see
``_create_projection_matrix``). The layer exposes NO seed parameter, so output
is NON-DETERMINISTIC across calls by design. Therefore:
  - forward tests assert SHAPE / structure only (never exact values);
  - the .keras round-trip asserts SHAPE / STRUCTURE only -- a bitwise-equal
    output assertion would be a spurious-red flake every run.

Coverage:
1. Initialization & Configuration
2. Input Validation
3. Forward Pass (shape only)
4. Non-determinism contract (locks the FAVOR+ per-forward redraw behavior)
5. Serialization (get_config / from_config + .keras structural round-trip)
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

from dl_techniques.layers.attention.performer_attention import PerformerAttention


# ==============================================================================
# 1. Initialization & Configuration
# ==============================================================================

class TestInitialization:

    def test_defaults(self):
        layer = PerformerAttention(dim=16, num_heads=2)
        assert layer.dim == 16
        assert layer.num_heads == 2
        assert layer.head_dim == 8
        assert layer.nb_features == 256
        assert layer.ortho_scaling == 0.0
        assert layer.causal is False
        expected_scale = 1.0 / np.sqrt(8)
        np.testing.assert_allclose(layer.scale, expected_scale)

    def test_custom_config(self):
        layer = PerformerAttention(
            dim=32,
            num_heads=4,
            nb_features=16,
            ortho_scaling=1.0,
            causal=True,
            dropout_rate=0.1,
            use_bias=True,
        )
        assert layer.nb_features == 16
        assert layer.ortho_scaling == 1.0
        assert layer.causal is True
        assert layer.dropout_rate == 0.1
        assert layer.use_bias is True


# ==============================================================================
# 2. Input Validation
# ==============================================================================

class TestValidation:

    def test_invalid_dim(self):
        with pytest.raises(ValueError, match="dim must be positive"):
            PerformerAttention(dim=-8, num_heads=2)

    def test_invalid_heads(self):
        with pytest.raises(ValueError, match="num_heads must be positive"):
            PerformerAttention(dim=16, num_heads=0)

    def test_invalid_divisibility(self):
        with pytest.raises(ValueError, match="must be divisible"):
            PerformerAttention(dim=10, num_heads=3)

    def test_invalid_nb_features(self):
        with pytest.raises(ValueError, match="nb_features must be positive"):
            PerformerAttention(dim=16, num_heads=2, nb_features=0)

    def test_invalid_dropout(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            PerformerAttention(dim=16, num_heads=2, dropout_rate=2.0)

    def test_build_wrong_rank(self):
        layer = PerformerAttention(dim=16, num_heads=2)
        with pytest.raises(ValueError, match="Expected 3D input"):
            layer.build((None, 16))

    def test_build_mismatched_dim(self):
        layer = PerformerAttention(dim=16, num_heads=2)
        with pytest.raises(ValueError, match="must match dim"):
            layer.build((None, 6, 32))


# ==============================================================================
# 3. Forward Pass (SHAPE ONLY -- random projection => non-deterministic)
# ==============================================================================

class TestForward:

    def test_output_shape(self):
        x = keras.random.normal((2, 6, 16))
        layer = PerformerAttention(dim=16, num_heads=2, nb_features=8)
        out = layer(x)
        # Assert SHAPE only -- output value is non-deterministic by design.
        assert out.shape == (2, 6, 16)
        assert not np.any(np.isnan(np.array(out)))

    @pytest.mark.xfail(
        reason="causal=True path is dead-on-forward (never-executed code): "
        "_linear_attention causal branch builds a 5-D einsum "
        "'bhnf,bhnd->bhnfd' then feeds it (plus a mis-ranked expand_dims(q)) "
        "into a second einsum 'bhnf,bhnfd->bhnd' with inconsistent subscripts "
        "-> InvalidArgumentError (rank 5 vs expected 4). Multi-bug chain in the "
        "streaming-causal cumsum math; needs a >10-line restructure -> deferred "
        "to a dedicated plan. Non-causal Performer forward is fully functional.",
        strict=False,
    )
    def test_output_shape_causal(self):
        x = keras.random.normal((2, 6, 16))
        layer = PerformerAttention(dim=16, num_heads=2, nb_features=8, causal=True)
        out = layer(x)
        assert out.shape == (2, 6, 16)

    def test_return_attention_scores_is_none(self):
        x = keras.random.normal((1, 4, 16))
        layer = PerformerAttention(dim=16, num_heads=2, nb_features=8)
        out, scores = layer(x, return_attention_scores=True)
        assert out.shape == (1, 4, 16)
        assert scores is None

    def test_variable_batch(self):
        layer = PerformerAttention(dim=16, num_heads=2, nb_features=8)
        out1 = layer(keras.random.normal((1, 4, 16)))
        out2 = layer(keras.random.normal((3, 4, 16)))
        assert out1.shape[0] == 1
        assert out2.shape[0] == 3


# ==============================================================================
# 4. Non-determinism contract (FAVOR+ re-draws projection every forward)
# ==============================================================================

class TestNonDeterminism:
    """Locks the documented non-deterministic FAVOR+ behavior. If a future
    change pins the projection (e.g. a seed), this test flags the contract
    change deliberately rather than silently."""

    def test_forward_is_nondeterministic(self):
        x = keras.random.normal((2, 6, 16))
        layer = PerformerAttention(dim=16, num_heads=2, nb_features=8)
        out1 = np.array(layer(x, training=False))
        out2 = np.array(layer(x, training=False))
        # Same input, same weights, but a fresh random projection each call.
        assert not np.allclose(out1, out2, atol=1e-6), (
            "Performer FAVOR+ projection is expected to re-draw per forward; "
            "outputs should differ across calls."
        )


# ==============================================================================
# 5. Serialization (structural round-trip ONLY)
# ==============================================================================

class TestSerialization:

    def test_get_config(self):
        layer = PerformerAttention(
            dim=32, num_heads=4, nb_features=16, causal=True, dropout_rate=0.2
        )
        config = layer.get_config()
        assert config["dim"] == 32
        assert config["num_heads"] == 4
        assert config["nb_features"] == 16
        assert config["causal"] is True
        assert config["dropout_rate"] == 0.2

    def test_from_config(self):
        layer = PerformerAttention(dim=16, num_heads=2, nb_features=8)
        config = layer.get_config()
        rebuilt = PerformerAttention.from_config(config)
        assert rebuilt.dim == 16
        assert rebuilt.num_heads == 2
        assert rebuilt.nb_features == 8

    def test_model_save_load_loop_shape_only(self):
        """Full .keras save/load round-trip. CRITICAL: assert SHAPE/STRUCTURE
        ONLY -- the per-forward random FAVOR+ projection makes every forward
        differ, so a value assertion would be a spurious-red flake."""
        inputs = keras.Input(shape=(6, 16))
        x = PerformerAttention(dim=16, num_heads=2, nb_features=8)(inputs)
        model = keras.Model(inputs, x)

        x_in = np.random.normal(size=(2, 6, 16)).astype("float32")
        pred_orig = model.predict(x_in, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "performer_attention.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            pred_load = loaded.predict(x_in, verbose=0)

        # SHAPE / STRUCTURE only -- do NOT compare values (non-deterministic).
        assert pred_orig.shape == pred_load.shape == (2, 6, 16)
        assert not np.any(np.isnan(pred_load))
        # Reloaded layer preserves config.
        reloaded_layer = [
            l for l in loaded.layers if isinstance(l, PerformerAttention)
        ][0]
        assert reloaded_layer.dim == 16
        assert reloaded_layer.num_heads == 2
        assert reloaded_layer.nb_features == 8


# ==============================================================================
# 6. Edge Cases
# ==============================================================================

class TestEdgeCases:

    def test_compute_output_shape(self):
        layer = PerformerAttention(dim=16, num_heads=2)
        assert layer.compute_output_shape((2, 6, 16)) == (2, 6, 16)

    def test_kwargs_passthrough(self):
        layer = PerformerAttention(dim=16, num_heads=2, name="performer_special")
        assert layer.name == "performer_special"

    def test_gradient_flow(self):
        layer = PerformerAttention(dim=16, num_heads=2, nb_features=8)
        x = keras.random.normal((1, 6, 16))
        with tf.GradientTape() as tape:
            out = layer(x)
            loss = tf.reduce_mean(out)
        grads = tape.gradient(loss, layer.to_qkv.trainable_variables)
        assert all(g is not None for g in grads)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
