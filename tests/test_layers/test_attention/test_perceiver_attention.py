"""Test suite for PerceiverAttention layer.

First-ever regression coverage for PerceiverAttention, the asymmetric
cross-attention block from the Perceiver / Perceiver IO architectures. It wraps
``MultiHeadCrossAttention`` so a small latent query array attends to a large
data (key/value) array. Covers:
1. Initialization & Configuration
2. Forward pass (asymmetric: distinct query / kv sequence lengths)
3. Self-attention mode (kv_input=None)
4. get_config / from_config round-trip
5. Full `.keras` model save/load round-trip (functional API; single tensor out)
"""

import os
import tempfile

import pytest
import numpy as np
import tensorflow as tf
import keras

from dl_techniques.layers.attention.perceiver_attention import PerceiverAttention


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def minimal_config():
    return {"dim": 32, "num_heads": 4}


@pytest.fixture
def query_input():
    """Small latent query array: [B, Q_seq, dim]."""
    return keras.random.normal((2, 8, 32))


@pytest.fixture
def kv_input():
    """Large data array (keys/values): [B, KV_seq, dim] — distinct seq len."""
    return keras.random.normal((2, 40, 32))


# ==============================================================================
# 1. Initialization & Configuration
# ==============================================================================

class TestInitialization:

    def test_defaults(self, minimal_config):
        layer = PerceiverAttention(**minimal_config)
        assert layer.dim == 32
        assert layer.num_heads == 4
        assert layer.dropout_rate == 0.0
        assert layer.use_bias is True
        assert layer.cross_attention is not None

    def test_invalid_dim_negative(self):
        with pytest.raises(ValueError, match="dim must be positive"):
            PerceiverAttention(dim=-10, num_heads=2)

    def test_invalid_heads_negative(self):
        with pytest.raises(ValueError, match="num_heads must be positive"):
            PerceiverAttention(dim=32, num_heads=0)

    def test_invalid_divisibility(self):
        with pytest.raises(ValueError, match="must be divisible"):
            PerceiverAttention(dim=30, num_heads=4)

    def test_invalid_dropout_range(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            PerceiverAttention(dim=32, num_heads=4, dropout_rate=1.5)


# ==============================================================================
# 2. Forward Pass (asymmetric cross-attention)
# ==============================================================================

class TestForwardPass:

    def test_asymmetric_output_shape(self, minimal_config, query_input, kv_input):
        """Output sequence length matches the QUERY length, not the KV length."""
        layer = PerceiverAttention(**minimal_config)
        out = layer(query_input, kv_input=kv_input)
        # [B, Q_seq, dim] — bottleneck preserves the small latent length (8).
        assert tuple(out.shape) == (2, 8, 32)

    def test_self_attention_mode(self, minimal_config, query_input):
        """kv_input=None falls back to self-attention over the query array."""
        layer = PerceiverAttention(**minimal_config)
        out = layer(query_input)
        assert tuple(out.shape) == (2, 8, 32)

    def test_forward_no_nans(self, minimal_config, query_input, kv_input):
        layer = PerceiverAttention(**minimal_config)
        out = layer(query_input, kv_input=kv_input)
        assert not np.any(np.isnan(np.asarray(out)))

    def test_determinism_inference(self, minimal_config, query_input, kv_input):
        layer = PerceiverAttention(**minimal_config)
        out1 = layer(query_input, kv_input=kv_input, training=False)
        out2 = layer(query_input, kv_input=kv_input, training=False)
        np.testing.assert_allclose(
            np.asarray(out1), np.asarray(out2), atol=1e-6
        )

    def test_gradient_flow(self, minimal_config, query_input, kv_input):
        layer = PerceiverAttention(**minimal_config)
        with tf.GradientTape() as tape:
            out = layer(query_input, kv_input=kv_input)
            loss = tf.reduce_mean(out)
        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(grads) > 0
        assert all(g is not None for g in grads)


# ==============================================================================
# 3. Serialization & Persistence
# ==============================================================================

class TestSerialization:

    def test_get_config(self, minimal_config):
        layer = PerceiverAttention(**minimal_config, dropout_rate=0.2)
        config = layer.get_config()
        assert config["dim"] == 32
        assert config["num_heads"] == 4
        assert config["dropout_rate"] == 0.2

    def test_from_config(self, minimal_config):
        original = PerceiverAttention(**minimal_config)
        rebuilt = PerceiverAttention.from_config(original.get_config())
        assert rebuilt.dim == original.dim
        assert rebuilt.num_heads == original.num_heads
        assert rebuilt.dropout_rate == original.dropout_rate

    def test_model_save_load_loop(self, minimal_config):
        """Full `.keras` round-trip via the functional API (two-input model)."""
        query_in = keras.Input(shape=(8, 32), name="query")
        kv_in = keras.Input(shape=(40, 32), name="kv")
        # Pass kv positionally so the functional API registers it as a second
        # input (the layer's call signature is call(query_input, kv_input, ...)).
        out = PerceiverAttention(**minimal_config)(query_in, kv_in)
        model = keras.Model([query_in, kv_in], out)

        q_data = np.random.normal(size=(2, 8, 32)).astype("float32")
        kv_data = np.random.normal(size=(2, 40, 32)).astype("float32")
        pred_orig = model.predict([q_data, kv_data], verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "perceiver_model.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            pred_load = loaded.predict([q_data, kv_data], verbose=0)

            np.testing.assert_allclose(pred_orig, pred_load, atol=1e-6)


# ==============================================================================
# 4. Misc
# ==============================================================================

class TestMisc:

    def test_compute_output_shape_single(self, minimal_config):
        layer = PerceiverAttention(**minimal_config)
        assert layer.compute_output_shape((2, 8, 32)) == (2, 8, 32)

    def test_compute_output_shape_list(self, minimal_config):
        layer = PerceiverAttention(**minimal_config)
        shape = layer.compute_output_shape([(2, 8, 32), (2, 40, 32)])
        assert shape == (2, 8, 32)

    def test_kwargs_passthrough(self, minimal_config):
        layer = PerceiverAttention(**minimal_config, name="perceiver_block")
        assert layer.name == "perceiver_block"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
