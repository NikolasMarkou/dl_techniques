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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
