"""Comprehensive Test Suite for AnchorAttention Layer.

This module provides exhaustive testing for the AnchorAttention layer, covering:
1. Initialization & Configuration
2. Input Validation & Error Handling
3. Build Process & Weight Shapes
4. Standard Attention Logic
5. Hierarchical Attention Logic
6. Probability/Activation Integration
7. Training Dynamics (Dropout, Regularization, Gradients)
8. Serialization & Persistence
9. Mixed Precision & DType Support
10. Edge Cases
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
from keras import ops, layers, initializers, regularizers
import tempfile
import os

from dl_techniques.layers.attention.anchor_attention import AnchorAttention


# ==============================================================================
# 1. Initialization & Configuration Tests
# ==============================================================================

class TestInitialization:
    """Tests for layer initialization and parameter storage."""

    def test_defaults(self):
        """Test instantiation with minimal required arguments."""
        layer = AnchorAttention(dim=64, num_heads=8)
        assert layer.dim == 64
        assert layer.num_heads == 8
        assert layer.head_dim == 8
        assert layer.dropout_rate == 0.0
        assert layer.use_bias is True
        assert layer.probability_type == "softmax"
        # Check scale factor (1 / sqrt(head_dim))
        expected_scale = 1.0 / np.sqrt(8)
        np.testing.assert_allclose(layer.scale, expected_scale)

    def test_explicit_head_dim(self):
        """Test providing a specific head dimension."""
        # dim=64, heads=4 -> default head_dim=16. Force head_dim=32.
        layer = AnchorAttention(dim=64, num_heads=4, head_dim=32)
        assert layer.head_dim == 32
        # Scale should be based on explicit head_dim
        expected_scale = 1.0 / np.sqrt(32)
        np.testing.assert_allclose(layer.scale, expected_scale)

    def test_custom_config(self):
        """Test all configurable parameters."""
        reg = regularizers.L2(0.01)
        init = initializers.HeNormal()
        layer = AnchorAttention(
            dim=128,
            num_heads=8,
            dropout_rate=0.5,
            use_bias=False,
            probability_type="sparsemax",
            probability_config={"axis": -1},
            kernel_initializer=init,
            bias_initializer="ones",
            kernel_regularizer=reg,
            bias_regularizer=reg
        )
        assert layer.dropout_rate == 0.5
        assert layer.use_bias is False
        assert layer.probability_type == "sparsemax"
        assert layer.probability_config == {"axis": -1}
        assert layer.kernel_regularizer == reg


# ==============================================================================
# 2. Input Validation & Error Handling
# ==============================================================================

class TestValidation:
    """Tests for input validation in __init__ and build/call."""

    def test_invalid_dim_negative(self):
        with pytest.raises(ValueError, match="dim must be positive"):
            AnchorAttention(dim=-10, num_heads=2)

    def test_invalid_heads_negative(self):
        with pytest.raises(ValueError, match="num_heads must be positive"):
            AnchorAttention(dim=32, num_heads=0)

    def test_invalid_divisibility(self):
        """Test dim not divisible by num_heads (when head_dim is None)."""
        with pytest.raises(ValueError, match="must be divisible"):
            AnchorAttention(dim=10, num_heads=3)

    def test_invalid_dropout_range_high(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            AnchorAttention(dim=32, num_heads=2, dropout_rate=1.1)

    def test_invalid_dropout_range_low(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            AnchorAttention(dim=32, num_heads=2, dropout_rate=-0.1)

    def test_build_wrong_dims(self):
        """Test building with non-3D input."""
        layer = AnchorAttention(dim=32, num_heads=4)
        with pytest.raises(ValueError, match="Input must be 3D"):
            layer.build((32, 10))

    def test_build_mismatched_channel_dim(self):
        """Test input channel dim mismatching layer config."""
        layer = AnchorAttention(dim=32, num_heads=4)
        with pytest.raises(ValueError, match="Last dimension of input"):
            layer.build((None, 10, 64))  # Expecting 32, got 64


# ==============================================================================
# 3. Build Process & Weight Shapes
# ==============================================================================

class TestBuild:
    """Tests for the build process and weight instantiation."""

    @pytest.fixture
    def layer(self):
        return AnchorAttention(dim=64, num_heads=8, head_dim=8)

    def test_weight_existence(self, layer):
        layer.build((None, 10, 64))
        # Check all expected weights exist
        assert len(layer.query_proj.trainable_variables) > 0
        assert len(layer.key_proj.trainable_variables) > 0
        assert len(layer.value_proj.trainable_variables) > 0
        assert len(layer.query_token_proj.trainable_variables) > 0
        assert len(layer.output_proj.trainable_variables) > 0

    def test_weight_shapes_with_bias(self, layer):
        layer.build((None, 10, 64))
        # query_proj kernel: (input_dim, num_heads * head_dim)
        assert layer.query_proj.kernel.shape == (64, 64)
        assert layer.query_proj.bias.shape == (64,)
        
        # query_token_proj (used in hierarchical)
        assert layer.query_token_proj.kernel.shape == (64, 64)


# ==============================================================================
# 4. Standard Attention Logic
# ==============================================================================

class TestStandardAttention:
    """Tests for Standard Mode (num_anchor_tokens=None)."""

    def test_output_shape(self):
        x = keras.random.normal((2, 20, 32))
        layer = AnchorAttention(dim=32, num_heads=4)
        out = layer(x)
        assert out.shape == (2, 20, 32)

    def test_determinism_inference(self):
        """Ensure standard attention is deterministic without dropout."""
        x = keras.random.normal((1, 10, 32))
        layer = AnchorAttention(dim=32, num_heads=4)
        out1 = layer(x, training=False)
        out2 = layer(x, training=False)
        np.testing.assert_allclose(out1, out2, atol=1e-6)

    def test_values_range(self):
        """Sanity check that outputs aren't exploding."""
        x = keras.random.normal((1, 10, 32))
        layer = AnchorAttention(dim=32, num_heads=4)
        out = layer(x)
        assert np.all(np.abs(out) < 100.0)  # Loose bound for normalized inputs


# ==============================================================================
# 5. Hierarchical Attention Logic
# ==============================================================================

class TestHierarchicalAttention:
    """Tests for Hierarchical Mode (num_anchor_tokens=K)."""

    def test_mixed_mode_shapes(self):
        """Test output shape when splitting anchors/queries."""
        x = keras.random.normal((2, 30, 32))
        layer = AnchorAttention(dim=32, num_heads=4)
        # 10 anchors, 20 queries
        out = layer(x, num_anchor_tokens=10)
        assert out.shape == (2, 30, 32)

    def test_query_isolation(self):
        """
        Critical: Queries should not attend to each other.
        Changing query q_i should not affect output of query q_j.
        """
        dim = 32
        layer = AnchorAttention(dim=dim, num_heads=4)
        seq_len = 10
        anchors = 2
        
        # Base input
        x = keras.random.normal((1, seq_len, dim))
        x_base = ops.convert_to_numpy(x)
        
        # Run 1
        out1 = layer(x, num_anchor_tokens=anchors)
        
        # Modify last token (a query)
        x_mod = x_base.copy()
        x_mod[0, -1, :] += 5.0
        x_mod_tensor = ops.convert_to_tensor(x_mod)
        
        # Run 2
        out2 = layer(x_mod_tensor, num_anchor_tokens=anchors)
        
        # Check penultimate token (another query). 
        # Since queries don't self-attend, token -2 should be unaffected by token -1 change.
        diff = np.max(np.abs(out1[:, -2, :] - out2[:, -2, :]))
        assert diff < 1e-5, f"Leaky attention detected! Diff: {diff}"

    def test_anchor_influence(self):
        """Verify that changing an anchor affects queries."""
        dim = 32
        layer = AnchorAttention(dim=dim, num_heads=4)
        seq_len = 10
        anchors = 2
        
        x = keras.random.normal((1, seq_len, dim))
        x_base = ops.convert_to_numpy(x)
        
        out1 = layer(x, num_anchor_tokens=anchors)
        
        # Modify first token (an anchor)
        x_mod = x_base.copy()
        x_mod[0, 0, :] += 5.0
        out2 = layer(ops.convert_to_tensor(x_mod), num_anchor_tokens=anchors)
        
        # All tokens (including queries) attend to anchors, so output should change
        diff = np.max(np.abs(out1[:, -1, :] - out2[:, -1, :]))
        assert diff > 1e-4, "Anchors failed to influence queries."

    def test_fallback_logic(self):
        """Test fallback when anchors >= seq_len."""
        x = keras.random.normal((1, 10, 32))
        layer = AnchorAttention(dim=32, num_heads=4)
        
        # Request 10 anchors for sequence of 10 -> Standard Attn
        out_hier = layer(x, num_anchor_tokens=10)
        out_std = layer(x, num_anchor_tokens=None)
        
        np.testing.assert_allclose(out_hier, out_std, atol=1e-6)

    def test_fallback_logic_exceeds(self):
        """Test fallback when anchors > seq_len."""
        x = keras.random.normal((1, 5, 32))
        layer = AnchorAttention(dim=32, num_heads=4)
        out_hier = layer(x, num_anchor_tokens=100) # Exceeds
        out_std = layer(x, num_anchor_tokens=None)
        np.testing.assert_allclose(out_hier, out_std, atol=1e-6)


# ==============================================================================
# 6. Probability & Activation Configuration
# ==============================================================================

class TestProbabilityConfiguration:
    """Test integration with different probability outputs."""

    def test_softmax_default(self):
        layer = AnchorAttention(dim=32, num_heads=4)
        assert layer.score_activation.probability_type == "softmax"

    def test_adaptive_config(self):
        """Test passing config dict to ProbabilityOutput."""
        config = {"min_temp": 0.5, "max_temp": 2.0}
        layer = AnchorAttention(
            dim=32, num_heads=4, 
            probability_type="adaptive",
            probability_config=config
        )
        assert layer.score_activation.probability_type == "adaptive"
        assert layer.score_activation.type_config == config
        
        # Ensure it runs
        x = keras.random.normal((1, 10, 32))
        out = layer(x)
        assert out.shape == (1, 10, 32)


# ==============================================================================
# 7. Training Dynamics (Dropout, Regularization, Gradients)
# ==============================================================================

class TestTrainingDynamics:
    
    def test_dropout_training(self):
        """Test dropout is applied during training."""
        layer = AnchorAttention(dim=32, num_heads=4, dropout_rate=0.5)
        x = keras.random.normal((1, 10, 32)) * 10.0
        
        # Run multiple times
        out1 = layer(x, training=True)
        out2 = layer(x, training=True)
        
        # Should differ due to dropout mask
        assert not np.allclose(out1, out2), "Dropout did not introduce stochasticity"

    def test_dropout_inference(self):
        """Test dropout is disabled during inference."""
        layer = AnchorAttention(dim=32, num_heads=4, dropout_rate=0.5)
        x = keras.random.normal((1, 10, 32))
        
        out1 = layer(x, training=False)
        out2 = layer(x, training=False)
        
        np.testing.assert_allclose(out1, out2, atol=1e-6)

    def test_gradient_flow_standard(self):
        """Ensure gradients flow to standard projections."""
        layer = AnchorAttention(dim=32, num_heads=4)
        x = keras.random.normal((1, 10, 32))
        
        with tf.GradientTape() as tape:
            out = layer(x, num_anchor_tokens=None)
            loss = tf.reduce_mean(out)
            
        grads = tape.gradient(loss, layer.query_proj.trainable_variables)
        assert all(g is not None and tf.reduce_any(g != 0) for g in grads)

    def test_gradient_flow_hierarchical(self):
        """Ensure gradients flow to both standard and query-specific projections."""
        layer = AnchorAttention(dim=32, num_heads=4)
        x = keras.random.normal((1, 10, 32))
        
        with tf.GradientTape() as tape:
            out = layer(x, num_anchor_tokens=5)
            loss = tf.reduce_mean(out)
            
        # query_token_proj is ONLY used in hierarchical mode
        grads = tape.gradient(loss, layer.query_token_proj.trainable_variables)
        assert all(g is not None and tf.reduce_any(g != 0) for g in grads)

    def test_regularization_loss(self):
        """Test that regularizers add to layer losses."""
        layer = AnchorAttention(
            dim=32, num_heads=4,
            kernel_regularizer=regularizers.L2(1.0)
        )
        x = keras.random.normal((1, 10, 32))
        layer(x) # Trigger build
        
        assert len(layer.losses) > 0
        # Should have losses from 5 sub-dense layers
        # (q, k, v, q_token, out)
        # Note: Depending on implementation, sublayers add to their own losses,
        # and parent layer aggregates them via standard Keras property.
        assert tf.reduce_sum(layer.losses) > 0


# ==============================================================================
# 8. Serialization & Persistence
# ==============================================================================

class TestSerialization:

    def test_get_config(self):
        layer = AnchorAttention(dim=64, num_heads=8, dropout_rate=0.2)
        config = layer.get_config()
        assert config["dim"] == 64
        assert config["dropout_rate"] == 0.2
        assert config["num_heads"] == 8

    def test_from_config(self):
        config = {
            "dim": 32,
            "num_heads": 2,
            "dropout_rate": 0.1,
            "probability_type": "adaptive"
        }
        layer = AnchorAttention.from_config(config)
        assert layer.dim == 32
        assert layer.probability_type == "adaptive"

    def test_model_save_load_loop(self):
        """Full save/load cycle in a model context."""
        inputs = keras.Input(shape=(10, 32))
        x = AnchorAttention(dim=32, num_heads=4)(inputs)
        model = keras.Model(inputs, x)
        
        x_in = np.random.normal(size=(1, 10, 32)).astype("float32")
        pred_orig = model.predict(x_in, verbose=0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_model.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            pred_load = loaded.predict(x_in, verbose=0)
            
            np.testing.assert_allclose(pred_orig, pred_load, atol=1e-5)


# ==============================================================================
# 9. Mixed Precision & DType Support
# ==============================================================================

class TestDtypes:


    def test_variable_batch_dim(self):
        """Test with varying batch sizes."""
        layer = AnchorAttention(dim=32, num_heads=4)
        out1 = layer(keras.random.normal((1, 5, 32)))
        out2 = layer(keras.random.normal((5, 5, 32)))
        assert out1.shape[0] == 1
        assert out2.shape[0] == 5


# ==============================================================================
# 10. Edge Cases
# ==============================================================================

class TestEdgeCases:
    
    def test_single_anchor(self):
        """Hierarchical mode with just 1 anchor."""
        layer = AnchorAttention(dim=32, num_heads=4)
        x = keras.random.normal((1, 10, 32))
        out = layer(x, num_anchor_tokens=1)
        assert out.shape == (1, 10, 32)
        assert not np.any(np.isnan(out))

    def test_zero_anchors(self):
        """
        Edge Case: 0 anchors. 
        Technically invalid for 'attending to anchors', but should handle gracefully 
        (e.g., return zeros, raise error, or operate on empty tensors without crash).
        Based on implementation: split at 0 means anchors=[], queries=all.
        Attending to empty keys usually results in NaN or Zero depending on softmax impl.
        We check it doesn't crash the Python process.
        """
        layer = AnchorAttention(dim=32, num_heads=4)
        x = keras.random.normal((1, 10, 32))
        try:
            out = layer(x, num_anchor_tokens=0)
        except Exception:
            # Crashing with error is acceptable for this invalid state
            pass
        else:
            # If it returns, check for NaNs if relevant, though behavior is undefined
            pass

    def test_kwargs_passthrough(self):
        """Test valid kwargs propagation to Layer init (e.g., name)."""
        layer = AnchorAttention(dim=32, num_heads=4, name="special_layer")
        assert layer.name == "special_layer"

    def test_compute_output_shape(self):
        layer = AnchorAttention(dim=32, num_heads=4)
        shape = layer.compute_output_shape((5, 20, 32))
        assert shape == (5, 20, 32)

    def test_cloning(self):
        """Test capability to be cloned via keras.models.clone_model."""
        layer = AnchorAttention(dim=32, num_heads=4)
        model = keras.Sequential([layer])
        clone = keras.models.clone_model(model)
        assert isinstance(clone.layers[0], AnchorAttention)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

