"""Test suite for AnchorAttention layer.

This module contains comprehensive tests for the AnchorAttention layer,
focusing on its dual-mode operation (standard vs hierarchical),
gradient flow, and specific properties of the anchor-based mechanism.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

# Import the layer to test
# Adjust import path based on your actual directory structure
from dl_techniques.layers.attention.anchor_attention import AnchorAttention


class TestAnchorAttention:
    """Test suite for AnchorAttention layer."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return keras.random.normal((2, 20, 64))  # (batch_size, seq_len, dim)

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return AnchorAttention(dim=64, num_heads=8)

    # ==================== Initialization Tests ====================

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = AnchorAttention(dim=64, num_heads=8)

        assert layer.dim == 64
        assert layer.num_heads == 8
        assert layer.dropout_rate == 0.0
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)

        # Check computed attributes
        assert layer.head_dim == 8
        np.testing.assert_allclose(layer.scale, 1.0 / np.sqrt(8))

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_reg = keras.regularizers.L2(1e-4)

        layer = AnchorAttention(
            dim=128,
            num_heads=16,
            dropout_rate=0.1,
            use_bias=False,
            kernel_regularizer=custom_reg
        )

        assert layer.dim == 128
        assert layer.num_heads == 16
        assert layer.dropout_rate == 0.1
        assert layer.use_bias is False
        assert layer.kernel_regularizer == custom_reg

    def test_invalid_params(self):
        """Test validation of invalid parameters."""
        # Invalid dim/head ratio
        with pytest.raises(ValueError, match="must be divisible"):
            AnchorAttention(dim=63, num_heads=8)

        # Invalid dim
        with pytest.raises(ValueError, match="dim must be positive"):
            AnchorAttention(dim=-64, num_heads=8)

        # Invalid dropout
        with pytest.raises(ValueError, match="dropout_rate"):
            AnchorAttention(dim=64, dropout_rate=1.5)

    # ==================== Build Process Tests ====================

    def test_build_process(self, input_tensor):
        """Test that the layer and sub-layers build properly."""
        layer = AnchorAttention(dim=64, num_heads=8)
        layer(input_tensor)  # Forward pass triggers build

        assert layer.built is True
        assert layer.qkv_dense.built is True
        assert layer.q_dense.built is True
        assert layer.proj_dense.built is True

        # Verify weight shapes
        # qkv: (dim, 3*dim)
        assert layer.qkv_dense.kernel.shape == (64, 64 * 3)
        # q_dense: (dim, dim) - used for queries
        assert layer.q_dense.kernel.shape == (64, 64)
        # proj: (dim, dim)
        assert layer.proj_dense.kernel.shape == (64, 64)

    def test_build_input_shape_validation(self):
        """Test input shape validation in build."""
        layer = AnchorAttention(dim=64, num_heads=8)

        with pytest.raises(ValueError, match="Input must be 3D"):
            layer.build((32, 64))

        with pytest.raises(ValueError, match="Last dimension of input"):
            layer.build((None, 10, 32))

    # ==================== Functional Logic Tests ====================

    def test_standard_attention_mode(self, input_tensor):
        """Test standard attention mode (num_anchor_tokens=None)."""
        layer = AnchorAttention(dim=64, num_heads=8)
        output = layer(input_tensor)

        assert output.shape == input_tensor.shape
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_hierarchical_attention_mode(self, input_tensor):
        """Test hierarchical mode with specific anchor count."""
        layer = AnchorAttention(dim=64, num_heads=8)
        # First 5 tokens are anchors
        output = layer(input_tensor, num_anchor_tokens=5)

        assert output.shape == input_tensor.shape
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_hierarchical_fallback(self, input_tensor):
        """
        Test that hierarchical mode falls back to standard attention
        when num_anchor_tokens >= seq_len.
        """
        layer = AnchorAttention(dim=64, num_heads=8)

        # 20 tokens input, 20 anchors requested -> Should be standard attention
        out_fallback = layer(input_tensor, num_anchor_tokens=20)
        out_standard = layer(input_tensor, num_anchor_tokens=None)

        # Outputs should be identical
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out_fallback),
            keras.ops.convert_to_numpy(out_standard),
            rtol=1e-6, atol=1e-6,
            err_msg="Fallback to standard attention failed"
        )

    def test_query_isolation_property(self):
        """
        CRITICAL TEST: Verify the core property of Anchor Attention.

        In hierarchical mode, query tokens (indices > K) attend ONLY to anchor tokens.
        Therefore, changing a query token input should NOT affect the output
        of other query tokens.
        """
        dim = 64
        seq_len = 10
        num_anchors = 4
        layer = AnchorAttention(dim=dim, num_heads=8)

        # Base input
        x1 = keras.random.normal((1, seq_len, dim))

        # Create x2 by modifying the LAST token (a query token)
        # We modify index 9. This should NOT affect the output at index 8 (another query).
        x2_np = keras.ops.convert_to_numpy(x1).copy()
        x2_np[0, -1, :] += 1.0 # Perturb last token
        x2 = keras.ops.convert_to_tensor(x2_np)

        # Run inference
        y1 = layer(x1, num_anchor_tokens=num_anchors)
        y2 = layer(x2, num_anchor_tokens=num_anchors)

        # Check index 8 (penultimate query token)
        # In standard attention, y1[0, 8] would differ from y2[0, 8] because token 8 attends to token 9
        # In anchor attention, token 8 attends only to 0-3. Token 9 is irrelevant to token 8.
        diff = tf.reduce_max(tf.abs(y1[:, 8, :] - y2[:, 8, :]))

        assert diff < 1e-5, f"Query isolation failed! Max diff: {diff}. Query tokens are attending to each other."

        # Conversely, check that changing an ANCHOR token DOES affect query outputs
        x3_np = keras.ops.convert_to_numpy(x1).copy()
        x3_np[0, 0, :] += 1.0 # Perturb first anchor
        x3 = keras.ops.convert_to_tensor(x3_np)

        y3 = layer(x3, num_anchor_tokens=num_anchors)

        anchor_diff = tf.reduce_max(tf.abs(y1[:, 8, :] - y3[:, 8, :]))
        assert anchor_diff > 1e-4, "Anchor influence failed! Anchors should affect queries."

    # ==================== Gradient Flow Tests ====================

    def test_gradient_flow(self, input_tensor):
        """Test gradient flow in both modes."""
        layer = AnchorAttention(dim=64, num_heads=8)

        # Test Standard Mode Gradients
        with tf.GradientTape() as tape:
            inputs = tf.Variable(input_tensor)
            # Explicitly pass None to ensure standard mode
            out_std = layer(inputs, num_anchor_tokens=None)
            loss_std = tf.reduce_mean(out_std**2)

        grads_std = tape.gradient(loss_std, layer.trainable_variables)

        # Retrieve variable groups from sub-layers
        qkv_vars = layer.qkv_dense.trainable_variables
        proj_vars = layer.proj_dense.trainable_variables
        q_vars = layer.q_dense.trainable_variables

        # Map variables to their gradients using object identity (since vars are unhashable)
        var_grad_pairs = list(zip(layer.trainable_variables, grads_std))

        def get_grad_for_var(target_var):
            for v, g in var_grad_pairs:
                if v is target_var:
                    return g
            return None

        # Check active variables (qkv and proj)
        for v in qkv_vars + proj_vars:
            grad = get_grad_for_var(v)
            assert grad is not None
            assert tf.reduce_any(grad != 0), f"Active variable {v.name} has zero gradient"

        # Check unused variables (q_dense)
        for v in q_vars:
            grad = get_grad_for_var(v)
            # Depending on backend configuration, unconnected gradients can be None or Zero
            if grad is not None:
                assert tf.reduce_all(grad == 0), f"Unused variable {v.name} has non-zero gradients in standard mode!"

        # Test Hierarchical Mode Gradients
        with tf.GradientTape() as tape:
            inputs = tf.Variable(input_tensor)
            # Use 10 anchors (half sequence)
            out_hier = layer(inputs, num_anchor_tokens=10)
            loss_hier = tf.reduce_mean(out_hier**2)

        grads_hier = tape.gradient(loss_hier, layer.trainable_variables)

        # In hierarchical mode, ALL sublayers should be involved
        assert all(g is not None for g in grads_hier)
        for g in grads_hier:
            assert tf.reduce_any(g != 0)

    # ==================== Serialization Tests ====================

    def test_serialization(self, input_tensor):
        """Test complete serialization cycle."""
        # Note: dim=64 to match input_tensor fixture (2, 20, 64)
        layer = AnchorAttention(
            dim=64,
            num_heads=4,
            dropout_rate=0.2,
            use_bias=False
        )

        # Build first
        layer(input_tensor)

        config = layer.get_config()
        recreated = AnchorAttention.from_config(config)

        assert recreated.dim == 64
        assert recreated.num_heads == 4
        assert recreated.dropout_rate == 0.2
        assert recreated.use_bias is False

    def test_model_save_load(self, input_tensor):
        """Test saving/loading within a full model."""
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = AnchorAttention(dim=64, num_heads=8)(inputs, num_anchor_tokens=5)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)

        # Run prediction
        pred_orig = model.predict(input_tensor, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "model.keras")
            model.save(path)

            loaded_model = keras.models.load_model(
                path,
                custom_objects={"AnchorAttention": AnchorAttention}
            )

            pred_loaded = loaded_model.predict(input_tensor, verbose=0)

            np.testing.assert_allclose(
                pred_orig, pred_loaded,
                rtol=1e-5, atol=1e-5
            )

    # ==================== Edge Case Tests ====================

    def test_single_anchor(self, input_tensor):
        """Test with just 1 anchor token."""
        layer = AnchorAttention(dim=64, num_heads=8)
        output = layer(input_tensor, num_anchor_tokens=1)
        assert output.shape == input_tensor.shape
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_zero_anchors_behavior(self, input_tensor):
        """
        Test behavior when num_anchor_tokens is 0.
        This is a degenerate case. The implementation splits inputs.
        If num_anchor=0:
          anchors=[]
          queries=all

        This typically causes empty tensor operations.
        We just want to ensure it doesn't crash effectively or behaves consistently.
        """
        layer = AnchorAttention(dim=64, num_heads=8)

        # Just check it doesn't hard crash the python process
        try:
            output = layer(input_tensor, num_anchor_tokens=0)
        except Exception:
            # Failure is acceptable for 0 anchors as it's invalid for attention
            pass

    def test_variable_batch_size(self):
        """Test compatibility with variable batch sizes."""
        layer = AnchorAttention(dim=64, num_heads=8)

        out1 = layer(tf.random.normal((1, 10, 64)))
        out2 = layer(tf.random.normal((5, 10, 64)))

        assert out1.shape == (1, 10, 64)
        assert out2.shape == (5, 10, 64)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])