"""Test suite for RPCAttention layer.

This module contains comprehensive tests for the RPCAttention layer,
focusing on the robustness mechanisms, PCP decomposition, and SVD stability
in addition to standard Keras layer functionality.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

from dl_techniques.layers.attention.rpc_attention import RPCAttention


class TestRPCAttention:
    """Test suite for RPCAttention layer."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return keras.random.normal((2, 10, 64))  # (batch_size, seq_len, dim)

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return RPCAttention(dim=64, num_heads=8)

    @pytest.fixture
    def different_configs(self):
        """Provide different layer configurations for testing."""
        return [
            {"dim": 32, "num_heads": 4, "lambda_sparse": 0.01},
            {"dim": 128, "num_heads": 8, "max_pcp_iter": 5},
            {"dim": 256, "num_heads": 16, "svd_threshold": 0.5},
            {"dim": 64, "num_heads": 8, "dropout_rate": 0.1, "qkv_bias": True},
        ]

    # ==================== Initialization Tests ====================

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = RPCAttention(dim=64, num_heads=8)

        assert layer.dim == 64
        assert layer.num_heads == 8
        assert layer.lambda_sparse == 0.1
        assert layer.max_pcp_iter == 10
        assert layer.svd_threshold == 1.0
        assert layer.dropout_rate == 0.0
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)

        # Check computed attributes
        assert layer.head_dim == 8
        np.testing.assert_allclose(layer.attention_scale, 1.0 / np.sqrt(8))

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_reg = keras.regularizers.L2(1e-4)

        layer = RPCAttention(
            dim=128,
            num_heads=8,
            lambda_sparse=0.5,
            max_pcp_iter=20,
            svd_threshold=0.1,
            qkv_bias=True,
            kernel_regularizer=custom_reg
        )

        assert layer.dim == 128
        assert layer.lambda_sparse == 0.5
        assert layer.max_pcp_iter == 20
        assert layer.svd_threshold == 0.1
        assert layer.qkv_bias is True
        assert layer.kernel_regularizer == custom_reg

    def test_invalid_dim_mismatch(self):
        """Test that invalid dim/head ratio raises ValueError."""
        with pytest.raises(ValueError, match="dim \\(63\\) must be divisible by num_heads \\(8\\)"):
            RPCAttention(dim=63, num_heads=8)

    def test_invalid_pcp_params(self):
        """Test validation of PCP-specific parameters."""
        with pytest.raises(ValueError, match="lambda_sparse must be positive"):
            RPCAttention(dim=64, lambda_sparse=-0.1)

        with pytest.raises(ValueError, match="max_pcp_iter must be positive"):
            RPCAttention(dim=64, max_pcp_iter=0)

        with pytest.raises(ValueError, match="svd_threshold must be positive"):
            RPCAttention(dim=64, svd_threshold=-1.0)

    # ==================== Build Process Tests ====================

    def test_build_process(self, input_tensor):
        """Test that the layer builds properly."""
        layer = RPCAttention(dim=64, num_heads=8)
        layer(input_tensor)  # Forward pass triggers build

        assert layer.built is True
        assert layer.to_qkv.built is True
        assert layer.to_out.built is True

        # Verify weight shapes
        # to_qkv: (dim, 3*dim) + bias if used
        expected_kernel_shape = (64, 64 * 3)
        assert layer.to_qkv.kernel.shape == expected_kernel_shape

    def test_build_input_shape_validation(self):
        """Test input shape validation in build."""
        layer = RPCAttention(dim=64, num_heads=8)

        # Test invalid input shape (2D instead of 3D)
        with pytest.raises(ValueError, match="Expected 3D input"):
            layer.build((32, 64))

        # Test dimension mismatch
        with pytest.raises(ValueError, match="Last dimension of input"):
            layer.build((None, 10, 32))

    # ==================== Forward Pass & Computation Tests ====================

    def test_forward_pass_basic(self, input_tensor):
        """Test basic forward pass functionality."""
        layer = RPCAttention(dim=64, num_heads=8)
        output = layer(input_tensor)

        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))
        assert output.shape == input_tensor.shape

    def test_return_attention_scores(self, input_tensor):
        """Test return_attention_scores=True behavior."""
        layer = RPCAttention(dim=64, num_heads=8)
        output, weights = layer(input_tensor, return_attention_scores=True)

        assert output.shape == input_tensor.shape
        # Weights shape: (batch, num_heads, seq_len, seq_len)
        expected_weights_shape = (2, 8, 10, 10)
        assert weights.shape == expected_weights_shape

        # Weights should sum to 1 on the last axis (softmax)
        sums = tf.reduce_sum(weights, axis=-1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(sums),
            np.ones(sums.shape),
            rtol=1e-5, atol=1e-5
        )

    def test_pcp_determinism(self, input_tensor):
        """
        Test that PCP decomposition is deterministic for the same input
        (SVD results should be consistent).
        """
        layer = RPCAttention(dim=64, num_heads=8, max_pcp_iter=5)

        # Run twice in inference mode
        out1 = layer(input_tensor, training=False)
        out2 = layer(input_tensor, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out1),
            keras.ops.convert_to_numpy(out2),
            rtol=1e-6, atol=1e-6,
            err_msg="RPC execution should be deterministic during inference"
        )

    # ==================== SVD & Numerical Stability Tests ====================

    def test_svd_stability_zeros(self):
        """Test stability when attention matrix is all zeros."""
        layer = RPCAttention(dim=64, num_heads=8)

        # Zero input -> Zero Q, K, V -> Zero Attention Matrix
        zeros_input = tf.zeros((2, 10, 64))
        output = layer(zeros_input)

        assert not tf.reduce_any(tf.math.is_nan(output))
        # With zero input, output should be zero (biases are zero by default)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            np.zeros((2, 10, 64)),
            atol=1e-6
        )

    def test_svd_stability_random(self):
        """Test stability with random normal inputs (checking for SVD convergence issues)."""
        # Use a larger matrix to stress the SVD
        layer = RPCAttention(dim=32, num_heads=4, max_pcp_iter=5)

        for _ in range(5):
            inp = keras.random.normal((2, 20, 32)) # (batch, seq, dim)
            output = layer(inp)
            assert not tf.reduce_any(tf.math.is_nan(output)), "NaN found in RPC output"

    def test_pcp_convergence_logic(self, input_tensor):
        """
        Indirectly test that increasing iterations or changing lambda affects output.
        This verifies the internal loops are actually running and using params.
        """
        # Layer with 1 iteration
        layer_fast = RPCAttention(dim=64, num_heads=8, max_pcp_iter=1, lambda_sparse=0.1)
        layer_fast.build(input_tensor.shape)
        # Force same weights
        weights = layer_fast.get_weights()

        # Layer with 10 iterations
        layer_deep = RPCAttention(dim=64, num_heads=8, max_pcp_iter=10, lambda_sparse=0.1)
        layer_deep.build(input_tensor.shape)
        layer_deep.set_weights(weights)

        out_fast = layer_fast(input_tensor, training=False)
        out_deep = layer_deep(input_tensor, training=False)

        # Outputs should be slightly different due to more refinement steps
        diff = tf.reduce_mean(tf.abs(out_fast - out_deep))
        assert diff > 0.0, "More PCP iterations should change the result"

    # ==================== Masking Tests ====================

    def test_mask_handling(self, input_tensor):
        """Test that attention masks are correctly applied before PCP."""
        layer = RPCAttention(dim=64, num_heads=8)

        seq_len = input_tensor.shape[1]
        # Mask the last 5 tokens
        mask = tf.concat([
            tf.ones((2, seq_len - 5)),
            tf.zeros((2, 5))
        ], axis=1)
        # Expand for attention logic usually handled by framework,
        # but here we pass raw mask to layer which expects (batch, seq, seq) or broadcastable
        mask_expanded = mask[:, None, :] * mask[:, :, None] # (batch, seq, seq)

        output_masked = layer(input_tensor, mask=mask_expanded)
        output_nomask = layer(input_tensor)

        assert not tf.reduce_all(tf.equal(output_masked, output_nomask))
        assert not tf.reduce_any(tf.math.is_nan(output_masked))

    # ==================== Gradient Flow Tests ====================

    def test_gradient_flow_through_svd(self, input_tensor):
        """
        Critical test: Ensure gradients propagate through SVD and the iterative loop.
        SVD gradients can be tricky in some backends.
        """
        layer = RPCAttention(dim=64, num_heads=8, max_pcp_iter=2) # Keep iter low for speed

        with tf.GradientTape() as tape:
            inputs = tf.Variable(input_tensor)
            outputs = layer(inputs)
            loss = tf.reduce_mean(tf.square(outputs))

        grads = tape.gradient(loss, layer.trainable_variables)

        # Check gradients exist
        assert len(grads) > 0
        assert all(g is not None for g in grads)

        # Check gradients are non-zero (implies flow through SVD)
        # We check the kernel weights of the projection layers
        for g in grads:
            assert tf.reduce_max(tf.abs(g)) > 0.0

    # ==================== Serialization Tests ====================

    def test_serialization(self, input_tensor):
        """Test complete serialization cycle."""
        # Note: dim=64 to match input_tensor fixture
        layer = RPCAttention(
            dim=64,
            num_heads=8,
            lambda_sparse=0.2,
            max_pcp_iter=5,
            svd_threshold=0.5,
            qkv_bias=True
        )

        # Build first
        layer(input_tensor)

        config = layer.get_config()
        recreated = RPCAttention.from_config(config)

        assert recreated.dim == 64
        assert recreated.lambda_sparse == 0.2
        assert recreated.max_pcp_iter == 5
        assert recreated.svd_threshold == 0.5
        assert recreated.qkv_bias is True

    def test_model_save_load(self, input_tensor):
        """Test saving/loading within a full model."""
        inputs = keras.Input(shape=(10, 64))
        x = RPCAttention(dim=64, num_heads=8)(inputs)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)

        # Run prediction
        pred_orig = model.predict(input_tensor, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "model.keras")
            model.save(path)

            loaded_model = keras.models.load_model(
                path,
                custom_objects={"RPCAttention": RPCAttention}
            )

            pred_loaded = loaded_model.predict(input_tensor, verbose=0)

            np.testing.assert_allclose(
                pred_orig, pred_loaded,
                rtol=1e-5, atol=1e-5
            )

    # ==================== Edge Case Tests ====================

    def test_single_token_sequence(self):
        """
        Test behavior with sequence length 1.
        SVD on 1x1 matrices should degenerate gracefully.
        """
        layer = RPCAttention(dim=64, num_heads=8)
        input_one = tf.random.normal((2, 1, 64))
        output = layer(input_one)

        assert output.shape == (2, 1, 64)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_batch_consistency(self):
        """Test that different batch sizes work."""
        layer = RPCAttention(dim=64, num_heads=8)

        # Batch size 1
        out1 = layer(tf.random.normal((1, 10, 64)))
        assert out1.shape == (1, 10, 64)

        # Batch size large
        out16 = layer(tf.random.normal((16, 10, 64)))
        assert out16.shape == (16, 10, 64)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])