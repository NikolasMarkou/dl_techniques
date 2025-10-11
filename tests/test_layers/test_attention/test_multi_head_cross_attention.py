"""Test suite for MultiHeadCrossAttention layer.

This module contains comprehensive tests for the MultiHeadCrossAttention layer,
validating its functionality in both cross-attention and self-attention modes,
including masking, adaptive softmax, and serialization.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

# Import the layer to test
from dl_techniques.layers.attention.multi_head_cross_attention import MultiHeadCrossAttention


class TestMultiHeadCrossAttention:
    """Test suite for MultiHeadCrossAttention layer."""

    @pytest.fixture
    def query_input(self):
        """Create a test query input tensor."""
        return tf.random.normal([2, 10, 64])  # (batch, query_seq_len, dim)

    @pytest.fixture
    def kv_input(self):
        """Create a test key-value input tensor."""
        return tf.random.normal([2, 20, 64])  # (batch, kv_seq_len, dim)

    @pytest.fixture
    def different_configs(self):
        """Provide different layer configurations for testing."""
        return [
            {"dim": 32, "num_heads": 4},
            {"dim": 128, "num_heads": 8, "dropout_rate": 0.1},
            {"dim": 256, "num_heads": 16, "use_bias": False},
            {
                "dim": 512,
                "num_heads": 8,
                "dropout_rate": 0.2,
                "use_adaptive_softmax": True,
                "adaptive_softmax_config": {"min_temp": 0.2, "max_temp": 2.0}
            },
        ]

    # ==================== Initialization Tests ====================

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8)

        assert layer.dim == 64
        assert layer.num_heads == 8
        assert layer.head_dim == 8
        assert layer.dropout_rate == 0.0
        assert layer.shared_qk_projections is False
        assert layer.use_bias is True
        assert layer.use_adaptive_softmax is False
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert layer.adaptive_softmax is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)
        adaptive_config = {"min_temp": 0.05, "max_temp": 5.0}

        layer = MultiHeadCrossAttention(
            dim=128,
            num_heads=16,
            dropout_rate=0.1,
            kernel_initializer="he_normal",
            kernel_regularizer=custom_regularizer,
            use_bias=False,
            use_adaptive_softmax=True,
            adaptive_softmax_config=adaptive_config
        )

        assert layer.dim == 128
        assert layer.num_heads == 16
        assert layer.dropout_rate == 0.1
        assert layer.use_bias is False
        assert layer.use_adaptive_softmax is True
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert layer.kernel_regularizer == custom_regularizer
        assert layer.adaptive_softmax is not None
        assert layer.adaptive_softmax.min_temp == 0.05
        assert layer.adaptive_softmax.max_temp == 5.0

    def test_invalid_dim_not_divisible(self):
        """Test that invalid dim raises ValueError."""
        with pytest.raises(ValueError, match="dim \\(63\\) must be divisible by num_heads \\(8\\)"):
            MultiHeadCrossAttention(dim=63, num_heads=8)

    def test_invalid_adaptive_softmax_config(self):
        """Test invalid parameters for adaptive softmax."""
        with pytest.raises(ValueError, match="min_temp must be positive, got 0"):
            MultiHeadCrossAttention(dim=64, num_heads=8, use_adaptive_softmax=True, adaptive_softmax_config={"min_temp": 0})
        with pytest.raises(ValueError, match="max_temp \\(0.5\\) must be greater than min_temp \\(1.0\\)"):
            MultiHeadCrossAttention(dim=64, num_heads=8, use_adaptive_softmax=True, adaptive_softmax_config={"min_temp": 1.0, "max_temp": 0.5})

    # ==================== Build Process Tests ====================

    def test_build_cross_attention(self, query_input, kv_input):
        """Test build process for cross-attention mode."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8)
        layer(query_input, kv_input)
        assert layer.built is True
        assert layer.q_dense is not None and layer.q_dense.built
        assert layer.kv_dense is not None and layer.kv_dense.built
        assert layer.qkv_dense is None

    def test_build_self_attention_shared(self, query_input):
        """Test build process for self-attention with shared projections."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8, shared_qk_projections=True)
        layer(query_input)
        assert layer.built is True
        assert layer.qkv_dense is not None and layer.qkv_dense.built
        assert layer.q_dense is None and layer.kv_dense is None

    def test_explicit_build(self, query_input, kv_input):
        """Test explicit build calls."""
        # Cross-attention
        layer_cross = MultiHeadCrossAttention(dim=64, num_heads=8)
        layer_cross.build([query_input.shape, kv_input.shape])
        assert layer_cross.built is True

        # Self-attention
        layer_self = MultiHeadCrossAttention(dim=64, num_heads=8)
        layer_self.build(query_input.shape)
        assert layer_self.built is True

    # ==================== Output Shape Tests ====================

    def test_output_shape_cross_attention(self, query_input, kv_input):
        """Test output shape for cross-attention."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8)
        output = layer(query_input, kv_input)
        assert output.shape == query_input.shape
        assert layer.compute_output_shape([query_input.shape, kv_input.shape]) == query_input.shape

    def test_output_shape_self_attention(self, query_input):
        """Test output shape for self-attention."""
        # Separate projections
        layer_sep = MultiHeadCrossAttention(dim=64, num_heads=8)
        output_sep = layer_sep(query_input)
        assert output_sep.shape == query_input.shape
        assert layer_sep.compute_output_shape(query_input.shape) == query_input.shape

        # Shared projections
        layer_shared = MultiHeadCrossAttention(dim=64, num_heads=8, shared_qk_projections=True)
        output_shared = layer_shared(query_input)
        assert output_shared.shape == query_input.shape

    # ==================== Forward Pass Tests ====================

    def test_forward_pass_modes(self, query_input, kv_input):
        """Test forward pass for all modes."""
        # Cross-attention
        cross_attn = MultiHeadCrossAttention(dim=64, num_heads=8)
        cross_output = cross_attn(query_input, kv_input)
        assert cross_output.shape == query_input.shape
        assert not tf.reduce_any(tf.math.is_nan(cross_output))

        # Self-attention (separate)
        self_attn_sep = MultiHeadCrossAttention(dim=64, num_heads=8)
        self_output_sep = self_attn_sep(query_input)
        assert self_output_sep.shape == query_input.shape
        assert not tf.reduce_any(tf.math.is_nan(self_output_sep))

        # Self-attention (shared)
        self_attn_shared = MultiHeadCrossAttention(dim=64, num_heads=8, shared_qk_projections=True)
        self_output_shared = self_attn_shared(query_input)
        assert self_output_shared.shape == query_input.shape
        assert not tf.reduce_any(tf.math.is_nan(self_output_shared))

    def test_shared_projections_with_kv_input_fails(self, query_input, kv_input):
        """Test that shared_qk_projections with kv_input raises an error."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8, shared_qk_projections=True)
        with pytest.raises(ValueError, match="When `shared_qk_projections=True`, `kv_input` must be None"):
            layer(query_input, kv_input)

    def test_adaptive_softmax_forward_pass(self, query_input, kv_input):
        """Test forward pass with adaptive softmax enabled."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8, use_adaptive_softmax=True)
        output = layer(query_input, kv_input)
        assert output.shape == query_input.shape
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_training_mode_differences(self, query_input):
        """Test that training mode affects dropout behavior."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8, dropout_rate=0.5)
        tf.random.set_seed(42)
        output_train = layer(query_input, training=True)
        tf.random.set_seed(42)
        output_inference = layer(query_input, training=False)
        assert not tf.reduce_all(tf.equal(output_train, output_inference))

    # ==================== Attention Mask Tests ====================

    def test_padding_mask(self, query_input, kv_input):
        """Test with a 2D padding mask."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8, dropout_rate=0.0)
        mask = tf.ones((kv_input.shape[0], kv_input.shape[1]))
        mask = tf.concat([mask[:, :-5], tf.zeros((kv_input.shape[0], 5))], axis=1)

        output_masked = layer(query_input, kv_input, attention_mask=mask, training=False)
        output_unmasked = layer(query_input, kv_input, training=False)

        assert not tf.reduce_all(tf.equal(output_masked, output_unmasked))
        assert not tf.reduce_any(tf.math.is_nan(output_masked))

    def test_full_attention_mask(self, query_input, kv_input):
        """Test with a 3D attention mask."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8, dropout_rate=0.0)
        mask = tf.ones((query_input.shape[0], query_input.shape[1], kv_input.shape[1]))
        # Mask some connections
        mask_np = mask.numpy()
        mask_np[:, 0, 0] = 0
        mask_np[:, -1, -1] = 0
        mask = tf.constant(mask_np)

        output_masked = layer(query_input, kv_input, attention_mask=mask, training=False)
        output_unmasked = layer(query_input, kv_input, training=False)
        assert not tf.reduce_all(tf.equal(output_masked, output_unmasked))

    def test_different_mask_dtypes(self, query_input, kv_input):
        """Test that masks with different dtypes work correctly."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8)
        for dtype in [tf.int32, tf.float32, tf.bool]:
            mask = tf.ones((kv_input.shape[0], kv_input.shape[1]), dtype=dtype)
            if dtype == tf.bool:
                mask = tf.cast(mask, tf.bool)
            output = layer(query_input, kv_input, attention_mask=mask)
            assert not tf.reduce_any(tf.math.is_nan(output))

    # ==================== Serialization Tests ====================

    def test_serialization_config_completeness(self):
        """Test that get_config captures all necessary parameters."""
        adaptive_config = {"min_temp": 0.1, "max_temp": 2.0}
        layer = MultiHeadCrossAttention(
            dim=256,
            num_heads=16,
            dropout_rate=0.2,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=keras.regularizers.L2(1e-4),
            use_bias=True,
            use_adaptive_softmax=True,
            adaptive_softmax_config=adaptive_config
        )
        config = layer.get_config()
        required_keys = [
            "dim", "num_heads", "dropout_rate", "shared_qk_projections",
            "use_bias", "kernel_initializer", "kernel_regularizer",
            "use_adaptive_softmax", "adaptive_softmax_config"
        ]
        for key in required_keys:
            assert key in config, f"Missing key {key} in config"
        assert config["adaptive_softmax_config"]["max_temp"] == 2.0

    def test_layer_recreation_from_config(self):
        """Test recreating layer from config."""
        original_layer = MultiHeadCrossAttention(
            dim=128,
            num_heads=8,
            dropout_rate=0.1,
            use_adaptive_softmax=True,
            adaptive_softmax_config={"polynomial_coeffs": [1.0, 0.5]}
        )
        config = original_layer.get_config()
        recreated_layer = MultiHeadCrossAttention.from_config(config)
        assert recreated_layer.dim == original_layer.dim
        assert recreated_layer.num_heads == original_layer.num_heads
        assert recreated_layer.use_adaptive_softmax is True
        assert recreated_layer.adaptive_softmax_config["polynomial_coeffs"] == [1.0, 0.5]

    # ==================== Model Integration Tests ====================

    def test_cross_attention_model_integration(self, query_input, kv_input):
        """Test the layer in a cross-attention model context."""
        query = keras.Input(shape=query_input.shape[1:])
        kv = keras.Input(shape=kv_input.shape[1:])
        x = MultiHeadCrossAttention(dim=64, num_heads=8)(query, kv)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)
        model = keras.Model(inputs=[query, kv], outputs=outputs)
        model.compile(optimizer="adam", loss="mse")
        y_pred = model([query_input, kv_input], training=False)
        assert y_pred.shape == (query_input.shape[0], 10)

    def test_self_attention_model_integration(self, query_input):
        """Test the layer in a self-attention model context."""
        inputs = keras.Input(shape=query_input.shape[1:])
        x = MultiHeadCrossAttention(dim=64, num_heads=8, shared_qk_projections=True)(inputs)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")
        y_pred = model(query_input, training=False)
        assert y_pred.shape == (query_input.shape[0], 10)

    # ==================== Model Save/Load Tests ====================

    def test_model_save_load(self, query_input, kv_input):
        """Test saving and loading a model with the custom layer."""
        query = keras.Input(shape=query_input.shape[1:])
        kv = keras.Input(shape=kv_input.shape[1:])
        x = MultiHeadCrossAttention(dim=64, num_heads=8, name="custom_cross_attention")(query, kv)
        outputs = keras.layers.GlobalAveragePooling1D()(x)
        model = keras.Model(inputs=[query, kv], outputs=outputs)

        original_prediction = model.predict([query_input, kv_input], verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"MultiHeadCrossAttention": MultiHeadCrossAttention}
            )
            loaded_prediction = loaded_model.predict([query_input, kv_input], verbose=0)
            np.testing.assert_allclose(
                original_prediction, loaded_prediction,
                rtol=1e-5, atol=1e-5,
                err_msg="Predictions should match after save/load"
            )
            assert isinstance(loaded_model.get_layer("custom_cross_attention"), MultiHeadCrossAttention)

    # ==================== Gradient Flow Tests ====================

    def test_gradient_flow(self, query_input, kv_input):
        """Test gradient flow through the layer."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8)
        with tf.GradientTape() as tape:
            q_var = tf.Variable(query_input)
            kv_var = tf.Variable(kv_input)
            outputs = layer(q_var, kv_var)
            loss = tf.reduce_mean(tf.square(outputs))
        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads), "All gradients should be non-None"
        assert all(tf.reduce_any(g != 0) for g in grads), "Gradients should have non-zero values"

    def test_gradient_flow_with_mask(self, query_input, kv_input):
        """Test gradient flow with attention mask."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8)
        mask = tf.ones((kv_input.shape[0], kv_input.shape[1]))
        with tf.GradientTape() as tape:
            q_var = tf.Variable(query_input)
            kv_var = tf.Variable(kv_input)
            outputs = layer(q_var, kv_var, attention_mask=mask)
            loss = tf.reduce_mean(tf.square(outputs))
        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads), "All gradients should be non-None"
        assert all(tf.reduce_any(g != 0) for g in grads), "Gradients should have non-zero values"

    # ==================== Edge Case Tests ====================

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8)
        test_cases = [
            tf.zeros((2, 10, 64)),
            tf.ones((2, 10, 64)) * 1e-10,
            tf.ones((2, 10, 64)) * 1e3,
            tf.random.normal((2, 10, 64)) * 10,
        ]
        for i, test_input in enumerate(test_cases):
            output = layer(test_input, test_input)
            assert not tf.reduce_any(tf.math.is_nan(output)), f"NaN values in test case {i}"
            assert not tf.reduce_any(tf.math.is_inf(output)), f"Inf values in test case {i}"

    def test_single_sequence_length(self):
        """Test with sequence length of 1."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8)
        query = tf.random.normal([2, 1, 64])
        kv = tf.random.normal([2, 1, 64])
        output = layer(query, kv)
        assert output.shape == (2, 1, 64)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_different_query_kv_seq_len(self):
        """Test with different query and kv sequence lengths."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8)
        query = tf.random.normal([2, 5, 64])
        kv = tf.random.normal([2, 50, 64])
        output = layer(query, kv)
        assert output.shape == query.shape
        assert not tf.reduce_any(tf.math.is_nan(output))

if __name__ == "__main__":
    pytest.main([__file__, "-v"])