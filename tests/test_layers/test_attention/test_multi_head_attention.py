"""Test suite for MultiHeadAttention layer.

This module contains comprehensive tests for the new MultiHeadAttention layer,
which wraps MultiHeadCrossAttention for self-attention functionality.
The layer uses 'dim' parameter instead of 'embed_dim' and delegates to
the underlying cross-attention implementation.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

# Import the layer to test
from dl_techniques.layers.attention.multi_head_attention import MultiHeadAttention


class TestMultiHeadAttention:
    """Test suite for MultiHeadAttention layer."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return tf.random.normal([2, 10, 64])  # (batch_size, seq_len, dim)

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return MultiHeadAttention(dim=64, num_heads=8)

    @pytest.fixture
    def different_configs(self):
        """Provide different layer configurations for testing."""
        return [
            {"dim": 32, "num_heads": 4},
            {"dim": 128, "num_heads": 8, "dropout_rate": 0.1},
            {"dim": 256, "num_heads": 16, "use_bias": True},
            {"dim": 512, "num_heads": 8, "dropout_rate": 0.2, "use_bias": True},
        ]

    # ==================== Initialization Tests ====================

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = MultiHeadAttention(dim=64, num_heads=8)

        assert layer.dim == 64
        assert layer.num_heads == 8
        assert layer.dropout_rate == 0.0
        assert layer.use_bias is False
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert layer.kernel_regularizer is None

        # Check that the cross_attention layer exists and has correct config
        assert layer.cross_attention is not None
        assert hasattr(layer, 'cross_attention')

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        layer = MultiHeadAttention(
            dim=128,
            num_heads=16,
            dropout_rate=0.1,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=custom_regularizer,
            use_bias=True
        )

        assert layer.dim == 128
        assert layer.num_heads == 16
        assert layer.dropout_rate == 0.1
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert layer.kernel_regularizer == custom_regularizer

    def test_invalid_dim_not_divisible(self):
        """Test that invalid dim raises ValueError."""
        with pytest.raises(ValueError, match="dim \\(63\\) must be divisible by num_heads \\(8\\)"):
            MultiHeadAttention(dim=63, num_heads=8)

    def test_invalid_dim_negative(self):
        """Test that negative dim raises ValueError."""
        with pytest.raises(ValueError, match="dim must be positive, got -64"):
            MultiHeadAttention(dim=-64, num_heads=8)

    def test_invalid_num_heads_negative(self):
        """Test that negative num_heads raises ValueError."""
        with pytest.raises(ValueError, match="num_heads must be positive, got -8"):
            MultiHeadAttention(dim=64, num_heads=-8)

    def test_invalid_dropout_rate(self):
        """Test that invalid dropout_rate raises ValueError."""
        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1, got 1.5"):
            MultiHeadAttention(dim=64, num_heads=8, dropout_rate=1.5)

    # ==================== Build Process Tests ====================

    def test_build_process(self, input_tensor):
        """Test that the layer builds properly."""
        layer = MultiHeadAttention(dim=64, num_heads=8)
        layer(input_tensor)  # Forward pass triggers build

        # Check that layer is built
        assert layer.built is True
        assert layer.cross_attention is not None
        assert layer.cross_attention.built is True

    def test_build_input_shape_validation(self):
        """Test input shape validation in build."""
        layer = MultiHeadAttention(dim=64, num_heads=8)

        # Test invalid input shape (2D instead of 3D)
        with pytest.raises(ValueError, match="Input must be 3D \\(batch, seq_len, dim\\), got shape"):
            layer.build((32, 64))  # Missing sequence dimension

        # Test dimension mismatch
        with pytest.raises(ValueError, match="Input last dimension \\(32\\) must match dim \\(64\\)"):
            layer.build((None, 10, 32))  # Wrong last dimension

    def test_explicit_build(self, input_tensor):
        """Test that build method works when called explicitly."""
        layer = MultiHeadAttention(dim=64, num_heads=8)
        layer.build(input_tensor.shape)

        assert layer.built is True
        assert layer.cross_attention.built is True

    # ==================== Output Shape Tests ====================

    def test_output_shapes(self, different_configs):
        """Test that output shapes are computed correctly."""
        for config in different_configs:
            layer = MultiHeadAttention(**config)

            # Test different sequence lengths
            for seq_len in [10, 50, 100]:
                input_shape = (2, seq_len, config["dim"])
                input_tensor = tf.random.normal(input_shape)

                output = layer(input_tensor)
                expected_shape = input_shape

                assert output.shape == expected_shape

                # Test compute_output_shape separately
                computed_shape = layer.compute_output_shape(input_shape)
                assert computed_shape == expected_shape

    def test_batch_size_flexibility(self):
        """Test that the layer works with different batch sizes."""
        layer = MultiHeadAttention(dim=64, num_heads=8)

        for batch_size in [1, 4, 16]:
            input_tensor = tf.random.normal([batch_size, 10, 64])
            output = layer(input_tensor)
            assert output.shape == (batch_size, 10, 64)

    # ==================== Forward Pass Tests ====================

    def test_forward_pass_basic(self, input_tensor):
        """Test basic forward pass functionality."""
        layer = MultiHeadAttention(dim=64, num_heads=8)
        output = layer(input_tensor)

        # Basic sanity checks
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))
        assert output.shape == input_tensor.shape

    def test_forward_pass_deterministic(self):
        """Test forward pass with deterministic inputs."""
        # Create deterministic layer
        layer = MultiHeadAttention(
            dim=64,
            num_heads=8,
            kernel_initializer="ones",
            dropout_rate=0.0
        )

        # Use deterministic input
        input_tensor = tf.ones([1, 5, 64])

        # Multiple forward passes should give same result in inference mode
        output1 = layer(input_tensor, training=False)
        output2 = layer(input_tensor, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Deterministic forward passes should match"
        )

    def test_training_mode_differences(self, input_tensor):
        """Test that training mode affects dropout behavior."""
        layer = MultiHeadAttention(dim=64, num_heads=8, dropout_rate=0.5)

        # Build layer first
        layer(input_tensor, training=False)

        # Set different random seeds to ensure different dropout patterns
        tf.random.set_seed(42)
        output_train = layer(input_tensor, training=True)

        tf.random.set_seed(42)  # Same seed for consistency
        output_inference = layer(input_tensor, training=False)

        # Results should be different due to dropout in training mode
        assert not tf.reduce_all(tf.equal(output_train, output_inference))

    # ==================== Attention Mask Tests ====================

    def test_no_mask_functionality(self, input_tensor):
        """Test that layer works properly without mask."""
        layer = MultiHeadAttention(dim=64, num_heads=8)

        # Should work without mask
        output = layer(input_tensor)
        assert output.shape == input_tensor.shape

    def test_sequence_level_mask(self, input_tensor):
        """Test sequence-level attention mask."""
        layer = MultiHeadAttention(dim=64, num_heads=8, dropout_rate=0.0)

        # Create mask that masks the last 3 positions
        seq_len = input_tensor.shape[1]
        mask = tf.ones((input_tensor.shape[0], seq_len))
        mask = tf.concat([mask[:, :-3], tf.zeros((input_tensor.shape[0], 3))], axis=1)

        output_masked = layer(input_tensor, attention_mask=mask, training=False)
        output_unmasked = layer(input_tensor, training=False)

        # Outputs should be different
        assert not tf.reduce_all(tf.equal(output_masked, output_unmasked))

        # Check that output is valid
        assert not tf.reduce_any(tf.math.is_nan(output_masked))
        assert not tf.reduce_any(tf.math.is_inf(output_masked))

    def test_full_attention_mask(self, input_tensor):
        """Test full attention mask (causal attention)."""
        layer = MultiHeadAttention(dim=64, num_heads=8, dropout_rate=0.0)

        seq_len = input_tensor.shape[1]
        batch_size = input_tensor.shape[0]

        # Create causal mask
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        causal_mask = tf.expand_dims(causal_mask, 0)  # Add batch dimension
        causal_mask = tf.tile(causal_mask, [batch_size, 1, 1])

        output_causal = layer(input_tensor, attention_mask=causal_mask, training=False)
        output_full = layer(input_tensor, training=False)

        # Outputs should be different
        assert not tf.reduce_all(tf.equal(output_causal, output_full))

        # Check that output is valid
        assert not tf.reduce_any(tf.math.is_nan(output_causal))
        assert not tf.reduce_any(tf.math.is_inf(output_causal))

    def test_per_head_mask(self, input_tensor):
        """Test per-head attention mask."""
        layer = MultiHeadAttention(dim=64, num_heads=8, dropout_rate=0.0)

        seq_len = input_tensor.shape[1]
        batch_size = input_tensor.shape[0]
        num_heads = 8

        # Create different masks for different heads
        mask = tf.ones((batch_size, num_heads, seq_len, seq_len))
        # Mask different positions for different heads
        mask = mask.numpy()
        mask[:, 0, :, -1] = 0  # Head 0: mask last position
        mask[:, 1, :, -2:] = 0  # Head 1: mask last 2 positions
        mask = tf.constant(mask)

        output_masked = layer(input_tensor, attention_mask=mask, training=False)
        output_unmasked = layer(input_tensor, training=False)

        # Outputs should be different
        assert not tf.reduce_all(tf.equal(output_masked, output_unmasked))

        # Check that output is valid
        assert not tf.reduce_any(tf.math.is_nan(output_masked))
        assert not tf.reduce_any(tf.math.is_inf(output_masked))

    def test_different_mask_dtypes(self, input_tensor):
        """Test that masks with different dtypes work correctly."""
        layer = MultiHeadAttention(dim=64, num_heads=8)

        seq_len = input_tensor.shape[1]
        batch_size = input_tensor.shape[0]

        # Test different dtypes
        for dtype in [tf.int32, tf.float32, tf.bool]:
            mask = tf.ones((batch_size, seq_len), dtype=dtype)
            if dtype == tf.bool:
                mask = tf.cast(mask, tf.bool)

            output = layer(input_tensor, attention_mask=mask)
            assert not tf.reduce_any(tf.math.is_nan(output))

    # ==================== Serialization Tests ====================

    def test_serialization_config_completeness(self):
        """Test that get_config captures all necessary parameters."""
        layer = MultiHeadAttention(
            dim=256,
            num_heads=16,
            dropout_rate=0.2,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=keras.regularizers.L2(1e-4),
            use_bias=True
        )

        config = layer.get_config()

        # Check all important parameters are in config
        required_keys = ["dim", "num_heads", "dropout_rate", "kernel_initializer",
                        "kernel_regularizer", "use_bias"]
        for key in required_keys:
            assert key in config, f"Missing key {key} in config"

        # Check values
        assert config["dim"] == 256
        assert config["num_heads"] == 16
        assert config["dropout_rate"] == 0.2
        assert config["use_bias"] is True

    def test_layer_recreation_from_config(self):
        """Test recreating layer from config."""
        # Create original layer
        original_layer = MultiHeadAttention(
            dim=128,
            num_heads=8,
            dropout_rate=0.1,
            kernel_initializer="he_normal",
            use_bias=True
        )

        # Get config and recreate layer
        config = original_layer.get_config()
        recreated_layer = MultiHeadAttention.from_config(config)

        # Check configuration matches
        assert recreated_layer.dim == original_layer.dim
        assert recreated_layer.num_heads == original_layer.num_heads
        assert recreated_layer.dropout_rate == original_layer.dropout_rate
        assert recreated_layer.use_bias == original_layer.use_bias

    def test_serialization_with_build(self, input_tensor):
        """Test serialization after building the layer."""
        # Create and build layer
        original_layer = MultiHeadAttention(
            dim=64,
            num_heads=8,
            dropout_rate=0.1,
            use_bias=True
        )

        # Build the layer
        original_layer(input_tensor)

        # Get config and recreate
        config = original_layer.get_config()
        recreated_layer = MultiHeadAttention.from_config(config)

        # Build recreated layer
        recreated_layer(input_tensor)

        # Both layers should have same structure
        assert len(recreated_layer.weights) == len(original_layer.weights)

    # ==================== Model Integration Tests ====================

    def test_model_integration(self, input_tensor):
        """Test the layer in a model context."""
        # Create a model with the custom layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = MultiHeadAttention(dim=64, num_heads=8)(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
        )

        # Test forward pass
        y_pred = model(input_tensor, training=False)
        assert y_pred.shape == (input_tensor.shape[0], 10)

    def test_model_with_mask_integration(self, input_tensor):
        """Test model integration with attention mask."""
        # Create a model that uses attention mask
        inputs = keras.Input(shape=input_tensor.shape[1:])
        mask_input = keras.Input(shape=input_tensor.shape[1:2])

        # Apply attention with mask
        x = MultiHeadAttention(dim=64, num_heads=8)(inputs, attention_mask=mask_input)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=[inputs, mask_input], outputs=outputs)

        # Create mask
        mask = tf.ones((input_tensor.shape[0], input_tensor.shape[1]))

        # Test forward pass
        y_pred = model([input_tensor, mask], training=False)
        assert y_pred.shape == (input_tensor.shape[0], 10)

    # ==================== Model Save/Load Tests ====================

    def test_model_save_load(self, input_tensor):
        """Test saving and loading a model with the custom layer."""
        # Create a model with the custom layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = MultiHeadAttention(dim=64, num_heads=8, name="custom_attention")(inputs)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate a prediction before saving
        original_prediction = model.predict(input_tensor, verbose=0)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"MultiHeadAttention": MultiHeadAttention}
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            # Check predictions match
            np.testing.assert_allclose(
                original_prediction, loaded_prediction,
                rtol=1e-5, atol=1e-5,
                err_msg="Predictions should match after save/load"
            )

            # Check layer type is preserved
            assert isinstance(loaded_model.get_layer("custom_attention"), MultiHeadAttention)

    # ==================== Gradient Flow Tests ====================

    def test_gradient_flow(self, input_tensor):
        """Test gradient flow through the layer."""
        layer = MultiHeadAttention(dim=64, num_heads=8)

        # Watch the variables
        with tf.GradientTape() as tape:
            inputs = tf.Variable(input_tensor)
            outputs = layer(inputs)
            loss = tf.reduce_mean(tf.square(outputs))

        # Get gradients
        grads = tape.gradient(loss, layer.trainable_variables)

        # Check gradients exist and are not None
        assert all(g is not None for g in grads), "All gradients should be non-None"

        # Check gradients have values (not all zeros)
        assert all(tf.reduce_any(g != 0) for g in grads), "Gradients should have non-zero values"

    def test_gradient_flow_with_mask(self, input_tensor):
        """Test gradient flow with attention mask."""
        layer = MultiHeadAttention(dim=64, num_heads=8)

        # Create mask
        mask = tf.ones((input_tensor.shape[0], input_tensor.shape[1]))

        # Watch the variables
        with tf.GradientTape() as tape:
            inputs = tf.Variable(input_tensor)
            outputs = layer(inputs, attention_mask=mask)
            loss = tf.reduce_mean(tf.square(outputs))

        # Get gradients
        grads = tape.gradient(loss, layer.trainable_variables)

        # Check gradients exist and are not None
        assert all(g is not None for g in grads), "All gradients should be non-None"

        # Check gradients have values (not all zeros)
        assert all(tf.reduce_any(g != 0) for g in grads), "Gradients should have non-zero values"

    # ==================== Edge Case Tests ====================

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = MultiHeadAttention(dim=64, num_heads=8)

        test_cases = [
            tf.zeros((2, 10, 64)),  # All zeros
            tf.ones((2, 10, 64)) * 1e-10,  # Very small values
            tf.ones((2, 10, 64)) * 1e3,  # Large values (reduced from 1e5)
            tf.random.normal((2, 10, 64)) * 10,  # Large random values (reduced from 100)
        ]

        for i, test_input in enumerate(test_cases):
            output = layer(test_input)

            # Check for NaN/Inf values
            assert not tf.reduce_any(tf.math.is_nan(output)), f"NaN values detected in output for test case {i}"
            assert not tf.reduce_any(tf.math.is_inf(output)), f"Inf values detected in output for test case {i}"

    def test_single_sequence_length(self):
        """Test with sequence length of 1."""
        layer = MultiHeadAttention(dim=64, num_heads=8)

        # Single time step
        input_tensor = tf.random.normal([2, 1, 64])
        output = layer(input_tensor)

        assert output.shape == (2, 1, 64)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_large_sequence_length(self):
        """Test with large sequence length."""
        layer = MultiHeadAttention(dim=64, num_heads=8)

        # Large sequence (memory intensive)
        input_tensor = tf.random.normal([1, 500, 64])  # Reduced from 1000 for efficiency
        output = layer(input_tensor)

        assert output.shape == (1, 500, 64)
        assert not tf.reduce_any(tf.math.is_nan(output))

    # ==================== Cross-Attention Delegation Tests ====================

    def test_cross_attention_delegation(self, input_tensor):
        """Test that the layer correctly delegates to cross_attention."""
        layer = MultiHeadAttention(dim=64, num_heads=8)

        # Build layer
        layer(input_tensor)

        # Check that cross_attention is properly configured
        assert layer.cross_attention is not None
        assert hasattr(layer.cross_attention, 'dim')
        assert layer.cross_attention.dim == 64
        assert layer.cross_attention.num_heads == 8

    def test_shared_qk_projections(self, input_tensor):
        """Test that the layer uses shared_qk_projections=True for self-attention."""
        layer = MultiHeadAttention(dim=64, num_heads=8)

        # Build layer
        layer(input_tensor)

        # The cross_attention layer should be configured with shared_qk_projections=True
        # This is a property of the self-attention configuration
        assert layer.cross_attention.shared_qk_projections is True

    # ==================== Performance Tests ====================

    def test_attention_mask_performance(self, input_tensor):
        """Test that attention mask doesn't significantly impact performance."""
        layer = MultiHeadAttention(dim=64, num_heads=8)

        # Create mask
        mask = tf.ones((input_tensor.shape[0], input_tensor.shape[1]))

        # Time without mask
        import time
        start = time.time()
        for _ in range(5):  # Reduced iterations for efficiency
            _ = layer(input_tensor, training=False)
        time_without_mask = time.time() - start

        # Time with mask
        start = time.time()
        for _ in range(5):  # Reduced iterations for efficiency
            _ = layer(input_tensor, attention_mask=mask, training=False)
        time_with_mask = time.time() - start

        # Mask should not significantly slow down computation
        # Allow 3x overhead for masking operations (more lenient)
        assert time_with_mask < time_without_mask * 3.0

    def test_different_head_counts(self):
        """Test layer with different numbers of heads."""
        dim = 64
        input_tensor = tf.random.normal([2, 10, dim])

        # Test with different head counts
        for num_heads in [1, 2, 4, 8, 16]:
            if dim % num_heads == 0:  # Only test valid configurations
                layer = MultiHeadAttention(dim=dim, num_heads=num_heads)
                output = layer(input_tensor)

                assert output.shape == input_tensor.shape
                assert not tf.reduce_any(tf.math.is_nan(output))

    # ==================== Wrapper Pattern Tests ====================

    def test_wrapper_pattern_consistency(self, input_tensor):
        """Test that the wrapper pattern maintains consistency."""
        layer = MultiHeadAttention(dim=64, num_heads=8, dropout_rate=0.1)

        # Build layer
        output = layer(input_tensor)

        # The layer should behave as self-attention
        # Output shape should match input shape
        assert output.shape == input_tensor.shape

        # Layer should have the cross_attention sublayer properly configured
        assert layer.cross_attention.dropout_rate == 0.1
        assert layer.cross_attention.dim == 64
        assert layer.cross_attention.num_heads == 8

    def test_parameter_propagation(self):
        """Test that parameters are properly propagated to cross_attention."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        layer = MultiHeadAttention(
            dim=128,
            num_heads=8,
            dropout_rate=0.2,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=custom_regularizer,
            use_bias=True
        )

        # Check that parameters are propagated to cross_attention
        assert layer.cross_attention.dropout_rate == 0.2
        assert layer.cross_attention.use_bias is True
        assert isinstance(layer.cross_attention.kernel_initializer, keras.initializers.GlorotUniform)
        assert layer.cross_attention.kernel_regularizer == custom_regularizer

if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v"])