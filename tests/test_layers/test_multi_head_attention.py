"""Test suite for MultiHeadAttention layer.

This module contains comprehensive tests for the MultiHeadAttention layer,
including initialization, forward pass, serialization, and attention mask
functionality tests.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

# Import the layer to test
from dl_techniques.layers.multi_head_attention import MultiHeadAttention


class TestMultiHeadAttention:
    """Test suite for MultiHeadAttention layer."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return tf.random.normal([2, 10, 64])  # (batch_size, seq_len, embed_dim)

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return MultiHeadAttention(embed_dim=64, num_heads=8)

    @pytest.fixture
    def different_configs(self):
        """Provide different layer configurations for testing."""
        return [
            {"embed_dim": 32, "num_heads": 4},
            {"embed_dim": 128, "num_heads": 8, "dropout_rate": 0.1},
            {"embed_dim": 256, "num_heads": 16, "use_bias": True},
            {"embed_dim": 512, "num_heads": 8, "dropout_rate": 0.2, "use_bias": True},
        ]

    # ==================== Initialization Tests ====================

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = MultiHeadAttention(embed_dim=64, num_heads=8)

        assert layer.embed_dim == 64
        assert layer.num_heads == 8
        assert layer.head_dim == 8
        assert layer.dropout_rate == 0.0
        assert layer.use_bias is False
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert layer.kernel_regularizer is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        layer = MultiHeadAttention(
            embed_dim=128,
            num_heads=16,
            dropout_rate=0.1,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=custom_regularizer,
            use_bias=True
        )

        assert layer.embed_dim == 128
        assert layer.num_heads == 16
        assert layer.head_dim == 8
        assert layer.dropout_rate == 0.1
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert layer.kernel_regularizer == custom_regularizer

    def test_invalid_embed_dim(self):
        """Test that invalid embed_dim raises ValueError."""
        with pytest.raises(ValueError, match="embed_dim \\(63\\) must be divisible by num_heads \\(8\\)"):
            MultiHeadAttention(embed_dim=63, num_heads=8)

    def test_head_dim_calculation(self):
        """Test correct head dimension calculation."""
        layer = MultiHeadAttention(embed_dim=512, num_heads=8)
        assert layer.head_dim == 64

        layer = MultiHeadAttention(embed_dim=768, num_heads=12)
        assert layer.head_dim == 64

    # ==================== Build Process Tests ====================

    def test_build_process(self, input_tensor):
        """Test that the layer builds properly."""
        layer = MultiHeadAttention(embed_dim=64, num_heads=8)
        layer(input_tensor)  # Forward pass triggers build

        # Check that layer is built
        assert layer.built is True
        assert layer.qkv is not None
        assert layer.proj is not None
        assert layer.dropout is not None

        # Check sublayer properties
        assert layer.qkv.units == 64 * 3  # embed_dim * 3 for Q, K, V
        assert layer.proj.units == 64
        assert layer.dropout.rate == 0.0

    def test_build_stores_input_shape(self, input_tensor):
        """Test that build stores input shape for serialization."""
        layer = MultiHeadAttention(embed_dim=64, num_heads=8)
        layer(input_tensor)

        assert layer._build_input_shape is not None
        assert layer._build_input_shape == input_tensor.shape

    # ==================== Output Shape Tests ====================

    def test_output_shapes(self, different_configs):
        """Test that output shapes are computed correctly."""
        for config in different_configs:
            layer = MultiHeadAttention(**config)

            # Test different sequence lengths
            for seq_len in [10, 50, 100]:
                input_shape = (2, seq_len, config["embed_dim"])
                input_tensor = tf.random.normal(input_shape)

                output = layer(input_tensor)
                expected_shape = input_shape

                assert output.shape == expected_shape

                # Test compute_output_shape separately
                computed_shape = layer.compute_output_shape(input_shape)
                assert computed_shape == expected_shape

    def test_batch_size_flexibility(self):
        """Test that the layer works with different batch sizes."""
        layer = MultiHeadAttention(embed_dim=64, num_heads=8)

        for batch_size in [1, 4, 16]:
            input_tensor = tf.random.normal([batch_size, 10, 64])
            output = layer(input_tensor)
            assert output.shape == (batch_size, 10, 64)

    # ==================== Forward Pass Tests ====================

    def test_forward_pass_basic(self, input_tensor):
        """Test basic forward pass functionality."""
        layer = MultiHeadAttention(embed_dim=64, num_heads=8)
        output = layer(input_tensor)

        # Basic sanity checks
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))
        assert output.shape == input_tensor.shape

    def test_forward_pass_deterministic(self):
        """Test forward pass with deterministic inputs."""
        # Create deterministic layer
        layer = MultiHeadAttention(
            embed_dim=64,
            num_heads=8,
            kernel_initializer="ones",
            dropout_rate=0.0
        )

        # Use deterministic input
        input_tensor = tf.ones([1, 5, 64])

        # Multiple forward passes should give same result
        output1 = layer(input_tensor, training=False)
        output2 = layer(input_tensor, training=False)

        assert tf.reduce_all(tf.equal(output1, output2))

    def test_training_mode_differences(self, input_tensor):
        """Test that training mode affects dropout behavior."""
        layer = MultiHeadAttention(embed_dim=64, num_heads=8, dropout_rate=0.5)

        # Build layer
        layer(input_tensor)

        # In training mode, dropout should be active
        output_train = layer(input_tensor, training=True)

        # In inference mode, dropout should be inactive
        output_inference = layer(input_tensor, training=False)

        # Results should be different due to dropout
        assert not tf.reduce_all(tf.equal(output_train, output_inference))

    # ==================== Attention Mask Tests ====================

    def test_no_mask_functionality(self, input_tensor):
        """Test that layer works properly without mask."""
        layer = MultiHeadAttention(embed_dim=64, num_heads=8)

        # Should work without mask
        output = layer(input_tensor)
        assert output.shape == input_tensor.shape

    def test_sequence_level_mask(self, input_tensor):
        """Test sequence-level attention mask."""
        layer = MultiHeadAttention(embed_dim=64, num_heads=8, dropout_rate=0.0)

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
        layer = MultiHeadAttention(embed_dim=64, num_heads=8, dropout_rate=0.0)

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
        layer = MultiHeadAttention(embed_dim=64, num_heads=8, dropout_rate=0.0)

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

    def test_mask_shape_validation(self, input_tensor):
        """Test that invalid mask shapes raise appropriate errors."""
        layer = MultiHeadAttention(embed_dim=64, num_heads=8)

        # Invalid mask shape (5D)
        invalid_mask = tf.ones((2, 8, 10, 10, 3))

        with pytest.raises(ValueError, match="Unsupported attention_mask rank"):
            layer(input_tensor, attention_mask=invalid_mask)

    def test_different_mask_dtypes(self, input_tensor):
        """Test that masks with different dtypes work correctly."""
        layer = MultiHeadAttention(embed_dim=64, num_heads=8)

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

    def test_serialization(self):
        """Test layer serialization and deserialization."""
        # Create and build layer
        original_layer = MultiHeadAttention(
            embed_dim=128,
            num_heads=8,
            dropout_rate=0.1,
            kernel_initializer="he_normal",
            use_bias=True
        )

        input_shape = (None, 20, 128)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate layer
        recreated_layer = MultiHeadAttention.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.embed_dim == original_layer.embed_dim
        assert recreated_layer.num_heads == original_layer.num_heads
        assert recreated_layer.dropout_rate == original_layer.dropout_rate
        assert recreated_layer.use_bias == original_layer.use_bias

        # Check that both layers have same number of weights
        assert len(recreated_layer.weights) == len(original_layer.weights)

    def test_get_config_completeness(self):
        """Test that get_config captures all necessary parameters."""
        layer = MultiHeadAttention(
            embed_dim=256,
            num_heads=16,
            dropout_rate=0.2,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=keras.regularizers.L2(1e-4),
            use_bias=True
        )

        config = layer.get_config()

        # Check all important parameters are in config
        assert "embed_dim" in config
        assert "num_heads" in config
        assert "dropout_rate" in config
        assert "kernel_initializer" in config
        assert "kernel_regularizer" in config
        assert "use_bias" in config

        # Check values
        assert config["embed_dim"] == 256
        assert config["num_heads"] == 16
        assert config["dropout_rate"] == 0.2
        assert config["use_bias"] is True

    # ==================== Model Integration Tests ====================

    def test_model_integration(self, input_tensor):
        """Test the layer in a model context."""
        # Create a model with the custom layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = MultiHeadAttention(embed_dim=64, num_heads=8)(inputs)
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
        x = MultiHeadAttention(embed_dim=64, num_heads=8)(inputs, attention_mask=mask_input)
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
        x = MultiHeadAttention(embed_dim=64, num_heads=8, name="custom_attention")(inputs)
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
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer type is preserved
            assert isinstance(loaded_model.get_layer("custom_attention"), MultiHeadAttention)

    # ==================== Gradient Flow Tests ====================

    def test_gradient_flow(self, input_tensor):
        """Test gradient flow through the layer."""
        layer = MultiHeadAttention(embed_dim=64, num_heads=8)

        # Watch the variables
        with tf.GradientTape() as tape:
            inputs = tf.Variable(input_tensor)
            outputs = layer(inputs)
            loss = tf.reduce_mean(tf.square(outputs))

        # Get gradients
        grads = tape.gradient(loss, layer.trainable_variables)

        # Check gradients exist and are not None
        assert all(g is not None for g in grads)

        # Check gradients have values (not all zeros)
        assert all(tf.reduce_any(g != 0) for g in grads)

    def test_gradient_flow_with_mask(self, input_tensor):
        """Test gradient flow with attention mask."""
        layer = MultiHeadAttention(embed_dim=64, num_heads=8)

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
        assert all(g is not None for g in grads)

        # Check gradients have values (not all zeros)
        assert all(tf.reduce_any(g != 0) for g in grads)

    # ==================== Edge Case Tests ====================

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = MultiHeadAttention(embed_dim=64, num_heads=8)

        test_cases = [
            tf.zeros((2, 10, 64)),  # All zeros
            tf.ones((2, 10, 64)) * 1e-10,  # Very small values
            tf.ones((2, 10, 64)) * 1e5,  # Very large values
            tf.random.normal((2, 10, 64)) * 100,  # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            assert not tf.reduce_any(tf.math.is_nan(output)), "NaN values detected in output"
            assert not tf.reduce_any(tf.math.is_inf(output)), "Inf values detected in output"

    def test_single_sequence_length(self):
        """Test with sequence length of 1."""
        layer = MultiHeadAttention(embed_dim=64, num_heads=8)

        # Single time step
        input_tensor = tf.random.normal([2, 1, 64])
        output = layer(input_tensor)

        assert output.shape == (2, 1, 64)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_large_sequence_length(self):
        """Test with large sequence length."""
        layer = MultiHeadAttention(embed_dim=64, num_heads=8)

        # Large sequence (memory intensive)
        input_tensor = tf.random.normal([1, 1000, 64])
        output = layer(input_tensor)

        assert output.shape == (1, 1000, 64)
        assert not tf.reduce_any(tf.math.is_nan(output))

    # ==================== Performance Tests ====================

    def test_attention_mask_performance(self, input_tensor):
        """Test that attention mask doesn't significantly impact performance."""
        layer = MultiHeadAttention(embed_dim=64, num_heads=8)

        # Create mask
        mask = tf.ones((input_tensor.shape[0], input_tensor.shape[1]))

        # Time without mask
        import time
        start = time.time()
        for _ in range(10):
            _ = layer(input_tensor, training=False)
        time_without_mask = time.time() - start

        # Time with mask
        start = time.time()
        for _ in range(10):
            _ = layer(input_tensor, attention_mask=mask, training=False)
        time_with_mask = time.time() - start

        # Mask should not significantly slow down computation
        # Allow 2x overhead for masking operations
        assert time_with_mask < time_without_mask * 2.0

    def test_different_head_counts(self):
        """Test layer with different numbers of heads."""
        embed_dim = 64
        input_tensor = tf.random.normal([2, 10, embed_dim])

        # Test with different head counts
        for num_heads in [1, 2, 4, 8, 16]:
            if embed_dim % num_heads == 0:  # Only test valid configurations
                layer = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
                output = layer(input_tensor)

                assert output.shape == input_tensor.shape
                assert not tf.reduce_any(tf.math.is_nan(output))