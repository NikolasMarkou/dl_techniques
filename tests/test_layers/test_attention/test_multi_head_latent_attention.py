"""Test suite for MultiHeadLatentAttention (MLA) layer.

This module contains comprehensive tests for the MultiHeadLatentAttention layer,
which implements the MLA mechanism from DeepSeek-V2 with low-rank KV compression
and decoupled RoPE for efficient inference.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

from dl_techniques.layers.attention.multi_head_latent_attention import (
    MultiHeadLatentAttention
)


class TestMultiHeadLatentAttention:
    """Test suite for MultiHeadLatentAttention layer."""

    # ==================== Fixtures ====================

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return tf.random.normal([2, 16, 256])  # (batch_size, seq_len, dim)

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return MultiHeadLatentAttention(
            dim=256,
            num_heads=8,
            kv_latent_dim=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32
        )

    @pytest.fixture
    def layer_with_q_compression(self):
        """Create a layer instance with query compression enabled."""
        return MultiHeadLatentAttention(
            dim=256,
            num_heads=8,
            kv_latent_dim=64,
            q_latent_dim=128,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32
        )

    @pytest.fixture
    def different_configs(self):
        """Provide different layer configurations for testing."""
        return [
            {
                "dim": 128,
                "num_heads": 4,
                "kv_latent_dim": 32,
                "qk_nope_head_dim": 16,
                "qk_rope_head_dim": 8,
                "v_head_dim": 16
            },
            {
                "dim": 256,
                "num_heads": 8,
                "kv_latent_dim": 64,
                "q_latent_dim": 128,
                "qk_nope_head_dim": 32,
                "qk_rope_head_dim": 16,
                "v_head_dim": 32,
                "dropout_rate": 0.1
            },
            {
                "dim": 512,
                "num_heads": 16,
                "kv_latent_dim": 128,
                "q_latent_dim": 256,
                "qk_nope_head_dim": 64,
                "qk_rope_head_dim": 32,
                "v_head_dim": 64,
                "use_bias": True
            },
            {
                "dim": 256,
                "num_heads": 8,
                "kv_latent_dim": 64,
                "qk_nope_head_dim": 32,
                "qk_rope_head_dim": 16,
                "v_head_dim": 32,
                "dropout_rate": 0.2,
                "use_bias": True,
                "max_seq_len": 2048,
                "rope_theta": 10000.0
            },
        ]

    # ==================== Initialization Tests ====================

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = MultiHeadLatentAttention(
            dim=256,
            num_heads=8,
            kv_latent_dim=64
        )

        assert layer.dim == 256
        assert layer.num_heads == 8
        assert layer.kv_latent_dim == 64
        assert layer.q_latent_dim is None  # Default: no query compression
        assert layer.qk_nope_head_dim == 128  # Default
        assert layer.qk_rope_head_dim == 64  # Default
        assert layer.v_head_dim == 128  # Default
        assert layer.dropout_rate == 0.0
        assert layer.use_bias is False
        assert layer.max_seq_len == 4096
        assert layer.rope_theta == 10000.0
        assert layer.rope_percentage == 1.0
        assert layer.normalization_type == "rms_norm"
        assert layer.kernel_regularizer is None

    def test_initialization_with_q_compression(self):
        """Test initialization with query compression enabled."""
        layer = MultiHeadLatentAttention(
            dim=256,
            num_heads=8,
            kv_latent_dim=64,
            q_latent_dim=128
        )

        assert layer.q_latent_dim == 128
        assert hasattr(layer, 'q_down_proj')
        assert hasattr(layer, 'q_norm')
        assert hasattr(layer, 'q_up_proj')

    def test_initialization_without_q_compression(self):
        """Test initialization without query compression (DeepSeek-V2 Lite style)."""
        layer = MultiHeadLatentAttention(
            dim=256,
            num_heads=8,
            kv_latent_dim=64,
            q_latent_dim=None
        )

        assert layer.q_latent_dim is None
        assert hasattr(layer, 'query_proj')
        assert not hasattr(layer, 'q_down_proj')

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        layer = MultiHeadLatentAttention(
            dim=512,
            num_heads=16,
            kv_latent_dim=128,
            q_latent_dim=256,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
            dropout_rate=0.1,
            use_bias=True,
            max_seq_len=8192,
            rope_theta=50000.0,
            rope_percentage=0.5,
            normalization_type="rms_norm",
            kernel_initializer="glorot_uniform",
            kernel_regularizer=custom_regularizer
        )

        assert layer.dim == 512
        assert layer.num_heads == 16
        assert layer.kv_latent_dim == 128
        assert layer.q_latent_dim == 256
        assert layer.qk_nope_head_dim == 64
        assert layer.qk_rope_head_dim == 32
        assert layer.v_head_dim == 64
        assert layer.dropout_rate == 0.1
        assert layer.use_bias is True
        assert layer.max_seq_len == 8192
        assert layer.rope_theta == 50000.0
        assert layer.rope_percentage == 0.5
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert layer.kernel_regularizer == custom_regularizer

    def test_invalid_dim_negative(self):
        """Test that negative dim raises ValueError."""
        with pytest.raises(ValueError, match="dim must be positive, got -256"):
            MultiHeadLatentAttention(dim=-256, num_heads=8, kv_latent_dim=64)

    def test_invalid_num_heads_negative(self):
        """Test that negative num_heads raises ValueError."""
        with pytest.raises(ValueError, match="num_heads must be positive, got -8"):
            MultiHeadLatentAttention(dim=256, num_heads=-8, kv_latent_dim=64)

    def test_invalid_kv_latent_dim_negative(self):
        """Test that negative kv_latent_dim raises ValueError."""
        with pytest.raises(ValueError, match="kv_latent_dim must be positive, got -64"):
            MultiHeadLatentAttention(dim=256, num_heads=8, kv_latent_dim=-64)

    def test_invalid_dropout_rate(self):
        """Test that invalid dropout_rate raises ValueError."""
        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1, got 1.5"):
            MultiHeadLatentAttention(
                dim=256, num_heads=8, kv_latent_dim=64, dropout_rate=1.5
            )

    def test_invalid_dropout_rate_negative(self):
        """Test that negative dropout_rate raises ValueError."""
        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1, got -0.1"):
            MultiHeadLatentAttention(
                dim=256, num_heads=8, kv_latent_dim=64, dropout_rate=-0.1
            )

    # ==================== Build Process Tests ====================

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer builds properly."""
        layer_instance(input_tensor)  # Forward pass triggers build

        assert layer_instance.built is True

    def test_build_process_with_q_compression(self, input_tensor, layer_with_q_compression):
        """Test that layer with query compression builds properly."""
        layer_with_q_compression(input_tensor)

        assert layer_with_q_compression.built is True
        assert layer_with_q_compression.q_down_proj.built is True
        assert layer_with_q_compression.q_norm.built is True
        assert layer_with_q_compression.q_up_proj.built is True

    def test_build_input_shape_validation(self):
        """Test input shape validation in build."""
        layer = MultiHeadLatentAttention(dim=256, num_heads=8, kv_latent_dim=64)

        # Test invalid input shape (2D instead of 3D)
        with pytest.raises(ValueError, match="Expected 3D input shape"):
            layer.build((32, 256))  # Missing sequence dimension

    def test_explicit_build(self, input_tensor, layer_instance):
        """Test that build method works when called explicitly."""
        layer_instance.build(input_tensor.shape)

        assert layer_instance.built is True

    def test_build_with_list_input_shape(self):
        """Test build with list input shape for cross-attention."""
        layer = MultiHeadLatentAttention(dim=256, num_heads=8, kv_latent_dim=64)

        # Cross-attention style: [query_shape, kv_shape]
        q_shape = (2, 10, 256)
        kv_shape = (2, 20, 256)
        layer.build([q_shape, kv_shape])

        assert layer.built is True

    # ==================== Output Shape Tests ====================

    def test_output_shapes(self, different_configs):
        """Test that output shapes are computed correctly."""
        for config in different_configs:
            layer = MultiHeadLatentAttention(**config)

            # Test different sequence lengths
            for seq_len in [8, 16, 32]:
                input_shape = (2, seq_len, config["dim"])
                input_tensor = tf.random.normal(input_shape)

                output = layer(input_tensor)
                expected_shape = input_shape

                assert output.shape == expected_shape, (
                    f"Expected shape {expected_shape}, got {output.shape} "
                    f"for config {config}"
                )

                # Test compute_output_shape separately
                computed_shape = layer.compute_output_shape(input_shape)
                assert computed_shape == expected_shape

    def test_batch_size_flexibility(self, layer_instance):
        """Test that the layer works with different batch sizes."""
        dim = layer_instance.dim

        for batch_size in [1, 4, 16]:
            input_tensor = tf.random.normal([batch_size, 16, dim])
            output = layer_instance(input_tensor)
            assert output.shape == (batch_size, 16, dim)

    def test_cross_attention_output_shape(self):
        """Test output shape for cross-attention (different Q and KV lengths)."""
        layer = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64
        )

        query = tf.random.normal([2, 10, 256])
        kv = tf.random.normal([2, 20, 256])

        output = layer(query, kv_input=kv)

        # Output shape should match query sequence length
        assert output.shape == (2, 10, 256)

    # ==================== Forward Pass Tests ====================

    def test_forward_pass_basic(self, input_tensor, layer_instance):
        """Test basic forward pass functionality."""
        output = layer_instance(input_tensor)

        # Basic sanity checks
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))
        assert output.shape == input_tensor.shape

    def test_forward_pass_with_q_compression(self, input_tensor, layer_with_q_compression):
        """Test forward pass with query compression enabled."""
        output = layer_with_q_compression(input_tensor)

        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))
        assert output.shape == input_tensor.shape

    def test_forward_pass_deterministic(self):
        """Test forward pass with deterministic inputs."""
        layer = MultiHeadLatentAttention(
            dim=256,
            num_heads=8,
            kv_latent_dim=64,
            kernel_initializer="ones",
            dropout_rate=0.0
        )

        # Use deterministic input
        input_tensor = tf.ones([1, 8, 256])

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
        layer = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64, dropout_rate=0.5
        )

        # Build layer first
        layer(input_tensor, training=False)

        # Set different random seeds to ensure different dropout patterns
        tf.random.set_seed(42)
        output_train = layer(input_tensor, training=True)

        tf.random.set_seed(42)  # Same seed for consistency
        output_inference = layer(input_tensor, training=False)

        # Results should be different due to dropout in training mode
        assert not tf.reduce_all(tf.equal(output_train, output_inference))

    def test_cross_attention_forward_pass(self):
        """Test forward pass for cross-attention."""
        layer = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64
        )

        query = tf.random.normal([2, 10, 256])
        kv = tf.random.normal([2, 20, 256])

        output = layer(query, kv_input=kv)

        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))

    # ==================== Attention Mask Tests ====================

    def test_no_mask_functionality(self, input_tensor, layer_instance):
        """Test that layer works properly without mask."""
        output = layer_instance(input_tensor)
        assert output.shape == input_tensor.shape

    def test_sequence_level_mask(self, input_tensor):
        """Test sequence-level attention mask."""
        layer = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64, dropout_rate=0.0
        )

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
        layer = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64, dropout_rate=0.0
        )

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
        layer = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64, dropout_rate=0.0
        )

        seq_len = input_tensor.shape[1]
        batch_size = input_tensor.shape[0]
        num_heads = 8

        # Create different masks for different heads
        mask = tf.ones((batch_size, num_heads, seq_len, seq_len))
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
        layer = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64
        )

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
        layer = MultiHeadLatentAttention(
            dim=512,
            num_heads=16,
            kv_latent_dim=128,
            q_latent_dim=256,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
            dropout_rate=0.2,
            use_bias=True,
            max_seq_len=8192,
            rope_theta=50000.0,
            rope_percentage=0.5,
            normalization_type="rms_norm",
            kernel_initializer="glorot_uniform",
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        config = layer.get_config()

        # Check all important parameters are in config
        required_keys = [
            "dim", "num_heads", "kv_latent_dim", "q_latent_dim",
            "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
            "dropout_rate", "use_bias", "max_seq_len", "rope_theta",
            "rope_percentage", "normalization_type",
            "kernel_initializer", "kernel_regularizer"
        ]
        for key in required_keys:
            assert key in config, f"Missing key {key} in config"

        # Check values
        assert config["dim"] == 512
        assert config["num_heads"] == 16
        assert config["kv_latent_dim"] == 128
        assert config["q_latent_dim"] == 256
        assert config["qk_nope_head_dim"] == 64
        assert config["qk_rope_head_dim"] == 32
        assert config["v_head_dim"] == 64
        assert config["dropout_rate"] == 0.2
        assert config["use_bias"] is True
        assert config["max_seq_len"] == 8192
        assert config["rope_theta"] == 50000.0
        assert config["rope_percentage"] == 0.5

    def test_layer_recreation_from_config(self):
        """Test recreating layer from config."""
        original_layer = MultiHeadLatentAttention(
            dim=256,
            num_heads=8,
            kv_latent_dim=64,
            q_latent_dim=128,
            dropout_rate=0.1,
            use_bias=True
        )

        # Get config and recreate layer
        config = original_layer.get_config()
        recreated_layer = MultiHeadLatentAttention.from_config(config)

        # Check configuration matches
        assert recreated_layer.dim == original_layer.dim
        assert recreated_layer.num_heads == original_layer.num_heads
        assert recreated_layer.kv_latent_dim == original_layer.kv_latent_dim
        assert recreated_layer.q_latent_dim == original_layer.q_latent_dim
        assert recreated_layer.dropout_rate == original_layer.dropout_rate
        assert recreated_layer.use_bias == original_layer.use_bias

    def test_serialization_with_build(self, input_tensor):
        """Test serialization after building the layer."""
        original_layer = MultiHeadLatentAttention(
            dim=256,
            num_heads=8,
            kv_latent_dim=64,
            dropout_rate=0.1,
            use_bias=True
        )

        # Build the layer
        original_layer(input_tensor)

        # Get config and recreate
        config = original_layer.get_config()
        recreated_layer = MultiHeadLatentAttention.from_config(config)

        # Build recreated layer
        recreated_layer(input_tensor)

        # Both layers should have same number of weights
        assert len(recreated_layer.weights) == len(original_layer.weights)

    # ==================== Model Integration Tests ====================

    def test_model_integration(self, input_tensor):
        """Test the layer in a model context."""
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64
        )(inputs)
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
        inputs = keras.Input(shape=input_tensor.shape[1:])
        mask_input = keras.Input(shape=(input_tensor.shape[1],))

        x = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64
        )(inputs, attention_mask=mask_input)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=[inputs, mask_input], outputs=outputs)

        # Create mask
        mask = tf.ones((input_tensor.shape[0], input_tensor.shape[1]))

        # Test forward pass
        y_pred = model([input_tensor, mask], training=False)
        assert y_pred.shape == (input_tensor.shape[0], 10)

    def test_stacked_mla_layers(self, input_tensor):
        """Test stacking multiple MLA layers."""
        inputs = keras.Input(shape=input_tensor.shape[1:])

        x = inputs
        for i in range(3):
            x = MultiHeadLatentAttention(
                dim=256, num_heads=8, kv_latent_dim=64,
                name=f"mla_{i}"
            )(x)
            x = keras.layers.LayerNormalization()(x)

        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Test forward pass
        y_pred = model(input_tensor, training=False)
        assert y_pred.shape == (input_tensor.shape[0], 10)
        assert not tf.reduce_any(tf.math.is_nan(y_pred))

    # ==================== Model Save/Load Tests ====================

    def test_model_save_load(self, input_tensor):
        """Test saving and loading a model with the custom layer."""
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64,
            name="custom_mla"
        )(inputs)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate a prediction before saving
        original_prediction = model.predict(input_tensor, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"MultiHeadLatentAttention": MultiHeadLatentAttention}
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
            assert isinstance(
                loaded_model.get_layer("custom_mla"),
                MultiHeadLatentAttention
            )

    def test_model_save_load_with_q_compression(self, input_tensor):
        """Test saving and loading a model with query compression enabled."""
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64,
            q_latent_dim=128,
            name="mla_with_q_compression"
        )(inputs)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        original_prediction = model.predict(input_tensor, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")
            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"MultiHeadLatentAttention": MultiHeadLatentAttention}
            )

            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            np.testing.assert_allclose(
                original_prediction, loaded_prediction,
                rtol=1e-5, atol=1e-5,
                err_msg="Predictions should match after save/load with q_compression"
            )

    # ==================== Gradient Flow Tests ====================

    def test_gradient_flow(self, input_tensor, layer_instance):
        """Test gradient flow through the layer."""
        with tf.GradientTape() as tape:
            inputs = tf.Variable(input_tensor)
            outputs = layer_instance(inputs)
            loss = tf.reduce_mean(tf.square(outputs))

        # Get gradients
        grads = tape.gradient(loss, layer_instance.trainable_variables)

        # Check gradients exist and are not None
        assert all(g is not None for g in grads), "All gradients should be non-None"

        # Check gradients have values (not all zeros)
        assert all(tf.reduce_any(g != 0) for g in grads), (
            "Gradients should have non-zero values"
        )

    def test_gradient_flow_with_mask(self, input_tensor, layer_instance):
        """Test gradient flow with attention mask."""
        mask = tf.ones((input_tensor.shape[0], input_tensor.shape[1]))

        with tf.GradientTape() as tape:
            inputs = tf.Variable(input_tensor)
            outputs = layer_instance(inputs, attention_mask=mask)
            loss = tf.reduce_mean(tf.square(outputs))

        grads = tape.gradient(loss, layer_instance.trainable_variables)

        assert all(g is not None for g in grads), "All gradients should be non-None"
        assert all(tf.reduce_any(g != 0) for g in grads), (
            "Gradients should have non-zero values"
        )

    def test_gradient_flow_with_q_compression(self, input_tensor, layer_with_q_compression):
        """Test gradient flow with query compression."""
        with tf.GradientTape() as tape:
            inputs = tf.Variable(input_tensor)
            outputs = layer_with_q_compression(inputs)
            loss = tf.reduce_mean(tf.square(outputs))

        grads = tape.gradient(loss, layer_with_q_compression.trainable_variables)

        assert all(g is not None for g in grads), "All gradients should be non-None"
        assert all(tf.reduce_any(g != 0) for g in grads), (
            "Gradients should have non-zero values"
        )

    # ==================== Edge Case Tests ====================

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64
        )

        test_cases = [
            tf.zeros((2, 16, 256)),  # All zeros
            tf.ones((2, 16, 256)) * 1e-10,  # Very small values
            tf.ones((2, 16, 256)) * 1e3,  # Large values
            tf.random.normal((2, 16, 256)) * 10,  # Large random values
        ]

        for i, test_input in enumerate(test_cases):
            output = layer(test_input)

            assert not tf.reduce_any(tf.math.is_nan(output)), (
                f"NaN values detected in output for test case {i}"
            )
            assert not tf.reduce_any(tf.math.is_inf(output)), (
                f"Inf values detected in output for test case {i}"
            )

    def test_single_sequence_length(self):
        """Test with sequence length of 1."""
        layer = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64
        )

        input_tensor = tf.random.normal([2, 1, 256])
        output = layer(input_tensor)

        assert output.shape == (2, 1, 256)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_large_sequence_length(self):
        """Test with large sequence length."""
        layer = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64
        )

        input_tensor = tf.random.normal([1, 256, 256])
        output = layer(input_tensor)

        assert output.shape == (1, 256, 256)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_single_head(self):
        """Test with single attention head."""
        layer = MultiHeadLatentAttention(
            dim=256, num_heads=1, kv_latent_dim=64
        )

        input_tensor = tf.random.normal([2, 16, 256])
        output = layer(input_tensor)

        assert output.shape == (2, 16, 256)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_single_batch(self):
        """Test with batch size of 1."""
        layer = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64
        )

        input_tensor = tf.random.normal([1, 16, 256])
        output = layer(input_tensor)

        assert output.shape == (1, 16, 256)
        assert not tf.reduce_any(tf.math.is_nan(output))

    # ==================== MLA-Specific Tests ====================

    def test_kv_compression_path(self, input_tensor):
        """Test that KV compression path is properly executed."""
        layer = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64
        )

        # Build the layer
        layer(input_tensor)

        # Verify KV compression layers exist and are built
        assert hasattr(layer, 'kv_down_proj')
        assert hasattr(layer, 'kv_norm')
        assert hasattr(layer, 'kv_up_proj')
        assert hasattr(layer, 'k_rope_proj')

        assert layer.kv_down_proj.built
        assert layer.kv_norm.built
        assert layer.kv_up_proj.built
        assert layer.k_rope_proj.built

    def test_rope_integration(self, input_tensor):
        """Test that RoPE is properly integrated."""
        layer = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64,
            max_seq_len=4096, rope_theta=10000.0
        )

        layer(input_tensor)

        # Verify RoPE layer exists and is built
        assert hasattr(layer, 'rope')
        assert layer.rope.built

    def test_different_normalization_types(self, input_tensor):
        """Test layer with different normalization types."""
        for norm_type in ["rms_norm", "layer_norm"]:
            layer = MultiHeadLatentAttention(
                dim=256, num_heads=8, kv_latent_dim=64,
                normalization_type=norm_type
            )

            output = layer(input_tensor)

            assert output.shape == input_tensor.shape
            assert not tf.reduce_any(tf.math.is_nan(output))

    def test_decoupled_rope_key_sharing(self, input_tensor):
        """Test that k_pe is properly shared across heads (decoupled RoPE)."""
        layer = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64,
            qk_rope_head_dim=16
        )

        layer(input_tensor)

        # k_rope_proj should output rope_dim, not num_heads * rope_dim
        # because K_pe is shared across heads
        assert layer.k_rope_proj.units == 16  # qk_rope_head_dim

    def test_output_projection_dimensions(self, input_tensor):
        """Test that output projection has correct dimensions."""
        v_head_dim = 32
        num_heads = 8
        dim = 256

        layer = MultiHeadLatentAttention(
            dim=dim, num_heads=num_heads, kv_latent_dim=64,
            v_head_dim=v_head_dim
        )

        layer(input_tensor)

        # Output projection should map from (num_heads * v_head_dim) to dim
        assert layer.output_proj.units == dim

    def test_attention_scale_factor(self):
        """Test that attention scale factor is correctly computed."""
        qk_nope_head_dim = 64
        qk_rope_head_dim = 32

        layer = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim
        )

        expected_scale = 1.0 / np.sqrt(qk_nope_head_dim + qk_rope_head_dim)
        assert np.isclose(layer._scale, expected_scale)

    # ==================== Performance Tests ====================

    def test_attention_mask_performance(self, input_tensor, layer_instance):
        """Test that attention mask doesn't significantly impact performance."""
        mask = tf.ones((input_tensor.shape[0], input_tensor.shape[1]))

        import time

        # Warm up
        _ = layer_instance(input_tensor, training=False)

        # Time without mask
        start = time.time()
        for _ in range(5):
            _ = layer_instance(input_tensor, training=False)
        time_without_mask = time.time() - start

        # Time with mask
        start = time.time()
        for _ in range(5):
            _ = layer_instance(input_tensor, attention_mask=mask, training=False)
        time_with_mask = time.time() - start

        # Mask should not significantly slow down computation
        assert time_with_mask < time_without_mask * 3.0

    def test_different_head_counts(self):
        """Test layer with different numbers of heads."""
        dim = 256
        input_tensor = tf.random.normal([2, 16, dim])

        for num_heads in [1, 2, 4, 8, 16]:
            layer = MultiHeadLatentAttention(
                dim=dim, num_heads=num_heads, kv_latent_dim=64
            )
            output = layer(input_tensor)

            assert output.shape == input_tensor.shape
            assert not tf.reduce_any(tf.math.is_nan(output))

    def test_different_latent_dimensions(self):
        """Test layer with different latent dimensions."""
        dim = 256
        input_tensor = tf.random.normal([2, 16, dim])

        for kv_latent_dim in [32, 64, 128, 256]:
            layer = MultiHeadLatentAttention(
                dim=dim, num_heads=8, kv_latent_dim=kv_latent_dim
            )
            output = layer(input_tensor)

            assert output.shape == input_tensor.shape
            assert not tf.reduce_any(tf.math.is_nan(output))

    # ==================== Trainable Variables Tests ====================

    def test_trainable_variables_count(self, input_tensor):
        """Test that all expected trainable variables are created."""
        layer = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64,
            use_bias=False
        )
        layer(input_tensor)

        # Without query compression:
        # - query_proj kernel
        # - kv_down_proj kernel
        # - kv_norm scale
        # - kv_up_proj kernel
        # - k_rope_proj kernel
        # - output_proj kernel
        # - rope layer variables

        trainable_vars = layer.trainable_variables
        assert len(trainable_vars) > 0, "Layer should have trainable variables"

    def test_trainable_variables_with_q_compression(self, input_tensor):
        """Test trainable variables with query compression enabled."""
        layer = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64,
            q_latent_dim=128, use_bias=False
        )
        layer(input_tensor)

        # With query compression:
        # - q_down_proj kernel
        # - q_norm scale
        # - q_up_proj kernel
        # + all KV path variables

        trainable_vars = layer.trainable_variables
        assert len(trainable_vars) > 0, "Layer should have trainable variables"

    def test_bias_variables(self, input_tensor):
        """Test that bias variables are created when use_bias=True."""
        layer_with_bias = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64, use_bias=True
        )
        layer_without_bias = MultiHeadLatentAttention(
            dim=256, num_heads=8, kv_latent_dim=64, use_bias=False
        )

        layer_with_bias(input_tensor)
        layer_without_bias(input_tensor)

        # Layer with bias should have more variables
        assert len(layer_with_bias.trainable_variables) > len(
            layer_without_bias.trainable_variables
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])