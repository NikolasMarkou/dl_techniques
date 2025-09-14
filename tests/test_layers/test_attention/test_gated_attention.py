"""
Test suite for the GatedAttention layer.

This module provides comprehensive testing for the GatedAttention layer,
covering initialization, forward pass, serialization, gradient flow,
and edge cases following modern Keras 3 testing best practices.
"""

import os
import tempfile
from typing import Any, Dict

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import layers, models, ops

from dl_techniques.layers.attention.gated_attention import GatedAttention


class TestGatedAttention:
    """
    Comprehensive test suite for the GatedAttention layer.
    Tests all aspects of the gated attention mechanism including
    normalization, RoPE, and output gating.
    """

    # --- Fixtures for Reusability ---
    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Provides a standard configuration for a small, testable layer."""
        return {
            "dim": 64,
            "num_heads": 4,
            "max_seq_len": 128,
            "rope_percentage": 0.5,
            "dropout_rate": 0.0,
        }

    @pytest.fixture
    def custom_head_config(self) -> Dict[str, Any]:
        """Provides configuration with custom head dimension."""
        return {
            "dim": 96,
            "num_heads": 6,
            "head_dim": 12,  # Custom head size
            "max_seq_len": 256,
            "rope_percentage": 0.25,
        }

    @pytest.fixture
    def regularized_config(self) -> Dict[str, Any]:
        """Provides configuration with regularization and custom initializers."""
        return {
            "dim": 32,
            "num_heads": 2,
            "max_seq_len": 64,
            "dropout_rate": 0.1,
            "use_bias": True,
            "kernel_initializer": "he_normal",
            "kernel_regularizer": keras.regularizers.L2(1e-4),
            "bias_regularizer": keras.regularizers.L1(1e-5),
        }

    @pytest.fixture
    def sample_input(self) -> tf.Tensor:
        """Provides a standard sample input tensor for testing."""
        return tf.random.normal(shape=(4, 16, 64))

    @pytest.fixture
    def custom_sample_input(self) -> tf.Tensor:
        """Provides sample input matching custom head configuration."""
        return tf.random.normal(shape=(2, 24, 96))

    @pytest.fixture
    def small_sample_input(self) -> tf.Tensor:
        """Provides sample input for regularized configuration."""
        return tf.random.normal(shape=(3, 8, 32))

    @pytest.fixture
    def attention_mask(self) -> tf.Tensor:
        """Provides a sample attention mask for testing masking functionality."""
        # Create a mask where some positions are masked (0) and others are not (1)
        return tf.constant([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=tf.float32)

    # ===============================================
    # 1. Initialization and Build Tests
    # ===============================================
    def test_initialization_defaults(self, layer_config):
        """Tests layer initialization with default parameters."""
        layer = GatedAttention(**layer_config)
        assert not layer.built
        assert layer.dim == 64
        assert layer.num_heads == 4
        assert layer.head_dim == 16  # 64 // 4
        assert layer.max_seq_len == 128
        assert layer.rope_percentage == 0.5
        assert layer.dropout_rate == 0.0
        assert not layer.use_bias
        assert layer.attention_dim == 64  # num_heads * head_dim

    def test_initialization_custom_head_dim(self, custom_head_config):
        """Tests initialization with custom head dimension."""
        layer = GatedAttention(**custom_head_config)
        assert layer.dim == 96
        assert layer.num_heads == 6
        assert layer.head_dim == 12  # Explicitly set
        assert layer.attention_dim == 72  # 6 * 12
        assert layer.rope_percentage == 0.25

    def test_initialization_with_regularization(self, regularized_config):
        """Tests initialization with regularization and custom parameters."""
        layer = GatedAttention(**regularized_config)
        assert layer.use_bias
        assert layer.dropout_rate == 0.1
        assert layer.kernel_regularizer is not None
        assert layer.bias_regularizer is not None

    def test_build_process_standard(self, layer_config, sample_input):
        """Tests that the layer and all its sub-layers are built correctly."""
        layer = GatedAttention(**layer_config)
        assert not layer.built

        # Build the layer by calling it
        output = layer(sample_input)
        assert layer.built
        assert output.shape == sample_input.shape

        # Check that all sub-layers are built
        assert layer.input_linear.built
        assert layer.q_linear.built
        assert layer.k_linear.built
        assert layer.v_linear.built
        assert layer.q_norm.built
        assert layer.k_norm.built
        assert layer.v_norm.built
        assert layer.rope.built
        assert layer.output_gate_linear.built

    def test_build_process_with_dropout(self, small_sample_input):
        """Tests build process with dropout enabled."""
        layer = GatedAttention(dim=32, num_heads=2, max_seq_len=64, dropout_rate=0.1)
        layer(small_sample_input)
        assert layer.built
        assert layer.dropout is not None
        assert layer.dropout.built

    def test_build_process_without_dropout(self, sample_input):
        """Tests build process with dropout disabled."""
        layer = GatedAttention(dim=64, num_heads=4, max_seq_len=128, dropout_rate=0.0)
        layer(sample_input)
        assert layer.dropout is None

    # ===============================================
    # 2. Parameter Validation Tests
    # ===============================================
    def test_parameter_validation_dim_positive(self):
        """Tests that dim must be positive."""
        with pytest.raises(ValueError, match="dim must be positive"):
            GatedAttention(dim=0, num_heads=4, max_seq_len=128)

        with pytest.raises(ValueError, match="dim must be positive"):
            GatedAttention(dim=-64, num_heads=4, max_seq_len=128)

    def test_parameter_validation_num_heads_positive(self):
        """Tests that num_heads must be positive."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            GatedAttention(dim=64, num_heads=0, max_seq_len=128)

        with pytest.raises(ValueError, match="num_heads must be positive"):
            GatedAttention(dim=64, num_heads=-4, max_seq_len=128)

    def test_parameter_validation_head_dim_positive(self):
        """Tests that head_dim must be positive when specified."""
        with pytest.raises(ValueError, match="head_dim must be positive"):
            GatedAttention(dim=64, num_heads=4, head_dim=0, max_seq_len=128)

        with pytest.raises(ValueError, match="head_dim must be positive"):
            GatedAttention(dim=64, num_heads=4, head_dim=-16, max_seq_len=128)

    def test_parameter_validation_divisibility(self):
        """Tests that dim must be divisible by num_heads when head_dim is None."""
        with pytest.raises(
                ValueError, match="dim .* must be divisible by num_heads"
        ):
            GatedAttention(dim=65, num_heads=4, max_seq_len=128)  # 65 is not divisible by 4

    def test_parameter_validation_max_seq_len(self):
        """Tests that max_seq_len must be positive."""
        with pytest.raises(ValueError, match="max_seq_len must be positive"):
            GatedAttention(dim=64, num_heads=4, max_seq_len=0)

        with pytest.raises(ValueError, match="max_seq_len must be positive"):
            GatedAttention(dim=64, num_heads=4, max_seq_len=-128)

    def test_parameter_validation_rope_percentage(self):
        """Tests that rope_percentage must be in (0, 1]."""
        with pytest.raises(ValueError, match="rope_percentage must be in"):
            GatedAttention(dim=64, num_heads=4, max_seq_len=128, rope_percentage=0.0)

        with pytest.raises(ValueError, match="rope_percentage must be in"):
            GatedAttention(dim=64, num_heads=4, max_seq_len=128, rope_percentage=1.5)

        with pytest.raises(ValueError, match="rope_percentage must be in"):
            GatedAttention(dim=64, num_heads=4, max_seq_len=128, rope_percentage=-0.1)

    def test_parameter_validation_dropout_rate(self):
        """Tests that dropout_rate must be in [0, 1]."""
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            GatedAttention(dim=64, num_heads=4, max_seq_len=128, dropout_rate=-0.1)

        with pytest.raises(ValueError, match="dropout_rate must be in"):
            GatedAttention(dim=64, num_heads=4, max_seq_len=128, dropout_rate=1.5)

    def test_build_validation_input_shape(self):
        """Tests build validation for input shape."""
        layer = GatedAttention(dim=64, num_heads=4, max_seq_len=128)

        # Test non-3D input
        with pytest.raises(ValueError, match="Expected 3D input shape"):
            layer.build((32, 64))  # 2D input

        # Test wrong feature dimension
        with pytest.raises(
                ValueError, match="Input feature dimension .* must match dim"
        ):
            layer.build((4, 16, 32))  # 32 != 64

    # ===============================================
    # 3. Forward Pass and Core Behavior Tests
    # ===============================================
    def test_forward_pass_basic(self, layer_config, sample_input):
        """Tests basic forward pass functionality."""
        layer = GatedAttention(**layer_config)
        output = layer(sample_input, training=False)

        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))
        assert not np.any(np.isinf(ops.convert_to_numpy(output)))

    def test_forward_pass_custom_head_dim(self, custom_head_config, custom_sample_input):
        """Tests forward pass with custom head dimension."""
        layer = GatedAttention(**custom_head_config)
        output = layer(custom_sample_input, training=False)

        assert output.shape == custom_sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_forward_pass_with_regularization(self, regularized_config, small_sample_input):
        """Tests forward pass with regularization enabled."""
        layer = GatedAttention(**regularized_config)
        output = layer(small_sample_input, training=True)

        assert output.shape == small_sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_forward_pass_with_attention_mask(self, layer_config, sample_input, attention_mask):
        """Tests forward pass with attention mask."""
        layer = GatedAttention(**layer_config)
        output = layer(sample_input, attention_mask=attention_mask, training=False)

        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_training_vs_inference_mode(self, layer_config, sample_input):
        """Tests that layer behaves differently in training vs inference mode."""
        config = layer_config.copy()
        config["dropout_rate"] = 0.1
        layer = GatedAttention(**config)

        output_train = layer(sample_input, training=True)
        output_infer = layer(sample_input, training=False)

        assert output_train.shape == output_infer.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output_train)))
        assert not np.any(np.isnan(ops.convert_to_numpy(output_infer)))
        # Outputs should be different due to dropout during training
        assert not np.allclose(
            ops.convert_to_numpy(output_train),
            ops.convert_to_numpy(output_infer),
            atol=1e-6,
        )

    def test_deterministic_inference(self, layer_config, sample_input):
        """Tests that inference is deterministic."""
        layer = GatedAttention(**layer_config)

        output1 = layer(sample_input, training=False)
        output2 = layer(sample_input, training=False)

        np.testing.assert_allclose(
            ops.convert_to_numpy(output1),
            ops.convert_to_numpy(output2),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Inference outputs should be identical",
        )

    @pytest.mark.parametrize("rope_percentage", [0.1, 0.25, 0.5, 0.75, 1.0])
    def test_different_rope_percentages(self, rope_percentage, sample_input):
        """Tests forward pass with different RoPE percentages."""
        layer = GatedAttention(
            dim=64, num_heads=4, max_seq_len=128, rope_percentage=rope_percentage
        )
        output = layer(sample_input, training=False)

        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
    def test_different_num_heads(self, num_heads, sample_input):
        """Tests forward pass with different numbers of heads."""
        layer = GatedAttention(dim=64, num_heads=num_heads, max_seq_len=128)
        output = layer(sample_input, training=False)

        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    # ===============================================
    # 4. Serialization Test (The Gold Standard)
    # ===============================================
    def test_full_serialization_cycle_basic(self, layer_config, sample_input):
        """Tests full serialization cycle with basic configuration."""
        inputs = layers.Input(shape=sample_input.shape[1:])
        outputs = GatedAttention(**layer_config)(inputs)
        model = models.Model(inputs, outputs)

        # Get prediction from original model
        original_prediction = model(sample_input, training=False)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_gated_attention_basic.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input, training=False)

            # Verify identical outputs
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6,
                atol=1e-6,
                err_msg="Predictions should match after serialization",
            )

    def test_full_serialization_cycle_custom_head(self, custom_head_config, custom_sample_input):
        """Tests full serialization cycle with custom head dimension."""
        inputs = layers.Input(shape=custom_sample_input.shape[1:])
        outputs = GatedAttention(**custom_head_config)(inputs)
        model = models.Model(inputs, outputs)

        original_prediction = model(custom_sample_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_gated_attention_custom.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(custom_sample_input, training=False)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6,
                atol=1e-6,
                err_msg="Custom head dim serialization should work",
            )

    def test_full_serialization_cycle_with_regularization(
            self, regularized_config, small_sample_input
    ):
        """Tests full serialization cycle with regularization and custom initializers."""
        inputs = layers.Input(shape=small_sample_input.shape[1:])
        outputs = GatedAttention(**regularized_config)(inputs)
        model = models.Model(inputs, outputs)

        original_prediction = model(small_sample_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_gated_attention_regularized.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(small_sample_input, training=False)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6,
                atol=1e-6,
                err_msg="Regularized layer serialization should work",
            )

    def test_full_serialization_cycle_with_mask(
            self, layer_config, sample_input, attention_mask
    ):
        """Tests full serialization cycle with attention mask support."""
        # Create model with two inputs: main input and mask
        main_input = layers.Input(shape=sample_input.shape[1:])
        mask_input = layers.Input(shape=(sample_input.shape[1],))

        attention_layer = GatedAttention(**layer_config)
        outputs = attention_layer(main_input, attention_mask=mask_input)
        model = models.Model([main_input, mask_input], outputs)

        # Get prediction from original model
        original_prediction = model([sample_input, attention_mask], training=False)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_gated_attention_mask.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model([sample_input, attention_mask], training=False)

            # Verify identical outputs
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6,
                atol=1e-6,
                err_msg="Masked attention serialization should work",
            )

    # ===============================================
    # 5. Configuration and Serialization Tests
    # ===============================================
    def test_get_config_completeness(self, layer_config):
        """Tests that get_config contains all __init__ parameters."""
        layer = GatedAttention(**layer_config)
        config = layer.get_config()

        # Check all required parameters are present
        required_params = [
            "dim",
            "num_heads",
            "head_dim",
            "max_seq_len",
            "rope_percentage",
            "dropout_rate",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
        ]

        for param in required_params:
            assert param in config, f"Missing {param} in get_config()"

    def test_from_config_reconstruction(self, regularized_config):
        """Tests that layer can be reconstructed from config."""
        original_layer = GatedAttention(**regularized_config)
        config = original_layer.get_config()
        reconstructed_layer = GatedAttention.from_config(config)

        # Check key parameters match
        assert reconstructed_layer.dim == original_layer.dim
        assert reconstructed_layer.num_heads == original_layer.num_heads
        assert reconstructed_layer.head_dim == original_layer.head_dim
        assert reconstructed_layer.max_seq_len == original_layer.max_seq_len
        assert reconstructed_layer.rope_percentage == original_layer.rope_percentage
        assert reconstructed_layer.dropout_rate == original_layer.dropout_rate
        assert reconstructed_layer.use_bias == original_layer.use_bias

    # ===============================================
    # 6. Gradient and Training Integration Tests
    # ===============================================
    def test_gradient_flow(self, layer_config, sample_input):
        """Tests gradient computation through the layer."""
        layer = GatedAttention(**layer_config)
        x_var = tf.Variable(sample_input)

        with tf.GradientTape() as tape:
            output = layer(x_var, training=True)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert len(gradients) > 0, "No gradients were computed"
        assert all(g is not None for g in gradients), "Some gradients are None"
        assert all(
            not np.any(np.isnan(ops.convert_to_numpy(g))) for g in gradients
        ), "NaN in gradients"

    def test_trainable_variables_count(self, layer_config, sample_input):
        """Tests that the layer has the expected number of trainable variables."""
        layer = GatedAttention(**layer_config)
        layer(sample_input)  # Build the layer

        expected_vars = 8
        actual_vars = len(layer.trainable_variables)
        assert actual_vars == expected_vars, (
            f"Expected {expected_vars} variables, got {actual_vars}"
        )

    def test_model_training_loop_integration(self, layer_config):
        """Tests integration in a standard training loop."""
        model = models.Sequential(
            [
                layers.InputLayer(shape=(16, 64)),
                GatedAttention(**layer_config),
                layers.GlobalAveragePooling1D(),
                layers.Dense(10),
            ]
        )

        model.compile(
            "adam",
            keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            jit_compile=False,
        )

        # Generate dummy data
        x_train = tf.random.normal((32, 16, 64))
        y_train = tf.random.uniform([32], 0, 10, dtype=tf.int32)

        # Train for one epoch
        history = model.fit(x_train, y_train, epochs=1, batch_size=8, verbose=0)

        assert "loss" in history.history
        assert not np.isnan(history.history["loss"][0]), "Loss became NaN during training"

    def test_stacked_layers(self, sample_input):
        """Tests stacking multiple GatedAttention layers."""
        inputs = layers.Input(shape=sample_input.shape[1:])
        x = GatedAttention(dim=64, num_heads=4, max_seq_len=128)(inputs)
        x = GatedAttention(dim=64, num_heads=8, max_seq_len=128)(x)
        outputs = layers.GlobalAveragePooling1D()(x)

        model = models.Model(inputs, outputs)
        prediction = model(sample_input, training=False)

        assert prediction.shape == (sample_input.shape[0], 64)
        assert not np.any(np.isnan(ops.convert_to_numpy(prediction)))

    # ===============================================
    # 7. Edge Cases and Robustness Tests
    # ===============================================
    def test_small_sequence_length(self):
        """Tests layer with very small sequence length."""
        layer = GatedAttention(dim=32, num_heads=2, max_seq_len=64)
        small_input = tf.random.normal((2, 3, 32))  # Very short sequence

        output = layer(small_input, training=False)
        assert output.shape == small_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_single_head(self, sample_input):
        """Tests layer with single attention head."""
        layer = GatedAttention(dim=64, num_heads=1, max_seq_len=128)
        output = layer(sample_input, training=False)

        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_max_rope_percentage(self, sample_input):
        """Tests layer with maximum RoPE percentage (1.0)."""
        layer = GatedAttention(
            dim=64,
            num_heads=4,
            max_seq_len=128,
            rope_percentage=1.0,  # Apply RoPE to all dimensions
        )
        output = layer(sample_input, training=False)

        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_batch_size_one(self):
        """Tests layer with batch size 1."""
        layer = GatedAttention(dim=32, num_heads=2, max_seq_len=64)
        single_batch_input = tf.random.normal((1, 10, 32))

        output = layer(single_batch_input, training=False)
        assert output.shape == single_batch_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_compute_output_shape(self, layer_config):
        """Tests compute_output_shape method."""
        layer = GatedAttention(**layer_config)
        input_shape = (None, 20, 64)
        output_shape = layer.compute_output_shape(input_shape)

        assert output_shape == input_shape

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_input, training):
        """Tests behavior in different training modes."""
        config = layer_config.copy()
        config["dropout_rate"] = 0.1
        layer = GatedAttention(**config)

        output = layer(sample_input, training=training)
        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_long_sequence(self):
        """Tests layer with long sequences."""
        layer = GatedAttention(dim=32, num_heads=2, max_seq_len=512)
        long_input = tf.random.normal((2, 256, 32))  # Long sequence

        output = layer(long_input, training=False)
        assert output.shape == long_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    # ===============================================
    # 8. Attention Mechanism Tests
    # ===============================================
    def test_attention_mask_effect(self, layer_config, sample_input, attention_mask):
        """Tests that attention mask affects the output."""
        layer = GatedAttention(**layer_config)

        # Run with and without mask
        output_without_mask = layer(sample_input, training=False)
        output_with_mask = layer(sample_input, attention_mask=attention_mask, training=False)

        # Outputs should be different due to masking
        assert not np.allclose(
            ops.convert_to_numpy(output_without_mask),
            ops.convert_to_numpy(output_with_mask),
            atol=1e-6,
        )

    def test_output_gating_effect(self, layer_config, sample_input):
        """Tests that output gating affects the final output."""
        layer = GatedAttention(**layer_config)
        output = layer(sample_input, training=False)

        # The output should be bounded due to sigmoid gating
        output_np = ops.convert_to_numpy(output)
        assert not np.any(np.abs(output_np) > 100), "Output values seem unbounded"

    def test_rope_application(self, layer_config, sample_input):
        """Tests that RoPE is being applied to queries and keys."""
        # Create two layers with different RoPE percentages
        layer1 = GatedAttention(dim=64, num_heads=4, max_seq_len=128, rope_percentage=0.1)
        layer2 = GatedAttention(dim=64, num_heads=4, max_seq_len=128, rope_percentage=0.9)

        output1 = layer1(sample_input, training=False)
        output2 = layer2(sample_input, training=False)

        # Outputs should be different due to different RoPE percentages
        assert not np.allclose(
            ops.convert_to_numpy(output1),
            ops.convert_to_numpy(output2),
            atol=1e-6,
        )

    def test_scaled_dot_product_attention_numerical_stability(self, layer_config):
        """Tests numerical stability of scaled dot-product attention."""
        layer = GatedAttention(**layer_config)

        # Create input with large values that could cause numerical issues
        large_input = tf.random.normal((2, 8, 64)) * 10.0

        output = layer(large_input, training=False)

        # Check that output doesn't contain NaN or Inf despite large inputs
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))
        assert not np.any(np.isinf(ops.convert_to_numpy(output)))

if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v"])
