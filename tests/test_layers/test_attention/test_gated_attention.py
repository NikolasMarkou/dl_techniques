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
        """Provides a standard configuration where attention_dim == dim."""
        return {
            "dim": 64,
            "num_heads": 4,
            "max_seq_len": 128,
            "rope_percentage": 0.5,
            "dropout_rate": 0.0,
        }

    @pytest.fixture
    def custom_head_config(self) -> Dict[str, Any]:
        """Provides configuration with custom head dimension where attention_dim != dim."""
        return {
            "dim": 96,
            "num_heads": 6,
            "head_dim": 12,  # Custom head size -> attention_dim = 72
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
    def padding_attention_mask(self, sample_input) -> tf.Tensor:
        """Provides a 2D sample padding mask for testing."""
        batch_size, seq_len = sample_input.shape[0], sample_input.shape[1]
        mask = np.ones((batch_size, seq_len), dtype="float32")
        # Mask out the second half of the sequence for the first batch item
        mask[0, seq_len // 2:] = 0
        return tf.constant(mask)

    @pytest.fixture
    def causal_attention_mask(self, sample_input) -> tf.Tensor:
        """Provides a 3D sample causal mask for testing."""
        seq_len = sample_input.shape[1]
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return tf.expand_dims(mask, 0)  # Shape: (1, seq_len, seq_len)

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
        assert isinstance(layer.kernel_regularizer, keras.regularizers.L2)
        assert isinstance(layer.bias_regularizer, keras.regularizers.L1)

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

    def test_output_proj_is_none_when_attention_dim_matches(self, layer_config, sample_input):
        """Tests that output_proj is None when attention_dim == dim."""
        layer = GatedAttention(**layer_config)
        assert layer.attention_dim == layer.dim
        layer(sample_input)  # Build
        assert layer.output_proj is None

    def test_output_proj_creation_when_attention_dim_mismatch(self, custom_head_config, custom_sample_input):
        """Tests that output_proj is created when attention_dim != dim."""
        layer = GatedAttention(**custom_head_config)
        assert layer.attention_dim != layer.dim
        layer(custom_sample_input)  # Build
        assert layer.output_proj is not None
        assert layer.output_proj.built

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
        """Tests forward pass with custom head dimension and output projection."""
        layer = GatedAttention(**custom_head_config)
        output = layer(custom_sample_input, training=False)

        assert output.shape == custom_sample_input.shape  # Verifies projection back to `dim`
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_forward_pass_with_regularization(self, regularized_config, small_sample_input):
        """Tests forward pass with regularization enabled."""
        layer = GatedAttention(**regularized_config)
        output = layer(small_sample_input, training=True)

        assert output.shape == small_sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_forward_pass_with_padding_mask(self, layer_config, sample_input, padding_attention_mask):
        """Tests forward pass with a 2D padding mask."""
        layer = GatedAttention(**layer_config)
        output = layer(sample_input, attention_mask=padding_attention_mask, training=False)

        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_forward_pass_with_causal_mask(self, layer_config, sample_input, causal_attention_mask):
        """Tests forward pass with a 3D causal mask."""
        layer = GatedAttention(**layer_config)
        output = layer(sample_input, attention_mask=causal_attention_mask, training=False)

        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_training_vs_inference_mode(self, layer_config, sample_input):
        """Tests that layer behaves differently in training vs inference mode due to dropout."""
        config = {**layer_config, "dropout_rate": 0.1}
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

        original_prediction = model(sample_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_gated_attention_basic.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input, training=False)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6
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
                rtol=1e-6, atol=1e-6
            )

    def test_full_serialization_cycle_with_padding_mask(self, layer_config, sample_input, padding_attention_mask):
        """Tests full serialization cycle with a 2D padding mask."""
        main_input = layers.Input(shape=sample_input.shape[1:])
        mask_input = layers.Input(shape=(sample_input.shape[1],))
        outputs = GatedAttention(**layer_config)(main_input, attention_mask=mask_input)
        model = models.Model([main_input, mask_input], outputs)

        original_prediction = model([sample_input, padding_attention_mask], training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_gated_attention_padding_mask.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model([sample_input, padding_attention_mask], training=False)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6
            )

    def test_full_serialization_cycle_with_causal_mask(self, layer_config, sample_input, causal_attention_mask):
        """Tests full serialization cycle with a 3D causal mask."""
        seq_len = sample_input.shape[1]
        main_input = layers.Input(shape=(seq_len, layer_config["dim"]))
        # FIX: The shape of a single mask sample is (seq_len, seq_len).
        # The provided tensor has shape (1, seq_len, seq_len), which is a batch of 1.
        # This is compatible with an input layer expecting samples of shape (seq_len, seq_len).
        mask_input = layers.Input(shape=(seq_len, seq_len))
        outputs = GatedAttention(**layer_config)(main_input, attention_mask=mask_input)
        model = models.Model([main_input, mask_input], outputs)

        original_prediction = model([sample_input, causal_attention_mask], training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_gated_attention_causal_mask.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model([sample_input, causal_attention_mask], training=False)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6
            )

    # ===============================================
    # 5. Configuration and Serialization Tests
    # ===============================================
    def test_get_config_completeness(self, regularized_config):
        """Tests that get_config contains all __init__ parameters."""
        layer = GatedAttention(**regularized_config)
        config = layer.get_config()

        # Check all required parameters are present
        for param in regularized_config:
            assert param in config, f"Missing {param} in get_config()"
        assert "head_dim" in config

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

        # 1 input_linear (W)
        # 3 QKV linear (W)
        # 3 Norms (scale)
        # 1 output_gate_linear (W)
        # Total = 1 + 3 + 3 + 1 = 8
        expected_vars = 8
        actual_vars = len(layer.trainable_variables)
        assert actual_vars == expected_vars

    def test_trainable_variables_count_custom_head(self, custom_head_config, custom_sample_input):
        """Tests trainable variables when an output_proj is created."""
        layer = GatedAttention(**custom_head_config)
        layer(custom_sample_input)  # Build the layer

        # Expected vars from standard + 1 for output_proj (W)
        expected_vars = 8 + 1
        actual_vars = len(layer.trainable_variables)
        assert actual_vars == expected_vars

    def test_model_training_loop_integration(self, layer_config):
        """Tests integration in a standard training loop."""
        model = models.Sequential([
            layers.InputLayer(shape=(16, 64)),
            GatedAttention(**layer_config),
            layers.GlobalAveragePooling1D(),
            layers.Dense(10)
        ])
        model.compile("adam", "sparse_categorical_crossentropy")
        x_train = tf.random.normal((32, 16, 64))
        y_train = tf.random.uniform([32], 0, 10, dtype=tf.int32)
        history = model.fit(x_train, y_train, epochs=1, verbose=0)
        assert "loss" in history.history
        assert not np.isnan(history.history["loss"][0])

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
        small_input = tf.random.normal((2, 1, 32))  # Seq len of 1
        output = layer(small_input, training=False)
        assert output.shape == small_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_single_head(self, sample_input):
        """Tests layer with single attention head."""
        layer = GatedAttention(dim=64, num_heads=1, max_seq_len=128)
        output = layer(sample_input, training=False)
        assert output.shape == sample_input.shape

    def test_max_rope_percentage(self, sample_input):
        """Tests layer with maximum RoPE percentage (1.0)."""
        layer = GatedAttention(dim=64, num_heads=4, max_seq_len=128, rope_percentage=1.0)
        output = layer(sample_input, training=False)
        assert output.shape == sample_input.shape

    def test_batch_size_one(self):
        """Tests layer with batch size 1."""
        layer = GatedAttention(dim=32, num_heads=2, max_seq_len=64)
        single_batch_input = tf.random.normal((1, 10, 32))
        output = layer(single_batch_input, training=False)
        assert output.shape == single_batch_input.shape

    def test_compute_output_shape(self, layer_config):
        """Tests compute_output_shape method."""
        layer = GatedAttention(**layer_config)
        input_shape = (None, 20, 64)
        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape == input_shape

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_input, training):
        """Tests behavior in different training modes."""
        config = {**layer_config, "dropout_rate": 0.1}
        layer = GatedAttention(**config)
        output = layer(sample_input, training=training)
        assert output.shape == sample_input.shape

    # ===============================================
    # 8. Attention Mechanism Tests
    # ===============================================
    @pytest.mark.parametrize("mask_type", ["padding", "causal"])
    def test_attention_mask_effect(self, layer_config, sample_input, padding_attention_mask, causal_attention_mask, mask_type):
        """Tests that both padding and causal attention masks affect the output."""
        layer = GatedAttention(**layer_config)
        mask = padding_attention_mask if mask_type == "padding" else causal_attention_mask

        output_without_mask = layer(sample_input, training=False)
        output_with_mask = layer(sample_input, attention_mask=mask, training=False)

        assert not np.allclose(
            ops.convert_to_numpy(output_without_mask),
            ops.convert_to_numpy(output_with_mask),
            atol=1e-6,
        )

    def test_output_gating_effect(self, layer_config, sample_input):
        """Tests that output gating affects the final output."""
        layer = GatedAttention(**layer_config)
        output = layer(sample_input, training=False)
        # A simple check: if gating is working, the output magnitude should be
        # somewhat controlled by the sigmoid gate. This is a heuristic check.
        assert np.mean(np.abs(ops.convert_to_numpy(output))) < 10.0

    def test_rope_application(self, layer_config, sample_input):
        """Tests that RoPE is being applied by comparing different percentages."""
        layer1 = GatedAttention(**{**layer_config, "rope_percentage": 0.1})
        layer2 = GatedAttention(**{**layer_config, "rope_percentage": 0.9})

        output1 = layer1(sample_input, training=False)
        output2 = layer2(sample_input, training=False)

        assert not np.allclose(
            ops.convert_to_numpy(output1),
            ops.convert_to_numpy(output2),
            atol=1e-6,
        )

    def test_scaled_dot_product_attention_numerical_stability(self, layer_config):
        """Tests numerical stability of scaled dot-product attention."""
        layer = GatedAttention(**layer_config)
        large_input = tf.random.normal((2, 8, 64)) * 10.0
        output = layer(large_input, training=False)
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))
        assert not np.any(np.isinf(ops.convert_to_numpy(output)))

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

# ---------------------------------------------------------------------
# Mixed-precision mask tests (plan-2026-07-27-b4ef45f0, step 4)
# ---------------------------------------------------------------------
#
# WHAT IS BEING GUARDED HERE.
#
# `GatedAttention._compute_attention` used the ARITHMETIC mask form
#
#     additive_mask = (1.0 - ops.cast(mask, scaled_attention_logits.dtype)) * -1e9
#
# which `common.MASK_BIAS_VALUE`'s own docstring rules out. Under
# `mixed_precision.set_global_policy('mixed_float16')` the literal `-1e9` is
# materialized in float16, where it is `-inf` (np.float16(-1e9) == -inf). At every
# UNMASKED position `(1.0 - mask) == 0`, so the product is `0 * -inf = NaN` — the
# NaN appears exactly where NOTHING was masked, and the following matmul spreads it
# across the whole batch.
#
# MEASURED on unfixed HEAD (B=2, N=64, D=64, num_heads=4), GPU 1 / TF 2.18,
# non-finite entries in the layer OUTPUT:
#
#     policy            no mask   all-ones mask   padding mask   causal mask
#     float32            0/8192      0/8192          0/8192        0/8192
#     mixed_float16      0/8192   8192/8192       8192/8192     8192/8192
#
# The all-ones column is the important one: a mask that masks NOTHING destroys the
# entire batch. That is not a pathological input — it is what a caller passes when
# every sequence in the batch happens to be full length.
#
# THE FIX is `common.apply_attention_mask`, which builds the bias with `ops.where`
# inside `common.mask_dtype(...)` (>= float32), so `0 * -inf` cannot be formed at
# all. Each site keeps its own broadcast/cast order and its own polarity spelling.
#
# FULLY-MASKED QUERY ROW (assumption A2 — FALSIFIED at step 4, remedied at step 4b).
# A query row that keeps NOTHING used to make every logit in that row
# `MASK_BIAS_VALUE`, which is `-inf` in float16, so `softmax(all -inf)` was `0/0 = NaN`
# and the row lost its own output (measured at step 4: 128/128 in the degenerate row,
# 0/8064 elsewhere — contained, but lost). Casting back was never the problem and
# `out_dtype=None` would not have helped: the softmax here is `self.attn_prob`, a Keras
# layer with autocasting ON, which drags a float32 tensor straight back to float16
# (pinned by
# `TestGatedAttentionMaskHazardIsReal::test_the_probability_sublayer_autocasts_a_float32_input`).
# Step 4b therefore HOISTED `capsule_routing_attention.py`'s predicate-level rescue
# (D-006) into the shared helper as the `rescue_axis=` argument, and step 4c made it the
# helper's DEFAULT (D-009), so this site gets it without asking: a row that keeps nothing
# is treated as keeping EVERYTHING, so
# no all-`-inf` row is ever FORMED and the answer is finite and identical in every
# dtype. ACCEPTED COST (user ruling, decisions.md D-008): a caller whose mask is wrong
# now gets finite garbage instead of a loud NaN.
#
# ANTI-VACUITY. The `N = 7`-hides-an-fp16-`-inf`-at-`N >= 512` trap does not transfer:
# this hazard is a per-ELEMENT dtype overflow of a constant multiplied by an exact
# zero, not a long reduction, so it appears at any N >= 1. It is nevertheless
# asserted reachable rather than assumed — see `TestGatedAttentionMaskHazardIsReal`, which
# checks that the policy really selects float16 compute, that
# `float16(MASK_BIAS_VALUE)` really is `-inf`, that `float16(0) * that` really is
# `NaN`, and that each mask really has the structure its name claims. N = 64 keys is
# a realistic sequence length for this layer rather than a toy one.

from dl_techniques.layers.attention.common import MASK_BIAS_VALUE

_MP_B, _MP_N, _MP_D, _MP_H = 2, 64, 64, 4
_MP_KEEP = _MP_N // 2            # first half kept, second half masked (padding mask)
_MP_DEG_ROW = 5                  # the query row the 'degenerate' mask blanks entirely
_MP_SEED = 1234

# Absolute tolerance for "this policy's masked forward agrees with the float32 one".
#
# PRE-REGISTERED, not tuned after the fact: sized from the layer's own NO-MASK dtype
# error measured on unfixed HEAD (the only forward that survives fp16 there), which is
# the honest budget — a correct mask fix should leave the masked path no worse
# conditioned than the unmasked one. Measured no-mask max |policy - float32|:
# mixed_float16 0.00081, against an output absmax of 0.66 (up to
# 2.19 once a mask is applied). The entries below carry ~10x headroom on that.
# float32 compares against a control computed the same way and is exact.
#
# The `float64` entry was added at step 5b, when the RoPE float64 defect was fixed
# and these parametrizations stopped skipping. Sized the SAME way and in the WORSE
# of the two TF32 regimes (TF32 ON, i.e. this file run alone — `test_linear_attention`
# disables TF32 process-globally at import time, which makes the identical
# measurement much smaller inside the directory gate): measured no-mask
# float64-vs-float32 0.000666, masked up to 0.00176.
_MP_ATOL = {"float32": 1e-6, "mixed_float16": 0.01, "float64": 0.01}

# `float64` is EXCLUDED from the mask tests at this site, and that exclusion is itself
# pinned by `TestGatedAttentionFloat64IsASeparatePreExistingDefect` below: under a float64
# policy this layer raises inside `RotaryPositionEmbedding` ("cannot compute Mul as
# input #1 was expected to be a double tensor but is a float tensor") with NO mask
# supplied at all. It is a pre-existing RoPE dtype defect, entirely independent of
# masking, and out of scope for this step — but skipping it silently would hide it.


def _mp_input():
    """Deterministic ``(B, N, D)`` float32 input, shared by every test below."""
    return np.random.default_rng(7).standard_normal(
        (_MP_B, _MP_N, _MP_D)
    ).astype("float32")


def _mp_mask(kind):
    """One of the masks these tests need, as a float32 ``1 = keep`` array.

    ``'all_ones'`` masks NOTHING and is the catastrophic case for the arithmetic
    form. ``'padding'`` masks the second half of the keys with a rank-2 ``(B, N)`` mask, exercising this site's rank-2 broadcast branch, ``'causal'`` is
    lower-triangular, and ``'degenerate'`` blanks query row ``_MP_DEG_ROW`` entirely
    (the A2 probe — see the note above; it is NOT part of the finiteness contract
    this step establishes).
    """
    if kind == "all_ones":
        return np.ones((_MP_B, _MP_N, _MP_N), dtype="float32")
    if kind == "padding":
        m = np.ones((_MP_B, _MP_N), dtype="float32")
        m[:, _MP_KEEP:] = 0.0
        return m
    if kind == "causal":
        return np.broadcast_to(
            np.tril(np.ones((_MP_N, _MP_N), dtype="float32")), (_MP_B, _MP_N, _MP_N)
        ).copy()
    if kind == "degenerate":
        m = np.ones((_MP_B, _MP_N, _MP_N), dtype="float32")
        m[:, _MP_DEG_ROW, :] = 0.0
        return m
    raise ValueError(f"unknown mask kind {kind!r}")


def _mp_layer(**kwargs):
    """A built layer whose TRAINABLE weights are identical under every dtype policy.

    Seeding the initializers is NOT sufficient: a ``glorot_uniform`` draw under a
    ``float64`` policy differs from the same-seed draw under ``float32`` (the
    initializer samples in the VARIABLE dtype), so a cross-policy comparison on
    seeded-but-not-assigned weights measures the initializer, not the code under
    test. Explicit values are assigned instead. Non-trainable buffers (e.g. the RoPE
    cos/sin cache) are left as the layer computes them.
    """
    layer = GatedAttention(dim=_MP_D, num_heads=_MP_H, max_seq_len=128, **kwargs)
    layer.build((_MP_B, _MP_N, _MP_D))
    rng = np.random.default_rng(_MP_SEED)
    for weight in layer.trainable_weights:
        shape = tuple(weight.shape)
        if len(shape) == 1 and ("bias" in weight.name or "beta" in weight.name):
            value = np.zeros(shape)
        elif len(shape) == 1:                     # a scale / gamma: keep it near 1
            value = 1.0 + 0.1 * rng.standard_normal(shape)
        else:
            value = 0.2 * rng.standard_normal(shape)
        weight.assign(keras.ops.cast(
            keras.ops.convert_to_tensor(value.astype("float32")), weight.dtype
        ))
    return layer


def _mp_forward(layer, array, mask):
    """One masked forward pass, returned as float64 numpy."""
    out = layer(keras.ops.convert_to_tensor(array), attention_mask=(None if mask is None else keras.ops.convert_to_tensor(mask)))
    if isinstance(out, (list, tuple)):
        out = out[0]
    return keras.ops.convert_to_numpy(out).astype("float64")


_F32_REFERENCE = {}


def _float32_reference(kind):
    """Masked float32 output for ``kind``, memoized, under an explicit policy.

    This is the CONTROL every mixed-precision assertion compares against. It sets and
    restores the policy itself, so it is valid whichever parametrization of
    ``dtype_policy`` happens to reach it first.
    """
    if kind not in _F32_REFERENCE:
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("float32")
        try:
            layer = _mp_layer()
            _F32_REFERENCE[kind] = _mp_forward(
                layer, _mp_input(), _mp_mask(kind)
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)
    return _F32_REFERENCE[kind]


class TestGatedAttentionMaskHazardIsReal:
    """Anti-vacuity. If these stop holding, every fp16 test below is worthless."""

    def test_policy_really_selects_float16_compute(self, dtype_policy):
        expected = {
            "float32": "float32",
            "mixed_float16": "float16",
            "float64": "float64",
        }[dtype_policy]
        assert keras.mixed_precision.global_policy().compute_dtype == expected

    def test_the_arithmetic_form_really_is_nan_in_the_compute_dtype(self):
        with np.errstate(over="ignore", invalid="ignore"):
            bias = np.float16(MASK_BIAS_VALUE)
            assert np.isneginf(bias), (
                "anti-vacuity FAILED: float16(MASK_BIAS_VALUE) is not -inf, so the "
                "`0 * -inf` hazard this module guards is not reproducible here."
            )
            assert np.isnan(np.float16(0.0) * bias), (
                "anti-vacuity FAILED: float16(0) * float16(MASK_BIAS_VALUE) is not "
                "NaN — the arithmetic mask form would be harmless."
            )
        assert np.isfinite(np.float32(MASK_BIAS_VALUE)), (
            "anti-vacuity FAILED: MASK_BIAS_VALUE is not finite in float32, so "
            "`mask_dtype(...)` would not be a fix."
        )

    def test_the_all_ones_mask_really_masks_nothing(self):
        mask = _mp_mask("all_ones")
        assert int((mask == 0).sum()) == 0, (
            "the 'all_ones' mask masks something; it no longer reproduces the "
            "signature catastrophe (a vacuous mask destroying the batch)"
        )

    def test_the_partial_masks_really_mask_something(self):
        for kind in ("padding", "causal"):
            mask = _mp_mask(kind)
            assert int((mask == 0).sum()) > 0, (
                f"the {kind!r} mask masks nothing; it cannot detect a regression "
                "in the masking code"
            )
            rows = mask if mask.ndim == 3 else mask[:, None, :]
            assert not (rows == 0).all(axis=-1).any(), (
                f"the {kind!r} mask has a fully-masked query row, so it no longer "
                "isolates the covered case from the A2 probe"
            )

    def test_the_degenerate_mask_really_has_exactly_one_fully_masked_row(self):
        mask = _mp_mask("degenerate")
        empty = (mask == 0).all(axis=-1)
        assert int(empty.sum()) == _MP_B, (
            f"expected exactly one fully-masked query row per batch element, got "
            f"{int(empty.sum())} across {_MP_B} batch elements"
        )
        assert empty[:, _MP_DEG_ROW].all()

    def test_the_probability_sublayer_autocasts_a_float32_input(self, dtype_policy):
        """Why ``out_dtype`` cannot rescue a fully-masked row at this site.

        The softmax here is a Keras LAYER with autocasting on, so handing it a
        carefully-promoted float32 tensor changes nothing — it is seen inside its
        own ``call()`` as the compute dtype. This is the same measurement that
        selected the predicate-level rescue in `capsule_routing_attention.py`
        (assumption A4). If Keras ever stops doing this, this test fails and the
        ``out_dtype`` choice at this site can be revisited.
        """
        layer = _mp_layer()
        prob = layer.attn_prob
        assert getattr(prob, "autocast", False) is True

        seen = {}
        original = prob.call

        def spy(x, *args, **kwargs):
            seen["dtype"] = keras.backend.standardize_dtype(x.dtype)
            return original(x, *args, **kwargs)

        prob.call = spy
        try:
            prob(keras.ops.convert_to_tensor(np.zeros((1, _MP_H, 4, 4), dtype="float32")))
        finally:
            prob.call = original

        expected = keras.mixed_precision.global_policy().compute_dtype
        assert seen["dtype"] == expected, (
            f"a float32 tensor entering `attn_prob` was seen inside its call() as "
            f"{seen['dtype']!r}, not the compute dtype {expected!r}"
        )


class TestGatedAttentionMixedPrecisionMask:
    """SC1 + SC2: finite AND agreeing with float32, for every legal mask."""

    @pytest.mark.parametrize("kind", ["all_ones", "padding", "causal"])
    def test_masked_forward_is_finite_and_matches_float32(self, dtype_policy, kind):
        layer = _mp_layer()
        out = _mp_forward(layer, _mp_input(), _mp_mask(kind))

        n_bad = int((~np.isfinite(out)).sum())
        assert n_bad == 0, (
            f"{n_bad}/{out.size} non-finite output entries for a {kind!r} mask "
            f"under policy {dtype_policy!r}"
        )

        reference = _float32_reference(kind)
        atol = _MP_ATOL[dtype_policy]
        max_dev = float(np.abs(out - reference).max())
        assert max_dev <= atol, (
            f"{kind!r}-masked forward under {dtype_policy!r} deviates from the "
            f"float32 control by {max_dev:.4g} > {atol:.4g}"
        )
        assert float(np.abs(out).max()) > 0.5 * float(np.abs(reference).max()), (
            f"output absmax {np.abs(out).max():.4g} collapsed relative to the "
            f"float32 control {np.abs(reference).max():.4g}"
        )


class TestGatedAttentionFullyMaskedRow:
    """Step 4b: a fully-masked query row is RESCUED, not merely contained.

    Assumption A2 predicted that casting the biased logits back to the compute dtype
    is safe here because every softmax row keeps at least one position. A caller CAN
    break that by masking a whole query row, and step 4 MEASURED that it does: under
    `mixed_float16` the degenerate row NaN'd its own 128 outputs (containment held —
    0/8064 elsewhere — but the row itself was lost). Step 4b removes that boundary at
    the source via `common.apply_attention_mask`'s default `rescue_axis=-1`: a row
    that keeps NOTHING is treated as keeping EVERYTHING, so the all-`-inf` row is
    never FORMED and no NaN gradient is created either.

    Both tests below were observed FAILING on the step-4 code before the fix landed.
    """

    def test_a_fully_masked_row_is_finite_and_matches_float32(self, dtype_policy):
        layer = _mp_layer()
        out = _mp_forward(layer, _mp_input(), _mp_mask("degenerate"))

        n_bad = int((~np.isfinite(out)).sum())
        assert n_bad == 0, (
            f"{n_bad}/{out.size} non-finite output entries for a mask with a "
            f"FULLY-MASKED query row, under policy {dtype_policy!r} — the "
            "degenerate-row rescue is not reaching this site"
        )

        reference = _float32_reference("degenerate")
        atol = _MP_ATOL[dtype_policy]
        max_dev = float(np.abs(out - reference).max())
        assert max_dev <= atol, (
            f"the degenerate-masked forward under {dtype_policy!r} deviates from the "
            f"float32 control by {max_dev:.4g} > {atol:.4g} — this compares the WHOLE "
            "output, including the rescued row"
        )

    def test_the_rescued_row_behaves_as_if_it_kept_everything(self, dtype_policy):
        """The rescue's SEMANTICS, not merely its finiteness.

        The ``'degenerate'`` mask is all-ones except for query row ``_MP_DEG_ROW``,
        which is blanked. "A row that keeps nothing is treated as keeping everything"
        therefore makes it EQUIVALENT to the ``'all_ones'`` mask. That equivalence is
        the convention this step chose — over zeroing the row, or a sentinel — so it
        is asserted directly rather than inferred from a finiteness check.

        ANTI-VACUITY: on the pre-4b code this assertion fails in **float32** too (the
        degenerate row was a uniform average over all keys, not `softmax(unmasked
        logits)`), so it can tell the two conventions apart and is not a restatement
        of the finiteness test above.
        """
        degenerate = _mp_forward(_mp_layer(), _mp_input(), _mp_mask("degenerate"))
        all_ones = _mp_forward(_mp_layer(), _mp_input(), _mp_mask("all_ones"))

        atol = _MP_ATOL[dtype_policy]
        max_dev = float(np.abs(degenerate - all_ones).max())
        assert max_dev <= atol, (
            f"under {dtype_policy!r} the 'degenerate' mask does not behave like the "
            f"'all_ones' mask: max deviation {max_dev:.4g} > {atol:.4g}. A query row "
            "that keeps nothing must be treated as keeping everything."
        )
        assert float(np.abs(all_ones).max()) > 0.0, (
            "anti-vacuity FAILED: the all-ones-masked output is identically zero, so "
            "the comparison above could not distinguish anything"
        )


class TestGatedAttentionMaskPolarity:
    """SC6: the mask must suppress the MASKED positions, not the kept ones.

    A polarity inversion at this site — passing the keep predicate where its
    complement is meant, or vice versa — raises nothing, changes no shape and leaves
    the output perfectly finite. Only an influence test can see it. MEASURED on
    unmodified HEAD by handing the layer ``1 - mask``: perturbing a "masked" token
    then moves the kept query rows by 0.547 instead of 0.0, against a
    kept-token influence of 0.72.

    The statement is EXACT here (not approximate): a masked key contributes exactly
    `exp(-1e9) == 0` weight, so a perturbation of a masked token cannot reach a kept
    query row at all. Measured 0.0 in float32 on unfixed HEAD, so a real inversion
    is separated from correct behavior by the full 0.547 signal.
    """

    @staticmethod
    def _influence(layer, mask):
        base_input = _mp_input()
        perturbed_masked = base_input.copy()
        perturbed_masked[:, _MP_KEEP + 3, :] += 5.0      # a MASKED token
        perturbed_kept = base_input.copy()
        perturbed_kept[:, 3, :] += 5.0                   # a KEPT token

        rows = slice(0, _MP_KEEP)
        base = _mp_forward(layer, base_input, mask)
        assert np.isfinite(base[:, rows]).all(), (
            "the kept query rows are not finite; the comparison below would be "
            "meaningless"
        )
        delta_masked = float(
            np.abs(_mp_forward(layer, perturbed_masked, mask)[:, rows]
                   - base[:, rows]).max()
        )
        delta_kept = float(
            np.abs(_mp_forward(layer, perturbed_kept, mask)[:, rows]
                   - base[:, rows]).max()
        )
        return delta_masked, delta_kept

    def test_a_masked_token_has_no_influence_on_the_kept_rows(self, dtype_policy):
        layer = _mp_layer()
        delta_masked, delta_kept = self._influence(layer, _mp_mask("padding"))

        # Measured EXACTLY 0.0 on unfixed float32. The 1e-3 budget is session-noise
        # headroom (see `test_rpc_attention.py`, where a batched op measured 0.0 in
        # isolation and 1.1e-06 inside the full suite).
        assert delta_masked <= 1e-3, (
            f"perturbing a MASKED token changed the kept query rows by "
            f"{delta_masked:.6g} under policy {dtype_policy!r} — this must be "
            f"exact, so the mask polarity is INVERTED (the layer is attending to "
            f"the padding; measured 0.547 with a deliberately inverted mask)"
        )
        assert delta_kept > 0.1, (
            f"perturbing a KEPT token changed the output by only {delta_kept:.6g}; "
            "the test is vacuous — the layer is ignoring its input"
        )


class TestGatedAttentionFloat64RunsThroughRoPE:
    """FLIPPED at step 5b. This class used to assert a RAISE.

    Until step 5b, a ``float64`` global policy made this layer raise inside
    :class:`RotaryPositionEmbedding` — with **no mask supplied** — because the
    cached cos/sin tables were hard-coded to ``float32`` while the projected
    queries were ``float64``::

        InvalidArgumentError: Exception encountered when calling
        RotaryPositionEmbedding.call().
        cannot compute Mul as input #1(zero-based) was expected to be a double
        tensor but is a float tensor

    Every ``float64`` parametrization of the mask tests above skipped for that
    reason. `RotaryPositionEmbedding._create_rope_cache` now builds the tables in
    ``variable_dtype`` and `_apply_rope_rotation` casts them to the input's dtype,
    so those skips are gone and this class asserts the forward instead.
    """

    def test_the_float64_forward_runs_through_rope_without_a_mask(self):
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("float64")
        try:
            out = _mp_forward(_mp_layer(), _mp_input(), None)
            assert np.isfinite(out).all(), (
                "the float64 UNMASKED forward is not finite"
            )
            assert float(np.abs(out).max()) > 0.0
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_the_rope_tables_are_float64_under_a_float64_policy(self):
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("float64")
        try:
            layer = _mp_layer()
            rope = getattr(layer, "rope", None)
            assert rope is not None, (
                "no RotaryPositionEmbedding found on this layer; the guard below "
                "would be vacuous"
            )
            assert keras.backend.standardize_dtype(
                rope.cos_cached.dtype
            ) == "float64", (
                "the RoPE cos/sin cache must be stored at the layer's "
                "variable_dtype — a float32 cache under a float64 policy is the "
                "exact defect this class used to pin"
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)


# ---------------------------------------------------------------------
# Step 10 (adversarial-review completion fix). Two silent-semantics gaps that no
# finiteness, polarity or bit-identity test in this file could see, both MEASURED
# on the shipped code before these guards were written.
# ---------------------------------------------------------------------


class TestGatedAttentionRejectsAQueryOnlyMask:
    """A ``(B, H, Q, 1)`` mask reaches the helper untouched and is now REJECTED.

    ``_apply_attention``'s rank dispatch ends in ``else: mask = attention_mask
    # Assume it's already broadcastable``, so a rank-4 mask is handed to
    ``apply_attention_mask`` exactly as supplied. When its LAST axis is 1 the
    predicate is constant along the axis ``self.attn_prob`` reduces over, which is
    mathematically a no-op for a softmax.

    MEASURED on the pre-step-10 code (float32, ``GatedAttention(dim=64,
    num_heads=4)``, ``(2, 4, 16, 1)`` mask blanking query rows 5+): the output was
    **bit-identical (maxdiff 0.0)** to a no-op all-ones mask, i.e. the mask was
    silently and completely ignored, while the pre-plan arithmetic form differed by
    1.849 (a uniform-attention artifact, not masking — see
    ``test_common.py::TestApplyAttentionMaskRejectsASizeOneRescueAxis``). The caller
    now gets a named error instead of a wrong answer.
    """

    @staticmethod
    def _query_only_mask():
        mask = np.ones((_MP_B, _MP_H, _MP_N, 1), dtype="float32")
        mask[:, :, _MP_DEG_ROW:, :] = 0.0
        return mask

    def test_a_query_only_rank_4_mask_raises(self):
        layer = _mp_layer()
        with pytest.raises(ValueError, match=r"rescue_axis.*size 1"):
            _mp_forward(layer, _mp_input(), self._query_only_mask())

    def test_a_key_axis_rank_4_mask_is_still_accepted_and_still_masks(self):
        """Anti-vacuity: the guard rejects the degenerate LAYOUT, not rank 4.

        A rank-4 mask whose last axis is the KEY axis must still work AND still
        mask — asserted semantically (perturbing a masked key must not move the
        kept rows), never merely by finiteness.
        """
        layer = _mp_layer()
        array = _mp_input()
        mask = np.ones((_MP_B, 1, _MP_N, _MP_N), dtype="float32")
        mask[:, :, :, _MP_KEEP:] = 0.0

        base = _mp_forward(layer, array, mask)
        assert np.isfinite(base).all()

        perturbed = array.copy()
        perturbed[:, _MP_KEEP + 1, :] += 25.0
        moved = _mp_forward(layer, perturbed, mask)

        delta = float(np.abs(moved[:, :_MP_KEEP, :] - base[:, :_MP_KEEP, :]).max())
        assert delta <= 1e-4, (
            f"perturbing a MASKED key moved the kept rows by {delta:.6g} — the "
            "rank-4 key-axis mask is not masking"
        )


class TestGatedAttentionRescueFollowsTheConfiguredProbabilityAxis:
    """The rescue axis is derived from ``probability_config['axis']`` (W2).

    ``ProbabilityOutput`` reads ``axis`` from its ``type_config``
    (``probability_output.py:180``) and this layer forwards ``probability_config``
    verbatim, so the softmax does NOT always reduce over the last axis. The
    per-site D-009 anchor used to claim ``-1`` was "checked, not assumed" — true
    only for the DEFAULT config.

    MEASURED on the pre-step-10 code under ``mixed_float16`` with
    ``probability_config={"axis": -2}``: a rank-3 mask blanking one KEY COLUMN gave
    **2048/2048 non-finite** outputs (float32 control 0/2048), i.e. the whole-batch
    NaN class this plan set out to eliminate survived at a supported public
    configuration, because the rescue was looking down the wrong axis.
    """

    @staticmethod
    def _dead_column_mask():
        mask = np.ones((_MP_B, _MP_N, _MP_N), dtype="float32")
        mask[:, :, 3] = 0.0
        return mask

    def test_a_dead_column_is_rescued_when_the_softmax_reduces_over_queries(
        self, dtype_policy
    ):
        layer = _mp_layer(probability_config={"axis": -2})
        out = _mp_forward(layer, _mp_input(), self._dead_column_mask())

        non_finite = int((~np.isfinite(out)).sum())
        assert non_finite == 0, (
            f"{non_finite}/{out.size} non-finite outputs under {dtype_policy!r} "
            "with probability_config={'axis': -2} and a fully-masked KEY COLUMN — "
            "the rescue is not following the configured softmax axis"
        )

    def test_a_key_mask_that_is_constant_over_queries_raises_under_axis_minus_2(
        self,
    ):
        """The C1 guard and the W2 derivation compose.

        With ``axis=-2`` the softmax reduces over QUERIES, so a rank-2 ``(B, N)``
        key-padding mask — broadcast to ``(B, 1, 1, N)`` — is constant along the
        reduced axis and cannot mask anything. Pre-step-10 that configuration
        returned 2048/2048 NaN under ``mixed_float16``; it is now a named error.
        """
        layer = _mp_layer(probability_config={"axis": -2})
        mask = np.ones((_MP_B, _MP_N), dtype="float32")
        mask[:, _MP_KEEP:] = 0.0

        with pytest.raises(ValueError, match=r"rescue_axis.*size 1"):
            _mp_forward(layer, _mp_input(), mask)

    def test_the_default_config_still_rescues_over_the_key_axis(self, dtype_policy):
        """Anti-vacuity: the derivation must not move the DEFAULT behavior."""
        layer = _mp_layer()
        out = _mp_forward(layer, _mp_input(), _mp_mask("degenerate"))
        assert np.isfinite(out).all()
        np.testing.assert_allclose(
            out,
            _float32_reference("degenerate"),
            atol=_MP_ATOL[dtype_policy],
        )


# ---------------------------------------------------------------------------
# RoPE axis order (plan-2026-08-14T233721-d4f9beb2, step 39.1, D-083)
# ---------------------------------------------------------------------------
#
# These tests exist because `GatedAttention.call` handed RoPE a `(B, S, H, D)`
# tensor while `RotaryPositionEmbedding.call` reads its sequence length from
# `ops.shape(inputs)[2]`. Axis 2 in that frame is HEADS, so every token was
# rotated by the angle for its HEAD INDEX and the layer carried no positional
# information at all. Nothing in the 267-test suite noticed, because a per-head
# constant rotation is orthogonal and cancelled between q and k.


def _positional_defect(layer, dim, seq=8, batch=2, seed=0):
    """max|f(x)[perm] - f(x[perm])| for a swap of tokens 1 and 2.

    A layer whose positional embedding works is NOT permutation-equivariant, so
    this is large. A positionless layer returns float32 noise.

    The layer is reached through a parent model's `call()` -- the realistic
    build path, and the one that exposed the sibling `.assign()` defect (D-021).
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(batch, seq, dim)).astype("float32")
    perm = np.arange(seq)
    perm[1], perm[2] = perm[2], perm[1]

    inp = keras.Input(shape=(seq, dim))
    model = models.Model(inp, layer(inp))

    y = ops.convert_to_numpy(model(x))
    y_perm_in = ops.convert_to_numpy(model(x[:, perm, :]))
    return float(np.max(np.abs(y[:, perm, :] - y_perm_in)))


class TestRoPECarriesPosition:
    """RoPE must contribute an actual positional signal, under every grouping."""

    @pytest.mark.parametrize("num_kv_heads", [None, 8, 2, 1])
    def test_permuting_two_tokens_changes_the_output(self, num_kv_heads):
        """The headline assertion.

        Pre-fix this measured 3.58e-07 at `num_kv_heads=None` and 2.98e-07 at
        `num_kv_heads=2` -- float32 noise, i.e. EXACT permutation equivariance.
        Post-fix: 1.05e+00 and 7.36e-01. The threshold sits ~4 orders of
        magnitude above the noise floor and ~4 below the signal.
        """
        keras.utils.set_random_seed(1234)
        layer = GatedAttention(
            dim=32, num_heads=8, num_kv_heads=num_kv_heads, max_seq_len=64,
            rope_percentage=1.0,
        )
        assert _positional_defect(layer, dim=32) > 1e-3

    def test_paired_query_and_kv_heads_get_the_same_rotation(self):
        """Pins the DEFINITION the two tests above are graded against.

        NOTE: this one does not fail when `GatedAttention` regresses -- it calls
        `RotaryPositionEmbedding` directly and does the transpose itself, so it
        passes both ways. It is here to make the expected property checkable on
        its own terms; the LAYER is guarded by the two tests above, both of
        which were verified RED against the pre-fix source.

        Derived from the RoPE DEFINITION, not from the implementation: the
        rotation angle at sequence position p is p * inv_freq -- a function of
        POSITION ONLY. So if a query head and the K/V head it is paired with
        carry the same vector, RoPE must return them bit-identical, whatever
        the head grouping is.

        Pre-fix the angle was a function of the HEAD INDEX instead, so query
        head h was rotated by angle h and its paired K/V head by angle
        h // num_kv_groups. Measured 4.04e+00 pre-fix, exactly 0.0 post-fix.
        """
        from dl_techniques.layers.embedding.rotary_position_embedding import (
            RotaryPositionEmbedding,
        )

        b, s, h, hkv, d = 1, 6, 8, 2, 8
        groups = h // hkv
        rng = np.random.default_rng(11)
        base = rng.normal(size=(b, s, d)).astype("float32")
        q = np.repeat(base[:, :, None, :], h, axis=2)
        k = np.repeat(base[:, :, None, :], hkv, axis=2)

        rope = RotaryPositionEmbedding(
            head_dim=d, max_seq_len=64, rope_percentage=1.0
        )
        rope.build((b, h, s, d))

        def apply(t):
            r = rope(ops.convert_to_tensor(np.transpose(t, (0, 2, 1, 3))))
            return np.transpose(ops.convert_to_numpy(r), (0, 2, 1, 3))

        q_rot = apply(q)
        k_rot = np.repeat(apply(k), groups, axis=2)

        np.testing.assert_allclose(q_rot, k_rot, atol=0.0, rtol=0.0)

        # Anti-vacuity: the rotation must not be the identity, or the equality
        # above would hold for any axis order at all.
        assert np.max(np.abs(q_rot - q)) > 1.0

    def test_rope_is_called_with_the_sequence_on_axis_2(self):
        """Pins the CONVENTION at the call site, not just its consequence.

        `RotaryPositionEmbedding` reads `ops.shape(inputs)[2]` as the sequence
        length. This records every tensor the layer actually hands it and
        requires axis 2 to be the sequence, distinguishable here because
        `seq_len` (7) differs from both `num_heads` (8) and `num_kv_heads` (2).
        """
        seq_len, num_heads, num_kv_heads, dim = 7, 8, 2, 32
        layer = GatedAttention(
            dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
            max_seq_len=64, rope_percentage=1.0,
        )

        seen = []
        original_call = layer.rope.call

        def recording_call(inputs, *args, **kwargs):
            seen.append(tuple(inputs.shape))
            return original_call(inputs, *args, **kwargs)

        layer.rope.call = recording_call

        inp = keras.Input(shape=(seq_len, dim))
        model = models.Model(inp, layer(inp))
        model(np.zeros((2, seq_len, dim), dtype="float32"))

        assert seen, "RoPE was never invoked; this test proves nothing."
        # Keras may trace `call` more than once; every invocation must agree.
        for shape in seen:
            assert len(shape) == 4, shape
            assert shape[2] == seq_len, (
                f"RoPE was handed {shape}: axis 2 is {shape[2]}, not the "
                f"sequence length {seq_len}. It will index its "
                f"position-indexed table by the wrong axis."
            )
        # Both head counts must appear on axis 1 -- q at num_heads, k at
        # num_kv_heads -- or the tensors were not the q/k pair we think.
        assert {s[1] for s in seen} == {num_heads, num_kv_heads}


# ---------------------------------------------------------------------------
# Grouped-query attention (plan-2026-08-14T233721-d4f9beb2, step 39.2, D-071)
# ---------------------------------------------------------------------------
#
# Step 39 shipped `num_kv_heads` -- a new public parameter, a narrowed weight
# layout and a new head-expansion code path -- with ZERO tests. Its commit
# reported "267 passed", which was 267 passes of tests that never constructed
# `num_kv_heads != None`. Everything below constructs it.


def _ga(seq_len=6, dim=32, **kwargs):
    cfg = dict(dim=dim, num_heads=8, max_seq_len=64, rope_percentage=1.0,
               dropout_rate=0.0)
    cfg.update(kwargs)
    return GatedAttention(**cfg)


def _forward(layer, x):
    """Through a parent model's call(), the realistic build path."""
    inp = keras.Input(shape=x.shape[1:])
    return ops.convert_to_numpy(models.Model(inp, layer(inp))(x))


class TestGroupedQueryAttention:
    """The `num_kv_heads` path: widths, head mapping, config, equivalence."""

    def test_none_is_exactly_num_heads(self):
        """`None` must mean "one K/V head per query head", i.e. plain MHA.

        This is the promise the D-071 anchor makes to every existing
        checkpoint: a layer built without `num_kv_heads` must be
        indistinguishable from one built with `num_kv_heads=num_heads`.
        """
        keras.utils.set_random_seed(7)
        rng = np.random.default_rng(3)
        x = rng.normal(size=(2, 6, 32)).astype("float32")

        a, b = _ga(num_kv_heads=None), _ga(num_kv_heads=8)
        ya, yb = _forward(a, x), _forward(b, x)

        assert a.num_kv_heads == b.num_kv_heads == 8
        assert a.num_kv_groups == b.num_kv_groups == 1
        assert a.kv_dim == b.kv_dim == a.attention_dim

        # Same architecture => same weight shapes, so weights transfer, so the
        # outputs must agree EXACTLY once they do.
        b.set_weights(a.get_weights())
        np.testing.assert_allclose(_forward(b, x), ya, atol=0.0, rtol=0.0)
        assert np.max(np.abs(ya)) > 0.0, "vacuous: the layer emitted zeros"
        del yb

    @pytest.mark.parametrize("num_kv_heads,groups", [(8, 1), (4, 2), (2, 4), (1, 8)])
    def test_kv_projection_and_norm_widths_follow_num_kv_heads(
        self, num_kv_heads, groups
    ):
        """The narrowed weight layout, asserted rather than assumed.

        `k_linear`/`v_linear` project to `kv_dim`, not `attention_dim`, and
        `k_norm`/`v_norm` are built at `kv_dim`. At the shipped Qwen3-Next
        shape (16 heads / 4 KV heads) that is a 4x narrowing of four weights
        per block -- a real checkpoint break, recorded in D-071.
        """
        layer = _ga(num_kv_heads=num_kv_heads)
        layer.build((None, 6, 32))

        assert layer.num_kv_groups == groups
        assert layer.kv_dim == num_kv_heads * layer.head_dim

        assert layer.q_linear.kernel.shape[-1] == layer.attention_dim
        assert layer.k_linear.kernel.shape[-1] == layer.kv_dim
        assert layer.v_linear.kernel.shape[-1] == layer.kv_dim

        # The Q/K norms are per-feature over the projection width; K's must
        # follow K's width or a GQA layer cannot even be built.
        k_norm_w = layer.k_norm.weights
        assert k_norm_w, "k_norm has no weights; this assertion is vacuous"
        for w in k_norm_w:
            assert w.shape[-1] == layer.kv_dim
        for w in layer.q_norm.weights:
            assert w.shape[-1] == layer.attention_dim

    def test_one_kv_head_means_every_query_head_sees_the_same_k(self):
        """At `num_kv_heads=1` the expansion must broadcast a single K/V head.

        Asserted on the expansion itself: `repeat` of a 1-head tensor must give
        H identical heads. If the expansion dropped or reordered heads this
        fails.
        """
        layer = _ga(num_kv_heads=1)
        layer.build((None, 6, 32))
        assert layer.num_kv_groups == layer.num_heads

        one = np.arange(2 * 6 * 1 * 4, dtype="float32").reshape(2, 6, 1, 4)
        expanded = ops.convert_to_numpy(
            ops.repeat(ops.convert_to_tensor(one), layer.num_kv_groups, axis=2)
        )
        assert expanded.shape[2] == layer.num_heads
        for h in range(layer.num_heads):
            np.testing.assert_allclose(expanded[:, :, h, :], one[:, :, 0, :],
                                       atol=0.0, rtol=0.0)

    def test_repeat_not_tile_puts_group_members_adjacent(self):
        """The isolating mutation the D-071 anchor's reasoning demands.

        The anchor claims `ops.repeat` (not `ops.tile`) is what makes query
        head g read K/V head `g // num_kv_groups`. Nothing asserted it. Here
        both are computed and the required mapping is checked against
        `repeat`, then shown to be VIOLATED by `tile` -- so the test cannot
        pass if the implementation is switched.
        """
        num_kv_heads, num_heads = 2, 8
        groups = num_heads // num_kv_heads
        # Head j carries the constant value j, so a head's identity is readable
        # straight off the tensor.
        k = np.arange(num_kv_heads, dtype="float32").reshape(1, 1, num_kv_heads, 1)
        k = np.tile(k, (1, 3, 1, 4))
        kt = ops.convert_to_tensor(k)

        repeated = ops.convert_to_numpy(ops.repeat(kt, groups, axis=2))
        tiled = ops.convert_to_numpy(ops.tile(kt, (1, 1, groups, 1)))

        expected = np.array(
            [h // groups for h in range(num_heads)], dtype="float32"
        )
        got_repeat = repeated[0, 0, :, 0]
        got_tile = tiled[0, 0, :, 0]

        np.testing.assert_allclose(got_repeat, expected, atol=0.0, rtol=0.0)
        # Anti-vacuity: `tile` must NOT satisfy it, or this proves nothing.
        assert not np.allclose(got_tile, expected), (
            "tile and repeat agree here, so this test cannot detect an "
            "inverted group-to-head mapping"
        )
        # And name what tile would have done: interleaved the groups.
        np.testing.assert_allclose(
            got_tile,
            np.array([h % num_kv_heads for h in range(num_heads)], dtype="float32"),
            atol=0.0, rtol=0.0,
        )

    @pytest.mark.parametrize("num_kv_heads", [None, 8, 4, 2, 1])
    def test_get_config_round_trip_carries_num_kv_heads(self, num_kv_heads):
        """A parameter that does not survive `get_config` is a parameter that
        silently resets to MHA on every `.keras` reload."""
        layer = _ga(num_kv_heads=num_kv_heads)
        config = layer.get_config()
        assert "num_kv_heads" in config

        restored = GatedAttention.from_config(config)
        assert restored.num_kv_heads == layer.num_kv_heads
        assert restored.num_kv_groups == layer.num_kv_groups
        assert restored.kv_dim == layer.kv_dim

    def test_full_keras_round_trip_preserves_values_under_gqa(self):
        """Save/load must restore the narrowed weights, not fresh ones.

        A shape-only check passes when a nested container silently restores
        re-initialized kernels; only a VALUE comparison sees it.
        """
        keras.utils.set_random_seed(21)
        rng = np.random.default_rng(5)
        x = rng.normal(size=(2, 6, 32)).astype("float32")

        layer = _ga(num_kv_heads=2)
        inp = keras.Input(shape=(6, 32))
        model = models.Model(inp, layer(inp))
        before = ops.convert_to_numpy(model(x))

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gqa.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            after = ops.convert_to_numpy(reloaded(x))

        np.testing.assert_allclose(after, before, atol=1e-6)
        assert np.max(np.abs(before)) > 1e-3, "vacuous: outputs are ~zero"

    @pytest.mark.parametrize(
        "num_kv_heads,message",
        [
            (0, "must be positive"),
            (-1, "must be positive"),
            (16, "cannot exceed num_heads"),
            (3, "must be divisible"),
        ],
    )
    def test_invalid_groupings_are_refused_at_construction(
        self, num_kv_heads, message
    ):
        """An indivisible grouping has no head mapping; it must not build one."""
        with pytest.raises(ValueError, match=message):
            _ga(num_kv_heads=num_kv_heads)

    def test_gqa_shrinks_the_parameter_count(self):
        """The whole point of GQA. Pre-step-39 this number did not move at all,
        which is how a serialized, `summary()`-printed knob stayed inert."""
        counts = {}
        for kv in (8, 4, 1):
            layer = _ga(num_kv_heads=kv)
            layer.build((None, 6, 32))
            counts[kv] = int(sum(np.prod(w.shape) for w in layer.weights))

        assert counts[8] > counts[4] > counts[1], counts

    def test_gqa_equals_mha_whose_kv_weights_are_the_group_expansion(self):
        """The group->query head mapping, read off the LAYER.

        `test_repeat_not_tile_puts_group_members_adjacent` above pins what
        `repeat` means but computes both candidates itself, so it cannot fail
        when the layer changes. This one can: it builds an ordinary
        `num_kv_heads=num_heads` layer whose K/V projections are the GQA
        layer's projections expanded GROUP-ADJACENTLY (head h of the reference
        gets K/V head `h // num_kv_groups`), and requires the two to agree
        exactly.

        If the implementation used `ops.tile`, the GQA layer would pair query
        head h with K/V head `h % num_kv_heads` while the reference pairs it
        with `h // num_kv_groups`, and the outputs diverge. Verified RED by
        substituting `ops.tile` for `ops.repeat` in `GatedAttention.call`.
        """
        keras.utils.set_random_seed(99)
        rng = np.random.default_rng(17)
        dim, num_heads, num_kv_heads = 32, 8, 2
        groups = num_heads // num_kv_heads
        x = rng.normal(size=(2, 6, dim)).astype("float32")

        gqa = _ga(num_heads=num_heads, num_kv_heads=num_kv_heads)
        ref = _ga(num_heads=num_heads, num_kv_heads=num_heads)
        gqa.build((None, 6, dim))
        ref.build((None, 6, dim))

        head_dim = gqa.head_dim

        def expand_last(arr):
            """(..., num_kv_heads*head_dim) -> (..., num_heads*head_dim),
            group-adjacent: output head h carries input head h // groups."""
            lead = arr.shape[:-1]
            a = arr.reshape(*lead, num_kv_heads, head_dim)
            a = np.repeat(a, groups, axis=-2)
            return a.reshape(*lead, num_heads * head_dim)

        # Copy every weight across by NAME-independent position, expanding the
        # four that are narrowed under GQA.
        narrowed = set()
        for sub in (gqa.k_linear, gqa.v_linear, gqa.k_norm, gqa.v_norm):
            narrowed.update(id(w) for w in sub.weights)

        gqa_w, ref_w = gqa.weights, ref.weights
        assert len(gqa_w) == len(ref_w), "layers are not structurally parallel"

        new_ref = []
        expanded_count = 0
        for gw, rw in zip(gqa_w, ref_w):
            v = ops.convert_to_numpy(gw)
            if id(gw) in narrowed:
                v = expand_last(v)
                expanded_count += 1
            assert v.shape == tuple(rw.shape), (gw.path, v.shape, rw.shape)
            new_ref.append(v)

        assert expanded_count == 4, (
            f"expected exactly 4 narrowed weights under GQA, expanded "
            f"{expanded_count}; the weight layout has changed"
        )
        ref.set_weights(new_ref)

        y_gqa = _forward(gqa, x)
        y_ref = _forward(ref, x)
        np.testing.assert_allclose(y_gqa, y_ref, atol=1e-6)
        assert np.max(np.abs(y_gqa)) > 1e-3, "vacuous: outputs are ~zero"
