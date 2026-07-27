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
# (D-006) into the shared helper as the opt-in `rescue_axis=` argument, and this site
# passes `rescue_axis=-1`: a row that keeps nothing is treated as keeping EVERYTHING, so
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
_MP_ATOL = {"float32": 1e-6, "mixed_float16": 0.01}

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
        if dtype_policy == "float64":
            pytest.skip(
                "float64 raises inside RotaryPositionEmbedding regardless of masking "
                "(pre-existing, pinned by "
                "TestGatedAttentionFloat64IsASeparatePreExistingDefect)"
            )

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
    the source by passing `rescue_axis=-1` to `common.apply_attention_mask`: a row
    that keeps NOTHING is treated as keeping EVERYTHING, so the all-`-inf` row is
    never FORMED and no NaN gradient is created either.

    Both tests below were observed FAILING on the step-4 code before the fix landed.
    """

    def test_a_fully_masked_row_is_finite_and_matches_float32(self, dtype_policy):
        if dtype_policy == "float64":
            pytest.skip(
                "float64 raises inside RotaryPositionEmbedding regardless of masking "
                "(pre-existing, pinned by "
                "TestGatedAttentionFloat64IsASeparatePreExistingDefect)"
            )

        layer = _mp_layer()
        out = _mp_forward(layer, _mp_input(), _mp_mask("degenerate"))

        n_bad = int((~np.isfinite(out)).sum())
        assert n_bad == 0, (
            f"{n_bad}/{out.size} non-finite output entries for a mask with a "
            f"FULLY-MASKED query row, under policy {dtype_policy!r} — the "
            "`rescue_axis=-1` rescue is not reaching this site"
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
        if dtype_policy == "float64":
            pytest.skip(
                "float64 raises inside RotaryPositionEmbedding regardless of masking "
                "(pre-existing, pinned by "
                "TestGatedAttentionFloat64IsASeparatePreExistingDefect)"
            )

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
        if dtype_policy == "float64":
            pytest.skip(
                "float64 raises inside RotaryPositionEmbedding regardless of masking "
                "(pre-existing, pinned by "
                "TestGatedAttentionFloat64IsASeparatePreExistingDefect)"
            )

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


class TestGatedAttentionFloat64IsASeparatePreExistingDefect:
    """Pins the reason the mask tests above skip ``float64`` at this site.

    Under a ``float64`` global policy this layer raises inside
    :class:`RotaryPositionEmbedding` — with **no mask supplied** — because the cached
    cos/sin buffer stays float32 while the projected queries are float64. It has
    nothing to do with the attention mask and is not fixed by this step; this test
    exists so the skip above is a DOCUMENTED exclusion rather than a silent one, and
    so it turns red the day someone fixes RoPE.
    """

    def test_float64_policy_raises_in_rope_even_without_a_mask(self):
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("float64")
        try:
            layer = _mp_layer()
            with pytest.raises(Exception, match="double tensor but is a float tensor"):
                _mp_forward(layer, _mp_input(), None)
        finally:
            keras.mixed_precision.set_global_policy(previous)
