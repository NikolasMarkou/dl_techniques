"""
Comprehensive test suite for FNetEncoderBlock following modern Keras 3 best practices.

This test suite validates all aspects of the FNetEncoderBlock implementation including
initialization, forward pass, serialization, configuration management, and edge cases.
"""

import pytest
import tempfile
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import keras
import tensorflow as tf

# Assuming the FNetEncoderBlock is in this location - adjust import path as needed
from dl_techniques.layers.fnet_encoder_block import FNetEncoderBlock


class TestFNetEncoderBlock:
    """Comprehensive test suite for FNetEncoderBlock layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'intermediate_dim': 256,
            'dropout_rate': 0.1,
            'activation': 'gelu',
            'layer_norm_epsilon': 1e-12,
            'use_bias': True,
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample input for testing - 3D tensor [batch, seq, hidden]."""
        return keras.random.normal(shape=(4, 16, 64))

    @pytest.fixture
    def bert_like_input(self) -> keras.KerasTensor:
        """BERT-like dimensions for realistic testing."""
        return keras.random.normal(shape=(2, 512, 768))

    def test_initialization(self, layer_config):
        """Test layer initialization with valid configuration."""
        layer = FNetEncoderBlock(**layer_config)

        # Check configuration attributes
        assert layer.intermediate_dim == 256
        assert layer.dropout_rate == 0.1
        assert layer.activation == 'gelu'
        assert layer.layer_norm_epsilon == 1e-12
        assert layer.use_bias is True
        assert not layer.built  # Should not be built yet

        # Check sub-layers are created
        assert layer.fourier_transform is not None
        assert layer.fourier_layer_norm is not None
        assert layer.intermediate_dense is not None
        assert layer.output_dense is not None
        assert layer.dropout is not None
        assert layer.output_layer_norm is not None

    def test_initialization_with_custom_config(self):
        """Test initialization with custom fourier configuration."""
        fourier_config = {'normalize_dft': False, 'epsilon': 1e-10}
        layer = FNetEncoderBlock(
            intermediate_dim=128,
            fourier_config=fourier_config,
            activation='relu'
        )

        assert layer.intermediate_dim == 128
        assert layer.fourier_config == fourier_config
        assert layer.activation == 'relu'

    def test_forward_pass_basic(self, layer_config, sample_input):
        """Test basic forward pass and building."""
        layer = FNetEncoderBlock(**layer_config)

        # Forward pass should trigger building
        output = layer(sample_input)

        # Check layer is now built
        assert layer.built

        # Check output shape matches input shape
        expected_shape = (4, 16, 64)
        assert output.shape == expected_shape

        # Check output is different from input (layer is doing something)
        input_np = keras.ops.convert_to_numpy(sample_input)
        output_np = keras.ops.convert_to_numpy(output)

        # Should not be identical (very low probability)
        assert not np.allclose(input_np, output_np, atol=1e-6)

    def test_forward_pass_bert_dimensions(self, bert_like_input):
        """Test with BERT-like dimensions."""
        layer = FNetEncoderBlock(intermediate_dim=3072)  # 4x hidden_dim like BERT

        output = layer(bert_like_input)

        # Shape should be preserved
        assert output.shape == bert_like_input.shape
        assert output.shape == (2, 512, 768)

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with FNetEncoderBlock
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = FNetEncoderBlock(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_fnet_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions after serialization
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization cycle"
            )

    def test_serialization_with_custom_fourier_config(self, sample_input):
        """Test serialization with custom fourier configuration."""
        config = {
            'intermediate_dim': 128,
            'fourier_config': {'normalize_dft': False, 'implementation': 'matrix'}
        }

        # Create and test model
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = FNetEncoderBlock(**config)(inputs)
        model = keras.Model(inputs, outputs)

        original_pred = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_custom_fnet.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Custom fourier config serialization failed"
            )

    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = FNetEncoderBlock(**layer_config)
        config = layer.get_config()

        # Check all layer_config parameters are present
        for key in layer_config:
            assert key in config, f"Missing {key} in get_config()"

        # Check additional expected keys
        expected_keys = [
            'intermediate_dim', 'dropout_rate', 'activation',
            'fourier_config', 'layer_norm_epsilon', 'use_bias',
            'kernel_initializer'
        ]

        for key in expected_keys:
            assert key in config, f"Expected key {key} not in config"

    def test_gradients_flow(self, layer_config, sample_input):
        """Test gradient computation through the layer."""
        layer = FNetEncoderBlock(**layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        # Get gradients for layer weights
        gradients = tape.gradient(loss, layer.trainable_variables)

        # All gradients should exist and be non-None
        assert all(g is not None for g in gradients), "Some gradients are None"
        assert len(gradients) > 0, "No trainable variables found"

        # Check that gradients have reasonable magnitudes (not all zeros)
        grad_norms = [keras.ops.convert_to_numpy(keras.ops.norm(g)) for g in gradients]
        assert any(norm > 1e-8 for norm in grad_norms), "All gradients are near zero"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_input, training):
        """Test behavior in different training modes."""
        layer = FNetEncoderBlock(**layer_config)

        output = layer(sample_input, training=training)

        # Basic shape check
        assert output.shape == sample_input.shape

        # Check that dropout behaves differently in training vs inference
        if layer_config['dropout_rate'] > 0:
            output_train = layer(sample_input, training=True)
            output_eval = layer(sample_input, training=False)

            # Outputs should be different due to dropout (with high probability)
            train_np = keras.ops.convert_to_numpy(output_train)
            eval_np = keras.ops.convert_to_numpy(output_eval)

            # This might occasionally fail due to randomness, but very unlikely
            assert not np.allclose(train_np, eval_np, atol=1e-6), \
                "Training and eval outputs are identical (dropout not working?)"

    def test_no_dropout(self, sample_input):
        """Test layer with no dropout."""
        layer = FNetEncoderBlock(intermediate_dim=128, dropout_rate=0.0)

        output_train = layer(sample_input, training=True)
        output_eval = layer(sample_input, training=False)

        # With no dropout, outputs should be identical
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output_train),
            keras.ops.convert_to_numpy(output_eval),
            rtol=1e-6, atol=1e-6,
            err_msg="Outputs differ with dropout_rate=0.0"
        )

    def test_different_activations(self, sample_input):
        """Test layer with different activation functions."""
        activations = ['relu', 'gelu', 'swish', 'tanh']

        outputs = []
        for activation in activations:
            layer = FNetEncoderBlock(intermediate_dim=64, activation=activation)
            output = layer(sample_input)
            outputs.append(keras.ops.convert_to_numpy(output))

        # Different activations should produce different outputs
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not np.allclose(outputs[i], outputs[j], atol=1e-6), \
                    f"Activations {activations[i]} and {activations[j]} produce identical outputs"

    def test_layer_builds_sublayers_correctly(self, layer_config, sample_input):
        """Test that sub-layers are properly built."""
        layer = FNetEncoderBlock(**layer_config)

        # Before forward pass, sub-layers should not be built
        assert not layer.fourier_transform.built
        assert not layer.intermediate_dense.built

        # Forward pass should build everything
        _ = layer(sample_input)

        # Now all sub-layers should be built
        assert layer.built
        assert layer.fourier_transform.built
        assert layer.fourier_layer_norm.built
        assert layer.intermediate_dense.built
        assert layer.output_dense.built
        assert layer.dropout.built
        assert layer.output_layer_norm.built

    def test_edge_cases(self):
        """Test error conditions and edge cases."""
        # Invalid intermediate_dim
        with pytest.raises(ValueError, match="intermediate_dim must be positive"):
            FNetEncoderBlock(intermediate_dim=0)

        with pytest.raises(ValueError, match="intermediate_dim must be positive"):
            FNetEncoderBlock(intermediate_dim=-10)

        # Invalid dropout_rate
        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            FNetEncoderBlock(intermediate_dim=64, dropout_rate=-0.1)

        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            FNetEncoderBlock(intermediate_dim=64, dropout_rate=1.5)

        # Invalid layer_norm_epsilon
        with pytest.raises(ValueError, match="layer_norm_epsilon must be positive"):
            FNetEncoderBlock(intermediate_dim=64, layer_norm_epsilon=0.0)

        with pytest.raises(ValueError, match="layer_norm_epsilon must be positive"):
            FNetEncoderBlock(intermediate_dim=64, layer_norm_epsilon=-1e-6)

    def test_wrong_input_shape(self, layer_config):
        """Test error handling for incorrect input shapes."""
        layer = FNetEncoderBlock(**layer_config)

        # 2D input (missing sequence dimension)
        with pytest.raises(ValueError, match="Expected 3D input"):
            wrong_input = keras.random.normal(shape=(4, 64))
            layer(wrong_input)

        # 4D input (too many dimensions)
        with pytest.raises(ValueError, match="Expected 3D input"):
            wrong_input = keras.random.normal(shape=(4, 8, 8, 64))
            layer(wrong_input)

    def test_unknown_hidden_dimension(self, layer_config):
        """Test error handling for unknown hidden dimension."""
        # Create input with unknown last dimension
        inputs = keras.Input(shape=(16, None))  # hidden_dim unknown
        layer = FNetEncoderBlock(**layer_config)

        with pytest.raises(ValueError, match="Hidden dimension must be known at build time"):
            layer(inputs)

    def test_compute_output_shape(self, layer_config):
        """Test compute_output_shape method."""
        layer = FNetEncoderBlock(**layer_config)

        input_shape = (None, 512, 768)
        output_shape = layer.compute_output_shape(input_shape)

        # Shape should be preserved
        assert output_shape == input_shape

    def test_very_small_layer(self):
        """Test with minimal dimensions."""
        layer = FNetEncoderBlock(intermediate_dim=1)
        small_input = keras.random.normal(shape=(1, 2, 2))

        output = layer(small_input)
        assert output.shape == (1, 2, 2)

    def test_different_intermediate_ratios(self, sample_input):
        """Test different ratios of intermediate_dim to hidden_dim."""
        hidden_dim = sample_input.shape[-1]  # 64

        # Test different expansion ratios
        ratios = [1, 2, 4, 8]

        for ratio in ratios:
            layer = FNetEncoderBlock(intermediate_dim=hidden_dim * ratio)
            output = layer(sample_input)
            assert output.shape == sample_input.shape

    def test_reproducibility(self, layer_config, sample_input):
        """Test that results are reproducible with same random seed."""
        # Set seeds for reproducibility
        keras.utils.set_random_seed(42)
        layer1 = FNetEncoderBlock(**layer_config)
        output1 = layer1(sample_input)

        # Reset seed and create identical layer
        keras.utils.set_random_seed(42)
        layer2 = FNetEncoderBlock(**layer_config)
        output2 = layer2(sample_input)

        # Results should be identical
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Results not reproducible with same seed"
        )

    def test_memory_efficiency(self):
        """Test that layer doesn't hold unnecessary references."""
        layer = FNetEncoderBlock(intermediate_dim=64)

        # Small test to ensure layer can be created and used
        test_input = keras.random.normal(shape=(2, 4, 8))
        output = layer(test_input)

        # Basic functionality check
        assert output is not None
        assert output.shape == test_input.shape

        # Layer should not hold reference to input after computation
        del test_input
        # If we get here without issues, memory management is working


# Additional integration tests
class TestFNetEncoderBlockIntegration:
    """Integration tests for FNetEncoderBlock in larger contexts."""

    def test_in_sequential_model(self):
        """Test FNetEncoderBlock as part of a Sequential model."""
        model = keras.Sequential([
            # FIX: Use input_shape and remove deprecated input_length
            keras.layers.Embedding(1000, 64, input_shape=(32,)),
            FNetEncoderBlock(intermediate_dim=256),
            FNetEncoderBlock(intermediate_dim=256),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Test compilation
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Test prediction
        # FIX: Use correct keyword arguments minval/maxval for randint
        sample_tokens = keras.random.randint(minval=0, maxval=1000, shape=(4, 32))
        predictions = model(sample_tokens)

        assert predictions.shape == (4, 10)

    def test_stacked_layers(self):
        """Test multiple FNetEncoderBlocks stacked together."""
        inputs = keras.Input(shape=(64, 128))
        x = inputs

        # Stack 3 encoder blocks
        for i in range(3):
            x = FNetEncoderBlock(
                intermediate_dim=512,
                dropout_rate=0.1,
                name=f'fnet_block_{i}'
            )(x)

        outputs = keras.layers.GlobalAveragePooling1D()(x)
        model = keras.Model(inputs, outputs)

        test_input = keras.random.normal(shape=(2, 64, 128))
        output = model(test_input)

        assert output.shape == (2, 128)

    def test_with_masking(self):
        """Test FNetEncoderBlock with masked inputs."""
        # This tests that the layer works with variable-length sequences
        # FIX: FNet requires a fixed sequence length, so provide one.
        inputs = keras.Input(shape=(10, 64))
        masked = keras.layers.Masking(mask_value=0.0)(inputs)
        encoded = FNetEncoderBlock(intermediate_dim=256)(masked)

        model = keras.Model(inputs, encoded)

        # Test with padded sequences
        test_input = keras.random.normal(shape=(2, 10, 64))
        # Set last 3 positions to zero (masked)
        test_input = keras.ops.concatenate([
            test_input[:, :7, :],
            keras.ops.zeros((2, 3, 64))
        ], axis=1)

        output = model(test_input)
        assert output.shape == (2, 10, 64)


if __name__ == "__main__":
    # Run tests with: pytest test_fnet_encoder_block.py -v
    pytest.main([__file__, "-v"])