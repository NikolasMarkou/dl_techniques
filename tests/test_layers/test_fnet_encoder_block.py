"""
Comprehensive test suite for FNetEncoderBlock following modern Keras 3 best practices.

This test suite validates all aspects of the refined FNetEncoderBlock implementation including
initialization, forward pass, serialization, configuration management, factory patterns, and edge cases.
"""

import pytest
import tempfile
import os
from typing import Any, Dict

import numpy as np
import keras
import tensorflow as tf

from dl_techniques.layers.fnet_encoder_block import FNetEncoderBlock


class TestFNetEncoderBlock:
    """Comprehensive test suite for FNetEncoderBlock layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing with new parameter structure."""
        return {
            'intermediate_dim': 256,
            'dropout_rate': 0.1,
            'normalization_type': 'layer_norm',
            'normalization_kwargs': {'epsilon': 1e-12},
            'ffn_type': 'mlp',
            'ffn_kwargs': {'activation': 'gelu', 'use_bias': True}
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample input for testing - 3D tensor [batch, seq, hidden]."""
        return keras.random.normal(shape=(4, 16, 64))

    @pytest.fixture
    def bert_like_input(self) -> keras.KerasTensor:
        """BERT-like dimensions for realistic testing."""
        return keras.random.normal(shape=(2, 512, 768))

    def test_initialization_basic(self):
        """Test basic layer initialization with default configuration."""
        layer = FNetEncoderBlock(intermediate_dim=256)

        # Check configuration attributes
        assert layer.intermediate_dim == 256
        assert layer.dropout_rate == 0.1  # default
        assert layer.normalization_type == 'layer_norm'  # default
        assert layer.ffn_type == 'mlp'  # default
        assert not layer.built  # Should not be built yet

        # Check that sub-layers will be created in build()
        assert layer.fourier_transform is not None
        assert layer.fourier_layer_norm is None  # Created in build()
        assert layer.ffn_layer is None  # Created in build()
        assert layer.output_layer_norm is None  # Created in build()

    def test_initialization_with_custom_config(self, layer_config):
        """Test initialization with custom configuration."""
        layer = FNetEncoderBlock(**layer_config)

        assert layer.intermediate_dim == 256
        assert layer.dropout_rate == 0.1
        assert layer.normalization_type == 'layer_norm'
        assert layer.ffn_type == 'mlp'
        assert layer.normalization_kwargs == {'epsilon': 1e-12}
        assert layer.ffn_kwargs == {'activation': 'gelu', 'use_bias': True}

    def test_initialization_modern_config(self):
        """Test initialization with modern configuration (RMS norm + SwiGLU)."""
        layer = FNetEncoderBlock(
            intermediate_dim=None,  # Not used for SwiGLU
            normalization_type='rms_norm',
            normalization_kwargs={'use_scale': True},
            ffn_type='swiglu',
            ffn_kwargs={'ffn_expansion_factor': 4}
        )

        assert layer.intermediate_dim is None
        assert layer.normalization_type == 'rms_norm'
        assert layer.ffn_type == 'swiglu'
        assert layer.ffn_kwargs['ffn_expansion_factor'] == 4

    def test_initialization_with_fourier_config(self):
        """Test initialization with custom fourier configuration."""
        fourier_config = {'normalize_dft': False, 'epsilon': 1e-10}
        layer = FNetEncoderBlock(
            intermediate_dim=128,
            fourier_config=fourier_config
        )

        assert layer.intermediate_dim == 128
        assert layer.fourier_config == fourier_config

    def test_forward_pass_basic(self, layer_config, sample_input):
        """Test basic forward pass and building."""
        layer = FNetEncoderBlock(**layer_config)

        # Forward pass should trigger building
        output = layer(sample_input)

        # Check layer is now built
        assert layer.built

        # Check sub-layers are created and built
        assert layer.fourier_layer_norm is not None
        assert layer.ffn_layer is not None
        assert layer.output_layer_norm is not None

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

    def test_forward_pass_different_normalization_types(self, sample_input):
        """Test forward pass with different normalization types."""
        norm_types = ['layer_norm', 'rms_norm', 'batch_norm', 'band_rms']

        outputs = []
        for norm_type in norm_types:
            layer = FNetEncoderBlock(
                intermediate_dim=128,
                normalization_type=norm_type
            )
            output = layer(sample_input)
            outputs.append(keras.ops.convert_to_numpy(output))

            # Shape should be preserved
            assert output.shape == sample_input.shape

        # Different normalizations should produce different outputs
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not np.allclose(outputs[i], outputs[j], atol=1e-6), \
                    f"Normalization types {norm_types[i]} and {norm_types[j]} produce identical outputs"

    def test_forward_pass_different_ffn_types(self, sample_input):
        """Test forward pass with different FFN types."""
        ffn_configs = [
            {'ffn_type': 'mlp', 'intermediate_dim': 128},
            {'ffn_type': 'glu', 'intermediate_dim': 128},
            {'ffn_type': 'geglu', 'intermediate_dim': 128},
            {'ffn_type': 'swiglu', 'intermediate_dim': None, 'ffn_kwargs': {'ffn_expansion_factor': 2}},
        ]

        outputs = []
        for config in ffn_configs:
            layer = FNetEncoderBlock(**config)
            output = layer(sample_input)
            outputs.append(keras.ops.convert_to_numpy(output))

            # Shape should be preserved
            assert output.shape == sample_input.shape

        # Different FFN types should produce different outputs
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not np.allclose(outputs[i], outputs[j], atol=1e-6), \
                    f"FFN types produce identical outputs for configs {i} and {j}"

    def test_advanced_normalization_types(self, sample_input):
        """Test forward pass with advanced normalization types."""
        advanced_norm_configs = [
            {'normalization_type': 'adaptive_band_rms', 'normalization_kwargs': {'max_band_width': 0.15}},
            {'normalization_type': 'global_response_norm'},
            {'normalization_type': 'dynamic_tanh', 'normalization_kwargs': {'alpha_init_value': 0.5}},
            {'normalization_type': 'logit_norm', 'normalization_kwargs': {'temperature': 1.5}},
        ]

        for norm_config in advanced_norm_configs:
            layer = FNetEncoderBlock(
                intermediate_dim=128,
                **norm_config
            )
            output = layer(sample_input)

            # Shape should be preserved
            assert output.shape == sample_input.shape

            # Check that layer is functioning
            output_np = keras.ops.convert_to_numpy(output)
            assert not np.isnan(output_np).any(), f"NaN values in output for {norm_config['normalization_type']}"
            assert np.isfinite(output_np).all(), f"Non-finite values in output for {norm_config['normalization_type']}"

    def test_advanced_ffn_types(self, sample_input):
        """Test forward pass with advanced FFN types."""
        advanced_ffn_configs = [
            {'ffn_type': 'differential', 'intermediate_dim': 128},
            {'ffn_type': 'residual', 'intermediate_dim': 128},
            {'ffn_type': 'swin_mlp', 'intermediate_dim': 128},
        ]

        for ffn_config in advanced_ffn_configs:
            layer = FNetEncoderBlock(**ffn_config)
            output = layer(sample_input)

            # Shape should be preserved
            assert output.shape == sample_input.shape

            # Check that layer is functioning
            output_np = keras.ops.convert_to_numpy(output)
            assert not np.isnan(output_np).any(), f"NaN values in output for {ffn_config['ffn_type']}"
            assert np.isfinite(output_np).all(), f"Non-finite values in output for {ffn_config['ffn_type']}"

    def test_serialization_cycle_basic(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle with basic config."""
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

    def test_serialization_modern_config(self, sample_input):
        """Test serialization with modern configuration (RMS + SwiGLU)."""
        config = {
            'intermediate_dim': None,
            'normalization_type': 'rms_norm',
            'normalization_kwargs': {'use_scale': True},
            'ffn_type': 'swiglu',
            'ffn_kwargs': {'ffn_expansion_factor': 4}
        }

        # Create and test model
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = FNetEncoderBlock(**config)(inputs)
        model = keras.Model(inputs, outputs)

        original_pred = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_modern_fnet.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Modern config serialization failed"
            )

    def test_serialization_with_fourier_config(self, sample_input):
        """Test serialization with custom fourier configuration."""
        config = {
            'intermediate_dim': 128,
            'fourier_config': {'normalize_dft': False, 'implementation': 'matrix'},
            'normalization_type': 'band_rms',
            'normalization_kwargs': {'max_band_width': 0.1}
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

    def test_serialization_complex_config(self, sample_input):
        """Test serialization with complex configuration combining various advanced features."""
        config = {
            'intermediate_dim': None,  # Using expansion factor instead
            'dropout_rate': 0.15,
            'fourier_config': {'normalize_dft': True, 'epsilon': 1e-8},
            'normalization_type': 'adaptive_band_rms',
            'normalization_kwargs': {
                'max_band_width': 0.2,
                'epsilon': 1e-7,
            },
            'ffn_type': 'swiglu',
            'ffn_kwargs': {
                'ffn_expansion_factor': 6,
                'dropout_rate': 0.12,
                'use_bias': False
            }
        }

        # Create and test model
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = FNetEncoderBlock(**config)(inputs)
        model = keras.Model(inputs, outputs)

        original_pred = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_complex_fnet.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Complex config serialization failed"
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
            'intermediate_dim', 'dropout_rate', 'fourier_config',
            'normalization_type', 'normalization_kwargs',
            'ffn_type', 'ffn_kwargs'
        ]

        for key in expected_keys:
            assert key in config, f"Expected key {key} not in config"

    def test_from_config(self, layer_config, sample_input):
        """Test creating layer from configuration dictionary."""
        original_layer = FNetEncoderBlock(**layer_config)
        original_output = original_layer(sample_input)

        # Get config and create new layer
        config = original_layer.get_config()
        restored_layer = FNetEncoderBlock.from_config(config)
        restored_output = restored_layer(sample_input)

        # Outputs should be identical (same weights initialization with same seed)
        # Note: This might not be exactly identical due to random initialization
        # So we'll just check that both layers work and have same config
        assert restored_layer.get_config() == original_layer.get_config()
        assert restored_output.shape == original_output.shape

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

    def test_layer_builds_sublayers_correctly(self, layer_config, sample_input):
        """Test that sub-layers are properly built."""
        layer = FNetEncoderBlock(**layer_config)

        # Before forward pass, sub-layers should not be built
        assert not layer.fourier_transform.built
        assert layer.fourier_layer_norm is None  # Created in build()
        assert layer.ffn_layer is None  # Created in build()

        # Forward pass should build everything
        _ = layer(sample_input)

        # Now all sub-layers should be built
        assert layer.built
        assert layer.fourier_transform.built
        assert layer.fourier_layer_norm.built
        assert layer.ffn_layer.built
        assert layer.output_layer_norm.built

    def test_ffn_fallback_mechanism(self, sample_input):
        """Test FFN fallback mechanism when factory creation fails."""
        # This test assumes the fallback mechanism is implemented as shown in the code
        # Create a configuration that might cause FFN creation to fail
        config = {
            'intermediate_dim': 128,
            'ffn_type': 'mlp',
            'ffn_kwargs': {'invalid_parameter': 'invalid_value'}  # Invalid parameter
        }

        # The layer should still work due to fallback mechanism
        layer = FNetEncoderBlock(**config)
        output = layer(sample_input)

        # Should still produce valid output despite invalid FFN config
        assert output.shape == sample_input.shape
        assert layer.ffn_layer is not None

    def test_edge_cases_validation(self):
        """Test error conditions and edge cases."""
        # Invalid intermediate_dim (when provided)
        with pytest.raises(ValueError, match="intermediate_dim must be positive"):
            FNetEncoderBlock(intermediate_dim=0)

        with pytest.raises(ValueError, match="intermediate_dim must be positive"):
            FNetEncoderBlock(intermediate_dim=-10)

        # Invalid dropout_rate
        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            FNetEncoderBlock(intermediate_dim=64, dropout_rate=-0.1)

        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            FNetEncoderBlock(intermediate_dim=64, dropout_rate=1.5)

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

    def test_compute_mask(self, layer_config):
        """Test compute_mask method."""
        layer = FNetEncoderBlock(**layer_config)

        # Input mask should be preserved
        input_mask = keras.random.uniform(minval=0, maxval=1, shape=(4, 16), dtype='float32')
        input_mask = input_mask > 0.5
        output_mask = layer.compute_mask(None, input_mask)

        assert output_mask is input_mask  # Should return the same mask

        # No mask case
        output_mask_none = layer.compute_mask(None, None)
        assert output_mask_none is None

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

    def test_attention_mask_propagation(self, layer_config, sample_input):
        """Test that attention mask is properly passed to Fourier transform."""
        layer = FNetEncoderBlock(**layer_config)

        # Create a simple attention mask
        attention_mask = keras.ops.ones(shape=(4, 16), dtype='bool')

        # Should not raise an error
        output = layer(sample_input, attention_mask=attention_mask)
        assert output.shape == sample_input.shape

    def test_supports_masking_attribute(self, layer_config):
        """Test that layer properly declares masking support."""
        layer = FNetEncoderBlock(**layer_config)
        assert layer.supports_masking is True


# Additional integration tests
class TestFNetEncoderBlockIntegration:
    """Integration tests for FNetEncoderBlock in larger contexts."""

    def test_in_sequential_model(self):
        """Test FNetEncoderBlock as part of a Sequential model."""
        model = keras.Sequential([
            keras.layers.Embedding(1000, 64, input_shape=(32,)),
            FNetEncoderBlock(intermediate_dim=256),
            FNetEncoderBlock(
                intermediate_dim=256,
                normalization_type='rms_norm',
                normalization_kwargs={'use_scale': True}
            ),
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
        sample_tokens = keras.random.randint(minval=0, maxval=1000, shape=(4, 32))
        predictions = model(sample_tokens)

        assert predictions.shape == (4, 10)

    def test_stacked_layers_different_configs(self):
        """Test multiple FNetEncoderBlocks with different configurations stacked together."""
        inputs = keras.Input(shape=(64, 128))
        x = inputs

        # Stack 3 encoder blocks with different configurations
        configs = [
            {
                'intermediate_dim': 512,
                'normalization_type': 'layer_norm',
                'ffn_type': 'mlp'
            },
            {
                'intermediate_dim': None,
                'normalization_type': 'rms_norm',
                'normalization_kwargs': {'use_scale': True},
                'ffn_type': 'swiglu',
                'ffn_kwargs': {'ffn_expansion_factor': 4}
            },
            {
                'intermediate_dim': 256,
                'normalization_type': 'band_rms',
                'normalization_kwargs': {'max_band_width': 0.1},
                'ffn_type': 'glu'
            }
        ]

        for i, config in enumerate(configs):
            x = FNetEncoderBlock(name=f'fnet_block_{i}', **config)(x)

        outputs = keras.layers.GlobalAveragePooling1D()(x)
        model = keras.Model(inputs, outputs)

        test_input = keras.random.normal(shape=(2, 64, 128))
        output = model(test_input)

        assert output.shape == (2, 128)

    def test_with_masking(self):
        """Test FNetEncoderBlock with masked inputs."""
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

    def test_mixed_precision_compatibility(self):
        """Test FNetEncoderBlock works with mixed precision."""
        # Enable mixed precision
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)

        try:
            layer = FNetEncoderBlock(intermediate_dim=128)
            test_input = keras.random.normal(shape=(2, 16, 64), dtype='float16')

            output = layer(test_input)
            assert output.shape == test_input.shape

        finally:
            # Reset policy
            keras.mixed_precision.set_global_policy('float32')

    def test_large_model_simulation(self):
        """Test FNetEncoderBlock in a larger transformer-like model."""
        # Simulate a small transformer model
        vocab_size = 1000
        seq_length = 128
        embed_dim = 256
        num_layers = 4

        inputs = keras.Input(shape=(seq_length,), dtype='int32')

        # Embedding
        x = keras.layers.Embedding(vocab_size, embed_dim)(inputs)

        # Multiple FNet encoder blocks with different configurations
        for i in range(num_layers):
            if i == 0:
                # First layer with standard config
                config = {
                    'intermediate_dim': embed_dim * 4,
                    'dropout_rate': 0.1,
                    'name': f'fnet_layer_{i}'
                }
            elif i == 1:
                # Second layer with modern config
                config = {
                    'intermediate_dim': None,
                    'dropout_rate': 0.1,
                    'normalization_type': 'rms_norm',
                    'normalization_kwargs': {'use_scale': True},
                    'ffn_type': 'swiglu',
                    'ffn_kwargs': {'ffn_expansion_factor': 4},
                    'name': f'fnet_layer_{i}'
                }
            else:
                # Remaining layers with mixed configs
                config = {
                    'intermediate_dim': embed_dim * 3,
                    'dropout_rate': 0.1,
                    'normalization_type': 'band_rms',
                    'normalization_kwargs': {'max_band_width': 0.1},
                    'ffn_type': 'glu',
                    'name': f'fnet_layer_{i}'
                }

            x = FNetEncoderBlock(**config)(x)

        # Final layers
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(2, activation='softmax')(x)

        model = keras.Model(inputs, outputs)

        # Test forward pass
        test_input = keras.random.randint(minval=0, maxval=vocab_size, shape=(4, seq_length))
        predictions = model(test_input)

        assert predictions.shape == (4, 2)

        # Test that model can be compiled and has reasonable number of parameters
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        assert model.count_params() > 0

    def test_factory_pattern_consistency(self):
        """Test that factory pattern creates consistent layer types."""
        sample_input = keras.random.normal(shape=(2, 16, 64))

        # Test different normalization factory usage
        norm_configs = [
            {'normalization_type': 'layer_norm'},
            {'normalization_type': 'rms_norm', 'normalization_kwargs': {'use_scale': True}},
            {'normalization_type': 'band_rms', 'normalization_kwargs': {'max_band_width': 0.1}},
        ]

        for config in norm_configs:
            layer = FNetEncoderBlock(intermediate_dim=128, **config)
            output = layer(sample_input)

            assert output.shape == sample_input.shape
            # Verify that the correct normalization layers were created
            assert layer.fourier_layer_norm is not None
            assert layer.output_layer_norm is not None

    def test_comprehensive_factory_combinations(self):
        """Test comprehensive combinations of factory-created layers."""
        sample_input = keras.random.normal(shape=(2, 32, 128))

        # Test matrix of normalization x FFN combinations
        norm_types = ['layer_norm', 'rms_norm', 'band_rms']
        ffn_types = ['mlp', 'glu', 'swiglu']

        for norm_type in norm_types:
            for ffn_type in ffn_types:
                config = {
                    'intermediate_dim': 256 if ffn_type != 'swiglu' else None,
                    'normalization_type': norm_type,
                    'ffn_type': ffn_type
                }

                if ffn_type == 'swiglu':
                    config['ffn_kwargs'] = {'ffn_expansion_factor': 4}

                layer = FNetEncoderBlock(**config)
                output = layer(sample_input)

                assert output.shape == sample_input.shape, \
                    f"Failed for norm={norm_type}, ffn={ffn_type}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])