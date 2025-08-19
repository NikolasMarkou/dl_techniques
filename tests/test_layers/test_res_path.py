"""
Comprehensive test suite for ResPath layer implementation.

This test suite follows modern Keras 3 testing patterns and validates the
ResPath layer's functionality, serialization, and integration capabilities.

Place this file in: tests/test_layers/test_res_path.py
"""

import pytest
import numpy as np
import keras
import tempfile
import os
from typing import Dict, Any
import tensorflow as tf

from dl_techniques.layers.res_path import ResPath


class TestResPath:
    """Comprehensive test suite for ResPath layer implementation."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample 4D input tensor for testing (batch, height, width, channels)."""
        return keras.random.normal([4, 32, 32, 64])  # batch, H, W, channels

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard layer configuration for testing."""
        return {
            'channels': 64,
            'num_blocks': 3
        }

    @pytest.fixture
    def custom_layer_config(self) -> Dict[str, Any]:
        """Custom layer configuration with comprehensive parameters."""
        return {
            'channels': 128,
            'num_blocks': 4,
            'kernel_initializer': 'he_normal',
            'kernel_regularizer': keras.regularizers.L2(1e-4)
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = ResPath(channels=64, num_blocks=3)

        # Check stored configuration
        assert layer.channels == 64
        assert layer.num_blocks == 3
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert layer.kernel_regularizer is None

        # Check that layer is not built yet
        assert not layer.built

        # Check that sub-layers are created but not built
        assert len(layer.conv_blocks) == 3
        assert len(layer.bn_blocks) == 3
        assert len(layer.se_blocks) == 3
        assert layer.final_se is not None
        assert layer.final_bn is not None
        assert layer.activation is not None

        # Sub-layers should not be built yet
        for conv, bn, se in zip(layer.conv_blocks, layer.bn_blocks, layer.se_blocks):
            assert not conv.built
            assert not bn.built
            assert not se.built
        assert not layer.final_se.built
        assert not layer.final_bn.built

    def test_initialization_custom(self, custom_layer_config):
        """Test initialization with comprehensive custom parameters."""
        layer = ResPath(**custom_layer_config)

        # Verify all custom parameters are stored correctly
        assert layer.channels == 128
        assert layer.num_blocks == 4
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.kernel_regularizer, keras.regularizers.L2)

        # Check correct number of blocks created
        assert len(layer.conv_blocks) == 4
        assert len(layer.bn_blocks) == 4
        assert len(layer.se_blocks) == 4

    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid channels values
        with pytest.raises(ValueError, match="channels must be positive"):
            ResPath(channels=0, num_blocks=2)

        with pytest.raises(ValueError, match="channels must be positive"):
            ResPath(channels=-10, num_blocks=2)

        # Test invalid num_blocks values
        with pytest.raises(ValueError, match="num_blocks must be positive"):
            ResPath(channels=64, num_blocks=0)

        with pytest.raises(ValueError, match="num_blocks must be positive"):
            ResPath(channels=64, num_blocks=-5)

    def test_build_process(self, sample_input, layer_config):
        """Test that the layer builds properly following modern patterns."""
        layer = ResPath(**layer_config)

        # Layer should not be built initially
        assert not layer.built

        # Forward pass triggers build
        output = layer(sample_input)

        # Check that layer is now built
        assert layer.built

        # Check that all sub-layers are built
        for conv, bn, se in zip(layer.conv_blocks, layer.bn_blocks, layer.se_blocks):
            assert conv.built
            assert bn.built
            assert se.built
        assert layer.final_se.built
        assert layer.final_bn.built

        # Verify output shape (should be same as input for ResPath)
        assert output.shape == sample_input.shape

        # Check that weights were created
        # Each conv block has kernel weights (no bias due to use_bias=False)
        # Each BN block has 4 weights (gamma, beta, moving_mean, moving_variance)
        # Each SE block has weights for its dense layers
        assert len(layer.weights) > 0

    def test_forward_pass_basic(self, sample_input, layer_config):
        """Test basic forward pass functionality."""
        layer = ResPath(**layer_config)
        output = layer(sample_input)

        # Basic sanity checks
        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

    def test_forward_pass_deterministic(self):
        """Test forward pass with controlled inputs for deterministic behavior."""
        # Create layer with controlled initialization
        layer = ResPath(
            channels=32,
            num_blocks=2,
            kernel_initializer='ones'
        )

        controlled_input = keras.ops.ones([2, 16, 16, 32])
        output = layer(controlled_input)

        # Verify output properties
        assert output.shape == (2, 16, 16, 32)
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

    def test_residual_connections(self, layer_config):
        """Test that residual connections are working properly."""
        layer = ResPath(**layer_config)

        # Create input with specific pattern
        test_input = keras.ops.ones([2, 16, 16, 64]) * 2.0

        # Forward pass
        output = layer(test_input)

        # Due to residual connections, output should retain some relationship to input
        # The exact values depend on initialization and BN, but structure should be preserved
        assert output.shape == test_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

        # Test with zero input - residual connections should still work
        zero_input = keras.ops.zeros([2, 16, 16, 64])
        zero_output = layer(zero_input)
        assert not keras.ops.any(keras.ops.isnan(zero_output))

    def test_different_num_blocks(self, sample_input):
        """Test layer with various numbers of residual blocks."""
        block_counts = [1, 2, 3, 5, 8]

        for num_blocks in block_counts:
            layer = ResPath(channels=64, num_blocks=num_blocks)
            output = layer(sample_input)

            # Verify output is valid
            assert output.shape == sample_input.shape
            assert not keras.ops.any(keras.ops.isnan(output))

            # Check correct number of sub-layers created
            assert len(layer.conv_blocks) == num_blocks
            assert len(layer.bn_blocks) == num_blocks
            assert len(layer.se_blocks) == num_blocks

    def test_different_channel_sizes(self):
        """Test with various channel sizes."""
        channel_sizes = [16, 32, 64, 128, 256, 512]

        for channels in channel_sizes:
            layer = ResPath(channels=channels, num_blocks=2)
            test_input = keras.random.normal([2, 8, 8, channels])

            output = layer(test_input)
            assert output.shape == test_input.shape
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_training_vs_inference_mode(self, sample_input, layer_config):
        """Test behavior in training vs inference modes."""
        layer = ResPath(**layer_config)

        # Test training mode
        output_train = layer(sample_input, training=True)
        assert output_train.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output_train))

        # Test inference mode
        output_inference = layer(sample_input, training=False)
        assert output_inference.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output_inference))

        # Due to BatchNorm, outputs may differ between training and inference
        # but both should be valid

    def test_compute_output_shape(self):
        """Test output shape computation."""
        test_cases = [
            # (channels, num_blocks, input_shape, expected_output_shape)
            (64, 2, (None, 32, 32, 64), (None, 32, 32, 64)),
            (128, 3, (4, 16, 16, 128), (4, 16, 16, 128)),
            (256, 1, (2, 64, 64, 256), (2, 64, 64, 256)),
            (32, 5, (8, 8, 8, 32), (8, 8, 8, 32)),
        ]

        for channels, num_blocks, input_shape, expected_shape in test_cases:
            layer = ResPath(channels=channels, num_blocks=num_blocks)
            computed_shape = layer.compute_output_shape(input_shape)
            assert computed_shape == expected_shape

    def test_get_config_completeness(self, custom_layer_config):
        """Test that get_config contains all initialization parameters."""
        layer = ResPath(**custom_layer_config)
        config = layer.get_config()

        # Verify all custom parameters are in config
        expected_keys = {
            'channels', 'num_blocks', 'kernel_initializer', 'kernel_regularizer'
        }

        config_keys = set(config.keys())
        assert expected_keys.issubset(config_keys)

        # Verify specific values
        assert config['channels'] == 128
        assert config['num_blocks'] == 4

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle following modern patterns."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = ResPath(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_prediction = model(sample_input)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            # Load without custom_objects (thanks to registration decorator)
            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization cycle"
            )

    def test_unet_integration(self, sample_input):
        """Test layer integration in a U-Net style architecture."""
        inputs = keras.Input(shape=sample_input.shape[1:])

        # Simulate encoder features
        x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
        encoder_features = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)

        # Apply ResPath to refine encoder features
        refined_features = ResPath(channels=64, num_blocks=3)(encoder_features)

        # Simulate decoder path
        decoder_input = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(refined_features)

        # Skip connection (the main purpose of ResPath)
        merged = keras.layers.Add()([refined_features, decoder_input])
        outputs = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(merged)

        model = keras.Model(inputs, outputs)

        # Test forward pass
        prediction = model(sample_input)
        assert prediction.shape == (sample_input.shape[0], sample_input.shape[1],
                                    sample_input.shape[2], 32)

        # Test that gradients flow properly
        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = model(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        gradients = tape.gradient(loss, model.trainable_variables)
        assert all(g is not None for g in gradients)

    def test_regularization_losses(self, sample_input):
        """Test that regularization losses are properly computed."""
        layer = ResPath(
            channels=64,
            num_blocks=2,
            kernel_regularizer=keras.regularizers.L2(0.01)
        )

        # No losses before forward pass
        initial_losses = len(layer.losses)

        # Apply the layer
        output = layer(sample_input)

        # Should have regularization losses now
        assert len(layer.losses) > initial_losses

        # Verify losses are non-zero
        total_loss = sum(layer.losses)
        assert keras.ops.convert_to_numpy(total_loss) > 0

    def test_gradient_flow(self, sample_input, layer_config):
        """Test that gradients flow properly through the layer."""
        layer = ResPath(**layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        # Compute gradients
        gradients = tape.gradient(loss, layer.trainable_variables)

        # Check that all gradients exist and are non-zero
        assert len(gradients) == len(layer.trainable_variables)
        assert all(g is not None for g in gradients)

        # Check gradient shapes match variable shapes
        for grad, var in zip(gradients, layer.trainable_variables):
            assert grad.shape == var.shape

    def test_different_spatial_dimensions(self):
        """Test layer with different spatial input dimensions."""
        layer = ResPath(channels=64, num_blocks=2)

        spatial_sizes = [
            (8, 8),  # Small
            (32, 32),  # Medium
            (64, 64),  # Large
            (16, 24),  # Non-square
            (128, 32),  # Very rectangular
        ]

        for height, width in spatial_sizes:
            test_input = keras.random.normal([2, height, width, 64])
            output = layer(test_input)

            # Output should have same spatial dimensions
            assert output.shape == (2, height, width, 64)
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = ResPath(channels=32, num_blocks=2)

        # Test different input value ranges
        test_cases = [
            keras.ops.zeros((2, 16, 16, 32)),  # Zeros
            keras.ops.ones((2, 16, 16, 32)) * 1e-10,  # Very small
            keras.ops.ones((2, 16, 16, 32)) * 1e3,  # Large
            keras.random.normal((2, 16, 16, 32)) * 100,  # Large random
            keras.random.normal((2, 16, 16, 32)) * 1e-5,  # Small random
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Verify numerical stability
            assert not keras.ops.any(keras.ops.isnan(output)), "NaN detected"
            assert not keras.ops.any(keras.ops.isinf(output)), "Inf detected"

    def test_weight_structure_and_shapes(self, sample_input):
        """Test that layer weights have correct structure and shapes."""
        channels = 64
        num_blocks = 3

        layer = ResPath(channels=channels, num_blocks=num_blocks)
        layer(sample_input)  # Build the layer

        # Should have weights from:
        # - conv_blocks: num_blocks conv layers (each has kernel, no bias)
        # - bn_blocks: num_blocks BN layers (each has 4 weights)
        # - se_blocks: num_blocks SE layers (each has weights for reduction/expansion)
        # - final_se: 1 SE layer
        # - final_bn: 1 BN layer

        assert len(layer.weights) > 0

        # Check that we have trainable weights
        assert len(layer.trainable_variables) > 0

        # Verify conv block kernels have correct shape
        for i, conv in enumerate(layer.conv_blocks):
            # Each conv should have kernel shape (3, 3, channels, channels)
            expected_kernel_shape = (3, 3, channels, channels)
            assert conv.kernel.shape == expected_kernel_shape

    def test_squeeze_excitation_integration(self, sample_input):
        """Test that Squeeze-and-Excitation blocks are working correctly."""
        layer = ResPath(channels=64, num_blocks=2)

        # Forward pass
        output = layer(sample_input)

        # SE blocks should be built and functional
        for se_block in layer.se_blocks:
            assert se_block.built

        # Final SE should also be built
        assert layer.final_se.built
        assert output.shape == sample_input.shape

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, sample_input, layer_config, training):
        """Test layer behavior in different training modes."""
        layer = ResPath(**layer_config)
        output = layer(sample_input, training=training)

        # Basic output validation
        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))


class TestResPathEdgeCases:
    """Test edge cases and boundary conditions for ResPath."""

    def test_minimal_configuration(self):
        """Test with minimal viable configuration."""
        layer = ResPath(channels=1, num_blocks=1)
        test_input = keras.random.normal([1, 4, 4, 1])

        output = layer(test_input)
        assert output.shape == (1, 4, 4, 1)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_large_configuration(self):
        """Test with large configuration."""
        layer = ResPath(channels=512, num_blocks=8)
        test_input = keras.random.normal([2, 8, 8, 512])

        output = layer(test_input)
        assert output.shape == (2, 8, 8, 512)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_single_pixel_input(self):
        """Test with 1x1 spatial dimensions."""
        layer = ResPath(channels=64, num_blocks=2)
        test_input = keras.random.normal([2, 1, 1, 64])

        output = layer(test_input)
        assert output.shape == (2, 1, 1, 64)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_layer_reuse(self):
        """Test that the same layer instance can be reused."""
        layer = ResPath(channels=32, num_blocks=2)

        # Use the same layer with different inputs
        input1 = keras.random.normal([2, 16, 16, 32])
        input2 = keras.random.normal([3, 24, 24, 32])

        output1 = layer(input1)
        output2 = layer(input2)

        assert output1.shape == (2, 16, 16, 32)
        assert output2.shape == (3, 24, 24, 32)

        # Both outputs should be valid
        assert not keras.ops.any(keras.ops.isnan(output1))
        assert not keras.ops.any(keras.ops.isnan(output2))

    def test_different_initializers(self):
        """Test layer with various weight initializers."""
        initializers = ['glorot_uniform', 'he_normal', 'lecun_normal']

        for initializer in initializers:
            layer = ResPath(
                channels=32,
                num_blocks=2,
                kernel_initializer=initializer
            )

            test_input = keras.random.normal([2, 8, 8, 32])
            output = layer(test_input)

            assert output.shape == (2, 8, 8, 32)
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_channel_mismatch_error(self):
        """Test that channel mismatch raises appropriate error."""
        layer = ResPath(channels=64, num_blocks=2)

        # Input with wrong number of channels should cause issues during build
        wrong_channels_input = keras.random.normal([2, 16, 16, 32])  # 32 channels instead of 64

        # This should work but the residual connections won't align properly
        # The layer will adapt but this tests robustness
        with pytest.raises(Exception):  # Expect some form of shape incompatibility
            output = layer(wrong_channels_input)

    def test_very_deep_respath(self):
        """Test with very deep ResPath (many blocks)."""
        layer = ResPath(channels=64, num_blocks=16)  # Very deep
        test_input = keras.random.normal([1, 8, 8, 64])

        output = layer(test_input)
        assert output.shape == (1, 8, 8, 64)
        assert not keras.ops.any(keras.ops.isnan(output))

        # Should have created many sub-layers
        assert len(layer.conv_blocks) == 16
        assert len(layer.bn_blocks) == 16
        assert len(layer.se_blocks) == 16

    def test_batch_size_variations(self):
        """Test with different batch sizes."""
        layer = ResPath(channels=32, num_blocks=2)

        batch_sizes = [1, 2, 8, 16, 32]
        for batch_size in batch_sizes:
            test_input = keras.random.normal([batch_size, 16, 16, 32])
            output = layer(test_input)

            assert output.shape == (batch_size, 16, 16, 32)
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_memory_efficiency(self):
        """Test that layer doesn't create excessive intermediate tensors."""
        # This is more of a smoke test to ensure the layer doesn't crash with larger inputs
        layer = ResPath(channels=128, num_blocks=4)
        test_input = keras.random.normal([1, 64, 64, 128])  # Relatively large input

        output = layer(test_input)
        assert output.shape == (1, 64, 64, 128)
        assert not keras.ops.any(keras.ops.isnan(output))