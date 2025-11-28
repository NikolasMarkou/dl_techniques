"""
Comprehensive test suite for HANCBlock layer implementation.

This test suite follows modern Keras 3 testing patterns and validates the
HANCBlock layer's functionality, serialization, and integration capabilities.
The tests cover the core innovation of ACC-UNet's hierarchical context aggregation.
"""

import pytest
import numpy as np
import keras
import tempfile
import os
from typing import Dict, Any
import tensorflow as tf

from dl_techniques.layers.hanc_block import HANCBlock


class TestHANCBlock:
    """Comprehensive test suite for HANCBlock layer implementation."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample 4D input tensor for testing (batch, height, width, channels)."""
        return keras.random.normal([4, 32, 32, 64])  # batch, H, W, channels

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard layer configuration for testing."""
        return {
            'filters': 64,
            'input_channels': 64,
            'k': 3,
            'inv_factor': 3
        }

    @pytest.fixture
    def custom_layer_config(self) -> Dict[str, Any]:
        """Custom layer configuration with comprehensive parameters."""
        return {
            'filters': 128,
            'input_channels': 64,
            'k': 4,
            'inv_factor': 4,
            'kernel_initializer': 'he_normal',
            'kernel_regularizer': keras.regularizers.L2(1e-4)
        }

    @pytest.fixture
    def residual_config(self) -> Dict[str, Any]:
        """Configuration that enables residual connections (input_channels == filters)."""
        return {
            'filters': 64,
            'input_channels': 64,  # Same as filters to enable residual
            'k': 3,
            'inv_factor': 3
        }

    @pytest.fixture
    def non_residual_config(self) -> Dict[str, Any]:
        """Configuration that disables residual connections (input_channels != filters)."""
        return {
            'filters': 128,
            'input_channels': 64,  # Different from filters to disable residual
            'k': 3,
            'inv_factor': 3
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = HANCBlock(filters=64, input_channels=32)

        # Check stored configuration
        assert layer.filters == 64
        assert layer.input_channels == 32
        assert layer.k == 3  # Default value
        assert layer.inv_factor == 3  # Default value
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert layer.kernel_regularizer is None

        # Check computed derived parameters
        assert layer.expanded_channels == 32 * 3  # input_channels * inv_factor
        assert layer.use_residual == False  # 32 != 64

        # Check that layer is not built yet
        assert not layer.built

        # Check that all sub-layers are created but not built
        assert layer.expand_conv is not None
        assert layer.expand_bn is not None
        assert layer.depthwise_conv is not None
        assert layer.depthwise_bn is not None
        assert layer.hanc_layer is not None
        assert layer.residual_bn is None  # Should be None when use_residual=False
        assert layer.output_conv is not None
        assert layer.output_bn is not None
        assert layer.squeeze_excitation is not None
        assert layer.activation is not None

        # Sub-layers should not be built yet
        assert not layer.expand_conv.built
        assert not layer.expand_bn.built
        assert not layer.depthwise_conv.built
        assert not layer.depthwise_bn.built

    def test_initialization_custom(self, custom_layer_config):
        """Test initialization with comprehensive custom parameters."""
        layer = HANCBlock(**custom_layer_config)

        # Verify all custom parameters are stored correctly
        assert layer.filters == 128
        assert layer.input_channels == 64
        assert layer.k == 4
        assert layer.inv_factor == 4
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.kernel_regularizer, keras.regularizers.L2)

        # Check computed parameters
        assert layer.expanded_channels == 64 * 4  # 256
        assert layer.use_residual == False  # 64 != 128

    def test_residual_vs_non_residual_initialization(self, residual_config, non_residual_config):
        """Test that residual connection logic is correctly initialized."""
        # Test residual configuration
        residual_layer = HANCBlock(**residual_config)
        assert residual_layer.use_residual == True
        assert residual_layer.residual_bn is not None

        # Test non-residual configuration
        non_residual_layer = HANCBlock(**non_residual_config)
        assert non_residual_layer.use_residual == False
        assert non_residual_layer.residual_bn is None

    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid filters values
        with pytest.raises(ValueError, match="filters must be positive"):
            HANCBlock(filters=0, input_channels=32)

        with pytest.raises(ValueError, match="filters must be positive"):
            HANCBlock(filters=-10, input_channels=32)

        # Test invalid input_channels values
        with pytest.raises(ValueError, match="input_channels must be positive"):
            HANCBlock(filters=64, input_channels=0)

        with pytest.raises(ValueError, match="input_channels must be positive"):
            HANCBlock(filters=64, input_channels=-5)

        # Test invalid k values
        with pytest.raises(ValueError, match="k must be between 1 and 5"):
            HANCBlock(filters=64, input_channels=32, k=0)

        with pytest.raises(ValueError, match="k must be between 1 and 5"):
            HANCBlock(filters=64, input_channels=32, k=6)

        # Test invalid inv_factor values
        with pytest.raises(ValueError, match="inv_factor must be positive"):
            HANCBlock(filters=64, input_channels=32, inv_factor=0)

        with pytest.raises(ValueError, match="inv_factor must be positive"):
            HANCBlock(filters=64, input_channels=32, inv_factor=-2)

    def test_build_process(self, sample_input, layer_config):
        """Test that the layer builds properly following modern patterns."""
        layer = HANCBlock(**layer_config)

        # Layer should not be built initially
        assert not layer.built

        # Forward pass triggers build
        output = layer(sample_input)

        # Check that layer is now built
        assert layer.built

        # Check that all sub-layers are built
        assert layer.expand_conv.built
        assert layer.expand_bn.built
        assert layer.depthwise_conv.built
        assert layer.depthwise_bn.built
        assert layer.hanc_layer.built
        assert layer.output_conv.built
        assert layer.output_bn.built
        assert layer.squeeze_excitation.built

        # For residual config, residual_bn should exist and be built
        if layer.residual_bn is not None:
            assert layer.residual_bn.built

        # Verify output shape
        expected_shape = (sample_input.shape[0], sample_input.shape[1],
                         sample_input.shape[2], layer_config['filters'])
        assert output.shape == expected_shape

        # Check that weights were created
        assert len(layer.weights) > 0
        assert len(layer.trainable_variables) > 0

    def test_build_input_shape_validation(self):
        """Test input shape validation during build."""
        layer = HANCBlock(filters=64, input_channels=32)

        # Test with wrong number of dimensions
        with pytest.raises(ValueError, match="Expected 4D input shape"):
            layer.build((None, 32))  # 2D instead of 4D

        # Test with channel mismatch
        with pytest.raises(ValueError, match="Input channels mismatch"):
            layer.build((None, 32, 32, 64))  # 64 channels instead of expected 32

    def test_forward_pass_basic(self, sample_input, layer_config):
        """Test basic forward pass functionality."""
        layer = HANCBlock(**layer_config)
        output = layer(sample_input)

        # Basic sanity checks
        expected_shape = (sample_input.shape[0], sample_input.shape[1],
                         sample_input.shape[2], layer_config['filters'])
        assert output.shape == expected_shape
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

    def test_forward_pass_residual_vs_non_residual(self):
        """Test forward pass with and without residual connections."""
        test_input = keras.random.normal([2, 16, 16, 64])

        # Test with residual connection (input_channels == filters)
        residual_layer = HANCBlock(filters=64, input_channels=64, k=2)
        residual_output = residual_layer(test_input)
        assert residual_output.shape == (2, 16, 16, 64)
        assert not keras.ops.any(keras.ops.isnan(residual_output))

        # Test without residual connection (input_channels != filters)
        non_residual_layer = HANCBlock(filters=128, input_channels=64, k=2)
        non_residual_output = non_residual_layer(test_input)
        assert non_residual_output.shape == (2, 16, 16, 128)
        assert not keras.ops.any(keras.ops.isnan(non_residual_output))

    def test_different_k_values(self, sample_input):
        """Test layer with various hierarchical levels (k values)."""
        k_values = [1, 2, 3, 4, 5]

        for k in k_values:
            layer = HANCBlock(filters=64, input_channels=64, k=k)
            output = layer(sample_input)

            # Verify output is valid
            assert output.shape == (sample_input.shape[0], sample_input.shape[1],
                                   sample_input.shape[2], 64)
            assert not keras.ops.any(keras.ops.isnan(output))

            # HANC layer should be configured with correct k
            assert layer.hanc_layer.k == k

    def test_different_inv_factors(self):
        """Test with various inverted bottleneck expansion factors."""
        inv_factors = [1, 2, 3, 4, 6, 8]
        test_input = keras.random.normal([2, 16, 16, 32])

        for inv_factor in inv_factors:
            layer = HANCBlock(filters=64, input_channels=32, inv_factor=inv_factor)
            output = layer(test_input)

            # Check expansion computation
            expected_expanded = 32 * inv_factor
            assert layer.expanded_channels == expected_expanded

            # Verify output
            assert output.shape == (2, 16, 16, 64)
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_different_channel_configurations(self):
        """Test with various input and output channel combinations."""
        test_cases = [
            # (input_channels, filters, expected_residual)
            (32, 32, True),   # Equal channels - residual enabled
            (64, 64, True),   # Equal channels - residual enabled
            (32, 64, False),  # Different channels - no residual
            (128, 64, False), # Different channels - no residual
            (16, 256, False), # Very different channels - no residual
        ]

        for input_channels, filters, expected_residual in test_cases:
            layer = HANCBlock(filters=filters, input_channels=input_channels)
            test_input = keras.random.normal([2, 8, 8, input_channels])

            # Check residual logic
            assert layer.use_residual == expected_residual
            if expected_residual:
                assert layer.residual_bn is not None
            else:
                assert layer.residual_bn is None

            # Test forward pass
            output = layer(test_input)
            assert output.shape == (2, 8, 8, filters)
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_training_vs_inference_mode(self, sample_input, layer_config):
        """Test behavior in training vs inference modes."""
        layer = HANCBlock(**layer_config)

        # Test training mode
        output_train = layer(sample_input, training=True)
        expected_shape = (sample_input.shape[0], sample_input.shape[1],
                         sample_input.shape[2], layer_config['filters'])
        assert output_train.shape == expected_shape
        assert not keras.ops.any(keras.ops.isnan(output_train))

        # Test inference mode
        output_inference = layer(sample_input, training=False)
        assert output_inference.shape == expected_shape
        assert not keras.ops.any(keras.ops.isnan(output_inference))

        # Due to BatchNorm, outputs may differ between training and inference
        # but both should be valid

    def test_compute_output_shape(self):
        """Test output shape computation."""
        test_cases = [
            # (filters, input_channels, input_shape, expected_output_shape)
            (64, 32, (None, 32, 32, 32), (None, 32, 32, 64)),
            (128, 64, (4, 16, 16, 64), (4, 16, 16, 128)),
            (256, 128, (2, 64, 64, 128), (2, 64, 64, 256)),
            (32, 64, (8, 8, 8, 64), (8, 8, 8, 32)),
        ]

        for filters, input_channels, input_shape, expected_shape in test_cases:
            layer = HANCBlock(filters=filters, input_channels=input_channels)
            computed_shape = layer.compute_output_shape(input_shape)
            assert computed_shape == expected_shape

    def test_compute_output_shape_validation(self):
        """Test output shape computation with invalid inputs."""
        layer = HANCBlock(filters=64, input_channels=32)

        # Test with wrong number of dimensions
        with pytest.raises(ValueError, match="Expected 4D input shape"):
            layer.compute_output_shape((None, 32, 32))  # 3D instead of 4D

    def test_get_config_completeness(self, custom_layer_config):
        """Test that get_config contains all initialization parameters."""
        layer = HANCBlock(**custom_layer_config)
        config = layer.get_config()

        # Verify all custom parameters are in config
        expected_keys = {
            'filters', 'input_channels', 'k', 'inv_factor',
            'kernel_initializer', 'kernel_regularizer'
        }

        config_keys = set(config.keys())
        assert expected_keys.issubset(config_keys)

        # Verify specific values
        assert config['filters'] == 128
        assert config['input_channels'] == 64
        assert config['k'] == 4
        assert config['inv_factor'] == 4

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle following modern patterns.

        NOTE: Currently skipped because HANCLayer (imported dependency) does not
        follow modern Keras 3 serialization patterns. The HANCLayer's sub-layers
        (hanc_conv, hanc_bn) are not properly built during the parent's build() method,
        causing serialization failures.

        This test should be re-enabled once HANCLayer is updated to follow
        the patterns described in "Complete Guide to Modern Keras 3 Custom Layers".
        """
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = HANCBlock(**layer_config)(inputs)
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

    def test_model_saving_config_only(self, layer_config, sample_input):
        """Test that HANCBlock's own configuration serializes correctly."""
        # Test that our layer's configuration works
        layer = HANCBlock(**layer_config)

        # Forward pass to build
        output = layer(sample_input)

        # Test config completeness
        config = layer.get_config()

        # Verify all parameters are present
        expected_keys = {'filters', 'input_channels', 'k', 'inv_factor',
                        'kernel_initializer', 'kernel_regularizer'}
        assert expected_keys.issubset(set(config.keys()))

        # Verify we can recreate the layer from config
        new_layer = HANCBlock(**{k: v for k, v in config.items()
                                if k in expected_keys})

        # New layer should have same configuration
        assert new_layer.filters == layer.filters
        assert new_layer.input_channels == layer.input_channels
        assert new_layer.k == layer.k
        assert new_layer.inv_factor == layer.inv_factor

    def test_acc_unet_integration(self, sample_input):
        """Test layer integration in an ACC-UNet style architecture."""
        inputs = keras.Input(shape=sample_input.shape[1:])

        # Simulate encoder path with HANC blocks
        x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)

        # First HANC block in encoder
        x = HANCBlock(filters=128, input_channels=64, k=3)(x)

        # Second HANC block in encoder
        encoder_features = HANCBlock(filters=128, input_channels=128, k=2)(x)

        # Simulate decoder path
        decoder_input = keras.layers.Conv2DTranspose(128, 3, padding='same', activation='relu')(encoder_features)

        # HANC block in decoder path
        decoder_features = HANCBlock(filters=64, input_channels=128, k=2)(decoder_input)

        # Skip connection
        merged = keras.layers.Add()([
            keras.layers.Conv2D(64, 1)(encoder_features),  # Channel adjustment for skip
            decoder_features
        ])

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
        layer = HANCBlock(
            filters=64,
            input_channels=64,
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
        layer = HANCBlock(**layer_config)

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
        layer = HANCBlock(filters=64, input_channels=32)

        spatial_sizes = [
            (8, 8),     # Small
            (32, 32),   # Medium
            (64, 64),   # Large
            (16, 24),   # Non-square
            (128, 32),  # Very rectangular
        ]

        for height, width in spatial_sizes:
            test_input = keras.random.normal([2, height, width, 32])
            output = layer(test_input)

            # Output should have same spatial dimensions but different channels
            assert output.shape == (2, height, width, 64)
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = HANCBlock(filters=64, input_channels=32)

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

    def test_hierarchical_context_aggregation(self, sample_input):
        """Test that hierarchical context aggregation is working correctly."""
        # Test with different k values to ensure HANC mechanism works
        k_values = [1, 3, 5]

        for k in k_values:
            layer = HANCBlock(filters=64, input_channels=64, k=k)
            output = layer(sample_input)

            # HANC layer should be properly configured
            assert layer.hanc_layer.k == k

            # Output should be valid regardless of k value
            assert output.shape == (sample_input.shape[0], sample_input.shape[1],
                                   sample_input.shape[2], 64)
            assert not keras.ops.any(keras.ops.isnan(output))

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes_parametrized(self, sample_input, layer_config, training):
        """Test layer behavior in different training modes using parametrize."""
        layer = HANCBlock(**layer_config)
        output = layer(sample_input, training=training)

        # Basic output validation
        expected_shape = (sample_input.shape[0], sample_input.shape[1],
                         sample_input.shape[2], layer_config['filters'])
        assert output.shape == expected_shape
        assert not keras.ops.any(keras.ops.isnan(output))


class TestHANCBlockEdgeCases:
    """Test edge cases and boundary conditions for HANCBlock."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample 4D input tensor for testing (batch, height, width, channels)."""
        return keras.random.normal([4, 32, 32, 64])  # batch, H, W, channels

    def test_minimal_configuration(self):
        """Test with minimal viable configuration."""
        layer = HANCBlock(filters=1, input_channels=1, k=1, inv_factor=1)
        test_input = keras.random.normal([1, 4, 4, 1])

        output = layer(test_input)
        assert output.shape == (1, 4, 4, 1)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_large_configuration(self):
        """Test with large configuration."""
        layer = HANCBlock(filters=512, input_channels=256, k=5, inv_factor=8)
        test_input = keras.random.normal([2, 8, 8, 256])

        output = layer(test_input)
        assert output.shape == (2, 8, 8, 512)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_single_pixel_input(self):
        """Test with 1x1 spatial dimensions."""
        layer = HANCBlock(filters=64, input_channels=32, k=2)
        test_input = keras.random.normal([2, 1, 1, 32])

        output = layer(test_input)
        assert output.shape == (2, 1, 1, 64)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_layer_reuse(self):
        """Test that the same layer instance can be reused."""
        layer = HANCBlock(filters=64, input_channels=32, k=2)

        # Use the same layer with different inputs (same channel count)
        input1 = keras.random.normal([2, 16, 16, 32])
        input2 = keras.random.normal([3, 24, 24, 32])

        output1 = layer(input1)
        output2 = layer(input2)

        assert output1.shape == (2, 16, 16, 64)
        assert output2.shape == (3, 24, 24, 64)

        # Both outputs should be valid
        assert not keras.ops.any(keras.ops.isnan(output1))
        assert not keras.ops.any(keras.ops.isnan(output2))

    def test_different_initializers(self):
        """Test layer with various weight initializers."""
        initializers = ['glorot_uniform', 'he_normal', 'lecun_normal']

        for initializer in initializers:
            layer = HANCBlock(
                filters=64,
                input_channels=32,
                kernel_initializer=initializer
            )

            test_input = keras.random.normal([2, 8, 8, 32])
            output = layer(test_input)

            assert output.shape == (2, 8, 8, 64)
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_very_high_expansion_factor(self):
        """Test with very high inverted bottleneck expansion factor."""
        layer = HANCBlock(filters=64, input_channels=16, k=2, inv_factor=16)
        test_input = keras.random.normal([1, 8, 8, 16])

        # Should create very wide intermediate representation
        assert layer.expanded_channels == 16 * 16  # 256 channels

        output = layer(test_input)
        assert output.shape == (1, 8, 8, 64)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_batch_size_variations(self):
        """Test with different batch sizes."""
        layer = HANCBlock(filters=64, input_channels=32, k=2)

        batch_sizes = [1, 2, 8, 16, 32]
        for batch_size in batch_sizes:
            test_input = keras.random.normal([batch_size, 16, 16, 32])
            output = layer(test_input)

            assert output.shape == (batch_size, 16, 16, 64)
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_memory_efficiency(self):
        """Test that layer doesn't create excessive intermediate tensors."""
        # This is more of a smoke test to ensure the layer doesn't crash with larger inputs
        layer = HANCBlock(filters=128, input_channels=128, k=4, inv_factor=4)
        test_input = keras.random.normal([1, 64, 64, 128])  # Relatively large input

        output = layer(test_input)
        assert output.shape == (1, 64, 64, 128)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_squeeze_excitation_integration(self, sample_input):
        """Test that Squeeze-and-Excitation is working correctly."""
        layer = HANCBlock(filters=64, input_channels=64, k=2)

        # Forward pass
        output = layer(sample_input)

        # SE blocks should be built and functional
        assert layer.squeeze_excitation.built
        assert output.shape == (sample_input.shape[0], sample_input.shape[1],
                               sample_input.shape[2], 64)

    def test_depthwise_convolution_functionality(self):
        """Test that depthwise convolution is working as expected."""
        layer = HANCBlock(filters=64, input_channels=32, k=2, inv_factor=3)
        test_input = keras.random.normal([2, 16, 16, 32])

        # Forward pass to build layer
        output = layer(test_input)

        # Check that depthwise conv was created with correct properties
        assert layer.depthwise_conv.kernel_size == (3, 3)
        assert layer.depthwise_conv.padding == 'same'
        assert layer.depthwise_conv.use_bias == False

        assert output.shape == (2, 16, 16, 64)

    def test_expansion_and_projection_pipeline(self):
        """Test the expansion -> processing -> projection pipeline."""
        input_channels = 64
        filters = 128
        inv_factor = 4

        layer = HANCBlock(
            filters=filters,
            input_channels=input_channels,
            k=3,
            inv_factor=inv_factor
        )

        test_input = keras.random.normal([2, 16, 16, input_channels])
        output = layer(test_input)

        # Check expansion computation
        expected_expanded = input_channels * inv_factor
        assert layer.expanded_channels == expected_expanded

        # Check final output
        assert output.shape == (2, 16, 16, filters)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_different_regularizers(self):
        """Test with different regularization strategies."""
        regularizers = [
            None,
            keras.regularizers.L1(0.01),
            keras.regularizers.L2(0.01),
            keras.regularizers.L1L2(l1=0.01, l2=0.01)
        ]

        for regularizer in regularizers:
            layer = HANCBlock(
                filters=64,
                input_channels=32,
                kernel_regularizer=regularizer
            )

            test_input = keras.random.normal([2, 8, 8, 32])
            output = layer(test_input)

            assert output.shape == (2, 8, 8, 64)
            assert not keras.ops.any(keras.ops.isnan(output))

            # Check regularization losses
            if regularizer is not None:
                assert len(layer.losses) > 0
            else:
                # May still have losses from sub-layers, but fewer
                pass

    def test_extreme_aspect_ratios(self):
        """Test with extreme spatial aspect ratios."""
        layer = HANCBlock(filters=64, input_channels=32, k=2)

        extreme_shapes = [
            (2, 1, 256, 32),    # Very wide
            (2, 256, 1, 32),    # Very tall
            (2, 4, 128, 32),    # Wide rectangle
            (2, 128, 4, 32),    # Tall rectangle
        ]

        for shape in extreme_shapes:
            test_input = keras.random.normal(shape)
            output = layer(test_input)

            expected_shape = (shape[0], shape[1], shape[2], 64)
            assert output.shape == expected_shape
            assert not keras.ops.any(keras.ops.isnan(output))