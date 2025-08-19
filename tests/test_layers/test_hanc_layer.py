"""
Comprehensive test suite for HANCLayer implementation.

This test suite follows modern Keras 3 testing patterns and validates the
HANCLayer's functionality, serialization, and integration capabilities.
The tests cover the core hierarchical context aggregation mechanism that
enables transformer-like global modeling through purely convolutional operations.

Place this file in: tests/test_layers/test_hanc_layer.py
"""

import pytest
import numpy as np
import keras
import tempfile
import os
from typing import Dict, Any
import tensorflow as tf

from dl_techniques.layers.hanc_layer import HANCLayer


class TestHANCLayer:
    """Comprehensive test suite for HANCLayer implementation."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample 4D input tensor for testing (batch, height, width, channels)."""
        return keras.random.normal([4, 32, 32, 64])  # batch, H, W, channels

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard layer configuration for testing."""
        return {
            'in_channels': 64,
            'out_channels': 64,
            'k': 3
        }

    @pytest.fixture
    def custom_layer_config(self) -> Dict[str, Any]:
        """Custom layer configuration with comprehensive parameters."""
        return {
            'in_channels': 128,
            'out_channels': 256,
            'k': 4,
            'kernel_initializer': 'he_normal',
            'kernel_regularizer': keras.regularizers.L2(1e-4)
        }

    @pytest.fixture
    def minimal_config(self) -> Dict[str, Any]:
        """Minimal configuration with k=1 (no hierarchical pooling)."""
        return {
            'in_channels': 32,
            'out_channels': 32,
            'k': 1
        }

    @pytest.fixture
    def maximal_config(self) -> Dict[str, Any]:
        """Maximal configuration with k=5 (maximum hierarchical levels)."""
        return {
            'in_channels': 64,
            'out_channels': 128,
            'k': 5
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = HANCLayer(in_channels=64, out_channels=32)

        # Check stored configuration
        assert layer.in_channels == 64
        assert layer.out_channels == 32
        assert layer.k == 3  # Default value
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert layer.kernel_regularizer is None

        # Check computed derived parameters
        assert layer.total_channels == 64 * (2 * 3 - 1)  # 64 * 5 = 320

        # Check that layer is not built yet
        assert not layer.built

        # Check that all main sub-layers are created but not built
        assert layer.conv is not None
        assert layer.batch_norm is not None
        assert layer.activation is not None
        assert layer.concatenate is not None

        # Check that pooling layers are pre-created (for all possible scales)
        assert len(layer.avg_pooling_layers) == 4  # scales 1,2,3,4 (for max k=5)
        assert len(layer.max_pooling_layers) == 4
        assert len(layer.avg_upsampling_layers) == 4
        assert len(layer.max_upsampling_layers) == 4

        # Sub-layers should not be built yet
        assert not layer.conv.built
        assert not layer.batch_norm.built

    def test_initialization_custom(self, custom_layer_config):
        """Test initialization with comprehensive custom parameters."""
        layer = HANCLayer(**custom_layer_config)

        # Verify all custom parameters are stored correctly
        assert layer.in_channels == 128
        assert layer.out_channels == 256
        assert layer.k == 4
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.kernel_regularizer, keras.regularizers.L2)

        # Check computed parameters for k=4
        assert layer.total_channels == 128 * (2 * 4 - 1)  # 128 * 7 = 896

    def test_channel_expansion_calculation(self):
        """Test that channel expansion is calculated correctly for different k values."""
        test_cases = [
            # (in_channels, k, expected_total_channels)
            (64, 1, 64 * 1),     # k=1: 64 * (2*1-1) = 64 * 1 = 64
            (64, 2, 64 * 3),     # k=2: 64 * (2*2-1) = 64 * 3 = 192
            (64, 3, 64 * 5),     # k=3: 64 * (2*3-1) = 64 * 5 = 320
            (64, 4, 64 * 7),     # k=4: 64 * (2*4-1) = 64 * 7 = 448
            (64, 5, 64 * 9),     # k=5: 64 * (2*5-1) = 64 * 9 = 576
            (32, 3, 32 * 5),     # Different in_channels: 32 * 5 = 160
            (128, 2, 128 * 3),   # Different in_channels: 128 * 3 = 384
        ]

        for in_channels, k, expected_total in test_cases:
            layer = HANCLayer(in_channels=in_channels, out_channels=64, k=k)
            assert layer.total_channels == expected_total

    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid in_channels values
        with pytest.raises(ValueError, match="in_channels must be positive"):
            HANCLayer(in_channels=0, out_channels=64)

        with pytest.raises(ValueError, match="in_channels must be positive"):
            HANCLayer(in_channels=-10, out_channels=64)

        # Test invalid out_channels values
        with pytest.raises(ValueError, match="out_channels must be positive"):
            HANCLayer(in_channels=64, out_channels=0)

        with pytest.raises(ValueError, match="out_channels must be positive"):
            HANCLayer(in_channels=64, out_channels=-5)

        # Test invalid k values
        with pytest.raises(ValueError, match="k must be between 1 and 5"):
            HANCLayer(in_channels=64, out_channels=64, k=0)

        with pytest.raises(ValueError, match="k must be between 1 and 5"):
            HANCLayer(in_channels=64, out_channels=64, k=6)

    def test_build_process(self, sample_input, layer_config):
        """Test that the layer builds properly following modern patterns."""
        layer = HANCLayer(**layer_config)

        # Layer should not be built initially
        assert not layer.built

        # Forward pass triggers build
        output = layer(sample_input)

        # Check that layer is now built
        assert layer.built

        # Check that main sub-layers are built
        assert layer.conv.built
        assert layer.batch_norm.built
        # activation and concatenate layers don't need explicit building

        # Check that appropriate pooling layers are built (based on k=3)
        # For k=3, we use scales 0,1 (representing 2^1=2, 2^2=4)
        for scale in range(min(layer_config['k'] - 1, len(layer.avg_pooling_layers))):
            assert layer.avg_pooling_layers[scale].built
            assert layer.max_pooling_layers[scale].built
            assert layer.avg_upsampling_layers[scale].built
            assert layer.max_upsampling_layers[scale].built

        # Verify output shape
        expected_shape = (sample_input.shape[0], sample_input.shape[1],
                         sample_input.shape[2], layer_config['out_channels'])
        assert output.shape == expected_shape

        # Check that weights were created
        assert len(layer.weights) > 0
        assert len(layer.trainable_variables) > 0

    def test_build_input_shape_validation(self):
        """Test input shape validation during build."""
        layer = HANCLayer(in_channels=32, out_channels=64)

        # Test with wrong number of dimensions
        with pytest.raises(ValueError, match="Expected 4D input shape"):
            layer.build((None, 32))  # 2D instead of 4D

        # Test with channel mismatch
        with pytest.raises(ValueError, match="Input channels mismatch"):
            layer.build((None, 32, 32, 64))  # 64 channels instead of expected 32

    def test_forward_pass_basic(self, sample_input, layer_config):
        """Test basic forward pass functionality."""
        layer = HANCLayer(**layer_config)
        output = layer(sample_input)

        # Basic sanity checks
        expected_shape = (sample_input.shape[0], sample_input.shape[1],
                         sample_input.shape[2], layer_config['out_channels'])
        assert output.shape == expected_shape
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

    def test_different_k_values(self, sample_input):
        """Test layer with various hierarchical levels (k values)."""
        k_values = [1, 2, 3, 4, 5]
        in_channels = sample_input.shape[-1]  # 64

        for k in k_values:
            layer = HANCLayer(in_channels=in_channels, out_channels=128, k=k)
            output = layer(sample_input)

            # Verify output is valid
            assert output.shape == (sample_input.shape[0], sample_input.shape[1],
                                   sample_input.shape[2], 128)
            assert not keras.ops.any(keras.ops.isnan(output))

            # Check that k is properly stored
            assert layer.k == k

            # Check that total_channels is computed correctly
            expected_total = in_channels * (2 * k - 1)
            assert layer.total_channels == expected_total

    def test_k_equals_1_no_pooling(self):
        """Test special case where k=1 means no hierarchical pooling."""
        layer = HANCLayer(in_channels=64, out_channels=64, k=1)
        test_input = keras.random.normal([2, 16, 16, 64])

        output = layer(test_input)

        # With k=1, total_channels should equal in_channels (no expansion)
        assert layer.total_channels == 64 * 1  # 64
        assert output.shape == (2, 16, 16, 64)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_different_channel_configurations(self):
        """Test with various input and output channel combinations."""
        test_cases = [
            # (in_channels, out_channels, k, expected_total_channels)
            (32, 32, 2, 32 * 3),     # Same channels, k=2
            (64, 128, 3, 64 * 5),    # Different channels, k=3
            (128, 64, 4, 128 * 7),   # Reduce channels, k=4
            (16, 256, 5, 16 * 9),    # Large increase, k=5
        ]

        for in_channels, out_channels, k, expected_total in test_cases:
            layer = HANCLayer(in_channels=in_channels, out_channels=out_channels, k=k)
            test_input = keras.random.normal([2, 8, 8, in_channels])

            # Check channel expansion calculation
            assert layer.total_channels == expected_total

            # Test forward pass
            output = layer(test_input)
            assert output.shape == (2, 8, 8, out_channels)
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_multi_scale_pooling_mechanism(self):
        """Test that multi-scale pooling is working correctly."""
        layer = HANCLayer(in_channels=32, out_channels=32, k=3)
        test_input = keras.random.normal([1, 16, 16, 32])

        # Forward pass to build and run
        output = layer(test_input)

        # Check that appropriate pooling layers were used
        # For k=3, we should have 2 scales: 2x2 and 4x4 pooling
        expected_scales = layer.k - 1  # 2 scales for k=3

        # Verify pooling layers exist and have correct pool sizes
        for scale in range(expected_scales):
            pool_size = 2 ** (scale + 1)  # 2, 4 for scales 0, 1

            assert layer.avg_pooling_layers[scale].pool_size == (pool_size, pool_size)
            assert layer.max_pooling_layers[scale].pool_size == (pool_size, pool_size)
            assert layer.avg_upsampling_layers[scale].size == (pool_size, pool_size)
            assert layer.max_upsampling_layers[scale].size == (pool_size, pool_size)

        assert output.shape == (1, 16, 16, 32)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_training_vs_inference_mode(self, sample_input, layer_config):
        """Test behavior in training vs inference modes."""
        layer = HANCLayer(**layer_config)

        # Test training mode
        output_train = layer(sample_input, training=True)
        expected_shape = (sample_input.shape[0], sample_input.shape[1],
                         sample_input.shape[2], layer_config['out_channels'])
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
            # (in_channels, out_channels, k, input_shape, expected_output_shape)
            (64, 32, 3, (None, 32, 32, 64), (None, 32, 32, 32)),
            (128, 256, 2, (4, 16, 16, 128), (4, 16, 16, 256)),
            (32, 64, 4, (2, 64, 64, 32), (2, 64, 64, 64)),
            (256, 128, 1, (8, 8, 8, 256), (8, 8, 8, 128)),
        ]

        for in_channels, out_channels, k, input_shape, expected_shape in test_cases:
            layer = HANCLayer(in_channels=in_channels, out_channels=out_channels, k=k)
            computed_shape = layer.compute_output_shape(input_shape)
            assert computed_shape == expected_shape

    def test_compute_output_shape_validation(self):
        """Test output shape computation with invalid inputs."""
        layer = HANCLayer(in_channels=64, out_channels=32)

        # Test with wrong number of dimensions
        with pytest.raises(ValueError, match="Expected 4D input shape"):
            layer.compute_output_shape((None, 32, 32))  # 3D instead of 4D

    def test_get_config_completeness(self, custom_layer_config):
        """Test that get_config contains all initialization parameters."""
        layer = HANCLayer(**custom_layer_config)
        config = layer.get_config()

        # Verify all custom parameters are in config
        expected_keys = {
            'in_channels', 'out_channels', 'k',
            'kernel_initializer', 'kernel_regularizer'
        }

        config_keys = set(config.keys())
        assert expected_keys.issubset(config_keys)

        # Verify specific values
        assert config['in_channels'] == 128
        assert config['out_channels'] == 256
        assert config['k'] == 4

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle following modern patterns."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = HANCLayer(**layer_config)(inputs)
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

    def test_hanc_block_integration(self, sample_input):
        """Test layer integration within HANCBlock architecture."""
        inputs = keras.Input(shape=sample_input.shape[1:])

        # Simulate usage within HANC block pipeline
        # Expansion phase
        x = keras.layers.Conv2D(128, 1, padding='same')(inputs)  # Expand channels
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(negative_slope=0.01)(x)

        # Depthwise convolution
        x = keras.layers.DepthwiseConv2D(3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(negative_slope=0.01)(x)

        # HANC Layer for hierarchical context aggregation
        x = HANCLayer(in_channels=128, out_channels=64, k=3)(x)

        # Output projection
        outputs = keras.layers.Conv2D(32, 1, padding='same')(x)

        model = keras.Model(inputs, outputs)

        # Test forward pass
        prediction = model(sample_input)
        assert prediction.shape == (sample_input.shape[0], sample_input.shape[1],
                                    sample_input.shape[2], 32)

        # Test gradient flow
        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = model(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        gradients = tape.gradient(loss, model.trainable_variables)
        assert all(g is not None for g in gradients)

    def test_regularization_losses(self, sample_input):
        """Test that regularization losses are properly computed."""
        layer = HANCLayer(
            in_channels=64,
            out_channels=64,
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
        layer = HANCLayer(**layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        # Compute gradients
        gradients = tape.gradient(loss, layer.trainable_variables)

        # Check that all gradients exist
        assert len(gradients) == len(layer.trainable_variables)
        assert all(g is not None for g in gradients)

        # Check gradient shapes match variable shapes
        for grad, var in zip(gradients, layer.trainable_variables):
            assert grad.shape == var.shape

    def test_different_spatial_dimensions(self):
        """Test layer with different spatial input dimensions."""
        layer = HANCLayer(in_channels=32, out_channels=64, k=3)

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
        layer = HANCLayer(in_channels=32, out_channels=64, k=3)

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

    def test_hierarchical_context_scales(self):
        """Test that different k values produce different context scales."""
        test_input = keras.random.normal([1, 32, 32, 64])

        # Test different k values and verify they use different pooling scales
        for k in [2, 3, 4, 5]:
            layer = HANCLayer(in_channels=64, out_channels=64, k=k)
            output = layer(test_input)

            # All outputs should be valid
            assert output.shape == (1, 32, 32, 64)
            assert not keras.ops.any(keras.ops.isnan(output))

            # Check expected channel expansion during concatenation
            expected_total = 64 * (2 * k - 1)
            assert layer.total_channels == expected_total

            # Verify correct number of pooling scales used
            expected_scales = k - 1
            # We should build exactly expected_scales number of pooling layers
            built_avg_pools = sum(1 for i in range(expected_scales)
                                if i < len(layer.avg_pooling_layers)
                                and layer.avg_pooling_layers[i].built)
            assert built_avg_pools == expected_scales

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes_parametrized(self, sample_input, layer_config, training):
        """Test layer behavior in different training modes using parametrize."""
        layer = HANCLayer(**layer_config)
        output = layer(sample_input, training=training)

        # Basic output validation
        expected_shape = (sample_input.shape[0], sample_input.shape[1],
                         sample_input.shape[2], layer_config['out_channels'])
        assert output.shape == expected_shape
        assert not keras.ops.any(keras.ops.isnan(output))


class TestHANCLayerEdgeCases:
    """Test edge cases and boundary conditions for HANCLayer."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample 4D input tensor for testing (batch, height, width, channels)."""
        return keras.random.normal([4, 32, 32, 64])  # batch, H, W, channels

    def test_minimal_configuration(self):
        """Test with minimal viable configuration."""
        layer = HANCLayer(in_channels=1, out_channels=1, k=1)
        test_input = keras.random.normal([1, 4, 4, 1])

        output = layer(test_input)
        assert output.shape == (1, 4, 4, 1)
        assert not keras.ops.any(keras.ops.isnan(output))

        # With k=1, no channel expansion should occur
        assert layer.total_channels == 1

    def test_large_configuration(self):
        """Test with large configuration."""
        layer = HANCLayer(in_channels=512, out_channels=256, k=5)
        test_input = keras.random.normal([2, 8, 8, 512])

        output = layer(test_input)
        assert output.shape == (2, 8, 8, 256)
        assert not keras.ops.any(keras.ops.isnan(output))

        # Check maximum channel expansion for k=5
        assert layer.total_channels == 512 * 9  # 512 * (2*5-1)

    def test_single_pixel_input(self):
        """Test with 1x1 spatial dimensions."""
        layer = HANCLayer(in_channels=64, out_channels=32, k=3)
        test_input = keras.random.normal([2, 1, 1, 64])

        output = layer(test_input)
        assert output.shape == (2, 1, 1, 32)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_layer_reuse(self):
        """Test that the same layer instance can be reused."""
        layer = HANCLayer(in_channels=32, out_channels=64, k=2)

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
            layer = HANCLayer(
                in_channels=32,
                out_channels=64,
                kernel_initializer=initializer
            )

            test_input = keras.random.normal([2, 8, 8, 32])
            output = layer(test_input)

            assert output.shape == (2, 8, 8, 64)
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_extreme_channel_expansion(self):
        """Test with configuration that causes large channel expansion."""
        # Large channel expansion case: 256 channels with k=5
        layer = HANCLayer(in_channels=256, out_channels=128, k=5)
        test_input = keras.random.normal([1, 8, 8, 256])

        # Should create very wide intermediate representation
        assert layer.total_channels == 256 * 9  # 2304 channels during concatenation

        output = layer(test_input)
        assert output.shape == (1, 8, 8, 128)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_batch_size_variations(self):
        """Test with different batch sizes."""
        layer = HANCLayer(in_channels=64, out_channels=32, k=3)

        batch_sizes = [1, 2, 8, 16, 32]
        for batch_size in batch_sizes:
            test_input = keras.random.normal([batch_size, 16, 16, 64])
            output = layer(test_input)

            assert output.shape == (batch_size, 16, 16, 32)
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_memory_efficiency(self):
        """Test that layer doesn't create excessive intermediate tensors."""
        # Test with relatively large input to check memory handling
        layer = HANCLayer(in_channels=128, out_channels=64, k=4)
        test_input = keras.random.normal([1, 64, 64, 128])  # Large spatial dimensions

        output = layer(test_input)
        assert output.shape == (1, 64, 64, 64)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_pooling_scale_boundaries(self):
        """Test pooling at different scales to verify correct behavior."""
        layer = HANCLayer(in_channels=32, out_channels=32, k=4)

        # Test with input size that will exercise different pooling scales
        test_input = keras.random.normal([1, 32, 32, 32])
        output = layer(test_input)

        # For k=4, we should have 3 pooling scales: 2x2, 4x4, 8x8
        expected_scales = 3

        # Check that pooling layers have correct sizes
        for scale in range(expected_scales):
            expected_pool_size = 2 ** (scale + 1)  # 2, 4, 8

            assert layer.avg_pooling_layers[scale].pool_size == (expected_pool_size, expected_pool_size)
            assert layer.max_pooling_layers[scale].pool_size == (expected_pool_size, expected_pool_size)

        assert output.shape == (1, 32, 32, 32)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_upsampling_spatial_alignment(self):
        """Test that upsampling maintains proper spatial alignment."""
        layer = HANCLayer(in_channels=32, out_channels=32, k=3)

        # Test with various input sizes to verify upsampling alignment
        test_sizes = [(16, 16), (17, 17), (20, 24), (33, 15)]  # Including odd sizes

        for height, width in test_sizes:
            test_input = keras.random.normal([1, height, width, 32])
            output = layer(test_input)

            # Output should maintain exact spatial dimensions
            assert output.shape == (1, height, width, 32)
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
            layer = HANCLayer(
                in_channels=64,
                out_channels=32,
                kernel_regularizer=regularizer
            )

            test_input = keras.random.normal([2, 8, 8, 64])
            output = layer(test_input)

            assert output.shape == (2, 8, 8, 32)
            assert not keras.ops.any(keras.ops.isnan(output))

            # Check regularization losses
            if regularizer is not None:
                assert len(layer.losses) > 0
            else:
                # May still have losses from batch norm, but should be minimal
                pass

    def test_extreme_aspect_ratios(self):
        """Test with extreme spatial aspect ratios."""
        layer = HANCLayer(in_channels=32, out_channels=64, k=3)

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

    def test_concatenation_mechanism(self):
        """Test that concatenation produces expected channel expansion."""
        test_cases = [
            # (k, expected_features_count)
            (1, 1),  # Original only
            (2, 3),  # Original + 1 avg + 1 max
            (3, 5),  # Original + 2 avg + 2 max
            (4, 7),  # Original + 3 avg + 3 max
            (5, 9),  # Original + 4 avg + 4 max
        ]

        for k, expected_features in test_cases:
            layer = HANCLayer(in_channels=16, out_channels=16, k=k)
            test_input = keras.random.normal([1, 8, 8, 16])

            # Forward pass
            output = layer(test_input)

            # Check channel expansion
            expected_channels = 16 * expected_features
            assert layer.total_channels == expected_channels

            assert output.shape == (1, 8, 8, 16)
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_batch_normalization_integration(self):
        """Test that batch normalization is working correctly."""
        layer = HANCLayer(in_channels=64, out_channels=64, k=3)
        test_input = keras.random.normal([4, 16, 16, 64])  # Larger batch for BN

        # Test training mode (BN should use batch statistics)
        output_train = layer(test_input, training=True)
        assert not keras.ops.any(keras.ops.isnan(output_train))

        # Test inference mode (BN should use moving statistics)
        output_infer = layer(test_input, training=False)
        assert not keras.ops.any(keras.ops.isnan(output_infer))

        # Both should be valid but may differ due to BN behavior
        assert output_train.shape == output_infer.shape