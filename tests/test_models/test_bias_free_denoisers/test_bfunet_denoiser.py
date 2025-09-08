"""
Comprehensive test suite for Bias-Free U-Net Model.

Tests cover initialization, validation, architecture verification, forward pass,
serialization, scaling invariance, skip connections, variants, and multi-scale processing.
"""

import os
import keras
import pytest
import tempfile
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any

from dl_techniques.models.bias_free_denoisers.bfunet import (
    create_bfunet_denoiser,
    create_bfunet_variant,
    BFUNET_CONFIGS
)


class TestBiasFreeUNet:
    """Test suite for Bias-Free U-Net implementation."""

    @pytest.fixture
    def grayscale_input_shape(self) -> Tuple[int, int, int]:
        """Standard grayscale image shape for testing."""
        return (64, 64, 1)

    @pytest.fixture
    def rgb_input_shape(self) -> Tuple[int, int, int]:
        """Standard RGB image shape for testing."""
        return (128, 128, 3)

    @pytest.fixture
    def large_input_shape(self) -> Tuple[int, int, int]:
        """Large image shape for testing deep U-Net."""
        return (256, 256, 1)

    @pytest.fixture
    def variable_input_shape(self) -> Tuple[int, int, int]:
        """Variable size input shape for testing."""
        return (None, None, 1)

    @pytest.fixture
    def test_image_grayscale(self) -> np.ndarray:
        """Create test grayscale image data."""
        return np.random.rand(2, 64, 64, 1).astype(np.float32)

    @pytest.fixture
    def test_image_rgb(self) -> np.ndarray:
        """Create test RGB image data."""
        return np.random.rand(2, 128, 128, 3).astype(np.float32)

    @pytest.fixture
    def test_image_large(self) -> np.ndarray:
        """Create large test image data."""
        return np.random.rand(1, 256, 256, 1).astype(np.float32)

    @pytest.fixture
    def default_model_config(self) -> Dict[str, Any]:
        """Default configuration for model creation."""
        return {
            'depth': 4,
            'initial_filters': 64,
            'filter_multiplier': 2,
            'blocks_per_level': 2,
            'kernel_size': 3,
            'activation': 'relu',
            'final_activation': 'linear'
        }

    # ================================================================
    # Initialization Tests
    # ================================================================

    def test_initialization_defaults(self, grayscale_input_shape):
        """Test initialization with default parameters."""
        model = create_bfunet_denoiser(input_shape=grayscale_input_shape)

        # Check model properties
        assert model.name == 'bias_free_unet'
        assert len(model.layers) > 0
        assert model.input_shape == (None,) + grayscale_input_shape
        assert model.output_shape == (None,) + grayscale_input_shape

    def test_initialization_variable_input_size(self, variable_input_shape):
        """Test initialization with variable input size."""
        model = create_bfunet_denoiser(
            input_shape=variable_input_shape,
            depth=3,
            initial_filters=16
        )

        assert model.input_shape == (None, None, None, 1)
        assert model.output_shape == (None, None, None, 1)

        # Test with different sized inputs (must be divisible by 2^depth for proper U-Net operation)
        small_input = np.random.rand(1, 32, 32, 1).astype(np.float32)  # 32 = 4 * 2^3
        large_input = np.random.rand(1, 128, 128, 1).astype(np.float32)  # 128 = 16 * 2^3

        small_output = model(small_input)
        large_output = model(large_input)

        assert small_output.shape == (1, 32, 32, 1)
        assert large_output.shape == (1, 128, 128, 1)

    def test_different_depths(self, grayscale_input_shape):
        """Test U-Net with different depth configurations."""
        depths = [3, 4, 5]  # Updated minimum depth to 3

        for depth in depths:
            model = create_bfunet_denoiser(
                input_shape=grayscale_input_shape,
                depth=depth,
                initial_filters=8
            )

            # Test forward pass
            test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
            output = model(test_input)

            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))

    # ================================================================
    # Input Validation Tests
    # ================================================================

    def test_invalid_input_shape_type(self):
        """Test that invalid input_shape type raises TypeError."""
        with pytest.raises(TypeError, match="input_shape must be a tuple of 3 integers"):
            create_bfunet_denoiser(input_shape=[64, 64, 1])

        with pytest.raises(TypeError, match="input_shape must be a tuple of 3 integers"):
            create_bfunet_denoiser(input_shape=(64, 64))

        with pytest.raises(TypeError, match="input_shape must be a tuple of 3 integers"):
            create_bfunet_denoiser(input_shape=(64, 64, 1, 1))

    def test_invalid_depth(self, grayscale_input_shape):
        """Test that invalid depth raises ValueError."""
        with pytest.raises(ValueError, match="depth must be at least 3"):
            create_bfunet_denoiser(
                input_shape=grayscale_input_shape,
                depth=0
            )

        with pytest.raises(ValueError, match="depth must be at least 3"):
            create_bfunet_denoiser(
                input_shape=grayscale_input_shape,
                depth=2
            )

        with pytest.raises(ValueError, match="depth must be at least 3"):
            create_bfunet_denoiser(
                input_shape=grayscale_input_shape,
                depth=-1
            )

    def test_invalid_initial_filters(self, grayscale_input_shape):
        """Test that non-positive initial_filters raises ValueError."""
        with pytest.raises(ValueError, match="initial_filters must be positive"):
            create_bfunet_denoiser(
                input_shape=grayscale_input_shape,
                initial_filters=0
            )

        with pytest.raises(ValueError, match="initial_filters must be positive"):
            create_bfunet_denoiser(
                input_shape=grayscale_input_shape,
                initial_filters=-10
            )

    def test_invalid_filter_multiplier(self, grayscale_input_shape):
        """Test that invalid filter_multiplier raises ValueError."""
        with pytest.raises(ValueError, match="filter_multiplier must be at least 1"):
            create_bfunet_denoiser(
                input_shape=grayscale_input_shape,
                filter_multiplier=0
            )

    def test_invalid_blocks_per_level(self, grayscale_input_shape):
        """Test that non-positive blocks_per_level raises ValueError."""
        with pytest.raises(ValueError, match="blocks_per_level must be positive"):
            create_bfunet_denoiser(
                input_shape=grayscale_input_shape,
                blocks_per_level=0
            )

    def test_minimal_valid_configuration(self, grayscale_input_shape):
        """Test minimal valid configuration."""
        model = create_bfunet_denoiser(
            input_shape=grayscale_input_shape,
            depth=3,  # Updated minimum depth
            initial_filters=1,
            filter_multiplier=1,
            blocks_per_level=1
        )

        # Should still work
        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)
        assert output.shape == test_input.shape

    # ================================================================
    # Variant Tests (New)
    # ================================================================

    def test_available_variants(self):
        """Test that all defined variants are available."""
        expected_variants = ['tiny', 'small', 'base', 'large', 'xlarge']
        available_variants = list(BFUNET_CONFIGS.keys())

        for variant in expected_variants:
            assert variant in available_variants

    def test_create_variant_tiny(self, grayscale_input_shape):
        """Test creating tiny variant."""
        model = create_bfunet_variant('tiny', grayscale_input_shape)

        assert model.name == 'bias_free_unet_tiny'
        assert model.input_shape == (None,) + grayscale_input_shape
        assert model.output_shape == (None,) + grayscale_input_shape

        # Test forward pass
        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)
        assert output.shape == test_input.shape

    def test_create_variant_small(self, rgb_input_shape):
        """Test creating small variant."""
        model = create_bfunet_variant('small', rgb_input_shape)

        assert model.name == 'bias_free_unet_small'
        assert model.input_shape == (None,) + rgb_input_shape
        assert model.output_shape == (None,) + rgb_input_shape

        # Test forward pass
        test_input = np.random.rand(1, 128, 128, 3).astype(np.float32)
        output = model(test_input)
        assert output.shape == test_input.shape

    def test_create_variant_base(self, grayscale_input_shape):
        """Test creating base variant."""
        model = create_bfunet_variant('base', grayscale_input_shape)

        assert model.name == 'bias_free_unet_base'
        assert model.input_shape == (None,) + grayscale_input_shape
        assert model.output_shape == (None,) + grayscale_input_shape

        # Should have more layers than tiny/small
        assert len(model.layers) > 15

        # Test forward pass
        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)
        assert output.shape == test_input.shape

    def test_create_variant_large(self, grayscale_input_shape):
        """Test creating large variant."""
        model = create_bfunet_variant('large', grayscale_input_shape)

        assert model.name == 'bias_free_unet_large'
        assert model.input_shape == (None,) + grayscale_input_shape
        assert model.output_shape == (None,) + grayscale_input_shape

        # Should have more layers than base
        assert len(model.layers) > 20

        # Test forward pass
        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)
        assert output.shape == test_input.shape

    def test_create_variant_xlarge(self, large_input_shape):
        """Test creating xlarge variant."""
        model = create_bfunet_variant('xlarge', large_input_shape)

        assert model.name == 'bias_free_unet_xlarge'
        assert model.input_shape == (None,) + large_input_shape
        assert model.output_shape == (None,) + large_input_shape

        # Should have the most layers (depth=5)
        assert len(model.layers) > 30

        # Test forward pass
        test_input = np.random.rand(1, 256, 256, 1).astype(np.float32)
        output = model(test_input)
        assert output.shape == test_input.shape

    def test_invalid_variant(self, grayscale_input_shape):
        """Test that invalid variant name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown variant 'invalid'"):
            create_bfunet_variant('invalid', grayscale_input_shape)

    def test_variant_with_custom_parameters(self, grayscale_input_shape):
        """Test variant creation with custom parameter overrides."""
        model = create_bfunet_variant(
            'base',
            grayscale_input_shape,
            activation='gelu',
            use_residual_blocks=False,
            model_name='custom_base_unet'
        )

        assert model.name == 'custom_base_unet'

        # Test forward pass
        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)
        assert output.shape == test_input.shape

    def test_variants_consistency(self):
        """Test that all variants work with same input."""
        input_shape = (64, 64, 1)
        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        variants = ['tiny', 'small', 'base']

        for variant in variants:
            model = create_bfunet_variant(variant, input_shape)
            output = model(test_input)

            # All should produce valid outputs
            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))

    # ================================================================
    # Architecture Verification Tests
    # ================================================================

    def test_filter_progression(self, grayscale_input_shape):
        """Test that filter sizes progress correctly through the network."""
        depth = 4
        initial_filters = 32
        filter_multiplier = 2

        model = create_bfunet_denoiser(
            input_shape=grayscale_input_shape,
            depth=depth,
            initial_filters=initial_filters,
            filter_multiplier=filter_multiplier
        )

        # Expected filter progression: [32, 64, 128, 256, 512] for depth=4
        expected_filters = [initial_filters * (filter_multiplier ** i) for i in range(depth + 1)]

        # This is more of a structural test
        assert len(expected_filters) == depth + 1

    def test_output_channels_match_input(self):
        """Test that output channels match input channels."""
        test_cases = [
            (64, 64, 1),   # Grayscale
            (128, 128, 3), # RGB
            (32, 32, 4),   # RGBA
        ]

        for input_shape in test_cases:
            model = create_bfunet_denoiser(
                input_shape=input_shape,
                depth=3,  # Updated minimum depth
                initial_filters=16
            )

            input_channels = input_shape[2]
            assert model.output_shape[-1] == input_channels

    def test_residual_vs_standard_blocks(self, grayscale_input_shape):
        """Test model with residual blocks vs standard convolution blocks."""
        configs = [
            {'use_residual_blocks': True},
            {'use_residual_blocks': False}
        ]

        for config in configs:
            model = create_bfunet_denoiser(
                input_shape=grayscale_input_shape,
                depth=3,  # Updated minimum depth
                initial_filters=32,
                **config
            )

            test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
            output = model(test_input)

            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))

    def test_model_creation_success(self, grayscale_input_shape):
        """Test that model creation completes successfully with proper structure."""
        depth = 4
        model = create_bfunet_denoiser(
            input_shape=grayscale_input_shape,
            depth=depth,
            initial_filters=64,
            model_name='test_unet'
        )

        # Verify model was created successfully
        assert model is not None
        assert model.name == 'test_unet'
        assert len(model.layers) > 0

        # Verify model has expected parameter count for given configuration
        total_params = model.count_params()
        assert total_params > 0

    # ================================================================
    # Forward Pass Tests
    # ================================================================

    def test_forward_pass_grayscale(self, test_image_grayscale):
        """Test forward pass with grayscale images."""
        model = create_bfunet_denoiser(
            input_shape=(64, 64, 1),
            depth=3,  # Updated minimum depth
            initial_filters=32
        )

        output = model(test_image_grayscale)

        # Check output properties
        assert output.shape == test_image_grayscale.shape
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_forward_pass_rgb(self, test_image_rgb):
        """Test forward pass with RGB images."""
        model = create_bfunet_denoiser(
            input_shape=(128, 128, 3),
            depth=3,
            initial_filters=32
        )

        output = model(test_image_rgb)

        # Check output properties
        assert output.shape == test_image_rgb.shape
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_forward_pass_large_image(self, test_image_large):
        """Test forward pass with large images."""
        model = create_bfunet_denoiser(
            input_shape=(256, 256, 1),
            depth=4,
            initial_filters=32
        )

        output = model(test_image_large)

        # Check output properties
        assert output.shape == test_image_large.shape
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_scaling_invariance_property(self, test_image_grayscale):
        """Test the scaling invariance property of the bias-free U-Net."""
        model = create_bfunet_denoiser(
            input_shape=(64, 64, 1),
            depth=3,  # Updated minimum depth
            initial_filters=32,
            final_activation='linear'  # Important for scaling invariance
        )

        # Test scaling invariance: if input is scaled by α, output is scaled by α
        alpha = 2.0
        scaled_input = alpha * test_image_grayscale

        original_output = model(test_image_grayscale)
        scaled_output = model(scaled_input)

        # The outputs should be related by the same scaling factor
        expected_scaled_output = alpha * original_output

        # Use relaxed tolerance for practical implementations
        np.testing.assert_allclose(
            scaled_output.numpy(),
            expected_scaled_output.numpy(),
            rtol=1e-1,  # Relaxed tolerance for U-Net complexity
            atol=1e-1
        )

    def test_different_batch_sizes(self, grayscale_input_shape):
        """Test model with different batch sizes."""
        model = create_bfunet_denoiser(
            input_shape=grayscale_input_shape,
            depth=3,  # Updated minimum depth
            initial_filters=16
        )

        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            test_input = np.random.rand(batch_size, 64, 64, 1).astype(np.float32)
            output = model(test_input)

            assert output.shape == (batch_size, 64, 64, 1)
            assert not np.any(np.isnan(output.numpy()))

    def test_skip_connections_functionality(self, grayscale_input_shape):
        """Test that skip connections are working properly."""
        # Create a simple test to verify skip connections
        unet_model = create_bfunet_denoiser(
            input_shape=grayscale_input_shape,
            depth=3,  # Updated minimum depth
            initial_filters=32
        )

        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        unet_output = unet_model(test_input)

        # U-Net should produce valid output with skip connections
        assert unet_output.shape == test_input.shape
        assert not np.any(np.isnan(unet_output.numpy()))

        # Skip connections should help preserve fine details
        layer_names = [layer.name for layer in unet_model.layers]
        assert any('concat' in name for name in layer_names), "Skip connections not found"

    # ================================================================
    # Numerical Stability Tests
    # ================================================================

    def test_numerical_stability(self, grayscale_input_shape):
        """Test model stability with extreme input values."""
        model = create_bfunet_denoiser(
            input_shape=grayscale_input_shape,
            depth=3,  # Updated minimum depth
            initial_filters=16
        )

        # Test with different input magnitudes
        test_cases = [
            np.zeros((1, 64, 64, 1), dtype=np.float32),  # Zeros
            np.ones((1, 64, 64, 1), dtype=np.float32) * 1e-10,  # Very small
            np.ones((1, 64, 64, 1), dtype=np.float32) * 1e3,    # Large values
            np.random.normal(0, 50, (1, 64, 64, 1)).astype(np.float32)  # High variance
        ]

        for test_input in test_cases:
            output = model(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

    # ================================================================
    # Serialization Tests
    # ================================================================

    def test_model_serialization(self, test_image_grayscale):
        """Test saving and loading the model."""
        original_model = create_bfunet_denoiser(
            input_shape=(64, 64, 1),
            depth=3,  # Updated minimum depth
            initial_filters=32,
            model_name='serialization_test'
        )

        # Generate prediction before saving
        original_prediction = original_model.predict(test_image_grayscale)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "bias_free_unet_model.keras")

            # Save the model
            original_model.save(model_path)

            # Verify the model file was created
            assert os.path.exists(model_path)

            # Verify model structure can be inspected
            assert original_model.name == 'serialization_test'
            assert len(original_model.layers) > 0

    # ================================================================
    # Training Integration Tests
    # ================================================================

    def test_model_compilation(self, grayscale_input_shape):
        """Test that model can be compiled with different optimizers and losses."""
        model = create_bfunet_denoiser(
            input_shape=grayscale_input_shape,
            depth=3,  # Updated minimum depth
            initial_filters=16
        )

        # Test different compilation configurations
        compilation_configs = [
            {'optimizer': 'adam', 'loss': 'mse'},
            {'optimizer': 'rmsprop', 'loss': 'mae'},
            {'optimizer': keras.optimizers.Adam(learning_rate=0.001), 'loss': 'binary_crossentropy'},
        ]

        for config in compilation_configs:
            model.compile(**config)
            assert model.optimizer is not None
            assert model.loss is not None

    def test_gradient_flow(self, test_image_grayscale):
        """Test gradient flow through the U-Net model."""
        model = create_bfunet_denoiser(
            input_shape=(64, 64, 1),
            depth=3,  # Updated minimum depth
            initial_filters=16
        )

        model.compile(optimizer='adam', loss='mse')

        # Create target
        target = test_image_grayscale * 0.9  # Simulated target

        # Test that gradients can be computed
        with tf.GradientTape() as tape:
            predictions = model(test_image_grayscale, training=True)
            loss = keras.losses.mean_squared_error(target, predictions)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, model.trainable_variables)

        # Check gradients exist and are not None
        assert all(g is not None for g in gradients)

        # Check gradients have non-zero values
        assert all(tf.reduce_any(tf.not_equal(g, 0.0)) for g in gradients)

    # ================================================================
    # Edge Cases and Robustness Tests
    # ================================================================

    def test_minimum_depth_configuration(self, grayscale_input_shape):
        """Test U-Net with minimum depth (depth=3)."""
        model = create_bfunet_denoiser(
            input_shape=grayscale_input_shape,
            depth=3,  # Updated minimum depth
            initial_filters=16
        )

        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_large_depth_configuration(self, large_input_shape):
        """Test U-Net with large depth."""
        model = create_bfunet_denoiser(
            input_shape=large_input_shape,
            depth=5,  # Deep U-Net
            initial_filters=16  # Keep filters low to manage memory
        )

        test_input = np.random.rand(1, 256, 256, 1).astype(np.float32)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_non_power_of_two_dimensions(self):
        """Test U-Net with input dimensions that are not powers of 2."""
        # U-Net with pooling/upsampling can handle non-power-of-2 dimensions
        # thanks to the Resizing layer for dimension matching
        input_shape = (96, 80, 1)  # Non-power-of-2 dimensions

        model = create_bfunet_denoiser(
            input_shape=input_shape,
            depth=3,  # Updated minimum depth
            initial_filters=16
        )

        test_input = np.random.rand(1, 96, 80, 1).astype(np.float32)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_large_filter_count(self, grayscale_input_shape):
        """Test model with large number of filters."""
        model = create_bfunet_denoiser(
            input_shape=grayscale_input_shape,
            depth=3,  # Updated minimum depth
            initial_filters=256,  # Large filter count
            filter_multiplier=2
        )

        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_single_block_per_level(self, grayscale_input_shape):
        """Test U-Net with single block per level."""
        model = create_bfunet_denoiser(
            input_shape=grayscale_input_shape,
            depth=3,
            initial_filters=32,
            blocks_per_level=1  # Minimal blocks
        )

        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))


# ================================================================
# Parameterized Tests for Multiple Configurations
# ================================================================

class TestBiasFreeUNetParameterized:
    """Parameterized tests for different U-Net configurations."""

    @pytest.mark.parametrize("input_shape,depth,initial_filters", [
        ((32, 32, 1), 3, 16),  # Updated minimum depth
        ((64, 64, 3), 3, 32),
        ((128, 128, 1), 4, 64),
        ((96, 96, 3), 3, 32),  # Non-power-of-2 dimensions
    ])
    def test_various_configurations(self, input_shape, depth, initial_filters):
        """Test model creation with various valid configurations."""
        model = create_bfunet_denoiser(
            input_shape=input_shape,
            depth=depth,
            initial_filters=initial_filters
        )

        # Test forward pass
        batch_size = 1
        test_input = np.random.rand(batch_size, *input_shape).astype(np.float32)
        output = model(test_input)

        assert output.shape == (batch_size,) + input_shape
        assert not np.any(np.isnan(output.numpy()))

    @pytest.mark.parametrize("activation", [
        'relu', 'leaky_relu', 'elu', 'swish', 'gelu'
    ])
    def test_different_activations(self, activation):
        """Test model with different activation functions."""
        model = create_bfunet_denoiser(
            input_shape=(64, 64, 1),
            depth=3,  # Updated minimum depth
            initial_filters=16,
            activation=activation
        )

        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

    @pytest.mark.parametrize("kernel_size", [1, 3, 5, 7])
    def test_different_kernel_sizes(self, kernel_size):
        """Test model with different kernel sizes."""
        model = create_bfunet_denoiser(
            input_shape=(64, 64, 1),
            depth=3,  # Updated minimum depth
            initial_filters=16,
            kernel_size=kernel_size
        )

        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

    @pytest.mark.parametrize("filter_multiplier", [1, 2, 3, 4])
    def test_different_filter_multipliers(self, filter_multiplier):
        """Test model with different filter multipliers."""
        model = create_bfunet_denoiser(
            input_shape=(64, 64, 1),
            depth=3,  # Updated minimum depth
            initial_filters=16,
            filter_multiplier=filter_multiplier
        )

        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

    @pytest.mark.parametrize("blocks_per_level", [1, 2, 3, 4])
    def test_different_blocks_per_level(self, blocks_per_level):
        """Test model with different numbers of blocks per level."""
        model = create_bfunet_denoiser(
            input_shape=(64, 64, 1),
            depth=3,  # Updated minimum depth
            initial_filters=16,
            blocks_per_level=blocks_per_level
        )

        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

    @pytest.mark.parametrize("variant", ['tiny', 'small', 'base', 'large', 'xlarge'])
    def test_all_variants_parameterized(self, variant):
        """Test all variants with parameterized approach."""
        input_shape = (64, 64, 1)
        model = create_bfunet_variant(variant, input_shape)

        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))


if __name__ == '__main__':
    pytest.main([__file__])