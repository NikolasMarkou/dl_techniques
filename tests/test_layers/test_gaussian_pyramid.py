import pytest
import tempfile
import os
import numpy as np
import keras
from typing import Any, Dict, List, Tuple

# Import the layer to test
from dl_techniques.layers.gaussian_pyramid import GaussianPyramid, gaussian_pyramid


class TestGaussianPyramid:
    """Comprehensive test suite for GaussianPyramid layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'levels': 3,
            'kernel_size': (5, 5),
            'sigma': 1.0,
            'scale_factor': 2,
            'padding': 'same',
            'data_format': 'channels_last'
        }

    @pytest.fixture
    def sample_input_channels_last(self) -> keras.KerasTensor:
        """Sample input for channels_last format."""
        return keras.random.normal(shape=(4, 32, 32, 3))

    @pytest.fixture
    def sample_input_channels_first(self) -> keras.KerasTensor:
        """Sample input for channels_first format."""
        return keras.random.normal(shape=(4, 3, 32, 32))

    def test_initialization(self, layer_config):
        """Test layer initialization."""
        layer = GaussianPyramid(**layer_config)

        # Check attributes are stored
        assert layer.levels == layer_config['levels']
        assert layer.kernel_size == layer_config['kernel_size']
        assert layer.scale_factor == layer_config['scale_factor']
        assert layer.padding == layer_config['padding']
        assert layer.data_format == layer_config['data_format']
        assert layer.sigma == (1.0, 1.0)  # Should be converted to tuple

        # Check sub-layers are created
        assert len(layer.gaussian_filters) == layer_config['levels']
        assert not layer.built  # Layer should not be built yet

        # Sub-layers should exist but not be built
        for gaussian_filter in layer.gaussian_filters:
            assert gaussian_filter is not None
            assert not gaussian_filter.built

    def test_forward_pass_channels_last(self, layer_config, sample_input_channels_last):
        """Test forward pass with channels_last format and downsampling."""
        layer = GaussianPyramid(**layer_config)

        outputs = layer(sample_input_channels_last)

        # Check layer is built
        assert layer.built

        # Check output structure
        assert isinstance(outputs, list)
        assert len(outputs) == layer_config['levels']

        # Check shapes progression with scale_factor=2
        expected_shapes = [
            (4, 32, 32, 3),  # Level 0: original size
            (4, 16, 16, 3),  # Level 1: downsampled by 2
            (4, 8, 8, 3),  # Level 2: downsampled by 4
        ]

        for i, (output, expected_shape) in enumerate(zip(outputs, expected_shapes)):
            assert output.shape == expected_shape, f"Level {i} shape mismatch"
            assert output.dtype == sample_input_channels_last.dtype

    def test_forward_pass_channels_first_config_only(self, sample_input_channels_first):
        """Test channels_first configuration without forward pass (CPU limitation)."""
        layer_config = {
            'levels': 3,
            'kernel_size': (5, 5),
            'sigma': 1.0,
            'scale_factor': 2,
            'data_format': 'channels_first'
        }
        layer = GaussianPyramid(**layer_config)

        # Test configuration is correct
        assert layer.data_format == 'channels_first'
        assert layer.levels == 3

        # Test compute_output_shape works for channels_first
        input_shape = (None, 3, 32, 32)
        output_shapes = layer.compute_output_shape(input_shape)

        expected_shapes = [
            (None, 3, 32, 32),  # Level 0: original size
            (None, 3, 16, 16),  # Level 1: downsampled by 2
            (None, 3, 8, 8),  # Level 2: downsampled by 4
        ]

        for i, (computed_shape, expected_shape) in enumerate(zip(output_shapes, expected_shapes)):
            assert computed_shape == expected_shape, f"Level {i} computed shape mismatch"

        # Note: Actual forward pass skipped due to TensorFlow CPU limitation
        # with depthwise convolution in NCHW format

    def test_downsampling_progression(self):
        """Test that downsampling works correctly across pyramid levels."""
        layer = GaussianPyramid(levels=4, kernel_size=(3, 3), scale_factor=2)
        sample_input = keras.random.normal(shape=(2, 64, 64, 1))

        outputs = layer(sample_input)

        # Expected shapes with scale_factor=2
        expected_shapes = [
            (2, 64, 64, 1),  # Level 0: original
            (2, 32, 32, 1),  # Level 1: /2
            (2, 16, 16, 1),  # Level 2: /4
            (2, 8, 8, 1),  # Level 3: /8
        ]

        for i, (output, expected_shape) in enumerate(zip(outputs, expected_shapes)):
            assert output.shape == expected_shape, f"Level {i} shape mismatch"

    def test_serialization_cycle(self, layer_config, sample_input_channels_last):
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input_channels_last.shape[1:])
        pyramid_outputs = GaussianPyramid(**layer_config)(inputs)

        # Since output is a list, we need to handle it properly for model creation
        # We'll use only the first level for simplicity
        model = keras.Model(inputs, pyramid_outputs[0])

        # Get original prediction
        original_pred = model(sample_input_channels_last)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_channels_last)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_serialization_cycle_full_pyramid(self, sample_input_channels_last):
        """Test serialization with all pyramid outputs."""
        # Simpler config for testing
        layer_config = {'levels': 2, 'kernel_size': (3, 3), 'sigma': 1.0}

        inputs = keras.Input(shape=sample_input_channels_last.shape[1:])
        pyramid_outputs = GaussianPyramid(**layer_config)(inputs)

        # Create a model that processes all pyramid levels
        # Concatenate all levels after flattening
        flattened_levels = [keras.layers.GlobalAveragePooling2D()(level) for level in pyramid_outputs]
        concatenated = keras.layers.Concatenate()(flattened_levels)
        model = keras.Model(inputs, concatenated)

        # Get original prediction
        original_pred = model(sample_input_channels_last)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_pyramid_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_channels_last)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Full pyramid predictions differ after serialization"
            )

    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = GaussianPyramid(**layer_config)
        config = layer.get_config()

        # Check all config parameters are present
        expected_keys = {
            'levels', 'kernel_size', 'sigma', 'scale_factor',
            'padding', 'data_format'
        }
        config_keys = set(config.keys())

        for key in expected_keys:
            assert key in config_keys, f"Missing {key} in get_config()"

        # Verify values match
        assert config['levels'] == layer_config['levels']
        assert config['kernel_size'] == layer_config['kernel_size']
        assert config['scale_factor'] == layer_config['scale_factor']
        assert config['padding'] == layer_config['padding']
        assert config['data_format'] == layer_config['data_format']

    def test_gradients_flow(self, layer_config, sample_input_channels_last):
        """Test gradient computation."""
        import tensorflow as tf

        layer = GaussianPyramid(**layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input_channels_last)
            outputs = layer(sample_input_channels_last)
            # Take mean of all pyramid levels for single scalar loss
            loss = keras.ops.mean([keras.ops.mean(keras.ops.square(output)) for output in outputs])

        # Get gradients with respect to input
        input_gradients = tape.gradient(loss, sample_input_channels_last)
        assert input_gradients is not None
        assert input_gradients.shape == sample_input_channels_last.shape

        # Check gradients with respect to trainable variables (if any)
        if layer.trainable and layer.trainable_variables:
            var_gradients = tape.gradient(loss, layer.trainable_variables)
            assert all(g is not None for g in var_gradients)

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_input_channels_last, training):
        """Test behavior in different training modes."""
        layer = GaussianPyramid(**layer_config)

        outputs = layer(sample_input_channels_last, training=training)

        assert isinstance(outputs, list)
        assert len(outputs) == layer_config['levels']
        for output in outputs:
            assert output.shape[0] == sample_input_channels_last.shape[0]

    @pytest.mark.parametrize("levels", [1, 2, 5])
    def test_different_levels(self, levels, sample_input_channels_last):
        """Test different numbers of pyramid levels."""
        layer = GaussianPyramid(levels=levels, kernel_size=(3, 3))

        outputs = layer(sample_input_channels_last)

        assert len(outputs) == levels

        # Check downsampling progression
        for i, output in enumerate(outputs):
            expected_size = 32 // (2 ** i)  # scale_factor=2 by default
            assert output.shape == (4, expected_size, expected_size, 3)

    @pytest.mark.parametrize("kernel_size", [(3, 3), (5, 5), (7, 7)])
    def test_different_kernel_sizes(self, kernel_size, sample_input_channels_last):
        """Test different kernel sizes."""
        layer = GaussianPyramid(levels=2, kernel_size=kernel_size)

        outputs = layer(sample_input_channels_last)

        assert len(outputs) == 2
        assert layer.kernel_size == kernel_size
        # Check first level is original size, second is downsampled
        assert outputs[0].shape == (4, 32, 32, 3)
        assert outputs[1].shape == (4, 16, 16, 3)

    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, (1.0, 2.0)])
    def test_different_sigma_values(self, sigma, sample_input_channels_last):
        """Test different sigma values."""
        layer = GaussianPyramid(levels=2, kernel_size=(5, 5), sigma=sigma)

        outputs = layer(sample_input_channels_last)

        assert len(outputs) == 2

        # Check sigma is processed correctly
        if isinstance(sigma, tuple):
            assert layer.sigma == sigma
        else:
            assert layer.sigma == (sigma, sigma)

    @pytest.mark.parametrize("scale_factor", [1, 2, 3, 4])
    def test_different_scale_factors(self, scale_factor):
        """Test different scale factors."""
        sample_input = keras.random.normal(shape=(2, 16, 16, 1))
        layer = GaussianPyramid(levels=3, scale_factor=scale_factor)

        outputs = layer(sample_input)

        assert len(outputs) == 3
        assert layer.scale_factor == scale_factor

        # Check progression
        for i, output in enumerate(outputs):
            expected_size = 16 // (scale_factor ** i)
            expected_size = max(1, expected_size)  # Minimum size is 1
            assert output.shape == (2, expected_size, expected_size, 1)

    @pytest.mark.parametrize("padding", ["same", "valid", "SAME", "VALID"])
    def test_different_padding(self, padding, sample_input_channels_last):
        """Test different padding modes (case insensitive)."""
        layer = GaussianPyramid(levels=2, padding=padding)

        outputs = layer(sample_input_channels_last)

        assert len(outputs) == 2
        assert layer.padding == padding.lower()

    @pytest.mark.parametrize("trainable", [True, False])
    def test_trainable_parameter(self, trainable, sample_input_channels_last):
        """Test trainable parameter."""
        layer = GaussianPyramid(levels=2, trainable=trainable)

        outputs = layer(sample_input_channels_last)

        assert len(outputs) == 2
        assert layer.trainable == trainable

        # Check that sub-layers have correct trainable setting
        for gaussian_filter in layer.gaussian_filters:
            assert gaussian_filter.trainable == trainable

    def test_edge_cases(self):
        """Test error conditions."""
        # Test invalid levels
        with pytest.raises(ValueError, match="levels must be >= 1"):
            GaussianPyramid(levels=0)

        with pytest.raises(ValueError, match="levels must be >= 1"):
            GaussianPyramid(levels=-1)

        # Test invalid kernel size
        with pytest.raises(ValueError, match="kernel_size must be length 2"):
            GaussianPyramid(kernel_size=(5,))

        with pytest.raises(ValueError, match="kernel_size must be length 2"):
            GaussianPyramid(kernel_size=(5, 5, 5))

        # Test invalid scale factor
        with pytest.raises(ValueError, match="scale_factor must be >= 1"):
            GaussianPyramid(scale_factor=0)

        with pytest.raises(ValueError, match="scale_factor must be >= 1"):
            GaussianPyramid(scale_factor=-1)

        # Test invalid padding
        with pytest.raises(ValueError, match="padding must be 'valid' or 'same'"):
            GaussianPyramid(padding="invalid")

        # Test invalid data format
        with pytest.raises(ValueError, match="data_format must be 'channels_first' or 'channels_last'"):
            GaussianPyramid(data_format="invalid")

    def test_compute_output_shape(self, layer_config):
        """Test compute_output_shape method."""
        layer = GaussianPyramid(**layer_config)
        input_shape = (None, 32, 32, 3)

        output_shapes = layer.compute_output_shape(input_shape)

        assert isinstance(output_shapes, list)
        assert len(output_shapes) == layer_config['levels']

        # Check downsampling progression in computed shapes
        expected_shapes = [
            (None, 32, 32, 3),  # Level 0
            (None, 16, 16, 3),  # Level 1
            (None, 8, 8, 3),  # Level 2
        ]

        for computed_shape, expected_shape in zip(output_shapes, expected_shapes):
            assert computed_shape == expected_shape

    def test_compute_output_shape_channels_first(self):
        """Test compute_output_shape with channels_first."""
        layer = GaussianPyramid(levels=3, data_format='channels_first')
        input_shape = (None, 3, 32, 32)

        output_shapes = layer.compute_output_shape(input_shape)

        expected_shapes = [
            (None, 3, 32, 32),  # Level 0
            (None, 3, 16, 16),  # Level 1
            (None, 3, 8, 8),  # Level 2
        ]

        for computed_shape, expected_shape in zip(output_shapes, expected_shapes):
            assert computed_shape == expected_shape

    def test_functional_interface(self, sample_input_channels_last):
        """Test functional interface."""
        outputs = gaussian_pyramid(
            sample_input_channels_last,
            levels=3,
            kernel_size=(5, 5),
            sigma=1.0
        )

        assert isinstance(outputs, list)
        assert len(outputs) == 3

        # Check downsampling progression
        expected_shapes = [
            (4, 32, 32, 3),  # Level 0
            (4, 16, 16, 3),  # Level 1
            (4, 8, 8, 3),  # Level 2
        ]

        for output, expected_shape in zip(outputs, expected_shapes):
            assert output.shape == expected_shape

    def test_functional_vs_layer_consistency(self, sample_input_channels_last):
        """Test that functional interface gives same results as layer interface."""
        # Layer interface
        layer = GaussianPyramid(levels=2, kernel_size=(3, 3), sigma=1.0)
        layer_outputs = layer(sample_input_channels_last)

        # Functional interface
        func_outputs = gaussian_pyramid(
            sample_input_channels_last,
            levels=2,
            kernel_size=(3, 3),
            sigma=1.0
        )

        # Compare outputs
        assert len(layer_outputs) == len(func_outputs)
        for layer_out, func_out in zip(layer_outputs, func_outputs):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(layer_out),
                keras.ops.convert_to_numpy(func_out),
                rtol=1e-6, atol=1e-6,
                err_msg="Layer and functional interfaces should give same results"
            )

    def test_sigma_auto_calculation(self, sample_input_channels_last):
        """Test automatic sigma calculation when sigma=-1."""
        layer = GaussianPyramid(levels=2, kernel_size=(7, 7), sigma=-1)

        # Should auto-calculate sigma based on kernel size
        expected_sigma = ((7 - 1) / 2, (7 - 1) / 2)
        assert layer.sigma == expected_sigma

        # Should still work
        outputs = layer(sample_input_channels_last)
        assert len(outputs) == 2

    def test_sigma_none_calculation(self, sample_input_channels_last):
        """Test automatic sigma calculation when sigma=None."""
        layer = GaussianPyramid(levels=2, kernel_size=(5, 5), sigma=None)

        # Should auto-calculate sigma based on kernel size
        expected_sigma = ((5 - 1) / 2, (5 - 1) / 2)
        assert layer.sigma == expected_sigma

        # Should still work
        outputs = layer(sample_input_channels_last)
        assert len(outputs) == 2

    def test_build_method_explicit(self):
        """Test that build method works with explicit shapes."""
        layer = GaussianPyramid(levels=3, kernel_size=(5, 5))
        input_shape = (None, 64, 64, 3)

        # Explicitly call build
        layer.build(input_shape)

        assert layer.built
        # Check that all sub-layers are built
        for gaussian_filter in layer.gaussian_filters:
            assert gaussian_filter.built

    def test_realistic_small_input_downsampling(self):
        """Test behavior with small inputs using realistic expectations."""
        # Use 8x8 input which can be reasonably downsampled
        small_input = keras.random.normal(shape=(2, 8, 8, 1))
        layer = GaussianPyramid(levels=3, scale_factor=2)

        outputs = layer(small_input)

        # Expected progression: 8 -> 4 -> 2
        expected_shapes = [
            (2, 8, 8, 1),  # Level 0: original
            (2, 4, 4, 1),  # Level 1: downsampled by 2
            (2, 2, 2, 1),  # Level 2: downsampled by 4
        ]

        for i, (output, expected_shape) in enumerate(zip(outputs, expected_shapes)):
            assert output.shape == expected_shape, f"Level {i} shape mismatch"

    def test_very_small_input_edge_case(self):
        """Test very small input that demonstrates downsampling limitations."""
        # Use 4x4 input to show what happens at the edge
        very_small_input = keras.random.normal(shape=(2, 4, 4, 1))
        layer = GaussianPyramid(levels=4, scale_factor=2)

        outputs = layer(very_small_input)

        # Actual progression: 4 -> 2 -> 1 -> 0 (this is what average pooling does)
        expected_shapes = [
            (2, 4, 4, 1),  # Level 0: original
            (2, 2, 2, 1),  # Level 1: downsampled by 2
            (2, 1, 1, 1),  # Level 2: downsampled by 4
            (2, 0, 0, 1),  # Level 3: downsampled to 0 (edge case behavior)
        ]

        for i, (output, expected_shape) in enumerate(zip(outputs, expected_shapes)):
            assert output.shape == expected_shape, f"Level {i} actual shape: {output.shape}, expected: {expected_shape}"

    def test_data_format_none_defaults(self, sample_input_channels_last):
        """Test that data_format=None uses Keras default."""
        # Save current setting
        original_format = keras.backend.image_data_format()

        try:
            # Set to channels_last
            keras.backend.set_image_data_format('channels_last')

            layer = GaussianPyramid(levels=2, data_format=None)
            assert layer.data_format == 'channels_last'

            # Set to channels_first
            keras.backend.set_image_data_format('channels_first')

            layer = GaussianPyramid(levels=2, data_format=None)
            assert layer.data_format == 'channels_first'

        finally:
            # Restore original setting
            keras.backend.set_image_data_format(original_format)
