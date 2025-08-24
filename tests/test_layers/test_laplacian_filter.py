import pytest
import tempfile
import os
import numpy as np
import keras
from typing import Any, Dict, List, Tuple, Literal

# Import the layers to test
from dl_techniques.layers.laplacian_filter import LaplacianFilter, AdvancedLaplacianFilter


class TestLaplacianFilter:
    """Comprehensive test suite for LaplacianFilter layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'kernel_size': (5, 5),
            'strides': (1, 1),
            'sigma': 1.0,
            'scale_factor': 1.0
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample input for testing."""
        return keras.random.normal(shape=(4, 32, 32, 3))

    def test_initialization(self, layer_config):
        """Test layer initialization."""
        layer = LaplacianFilter(**layer_config)

        # Check attributes are stored
        assert layer.kernel_size == layer_config['kernel_size']
        assert layer.strides == layer_config['strides']
        assert layer.scale_factor == layer_config['scale_factor']
        assert layer.sigma == (1.0, 1.0)  # Should be converted to tuple

        # Check sub-layer is created
        assert layer.gaussian_filter is not None
        assert not layer.built  # Layer should not be built yet
        assert not layer.gaussian_filter.built  # Sub-layer should not be built yet

    def test_forward_pass(self, layer_config, sample_input):
        """Test forward pass functionality."""
        layer = LaplacianFilter(**layer_config)

        output = layer(sample_input)

        # Check layer is built
        assert layer.built
        assert layer.gaussian_filter.built

        # Check output properties
        assert output.shape == sample_input.shape  # Same shape as input
        assert output.dtype == sample_input.dtype

        # Check that output is different from input (edges detected)
        assert not np.allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(sample_input),
            rtol=1e-5, atol=1e-5
        )

    def test_edge_detection_properties(self, sample_input):
        """Test that the filter actually detects edges."""
        layer = LaplacianFilter(kernel_size=(5, 5), sigma=1.0, scale_factor=1.0)

        # Create an image with a clear vertical edge (step function)
        batch_size, height, width, channels = sample_input.shape
        edge_image = keras.ops.zeros((batch_size, height, width, channels))

        # Create a simple step function: left half = 0, right half = 1
        left_half = keras.ops.zeros((batch_size, height, width // 2, channels))
        right_half = keras.ops.ones((batch_size, height, width - width // 2, channels))
        edge_image = keras.ops.concatenate([left_half, right_half], axis=2)

        output = layer(edge_image)

        # The Laplacian should produce high absolute responses at edge locations
        # Check that the output has significant non-zero values (indicating edge detection)
        output_abs_mean = keras.ops.mean(keras.ops.abs(output))

        # For a step edge, the Laplacian should produce significant responses
        # The mean absolute value should be greater than a small threshold
        assert keras.ops.convert_to_numpy(
            output_abs_mean) > 0.01, f"Expected edge response > 0.01, got {keras.ops.convert_to_numpy(output_abs_mean)}"

        # FIX: Also check that it's different from applying to uniform ZERO image
        # Using a zero image ensures the output is zero, avoiding padding artifacts.
        uniform_image = keras.ops.zeros_like(edge_image)
        uniform_output = layer(uniform_image)
        uniform_abs_mean = keras.ops.mean(keras.ops.abs(uniform_output))

        # Edge image should have higher response than uniform image
        assert keras.ops.convert_to_numpy(output_abs_mean) > keras.ops.convert_to_numpy(uniform_abs_mean)

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = LaplacianFilter(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = LaplacianFilter(**layer_config)
        config = layer.get_config()

        # Check all config parameters are present
        expected_keys = {
            'kernel_size', 'strides', 'sigma', 'scale_factor',
            'kernel_initializer', 'kernel_regularizer'
        }
        config_keys = set(config.keys())

        for key in expected_keys:
            assert key in config_keys, f"Missing {key} in get_config()"

        # Verify values match
        assert config['kernel_size'] == layer_config['kernel_size']
        assert config['strides'] == layer_config['strides']
        assert config['scale_factor'] == layer_config['scale_factor']
        assert config['sigma'] == (1.0, 1.0)  # Should be tuple

    def test_gradients_flow(self, layer_config, sample_input):
        """Test gradient computation."""
        import tensorflow as tf

        layer = LaplacianFilter(**layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        # Get gradients with respect to input
        input_gradients = tape.gradient(loss, sample_input)
        assert input_gradients is not None
        assert input_gradients.shape == sample_input.shape

        # Check gradients with respect to trainable variables (if any)
        if layer.trainable and layer.trainable_variables:
            var_gradients = tape.gradient(loss, layer.trainable_variables)
            assert all(g is not None for g in var_gradients)

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_input, training):
        """Test behavior in different training modes."""
        layer = LaplacianFilter(**layer_config)

        output = layer(sample_input, training=training)

        assert output.shape == sample_input.shape
        assert output.dtype == sample_input.dtype

    @pytest.mark.parametrize("kernel_size", [(3, 3), (5, 5), (7, 7)])
    def test_different_kernel_sizes(self, kernel_size, sample_input):
        """Test different kernel sizes."""
        layer = LaplacianFilter(kernel_size=kernel_size, sigma=1.0)

        output = layer(sample_input)

        assert output.shape == sample_input.shape
        assert layer.kernel_size == kernel_size
        assert layer.gaussian_filter.kernel_size == kernel_size

    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, (1.0, 1.5)])
    def test_different_sigma_values(self, sigma, sample_input):
        """Test different sigma values."""
        layer = LaplacianFilter(kernel_size=(5, 5), sigma=sigma)

        output = layer(sample_input)

        assert output.shape == sample_input.shape

        # Check sigma is processed correctly
        if isinstance(sigma, tuple):
            assert layer.sigma == sigma
        else:
            assert layer.sigma == (sigma, sigma)

    @pytest.mark.parametrize("scale_factor", [0.5, 1.0, 2.0, 5.0])
    def test_different_scale_factors(self, scale_factor, sample_input):
        """Test different scale factors."""
        layer1 = LaplacianFilter(kernel_size=(5, 5), scale_factor=1.0)
        layer2 = LaplacianFilter(kernel_size=(5, 5), scale_factor=scale_factor)

        output1 = layer1(sample_input)
        output2 = layer2(sample_input)

        if scale_factor != 1.0:
            # FIX: Use np.allclose as keras.ops.allclose does not exist.
            # Outputs should be different when scale factors differ
            assert not np.allclose(
                keras.ops.convert_to_numpy(output1),
                keras.ops.convert_to_numpy(output2)
            )

        # Check that scale_factor is applied correctly
        assert layer2.scale_factor == scale_factor

    @pytest.mark.parametrize("strides", [(1, 1), (2, 2)])
    def test_different_strides(self, strides, sample_input):
        """Test different stride values."""
        layer = LaplacianFilter(kernel_size=(5, 5), strides=strides)

        output = layer(sample_input)

        # Note: For LaplacianFilter, strides are ignored and the effective stride is
        # always (1, 1) to maintain shape for subtraction.
        assert output.shape == sample_input.shape
        assert layer.strides == strides

    def test_sigma_auto_calculation(self, sample_input):
        """Test automatic sigma calculation when sigma=None."""
        layer = LaplacianFilter(kernel_size=(7, 7), sigma=None)

        # Should auto-calculate sigma based on kernel size
        expected_sigma = ((7 - 1) / 2, (7 - 1) / 2)
        assert layer.sigma == expected_sigma

        # Should still work
        output = layer(sample_input)
        assert output.shape == sample_input.shape

    def test_sigma_zero_calculation(self, sample_input):
        """Test automatic sigma calculation when sigma=0."""
        layer = LaplacianFilter(kernel_size=(5, 5), sigma=0.0)

        # Should auto-calculate sigma based on kernel size
        expected_sigma = ((5 - 1) / 2, (5 - 1) / 2)
        assert layer.sigma == expected_sigma

        # Should still work
        output = layer(sample_input)
        assert output.shape == sample_input.shape

    def test_edge_cases(self):
        """Test error conditions."""
        # Test invalid kernel size
        with pytest.raises(ValueError, match="kernel_size must be length 2"):
            LaplacianFilter(kernel_size=(5,))

        with pytest.raises(ValueError, match="kernel_size must be length 2"):
            LaplacianFilter(kernel_size=(5, 5, 5))

        # Test invalid sigma
        with pytest.raises(ValueError, match="Invalid sigma value"):
            LaplacianFilter(kernel_size=(5, 5), sigma="invalid")

    def test_compute_output_shape(self, layer_config):
        """Test compute_output_shape method."""
        layer = LaplacianFilter(**layer_config)
        input_shape = (None, 64, 64, 3)

        output_shape = layer.compute_output_shape(input_shape)

        # Output shape should be same as input shape
        assert output_shape == input_shape

    def test_build_method_explicit(self):
        """Test that build method works with explicit shapes."""
        layer = LaplacianFilter(kernel_size=(5, 5), sigma=1.0)
        input_shape = (None, 32, 32, 3)

        # Explicitly call build
        layer.build(input_shape)

        assert layer.built
        assert layer.gaussian_filter.built

    def test_list_strides_conversion(self, sample_input):
        """Test that list strides are converted to tuples."""
        layer = LaplacianFilter(kernel_size=(5, 5), strides=[2, 2])

        output = layer(sample_input)

        assert layer.strides == (2, 2)  # Should be converted to tuple
        assert output.shape == sample_input.shape


class TestAdvancedLaplacianFilter:
    """Comprehensive test suite for AdvancedLaplacianFilter layer."""

    @pytest.fixture
    def layer_config_dog(self) -> Dict[str, Any]:
        """Configuration for DoG method."""
        return {
            'method': 'dog',
            'kernel_size': (5, 5),
            'strides': (1, 1),
            'sigma': 1.0,
            'scale_factor': 1.0
        }

    @pytest.fixture
    def layer_config_log(self) -> Dict[str, Any]:
        """Configuration for LoG method."""
        return {
            'method': 'log',
            'kernel_size': (5, 5),
            'strides': (1, 1),
            'sigma': 1.0,
            'scale_factor': 1.0
        }

    @pytest.fixture
    def layer_config_kernel(self) -> Dict[str, Any]:
        """Configuration for kernel method."""
        return {
            'method': 'kernel',
            'kernel_size': (3, 3),
            'strides': (1, 1),
            'sigma': 1.0,
            'scale_factor': 1.0
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample input for testing."""
        return keras.random.normal(shape=(4, 32, 32, 3))

    def test_initialization_dog(self, layer_config_dog):
        """Test initialization with DoG method."""
        layer = AdvancedLaplacianFilter(**layer_config_dog)

        # Check attributes
        assert layer.method == 'dog'
        assert layer.kernel_size == layer_config_dog['kernel_size']
        assert layer.sigma == (1.0, 1.0)

        # DoG method should create GaussianFilter sub-layer
        assert layer.gaussian_filter is not None
        assert not layer.gaussian_filter.built
        assert layer.filter_kernel is None

    def test_initialization_log(self, layer_config_log):
        """Test initialization with LoG method."""
        layer = AdvancedLaplacianFilter(**layer_config_log)

        # Check attributes
        assert layer.method == 'log'
        assert layer.kernel_size == layer_config_log['kernel_size']

        # LoG method should not create GaussianFilter
        assert layer.gaussian_filter is None
        assert layer.filter_kernel is None  # Created in build()

    def test_initialization_kernel(self, layer_config_kernel):
        """Test initialization with kernel method."""
        layer = AdvancedLaplacianFilter(**layer_config_kernel)

        # Check attributes
        assert layer.method == 'kernel'
        assert layer.kernel_size == layer_config_kernel['kernel_size']

        # Kernel method should not create GaussianFilter
        assert layer.gaussian_filter is None
        assert layer.filter_kernel is None  # Created in build()

    def test_forward_pass_dog_method(self, layer_config_dog, sample_input):
        """Test forward pass with DoG method."""
        layer = AdvancedLaplacianFilter(**layer_config_dog)

        output = layer(sample_input)

        assert layer.built
        assert layer.gaussian_filter.built
        assert output.shape == sample_input.shape
        assert output.dtype == sample_input.dtype

    def test_forward_pass_log_method_channels_last_only(self, sample_input):
        """Test LoG method configuration (CPU limitation noted)."""
        layer_config = {
            'method': 'log',
            'kernel_size': (5, 5),
            'sigma': 1.0,
            'scale_factor': 1.0
        }
        layer = AdvancedLaplacianFilter(**layer_config)

        # Test that layer builds correctly and runs
        output = layer(sample_input)
        assert layer.built
        assert layer.filter_kernel is not None
        assert output.shape == sample_input.shape

    def test_forward_pass_kernel_method_channels_last_only(self, sample_input):
        """Test kernel method configuration (CPU limitation noted)."""
        layer_config = {
            'method': 'kernel',
            'kernel_size': (3, 3),
            'sigma': 1.0,
            'scale_factor': 1.0
        }
        layer = AdvancedLaplacianFilter(**layer_config)

        # Test that layer builds correctly and runs
        output = layer(sample_input)
        assert layer.built
        assert layer.filter_kernel is not None
        assert output.shape == sample_input.shape

    @pytest.mark.parametrize("method", ['dog', 'log', 'kernel'])
    def test_serialization_cycle_all_methods(self, method, sample_input):
        """Test serialization cycle for all methods."""
        layer_config = {
            'method': method,
            'kernel_size': (5, 5),
            'sigma': 1.0,
            'scale_factor': 1.0
        }

        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = AdvancedLaplacianFilter(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_advanced_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_config_completeness_all_methods(self):
        """Test that get_config contains all __init__ parameters for all methods."""
        methods = ['dog', 'log', 'kernel']

        for method in methods:
            layer_config = {
                'method': method,
                'kernel_size': (5, 5),
                'sigma': 1.5,
                'scale_factor': 2.0
            }
            layer = AdvancedLaplacianFilter(**layer_config)
            config = layer.get_config()

            # Check all config parameters are present
            expected_keys = {
                'method', 'kernel_size', 'strides', 'sigma', 'scale_factor',
                'kernel_initializer', 'kernel_regularizer'
            }
            config_keys = set(config.keys())

            for key in expected_keys:
                assert key in config_keys, f"Missing {key} in get_config() for method {method}"

            # Verify values match
            assert config['method'] == method
            assert config['kernel_size'] == layer_config['kernel_size']
            assert config['scale_factor'] == layer_config['scale_factor']
            assert config['sigma'] == (1.5, 1.5)  # Should be tuple

    def test_gradients_flow_dog_method(self, layer_config_dog, sample_input):
        """Test gradient computation for DoG method."""
        import tensorflow as tf

        layer = AdvancedLaplacianFilter(**layer_config_dog)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        # Get gradients with respect to input
        input_gradients = tape.gradient(loss, sample_input)
        assert input_gradients is not None
        assert input_gradients.shape == sample_input.shape

    @pytest.mark.parametrize("method", ['dog', 'log', 'kernel'])
    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, method, training, sample_input):
        """Test behavior in different training modes."""
        layer_config = {'method': method, 'kernel_size': (5, 5), 'sigma': 1.0}
        layer = AdvancedLaplacianFilter(**layer_config)

        output = layer(sample_input, training=training)

        assert output.shape == sample_input.shape
        assert output.dtype == sample_input.dtype

    def test_method_comparison_dog_vs_log_config(self, sample_input):
        """Compare DoG vs LoG method configurations."""
        dog_layer = AdvancedLaplacianFilter(method='dog', kernel_size=(5, 5), sigma=1.0)
        log_layer = AdvancedLaplacianFilter(method='log', kernel_size=(5, 5), sigma=1.0)

        # Build both layers
        dog_layer.build(sample_input.shape)
        log_layer.build(sample_input.shape)

        # DoG should have GaussianFilter, LoG should have filter_kernel
        assert dog_layer.gaussian_filter is not None
        assert dog_layer.filter_kernel is None

        assert log_layer.gaussian_filter is None
        assert log_layer.filter_kernel is not None

    def test_log_kernel_creation(self, sample_input):
        """Test LoG kernel creation."""
        layer = AdvancedLaplacianFilter(method='log', kernel_size=(5, 5), sigma=1.0)

        # Build layer to create kernel
        layer.build(sample_input.shape)

        # Check kernel properties
        assert layer.filter_kernel is not None
        expected_shape = (5, 5, 3, 1)  # (height, width, channels, depth_multiplier)
        assert layer.filter_kernel.shape == expected_shape

    def test_discrete_kernel_creation(self, sample_input):
        """Test discrete Laplacian kernel creation."""
        # Test with 3x3 kernel (should use standard Laplacian)
        layer = AdvancedLaplacianFilter(method='kernel', kernel_size=(3, 3), sigma=1.0)
        layer.build(sample_input.shape)

        assert layer.filter_kernel is not None
        expected_shape = (3, 3, 3, 1)
        assert layer.filter_kernel.shape == expected_shape

        # Test with larger kernel (should use LoG approximation)
        layer_large = AdvancedLaplacianFilter(method='kernel', kernel_size=(5, 5), sigma=1.0)
        layer_large.build(sample_input.shape)

        assert layer_large.filter_kernel is not None
        expected_shape_large = (5, 5, 3, 1)
        assert layer_large.filter_kernel.shape == expected_shape_large

    @pytest.mark.parametrize("sigma", [(1.0, 1.0), (1.0, 2.0), 1.5])
    def test_different_sigma_formats(self, sigma, sample_input):
        """Test different sigma format handling."""
        layer = AdvancedLaplacianFilter(method='dog', kernel_size=(5, 5), sigma=sigma)

        if isinstance(sigma, tuple):
            assert layer.sigma == sigma
        else:
            assert layer.sigma == (sigma, sigma)

        # Should work with DoG method
        output = layer(sample_input)
        assert output.shape == sample_input.shape

    def test_edge_cases(self):
        """Test error conditions."""
        # Test invalid method
        with pytest.raises(ValueError, match="Method 'invalid' not supported"):
            AdvancedLaplacianFilter(method='invalid')

        # Test invalid sigma
        with pytest.raises(ValueError, match="Invalid sigma value"):
            AdvancedLaplacianFilter(method='dog', sigma='invalid')

    def test_compute_output_shape_all_methods(self):
        """Test compute_output_shape for all methods."""
        methods = ['dog', 'log', 'kernel']
        input_shape = (None, 64, 64, 3)

        for method in methods:
            layer = AdvancedLaplacianFilter(method=method, kernel_size=(5, 5))
            output_shape = layer.compute_output_shape(input_shape)

            # Output shape should be same as input shape for all methods with stride=1
            assert output_shape == input_shape, f"Failed for method {method}"

    def test_build_channel_none_error(self):
        """Test error when channels dimension is None."""
        layer = AdvancedLaplacianFilter(method='log', kernel_size=(5, 5))

        # Input shape with None channels should raise error
        with pytest.raises(ValueError, match="Last dimension \\(channels\\) of input must be defined"):
            layer.build((None, 32, 32, None))

    def test_list_strides_conversion(self, sample_input):
        """Test that list strides are converted to tuples."""
        layer = AdvancedLaplacianFilter(method='dog', kernel_size=(5, 5), strides=[2, 2])

        output = layer(sample_input)

        assert layer.strides == (2, 2)  # Should be converted to tuple
        assert output.shape == sample_input.shape


class TestLaplacianFilterComparison:
    """Tests comparing LaplacianFilter and AdvancedLaplacianFilter behavior."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample input for comparison tests."""
        return keras.random.normal(shape=(2, 16, 16, 1))

    def test_basic_vs_advanced_dog_consistency(self, sample_input):
        """Test that LaplacianFilter and AdvancedLaplacianFilter DoG give similar results."""
        # Same configuration
        config = {'kernel_size': (5, 5), 'sigma': 1.0, 'scale_factor': 1.0}

        basic_layer = LaplacianFilter(**config)
        advanced_layer = AdvancedLaplacianFilter(method='dog', **config)

        basic_output = basic_layer(sample_input)
        advanced_output = advanced_layer(sample_input)

        # Both should have same shape
        assert basic_output.shape == advanced_output.shape

        # Results should be very close (may have slight numerical differences)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(basic_output),
            keras.ops.convert_to_numpy(advanced_output),
            rtol=1e-5, atol=1e-5,
            err_msg="Basic and Advanced DoG methods should give similar results"
        )

    def test_different_methods_produce_different_results(self, sample_input):
        """Test that different methods in AdvancedLaplacianFilter produce different results."""
        config = {'kernel_size': (5, 5), 'sigma': 1.0, 'scale_factor': 1.0}

        dog_layer = AdvancedLaplacianFilter(method='dog', **config)
        log_layer = AdvancedLaplacianFilter(method='log', **config)

        dog_output = dog_layer(sample_input)
        log_output = log_layer(sample_input)

        # The outputs should be different
        assert not np.allclose(
            keras.ops.convert_to_numpy(dog_output),
            keras.ops.convert_to_numpy(log_output)
        )


# Additional integration test
class TestLaplacianFilterIntegration:
    """Integration tests with realistic use cases."""

    def test_edge_detection_pipeline(self):
        """Test a realistic edge detection pipeline."""
        # Create a model with Laplacian edge detection
        inputs = keras.Input(shape=(64, 64, 1))

        # Apply Laplacian filter for edge detection
        edges = LaplacianFilter(kernel_size=(5, 5), sigma=1.0)(inputs)

        # Apply some processing
        processed = keras.layers.Conv2D(8, (3, 3), activation='relu')(edges)
        pooled = keras.layers.MaxPooling2D((2, 2))(processed)
        flattened = keras.layers.Flatten()(pooled)
        outputs = keras.layers.Dense(1, activation='sigmoid')(flattened)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy')

        # Test with sample data
        sample_data = keras.random.normal(shape=(4, 64, 64, 1))
        predictions = model(sample_data)

        assert predictions.shape == (4, 1)
        assert predictions.dtype == keras.mixed_precision.global_policy().compute_dtype

    def test_multi_scale_edge_detection(self):
        """Test multi-scale edge detection using different kernel sizes."""
        inputs = keras.Input(shape=(32, 32, 3))

        # Different scales of edge detection
        edges_fine = LaplacianFilter(kernel_size=(3, 3), sigma=0.5, scale_factor=1.0)(inputs)
        edges_medium = LaplacianFilter(kernel_size=(5, 5), sigma=1.0, scale_factor=1.0)(inputs)
        edges_coarse = LaplacianFilter(kernel_size=(7, 7), sigma=1.5, scale_factor=1.0)(inputs)

        # Combine multi-scale features
        combined = keras.layers.Concatenate()([edges_fine, edges_medium, edges_coarse])

        # Process combined edges
        processed = keras.layers.Conv2D(16, (1, 1), activation='relu')(combined)
        outputs = keras.layers.GlobalAveragePooling2D()(processed)

        model = keras.Model(inputs, outputs)

        # Test with sample data
        sample_data = keras.random.normal(shape=(2, 32, 32, 3))
        features = model(sample_data)

        assert features.shape == (2, 16)