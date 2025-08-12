import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

from dl_techniques.layers.gaussian_pyramid import (
    GaussianPyramid, gaussian_pyramid)


class TestGaussianPyramid:
    """Comprehensive test suite for GaussianPyramid layer."""

    @pytest.fixture
    def input_tensor_channels_last(self):
        """Create test input tensor with channels_last format."""
        return tf.random.normal([2, 32, 32, 3])

    @pytest.fixture
    def input_tensor_channels_first(self):
        """Create test input tensor with channels_first format."""
        return tf.random.normal([2, 3, 32, 32])

    @pytest.fixture
    def layer_instance(self):
        """Create default layer instance for testing."""
        return GaussianPyramid(levels=3)

    # =====================================================================
    # Initialization Tests
    # =====================================================================

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = GaussianPyramid()

        assert layer.levels == 3
        assert layer.kernel_size == (5, 5)
        assert layer.sigma == (2.0, 2.0)
        assert layer.scale_factor == 2
        assert layer.padding == "same"
        assert layer.data_format == keras.backend.image_data_format()
        assert layer.trainable is False
        assert layer.gaussian_filters == []

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        layer = GaussianPyramid(
            levels=4,
            kernel_size=(7, 7),
            sigma=1.5,
            scale_factor=3,
            padding="valid",
            data_format="channels_first",
            trainable=True
        )

        assert layer.levels == 4
        assert layer.kernel_size == (7, 7)
        assert layer.sigma == (1.5, 1.5)
        assert layer.scale_factor == 3
        assert layer.padding == "valid"
        assert layer.data_format == "channels_first"
        assert layer.trainable is True

    def test_initialization_sigma_variations(self):
        """Test initialization with different sigma values."""
        # Single float sigma
        layer1 = GaussianPyramid(sigma=2.0)
        assert layer1.sigma == (2.0, 2.0)

        # Tuple sigma
        layer2 = GaussianPyramid(sigma=(1.5, 2.0))
        assert layer2.sigma == (1.5, 2.0)

        # None sigma (should use default calculation)
        layer3 = GaussianPyramid(sigma=None)
        assert layer3.sigma == ((5 - 1) / 2, (5 - 1) / 2)  # Based on default kernel_size

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Invalid levels
        with pytest.raises(ValueError, match="levels must be >= 1"):
            GaussianPyramid(levels=0)

        # Invalid kernel_size
        with pytest.raises(ValueError, match="kernel_size must be length 2"):
            GaussianPyramid(kernel_size=(5,))

        # Invalid scale_factor
        with pytest.raises(ValueError, match="scale_factor must be >= 1"):
            GaussianPyramid(scale_factor=0)

        # Invalid padding
        with pytest.raises(ValueError, match="padding must be 'valid' or 'same'"):
            GaussianPyramid(padding="invalid")

        # Invalid data_format
        with pytest.raises(ValueError, match="data_format must be 'channels_first' or 'channels_last'"):
            GaussianPyramid(data_format="invalid")

    # =====================================================================
    # Build Process Tests
    # =====================================================================

    def test_build_process_channels_last(self, input_tensor_channels_last):
        """Test that the layer builds properly with channels_last format."""
        layer = GaussianPyramid(levels=3, data_format="channels_last")
        layer(input_tensor_channels_last)  # Trigger build

        assert layer.built is True
        assert len(layer.gaussian_filters) == 3
        assert all(filter.built for filter in layer.gaussian_filters)
        assert layer._build_input_shape is not None

    def test_build_different_levels(self, input_tensor_channels_last):
        """Test building with different number of levels."""
        for levels in [1, 2, 4, 5]:
            layer = GaussianPyramid(levels=levels)
            layer(input_tensor_channels_last)

            assert len(layer.gaussian_filters) == levels
            assert all(filter.built for filter in layer.gaussian_filters)

    # =====================================================================
    # Output Shape Tests
    # =====================================================================

    def test_output_shapes_channels_last(self, input_tensor_channels_last):
        """Test output shapes with channels_last format."""
        layer = GaussianPyramid(levels=3, scale_factor=2, data_format="channels_last")
        outputs = layer(input_tensor_channels_last)

        # Check number of outputs
        assert len(outputs) == 3

        # Check shapes: (2, 32, 32, 3) -> (2, 16, 16, 3) -> (2, 8, 8, 3)
        expected_shapes = [(2, 32, 32, 3), (2, 16, 16, 3), (2, 8, 8, 3)]
        actual_shapes = [tuple(output.shape) for output in outputs]

        assert actual_shapes == expected_shapes

    def test_compute_output_shape(self):
        """Test compute_output_shape method."""
        layer = GaussianPyramid(levels=3, scale_factor=2, data_format="channels_last")

        input_shape = (None, 64, 64, 3)
        output_shapes = layer.compute_output_shape(input_shape)

        expected_shapes = [
            (None, 64, 64, 3),
            (None, 32, 32, 3),
            (None, 16, 16, 3)
        ]

        assert output_shapes == expected_shapes

    def test_different_scale_factors(self, input_tensor_channels_last):
        """Test with different scale factors."""
        # Scale factor 3
        layer = GaussianPyramid(levels=3, scale_factor=3)
        outputs = layer(input_tensor_channels_last)

        expected_shapes = [(2, 32, 32, 3), (2, 10, 10, 3), (2, 3, 3, 3)]
        actual_shapes = [tuple(output.shape) for output in outputs]

        assert actual_shapes == expected_shapes

    def test_scale_factor_one(self, input_tensor_channels_last):
        """Test with scale factor 1 (no downsampling)."""
        layer = GaussianPyramid(levels=3, scale_factor=1)
        outputs = layer(input_tensor_channels_last)

        # All outputs should have the same shape
        expected_shapes = [(2, 32, 32, 3)] * 3
        actual_shapes = [tuple(output.shape) for output in outputs]

        assert actual_shapes == expected_shapes

    # =====================================================================
    # Forward Pass Tests
    # =====================================================================

    def test_forward_pass_validity(self, input_tensor_channels_last):
        """Test that forward pass produces valid outputs."""
        layer = GaussianPyramid(levels=3)
        outputs = layer(input_tensor_channels_last)

        # Check that all outputs are valid tensors
        for output in outputs:
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))
            assert output.dtype == input_tensor_channels_last.dtype

    def test_forward_pass_deterministic(self):
        """Test that forward pass is deterministic."""
        layer = GaussianPyramid(levels=2, trainable=False)

        # Same input should produce same output
        x = tf.constant(np.random.rand(1, 16, 16, 1).astype(np.float32))

        output1 = layer(x)
        output2 = layer(x)

        for o1, o2 in zip(output1, output2):
            assert np.allclose(o1.numpy(), o2.numpy())

    def test_training_mode_parameter(self, input_tensor_channels_last):
        """Test that training parameter is handled correctly."""
        layer = GaussianPyramid(levels=2)

        # Should work in both training modes
        output_train = layer(input_tensor_channels_last, training=True)
        output_eval = layer(input_tensor_channels_last, training=False)

        assert len(output_train) == len(output_eval) == 2

    def test_single_level_pyramid(self, input_tensor_channels_last):
        """Test pyramid with only one level."""
        layer = GaussianPyramid(levels=1)
        outputs = layer(input_tensor_channels_last)

        assert len(outputs) == 1
        assert outputs[0].shape == input_tensor_channels_last.shape

    # =====================================================================
    # Serialization Tests
    # =====================================================================

    def test_serialization_basic(self):
        """Test basic serialization and deserialization."""
        original_layer = GaussianPyramid(
            levels=3,
            kernel_size=(7, 7),
            sigma=1.5,
            scale_factor=2,
            padding="valid",
            data_format="channels_last"
        )

        # Build the layer
        original_layer.build((None, 32, 32, 3))

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate layer
        recreated_layer = GaussianPyramid.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.levels == original_layer.levels
        assert recreated_layer.kernel_size == original_layer.kernel_size
        assert recreated_layer.sigma == original_layer.sigma
        assert recreated_layer.scale_factor == original_layer.scale_factor
        assert recreated_layer.padding == original_layer.padding
        assert recreated_layer.data_format == original_layer.data_format

    def test_serialization_with_outputs(self):
        """Test serialization with actual forward pass."""
        original_layer = GaussianPyramid(levels=2, sigma=1.0)

        # Test input
        x = tf.random.normal((1, 16, 16, 3))

        # Build and get output
        original_outputs = original_layer(x)

        # Serialize and recreate
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        recreated_layer = GaussianPyramid.from_config(config)
        recreated_layer.build_from_config(build_config)

        recreated_outputs = recreated_layer(x)

        # Check outputs match
        assert len(original_outputs) == len(recreated_outputs)
        for orig, recreated in zip(original_outputs, recreated_outputs):
            assert orig.shape == recreated.shape

    # =====================================================================
    # Model Integration Tests
    # =====================================================================

    def test_model_integration(self, input_tensor_channels_last):
        """Test the layer in a model context."""
        # Create model with GaussianPyramid
        inputs = keras.Input(shape=(32, 32, 3))
        pyramid_outputs = GaussianPyramid(levels=3)(inputs)

        # Use the largest scale output for classification
        x = pyramid_outputs[0]  # First level (original size)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        # Test forward pass
        y_pred = model(input_tensor_channels_last)
        assert y_pred.shape == (2, 10)

    def test_model_with_multiple_pyramid_outputs(self, input_tensor_channels_last):
        """Test model that uses multiple pyramid outputs."""
        inputs = keras.Input(shape=(32, 32, 3))
        pyramid_outputs = GaussianPyramid(levels=3)(inputs)

        # Process each scale differently
        features = []
        for i, output in enumerate(pyramid_outputs):
            x = keras.layers.GlobalAveragePooling2D()(output)
            x = keras.layers.Dense(16, activation='relu')(x)
            features.append(x)

        # Combine features
        combined = keras.layers.Concatenate()(features)
        final_output = keras.layers.Dense(10, activation='softmax')(combined)

        model = keras.Model(inputs=inputs, outputs=final_output)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        # Test forward pass
        y_pred = model(input_tensor_channels_last)
        assert y_pred.shape == (2, 10)

    # =====================================================================
    # Model Save/Load Tests
    # =====================================================================

    def test_model_save_load(self, input_tensor_channels_last):
        """Test saving and loading a model with GaussianPyramid."""
        # Create model
        inputs = keras.Input(shape=(32, 32, 3))
        pyramid_outputs = GaussianPyramid(levels=2, name="gaussian_pyramid")(inputs)
        x = keras.layers.GlobalAveragePooling2D()(pyramid_outputs[0])
        outputs = keras.layers.Dense(5)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate prediction
        original_prediction = model.predict(input_tensor_channels_last)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")

            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"GaussianPyramid": GaussianPyramid}
            )

            # Test loaded model
            loaded_prediction = loaded_model.predict(input_tensor_channels_last)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer type is preserved
            assert isinstance(loaded_model.get_layer("gaussian_pyramid"), GaussianPyramid)

    # =====================================================================
    # Edge Case Tests
    # =====================================================================

    def test_small_input_sizes(self):
        """Test with very small input sizes."""
        layer = GaussianPyramid(levels=3, scale_factor=2)

        # 4x4 input should still work
        x = tf.random.normal((1, 4, 4, 3))
        outputs = layer(x)

        assert len(outputs) == 3
        # Check that dimensions don't go below 1
        assert all(output.shape[1] >= 1 and output.shape[2] >= 1 for output in outputs)

    def test_large_scale_factor(self):
        """Test with large scale factor."""
        layer = GaussianPyramid(levels=2, scale_factor=10)

        x = tf.random.normal((1, 32, 32, 3))
        outputs = layer(x)

        assert len(outputs) == 2
        assert outputs[0].shape == (1, 32, 32, 3)
        assert outputs[1].shape == (1, 3, 3, 3)  # 32//10 = 3

    def test_different_kernel_sizes(self, input_tensor_channels_last):
        """Test with different kernel sizes."""
        kernel_sizes = [(3, 3), (7, 7), (9, 9)]

        for kernel_size in kernel_sizes:
            layer = GaussianPyramid(levels=2, kernel_size=kernel_size)
            outputs = layer(input_tensor_channels_last)

            assert len(outputs) == 2
            assert all(not np.any(np.isnan(output.numpy())) for output in outputs)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        layer = GaussianPyramid(levels=2)

        # Test with very small values
        x_small = tf.ones((1, 16, 16, 3)) * 1e-8
        outputs_small = layer(x_small)

        for output in outputs_small:
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))

        # Test with very large values
        x_large = tf.ones((1, 16, 16, 3)) * 1e8
        outputs_large = layer(x_large)

        for output in outputs_large:
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))

    # =====================================================================
    # Functional Interface Tests
    # =====================================================================

    def test_functional_interface(self, input_tensor_channels_last):
        """Test the functional interface gaussian_pyramid."""
        # Test functional interface
        outputs = gaussian_pyramid(
            input_tensor_channels_last,
            levels=3,
            kernel_size=(5, 5),
            sigma=1.0,
            scale_factor=2
        )

        assert len(outputs) == 3
        expected_shapes = [(2, 32, 32, 3), (2, 16, 16, 3), (2, 8, 8, 3)]
        actual_shapes = [tuple(output.shape) for output in outputs]

        assert actual_shapes == expected_shapes

    def test_functional_interface_equivalence(self, input_tensor_channels_last):
        """Test that functional interface produces same results as layer."""
        # Layer interface
        layer = GaussianPyramid(levels=2, sigma=1.0)
        layer_outputs = layer(input_tensor_channels_last)

        # Functional interface
        func_outputs = gaussian_pyramid(
            input_tensor_channels_last,
            levels=2,
            sigma=1.0
        )

        # Should produce same results
        assert len(layer_outputs) == len(func_outputs)
        for layer_out, func_out in zip(layer_outputs, func_outputs):
            assert layer_out.shape == func_out.shape

    # =====================================================================
    # Performance Tests
    # =====================================================================

    def test_memory_efficiency(self):
        """Test that the layer doesn't create unnecessary memory overhead."""
        import gc

        # Create layer and test input
        layer = GaussianPyramid(levels=3)
        x = tf.random.normal((1, 64, 64, 3))

        # Run forward pass
        outputs = layer(x)

        # Clean up
        del outputs
        gc.collect()

        # Should complete without memory issues
        assert True

    def test_repeated_calls(self, input_tensor_channels_last):
        """Test that repeated calls work correctly."""
        layer = GaussianPyramid(levels=2)

        # Multiple calls should work
        for _ in range(5):
            outputs = layer(input_tensor_channels_last)
            assert len(outputs) == 2