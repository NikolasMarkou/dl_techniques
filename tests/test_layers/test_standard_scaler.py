import pytest
import numpy as np
import keras
import os
import tempfile

from dl_techniques.layers.standard_scaler import StandardScaler


class TestStandardScaler:
    """Test suite for StandardScaler implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        # Shape: (batch_size, sequence_length, features)
        # Using realistic time series data with varying scales
        np.random.seed(42)
        batch_size, seq_length, features = 4, 100, 10

        # Create data with different scales per feature
        data = np.random.randn(batch_size, seq_length, features)
        # Scale different features differently
        scales = np.array([0.1, 1.0, 10.0, 100.0, 0.01, 5.0, 50.0, 2.0, 0.5, 20.0])
        data = data * scales.reshape(1, 1, -1)

        return keras.ops.convert_to_tensor(data.astype(np.float32))

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return StandardScaler()

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = StandardScaler()

        # Check default values
        assert layer.epsilon == 1e-5
        assert layer.nan_replacement == 0.0
        assert layer.store_stats is False
        assert layer.axis == -1
        assert layer.stored_mean is None
        assert layer.stored_std is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        layer = StandardScaler(
            epsilon=1e-6,
            nan_replacement=-1.0,
            store_stats=True,
            axis=1,
        )

        # Check custom values
        assert layer.epsilon == 1e-6
        assert layer.nan_replacement == -1.0
        assert layer.store_stats is True
        assert layer.axis == 1

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid epsilon
        with pytest.raises(ValueError, match="epsilon must be positive"):
            StandardScaler(epsilon=0)

        with pytest.raises(ValueError, match="epsilon must be positive"):
            StandardScaler(epsilon=-1e-5)

    def test_build_process_no_stats(self, input_tensor):
        """Test that the layer builds properly without stats storage."""
        layer = StandardScaler(store_stats=False)

        # Trigger build through forward pass
        output = layer(input_tensor)

        # Check that the layer was built
        assert layer.built is True
        assert layer.stored_mean is None
        assert layer.stored_std is None
        assert layer._built_input_shape == input_tensor.shape

    def test_build_process_with_stats(self, input_tensor):
        """Test that the layer builds properly with stats storage."""
        layer = StandardScaler(store_stats=True)

        # Trigger build through forward pass
        output = layer(input_tensor)

        # Check that the layer was built with stats storage
        assert layer.built is True
        assert layer.stored_mean is not None
        assert layer.stored_std is not None
        assert len(layer.weights) == 2  # stored_mean and stored_std
        assert layer._built_input_shape == input_tensor.shape

        # Check stats shapes
        expected_stats_shape = list(input_tensor.shape[1:])  # Remove batch dimension
        expected_stats_shape[-1] = 1  # Stats along last axis
        assert layer.stored_mean.shape == tuple(expected_stats_shape)
        assert layer.stored_std.shape == tuple(expected_stats_shape)

    def test_build_different_axis(self):
        """Test build with different normalization axes."""
        # Test with axis=1 (normalize along sequence dimension)
        layer = StandardScaler(store_stats=True, axis=1)
        input_shape = (None, 50, 10)
        layer.build(input_shape)

        expected_stats_shape = (1, 10)  # axis=1 becomes 1, others preserved
        assert layer.stored_mean.shape == expected_stats_shape
        assert layer.stored_std.shape == expected_stats_shape

    def test_output_shapes(self, input_tensor):
        """Test that output shapes are computed correctly."""
        configs_to_test = [
            {"axis": -1},
            {"axis": 1},
            {"store_stats": True},
            {"store_stats": False},
        ]

        for config in configs_to_test:
            layer = StandardScaler(**config)
            output = layer(input_tensor)

            # Check output shape matches input shape
            assert output.shape == input_tensor.shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(input_tensor.shape)
            assert computed_shape == input_tensor.shape

    def test_forward_pass_basic(self, input_tensor, layer_instance):
        """Test that forward pass produces expected values."""
        output = layer_instance(input_tensor)

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))
        assert output.shape == input_tensor.shape

        # Check that output is approximately normalized (mean≈0, std≈1)
        output_np = output.numpy()
        output_mean = np.mean(output_np, axis=-1, keepdims=True)
        output_std = np.std(output_np, axis=-1, keepdims=True)

        # Should be close to 0 mean and 1 std along the normalized axis
        assert np.allclose(output_mean, 0, atol=1e-6)
        assert np.allclose(output_std, 1, atol=1e-6)

    def test_forward_pass_with_training_modes(self, input_tensor):
        """Test forward pass with different training modes."""
        layer = StandardScaler(store_stats=True)

        # Test training mode
        training_output = layer(input_tensor, training=True)

        # Test inference mode
        inference_output = layer(input_tensor, training=False)

        # Both should produce valid outputs
        assert not np.any(np.isnan(training_output.numpy()))
        assert not np.any(np.isnan(inference_output.numpy()))
        assert training_output.shape == input_tensor.shape
        assert inference_output.shape == input_tensor.shape

    def test_nan_handling(self):
        """Test that NaN values are handled correctly."""
        # Create input with NaN values
        input_data = np.random.randn(2, 10, 5).astype(np.float32)
        input_data[0, 5, 2] = np.nan  # Insert NaN values
        input_data[1, 3, 0] = np.nan
        input_tensor = keras.ops.convert_to_tensor(input_data)

        layer = StandardScaler(nan_replacement=-999.0)
        output = layer(input_tensor)

        # Check that NaN values are handled
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_different_axis_normalization(self):
        """Test normalization along different axes."""
        input_data = keras.random.normal([2, 20, 5])

        # Test axis=-1 (feature-wise normalization)
        layer_axis_neg1 = StandardScaler(axis=-1)
        output_neg1 = layer_axis_neg1(input_data)

        # Check normalization along last axis
        output_mean = np.mean(output_neg1.numpy(), axis=-1, keepdims=True)
        output_std = np.std(output_neg1.numpy(), axis=-1, keepdims=True)
        assert np.allclose(output_mean, 0, atol=1e-6)
        # Use more lenient tolerance for std due to numerical precision
        assert np.allclose(output_std, 1, atol=1e-4)

        # Test axis=1 (sequence-wise normalization)
        layer_axis_1 = StandardScaler(axis=1)
        output_1 = layer_axis_1(input_data)

        # Check normalization along axis 1
        output_mean = np.mean(output_1.numpy(), axis=1, keepdims=True)
        output_std = np.std(output_1.numpy(), axis=1, keepdims=True)
        assert np.allclose(output_mean, 0, atol=1e-6)
        # Use more lenient tolerance for std due to numerical precision
        assert np.allclose(output_std, 1, atol=1e-4)

    def test_inverse_transform_functionality(self, input_tensor):
        """Test inverse transform functionality."""
        # Test with any layer (no longer requires store_stats=True)
        layer = StandardScaler()

        # Forward transform
        normalized = layer(input_tensor)

        # Inverse transform
        reconstructed = layer.inverse_transform(normalized)

        # Should approximately recover original data
        # Use more realistic tolerance for floating point precision
        assert np.allclose(input_tensor.numpy(), reconstructed.numpy(), rtol=1e-4, atol=1e-5)

    def test_inverse_transform_with_stats_storage(self, input_tensor):
        """Test inverse transform with persistent stats storage."""
        # Test with store_stats=True as well
        layer = StandardScaler(store_stats=True)

        # Forward transform
        normalized = layer(input_tensor)

        # Inverse transform
        reconstructed = layer.inverse_transform(normalized)

        # Should approximately recover original data
        # Use more realistic tolerance for floating point precision
        assert np.allclose(input_tensor.numpy(), reconstructed.numpy(), rtol=1e-4, atol=1e-5)

    def test_inverse_transform_before_call(self):
        """Test that inverse transform fails before calling the layer."""
        layer = StandardScaler()
        dummy_input = keras.random.normal([2, 10, 5])

        with pytest.raises(RuntimeError, match="Layer must be called at least once"):
            layer.inverse_transform(dummy_input)

    def test_stats_management(self, input_tensor):
        """Test statistics storage and management."""
        layer = StandardScaler(store_stats=True)

        # Initially no stats
        assert layer.get_stats() is None

        # After calling, stats should be available
        layer(input_tensor)
        stats = layer.get_stats()
        assert stats is not None
        mean, std = stats
        assert mean.shape == (input_tensor.shape[1], 1)
        assert std.shape == (input_tensor.shape[1], 1)

        # Reset stats
        layer.reset_stats()
        stats = layer.get_stats()
        mean, std = stats
        assert np.allclose(mean.numpy(), 0)
        assert np.allclose(std.numpy(), 1)

    def test_stats_management_without_storage(self):
        """Test stats management methods when store_stats=False."""
        layer = StandardScaler(store_stats=False)

        # Should return None
        assert layer.get_stats() is None

        # Reset should warn but not fail
        layer.reset_stats()  # Should log warning but not crash

    def test_zero_variance_handling(self):
        """Test handling of zero variance (constant) features."""
        # Create input with constant features
        input_data = np.ones((2, 10, 3), dtype=np.float32)
        input_data[:, :, 0] = 5.0  # Constant feature
        input_data[:, :, 1] = np.random.randn(2, 10)  # Variable feature
        input_data[:, :, 2] = -2.0  # Another constant feature

        input_tensor = keras.ops.convert_to_tensor(input_data)

        layer = StandardScaler(epsilon=1e-5)
        output = layer(input_tensor)

        # Should handle zero variance gracefully
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = StandardScaler(
            epsilon=1e-6,
            nan_replacement=-1.0,
            store_stats=True,
            axis=1,
        )

        # Build the layer
        input_shape = (None, 20, 10)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = StandardScaler.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.epsilon == original_layer.epsilon
        assert recreated_layer.nan_replacement == original_layer.nan_replacement
        assert recreated_layer.store_stats == original_layer.store_stats
        assert recreated_layer.axis == original_layer.axis

        # Check weights match (shapes should be the same)
        if original_layer.store_stats:
            assert len(recreated_layer.weights) == len(original_layer.weights)
            for w1, w2 in zip(original_layer.weights, recreated_layer.weights):
                assert w1.shape == w2.shape

    def test_model_integration(self, input_tensor):
        """Test the layer in a model context."""
        # Create a simple model with the custom layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = StandardScaler(store_stats=True)(inputs)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
        )

        # Test forward pass
        y_pred = model(input_tensor, training=False)
        assert y_pred.shape == (input_tensor.shape[0], 10)

    def test_model_save_load(self, input_tensor):
        """Test saving and loading a model with the custom layer."""
        # Create a model with the custom layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = StandardScaler(store_stats=True, name="standard_scaler")(inputs)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate a prediction before saving
        original_prediction = model.predict(input_tensor, verbose=0)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"StandardScaler": StandardScaler}
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer type is preserved
            assert isinstance(loaded_model.get_layer("standard_scaler"), StandardScaler)

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = StandardScaler(epsilon=1e-8)

        # Create inputs with different magnitudes
        batch_size, seq_len, features = 2, 20, 5

        test_cases = [
            keras.ops.zeros((batch_size, seq_len, features)),  # Zeros
            keras.ops.ones((batch_size, seq_len, features)) * 1e-15,  # Very small values
            keras.ops.ones((batch_size, seq_len, features)) * 1e10,  # Very large values
            keras.random.normal((batch_size, seq_len, features)) * 1000,  # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

    def test_batch_consistency(self):
        """Test that normalization is consistent across different batch sizes."""
        # Create deterministic input
        np.random.seed(42)
        data = np.random.randn(10, 20, 5).astype(np.float32)

        layer = StandardScaler()

        # Test with different batch sizes of the same data
        batch_2 = keras.ops.convert_to_tensor(data[:2])
        batch_5 = keras.ops.convert_to_tensor(data[:5])

        output_2 = layer(batch_2)
        output_5 = layer(batch_5)

        # First 2 samples should be identical
        assert np.allclose(output_2.numpy(), output_5.numpy()[:2], rtol=1e-6)

    def test_different_input_shapes(self):
        """Test layer with different input shapes."""
        layer = StandardScaler()

        # Test different shapes
        input_shapes = [
            (2, 10),  # 2D input
            (2, 10, 5),  # 3D input
            (2, 10, 5, 3),  # 4D input
        ]

        for shape in input_shapes:
            test_input = keras.random.normal(shape)
            output = layer(test_input)

            # Check output shape matches input shape
            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))

    def test_multiple_calls_with_stats(self, input_tensor):
        """Test multiple calls with statistics storage."""
        layer = StandardScaler(store_stats=True)

        # First call
        output1 = layer(input_tensor)
        stats1 = layer.get_stats()

        # Copy the stats to avoid reference issues
        mean1, std1 = stats1
        mean1_copy = keras.ops.convert_to_tensor(mean1.numpy().copy())
        std1_copy = keras.ops.convert_to_tensor(std1.numpy().copy())

        # Second call with significantly different data
        input2 = keras.random.normal(input_tensor.shape) * 10 + 100  # Very different distribution
        output2 = layer(input2)
        stats2 = layer.get_stats()

        # Stats should be updated (different from first call)
        mean2, std2 = stats2

        # Stats should be different with such different input distributions
        # Use a more lenient tolerance since we're comparing stored (averaged) stats
        assert not np.allclose(mean1_copy.numpy(), mean2.numpy(), rtol=0.1)

    def test_edge_case_single_sample(self):
        """Test with single sample input."""
        # Single sample case
        single_sample = keras.random.normal([1, 10, 5])
        layer = StandardScaler()

        output = layer(single_sample)

        # Should handle single sample gracefully
        assert output.shape == single_sample.shape
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_deterministic_behavior(self):
        """Test that the layer produces deterministic results."""
        layer = StandardScaler()
        test_input = keras.random.normal([2, 10, 5])

        # Multiple calls should produce identical results
        output1 = layer(test_input)
        output2 = layer(test_input)

        assert np.allclose(output1.numpy(), output2.numpy(), rtol=1e-10)