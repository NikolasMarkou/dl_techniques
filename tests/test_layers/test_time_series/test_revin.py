"""
Comprehensive Test Suite for RevIN Layer.

This module contains comprehensive tests for the RevIN (Reversible Instance Normalization)
layer implementation, covering initialization, functionality, serialization, and integration.
"""

import pytest
import numpy as np
import keras
import tensorflow as tf
import tempfile
import os
from typing import Tuple

from dl_techniques.layers.time_series.revin import RevIN


class TestRevIN:
    """Test suite for RevIN layer implementation."""

    @pytest.fixture
    def input_tensor_3d(self):
        """Create a 3D test input tensor (batch, seq_len, features)."""
        np.random.seed(42)
        return tf.constant(np.random.randn(4, 10, 5), dtype=tf.float32)

    @pytest.fixture
    def input_tensor_large(self):
        """Create a larger test input tensor for robustness testing."""
        np.random.seed(123)
        return tf.constant(np.random.randn(16, 50, 12), dtype=tf.float32)

    @pytest.fixture
    def layer_instance(self):
        """Create a default RevIN layer instance for testing."""
        return RevIN(num_features=5)

    @pytest.fixture
    def layer_instance_no_affine(self):
        """Create a RevIN layer instance without affine transformation."""
        return RevIN(num_features=5, affine=False)

    # ==================== Initialization Tests ====================

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = RevIN(num_features=10)

        # Check default values
        assert layer.num_features == 10
        assert layer.eps == 1e-5
        assert layer.affine is True
        assert isinstance(layer.affine_weight_initializer, keras.initializers.Ones)
        assert isinstance(layer.affine_bias_initializer, keras.initializers.Zeros)

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        layer = RevIN(
            num_features=8,
            eps=1e-6,
            affine=False,
            affine_weight_initializer="glorot_uniform",
            affine_bias_initializer="ones"
        )

        # Check custom values
        assert layer.num_features == 8
        assert layer.eps == 1e-6
        assert layer.affine is False
        assert isinstance(layer.affine_weight_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.affine_bias_initializer, keras.initializers.Ones)

    # ==================== Build Process Tests ====================

    def test_build_process_with_affine(self, input_tensor_3d):
        """Test that the layer builds properly with affine transformation."""
        layer = RevIN(num_features=5, affine=True)
        layer(input_tensor_3d)  # Trigger build

        # Check that weights were created
        assert layer.built is True
        assert len(layer.weights) == 2  # affine_weight and affine_bias
        assert hasattr(layer, "affine_weight")
        assert hasattr(layer, "affine_bias")
        assert layer.affine_weight.shape == (5,)
        assert layer.affine_bias.shape == (5,)

    def test_build_process_without_affine(self, input_tensor_3d):
        """Test that the layer builds properly without affine transformation."""
        layer = RevIN(num_features=5, affine=False)
        layer(input_tensor_3d)  # Trigger build

        # Check that no weights were created
        assert layer.built is True
        assert len(layer.weights) == 0
        assert layer.affine_weight is None
        assert layer.affine_bias is None

    # ==================== Output Shape Tests ====================

    def test_output_shapes(self, input_tensor_3d, input_tensor_large):
        """Test that output shapes are computed correctly."""
        test_cases = [
            (input_tensor_3d, 5),
            (input_tensor_large, 12)
        ]

        for tensor, num_features in test_cases:
            layer = RevIN(num_features=num_features)
            output = layer(tensor)

            # Check output shape matches input shape
            assert output.shape == tensor.shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(tensor.shape)
            assert computed_shape == tensor.shape

    # ==================== Forward Pass and Functionality Tests ====================

    def test_forward_pass_basic(self, input_tensor_3d):
        """Test basic forward pass functionality."""
        layer = RevIN(num_features=5)
        output = layer(input_tensor_3d)

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))
        assert output.shape == input_tensor_3d.shape

    def test_statistics_computation(self, input_tensor_3d):
        """Test that statistics are computed correctly."""
        layer = RevIN(num_features=5)

        # Before calling the layer, statistics should be None
        assert layer._mean is None
        assert layer._stdev is None

        # After calling the layer, statistics should be computed
        _ = layer(input_tensor_3d)
        assert layer._mean is not None
        assert layer._stdev is not None

        # Check statistics shapes
        assert layer._mean.shape == (4, 1, 5)  # (batch, 1, features)
        assert layer._stdev.shape == (4, 1, 5)

    def test_normalization_properties(self):
        """Test that normalization produces expected statistical properties."""
        # Create input with known statistics
        np.random.seed(42)
        input_data = tf.constant(np.random.randn(8, 20, 3) * 5 + 10, dtype=tf.float32)

        layer = RevIN(num_features=3, affine=False)  # No affine for cleaner test
        normalized = layer(input_data)

        # Check that normalized data has approximately zero mean and unit variance
        # along the sequence dimension for each batch and feature
        for batch_idx in range(8):
            for feature_idx in range(3):
                sequence = normalized[batch_idx, :, feature_idx]
                mean_val = tf.reduce_mean(sequence)
                var_val = tf.reduce_mean(tf.square(sequence - mean_val))

                assert abs(mean_val.numpy()) < 1e-5  # Very close to 0
                assert abs(var_val.numpy() - 1.0) < 1e-5  # Very close to 1

    def test_denormalization_without_normalization(self):
        """Test that denormalization without prior normalization raises error."""
        layer = RevIN(num_features=5)
        dummy_input = tf.random.normal((4, 10, 5))

        with pytest.raises(ValueError, match="Cannot denormalize: statistics not computed"):
            layer.denormalize(dummy_input)

    def test_reversibility(self, input_tensor_3d):
        """Test that normalization followed by denormalization is reversible."""
        layer = RevIN(num_features=5)

        # Apply normalization
        normalized = layer(input_tensor_3d)

        # Apply denormalization
        denormalized = layer.denormalize(normalized)

        # Check that denormalized data matches original input
        np.testing.assert_allclose(
            input_tensor_3d.numpy(),
            denormalized.numpy(),
            rtol=1e-5,
            atol=1e-6
        )

    def test_affine_transformation(self):
        """Test that affine transformation is applied correctly."""
        # Create controlled input
        input_data = tf.ones((2, 5, 3))  # All ones

        # Create layer with specific affine parameters
        layer = RevIN(
            num_features=3,
            affine_weight_initializer=keras.initializers.Constant(2.0),
            affine_bias_initializer=keras.initializers.Constant(1.0)
        )

        normalized = layer(input_data)

        # With all ones input, after normalization (subtract mean, divide by std)
        # we should get zeros, then affine transform: 0 * 2 + 1 = 1
        expected = tf.ones_like(input_data)
        np.testing.assert_allclose(normalized.numpy(), expected.numpy(), rtol=1e-5)

    def test_different_batch_sizes(self):
        """Test layer works with different batch sizes."""
        layer = RevIN(num_features=4)

        batch_sizes = [1, 3, 8, 16]
        seq_len, features = 15, 4

        for batch_size in batch_sizes:
            input_tensor = tf.random.normal((batch_size, seq_len, features))
            output = layer(input_tensor)

            assert output.shape == (batch_size, seq_len, features)
            assert not np.any(np.isnan(output.numpy()))

    # ==================== Serialization Tests ====================

    def test_serialization_with_affine(self):
        """Test serialization and deserialization with affine transformation."""
        original_layer = RevIN(
            num_features=6,
            eps=1e-6,
            affine=True,
            affine_weight_initializer="he_normal"
        )

        # Build the layer
        input_shape = (None, 12, 6)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = RevIN.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.num_features == original_layer.num_features
        assert recreated_layer.eps == original_layer.eps
        assert recreated_layer.affine == original_layer.affine

    def test_serialization_without_affine(self):
        """Test serialization and deserialization without affine transformation."""
        original_layer = RevIN(num_features=4, affine=False, eps=1e-7)

        # Build the layer
        input_shape = (None, 20, 4)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = RevIN.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.num_features == original_layer.num_features
        assert recreated_layer.eps == original_layer.eps
        assert recreated_layer.affine == original_layer.affine
        assert recreated_layer.affine_weight is None
        assert recreated_layer.affine_bias is None

    # ==================== Model Integration Tests ====================

    def test_model_integration(self, input_tensor_3d):
        """Test the layer in a model context."""
        # Create a simple model with RevIN layer
        inputs = keras.Input(shape=(10, 5))
        x = RevIN(num_features=5)(inputs)
        x = keras.layers.LSTM(16, return_sequences=True)(x)
        x = keras.layers.Dense(5)(x)
        outputs = x

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(optimizer="adam", loss="mse")

        # Test forward pass
        y_pred = model(input_tensor_3d, training=False)
        assert y_pred.shape == (4, 10, 5)
        assert not np.any(np.isnan(y_pred.numpy()))

    def test_forecasting_workflow(self):
        """Test a complete forecasting workflow with RevIN."""
        # Create synthetic time series data
        np.random.seed(42)
        batch_size, seq_len, features = 8, 30, 4

        # Create data with trend and seasonality
        t = np.linspace(0, 10, seq_len)
        x_data = np.zeros((batch_size, seq_len, features))

        for b in range(batch_size):
            for f in range(features):
                trend = 0.1 * t + np.random.randn() * 0.1
                seasonal = np.sin(2 * np.pi * t / 5) * (0.5 + np.random.randn() * 0.1)
                noise = np.random.randn(seq_len) * 0.1
                x_data[b, :, f] = trend + seasonal + noise

        x_data = tf.constant(x_data, dtype=tf.float32)

        # Create RevIN layer
        revin = RevIN(num_features=features)

        # Normalize input
        x_norm = revin(x_data)

        # Pass through simple forecasting model
        model = keras.Sequential([
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.Dense(features)
        ])
        predictions_norm = model(x_norm)

        # Denormalize predictions
        predictions = revin.denormalize(predictions_norm)

        # Check that predictions have similar scale to input
        input_scale = tf.math.reduce_std(x_data)
        pred_scale = tf.math.reduce_std(predictions)

        # Scales should be similar (within an order of magnitude)
        assert 0.1 < (pred_scale / input_scale) < 10.0

    # ==================== Model Save/Load Tests ====================

    def test_model_save_load(self, input_tensor_3d):
        """Test saving and loading a model with RevIN layer."""
        # Create a model with RevIN layer
        inputs = keras.Input(shape=(10, 5))
        x = RevIN(num_features=5, name="revin_layer")(inputs)
        x = keras.layers.Dense(8, activation="relu")(x)
        x = keras.layers.Dense(5)(x)
        outputs = x

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate a prediction before saving
        original_prediction = model.predict(input_tensor_3d, verbose=0)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"RevIN": RevIN}
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor_3d, verbose=0)

            # Check predictions match
            np.testing.assert_allclose(
                original_prediction,
                loaded_prediction,
                rtol=1e-5
            )

            # Check layer types are preserved
            assert isinstance(loaded_model.get_layer("revin_layer"), RevIN)

    # ==================== Edge Case and Robustness Tests ====================

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = RevIN(num_features=3)

        # Test cases with different magnitudes
        test_cases = [
            tf.zeros((2, 5, 3)),  # All zeros
            tf.ones((2, 5, 3)) * 1e-10,  # Very small values
            tf.ones((2, 5, 3)) * 1e10,   # Very large values
            tf.random.normal((2, 5, 3)) * 1e5  # Large random values
        ]

        for test_input in test_cases:
            try:
                output = layer(test_input)

                # Check for NaN/Inf values
                assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
                assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

                # Test denormalization
                denorm = layer.denormalize(output)
                assert not np.any(np.isnan(denorm.numpy())), "NaN values in denormalization"
                assert not np.any(np.isinf(denorm.numpy())), "Inf values in denormalization"

            except Exception as e:
                pytest.fail(f"Layer failed with input of scale {tf.reduce_max(tf.abs(test_input))}: {e}")

    def test_constant_input(self):
        """Test behavior with constant input values."""
        # All values are the same - this creates zero variance
        constant_input = tf.ones((3, 8, 4)) * 5.0
        layer = RevIN(num_features=4, affine=False)

        # This should not crash due to division by zero (eps should prevent it)
        output = layer(constant_input)

        # Output should be finite
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

        # Test denormalization
        denorm = layer.denormalize(output)

        # Should approximately reconstruct the original constant
        np.testing.assert_allclose(
            constant_input.numpy(),
            denorm.numpy(),
            rtol=1e-3  # Allow some tolerance due to eps
        )

    def test_single_timestep(self):
        """Test layer with single timestep input."""
        single_step_input = tf.random.normal((4, 1, 6))
        layer = RevIN(num_features=6)

        # This is an edge case - statistics computed over single timestep
        output = layer(single_step_input)

        assert output.shape == single_step_input.shape
        assert not np.any(np.isnan(output.numpy()))

        # Test reversibility
        denorm = layer.denormalize(output)
        np.testing.assert_allclose(
            single_step_input.numpy(),
            denorm.numpy(),
            rtol=1e-5
        )

    def test_different_eps_values(self):
        """Test layer with different epsilon values."""
        input_data = tf.random.normal((2, 10, 3))

        eps_values = [1e-3, 1e-5, 1e-7, 1e-9]

        for eps in eps_values:
            layer = RevIN(num_features=3, eps=eps, affine=False)

            normalized = layer(input_data)
            denormalized = layer.denormalize(normalized)

            # Should be able to reconstruct original
            np.testing.assert_allclose(
                input_data.numpy(),
                denormalized.numpy(),
                rtol=1e-4,
                atol=1e-6
            )

    # ==================== Performance and Memory Tests ====================

    def test_memory_efficiency(self):
        """Test that layer doesn't consume excessive memory."""
        # Create a reasonably large input
        large_input = tf.random.normal((32, 100, 64))
        layer = RevIN(num_features=64)

        # This should complete without memory issues
        output = layer(large_input)
        denorm = layer.denormalize(output)

        assert output.shape == large_input.shape
        assert denorm.shape == large_input.shape

    def test_multiple_calls_consistency(self):
        """Test that multiple calls with same input produce consistent results."""
        input_data = tf.random.normal((4, 15, 5))
        layer = RevIN(num_features=5)

        # First call
        output1 = layer(input_data)
        denorm1 = layer.denormalize(output1)

        # Second call with same input
        output2 = layer(input_data)  # This will recompute statistics
        denorm2 = layer.denormalize(output2)

        # Outputs should be identical
        np.testing.assert_array_equal(output1.numpy(), output2.numpy())
        np.testing.assert_array_equal(denorm1.numpy(), denorm2.numpy())


# ==================== Standalone Test Functions ====================

def test_revin_registration():
    """Test that RevIN is properly registered for serialization."""
    # This test ensures the @keras.saving.register_keras_serializable() decorator works
    layer = RevIN(num_features=5)
    config = layer.get_config()

    # Should be able to recreate from config
    new_layer = RevIN.from_config(config)
    assert new_layer.num_features == layer.num_features


def test_revin_with_different_dtypes():
    """Test RevIN with different input dtypes."""
    # Test with float32
    layer_f32 = RevIN(num_features=3)
    input_f32 = tf.random.normal((2, 8, 3), dtype=tf.float32)
    output_f32 = layer_f32(input_f32)
    assert output_f32.dtype == tf.float32

    # Test with float64 - just ensure it works
    layer_f64 = RevIN(num_features=3)
    input_f64 = tf.cast(tf.random.normal((2, 8, 3)), tf.float64)
    # Just ensure it runs without error and produces valid output
    output_f64 = layer_f64(input_f64)
    assert not np.any(np.isnan(output_f64.numpy()))
    assert not np.any(np.isinf(output_f64.numpy()))


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__])