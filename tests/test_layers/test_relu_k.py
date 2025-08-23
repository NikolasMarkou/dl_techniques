"""Comprehensive test suite for ReLUK activation layer.

This module contains tests for the ReLUK layer implementation,
following the project's testing standards and patterns.
"""

import pytest
import numpy as np
import keras
import tempfile
import os
import tensorflow as tf

from dl_techniques.layers.activations.relu_k import ReLUK


class TestReLUK:
    """Test suite for ReLUK activation layer implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return keras.random.normal([4, 32, 64])  # 4 batches, 32 features, 64 dims

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return ReLUK(k=3)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = ReLUK()

        # Check default values
        assert layer.k == 3
        assert layer.built is False

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        layer = ReLUK(
            k=5,
            name="custom_reluk",
            dtype="float32",
            trainable=True
        )

        # Check custom values
        assert layer.k == 5
        assert layer.name == "custom_reluk"
        assert layer.dtype == "float32"
        assert layer.trainable is True

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test negative k
        with pytest.raises(ValueError, match="k must be a positive integer"):
            ReLUK(k=-1)

        # Test zero k
        with pytest.raises(ValueError, match="k must be a positive integer"):
            ReLUK(k=0)

        # Test non-integer k
        with pytest.raises(TypeError, match="k must be an integer"):
            ReLUK(k=2.5)

        with pytest.raises(TypeError, match="k must be an integer"):
            ReLUK(k="3")

        with pytest.raises(TypeError, match="k must be an integer"):
            ReLUK(k=[3])

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer builds properly."""
        # Trigger build through forward pass
        output = layer_instance(input_tensor)

        # Check that layer was built
        assert layer_instance.built is True

        # For activation layers, there are typically no trainable weights
        assert len(layer_instance.trainable_weights) == 0
        assert len(layer_instance.non_trainable_weights) == 0

    def test_output_shapes(self, input_tensor):
        """Test that output shapes are computed correctly."""
        k_values_to_test = [1, 2, 3, 5, 10]

        for k in k_values_to_test:
            layer = ReLUK(k=k)

            output = layer(input_tensor)

            # Check output shape matches input shape
            assert output.shape == input_tensor.shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(input_tensor.shape)
            assert computed_shape == input_tensor.shape

    def test_forward_pass(self, input_tensor, layer_instance):
        """Test that forward pass produces expected values."""
        output = layer_instance(input_tensor)

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

        # Check output shape
        assert output.shape == input_tensor.shape

        # Test with training=False
        output_inference = layer_instance(input_tensor, training=False)
        assert output_inference.shape == input_tensor.shape

        # Test with training=True
        output_training = layer_instance(input_tensor, training=True)
        assert output_training.shape == input_tensor.shape

        # Training and inference should produce same results for activation layers
        assert np.allclose(output_inference.numpy(), output_training.numpy())

    def test_different_configurations(self):
        """Test layer with different k values."""
        configurations = [
            {"k": 1},
            {"k": 2},
            {"k": 3},
            {"k": 5},
            {"k": 10},
        ]

        for config in configurations:
            layer = ReLUK(**config)

            # Create test input
            test_input = keras.random.normal([2, 16, 32])

            # Test forward pass
            output = layer(test_input)

            # Check output is valid
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))
            assert output.shape == test_input.shape

    def test_mathematical_correctness(self):
        """Test the mathematical correctness of the ReLU-k function."""
        # Test with known values for different k values
        test_cases = [
            {
                "k": 1,
                "input": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                "expected": [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
            },
            {
                "k": 2,
                "input": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                "expected": [0.0, 0.0, 0.0, 1.0, 4.0, 9.0]
            },
            {
                "k": 3,
                "input": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                "expected": [0.0, 0.0, 0.0, 1.0, 8.0, 27.0]
            }
        ]

        for case in test_cases:
            layer = ReLUK(k=case["k"])
            test_input = keras.ops.convert_to_tensor([case["input"]])
            expected = keras.ops.convert_to_tensor([case["expected"]])

            output = layer(test_input)

            # Check values are close (accounting for floating point precision)
            assert np.allclose(output.numpy(), expected.numpy(), rtol=1e-6)

    def test_negative_inputs_handling(self):
        """Test that negative inputs are properly zeroed."""
        layer = ReLUK(k=3)

        # All negative inputs should result in zeros
        negative_input = keras.ops.convert_to_tensor([
            [-10.0, -5.0, -2.0, -1.0, -0.1, -0.001]
        ])
        output = layer(negative_input)
        expected = keras.ops.convert_to_tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])

        assert np.allclose(output.numpy(), expected.numpy(), atol=1e-7)

    def test_positive_inputs_handling(self):
        """Test that positive inputs are properly transformed."""
        layer = ReLUK(k=2)

        # Test with positive inputs
        positive_input = keras.ops.convert_to_tensor([
            [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
        ])
        output = layer(positive_input)
        expected = keras.ops.convert_to_tensor([
            [0.01, 0.25, 1.0, 4.0, 9.0, 25.0]  # Each value squared
        ])

        assert np.allclose(output.numpy(), expected.numpy(), rtol=1e-6)

    def test_mixed_inputs(self):
        """Test with mixed positive and negative inputs."""
        layer = ReLUK(k=2)

        mixed_input = keras.ops.convert_to_tensor([
            [-3.0, -1.0, 0.0, 1.0, 2.0, 4.0]
        ])
        output = layer(mixed_input)
        expected = keras.ops.convert_to_tensor([
            [0.0, 0.0, 0.0, 1.0, 4.0, 16.0]
        ])

        assert np.allclose(output.numpy(), expected.numpy(), rtol=1e-6)

    def test_k_equals_one_optimization(self):
        """Test that k=1 case is equivalent to standard ReLU."""
        layer_k1 = ReLUK(k=1)
        relu_layer = keras.layers.ReLU()

        test_input = keras.random.normal([8, 16])

        # Apply both layers
        output_k1 = layer_k1(test_input)
        output_relu = relu_layer(test_input)

        # Results should be identical
        assert np.allclose(output_k1.numpy(), output_relu.numpy(), rtol=1e-6)

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = ReLUK(
            k=5,
            name="test_reluk"
        )

        # Build the layer
        input_shape = (None, 32, 64)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = ReLUK.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.k == original_layer.k
        assert recreated_layer.name == original_layer.name

        # Test that both layers produce same output
        test_input = keras.random.normal([2, 32, 64])
        original_output = original_layer(test_input)
        recreated_output = recreated_layer(test_input)

        assert np.allclose(original_output.numpy(), recreated_output.numpy())

    def test_model_integration(self, input_tensor):
        """Test the layer in a model context."""
        # Create a simple model with the ReLU-k layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = keras.layers.Dense(64)(inputs)
        x = ReLUK(k=2)(x)
        x = keras.layers.Dense(32)(x)
        x = ReLUK(k=3)(x)
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

        # Test that output is valid
        assert not np.any(np.isnan(y_pred.numpy()))
        assert not np.any(np.isinf(y_pred.numpy()))

    def test_model_save_load(self, input_tensor):
        """Test saving and loading a model with the ReLU-k layer."""
        # Create a model with the ReLU-k layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = keras.layers.Dense(32)(inputs)
        x = ReLUK(k=2, name="reluk_hidden")(x)
        x = keras.layers.Dense(16)(x)
        x = ReLUK(k=3, name="reluk_output")(x)
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
                custom_objects={"ReLUK": ReLUK}
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer types are preserved
            assert isinstance(loaded_model.get_layer("reluk_hidden"), ReLUK)
            assert isinstance(loaded_model.get_layer("reluk_output"), ReLUK)

            # Check layer parameters are preserved
            assert loaded_model.get_layer("reluk_hidden").k == 2
            assert loaded_model.get_layer("reluk_output").k == 3

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = ReLUK(k=2)

        # Create inputs with different magnitudes
        batch_size = 2
        feature_dim = 16

        test_cases = [
            keras.ops.zeros((batch_size, feature_dim)),  # Zeros
            keras.ops.ones((batch_size, feature_dim)) * 1e-10,  # Very small values
            keras.ops.ones((batch_size, feature_dim)) * 1e5,  # Large values
            keras.random.normal((batch_size, feature_dim)) * 100  # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

            # Check that output shape is preserved
            assert output.shape == test_input.shape

    def test_gradient_flow(self):
        """Test that gradients flow properly through the layer."""
        layer = ReLUK(k=2)

        # Create variable input
        test_input = tf.Variable(keras.random.normal([4, 8]))

        # Compute gradients
        with tf.GradientTape() as tape:
            output = layer(test_input)
            loss = keras.ops.mean(keras.ops.square(output))

        gradients = tape.gradient(loss, test_input)

        # Check gradients exist and are not None
        assert gradients is not None

        # Check gradients have proper shape
        assert gradients.shape == test_input.shape

        # For positive inputs, gradients should be non-zero (in most cases)
        # For negative inputs, gradients should be zero
        positive_mask = test_input > 0
        negative_mask = test_input <= 0

        # Get gradients for positive and negative regions
        positive_grads = tf.where(positive_mask, gradients, 0.0)
        negative_grads = tf.where(negative_mask, gradients, 0.0)

        # Gradients for negative inputs should be zero (dead neurons)
        assert np.allclose(negative_grads.numpy(), 0.0, atol=1e-7)

    def test_different_input_shapes(self):
        """Test layer with different input shapes."""
        layer = ReLUK(k=3)

        input_shapes = [
            (8, 10),           # 2D
            (4, 16, 32),       # 3D
            (2, 8, 8, 3),      # 4D (image-like)
            (1, 4, 6, 8, 2),   # 5D
        ]

        for shape in input_shapes:
            test_input = keras.random.normal(shape)
            output = layer(test_input)

            # Check output shape matches input
            assert output.shape == test_input.shape

            # Check output is valid
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))

    def test_training_compatibility(self):
        """Test that the layer works correctly during training."""
        # Create a simple model
        model = keras.Sequential([
            keras.layers.Dense(16, input_shape=(8,)),
            ReLUK(k=2),
            keras.layers.Dense(8),
            ReLUK(k=3),
            keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        # Create synthetic training data
        x_train = keras.random.normal([32, 8])
        y_train = keras.random.normal([32, 1])

        # Initial loss
        initial_loss = model.evaluate(x_train, y_train, verbose=0)

        # Train for a few epochs
        model.fit(x_train, y_train, epochs=3, batch_size=8, verbose=0)

        # Final loss
        final_loss = model.evaluate(x_train, y_train, verbose=0)

        # Training should complete without errors
        assert isinstance(final_loss, (int, float))
        assert not np.isnan(final_loss)
        assert not np.isinf(final_loss)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        layer = ReLUK(k=3)

        # Test with exactly zero
        zero_input = keras.ops.zeros([2, 4])
        zero_output = layer(zero_input)
        assert np.allclose(zero_output.numpy(), keras.ops.zeros([2, 4]).numpy())

        # Test with very small positive values
        tiny_input = keras.ops.convert_to_tensor([[1e-8, 1e-6, 1e-4, 1e-2]])
        tiny_output = layer(tiny_input)

        # Should be very small but not zero, and follow the k^3 relationship
        expected = keras.ops.convert_to_tensor([[1e-24, 1e-18, 1e-12, 1e-6]])
        assert np.allclose(tiny_output.numpy(), expected.numpy(), rtol=1e-6)

        # Test with exactly 1.0 (should remain 1.0 for any k)
        ones_input = keras.ops.ones([2, 3])
        ones_output = layer(ones_input)
        assert np.allclose(ones_output.numpy(), keras.ops.ones([2, 3]).numpy())

    def test_large_k_values(self):
        """Test layer with large k values."""
        large_k_values = [50, 100]

        for k in large_k_values:
            layer = ReLUK(k=k)

            # Use small positive inputs to avoid overflow
            test_input = keras.ops.convert_to_tensor([[0.1, 0.5, 0.9, 1.0, 1.1]])
            output = layer(test_input)

            # Check output is valid (no NaN/Inf)
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))

            # Check that 1.0 remains 1.0
            ones_idx = 3  # Index where input is 1.0
            assert abs(output.numpy()[0, ones_idx] - 1.0) < 1e-6

    def test_repr_method(self, layer_instance):
        """Test the string representation of the layer."""
        repr_str = repr(layer_instance)

        assert "ReLUK" in repr_str
        assert f"k={layer_instance.k}" in repr_str
        assert layer_instance.name in repr_str