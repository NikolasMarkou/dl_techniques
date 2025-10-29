"""Comprehensive test suite for BasisFunction activation layer.

This module contains tests for the BasisFunction layer implementation,
following the project's testing standards and patterns.
"""

import pytest
import numpy as np
import keras
import tempfile
import os
import tensorflow as tf

from dl_techniques.layers.activations.basis_function import BasisFunction


class TestBasisFunction:
    """Test suite for BasisFunction activation layer implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return keras.random.normal([4, 32, 64])  # 4 batches, 32 features, 64 dims

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return BasisFunction()

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = BasisFunction()

        # Check default values
        assert layer.built is False

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        layer = BasisFunction(
            name="custom_basis_function",
            dtype="float32",
            trainable=True
        )

        # Check custom values
        assert layer.name == "custom_basis_function"
        assert layer.dtype == "float32"
        assert layer.trainable is True

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
        layer = BasisFunction()

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

    def test_mathematical_correctness(self):
        """Test the mathematical correctness of the basis function."""
        layer = BasisFunction()

        # Test with known values
        test_input = keras.ops.convert_to_tensor([
            [-3.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        ])

        output = layer(test_input)

        # Calculate expected values: b(x) = x / (1 + e^(-x))
        x_values = test_input.numpy()[0]
        expected_values = []
        for x in x_values:
            expected = x / (1.0 + np.exp(-x))
            expected_values.append(expected)

        expected = keras.ops.convert_to_tensor([expected_values])

        # Check values are close (accounting for floating point precision)
        assert np.allclose(output.numpy(), expected.numpy(), rtol=1e-6)

    def test_equivalence_to_swish(self):
        """Test that basis function is equivalent to Swish (x * sigmoid(x))."""
        layer = BasisFunction()

        test_input = keras.random.normal([8, 16])

        # Apply basis function
        output_basis = layer(test_input)

        # Calculate Swish: x * sigmoid(x)
        sigmoid_x = keras.ops.sigmoid(test_input)
        output_swish = test_input * sigmoid_x

        # Results should be identical
        assert np.allclose(output_basis.numpy(), output_swish.numpy(), rtol=1e-6)

    def test_zero_input_handling(self):
        """Test that zero inputs are properly handled."""
        layer = BasisFunction()

        # Test with exactly zero
        zero_input = keras.ops.zeros([2, 4])
        output = layer(zero_input)

        # b(0) = 0 / (1 + e^0) = 0 / 2 = 0
        expected = keras.ops.zeros([2, 4])
        assert np.allclose(output.numpy(), expected.numpy(), atol=1e-7)

    def test_positive_inputs_handling(self):
        """Test behavior with positive inputs."""
        layer = BasisFunction()

        # Test with positive inputs
        positive_input = keras.ops.convert_to_tensor([
            [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        ])
        output = layer(positive_input)

        # For large positive values, basis function should approach x
        large_positive_input = positive_input.numpy()[0][-1]  # 10.0
        large_positive_output = output.numpy()[0][-1]

        # Should be close to the input value for large positive inputs
        assert abs(large_positive_output - large_positive_input) < 0.1

        # All outputs should be positive for positive inputs
        assert np.all(output.numpy() >= 0)

    def test_negative_inputs_handling(self):
        """Test behavior with negative inputs."""
        layer = BasisFunction()

        # Test with negative inputs
        negative_input = keras.ops.convert_to_tensor([
            [-10.0, -5.0, -2.0, -1.0, -0.1]
        ])
        output = layer(negative_input)

        # For large negative values, basis function should approach 0
        large_negative_output = output.numpy()[0][0]  # Output for -10.0
        assert abs(large_negative_output) < 0.01

        # All outputs should be negative for negative inputs
        assert np.all(output.numpy() <= 0)

    def test_mixed_inputs(self):
        """Test with mixed positive and negative inputs."""
        layer = BasisFunction()

        mixed_input = keras.ops.convert_to_tensor([
            [-2.0, -1.0, 0.0, 1.0, 2.0]
        ])
        output = layer(mixed_input)

        # Check that the function produces reasonable outputs
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

        # Check that zero input produces zero output
        zero_idx = 2  # Index where input is 0.0
        assert abs(output.numpy()[0, zero_idx]) < 1e-6

    def test_smoothness_property(self):
        """Test that the function is smooth (no discontinuities)."""
        layer = BasisFunction()

        # Create a range of inputs around potentially problematic points
        test_points = np.linspace(-5, 5, 1000)
        test_input = keras.ops.convert_to_tensor(test_points.reshape(1, -1))

        output = layer(test_input)

        # Check that output is finite everywhere
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

        # Check smoothness by ensuring no large jumps in output
        output_values = output.numpy().flatten()
        differences = np.diff(output_values)
        max_difference = np.max(np.abs(differences))

        # Should be smooth, so max difference should be reasonable
        assert max_difference < 0.1  # Adjust threshold as needed

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = BasisFunction(name="test_basis_function")

        # Build the layer
        input_shape = (None, 32, 64)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = BasisFunction.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.name == original_layer.name

        # Test that both layers produce same output
        test_input = keras.random.normal([2, 32, 64])
        original_output = original_layer(test_input)
        recreated_output = recreated_layer(test_input)

        assert np.allclose(original_output.numpy(), recreated_output.numpy())

    def test_model_integration(self, input_tensor):
        """Test the layer in a model context."""
        # Create a simple model with the BasisFunction layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = keras.layers.Dense(64)(inputs)
        x = BasisFunction()(x)
        x = keras.layers.Dense(32)(x)
        x = BasisFunction()(x)
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
        """Test saving and loading a model with the BasisFunction layer."""
        # Create a model with the BasisFunction layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = keras.layers.Dense(32)(inputs)
        x = BasisFunction(name="basis_hidden")(x)
        x = keras.layers.Dense(16)(x)
        x = BasisFunction(name="basis_output")(x)
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
                custom_objects={"BasisFunction": BasisFunction}
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer types are preserved
            assert isinstance(loaded_model.get_layer("basis_hidden"), BasisFunction)
            assert isinstance(loaded_model.get_layer("basis_output"), BasisFunction)

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = BasisFunction()

        # Create inputs with different magnitudes
        batch_size = 2
        feature_dim = 16

        test_cases = [
            keras.ops.zeros((batch_size, feature_dim)),  # Zeros
            keras.ops.ones((batch_size, feature_dim)) * 1e-10,  # Very small values
            keras.ops.ones((batch_size, feature_dim)) * 1e5,  # Large positive values
            keras.ops.ones((batch_size, feature_dim)) * -1e5,  # Large negative values
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
        layer = BasisFunction()

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

        # Gradients should be finite everywhere (smooth function)
        assert not np.any(np.isnan(gradients.numpy()))
        assert not np.any(np.isinf(gradients.numpy()))

        # For basis function, gradients should generally be non-zero
        # (unlike ReLU which has zero gradients for negative inputs)
        non_zero_grads = np.count_nonzero(gradients.numpy())
        total_grads = gradients.numpy().size

        # Most gradients should be non-zero for a smooth function
        assert non_zero_grads > total_grads * 0.8  # At least 80% non-zero

    def test_different_input_shapes(self):
        """Test layer with different input shapes."""
        layer = BasisFunction()

        input_shapes = [
            (8, 10),  # 2D
            (4, 16, 32),  # 3D
            (2, 8, 8, 3),  # 4D (image-like)
            (1, 4, 6, 8, 2),  # 5D
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
            BasisFunction(),
            keras.layers.Dense(8),
            BasisFunction(),
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
        layer = BasisFunction()

        # Test with exactly zero
        zero_input = keras.ops.zeros([2, 4])
        zero_output = layer(zero_input)
        assert np.allclose(zero_output.numpy(), keras.ops.zeros([2, 4]).numpy())

        # Test with very small values
        tiny_positive = keras.ops.convert_to_tensor([[1e-8, 1e-6, 1e-4]])
        tiny_negative = keras.ops.convert_to_tensor([[-1e-8, -1e-6, -1e-4]])

        output_pos = layer(tiny_positive)
        output_neg = layer(tiny_negative)

        # Should be very small but not zero, and have appropriate signs
        assert np.all(output_pos.numpy() >= 0)
        assert np.all(output_neg.numpy() <= 0)

        # Test asymptotic behavior
        large_positive = keras.ops.convert_to_tensor([[10.0, 20.0, 50.0]])
        large_negative = keras.ops.convert_to_tensor([[-10.0, -20.0, -50.0]])

        output_large_pos = layer(large_positive)
        output_large_neg = layer(large_negative)

        # For large positive x, b(x) ≈ x
        assert np.allclose(output_large_pos.numpy(), large_positive.numpy(), rtol=0.01)

        # For large negative x, b(x) ≈ 0
        assert np.all(np.abs(output_large_neg.numpy()) < 0.01)

    def test_powermlp_context(self):
        """Test the layer in the context it was designed for (PowerMLP)."""
        # Simulate a PowerMLP-style architecture
        inputs = keras.Input(shape=(128,))

        # Main branch
        main_branch = keras.layers.Dense(64)(inputs)

        # Basis function branch (this layer's purpose)
        basis_branch = keras.layers.Dense(64)(inputs)
        basis_branch = BasisFunction()(basis_branch)

        # Combine branches (element-wise multiplication)
        combined = keras.layers.Multiply()([main_branch, basis_branch])

        # Output
        outputs = keras.layers.Dense(10)(combined)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')

        # Test forward pass
        test_input = keras.random.normal([4, 128])
        output = model(test_input)

        assert output.shape == (4, 10)
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_repr_method(self, layer_instance):
        """Test the string representation of the layer."""
        repr_str = repr(layer_instance)

        assert "BasisFunction" in repr_str
        assert layer_instance.name in repr_str

    def test_comparison_with_standard_activations(self):
        """Compare performance characteristics with standard activations."""
        # Test that BasisFunction produces different outputs than standard activations
        test_input = keras.random.normal([8, 16])

        basis_layer = BasisFunction()
        relu_layer = keras.layers.ReLU()
        sigmoid_layer = keras.layers.Activation('sigmoid')

        basis_output = basis_layer(test_input)
        relu_output = relu_layer(test_input)
        sigmoid_output = sigmoid_layer(test_input)

        # Should produce different outputs
        assert not np.allclose(basis_output.numpy(), relu_output.numpy())
        assert not np.allclose(basis_output.numpy(), sigmoid_output.numpy())

        # But should be well-behaved (finite values)
        for output in [basis_output, relu_output, sigmoid_output]:
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))


if __name__ == "__main__":
    pytest.main([__file__])