"""Comprehensive test suite for PowerMLPLayer.

This module contains tests for the PowerMLPLayer implementation,
following the project's testing standards and patterns.
"""

import pytest
import numpy as np
import keras
import tempfile
import os
import tensorflow as tf

from dl_techniques.layers.ffn.power_mlp_layer import PowerMLPLayer
from dl_techniques.layers.activations.relu_k import ReLUK
from dl_techniques.layers.activations.basis_function import BasisFunction


class TestPowerMLPLayer:
    """Test suite for PowerMLPLayer implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return keras.random.normal([4, 32])  # 4 batches, 32 input features

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return PowerMLPLayer(units=16, k=3)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = PowerMLPLayer(units=64)

        # Check default values
        assert layer.units == 64
        assert layer.k == 3  # Default k
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None
        assert layer.use_bias is True
        assert layer._build_input_shape is None
        assert layer.built is False

        # Sublayers should be None before build
        assert layer.main_dense is None
        assert layer.relu_k is None
        assert layer.basis_function is None
        assert layer.basis_dense is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        layer = PowerMLPLayer(
            units=128,
            k=5,
            kernel_initializer="glorot_uniform",
            bias_initializer="ones",
            kernel_regularizer=custom_regularizer,
            bias_regularizer=custom_regularizer,
            use_bias=False,
            name="custom_power_mlp"
        )

        # Check custom values
        assert layer.units == 128
        assert layer.k == 5
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)
        assert layer.kernel_regularizer == custom_regularizer
        assert layer.bias_regularizer == custom_regularizer
        assert layer.use_bias is False
        assert layer.name == "custom_power_mlp"

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test negative units
        with pytest.raises(ValueError, match="units must be positive"):
            PowerMLPLayer(units=-10)

        # Test zero units
        with pytest.raises(ValueError, match="units must be positive"):
            PowerMLPLayer(units=0)

        # Test non-integer units
        with pytest.raises(TypeError, match="units must be an integer"):
            PowerMLPLayer(units=64.5)

        # Test negative k
        with pytest.raises(ValueError, match="k must be positive"):
            PowerMLPLayer(units=64, k=-1)

        # Test zero k
        with pytest.raises(ValueError, match="k must be positive"):
            PowerMLPLayer(units=64, k=0)

        # Test non-integer k
        with pytest.raises(TypeError, match="k must be an integer"):
            PowerMLPLayer(units=64, k=2.5)

        # Test string parameters
        with pytest.raises(TypeError, match="units must be an integer"):
            PowerMLPLayer(units="64")

        with pytest.raises(TypeError, match="k must be an integer"):
            PowerMLPLayer(units=64, k="3")

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer builds properly and creates sublayers."""
        # Trigger build through forward pass
        output = layer_instance(input_tensor)

        # Check that layer was built
        assert layer_instance.built is True
        assert layer_instance._build_input_shape == input_tensor.shape

        # Check that all sublayers were created and built
        assert layer_instance.main_dense is not None
        assert layer_instance.relu_k is not None
        assert layer_instance.basis_function is not None
        assert layer_instance.basis_dense is not None

        # Check that sublayers are built
        assert layer_instance.main_dense.built is True
        assert layer_instance.relu_k.built is True
        assert layer_instance.basis_function.built is True
        assert layer_instance.basis_dense.built is True

        # Check sublayer configurations
        assert layer_instance.main_dense.units == layer_instance.units
        assert layer_instance.relu_k.k == layer_instance.k
        assert layer_instance.basis_dense.units == layer_instance.units
        assert layer_instance.basis_dense.use_bias is False  # Should be False

    def test_output_shapes(self, input_tensor):
        """Test that output shapes are computed correctly."""
        configs_to_test = [
            {"units": 32, "k": 1},
            {"units": 64, "k": 2},
            {"units": 128, "k": 3},
            {"units": 256, "k": 5},
        ]

        for config in configs_to_test:
            layer = PowerMLPLayer(**config)

            output = layer(input_tensor)

            # Check output shape
            expected_shape = (input_tensor.shape[0], config["units"])
            assert output.shape == expected_shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(input_tensor.shape)
            assert computed_shape == expected_shape

    def test_forward_pass(self, input_tensor, layer_instance):
        """Test that forward pass produces expected values."""
        output = layer_instance(input_tensor)

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

        # Check output shape
        expected_shape = (input_tensor.shape[0], layer_instance.units)
        assert output.shape == expected_shape

        # Test with training=False
        output_inference = layer_instance(input_tensor, training=False)
        assert output_inference.shape == expected_shape

        # Test with training=True
        output_training = layer_instance(input_tensor, training=True)
        assert output_training.shape == expected_shape

    def test_dual_branch_architecture(self, input_tensor):
        """Test that the dual-branch architecture works correctly."""
        layer = PowerMLPLayer(units=32, k=2)

        # Get output from the full layer
        full_output = layer(input_tensor)

        # Manually compute each branch to verify the architecture
        main_branch = layer.main_dense(input_tensor)
        main_branch = layer.relu_k(main_branch)

        basis_branch = layer.basis_function(input_tensor)
        basis_branch = layer.basis_dense(basis_branch)

        # Manually combine branches
        manual_output = main_branch + basis_branch

        # Should match the layer's output
        assert np.allclose(full_output.numpy(), manual_output.numpy(), rtol=1e-6)

    def test_different_configurations(self):
        """Test layer with different configurations."""
        configurations = [
            {"units": 16, "k": 1, "use_bias": True},
            {"units": 32, "k": 2, "use_bias": False},
            {"units": 64, "k": 3, "kernel_initializer": "glorot_uniform"},
            {"units": 128, "k": 5, "kernel_regularizer": keras.regularizers.L2(0.01)},
        ]

        for config in configurations:
            layer = PowerMLPLayer(**config)

            # Create test input
            test_input = keras.random.normal([2, 20])

            # Test forward pass
            output = layer(test_input)

            # Check output is valid
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))
            assert output.shape == (2, config["units"])

    def test_branch_independence(self, input_tensor):
        """Test that branches operate independently before combination."""
        layer = PowerMLPLayer(units=32, k=3)

        # Build the layer
        _ = layer(input_tensor)

        # Test main branch in isolation
        main_only = layer.main_dense(input_tensor)
        main_only = layer.relu_k(main_only)

        # Test basis branch in isolation
        basis_only = layer.basis_function(input_tensor)
        basis_only = layer.basis_dense(basis_only)

        # Both should produce valid outputs
        assert not np.any(np.isnan(main_only.numpy()))
        assert not np.any(np.isinf(main_only.numpy()))
        assert not np.any(np.isnan(basis_only.numpy()))
        assert not np.any(np.isinf(basis_only.numpy()))

        # Shapes should match
        assert main_only.shape == (input_tensor.shape[0], 32)
        assert basis_only.shape == (input_tensor.shape[0], 32)

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = PowerMLPLayer(
            units=64,
            k=3,
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            kernel_regularizer=keras.regularizers.L2(0.01),
            use_bias=True,
            name="test_power_mlp"
        )

        # Build the layer
        input_shape = (None, 32)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = PowerMLPLayer.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.units == original_layer.units
        assert recreated_layer.k == original_layer.k
        assert recreated_layer.use_bias == original_layer.use_bias
        assert recreated_layer.name == original_layer.name
        assert recreated_layer._build_input_shape == original_layer._build_input_shape

        # Check sublayers were recreated
        assert recreated_layer.main_dense is not None
        assert recreated_layer.relu_k is not None
        assert recreated_layer.basis_function is not None
        assert recreated_layer.basis_dense is not None

        # Test that both layers produce same output
        test_input = keras.random.normal([2, 32])
        original_output = original_layer(test_input)
        recreated_output = recreated_layer(test_input)

        # Note: Weights are randomly initialized, so outputs won't be identical
        # But shapes and validity should match
        assert original_output.shape == recreated_output.shape
        assert not np.any(np.isnan(recreated_output.numpy()))

    def test_model_integration(self, input_tensor):
        """Test the layer in a model context."""
        # Create a model with PowerMLP layers
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = PowerMLPLayer(units=64, k=2)(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = PowerMLPLayer(units=32, k=3)(x)
        x = keras.layers.Dropout(0.2)(x)
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
        """Test saving and loading a model with PowerMLP layers."""
        # Create a model with PowerMLP layers
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = PowerMLPLayer(units=32, k=2, name="power_mlp_1")(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = PowerMLPLayer(units=16, k=3, name="power_mlp_2")(x)
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
                custom_objects={
                    "PowerMLPLayer": PowerMLPLayer,
                    "ReLUK": ReLUK,
                    "BasisFunction": BasisFunction
                }
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer types are preserved
            assert isinstance(loaded_model.get_layer("power_mlp_1"), PowerMLPLayer)
            assert isinstance(loaded_model.get_layer("power_mlp_2"), PowerMLPLayer)

            # Check layer parameters are preserved
            assert loaded_model.get_layer("power_mlp_1").units == 32
            assert loaded_model.get_layer("power_mlp_1").k == 2
            assert loaded_model.get_layer("power_mlp_2").units == 16
            assert loaded_model.get_layer("power_mlp_2").k == 3

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = PowerMLPLayer(units=16, k=2)

        # Create inputs with different magnitudes
        batch_size = 2
        input_dim = 10

        test_cases = [
            keras.ops.zeros((batch_size, input_dim)),  # Zeros
            keras.ops.ones((batch_size, input_dim)) * 1e-10,  # Very small values
            keras.ops.ones((batch_size, input_dim)) * 1e5,  # Large positive values
            keras.ops.ones((batch_size, input_dim)) * -1e5,  # Large negative values
            keras.random.normal((batch_size, input_dim)) * 100  # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

            # Check that output shape is preserved
            assert output.shape == (batch_size, 16)

    def test_gradient_flow(self):
        """Test that gradients flow properly through the layer."""
        layer = PowerMLPLayer(units=16, k=2)

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

        # Gradients should be finite
        assert not np.any(np.isnan(gradients.numpy()))
        assert not np.any(np.isinf(gradients.numpy()))

        # Should have non-zero gradients (due to basis function branch)
        non_zero_grads = np.count_nonzero(gradients.numpy())
        total_grads = gradients.numpy().size
        assert non_zero_grads > total_grads * 0.5  # At least 50% non-zero

    def test_regularization_effects(self, input_tensor):
        """Test that regularization is properly applied to both branches."""
        # Create layer with regularization
        layer = PowerMLPLayer(
            units=32,
            k=2,
            kernel_regularizer=keras.regularizers.L2(0.1),
            bias_regularizer=keras.regularizers.L1(0.1)
        )

        # Build layer
        layer.build(input_tensor.shape)

        # Apply the layer
        _ = layer(input_tensor)

        # Check that regularization losses exist
        assert len(layer.losses) > 0

        # Both main_dense and basis_dense should contribute to regularization
        main_losses = len(layer.main_dense.losses)
        basis_losses = len(layer.basis_dense.losses)
        assert main_losses > 0 or basis_losses > 0

    def test_different_input_shapes(self):
        """Test layer with different input shapes."""
        input_shapes = [
            (8, 10),           # 2D
            (4, 16, 32),       # 3D
            (2, 8, 16),        # 3D different size
        ]

        for shape in input_shapes:
            # Create a new layer instance for each shape
            layer = PowerMLPLayer(units=16, k=2)
            test_input = keras.random.normal(shape)
            output = layer(test_input)

            # Check output shape
            expected_shape = shape[:-1] + (16,)
            assert output.shape == expected_shape

            # Check output is valid
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))

    def test_training_compatibility(self):
        """Test that the layer works correctly during training."""
        # Create a simple model
        model = keras.Sequential([
            keras.layers.Input(shape=(20,)),
            PowerMLPLayer(units=32, k=2),
            PowerMLPLayer(units=16, k=3),
            keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        # Create synthetic training data
        x_train = keras.random.normal([64, 20])
        y_train = keras.random.normal([64, 1])

        # Initial loss
        initial_loss = model.evaluate(x_train, y_train, verbose=0)

        # Train for a few epochs
        model.fit(x_train, y_train, epochs=3, batch_size=16, verbose=0)

        # Final loss
        final_loss = model.evaluate(x_train, y_train, verbose=0)

        # Training should complete without errors
        assert isinstance(final_loss, (int, float))
        assert not np.isnan(final_loss)
        assert not np.isinf(final_loss)

    def test_branch_contribution_analysis(self, input_tensor):
        """Test that both branches contribute meaningfully to the output."""
        layer = PowerMLPLayer(units=32, k=2)

        # Get full output
        full_output = layer(input_tensor)

        # Get individual branch outputs
        main_output = layer.main_dense(input_tensor)
        main_output = layer.relu_k(main_output)

        basis_output = layer.basis_function(input_tensor)
        basis_output = layer.basis_dense(basis_output)

        # Check that both branches produce non-trivial outputs
        main_magnitude = np.mean(np.abs(main_output.numpy()))
        basis_magnitude = np.mean(np.abs(basis_output.numpy()))

        assert main_magnitude > 1e-6, "Main branch output is too small"
        assert basis_magnitude > 1e-6, "Basis branch output is too small"

        # Check that the combination is indeed addition
        combined_manual = main_output + basis_output
        assert np.allclose(full_output.numpy(), combined_manual.numpy(), rtol=1e-6)

    def test_k_parameter_effects(self):
        """Test that different k values produce different behaviors."""
        test_input = keras.random.normal([4, 16])

        # Test different k values
        k_values = [1, 2, 3, 5]
        outputs = []

        for k in k_values:
            layer = PowerMLPLayer(units=16, k=k)
            output = layer(test_input)
            outputs.append(output.numpy())

        # Different k values should produce different outputs
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not np.allclose(outputs[i], outputs[j], rtol=0.1), \
                    f"k={k_values[i]} and k={k_values[j]} produced too similar outputs"

    def test_use_bias_parameter(self):
        """Test that use_bias parameter affects only the main branch."""
        test_input = keras.random.normal([2, 8])

        # Test with bias
        layer_with_bias = PowerMLPLayer(units=16, k=2, use_bias=True)
        output_with_bias = layer_with_bias(test_input)

        # Test without bias
        layer_without_bias = PowerMLPLayer(units=16, k=2, use_bias=False)
        output_without_bias = layer_without_bias(test_input)

        # Should produce different outputs
        assert not np.allclose(
            output_with_bias.numpy(),
            output_without_bias.numpy(),
            rtol=0.1
        )

        # Check that main branch has/doesn't have bias as expected
        assert layer_with_bias.main_dense.use_bias is True
        assert layer_without_bias.main_dense.use_bias is False

        # Basis branch should never have bias
        assert layer_with_bias.basis_dense.use_bias is False
        assert layer_without_bias.basis_dense.use_bias is False

    def test_repr_method(self, layer_instance):
        """Test the string representation of the layer."""
        repr_str = repr(layer_instance)

        assert "PowerMLPLayer" in repr_str
        assert f"units={layer_instance.units}" in repr_str
        assert f"k={layer_instance.k}" in repr_str
        assert layer_instance.name in repr_str

    def test_comparison_with_standard_dense(self):
        """Compare PowerMLP with standard Dense layer."""
        test_input = keras.random.normal([8, 16])

        # PowerMLP layer
        power_mlp = PowerMLPLayer(units=32, k=2)
        power_output = power_mlp(test_input)

        # Standard Dense layer
        dense = keras.layers.Dense(32)
        dense_output = dense(test_input)

        # Should produce different outputs (PowerMLP is more expressive)
        assert not np.allclose(power_output.numpy(), dense_output.numpy(), rtol=0.1)

        # Both should be valid
        assert not np.any(np.isnan(power_output.numpy()))
        assert not np.any(np.isnan(dense_output.numpy()))

        # Same output shape
        assert power_output.shape == dense_output.shape


if __name__ == "__main__":
    pytest.main([__file__])