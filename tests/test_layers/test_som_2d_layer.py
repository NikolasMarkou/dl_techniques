"""
Comprehensive test suite for SOM2dLayer implementation.

This test suite covers:
- Initialization with default and custom parameters
- Build process and weight creation
- Forward pass functionality
- Shape computation
- Serialization and deserialization
- Model integration
- Edge cases and error handling
- Training behavior
- Numerical stability
"""

import pytest
import numpy as np
import keras
import tempfile
import os
from typing import Tuple

# Import the refined SOM2dLayer - adjust import path as needed
from dl_techniques.layers.som_2d_layer import SOM2dLayer


class TestSOM2dLayer:
    """Test suite for SOM2dLayer implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return keras.random.normal([32, 784])  # MNIST-like data

    @pytest.fixture
    def small_input_tensor(self):
        """Create a smaller test input tensor."""
        return keras.random.normal([8, 10])

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return SOM2dLayer(map_size=(10, 10), input_dim=784)

    @pytest.fixture
    def small_layer_instance(self):
        """Create a smaller layer instance for testing."""
        return SOM2dLayer(map_size=(5, 5), input_dim=10)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = SOM2dLayer(map_size=(10, 10), input_dim=784)

        # Check default values
        assert layer.map_size == (10, 10)
        assert layer.grid_height == 10
        assert layer.grid_width == 10
        assert layer.input_dim == 784
        assert layer.initial_learning_rate == 0.1
        assert layer.sigma == 1.0
        assert layer.neighborhood_function == 'gaussian'
        assert isinstance(layer.weights_initializer, keras.initializers.RandomUniform)
        assert layer.regularizer is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)
        custom_initializer = keras.initializers.HeNormal()

        layer = SOM2dLayer(
            map_size=(8, 12),
            input_dim=256,
            initial_learning_rate=0.05,
            sigma=2.0,
            neighborhood_function='bubble',
            weights_initializer=custom_initializer,
            regularizer=custom_regularizer
        )

        # Check custom values
        assert layer.map_size == (8, 12)
        assert layer.grid_height == 8
        assert layer.grid_width == 12
        assert layer.input_dim == 256
        assert layer.initial_learning_rate == 0.05
        assert layer.sigma == 2.0
        assert layer.neighborhood_function == 'bubble'
        assert layer.weights_initializer == custom_initializer
        assert layer.regularizer == custom_regularizer

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Negative map size
        with pytest.raises(ValueError, match="Map size must be positive"):
            SOM2dLayer(map_size=(-1, 10), input_dim=784)

        # Zero map size
        with pytest.raises(ValueError, match="Map size must be positive"):
            SOM2dLayer(map_size=(0, 10), input_dim=784)

        # Negative input dimension
        with pytest.raises(ValueError, match="Input dimension must be positive"):
            SOM2dLayer(map_size=(10, 10), input_dim=-1)

        # Zero input dimension
        with pytest.raises(ValueError, match="Input dimension must be positive"):
            SOM2dLayer(map_size=(10, 10), input_dim=0)

        # Negative learning rate
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            SOM2dLayer(map_size=(10, 10), input_dim=784, initial_learning_rate=-0.1)

        # Zero learning rate
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            SOM2dLayer(map_size=(10, 10), input_dim=784, initial_learning_rate=0.0)

        # Negative sigma
        with pytest.raises(ValueError, match="Sigma must be positive"):
            SOM2dLayer(map_size=(10, 10), input_dim=784, sigma=-1.0)

        # Invalid neighborhood function
        with pytest.raises(ValueError, match="Neighborhood function must be"):
            SOM2dLayer(map_size=(10, 10), input_dim=784, neighborhood_function='invalid')

    def test_build_process(self, small_input_tensor, small_layer_instance):
        """Test that the layer builds properly."""
        layer = small_layer_instance

        # Before build
        assert layer.weights_map is None
        assert layer.iterations is None
        assert layer.max_iterations is None
        assert layer.grid_positions is None
        assert not layer.built

        # Trigger build
        layer(small_input_tensor)

        # After build
        assert layer.built is True
        assert layer.weights_map is not None
        assert layer.iterations is not None
        assert layer.max_iterations is not None
        assert layer.grid_positions is not None

        # Check weight shapes
        assert layer.weights_map.shape == (5, 5, 10)
        assert layer.grid_positions.shape == (5, 5, 2)
        assert layer.iterations.shape == ()
        assert layer.max_iterations.shape == ()

    def test_build_input_shape_mismatch(self):
        """Test that build fails with mismatched input shape."""
        layer = SOM2dLayer(map_size=(5, 5), input_dim=10)

        with pytest.raises(ValueError, match="Expected input shape with dimension 10"):
            layer.build((None, 20))  # Wrong input dimension

    def test_output_shapes(self, small_input_tensor, small_layer_instance):
        """Test that output shapes are computed correctly."""
        layer = small_layer_instance
        bmu_indices, quantization_errors = layer(small_input_tensor)

        # Check output shapes
        assert bmu_indices.shape == (8, 2)  # [batch_size, 2]
        assert quantization_errors.shape == (8,)  # [batch_size]

        # Test compute_output_shape separately
        bmu_shape, error_shape = layer.compute_output_shape((8, 10))
        assert bmu_shape == (8, 2)
        assert error_shape == (8,)

    def test_forward_pass_inference(self, small_input_tensor, small_layer_instance):
        """Test forward pass in inference mode."""
        layer = small_layer_instance

        # Get initial weights
        initial_weights = layer.weights_map.numpy() if layer.weights_map is not None else None

        # Forward pass in inference mode
        bmu_indices, quantization_errors = layer(small_input_tensor, training=False)

        # Check output types and ranges
        # During actual computation, Keras returns backend tensors, not KerasTensor
        assert hasattr(bmu_indices, 'numpy')  # Check it's a tensor-like object
        assert hasattr(quantization_errors, 'numpy')  # Check it's a tensor-like object

        # BMU indices should be within grid bounds
        bmu_indices_np = bmu_indices.numpy()
        assert np.all(bmu_indices_np[:, 0] >= 0)
        assert np.all(bmu_indices_np[:, 0] < 5)
        assert np.all(bmu_indices_np[:, 1] >= 0)
        assert np.all(bmu_indices_np[:, 1] < 5)

        # Quantization errors should be non-negative
        assert np.all(quantization_errors.numpy() >= 0)

    def test_forward_pass_training(self, small_input_tensor, small_layer_instance):
        """Test forward pass in training mode."""
        layer = small_layer_instance

        # Get initial weights
        layer(small_input_tensor, training=False)  # Initialize weights
        initial_weights = layer.weights_map.numpy().copy()
        initial_iterations = layer.iterations.numpy()

        # Forward pass in training mode
        bmu_indices, quantization_errors = layer(small_input_tensor, training=True)

        # Check that weights were updated
        updated_weights = layer.weights_map.numpy()
        assert not np.allclose(initial_weights, updated_weights)

        # Check that iterations were updated
        updated_iterations = layer.iterations.numpy()
        assert updated_iterations == initial_iterations + 1

    def test_neighborhood_functions(self, small_input_tensor):
        """Test different neighborhood functions."""
        # Test Gaussian neighborhood
        gaussian_layer = SOM2dLayer(
            map_size=(5, 5),
            input_dim=10,
            neighborhood_function='gaussian'
        )
        bmu_g, error_g = gaussian_layer(small_input_tensor)

        # Test Bubble neighborhood
        bubble_layer = SOM2dLayer(
            map_size=(5, 5),
            input_dim=10,
            neighborhood_function='bubble'
        )
        bmu_b, error_b = bubble_layer(small_input_tensor)

        # Both should produce valid outputs
        assert bmu_g.shape == bmu_b.shape == (8, 2)
        assert error_g.shape == error_b.shape == (8,)

    def test_serialization(self, small_layer_instance):
        """Test serialization and deserialization of the layer."""
        original_layer = small_layer_instance

        # Build the layer
        input_shape = (None, 10)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = SOM2dLayer.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.map_size == original_layer.map_size
        assert recreated_layer.input_dim == original_layer.input_dim
        assert recreated_layer.initial_learning_rate == original_layer.initial_learning_rate
        assert recreated_layer.sigma == original_layer.sigma
        assert recreated_layer.neighborhood_function == original_layer.neighborhood_function

        # Check weights have same shape
        assert recreated_layer.weights_map.shape == original_layer.weights_map.shape

    def test_serialization_with_custom_objects(self):
        """Test serialization with custom initializers and regularizers."""
        # Create layer with custom objects
        original_layer = SOM2dLayer(
            map_size=(3, 3),
            input_dim=5,
            weights_initializer=keras.initializers.HeNormal(),
            regularizer=keras.regularizers.L2(0.01)
        )

        # Build the layer
        original_layer.build((None, 5))

        # Serialize and deserialize
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        recreated_layer = SOM2dLayer.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check that custom objects were preserved
        assert isinstance(recreated_layer.weights_initializer, keras.initializers.HeNormal)
        assert isinstance(recreated_layer.regularizer, keras.regularizers.L2)

    def test_model_integration(self, small_input_tensor):
        """Test the layer in a model context."""
        # Create a simple model with the SOM layer
        inputs = keras.Input(shape=(10,))
        bmu_indices, quantization_errors = SOM2dLayer(map_size=(5, 5), input_dim=10)(inputs)

        # Use the BMU indices in a dense layer
        flattened = keras.layers.Flatten()(bmu_indices)
        outputs = keras.layers.Dense(3, activation='softmax')(flattened)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Test forward pass
        predictions = model(small_input_tensor, training=False)
        assert predictions.shape == (8, 3)

    def test_model_save_load(self, small_input_tensor):
        """Test saving and loading a model with the SOM layer."""
        # Create a model with the SOM layer
        inputs = keras.Input(shape=(10,))
        bmu_indices, quantization_errors = SOM2dLayer(
            map_size=(5, 5),
            input_dim=10,
            name="som_layer"
        )(inputs)

        # Simple output layer
        flattened = keras.layers.Flatten()(bmu_indices)
        outputs = keras.layers.Dense(3)(flattened)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate prediction before saving
        original_prediction = model.predict(small_input_tensor)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"SOM2dLayer": SOM2dLayer}
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(small_input_tensor)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer type is preserved
            assert isinstance(loaded_model.get_layer("som_layer"), SOM2dLayer)

    def test_training_behavior(self, small_input_tensor, small_layer_instance):
        """Test that training reduces quantization error over time."""
        layer = small_layer_instance

        # Multiple training steps
        errors = []
        for i in range(5):
            _, quantization_errors = layer(small_input_tensor, training=True)
            mean_error = np.mean(quantization_errors.numpy())
            errors.append(mean_error)

        # Check that iterations increased
        assert layer.iterations.numpy() == 5

        # Check that errors generally decrease (allowing for some variance)
        first_error = errors[0]
        last_error = errors[-1]
        assert last_error <= first_error * 1.1  # Allow 10% tolerance

    def test_numerical_stability(self, small_layer_instance):
        """Test layer stability with extreme input values."""
        layer = small_layer_instance

        # Test with different input magnitudes
        test_cases = [
            keras.ops.zeros((4, 10)),  # Zeros
            keras.ops.ones((4, 10)) * 1e-10,  # Very small values
            keras.ops.ones((4, 10)) * 1e10,  # Very large values
            keras.random.normal((4, 10)) * 1e5  # Large random values
        ]

        for test_input in test_cases:
            bmu_indices, quantization_errors = layer(test_input, training=False)

            # Check for NaN/Inf values
            bmu_numpy = bmu_indices.numpy()
            error_numpy = quantization_errors.numpy()

            assert not np.any(np.isnan(bmu_numpy)), "NaN values detected in BMU indices"
            assert not np.any(np.isinf(bmu_numpy)), "Inf values detected in BMU indices"
            assert not np.any(np.isnan(error_numpy)), "NaN values detected in errors"
            assert not np.any(np.isinf(error_numpy)), "Inf values detected in errors"

    def test_regularization(self, small_input_tensor):
        """Test that regularization losses are properly applied."""
        # Create layer with regularization
        layer = SOM2dLayer(
            map_size=(5, 5),
            input_dim=10,
            regularizer=keras.regularizers.L2(0.1)
        )

        # No regularization losses before calling the layer
        assert len(layer.losses) == 0

        # Build and apply the layer (this should trigger regularization)
        _ = layer(small_input_tensor)

        # Check that regularization losses exist after calling the layer
        assert len(layer.losses) > 0

        # Check that the loss is a scalar
        reg_loss = layer.losses[0]
        assert hasattr(reg_loss, 'numpy')
        assert reg_loss.numpy().shape == ()  # Should be scalar

    def test_custom_decay_function(self, small_input_tensor):
        """Test layer with custom decay function."""
        # Custom exponential decay
        def exp_decay(iteration, max_iter):
            return 0.1 * np.exp(-iteration / max_iter)

        layer = SOM2dLayer(
            map_size=(5, 5),
            input_dim=10,
            decay_function=exp_decay
        )

        # Test that it works
        bmu_indices, quantization_errors = layer(small_input_tensor, training=True)

        # Check outputs are valid
        assert bmu_indices.shape == (8, 2)
        assert quantization_errors.shape == (8,)

    def test_get_weights_as_grid(self, small_input_tensor, small_layer_instance):
        """Test getting weights as grid."""
        layer = small_layer_instance

        # Build layer
        layer(small_input_tensor, training=False)

        # Get weights as grid
        weights_grid = layer.get_weights_as_grid()

        # Check shape
        assert weights_grid.shape == (5, 5, 10)

        # Check that it's the same as the internal weights
        assert np.allclose(weights_grid.numpy(), layer.weights_map.numpy())

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Single neuron SOM
        single_layer = SOM2dLayer(map_size=(1, 1), input_dim=5)
        single_input = keras.random.normal((3, 5))

        bmu_indices, errors = single_layer(single_input, training=False)

        # All BMUs should be (0, 0)
        assert np.all(bmu_indices.numpy() == 0)

        # Single input dimension
        narrow_layer = SOM2dLayer(map_size=(3, 3), input_dim=1)
        narrow_input = keras.random.normal((5, 1))

        bmu_indices, errors = narrow_layer(narrow_input, training=False)

        # Should produce valid outputs
        assert bmu_indices.shape == (5, 2)
        assert errors.shape == (5,)

    def test_batch_processing(self):
        """Test processing different batch sizes."""
        layer = SOM2dLayer(map_size=(4, 4), input_dim=8)

        # Test different batch sizes
        for batch_size in [1, 5, 10, 32]:
            test_input = keras.random.normal((batch_size, 8))
            bmu_indices, errors = layer(test_input, training=False)

            assert bmu_indices.shape == (batch_size, 2)
            assert errors.shape == (batch_size,)

            # Test training mode as well
            bmu_indices_train, errors_train = layer(test_input, training=True)
            assert bmu_indices_train.shape == (batch_size, 2)
            assert errors_train.shape == (batch_size,)

    def test_layer_naming(self):
        """Test that layer naming works correctly."""
        layer = SOM2dLayer(map_size=(5, 5), input_dim=10, name="test_som")

        assert layer.name == "test_som"

        # Test in model context
        inputs = keras.Input(shape=(10,))
        bmu_indices, _ = layer(inputs)
        model = keras.Model(inputs=inputs, outputs=bmu_indices)

        assert model.get_layer("test_som") is layer

    def test_dtype_consistency(self, small_input_tensor):
        """Test that the layer handles different dtypes consistently."""
        layer = SOM2dLayer(map_size=(3, 3), input_dim=10)

        # Test with float32 input
        float32_input = keras.ops.cast(small_input_tensor, "float32")
        bmu_indices, errors = layer(float32_input, training=False)

        # Check dtypes using .name attribute
        assert bmu_indices.dtype.name == "int32"
        assert errors.dtype.name == "float32"

    def test_memory_efficiency(self):
        """Test that layer doesn't consume excessive memory."""
        # Create a moderately large SOM
        layer = SOM2dLayer(map_size=(20, 20), input_dim=100)

        # Process some data
        test_input = keras.random.normal((16, 100))
        bmu_indices, errors = layer(test_input, training=True)

        # Should complete without memory issues
        assert bmu_indices.shape == (16, 2)
        assert errors.shape == (16,)


if __name__ == '__main__':
    pytest.main([__file__])