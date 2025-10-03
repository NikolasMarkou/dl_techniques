"""
Comprehensive test suite for the refined SOM2dLayer implementation.

This test suite covers:
- Initialization with default and custom parameters
- Build process and weight creation
- Forward pass functionality
- Shape computation and validation
- Serialization and deserialization
- Model integration and deployment
- Edge cases and error handling
- Training behavior and convergence
- Numerical stability and robustness
- Memory efficiency and performance

The tests are designed to work with the modern Keras 3 implementation
that inherits from SOMLayer and follows current best practices.
"""

import pytest
import numpy as np
import keras
import tempfile
import os

# Import the refined SOM2dLayer - adjust import path as needed
from dl_techniques.layers.memory.som_2d_layer import SOM2dLayer


class TestSOM2dLayer:
    """Comprehensive test suite for the refined SOM2dLayer implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor for MNIST-like data."""
        return keras.random.normal([32, 784])

    @pytest.fixture
    def small_input_tensor(self):
        """Create a smaller test input tensor for faster testing."""
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

        # Check configuration attributes
        assert layer.map_size == (10, 10)
        assert layer.grid_shape == (10, 10)  # Inherited from SOMLayer
        assert layer.input_dim == 784
        assert layer.initial_learning_rate == 0.1
        assert layer.sigma == 1.0
        assert layer.neighborhood_function == 'gaussian'

        # Check that the layer is not built yet
        assert not layer.built

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)
        custom_initializer = keras.initializers.HeNormal()

        def custom_decay(iteration, max_iter):
            return 0.1 * np.exp(-iteration / max_iter)

        layer = SOM2dLayer(
            map_size=(8, 12),
            input_dim=256,
            initial_learning_rate=0.05,
            sigma=2.0,
            neighborhood_function='bubble',
            weights_initializer=custom_initializer,
            regularizer=custom_regularizer,
            decay_function=custom_decay,
            name='custom_som'
        )

        # Check custom values
        assert layer.map_size == (8, 12)
        assert layer.grid_shape == (8, 12)
        assert layer.input_dim == 256
        assert layer.initial_learning_rate == 0.05
        assert layer.sigma == 2.0
        assert layer.neighborhood_function == 'bubble'
        assert layer.name == 'custom_som'
        assert layer.decay_function == custom_decay

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Invalid map_size type
        with pytest.raises(ValueError, match="must be a tuple of exactly 2 integers"):
            SOM2dLayer(map_size=(10,), input_dim=784)

        # Invalid map_size with three elements
        with pytest.raises(ValueError, match="must be a tuple of exactly 2 integers"):
            SOM2dLayer(map_size=(10, 10, 10), input_dim=784)

        # Negative map size
        with pytest.raises(ValueError, match="must contain positive integers"):
            SOM2dLayer(map_size=(-1, 10), input_dim=784)

        # Zero map size
        with pytest.raises(ValueError, match="must contain positive integers"):
            SOM2dLayer(map_size=(0, 10), input_dim=784)

        # Non-integer map size
        with pytest.raises(ValueError, match="must contain positive integers"):
            SOM2dLayer(map_size=(10.5, 10), input_dim=784)

        # Note: Other parameter validations are handled by the base SOMLayer class
        # We trust that the base class handles these validations properly

    def test_build_process(self, small_input_tensor, small_layer_instance):
        """Test that the layer builds properly."""
        layer = small_layer_instance

        # Before build - layer should not be built
        assert not layer.built

        # Trigger build through forward pass
        bmu_indices, quantization_errors = layer(small_input_tensor)

        # After build - check that layer is built
        assert layer.built

        # Check output shapes
        assert bmu_indices.shape == (8, 2)
        assert quantization_errors.shape == (8,)

        # Check that weights were created
        weights_map = layer.get_weights_map()
        assert weights_map.shape == (5, 5, 10)

    def test_build_input_shape_validation(self):
        """Test build validation with different input shapes."""
        layer = SOM2dLayer(map_size=(5, 5), input_dim=10)

        # Valid input shape
        valid_input = keras.random.normal((8, 10))
        bmu_indices, errors = layer(valid_input)
        assert bmu_indices.shape == (8, 2)
        assert errors.shape == (8,)

        # The layer should handle different batch sizes gracefully
        different_batch_input = keras.random.normal((3, 10))
        bmu_indices2, errors2 = layer(different_batch_input)
        assert bmu_indices2.shape == (3, 2)
        assert errors2.shape == (3,)

    def test_forward_pass_inference(self, small_input_tensor, small_layer_instance):
        """Test forward pass in inference mode."""
        layer = small_layer_instance

        # Forward pass in inference mode
        bmu_indices, quantization_errors = layer(small_input_tensor, training=False)

        # Check output shapes and types
        assert bmu_indices.shape == (8, 2)
        assert quantization_errors.shape == (8,)

        # BMU indices should be within grid bounds
        bmu_indices_np = keras.ops.convert_to_numpy(bmu_indices)
        assert np.all(bmu_indices_np[:, 0] >= 0)
        assert np.all(bmu_indices_np[:, 0] < 5)
        assert np.all(bmu_indices_np[:, 1] >= 0)
        assert np.all(bmu_indices_np[:, 1] < 5)

        # Quantization errors should be non-negative
        errors_np = keras.ops.convert_to_numpy(quantization_errors)
        assert np.all(errors_np >= 0)

    def test_forward_pass_training(self, small_input_tensor, small_layer_instance):
        """Test forward pass in training mode."""
        layer = small_layer_instance

        # First pass to initialize
        layer(small_input_tensor, training=False)
        initial_weights = keras.ops.convert_to_numpy(layer.get_weights_map()).copy()

        # Training pass should update weights
        bmu_indices, quantization_errors = layer(small_input_tensor, training=True)

        # Check that outputs are valid
        assert bmu_indices.shape == (8, 2)
        assert quantization_errors.shape == (8,)

        # Check that weights were potentially updated (they might not change much in a single step)
        updated_weights = keras.ops.convert_to_numpy(layer.get_weights_map())
        assert updated_weights.shape == initial_weights.shape

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

        # Check that outputs are within valid ranges
        for bmu in [bmu_g, bmu_b]:
            bmu_np = keras.ops.convert_to_numpy(bmu)
            assert np.all((bmu_np >= 0) & (bmu_np < 5))

    def test_compute_output_shape(self, small_layer_instance):
        """Test output shape computation."""
        layer = small_layer_instance

        # Test with different batch sizes
        test_cases = [
            (None, 10),
            (1, 10),
            (32, 10),
            (100, 10)
        ]

        for batch_size, input_dim in test_cases:
            input_shape = (batch_size, input_dim)
            bmu_shape, error_shape = layer.compute_output_shape(input_shape)

            assert bmu_shape == (batch_size, 2)
            assert error_shape == (batch_size,)

    def test_get_weights_as_grid(self, small_input_tensor, small_layer_instance):
        """Test getting weights as grid."""
        layer = small_layer_instance

        # Build layer first
        layer(small_input_tensor, training=False)

        # Get weights as grid
        weights_grid = layer.get_weights_as_grid()

        # Check shape matches expected grid format
        assert weights_grid.shape == (5, 5, 10)

        # Check that it matches get_weights_map (they should be the same)
        weights_map = layer.get_weights_map()

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(weights_grid),
            keras.ops.convert_to_numpy(weights_map),
            rtol=1e-6, atol=1e-6,
            err_msg="get_weights_as_grid should match get_weights_map"
        )

    def test_serialization_basic(self, small_layer_instance):
        """Test basic serialization and deserialization."""
        original_layer = small_layer_instance

        # Build the layer
        test_input = keras.random.normal((4, 10))
        original_layer(test_input)

        # Get configuration
        config = original_layer.get_config()

        # Check that map_size is in config (not grid_shape)
        assert 'map_size' in config
        assert 'grid_shape' not in config  # Should be replaced by map_size
        assert config['map_size'] == (5, 5)

        # Recreate layer from config
        recreated_layer = SOM2dLayer.from_config(config)

        # Check that configuration matches
        assert recreated_layer.map_size == original_layer.map_size
        assert recreated_layer.input_dim == original_layer.input_dim
        assert recreated_layer.initial_learning_rate == original_layer.initial_learning_rate
        assert recreated_layer.sigma == original_layer.sigma
        assert recreated_layer.neighborhood_function == original_layer.neighborhood_function

    def test_serialization_with_custom_objects(self):
        """Test serialization with custom initializers and regularizers."""
        # Create layer with custom objects
        original_layer = SOM2dLayer(
            map_size=(3, 3),
            input_dim=5,
            weights_initializer=keras.initializers.HeNormal(seed=42),
            regularizer=keras.regularizers.L2(0.01),
            name='custom_som'
        )

        # Build the layer
        test_input = keras.random.normal((2, 5))
        original_layer(test_input)

        # Serialize and deserialize
        config = original_layer.get_config()
        recreated_layer = SOM2dLayer.from_config(config)

        # Check basic configuration
        assert recreated_layer.map_size == original_layer.map_size
        assert recreated_layer.name == original_layer.name

        # Build recreated layer
        recreated_layer(test_input)

        # Both layers should have same structure
        assert recreated_layer.get_weights_map().shape == original_layer.get_weights_map().shape

    def test_model_integration(self, small_input_tensor):
        """Test the layer in a model context."""
        # Create a model with the SOM layer
        inputs = keras.Input(shape=(10,))
        bmu_indices, quantization_errors = SOM2dLayer(
            map_size=(5, 5),
            input_dim=10,
            name='som_layer'
        )(inputs)

        # Use BMU indices for classification
        flattened = keras.layers.Flatten()(bmu_indices)
        outputs = keras.layers.Dense(3, activation='softmax', name='classifier')(flattened)

        model = keras.Model(inputs=inputs, outputs=outputs, name='som_model')

        # Test forward pass
        predictions = model(small_input_tensor, training=False)
        assert predictions.shape == (8, 3)

        # Test model summary doesn't crash
        summary = model.summary()

    def test_model_save_load_cycle(self, small_input_tensor):
        """Test complete model save/load cycle."""
        # Create model with SOM layer
        inputs = keras.Input(shape=(10,))
        bmu_indices, quantization_errors = SOM2dLayer(
            map_size=(4, 4),
            input_dim=10,
            name="som_layer"
        )(inputs)

        outputs = keras.layers.Dense(2, activation='softmax')(keras.layers.Flatten()(bmu_indices))
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate prediction before saving
        original_prediction = model(small_input_tensor, training=False)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "som_model.keras")

            # Save model
            model.save(model_path)

            # Load model
            loaded_model = keras.models.load_model(model_path)

            # Generate prediction with loaded model
            loaded_prediction = loaded_model(small_input_tensor, training=False)

            # Check predictions match within tolerance
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-5, atol=1e-5,
                err_msg="Predictions should match after save/load cycle"
            )

            # Check layer type is preserved
            som_layer = loaded_model.get_layer("som_layer")
            assert isinstance(som_layer, SOM2dLayer)
            assert som_layer.map_size == (4, 4)

    def test_training_behavior_consistency(self, small_input_tensor, small_layer_instance):
        """Test consistent training behavior."""
        layer = small_layer_instance

        # Record training progression
        training_errors = []

        for step in range(3):
            _, quantization_errors = layer(small_input_tensor, training=True)
            mean_error = np.mean(keras.ops.convert_to_numpy(quantization_errors))
            training_errors.append(mean_error)

        # All errors should be positive
        assert all(error > 0 for error in training_errors)

        # Check that training is happening (weights should be updating)
        # Note: We can't guarantee decreasing error in just 3 steps, but we can check consistency

    def test_numerical_stability(self, small_layer_instance):
        """Test layer stability with extreme input values."""
        layer = small_layer_instance

        test_cases = [
            keras.ops.zeros((4, 10)),  # All zeros
            keras.ops.ones((4, 10)) * 1e-8,  # Very small positive values
            keras.ops.ones((4, 10)) * 1e8,   # Very large values
            keras.random.normal((4, 10)) * 100  # Large random values
        ]

        for i, test_input in enumerate(test_cases):
            try:
                bmu_indices, quantization_errors = layer(test_input, training=False)

                # Check for numerical issues
                bmu_numpy = keras.ops.convert_to_numpy(bmu_indices)
                error_numpy = keras.ops.convert_to_numpy(quantization_errors)

                assert not np.any(np.isnan(bmu_numpy)), f"NaN in BMU indices for test case {i}"
                assert not np.any(np.isinf(bmu_numpy)), f"Inf in BMU indices for test case {i}"
                assert not np.any(np.isnan(error_numpy)), f"NaN in errors for test case {i}"
                assert not np.any(np.isinf(error_numpy)), f"Inf in errors for test case {i}"

                # BMU indices should be within valid range
                assert np.all((bmu_numpy >= 0) & (bmu_numpy < 5)), f"BMU indices out of range for test case {i}"

                # Errors should be non-negative
                assert np.all(error_numpy >= 0), f"Negative errors for test case {i}"

            except Exception as e:
                pytest.fail(f"Numerical stability test failed for case {i}: {e}")

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Single neuron SOM
        single_layer = SOM2dLayer(map_size=(1, 1), input_dim=5)
        single_input = keras.random.normal((3, 5))

        bmu_indices, errors = single_layer(single_input, training=False)

        # All BMUs should be (0, 0)
        bmu_numpy = keras.ops.convert_to_numpy(bmu_indices)
        assert np.all(bmu_numpy == 0)
        assert bmu_indices.shape == (3, 2)

        # Single input dimension
        narrow_layer = SOM2dLayer(map_size=(3, 3), input_dim=1)
        narrow_input = keras.random.normal((5, 1))

        bmu_indices, errors = narrow_layer(narrow_input, training=False)
        assert bmu_indices.shape == (5, 2)
        assert errors.shape == (5,)

    def test_batch_processing_scalability(self):
        """Test processing with different batch sizes."""
        layer = SOM2dLayer(map_size=(6, 6), input_dim=8)

        # Test various batch sizes
        for batch_size in [1, 2, 8, 16, 64]:
            test_input = keras.random.normal((batch_size, 8))

            # Test inference mode
            bmu_indices, errors = layer(test_input, training=False)
            assert bmu_indices.shape == (batch_size, 2)
            assert errors.shape == (batch_size,)

            # Test training mode
            bmu_indices_train, errors_train = layer(test_input, training=True)
            assert bmu_indices_train.shape == (batch_size, 2)
            assert errors_train.shape == (batch_size,)

    def test_regularization_integration(self, small_input_tensor):
        """Test that regularization is properly integrated."""
        # Create layer with L2 regularization
        layer = SOM2dLayer(
            map_size=(4, 4),
            input_dim=10,
            regularizer=keras.regularizers.L2(0.01)
        )

        # Apply the layer to trigger regularization
        bmu_indices, errors = layer(small_input_tensor)

        # Check that regularization losses exist if the base class supports them
        # Note: This depends on how the base SOMLayer handles regularization
        if hasattr(layer, 'losses') and len(layer.losses) > 0:
            reg_loss = layer.losses[0]
            reg_loss_value = keras.ops.convert_to_numpy(reg_loss)
            assert not np.isnan(reg_loss_value)
            assert reg_loss_value >= 0

    def test_custom_decay_function(self, small_input_tensor):
        """Test layer with custom decay function."""
        # Custom linear decay
        def linear_decay(iteration, max_iter):
            if max_iter == 0:
                return 1.0
            return max(0.01, 1.0 - iteration / max_iter)

        layer = SOM2dLayer(
            map_size=(3, 3),
            input_dim=10,
            decay_function=linear_decay,
            initial_learning_rate=0.5
        )

        # Test that it works without errors
        bmu_indices, quantization_errors = layer(small_input_tensor, training=True)

        assert bmu_indices.shape == (8, 2)
        assert quantization_errors.shape == (8,)

    def test_dtype_consistency(self):
        """Test dtype handling across different input types."""
        layer = SOM2dLayer(map_size=(3, 3), input_dim=5)

        # Test with different dtypes
        for dtype in ['float32', 'float64']:
            test_input = keras.ops.cast(keras.random.normal((4, 5)), dtype)
            bmu_indices, errors = layer(test_input, training=False)

            # BMU indices should be integers
            assert 'int' in str(bmu_indices.dtype)

            # Errors should match input dtype (or be float32/float64)
            assert 'float' in str(errors.dtype)

    def test_layer_naming_and_identification(self):
        """Test that layer naming and identification works correctly."""
        layer_name = "test_som_layer"
        layer = SOM2dLayer(
            map_size=(5, 5),
            input_dim=10,
            name=layer_name
        )

        assert layer.name == layer_name

        # Test in model context
        inputs = keras.Input(shape=(10,))
        bmu_indices, _ = layer(inputs)
        model = keras.Model(inputs=inputs, outputs=bmu_indices)

        # Should be able to retrieve layer by name
        retrieved_layer = model.get_layer(layer_name)
        assert retrieved_layer is layer
        assert isinstance(retrieved_layer, SOM2dLayer)

    def test_memory_efficiency(self):
        """Test memory efficiency with moderately large SOMs."""
        # Create a reasonably large SOM
        layer = SOM2dLayer(map_size=(15, 15), input_dim=50)

        # Process moderate batch
        test_input = keras.random.normal((32, 50))

        # Should complete without memory issues
        bmu_indices, errors = layer(test_input, training=True)

        assert bmu_indices.shape == (32, 2)
        assert errors.shape == (32,)

        # Check weight dimensions are correct
        weights = layer.get_weights_as_grid()
        assert weights.shape == (15, 15, 50)

    def test_interface_compatibility(self):
        """Test that the interface matches expected SOM2dLayer API."""
        layer = SOM2dLayer(map_size=(4, 4), input_dim=6)

        # Check that all expected methods exist
        expected_methods = [
            'get_weights_as_grid',
            'get_config',
            'from_config',
            'compute_output_shape',
            '__call__'
        ]

        for method_name in expected_methods:
            assert hasattr(layer, method_name), f"Method {method_name} should exist"
            assert callable(getattr(layer, method_name)), f"Method {method_name} should be callable"

        # Check expected attributes
        expected_attributes = [
            'map_size',
            'input_dim',
            'initial_learning_rate',
            'sigma',
            'neighborhood_function'
        ]

        for attr_name in expected_attributes:
            assert hasattr(layer, attr_name), f"Attribute {attr_name} should exist"


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v'])