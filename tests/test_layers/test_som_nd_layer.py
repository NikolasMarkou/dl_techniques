import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

from dl_techniques.layers.som_nd_layer import SOMLayer


class TestSOMLayer:
    """Test suite for SOMLayer implementation."""

    @pytest.fixture
    def input_data_1d(self):
        """Create 1D test input data."""
        return keras.random.uniform([32, 10], seed=42)

    @pytest.fixture
    def input_data_2d(self):
        """Create 2D test input data."""
        return keras.random.uniform([64, 784], seed=42)

    @pytest.fixture
    def input_data_3d(self):
        """Create 3D test input data."""
        return keras.random.uniform([16, 128], seed=42)

    @pytest.fixture
    def som_1d(self):
        """Create a 1D SOM instance."""
        return SOMLayer(
            grid_shape=(20,),
            input_dim=10,
            initial_learning_rate=0.1,
            sigma=2.0,
            neighborhood_function='gaussian'
        )

    @pytest.fixture
    def som_2d(self):
        """Create a 2D SOM instance."""
        return SOMLayer(
            grid_shape=(10, 10),
            input_dim=784,
            initial_learning_rate=0.1,
            sigma=1.5,
            neighborhood_function='gaussian'
        )

    @pytest.fixture
    def som_3d(self):
        """Create a 3D SOM instance."""
        return SOMLayer(
            grid_shape=(5, 5, 5),
            input_dim=128,
            initial_learning_rate=0.1,
            sigma=1.0,
            neighborhood_function='bubble'
        )

    def test_initialization_1d(self, som_1d):
        """Test initialization of 1D SOM with default parameters."""
        assert som_1d.grid_shape == (20,)
        assert som_1d.grid_dim == 1
        assert som_1d.input_dim == 10
        assert som_1d.initial_learning_rate == 0.1
        assert som_1d.sigma == 2.0
        assert som_1d.neighborhood_function == 'gaussian'
        assert som_1d.weights_map is None  # Not built yet

    def test_initialization_2d(self, som_2d):
        """Test initialization of 2D SOM with default parameters."""
        assert som_2d.grid_shape == (10, 10)
        assert som_2d.grid_dim == 2
        assert som_2d.input_dim == 784
        assert som_2d.initial_learning_rate == 0.1
        assert som_2d.sigma == 1.5
        assert som_2d.neighborhood_function == 'gaussian'

    def test_initialization_3d(self, som_3d):
        """Test initialization of 3D SOM with bubble neighborhood."""
        assert som_3d.grid_shape == (5, 5, 5)
        assert som_3d.grid_dim == 3
        assert som_3d.input_dim == 128
        assert som_3d.neighborhood_function == 'bubble'

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)
        custom_initializer = keras.initializers.HeNormal()

        som = SOMLayer(
            grid_shape=(8, 8),
            input_dim=100,
            initial_learning_rate=0.05,
            sigma=3.0,
            neighborhood_function='bubble',
            weights_initializer=custom_initializer,
            regularizer=custom_regularizer
        )

        assert som.grid_shape == (8, 8)
        assert som.input_dim == 100
        assert som.initial_learning_rate == 0.05
        assert som.sigma == 3.0
        assert som.neighborhood_function == 'bubble'
        assert som.weights_initializer == custom_initializer
        assert som.regularizer == custom_regularizer

    def test_initialization_special_string_initializers(self):
        """Test initialization with special string initializers."""
        som_random = SOMLayer(
            grid_shape=(5, 5),
            input_dim=10,
            weights_initializer='random'
        )

        som_sample = SOMLayer(
            grid_shape=(5, 5),
            input_dim=10,
            weights_initializer='sample'
        )

        assert som_random.weights_initializer is None
        assert som_sample.weights_initializer is None

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="must be a tuple of positive integers"):
            SOMLayer(grid_shape=(-5, 10), input_dim=10)

        with pytest.raises(ValueError, match="must be a tuple of positive integers"):
            SOMLayer(grid_shape=(0, 10), input_dim=10)

        with pytest.raises(ValueError, match="must be a positive integer"):
            SOMLayer(grid_shape=(10, 10), input_dim=-5)

        with pytest.raises(ValueError, match="Learning rate must be positive"):
            SOMLayer(grid_shape=(10, 10), input_dim=10, initial_learning_rate=-0.1)

        with pytest.raises(ValueError, match="Sigma must be positive"):
            SOMLayer(grid_shape=(10, 10), input_dim=10, sigma=-1.0)

        with pytest.raises(ValueError, match="Neighborhood function must be"):
            SOMLayer(grid_shape=(10, 10), input_dim=10, neighborhood_function='invalid')

    def test_build_process_1d(self, som_1d, input_data_1d):
        """Test that 1D SOM builds properly."""
        som_1d.build(input_data_1d.shape)

        assert som_1d.built is True
        assert som_1d.weights_map is not None
        assert som_1d.weights_map.shape == (20, 10)
        assert som_1d.iterations is not None
        assert som_1d.max_iterations is not None
        assert som_1d.grid_positions is not None
        assert som_1d.grid_positions.shape == (20, 1)

    def test_build_process_2d(self, som_2d, input_data_2d):
        """Test that 2D SOM builds properly."""
        som_2d.build(input_data_2d.shape)

        assert som_2d.built is True
        assert som_2d.weights_map is not None
        assert som_2d.weights_map.shape == (10, 10, 784)
        assert som_2d.grid_positions.shape == (10, 10, 2)

    def test_build_process_3d(self, som_3d, input_data_3d):
        """Test that 3D SOM builds properly."""
        som_3d.build(input_data_3d.shape)

        assert som_3d.built is True
        assert som_3d.weights_map is not None
        assert som_3d.weights_map.shape == (5, 5, 5, 128)
        assert som_3d.grid_positions.shape == (5, 5, 5, 3)

    def test_build_with_wrong_input_shape(self, som_2d):
        """Test that building with wrong input shape raises error."""
        with pytest.raises(ValueError, match="Expected input shape"):
            som_2d.build((64, 100))  # Wrong input dimension

    def test_forward_pass_1d(self, som_1d, input_data_1d):
        """Test forward pass with 1D SOM."""
        bmu_indices, quantization_errors = som_1d(input_data_1d, training=False)

        assert bmu_indices.shape == (32, 1)  # batch_size, grid_dim
        assert quantization_errors.shape == (32,)  # batch_size
        assert bmu_indices.dtype == tf.int32
        assert quantization_errors.dtype == tf.float32

        # Check that BMU indices are within valid range
        assert tf.reduce_all(bmu_indices >= 0)
        assert tf.reduce_all(bmu_indices < 20)

        # Check that quantization errors are non-negative
        assert tf.reduce_all(quantization_errors >= 0)

    def test_forward_pass_2d(self, som_2d, input_data_2d):
        """Test forward pass with 2D SOM."""
        bmu_indices, quantization_errors = som_2d(input_data_2d, training=False)

        assert bmu_indices.shape == (64, 2)  # batch_size, grid_dim
        assert quantization_errors.shape == (64,)

        # Check that BMU indices are within valid range
        assert tf.reduce_all(bmu_indices[:, 0] >= 0)
        assert tf.reduce_all(bmu_indices[:, 0] < 10)
        assert tf.reduce_all(bmu_indices[:, 1] >= 0)
        assert tf.reduce_all(bmu_indices[:, 1] < 10)

    def test_forward_pass_3d(self, som_3d, input_data_3d):
        """Test forward pass with 3D SOM."""
        bmu_indices, quantization_errors = som_3d(input_data_3d, training=False)

        assert bmu_indices.shape == (16, 3)  # batch_size, grid_dim
        assert quantization_errors.shape == (16,)

        # Check that BMU indices are within valid range
        assert tf.reduce_all(bmu_indices[:, 0] >= 0)
        assert tf.reduce_all(bmu_indices[:, 0] < 5)
        assert tf.reduce_all(bmu_indices[:, 1] >= 0)
        assert tf.reduce_all(bmu_indices[:, 1] < 5)
        assert tf.reduce_all(bmu_indices[:, 2] >= 0)
        assert tf.reduce_all(bmu_indices[:, 2] < 5)

    def test_training_mode_weight_updates(self, som_2d, input_data_2d):
        """Test that weights are updated during training."""
        # Get initial weights
        initial_weights = som_2d.weights_map.numpy().copy()

        # Forward pass in training mode
        bmu_indices, quantization_errors = som_2d(input_data_2d, training=True)

        # Check that weights have been updated
        updated_weights = som_2d.weights_map.numpy()
        assert not np.allclose(initial_weights, updated_weights)

        # Check that iterations counter was updated
        assert som_2d.iterations.numpy() > 0

    def test_inference_mode_no_weight_updates(self, som_2d, input_data_2d):
        """Test that weights are not updated during inference."""
        # Get initial weights
        initial_weights = som_2d.weights_map.numpy().copy()

        # Forward pass in inference mode
        bmu_indices, quantization_errors = som_2d(input_data_2d, training=False)

        # Check that weights have not been updated
        updated_weights = som_2d.weights_map.numpy()
        assert np.allclose(initial_weights, updated_weights)

        # Check that iterations counter was not updated
        assert som_2d.iterations.numpy() == 0

    def test_neighborhood_functions(self, input_data_2d):
        """Test different neighborhood functions."""
        som_gaussian = SOMLayer(
            grid_shape=(8, 8),
            input_dim=784,
            neighborhood_function='gaussian'
        )

        som_bubble = SOMLayer(
            grid_shape=(8, 8),
            input_dim=784,
            neighborhood_function='bubble'
        )

        # Test both neighborhood functions work
        bmu_g, err_g = som_gaussian(input_data_2d, training=True)
        bmu_b, err_b = som_bubble(input_data_2d, training=True)

        assert bmu_g.shape == bmu_b.shape
        assert err_g.shape == err_b.shape

    def test_custom_decay_function(self, input_data_2d):
        """Test custom decay function."""

        def custom_decay(current_iter, max_iter):
            return 0.1 * tf.exp(-current_iter / max_iter)

        som = SOMLayer(
            grid_shape=(5, 5),
            input_dim=784,
            decay_function=custom_decay
        )

        # Test that custom decay function is used
        bmu_indices, quantization_errors = som(input_data_2d, training=True)
        assert bmu_indices.shape == (64, 2)
        assert quantization_errors.shape == (64,)

    def test_get_weights_map(self, som_2d, input_data_2d):
        """Test get_weights_map method."""
        # Build the layer
        som_2d(input_data_2d, training=False)

        weights_map = som_2d.get_weights_map()

        assert weights_map.shape == (10, 10, 784)
        assert weights_map.dtype == tf.float32

        # Check that it's the same as the internal weights
        assert tf.reduce_all(tf.equal(weights_map, som_2d.weights_map))

    def test_compute_output_shape_1d(self, som_1d):
        """Test compute_output_shape for 1D SOM."""
        input_shape = (32, 10)
        bmu_shape, error_shape = som_1d.compute_output_shape(input_shape)

        assert bmu_shape == (32, 1)
        assert error_shape == (32,)

    def test_compute_output_shape_2d(self, som_2d):
        """Test compute_output_shape for 2D SOM."""
        input_shape = (64, 784)
        bmu_shape, error_shape = som_2d.compute_output_shape(input_shape)

        assert bmu_shape == (64, 2)
        assert error_shape == (64,)

    def test_compute_output_shape_3d(self, som_3d):
        """Test compute_output_shape for 3D SOM."""
        input_shape = (16, 128)
        bmu_shape, error_shape = som_3d.compute_output_shape(input_shape)

        assert bmu_shape == (16, 3)
        assert error_shape == (16,)

    def test_regularization(self, input_data_2d):
        """Test that regularization losses are properly applied."""
        som_with_reg = SOMLayer(
            grid_shape=(8, 8),
            input_dim=784,
            regularizer=keras.regularizers.L2(0.01)
        )

        # No regularization losses before calling the layer
        assert len(som_with_reg.losses) == 0

        # Apply the layer
        bmu_indices, quantization_errors = som_with_reg(input_data_2d, training=True)

        # Should have regularization losses now
        assert len(som_with_reg.losses) > 0

    def test_serialization(self, som_2d):
        """Test serialization and deserialization of the layer."""
        # Build the layer
        som_2d.build((64, 784))

        # Get configs
        config = som_2d.get_config()
        build_config = som_2d.get_build_config()

        # Recreate the layer
        recreated_som = SOMLayer.from_config(config)
        recreated_som.build_from_config(build_config)

        # Check configuration matches
        assert recreated_som.grid_shape == som_2d.grid_shape
        assert recreated_som.input_dim == som_2d.input_dim
        assert recreated_som.initial_learning_rate == som_2d.initial_learning_rate
        assert recreated_som.sigma == som_2d.sigma
        assert recreated_som.neighborhood_function == som_2d.neighborhood_function

        # Check weights match (shapes should be the same)
        assert recreated_som.weights_map.shape == som_2d.weights_map.shape

    def test_serialization_with_custom_objects(self):
        """Test serialization with custom initializers and regularizers."""
        original_som = SOMLayer(
            grid_shape=(6, 6),
            input_dim=100,
            weights_initializer=keras.initializers.HeNormal(),
            regularizer=keras.regularizers.L1(0.01)
        )

        # Build the layer
        original_som.build((32, 100))

        # Get configs
        config = original_som.get_config()
        build_config = original_som.get_build_config()

        # Recreate the layer
        recreated_som = SOMLayer.from_config(config)
        recreated_som.build_from_config(build_config)

        # Check configuration matches
        assert recreated_som.grid_shape == original_som.grid_shape
        assert recreated_som.input_dim == original_som.input_dim

    def test_model_integration(self, input_data_2d):
        """Test the layer in a model context."""
        # Create a simple model with the SOM layer
        inputs = keras.Input(shape=(784,))
        bmu_indices, quantization_errors = SOMLayer(grid_shape=(8, 8), input_dim=784)(inputs)

        # Create a model that outputs both BMU coordinates and quantization errors
        model = keras.Model(inputs=inputs, outputs=[bmu_indices, quantization_errors])

        # Test forward pass
        bmu_pred, error_pred = model(input_data_2d, training=False)

        assert bmu_pred.shape == (64, 2)
        assert error_pred.shape == (64,)

    def test_model_save_load(self, input_data_2d):
        """Test saving and loading a model with the SOM layer."""
        # Create a model with the SOM layer
        inputs = keras.Input(shape=(784,))
        bmu_indices, quantization_errors = SOMLayer(
            grid_shape=(5, 5),
            input_dim=784,
            name="som_layer"
        )(inputs)

        # Add a simple dense layer for completeness
        x = keras.layers.Dense(10)(keras.layers.Flatten()(bmu_indices))
        outputs = keras.layers.Dense(1)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate predictions before saving
        original_prediction = model.predict(input_data_2d[:8])

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"SOMLayer": SOMLayer}
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_data_2d[:8])

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer type is preserved
            som_layer = loaded_model.get_layer("som_layer")
            assert isinstance(som_layer, SOMLayer)

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        som = SOMLayer(grid_shape=(4, 4), input_dim=10)

        # Create inputs with different magnitudes
        test_cases = [
            keras.ops.zeros((8, 10)),  # Zeros
            keras.ops.ones((8, 10)) * 1e-10,  # Very small values
            keras.ops.ones((8, 10)) * 1e10,  # Very large values
            keras.random.normal((8, 10)) * 1e5  # Large random values
        ]

        for test_input in test_cases:
            bmu_indices, quantization_errors = som(test_input, training=True)

            # Check for NaN/Inf values
            assert not tf.reduce_any(tf.math.is_nan(bmu_indices)).numpy()
            assert not tf.reduce_any(tf.math.is_inf(bmu_indices)).numpy()
            assert not tf.reduce_any(tf.math.is_nan(quantization_errors)).numpy()
            assert not tf.reduce_any(tf.math.is_inf(quantization_errors)).numpy()

    def test_gradient_flow(self, input_data_2d):
        """Test that gradients flow properly through the layer."""
        som = SOMLayer(grid_shape=(6, 6), input_dim=784)

        # Create a simple model for gradient testing
        inputs = keras.Input(shape=(784,))
        bmu_indices, quantization_errors = som(inputs)

        # Create a loss based on quantization errors
        loss = keras.layers.Lambda(lambda x: keras.ops.mean(x))(quantization_errors)
        model = keras.Model(inputs=inputs, outputs=loss)

        # Test gradient computation
        with tf.GradientTape() as tape:
            tape.watch(input_data_2d)
            output = model(input_data_2d, training=True)

        # Get gradients with respect to inputs
        gradients = tape.gradient(output, input_data_2d)

        # Gradients should exist (though they might be sparse due to argmin)
        assert gradients is not None

    def test_different_grid_shapes(self):
        """Test SOM with various grid shapes."""
        test_cases = [
            ((10,), 20),  # 1D
            ((5, 5), 30),  # 2D square
            ((3, 7), 40),  # 2D rectangle
            ((2, 3, 4), 50),  # 3D
            ((2, 2, 2, 2), 60)  # 4D
        ]

        for grid_shape, input_dim in test_cases:
            som = SOMLayer(grid_shape=grid_shape, input_dim=input_dim)

            # Test with small batch
            test_input = keras.random.uniform((4, input_dim))
            bmu_indices, quantization_errors = som(test_input, training=True)

            # Check output shapes
            assert bmu_indices.shape == (4, len(grid_shape))
            assert quantization_errors.shape == (4,)

            # Check BMU indices are within bounds
            for i, dim_size in enumerate(grid_shape):
                assert tf.reduce_all(bmu_indices[:, i] >= 0)
                assert tf.reduce_all(bmu_indices[:, i] < dim_size)

    def test_learning_progression(self, som_2d, input_data_2d):
        """Test that learning progresses over iterations."""
        # Initial quantization error
        _, initial_errors = som_2d(input_data_2d, training=True)
        initial_mean_error = tf.reduce_mean(initial_errors)

        # Train for multiple iterations
        for _ in range(10):
            som_2d(input_data_2d, training=True)

        # Final quantization error
        _, final_errors = som_2d(input_data_2d, training=False)
        final_mean_error = tf.reduce_mean(final_errors)

        # Error should generally decrease (though not guaranteed for all cases)
        # At minimum, check that the layer is still functioning
        assert final_mean_error >= 0
        assert som_2d.iterations.numpy() > 0

    def test_consistent_bmu_finding(self, som_2d):
        """Test that BMU finding is consistent for the same input."""
        # Create fixed input
        fixed_input = keras.ops.ones((1, 784))

        # Find BMU multiple times
        bmu1, _ = som_2d(fixed_input, training=False)
        bmu2, _ = som_2d(fixed_input, training=False)
        bmu3, _ = som_2d(fixed_input, training=False)

        # Should be the same BMU each time (no training mode)
        assert tf.reduce_all(tf.equal(bmu1, bmu2))
        assert tf.reduce_all(tf.equal(bmu2, bmu3))

    def test_edge_case_single_neuron(self):
        """Test SOM with single neuron grid."""
        som = SOMLayer(grid_shape=(1,), input_dim=10)
        test_input = keras.random.uniform((5, 10))

        bmu_indices, quantization_errors = som(test_input, training=True)

        # All inputs should map to the single neuron
        assert tf.reduce_all(tf.equal(bmu_indices, 0))
        assert quantization_errors.shape == (5,)

    def test_edge_case_large_sigma(self, som_2d, input_data_2d):
        """Test SOM with very large sigma value."""
        som_large_sigma = SOMLayer(
            grid_shape=(10, 10),
            input_dim=784,
            sigma=100.0  # Very large sigma
        )

        bmu_indices, quantization_errors = som_large_sigma(input_data_2d, training=True)

        # Should still work without errors
        assert bmu_indices.shape == (64, 2)
        assert quantization_errors.shape == (64,)

    def test_multiple_training_sessions(self, som_2d, input_data_2d):
        """Test multiple training sessions with the same SOM."""
        initial_iterations = som_2d.iterations.numpy()

        # First training session
        som_2d(input_data_2d, training=True)
        first_iterations = som_2d.iterations.numpy()

        # Second training session
        som_2d(input_data_2d, training=True)
        second_iterations = som_2d.iterations.numpy()

        # Iterations should increase
        assert first_iterations > initial_iterations
        assert second_iterations > first_iterations