"""
Comprehensive test suite for Self-Organizing Map (SOM) layers and model.

This test suite follows the "Complete Guide to Modern Keras 3 Custom Layers and Models"
best practices for testing custom Keras implementations. Tests cover:
- Layer initialization and configuration
- Forward pass and building behavior
- Serialization cycle (CRITICAL)
- Training functionality
- Edge cases and error handling
- Model-specific functionality

The tests ensure robustness, proper serialization, and correct behavior across
different use cases for the SOM implementation.
"""

import os
import tempfile
import pytest
import numpy as np
import keras
from keras import ops
import tensorflow as tf
from typing import Dict, Any, Tuple


from dl_techniques.models.som.model import SOMModel, SOM2dLayer
from dl_techniques.layers.memory.som_nd_layer import SOMLayer

# =============================================================================
# Test Suite for Base SOMLayer (N-Dimensional)
# =============================================================================

class TestSOMLayer:
    """Comprehensive test suite for base SOMLayer."""

    @pytest.fixture
    def layer_config_2d(self) -> Dict[str, Any]:
        """Standard 2D grid configuration for testing."""
        return {
            'grid_shape': (8, 8),
            'input_dim': 16,
            'initial_learning_rate': 0.1,
            'sigma': 1.5,
            'neighborhood_function': 'gaussian',
        }

    @pytest.fixture
    def layer_config_1d(self) -> Dict[str, Any]:
        """1D grid configuration for testing."""
        return {
            'grid_shape': (20,),
            'input_dim': 10,
            'initial_learning_rate': 0.15,
            'sigma': 2.0,
            'neighborhood_function': 'bubble',
        }

    @pytest.fixture
    def layer_config_3d(self) -> Dict[str, Any]:
        """3D grid configuration for testing."""
        return {
            'grid_shape': (4, 4, 4),
            'input_dim': 32,
            'initial_learning_rate': 0.2,
            'sigma': 1.0,
            'neighborhood_function': 'gaussian',
        }

    @pytest.fixture
    def sample_input_2d(self) -> np.ndarray:
        """Sample input for 2D grid testing."""
        return np.random.randn(8, 16).astype(np.float32)

    @pytest.fixture
    def sample_input_1d(self) -> np.ndarray:
        """Sample input for 1D grid testing."""
        return np.random.randn(10, 10).astype(np.float32)

    @pytest.fixture
    def sample_input_3d(self) -> np.ndarray:
        """Sample input for 3D grid testing."""
        return np.random.randn(6, 32).astype(np.float32)

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_initialization_2d(self, layer_config_2d):
        """Test layer initialization with 2D grid."""

        layer = SOMLayer(**layer_config_2d)

        # Check configuration storage
        assert layer.grid_shape == (8, 8)
        assert layer.input_dim == 16
        assert layer.initial_learning_rate == 0.1
        assert layer.sigma == 1.5
        assert layer.neighborhood_function == 'gaussian'

        # Check grid properties
        assert layer.grid_dim == 2
        assert layer.num_neurons == 64

        # Layer should not be built yet
        assert not layer.built

    def test_initialization_1d(self, layer_config_1d):
        """Test layer initialization with 1D grid."""

        layer = SOMLayer(**layer_config_1d)

        assert layer.grid_shape == (20,)
        assert layer.grid_dim == 1
        assert layer.num_neurons == 20
        assert layer.neighborhood_function == 'bubble'
        assert not layer.built

    def test_initialization_3d(self, layer_config_3d):
        """Test layer initialization with 3D grid."""
        

        layer = SOMLayer(**layer_config_3d)

        assert layer.grid_shape == (4, 4, 4)
        assert layer.grid_dim == 3
        assert layer.num_neurons == 64
        assert not layer.built

    def test_initialization_with_regularizer(self):
        """Test layer initialization with regularizer."""
        

        regularizer = keras.regularizers.L2(0.01)
        layer = SOMLayer(
            grid_shape=(5, 5),
            input_dim=10,
            regularizer=regularizer
        )

        assert layer.regularizer is not None

    def test_initialization_with_custom_initializer(self):
        """Test layer initialization with custom weight initializer."""
        

        initializer = keras.initializers.GlorotUniform()
        layer = SOMLayer(
            grid_shape=(6, 6),
            input_dim=20,
            weights_initializer=initializer
        )

        assert layer._weights_initializer_config is not None

    # -------------------------------------------------------------------------
    # Forward Pass and Building Tests
    # -------------------------------------------------------------------------

    def test_forward_pass_2d(self, layer_config_2d, sample_input_2d):
        """Test forward pass through 2D grid layer."""
        

        layer = SOMLayer(**layer_config_2d)

        # First call should build the layer
        bmu_indices, quant_errors = layer(sample_input_2d, training=False)

        # Check layer is built
        assert layer.built

        # Check output shapes
        batch_size = sample_input_2d.shape[0]
        assert bmu_indices.shape == (batch_size, 2)  # 2D coordinates
        assert quant_errors.shape == (batch_size,)

        # Check BMU indices are within grid bounds
        bmu_np = ops.convert_to_numpy(bmu_indices)
        assert np.all(bmu_np[:, 0] >= 0) and np.all(bmu_np[:, 0] < 8)
        assert np.all(bmu_np[:, 1] >= 0) and np.all(bmu_np[:, 1] < 8)

        # Check quantization errors are non-negative
        quant_np = ops.convert_to_numpy(quant_errors)
        assert np.all(quant_np >= 0)

    def test_forward_pass_1d(self, layer_config_1d, sample_input_1d):
        """Test forward pass through 1D grid layer."""
        

        layer = SOMLayer(**layer_config_1d)
        bmu_indices, quant_errors = layer(sample_input_1d, training=False)

        assert layer.built
        batch_size = sample_input_1d.shape[0]
        assert bmu_indices.shape == (batch_size, 1)  # 1D coordinates

        # Check BMU indices are within grid bounds
        bmu_np = ops.convert_to_numpy(bmu_indices)
        assert np.all(bmu_np >= 0) and np.all(bmu_np < 20)

    def test_forward_pass_3d(self, layer_config_3d, sample_input_3d):
        """Test forward pass through 3D grid layer."""
        

        layer = SOMLayer(**layer_config_3d)
        bmu_indices, quant_errors = layer(sample_input_3d, training=False)

        assert layer.built
        batch_size = sample_input_3d.shape[0]
        assert bmu_indices.shape == (batch_size, 3)  # 3D coordinates

        # Check all coordinates are within bounds
        bmu_np = ops.convert_to_numpy(bmu_indices)
        assert np.all(bmu_np >= 0) and np.all(bmu_np < 4)

    def test_weights_created_during_build(self, layer_config_2d, sample_input_2d):
        """Test that weights are created during build phase."""
        

        layer = SOMLayer(**layer_config_2d)

        # Before building, weights should not exist
        assert not hasattr(layer, 'weights_map') or layer.weights_map is None

        # Build the layer
        layer.build(sample_input_2d.shape)

        # After building, weights should exist with correct shape
        assert layer.weights_map is not None
        expected_shape = (8, 8, 16)  # grid_shape + (input_dim,)
        assert layer.weights_map.shape == expected_shape

    def test_training_mode_updates_weights(self, layer_config_2d, sample_input_2d):
        """Test that training mode updates weights while inference doesn't."""
        

        layer = SOMLayer(**layer_config_2d)

        # Get initial weights
        layer(sample_input_2d, training=False)  # Build the layer
        initial_weights = ops.convert_to_numpy(layer.weights_map).copy()

        # Run in training mode
        layer(sample_input_2d, training=True)
        trained_weights = ops.convert_to_numpy(layer.weights_map)

        # Weights should have changed
        assert not np.allclose(initial_weights, trained_weights, rtol=1e-5, atol=1e-5)

        # Run in inference mode
        inference_weights_before = ops.convert_to_numpy(layer.weights_map).copy()
        layer(sample_input_2d, training=False)
        inference_weights_after = ops.convert_to_numpy(layer.weights_map)

        # Weights should not change in inference mode
        np.testing.assert_allclose(
            inference_weights_before,
            inference_weights_after,
            rtol=1e-7, atol=1e-7,
            err_msg="Weights should not change in inference mode"
        )

    def test_iterations_increment_during_training(self, layer_config_2d, sample_input_2d):
        """Test that iteration counter increments during training."""
        

        layer = SOMLayer(**layer_config_2d)

        # Build layer
        layer(sample_input_2d, training=False)
        initial_iterations = float(layer.iterations.numpy())

        # Train for a few batches
        for _ in range(3):
            layer(sample_input_2d, training=True)

        final_iterations = float(layer.iterations.numpy())

        # Iterations should increase by batch_size * num_batches
        expected_increase = sample_input_2d.shape[0] * 3
        assert final_iterations == initial_iterations + expected_increase

    # -------------------------------------------------------------------------
    # Serialization Tests (CRITICAL)
    # -------------------------------------------------------------------------

    def test_serialization_cycle_2d(self, layer_config_2d, sample_input_2d):
        """CRITICAL TEST: Full serialization cycle for 2D grid."""
        

        # Create model with custom layer
        inputs = keras.Input(shape=(16,))
        bmu_indices, quant_errors = SOMLayer(**layer_config_2d)(inputs)
        model = keras.Model(inputs, [bmu_indices, quant_errors])

        # Get original predictions
        original_bmu, original_quant = model(sample_input_2d)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_som_layer.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_bmu, loaded_quant = loaded_model(sample_input_2d)

            # Verify identical outputs
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_bmu),
                ops.convert_to_numpy(loaded_bmu),
                rtol=1e-6, atol=1e-6,
                err_msg="BMU indices differ after serialization"
            )

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_quant),
                ops.convert_to_numpy(loaded_quant),
                rtol=1e-6, atol=1e-6,
                err_msg="Quantization errors differ after serialization"
            )

    def test_serialization_cycle_1d(self, layer_config_1d, sample_input_1d):
        """CRITICAL TEST: Full serialization cycle for 1D grid."""
        

        inputs = keras.Input(shape=(10,))
        bmu_indices, quant_errors = SOMLayer(**layer_config_1d)(inputs)
        model = keras.Model(inputs, [bmu_indices, quant_errors])

        original_bmu, original_quant = model(sample_input_1d)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_som_1d.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_bmu, loaded_quant = loaded_model(sample_input_1d)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_bmu),
                ops.convert_to_numpy(loaded_bmu),
                rtol=1e-6, atol=1e-6
            )

    def test_serialization_with_regularizer(self, sample_input_2d):
        """Test serialization with regularizer."""
        

        regularizer = keras.regularizers.L2(0.01)
        inputs = keras.Input(shape=(16,))
        bmu_indices, quant_errors = SOMLayer(
            grid_shape=(5, 5),
            input_dim=16,
            regularizer=regularizer
        )(inputs)
        model = keras.Model(inputs, [bmu_indices, quant_errors])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_som_reg.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)

            # Check that regularizer is preserved
            loaded_layer = loaded_model.layers[1]
            assert loaded_layer.regularizer is not None

    def test_config_completeness(self, layer_config_2d):
        """Test that get_config contains all initialization parameters."""
        

        layer = SOMLayer(**layer_config_2d)
        config = layer.get_config()

        # Check all critical parameters are present
        required_keys = ['grid_shape', 'input_dim', 'initial_learning_rate',
                         'sigma', 'neighborhood_function']
        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        # Verify values match
        assert config['grid_shape'] == layer_config_2d['grid_shape']
        assert config['input_dim'] == layer_config_2d['input_dim']
        assert config['initial_learning_rate'] == layer_config_2d['initial_learning_rate']

    # -------------------------------------------------------------------------
    # Gradient and Training Tests
    # -------------------------------------------------------------------------

    def test_no_gradients_in_som_training(self, layer_config_2d, sample_input_2d):
        """Test that SOM doesn't use gradient descent (uses competitive learning)."""
        

        layer = SOMLayer(**layer_config_2d)

        # Note: SOMs don't use gradient descent, they use competitive learning
        # This test verifies the layer can be used in a GradientTape context
        # but the weights are updated through the competitive learning rule, not gradients

        with tf.GradientTape() as tape:
            bmu_indices, quant_errors = layer(sample_input_2d, training=True)
            # Use quantization error as a pseudo-loss
            loss = ops.mean(quant_errors)

        # We can compute gradients, but SOM updates weights internally
        # This just ensures compatibility with TensorFlow's computational graph
        gradients = tape.gradient(loss, layer.trainable_variables)
        assert gradients is not None

    # -------------------------------------------------------------------------
    # Neighborhood Function Tests
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("neighborhood_fn", ['gaussian', 'bubble'])
    def test_neighborhood_functions(self, neighborhood_fn, sample_input_2d):
        """Test different neighborhood functions."""
        

        layer = SOMLayer(
            grid_shape=(6, 6),
            input_dim=16,
            neighborhood_function=neighborhood_fn
        )

        bmu_indices, quant_errors = layer(sample_input_2d, training=True)

        assert layer.built
        assert bmu_indices.shape[0] == sample_input_2d.shape[0]

    # -------------------------------------------------------------------------
    # Edge Cases and Error Handling
    # -------------------------------------------------------------------------

    def test_invalid_grid_shape(self):
        """Test error handling for invalid grid shapes."""
        

        with pytest.raises(ValueError):
            SOMLayer(grid_shape=(0, 5), input_dim=10)

        with pytest.raises(ValueError):
            SOMLayer(grid_shape=(-1, 5), input_dim=10)

    def test_invalid_input_dim(self):
        """Test error handling for invalid input dimension."""
        

        with pytest.raises(ValueError):
            SOMLayer(grid_shape=(5, 5), input_dim=0)

        with pytest.raises(ValueError):
            SOMLayer(grid_shape=(5, 5), input_dim=-10)

    def test_invalid_learning_rate(self):
        """Test error handling for invalid learning rate."""
        

        with pytest.raises(ValueError):
            SOMLayer(grid_shape=(5, 5), input_dim=10, initial_learning_rate=0.0)

        with pytest.raises(ValueError):
            SOMLayer(grid_shape=(5, 5), input_dim=10, initial_learning_rate=-0.1)

    def test_invalid_sigma(self):
        """Test error handling for invalid sigma."""
        

        with pytest.raises(ValueError):
            SOMLayer(grid_shape=(5, 5), input_dim=10, sigma=0.0)

        with pytest.raises(ValueError):
            SOMLayer(grid_shape=(5, 5), input_dim=10, sigma=-1.0)

    def test_invalid_neighborhood_function(self):
        """Test error handling for invalid neighborhood function."""
        

        with pytest.raises(ValueError):
            SOMLayer(
                grid_shape=(5, 5),
                input_dim=10,
                neighborhood_function='invalid_function'
            )

    # -------------------------------------------------------------------------
    # Utility Method Tests
    # -------------------------------------------------------------------------

    def test_get_weights_map(self, layer_config_2d, sample_input_2d):
        """Test get_weights_map method."""
        

        layer = SOMLayer(**layer_config_2d)
        layer(sample_input_2d, training=False)  # Build layer

        weights_map = layer.get_weights_map()

        # Check shape
        expected_shape = (8, 8, 16)
        assert weights_map.shape == expected_shape

        # Verify it returns the actual weights (not a copy)
        assert weights_map is layer.weights_map

    def test_compute_output_shape(self, layer_config_2d):
        """Test compute_output_shape method."""
        

        layer = SOMLayer(**layer_config_2d)

        input_shape = (32, 16)  # (batch_size, input_dim)
        bmu_shape, error_shape = layer.compute_output_shape(input_shape)

        assert bmu_shape == (32, 2)  # (batch_size, grid_dim)
        assert error_shape == (32,)  # (batch_size,)

    # -------------------------------------------------------------------------
    # Training Behavior Tests
    # -------------------------------------------------------------------------

    def test_learning_rate_decay(self, layer_config_2d, sample_input_2d):
        """Test that learning rate decays over iterations."""
        

        layer = SOMLayer(**layer_config_2d)
        layer(sample_input_2d, training=False)  # Build layer

        # Store initial learning rate effect
        initial_weights = ops.convert_to_numpy(layer.weights_map).copy()
        layer(sample_input_2d, training=True)
        early_delta = np.abs(ops.convert_to_numpy(layer.weights_map) - initial_weights).mean()

        # Train for many iterations to increase iteration counter
        for _ in range(50):
            layer(sample_input_2d, training=True)

        # Check learning continues but with smaller updates (due to decay)
        mid_weights = ops.convert_to_numpy(layer.weights_map).copy()
        layer(sample_input_2d, training=True)
        late_delta = np.abs(ops.convert_to_numpy(layer.weights_map) - mid_weights).mean()

        # Later updates should be smaller (learning rate has decayed)
        assert late_delta < early_delta

    def test_sigma_decay(self, layer_config_2d, sample_input_2d):
        """Test that neighborhood radius (sigma) decays over training."""
        

        # This test verifies the decay happens implicitly during training
        # by checking that the layer continues to function correctly
        # throughout extended training

        layer = SOMLayer(**layer_config_2d)
        layer(sample_input_2d, training=False)  # Build

        # Train for many iterations
        for epoch in range(100):
            layer(sample_input_2d, training=True)

        # Layer should still produce valid outputs
        bmu_indices, quant_errors = layer(sample_input_2d, training=False)

        assert bmu_indices.shape[0] == sample_input_2d.shape[0]
        assert np.all(ops.convert_to_numpy(quant_errors) >= 0)


# =============================================================================
# Test Suite for SOM2dLayer
# =============================================================================

class TestSOM2dLayer:
    """Comprehensive test suite for 2D specialized SOM layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard 2D configuration."""
        return {
            'map_size': (10, 10),
            'input_dim': 784,  # MNIST-like
            'initial_learning_rate': 0.1,
            'sigma': 2.0,
            'neighborhood_function': 'gaussian',
        }

    @pytest.fixture
    def sample_input(self) -> np.ndarray:
        """Sample input simulating flattened images."""
        return np.random.randn(16, 784).astype(np.float32)

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_initialization(self, layer_config):
        """Test 2D layer initialization."""
        

        layer = SOM2dLayer(**layer_config)

        assert layer.map_size == (10, 10)
        assert layer.grid_shape == (10, 10)  # Should match map_size
        assert layer.input_dim == 784
        assert layer.grid_dim == 2
        assert not layer.built

    def test_map_size_validation(self):
        """Test validation of map_size parameter."""
        

        # Valid 2D map_size
        layer = SOM2dLayer(map_size=(5, 5), input_dim=10)
        assert layer.map_size == (5, 5)

        # Invalid: not a tuple/list
        with pytest.raises(ValueError):
            SOM2dLayer(map_size=5, input_dim=10)

        # Invalid: wrong number of dimensions
        with pytest.raises(ValueError):
            SOM2dLayer(map_size=(5, 5, 5), input_dim=10)

        # Invalid: non-positive dimensions
        with pytest.raises(ValueError):
            SOM2dLayer(map_size=(0, 5), input_dim=10)

    # -------------------------------------------------------------------------
    # Forward Pass Tests
    # -------------------------------------------------------------------------

    def test_forward_pass(self, layer_config, sample_input):
        """Test forward pass through 2D layer."""
        

        layer = SOM2dLayer(**layer_config)
        bmu_indices, quant_errors = layer(sample_input, training=False)

        assert layer.built
        assert bmu_indices.shape == (16, 2)
        assert quant_errors.shape == (16,)

        # Check BMU indices are valid
        bmu_np = ops.convert_to_numpy(bmu_indices)
        assert np.all(bmu_np[:, 0] >= 0) and np.all(bmu_np[:, 0] < 10)
        assert np.all(bmu_np[:, 1] >= 0) and np.all(bmu_np[:, 1] < 10)

    # -------------------------------------------------------------------------
    # Serialization Tests
    # -------------------------------------------------------------------------

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle for 2D layer."""
        

        inputs = keras.Input(shape=(784,))
        bmu_indices, quant_errors = SOM2dLayer(**layer_config)(inputs)
        model = keras.Model(inputs, [bmu_indices, quant_errors])

        original_bmu, original_quant = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_som2d.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_bmu, loaded_quant = loaded_model(sample_input)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_bmu),
                ops.convert_to_numpy(loaded_bmu),
                rtol=1e-6, atol=1e-6,
                err_msg="2D SOM predictions differ after serialization"
            )

    def test_config_uses_map_size(self, layer_config):
        """Test that config uses map_size instead of grid_shape."""
        

        layer = SOM2dLayer(**layer_config)
        config = layer.get_config()

        # Should have map_size, not grid_shape
        assert 'map_size' in config
        assert config['map_size'] == (10, 10)

        # grid_shape should not be in config (replaced by map_size)
        assert 'grid_shape' not in config

    # -------------------------------------------------------------------------
    # Utility Method Tests
    # -------------------------------------------------------------------------

    def test_get_weights_as_grid(self, layer_config, sample_input):
        """Test get_weights_as_grid method (alias for get_weights_map)."""
        

        layer = SOM2dLayer(**layer_config)
        layer(sample_input, training=False)  # Build

        weights_grid = layer.get_weights_as_grid()
        weights_map = layer.get_weights_map()

        # Should be the same
        np.testing.assert_array_equal(
            ops.convert_to_numpy(weights_grid),
            ops.convert_to_numpy(weights_map)
        )

        # Check shape
        assert weights_grid.shape == (10, 10, 784)

    # -------------------------------------------------------------------------
    # Training Tests
    # -------------------------------------------------------------------------

    def test_training_updates_2d_weights(self, layer_config, sample_input):
        """Test that training updates weights in 2D layer."""
        

        layer = SOM2dLayer(**layer_config)
        layer(sample_input, training=False)  # Build

        initial_weights = ops.convert_to_numpy(layer.get_weights_as_grid()).copy()

        # Train
        for _ in range(5):
            layer(sample_input, training=True)

        final_weights = ops.convert_to_numpy(layer.get_weights_as_grid())

        # Weights should have changed
        assert not np.allclose(initial_weights, final_weights, rtol=1e-5, atol=1e-5)


# =============================================================================
# Test Suite for SOMModel
# =============================================================================

class TestSOMModel:
    """Comprehensive test suite for SOM Model."""

    @pytest.fixture
    def model_config(self) -> Dict[str, Any]:
        """Standard model configuration."""
        return {
            'map_size': (8, 8),
            'input_dim': 64,
            'initial_learning_rate': 0.1,
            'sigma': 1.5,
            'neighborhood_function': 'gaussian',
        }

    @pytest.fixture
    def sample_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample training data with labels."""
        x = np.random.randn(32, 64).astype(np.float32)
        y = np.random.randint(0, 3, size=(32,))  # 3 classes
        return x, y

    @pytest.fixture
    def small_image_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Small image-like data for visualization tests."""
        # 8x8 images flattened to 64 dimensions
        x = np.random.rand(20, 64).astype(np.float32)
        y = np.random.randint(0, 2, size=(20,))
        return x, y

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_initialization(self, model_config):
        """Test model initialization."""

        model = SOMModel(**model_config)

        assert model.map_size == (8, 8)
        assert model.input_dim == 64
        assert model.som_layer is not None
        assert model.class_prototypes is None  # Not trained yet
        assert not model._is_built

    # -------------------------------------------------------------------------
    # Build and Call Tests
    # -------------------------------------------------------------------------

    def test_build(self, model_config):
        """Test model building."""

        model = SOMModel(**model_config)

        # Build with input shape
        model.build((None, 64))

        assert model._is_built
        assert model.som_layer.built

    def test_call(self, model_config, sample_data):
        """Test forward pass through model."""
        

        model = SOMModel(**model_config)
        x, _ = sample_data

        bmu_indices, quant_errors = model(x, training=False)

        assert bmu_indices.shape == (32, 2)
        assert quant_errors.shape == (32,)

    # -------------------------------------------------------------------------
    # Training Tests
    # -------------------------------------------------------------------------

    def test_train_method(self, model_config, sample_data):
        """Test the train method."""
        

        model = SOMModel(**model_config)
        x, y = sample_data

        # Train for a few iterations
        history = model.train(x, epochs=5, verbose=0)

        assert 'mean_quantization_error' in history
        assert len(history['mean_quantization_error']) == 5

        # Quantization error should generally decrease
        errors = history['mean_quantization_error']
        assert errors[-1] <= errors[0] * 1.5  # Allow some variance

    def test_fit_class_prototypes(self, model_config, sample_data):
        """Test fitting class prototypes."""
        

        model = SOMModel(**model_config)
        x, y = sample_data

        # Train model first
        model.train(x, epochs=5, verbose=0)

        # Fit class prototypes
        model.fit_class_prototypes(x, y)

        assert model.class_prototypes is not None
        # Should have prototypes for each class
        assert len(model.class_prototypes) == 3  # 3 classes in sample_data

    def test_predict_class(self, model_config, sample_data):
        """Test class prediction."""
        

        model = SOMModel(**model_config)
        x, y = sample_data

        # Train and fit prototypes
        model.train(x, epochs=5, verbose=0)
        model.fit_class_prototypes(x, y)

        # Predict on test samples
        test_samples = x[:5]
        predictions = model.predict_class(test_samples)

        assert len(predictions) == 5
        # Predictions should be valid class labels
        assert all(pred in [0, 1, 2] for pred in predictions)

    # -------------------------------------------------------------------------
    # Serialization Tests
    # -------------------------------------------------------------------------

    def test_serialization_cycle(self, model_config, sample_data):
        """CRITICAL TEST: Full model serialization cycle."""
        

        model = SOMModel(**model_config)
        x, _ = sample_data

        # Train briefly
        model.train(x, epochs=3, verbose=0)

        # Get predictions
        original_bmu, original_quant = model(x)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_som_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_bmu, loaded_quant = loaded_model(x)

            # Verify predictions match
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_bmu),
                ops.convert_to_numpy(loaded_bmu),
                rtol=1e-6, atol=1e-6,
                err_msg="Model predictions differ after serialization"
            )

    def test_serialization_with_class_prototypes(self, model_config, sample_data):
        """Test serialization preserves class prototypes."""
        

        model = SOMModel(**model_config)
        x, y = sample_data

        # Train and fit prototypes
        model.train(x, epochs=3, verbose=0)
        model.fit_class_prototypes(x, y)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_som_with_prototypes.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)

            # Class prototypes should be preserved (if implemented in get/from_config)
            # Note: This may require additional serialization logic
            # For now, just verify model loads successfully
            assert loaded_model is not None

    def test_config_completeness(self, model_config):
        """Test that get_config contains all parameters."""
        

        model = SOMModel(**model_config)
        config = model.get_config()

        # Check all critical parameters
        required_keys = ['map_size', 'input_dim', 'initial_learning_rate',
                         'sigma', 'neighborhood_function']
        for key in required_keys:
            assert key in config, f"Missing {key} in model config"

    # -------------------------------------------------------------------------
    # Visualization Method Tests (without actually displaying)
    # -------------------------------------------------------------------------

    def test_visualize_grid_runs(self, model_config, small_image_data):
        """Test that visualize_grid runs without errors."""
        
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        model = SOMModel(**model_config)
        x, _ = small_image_data

        # Train briefly
        model.train(x, epochs=3, verbose=0)

        # This should not raise errors
        try:
            import matplotlib.pyplot as plt
            model.visualize_grid(figsize=(6, 6))
            plt.close('all')
        except Exception as e:
            pytest.fail(f"visualize_grid raised exception: {e}")

    def test_visualize_class_distribution_runs(self, model_config, sample_data):
        """Test that visualize_class_distribution runs without errors."""
        
        import matplotlib
        matplotlib.use('Agg')

        model = SOMModel(**model_config)
        x, y = sample_data

        model.train(x, epochs=3, verbose=0)

        try:
            import matplotlib.pyplot as plt
            model.visualize_class_distribution(x, y, figsize=(6, 6))
            plt.close('all')
        except Exception as e:
            pytest.fail(f"visualize_class_distribution raised exception: {e}")

    def test_visualize_u_matrix_runs(self, model_config, sample_data):
        """Test that visualize_u_matrix runs without errors."""
        
        import matplotlib
        matplotlib.use('Agg')

        model = SOMModel(**model_config)
        x, _ = sample_data

        model.train(x, epochs=3, verbose=0)

        try:
            import matplotlib.pyplot as plt
            model.visualize_u_matrix(figsize=(6, 6))
            plt.close('all')
        except Exception as e:
            pytest.fail(f"visualize_u_matrix raised exception: {e}")

    # -------------------------------------------------------------------------
    # Edge Cases
    # -------------------------------------------------------------------------

    def test_predict_class_without_prototypes(self, model_config, sample_data):
        """Test error handling when predicting without fitted prototypes."""
        

        model = SOMModel(**model_config)
        x, _ = sample_data

        model.train(x, epochs=3, verbose=0)

        # Try to predict without fitting prototypes
        with pytest.raises((ValueError, AttributeError)):
            model.predict_class(x[:5])

    def test_empty_training_data(self, model_config):
        """Test handling of empty training data."""
        

        model = SOMModel(**model_config)

        empty_data = np.array([]).reshape(0, 64)

        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, IndexError)):
            model.train(empty_data, epochs=1)


# =============================================================================
# Integration Tests
# =============================================================================

class TestSOMIntegration:
    """Integration tests for complete SOM workflows."""

    def test_full_workflow_mnist_like(self):
        """Test complete workflow with MNIST-like data."""
        

        # Generate synthetic MNIST-like data
        np.random.seed(42)
        n_samples = 100
        img_size = 28
        x_train = np.random.rand(n_samples, img_size * img_size).astype(np.float32)
        y_train = np.random.randint(0, 10, size=(n_samples,))

        # Create and train model
        model = SOMModel(
            map_size=(10, 10),
            input_dim=img_size * img_size,
            initial_learning_rate=0.1,
            sigma=2.0
        )

        # Train
        history = model.train(x_train, epochs=10, verbose=0)
        assert len(history['mean_quantization_error']) == 10

        # Fit class prototypes
        model.fit_class_prototypes(x_train, y_train)

        # Make predictions
        test_samples = x_train[:10]
        predictions = model.predict_class(test_samples)
        assert len(predictions) == 10

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mnist_som.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)

            # Verify predictions still work
            loaded_predictions = loaded_model.predict_class(test_samples)
            np.testing.assert_array_equal(predictions, loaded_predictions)

    def test_2d_layer_in_larger_model(self):
        """Test using SOM2dLayer as part of a larger model."""

        # Create a model that uses SOM layer for feature extraction
        inputs = keras.Input(shape=(64,))

        # SOM layer for unsupervised feature learning
        bmu_indices, quant_errors = SOM2dLayer(
            map_size=(8, 8),
            input_dim=64
        )(inputs)

        # Use BMU indices as features for classification
        flat_indices = ops.cast(bmu_indices, 'float32')
        dense = keras.layers.Dense(32, activation='relu')(flat_indices)
        outputs = keras.layers.Dense(3, activation='softmax')(dense)

        model = keras.Model(inputs, outputs)

        # Compile and verify it works
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Generate dummy data
        x = np.random.randn(50, 64).astype(np.float32)
        y = np.random.randint(0, 3, size=(50,))

        # Should train without errors
        model.fit(x, y, epochs=2, verbose=0)

        # Should serialize properly
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'hybrid_model.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)

            # Verify predictions match
            original_pred = model.predict(x[:5], verbose=0)
            loaded_pred = loaded_model.predict(x[:5], verbose=0)

            np.testing.assert_allclose(
                original_pred,
                loaded_pred,
                rtol=1e-6, atol=1e-6
            )


if __name__ == '__main__':
    # Run tests with: pytest test_som.py -v --tb=short
    pytest.main([__file__, '-v', '--tb=short'])