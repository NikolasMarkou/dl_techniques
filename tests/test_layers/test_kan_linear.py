"""
Comprehensive Test Suite for the KANLinear Layer
================================================

This test suite provides thorough testing for the new KANLinear
(Kolmogorov-Arnold Network) layer implementation. It has been updated to
match the new API and features of the provided `kan_linear.py`.

Test Categories:
    1. Initialization and Configuration
    2. Forward Pass and Building
    3. Serialization Cycle (Critical Test)
    4. Gradient Computation and Training
    5. Different Training Modes
    6. Edge Cases and Error Conditions
    7. Output Shape Validation
    8. New Functionality (Grid Updates)
"""

import pytest
import tempfile
import os
import numpy as np
from typing import Any, Dict, Tuple

import keras
from keras import ops, layers
import tensorflow as tf

from dl_techniques.layers.kan_linear import KANLinear


class TestKANLinear:
    """Comprehensive test suite for the KANLinear layer."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Basic configuration for standard testing."""
        return {
            'features': 64,
            'grid_size': 5,
            'spline_order': 3,
            'activation': 'swish',
            'base_trainable': True,
            'spline_trainable': True,
        }

    @pytest.fixture
    def advanced_config(self) -> Dict[str, Any]:
        """Advanced configuration for comprehensive testing."""
        return {
            'features': 128,
            'grid_size': 8,
            'spline_order': 4,
            'activation': 'gelu',
            'grid_range': (-3.0, 3.0),
            'epsilon': 1e-6,
            'kernel_initializer': 'he_normal',
            'base_trainable': False,
            'spline_trainable': True,
        }

    @pytest.fixture
    def sample_input_2d(self) -> keras.KerasTensor:
        """Sample 2D input for testing."""
        return keras.random.normal(shape=(4, 32))

    @pytest.fixture
    def sample_input_3d(self) -> keras.KerasTensor:
        """Sample 3D input for testing batch processing."""
        return keras.random.normal(shape=(2, 8, 16))

    @pytest.fixture
    def large_input(self) -> keras.KerasTensor:
        """Large input for stress testing."""
        return keras.random.normal(shape=(16, 256))

    # ========================================================================
    # 1. Initialization and Configuration Tests
    # ========================================================================

    def test_initialization_basic(self, basic_config: Dict[str, Any]) -> None:
        """Test basic layer initialization."""
        layer = KANLinear(**basic_config)

        # Check configuration storage
        assert layer.features == basic_config['features']
        assert layer.grid_size == basic_config['grid_size']
        assert layer.spline_order == basic_config['spline_order']
        assert layer.base_activation_name == basic_config['activation']
        assert layer.base_trainable == basic_config['base_trainable']
        assert layer.spline_trainable == basic_config['spline_trainable']

        # Check that layer is not built initially
        assert not layer.built
        assert layer.base_scaler is None
        assert layer.spline_weight is None
        assert layer.spline_scaler is None
        assert layer.input_features is None

    def test_initialization_advanced(self, advanced_config: Dict[str, Any]) -> None:
        """Test initialization with advanced configuration."""
        layer = KANLinear(**advanced_config)

        # Check all advanced parameters
        assert layer.features == advanced_config['features']
        assert layer.grid_size == advanced_config['grid_size']
        assert layer.spline_order == advanced_config['spline_order']
        assert layer.base_activation_name == advanced_config['activation']
        assert layer.grid_range == advanced_config['grid_range']
        assert layer.epsilon == advanced_config['epsilon']
        assert layer.base_trainable == advanced_config['base_trainable']

        # Check initializer is properly set
        assert isinstance(layer.kernel_initializer, keras.initializers.Initializer)

    def test_invalid_initialization(self) -> None:
        """Test error conditions during initialization."""
        with pytest.raises(ValueError, match="Features must be a positive integer."):
            KANLinear(features=0)

        with pytest.raises(ValueError, match="Grid size must be a positive integer."):
            KANLinear(features=10, grid_size=0)

        with pytest.raises(ValueError, match="Spline order must be a non-negative integer."):
            KANLinear(features=10, spline_order=-1)

        with pytest.raises(ValueError, match="Invalid grid range: min must be less than max."):
            KANLinear(features=32, grid_range=(2.0, 1.0))

        with pytest.raises(ValueError, match="Invalid grid range: min must be less than max."):
            KANLinear(features=32, grid_range=(1.0, 1.0))

    # ========================================================================
    # 2. Forward Pass and Building Tests
    # ========================================================================

    def test_forward_pass_2d(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """Test forward pass with 2D input."""
        layer = KANLinear(**basic_config)
        output = layer(sample_input_2d)

        # Check that layer is now built
        assert layer.built
        assert layer.input_features == sample_input_2d.shape[-1]

        # Check output shape
        expected_shape = (sample_input_2d.shape[0], basic_config['features'])
        assert output.shape == expected_shape

        # Check weights are created
        assert layer.base_scaler is not None
        assert layer.spline_weight is not None
        assert layer.spline_scaler is not None

        # Check weight shapes
        num_basis_fns = basic_config['grid_size'] + basic_config['spline_order']
        expected_scaler_shape = (sample_input_2d.shape[-1], basic_config['features'])
        expected_spline_shape = (sample_input_2d.shape[-1], basic_config['features'], num_basis_fns)

        assert layer.base_scaler.shape == expected_scaler_shape
        assert layer.spline_scaler.shape == expected_scaler_shape
        assert layer.spline_weight.shape == expected_spline_shape

    def test_forward_pass_3d(self, basic_config: Dict[str, Any], sample_input_3d: keras.KerasTensor) -> None:
        """Test forward pass with 3D input (batch processing)."""
        layer = KANLinear(**basic_config)
        output = layer(sample_input_3d)
        expected_shape = sample_input_3d.shape[:-1] + (basic_config['features'],)
        assert output.shape == expected_shape
        assert layer.input_features == sample_input_3d.shape[-1]

    def test_build_invalid_input_shape(self, basic_config: Dict[str, Any]) -> None:
        """Test building with invalid input shapes."""
        layer = KANLinear(**basic_config)
        with pytest.raises(ValueError, match="Input must be at least 2D"):
            layer.build((32,))
        with pytest.raises(ValueError, match="Input features dimension cannot be None"):
            layer.build((None, None))

    # ========================================================================
    # 3. Serialization Cycle Test (CRITICAL)
    # ========================================================================

    def test_serialization_cycle_basic(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """CRITICAL TEST: Full serialization cycle with basic config."""
        inputs = keras.Input(shape=sample_input_2d.shape[1:])
        outputs = KANLinear(**basic_config)(inputs)
        model = keras.Model(inputs, outputs)

        original_prediction = model(sample_input_2d)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_kan_model.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input_2d)
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions should match after serialization"
            )

    def test_serialization_cycle_advanced(self, advanced_config: Dict[str, Any], sample_input_3d: keras.KerasTensor) -> None:
        """CRITICAL TEST: Serialization cycle with advanced configuration."""
        inputs = keras.Input(shape=sample_input_3d.shape[1:])
        x = KANLinear(**advanced_config)(inputs)
        x = KANLinear(64, activation='relu', grid_size=6)(x)
        outputs = KANLinear(10, activation='softmax')(x)
        model = keras.Model(inputs, outputs)

        original_prediction = model(sample_input_3d)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_kan_complex.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input_3d)
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Complex model predictions should match after serialization"
            )

    # ========================================================================
    # Configuration and Reconstruction Tests
    # ========================================================================

    def test_config_completeness(self, advanced_config: Dict[str, Any]) -> None:
        """Test that get_config contains all initialization parameters."""
        layer = KANLinear(**advanced_config)
        config = layer.get_config()

        # Check that all keys from __init__ are present
        # FIX 1: Check for 'activation' instead of 'base_activation'
        required_keys = {
            'features', 'grid_size', 'spline_order', 'grid_range',
            'activation', 'base_trainable', 'spline_trainable',
            'kernel_initializer', 'epsilon'
        }
        for key in required_keys:
            assert key in config, f"Required serialization key '{key}' missing"

        # Check values
        assert config['features'] == advanced_config['features']
        assert config['activation'] == advanced_config['activation']
        assert config['grid_range'] == advanced_config['grid_range']

    def test_layer_reconstruction_from_config(self, basic_config: Dict[str, Any]) -> None:
        """Test that a layer can be reconstructed from its config."""
        original_layer = KANLinear(**basic_config)
        config = original_layer.get_config()

        reconstructed_layer = KANLinear.from_config(config)

        assert reconstructed_layer.features == original_layer.features
        assert reconstructed_layer.grid_size == original_layer.grid_size
        assert reconstructed_layer.spline_order == original_layer.spline_order
        assert reconstructed_layer.base_activation_name == original_layer.base_activation_name
        assert reconstructed_layer.base_trainable == original_layer.base_trainable

    # ========================================================================
    # 4. Gradient Flow and Training Tests
    # ========================================================================

    def test_gradient_flow(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """Test that gradients flow through all trainable variables."""
        layer = KANLinear(**basic_config)
        sample_input_var = keras.Variable(sample_input_2d)

        with tf.GradientTape() as tape:
            output = layer(sample_input_var)
            loss = ops.mean(ops.square(output))

        trainable_gradients = tape.gradient(loss, layer.trainable_variables)

        assert len(trainable_gradients) == 3 # base_scaler, spline_scaler, spline_weight
        assert all(g is not None for g in trainable_gradients), "Some gradients are None"

        for grad in trainable_gradients:
            grad_norm = ops.norm(ops.reshape(grad, newshape=(-1,)))
            assert grad_norm > 0, "Gradient should not be zero"
            assert not ops.isnan(grad_norm), "Gradient should not be NaN"

    def test_gradient_flow_input(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """Test gradients with respect to inputs."""
        layer = KANLinear(**basic_config)
        sample_input_var = keras.Variable(sample_input_2d)

        with tf.GradientTape() as tape:
            output = layer(sample_input_var)
            loss = ops.mean(ops.square(output))

        input_gradient = tape.gradient(loss, sample_input_var)
        assert input_gradient is not None, "Input gradient should not be None"
        assert input_gradient.shape == sample_input_2d.shape
        assert not ops.any(ops.isnan(input_gradient))

    def test_trainable_parameters(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """Test correct counting and identification of trainable parameters."""
        # Case 1: All components trainable
        layer_all_trainable = KANLinear(**basic_config)
        _ = layer_all_trainable(sample_input_2d)
        assert len(layer_all_trainable.trainable_variables) == 3
        assert len(layer_all_trainable.non_trainable_variables) == 1

        # Case 2: Scalers not trainable
        # FIX 2: Create a modified config instead of overriding kwargs
        config_not_trainable = basic_config.copy()
        config_not_trainable['base_trainable'] = False
        config_not_trainable['spline_trainable'] = False

        layer_some_not_trainable = KANLinear(**config_not_trainable)
        _ = layer_some_not_trainable(sample_input_2d)
        assert len(layer_some_not_trainable.trainable_variables) == 1 # spline_weight only
        assert len(layer_some_not_trainable.non_trainable_variables) == 3

    # ========================================================================
    # 5. Different Training Modes
    # ========================================================================

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor, training: bool) -> None:
        """Test behavior in different training modes."""
        layer = KANLinear(**basic_config)
        output = layer(sample_input_2d, training=training)

        expected_shape = (sample_input_2d.shape[0], basic_config['features'])
        assert output.shape == expected_shape
        assert not ops.any(ops.isnan(output)), f"Output contains NaN in training={training}"

    # ========================================================================
    # 6. Edge Cases and Stress Tests
    # ========================================================================

    def test_empty_input(self, basic_config: Dict[str, Any]) -> None:
        """Test behavior with empty input."""
        layer = KANLinear(**basic_config)
        empty_input = ops.zeros((0, 32))
        output = layer(empty_input)
        assert output.shape == (0, basic_config['features'])

    def test_extreme_values(self, basic_config: Dict[str, Any]) -> None:
        """Test with extreme input values for numerical stability."""
        layer = KANLinear(**basic_config)
        large_values = ops.ones((4, 32)) * 1e4
        output_large = layer(large_values)
        assert not ops.any(ops.isnan(output_large))

        small_values = ops.ones((4, 32)) * -1e4
        output_small = layer(small_values)
        assert not ops.any(ops.isnan(output_small))

    # ========================================================================
    # 7. Output Shape and Compatibility Tests
    # ========================================================================

    def test_compute_output_shape(self, basic_config: Dict[str, Any]) -> None:
        """Test compute_output_shape method."""
        layer = KANLinear(**basic_config)
        test_shapes = [(None, 32), (10, 64), (None, 16, 32)]
        for input_shape in test_shapes:
            output_shape = layer.compute_output_shape(input_shape)
            expected_shape = input_shape[:-1] + (basic_config['features'],)
            assert output_shape == expected_shape

    def test_in_sequential_model(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """Test KANLinear in a Sequential model."""
        # FIX 3: Add a separate Softmax layer for final activation
        model = keras.Sequential([
            keras.Input(shape=sample_input_2d.shape[1:]),
            KANLinear(**basic_config),
            layers.LayerNormalization(),
            KANLinear(10, activation='linear'), # Output logits
            layers.Activation('softmax')      # Apply softmax to logits
        ])
        output = model(sample_input_2d)
        assert output.shape == (sample_input_2d.shape[0], 10)
        prob_sums = ops.sum(output, axis=-1)
        np.testing.assert_allclose(
            ops.convert_to_numpy(prob_sums),
            np.ones(sample_input_2d.shape[0]),
            rtol=1e-5, atol=1e-5
        )

    # ========================================================================
    # 8. New Functionality Tests
    # ========================================================================

    def test_update_grid(self, basic_config: Dict[str, Any]) -> None:
        """Test the update_grid_from_samples method."""
        layer = KANLinear(**basic_config)

        # Build the layer to initialize the grid
        sample_data = keras.random.normal(shape=(100, 10))
        layer.build(sample_data.shape)

        # Store original grid properties
        original_grid = ops.convert_to_numpy(layer.grid)
        original_grid_range = layer.grid_range

        # Create new data with a different distribution
        update_data = keras.random.uniform(shape=(100, 10), minval=10.0, maxval=20.0)

        # Update the grid
        layer.update_grid_from_samples(update_data)

        # Check that grid and range have changed
        new_grid = ops.convert_to_numpy(layer.grid)
        new_grid_range = layer.grid_range

        assert not np.allclose(original_grid, new_grid), "Grid should have been updated."
        assert original_grid_range != new_grid_range, "Grid range should have been updated."
        assert 10.0 <= new_grid_range[0] < 11.0, "New grid min is out of expected range."
        assert 19.0 < new_grid_range[1] <= 20.0, "New grid max is out of expected range."

        # Check invalid input for grid update
        with pytest.raises(ValueError, match="Input 'x' for grid update must be 2D."):
            layer.update_grid_from_samples(ops.zeros((2, 4, 6)))

# ============================================================================
# Integration Test with Model Training
# ============================================================================

class TestKANLinearTraining:
    """Test KANLinear layer in an actual training scenario."""

    @pytest.fixture
    def training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a simple synthetic regression problem."""
        x = np.random.uniform(-2, 2, (200, 10))
        y = np.sum(np.sin(x) + x**2, axis=1, keepdims=True)
        return x, y

    def test_training_convergence(self, training_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test that a model with KANLinear layers can learn."""
        x_train, y_train = training_data

        model = keras.Sequential([
            keras.Input(shape=(x_train.shape[1],)),
            KANLinear(features=32, grid_size=8, activation='swish'),
            KANLinear(features=1, activation='linear')
        ])

        model.compile(optimizer='adam', loss='mse')
        history = model.fit(x_train, y_train, epochs=5, verbose=0)

        # Check that loss decreased
        initial_loss = history.history['loss'][0]
        final_loss = history.history['loss'][-1]
        assert final_loss < initial_loss, "Loss should decrease during training"
        assert not np.isnan(final_loss), "Final loss should not be NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])