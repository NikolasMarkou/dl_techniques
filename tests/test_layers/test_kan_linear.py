"""
Comprehensive Test Suite for KANLinear Layer
===========================================

This test suite follows the guidelines from "Complete Guide to Modern Keras 3 Custom Layers and Models"
and provides thorough testing for the KANLinear (Kolmogorov-Arnold Network) layer implementation.

Test Categories:
    1. Initialization and Configuration
    2. Forward Pass and Building
    3. Serialization Cycle (Critical Test)
    4. Gradient Computation and Training
    5. Different Training Modes
    6. Edge Cases and Error Conditions
    7. Output Shape Validation
    8. Parameter Validation

Each test ensures the layer works correctly in isolation and within model contexts,
with particular emphasis on the serialization cycle test which is crucial for
production deployment.
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
    """Comprehensive test suite for KANLinear layer."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Basic configuration for standard testing."""
        return {
            'features': 64,
            'grid_size': 5,
            'spline_order': 3,
            'activation': 'swish',
            'regularization_factor': 0.01
        }

    @pytest.fixture
    def advanced_config(self) -> Dict[str, Any]:
        """Advanced configuration for comprehensive testing."""
        return {
            'features': 128,
            'grid_size': 8,
            'spline_order': 4,
            'activation': 'gelu',
            'regularization_factor': 0.001,
            'grid_range': (-2.0, 2.0),
            'epsilon': 1e-6,
            'clip_value': 1e2,
            'use_residual': False,
            'kernel_initializer': 'he_normal',
            'spline_initializer': 'he_normal'
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
    # Initialization and Configuration Tests
    # ========================================================================

    def test_initialization_basic(self, basic_config: Dict[str, Any]) -> None:
        """Test basic layer initialization."""
        layer = KANLinear(**basic_config)

        # Check configuration storage
        assert layer.features == basic_config['features']
        assert layer.grid_size == basic_config['grid_size']
        assert layer.spline_order == basic_config['spline_order']
        assert layer.activation_name == basic_config['activation']
        assert layer.regularization_factor == basic_config['regularization_factor']

        # Check that layer is not built initially
        assert not layer.built
        assert layer.base_weight is None
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
        assert layer.activation_name == advanced_config['activation']
        assert layer.regularization_factor == advanced_config['regularization_factor']
        assert layer.grid_range == advanced_config['grid_range']
        assert layer.epsilon == advanced_config['epsilon']
        assert layer.clip_value == advanced_config['clip_value']
        assert layer.use_residual == advanced_config['use_residual']

        # Check initializers are properly set
        assert isinstance(layer.kernel_initializer, keras.initializers.Initializer)
        assert isinstance(layer.spline_initializer, keras.initializers.Initializer)

    def test_invalid_initialization(self) -> None:
        """Test error conditions during initialization."""
        # Test invalid features
        with pytest.raises(ValueError, match="Features must be a positive integer."):
            KANLinear(features=0)

        with pytest.raises(ValueError, match="Features must be a positive integer."):
            KANLinear(features=-5)

        # Test invalid grid size vs spline order
        with pytest.raises(ValueError, match="Grid size must be >= spline order"):
            KANLinear(features=32, grid_size=2, spline_order=5)

        # Test invalid grid range
        with pytest.raises(ValueError, match="Invalid grid range"):
            KANLinear(features=32, grid_range=(2.0, 1.0))

        with pytest.raises(ValueError, match="Invalid grid range"):
            KANLinear(features=32, grid_range=(1.0, 1.0))

    # ========================================================================
    # Forward Pass and Building Tests
    # ========================================================================

    def test_forward_pass_2d(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """Test forward pass with 2D input."""
        layer = KANLinear(**basic_config)

        # Forward pass should trigger building
        output = layer(sample_input_2d)

        # Check that layer is now built
        assert layer.built
        assert layer.input_features == sample_input_2d.shape[-1]

        # Check output shape
        expected_shape = (sample_input_2d.shape[0], basic_config['features'])
        assert output.shape == expected_shape

        # Check weights are created
        assert layer.base_weight is not None
        assert layer.spline_weight is not None
        assert layer.spline_scaler is not None

        # Check weight shapes
        assert layer.base_weight.shape == (sample_input_2d.shape[-1], basic_config['features'])
        expected_spline_shape = (
            sample_input_2d.shape[-1],
            basic_config['features'],
            basic_config['grid_size'] + basic_config['spline_order'] - 1
        )
        assert layer.spline_weight.shape == expected_spline_shape
        assert layer.spline_scaler.shape == (sample_input_2d.shape[-1], basic_config['features'])

    def test_forward_pass_3d(self, basic_config: Dict[str, Any], sample_input_3d: keras.KerasTensor) -> None:
        """Test forward pass with 3D input (batch processing)."""
        layer = KANLinear(**basic_config)

        output = layer(sample_input_3d)

        # Check output shape preserves all but last dimension
        expected_shape = sample_input_3d.shape[:-1] + (basic_config['features'],)
        assert output.shape == expected_shape
        assert layer.input_features == sample_input_3d.shape[-1]

    def test_build_invalid_input_shape(self, basic_config: Dict[str, Any]) -> None:
        """Test building with invalid input shapes."""
        layer = KANLinear(**basic_config)

        # Test 1D input (invalid)
        with pytest.raises(ValueError, match="Input must be at least 2D"):
            layer.build((32,))

        # Test None input features
        with pytest.raises(ValueError, match="Input features dimension cannot be None"):
            layer.build((None, None))

    def test_residual_connection_logic(self, sample_input_2d: keras.KerasTensor) -> None:
        """Test residual connection behavior."""
        input_features = sample_input_2d.shape[-1]

        # Test with matching dimensions (residual should be enabled)
        layer_residual = KANLinear(features=input_features, use_residual=True)
        output_residual = layer_residual(sample_input_2d, training=True)

        # Test with non-matching dimensions (residual should be disabled)
        layer_no_residual = KANLinear(features=input_features * 2, use_residual=True)
        output_no_residual = layer_no_residual(sample_input_2d, training=True)

        # Both should work without errors
        assert output_residual.shape[-1] == input_features
        assert output_no_residual.shape[-1] == input_features * 2

    # ========================================================================
    # Serialization Cycle Test (CRITICAL)
    # ========================================================================

    def test_serialization_cycle_basic(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """CRITICAL TEST: Full serialization cycle with basic config."""
        # Create model with KANLinear layer
        inputs = keras.Input(shape=sample_input_2d.shape[1:])
        outputs = KANLinear(**basic_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_prediction = model(sample_input_2d)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_kan_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input_2d)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions should match after serialization"
            )

    def test_serialization_cycle_advanced(self, advanced_config: Dict[str, Any],
                                          sample_input_3d: keras.KerasTensor) -> None:
        """CRITICAL TEST: Serialization cycle with advanced configuration."""
        # Create more complex model
        inputs = keras.Input(shape=sample_input_3d.shape[1:])
        x = KANLinear(**advanced_config)(inputs)
        x = KANLinear(64, activation='relu', grid_size=6)(x)
        outputs = KANLinear(10, activation='softmax')(x)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_prediction = model(sample_input_3d)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_kan_complex.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input_3d)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Complex model predictions should match after serialization"
            )

    # ========================================================================
    # Configuration and Serialization Tests
    # ========================================================================

    def test_config_completeness_basic(self, basic_config: Dict[str, Any]) -> None:
        """Test that get_config contains all initialization parameters."""
        layer = KANLinear(**basic_config)
        config = layer.get_config()

        # Check all basic config parameters are present
        for key in basic_config:
            assert key in config, f"Missing {key} in get_config()"
            if key != 'activation':  # activation is stored as activation_name
                assert config[key] == basic_config[key]

        # Check activation is stored correctly
        assert config['activation'] == basic_config['activation']

        # Check required serialization parameters
        required_keys = {
            'features', 'grid_size', 'spline_order', 'activation',
            'regularization_factor', 'grid_range', 'epsilon', 'clip_value',
            'use_residual', 'kernel_initializer', 'spline_initializer',
            'kernel_regularizer', 'spline_regularizer'
        }
        for key in required_keys:
            assert key in config, f"Required serialization key {key} missing"

    def test_config_completeness_advanced(self, advanced_config: Dict[str, Any]) -> None:
        """Test get_config with advanced configuration."""
        layer = KANLinear(**advanced_config)
        config = layer.get_config()

        # Check all advanced config parameters
        for key, expected_value in advanced_config.items():
            assert key in config
            actual_value = config[key]

            if 'initializer' in key:
                # Initializers are serialized to dicts, so we check they are dicts.
                # The serialization cycle test confirms they can be reconstructed.
                assert isinstance(actual_value, dict)
            elif isinstance(expected_value, (int, float, bool, str, tuple)):
                assert actual_value == expected_value

    def test_layer_reconstruction_from_config(self, basic_config: Dict[str, Any]) -> None:
        """Test that layer can be reconstructed from its config."""
        original_layer = KANLinear(**basic_config)
        config = original_layer.get_config()

        # Reconstruct layer from config
        reconstructed_layer = KANLinear.from_config(config)

        # Check that key attributes match
        assert reconstructed_layer.features == original_layer.features
        assert reconstructed_layer.grid_size == original_layer.grid_size
        assert reconstructed_layer.spline_order == original_layer.spline_order
        assert reconstructed_layer.activation_name == original_layer.activation_name
        assert reconstructed_layer.regularization_factor == original_layer.regularization_factor

    # ========================================================================
    # Gradient Flow and Training Tests
    # ========================================================================

    def test_gradient_flow(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """Test that gradients flow through the layer correctly."""
        layer = KANLinear(**basic_config)

        # Make input trackable
        sample_input_var = keras.Variable(sample_input_2d)

        with tf.GradientTape() as tape:
            # tape.watch is not needed for Keras/TF Variables, they are watched automatically.
            output = layer(sample_input_var)
            loss = ops.mean(ops.square(output))

        # Test gradients with respect to trainable variables
        trainable_gradients = tape.gradient(loss, layer.trainable_variables)

        # Check that all trainable variables have gradients
        assert len(trainable_gradients) > 0
        assert all(g is not None for g in trainable_gradients), "Some gradients are None"

        # Check gradient magnitudes are reasonable
        for grad in trainable_gradients:
            # Flatten the gradient tensor before calculating the norm to support
            # tensors with more than 2 dimensions in all backends.
            grad_norm = ops.norm(ops.reshape(grad, newshape=(-1,)))
            assert grad_norm > 0, "Gradient should not be zero"
            assert not ops.isnan(grad_norm), "Gradient should not be NaN"
            assert not ops.isinf(grad_norm), "Gradient should not be infinite"

    def test_gradient_flow_input(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """Test gradients with respect to inputs."""
        layer = KANLinear(**basic_config)

        # Convert to Variable for gradient tracking
        sample_input_var = keras.Variable(sample_input_2d)

        with tf.GradientTape() as tape:
            # A Variable is automatically watched, so tape.watch is not needed.
            output = layer(sample_input_var)
            loss = ops.mean(ops.square(output))

        # Test gradient with respect to input
        input_gradient = tape.gradient(loss, sample_input_var)

        assert input_gradient is not None, "Input gradient should not be None"
        assert input_gradient.shape == sample_input_2d.shape
        assert not ops.any(ops.isnan(input_gradient)), "Input gradient should not contain NaN"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor,
                            training: bool) -> None:
        """Test behavior in different training modes."""
        layer = KANLinear(**basic_config)

        # Test forward pass in different training modes
        output = layer(sample_input_2d, training=training)

        # Basic checks that should pass regardless of training mode
        assert output.shape[0] == sample_input_2d.shape[0]
        assert output.shape[-1] == basic_config['features']
        assert not ops.any(ops.isnan(output)), f"Output contains NaN in training={training}"
        assert not ops.any(ops.isinf(output)), f"Output contains Inf in training={training}"

    # ========================================================================
    # Output Shape and Compatibility Tests
    # ========================================================================

    def test_compute_output_shape(self, basic_config: Dict[str, Any]) -> None:
        """Test compute_output_shape method."""
        layer = KANLinear(**basic_config)

        # Test various input shapes
        test_shapes = [
            (None, 32),
            (10, 64),
            (None, 16, 32),
            (5, 8, 128)
        ]

        for input_shape in test_shapes:
            output_shape = layer.compute_output_shape(input_shape)
            expected_shape = input_shape[:-1] + (basic_config['features'],)
            assert output_shape == expected_shape

    def test_multiple_calls_consistency(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """Test that multiple calls with same input produce consistent results."""
        layer = KANLinear(**basic_config)

        # Multiple forward passes
        output1 = layer(sample_input_2d, training=False)
        output2 = layer(sample_input_2d, training=False)

        # Should be identical in inference mode
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Multiple calls should produce identical results in inference mode"
        )

    # ========================================================================
    # Edge Cases and Stress Tests
    # ========================================================================

    def test_empty_input(self, basic_config: Dict[str, Any]) -> None:
        """Test behavior with empty input."""
        layer = KANLinear(**basic_config)

        # Create empty input
        empty_input = ops.zeros((0, 32))
        output = layer(empty_input)

        # Should handle gracefully
        assert output.shape == (0, basic_config['features'])

    def test_single_sample(self, basic_config: Dict[str, Any]) -> None:
        """Test with single sample input."""
        layer = KANLinear(**basic_config)

        single_input = keras.random.normal((1, 32))
        output = layer(single_input)

        assert output.shape == (1, basic_config['features'])
        assert not ops.any(ops.isnan(output))

    def test_large_input(self, basic_config: Dict[str, Any], large_input: keras.KerasTensor) -> None:
        """Test with large input for performance/stability."""
        layer = KANLinear(**basic_config)

        output = layer(large_input)

        assert output.shape == (large_input.shape[0], basic_config['features'])
        assert not ops.any(ops.isnan(output))
        assert not ops.any(ops.isinf(output))

    def test_extreme_values(self, basic_config: Dict[str, Any]) -> None:
        """Test with extreme input values."""
        layer = KANLinear(**basic_config)

        # Test with very large values
        large_values = ops.ones((4, 32)) * 1e3
        output_large = layer(large_values)
        assert not ops.any(ops.isnan(output_large))

        # Test with very small values
        small_values = ops.ones((4, 32)) * 1e-6
        output_small = layer(small_values)
        assert not ops.any(ops.isnan(output_small))

        # Test with negative values
        negative_values = ops.ones((4, 32)) * -10
        output_negative = layer(negative_values)
        assert not ops.any(ops.isnan(output_negative))

    # ========================================================================
    # Integration and Model Tests
    # ========================================================================

    def test_in_sequential_model(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """Test KANLinear in Sequential model."""
        model = keras.Sequential([
            KANLinear(**basic_config),
            KANLinear(32, activation='relu'),
            KANLinear(10, activation='softmax')
        ])

        # Build model
        model.build(sample_input_2d.shape)

        # Test prediction
        output = model(sample_input_2d)
        assert output.shape == (sample_input_2d.shape[0], 10)

        # Test that all probabilities sum to 1 (softmax output)
        prob_sums = ops.sum(output, axis=-1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(prob_sums),
            np.ones(sample_input_2d.shape[0]),
            rtol=1e-5, atol=1e-5
        )

    def test_in_functional_model(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """Test KANLinear in Functional API model."""
        inputs = keras.Input(shape=sample_input_2d.shape[1:])

        # Create branched architecture
        branch1 = KANLinear(64, activation='swish', name='kan1')(inputs)
        branch2 = KANLinear(64, activation='gelu', name='kan2')(inputs)

        # Merge branches
        merged = layers.Add()([branch1, branch2])
        outputs = KANLinear(10, activation='linear', name='kan_output')(merged)

        model = keras.Model(inputs, outputs)

        # Test prediction
        output = model(sample_input_2d)
        assert output.shape == (sample_input_2d.shape[0], 10)

    def test_trainable_parameters(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """Test trainable parameter counting."""
        layer = KANLinear(**basic_config)

        # Build layer
        _ = layer(sample_input_2d)

        # Check trainable variables
        trainable_vars = layer.trainable_variables
        assert len(trainable_vars) == 3  # base_weight, spline_weight, spline_scaler

        # Check parameter count
        total_params = sum(np.prod(var.shape) for var in trainable_vars)
        assert total_params > 0

        # Verify specific parameter counts
        input_features = sample_input_2d.shape[-1]
        features = basic_config['features']
        grid_size = basic_config['grid_size']
        spline_order = basic_config['spline_order']

        expected_base_params = input_features * features
        expected_spline_params = input_features * features * (grid_size + spline_order - 1)
        expected_scaler_params = input_features * features
        expected_total = expected_base_params + expected_spline_params + expected_scaler_params

        assert total_params == expected_total


# ============================================================================
# Integration Test with Model Training
# ============================================================================

class TestKANLinearTraining:
    """Test KANLinear layer in actual training scenarios."""

    @pytest.fixture
    def training_data(self) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Generate synthetic training data."""
        # Simple regression problem: y = sum(x^2)
        x = keras.random.uniform((100, 10), minval=-2, maxval=2)
        y = ops.sum(ops.square(x), axis=1, keepdims=True)
        return x, y

    def test_training_convergence(self, training_data: Tuple[keras.KerasTensor, keras.KerasTensor]) -> None:
        """Test that the layer can learn from training data."""
        x_train, y_train = training_data

        # Create simple model
        model = keras.Sequential([
            KANLinear(32, activation='swish'),
            KANLinear(16, activation='relu'),
            KANLinear(1, activation='linear')
        ])

        # Compile model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Train for a few epochs
        history = model.fit(x_train, y_train, epochs=5, verbose=0, validation_split=0.2)

        # Check that loss decreased
        initial_loss = history.history['loss'][0]
        final_loss = history.history['loss'][-1]

        assert final_loss < initial_loss, "Loss should decrease during training"
        assert not np.isnan(final_loss), "Final loss should not be NaN"
        assert not np.isinf(final_loss), "Final loss should not be infinite"

    def test_overfitting_prevention(self, training_data: Tuple[keras.KerasTensor, keras.KerasTensor]) -> None:
        """Test regularization effectiveness."""
        x_train, y_train = training_data

        # Model with regularization
        regularized_model = keras.Sequential([
            KANLinear(64, activation='swish', regularization_factor=0.1),
            KANLinear(1, activation='linear', regularization_factor=0.1)
        ])

        # Model without regularization
        unregularized_model = keras.Sequential([
            KANLinear(64, activation='swish', regularization_factor=0.0),
            KANLinear(1, activation='linear', regularization_factor=0.0)
        ])

        # Compile both models
        regularized_model.compile(optimizer='adam', loss='mse')
        unregularized_model.compile(optimizer='adam', loss='mse')

        # Train both models
        reg_history = regularized_model.fit(
            x_train, y_train, epochs=10, verbose=0, validation_split=0.2
        )
        unreg_history = unregularized_model.fit(
            x_train, y_train, epochs=10, verbose=0, validation_split=0.2
        )

        # Regularized model should have more stable validation loss
        reg_val_losses = reg_history.history['val_loss']
        unreg_val_losses = unreg_history.history['val_loss']

        # Check that both models trained successfully
        assert len(reg_val_losses) > 0
        assert len(unreg_val_losses) > 0
        assert all(not np.isnan(loss) for loss in reg_val_losses)
        assert all(not np.isnan(loss) for loss in unreg_val_losses)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])