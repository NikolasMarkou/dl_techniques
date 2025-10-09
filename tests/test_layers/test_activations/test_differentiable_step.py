"""
Comprehensive test suite for the DifferentiableStep layer.

This test suite follows the modern Keras 3 testing patterns from the
"Complete Guide to Modern Keras 3 Custom Layers and Models" ensuring
robust validation of all layer functionality, including its scalar and
per-axis modes.
"""

import pytest
import tempfile
import os
import numpy as np
import keras
from typing import Any, Dict

# Import the layer to test
from dl_techniques.layers.activations.differentiable_step import DifferentiableStep

# TensorFlow for gradient testing
import tensorflow as tf


class TestDifferentiableStep:
    """Comprehensive test suite for the DifferentiableStep layer."""

    @pytest.fixture
    def scalar_config(self) -> Dict[str, Any]:
        """Configuration for scalar mode testing."""
        return {
            'axis': None,
            'slope_initializer': keras.initializers.Constant(2.0),
            'shift_initializer': keras.initializers.Constant(0.5)
        }

    @pytest.fixture
    def per_axis_config(self) -> Dict[str, Any]:
        """Configuration for per-axis mode testing."""
        return {
            'axis': -1,
            'slope_initializer': 'ones',
            'shift_initializer': 'zeros'
        }

    @pytest.fixture
    def sample_input_2d(self) -> keras.KerasTensor:
        """Sample 2D input for basic testing."""
        return keras.ops.convert_to_tensor(
            np.linspace(-2.0, 2.0, 40).reshape(4, 10).astype('float32')
        )

    @pytest.fixture
    def sample_input_3d(self) -> keras.KerasTensor:
        """Sample 3D input for sequence/image testing."""
        return keras.random.normal(shape=(2, 8, 16))

    def test_initialization(self, scalar_config, per_axis_config):
        """Test layer initialization with valid parameters."""
        # Test scalar mode
        scalar_layer = DifferentiableStep(**scalar_config)
        assert scalar_layer.axis is None
        assert not scalar_layer.built

        # Test per-axis mode
        per_axis_layer = DifferentiableStep(**per_axis_config)
        assert per_axis_layer.axis == -1
        assert not per_axis_layer.built

    @pytest.mark.parametrize("config_fixture", ["scalar_config", "per_axis_config"])
    def test_forward_pass(self, config_fixture, sample_input_2d, request):
        """Test forward pass and building for both scalar and per-axis modes."""
        config = request.getfixturevalue(config_fixture)
        layer = DifferentiableStep(**config)

        output = layer(sample_input_2d)

        # Check layer is built
        assert layer.built

        # Check output shape
        assert output.shape == sample_input_2d.shape

        # Check output is in the valid range (0, 1)
        output_np = keras.ops.convert_to_numpy(output)
        assert np.all(output_np > 0), "All outputs should be > 0"
        assert np.all(output_np < 1), "All outputs should be < 1"

    @pytest.mark.parametrize("config_fixture", ["scalar_config", "per_axis_config"])
    def test_serialization_cycle(self, config_fixture, sample_input_2d, request):
        """CRITICAL TEST: Full serialization cycle for both modes."""
        config = request.getfixturevalue(config_fixture)

        # Create model with custom layer
        inputs = keras.Input(shape=sample_input_2d.shape[1:])
        outputs = DifferentiableStep(**config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input_2d)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_2d)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg=f"Predictions differ after serialization for config: {config}"
            )

    @pytest.mark.parametrize("config_fixture", ["scalar_config", "per_axis_config"])
    def test_config_completeness(self, config_fixture, request):
        """Test that get_config contains all __init__ parameters."""
        config = request.getfixturevalue(config_fixture)
        layer = DifferentiableStep(**config)
        retrieved_config = layer.get_config()

        # Check all config parameters are present
        assert 'axis' in retrieved_config
        assert 'slope_initializer' in retrieved_config
        assert 'shift_initializer' in retrieved_config

    def test_gradients_flow(self, per_axis_config, sample_input_2d):
        """Test gradient computation for inputs and trainable variables."""
        layer = DifferentiableStep(**per_axis_config)

        tf_input = tf.convert_to_tensor(keras.ops.convert_to_numpy(sample_input_2d))

        # FIX: Use persistent=True to allow multiple gradient calls
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(tf_input)
            output = layer(tf_input)
            loss = tf.reduce_mean(tf.square(output))

        # Check gradients w.r.t input
        grad_input = tape.gradient(loss, tf_input)
        assert grad_input is not None, "Gradients w.r.t input should not be None"
        assert np.any(grad_input.numpy() != 0), "Input gradients should not be all zero"

        # Check gradients w.r.t trainable variables
        grad_vars = tape.gradient(loss, layer.trainable_variables)
        assert len(grad_vars) == 2  # slope and shift
        for grad in grad_vars:
            assert grad is not None, "Trainable variable gradients should not be None"
            assert np.any(grad.numpy() != 0), "Variable gradients should not be all zero"

    def test_slope_effect(self, sample_input_2d):
        """Test that a higher slope creates a sharper transition."""
        gentle_layer = DifferentiableStep(slope_initializer=keras.initializers.Constant(1.0))
        sharp_layer = DifferentiableStep(slope_initializer=keras.initializers.Constant(10.0))

        gentle_output = gentle_layer(sample_input_2d)
        sharp_output = sharp_layer(sample_input_2d)

        # Sharp output should be closer to the extremes (0 or 1)
        dist_from_center_gentle = keras.ops.mean(keras.ops.abs(gentle_output - 0.5))
        dist_from_center_sharp = keras.ops.mean(keras.ops.abs(sharp_output - 0.5))

        assert keras.ops.convert_to_numpy(dist_from_center_sharp) > \
               keras.ops.convert_to_numpy(dist_from_center_gentle)

    def test_shift_effect(self):
        """Test that the shift parameter correctly moves the step's center."""
        layer = DifferentiableStep(shift_initializer=keras.initializers.Constant(0.5))

        # Input centered at the shift value
        center_input = keras.ops.convert_to_tensor([[0.5]])
        # Input below and above the shift value
        side_inputs = keras.ops.convert_to_tensor([[-1.0, 2.0]])

        center_output = layer(center_input)
        side_outputs = layer(side_inputs)

        # Output at shift center should be ~0.5
        np.testing.assert_allclose(keras.ops.convert_to_numpy(center_output), [[0.5]], atol=1e-7)

        # Output below shift should be < 0.5, above should be > 0.5
        side_outputs_np = keras.ops.convert_to_numpy(side_outputs)
        assert side_outputs_np[0, 0] < 0.5
        assert side_outputs_np[0, 1] > 0.5

    def test_edge_cases(self):
        """Test error conditions for initialization."""
        with pytest.raises(TypeError, match="Expected `axis` to be an int or None"):
            DifferentiableStep(axis='invalid')

    def test_build_validation(self, sample_input_2d):
        """Test build method validation for axis and input shape."""
        # Test out-of-bounds axis
        layer = DifferentiableStep(axis=2)
        with pytest.raises(ValueError, match="Invalid axis"):
            layer.build(sample_input_2d.shape)

        # Test undefined dimension on the specified axis
        layer = DifferentiableStep(axis=-1)
        # FIX: Make regex more robust to match the layer's helpful error message
        with pytest.raises(ValueError, match="dimension for axis .* must be defined"):
            layer.build((4, None))

    def test_numerical_stability(self):
        """Test numerical stability with extreme inputs."""
        layer = DifferentiableStep()

        extreme_inputs = keras.ops.convert_to_tensor([[-1e6, 1e6]])
        output = layer(extreme_inputs)
        output_np = keras.ops.convert_to_numpy(output)

        assert np.all(np.isfinite(output_np))
        np.testing.assert_allclose(output_np, [[0.0, 1.0]], atol=1e-6)


class TestDifferentiableStepIntegration:
    """Integration tests for DifferentiableStep in complete models."""

    @pytest.mark.parametrize("axis_mode", [None, -1])
    def test_in_gating_model(self, axis_mode):
        """Test layer in a complete gating model for both modes."""
        inputs = keras.Input(shape=(64,))
        features = keras.layers.Dense(32, activation='relu')(inputs)

        # Gate controls which features pass through
        gate = DifferentiableStep(axis=axis_mode)(features)

        gated_features = keras.layers.multiply([features, gate])
        outputs = keras.layers.Dense(10, activation='softmax')(gated_features)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        # Test with dummy data
        dummy_x = keras.random.normal((16, 64))
        dummy_y = keras.utils.to_categorical(np.random.randint(0, 10, 16))

        # Should be trainable
        history = model.fit(dummy_x, dummy_y, epochs=1, verbose=0)
        assert len(history.history['loss']) == 1

        # Should produce valid predictions
        predictions = model.predict(dummy_x, verbose=0)
        assert predictions.shape == (16, 10)

    @pytest.mark.parametrize("axis_mode", [None, -1])
    def test_serialization_in_complete_model(self, axis_mode):
        """Test serialization of a complete model with the layer."""
        inputs = keras.Input(shape=(20,))
        x = keras.layers.Dense(32)(inputs)
        outputs = DifferentiableStep(axis=axis_mode)(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')

        test_input = keras.random.normal((8, 20))
        original_output = model.predict(test_input, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'complete_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_output = loaded_model.predict(test_input, verbose=0)

            np.testing.assert_allclose(
                original_output,
                loaded_output,
                rtol=1e-6, atol=1e-6,
                err_msg=f"Model predictions should match after serialization for axis={axis_mode}"
            )