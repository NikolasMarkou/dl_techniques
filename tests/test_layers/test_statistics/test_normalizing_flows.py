"""
Comprehensive test suite for normalizing flow layers.

This test suite follows modern Keras 3 testing best practices and ensures
robust functionality, serialization compatibility, and numerical accuracy
for both AffineCouplingLayer and NormalizingFlowLayer components.
"""

import pytest
import tempfile
import os
import numpy as np
from typing import Any, Dict, List, Tuple

import keras
from keras import ops
import tensorflow as tf

# Import the normalizing flow layers
# Assuming they're in dl_techniques.layers.statistics.normalizing_flow
from dl_techniques.layers.statistics.normalizing_flow import (
    AffineCouplingLayer,
    NormalizingFlowLayer,
    EPSILON_CONSTANT
)


class TestAffineCouplingLayer:
    """Comprehensive test suite for AffineCouplingLayer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'input_dim': 6,
            'context_dim': 4,
            'hidden_units': 32,
            'reverse': False,
            'activation': 'relu',
            'use_tanh_stabilization': True
        }

    @pytest.fixture
    def sample_data(self) -> keras.KerasTensor:
        """Sample input data for testing."""
        return keras.random.normal(shape=(8, 6), seed=42)

    @pytest.fixture
    def sample_context(self) -> keras.KerasTensor:
        """Sample context data for testing."""
        return keras.random.normal(shape=(8, 4), seed=123)

    def test_initialization(self, layer_config: Dict[str, Any]) -> None:
        """Test layer initialization with valid parameters."""
        layer = AffineCouplingLayer(**layer_config)

        # Check attributes are set correctly
        assert layer.input_dim == 6
        assert layer.context_dim == 4
        assert layer.hidden_units == 32
        assert layer.reverse is False
        assert layer.activation == 'relu'
        assert layer.use_tanh_stabilization is True
        assert layer.split_dim == 3  # input_dim // 2

        # Check layer not built yet
        assert not layer.built

        # Check transformation network exists
        assert layer.transformation_net is not None
        assert isinstance(layer.transformation_net, keras.Sequential)

    def test_initialization_edge_cases(self) -> None:
        """Test initialization with edge cases and error conditions."""
        # Valid minimum case
        layer = AffineCouplingLayer(input_dim=2, context_dim=1, hidden_units=1)
        assert layer.input_dim == 2
        assert layer.context_dim == 1
        assert layer.split_dim == 1

    def test_initialization_errors(self) -> None:
        """Test initialization error conditions."""
        with pytest.raises(ValueError, match="input_dim must be >= 2"):
            AffineCouplingLayer(input_dim=1, context_dim=4)

        with pytest.raises(ValueError, match="context_dim must be >= 1"):
            AffineCouplingLayer(input_dim=6, context_dim=0)

        with pytest.raises(ValueError, match="hidden_units must be >= 1"):
            AffineCouplingLayer(input_dim=6, context_dim=4, hidden_units=0)

    def test_build_process(
        self,
        layer_config: Dict[str, Any],
        sample_data: keras.KerasTensor,
        sample_context: keras.KerasTensor
    ) -> None:
        """Test the build process and layer building."""
        layer = AffineCouplingLayer(**layer_config)

        # Layer should not be built initially
        assert not layer.built

        # Build with correct input shapes
        input_shapes = [sample_data.shape, sample_context.shape]
        layer.build(input_shapes)

        # Layer should be built now
        assert layer.built
        assert layer.transformation_net.built

    def test_build_errors(self, layer_config: Dict[str, Any]) -> None:
        """Test build method error conditions."""
        layer = AffineCouplingLayer(**layer_config)

        # Invalid input shapes
        with pytest.raises(ValueError, match="input_shapes must be a list of two shape tuples"):
            layer.build([(None, 6)])  # Only one shape

        with pytest.raises(ValueError, match="input_shapes must be a list of two shape tuples"):
            layer.build((None, 6))  # Not a list

    def test_forward_transformation(
        self,
        layer_config: Dict[str, Any],
        sample_data: keras.KerasTensor,
        sample_context: keras.KerasTensor
    ) -> None:
        """Test forward transformation z → y."""
        layer = AffineCouplingLayer(**layer_config)

        # Build layer first to ensure it's built
        layer.build([sample_data.shape, sample_context.shape])

        # Forward pass
        y = layer.forward(sample_data, sample_context)

        # Check output properties
        assert y.shape == sample_data.shape
        assert y.dtype == sample_data.dtype
        assert layer.built  # Should be built

        # Check values are different (transformation occurred)
        y_np = keras.ops.convert_to_numpy(y)
        data_np = keras.ops.convert_to_numpy(sample_data)
        assert not np.allclose(y_np, data_np, rtol=1e-6, atol=1e-6)

    def test_inverse_transformation(
        self,
        layer_config: Dict[str, Any],
        sample_data: keras.KerasTensor,
        sample_context: keras.KerasTensor
    ) -> None:
        """Test inverse transformation y → z with log-determinant."""
        layer = AffineCouplingLayer(**layer_config)

        # Build layer first to ensure it's built
        layer.build([sample_data.shape, sample_context.shape])

        # Inverse pass
        z, log_det_jac = layer.inverse(sample_data, sample_context)

        # Check output properties
        assert z.shape == sample_data.shape
        assert z.dtype == sample_data.dtype
        assert log_det_jac.shape == (sample_data.shape[0],)
        assert log_det_jac.dtype == sample_data.dtype

        # Check layer is built
        assert layer.built

        # Log-determinant should be finite
        assert ops.all(ops.isfinite(log_det_jac))

    def test_invertibility(
        self,
        layer_config: Dict[str, Any],
        sample_data: keras.KerasTensor,
        sample_context: keras.KerasTensor
    ) -> None:
        """Test that forward and inverse are actually inverse operations."""
        layer = AffineCouplingLayer(**layer_config)

        # Forward then inverse
        y = layer.forward(sample_data, sample_context)
        z_reconstructed, _ = layer.inverse(y, sample_context)

        # Should recover original data
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(sample_data),
            keras.ops.convert_to_numpy(z_reconstructed),
            rtol=1e-5, atol=1e-6,
            err_msg="Forward-inverse cycle should recover original data"
        )

        # Inverse then forward
        z, log_det_jac = layer.inverse(sample_data, sample_context)
        y_reconstructed = layer.forward(z, sample_context)

        # Should recover original data
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(sample_data),
            keras.ops.convert_to_numpy(y_reconstructed),
            rtol=1e-4, atol=1e-5,
            err_msg="Inverse-forward cycle should recover original data"
        )

    @pytest.mark.parametrize("reverse", [True, False])
    def test_reverse_parameter(
        self,
        sample_data: keras.KerasTensor,
        sample_context: keras.KerasTensor,
        reverse: bool
    ) -> None:
        """Test behavior with reverse parameter."""
        config = {
            'input_dim': 6,
            'context_dim': 4,
            'hidden_units': 32,
            'reverse': reverse
        }

        layer = AffineCouplingLayer(**config)
        y = layer.forward(sample_data, sample_context)

        # Should still maintain invertibility
        z_reconstructed, _ = layer.inverse(y, sample_context)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(sample_data),
            keras.ops.convert_to_numpy(z_reconstructed),
            rtol=1e-5, atol=1e-6,
            err_msg=f"Invertibility should hold with reverse={reverse}"
        )

    @pytest.mark.parametrize("use_tanh_stabilization", [True, False])
    def test_stabilization_modes(
        self,
        sample_data: keras.KerasTensor,
        sample_context: keras.KerasTensor,
        use_tanh_stabilization: bool
    ) -> None:
        """Test different stabilization modes."""
        config = {
            'input_dim': 6,
            'context_dim': 4,
            'hidden_units': 32,
            'use_tanh_stabilization': use_tanh_stabilization
        }

        layer = AffineCouplingLayer(**config)

        # Should work with both stabilization modes
        y = layer.forward(sample_data, sample_context)
        z, log_det_jac = layer.inverse(y, sample_context)

        # Check no NaN or infinite values
        assert ops.all(ops.isfinite(y))
        assert ops.all(ops.isfinite(z))
        assert ops.all(ops.isfinite(log_det_jac))

    def test_gradients_flow(
        self,
        layer_config: Dict[str, Any],
        sample_data: keras.KerasTensor,
        sample_context: keras.KerasTensor
    ) -> None:
        """Test gradient computation through the layer."""
        layer = AffineCouplingLayer(**layer_config)

        # Build layer first
        layer.build([sample_data.shape, sample_context.shape])

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([sample_data, sample_context])
            y = layer.forward(sample_data, sample_context)
            loss = ops.mean(ops.square(y))

        # Check gradients flow to layer parameters
        layer_gradients = tape.gradient(loss, layer.trainable_variables)
        assert len(layer_gradients) > 0
        assert all(g is not None for g in layer_gradients)

        # Check gradients flow to inputs
        input_gradients = tape.gradient(loss, [sample_data, sample_context])
        assert all(g is not None for g in input_gradients)

        # Clean up persistent tape
        del tape

    def test_serialization_cycle(
        self,
        layer_config: Dict[str, Any],
        sample_data: keras.KerasTensor,
        sample_context: keras.KerasTensor
    ) -> None:
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        data_input = keras.Input(shape=(6,), name='data')
        context_input = keras.Input(shape=(4,), name='context')

        layer = AffineCouplingLayer(**layer_config)
        y = layer.forward(data_input, context_input)

        model = keras.Model([data_input, context_input], y)

        # Get original prediction
        original_pred = model([sample_data, sample_context])

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model([sample_data, sample_context])

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_odd_dimensions_support(self) -> None:
        """Test that the layer works correctly with odd input dimensions after fix."""
        # This test verifies the fix for the shape mismatch bug
        layer_config_odd = {
            'input_dim': 3,  # Odd dimension
            'context_dim': 4,
            'hidden_units': 32
        }

        layer = AffineCouplingLayer(**layer_config_odd)
        sample_data = keras.random.normal((4, 3))
        sample_context = keras.random.normal((4, 4))

        # Build the layer
        layer.build([sample_data.shape, sample_context.shape])

        # This should now work without errors
        y = layer.forward(sample_data, sample_context)
        assert y.shape == sample_data.shape

        # Verify invertibility
        z, _ = layer.inverse(y, sample_context)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(sample_data),
            keras.ops.convert_to_numpy(z),
            rtol=1e-5, atol=1e-6,
            err_msg="Invertibility should hold for odd dimensions"
        )


class TestNormalizingFlowLayer:
    """Comprehensive test suite for NormalizingFlowLayer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'output_dimension': 4,
            'num_flow_steps': 3,
            'context_dim': 8,
            'hidden_units_coupling': 32,
            'activation': 'relu',
            'use_tanh_stabilization': True
        }

    @pytest.fixture
    def sample_data(self) -> keras.KerasTensor:
        """Sample output data for testing."""
        return keras.random.normal(shape=(16, 4), seed=42)

    @pytest.fixture
    def sample_context(self) -> keras.KerasTensor:
        """Sample context data for testing."""
        return keras.random.normal(shape=(16, 8), seed=123)

    def test_initialization(self, layer_config: Dict[str, Any]) -> None:
        """Test layer initialization."""
        layer = NormalizingFlowLayer(**layer_config)

        # Check attributes
        assert layer.output_dim == 4
        assert layer.num_flow_steps == 3
        assert layer.context_dim == 8
        assert layer.hidden_units_coupling == 32

        # Check coupling layers created
        assert len(layer.coupling_layers) == 3
        assert all(isinstance(cl, AffineCouplingLayer) for cl in layer.coupling_layers)

        # Check alternating reverse pattern
        assert layer.coupling_layers[0].reverse is False
        assert layer.coupling_layers[1].reverse is True
        assert layer.coupling_layers[2].reverse is False

        assert not layer.built

    def test_initialization_errors(self) -> None:
        """Test initialization error conditions."""
        with pytest.raises(ValueError, match="output_dimension must be >= 2"):
            NormalizingFlowLayer(output_dimension=1, num_flow_steps=3, context_dim=8)

        with pytest.raises(ValueError, match="num_flow_steps must be >= 1"):
            NormalizingFlowLayer(output_dimension=4, num_flow_steps=0, context_dim=8)

        with pytest.raises(ValueError, match="context_dim must be >= 1"):
            NormalizingFlowLayer(output_dimension=4, num_flow_steps=3, context_dim=0)

        with pytest.raises(ValueError, match="hidden_units_coupling must be >= 1"):
            NormalizingFlowLayer(
                output_dimension=4, num_flow_steps=3,
                context_dim=8, hidden_units_coupling=0
            )

    def test_forward_pass(
        self,
        layer_config: Dict[str, Any],
        sample_data: keras.KerasTensor,
        sample_context: keras.KerasTensor
    ) -> None:
        """Test forward pass (inverse transformation for likelihood)."""
        layer = NormalizingFlowLayer(**layer_config)

        # Forward pass
        z, log_det_jac = layer([sample_data, sample_context])

        # Check output shapes
        assert z.shape == sample_data.shape
        assert log_det_jac.shape == (sample_data.shape[0],)
        assert layer.built

        # Check values are finite
        assert ops.all(ops.isfinite(z))
        assert ops.all(ops.isfinite(log_det_jac))

    def test_loss_function(
        self,
        layer_config: Dict[str, Any],
        sample_data: keras.KerasTensor,
        sample_context: keras.KerasTensor
    ) -> None:
        """Test loss function computation."""
        layer = NormalizingFlowLayer(**layer_config)

        # Get layer outputs
        z, log_det_jac = layer([sample_data, sample_context])

        # Compute loss
        loss = layer.loss_func(sample_data, (z, log_det_jac))

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert ops.isfinite(loss)

        # Loss should be reasonable (not too extreme)
        loss_value = keras.ops.convert_to_numpy(loss)
        assert -1000 < loss_value < 1000

    def test_sampling(
        self,
        layer_config: Dict[str, Any],
        sample_context: keras.KerasTensor
    ) -> None:
        """Test sampling from the learned distribution."""
        layer = NormalizingFlowLayer(**layer_config)

        # Build layer first with proper shapes
        dummy_data_shape = (sample_context.shape[0], layer.output_dim)
        layer.build([dummy_data_shape, sample_context.shape])

        # Check layer is built
        assert layer.built
        assert len(layer.coupling_layers) == layer_config['num_flow_steps']

        # Test sampling implementation
        num_samples = 5
        samples = layer.sample(num_samples, sample_context)
        expected_shape = (sample_context.shape[0], num_samples, layer.output_dim)
        assert samples.shape == expected_shape
        assert ops.all(ops.isfinite(samples))

    def test_sampling_errors(
        self,
        layer_config: Dict[str, Any],
        sample_context: keras.KerasTensor
    ) -> None:
        """Test sampling error conditions."""
        layer = NormalizingFlowLayer(**layer_config)
        layer.build([sample_context.shape[:-1] + (4,), sample_context.shape])

        with pytest.raises(ValueError, match="num_samples must be >= 1"):
            layer.sample(0, sample_context)

    def test_call_errors(self, layer_config: Dict[str, Any]) -> None:
        """Test call method error conditions."""
        layer = NormalizingFlowLayer(**layer_config)

        # Wrong number of inputs - this will trigger build error first
        with pytest.raises(ValueError, match="input_shapes must be a list of two shape tuples"):
            dummy_input = keras.random.normal((1, 4))
            layer([dummy_input])  # Only one input

    @pytest.mark.parametrize("num_flow_steps", [1, 2, 4, 6])
    def test_different_flow_depths(
        self,
        sample_data: keras.KerasTensor,
        sample_context: keras.KerasTensor,
        num_flow_steps: int
    ) -> None:
        """Test with different numbers of flow steps."""
        config = {
            'output_dimension': 4,
            'num_flow_steps': num_flow_steps,
            'context_dim': 8,
            'hidden_units_coupling': 16  # Smaller for speed
        }

        layer = NormalizingFlowLayer(**config)

        # Should work with any number of steps
        z, log_det_jac = layer([sample_data, sample_context])

        assert z.shape == sample_data.shape
        assert log_det_jac.shape == (sample_data.shape[0],)
        assert ops.all(ops.isfinite(z))
        assert ops.all(ops.isfinite(log_det_jac))

    def test_gradients_flow(
        self,
        layer_config: Dict[str, Any],
        sample_data: keras.KerasTensor,
        sample_context: keras.KerasTensor
    ) -> None:
        """Test gradient computation."""
        layer = NormalizingFlowLayer(**layer_config)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([sample_data, sample_context])
            z, log_det_jac = layer([sample_data, sample_context])
            loss = layer.loss_func(sample_data, (z, log_det_jac))

        # Check gradients flow to layer parameters
        layer_gradients = tape.gradient(loss, layer.trainable_variables)
        assert len(layer_gradients) > 0
        assert all(g is not None for g in layer_gradients)

        # Check gradients flow to inputs
        input_gradients = tape.gradient(loss, [sample_data, sample_context])
        assert all(g is not None for g in input_gradients)

        # Clean up persistent tape
        del tape

    def test_serialization_cycle(
        self,
        layer_config: Dict[str, Any],
        sample_data: keras.KerasTensor,
        sample_context: keras.KerasTensor
    ) -> None:
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        data_input = keras.Input(shape=(4,), name='data')
        context_input = keras.Input(shape=(8,), name='context')

        layer = NormalizingFlowLayer(**layer_config)
        z, log_det_jac = layer([data_input, context_input])

        model = keras.Model([data_input, context_input], [z, log_det_jac])

        # Get original predictions
        original_z, original_ldj = model([sample_data, sample_context])

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_flow_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_z, loaded_ldj = loaded_model([sample_data, sample_context])

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_z),
                keras.ops.convert_to_numpy(loaded_z),
                rtol=1e-6, atol=1e-6,
                err_msg="Z predictions differ after serialization"
            )

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_ldj),
                keras.ops.convert_to_numpy(loaded_ldj),
                rtol=1e-6, atol=1e-6,
                err_msg="Log-det-jacobian predictions differ after serialization"
            )

    def test_config_completeness(self, layer_config: Dict[str, Any]) -> None:
        """Test that get_config contains all __init__ parameters."""
        layer = NormalizingFlowLayer(**layer_config)
        config = layer.get_config()

        # Check all initialization parameters are present
        for key in layer_config:
            if key == 'output_dimension':
                assert 'output_dimension' in config
            else:
                assert key in config, f"Missing {key} in get_config()"

        # Test config can recreate layer
        new_layer = NormalizingFlowLayer.from_config(config)
        assert new_layer.output_dim == layer.output_dim
        assert new_layer.num_flow_steps == layer.num_flow_steps

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(
        self,
        layer_config: Dict[str, Any],
        sample_data: keras.KerasTensor,
        sample_context: keras.KerasTensor,
        training: bool
    ) -> None:
        """Test behavior in different training modes."""
        layer = NormalizingFlowLayer(**layer_config)

        # Should work in all training modes
        z, log_det_jac = layer([sample_data, sample_context], training=training)

        assert z.shape == sample_data.shape
        assert log_det_jac.shape == (sample_data.shape[0],)

    def test_numerical_stability_extreme_values(self, layer_config: Dict[str, Any]) -> None:
        """Test numerical stability with extreme input values."""
        layer = NormalizingFlowLayer(**layer_config)

        # Test with large values
        large_data = keras.ops.ones((4, 4)) * 10.0
        large_context = keras.ops.ones((4, 8)) * 5.0

        z, log_det_jac = layer([large_data, large_context])

        # Should remain finite
        assert ops.all(ops.isfinite(z))
        assert ops.all(ops.isfinite(log_det_jac))

        # Test with small values
        small_data = keras.ops.ones((4, 4)) * 0.01
        small_context = keras.ops.ones((4, 8)) * 0.01

        z, log_det_jac = layer([small_data, small_context])

        # Should remain finite
        assert ops.all(ops.isfinite(z))
        assert ops.all(ops.isfinite(log_det_jac))


class TestNormalizingFlowIntegration:
    """Integration tests for complete normalizing flow usage."""

    def test_end_to_end_training_setup(self) -> None:
        """Test complete training setup with normalizing flow using GradientTape."""
        # Create a simple regression problem
        input_dim = 10
        output_dim = 2
        context_dim = 16
        batch_size = 32

        # Model inputs
        inputs = keras.Input(shape=(input_dim,), name='features')
        targets = keras.Input(shape=(output_dim,), name='targets')

        # Feature processing
        x = keras.layers.Dense(64, activation='relu')(inputs)
        x = keras.layers.Dense(64, activation='relu')(x)
        context = keras.layers.Dense(context_dim, name='context')(x)

        # Normalizing flow
        flow = NormalizingFlowLayer(
            output_dimension=output_dim,
            num_flow_steps=4,
            context_dim=context_dim,
            hidden_units_coupling=64
        )

        z, log_det_jac = flow([targets, context])

        # Create training model
        training_model = keras.Model([inputs, targets], [z, log_det_jac])

        # Test with dummy data
        x_data = keras.random.normal((batch_size, input_dim))
        y_data = keras.random.normal((batch_size, output_dim))

        # Use GradientTape to verify trainability, which is the standard
        # approach for models with complex, multi-output loss functions.
        with tf.GradientTape() as tape:
            z_pred, ldj_pred = training_model([x_data, y_data], training=True)
            loss = flow.loss_func(y_data, (z_pred, ldj_pred))

        # Check that gradients are computable for all trainable variables
        gradients = tape.gradient(loss, training_model.trainable_variables)
        assert len(gradients) > 0
        assert all(g is not None for g in gradients)
        assert np.isfinite(keras.ops.convert_to_numpy(loss))

    def test_complete_sampling_workflow(self) -> None:
        """Test complete workflow from training to sampling."""
        # Setup - Use odd output_dim to verify fix
        output_dim = 3
        context_dim = 8

        # Create flow
        flow = NormalizingFlowLayer(
            output_dimension=output_dim,
            num_flow_steps=3,
            context_dim=context_dim
        )

        # Build the layer properly
        dummy_data = keras.random.normal((4, output_dim))
        dummy_context = keras.random.normal((4, context_dim))

        # Test likelihood computation (this should build the layer)
        z, log_det_jac = flow([dummy_data, dummy_context])
        loss = flow.loss_func(dummy_data, (z, log_det_jac))

        # Check basic functionality works
        assert z.shape == (4, output_dim)
        assert log_det_jac.shape == (4,)
        assert ops.all(ops.isfinite(z))
        assert ops.all(ops.isfinite(log_det_jac))
        assert np.isfinite(keras.ops.convert_to_numpy(loss))

        # Test sampling now that it is fixed
        test_context = keras.random.normal((10, context_dim))
        samples = flow.sample(5, test_context)
        assert samples.shape == (10, 5, output_dim)
        assert ops.all(ops.isfinite(samples))

    def test_different_activation_functions(self) -> None:
        """Test normalizing flow with different activation functions."""
        activations = ['relu', 'gelu', 'swish', 'tanh']

        for activation in activations:
            flow = NormalizingFlowLayer(
                output_dimension=4,
                num_flow_steps=2,
                context_dim=6,
                activation=activation
            )

            # Test basic functionality
            data = keras.random.normal((8, 4))
            context = keras.random.normal((8, 6))

            z, log_det_jac = flow([data, context])

            # Should work with all activations
            assert ops.all(ops.isfinite(z))
            assert ops.all(ops.isfinite(log_det_jac))

    def test_memory_efficiency_large_batch(self) -> None:
        """Test memory efficiency with larger batches."""
        # Larger batch size test
        batch_size = 128
        output_dim = 8
        context_dim = 16

        flow = NormalizingFlowLayer(
            output_dimension=output_dim,
            num_flow_steps=4,
            context_dim=context_dim,
            hidden_units_coupling=32  # Smaller to save memory
        )

        # Large batch test
        large_data = keras.random.normal((batch_size, output_dim))
        large_context = keras.random.normal((batch_size, context_dim))

        # Should handle large batches without memory issues
        z, log_det_jac = flow([large_data, large_context])

        # Check outputs
        assert z.shape == (batch_size, output_dim)
        assert log_det_jac.shape == (batch_size,)

        # Test sampling with large batch
        samples = flow.sample(3, large_context)
        assert samples.shape == (batch_size, 3, output_dim)