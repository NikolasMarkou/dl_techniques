"""
Comprehensive test suite for BiasFreeConv1D and BiasFreeResidualBlock1D layers.

Following modern Keras 3 testing patterns with emphasis on serialization robustness
and gradient flow validation.
"""

import pytest
import tempfile
import os
import numpy as np
from typing import Any, Dict

import keras
import tensorflow as tf

from dl_techniques.layers.bias_free_conv1d import BiasFreeConv1D, BiasFreeResidualBlock1D


class TestBiasFreeConv1D:
    """Comprehensive test suite for BiasFreeConv1D layer."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Basic configuration for testing."""
        return {
            'filters': 32,
            'kernel_size': 3,
            'activation': 'relu',
            'use_batch_norm': True
        }

    @pytest.fixture
    def advanced_config(self) -> Dict[str, Any]:
        """Advanced configuration with regularization."""
        return {
            'filters': 64,
            'kernel_size': 5,
            'activation': 'gelu',
            'kernel_initializer': 'he_normal',
            'kernel_regularizer': keras.regularizers.L2(1e-4),
            'use_batch_norm': True
        }

    @pytest.fixture
    def no_bn_config(self) -> Dict[str, Any]:
        """Configuration without batch normalization."""
        return {
            'filters': 16,
            'kernel_size': 7,
            'activation': 'swish',
            'use_batch_norm': False
        }

    @pytest.fixture
    def sample_input_1d(self) -> keras.KerasTensor:
        """Sample 1D input tensor for testing."""
        return keras.random.normal(shape=(4, 100, 8))  # batch=4, time=100, features=8

    @pytest.fixture
    def large_input_1d(self) -> keras.KerasTensor:
        """Larger 1D input tensor for gradient testing."""
        return keras.random.normal(shape=(2, 500, 16))

    def test_initialization(self, basic_config: Dict[str, Any]) -> None:
        """Test layer initialization and attribute setting."""
        layer = BiasFreeConv1D(**basic_config)

        # Check configuration attributes
        assert layer.filters == basic_config['filters']
        assert layer.kernel_size == basic_config['kernel_size']
        assert layer.activation == basic_config['activation']
        assert layer.use_batch_norm == basic_config['use_batch_norm']

        # Check layer is not built yet
        assert not layer.built

        # Check sub-layers are created
        assert layer.conv is not None
        assert layer.batch_norm is not None
        assert layer.activation_layer is not None

        # Verify conv layer has no bias
        assert not layer.conv.use_bias

        # Verify batch norm has no center (no bias)
        assert not layer.batch_norm.center
        assert layer.batch_norm.scale  # Should still have scale

    def test_initialization_no_batch_norm(self, no_bn_config: Dict[str, Any]) -> None:
        """Test initialization without batch normalization."""
        layer = BiasFreeConv1D(**no_bn_config)

        assert layer.filters == no_bn_config['filters']
        assert not layer.use_batch_norm
        assert layer.batch_norm is None
        assert layer.conv is not None
        assert layer.activation_layer is not None

    def test_forward_pass(self, basic_config: Dict[str, Any], sample_input_1d: keras.KerasTensor) -> None:
        """Test forward pass and automatic building."""
        layer = BiasFreeConv1D(**basic_config)

        # Forward pass should trigger building
        output = layer(sample_input_1d)

        # Check layer is now built
        assert layer.built

        # Check output shape
        expected_shape = (sample_input_1d.shape[0], sample_input_1d.shape[1], basic_config['filters'])
        assert output.shape == expected_shape

        # Check output is not NaN or infinite
        output_np = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_np).all()
        assert not np.isnan(output_np).any()

    def test_forward_pass_no_batch_norm(self, no_bn_config: Dict[str, Any], sample_input_1d: keras.KerasTensor) -> None:
        """Test forward pass without batch normalization."""
        layer = BiasFreeConv1D(**no_bn_config)

        output = layer(sample_input_1d)

        expected_shape = (sample_input_1d.shape[0], sample_input_1d.shape[1], no_bn_config['filters'])
        assert output.shape == expected_shape

        # Output should still be valid
        output_np = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_np).all()

    def test_serialization_cycle(self, basic_config: Dict[str, Any], sample_input_1d: keras.KerasTensor) -> None:
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input_1d.shape[1:])
        outputs = BiasFreeConv1D(**basic_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input_1d)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            # Load model
            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_1d)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_serialization_cycle_advanced_config(self, advanced_config: Dict[str, Any], sample_input_1d: keras.KerasTensor) -> None:
        """Test serialization with advanced configuration."""
        inputs = keras.Input(shape=sample_input_1d.shape[1:])
        outputs = BiasFreeConv1D(**advanced_config)(inputs)
        model = keras.Model(inputs, outputs)

        original_pred = model(sample_input_1d)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_advanced_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_1d)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Advanced config predictions differ after serialization"
            )

    def test_config_completeness(self, basic_config: Dict[str, Any]) -> None:
        """Test that get_config contains all __init__ parameters."""
        layer = BiasFreeConv1D(**basic_config)
        config = layer.get_config()

        # Check all basic config parameters are present
        for key in basic_config:
            assert key in config, f"Missing {key} in get_config()"

        # Check additional default parameters
        assert 'kernel_initializer' in config
        assert 'kernel_regularizer' in config

        # Verify serialized values are correct
        assert config['filters'] == basic_config['filters']
        assert config['kernel_size'] == basic_config['kernel_size']
        assert config['use_batch_norm'] == basic_config['use_batch_norm']

    def test_gradients_flow(self, basic_config: Dict[str, Any], large_input_1d: keras.KerasTensor) -> None:
        """Test gradient computation and flow through the layer."""
        layer = BiasFreeConv1D(**basic_config)

        with tf.GradientTape() as tape:
            tape.watch(large_input_1d)
            output = layer(large_input_1d, training=True)
            loss = keras.ops.mean(keras.ops.square(output))

        # Compute gradients
        gradients = tape.gradient(loss, layer.trainable_variables)

        # Check gradients exist and are not None
        assert gradients is not None
        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

        # Check gradients are not all zeros
        for grad in gradients:
            grad_np = keras.ops.convert_to_numpy(grad)
            assert not np.allclose(grad_np, 0), "Gradient should not be all zeros"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, basic_config: Dict[str, Any], sample_input_1d: keras.KerasTensor, training: bool) -> None:
        """Test behavior in different training modes."""
        layer = BiasFreeConv1D(**basic_config)

        # Forward pass in specified training mode
        output = layer(sample_input_1d, training=training)

        # Basic shape and validity checks
        expected_shape = (sample_input_1d.shape[0], sample_input_1d.shape[1], basic_config['filters'])
        assert output.shape == expected_shape

        output_np = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_np).all()

    def test_different_input_shapes(self, basic_config: Dict[str, Any]) -> None:
        """Test layer with different input shapes."""
        # Test various input shapes
        test_shapes = [
            (1, 50, 1),   # Single batch, 50 timesteps, 1 feature
            (8, 200, 32), # Batch of 8, 200 timesteps, 32 features
            (2, 10, 128)  # Small batch, short sequence, many features
        ]

        for shape in test_shapes:
            # Create a new layer instance for each shape since layers get locked to input shape after building
            layer = BiasFreeConv1D(**basic_config)
            inputs = keras.random.normal(shape=shape)
            output = layer(inputs)

            expected_shape = (shape[0], shape[1], basic_config['filters'])
            assert output.shape == expected_shape

    def test_edge_cases_and_errors(self) -> None:
        """Test error conditions and edge cases."""
        # Test invalid filters
        with pytest.raises(ValueError):
            BiasFreeConv1D(filters=0)  # Zero filters

        with pytest.raises(ValueError):
            BiasFreeConv1D(filters=-5)  # Negative filters

        # Test invalid kernel_size
        with pytest.raises(ValueError):
            BiasFreeConv1D(filters=32, kernel_size=0)  # Zero kernel size

        with pytest.raises(ValueError):
            BiasFreeConv1D(filters=32, kernel_size=-3)  # Negative kernel size

        # Test invalid use_batch_norm type
        with pytest.raises(TypeError):
            BiasFreeConv1D(filters=32, use_batch_norm="invalid")  # String instead of bool

    def test_compute_output_shape(self, basic_config: Dict[str, Any]) -> None:
        """Test output shape computation."""
        layer = BiasFreeConv1D(**basic_config)

        input_shape = (None, 100, 16)  # Batch can be None
        output_shape = layer.compute_output_shape(input_shape)

        expected_shape = (None, 100, basic_config['filters'])
        assert output_shape == expected_shape

    def test_no_activation(self) -> None:
        """Test layer with no activation function."""
        layer = BiasFreeConv1D(filters=16, activation=None)

        inputs = keras.random.normal(shape=(2, 50, 8))
        output = layer(inputs)

        # Should work without activation
        assert output.shape == (2, 50, 16)
        assert layer.activation_layer is None


class TestBiasFreeResidualBlock1D:
    """Comprehensive test suite for BiasFreeResidualBlock1D layer."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Basic configuration for testing."""
        return {
            'filters': 32,
            'kernel_size': 3,
            'activation': 'relu'
        }

    @pytest.fixture
    def advanced_config(self) -> Dict[str, Any]:
        """Advanced configuration with regularization."""
        return {
            'filters': 64,
            'kernel_size': 5,
            'activation': 'gelu',
            'kernel_initializer': 'he_uniform',
            'kernel_regularizer': keras.regularizers.L1L2(l1=1e-5, l2=1e-4)
        }

    @pytest.fixture
    def sample_input_1d(self) -> keras.KerasTensor:
        """Sample input tensor matching the output filters (no shortcut needed)."""
        return keras.random.normal(shape=(4, 100, 32))  # features=32 to match basic_config filters

    @pytest.fixture
    def different_input_1d(self) -> keras.KerasTensor:
        """Input tensor with different feature dimension (shortcut needed)."""
        return keras.random.normal(shape=(4, 100, 16))  # features=16, different from output

    def test_initialization(self, basic_config: Dict[str, Any]) -> None:
        """Test layer initialization and attribute setting."""
        layer = BiasFreeResidualBlock1D(**basic_config)

        # Check configuration attributes
        assert layer.filters == basic_config['filters']
        assert layer.kernel_size == basic_config['kernel_size']
        assert layer.activation == basic_config['activation']

        # Check layer is not built yet
        assert not layer.built

        # Check sub-layers are created
        assert layer.conv1 is not None
        assert layer.conv2 is not None
        assert layer.add_layer is not None
        assert layer.final_activation is not None

        # Shortcut should be None until build is called
        assert layer.shortcut_conv is None

    def test_forward_pass_no_shortcut(self, basic_config: Dict[str, Any], sample_input_1d: keras.KerasTensor) -> None:
        """Test forward pass when input and output dimensions match (no shortcut conv needed)."""
        layer = BiasFreeResidualBlock1D(**basic_config)

        output = layer(sample_input_1d)

        # Check layer is built
        assert layer.built

        # With matching dimensions, no shortcut conv should be created
        assert layer.shortcut_conv is None

        # Check output shape
        expected_shape = sample_input_1d.shape
        assert output.shape == expected_shape

        # Verify output is valid
        output_np = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_np).all()

    def test_forward_pass_with_shortcut(self, basic_config: Dict[str, Any], different_input_1d: keras.KerasTensor) -> None:
        """Test forward pass when input and output dimensions differ (shortcut conv needed)."""
        layer = BiasFreeResidualBlock1D(**basic_config)

        output = layer(different_input_1d)

        # Check layer is built
        assert layer.built

        # With different dimensions, shortcut conv should be created
        assert layer.shortcut_conv is not None
        assert not layer.shortcut_conv.use_bias  # Should be bias-free

        # Check output shape matches the configured filters
        expected_shape = (different_input_1d.shape[0], different_input_1d.shape[1], basic_config['filters'])
        assert output.shape == expected_shape

        # Verify output is valid
        output_np = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_np).all()

    def test_serialization_cycle(self, basic_config: Dict[str, Any], sample_input_1d: keras.KerasTensor) -> None:
        """CRITICAL TEST: Full serialization cycle."""
        inputs = keras.Input(shape=sample_input_1d.shape[1:])
        outputs = BiasFreeResidualBlock1D(**basic_config)(inputs)
        model = keras.Model(inputs, outputs)

        original_pred = model(sample_input_1d)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_residual_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_1d)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Residual block predictions differ after serialization"
            )

    def test_serialization_cycle_with_shortcut(self, basic_config: Dict[str, Any], different_input_1d: keras.KerasTensor) -> None:
        """Test serialization when shortcut conv is needed."""
        inputs = keras.Input(shape=different_input_1d.shape[1:])
        outputs = BiasFreeResidualBlock1D(**basic_config)(inputs)
        model = keras.Model(inputs, outputs)

        original_pred = model(different_input_1d)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_residual_shortcut_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(different_input_1d)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Residual block with shortcut predictions differ after serialization"
            )

    def test_config_completeness(self, basic_config: Dict[str, Any]) -> None:
        """Test that get_config contains all __init__ parameters."""
        layer = BiasFreeResidualBlock1D(**basic_config)
        config = layer.get_config()

        # Check all config parameters are present
        for key in basic_config:
            assert key in config, f"Missing {key} in get_config()"

        # Check additional default parameters
        assert 'kernel_initializer' in config
        assert 'kernel_regularizer' in config

        # Verify values
        assert config['filters'] == basic_config['filters']
        assert config['kernel_size'] == basic_config['kernel_size']

    def test_gradients_flow(self, basic_config: Dict[str, Any], sample_input_1d: keras.KerasTensor) -> None:
        """Test gradient computation through the residual block."""
        layer = BiasFreeResidualBlock1D(**basic_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input_1d)
            output = layer(sample_input_1d, training=True)
            loss = keras.ops.mean(keras.ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert gradients is not None
        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

        # Verify gradients are not all zeros
        for grad in gradients:
            grad_np = keras.ops.convert_to_numpy(grad)
            assert not np.allclose(grad_np, 0), "Gradient should not be all zeros"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, basic_config: Dict[str, Any], sample_input_1d: keras.KerasTensor, training: bool) -> None:
        """Test behavior in different training modes."""
        layer = BiasFreeResidualBlock1D(**basic_config)

        output = layer(sample_input_1d, training=training)

        expected_shape = sample_input_1d.shape
        assert output.shape == expected_shape

        output_np = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_np).all()

    def test_residual_connection_effectiveness(self, basic_config: Dict[str, Any]) -> None:
        """Test that residual connection actually works by comparing with and without."""
        # Create input that exactly matches the filter dimension
        inputs = keras.random.normal(shape=(2, 50, basic_config['filters']))

        # Create layer and get output
        layer = BiasFreeResidualBlock1D(**basic_config)
        output = layer(inputs)

        # The output should be different from input due to residual transformation
        # but should maintain the same shape
        assert output.shape == inputs.shape

        # Verify the output is not identical to input (residual function adds something)
        output_np = keras.ops.convert_to_numpy(output)
        inputs_np = keras.ops.convert_to_numpy(inputs)

        # Should not be identical due to the residual function F(x)
        assert not np.allclose(output_np, inputs_np, rtol=1e-3)

    def test_edge_cases_and_errors(self) -> None:
        """Test error conditions."""
        # Test invalid filters
        with pytest.raises(ValueError):
            BiasFreeResidualBlock1D(filters=0)

        with pytest.raises(ValueError):
            BiasFreeResidualBlock1D(filters=-10)

        # Test invalid kernel_size
        with pytest.raises(ValueError):
            BiasFreeResidualBlock1D(filters=32, kernel_size=0)

        with pytest.raises(ValueError):
            BiasFreeResidualBlock1D(filters=32, kernel_size=-2)

    def test_compute_output_shape(self, basic_config: Dict[str, Any]) -> None:
        """Test output shape computation."""
        layer = BiasFreeResidualBlock1D(**basic_config)

        input_shape = (None, 100, 16)  # Different from output filters
        output_shape = layer.compute_output_shape(input_shape)

        expected_shape = (None, 100, basic_config['filters'])
        assert output_shape == expected_shape

    def test_deep_residual_stack(self, basic_config: Dict[str, Any]) -> None:
        """Test stacking multiple residual blocks."""
        inputs = keras.Input(shape=(100, 32))

        # Stack multiple residual blocks
        x = BiasFreeResidualBlock1D(32)(inputs)  # Same dimension
        x = BiasFreeResidualBlock1D(32)(x)       # Same dimension
        x = BiasFreeResidualBlock1D(64)(x)       # Different dimension (shortcut needed)
        x = BiasFreeResidualBlock1D(64)(x)       # Same dimension
        outputs = BiasFreeConv1D(1, activation='linear')(x)  # Final conv

        model = keras.Model(inputs, outputs)

        # Test forward pass
        test_input = keras.random.normal(shape=(2, 100, 32))
        output = model(test_input)

        assert output.shape == (2, 100, 1)

        # Test gradient flow through deep stack
        with tf.GradientTape() as tape:
            tape.watch(test_input)
            pred = model(test_input, training=True)
            loss = keras.ops.mean(keras.ops.square(pred))

        gradients = tape.gradient(loss, model.trainable_variables)
        assert all(g is not None for g in gradients)


class TestIntegrationAndCompatibility:
    """Integration tests and compatibility checks."""

    def test_mixed_layer_compatibility(self) -> None:
        """Test compatibility with standard Keras layers."""
        inputs = keras.Input(shape=(200, 16))

        # Mix bias-free and standard layers
        x = BiasFreeConv1D(32)(inputs)
        x = keras.layers.Dropout(0.1)(x)
        x = BiasFreeResidualBlock1D(32)(x)
        x = keras.layers.GlobalMaxPooling1D()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs, outputs)

        # Test compilation and forward pass
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        test_input = keras.random.normal(shape=(4, 200, 16))
        output = model(test_input)

        assert output.shape == (4, 10)

    def test_different_data_types(self) -> None:
        """Test layer with different input data types."""
        layer = BiasFreeConv1D(filters=16, kernel_size=3)

        # Test with float32 (standard)
        inputs_f32 = keras.ops.cast(keras.random.normal(shape=(2, 50, 8)), 'float32')
        output_f32 = layer(inputs_f32)
        assert output_f32.dtype == 'float32'

        # Test output is valid
        assert np.isfinite(keras.ops.convert_to_numpy(output_f32)).all()


# Additional utility functions for testing
def create_test_model_with_bias_free_layers() -> keras.Model:
    """Create a test model using both bias-free layer types."""
    inputs = keras.Input(shape=(100, 1), name='input')

    # Encoder path
    x = BiasFreeConv1D(32, kernel_size=3, name='conv1')(inputs)
    x = BiasFreeResidualBlock1D(32, kernel_size=3, name='res1')(x)
    x = BiasFreeResidualBlock1D(64, kernel_size=3, name='res2')(x)  # Dimension change

    # Decoder path
    x = BiasFreeConv1D(32, kernel_size=3, name='conv2')(x)
    outputs = BiasFreeConv1D(1, kernel_size=1, activation='linear', name='output')(x)

    return keras.Model(inputs, outputs, name='bias_free_test_model')


if __name__ == "__main__":
    # Run tests with: pytest test_bias_free_conv1d.py -v
    # Or run specific test class: pytest test_bias_free_conv1d.py::TestBiasFreeConv1D -v
    pytest.main([__file__, "-v"])