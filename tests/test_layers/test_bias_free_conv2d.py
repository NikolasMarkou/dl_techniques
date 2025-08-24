"""
Comprehensive test suite for BiasFreeConv2D and BiasFreeResidualBlock layers.

Following modern Keras 3 testing patterns with emphasis on serialization robustness
and gradient flow validation for 2D image processing layers.
"""

import pytest
import tempfile
import os
import numpy as np
from typing import Any, Dict

import keras
import tensorflow as tf

from dl_techniques.layers.bias_free_conv2d import BiasFreeConv2D, BiasFreeResidualBlock


class TestBiasFreeConv2D:
    """Comprehensive test suite for BiasFreeConv2D layer."""

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
        """Advanced configuration with regularization and asymmetric kernel."""
        return {
            'filters': 64,
            'kernel_size': (3, 5),  # Asymmetric kernel
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
            'kernel_size': (7, 7),
            'activation': 'swish',
            'use_batch_norm': False
        }

    @pytest.fixture
    def asymmetric_config(self) -> Dict[str, Any]:
        """Configuration with asymmetric kernel for edge detection."""
        return {
            'filters': 8,
            'kernel_size': (1, 7),  # Horizontal edge detector
            'activation': 'relu',
            'use_batch_norm': True
        }

    @pytest.fixture
    def sample_input_2d(self) -> keras.KerasTensor:
        """Sample 2D input tensor for testing."""
        return keras.random.normal(shape=(4, 32, 32, 3))  # batch=4, 32x32 RGB images

    @pytest.fixture
    def large_input_2d(self) -> keras.KerasTensor:
        """Larger 2D input tensor for gradient testing."""
        return keras.random.normal(shape=(2, 64, 64, 16))  # batch=2, 64x64, 16 channels

    @pytest.fixture
    def grayscale_input(self) -> keras.KerasTensor:
        """Grayscale input for denoising scenarios."""
        return keras.random.normal(shape=(8, 128, 128, 1))  # batch=8, 128x128 grayscale

    def test_initialization(self, basic_config: Dict[str, Any]) -> None:
        """Test layer initialization and attribute setting."""
        layer = BiasFreeConv2D(**basic_config)

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

    def test_initialization_asymmetric_kernel(self, asymmetric_config: Dict[str, Any]) -> None:
        """Test initialization with asymmetric kernel."""
        layer = BiasFreeConv2D(**asymmetric_config)

        assert layer.kernel_size == asymmetric_config['kernel_size']
        assert layer.filters == asymmetric_config['filters']
        assert layer.conv.kernel_size == asymmetric_config['kernel_size']

    def test_initialization_no_batch_norm(self, no_bn_config: Dict[str, Any]) -> None:
        """Test initialization without batch normalization."""
        layer = BiasFreeConv2D(**no_bn_config)

        assert layer.filters == no_bn_config['filters']
        assert not layer.use_batch_norm
        assert layer.batch_norm is None
        assert layer.conv is not None
        assert layer.activation_layer is not None

    def test_forward_pass(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """Test forward pass and automatic building."""
        layer = BiasFreeConv2D(**basic_config)

        # Forward pass should trigger building
        output = layer(sample_input_2d)

        # Check layer is now built
        assert layer.built

        # Check output shape (spatial dimensions preserved, channels changed)
        expected_shape = (sample_input_2d.shape[0], sample_input_2d.shape[1],
                         sample_input_2d.shape[2], basic_config['filters'])
        assert output.shape == expected_shape

        # Check output is not NaN or infinite
        output_np = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_np).all()
        assert not np.isnan(output_np).any()

    def test_forward_pass_asymmetric(self, asymmetric_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """Test forward pass with asymmetric kernel."""
        layer = BiasFreeConv2D(**asymmetric_config)

        output = layer(sample_input_2d)

        expected_shape = (sample_input_2d.shape[0], sample_input_2d.shape[1],
                         sample_input_2d.shape[2], asymmetric_config['filters'])
        assert output.shape == expected_shape

        # Output should still be valid with asymmetric kernel
        output_np = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_np).all()

    def test_forward_pass_no_batch_norm(self, no_bn_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """Test forward pass without batch normalization."""
        layer = BiasFreeConv2D(**no_bn_config)

        output = layer(sample_input_2d)

        expected_shape = (sample_input_2d.shape[0], sample_input_2d.shape[1],
                         sample_input_2d.shape[2], no_bn_config['filters'])
        assert output.shape == expected_shape

        # Output should still be valid
        output_np = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_np).all()

    def test_serialization_cycle(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input_2d.shape[1:])
        outputs = BiasFreeConv2D(**basic_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input_2d)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_conv2d_model.keras')
            model.save(filepath)

            # Load model
            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_2d)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_serialization_cycle_advanced_config(self, advanced_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """Test serialization with advanced configuration and asymmetric kernel."""
        inputs = keras.Input(shape=sample_input_2d.shape[1:])
        outputs = BiasFreeConv2D(**advanced_config)(inputs)
        model = keras.Model(inputs, outputs)

        original_pred = model(sample_input_2d)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_advanced_conv2d_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_2d)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Advanced config predictions differ after serialization"
            )

    def test_config_completeness(self, basic_config: Dict[str, Any]) -> None:
        """Test that get_config contains all __init__ parameters."""
        layer = BiasFreeConv2D(**basic_config)
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

    def test_config_completeness_asymmetric(self, asymmetric_config: Dict[str, Any]) -> None:
        """Test config completeness with asymmetric kernel."""
        layer = BiasFreeConv2D(**asymmetric_config)
        config = layer.get_config()

        # Check tuple kernel_size is preserved
        assert config['kernel_size'] == asymmetric_config['kernel_size']
        assert isinstance(config['kernel_size'], tuple)

    def test_gradients_flow(self, basic_config: Dict[str, Any], large_input_2d: keras.KerasTensor) -> None:
        """Test gradient computation and flow through the layer."""
        layer = BiasFreeConv2D(**basic_config)

        with tf.GradientTape() as tape:
            tape.watch(large_input_2d)
            output = layer(large_input_2d, training=True)
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
    def test_training_modes(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor, training: bool) -> None:
        """Test behavior in different training modes."""
        layer = BiasFreeConv2D(**basic_config)

        # Forward pass in specified training mode
        output = layer(sample_input_2d, training=training)

        # Basic shape and validity checks
        expected_shape = (sample_input_2d.shape[0], sample_input_2d.shape[1],
                         sample_input_2d.shape[2], basic_config['filters'])
        assert output.shape == expected_shape

        output_np = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_np).all()

    def test_different_input_shapes(self, basic_config: Dict[str, Any]) -> None:
        """Test layer with different input shapes."""
        # Test various input shapes - create new layer for each to avoid shape locking
        test_shapes = [
            (1, 28, 28, 1),    # Single MNIST-like grayscale
            (4, 224, 224, 3),  # ImageNet-like RGB batch
            (2, 64, 32, 16),   # Non-square images with many channels
            (8, 512, 512, 1)   # Large grayscale images
        ]

        for shape in test_shapes:
            # Create a new layer instance for each shape since layers get locked to input shape after building
            layer = BiasFreeConv2D(**basic_config)
            inputs = keras.random.normal(shape=shape)
            output = layer(inputs)

            expected_shape = (shape[0], shape[1], shape[2], basic_config['filters'])
            assert output.shape == expected_shape

    def test_edge_cases_and_errors(self) -> None:
        """Test error conditions and edge cases."""
        # Test invalid filters
        with pytest.raises(ValueError):
            BiasFreeConv2D(filters=0)  # Zero filters

        with pytest.raises(ValueError):
            BiasFreeConv2D(filters=-5)  # Negative filters

        # Test invalid kernel_size
        with pytest.raises(ValueError):
            BiasFreeConv2D(filters=32, kernel_size=0)  # Zero kernel size

        with pytest.raises(ValueError):
            BiasFreeConv2D(filters=32, kernel_size=-3)  # Negative kernel size

        with pytest.raises(ValueError):
            BiasFreeConv2D(filters=32, kernel_size=(3, 0))  # Zero in tuple

        with pytest.raises(ValueError):
            BiasFreeConv2D(filters=32, kernel_size=(3, -1))  # Negative in tuple

        with pytest.raises(ValueError):
            BiasFreeConv2D(filters=32, kernel_size=(3,))  # Wrong tuple length

        with pytest.raises(TypeError):
            BiasFreeConv2D(filters=32, kernel_size="invalid")  # Wrong type

        # Test invalid use_batch_norm type
        with pytest.raises(TypeError):
            BiasFreeConv2D(filters=32, use_batch_norm="invalid")  # String instead of bool

    def test_compute_output_shape(self, basic_config: Dict[str, Any]) -> None:
        """Test output shape computation."""
        layer = BiasFreeConv2D(**basic_config)

        input_shape = (None, 64, 64, 16)  # Batch can be None
        output_shape = layer.compute_output_shape(input_shape)

        expected_shape = (None, 64, 64, basic_config['filters'])
        assert output_shape == expected_shape

    def test_no_activation(self) -> None:
        """Test layer with no activation function."""
        layer = BiasFreeConv2D(filters=16, activation=None)

        inputs = keras.random.normal(shape=(2, 32, 32, 8))
        output = layer(inputs)

        # Should work without activation
        assert output.shape == (2, 32, 32, 16)
        assert layer.activation_layer is None

    def test_grayscale_denoising_scenario(self, grayscale_input: keras.KerasTensor) -> None:
        """Test layer in typical grayscale denoising scenario."""
        layer = BiasFreeConv2D(filters=64, kernel_size=5, activation='relu')

        output = layer(grayscale_input)

        # Should handle grayscale input properly
        assert output.shape == (8, 128, 128, 64)

        # Output should be valid
        output_np = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_np).all()


class TestBiasFreeResidualBlock:
    """Comprehensive test suite for BiasFreeResidualBlock layer."""

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
        """Advanced configuration with regularization and asymmetric kernel."""
        return {
            'filters': 64,
            'kernel_size': (3, 5),  # Asymmetric kernel
            'activation': 'gelu',
            'kernel_initializer': 'he_uniform',
            'kernel_regularizer': keras.regularizers.L1L2(l1=1e-5, l2=1e-4)
        }

    @pytest.fixture
    def sample_input_2d(self) -> keras.KerasTensor:
        """Sample input tensor matching the output filters (no shortcut needed)."""
        return keras.random.normal(shape=(4, 64, 64, 32))  # channels=32 to match basic_config filters

    @pytest.fixture
    def different_input_2d(self) -> keras.KerasTensor:
        """Input tensor with different channel dimension (shortcut needed)."""
        return keras.random.normal(shape=(4, 64, 64, 16))  # channels=16, different from output

    @pytest.fixture
    def large_input_2d(self) -> keras.KerasTensor:
        """Large input for deep network testing."""
        return keras.random.normal(shape=(2, 128, 128, 64))

    def test_initialization(self, basic_config: Dict[str, Any]) -> None:
        """Test layer initialization and attribute setting."""
        layer = BiasFreeResidualBlock(**basic_config)

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

    def test_initialization_asymmetric(self, advanced_config: Dict[str, Any]) -> None:
        """Test initialization with asymmetric kernel."""
        layer = BiasFreeResidualBlock(**advanced_config)

        assert layer.kernel_size == advanced_config['kernel_size']
        assert isinstance(layer.kernel_size, tuple)
        assert layer.filters == advanced_config['filters']

    def test_forward_pass_no_shortcut(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """Test forward pass when input and output dimensions match (no shortcut conv needed)."""
        layer = BiasFreeResidualBlock(**basic_config)

        output = layer(sample_input_2d)

        # Check layer is built
        assert layer.built

        # With matching dimensions, no shortcut conv should be created
        assert layer.shortcut_conv is None

        # Check output shape
        expected_shape = sample_input_2d.shape
        assert output.shape == expected_shape

        # Verify output is valid
        output_np = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_np).all()

    def test_forward_pass_with_shortcut(self, basic_config: Dict[str, Any], different_input_2d: keras.KerasTensor) -> None:
        """Test forward pass when input and output dimensions differ (shortcut conv needed)."""
        layer = BiasFreeResidualBlock(**basic_config)

        output = layer(different_input_2d)

        # Check layer is built
        assert layer.built

        # With different dimensions, shortcut conv should be created
        assert layer.shortcut_conv is not None
        assert not layer.shortcut_conv.use_bias  # Should be bias-free
        assert layer.shortcut_conv.kernel_size == (1, 1)  # Should be 1x1 conv

        # Check output shape matches the configured filters
        expected_shape = (different_input_2d.shape[0], different_input_2d.shape[1],
                         different_input_2d.shape[2], basic_config['filters'])
        assert output.shape == expected_shape

        # Verify output is valid
        output_np = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_np).all()

    def test_forward_pass_asymmetric(self, advanced_config: Dict[str, Any], different_input_2d: keras.KerasTensor) -> None:
        """Test forward pass with asymmetric kernel and shortcut."""
        layer = BiasFreeResidualBlock(**advanced_config)

        output = layer(different_input_2d)

        # Should work with asymmetric kernels
        expected_shape = (different_input_2d.shape[0], different_input_2d.shape[1],
                         different_input_2d.shape[2], advanced_config['filters'])
        assert output.shape == expected_shape

        # Verify shortcut was created due to dimension mismatch
        assert layer.shortcut_conv is not None

    def test_serialization_cycle(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """CRITICAL TEST: Full serialization cycle."""
        inputs = keras.Input(shape=sample_input_2d.shape[1:])
        outputs = BiasFreeResidualBlock(**basic_config)(inputs)
        model = keras.Model(inputs, outputs)

        original_pred = model(sample_input_2d)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_residual2d_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_2d)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Residual block predictions differ after serialization"
            )

    def test_serialization_cycle_with_shortcut(self, basic_config: Dict[str, Any], different_input_2d: keras.KerasTensor) -> None:
        """Test serialization when shortcut conv is needed."""
        inputs = keras.Input(shape=different_input_2d.shape[1:])
        outputs = BiasFreeResidualBlock(**basic_config)(inputs)
        model = keras.Model(inputs, outputs)

        original_pred = model(different_input_2d)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_residual2d_shortcut_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(different_input_2d)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Residual block with shortcut predictions differ after serialization"
            )

    def test_config_completeness(self, basic_config: Dict[str, Any]) -> None:
        """Test that get_config contains all __init__ parameters."""
        layer = BiasFreeResidualBlock(**basic_config)
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

    def test_config_completeness_asymmetric(self, advanced_config: Dict[str, Any]) -> None:
        """Test config completeness with asymmetric kernel."""
        layer = BiasFreeResidualBlock(**advanced_config)
        config = layer.get_config()

        # Check tuple kernel_size is preserved
        assert config['kernel_size'] == advanced_config['kernel_size']
        assert isinstance(config['kernel_size'], tuple)

    def test_gradients_flow(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor) -> None:
        """Test gradient computation through the residual block."""
        layer = BiasFreeResidualBlock(**basic_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input_2d)
            output = layer(sample_input_2d, training=True)
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
    def test_training_modes(self, basic_config: Dict[str, Any], sample_input_2d: keras.KerasTensor, training: bool) -> None:
        """Test behavior in different training modes."""
        layer = BiasFreeResidualBlock(**basic_config)

        output = layer(sample_input_2d, training=training)

        expected_shape = sample_input_2d.shape
        assert output.shape == expected_shape

        output_np = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_np).all()

    def test_residual_connection_effectiveness(self, basic_config: Dict[str, Any]) -> None:
        """Test that residual connection actually works by comparing with and without."""
        # Create input that exactly matches the filter dimension
        inputs = keras.random.normal(shape=(2, 32, 32, basic_config['filters']))

        # Create layer and get output
        layer = BiasFreeResidualBlock(**basic_config)
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
            BiasFreeResidualBlock(filters=0)

        with pytest.raises(ValueError):
            BiasFreeResidualBlock(filters=-10)

        # Test invalid kernel_size
        with pytest.raises(ValueError):
            BiasFreeResidualBlock(filters=32, kernel_size=0)

        with pytest.raises(ValueError):
            BiasFreeResidualBlock(filters=32, kernel_size=(3, 0))

        with pytest.raises(TypeError):
            BiasFreeResidualBlock(filters=32, kernel_size="invalid")

    def test_compute_output_shape(self, basic_config: Dict[str, Any]) -> None:
        """Test output shape computation."""
        layer = BiasFreeResidualBlock(**basic_config)

        input_shape = (None, 128, 128, 16)  # Different from output filters
        output_shape = layer.compute_output_shape(input_shape)

        expected_shape = (None, 128, 128, basic_config['filters'])
        assert output_shape == expected_shape

    def test_deep_residual_stack(self, basic_config: Dict[str, Any]) -> None:
        """Test stacking multiple residual blocks."""
        inputs = keras.Input(shape=(64, 64, 16))

        # Stack multiple residual blocks
        x = BiasFreeConv2D(32)(inputs)            # Change to 32 channels first
        x = BiasFreeResidualBlock(32)(x)          # Same dimension
        x = BiasFreeResidualBlock(32)(x)          # Same dimension
        x = BiasFreeResidualBlock(64)(x)          # Different dimension (shortcut needed)
        x = BiasFreeResidualBlock(64)(x)          # Same dimension
        outputs = BiasFreeConv2D(1, activation='sigmoid')(x)  # Final conv for denoising

        model = keras.Model(inputs, outputs)

        # Test forward pass
        test_input = keras.random.normal(shape=(2, 64, 64, 16))
        output = model(test_input)

        assert output.shape == (2, 64, 64, 1)

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
        inputs = keras.Input(shape=(128, 128, 3))

        # Mix bias-free and standard layers
        x = BiasFreeConv2D(32)(inputs)
        x = keras.layers.Dropout(0.1)(x)
        x = BiasFreeResidualBlock(64)(x)
        x = keras.layers.MaxPooling2D(2)(x)
        x = BiasFreeConv2D(128)(x)
        x = keras.layers.GlobalMaxPooling2D()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs, outputs)

        # Test compilation and forward pass
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        test_input = keras.random.normal(shape=(4, 128, 128, 3))
        output = model(test_input)

        assert output.shape == (4, 10)

    def test_denoising_architecture(self) -> None:
        """Test typical denoising architecture with bias-free layers."""
        inputs = keras.Input(shape=(256, 256, 1), name='noisy_input')

        # Encoder path
        x1 = BiasFreeConv2D(32, kernel_size=3, name='conv1')(inputs)
        x2 = BiasFreeResidualBlock(32, kernel_size=3, name='res1')(x1)
        x3 = BiasFreeResidualBlock(64, kernel_size=3, name='res2')(x2)  # Dimension change

        # Bottleneck
        x4 = BiasFreeResidualBlock(64, kernel_size=3, name='bottleneck')(x3)

        # Decoder path
        x5 = BiasFreeResidualBlock(32, kernel_size=3, name='res3')(x4)
        x6 = BiasFreeConv2D(16, kernel_size=3, name='conv2')(x5)
        outputs = BiasFreeConv2D(1, kernel_size=1, activation='sigmoid', name='clean_output')(x6)

        model = keras.Model(inputs, outputs, name='bias_free_denoiser')

        # Test forward pass
        test_input = keras.random.normal(shape=(2, 256, 256, 1))
        output = model(test_input)

        assert output.shape == (2, 256, 256, 1)

        # Test model compilation
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Model should be trainable
        assert len(model.trainable_variables) > 0

    def test_different_data_types(self) -> None:
        """Test layer with different input data types."""
        layer = BiasFreeConv2D(filters=16, kernel_size=3)

        # Test with float32 (standard)
        inputs_f32 = keras.ops.cast(keras.random.normal(shape=(2, 32, 32, 8)), 'float32')
        output_f32 = layer(inputs_f32)
        assert output_f32.dtype == 'float32'

        # Test output is valid
        assert np.isfinite(keras.ops.convert_to_numpy(output_f32)).all()


# Additional utility functions for testing
def create_test_denoising_model() -> keras.Model:
    """Create a complete bias-free denoising model for testing."""
    inputs = keras.Input(shape=(128, 128, 1), name='noisy_input')

    # Feature extraction
    x = BiasFreeConv2D(32, kernel_size=5, name='initial_conv')(inputs)

    # Residual blocks for deep feature learning
    for i in range(4):
        x = BiasFreeResidualBlock(32, kernel_size=3, name=f'res_block_{i}')(x)

    # Final denoising layer
    outputs = BiasFreeConv2D(1, kernel_size=3, activation='sigmoid', name='denoised_output')(x)

    return keras.Model(inputs, outputs, name='bias_free_denoising_model')


def test_full_denoising_pipeline():
    """Test complete denoising pipeline functionality."""
    model = create_test_denoising_model()

    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae', 'mse']
    )

    # Test with synthetic noisy data
    batch_size = 4
    noisy_images = keras.random.normal(shape=(batch_size, 128, 128, 1))

    # Forward pass
    denoised = model(noisy_images)

    # Check output properties
    assert denoised.shape == noisy_images.shape
    assert denoised.dtype == noisy_images.dtype

    # Output should be in valid range for sigmoid activation
    denoised_np = keras.ops.convert_to_numpy(denoised)
    assert denoised_np.min() >= 0.0
    assert denoised_np.max() <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])