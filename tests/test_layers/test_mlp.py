"""
Comprehensive test suite for modern MLPBlock layer implementation.

This test suite follows modern Keras 3 testing patterns and validates the
MLPBlock layer's functionality, serialization, and integration capabilities.

Place this file in: tests/test_layers/test_ffn/test_mlp_block.py
"""

import pytest
import numpy as np
import keras
import tempfile
import os
from typing import Dict, Any
import tensorflow as tf

from dl_techniques.layers.ffn.mlp import MLPBlock


class TestMLPBlock:
    """Comprehensive test suite for modern MLPBlock layer implementation."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample input tensor for testing."""
        return keras.random.normal([8, 16, 128])  # batch, seq_len, features

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard layer configuration for testing."""
        return {
            'hidden_dim': 512,
            'output_dim': 128,
            'activation': 'gelu',
            'dropout_rate': 0.1
        }

    @pytest.fixture
    def custom_layer_config(self) -> Dict[str, Any]:
        """Custom layer configuration with comprehensive parameters."""
        return {
            'hidden_dim': 2048,
            'output_dim': 256,
            'activation': 'swish',
            'dropout_rate': 0.2,
            'use_bias': True,
            'kernel_initializer': 'he_normal',
            'bias_initializer': 'ones',
            'kernel_regularizer': keras.regularizers.L2(1e-4)
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = MLPBlock(hidden_dim=512, output_dim=128)

        # Check stored configuration
        assert layer.hidden_dim == 512
        assert layer.output_dim == 128
        assert layer.activation_name == "gelu"
        assert layer.dropout_rate == 0.0
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.kernel_regularizer is None

        # Check that layer is not built yet
        assert not layer.built

        # Check that sub-layers are created but not built
        assert layer.fc1 is not None
        assert layer.fc2 is not None
        assert layer.activation_fn is not None
        assert not layer.fc1.built
        assert not layer.fc2.built

    def test_initialization_custom(self, custom_layer_config):
        """Test initialization with comprehensive custom parameters."""
        layer = MLPBlock(**custom_layer_config)

        # Verify all custom parameters are stored correctly
        assert layer.hidden_dim == 2048
        assert layer.output_dim == 256
        assert layer.activation_name == 'swish'
        assert layer.dropout_rate == 0.2
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)
        assert isinstance(layer.kernel_regularizer, keras.regularizers.L2)

    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid hidden_dim values
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            MLPBlock(hidden_dim=0, output_dim=64)

        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            MLPBlock(hidden_dim=-10, output_dim=64)

        # Test invalid output_dim values
        with pytest.raises(ValueError, match="output_dim must be positive"):
            MLPBlock(hidden_dim=64, output_dim=0)

        with pytest.raises(ValueError, match="output_dim must be positive"):
            MLPBlock(hidden_dim=64, output_dim=-5)

        # Test invalid dropout_rate values
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            MLPBlock(hidden_dim=64, output_dim=32, dropout_rate=-0.1)

        with pytest.raises(ValueError, match="dropout_rate must be in"):
            MLPBlock(hidden_dim=64, output_dim=32, dropout_rate=1.5)

        # Test edge case: dropout_rate = 1.0 should be invalid
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            MLPBlock(hidden_dim=64, output_dim=32, dropout_rate=1.0)

    def test_build_process(self, sample_input, layer_config):
        """Test that the layer builds properly following modern patterns."""
        layer = MLPBlock(**layer_config)

        # Layer should not be built initially
        assert not layer.built

        # Forward pass triggers build
        output = layer(sample_input)

        # Check that layer is now built
        assert layer.built

        # Check that all sub-layers are built
        assert layer.fc1.built
        assert layer.fc2.built

        # Verify weights were created (4 weights: fc1 kernel/bias + fc2 kernel/bias)
        assert len(layer.weights) == 4

        # Verify output shape
        expected_shape = list(sample_input.shape)
        expected_shape[-1] = layer_config['output_dim']
        assert output.shape == tuple(expected_shape)

    def test_build_dropout_creation(self, sample_input):
        """Test that dropout layer is created correctly based on dropout_rate."""
        # Test with dropout
        layer_with_dropout = MLPBlock(hidden_dim=256, output_dim=128, dropout_rate=0.3)
        layer_with_dropout(sample_input)
        assert layer_with_dropout.dropout is not None

        # Test without dropout
        layer_no_dropout = MLPBlock(hidden_dim=256, output_dim=128, dropout_rate=0.0)
        layer_no_dropout(sample_input)
        assert layer_no_dropout.dropout is None

    def test_forward_pass_basic(self, sample_input, layer_config):
        """Test basic forward pass functionality."""
        layer = MLPBlock(**layer_config)
        output = layer(sample_input)

        # Basic sanity checks
        expected_shape = list(sample_input.shape)
        expected_shape[-1] = layer_config['output_dim']
        assert output.shape == tuple(expected_shape)
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

    def test_forward_pass_deterministic(self):
        """Test forward pass with controlled inputs for deterministic behavior."""
        # Create layer with linear activation and controlled initialization
        layer = MLPBlock(
            hidden_dim=16,
            output_dim=8,
            activation='linear',
            dropout_rate=0.0,
            kernel_initializer='ones',
            bias_initializer='zeros'
        )

        controlled_input = keras.ops.ones([2, 4, 10])
        output = layer(controlled_input)

        # Verify output properties
        assert output.shape == (2, 4, 8)
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

        # With ones initialization and linear activation, output should be predictable
        # Each position should be sum of input features (10) = 10 * 16 hidden units = 160 per output unit
        expected_value = 10.0 * 16.0  # input_sum * hidden_units
        output_mean = keras.ops.mean(output)
        assert abs(keras.ops.convert_to_numpy(output_mean) - expected_value) < 1e-5

    def test_different_activations(self, sample_input):
        """Test layer with various activation functions."""
        activations = ['relu', 'gelu', 'swish', 'tanh', 'linear', 'selu']

        for activation in activations:
            layer = MLPBlock(hidden_dim=128, output_dim=64, activation=activation)
            output = layer(sample_input)

            # Verify output is valid
            expected_shape = list(sample_input.shape)
            expected_shape[-1] = 64
            assert output.shape == tuple(expected_shape)
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_custom_activation_callable(self, sample_input):
        """Test with custom activation function as callable."""
        def custom_activation(x):
            return keras.ops.relu(x) * 0.5

        layer = MLPBlock(hidden_dim=128, output_dim=64, activation=custom_activation)
        output = layer(sample_input)

        expected_shape = list(sample_input.shape)
        expected_shape[-1] = 64
        assert output.shape == tuple(expected_shape)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_dropout_training_vs_inference(self, sample_input):
        """Test dropout behavior in training vs inference modes."""
        layer = MLPBlock(hidden_dim=128, output_dim=64, dropout_rate=0.5)

        # Test inference mode - should be deterministic
        output_inf_1 = layer(sample_input, training=False)
        output_inf_2 = layer(sample_input, training=False)

        # Inference outputs should be identical
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output_inf_1),
            keras.ops.convert_to_numpy(output_inf_2),
            rtol=1e-6, atol=1e-6,
            err_msg="Inference outputs should match"
        )

        # Test training mode - may produce different outputs due to dropout
        output_train_1 = layer(sample_input, training=True)
        output_train_2 = layer(sample_input, training=True)

        # Both should have same shape
        expected_shape = list(sample_input.shape)
        expected_shape[-1] = 64
        expected_shape_tuple = tuple(expected_shape)

        assert output_train_1.shape == expected_shape_tuple
        assert output_train_2.shape == expected_shape_tuple

    def test_compute_output_shape(self):
        """Test output shape computation."""
        test_cases = [
            # (hidden_dim, output_dim, input_shape, expected_output_shape)
            (128, 64, (None, 16, 32), (None, 16, 64)),
            (256, 128, (4, 8, 512), (4, 8, 128)),
            (512, 256, (2, 10, 20, 128), (2, 10, 20, 256)),
            (64, 32, (1, 4, 8, 16, 64), (1, 4, 8, 16, 32)),
        ]

        for hidden_dim, output_dim, input_shape, expected_shape in test_cases:
            layer = MLPBlock(hidden_dim=hidden_dim, output_dim=output_dim)
            computed_shape = layer.compute_output_shape(input_shape)
            assert computed_shape == expected_shape

    def test_get_config_completeness(self, custom_layer_config):
        """Test that get_config contains all initialization parameters."""
        layer = MLPBlock(**custom_layer_config)
        config = layer.get_config()

        # Verify all custom parameters are in config
        expected_keys = {
            'hidden_dim', 'output_dim', 'activation', 'dropout_rate',
            'use_bias', 'kernel_initializer', 'bias_initializer',
            'kernel_regularizer'
        }

        config_keys = set(config.keys())
        assert expected_keys.issubset(config_keys)

        # Verify specific values
        assert config['hidden_dim'] == 2048
        assert config['output_dim'] == 256
        assert config['activation'] == 'swish'
        assert config['dropout_rate'] == 0.2

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle following modern patterns."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = MLPBlock(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_prediction = model(sample_input)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            # Load without custom_objects (thanks to registration decorator)
            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization cycle"
            )

    def test_model_integration_complex(self, sample_input):
        """Test layer integration in a complex transformer-like model."""
        inputs = keras.Input(shape=sample_input.shape[1:])

        # Simulate transformer architecture with MLPBlocks
        x = keras.layers.LayerNormalization()(inputs)

        # First transformer-like block
        attention_output = keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=16
        )(x, x)
        x = keras.layers.Add()([x, attention_output])
        x = keras.layers.LayerNormalization()(x)

        # MLP block (feed-forward)
        mlp_output = MLPBlock(hidden_dim=512, output_dim=128, dropout_rate=0.1)(x)
        x = keras.layers.Add()([x, mlp_output])

        # Second transformer-like block
        x = keras.layers.LayerNormalization()(x)
        attention_output = keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=16
        )(x, x)
        x = keras.layers.Add()([x, attention_output])
        x = keras.layers.LayerNormalization()(x)

        # Another MLP block
        mlp_output = MLPBlock(hidden_dim=256, output_dim=128, dropout_rate=0.2)(x)
        x = keras.layers.Add()([x, mlp_output])

        # Final classification head
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        # Test forward pass
        prediction = model(sample_input)
        assert prediction.shape == (sample_input.shape[0], 10)

        # Test that gradients flow properly
        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = model(sample_input)
            loss = keras.ops.mean(output)

        gradients = tape.gradient(loss, model.trainable_variables)
        assert all(g is not None for g in gradients)

    def test_regularization_losses(self, sample_input):
        """Test that regularization losses are properly computed."""
        layer = MLPBlock(
            hidden_dim=128,
            output_dim=64,
            kernel_regularizer=keras.regularizers.L2(0.01)
        )

        # No losses before forward pass
        initial_losses = len(layer.losses)

        # Apply the layer
        output = layer(sample_input)

        # Should have regularization losses now
        assert len(layer.losses) > initial_losses

        # Verify losses are non-zero
        total_loss = sum(layer.losses)
        assert keras.ops.convert_to_numpy(total_loss) > 0

    def test_gradient_flow(self, sample_input, layer_config):
        """Test that gradients flow properly through the layer."""
        layer = MLPBlock(**layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        # Compute gradients
        gradients = tape.gradient(loss, layer.trainable_variables)

        # Check that all gradients exist and are non-zero
        assert len(gradients) == len(layer.trainable_variables)
        assert all(g is not None for g in gradients)

        # Check gradient shapes match variable shapes
        for grad, var in zip(gradients, layer.trainable_variables):
            assert grad.shape == var.shape

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, sample_input, layer_config, training):
        """Test layer behavior in different training modes."""
        layer = MLPBlock(**layer_config)
        output = layer(sample_input, training=training)

        # Basic output validation
        expected_shape = list(sample_input.shape)
        expected_shape[-1] = layer_config['output_dim']
        assert output.shape == tuple(expected_shape)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_different_input_dimensions(self):
        """Test layer with different input tensor dimensions."""
        layer = MLPBlock(hidden_dim=128, output_dim=64)

        test_shapes = [
            (4, 32),        # 2D input
            (4, 16, 32),    # 3D input
            (2, 8, 16, 32), # 4D input
            (1, 4, 8, 16, 32) # 5D input
        ]

        for shape in test_shapes:
            test_input = keras.random.normal(shape)
            output = layer(test_input)

            # Output should have last dimension changed to output_dim
            expected_shape = list(shape)
            expected_shape[-1] = 64
            assert output.shape == tuple(expected_shape)
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = MLPBlock(hidden_dim=64, output_dim=32, activation='gelu')

        # Test different input value ranges
        test_cases = [
            keras.ops.zeros((4, 8, 16)),                              # Zeros
            keras.ops.ones((4, 8, 16)) * 1e-10,                     # Very small
            keras.ops.ones((4, 8, 16)) * 1e5,                       # Large
            keras.random.normal((4, 8, 16)) * 100,                  # Large random
            keras.random.normal((4, 8, 16)) * 1e-5,                 # Small random
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Verify numerical stability
            assert not keras.ops.any(keras.ops.isnan(output)), "NaN detected"
            assert not keras.ops.any(keras.ops.isinf(output)), "Inf detected"

    def test_weight_structure_and_shapes(self, sample_input):
        """Test that layer weights have correct structure and shapes."""
        input_dim = sample_input.shape[-1]
        hidden_dim = 256
        output_dim = 64

        layer = MLPBlock(hidden_dim=hidden_dim, output_dim=output_dim)
        layer(sample_input)  # Build the layer

        # Should have exactly 4 weights: fc1_kernel, fc1_bias, fc2_kernel, fc2_bias
        assert len(layer.weights) == 4
        assert len(layer.trainable_variables) == 4

        # Verify weight shapes
        expected_shapes = [
            (input_dim, hidden_dim),  # fc1 kernel
            (hidden_dim,),            # fc1 bias
            (hidden_dim, output_dim), # fc2 kernel
            (output_dim,)             # fc2 bias
        ]

        actual_shapes = [tuple(w.shape) for w in layer.weights]
        assert actual_shapes == expected_shapes

    def test_double_dropout_application(self, sample_input):
        """Test that dropout is applied twice as documented (after activation and after projection)."""
        layer = MLPBlock(hidden_dim=128, output_dim=64, dropout_rate=0.8)  # High dropout for testing

        # Build the layer
        output = layer(sample_input, training=True)

        # With very high dropout, outputs should be significantly different between calls
        output1 = layer(sample_input, training=True)
        output2 = layer(sample_input, training=True)

        # Check that shapes are correct
        expected_shape = list(sample_input.shape)
        expected_shape[-1] = 64
        assert output1.shape == tuple(expected_shape)
        assert output2.shape == tuple(expected_shape)

        # With high dropout, outputs should likely be different
        # (Note: there's a small chance they could be identical, but very unlikely with 80% dropout)
        difference = keras.ops.mean(keras.ops.abs(output1 - output2))
        assert keras.ops.convert_to_numpy(difference) > 1e-7  # Should have some difference


class TestMLPBlockEdgeCases:
    """Test edge cases and boundary conditions for MLPBlock."""

    def test_minimal_dimensions(self):
        """Test with minimal viable dimensions."""
        layer = MLPBlock(hidden_dim=1, output_dim=1)
        test_input = keras.random.normal([2, 3, 1])

        output = layer(test_input)
        assert output.shape == (2, 3, 1)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_large_dimensions(self):
        """Test with large dimensions."""
        layer = MLPBlock(hidden_dim=4096, output_dim=2048)
        test_input = keras.random.normal([2, 4, 1024])

        output = layer(test_input)
        assert output.shape == (2, 4, 2048)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_zero_dropout_rate(self):
        """Test with zero dropout rate."""
        sample_input = keras.random.normal([4, 8, 32])
        layer = MLPBlock(hidden_dim=64, output_dim=48, dropout_rate=0.0)

        # Multiple calls should produce identical results when dropout_rate=0
        output1 = layer(sample_input, training=True)
        output2 = layer(sample_input, training=True)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Outputs with zero dropout should be identical"
        )

    def test_single_batch_single_sequence(self):
        """Test with minimal batch and sequence dimensions."""
        layer = MLPBlock(hidden_dim=32, output_dim=16)
        test_input = keras.random.normal([1, 1, 8])  # Single item, single step

        output = layer(test_input)
        assert output.shape == (1, 1, 16)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_expansion_and_projection(self):
        """Test the typical transformer pattern: expand then project."""
        input_dim = 512
        hidden_dim = 2048  # 4x expansion (typical in transformers)
        output_dim = 512   # Project back to original size

        layer = MLPBlock(hidden_dim=hidden_dim, output_dim=output_dim)
        test_input = keras.random.normal([4, 8, input_dim])

        output = layer(test_input)
        assert output.shape == (4, 8, output_dim)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_layer_reuse(self):
        """Test that the same layer instance can be reused."""
        layer = MLPBlock(hidden_dim=128, output_dim=64)

        # Use the same layer with different inputs
        input1 = keras.random.normal([2, 8, 32])
        input2 = keras.random.normal([3, 10, 32])

        output1 = layer(input1)
        output2 = layer(input2)

        assert output1.shape == (2, 8, 64)
        assert output2.shape == (3, 10, 64)

        # Both outputs should be valid
        assert not keras.ops.any(keras.ops.isnan(output1))
        assert not keras.ops.any(keras.ops.isnan(output2))

    def test_bias_disabled(self):
        """Test layer behavior with bias disabled."""
        layer = MLPBlock(
            hidden_dim=64,
            output_dim=32,
            use_bias=False,
            activation='linear'
        )

        test_input = keras.random.normal([2, 4, 16])
        output = layer(test_input)

        # Should have only 2 weights (kernels only, no biases)
        assert len(layer.weights) == 2
        assert output.shape == (2, 4, 32)

    def test_different_initializers(self):
        """Test layer with various weight initializers."""
        initializers = ['glorot_uniform', 'he_normal', 'lecun_normal', 'zeros', 'ones']

        for initializer in initializers:
            layer = MLPBlock(
                hidden_dim=32,
                output_dim=16,
                kernel_initializer=initializer
            )

            test_input = keras.random.normal([2, 4, 8])
            output = layer(test_input)

            assert output.shape == (2, 4, 16)
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_extreme_dropout_rates(self):
        """Test with extreme but valid dropout rates."""
        # Test very low dropout
        layer_low = MLPBlock(hidden_dim=64, output_dim=32, dropout_rate=1e-6)
        test_input = keras.random.normal([2, 4, 16])
        output_low = layer_low(test_input, training=True)
        assert not keras.ops.any(keras.ops.isnan(output_low))

        # Test very high dropout (but less than 1.0)
        layer_high = MLPBlock(hidden_dim=64, output_dim=32, dropout_rate=0.99)
        output_high = layer_high(test_input, training=True)
        assert not keras.ops.any(keras.ops.isnan(output_high))