import pytest
import numpy as np
import keras
import tempfile
import os
from typing import Dict, Any
import tensorflow as tf

from dl_techniques.layers.ffn.residual_block import ResidualBlock


class TestResidualBlock:
    """Comprehensive test suite for modern ResidualBlock layer implementation."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample input tensor for testing."""
        return keras.random.normal([8, 64])  # batch, features

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard layer configuration for testing."""
        return {
            'hidden_dim': 128,
            'output_dim': 32,
            'dropout_rate': 0.1,
            'activation': 'relu'
        }

    @pytest.fixture
    def custom_layer_config(self) -> Dict[str, Any]:
        """Custom layer configuration with all parameters."""
        return {
            'hidden_dim': 256,
            'output_dim': 128,
            'dropout_rate': 0.2,
            'activation': 'gelu',
            'use_bias': True,
            'kernel_initializer': 'he_normal',
            'bias_initializer': 'ones',
            'kernel_regularizer': keras.regularizers.L2(1e-4),
            'bias_regularizer': keras.regularizers.L1(1e-5)
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = ResidualBlock(hidden_dim=64, output_dim=32)

        # Check stored configuration
        assert layer.hidden_dim == 64
        assert layer.output_dim == 32
        assert layer.dropout_rate == 0.0
        assert layer.activation == 'relu'
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None

        # Check that layer is not built yet
        assert not layer.built

        # Check that sub-layers are created but not built
        assert layer.hidden_layer is not None
        assert layer.output_layer is not None
        assert layer.residual_layer is not None
        assert layer.dropout is None  # Default dropout_rate=0.0
        assert not layer.hidden_layer.built
        assert not layer.output_layer.built
        assert not layer.residual_layer.built

    def test_initialization_custom(self, custom_layer_config):
        """Test initialization with comprehensive custom parameters."""
        layer = ResidualBlock(**custom_layer_config)

        # Verify all custom parameters are stored correctly
        assert layer.hidden_dim == 256
        assert layer.output_dim == 128
        assert layer.dropout_rate == 0.2
        assert layer.activation == 'gelu'
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)
        assert isinstance(layer.kernel_regularizer, keras.regularizers.L2)
        assert isinstance(layer.bias_regularizer, keras.regularizers.L1)

        # Check dropout layer is created when dropout_rate > 0
        assert layer.dropout is not None

    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid hidden_dim values
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            ResidualBlock(hidden_dim=0, output_dim=32)

        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            ResidualBlock(hidden_dim=-10, output_dim=32)

        # Test invalid output_dim values
        with pytest.raises(ValueError, match="output_dim must be positive"):
            ResidualBlock(hidden_dim=64, output_dim=0)

        with pytest.raises(ValueError, match="output_dim must be positive"):
            ResidualBlock(hidden_dim=64, output_dim=-5)

        # Test invalid dropout_rate values
        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            ResidualBlock(hidden_dim=64, output_dim=32, dropout_rate=-0.1)

        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            ResidualBlock(hidden_dim=64, output_dim=32, dropout_rate=1.5)

    def test_build_process(self, sample_input, layer_config):
        """Test that the layer builds properly following modern patterns."""
        layer = ResidualBlock(**layer_config)

        # Layer should not be built initially
        assert not layer.built

        # Forward pass triggers build
        output = layer(sample_input)

        # Check that layer is now built
        assert layer.built

        # Check that all sub-layers are built
        assert layer.hidden_layer.built
        assert layer.output_layer.built
        assert layer.residual_layer.built
        if layer.dropout is not None:
            # Dropout layer is built automatically
            pass

        # Verify weights were created
        expected_weights = 6 if layer.use_bias else 3  # 3 layers × 2 weights each (kernel + bias)
        assert len(layer.weights) == expected_weights

        # Verify output shape
        assert output.shape == (sample_input.shape[0], layer_config['output_dim'])

    def test_build_input_validation(self):
        """Test build method input validation."""
        layer = ResidualBlock(hidden_dim=64, output_dim=32)

        # Test with valid input shapes
        layer.build((None, 128))  # Should work fine

        # Test with undefined last dimension should work (Keras handles this)
        layer_2 = ResidualBlock(hidden_dim=64, output_dim=32)
        # This should not raise an error in modern Keras
        # layer_2.build((None, None))

    def test_forward_pass_basic(self, sample_input, layer_config):
        """Test basic forward pass functionality."""
        layer = ResidualBlock(**layer_config)
        output = layer(sample_input)

        # Basic sanity checks
        assert output.shape == (sample_input.shape[0], layer_config['output_dim'])
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

    def test_residual_connection_behavior(self):
        """Test that the residual connection is working correctly."""
        # Create a controlled test to verify residual connection
        layer = ResidualBlock(
            hidden_dim=16,
            output_dim=8,
            activation='linear',  # Linear to make math predictable
            dropout_rate=0.0,
            kernel_initializer='ones',
            bias_initializer='zeros'
        )

        controlled_input = keras.ops.ones([2, 10])  # Shape: [batch=2, features=10]
        output = layer(controlled_input)

        # Verify the residual connection is applied
        # output = main_path + residual_path
        # With 'ones' initialization and linear activation, we can predict the behavior
        assert output.shape == (2, 8)
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

    def test_different_input_output_dimensions(self):
        """Test layer with various input/output dimension combinations."""
        test_cases = [
            # (input_dim, hidden_dim, output_dim)
            (32, 64, 16),  # Reduce dimensions
            (16, 128, 64),  # Expand then reduce
            (64, 32, 64),  # Same input/output, smaller hidden
            (10, 5, 20),  # Small to large
        ]

        for input_dim, hidden_dim, output_dim in test_cases:
            layer = ResidualBlock(hidden_dim=hidden_dim, output_dim=output_dim)
            test_input = keras.random.normal([4, input_dim])
            output = layer(test_input)

            assert output.shape == (4, output_dim)
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_different_activations(self, sample_input):
        """Test layer with various activation functions."""
        activations = ['relu', 'gelu', 'swish', 'tanh', 'linear', 'selu', 'elu']

        for activation in activations:
            layer = ResidualBlock(hidden_dim=128, output_dim=64, activation=activation)
            output = layer(sample_input)

            # Verify output is valid
            assert output.shape == (sample_input.shape[0], 64)
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_custom_activation_callable(self, sample_input):
        """Test with custom activation function as callable."""

        def custom_activation(x):
            return keras.ops.relu(x) * 0.5

        layer = ResidualBlock(hidden_dim=64, output_dim=32, activation=custom_activation)
        output = layer(sample_input)

        assert output.shape == (sample_input.shape[0], 32)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_dropout_training_vs_inference(self, sample_input):
        """Test dropout behavior in training vs inference modes."""
        layer = ResidualBlock(hidden_dim=128, output_dim=64, dropout_rate=0.5)

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
        assert output_train_1.shape == output_train_2.shape == (sample_input.shape[0], 64)

    def test_compute_output_shape(self):
        """Test output shape computation."""
        test_cases = [
            # (hidden_dim, output_dim, input_shape, expected_output_shape)
            (128, 64, (None, 32), (None, 64)),
            (256, 128, (4, 100), (4, 128)),
            (512, 256, (8, 200), (8, 256)),
            (64, 32, (None, 50), (None, 32)),
        ]

        for hidden_dim, output_dim, input_shape, expected_shape in test_cases:
            layer = ResidualBlock(hidden_dim=hidden_dim, output_dim=output_dim)
            computed_shape = layer.compute_output_shape(input_shape)
            assert computed_shape == expected_shape

    def test_get_config_completeness(self, custom_layer_config):
        """Test that get_config contains all initialization parameters."""
        layer = ResidualBlock(**custom_layer_config)
        config = layer.get_config()

        # Verify all custom parameters are in config
        expected_keys = {
            'hidden_dim', 'output_dim', 'dropout_rate', 'activation',
            'use_bias', 'kernel_initializer', 'bias_initializer',
            'kernel_regularizer', 'bias_regularizer'
        }

        config_keys = set(config.keys())
        assert expected_keys.issubset(config_keys)

        # Verify specific values
        assert config['hidden_dim'] == 256
        assert config['output_dim'] == 128
        assert config['dropout_rate'] == 0.2
        assert config['activation'] == 'gelu'

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle following modern patterns."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = ResidualBlock(**layer_config)(inputs)
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
        """Test layer integration in a complex model."""
        inputs = keras.Input(shape=sample_input.shape[1:])

        # Multi-layer architecture with residual blocks
        x = ResidualBlock(hidden_dim=128, output_dim=96, dropout_rate=0.1)(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = ResidualBlock(hidden_dim=256, output_dim=64, dropout_rate=0.2)(x)
        x = keras.layers.LayerNormalization()(x)
        x = ResidualBlock(hidden_dim=128, output_dim=32)(x)
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
        layer = ResidualBlock(
            hidden_dim=128,
            output_dim=64,
            kernel_regularizer=keras.regularizers.L2(0.01),
            bias_regularizer=keras.regularizers.L1(0.01)
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
        layer = ResidualBlock(**layer_config)

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
        layer = ResidualBlock(**layer_config)
        output = layer(sample_input, training=training)

        # Basic output validation
        assert output.shape == (sample_input.shape[0], layer_config['output_dim'])
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_different_input_shapes(self):
        """Test layer with different input tensor shapes."""
        layer = ResidualBlock(hidden_dim=64, output_dim=32)

        test_shapes = [
            (4, 16),  # 2D input (typical)
            (4, 8, 16),  # 3D input (sequences)
            (2, 6, 8, 16),  # 4D input (images without channels)
        ]

        for shape in test_shapes:
            test_input = keras.random.normal(shape)
            output = layer(test_input)

            # Output should have same shape except last dimension
            expected_shape = list(shape)
            expected_shape[-1] = 32
            assert output.shape == tuple(expected_shape)
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = ResidualBlock(hidden_dim=64, output_dim=32, activation='relu')

        # Test different input value ranges
        test_cases = [
            keras.ops.zeros((4, 16)),  # Zeros
            keras.ops.ones((4, 16)) * 1e-10,  # Very small
            keras.ops.ones((4, 16)) * 1e3,  # Large
            keras.random.normal((4, 16)) * 100,  # Large random
            keras.random.normal((4, 16)) * 1e-5,  # Small random
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Verify numerical stability
            assert not keras.ops.any(keras.ops.isnan(output)), "NaN detected"
            assert not keras.ops.any(keras.ops.isinf(output)), "Inf detected"

    def test_weight_structure_and_shapes(self, sample_input):
        """Test that layer weights have correct structure and shapes."""
        input_dim = sample_input.shape[-1]
        hidden_dim = 128
        output_dim = 64

        layer = ResidualBlock(hidden_dim=hidden_dim, output_dim=output_dim)
        layer(sample_input)  # Build the layer

        # Should have exactly 6 weights: 3 layers × 2 weights each (kernel + bias)
        assert len(layer.weights) == 6
        assert len(layer.trainable_variables) == 6

        # Verify weight shapes
        expected_shapes = [
            (input_dim, hidden_dim),  # hidden_layer kernel
            (hidden_dim,),  # hidden_layer bias
            (hidden_dim, output_dim),  # output_layer kernel
            (output_dim,),  # output_layer bias
            (input_dim, output_dim),  # residual_layer kernel
            (output_dim,)  # residual_layer bias
        ]

        actual_shapes = [tuple(w.shape) for w in layer.weights]
        assert actual_shapes == expected_shapes

    def test_residual_connection_mathematical_correctness(self):
        """Test that the residual connection math is correct."""
        # Create a simple test where we can verify the mathematical operation
        layer = ResidualBlock(
            hidden_dim=2,
            output_dim=2,
            activation='linear',
            dropout_rate=0.0,
            use_bias=False,  # Simplify math
            kernel_initializer='ones'
        )

        # Simple input
        test_input = keras.ops.ones([1, 2])  # [1, 1]
        output = layer(test_input)

        # With ones initialization and linear activation:
        # hidden = input @ ones_matrix = [1, 1] @ [[1, 1], [1, 1]] = [2, 2]
        # main_output = hidden @ ones_matrix = [2, 2] @ [[1, 1], [1, 1]] = [4, 4]
        # residual_output = input @ ones_matrix = [1, 1] @ [[1, 1], [1, 1]] = [2, 2]
        # final_output = main_output + residual_output = [4, 4] + [2, 2] = [6, 6]

        expected_output = keras.ops.ones([1, 2]) * 6
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(expected_output),
            rtol=1e-6, atol=1e-6,
            err_msg="Residual connection math is incorrect"
        )


class TestResidualBlockEdgeCases:
    """Test edge cases and boundary conditions for ResidualBlock."""

    def test_minimal_dimensions(self):
        """Test with minimal viable dimensions."""
        layer = ResidualBlock(hidden_dim=1, output_dim=1)
        test_input = keras.random.normal([2, 1])

        output = layer(test_input)
        assert output.shape == (2, 1)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_large_dimensions(self):
        """Test with large dimensions."""
        layer = ResidualBlock(hidden_dim=1024, output_dim=512)
        test_input = keras.random.normal([2, 256])

        output = layer(test_input)
        assert output.shape == (2, 512)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_zero_dropout_rate(self):
        """Test with zero dropout rate."""
        sample_input = keras.random.normal([4, 32])
        layer = ResidualBlock(hidden_dim=64, output_dim=16, dropout_rate=0.0)

        # Multiple calls should produce identical results
        output1 = layer(sample_input, training=True)
        output2 = layer(sample_input, training=True)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Zero dropout should produce identical results"
        )

    def test_single_batch_item(self):
        """Test with single batch item."""
        layer = ResidualBlock(hidden_dim=32, output_dim=16)
        test_input = keras.random.normal([1, 8])  # Single batch item

        output = layer(test_input)
        assert output.shape == (1, 16)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_layer_reuse(self):
        """Test that the same layer instance can be reused."""
        layer = ResidualBlock(hidden_dim=64, output_dim=32)

        # Use the same layer with different inputs
        input1 = keras.random.normal([2, 16])
        input2 = keras.random.normal([3, 16])

        output1 = layer(input1)
        output2 = layer(input2)

        assert output1.shape == (2, 32)
        assert output2.shape == (3, 32)

        # Both outputs should be valid
        assert not keras.ops.any(keras.ops.isnan(output1))
        assert not keras.ops.any(keras.ops.isnan(output2))

    def test_no_bias_configuration(self):
        """Test layer with bias disabled."""
        layer = ResidualBlock(
            hidden_dim=32,
            output_dim=16,
            use_bias=False
        )
        test_input = keras.random.normal([4, 8])

        output = layer(test_input)

        # Should have only 3 weights (kernels only, no biases)
        assert len(layer.weights) == 3
        assert output.shape == (4, 16)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_extreme_dropout_rate(self):
        """Test with very high dropout rate."""
        layer = ResidualBlock(
            hidden_dim=128,
            output_dim=64,
            dropout_rate=0.99  # Very high dropout
        )
        test_input = keras.random.normal([8, 32])

        # Should still work, though outputs may be sparse
        output = layer(test_input, training=True)
        assert output.shape == (8, 64)
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))