import pytest
import numpy as np
import keras
import tempfile
import os
from typing import Dict, Any
import tensorflow as tf

from dl_techniques.layers.ffn.swin_mlp import SwinMLP


class TestSwinMLP:
    """Comprehensive test suite for modern SwinMLP layer implementation."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample input tensor for testing."""
        return keras.random.normal([8, 16, 128])  # batch, seq_len, features

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard layer configuration for testing."""
        return {
            'hidden_dim': 256,
            'dropout_rate': 0.1,
            'activation': 'gelu'
        }

    @pytest.fixture
    def custom_layer_config(self) -> Dict[str, Any]:
        """Custom layer configuration with regularization."""
        return {
            'hidden_dim': 512,
            'out_dim': 128,
            'activation': 'swish',
            'dropout_rate': 0.2,
            'use_bias': True,
            'kernel_initializer': 'he_normal',
            'bias_initializer': 'ones',
            'kernel_regularizer': keras.regularizers.L2(1e-4),
            'bias_regularizer': keras.regularizers.L1(1e-5),
            'activity_regularizer': keras.regularizers.L2(1e-6)
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = SwinMLP(hidden_dim=128)

        # Check stored configuration
        assert layer.hidden_dim == 128
        assert layer.out_dim is None
        assert layer.activation == "gelu"
        assert layer.dropout_rate == 0.0
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None
        assert layer.activity_regularizer is None

        # Check that layer is not built yet
        assert not layer.built

        # Check that sub-layers are created but not built
        assert layer.fc1 is not None
        assert layer.act is not None
        assert layer.drop1 is not None
        assert layer.drop2 is not None
        assert not layer.fc1.built

    def test_initialization_custom(self, custom_layer_config):
        """Test initialization with comprehensive custom parameters."""
        layer = SwinMLP(**custom_layer_config)

        # Verify all custom parameters are stored correctly
        assert layer.hidden_dim == 512
        assert layer.out_dim == 128
        assert layer.activation == 'swish'
        assert layer.dropout_rate == 0.2
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)
        assert isinstance(layer.kernel_regularizer, keras.regularizers.L2)
        assert isinstance(layer.bias_regularizer, keras.regularizers.L1)
        assert isinstance(layer.activity_regularizer, keras.regularizers.L2)

    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid hidden_dim values
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            SwinMLP(hidden_dim=0)

        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            SwinMLP(hidden_dim=-10)

        # Test invalid dropout_rate values
        with pytest.raises(ValueError, match="dropout_rate must be between 0.0 and 1.0"):
            SwinMLP(hidden_dim=64, dropout_rate=-0.1)

        with pytest.raises(ValueError, match="dropout_rate must be between 0.0 and 1.0"):
            SwinMLP(hidden_dim=64, dropout_rate=1.5)

        # Test invalid out_dim when specified
        with pytest.raises(ValueError, match="out_dim must be positive when specified"):
            SwinMLP(hidden_dim=64, out_dim=0)

        with pytest.raises(ValueError, match="out_dim must be positive when specified"):
            SwinMLP(hidden_dim=64, out_dim=-5)

    def test_build_process(self, sample_input, layer_config):
        """Test that the layer builds properly following modern patterns."""
        layer = SwinMLP(**layer_config)

        # Layer should not be built initially
        assert not layer.built
        assert layer.fc2 is None  # Created in build()

        # Forward pass triggers build
        output = layer(sample_input)

        # Check that layer is now built
        assert layer.built
        assert layer.fc2 is not None

        # Check that all sub-layers are built
        assert layer.fc1.built
        assert layer.fc2.built
        # Note: Activation and Dropout layers don't have built attribute

        # Verify weights were created
        assert len(layer.weights) == 4  # fc1 kernel/bias + fc2 kernel/bias

        # Verify output shape
        assert output.shape == sample_input.shape  # out_dim=None preserves input shape

    def test_build_with_output_dimension(self, sample_input):
        """Test building with custom output dimension."""
        layer = SwinMLP(hidden_dim=256, out_dim=64)
        output = layer(sample_input)

        # Check output shape is correctly modified
        expected_shape = list(sample_input.shape)
        expected_shape[-1] = 64
        assert output.shape == tuple(expected_shape)

    def test_build_input_validation(self):
        """Test build method input validation."""
        layer = SwinMLP(hidden_dim=128)

        # Test with invalid input shapes
        with pytest.raises(ValueError, match="Input must be at least 2D"):
            layer.build((10,))  # 1D input

        with pytest.raises(ValueError, match="Last dimension of input must be defined"):
            layer.build((None, 10, None))  # Undefined last dimension

    def test_forward_pass_basic(self, sample_input, layer_config):
        """Test basic forward pass functionality."""
        layer = SwinMLP(**layer_config)
        output = layer(sample_input)

        # Basic sanity checks
        assert output.shape == sample_input.shape  # No out_dim specified
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

    def test_forward_pass_deterministic(self):
        """Test forward pass with controlled inputs for deterministic behavior."""
        # Create layer with linear activation and controlled initialization
        layer = SwinMLP(
            hidden_dim=16,
            out_dim=8,
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

    def test_different_activations(self, sample_input):
        """Test layer with various activation functions."""
        activations = ['relu', 'gelu', 'swish', 'tanh', 'linear', 'selu']

        for activation in activations:
            layer = SwinMLP(hidden_dim=128, activation=activation)
            output = layer(sample_input)

            # Verify output is valid
            assert output.shape == sample_input.shape
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_custom_activation_callable(self, sample_input):
        """Test with custom activation function as callable."""
        def custom_activation(x):
            return keras.ops.relu(x) * 0.5

        layer = SwinMLP(hidden_dim=64, activation=custom_activation)
        output = layer(sample_input)

        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_dropout_training_vs_inference(self, sample_input):
        """Test dropout behavior in training vs inference modes."""
        layer = SwinMLP(hidden_dim=128, dropout_rate=0.5)

        # Test inference mode - should be deterministic
        output_inf_1 = layer(sample_input, training=False)
        output_inf_2 = layer(sample_input, training=False)

        # Inference outputs should be identical
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output_inf_1),
            keras.ops.convert_to_numpy(output_inf_2),
            rtol=1e-6, atol=1e-6,
            err_msg="should match"
        )

        # Test training mode - may produce different outputs due to dropout
        output_train_1 = layer(sample_input, training=True)
        output_train_2 = layer(sample_input, training=True)

        # Both should have same shape
        assert output_train_1.shape == output_train_2.shape == sample_input.shape

    def test_compute_output_shape(self):
        """Test output shape computation."""
        test_cases = [
            # (hidden_dim, out_dim, input_shape, expected_output_shape)
            (128, None, (None, 16, 64), (None, 16, 64)),
            (256, 32, (None, 16, 64), (None, 16, 32)),
            (512, 128, (4, 8, 256), (4, 8, 128)),
            (64, 16, (2, 10, 20, 32), (2, 10, 20, 16)),
        ]

        for hidden_dim, out_dim, input_shape, expected_shape in test_cases:
            layer = SwinMLP(hidden_dim=hidden_dim, out_dim=out_dim)
            computed_shape = layer.compute_output_shape(input_shape)
            assert computed_shape == expected_shape

    def test_get_config_completeness(self, custom_layer_config):
        """Test that get_config contains all initialization parameters."""
        layer = SwinMLP(**custom_layer_config)
        config = layer.get_config()

        # Verify all custom parameters are in config
        expected_keys = {
            'hidden_dim', 'use_bias', 'out_dim', 'activation', 'dropout_rate',
            'kernel_initializer', 'bias_initializer', 'kernel_regularizer',
            'bias_regularizer', 'activity_regularizer'
        }

        config_keys = set(config.keys())
        assert expected_keys.issubset(config_keys)

        # Verify specific values
        assert config['hidden_dim'] == 512
        assert config['out_dim'] == 128
        assert config['activation'] == 'swish'
        assert config['dropout_rate'] == 0.2

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle following modern patterns."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = SwinMLP(**layer_config)(inputs)
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

        # Multi-layer architecture
        x = SwinMLP(hidden_dim=256, dropout_rate=0.1)(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = SwinMLP(hidden_dim=512, out_dim=128, dropout_rate=0.2)(x)
        x = keras.layers.LayerNormalization()(x)
        x = SwinMLP(hidden_dim=64, out_dim=32)(x)
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
        layer = SwinMLP(
            hidden_dim=128,
            kernel_regularizer=keras.regularizers.L2(0.01),
            bias_regularizer=keras.regularizers.L1(0.01),
            activity_regularizer=keras.regularizers.L2(0.005)
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
        layer = SwinMLP(**layer_config)

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
        layer = SwinMLP(**layer_config)
        output = layer(sample_input, training=training)

        # Basic output validation
        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_different_input_dimensions(self):
        """Test layer with different input tensor dimensions."""
        layer = SwinMLP(hidden_dim=64)

        test_shapes = [
            (4, 32),        # 2D input
            (4, 16, 32),    # 3D input
            (2, 8, 16, 32), # 4D input
            (1, 4, 8, 16, 32) # 5D input
        ]

        for shape in test_shapes:
            test_input = keras.random.normal(shape)
            output = layer(test_input)

            # Output should preserve input shape (no out_dim specified)
            assert output.shape == test_input.shape
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = SwinMLP(hidden_dim=32, activation='gelu')

        # Test different input value ranges
        test_cases = [
            keras.ops.zeros((4, 8, 16)),                              # Zeros
            keras.ops.ones((4, 8, 16)) * 1e-10,                     # Very small
            keras.ops.ones((4, 8, 16)) * 1e5,                       # Large
            keras.random.normal((4, 8, 16)) * 100,              # Large random
            keras.random.normal((4, 8, 16)) * 1e-5,            # Small random
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
        out_dim = 64

        layer = SwinMLP(hidden_dim=hidden_dim, out_dim=out_dim)
        layer(sample_input)  # Build the layer

        # Should have exactly 4 weights: fc1_kernel, fc1_bias, fc2_kernel, fc2_bias
        assert len(layer.weights) == 4
        assert len(layer.trainable_variables) == 4

        # Verify weight shapes
        expected_shapes = [
            (input_dim, hidden_dim),  # fc1 kernel
            (hidden_dim,),            # fc1 bias
            (hidden_dim, out_dim),    # fc2 kernel
            (out_dim,)               # fc2 bias
        ]

        actual_shapes = [tuple(w.shape) for w in layer.weights]
        assert actual_shapes == expected_shapes


class TestSwinMLPEdgeCases:
    """Test edge cases and boundary conditions for SwinMLP."""

    def test_minimal_dimensions(self):
        """Test with minimal viable dimensions."""
        layer = SwinMLP(hidden_dim=1, out_dim=1)
        test_input = keras.random.normal([2, 3, 1])

        output = layer(test_input)
        assert output.shape == (2, 3, 1)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_large_dimensions(self):
        """Test with large dimensions."""
        layer = SwinMLP(hidden_dim=2048, out_dim=512)
        test_input = keras.random.normal([2, 4, 256])

        output = layer(test_input)
        assert output.shape == (2, 4, 512)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_zero_dropout_rate(self, sample_input=None):
        """Test with zero dropout rate."""
        if sample_input is None:
            sample_input = keras.random.normal([4, 8, 32])

        layer = SwinMLP(hidden_dim=64, dropout_rate=0.0)

        # Multiple calls should produce identical results
        output1 = layer(sample_input, training=True)
        output2 = layer(sample_input, training=True)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="should match"
        )

    def test_single_batch_single_sequence(self):
        """Test with minimal batch and sequence dimensions."""
        layer = SwinMLP(hidden_dim=32, out_dim=16)
        test_input = keras.random.normal([1, 1, 8])  # Single item, single step

        output = layer(test_input)
        assert output.shape == (1, 1, 16)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_identity_transformation(self):
        """Test configuration that approximates identity transformation."""
        # Create a layer that should approximate identity
        layer = SwinMLP(
            hidden_dim=128,  # Same as input
            out_dim=None,    # Preserve input dim
            activation='linear',
            dropout_rate=0.0,
            kernel_initializer='identity',  # Start with identity-like weights
            bias_initializer='zeros'
        )

        test_input = keras.random.normal([2, 4, 128])
        output = layer(test_input)

        assert output.shape == test_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_layer_reuse(self):
        """Test that the same layer instance can be reused."""
        layer = SwinMLP(hidden_dim=64, out_dim=32)

        # Use the same layer with different inputs
        input1 = keras.random.normal([2, 8, 16])
        input2 = keras.random.normal([3, 10, 16])

        output1 = layer(input1)
        output2 = layer(input2)

        assert output1.shape == (2, 8, 32)
        assert output2.shape == (3, 10, 32)

        # Both outputs should be valid
        assert not keras.ops.any(keras.ops.isnan(output1))
        assert not keras.ops.any(keras.ops.isnan(output2))