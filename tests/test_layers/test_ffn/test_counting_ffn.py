import pytest
import numpy as np
import keras
import tempfile
import os
from typing import Dict, Any
import tensorflow as tf

from dl_techniques.layers.ffn.counting_ffn import CountingFFN


class TestCountingFFN:
    """Comprehensive test suite for modern CountingFFN layer implementation."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample input tensor for testing."""
        return keras.random.normal([4, 12, 64])  # batch, seq_len, features

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard layer configuration for testing."""
        return {
            'output_dim': 64,
            'count_dim': 32,
            'counting_scope': 'local',
            'activation': 'gelu'
        }

    @pytest.fixture
    def custom_layer_config(self) -> Dict[str, Any]:
        """Custom layer configuration with comprehensive parameters."""
        return {
            'output_dim': 128,
            'count_dim': 64,
            'counting_scope': 'causal',
            'activation': 'swish',
            'use_bias': True,
            'kernel_initializer': 'he_normal',
            'bias_initializer': 'ones',
            'kernel_regularizer': keras.regularizers.L2(1e-4),
            'bias_regularizer': keras.regularizers.L1(1e-5)
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = CountingFFN(output_dim=128, count_dim=64)

        # Check stored configuration
        assert layer.output_dim == 128
        assert layer.count_dim == 64
        assert layer.counting_scope == "local"
        assert layer._activation_identifier == "gelu"
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None

        # Check that layer is not built yet
        assert not layer.built

        # Check that sub-layers are created but not built
        assert layer.key_projection is not None
        assert layer.count_transform is not None
        assert layer.gate is not None
        assert not layer.key_projection.built
        assert not layer.count_transform.built
        assert not layer.gate.built

    def test_initialization_custom(self, custom_layer_config):
        """Test initialization with comprehensive custom parameters."""
        layer = CountingFFN(**custom_layer_config)

        # Verify all custom parameters are stored correctly
        assert layer.output_dim == 128
        assert layer.count_dim == 64
        assert layer.counting_scope == 'causal'
        assert layer._activation_identifier == 'swish'
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)
        assert isinstance(layer.kernel_regularizer, keras.regularizers.L2)
        assert isinstance(layer.bias_regularizer, keras.regularizers.L1)

    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid output_dim values
        with pytest.raises(ValueError, match="output_dim must be positive"):
            CountingFFN(output_dim=0, count_dim=32)

        with pytest.raises(ValueError, match="output_dim must be positive"):
            CountingFFN(output_dim=-10, count_dim=32)

        # Test invalid count_dim values
        with pytest.raises(ValueError, match="count_dim must be positive"):
            CountingFFN(output_dim=64, count_dim=0)

        with pytest.raises(ValueError, match="count_dim must be positive"):
            CountingFFN(output_dim=64, count_dim=-5)

        # Test invalid counting_scope values
        with pytest.raises(ValueError, match="counting_scope must be one of"):
            CountingFFN(output_dim=64, count_dim=32, counting_scope='invalid')

        with pytest.raises(ValueError, match="counting_scope must be one of"):
            CountingFFN(output_dim=64, count_dim=32, counting_scope='bidirectional')

    def test_build_process(self, sample_input, layer_config):
        """Test that the layer builds properly following modern patterns."""
        layer = CountingFFN(**layer_config)

        # Layer should not be built initially
        assert not layer.built

        # Forward pass triggers build
        output = layer(sample_input)

        # Check that layer is now built
        assert layer.built

        # Check that all sub-layers are built
        assert layer.key_projection.built
        assert layer.count_transform.built
        assert layer.gate.built

        # Verify weights were created
        expected_weight_count = 6  # 3 Dense layers × 2 weights each (kernel + bias)
        assert len(layer.weights) == expected_weight_count

        # Verify output shape matches expected
        assert output.shape == sample_input.shape

    def test_build_input_validation(self):
        """Test build method input validation."""
        layer = CountingFFN(output_dim=64, count_dim=32)

        # Test with invalid input shapes
        with pytest.raises(ValueError, match="Input must be at least 2D"):
            layer.build((10,))  # 1D input

        with pytest.raises(ValueError, match="Input feature dimension must be specified"):
            layer.build((None, 10, None))  # Undefined last dimension

    def test_forward_pass_basic(self, sample_input, layer_config):
        """Test basic forward pass functionality."""
        layer = CountingFFN(**layer_config)
        output = layer(sample_input)

        # Basic sanity checks
        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

    def test_counting_scope_global(self, sample_input):
        """Test global counting scope behavior."""
        layer = CountingFFN(
            output_dim=64,
            count_dim=16,
            counting_scope='global',
            activation='linear',
            kernel_initializer='ones',
            bias_initializer='zeros'
        )

        output = layer(sample_input)

        # Global scope should produce same count information for all positions
        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_counting_scope_causal(self, sample_input):
        """Test causal counting scope behavior."""
        layer = CountingFFN(
            output_dim=64,
            count_dim=16,
            counting_scope='causal',
            activation='linear'
        )

        output = layer(sample_input)

        # Causal scope should work with cumulative sums
        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_counting_scope_local(self, sample_input):
        """Test local (bidirectional) counting scope behavior."""
        layer = CountingFFN(
            output_dim=64,
            count_dim=16,
            counting_scope='local',
            activation='linear'
        )

        output = layer(sample_input)

        # Local scope uses bidirectional cumulative sums
        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_different_counting_scopes_consistency(self, sample_input):
        """Test that different counting scopes all produce valid outputs."""
        scopes = ['global', 'causal', 'local']
        outputs = {}

        for scope in scopes:
            layer = CountingFFN(
                output_dim=64,
                count_dim=16,
                counting_scope=scope,
                activation='relu'
            )
            outputs[scope] = layer(sample_input)

            # All outputs should have same shape and be valid
            assert outputs[scope].shape == sample_input.shape
            assert not keras.ops.any(keras.ops.isnan(outputs[scope]))

        # Outputs should be different for different scopes
        for i, scope1 in enumerate(scopes):
            for scope2 in scopes[i + 1:]:
                # Check that outputs are actually different
                diff = keras.ops.mean(keras.ops.abs(outputs[scope1] - outputs[scope2]))
                assert keras.ops.convert_to_numpy(diff) > 1e-6

    def test_different_activations(self, sample_input):
        """Test layer with various activation functions."""
        activations = ['relu', 'gelu', 'swish', 'tanh', 'linear', 'selu']

        for activation in activations:
            layer = CountingFFN(
                output_dim=64,
                count_dim=32,
                activation=activation
            )
            output = layer(sample_input)

            # Verify output is valid
            assert output.shape == sample_input.shape
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_custom_activation_callable(self, sample_input):
        """Test with custom activation function as callable."""

        def custom_activation(x):
            return keras.ops.relu(x) * 0.5

        layer = CountingFFN(
            output_dim=64,
            count_dim=32,
            activation=custom_activation
        )
        output = layer(sample_input)

        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_gating_mechanism(self, sample_input):
        """Test that the gating mechanism works properly."""
        # Create layer with extreme gate initialization to test blending
        layer = CountingFFN(
            output_dim=64,
            count_dim=16,
            activation='linear',
            kernel_initializer='ones',
            bias_initializer='zeros'
        )

        output = layer(sample_input)

        # Output should be a blend of original input and transformed counts
        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

        # The output should be different from the input (gating should have effect)
        input_mean = keras.ops.mean(sample_input)
        output_mean = keras.ops.mean(output)
        assert keras.ops.convert_to_numpy(keras.ops.abs(input_mean - output_mean)) > 1e-6

    def test_different_output_dimensions(self, sample_input):
        """Test with different output dimensions."""
        input_dim = sample_input.shape[-1]  # 64

        test_cases = [
            32,  # Smaller than input
            64,  # Same as input
            128,  # Larger than input
        ]

        for output_dim in test_cases:
            layer = CountingFFN(
                output_dim=output_dim,
                count_dim=16
            )
            output = layer(sample_input)

            expected_shape = list(sample_input.shape)
            expected_shape[-1] = output_dim
            assert output.shape == tuple(expected_shape)
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_compute_output_shape(self):
        """Test output shape computation."""
        test_cases = [
            # (output_dim, count_dim, input_shape, expected_output_shape)
            (64, 32, (None, 16, 128), (None, 16, 64)),
            (128, 64, (4, 8, 256), (4, 8, 128)),
            (32, 16, (2, 10, 20, 64), (2, 10, 20, 32)),
            (256, 128, (1, 4, 8, 16, 128), (1, 4, 8, 16, 256)),
        ]

        for output_dim, count_dim, input_shape, expected_shape in test_cases:
            layer = CountingFFN(output_dim=output_dim, count_dim=count_dim)
            computed_shape = layer.compute_output_shape(input_shape)
            assert computed_shape == expected_shape

    def test_get_config_completeness(self, custom_layer_config):
        """Test that get_config contains all initialization parameters."""
        layer = CountingFFN(**custom_layer_config)
        config = layer.get_config()

        # Verify all custom parameters are in config
        expected_keys = {
            'output_dim', 'count_dim', 'counting_scope', 'activation',
            'use_bias', 'kernel_initializer', 'bias_initializer',
            'kernel_regularizer', 'bias_regularizer'
        }

        config_keys = set(config.keys())
        assert expected_keys.issubset(config_keys)

        # Verify specific values
        assert config['output_dim'] == 128
        assert config['count_dim'] == 64
        assert config['counting_scope'] == 'causal'

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle following modern patterns."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = CountingFFN(**layer_config)(inputs)
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

        # Multi-layer architecture with different counting scopes
        x = CountingFFN(output_dim=128, count_dim=32, counting_scope='global')(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = CountingFFN(output_dim=128, count_dim=64, counting_scope='causal')(x)
        x = keras.layers.LayerNormalization()(x)
        x = CountingFFN(output_dim=64, count_dim=32, counting_scope='local')(x)
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
        layer = CountingFFN(
            output_dim=64,
            count_dim=32,
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
        layer = CountingFFN(**layer_config)

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
        layer = CountingFFN(**layer_config)
        output = layer(sample_input, training=training)

        # Basic output validation
        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

    @pytest.mark.parametrize("counting_scope", ["global", "causal", "local"])
    def test_all_counting_scopes(self, sample_input, counting_scope):
        """Test all counting scopes systematically."""
        layer = CountingFFN(
            output_dim=64,
            count_dim=32,
            counting_scope=counting_scope
        )
        output = layer(sample_input)

        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_different_input_dimensions(self):
        """Test layer with different input tensor dimensions."""
        layer = CountingFFN(output_dim=32, count_dim=16)

        test_shapes = [
            (4, 64),  # 2D input
            (4, 16, 64),  # 3D input
            (2, 8, 16, 64),  # 4D input
            (1, 4, 8, 16, 64)  # 5D input
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
        layer = CountingFFN(output_dim=32, count_dim=16, activation='gelu')

        # Test different input value ranges
        test_cases = [
            keras.ops.zeros((4, 8, 64)),  # Zeros
            keras.ops.ones((4, 8, 64)) * 1e-10,  # Very small
            keras.ops.ones((4, 8, 64)) * 1e3,  # Large
            keras.random.normal((4, 8, 64)) * 100,  # Large random
            keras.random.normal((4, 8, 64)) * 1e-5,  # Small random
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Verify numerical stability
            assert not keras.ops.any(keras.ops.isnan(output)), "NaN detected"
            assert not keras.ops.any(keras.ops.isinf(output)), "Inf detected"

    def test_weight_structure_and_shapes(self, sample_input):
        """Test that layer weights have correct structure and shapes."""
        input_dim = sample_input.shape[-1]  # 64
        output_dim = 128
        count_dim = 32

        layer = CountingFFN(output_dim=output_dim, count_dim=count_dim, counting_scope='local')
        layer(sample_input)  # Build the layer

        # Should have exactly 6 weights: 3 Dense layers × 2 weights each
        assert len(layer.weights) == 6
        assert len(layer.trainable_variables) == 6

        # Verify weight shapes
        expected_shapes = [
            (input_dim, count_dim),  # key_projection kernel
            (count_dim,),  # key_projection bias
            (count_dim * 2, output_dim),  # count_transform kernel (local uses 2 × count_dim)
            (output_dim,),  # count_transform bias
            (input_dim, output_dim),  # gate kernel
            (output_dim,)  # gate bias
        ]

        actual_shapes = [tuple(w.shape) for w in layer.weights]
        assert actual_shapes == expected_shapes

    def test_count_transform_input_dimension_logic(self, sample_input):
        """Test that count_transform receives correct input dimensions for different scopes."""
        input_dim = sample_input.shape[-1]
        count_dim = 16

        # Test each scope
        scope_configs = {
            'global': count_dim,  # Uses count_dim
            'causal': count_dim,  # Uses count_dim
            'local': count_dim * 2  # Uses count_dim * 2 (bidirectional)
        }

        for scope, expected_input_dim in scope_configs.items():
            layer = CountingFFN(
                output_dim=64,
                count_dim=count_dim,
                counting_scope=scope
            )
            layer(sample_input)  # Build the layer

            # Check count_transform input dimension
            count_transform_kernel_shape = layer.count_transform.kernel.shape
            assert count_transform_kernel_shape[0] == expected_input_dim


class TestCountingFFNEdgeCases:
    """Test edge cases and boundary conditions for CountingFFN."""

    def test_minimal_dimensions(self):
        """Test with minimal viable dimensions."""
        layer = CountingFFN(output_dim=1, count_dim=1)
        test_input = keras.random.normal([2, 3, 1])

        output = layer(test_input)
        assert output.shape == (2, 3, 1)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_large_dimensions(self):
        """Test with large dimensions."""
        layer = CountingFFN(output_dim=512, count_dim=256)
        test_input = keras.random.normal([2, 4, 128])

        output = layer(test_input)
        assert output.shape == (2, 4, 512)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_single_batch_single_sequence(self):
        """Test with minimal batch and sequence dimensions."""
        layer = CountingFFN(output_dim=32, count_dim=16)
        test_input = keras.random.normal([1, 1, 8])  # Single item, single step

        output = layer(test_input)
        assert output.shape == (1, 1, 32)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_very_long_sequence(self):
        """Test with very long sequences."""
        layer = CountingFFN(output_dim=32, count_dim=16, counting_scope='causal')
        test_input = keras.random.normal([1, 1000, 8])  # Very long sequence

        output = layer(test_input)
        assert output.shape == (1, 1000, 32)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_layer_reuse(self):
        """Test that the same layer instance can be reused."""
        layer = CountingFFN(output_dim=32, count_dim=16)

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

    def test_deterministic_behavior_with_fixed_inputs(self):
        """Test deterministic behavior with controlled inputs."""
        layer = CountingFFN(
            output_dim=16,
            count_dim=8,
            activation='linear',
            kernel_initializer='ones',
            bias_initializer='zeros'
        )

        # Fixed input should produce deterministic output
        fixed_input = keras.ops.ones([2, 4, 8])

        output1 = layer(fixed_input, training=False)
        output2 = layer(fixed_input, training=False)

        # Outputs should be identical
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="should match"
        )

    def test_counting_mechanism_sanity_check(self):
        """Test that counting mechanism produces different outputs for different sequences."""
        layer = CountingFFN(
            output_dim=32,
            count_dim=16,
            counting_scope='causal',
            activation='relu'
        )

        # Create two different input patterns
        input1 = keras.ops.ones([1, 5, 8])  # Uniform pattern
        input2 = keras.random.normal([1, 5, 8])  # Random pattern

        output1 = layer(input1)
        output2 = layer(input2)

        # Outputs should be different
        diff = keras.ops.mean(keras.ops.abs(output1 - output2))
        assert keras.ops.convert_to_numpy(diff) > 1e-6

    def test_sequence_position_sensitivity(self):
        """Test that the layer is sensitive to sequence position (especially for causal scope)."""
        layer = CountingFFN(
            output_dim=16,
            count_dim=8,
            counting_scope='causal',
            activation='linear'
        )

        # Create input where each position has different values
        seq_len = 6
        test_input = keras.ops.zeros([1, seq_len, 4])

        # Set each position to a different value
        for i in range(seq_len):
            test_input = keras.ops.scatter_update(
                test_input,
                [[0, i, 0]],
                [float(i + 1)]
            )

        output = layer(test_input)

        # For causal counting, later positions should have different values than earlier ones
        first_pos = output[0, 0, :]
        last_pos = output[0, -1, :]

        position_diff = keras.ops.mean(keras.ops.abs(first_pos - last_pos))
        assert keras.ops.convert_to_numpy(position_diff) > 1e-6