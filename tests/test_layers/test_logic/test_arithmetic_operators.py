import pytest
import tempfile
import os
import numpy as np
import keras
from keras import ops
from typing import Any, Dict

from dl_techniques.layers.logic.arithmetic_operators import LearnableArithmeticOperator


class TestLearnableArithmeticOperator:
    """Comprehensive test suite for LearnableArithmeticOperator."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'operation_types': ['add', 'multiply', 'subtract'],
            'use_temperature': True,
            'temperature_init': 1.0,
            'use_scaling': True,
            'scaling_init': 1.0
        }

    @pytest.fixture
    def minimal_config(self) -> Dict[str, Any]:
        """Minimal configuration for testing."""
        return {
            'operation_types': ['add', 'multiply']
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample input for testing."""
        return ops.convert_to_tensor(np.random.normal(0, 1, (4, 32)).astype(np.float32))

    @pytest.fixture
    def sample_input_2d(self) -> keras.KerasTensor:
        """Sample 2D input for testing."""
        return ops.convert_to_tensor(np.random.normal(0, 1, (4, 16, 16, 8)).astype(np.float32))

    @pytest.fixture
    def dual_inputs(self) -> tuple:
        """Dual inputs for binary operations testing."""
        x1 = ops.convert_to_tensor(np.random.normal(0, 1, (4, 32)).astype(np.float32))
        x2 = ops.convert_to_tensor(np.random.normal(0, 1, (4, 32)).astype(np.float32))
        return x1, x2

    def test_initialization(self, layer_config):
        """Test layer initialization."""
        layer = LearnableArithmeticOperator(**layer_config)

        assert hasattr(layer, 'operation_types')
        assert hasattr(layer, 'use_temperature')
        assert hasattr(layer, 'temperature_init')
        assert hasattr(layer, 'use_scaling')
        assert not layer.built
        assert layer.operation_weights is None  # Not built yet
        assert layer.temperature is None  # Not built yet
        assert layer.scaling_factor is None  # Not built yet

    def test_minimal_initialization(self, minimal_config):
        """Test layer initialization with minimal config."""
        layer = LearnableArithmeticOperator(**minimal_config)

        assert layer.operation_types == ['add', 'multiply']
        assert layer.use_temperature is True  # Default
        assert layer.use_scaling is True  # Default
        assert not layer.built

    def test_forward_pass_single_input(self, layer_config, sample_input):
        """Test forward pass with single input and building."""
        layer = LearnableArithmeticOperator(**layer_config)

        output = layer(sample_input)

        assert layer.built
        assert output.shape == sample_input.shape
        assert layer.operation_weights is not None
        assert layer.temperature is not None  # use_temperature=True
        assert layer.scaling_factor is not None  # use_scaling=True

        # Check that weights have correct shapes
        assert len(layer.operation_weights.shape) == 1
        assert layer.operation_weights.shape[0] == len(layer.operation_types)
        assert len(layer.temperature.shape) == 0  # Scalar
        assert len(layer.scaling_factor.shape) == 0  # Scalar

    def test_forward_pass_dual_inputs(self, layer_config, dual_inputs):
        """Test forward pass with dual inputs."""
        layer = LearnableArithmeticOperator(**layer_config)
        x1, x2 = dual_inputs

        output = layer([x1, x2])

        assert layer.built
        assert output.shape == x1.shape
        assert output.shape == x2.shape

    def test_forward_pass_2d_input(self, layer_config, sample_input_2d):
        """Test forward pass with 2D feature maps."""
        layer = LearnableArithmeticOperator(**layer_config)

        output = layer(sample_input_2d)

        assert layer.built
        assert output.shape == sample_input_2d.shape

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = LearnableArithmeticOperator(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_pred),
                ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_serialization_cycle_dual_inputs(self, layer_config, dual_inputs):
        """Test serialization with dual inputs."""
        x1, x2 = dual_inputs

        # Create model with dual inputs
        input1 = keras.Input(shape=x1.shape[1:], name='input1')
        input2 = keras.Input(shape=x2.shape[1:], name='input2')
        outputs = LearnableArithmeticOperator(**layer_config)([input1, input2])
        model = keras.Model([input1, input2], outputs)

        # Get original prediction
        original_pred = model([x1, x2])

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_dual_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model([x1, x2])

            # Verify identical predictions
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_pred),
                ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Dual input predictions differ after serialization"
            )

    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = LearnableArithmeticOperator(**layer_config)
        config = layer.get_config()

        # Check all config parameters are present
        expected_keys = {
            'operation_types', 'use_temperature', 'temperature_init',
            'use_scaling', 'scaling_init', 'operation_initializer',
            'temperature_initializer', 'scaling_initializer',
            'epsilon', 'power_clip_range', 'exponent_clip_range'
        }

        for key in expected_keys:
            assert key in config, f"Missing {key} in get_config()"

        # Check that operation_types match
        assert config['operation_types'] == layer_config['operation_types']
        assert config['use_temperature'] == layer_config['use_temperature']
        assert config['temperature_init'] == layer_config['temperature_init']

    def test_gradients_flow(self, layer_config, sample_input):
        """Test gradient computation."""
        import tensorflow as tf  # For GradientTape

        layer = LearnableArithmeticOperator(**layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

        # Should have gradients for operation_weights, temperature, scaling_factor
        expected_num_weights = 1  # operation_weights
        if layer.use_temperature:
            expected_num_weights += 1  # temperature
        if layer.use_scaling:
            expected_num_weights += 1  # scaling_factor

        assert len(gradients) == expected_num_weights

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_input, training):
        """Test behavior in different training modes."""
        layer = LearnableArithmeticOperator(**layer_config)

        output = layer(sample_input, training=training)
        assert output.shape == sample_input.shape

    def test_different_operation_combinations(self, sample_input):
        """Test different combinations of operations."""
        operation_sets = [
            ['add', 'multiply'],
            ['subtract', 'divide'],
            ['power', 'max', 'min'],
            ['add', 'multiply', 'subtract', 'divide', 'power', 'max', 'min']  # All
        ]

        for ops_set in operation_sets:
            layer = LearnableArithmeticOperator(operation_types=ops_set)
            output = layer(sample_input)
            assert output.shape == sample_input.shape

    def test_without_temperature_scaling(self, sample_input):
        """Test layer without temperature scaling."""
        layer = LearnableArithmeticOperator(
            operation_types=['add', 'multiply'],
            use_temperature=False
        )

        output = layer(sample_input)
        assert output.shape == sample_input.shape
        assert layer.temperature is None

    def test_without_scaling_factor(self, sample_input):
        """Test layer without output scaling."""
        layer = LearnableArithmeticOperator(
            operation_types=['add', 'multiply'],
            use_scaling=False
        )

        output = layer(sample_input)
        assert output.shape == sample_input.shape
        assert layer.scaling_factor is None

    def test_safe_divide_operation(self, sample_input):
        """Test safe division with potential zero denominators."""
        # Create input with some zeros
        zero_input = ops.convert_to_tensor(
            np.array([[0.0, 1.0, 0.0, 2.0]], dtype=np.float32)
        )

        layer = LearnableArithmeticOperator(operation_types=['divide'])
        output = layer(zero_input)

        # Should not contain NaN or Inf
        assert not ops.any(ops.isnan(output))
        assert not ops.any(ops.isinf(output))

    def test_safe_power_operation(self):
        """Test safe power operation with extreme values."""
        # Create inputs with extreme values
        extreme_input = ops.convert_to_tensor(
            np.array([[1e6, 1e-6, -1e6, 0.0]], dtype=np.float32)
        )

        layer = LearnableArithmeticOperator(operation_types=['power'])
        output = layer(extreme_input)

        # Should not contain NaN or Inf
        assert not ops.any(ops.isnan(output))
        assert not ops.any(ops.isinf(output))

    def test_edge_cases(self):
        """Test error conditions."""
        # Invalid operation types
        with pytest.raises(ValueError, match="Invalid operation types"):
            LearnableArithmeticOperator(operation_types=['invalid_op'])

        # Invalid temperature_init
        with pytest.raises(ValueError, match="temperature_init must be positive"):
            LearnableArithmeticOperator(temperature_init=0.0)

        # Invalid scaling_init
        with pytest.raises(ValueError, match="scaling_init must be positive"):
            LearnableArithmeticOperator(scaling_init=-1.0)

        # Invalid epsilon
        with pytest.raises(ValueError, match="epsilon must be positive"):
            LearnableArithmeticOperator(epsilon=0.0)

        # Invalid power_clip_range
        with pytest.raises(ValueError, match="power_clip_range must be"):
            LearnableArithmeticOperator(power_clip_range=(0.0, 1.0))  # min not > 0

        with pytest.raises(ValueError, match="power_clip_range must be"):
            LearnableArithmeticOperator(power_clip_range=(2.0, 1.0))  # min > max

        # Invalid exponent_clip_range
        with pytest.raises(ValueError, match="exponent_clip_range must be"):
            LearnableArithmeticOperator(exponent_clip_range=(2.0, 1.0))  # min > max

    def test_mismatched_input_shapes(self):
        """Test error with mismatched input shapes."""
        layer = LearnableArithmeticOperator(operation_types=['add'])

        x1 = ops.convert_to_tensor(np.random.normal(0, 1, (4, 32)).astype(np.float32))
        x2 = ops.convert_to_tensor(np.random.normal(0, 1, (4, 16)).astype(np.float32))

        with pytest.raises(ValueError, match="Input tensors must have the same shape"):
            layer.build([x1.shape, x2.shape])

    def test_too_many_inputs(self):
        """Test error with too many inputs."""
        layer = LearnableArithmeticOperator(operation_types=['add'])

        x1 = ops.convert_to_tensor(np.random.normal(0, 1, (4, 32)).astype(np.float32))
        x2 = ops.convert_to_tensor(np.random.normal(0, 1, (4, 32)).astype(np.float32))
        x3 = ops.convert_to_tensor(np.random.normal(0, 1, (4, 32)).astype(np.float32))

        with pytest.raises(ValueError, match="Expected 1 or 2 inputs"):
            layer([x1, x2, x3])

    def test_compute_output_shape(self, layer_config):
        """Test output shape computation."""
        layer = LearnableArithmeticOperator(**layer_config)

        # Single input
        input_shape = (None, 32, 32, 16)
        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape == input_shape

        # Dual inputs
        dual_input_shapes = [(None, 32, 32, 16), (None, 32, 32, 16)]
        output_shape = layer.compute_output_shape(dual_input_shapes)
        assert output_shape == dual_input_shapes[0]

    def test_custom_initializers(self, sample_input):
        """Test with custom initializers."""
        layer = LearnableArithmeticOperator(
            operation_types=['add', 'multiply'],
            operation_initializer='he_normal',
            temperature_initializer='constant',
            scaling_initializer='ones'
        )

        output = layer(sample_input)
        assert output.shape == sample_input.shape
        assert layer.built

    def test_deterministic_with_fixed_seed(self, sample_input):
        """Test deterministic behavior with fixed random seed."""
        keras.utils.set_random_seed(42)

        layer1 = LearnableArithmeticOperator(operation_types=['add', 'multiply'])
        output1 = layer1(sample_input)

        keras.utils.set_random_seed(42)

        layer2 = LearnableArithmeticOperator(operation_types=['add', 'multiply'])
        output2 = layer2(sample_input)

        # Should be identical with same seed
        np.testing.assert_allclose(
            ops.convert_to_numpy(output1),
            ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Outputs should be identical with same random seed"
        )