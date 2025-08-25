import pytest
import tempfile
import os
import numpy as np
import keras
from keras import ops
from typing import Any, Dict

from dl_techniques.layers.logic.neural_circuit import CircuitDepthLayer, LearnableNeuralCircuit


class TestCircuitDepthLayer:
    """Comprehensive test suite for CircuitDepthLayer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'num_logic_ops': 2,
            'num_arithmetic_ops': 2,
            'use_residual': True
        }

    @pytest.fixture
    def minimal_config(self) -> Dict[str, Any]:
        """Minimal configuration for testing."""
        return {
            'num_logic_ops': 1,
            'num_arithmetic_ops': 1
        }

    @pytest.fixture
    def sample_input_4d(self) -> keras.KerasTensor:
        """Sample 4D input for testing."""
        return ops.convert_to_tensor(np.random.normal(0, 1, (2, 16, 16, 8)).astype(np.float32))

    @pytest.fixture
    def large_input_4d(self) -> keras.KerasTensor:
        """Larger 4D input for testing."""
        return ops.convert_to_tensor(np.random.normal(0, 1, (4, 32, 32, 16)).astype(np.float32))

    def test_initialization(self, layer_config):
        """Test layer initialization."""
        layer = CircuitDepthLayer(**layer_config)

        assert hasattr(layer, 'num_logic_ops')
        assert hasattr(layer, 'num_arithmetic_ops')
        assert hasattr(layer, 'use_residual')
        assert not layer.built
        assert layer.logic_operators == []  # Empty until built
        assert layer.arithmetic_operators == []  # Empty until built
        assert layer.routing_weights is None  # Not built yet
        assert layer.combination_weights is None  # Not built yet

    def test_minimal_initialization(self, minimal_config):
        """Test layer initialization with minimal config."""
        layer = CircuitDepthLayer(**minimal_config)

        assert layer.num_logic_ops == 1
        assert layer.num_arithmetic_ops == 1
        assert layer.use_residual is True  # Default
        assert not layer.built

    def test_forward_pass_and_building(self, layer_config, sample_input_4d):
        """Test forward pass and building with 4D input."""
        layer = CircuitDepthLayer(**layer_config)

        output = layer(sample_input_4d)

        assert layer.built
        assert output.shape == sample_input_4d.shape

        # Check sub-layers were created
        assert len(layer.logic_operators) == layer_config['num_logic_ops']
        assert len(layer.arithmetic_operators) == layer_config['num_arithmetic_ops']

        # Check weights were created
        assert layer.routing_weights is not None
        assert layer.combination_weights is not None

        total_ops = layer_config['num_logic_ops'] + layer_config['num_arithmetic_ops']
        assert layer.routing_weights.shape[0] == total_ops
        assert layer.combination_weights.shape[0] == total_ops

    def test_residual_connection_effect(self, sample_input_4d):
        """Test effect of residual connections."""
        # Layer with residual connection
        layer_with_residual = CircuitDepthLayer(
            num_logic_ops=1,
            num_arithmetic_ops=1,
            use_residual=True
        )

        # Layer without residual connection
        layer_without_residual = CircuitDepthLayer(
            num_logic_ops=1,
            num_arithmetic_ops=1,
            use_residual=False
        )

        output_with = layer_with_residual(sample_input_4d)
        output_without = layer_without_residual(sample_input_4d)

        # Both should have correct shape
        assert output_with.shape == sample_input_4d.shape
        assert output_without.shape == sample_input_4d.shape

        # Outputs should generally be different due to residual connection
        assert not np.allclose(
            ops.convert_to_numpy(output_with),
            ops.convert_to_numpy(output_without),
            atol=1e-6
        )

    def test_different_operator_counts(self, sample_input_4d):
        """Test with different numbers of operators."""
        configs = [
            {'num_logic_ops': 1, 'num_arithmetic_ops': 1},
            {'num_logic_ops': 3, 'num_arithmetic_ops': 2},
            {'num_logic_ops': 2, 'num_arithmetic_ops': 4},
            {'num_logic_ops': 5, 'num_arithmetic_ops': 3}
        ]

        for config in configs:
            layer = CircuitDepthLayer(**config)
            output = layer(sample_input_4d)

            assert output.shape == sample_input_4d.shape
            assert len(layer.logic_operators) == config['num_logic_ops']
            assert len(layer.arithmetic_operators) == config['num_arithmetic_ops']

    def test_custom_operation_types(self, sample_input_4d):
        """Test with custom operation types."""
        layer = CircuitDepthLayer(
            num_logic_ops=2,
            num_arithmetic_ops=2,
            logic_op_types=['and', 'or'],
            arithmetic_op_types=['add', 'multiply']
        )

        output = layer(sample_input_4d)
        assert output.shape == sample_input_4d.shape

        # Check that operators were created with custom types
        for logic_op in layer.logic_operators:
            assert logic_op.operation_types == ['and', 'or']

        for arith_op in layer.arithmetic_operators:
            assert arith_op.operation_types == ['add', 'multiply']

    def test_serialization_cycle(self, layer_config, sample_input_4d):
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input_4d.shape[1:])
        outputs = CircuitDepthLayer(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input_4d)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_circuit_depth.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_4d)

            # Verify identical predictions
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_pred),
                ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Circuit depth predictions differ after serialization"
            )

    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = CircuitDepthLayer(**layer_config)
        config = layer.get_config()

        expected_keys = {
            'num_logic_ops', 'num_arithmetic_ops', 'use_residual',
            'logic_op_types', 'arithmetic_op_types',
            'routing_initializer', 'combination_initializer'
        }

        for key in expected_keys:
            assert key in config, f"Missing {key} in get_config()"

        # Check values match
        assert config['num_logic_ops'] == layer_config['num_logic_ops']
        assert config['num_arithmetic_ops'] == layer_config['num_arithmetic_ops']
        assert config['use_residual'] == layer_config['use_residual']

    def test_gradients_flow(self, layer_config, sample_input_4d):
        """Test gradient computation."""
        import tensorflow as tf

        layer = CircuitDepthLayer(**layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input_4d)
            output = layer(sample_input_4d)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

        # Should have gradients from routing, combination, and all sub-layer weights
        assert len(gradients) >= 2  # At least routing and combination weights

    def test_edge_cases(self):
        """Test error conditions."""
        # Invalid num_logic_ops
        with pytest.raises(ValueError, match="num_logic_ops must be positive"):
            CircuitDepthLayer(num_logic_ops=0, num_arithmetic_ops=1)

        # Invalid num_arithmetic_ops
        with pytest.raises(ValueError, match="num_arithmetic_ops must be positive"):
            CircuitDepthLayer(num_logic_ops=1, num_arithmetic_ops=0)

        # Non-4D input
        layer = CircuitDepthLayer(num_logic_ops=1, num_arithmetic_ops=1)
        invalid_input_3d = (None, 32, 16)  # Only 3D

        with pytest.raises(ValueError, match="CircuitDepthLayer expects 4D input"):
            layer.build(invalid_input_3d)

    def test_compute_output_shape(self, layer_config):
        """Test output shape computation."""
        layer = CircuitDepthLayer(**layer_config)

        input_shape = (None, 32, 32, 16)
        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape == input_shape


class TestLearnableNeuralCircuit:
    """Comprehensive test suite for LearnableNeuralCircuit."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'circuit_depth': 3,
            'num_logic_ops_per_depth': 2,
            'num_arithmetic_ops_per_depth': 2,
            'use_residual': False,
            'use_layer_norm': False
        }

    @pytest.fixture
    def minimal_config(self) -> Dict[str, Any]:
        """Minimal configuration for testing."""
        return {
            'circuit_depth': 2,
            'num_logic_ops_per_depth': 1,
            'num_arithmetic_ops_per_depth': 1
        }

    @pytest.fixture
    def sample_input_4d(self) -> keras.KerasTensor:
        """Sample 4D input for testing."""
        return ops.convert_to_tensor(np.random.normal(0, 1, (2, 16, 16, 8)).astype(np.float32))

    def test_initialization(self, layer_config):
        """Test layer initialization."""
        layer = LearnableNeuralCircuit(**layer_config)

        assert hasattr(layer, 'circuit_depth')
        assert hasattr(layer, 'num_logic_ops_per_depth')
        assert hasattr(layer, 'num_arithmetic_ops_per_depth')
        assert hasattr(layer, 'use_residual')
        assert hasattr(layer, 'use_layer_norm')
        assert not layer.built
        assert layer.circuit_layers == []  # Empty until built
        assert layer.layer_norms == []  # Empty until built

    def test_forward_pass_and_building(self, layer_config, sample_input_4d):
        """Test forward pass and building."""
        layer = LearnableNeuralCircuit(**layer_config)

        output = layer(sample_input_4d)

        assert layer.built
        assert output.shape == sample_input_4d.shape

        # Check circuit layers were created
        assert len(layer.circuit_layers) == layer_config['circuit_depth']

        # Check layer norms (should be empty since use_layer_norm=False)
        assert len(layer.layer_norms) == 0

        # Check each circuit layer
        for circuit_layer in layer.circuit_layers:
            assert isinstance(circuit_layer, CircuitDepthLayer)
            assert circuit_layer.built

    def test_with_layer_normalization(self, sample_input_4d):
        """Test with layer normalization enabled."""
        layer = LearnableNeuralCircuit(
            circuit_depth=2,
            num_logic_ops_per_depth=1,
            num_arithmetic_ops_per_depth=1,
            use_layer_norm=True
        )

        output = layer(sample_input_4d)

        assert output.shape == sample_input_4d.shape
        assert len(layer.layer_norms) == 2  # One per depth

        for norm_layer in layer.layer_norms:
            assert isinstance(norm_layer, keras.layers.LayerNormalization)

    def test_different_depths(self, sample_input_4d):
        """Test with different circuit depths."""
        depths = [1, 3, 5, 7]

        for depth in depths:
            layer = LearnableNeuralCircuit(
                circuit_depth=depth,
                num_logic_ops_per_depth=1,
                num_arithmetic_ops_per_depth=1
            )

            output = layer(sample_input_4d)

            assert output.shape == sample_input_4d.shape
            assert len(layer.circuit_layers) == depth

    def test_with_residual_connections(self, sample_input_4d):
        """Test with residual connections in depth layers."""
        layer_with_residual = LearnableNeuralCircuit(
            circuit_depth=2,
            num_logic_ops_per_depth=1,
            num_arithmetic_ops_per_depth=1,
            use_residual=True
        )

        layer_without_residual = LearnableNeuralCircuit(
            circuit_depth=2,
            num_logic_ops_per_depth=1,
            num_arithmetic_ops_per_depth=1,
            use_residual=False
        )

        output_with = layer_with_residual(sample_input_4d)
        output_without = layer_without_residual(sample_input_4d)

        assert output_with.shape == sample_input_4d.shape
        assert output_without.shape == sample_input_4d.shape

        # Check that circuit layers have correct residual setting
        for circuit_layer in layer_with_residual.circuit_layers:
            assert circuit_layer.use_residual is True

        for circuit_layer in layer_without_residual.circuit_layers:
            assert circuit_layer.use_residual is False

    def test_custom_operation_types(self, sample_input_4d):
        """Test with custom operation types."""
        layer = LearnableNeuralCircuit(
            circuit_depth=2,
            num_logic_ops_per_depth=1,
            num_arithmetic_ops_per_depth=1,
            logic_op_types=['and', 'or'],
            arithmetic_op_types=['add', 'multiply']
        )

        output = layer(sample_input_4d)
        assert output.shape == sample_input_4d.shape

        # Check that all circuit layers use custom operation types
        for circuit_layer in layer.circuit_layers:
            assert circuit_layer.logic_op_types == ['and', 'or']
            assert circuit_layer.arithmetic_op_types == ['add', 'multiply']

    def test_serialization_cycle(self, layer_config, sample_input_4d):
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input_4d.shape[1:])
        outputs = LearnableNeuralCircuit(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input_4d)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_neural_circuit.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_4d)

            # Verify identical predictions
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_pred),
                ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Neural circuit predictions differ after serialization"
            )

    def test_serialization_with_layer_norm(self, sample_input_4d):
        """Test serialization with layer normalization."""
        config = {
            'circuit_depth': 2,
            'num_logic_ops_per_depth': 1,
            'num_arithmetic_ops_per_depth': 1,
            'use_layer_norm': True
        }

        # Create model
        inputs = keras.Input(shape=sample_input_4d.shape[1:])
        outputs = LearnableNeuralCircuit(**config)(inputs)
        model = keras.Model(inputs, outputs)

        original_pred = model(sample_input_4d)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_circuit_norm.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_4d)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_pred),
                ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Circuit with layer norm predictions differ after serialization"
            )

    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = LearnableNeuralCircuit(**layer_config)
        config = layer.get_config()

        expected_keys = {
            'circuit_depth', 'num_logic_ops_per_depth', 'num_arithmetic_ops_per_depth',
            'use_residual', 'use_layer_norm', 'logic_op_types', 'arithmetic_op_types',
            'routing_initializer', 'combination_initializer'
        }

        for key in expected_keys:
            assert key in config, f"Missing {key} in get_config()"

        # Check values match
        assert config['circuit_depth'] == layer_config['circuit_depth']
        assert config['num_logic_ops_per_depth'] == layer_config['num_logic_ops_per_depth']
        assert config['num_arithmetic_ops_per_depth'] == layer_config['num_arithmetic_ops_per_depth']

    def test_gradients_flow(self, minimal_config, sample_input_4d):
        """Test gradient computation."""
        import tensorflow as tf

        layer = LearnableNeuralCircuit(**minimal_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input_4d)
            output = layer(sample_input_4d)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, minimal_config, sample_input_4d, training):
        """Test behavior in different training modes."""
        layer = LearnableNeuralCircuit(**minimal_config)

        output = layer(sample_input_4d, training=training)
        assert output.shape == sample_input_4d.shape

    def test_edge_cases(self):
        """Test error conditions."""
        # Invalid circuit_depth
        with pytest.raises(ValueError, match="circuit_depth must be positive"):
            LearnableNeuralCircuit(circuit_depth=0)

        # Invalid num_logic_ops_per_depth
        with pytest.raises(ValueError, match="num_logic_ops_per_depth must be positive"):
            LearnableNeuralCircuit(num_logic_ops_per_depth=0)

        # Invalid num_arithmetic_ops_per_depth
        with pytest.raises(ValueError, match="num_arithmetic_ops_per_depth must be positive"):
            LearnableNeuralCircuit(num_arithmetic_ops_per_depth=0)

        # Non-4D input
        layer = LearnableNeuralCircuit(circuit_depth=1)
        invalid_input_3d = (None, 32, 16)  # Only 3D

        with pytest.raises(ValueError, match="LearnableNeuralCircuit expects 4D input"):
            layer.build(invalid_input_3d)

    def test_compute_output_shape(self, layer_config):
        """Test output shape computation."""
        layer = LearnableNeuralCircuit(**layer_config)

        input_shape = (None, 32, 32, 16)
        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape == input_shape

    def test_complex_configuration(self, sample_input_4d):
        """Test with complex configuration combining all features."""
        layer = LearnableNeuralCircuit(
            circuit_depth=3,
            num_logic_ops_per_depth=3,
            num_arithmetic_ops_per_depth=2,
            use_residual=True,
            use_layer_norm=True,
            logic_op_types=['and', 'or', 'xor'],
            arithmetic_op_types=['add', 'multiply', 'subtract'],
            routing_initializer='he_normal',
            combination_initializer='glorot_uniform'
        )

        output = layer(sample_input_4d)

        assert output.shape == sample_input_4d.shape
        assert len(layer.circuit_layers) == 3
        assert len(layer.layer_norms) == 3
        assert layer.built

    def test_deterministic_with_fixed_seed(self, minimal_config, sample_input_4d):
        """Test deterministic behavior with fixed random seed."""
        keras.utils.set_random_seed(42)

        layer1 = LearnableNeuralCircuit(**minimal_config)
        output1 = layer1(sample_input_4d)

        keras.utils.set_random_seed(42)

        layer2 = LearnableNeuralCircuit(**minimal_config)
        output2 = layer2(sample_input_4d)

        # Should be identical with same seed
        np.testing.assert_allclose(
            ops.convert_to_numpy(output1),
            ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Outputs should be identical with same random seed"
        )

    def test_large_circuit(self):
        """Test with large circuit configuration."""
        large_input = ops.convert_to_tensor(
            np.random.normal(0, 1, (2, 64, 64, 32)).astype(np.float32)
        )

        layer = LearnableNeuralCircuit(
            circuit_depth=5,
            num_logic_ops_per_depth=4,
            num_arithmetic_ops_per_depth=4,
            use_residual=True,
            use_layer_norm=True
        )

        output = layer(large_input)

        assert output.shape == large_input.shape
        assert len(layer.circuit_layers) == 5
        assert len(layer.layer_norms) == 5

    def test_custom_initializers(self, sample_input_4d):
        """Test with custom initializers."""
        layer = LearnableNeuralCircuit(
            circuit_depth=2,
            num_logic_ops_per_depth=1,
            num_arithmetic_ops_per_depth=1,
            routing_initializer='he_normal',
            combination_initializer='glorot_normal'
        )

        output = layer(sample_input_4d)
        assert output.shape == sample_input_4d.shape

        # Check that circuit layers use the custom initializers
        for circuit_layer in layer.circuit_layers:
            assert isinstance(circuit_layer.routing_initializer, keras.initializers.HeNormal)
            assert isinstance(circuit_layer.combination_initializer, keras.initializers.GlorotNormal)
