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
        # Post plan_2026-05-13_a2b0f17b/D-002: children created in __init__,
        # built lazily on first __call__ via parent.build().
        assert len(layer.logic_operators) == layer.num_logic_ops
        assert len(layer.arithmetic_operators) == layer.num_arithmetic_ops
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

        # In default 'output_only' routing mode, `routing_weights` is created
        # for serialization compatibility but unused at call time, so its
        # gradient is None. All OTHER variables must have gradients.
        for var, grad in zip(layer.trainable_variables, gradients):
            if "routing_weights" in var.name and layer.circuit_routing == "output_only":
                continue
            assert grad is not None, f"None gradient for {var.name}"
        assert len(gradients) >= 2

    def test_edge_cases(self):
        """Test error conditions."""
        # Invalid num_logic_ops
        with pytest.raises(ValueError, match="num_logic_ops must be positive"):
            CircuitDepthLayer(num_logic_ops=0, num_arithmetic_ops=1)

        # Invalid num_arithmetic_ops
        with pytest.raises(ValueError, match="num_arithmetic_ops must be positive"):
            CircuitDepthLayer(num_logic_ops=1, num_arithmetic_ops=0)

        # Rank < 2 input rejected (rank-relaxed contract: rank >= 2 allowed).
        layer = CircuitDepthLayer(num_logic_ops=1, num_arithmetic_ops=1)
        invalid_input_1d = (16,)

        with pytest.raises(ValueError, match="rank >= 2"):
            layer.build(invalid_input_1d)

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
        # Children created eagerly in __init__ post plan_2026-05-13_a2b0f17b/D-002.
        assert len(layer.circuit_layers) == layer.circuit_depth
        expected_norms = layer.circuit_depth if layer.use_layer_norm else 0
        assert len(layer.layer_norms) == expected_norms

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

        # Routing weights are unused in 'output_only' (the new default).
        for var, grad in zip(layer.trainable_variables, gradients):
            if "routing_weights" in var.name and layer.circuit_routing == "output_only":
                continue
            assert grad is not None, f"None gradient for {var.name}"
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

        # Rank < 2 input rejected (rank-relaxed contract: rank >= 2 allowed).
        layer = LearnableNeuralCircuit(circuit_depth=1)
        invalid_input_1d = (16,)

        with pytest.raises(ValueError, match="rank >= 2"):
            layer.build(invalid_input_1d)

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


# ---------------------------------------------------------------------
# Rank-relaxation tests (rank >= 2 supported after iter-1).
# ---------------------------------------------------------------------


class TestRankRelaxation:
    """Verify CircuitDepthLayer / LearnableNeuralCircuit accept rank-2 and rank-3 inputs."""

    @pytest.mark.parametrize("layer_cls", [CircuitDepthLayer, LearnableNeuralCircuit])
    def test_rank2_input_forward_and_shape(self, layer_cls):
        x = ops.convert_to_tensor(np.random.normal(0, 1, (3, 16)).astype(np.float32))
        if layer_cls is CircuitDepthLayer:
            layer = layer_cls(num_logic_ops=2, num_arithmetic_ops=2, use_residual=True)
        else:
            layer = layer_cls(circuit_depth=2, num_logic_ops_per_depth=2, num_arithmetic_ops_per_depth=2)
        y = layer(x)
        assert tuple(y.shape) == (3, 16)

    @pytest.mark.parametrize("layer_cls", [CircuitDepthLayer, LearnableNeuralCircuit])
    def test_rank3_input_forward_and_shape(self, layer_cls):
        x = ops.convert_to_tensor(np.random.normal(0, 1, (2, 7, 16)).astype(np.float32))
        if layer_cls is CircuitDepthLayer:
            layer = layer_cls(num_logic_ops=2, num_arithmetic_ops=2, use_residual=True)
        else:
            layer = layer_cls(circuit_depth=2, num_logic_ops_per_depth=2, num_arithmetic_ops_per_depth=2)
        y = layer(x)
        assert tuple(y.shape) == (2, 7, 16)

    @pytest.mark.parametrize("layer_cls", [CircuitDepthLayer, LearnableNeuralCircuit])
    def test_rank1_input_rejected(self, layer_cls):
        # rank < 2 must still raise
        if layer_cls is CircuitDepthLayer:
            layer = layer_cls(num_logic_ops=1, num_arithmetic_ops=1)
        else:
            layer = layer_cls(circuit_depth=1, num_logic_ops_per_depth=1, num_arithmetic_ops_per_depth=1)
        with pytest.raises(ValueError, match="rank >= 2"):
            layer.build((16,))

    def test_rank2_circuit_depth_save_load(self):
        layer = CircuitDepthLayer(num_logic_ops=2, num_arithmetic_ops=2, use_residual=True)
        inp = keras.Input(shape=(16,))
        out = layer(inp)
        model = keras.Model(inp, out)

        x = ops.convert_to_tensor(np.random.normal(0, 1, (3, 16)).astype(np.float32))
        y1 = model(x)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            y2 = reloaded(x)
        np.testing.assert_allclose(ops.convert_to_numpy(y1), ops.convert_to_numpy(y2), atol=1e-6)

    def test_rank3_neural_circuit_save_load(self):
        layer = LearnableNeuralCircuit(circuit_depth=2)
        inp = keras.Input(shape=(7, 16))
        out = layer(inp)
        model = keras.Model(inp, out)

        x = ops.convert_to_tensor(np.random.normal(0, 1, (2, 7, 16)).astype(np.float32))
        y1 = model(x)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            y2 = reloaded(x)
        np.testing.assert_allclose(ops.convert_to_numpy(y1), ops.convert_to_numpy(y2), atol=1e-6)


# ---------------------------------------------------------------------------
# Regression tests added in plan_2026-05-13_e52a5ac8
# ---------------------------------------------------------------------------

class TestPlanE52a5ac8Regressions:
    """Lock in fixes from plan_2026-05-13_e52a5ac8."""

    def test_default_use_residual_is_true(self):
        """C1: LearnableNeuralCircuit() default must use residuals.

        Original default was False, contradicting the layer docstring which
        promised residuals would "stabilize gradient flow in deeply stacked
        circuits." Default-False shrank ||y||/||x|| to ~0.38 at init.
        """
        layer = LearnableNeuralCircuit()
        assert layer.use_residual is True
        # Inner CircuitDepthLayer instances inherit the value.
        x = ops.convert_to_tensor(np.random.normal(0, 1, (2, 8)).astype(np.float32))
        _ = layer(x)
        for circuit_layer in layer.circuit_layers:
            assert circuit_layer.use_residual is True

    def test_signal_preserved_with_default_config(self):
        """C1 regression: default LearnableNeuralCircuit preserves signal magnitude."""
        keras.utils.set_random_seed(0)
        layer = LearnableNeuralCircuit()
        x = ops.convert_to_tensor(np.random.normal(0, 1, (4, 16)).astype(np.float32))
        y = layer(x)
        ratio = float(ops.norm(y)) / float(ops.norm(x))
        # With residual, init output should be at least ~0.5x input norm.
        # Pre-fix value was 0.38; with residual, observed ~1.7.
        assert ratio > 0.5, f"signal collapsed at init: ||y||/||x|| = {ratio:.3f}"

    def test_compute_output_shape_accepts_list_form(self):
        """M2: compute_output_shape must handle list-form single shapes."""
        layer = CircuitDepthLayer()
        # Single shape deserialized as list — previously returned None.
        out_list = layer.compute_output_shape([None, 32])
        out_tuple = layer.compute_output_shape((None, 32))
        # CircuitDepthLayer is shape-preserving and was already correct;
        # this just confirms no regression.
        assert tuple(out_list) == tuple(out_tuple) == (None, 32)


# ---------------------------------------------------------------------------
# Regression tests added in plan_2026-05-13_a2b0f17b
# ---------------------------------------------------------------------------

class TestPlanA2b0f17bCircuit:
    """Regressions for full-rewrite plan."""

    def test_default_routing_is_output_only(self):
        layer = CircuitDepthLayer()
        assert layer.circuit_routing == "output_only"

    def test_classic_routing_preserves_attenuation(self):
        """Classic mode should still attenuate signal (~1/N before residual)."""
        np.random.seed(0)
        x = ops.convert_to_tensor(np.random.randn(2, 16).astype(np.float32))
        layer = CircuitDepthLayer(
            num_logic_ops=2, num_arithmetic_ops=2,
            use_residual=False, circuit_routing='classic',
        )
        y = layer(x)
        ratio = float(ops.norm(y)) / float(ops.norm(x))
        assert ratio < 0.6, f"classic mode should attenuate, got {ratio:.3f}"

    def test_output_only_routing_does_not_attenuate(self):
        """Output-only mode should preserve magnitude better than 0.6×."""
        np.random.seed(0)
        x = ops.convert_to_tensor(np.random.randn(2, 16).astype(np.float32))
        layer = CircuitDepthLayer(
            num_logic_ops=2, num_arithmetic_ops=2,
            use_residual=False, circuit_routing='output_only',
        )
        y = layer(x)
        ratio = float(ops.norm(y)) / float(ops.norm(x))
        # Soft signal not attenuated by 1/N input gating.
        assert ratio > 0.5, f"output-only mode unexpectedly attenuated: {ratio:.3f}"

    def test_load_balance_loss_added_when_coef_positive(self):
        layer = CircuitDepthLayer(load_balance_coefficient=0.1)
        x = ops.convert_to_tensor(np.random.randn(2, 8).astype(np.float32))
        _ = layer(x)
        assert len(layer.losses) >= 1
        assert float(layer.losses[0]) > 0

    def test_load_balance_loss_absent_when_coef_zero(self):
        layer = CircuitDepthLayer()
        x = ops.convert_to_tensor(np.random.randn(2, 8).astype(np.float32))
        _ = layer(x)
        # No reg → no losses (children also have entropy_coefficient=0).
        assert len(layer.losses) == 0

    def test_channel_mix_dense_preserves_shape(self):
        layer = CircuitDepthLayer(channel_mix='dense')
        x = ops.convert_to_tensor(np.random.randn(2, 16).astype(np.float32))
        y = layer(x)
        assert y.shape == (2, 16)
        # Channel-mix layer should add new weights.
        assert layer._channel_mix_layer is not None
        assert len(layer._channel_mix_layer.trainable_variables) >= 1

    def test_invalid_circuit_routing_raises(self):
        with pytest.raises(ValueError, match="circuit_routing"):
            CircuitDepthLayer(circuit_routing='nope')

    def test_invalid_channel_mix_raises(self):
        with pytest.raises(ValueError, match="channel_mix"):
            CircuitDepthLayer(channel_mix='conv7x7')

    def test_negative_load_balance_raises(self):
        # After H6 (plan_3a2f1d23), the message names the canonical key.
        with pytest.raises(ValueError, match="gate_entropy_coefficient"):
            CircuitDepthLayer(gate_entropy_coefficient=-0.1)

    def test_neural_circuit_apply_sigmoid_first_only(self):
        """C2: with first_only mode, only depth 0 gets apply_sigmoid=True."""
        nc = LearnableNeuralCircuit(circuit_depth=3, apply_sigmoid_per_depth='first_only')
        assert nc.circuit_layers[0].apply_sigmoid is True
        assert nc.circuit_layers[1].apply_sigmoid is False
        assert nc.circuit_layers[2].apply_sigmoid is False

    def test_neural_circuit_apply_sigmoid_all(self):
        nc = LearnableNeuralCircuit(circuit_depth=3, apply_sigmoid_per_depth='all')
        for depth_layer in nc.circuit_layers:
            assert depth_layer.apply_sigmoid is True

    def test_neural_circuit_apply_sigmoid_none(self):
        nc = LearnableNeuralCircuit(circuit_depth=3, apply_sigmoid_per_depth='none')
        for depth_layer in nc.circuit_layers:
            assert depth_layer.apply_sigmoid is False

    def test_invalid_apply_sigmoid_per_depth_raises(self):
        with pytest.raises(ValueError, match="apply_sigmoid_per_depth"):
            LearnableNeuralCircuit(apply_sigmoid_per_depth='whenever')

    def test_neural_circuit_round_trip_with_new_params(self):
        """Full Keras serialization with the new fields."""
        nc = LearnableNeuralCircuit(
            circuit_depth=2,
            num_logic_ops_per_depth=2,
            num_arithmetic_ops_per_depth=2,
            use_residual=True,
            circuit_routing='output_only',
            apply_sigmoid_per_depth='first_only',
            load_balance_coefficient=0.05,
            channel_mix='dense',
        )
        x = keras.Input(shape=(16,))
        y = nc(x)
        m = keras.Model(x, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'm.keras')
            m.save(path)
            m2 = keras.models.load_model(path)
        sample = np.random.randn(2, 16).astype(np.float32)
        np.testing.assert_allclose(
            ops.convert_to_numpy(m(sample)),
            ops.convert_to_numpy(m2(sample)),
            atol=1e-5,
        )


class TestPlan3a2f1d23DiversityRegularizer:
    """M5: diversity regularizer adds an aux loss when coefficient > 0."""

    def test_diversity_active_adds_loss(self):
        l = CircuitDepthLayer(diversity_coefficient=0.1)
        x = ops.convert_to_tensor(np.random.randn(2, 4).astype(np.float32))
        _ = l(x)
        # combination_probs loss alone would NOT add when coef=0; expect at
        # least the diversity loss term.
        assert len(l.losses) >= 1

    def test_diversity_inactive_no_loss(self):
        l = CircuitDepthLayer(diversity_coefficient=0.0)
        x = ops.convert_to_tensor(np.random.randn(2, 4).astype(np.float32))
        _ = l(x)
        assert len(l.losses) == 0

    def test_diversity_round_trip(self):
        l = CircuitDepthLayer(diversity_coefficient=0.05)
        cfg = l.get_config()
        assert cfg['diversity_coefficient'] == 0.05
        l2 = CircuitDepthLayer.from_config(cfg)
        assert l2.diversity_coefficient == 0.05

    def test_diversity_negative_raises(self):
        with pytest.raises(ValueError, match="diversity_coefficient"):
            CircuitDepthLayer(diversity_coefficient=-0.1)


class TestPlan3a2f1d23ToSymbolicWalker:
    """M1: LearnableNeuralCircuit.to_symbolic walker."""

    def test_to_symbolic_returns_multiline(self):
        nc = LearnableNeuralCircuit(circuit_depth=2)
        # Force build.
        x = ops.convert_to_tensor(np.random.randn(2, 8).astype(np.float32))
        _ = nc(x)
        s = nc.to_symbolic(top_k=1)
        assert isinstance(s, str)
        # Two depths -> at least 2 "depth N:" lines.
        assert s.count("depth 0:") == 1
        assert s.count("depth 1:") == 1
        assert "combination:" in s

    def test_to_symbolic_raises_before_build(self):
        nc = LearnableNeuralCircuit(circuit_depth=2)
        with pytest.raises(RuntimeError, match="built"):
            nc.to_symbolic()

    def test_to_symbolic_deterministic(self):
        nc = LearnableNeuralCircuit(circuit_depth=2)
        x = ops.convert_to_tensor(np.random.randn(2, 8).astype(np.float32))
        _ = nc(x)
        outputs = {nc.to_symbolic() for _ in range(5)}
        assert len(outputs) == 1


class TestPlan3a2f1d23PerChannelCircuit:
    """C3: per-channel selection mode propagates through CircuitDepthLayer
    and LearnableNeuralCircuit."""

    def test_circuit_depth_per_channel_weight_shape(self):
        l = CircuitDepthLayer(
            num_logic_ops=2, num_arithmetic_ops=2,
            selection_mode='per_channel',
        )
        l.build((None, 8))
        # 4 total operators, 8 channels.
        assert l.combination_weights.shape == (8, 4)
        # Inner experts also per-channel.
        assert l.logic_operators[0].selection_mode == 'per_channel'
        assert l.arithmetic_operators[0].selection_mode == 'per_channel'

    def test_circuit_depth_per_channel_forward_preserves_shape(self):
        l = CircuitDepthLayer(
            num_logic_ops=2, num_arithmetic_ops=2,
            selection_mode='per_channel',
        )
        x = ops.convert_to_tensor(np.random.randn(2, 6).astype(np.float32))
        y = l(x)
        assert y.shape == (2, 6)
        assert bool(ops.all(ops.isfinite(y)))

    def test_neural_circuit_per_channel_round_trip(self):
        nc = LearnableNeuralCircuit(
            circuit_depth=2, selection_mode='per_channel'
        )
        x = ops.convert_to_tensor(np.random.randn(2, 8).astype(np.float32))
        y1 = ops.convert_to_numpy(nc(x))
        cfg = nc.get_config()
        assert cfg['selection_mode'] == 'per_channel'
        # Build a fresh layer and check forward shape works.
        nc2 = LearnableNeuralCircuit.from_config(cfg)
        y2 = nc2(x)
        assert y2.shape == y1.shape

    def test_neural_circuit_per_channel_keras_save_load(self):
        inp = keras.Input(shape=(8,))
        out = LearnableNeuralCircuit(
            circuit_depth=2, selection_mode='per_channel'
        )(inp)
        m = keras.Model(inp, out)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'm.keras')
            m.save(path)
            m2 = keras.models.load_model(path)
        sample = np.random.randn(2, 8).astype(np.float32)
        np.testing.assert_allclose(
            ops.convert_to_numpy(m(sample)),
            ops.convert_to_numpy(m2(sample)),
            atol=1e-5,
        )


class TestPlan3a2f1d23ForceClip:
    """C4: stacked logic-after-arithmetic must clip inputs to [0, 1] when
    apply_sigmoid=False on depths >= 1."""

    def test_force_clip_auto_enabled_for_risky_stack(self, caplog):
        import logging
        caplog.set_level(logging.WARNING)
        nc = LearnableNeuralCircuit(
            circuit_depth=3,
            num_arithmetic_ops_per_depth=2,
            apply_sigmoid_per_depth='first_only',
        )
        # Depths >= 1 have apply_sigmoid=False and must force_logic_input_clip.
        assert nc.circuit_layers[0].force_logic_input_clip is False
        assert nc.circuit_layers[1].force_logic_input_clip is True
        assert nc.circuit_layers[2].force_logic_input_clip is True
        # Inner LearnableLogicOperator instances inherit it.
        assert nc.circuit_layers[1].logic_operators[0].force_clip_when_no_sigmoid is True
        # Warning emitted.
        assert any("force_logic_input_clip" in r.getMessage() for r in caplog.records)

    def test_no_force_clip_when_all_sigmoid(self):
        nc = LearnableNeuralCircuit(
            circuit_depth=3,
            num_arithmetic_ops_per_depth=2,
            apply_sigmoid_per_depth='all',
        )
        for cl in nc.circuit_layers:
            assert cl.force_logic_input_clip is False

    def test_force_clip_keeps_inputs_in_unit_interval(self):
        """When force_clip is on, the logic op tolerates arbitrary inputs."""
        op = LearnableLogicOperator_for_test_import = None  # noqa
        from dl_techniques.layers.logic.logic_operators import LearnableLogicOperator
        op = LearnableLogicOperator(
            operation_types=['and', 'or'],
            apply_sigmoid=False,
            allow_unary_degenerate=True,
            force_clip_when_no_sigmoid=True,
        )
        x = ops.convert_to_tensor(
            np.array([[-5.0, -1.0, 0.0, 0.5, 1.0, 5.0]], dtype=np.float32)
        )
        y = op(x)
        y_np = ops.convert_to_numpy(y)
        assert np.all(y_np >= 0.0) and np.all(y_np <= 1.0)


class TestPlan3a2f1d23GateEntropyAlias:
    """H6: gate_entropy_coefficient is canonical; load_balance_coefficient is
    a deprecated alias that emits DeprecationWarning."""

    def test_canonical_name_accepted_on_depth(self):
        l = CircuitDepthLayer(gate_entropy_coefficient=0.1)
        assert l.gate_entropy_coefficient == 0.1
        assert l.load_balance_coefficient == 0.1  # legacy attribute alias

    def test_deprecated_name_warns_on_depth(self):
        with pytest.warns(DeprecationWarning, match="load_balance_coefficient"):
            l = CircuitDepthLayer(load_balance_coefficient=0.2)
        assert l.gate_entropy_coefficient == 0.2

    def test_canonical_name_accepted_on_circuit(self):
        c = LearnableNeuralCircuit(
            circuit_depth=2, gate_entropy_coefficient=0.15
        )
        assert c.gate_entropy_coefficient == 0.15

    def test_deprecated_name_warns_on_circuit(self):
        with pytest.warns(DeprecationWarning, match="load_balance_coefficient"):
            c = LearnableNeuralCircuit(
                circuit_depth=2, load_balance_coefficient=0.25
            )
        assert c.gate_entropy_coefficient == 0.25

    def test_round_trip_uses_canonical_name(self):
        l = CircuitDepthLayer(gate_entropy_coefficient=0.3)
        cfg = l.get_config()
        assert cfg["gate_entropy_coefficient"] == 0.3
        assert "load_balance_coefficient" not in cfg
        l2 = CircuitDepthLayer.from_config(cfg)
        assert l2.gate_entropy_coefficient == 0.3


# ---------------------------------------------------------------------
# plan_2026-05-13_e33114da regression tests
# ---------------------------------------------------------------------

class TestPlanE33114daNeuralCircuit:
    """Regression tests for plan_2026-05-13_e33114da."""

    def test_risky_stack_triggers_with_residual_no_arith_widened(self):
        """B3 (widened, defensive): force_clip propagates on depth>=1 when
        first_only + use_residual=True. Tested with the smallest legal
        configuration (num_arith_per_depth=1, the smallest allowed).
        """
        nc = LearnableNeuralCircuit(
            circuit_depth=2,
            num_logic_ops_per_depth=2,
            num_arithmetic_ops_per_depth=1,
            use_residual=True,
            apply_sigmoid_per_depth='first_only',
        )
        # Depth-1 inner logic op should have force_clip=True
        assert nc.circuit_layers[1].force_logic_input_clip is True
        assert nc.circuit_layers[1].logic_operators[0].force_clip_when_no_sigmoid is True

    def test_per_channel_load_balance_catches_peaky(self):
        """B4: per-channel mode now properly penalizes channel-wise peaky β."""
        import keras
        cd = CircuitDepthLayer(
            num_logic_ops=2, num_arithmetic_ops=2,
            gate_entropy_coefficient=0.5, selection_mode='per_channel',
        )
        inp = ops.convert_to_tensor(np.random.randn(2, 4).astype(np.float32))
        _ = cd(inp)
        # Set channel-wise peaky weights (eye scaled, softmax -> ~one-hot)
        peaky = np.eye(4, dtype=np.float32) * 10.0
        cd.combination_weights.assign(peaky)
        _ = cd(inp)
        # The most recent loss should be roughly N=4 (4 * mean(1.0)) for peaky.
        losses = [float(l) for l in cd.losses]
        assert any(l > 1.0 for l in losses), (
            f"Per-channel peaky distribution should produce loss > 1; got {losses}"
        )

    def test_diversity_coefficient_reachable_via_wrapper(self):
        """B5: diversity_coefficient now goes through LearnableNeuralCircuit."""
        nc = LearnableNeuralCircuit(
            circuit_depth=2, diversity_coefficient=0.5,
        )
        # The attribute should be set and forwarded
        assert nc.diversity_coefficient == 0.5
        # Children should receive it
        for cl in nc.circuit_layers:
            assert cl.diversity_coefficient == 0.5

    def test_diversity_coefficient_reachable_via_factory(self):
        """B5: factory must accept diversity_coefficient (was silently dropped)."""
        from dl_techniques.layers.logic.factory import create_logic_layer
        layer = create_logic_layer(
            'neural_circuit', diversity_coefficient=0.7,
        )
        assert layer.diversity_coefficient == 0.7

    def test_inner_logic_kwargs_forwarded(self):
        """G1: inner_logic_kwargs forwarded to inner LearnableLogicOperator."""
        nc = LearnableNeuralCircuit(
            circuit_depth=1,
            inner_logic_kwargs={'gumbel_softmax': True, 'yager_p': 3.5},
        )
        inner = nc.circuit_layers[0].logic_operators[0]
        assert inner.gumbel_softmax is True
        assert inner.yager_p == 3.5

    def test_inner_arithmetic_kwargs_forwarded(self):
        nc = LearnableNeuralCircuit(
            circuit_depth=1,
            inner_arithmetic_kwargs={
                'safe_divide_mode': 'smooth',
                'exponent_clip_mode': 'smooth',
            },
        )
        inner = nc.circuit_layers[0].arithmetic_operators[0]
        assert inner.safe_divide_mode == 'smooth'
        assert inner.exponent_clip_mode == 'smooth'

    def test_inner_kwargs_collision_warns_and_ignores(self):
        """G1: wrapper-owned keys in inner_*_kwargs are warned and ignored."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CircuitDepthLayer(
                num_logic_ops=2,
                num_arithmetic_ops=2,
                # operation_types is wrapper-owned via logic_op_types
                inner_logic_kwargs={'apply_sigmoid': False, 'gumbel_softmax': True},
            )
            assert any('wrapper-controlled' in str(item.message) for item in w)

    def test_to_symbolic_on_circuit_depth(self):
        """G2: CircuitDepthLayer.to_symbolic works standalone."""
        cd = CircuitDepthLayer(num_logic_ops=2, num_arithmetic_ops=2)
        x = ops.convert_to_tensor(np.random.randn(2, 4).astype(np.float32))
        _ = cd(x)
        s = cd.to_symbolic(top_k=1)
        assert 'logic_op_0' in s
        assert 'arithmetic_op_0' in s
        assert 'combination:' in s

    def test_factory_rejects_bad_enum(self):
        """G3: factory validates enums early with helpful message."""
        from dl_techniques.layers.logic.factory import create_logic_layer
        with pytest.raises(ValueError, match="selection_mode"):
            create_logic_layer('neural_circuit', selection_mode='garbage')
        with pytest.raises(ValueError, match="circuit_routing"):
            create_logic_layer('circuit_depth', circuit_routing='nope')

    def test_neural_circuit_full_round_trip_with_new_params(self):
        """B5 + G1 round-trip: new params survive save/load."""
        import tempfile, os, keras
        nc = LearnableNeuralCircuit(
            circuit_depth=2,
            diversity_coefficient=0.3,
            inner_logic_kwargs={'gumbel_softmax': False, 'yager_p': 2.5},
        )
        inputs = keras.Input(shape=(4, 8))
        outputs = nc(inputs)
        m = keras.Model(inputs, outputs)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'model.keras')
            m.save(path)
            m2 = keras.models.load_model(path)
            nc2 = m2.layers[-1]
            assert nc2.diversity_coefficient == 0.3
            assert nc2.inner_logic_kwargs.get('yager_p') == 2.5

    def test_diversity_loss_vectorized_no_python_pair_loop(self):
        """D4: diversity loss uses Gram matrix, not nested Python loop.
        Smoke test: it still produces a loss > 0 when experts converge."""
        nc = LearnableNeuralCircuit(
            circuit_depth=1,
            num_logic_ops_per_depth=3,
            num_arithmetic_ops_per_depth=3,
            diversity_coefficient=0.5,
        )
        x = ops.convert_to_tensor(np.random.randn(2, 4).astype(np.float32))
        _ = nc(x)
        # Diversity loss should be among the model losses
        assert len(nc.losses) > 0
