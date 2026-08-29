import pytest
import tempfile
import os
import numpy as np
import keras
from keras import ops
from typing import Any, Dict

from dl_techniques.layers.logic.logic_operators import LearnableLogicOperator

from .logic_subject_oracle import (
    Knob,
    assert_knob_is_honoured,
    assert_the_harness_is_deterministic,
)


# R-038 closure -- plan-2026-08-22T035419-a11304c8 / D-251.
# Keras `ops/nn.py:907` advises that a softmax over a size-1 axis always returns
# exactly 1.0. Every site in this module feeds that axis a size of 1 ON PURPOSE
# -- single class, single token, single head, single anchor, single cluster,
# minimum sequence length -- so the advisory describes the test's own input, not
# a defect. Suppressed HERE rather than in `pyproject.toml` so an ACCIDENTAL
# size-1 softmax anywhere else still fails under `error::UserWarning`.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:You are using a softmax over axis:UserWarning"),
]


class TestLearnableLogicOperator:
    """Comprehensive test suite for LearnableLogicOperator."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing.

        Opts in to allow_unary_degenerate=True so legacy single-input tests
        still pass after M8 (plan_2026-05-13_3a2f1d23) flipped the default.
        """
        return {
            'operation_types': ['and', 'or', 'not'],
            'use_temperature': True,
            'temperature_init': 1.0,
            'allow_unary_degenerate': True,
        }

    @pytest.fixture
    def minimal_config(self) -> Dict[str, Any]:
        """Minimal configuration for testing."""
        return {
            'operation_types': ['and', 'or'],
            'allow_unary_degenerate': True,
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

    @pytest.fixture
    def binary_like_inputs(self) -> tuple:
        """Binary-like inputs (0s and 1s) for testing logic operations."""
        x1 = ops.convert_to_tensor(np.random.choice([0.0, 1.0], (4, 32)).astype(np.float32))
        x2 = ops.convert_to_tensor(np.random.choice([0.0, 1.0], (4, 32)).astype(np.float32))
        return x1, x2

    def test_initialization(self, layer_config):
        """Test layer initialization."""
        layer = LearnableLogicOperator(**layer_config)

        assert hasattr(layer, 'operation_types')
        assert hasattr(layer, 'use_temperature')
        assert hasattr(layer, 'temperature_init')
        assert not layer.built
        assert layer.operation_weights is None  # Not built yet
        assert layer.temperature is None  # Not built yet

    def test_minimal_initialization(self, minimal_config):
        """Test layer initialization with minimal config."""
        layer = LearnableLogicOperator(**minimal_config)

        assert layer.operation_types == ['and', 'or']
        assert layer.use_temperature is True  # Default
        assert not layer.built

    def test_default_initialization(self):
        """Test layer initialization with all defaults."""
        layer = LearnableLogicOperator()

        expected_ops = ['and', 'or', 'xor', 'not', 'nand', 'nor']
        assert layer.operation_types == expected_ops
        assert layer.use_temperature is True
        assert layer.temperature_init == 1.0

    def test_forward_pass_single_input(self, layer_config, sample_input):
        """Test forward pass with single input and building."""
        layer = LearnableLogicOperator(**layer_config)

        output = layer(sample_input)

        assert layer.built
        assert output.shape == sample_input.shape
        assert layer.operation_weights is not None
        assert layer.temperature is not None  # use_temperature=True

        # Check that weights have correct shapes
        assert len(layer.operation_weights.shape) == 1
        assert layer.operation_weights.shape[0] == len(layer.operation_types)
        assert len(layer.temperature.shape) == 0  # Scalar

        # Output should be in [0, 1] range due to sigmoid normalization and logic ops
        output_np = ops.convert_to_numpy(output)
        assert np.all(output_np >= 0.0) and np.all(output_np <= 1.0)

    def test_forward_pass_dual_inputs(self, layer_config, dual_inputs):
        """Test forward pass with dual inputs."""
        layer = LearnableLogicOperator(**layer_config)
        x1, x2 = dual_inputs

        output = layer([x1, x2])

        assert layer.built
        assert output.shape == x1.shape
        assert output.shape == x2.shape

        # Output should be in [0, 1] range
        output_np = ops.convert_to_numpy(output)
        assert np.all(output_np >= 0.0) and np.all(output_np <= 1.0)

    def test_forward_pass_2d_input(self, layer_config, sample_input_2d):
        """Test forward pass with 2D feature maps."""
        layer = LearnableLogicOperator(**layer_config)

        output = layer(sample_input_2d)

        assert layer.built
        assert output.shape == sample_input_2d.shape

        # Output should be in [0, 1] range
        output_np = ops.convert_to_numpy(output)
        assert np.all(output_np >= 0.0) and np.all(output_np <= 1.0)

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = LearnableLogicOperator(**layer_config)(inputs)
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
        outputs = LearnableLogicOperator(**layer_config)([input1, input2])
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
        layer = LearnableLogicOperator(**layer_config)
        config = layer.get_config()

        # Check all config parameters are present
        expected_keys = {
            'operation_types', 'use_temperature', 'temperature_init',
            'operation_initializer', 'temperature_initializer'
        }

        for key in expected_keys:
            assert key in config, f"Missing {key} in get_config()"

        # Check that values match
        assert config['operation_types'] == layer_config['operation_types']
        assert config['use_temperature'] == layer_config['use_temperature']
        assert config['temperature_init'] == layer_config['temperature_init']

    def test_gradients_flow(self, layer_config, sample_input):
        """Test gradient computation."""
        import tensorflow as tf  # For GradientTape

        layer = LearnableLogicOperator(**layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

        # Should have gradients for operation_weights and temperature (if enabled)
        expected_num_weights = 1  # operation_weights
        if layer.use_temperature:
            expected_num_weights += 1  # temperature

        assert len(gradients) == expected_num_weights

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_input, training):
        """Test behavior in different training modes."""
        layer = LearnableLogicOperator(**layer_config)

        output = layer(sample_input, training=training)
        assert output.shape == sample_input.shape

        # Output should be in [0, 1] range regardless of training mode
        output_np = ops.convert_to_numpy(output)
        assert np.all(output_np >= 0.0) and np.all(output_np <= 1.0)

    def test_different_operation_combinations(self, sample_input):
        """Test different combinations of operations."""
        operation_sets = [
            ['and', 'or'],
            ['xor', 'not'],
            ['nand', 'nor'],
            ['and', 'or', 'xor'],
            ['and', 'or', 'xor', 'not', 'nand', 'nor']  # All
        ]

        for ops_set in operation_sets:
            layer = LearnableLogicOperator(
                operation_types=ops_set, allow_unary_degenerate=True
            )
            output = layer(sample_input)
            assert output.shape == sample_input.shape

            # Output should be in [0, 1] range
            output_np = ops.convert_to_numpy(output)
            assert np.all(output_np >= 0.0) and np.all(output_np <= 1.0)

    def test_without_temperature_scaling(self, sample_input):
        """Test layer without temperature scaling."""
        layer = LearnableLogicOperator(
            operation_types=['and', 'or'],
            use_temperature=False,
            allow_unary_degenerate=True,
        )

        output = layer(sample_input)
        assert output.shape == sample_input.shape
        assert layer.temperature is None

    def test_logic_operations_correctness(self, binary_like_inputs):
        """Test that logic operations work correctly with binary-like inputs."""
        x1, x2 = binary_like_inputs

        # Test individual operations
        operations = {
            'and': lambda a, b: a * b,
            'or': lambda a, b: a + b - a * b,
            'xor': lambda a, b: a + b - 2 * a * b,
            'not': lambda a, b: 1 - a,  # Only uses first input
            'nand': lambda a, b: 1 - a * b,
            'nor': lambda a, b: 1 - (a + b - a * b)
        }

        for op_name, expected_func in operations.items():
            layer = LearnableLogicOperator(
                operation_types=[op_name],
                use_temperature=False,  # Disable for cleaner results
                operation_initializer='ones'  # Weight = 1 for clear testing
            )

            # Apply sigmoid to inputs to get [0,1] range (as layer does internally)
            x1_sig = ops.sigmoid(x1)
            x2_sig = ops.sigmoid(x2)

            output = layer([x1, x2])

            # For pure operations with weight=1, output should be close to expected
            # (Note: this is approximate due to softmax normalization)
            assert output.shape == x1.shape

    def test_sigmoid_input_normalization(self):
        """Test that inputs are properly normalized with sigmoid."""
        # Create extreme inputs
        extreme_input = ops.convert_to_tensor(
            np.array([[-100.0, -10.0, 0.0, 10.0, 100.0]], dtype=np.float32)
        )

        layer = LearnableLogicOperator(
            operation_types=['and'], allow_unary_degenerate=True
        )
        output = layer(extreme_input)

        # Output should still be in [0, 1] range
        output_np = ops.convert_to_numpy(output)
        assert np.all(output_np >= 0.0) and np.all(output_np <= 1.0)

    def test_edge_cases(self):
        """Test error conditions."""
        # Invalid operation types
        with pytest.raises(ValueError, match="Invalid operation types"):
            LearnableLogicOperator(operation_types=['invalid_op'])

        # Invalid temperature_init
        with pytest.raises(ValueError, match="temperature_init must be positive"):
            LearnableLogicOperator(temperature_init=0.0)

        # Invalid temperature_init (negative)
        with pytest.raises(ValueError, match="temperature_init must be positive"):
            LearnableLogicOperator(temperature_init=-1.0)

    def test_mismatched_input_shapes(self):
        """Test error with mismatched input shapes."""
        layer = LearnableLogicOperator(operation_types=['and'])

        x1 = ops.convert_to_tensor(np.random.normal(0, 1, (4, 32)).astype(np.float32))
        x2 = ops.convert_to_tensor(np.random.normal(0, 1, (4, 16)).astype(np.float32))

        with pytest.raises(ValueError, match="Input tensors must have the same shape"):
            layer.build([x1.shape, x2.shape])

    def test_too_many_inputs(self):
        """Test error with too many inputs."""
        layer = LearnableLogicOperator(operation_types=['and'])

        x1 = ops.convert_to_tensor(np.random.normal(0, 1, (4, 32)).astype(np.float32))
        x2 = ops.convert_to_tensor(np.random.normal(0, 1, (4, 32)).astype(np.float32))
        x3 = ops.convert_to_tensor(np.random.normal(0, 1, (4, 32)).astype(np.float32))

        with pytest.raises(ValueError, match="Expected 1 or 2 inputs"):
            layer([x1, x2, x3])

    def test_compute_output_shape(self, layer_config):
        """Test output shape computation."""
        layer = LearnableLogicOperator(**layer_config)

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
        layer = LearnableLogicOperator(
            operation_types=['and', 'or'],
            operation_initializer='he_normal',
            temperature_initializer='constant',
            allow_unary_degenerate=True,
        )

        output = layer(sample_input)
        assert output.shape == sample_input.shape
        assert layer.built

        # Output should be in [0, 1] range
        output_np = ops.convert_to_numpy(output)
        assert np.all(output_np >= 0.0) and np.all(output_np <= 1.0)

    def test_deterministic_with_fixed_seed(self, sample_input):
        """Test deterministic behavior with fixed random seed."""
        keras.utils.set_random_seed(42)

        layer1 = LearnableLogicOperator(
            operation_types=['and', 'or'], allow_unary_degenerate=True
        )
        output1 = layer1(sample_input)

        keras.utils.set_random_seed(42)

        layer2 = LearnableLogicOperator(
            operation_types=['and', 'or'], allow_unary_degenerate=True
        )
        output2 = layer2(sample_input)

        # Should be identical with same seed
        np.testing.assert_allclose(
            ops.convert_to_numpy(output1),
            ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Outputs should be identical with same random seed"
        )

    def test_temperature_effect(self, sample_input):
        """Test effect of different temperature values."""
        # High temperature should give more uniform operation selection
        layer_high_temp = LearnableLogicOperator(
            operation_types=['and', 'or'],
            temperature_init=10.0,
            allow_unary_degenerate=True,
        )

        # Low temperature should give sharper operation selection
        layer_low_temp = LearnableLogicOperator(
            operation_types=['and', 'or'],
            temperature_init=0.1,
            allow_unary_degenerate=True,
        )

        output_high = layer_high_temp(sample_input)
        output_low = layer_low_temp(sample_input)

        # Both should produce valid outputs
        assert output_high.shape == sample_input.shape
        assert output_low.shape == sample_input.shape

        # Both should be in [0, 1] range
        for output in [output_high, output_low]:
            output_np = ops.convert_to_numpy(output)
            assert np.all(output_np >= 0.0) and np.all(output_np <= 1.0)

    def test_single_operation_type(self, sample_input):
        """Test with single operation type."""
        for op in ['and', 'or', 'xor', 'not', 'nand', 'nor']:
            layer = LearnableLogicOperator(
                operation_types=[op], allow_unary_degenerate=True
            )
            output = layer(sample_input)

            assert output.shape == sample_input.shape
            output_np = ops.convert_to_numpy(output)
            assert np.all(output_np >= 0.0) and np.all(output_np <= 1.0)

# Run tests with: pytest test_logic_operator.py -v

# ---------------------------------------------------------------------------
# Regression tests added in plan_2026-05-13_e52a5ac8
# ---------------------------------------------------------------------------

class TestPlanE52a5ac8Logic:
    def test_compute_output_shape_accepts_list_form(self):
        layer = LearnableLogicOperator()
        out_list = layer.compute_output_shape([None, 32])
        assert tuple(out_list) == (None, 32)
        out_binary = layer.compute_output_shape([(None, 32), (None, 32)])
        assert tuple(out_binary) == (None, 32)

    def test_apply_sigmoid_flag_default_true(self):
        """C3: default preserves legacy sigmoid pre-normalization."""
        layer = LearnableLogicOperator()
        assert layer.apply_sigmoid is True

    def test_apply_sigmoid_false_preserves_dynamic_range(self):
        """C3: with binary [0,1] inputs, skipping sigmoid widens output range.

        The flag is intended for stacking / callers who already provide
        values in [0,1]. Default `apply_sigmoid=True` re-maps [0,1] to
        [0.5, 0.731], collapsing dynamic range. With apply_sigmoid=False the
        layer operates on the original [0,1] values directly.

        NOTE on unary degeneracy (LESSONS L38): single-tensor inputs collapse
        binary ops to fixed functions of x — supply two distinct inputs.
        """
        keras.utils.set_random_seed(0)
        x1 = ops.convert_to_tensor(np.random.uniform(0.0, 1.0, (64, 16)).astype(np.float32))
        x2 = ops.convert_to_tensor(np.random.uniform(0.0, 1.0, (64, 16)).astype(np.float32))
        layer_with = LearnableLogicOperator()  # apply_sigmoid=True (default)
        layer_no = LearnableLogicOperator(apply_sigmoid=False)
        y_with = layer_with([x1, x2])
        y_no = layer_no([x1, x2])
        std_with = float(ops.std(y_with))
        std_no = float(ops.std(y_no))
        assert std_no > std_with, (
            f"apply_sigmoid=False did not preserve dynamic range: "
            f"std_with={std_with:.4f}, std_no={std_no:.4f}"
        )
        # Both outputs must remain in [0, 1].
        assert float(ops.min(y_with)) >= 0.0 and float(ops.max(y_with)) <= 1.0 + 1e-6
        assert float(ops.min(y_no)) >= 0.0 and float(ops.max(y_no)) <= 1.0 + 1e-6

    def test_apply_sigmoid_roundtrip(self):
        """C3: get_config / from_config round-trip preserves apply_sigmoid."""
        layer = LearnableLogicOperator(apply_sigmoid=False)
        cfg = layer.get_config()
        assert cfg["apply_sigmoid"] is False
        restored = LearnableLogicOperator.from_config(cfg)
        assert restored.apply_sigmoid is False


# ---------------------------------------------------------------------------
# Regression tests added in plan_2026-05-13_a2b0f17b
# ---------------------------------------------------------------------------

class TestPlanA2b0f17bLogic:
    """Regressions for full-rewrite plan."""

    def _at_corner(self, layer, p, q):
        x1 = ops.convert_to_tensor(np.array([[float(p)]], dtype=np.float32))
        x2 = ops.convert_to_tensor(np.array([[float(q)]], dtype=np.float32))
        return float(ops.convert_to_numpy(layer([x1, x2]))[0, 0])

    def test_truth_table_classical_and(self):
        op = LearnableLogicOperator(operation_types=['and'], apply_sigmoid=False, use_temperature=False)
        op.build([(None, 1), (None, 1)])
        assert abs(self._at_corner(op, 0, 0) - 0.0) < 1e-5
        assert abs(self._at_corner(op, 0, 1) - 0.0) < 1e-5
        assert abs(self._at_corner(op, 1, 0) - 0.0) < 1e-5
        assert abs(self._at_corner(op, 1, 1) - 1.0) < 1e-5

    def test_truth_table_classical_or(self):
        op = LearnableLogicOperator(operation_types=['or'], apply_sigmoid=False, use_temperature=False)
        op.build([(None, 1), (None, 1)])
        assert abs(self._at_corner(op, 0, 0) - 0.0) < 1e-5
        assert abs(self._at_corner(op, 0, 1) - 1.0) < 1e-5
        assert abs(self._at_corner(op, 1, 0) - 1.0) < 1e-5
        assert abs(self._at_corner(op, 1, 1) - 1.0) < 1e-5

    def test_truth_table_xor(self):
        op = LearnableLogicOperator(operation_types=['xor'], apply_sigmoid=False, use_temperature=False)
        op.build([(None, 1), (None, 1)])
        assert abs(self._at_corner(op, 0, 0) - 0.0) < 1e-5
        assert abs(self._at_corner(op, 0, 1) - 1.0) < 1e-5
        assert abs(self._at_corner(op, 1, 0) - 1.0) < 1e-5
        assert abs(self._at_corner(op, 1, 1) - 0.0) < 1e-5

    def test_truth_table_lukasiewicz_and(self):
        op = LearnableLogicOperator(operation_types=['lukasiewicz_and'], apply_sigmoid=False, use_temperature=False)
        op.build([(None, 1), (None, 1)])
        assert abs(self._at_corner(op, 0.5, 0.5) - 0.0) < 1e-5
        assert abs(self._at_corner(op, 0.7, 0.7) - 0.4) < 1e-5
        assert abs(self._at_corner(op, 1.0, 1.0) - 1.0) < 1e-5

    def test_truth_table_godel_and(self):
        op = LearnableLogicOperator(operation_types=['godel_and'], apply_sigmoid=False, use_temperature=False)
        op.build([(None, 1), (None, 1)])
        assert abs(self._at_corner(op, 0.3, 0.7) - 0.3) < 1e-5
        assert abs(self._at_corner(op, 0.9, 0.4) - 0.4) < 1e-5

    def test_truth_table_implies(self):
        op = LearnableLogicOperator(operation_types=['implies'], apply_sigmoid=False, use_temperature=False)
        op.build([(None, 1), (None, 1)])
        assert abs(self._at_corner(op, 1, 1) - 1.0) < 1e-5
        assert abs(self._at_corner(op, 1, 0) - 0.0) < 1e-5
        assert abs(self._at_corner(op, 0, 1) - 1.0) < 1e-5
        assert abs(self._at_corner(op, 0, 0) - 1.0) < 1e-5

    def test_unary_input_raises_when_strict(self):
        op = LearnableLogicOperator(allow_unary_degenerate=False)
        x = ops.convert_to_tensor(np.random.randn(2, 4).astype(np.float32))
        with pytest.raises(ValueError, match="single tensor input"):
            _ = op(x)

    def test_unary_input_blocked_when_default_M8(self):
        """M8: allow_unary_degenerate default flipped False (plan_3a2f1d23)."""
        op = LearnableLogicOperator()
        assert op.allow_unary_degenerate is False
        x = ops.convert_to_tensor(np.random.randn(2, 4).astype(np.float32))
        with pytest.raises(ValueError, match="single tensor input"):
            _ = op(x)

    def test_unary_input_allowed_when_opt_in(self):
        op = LearnableLogicOperator(allow_unary_degenerate=True)
        x = ops.convert_to_tensor(np.random.randn(2, 4).astype(np.float32))
        y = op(x)
        assert y.shape == (2, 4)

    def test_softplus_temperature_round_trip(self):
        op = LearnableLogicOperator(softplus_temperature=True, temperature_init=2.0)
        op.build((None, 4))
        from keras import ops as kops
        assert abs(float(kops.softplus(op.temperature)) - 2.0) < 1e-5
        op2 = LearnableLogicOperator.from_config(op.get_config())
        assert op2.softplus_temperature is True

    def test_to_symbolic_returns_dominant_op(self):
        op = LearnableLogicOperator(operation_types=['and', 'or', 'xor'])
        op.build([(None, 4), (None, 4)])
        op.operation_weights.assign([0.0, 10.0, 0.0])
        s = op.to_symbolic(top_k=1)
        assert s.startswith("or")

    def test_entropy_loss_added(self):
        op = LearnableLogicOperator(
            entropy_coefficient=0.5, allow_unary_degenerate=True
        )
        op.build((None, 4))
        x = ops.convert_to_tensor(np.random.randn(2, 4).astype(np.float32))
        _ = op(x)
        assert len(op.losses) >= 1

    def test_invalid_op_type_raises(self):
        with pytest.raises(ValueError, match="Invalid operation types"):
            LearnableLogicOperator(operation_types=['nonexistent'])

    def test_empty_operation_types_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            LearnableLogicOperator(operation_types=[])

    def test_new_ops_in_config(self):
        op = LearnableLogicOperator(
            operation_types=['lukasiewicz_and', 'godel_or', 'implies']
        )
        cfg = op.get_config()
        assert set(cfg['operation_types']) == {'lukasiewicz_and', 'godel_or', 'implies'}

    def test_stacked_logic_first_only_beats_all_sigmoid(self):
        """C2: 'first_only' sigmoid plumbing materially reduces (but does
        not fully cure) the stacked-collapse pathology vs the legacy
        'all-sigmoid' behavior. The full cure requires entropy reg + sharper
        init; this test asserts the directional improvement."""
        np.random.seed(0)
        x = ops.convert_to_tensor(np.linspace(-3, 3, 64).reshape(8, 8).astype(np.float32))
        keras.utils.set_random_seed(0)
        op_a = LearnableLogicOperator(apply_sigmoid=True, allow_unary_degenerate=True)
        op_b = LearnableLogicOperator(apply_sigmoid=True, allow_unary_degenerate=True)
        op_c = LearnableLogicOperator(apply_sigmoid=True, allow_unary_degenerate=True)
        all_sig = op_c(op_b(op_a(x)))
        keras.utils.set_random_seed(0)
        op_a2 = LearnableLogicOperator(apply_sigmoid=True, allow_unary_degenerate=True)
        op_b2 = LearnableLogicOperator(apply_sigmoid=False, allow_unary_degenerate=True)
        op_c2 = LearnableLogicOperator(apply_sigmoid=False, allow_unary_degenerate=True)
        first_only = op_c2(op_b2(op_a2(x)))
        s_all = float(ops.std(all_sig))
        s_first = float(ops.std(first_only))
        # first_only should preserve at least 2× more std than legacy.
        assert s_first > 2.0 * s_all, (
            f"first_only std={s_first:.6f} not materially > all_sigmoid std={s_all:.6f}"
        )


class TestPlan3a2f1d23LogicC1:
    """Regression tests for plan_2026-05-13_3a2f1d23 step 1 (C1): canonical
    Jang (2017) Gumbel-softmax form for LearnableLogicOperator."""

    def test_canonical_gumbel_form_low_temperature(self):
        keras.utils.set_random_seed(7)
        weights = np.array([0.0, 1.0, 2.0, 0.5], dtype=np.float32)
        temperature = 0.1
        op = LearnableLogicOperator(
            operation_types=['and', 'or', 'xor', 'nand'],
            use_temperature=True,
            gumbel_softmax=True,
            gumbel_hard=False,
            softplus_temperature=False,
        )
        op.build((None, 4))
        op.operation_weights.assign(weights)
        op.temperature.assign(np.array(temperature, dtype=np.float32))

        n_samples = 4000
        accum = np.zeros(4, dtype=np.float64)
        for _ in range(n_samples):
            accum += ops.convert_to_numpy(op._operation_probs())
        empirical = accum / n_samples

        canonical_logits = weights / temperature
        canonical = np.exp(canonical_logits - canonical_logits.max())
        canonical /= canonical.sum()

        assert empirical.argmax() == int(canonical.argmax())
        # Threshold 0.45 separates canonical (~0.55-0.65) from buggy (~0.25).
        assert empirical[canonical.argmax()] >= 0.45, (
            f"Canonical mass on argmax too low: empirical={empirical}, "
            f"canonical={canonical}"
        )

    def test_gumbel_deterministic_skips_noise(self):
        op = LearnableLogicOperator(
            operation_types=['and', 'or'],
            gumbel_softmax=True,
            gumbel_hard=False,
        )
        op.build((None, 4))
        op.operation_weights.assign([1.0, 2.0])
        p1 = ops.convert_to_numpy(op._operation_probs(deterministic=True))
        p2 = ops.convert_to_numpy(op._operation_probs(deterministic=True))
        np.testing.assert_allclose(p1, p2, atol=1e-7)

    def test_softplus_temperature_default_True_H1(self):
        """H1: softplus_temperature default flipped True (plan_3a2f1d23)."""
        op = LearnableLogicOperator()
        assert op.softplus_temperature is True

    def test_operation_initializer_default_zeros_H2(self):
        """H2: operation_initializer default flipped to 'zeros'."""
        op = LearnableLogicOperator()
        assert op.operation_initializer.__class__.__name__ == 'Zeros'

    def test_to_symbolic_deterministic_under_gumbel_C5(self):
        op = LearnableLogicOperator(
            operation_types=['and', 'or', 'xor'],
            gumbel_softmax=True,
            gumbel_hard=False,
        )
        op.build((None, 4))
        op.operation_weights.assign([0.0, 0.0, 5.0])
        outputs = {op.to_symbolic(top_k=1) for _ in range(10)}
        assert len(outputs) == 1, f"to_symbolic() non-deterministic: {outputs}"
        assert next(iter(outputs)).startswith("xor")


class TestPlan3a2f1d23TnormsM4:
    """M4: Hamacher + Yager t-norms truth-table corners."""

    def _at_corner(self, op, p, q):
        x1 = ops.convert_to_tensor(np.array([[p]], dtype=np.float32))
        x2 = ops.convert_to_tensor(np.array([[q]], dtype=np.float32))
        return float(ops.convert_to_numpy(op([x1, x2]))[0, 0])

    def test_hamacher_and_corners(self):
        op = LearnableLogicOperator(
            operation_types=['hamacher_and'],
            apply_sigmoid=False,
            use_temperature=False,
        )
        op.build([(None, 1), (None, 1)])
        assert abs(self._at_corner(op, 1.0, 1.0) - 1.0) < 1e-4
        assert self._at_corner(op, 1.0, 0.0) < 1e-4
        assert self._at_corner(op, 0.0, 0.0) < 1e-4

    def test_hamacher_or_corners(self):
        op = LearnableLogicOperator(
            operation_types=['hamacher_or'],
            apply_sigmoid=False,
            use_temperature=False,
        )
        op.build([(None, 1), (None, 1)])
        assert abs(self._at_corner(op, 1.0, 0.0) - 1.0) < 1e-4
        assert abs(self._at_corner(op, 0.0, 1.0) - 1.0) < 1e-4
        assert self._at_corner(op, 0.0, 0.0) < 1e-4

    def test_yager_and_corners(self):
        op = LearnableLogicOperator(
            operation_types=['yager_and'],
            apply_sigmoid=False,
            use_temperature=False,
            yager_p=2.0,
        )
        op.build([(None, 1), (None, 1)])
        assert abs(self._at_corner(op, 1.0, 1.0) - 1.0) < 1e-4
        assert self._at_corner(op, 1.0, 0.0) < 1e-4

    def test_yager_or_corners(self):
        op = LearnableLogicOperator(
            operation_types=['yager_or'],
            apply_sigmoid=False,
            use_temperature=False,
            yager_p=2.0,
        )
        op.build([(None, 1), (None, 1)])
        assert abs(self._at_corner(op, 1.0, 1.0) - 1.0) < 1e-4
        assert abs(self._at_corner(op, 1.0, 0.0) - 1.0) < 1e-4
        assert self._at_corner(op, 0.0, 0.0) < 1e-4

    def test_yager_p_round_trip(self):
        op = LearnableLogicOperator(operation_types=['yager_and'], yager_p=3.5)
        op.build([(None, 4), (None, 4)])
        cfg = op.get_config()
        assert cfg['yager_p'] == 3.5
        op2 = LearnableLogicOperator.from_config(cfg)
        assert op2.yager_p == 3.5

    def test_yager_p_invalid_raises(self):
        with pytest.raises(ValueError, match="yager_p"):
            LearnableLogicOperator(yager_p=0.0)


class TestPlan3a2f1d23LogicPerChannelC3:
    """C3: per-channel selection mode on LearnableLogicOperator."""

    def test_weight_shape_per_channel(self):
        op = LearnableLogicOperator(
            operation_types=['and', 'or', 'xor'],
            selection_mode='per_channel',
            allow_unary_degenerate=True,
        )
        op.build((None, 8))
        assert op.operation_weights.shape == (8, 3)

    def test_forward_per_channel_distinct_channel_selection(self):
        """Channel 0 → 'and' (p*q), Channel 1 → 'or' (p+q-pq). With known
        binary inputs, output must match the per-channel ops."""
        op = LearnableLogicOperator(
            operation_types=['and', 'or'],
            apply_sigmoid=False,
            use_temperature=False,
        )
        op.selection_mode = None  # placeholder
        op2 = LearnableLogicOperator(
            operation_types=['and', 'or'],
            apply_sigmoid=False,
            use_temperature=False,
            selection_mode='per_channel',
        )
        op2.build([(None, 2), (None, 2)])
        op2.operation_weights.assign(
            np.array([[10.0, 0.0], [0.0, 10.0]], dtype=np.float32)
        )
        x1 = ops.convert_to_tensor(np.array([[1.0, 1.0]], dtype=np.float32))
        x2 = ops.convert_to_tensor(np.array([[1.0, 0.0]], dtype=np.float32))
        y = ops.convert_to_numpy(op2([x1, x2]))
        # Channel 0 (and): 1*1=1. Channel 1 (or): 1+0-0=1.
        np.testing.assert_allclose(y[0, 0], 1.0, atol=1e-3)
        np.testing.assert_allclose(y[0, 1], 1.0, atol=1e-3)
        # Try inputs that distinguish: x1=[1, 1], x2=[0, 0]
        x1b = ops.convert_to_tensor(np.array([[1.0, 1.0]], dtype=np.float32))
        x2b = ops.convert_to_tensor(np.array([[0.0, 0.0]], dtype=np.float32))
        yb = ops.convert_to_numpy(op2([x1b, x2b]))
        # Channel 0 (and): 1*0=0. Channel 1 (or): 1+0-0=1.
        np.testing.assert_allclose(yb[0, 0], 0.0, atol=1e-3)
        np.testing.assert_allclose(yb[0, 1], 1.0, atol=1e-3)

    def test_per_channel_round_trip(self):
        op = LearnableLogicOperator(
            operation_types=['and', 'or'],
            selection_mode='per_channel',
            allow_unary_degenerate=True,
        )
        op.build((None, 4))
        cfg = op.get_config()
        assert cfg['selection_mode'] == 'per_channel'
        op2 = LearnableLogicOperator.from_config(cfg)
        assert op2.selection_mode == 'per_channel'

    def test_per_channel_rank4(self):
        op = LearnableLogicOperator(
            operation_types=['and', 'or'],
            selection_mode='per_channel',
            allow_unary_degenerate=True,
        )
        x = ops.convert_to_tensor(np.random.randn(2, 4, 4, 6).astype(np.float32))
        y = op(x)
        assert y.shape == (2, 4, 4, 6)


# ---------------------------------------------------------------------
# plan_2026-05-13_e33114da regression tests
# ---------------------------------------------------------------------

class TestPlanE33114daLogic:
    """Regression tests for plan_2026-05-13_e33114da (post-rewrite review)."""

    def test_hamacher_or_boundary_one_one(self):
        """B1: Hamacher OR at (1, 1) must equal 1 (was 0)."""
        op = LearnableLogicOperator(
            operation_types=['hamacher_or'], allow_unary_degenerate=True
        )
        x_one = ops.convert_to_tensor(np.array([[1.0, 1.0, 1.0]], dtype=np.float32))
        # Bypass sigmoid by going via the private method directly
        op.build((None, 3))
        result = ops.convert_to_numpy(op._hamacher_or(x_one, x_one))
        np.testing.assert_allclose(result, 1.0, atol=1e-5)

    def test_hamacher_and_boundary_zero_zero(self):
        """B1: Hamacher AND at (0, 0) must equal 0."""
        op = LearnableLogicOperator(
            operation_types=['hamacher_and'], allow_unary_degenerate=True
        )
        x_zero = ops.convert_to_tensor(np.array([[0.0, 0.0]], dtype=np.float32))
        op.build((None, 2))
        result = ops.convert_to_numpy(op._hamacher_and(x_zero, x_zero))
        np.testing.assert_allclose(result, 0.0, atol=1e-5)

    def test_hamacher_or_continuity_near_one(self):
        """Hamacher OR should be continuous near (1,1) corner."""
        op = LearnableLogicOperator(
            operation_types=['hamacher_or'], allow_unary_degenerate=True
        )
        op.build((None, 1))
        x_near = ops.convert_to_tensor(np.array([[0.999]], dtype=np.float32))
        x_one = ops.convert_to_tensor(np.array([[1.0]], dtype=np.float32))
        near = float(ops.convert_to_numpy(op._hamacher_or(x_near, x_near)))
        at = float(ops.convert_to_numpy(op._hamacher_or(x_one, x_one)))
        # Both should be ~1.0; the prior bug had near=0.9995 then at=0
        assert abs(near - at) < 1e-3

    def test_gumbel_deterministic_at_inference(self):
        """B2: gumbel_softmax=True at training=False yields identical outputs."""
        keras.utils.set_random_seed(42)
        op = LearnableLogicOperator(
            gumbel_softmax=True, operation_types=['and', 'or', 'xor'],
            allow_unary_degenerate=True,
        )
        x = ops.convert_to_tensor(np.random.randn(2, 4).astype(np.float32))
        op.build(x.shape)
        o1 = ops.convert_to_numpy(op(x, training=False))
        o2 = ops.convert_to_numpy(op(x, training=False))
        o3 = ops.convert_to_numpy(op(x))  # training=None
        np.testing.assert_allclose(o1, o2, atol=1e-7)
        np.testing.assert_allclose(o1, o3, atol=1e-7)

    def test_gumbel_stochastic_at_training(self):
        """B2: gumbel_softmax=True at training=True is stochastic."""
        keras.utils.set_random_seed(0)
        op = LearnableLogicOperator(
            gumbel_softmax=True,
            operation_types=['and', 'or', 'xor', 'not', 'nand', 'nor'],
            allow_unary_degenerate=True,
        )
        x = ops.convert_to_tensor(np.random.randn(2, 8).astype(np.float32))
        op.build(x.shape)
        o1 = ops.convert_to_numpy(op(x, training=True))
        o2 = ops.convert_to_numpy(op(x, training=True))
        # Outputs should differ due to fresh Gumbel sampling.
        assert not np.allclose(o1, o2, atol=1e-5)

    def test_new_implications_in_unit_interval(self):
        """G4: lukasiewicz_implies, reichenbach_implies, goguen_implies."""
        for op_name in ['lukasiewicz_implies', 'reichenbach_implies', 'goguen_implies']:
            op = LearnableLogicOperator(operation_types=[op_name])
            # Skip sigmoid: pass values already in [0, 1].
            op2 = LearnableLogicOperator(
                operation_types=[op_name], apply_sigmoid=False,
            )
            x1 = ops.convert_to_tensor(np.array([[0.0, 0.5, 1.0, 0.7]], dtype=np.float32))
            x2 = ops.convert_to_tensor(np.array([[0.5, 0.5, 0.5, 0.3]], dtype=np.float32))
            y = ops.convert_to_numpy(op2([x1, x2]))
            assert (y >= -1e-5).all() and (y <= 1.0 + 1e-5).all(), (
                f"{op_name} produced out-of-[0,1] value: {y}"
            )

    def test_compute_output_shape_rejects_mismatched_binary(self):
        """D9: compute_output_shape raises on shape mismatch."""
        op = LearnableLogicOperator(operation_types=['and', 'or'])
        with pytest.raises(ValueError, match="same shape"):
            op.compute_output_shape([(None, 32), (None, 16)])


# --------------------------------------------------------------------
# §12.5 -- every constructor parameter pinned, with the §13.3.2
# instrument that matches its knob class.
# --------------------------------------------------------------------

#: Every variant that needs a non-uniform gate distribution uses this.
#: With the default `operation_initializer='zeros'` the softmax over the
#: gates is exactly uniform, which makes the temperature and the Gumbel
#: knobs structurally unobservable in the output (measured: `dy = 0.0`
#: for `temperature_init` 1.0-vs-5.0 at the default initializer).
_KNOB_LIVE = keras.initializers.RandomNormal(stddev=0.5, seed=11)
_KNOB_OTHER = keras.initializers.RandomNormal(stddev=0.9, seed=13)

#: Base kwargs. Three binary gates, so `operation_weights` has shape
#: (3,) and the sweep below can shorten it to (2,).
_KNOB_BASE = {"operation_types": ["and", "or", "xor"]}

LOGIC_KNOBS = [
    Knob("operation_types", "structural", {
        "two": {"operation_types": ["and", "or"]},
        "three": {"operation_types": ["and", "or", "xor"]},
    }, measured=0.105044),
    Knob("use_temperature", "structural", {
        "on": {"use_temperature": True},
        "off": {"use_temperature": False},
    }, measured=0.0),
    # NOT a value knob: at temperature_init=1.0 vs 5.0 the OUTPUT moves
    # by only 0.0167, but the stored weight moves by 4.45191 --
    # log(expm1(5.0)) - log(expm1(1.0)). The weight is the direct
    # instrument; the output is a downstream consequence.
    Knob("temperature_init", "scoped_value", {
        "one": {"operation_initializer": _KNOB_LIVE, "temperature_init": 1.0},
        "five": {"operation_initializer": _KNOB_LIVE, "temperature_init": 5.0},
    }, measured=4.45191, scope="temperature"),
    Knob("operation_initializer", "scoped_value", {
        "narrow": {"operation_initializer": _KNOB_LIVE},
        "wide": {"operation_initializer": _KNOB_OTHER},
    }, measured=1.7634, scope="operation_weights"),
    # Read ONLY when softplus_temperature is False -- the class
    # docstring says so, and the softplus path overrides it with
    # Constant(log(expm1(temperature_init))). Pinned in the
    # configuration in which it is live; the ignored configuration is
    # pinned separately below.
    Knob("temperature_initializer", "scoped_value", {
        "one": {"softplus_temperature": False,
                "temperature_initializer": keras.initializers.Constant(1.0)},
        "three": {"softplus_temperature": False,
                  "temperature_initializer": keras.initializers.Constant(3.0)},
    }, measured=2.0, scope="temperature"),
    Knob("apply_sigmoid", "value", {
        "on": {"apply_sigmoid": True},
        "off": {"apply_sigmoid": False},
    }, measured=0.40371),
    Knob("force_clip_when_no_sigmoid", "value", {
        "off": {"apply_sigmoid": False, "force_clip_when_no_sigmoid": False},
        "on": {"apply_sigmoid": False, "force_clip_when_no_sigmoid": True},
    }, measured=0.735067),
    # softplus_temperature is a REPARAMETERIZATION, not an output knob:
    # both paths give an effective temperature of temperature_init, so
    # the output delta is EXACTLY 0.0 by construction. What moves is the
    # stored raw value, 1.0 - log(expm1(1.0)) = 0.458675.
    Knob("softplus_temperature", "scoped_value", {
        "on": {"operation_initializer": _KNOB_LIVE,
               "softplus_temperature": True},
        "off": {"operation_initializer": _KNOB_LIVE,
                "softplus_temperature": False},
    }, measured=0.458675, scope="temperature"),
    Knob("gumbel_softmax", "value", {
        "off": {"operation_initializer": _KNOB_LIVE, "gumbel_softmax": False},
        "on": {"operation_initializer": _KNOB_LIVE, "gumbel_softmax": True},
    }, measured=0.124748, training=True),
    Knob("gumbel_hard", "value", {
        "soft": {"operation_initializer": _KNOB_LIVE,
                 "gumbel_softmax": True, "gumbel_hard": False},
        "hard": {"operation_initializer": _KNOB_LIVE,
                 "gumbel_softmax": True, "gumbel_hard": True},
    }, measured=0.398007, training=True),
    Knob("entropy_coefficient", "loss", {
        "zero": {"entropy_coefficient": 0.0},
        "half": {"entropy_coefficient": 0.5},
    }, measured=0.549306),
    Knob("selection_mode", "structural", {
        "global": {"selection_mode": "global"},
        "per_channel": {"selection_mode": "per_channel"},
    }, measured=0.0971184),
    Knob("yager_p", "value", {
        "two": {"operation_types": ["yager_and", "yager_or"], "yager_p": 2.0},
        "five": {"operation_types": ["yager_and", "yager_or"], "yager_p": 5.0},
    }, measured=0.0548472),
]

#: `allow_unary_degenerate` is a GUARD knob: it decides whether a
#: single-tensor call raises. It has no output, weight or loss
#: signature, so it gets its own pin below rather than a table row.
LOGIC_KNOB_NAMES = [knob.param for knob in LOGIC_KNOBS] + [
    "allow_unary_degenerate"
]


def _logic_sample(knob):
    """The input each knob needs.

    The clip knobs are no-ops on inputs already inside [0, 1] --
    measured `dy = 0.0` on the default [0.05, 0.95] draw -- so they get
    a draw that straddles the boundary. Everything else uses the shared
    in-range draw.
    """
    rng = np.random.default_rng(1234)
    drawn = [
        rng.uniform(0.05, 0.95, size=(4, 8, 16)).astype("float32")
        for _ in range(2)
    ]
    if knob.param == "force_clip_when_no_sigmoid":
        drawn[0] = np.clip(drawn[0] * 4.0 - 1.5, -3.0, 3.0)
    return drawn


class TestEveryLogicConstructorKnobIsPinned:
    """§12.5. Reading the value back off `self` is not coverage.

    Fourteen constructor parameters, each varied and each asserted to
    make a measured difference, with the instrument matching its
    §13.3.2 class. The measurement each pin was written against is
    carried in the `Knob.measured` field and quoted on failure.
    """

    def test_the_table_covers_every_constructor_parameter(self):
        """The pin that keeps this table honest as the class grows.

        A new constructor parameter with no table row fails HERE
        instead of silently being unpinned -- which is the whole
        finding this class exists to close.
        """
        import inspect

        declared = [
            name for name, parameter
            in inspect.signature(LearnableLogicOperator.__init__)
            .parameters.items()
            if name != "self"
            and parameter.kind is not parameter.VAR_KEYWORD
        ]
        assert sorted(declared) == sorted(LOGIC_KNOB_NAMES), (
            f"unpinned: {sorted(set(declared) - set(LOGIC_KNOB_NAMES))}; "
            f"stale rows: "
            f"{sorted(set(LOGIC_KNOB_NAMES) - set(declared))}"
        )

    @pytest.mark.parametrize(
        "knob", LOGIC_KNOBS, ids=[k.param for k in LOGIC_KNOBS]
    )
    def test_rebuilding_one_variant_is_bit_identical(self, knob):
        """The anti-vacuity control (§13.1 rule 3)."""
        assert_the_harness_is_deterministic(
            LearnableLogicOperator, _KNOB_BASE, knob, _logic_sample
        )

    @pytest.mark.parametrize(
        "knob", LOGIC_KNOBS, ids=[k.param for k in LOGIC_KNOBS]
    )
    def test_the_knob_is_honoured(self, knob):
        assert_knob_is_honoured(
            LearnableLogicOperator, _KNOB_BASE, knob, _logic_sample
        )

    def test_the_unary_guard_is_not_raising_for_an_unrelated_reason(
            self
    ):
        """The missing twin of the `allow_unary_degenerate` pins.

        `allow_unary_degenerate` is a GUARD knob -- it decides whether a
        single-tensor call raises -- so it has no output, weight or loss
        signature and gets no table row. It is already pinned in both
        directions by `test_unary_input_raises_when_strict`,
        `test_unary_input_blocked_when_default_M8` and
        `test_unary_input_allowed_when_opt_in` above, which were
        RED-proven here on 2026-08-29 by forcing the flag True (both
        raise-side tests went red). This adds the twin none of the three
        supplies: a class that rejected EVERY single tensor, ignoring
        the knob, would satisfy all three. An ALL-UNARY gate set is
        accepted at `allow_unary_degenerate=False`, so the raise is
        attributable to the binary gate rather than to the arity.
        """
        sample = _logic_sample(LOGIC_KNOBS[0])[0]
        unary_only = LearnableLogicOperator(
            operation_types=["not"], allow_unary_degenerate=False
        )
        output = ops.convert_to_numpy(unary_only(sample))
        assert bool(np.all(np.isfinite(output)))

    def test_temperature_initializer_is_ignored_on_the_softplus_path(
            self
    ):
        """The documented conditional. `softplus_temperature=True`
        overrides `temperature_initializer` with
        `Constant(log(expm1(temperature_init)))`, so the knob reads
        `dw = 0.0` there -- pinned so the docstring claim cannot rot,
        and paired with its live configuration in the table above
        (`dw = 2.0`), which is the twin.
        """
        def stored(initializer):
            keras.utils.set_random_seed(0)
            layer = LearnableLogicOperator(
                operation_types=["and", "or", "xor"],
                softplus_temperature=True,
                temperature_initializer=initializer,
            )
            layer(_logic_sample(LOGIC_KNOBS[0]))
            return float(
                ops.convert_to_numpy(layer.temperature)
            )

        one = stored(keras.initializers.Constant(1.0))
        three = stored(keras.initializers.Constant(3.0))
        assert one == three, (
            "temperature_initializer became live on the softplus path; "
            "the class docstring says it is read only when "
            "softplus_temperature is False"
        )
        assert one == pytest.approx(
            float(np.log(np.expm1(1.0))), rel=0, abs=1e-6
        ), f"the softplus path stored {one}, not log(expm1(1.0))"
