import pytest
import tempfile
import os
import numpy as np
import keras
from keras import ops
from typing import Any, Dict

from dl_techniques.layers.logic.arithmetic_operators import LearnableArithmeticOperator

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

# ---------------------------------------------------------------------------
# Regression tests added in plan_2026-05-13_e52a5ac8
# ---------------------------------------------------------------------------

class TestPlanE52a5ac8Arithmetic:
    def test_compute_output_shape_accepts_list_form(self):
        """M2: previously returned None for list-form single shapes."""
        layer = LearnableArithmeticOperator()
        out_list = layer.compute_output_shape([None, 32])
        out_tuple = layer.compute_output_shape((None, 32))
        assert tuple(out_list) == (None, 32)
        assert tuple(out_tuple) == (None, 32)
        # Rank-3 list form
        out3 = layer.compute_output_shape([None, 16, 32])
        assert tuple(out3) == (None, 16, 32)
        # List-of-shapes (binary inputs) still works
        out_binary = layer.compute_output_shape([(None, 32), (None, 32)])
        assert tuple(out_binary) == (None, 32)


# ---------------------------------------------------------------------------
# Regression tests added in plan_2026-05-13_a2b0f17b
# ---------------------------------------------------------------------------

class TestPlanA2b0f17bArithmetic:
    """Regressions for full-rewrite plan."""

    def test_safe_power_preserves_sign(self):
        """C3: power(-2, 3) == -8 via Re((-|x|)^y) = cos(pi*y)*|x|^y."""
        op = LearnableArithmeticOperator(
            operation_types=['power'], use_scaling=False,
            exponent_clip_range=(-3.0, 3.0),
        )
        op.build((None, 4))
        x1 = ops.convert_to_tensor(np.array([[-2.0, 2.0, -3.0, 3.0]], dtype=np.float32))
        x2 = ops.convert_to_tensor(np.array([[3.0, 3.0, 2.0, 2.0]], dtype=np.float32))
        y = ops.convert_to_numpy(op([x1, x2]))
        np.testing.assert_allclose(y[0], [-8.0, 8.0, 9.0, 9.0], atol=1e-5)

    def test_safe_power_half_integer_negative_base_is_zero(self):
        op = LearnableArithmeticOperator(
            operation_types=['power'], use_scaling=False,
            exponent_clip_range=(-1.0, 1.0),
        )
        op.build((None, 1))
        x1 = ops.convert_to_tensor(np.array([[-4.0]], dtype=np.float32))
        x2 = ops.convert_to_tensor(np.array([[0.5]], dtype=np.float32))
        y = ops.convert_to_numpy(op([x1, x2]))
        np.testing.assert_allclose(y, [[0.0]], atol=1e-5)

    def test_smooth_divide_bounded_and_differentiable_at_zero(self):
        """H4/C: smooth mode gives finite forward AND non-zero gradient at
        x2=0, vs hard_clamp which gives 1/eps forward and ZERO gradient at
        the non-differentiable point (so optimizer cannot escape x2=0).

        Analytical: at x2=0, smooth |d/dx2| = |x1|/eps^2 (continuous).
        Hard-clamp at x2=0 has |grad| = 0 (sub-gradient of |.|max(.,eps)).
        """
        import tensorflow as tf
        eps = 1e-3
        x1 = tf.Variable(np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32))
        x2 = tf.Variable(np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        # Smooth
        op = LearnableArithmeticOperator(
            operation_types=['divide'], use_scaling=False,
            safe_divide_mode='smooth', epsilon=eps,
        )
        op.build((None, 4))
        with tf.GradientTape() as t:
            y = op([x1, x2])
            loss = tf.reduce_sum(y)
        _, g2 = t.gradient(loss, [x1, x2])
        smooth_max_grad = float(tf.reduce_max(tf.abs(g2)))
        # Forward bounded (=0 at x2=0), gradient finite and non-zero.
        np.testing.assert_allclose(y.numpy(), 0.0, atol=1e-9)
        assert 0.5 / eps**2 <= smooth_max_grad <= 2.0 / eps**2
        # Hard clamp comparison.
        op_hard = LearnableArithmeticOperator(
            operation_types=['divide'], use_scaling=False,
            safe_divide_mode='hard_clamp', epsilon=eps,
        )
        op_hard.build((None, 4))
        with tf.GradientTape() as t:
            y_hard = op_hard([x1, x2])
            loss = tf.reduce_sum(y_hard)
        _, g2_hard = t.gradient(loss, [x1, x2])
        # Hard-clamp at x2=0: sub-gradient of clamp is 0, so grad x2 = 0.
        # Forward shoots to 1/eps (=1000) — large but finite.
        assert float(tf.reduce_max(tf.abs(g2_hard))) < 1e-3
        assert float(tf.reduce_max(tf.abs(y_hard))) > 100.0
        # Smooth mode: forward zero but gradient pushes x2 off zero.
        assert smooth_max_grad > 0

    def test_softplus_temperature_round_trip(self):
        op = LearnableArithmeticOperator(
            softplus_temperature=True, temperature_init=2.0
        )
        op.build((None, 4))
        from keras import ops as kops
        assert abs(float(kops.softplus(op.temperature)) - 2.0) < 1e-5
        cfg = op.get_config()
        op2 = LearnableArithmeticOperator.from_config(cfg)
        op2.build((None, 4))
        assert op2.softplus_temperature is True

    def test_entropy_loss_added_when_coef_positive(self):
        op = LearnableArithmeticOperator(entropy_coefficient=0.5)
        op.build((None, 4))
        x = ops.convert_to_tensor(np.random.randn(2, 4).astype(np.float32))
        _ = op([x, x])
        assert len(op.losses) >= 1
        assert float(op.losses[0]) > 0

    def test_entropy_loss_absent_when_coef_zero(self):
        op = LearnableArithmeticOperator(entropy_coefficient=0.0)
        op.build((None, 4))
        x = ops.convert_to_tensor(np.random.randn(2, 4).astype(np.float32))
        _ = op([x, x])
        assert len(op.losses) == 0

    def test_gumbel_softmax_finite_output(self):
        op = LearnableArithmeticOperator(
            gumbel_softmax=True, gumbel_hard=True
        )
        op.build((None, 4))
        x = ops.convert_to_tensor(np.random.randn(2, 4).astype(np.float32))
        y = op([x, x])
        assert bool(ops.all(ops.isfinite(y)))

    def test_to_symbolic_returns_dominant_op(self):
        op = LearnableArithmeticOperator(
            operation_types=['add', 'multiply', 'subtract']
        )
        op.build((None, 4))
        op.operation_weights.assign([0.0, 10.0, 0.0])
        s = op.to_symbolic(top_k=1)
        assert s.startswith("multiply")

    def test_empty_operation_types_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            LearnableArithmeticOperator(operation_types=[])

    def test_invalid_safe_divide_mode_raises(self):
        with pytest.raises(ValueError, match="safe_divide_mode"):
            LearnableArithmeticOperator(safe_divide_mode='nope')

    def test_negative_entropy_coefficient_raises(self):
        with pytest.raises(ValueError, match="entropy_coefficient"):
            LearnableArithmeticOperator(entropy_coefficient=-0.1)


class TestPlan3a2f1d23ArithmeticC1:
    """Regression tests for plan_2026-05-13_3a2f1d23 step 1 (C1): canonical
    Jang (2017) Gumbel-softmax form. Expected: softmax((w + g) / T), NOT
    softmax((w/T) + g)."""

    def test_canonical_gumbel_form_low_temperature(self):
        """At low T, the Monte-Carlo mean of one-hot draws should converge to
        softmax(w/T) (the standard Gumbel-softmax marginal). The previous
        buggy form would over-weight the noise term and produce a far more
        uniform distribution at low T."""
        keras.utils.set_random_seed(42)
        weights = np.array([0.0, 1.0, 2.0, 0.5], dtype=np.float32)
        temperature = 0.1
        op = LearnableArithmeticOperator(
            operation_types=['add', 'subtract', 'multiply', 'min'],
            use_temperature=True,
            gumbel_softmax=True,
            gumbel_hard=False,
            softplus_temperature=False,
        )
        op.build((None, 4))
        op.operation_weights.assign(weights)
        op.temperature.assign(np.array(temperature, dtype=np.float32))

        # Monte-Carlo: average over many samples.
        n_samples = 4000
        accum = np.zeros(4, dtype=np.float64)
        for _ in range(n_samples):
            probs = ops.convert_to_numpy(op._operation_probs())
            accum += probs
        empirical = accum / n_samples

        # Canonical marginal: softmax(w/T).
        canonical_logits = weights / temperature
        canonical = np.exp(canonical_logits - canonical_logits.max())
        canonical /= canonical.sum()

        # Under canonical form at T=0.1, the MC mean of the soft samples
        # concentrates strongly on the argmax (empirically ~0.55-0.65).
        # Under the buggy softmax((w/T) + g) form, the noise dominates and
        # the distribution stays close to uniform (~0.25 on each index).
        # The threshold 0.45 cleanly separates canonical from buggy.
        assert empirical.argmax() == int(canonical.argmax())
        assert empirical[canonical.argmax()] >= 0.45, (
            f"Canonical mass on argmax too low: empirical={empirical}, "
            f"canonical={canonical}"
        )

    def test_gumbel_deterministic_skips_noise(self):
        """deterministic=True must produce the same probs as non-gumbel."""
        op = LearnableArithmeticOperator(
            operation_types=['add', 'subtract'],
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
        op = LearnableArithmeticOperator()
        assert op.softplus_temperature is True

    def test_operation_initializer_default_zeros_H2(self):
        """H2: operation_initializer default flipped to 'zeros' (plan_3a2f1d23)."""
        op = LearnableArithmeticOperator()
        assert op.operation_initializer.__class__.__name__ == 'Zeros'

    def test_to_symbolic_deterministic_under_gumbel_C5(self):
        """C5: to_symbolic() must be deterministic regardless of gumbel mode."""
        op = LearnableArithmeticOperator(
            operation_types=['add', 'subtract', 'multiply'],
            gumbel_softmax=True,
            gumbel_hard=False,
        )
        op.build((None, 4))
        op.operation_weights.assign([0.0, 0.0, 5.0])
        outputs = {op.to_symbolic(top_k=1) for _ in range(10)}
        assert len(outputs) == 1, f"to_symbolic() non-deterministic: {outputs}"
        assert next(iter(outputs)).startswith("multiply")


class TestPlan3a2f1d23ArithmeticPerChannelC3:
    """C3: per-channel selection mode on LearnableArithmeticOperator."""

    def test_weight_shape_per_channel(self):
        op = LearnableArithmeticOperator(
            operation_types=['add', 'multiply', 'subtract'],
            selection_mode='per_channel',
        )
        op.build((None, 8))
        assert op.operation_weights.shape == (8, 3)

    def test_weight_shape_global(self):
        op = LearnableArithmeticOperator(
            operation_types=['add', 'multiply', 'subtract'],
        )
        op.build((None, 8))
        assert op.operation_weights.shape == (3,)

    def test_forward_per_channel_rank2(self):
        op = LearnableArithmeticOperator(
            operation_types=['add', 'multiply'],
            selection_mode='per_channel',
            use_scaling=False,
        )
        x = ops.convert_to_tensor(np.random.randn(2, 4).astype(np.float32))
        y = op([x, x])
        assert y.shape == (2, 4)
        assert bool(ops.all(ops.isfinite(y)))

    def test_forward_per_channel_rank4(self):
        op = LearnableArithmeticOperator(
            operation_types=['add', 'multiply'],
            selection_mode='per_channel',
            use_scaling=False,
        )
        x = ops.convert_to_tensor(np.random.randn(2, 5, 5, 6).astype(np.float32))
        y = op([x, x])
        assert y.shape == (2, 5, 5, 6)

    def test_per_channel_distinct_channel_selection(self):
        """Each channel selects its own operator: with weights biasing channel
        0 toward 'add' and channel 1 toward 'multiply', outputs should
        reflect that."""
        op = LearnableArithmeticOperator(
            operation_types=['add', 'multiply'],
            selection_mode='per_channel',
            use_temperature=False,
            use_scaling=False,
        )
        op.build((None, 2))
        op.operation_weights.assign(np.array([[10.0, 0.0], [0.0, 10.0]], dtype=np.float32))
        x1 = ops.convert_to_tensor(np.array([[2.0, 3.0]], dtype=np.float32))
        x2 = ops.convert_to_tensor(np.array([[5.0, 7.0]], dtype=np.float32))
        y = ops.convert_to_numpy(op([x1, x2]))
        # Channel 0 picks add: 2+5=7. Channel 1 picks multiply: 3*7=21.
        np.testing.assert_allclose(y[0, 0], 7.0, atol=1e-3)
        np.testing.assert_allclose(y[0, 1], 21.0, atol=1e-3)

    def test_per_channel_round_trip(self):
        op = LearnableArithmeticOperator(
            operation_types=['add', 'multiply'],
            selection_mode='per_channel',
        )
        op.build((None, 4))
        cfg = op.get_config()
        assert cfg['selection_mode'] == 'per_channel'
        op2 = LearnableArithmeticOperator.from_config(cfg)
        assert op2.selection_mode == 'per_channel'

    def test_per_channel_requires_concrete_channel_dim(self):
        op = LearnableArithmeticOperator(selection_mode='per_channel')
        with pytest.raises(ValueError, match="last-axis"):
            op.build((None, None))

    def test_invalid_selection_mode_raises(self):
        with pytest.raises(ValueError, match="selection_mode"):
            LearnableArithmeticOperator(selection_mode='bogus')


# ---------------------------------------------------------------------
# plan_2026-05-13_e33114da regression tests
# ---------------------------------------------------------------------

class TestPlanE33114daArithmetic:
    """Regression tests for plan_2026-05-13_e33114da."""

    def test_gumbel_deterministic_at_inference(self):
        """B2: arithmetic Gumbel is deterministic at training=False."""
        keras.utils.set_random_seed(0)
        op = LearnableArithmeticOperator(
            gumbel_softmax=True, operation_types=['add', 'multiply', 'subtract'],
        )
        x = ops.convert_to_tensor(np.random.randn(2, 4).astype(np.float32))
        op.build(x.shape)
        o1 = ops.convert_to_numpy(op([x, x], training=False))
        o2 = ops.convert_to_numpy(op([x, x], training=False))
        np.testing.assert_allclose(o1, o2, atol=1e-7)

    def test_scaling_factor_sign_preserved(self):
        """D5: negative scaling_factor produces sign-flipped output."""
        op = LearnableArithmeticOperator(
            operation_types=['add'], use_scaling=True, scaling_init=1.0,
        )
        x = ops.convert_to_tensor(np.array([[1.0, 2.0]], dtype=np.float32))
        op([x, x])  # build
        # Set scale to negative
        op.scaling_factor.assign(np.array(-2.0, dtype=np.float32))
        y = ops.convert_to_numpy(op([x, x]))
        # add(x, x) = 2x = [[2, 4]]; * -2 = [[-4, -8]]
        np.testing.assert_allclose(y, [[-4.0, -8.0]], atol=1e-5)

    def test_exponent_clip_mode_smooth_gradient(self):
        """D7: smooth exponent_clip has non-zero gradient even outside range."""
        import tensorflow as tf
        op = LearnableArithmeticOperator(
            operation_types=['power'],
            exponent_clip_mode='smooth',
            exponent_clip_range=(-2.0, 2.0),
            use_scaling=False,
        )
        base = tf.constant([[2.0]], dtype=tf.float32)
        # exponent OUTSIDE the clip range -> hard mode would have zero gradient
        exp_val = tf.Variable(3.0, dtype=tf.float32)
        with tf.GradientTape() as tape:
            x2 = tf.reshape(exp_val, (1, 1))
            y = op([base, x2])
        grad = tape.gradient(y, exp_val)
        assert grad is not None and abs(float(grad.numpy())) > 1e-7, (
            f"Smooth mode gradient should be non-zero outside clip range; got {grad}"
        )

    def test_exponent_clip_mode_hard_default(self):
        """D7: default exponent_clip_mode is 'hard' (back-compat)."""
        op = LearnableArithmeticOperator(operation_types=['power'])
        assert op.exponent_clip_mode == 'hard'

    def test_exponent_clip_mode_round_trip(self):
        """D7: exponent_clip_mode survives get_config / from_config."""
        op = LearnableArithmeticOperator(
            operation_types=['power'], exponent_clip_mode='smooth'
        )
        cfg = op.get_config()
        assert cfg['exponent_clip_mode'] == 'smooth'
        op2 = LearnableArithmeticOperator.from_config(cfg)
        assert op2.exponent_clip_mode == 'smooth'

    def test_compute_output_shape_rejects_mismatched_binary(self):
        """D9: compute_output_shape raises on shape mismatch."""
        op = LearnableArithmeticOperator(operation_types=['add'])
        with pytest.raises(ValueError, match="same shape"):
            op.compute_output_shape([(None, 32), (None, 16)])

    def test_invalid_exponent_clip_mode_raises(self):
        with pytest.raises(ValueError, match="exponent_clip_mode"):
            LearnableArithmeticOperator(exponent_clip_mode='bogus')


# --------------------------------------------------------------------
# §12.5 -- every constructor parameter pinned, with the §13.3.2
# instrument that matches its knob class.
# --------------------------------------------------------------------

#: See the note in `test_logic_operators.py`: the default
#: `operation_initializer='zeros'` makes the gate softmax exactly
#: uniform, which hides the temperature and Gumbel knobs.
_KNOB_LIVE = keras.initializers.RandomNormal(stddev=0.5, seed=11)
_KNOB_OTHER = keras.initializers.RandomNormal(stddev=0.9, seed=13)

_KNOB_BASE = {"operation_types": ["add", "multiply"]}

ARITHMETIC_KNOBS = [
    Knob("operation_types", "structural", {
        "two": {"operation_types": ["add", "multiply"]},
        "three": {"operation_types": ["add", "multiply", "divide"]},
    }, measured=5.30675),
    Knob("use_temperature", "structural", {
        "on": {"operation_initializer": _KNOB_LIVE, "use_temperature": True},
        "off": {"operation_initializer": _KNOB_LIVE, "use_temperature": False},
    }, measured=0.0),
    Knob("temperature_init", "scoped_value", {
        "one": {"operation_initializer": _KNOB_LIVE, "temperature_init": 1.0},
        "five": {"operation_initializer": _KNOB_LIVE, "temperature_init": 5.0},
    }, measured=4.45191, scope="temperature"),
    Knob("use_scaling", "structural", {
        "on": {"operation_initializer": _KNOB_LIVE, "use_scaling": True},
        "off": {"operation_initializer": _KNOB_LIVE, "use_scaling": False},
    }, measured=0.0),
    Knob("scaling_init", "scoped_value", {
        "one": {"operation_initializer": _KNOB_LIVE, "scaling_init": 1.0},
        "two_five": {"operation_initializer": _KNOB_LIVE,
                     "scaling_init": 2.5},
    }, measured=1.5, scope="scaling_factor"),
    Knob("operation_initializer", "scoped_value", {
        "narrow": {"operation_initializer": _KNOB_LIVE},
        "wide": {"operation_initializer": _KNOB_OTHER},
    }, measured=0.526772, scope="operation_weights"),
    # Live only when softplus_temperature is False; the docstring says
    # so and the ignored configuration is pinned separately below.
    Knob("temperature_initializer", "scoped_value", {
        "one": {"softplus_temperature": False,
                "temperature_initializer": keras.initializers.Constant(1.0)},
        "three": {"softplus_temperature": False,
                  "temperature_initializer": keras.initializers.Constant(3.0)},
    }, measured=2.0, scope="temperature"),
    Knob("scaling_initializer", "scoped_value", {
        "one": {"scaling_initializer": keras.initializers.Constant(1.0)},
        "three": {"scaling_initializer": keras.initializers.Constant(3.0)},
    }, measured=2.0, scope="scaling_factor"),
    # The divide guards are invisible on a denominator far from zero:
    # measured dy = 9.5e-07 on the ordinary draw, which is float32
    # noise. `_arithmetic_sample` hands these two a denominator that
    # sweeps through 0.
    Knob("epsilon", "value", {
        "tiny": {"operation_types": ["divide"], "epsilon": 1e-7},
        "large": {"operation_types": ["divide"], "epsilon": 1e-2},
    }, measured=255432.0),
    Knob("safe_divide_mode", "value", {
        "hard_clamp": {"operation_types": ["divide"],
                       "safe_divide_mode": "hard_clamp"},
        "smooth": {"operation_types": ["divide"],
                   "safe_divide_mode": "smooth"},
    }, measured=665.328),
    Knob("power_clip_range", "value", {
        "wide": {"operation_types": ["power"],
                 "power_clip_range": (1e-7, 10.0)},
        "narrow": {"operation_types": ["power"],
                   "power_clip_range": (0.5, 0.6)},
    }, measured=0.494605),
    Knob("exponent_clip_range", "value", {
        "wide": {"operation_types": ["power"],
                 "exponent_clip_range": (-2.0, 2.0)},
        "narrow": {"operation_types": ["power"],
                   "exponent_clip_range": (0.1, 0.2)},
    }, measured=0.511649),
    Knob("exponent_clip_mode", "value", {
        "hard": {"operation_types": ["power"], "exponent_clip_mode": "hard"},
        "smooth": {"operation_types": ["power"],
                   "exponent_clip_mode": "smooth"},
    }, measured=0.0259506),
    # A reparameterization, not an output knob: both paths give an
    # effective temperature of temperature_init, so the output delta is
    # exactly 0.0 by construction and only the stored raw value moves.
    Knob("softplus_temperature", "scoped_value", {
        "on": {"operation_initializer": _KNOB_LIVE,
               "softplus_temperature": True},
        "off": {"operation_initializer": _KNOB_LIVE,
                "softplus_temperature": False},
    }, measured=0.458675, scope="temperature"),
    Knob("gumbel_softmax", "value", {
        "off": {"operation_initializer": _KNOB_LIVE, "gumbel_softmax": False},
        "on": {"operation_initializer": _KNOB_LIVE, "gumbel_softmax": True},
    }, measured=0.132332, training=True),
    Knob("gumbel_hard", "value", {
        "soft": {"operation_initializer": _KNOB_LIVE,
                 "gumbel_softmax": True, "gumbel_hard": False},
        "hard": {"operation_initializer": _KNOB_LIVE,
                 "gumbel_softmax": True, "gumbel_hard": True},
    }, measured=0.551703, training=True),
    Knob("entropy_coefficient", "loss", {
        "zero": {"entropy_coefficient": 0.0},
        "half": {"entropy_coefficient": 0.5},
    }, measured=0.346574),
    Knob("selection_mode", "structural", {
        "global": {"operation_initializer": _KNOB_LIVE,
                   "selection_mode": "global"},
        "per_channel": {"operation_initializer": _KNOB_LIVE,
                        "selection_mode": "per_channel"},
    }, measured=0.26746),
]

ARITHMETIC_KNOB_NAMES = [knob.param for knob in ARITHMETIC_KNOBS]


def _arithmetic_sample(knob):
    """The input each knob needs.

    `epsilon` and `safe_divide_mode` guard a division, and both are
    exactly invisible when the denominator is far from zero: measured
    `dy = 9.5e-07` (float32 noise) on the ordinary [0.05, 0.95] draw
    versus 255432 and 665.328 on a denominator that sweeps through 0.
    """
    rng = np.random.default_rng(1234)
    drawn = [
        rng.uniform(0.05, 0.95, size=(4, 8, 16)).astype("float32")
        for _ in range(2)
    ]
    if knob.param in ("epsilon", "safe_divide_mode"):
        drawn[1] = np.linspace(
            -1e-3, 1e-3, drawn[1].size
        ).reshape(drawn[1].shape).astype("float32")
    return drawn


class TestEveryArithmeticConstructorKnobIsPinned:
    """§12.5. Eighteen constructor parameters, each varied and each
    asserted to make a measured difference, with the instrument
    matching its §13.3.2 class.
    """

    def test_the_table_covers_every_constructor_parameter(self):
        """A new constructor parameter with no table row fails HERE."""
        import inspect

        declared = [
            name for name, parameter
            in inspect.signature(LearnableArithmeticOperator.__init__)
            .parameters.items()
            if name != "self"
            and parameter.kind is not parameter.VAR_KEYWORD
        ]
        assert sorted(declared) == sorted(ARITHMETIC_KNOB_NAMES), (
            f"unpinned: "
            f"{sorted(set(declared) - set(ARITHMETIC_KNOB_NAMES))}; "
            f"stale rows: "
            f"{sorted(set(ARITHMETIC_KNOB_NAMES) - set(declared))}"
        )

    @pytest.mark.parametrize(
        "knob", ARITHMETIC_KNOBS, ids=[k.param for k in ARITHMETIC_KNOBS]
    )
    def test_rebuilding_one_variant_is_bit_identical(self, knob):
        """The anti-vacuity control (§13.1 rule 3)."""
        assert_the_harness_is_deterministic(
            LearnableArithmeticOperator, _KNOB_BASE, knob,
            _arithmetic_sample,
        )

    @pytest.mark.parametrize(
        "knob", ARITHMETIC_KNOBS, ids=[k.param for k in ARITHMETIC_KNOBS]
    )
    def test_the_knob_is_honoured(self, knob):
        assert_knob_is_honoured(
            LearnableArithmeticOperator, _KNOB_BASE, knob,
            _arithmetic_sample,
        )

    def test_temperature_initializer_is_ignored_on_the_softplus_path(
            self
    ):
        """The documented conditional, and the twin of the live pin in
        the table above (`dw = 2.0` at `softplus_temperature=False`).
        """
        def stored(initializer):
            keras.utils.set_random_seed(0)
            layer = LearnableArithmeticOperator(
                operation_types=["add", "multiply"],
                softplus_temperature=True,
                temperature_initializer=initializer,
            )
            layer(_arithmetic_sample(ARITHMETIC_KNOBS[0]))
            return float(ops.convert_to_numpy(layer.temperature))

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
