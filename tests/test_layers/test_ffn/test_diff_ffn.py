import pytest
import numpy as np
import keras
import tempfile
import os
from typing import Dict, Any
import tensorflow as tf

from dl_techniques.layers.ffn.diff_ffn import DifferentialFFN
from dl_techniques.regularizers.soft_orthogonal import (
    SoftOrthonormalConstraintRegularizer,
)


class TestDifferentialFFN:
    """Comprehensive test suite for modern DifferentialFFN layer implementation."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample input tensor for testing."""
        return keras.random.normal([8, 16, 128])  # batch, seq_len, features

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard layer configuration for testing."""
        return {
            'hidden_dim': 256,
            'output_dim': 128,
            'dropout_rate': 0.1,
            'branch_activation': 'gelu'
        }

    @pytest.fixture
    def custom_layer_config(self) -> Dict[str, Any]:
        """Custom layer configuration with comprehensive parameters."""
        return {
            'hidden_dim': 512,
            'output_dim': 256,
            'branch_activation': 'swish',
            'dropout_rate': 0.2,
            'use_bias': True,
            'tie_branches': False,
            'kernel_initializer': 'he_normal',
            'bias_initializer': 'ones',
            'kernel_regularizer': keras.regularizers.L2(1e-4),
            'bias_regularizer': keras.regularizers.L1(1e-5)
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = DifferentialFFN(hidden_dim=128, output_dim=64)

        # Check stored configuration
        assert layer.hidden_dim == 128
        assert layer.output_dim == 64
        assert layer.branch_activation.__name__ == 'gelu'
        assert layer.dropout_rate == 0.0
        assert layer.use_bias is True
        # tie_branches defaults to True -- the only setting that yields the
        # push-pull odd-symmetry guarantee (see diff_ffn.py module docstring).
        assert layer.tie_branches is True
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        # No kernel regularizer by default -- None means OFF, not "substitute one".
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None

        # Check that layer is not built yet
        assert not layer.built

        # Check that the shared-branch sub-layers are created but not built.
        # With tie_branches=True (the default), there is no separate "negative"
        # Dense/LayerNorm pair -- both polarities traverse branch_dense/branch_norm.
        assert layer.branch_dense is not None
        assert layer.branch_norm is not None
        assert layer.branch_dense_neg is None
        assert layer.branch_norm_neg is None
        assert layer.norm_diff is not None
        assert layer.dropout is not None
        assert layer.output_proj is not None

        # Sub-layers should not be built yet (but should exist)
        assert hasattr(layer.branch_dense, 'built')
        assert hasattr(layer.output_proj, 'built')
        assert not layer.branch_dense.built
        assert not layer.output_proj.built

    def test_initialization_custom(self, custom_layer_config):
        """Test initialization with comprehensive custom parameters."""
        layer = DifferentialFFN(**custom_layer_config)

        # Verify all custom parameters are stored correctly
        assert layer.hidden_dim == 512
        assert layer.output_dim == 256
        # Note: 'swish' is internally converted to 'silu' in newer Keras versions
        assert layer.branch_activation.__name__ in ['swish', 'silu']
        assert layer.dropout_rate == 0.2
        assert layer.use_bias is True
        assert layer.tie_branches is False
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)
        assert isinstance(layer.kernel_regularizer, keras.regularizers.L2)
        assert isinstance(layer.bias_regularizer, keras.regularizers.L1)

        # tie_branches=False must actually create the untied ablation pair.
        assert layer.branch_dense_neg is not None
        assert layer.branch_norm_neg is not None

    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid hidden_dim values
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            DifferentialFFN(hidden_dim=0, output_dim=32)

        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            DifferentialFFN(hidden_dim=-10, output_dim=32)

        # hidden_dim has NO parity requirement in the current push-pull design:
        # both polarities traverse the same shared branch map and land on the
        # same width by construction (see class docstring, "D = hidden_dim,
        # unconstrained"). An odd hidden_dim must build and run cleanly.
        odd_layer = DifferentialFFN(hidden_dim=15, output_dim=32)
        odd_output = odd_layer(keras.ops.ones((2, 4)))
        assert odd_output.shape == (2, 32)

        # Test invalid output_dim values
        with pytest.raises(ValueError, match="output_dim must be positive"):
            DifferentialFFN(hidden_dim=128, output_dim=0)

        with pytest.raises(ValueError, match="output_dim must be positive"):
            DifferentialFFN(hidden_dim=128, output_dim=-5)

        # Test invalid dropout_rate values -- half-open [0.0, 1.0), matching
        # what keras.layers.Dropout.call() itself accepts under training=True.
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            DifferentialFFN(hidden_dim=128, output_dim=64, dropout_rate=-0.1)

        with pytest.raises(ValueError, match="dropout_rate must be in"):
            DifferentialFFN(hidden_dim=128, output_dim=64, dropout_rate=1.5)

        with pytest.raises(ValueError, match="dropout_rate must be in"):
            DifferentialFFN(hidden_dim=128, output_dim=64, dropout_rate=1.0)

    def test_forward_pass_basic(self, sample_input, layer_config):
        """Test basic forward pass functionality."""
        layer = DifferentialFFN(**layer_config)
        output = layer(sample_input)

        # Check output shape
        expected_shape = list(sample_input.shape)
        expected_shape[-1] = layer_config['output_dim']
        assert output.shape == tuple(expected_shape)

        # Basic sanity checks
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

        # Check that layer is now built
        assert layer.built
        assert layer.branch_dense.built
        assert layer.output_proj.built
        # tie_branches defaults to True, so there is no separate negative branch.
        assert layer.branch_dense_neg is None
        # Note: LayerNormalization and Dropout layers don't have explicit built checks

    def test_forward_pass_deterministic(self):
        """Test forward pass with controlled inputs for deterministic behavior."""
        # Create layer with linear activations and controlled initialization
        layer = DifferentialFFN(
            hidden_dim=16,
            output_dim=8,
            branch_activation='linear',
            dropout_rate=0.0,
            kernel_initializer='ones',
            bias_initializer='zeros',
            # None is a true off switch here: no regularizer is installed or
            # substituted (see test_kernel_regularizer_none_is_a_true_off_switch).
            kernel_regularizer=None
        )

        controlled_input = keras.ops.ones([2, 4, 10])
        output = layer(controlled_input)

        # Verify output properties
        assert output.shape == (2, 4, 8)
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

    def test_differential_computation(self):
        """Test that differential computation works correctly."""
        # Create a simple case where we can verify the differential logic
        layer = DifferentialFFN(
            hidden_dim=4,
            output_dim=2,
            branch_activation='linear',
            dropout_rate=0.0,
            kernel_initializer='ones',
            bias_initializer='zeros',
            kernel_regularizer=None
        )

        test_input = keras.ops.ones([1, 1, 4])
        output = layer(test_input)

        # Since we're using a linear branch activation and ones initialization,
        # the differential should be computed as push - pull (see module
        # docstring). With an all-ones input, x_neg = ReLU(-1) = 0, so pull
        # collapses to the branch map evaluated at zero -- a nonzero constant
        # here because branch_dense carries a bias (use_bias defaults True).
        assert output.shape == (1, 1, 2)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_different_activations(self, sample_input):
        """Test layer with various branch activation functions.

        `gate_activation` does not exist on this layer -- there is no gate.
        The current design applies one shared `branch_activation` to both
        polarities (see module docstring); this only needs to vary that one
        parameter, not a cross product with a nonexistent second one.
        """
        branch_activations = ['relu', 'gelu', 'swish', 'tanh', 'linear', 'selu']

        for branch_act in branch_activations:
            layer = DifferentialFFN(
                hidden_dim=64,
                output_dim=32,
                branch_activation=branch_act
            )
            output = layer(sample_input)

            # Verify output is valid
            assert output.shape == (*sample_input.shape[:-1], 32)
            assert not keras.ops.any(keras.ops.isnan(output))

            # Verify activations are set (handle swish->silu conversion)
            expected_branch_names = [branch_act] if branch_act != 'swish' else ['swish', 'silu']
            assert layer.branch_activation.__name__ in expected_branch_names

    def test_custom_activation_callable(self, sample_input):
        """Test with a custom branch activation function as a callable."""

        def custom_branch_activation(x):
            return keras.ops.relu(x) * 0.5

        layer = DifferentialFFN(
            hidden_dim=64,
            output_dim=32,
            branch_activation=custom_branch_activation
        )
        output = layer(sample_input)

        assert output.shape == (*sample_input.shape[:-1], 32)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_dropout_training_vs_inference(self, sample_input):
        """Test dropout behavior in training vs inference modes."""
        layer = DifferentialFFN(hidden_dim=128, output_dim=64, dropout_rate=0.5)

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
        expected_shape = (*sample_input.shape[:-1], 64)
        assert output_train_1.shape == output_train_2.shape == expected_shape

    def test_compute_output_shape(self):
        """Test output shape computation."""
        test_cases = [
            # (hidden_dim, output_dim, input_shape, expected_output_shape)
            (128, 64, (None, 16, 32), (None, 16, 64)),
            (256, 128, (4, 8, 64), (4, 8, 128)),
            (512, 256, (2, 10, 20, 128), (2, 10, 20, 256)),
            (64, 32, (1, 4, 8, 16), (1, 4, 8, 32)),
        ]

        for hidden_dim, output_dim, input_shape, expected_shape in test_cases:
            layer = DifferentialFFN(hidden_dim=hidden_dim, output_dim=output_dim)
            computed_shape = layer.compute_output_shape(input_shape)
            assert computed_shape == expected_shape

    def test_get_config_completeness(self, custom_layer_config):
        """Test that get_config contains all initialization parameters."""
        layer = DifferentialFFN(**custom_layer_config)
        config = layer.get_config()

        # Verify all custom parameters are in config. No 'gate_activation' key
        # exists on this layer -- there is no gate in the current push-pull
        # design (see module docstring).
        expected_keys = {
            'hidden_dim', 'output_dim', 'branch_activation',
            'dropout_rate', 'use_bias', 'tie_branches', 'epsilon',
            'kernel_initializer', 'bias_initializer',
            'kernel_regularizer', 'bias_regularizer'
        }

        config_keys = set(config.keys())
        assert expected_keys.issubset(config_keys)
        assert 'gate_activation' not in config_keys

        # Verify specific values
        assert config['hidden_dim'] == 512
        assert config['output_dim'] == 256
        assert config['dropout_rate'] == 0.2
        assert config['tie_branches'] is False

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle following modern patterns."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        layer_instance = DifferentialFFN(**layer_config)
        outputs = layer_instance(inputs)
        model = keras.Model(inputs, outputs)

        # Ensure the layer is properly built. layer_config uses tie_branches'
        # default (True), so branch_dense_neg stays None -- only branch_dense
        # and output_proj are real sub-layers to check here.
        assert layer_instance.built, "Layer should be built after model creation"
        assert all(layer.built for layer in [
            layer_instance.branch_dense,
            layer_instance.output_proj
        ]), "All sub-layers should be built"

        # Get original prediction
        original_prediction = model(sample_input)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')

            try:
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

            except Exception as e:
                # Print debug information if serialization fails
                print(f"Serialization failed: {e}")
                print(f"Layer built: {layer_instance.built}")
                print("Sub-layer build status:")
                for name, sublayer in [
                    ('branch_dense', layer_instance.branch_dense),
                    ('branch_norm', layer_instance.branch_norm),
                    ('norm_diff', layer_instance.norm_diff),
                    ('dropout', layer_instance.dropout),
                    ('output_proj', layer_instance.output_proj),
                ]:
                    built_status = getattr(sublayer, 'built', 'N/A')
                    print(f"  {name}: built={built_status}")
                raise

    def test_model_integration_complex(self, sample_input):
        """Test layer integration in a complex model."""
        inputs = keras.Input(shape=sample_input.shape[1:])

        # Multi-layer architecture with DifferentialFFN
        x = DifferentialFFN(hidden_dim=256, output_dim=128, dropout_rate=0.1)(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = DifferentialFFN(hidden_dim=512, output_dim=64, dropout_rate=0.2)(x)
        x = keras.layers.LayerNormalization()(x)
        x = DifferentialFFN(hidden_dim=128, output_dim=32)(x)
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
        layer = DifferentialFFN(
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
        layer = DifferentialFFN(**layer_config)

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
        layer = DifferentialFFN(**layer_config)
        output = layer(sample_input, training=training)

        # Basic output validation
        expected_shape = (*sample_input.shape[:-1], layer_config['output_dim'])
        assert output.shape == expected_shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_different_input_dimensions(self):
        """Test layer with different input tensor dimensions."""
        layer = DifferentialFFN(hidden_dim=64, output_dim=32)

        test_shapes = [
            (4, 16),  # 2D input
            (4, 8, 16),  # 3D input
            (2, 4, 8, 16),  # 4D input
            (1, 2, 4, 8, 16)  # 5D input
        ]

        for shape in test_shapes:
            test_input = keras.random.normal(shape)
            output = layer(test_input)

            # Output should have same shape except last dimension
            expected_shape = (*shape[:-1], 32)
            assert output.shape == expected_shape
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = DifferentialFFN(
            hidden_dim=64,
            output_dim=32,
            branch_activation='gelu'
        )

        # Test different input value ranges
        test_cases = [
            keras.ops.zeros((4, 8, 16)),  # Zeros
            keras.ops.ones((4, 8, 16)) * 1e-10,  # Very small
            keras.ops.ones((4, 8, 16)) * 1e3,  # Large
            keras.random.normal((4, 8, 16)) * 100,  # Large random
            keras.random.normal((4, 8, 16)) * 1e-5,  # Small random
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Verify numerical stability
            assert not keras.ops.any(keras.ops.isnan(output)), "NaN detected"
            assert not keras.ops.any(keras.ops.isinf(output)), "Inf detected"


def test_kernel_regularizer_none_is_a_true_off_switch():
    """`kernel_regularizer=None` means NO regularizer, and it reaches the Dense layers.

    Supersedes DECISION plan-2026-09-01T201957-15d7a40e/D-002, which pinned the
    coefficients of a substituted default. The substitution is gone: this layer
    used to install a `SoftOrthonormalConstraintRegularizer` whenever the
    argument was None, which made the parameter impossible to switch off at all.

    Asserting `layer.losses == []` after a real forward pass is the
    discriminating check -- an attribute-only assertion would still pass if a
    regularizer were attached to the sub-layers by some other route.
    """
    layer = DifferentialFFN(hidden_dim=8, output_dim=4)
    assert layer.kernel_regularizer is None

    layer(keras.random.normal([2, 3, 6]))
    assert layer.losses == []
    for sub in (layer.branch_dense, layer.output_proj):
        assert sub.kernel_regularizer is None


def test_kernel_regularizer_opt_in_reaches_the_sublayers():
    """An explicitly passed regularizer is installed and actually contributes a loss.

    The anti-vacuity partner to the test above: together they show the switch
    moves in both directions, so neither is satisfied by a layer that simply
    never regularizes anything.
    """
    reg = SoftOrthonormalConstraintRegularizer(
        lambda_coefficient=1e-3, l1_coefficient=0.0,
        l2_coefficient=0.0, use_matrix_scaling=True)
    layer = DifferentialFFN(hidden_dim=8, output_dim=4, kernel_regularizer=reg)

    layer(keras.random.normal([2, 3, 6]))
    assert len(layer.losses) > 0
    assert all(float(loss) > 0.0 for loss in layer.losses)


class TestDifferentialFFNEdgeCases:
    """Test edge cases and boundary conditions for DifferentialFFN."""

    def test_minimal_dimensions(self):
        """Test with minimal viable dimensions."""
        layer = DifferentialFFN(hidden_dim=2, output_dim=1)  # Minimal positive hidden_dim; no parity constraint
        test_input = keras.random.normal([2, 3, 4])

        output = layer(test_input)
        assert output.shape == (2, 3, 1)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_large_dimensions(self):
        """Test with large dimensions."""
        layer = DifferentialFFN(hidden_dim=2048, output_dim=512)
        test_input = keras.random.normal([2, 4, 256])

        output = layer(test_input)
        assert output.shape == (2, 4, 512)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_zero_dropout_rate(self):
        """Test with zero dropout rate."""
        sample_input = keras.random.normal([4, 8, 32])
        layer = DifferentialFFN(hidden_dim=64, output_dim=32, dropout_rate=0.0)

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
        layer = DifferentialFFN(hidden_dim=32, output_dim=16)
        test_input = keras.random.normal([1, 1, 8])  # Single item, single step

        output = layer(test_input)
        assert output.shape == (1, 1, 16)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_layer_reuse(self):
        """Test that the same layer instance can be reused."""
        layer = DifferentialFFN(hidden_dim=64, output_dim=32)

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

    def test_branch_symmetry(self):
        """Test the push-pull branch relationship the current design actually has.

        The old "Dense(D) -> Dense(D/2) projection per polarity" architecture
        is gone (see module docstring, "Differences from the earlier untied,
        gated design"); there is no `positive_proj`/`negative_proj` bottleneck
        to compare units on. What must actually hold: `tie_branches=True`
        (the default) means both polarities share ONE branch map -- no second
        Dense/LayerNorm pair is even created -- and `tie_branches=False`
        creates a structurally matching but independently-initialized pair
        (same bug class as the GatedMLP gate/value tie fixed elsewhere in
        this package; verified here by measurement, not by reading).
        """
        tied = DifferentialFFN(hidden_dim=128, output_dim=64, tie_branches=True)
        test_input = keras.random.normal([2, 4, 32])
        tied(test_input)

        # tie_branches=True: no independent negative branch exists at all.
        assert tied.branch_dense_neg is None
        assert tied.branch_norm_neg is None

        untied = DifferentialFFN(hidden_dim=128, output_dim=64, tie_branches=False)
        untied(test_input)

        # tie_branches=False: a structurally matching pair is created ...
        assert untied.branch_dense_neg is not None
        assert untied.branch_norm_neg is not None
        assert untied.branch_dense.units == untied.branch_dense_neg.units

        # ... but the two kernels must be genuinely independent draws, not
        # tied by a shared initializer instance (max|delta| must never be 0.0).
        kernel_pos = keras.ops.convert_to_numpy(untied.branch_dense.kernel)
        kernel_neg = keras.ops.convert_to_numpy(untied.branch_dense_neg.kernel)
        max_delta = float(np.max(np.abs(kernel_pos - kernel_neg)))
        assert max_delta > 0.0, "untied branch kernels must not be tied"

    def test_no_bias_configuration(self):
        """Test layer with bias disabled."""
        layer = DifferentialFFN(
            hidden_dim=64,
            output_dim=32,
            use_bias=False,
            # None is a true off switch here: no regularizer is installed or
            # substituted (see test_kernel_regularizer_none_is_a_true_off_switch).
            kernel_regularizer=None
        )
        test_input = keras.random.normal([2, 4, 16])

        output = layer(test_input)

        # Should still work without bias
        assert output.shape == (2, 4, 32)
        assert not keras.ops.any(keras.ops.isnan(output))

        # use_bias governs the shared branch Dense; output_proj is
        # unconditionally bias-free regardless of use_bias (structural
        # requirement for the odd-symmetry proof, see module docstring).
        assert not layer.branch_dense.use_bias
        assert not layer.output_proj.use_bias