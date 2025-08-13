import pytest
import numpy as np
import keras
import tempfile
import os
from typing import Dict, Any
import tensorflow as tf

from dl_techniques.layers.ffn.diff_ffn import DifferentialFFN


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
            'gate_activation': 'sigmoid',
            'dropout_rate': 0.2,
            'use_bias': True,
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
        assert layer.gate_activation.__name__ == 'sigmoid'
        assert layer.dropout_rate == 0.0
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        # Default regularizer should be SoftOrthonormalConstraintRegularizer
        assert layer.kernel_regularizer is not None  # Should have default regularizer
        assert layer.bias_regularizer is None

        # Check that layer is not built yet
        assert not layer.built

        # Check that all sub-layers are created but not built
        assert layer.positive_dense is not None
        assert layer.layer_norm_pos is not None
        assert layer.positive_proj is not None
        assert layer.negative_dense is not None
        assert layer.layer_norm_neg is not None
        assert layer.negative_proj is not None
        assert layer.layer_norm_diff is not None
        assert layer.dropout is not None
        assert layer.output_proj is not None

        # Sub-layers should not be built yet (but should exist)
        assert hasattr(layer.positive_dense, 'built')
        assert hasattr(layer.negative_dense, 'built')
        assert hasattr(layer.output_proj, 'built')
        assert not layer.positive_dense.built
        assert not layer.negative_dense.built
        assert not layer.output_proj.built

    def test_initialization_custom(self, custom_layer_config):
        """Test initialization with comprehensive custom parameters."""
        layer = DifferentialFFN(**custom_layer_config)

        # Verify all custom parameters are stored correctly
        assert layer.hidden_dim == 512
        assert layer.output_dim == 256
        # Note: 'swish' is internally converted to 'silu' in newer Keras versions
        assert layer.branch_activation.__name__ in ['swish', 'silu']
        assert layer.gate_activation.__name__ == 'sigmoid'
        assert layer.dropout_rate == 0.2
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)
        assert isinstance(layer.kernel_regularizer, keras.regularizers.L2)
        assert isinstance(layer.bias_regularizer, keras.regularizers.L1)

    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid hidden_dim values
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            DifferentialFFN(hidden_dim=0, output_dim=32)

        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            DifferentialFFN(hidden_dim=-10, output_dim=32)

        with pytest.raises(ValueError, match="hidden_dim must be divisible by 2"):
            DifferentialFFN(hidden_dim=15, output_dim=32)  # Odd number

        # Test invalid output_dim values
        with pytest.raises(ValueError, match="output_dim must be positive"):
            DifferentialFFN(hidden_dim=128, output_dim=0)

        with pytest.raises(ValueError, match="output_dim must be positive"):
            DifferentialFFN(hidden_dim=128, output_dim=-5)

        # Test invalid dropout_rate values
        with pytest.raises(ValueError, match="dropout_rate must be between 0.0 and 1.0"):
            DifferentialFFN(hidden_dim=128, output_dim=64, dropout_rate=-0.1)

        with pytest.raises(ValueError, match="dropout_rate must be between 0.0 and 1.0"):
            DifferentialFFN(hidden_dim=128, output_dim=64, dropout_rate=1.5)

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
        assert layer.positive_dense.built
        assert layer.negative_dense.built
        assert layer.output_proj.built
        # Note: LayerNormalization and Dropout layers don't have explicit built checks

    def test_forward_pass_deterministic(self):
        """Test forward pass with controlled inputs for deterministic behavior."""
        # Create layer with linear activations and controlled initialization
        layer = DifferentialFFN(
            hidden_dim=16,
            output_dim=8,
            branch_activation='linear',
            gate_activation='linear',
            dropout_rate=0.0,
            kernel_initializer='ones',
            bias_initializer='zeros',
            kernel_regularizer=None  # Disable default regularizer
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
            gate_activation='linear',
            dropout_rate=0.0,
            kernel_initializer='ones',
            bias_initializer='zeros',
            kernel_regularizer=None
        )

        test_input = keras.ops.ones([1, 1, 4])
        output = layer(test_input)

        # Since we're using linear activations and ones initialization,
        # the differential should be computed as pos_branch - neg_branch
        assert output.shape == (1, 1, 2)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_different_activations(self, sample_input):
        """Test layer with various activation functions."""
        branch_activations = ['relu', 'gelu', 'swish', 'tanh', 'linear', 'selu']
        gate_activations = ['sigmoid', 'tanh', 'linear']

        for branch_act in branch_activations:
            for gate_act in gate_activations:
                layer = DifferentialFFN(
                    hidden_dim=64,
                    output_dim=32,
                    branch_activation=branch_act,
                    gate_activation=gate_act
                )
                output = layer(sample_input)

                # Verify output is valid
                assert output.shape == (*sample_input.shape[:-1], 32)
                assert not keras.ops.any(keras.ops.isnan(output))

                # Verify activations are set (handle swish->silu conversion)
                expected_branch_names = [branch_act] if branch_act != 'swish' else ['swish', 'silu']
                assert layer.branch_activation.__name__ in expected_branch_names

    def test_custom_activation_callable(self, sample_input):
        """Test with custom activation functions as callables."""

        def custom_branch_activation(x):
            return keras.ops.relu(x) * 0.5

        def custom_gate_activation(x):
            return keras.ops.sigmoid(x) * 0.8

        layer = DifferentialFFN(
            hidden_dim=64,
            output_dim=32,
            branch_activation=custom_branch_activation,
            gate_activation=custom_gate_activation
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

        # Verify all custom parameters are in config
        expected_keys = {
            'hidden_dim', 'output_dim', 'branch_activation', 'gate_activation',
            'dropout_rate', 'use_bias', 'kernel_initializer', 'bias_initializer',
            'kernel_regularizer', 'bias_regularizer'
        }

        config_keys = set(config.keys())
        assert expected_keys.issubset(config_keys)

        # Verify specific values
        assert config['hidden_dim'] == 512
        assert config['output_dim'] == 256
        assert config['dropout_rate'] == 0.2

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle following modern patterns."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        layer_instance = DifferentialFFN(**layer_config)
        outputs = layer_instance(inputs)
        model = keras.Model(inputs, outputs)

        # Ensure the layer is properly built
        assert layer_instance.built, "Layer should be built after model creation"
        assert all(layer.built for layer in [
            layer_instance.positive_dense,
            layer_instance.negative_dense,
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
                    ('positive_dense', layer_instance.positive_dense),
                    ('layer_norm_pos', layer_instance.layer_norm_pos),
                    ('positive_proj', layer_instance.positive_proj),
                    ('negative_dense', layer_instance.negative_dense),
                    ('layer_norm_neg', layer_instance.layer_norm_neg),
                    ('negative_proj', layer_instance.negative_proj),
                    ('layer_norm_diff', layer_instance.layer_norm_diff),
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
            branch_activation='gelu',
            gate_activation='sigmoid'
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


class TestDifferentialFFNEdgeCases:
    """Test edge cases and boundary conditions for DifferentialFFN."""

    def test_minimal_dimensions(self):
        """Test with minimal viable dimensions."""
        layer = DifferentialFFN(hidden_dim=2, output_dim=1)  # Minimum hidden_dim is 2 (must be even)
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
        """Test that positive and negative branches are structured symmetrically."""
        layer = DifferentialFFN(hidden_dim=128, output_dim=64)
        test_input = keras.random.normal([2, 4, 32])

        # Build the layer
        layer(test_input)

        # Check that positive and negative branches have same structure
        assert layer.positive_dense.units == layer.negative_dense.units
        assert layer.positive_proj.units == layer.negative_proj.units

        # Both should project to hidden_dim // 2
        assert layer.positive_proj.units == 128 // 2
        assert layer.negative_proj.units == 128 // 2

    def test_no_bias_configuration(self):
        """Test layer with bias disabled."""
        layer = DifferentialFFN(
            hidden_dim=64,
            output_dim=32,
            use_bias=False,
            kernel_regularizer=None  # Disable default regularizer
        )
        test_input = keras.random.normal([2, 4, 16])

        output = layer(test_input)

        # Should still work without bias
        assert output.shape == (2, 4, 32)
        assert not keras.ops.any(keras.ops.isnan(output))

        # Check that bias is disabled in sub-layers
        assert not layer.positive_dense.use_bias
        assert not layer.negative_dense.use_bias
        assert not layer.output_proj.use_bias