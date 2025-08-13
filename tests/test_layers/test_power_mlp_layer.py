"""
Comprehensive test suite for modern PowerMLPLayer implementation.

This test suite follows modern Keras 3 testing patterns and validates the
PowerMLPLayer's functionality, serialization, and integration capabilities.

Place this file in: tests/test_layers/test_power_mlp_layer.py
"""

import pytest
import numpy as np
import keras
import tempfile
import os
from typing import Dict, Any
import tensorflow as tf

from dl_techniques.layers.ffn.power_mlp_layer import PowerMLPLayer


class TestPowerMLPLayer:
    """Comprehensive test suite for modern PowerMLPLayer implementation."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample input tensor for testing."""
        return keras.random.normal([4, 32])  # batch_size, input_dim

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard layer configuration for testing."""
        return {
            'units': 64,
            'k': 3,
            'kernel_initializer': 'he_normal',
            'use_bias': True
        }

    @pytest.fixture
    def custom_layer_config(self) -> Dict[str, Any]:
        """Custom layer configuration with regularization."""
        return {
            'units': 128,
            'k': 5,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'ones',
            'kernel_regularizer': keras.regularizers.L2(1e-4),
            'bias_regularizer': keras.regularizers.L1(1e-5),
            'use_bias': True
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = PowerMLPLayer(units=64)

        # Check stored configuration
        assert layer.units == 64
        assert layer.k == 3  # Default k
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None
        assert layer.use_bias is True

        # Check that layer is not built yet
        assert not layer.built

        # Following modern Keras 3 pattern: sub-layers are created in __init__
        assert layer.main_dense is not None
        assert layer.relu_k is not None
        assert layer.basis_function is not None
        assert layer.basis_dense is not None

        # But sub-layers should not be built yet
        assert not layer.main_dense.built
        assert not layer.relu_k.built
        assert not layer.basis_function.built
        assert not layer.basis_dense.built

    def test_initialization_custom(self, custom_layer_config):
        """Test initialization with comprehensive custom parameters."""
        layer = PowerMLPLayer(**custom_layer_config)

        # Verify all custom parameters are stored correctly
        assert layer.units == 128
        assert layer.k == 5
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)
        assert isinstance(layer.kernel_regularizer, keras.regularizers.L2)
        assert isinstance(layer.bias_regularizer, keras.regularizers.L1)
        assert layer.use_bias is True

        # Sub-layers should be created but not built
        assert layer.main_dense is not None
        assert layer.basis_dense is not None
        assert not layer.main_dense.built

    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid units values
        with pytest.raises(ValueError, match="units must be positive"):
            PowerMLPLayer(units=0)

        with pytest.raises(ValueError, match="units must be positive"):
            PowerMLPLayer(units=-10)

        with pytest.raises(TypeError, match="units must be an integer"):
            PowerMLPLayer(units=64.5)

        with pytest.raises(TypeError, match="units must be an integer"):
            PowerMLPLayer(units="64")

        # Test invalid k values
        with pytest.raises(ValueError, match="k must be positive"):
            PowerMLPLayer(units=64, k=0)

        with pytest.raises(ValueError, match="k must be positive"):
            PowerMLPLayer(units=64, k=-1)

        with pytest.raises(TypeError, match="k must be an integer"):
            PowerMLPLayer(units=64, k=2.5)

        with pytest.raises(TypeError, match="k must be an integer"):
            PowerMLPLayer(units=64, k="3")

    def test_build_process(self, sample_input, layer_config):
        """Test that the layer builds properly following modern patterns."""
        layer = PowerMLPLayer(**layer_config)

        # Layer should not be built initially
        assert not layer.built

        # Sub-layers should exist but not be built
        assert layer.main_dense is not None
        assert not layer.main_dense.built

        # Forward pass triggers build
        output = layer(sample_input)

        # Check that layer is now built
        assert layer.built

        # Check that all sub-layers are built
        assert layer.main_dense.built
        assert layer.relu_k.built
        assert layer.basis_function.built
        assert layer.basis_dense.built

        # Verify weights were created
        expected_weights = 3  # main_dense (kernel+bias) + basis_dense (kernel only, no bias)
        assert len(layer.weights) == expected_weights

        # Verify output shape
        assert output.shape == (sample_input.shape[0], layer_config['units'])

    def test_sublayer_configurations(self, sample_input, layer_config):
        """Test that sub-layers are configured correctly."""
        layer = PowerMLPLayer(**layer_config)
        layer(sample_input)  # Build the layer

        # Check main branch configuration
        assert layer.main_dense.units == layer_config['units']
        assert layer.main_dense.use_bias == layer_config['use_bias']

        # Check ReLU-k configuration
        assert layer.relu_k.k == layer_config['k']

        # Check basis branch configuration (should never use bias)
        assert layer.basis_dense.units == layer_config['units']
        assert layer.basis_dense.use_bias is False  # Always False by design

    def test_forward_pass_basic(self, sample_input, layer_config):
        """Test basic forward pass functionality."""
        layer = PowerMLPLayer(**layer_config)
        output = layer(sample_input)

        # Basic sanity checks
        expected_shape = (sample_input.shape[0], layer_config['units'])
        assert output.shape == expected_shape
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

    def test_dual_branch_architecture(self, sample_input):
        """Test that the dual-branch architecture works correctly."""
        layer = PowerMLPLayer(units=32, k=2)

        # Get output from the full layer
        full_output = layer(sample_input)

        # Manually compute each branch to verify the architecture
        main_branch = layer.main_dense(sample_input)
        main_branch = layer.relu_k(main_branch)

        basis_branch = layer.basis_function(sample_input)
        basis_branch = layer.basis_dense(basis_branch)

        # Manually combine branches
        manual_output = main_branch + basis_branch

        # Should match the layer's output
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(full_output),
            keras.ops.convert_to_numpy(manual_output),
            rtol=1e-6, atol=1e-6,
            err_msg="Full layer output should match manual branch combination"
        )

    def test_branch_independence(self, sample_input):
        """Test that branches operate independently before combination."""
        layer = PowerMLPLayer(units=32, k=3)
        layer(sample_input)  # Build the layer

        # Test main branch in isolation
        main_only = layer.main_dense(sample_input)
        main_only = layer.relu_k(main_only)

        # Test basis branch in isolation
        basis_only = layer.basis_function(sample_input)
        basis_only = layer.basis_dense(basis_only)

        # Both should produce valid outputs
        assert not keras.ops.any(keras.ops.isnan(main_only))
        assert not keras.ops.any(keras.ops.isinf(main_only))
        assert not keras.ops.any(keras.ops.isnan(basis_only))
        assert not keras.ops.any(keras.ops.isinf(basis_only))

        # Shapes should match
        assert main_only.shape == (sample_input.shape[0], 32)
        assert basis_only.shape == (sample_input.shape[0], 32)

    def test_compute_output_shape(self):
        """Test output shape computation."""
        test_cases = [
            # (units, input_shape, expected_output_shape)
            (64, (None, 32), (None, 64)),
            (128, (4, 16), (4, 128)),
            (256, (8, 10, 32), (8, 10, 256)),
            (32, (2, 5, 8, 16), (2, 5, 8, 32)),
        ]

        for units, input_shape, expected_shape in test_cases:
            layer = PowerMLPLayer(units=units)
            computed_shape = layer.compute_output_shape(input_shape)
            assert computed_shape == expected_shape

    def test_different_k_values(self, sample_input):
        """Test that different k values produce different behaviors."""
        k_values = [1, 2, 3, 5]
        outputs = []

        for k in k_values:
            layer = PowerMLPLayer(units=32, k=k)
            output = layer(sample_input)
            outputs.append(keras.ops.convert_to_numpy(output))

        # Different k values should produce different outputs
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not np.allclose(outputs[i], outputs[j], rtol=0.1), \
                    f"k={k_values[i]} and k={k_values[j]} produced too similar outputs"

    def test_use_bias_parameter(self, sample_input):
        """Test that use_bias parameter affects only the main branch."""
        # Test with bias
        layer_with_bias = PowerMLPLayer(units=32, k=2, use_bias=True)
        output_with_bias = layer_with_bias(sample_input)

        # Test without bias
        layer_without_bias = PowerMLPLayer(units=32, k=2, use_bias=False)
        output_without_bias = layer_without_bias(sample_input)

        # Should produce different outputs
        assert not np.allclose(
            keras.ops.convert_to_numpy(output_with_bias),
            keras.ops.convert_to_numpy(output_without_bias),
            rtol=0.1
        )

        # Check that main branch has/doesn't have bias as expected
        assert layer_with_bias.main_dense.use_bias is True
        assert layer_without_bias.main_dense.use_bias is False

        # Basis branch should never have bias
        assert layer_with_bias.basis_dense.use_bias is False
        assert layer_without_bias.basis_dense.use_bias is False

    def test_get_config_completeness(self, custom_layer_config):
        """Test that get_config contains all initialization parameters."""
        layer = PowerMLPLayer(**custom_layer_config)
        config = layer.get_config()

        # Verify all custom parameters are in config
        expected_keys = {
            'units', 'k', 'kernel_initializer', 'bias_initializer',
            'kernel_regularizer', 'bias_regularizer', 'use_bias'
        }

        config_keys = set(config.keys())
        assert expected_keys.issubset(config_keys)

        # Verify specific values
        assert config['units'] == 128
        assert config['k'] == 5
        assert config['use_bias'] is True

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle following modern patterns."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = PowerMLPLayer(**layer_config)(inputs)
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
        x = PowerMLPLayer(units=128, k=2)(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = PowerMLPLayer(units=64, k=3)(x)
        x = keras.layers.Dropout(0.2)(x)
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
        layer = PowerMLPLayer(
            units=64,
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
        if layer.losses:
            total_loss = sum(layer.losses)
            assert keras.ops.convert_to_numpy(total_loss) > 0

    def test_gradient_flow(self, sample_input, layer_config):
        """Test that gradients flow properly through the layer."""
        layer = PowerMLPLayer(**layer_config)

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
        layer = PowerMLPLayer(**layer_config)
        output = layer(sample_input, training=training)

        # Basic output validation
        expected_shape = (sample_input.shape[0], layer_config['units'])
        assert output.shape == expected_shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_different_input_dimensions(self):
        """Test layer with different input tensor dimensions."""
        layer = PowerMLPLayer(units=32, k=2)

        test_shapes = [
            (4, 16),        # 2D input
            (4, 8, 16),     # 3D input
            (2, 4, 8, 16),  # 4D input
            (1, 2, 4, 8, 16) # 5D input
        ]

        for shape in test_shapes:
            test_input = keras.random.normal(shape)
            output = layer(test_input)

            # Output should change only the last dimension
            expected_shape = shape[:-1] + (32,)
            assert output.shape == expected_shape
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = PowerMLPLayer(units=16, k=2)

        # Test different input value ranges
        test_cases = [
            keras.ops.zeros((4, 8)),                              # Zeros
            keras.ops.ones((4, 8)) * 1e-10,                     # Very small
            keras.ops.ones((4, 8)) * 1e5,                       # Large
            keras.random.normal((4, 8)) * 100,                  # Large random
            keras.random.normal((4, 8)) * 1e-5,                 # Small random
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Verify numerical stability
            assert not keras.ops.any(keras.ops.isnan(output)), "NaN detected"
            assert not keras.ops.any(keras.ops.isinf(output)), "Inf detected"

    def test_weight_structure_and_shapes(self, sample_input):
        """Test that layer weights have correct structure and shapes."""
        input_dim = sample_input.shape[-1]
        units = 64

        layer = PowerMLPLayer(units=units, k=2)
        layer(sample_input)  # Build the layer

        # Should have exactly 3 weights: main_dense (kernel+bias) + basis_dense (kernel only)
        assert len(layer.weights) == 3
        assert len(layer.trainable_variables) == 3

        # Verify weight shapes
        expected_shapes = [
            (input_dim, units),  # main_dense kernel
            (units,),            # main_dense bias
            (input_dim, units),  # basis_dense kernel (no bias)
        ]

        actual_shapes = [tuple(w.shape) for w in layer.weights]
        assert actual_shapes == expected_shapes

    def test_comparison_with_standard_dense(self, sample_input):
        """Compare PowerMLP with standard Dense layer to verify enhanced expressiveness."""
        # PowerMLP layer
        power_mlp = PowerMLPLayer(units=32, k=2)
        power_output = power_mlp(sample_input)

        # Standard Dense layer
        dense = keras.layers.Dense(32)
        dense_output = dense(sample_input)

        # Should produce different outputs (PowerMLP is more expressive)
        assert not np.allclose(
            keras.ops.convert_to_numpy(power_output),
            keras.ops.convert_to_numpy(dense_output),
            rtol=0.1
        )

        # Both should be valid
        assert not keras.ops.any(keras.ops.isnan(power_output))
        assert not keras.ops.any(keras.ops.isnan(dense_output))

        # Same output shape
        assert power_output.shape == dense_output.shape

    def test_branch_contribution_analysis(self, sample_input):
        """Test that both branches contribute meaningfully to the output."""
        layer = PowerMLPLayer(units=32, k=2)
        layer(sample_input)  # Build the layer

        # Get individual branch outputs
        main_output = layer.main_dense(sample_input)
        main_output = layer.relu_k(main_output)

        basis_output = layer.basis_function(sample_input)
        basis_output = layer.basis_dense(basis_output)

        # Check that both branches produce non-trivial outputs
        main_magnitude = keras.ops.mean(keras.ops.abs(main_output))
        basis_magnitude = keras.ops.mean(keras.ops.abs(basis_output))

        main_val = keras.ops.convert_to_numpy(main_magnitude)
        basis_val = keras.ops.convert_to_numpy(basis_magnitude)

        assert main_val > 1e-6, "Main branch output is too small"
        assert basis_val > 1e-6, "Basis branch output is too small"

    def test_repr_method(self):
        """Test the string representation of the layer."""
        layer = PowerMLPLayer(units=64, k=3, name="test_power_mlp")
        repr_str = repr(layer)

        assert "PowerMLPLayer" in repr_str
        assert "units=64" in repr_str
        assert "k=3" in repr_str
        assert "test_power_mlp" in repr_str

    def test_training_compatibility(self):
        """Test that the layer works correctly during training."""
        # Create a simple model
        model = keras.Sequential([
            keras.layers.Input(shape=(20,)),
            PowerMLPLayer(units=32, k=2),
            PowerMLPLayer(units=16, k=3),
            keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        # Create synthetic training data
        x_train = keras.random.normal([64, 20])
        y_train = keras.random.normal([64, 1])

        # Initial loss
        initial_loss = model.evaluate(x_train, y_train, verbose=0)

        # Train for a few epochs
        model.fit(x_train, y_train, epochs=3, batch_size=16, verbose=0)

        # Final loss
        final_loss = model.evaluate(x_train, y_train, verbose=0)

        # Training should complete without errors
        assert isinstance(final_loss, (int, float))
        assert not np.isnan(final_loss)
        assert not np.isinf(final_loss)


class TestPowerMLPLayerEdgeCases:
    """Test edge cases and boundary conditions for PowerMLPLayer."""

    def test_minimal_dimensions(self):
        """Test with minimal viable dimensions."""
        layer = PowerMLPLayer(units=1, k=1)
        test_input = keras.random.normal([2, 1])

        output = layer(test_input)
        assert output.shape == (2, 1)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_large_dimensions(self):
        """Test with large dimensions."""
        layer = PowerMLPLayer(units=1024, k=2)
        test_input = keras.random.normal([2, 512])

        output = layer(test_input)
        assert output.shape == (2, 1024)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_single_batch_item(self):
        """Test with single batch item."""
        layer = PowerMLPLayer(units=32, k=2)
        test_input = keras.random.normal([1, 16])  # Single item

        output = layer(test_input)
        assert output.shape == (1, 32)
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_layer_reuse(self):
        """Test that the same layer instance can be reused."""
        layer = PowerMLPLayer(units=32, k=2)

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

    def test_extreme_k_values(self):
        """Test with extreme k values."""
        # Very high k value
        layer_high_k = PowerMLPLayer(units=16, k=10)
        test_input = keras.random.normal([2, 8])

        output = layer_high_k(test_input)
        assert output.shape == (2, 16)
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

    def test_no_bias_configuration(self):
        """Test layer with bias disabled."""
        layer = PowerMLPLayer(units=32, k=2, use_bias=False)
        test_input = keras.random.normal([4, 16])

        output = layer(test_input)
        layer(test_input)  # Build

        # Should have only 2 weights (main_dense kernel + basis_dense kernel, no bias)
        assert len(layer.weights) == 2
        assert layer.main_dense.use_bias is False
        assert layer.basis_dense.use_bias is False  # Always False

        assert output.shape == (4, 32)
        assert not keras.ops.any(keras.ops.isnan(output))


if __name__ == "__main__":
    pytest.main([__file__])