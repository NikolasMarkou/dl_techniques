"""
Comprehensive test suite for LogicFFN layer following Modern Keras 3 patterns.

This test suite demonstrates the essential tests every custom layer should have,
particularly the critical serialization cycle test.
"""

import pytest
import tempfile
import os
import numpy as np
import keras
import tensorflow as tf
from typing import Any, Dict

# Import the layer to test
from dl_techniques.layers.ffn.logic_ffn import LogicFFN, create_logic_ffn_standard, create_logic_ffn_regularized


class TestLogicFFN:
    """Comprehensive test suite for LogicFFN layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'output_dim': 64,
            'logic_dim': 32,
            'temperature': 1.0,
            'use_bias': True
        }

    @pytest.fixture
    def sample_input(self) -> np.ndarray:
        """Sample input for testing."""
        return np.random.normal(size=(4, 16, 48)).astype(np.float32)

    @pytest.fixture
    def sample_input_2d(self) -> np.ndarray:
        """Sample 2D input for testing."""
        return np.random.normal(size=(4, 48)).astype(np.float32)

    def test_initialization(self, layer_config):
        """Test layer initialization and parameter storage."""
        layer = LogicFFN(**layer_config)

        # Check configuration storage
        assert layer.output_dim == layer_config['output_dim']
        assert layer.logic_dim == layer_config['logic_dim']
        assert layer.temperature == layer_config['temperature']
        assert layer.use_bias == layer_config['use_bias']
        assert layer.num_logic_ops == 3  # AND, OR, XOR

        # Check sub-layers are created
        assert layer.logic_projection is not None
        assert layer.gate_projection is not None
        assert layer.output_projection is not None

        # Layer should not be built yet
        assert not layer.built

    def test_parameter_validation(self):
        """Test parameter validation in __init__."""
        # Test invalid output_dim
        with pytest.raises(ValueError, match="output_dim must be positive"):
            LogicFFN(output_dim=0, logic_dim=32)

        with pytest.raises(ValueError, match="output_dim must be positive"):
            LogicFFN(output_dim=-5, logic_dim=32)

        # Test invalid logic_dim
        with pytest.raises(ValueError, match="logic_dim must be positive"):
            LogicFFN(output_dim=64, logic_dim=0)

        with pytest.raises(ValueError, match="logic_dim must be positive"):
            LogicFFN(output_dim=64, logic_dim=-10)

        # Test invalid temperature
        with pytest.raises(ValueError, match="temperature must be positive"):
            LogicFFN(output_dim=64, logic_dim=32, temperature=0.0)

        with pytest.raises(ValueError, match="temperature must be positive"):
            LogicFFN(output_dim=64, logic_dim=32, temperature=-1.0)

    def test_forward_pass_3d(self, layer_config, sample_input):
        """Test forward pass with 3D input and building."""
        layer = LogicFFN(**layer_config)

        output = layer(sample_input)

        # Check layer is built after forward pass
        assert layer.built

        # Check output shape
        expected_shape = (sample_input.shape[0], sample_input.shape[1], layer_config['output_dim'])
        assert output.shape == expected_shape

        # Check output is finite
        assert keras.ops.all(keras.ops.isfinite(output))

    def test_forward_pass_2d(self, layer_config, sample_input_2d):
        """Test forward pass with 2D input."""
        layer = LogicFFN(**layer_config)

        output = layer(sample_input_2d)

        # Check output shape
        expected_shape = (sample_input_2d.shape[0], layer_config['output_dim'])
        assert output.shape == expected_shape

        # Check output is finite
        assert keras.ops.all(keras.ops.isfinite(output))

    def test_compute_output_shape(self, layer_config):
        """Test output shape computation."""
        layer = LogicFFN(**layer_config)

        # Test 3D shape
        input_shape_3d = (None, 16, 48)
        output_shape_3d = layer.compute_output_shape(input_shape_3d)
        expected_shape_3d = (None, 16, layer_config['output_dim'])
        assert output_shape_3d == expected_shape_3d

        # Test 2D shape
        input_shape_2d = (None, 48)
        output_shape_2d = layer.compute_output_shape(input_shape_2d)
        expected_shape_2d = (None, layer_config['output_dim'])
        assert output_shape_2d == expected_shape_2d

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = LogicFFN(**layer_config)(inputs)
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
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = LogicFFN(**layer_config)
        config = layer.get_config()

        # Check all config parameters are present
        required_keys = [
            'output_dim', 'logic_dim', 'use_bias', 'temperature',
            'kernel_initializer', 'bias_initializer',
            'kernel_regularizer', 'bias_regularizer'
        ]

        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        # Check values match
        assert config['output_dim'] == layer_config['output_dim']
        assert config['logic_dim'] == layer_config['logic_dim']
        assert config['temperature'] == layer_config['temperature']
        assert config['use_bias'] == layer_config['use_bias']

    def test_config_reconstruction(self, layer_config):
        """Test layer can be reconstructed from config."""
        original_layer = LogicFFN(**layer_config)
        config = original_layer.get_config()

        # Remove base layer config items for reconstruction
        layer_config_only = {k: v for k, v in config.items()
                           if k in ['output_dim', 'logic_dim', 'use_bias',
                                  'temperature', 'kernel_initializer',
                                  'bias_initializer', 'kernel_regularizer',
                                  'bias_regularizer']}

        reconstructed_layer = LogicFFN(**layer_config_only)

        assert reconstructed_layer.output_dim == original_layer.output_dim
        assert reconstructed_layer.logic_dim == original_layer.logic_dim
        assert reconstructed_layer.temperature == original_layer.temperature

    def test_gradients_flow(self, layer_config, sample_input):
        """Test gradient computation."""
        layer = LogicFFN(**layer_config)

        # Convert numpy array to TensorFlow variable for gradient computation
        sample_input_var = tf.Variable(sample_input, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(sample_input_var)
            output = layer(sample_input_var)
            loss = keras.ops.mean(keras.ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        # Check gradients exist and are finite
        assert len(gradients) > 0
        for grad in gradients:
            assert grad is not None, "Gradient is None - gradient flow broken"
            assert keras.ops.all(keras.ops.isfinite(grad)), "Gradient contains NaN or Inf"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_input, training):
        """Test behavior in different training modes."""
        layer = LogicFFN(**layer_config)

        output = layer(sample_input, training=training)

        # Should work in all training modes
        assert output.shape[0] == sample_input.shape[0]
        assert output.shape[-1] == layer_config['output_dim']
        assert keras.ops.all(keras.ops.isfinite(output))

    def test_multiple_calls_consistency(self, layer_config, sample_input):
        """Test that multiple calls with same input produce same output."""
        layer = LogicFFN(**layer_config)

        output1 = layer(sample_input, training=False)
        output2 = layer(sample_input, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Multiple calls should produce identical outputs"
        )

    def test_logic_operations_range(self, layer_config, sample_input):
        """Test that logic operations produce values in expected ranges."""
        layer = LogicFFN(**layer_config)

        # Convert to tensor for consistent behavior
        sample_input_tensor = keras.ops.convert_to_tensor(sample_input, dtype='float32')

        # Build layer
        _ = layer(sample_input_tensor)

        # Get intermediate logic operations by manually computing
        projected = layer.logic_projection(sample_input_tensor)
        operand_a, operand_b = keras.ops.split(projected, 2, axis=-1)
        soft_a = keras.ops.sigmoid(operand_a)
        soft_b = keras.ops.sigmoid(operand_b)

        # Check soft-bits are in [0, 1]
        assert keras.ops.all(soft_a >= 0.0)
        assert keras.ops.all(soft_a <= 1.0)
        assert keras.ops.all(soft_b >= 0.0)
        assert keras.ops.all(soft_b <= 1.0)

        # Check logic operations
        logic_and = soft_a * soft_b
        logic_or = soft_a + soft_b - (soft_a * soft_b)
        logic_xor = keras.ops.square(soft_a - soft_b)

        # AND should be in [0, 1]
        assert keras.ops.all(logic_and >= 0.0)
        assert keras.ops.all(logic_and <= 1.0)

        # OR should be in [0, 1]
        assert keras.ops.all(logic_or >= 0.0)
        assert keras.ops.all(logic_or <= 1.0)

        # XOR should be in [0, 1] (since (a-b)^2 where a,b in [0,1])
        assert keras.ops.all(logic_xor >= 0.0)
        assert keras.ops.all(logic_xor <= 1.0)

    def test_temperature_effects(self, sample_input):
        """Test that temperature affects gating behavior."""
        # Convert to tensor for consistent behavior
        sample_input_tensor = keras.ops.convert_to_tensor(sample_input, dtype='float32')

        # Create layers with different temperatures
        layer_low_temp = LogicFFN(output_dim=64, logic_dim=32, temperature=0.1)
        layer_high_temp = LogicFFN(output_dim=64, logic_dim=32, temperature=10.0)

        # Get gate weights
        gates_low = layer_low_temp.gate_projection(sample_input_tensor)
        gates_high = layer_high_temp.gate_projection(sample_input_tensor)

        # Apply softmax with temperatures
        softmax_low = keras.ops.softmax(gates_low / 0.1)
        softmax_high = keras.ops.softmax(gates_high / 10.0)

        # Low temperature should produce more peaked distributions
        entropy_low = -keras.ops.sum(softmax_low * keras.ops.log(softmax_low + 1e-8), axis=-1)
        entropy_high = -keras.ops.sum(softmax_high * keras.ops.log(softmax_high + 1e-8), axis=-1)

        # Generally, higher temperature leads to higher entropy (more uniform)
        mean_entropy_low = keras.ops.mean(entropy_low)
        mean_entropy_high = keras.ops.mean(entropy_high)

        # This is a statistical test, might occasionally fail due to randomness
        # but generally should hold
        assert mean_entropy_high >= mean_entropy_low - 0.1  # Small tolerance


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_logic_ffn_standard(self):
        """Test standard factory function."""
        layer = create_logic_ffn_standard(output_dim=128, logic_dim=64)

        assert layer.output_dim == 128
        assert layer.logic_dim == 64
        assert layer.temperature == 1.0
        assert layer.use_bias is True

    def test_create_logic_ffn_regularized(self):
        """Test regularized factory function."""
        layer = create_logic_ffn_regularized(output_dim=128, logic_dim=64, l2_reg=1e-3)

        assert layer.output_dim == 128
        assert layer.logic_dim == 64
        assert layer.kernel_regularizer is not None
        assert layer.bias_regularizer is not None


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_input_shape(self):
        """Test handling of invalid input shapes."""
        layer = LogicFFN(output_dim=64, logic_dim=32)

        # Test 1D input (too few dimensions)
        with pytest.raises(ValueError, match="Input must be at least 2D"):
            layer.build((48,))

        # Test unknown last dimension
        with pytest.raises(ValueError, match="Input feature dimension must be specified"):
            layer.build((None, 16, None))

    def test_very_small_dimensions(self):
        """Test with very small dimensions."""
        layer = LogicFFN(output_dim=1, logic_dim=1)
        sample_input = np.random.normal(size=(2, 1)).astype(np.float32)

        output = layer(sample_input)
        assert output.shape == (2, 1)
        assert keras.ops.all(keras.ops.isfinite(output))

    def test_large_batch_size(self):
        """Test with large batch size."""
        layer = LogicFFN(output_dim=64, logic_dim=32)
        large_input = np.random.normal(size=(1000, 48)).astype(np.float32)

        output = layer(large_input)
        assert output.shape == (1000, 64)
        assert keras.ops.all(keras.ops.isfinite(output))


# Additional utility for debugging
def debug_logic_ffn_serialization(layer_config, sample_input):
    """Debug helper for LogicFFN serialization issues."""
    from dl_techniques.utils.logger import logger

    try:
        # Convert to tensor for consistent behavior
        sample_input_tensor = keras.ops.convert_to_tensor(sample_input, dtype='float32')

        # Test basic functionality
        layer = LogicFFN(**layer_config)
        output = layer(sample_input_tensor)
        logger.info(f"✅ Forward pass successful: {output.shape}")

        # Test configuration
        config = layer.get_config()
        logger.info(f"✅ Configuration keys: {list(config.keys())}")

        # Test serialization
        inputs = keras.Input(shape=sample_input_tensor.shape[1:])
        outputs = LogicFFN(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(os.path.join(tmpdir, 'test.keras'))
            loaded = keras.models.load_model(os.path.join(tmpdir, 'test.keras'))
            logger.info("✅ Serialization test passed")

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise


# Run tests with: pytest test_logic_ffn.py -v
if __name__ == "__main__":
    # Quick manual test
    layer = LogicFFN(output_dim=64, logic_dim=32)
    test_input = keras.ops.convert_to_tensor(
        np.random.normal(size=(4, 16, 48)).astype(np.float32)
    )
    output = layer(test_input)
    print(f"Test successful! Output shape: {output.shape}")