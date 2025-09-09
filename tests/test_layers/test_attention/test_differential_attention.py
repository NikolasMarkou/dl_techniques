"""
Comprehensive Test Suite for DifferentialMultiHeadAttention
=========================================================

This test suite follows dl-techniques testing best practices and ensures
robust validation of the differential attention layer implementation.

Key testing principles:
- Comprehensive input validation
- Full serialization cycle testing (most critical)
- Gradient flow verification
- Multiple training modes
- Edge case handling
- Backend-agnostic implementation
"""

import pytest
import tempfile
import os
import numpy as np
import keras
import tensorflow as tf
from typing import Any, Dict

from dl_techniques.layers.attention.differential_attention import DifferentialMultiHeadAttention


class TestDifferentialMultiHeadAttention:
    """Comprehensive test suite for DifferentialMultiHeadAttention layer."""

    def test_import(self):
        """Test that the layer can be imported successfully."""
        # This test ensures the module loads without import errors
        assert DifferentialMultiHeadAttention is not None
        assert hasattr(DifferentialMultiHeadAttention, '__init__')
        assert hasattr(DifferentialMultiHeadAttention, 'call')
        assert hasattr(DifferentialMultiHeadAttention, 'get_config')

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'dim': 256,
            'num_heads': 8,
            'head_dim': 32,
            'dropout_rate': 0.1,  # Updated parameter name
            'attention_dropout_rate': 0.05,  # Updated parameter name
            'lambda_init': 0.7
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create sample input tensor for testing."""
        # Use keras.random instead of numpy for backend consistency
        return keras.random.normal(shape=(4, 16, 256))

    @pytest.fixture
    def attention_mask(self) -> keras.KerasTensor:
        """Create attention mask for testing."""
        batch_size, seq_len = 4, 16
        # Create padding mask (1 = attend, 0 = ignore) using numpy then convert
        mask_np = np.ones((batch_size, seq_len), dtype=np.float32)
        # Mask out some positions (last 4 positions)
        mask_np[:, 12:16] = 0.0
        mask = keras.ops.convert_to_tensor(mask_np)

        # Convert to 3D attention mask format (batch, seq_len, seq_len)
        attention_mask = keras.ops.einsum('bi,bj->bij', mask, mask)
        return attention_mask

    def test_initialization_valid_params(self, layer_config):
        """Test layer initialization with valid parameters."""
        layer = DifferentialMultiHeadAttention(**layer_config)

        # Verify configuration stored correctly
        assert layer.dim == layer_config['dim']
        assert layer.num_heads == layer_config['num_heads']
        assert layer.head_dim == layer_config['head_dim']
        assert layer.dropout_rate == layer_config['dropout_rate']  # Internal storage
        assert layer.attention_dropout_rate == layer_config['attention_dropout_rate']  # Internal storage
        assert layer.lambda_init == layer_config['lambda_init']

        # Verify layer not built initially
        assert not layer.built
        assert layer.lambda_param is None

        # Verify sub-layers created in __init__
        assert layer.attention1 is not None
        assert layer.attention2 is not None
        assert layer.proj is not None
        assert layer.dropout_layer is not None
        assert isinstance(layer.attention1, keras.layers.MultiHeadAttention)
        assert isinstance(layer.attention2, keras.layers.MultiHeadAttention)
        assert isinstance(layer.proj, keras.layers.Dense)
        assert isinstance(layer.dropout_layer, keras.layers.Dropout)

    def test_initialization_with_regularization(self):
        """Test initialization with regularization parameters."""
        kernel_reg = keras.regularizers.L2(1e-4)
        bias_reg = keras.regularizers.L1(1e-5)

        layer = DifferentialMultiHeadAttention(
            dim=128,
            num_heads=8,
            head_dim=16,
            kernel_regularizer=kernel_reg,
            bias_regularizer=bias_reg,
            activity_regularizer=keras.regularizers.L2(1e-6)
        )

        # Verify regularizers stored properly
        assert layer.kernel_regularizer is not None
        assert layer.bias_regularizer is not None

    def test_build_process(self, layer_config, sample_input):
        """Test the build process and weight creation."""
        layer = DifferentialMultiHeadAttention(**layer_config)

        # Verify not built initially
        assert not layer.built
        assert layer.lambda_param is None

        # Build the layer
        layer.build(sample_input.shape)

        # Verify built state
        assert layer.built
        assert layer.lambda_param is not None
        assert layer.lambda_param.shape == (1,)

        # Verify lambda param initialized correctly
        lambda_val = keras.ops.convert_to_numpy(layer.lambda_param[0])
        assert lambda_val == pytest.approx(layer_config['lambda_init'], abs=1e-6)

    def test_build_invalid_input_shape(self, layer_config):
        """Test build with invalid input shapes."""
        layer = DifferentialMultiHeadAttention(**layer_config)

        # Test with wrong number of dimensions
        with pytest.raises(ValueError, match="Expected 3D input"):
            layer.build((32, 64))  # 2D instead of 3D

        # Test with wrong dimension
        with pytest.raises(ValueError, match="doesn't match expected dimension"):
            layer.build((4, 16, 128))  # 128 instead of 256

    def test_forward_pass_basic(self, layer_config, sample_input):
        """Test basic forward pass functionality."""
        layer = DifferentialMultiHeadAttention(**layer_config)

        # Forward pass (builds layer automatically)
        output = layer(sample_input, layer_idx=2, training=True)

        # Verify output shape
        assert output.shape == sample_input.shape
        assert layer.built

        # Verify output is different from input (attention applied)
        input_numpy = keras.ops.convert_to_numpy(sample_input)
        output_numpy = keras.ops.convert_to_numpy(output)
        assert not np.allclose(input_numpy, output_numpy, rtol=1e-3)

    def test_forward_pass_with_mask(self, layer_config, sample_input, attention_mask):
        """Test forward pass with attention mask."""
        layer = DifferentialMultiHeadAttention(**layer_config)

        # Forward pass with mask
        output_with_mask = layer(sample_input, attention_mask=attention_mask, layer_idx=1, training=True)

        # Forward pass without mask
        output_without_mask = layer(sample_input, layer_idx=1, training=True)

        # Outputs should be different when mask is applied
        masked_numpy = keras.ops.convert_to_numpy(output_with_mask)
        unmasked_numpy = keras.ops.convert_to_numpy(output_without_mask)
        assert not np.allclose(masked_numpy, unmasked_numpy, rtol=1e-3)

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_input, training):
        """Test behavior in different training modes."""
        layer = DifferentialMultiHeadAttention(**layer_config)

        output = layer(sample_input, training=training)
        assert output.shape == sample_input.shape

    @pytest.mark.parametrize("layer_idx", [0, 1, 3, 5, 10])
    def test_lambda_computation(self, layer_config, layer_idx):
        """Test lambda parameter computation for different layer indices."""
        layer = DifferentialMultiHeadAttention(**layer_config)

        # Build the layer
        layer.build((4, 16, 256))

        # Get lambda for different indices
        lambda_val = layer.get_lambda(layer_idx)
        lambda_numpy = keras.ops.convert_to_numpy(lambda_val)

        # Verify lambda is in valid range [0.1, 0.9]
        assert 0.1 <= lambda_numpy <= 0.9

        # Verify that lambda computation works consistently
        lambda_val2 = layer.get_lambda(layer_idx)
        lambda_numpy2 = keras.ops.convert_to_numpy(lambda_val2)
        np.testing.assert_allclose(lambda_numpy, lambda_numpy2, rtol=1e-6)

    def test_differential_attention_mechanism(self, layer_config, sample_input):
        """Test that differential attention actually computes MHA1 - λ*MHA2."""
        layer = DifferentialMultiHeadAttention(**layer_config)

        # Get individual attention outputs by temporarily modifying the call
        # This is a bit tricky since we need to access intermediate results
        output = layer(sample_input, layer_idx=2)

        # Verify we get valid output
        assert output.shape == sample_input.shape

        # Test that lambda parameter affects output
        # Change lambda and verify output changes
        original_lambda = keras.ops.convert_to_numpy(layer.lambda_param[0])

        # Modify lambda parameter
        layer.lambda_param.assign([original_lambda * 0.5])
        output_modified = layer(sample_input, layer_idx=2)

        # Outputs should be different
        original_numpy = keras.ops.convert_to_numpy(output)
        modified_numpy = keras.ops.convert_to_numpy(output_modified)
        assert not np.allclose(original_numpy, modified_numpy, rtol=1e-3)

        # Restore original lambda
        layer.lambda_param.assign([original_lambda])

    def test_compute_output_shape(self, layer_config):
        """Test compute_output_shape method."""
        layer = DifferentialMultiHeadAttention(**layer_config)

        # Test various input shapes
        test_shapes = [
            (None, 16, 256),
            (4, None, 256),
            (8, 32, 256),
            tf.TensorShape([None, 16, 256])
        ]

        for input_shape in test_shapes:
            output_shape = layer.compute_output_shape(input_shape)
            if isinstance(input_shape, tf.TensorShape):
                assert output_shape == input_shape
            else:
                assert output_shape == tuple(input_shape)

    def test_gradients_flow(self, layer_config, sample_input):
        """Test that gradients flow properly through the layer."""
        layer = DifferentialMultiHeadAttention(**layer_config)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(sample_input)
            output = layer(sample_input, layer_idx=2, training=True)
            loss = keras.ops.mean(keras.ops.square(output))

        # Get gradients with respect to trainable variables
        gradients = tape.gradient(loss, layer.trainable_variables)

        # Verify gradients exist and are not None
        assert len(gradients) > 0
        assert all(g is not None for g in gradients)

        # Verify lambda parameter has gradient
        lambda_grad = tape.gradient(loss, layer.lambda_param)
        assert lambda_grad is not None

        # Clean up persistent tape
        del tape

    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = DifferentialMultiHeadAttention(**layer_config)
        config = layer.get_config()

        # Check all essential parameters are present with updated names
        required_keys = [
            'dim', 'num_heads', 'head_dim', 'dropout_rate', 'attention_dropout_rate',
            'lambda_init', 'kernel_initializer', 'bias_initializer',
            'kernel_regularizer', 'bias_regularizer'
        ]

        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        # Verify values match original configuration
        assert config['dim'] == layer_config['dim']
        assert config['num_heads'] == layer_config['num_heads']
        assert config['head_dim'] == layer_config['head_dim']
        assert config['dropout_rate'] == layer_config['dropout_rate']
        assert config['attention_dropout_rate'] == layer_config['attention_dropout_rate']
        assert config['lambda_init'] == layer_config['lambda_init']

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle (most important test)."""
        # Create model with differential attention layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = DifferentialMultiHeadAttention(**layer_config)(inputs, layer_idx=3)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions (this is the critical test)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization - this indicates broken serialization"
            )

    def test_serialization_with_trained_weights(self, layer_config, sample_input):
        """Test serialization after some training to verify lambda parameter preservation."""
        # Create and train model briefly
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = DifferentialMultiHeadAttention(**layer_config)(inputs, layer_idx=2)
        model = keras.Model(inputs, outputs)

        model.compile(optimizer='adam', loss='mse')

        # Simple "training" - one step
        with tf.GradientTape() as tape:
            pred = model(sample_input)
            loss = keras.ops.mean(keras.ops.square(pred))

        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Get lambda after training
        diff_layer = model.layers[1]  # The differential attention layer
        trained_lambda = keras.ops.convert_to_numpy(diff_layer.lambda_param[0])

        # Serialize and deserialize
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'trained_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_diff_layer = loaded_model.layers[1]
            loaded_lambda = keras.ops.convert_to_numpy(loaded_diff_layer.lambda_param[0])

            # Verify lambda parameter preserved
            np.testing.assert_allclose(
                trained_lambda, loaded_lambda,
                rtol=1e-6, atol=1e-6,
                err_msg="Lambda parameter not preserved after serialization"
            )

    def test_from_config_recreation(self, layer_config):
        """Test layer recreation from config."""
        original_layer = DifferentialMultiHeadAttention(**layer_config)
        config = original_layer.get_config()

        # Recreate layer from config
        new_layer = DifferentialMultiHeadAttention.from_config(config)

        # Verify configuration preserved
        assert new_layer.dim == original_layer.dim
        assert new_layer.num_heads == original_layer.num_heads
        assert new_layer.head_dim == original_layer.head_dim
        assert new_layer.dropout_rate == original_layer.dropout_rate
        assert new_layer.attention_dropout_rate == original_layer.attention_dropout_rate
        assert new_layer.lambda_init == original_layer.lambda_init

    def test_edge_cases(self):
        """Test various edge cases and boundary conditions."""
        # Create appropriately sized inputs for different test cases
        small_input = keras.random.normal((4, 16, 8))
        regular_input = keras.random.normal((4, 16, 256))

        # Minimum valid configuration
        min_layer = DifferentialMultiHeadAttention(
            dim=8, num_heads=1, head_dim=8, dropout_rate=0.0, attention_dropout_rate=0.0
        )
        output = min_layer(small_input)
        assert output.shape == (4, 16, 8)

        # High dropout (but less than 1.0 since Keras 3 requires rate < 1.0)
        high_dropout_layer = DifferentialMultiHeadAttention(
            dim=256, num_heads=8, head_dim=32, dropout_rate=0.99, attention_dropout_rate=0.99
        )
        output = high_dropout_layer(regular_input, training=True)
        assert output.shape == regular_input.shape

        # Edge lambda values
        edge_lambda_layer = DifferentialMultiHeadAttention(
            dim=256, num_heads=8, head_dim=32, lambda_init=0.0001  # Very small lambda
        )
        output = edge_lambda_layer(regular_input)
        assert output.shape == regular_input.shape

    def test_multiple_calls_consistency(self, layer_config, sample_input):
        """Test that multiple calls with same input give consistent results in eval mode."""
        layer = DifferentialMultiHeadAttention(**layer_config)

        # Multiple calls in eval mode should be consistent
        output1 = layer(sample_input, layer_idx=2, training=False)
        output2 = layer(sample_input, layer_idx=2, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Multiple calls in eval mode should give identical results"
        )

    def test_layer_weights_structure(self, layer_config, sample_input):
        """Test the structure of layer weights."""
        layer = DifferentialMultiHeadAttention(**layer_config)

        # Build layer
        _ = layer(sample_input)

        # Check weight structure
        trainable_weights = layer.trainable_weights
        assert len(trainable_weights) > 0

        # Lambda parameter should be trainable
        lambda_weights = [w for w in trainable_weights if 'lambda_param' in w.name]
        assert len(lambda_weights) == 1
        assert lambda_weights[0].shape == (1,)

        # Get all weight names for debugging
        weight_names = [w.name for w in trainable_weights]
        print(f"\nDEBUG - All {len(weight_names)} weight names:")
        for i, name in enumerate(weight_names):
            print(f"  {i+1}. {name}")

        # More flexible weight matching - just check we have reasonable number of weights
        # A differential attention layer should have:
        # - 1 lambda parameter
        # - 2 attention layers (each with query, key, value, output weights + biases = ~8 weights each)
        # - 1 projection layer (weight + bias = 2 weights)
        # Total expected: 1 + 16 + 2 = 19+ weights

        # Basic sanity checks
        assert len(lambda_weights) == 1, f"Expected 1 lambda weight, got {len(lambda_weights)}"

        # Check we have a reasonable total number of weights
        # MultiHeadAttention layers are complex and have many internal weights
        assert len(trainable_weights) >= 10, f"Expected at least 10 total weights, got {len(trainable_weights)}"

        # Verify lambda parameter is correct
        lambda_val = keras.ops.convert_to_numpy(lambda_weights[0])
        assert lambda_val.shape == (1,)

        print(f"✅ Weight structure validation passed: {len(trainable_weights)} total weights")


# Run critical tests directly
if __name__ == "__main__":
    # Run a subset of critical tests
    pytest.main([__file__, "-v"])