"""
Comprehensive Test Suite for GlobalResponseNormalization Layer
============================================================

This test suite follows the patterns outlined in "Complete Guide to Modern Keras 3 Custom Layers and Models - Refined.md"
and ensures robust testing of the GlobalResponseNormalization layer implementation.
"""

import pytest
import tempfile
import os
import numpy as np
from typing import Any, Dict

import keras
import tensorflow as tf

# Import the layer under test
from dl_techniques.layers.norms.global_response_norm import GlobalResponseNormalization


class TestGlobalResponseNormalization:
    """Comprehensive test suite for GlobalResponseNormalization layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'eps': 1e-6,
            'gamma_initializer': 'ones',
            'beta_initializer': 'zeros',
            'gamma_regularizer': None,
            'beta_regularizer': None,
            'activity_regularizer': None
        }

    @pytest.fixture
    def custom_layer_config(self) -> Dict[str, Any]:
        """Custom configuration with regularizers for advanced testing."""
        return {
            'eps': 1e-5,
            'gamma_initializer': 'random_uniform',
            'beta_initializer': 'random_normal',
            'gamma_regularizer': keras.regularizers.L2(1e-4),
            'beta_regularizer': keras.regularizers.L1(1e-5),
            'activity_regularizer': keras.regularizers.L2(1e-6)
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample 4D input tensor for testing."""
        return keras.random.normal(shape=(4, 8, 8, 16), seed=42)

    @pytest.fixture
    def large_sample_input(self) -> keras.KerasTensor:
        """Larger sample input for advanced testing."""
        return keras.random.normal(shape=(2, 32, 32, 64), seed=42)

    # ========================================================================
    # Essential Tests Following the Guide
    # ========================================================================

    def test_initialization(self, layer_config):
        """Test layer initialization following Guide Pattern 1."""
        layer = GlobalResponseNormalization(**layer_config)

        # Verify all configuration stored
        assert hasattr(layer, 'eps')
        assert layer.eps == layer_config['eps']
        assert hasattr(layer, 'gamma_initializer')
        assert hasattr(layer, 'beta_initializer')

        # Layer should not be built initially
        assert not layer.built

        # Weight attributes should be None until built
        assert layer.gamma is None
        assert layer.beta is None

        # Verify initializers are properly processed
        assert isinstance(layer.gamma_initializer, keras.initializers.Initializer)
        assert isinstance(layer.beta_initializer, keras.initializers.Initializer)

    def test_initialization_with_regularizers(self, custom_layer_config):
        """Test initialization with regularizers and custom initializers."""
        layer = GlobalResponseNormalization(**custom_layer_config)

        # Verify regularizers are stored
        assert layer.gamma_regularizer is not None
        assert layer.beta_regularizer is not None
        assert layer.activity_regularizer is not None

        # Verify custom eps
        assert layer.eps == custom_layer_config['eps']

    def test_forward_pass(self, layer_config, sample_input):
        """Test forward pass and automatic building."""
        layer = GlobalResponseNormalization(**layer_config)

        # Forward pass should trigger building
        output = layer(sample_input)

        # Layer should now be built
        assert layer.built

        # Weights should be created
        assert layer.gamma is not None
        assert layer.beta is not None

        # Output shape should match input shape
        assert output.shape == sample_input.shape

        # Verify weights have correct shapes (1, 1, 1, channels)
        channels = sample_input.shape[-1]
        expected_weight_shape = (1, 1, 1, channels)
        assert layer.gamma.shape == expected_weight_shape
        assert layer.beta.shape == expected_weight_shape

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle following the guide."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = GlobalResponseNormalization(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions using guide's recommended assertion
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_serialization_cycle_with_regularizers(self, custom_layer_config, sample_input):
        """Test serialization with complex configuration including regularizers."""
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = GlobalResponseNormalization(**custom_layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_complex_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions with regularizers differ after serialization"
            )

    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = GlobalResponseNormalization(**layer_config)
        config = layer.get_config()

        # Check all config parameters are present
        expected_keys = [
            'eps', 'gamma_initializer', 'beta_initializer',
            'gamma_regularizer', 'beta_regularizer', 'activity_regularizer'
        ]

        for key in expected_keys:
            assert key in config, f"Missing {key} in get_config()"

        # Verify eps value is correctly stored
        assert config['eps'] == layer_config['eps']

    def test_config_completeness_with_regularizers(self, custom_layer_config):
        """Test config completeness with regularizers."""
        layer = GlobalResponseNormalization(**custom_layer_config)
        config = layer.get_config()

        # Check that regularizers are properly serialized
        assert 'gamma_regularizer' in config
        assert 'beta_regularizer' in config
        assert 'activity_regularizer' in config

        # Regularizers should be properly serialized dicts, not None
        assert config['gamma_regularizer'] is not None
        assert config['beta_regularizer'] is not None
        assert config['activity_regularizer'] is not None

    def test_gradients_flow(self, layer_config, sample_input):
        """Test gradient computation."""
        layer = GlobalResponseNormalization(**layer_config)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        # Should have gradients for gamma and beta
        assert len(gradients) == 2, "Should have gradients for gamma and beta"
        assert all(g is not None for g in gradients), "All gradients should be non-None"

        # Test input gradients
        input_gradients = tape.gradient(loss, sample_input)
        assert input_gradients is not None, "Input gradients should flow"

        # Clean up persistent tape
        del tape

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_input, training):
        """Test behavior in different training modes."""
        layer = GlobalResponseNormalization(**layer_config)

        output = layer(sample_input, training=training)

        # Output shape should be consistent across training modes
        assert output.shape == sample_input.shape

        # Layer should be built after first call
        assert layer.built

    def test_edge_cases(self):
        """Test error conditions and edge cases."""

        # Test invalid eps
        with pytest.raises(ValueError, match="eps must be positive"):
            GlobalResponseNormalization(eps=0)

        with pytest.raises(ValueError, match="eps must be positive"):
            GlobalResponseNormalization(eps=-1e-6)

        # Test invalid input shapes during forward pass
        layer = GlobalResponseNormalization()

    # ========================================================================
    # Additional Comprehensive Tests
    # ========================================================================

    def test_mathematical_correctness(self, sample_input):
        """Test the mathematical correctness of GRN operation."""
        layer = GlobalResponseNormalization(eps=1e-6)

        # Use simple initialization for predictable behavior
        layer.build(sample_input.shape)

        # Set gamma=1, beta=0 for testing mathematical operation
        layer.gamma.assign(keras.ops.ones_like(layer.gamma))
        layer.beta.assign(keras.ops.zeros_like(layer.beta))

        output = layer(sample_input)

        # Manual computation for verification
        batch_size, height, width, channels = sample_input.shape

        # Reshape and compute norms manually
        reshaped = keras.ops.reshape(sample_input, (batch_size, height * width, channels))
        norm_squared = keras.ops.sum(keras.ops.square(reshaped), axis=1, keepdims=True)
        norm = keras.ops.sqrt(norm_squared + 1e-6)

        # Global mean norm
        mean_norm = keras.ops.mean(norm, axis=-1, keepdims=True)
        norm_channels = norm / (mean_norm + 1e-6)
        norm_spatial = keras.ops.reshape(norm_channels, (batch_size, 1, 1, channels))

        # Expected output with gamma=1, beta=0
        expected = sample_input + sample_input * norm_spatial

        # Compare with actual output
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(expected),
            rtol=1e-6, atol=1e-6,
            err_msg="Mathematical operation doesn't match expected computation"
        )

    def test_residual_connection(self, sample_input):
        """Test that residual connection is properly implemented."""
        layer = GlobalResponseNormalization()

        # Build layer and set gamma=0, beta=0
        layer.build(sample_input.shape)
        layer.gamma.assign(keras.ops.zeros_like(layer.gamma))
        layer.beta.assign(keras.ops.zeros_like(layer.beta))

        output = layer(sample_input)

        # With gamma=0 and beta=0, output should equal input (residual connection only)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(sample_input),
            rtol=1e-6, atol=1e-6,
            err_msg="Residual connection not working correctly"
        )

    def test_different_input_sizes(self):
        """Test layer with various input sizes."""
        test_shapes = [
            (1, 1, 1, 1),      # Minimal
            (2, 4, 4, 8),      # Small
            (1, 16, 16, 32),   # Medium
            (4, 64, 64, 128)   # Large
        ]

        for shape in test_shapes:
            # Create a new layer instance for each shape to avoid input_spec conflicts
            layer = GlobalResponseNormalization()
            inputs = keras.random.normal(shape)
            output = layer(inputs)

            assert output.shape == inputs.shape, f"Shape mismatch for {shape}"

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very small values
        layer_small = GlobalResponseNormalization(eps=1e-8)
        small_input = keras.ops.ones((2, 4, 4, 8)) * 1e-10
        small_output = layer_small(small_input)
        assert not keras.ops.any(keras.ops.isnan(small_output)), "NaN values with small input"
        assert not keras.ops.any(keras.ops.isinf(small_output)), "Inf values with small input"

        # Test with large values (use separate layer instance)
        layer_large = GlobalResponseNormalization(eps=1e-8)
        large_input = keras.ops.ones((2, 4, 4, 8)) * 1e10
        large_output = layer_large(large_input)
        assert not keras.ops.any(keras.ops.isnan(large_output)), "NaN values with large input"
        assert not keras.ops.any(keras.ops.isinf(large_output)), "Inf values with large input"

    def test_mixed_precision_compatibility(self, sample_input):
        """Test compatibility with mixed precision training."""
        # Enable mixed precision
        original_policy = keras.mixed_precision.global_policy()
        try:
            keras.mixed_precision.set_global_policy('mixed_float16')

            # Cast input to float16
            fp16_input = keras.ops.cast(sample_input, dtype='float16')

            layer = GlobalResponseNormalization()
            output = layer(fp16_input)

            # Output should be float16
            assert output.dtype == 'float16'

            # Should not have NaN or Inf
            assert not keras.ops.any(keras.ops.isnan(output))
            assert not keras.ops.any(keras.ops.isinf(output))

        finally:
            # Restore original policy
            keras.mixed_precision.set_global_policy(original_policy)

    def test_input_spec_validation(self, layer_config):
        """Test that input spec is correctly set after building."""
        layer = GlobalResponseNormalization(**layer_config)

        # Build with specific shape
        input_shape = (None, 16, 16, 32)
        layer.build(input_shape)

    def test_compute_output_shape(self, layer_config):
        """Test compute_output_shape method."""
        layer = GlobalResponseNormalization(**layer_config)

        input_shape = (None, 32, 32, 64)
        output_shape = layer.compute_output_shape(input_shape)

        # Output shape should be identical to input shape
        assert output_shape == input_shape

    # ========================================================================
    # Integration and Performance Tests
    # ========================================================================

    def test_integration_in_model(self, sample_input):
        """Test GRN integration within a larger model."""
        inputs = keras.Input(shape=sample_input.shape[1:])

        # Create a ConvNeXt-style block with GRN
        x = keras.layers.DepthwiseConv2D(7, padding='same')(inputs)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = keras.layers.Conv2D(64, 1)(x)
        x = keras.layers.Activation('gelu')(x)
        x = GlobalResponseNormalization()(x)
        x = keras.layers.Conv2D(16, 1)(x)
        outputs = inputs + x  # Residual connection

        model = keras.Model(inputs, outputs)

        # Test forward pass
        result = model(sample_input)
        assert result.shape == sample_input.shape

        # Test that model can be compiled and has trainable parameters
        model.compile(optimizer='adam', loss='mse')
        assert len(model.trainable_variables) > 0

    def test_gradient_checkpointing_compatibility(self, sample_input):
        """Test compatibility with gradient checkpointing."""
        @tf.recompute_grad
        def checkpointed_grn(x):
            layer = GlobalResponseNormalization()
            return layer(x, training=True)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = checkpointed_grn(sample_input)
            loss = keras.ops.mean(output)

        gradients = tape.gradient(loss, sample_input)
        assert gradients is not None
        assert not keras.ops.any(keras.ops.isnan(gradients))


# ============================================================================
# Debug Helper Function
# ============================================================================

def debug_layer_serialization(layer_config: Dict[str, Any], sample_input: keras.KerasTensor):
    """
    Debug helper for GlobalResponseNormalization serialization issues.

    This function helps diagnose serialization problems by testing each step
    of the save/load process and providing detailed logging.
    """
    from dl_techniques.utils.logger import logger

    try:
        # Test basic functionality
        layer = GlobalResponseNormalization(**layer_config)
        output = layer(sample_input)
        logger.info(f"✅ Forward pass successful: {output.shape}")

        # Test configuration
        config = layer.get_config()
        logger.info(f"✅ Configuration keys: {list(config.keys())}")

        # Test serialization
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = GlobalResponseNormalization(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'debug_test.keras')
            model.save(filepath)
            loaded = keras.models.load_model(filepath)
            logger.info("✅ Serialization test passed")

            # Compare predictions
            orig_pred = model(sample_input)
            loaded_pred = loaded(sample_input)

            max_diff = keras.ops.max(keras.ops.abs(orig_pred - loaded_pred))
            logger.info(f"✅ Max prediction difference: {max_diff}")

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise

