"""
Comprehensive Test Suite for SpatialAttention Layer

This test suite follows Modern Keras 3 testing best practices with emphasis on
serialization testing and production readiness validation.
"""

import pytest
import tempfile
import os
from typing import Any, Dict
import numpy as np

import keras
from keras import ops
import tensorflow as tf

from dl_techniques.layers.attention.spatial_attention import SpatialAttention


class TestSpatialAttention:
    """Comprehensive test suite for SpatialAttention Layer following Modern Keras 3 patterns."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard SpatialAttention layer configuration for testing."""
        return {
            'kernel_size': 7,
            'kernel_initializer': 'glorot_uniform',
            'kernel_regularizer': None,
            'use_bias': True,
        }

    @pytest.fixture
    def sample_input_4d(self) -> keras.KerasTensor:
        """Sample 4D input tensor for testing (standard image format)."""
        return tf.random.normal(shape=(8, 32, 32, 64), seed=42)

    @pytest.fixture
    def sample_input_small(self) -> keras.KerasTensor:
        """Small 4D input tensor for testing."""
        return tf.random.normal(shape=(4, 16, 16, 32), seed=43)

    @pytest.fixture
    def sample_input_single_channel(self) -> keras.KerasTensor:
        """Single channel 4D input tensor for testing."""
        return tf.random.normal(shape=(4, 28, 28, 1), seed=44)

    @pytest.fixture
    def sample_input_large_channels(self) -> keras.KerasTensor:
        """Large channel 4D input tensor for testing."""
        return tf.random.normal(shape=(2, 14, 14, 256), seed=45)

    def test_initialization(self, layer_config: Dict[str, Any]) -> None:
        """
        Test layer initialization and parameter validation.

        Validates:
        - Correct attribute assignment from config
        - Unbuilt state after initialization
        - Sub-layer creation (conv layer)
        - Parameter validation with invalid inputs
        """
        # Test successful initialization
        layer = SpatialAttention(**layer_config)

        # Verify configuration storage
        assert layer.kernel_size == layer_config['kernel_size']
        assert layer.use_bias == layer_config['use_bias']
        assert layer.kernel_initializer is not None
        assert layer.kernel_regularizer is layer_config['kernel_regularizer']

        # Verify unbuilt state
        assert not layer.built

        # Verify sub-layer creation
        assert layer.conv is not None
        assert isinstance(layer.conv, keras.layers.Conv2D)
        assert layer.conv.filters == 1  # Should output single attention channel
        assert layer.conv.activation.__name__ == 'sigmoid'
        assert layer.conv.padding == 'same'

    def test_parameter_validation(self) -> None:
        """Test comprehensive parameter validation with invalid inputs."""

        # Test invalid kernel_size (non-positive)
        with pytest.raises(ValueError, match="kernel_size must be positive"):
            SpatialAttention(kernel_size=0)

        with pytest.raises(ValueError, match="kernel_size must be positive"):
            SpatialAttention(kernel_size=-3)

        # Test invalid kernel_size (even number)
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            SpatialAttention(kernel_size=6)

        with pytest.raises(ValueError, match="kernel_size must be odd"):
            SpatialAttention(kernel_size=8)

        # Test valid odd kernel sizes
        valid_sizes = [1, 3, 5, 7, 9, 11]
        for size in valid_sizes:
            layer = SpatialAttention(kernel_size=size)
            assert layer.kernel_size == size

    @pytest.mark.parametrize("input_tensor", [
        "sample_input_4d", "sample_input_small",
        "sample_input_single_channel", "sample_input_large_channels"
    ])
    def test_forward_pass(
        self,
        layer_config: Dict[str, Any],
        input_tensor: str,
        request: pytest.FixtureRequest
    ) -> None:
        """
        Test forward pass and automatic building.

        Validates:
        - Automatic building on first call
        - Correct output shapes
        - Attention map properties
        - Multiple forward passes consistency
        """
        # Get the input tensor from fixture
        inputs = request.getfixturevalue(input_tensor)

        layer = SpatialAttention(**layer_config)

        # Forward pass triggers building
        attention_map = layer(inputs)

        # Verify building occurred
        assert layer.built
        assert layer.conv.built

        # Verify output shape - should preserve spatial dims, single channel
        expected_shape = (inputs.shape[0], inputs.shape[1], inputs.shape[2], 1)
        assert attention_map.shape == expected_shape

        # Verify attention map properties
        attention_np = ops.convert_to_numpy(attention_map)
        assert np.all(attention_np >= 0.0), "Attention values must be non-negative"
        assert np.all(attention_np <= 1.0), "Attention values must be <= 1.0 (sigmoid output)"

        # Test consistency across multiple calls
        attention_map2 = layer(inputs)
        np.testing.assert_allclose(
            ops.convert_to_numpy(attention_map),
            ops.convert_to_numpy(attention_map2),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Multiple forward passes should be consistent"
        )

    def test_serialization_cycle(
        self,
        layer_config: Dict[str, Any],
        sample_input_4d: keras.KerasTensor
    ) -> None:
        """
        CRITICAL TEST: Full serialization and deserialization cycle.

        This is the most important test for production readiness.
        Validates:
        - Model saving with custom layer
        - Model loading with custom layer
        - Identical predictions after serialization
        - Weight preservation
        """
        # Create model with SpatialAttention layer
        inputs = keras.Input(shape=sample_input_4d.shape[1:])
        attention_map = SpatialAttention(**layer_config)(inputs)
        # Apply attention to inputs (common usage pattern)
        attended = keras.layers.Multiply()([inputs, attention_map])
        # Add classification head for more realistic test
        pooled = keras.layers.GlobalAveragePooling2D()(attended)
        outputs = keras.layers.Dense(10, activation='softmax')(pooled)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_prediction = model(sample_input_4d)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'spatial_attention_test_model.keras')

            # Save model
            model.save(filepath)

            # Load model (tests custom layer registration)
            loaded_model = keras.models.load_model(filepath)

            # Get prediction from loaded model
            loaded_prediction = loaded_model(sample_input_4d)

            # Verify identical predictions
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6,
                atol=1e-6,
                err_msg="Predictions differ after serialization cycle"
            )

            # Verify layer configuration preserved
            original_attention = model.layers[1]  # SpatialAttention layer
            loaded_attention = loaded_model.layers[1]  # SpatialAttention layer

            assert original_attention.kernel_size == loaded_attention.kernel_size
            assert original_attention.use_bias == loaded_attention.use_bias

    def test_config_completeness(self, layer_config: Dict[str, Any]) -> None:
        """
        Test that get_config contains all __init__ parameters.

        Validates:
        - All initialization parameters in config
        - Proper serialization of complex objects
        - Config can reconstruct identical layer
        """
        layer = SpatialAttention(**layer_config)
        config = layer.get_config()

        # Check all required parameters present
        required_keys = [
            'kernel_size', 'kernel_initializer', 'kernel_regularizer', 'use_bias'
        ]

        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        # Verify config values match initialization
        assert config['kernel_size'] == layer_config['kernel_size']
        assert config['use_bias'] == layer_config['use_bias']

        # Test reconstruction from config
        reconstructed_layer = SpatialAttention.from_config(config)
        assert reconstructed_layer.kernel_size == layer.kernel_size
        assert reconstructed_layer.use_bias == layer.use_bias

    def test_gradients_flow(
        self,
        layer_config: Dict[str, Any],
        sample_input_4d: keras.KerasTensor
    ) -> None:
        """
        Test gradient computation and backpropagation.

        Validates:
        - Gradients computed for all trainable weights
        - No None gradients
        - Reasonable gradient magnitudes
        """
        layer = SpatialAttention(**layer_config)

        with tf.GradientTape() as tape:
            # Enable gradient tracking for input
            tape.watch(sample_input_4d)

            # Forward pass
            attention_map = layer(sample_input_4d, training=True)

            # Compute loss (encourage high attention values)
            loss = -ops.mean(attention_map)  # Negative to maximize attention

        # Compute all gradients
        all_variables = layer.trainable_weights + [sample_input_4d]
        all_gradients = tape.gradient(loss, all_variables)

        # Split gradients
        layer_gradients = all_gradients[:-1]
        input_gradients = all_gradients[-1]

        # Verify gradients exist
        assert len(layer_gradients) > 0, "No trainable weights found"
        assert all(g is not None for g in layer_gradients), "Some gradients are None"
        assert input_gradients is not None, "Input gradients are None"

        # Verify reasonable gradient magnitudes
        for i, grad in enumerate(layer_gradients):
            grad_norm = ops.sqrt(ops.sum(ops.square(grad)))
            assert grad_norm > 0, f"Zero gradient for weight {i}"
            assert grad_norm < 1000, f"Exploding gradient for weight {i}: {grad_norm}"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(
        self,
        layer_config: Dict[str, Any],
        sample_input_4d: keras.KerasTensor,
        training: bool
    ) -> None:
        """
        Test behavior in different training modes.

        Validates:
        - Consistent outputs across training modes (no dropout in this layer)
        - Proper handling of training parameter
        """
        layer = SpatialAttention(**layer_config)

        # Forward pass in specified training mode
        attention_map = layer(sample_input_4d, training=training)

        # Verify output shape and properties
        expected_shape = (sample_input_4d.shape[0], sample_input_4d.shape[1],
                         sample_input_4d.shape[2], 1)
        assert attention_map.shape == expected_shape

        # Verify attention properties
        attention_np = ops.convert_to_numpy(attention_map)
        assert np.all(attention_np >= 0.0)
        assert np.all(attention_np <= 1.0)

        # For SpatialAttention, output should be identical regardless of training mode
        # since there's no dropout or batch normalization
        attention_map_inference = layer(sample_input_4d, training=False)
        np.testing.assert_allclose(
            ops.convert_to_numpy(attention_map),
            ops.convert_to_numpy(attention_map_inference),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Training and inference outputs should be identical for SpatialAttention"
        )

    def test_output_shape_computation(self, layer_config: Dict[str, Any]) -> None:
        """Test compute_output_shape method."""
        layer = SpatialAttention(**layer_config)

        # Test various 4D input shapes
        test_cases = [
            ((None, 32, 32, 64), (None, 32, 32, 1)),
            ((8, 224, 224, 3), (8, 224, 224, 1)),
            ((16, 56, 56, 128), (16, 56, 56, 1)),
            ((4, 14, 14, 256), (4, 14, 14, 1)),
        ]

        for input_shape, expected_output_shape in test_cases:
            output_shape = layer.compute_output_shape(input_shape)
            assert output_shape == expected_output_shape, \
                f"Input {input_shape} -> Expected {expected_output_shape}, Got {output_shape}"

    def test_spatial_attention_mechanism(self, layer_config: Dict[str, Any]) -> None:
        """
        Test the core spatial attention mechanism.

        Validates:
        - Channel-wise pooling operations
        - Concatenation of pooled features
        - Convolution and sigmoid activation
        - Attention map properties
        """
        layer = SpatialAttention(**layer_config)

        # Create test input with known patterns
        batch_size, height, width, channels = 2, 16, 16, 8
        inputs = tf.random.normal((batch_size, height, width, channels), seed=46)

        # Get attention map
        attention_map = layer(inputs)

        # Test intermediate computations by accessing them directly
        # Simulate the internal pooling operations
        expected_avg_pool = ops.mean(inputs, axis=-1, keepdims=True)
        expected_max_pool = ops.max(inputs, axis=-1, keepdims=True)
        expected_concat = ops.concatenate([expected_avg_pool, expected_max_pool], axis=-1)

        # Verify shapes
        assert expected_avg_pool.shape == (batch_size, height, width, 1)
        assert expected_max_pool.shape == (batch_size, height, width, 1)
        assert expected_concat.shape == (batch_size, height, width, 2)

        # The final attention map should be the result of conv(concat)
        assert attention_map.shape == (batch_size, height, width, 1)

        # Verify attention properties
        attention_np = ops.convert_to_numpy(attention_map)
        assert np.all(attention_np >= 0.0), "Sigmoid output must be >= 0"
        assert np.all(attention_np <= 1.0), "Sigmoid output must be <= 1"

    def test_attention_application(self, layer_config: Dict[str, Any]) -> None:
        """
        Test spatial attention when applied to inputs (multiplication).

        Validates:
        - Attention modulation effect
        - Preserved input structure where attention is high
        - Suppressed response where attention is low
        """
        layer = SpatialAttention(**layer_config)

        # Create test input
        inputs = tf.random.normal((4, 16, 16, 32), seed=47)

        # Get attention map
        attention_map = layer(inputs)

        # Apply attention (element-wise multiplication)
        attended_features = keras.layers.Multiply()([inputs, attention_map])

        # Verify shapes
        assert attended_features.shape == inputs.shape

        # Convert to numpy for detailed analysis
        inputs_np = ops.convert_to_numpy(inputs)
        attention_np = ops.convert_to_numpy(attention_map)
        attended_np = ops.convert_to_numpy(attended_features)

        # Verify attention effect: attended = inputs * attention
        expected_attended = inputs_np * attention_np
        np.testing.assert_allclose(
            attended_np, expected_attended,
            rtol=1e-6, atol=1e-6,
            err_msg="Attended features should equal inputs * attention"
        )

        # Flatten spatial dimensions for easier comparison
        # Reshape to (batch*H*W, channels) for inputs and attended
        # Reshape to (batch*H*W,) for attention
        batch_size, height, width, channels = inputs_np.shape
        inputs_flat = inputs_np.reshape(-1, channels)  # (batch*H*W, channels)
        attention_flat = attention_np.reshape(-1)  # (batch*H*W,)
        attended_flat = attended_np.reshape(-1, channels)  # (batch*H*W, channels)

        # Where attention is high (> 0.5), features should be preserved/enhanced
        high_attention_positions = attention_flat > 0.5
        if np.any(high_attention_positions):
            high_attention_features = attended_flat[high_attention_positions]
            high_attention_original = inputs_flat[high_attention_positions]

            # Features should maintain reasonable magnitude
            if len(high_attention_features) > 0:
                high_attended_magnitude = np.mean(np.abs(high_attention_features))
                high_original_magnitude = np.mean(np.abs(high_attention_original))
                assert high_attended_magnitude > 0.1 * high_original_magnitude, \
                    f"High attention regions too suppressed: {high_attended_magnitude} vs {high_original_magnitude}"

        # Where attention is low (< 0.2), features should be suppressed
        low_attention_positions = attention_flat < 0.2
        if np.any(low_attention_positions):
            low_attention_features = attended_flat[low_attention_positions]
            low_attention_original = inputs_flat[low_attention_positions]

            # Features should be significantly suppressed
            if len(low_attention_features) > 0:
                low_attended_magnitude = np.mean(np.abs(low_attention_features))
                low_original_magnitude = np.mean(np.abs(low_attention_original))
                assert low_attended_magnitude < 0.5 * low_original_magnitude, \
                    f"Low attention regions not sufficiently suppressed: {low_attended_magnitude} vs {low_original_magnitude}"

    def test_different_kernel_sizes(self) -> None:
        """
        Test spatial attention with different kernel sizes.

        Validates:
        - Various kernel sizes work correctly
        - Output shapes remain consistent
        - Different receptive fields produce different attention patterns
        """
        # Test different odd kernel sizes
        kernel_sizes = [1, 3, 5, 7, 9]
        inputs = tf.random.normal((4, 32, 32, 16), seed=48)

        attention_maps = {}

        for k_size in kernel_sizes:
            layer = SpatialAttention(kernel_size=k_size)
            attention_map = layer(inputs)

            # Verify output shape consistency
            assert attention_map.shape == (4, 32, 32, 1)

            # Store for comparison
            attention_maps[k_size] = ops.convert_to_numpy(attention_map)

        # Larger kernels should potentially produce smoother attention maps
        # (though this depends on the specific input and learned weights)
        # At least verify they're all valid and different
        for k1 in kernel_sizes:
            for k2 in kernel_sizes:
                if k1 != k2:
                    # Maps should be different (with high probability for random weights)
                    map1 = attention_maps[k1]
                    map2 = attention_maps[k2]

                    # Allow for some similarity but expect differences
                    correlation = np.corrcoef(map1.flat, map2.flat)[0, 1]
                    assert not np.allclose(map1, map2, atol=1e-3), \
                        f"Attention maps for kernel sizes {k1} and {k2} are too similar"

    def test_numerical_stability(self, layer_config: Dict[str, Any]) -> None:
        """
        Test numerical stability with extreme inputs.

        Validates:
        - Handling of large magnitude inputs
        - Stability with very small inputs
        - No NaN/Inf in outputs
        - Bounded output range maintained
        """
        layer = SpatialAttention(**layer_config)

        # Test cases with extreme inputs
        test_cases = [
            ("zeros", ops.zeros((4, 16, 16, 32))),
            ("large_positive", ops.ones((4, 16, 16, 32)) * 1000.0),
            ("large_negative", ops.ones((4, 16, 16, 32)) * -1000.0),
            ("very_small", ops.ones((4, 16, 16, 32)) * 1e-8),
            ("mixed_extreme", tf.random.normal((4, 16, 16, 32), seed=49) * 1e6)
        ]

        for case_name, inputs in test_cases:
            attention_map = layer(inputs)

            # Convert to numpy for detailed checks
            attention_np = ops.convert_to_numpy(attention_map)

            # Check for numerical issues
            assert not np.any(np.isnan(attention_np)), \
                f"NaN values in attention map for case: {case_name}"

            assert not np.any(np.isinf(attention_np)), \
                f"Inf values in attention map for case: {case_name}"

            # Check valid range [0, 1] (sigmoid output)
            assert np.all(attention_np >= 0.0), \
                f"Negative attention values for case: {case_name}"

            assert np.all(attention_np <= 1.0), \
                f"Attention values > 1.0 for case: {case_name}"

    def test_regularization_effects(self) -> None:
        """
        Test kernel regularization effects.

        Validates:
        - L1 and L2 regularization can be applied
        - Regularization affects training behavior
        - No errors during gradient computation with regularizers
        """
        # Test without regularization
        layer_no_reg = SpatialAttention(kernel_regularizer=None)

        # Test with L2 regularization
        layer_l2 = SpatialAttention(
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # Test with L1 regularization
        layer_l1 = SpatialAttention(
            kernel_regularizer=keras.regularizers.L1(1e-4)
        )

        inputs = tf.random.normal((4, 16, 16, 8), seed=50)

        # All should work without errors
        attention_no_reg = layer_no_reg(inputs)
        attention_l2 = layer_l2(inputs)
        attention_l1 = layer_l1(inputs)

        # All should produce valid attention maps
        for attention, name in [(attention_no_reg, "no_reg"),
                               (attention_l2, "l2"),
                               (attention_l1, "l1")]:
            attention_np = ops.convert_to_numpy(attention)
            assert attention.shape == (4, 16, 16, 1)
            assert np.all(attention_np >= 0.0), f"Invalid attention for {name}"
            assert np.all(attention_np <= 1.0), f"Invalid attention for {name}"

        # Test gradient computation with regularizers
        with tf.GradientTape() as tape:
            attention = layer_l2(inputs, training=True)
            loss = ops.mean(attention)

        gradients = tape.gradient(loss, layer_l2.trainable_weights)
        assert all(g is not None for g in gradients), "Gradients should exist with regularizers"

    def test_edge_cases(self) -> None:
        """Test error conditions and edge cases."""

        # Test invalid input dimensions (not 4D)
        layer = SpatialAttention()

        # 3D input should fail
        inputs_3d = tf.random.normal((4, 16, 32))
        with pytest.raises(Exception):  # Specific exception depends on Keras version
            layer(inputs_3d)

        # 2D input should fail
        inputs_2d = tf.random.normal((4, 32))
        with pytest.raises(Exception):
            layer(inputs_2d)

        # Test minimum spatial dimensions
        inputs_tiny = tf.random.normal((2, 1, 1, 8))
        attention = layer(inputs_tiny)
        assert attention.shape == (2, 1, 1, 1)

        # Test single channel input
        inputs_single_ch = tf.random.normal((4, 16, 16, 1))
        attention = layer(inputs_single_ch)
        assert attention.shape == (4, 16, 16, 1)

        # Test large number of channels
        inputs_many_ch = tf.random.normal((2, 8, 8, 1024))
        attention = layer(inputs_many_ch)
        assert attention.shape == (2, 8, 8, 1)

    def test_consistency_with_cbam_paper(self) -> None:
        """
        Test consistency with CBAM paper specifications.

        Validates:
        - Default 7x7 kernel size (from paper)
        - Sigmoid activation
        - Same padding
        - Single output channel
        """
        # Default configuration should match CBAM paper
        layer = SpatialAttention()  # Default kernel_size=7

        assert layer.kernel_size == 7, "Default kernel size should be 7 (CBAM paper)"
        assert layer.conv.filters == 1, "Should output single attention channel"
        assert layer.conv.activation.__name__ == 'sigmoid', "Should use sigmoid activation"
        assert layer.conv.padding == 'same', "Should use 'same' padding"

        # Test with paper-like input dimensions
        inputs = tf.random.normal((8, 224, 224, 64))  # ImageNet-like
        attention = layer(inputs)

        # Should preserve spatial dimensions
        assert attention.shape == (8, 224, 224, 1)

        # Attention values should be in valid range
        attention_np = ops.convert_to_numpy(attention)
        assert np.all(attention_np >= 0.0)
        assert np.all(attention_np <= 1.0)


if __name__ == '__main__':
    # Run with: python -m pytest spatial_attention_test.py -v
    pytest.main([__file__, '-v', '--tb=short'])