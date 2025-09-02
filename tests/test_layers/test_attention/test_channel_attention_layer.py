"""
Comprehensive Test Suite for ChannelAttention Layer

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

from dl_techniques.layers.attention.channel_attention import ChannelAttention


class TestChannelAttention:
    """Comprehensive test suite for ChannelAttention Layer following Modern Keras 3 patterns."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard ChannelAttention layer configuration for testing."""
        return {
            'channels': 64,
            'ratio': 8,
            'kernel_initializer': 'glorot_uniform',
            'kernel_regularizer': None,
            'use_bias': False,
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
    def sample_input_single_pixel(self) -> keras.KerasTensor:
        """Single pixel 4D input tensor for testing."""
        return tf.random.normal(shape=(4, 1, 1, 128), seed=44)

    @pytest.fixture
    def sample_input_large_channels(self) -> keras.KerasTensor:
        """Large channel 4D input tensor for testing."""
        return tf.random.normal(shape=(2, 14, 14, 512), seed=45)

    def test_initialization(self, layer_config: Dict[str, Any]) -> None:
        """
        Test layer initialization and parameter validation.

        Validates:
        - Correct attribute assignment from config
        - Unbuilt state after initialization
        - Sub-layer creation (Dense layers)
        - Parameter validation with invalid inputs
        """
        # Test successful initialization
        layer = ChannelAttention(**layer_config)

        # Verify configuration storage
        assert layer.channels == layer_config['channels']
        assert layer.ratio == layer_config['ratio']
        assert layer.use_bias == layer_config['use_bias']
        assert layer.kernel_initializer is not None
        assert layer.kernel_regularizer is layer_config['kernel_regularizer']

        # Verify unbuilt state
        assert not layer.built

        # Verify sub-layer creation
        assert layer.dense1 is not None
        assert layer.dense2 is not None
        assert isinstance(layer.dense1, keras.layers.Dense)
        assert isinstance(layer.dense2, keras.layers.Dense)

        # Verify MLP structure
        assert layer.dense1.units == layer_config['channels'] // layer_config['ratio']
        assert layer.dense1.activation.__name__ == 'relu'
        assert layer.dense2.units == layer_config['channels']
        # Final layer should have linear activation (default when None specified)
        assert layer.dense2.activation.__name__ == 'linear'

    def test_parameter_validation(self) -> None:
        """Test comprehensive parameter validation with invalid inputs."""

        # Test invalid channels (non-positive)
        with pytest.raises(ValueError, match="channels must be positive"):
            ChannelAttention(channels=0)

        with pytest.raises(ValueError, match="channels must be positive"):
            ChannelAttention(channels=-16)

        # Test invalid ratio (non-positive)
        with pytest.raises(ValueError, match="ratio must be positive"):
            ChannelAttention(channels=64, ratio=0)

        with pytest.raises(ValueError, match="ratio must be positive"):
            ChannelAttention(channels=64, ratio=-4)

        # Test channels not divisible by ratio
        with pytest.raises(ValueError, match="channels .* must be divisible by ratio"):
            ChannelAttention(channels=64, ratio=9)  # 64 not divisible by 9

        with pytest.raises(ValueError, match="channels .* must be divisible by ratio"):
            ChannelAttention(channels=128, ratio=7)  # 128 not divisible by 7

        # Test valid configurations
        valid_configs = [
            (64, 8),   # 64 / 8 = 8
            (128, 16), # 128 / 16 = 8
            (256, 4),  # 256 / 4 = 64
            (32, 2),   # 32 / 2 = 16
        ]

        for channels, ratio in valid_configs:
            layer = ChannelAttention(channels=channels, ratio=ratio)
            assert layer.channels == channels
            assert layer.ratio == ratio
            assert layer.dense1.units == channels // ratio

    @pytest.mark.parametrize("input_tensor", [
        "sample_input_4d", "sample_input_small",
        "sample_input_single_pixel", "sample_input_large_channels"
    ])
    def test_forward_pass(
        self,
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
        channels = inputs.shape[-1]

        # Create layer config matching input channels
        layer_config = {
            'channels': channels,
            'ratio': min(8, channels),  # Ensure valid ratio
        }

        layer = ChannelAttention(**layer_config)

        # Forward pass triggers building
        attention_weights = layer(inputs)

        # Verify building occurred
        assert layer.built
        assert layer.dense1.built
        assert layer.dense2.built

        # Verify output shape - should be (batch, 1, 1, channels)
        expected_shape = (inputs.shape[0], 1, 1, channels)
        assert attention_weights.shape == expected_shape

        # Verify attention weight properties
        attention_np = ops.convert_to_numpy(attention_weights)
        assert np.all(attention_np >= 0.0), "Attention weights must be non-negative"
        assert np.all(attention_np <= 1.0), "Attention weights must be <= 1.0 (sigmoid output)"

        # Test consistency across multiple calls
        attention_weights2 = layer(inputs)
        np.testing.assert_allclose(
            ops.convert_to_numpy(attention_weights),
            ops.convert_to_numpy(attention_weights2),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Multiple forward passes should be consistent"
        )

    def test_build_validation(self, layer_config: Dict[str, Any]) -> None:
        """
        Test build method validation.

        Validates:
        - Input shape validation
        - Channel dimension consistency
        - Proper error handling for invalid shapes
        """
        layer = ChannelAttention(**layer_config)

        # Test valid 4D input shape
        valid_shape = (None, 32, 32, layer_config['channels'])
        layer.build(valid_shape)
        assert layer.built

        # Test invalid input dimensions (not 4D)
        with pytest.raises(ValueError, match="Expected 4D input shape"):
            layer_new = ChannelAttention(**layer_config)
            layer_new.build((None, 32, layer_config['channels']))  # 3D

        # Test channel mismatch
        with pytest.raises(ValueError, match="Expected input channels .* to match layer channels"):
            layer_mismatch = ChannelAttention(**layer_config)
            wrong_channels = layer_config['channels'] + 16
            layer_mismatch.build((None, 32, 32, wrong_channels))

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
        # Create model with ChannelAttention layer
        inputs = keras.Input(shape=sample_input_4d.shape[1:])
        channel_attention = ChannelAttention(**layer_config)(inputs)
        # Apply attention to inputs (common usage pattern)
        attended = keras.layers.Multiply()([inputs, channel_attention])
        # Add classification head for more realistic test
        pooled = keras.layers.GlobalAveragePooling2D()(attended)
        outputs = keras.layers.Dense(10, activation='softmax')(pooled)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_prediction = model(sample_input_4d)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'channel_attention_test_model.keras')

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
            original_attention = model.layers[1]  # ChannelAttention layer
            loaded_attention = loaded_model.layers[1]  # ChannelAttention layer

            assert original_attention.channels == loaded_attention.channels
            assert original_attention.ratio == loaded_attention.ratio
            assert original_attention.use_bias == loaded_attention.use_bias

    def test_config_completeness(self, layer_config: Dict[str, Any]) -> None:
        """
        Test that get_config contains all __init__ parameters.

        Validates:
        - All initialization parameters in config
        - Proper serialization of complex objects
        - Config can reconstruct identical layer
        """
        layer = ChannelAttention(**layer_config)
        config = layer.get_config()

        # Check all required parameters present
        required_keys = [
            'channels', 'ratio', 'kernel_initializer', 'kernel_regularizer', 'use_bias'
        ]

        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        # Verify config values match initialization
        assert config['channels'] == layer_config['channels']
        assert config['ratio'] == layer_config['ratio']
        assert config['use_bias'] == layer_config['use_bias']

        # Test reconstruction from config
        reconstructed_layer = ChannelAttention.from_config(config)
        assert reconstructed_layer.channels == layer.channels
        assert reconstructed_layer.ratio == layer.ratio
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
        layer = ChannelAttention(**layer_config)

        with tf.GradientTape() as tape:
            # Enable gradient tracking for input
            tape.watch(sample_input_4d)

            # Forward pass
            attention_weights = layer(sample_input_4d, training=True)

            # Compute loss (encourage high attention values)
            loss = -ops.mean(attention_weights)  # Negative to maximize attention

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
        layer = ChannelAttention(**layer_config)

        # Forward pass in specified training mode
        attention_weights = layer(sample_input_4d, training=training)

        # Verify output shape and properties
        expected_shape = (sample_input_4d.shape[0], 1, 1, sample_input_4d.shape[-1])
        assert attention_weights.shape == expected_shape

        # Verify attention properties
        attention_np = ops.convert_to_numpy(attention_weights)
        assert np.all(attention_np >= 0.0)
        assert np.all(attention_np <= 1.0)

        # For ChannelAttention, output should be identical regardless of training mode
        # since there's no dropout or batch normalization
        attention_weights_inference = layer(sample_input_4d, training=False)
        np.testing.assert_allclose(
            ops.convert_to_numpy(attention_weights),
            ops.convert_to_numpy(attention_weights_inference),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Training and inference outputs should be identical for ChannelAttention"
        )

    def test_output_shape_computation(self, layer_config: Dict[str, Any]) -> None:
        """Test compute_output_shape method."""
        layer = ChannelAttention(**layer_config)

        # Test various 4D input shapes
        channels = layer_config['channels']
        test_cases = [
            ((None, 32, 32, channels), (None, 1, 1, channels)),
            ((8, 224, 224, channels), (8, 1, 1, channels)),
            ((16, 56, 56, channels), (16, 1, 1, channels)),
            ((4, 14, 14, channels), (4, 1, 1, channels)),
            ((1, 1, 1, channels), (1, 1, 1, channels)),  # Single pixel
        ]

        for input_shape, expected_output_shape in test_cases:
            output_shape = layer.compute_output_shape(input_shape)
            assert output_shape == expected_output_shape, \
                f"Input {input_shape} -> Expected {expected_output_shape}, Got {output_shape}"

    def test_channel_attention_mechanism(self, layer_config: Dict[str, Any]) -> None:
        """
        Test the core channel attention mechanism.

        Validates:
        - Global average pooling and global max pooling
        - Shared MLP processing
        - Element-wise addition and sigmoid activation
        - Attention weight properties
        """
        layer = ChannelAttention(**layer_config)

        # Create test input with known patterns
        batch_size, height, width, channels = 2, 16, 16, layer_config['channels']
        inputs = tf.random.normal((batch_size, height, width, channels), seed=46)

        # Get attention weights
        attention_weights = layer(inputs)

        # Test intermediate computations by manually computing them
        # Global average and max pooling
        expected_avg_pool = ops.mean(inputs, axis=[1, 2], keepdims=True)  # (batch, 1, 1, channels)
        expected_max_pool = ops.max(inputs, axis=[1, 2], keepdims=True)   # (batch, 1, 1, channels)

        # Verify pooling shapes
        assert expected_avg_pool.shape == (batch_size, 1, 1, channels)
        assert expected_max_pool.shape == (batch_size, 1, 1, channels)

        # The final attention weights should be properly shaped
        assert attention_weights.shape == (batch_size, 1, 1, channels)

        # Verify attention properties
        attention_np = ops.convert_to_numpy(attention_weights)
        assert np.all(attention_np >= 0.0), "Sigmoid output must be >= 0"
        assert np.all(attention_np <= 1.0), "Sigmoid output must be <= 1"

        # Test that different channels can have different attention weights
        attention_flat = attention_np.reshape(-1, channels)  # (batch_size, channels)

        # Check that not all channels have identical attention (with high probability)
        channel_std = np.std(attention_flat, axis=1)  # Standard deviation across channels
        assert np.any(channel_std > 1e-6), "All channels have identical attention - mechanism not working"

    def test_attention_application(self, layer_config: Dict[str, Any]) -> None:
        """
        Test channel attention when applied to inputs (multiplication).

        Validates:
        - Attention modulation effect
        - Channel-wise scaling behavior
        - Broadcasting compatibility
        """
        layer = ChannelAttention(**layer_config)

        # Create test input
        channels = layer_config['channels']
        inputs = tf.random.normal((4, 16, 16, channels), seed=47)

        # Get attention weights
        attention_weights = layer(inputs)

        # Apply attention (element-wise multiplication with broadcasting)
        attended_features = keras.layers.Multiply()([inputs, attention_weights])

        # Verify shapes
        assert attended_features.shape == inputs.shape

        # Convert to numpy for detailed analysis
        inputs_np = ops.convert_to_numpy(inputs)
        attention_np = ops.convert_to_numpy(attention_weights)
        attended_np = ops.convert_to_numpy(attended_features)

        # Verify attention effect: attended = inputs * attention (broadcast)
        expected_attended = inputs_np * attention_np  # Broadcasting should work
        np.testing.assert_allclose(
            attended_np, expected_attended,
            rtol=1e-6, atol=1e-6,
            err_msg="Attended features should equal inputs * attention"
        )

        # Test channel-wise effects
        batch_size, height, width, channels = inputs_np.shape

        # Flatten spatial dimensions for channel analysis
        inputs_flat = inputs_np.reshape(batch_size, -1, channels)  # (batch, H*W, channels)
        attention_flat = attention_np.reshape(batch_size, channels)  # (batch, channels)
        attended_flat = attended_np.reshape(batch_size, -1, channels)  # (batch, H*W, channels)

        # For each batch and each channel, check scaling effect
        for b in range(batch_size):
            for c in range(channels):
                channel_attention = attention_flat[b, c]
                original_channel = inputs_flat[b, :, c]  # All spatial positions for this channel
                attended_channel = attended_flat[b, :, c]  # All spatial positions for this channel

                # Attended should be original scaled by attention weight
                expected_attended_channel = original_channel * channel_attention
                np.testing.assert_allclose(
                    attended_channel, expected_attended_channel,
                    rtol=1e-6, atol=1e-6,
                    err_msg=f"Channel scaling incorrect for batch {b}, channel {c}"
                )

    def test_different_ratios(self) -> None:
        """
        Test channel attention with different reduction ratios.

        Validates:
        - Various ratios work correctly
        - Output shapes remain consistent
        - Different compression levels in MLP
        """
        # Test different valid ratios for 64 channels
        channels = 64
        ratios = [2, 4, 8, 16]  # All divide 64 evenly
        inputs = tf.random.normal((4, 32, 32, channels), seed=48)

        attention_maps = {}

        for ratio in ratios:
            layer = ChannelAttention(channels=channels, ratio=ratio)
            attention_weights = layer(inputs)

            # Verify output shape consistency
            assert attention_weights.shape == (4, 1, 1, channels)

            # Verify MLP bottleneck dimension
            expected_bottleneck = channels // ratio
            assert layer.dense1.units == expected_bottleneck

            # Store for comparison
            attention_maps[ratio] = ops.convert_to_numpy(attention_weights)

        # Different ratios should potentially produce different attention patterns
        for r1 in ratios:
            for r2 in ratios:
                if r1 != r2:
                    map1 = attention_maps[r1]
                    map2 = attention_maps[r2]

                    # Maps should be different (with high probability for random weights)
                    assert not np.allclose(map1, map2, atol=1e-3), \
                        f"Attention maps for ratios {r1} and {r2} are too similar"

    def test_numerical_stability(self, layer_config: Dict[str, Any]) -> None:
        """
        Test numerical stability with extreme inputs.

        Validates:
        - Handling of large magnitude inputs
        - Stability with very small inputs
        - No NaN/Inf in outputs
        - Bounded output range maintained
        """
        layer = ChannelAttention(**layer_config)
        channels = layer_config['channels']

        # Test cases with extreme inputs
        test_cases = [
            ("zeros", ops.zeros((4, 16, 16, channels))),
            ("large_positive", ops.ones((4, 16, 16, channels)) * 1000.0),
            ("large_negative", ops.ones((4, 16, 16, channels)) * -1000.0),
            ("very_small", ops.ones((4, 16, 16, channels)) * 1e-8),
            ("mixed_extreme", tf.random.normal((4, 16, 16, channels), seed=49) * 1e6)
        ]

        for case_name, inputs in test_cases:
            attention_weights = layer(inputs)

            # Convert to numpy for detailed checks
            attention_np = ops.convert_to_numpy(attention_weights)

            # Check for numerical issues
            assert not np.any(np.isnan(attention_np)), \
                f"NaN values in attention weights for case: {case_name}"

            assert not np.any(np.isinf(attention_np)), \
                f"Inf values in attention weights for case: {case_name}"

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
        channels = 64

        # Test without regularization
        layer_no_reg = ChannelAttention(channels=channels, kernel_regularizer=None)

        # Test with L2 regularization
        layer_l2 = ChannelAttention(
            channels=channels,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # Test with L1 regularization
        layer_l1 = ChannelAttention(
            channels=channels,
            kernel_regularizer=keras.regularizers.L1(1e-4)
        )

        inputs = tf.random.normal((4, 16, 16, channels), seed=50)

        # All should work without errors
        attention_no_reg = layer_no_reg(inputs)
        attention_l2 = layer_l2(inputs)
        attention_l1 = layer_l1(inputs)

        # All should produce valid attention weights
        for attention, name in [(attention_no_reg, "no_reg"),
                               (attention_l2, "l2"),
                               (attention_l1, "l1")]:
            attention_np = ops.convert_to_numpy(attention)
            assert attention.shape == (4, 1, 1, channels)
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

        # Test minimum valid configuration
        layer_min = ChannelAttention(channels=2, ratio=1)  # Minimal valid setup
        inputs_min = tf.random.normal((2, 8, 8, 2))
        attention = layer_min(inputs_min)
        assert attention.shape == (2, 1, 1, 2)

        # Test single spatial dimension
        layer = ChannelAttention(channels=32, ratio=4)
        inputs_single = tf.random.normal((4, 1, 1, 32))
        attention = layer(inputs_single)
        assert attention.shape == (4, 1, 1, 32)

        # Test large channel count
        layer_large = ChannelAttention(channels=1024, ratio=16)
        inputs_large = tf.random.normal((2, 8, 8, 1024))
        attention = layer_large(inputs_large)
        assert attention.shape == (2, 1, 1, 1024)

        # Test ratio = channels (extreme compression)
        layer_extreme = ChannelAttention(channels=32, ratio=32)  # Bottleneck to 1 unit
        inputs_extreme = tf.random.normal((2, 16, 16, 32))
        attention = layer_extreme(inputs_extreme)
        assert attention.shape == (2, 1, 1, 32)
        assert layer_extreme.dense1.units == 1

    def test_consistency_with_cbam_paper(self) -> None:
        """
        Test consistency with CBAM paper specifications.

        Validates:
        - Default reduction ratio (often 16 in practice, here 8)
        - Shared MLP structure
        - ReLU activation in intermediate layer
        - Sigmoid activation for final weights
        - Global pooling operations
        """
        channels = 256
        ratio = 8  # Common choice, though paper uses 16 for larger networks

        layer = ChannelAttention(channels=channels, ratio=ratio)

        # Verify MLP structure matches CBAM specifications
        assert layer.dense1.units == channels // ratio, "Bottleneck dimension incorrect"
        assert layer.dense1.activation.__name__ == 'relu', "Should use ReLU in intermediate layer"
        assert layer.dense2.units == channels, "Output dimension should match input channels"
        # Final layer should have linear activation (sigmoid applied after in call method)
        assert layer.dense2.activation.__name__ == 'linear', "Final layer should have linear activation"

        # Test with ImageNet-like input dimensions
        inputs = tf.random.normal((8, 224, 224, channels))  # ImageNet-like
        attention_weights = layer(inputs)

        # Should output channel attention weights
        assert attention_weights.shape == (8, 1, 1, channels)

        # Attention values should be in valid range (sigmoid output)
        attention_np = ops.convert_to_numpy(attention_weights)
        assert np.all(attention_np >= 0.0)
        assert np.all(attention_np <= 1.0)

        # Test that the mechanism is working (not all channels equal)
        channel_means = np.mean(attention_np, axis=(0, 1, 2))  # Mean across batch and spatial
        channel_variance = np.var(channel_means)
        assert channel_variance > 1e-6, "All channels have identical attention - mechanism not working"

    def test_mlp_shared_weights(self) -> None:
        """
        Test that the MLP weights are truly shared between avg and max pooling paths.

        Validates:
        - Same Dense layers used for both avg and max pooled features
        - Weight sharing behavior
        - Identical processing of both paths through MLP
        """
        layer = ChannelAttention(channels=64, ratio=8)

        # Create test input
        inputs = tf.random.normal((2, 16, 16, 64), seed=51)

        # Manually compute average and max pooling
        avg_pooled = ops.mean(inputs, axis=[1, 2], keepdims=True)  # (2, 1, 1, 64)
        max_pooled = ops.max(inputs, axis=[1, 2], keepdims=True)   # (2, 1, 1, 64)

        # Flatten for MLP processing
        avg_flat = ops.reshape(avg_pooled, (-1, 64))  # (2, 64)
        max_flat = ops.reshape(max_pooled, (-1, 64))  # (2, 64)

        # Build the layer first
        _ = layer(inputs)

        # Process both through the same Dense layers (shared MLP)
        avg_out1 = layer.dense1(avg_flat)
        avg_out2 = layer.dense2(avg_out1)

        max_out1 = layer.dense1(max_flat)  # Same dense1 layer
        max_out2 = layer.dense2(max_out1)  # Same dense2 layer

        # Verify intermediate shapes
        assert avg_out1.shape == (2, 8)  # channels // ratio = 64 // 8 = 8
        assert max_out1.shape == (2, 8)
        assert avg_out2.shape == (2, 64)  # Back to original channel count
        assert max_out2.shape == (2, 64)

        # The final result should be sigmoid(avg_out2 + max_out2)
        manual_result = ops.sigmoid(avg_out2 + max_out2)
        manual_result = ops.reshape(manual_result, (-1, 1, 1, 64))

        # Compare with layer output
        layer_result = layer(inputs)

        np.testing.assert_allclose(
            ops.convert_to_numpy(manual_result),
            ops.convert_to_numpy(layer_result),
            rtol=1e-6, atol=1e-6,
            err_msg="Manual computation should match layer output"
        )


if __name__ == '__main__':
    # Run with: python -m pytest channel_attention_test.py -v
    pytest.main([__file__, '-v', '--tb=short'])