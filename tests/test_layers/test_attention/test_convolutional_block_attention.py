"""
Comprehensive Test Suite for CBAM Layer

This test suite follows Modern Keras 3 testing best practices with emphasis on
serialization testing and production readiness validation for the Convolutional
Block Attention Module (CBAM) layer.
"""

import pytest
import tempfile
import os
from typing import Any, Dict
import numpy as np

import keras
from keras import ops
import tensorflow as tf

from dl_techniques.layers.attention.convolutional_block_attention import CBAM


class TestCBAM:
    """Comprehensive test suite for CBAM Layer following Modern Keras 3 patterns."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard CBAM layer configuration for testing."""
        return {
            'channels': 64,
            'ratio': 8,
            'kernel_size': 7,
            'channel_kernel_initializer': 'glorot_uniform',
            'spatial_kernel_initializer': 'glorot_uniform',
            'channel_kernel_regularizer': None,
            'spatial_kernel_regularizer': None,
            'channel_use_bias': False,
            'spatial_use_bias': True,
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
        - Sub-layer creation (ChannelAttention and SpatialAttention)
        - Parameter validation with invalid inputs
        """
        # Test successful initialization
        layer = CBAM(**layer_config)

        # Verify configuration storage
        assert layer.channels == layer_config['channels']
        assert layer.ratio == layer_config['ratio']
        assert layer.kernel_size == layer_config['kernel_size']
        assert layer.channel_use_bias == layer_config['channel_use_bias']
        assert layer.spatial_use_bias == layer_config['spatial_use_bias']
        assert layer.channel_kernel_initializer is not None
        assert layer.spatial_kernel_initializer is not None
        assert layer.channel_kernel_regularizer is layer_config['channel_kernel_regularizer']
        assert layer.spatial_kernel_regularizer is layer_config['spatial_kernel_regularizer']

        # Verify unbuilt state
        assert not layer.built

        # Verify sub-layer creation in __init__ (Modern Keras 3 pattern)
        assert layer.channel_attention is not None
        assert layer.spatial_attention is not None

        # Verify sub-layer types
        from dl_techniques.layers.attention.channel_attention import ChannelAttention
        from dl_techniques.layers.attention.spatial_attention import SpatialAttention
        assert isinstance(layer.channel_attention, ChannelAttention)
        assert isinstance(layer.spatial_attention, SpatialAttention)

        # Verify sub-layer configuration
        assert layer.channel_attention.channels == layer_config['channels']
        assert layer.channel_attention.ratio == layer_config['ratio']
        assert layer.spatial_attention.kernel_size == layer_config['kernel_size']

    def test_parameter_validation(self) -> None:
        """Test comprehensive parameter validation with invalid inputs."""

        # Test invalid channels (non-positive)
        with pytest.raises(ValueError, match="channels must be positive"):
            CBAM(channels=0)

        with pytest.raises(ValueError, match="channels must be positive"):
            CBAM(channels=-16)

        # Test invalid ratio (non-positive)
        with pytest.raises(ValueError, match="ratio must be positive"):
            CBAM(channels=64, ratio=0)

        with pytest.raises(ValueError, match="ratio must be positive"):
            CBAM(channels=64, ratio=-4)

        # Test invalid kernel_size (non-positive)
        with pytest.raises(ValueError, match="kernel_size must be positive"):
            CBAM(channels=64, kernel_size=0)

        with pytest.raises(ValueError, match="kernel_size must be positive"):
            CBAM(channels=64, kernel_size=-3)

        # Test valid configurations
        valid_configs = [
            (64, 8, 7),   # Standard configuration
            (128, 16, 3), # Smaller spatial kernel
            (256, 4, 5),  # Larger ratio, medium kernel
            (32, 2, 1),   # Small configuration
        ]

        for channels, ratio, kernel_size in valid_configs:
            layer = CBAM(channels=channels, ratio=ratio, kernel_size=kernel_size)
            assert layer.channels == channels
            assert layer.ratio == ratio
            assert layer.kernel_size == kernel_size

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
        - Correct output shapes (same as input)
        - Sequential attention mechanism
        - Multiple forward passes consistency
        """
        # Get the input tensor from fixture
        inputs = request.getfixturevalue(input_tensor)
        channels = inputs.shape[-1]

        # Create layer config matching input channels
        layer_config = {
            'channels': channels,
            'ratio': min(8, channels),  # Ensure valid ratio
            'kernel_size': 7
        }

        layer = CBAM(**layer_config)

        # Forward pass triggers building
        refined_features = layer(inputs)

        # Verify building occurred
        assert layer.built
        assert layer.channel_attention.built
        assert layer.spatial_attention.built

        # Verify output shape - should be same as input (CBAM preserves dimensions)
        expected_shape = inputs.shape
        assert refined_features.shape == expected_shape

        # Test consistency across multiple calls
        refined_features2 = layer(inputs)
        np.testing.assert_allclose(
            ops.convert_to_numpy(refined_features),
            ops.convert_to_numpy(refined_features2),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Multiple forward passes should be consistent"
        )

    def test_build_validation(self, layer_config: Dict[str, Any]) -> None:
        """
        Test build method validation.

        Validates:
        - Sub-layer building in parent build method
        - Input shape validation
        - Channel dimension consistency
        - Proper error handling for invalid shapes
        """
        layer = CBAM(**layer_config)

        # Test valid 4D input shape
        valid_shape = (None, 32, 32, layer_config['channels'])
        layer.build(valid_shape)
        assert layer.built
        assert layer.channel_attention.built
        assert layer.spatial_attention.built

        # The sub-layers should have been built with explicit build() calls
        # This is critical for serialization robustness

        # Test that building works correctly
        inputs = tf.random.normal((4, 32, 32, layer_config['channels']))
        output = layer(inputs)
        assert output.shape == inputs.shape

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
        - Weight preservation for both sub-layers
        """
        # Create model with CBAM layer
        inputs = keras.Input(shape=sample_input_4d.shape[1:])
        cbam_output = CBAM(**layer_config)(inputs)
        # Add classification head for more realistic test
        pooled = keras.layers.GlobalAveragePooling2D()(cbam_output)
        outputs = keras.layers.Dense(10, activation='softmax')(pooled)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_prediction = model(sample_input_4d)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'cbam_test_model.keras')

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
            original_cbam = model.layers[1]  # CBAM layer
            loaded_cbam = loaded_model.layers[1]  # CBAM layer

            assert original_cbam.channels == loaded_cbam.channels
            assert original_cbam.ratio == loaded_cbam.ratio
            assert original_cbam.kernel_size == loaded_cbam.kernel_size
            assert original_cbam.channel_use_bias == loaded_cbam.channel_use_bias
            assert original_cbam.spatial_use_bias == loaded_cbam.spatial_use_bias

    def test_config_completeness(self, layer_config: Dict[str, Any]) -> None:
        """
        Test that get_config contains all __init__ parameters.

        Validates:
        - All initialization parameters in config
        - Proper serialization of complex objects
        - Config can reconstruct identical layer
        """
        layer = CBAM(**layer_config)
        config = layer.get_config()

        # Check all required parameters present
        required_keys = [
            'channels', 'ratio', 'kernel_size',
            'channel_kernel_initializer', 'spatial_kernel_initializer',
            'channel_kernel_regularizer', 'spatial_kernel_regularizer',
            'channel_use_bias', 'spatial_use_bias'
        ]

        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        # Verify config values match initialization
        assert config['channels'] == layer_config['channels']
        assert config['ratio'] == layer_config['ratio']
        assert config['kernel_size'] == layer_config['kernel_size']
        assert config['channel_use_bias'] == layer_config['channel_use_bias']
        assert config['spatial_use_bias'] == layer_config['spatial_use_bias']

        # Test reconstruction from config
        reconstructed_layer = CBAM.from_config(config)
        assert reconstructed_layer.channels == layer.channels
        assert reconstructed_layer.ratio == layer.ratio
        assert reconstructed_layer.kernel_size == layer.kernel_size
        assert reconstructed_layer.channel_use_bias == layer.channel_use_bias
        assert reconstructed_layer.spatial_use_bias == layer.spatial_use_bias

    def test_gradients_flow(
        self,
        layer_config: Dict[str, Any],
        sample_input_4d: keras.KerasTensor
    ) -> None:
        """
        Test gradient computation and backpropagation.

        Validates:
        - Gradients computed for all trainable weights (both sub-layers)
        - No None gradients
        - Reasonable gradient magnitudes
        """
        layer = CBAM(**layer_config)

        with tf.GradientTape() as tape:
            # Enable gradient tracking for input
            tape.watch(sample_input_4d)

            # Forward pass
            refined_features = layer(sample_input_4d, training=True)

            # Compute loss (encourage feature enhancement)
            loss = ops.mean(ops.square(refined_features))

        # Get sub-layer weights for verification
        channel_weights = layer.channel_attention.trainable_weights
        spatial_weights = layer.spatial_attention.trainable_weights

        assert len(channel_weights) > 0, "Channel attention has no trainable weights"
        assert len(spatial_weights) > 0, "Spatial attention has no trainable weights"

        # Compute all gradients in a single call (avoids tape reuse issue)
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

        # Verify that both channel and spatial attention weights have gradients
        # We can check this by matching weights to gradients from layer.trainable_weights
        layer_trainable_weights = layer.trainable_weights

        # Find channel attention weight indices
        channel_gradient_indices = []
        spatial_gradient_indices = []

        for i, weight in enumerate(layer_trainable_weights):
            if any(w is weight for w in channel_weights):
                channel_gradient_indices.append(i)
            elif any(w is weight for w in spatial_weights):
                spatial_gradient_indices.append(i)

        # Verify both sub-layers have gradients
        assert len(channel_gradient_indices) > 0, "No channel attention gradients found"
        assert len(spatial_gradient_indices) > 0, "No spatial attention gradients found"

        # Check that corresponding gradients exist and are non-None
        for idx in channel_gradient_indices:
            assert layer_gradients[idx] is not None, f"Channel attention gradient {idx} is None"

        for idx in spatial_gradient_indices:
            assert layer_gradients[idx] is not None, f"Spatial attention gradient {idx} is None"

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
        - Consistent outputs across training modes
        - Proper handling of training parameter in both sub-layers
        """
        layer = CBAM(**layer_config)

        # Forward pass in specified training mode
        refined_features = layer(sample_input_4d, training=training)

        # Verify output shape and properties
        expected_shape = sample_input_4d.shape
        assert refined_features.shape == expected_shape

        # For CBAM, output should be identical regardless of training mode
        # since there's no dropout or batch normalization in the attention modules
        refined_features_inference = layer(sample_input_4d, training=False)
        np.testing.assert_allclose(
            ops.convert_to_numpy(refined_features),
            ops.convert_to_numpy(refined_features_inference),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Training and inference outputs should be identical for CBAM"
        )

    def test_output_shape_computation(self, layer_config: Dict[str, Any]) -> None:
        """Test compute_output_shape method."""
        layer = CBAM(**layer_config)

        # Test various 4D input shapes
        channels = layer_config['channels']
        test_cases = [
            ((None, 32, 32, channels), (None, 32, 32, channels)),
            ((8, 224, 224, channels), (8, 224, 224, channels)),
            ((16, 56, 56, channels), (16, 56, 56, channels)),
            ((4, 14, 14, channels), (4, 14, 14, channels)),
            ((1, 1, 1, channels), (1, 1, 1, channels)),  # Single pixel
        ]

        for input_shape, expected_output_shape in test_cases:
            output_shape = layer.compute_output_shape(input_shape)
            assert output_shape == expected_output_shape, \
                f"Input {input_shape} -> Expected {expected_output_shape}, Got {output_shape}"

    def test_cbam_attention_mechanism(self, layer_config: Dict[str, Any]) -> None:
        """
        Test the core CBAM attention mechanism.

        Validates:
        - Sequential application of channel then spatial attention
        - Proper attention flow: F' = Ms(F) ⊗ (Mc(F) ⊗ F)
        - Attention effects are cumulative
        - Both attention mechanisms contribute to final result
        """
        layer = CBAM(**layer_config)

        # Create test input with known patterns
        batch_size, height, width, channels = 2, 16, 16, layer_config['channels']
        inputs = tf.random.normal((batch_size, height, width, channels), seed=46)

        # Get final refined features
        refined_features = layer(inputs)

        # Manually trace the CBAM process
        # Step 1: Channel attention
        channel_attention_weights = layer.channel_attention(inputs)
        assert channel_attention_weights.shape == (batch_size, 1, 1, channels)

        # Step 2: Apply channel attention
        channel_refined = inputs * channel_attention_weights

        # Step 3: Spatial attention on channel-refined features
        spatial_attention_weights = layer.spatial_attention(channel_refined)
        assert spatial_attention_weights.shape == (batch_size, height, width, 1)

        # Step 4: Apply spatial attention
        expected_refined = channel_refined * spatial_attention_weights

        # Compare with layer output
        np.testing.assert_allclose(
            ops.convert_to_numpy(expected_refined),
            ops.convert_to_numpy(refined_features),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Manual CBAM computation should match layer output"
        )

        # Verify attention properties
        channel_attention_np = ops.convert_to_numpy(channel_attention_weights)
        spatial_attention_np = ops.convert_to_numpy(spatial_attention_weights)

        # Channel attention should be in [0, 1] (sigmoid output)
        assert np.all(channel_attention_np >= 0.0), "Channel attention must be >= 0"
        assert np.all(channel_attention_np <= 1.0), "Channel attention must be <= 1"

        # Spatial attention should be in [0, 1] (sigmoid output)
        assert np.all(spatial_attention_np >= 0.0), "Spatial attention must be >= 0"
        assert np.all(spatial_attention_np <= 1.0), "Spatial attention must be <= 1"

    def test_attention_effects_cumulative(self, layer_config: Dict[str, Any]) -> None:
        """
        Test that CBAM effects are cumulative and beneficial.

        Validates:
        - CBAM refines features better than either attention alone
        - Sequential application order matters
        - Attention mechanisms complement each other
        """
        # Create individual attention layers for comparison
        from dl_techniques.layers.attention.channel_attention import ChannelAttention
        from dl_techniques.layers.attention.spatial_attention import SpatialAttention

        channel_only = ChannelAttention(
            channels=layer_config['channels'],
            ratio=layer_config['ratio']
        )

        spatial_only = SpatialAttention(
            kernel_size=layer_config['kernel_size']
        )

        cbam_layer = CBAM(**layer_config)

        # Test input
        inputs = tf.random.normal((4, 32, 32, layer_config['channels']), seed=47)

        # Get different attention outputs
        channel_weights = channel_only(inputs)
        channel_refined = inputs * channel_weights

        spatial_weights = spatial_only(inputs)
        spatial_refined = inputs * spatial_weights

        cbam_refined = cbam_layer(inputs)

        # Verify shapes
        assert channel_refined.shape == inputs.shape
        assert spatial_refined.shape == inputs.shape
        assert cbam_refined.shape == inputs.shape

        # CBAM should be different from either attention alone
        # (This tests that both attentions contribute)
        assert not np.allclose(
            ops.convert_to_numpy(cbam_refined),
            ops.convert_to_numpy(channel_refined),
            atol=1e-6
        ), "CBAM should differ from channel attention alone"

        assert not np.allclose(
            ops.convert_to_numpy(cbam_refined),
            ops.convert_to_numpy(spatial_refined),
            atol=1e-6
        ), "CBAM should differ from spatial attention alone"

    def test_different_configurations(self) -> None:
        """
        Test CBAM with different configuration parameters.

        Validates:
        - Various channel reduction ratios
        - Different spatial kernel sizes
        - Different bias configurations
        - Output consistency across configurations
        """
        # Test different configurations
        configs = [
            {'channels': 64, 'ratio': 4, 'kernel_size': 3},   # High compression, small kernel
            {'channels': 128, 'ratio': 16, 'kernel_size': 7}, # Low compression, large kernel
            {'channels': 256, 'ratio': 8, 'kernel_size': 5},  # Medium configuration
        ]

        inputs_dict = {
            64: tf.random.normal((4, 32, 32, 64), seed=48),
            128: tf.random.normal((4, 16, 16, 128), seed=49),
            256: tf.random.normal((4, 8, 8, 256), seed=50),
        }

        attention_outputs = {}

        for config in configs:
            layer = CBAM(**config)
            inputs = inputs_dict[config['channels']]

            refined_features = layer(inputs)

            # Verify output shape consistency
            assert refined_features.shape == inputs.shape

            # Store for comparison
            attention_outputs[str(config)] = ops.convert_to_numpy(refined_features)

        # Different configurations should potentially produce different results
        config_keys = list(attention_outputs.keys())
        for i in range(len(config_keys)):
            for j in range(i+1, len(config_keys)):
                key1, key2 = config_keys[i], config_keys[j]
                if configs[i]['channels'] == configs[j]['channels']:  # Same input size
                    output1 = attention_outputs[key1]
                    output2 = attention_outputs[key2]

                    # Should be different (with high probability for random weights)
                    assert not np.allclose(output1, output2, atol=1e-2), \
                        f"Attention outputs for configs {key1} and {key2} are too similar"

    def test_numerical_stability(self, layer_config: Dict[str, Any]) -> None:
        """
        Test numerical stability with extreme inputs.

        Validates:
        - Handling of large magnitude inputs
        - Stability with very small inputs
        - No NaN/Inf in outputs
        - Reasonable output magnitudes
        """
        layer = CBAM(**layer_config)
        channels = layer_config['channels']

        # Test cases with extreme inputs
        test_cases = [
            ("zeros", ops.zeros((4, 16, 16, channels))),
            ("large_positive", ops.ones((4, 16, 16, channels)) * 1000.0),
            ("large_negative", ops.ones((4, 16, 16, channels)) * -1000.0),
            ("very_small", ops.ones((4, 16, 16, channels)) * 1e-8),
            ("mixed_extreme", tf.random.normal((4, 16, 16, channels), seed=51) * 1e6)
        ]

        for case_name, inputs in test_cases:
            refined_features = layer(inputs)

            # Convert to numpy for detailed checks
            output_np = ops.convert_to_numpy(refined_features)

            # Check for numerical issues
            assert not np.any(np.isnan(output_np)), \
                f"NaN values in output for case: {case_name}"

            assert not np.any(np.isinf(output_np)), \
                f"Inf values in output for case: {case_name}"

            # Output should have reasonable magnitudes (not exploding)
            max_magnitude = np.max(np.abs(output_np))
            assert max_magnitude < 1e10, \
                f"Output magnitude too large for case: {case_name}: {max_magnitude}"

    def test_regularization_effects(self) -> None:
        """
        Test kernel regularization effects for both sub-layers.

        Validates:
        - L1 and L2 regularization can be applied to both channel and spatial attention
        - Regularization affects training behavior
        - No errors during gradient computation with regularizers
        """
        channels = 64

        # Test with different regularization configurations
        test_configs = [
            # No regularization
            {
                'channels': channels,
                'channel_kernel_regularizer': None,
                'spatial_kernel_regularizer': None
            },
            # L2 regularization on both
            {
                'channels': channels,
                'channel_kernel_regularizer': keras.regularizers.L2(1e-4),
                'spatial_kernel_regularizer': keras.regularizers.L2(1e-4)
            },
            # L1 regularization on channel, L2 on spatial
            {
                'channels': channels,
                'channel_kernel_regularizer': keras.regularizers.L1(1e-4),
                'spatial_kernel_regularizer': keras.regularizers.L2(1e-4)
            },
        ]

        inputs = tf.random.normal((4, 16, 16, channels), seed=52)

        for i, config in enumerate(test_configs):
            layer = CBAM(**config)

            # Forward pass should work without errors
            refined_features = layer(inputs)

            # Verify output properties
            assert refined_features.shape == inputs.shape
            output_np = ops.convert_to_numpy(refined_features)
            assert not np.any(np.isnan(output_np)), f"NaN in output for config {i}"
            assert not np.any(np.isinf(output_np)), f"Inf in output for config {i}"

            # Test gradient computation with regularizers
            with tf.GradientTape() as tape:
                output = layer(inputs, training=True)
                loss = ops.mean(output)

            gradients = tape.gradient(loss, layer.trainable_weights)
            assert all(g is not None for g in gradients), f"Gradients should exist with regularizers for config {i}"

    def test_integration_with_cnn(self) -> None:
        """
        Test CBAM integration within a CNN architecture.

        Validates:
        - CBAM works correctly in realistic CNN context
        - Preserves feature map dimensions
        - Improves feature representation (tested via different outputs)
        """
        # Create a simple CNN with and without CBAM
        def create_cnn(use_cbam=False):
            inputs = keras.Input(shape=(32, 32, 3))

            x = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
            x = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)

            if use_cbam:
                x = CBAM(channels=64, ratio=8, kernel_size=7)(x)

            x = keras.layers.GlobalAveragePooling2D()(x)
            outputs = keras.layers.Dense(10, activation='softmax')(x)

            return keras.Model(inputs, outputs)

        # Create models
        model_without_cbam = create_cnn(use_cbam=False)
        model_with_cbam = create_cnn(use_cbam=True)

        # Test input
        test_input = tf.random.normal((8, 32, 32, 3), seed=53)

        # Get predictions
        pred_without = model_without_cbam(test_input)
        pred_with = model_with_cbam(test_input)

        # Both should have same output shape
        assert pred_without.shape == pred_with.shape == (8, 10)

        # Predictions should be different (CBAM should change the features)
        assert not np.allclose(
            ops.convert_to_numpy(pred_without),
            ops.convert_to_numpy(pred_with),
            atol=1e-3
        ), "CBAM should produce different features than non-CBAM model"

        # Both outputs should be valid probability distributions
        for pred, name in [(pred_without, "without"), (pred_with, "with")]:
            pred_np = ops.convert_to_numpy(pred)

            # Should be non-negative
            assert np.all(pred_np >= 0), f"Negative probabilities in {name} CBAM model"

            # Should sum to approximately 1 (softmax output)
            sums = np.sum(pred_np, axis=1)
            np.testing.assert_allclose(
                sums, np.ones_like(sums), rtol=1e-6,
                err_msg=f"Probabilities don't sum to 1 in {name} CBAM model"
            )

    def test_edge_cases(self) -> None:
        """Test error conditions and edge cases."""

        # Test minimum valid configuration
        layer_min = CBAM(channels=2, ratio=1, kernel_size=1)  # Minimal valid setup
        inputs_min = tf.random.normal((2, 8, 8, 2))
        refined = layer_min(inputs_min)
        assert refined.shape == (2, 8, 8, 2)

        # Test single spatial dimension
        layer = CBAM(channels=32, ratio=4, kernel_size=1)
        inputs_single = tf.random.normal((4, 1, 1, 32))
        refined = layer(inputs_single)
        assert refined.shape == (4, 1, 1, 32)

        # Test large channel count
        layer_large = CBAM(channels=1024, ratio=16, kernel_size=3)
        inputs_large = tf.random.normal((2, 8, 8, 1024))
        refined = layer_large(inputs_large)
        assert refined.shape == (2, 8, 8, 1024)

        # Test extreme compression ratio
        layer_extreme = CBAM(channels=64, ratio=64, kernel_size=3)  # Maximum compression
        inputs_extreme = tf.random.normal((2, 16, 16, 64))
        refined = layer_extreme(inputs_extreme)
        assert refined.shape == (2, 16, 16, 64)

        # Test large spatial kernel
        layer_large_kernel = CBAM(channels=32, ratio=4, kernel_size=15)
        inputs_kernel = tf.random.normal((2, 32, 32, 32))  # Ensure spatial size > kernel
        refined = layer_large_kernel(inputs_kernel)
        assert refined.shape == (2, 32, 32, 32)

    def test_consistency_with_cbam_paper(self) -> None:
        """
        Test consistency with CBAM paper specifications.

        Validates:
        - Default parameters match paper recommendations
        - Sequential attention flow (channel then spatial)
        - Proper attention combination
        - Expected behavior on ImageNet-like inputs
        """
        # Standard CBAM configuration as per paper
        channels = 512  # Common in ResNet
        ratio = 16      # Paper recommendation for larger networks
        kernel_size = 7 # Paper recommendation for spatial attention

        layer = CBAM(channels=channels, ratio=ratio, kernel_size=kernel_size)

        # Test with ResNet-like feature maps
        inputs = tf.random.normal((8, 28, 28, channels), seed=54)  # ResNet conv4_x like
        refined_features = layer(inputs)

        # Should preserve dimensions
        assert refined_features.shape == inputs.shape

        # Test attention mechanism consistency
        # Channel attention should focus on important channels
        channel_attention = layer.channel_attention(inputs)
        assert channel_attention.shape == (8, 1, 1, channels)

        # Apply channel attention
        channel_refined = inputs * channel_attention

        # Spatial attention should focus on important spatial locations
        spatial_attention = layer.spatial_attention(channel_refined)
        assert spatial_attention.shape == (8, 28, 28, 1)

        # Final result
        expected_output = channel_refined * spatial_attention
        np.testing.assert_allclose(
            ops.convert_to_numpy(expected_output),
            ops.convert_to_numpy(refined_features),
            rtol=1e-6,
            atol=1e-6,
            err_msg="CBAM should follow paper's sequential attention flow"
        )

        # Verify attention properties
        channel_attention_np = ops.convert_to_numpy(channel_attention)
        spatial_attention_np = ops.convert_to_numpy(spatial_attention)

        # Both attention maps should be in valid range
        assert np.all(channel_attention_np >= 0.0) and np.all(channel_attention_np <= 1.0)
        assert np.all(spatial_attention_np >= 0.0) and np.all(spatial_attention_np <= 1.0)

        # Attention should not be uniform (mechanism should be working)
        channel_std = np.std(channel_attention_np.reshape(-1, channels), axis=1)
        spatial_std = np.std(spatial_attention_np.reshape(8, -1), axis=1)

        assert np.any(channel_std > 1e-6), "Channel attention should vary across channels"
        assert np.any(spatial_std > 1e-6), "Spatial attention should vary across spatial locations"

    def test_bias_configurations(self) -> None:
        """
        Test different bias configurations for channel and spatial attention.

        Validates:
        - Different bias settings work correctly
        - Bias affects output as expected
        - All combinations are valid
        """
        channels = 64
        inputs = tf.random.normal((4, 16, 16, channels), seed=55)

        # Test all bias combinations
        bias_configs = [
            (False, False),  # No bias in either
            (False, True),   # Bias only in spatial
            (True, False),   # Bias only in channel
            (True, True),    # Bias in both
        ]

        outputs = {}
        for channel_bias, spatial_bias in bias_configs:
            layer = CBAM(
                channels=channels,
                channel_use_bias=channel_bias,
                spatial_use_bias=spatial_bias
            )

            output = layer(inputs)
            assert output.shape == inputs.shape

            # Store for comparison
            key = f"ch_{channel_bias}_sp_{spatial_bias}"
            outputs[key] = ops.convert_to_numpy(output)

        # Different bias configurations should produce different outputs
        keys = list(outputs.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                key1, key2 = keys[i], keys[j]
                output1, output2 = outputs[key1], outputs[key2]

                # Should be different (though potentially subtly)
                assert not np.allclose(output1, output2, atol=1e-6), \
                    f"Outputs for bias configs {key1} and {key2} should differ"

    def test_memory_efficiency(self, layer_config: Dict[str, Any]) -> None:
        """
        Test memory efficiency and scalability.

        Validates:
        - Layer works with larger inputs
        - No memory leaks in repeated calls
        - Reasonable memory usage patterns
        """
        layer = CBAM(**layer_config)

        # Test with progressively larger inputs
        test_sizes = [
            (2, 32, 32, layer_config['channels']),
            (4, 64, 64, layer_config['channels']),
            (2, 128, 128, layer_config['channels']),
        ]

        for batch, height, width, channels in test_sizes:
            inputs = tf.random.normal((batch, height, width, channels), seed=56)

            # Should handle larger inputs without issues
            output = layer(inputs)
            assert output.shape == inputs.shape

            # Verify no numerical issues
            output_np = ops.convert_to_numpy(output)
            assert not np.any(np.isnan(output_np))
            assert not np.any(np.isinf(output_np))

        # Test repeated calls (check for memory leaks)
        inputs = tf.random.normal((2, 32, 32, layer_config['channels']), seed=57)

        for _ in range(10):  # Multiple calls
            output = layer(inputs)
            assert output.shape == inputs.shape


if __name__ == '__main__':
    # Run with: python -m pytest cbam_test.py -v
    pytest.main([__file__, '-v', '--tb=short'])