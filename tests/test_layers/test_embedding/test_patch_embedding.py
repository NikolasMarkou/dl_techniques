import pytest
import numpy as np
import tempfile
import os
from typing import Any, Dict

import keras
from keras import ops
import tensorflow as tf

from dl_techniques.layers.embedding.patch_embedding import (
    PatchEmbedding2D, PatchEmbedding1D
)


class TestPatchEmbedding2D:
    """Comprehensive test suite for PatchEmbedding2D following modern Keras 3 patterns."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create sample input tensor using keras.random."""
        return keras.random.normal([4, 224, 224, 3])

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'patch_size': 16,
            'embed_dim': 768
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = PatchEmbedding2D(patch_size=16, embed_dim=512)

        # Verify attributes are set correctly
        assert layer.patch_size == (16, 16)
        assert layer.embed_dim == 512
        assert layer.use_bias is True
        assert not layer.built  # Should not be built yet

        # Sub-layers should be created in __init__
        assert hasattr(layer, 'proj')
        assert layer.proj is not None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_config = {
            'patch_size': (8, 16),
            'embed_dim': 256,
            'kernel_initializer': 'he_normal',
            'kernel_regularizer': keras.regularizers.L2(1e-4),
            'bias_initializer': 'ones',
            'bias_regularizer': keras.regularizers.L1(1e-5),
            'activation': 'relu',
            'use_bias': False
        }

        layer = PatchEmbedding2D(**custom_config)

        # Verify custom values
        assert layer.patch_size == (8, 16)
        assert layer.embed_dim == 256
        assert layer.use_bias is False

    def test_forward_pass(self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor):
        """Test forward pass and building."""
        layer = PatchEmbedding2D(**layer_config)

        output = layer(sample_input)

        # Layer should be built after first call
        assert layer.built

        # Check output shape
        expected_num_patches = (224 // 16) * (224 // 16)  # 14 * 14 = 196
        assert output.shape == (4, expected_num_patches, 768)

        # Check for numerical stability
        output_np = ops.convert_to_numpy(output)
        assert not np.any(np.isnan(output_np))
        assert not np.any(np.isinf(output_np))

    def test_serialization_cycle(self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor):
        """CRITICAL TEST: Full serialization cycle following guide pattern."""
        # 1. Create original layer in a model
        inputs = keras.Input(shape=sample_input.shape[1:])
        layer_output = PatchEmbedding2D(**layer_config)(inputs)
        model = keras.Model(inputs, layer_output)

        # 2. Get prediction from original
        original_prediction = model(sample_input)

        # 3. Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            # Load without custom_objects since layer is properly registered
            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            # 4. Verify identical outputs using recommended method
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_config_completeness(self, layer_config: Dict[str, Any]):
        """Test that get_config contains all __init__ parameters."""
        layer = PatchEmbedding2D(**layer_config)
        config = layer.get_config()

        # Check all config parameters are present
        for key in layer_config:
            assert key in config, f"Missing {key} in get_config()"

        # Check essential configuration keys
        essential_keys = ['patch_size', 'embed_dim', 'use_bias']
        for key in essential_keys:
            assert key in config, f"Missing essential key {key} in get_config()"

    def test_build_invalid_input_shape(self):
        """Test build with invalid input shapes."""
        layer = PatchEmbedding2D(patch_size=16, embed_dim=768)

        # Test with invalid dimensions
        with pytest.raises(ValueError, match="Expected 4D input"):
            layer.build((32, 224, 224))

        # Non-divisible dimensions should fail
        with pytest.raises(ValueError):
            layer.build((4, 225, 224, 3))  # 225 not divisible by 16

    def test_gradients_flow(self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor):
        """Test gradient computation using TensorFlow GradientTape as specified."""
        layer = PatchEmbedding2D(**layer_config)

        with tf.GradientTape() as tape:
            # Convert to TensorFlow Variable for gradient tracking
            tf_input = tf.Variable(ops.convert_to_numpy(sample_input))
            output = layer(tf_input)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor, training):
        """Test behavior in different training modes."""
        layer = PatchEmbedding2D(**layer_config)

        output = layer(sample_input, training=training)
        assert output.shape[0] == sample_input.shape[0]

    def test_edge_cases(self):
        """Test error conditions and edge cases."""
        with pytest.raises(ValueError):
            PatchEmbedding2D(patch_size=0, embed_dim=32)  # Invalid patch_size

        with pytest.raises(ValueError):
            PatchEmbedding2D(patch_size=16, embed_dim=-5)  # Invalid embed_dim

    def test_compute_output_shape(self):
        """Test compute_output_shape method."""
        layer = PatchEmbedding2D(patch_size=16, embed_dim=768)

        input_shape = (None, 224, 224, 3)
        output_shape = layer.compute_output_shape(input_shape)

        expected_num_patches = (224 // 16) * (224 // 16)
        expected_shape = (None, expected_num_patches, 768)

        assert output_shape == expected_shape

    def test_different_patch_sizes(self):
        """Test with various patch sizes."""
        test_configs = [
            (16, 768, 224, 224),  # Standard ViT-Base
            (32, 512, 224, 224),  # Larger patches
            (8, 256, 224, 224),  # Smaller patches
        ]

        for patch_size, embed_dim, height, width in test_configs:
            layer = PatchEmbedding2D(patch_size=patch_size, embed_dim=embed_dim)
            test_input = keras.random.normal([2, height, width, 3])

            output = layer(test_input)

            expected_num_patches = (height // patch_size) * (width // patch_size)
            expected_shape = (2, expected_num_patches, embed_dim)

            assert output.shape == expected_shape

    def test_rectangular_patches(self):
        """Test with non-square patches."""
        layer = PatchEmbedding2D(patch_size=(8, 16), embed_dim=256)

        # Input divisible by patch dimensions
        test_input = keras.random.normal([2, 64, 128, 3])  # 64/8=8, 128/16=8
        output = layer(test_input)

        # Expected: 8 * 8 = 64 patches
        expected_shape = (2, 64, 256)
        assert output.shape == expected_shape

    def test_activations(self):
        """Test with different activation functions."""
        activations = ['relu', 'gelu', 'swish', None]

        for activation in activations:
            layer = PatchEmbedding2D(patch_size=16, embed_dim=256, activation=activation)
            test_input = keras.random.normal([2, 64, 64, 3])

            output = layer(test_input)

            assert output.shape == (2, 16, 256)  # 4*4 patches
            output_np = ops.convert_to_numpy(output)
            assert not np.any(np.isnan(output_np))

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        layer = PatchEmbedding2D(patch_size=16, embed_dim=256)

        # Test with extreme values
        test_cases = [
            ops.zeros((2, 64, 64, 3)),
            ops.ones((2, 64, 64, 3)) * 1e-10,
            ops.ones((2, 64, 64, 3)) * 1e5,
        ]

        for test_input in test_cases:
            output = layer(test_input)
            output_np = ops.convert_to_numpy(output)

            assert not np.any(np.isnan(output_np)), "NaN values detected"
            assert not np.any(np.isinf(output_np)), "Inf values detected"


class TestPatchEmbedding1D:
    """Comprehensive test suite for PatchEmbedding1D following modern Keras 3 patterns."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create sample input tensor."""
        return keras.random.normal([4, 128, 64])

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'patch_size': 16,
            'embed_dim': 256
        }

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = PatchEmbedding1D(patch_size=16, embed_dim=256)

        assert layer.patch_size == 16
        assert layer.embed_dim == 256
        assert layer.stride == 16  # Should default to patch_size
        assert layer.padding == 'causal'
        assert layer.use_bias is True
        assert not layer.built

        # Sub-layer should be created in __init__
        assert hasattr(layer, 'embedding')
        assert layer.embedding is not None

    def test_forward_pass(self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor):
        """Test forward pass and building."""
        layer = PatchEmbedding1D(**layer_config)

        output = layer(sample_input)

        assert layer.built
        assert len(output.shape) == 3
        assert output.shape[0] == sample_input.shape[0]  # batch size preserved
        assert output.shape[2] == 256  # embed_dim

        # Check numerical stability
        output_np = ops.convert_to_numpy(output)
        assert not np.any(np.isnan(output_np))
        assert not np.any(np.isinf(output_np))

    def test_serialization_cycle(self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor):
        """CRITICAL TEST: Full serialization cycle."""
        # 1. Create original layer in a model
        inputs = keras.Input(shape=sample_input.shape[1:])
        layer_output = PatchEmbedding1D(**layer_config)(inputs)
        model = keras.Model(inputs, layer_output)

        # 2. Get prediction from original
        original_prediction = model(sample_input)

        # 3. Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            # 4. Verify identical outputs
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_config_completeness(self, layer_config: Dict[str, Any]):
        """Test that get_config contains all __init__ parameters."""
        layer = PatchEmbedding1D(**layer_config)
        config = layer.get_config()

        # Check all config parameters are present
        for key in layer_config:
            assert key in config, f"Missing {key} in get_config()"

        # Check essential configuration keys
        essential_keys = ['patch_size', 'embed_dim', 'stride', 'padding', 'use_bias']
        for key in essential_keys:
            assert key in config, f"Missing essential key {key} in get_config()"

    def test_padding_modes(self):
        """Test different padding modes and their output shapes."""
        seq_len = 128
        patch_size = 16
        embed_dim = 256
        test_input = keras.random.normal([2, seq_len, 32])

        # Test different padding modes
        padding_modes = ['valid', 'same', 'causal']

        for padding in padding_modes:
            layer = PatchEmbedding1D(
                patch_size=patch_size,
                embed_dim=embed_dim,
                padding=padding
            )
            output = layer(test_input)

            # All should produce valid outputs
            assert output.shape[0] == 2  # batch size
            assert output.shape[2] == embed_dim  # embedding dimension
            assert output.shape[1] > 0  # sequence length

    def test_overlapping_patches(self):
        """Test with overlapping patches (stride < patch_size)."""
        layer = PatchEmbedding1D(
            patch_size=16,
            embed_dim=256,
            stride=8,
            padding='valid'
        )

        test_input = keras.random.normal([2, 128, 32])
        output = layer(test_input)

        # With stride=8, patch_size=16, we get overlapping patches
        expected_len = (128 - 16) // 8 + 1
        assert output.shape == (2, expected_len, 256)

    def test_gradients_flow(self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor):
        """Test gradient computation."""
        layer = PatchEmbedding1D(**layer_config)

        with tf.GradientTape() as tape:
            tf_input = tf.Variable(ops.convert_to_numpy(sample_input))
            output = layer(tf_input)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

    def test_edge_cases(self):
        """Test error conditions."""
        with pytest.raises(ValueError):
            PatchEmbedding1D(patch_size=0, embed_dim=256)  # Invalid patch_size

        with pytest.raises(ValueError):
            PatchEmbedding1D(patch_size=16, embed_dim=-5)  # Invalid embed_dim

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        layer = PatchEmbedding1D(patch_size=8, embed_dim=128, padding='causal')

        # Test with different sequence lengths
        lengths = [32, 64, 128, 256]

        for length in lengths:
            test_input = keras.random.normal([2, length, 16])
            output = layer(test_input)

            expected_patches = length // 8
            assert output.shape == (2, expected_patches, 128)

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor, training):
        """Test behavior in different training modes."""
        layer = PatchEmbedding1D(**layer_config)

        output = layer(sample_input, training=training)
        assert output.shape[0] == sample_input.shape[0]

    def test_compute_output_shape(self):
        """Test compute_output_shape method."""
        test_cases = [
            (PatchEmbedding1D(patch_size=8, embed_dim=128, padding='valid'), (2, 64, 16)),
            (PatchEmbedding1D(patch_size=8, embed_dim=128, padding='same'), (2, 64, 16)),
            (PatchEmbedding1D(patch_size=8, embed_dim=128, padding='causal'), (2, 64, 16)),
        ]

        for layer, input_shape in test_cases:
            output_shape = layer.compute_output_shape(input_shape)

            assert output_shape[0] == input_shape[0]  # batch size
            assert output_shape[2] == 128  # embed_dim
            assert output_shape[1] is not None  # sequence length computed

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        layer = PatchEmbedding1D(patch_size=8, embed_dim=128)

        test_cases = [
            ops.zeros((2, 64, 32)),
            ops.ones((2, 64, 32)) * 1e-10,
            ops.ones((2, 64, 32)) * 1e5,
        ]

        for test_input in test_cases:
            output = layer(test_input)
            output_np = ops.convert_to_numpy(output)

            assert not np.any(np.isnan(output_np)), "NaN values detected"
            assert not np.any(np.isinf(output_np)), "Inf values detected"


def test_both_layers_in_model():
    """Integration test using both layers in a single model."""
    # Test that both layers work together
    inputs_2d = keras.Input(shape=(64, 64, 3), name='input_2d')
    inputs_1d = keras.Input(shape=(128, 32), name='input_1d')

    # Process 2D input
    patches_2d = PatchEmbedding2D(patch_size=16, embed_dim=256)(inputs_2d)
    x2d = keras.layers.GlobalAveragePooling1D()(patches_2d)

    # Process 1D input
    patches_1d = PatchEmbedding1D(patch_size=16, embed_dim=256)(inputs_1d)
    x1d = keras.layers.GlobalAveragePooling1D()(patches_1d)

    # Combine and classify
    combined = keras.layers.Concatenate()([x2d, x1d])
    outputs = keras.layers.Dense(10, activation='softmax')(combined)

    model = keras.Model(inputs=[inputs_2d, inputs_1d], outputs=outputs)

    # Test forward pass
    test_2d = keras.random.normal([2, 64, 64, 3])
    test_1d = keras.random.normal([2, 128, 32])

    predictions = model([test_2d, test_1d])
    assert predictions.shape == (2, 10)


# Debug helper from the guide
def debug_layer_serialization(layer_class, layer_config, sample_input):
    """Debug helper for layer serialization issues."""
    from dl_techniques.utils.logger import logger

    try:
        # Test basic functionality
        layer = layer_class(**layer_config)
        output = layer(sample_input)
        logger.info(f"✅ Forward pass successful: {output.shape}")

        # Test configuration
        config = layer.get_config()
        logger.info(f"✅ Configuration keys: {list(config.keys())}")

        # Test serialization
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = layer_class(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(os.path.join(tmpdir, 'test.keras'))
            loaded = keras.models.load_model(os.path.join(tmpdir, 'test.keras'))
            logger.info("✅ Serialization test passed")

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])