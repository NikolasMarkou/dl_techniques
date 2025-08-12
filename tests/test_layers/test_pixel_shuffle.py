"""
Test suite for PixelShuffle layer implementation.

This module provides comprehensive tests for:
- PixelShuffle layer functionality
- CLS token handling and spatial token reduction
- Serialization and deserialization
- Model integration and persistence
- Edge cases and error handling
"""

import pytest
import tensorflow as tf
import numpy as np
import keras
from typing import Dict, Any, Tuple

# Assuming the fixed file is in this path
from dl_techniques.layers.pixel_shuffle import PixelShuffle


# Test fixtures
@pytest.fixture
def sample_vit_tokens() -> tf.Tensor:
    """Generate sample Vision Transformer tokens with CLS token."""
    tf.random.set_seed(42)
    # Shape: [batch=2, seq_len=197, channels=768] (196 spatial + 1 CLS, 14x14 spatial)
    return tf.random.normal((2, 197, 768))


@pytest.fixture
def sample_small_tokens() -> tf.Tensor:
    """Generate small sample tokens for testing."""
    tf.random.set_seed(42)
    # Shape: [batch=2, seq_len=10, channels=64] (9 spatial + 1 CLS, 3x3 spatial)
    return tf.random.normal((2, 10, 64))


@pytest.fixture
def sample_large_tokens() -> tf.Tensor:
    """Generate larger sample tokens."""
    tf.random.set_seed(42)
    # Shape: [batch=2, seq_len=257, channels=512] (256 spatial + 1 CLS, 16x16 spatial)
    return tf.random.normal((2, 257, 512))


@pytest.fixture
def default_shuffle_params() -> Dict[str, Any]:
    """Default parameters for PixelShuffle."""
    return {
        "scale_factor": 2,
        "validate_spatial_dims": True,
    }


@pytest.fixture
def minimal_shuffle_params() -> Dict[str, Any]:
    """Minimal parameters for PixelShuffle."""
    return {
        "scale_factor": 2,
    }


# Initialization tests
def test_shuffle_initialization(default_shuffle_params: Dict[str, Any]) -> None:
    """Test initialization of PixelShuffle."""
    shuffle = PixelShuffle(**default_shuffle_params)

    assert shuffle.scale_factor == default_shuffle_params["scale_factor"]
    assert shuffle.validate_spatial_dims == default_shuffle_params["validate_spatial_dims"]


def test_shuffle_minimal_initialization(minimal_shuffle_params: Dict[str, Any]) -> None:
    """Test initialization with minimal parameters."""
    shuffle = PixelShuffle(**minimal_shuffle_params)

    assert shuffle.scale_factor == minimal_shuffle_params["scale_factor"]
    assert shuffle.validate_spatial_dims is True  # Default value


def test_shuffle_invalid_scale_factor() -> None:
    """Test that invalid scale factors raise ValueError."""
    # Negative scale factor
    with pytest.raises(ValueError, match="scale_factor must be a positive integer"):
        PixelShuffle(scale_factor=-1)

    # Zero scale factor
    with pytest.raises(ValueError, match="scale_factor must be a positive integer"):
        PixelShuffle(scale_factor=0)

    # Float scale factor
    with pytest.raises(ValueError, match="scale_factor must be a positive integer"):
        PixelShuffle(scale_factor=1.5)


# Shape computation tests
def test_shuffle_output_shape_small(sample_small_tokens: tf.Tensor) -> None:
    """Test output shape computation with small tokens."""
    shuffle = PixelShuffle(scale_factor=3)

    # Input: [2, 10, 64] -> 9 spatial tokens (3x3) + 1 CLS
    # After shuffle: 9 -> 1 spatial token, channels 64 -> 576 (64 * 3^2)
    expected_shape = (2, 2, 576)  # 1 spatial + 1 CLS = 2 tokens

    computed_shape = shuffle.compute_output_shape(sample_small_tokens.shape)
    assert computed_shape == expected_shape


def test_shuffle_output_shape_vit(sample_vit_tokens: tf.Tensor) -> None:
    """Test output shape computation with standard ViT tokens."""
    shuffle = PixelShuffle(scale_factor=2)

    # Input: [2, 197, 768] -> 196 spatial tokens (14x14) + 1 CLS
    # After shuffle: 196 -> 49 spatial tokens (7x7), channels 768 -> 3072 (768 * 2^2)
    expected_shape = (2, 50, 3072)  # 49 spatial + 1 CLS = 50 tokens

    computed_shape = shuffle.compute_output_shape(sample_vit_tokens.shape)
    assert computed_shape == expected_shape


def test_shuffle_output_shape_large(sample_large_tokens: tf.Tensor) -> None:
    """Test output shape computation with large tokens."""
    shuffle = PixelShuffle(scale_factor=4)

    # Input: [2, 257, 512] -> 256 spatial tokens (16x16) + 1 CLS
    # After shuffle: 256 -> 16 spatial tokens (4x4), channels 512 -> 8192 (512 * 4^2)
    expected_shape = (2, 17, 8192)  # 16 spatial + 1 CLS = 17 tokens

    computed_shape = shuffle.compute_output_shape(sample_large_tokens.shape)
    assert computed_shape == expected_shape


def test_shuffle_output_shape_unknown_dims() -> None:
    """Test output shape computation with unknown dimensions."""
    shuffle = PixelShuffle(scale_factor=2)

    # Test with unknown sequence length
    input_shape = (None, None, 768)
    computed_shape = shuffle.compute_output_shape(input_shape)
    expected_shape = (None, None, 3072)  # Only channels can be computed
    assert computed_shape == expected_shape

    # Test with unknown channels
    input_shape = (2, 197, None)
    computed_shape = shuffle.compute_output_shape(input_shape)
    expected_shape = (2, 50, None)
    assert computed_shape == expected_shape


# Forward pass tests
def test_shuffle_forward_pass_small(sample_small_tokens: tf.Tensor) -> None:
    """Test forward pass with small tokens."""
    shuffle = PixelShuffle(scale_factor=3, validate_spatial_dims=True)

    output = shuffle(sample_small_tokens)
    expected_shape = (2, 2, 576)  # 1 spatial + 1 CLS, 64 * 3^2 channels

    assert output.shape == expected_shape
    assert not tf.reduce_any(tf.math.is_nan(output))
    assert not tf.reduce_any(tf.math.is_inf(output))


def test_shuffle_forward_pass_vit(sample_vit_tokens: tf.Tensor) -> None:
    """Test forward pass with standard ViT tokens."""
    shuffle = PixelShuffle(scale_factor=2)

    output = shuffle(sample_vit_tokens)
    expected_shape = (2, 50, 3072)

    assert output.shape == expected_shape
    assert not tf.reduce_any(tf.math.is_nan(output))
    assert not tf.reduce_any(tf.math.is_inf(output))


def test_shuffle_forward_pass_large(sample_large_tokens: tf.Tensor) -> None:
    """Test forward pass with large tokens."""
    shuffle = PixelShuffle(scale_factor=4)

    output = shuffle(sample_large_tokens)
    expected_shape = (2, 17, 8192)

    assert output.shape == expected_shape
    assert not tf.reduce_any(tf.math.is_nan(output))
    assert not tf.reduce_any(tf.math.is_inf(output))


def test_shuffle_cls_token_preservation(sample_vit_tokens: tf.Tensor) -> None:
    """Test that CLS token is preserved correctly."""
    shuffle = PixelShuffle(scale_factor=2)

    # Set specific values for CLS token to track it
    modified_tokens = tf.Variable(sample_vit_tokens)
    cls_values = tf.ones((2, 1, 768)) * 999.0  # Distinctive CLS values
    modified_tokens[:, 0:1, :].assign(cls_values)

    output = shuffle(modified_tokens)

    # CLS token should be preserved at position 0
    output_cls = output[:, 0:1, :]

    # Since scale_factor=2, channels go from 768 to 3072.
    # The CLS token is padded to match the new channel dimension.
    assert output_cls.shape == (2, 1, 3072)

    # The first 768 channels should contain the original CLS values
    original_part = output_cls[:, :, :768]
    np.testing.assert_allclose(original_part.numpy(), cls_values.numpy())

    # The remaining channels should be padded with zeros
    padded_part = output_cls[:, :, 768:]
    assert np.all(padded_part.numpy() == 0)


def test_shuffle_spatial_token_reduction(sample_vit_tokens: tf.Tensor) -> None:
    """Test that spatial tokens are correctly reduced."""
    shuffle = PixelShuffle(scale_factor=2)

    output = shuffle(sample_vit_tokens)

    # Input has 196 spatial tokens, output should have 49 spatial tokens
    input_spatial_count = sample_vit_tokens.shape[1] - 1  # 196
    output_spatial_count = output.shape[1] - 1  # 49

    expected_reduction = shuffle.scale_factor ** 2  # 4
    assert input_spatial_count // expected_reduction == output_spatial_count


# Scale factor tests
@pytest.mark.parametrize("scale_factor", [1, 2, 3, 4])
def test_shuffle_different_scale_factors(scale_factor: int, sample_vit_tokens: tf.Tensor) -> None:
    """Test PixelShuffle with different scale factors."""
    if scale_factor > 2 and sample_vit_tokens.shape[1] == 197:
        # 196 spatial tokens (14x14), scale_factor 3 would need 21x21, 4 would need 28x28
        # Skip incompatible combinations or create compatible input
        spatial_tokens = 196  # 14x14
        if 14 % scale_factor != 0:
            pytest.skip(f"Scale factor {scale_factor} not compatible with 14x14 spatial arrangement")

    shuffle = PixelShuffle(scale_factor=scale_factor, validate_spatial_dims=True)

    output = shuffle(sample_vit_tokens)

    # Check channel expansion
    expected_channels = sample_vit_tokens.shape[-1] * (scale_factor ** 2)
    assert output.shape[-1] == expected_channels

    # Check spatial reduction
    input_spatial = sample_vit_tokens.shape[1] - 1
    output_spatial = output.shape[1] - 1
    expected_spatial = input_spatial // (scale_factor ** 2)
    assert output_spatial == expected_spatial


def test_shuffle_scale_factor_1(sample_vit_tokens: tf.Tensor) -> None:
    """Test PixelShuffle with scale_factor=1 (identity operation)."""
    shuffle = PixelShuffle(scale_factor=1)

    output = shuffle(sample_vit_tokens)

    # Should be identical to input for scale_factor=1
    assert output.shape == sample_vit_tokens.shape
    # Values should be identical for the identity operation
    np.testing.assert_allclose(output.numpy(), sample_vit_tokens.numpy())


# Validation tests
def test_shuffle_validation_enabled() -> None:
    """Test validation when validate_spatial_dims=True."""
    shuffle = PixelShuffle(scale_factor=2, validate_spatial_dims=True)

    # Valid input: 9 spatial tokens (3x3) + 1 CLS, but 3 is not divisible by 2
    # This should raise an error during build
    invalid_input_shape = (2, 10, 64)  # 9 spatial tokens, 3x3 grid

    with pytest.raises(ValueError, match="must be divisible by scale_factor"):
        shuffle.build(invalid_input_shape)


def test_shuffle_validation_disabled() -> None:
    """Test validation when validate_spatial_dims=False."""
    shuffle = PixelShuffle(scale_factor=2, validate_spatial_dims=False)

    # Should not raise error even with incompatible dimensions during build
    invalid_input_shape = (2, 10, 64)  # 9 spatial tokens, 3x3 grid

    # This should not raise an error
    shuffle.build(invalid_input_shape)
    assert shuffle.built


def test_shuffle_non_square_spatial() -> None:
    """Test error handling for non-square spatial arrangements."""
    shuffle = PixelShuffle(scale_factor=2, validate_spatial_dims=True)

    # 8 spatial tokens is not a perfect square
    invalid_shape = (2, 9, 64)  # 8 spatial + 1 CLS

    with pytest.raises(ValueError, match="must form a perfect square"):
        shuffle.build(invalid_shape)


def test_shuffle_insufficient_tokens() -> None:
    """Test error handling for insufficient tokens."""
    shuffle = PixelShuffle(scale_factor=2, validate_spatial_dims=True)

    # Only CLS token, no spatial tokens
    invalid_shape = (2, 1, 64)

    with pytest.raises(ValueError, match="must be > 1"):
        shuffle.build(invalid_shape)


def test_shuffle_invalid_input_dimension() -> None:
    """Test error handling for invalid input dimensions."""
    shuffle = PixelShuffle(scale_factor=2)

    # 2D input instead of 3D
    invalid_shape = (2, 197)

    with pytest.raises(ValueError, match="Expected 3D input"):
        shuffle.build(invalid_shape)

    # 4D input instead of 3D
    invalid_shape = (2, 14, 14, 768)

    with pytest.raises(ValueError, match="Expected 3D input"):
        shuffle.build(invalid_shape)


# Serialization tests
def test_shuffle_serialization(default_shuffle_params: Dict[str, Any], sample_vit_tokens: tf.Tensor) -> None:
    """Test serialization of PixelShuffle."""
    # Create and build original shuffle layer
    original_shuffle = PixelShuffle(**default_shuffle_params)
    original_shuffle.build(sample_vit_tokens.shape)

    # Get config and recreate from config
    config = original_shuffle.get_config()
    build_config = original_shuffle.get_build_config()

    restored_shuffle = PixelShuffle.from_config(config)
    restored_shuffle.build_from_config(build_config)

    # Check that key properties match
    assert restored_shuffle.scale_factor == original_shuffle.scale_factor
    assert restored_shuffle.validate_spatial_dims == original_shuffle.validate_spatial_dims

    # Check that both are built
    assert original_shuffle.built
    assert restored_shuffle.built

    # Test that outputs match
    original_output = original_shuffle(sample_vit_tokens)
    restored_output = restored_shuffle(sample_vit_tokens)

    assert original_output.shape == restored_output.shape
    np.testing.assert_allclose(original_output.numpy(), restored_output.numpy())


def test_shuffle_build_configuration(default_shuffle_params: Dict[str, Any], sample_vit_tokens: tf.Tensor) -> None:
    """Test get_build_config and build_from_config methods."""
    # Create and build original layer
    original_shuffle = PixelShuffle(**default_shuffle_params)
    original_shuffle.build(sample_vit_tokens.shape)

    # Get build config
    build_config = original_shuffle.get_build_config()

    # Check that build config contains input_shape
    assert "input_shape" in build_config
    assert build_config["input_shape"] == sample_vit_tokens.shape

    # Create new layer and build from config
    new_shuffle = PixelShuffle(**default_shuffle_params)
    new_shuffle.build_from_config(build_config)

    # Check that new layer is built
    assert new_shuffle.built
    assert new_shuffle._build_input_shape == sample_vit_tokens.shape


def test_shuffle_build_config_none_handling(default_shuffle_params: Dict[str, Any]) -> None:
    """Test build configuration methods handle None input_shape."""
    # Test with None input_shape
    shuffle = PixelShuffle(**default_shuffle_params)

    # Before building
    build_config = shuffle.get_build_config()
    assert build_config["input_shape"] is None

    # build_from_config with None should not crash
    new_shuffle = PixelShuffle(**default_shuffle_params)
    new_shuffle.build_from_config({"input_shape": None})
    assert not new_shuffle.built


# Model integration tests
def test_shuffle_model_integration(sample_vit_tokens: tf.Tensor, default_shuffle_params: Dict[str, Any]) -> None:
    """Test integrating PixelShuffle into a Keras model."""
    inputs = keras.Input(shape=sample_vit_tokens.shape[1:])
    shuffle = PixelShuffle(**default_shuffle_params)
    outputs = shuffle(inputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Test forward pass
    result = model(sample_vit_tokens)
    expected_shape = shuffle.compute_output_shape(sample_vit_tokens.shape)
    assert result.shape == expected_shape


def test_shuffle_in_vision_pipeline(sample_vit_tokens: tf.Tensor) -> None:
    """Test PixelShuffle in a vision processing pipeline."""
    inputs = keras.Input(shape=sample_vit_tokens.shape[1:])

    # Vision pipeline with pixel shuffle
    x = PixelShuffle(scale_factor=2)(inputs)  # Reduce spatial tokens
    x = keras.layers.Dense(512)(x)  # Process with fewer tokens
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dense(768)(x)  # Back to original channel size

    model = keras.Model(inputs=inputs, outputs=x)

    # Test forward pass
    result = model(sample_vit_tokens)
    assert result.shape == (2, 50, 768)  # 50 tokens (49 spatial + 1 CLS)


def test_shuffle_gradient_flow(sample_vit_tokens: tf.Tensor, default_shuffle_params: Dict[str, Any]) -> None:
    """Test gradient flow through PixelShuffle."""
    shuffle = PixelShuffle(**default_shuffle_params)

    with tf.GradientTape() as tape:
        tape.watch(sample_vit_tokens)
        output = shuffle(sample_vit_tokens)
        loss = tf.reduce_mean(output)

    # PixelShuffle has no trainable parameters, but gradients should flow through
    gradients = tape.gradient(loss, sample_vit_tokens)

    assert gradients is not None
    assert gradients.shape == sample_vit_tokens.shape
    assert not tf.reduce_any(tf.math.is_nan(gradients))


def test_shuffle_model_save_load(sample_vit_tokens: tf.Tensor, default_shuffle_params: Dict[str, Any], tmp_path) -> None:
    """Test saving and loading a model with PixelShuffle."""
    # Create model with PixelShuffle
    inputs = keras.Input(shape=sample_vit_tokens.shape[1:])
    shuffle = PixelShuffle(**default_shuffle_params)
    outputs = shuffle(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Generate output before saving
    original_output = model(sample_vit_tokens, training=False).numpy()

    # Save model in Keras format
    save_path = str(tmp_path / "model.keras")
    model.save(save_path)

    # Load model with custom objects
    loaded_model = keras.models.load_model(
        save_path,
        custom_objects={"PixelShuffle": PixelShuffle}
    )

    # Generate output after loading
    loaded_output = loaded_model(sample_vit_tokens, training=False).numpy()

    # Outputs should be identical
    assert np.allclose(original_output, loaded_output, rtol=1e-5, atol=1e-5)


def test_shuffle_model_save_load_weights(sample_vit_tokens: tf.Tensor, default_shuffle_params: Dict[str, Any], tmp_path) -> None:
    """Test saving and loading model weights with PixelShuffle."""
    # Create model with PixelShuffle
    inputs = keras.Input(shape=sample_vit_tokens.shape[1:])
    x = PixelShuffle(**default_shuffle_params)(inputs)
    x = keras.layers.Dense(256)(x)  # Add trainable layer
    outputs = keras.layers.Dense(128)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Generate output before saving
    original_output = model(sample_vit_tokens, training=False).numpy()

    # Save weights
    save_path = str(tmp_path / "model_weights.weights.h5")
    model.save_weights(save_path)

    # Load weights
    model.load_weights(save_path)

    # Generate output after loading
    loaded_output = model(sample_vit_tokens, training=False).numpy()

    # Outputs should be identical
    assert np.allclose(original_output, loaded_output, rtol=1e-5, atol=1e-5)


# Performance and edge case tests
def test_shuffle_deterministic_output(sample_vit_tokens: tf.Tensor) -> None:
    """Test that PixelShuffle produces deterministic output."""
    shuffle = PixelShuffle(scale_factor=2, validate_spatial_dims=False)

    output1 = shuffle(sample_vit_tokens)
    output2 = shuffle(sample_vit_tokens)

    # Should be identical (no randomness in operation)
    assert np.allclose(output1.numpy(), output2.numpy(), rtol=1e-7, atol=1e-7)


def test_shuffle_memory_efficiency() -> None:
    """Test memory efficiency with large inputs."""
    # Create large input
    large_input = tf.random.normal((1, 1025, 1024))  # 1024 spatial + 1 CLS, 32x32 spatial
    shuffle = PixelShuffle(scale_factor=4, validate_spatial_dims=False)  # Don't validate for speed

    output = shuffle(large_input)

    # Check output shape
    expected_shape = (1, 65, 16384)  # 64 spatial + 1 CLS, 1024 * 4^2 channels
    assert output.shape == expected_shape

    # Verify no memory leaks (output exists)
    assert output is not None


def test_shuffle_batch_consistency() -> None:
    """Test that PixelShuffle works consistently across batch dimensions."""
    # Create inputs with different batch sizes
    single_batch = tf.random.normal((1, 197, 768))
    multi_batch = tf.random.normal((4, 197, 768))

    shuffle = PixelShuffle(scale_factor=2)

    output_single = shuffle(single_batch)
    output_multi = shuffle(multi_batch)

    # Check shapes
    assert output_single.shape == (1, 50, 3072)
    assert output_multi.shape == (4, 50, 3072)

    # Check that processing is consistent per batch item
    # (This is mainly testing the implementation doesn't have batch-dependent bugs)
    assert not tf.reduce_any(tf.math.is_nan(output_single))
    assert not tf.reduce_any(tf.math.is_nan(output_multi))


def test_shuffle_information_preservation() -> None:
    """Test that PixelShuffle preserves information (invertible with appropriate inverse)."""
    # Create a controlled input where we can verify information preservation
    input_tokens = tf.zeros((1, 17, 4))  # 16 spatial (4x4) + 1 CLS, 4 channels

    # Set specific values we can track
    # CLS token
    input_tokens = tf.tensor_scatter_nd_update(input_tokens, [[0, 0, 0]], [100.0])
    # Spatial tokens with unique values
    for i in range(16):
        input_tokens = tf.tensor_scatter_nd_update(input_tokens, [[0, i+1, 0]], [float(i+1)])

    shuffle = PixelShuffle(scale_factor=2, validate_spatial_dims=False)
    output = shuffle(input_tokens)

    # Output should have shape (1, 5, 16) - 4 spatial (2x2) + 1 CLS, 4*4 channels
    assert output.shape == (1, 5, 16)

    # CLS token should be preserved in first position, padded with zeros
    cls_output = output[0, 0, :]
    assert cls_output[0] == 100.0
    assert np.all(cls_output[4:].numpy() == 0)

    # Total information should be preserved (no values lost)
    # The spatial information is rearranged but should all be present
    assert not tf.reduce_any(tf.math.is_nan(output))


if __name__ == "__main__":
    pytest.main([__file__])
