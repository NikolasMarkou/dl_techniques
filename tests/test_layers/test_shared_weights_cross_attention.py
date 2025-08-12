"""
Test suite for SharedWeightsCrossAttention implementation.

This module provides comprehensive tests for:
- SharedWeightsCrossAttention layer
- Layer behavior under different configurations
- Multi-modal cross-attention functionality
- Serialization and deserialization
- Model integration and persistence
"""

import pytest
import tensorflow as tf
import numpy as np
import keras
from typing import Dict, Any, List, Tuple

from dl_techniques.layers.attention.shared_weights_cross_attention import SharedWeightsCrossAttention


# Test fixtures
@pytest.fixture
def two_modality_inputs() -> Tuple[tf.Tensor, List[int]]:
    """Generate sample input tensor for two modalities."""
    tf.random.set_seed(42)
    # Surface features: 100 tokens, Volume features: 150 tokens
    surface_features = tf.random.normal((2, 100, 256))
    volume_features = tf.random.normal((2, 150, 256))
    combined = tf.concat([surface_features, volume_features], axis=1)
    split_sizes = [100, 150]
    return combined, split_sizes


@pytest.fixture
def equal_modality_inputs() -> Tuple[tf.Tensor, List[int]]:
    """Generate sample input tensor for two equal-sized modalities."""
    tf.random.set_seed(42)
    # Equal sized modalities for testing optimization path
    mod_a_features = tf.random.normal((2, 128, 256))
    mod_b_features = tf.random.normal((2, 128, 256))
    combined = tf.concat([mod_a_features, mod_b_features], axis=1)
    split_sizes = [128, 128]
    return combined, split_sizes


@pytest.fixture
def anchor_query_inputs() -> Tuple[tf.Tensor, List[int]]:
    """Generate sample input tensor for anchor-query structure."""
    tf.random.set_seed(42)
    # Surface: 50 anchors + 50 queries, Volume: 75 anchors + 75 queries
    surface_anchors = tf.random.normal((2, 50, 256))
    surface_queries = tf.random.normal((2, 50, 256))
    volume_anchors = tf.random.normal((2, 75, 256))
    volume_queries = tf.random.normal((2, 75, 256))
    combined = tf.concat([surface_anchors, surface_queries, volume_anchors, volume_queries], axis=1)
    split_sizes = [50, 50, 75, 75]
    return combined, split_sizes


@pytest.fixture
def default_attention_params() -> Dict[str, Any]:
    """Default parameters for SharedWeightsCrossAttention."""
    return {
        "dim": 256,
        "num_heads": 8,
        "dropout_rate": 0.1,
        "use_bias": True,
        "kernel_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
    }


@pytest.fixture
def minimal_attention_params() -> Dict[str, Any]:
    """Minimal parameters for SharedWeightsCrossAttention."""
    return {
        "dim": 256,
    }


# SharedWeightsCrossAttention tests
def test_attention_initialization(default_attention_params: Dict[str, Any]) -> None:
    """Test initialization of SharedWeightsCrossAttention."""
    attention = SharedWeightsCrossAttention(**default_attention_params)

    assert attention.dim == default_attention_params["dim"]
    assert attention.num_heads == default_attention_params["num_heads"]
    assert attention.head_dim == default_attention_params["dim"] // default_attention_params["num_heads"]
    assert attention.dropout_rate == default_attention_params["dropout_rate"]
    assert attention.use_bias == default_attention_params["use_bias"]
    assert attention.scale == 1.0 / np.sqrt(float(attention.head_dim))


def test_attention_minimal_initialization(minimal_attention_params: Dict[str, Any]) -> None:
    """Test initialization with minimal parameters."""
    attention = SharedWeightsCrossAttention(**minimal_attention_params)

    assert attention.dim == minimal_attention_params["dim"]
    assert attention.num_heads == 8  # default
    assert attention.head_dim == 256 // 8
    assert attention.dropout_rate == 0.0  # default
    assert attention.use_bias is True  # default


def test_attention_invalid_dim_heads_ratio() -> None:
    """Test initialization with invalid dim/num_heads ratio."""
    with pytest.raises(ValueError, match="dim .* must be divisible by num_heads"):
        SharedWeightsCrossAttention(dim=257, num_heads=8)


def test_attention_two_modality_forward_pass(
        two_modality_inputs: Tuple[tf.Tensor, List[int]],
        default_attention_params: Dict[str, Any]
) -> None:
    """Test forward pass with two modalities."""
    combined_input, split_sizes = two_modality_inputs
    attention = SharedWeightsCrossAttention(**default_attention_params)

    output = attention(combined_input, split_sizes=split_sizes)

    # Output should have same shape as input
    assert output.shape == combined_input.shape
    assert not tf.reduce_any(tf.math.is_nan(output))
    assert not tf.reduce_any(tf.math.is_inf(output))


def test_attention_equal_modality_forward_pass(
        equal_modality_inputs: Tuple[tf.Tensor, List[int]],
        default_attention_params: Dict[str, Any]
) -> None:
    """Test forward pass with equal-sized modalities (optimized path)."""
    combined_input, split_sizes = equal_modality_inputs
    attention = SharedWeightsCrossAttention(**default_attention_params)

    output = attention(combined_input, split_sizes=split_sizes)

    # Output should have same shape as input
    assert output.shape == combined_input.shape
    assert not tf.reduce_any(tf.math.is_nan(output))
    assert not tf.reduce_any(tf.math.is_inf(output))


def test_attention_anchor_query_forward_pass(
        anchor_query_inputs: Tuple[tf.Tensor, List[int]],
        default_attention_params: Dict[str, Any]
) -> None:
    """Test forward pass with anchor-query structure."""
    combined_input, split_sizes = anchor_query_inputs
    attention = SharedWeightsCrossAttention(**default_attention_params)

    output = attention(combined_input, split_sizes=split_sizes)

    # Output should have same shape as input
    assert output.shape == combined_input.shape
    assert not tf.reduce_any(tf.math.is_nan(output))
    assert not tf.reduce_any(tf.math.is_inf(output))


def test_attention_compute_output_shape(
        two_modality_inputs: Tuple[tf.Tensor, List[int]],
        default_attention_params: Dict[str, Any]
) -> None:
    """Test compute_output_shape method."""
    combined_input, _ = two_modality_inputs
    attention = SharedWeightsCrossAttention(**default_attention_params)

    computed_shape = attention.compute_output_shape(combined_input.shape)
    assert computed_shape == combined_input.shape


def test_attention_training_behavior(
        two_modality_inputs: Tuple[tf.Tensor, List[int]],
        default_attention_params: Dict[str, Any]
) -> None:
    """Test attention behavior in training vs inference modes."""
    combined_input, split_sizes = two_modality_inputs
    attention = SharedWeightsCrossAttention(**default_attention_params)

    # Training mode
    train_output = attention(combined_input, split_sizes=split_sizes, training=True)

    # Inference mode
    inference_output = attention(combined_input, split_sizes=split_sizes, training=False)

    # Outputs should be different due to dropout
    assert not np.allclose(train_output.numpy(), inference_output.numpy())


def test_attention_without_dropout(
        two_modality_inputs: Tuple[tf.Tensor, List[int]],
        default_attention_params: Dict[str, Any]
) -> None:
    """Test attention without dropout."""
    combined_input, split_sizes = two_modality_inputs
    no_dropout_params = default_attention_params.copy()
    no_dropout_params["dropout_rate"] = 0.0

    attention = SharedWeightsCrossAttention(**no_dropout_params)

    # Training and inference outputs should be the same when no dropout is used
    train_output = attention(combined_input, split_sizes=split_sizes, training=True)
    inference_output = attention(combined_input, split_sizes=split_sizes, training=False)

    assert np.allclose(train_output.numpy(), inference_output.numpy(), rtol=1e-5, atol=1e-5)


def test_attention_serialization(
        default_attention_params: Dict[str, Any],
        two_modality_inputs: Tuple[tf.Tensor, List[int]]
) -> None:
    """Test serialization of SharedWeightsCrossAttention."""
    combined_input, _ = two_modality_inputs

    # Create and build the original attention layer
    original_attention = SharedWeightsCrossAttention(**default_attention_params)
    original_attention.build(combined_input.shape)

    # Get config and recreate from config
    config = original_attention.get_config()
    build_config = original_attention.get_build_config()

    restored_attention = SharedWeightsCrossAttention.from_config(config)
    restored_attention.build_from_config(build_config)

    # Check if the key properties match
    assert restored_attention.dim == original_attention.dim
    assert restored_attention.num_heads == original_attention.num_heads
    assert restored_attention.head_dim == original_attention.head_dim
    assert restored_attention.dropout_rate == original_attention.dropout_rate
    assert restored_attention.use_bias == original_attention.use_bias

    # Check that both layers are built
    assert original_attention.built
    assert restored_attention.built

    # Check that restored layer's config has the same keys as original
    restored_config = restored_attention.get_config()
    assert set(restored_config.keys()) == set(config.keys())


def test_attention_build_configuration(
        default_attention_params: Dict[str, Any],
        two_modality_inputs: Tuple[tf.Tensor, List[int]]
) -> None:
    """Test get_build_config and build_from_config methods."""
    combined_input, _ = two_modality_inputs

    # Create and build the original attention layer
    original_attention = SharedWeightsCrossAttention(**default_attention_params)
    original_attention.build(combined_input.shape)

    # Get build config
    build_config = original_attention.get_build_config()

    # Check that build config contains input_shape
    assert "input_shape" in build_config
    assert build_config["input_shape"] == combined_input.shape

    # Create new attention layer and build from config
    new_attention = SharedWeightsCrossAttention(**default_attention_params)
    new_attention.build_from_config(build_config)

    # Check that new attention layer is built
    assert new_attention.built
    assert new_attention._build_input_shape == combined_input.shape


def test_attention_build_configuration_methods(default_attention_params: Dict[str, Any]) -> None:
    """Test build configuration methods handle None input_shape."""
    # Test with None input_shape
    attention = SharedWeightsCrossAttention(**default_attention_params)

    # Before building
    build_config = attention.get_build_config()
    assert build_config["input_shape"] is None

    # build_from_config with None should not crash
    new_attention = SharedWeightsCrossAttention(**default_attention_params)
    new_attention.build_from_config({"input_shape": None})
    assert not new_attention.built  # Should not be built if input_shape is None


@pytest.mark.parametrize("num_heads", [1, 2, 4, 8, 16])
def test_attention_different_head_counts(
        num_heads: int,
        two_modality_inputs: Tuple[tf.Tensor, List[int]]
) -> None:
    """Test different numbers of attention heads."""
    combined_input, split_sizes = two_modality_inputs
    attention = SharedWeightsCrossAttention(dim=256, num_heads=num_heads, dropout_rate=0.0)

    output = attention(combined_input, split_sizes=split_sizes)
    assert output.shape == combined_input.shape
    assert not tf.reduce_any(tf.math.is_nan(output))


@pytest.mark.parametrize("dim", [64, 128, 256, 512])
def test_attention_different_dimensions(
        dim: int,
        two_modality_inputs: Tuple[tf.Tensor, List[int]]
) -> None:
    """Test different attention dimensions."""
    # Adjust input to match dimension
    combined_input, split_sizes = two_modality_inputs
    # Resize input to match the test dimension
    if dim != 256:
        resized_input = tf.random.normal((combined_input.shape[0], combined_input.shape[1], dim))
    else:
        resized_input = combined_input

    attention = SharedWeightsCrossAttention(dim=dim, num_heads=8, dropout_rate=0.0)

    output = attention(resized_input, split_sizes=split_sizes)
    assert output.shape == resized_input.shape
    assert not tf.reduce_any(tf.math.is_nan(output))


def test_attention_gradient_flow(
        two_modality_inputs: Tuple[tf.Tensor, List[int]],
        default_attention_params: Dict[str, Any]
) -> None:
    """Test gradient flow through the attention layer."""
    combined_input, split_sizes = two_modality_inputs
    attention = SharedWeightsCrossAttention(**default_attention_params)

    with tf.GradientTape() as tape:
        output = attention(combined_input, split_sizes=split_sizes, training=True)
        loss = tf.reduce_mean(output)

    gradients = tape.gradient(loss, attention.trainable_variables)

    # Check if gradients exist for all trainable variables
    assert all(g is not None for g in gradients)

    # Check if we have trainable variables
    assert len(attention.trainable_variables) > 0


def test_attention_different_modality_sizes(default_attention_params: Dict[str, Any]) -> None:
    """Test attention with various modality size combinations."""
    tf.random.set_seed(42)

    test_cases = [
        ([50, 100], "Different sized modalities"),
        ([200, 50], "Large vs small modality"),
        ([1, 299], "Very uneven modalities"),
        ([25, 25], "Small equal modalities"),
    ]

    for split_sizes, description in test_cases:
        # Create input with these split sizes
        total_len = sum(split_sizes)
        combined_input = tf.random.normal((2, total_len, 256))

        attention = SharedWeightsCrossAttention(**default_attention_params)
        output = attention(combined_input, split_sizes=split_sizes)

        assert output.shape == combined_input.shape, f"Failed for {description}"
        assert not tf.reduce_any(tf.math.is_nan(output)), f"NaN values for {description}"


def test_attention_model_integration(
        two_modality_inputs: Tuple[tf.Tensor, List[int]],
        default_attention_params: Dict[str, Any]
) -> None:
    """Test integrating SharedWeightsCrossAttention into a Keras model."""
    combined_input, split_sizes = two_modality_inputs

    # Create a model that uses the attention layer
    inputs = keras.Input(shape=combined_input.shape[1:])
    attention_layer = SharedWeightsCrossAttention(**default_attention_params)

    # Note: In a real model, split_sizes would need to be handled differently
    # For testing, we'll create a simple wrapper
    class AttentionWrapper(keras.layers.Layer):
        def __init__(self, attention_layer, split_sizes, **kwargs):
            super().__init__(**kwargs)
            self.attention_layer = attention_layer
            self.split_sizes = split_sizes

        def call(self, inputs, training=None):
            return self.attention_layer(inputs, split_sizes=self.split_sizes, training=training)

    wrapped_attention = AttentionWrapper(attention_layer, split_sizes)
    outputs = wrapped_attention(inputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Test forward pass
    result = model(combined_input)
    assert result.shape == combined_input.shape


def test_attention_model_compilation_and_training(
        two_modality_inputs: Tuple[tf.Tensor, List[int]],
        default_attention_params: Dict[str, Any]
) -> None:
    """Test compiling and training a model with SharedWeightsCrossAttention."""
    combined_input, split_sizes = two_modality_inputs

    # Create a classification model using attention
    inputs = keras.Input(shape=combined_input.shape[1:])

    class AttentionWrapper(keras.layers.Layer):
        def __init__(self, attention_layer, split_sizes, **kwargs):
            super().__init__(**kwargs)
            self.attention_layer = attention_layer
            self.split_sizes = split_sizes

        def call(self, inputs, training=None):
            return self.attention_layer(inputs, split_sizes=self.split_sizes, training=training)

    attention_layer = SharedWeightsCrossAttention(**default_attention_params)
    wrapped_attention = AttentionWrapper(attention_layer, split_sizes)

    x = wrapped_attention(inputs)
    x = keras.layers.GlobalAveragePooling1D()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Create dummy labels
    labels = tf.random.uniform((combined_input.shape[0],), 0, 10, dtype=tf.int32)

    # Test training for one step
    history = model.fit(combined_input, labels, epochs=1, verbose=0)
    assert len(history.history["loss"]) == 1


def test_attention_model_save_load_weights(
        two_modality_inputs: Tuple[tf.Tensor, List[int]],
        default_attention_params: Dict[str, Any],
        tmp_path
) -> None:
    """Test saving and loading model weights with SharedWeightsCrossAttention."""
    combined_input, split_sizes = two_modality_inputs

    # Create a simple model with the attention layer
    inputs = keras.Input(shape=combined_input.shape[1:])

    class AttentionWrapper(keras.layers.Layer):
        def __init__(self, attention_layer, split_sizes, **kwargs):
            super().__init__(**kwargs)
            self.attention_layer = attention_layer
            self.split_sizes = split_sizes

        def call(self, inputs, training=None):
            return self.attention_layer(inputs, split_sizes=self.split_sizes, training=training)

    attention_layer = SharedWeightsCrossAttention(**default_attention_params)
    wrapped_attention = AttentionWrapper(attention_layer, split_sizes)
    outputs = wrapped_attention(inputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Generate output before saving
    original_output = model(combined_input, training=False).numpy()

    # Save model weights
    save_path = str(tmp_path / "model.weights.h5")
    model.save_weights(save_path)

    # Load model weights
    model.load_weights(save_path)

    # Generate output after loading
    loaded_output = model(combined_input, training=False).numpy()

    # Outputs should be identical
    assert np.allclose(original_output, loaded_output, rtol=1e-5, atol=1e-5)


def test_error_handling_invalid_split_sizes(
        two_modality_inputs: Tuple[tf.Tensor, List[int]],
        default_attention_params: Dict[str, Any]
) -> None:
    """Test error handling for invalid split_sizes."""
    combined_input, _ = two_modality_inputs
    attention = SharedWeightsCrossAttention(**default_attention_params)

    # Test with non-list/tuple split_sizes
    with pytest.raises(ValueError, match="split_sizes must be a list or tuple"):
        attention(combined_input, split_sizes=250)

    # Test with wrong length split_sizes
    with pytest.raises(ValueError, match="split_sizes must have length 2 or 4"):
        attention(combined_input, split_sizes=[100, 75, 75])

    # Test with split_sizes that don't sum to total length
    with pytest.raises(ValueError, match="Sum of split_sizes .* must equal total sequence length"):
        attention(combined_input, split_sizes=[100, 200])  # Sums to 300, but input has 250


def test_error_handling_invalid_input_shape(default_attention_params: Dict[str, Any]) -> None:
    """Test error handling for invalid input shapes."""
    attention = SharedWeightsCrossAttention(**default_attention_params)

    # Test with wrong input dimensionality during build
    with pytest.raises(ValueError, match="Input must be 3D"):
        attention.build((32, 256))  # Only 2D

    # Test with wrong feature dimension during build
    with pytest.raises(ValueError, match="Last dimension of input .* must match dim"):
        attention.build((2, 100, 128))  # Last dim is 128, but dim is 256


def test_cross_attention_behavior(
        two_modality_inputs: Tuple[tf.Tensor, List[int]],
        default_attention_params: Dict[str, Any]
) -> None:
    """Test that cross-attention produces different outputs than self-attention."""
    combined_input, split_sizes = two_modality_inputs

    # Cross-attention
    cross_attention = SharedWeightsCrossAttention(**default_attention_params)
    cross_output = cross_attention(combined_input, split_sizes=split_sizes)

    # Self-attention (using standard multi-head attention for comparison)
    self_attention = keras.layers.MultiHeadAttention(
        num_heads=default_attention_params["num_heads"],
        key_dim=default_attention_params["dim"] // default_attention_params["num_heads"],
        dropout=default_attention_params["dropout_rate"]
    )
    self_output = self_attention(combined_input, combined_input)

    # Outputs should be different (cross vs self attention)
    assert not np.allclose(cross_output.numpy(), self_output.numpy(), rtol=1e-2)


def test_equal_vs_unequal_modality_optimization(default_attention_params: Dict[str, Any]) -> None:
    """Test that equal-sized modalities use optimization path correctly."""
    tf.random.set_seed(42)

    # Equal-sized modalities (should use optimized path)
    equal_input = tf.random.normal((2, 200, 256))  # 100 + 100
    equal_split = [100, 100]

    # Unequal-sized modalities (should use general path)
    unequal_input = tf.random.normal((2, 200, 256))  # 80 + 120
    unequal_split = [80, 120]

    attention = SharedWeightsCrossAttention(**default_attention_params)

    # Both should work without errors
    equal_output = attention(equal_input, split_sizes=equal_split)
    unequal_output = attention(unequal_input, split_sizes=unequal_split)

    assert equal_output.shape == equal_input.shape
    assert unequal_output.shape == unequal_input.shape
    assert not tf.reduce_any(tf.math.is_nan(equal_output))
    assert not tf.reduce_any(tf.math.is_nan(unequal_output))


def test_anchor_query_vs_simple_cross_attention(
        default_attention_params: Dict[str, Any]
) -> None:
    """Test that anchor-query attention produces different results than simple cross-attention."""
    tf.random.set_seed(42)

    # Create input for both configurations (same total size)
    total_tokens = 200
    input_tensor = tf.random.normal((2, total_tokens, 256))

    attention = SharedWeightsCrossAttention(**default_attention_params)

    # Simple cross-attention (2 modalities)
    simple_output = attention(input_tensor, split_sizes=[100, 100])

    # Anchor-query attention (4 splits)
    anchor_query_output = attention(input_tensor, split_sizes=[50, 50, 50, 50])

    # Outputs should be different due to different attention patterns
    assert not np.allclose(simple_output.numpy(), anchor_query_output.numpy(), rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])