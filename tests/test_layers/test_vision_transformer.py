import pytest
import tensorflow as tf
from keras.api.regularizers import L2

from dl_techniques.layers.patch_embedding import PatchEmbedding2D
from dl_techniques.layers.vision_transformer import (
    MultiHeadAttention,
    VisionTransformerLayer
)


@pytest.fixture
def sample_image():
    return tf.random.normal((2, 224, 224, 3))


@pytest.fixture
def sample_sequence():
    return tf.random.normal((2, 196, 768))


def test_patch_embed():
    """Test PatchEmbed layer."""
    layer = PatchEmbedding2D(
        patch_size=16,
        embed_dim=768,
        kernel_initializer="he_normal",
        kernel_regularizer=L2(1e-4)
    )

    x = tf.random.normal((2, 224, 224, 3))
    output = layer(x)

    # Check output shape (batch_size, n_patches, embed_dim)
    assert output.shape == (2, 196, 768)  # 224/16 = 14, 14*14 = 196 patches


def test_multi_head_attention():
    """Test MultiHeadAttention layer."""
    layer = MultiHeadAttention(
        embed_dim=768,
        num_heads=12,
        dropout_rate=0.1,
        kernel_initializer="he_normal",
        kernel_regularizer=L2(1e-4)
    )

    x = tf.random.normal((2, 196, 768))
    output = layer(x, training=True)

    # Check output shape
    assert output.shape == x.shape


def test_vision_transformer_layer():
    """Test VisionTransformerLayer."""
    layer = VisionTransformerLayer(
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        kernel_initializer="he_normal",
        kernel_regularizer=L2(1e-4)
    )

    x = tf.random.normal((2, 196, 768))
    output = layer(x, training=True)

    # Check output shape
    assert output.shape == x.shape


def test_serialization():
    """Test layer serialization."""
    layer = VisionTransformerLayer(
        embed_dim=768,
        num_heads=12,
        kernel_regularizer=L2(1e-4)
    )

    config = layer.get_config()
    reconstructed_layer = VisionTransformerLayer.from_config(config)

    assert isinstance(reconstructed_layer, VisionTransformerLayer)
    assert reconstructed_layer.embed_dim == 768
    assert reconstructed_layer.num_heads == 12


def test_end_to_end(sample_image):
    """Test end-to-end pipeline."""
    # Create patch embedding
    patch_embed = PatchEmbedding2D(
        patch_size=16,
        embed_dim=768,
        kernel_regularizer=L2(1e-4)
    )

    # Create transformer layer
    transformer = VisionTransformerLayer(
        embed_dim=768,
        num_heads=12,
        kernel_regularizer=L2(1e-4)
    )

    # Process image
    x = patch_embed(sample_image)
    output = transformer(x, training=True)

    # Check final output shape
    assert output.shape == (2, 196, 768)


if __name__ == "__main__":
    pytest.main([__file__])