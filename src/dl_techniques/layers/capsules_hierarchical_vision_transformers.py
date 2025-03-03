"""
Hierarchical Capsule Stem for Vision Transformers
==================================================

This module implements a hierarchical capsule stem for Vision Transformers,
combining the hierarchical structure from "Three things everyone should know about
Vision Transformers" by Touvron et al. with capsule networks from Hinton et al.

HYBRID DESIGN:
--------------
This implementation fuses:
1. The hierarchical patch processing approach of hMLP stems
2. The dynamic routing and part-whole relationships of capsule networks

CAPSULE ADVANTAGES:
------------------
- Preserves spatial relationships between features
- Better handles viewpoint changes and transformations
- More robust representation of visual entities

KEY FEATURES:
------------
- Processes each patch independently (no information leakage between patches)
- Progressively processes patches from 2×2 → 4×4 → 8×8 → 16×16
- Uses capsule routing instead of traditional linear projections
- Maintains compatibility with masked self-supervised learning
- Potentially better at capturing hierarchical relationships in image data
"""

import keras
import tensorflow as tf
from typing import Tuple, Optional, Union, List


from .capsules import CapsuleBlock


class HierarchicalCapsuleStem(keras.layers.Layer):
    """
    Hierarchical Capsule Stem for Vision Transformers.

    This stem processes patches independently through a sequence of capsule layers,
    gradually increasing the patch size from 2×2 to 16×16 without any cross-patch
    communication.
    """

    def __init__(
            self,
            embed_dim: int = 768,
            img_size: Tuple[int, int] = (224, 224),
            patch_size: Tuple[int, int] = (16, 16),
            in_channels: int = 3,
            norm_layer: str = "batch",
            drop_rate: float = 0.0,
            name: Optional[str] = None,
    ):
        """
        Initialize the Hierarchical Capsule Stem.

        Args:
            embed_dim: Final embedding dimension
            img_size: Input image dimensions (height, width)
            patch_size: Final patch dimensions (height, width)
            in_channels: Number of input channels (3 for RGB)
            norm_layer: Normalization type ('batch' or 'layer')
            drop_rate: Dropout rate
            name: Layer name
        """
        super().__init__(name=name)

        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        # Verify patch size is 16x16 (could be extended to support other sizes)
        if patch_size != (16, 16):
            raise ValueError(f"Current implementation only supports 16x16 patches, got {patch_size}")

        # Determine intermediate embedding dimensions
        self.dim1 = embed_dim // 4  # Dimension after first stage

        # Create normalization layers based on specified type
        if norm_layer == "batch":
            self.norm1 = keras.layers.BatchNormalization(name=f"{name}_bn1" if name else None)
            self.norm2 = keras.layers.BatchNormalization(name=f"{name}_bn2" if name else None)
            self.norm3 = keras.layers.BatchNormalization(name=f"{name}_bn3" if name else None)
        elif norm_layer == "layer":
            self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1" if name else None)
            self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2" if name else None)
            self.norm3 = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln3" if name else None)
        else:
            raise ValueError(f"Unsupported normalization layer: {norm_layer}")

        # Initial projection from image patches
        self.projection = keras.layers.Conv2D(
            filters=self.dim1,
            kernel_size=4,
            strides=4,
            padding="valid",
            name=f"{name}_proj" if name else None
        )

        # Capsule layers for each hierarchical stage
        self.capsule1 = CapsuleBlock(
            dim=self.dim1,
            drop=drop_rate,
            name=f"{name}_capsule1" if name else None
        )

        self.capsule2 = CapsuleBlock(
            dim=self.dim1,
            drop=drop_rate,
            name=f"{name}_capsule2" if name else None
        )

        self.capsule3 = CapsuleBlock(
            dim=embed_dim,
            drop=drop_rate,
            name=f"{name}_capsule3" if name else None
        )

        # Pooling operations to reduce spatial dimensions (equiv to strided conv)
        self.pool1 = keras.layers.AveragePooling2D(
            pool_size=2,
            strides=2,
            name=f"{name}_pool1" if name else None
        )

        self.pool2 = keras.layers.AveragePooling2D(
            pool_size=2,
            strides=2,
            name=f"{name}_pool2" if name else None
        )

        # Calculate the number of patches
        h_patches = img_size[0] // patch_size[0]
        w_patches = img_size[1] // patch_size[1]
        self.num_patches = h_patches * w_patches

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Apply the hierarchical capsule stem to input images.

        Args:
            x: Input tensor of shape [batch_size, height, width, channels]
            training: Whether in training mode

        Returns:
            Tensor of shape [batch_size, num_patches, embed_dim]
        """
        batch_size = tf.shape(x)[0]

        # Stage 1: Initial projection to 4x4 patches
        x = self.projection(x)
        x = self.norm1(x, training=training)

        # Reshape for capsule processing
        h1, w1 = tf.shape(x)[1], tf.shape(x)[2]
        x_reshaped = tf.reshape(x, [batch_size, h1 * w1, self.dim1])

        # Apply first capsule layer
        x = self.capsule1(x_reshaped, training=training)

        # Reshape back to spatial form for pooling
        x = tf.reshape(x, [batch_size, h1, w1, self.dim1])

        # Stage 2: Pool to 8x8 patches
        x = self.pool1(x)
        x = self.norm2(x, training=training)

        # Reshape for second capsule layer
        h2, w2 = tf.shape(x)[1], tf.shape(x)[2]
        x_reshaped = tf.reshape(x, [batch_size, h2 * w2, self.dim1])

        # Apply second capsule layer
        x = self.capsule2(x_reshaped, training=training)

        # Reshape back to spatial form
        x = tf.reshape(x, [batch_size, h2, w2, self.dim1])

        # Stage 3: Final pooling to reach 16x16 patch size
        x = self.pool2(x)

        # Up-dimension to final embedding size
        x = keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=1,
            padding="valid"
        )(x)

        x = self.norm3(x, training=training)

        # Final reshape for capsule processing
        h3, w3 = tf.shape(x)[1], tf.shape(x)[2]
        x_reshaped = tf.reshape(x, [batch_size, h3 * w3, self.embed_dim])

        # Apply final capsule layer
        x = self.capsule3(x_reshaped, training=training)

        return x


class MultiHeadSelfAttention(keras.layers.Layer):
    """
    Multi-Head Self-Attention module.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            name: Optional[str] = None,
    ):
        """
        Initialize Multi-Head Self-Attention.

        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in qkv projections
            attn_drop: Attention dropout rate
            proj_drop: Output projection dropout rate
            name: Layer name
        """
        super().__init__(name=name)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = keras.layers.Dense(dim * 3, use_bias=qkv_bias, name=f"{name}_qkv" if name else None)
        self.attn_drop = keras.layers.Dropout(attn_drop)
        self.proj = keras.layers.Dense(dim, name=f"{name}_proj" if name else None)
        self.proj_drop = keras.layers.Dropout(proj_drop)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Apply Multi-Head Self-Attention.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            training: Whether in training mode

        Returns:
            Attention output of shape [batch_size, seq_len, dim]
        """
        batch_size, seq_len, dim = tf.shape(x)[0], tf.shape(x)[1], x.shape[2]

        # Compute qkv
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [batch_size, seq_len, 3, self.num_heads, dim // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        # Apply attention to values
        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [batch_size, seq_len, dim])

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x, training=training)

        return x


class TransformerBlock(keras.layers.Layer):
    """
    Transformer block with Multi-Head Self-Attention and Capsule Block.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: int = 4,
            qkv_bias: bool = False,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            name: Optional[str] = None,
    ):
        """
        Initialize a Transformer block.

        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: Not used directly, kept for API compatibility
            qkv_bias: Whether to use bias in qkv projections
            drop: Dropout rate
            attn_drop: Attention dropout rate
            name: Layer name
        """
        super().__init__(name=name)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm1" if name else None)
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            name=f"{name}_attn" if name else None,
        )
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm2" if name else None)

        # Replace MLP with CapsuleBlock
        self.capsule_block = CapsuleBlock(
            dim=dim,
            drop=drop,
            name=f"{name}_capsule" if name else None
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Apply the transformer block to the input.

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            Processed tensor
        """
        # Attention block with residual connection
        attn_output = self.attn(self.norm1(x), training=training)
        x = x + attn_output

        # Capsule block with residual connection
        capsule_output = self.capsule_block(self.norm2(x), training=training)
        x = x + capsule_output

        return x


class ViTWithCapsuleStem(keras.Model):
    """
    Vision Transformer with the Hierarchical Capsule Stem.

    This combines the capsule stem with a Vision Transformer where MLP blocks
    are replaced with Capsule blocks.
    """

    def __init__(
            self,
            image_size: Tuple[int, int] = (224, 224),
            patch_size: Tuple[int, int] = (16, 16),
            num_classes: int = 1000,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: int = 4,
            qkv_bias: bool = True,
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            norm_layer: str = "batch",
    ):
        """
        Initialize the ViT with capsule stem.

        Args:
            image_size: Input image dimensions
            patch_size: Patch dimensions
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Kept for API compatibility
            qkv_bias: Whether to use bias in qkv projections
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            norm_layer: Normalization layer type
        """
        super().__init__()

        # Create the hierarchical capsule stem
        self.stem = HierarchicalCapsuleStem(
            embed_dim=embed_dim,
            img_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            norm_layer=norm_layer,
            drop_rate=drop_rate,
        )

        num_patches = self.stem.num_patches

        # Positional embedding
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, num_patches + 1, embed_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )

        # Class token
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, embed_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )

        # Dropout
        self.pos_drop = keras.layers.Dropout(drop_rate)

        # Transformer blocks
        self.blocks = [
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                name=f"transformer_block_{i}"
            )
            for i in range(depth)
        ]

        # Final normalization and classifier
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name="encoder_norm")
        self.head = keras.layers.Dense(num_classes, name="classification_head")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape [batch_size, height, width, channels]
            training: Whether in training mode

        Returns:
            Classification logits
        """
        batch_size = tf.shape(x)[0]

        # Process patches with capsule stem
        x = self.stem(x, training=training)

        # Add class token
        cls_tokens = tf.repeat(self.cls_token, batch_size, axis=0)
        x = tf.concat([cls_tokens, x], axis=1)

        # Add position embedding and apply dropout
        x = x + self.pos_embed
        x = self.pos_drop(x, training=training)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, training=training)

        # Apply final normalization
        x = self.norm(x)

        # Use [CLS] token for classification
        x = x[:, 0]

        # Classification head
        x = self.head(x)

        return x


def create_capsule_vit_model(
        image_size: Tuple[int, int] = (224, 224),
        patch_size: Tuple[int, int] = (16, 16),
        num_classes: int = 1000,
        model_size: str = "base"
) -> keras.Model:
    """
    Create a Vision Transformer with hierarchical capsule stem.

    Args:
        image_size: Input image dimensions (height, width)
        patch_size: Patch dimensions (height, width)
        num_classes: Number of output classes
        model_size: Model size ('tiny', 'small', 'base', or 'large')

    Returns:
        ViT model with capsule stem
    """
    # Model configurations
    configs = {
        "tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3},
        "small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
        "base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
        "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    }

    if model_size not in configs:
        raise ValueError(f"Unsupported model size: {model_size}")

    config = configs[model_size]

    return ViTWithCapsuleStem(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer="batch",
    )


# Example usage
if __name__ == "__main__":
    # Create a model
    model = create_capsule_vit_model(model_size="small")

    # Create dummy input
    x = tf.random.normal([2, 224, 224, 3])

    # Test forward pass
    output = model(x, training=True)

    print(f"Model output shape: {output.shape}")
    print(f"Number of parameters: {model.count_params()}")