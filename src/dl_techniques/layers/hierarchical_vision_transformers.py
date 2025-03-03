"""
Hierarchical MLP (hMLP) Stem for Vision Transformers

This module implements the hierarchical MLP stem as described in
"Three things everyone should know about Vision Transformers" by Touvron et al.

Key features:
1. Processes each patch independently (no information sharing between patches)
2. Progressively increases patch size from 2×2 → 4×4 → 8×8 → 16×16
3. Compatible with masked self-supervised learning methods
4. Minimal computational overhead (<1% increase in FLOPs)
5. Available with BatchNorm or LayerNorm variants
"""

import keras
import tensorflow as tf
from typing import Tuple, Optional, Union, List


class HierarchicalMLPStem(keras.layers.Layer):
    """
    Hierarchical MLP Stem for Vision Transformers.

    This stem processes patches independently through a sequence of linear projections,
    normalizations, and activations, gradually increasing the patch size from
    2×2 to 16×16 without any cross-patch communication.
    """

    def __init__(
            self,
            embed_dim: int = 768,
            img_size: Tuple[int, int] = (224, 224),
            patch_size: Tuple[int, int] = (16, 16),
            in_channels: int = 3,
            norm_layer: str = "batch",
            activation: str = "gelu",
            name: Optional[str] = None,
    ):
        """
        Initialize the Hierarchical MLP Stem.

        Args:
            embed_dim: Final embedding dimension
            img_size: Input image dimensions (height, width)
            patch_size: Final patch dimensions (height, width)
            in_channels: Number of input channels (3 for RGB)
            norm_layer: Normalization type ('batch' or 'layer')
            activation: Activation function ('gelu' or 'relu')
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

        # Activation function
        if activation == "gelu":
            self.activation = keras.activations.gelu
        elif activation == "relu":
            self.activation = keras.activations.relu
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Define the hierarchical convolutions (equivalent to independent patch processing)
        # Stage 1: Process 4x4 patches (kernel_size=4, stride=4)
        self.stage1_conv = keras.layers.Conv2D(
            filters=self.dim1,
            kernel_size=4,
            strides=4,
            padding="valid",
            name=f"{name}_conv1" if name else None
        )

        # Stage 2: Process 2x2 patches within the previous patches (kernel_size=2, stride=2)
        self.stage2_conv = keras.layers.Conv2D(
            filters=self.dim1,
            kernel_size=2,
            strides=2,
            padding="valid",
            name=f"{name}_conv2" if name else None
        )

        # Stage 3: Final 2x2 patches to get 16x16 total patch size (kernel_size=2, stride=2)
        self.stage3_conv = keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=2,
            strides=2,
            padding="valid",
            name=f"{name}_conv3" if name else None
        )

        # Calculate the number of patches
        h_patches = img_size[0] // patch_size[0]
        w_patches = img_size[1] // patch_size[1]
        self.num_patches = h_patches * w_patches

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Apply the hierarchical MLP stem to input images.

        Args:
            x: Input tensor of shape [batch_size, height, width, channels]
            training: Whether in training mode

        Returns:
            Tensor of shape [batch_size, num_patches, embed_dim]
        """
        # Stage 1: 4x4 patches
        x = self.stage1_conv(x)
        if isinstance(self.norm1, keras.layers.BatchNormalization):
            x = self.norm1(x, training=training)
        else:
            # Reshape for layer norm if needed
            orig_shape = tf.shape(x)
            x = tf.reshape(x, [-1, x.shape[-1]])
            x = self.norm1(x)
            x = tf.reshape(x, orig_shape)
        x = self.activation(x)

        # Stage 2: 8x8 patches (4x4 + 2x2 processing)
        x = self.stage2_conv(x)
        if isinstance(self.norm2, keras.layers.BatchNormalization):
            x = self.norm2(x, training=training)
        else:
            # Reshape for layer norm if needed
            orig_shape = tf.shape(x)
            x = tf.reshape(x, [-1, x.shape[-1]])
            x = self.norm2(x)
            x = tf.reshape(x, orig_shape)
        x = self.activation(x)

        # Stage 3: 16x16 patches (8x8 + 2x2 processing)
        x = self.stage3_conv(x)
        if isinstance(self.norm3, keras.layers.BatchNormalization):
            x = self.norm3(x, training=training)
        else:
            # Reshape for layer norm if needed
            orig_shape = tf.shape(x)
            x = tf.reshape(x, [-1, x.shape[-1]])
            x = self.norm3(x)
            x = tf.reshape(x, orig_shape)

        # Reshape from [B, H, W, C] to [B, HW, C]
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, [B, H * W, C])

        return x


class ViTWithHMLPStem(keras.Model):
    """
    Vision Transformer with the Hierarchical MLP Stem.

    This class combines the hMLP stem with a standard Vision Transformer.
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
        Initialize the ViT with hMLP stem.

        Args:
            image_size: Input image dimensions
            patch_size: Patch dimensions
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Whether to use bias in qkv projections
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            norm_layer: Normalization layer type
        """
        super().__init__()

        # Create the hierarchical MLP stem
        self.stem = HierarchicalMLPStem(
            embed_dim=embed_dim,
            img_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            norm_layer=norm_layer,
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
                norm_layer="layer",  # ViTs typically use LayerNorm
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

        # Process patches with hMLP stem
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


class TransformerBlock(keras.layers.Layer):
    """
    Standard Transformer block with multi-head self-attention and MLP.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: int = 4,
            qkv_bias: bool = False,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            norm_layer: str = "layer",
            name: Optional[str] = None,
    ):
        """
        Initialize a Transformer block.

        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Whether to use bias in qkv projections
            drop: Dropout rate
            attn_drop: Attention dropout rate
            norm_layer: Normalization layer type
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
        self.mlp = MLPBlock(
            hidden_dim=dim * mlp_ratio,
            output_dim=dim,
            drop=drop,
            name=f"{name}_mlp" if name else None,
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

        # MLP block with residual connection
        mlp_output = self.mlp(self.norm2(x), training=training)
        x = x + mlp_output

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


class MLPBlock(keras.layers.Layer):
    """
    MLP block used in Vision Transformers.
    """

    def __init__(
            self,
            hidden_dim: int,
            output_dim: int,
            drop: float = 0.0,
            name: Optional[str] = None,
    ):
        """
        Initialize MLP block.

        Args:
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            drop: Dropout rate
            name: Layer name
        """
        super().__init__(name=name)
        self.fc1 = keras.layers.Dense(hidden_dim, name=f"{name}_fc1" if name else None)
        self.act = keras.activations.gelu
        self.fc2 = keras.layers.Dense(output_dim, name=f"{name}_fc2" if name else None)
        self.drop = keras.layers.Dropout(drop)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Apply MLP block.

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            Output tensor
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x


# Example usage
def create_vit_model(
        image_size: Tuple[int, int] = (224, 224),
        patch_size: Tuple[int, int] = (16, 16),
        num_classes: int = 1000,
        model_size: str = "base"
) -> keras.Model:
    """
    Create a Vision Transformer with hierarchical MLP stem.

    Args:
        image_size: Input image dimensions (height, width)
        patch_size: Patch dimensions (height, width)
        num_classes: Number of output classes
        model_size: Model size ('tiny', 'small', 'base', or 'large')

    Returns:
        ViT model with hMLP stem
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

    return ViTWithHMLPStem(
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


# Example demonstrating how to use the hMLP stem with masked self-supervised learning like BeiT
def create_inputs_with_masking(
        batch_size: int = 8,
        image_size: Tuple[int, int] = (224, 224),
        patch_size: Tuple[int, int] = (16, 16),
        mask_ratio: float = 0.4,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Create masked input images and corresponding mask.

    Args:
        batch_size: Batch size
        image_size: Image dimensions (height, width)
        patch_size: Patch dimensions (height, width)
        mask_ratio: Ratio of patches to mask

    Returns:
        Tuple of (images, mask) where mask is 1 for masked patches
    """
    # Create random images
    images = tf.random.normal([batch_size, image_size[0], image_size[1], 3])

    # Create random mask
    num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
    num_mask = int(mask_ratio * num_patches)

    # For each sample in the batch, randomly select indices to mask
    mask = tf.zeros([batch_size, num_patches], dtype=tf.float32)
    for i in range(batch_size):
        indices = tf.random.shuffle(tf.range(num_patches))[:num_mask]
        mask_sample = tf.zeros([num_patches], dtype=tf.float32)
        mask_sample = tf.tensor_scatter_nd_update(
            mask_sample,
            indices[:, tf.newaxis],
            tf.ones([num_mask], dtype=tf.float32)
        )
        mask = tf.tensor_scatter_nd_update(
            mask,
            [[i]],
            [mask_sample][tf.newaxis]
        )

    return images, mask


def apply_mask_after_stem(
        stem: HierarchicalMLPStem,
        images: tf.Tensor,
        mask: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Process images with hMLP stem then apply masking.

    This function demonstrates the key advantage of hMLP stem:
    it can be used with masked self-supervised learning by
    applying masking after the stem.

    Args:
        stem: Hierarchical MLP stem
        images: Input images
        mask: Mask tensor (1 for masked patches)

    Returns:
        Tuple of (processed_patches, mask)
    """
    # Process images with stem
    patches = stem(images)

    # Apply mask (set masked patches to zero)
    batch_size, num_patches, embed_dim = tf.shape(patches)[0], tf.shape(patches)[1], tf.shape(patches)[2]
    mask_expanded = tf.expand_dims(mask, -1)  # [B, N, 1]
    mask_expanded = tf.tile(mask_expanded, [1, 1, embed_dim])  # [B, N, D]

    # Zero out masked patches
    masked_patches = patches * (1 - mask_expanded)

    return masked_patches, mask


if __name__ == "__main__":
    # Create a model
    model = create_vit_model(model_size="small")

    # Create dummy input
    x = tf.random.normal([2, 224, 224, 3])

    # Test forward pass
    output = model(x, training=True)

    print(f"Model output shape: {output.shape}")
    print(f"Number of parameters: {model.count_params()}")

    # Test with masking
    images, mask = create_inputs_with_masking(batch_size=2)
    stem = HierarchicalMLPStem(embed_dim=384)
    masked_patches, _ = apply_mask_after_stem(stem, images, mask)

    print(f"Masked patches shape: {masked_patches.shape}")