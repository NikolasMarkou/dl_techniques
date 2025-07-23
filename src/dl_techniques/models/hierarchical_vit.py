"""
Hierarchical MLP (hMLP) Stem for Vision Transformers
==================================================

This module implements the hierarchical MLP stem as described in
"Three things everyone should know about Vision Transformers" by Touvron et al.

PAPER OVERVIEW:
--------------
The paper introduces three key insights about Vision Transformers:
1. Parallelizing ViT layers can improve efficiency without affecting accuracy
2. Fine-tuning only attention layers is sufficient for adaptation
3. Using hierarchical MLP stems improves compatibility with masked self-supervised learning

HMLP STEM DESIGN:
---------------
The hMLP stem is a patch pre-processing technique that:
- Processes each patch independently (no information leakage between patches)
- Progressively processes patches from 2×2 → 4×4 → 8×8 → 16×16
- Uses linear projections with normalization and non-linearity at each stage
- Has minimal computational overhead (<1% increase in FLOPs vs. standard ViT)

KEY ADVANTAGES:
-------------
1. Compatible with Masked Self-supervised Learning:
   - Unlike conventional convolutional stems which cause information leakage between patches
   - Works with BeiT, MAE, and other mask-based approaches
   - Masking can be applied either before or after the stem with identical results

2. Performance Benefits:
   - Supervised learning: ~0.3% accuracy improvement over standard ViT
   - BeiT pre-training: +0.4% accuracy improvement over linear projection
   - On par with the best convolutional stems for supervised learning

3. Implementation:
   - Uses convolutions with matching kernel size and stride for efficiency
   - Each patch is processed independently despite using convolutional layers
   - Works with both BatchNorm (better performance) and LayerNorm (stable for small batches)

EXPERIMENTAL RESULTS:
------------------
From the paper:
- Supervised ViT-B with Linear stem: 82.2% top-1 accuracy on ImageNet
- Supervised ViT-B with hMLP stem: 82.5% top-1 accuracy
- BeiT+FT ViT-B with Linear stem: 83.1% top-1 accuracy
- BeiT+FT ViT-B with hMLP stem: 83.4% top-1 accuracy

When used with BeiT, existing convolutional stems show no improvement (83.0%)
while hMLP stem provides significant gains, demonstrating its effectiveness
for masked self-supervised learning approaches.
"""

import keras
import tensorflow as tf
from typing import Tuple, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.ffn.mlp import MLPBlock
from dl_techniques.layers.multi_head_attention import MultiHeadAttention
from dl_techniques.layers.hierarchical_mlp_stem import HierarchicalMLPStem

# ---------------------------------------------------------------------


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
        self.attn = MultiHeadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            use_bias=qkv_bias,
            dropout_rate=attn_drop,
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

# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Create a model
    model = create_vit_model(model_size="small")

    # Create dummy input
    x = tf.random.normal([2, 224, 224, 3])

    # Test forward pass
    output = model(x, training=True)

    logger.info(f"Model output shape: {output.shape}")
    logger.info(f"Number of parameters: {model.count_params()}")

    # Test with masking
    images, mask = create_inputs_with_masking(batch_size=2)
    stem = HierarchicalMLPStem(embed_dim=384)
    masked_patches, _ = apply_mask_after_stem(stem, images, mask)

    logger.info(f"Masked patches shape: {masked_patches.shape}")

# ---------------------------------------------------------------------

