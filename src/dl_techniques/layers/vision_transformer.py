"""
# Vision Transformer (ViT) Layer Implementation Documentation

## Table of Contents
1. [Overview](#overview)
2. [Theoretical Background](#theoretical-background)
3. [Architecture Details](#architecture-details)
4. [Implementation Details](#implementation-details)
5. [Usage Guidelines](#usage-guidelines)
6. [Performance Considerations](#performance-considerations)

## Overview

This implementation provides a modular Vision Transformer (ViT) layer based on the architecture proposed in "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020). The implementation follows Keras best practices and includes comprehensive type hinting and documentation.

## Theoretical Background

### Vision Transformer Principles

Vision Transformers adapt the transformer architecture, originally designed for NLP tasks, to computer vision by:

1. **Patch Embedding**:
   - Images are split into fixed-size patches (e.g., 16x16 pixels)
   - Each patch is flattened and linearly projected to create a sequence of embeddings
   - Position embeddings are added to maintain spatial information

2. **Self-Attention Mechanism**:
   - Enables global interaction between all patches
   - Computes attention weights using scaled dot-product attention:
     ```
     Attention(Q, K, V) = softmax(QK^T / √d_k)V
     ```
   - Uses multiple attention heads for parallel attention computation

3. **Transformer Encoder**:
   - Combines self-attention with MLP blocks
   - Uses Layer Normalization and residual connections
   - Pre-norm architecture for improved training stability

## Architecture Details

### Component Breakdown

1. **PatchEmbed Layer**:
   ```python
   Input Image (H×W×C) → Patches ((H/P)×(W/P)×(P²×C)) → Linear Projection (N×D)
   ```
   - H, W: Image height and width
   - C: Number of channels
   - P: Patch size
   - N: Number of patches ((H×W)/(P×P))
   - D: Embedding dimension

2. **Multi-Head Attention**:
   ```
   MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
   where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
   ```
   - Splits embedding dimension into h heads
   - Each head processes a different projection of the input
   - Outputs are concatenated and linearly projected

3. **MLP Block**:
   ```
   Linear → GELU → Dropout → Linear → Dropout
   ```
   - Hidden dimension typically 4x input dimension
   - GELU activation for better performance
   - Dropout for regularization

### Layer Normalization Strategy

The implementation uses pre-norm architecture:
```
x → LayerNorm → Attention → Dropout → Add → LayerNorm → MLP → Add
```

Benefits of pre-norm:
- More stable gradients
- Enables deeper architectures
- Better training dynamics

## Implementation Details

### Key Features

1. **Modular Design**:
   ```python
   PatchEmbed → Optional[CLS Token + Position Embedding] → TransformerLayer
   ```
   - Each component is a separate layer
   - Easy to modify or extend

2. **Initialization and Regularization**:
   - He initialization for better gradient flow
   - L2 regularization support
   - Dropout in attention and MLP paths

3. **Shape Handling**:
   ```python
   # For 224×224 image, 16×16 patches, 768-dim embedding
   Input: (B, 224, 224, 3)
   Patches: (B, 196, 768)  # 196 = (224/16)²
   Output: (B, 196, 768)
   ```
   - B: Batch size
   - Automatic shape inference
   - Dynamic batch size support

### Optimization Techniques

1. **Attention Computation**:
   ```python
   scale = (embed_dim // num_heads) ** -0.5  # Scaling factor
   attention = softmax(QK^T * scale)
   ```
   - Scaled dot-product attention
   - Efficient matrix multiplication
   - Proper numerical stability

2. **Memory Efficiency**:
   - Fused QKV computation
   - Proper reshaping for attention heads
   - Minimal tensor copies

## Usage Guidelines

### Basic Usage

```python
# Create patch embedding
patch_embed = PatchEmbed(
    patch_size=16,
    embed_dim=768,
    kernel_regularizer=L2(1e-4)
)

# Create transformer layer
transformer = VisionTransformerLayer(
    embed_dim=768,
    num_heads=12,
    mlp_ratio=4.0,
    dropout_rate=0.1
)

# Process image
x = patch_embed(images)  # (B, N, D)
x = transformer(x)       # (B, N, D)
```

### Hyperparameter Guidelines

1. **Patch Size**:
   - Typical values: 16×16 or 32×32
   - Larger patches → fewer tokens → faster computation
   - Smaller patches → finer granularity → better detail

2. **Embedding Dimension**:
   - Should be divisible by number of heads
   - Typical values: 768 (ViT-Base), 1024 (ViT-Large)
   - Affects model capacity and computation

3. **Number of Heads**:
   - Typical values: 12 (ViT-Base), 16 (ViT-Large)
   - More heads → more parallel attention computation
   - Each head should have reasonable dimension (>=64)

4. **Dropout Rates**:
   - Attention dropout: 0.0-0.1
   - MLP dropout: 0.1-0.2
   - Higher for smaller datasets

## Performance Considerations

### Memory Usage

Memory complexity of self-attention:
- O(N²) memory usage where N is number of patches
- For 224×224 image with 16×16 patches:
  - N = 196 patches
  - Attention matrix size: 196×196 = 38,416 elements per head

### Computation Efficiency

1. **Patch Embedding**:
   - Uses Conv2D for efficient patch extraction
   - Single matrix multiplication for embedding

2. **Attention Computation**:
   - Fused QKV transformation
   - Parallel processing across heads
   - Efficient batch matrix multiplication

3. **Training Tips**:
   - Gradient checkpointing for large models
   - Mixed precision training recommended
   - Proper batch size selection important

### Recommended Configurations

1. **Small-Scale (ViT-Tiny)**:
   ```python
   patch_size=16
   embed_dim=192
   num_heads=3
   mlp_ratio=4.0
   ```

2. **Medium-Scale (ViT-Base)**:
   ```python
   patch_size=16
   embed_dim=768
   num_heads=12
   mlp_ratio=4.0
   ```

3. **Large-Scale (ViT-Large)**:
   ```python
   patch_size=16
   embed_dim=1024
   num_heads=16
   mlp_ratio=4.0
   ```
"""

import keras
import tensorflow as tf
from typing import Optional, Tuple, Union
from keras import Layer, Regularizer, Initializer

# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class PatchEmbed(Layer):
    """2D Image to Patch Embedding Layer.

    Splits images into patches and linearly embeds each patch.

    Args:
        patch_size: Size of patches to split the input image into
        embed_dim: Embedding dimension for patches
        kernel_initializer: Initializer for the projection matrix
        kernel_regularizer: Regularizer function for the projection matrix
        name: Optional name for the layer
        **kwargs: Additional layer arguments
    """

    def __init__(
            self,
            patch_size: Union[int, Tuple[int, int]],
            embed_dim: int,
            kernel_initializer: Union[str, Initializer] = "he_normal",
            kernel_regularizer: Optional[Regularizer] = None,
            name: Optional[str] = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.embed_dim = embed_dim
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

        self.proj = keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            padding="valid",
            name=f"{name}_projection" if name else None
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            x: Input tensor of shape (batch_size, height, width, channels)

        Returns:
            Embedded patches tensor of shape (batch_size, n_patches, embed_dim)
        """
        x = self.proj(x)  # (batch_size, h', w', embed_dim)
        batch_size = tf.shape(x)[0]
        # Rearrange to (batch_size, n_patches, embed_dim)
        return tf.reshape(x, (batch_size, -1, self.embed_dim))

# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class MultiHeadAttention(Layer):
    """Multi-Head Self Attention mechanism.

    Args:
        embed_dim: Dimension of input embeddings
        num_heads: Number of attention heads
        dropout_rate: Dropout rate for attention weights
        kernel_initializer: Initializer for weight matrices
        kernel_regularizer: Regularizer for weight matrices
        name: Optional name for the layer
        **kwargs: Additional layer arguments
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            kernel_initializer: Union[str, Initializer] = "he_normal",
            kernel_regularizer: Optional[Regularizer] = None,
            name: Optional[str] = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5

        self.qkv = keras.layers.Dense(
            embed_dim * 3,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=f"{name}_qkv" if name else None
        )
        self.proj = keras.layers.Dense(
            embed_dim,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=f"{name}_proj" if name else None
        )
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            training: Whether in training mode

        Returns:
            Attention output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = tf.unstack(tf.shape(x))
        head_dim = self.embed_dim // self.num_heads

        qkv = self.qkv(x)  # (batch_size, seq_len, embed_dim * 3)
        qkv = tf.reshape(qkv, (batch_size, seq_len, 3, self.num_heads, head_dim))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn, training=training)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, (0, 2, 1, 3))
        x = tf.reshape(x, (batch_size, seq_len, self.embed_dim))

        return self.proj(x)

# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class VisionTransformerLayer(Layer):
    """Vision Transformer (ViT) Layer.

    Implements a single transformer encoder layer for vision tasks.

    Args:
        embed_dim: Dimension of input embeddings
        num_heads: Number of attention heads
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        dropout_rate: Dropout rate
        attention_dropout_rate: Dropout rate for attention weights
        kernel_initializer: Initializer for weight matrices
        kernel_regularizer: Regularizer for weight matrices
        name: Optional name for the layer
        **kwargs: Additional layer arguments
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            dropout_rate: float = 0.0,
            attention_dropout_rate: float = 0.0,
            kernel_initializer: Union[str, Initializer] = "he_normal",
            kernel_regularizer: Optional[Regularizer] = None,
            name: Optional[str] = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)

        # Layer Norm 1 (before attention)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm1" if name else None)

        # Multi-head Self Attention
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=attention_dropout_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=f"{name}_attn" if name else None
        )
        self.dropout1 = keras.layers.Dropout(dropout_rate)

        # Layer Norm 2 (before MLP)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm2" if name else None)

        # MLP block
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = keras.Sequential([
            keras.layers.Dense(
                mlp_hidden_dim,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f"{name}_mlp1" if name else None
            ),
            keras.layers.Activation("gelu"),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(
                embed_dim,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f"{name}_mlp2" if name else None
            ),
            keras.layers.Dropout(dropout_rate)
        ], name=f"{name}_mlp" if name else None)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            training: Whether in training mode

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Attention block (with residual)
        x1 = self.norm1(x)
        x1 = self.attn(x1, training=training)
        x1 = self.dropout1(x1, training=training)
        x = x + x1

        # MLP block (with residual)
        x2 = self.norm2(x)
        x2 = self.mlp(x2, training=training)
        return x + x2

    def get_config(self) -> dict:
        """Gets layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "embed_dim": self.attn.embed_dim,
            "num_heads": self.attn.num_heads,
            "mlp_ratio": self.mlp.layers[0].units / self.attn.embed_dim,
            "dropout_rate": self.dropout1.rate,
            "attention_dropout_rate": self.attn.dropout.rate,
            "kernel_initializer": keras.initializers.serialize(self.attn.qkv.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.attn.qkv.kernel_regularizer)
        })
        return config

# ---------------------------------------------------------------------