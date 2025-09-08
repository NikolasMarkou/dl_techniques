"""
DINOv2 Vision Transformer Implementation

This module provides a complete implementation of the DINOv2 (DINO version 2) Vision Transformer
architecture using modern Keras 3.x best practices and the dl_techniques framework.

Key Features:
- Configurable ViT backbone with multiple size variants (Tiny, Small, Base, Large, Giant)
- Support for register tokens for improved attention quality
- LearnableMultiplier for layer scaling (LayerScale replacement)
- Stochastic depth regularization with proper linear decay
- Flexible positional embedding with interpolation support
- Memory-efficient attention mechanisms
- Pre-normalization architecture for better gradient flow

The implementation is designed to be compatible with both self-supervised pre-training
(DINO, iBOT, KoLeo losses) and downstream supervised fine-tuning tasks.

Architecture Overview:
```
Input Image (B, H, W, C)
     ↓
Patch Embedding → (B, N, D)
     ↓
[CLS] + [REG] + Patches + PosEmb → (B, 1+R+N, D)
     ↓
DINOv2Block_1 → ... → DINOv2Block_L
     ↓
Layer Normalization
     ↓
CLS Token Features (B, D)
```

Where:
- B: Batch size
- H, W: Image height and width
- C: Input channels (typically 3)
- N: Number of patches = (H/P) * (W/P)
- P: Patch size (typically 14)
- D: Embedding dimension
- R: Number of register tokens (0 or 4)
- L: Number of transformer layers (depth)
"""

import keras
import numpy as np
import tensorflow as tf
from keras import layers, initializers
from typing import Optional, Union, Dict, Any, Tuple, Literal

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.embedding import create_embedding_layer
from dl_techniques.layers.attention import create_attention_layer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.layers.layer_scale import LearnableMultiplier

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DINOv2Block(keras.layers.Layer):
    """
    DINOv2 Transformer Block with LearnableMultiplier scaling and configurable components.

    This block implements the DINOv2 transformer architecture with:
    - Pre-normalization layout (LayerNorm → Attention → Residual)
    - LearnableMultiplier for training stability (replaces LayerScale)
    - Configurable attention mechanisms via factory
    - Configurable FFN types via factory
    - Optional stochastic depth regularization

    **Architecture**:
    ```
    Input x (B, N, D) ──────────────────────────────────+
       ↓                                                │
    LayerNorm → MultiHeadAttention → LearnableMultiplier → DropPath ──(+)─→ x_mid
       ↓                                                                   │
    LayerNorm → FFN → LearnableMultiplier → DropPath ─────────────────(+)─→ Output
    ```

    **Intent**: Implement the core transformer block used in DINOv2 with modern
    enhancements like learnable scaling and stochastic depth for improved training
    stability and regularization.

    Args:
        dim: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dimension to embedding dimension.
        attention_type: Type of attention mechanism ('multi_head_attention', etc.).
        ffn_type: Type of FFN ('mlp', 'swiglu', etc.).
        normalization_type: Type of normalization ('layer_norm', 'rms_norm', etc.).
        qkv_bias: Whether to use bias in QKV projection.
        proj_bias: Whether to use bias in attention projection.
        ffn_bias: Whether to use bias in FFN layers.
        drop_path_rate: Stochastic depth drop probability.
        init_values: LearnableMultiplier initialization value (None disables scaling).
        attention_dropout: Dropout rate for attention.
        ffn_dropout: Dropout rate for FFN.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, embedding_dim)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, embedding_dim)`.
        Same shape as input due to residual connections.

    Example:
        ```python
        # Standard DINOv2 block
        block = DINOv2Block(
            dim=768,
            num_heads=12,
            mlp_ratio=4.0,
            drop_path_rate=0.1,
            init_values=1e-5
        )

        # With SwiGLU FFN (for giant model)
        block = DINOv2Block(
            dim=1536,
            num_heads=24,
            ffn_type='swiglu',
            drop_path_rate=0.3
        )
        ```
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            attention_type: str = 'multi_head_attention',
            ffn_type: str = 'mlp',
            normalization_type: str = 'layer_norm',
            qkv_bias: bool = True,
            proj_bias: bool = True,
            ffn_bias: bool = True,
            drop_path_rate: float = 0.0,
            init_values: Optional[float] = None,
            attention_dropout: float = 0.0,
            ffn_dropout: float = 0.0,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")

        # Store all configuration
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.qkv_bias = qkv_bias
        self.proj_bias = proj_bias
        self.ffn_bias = ffn_bias
        self.drop_path_rate = drop_path_rate
        self.init_values = init_values
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout

        # Create sub-layers in __init__ following Modern Keras 3 patterns

        # Normalization layers
        self.norm1 = create_normalization_layer(
            normalization_type=self.normalization_type,
            name="norm1"
        )
        self.norm2 = create_normalization_layer(
            normalization_type=self.normalization_type,
            name="norm2"
        )

        # Attention layer - map parameters appropriately
        attention_args = {
            'num_heads': self.num_heads,
            'dropout_rate': self.attention_dropout
        }

        if self.attention_type == 'multi_head_attention':
            attention_args['embed_dim'] = self.dim
            attention_args['use_bias'] = self.qkv_bias
        else:
            attention_args['dim'] = self.dim

        self.attention = create_attention_layer(
            attention_type=self.attention_type,
            name="attention",
            **attention_args
        )

        # FFN layer
        hidden_dim = int(self.dim * self.mlp_ratio)
        ffn_args = {
            'output_dim': self.dim,
            'dropout_rate': self.ffn_dropout,
            'use_bias': self.ffn_bias
        }

        if self.ffn_type in ['mlp', 'glu', 'geglu']:
            ffn_args['hidden_dim'] = hidden_dim
        elif self.ffn_type in ['swiglu']:
            ffn_args['ffn_expansion_factor'] = self.mlp_ratio

        self.ffn = create_ffn_layer(
            ffn_type=self.ffn_type,
            name="ffn",
            **ffn_args
        )

        # LearnableMultiplier for layer scaling (replaces LayerScale)
        if self.init_values is not None:
            self.ls1 = LearnableMultiplier(
                multiplier_type='CHANNEL',
                initializer=initializers.Constant(self.init_values),
                constraint='non_neg',
                name="ls1"
            )
            self.ls2 = LearnableMultiplier(
                multiplier_type='CHANNEL',
                initializer=initializers.Constant(self.init_values),
                constraint='non_neg',
                name="ls2"
            )
        else:
            self.ls1 = None
            self.ls2 = None

        # Stochastic depth (optional)
        if self.drop_path_rate > 0.0:
            self.drop_path = StochasticDepth(self.drop_path_rate, name="drop_path")
        else:
            self.drop_path = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the transformer block with all sub-components.

        **Critical**: Explicitly builds all sub-layers for proper serialization.
        """
        # Build normalization layers
        self.norm1.build(input_shape)
        self.norm2.build(input_shape)

        # Build attention layer
        self.attention.build(input_shape)

        # Build FFN layer
        self.ffn.build(input_shape)

        # Build LearnableMultiplier layers if used
        if self.ls1 is not None:
            self.ls1.build(input_shape)
            self.ls2.build(input_shape)

        # Build stochastic depth if used
        if self.drop_path is not None:
            self.drop_path.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """Forward pass of the transformer block."""
        # Pre-norm attention block
        x = self.norm1(inputs, training=training)
        x = self.attention(x, training=training)

        if self.ls1 is not None:
            x = self.ls1(x, training=training)

        if self.drop_path is not None:
            x = self.drop_path(x, training=training)

        x = inputs + x

        # Pre-norm FFN block
        y = self.norm2(x, training=training)
        y = self.ffn(y, training=training)

        if self.ls2 is not None:
            y = self.ls2(y, training=training)

        if self.drop_path is not None:
            y = self.drop_path(y, training=training)

        x = x + y

        return x

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute the output shape of the layer."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the layer."""
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "attention_type": self.attention_type,
            "ffn_type": self.ffn_type,
            "normalization_type": self.normalization_type,
            "qkv_bias": self.qkv_bias,
            "proj_bias": self.proj_bias,
            "ffn_bias": self.ffn_bias,
            "drop_path_rate": self.drop_path_rate,
            "init_values": self.init_values,
            "attention_dropout": self.attention_dropout,
            "ffn_dropout": self.ffn_dropout,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DINOv2VisionTransformer(keras.layers.Layer):
    """
    DINOv2 Vision Transformer backbone implementation following Modern Keras 3 patterns.

    This implementation provides a complete DINOv2 ViT backbone with:
    - Modern Keras 3 patterns: sub-layers created in __init__, built in build()
    - Configurable model sizes (tiny, small, base, large, giant)
    - Patch embedding with customizable patch sizes
    - Learnable positional embeddings with interpolation
    - Optional register tokens for improved attention quality
    - LearnableMultiplier for training stability
    - Stochastic depth regularization with linear decay
    - Support for both training and inference modes
    - Mask token for masked image modeling (iBOT objective)

    **Architecture**:
    ```
    Input Image (B, H, W, C)
         ↓
    PatchEmbed → (B, N, D)
         ↓
    [CLS] + [REG]₀₋ᵣ + Patches + PosEmbed → (B, 1+R+N, D)
         ↓
    TransformerBlock₁ → ... → TransformerBlockₗ
         ↓
    LayerNorm → Split: [CLS] | [REG] | [Patches]
    ```

    **Intent**: Provide the core Vision Transformer backbone used in DINOv2,
    supporting both self-supervised pre-training and downstream fine-tuning
    with configurable architecture and modern training enhancements.

    Args:
        img_size: Input image size (int or tuple).
        patch_size: Patch size for embedding (int or tuple).
        in_chans: Number of input channels.
        embed_dim: Embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        qkv_bias: Enable bias for QKV projections.
        proj_bias: Enable bias for attention projection.
        ffn_bias: Enable bias for FFN layers.
        drop_path_rate: Maximum stochastic depth rate.
        drop_path_uniform: Use uniform drop rate across blocks.
        init_values: LearnableMultiplier initialization value.
        attention_type: Type of attention mechanism.
        ffn_type: Type of feed-forward network.
        normalization_type: Type of normalization.
        num_register_tokens: Number of register tokens to use.
        interpolate_antialias: Use anti-aliasing for positional embedding interpolation.
        interpolate_offset: Offset for positional embedding interpolation.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.

    Output shape:
        - If `is_training=False`: 2D tensor with shape: `(batch_size, embed_dim)` (CLS token)
        - If `is_training=True`: Dictionary with keys:
            - 'x_norm_clstoken': CLS token features (B, D)
            - 'x_norm_regtokens': Register token features (B, R, D)
            - 'x_norm_patchtokens': Patch token features (B, N, D)
            - 'x_prenorm': Pre-normalization features (B, 1+R+N, D)
            - 'masks': Input masks (if provided)

    Example:
        ```python
        # Standard DINOv2-Base
        backbone = DINOv2VisionTransformer(
            embed_dim=768,
            depth=12,
            num_heads=12,
            drop_path_rate=0.1
        )

        # With register tokens
        backbone = DINOv2VisionTransformer(
            embed_dim=768,
            depth=12,
            num_heads=12,
            num_register_tokens=4
        )
        ```
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            proj_bias: bool = True,
            ffn_bias: bool = True,
            drop_path_rate: float = 0.0,
            drop_path_uniform: bool = False,
            init_values: Optional[float] = None,
            attention_type: str = 'multi_head_attention',
            ffn_type: str = 'mlp',
            normalization_type: str = 'layer_norm',
            num_register_tokens: int = 0,
            interpolate_antialias: bool = False,
            interpolate_offset: float = 0.1,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        # Store configuration
        self.img_size = img_size if isinstance(img_size, (tuple, list)) else (img_size, img_size)
        self.patch_size = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.proj_bias = proj_bias
        self.ffn_bias = ffn_bias
        self.drop_path_rate = drop_path_rate
        self.drop_path_uniform = drop_path_uniform
        self.init_values = init_values
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        # Computed attributes
        self.num_features = embed_dim
        self.num_tokens = 1  # CLS token
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])

        # Create sub-layers in __init__ following Modern Keras 3 patterns

        # Patch embedding using factory
        self.patch_embed = create_embedding_layer(
            'patch_2d',
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            name='patch_embed'
        )

        # Positional embeddings - use learned embeddings from factory
        pos_embed_seq_len = self.num_patches + self.num_tokens
        self.pos_embed = create_embedding_layer(
            'positional_learned',
            max_seq_len=pos_embed_seq_len,
            dim=self.embed_dim,
            name='pos_embed'
        )

        # Build transformer blocks with stochastic depth
        if self.drop_path_uniform:
            dpr = [self.drop_path_rate] * self.depth
        else:
            dpr = np.linspace(0, self.drop_path_rate, self.depth).tolist()

        self.blocks = []
        for i in range(self.depth):
            block = DINOv2Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attention_type=self.attention_type,
                ffn_type=self.ffn_type,
                normalization_type=self.normalization_type,
                qkv_bias=self.qkv_bias,
                proj_bias=self.proj_bias,
                ffn_bias=self.ffn_bias,
                drop_path_rate=dpr[i],
                init_values=self.init_values,
                name=f"block_{i}"
            )
            self.blocks.append(block)

        # Final normalization
        self.norm = create_normalization_layer(
            normalization_type=self.normalization_type,
            name="norm"
        )

        # Identity head (for pre-training)
        self.head = layers.Identity(name="head")

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the vision transformer with all components.

        **Critical**: Explicitly builds all sub-layers and creates trainable weights.
        """

        # Create trainable weights that couldn't be created in __init__

        # CLS token
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.embed_dim),
            initializer=initializers.TruncatedNormal(stddev=1e-6),
            trainable=True,
        )

        # Register tokens (optional)
        if self.num_register_tokens > 0:
            self.register_tokens = self.add_weight(
                name="register_tokens",
                shape=(1, self.num_register_tokens, self.embed_dim),
                initializer=initializers.TruncatedNormal(stddev=1e-6),
                trainable=True,
            )
        else:
            self.register_tokens = None

        # Mask token for iBOT objective
        self.mask_token = self.add_weight(
            name="mask_token",
            shape=(1, self.embed_dim),
            initializer=initializers.Zeros(),
            trainable=True,
        )

        # Build sub-layers explicitly for proper serialization

        # Build patch embedding
        patch_input_shape = input_shape
        self.patch_embed.build(patch_input_shape)

        # Compute shape after patch embedding
        batch_size = input_shape[0]
        embed_shape = (batch_size, self.num_patches, self.embed_dim)

        # Build positional embedding
        pos_embed_shape = (batch_size, self.num_patches + self.num_tokens, self.embed_dim)
        self.pos_embed.build(pos_embed_shape)

        # Build transformer blocks
        current_shape = pos_embed_shape
        for block in self.blocks:
            block.build(current_shape)

        # Build final normalization
        self.norm.build(current_shape)

        # Build head
        cls_shape = (batch_size, self.embed_dim)
        self.head.build(cls_shape)

        super().build(input_shape)

    def interpolate_pos_encoding(
            self,
            x: tf.Tensor,
            w: int,
            h: int
    ) -> tf.Tensor:
        """
        Interpolate positional embeddings for different input resolutions.

        **Mathematical Operation**:
        For resolution adaptation:
        1. Extract patch embeddings (excluding CLS token)
        2. Reshape to spatial grid: (1, N, D) → (1, H_orig, W_orig, D)
        3. Bilinear interpolate to target: (1, H_new, W_new, D)
        4. Reshape back: (1, N_new, D)
        5. Concatenate with CLS embedding: (1, 1+N_new, D)

        Args:
            x: Input tensor with sequence of patch embeddings.
            w: Width of input image.
            h: Height of input image.

        Returns:
            Interpolated positional embeddings.
        """
        npatch = keras.ops.shape(x)[1] - 1  # Exclude CLS token
        N = self.num_patches

        if npatch == N and w == h:
            # No interpolation needed
            pos_embed_weights = self.pos_embed.pos_embed
            return pos_embed_weights

        # Get weights from the positional embedding layer
        pos_embed_weights = self.pos_embed.pos_embed
        pos_embed = keras.ops.cast(pos_embed_weights, x.dtype)

        class_pos_embed = pos_embed[:, 0:1]  # CLS token embedding
        patch_pos_embed = pos_embed[:, 1:]  # Patch embeddings

        dim = keras.ops.shape(x)[-1]
        w0 = w // self.patch_size[0]
        h0 = h // self.patch_size[1]

        # Calculate original grid size
        M = int(np.sqrt(N))
        assert N == M * M, f"Number of patches {N} is not a perfect square"

        # Reshape and interpolate using tf.image.resize
        patch_pos_embed = keras.ops.reshape(patch_pos_embed, (1, M, M, dim))

        if self.interpolate_offset:
            # Historical offset for backward compatibility
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            new_size = (int(M * sx), int(M * sy))
        else:
            new_size = (w0, h0)

        # Use tf.image.resize for interpolation
        patch_pos_embed = tf.image.resize(
            patch_pos_embed,
            size=new_size,
            method='bicubic',
            antialias=self.interpolate_antialias
        )

        assert (w0, h0) == patch_pos_embed.shape[1:3], \
            f"Size mismatch: expected {(w0, h0)}, got {patch_pos_embed.shape[1:3]}"

        patch_pos_embed = keras.ops.reshape(patch_pos_embed, (1, -1, dim))

        return keras.ops.concatenate([class_pos_embed, patch_pos_embed], axis=1)

    def prepare_tokens_with_masks(
            self,
            x: tf.Tensor,
            masks: Optional[tf.Tensor] = None,
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """
        Prepare input tokens with optional masking for iBOT objective.

        **Token Preparation Pipeline**:
        1. PatchEmbed: (B, H, W, C) → (B, N, D)
        2. Masking (optional): Replace masked patches with mask_token
        3. Add CLS token: (B, N, D) → (B, 1+N, D)
        4. Add positional embeddings: x = x + pos_embed
        5. Add register tokens (optional): (B, 1+N, D) → (B, 1+R+N, D)

        Args:
            x: Input image tensor of shape (B, H, W, C).
            masks: Optional mask tensor for masked patches.
            training: Whether in training mode.

        Returns:
            Prepared token sequence with CLS, register, and patch tokens.
        """
        B, H, W, C = keras.ops.shape(x)[0], keras.ops.shape(x)[1], keras.ops.shape(x)[2], keras.ops.shape(x)[3]

        # Extract patches
        x = self.patch_embed(x, training=training)

        # Apply masking if provided (for iBOT)
        if masks is not None:
            mask_tokens = keras.ops.broadcast_to(self.mask_token, keras.ops.shape(x))
            mask_expanded = keras.ops.expand_dims(masks, -1)  # (B, N, 1)
            x = keras.ops.where(mask_expanded, mask_tokens, x)

        # Add CLS token
        cls_tokens = keras.ops.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        x = keras.ops.concatenate([cls_tokens, x], axis=1)

        # Add positional embeddings
        pos_embed = self.interpolate_pos_encoding(x, W, H)
        x = x + pos_embed

        # Add register tokens if configured
        if self.register_tokens is not None:
            register_tokens = keras.ops.broadcast_to(self.register_tokens,
                                                     (B, self.num_register_tokens, self.embed_dim))
            x = keras.ops.concatenate([
                x[:, :1],  # CLS token
                register_tokens,  # Register tokens
                x[:, 1:]  # Patch tokens
            ], axis=1)

        return x

    def forward_features(
            self,
            x: tf.Tensor,
            masks: Optional[tf.Tensor] = None,
            training: Optional[bool] = None
    ) -> Dict[str, tf.Tensor]:
        """
        Forward pass through the transformer backbone.

        Args:
            x: Input image tensor.
            masks: Optional mask tensor.
            training: Whether in training mode.

        Returns:
            Dictionary containing:
                - x_norm_clstoken: Normalized CLS token
                - x_norm_regtokens: Normalized register tokens
                - x_norm_patchtokens: Normalized patch tokens
                - x_prenorm: Pre-normalization features
                - masks: Input masks
        """
        x = self.prepare_tokens_with_masks(x, masks, training=training)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, training=training)

        # Final normalization
        x_norm = self.norm(x, training=training)

        # Split outputs
        cls_token = x_norm[:, 0]  # CLS token

        if self.register_tokens is not None:
            reg_start = 1
            reg_end = 1 + self.num_register_tokens
            reg_tokens = x_norm[:, reg_start:reg_end]  # Register tokens
            patch_tokens = x_norm[:, reg_end:]  # Patch tokens
        else:
            reg_tokens = tf.zeros((keras.ops.shape(x_norm)[0], 0, self.embed_dim), dtype=x_norm.dtype)
            patch_tokens = x_norm[:, 1:]  # Patch tokens

        return {
            "x_norm_clstoken": cls_token,
            "x_norm_regtokens": reg_tokens,
            "x_norm_patchtokens": patch_tokens,
            "x_prenorm": x,
            "masks": masks,
        }

    def call(
            self,
            inputs: tf.Tensor,
            masks: Optional[tf.Tensor] = None,
            is_training: bool = False,
            training: Optional[bool] = None
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Forward pass of the DINOv2 Vision Transformer.

        Args:
            inputs: Input image tensor of shape (B, H, W, C).
            masks: Optional mask tensor for iBOT objective.
            is_training: Whether to return full feature dictionary for training.
            training: Keras training flag.

        Returns:
            If is_training=True: Dictionary with all features
            If is_training=False: CLS token features only
        """
        ret = self.forward_features(inputs, masks, training=training)

        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute the output shape of the layer."""
        batch_size = input_shape[0]
        return tf.TensorShape([batch_size, self.embed_dim])

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the layer."""
        config = super().get_config()
        config.update({
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "in_chans": self.in_chans,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "proj_bias": self.proj_bias,
            "ffn_bias": self.ffn_bias,
            "drop_path_rate": self.drop_path_rate,
            "drop_path_uniform": self.drop_path_uniform,
            "init_values": self.init_values,
            "attention_type": self.attention_type,
            "ffn_type": self.ffn_type,
            "normalization_type": self.normalization_type,
            "num_register_tokens": self.num_register_tokens,
            "interpolate_antialias": self.interpolate_antialias,
            "interpolate_offset": self.interpolate_offset,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DINOv2Model(keras.Model):
    """
    Complete DINOv2 Model with proper call method following modern Keras 3 patterns.

    This model provides a high-level interface for DINOv2 with automatic
    input/output handling and model variant support. Unlike functional API
    construction, this uses a proper call method for better control and flexibility.

    **Architecture**:
    ```
    Input (B, H, W, C)
         ↓
    DINOv2VisionTransformer → Features (B, D)
         ↓
    Dense Classifier (optional) → Predictions (B, num_classes)
    ```

    **Intent**: Provide a complete model interface that can be used for both
    pre-training and fine-tuning, with proper call method implementation
    following modern Keras 3 best practices.

    Args:
        img_size: Input image size (int or tuple).
        patch_size: Patch size for patch embedding (int or tuple).
        num_classes: Number of output classes.
        include_top: Whether to include classification head.
        **backbone_kwargs: Additional arguments passed to backbone.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.

    Output shape:
        - If `include_top=True`: 2D tensor with shape: `(batch_size, num_classes)`
        - If `include_top=False`: 2D tensor with shape: `(batch_size, embed_dim)`

    Example:
        ```python
        # Pre-training model (no classification head)
        model = DINOv2Model(
            img_size=224,
            patch_size=14,
            include_top=False,
            embed_dim=768,
            depth=12,
            num_heads=12
        )

        # Fine-tuning model with classification head
        model = DINOv2Model(
            img_size=224,
            patch_size=14,
            num_classes=1000,
            include_top=True,
            embed_dim=768,
            depth=12,
            num_heads=12
        )

        # From variant
        model = DINOv2Model.from_variant('base', num_classes=100)
        ```
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        'tiny': {
            'embed_dim': 192,
            'depth': 12,
            'num_heads': 3,
            'mlp_ratio': 4.0,
        },
        'small': {
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6,
            'mlp_ratio': 4.0,
        },
        'base': {
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4.0,
        },
        'large': {
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'mlp_ratio': 4.0,
        },
        'giant': {
            'embed_dim': 1536,
            'depth': 40,
            'num_heads': 24,
            'mlp_ratio': 4.0,
        }
    }

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 14,
            num_classes: int = 1000,
            include_top: bool = True,
            **backbone_kwargs
    ) -> None:
        super().__init__(name='dinov2_model')

        # Validate inputs
        if num_classes <= 0 and include_top:
            raise ValueError(f"num_classes must be positive when include_top=True, got {num_classes}")

        # Store configuration
        self.img_size = img_size if isinstance(img_size, (tuple, list)) else (img_size, img_size)
        self.patch_size = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
        self.num_classes = num_classes
        self.include_top = include_top
        self.backbone_kwargs = backbone_kwargs

        # Create backbone
        self.backbone = DINOv2VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            name='dinov2_backbone',
            **backbone_kwargs
        )

        # Create classifier if needed
        if include_top and num_classes > 0:
            self.classifier = layers.Dense(
                num_classes,
                kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
                name='classifier'
            )
        else:
            self.classifier = None

    def call(
            self,
            inputs: tf.Tensor,
            masks: Optional[tf.Tensor] = None,
            is_training: bool = False,
            training: Optional[bool] = None
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Forward pass of the DINOv2 model.

        **Call Method Implementation**: This method implements the forward pass
        directly rather than using functional API construction, following
        modern Keras 3 best practices for custom models.

        Args:
            inputs: Input image tensor of shape (B, H, W, C).
            masks: Optional mask tensor for iBOT objective.
            is_training: Whether to return full feature dictionary (used in pre-training).
            training: Keras training flag.

        Returns:
            - If is_training=True: Dictionary with all backbone features
            - If is_training=False and include_top=True: Class predictions (B, num_classes)
            - If is_training=False and include_top=False: Backbone features (B, embed_dim)
        """
        # Get backbone output
        backbone_output = self.backbone(
            inputs,
            masks=masks,
            is_training=is_training,
            training=training
        )

        # Handle different output modes
        if is_training:
            # Return full feature dictionary for pre-training losses
            return backbone_output

        # backbone_output is CLS token features (B, embed_dim)
        if self.include_top and self.classifier is not None:
            # Apply classification head
            return self.classifier(backbone_output, training=training)

        # Return raw features
        return backbone_output

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute the output shape of the model."""
        batch_size = input_shape[0]
        if self.include_top and self.classifier is not None:
            return tf.TensorShape([batch_size, self.num_classes])
        else:
            embed_dim = self.backbone_kwargs.get('embed_dim', 768)
            return tf.TensorShape([batch_size, embed_dim])

    @classmethod
    def from_variant(
            cls,
            variant: Literal['tiny', 'small', 'base', 'large', 'giant'],
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 14,
            num_classes: int = 1000,
            include_top: bool = True,
            **kwargs
    ) -> "DINOv2Model":
        """
        Create DINOv2 model from predefined variant.

        Args:
            variant: Size variant ('tiny', 'small', 'base', 'large', 'giant').
            img_size: Input image size.
            patch_size: Patch size for patch embedding.
            num_classes: Number of output classes.
            include_top: Whether to include classification head.
            **kwargs: Additional arguments for the model.

        Returns:
            DINOv2Model instance.

        Raises:
            ValueError: If variant is not recognized.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(cls.MODEL_VARIANTS.keys())}")

        config = cls.MODEL_VARIANTS[variant].copy()
        config.update(kwargs)

        return cls(
            img_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            include_top=include_top,
            **config
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'num_classes': self.num_classes,
            'include_top': self.include_top,
            **self.backbone_kwargs
        }

# ---------------------------------------------------------------------

def create_dinov2_model(
        variant: Literal['tiny', 'small', 'base', 'large', 'giant'] = 'base',
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 14,
        num_register_tokens: int = 0,
        init_values: Optional[float] = 1e-5,
        drop_path_rate: float = 0.0,
        ffn_type: str = 'mlp',
        num_classes: int = 1000,
        include_top: bool = True,
        **kwargs
) -> DINOv2Model:
    """
    Factory function to create DINOv2 model variants with sensible defaults.

    **Recommended Configurations**:
    - **tiny/small/base/large**: Use 'mlp' FFN, 0 or 4 register tokens
    - **giant**: Use 'swiglu' FFN, 4 register tokens, higher drop_path_rate

    Args:
        variant: Size variant ('tiny', 'small', 'base', 'large', 'giant').
        img_size: Input image size.
        patch_size: Patch size for patch embedding.
        num_register_tokens: Number of register tokens (0 or 4 typically).
        init_values: LearnableMultiplier initialization value.
        drop_path_rate: Maximum stochastic depth rate.
        ffn_type: Type of FFN ('mlp' for small models, 'swiglu' for giant).
        num_classes: Number of output classes.
        include_top: Whether to include classification head.
        **kwargs: Additional arguments for the model.

    Returns:
        DINOv2Model instance.

    Example:
        ```python
        # Standard DINOv2-Base for ImageNet
        model = create_dinov2_model('base', num_classes=1000)

        # DINOv2-Giant with SwiGLU and register tokens
        model = create_dinov2_model(
            'giant',
            ffn_type='swiglu',
            num_register_tokens=4,
            drop_path_rate=0.3
        )

        # Pre-training model (no classification head)
        model = create_dinov2_model('base', include_top=False)
        ```
    """
    # Use SwiGLU for giant model by default
    if variant == 'giant' and ffn_type == 'mlp':
        ffn_type = 'swiglu'

    return DINOv2Model.from_variant(
        variant,
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        include_top=include_top,
        num_register_tokens=num_register_tokens,
        init_values=init_values,
        drop_path_rate=drop_path_rate,
        ffn_type=ffn_type,
        **kwargs
    )