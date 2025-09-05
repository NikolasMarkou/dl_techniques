"""
JEPA Encoder Implementation using Vision Transformer Architecture.

This module implements the context and target encoders for JEPA, following modern
Keras 3.8.0 patterns with full serialization support and memory optimizations.
"""

import keras
import tensorflow as tf
from keras import layers, ops, initializers
from typing import Optional, Union, Tuple, Dict, Any
import numpy as np

from dl_techniques.layers.transformer import TransformerLayer


@keras.utils.register_keras_serializable(package="JEPA")
class JEPAPatchEmbedding(layers.Layer):
    """
    Advanced patch embedding layer for JEPA with support for different modalities.

    This layer handles patch tokenization for images, video, and audio spectrograms
    with optimized memory usage and proper positional encoding.

    Args:
        patch_size: Size of patches (int or tuple for 2D, tuple for 3D video).
        embed_dim: Embedding dimension for patch tokens.
        img_size: Input image/spectrogram size.
        variant: Modality variant ("image", "video", "audio").
        kernel_initializer: Weight initialization strategy.
        **kwargs: Additional layer arguments.

    Input shapes:
        - Images: (batch_size, height, width, channels)
        - Video: (batch_size, num_frames, height, width, channels)
        - Audio: (batch_size, freq_bins, time_frames, 1)

    Output shape:
        (batch_size, num_patches, embed_dim)
    """

    def __init__(
            self,
            patch_size: Union[int, Tuple[int, ...]],
            embed_dim: int,
            img_size: Tuple[int, ...],
            variant: str = "image",
            kernel_initializer: Union[str, initializers.Initializer] = "truncated_normal",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.patch_size = patch_size if isinstance(patch_size, (list, tuple)) else (patch_size, patch_size)
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.variant = variant
        self.kernel_initializer = initializers.get(kernel_initializer)

        # Compute number of patches
        if variant == "video":
            # 3D patches for video: (time, height, width)
            self.num_patches = (
                (img_size[0] // self.patch_size[0]) *
                (img_size[1] // self.patch_size[1]) *
                (img_size[2] // self.patch_size[2]) if len(img_size) == 3
                else (img_size[0] // self.patch_size[0]) * (img_size[1] // self.patch_size[1])
            )
        else:
            # 2D patches for images and audio spectrograms
            self.num_patches = (img_size[0] // self.patch_size[0]) * (img_size[1] // self.patch_size[1])

        # Create projection layer
        if variant == "video" and len(img_size) == 3:
            # 3D convolution for video
            self.projection = layers.Conv3D(
                embed_dim,
                kernel_size=self.patch_size,
                strides=self.patch_size,
                kernel_initializer=self.kernel_initializer,
                name="video_projection"
            )
        else:
            # 2D convolution for images and audio
            self.projection = layers.Conv2D(
                embed_dim,
                kernel_size=self.patch_size,
                strides=self.patch_size,
                kernel_initializer=self.kernel_initializer,
                name="patch_projection"
            )

        self.flatten = layers.Reshape((-1, embed_dim))

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """
        Apply patch embedding to inputs.

        Args:
            inputs: Input tensor.
            training: Training mode flag.

        Returns:
            Patch embeddings of shape (batch_size, num_patches, embed_dim).
        """
        batch_size = ops.shape(inputs)[0]

        # Apply patch projection
        x = self.projection(inputs)

        # Reshape to sequence format
        if self.variant == "video" and len(self.img_size) == 3:
            # For video: (B, T/pt, H/ph, W/pw, D) -> (B, T*H*W/pt*ph*pw, D)
            x = self.flatten(x)
        else:
            # For images/audio: (B, H/ph, W/pw, D) -> (B, H*W/ph*pw, D)
            x = self.flatten(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        batch_size = input_shape[0]
        return (batch_size, self.num_patches, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "img_size": self.img_size,
            "variant": self.variant,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
        })
        return config


@keras.utils.register_keras_serializable(package="JEPA")
class JEPAEncoder(layers.Layer):
    """
    JEPA Encoder using Vision Transformer architecture with modern optimizations.

    This encoder processes patch sequences through transformer blocks with optional
    gradient checkpointing for memory efficiency and mixed precision support.

    Args:
        embed_dim: Embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        patch_size: Patch size for tokenization.
        img_size: Input image size.
        variant: Modality variant.
        dropout_rate: Dropout rate.
        drop_path_rate: Stochastic depth drop path rate.
        layer_scale_init: Layer scale initialization.
        use_layer_scale: Enable layer scale.
        use_gradient_checkpointing: Enable gradient checkpointing.
        activation: Activation function.
        norm_type: Normalization type.
        kernel_initializer: Weight initialization.
        bias_initializer: Bias initialization.
        **kwargs: Additional layer arguments.

    Input shape:
        (batch_size, height, width, channels) for images
        (batch_size, num_frames, height, width, channels) for video

    Output shape:
        (batch_size, num_patches, embed_dim)
    """

    def __init__(
            self,
            embed_dim: int,
            depth: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            patch_size: Union[int, Tuple[int, ...]] = 16,
            img_size: Tuple[int, ...] = (224, 224),
            variant: str = "image",
            dropout_rate: float = 0.0,
            drop_path_rate: float = 0.1,
            layer_scale_init: float = 1e-4,
            use_layer_scale: bool = True,
            use_gradient_checkpointing: bool = False,
            activation: str = "gelu",
            norm_type: str = "layer_norm",
            kernel_initializer: Union[str, initializers.Initializer] = "truncated_normal",
            bias_initializer: Union[str, initializers.Initializer] = "zeros",
            **kwargs
    ):
        super().__init__(**kwargs)

        # Store configuration
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.img_size = img_size
        self.variant = variant
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init = layer_scale_init
        self.use_layer_scale = use_layer_scale
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.activation = activation
        self.norm_type = norm_type
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # Create patch embedding
        self.patch_embed = JEPAPatchEmbedding(
            patch_size=patch_size,
            embed_dim=embed_dim,
            img_size=img_size,
            variant=variant,
            kernel_initializer=kernel_initializer,
            name="patch_embed"
        )

        # Class token for global representation
        self.use_cls_token = True
        if self.use_cls_token:
            self.cls_token = self.add_weight(
                name="cls_token",
                shape=(1, 1, embed_dim),
                initializer="truncated_normal",
                trainable=True
            )

        # Positional embeddings
        num_patches = self.patch_embed.num_patches
        pos_embed_length = num_patches + (1 if self.use_cls_token else 0)

        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, pos_embed_length, embed_dim),
            initializer="truncated_normal",
            trainable=True
        )

        # Dropout after embeddings
        self.pos_drop = layers.Dropout(dropout_rate, name="pos_drop")

        # Transformer blocks with stochastic depth
        self.blocks = []
        dpr = np.linspace(0, drop_path_rate, depth)  # Stochastic depth decay rule

        for i in range(depth):
            block = TransformerLayer(
                hidden_size=embed_dim,
                num_heads=num_heads,
                intermediate_size=int(embed_dim * mlp_ratio),
                attention_type="multi_head_attention",
                normalization_type=norm_type,
                normalization_position="pre",
                ffn_type="mlp",
                dropout_rate=dropout_rate,
                use_stochastic_depth=drop_path_rate > 0,
                stochastic_depth_rate=dpr[i],
                activation=activation,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name=f"block_{i}"
            )
            self.blocks.append(block)

        # Final normalization
        if norm_type == "layer_norm":
            self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")
        else:
            from dl_techniques.layers.norms.rms_norm import RMSNorm
            self.norm = RMSNorm(name="norm")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
            return_all_tokens: bool = False
    ) -> keras.KerasTensor:
        """
        Forward pass through JEPA encoder.

        Args:
            inputs: Input tensor.
            training: Training mode flag.
            return_all_tokens: Return all tokens or just CLS token.

        Returns:
            Encoded representations.
        """
        batch_size = ops.shape(inputs)[0]

        # Patch embedding
        x = self.patch_embed(inputs, training=training)

        # Add CLS token
        if self.use_cls_token:
            cls_token = ops.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))
            x = ops.concatenate([cls_token, x], axis=1)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x, training=training)

        # Apply transformer blocks
        for i, block in enumerate(self.blocks):
            if self.use_gradient_checkpointing and training:
                # Use gradient checkpointing for memory efficiency
                x = tf.recompute_grad(lambda x, b=block: b(x, training=True))(x)
            else:
                x = block(x, training=training)

        # Final normalization
        x = self.norm(x, training=training)

        if return_all_tokens or not self.use_cls_token:
            return x
        else:
            # Return only CLS token for classification
            return x[:, 0]

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        batch_size = input_shape[0]
        num_patches = self.patch_embed.num_patches
        seq_length = num_patches + (1 if self.use_cls_token else 0)
        return (batch_size, seq_length, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "patch_size": self.patch_size,
            "img_size": self.img_size,
            "variant": self.variant,
            "dropout_rate": self.dropout_rate,
            "drop_path_rate": self.drop_path_rate,
            "layer_scale_init": self.layer_scale_init,
            "use_layer_scale": self.use_layer_scale,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "activation": self.activation,
            "norm_type": self.norm_type,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
        })
        return config


@keras.utils.register_keras_serializable(package="JEPA")
class JEPAPredictor(layers.Layer):
    """
    JEPA Predictor network for masked token prediction.

    The predictor is intentionally smaller than the encoder to enforce abstraction
    and prevent trivial pixel-level memorization. It takes context tokens and
    masked position embeddings to predict target representations.

    Args:
        embed_dim: Embedding dimension.
        depth: Number of transformer blocks (typically 50% of encoder depth).
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        dropout_rate: Dropout rate.
        drop_path_rate: Stochastic depth drop path rate.
        activation: Activation function.
        norm_type: Normalization type.
        kernel_initializer: Weight initialization.
        bias_initializer: Bias initialization.
        **kwargs: Additional layer arguments.

    Input shapes:
        - context_tokens: (batch_size, num_context, embed_dim)
        - mask_tokens: (batch_size, num_masked, embed_dim)

    Output shape:
        (batch_size, num_masked, embed_dim)
    """

    def __init__(
            self,
            embed_dim: int,
            depth: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            dropout_rate: float = 0.0,
            drop_path_rate: float = 0.1,
            activation: str = "gelu",
            norm_type: str = "layer_norm",
            kernel_initializer: Union[str, initializers.Initializer] = "truncated_normal",
            bias_initializer: Union[str, initializers.Initializer] = "zeros",
            **kwargs
    ):
        super().__init__(**kwargs)

        # Store configuration
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        self.activation = activation
        self.norm_type = norm_type
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # Learnable mask token
        self.mask_token = self.add_weight(
            name="mask_token",
            shape=(1, 1, embed_dim),
            initializer="truncated_normal",
            trainable=True
        )

        # Transformer blocks for prediction
        self.blocks = []
        dpr = np.linspace(0, drop_path_rate, depth)  # Stochastic depth decay

        for i in range(depth):
            block = TransformerLayer(
                hidden_size=embed_dim,
                num_heads=num_heads,
                intermediate_size=int(embed_dim * mlp_ratio),
                attention_type="multi_head_attention",
                normalization_type=norm_type,
                normalization_position="pre",
                ffn_type="mlp",
                dropout_rate=dropout_rate,
                use_stochastic_depth=drop_path_rate > 0,
                stochastic_depth_rate=dpr[i],
                activation=activation,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name=f"pred_block_{i}"
            )
            self.blocks.append(block)

        # Final normalization
        if norm_type == "layer_norm":
            self.norm = layers.LayerNormalization(epsilon=1e-6, name="pred_norm")
        else:
            from dl_techniques.layers.norms.rms_norm import RMSNorm
            self.norm = RMSNorm(name="pred_norm")

    def call(
            self,
            context_tokens: keras.KerasTensor,
            mask_positions: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Predict masked token representations.

        Args:
            context_tokens: Context (visible) patch embeddings.
            mask_positions: Positional embeddings for masked locations.
            training: Training mode flag.

        Returns:
            Predicted representations for masked positions.
        """
        batch_size = ops.shape(context_tokens)[0]
        num_masked = ops.shape(mask_positions)[1]

        # Create mask tokens with positional embeddings
        mask_tokens = ops.broadcast_to(self.mask_token, (batch_size, num_masked, self.embed_dim))
        mask_tokens = mask_tokens + mask_positions

        # Combine context and mask tokens
        x = ops.concatenate([context_tokens, mask_tokens], axis=1)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, training=training)

        # Final normalization
        x = self.norm(x, training=training)

        # Extract predictions for masked positions
        num_context = ops.shape(context_tokens)[1]
        predictions = x[:, num_context:]

        return predictions

    def compute_output_shape(
            self,
            input_shape: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        context_shape, mask_shape = input_shape
        batch_size = mask_shape[0]
        num_masked = mask_shape[1]
        return (batch_size, num_masked, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout_rate": self.dropout_rate,
            "drop_path_rate": self.drop_path_rate,
            "activation": self.activation,
            "norm_type": self.norm_type,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
        })
        return config