"""
Vision Language Model (VLM) Implementation using dl_techniques components.

This module implements a comprehensive Vision Language Model that can handle
various multimodal tasks such as image captioning, visual question answering,
and image-text matching.
"""

import keras
from keras import ops, layers
from typing import Dict, Optional, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from ..layers.norms.rms_norm import RMSNorm
from ..layers.ffn.swiglu_ffn import SwiGLUFFN
from ..layers.transformer import TransformerLayer
from ..layers.tokenizers.bpe import TokenEmbedding
from ..layers.patch_embedding import PatchEmbedding2D
from ..layers.positional_embedding import PositionalEmbedding
from ..layers.vision_transformer import VisionTransformerLayer
from ..layers.attention.multi_head_attention import MultiHeadAttention
from ..layers.geometric.shared_weights_cross_attention import SharedWeightsCrossAttention

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class VisionEncoder(keras.layers.Layer):
    """
    Vision encoder using Vision Transformer architecture.

    Args:
        image_size: Input image size as (height, width).
        patch_size: Size of image patches.
        embed_dim: Embedding dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dimension to embedding dimension.
        dropout_rate: Dropout rate.
        layer_norm_epsilon: Layer normalization epsilon.
        use_cls_token: Whether to use a classification token.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
            self,
            image_size: Tuple[int, int] = (224, 224),
            patch_size: Tuple[int, int] = (16, 16),
            embed_dim: int = 768,
            num_layers: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            dropout_rate: float = 0.1,
            layer_norm_epsilon: float = 1e-6,
            use_cls_token: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_cls_token = use_cls_token

        # Calculate number of patches
        self.num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])

        # Components will be initialized in build()
        self.patch_embedding = None
        self.cls_token = None
        self.position_embedding = None
        self.transformer_layers = []
        self.layer_norm = None
        self.dropout = None

    def build(self, input_shape):
        """Build the vision encoder layers."""
        # Patch embedding
        self.patch_embedding = PatchEmbedding2D(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            name="patch_embedding"
        )

        # CLS token
        if self.use_cls_token:
            self.cls_token = self.add_weight(
                name="cls_token",
                shape=(1, 1, self.embed_dim),
                initializer="truncated_normal",
                trainable=True
            )

        # Positional embedding
        seq_len = self.num_patches + (1 if self.use_cls_token else 0)
        self.position_embedding = PositionalEmbedding(
            max_seq_len=seq_len,
            dim=self.embed_dim,
            dropout=self.dropout_rate,
            name="position_embedding"
        )

        # Transformer layers
        for i in range(self.num_layers):
            transformer_layer = VisionTransformerLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                layer_norm_epsilon=self.layer_norm_epsilon,
                name=f"transformer_layer_{i}"
            )
            self.transformer_layers.append(transformer_layer)

        # Final layer norm
        self.layer_norm = RMSNorm(
            epsilon=self.layer_norm_epsilon,
            name="final_layer_norm"
        )

        # Dropout
        self.dropout = layers.Dropout(self.dropout_rate, name="dropout")

        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Forward pass of the vision encoder.

        Args:
            inputs: Input images of shape (batch_size, height, width, channels).
            training: Whether in training mode.

        Returns:
            Encoded vision features of shape (batch_size, seq_len, embed_dim).
        """
        batch_size = ops.shape(inputs)[0]

        # Patch embedding
        x = self.patch_embedding(inputs, training=training)  # (batch, num_patches, embed_dim)

        # Add CLS token if used
        if self.use_cls_token:
            cls_tokens = ops.tile(self.cls_token, [batch_size, 1, 1])
            x = ops.concatenate([cls_tokens, x], axis=1)

        # Add positional embedding
        x = self.position_embedding(x, training=training)

        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, training=training)

        # Final layer norm and dropout
        x = self.layer_norm(x)
        x = self.dropout(x, training=training)

        return x

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout_rate": self.dropout_rate,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "use_cls_token": self.use_cls_token,
        })
        return config


@keras.saving.register_keras_serializable()
class TextEncoder(keras.layers.Layer):
    """
    Text encoder using Transformer architecture.

    Args:
        vocab_size: Size of the vocabulary.
        embed_dim: Embedding dimension.
        max_seq_len: Maximum sequence length.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        intermediate_size: Size of the intermediate layer in FFN.
        dropout_rate: Dropout rate.
        layer_norm_epsilon: Layer normalization epsilon.
        use_causal_mask: Whether to use causal masking.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 768,
            max_seq_len: int = 512,
            num_layers: int = 12,
            num_heads: int = 12,
            intermediate_size: int = 3072,
            dropout_rate: float = 0.1,
            layer_norm_epsilon: float = 1e-6,
            use_causal_mask: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_causal_mask = use_causal_mask

        # Components will be initialized in build()
        self.token_embedding = None
        self.position_embedding = None
        self.transformer_layers = []
        self.layer_norm = None
        self.dropout = None

    def build(self, input_shape):
        """Build the text encoder layers."""
        # Token embedding
        self.token_embedding = TokenEmbedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.embed_dim,
            mask_zero=True,
            name="token_embedding"
        )

        # Positional embedding
        self.position_embedding = PositionalEmbedding(
            max_seq_len=self.max_seq_len,
            dim=self.embed_dim,
            dropout=self.dropout_rate,
            name="position_embedding"
        )

        # Transformer layers
        for i in range(self.num_layers):
            transformer_layer = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                dropout_rate=self.dropout_rate,
                layer_norm_epsilon=self.layer_norm_epsilon,
                use_causal_mask=self.use_causal_mask,
                name=f"transformer_layer_{i}"
            )
            self.transformer_layers.append(transformer_layer)

        # Final layer norm
        self.layer_norm = RMSNorm(
            epsilon=self.layer_norm_epsilon,
            name="final_layer_norm"
        )

        # Dropout
        self.dropout = layers.Dropout(self.dropout_rate, name="dropout")

        super().build(input_shape)

    def call(self, inputs, attention_mask=None, training=None):
        """
        Forward pass of the text encoder.

        Args:
            inputs: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            training: Whether in training mode.

        Returns:
            Encoded text features of shape (batch_size, seq_len, embed_dim).
        """
        # Token embedding
        x = self.token_embedding(inputs, training=training)

        # Add positional embedding
        x = self.position_embedding(x, training=training)

        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, attention_mask=attention_mask, training=training)

        # Final layer norm and dropout
        x = self.layer_norm(x)
        x = self.dropout(x, training=training)

        return x

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "max_seq_len": self.max_seq_len,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "dropout_rate": self.dropout_rate,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "use_causal_mask": self.use_causal_mask,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class CrossModalFusion(keras.layers.Layer):
    """
    Cross-modal fusion layer using cross-attention mechanisms.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        num_fusion_layers: Number of cross-attention layers.
        dropout_rate: Dropout rate.
        use_shared_weights: Whether to use shared weights cross-attention.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
            self,
            embed_dim: int = 768,
            num_heads: int = 12,
            num_fusion_layers: int = 6,
            dropout_rate: float = 0.1,
            use_shared_weights: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_fusion_layers = num_fusion_layers
        self.dropout_rate = dropout_rate
        self.use_shared_weights = use_shared_weights

        # Components will be initialized in build()
        self.vision_to_text_layers = []
        self.text_to_vision_layers = []
        self.vision_ffn_layers = []
        self.text_ffn_layers = []
        self.vision_norm_layers = []
        self.text_norm_layers = []

    def build(self, input_shape):
        """Build the cross-modal fusion layers."""
        for i in range(self.num_fusion_layers):
            # Cross-attention layers
            if self.use_shared_weights:
                v2t_attention = SharedWeightsCrossAttention(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout_rate,
                    name=f"vision_to_text_attention_{i}"
                )
                t2v_attention = SharedWeightsCrossAttention(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout_rate,
                    name=f"text_to_vision_attention_{i}"
                )
            else:
                v2t_attention = MultiHeadAttention(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    name=f"vision_to_text_attention_{i}"
                )
                t2v_attention = MultiHeadAttention(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    name=f"text_to_vision_attention_{i}"
                )

            self.vision_to_text_layers.append(v2t_attention)
            self.text_to_vision_layers.append(t2v_attention)

            # FFN layers
            vision_ffn = SwiGLUFFN(
                d_model=self.embed_dim,
                name=f"vision_ffn_{i}"
            )
            text_ffn = SwiGLUFFN(
                d_model=self.embed_dim,
                name=f"text_ffn_{i}"
            )

            self.vision_ffn_layers.append(vision_ffn)
            self.text_ffn_layers.append(text_ffn)

            # Normalization layers
            vision_norm = RMSNorm(name=f"vision_norm_{i}")
            text_norm = RMSNorm(name=f"text_norm_{i}")

            self.vision_norm_layers.append(vision_norm)
            self.text_norm_layers.append(text_norm)

        super().build(input_shape)

    def call(self, vision_features, text_features, attention_mask=None, training=None):
        """
        Forward pass of the cross-modal fusion.

        Args:
            vision_features: Vision features of shape (batch_size, vision_seq_len, embed_dim).
            text_features: Text features of shape (batch_size, text_seq_len, embed_dim).
            attention_mask: Text attention mask of shape (batch_size, text_seq_len).
            training: Whether in training mode.

        Returns:
            Tuple of (fused_vision_features, fused_text_features).
        """
        v_features = vision_features
        t_features = text_features

        for i in range(self.num_fusion_layers):
            # Cross-attention
            if self.use_shared_weights:
                # Vision attending to text
                v_attended = self.vision_to_text_layers[i](
                    v_features, t_features, training=training
                )
                # Text attending to vision
                t_attended = self.text_to_vision_layers[i](
                    t_features, v_features, training=training
                )
            else:
                # Vision attending to text (query=vision, key=value=text)
                v_attended = self.vision_to_text_layers[i](
                    v_features, context=t_features, training=training
                )
                # Text attending to vision (query=text, key=value=vision)
                t_attended = self.text_to_vision_layers[i](
                    t_features, context=v_features, training=training
                )

            # Residual connection and normalization
            v_features = self.vision_norm_layers[i](v_features + v_attended)
            t_features = self.text_norm_layers[i](t_features + t_attended)

            # FFN
            v_ffn_out = self.vision_ffn_layers[i](v_features, training=training)
            t_ffn_out = self.text_ffn_layers[i](t_features, training=training)

            # Residual connection
            v_features = v_features + v_ffn_out
            t_features = t_features + t_ffn_out

        return v_features, t_features

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_fusion_layers": self.num_fusion_layers,
            "dropout_rate": self.dropout_rate,
            "use_shared_weights": self.use_shared_weights,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class VisionLanguageModel(keras.Model):
    """
    Vision Language Model for multimodal tasks.

    This model combines vision and text encoders with cross-modal fusion
    to handle various vision-language tasks.

    Args:
        vision_config: Configuration for the vision encoder.
        text_config: Configuration for the text encoder.
        fusion_config: Configuration for cross-modal fusion.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
            self,
            vision_config: Dict[str, Any],
            text_config: Dict[str, Any],
            fusion_config: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_config = vision_config
        self.text_config = text_config
        self.fusion_config = fusion_config or {}

        # Initialize encoders
        self.vision_encoder = VisionEncoder(**vision_config)
        self.text_encoder = TextEncoder(**text_config)

        # Initialize fusion layer if specified
        if fusion_config:
            self.cross_modal_fusion = CrossModalFusion(**fusion_config)
        else:
            self.cross_modal_fusion = None

        # Projection layers for similarity computation
        self.vision_projection = layers.Dense(
            vision_config.get("embed_dim", 768),
            name="vision_projection",
            kernel_initializer="glorot_normal"
        )
        self.text_projection = layers.Dense(
            text_config.get("embed_dim", 768),
            name="text_projection",
            kernel_initializer="glorot_normal"
        )

        logger.info("VisionLanguageModel initialized successfully")

    def call(self, inputs, training=None):
        """
        Forward pass of the vision language model.

        Args:
            inputs: Dictionary containing 'images' and 'text_tokens' keys.
                   Optionally contains 'attention_mask'.
            training: Whether in training mode.

        Returns:
            Dictionary containing various model outputs.
        """
        images = inputs["images"]
        text_tokens = inputs["text_tokens"]
        attention_mask = inputs.get("attention_mask", None)

        # Encode vision and text
        vision_features = self.vision_encoder(images, training=training)
        text_features = self.text_encoder(
            text_tokens, attention_mask=attention_mask, training=training
        )

        # Cross-modal fusion if enabled
        if self.cross_modal_fusion is not None:
            fused_vision, fused_text = self.cross_modal_fusion(
                vision_features, text_features,
                attention_mask=attention_mask, training=training
            )
        else:
            fused_vision = vision_features
            fused_text = text_features

        # Global pooling for similarity computation
        # For vision: use CLS token if available, otherwise mean pooling
        if self.vision_encoder.use_cls_token:
            vision_global = fused_vision[:, 0]  # CLS token
        else:
            vision_global = ops.mean(fused_vision, axis=1)  # Mean pooling

        # For text: use first token (often CLS) or mean pooling of non-masked tokens
        if attention_mask is not None:
            # Masked mean pooling
            mask_expanded = ops.expand_dims(ops.cast(attention_mask, fused_text.dtype), -1)
            text_sum = ops.sum(fused_text * mask_expanded, axis=1)
            mask_sum = ops.sum(mask_expanded, axis=1)
            text_global = text_sum / (mask_sum + 1e-8)
        else:
            text_global = ops.mean(fused_text, axis=1)  # Simple mean pooling

        # Project to common space for similarity computation
        vision_projected = self.vision_projection(vision_global)
        text_projected = self.text_projection(text_global)

        # Normalize for cosine similarity
        vision_projected = ops.l2_normalize(vision_projected, axis=-1)
        text_projected = ops.l2_normalize(text_projected, axis=-1)

        # Compute similarity matrix
        similarity_matrix = ops.matmul(vision_projected, ops.transpose(text_projected))

        return {
            "vision_features": vision_features,
            "text_features": text_features,
            "fused_vision_features": fused_vision,
            "fused_text_features": fused_text,
            "vision_global": vision_projected,
            "text_global": text_projected,
            "similarity_matrix": similarity_matrix,
        }

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "vision_config": self.vision_config,
            "text_config": self.text_config,
            "fusion_config": self.fusion_config,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        return cls(**config)

# ---------------------------------------------------------------------

def create_vlm_for_image_captioning(
        image_size: Tuple[int, int] = (224, 224),
        vocab_size: int = 50000,
        max_text_length: int = 128
) -> VisionLanguageModel:
    """
    Create a VLM optimized for image captioning tasks.

    Args:
        image_size: Input image size.
        vocab_size: Size of text vocabulary.
        max_text_length: Maximum text sequence length.

    Returns:
        Configured VisionLanguageModel instance.
    """
    vision_config = {
        "image_size": image_size,
        "patch_size": (16, 16),
        "embed_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "dropout_rate": 0.1,
        "use_cls_token": True,
    }

    text_config = {
        "vocab_size": vocab_size,
        "embed_dim": 768,
        "max_seq_len": max_text_length,
        "num_layers": 12,
        "num_heads": 12,
        "intermediate_size": 3072,
        "dropout_rate": 0.1,
        "use_causal_mask": True,  # For autoregressive generation
    }

    fusion_config = {
        "embed_dim": 768,
        "num_heads": 12,
        "num_fusion_layers": 6,
        "dropout_rate": 0.1,
        "use_shared_weights": True,
    }

    return VisionLanguageModel(
        vision_config=vision_config,
        text_config=text_config,
        fusion_config=fusion_config,
        name="image_captioning_vlm"
    )

# ---------------------------------------------------------------------

def create_vlm_for_vqa(
        image_size: Tuple[int, int] = (224, 224),
        vocab_size: int = 50000,
        max_text_length: int = 256
) -> VisionLanguageModel:
    """
    Create a VLM optimized for Visual Question Answering tasks.

    Args:
        image_size: Input image size.
        vocab_size: Size of text vocabulary.
        max_text_length: Maximum text sequence length.

    Returns:
        Configured VisionLanguageModel instance.
    """
    vision_config = {
        "image_size": image_size,
        "patch_size": (16, 16),
        "embed_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "dropout_rate": 0.1,
        "use_cls_token": True,
    }

    text_config = {
        "vocab_size": vocab_size,
        "embed_dim": 768,
        "max_seq_len": max_text_length,
        "num_layers": 12,
        "num_heads": 12,
        "intermediate_size": 3072,
        "dropout_rate": 0.1,
        "use_causal_mask": False,  # Bidirectional for understanding questions
    }

    fusion_config = {
        "embed_dim": 768,
        "num_heads": 12,
        "num_fusion_layers": 8,  # More fusion for complex reasoning
        "dropout_rate": 0.1,
        "use_shared_weights": False,  # More flexible cross-attention
    }

    return VisionLanguageModel(
        vision_config=vision_config,
        text_config=text_config,
        fusion_config=fusion_config,
        name="vqa_vlm"
    )

# ---------------------------------------------------------------------

def create_clip_style_vlm(
        image_size: Tuple[int, int] = (224, 224),
        vocab_size: int = 50000,
        max_text_length: int = 77
) -> VisionLanguageModel:
    """
    Create a CLIP-style VLM for image-text matching.

    Args:
        image_size: Input image size.
        vocab_size: Size of text vocabulary.
        max_text_length: Maximum text sequence length.

    Returns:
        Configured VisionLanguageModel instance.
    """
    vision_config = {
        "image_size": image_size,
        "patch_size": (32, 32),  # Larger patches for efficiency
        "embed_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "dropout_rate": 0.0,  # No dropout for contrastive learning
        "use_cls_token": True,
    }

    text_config = {
        "vocab_size": vocab_size,
        "embed_dim": 768,
        "max_seq_len": max_text_length,
        "num_layers": 12,
        "num_heads": 12,
        "intermediate_size": 3072,
        "dropout_rate": 0.0,  # No dropout for contrastive learning
        "use_causal_mask": False,  # Bidirectional for text understanding
    }

    # No cross-modal fusion for CLIP-style architecture
    return VisionLanguageModel(
        vision_config=vision_config,
        text_config=text_config,
        fusion_config=None,
        name="clip_style_vlm"
    )

# ---------------------------------------------------------------------
