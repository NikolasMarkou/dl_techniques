"""
Task-specific heads and utilities for Vision Language Models.

This module provides specialized heads for different vision-language tasks
such as image captioning, visual question answering, and contrastive learning.
"""

import keras
import numpy as np
from keras import ops, layers
from typing import Dict, List, Optional, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .norms.rms_norm import RMSNorm
from .ffn.swiglu_ffn import SwiGLUFFN
from .attention.multi_head_attention import MultiHeadAttention

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class CaptioningHead(keras.layers.Layer):
    """
    Captioning head for autoregressive text generation.

    Args:
        vocab_size: Size of the vocabulary.
        embed_dim: Embedding dimension.
        num_layers: Number of decoder layers.
        num_heads: Number of attention heads.
        intermediate_size: Size of the intermediate layer in FFN.
        dropout_rate: Dropout rate.
        max_length: Maximum generation length.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 768,
            num_layers: int = 6,
            num_heads: int = 12,
            intermediate_size: int = 3072,
            dropout_rate: float = 0.1,
            max_length: int = 128,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate
        self.max_length = max_length

        # Components will be initialized in build()
        self.cross_attention_layers = []
        self.self_attention_layers = []
        self.ffn_layers = []
        self.norm_layers = []
        self.output_projection = None

    def build(self, input_shape):
        """Build the captioning head layers."""
        for i in range(self.num_layers):
            # Cross-attention to vision features
            cross_attn = MultiHeadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                name=f"cross_attention_{i}"
            )
            self.cross_attention_layers.append(cross_attn)

            # Self-attention for text
            self_attn = MultiHeadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                name=f"self_attention_{i}"
            )
            self.self_attention_layers.append(self_attn)

            # FFN
            ffn = SwiGLUFFN(
                d_model=self.embed_dim,
                name=f"ffn_{i}"
            )
            self.ffn_layers.append(ffn)

            # Layer norms
            norm1 = RMSNorm(name=f"norm1_{i}")
            norm2 = RMSNorm(name=f"norm2_{i}")
            norm3 = RMSNorm(name=f"norm3_{i}")
            self.norm_layers.extend([norm1, norm2, norm3])

        # Output projection to vocabulary
        self.output_projection = layers.Dense(
            self.vocab_size,
            name="output_projection",
            kernel_initializer="glorot_normal"
        )

        super().build(input_shape)

    def call(self, text_features, vision_features, causal_mask=None, training=None):
        """
        Forward pass of the captioning head.

        Args:
            text_features: Text features of shape (batch_size, text_seq_len, embed_dim).
            vision_features: Vision features of shape (batch_size, vision_seq_len, embed_dim).
            causal_mask: Causal mask for text generation.
            training: Whether in training mode.

        Returns:
            Logits for next token prediction of shape (batch_size, text_seq_len, vocab_size).
        """
        x = text_features

        for i in range(self.num_layers):
            # Self-attention with causal mask
            attn_output = self.self_attention_layers[i](
                x, attention_mask=causal_mask, training=training
            )
            x = self.norm_layers[i * 3](x + attn_output)

            # Cross-attention to vision features
            cross_attn_output = self.cross_attention_layers[i](
                x, context=vision_features, training=training
            )
            x = self.norm_layers[i * 3 + 1](x + cross_attn_output)

            # FFN
            ffn_output = self.ffn_layers[i](x, training=training)
            x = self.norm_layers[i * 3 + 2](x + ffn_output)

        # Project to vocabulary
        logits = self.output_projection(x)

        return logits

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "dropout_rate": self.dropout_rate,
            "max_length": self.max_length,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class VQAHead(keras.layers.Layer):
    """
    Visual Question Answering head for answer classification/generation.

    Args:
        num_answers: Number of possible answers (for classification).
        embed_dim: Embedding dimension.
        hidden_dims: List of hidden layer dimensions.
        dropout_rate: Dropout rate.
        pooling_strategy: Strategy for pooling multimodal features.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
            self,
            num_answers: int,
            embed_dim: int = 768,
            hidden_dims: List[int] = [512, 256],
            dropout_rate: float = 0.1,
            pooling_strategy: str = "attention",  # "mean", "max", "attention"
            **kwargs
    ):
        super().__init__(**kwargs)
        self.num_answers = num_answers
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.pooling_strategy = pooling_strategy

        # Components will be initialized in build()
        self.attention_pooling = None
        self.hidden_layers = []
        self.output_layer = None
        self.dropout_layers = []

    def build(self, input_shape):
        """Build the VQA head layers."""
        # Attention pooling if specified
        if self.pooling_strategy == "attention":
            self.attention_pooling = MultiHeadAttention(
                embed_dim=self.embed_dim,
                num_heads=8,
                dropout_rate=self.dropout_rate,
                name="attention_pooling"
            )

        # Hidden layers
        prev_dim = self.embed_dim * 2  # Concatenated vision and text features
        for i, hidden_dim in enumerate(self.hidden_dims):
            hidden_layer = layers.Dense(
                hidden_dim,
                activation="gelu",
                kernel_initializer="glorot_normal",
                name=f"hidden_{i}"
            )
            dropout_layer = layers.Dropout(self.dropout_rate, name=f"dropout_{i}")

            self.hidden_layers.append(hidden_layer)
            self.dropout_layers.append(dropout_layer)
            prev_dim = hidden_dim

        # Output layer
        self.output_layer = layers.Dense(
            self.num_answers,
            kernel_initializer="glorot_normal",
            name="output_layer"
        )

        super().build(input_shape)

    def call(self, vision_features, text_features, training=None):
        """
        Forward pass of the VQA head.

        Args:
            vision_features: Vision features of shape (batch_size, vision_seq_len, embed_dim).
            text_features: Text features of shape (batch_size, text_seq_len, embed_dim).
            training: Whether in training mode.

        Returns:
            Answer logits of shape (batch_size, num_answers).
        """
        # Pool features
        if self.pooling_strategy == "mean":
            vision_pooled = ops.mean(vision_features, axis=1)
            text_pooled = ops.mean(text_features, axis=1)
        elif self.pooling_strategy == "max":
            vision_pooled = ops.max(vision_features, axis=1)
            text_pooled = ops.max(text_features, axis=1)
        elif self.pooling_strategy == "attention":
            # Use cross-attention between vision and text for pooling
            vision_attended = self.attention_pooling(
                vision_features, context=text_features, training=training
            )
            text_attended = self.attention_pooling(
                text_features, context=vision_features, training=training
            )
            vision_pooled = ops.mean(vision_attended, axis=1)
            text_pooled = ops.mean(text_attended, axis=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # Concatenate features
        x = ops.concatenate([vision_pooled, text_pooled], axis=-1)

        # Pass through hidden layers
        for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            x = hidden_layer(x)
            x = dropout_layer(x, training=training)

        # Output layer
        logits = self.output_layer(x)

        return logits

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "num_answers": self.num_answers,
            "embed_dim": self.embed_dim,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "pooling_strategy": self.pooling_strategy,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ContrastiveHead(keras.layers.Layer):
    """
    Contrastive learning head for image-text matching.

    Args:
        embed_dim: Embedding dimension.
        projection_dim: Projection dimension for contrastive learning.
        temperature: Temperature parameter for contrastive loss.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
            self,
            embed_dim: int = 768,
            projection_dim: int = 256,
            temperature: float = 0.07,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim
        self.temperature = temperature

        # Components will be initialized in build()
        self.vision_projection = None
        self.text_projection = None

    def build(self, input_shape):
        """Build the contrastive head layers."""
        self.vision_projection = layers.Dense(
            self.projection_dim,
            kernel_initializer="glorot_normal",
            name="vision_projection"
        )

        self.text_projection = layers.Dense(
            self.projection_dim,
            kernel_initializer="glorot_normal",
            name="text_projection"
        )

        super().build(input_shape)

    def call(self, vision_features, text_features, training=None):
        """
        Forward pass of the contrastive head.

        Args:
            vision_features: Vision features of shape (batch_size, embed_dim).
            text_features: Text features of shape (batch_size, embed_dim).
            training: Whether in training mode.

        Returns:
            Dictionary containing projected features and similarity logits.
        """
        # Project to contrastive space
        vision_projected = self.vision_projection(vision_features)
        text_projected = self.text_projection(text_features)

        # L2 normalize
        vision_projected = ops.l2_normalize(vision_projected, axis=-1)
        text_projected = ops.l2_normalize(text_projected, axis=-1)

        # Compute similarity matrix
        similarity_matrix = ops.matmul(vision_projected, ops.transpose(text_projected))
        logits = similarity_matrix / self.temperature

        return {
            "vision_projected": vision_projected,
            "text_projected": text_projected,
            "similarity_matrix": similarity_matrix,
            "logits": logits,
        }

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "projection_dim": self.projection_dim,
            "temperature": self.temperature,
        })
        return config

# ---------------------------------------------------------------------