"""
Task-specific heads and utilities for Vision Language Models.

This module provides specialized heads for different vision-language tasks
such as image captioning, visual question answering, and contrastive learning.
"""

import keras
import numpy as np
from keras import ops, layers
from typing import Dict, List, Optional, Any

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.ffn.swiglu_ffn import SwiGLUFFN
from dl_techniques.layers.attention.multi_head_attention import MultiHeadAttention


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


class VLMTrainingUtils:
    """Utility functions for training Vision Language Models."""

    @staticmethod
    def create_causal_mask(seq_len: int) -> np.ndarray:
        """
        Create a causal mask for autoregressive generation.

        Args:
            seq_len: Sequence length.

        Returns:
            Causal mask of shape (seq_len, seq_len).
        """
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        return mask.astype(np.bool_)

    @staticmethod
    def create_attention_mask(input_ids: np.ndarray, pad_token_id: int = 0) -> np.ndarray:
        """
        Create attention mask from input token IDs.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            pad_token_id: Padding token ID.

        Returns:
            Attention mask of shape (batch_size, seq_len).
        """
        return (input_ids != pad_token_id).astype(np.float32)

    @staticmethod
    def compute_contrastive_loss(
            vision_features: keras.KerasTensor,
            text_features: keras.KerasTensor,
            temperature: float = 0.07
    ) -> keras.KerasTensor:
        """
        Compute contrastive loss for vision-text pairs.

        Args:
            vision_features: Normalized vision features.
            text_features: Normalized text features.
            temperature: Temperature parameter.

        Returns:
            Contrastive loss value.
        """
        batch_size = ops.shape(vision_features)[0]

        # Compute similarity matrix
        similarity_matrix = ops.matmul(vision_features, ops.transpose(text_features))
        logits = similarity_matrix / temperature

        # Create labels (diagonal matrix for positive pairs)
        labels = ops.cast(ops.eye(batch_size), logits.dtype)

        # Compute cross-entropy loss
        vision_loss = keras.losses.categorical_crossentropy(
            labels, ops.softmax(logits, axis=1), from_logits=False
        )
        text_loss = keras.losses.categorical_crossentropy(
            labels, ops.softmax(ops.transpose(logits), axis=1), from_logits=False
        )

        return (ops.mean(vision_loss) + ops.mean(text_loss)) / 2

    @staticmethod
    def compute_captioning_loss(
            predictions: keras.KerasTensor,
            targets: keras.KerasTensor,
            mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """
        Compute captioning loss with optional masking.

        Args:
            predictions: Predicted logits of shape (batch_size, seq_len, vocab_size).
            targets: Target token IDs of shape (batch_size, seq_len).
            mask: Optional mask of shape (batch_size, seq_len).

        Returns:
            Captioning loss value.
        """
        # Compute cross-entropy loss
        loss = keras.losses.sparse_categorical_crossentropy(
            targets, predictions, from_logits=True
        )

        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            return ops.sum(loss) / ops.sum(mask)
        else:
            return ops.mean(loss)


# Complete VLM with task-specific heads
@keras.saving.register_keras_serializable()
class CompleteVLM(keras.Model):
    """
    Complete Vision Language Model with task-specific heads.

    Args:
        base_model: Base VisionLanguageModel instance.
        task_configs: Dictionary of task configurations.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
            self,
            base_model: keras.Model,
            task_configs: Dict[str, Dict[str, Any]],
            **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.task_configs = task_configs

        # Initialize task-specific heads
        self.heads = {}
        for task_name, config in task_configs.items():
            if task_name == "captioning":
                self.heads[task_name] = CaptioningHead(**config)
            elif task_name == "vqa":
                self.heads[task_name] = VQAHead(**config)
            elif task_name == "contrastive":
                self.heads[task_name] = ContrastiveHead(**config)
            else:
                logger.warning(f"Unknown task: {task_name}")

    def call(self, inputs, task="contrastive", training=None):
        """
        Forward pass with task-specific head.

        Args:
            inputs: Model inputs.
            task: Task name to use.
            training: Whether in training mode.

        Returns:
            Task-specific outputs.
        """
        # Get base model outputs
        base_outputs = self.base_model(inputs, training=training)

        # Apply task-specific head
        if task in self.heads:
            if task == "captioning":
                return self.heads[task](
                    base_outputs["fused_text_features"],
                    base_outputs["fused_vision_features"],
                    training=training
                )
            elif task == "vqa":
                return self.heads[task](
                    base_outputs["fused_vision_features"],
                    base_outputs["fused_text_features"],
                    training=training
                )
            elif task == "contrastive":
                return self.heads[task](
                    base_outputs["vision_global"],
                    base_outputs["text_global"],
                    training=training
                )

        return base_outputs

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "base_model": keras.saving.serialize_keras_object(self.base_model),
            "task_configs": self.task_configs,
        })
        return config