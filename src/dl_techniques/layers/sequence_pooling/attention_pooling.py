"""Attention-based pooling layer for sequence data.

This module provides :class:`AttentionPooling`, a learnable, content-aware
pooling layer that collapses a sequence of token vectors
``(batch, seq_len, embed_dim)`` into a single summary vector
``(batch, embed_dim)``.

Each token is transformed through a ``tanh`` dense projection and scored
against a learnable context vector, producing per-token importance weights
via softmax. The output is the weighted sum of the original input tokens,
optionally using multiple attention heads whose outputs are averaged. This
mechanism follows the structured self-attentive pooling of Lin et al. (2017),
allowing the model to dynamically focus on the most relevant parts of the
sequence for a given task.
"""

import keras
from keras import ops, layers, initializers, regularizers
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class AttentionPooling(keras.layers.Layer):
    """Attention-based pooling that learns to weight sequence elements.

    Each token is transformed through a ``tanh`` dense layer and scored
    against a learnable context vector, producing per-token importance
    weights via softmax. The output is the weighted sum of the original
    input tokens, optionally using multiple attention heads whose outputs
    are averaged.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input [B, seq_len, embed_dim]   │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Dense(hidden*heads, tanh)       │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Reshape → [B, S, heads, hidden] │
        │  Score = einsum(context_vector)  │
        │  Softmax → attention weights     │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Weighted sum over sequence      │
        │  → Output [B, embed_dim]         │
        └──────────────────────────────────┘

    Args:
        hidden_dim: Hidden dimension for attention computation.
        num_heads: Number of attention heads.
        dropout_rate: Dropout rate for attention weights.
        use_bias: Whether to use bias in attention layers.
        temperature: Temperature for attention softmax.
        kernel_initializer: Initializer for kernel weights.
        kernel_regularizer: Optional regularizer for kernel weights.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 1,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        temperature: float = 1.0,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialise the attention pooling layer."""
        super().__init__(**kwargs)

        # Store all configuration
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.temperature = temperature
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer) if kernel_regularizer else None

        # Create sub-layers in __init__ (Golden Rule)
        self.attention_dense = layers.Dense(
            self.hidden_dim * self.num_heads,
            activation='tanh',
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='attention_transform'
        )

        if self.dropout_rate > 0:
            self.dropout = layers.Dropout(self.dropout_rate)
        else:
            self.dropout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build attention layers based on input shape.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Build sub-layers explicitly (Critical for serialization)
        self.attention_dense.build(input_shape)

        # Create context vector weight
        self.context_vector = self.add_weight(
            name='context_vector',
            shape=(self.num_heads, self.hidden_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        # Build dropout if exists
        if self.dropout is not None:
            # Dropout expects shape (batch, seq_len, num_heads)
            dropout_shape = (input_shape[0], input_shape[1], self.num_heads)
            self.dropout.build(dropout_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply attention-based pooling.

        Args:
            inputs: Input sequence ``(batch, seq_len, embed_dim)``.
            mask: Optional boolean mask ``(batch, seq_len)``.
            training: Whether in training mode.

        Returns:
            Pooled output ``(batch, embed_dim)``.
        """
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # Compute attention scores
        attention_hidden = self.attention_dense(inputs)

        # Reshape for multi-head attention
        attention_hidden = ops.reshape(
            attention_hidden,
            (batch_size, seq_len, self.num_heads, self.hidden_dim)
        )

        # Compute attention scores with context vector
        scores = ops.einsum('bsnh,nh->bsn', attention_hidden, self.context_vector)
        scores = scores / self.temperature

        # Apply mask if provided
        if mask is not None:
            # DECISION plan-2026-07-15T114613-5add9baa/D-001: dtype-safe mask sentinel (ops.where, finite -1e4) — a -1e9 literal casts to -inf under fp16 and 0*-inf=NaN poisons unmasked tokens; do NOT revert to an additive -1e9/-inf mask.
            mask_bool = ops.expand_dims(ops.cast(mask, "bool"), -1)
            neg = ops.cast(-1e4, scores.dtype)
            scores = ops.where(mask_bool, scores, neg)

        # Compute attention weights
        attention_weights = ops.softmax(scores, axis=1)

        # Apply dropout to attention weights
        if self.dropout is not None:
            attention_weights = self.dropout(attention_weights, training=training)

        # Apply attention weights to input
        weighted_sum = ops.einsum('bsn,bsd->bnd', attention_weights, inputs)

        # Average or concatenate heads
        if self.num_heads == 1:
            output = weighted_sum[:, 0, :]
        else:
            # Average across heads
            output = ops.mean(weighted_sum, axis=1)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple.
        """
        return (input_shape[0], input_shape[-1])

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        Returns:
            Dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'temperature': self.temperature,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------
