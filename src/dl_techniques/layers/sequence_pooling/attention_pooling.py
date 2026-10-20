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
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers.clone import clone_initializer
from dl_techniques.utils.dtype_policy import mask_sentinel
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.validation import positive_float, positive_int

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.sequence_pooling.attention_pooling")
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
        use_bias: Whether the ``tanh`` projection has a bias.
        temperature: Temperature for attention softmax.
        kernel_initializer: Initializer for the projection kernel and the
            context vector. Each of the two weights gets its own clone, so a
            seedless initializer draws them independently.
        bias_initializer: Initializer for the projection bias (read only when
            ``use_bias`` is True).
        kernel_regularizer: Optional regularizer for the projection kernel and
            the context vector.
        bias_regularizer: Optional regularizer for the projection bias.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 1,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        temperature: float = 1.0,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialise the attention pooling layer."""
        super().__init__(**kwargs)

        # Store all configuration (checked here, not first at call time)
        self.hidden_dim = positive_int(hidden_dim, 'hidden_dim', 'AttentionPooling')
        self.num_heads = positive_int(num_heads, 'num_heads', 'AttentionPooling')
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(
                f"AttentionPooling: dropout_rate must be in [0, 1), got {dropout_rate!r}"
            )
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.temperature = positive_float(temperature, 'temperature', 'AttentionPooling')
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer) if kernel_regularizer else None
        self.bias_regularizer = keras.regularizers.get(bias_regularizer) if bias_regularizer else None

        # DECISION plan-2026-10-14T062616-81938e81/D-201: one clone of the
        # kernel initializer per weight (the Dense kernel here, context_vector
        # in build). A seedless instance shared by both replayed one sample:
        # context_vector was the kernel's leading values (r = 1.0). Do NOT
        # hand self.kernel_initializer to either site directly.
        self.attention_dense = keras.layers.Dense(
            self.hidden_dim * self.num_heads,
            activation='tanh',
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='attention_transform'
        )

        if self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(self.dropout_rate)
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
            initializer=clone_initializer(self.kernel_initializer),
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
            mask: Optional boolean mask ``(batch, seq_len)``, 1/True = keep.
                A masked position contributes exactly zero to the output while
                every kept logit is above the dtype's mask sentinel
                (``utils.dtype_policy.mask_sentinel``: -1e9 in float32, -1e4
                in float16). A fully-masked row is not rescued: every logit
                becomes the same sentinel, so the softmax is uniform and the
                result is the plain mean over the sequence (finite, never NaN).
                The sibling ``WeightedPooling`` behaves the same way.
            training: Whether in training mode.

        Returns:
            Pooled output ``(batch, embed_dim)``.
        """
        batch_size = keras.ops.shape(inputs)[0]
        seq_len = keras.ops.shape(inputs)[1]

        # Compute attention scores
        attention_hidden = self.attention_dense(inputs)

        # Reshape for multi-head attention
        attention_hidden = keras.ops.reshape(
            attention_hidden,
            (batch_size, seq_len, self.num_heads, self.hidden_dim)
        )

        # Compute attention scores with context vector
        scores = keras.ops.einsum('bsnh,nh->bsn', attention_hidden, self.context_vector)
        scores = scores / self.temperature

        # Apply mask if provided
        if mask is not None:
            # DECISION plan-2026-10-14T062616-81938e81/D-203: select the
            # dtype's mask_sentinel with ops.where. The old float32 literal
            # -1e4 let a masked position take all the weight once the kept
            # logits fell below -1e4 (a small temperature). Do NOT revert to a
            # literal, and do NOT use an additive -1e9 mask (-inf under fp16,
            # and 0 * -inf = NaN poisons the kept tokens).
            mask_bool = keras.ops.expand_dims(keras.ops.cast(mask, "bool"), -1)
            neg = keras.ops.cast(mask_sentinel(self.compute_dtype), scores.dtype)
            scores = keras.ops.where(mask_bool, scores, neg)

        # Compute attention weights
        attention_weights = keras.ops.softmax(scores, axis=1)

        # Apply dropout to attention weights
        if self.dropout is not None:
            attention_weights = self.dropout(attention_weights, training=training)

        # Apply attention weights to input
        weighted_sum = keras.ops.einsum('bsn,bsd->bnd', attention_weights, inputs)

        # Average or concatenate heads
        if self.num_heads == 1:
            output = weighted_sum[:, 0, :]
        else:
            # Average across heads
            output = keras.ops.mean(weighted_sum, axis=1)

        return output

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
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
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
