"""Learnable position-weighted pooling layer for sequence data.

This module provides :class:`WeightedPooling`, a learnable pooling layer that
collapses a sequence of token vectors ``(batch, seq_len, embed_dim)`` into a
single summary vector ``(batch, embed_dim)`` using per-position learnable
weights.

A scalar learnable weight is assigned to each position up to ``max_seq_len``.
At inference the weights for the current sequence length are softmax-normalised
and used to compute a weighted sum. Unlike attention pooling these weights are
content-independent: they capture positional importance patterns rather than
input-dependent relevance, providing a middle ground between simple mean
pooling and full attention pooling.
"""

import keras
from keras import ops, layers, initializers, regularizers
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class WeightedPooling(keras.layers.Layer):
    """Learnable position-weighted pooling for sequences.

    A scalar learnable weight is assigned to each position up to
    ``max_seq_len``. At inference the weights for the current sequence
    length are softmax-normalised and used to compute a weighted sum,
    producing a fixed-size vector. Unlike attention pooling these weights
    are content-independent: they capture positional importance patterns
    rather than input-dependent relevance.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input [B, seq_len, embed_dim]   │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  position_weights[:seq_len]      │
        │  / temperature → softmax         │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Weighted sum over sequence      │
        │  → Output [B, embed_dim]         │
        └──────────────────────────────────┘

    Args:
        max_seq_len: Maximum sequence length for weight allocation.
        dropout_rate: Dropout rate applied to normalised weights.
        temperature: Temperature for weight softmax.
        initializer: Initializer for position weights.
        regularizer: Optional regularizer for weights.
    """

    def __init__(
        self,
        max_seq_len: int = 512,
        dropout_rate: float = 0.0,
        temperature: float = 1.0,
        initializer: Union[str, initializers.Initializer] = 'ones',
        regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialise the weighted pooling layer."""
        super().__init__(**kwargs)

        # Store all configuration
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.temperature = temperature
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer) if regularizer else None

        # Create dropout layer in __init__
        if self.dropout_rate > 0:
            self.dropout = layers.Dropout(self.dropout_rate)
        else:
            self.dropout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build learnable position weights.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Create learnable position weights
        self.position_weights = self.add_weight(
            name='position_weights',
            shape=(self.max_seq_len,),
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True
        )

        # Build dropout if exists
        if self.dropout is not None:
            # Dropout expects shape (batch, seq_len)
            dropout_shape = (input_shape[0], input_shape[1])
            self.dropout.build(dropout_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply position-weighted pooling.

        Args:
            inputs: Input sequence ``(batch, seq_len, embed_dim)``.
            mask: Optional boolean mask ``(batch, seq_len)``.
            training: Whether in training mode.

        Returns:
            Pooled output ``(batch, embed_dim)``.
        """
        seq_len = ops.shape(inputs)[1]

        # Get weights for current sequence length
        weights = self.position_weights[:seq_len]
        weights = weights / self.temperature

        # Apply mask if provided
        if mask is not None:
            weights = weights * ops.cast(mask, weights.dtype)

        # Normalize weights
        weights = ops.softmax(weights, axis=-1)

        # Apply dropout
        if self.dropout is not None:
            weights = self.dropout(weights, training=training)

        # Compute weighted sum
        weights_expanded = ops.expand_dims(weights, -1)
        weighted_sum = ops.sum(inputs * weights_expanded, axis=1)

        return weighted_sum

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
            'max_seq_len': self.max_seq_len,
            'dropout_rate': self.dropout_rate,
            'temperature': self.temperature,
            'initializer': initializers.serialize(self.initializer),
            'regularizer': regularizers.serialize(self.regularizer),
        })
        return config

# ---------------------------------------------------------------------
