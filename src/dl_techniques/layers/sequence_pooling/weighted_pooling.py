"""
Learnable position-weighted pooling layer for sequence data.

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
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.dtype_policy import mask_sentinel
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.validation import positive_float, positive_int

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.sequence_pooling.weighted_pooling")
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

    A sequence longer than ``max_seq_len`` has no weights for its tail. With a
    static length (every eager call, and a functional model built for a fixed
    length) ``call`` raises a ``ValueError`` naming ``max_seq_len``. With an
    unknown length no assert is traced: the run reaches a longer input and the
    backend refuses the broadcast (``Incompatible shapes``). Nothing is
    truncated silently.

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
        initializer: Union[str, keras.initializers.Initializer] = 'ones',
        regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialise the weighted pooling layer."""
        super().__init__(**kwargs)

        # Store all configuration (checked here, not first at call time)
        self.max_seq_len = positive_int(max_seq_len, 'max_seq_len', 'WeightedPooling')
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(
                f"WeightedPooling: dropout_rate must be in [0, 1), got {dropout_rate!r}"
            )
        self.dropout_rate = dropout_rate
        self.temperature = positive_float(temperature, 'temperature', 'WeightedPooling')
        self.initializer = keras.initializers.get(initializer)
        self.regularizer = keras.regularizers.get(regularizer) if regularizer else None

        # Create dropout layer in __init__
        if self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(self.dropout_rate)
        else:
            self.dropout = None

    def _check_length(self, seq_len: Optional[int]) -> None:
        """Raise if a static ``seq_len`` exceeds ``max_seq_len``.

        Called from ``build`` (a functional model infers shapes without
        running ``call``) and from ``call`` (an already-built layer meeting a
        longer input). A ``None`` length passes; see the class docstring.
        """
        if seq_len is not None and seq_len > self.max_seq_len:
            raise ValueError(
                f"WeightedPooling: the sequence length {seq_len} is longer "
                f"than max_seq_len={self.max_seq_len}; there is one position "
                f"weight per position up to max_seq_len."
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build learnable position weights.

        Args:
            input_shape: Shape tuple of the input tensor.

        Raises:
            ValueError: If the static sequence length exceeds ``max_seq_len``.
        """
        self._check_length(input_shape[1])
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
            mask: Optional boolean mask ``(batch, seq_len)``, 1/True = keep.
                A masked position contributes EXACTLY zero to the output.
                A fully-masked row is not rescued: every logit becomes the same
                sentinel, so the softmax is uniform and the result is the plain
                mean over the sequence (finite, never NaN). The sibling
                ``AttentionPooling`` behaves the same way.
            training: Whether in training mode.

        Returns:
            Pooled output ``(batch, embed_dim)``.
        """
        self._check_length(inputs.shape[1])
        seq_len = keras.ops.shape(inputs)[1]

        # Get weights for current sequence length
        weights = self.position_weights[:seq_len]
        weights = weights / self.temperature

        # Apply mask if provided
        if mask is not None:
            # DECISION plan-2026-07-31T132403-b3f540cb/D-002: mask the softmax
            # LOGITS with `ops.where` and the finite, fp16-representable `-1e4`
            # sentinel, after broadcasting the shared position weights to
            # (batch, seq_len).
            # Do NOT revert to `weights * cast(mask, dtype)` before the softmax:
            # that sends a masked position in as logit 0, and `softmax(0)`
            # among finite logits is NOT zero, so the masked token keeps real
            # weight in the weighted sum (measured leak 1.129423e+00 where 0.0
            # is required - finding F-24).
            # Do NOT use `-1e9` (the `layers/attention/common.py` value):
            # `float16(-1e9)` is `-inf`, and Do NOT use the additive form
            # `weights + (1 - mask) * bias` either - `0 * -inf` is NaN under
            # `mixed_float16`. The `ops.where` + `-1e4` shape mirrors the
            # in-package precedent in `attention_pooling.py`
            # (`# DECISION plan-2026-07-15T114613-5add9baa/D-001`).
            # The broadcast is load-bearing: `position_weights` is ONE vector
            # shared by the whole batch, while the mask is per-batch-row.
            # Superseded in part by plan-2026-10-14T062616-81938e81/D-203: the
            # sentinel is now mask_sentinel(compute_dtype) (-1e4 in float16 as
            # before, -1e9 in float32), so kept logits below -1e4 no longer
            # lose to a masked position in float32.
            weights = keras.ops.broadcast_to(weights, keras.ops.shape(mask))
            keep = keras.ops.cast(mask, "bool")
            neg = keras.ops.cast(mask_sentinel(self.compute_dtype), weights.dtype)
            weights = keras.ops.where(keep, weights, neg)

        # Normalize weights
        weights = keras.ops.softmax(weights, axis=-1)

        # Apply dropout
        if self.dropout is not None:
            weights = self.dropout(weights, training=training)

        # Compute weighted sum
        weights_expanded = keras.ops.expand_dims(weights, -1)
        weighted_sum = keras.ops.sum(inputs * weights_expanded, axis=1)

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
            'initializer': keras.initializers.serialize(self.initializer),
            'regularizer': keras.regularizers.serialize(self.regularizer),
        })
        return config

# ---------------------------------------------------------------------
