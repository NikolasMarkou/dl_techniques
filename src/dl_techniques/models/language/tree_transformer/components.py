"""
Tree Transformer sub-layer components: ``PositionalEncoding`` for sinusoidal
positions, ``GroupAttention`` for tree induction, ``TreeMHA`` for attention
modulated by group probabilities, and ``TreeTransformerBlock``, the encoder
block that combines the last two with a feed-forward network. What separates
this from a plain transformer block is ``GroupAttention``: it turns the
attention between adjacent tokens into break probabilities, runs a parallel
CKY-style dynamic program over them to get a probability that any two tokens
belong to the same constituent, and ``TreeMHA`` multiplies its attention
weights by that matrix. The break probabilities pass to the next block as a
prior, so structure accumulates with depth. Callers should know that every
``call`` here takes a tuple, that ``GroupAttention`` returns two tensors and
``TreeTransformerBlock`` returns three, that the mask arrives shaped
``(batch, 1, seq_len)``, and that the dynamic program runs in float32
whatever the compute dtype is. ``PositionalEncoding`` precomputes ``max_len``
positions, which caps the sequence length it can serve. The classes are
re-exported from ``model.py``.
"""

import math
import keras
import numpy as np
from keras import ops
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------

from dl_techniques.layers.ffn import create_ffn_layer, FFNType
from dl_techniques.layers.ffn.factory import assemble_ffn_config
from dl_techniques.layers.norms import (
    create_normalization_layer,
    NormalizationType,
)
from dl_techniques.utils.dtype_policy import mask_sentinel
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.tree_transformer.components")
class PositionalEncoding(keras.layers.Layer):
    """
    Add fixed sinusoidal positional encoding to input embeddings.

    The encoding is precomputed for ``max_len`` positions and stored as a
    non-trainable weight, and ``call`` slices it to the sequence length it
    receives. A sequence longer than ``max_len`` has no encoding to slice.

    Architecture:

    .. code-block:: text

            x [B, L, hidden_size]
                      │
                      ▼
           ┌─────────────────────┐
           │  add pe[:, :L, :]   │
           └─────────────────────┘
                      │
                      ▼
                   dropout
                      │
                      ▼
             [B, L, hidden_size]

    Formula:
        :math:`PE(pos, 2i) = \\sin(pos / 10000^{2i/d_{model}})`
        :math:`PE(pos, 2i+1) = \\cos(pos / 10000^{2i/d_{model}})`
        :math:`output = dropout(input + PE_{slice})`

    :param hidden_size: The dimensionality of the embeddings. Must be positive.
    :type hidden_size: int
    :param dropout_rate: Dropout probability after adding encodings. Must be in [0, 1].
    :type dropout_rate: float
    :param max_len: Maximum sequence length for pre-computation.
    :type max_len: int
    :param kwargs: Additional keyword arguments for `keras.layers.Layer`.
    :raises ValueError: If ``hidden_size`` is not positive or ``dropout_rate``
        is outside [0, 1].

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_size)`.

    Output shape:
        3D tensor with the same shape as input: `(batch_size, sequence_length, hidden_size)`.

    :ivar pe: Non-trainable weight matrix of shape `(1, max_len, hidden_size)`.
    :ivar dropout: `keras.layers.Dropout` layer.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout_rate: float,
        max_len: int = 5000,
        **kwargs: Any,
    ) -> None:
        """Initializes the PositionalEncoding layer."""
        super().__init__(**kwargs)
        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {hidden_size}"
            )
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be in [0, 1], got {dropout_rate}"
            )
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        self.dropout = keras.layers.Dropout(dropout_rate)
        # Created in `build`.
        self.pe = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Creates the non-trainable positional encoding matrix.

        :param input_shape: Shape of the input to ``call``.
        """
        pe = np.zeros((self.max_len, self.hidden_size), dtype=np.float32)
        position = np.arange(0, self.max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(
            np.arange(0, self.hidden_size, 2, dtype=np.float32)
            * -(math.log(10000.0) / self.hidden_size)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        # The leading axis broadcasts over the batch.
        pe = pe[np.newaxis, :, :]

        self.pe = self.add_weight(
            name="positional_encoding",
            shape=(1, self.max_len, self.hidden_size),
            initializer=keras.initializers.Constant(pe),
            trainable=False,
        )
        super().build(input_shape)

    def call(
        self, x: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Adds positional encodings to the input tensor.

        :param x: Embeddings of shape (batch, seq_len, hidden_size).
        :param training: Whether to apply dropout. Defaults to None.
        :return: Tensor of the same shape as ``x``.
        """
        seq_len = ops.shape(x)[1]
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x, training=training)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Output shape is identical to the input shape."""
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer's configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "dropout_rate": self.dropout_rate,
                "max_len": self.max_len,
            }
        )
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.tree_transformer.components")
class GroupAttention(keras.layers.Layer):
    """
    Induce a constituency tree and return it as two attention matrices.

    The layer computes ``neibor_attn``, the probability of a break between
    adjacent tokens, and ``g_attn``, the probability that two tokens sit in
    the same syntactic group. Neighbor attention is a dot-product attention
    restricted to the two diagonals next to the main one, symmetrized and then
    blended with the ``prior`` from the previous block. Group attention comes
    from a dynamic program over those neighbor values, run as matrix products
    in log space so all spans are computed at once.

    Architecture:

    .. code-block:: text

        context [B, L, H]  mask [B, 1, L]         prior
                │                 │                 │
                └────────┬────────┘                 │
                         ▼                          │
                neighbor attention                  │
                         │                          │
                         │                          │
                         └────────────┬─────────────┘
                                      ▼
                           prior + (1 - prior) * a
                                      │
                            ┌─────────┴─────────┐
                            ▼                   │
                     tree induction             │
                            │                   │
                            ▼                   ▼
                    g_attn [B, L, L]  neibor_attn [B, L, L]

    Both outputs are zeroed at padded pairs before they are returned.

    Neighbor attention:

    .. code-block:: text

                   context                    mask
                      │                         │
                      ▼                         ▼
             ┌─────────────────┐         adjacency band
             │      norm       │           and padding
             └─────────────────┘                │
                      │                         │
                      ▼                         │
             ┌─────────────────┐                │
             │   query, key    │                │
             └─────────────────┘                │
                      │                         │
                      ▼                         │
          q kᵀ / sqrt(hidden_size)              │
                      │                         │
                      │                         │
                      └────────────┬────────────┘
                                   ▼
                       where(keep, scores, -1e9)
                                   │
                                   ▼
                                softmax
                                   │
                                   ▼
                          sqrt(a * aᵀ + 1e-9)

    The sentinel is -1e4 instead under a float16 compute dtype.

    Tree induction:

    .. code-block:: text

                neibor_attn [B, L, L]
                          │
                          ▼
        ┌───────────────────────────────────┐
        │  cast to float32                  │
        │  t = log(a + 1e-9)                │
        │  t = where(super-diagonal, t, 0)  │
        │  g = exp(tri @ (t @ tri))         │
        └───────────────────────────────────┘
                          │ cast to compute dtype
                          ▼
             keep strict upper triangle
                          │
                          ▼
         g + gᵀ + diagonal from neibor_attn
                          │
                          ▼
                  divide by row sum
                          │
                          ▼
                  g_attn [B, L, L]

    The float32 island is fixed: 1e-9 underflows in float16, and log(0) then
    sends NaN through the exp.

    :param hidden_size: The dimensionality of the model. Must be positive.
    :type hidden_size: int
    :param normalization_type: The type of normalization layer to use (e.g., "layer_norm").
    :type normalization_type: str
    :param layer_norm_eps: Epsilon for this layer's own normalization. Defaults
        to 1e-12, matching :class:`TreeTransformerBlock`'s default.
    :type layer_norm_eps: float
    :param kwargs: Additional keyword arguments for `keras.layers.Layer`.
    :raises ValueError: If ``hidden_size`` is not positive.

    Input shape:
        A tuple of three tensors:
        - `context`: `(batch_size, sequence_length, hidden_size)`
        - `mask`: `(batch_size, 1, sequence_length)`
        - `prior`: Scalar tensor or `(batch_size, sequence_length, sequence_length)`

    Output shape:
        A tuple of two tensors, both with shape: `(batch_size, sequence_length, sequence_length)`
        - `g_attn`: Group attention probabilities.
        - `neibor_attn`: Neighbor attention probabilities (break probabilities).
    """

    def __init__(
        self,
        hidden_size: int,
        normalization_type: str = "layer_norm",
        layer_norm_eps: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        """Initializes the GroupAttention layer."""
        super().__init__(**kwargs)
        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {hidden_size}"
            )
        self.hidden_size = hidden_size
        self.normalization_type = normalization_type
        self.layer_norm_eps = layer_norm_eps

        # DECISION plan-2026-08-19T070627-a616f581/D-007: layer_norm_eps is a real
        # parameter, not the factory's 1e-6 default. See decisions.md.
        self.norm = create_normalization_layer(
            normalization_type, epsilon=layer_norm_eps
        )
        self.linear_key = keras.layers.Dense(
            hidden_size, name="key_projection"
        )
        self.linear_query = keras.layers.Dense(
            hidden_size, name="query_projection"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Builds the sub-layers, which is critical for serialization.

        :param input_shape: Tuple of the ``(context, mask, prior)`` shapes.
        """
        context_shape, _, _ = input_shape
        self.norm.build(context_shape)
        self.linear_key.build(context_shape)
        self.linear_query.build(context_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: Tuple[
            keras.KerasTensor, keras.KerasTensor, keras.KerasTensor
        ],
        training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Computes group attention probabilities.

        :param inputs: Tuple of ``(context, mask, prior)``.
        :param training: Whether to run in training mode. Defaults to None.
        :return: Tuple of ``(g_attn, neibor_attn)``, each (batch, seq_len, seq_len).
        """
        context, mask, prior = inputs
        current_seq_len = ops.shape(context)[1]

        # Backend-agnostic ops, so the band survives graph tracing.
        i = ops.arange(current_seq_len)
        adj_mask_upper = ops.equal(ops.expand_dims(i, 1), ops.expand_dims(i, 0) - 1)
        adj_mask_lower = ops.equal(ops.expand_dims(i, 1), ops.expand_dims(i, 0) + 1)
        adj_mask = ops.logical_or(adj_mask_upper, adj_mask_lower)

        # Padding positions drop out of the band, so nothing leaks through them.
        padding_mask = ops.cast(ops.squeeze(mask, axis=1), "bool")
        padding_mask_2d = ops.logical_and(
            ops.expand_dims(padding_mask, axis=2),
            ops.expand_dims(padding_mask, axis=1),
        )
        final_adj_mask = ops.logical_and(adj_mask, padding_mask_2d)

        context_norm = self.norm(context, training=training)
        key = self.linear_key(context_norm)
        query = self.linear_query(context_norm)
        scores = ops.matmul(query, ops.transpose(key, axes=(0, 2, 1)))
        scores = scores / math.sqrt(self.hidden_size)
        # DECISION plan_2026-05-11_3c3ed037/D-001: -1e9 overflows fp16 (min about
        # -6.5e4) and softmax then returns NaN. See decisions.md.
        neg_inf = -1e4 if self.compute_dtype == "float16" else -1e9
        scores = ops.where(final_adj_mask, scores, neg_inf)
        neibor_attn = ops.softmax(scores, axis=-1)

        # The geometric mean of a pair and its transpose symmetrizes the matrix.
        neibor_attn = ops.sqrt(
            neibor_attn * ops.transpose(neibor_attn, axes=(0, 2, 1)) + 1e-9
        )

        # Blend in the prior, so structure accumulates across blocks.
        neibor_attn = prior + (1.0 - prior) * neibor_attn

        # Parallel CKY: the dynamic program runs as matrix products.
        row_indices = ops.expand_dims(ops.arange(current_seq_len), 1)
        col_indices = ops.expand_dims(ops.arange(current_seq_len), 0)
        triu_mask = ops.greater_equal(col_indices, row_indices)
        tri_matrix = ops.cast(triu_mask, self.compute_dtype)
        b = ops.cast(ops.eye(current_seq_len, dtype="int32"), "bool")

        # Float32 for the whole block: 1e-9 underflows in fp16 (min positive
        # about 6e-5), and log(0) then sends NaN through the exp.
        neibor_attn_f32 = ops.cast(neibor_attn, "float32")
        tri_matrix_f32 = ops.cast(tri_matrix, "float32")
        t = ops.log(neibor_attn_f32 + 1e-9)
        # Only the super-diagonal carries a break probability.
        t = ops.where(adj_mask_upper, t, 0.0)
        t = ops.matmul(t, tri_matrix_f32)
        g_attn = ops.exp(ops.matmul(tri_matrix_f32, t))
        g_attn = ops.cast(g_attn, self.compute_dtype)

        g_attn = ops.where(
            ops.logical_xor(triu_mask, b), g_attn, 0.0
        )
        neibor_attn_diag_filled = ops.where(b, 1.0, neibor_attn)
        g_attn = (
            g_attn
            + ops.transpose(g_attn, axes=(0, 2, 1))
            + neibor_attn_diag_filled
        )

        # The 1e-9 keeps an all-padding row from dividing by zero.
        g_attn_sum = ops.sum(g_attn, axis=-1, keepdims=True)
        g_attn = ops.divide(g_attn, g_attn_sum + 1e-9)

        float_padding_mask_2d = ops.cast(padding_mask_2d, self.compute_dtype)
        g_attn *= float_padding_mask_2d
        neibor_attn *= float_padding_mask_2d

        return g_attn, neibor_attn

    def compute_output_shape(
        self, input_shape: Any
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Return ``(g_attn, neibor_attn)`` shapes ``(b, s, s)`` each."""
        context_shape = input_shape[0]
        b, s = context_shape[0], context_shape[1]
        return ((b, s, s), (b, s, s))

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer's configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "normalization_type": self.normalization_type,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.tree_transformer.components")
class TreeMHA(keras.layers.Layer):
    """
    Run multi-head attention whose weights are scaled by group probabilities.

    The layer is a standard scaled dot-product multi-head attention up to the
    softmax. After it, the attention weights are multiplied element-wise by the
    ``group_prob`` matrix from :class:`GroupAttention`, broadcast across heads,
    so a pair of tokens in different constituents is attenuated. Both
    ``group_prob`` and ``mask`` may be ``None``, in which case the
    corresponding stage is skipped.

    Architecture:

    .. code-block:: text

           query, key, value [B, L, H]     mask [B, 1, L]
                        │                         │
                        ▼                         │
              ┌───────────────────┐               │
              │    wq, wk, wv     │               │
              └───────────────────┘               │
                        │ split heads             │
                        ▼                         │
                 q kᵀ / sqrt(d)                   │
                        │                         │
                        │                         │
                        └────────────┬────────────┘
                                     ▼
                       where(keep, logits, sentinel)
                                     │
                                     ▼
                                  softmax
                                     │    group_prob (optional)
                                     │              │
                                     ▼              ▼
                                     └──────┬───────┘
                                            ▼
                                  weights * group_prob
                                            │
                                            ▼
                                         dropout
                                            │
                                            ▼
                                  matmul v, merge heads
                                            │
                                            ▼
                                    ┌───────────────┐
                                    │     dense     │
                                    └───────────────┘
                                            │
                                            ▼
                                    output [B, L, H]

    The mask selects the sentinel rather than being added to the logits.

    :param num_heads: The number of attention heads.
    :type num_heads: int
    :param hidden_size: The dimensionality of the model. Must be divisible by `num_heads`.
    :type hidden_size: int
    :param attention_dropout_rate: Dropout probability for attention scores.
    :type attention_dropout_rate: float
    :param kwargs: Additional keyword arguments for `keras.layers.Layer`.
    :raises ValueError: If ``hidden_size`` or ``num_heads`` is not positive, if
        ``hidden_size`` is not divisible by ``num_heads``, or if
        ``attention_dropout_rate`` is outside [0, 1].

    Input shape:
        A tuple of tensors:
        - `query`, `key`, `value`: `(batch_size, sequence_length, hidden_size)`
        - `group_prob`: `(batch_size, sequence_length, sequence_length)`
        - `mask`: `(batch_size, 1, sequence_length)`

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_size)`.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        attention_dropout_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initializes the TreeMHA layer."""
        super().__init__(**kwargs)
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads != 0:
            raise ValueError("Invalid hidden_size or num_heads configuration.")
        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError("attention_dropout_rate must be in [0, 1].")

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.attention_dropout_rate = attention_dropout_rate
        self.depth = hidden_size // num_heads

        self.wq = keras.layers.Dense(hidden_size, name="query")
        self.wk = keras.layers.Dense(hidden_size, name="key")
        self.wv = keras.layers.Dense(hidden_size, name="value")
        self.dense = keras.layers.Dense(hidden_size, name="output_projection")
        self.dropout = keras.layers.Dropout(attention_dropout_rate)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Builds all sub-layers for robust serialization.

        :param input_shape: Tuple of the five input shapes; only the query
            shape is needed, since all four projections are square.
        """
        query_shape, _, _, _, _ = input_shape
        self.wq.build(query_shape)
        self.wk.build(query_shape)
        self.wv.build(query_shape)
        self.dense.build(query_shape)
        super().build(input_shape)

    def _split_heads(
        self, x: keras.KerasTensor, batch_size: int
    ) -> keras.KerasTensor:
        """Splits the last dimension into (num_heads, depth)."""
        x = ops.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return ops.transpose(x, axes=[0, 2, 1, 3])

    def call(
        self,
        inputs: Tuple[
            keras.KerasTensor,
            keras.KerasTensor,
            keras.KerasTensor,
            Optional[keras.KerasTensor],
            Optional[keras.KerasTensor],
        ],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass for tree-modulated multi-head attention.

        :param inputs: Tuple of ``(query, key, value, group_prob, mask)``, with
            the last two optional.
        :param training: Whether to apply attention dropout. Defaults to None.
        :return: Tensor of shape (batch, seq_len, hidden_size).
        """
        query, key, value, group_prob, mask = inputs
        batch_size = ops.shape(query)[0]

        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)

        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        matmul_qk = ops.matmul(q, ops.transpose(k, axes=(0, 1, 3, 2)))
        dk = ops.cast(ops.shape(k)[-1], self.compute_dtype)
        scaled_attention_logits = matmul_qk / ops.sqrt(dk)

        if mask is not None:
            # DECISION plan-2026-08-31T134711-6271592d/D-007: select the sentinel,
            # never add it; the additive form gives 0 * -inf = NaN. See decisions.md.
            keep = ops.expand_dims(ops.cast(mask, "bool"), axis=1)
            sentinel = ops.cast(
                mask_sentinel(self.compute_dtype), scaled_attention_logits.dtype
            )
            scaled_attention_logits = ops.where(
                keep, scaled_attention_logits, sentinel
            )

        attention_weights = ops.softmax(scaled_attention_logits, axis=-1)

        # group_prob is (B, L, L), so it broadcasts across heads.
        if group_prob is not None:
            attention_weights *= ops.expand_dims(group_prob, axis=1)

        attention_weights = self.dropout(attention_weights, training=training)
        output = ops.matmul(attention_weights, v)
        output = ops.transpose(output, axes=[0, 2, 1, 3])
        output = ops.reshape(output, (batch_size, -1, self.hidden_size))
        return self.dense(output)

    def compute_output_shape(
        self, input_shape: Any
    ) -> Tuple[Optional[int], ...]:
        """Output shape is ``(batch, seq_len, hidden_size)``."""
        query_shape = input_shape[0]
        return (query_shape[0], query_shape[1], self.hidden_size)

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer's configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_size": self.hidden_size,
                "attention_dropout_rate": self.attention_dropout_rate,
            }
        )
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.tree_transformer.components")
class TreeTransformerBlock(keras.layers.Layer):
    """
    Run one Tree Transformer encoder block.

    The block computes the constituency structure with :class:`GroupAttention`,
    feeds it to :class:`TreeMHA`, and finishes with a feed-forward network,
    both sub-blocks wired pre-norm. It returns three tensors: the updated
    hidden states, the group probabilities, and the break probabilities, which
    the next block takes as its ``prior``.

    Architecture (pre-norm):

    .. code-block:: text

                       x [B, L, H]
                            │
              ┌─────────────┤
              │             ▼
              │      ┌─────────────┐
              │      │    norm1    │
              │      └─────────────┘
              │             │
              │             ▼
              │      ┌─────────────┐
              │      │  self_attn  │
              │      └─────────────┘
              │             │
              │             ▼
              │         dropout1
              │             │
              └──────┬──────┘
                     ▼
                    add
                     │
              ┌──────┴──────┐
              │             ▼
              │      ┌─────────────┐
              │      │    norm2    │
              │      └─────────────┘
              │             │
              │             ▼
              │      ┌─────────────┐
              │      │     ffn     │
              │      └─────────────┘
              │             │
              │             ▼
              │         dropout2
              │             │
              └──────┬──────┘
                     ▼
                   x_out

    The residual stream is never normalized; norm1 and norm2 see copies.

    Structure path:

    .. code-block:: text

                    x, mask, prior
                           │
                           ▼
        ┌─────────────────────────────────────┐
        │             group_attn              │
        └─────────────────────────────────────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
         group_prob_out          break_prob
                │                     │
                ▼                     ▼
            self_attn             returned

    group_prob_out modulates self_attn and is returned as well.

    :param hidden_size: Dimensionality of the model.
    :type hidden_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param intermediate_size: Dimensionality of the FFN layer.
    :type intermediate_size: int
    :param hidden_dropout_rate: Dropout for hidden layers.
    :type hidden_dropout_rate: float
    :param attention_dropout_rate: Dropout for attention scores.
    :type attention_dropout_rate: float
    :param normalization_type: Type of normalization layer.
    :type normalization_type: NormalizationType
    :param ffn_type: Type of feed-forward network.
    :type ffn_type: FFNType
    :param hidden_act: Activation function for the FFN.
    :type hidden_act: str
    :param layer_norm_eps: Epsilon for normalization layers. Applies to
        ``norm1``, ``norm2`` and the :class:`GroupAttention` sub-layer's own
        norm.
    :type layer_norm_eps: float
    :param kwargs: Additional keyword arguments for `keras.layers.Layer`.
    :raises ValueError: If ``hidden_size``, ``num_heads`` or
        ``intermediate_size`` is not positive, or if ``hidden_size`` is not
        divisible by ``num_heads``.

    Input shape:
        Tuple of `(x, mask, group_prob_prior)` as defined in `GroupAttention`.

    Output shape:
        Tuple of `(x_out, group_prob_out, break_prob)`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        hidden_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        normalization_type: NormalizationType = "layer_norm",
        ffn_type: FFNType = "mlp",
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        """Initializes the TreeTransformerBlock."""
        super().__init__(**kwargs)

        if hidden_size <= 0 or num_heads <= 0:
            raise ValueError(
                f"hidden_size and num_heads must be positive, got "
                f"hidden_size={hidden_size}, num_heads={num_heads}"
            )
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        if intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be positive, got {intermediate_size}"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.normalization_type = normalization_type
        self.ffn_type = ffn_type
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps

        # DECISION plan-2026-08-19T070627-a616f581/D-007: layer_norm_eps reaches
        # GroupAttention too, not just norm1 and norm2. See decisions.md.
        self.group_attn = GroupAttention(
            hidden_size=hidden_size,
            normalization_type=normalization_type,
            layer_norm_eps=layer_norm_eps,
        )
        self.self_attn = TreeMHA(
            num_heads=num_heads,
            hidden_size=hidden_size,
            attention_dropout_rate=attention_dropout_rate,
        )
        # DECISION plan-2026-07-30T140922-8af1028f/D-022: pre-filter the config
        # against what ffn_type accepts; create_ffn_layer raises on a key it
        # would drop. See decisions.md, nam/cell.py.
        self.ffn = create_ffn_layer(
            ffn_type=ffn_type,
            name="ffn",
            **assemble_ffn_config(
                ffn_type,
                {
                    "hidden_dim": intermediate_size,
                    "output_dim": hidden_size,
                    "activation": hidden_act,
                    "dropout_rate": hidden_dropout_rate,
                },
            ),
        )
        self.norm1 = create_normalization_layer(
            normalization_type=normalization_type,
            epsilon=layer_norm_eps,
            name="norm1",
        )
        self.norm2 = create_normalization_layer(
            normalization_type=normalization_type,
            epsilon=layer_norm_eps,
            name="norm2",
        )
        self.dropout1 = keras.layers.Dropout(hidden_dropout_rate)
        self.dropout2 = keras.layers.Dropout(hidden_dropout_rate)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Builds all sub-layers explicitly for robust serialization.

        :param input_shape: Tuple of the ``(x, mask, prior)`` shapes.
        """
        x_shape, mask_shape, prior_shape = input_shape
        group_attn_input_shape = (x_shape, mask_shape, prior_shape)

        self.group_attn.build(group_attn_input_shape)
        self.norm1.build(x_shape)

        # TreeMHA takes a five-tuple; the group_prob slot has no shape to give.
        mha_input_shape = (x_shape, x_shape, x_shape, None, mask_shape)
        self.self_attn.build(mha_input_shape)

        self.norm2.build(x_shape)
        self.ffn.build(x_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: Tuple[
            keras.KerasTensor, keras.KerasTensor, keras.KerasTensor
        ],
        training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Forward pass for the Tree Transformer block.

        :param inputs: Tuple of ``(x, mask, group_prob_prior)``.
        :param training: Whether to run in training mode. Defaults to None.
        :return: Tuple of ``(x_out, group_prob_out, break_prob)``.
        """
        x, mask, group_prob = inputs

        group_prob_out, break_prob = self.group_attn(
            (x, mask, group_prob), training=training
        )

        x_norm1 = self.norm1(x)
        attn_output = self.self_attn(
            (x_norm1, x_norm1, x_norm1, group_prob_out, mask),
            training=training,
        )
        x = x + self.dropout1(attn_output, training=training)

        x_norm2 = self.norm2(x)
        ffn_output = self.ffn(x_norm2, training=training)
        x = x + self.dropout2(ffn_output, training=training)

        return x, group_prob_out, break_prob

    def compute_output_shape(
        self, input_shape: Any
    ) -> Tuple[Tuple[Optional[int], ...], ...]:
        """Return ``(x_out, group_prob_out, break_prob)`` shapes."""
        x_shape = input_shape[0]
        b, s = x_shape[0], x_shape[1]
        return ((b, s, self.hidden_size), (b, s, s), (b, s, s))

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer's configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "hidden_dropout_rate": self.hidden_dropout_rate,
                "attention_dropout_rate": self.attention_dropout_rate,
                "normalization_type": self.normalization_type,
                "ffn_type": self.ffn_type,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config

# ---------------------------------------------------------------------