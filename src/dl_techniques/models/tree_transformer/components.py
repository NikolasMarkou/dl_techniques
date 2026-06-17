"""
Tree Transformer sub-layer components.

Contains the four sub-layer classes used by the top-level
`TreeTransformer` model:

- `PositionalEncoding` — sinusoidal positional encoding.
- `GroupAttention` — hierarchical group attention (tree induction).
- `TreeMHA` — multi-head attention modulated by group probabilities.
- `TreeTransformerBlock` — single encoder block (GroupAttention + TreeMHA + FFN).

These classes were extracted from `model.py` to keep the top-level model
file focused on the model class and its factory functions. The classes
are re-exported from `model.py` so that callers importing from
`dl_techniques.models.tree_transformer.model` continue to resolve.
"""

import math
import keras
import numpy as np
from keras import ops
from typing import Optional, Tuple, Dict, Any

from dl_techniques.utils.logger import logger  # noqa: F401  (kept for parity with model.py imports)
from dl_techniques.layers.ffn import create_ffn_layer, FFNType
from dl_techniques.layers.norms import (
    create_normalization_layer,
    NormalizationType,
)

@keras.saving.register_keras_serializable()
class PositionalEncoding(keras.layers.Layer):
    """
    Injects sinusoidal positional encoding into input embeddings.

    This layer adds positional information to token embeddings, allowing the
    model to understand the order of tokens in a sequence. It uses a fixed,
    non-trainable sinusoidal function.

    **Intent**: To provide a standard, non-learnable method for incorporating
    sequence order into token representations, essential for self-attention
    mechanisms which are otherwise permutation-invariant.

    **Architecture**:
    .. code-block:: text

        Input(shape=[batch, seq_len, hidden_size])
               │
               ▼
        Add Positional Encoding Matrix (precomputed)
               │
               ▼
        Dropout(rate=dropout_rate)
               │
               ▼
        Output(shape=[batch, seq_len, hidden_size])

    **Mathematical Operation**:
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

    **Input shape**:
        3D tensor with shape: `(batch_size, sequence_length, hidden_size)`.

    **Output shape**:
        3D tensor with the same shape as input: `(batch_size, sequence_length, hidden_size)`.

    **Attributes**:
        pe: Non-trainable weight matrix of shape `(1, max_len, hidden_size)`.
        dropout: `keras.layers.Dropout` layer.
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

        # Create sub-layers in __init__
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.pe = None # Weight created in build()

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Creates the non-trainable positional encoding matrix."""
        pe = np.zeros((self.max_len, self.hidden_size), dtype=np.float32)
        position = np.arange(0, self.max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(
            np.arange(0, self.hidden_size, 2, dtype=np.float32)
            * -(math.log(10000.0) / self.hidden_size)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]  # Add batch dimension

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
        """Adds positional encodings to the input tensor."""
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


@keras.saving.register_keras_serializable()
class GroupAttention(keras.layers.Layer):
    """
    Hierarchical group attention for Tree Transformer.

    This layer implements the core mechanism of the Tree Transformer. It computes
    two attention distributions: `neibor_attn` (break probability between adjacent
    tokens) and `g_attn` (probability of tokens belonging to the same syntactic group).

    **Intent**: To learn soft, hierarchical constituency trees directly from text
    without explicit syntactic supervision. The computed `g_attn` serves as a
    structural prior for the subsequent multi-head attention layer.

    **Architecture**:
    .. code-block:: text

        Input(context, mask, prior)
               │
               ▼
        Normalization(context) -> Linear Projections (Q, K)
               │
               ▼
        1. Compute Neighbor Attention (`neibor_attn`):
           - Masked dot-product attention between adjacent tokens only.
           - Symmetrize and combine with `prior` from previous layer.
               │
               ▼
        2. Tree Induction (`g_attn`):
           - Use dynamic programming (in log-space) on `neibor_attn`
             to compute group probabilities for all token spans.
           - Symmetrize and normalize.
               │
               ▼
        Output(`g_attn`, `neibor_attn`)

    :param hidden_size: The dimensionality of the model. Must be positive.
    :type hidden_size: int
    :param normalization_type: The type of normalization layer to use (e.g., "layer_norm").
    :type normalization_type: str
    :param kwargs: Additional keyword arguments for `keras.layers.Layer`.

    **Input shape**:
        A tuple of three tensors:
        - `context`: `(batch_size, sequence_length, hidden_size)`
        - `mask`: `(batch_size, 1, sequence_length)`
        - `prior`: Scalar tensor or `(batch_size, sequence_length, sequence_length)`

    **Output shape**:
        A tuple of two tensors, both with shape: `(batch_size, sequence_length, sequence_length)`
        - `g_attn`: Group attention probabilities.
        - `neibor_attn`: Neighbor attention probabilities (break probabilities).
    """

    def __init__(
        self,
        hidden_size: int,
        normalization_type: str = "layer_norm",
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

        # Create sub-layers in __init__
        self.norm = create_normalization_layer(normalization_type)
        self.linear_key = keras.layers.Dense(
            hidden_size, name="key_projection"
        )
        self.linear_query = keras.layers.Dense(
            hidden_size, name="query_projection"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Builds the sub-layers, which is critical for serialization."""
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
        """Computes group attention probabilities."""
        context, mask, prior = inputs
        current_seq_len = ops.shape(context)[1]

        # --- 1. Compute Neighbor Attention (between adjacent tokens) ---
        # Create an adjacency matrix for immediate neighbors (super/sub-diagonal)
        # using backend-agnostic ops for robust graph tracing.
        i = ops.arange(current_seq_len)
        adj_mask_upper = ops.equal(ops.expand_dims(i, 1), ops.expand_dims(i, 0) - 1)
        adj_mask_lower = ops.equal(ops.expand_dims(i, 1), ops.expand_dims(i, 0) + 1)
        adj_mask = ops.logical_or(adj_mask_upper, adj_mask_lower)

        # Mask out padding positions to prevent attention leakage.
        padding_mask = ops.cast(ops.squeeze(mask, axis=1), "bool")
        padding_mask_2d = ops.logical_and(
            ops.expand_dims(padding_mask, axis=2),
            ops.expand_dims(padding_mask, axis=1),
        )
        final_adj_mask = ops.logical_and(adj_mask, padding_mask_2d)

        # Scaled dot-product attention, masked to only consider neighbors.
        context_norm = self.norm(context, training=training)
        key = self.linear_key(context_norm)
        query = self.linear_query(context_norm)
        scores = ops.matmul(query, ops.transpose(key, axes=(0, 2, 1)))
        scores = scores / math.sqrt(self.hidden_size)
        # DECISION plan_2026-05-11_3c3ed037/D-001
        # dtype-aware mask sentinel: -1e9 overflows fp16 (min ~ -6.5e4),
        # producing NaN after softmax. Use -1e4 under float16 compute_dtype.
        neg_inf = -1e4 if self.compute_dtype == "float16" else -1e9
        scores = ops.where(final_adj_mask, scores, neg_inf)
        neibor_attn = ops.softmax(scores, axis=-1)

        # Symmetrize and stabilize.
        neibor_attn = ops.sqrt(
            neibor_attn * ops.transpose(neibor_attn, axes=(0, 2, 1)) + 1e-9
        )

        # Combine with prior from the previous layer.
        neibor_attn = prior + (1.0 - prior) * neibor_attn

        # --- 2. Tree Induction (Dynamic Programming with Matrix Ops) ---
        # This is a numerically stable, parallelized version of the CKY algorithm.
        row_indices = ops.expand_dims(ops.arange(current_seq_len), 1)
        col_indices = ops.expand_dims(ops.arange(current_seq_len), 0)
        triu_mask = ops.greater_equal(col_indices, row_indices)
        tri_matrix = ops.cast(triu_mask, self.compute_dtype)
        b = ops.cast(ops.eye(current_seq_len, dtype="int32"), "bool")

        # Use log-space for numerical stability to avoid underflow.
        # B-1 fp16 fix: the additive eps must be representable in compute_dtype
        # (fp16 min positive ≈ 6e-5 — `1e-9` collapses to 0 and log(0) → -inf,
        # which then propagates NaN through exp(matmul(...))). Cast the DP
        # block to float32 explicitly so log/exp are always stable, then cast
        # back to compute_dtype.
        neibor_attn_f32 = ops.cast(neibor_attn, "float32")
        tri_matrix_f32 = ops.cast(tri_matrix, "float32")
        t = ops.log(neibor_attn_f32 + 1e-9)
        t = ops.where(adj_mask_upper, t, 0.0)  # Consider only super-diagonal.
        t = ops.matmul(t, tri_matrix_f32)
        g_attn = ops.exp(ops.matmul(tri_matrix_f32, t))
        g_attn = ops.cast(g_attn, self.compute_dtype)

        # Finalize group attention scores.
        g_attn = ops.where(
            ops.logical_xor(triu_mask, b), g_attn, 0.0
        )
        neibor_attn_diag_filled = ops.where(b, 1.0, neibor_attn)
        g_attn = (
            g_attn
            + ops.transpose(g_attn, axes=(0, 2, 1))
            + neibor_attn_diag_filled
        )

        # Normalize g_attn and explicitly zero out padded token interactions.
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
            }
        )
        return config


@keras.saving.register_keras_serializable()
class TreeMHA(keras.layers.Layer):
    """
    Multi-Head Attention modulated by Tree Transformer group probabilities.

    This layer performs standard scaled dot-product multi-head attention, but
    with a key modification: the final attention weights are element-wise
    multiplied by the `group_prob` matrix from the `GroupAttention` layer.

    **Intent**: To bias the self-attention mechanism to focus on tokens within
    the same learned syntactic constituents, integrating the induced tree
    structure directly into the representation learning process.

    **Architecture**:
    .. code-block:: text

        Input(query, key, value, group_prob, mask)
               │
               ▼
        Linear Projections -> Split Heads (Q, K, V)
               │
               ▼
        Scaled Dot-Product Attention: softmax(Q @ Kᵀ / √dₖ)
               │
               ▼
        Modulate with Structure: AttentionWeights * group_prob
               │
               ▼
        Dropout -> Weighted Sum with V -> Concat Heads -> Output Projection
               │
               ▼
        Output(shape=[batch, seq_len, hidden_size])

    :param num_heads: The number of attention heads.
    :type num_heads: int
    :param hidden_size: The dimensionality of the model. Must be divisible by `num_heads`.
    :type hidden_size: int
    :param attention_dropout_rate: Dropout probability for attention scores.
    :type attention_dropout_rate: float
    :param kwargs: Additional keyword arguments for `keras.layers.Layer`.

    **Input shape**:
        A tuple of tensors:
        - `query`, `key`, `value`: `(batch_size, sequence_length, hidden_size)`
        - `group_prob`: `(batch_size, sequence_length, sequence_length)`
        - `mask`: `(batch_size, 1, sequence_length)`

    **Output shape**:
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

        # Create sub-layers in __init__
        self.wq = keras.layers.Dense(hidden_size, name="query")
        self.wk = keras.layers.Dense(hidden_size, name="key")
        self.wv = keras.layers.Dense(hidden_size, name="value")
        self.dense = keras.layers.Dense(hidden_size, name="output_projection")
        self.dropout = keras.layers.Dropout(attention_dropout_rate)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Builds all sub-layers for robust serialization."""
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
        """Forward pass for tree-modulated multi-head attention."""
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
            # dtype-aware mask sentinel (see GroupAttention D-001).
            neg_inf = -1e4 if self.compute_dtype == "float16" else -1e9
            attention_mask = (1.0 - ops.cast(mask, self.compute_dtype)) * neg_inf
            broadcast_mask = ops.expand_dims(attention_mask, axis=1)
            scaled_attention_logits += broadcast_mask

        attention_weights = ops.softmax(scaled_attention_logits, axis=-1)

        # This is the core innovation: modulate attention with learned structure.
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


@keras.saving.register_keras_serializable()
class TreeTransformerBlock(keras.layers.Layer):
    """
    Single block of the Tree Transformer encoder.

    This composite layer encapsulates one full cycle of the Tree Transformer's
    logic: computing syntactic structure, applying structure-aware attention,
    and processing through a feed-forward network. It uses a Pre-LayerNorm
    architecture for improved training stability.

    **Intent**: To provide a modular, repeatable building block for constructing
    the full Tree Transformer encoder stack.

    **Architecture (Pre-LN)**:
    .. code-block:: text

        Input(x, mask, group_prob_prior)
          │
          ├─► GroupAttention(x, mask, group_prob_prior) ──► group_prob_out, break_prob
          │
          └─► LayerNorm₁ ──► TreeMHA(q, k, v, group_prob_out, mask) ─► Add & Dropout ─► x'
                                 ▲
                                 │
              ───────────────────┘ (Residual Connection)

        x' ─► LayerNorm₂ ──► FFN ────────────────────────────────────► Add & Dropout ─► Output(x_out)
          ▲
          │
          └──────────────────────────────────────────────────────────────────────────┘ (Residual Connection)

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
    :param layer_norm_eps: Epsilon for normalization layers.
    :type layer_norm_eps: float
    :param kwargs: Additional keyword arguments for `keras.layers.Layer`.

    **Input shape**:
        Tuple of `(x, mask, group_prob_prior)` as defined in `GroupAttention`.

    **Output shape**:
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

        # Create all sub-layers in __init__
        self.group_attn = GroupAttention(
            hidden_size=hidden_size, normalization_type=normalization_type
        )
        self.self_attn = TreeMHA(
            num_heads=num_heads,
            hidden_size=hidden_size,
            attention_dropout_rate=attention_dropout_rate,
        )
        self.ffn = create_ffn_layer(
            ffn_type=ffn_type,
            hidden_dim=intermediate_size,
            output_dim=hidden_size,
            activation=hidden_act,
            dropout_rate=hidden_dropout_rate,
            name="ffn",
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
        """Builds all sub-layers explicitly for robust serialization."""
        x_shape, mask_shape, prior_shape = input_shape
        group_attn_input_shape = (x_shape, mask_shape, prior_shape)

        self.group_attn.build(group_attn_input_shape)
        self.norm1.build(x_shape)

        # The TreeMHA layer expects a tuple of inputs.
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
        """Forward pass for the Tree Transformer block."""
        x, mask, group_prob = inputs

        # 1. Group Attention to compute constituency structure
        group_prob_out, break_prob = self.group_attn(
            (x, mask, group_prob), training=training
        )

        # 2. Pre-LN MHA with residual connection
        x_norm1 = self.norm1(x)
        attn_output = self.self_attn(
            (x_norm1, x_norm1, x_norm1, group_prob_out, mask),
            training=training,
        )
        x = x + self.dropout1(attn_output, training=training)

        # 3. Pre-LN FFN with residual connection
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

