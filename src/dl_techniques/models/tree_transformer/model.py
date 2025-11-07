"""
Tree Transformer: Grammar Induction with Hierarchical Attention
===============================================================

A complete and refactored implementation of the Tree Transformer architecture
with support for loading (hypothetical) pretrained weights. This version is
designed as a pure foundation model, separating the core encoding logic from
task-specific heads for maximum flexibility.

The Tree Transformer introduces a hierarchical group attention mechanism that learns
soft constituency trees from raw text, without explicit syntactic supervision.

Based on: "Tree Transformer: Integrating Tree Structures into Self-Attention"
(Shen et al., 2019) https://arxiv.org/abs/1904.00035

Usage Examples:
--------------

.. code-block:: python

    import keras
    from dl_techniques.nlp.heads.factory import create_nlp_head
    from dl_techniques.nlp.heads.task_types import NLPTaskConfig, NLPTaskType

    # 1. Load (hypothetically) pretrained Tree Transformer
    tree_transformer = TreeTransformer.from_variant("base", pretrained=True)

    # 2. Load from local weights file
    tree_transformer = TreeTransformer.from_variant(
        "large", pretrained="path/to/weights.keras"
    )

    # 3. Create Tree Transformer with custom configuration
    tree_transformer = TreeTransformer.from_variant("base", vocab_size=50000)

    # 4. Combine with a task-specific head (e.g., for sequence tagging)
    ner_config = NLPTaskConfig(
        name="ner",
        task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
        num_classes=9
    )
    ner_head = create_nlp_head(
        task_config=ner_config,
        input_dim=tree_transformer.hidden_size
    )

    # 5. Build a complete end-to-end model
    inputs = {
        "input_ids": keras.Input(shape=(None,), dtype="int32", name="input_ids"),
    }
    encoder_outputs = tree_transformer(inputs)
    head_inputs = {"hidden_states": encoder_outputs["last_hidden_state"]}
    task_outputs = ner_head(head_inputs)
    ner_model = keras.Model(inputs, task_outputs)

"""

import os
import math
import keras
import numpy as np
from keras import ops
from typing import Optional, Union, Tuple, Dict, Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.ffn import create_ffn_layer, FFNType
from dl_techniques.layers.norms import (
    create_normalization_layer,
    NormalizationType,
)
from dl_techniques.layers.nlp_heads import create_nlp_head, NLPTaskConfig

# ---------------------------------------------------------------------
# Tree Transformer Sub-Layers
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class PositionalEncoding(keras.layers.Layer):
    """Injects sinusoidal positional encoding into input embeddings."""

    def __init__(
        self,
        hidden_size: int,
        dropout_prob: float,
        max_len: int = 5000,
        **kwargs: Any,
    ) -> None:
        """Initializes the PositionalEncoding layer.

        :param hidden_size: The dimensionality of the embeddings.
        :type hidden_size: int
        :param dropout_prob: Dropout probability after adding encodings.
        :type dropout_prob: float
        :param max_len: Maximum sequence length for pre-computation.
        :type max_len: int
        :param kwargs: Additional keyword arguments for `keras.layers.Layer`.
        """
        super().__init__(**kwargs)
        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {hidden_size}"
            )
        if not (0.0 <= dropout_prob <= 1.0):
            raise ValueError(
                f"dropout_prob must be in [0, 1], got {dropout_prob}"
            )
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.max_len = max_len
        self.dropout = keras.layers.Dropout(dropout_prob)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Creates the non-trainable positional encoding matrix.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        pe = np.zeros((self.max_len, self.hidden_size), dtype=np.float32)
        position = np.arange(0, self.max_len, dtype=np.float32).reshape(-1, 1)
        # The geometric progression of wavelengths for the sine/cosine functions.
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
        """Adds positional encodings to the input tensor.

        :param x: Input tensor of shape (batch, seq_len, hidden_size).
        :type x: keras.KerasTensor
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :return: Tensor with added positional encodings.
        :rtype: keras.KerasTensor
        """
        seq_len = ops.shape(x)[1]
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x, training=training)

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer's configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "dropout_prob": self.dropout_prob,
                "max_len": self.max_len,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class GroupAttention(keras.layers.Layer):
    """Hierarchical group attention for Tree Transformer."""

    def __init__(
        self,
        hidden_size: int,
        normalization_type: str = "layer_norm",
        **kwargs: Any,
    ) -> None:
        """Initializes the GroupAttention layer.

        :param hidden_size: The dimensionality of the model.
        :type hidden_size: int
        :param normalization_type: The type of normalization layer to use.
        :type normalization_type: str
        :param kwargs: Additional keyword arguments for `keras.layers.Layer`.
        """
        super().__init__(**kwargs)
        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {hidden_size}"
            )
        self.hidden_size = hidden_size
        self.normalization_type = normalization_type
        self.norm = create_normalization_layer(normalization_type)
        self.linear_key = keras.layers.Dense(
            hidden_size, name="key_projection"
        )
        self.linear_query = keras.layers.Dense(
            hidden_size, name="query_projection"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Builds the sub-layers.

        :param input_shape: Tuple of shapes for (context, mask, prior).
        :type input_shape: Tuple[Optional[int], ...]
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

        :param inputs: Tuple of (context, mask, prior).
        :type inputs: Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :return: A tuple containing (g_attn, neibor_attn).
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        context, mask, prior = inputs
        # Use ops.shape to handle both eager and symbolic tensors
        current_seq_len = ops.shape(context)[1]

        # --- 1. Compute Neighbor Attention (between adjacent tokens) ---
        # Create adjacency matrix for immediate neighbors (super/sub-diagonal)
        # This implementation is symbolic-friendly and avoids Python conditionals
        # on tensor shapes, which fixes graph tracing issues.
        i = ops.arange(current_seq_len)
        # Super-diagonal mask (i == j - 1)
        adj_mask_upper = ops.equal(
            ops.expand_dims(i, 1), ops.expand_dims(i, 0) - 1
        )
        # Sub-diagonal mask (i == j + 1)
        adj_mask_lower = ops.equal(
            ops.expand_dims(i, 1), ops.expand_dims(i, 0) + 1
        )
        adj_mask = ops.logical_or(adj_mask_upper, adj_mask_lower)

        # Mask out padding positions
        padding_mask = ops.cast(ops.squeeze(mask, axis=1), "bool")
        padding_mask_2d = ops.logical_and(
            ops.expand_dims(padding_mask, axis=2),
            ops.expand_dims(padding_mask, axis=1),
        )
        final_adj_mask = ops.logical_and(adj_mask, padding_mask_2d)

        # Scaled dot-product attention, masked to only neighbors
        context_norm = self.norm(context)
        key = self.linear_key(context_norm)
        query = self.linear_query(context_norm)
        scores = ops.matmul(query, ops.transpose(key, axes=(0, 2, 1)))
        scores = scores / math.sqrt(self.hidden_size)
        scores = ops.where(final_adj_mask, scores, -1e9)
        neibor_attn = ops.softmax(scores, axis=-1)

        # Symmetrize and stabilize
        neibor_attn = ops.sqrt(
            neibor_attn * ops.transpose(neibor_attn, axes=(0, 2, 1)) + 1e-9
        )

        # Combine with prior from previous layer. Prior is expected to be either
        # a scalar tensor (initial) or a (B, L, L) tensor (from previous block).
        neibor_attn = prior + (1.0 - prior) * neibor_attn

        # --- 2. Tree Induction (Dynamic Programming with Matrix Ops) ---
        # Construct masks using fundamental ops for robust serialization
        row_indices = ops.expand_dims(ops.arange(current_seq_len), 1)
        col_indices = ops.expand_dims(ops.arange(current_seq_len), 0)
        triu_mask = ops.greater_equal(col_indices, row_indices)
        tri_matrix = ops.cast(triu_mask, self.compute_dtype)
        b = ops.cast(ops.eye(current_seq_len, dtype="int32"), "bool")

        # Use log-space for numerical stability (avoids underflow)
        t = ops.log(neibor_attn + 1e-9)
        # Mask non-super-diagonal using the symbolic-friendly mask from above
        t = ops.where(adj_mask_upper, t, 0.0)

        t = ops.matmul(t, tri_matrix)
        g_attn = ops.exp(ops.matmul(tri_matrix, t))

        # Finalize group attention scores
        g_attn = ops.where(
            ops.logical_not(ops.logical_xor(triu_mask, b)), g_attn, 0.0
        )
        neibor_attn_diag_filled = ops.where(b, 1e-9, neibor_attn)
        g_attn = (
            g_attn
            + ops.transpose(g_attn, axes=(0, 2, 1))
            + neibor_attn_diag_filled
        )

        # Normalize g_attn to be a valid probability distribution (prior for next layer)
        # and explicitly zero out padded token interactions to prevent information leak.
        g_attn_sum = ops.sum(g_attn, axis=-1, keepdims=True)
        g_attn = ops.divide(g_attn, g_attn_sum + 1e-9)

        float_padding_mask_2d = ops.cast(padding_mask_2d, self.compute_dtype)
        g_attn *= float_padding_mask_2d
        neibor_attn *= float_padding_mask_2d

        return g_attn, neibor_attn

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
    """Multi-Head Attention modulated by Tree Transformer group probabilities."""

    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        attention_dropout_prob: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initializes the TreeMHA layer.

        :param num_heads: The number of attention heads.
        :type num_heads: int
        :param hidden_size: The dimensionality of the model.
        :type hidden_size: int
        :param attention_dropout_prob: Dropout probability for attention scores.
        :type attention_dropout_prob: float
        :param kwargs: Additional keyword arguments for `keras.layers.Layer`.
        """
        super().__init__(**kwargs)
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads != 0:
            raise ValueError("Invalid hidden_size or num_heads configuration.")
        if not (0.0 <= attention_dropout_prob <= 1.0):
            raise ValueError("attention_dropout_prob must be in [0, 1].")

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.attention_dropout_prob = attention_dropout_prob
        self.depth = hidden_size // num_heads
        self.wq = keras.layers.Dense(hidden_size, name="query")
        self.wk = keras.layers.Dense(hidden_size, name="key")
        self.wv = keras.layers.Dense(hidden_size, name="value")
        self.dense = keras.layers.Dense(hidden_size, name="output_projection")
        self.dropout = keras.layers.Dropout(attention_dropout_prob)

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

        :param inputs: Tuple of (query, key, value, group_prob, mask).
        :type inputs: Tuple[...]
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :return: The output tensor of shape (batch, seq_len, hidden_size).
        :rtype: keras.KerasTensor
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
            # The input mask is (B, 1, L) with 1s for valid tokens and 0s for padding.
            # Convert to a float mask with 0.0 for valid and -1e9 for padding.
            attention_mask = (1.0 - ops.cast(mask, self.compute_dtype)) * -1e9
            # Expand to (B, 1, 1, L) for broadcasting over heads and query sequence length.
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

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer's configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_size": self.hidden_size,
                "attention_dropout_prob": self.attention_dropout_prob,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class TreeTransformerBlock(keras.layers.Layer):
    """Single block of the Tree Transformer encoder."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        hidden_dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
        normalization_type: NormalizationType = "layer_norm",
        ffn_type: FFNType = "mlp",
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        """Initializes the TreeTransformerBlock.

        :param hidden_size: Dimensionality of the model.
        :type hidden_size: int
        :param num_heads: Number of attention heads.
        :type num_heads: int
        :param intermediate_size: Dimensionality of the FFN layer.
        :type intermediate_size: int
        :param hidden_dropout_prob: Dropout for hidden layers.
        :type hidden_dropout_prob: float
        :param attention_dropout_prob: Dropout for attention scores.
        :type attention_dropout_prob: float
        :param normalization_type: Type of normalization layer.
        :type normalization_type: NormalizationType
        :param ffn_type: Type of feed-forward network.
        :type ffn_type: FFNType
        :param hidden_act: Activation function for the FFN.
        :type hidden_act: str
        :param layer_norm_eps: Epsilon for normalization layers.
        :type layer_norm_eps: float
        :param kwargs: Additional keyword arguments for `keras.layers.Layer`.
        """
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.normalization_type = normalization_type
        self.ffn_type = ffn_type
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps

        self.group_attn = GroupAttention(
            hidden_size=hidden_size, normalization_type=normalization_type
        )
        self.self_attn = TreeMHA(
            num_heads=num_heads,
            hidden_size=hidden_size,
            attention_dropout_prob=attention_dropout_prob,
        )
        self.ffn = create_ffn_layer(
            ffn_type=ffn_type,
            hidden_dim=intermediate_size,
            output_dim=hidden_size,
            activation=hidden_act,
            dropout_rate=hidden_dropout_prob,
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
        self.dropout1 = keras.layers.Dropout(hidden_dropout_prob)
        self.dropout2 = keras.layers.Dropout(hidden_dropout_prob)

    def call(
        self,
        inputs: Tuple[
            keras.KerasTensor, keras.KerasTensor, keras.KerasTensor
        ],
        training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Forward pass for the Tree Transformer block.

        :param inputs: Tuple of (x, mask, group_prob_prior).
        :type inputs: Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :return: Tuple of (x_out, group_prob_out, break_prob).
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
        """
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

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer's configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "hidden_dropout_prob": self.hidden_dropout_prob,
                "attention_dropout_prob": self.attention_dropout_prob,
                "normalization_type": self.normalization_type,
                "ffn_type": self.ffn_type,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config


# ---------------------------------------------------------------------
# Main TreeTransformer Model
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TreeTransformer(keras.Model):
    """Tree Transformer model for grammar induction and language modeling.

    This is a pure encoder implementation with pretrained weights support,
    designed to produce contextual token representations alongside learned
    syntactic structures (break probabilities). It separates the core transformer
    architecture from any task-specific layers, making it highly flexible.

    The model expects input as a dictionary containing 'input_ids'. It outputs a
    dictionary containing 'last_hidden_state', 'logits' (from LM head), and
    'break_probs' from all layers.

    **Architecture Overview:**

    .. code-block:: text

        Input(input_ids)
               │
               ▼
        Embedding -> PositionalEncoding
               │
               ▼
        TreeTransformerBlock₁ (GroupAttention -> TreeMHA -> FFN)
               │
               ▼
              ...
               │
               ▼
        TreeTransformerBlockₙ (GroupAttention -> TreeMHA -> FFN)
               │
               ▼
        Final LayerNorm -> LM Head
               │
               ▼
        Output Dictionary {
            "last_hidden_state": [batch, seq_len, hidden_size],
            "logits": [batch, seq_len, vocab_size],
            "break_probs": [batch, num_layers, seq_len, seq_len]
        }

    :param vocab_size: Size of the vocabulary. Defaults to 30000.
    :type vocab_size: int
    :param hidden_size: Dimensionality of encoder layers. Defaults to 512.
    :type hidden_size: int
    :param num_layers: Number of hidden transformer layers. Defaults to 10.
    :type num_layers: int
    :param num_heads: Number of attention heads. Defaults to 8.
    :type num_heads: int
    :param intermediate_size: Dimensionality of the FFN layer. Defaults to 2048.
    :type intermediate_size: int
    :param hidden_act: Activation function in the encoder. Defaults to "gelu".
    :type hidden_act: str
    :param hidden_dropout_prob: Dropout for embeddings/encoder. Defaults to 0.1.
    :type hidden_dropout_prob: float
    :param attention_dropout_prob: Dropout for attention scores. Defaults to 0.1.
    :type attention_dropout_prob: float
    :param max_len: Maximum sequence length. Defaults to 256.
    :type max_len: int
    :param layer_norm_eps: Epsilon for normalization layers. Defaults to 1e-6.
    :type layer_norm_eps: float
    :param pad_token_id: ID of the padding token. Defaults to 0.
    :type pad_token_id: int
    :param normalization_type: Type of normalization layer. Defaults to "layer_norm".
    :type normalization_type: str
    :param ffn_type: Type of feed-forward network. Defaults to "mlp".
    :type ffn_type: str
    :param kwargs: Additional keyword arguments for the `keras.Model`.

    :raises ValueError: If invalid configuration parameters are provided.
    """

    MODEL_VARIANTS = {
        "large": {
            "hidden_size": 1024,
            "num_layers": 16,
            "num_heads": 16,
            "intermediate_size": 4096,
            "description": "TreeTransformer-Large: High capacity for large datasets",
        },
        "base": {
            "hidden_size": 512,
            "num_layers": 10,
            "num_heads": 8,
            "intermediate_size": 2048,
            "description": "TreeTransformer-Base: Balanced performance, based on original paper",
        },
        "small": {
            "hidden_size": 256,
            "num_layers": 6,
            "num_heads": 4,
            "intermediate_size": 1024,
            "description": "TreeTransformer-Small: Lightweight for faster training",
        },
        "tiny": {
            "hidden_size": 128,
            "num_layers": 4,
            "num_heads": 4,
            "intermediate_size": 512,
            "description": "TreeTransformer-Tiny: Ultra-lightweight for research",
        },
    }

    PRETRAINED_WEIGHTS = {
        "base": {
            "uncased": "https://example.com/tree_transformer_base_uncased.keras"
        },
        "large": {
            "uncased": "https://example.com/tree_transformer_large_uncased.keras"
        },
    }

    DEFAULT_VOCAB_SIZE = 30000
    DEFAULT_MAX_LEN = 256
    DEFAULT_LAYER_NORM_EPSILON = 1e-6
    DEFAULT_HIDDEN_ACT = "gelu"
    DEFAULT_PAD_TOKEN_ID = 0

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        hidden_size: int = 512,
        num_layers: int = 10,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        hidden_act: str = DEFAULT_HIDDEN_ACT,
        hidden_dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
        max_len: int = DEFAULT_MAX_LEN,
        layer_norm_eps: float = DEFAULT_LAYER_NORM_EPSILON,
        pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
        normalization_type: NormalizationType = "layer_norm",
        ffn_type: FFNType = "mlp",
        **kwargs: Any,
    ) -> None:
        """Initializes the TreeTransformer model instance."""
        super().__init__(**kwargs)

        self._validate_config(
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            hidden_dropout_prob,
            attention_dropout_prob,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.max_len = max_len
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.normalization_type = normalization_type
        self.ffn_type = ffn_type

        self._build_architecture()
        logger.info(
            f"Created Tree Transformer foundation model: {self.num_layers} layers, "
            f"hidden_size={self.hidden_size}, heads={self.num_heads}"
        )

    def _validate_config(
        self,
        vocab_size,
        hidden_size,
        num_layers,
        num_heads,
        hidden_dropout_prob,
        attention_dropout_prob,
    ) -> None:
        """Validates model configuration parameters."""
        if vocab_size <= 0:
            raise ValueError(
                f"vocab_size must be positive, got {vocab_size}"
            )
        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {hidden_size}"
            )
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be positive, got {num_layers}"
            )
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if not (0.0 <= hidden_dropout_prob <= 1.0):
            raise ValueError(
                f"hidden_dropout_prob must be in [0, 1], got {hidden_dropout_prob}"
            )
        if not (0.0 <= attention_dropout_prob <= 1.0):
            raise ValueError(
                f"attention_dropout_prob must be in [0, 1], got {attention_dropout_prob}"
            )

    def _build_architecture(self) -> None:
        """Builds all model components."""
        self.embedding = keras.layers.Embedding(
            self.vocab_size, self.hidden_size, name="embedding"
        )
        self.pos_encoding = PositionalEncoding(
            hidden_size=self.hidden_size,
            dropout_prob=self.hidden_dropout_prob,
            max_len=self.max_len,
            name="pos_encoding",
        )
        self.blocks: List[TreeTransformerBlock] = [
            TreeTransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attention_dropout_prob=self.attention_dropout_prob,
                normalization_type=self.normalization_type,
                ffn_type=self.ffn_type,
                hidden_act=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps,
                name=f"block_{i}",
            )
            for i in range(self.num_layers)
        ]
        self.final_norm = create_normalization_layer(
            normalization_type=self.normalization_type,
            epsilon=self.layer_norm_eps,
            name="final_norm",
        )
        self.lm_head = keras.layers.Dense(
            self.vocab_size, name="lm_head_projection"
        )

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer."""
        if isinstance(input_shape, dict):
            input_shape = input_shape["input_ids"]

        batch_size, seq_len = input_shape

        return {
            "last_hidden_state": (batch_size, seq_len, self.hidden_size),
            "logits": (batch_size, seq_len, self.vocab_size),
            "break_probs": (batch_size, self.num_layers, seq_len, seq_len),
        }

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass of the TreeTransformer model.

        :param inputs: Input token IDs or a dictionary with 'input_ids'.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param training: Indicates if the model is in training mode.
        :type training: Optional[bool]
        :return: A dictionary with 'last_hidden_state', 'logits', 'break_probs'.
        :rtype: Dict[str, keras.KerasTensor]
        :raises ValueError: If dictionary input does not contain 'input_ids'.
        """
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError(
                    "Dictionary input must contain 'input_ids' key"
                )
        else:
            input_ids = inputs

        mask = ops.cast(
            ops.not_equal(input_ids, self.pad_token_id), "int32"
        )
        mask = ops.expand_dims(mask, axis=1)

        x = self.embedding(input_ids)
        # Standard transformer practice: scale embeddings by sqrt of model dimension.
        x *= ops.cast(self.hidden_size, x.dtype) ** 0.5
        x = self.pos_encoding(x, training=training)

        # Initialize group probability for the first layer as a scalar tensor (0.0).
        # FIX: Must be a tensor (not a Python float) for Keras tracing/building.
        group_prob: keras.KerasTensor = ops.convert_to_tensor(
            0.0, dtype=self.compute_dtype
        )
        all_break_probs = []

        # Pass through the stack of transformer blocks.
        for block in self.blocks:
            x, group_prob, break_prob = block(
                (x, mask, group_prob), training=training
            )
            all_break_probs.append(break_prob)

        last_hidden_state = self.final_norm(x)
        logits = self.lm_head(last_hidden_state)
        stacked_break_probs = ops.stack(all_break_probs, axis=1)

        return {
            "last_hidden_state": last_hidden_state,
            "logits": logits,
            "break_probs": stacked_break_probs,
        }

    def load_pretrained_weights(
        self,
        weights_path: str,
        skip_mismatch: bool = True,
        by_name: bool = True,
    ) -> None:
        """Loads pretrained weights into the model.

        :param weights_path: Path to the weights file (.keras format).
        :type weights_path: str
        :param skip_mismatch: Whether to skip layers with mismatched shapes.
        :type skip_mismatch: bool
        :param by_name: Whether to load weights by layer name.
        :type by_name: bool
        :raises FileNotFoundError: If weights_path doesn't exist.
        :raises ValueError: If weights cannot be loaded.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        try:
            if not self.built:
                dummy_input = {
                    "input_ids": keras.random.uniform(
                        (1, 64), 0, self.vocab_size, dtype="int32"
                    )
                }
                self(dummy_input, training=False)
            logger.info(f"Loading pretrained weights from {weights_path}")
            self.load_weights(
                weights_path, skip_mismatch=skip_mismatch, by_name=by_name
            )
            if skip_mismatch:
                logger.info(
                    "Weights loaded with skip_mismatch=True. "
                    "Layers with shape mismatches were skipped."
                )
            else:
                logger.info("All weights loaded successfully.")
        except Exception as e:
            raise ValueError(
                f"Failed to load weights from {weights_path}: {str(e)}"
            )

    @staticmethod
    def _download_weights(
        variant: str, dataset: str = "uncased", cache_dir: Optional[str] = None
    ) -> str:
        """Downloads pretrained weights from URL.

        :param variant: Model variant name.
        :type variant: str
        :param dataset: Dataset version (e.g., "uncased").
        :type dataset: str
        :param cache_dir: Directory to cache downloaded weights.
        :type cache_dir: Optional[str]
        :return: Path to the downloaded weights file.
        :rtype: str
        :raises ValueError: If variant or dataset is not available.
        """
        if (
            variant not in TreeTransformer.PRETRAINED_WEIGHTS
            or dataset not in TreeTransformer.PRETRAINED_WEIGHTS[variant]
        ):
            raise ValueError(
                f"No pretrained weights available for variant '{variant}' "
                f"with dataset '{dataset}'."
            )
        url = TreeTransformer.PRETRAINED_WEIGHTS[variant][dataset]
        logger.info(
            f"Downloading TreeTransformer-{variant} ({dataset}) weights..."
        )
        weights_path = keras.utils.get_file(
            fname=f"tree_transformer_{variant}_{dataset}.keras",
            origin=url,
            cache_dir=cache_dir,
            cache_subdir="models/tree_transformer",
        )
        logger.info(f"Weights downloaded to: {weights_path}")
        return weights_path

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "uncased",
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> "TreeTransformer":
        """Creates a TreeTransformer model from a predefined variant.

        :param variant: Name of the variant ("base", "large", etc.).
        :type variant: str
        :param pretrained: If True, loads pretrained weights. If str, path to local weights.
        :type pretrained: Union[bool, str]
        :param weights_dataset: Dataset for pretrained weights ("uncased").
        :type weights_dataset: str
        :param cache_dir: Directory to cache downloaded weights.
        :type cache_dir: Optional[str]
        :param kwargs: Additional arguments to override variant defaults.
        :type kwargs: Any
        :return: A TreeTransformer model instance.
        :rtype: TreeTransformer
        :raises ValueError: If the variant is not recognized.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(cls.MODEL_VARIANTS.keys())}"
            )
        config = cls.MODEL_VARIANTS[variant].copy()
        description = config.pop("description", "")
        logger.info(
            f"Creating TreeTransformer-{variant.upper()} model: {description}"
        )
        load_weights_path, skip_mismatch = None, False
        if pretrained:
            if isinstance(pretrained, str):
                load_weights_path = pretrained
            else:
                try:
                    load_weights_path = cls._download_weights(
                        variant, weights_dataset, cache_dir
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to download pretrained weights: {e}. "
                        "Continuing with random initialization."
                    )
            # If a custom vocab size is passed, we must skip the embedding layer
            # during weight loading as shapes will not match.
            if (
                "vocab_size" in kwargs
                and kwargs["vocab_size"] != cls.DEFAULT_VOCAB_SIZE
            ):
                skip_mismatch = True
                logger.info(
                    "Custom vocab_size differs from pretrained, will skip "
                    "embedding and LM head weights."
                )
        config.update(kwargs)
        model = cls(**config)
        if load_weights_path:
            try:
                model.load_pretrained_weights(
                    load_weights_path,
                    skip_mismatch=skip_mismatch,
                    by_name=True,
                )
            except Exception as e:
                logger.error(f"Failed to load pretrained weights: {e}")
                raise
        return model

    def get_config(self) -> Dict[str, Any]:
        """Returns the model's configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "hidden_act": self.hidden_act,
                "hidden_dropout_prob": self.hidden_dropout_prob,
                "attention_dropout_prob": self.attention_dropout_prob,
                "max_len": self.max_len,
                "layer_norm_eps": self.layer_norm_eps,
                "pad_token_id": self.pad_token_id,
                "normalization_type": self.normalization_type,
                "ffn_type": self.ffn_type,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TreeTransformer":
        """Creates a model instance from its configuration."""
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Prints the model summary with additional configuration details."""
        super().summary(**kwargs)
        logger.info("Tree Transformer Foundation Model Configuration:")
        logger.info(
            f"  - Architecture: {self.num_layers} layers, {self.hidden_size} hidden size"
        )
        logger.info(f"  - Attention: {self.num_heads} heads")
        logger.info(f"  - Vocabulary: {self.vocab_size} tokens")
        logger.info(f"  - Max sequence length: {self.max_len}")
        logger.info(
            f"  - Normalization: {self.normalization_type} (Pre-LN in blocks)"
        )
        logger.info(
            f"  - Feed-forward: {self.ffn_type}, {self.intermediate_size} intermediate size"
        )


# ---------------------------------------------------------------------
# Integration with NLP Task Heads
# ---------------------------------------------------------------------


def create_tree_transformer_with_head(
    tree_transformer_variant: str,
    task_config: NLPTaskConfig,
    pretrained: Union[bool, str] = False,
    weights_dataset: str = "uncased",
    cache_dir: Optional[str] = None,
    encoder_config_overrides: Optional[Dict[str, Any]] = None,
    head_config_overrides: Optional[Dict[str, Any]] = None,
) -> keras.Model:
    """Factory function to create a Tree Transformer model with a task-specific head.

    :param tree_transformer_variant: The Tree Transformer variant (e.g., "base").
    :type tree_transformer_variant: str
    :param task_config: An `NLPTaskConfig` object defining the task.
    :type task_config: NLPTaskConfig
    :param pretrained: If True, loads pretrained weights. If str, path to local weights.
    :type pretrained: Union[bool, str]
    :param weights_dataset: Dataset for pretrained weights ("uncased").
    :type weights_dataset: str
    :param cache_dir: Directory to cache downloaded weights.
    :type cache_dir: Optional[str]
    :param encoder_config_overrides: Dict to override default encoder config.
    :type encoder_config_overrides: Optional[Dict[str, Any]]
    :param head_config_overrides: Dict to override default head config.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :return: A complete `keras.Model` ready for the specified task.
    :rtype: keras.Model
    """
    encoder_config_overrides = encoder_config_overrides or {}
    head_config_overrides = head_config_overrides or {}
    logger.info(
        f"Creating TreeTransformer-{tree_transformer_variant} with a '{task_config.name}' head."
    )

    # 1. Create the foundational TreeTransformer model
    tree_encoder = TreeTransformer.from_variant(
        tree_transformer_variant,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        cache_dir=cache_dir,
        **encoder_config_overrides,
    )

    # 2. Create the task-specific head
    task_head = create_nlp_head(
        task_config=task_config,
        input_dim=tree_encoder.hidden_size,
        **head_config_overrides,
    )

    # 3. Define inputs and build the end-to-end model
    inputs = {
        "input_ids": keras.Input(
            shape=(None,), dtype="int32", name="input_ids"
        )
    }
    encoder_outputs = tree_encoder(inputs)

    # Pass encoder outputs to the task head.
    # Note: Depending on the task, you might also want to use the 'break_probs'
    # from encoder_outputs for syntax-aware tasks.
    head_inputs = {"hidden_states": encoder_outputs["last_hidden_state"]}
    task_outputs = task_head(head_inputs)

    model_name = (
        f"tree_transformer_{tree_transformer_variant}_with_{task_config.name}_head"
    )
    model = keras.Model(inputs=inputs, outputs=task_outputs, name=model_name)
    logger.info(
        f"Successfully created model with {model.count_params():,} parameters."
    )
    return model