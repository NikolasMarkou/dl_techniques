import keras
from keras import ops, initializers, layers
from typing import Optional, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.ffn.geglu_ffn import GeGLUFFN
from dl_techniques.layers.embedding.rotary_position_embedding import (
    RotaryPositionEmbedding,
)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ModernBertEmbeddings(keras.layers.Layer):
    """
    Computes embeddings for ModernBERT from token and type IDs.

    This layer handles the initial embedding lookup for input tokens and segment
    (token type) IDs. It combines these two embeddings and then applies layer
    normalization and dropout. Notably, it does not include absolute position
    embeddings, as positional information is handled by Rotary Position
    Embeddings (RoPE) within the attention layers.

    **Intent**:
    To provide the initial, fixed-dimensional vector representations for input
    tokens that serve as the input to the main transformer encoder stack,
    following modern Keras patterns for robust serialization.

    **Architecture**:
    ```
    Input (input_ids, token_type_ids)
           ↓
    Word Embeddings + Token Type Embeddings
           ↓
    Layer Normalization
           ↓
    Dropout
           ↓
    Output [batch, seq_len, hidden_size]
    ```

    Args:
        vocab_size: Integer, the size of the vocabulary.
        hidden_size: Integer, the dimensionality of the embedding vectors.
        type_vocab_size: Integer, the number of segment types (e.g., 2 for
            sentence A/B).
        initializer_range: Float, standard deviation for the truncated normal
            initializer used for embedding weights.
        layer_norm_eps: Float, a small epsilon value for numerical stability in
            the layer normalization.
        hidden_dropout_prob: Float, dropout rate applied to the final embeddings.
        use_bias: Boolean, whether the layer normalization sub-layer should use
            a bias term.
        **kwargs: Additional arguments for the `keras.layers.Layer` base class.

    Input shape:
        - `input_ids`: 2D tensor of shape `(batch_size, sequence_length)`.
        - `token_type_ids`: (Optional) 2D tensor of shape
          `(batch_size, sequence_length)`.

    Output shape:
        A 3D tensor of shape `(batch_size, sequence_length, hidden_size)`.

    Attributes:
        word_embeddings: `layers.Embedding` for token IDs.
        token_type_embeddings: `layers.Embedding` for segment type IDs.
        layer_norm: `layers.LayerNormalization` applied after embedding summation.
        dropout: `layers.Dropout` applied as the final step.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        type_vocab_size: int,
        initializer_range: float,
        layer_norm_eps: float,
        hidden_dropout_prob: float,
        use_bias: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Store all configuration
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use_bias = use_bias

        # CREATE all sub-layers in __init__ (they remain unbuilt)
        self.word_embeddings = layers.Embedding(
            self.vocab_size,
            self.hidden_size,
            embeddings_initializer=initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="word_embeddings",
        )
        self.token_type_embeddings = layers.Embedding(
            self.type_vocab_size,
            self.hidden_size,
            embeddings_initializer=initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="token_type_embeddings",
        )
        self.layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, center=self.use_bias, name="layer_norm"
        )
        self.dropout = layers.Dropout(self.hidden_dropout_prob, name="dropout")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Creates the weights for the embedding, norm, and dropout layers."""
        # Build sub-layers explicitly in computational order for robust serialization
        self.word_embeddings.build(input_shape)
        self.token_type_embeddings.build(input_shape)

        # The output shape of the embedding summation is needed to build subsequent layers
        embedding_shape = tuple(input_shape) + (self.hidden_size,)
        self.layer_norm.build(embedding_shape)
        self.dropout.build(embedding_shape)

        super().build(input_shape)

    def call(
        self,
        input_ids: keras.KerasTensor,
        token_type_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """
        Computes the final embedding vectors.

        Args:
            input_ids: Tensor of token indices.
            token_type_ids: (Optional) Tensor of segment indices.
            training: (Optional) Boolean indicating training mode for dropout.

        Returns:
            The combined, normalized, and regularized embedding tensor.
        """
        seq_length = ops.shape(input_ids)[1]
        # Default token_type_ids to zeros if not provided
        if token_type_ids is None:
            token_type_ids = ops.zeros(
                (ops.shape(input_ids)[0], seq_length), dtype="int32"
            )

        word_embeds = self.word_embeddings(input_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        embeddings = word_embeds + token_type_embeds

        embeddings = self.layer_norm(embeddings, training=training)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer's configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "type_vocab_size": self.type_vocab_size,
                "initializer_range": self.initializer_range,
                "layer_norm_eps": self.layer_norm_eps,
                "hidden_dropout_prob": self.hidden_dropout_prob,
                "use_bias": self.use_bias,
            }
        )
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ModernBertAttention(keras.layers.Layer):
    """
    Multi-head attention layer with Rotary Position Embeddings (RoPE).

    This layer implements multi-head self-attention, incorporating relative
    positional information by applying RoPE to the query and key projections.
    It supports two modes of operation controlled by the `is_global` flag:
    1.  **Global Attention**: Standard full self-attention.
    2.  **Local Attention**: A sliding-window attention mechanism for
        computational efficiency with long sequences.

    **Intent**:
    To provide a flexible and powerful attention mechanism that captures contextual
    relationships between tokens while respecting their relative positions,
    optimized for both global context and local efficiency.

    **Architecture**:
    ```
    Input [batch, seq_len, hidden_size]
           ↓
    Reshape & Transpose for RoPE
           ↓
    Apply RotaryPositionEmbedding to Q, K
           ↓
    Reshape & Transpose back for MHA
           ↓
    Generate Attention Mask (Padding + optional Sliding Window)
           ↓
    MultiHeadAttention(Q_rotated, K_rotated, V_original)
           ↓
    Output [batch, seq_len, hidden_size]
    ```

    Args:
        hidden_size: Integer, the dimensionality of the input and output.
        num_heads: Integer, the number of attention heads.
        attention_probs_dropout_prob: Float, dropout rate for attention scores.
        initializer_range: Float, stddev for weight initialization.
        rope_theta: Float, the theta parameter for RoPE.
        max_seq_len: Integer, maximum sequence length for RoPE cache.
        local_attention_window_size: Integer, window size for local attention.
        use_bias: Boolean, whether to use bias in the MHA sub-layer.
        is_global: Boolean, if True, performs global attention; otherwise,
            performs local sliding-window attention.
        **kwargs: Additional arguments for the `keras.layers.Layer` base class.

    Input shape:
        - `hidden_states`: 3D tensor of shape `(batch_size, sequence_length, hidden_size)`.
        - `attention_mask`: (Optional) 2D tensor for padding, shape
          `(batch_size, sequence_length)`.

    Output shape:
        A 3D tensor of the same shape as the input:
        `(batch_size, sequence_length, hidden_size)`.

    Attributes:
        mha: `layers.MultiHeadAttention` sub-layer.
        rotary_embedding: `RotaryPositionEmbedding` sub-layer.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_probs_dropout_prob: float,
        initializer_range: float,
        rope_theta: float,
        max_seq_len: int,
        local_attention_window_size: int,
        use_bias: bool,
        is_global: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        # Store all configuration
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.local_attention_window_size = local_attention_window_size
        self.use_bias = use_bias
        self.is_global = is_global
        self.head_dim = hidden_size // num_heads

        # CREATE all sub-layers in __init__
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_dim,
            dropout=self.attention_probs_dropout_prob,
            use_bias=self.use_bias,
            kernel_initializer=initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="multi_head_attention",
        )
        self.rotary_embedding = RotaryPositionEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            rope_theta=self.rope_theta,
            name="rotary_embedding",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Builds the MHA and RoPE sub-layers."""
        self.mha.build(query_shape=input_shape, value_shape=input_shape)

        # Build RoPE with its expected 4D input shape (batch, heads, seq, dim)
        dummy_4d_shape = tuple(input_shape[:-1]) + (
            self.num_heads,
            self.head_dim,
        )
        self.rotary_embedding.build(dummy_4d_shape)

        super().build(input_shape)

    def _create_sliding_window_mask(self, seq_len: int) -> keras.KerasTensor:
        """Creates a boolean mask for sliding window local attention."""
        # Creates a [seq_len, seq_len] matrix where entry (i, j) is |i - j|.
        positions = ops.arange(seq_len, dtype="int32")
        distance_matrix = ops.abs(positions[:, None] - positions[None, :])
        # The mask is True where the distance is within the window size.
        mask = distance_matrix < self.local_attention_window_size
        return ops.cast(mask, "bool")

    def call(
        self,
        hidden_states: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass for attention, applying RoPE and attention masking."""
        batch_size = ops.shape(hidden_states)[0]
        seq_len = ops.shape(hidden_states)[1]

        # Reshape to (batch, heads, seq, dim) for applying RoPE
        query_reshaped = ops.reshape(
            hidden_states, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        query_reshaped = ops.transpose(query_reshaped, [0, 2, 1, 3])
        key_reshaped = ops.reshape(
            hidden_states, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        key_reshaped = ops.transpose(key_reshaped, [0, 2, 1, 3])

        # Apply Rotary Position Embeddings
        query_rotated = self.rotary_embedding(query_reshaped, training=training)
        key_rotated = self.rotary_embedding(key_reshaped, training=training)

        # Transpose back and reshape to 3D for MultiHeadAttention layer
        query_rotated = ops.transpose(query_rotated, [0, 2, 1, 3])
        key_rotated = ops.transpose(key_rotated, [0, 2, 1, 3])
        query = ops.reshape(
            query_rotated, (batch_size, seq_len, self.hidden_size)
        )
        key = ops.reshape(key_rotated, (batch_size, seq_len, self.hidden_size))
        value = hidden_states

        # --- Attention Masking Logic ---
        final_attention_mask = None
        # 1. Handle the padding mask if it exists
        if attention_mask is not None:
            attention_mask = ops.cast(attention_mask, "bool")
            # Expand 2D mask to 4D for MHA broadcasting:
            # (batch, seq) -> (batch, 1, 1, seq)
            if ops.ndim(attention_mask) == 2:
                final_attention_mask = attention_mask[:, None, None, :]
            else:  # Assume it's already broadcastable
                final_attention_mask = attention_mask

        # 2. If local attention, create and combine the sliding window mask
        if not self.is_global:
            sliding_mask = self._create_sliding_window_mask(seq_len)
            if final_attention_mask is not None:
                # Combine padding mask and sliding mask via logical AND.
                # Broadcasting (batch, 1, 1, seq) & (seq, seq) -> (batch, 1, seq, seq)
                final_attention_mask = ops.logical_and(
                    final_attention_mask, sliding_mask
                )
            else:
                # No padding mask, just use the sliding mask.
                # Expand dims to broadcast across batch and heads.
                final_attention_mask = sliding_mask[None, None, :, :]

        return self.mha(
            query=query,
            value=value,
            key=key,
            attention_mask=final_attention_mask,
            training=training,
        )

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer's configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
                "initializer_range": self.initializer_range,
                "rope_theta": self.rope_theta,
                "max_seq_len": self.max_seq_len,
                "local_attention_window_size": self.local_attention_window_size,
                "use_bias": self.use_bias,
                "is_global": self.is_global,
            }
        )
        return config

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ModernBertEncoderLayer(keras.layers.Layer):
    """
    A complete ModernBERT Transformer encoder layer.

    This layer implements a single block of the Transformer encoder using a
    Pre-Normalization (Pre-LN) architecture. It is composed of a
    `ModernBertAttention` layer and a `GeGLUFFN` (feed-forward) layer, each
    preceded by layer normalization and followed by a residual connection.

    **Intent**:
    To encapsulate one full block of transformer processing, making it easy to
    stack these layers to build the full encoder, while handling the logic for
    selecting global or local attention.

    **Architecture (Pre-LN)**:
    ```
    Input
      | ------------------------------------┐
      ↓                                     |
    LayerNorm                               |
      ↓                                     |
    ModernBertAttention                     |
      ↓                                     |
    Add (Residual Connection) ---------------+
      | ------------------------------------┐
      ↓                                     |
    LayerNorm                               |
      ↓                                     |
    GeGLUFFN                                |
      ↓                                     |
    Add (Residual Connection) ---------------+
      ↓
    Output
    ```

    Args:
        is_global: Boolean, passed to the attention sub-layer to determine
            whether to use global or local attention.
        **kwargs: A dictionary of configuration arguments that are passed to
            the sub-layers. Includes `hidden_size`, `num_heads`,
            `intermediate_size`, etc.

    Input shape:
        - `hidden_states`: 3D tensor of shape `(batch_size, sequence_length, hidden_size)`.
        - `attention_mask`: (Optional) 2D tensor for padding, shape
          `(batch_size, sequence_length)`.

    Output shape:
        A 3D tensor of the same shape as the input:
        `(batch_size, sequence_length, hidden_size)`.

    Attributes:
        attention_norm: First `LayerNormalization`.
        attention: `ModernBertAttention` instance.
        ffn_norm: Second `LayerNormalization`.
        ffn: `GeGLUFFN` instance.
    """

    def __init__(self, is_global: bool, **kwargs: Any) -> None:
        # Store all kwargs for serialization and sub-layer construction
        self.all_config = kwargs.copy()
        self.is_global = is_global

        # Filter kwargs to only pass base Layer arguments to super().__init__
        base_kwargs = {
            k: v for k, v in kwargs.items() if k in ["name", "trainable", "dtype"]
        }
        super().__init__(**base_kwargs)

        self.attention_norm = layers.LayerNormalization(
            epsilon=self.all_config["layer_norm_eps"],
            center=self.all_config["use_bias"],
            name="attention_norm",
        )
        self.attention = ModernBertAttention(
            is_global=self.is_global,
            rope_theta=(
                self.all_config["rope_theta_global"]
                if self.is_global
                else self.all_config["rope_theta_local"]
            ),
            hidden_size=self.all_config["hidden_size"],
            num_heads=self.all_config["num_heads"],
            attention_probs_dropout_prob=self.all_config[
                "attention_probs_dropout_prob"
            ],
            initializer_range=self.all_config["initializer_range"],
            max_seq_len=self.all_config["max_seq_len"],
            local_attention_window_size=self.all_config[
                "local_attention_window_size"
            ],
            use_bias=self.all_config["use_bias"],
            name="attention",
        )
        self.ffn_norm = layers.LayerNormalization(
            epsilon=self.all_config["layer_norm_eps"],
            center=self.all_config["use_bias"],
            name="ffn_norm",
        )
        self.ffn = GeGLUFFN(
            hidden_dim=self.all_config["intermediate_size"],
            output_dim=self.all_config["hidden_size"],
            activation=self.all_config["hidden_act"],
            dropout_rate=self.all_config["hidden_dropout_prob"],
            use_bias=self.all_config["use_bias"],
            kernel_initializer=initializers.TruncatedNormal(
                stddev=self.all_config["initializer_range"]
            ),
            name="ffn",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Builds all sub-layers in the encoder block."""
        self.attention_norm.build(input_shape)
        self.attention.build(input_shape)
        self.ffn_norm.build(input_shape)
        self.ffn.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        hidden_states: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass implementing the Pre-LN Transformer architecture."""
        # --- First sub-layer: Self-Attention ---
        residual = hidden_states
        x = self.attention_norm(hidden_states, training=training)
        x = self.attention(x, attention_mask=attention_mask, training=training)
        attention_output = x + residual

        # --- Second sub-layer: Feed-Forward Network ---
        residual = attention_output
        x = self.ffn_norm(attention_output, training=training)
        x = self.ffn(x, training=training)
        layer_output = x + residual

        return layer_output

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer's configuration for serialization."""
        config = super().get_config()
        config.update(self.all_config)
        config["is_global"] = self.is_global
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModernBertEncoderLayer":
        """Creates a layer from its config, handling the kwargs-based init."""
        is_global = config.pop("is_global")
        # The remaining items in the config dictionary are the kwargs
        return cls(is_global=is_global, **config)

# ---------------------------------------------------------------------
