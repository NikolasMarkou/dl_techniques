import keras
from keras import ops, initializers
from typing import Optional, Any, Dict, Tuple, List, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.ffn.geglu_ffn import GeGLUFFN
from dl_techniques.layers.embedding.rotary_position_embedding import RotaryPositionEmbedding


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ModernBertEmbeddings(keras.layers.Layer):
    """
    ModernBERT embeddings layer combining word and token type embeddings.

    This layer implements the embedding component for a BERT model that uses
    relative position encodings (like RoPE) applied in the attention mechanism,
    and therefore does not include a dedicated absolute position embedding layer.

    **Intent**: Provide a clean, serializable embedding block that generates
    the initial hidden states for the transformer encoder by summing word and
    segment (token type) embeddings, followed by normalization and dropout.

    **Architecture**:
    ```
    Input(input_ids, token_type_ids)
           │
           ├─> Word Embeddings(input_ids) ───┐
           │                                 ▼
           └─> Token Type Embeddings(token_type_ids) ───> Sum -> LayerNorm -> Dropout
                                                            │
                                                            ▼
                                        Output(shape=[batch, seq_len, hidden_size])
    ```

    Args:
        vocab_size: Integer, size of the vocabulary. Must be positive.
        hidden_size: Integer, dimensionality of the embedding vectors. Must be positive.
        type_vocab_size: Integer, vocabulary size for token type IDs (segments). Defaults to 2.
        initializer_range: Float, stddev of the truncated normal initializer. Defaults to 0.02.
        layer_norm_eps: Float, epsilon for the normalization layer. Defaults to 1e-12.
        hidden_dropout_prob: Float, dropout rate for the final embeddings. Defaults to 0.1.
        use_bias: Boolean, whether the normalization layer should use a bias term. Defaults to False.
        **kwargs: Additional keyword arguments for the `keras.layers.Layer` base class.

    Input shape:
        - `input_ids`: 2D tensor of shape `(batch_size, sequence_length)`.
        - `token_type_ids`: (Optional) 2D tensor of shape `(batch_size, sequence_length)`.

    Output shape:
        3D tensor of shape `(batch_size, sequence_length, hidden_size)`.

    Example:
        ```python
        embeddings = ModernBertEmbeddings(
            vocab_size=50368,
            hidden_size=768,
            type_vocab_size=2
        )
        input_ids = keras.random.uniform((2, 128), 0, 50368, dtype='int32')
        token_type_ids = keras.random.uniform((2, 128), 0, 2, dtype='int32')
        output = embeddings(input_ids, token_type_ids)
        # output.shape: (2, 128, 768)
        ```
    """

    def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            type_vocab_size: int = 2,
            initializer_range: float = 0.02,
            layer_norm_eps: float = 1e-12,
            hidden_dropout_prob: float = 0.1,
            use_bias: bool = False,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if type_vocab_size <= 0:
            raise ValueError(f"type_vocab_size must be positive, got {type_vocab_size}")
        if not (0.0 <= hidden_dropout_prob <= 1.0):
            raise ValueError(f"hidden_dropout_prob must be between 0 and 1, got {hidden_dropout_prob}")

        # Store all configuration parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use_bias = use_bias

        # CREATE all sub-layers in __init__
        self.word_embeddings = keras.layers.Embedding(
            self.vocab_size,
            self.hidden_size,
            embeddings_initializer=initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="word_embeddings"
        )
        self.token_type_embeddings = keras.layers.Embedding(
            self.type_vocab_size,
            self.hidden_size,
            embeddings_initializer=initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="token_type_embeddings"
        )
        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            center=self.use_bias,
            name="layer_norm"
        )
        self.dropout = keras.layers.Dropout(
            self.hidden_dropout_prob,
            name="dropout"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build all sub-layers for robust serialization."""
        # Build sub-layers with appropriate shapes
        embedding_shape = (*input_shape, self.hidden_size)

        self.word_embeddings.build(input_shape)
        self.token_type_embeddings.build(input_shape)
        self.layer_norm.build(embedding_shape)
        self.dropout.build(embedding_shape)

        super().build(input_shape)

    def call(
            self,
            input_ids: keras.KerasTensor,
            token_type_ids: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through embeddings."""
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids, dtype="int32")

        word_embeds = self.word_embeddings(input_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        embeddings = word_embeds + token_type_embeds

        embeddings = self.layer_norm(embeddings, training=training)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return (*input_shape, self.hidden_size)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'type_vocab_size': self.type_vocab_size,
            'initializer_range': self.initializer_range,
            'layer_norm_eps': self.layer_norm_eps,
            'hidden_dropout_prob': self.hidden_dropout_prob,
            'use_bias': self.use_bias,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ModernBertAttention(keras.layers.Layer):
    """
    Attention layer for ModernBERT using Rotary Position Embeddings (RoPE).

    This layer implements self-attention where positional information is
    injected into the query and key vectors using RoPE. It supports both
    global (full) attention and local (sliding window) attention patterns.

    **Intent**: Provide a self-contained, serializable attention block that
    integrates RoPE for relative position encoding, which is a key feature
    of modern transformer architectures.

    **Architecture**:
    ```
    Input(hidden_states)
           │
           ├─> Query(Q) ─> Reshape ─> Apply RoPE ─> Reshape back ──┐
           │                                                       │
           ├─> Key(K)   ─> Reshape ─> Apply RoPE ─> Reshape back ──┼──> MultiHeadAttention(Q,K,V)
           │                                                       │
           └─> Value(V) ──────────────────────────────────────────┘
                                                                   │
                                                                   ▼
                                                       Output(shape=[batch, seq_len, hidden_size])
    ```

    Args:
        hidden_size: Integer, hidden size of the attention layer. Must be divisible by num_heads.
        num_heads: Integer, number of attention heads. Must be positive.
        attention_probs_dropout_prob: Float, dropout probability for attention weights.
        use_bias: Boolean, whether to use bias in linear projections.
        initializer_range: Float, standard deviation for weight initialization.
        is_global: Boolean, if True uses global attention, otherwise uses sliding window.
        rope_theta_local: Float, theta parameter for local RoPE. Defaults to 10000.0.
        rope_theta_global: Float, theta parameter for global RoPE. Defaults to 160000.0.
        local_attention_window_size: Integer, window size for local attention. Defaults to 128.
        max_seq_len: Integer, maximum sequence length for RoPE. Defaults to 8192.
        **kwargs: Additional keyword arguments.

    Example:
        ```python
        attention = ModernBertAttention(
            hidden_size=768,
            num_heads=12,
            is_global=True
        )
        hidden_states = keras.random.normal((2, 128, 768))
        output = attention(hidden_states)
        # output.shape: (2, 128, 768)
        ```
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            attention_probs_dropout_prob: float = 0.1,
            use_bias: bool = False,
            initializer_range: float = 0.02,
            is_global: bool = True,
            rope_theta_local: float = 10000.0,
            rope_theta_global: float = 160000.0,
            local_attention_window_size: int = 128,
            max_seq_len: int = 8192,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        if not (0.0 <= attention_probs_dropout_prob <= 1.0):
            raise ValueError(
                f"attention_probs_dropout_prob must be between 0 and 1, got {attention_probs_dropout_prob}"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.use_bias = use_bias
        self.initializer_range = initializer_range
        self.is_global = is_global
        self.rope_theta_local = rope_theta_local
        self.rope_theta_global = rope_theta_global
        self.local_attention_window_size = local_attention_window_size
        self.max_seq_len = max_seq_len

        self.head_dim = hidden_size // num_heads

        # CREATE all sub-layers in __init__
        self.mha = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_dim,
            dropout=self.attention_probs_dropout_prob,
            use_bias=self.use_bias,
            kernel_initializer=initializers.TruncatedNormal(stddev=self.initializer_range),
            name="multi_head_attention"
        )
        self.rotary_embedding = RotaryPositionEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            rope_theta=(self.rope_theta_global if self.is_global else self.rope_theta_local),
            name="rotary_embedding"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build all sub-layers for robust serialization."""
        # Build MultiHeadAttention
        self.mha.build(input_shape, input_shape)

        # Build RotaryPositionEmbedding with 4D shape (batch, seq_len, num_heads, head_dim)
        rope_input_shape = (*input_shape[:-1], self.num_heads, self.head_dim)
        self.rotary_embedding.build(rope_input_shape)

        super().build(input_shape)

    def _apply_rotary_pos_emb(
            self,
            tensor: keras.KerasTensor,
            cos_emb: keras.KerasTensor,
            sin_emb: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Apply rotary position embedding to tensor."""
        x1, x2 = ops.split(tensor, 2, axis=-1)
        half_rot_tensor = ops.concatenate([-x2, x1], axis=-1)
        return (tensor * cos_emb) + (half_rot_tensor * sin_emb)

    def _create_sliding_window_mask(self, seq_len: int) -> keras.KerasTensor:
        """Create sliding window attention mask."""
        positions = ops.arange(seq_len, dtype="int32")
        mask = ops.abs(positions[:, None] - positions[None, :]) < self.local_attention_window_size
        return ops.cast(mask, "bool")

    def call(
            self,
            hidden_states: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through attention."""
        batch_size = ops.shape(hidden_states)[0]
        seq_len = ops.shape(hidden_states)[1]

        # Reshape for RoPE application
        hidden_states_reshaped = ops.reshape(
            hidden_states, (batch_size, seq_len, self.num_heads, self.head_dim)
        )

        # Get RoPE embeddings
        cos_emb, sin_emb = self.rotary_embedding(hidden_states_reshaped)

        # Apply RoPE to query and key
        query = self._apply_rotary_pos_emb(hidden_states_reshaped, cos_emb, sin_emb)
        key = self._apply_rotary_pos_emb(hidden_states_reshaped, cos_emb, sin_emb)

        # Reshape back for attention
        query = ops.reshape(query, (batch_size, seq_len, self.hidden_size))
        key = ops.reshape(key, (batch_size, seq_len, self.hidden_size))
        value = hidden_states

        # Handle attention mask
        final_attention_mask = None
        if attention_mask is not None:
            # The Keras MHA layer expects a mask that is broadcastable to (B, T, S).
            # A 2D padding mask (B, S) must be expanded to (B, 1, S).
            final_attention_mask = ops.expand_dims(attention_mask, axis=1)

        if not self.is_global:
            # Create and combine with a sliding window mask for local attention.
            sliding_mask = self._create_sliding_window_mask(seq_len)
            if final_attention_mask is not None:
                final_attention_mask = ops.logical_and(final_attention_mask, sliding_mask)
            else:
                final_attention_mask = sliding_mask

        # Apply multi-head attention
        return self.mha(
            query=query,
            value=value,
            key=key,
            attention_mask=final_attention_mask,
            training=training
        )

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'attention_probs_dropout_prob': self.attention_probs_dropout_prob,
            'use_bias': self.use_bias,
            'initializer_range': self.initializer_range,
            'is_global': self.is_global,
            'rope_theta_local': self.rope_theta_local,
            'rope_theta_global': self.rope_theta_global,
            'local_attention_window_size': self.local_attention_window_size,
            'max_seq_len': self.max_seq_len,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ModernBertTransformerLayer(keras.layers.Layer):
    """
    A single transformer layer for ModernBERT with pre-normalization.

    This layer combines the self-attention mechanism and a GeGLU feed-forward
    network with residual connections and pre-layer normalization, forming a
    complete transformer block optimized for ModernBERT's architecture.

    **Intent**: To create a modular, serializable building block for the full
    ModernBERT encoder stack by orchestrating specialized sub-components.

    **Architecture (Pre-Normalization)**:
    ```
    Input(hidden_states) ───> Residual Connection ───┐
           │                                          │
           ▼                                          ▼
    LayerNorm ─> ModernBertAttention ─> Add ──────────> Residual Connection ───┐
                                          │                                    │
                                          ▼                                    ▼
                                   LayerNorm ─> GeGLUFFN ───────> Add ──────────> Output
    ```

    Args:
        hidden_size: Integer, hidden size of the layer. Must be positive.
        num_heads: Integer, number of attention heads. Must be positive and divide hidden_size.
        intermediate_size: Integer, size of the intermediate FFN layer. Must be positive.
        hidden_act: String, activation function for FFN. Defaults to 'gelu'.
        hidden_dropout_prob: Float, dropout probability. Defaults to 0.1.
        attention_probs_dropout_prob: Float, attention dropout probability. Defaults to 0.1.
        use_bias: Boolean, whether to use bias in linear layers. Defaults to False.
        initializer_range: Float, standard deviation for weight initialization. Defaults to 0.02.
        layer_norm_eps: Float, epsilon for layer normalization. Defaults to 1e-12.
        is_global: Boolean, whether to use global or local attention. Defaults to True.
        rope_theta_local: Float, RoPE theta for local attention. Defaults to 10000.0.
        rope_theta_global: Float, RoPE theta for global attention. Defaults to 160000.0.
        local_attention_window_size: Integer, window size for local attention. Defaults to 128.
        **kwargs: Additional keyword arguments.

    Example:
        ```python
        transformer = ModernBertTransformerLayer(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            is_global=True
        )
        hidden_states = keras.random.normal((2, 128, 768))
        output = transformer(hidden_states)
        # output.shape: (2, 128, 768)
        ```
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            intermediate_size: int,
            hidden_act: str = "gelu",
            hidden_dropout_prob: float = 0.1,
            attention_probs_dropout_prob: float = 0.1,
            use_bias: bool = False,
            initializer_range: float = 0.02,
            layer_norm_eps: float = 1e-12,
            is_global: bool = True,
            rope_theta_local: float = 10000.0,
            rope_theta_global: float = 160000.0,
            local_attention_window_size: int = 128,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        if intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be positive, got {intermediate_size}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.use_bias = use_bias
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.is_global = is_global
        self.rope_theta_local = rope_theta_local
        self.rope_theta_global = rope_theta_global
        self.local_attention_window_size = local_attention_window_size

        # CREATE sub-layers in __init__
        self.attention = ModernBertAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_bias=use_bias,
            initializer_range=initializer_range,
            is_global=is_global,
            rope_theta_local=rope_theta_local,
            rope_theta_global=rope_theta_global,
            local_attention_window_size=local_attention_window_size,
            name="attention"
        )

        # Use the existing GeGLUFFN from the framework
        self.ffn = GeGLUFFN(
            hidden_dim=intermediate_size,
            output_dim=hidden_size,
            activation=hidden_act,
            dropout_rate=hidden_dropout_prob,
            use_bias=use_bias,
            kernel_initializer=initializers.TruncatedNormal(stddev=initializer_range),
            name="ffn"
        )

        self.attention_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_eps,
            center=use_bias,
            name="attention_norm"
        )
        self.ffn_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_eps,
            center=use_bias,
            name="ffn_norm"
        )
        self.attention_dropout = keras.layers.Dropout(
            hidden_dropout_prob,
            name="attention_dropout"
        )
        self.ffn_dropout = keras.layers.Dropout(
            hidden_dropout_prob,
            name="ffn_dropout"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build all sub-layers for robust serialization."""
        self.attention.build(input_shape)
        self.ffn.build(input_shape)
        self.attention_norm.build(input_shape)
        self.ffn_norm.build(input_shape)
        self.attention_dropout.build(input_shape)
        self.ffn_dropout.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            hidden_states: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through transformer layer."""
        # Pre-normalization attention block
        residual = hidden_states
        hidden_states_norm = self.attention_norm(hidden_states, training=training)
        attention_output = self.attention(
            hidden_states_norm,
            attention_mask=attention_mask,
            training=training
        )
        hidden_states = residual + self.attention_dropout(attention_output, training=training)

        # Pre-normalization FFN block
        residual = hidden_states
        hidden_states_norm = self.ffn_norm(hidden_states, training=training)
        ffn_output = self.ffn(hidden_states_norm, training=training)
        hidden_states = residual + self.ffn_dropout(ffn_output, training=training)

        return hidden_states

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'intermediate_size': self.intermediate_size,
            'hidden_act': self.hidden_act,
            'hidden_dropout_prob': self.hidden_dropout_prob,
            'attention_probs_dropout_prob': self.attention_probs_dropout_prob,
            'use_bias': self.use_bias,
            'initializer_range': self.initializer_range,
            'layer_norm_eps': self.layer_norm_eps,
            'is_global': self.is_global,
            'rope_theta_local': self.rope_theta_local,
            'rope_theta_global': self.rope_theta_global,
            'local_attention_window_size': self.local_attention_window_size,
        })
        return config

# ---------------------------------------------------------------------