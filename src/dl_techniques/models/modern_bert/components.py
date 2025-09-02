import keras
from keras import ops, initializers
from typing import Optional, Any, Dict, Tuple, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.ffn.ge_glu_fnn import GeGLUFFN
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
        vocab_size: Integer, size of the vocabulary.
        hidden_size: Integer, dimensionality of the embedding vectors.
        type_vocab_size: Integer, vocabulary size for token type IDs (segments).
        initializer_range: Float, stddev of the truncated normal initializer.
        layer_norm_eps: Float, epsilon for the normalization layer.
        hidden_dropout_prob: Float, dropout rate for the final embeddings.
        use_bias: Boolean, whether the normalization layer should use a bias term.
        **kwargs: Additional keyword arguments for the `keras.layers.Layer` base class.

    Input shape:
        - `input_ids`: 2D tensor of shape `(batch_size, sequence_length)`.
        - `token_type_ids`: (Optional) 2D tensor of shape `(batch_size, sequence_length)`.

    Output shape:
        3D tensor of shape `(batch_size, sequence_length, hidden_size)`.
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
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

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


@keras.saving.register_keras_serializable()
class ModernBertAttention(keras.layers.Layer):
    """
    Attention layer for ModernBERT using Rotary Position Embeddings (RoPE).

    This layer implements self-attention where positional information is
    injected into the query and key vectors using RoPE. It supports both
    global (full) attention and local (sliding window) attention.

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
        hidden_size, num_heads, etc.: Standard transformer parameters.
        is_global: Boolean, if True, uses global attention. Otherwise, uses a
            sliding window attention mask.
        rope_theta_local/global: Theta parameters for RoPE.
        local_attention_window_size: Integer, window size for local attention.
        **kwargs: Additional keyword arguments for the `keras.layers.Layer` base class.
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            attention_probs_dropout_prob: float,
            use_bias: bool,
            initializer_range: float,
            is_global: bool,
            rope_theta_local: float,
            rope_theta_global: float,
            local_attention_window_size: int,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.use_bias = use_bias
        self.initializer_range = initializer_range
        self.is_global = is_global
        self.rope_theta_local = rope_theta_local
        self.rope_theta_global = rope_theta_global
        self.local_attention_window_size = local_attention_window_size

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
            max_seq_len=8192,  # A large enough value
            rope_theta=(self.rope_theta_global if self.is_global else self.rope_theta_local),
            name="rotary_embedding"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build all sub-layers."""
        self.mha.build(query_shape=input_shape, value_shape=input_shape)

        # RoPE expects a 4D shape for building
        dummy_4d_shape = (*input_shape[:-1], self.num_heads, self.head_dim)
        self.rotary_embedding.build(dummy_4d_shape)
        super().build(input_shape)

    def _apply_rotary_pos_emb(self, tensor: keras.KerasTensor, cos_emb: keras.KerasTensor,
                              sin_emb: keras.KerasTensor) -> keras.KerasTensor:
        x1, x2 = ops.split(tensor, 2, axis=-1)
        half_rot_tensor = ops.concatenate([-x2, x1], axis=-1)
        return (tensor * cos_emb) + (half_rot_tensor * sin_emb)

    def _create_sliding_window_mask(self, seq_len: int) -> keras.KerasTensor:
        positions = ops.arange(seq_len, dtype="int32")
        mask = ops.abs(positions[:, None] - positions[None, :]) < self.local_attention_window_size
        return ops.cast(mask, "bool")

    def call(self, hidden_states: keras.KerasTensor, attention_mask: Optional[keras.KerasTensor] = None,
             training: Optional[bool] = None) -> keras.KerasTensor:
        batch_size, seq_len, _ = ops.shape(hidden_states)

        hidden_states_reshaped = ops.reshape(hidden_states, (batch_size, seq_len, self.num_heads, self.head_dim))
        cos_emb, sin_emb = self.rotary_embedding(hidden_states_reshaped)

        query = self._apply_rotary_pos_emb(hidden_states_reshaped, cos_emb, sin_emb)
        key = self._apply_rotary_pos_emb(hidden_states_reshaped, cos_emb, sin_emb)

        query = ops.reshape(query, (batch_size, seq_len, self.hidden_size))
        key = ops.reshape(key, (batch_size, seq_len, self.hidden_size))
        value = hidden_states

        final_attention_mask = attention_mask
        if not self.is_global:
            sliding_mask = self._create_sliding_window_mask(seq_len)
            if attention_mask is not None:
                expanded_padding_mask = ops.expand_dims(attention_mask, axis=1)
                final_attention_mask = ops.logical_and(expanded_padding_mask, sliding_mask)
            else:
                final_attention_mask = sliding_mask

        return self.mha(query=query, value=value, key=key, attention_mask=final_attention_mask, training=training)

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
        })
        return config


@keras.saving.register_keras_serializable()
class ModernBertTransformerLayer(keras.layers.Layer):
    """
    A single transformer layer for ModernBERT.

    This layer combines the self-attention mechanism (`ModernBertAttention`) and
    a feed-forward network (`GeGLUFFN`) with residual connections and
    pre-layer normalization, forming a complete transformer block.

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
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            intermediate_size: int,
            hidden_act: str,
            hidden_dropout_prob: float,
            attention_probs_dropout_prob: float,
            use_bias: bool,
            initializer_range: float,
            layer_norm_eps: float,
            is_global: bool,
            rope_theta_local: float,
            rope_theta_global: float,
            local_attention_window_size: int,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
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
            hidden_size=hidden_size, num_heads=num_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob, use_bias=use_bias,
            initializer_range=initializer_range, is_global=is_global,
            rope_theta_local=rope_theta_local, rope_theta_global=rope_theta_global,
            local_attention_window_size=local_attention_window_size, name="attention"
        )
        # REFACTOR: Use the new, modular GeGLUFFN layer
        self.ffn = GeGLUFFN(
            hidden_dim=intermediate_size,
            output_dim=hidden_size,
            activation=hidden_act,
            dropout_rate=hidden_dropout_prob,
            use_bias=use_bias,
            kernel_initializer=initializers.TruncatedNormal(stddev=initializer_range),
            name="ffn"
        )
        self.norm1 = keras.layers.LayerNormalization(epsilon=layer_norm_eps, center=use_bias, name="norm1")
        self.norm2 = keras.layers.LayerNormalization(epsilon=layer_norm_eps, center=use_bias, name="norm2")
        self.dropout1 = keras.layers.Dropout(hidden_dropout_prob)
        self.dropout2 = keras.layers.Dropout(hidden_dropout_prob)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build all sub-layers for robust serialization."""
        self.attention.build(input_shape)
        self.ffn.build(input_shape)
        self.norm1.build(input_shape)
        self.norm2.build(input_shape)
        self.dropout1.build(input_shape)
        self.dropout2.build(input_shape)
        super().build(input_shape)

    def call(self, hidden_states: keras.KerasTensor, attention_mask: Optional[keras.KerasTensor] = None,
             training: Optional[bool] = None) -> keras.KerasTensor:
        # Pre-normalization attention block
        residual = hidden_states
        hidden_states_norm = self.norm1(hidden_states, training=training)
        attention_output = self.attention(hidden_states_norm, attention_mask=attention_mask, training=training)
        hidden_states = residual + self.dropout1(attention_output, training=training)

        # Pre-normalization FFN block
        residual = hidden_states
        hidden_states_norm = self.norm2(hidden_states, training=training)
        ffn_output = self.ffn(hidden_states_norm, training=training)
        hidden_states = residual + self.dropout2(ffn_output, training=training)
        return hidden_states

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size, 'num_heads': self.num_heads, 'intermediate_size': self.intermediate_size,
            'hidden_act': self.hidden_act, 'hidden_dropout_prob': self.hidden_dropout_prob,
            'attention_probs_dropout_prob': self.attention_probs_dropout_prob, 'use_bias': self.use_bias,
            'initializer_range': self.initializer_range, 'layer_norm_eps': self.layer_norm_eps,
            'is_global': self.is_global, 'rope_theta_local': self.rope_theta_local,
            'rope_theta_global': self.rope_theta_global,
            'local_attention_window_size': self.local_attention_window_size,
        })
        return config