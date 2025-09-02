import keras
from keras import ops, initializers, layers
from typing import Optional, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.ffn.geglu_ffn import GeGLUFFN
from dl_techniques.layers.embedding.rotary_position_embedding import RotaryPositionEmbedding


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ModernBertEmbeddings(keras.layers.Layer):
    """
    ModernBERT embeddings layer without absolute position embeddings.

    This layer creates embeddings from input_ids and token_type_ids,
    then applies layer normalization and dropout. It follows modern Keras
    patterns for robust serialization.
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
        # Store all configuration
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use_bias = use_bias

        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.word_embeddings = layers.Embedding(
            self.vocab_size,
            self.hidden_size,
            embeddings_initializer=initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="word_embeddings"
        )
        self.token_type_embeddings = layers.Embedding(
            self.type_vocab_size,
            self.hidden_size,
            embeddings_initializer=initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="token_type_embeddings"
        )
        self.layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            center=self.use_bias,
            name="layer_norm"
        )
        self.dropout = layers.Dropout(
            self.hidden_dropout_prob,
            name="dropout"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the embedding layers."""
        # Build sub-layers in computational order
        self.word_embeddings.build(input_shape)
        self.token_type_embeddings.build(input_shape)

        embedding_shape = tuple(input_shape) + (self.hidden_size,)
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
        seq_length = ops.shape(input_ids)[1]
        if token_type_ids is None:
            token_type_ids = ops.zeros(
                (ops.shape(input_ids)[0], seq_length),
                dtype="int32"
            )

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


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ModernBertAttention(keras.layers.Layer):
    """
    Attention layer for ModernBERT with Rotary Position Embeddings (RoPE).

    This layer integrates Multi-Head Attention with RoPE and supports both global
    and local (sliding window) attention masks.
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
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
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
            kernel_initializer=initializers.TruncatedNormal(stddev=self.initializer_range),
            name="multi_head_attention"
        )
        self.rotary_embedding = RotaryPositionEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            rope_theta=self.rope_theta,
            name="rotary_embedding"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the attention and RoPE layers."""
        self.mha.build(query_shape=input_shape, value_shape=input_shape)

        # Build rotary embedding with its expected 4D shape
        dummy_4d_shape = tuple(input_shape[:-1]) + (self.num_heads, self.head_dim)
        self.rotary_embedding.build(dummy_4d_shape)

        super().build(input_shape)

    def _create_sliding_window_mask(self, seq_len: int) -> keras.KerasTensor:
        """Creates a boolean mask for sliding window local attention."""
        positions = ops.arange(seq_len, dtype="int32")
        mask = ops.abs(positions[:, None] - positions[None, :]) < self.local_attention_window_size
        return ops.cast(mask, "bool")

    def call(
        self,
        hidden_states: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass for attention with RoPE."""
        batch_size = ops.shape(hidden_states)[0]
        seq_len = ops.shape(hidden_states)[1]

        # Reshape to (batch, heads, seq, dim) for RoPE
        query_reshaped = ops.reshape(hidden_states, (batch_size, seq_len, self.num_heads, self.head_dim))
        query_reshaped = ops.transpose(query_reshaped, [0, 2, 1, 3])

        key_reshaped = ops.reshape(hidden_states, (batch_size, seq_len, self.num_heads, self.head_dim))
        key_reshaped = ops.transpose(key_reshaped, [0, 2, 1, 3])

        # Apply RoPE using the imported layer
        query_rotated = self.rotary_embedding(query_reshaped, training=training)
        key_rotated = self.rotary_embedding(key_reshaped, training=training)

        # Transpose back and reshape to 3D for MultiHeadAttention
        query_rotated = ops.transpose(query_rotated, [0, 2, 1, 3])
        key_rotated = ops.transpose(key_rotated, [0, 2, 1, 3])
        query = ops.reshape(query_rotated, (batch_size, seq_len, self.hidden_size))
        key = ops.reshape(key_rotated, (batch_size, seq_len, self.hidden_size))
        value = hidden_states

        # --- Definitive Masking Logic ---
        final_attention_mask = None
        if attention_mask is not None:
            # Ensure mask is boolean
            attention_mask = ops.cast(attention_mask, "bool")
            # Expand 2D padding mask to 4D for MHA broadcasting: (batch, seq) -> (batch, 1, 1, seq)
            if ops.ndim(attention_mask) == 2:
                 final_attention_mask = attention_mask[:, None, None, :]
            else: # Should already be broadcastable
                 final_attention_mask = attention_mask

        if not self.is_global:
            # Create a (seq, seq) sliding window mask.
            sliding_mask = self._create_sliding_window_mask(seq_len)

            if final_attention_mask is not None:
                # Combine the (batch, 1, 1, seq) padding mask with the (seq, seq) sliding mask.
                # Broadcasting results in a (batch, 1, seq, seq) combined mask.
                final_attention_mask = ops.logical_and(final_attention_mask, sliding_mask)
            else:
                # No padding mask exists, just use the sliding mask.
                # Expand to (1, 1, seq, seq) to broadcast over batch and heads.
                final_attention_mask = sliding_mask[None, None, :, :]

        return self.mha(
            query=query,
            value=value,
            key=key,
            attention_mask=final_attention_mask,
            training=training
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'attention_probs_dropout_prob': self.attention_probs_dropout_prob,
            'initializer_range': self.initializer_range,
            'rope_theta': self.rope_theta,
            'max_seq_len': self.max_seq_len,
            'local_attention_window_size': self.local_attention_window_size,
            'use_bias': self.use_bias,
            'is_global': self.is_global,
        })
        return config


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ModernBertEncoderLayer(keras.layers.Layer):
    """
    A complete ModernBERT Transformer encoder layer.

    This layer implements a Pre-Normalization (Pre-LN) architecture, combining
    the custom ModernBertAttention layer (with RoPE) and the reusable GeGLUFFN.
    """

    def __init__(self, is_global: bool, **kwargs: Any) -> None:
        self.all_config = kwargs.copy()
        self.is_global = is_global

        for key in list(kwargs.keys()):
            if key not in ['name', 'trainable', 'dtype']:
                kwargs.pop(key)
        super().__init__(**kwargs)

        self.attention_norm = layers.LayerNormalization(
            epsilon=self.all_config['layer_norm_eps'],
            center=self.all_config['use_bias'],
            name="attention_norm"
        )
        self.attention = ModernBertAttention(
            is_global=self.is_global,
            rope_theta=(
                self.all_config['rope_theta_global']
                if self.is_global
                else self.all_config['rope_theta_local']
            ),
            hidden_size=self.all_config['hidden_size'],
            num_heads=self.all_config['num_heads'],
            attention_probs_dropout_prob=self.all_config['attention_probs_dropout_prob'],
            initializer_range=self.all_config['initializer_range'],
            max_seq_len=self.all_config['max_seq_len'],
            local_attention_window_size=self.all_config['local_attention_window_size'],
            use_bias=self.all_config['use_bias'],
            name="attention"
        )
        self.ffn_norm = layers.LayerNormalization(
            epsilon=self.all_config['layer_norm_eps'],
            center=self.all_config['use_bias'],
            name="ffn_norm"
        )
        self.ffn = GeGLUFFN(
            hidden_dim=self.all_config['intermediate_size'],
            output_dim=self.all_config['hidden_size'],
            activation=self.all_config['hidden_act'],
            dropout_rate=self.all_config['hidden_dropout_prob'],
            use_bias=self.all_config['use_bias'],
            kernel_initializer=initializers.TruncatedNormal(stddev=self.all_config['initializer_range']),
            name="ffn"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all sub-layers."""
        self.attention_norm.build(input_shape)
        self.attention.build(input_shape)
        self.ffn_norm.build(input_shape)
        self.ffn.build(input_shape)
        super().build(input_shape)

    def call(
            self,
            hidden_states: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass implementing the Pre-LN Transformer architecture."""
        residual = hidden_states
        x = self.attention_norm(hidden_states, training=training)
        x = self.attention(x, attention_mask=attention_mask, training=training)
        attention_output = x + residual

        residual = attention_output
        x = self.ffn_norm(attention_output, training=training)
        x = self.ffn(x, training=training)
        layer_output = x + residual

        return layer_output

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update(self.all_config)
        config['is_global'] = self.is_global
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ModernBertEncoderLayer':
        """Creates layer from config, handling the kwargs-based init."""
        is_global = config.pop('is_global')
        return cls(is_global=is_global, **config)