import keras
from keras import ops, layers, initializers, regularizers
from typing import Optional, Union, Tuple, Dict, Any, Literal, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------
from ..utils.logger import logger
from .embedding import create_embedding_layer
from .norms import create_normalization_layer
from .transformer import TransformerLayer
from ..utils.masking import create_causal_mask

# ---------------------------------------------------------------------
# Type definitions for enhanced type safety
# ---------------------------------------------------------------------

EmbeddingType = Literal['learned', 'shared', 'factorized']
PositionalType = Literal['learned', 'sincos']  # RoPE types are not typically used in decoders this way
AttentionType = Literal['multi_head_attention', 'window_attention', 'group_query_attention', 'differential_attention']
NormalizationType = Literal['layer_norm', 'rms_norm', 'batch_norm', 'band_rms', 'adaptive_band_rms', 'dynamic_tanh']
NormalizationPosition = Literal['pre', 'post']
FFNType = Literal['mlp', 'swiglu', 'differential', 'glu', 'geglu', 'residual', 'swin_mlp']


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TextDecoder(keras.layers.Layer):
    """
    General-purpose configurable text decoder with causal attention.

    This layer implements a transformer-based text decoder with configurable
    components (embeddings, attention, FFN, normalization) and automatically
    handles causal masking for autoregressive text generation.

    Args:
        vocab_size: Size of the vocabulary.
        embed_dim: Embedding dimension for tokens and hidden states.
        depth: Number of transformer decoder layers.
        num_heads: Number of attention heads.
        mlp_ratio: Expansion ratio for the feed-forward network's intermediate dimension.
        max_seq_len: Maximum sequence length for positional embeddings.
        embedding_type: Strategy for word embeddings ('learned', 'shared', 'factorized').
        positional_type: Strategy for positional embeddings ('learned', 'sincos').
        attention_type: Attention mechanism for transformer layers.
        normalization_type: Normalization layer type.
        normalization_position: Position of normalization ('pre' or 'post').
        ffn_type: Feed-forward network architecture.
        use_token_type_embedding: Whether to include token type embeddings.
        type_vocab_size: Vocabulary size for token types.
        dropout: Dropout rate for FFN.
        attention_dropout: Dropout rate for attention weights.
        embed_dropout: Dropout rate after embeddings.
        stochastic_depth_rate: Rate for stochastic depth regularization.
        **kwargs: Additional keyword arguments for base Layer class and sub-components.

    Input shape:
        - `input_ids`: 2D tensor of shape `(batch_size, sequence_length)`
        - `attention_mask` (optional): 2D tensor of shape `(batch_size, sequence_length)` for padding.

    Output shape:
        3D tensor of shape `(batch_size, sequence_length, embed_dim)`
    """

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int,
            depth: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            max_seq_len: int = 512,
            embedding_type: EmbeddingType = 'learned',
            positional_type: PositionalType = 'learned',
            attention_type: AttentionType = 'multi_head_attention',
            normalization_type: NormalizationType = 'layer_norm',
            normalization_position: NormalizationPosition = 'post',
            ffn_type: FFNType = 'mlp',
            use_token_type_embedding: bool = False,
            type_vocab_size: int = 2,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            embed_dropout: float = 0.1,
            stochastic_depth_rate: float = 0.0,
            activation: Union[str, Callable] = 'gelu',
            use_bias: bool = True,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            initializer_range: float = 0.02,
            layer_norm_eps: float = 1e-12,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Parameter validation
        if vocab_size <= 0 or embed_dim <= 0 or depth <= 0 or num_heads <= 0 or max_seq_len <= 0:
            raise ValueError("All dimension and size parameters must be positive.")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        # Store configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.max_seq_len = max_seq_len
        self.embedding_type = embedding_type
        self.positional_type = positional_type
        self.attention_type = attention_type
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.ffn_type = ffn_type
        self.use_token_type_embedding = use_token_type_embedding
        self.type_vocab_size = type_vocab_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.embed_dropout = embed_dropout
        self.stochastic_depth_rate = stochastic_depth_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        # Computed properties
        self.intermediate_size = int(embed_dim * mlp_ratio)

        # --- Sub-layer Creation ---
        self._create_word_embeddings()

        pos_embed_type = 'positional_learned' if self.positional_type == 'learned' else 'continuous_sincos'
        pos_embed_kwargs = {'dim': embed_dim, 'max_seq_len': max_seq_len}
        if pos_embed_type == 'continuous_sincos':
            pos_embed_kwargs['ndim'] = 1 # Required for this type

        self.positional_embeddings = create_embedding_layer(
            pos_embed_type,
            name="positional_embeddings",
            **pos_embed_kwargs
        )

        self.token_type_embeddings = None
        if self.use_token_type_embedding:
            self.token_type_embeddings = layers.Embedding(
                input_dim=self.type_vocab_size,
                output_dim=self.embed_dim,
                embeddings_initializer=initializers.TruncatedNormal(stddev=self.initializer_range),
                name="token_type_embeddings"
            )

        self.embed_norm = create_normalization_layer(
            self.normalization_type, epsilon=self.layer_norm_eps, name="embed_norm"
        )
        self.embed_dropout_layer = layers.Dropout(self.embed_dropout)

        self.decoder_layers = []
        for i in range(self.depth):
            layer_drop_rate = self.stochastic_depth_rate * i / max(1, self.depth - 1)
            layer = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                attention_type=self.attention_type,
                normalization_type=self.normalization_type,
                normalization_position=self.normalization_position,
                ffn_type=self.ffn_type,
                dropout_rate=self.dropout,
                attention_dropout_rate=self.attention_dropout,
                use_stochastic_depth=self.stochastic_depth_rate > 0.0,
                stochastic_depth_rate=layer_drop_rate,
                name=f'decoder_layer_{i}'
            )
            self.decoder_layers.append(layer)

        self.final_norm = create_normalization_layer(
            self.normalization_type, epsilon=self.layer_norm_eps, name='final_norm'
        )

    def _create_word_embeddings(self) -> None:
        """Create word embedding layers based on the specified strategy."""
        self.word_embeddings = None
        self.factorized_embed_layer = None
        self.embed_projection_layer = None

        if self.embedding_type in ['learned', 'shared']:
            self.word_embeddings = layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embed_dim,
                embeddings_initializer=initializers.TruncatedNormal(stddev=self.initializer_range),
                name="word_embeddings"
            )
        elif self.embedding_type == 'factorized':
            factorized_dim = min(self.embed_dim, 128)
            self.factorized_embed_layer = layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=factorized_dim,
                embeddings_initializer=initializers.TruncatedNormal(stddev=self.initializer_range),
                name='factorized_embed'
            )
            self.embed_projection_layer = layers.Dense(
                self.embed_dim,
                use_bias=False,
                kernel_initializer=initializers.TruncatedNormal(stddev=self.initializer_range),
                name='embed_projection'
            )

    def build(self, input_shape):
        if self.word_embeddings:
            self.word_embeddings.build(input_shape)
        if self.factorized_embed_layer:
            self.factorized_embed_layer.build(input_shape)
            factorized_shape = self.factorized_embed_layer.compute_output_shape(input_shape)
            self.embed_projection_layer.build(factorized_shape)

        embedding_output_shape = (*input_shape, self.embed_dim)
        if self.positional_type == 'learned':
             self.positional_embeddings.build(embedding_output_shape)
        else: # sincos
            # It expects coordinate shape (batch, seq_len, ndim)
            self.positional_embeddings.build((*input_shape, 1))

        if self.use_token_type_embedding:
            self.token_type_embeddings.build(input_shape)

        self.embed_norm.build(embedding_output_shape)

        for layer in self.decoder_layers:
            layer.build(embedding_output_shape)

        self.final_norm.build(embedding_output_shape)
        super().build(input_shape)

    def call(self, input_ids, attention_mask=None, token_type_ids=None, training=None):
        seq_len = ops.shape(input_ids)[1]
        batch_size = ops.shape(input_ids)[0]

        # 1. Create Embeddings
        if self.embedding_type == 'factorized':
            x = self.factorized_embed_layer(input_ids)
            x = self.embed_projection_layer(x)
        else:
            x = self.word_embeddings(input_ids)

        if self.positional_type == 'learned':
            x = self.positional_embeddings(x)
        else:  # sincos
            positions = ops.arange(start=0, stop=seq_len, dtype="float32")
            positions = ops.expand_dims(positions, axis=-1)  # (seq_len, 1)
            positions = ops.expand_dims(positions, axis=0)    # (1, seq_len, 1)
            positions = ops.broadcast_to(positions, (batch_size, seq_len, 1))
            x += self.positional_embeddings(positions)

        if self.use_token_type_embedding:
            if token_type_ids is None:
                token_type_ids = ops.zeros_like(input_ids)
            x += self.token_type_embeddings(token_type_ids)

        x = self.embed_norm(x)
        x = self.embed_dropout_layer(x, training=training)

        # 2. Create Causal + Padding Mask
        causal_mask = create_causal_mask(seq_len)  # Shape: (seq_len, seq_len)
        causal_mask = ops.expand_dims(causal_mask, axis=0) # Shape: (1, seq_len, seq_len)

        if attention_mask is not None:
            padding_mask = ops.expand_dims(attention_mask == 0, axis=1)  # True where padded -> (batch, 1, seq_len)
            combined_mask = ops.logical_or(padding_mask, causal_mask)
        else:
            combined_mask = ops.broadcast_to(causal_mask, (batch_size, seq_len, seq_len))

        # 3. Apply Transformer Layers
        for layer in self.decoder_layers:
            x = layer(x, attention_mask=combined_mask, training=training)

        # 4. Final Normalization
        x = self.final_norm(x)
        return x

    def compute_output_shape(self, input_shape):
        return (*input_shape, self.embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'max_seq_len': self.max_seq_len,
            'embedding_type': self.embedding_type,
            'positional_type': self.positional_type,
            'attention_type': self.attention_type,
            'normalization_type': self.normalization_type,
            'normalization_position': self.normalization_position,
            'ffn_type': self.ffn_type,
            'use_token_type_embedding': self.use_token_type_embedding,
            'type_vocab_size': self.type_vocab_size,
            'dropout': self.dropout,
            'attention_dropout': self.attention_dropout,
            'embed_dropout': self.embed_dropout,
            'stochastic_depth_rate': self.stochastic_depth_rate,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'initializer_range': self.initializer_range,
            'layer_norm_eps': self.layer_norm_eps,
        })
        return config