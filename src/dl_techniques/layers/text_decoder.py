import keras
from keras import ops, layers, initializers
from typing import Optional, Dict, Any, Literal, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------
from ..utils.masking import create_causal_mask
from .embedding import create_embedding_layer
from .norms import create_normalization_layer, NormalizationType
from .transformer import TransformerLayer, AttentionType, FFNType, NormalizationPosition

# ---------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------

EmbeddingType = Literal['learned', 'shared', 'factorized']
PositionalType = Literal['learned', 'sincos']


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TextDecoder(keras.layers.Layer):
    """
    General-purpose configurable text decoder built upon a stack of TransformerLayers.

    This layer orchestrates token and positional embeddings, a stack of configurable
    transformer decoder blocks, and appropriate masking for autoregressive text generation.
    It serves as a high-level interface for building decoder-only models, exposing key
    architectural choices of its underlying components.

    **Intent**: Provide a flexible, configurable transformer decoder that can be easily
    adapted for different language modeling tasks by adjusting embedding strategies,
    attention mechanisms, normalization types, and architectural depth.

    **Architecture**:
    ```
    Input IDs (batch_size, seq_len)
           ↓
    Word Embeddings → Positional Embeddings
           ↓                    ↓
           └────── ADD ──────────┘
                   ↓
           Embedding Normalization
                   ↓
           Embedding Dropout
                   ↓
           Causal + Padding Mask
                   ↓
    ┌─────────────────────────────┐
    │   TransformerLayer 0        │
    │   TransformerLayer 1        │
    │   ...                       │
    │   TransformerLayer (depth-1)│
    └─────────────────────────────┘
                   ↓
           Final Normalization
                   ↓
    Output (batch_size, seq_len, embed_dim)
    ```

    **Embedding Strategies**:
    - **learned**: Standard embedding lookup table
    - **shared**: Weight sharing between input/output embeddings
    - **factorized**: Low-rank factorized embeddings for memory efficiency

    **Positional Encoding**:
    - **learned**: Trainable positional embeddings
    - **sincos**: Fixed sinusoidal positional encodings

    Args:
        vocab_size: Size of the vocabulary.
        embed_dim: Embedding dimension for tokens and hidden states.
        depth: Number of transformer decoder layers.
        num_heads: Number of attention heads.
        max_seq_len: Maximum sequence length for positional embeddings.
        embedding_type: Strategy for word embeddings ('learned', 'shared', 'factorized').
        positional_type: Strategy for positional embeddings ('learned', 'sincos').
        attention_type: Attention mechanism for transformer layers.
        normalization_type: Normalization layer type for all layers.
        normalization_position: Position of normalization ('pre' or 'post').
        ffn_type: Feed-forward network architecture for transformer layers.
        stochastic_depth_rate: Rate for stochastic depth regularization.
        dropout_rate: Dropout rate for FFN and final embedding dropout.
        attention_dropout: Dropout rate for attention weights.
        initializer_range: Standard deviation for the TruncatedNormal initializer.
        layer_norm_eps: Epsilon for normalization layers.
        **kwargs: Additional keyword arguments for base Layer class.

    Input shape:
        - `input_ids`: 2D tensor of shape `(batch_size, sequence_length)` with integer token IDs
        - `attention_mask` (optional): 2D tensor of shape `(batch_size, sequence_length)`
          where 1 indicates valid tokens and 0 indicates padding

    Output shape:
        3D tensor of shape `(batch_size, sequence_length, embed_dim)` containing
        contextualized token representations

    Raises:
        ValueError: If embed_dim is not divisible by num_heads, or if any dimension
            parameters are not positive integers.

    Example:
        ```python
        # Basic GPT-style decoder
        decoder = TextDecoder(
            vocab_size=50000,
            embed_dim=768,
            depth=12,
            num_heads=12,
            max_seq_len=2048
        )

        # Modern architecture with RMSNorm and SwiGLU
        decoder = TextDecoder(
            vocab_size=32000,
            embed_dim=512,
            depth=6,
            num_heads=8,
            max_seq_len=1024,
            positional_type='sincos',
            normalization_type='rms_norm',
            normalization_position='pre',
            ffn_type='swiglu'
        )

        # Memory-efficient with factorized embeddings
        decoder = TextDecoder(
            vocab_size=100000,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            embedding_type='factorized',
            stochastic_depth_rate=0.1
        )
        ```

    Note:
        This layer implements causal masking automatically for autoregressive generation.
        The attention mask parameter allows for handling variable-length sequences with
        padding, which is combined with the causal mask during forward pass.
    """

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int,
            depth: int,
            num_heads: int,
            max_seq_len: int = 512,
            embedding_type: EmbeddingType = 'learned',
            positional_type: PositionalType = 'learned',
            attention_type: AttentionType = 'multi_head_attention',
            normalization_type: NormalizationType = 'layer_norm',
            normalization_position: NormalizationPosition = 'post',
            ffn_type: FFNType = 'mlp',
            stochastic_depth_rate: float = 0.0,
            dropout_rate: float = 0.1,
            attention_dropout: float = 0.1,
            initializer_range: float = 0.02,
            layer_norm_eps: float = 1e-12,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # --- Parameter Validation ---
        if not all(isinstance(p, int) and p > 0 for p in [vocab_size, embed_dim, depth, num_heads, max_seq_len]):
            raise ValueError("All dimension and size parameters must be positive integers.")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout must be between 0.0 and 1.0, got {dropout_rate}")
        if not 0.0 <= attention_dropout <= 1.0:
            raise ValueError(f"attention_dropout must be between 0.0 and 1.0, got {attention_dropout}")
        if not 0.0 <= stochastic_depth_rate <= 1.0:
            raise ValueError(f"stochastic_depth_rate must be between 0.0 and 1.0, got {stochastic_depth_rate}")

        # --- Store Configuration ---
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.embedding_type = embedding_type
        self.positional_type = positional_type
        self.attention_type = attention_type
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.ffn_type = ffn_type
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        # --- Create Sub-layers in __init__ ---
        self._create_word_embeddings()
        self._create_positional_embeddings()

        # Embedding processing layers
        self.embed_dropout_layer = layers.Dropout(rate=self.dropout_rate, name="embed_dropout")
        self.embed_norm = create_normalization_layer(
            self.normalization_type, epsilon=self.layer_norm_eps, name="embed_norm"
        )

        # Create transformer decoder layers
        self.decoder_layers = []
        for i in range(self.depth):
            # Linearly increase drop rate per layer
            layer_drop_rate = self.stochastic_depth_rate * i / max(1, self.depth - 1)
            layer = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=int(self.embed_dim * 4),  # Standard 4x expansion
                attention_type=self.attention_type,
                normalization_type=self.normalization_type,
                normalization_position=self.normalization_position,
                ffn_type=self.ffn_type,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout,
                use_stochastic_depth=self.stochastic_depth_rate > 0.0,
                stochastic_depth_rate=layer_drop_rate,
                name=f'decoder_layer_{i}'
            )
            self.decoder_layers.append(layer)

        # Final normalization layer
        self.final_norm = create_normalization_layer(
            self.normalization_type, epsilon=self.layer_norm_eps, name='final_norm'
        )

    def _create_word_embeddings(self) -> None:
        """Create word embedding layers based on the specified strategy."""
        initializer = initializers.TruncatedNormal(stddev=self.initializer_range)

        if self.embedding_type in ['learned', 'shared']:
            self.word_embeddings = layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embed_dim,
                embeddings_initializer=initializer,
                name="word_embeddings"
            )
        elif self.embedding_type == 'factorized':
            # Use factorized embeddings for memory efficiency
            factorized_dim = min(self.embed_dim, 128)
            self.factorized_embed_layer = layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=factorized_dim,
                embeddings_initializer=initializer,
                name='factorized_embed'
            )
            self.embed_projection_layer = layers.Dense(
                units=self.embed_dim,
                use_bias=False,
                kernel_initializer=initializer,
                name='embed_projection'
            )

    def _create_positional_embeddings(self) -> None:
        """Create positional embedding layer based on the specified strategy."""
        if self.positional_type == 'learned':
            self.positional_embeddings = layers.Embedding(
                input_dim=self.max_seq_len,
                output_dim=self.embed_dim,
                embeddings_initializer=initializers.TruncatedNormal(stddev=self.initializer_range),
                name="positional_embeddings"
            )
        elif self.positional_type == 'sincos':
            self.positional_embeddings = create_embedding_layer(
                'continuous_sincos',
                dim=self.embed_dim,
                max_seq_len=self.max_seq_len,
                ndim=1,
                name="positional_embeddings"
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        This ensures all weight variables exist before weight restoration during loading.
        """
        # Build word embedding layers
        if hasattr(self, 'word_embeddings'):
            self.word_embeddings.build(input_shape)
        elif hasattr(self, 'factorized_embed_layer'):
            self.factorized_embed_layer.build(input_shape)
            # Compute shape after factorized embedding
            factorized_output_shape = self.factorized_embed_layer.compute_output_shape(input_shape)
            self.embed_projection_layer.build(factorized_output_shape)

        # Build positional embeddings with appropriate input shapes
        if self.positional_type == 'learned':
            # Learned embeddings take position indices as input
            position_input_shape = (None,)  # 1D sequence of positions
            self.positional_embeddings.build(position_input_shape)
        elif self.positional_type == 'sincos':
            # Continuous sincos embeddings take coordinates as input
            sincos_input_shape = (input_shape[0], input_shape[1], 1)  # (batch, seq, coord_dim)
            self.positional_embeddings.build(sincos_input_shape)

        # Compute embedding output shape for subsequent layers
        embedding_output_shape = (*input_shape, self.embed_dim)

        # Build embedding processing layers
        self.embed_norm.build(embedding_output_shape)
        self.embed_dropout_layer.build(embedding_output_shape)

        # Build all transformer decoder layers
        for layer in self.decoder_layers:
            layer.build(embedding_output_shape)

        # Build final normalization
        self.final_norm.build(embedding_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            input_ids: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the text decoder.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional padding mask of shape (batch_size, seq_len)
            training: Whether layer is in training mode

        Returns:
            Contextualized token representations of shape (batch_size, seq_len, embed_dim)
        """
        seq_len = ops.shape(input_ids)[1]
        batch_size = ops.shape(input_ids)[0]

        # 1. Word Embeddings
        if self.embedding_type == 'factorized':
            x = self.factorized_embed_layer(input_ids)
            x = self.embed_projection_layer(x)
        else:
            x = self.word_embeddings(input_ids)

        # 2. Positional Embeddings
        positions = ops.arange(start=0, stop=seq_len)
        if self.positional_type == 'learned':
            pos_embed = self.positional_embeddings(positions)
            x = ops.add(x, pos_embed)
        elif self.positional_type == 'sincos':
            # Reshape positions to (batch, seq_len, 1) for sincos layer
            pos_coords = ops.cast(positions, "float32")
            pos_coords = ops.expand_dims(pos_coords, axis=-1)
            pos_coords = ops.expand_dims(pos_coords, axis=0)
            pos_coords = ops.broadcast_to(pos_coords, (batch_size, seq_len, 1))
            pos_embed = self.positional_embeddings(pos_coords)
            x = ops.add(x, pos_embed)

        # 3. Embedding normalization and dropout
        x = self.embed_norm(x)
        x = self.embed_dropout_layer(x, training=training)

        # 4. Create Causal + Padding Mask
        causal_mask = create_causal_mask(seq_len, dtype=bool)
        causal_mask = ops.expand_dims(causal_mask, axis=0)  # Add batch dimension

        if attention_mask is not None:
            # Create padding mask (True where padded)
            padding_mask = ops.equal(attention_mask, 0)
            padding_mask = ops.expand_dims(padding_mask, axis=1)  # Add seq dimension for broadcasting
            # Combine causal and padding masks
            combined_mask = ops.logical_or(padding_mask, causal_mask)
        else:
            # Use only causal mask
            combined_mask = ops.broadcast_to(causal_mask, (batch_size, seq_len, seq_len))

        # 5. Apply Transformer Layers
        for layer in self.decoder_layers:
            x = layer(x, attention_mask=combined_mask, training=training)

        # 6. Final Normalization
        x = self.final_norm(x)
        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape given input shape."""
        return (*input_shape, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'max_seq_len': self.max_seq_len,
            'embedding_type': self.embedding_type,
            'positional_type': self.positional_type,
            'attention_type': self.attention_type,
            'normalization_type': self.normalization_type,
            'normalization_position': self.normalization_position,
            'ffn_type': self.ffn_type,
            'stochastic_depth_rate': self.stochastic_depth_rate,
            'dropout_rate': self.dropout_rate,
            'attention_dropout': self.attention_dropout,
            'initializer_range': self.initializer_range,
            'layer_norm_eps': self.layer_norm_eps,
        })
        return config

# ---------------------------------------------------------------------
