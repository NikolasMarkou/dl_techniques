"""
A configurable, Transformer-based text decoder stack.

This layer serves as a high-level component for building decoder-only
autoregressive language models, encapsulating the core logic of token
embedding, positional encoding, and a stack of causal self-attention blocks.
It is designed to be highly configurable, allowing for the construction of
various modern transformer architectures by composing different underlying
mechanisms for embeddings, attention, and normalization.

Architecture and Core Concepts:

The fundamental design follows the decoder side of the original Transformer
architecture, which has become the de facto standard for large language
models (LLMs). The primary purpose of this architecture is to model the
probability distribution of a sequence of tokens, P(x_i | x_1, ..., x_{i-1}),
making it inherently suited for text generation.

The core operational principles are:

1.  **Input Representation:** The process begins by converting a sequence of
    integer token IDs into a dense vector space representation. This is
    achieved by summing a token embedding (which captures semantic meaning)
    and a positional embedding (which injects information about the token's
    position in the sequence, compensating for the architecture's lack of
    inherent sequential awareness).

2.  **Causal Self-Attention:** The sequence of embeddings is then processed by
    a stack of identical transformer layers. The central mechanism in each
    layer is masked multi-head self-attention. The "causal" or "look-ahead"
    mask is the defining feature of a decoder. It ensures that the
    representation for a token at position `i` can only be influenced by
    tokens at positions less than or equal to `i`. This restriction is
    critical for maintaining the autoregressive property, preventing the model
    from "cheating" by looking at future tokens during training.

3.  **Feed-Forward Networks:** Each attention sub-layer is followed by a
    pointwise feed-forward network (FFN), which introduces additional
    non-linearity and capacity, allowing the model to learn more complex
    transformations of the token representations.

4.  **Layer Normalization and Residuals:** The entire stack is stabilized by
    residual connections and layer normalization, which are applied around
    each sub-layer (attention and FFN). This layer supports both "pre-norm"
    and "post-norm" configurations, a key architectural choice affecting
    training dynamics and stability.

This layer's configurability allows for the exploration of architectural
variants popularized by recent research, such as substituting standard Layer
Normalization with RMSNorm, or replacing the standard FFN with more advanced
variants like SwiGLU.

Mathematical Foundation:

The attention mechanism calculates a weighted sum of value vectors, where the
weights are determined by the similarity between query and key vectors. For a
sequence `X`, the output `Z` is computed as:
`Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k) + M) V`

Here, `Q`, `K`, and `V` are linear projections of the input `X`. The term `d_k`
is the dimension of the key vectors, used for scaling. The crucial component
for a decoder is the mask `M`, a matrix where `M_ij = -inf` for `j > i` and
`0` otherwise. This ensures that the softmax output for connections to future
tokens is zero, enforcing causality.

References:

The architecture implemented here is a configurable version of the decoder
stack first proposed in:
-   Vaswani, A., et al. (2017). "Attention Is All You Need." This paper
    introduced the original Transformer architecture.

The decoder-only variant was popularized by the GPT series of models, which
demonstrated its effectiveness for generative pre-training:
-   Radford, A., et al. (2018). "Improving Language Understanding by
    Generative Pre-Training."

Modern components available through this layer's configuration options are
based on subsequent research, such as:
-   Zhang, B., & Sennrich, R. (2019). "Root Mean Square Layer Normalization."
-   Shazeer, N. (2020). "GLU Variants Improve Transformer."

"""

import keras
from keras import ops, layers, initializers
from typing import Optional, Dict, Any, Literal, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.masking import create_mask, MaskConfig, combine_masks
from ..embedding import create_embedding_layer
from ..norms import create_normalization_layer, NormalizationType
from .transformer import TransformerLayer, AttentionType, FFNType, NormalizationPositionType

# ---------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------

PositionalType = Literal['learned', 'sincos']
EmbeddingType = Literal['learned', 'shared', 'factorized']

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TextDecoder(keras.layers.Layer):
    """
    General-purpose configurable text decoder built on a TransformerLayer stack.

    Orchestrates token and positional embeddings, causal masking, a stack of
    configurable transformer decoder blocks, and final normalization for
    autoregressive text generation. Causal masking is applied automatically;
    an optional padding mask can be combined with it.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────┐
        │  Input IDs (B, seq_len)                  │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Word Embedding + Positional Embedding   │
        │  ─► Embed Norm ─► Embed Dropout          │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Causal + Padding Mask                   │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  TransformerLayer x depth                │
        │  (causal self-attention + FFN)           │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Final Normalization                     │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Output (B, seq_len, embed_dim)          │
        └──────────────────────────────────────────┘

    :param vocab_size: Vocabulary size.
    :type vocab_size: int
    :param embed_dim: Token embedding / hidden dimension.
    :type embed_dim: int
    :param depth: Number of decoder layers.
    :type depth: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param max_seq_len: Maximum sequence length. Default: 512.
    :type max_seq_len: int
    :param embedding_type: Word embedding strategy. Default: ``'learned'``.
    :type embedding_type: EmbeddingType
    :param positional_type: Positional encoding strategy. Default: ``'learned'``.
    :type positional_type: PositionalType
    :param attention_type: Attention mechanism type. Default: ``'multi_head'``.
    :type attention_type: AttentionType
    :param normalization_type: Normalization type. Default: ``'layer_norm'``.
    :type normalization_type: NormalizationType
    :param normalization_position: ``'pre'`` or ``'post'``. Default: ``'post'``.
    :type normalization_position: NormalizationPositionType
    :param ffn_type: FFN architecture type. Default: ``'mlp'``.
    :type ffn_type: FFNType
    :param stochastic_depth_rate: Drop-path rate. Default: 0.0.
    :type stochastic_depth_rate: float
    :param dropout_rate: Dropout rate. Default: 0.1.
    :type dropout_rate: float
    :param attention_dropout_rate: Attention dropout. Default: 0.1.
    :type attention_dropout_rate: float
    :param initializer_range: Std-dev for TruncatedNormal. Default: 0.02.
    :type initializer_range: float
    :param layer_norm_eps: Normalization epsilon. Default: 1e-12.
    :type layer_norm_eps: float
    :param kwargs: Additional keyword arguments for the base Layer.
    :type kwargs: Any

    :raises ValueError: If dimension parameters are invalid.
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
            attention_type: AttentionType = 'multi_head',
            normalization_type: NormalizationType = 'layer_norm',
            normalization_position: NormalizationPositionType = 'post',
            ffn_type: FFNType = 'mlp',
            stochastic_depth_rate: float = 0.0,
            dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.1,
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
        if not 0.0 <= attention_dropout_rate <= 1.0:
            raise ValueError(f"attention_dropout must be between 0.0 and 1.0, got {attention_dropout_rate}")
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
        self.attention_dropout_rate = attention_dropout_rate
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
                attention_dropout_rate=self.attention_dropout_rate,
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
        """Forward pass through the text decoder.

        :param input_ids: Token IDs ``(B, seq_len)``.
        :type input_ids: keras.KerasTensor
        :param attention_mask: Optional padding mask ``(B, seq_len)``.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Contextualized representations ``(B, seq_len, embed_dim)``.
        :rtype: keras.KerasTensor
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

        # 4. Create Attention Mask using the Masking Factory
        # Create causal mask
        causal_mask = create_mask('causal', seq_len=seq_len, dtype='bool')
        # Add batch dimension and broadcast
        causal_mask = ops.expand_dims(causal_mask, axis=0)
        causal_mask = ops.broadcast_to(causal_mask, (batch_size, seq_len, seq_len))

        if attention_mask is not None:
            # Convert attention_mask from 1/0 format to boolean padding mask
            # True indicates padding (positions to mask)
            padding_mask_1d = ops.equal(attention_mask, 0)  # Shape: (batch, seq_len)

            # Create padding attention mask using the factory
            # The factory expects the padding mask in extra_params
            padding_config = MaskConfig(
                mask_type='padding',
                dtype='bool',
                extra_params={'padding_mask': padding_mask_1d}
            )
            padding_mask_3d = create_mask(config=padding_config)  # Shape: (batch, seq_len, seq_len)

            # Combine causal and padding masks using the factory's combine function
            # Use 'or' to mask positions that are either future tokens OR padding
            combined_mask = combine_masks(causal_mask, padding_mask_3d, combination='or')
        else:
            # Use only causal mask
            combined_mask = causal_mask

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
            'attention_dropout_rate': self.attention_dropout_rate,
            'initializer_range': self.initializer_range,
            'layer_norm_eps': self.layer_norm_eps,
        })
        return config

# ---------------------------------------------------------------------
