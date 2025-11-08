"""
A highly configurable, Transformer-based text encoder.

This layer provides a versatile and modular framework for constructing a wide
range of Transformer encoder architectures. It is designed to create deep,
bidirectional representations of text, making it suitable for natural
language understanding tasks like text classification, named entity
recognition, and question answering. Its primary design principle is
configurability, allowing researchers and engineers to compose different
architectural components to build models ranging from classic BERT-style
architectures to more modern variants.

Architecture and Core Concepts:

The layer follows the fundamental architecture of the encoder side of the
original Transformer model. The core operation is to process an entire
sequence of text at once, allowing every token to attend to every other token.
This "bidirectional" self-attention mechanism is what enables the model to
build a contextually rich understanding of each token based on its complete
surrounding text.

The architecture is composed of three main stages:

1.  **Input Embedding Stage:** This stage converts input token IDs into a
    continuous representation that incorporates semantic, positional, and (if
    applicable) segment information. The key innovation here is the
    flexibility in positional encoding. Beyond traditional learned or fixed
    sinusoidal embeddings, it supports modern techniques like Rotary Position
    Embeddings (RoPE), which inject relative positional information more
    effectively.

2.  **Transformer Layer Stack:** The core of the encoder is a stack of
    identical Transformer layers. Each layer refines the token
    representations through two main sub-layers: a self-attention mechanism
    and a position-wise feed-forward network (FFN). This class allows for
    extensive configuration of these layers, including the choice between
    pre-layer and post-layer normalization (a critical factor for training
    stability), different normalization techniques (e.g., LayerNorm vs.
    RMSNorm), and various FFN architectures (e.g., standard MLP vs. gated
    variants like SwiGLU).

3.  **Output Pooling Stage:** For tasks that require a single, fixed-size
    vector representation of the entire input sequence, the layer can apply a
    pooling strategy to the final token representations. This might involve
    using the representation of a special `[CLS]` token, or applying mean or
    max pooling across the sequence.

By exposing these architectural choices as configuration parameters, this
class serves as a factory for a family of encoder models, promoting rapid
experimentation and adaptation to new research findings.

Mathematical Foundation:

The cornerstone of the encoder is the self-attention mechanism, calculated as:
`Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k)) V`

Unlike a decoder, the attention mechanism here is not causally masked,
meaning the softmax is computed over the entire sequence. This allows
information to flow bidirectionally.

A key mathematical concept supported is Rotary Position Embeddings (RoPE).
Instead of adding positional vectors to the token embeddings, RoPE rotates
the query and key vectors based on their absolute position. The rotation is
designed such that the dot product between a query at position `m` and a key
at position `n` inherently depends only on their relative position `m-n`.
This elegantly injects relative positional awareness directly into the
attention mechanism, often leading to improved performance on long sequences.

References:

The design of this layer is based on a rich history of research in natural
language processing, primarily originating from:
-   Vaswani, A., et al. (2017). "Attention Is All You Need." This paper
    introduced the original Transformer architecture.
-   Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional
    Transformers for Language Understanding." This work established the
    Transformer encoder as a dominant paradigm for NLU tasks.

The modern, configurable components are based on subsequent innovations:
-   Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary
    Position Embedding." This paper introduced RoPE.
-   Zhang, B., & Sennrich, R. (2019). "Root Mean Square Layer
    Normalization."
-   Shazeer, N. (2020). "GLU Variants Improve Transformer," which popularized
    architectures like SwiGLU.

"""

import keras
from keras import ops, layers, initializers, regularizers
from typing import Optional, Union, Tuple, Dict, Any, Literal, Callable, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from ..embedding import create_embedding_layer
from ..sequence_pooling import SequencePooling, PoolingStrategy
from ..norms import create_normalization_layer, NormalizationType
from .transformer import TransformerLayer, AttentionType, NormalizationPositionType, FFNType

# ---------------------------------------------------------------------
# Type definitions for enhanced type safety
# ---------------------------------------------------------------------

EmbeddingType = Literal['learned', 'shared', 'factorized']
PositionalType = Literal['learned', 'rope', 'dual_rope', 'sincos']

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TextEncoder(keras.layers.Layer):
    """
    General purpose configurable text encoder using factory-based components.

    This layer provides a highly flexible text encoder that can be configured to create
    various natural language processing architectures. It uses factory patterns for all
    major components (embeddings, attention, normalization, FFN) to enable easy
    experimentation and architectural exploration.

    **Intent**: Provide a single, configurable text encoder that can replace multiple
    specialized implementations by supporting different embedding strategies, attention
    mechanisms, normalization types, and feed-forward networks through factory-based
    component creation.

    **Architecture**:
    ```
    Input Token IDs (batch, seq_len)
           ↓
    Configurable Word Embedding:
      - 'learned': Standard embedding table
      - 'shared': Shared input/output embeddings
      - 'factorized': Factorized embeddings for large vocab
           ↓
    Add Token Type Embeddings (optional)
           ↓
    Configurable Positional Encoding:
      - 'learned': Learned absolute positions
      - 'rope': Rotary position embeddings
      - 'dual_rope': Dual RoPE for long context
      - 'sincos': Fixed sinusoidal embeddings
           ↓
    Embedding Dropout + Normalization
           ↓
    TransformerLayer × depth (configurable components)
           ↓
    Optional Final Normalization
           ↓
    Output Features (configurable pooling via SequencePooling layer)
    ```

    **Key Features**:
    - Factory-based component creation for maximum flexibility
    - Support for multiple embedding and positional encoding strategies
    - Configurable attention mechanisms (MHA, Window, GQA, etc.)
    - Multiple normalization options (LayerNorm, RMSNorm, etc.)
    - Various FFN architectures (MLP, SwiGLU, etc.)
    - Flexible output modes using SequencePooling layer
    - Support for token type embeddings for multi-segment inputs

    Args:
        vocab_size: Integer, size of the vocabulary. Must be positive. This determines
            the number of unique tokens the encoder can handle.
        embed_dim: Integer, embedding dimension. Must be positive. This is the
            dimensionality of the token embeddings and hidden states.
        depth: Integer, number of transformer layers. Must be positive. Defaults to 12.
        num_heads: Integer, number of attention heads. Must be positive and divide
            embed_dim. Defaults to 12.
        mlp_ratio: Float, MLP expansion ratio. Must be positive. The intermediate
            size will be embed_dim * mlp_ratio. Defaults to 4.0.
        max_seq_len: Integer, maximum sequence length. Must be positive. This determines
            the maximum number of tokens the encoder can process. Defaults to 512.
        embedding_type: EmbeddingType, word embedding strategy:
            - 'learned': Standard learnable embedding table
            - 'shared': Shared input/output embeddings (ties weights)
            - 'factorized': Factorized embeddings for large vocabularies
            Defaults to 'learned'.
        positional_type: PositionalType, positional encoding strategy:
            - 'learned': Learned absolute positional embeddings
            - 'rope': Rotary position embeddings
            - 'dual_rope': Dual RoPE for long context modeling
            - 'sincos': Fixed sinusoidal positional embeddings
            Defaults to 'learned'.
        attention_type: AttentionType, attention mechanism to use:
            - 'multi_head_attention': Standard multi-head self-attention
            - 'window_attention': Windowed attention for efficiency
            - 'group_query_attention': Grouped query attention
            - 'differential_attention': Differential attention for noise reduction
            Defaults to 'multi_head_attention'.
        normalization_type: NormalizationType, normalization layer type:
            - 'layer_norm': Standard layer normalization
            - 'rms_norm': Root mean square normalization
            - 'band_rms': Band-constrained RMS normalization
            - 'dynamic_tanh': Dynamic Tanh normalization
            Defaults to 'layer_norm'.
        normalization_position: NormalizationPosition, normalization position:
            - 'post': Post-normalization (original Transformer)
            - 'pre': Pre-normalization (often more stable)
            Defaults to 'post'.
        ffn_type: FFNType, feed-forward network architecture:
            - 'mlp': Standard MLP with intermediate expansion
            - 'swiglu': SwiGLU activation with gating mechanism
            - 'differential': Differential FFN with separate pathways
            - 'geglu': GELU-based Gated Linear Unit
            Defaults to 'mlp'.
        use_token_type_embedding: Boolean, whether to add token type embeddings for
            multi-segment inputs (like BERT's segment embeddings). Defaults to False.
        type_vocab_size: Integer, size of token type vocabulary. Only used when
            use_token_type_embedding=True. Defaults to 2.
        use_cls_token: Boolean, whether to prepend a special CLS token for classification.
            When True, adds learnable CLS token at sequence start. Defaults to False.
        output_mode: PoolingMode, output mode for feature extraction:
            - 'cls': Return CLS token features (requires use_cls_token=True)
            - 'mean': Global average pooling over sequence
            - 'max': Global max pooling over sequence
            - 'first': Return first token features
            - 'last': Return last token features
            - 'none': Return full sequence features
            Defaults to 'none'.
        dropout_rate: Float, general dropout rate between 0 and 1. Applied to embeddings
            and in transformer layers. Defaults to 0.1.
        attention_dropout_rate: Float, attention-specific dropout rate. Applied to
            attention weights. Defaults to 0.1.
        embed_dropout_rate: Float, embedding dropout rate. Applied after embedding layers.
            Defaults to 0.1.
        stochastic_depth_rate: Float, stochastic depth drop path rate for regularization.
            When > 0, applies stochastic depth to transformer layers. Defaults to 0.0.
        activation: Union[str, Callable], activation function for FFN layers.
            Defaults to 'gelu'.
        use_bias: Boolean, whether to use bias in linear layers. Defaults to True.
        kernel_initializer: Union[str, Initializer], initializer for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: Union[str, Initializer], initializer for bias weights.
            Defaults to 'zeros'.
        kernel_regularizer: Optional[Regularizer], regularizer for kernel weights.
            Defaults to None.
        bias_regularizer: Optional[Regularizer], regularizer for bias weights.
            Defaults to None.
        initializer_range: Float, standard deviation for weight initialization.
            Used for embedding layers and custom initializers. Defaults to 0.02.
        layer_norm_eps: Float, epsilon value for normalization layers. Must be positive.
            Used in LayerNorm and RMSNorm layers. Defaults to 1e-12.
        rope_theta: Float, theta parameter for RoPE. Only used when positional_type
            is 'rope' or 'dual_rope'. Defaults to 10000.0.
        rope_percentage: Float, percentage of dimensions to apply RoPE to. Only used
            when positional_type is 'rope'. Must be in (0, 1]. Defaults to 1.0.
        attention_args: Optional[Dict[str, Any]], custom arguments for attention layer.
            These override default parameters for the specific attention type.
            Defaults to None.
        norm_args: Optional[Dict[str, Any]], custom arguments for normalization layers.
            Applied to all normalization layers in the encoder. Defaults to None.
        ffn_args: Optional[Dict[str, Any]], custom arguments for FFN layers.
            These override default parameters for the FFN type. Defaults to None.
        embedding_args: Optional[Dict[str, Any]], custom arguments for embedding layers.
            Type-specific parameters for embedding layers. Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        2D tensor with shape: `(batch_size, sequence_length)` containing token IDs

        Optional additional inputs (when use_token_type_embedding=True):
        - token_type_ids: 2D tensor with shape: `(batch_size, sequence_length)`
        - attention_mask: 2D/3D tensor for masking padded positions

    Output shape:
        Depends on output_mode:
        - 'cls', 'mean', 'max', 'first', 'last': `(batch_size, embed_dim)`
        - 'none': `(batch_size, sequence_length, embed_dim)`

    Attributes:
        seq_len: Integer, effective sequence length including CLS token if used.
        intermediate_size: Integer, intermediate dimension for FFN layers.
        word_embeddings: Word embedding layer created by factory.
        token_type_embeddings: Optional token type embedding layer.
        positional_embeddings: Positional encoding layer.
        embed_norm: Embedding normalization layer.
        embed_dropout_rate: Embedding dropout layer.
        transformer_layers: List of TransformerLayer instances.
        final_norm: Optional final normalization layer.
        cls_token: Optional learnable CLS token weight.
        pooling_layer: SequencePooling layer for output pooling.

    Example:
        ```python
        # Standard BERT-like encoder
        encoder = TextEncoder(
            vocab_size=30522,
            embed_dim=768,
            depth=12,
            num_heads=12,
            max_seq_len=512,
            embedding_type='learned',
            positional_type='learned',
            use_token_type_embedding=True,
            use_cls_token=True,
            output_mode='cls'
        )

        # Modern encoder with advanced components
        modern_encoder = TextEncoder(
            vocab_size=50000,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            max_seq_len=2048,
            embedding_type='factorized',
            positional_type='rope',
            attention_type='differential_attention',
            normalization_type='rms_norm',
            normalization_position='pre',
            ffn_type='swiglu',
            rope_percentage=0.5
        )

        # Efficient encoder for long sequences
        efficient_encoder = TextEncoder(
            vocab_size=32000,
            embed_dim=512,
            depth=8,
            num_heads=8,
            max_seq_len=4096,
            positional_type='dual_rope',
            attention_type='window_attention',
            normalization_type='rms_norm',
            ffn_type='swiglu',
            attention_args={'window_size': 128},
            stochastic_depth_rate=0.1
        )

        # Usage
        input_ids = keras.Input(shape=(512,), dtype='int32', name='input_ids')
        features = encoder(input_ids)  # Shape depends on output_mode
        ```

    Note:
        All components are created using factory functions to ensure consistency
        and enable easy experimentation with different architectural choices.
        The encoder follows modern Keras 3 patterns for robust serialization.

    Raises:
        ValueError: If vocab_size or embed_dim are not positive.
        ValueError: If embed_dim is not divisible by num_heads.
        ValueError: If any dimension parameter is not positive.
        ValueError: If use_cls_token=False but output_mode='cls'.
        ValueError: If positional_type requires parameters not provided.
    """

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            max_seq_len: int = 512,
            embedding_type: EmbeddingType = 'learned',
            positional_type: PositionalType = 'learned',
            attention_type: AttentionType = 'multi_head',
            normalization_type: NormalizationType = 'layer_norm',
            normalization_position: NormalizationPositionType = 'post',
            ffn_type: FFNType = 'mlp',
            use_token_type_embedding: bool = False,
            type_vocab_size: int = 2,
            use_cls_token: bool = False,
            output_mode: PoolingStrategy = 'none',
            dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.1,
            embed_dropout_rate: float = 0.1,
            stochastic_depth_rate: float = 0.0,
            activation: Union[str, Callable] = 'gelu',
            use_bias: bool = True,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            initializer_range: float = 0.02,
            layer_norm_eps: float = 1e-12,
            rope_theta: float = 10000.0,
            rope_percentage: float = 1.0,
            attention_args: Optional[Dict[str, Any]] = None,
            norm_args: Optional[Dict[str, Any]] = None,
            ffn_args: Optional[Dict[str, Any]] = None,
            embedding_args: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if type_vocab_size <= 0:
            raise ValueError(f"type_vocab_size must be positive, got {type_vocab_size}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(f"attention_dropout_rate must be between 0 and 1, got {attention_dropout_rate}")
        if not (0.0 <= embed_dropout_rate <= 1.0):
            raise ValueError(f"embed_dropout_rate must be between 0 and 1, got {embed_dropout_rate}")
        if initializer_range <= 0:
            raise ValueError(f"initializer_range must be positive, got {initializer_range}")
        if layer_norm_eps <= 0:
            raise ValueError(f"layer_norm_eps must be positive, got {layer_norm_eps}")
        if not use_cls_token and output_mode == 'cls':
            raise ValueError("output_mode='cls' requires use_cls_token=True")
        if positional_type in ['rope', 'dual_rope'] and rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {rope_theta}")
        if positional_type == 'rope' and not (0.0 < rope_percentage <= 1.0):
            raise ValueError(f"rope_percentage must be in (0, 1], got {rope_percentage}")

        # Store ALL configuration parameters for serialization
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
        self.use_cls_token = use_cls_token
        self.output_mode = output_mode
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.embed_dropout_rate = embed_dropout_rate
        self.stochastic_depth_rate = stochastic_depth_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.rope_theta = rope_theta
        self.rope_percentage = rope_percentage
        self.attention_args = attention_args or {}
        self.norm_args = norm_args or {}
        self.ffn_args = ffn_args or {}
        self.embedding_args = embedding_args or {}

        # Computed properties
        self.seq_len = max_seq_len + (1 if use_cls_token else 0)
        self.intermediate_size = int(embed_dim * mlp_ratio)

        # CREATE all sub-layers in __init__ (modern Keras 3 pattern)

        # Create word embeddings
        self._create_word_embeddings()

        # Create token type embeddings if needed
        self.token_type_embeddings = None
        if use_token_type_embedding:
            self.token_type_embeddings = layers.Embedding(
                input_dim=type_vocab_size,
                output_dim=embed_dim,
                mask_zero=False,
                embeddings_initializer=initializers.TruncatedNormal(stddev=initializer_range),
                embeddings_regularizer=kernel_regularizer,
                name='token_type_embeddings'
            )

        # Create positional encodings
        self.positional_embeddings = self._create_positional_embeddings()

        # Create embedding normalization and dropout
        norm_config = {'epsilon': layer_norm_eps, **self.norm_args}
        self.embed_norm = create_normalization_layer(
            normalization_type,
            name='embed_norm',
            **norm_config
        )

        self.embed_dropout_layer = layers.Dropout(
            embed_dropout_rate,
            name='embed_dropout'
        ) if embed_dropout_rate > 0.0 else None

        # Create transformer layers using factory components
        self.transformer_layers = []
        for i in range(depth):
            # Calculate stochastic depth rate (linearly increasing)
            layer_drop_rate = stochastic_depth_rate * i / max(1, depth - 1)

            layer = TransformerLayer(
                hidden_size=embed_dim,
                num_heads=num_heads,
                intermediate_size=self.intermediate_size,
                attention_type=attention_type,
                attention_args=self.attention_args,
                normalization_type=normalization_type,
                normalization_position=normalization_position,
                attention_norm_args=self.norm_args,
                ffn_norm_args=self.norm_args,
                ffn_type=ffn_type,
                ffn_args=self.ffn_args,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                use_stochastic_depth=stochastic_depth_rate > 0.0,
                stochastic_depth_rate=layer_drop_rate,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=initializers.TruncatedNormal(stddev=initializer_range),
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"transformer_layer_{i}"
            )
            self.transformer_layers.append(layer)

        # Create final normalization layer (only for pre-norm)
        self.final_norm = None
        if normalization_position == 'pre':
            self.final_norm = create_normalization_layer(
                normalization_type,
                name="final_norm",
                **norm_config
            )

        # Create pooling layer using SequencePooling
        self.pooling_layer = SequencePooling(
            strategy=output_mode,
            name='output_pooling'
        )

        # CLS token will be created in build()
        self.cls_token = None

        logger.info(f"Created TextEncoder with vocab_size={vocab_size}, embed_dim={embed_dim}, "
                    f"depth={depth}, max_seq_len={max_seq_len}, output_mode={output_mode}")

    def _create_word_embeddings(self) -> None:
        """
        Create word embedding layer(s) based on specified strategy.
        This method sets attributes on `self` instead of returning for serialization robustness.
        """
        base_args = {
            'embeddings_initializer': initializers.TruncatedNormal(stddev=self.initializer_range),
            'embeddings_regularizer': self.kernel_regularizer,
            'mask_zero': True,
        }
        base_args.update(self.embedding_args)

        self.word_embeddings = None
        self.factorized_embed_layer = None
        self.embed_projection_layer = None

        if self.embedding_type in ['learned', 'shared']:
            self.word_embeddings = layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embed_dim,
                name='word_embeddings',
                **base_args
            )
        else:  # factorized
            factorized_dim = min(self.embed_dim, 128)
            factorized_dim = self.embedding_args.get('factorized_dim', factorized_dim)

            self.factorized_embed_layer = layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=factorized_dim,
                embeddings_initializer=initializers.TruncatedNormal(stddev=self.initializer_range),
                embeddings_regularizer=self.kernel_regularizer,
                mask_zero=True,
                name='factorized_embed'
            )
            self.embed_projection_layer = layers.Dense(
                self.embed_dim,
                use_bias=False,
                kernel_initializer=initializers.TruncatedNormal(stddev=self.initializer_range),
                kernel_regularizer=self.kernel_regularizer,
                name='embed_projection'
            )

    def _create_positional_embeddings(self) -> Optional[keras.layers.Layer]:
        """
        Create positional embedding layer based on specified type.

        Returns:
            Positional embedding layer or None for RoPE (applied in attention).
        """
        if self.positional_type == 'learned':
            # Standard learned positional embeddings
            return create_embedding_layer(
                'positional_learned',
                max_seq_len=self.seq_len,
                dim=self.embed_dim,
                dropout_rate=0.0,  # We handle dropout separately
                scale=self.initializer_range,
                name="positional_embeddings"
            )

        elif self.positional_type == 'rope':
            # RoPE is applied within attention layers, but the provided TransformerLayer
            # uses attention mechanisms that do not support RoPE parameters in their
            # constructor. To prevent crashing, we avoid adding these to attention_args.
            # This means RoPE is effectively NOT applied with the current setup.
            # self.attention_args.update({
            #     'rope_theta': self.rope_theta,
            #     'rope_percentage': self.rope_percentage
            # })
            return None

        elif self.positional_type == 'dual_rope':
            # See comment for 'rope'.
            # self.attention_args.update({
            #     'global_theta_base': self.rope_theta,
            #     'local_theta_base': 10000.0  # Standard local theta
            # })
            return None

        else:  # sincos
            # Fixed sinusoidal positional embeddings
            return create_embedding_layer(
                'continuous_sincos',
                dim=self.embed_dim,
                ndim=1,  # 1D sequence positions
                max_wavelength=self.rope_theta,
                name="sincos_positional_embeddings"
            )

    def build(self, input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]], Dict[str, Any]]) -> None:
        """
        Build the text encoder and all its sub-layers.

        This method validates input shape and explicitly builds sub-layers
        for robust serialization following modern Keras 3 patterns.

        Args:
            input_shape: Shape tuple(s) of input tensors. Can be:
                - Single tuple (batch_size, seq_len) for input_ids only
                - List of tuples for multiple inputs (input_ids, token_type_ids, etc.)
                - Dict of shapes for named inputs

        Raises:
            ValueError: If input shape is invalid or incompatible with configuration.
        """
        # Handle multiple input shapes
        if isinstance(input_shape, dict):
            main_input_shape = input_shape['input_ids']
        elif isinstance(input_shape, list):
            main_input_shape = input_shape[0]
        else:
            main_input_shape = input_shape

        if main_input_shape is not None and len(main_input_shape) != 2:
            raise ValueError(
                f"Expected 2D input shape (batch_size, seq_len), got {main_input_shape}"
            )

        # Create CLS token weight if needed
        if self.use_cls_token:
            self.cls_token = self.add_weight(
                name="cls_token",
                shape=(1, 1, self.embed_dim),
                initializer=initializers.TruncatedNormal(stddev=self.initializer_range),
                trainable=True
            )

        # Build word embeddings
        if self.word_embeddings is not None:
            self.word_embeddings.build(main_input_shape)
        if self.factorized_embed_layer is not None:
            self.factorized_embed_layer.build(main_input_shape)
            # Compute the symbolic output shape of the first embedding layer
            factorized_shape = self.factorized_embed_layer.compute_output_shape(
                main_input_shape if main_input_shape else (None, None)
            )
            self.embed_projection_layer.build(factorized_shape)


        # Build token type embeddings if present
        if self.token_type_embeddings is not None:
            self.token_type_embeddings.build(main_input_shape)

        # Build positional embeddings if present
        embedding_output_shape = (None, self.seq_len, self.embed_dim)
        if self.positional_embeddings is not None:
            if self.positional_type == 'learned':
                self.positional_embeddings.build(embedding_output_shape)
            else:  # sincos - expects coordinate input
                coord_shape = (None, self.seq_len, 1)
                self.positional_embeddings.build(coord_shape)

        # Build embedding normalization and dropout
        self.embed_norm.build(embedding_output_shape)
        if self.embed_dropout_layer is not None:
            self.embed_dropout_layer.build(embedding_output_shape)

        # Build transformer layers
        for layer in self.transformer_layers:
            layer.build(embedding_output_shape)

        # Build final normalization if present
        if self.final_norm is not None:
            self.final_norm.build(embedding_output_shape)

        # Build pooling layer
        self.pooling_layer.build(embedding_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            token_type_ids: Optional[keras.KerasTensor] = None,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the text encoder.

        Args:
            inputs: Input tensor(s). Can be:
                - Single tensor: input_ids of shape [batch_size, seq_len]
                - Dict: {'input_ids': tensor, 'token_type_ids': tensor, ...}
            token_type_ids: Optional token type IDs for segment embeddings
                of shape [batch_size, seq_len]. Used when use_token_type_embedding=True.
            attention_mask: Optional attention mask to avoid attention on padding
                tokens of shape [batch_size, seq_len] or [batch_size, seq_len, seq_len].
            training: Optional boolean indicating training mode.

        Returns:
            Output tensor with shape depending on output_mode configuration.
        """
        # Extract input_ids and optional token_type_ids from inputs
        if isinstance(inputs, dict):
            input_ids = inputs['input_ids']
            if token_type_ids is None and 'token_type_ids' in inputs:
                token_type_ids = inputs['token_type_ids']
            if attention_mask is None and 'attention_mask' in inputs:
                attention_mask = inputs['attention_mask']
        else:
            input_ids = inputs

        batch_size = ops.shape(input_ids)[0]

        # Word embeddings
        if self.embedding_type == 'factorized':
            x = self.factorized_embed_layer(input_ids, training=training)
            x = self.embed_projection_layer(x, training=training)
        else:
            x = self.word_embeddings(input_ids, training=training)

        # Add token type embeddings if configured
        if self.use_token_type_embedding and self.token_type_embeddings is not None:
            if token_type_ids is None:
                # Default to all zeros (first segment)
                token_type_ids = ops.zeros_like(input_ids)
            type_embeddings = self.token_type_embeddings(token_type_ids, training=training)
            x = x + type_embeddings

        # Add CLS token if configured
        if self.use_cls_token:
            cls_tokens = ops.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))
            x = ops.concatenate([cls_tokens, x], axis=1)
            # Update attention mask to account for CLS token
            if attention_mask is not None:
                cls_mask = ops.ones((batch_size, 1), dtype=attention_mask.dtype)
                attention_mask = ops.concatenate([cls_mask, attention_mask], axis=1)

        # Add positional embeddings
        if self.positional_embeddings is not None:
            if self.positional_type == 'learned':
                x = self.positional_embeddings(x, training=training)
            else:  # sincos
                # Create position coordinates
                current_seq_len = ops.shape(x)[1]
                positions = ops.cast(ops.arange(current_seq_len), dtype='float32')
                positions = ops.expand_dims(positions, axis=-1)  # (seq_len, 1)
                positions = ops.expand_dims(positions, axis=0)  # (1, seq_len, 1)
                positions = ops.broadcast_to(positions, (batch_size, current_seq_len, 1))

                pos_embeddings = self.positional_embeddings(positions, training=training)
                x = x + pos_embeddings

        # Apply embedding normalization and dropout
        x = self.embed_norm(x, training=training)
        if self.embed_dropout_layer is not None:
            x = self.embed_dropout_layer(x, training=training)

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, attention_mask=attention_mask, training=training)

        # Apply final normalization if present
        if self.final_norm is not None:
            x = self.final_norm(x, training=training)

        # Apply pooling using SequencePooling layer
        # Convert attention mask to format expected by SequencePooling (batch, seq_len)
        pooling_mask = None
        if attention_mask is not None:
            # If attention_mask is 2D (batch, seq_len), use as is
            # If attention_mask is 3D (batch, seq_len, seq_len), take diagonal or first row
            if len(ops.shape(attention_mask)) == 2:
                pooling_mask = attention_mask
            elif len(ops.shape(attention_mask)) == 3:
                # Use the first row which indicates which positions are valid
                pooling_mask = attention_mask[:, 0, :]

        # Apply pooling
        output = self.pooling_layer(x, mask=pooling_mask, training=training)

        return output

    def get_sequence_features(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            token_type_ids: Optional[keras.KerasTensor] = None,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Get full sequence features regardless of output_mode.

        Args:
            inputs: Input tensor(s) as in call method.
            token_type_ids: Optional token type IDs.
            attention_mask: Optional attention mask.
            training: Optional boolean indicating training mode.

        Returns:
            Full sequence features tensor of shape [batch_size, seq_len, embed_dim].
        """
        # Temporarily save original pooling strategy and set to 'none'
        original_strategy = self.pooling_layer.strategy
        self.pooling_layer.strategy = ['none']

        try:
            features = self(
                inputs=inputs,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                training=training
            )
            return features
        finally:
            # Restore original pooling strategy
            self.pooling_layer.strategy = original_strategy

    def get_pooled_features(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            pooling_mode: PoolingStrategy = 'mean',
            token_type_ids: Optional[keras.KerasTensor] = None,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Get pooled features with specified pooling strategy.

        Args:
            inputs: Input tensor(s) as in call method.
            pooling_mode: Pooling strategy to use.
            token_type_ids: Optional token type IDs.
            attention_mask: Optional attention mask.
            training: Optional boolean indicating training mode.

        Returns:
            Pooled features tensor of shape [batch_size, embed_dim].
        """
        # Temporarily save original pooling strategy and set to requested mode
        original_strategy = self.pooling_layer.strategy
        self.pooling_layer.strategy = [pooling_mode]

        try:
            features = self(
                inputs=inputs,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                training=training
            )
            return features
        finally:
            # Restore original pooling strategy
            self.pooling_layer.strategy = original_strategy

    def compute_output_shape(self, input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]], Dict[str, Any]]) -> \
    Tuple[Optional[int], ...]:
        """
        Compute output shape given input shape.

        Args:
            input_shape: Input shape tuple(s) or dict.

        Returns:
            Output shape tuple based on output_mode configuration.
        """
        if isinstance(input_shape, dict):
            main_input_shape = input_shape['input_ids']
        elif isinstance(input_shape, list):
            main_input_shape = input_shape[0]
        else:
            main_input_shape = input_shape

        batch_size = main_input_shape[0]
        seq_len = main_input_shape[1]
        if self.use_cls_token and seq_len is not None:
            seq_len += 1

        # Create dummy input shape for pooling layer
        transformer_output_shape = (batch_size, seq_len, self.embed_dim)
        return self.pooling_layer.compute_output_shape(transformer_output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns ALL parameters passed to __init__ for complete reconstruction.

        Returns:
            Dictionary containing all layer configuration parameters.
        """
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
            'use_cls_token': self.use_cls_token,
            'output_mode': self.output_mode,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'embed_dropout_rate': self.embed_dropout_rate,
            'stochastic_depth_rate': self.stochastic_depth_rate,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'initializer_range': self.initializer_range,
            'layer_norm_eps': self.layer_norm_eps,
            'rope_theta': self.rope_theta,
            'rope_percentage': self.rope_percentage,
            'attention_args': self.attention_args,
            'norm_args': self.norm_args,
            'ffn_args': self.ffn_args,
            'embedding_args': self.embedding_args,
        })
        return config


# ---------------------------------------------------------------------
# Factory Functions for Convenient Encoder Creation
# ---------------------------------------------------------------------


def create_text_encoder(
        vocab_size: int,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 512,
        embedding_type: EmbeddingType = 'learned',
        positional_type: PositionalType = 'learned',
        attention_type: AttentionType = 'multi_head',
        normalization_type: NormalizationType = 'layer_norm',
        normalization_position: NormalizationPositionType = 'post',
        ffn_type: FFNType = 'mlp',
        **kwargs: Any
) -> TextEncoder:
    """
    Factory function to create a TextEncoder with validated parameters.

    This function provides parameter validation and sensible defaults for creating
    text encoders with different architectural configurations. It supports all
    major transformer variants through configurable components.

    Args:
        vocab_size: Size of the vocabulary.
        embed_dim: Embedding dimension.
        depth: Number of transformer layers.
        num_heads: Number of attention heads.
        max_seq_len: Maximum sequence length.
        embedding_type: Type of word embedding strategy.
        positional_type: Type of positional encoding.
        attention_type: Type of attention mechanism.
        normalization_type: Type of normalization.
        normalization_position: Position of normalization layers.
        ffn_type: Type of feed-forward network.
        **kwargs: Additional arguments for TextEncoder constructor.

    Returns:
        Configured TextEncoder instance.

    Raises:
        ValueError: If any parameter validation fails.

    Example:
        ```python
        # Standard BERT-style encoder
        encoder = create_text_encoder(
            vocab_size=30522,
            embed_dim=768,
            depth=12,
            num_heads=12
        )

        # Modern encoder with advanced components
        encoder = create_text_encoder(
            vocab_size=50000,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            positional_type='rope',
            attention_type='differential_attention',
            normalization_type='rms_norm',
            ffn_type='swiglu'
        )
        ```
    """
    # Validate basic parameters
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")

    if embed_dim <= 0:
        raise ValueError(f"embed_dim must be positive, got {embed_dim}")

    if depth <= 0 or num_heads <= 0:
        raise ValueError("depth and num_heads must be positive")

    if embed_dim % num_heads != 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

    return TextEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        embedding_type=embedding_type,
        positional_type=positional_type,
        attention_type=attention_type,
        normalization_type=normalization_type,
        normalization_position=normalization_position,
        ffn_type=ffn_type,
        **kwargs
    )

# ---------------------------------------------------------------------

def create_bert_encoder(
        vocab_size: int = 30522,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 512,
        **kwargs: Any
) -> TextEncoder:
    """Create BERT-style encoder configuration."""
    return create_text_encoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        embedding_type='learned',
        positional_type='learned',
        attention_type='multi_head',
        normalization_type='layer_norm',
        normalization_position='post',
        ffn_type='mlp',
        use_token_type_embedding=True,
        use_cls_token=True,
        output_mode='cls',
        **kwargs
    )

# ---------------------------------------------------------------------

def create_roberta_encoder(
        vocab_size: int = 50265,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 512,
        **kwargs: Any
) -> TextEncoder:
    """Create RoBERTa-style encoder configuration."""
    return create_text_encoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        embedding_type='learned',
        positional_type='learned',
        attention_type='multi_head',
        normalization_type='layer_norm',
        normalization_position='post',
        ffn_type='mlp',
        use_token_type_embedding=False,
        use_cls_token=True,
        output_mode='cls',
        **kwargs
    )

# ---------------------------------------------------------------------

def create_modern_encoder(
        vocab_size: int = 50000,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        max_seq_len: int = 2048,
        **kwargs: Any
) -> TextEncoder:
    """Create modern encoder with advanced components."""
    return create_text_encoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        embedding_type='factorized',
        positional_type='rope',
        attention_type='differential',
        normalization_type='rms_norm',
        normalization_position='pre',
        ffn_type='swiglu',
        use_token_type_embedding=False,
        use_cls_token=False,
        output_mode='mean',
        **kwargs
    )

# ---------------------------------------------------------------------

def create_efficient_encoder(
        vocab_size: int = 32000,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        max_seq_len: int = 1024,
        **kwargs: Any
) -> TextEncoder:
    """Create efficient encoder for resource-constrained environments."""
    return create_text_encoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        embedding_type='factorized',
        positional_type='rope',
        attention_type='multi_head',
        normalization_type='rms_norm',
        normalization_position='pre',
        ffn_type='swiglu',
        stochastic_depth_rate=0.1,
        embedding_args={'factorized_dim': 64},
        **kwargs
    )

# ---------------------------------------------------------------------