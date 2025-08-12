"""
Byte-BERT: BERT with Byte Latent Transformer Features

This module implements a BERT model that incorporates key innovations from the
Byte Latent Transformer (BLT), creating a powerful byte-level bidirectional
encoder that combines the understanding capabilities of BERT with the efficiency
and robustness advantages of byte-level processing.

Key Features from BLT Integration:
=================================

1. **Byte-Level Processing**: Operates directly on UTF-8 bytes instead of
   traditional subword tokens, providing language-agnostic capabilities and
   enhanced robustness to noisy or corrupted text.

2. **Dynamic Entropy-Based Patching**: Uses learned entropy patterns to
   dynamically group bytes into patches of varying length, allocating compute
   where linguistic complexity demands it.

3. **Hierarchical Architecture**:
   - Local Encoder: Processes bytes within patches using local attention
   - Global Encoder: Processes patch representations bidirectionally
   - Cross-Attention: Enables rich interaction between byte and patch levels

4. **Hash N-Gram Embeddings**: Incorporates contextual byte patterns (3-8 grams)
   using rolling polynomial hash functions for enhanced representations.

5. **Bidirectional Understanding**: Maintains BERT's bidirectional attention
   pattern (unlike BLT's causal generation focus) for better understanding tasks.

Architecture Overview:
=====================

Input Bytes → [Hash N-gram Embeddings] → [Dynamic Patching] →
[Local Byte Encoding] → [Global Patch Processing] → [Cross-Attention Fusion] →
[Task-Specific Heads]

This creates a model that:
- Processes text at the fundamental byte level
- Adapts compute allocation based on content complexity
- Maintains full bidirectional understanding
- Provides enhanced multilingual and noise robustness
- Scales efficiently with dynamic patch sizing
"""

import keras
from keras import ops, layers
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, List, Tuple

from ..utils.logger import logger
from ..layers.norms.rms_norm import RMSNorm
from ..layers.transformer import TransformerLayer
from ..layers.byte_latent_transformer_blocks import (
    ByteTokenizer, EntropyModel, DynamicPatcher,
    LocalEncoder, PatchPooling
)


@dataclass
class ByteBertConfig:
    """
    Configuration for Byte-BERT model combining BERT and BLT features.

    This configuration extends traditional BERT parameters with BLT-specific
    settings for byte-level processing and dynamic patching.

    Attributes:
        # Core BERT parameters
        vocab_size: Size of byte vocabulary (typically 256 + special tokens).
        hidden_size: Hidden dimension of the global transformer layers.
        num_layers: Number of global transformer encoder layers.
        num_heads: Number of attention heads in global transformer.
        intermediate_size: Size of intermediate layer in feed-forward network.

        # BLT-specific parameters
        local_hidden_size: Hidden dimension for local byte processing.
        num_local_layers: Number of local transformer layers.
        num_local_heads: Number of attention heads in local transformers.
        max_sequence_length: Maximum sequence length in bytes.
        max_patches: Maximum number of patches per sequence.
        entropy_threshold: Threshold for dynamic patch boundary detection.
        patch_pooling_method: Method for aggregating bytes into patches.
        cross_attention_queries: Number of query vectors for cross-attention.

        # Hash embedding parameters
        use_hash_embeddings: Whether to use hash n-gram embeddings.
        hash_vocab_size: Size of hash embedding vocabulary.
        ngram_sizes: List of n-gram sizes for hash embeddings.

        # Standard parameters
        hidden_dropout_prob: Dropout probability for hidden layers.
        attention_probs_dropout_prob: Dropout probability for attention.
        hidden_act: Activation function for feed-forward networks.
        initializer_range: Standard deviation for weight initialization.
        layer_norm_eps: Epsilon for normalization layers.
        normalization_type: Type of normalization ('layer_norm', 'rms_norm', etc.).
        position_embedding_type: Type of position embeddings.

        # Training parameters
        use_stochastic_depth: Whether to use stochastic depth regularization.
        stochastic_depth_rate: Drop path rate for stochastic depth.
        classifier_dropout: Dropout rate for classification head.
    """
    # Core BERT parameters
    vocab_size: int = 260  # 256 bytes + special tokens
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072

    # BLT-specific parameters
    local_hidden_size: int = 384
    num_local_layers: int = 4
    num_local_heads: int = 6
    max_sequence_length: int = 2048
    max_patches: int = 512
    entropy_threshold: float = 1.5
    patch_pooling_method: str = 'attention'
    cross_attention_queries: int = 4

    # Hash embedding parameters
    use_hash_embeddings: bool = True
    hash_vocab_size: int = 500000
    ngram_sizes: List[int] = None

    # Standard parameters
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    hidden_act: str = "gelu"
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    normalization_type: str = "layer_norm"
    position_embedding_type: str = "absolute"

    # Training parameters
    use_stochastic_depth: bool = False
    stochastic_depth_rate: float = 0.1
    classifier_dropout: Optional[float] = None

    def __post_init__(self):
        """Initialize default values after creation."""
        if self.ngram_sizes is None:
            self.ngram_sizes = [3, 4, 5, 6, 7, 8]

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.local_hidden_size % self.num_local_heads != 0:
            raise ValueError(
                f"local_hidden_size ({self.local_hidden_size}) must be divisible by "
                f"num_local_heads ({self.num_local_heads})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ByteBertConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


@keras.saving.register_keras_serializable()
class HashNGramEmbedding(layers.Layer):
    """
    Hash-based n-gram embedding layer for enhanced byte representations.

    Uses rolling polynomial hash functions to map byte n-grams to embedding
    vectors without requiring explicit vocabulary storage, inspired by BLT's
    hash embedding approach.

    Args:
        hash_vocab_size: Size of hash embedding vocabulary.
        embed_dim: Dimension of embedding vectors.
        ngram_sizes: List of n-gram sizes to compute embeddings for.
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        hash_vocab_size: int,
        embed_dim: int,
        ngram_sizes: List[int] = [3, 4, 5, 6, 7, 8],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hash_vocab_size = hash_vocab_size
        self.embed_dim = embed_dim
        self.ngram_sizes = ngram_sizes

        # Hash embedding tables for each n-gram size
        self.hash_embeddings = {}
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build hash embedding tables."""
        if self.built:
            return

        self._build_input_shape = input_shape

        # Create embedding table for each n-gram size
        for n in self.ngram_sizes:
            self.hash_embeddings[str(n)] = self.add_weight(
                name=f'hash_embedding_{n}gram',
                shape=(self.hash_vocab_size, self.embed_dim),
                initializer='glorot_uniform',
                trainable=True
            )

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute hash n-gram embeddings for byte sequence.

        Args:
            inputs: Byte token sequence of shape (batch_size, seq_len).

        Returns:
            Combined n-gram embeddings of shape (batch_size, seq_len, embed_dim).
        """
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # Initialize combined embeddings
        combined_embeddings = ops.zeros((batch_size, seq_len, self.embed_dim))

        # Compute embeddings for each n-gram size
        for n in self.ngram_sizes:
            ngram_embeddings = self._compute_ngram_embeddings(inputs, n)
            combined_embeddings = combined_embeddings + ngram_embeddings

        # Average across n-gram sizes
        combined_embeddings = combined_embeddings / len(self.ngram_sizes)

        return combined_embeddings

    def _compute_ngram_embeddings(
        self,
        inputs: keras.KerasTensor,
        n: int
    ) -> keras.KerasTensor:
        """Compute embeddings for specific n-gram size using polynomial hashing."""
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # Rolling polynomial hash with base 257 (larger than byte vocab)
        base = 257

        # Initialize hash values
        hash_values = ops.zeros((batch_size, seq_len), dtype='int32')

        # Compute rolling hash for each position
        for i in range(seq_len):
            if i < n - 1:
                # Not enough context for full n-gram, use partial
                current_hash = ops.cast(inputs[:, i], 'int32')
            else:
                # Compute full n-gram hash
                current_hash = ops.cast(inputs[:, i], 'int32')
                for j in range(1, n):
                    byte_val = ops.cast(inputs[:, i - j], 'int32')
                    current_hash = (current_hash * base + byte_val) % self.hash_vocab_size

            hash_values = ops.slice_update(hash_values, [slice(None), i], current_hash)

        # Look up embeddings
        embedding_table = self.hash_embeddings[str(n)]
        ngram_embeddings = ops.take(embedding_table, hash_values, axis=0)

        return ngram_embeddings

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'hash_vocab_size': self.hash_vocab_size,
            'embed_dim': self.embed_dim,
            'ngram_sizes': self.ngram_sizes,
        })
        return config


@keras.saving.register_keras_serializable()
class ByteBertEmbeddings(layers.Layer):
    """
    Byte-BERT embeddings combining byte-level and hash n-gram representations.

    This layer creates rich byte representations by combining:
    - ByteTokenizer for proper byte-level tokenization
    - Direct byte embeddings for each byte value
    - Hash n-gram embeddings for contextual patterns
    - Positional embeddings for sequence information

    Args:
        config: ByteBertConfig containing all hyperparameters.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, config: ByteBertConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Initialize ByteTokenizer for proper byte handling
        self.tokenizer = ByteTokenizer(
            vocab_size=config.vocab_size,
            name="byte_tokenizer"
        )

        # Initialize sublayers to None - created in build()
        self.byte_embeddings = None
        self.position_embeddings = None
        self.hash_embeddings = None
        self.layer_norm = None
        self.dropout = None

        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all embedding sublayers."""
        if self.built:
            return

        self._build_input_shape = input_shape

        # Build ByteTokenizer first
        self.tokenizer.build(input_shape)

        # Direct byte embeddings using tokenizer's vocabulary
        self.byte_embeddings = layers.Embedding(
            input_dim=self.tokenizer.vocab_size,
            output_dim=self.config.local_hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=self.config.initializer_range
            ),
            mask_zero=True,
            name="byte_embeddings"
        )

        # Position embeddings
        self.position_embeddings = layers.Embedding(
            input_dim=self.config.max_sequence_length,
            output_dim=self.config.local_hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=self.config.initializer_range
            ),
            name="position_embeddings"
        )

        # Hash n-gram embeddings if enabled
        if self.config.use_hash_embeddings:
            self.hash_embeddings = HashNGramEmbedding(
                hash_vocab_size=self.config.hash_vocab_size,
                embed_dim=self.config.local_hidden_size,
                ngram_sizes=self.config.ngram_sizes,
                name="hash_embeddings"
            )

        # Normalization and dropout
        if self.config.normalization_type == 'layer_norm':
            self.layer_norm = layers.LayerNormalization(
                epsilon=self.config.layer_norm_eps,
                name="layer_norm"
            )
        elif self.config.normalization_type == 'rms_norm':
            self.layer_norm = RMSNorm(
                epsilon=self.config.layer_norm_eps,
                name="layer_norm"
            )

        self.dropout = layers.Dropout(
            rate=self.config.hidden_dropout_prob,
            name="dropout"
        )

        # Build all sublayers
        self.byte_embeddings.build(input_shape)
        self.position_embeddings.build(input_shape)

        if self.hash_embeddings is not None:
            self.hash_embeddings.build(input_shape)

        embeddings_shape = (*input_shape, self.config.local_hidden_size)
        self.layer_norm.build(embeddings_shape)
        self.dropout.build(embeddings_shape)

        super().build(input_shape)

    def call(
        self,
        input_ids: keras.KerasTensor,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply byte-level embeddings with hash n-gram enhancement.

        Args:
            input_ids: Byte token IDs of shape (batch_size, seq_len).
            position_ids: Position IDs of shape (batch_size, seq_len).
            training: Whether in training mode.

        Returns:
            Enhanced byte embeddings of shape (batch_size, seq_len, local_hidden_size).
        """
        input_shape = ops.shape(input_ids)
        batch_size, seq_length = input_shape[0], input_shape[1]

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = ops.arange(seq_length, dtype="int32")
            position_ids = ops.expand_dims(position_ids, axis=0)
            position_ids = ops.broadcast_to(position_ids, (batch_size, seq_length))

        # Get base byte embeddings
        byte_embeds = self.byte_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)

        # Combine embeddings
        embeddings = byte_embeds + position_embeds

        # Add hash n-gram embeddings if enabled
        if self.hash_embeddings is not None:
            hash_embeds = self.hash_embeddings(input_ids)
            embeddings = embeddings + hash_embeds

        # Apply normalization and dropout
        embeddings = self.layer_norm(embeddings, training=training)
        embeddings = self.dropout(embeddings, training=training)

        return embeddings

    def encode_text(self, text: str, max_length: Optional[int] = None) -> keras.KerasTensor:
        """
        Encode text to byte token IDs using the ByteTokenizer.

        Args:
            text: Input text string.
            max_length: Maximum sequence length (uses config default if None).

        Returns:
            Byte token IDs tensor.
        """
        if max_length is None:
            max_length = self.config.max_sequence_length

        # Convert text to byte tokens using ByteTokenizer
        byte_tokens = self.tokenizer.text_to_bytes(
            text,
            add_bos=True,
            add_eos=True
        )

        # Pad or truncate to max_length
        if len(byte_tokens) > max_length:
            byte_tokens = byte_tokens[:max_length]
        else:
            byte_tokens.extend([0] * (max_length - len(byte_tokens)))  # Pad with 0

        return ops.array([byte_tokens], dtype='int32')

    def decode_tokens(self, token_ids: keras.KerasTensor) -> str:
        """
        Decode byte token IDs back to text using the ByteTokenizer.

        Args:
            token_ids: Byte token IDs tensor.

        Returns:
            Decoded text string.
        """
        # Convert to numpy and get first sequence
        tokens = token_ids.numpy()[0].tolist()

        # Remove padding tokens (0)
        tokens = [t for t in tokens if t != 0]

        # Decode using ByteTokenizer
        return self.tokenizer.tokens_to_text(tokens)

    def get_vocab_size(self) -> int:
        """Get vocabulary size from ByteTokenizer."""
        return self.tokenizer.vocab_size

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({'config': self.config.to_dict()})
        return config


@keras.saving.register_keras_serializable()
class ByteBertEncoder(layers.Layer):
    """
    Byte-BERT encoder with hierarchical processing.

    Implements the core hierarchical architecture:
    1. Local encoding of byte sequences within patches
    2. Dynamic patching based on entropy patterns
    3. Global processing of patch representations
    4. Cross-attention between byte and patch levels

    Args:
        config: ByteBertConfig containing all hyperparameters.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, config: ByteBertConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Initialize components to None - created in build()
        self.entropy_model = None
        self.patcher = None
        self.local_encoder = None
        self.patch_pooling = None
        self.global_layers = None
        self.cross_attention = None

        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all encoder components."""
        if self.built:
            return

        self._build_input_shape = input_shape

        # Entropy model for dynamic patching
        self.entropy_model = EntropyModel(
            vocab_size=self.config.vocab_size,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            max_seq_len=self.config.max_sequence_length,
            name='entropy_model'
        )

        # Dynamic patcher
        self.patcher = DynamicPatcher(
            entropy_threshold=self.config.entropy_threshold,
            max_patches=self.config.max_patches,
            name='patcher'
        )

        # Local encoder for processing bytes within patches
        self.local_encoder = LocalEncoder(
            vocab_size=self.config.vocab_size,
            local_dim=self.config.local_hidden_size,
            num_local_layers=self.config.num_local_layers,
            num_heads_local=self.config.num_local_heads,
            max_sequence_length=self.config.max_sequence_length,
            max_patches=self.config.max_patches,
            dropout_rate=self.config.hidden_dropout_prob,
            patch_pooling_method=self.config.patch_pooling_method,
            global_dim=self.config.hidden_size,
            cross_attention_queries=self.config.cross_attention_queries,
            name='local_encoder'
        )

        # Patch pooling for creating global representations
        self.patch_pooling = PatchPooling(
            pooling_method=self.config.patch_pooling_method,
            output_dim=self.config.hidden_size,
            num_queries=self.config.cross_attention_queries,
            name='patch_pooling'
        )

        # Global transformer layers for patch-level processing
        self.global_layers = []
        for i in range(self.config.num_layers):
            layer = TransformerLayer(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_heads,
                intermediate_size=self.config.intermediate_size,
                normalization_type=self.config.normalization_type,
                dropout_rate=self.config.hidden_dropout_prob,
                attention_dropout_rate=self.config.attention_probs_dropout_prob,
                use_stochastic_depth=self.config.use_stochastic_depth,
                stochastic_depth_rate=self.config.stochastic_depth_rate,
                activation=self.config.hidden_act,
                use_bias=True,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.config.initializer_range
                ),
                name=f'global_layer_{i}'
            )
            self.global_layers.append(layer)

        # Cross-attention for byte-patch fusion
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=self.config.num_heads,
            key_dim=self.config.hidden_size // self.config.num_heads,
            dropout=self.config.attention_probs_dropout_prob,
            name='cross_attention'
        )

        # Build all components with appropriate shapes
        self.entropy_model.build(input_shape)
        self.patcher.build((None,))  # Entropy shape

        # Local encoder expects byte embeddings
        local_input_shape = (*input_shape, self.config.local_hidden_size)
        self.local_encoder.build(local_input_shape)

        # Patch pooling expects local encoder output
        patch_shape = (None, self.config.max_patches, self.config.local_hidden_size)
        self.patch_pooling.build(patch_shape)

        # Global layers expect patch representations
        global_input_shape = (None, self.config.max_patches, self.config.hidden_size)
        for layer in self.global_layers:
            layer.build(global_input_shape)

        # Cross-attention
        self.cross_attention.build(global_input_shape)

        super().build(input_shape)

    def call(
        self,
        embeddings: keras.KerasTensor,
        byte_tokens: keras.KerasTensor,  # Now explicitly require byte tokens
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Hierarchical encoding with dynamic patching.

        Args:
            embeddings: Byte embeddings of shape (batch_size, seq_len, local_hidden_size).
            byte_tokens: Original byte token IDs for entropy computation.
            attention_mask: Optional attention mask.
            training: Whether in training mode.

        Returns:
            Tuple of (sequence_output, patch_output) where:
            - sequence_output: Final byte-level representations
            - patch_output: Global patch representations
        """
        # Compute entropy for dynamic patching using actual byte tokens
        entropy_logits = self.entropy_model(byte_tokens, training=training)
        entropy = self.entropy_model.compute_entropy(entropy_logits)

        # Create dynamic patches
        patch_lengths = self.patcher(entropy, training=training)
        patch_ids = self.patcher.compute_patch_ids(patch_lengths)

        # Local encoding within patches
        local_output = self.local_encoder(
            byte_tokens, patch_ids, training=training
        )

        # Pool bytes to patch representations
        patch_representations = self.patch_pooling(
            local_output, patch_ids, training=training
        )

        # Global processing of patch representations
        global_output = patch_representations
        for layer in self.global_layers:
            global_output = layer(
                global_output,
                attention_mask=attention_mask,
                training=training
            )

        # Cross-attention to get final sequence output
        sequence_output = self.cross_attention(
            query=embeddings,  # Original byte embeddings as queries
            key=global_output,  # Global patch representations as keys
            value=global_output,  # Global patch representations as values
            attention_mask=attention_mask,
            training=training
        )

        return sequence_output, global_output

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({'config': self.config.to_dict()})
        return config


@keras.saving.register_keras_serializable()
class ByteBERT(keras.Model):
    """
    Byte-BERT: BERT with Byte Latent Transformer features.

    This model combines BERT's bidirectional understanding with BLT's efficient
    byte-level processing, creating a robust and language-agnostic text encoder.

    Key innovations:
    - Operates directly on UTF-8 bytes for universal language support
    - Uses dynamic entropy-based patching for efficient compute allocation
    - Employs hierarchical processing with local and global attention
    - Incorporates hash n-gram embeddings for rich byte representations
    - Maintains bidirectional attention for understanding tasks

    Args:
        config: ByteBertConfig containing all hyperparameters.
        add_pooling_layer: Whether to add pooling layer for classification.
        **kwargs: Additional model arguments.
    """

    def __init__(
        self,
        config: ByteBertConfig,
        add_pooling_layer: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        config.validate()
        self.config = config
        self.add_pooling_layer = add_pooling_layer

        # Initialize components to None - created in build()
        self.embeddings = None
        self.encoder = None
        self.pooler = None

        self._build_input_shape = None

        logger.info(f"Created Byte-BERT with {config.num_layers} global layers, "
                   f"{config.num_local_layers} local layers, hierarchical processing enabled")

    def build(self, input_shape: Union[Tuple[Optional[int], ...], Dict[str, Tuple[Optional[int], ...]]]) -> None:
        """Build the complete Byte-BERT model."""
        if self.built:
            return

        self._build_input_shape = input_shape

        # Handle different input formats
        if isinstance(input_shape, dict):
            byte_ids_shape = input_shape.get('input_ids', input_shape.get('inputs'))
        else:
            byte_ids_shape = input_shape

        logger.info(f"Building Byte-BERT with input shape: {byte_ids_shape}")

        # Create embeddings layer
        self.embeddings = ByteBertEmbeddings(
            config=self.config,
            name="embeddings"
        )
        self.embeddings.build(byte_ids_shape)

        # Create encoder
        embeddings_shape = (*byte_ids_shape, self.config.local_hidden_size)
        self.encoder = ByteBertEncoder(
            config=self.config,
            name="encoder"
        )
        self.encoder.build(embeddings_shape)

        # Create pooler if needed
        if self.add_pooling_layer:
            self.pooler = layers.Dense(
                units=self.config.hidden_size,
                activation="tanh",
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.config.initializer_range
                ),
                name="pooler"
            )
            pooler_input_shape = (None, self.config.hidden_size)
            self.pooler.build(pooler_input_shape)

        super().build(input_shape)
        logger.info("Byte-BERT model built successfully")

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        attention_mask: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        return_dict: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor], Dict[str, keras.KerasTensor]]:
        """
        Forward pass of Byte-BERT.

        Args:
            inputs: Byte token IDs or dictionary of inputs.
            attention_mask: Mask for attention computation.
            position_ids: Position IDs for embeddings.
            training: Whether in training mode.
            return_dict: Whether to return outputs as dictionary.

        Returns:
            Model outputs in requested format.
        """
        # Parse inputs
        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", attention_mask)
            position_ids = inputs.get("position_ids", position_ids)
        else:
            input_ids = inputs

        # Get embeddings with hash n-gram enhancement
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            training=training
        )

        # Hierarchical encoding (now properly passing byte tokens)
        sequence_output, patch_output = self.encoder(
            embeddings=embedding_output,
            byte_tokens=input_ids,  # Pass original byte tokens
            attention_mask=attention_mask,
            training=training
        )

        # Apply pooling if available
        pooled_output = None
        if self.pooler is not None:
            # Pool first token representation
            first_token_tensor = sequence_output[:, 0]
            pooled_output = self.pooler(first_token_tensor)

        # Return in requested format
        if return_dict:
            outputs = {
                "last_hidden_state": sequence_output,
                "patch_representations": patch_output,
            }
            if pooled_output is not None:
                outputs["pooler_output"] = pooled_output
            return outputs
        else:
            if pooled_output is not None:
                return sequence_output, pooled_output
            else:
                return sequence_output

    def encode_text(
        self,
        text: str,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True
    ) -> keras.KerasTensor:
        """
        Encode text to byte tokens using the ByteTokenizer.

        Args:
            text: Input text string.
            max_length: Maximum sequence length.
            add_special_tokens: Whether to add BOS/EOS tokens.

        Returns:
            Byte token IDs tensor of shape (1, seq_len).
        """
        return self.embeddings.encode_text(text, max_length)

    def decode_tokens(self, token_ids: keras.KerasTensor) -> str:
        """
        Decode byte token IDs back to text.

        Args:
            token_ids: Byte token IDs tensor.

        Returns:
            Decoded text string.
        """
        return self.embeddings.decode_tokens(token_ids)

    def encode_and_predict(
        self,
        text: str,
        max_length: Optional[int] = None,
        return_dict: bool = True
    ) -> Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """
        Convenience method to encode text and get predictions.

        Args:
            text: Input text string.
            max_length: Maximum sequence length.
            return_dict: Whether to return as dictionary.

        Returns:
            Model outputs for the encoded text.
        """
        # Encode text to byte tokens
        token_ids = self.encode_text(text, max_length)

        # Get predictions
        outputs = self(token_ids, return_dict=return_dict, training=False)

        return outputs

    def get_byte_tokenizer(self) -> ByteTokenizer:
        """Get the underlying ByteTokenizer."""
        return self.embeddings.tokenizer

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'config': self.config.to_dict(),
            'add_pooling_layer': self.add_pooling_layer,
        })
        return config


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_byte_bert_base() -> ByteBertConfig:
    """Create base Byte-BERT configuration."""
    return ByteBertConfig(
        vocab_size=260,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072,
        local_hidden_size=384,
        num_local_layers=4,
        num_local_heads=6
    )


def create_byte_bert_large() -> ByteBertConfig:
    """Create large Byte-BERT configuration."""
    return ByteBertConfig(
        vocab_size=260,
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        intermediate_size=4096,
        local_hidden_size=512,
        num_local_layers=6,
        num_local_heads=8,
        max_sequence_length=4096,
        max_patches=1024
    )


def create_byte_bert_for_classification(
    config: ByteBertConfig,
    num_labels: int,
    classifier_dropout: Optional[float] = None
) -> keras.Model:
    """
    Create Byte-BERT model for classification tasks.

    Args:
        config: Byte-BERT configuration.
        num_labels: Number of classification labels.
        classifier_dropout: Dropout rate for classifier head.

    Returns:
        Complete classification model.
    """
    logger.info(f"Creating Byte-BERT classification model with {num_labels} labels")

    # Create base model
    bert = ByteBERT(config=config, add_pooling_layer=True, name="byte_bert")

    # Define inputs
    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(shape=(None,), dtype="int32", name="attention_mask")

    # Get BERT outputs
    bert_outputs = bert(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        return_dict=True
    )

    # Classification head
    pooled_output = bert_outputs["pooler_output"]

    if classifier_dropout is None:
        classifier_dropout = config.classifier_dropout or config.hidden_dropout_prob

    if classifier_dropout > 0.0:
        pooled_output = layers.Dropout(
            classifier_dropout, name="classifier_dropout"
        )(pooled_output)

    # Final classification layer
    logits = layers.Dense(
        units=num_labels,
        kernel_initializer=keras.initializers.TruncatedNormal(
            stddev=config.initializer_range
        ),
        name="classifier"
    )(pooled_output)

    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=logits,
        name="byte_bert_for_classification"
    )

    logger.info(f"Created Byte-BERT classification model")
    return model


def create_robust_byte_bert() -> ByteBertConfig:
    """Create Byte-BERT optimized for noisy/corrupted text."""
    config = create_byte_bert_base()

    # Optimize for robustness
    config.use_hash_embeddings = True
    config.entropy_threshold = 1.8  # More conservative patching
    config.patch_pooling_method = 'attention'  # Better aggregation
    config.hidden_dropout_prob = 0.15  # Higher regularization
    config.attention_probs_dropout_prob = 0.15

    logger.info("Created robust Byte-BERT configuration")
    return config

