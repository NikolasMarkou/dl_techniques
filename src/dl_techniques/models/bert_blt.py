"""
BertBlt: BERT with Byte Latent Transformer Features

This module implements a BERT model that incorporates key innovations from the
Byte Latent Transformer (BLT), creating a powerful byte-level bidirectional
encoder that combines BERT's understanding capabilities with BLT's efficiency
and robustness advantages of byte-level processing.

Key Features:
=============

1. **Byte-Level Processing**: Operates directly on UTF-8 bytes instead of
   traditional subword tokens, providing language-agnostic capabilities and
   enhanced robustness to noisy or corrupted text.

2. **Hash N-Gram Embeddings**: Incorporates contextual byte patterns (3-8 grams)
   using rolling polynomial hash functions for enhanced representations.

3. **Bidirectional Understanding**: Maintains BERT's bidirectional attention
   pattern for better understanding tasks.

4. **Self-Contained Architecture**: All components are implemented within this
   module without external dependencies, following the dl-techniques patterns.

Architecture Overview:
=====================

Input Bytes → [Byte Embeddings + Hash N-gram Embeddings + Position Embeddings] →
[Layer Normalization + Dropout] → [Stack of Transformer Layers] → [Optional Pooling]

This creates a model that:
- Processes text at the fundamental byte level
- Maintains full bidirectional understanding
- Provides enhanced multilingual and noise robustness
- Uses familiar BERT architecture with byte-level improvements
"""

import keras
from keras import ops
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, List, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from ..layers.norms.rms_norm import RMSNorm
from ..layers.norms.band_rms import BandRMS
from ..layers.transformer import TransformerLayer


# ---------------------------------------------------------------------

@dataclass
class BertBltConfig:
    """
    Configuration class for BertBlt model parameters.

    This dataclass contains all the hyperparameters for the BertBlt model,
    extending traditional BERT parameters with byte-level processing features.

    Attributes:
        vocab_size: Size of byte vocabulary (typically 256 + special tokens).
        hidden_size: Hidden dimension of the transformer layers.
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads in each transformer layer.
        intermediate_size: Size of the intermediate layer in the feed-forward network.
        hidden_act: Activation function for the feed-forward network.
        hidden_dropout_prob: Dropout probability for hidden layers.
        attention_probs_dropout_prob: Dropout probability for attention weights.
        max_position_embeddings: Maximum sequence length for positional embeddings.
        initializer_range: Standard deviation for weight initialization.
        layer_norm_eps: Epsilon value for normalization layers.

        # BLT-specific parameters
        use_hash_embeddings: Whether to use hash n-gram embeddings.
        hash_vocab_size: Size of hash embedding vocabulary.
        ngram_sizes: List of n-gram sizes for hash embeddings.
        hash_embedding_dim: Dimension for hash embeddings (if different from hidden_size).

        # Advanced features
        normalization_type: Type of normalization layer to use.
        normalization_position: Position of normalization ('pre' or 'post').
        attention_type: Type of attention mechanism.
        ffn_type: Type of feed-forward network architecture.
        use_stochastic_depth: Whether to enable stochastic depth regularization.
        stochastic_depth_rate: Drop path rate for stochastic depth.
        classifier_dropout: Dropout probability for classification head.
    """
    vocab_size: int = 260  # 256 bytes + special tokens
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12

    # BLT-specific parameters
    use_hash_embeddings: bool = True
    hash_vocab_size: int = 500000
    ngram_sizes: Optional[List[int]] = None
    hash_embedding_dim: Optional[int] = None

    # Advanced features
    normalization_type: str = "layer_norm"
    normalization_position: str = "post"
    attention_type: str = "multi_head_attention"
    ffn_type: str = "mlp"
    use_stochastic_depth: bool = False
    stochastic_depth_rate: float = 0.1
    classifier_dropout: Optional[float] = None

    def __post_init__(self):
        """Initialize default values after creation."""
        if self.ngram_sizes is None:
            self.ngram_sizes = [3, 4, 5, 6, 7, 8]
        if self.hash_embedding_dim is None:
            self.hash_embedding_dim = self.hidden_size

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if not (0.0 <= self.hidden_dropout_prob <= 1.0):
            raise ValueError(f"hidden_dropout_prob must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BertBltConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ByteTokenizer(keras.layers.Layer):
    """
    Self-contained byte tokenizer for converting text to byte tokens.

    This layer handles the conversion between text and byte token IDs,
    supporting special tokens for sequence boundaries.

    Args:
        vocab_size: Size of vocabulary including special tokens.
        byte_offset: Offset for byte values (to reserve space for special tokens).
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        vocab_size: int = 260,
        byte_offset: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.byte_offset = byte_offset

        # Special token IDs
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

    def text_to_bytes(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True
    ) -> List[int]:
        """
        Convert text to byte token IDs.

        Args:
            text: Input text string.
            add_bos: Whether to add beginning-of-sequence token.
            add_eos: Whether to add end-of-sequence token.

        Returns:
            List of byte token IDs.
        """
        # Convert text to UTF-8 bytes
        byte_values = text.encode('utf-8')

        # Convert bytes to token IDs (offset by byte_offset)
        token_ids = [b + self.byte_offset for b in byte_values]

        # Add special tokens
        if add_bos:
            token_ids = [self.bos_token_id] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_token_id]

        return token_ids

    def tokens_to_text(self, token_ids: List[int]) -> str:
        """
        Convert byte token IDs back to text.

        Args:
            token_ids: List of byte token IDs.

        Returns:
            Decoded text string.
        """
        # Filter out special tokens and convert back to bytes
        byte_values = []
        for token_id in token_ids:
            if token_id >= self.byte_offset:
                byte_value = token_id - self.byte_offset
                if 0 <= byte_value <= 255:
                    byte_values.append(byte_value)

        # Convert bytes back to text
        try:
            byte_string = bytes(byte_values)
            return byte_string.decode('utf-8', errors='ignore')
        except Exception:
            return ""

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'byte_offset': self.byte_offset,
        })
        return config


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class HashNGramEmbedding(keras.layers.Layer):
    """
    Hash-based n-gram embedding layer for enhanced byte representations.

    Uses rolling polynomial hash functions to map byte n-grams to embedding
    vectors without requiring explicit vocabulary storage.

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

        # Validate parameters
        if hash_vocab_size <= 0:
            raise ValueError(f"hash_vocab_size must be positive, got {hash_vocab_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if not ngram_sizes or any(n <= 0 for n in ngram_sizes):
            raise ValueError(f"ngram_sizes must be non-empty with positive values, got {ngram_sizes}")

        self.hash_vocab_size = hash_vocab_size
        self.embed_dim = embed_dim
        self.ngram_sizes = ngram_sizes

        # Hash embedding tables - created in build()
        self.hash_embeddings = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build hash embedding tables."""
        if self.built:
            return

        # Create embedding table for each n-gram size
        self.hash_embeddings = {}
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
        combined_embeddings = ops.zeros((batch_size, seq_len, self.embed_dim), dtype=self.compute_dtype)

        # Compute embeddings for each n-gram size
        for n in self.ngram_sizes:
            ngram_embeddings = self._compute_ngram_embeddings(inputs, n)
            combined_embeddings = combined_embeddings + ngram_embeddings

        # Average across n-gram sizes
        if self.ngram_sizes:
            combined_embeddings = combined_embeddings / len(self.ngram_sizes)

        return combined_embeddings

    def _compute_ngram_embeddings(
        self,
        inputs: keras.KerasTensor,
        n: int
    ) -> keras.KerasTensor:
        """Compute embeddings for specific n-gram size using polynomial hashing."""
        seq_len = ops.shape(inputs)[1]

        # Rolling polynomial hash with base 257 (larger than byte vocab)
        base = 257

        # Use a list to collect hash values for each position, then stack.
        # This is a graph-compatible way to handle the sequential computation.
        hash_values_list = []

        # Compute rolling hash for each position
        for i in range(seq_len):
            if i < n - 1:
                # Not enough context for full n-gram, use single byte's value as hash.
                # The hash must be taken modulo hash_vocab_size to prevent out-of-bounds access.
                current_hash = ops.cast(inputs[:, i], 'int32') % self.hash_vocab_size
            else:
                # Compute full n-gram hash using Horner's method with modulo at each step
                current_hash = ops.cast(inputs[:, i], 'int32')
                for j in range(1, n):
                    byte_val = ops.cast(inputs[:, i - j], 'int32')
                    # The modulo is applied at each step to keep the hash value bounded
                    current_hash = (current_hash * base + byte_val) % self.hash_vocab_size

            hash_values_list.append(current_hash)

        # Stack the list of hashes into a single tensor
        hash_values = ops.stack(hash_values_list, axis=1)

        # Look up embeddings from the corresponding table
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


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class BertBltEmbeddings(keras.layers.Layer):
    """
    BertBlt embeddings layer combining byte-level and hash n-gram representations.

    This layer creates rich byte representations by combining:
    - Direct byte embeddings for each byte value
    - Hash n-gram embeddings for contextual patterns
    - Positional embeddings for sequence information
    - Layer normalization and dropout for regularization

    Args:
        vocab_size: Size of byte vocabulary.
        hidden_size: Hidden dimension for embeddings.
        max_position_embeddings: Maximum sequence length.
        initializer_range: Standard deviation for weight initialization.
        layer_norm_eps: Epsilon value for normalization layers.
        hidden_dropout_prob: Dropout probability for embeddings.
        use_hash_embeddings: Whether to use hash n-gram embeddings.
        hash_vocab_size: Size of hash embedding vocabulary.
        ngram_sizes: List of n-gram sizes for hash embeddings.
        hash_embedding_dim: Dimension for hash embeddings.
        normalization_type: Type of normalization layer.
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        initializer_range: float,
        layer_norm_eps: float,
        hidden_dropout_prob: float,
        use_hash_embeddings: bool = True,
        hash_vocab_size: int = 500000,
        ngram_sizes: List[int] = [3, 4, 5, 6, 7, 8],
        hash_embedding_dim: int = 768,
        normalization_type: str = "layer_norm",
        **kwargs
    ):
        super().__init__(**kwargs)

        # Validate parameters
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if max_position_embeddings <= 0:
            raise ValueError(f"max_position_embeddings must be positive, got {max_position_embeddings}")

        # Store parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use_hash_embeddings = use_hash_embeddings
        self.hash_vocab_size = hash_vocab_size
        self.ngram_sizes = ngram_sizes
        self.hash_embedding_dim = hash_embedding_dim
        self.normalization_type = normalization_type

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)

        # Byte tokenizer for text processing
        self.tokenizer = ByteTokenizer(
            vocab_size=vocab_size,
            name="tokenizer"
        )

        # Direct byte embeddings
        self.byte_embeddings = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=initializer_range
            ),
            mask_zero=True,
            name="byte_embeddings"
        )

        # Position embeddings
        self.position_embeddings = keras.layers.Embedding(
            input_dim=max_position_embeddings,
            output_dim=hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=initializer_range
            ),
            name="position_embeddings"
        )

        # Hash n-gram embeddings if enabled
        self.hash_embeddings = None
        self.hash_projection = None
        if use_hash_embeddings:
            self.hash_embeddings = HashNGramEmbedding(
                hash_vocab_size=hash_vocab_size,
                embed_dim=hash_embedding_dim,
                ngram_sizes=ngram_sizes,
                name="hash_embeddings"
            )

            # Project hash embeddings to hidden_size if dimensions differ
            if hash_embedding_dim != hidden_size:
                self.hash_projection = keras.layers.Dense(
                    hidden_size,
                    use_bias=False,
                    kernel_initializer=keras.initializers.TruncatedNormal(
                        stddev=initializer_range
                    ),
                    name="hash_projection"
                )

        # Normalization layer
        self.layer_norm = self._create_normalization_layer("layer_norm")

        # Dropout
        self.dropout = keras.layers.Dropout(
            rate=hidden_dropout_prob,
            name="dropout"
        )

        logger.info(f"Created BertBltEmbeddings with hidden_size={hidden_size}, "
                   f"vocab_size={vocab_size}, hash_embeddings={use_hash_embeddings}")

    def _create_normalization_layer(self, name: str) -> keras.layers.Layer:
        """Create normalization layer based on type."""
        if self.normalization_type == 'layer_norm':
            return keras.layers.LayerNormalization(
                epsilon=self.layer_norm_eps,
                name=name
            )
        elif self.normalization_type == 'rms_norm':
            return RMSNorm(
                epsilon=self.layer_norm_eps,
                name=name
            )
        elif self.normalization_type == 'band_rms':
            return BandRMS(
                epsilon=self.layer_norm_eps,
                name=name
            )
        elif self.normalization_type == 'batch_norm':
            return keras.layers.BatchNormalization(
                epsilon=self.layer_norm_eps,
                name=name
            )
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization_type}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all embedding sublayers."""
        if self.built:
            return

        # CRITICAL: Explicitly build all sub-layers for robust serialization
        self.tokenizer.build(input_shape)
        self.byte_embeddings.build(input_shape)
        # Position embeddings input shape is the same as input_ids
        self.position_embeddings.build(input_shape)

        if self.hash_embeddings is not None:
            self.hash_embeddings.build(input_shape)

            if self.hash_projection is not None:
                hash_shape = (*input_shape, self.hash_embedding_dim)
                self.hash_projection.build(hash_shape)

        # Build normalization and dropout with embeddings output shape
        embeddings_output_shape = (*input_shape, self.hidden_size)
        self.layer_norm.build(embeddings_output_shape)
        self.dropout.build(embeddings_output_shape)

        super().build(input_shape)

    def call(
        self,
        input_ids: keras.KerasTensor,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply byte-level embeddings with optional hash n-gram enhancement.

        Args:
            input_ids: Byte token IDs of shape (batch_size, seq_len).
            position_ids: Position IDs of shape (batch_size, seq_len).
            training: Whether in training mode.

        Returns:
            Enhanced byte embeddings of shape (batch_size, seq_len, hidden_size).
        """
        input_shape = ops.shape(input_ids)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = ops.arange(seq_length, dtype="int32")
            position_ids = ops.expand_dims(position_ids, axis=0)
            position_ids = ops.broadcast_to(position_ids, (batch_size, seq_length))

        # Get base byte embeddings
        byte_embeds = self.byte_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)

        # Combine base embeddings
        embeddings = byte_embeds + position_embeds

        # Add hash n-gram embeddings if enabled
        if self.hash_embeddings is not None:
            hash_embeds = self.hash_embeddings(input_ids)

            # Project hash embeddings if dimension mismatch
            if self.hash_projection is not None:
                hash_embeds = self.hash_projection(hash_embeds)

            embeddings = embeddings + hash_embeds

        # Apply normalization and dropout
        embeddings = self.layer_norm(embeddings, training=training)
        embeddings = self.dropout(embeddings, training=training)

        return embeddings

    def encode_text(
        self,
        text: str,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True
    ) -> keras.KerasTensor:
        """
        Encode text to byte token IDs using the ByteTokenizer.

        Args:
            text: Input text string.
            max_length: Maximum sequence length.
            add_special_tokens: Whether to add BOS/EOS tokens.

        Returns:
            Byte token IDs tensor of shape (1, seq_len).
        """
        if max_length is None:
            max_length = self.max_position_embeddings

        # Convert text to byte tokens
        byte_tokens = self.tokenizer.text_to_bytes(
            text,
            add_bos=add_special_tokens,
            add_eos=add_special_tokens
        )

        # Pad or truncate to max_length
        if len(byte_tokens) > max_length:
            byte_tokens = byte_tokens[:max_length]
        else:
            # Pad with pad_token_id (0)
            byte_tokens.extend([self.tokenizer.pad_token_id] * (max_length - len(byte_tokens)))

        return ops.array([byte_tokens], dtype='int32')

    def decode_tokens(self, token_ids: keras.KerasTensor) -> str:
        """
        Decode byte token IDs back to text.

        Args:
            token_ids: Byte token IDs tensor.

        Returns:
            Decoded text string.
        """
        # Convert to numpy and get first sequence
        tokens = ops.convert_to_numpy(token_ids)[0].tolist()
        return self.tokenizer.tokens_to_text(tokens)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape given input shape."""
        return (*input_shape, self.hidden_size)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'max_position_embeddings': self.max_position_embeddings,
            'initializer_range': self.initializer_range,
            'layer_norm_eps': self.layer_norm_eps,
            'hidden_dropout_prob': self.hidden_dropout_prob,
            'use_hash_embeddings': self.use_hash_embeddings,
            'hash_vocab_size': self.hash_vocab_size,
            'ngram_sizes': self.ngram_sizes,
            'hash_embedding_dim': self.hash_embedding_dim,
            'normalization_type': self.normalization_type,
        })
        return config


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class BertBlt(keras.Model):
    """
    BertBlt: BERT with Byte Latent Transformer features.

    This model combines BERT's bidirectional understanding with BLT's efficient
    byte-level processing, creating a robust and language-agnostic text encoder.

    Key features:
    - Operates directly on UTF-8 bytes for universal language support
    - Uses hash n-gram embeddings for rich byte representations
    - Maintains bidirectional attention for understanding tasks
    - Self-contained implementation following dl-techniques patterns

    Args:
        config: BertBltConfig containing all hyperparameters.
        add_pooling_layer: Whether to add pooling layer for classification.
        **kwargs: Additional model arguments.
    """

    def __init__(
        self,
        config: BertBltConfig,
        add_pooling_layer: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Validate and store configuration
        config.validate()
        self.config = config
        self.add_pooling_layer = add_pooling_layer

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)

        # Embeddings layer with byte-level processing
        self.embeddings = BertBltEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            initializer_range=config.initializer_range,
            layer_norm_eps=config.layer_norm_eps,
            hidden_dropout_prob=config.hidden_dropout_prob,
            use_hash_embeddings=config.use_hash_embeddings,
            hash_vocab_size=config.hash_vocab_size,
            ngram_sizes=config.ngram_sizes,
            hash_embedding_dim=config.hash_embedding_dim,
            normalization_type=config.normalization_type,
            name="embeddings"
        )

        # Transformer encoder layers using existing TransformerLayer
        self.encoder_layers: List[TransformerLayer] = []
        for i in range(config.num_layers):
            transformer_layer = TransformerLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                intermediate_size=config.intermediate_size,
                normalization_type=config.normalization_type,
                normalization_position=config.normalization_position,
                attention_type=config.attention_type,
                ffn_type=config.ffn_type,
                dropout_rate=config.hidden_dropout_prob,
                attention_dropout_rate=config.attention_probs_dropout_prob,
                use_stochastic_depth=config.use_stochastic_depth,
                stochastic_depth_rate=config.stochastic_depth_rate,
                activation=config.hidden_act,
                use_bias=True,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=config.initializer_range
                ),
                bias_initializer="zeros",
                name=f"encoder_layer_{i}"
            )
            self.encoder_layers.append(transformer_layer)

        # Pooler for classification tasks
        self.pooler = None
        if add_pooling_layer:
            self.pooler = keras.layers.Dense(
                units=config.hidden_size,
                activation="tanh",
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=config.initializer_range
                ),
                name="pooler"
            )

        logger.info(f"Created BertBlt model with {config.num_layers} layers, "
                   f"hidden_size={config.hidden_size}, byte_vocab={config.vocab_size}, "
                   f"hash_embeddings={config.use_hash_embeddings}")

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        attention_mask: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        return_dict: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor], Dict[str, keras.KerasTensor]]:
        """
        Forward pass of BertBlt model.

        Args:
            inputs: Byte token IDs or dictionary containing inputs.
            attention_mask: Mask to avoid attention on padding tokens.
            position_ids: Position IDs for positional embeddings.
            training: Whether the model is in training mode.
            return_dict: Whether to return outputs as a dictionary.

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

        # Get byte-level embeddings with hash enhancement
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            training=training
        )

        # Pass through transformer encoder layers
        hidden_states = embedding_output
        for i, encoder_layer in enumerate(self.encoder_layers):
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_idx=i,  # For differential attention
                training=training
            )

        sequence_output = hidden_states

        # Apply pooling if available
        pooled_output = None
        if self.pooler is not None:
            # Pool the representation of the first token
            first_token_tensor = sequence_output[:, 0]  # (batch_size, hidden_size)
            pooled_output = self.pooler(first_token_tensor)

        # Return in requested format
        if return_dict:
            outputs = {
                "last_hidden_state": sequence_output,
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
        Encode text to byte tokens.

        Args:
            text: Input text string.
            max_length: Maximum sequence length.
            add_special_tokens: Whether to add BOS/EOS tokens.

        Returns:
            Byte token IDs tensor of shape (1, seq_len).
        """
        return self.embeddings.encode_text(text, max_length, add_special_tokens)

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

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = {
            'config': self.config.to_dict(),
            'add_pooling_layer': self.add_pooling_layer,
        }
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BertBlt':
        """Create model from configuration."""
        bert_config = BertBltConfig.from_dict(config['config'])
        return cls(
            config=bert_config,
            add_pooling_layer=config.get('add_pooling_layer', True)
        )


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_bert_blt_base() -> BertBltConfig:
    """
    Create base BertBlt configuration.

    Returns:
        BertBltConfig configured for the base model size.
    """
    return BertBltConfig(
        vocab_size=260,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072,
        use_hash_embeddings=True,
        hash_vocab_size=500000,
        ngram_sizes=[3, 4, 5, 6, 7, 8]
    )


def create_bert_blt_large() -> BertBltConfig:
    """
    Create large BertBlt configuration.

    Returns:
        BertBltConfig configured for the large model size.
    """
    return BertBltConfig(
        vocab_size=260,
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        intermediate_size=4096,
        max_position_embeddings=4096,
        use_hash_embeddings=True,
        hash_vocab_size=1000000,
        ngram_sizes=[3, 4, 5, 6, 7, 8]
    )


def create_bert_blt_for_classification(
    config: BertBltConfig,
    num_labels: int,
    classifier_dropout: Optional[float] = None
) -> keras.Model:
    """
    Create BertBlt model for classification tasks.

    Args:
        config: BertBlt configuration object.
        num_labels: Number of classification labels.
        classifier_dropout: Dropout rate for the classifier head.

    Returns:
        Complete BertBlt model for classification.

    Example:
        ```python
        config = create_bert_blt_base()
        model = create_bert_blt_for_classification(config, num_labels=2)

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        ```
    """
    logger.info(f"Creating BertBlt classification model with {num_labels} labels")

    # Create base BertBlt model with pooling
    bert = BertBlt(config=config, add_pooling_layer=True, name="bert_blt")

    # Define inputs
    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(shape=(None,), dtype="int32", name="attention_mask")

    # Get BERT outputs
    bert_outputs = bert(
        inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        return_dict=True
    )

    # Classification head
    pooled_output = bert_outputs["pooler_output"]

    # Apply classifier dropout
    if classifier_dropout is None:
        classifier_dropout = config.classifier_dropout or config.hidden_dropout_prob

    if classifier_dropout > 0.0:
        pooled_output = keras.layers.Dropout(
            classifier_dropout,
            name="classifier_dropout"
        )(pooled_output)

    # Final classification layer
    logits = keras.layers.Dense(
        units=num_labels,
        kernel_initializer=keras.initializers.TruncatedNormal(
            stddev=config.initializer_range
        ),
        name="classifier"
    )(pooled_output)

    # Create the final model
    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=logits,
        name="bert_blt_for_classification"
    )

    logger.info(f"Created BertBlt classification model with {model.count_params()} parameters")
    return model


def create_bert_blt_for_sequence_output(
    config: BertBltConfig
) -> keras.Model:
    """
    Create BertBlt model for sequence-level output tasks.

    Args:
        config: BertBlt configuration object.

    Returns:
        BertBlt model outputting sequence-level representations.

    Example:
        ```python
        config = create_bert_blt_base()
        model = create_bert_blt_for_sequence_output(config)

        # For token classification, add a classification head
        num_tags = 9  # e.g., for NER
        sequence_output = model.output
        logits = keras.layers.Dense(num_tags)(sequence_output)
        token_classifier = keras.Model(model.input, logits)
        ```
    """
    logger.info("Creating BertBlt model for sequence output tasks")

    # Create base BertBlt model without pooling
    bert = BertBlt(config=config, add_pooling_layer=False, name="bert_blt")

    # Define inputs
    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(shape=(None,), dtype="int32", name="attention_mask")

    # Get BERT sequence output
    sequence_output = bert(
        inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )

    # Create the final model
    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=sequence_output,
        name="bert_blt_for_sequence_output"
    )

    logger.info(f"Created BertBlt sequence model with {model.count_params()} parameters")
    return model


def create_robust_bert_blt() -> BertBltConfig:
    """
    Create BertBlt configuration optimized for noisy/corrupted text.

    Returns:
        BertBltConfig optimized for robustness to text corruptions.
    """
    config = create_bert_blt_base()

    # Optimize for robustness
    config.use_hash_embeddings = True
    config.hash_vocab_size = 1000000  # Larger hash space
    config.ngram_sizes = [2, 3, 4, 5, 6, 7, 8]  # Include bigrams
    config.hidden_dropout_prob = 0.15  # Higher regularization
    config.attention_probs_dropout_prob = 0.15

    logger.info("Created robust BertBlt configuration")
    return config