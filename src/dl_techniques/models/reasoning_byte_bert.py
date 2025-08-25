"""
ReasoningByteBERT: Combining ByteBERT with Hierarchical Reasoning

This module creates a powerful combination of ByteBERT's byte-level processing
capabilities with the HierarchicalReasoningModel's iterative reasoning and
Adaptive Computation Time (ACT) mechanisms.

Key Innovations:
===============

1. **Byte-Level Reasoning**: Operates on UTF-8 bytes with dynamic patching
   while performing iterative hierarchical reasoning.

2. **Adaptive Computation on Bytes**: Uses ACT to dynamically allocate
   computation time for complex byte sequences.

3. **Hierarchical Byte Processing**:
   - ByteBERT embeddings with hash n-grams
   - Dynamic entropy-based patching
   - HRM-style iterative reasoning on patch representations
   - Cross-attention between byte and reasoning levels

4. **Multi-Level State Management**:
   - Byte-level states for local processing
   - Patch-level states for global reasoning
   - Hierarchical reasoning states (z_h, z_l)

5. **Task-Aware Byte Processing**: Combines puzzle embeddings with byte
   representations for task-specific reasoning.

Architecture Flow:
=================
Input Bytes → [Hash N-gram + Byte Embeddings] → [Dynamic Patching] →
[Local Byte Processing] → [Hierarchical Reasoning Core] →
[Adaptive Computation Time] → [Task Outputs + Halt Decisions]
"""

import math
import keras
from keras import ops
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, List, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from ..layers.norms.rms_norm import RMSNorm
from ..layers.byte_latent_transformer_blocks import (
    ByteTokenizer, EntropyModel, DynamicPatcher,
    LocalEncoder, PatchPooling
)
from ..layers.positional_embedding import PositionalEmbedding
from ..layers.hrm_reasoning_module import HierarchicalReasoningModule
from ..layers.hrm_sparse_puzzle_embedding import SparsePuzzleEmbedding
from ..layers.rotary_position_embedding import RotaryPositionEmbedding

# ---------------------------------------------------------------------

@dataclass
class ReasoningByteBertConfig:
    """
    Configuration for ReasoningByteBERT combining ByteBERT and HRM features.

    This configuration merges byte-level processing parameters with
    hierarchical reasoning and adaptive computation settings.

    Attributes:
        # Core parameters
        vocab_size: Size of byte vocabulary (typically 256 + special tokens).
        seq_len: Maximum sequence length in bytes.
        embed_dim: Embedding dimension for global processing.

        # Byte-level processing (from ByteBERT)
        local_hidden_size: Hidden dimension for local byte processing.
        max_patches: Maximum number of patches per sequence.
        entropy_threshold: Threshold for dynamic patch boundary detection.
        patch_pooling_method: Method for aggregating bytes into patches.
        cross_attention_queries: Number of query vectors for cross-attention.

        # Hash embedding parameters
        use_hash_embeddings: Whether to use hash n-gram embeddings.
        hash_vocab_size: Size of hash embedding vocabulary.
        ngram_sizes: List of n-gram sizes for hash embeddings.

        # Hierarchical reasoning (from HRM)
        num_puzzle_identifiers: Number of puzzle identifiers.
        puzzle_emb_dim: Puzzle embedding dimension (0 to disable).
        h_layers: Number of high-level reasoning layers.
        l_layers: Number of low-level reasoning layers.
        h_cycles: Number of high-level reasoning cycles.
        l_cycles: Number of low-level reasoning cycles.
        num_heads: Number of attention heads.
        ffn_expansion_factor: Feed-forward expansion factor.

        # Adaptive computation
        halt_max_steps: Maximum computation steps before forced halt.
        halt_exploration_prob: Probability of exploration in Q-learning.

        # Training parameters
        batch_size: Batch size for training.
        dropout_rate: Dropout rate.
        hidden_dropout_prob: Dropout probability for hidden layers.
        attention_probs_dropout_prob: Dropout probability for attention.

        # Architecture parameters
        pos_encodings: Type of positional encodings ("rope" or "learned").
        rope_theta: RoPE theta parameter.
        normalization_type: Type of normalization ('layer_norm', 'rms_norm').
        hidden_act: Activation function for feed-forward networks.
        use_bias: Whether to use bias in linear layers.
        initializer_range: Standard deviation for weight initialization.
        layer_norm_eps: Epsilon for normalization layers.
    """
    # Core parameters
    vocab_size: int = 260  # 256 bytes + special tokens
    seq_len: int = 2048
    embed_dim: int = 768

    # Byte-level processing
    local_hidden_size: int = 384
    max_patches: int = 512
    entropy_threshold: float = 1.5
    patch_pooling_method: str = 'attention'
    cross_attention_queries: int = 4

    # Hash embedding parameters
    use_hash_embeddings: bool = True
    hash_vocab_size: int = 500000
    ngram_sizes: List[int] = None

    # Hierarchical reasoning
    num_puzzle_identifiers: int = 1000
    puzzle_emb_dim: int = 512
    h_layers: int = 4
    l_layers: int = 4
    h_cycles: int = 2
    l_cycles: int = 2
    num_heads: int = 12
    ffn_expansion_factor: int = 4

    # Adaptive computation
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.1

    # Training parameters
    batch_size: int = 32
    dropout_rate: float = 0.1
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # Architecture parameters
    pos_encodings: str = "rope"
    rope_theta: float = 10000.0
    normalization_type: str = "rms_norm"
    hidden_act: str = "gelu"
    use_bias: bool = False
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12

    def __post_init__(self):
        """Initialize default values after creation."""
        if self.ngram_sizes is None:
            self.ngram_sizes = [3, 4, 5, 6, 7, 8]

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.local_hidden_size <= 0:
            raise ValueError(f"local_hidden_size must be positive, got {self.local_hidden_size}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ReasoningByteBertConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

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
        self.hash_vocab_size = hash_vocab_size
        self.embed_dim = embed_dim
        self.ngram_sizes = ngram_sizes
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

        # Rolling polynomial hash with base 257
        base = 257

        # Initialize hash values
        hash_values = ops.zeros((batch_size, seq_len), dtype='int32')

        # Compute rolling hash for each position
        for i in range(seq_len):
            if i < n - 1:
                current_hash = ops.cast(inputs[:, i], 'int32')
            else:
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

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ReasoningByteEmbeddings(keras.layers.Layer):
    """
    ReasoningByte embeddings combining byte-level processing with puzzle context.

    This layer creates rich representations by combining:
    - ByteTokenizer for proper byte-level tokenization
    - Direct byte embeddings for each byte value
    - Hash n-gram embeddings for contextual patterns
    - Puzzle embeddings for task-specific context
    - Positional embeddings for sequence information

    Args:
        config: ReasoningByteBertConfig containing all hyperparameters.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, config: ReasoningByteBertConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Initialize ByteTokenizer
        self.tokenizer = ByteTokenizer(
            vocab_size=config.vocab_size,
            name="byte_tokenizer"
        )

        # Initialize sublayers to None - created in build()
        self.byte_embeddings = None
        self.position_embeddings = None
        self.rope = None
        self.hash_embeddings = None
        self.puzzle_embedding = None
        self.layer_norm = None
        self.dropout = None

        # Calculate puzzle embedding sequence length
        self.puzzle_emb_len = max(1, (
                    config.puzzle_emb_dim + config.embed_dim - 1) // config.embed_dim) if config.puzzle_emb_dim > 0 else 0
        self.total_seq_len = config.seq_len + self.puzzle_emb_len

        # Embedding scale
        self.embed_scale = math.sqrt(config.embed_dim)

        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all embedding sublayers."""
        if self.built:
            return

        self._build_input_shape = input_shape

        # Build ByteTokenizer
        self.tokenizer.build(input_shape)

        # Direct byte embeddings
        self.byte_embeddings = keras.layers.Embedding(
            input_dim=self.tokenizer.vocab_size,
            output_dim=self.config.embed_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=self.config.initializer_range
            ),
            mask_zero=True,
            name="byte_embeddings"
        )

        # Puzzle embeddings (if enabled)
        if self.config.puzzle_emb_dim > 0:
            self.puzzle_embedding = SparsePuzzleEmbedding(
                num_embeddings=self.config.num_puzzle_identifiers,
                embedding_dim=self.config.puzzle_emb_dim,
                batch_size=self.config.batch_size,
                embeddings_initializer="zeros",
                name="puzzle_embedding"
            )

        # Positional embeddings
        if self.config.pos_encodings == "rope":
            self.rope = RotaryPositionEmbedding(
                head_dim=self.config.embed_dim // self.config.num_heads,
                max_seq_len=self.total_seq_len,
                rope_theta=self.config.rope_theta,
                name="rope"
            )
        elif self.config.pos_encodings == "learned":
            self.position_embeddings = PositionalEmbedding(
                max_seq_len=self.total_seq_len,
                dim=self.config.embed_dim,
                dropout=0.0,
                name="position_embeddings"
            )

        # Hash n-gram embeddings if enabled
        if self.config.use_hash_embeddings:
            self.hash_embeddings = HashNGramEmbedding(
                hash_vocab_size=self.config.hash_vocab_size,
                embed_dim=self.config.embed_dim,
                ngram_sizes=self.config.ngram_sizes,
                name="hash_embeddings"
            )

        # Normalization and dropout
        if self.config.normalization_type == 'layer_norm':
            self.layer_norm = keras.layers.LayerNormalization(
                epsilon=self.config.layer_norm_eps,
                name="layer_norm"
            )
        elif self.config.normalization_type == 'rms_norm':
            self.layer_norm = RMSNorm(
                epsilon=self.config.layer_norm_eps,
                name="layer_norm"
            )

        self.dropout = keras.layers.Dropout(
            rate=self.config.hidden_dropout_prob,
            name="dropout"
        )

        # Build all sublayers
        self.byte_embeddings.build(input_shape)

        if self.puzzle_embedding is not None:
            puzzle_input_shape = (None,)  # Single puzzle ID per sequence
            self.puzzle_embedding.build(puzzle_input_shape)

        if self.position_embeddings is not None:
            self.position_embeddings.build(input_shape)

        if self.hash_embeddings is not None:
            self.hash_embeddings.build(input_shape)

        embeddings_shape = (*input_shape, self.config.embed_dim)
        self.layer_norm.build(embeddings_shape)
        self.dropout.build(embeddings_shape)

        super().build(input_shape)

    def call(
            self,
            token_ids: keras.KerasTensor,
            puzzle_ids: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Create combined embeddings from tokens and puzzle context.

        Args:
            token_ids: Byte token IDs of shape (batch_size, seq_len).
            puzzle_ids: Puzzle IDs of shape (batch_size,).
            training: Whether in training mode.

        Returns:
            Enhanced embeddings of shape (batch_size, total_seq_len, embed_dim).
        """
        # Get base byte embeddings
        byte_embeds = self.byte_embeddings(token_ids)

        # Add puzzle embeddings if enabled
        if self.config.puzzle_emb_dim > 0 and puzzle_ids is not None:
            puzzle_emb = self.puzzle_embedding(puzzle_ids)

            # Pad and reshape puzzle embedding
            pad_size = self.puzzle_emb_len * self.config.embed_dim - self.config.puzzle_emb_dim
            if pad_size > 0:
                puzzle_emb = ops.pad(puzzle_emb, [[0, 0], [0, pad_size]])

            batch_size = ops.shape(byte_embeds)[0]
            puzzle_emb = ops.reshape(puzzle_emb, [batch_size, self.puzzle_emb_len, self.config.embed_dim])
            embeddings = ops.concatenate([puzzle_emb, byte_embeds], axis=1)
        else:
            embeddings = byte_embeds

        # Add positional embeddings
        if self.config.pos_encodings == "learned":
            # Scale by 1/sqrt(2) to maintain variance
            embeddings = 0.707106781 * (embeddings + self.position_embeddings.position_embeddings)

        # Add hash n-gram embeddings if enabled
        if self.hash_embeddings is not None:
            # Create padded token_ids for hash embeddings
            if self.puzzle_emb_len > 0:
                padded_tokens = ops.pad(token_ids, [[0, 0], [self.puzzle_emb_len, 0]], constant_values=0)
            else:
                padded_tokens = token_ids

            hash_embeds = self.hash_embeddings(padded_tokens)
            embeddings = embeddings + hash_embeds

        # Scale embeddings
        embeddings = self.embed_scale * embeddings

        # Apply normalization and dropout
        embeddings = self.layer_norm(embeddings, training=training)
        embeddings = self.dropout(embeddings, training=training)

        return embeddings

    def get_vocab_size(self) -> int:
        """Get vocabulary size from ByteTokenizer."""
        return self.tokenizer.vocab_size

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({'config': self.config.to_dict()})
        return config


@keras.saving.register_keras_serializable()
class ReasoningByteCore(keras.layers.Layer):
    """
    Core reasoning engine combining byte-level processing with hierarchical reasoning.

    This layer integrates:
    1. Dynamic entropy-based patching on byte sequences
    2. Local encoding within patches
    3. Hierarchical reasoning on patch representations
    4. Iterative refinement with cycles

    Args:
        config: ReasoningByteBertConfig containing all hyperparameters.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, config: ReasoningByteBertConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Initialize components to None - created in build()
        self.entropy_model = None
        self.patcher = None
        self.local_encoder = None
        self.patch_pooling = None
        self.h_reasoning = None
        self.l_reasoning = None
        self.lm_head = None
        self.q_head = None

        # Initial states for reasoning
        self.h_init = None
        self.l_init = None

        # Calculate sequence dimensions
        self.puzzle_emb_len = max(1, (
                    config.puzzle_emb_dim + config.embed_dim - 1) // config.embed_dim) if config.puzzle_emb_dim > 0 else 0
        self.total_seq_len = config.seq_len + self.puzzle_emb_len

        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all reasoning components."""
        if self.built:
            return

        self._build_input_shape = input_shape

        # Entropy model for dynamic patching
        self.entropy_model = EntropyModel(
            vocab_size=self.config.vocab_size,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            max_seq_len=self.config.seq_len,
            name='entropy_model'
        )

        # Dynamic patcher
        self.patcher = DynamicPatcher(
            entropy_threshold=self.config.entropy_threshold,
            max_patches=self.config.max_patches,
            name='patcher'
        )

        # Local encoder for byte-level processing
        self.local_encoder = LocalEncoder(
            vocab_size=self.config.vocab_size,
            local_dim=self.config.local_hidden_size,
            num_local_layers=4,
            num_heads_local=8,
            max_sequence_length=self.config.seq_len,
            max_patches=self.config.max_patches,
            dropout_rate=self.config.dropout_rate,
            patch_pooling_method=self.config.patch_pooling_method,
            global_dim=self.config.embed_dim,
            cross_attention_queries=self.config.cross_attention_queries,
            name='local_encoder'
        )

        # Patch pooling
        self.patch_pooling = PatchPooling(
            pooling_method=self.config.patch_pooling_method,
            output_dim=self.config.embed_dim,
            num_queries=self.config.cross_attention_queries,
            name='patch_pooling'
        )

        # Hierarchical reasoning modules
        self.h_reasoning = HierarchicalReasoningModule(
            num_layers=self.config.h_layers,
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            ffn_expansion_factor=self.config.ffn_expansion_factor,
            dropout_rate=self.config.dropout_rate,
            use_bias=self.config.use_bias,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.config.initializer_range
            ),
            name="h_reasoning"
        )

        self.l_reasoning = HierarchicalReasoningModule(
            num_layers=self.config.l_layers,
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            ffn_expansion_factor=self.config.ffn_expansion_factor,
            dropout_rate=self.config.dropout_rate,
            use_bias=self.config.use_bias,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.config.initializer_range
            ),
            name="l_reasoning"
        )

        # Output heads
        self.lm_head = keras.layers.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.config.initializer_range
            ),
            name="lm_head"
        )

        # Q head for halt decisions
        self.q_head = keras.layers.Dense(
            2,  # halt, continue
            use_bias=True,
            kernel_initializer="zeros",
            bias_initializer=keras.initializers.Constant(-5.0),
            name="q_head"
        )

        # Initial states
        self.h_init = self.add_weight(
            name="h_init",
            shape=(self.config.embed_dim,),
            initializer=keras.initializers.TruncatedNormal(stddev=1.0),
            trainable=True
        )

        self.l_init = self.add_weight(
            name="l_init",
            shape=(self.config.embed_dim,),
            initializer=keras.initializers.TruncatedNormal(stddev=1.0),
            trainable=True
        )

        # Build components with appropriate shapes
        token_input_shape = (None, self.config.seq_len)
        self.entropy_model.build(token_input_shape)
        self.patcher.build((None,))

        # Local encoder expects byte tokens
        self.local_encoder.build(token_input_shape)

        # Reasoning modules expect patch representations
        reasoning_input_shape = (None, self.config.max_patches, self.config.embed_dim)
        self.h_reasoning.build(reasoning_input_shape)
        self.l_reasoning.build(reasoning_input_shape)

        # Output heads
        self.lm_head.build((None, self.config.embed_dim))
        self.q_head.build((None, self.config.embed_dim))

        super().build(input_shape)

    def empty_carry(self, batch_size: int) -> Dict[str, keras.KerasTensor]:
        """Create empty carry state for reasoning."""
        return {
            "z_h": ops.zeros((batch_size, self.config.max_patches, self.config.embed_dim)),
            "z_l": ops.zeros((batch_size, self.config.max_patches, self.config.embed_dim))
        }

    def reset_carry(
            self,
            reset_flag: keras.KerasTensor,
            carry: Dict[str, keras.KerasTensor]
    ) -> Dict[str, keras.KerasTensor]:
        """Reset carry state for halted sequences."""
        batch_size = ops.shape(reset_flag)[0]

        # Broadcast initial states
        h_init_broadcast = ops.broadcast_to(
            ops.reshape(self.h_init, [1, 1, self.config.embed_dim]),
            [batch_size, self.config.max_patches, self.config.embed_dim]
        )
        l_init_broadcast = ops.broadcast_to(
            ops.reshape(self.l_init, [1, 1, self.config.embed_dim]),
            [batch_size, self.config.max_patches, self.config.embed_dim]
        )

        # Reset based on halt flag
        reset_flag = ops.reshape(reset_flag, [-1, 1, 1])
        new_z_h = ops.where(reset_flag, h_init_broadcast, carry["z_h"])
        new_z_l = ops.where(reset_flag, l_init_broadcast, carry["z_l"])

        return {"z_h": new_z_h, "z_l": new_z_l}

    def call(
            self,
            carry: Dict[str, keras.KerasTensor],
            embeddings: keras.KerasTensor,
            byte_tokens: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[Dict[str, keras.KerasTensor], Dict[str, keras.KerasTensor]]:
        """
        Perform hierarchical reasoning on byte-level representations.

        Args:
            carry: Current reasoning state with "z_h" and "z_l".
            embeddings: Combined embeddings with puzzle context.
            byte_tokens: Original byte token IDs for entropy computation.
            training: Whether in training mode.

        Returns:
            Tuple of (new_carry, outputs_dict).
        """
        # Extract puzzle-free byte tokens for entropy computation
        if self.puzzle_emb_len > 0:
            pure_byte_tokens = byte_tokens  # These should already be pure byte tokens
        else:
            pure_byte_tokens = byte_tokens

        # Compute entropy for dynamic patching
        entropy_logits = self.entropy_model(pure_byte_tokens, training=training)
        entropy = self.entropy_model.compute_entropy(entropy_logits)

        # Create dynamic patches
        patch_lengths = self.patcher(entropy, training=training)
        patch_ids = self.patcher.compute_patch_ids(patch_lengths)

        # Local byte encoding within patches
        local_output = self.local_encoder(
            pure_byte_tokens, patch_ids, training=training
        )

        # Pool bytes to patch representations
        patch_representations = self.patch_pooling(
            local_output, patch_ids, training=training
        )

        # Get current reasoning states
        z_h, z_l = carry["z_h"], carry["z_l"]

        # Hierarchical reasoning cycles (detached for efficiency)
        with ops.stop_gradient():
            for h_step in range(self.config.h_cycles):
                for l_step in range(self.config.l_cycles):
                    # Skip last L step of last H cycle
                    if not (h_step == self.config.h_cycles - 1 and l_step == self.config.l_cycles - 1):
                        z_l = self.l_reasoning(z_l, z_h + patch_representations, training=training)

                # Skip last H step
                if h_step != self.config.h_cycles - 1:
                    z_h = self.h_reasoning(z_h, z_l, training=training)

        # Final step with gradients
        z_l = self.l_reasoning(z_l, z_h + patch_representations, training=training)
        z_h = self.h_reasoning(z_h, z_l, training=training)

        # Generate outputs using first patch representation (global context)
        global_representation = z_h[:, 0]  # Shape: (batch_size, embed_dim)

        # Language modeling head (predict next bytes)
        lm_logits = self.lm_head(global_representation)

        # Q head for halt decisions
        q_logits = self.q_head(global_representation)
        q_halt_logits = q_logits[..., 0]
        q_continue_logits = q_logits[..., 1]

        outputs = {
            "logits": lm_logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "patch_representations": patch_representations,
            "entropy": entropy
        }

        # New carry (detached)
        new_carry = {
            "z_h": ops.stop_gradient(z_h),
            "z_l": ops.stop_gradient(z_l)
        }

        return new_carry, outputs

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({'config': self.config.to_dict()})
        return config


@keras.saving.register_keras_serializable()
class ReasoningByteBERT(keras.Model):
    """
    ReasoningByteBERT: Combining ByteBERT with Hierarchical Reasoning and ACT.

    This model integrates byte-level processing with iterative reasoning
    and adaptive computation time, creating a powerful system for
    complex language understanding and reasoning tasks.

    Key Features:
    - Byte-level processing with dynamic patching
    - Hierarchical reasoning with high-level and low-level states
    - Adaptive computation time with Q-learning based halting
    - Task-specific puzzle embeddings
    - Hash n-gram enhanced representations

    Args:
        config: ReasoningByteBertConfig containing all hyperparameters.
        **kwargs: Additional model arguments.
    """

    def __init__(self, config: ReasoningByteBertConfig, **kwargs):
        super().__init__(**kwargs)

        config.validate()
        self.config = config

        # Initialize components to None - created in build()
        self.embeddings = None
        self.reasoning_core = None

        self._build_input_shape = None

        logger.info(f"Created ReasoningByteBERT with byte-level processing, "
                    f"{config.h_layers}+{config.l_layers} reasoning layers, "
                    f"ACT with max {config.halt_max_steps} steps")

    def build(self, input_shape: Union[Tuple[Optional[int], ...], Dict[str, Tuple[Optional[int], ...]]]) -> None:
        """Build the complete ReasoningByteBERT model."""
        if self.built:
            return

        self._build_input_shape = input_shape

        # Handle different input formats
        if isinstance(input_shape, dict):
            token_shape = input_shape.get('token_ids', input_shape.get('inputs'))
        else:
            token_shape = input_shape

        logger.info(f"Building ReasoningByteBERT with token shape: {token_shape}")

        # Create embeddings layer
        self.embeddings = ReasoningByteEmbeddings(
            config=self.config,
            name="embeddings"
        )
        self.embeddings.build(token_shape)

        # Create reasoning core
        embeddings_shape = (*token_shape, self.config.embed_dim)
        self.reasoning_core = ReasoningByteCore(
            config=self.config,
            name="reasoning_core"
        )
        self.reasoning_core.build(embeddings_shape)

        super().build(input_shape)
        logger.info("ReasoningByteBERT model built successfully")

    def initial_carry(self, batch: Dict[str, keras.KerasTensor]) -> Dict[str, keras.KerasTensor]:
        """Initialize carry state for a batch."""
        batch_size = ops.shape(batch["token_ids"])[0]

        return {
            # Core reasoning state
            "inner_carry": self.reasoning_core.empty_carry(batch_size),

            # ACT state
            "steps": ops.zeros((batch_size,), dtype="int32"),
            "halted": ops.ones((batch_size,), dtype="bool"),  # Start halted

            # Current data cache
            "current_data": {k: ops.zeros_like(v) for k, v in batch.items()}
        }

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor], Tuple],
            training: Optional[bool] = None,
            return_dict: bool = True
    ) -> Union[Dict[str, keras.KerasTensor], keras.KerasTensor]:
        """
        Forward pass through ReasoningByteBERT.

        This method supports three calling modes:
        1. Standard call: call(batch_dict) - runs until convergence
        2. Step call: call((carry, batch_dict)) - single reasoning step
        3. Tensor call: call(token_ids) - simple forward pass

        Args:
            inputs: Input data in various formats.
            training: Whether in training mode.
            return_dict: Whether to return outputs as dictionary.

        Returns:
            Model outputs in requested format.
        """
        if isinstance(inputs, tuple):
            # Step call: (carry, batch)
            carry, batch = inputs
            return self._forward_step(carry, batch, training=training, return_dict=return_dict)
        elif isinstance(inputs, dict):
            # Standard batch call
            return self._forward_complete(inputs, training=training, return_dict=return_dict)
        else:
            # Simple tensor call
            batch = {"token_ids": inputs}
            return self._forward_complete(batch, training=training, return_dict=return_dict)

    def _forward_complete(
            self,
            batch: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None,
            return_dict: bool = True
    ) -> Union[Dict[str, keras.KerasTensor], keras.KerasTensor]:
        """Run complete forward pass until all sequences halt."""
        carry = self.initial_carry(batch)
        outputs = None

        # Run steps until all sequences halt
        max_iterations = self.config.halt_max_steps * 2  # Safety limit
        for _ in range(max_iterations):
            carry, outputs, all_finished = self._forward_step(
                carry, batch, training=training, return_dict=True
            )
            if all_finished:
                break

        if return_dict:
            return outputs
        else:
            return outputs.get("logits", outputs)

    def _forward_step(
            self,
            carry: Dict[str, keras.KerasTensor],
            batch: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None,
            return_dict: bool = True
    ) -> Tuple[Dict[str, keras.KerasTensor], Dict[str, keras.KerasTensor], bool]:
        """Single reasoning step with ACT logic."""
        # Update carry for new sequences (halted ones get reset)
        new_inner_carry = self.reasoning_core.reset_carry(
            carry["halted"], carry["inner_carry"]
        )

        # Reset steps for halted sequences
        new_steps = ops.where(carry["halted"], 0, carry["steps"])

        # Update current data for halted sequences
        new_current_data = {}
        for k, v in carry["current_data"].items():
            reset_mask = ops.reshape(carry["halted"], [-1] + [1] * (len(v.shape) - 1))
            new_current_data[k] = ops.where(reset_mask, batch[k], v)

        # Get embeddings
        embeddings = self.embeddings(
            token_ids=new_current_data["token_ids"],
            puzzle_ids=new_current_data.get("puzzle_ids"),
            training=training
        )

        # Forward pass through reasoning core
        new_inner_carry, outputs = self.reasoning_core(
            carry=new_inner_carry,
            embeddings=embeddings,
            byte_tokens=new_current_data["token_ids"],
            training=training
        )

        # Update steps
        new_steps = new_steps + 1
        is_last_step = new_steps >= self.config.halt_max_steps

        # Determine halting
        halted = is_last_step

        if training and self.config.halt_max_steps > 1:
            # Q-learning based halting
            q_halt = outputs["q_halt_logits"]
            q_continue = outputs["q_continue_logits"]

            # Halt if q_halt > q_continue
            halted = halted | (q_halt > q_continue)

            # Exploration: random minimum halt steps
            if self.config.halt_exploration_prob > 0:
                explore_mask = keras.random.uniform(ops.shape(q_halt)) < self.config.halt_exploration_prob
                min_steps = keras.random.uniform(
                    ops.shape(new_steps),
                    minval=2,
                    maxval=self.config.halt_max_steps + 1,
                    dtype="int32"
                )
                min_halt_steps = ops.where(explore_mask, min_steps, 1)
                halted = halted & (new_steps >= min_halt_steps)

        # Create new carry
        new_carry = {
            "inner_carry": new_inner_carry,
            "steps": new_steps,
            "halted": halted,
            "current_data": new_current_data
        }

        # Check if all sequences are finished
        all_finished = ops.all(halted)

        return new_carry, outputs, all_finished

    def encode_text(
            self,
            text: str,
            max_length: Optional[int] = None
    ) -> keras.KerasTensor:
        """
        Encode text to byte tokens.

        Args:
            text: Input text string.
            max_length: Maximum sequence length.

        Returns:
            Byte token IDs tensor.
        """
        return self.embeddings.tokenizer.text_to_bytes(text, max_length or self.config.seq_len)

    def decode_tokens(self, token_ids: keras.KerasTensor) -> str:
        """
        Decode byte tokens back to text.

        Args:
            token_ids: Byte token IDs tensor.

        Returns:
            Decoded text string.
        """
        return self.embeddings.tokenizer.tokens_to_text(token_ids)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({'config': self.config.to_dict()})
        return config


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_reasoning_byte_bert_base() -> ReasoningByteBertConfig:
    """Create base ReasoningByteBERT configuration."""
    return ReasoningByteBertConfig(
        vocab_size=260,
        seq_len=2048,
        embed_dim=768,
        local_hidden_size=384,
        h_layers=4,
        l_layers=4,
        h_cycles=2,
        l_cycles=2,
        num_heads=12,
        ffn_expansion_factor=4
    )


def create_reasoning_byte_bert_large() -> ReasoningByteBertConfig:
    """Create large ReasoningByteBERT configuration."""
    return ReasoningByteBertConfig(
        vocab_size=260,
        seq_len=4096,
        embed_dim=1024,
        local_hidden_size=512,
        h_layers=6,
        l_layers=6,
        h_cycles=3,
        l_cycles=3,
        num_heads=16,
        ffn_expansion_factor=4,
        max_patches=1024
    )


def create_reasoning_byte_bert_for_reasoning_tasks(
        config: ReasoningByteBertConfig,
        num_puzzle_types: int = 1000
) -> ReasoningByteBERT:
    """
    Create ReasoningByteBERT optimized for reasoning tasks.

    Args:
        config: Base configuration.
        num_puzzle_types: Number of different puzzle/task types.

    Returns:
        Configured ReasoningByteBERT model.
    """
    # Optimize for reasoning
    config.num_puzzle_identifiers = num_puzzle_types
    config.puzzle_emb_dim = config.embed_dim
    config.halt_max_steps = 20  # Allow more reasoning steps
    config.halt_exploration_prob = 0.15  # More exploration
    config.h_cycles = 3  # More reasoning cycles
    config.l_cycles = 3

    logger.info(f"Created ReasoningByteBERT for reasoning with {num_puzzle_types} puzzle types")

    return ReasoningByteBERT(config=config, name="reasoning_byte_bert_reasoning")


def create_fast_reasoning_byte_bert() -> ReasoningByteBertConfig:
    """Create fast ReasoningByteBERT configuration for quick inference."""
    return ReasoningByteBertConfig(
        vocab_size=260,
        seq_len=1024,
        embed_dim=512,
        local_hidden_size=256,
        h_layers=2,
        l_layers=2,
        h_cycles=1,
        l_cycles=2,
        num_heads=8,
        ffn_expansion_factor=2,
        halt_max_steps=8,
        max_patches=256
    )