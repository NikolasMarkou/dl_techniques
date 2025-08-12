import math
import keras
from typing import Optional, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .positional_embedding import PositionalEmbedding
from .hrm_reasoning_module import HierarchicalReasoningModule
from .hrm_sparse_puzzle_embedding import SparsePuzzleEmbedding
from .rotary_position_embedding import RotaryPositionEmbedding
from .byte_latent_transformer_blocks import (
    ByteTokenizer, EntropyModel,
    DynamicPatcher, LocalEncoder,
    GlobalTransformer, LocalDecoder
)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ByteLatentReasoningCore(keras.layers.Layer):
    """
    Core hierarchical reasoning model that operates on dynamic byte patches.

    This layer combines the Byte Latent Transformer's dynamic patching with
    Hierarchical Reasoning Model's iterative reasoning capabilities. It processes
    raw byte sequences through entropy-based patching and applies hierarchical
    reasoning cycles at both local (byte) and global (patch) levels.

    Args:
        vocab_size: Size of byte vocabulary (typically 256 + special tokens).
        seq_len: Maximum sequence length in bytes.
        embed_dim: Embedding dimension for reasoning states.
        local_dim: Hidden dimension for local byte processing.
        global_dim: Hidden dimension for global patch processing.
        max_patches: Maximum number of patches per sequence.
        num_puzzle_identifiers: Number of puzzle identifiers.
        puzzle_emb_dim: Puzzle embedding dimension (0 to disable).
        batch_size: Batch size for training.
        h_layers: Number of high-level (global) reasoning layers.
        l_layers: Number of low-level (local) reasoning layers.
        h_cycles: Number of high-level reasoning cycles.
        l_cycles: Number of low-level reasoning cycles.
        num_heads: Number of attention heads.
        entropy_threshold: Threshold for dynamic byte patching.
        pos_encodings: Type of positional encodings ("rope" or "learned").
        rope_theta: RoPE theta parameter.
        dropout_rate: Dropout rate.
        use_bias: Whether to use bias in linear layers.
        embeddings_initializer: Initializer for embeddings.
        kernel_initializer: Initializer for kernel weights.
        embeddings_regularizer: Regularizer for embeddings.
        kernel_regularizer: Regularizer for kernel weights.
        **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            vocab_size: int,
            seq_len: int,
            embed_dim: int,
            local_dim: int = 512,
            global_dim: int = 768,
            max_patches: int = 512,
            num_puzzle_identifiers: int = 1000,
            puzzle_emb_dim: int = 512,
            batch_size: int = 32,
            h_layers: int = 4,
            l_layers: int = 4,
            h_cycles: int = 2,
            l_cycles: int = 2,
            num_heads: int = 8,
            entropy_threshold: float = 1.5,
            pos_encodings: str = "rope",
            rope_theta: float = 10000.0,
            dropout_rate: float = 0.0,
            use_bias: bool = False,
            embeddings_initializer: Union[str, keras.initializers.Initializer] = "truncated_normal",
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            embeddings_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Store configuration
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.max_patches = max_patches
        self.num_puzzle_identifiers = num_puzzle_identifiers
        self.puzzle_emb_dim = puzzle_emb_dim
        self.batch_size = batch_size
        self.h_layers = h_layers
        self.l_layers = l_layers
        self.h_cycles = h_cycles
        self.l_cycles = l_cycles
        self.num_heads = num_heads
        self.entropy_threshold = entropy_threshold
        self.pos_encodings = pos_encodings
        self.rope_theta = rope_theta
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.embeddings_initializer = embeddings_initializer
        self.kernel_initializer = kernel_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.kernel_regularizer = kernel_regularizer

        # Calculate embedding scale
        self.embed_scale = math.sqrt(embed_dim)

        # Will be built in build()
        self.tokenizer = None
        self.entropy_model = None
        self.patcher = None
        self.local_encoder = None
        self.global_transformer = None
        self.local_decoder = None
        self.puzzle_embedding = None
        self.position_embedding = None
        self.rope = None
        self.h_reasoning = None
        self.l_reasoning = None
        self.lm_head = None
        self.q_head = None
        self.h_init = None
        self.l_init = None

        # Byte-to-reasoning projection layers
        self.byte_to_reasoning_proj = None
        self.reasoning_to_byte_proj = None
        self.entropy_to_reasoning_proj = None

    def build(self, input_shape):
        """Build the model components."""
        # Byte processing components
        self.tokenizer = ByteTokenizer(
            vocab_size=self.vocab_size,
            name="tokenizer"
        )

        self.entropy_model = EntropyModel(
            vocab_size=self.vocab_size,
            hidden_dim=self.local_dim,
            num_layers=4,
            num_heads=self.num_heads,
            max_seq_len=self.seq_len,
            dropout_rate=self.dropout_rate,
            name="entropy_model"
        )

        self.patcher = DynamicPatcher(
            entropy_threshold=self.entropy_threshold,
            max_patches=self.max_patches,
            name="patcher"
        )

        self.local_encoder = LocalEncoder(
            vocab_size=self.vocab_size,
            local_dim=self.local_dim,
            num_local_layers=self.l_layers,
            num_heads_local=self.num_heads,
            max_sequence_length=self.seq_len,
            max_patches=self.max_patches,
            dropout_rate=self.dropout_rate,
            global_dim=self.global_dim,
            name="local_encoder"
        )

        self.global_transformer = GlobalTransformer(
            global_dim=self.global_dim,
            num_global_layers=self.h_layers,
            num_heads_global=self.num_heads,
            max_patches=self.max_patches,
            dropout_rate=self.dropout_rate,
            name="global_transformer"
        )

        self.local_decoder = LocalDecoder(
            vocab_size=self.vocab_size,
            local_dim=self.local_dim,
            global_dim=self.global_dim,
            num_local_layers=self.l_layers,
            num_heads_local=self.num_heads,
            max_sequence_length=self.seq_len,
            dropout_rate=self.dropout_rate,
            name="local_decoder"
        )

        # Puzzle embeddings (adapted for byte sequences)
        if self.puzzle_emb_dim > 0:
            self.puzzle_embedding = SparsePuzzleEmbedding(
                num_embeddings=self.num_puzzle_identifiers,
                embedding_dim=self.puzzle_emb_dim,
                batch_size=self.batch_size,
                embeddings_initializer="zeros",
                embeddings_regularizer=self.embeddings_regularizer,
                name="puzzle_embedding"
            )

        # Positional embeddings for reasoning states
        if self.pos_encodings == "rope":
            self.rope = RotaryPositionEmbedding(
                head_dim=self.embed_dim // self.num_heads,
                max_seq_len=self.max_patches,
                rope_theta=self.rope_theta,
                name="rope"
            )
        elif self.pos_encodings == "learned":
            self.position_embedding = PositionalEmbedding(
                max_seq_len=self.max_patches,
                dim=self.embed_dim,
                dropout=0.0,
                name="position_embedding"
            )

        # Hierarchical reasoning modules (adapted for byte-patch processing)
        self.h_reasoning = HierarchicalReasoningModule(
            num_layers=self.h_layers,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ffn_expansion_factor=4,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="h_reasoning"
        )

        self.l_reasoning = HierarchicalReasoningModule(
            num_layers=self.l_layers,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ffn_expansion_factor=4,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="l_reasoning"
        )

        # Cross-modal projection layers
        self.byte_to_reasoning_proj = keras.layers.Dense(
            self.embed_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="byte_to_reasoning_proj"
        )

        self.reasoning_to_byte_proj = keras.layers.Dense(
            self.global_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="reasoning_to_byte_proj"
        )

        self.entropy_to_reasoning_proj = keras.layers.Dense(
            self.embed_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="entropy_to_reasoning_proj"
        )

        # Output heads
        self.lm_head = keras.layers.Dense(
            self.vocab_size,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="lm_head"
        )

        # Q head for adaptive computation (enhanced with entropy information)
        self.q_head = keras.layers.Dense(
            2,  # halt, continue
            use_bias=True,
            kernel_initializer="zeros",
            bias_initializer=keras.initializers.Constant(-5.0),
            name="q_head"
        )

        # Initial reasoning states
        self.h_init = self.add_weight(
            name="h_init",
            shape=(self.embed_dim,),
            initializer=keras.initializers.TruncatedNormal(stddev=1.0),
            trainable=True
        )

        self.l_init = self.add_weight(
            name="l_init",
            shape=(self.embed_dim,),
            initializer=keras.initializers.TruncatedNormal(stddev=1.0),
            trainable=True
        )

        super().build(input_shape)

    def _create_reasoning_embeddings(self, byte_tokens, puzzle_ids, patch_representations, entropy_values):
        """Create input embeddings for reasoning modules."""
        batch_size = keras.ops.shape(byte_tokens)[0]

        # Project patch representations to reasoning dimension
        patch_reasoning = self.byte_to_reasoning_proj(patch_representations)

        # Add puzzle embeddings if enabled
        if self.puzzle_emb_dim > 0 and puzzle_ids is not None:
            puzzle_emb = self.puzzle_embedding(puzzle_ids)
            # Broadcast puzzle embedding to all patches
            puzzle_emb_expanded = keras.ops.expand_dims(puzzle_emb, axis=1)
            puzzle_emb_expanded = keras.ops.tile(
                puzzle_emb_expanded,
                [1, keras.ops.shape(patch_reasoning)[1], 1]
            )
            patch_reasoning = patch_reasoning + puzzle_emb_expanded[:, :, :self.embed_dim]

        # Incorporate entropy information into reasoning
        if entropy_values is not None:
            # Pool entropy values by patches (simplified approach)
            pooled_entropy = keras.ops.expand_dims(
                keras.ops.mean(entropy_values, axis=1),
                axis=1
            )
            pooled_entropy = keras.ops.tile(
                pooled_entropy,
                [1, keras.ops.shape(patch_reasoning)[1], 1]
            )
            entropy_proj = self.entropy_to_reasoning_proj(pooled_entropy)
            patch_reasoning = patch_reasoning + entropy_proj

        # Add positional embeddings to reasoning states
        if self.pos_encodings == "learned" and self.position_embedding is not None:
            seq_len = keras.ops.shape(patch_reasoning)[1]
            pos_emb = self.position_embedding.position_embeddings[:seq_len]
            pos_emb = keras.ops.expand_dims(pos_emb, axis=0)
            patch_reasoning = 0.707106781 * (patch_reasoning + pos_emb)

        # Scale embeddings
        return self.embed_scale * patch_reasoning

    def empty_carry(self, batch_size):
        """Create empty carry state for byte-level reasoning."""
        return {
            "z_h": keras.ops.zeros((batch_size, self.max_patches, self.embed_dim)),
            "z_l": keras.ops.zeros((batch_size, self.max_patches, self.embed_dim)),
            "patch_context": keras.ops.zeros((batch_size, self.max_patches, self.global_dim)),
            "entropy_state": keras.ops.zeros((batch_size, self.seq_len))
        }

    def reset_carry(self, reset_flag, carry):
        """Reset carry state for halted sequences."""
        batch_size = keras.ops.shape(reset_flag)[0]

        # Broadcast initial states
        h_init_broadcast = keras.ops.broadcast_to(
            keras.ops.reshape(self.h_init, [1, 1, self.embed_dim]),
            [batch_size, self.max_patches, self.embed_dim]
        )
        l_init_broadcast = keras.ops.broadcast_to(
            keras.ops.reshape(self.l_init, [1, 1, self.embed_dim]),
            [batch_size, self.max_patches, self.embed_dim]
        )

        # Reset based on halt flag
        reset_flag = keras.ops.reshape(reset_flag, [-1, 1, 1])
        new_z_h = keras.ops.where(reset_flag, h_init_broadcast, carry["z_h"])
        new_z_l = keras.ops.where(reset_flag, l_init_broadcast, carry["z_l"])

        # Reset other states
        zero_context = keras.ops.zeros_like(carry["patch_context"])
        zero_entropy = keras.ops.zeros_like(carry["entropy_state"])

        new_patch_context = keras.ops.where(reset_flag, zero_context, carry["patch_context"])
        reset_flag_seq = keras.ops.reshape(reset_flag[:, :, 0], [-1, 1])
        new_entropy_state = keras.ops.where(reset_flag_seq, zero_entropy, carry["entropy_state"])

        return {
            "z_h": new_z_h,
            "z_l": new_z_l,
            "patch_context": new_patch_context,
            "entropy_state": new_entropy_state
        }

    def call(self, carry, inputs, training=None):
        """
        Forward pass through the byte latent hierarchical reasoning core.

        Args:
            carry: Current carry state dict with reasoning and byte processing states.
            inputs: Dict with "byte_tokens" and optionally "puzzle_ids".
            training: Whether in training mode.

        Returns:
            Tuple of (new_carry, outputs_dict).
        """
        byte_tokens = inputs["byte_tokens"]
        puzzle_ids = inputs.get("puzzle_ids")

        # Step 1: Compute entropy for dynamic patching
        entropy_logits = self.entropy_model(byte_tokens, training=training)
        entropy = self.entropy_model.compute_entropy(entropy_logits)

        # Step 2: Create dynamic patches based on entropy
        patch_lengths = self.patcher(entropy, training=training)
        patch_ids = self.patcher.compute_patch_ids(patch_lengths)

        # Step 3: Encode bytes to patch representations (local processing)
        patch_representations = self.local_encoder(
            byte_tokens, patch_ids, training=training
        )

        # Step 4: Create reasoning embeddings from byte processing
        reasoning_input = self._create_reasoning_embeddings(
            byte_tokens, puzzle_ids, patch_representations, entropy
        )

        # Step 5: Hierarchical reasoning cycles with byte-patch interaction
        z_h, z_l = carry["z_h"], carry["z_l"]

        # Forward iterations (detached for efficiency)
        with keras.ops.stop_gradient():
            for h_step in range(self.h_cycles):
                for l_step in range(self.l_cycles):
                    # Skip last L step of last H cycle
                    if not (h_step == self.h_cycles - 1 and l_step == self.l_cycles - 1):
                        # Low-level reasoning: incorporate patch information
                        z_l = self.l_reasoning(
                            z_l, z_h + reasoning_input, training=training
                        )

                # Skip last H step
                if h_step != self.h_cycles - 1:
                    # High-level reasoning: abstract patch representations
                    z_h = self.h_reasoning(z_h, z_l, training=training)

        # Final step with gradients
        z_l = self.l_reasoning(z_l, z_h + reasoning_input, training=training)
        z_h = self.h_reasoning(z_h, z_l, training=training)

        # Step 6: Process patch representations through global transformer
        # Convert reasoning states back to byte processing format
        reasoning_to_byte_context = self.reasoning_to_byte_proj(z_h)
        global_context = self.global_transformer(
            reasoning_to_byte_context, training=training
        )

        # Step 7: Generate byte-level predictions with hierarchical context
        logits = self.local_decoder(
            byte_tokens, global_context, patch_ids, training=training
        )

        # Step 8: Compute Q-values for adaptive computation (enhanced with entropy)
        # Use both reasoning state and entropy information for halting decisions
        entropy_summary = keras.ops.mean(entropy, axis=1, keepdims=True)  # (batch, 1)
        q_input = keras.ops.concatenate([
            keras.ops.mean(z_h, axis=1),  # Global reasoning state summary
            entropy_summary
        ], axis=-1)

        q_logits = self.q_head(q_input)
        q_halt_logits = q_logits[..., 0]
        q_continue_logits = q_logits[..., 1]

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "entropy": entropy,
            "patch_lengths": patch_lengths
        }

        # New carry (detached)
        new_carry = {
            "z_h": keras.ops.stop_gradient(z_h),
            "z_l": keras.ops.stop_gradient(z_l),
            "patch_context": keras.ops.stop_gradient(global_context),
            "entropy_state": keras.ops.stop_gradient(entropy)
        }

        return new_carry, outputs

    def compute_output_shape(self, input_shape):
        """Compute output shapes."""
        batch_size = input_shape.get("byte_tokens", [None])[0]
        return {
            "logits": (batch_size, self.seq_len, self.vocab_size),
            "q_halt_logits": (batch_size,),
            "q_continue_logits": (batch_size,),
            "entropy": (batch_size, self.seq_len),
            "patch_lengths": (batch_size, self.max_patches)
        }

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len,
            "embed_dim": self.embed_dim,
            "local_dim": self.local_dim,
            "global_dim": self.global_dim,
            "max_patches": self.max_patches,
            "num_puzzle_identifiers": self.num_puzzle_identifiers,
            "puzzle_emb_dim": self.puzzle_emb_dim,
            "batch_size": self.batch_size,
            "h_layers": self.h_layers,
            "l_layers": self.l_layers,
            "h_cycles": self.h_cycles,
            "l_cycles": self.l_cycles,
            "num_heads": self.num_heads,
            "entropy_threshold": self.entropy_threshold,
            "pos_encodings": self.pos_encodings,
            "rope_theta": self.rope_theta,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "embeddings_initializer": keras.initializers.serialize(self.embeddings_initializer),
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "embeddings_regularizer": keras.regularizers.serialize(self.embeddings_regularizer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
