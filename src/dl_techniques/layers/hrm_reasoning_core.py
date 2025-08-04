"""
This module defines the HierarchicalReasoningCore, a stateful, recurrent Keras layer
that represents the central engine of the Hierarchical Reasoning Model (HRM) architecture.

This is not a standard Transformer. Instead, it's a sophisticated recurrent model
that uses Transformer-like blocks to perform iterative, multi-level reasoning.
Its design is intended for complex tasks that may require a variable amount of
computational effort, which it can learn to allocate dynamically.

Core Concepts:

1.  **Stateful, Recurrent Operation:**
    The layer is stateful. It takes a 'carry' state as input and returns an updated
    'carry' as output. This state consists of two tensors: a high-level state (`z_h`)
    and a low-level state (`z_l`), allowing the model to maintain and refine its
    "thoughts" over multiple steps or calls.

2.  **Hierarchical Reasoning Modules:**
    The model features two distinct reasoning pathways, implemented as stacks of
    Transformer blocks:
    - `h_reasoning` (High-Level): Processes the low-level state (`z_l`) to produce a more
      abstract, high-level representation (`z_h`).
    - `l_reasoning` (Low-Level): Processes the high-level state (`z_h`) combined with the
      initial input embeddings to refine the detailed, low-level representation (`z_l`).

3.  **Iterative Refinement via Cycles:**
    A key feature is the use of reasoning cycles (`h_cycles`, `l_cycles`). Instead of a
    single pass, the model iterates internally, feeding the output of one module back
    into the other. This allows for a progressive refinement of the internal state,
e.g., performing two low-level cycles for every one high-level cycle, enabling the
    model to "think" or reason for multiple steps before producing an output.

4.  **Specialized Inputs:**
    The model combines standard token embeddings with an optional, special-purpose
    `SparsePuzzleEmbedding`. This puzzle embedding is prepended to the sequence,
    providing a global context or task identifier that informs the entire reasoning process.

5.  **Dual Output Heads for Adaptive Computation:**
    The layer produces two distinct outputs, enabling adaptive computation strategies
    (like Adaptive Computation Time - ACT):
    - `lm_head`: A standard language modeling head that predicts the next token in
      the sequence based on the final high-level state.
    - `q_head`: A Q-learning head that outputs logits for a "halt" or "continue"
      decision. This allows an external mechanism to decide whether to stop the
      reasoning process or to feed the new carry state back into this layer for
      another cycle of refinement.

6.  **Efficient Training with Detached Gradients:**
    To make training feasible across many reasoning cycles, gradients are explicitly
    detached (`ops.stop_gradient`) for all but the final iteration. This means the
    model is trained to perform one full step of reasoning correctly, while the
    intermediate "thought" steps guide the process without accumulating gradients,
    preventing vanishing/exploding gradient issues.

Architectural Flow (`call` method):
1.  Receives the `carry` state (`z_h`, `z_l`) and input IDs (`token_ids`, `puzzle_ids`).
2.  Generates combined input embeddings from tokens and the special puzzle ID.
3.  Iterates through `h_cycles` and `l_cycles`, refining `z_h` and `z_l` with gradients turned off.
4.  Performs one final reasoning step with gradients enabled.
5.  Uses the final `z_h` state to compute logits for the language model and Q-values.
6.  Returns the new, detached `carry` state and the computed outputs.
"""

import math
import keras
from typing import Optional, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .rope import RotaryPositionEmbedding
from .positional_embedding import PositionalEmbedding
from .hrm_reasoning_module import HierarchicalReasoningModule
from .hrm_sparse_puzzle_embedding import SparsePuzzleEmbedding

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class HierarchicalReasoningCore(keras.layers.Layer):
    """
    Core hierarchical reasoning model without ACT wrapper.

    This implements the main HRM logic with high-level and low-level reasoning
    modules, puzzle embeddings, and output heads for language modeling and Q-learning.

    Args:
        vocab_size: Size of vocabulary
        seq_len: Maximum sequence length
        embed_dim: Embedding dimension
        num_puzzle_identifiers: Number of puzzle identifiers
        puzzle_emb_dim: Puzzle embedding dimension (0 to disable)
        batch_size: Batch size for training
        h_layers: Number of high-level reasoning layers
        l_layers: Number of low-level reasoning layers
        h_cycles: Number of high-level reasoning cycles
        l_cycles: Number of low-level reasoning cycles
        num_heads: Number of attention heads
        ffn_expansion_factor: Feed-forward expansion factor
        pos_encodings: Type of positional encodings ("rope" or "learned")
        rope_theta: RoPE theta parameter
        dropout_rate: Dropout rate
        use_bias: Whether to use bias in linear layers
        embeddings_initializer: Initializer for embeddings
        kernel_initializer: Initializer for kernel weights
        embeddings_regularizer: Regularizer for embeddings
        kernel_regularizer: Regularizer for kernel weights
        **kwargs: Additional layer arguments
    """

    def __init__(
            self,
            vocab_size: int,
            seq_len: int,
            embed_dim: int,
            num_puzzle_identifiers: int,
            puzzle_emb_dim: int = 0,
            batch_size: int = 32,
            h_layers: int = 4,
            l_layers: int = 4,
            h_cycles: int = 2,
            l_cycles: int = 2,
            num_heads: int = 8,
            ffn_expansion_factor: int = 4,
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
        self.num_puzzle_identifiers = num_puzzle_identifiers
        self.puzzle_emb_dim = puzzle_emb_dim
        self.batch_size = batch_size
        self.h_layers = h_layers
        self.l_layers = l_layers
        self.h_cycles = h_cycles
        self.l_cycles = l_cycles
        self.num_heads = num_heads
        self.ffn_expansion_factor = ffn_expansion_factor
        self.pos_encodings = pos_encodings
        self.rope_theta = rope_theta
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.embeddings_initializer = embeddings_initializer
        self.kernel_initializer = kernel_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.kernel_regularizer = kernel_regularizer

        # Calculate puzzle embedding sequence length
        self.puzzle_emb_len = max(1, (puzzle_emb_dim + embed_dim - 1) // embed_dim) if puzzle_emb_dim > 0 else 0
        self.total_seq_len = seq_len + self.puzzle_emb_len

        # Embedding scale (like in original code)
        self.embed_scale = math.sqrt(embed_dim)
        embed_init_std = 1.0 / self.embed_scale

        # Will be built in build()
        self.token_embedding = None
        self.puzzle_embedding = None
        self.position_embedding = None
        self.rope = None
        self.h_reasoning = None
        self.l_reasoning = None
        self.lm_head = None
        self.q_head = None
        self.h_init = None
        self.l_init = None

    def build(self, input_shape):
        """Build the model components."""
        # Token embeddings
        self.token_embedding = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embed_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=1.0 / self.embed_scale),
            embeddings_regularizer=self.embeddings_regularizer,
            name="token_embedding"
        )

        # Puzzle embeddings (if enabled)
        if self.puzzle_emb_dim > 0:
            self.puzzle_embedding = SparsePuzzleEmbedding(
                num_embeddings=self.num_puzzle_identifiers,
                embedding_dim=self.puzzle_emb_dim,
                batch_size=self.batch_size,
                embeddings_initializer="zeros",  # Zero init as in original
                embeddings_regularizer=self.embeddings_regularizer,
                name="puzzle_embedding"
            )

        # Positional embeddings
        if self.pos_encodings == "rope":
            self.rope = RotaryPositionEmbedding(
                head_dim=self.embed_dim // self.num_heads,
                max_seq_len=self.total_seq_len,
                rope_theta=self.rope_theta,
                name="rope"
            )
        elif self.pos_encodings == "learned":
            self.position_embedding = PositionalEmbedding(
                max_seq_len=self.total_seq_len,
                dim=self.embed_dim,
                dropout=0.0,  # No dropout in original
                name="position_embedding"
            )
        else:
            raise ValueError(f"Unknown pos_encodings: {self.pos_encodings}")

        # Reasoning modules
        self.h_reasoning = HierarchicalReasoningModule(
            num_layers=self.h_layers,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ffn_expansion_factor=self.ffn_expansion_factor,
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
            ffn_expansion_factor=self.ffn_expansion_factor,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="l_reasoning"
        )

        # Output heads
        self.lm_head = keras.layers.Dense(
            self.vocab_size,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="lm_head"
        )

        # Q head for halt decisions (2 outputs: halt, continue)
        self.q_head = keras.layers.Dense(
            2,
            use_bias=True,
            kernel_initializer="zeros",  # Zero init as in original
            bias_initializer=keras.initializers.Constant(-5.0),  # Bias init as in original
            name="q_head"
        )

        # Initial states for reasoning modules
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

    def _input_embeddings(self, token_ids, puzzle_ids):
        """Create input embeddings from token and puzzle IDs."""
        # Token embeddings
        token_emb = self.token_embedding(token_ids)

        # Add puzzle embeddings if enabled
        if self.puzzle_emb_dim > 0:
            puzzle_emb = self.puzzle_embedding(puzzle_ids)

            # Pad puzzle embedding to match sequence embedding format
            pad_size = self.puzzle_emb_len * self.embed_dim - self.puzzle_emb_dim
            if pad_size > 0:
                puzzle_emb = keras.ops.pad(puzzle_emb, [[0, 0], [0, pad_size]])

            # Reshape and concatenate
            batch_size = keras.ops.shape(token_emb)[0]
            puzzle_emb = keras.ops.reshape(puzzle_emb, [batch_size, self.puzzle_emb_len, self.embed_dim])
            embeddings = keras.ops.concatenate([puzzle_emb, token_emb], axis=1)
        else:
            embeddings = token_emb

        # Add positional embeddings
        if self.pos_encodings == "learned":
            # Scale by 1/sqrt(2) to maintain variance as in original
            embeddings = 0.707106781 * (embeddings + self.position_embedding.position_embeddings)

        # Scale embeddings
        return self.embed_scale * embeddings

    def empty_carry(self, batch_size):
        """Create empty carry state."""
        return {
            "z_h": keras.ops.zeros((batch_size, self.total_seq_len, self.embed_dim)),
            "z_l": keras.ops.zeros((batch_size, self.total_seq_len, self.embed_dim))
        }

    def reset_carry(self, reset_flag, carry):
        """Reset carry state for halted sequences."""
        batch_size = keras.ops.shape(reset_flag)[0]

        # Broadcast initial states
        h_init_broadcast = keras.ops.broadcast_to(
            keras.ops.reshape(self.h_init, [1, 1, self.embed_dim]),
            [batch_size, self.total_seq_len, self.embed_dim]
        )
        l_init_broadcast = keras.ops.broadcast_to(
            keras.ops.reshape(self.l_init, [1, 1, self.embed_dim]),
            [batch_size, self.total_seq_len, self.embed_dim]
        )

        # Reset based on halt flag
        reset_flag = keras.ops.reshape(reset_flag, [-1, 1, 1])
        new_z_h = keras.ops.where(reset_flag, h_init_broadcast, carry["z_h"])
        new_z_l = keras.ops.where(reset_flag, l_init_broadcast, carry["z_l"])

        return {"z_h": new_z_h, "z_l": new_z_l}

    def call(self, carry, inputs, training=None):
        """
        Forward pass through the hierarchical reasoning core.

        Args:
            carry: Current carry state dict with "z_h" and "z_l"
            inputs: Dict with "token_ids" and "puzzle_ids"
            training: Whether in training mode

        Returns:
            Tuple of (new_carry, outputs_dict)
        """
        # Get input embeddings
        input_emb = self._input_embeddings(inputs["token_ids"], inputs["puzzle_ids"])

        z_h, z_l = carry["z_h"], carry["z_l"]

        # Forward iterations (detached for efficiency as in original)
        with keras.ops.stop_gradient():
            for h_step in range(self.h_cycles):
                for l_step in range(self.l_cycles):
                    # Skip last L step of last H cycle (will be done with gradients)
                    if not (h_step == self.h_cycles - 1 and l_step == self.l_cycles - 1):
                        z_l = self.l_reasoning(z_l, z_h + input_emb, training=training)

                # Skip last H step (will be done with gradients)
                if h_step != self.h_cycles - 1:
                    z_h = self.h_reasoning(z_h, z_l, training=training)

        # Final step with gradients
        z_l = self.l_reasoning(z_l, z_h + input_emb, training=training)
        z_h = self.h_reasoning(z_h, z_l, training=training)

        # Generate outputs
        # Language modeling head (skip puzzle embedding positions)
        lm_logits = self.lm_head(z_h[:, self.puzzle_emb_len:])

        # Q head (use first position)
        q_logits = self.q_head(z_h[:, 0])
        q_halt_logits = q_logits[..., 0]
        q_continue_logits = q_logits[..., 1]

        outputs = {
            "logits": lm_logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        # New carry (detached)
        new_carry = {
            "z_h": keras.ops.stop_gradient(z_h),
            "z_l": keras.ops.stop_gradient(z_l)
        }

        return new_carry, outputs

    def compute_output_shape(self, input_shape):
        """Compute output shapes."""
        batch_size = input_shape.get("token_ids", [None])[0]
        return {
            "logits": (batch_size, self.seq_len, self.vocab_size),
            "q_halt_logits": (batch_size,),
            "q_continue_logits": (batch_size,)
        }

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len,
            "embed_dim": self.embed_dim,
            "num_puzzle_identifiers": self.num_puzzle_identifiers,
            "puzzle_emb_dim": self.puzzle_emb_dim,
            "batch_size": self.batch_size,
            "h_layers": self.h_layers,
            "l_layers": self.l_layers,
            "h_cycles": self.h_cycles,
            "l_cycles": self.l_cycles,
            "num_heads": self.num_heads,
            "ffn_expansion_factor": self.ffn_expansion_factor,
            "pos_encodings": self.pos_encodings,
            "rope_theta": self.rope_theta,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "embeddings_initializer": keras.initializers.serialize(self.embeddings_initializer),
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "embeddings_regularizer": keras.regularizers.serialize(self.embeddings_regularizer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------
