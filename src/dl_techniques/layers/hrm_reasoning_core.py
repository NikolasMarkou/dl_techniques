"""
This module defines the HierarchicalReasoningCore, a stateful, recurrent Keras layer
that represents the central engine of the Hierarchical Reasoning Model (HRM) architecture.

The HierarchicalReasoningCore implements a sophisticated recurrent reasoning system that
uses Transformer-like blocks to perform iterative, multi-level reasoning with adaptive
computation capabilities. Unlike standard Transformers, this core maintains persistent
state across reasoning cycles and can dynamically allocate computational resources.

The architecture combines hierarchical reasoning modules, specialized embeddings, and
adaptive computation mechanisms to enable complex reasoning tasks that require variable
computational effort and multi-step "thinking" processes.
"""

import math
import keras
from typing import Optional, Union, Dict, Any, Tuple, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .hrm_reasoning_module import HierarchicalReasoningModule
from .hrm_sparse_puzzle_embedding import SparsePuzzleEmbedding
from .embedding.positional_embedding import PositionalEmbedding
from .embedding.rotary_position_embedding import RotaryPositionEmbedding

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class HierarchicalReasoningCore(keras.layers.Layer):
    """
    Stateful hierarchical reasoning core for complex multi-step reasoning tasks.

    This layer implements the central engine of the Hierarchical Reasoning Model (HRM),
    featuring dual-level reasoning modules, persistent state management, and adaptive
    computation capabilities. The core performs iterative refinement of high-level and
    low-level representations through multiple reasoning cycles, enabling sophisticated
    reasoning patterns for complex problem solving.

    **Intent**: Provide a recurrent reasoning architecture that can perform variable-depth
    computation through hierarchical reasoning cycles, with integrated puzzle context and
    adaptive computation mechanisms for complex cognitive tasks.

    **Architecture**:
    ```
    Input: {token_ids, puzzle_ids} + Carry: {z_h, z_l}
           ↓
    Token Embedding + Puzzle Embedding (optional)
           ↓
    Positional Encoding (Learned or RoPE)
           ↓
    ┌─────── Reasoning Cycles (with stop_gradient) ──────┐
    │  for h_cycle in range(h_cycles):                   │
    │    for l_cycle in range(l_cycles):                 │
    │      z_l = L_Reasoning(z_l, z_h + input_emb)       │
    │    z_h = H_Reasoning(z_h, z_l)                     │
    └────────────────────────────────────────────────────┘
           ↓
    Final Step (with gradients):
      z_l = L_Reasoning(z_l, z_h + input_emb)
      z_h = H_Reasoning(z_h, z_l)
           ↓
    Output Heads:
      - LM Head: z_h → logits (language modeling)
      - Q Head: z_h[0] → {halt, continue} (adaptive computation)
           ↓
    New Carry: {z_h, z_l} (detached) + Outputs
    ```

    **Key Components**:
    1. **Stateful Operation**: Maintains z_h (high-level) and z_l (low-level) states
    2. **Hierarchical Reasoning**: Separate H and L reasoning modules for multi-level processing
    3. **Iterative Cycles**: Multiple reasoning cycles with gradient control for efficiency
    4. **Puzzle Context**: Optional sparse puzzle embeddings for task-specific context
    5. **Adaptive Computation**: Q-learning head for halt/continue decisions
    6. **Efficient Training**: Gradient detachment for all but final reasoning step

    **Reasoning Cycle Details**:
    - **High-level cycles** (h_cycles): Abstract reasoning steps
    - **Low-level cycles** (l_cycles): Detailed processing steps
    - **Gradient Management**: Only final step trains to prevent vanishing gradients
    - **State Updates**: Progressive refinement of both z_h and z_l representations

    Args:
        vocab_size: Integer, size of the vocabulary for token embeddings.
            Must be positive. Determines the input space for language modeling.
        seq_len: Integer, maximum sequence length for input tokens.
            Must be positive. Defines the temporal processing window.
        embed_dim: Integer, embedding dimension throughout the model.
            Must be positive and typically divisible by num_heads.
        num_puzzle_identifiers: Integer, number of unique puzzle identifiers.
            Must be positive. Used for task-specific context embedding.
        puzzle_emb_dim: Integer, dimension of puzzle embeddings.
            Use 0 to disable puzzle embeddings. When > 0, provides task context.
        batch_size: Integer, batch size for training and state management.
            Must be positive. Used for stateful operations and initialization.
        h_layers: Integer, number of layers in high-level reasoning module.
            Must be positive. Controls depth of abstract reasoning.
        l_layers: Integer, number of layers in low-level reasoning module.
            Must be positive. Controls depth of detailed processing.
        h_cycles: Integer, number of high-level reasoning cycles per forward pass.
            Must be positive. More cycles enable deeper abstract reasoning.
        l_cycles: Integer, number of low-level reasoning cycles per h_cycle.
            Must be positive. More cycles enable more detailed processing.
        num_heads: Integer, number of attention heads in reasoning modules.
            Must be positive and divide embed_dim evenly.
        ffn_expansion_factor: Integer, expansion factor for feed-forward networks.
            Must be positive. Typically 4 for standard transformer scaling.
        pos_encodings: String, type of positional encoding to use.
            Must be "rope" or "learned". Affects how sequence positions are encoded.
        rope_theta: Float, theta parameter for rotary position embeddings.
            Only used when pos_encodings="rope". Defaults to 10000.0.
        dropout_rate: Float between 0 and 1, dropout rate for regularization.
            Applied throughout reasoning modules. Defaults to 0.0.
        use_bias: Boolean, whether to use bias parameters in linear layers.
            Defaults to False following modern practices.
        embeddings_initializer: String or Initializer, initialization for embeddings.
            Defaults to 'truncated_normal' for stable training.
        kernel_initializer: String or Initializer, initialization for kernel weights.
            Defaults to 'he_normal' for good gradient flow.
        embeddings_regularizer: Optional Regularizer for embedding weights.
        kernel_regularizer: Optional Regularizer for kernel weights.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        Dictionary with:
        - carry: Dict with "z_h" and "z_l" tensors of shape
          `(batch_size, total_seq_len, embed_dim)`
        - inputs: Dict with:
          - "token_ids": `(batch_size, seq_len)`
          - "puzzle_ids": `(batch_size,)` if puzzle_emb_dim > 0

    Output shape:
        Tuple of (new_carry, outputs) where:
        - new_carry: Dict with "z_h", "z_l" of shape `(batch_size, total_seq_len, embed_dim)`
        - outputs: Dict with:
          - "logits": `(batch_size, seq_len, vocab_size)`
          - "q_halt_logits": `(batch_size,)`
          - "q_continue_logits": `(batch_size,)`

    Attributes:
        token_embedding: Embedding layer for input tokens.
        puzzle_embedding: SparsePuzzleEmbedding for task context (if enabled).
        position_embedding: PositionalEmbedding for learned positions (if used).
        rope: RotaryPositionEmbedding for rotary positions (if used).
        h_reasoning: HierarchicalReasoningModule for high-level processing.
        l_reasoning: HierarchicalReasoningModule for low-level processing.
        lm_head: Dense layer for language modeling predictions.
        q_head: Dense layer for adaptive computation decisions.
        h_init: Initial high-level state parameter.
        l_init: Initial low-level state parameter.

    Methods:
        empty_carry(batch_size): Create empty carry state for initialization.
        reset_carry(reset_flag, carry): Reset carry state based on halt decisions.

    Example:
        ```python
        # Basic configuration
        core = HierarchicalReasoningCore(
            vocab_size=10000,
            seq_len=128,
            embed_dim=512,
            num_puzzle_identifiers=100,
            puzzle_emb_dim=64,
            batch_size=32
        )

        # Initialize carry state
        carry = core.empty_carry(batch_size=32)

        # Prepare inputs
        inputs = {
            "token_ids": keras.random.randint((32, 128), 0, 10000),
            "puzzle_ids": keras.random.randint((32,), 0, 100)
        }

        # Forward pass
        new_carry, outputs = core(carry, inputs)

        # Access outputs
        logits = outputs["logits"]  # (32, 128, 10000)
        halt_logits = outputs["q_halt_logits"]  # (32,)

        # Advanced configuration with more reasoning cycles
        advanced_core = HierarchicalReasoningCore(
            vocab_size=32000,
            seq_len=256,
            embed_dim=768,
            num_puzzle_identifiers=500,
            puzzle_emb_dim=128,
            batch_size=16,
            h_layers=6,
            l_layers=6,
            h_cycles=3,
            l_cycles=3,
            num_heads=12,
            pos_encodings="rope",
            dropout_rate=0.1
        )
        ```

    Note:
        This layer implements sophisticated gradient management where only the final
        reasoning step accumulates gradients, making training efficient across many
        cycles while still learning effective reasoning patterns.
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
        pos_encodings: Literal["rope", "learned"] = "rope",
        rope_theta: float = 10000.0,
        dropout_rate: float = 0.0,
        use_bias: bool = False,
        embeddings_initializer: Union[str, keras.initializers.Initializer] = "truncated_normal",
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        embeddings_regularizer: Optional[keras.regularizers.Regularizer] = None,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_puzzle_identifiers <= 0:
            raise ValueError(f"num_puzzle_identifiers must be positive, got {num_puzzle_identifiers}")
        if puzzle_emb_dim < 0:
            raise ValueError(f"puzzle_emb_dim must be non-negative, got {puzzle_emb_dim}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if h_layers <= 0:
            raise ValueError(f"h_layers must be positive, got {h_layers}")
        if l_layers <= 0:
            raise ValueError(f"l_layers must be positive, got {l_layers}")
        if h_cycles <= 0:
            raise ValueError(f"h_cycles must be positive, got {h_cycles}")
        if l_cycles <= 0:
            raise ValueError(f"l_cycles must be positive, got {l_cycles}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if ffn_expansion_factor <= 0:
            raise ValueError(f"ffn_expansion_factor must be positive, got {ffn_expansion_factor}")
        if pos_encodings not in ["rope", "learned"]:
            raise ValueError(f"pos_encodings must be 'rope' or 'learned', got {pos_encodings}")
        if rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {rope_theta}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store configuration parameters
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
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Calculate puzzle embedding sequence length
        self.puzzle_emb_len = max(1, (puzzle_emb_dim + embed_dim - 1) // embed_dim) if puzzle_emb_dim > 0 else 0
        self.total_seq_len = seq_len + self.puzzle_emb_len

        # Embedding scale (like in original Transformer)
        self.embed_scale = math.sqrt(embed_dim)

        # CREATE all sub-layers in __init__ (they are unbuilt)

        # Token embeddings with scaled initialization
        embed_init_std = 1.0 / self.embed_scale
        self.token_embedding = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=embed_init_std),
            embeddings_regularizer=self.embeddings_regularizer,
            name="token_embedding"
        )

        # Puzzle embeddings (if enabled)
        if puzzle_emb_dim > 0:
            self.puzzle_embedding = SparsePuzzleEmbedding(
                num_embeddings=num_puzzle_identifiers,
                embedding_dim=puzzle_emb_dim,
                batch_size=batch_size,
                embeddings_initializer="zeros",  # Zero init as in original
                embeddings_regularizer=self.embeddings_regularizer,
                name="puzzle_embedding"
            )
        else:
            self.puzzle_embedding = None

        # Positional embeddings
        if pos_encodings == "rope":
            self.rope = RotaryPositionEmbedding(
                head_dim=embed_dim // num_heads,
                max_seq_len=self.total_seq_len,
                rope_theta=rope_theta,
                name="rope"
            )
            self.position_embedding = None
        elif pos_encodings == "learned":
            self.position_embedding = PositionalEmbedding(
                max_seq_len=self.total_seq_len,
                dim=embed_dim,
                dropout=0.0,  # No dropout in original
                name="position_embedding"
            )
            self.rope = None

        # Reasoning modules
        self.h_reasoning = HierarchicalReasoningModule(
            num_layers=h_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="h_reasoning"
        )

        self.l_reasoning = HierarchicalReasoningModule(
            num_layers=l_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="l_reasoning"
        )

        # Output heads
        self.lm_head = keras.layers.Dense(
            vocab_size,
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

        # Weight attributes (created in build)
        self.h_init = None
        self.l_init = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        """
        # Create initial states for reasoning modules
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

        # Build sub-layers in computational order
        # Token embeddings expect (batch_size, seq_len)
        token_input_shape = (None, self.seq_len)
        self.token_embedding.build(token_input_shape)

        # Puzzle embeddings expect (batch_size,)
        if self.puzzle_embedding is not None:
            puzzle_input_shape = (None,)
            self.puzzle_embedding.build(puzzle_input_shape)

        # Positional embeddings
        if self.position_embedding is not None:
            # PositionalEmbedding doesn't need explicit building
            pass
        if self.rope is not None:
            # RoPE doesn't need explicit building for its parameters
            pass

        # Reasoning modules expect (batch_size, seq_len, embed_dim)
        reasoning_input_shape = (None, self.total_seq_len, self.embed_dim)
        self.h_reasoning.build(reasoning_input_shape)
        self.l_reasoning.build(reasoning_input_shape)

        # Output heads
        # LM head expects (batch_size, seq_len, embed_dim)
        lm_input_shape = (None, self.seq_len, self.embed_dim)
        self.lm_head.build(lm_input_shape)

        # Q head expects (batch_size, embed_dim)
        q_input_shape = (None, self.embed_dim)
        self.q_head.build(q_input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def _input_embeddings(
        self,
        token_ids: keras.KerasTensor,
        puzzle_ids: Optional[keras.KerasTensor]
    ) -> keras.KerasTensor:
        """
        Create input embeddings from token and puzzle IDs.

        Args:
            token_ids: Token ID tensor of shape (batch_size, seq_len).
            puzzle_ids: Puzzle ID tensor of shape (batch_size,) or None.

        Returns:
            Combined embeddings tensor of shape (batch_size, total_seq_len, embed_dim).
        """
        # Token embeddings
        token_emb = self.token_embedding(token_ids)

        # Add puzzle embeddings if enabled
        if self.puzzle_emb_dim > 0 and self.puzzle_embedding is not None:
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
        if self.pos_encodings == "learned" and self.position_embedding is not None:
            # Scale by 1/sqrt(2) to maintain variance as in original
            embeddings = 0.707106781 * (embeddings + self.position_embedding.position_embeddings)

        # Scale embeddings
        return self.embed_scale * embeddings

    def empty_carry(self, batch_size: int) -> Dict[str, keras.KerasTensor]:
        """
        Create empty carry state for initialization.

        Args:
            batch_size: Batch size for state tensors.

        Returns:
            Dictionary with "z_h" and "z_l" zero-initialized tensors.
        """
        return {
            "z_h": keras.ops.zeros((batch_size, self.total_seq_len, self.embed_dim)),
            "z_l": keras.ops.zeros((batch_size, self.total_seq_len, self.embed_dim))
        }

    def reset_carry(
        self,
        reset_flag: keras.KerasTensor,
        carry: Dict[str, keras.KerasTensor]
    ) -> Dict[str, keras.KerasTensor]:
        """
        Reset carry state for halted sequences.

        Args:
            reset_flag: Boolean tensor of shape (batch_size,) indicating which
                sequences to reset.
            carry: Current carry state dictionary.

        Returns:
            Updated carry state with reset sequences reinitialized.
        """
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

    def call(
        self,
        carry: Dict[str, keras.KerasTensor],
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None
    ) -> Tuple[Dict[str, keras.KerasTensor], Dict[str, keras.KerasTensor]]:
        """
        Forward pass through the hierarchical reasoning core.

        Args:
            carry: Current carry state dict with "z_h" and "z_l" tensors.
            inputs: Input dict with "token_ids" and optionally "puzzle_ids".
            training: Boolean indicating training mode.

        Returns:
            Tuple of (new_carry, outputs) where:
            - new_carry: Updated carry state dict (detached from gradients)
            - outputs: Dict with "logits", "q_halt_logits", "q_continue_logits"
        """
        # Get input embeddings
        puzzle_ids = inputs.get("puzzle_ids") if self.puzzle_emb_dim > 0 else None
        input_emb = self._input_embeddings(inputs["token_ids"], puzzle_ids)

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

        # New carry (detached from gradients for efficiency)
        new_carry = {
            "z_h": keras.ops.stop_gradient(z_h),
            "z_l": keras.ops.stop_gradient(z_l)
        }

        return new_carry, outputs

    def compute_output_shape(
        self,
        input_shape: Dict[str, Tuple[Optional[int], ...]]
    ) -> Tuple[Dict[str, Tuple[Optional[int], ...]], Dict[str, Tuple[Optional[int], ...]]]:
        """
        Compute output shapes for new_carry and outputs.

        Args:
            input_shape: Input shape dictionary.

        Returns:
            Tuple of (new_carry_shape, outputs_shape) dictionaries.
        """
        batch_size = None
        if "token_ids" in input_shape:
            batch_size = input_shape["token_ids"][0]

        new_carry_shape = {
            "z_h": (batch_size, self.total_seq_len, self.embed_dim),
            "z_l": (batch_size, self.total_seq_len, self.embed_dim)
        }

        outputs_shape = {
            "logits": (batch_size, self.seq_len, self.vocab_size),
            "q_halt_logits": (batch_size,),
            "q_continue_logits": (batch_size,)
        }

        return new_carry_shape, outputs_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns:
            Dictionary containing all constructor parameters for proper serialization.
        """
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