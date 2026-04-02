"""
A stateful, recurrent engine for the Hierarchical Reasoning Model (HRM).

This layer serves as the "brain" of the HRM, designed to tackle complex tasks requiring
multi-step, iterative reasoning. It operates as a sophisticated Recurrent Neural
Network (RNN) cell that maintains persistent state across time steps, but internally
uses Transformer-based blocks to perform deep computation. Its key innovation lies
in its dual-state memory, adaptive computation mechanism, and efficient gradient
management for "thinking" cycles.

Architecture:
    The core maintains two distinct, persistent memory states that evolve over time:
    1.  `z_h` (High-level): Represents abstract plans, global context, or "System 2"
        slow thinking.
    2.  `z_l` (Low-level): Represents detailed execution, local features, or "System 1"
        fast processing.

    In each forward pass, the core accepts new inputs (token embeddings and optional
    task-specific puzzle embeddings) and the previous states (`carry`). It then
    engages in multiple internal "reasoning cycles." In these cycles, two
    specialized Transformer modules interact:
    -   The Low-Level module (`l_reasoning`) updates `z_l` by attending to the
        current `z_h` and the new input.
    -   The High-Level module (`h_reasoning`) updates `z_h` by attending to the
        newly updated `z_l`.

    This ping-pong structure allows the high-level state to guide the low-level
    processing, and the low-level results to inform and refine the high-level plan.

Foundational Mathematics and Concepts:
    -   **Hierarchical/Dual-Process Reasoning:** The architecture is inspired by
        cognitive theories positing two systems of thought. `z_h` acts as a
        Global Workspace, broadcasting high-level directives, while `z_l` performs
        specialized processing grounded in the input.
        `z_l_new = L_Module(z_l, condition=z_h + input)`
        `z_h_new = H_Module(z_h, condition=z_l_new)`

    -   **Adaptive Computation Time (ACT):** The layer includes a `q_head` which
        projects the high-level state `z_h` to a halting probability (logits for
        `halt` vs. `continue`). This allows an external control loop to dynamically
        decide how many times to invoke this core (how long to "think") for a
        given problem, allocating more computation to harder tasks.

    -   **Gradient Detachment and Truncated Backpropagation:** To enable deep
        reasoning (many cycles) without the prohibitive memory cost and instability
        of full Backpropagation Through Time (BPTT), this layer employs a crucial
        optimization. It runs multiple inner reasoning cycles (`h_cycles` * `l_cycles`)
        where the intermediate states are explicitly detached from the computation
        graph using `stop_gradient`. Only the operations in the very final cycle
        are recorded for backpropagation. This allows the model to leverage the
        informational benefit of many thinking steps while only training on the
        final output, similar to Truncated BPTT but applied to "thinking depth."

References:
    1.  Graves, A. (2016). "Adaptive Computation Time for Recurrent Neural Networks."
        Provides the theoretical basis for the `q_head` and dynamic halting.
    2.  Baars, B. J. (1988). "A Cognitive Theory of Consciousness." (Global Workspace
        Theory). The interaction between the global `z_h` and the specialized `z_l`
        shares conceptual similarities with GWT.
    3.  Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position
        Embedding." The layer supports RoPE (`rope_theta`) as a modern positional
        encoding strategy.
"""

import math
import keras
from typing import Optional, Union, Dict, Any, Tuple, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .hrm_reasoning_module import HierarchicalReasoningModule
from .hrm_sparse_puzzle_embedding import SparsePuzzleEmbedding
from ..embedding.positional_embedding import PositionalEmbedding
from ..embedding.rotary_position_embedding import RotaryPositionEmbedding

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class HierarchicalReasoningCore(keras.layers.Layer):
    """
    Stateful hierarchical reasoning core for multi-step cognitive tasks.

    Implements the central engine of the Hierarchical Reasoning Model (HRM),
    maintaining dual persistent states: ``z_h`` (high-level abstract plans)
    and ``z_l`` (low-level detailed features). Each forward pass performs
    iterative reasoning cycles where ``z_l = L_Module(z_l, z_h + input_emb)``
    and ``z_h = H_Module(z_h, z_l)`` refine both representations. Only the
    final cycle accumulates gradients (truncated BPTT), enabling deep
    reasoning without prohibitive memory cost. An adaptive computation
    ``q_head`` produces halt/continue logits for dynamic depth control.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────────────────┐
        │           HierarchicalReasoningCore                  │
        │                                                      │
        │  {token_ids, puzzle_ids} + Carry: {z_h, z_l}         │
        │              │                                       │
        │              ▼                                       │
        │  Token Embedding + Puzzle Embedding (optional)       │
        │  + Positional Encoding (Learned / RoPE)              │
        │              │                                       │
        │              ▼                                       │
        │  ┌─── Reasoning Cycles (stop_gradient) ────┐         │
        │  │  for h in h_cycles:                     │         │
        │  │    for l in l_cycles:                   │         │
        │  │      z_l ← L_Reasoning(z_l, z_h+emb)    │         │
        │  │    z_h ← H_Reasoning(z_h, z_l)          │         │
        │  └─────────────────────────────────────────┘         │
        │              │                                       │
        │              ▼                                       │
        │  Final Step (with gradients):                        │
        │    z_l ← L_Reasoning(z_l, z_h + emb)                 │
        │    z_h ← H_Reasoning(z_h, z_l)                       │
        │              │                                       │
        │         ┌────┴────┐                                  │
        │         ▼         ▼                                  │
        │  LM Head(logits)  Q Head(halt/continue)              │
        │              │                                       │
        │              ▼                                       │
        │  New Carry: {z_h, z_l} (detached)                    │
        └──────────────────────────────────────────────────────┘

    :param vocab_size: Size of the vocabulary for token embeddings.
    :type vocab_size: int
    :param seq_len: Maximum sequence length for input tokens.
    :type seq_len: int
    :param embed_dim: Embedding dimension throughout the model.
    :type embed_dim: int
    :param num_puzzle_identifiers: Number of unique puzzle identifiers.
    :type num_puzzle_identifiers: int
    :param puzzle_emb_dim: Dimension of puzzle embeddings. Use 0 to disable.
    :type puzzle_emb_dim: int
    :param batch_size: Batch size for stateful operations.
    :type batch_size: int
    :param h_layers: Number of layers in high-level reasoning module.
    :type h_layers: int
    :param l_layers: Number of layers in low-level reasoning module.
    :type l_layers: int
    :param h_cycles: Number of high-level reasoning cycles per forward pass.
    :type h_cycles: int
    :param l_cycles: Number of low-level reasoning cycles per h_cycle.
    :type l_cycles: int
    :param num_heads: Number of attention heads in reasoning modules.
    :type num_heads: int
    :param ffn_expansion_factor: FFN expansion factor. Defaults to 4.
    :type ffn_expansion_factor: int
    :param pos_encodings: Positional encoding type (``"rope"`` or ``"learned"``).
    :type pos_encodings: Literal["rope", "learned"]
    :param rope_theta: Theta parameter for RoPE. Defaults to 10000.0.
    :type rope_theta: float
    :param dropout_rate: Dropout rate for regularization. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether to use bias in linear layers. Defaults to False.
    :type use_bias: bool
    :param embeddings_initializer: Initialization for embeddings.
    :type embeddings_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_initializer: Initialization for kernel weights.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param embeddings_regularizer: Optional regularizer for embeddings.
    :type embeddings_regularizer: Optional[keras.regularizers.Regularizer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional arguments for Layer base class.
    :type kwargs: Any
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
                dropout_rate=0.0,  # No dropout in original
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

        # Positional embeddings - explicitly build for robust serialization
        if self.position_embedding is not None:
            pos_input_shape = (None, self.total_seq_len, self.embed_dim)
            self.position_embedding.build(pos_input_shape)
        if self.rope is not None:
            head_dim = self.embed_dim // self.num_heads
            rope_input_shape = (None, self.num_heads, self.total_seq_len, head_dim)
            self.rope.build(rope_input_shape)

        # Reasoning modules expect list of two shapes: [hidden_states, input_injection]
        reasoning_input_shape = (None, self.total_seq_len, self.embed_dim)
        self.h_reasoning.build([reasoning_input_shape, reasoning_input_shape])
        self.l_reasoning.build([reasoning_input_shape, reasoning_input_shape])

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

        :param token_ids: Token ID tensor of shape ``(batch_size, seq_len)``.
        :type token_ids: keras.KerasTensor
        :param puzzle_ids: Puzzle ID tensor of shape ``(batch_size,)`` or None.
        :type puzzle_ids: Optional[keras.KerasTensor]
        :return: Combined embeddings of shape ``(batch_size, total_seq_len, embed_dim)``.
        :rtype: keras.KerasTensor
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
            embeddings = 0.707106781 * (embeddings + self.position_embedding.pos_embedding)

        # Scale embeddings
        return self.embed_scale * embeddings

    def empty_carry(self, batch_size: int) -> Dict[str, keras.KerasTensor]:
        """
        Create empty carry state for initialization.

        :param batch_size: Batch size for state tensors.
        :type batch_size: int
        :return: Dictionary with ``"z_h"`` and ``"z_l"`` zero-initialized tensors.
        :rtype: Dict[str, keras.KerasTensor]
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

        :param reset_flag: Boolean tensor of shape ``(batch_size,)`` indicating
            which sequences to reset.
        :type reset_flag: keras.KerasTensor
        :param carry: Current carry state dictionary.
        :type carry: Dict[str, keras.KerasTensor]
        :return: Updated carry state with reset sequences reinitialized.
        :rtype: Dict[str, keras.KerasTensor]
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

        :param carry: Current carry state dict with ``"z_h"`` and ``"z_l"`` tensors.
        :type carry: Dict[str, keras.KerasTensor]
        :param inputs: Input dict with ``"token_ids"`` and optionally ``"puzzle_ids"``.
        :type inputs: Dict[str, keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Tuple of (new_carry, outputs).
        :rtype: Tuple[Dict[str, keras.KerasTensor], Dict[str, keras.KerasTensor]]
        """
        # Get input embeddings
        puzzle_ids = inputs.get("puzzle_ids") if self.puzzle_emb_dim > 0 else None
        input_emb = self._input_embeddings(inputs["token_ids"], puzzle_ids)

        z_h, z_l = carry["z_h"], carry["z_l"]

        # Forward iterations (detached for efficiency as in original)
        for h_step in range(self.h_cycles):
            for l_step in range(self.l_cycles):
                # Skip last L step of last H cycle (will be done with gradients)
                if not (h_step == self.h_cycles - 1 and l_step == self.l_cycles - 1):
                    z_l = self.l_reasoning([z_l, z_h + input_emb], training=training)

            # Skip last H step (will be done with gradients)
            if h_step != self.h_cycles - 1:
                z_h = self.h_reasoning([z_h, z_l], training=training)

        # Detach states before final step (truncated BPTT)
        z_h = keras.ops.stop_gradient(z_h)
        z_l = keras.ops.stop_gradient(z_l)

        # Final step with gradients
        z_l = self.l_reasoning([z_l, z_h + input_emb], training=training)
        z_h = self.h_reasoning([z_h, z_l], training=training)

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

        :param input_shape: Input shape dictionary.
        :type input_shape: Dict[str, Tuple[Optional[int], ...]]
        :return: Tuple of (new_carry_shape, outputs_shape) dictionaries.
        :rtype: Tuple[Dict, Dict]
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

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]
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