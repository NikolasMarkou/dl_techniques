"""
This module defines the HierarchicalReasoningModel, a Keras Model that wraps the
HierarchicalReasoningCore with a stateful mechanism for Adaptive Computation Time (ACT).

This model acts as a high-level controller, orchestrating the iterative reasoning
process performed by its inner `core`. Instead of a fixed number of layers, this model
can learn to dynamically allocate computational resources, running the `core` for a
variable number of "thinking steps" depending on the input's complexity.

Core Concepts:

1.  **Wrapper Architecture:**
    This class is primarily a stateful wrapper. It contains an instance of the
    `HierarchicalReasoningCore` and manages the state required for iterative
    computation.

2.  **Adaptive Computation Time (ACT):**
    The model's main feature is its ability to perform a variable number of
    reasoning steps. The `call` method can run a loop, repeatedly invoking the `core`
    until a halting condition is met for all items in a batch. This allows the model
    to "think" longer about more difficult problems.

3.  **State Management (The 'carry'):**
    The model is stateful and operates on a `carry` dictionary that tracks:
    -   `inner_carry`: The high-level (`z_h`) and low-level (`z_l`) states of the
        `HierarchicalReasoningCore`.
    -   `steps`: The number of computation steps taken for each sequence.
    -   `halted`: A boolean flag indicating if a sequence has finished its computation.
    -   `current_data`: A cache of the input data for each sequence.

4.  **Learned Halting Mechanism:**
    During training, the decision to halt is learned via Q-learning. The `core`
    produces Q-values for "halt" and "continue" actions. This model uses these
    values to decide whether to stop or perform another reasoning step. The process
    includes:
    -   An exploration strategy (`halt_exploration_prob`) to encourage trying
        different computation depths.
    -   A hard limit (`halt_max_steps`) to prevent infinite loops.
    -   Calculation of a target Q-value for training the Q-learning head via
        bootstrapping from the next state's value.

5.  **Dual `call` Interface:**
    The model supports two modes of operation through its `call` method:
    -   **Complete Mode `model(batch)`:** Runs the internal loop until all sequences
        in the batch have halted, returning the final outputs.
    -   **Single-Step Mode `model((carry, batch))`:** Executes exactly one reasoning
        step and returns the new state, the step's outputs, and a flag
        indicating if all sequences are now finished. This is useful for custom
        training loops or detailed analysis.
"""

import keras
from typing import Optional, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.hrm_reasoning_core import HierarchicalReasoningCore

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class HierarchicalReasoningModel(keras.Model):
    """
    Hierarchical Reasoning Model with Adaptive Computation Time.

    This model wraps the hierarchical reasoning core with ACT mechanisms
    for variable computation depth and halting decisions.

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
        halt_max_steps: Maximum computation steps before forced halt
        halt_exploration_prob: Probability of exploration in Q-learning
        dropout_rate: Dropout rate
        use_bias: Whether to use bias in linear layers
        embeddings_initializer: Initializer for embeddings
        kernel_initializer: Initializer for kernel weights
        embeddings_regularizer: Regularizer for embeddings
        kernel_regularizer: Regularizer for kernel weights
        **kwargs: Additional model arguments
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
            halt_max_steps: int = 16,
            halt_exploration_prob: float = 0.1,
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
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.embeddings_initializer = embeddings_initializer
        self.kernel_initializer = kernel_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.kernel_regularizer = kernel_regularizer

        # Core reasoning model
        self.core = HierarchicalReasoningCore(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_puzzle_identifiers=num_puzzle_identifiers,
            puzzle_emb_dim=puzzle_emb_dim,
            batch_size=batch_size,
            h_layers=h_layers,
            l_layers=l_layers,
            h_cycles=h_cycles,
            l_cycles=l_cycles,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            pos_encodings=pos_encodings,
            rope_theta=rope_theta,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            embeddings_initializer=embeddings_initializer,
            kernel_initializer=kernel_initializer,
            embeddings_regularizer=embeddings_regularizer,
            kernel_regularizer=kernel_regularizer,
            name="core"
        )

    def initial_carry(self, batch):
        """Initialize carry state for a batch."""
        batch_size = keras.ops.shape(batch["token_ids"])[0]

        return {
            # Core reasoning state
            "inner_carry": self.core.empty_carry(batch_size),

            # ACT state
            "steps": keras.ops.zeros((batch_size,), dtype="int32"),
            "halted": keras.ops.ones((batch_size,), dtype="bool"),  # Start halted

            # Current data cache
            "current_data": {k: keras.ops.zeros_like(v) for k, v in batch.items()}
        }

    def call(self, inputs, training=None):
        """
        Forward pass through the model.

        This method can be called in two ways:
        1. Standard call: call(batch) - runs until convergence
        2. Step call: call(carry, batch) - single reasoning step

        Args:
            inputs: Either batch dict or (carry, batch) tuple
            training: Whether in training mode

        Returns:
            If standard call: final outputs
            If step call: (new_carry, outputs, all_finished)
        """
        if isinstance(inputs, dict):
            # Standard call - run until convergence
            return self._forward_complete(inputs, training=training)
        else:
            # Step call
            carry, batch = inputs
            return self._forward_step(carry, batch, training=training)

    def _forward_complete(self, batch, training=None):
        """Run complete forward pass until all sequences halt."""
        carry = self.initial_carry(batch)
        outputs = None

        # Run steps until all sequences halt
        max_iterations = self.halt_max_steps * 2  # Safety limit
        for _ in range(max_iterations):
            carry, outputs, all_finished = self._forward_step(carry, batch, training=training)
            if all_finished:
                break

        return outputs

    def _forward_step(self, carry, batch, training=None):
        """Single reasoning step with ACT logic."""
        # Update carry for new sequences (halted ones get reset)
        new_inner_carry = self.core.reset_carry(carry["halted"], carry["inner_carry"])

        # Reset steps for halted sequences
        new_steps = keras.ops.where(carry["halted"], 0, carry["steps"])

        # Update current data for halted sequences
        new_current_data = {}
        for k, v in carry["current_data"].items():
            reset_mask = keras.ops.reshape(carry["halted"], [-1] + [1] * (len(v.shape) - 1))
            new_current_data[k] = keras.ops.where(reset_mask, batch[k], v)

        # Forward pass through core
        new_inner_carry, outputs = self.core(
            new_inner_carry,
            {"token_ids": new_current_data["token_ids"],
             "puzzle_ids": new_current_data["puzzle_ids"]},
            training=training
        )

        # Update steps
        new_steps = new_steps + 1
        is_last_step = new_steps >= self.halt_max_steps

        # Determine halting
        halted = is_last_step

        if training and self.halt_max_steps > 1:
            # Q-learning based halting
            q_halt = outputs["q_halt_logits"]
            q_continue = outputs["q_continue_logits"]

            # Halt if q_halt > q_continue
            halted = halted | (q_halt > q_continue)

            # Exploration: random minimum halt steps
            if self.halt_exploration_prob > 0:
                explore_mask = keras.random.uniform(keras.ops.shape(q_halt)) < self.halt_exploration_prob
                min_steps = keras.random.uniform(
                    keras.ops.shape(new_steps),
                    minval=2,
                    maxval=self.halt_max_steps + 1,
                    dtype="int32"
                )
                min_halt_steps = keras.ops.where(explore_mask, min_steps, 1)
                halted = halted & (new_steps >= min_halt_steps)

            # Compute target Q for bootstrapping (as in original)
            if not is_last_step:
                # Get next step Q values for target computation
                next_inner_carry, next_outputs = self.core(
                    new_inner_carry,
                    {"token_ids": new_current_data["token_ids"],
                     "puzzle_ids": new_current_data["puzzle_ids"]},
                    training=training
                )

                next_q_halt = next_outputs["q_halt_logits"]
                next_q_continue = next_outputs["q_continue_logits"]

                # Target Q: if last step, use halt; otherwise use max
                target_q = keras.ops.where(
                    is_last_step,
                    keras.ops.sigmoid(next_q_halt),
                    keras.ops.sigmoid(keras.ops.maximum(next_q_halt, next_q_continue))
                )
                outputs["target_q_continue"] = target_q

        # Create new carry
        new_carry = {
            "inner_carry": new_inner_carry,
            "steps": new_steps,
            "halted": halted,
            "current_data": new_current_data
        }

        # Check if all sequences are finished
        all_finished = keras.ops.all(halted)

        return new_carry, outputs, all_finished

    def get_config(self):
        """Get model configuration."""
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
            "halt_max_steps": self.halt_max_steps,
            "halt_exploration_prob": self.halt_exploration_prob,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "embeddings_initializer": keras.initializers.serialize(self.embeddings_initializer),
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "embeddings_regularizer": keras.regularizers.serialize(self.embeddings_regularizer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------

def create_hierarchical_reasoning_model(
        vocab_size: int,
        seq_len: int,
        embed_dim: int = 512,
        num_puzzle_identifiers: int = 1000,
        puzzle_emb_dim: int = 512,
        batch_size: int = 32,
        h_layers: int = 4,
        l_layers: int = 4,
        h_cycles: int = 2,
        l_cycles: int = 2,
        num_heads: int = 8,
        ffn_expansion_factor: int = 4,
        pos_encodings: str = "rope",
        rope_theta: float = 10000.0,
        halt_max_steps: int = 16,
        halt_exploration_prob: float = 0.1,
        dropout_rate: float = 0.0,
        use_bias: bool = False,
        **kwargs
) -> HierarchicalReasoningModel:
    """
    Create a Hierarchical Reasoning Model.

    Args:
        vocab_size: Size of vocabulary
        seq_len: Maximum sequence length
        embed_dim: Embedding dimension
        num_puzzle_identifiers: Number of puzzle identifiers
        puzzle_emb_dim: Puzzle embedding dimension
        batch_size: Batch size for training
        h_layers: Number of high-level reasoning layers
        l_layers: Number of low-level reasoning layers
        h_cycles: Number of high-level reasoning cycles
        l_cycles: Number of low-level reasoning cycles
        num_heads: Number of attention heads
        ffn_expansion_factor: Feed-forward expansion factor
        pos_encodings: Type of positional encodings
        rope_theta: RoPE theta parameter
        halt_max_steps: Maximum computation steps
        halt_exploration_prob: Exploration probability for Q-learning
        dropout_rate: Dropout rate
        use_bias: Whether to use bias
        **kwargs: Additional arguments

    Returns:
        Configured HierarchicalReasoningModel
    """
    return HierarchicalReasoningModel(
        vocab_size=vocab_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_puzzle_identifiers=num_puzzle_identifiers,
        puzzle_emb_dim=puzzle_emb_dim,
        batch_size=batch_size,
        h_layers=h_layers,
        l_layers=l_layers,
        h_cycles=h_cycles,
        l_cycles=l_cycles,
        num_heads=num_heads,
        ffn_expansion_factor=ffn_expansion_factor,
        pos_encodings=pos_encodings,
        rope_theta=rope_theta,
        halt_max_steps=halt_max_steps,
        halt_exploration_prob=halt_exploration_prob,
        dropout_rate=dropout_rate,
        use_bias=use_bias,
        **kwargs
    )

# ---------------------------------------------------------------------
