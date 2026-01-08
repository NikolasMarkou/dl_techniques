"""
Tiny Recursive Model (TRM) with Adaptive Computation Time (ACT).

This model performs recursive reasoning to solve complex tasks by repeatedly
applying a small, shared neural network over a variable number of steps. It
integrates the principles of Adaptive Computation Time (ACT) to learn how
many computational steps are necessary for a given problem, allowing it to
dynamically allocate resources. The model's `call` method encapsulates
a single step of this adaptive, recursive process.

The architecture is conceptually divided into two nested levels of recursion:
1.  An outer ACT loop, managed externally by the training script, which
    calls this model's `call` method repeatedly. This loop allows the
    model to progressively refine its solution over multiple "thought" steps.
2.  An inner, fixed-cycle reasoning process within the `TRMInner` submodule.
    During each single call to this model's `call` method, the inner
    module performs a multi-step update of its latent states (`z_H`, `z_L`),
    representing a focused burst of computation.

The core mathematical principle is Adaptive Computation Time (ACT). At each
step of the outer loop, the model produces not only a prediction but also a
halting probability via a dedicated `q_head`. This probability is used to
decide whether to continue processing or to terminate for a given example
in the batch. The model's state (the `carry` object) is passed from one
step to the next. For sequences that have not halted, their latent states
are updated by the `TRMInner` module. This design makes the model highly
parameter-efficient, as the same small network is reused across all steps,
and computationally efficient, as it can stop processing easier inputs
early.

References:
    - Jolicoeur-Martineau, A. (2025). Less is More: Recursive Reasoning
      with Tiny Networks. arXiv preprint arXiv:2510.04871.
    - Graves, A. (2016). Adaptive Computation Time for Recurrent Neural
      Networks. Advances in Neural Information Processing Systems, 29.
      https://proceedings.neurips.cc/paper/2016/file/bf69145244511593c662c12aee4608c0-Paper.pdf
    - Wang, G., et al. (2025). Hierarchical Reasoning Model. arXiv
      preprint arXiv:2506.21734. (Inspirational basis for TRM).
"""

import keras
from keras import ops
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.transformers import (
    FFNType,
    AttentionType,
    NormalizationType,
    NormalizationPositionType
)

from .components import TRMInner

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TRM(keras.Model):
    """
    Tiny Recursive Model (TRM) with Adaptive Computation Time (ACT).

    This model recursively refines its predictions over a variable number of steps.
    It starts with an initial state and iteratively updates it using a core reasoning
    module (`TRMInner`). The decision to continue or halt is learned, allowing
    the model to allocate more computation to harder problems.

    **Intent**: To solve complex reasoning tasks with a highly parameter-efficient
    model by applying the same small network (`TRMInner`) recursively, progressively
    improving the solution.

    **Architecture**:
    The model's `call` method implements a single step of the ACT loop. An external
    training script is expected to manage the state (`carry`) and loop until all
    sequences in the batch have halted. The core computation is delegated to the
    `TRMInner` layer.

    **State Management**:
    - `carry`: A dictionary holding the state between steps, including:
      - `inner_carry`: The `z_H` and `z_L` latent states for `TRMInner`.
      - `steps`: The current step count for each item in the batch.
      - `halted`: A boolean mask indicating which items have halted.
      - `current_data`: The input data for non-halted items.

    Args:
        vocab_size (int): Size of the vocabulary for token embeddings.
        hidden_size (int): Dimensionality of hidden states.
        num_heads (int): Number of attention heads in transformer layers.
        expansion (float): Factor to determine FFN intermediate size.
        seq_len (int): Length of the input sequence (excluding puzzle embedding).
        puzzle_emb_len (int): Length of the puzzle embedding prefix. Default is 16.
        h_layers (int): Number of layers in the H_level reasoning module. Default is 2.
        l_layers (int): Number of layers in the L_level reasoning module. Default is 2.
        halt_max_steps (int): Maximum number of ACT steps allowed. Default is 10.
        halt_exploration_prob (float): Probability of exploration during halting
            decisions. Default is 0.1.
        no_act_continue (bool): Whether to use simple halting (True) or Q-learning
            based halting (False). Default is True.
        rope_theta (float): Theta value for RoPE (Rotary Position Embedding).
            Default is 10000.0.
        attention_type (str): Type of attention mechanism to use. Default is 'multi_head'.
        ffn_type (str): Type of feed-forward network to use. Default is 'swiglu'.
        normalization_type (str): Type of normalization layer to use. Default is 'rms_norm'.
        normalization_position (str): Position of normalization ('pre' or 'post').
            Default is 'post'.
        dropout_rate (float): Dropout rate for transformer layers. Default is 0.0.
        attention_dropout_rate (float): Dropout rate specifically for attention.
            Default is 0.0.
        **kwargs: Additional arguments for the `keras.Model` base class.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        seq_len: int,
        puzzle_emb_len: int = 16,
        h_layers: int = 2,
        l_layers: int = 2,
        halt_max_steps: int = 10,
        halt_exploration_prob: float = 0.1,
        no_act_continue: bool = True,
        rope_theta: float = 10000.0,
        attention_type: AttentionType = 'multi_head',
        ffn_type: FFNType = 'swiglu',
        normalization_type: NormalizationType = 'rms_norm',
        normalization_position: NormalizationPositionType = 'post',
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store all configuration parameters as instance attributes
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expansion = expansion
        self.seq_len = seq_len
        self.puzzle_emb_len = puzzle_emb_len
        self.h_layers = h_layers
        self.l_layers = l_layers
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob
        self.no_act_continue = no_act_continue
        self.rope_theta = rope_theta
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        # CREATE the main sub-layer in __init__ following the Golden Rule.
        # We will explicitly build it in this model's `build` method to ensure
        # its weights are available before the first `call`.
        self.inner = TRMInner(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            expansion=expansion,
            seq_len=seq_len,
            puzzle_emb_len=puzzle_emb_len,
            h_layers=h_layers,
            l_layers=l_layers,
            rope_theta=rope_theta,
            attention_type=attention_type,
            ffn_type=ffn_type,
            normalization_type=normalization_type,
            normalization_position=normalization_position,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            name="trm_inner"
        )

    def build(self, input_shape: Optional[Any] = None) -> None:
        """
        Build the model and its inner layer.

        This explicit build call is crucial. It ensures that `self.inner.H_init`
        and `self.inner.L_init` are created before the `call` method tries to
        access them for the state reset logic. Without this, an error occurs
        because the weights don't exist yet on the first call.

        Args:
            input_shape (Optional[Any]): Shape of the input. Not used since the
                inner layer handles its own shape inference.
        """
        if not self.inner.built:
            self.inner.build()
        super().build(input_shape)

    def initial_carry(self, batch: Dict[str, keras.KerasTensor]) -> Dict[str, Any]:
        """
        Create the initial state for the ACT loop.

        This method initializes all state variables needed for the recursive
        reasoning process, including latent states, step counters, halting flags,
        and current data.

        Args:
            batch (Dict[str, keras.KerasTensor]): A batch of input data containing:
                - `inputs` (keras.KerasTensor): Input token IDs with shape
                    (batch_size, seq_len).

        Returns:
            Dict[str, Any]: The initial `carry` dictionary containing:
                - `inner_carry`: Initial latent states (all zeros).
                - `steps`: Step counter initialized to 0.
                - `halted`: Boolean mask initialized to True (triggers reset on first step).
                - `current_data`: Data tensor initialized to zeros.
        """
        batch_size = ops.shape(batch["inputs"])[0]
        full_shape = (
            batch_size,
            self.seq_len + self.puzzle_emb_len,
            self.hidden_size
        )

        return {
            "inner_carry": {
                "z_H": ops.zeros(full_shape, dtype=self.compute_dtype),
                "z_L": ops.zeros(full_shape, dtype=self.compute_dtype),
            },
            "steps": ops.zeros((batch_size,), dtype="int32"),
            # Start with `halted` as True to trigger a reset on the first step.
            "halted": ops.ones((batch_size,), dtype="bool"),
            "current_data": {k: ops.zeros_like(v) for k, v in batch.items()},
        }

    def call(
            self,
            carry: Dict[str, Any],
            batch: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Tuple[Dict[str, Any], Dict[str, keras.KerasTensor]]:
        """
        Perform one step of the ACT reasoning process.

        This method implements a single iteration of the adaptive computation loop.
        It handles state resetting for newly started sequences, delegates computation
        to the inner layer, and manages the halting logic.

        Args:
            carry (Dict[str, Any]): The state from the previous step containing:
                - `inner_carry`: Latent states from previous step.
                - `steps`: Current step count.
                - `halted`: Boolean mask of halted sequences.
                - `current_data`: Current input data.
            batch (Dict[str, keras.KerasTensor]): The current batch of data containing:
                - `inputs`: Input token IDs.
            training (Optional[bool]): Boolean flag for training mode. Affects halting
                behavior (training uses learned halting, inference uses max steps).

        Returns:
            Tuple containing:
            - new_carry (Dict[str, Any]): The updated state for the next step.
            - outputs (Dict[str, keras.KerasTensor]): The model outputs for this step:
                - `logits`: Prediction logits.
                - `q_halt_logits`: Halting probability logits.
                - `q_continue_logits`: Continuation probability logits.
                - `target_q_continue` (optional): Target Q-value for Bellman update
                    (only present during training with Q-learning).
        """
        inner_carry = carry["inner_carry"]
        halted = carry["halted"]

        # Reset inner state (z_H, z_L) for newly started sequences using
        # the initial state weights from the (now built) `inner` layer.
        # Broadcasting is handled by ops.where with appropriate expansion.
        reset_flag = ops.expand_dims(halted, axis=(-1, -2))
        z_H = ops.where(reset_flag, self.inner.H_init, inner_carry["z_H"])
        z_L = ops.where(reset_flag, self.inner.L_init, inner_carry["z_L"])

        # Reset step counter for newly started sequences
        steps = ops.where(halted, 0, carry["steps"])

        # Update the data for sequences that have not yet halted.
        # For halted sequences, use new batch data; for non-halted, keep current.
        current_data = {}
        for k, v in batch.items():
            # Expand halted mask to match data dimensions
            expand_dims = (1,) * (len(v.shape) - 1)
            halted_expanded = ops.reshape(halted, (-1, *expand_dims))
            current_data[k] = ops.where(halted_expanded, v, carry["current_data"][k])

        # Perform inner reasoning step
        new_inner_carry, logits, (q_halt, q_continue) = self.inner(
            {"z_H": z_H, "z_L": z_L}, current_data, training=training
        )

        # Prepare outputs
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt,
            "q_continue_logits": q_continue
        }

        # --- Halting Logic (No Gradients) ---
        # Increment step counter
        steps = steps + 1

        # Check if maximum steps reached
        is_last_step = steps >= self.halt_max_steps
        new_halted = is_last_step

        if training and self.halt_max_steps > 1:
            # Training mode: use learned halting signals
            if self.no_act_continue:
                # Simple halting: halt if q_halt > 0
                halt_signal = q_halt > 0
            else:
                # Q-learning halting: halt if q_halt > q_continue
                halt_signal = q_halt > q_continue
            new_halted = new_halted | halt_signal

            # Exploration: randomly force continuation for some sequences
            rand_val = keras.random.uniform(ops.shape(q_halt))
            explore_halt = rand_val < self.halt_exploration_prob
            min_halt_steps = ops.cast(explore_halt, "int32") * keras.random.randint(
                ops.shape(steps), 2, self.halt_max_steps + 1
            )
            new_halted = new_halted & (steps >= min_halt_steps)

            if not self.no_act_continue:
                # Q-learning: compute target Q-value for Bellman update
                # Lookahead one step to get target Q-value
                _, _, (next_q_halt, next_q_continue) = self.inner(
                    new_inner_carry, current_data, training=training
                )
                # Target is the maximum Q-value at the next state
                target_q = ops.where(
                    is_last_step,
                    next_q_halt,
                    ops.maximum(next_q_halt, next_q_continue)
                )
                outputs["target_q_continue"] = ops.sigmoid(target_q)

        if not training:
            # Inference mode: only halt at the maximum step limit
            new_halted = is_last_step

        # Construct new carry state
        new_carry = {
            "inner_carry": new_inner_carry,
            "steps": steps,
            "halted": new_halted,
            "current_data": current_data,
        }

        return new_carry, outputs

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns:
            Dict[str, Any]: Configuration dictionary containing all parameters needed
                to reconstruct this model.
        """
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'expansion': self.expansion,
            'seq_len': self.seq_len,
            'puzzle_emb_len': self.puzzle_emb_len,
            'h_layers': self.h_layers,
            'l_layers': self.l_layers,
            'halt_max_steps': self.halt_max_steps,
            'halt_exploration_prob': self.halt_exploration_prob,
            'no_act_continue': self.no_act_continue,
            'rope_theta': self.rope_theta,
            'attention_type': self.attention_type,
            'ffn_type': self.ffn_type,
            'normalization_type': self.normalization_type,
            'normalization_position': self.normalization_position,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
        })
        return config


# ---------------------------------------------------------------------