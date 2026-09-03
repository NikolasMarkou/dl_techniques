"""
Tiny Recursive Model: a small shared reasoning network applied repeatedly
under Adaptive Computation Time, with optional Q-learned halting.

A feedforward network spends the same compute on every input, so its
budget must fit the hardest case and is wasted on the rest. TRM applies
one small network recursively instead: depth becomes iteration count, not
parameter count, and each example in the batch picks its own iteration
count. The outer ACT loop lives in the training script, not this model:
`call` performs one outer step, taking a `carry` dict in and returning the
updated one. Inside a step, `TRMInner` updates the low-level state `z_L`
from the previous `z_L` and the token embeddings, then the high-level
state `z_H` from the previous `z_H` and the fresh `z_L`.

Halting is learned: a `q_head` reads two logits off `z_H`'s first
position. Under Q-learning an example halts when `q_halt > q_continue`;
with `no_act_continue` the rule is `q_halt > 0`. `halt_max_steps` is a
hard ceiling either way. Training fits the Q-values as a Bellman target,
looking one step ahead under `training=False` and detaching it with
`stop_gradient`, and also forces a random subset of examples to keep
going for extra steps so the halting head sees states beyond an immediate
halt. Inference uses the learned halt signal with no exploration.

The carry's latent states pass forward through `stop_gradient`, so
gradients flow within one outer step but not across steps: memory stays
constant in the number of ACT steps, at the cost of a one-step-truncated
approximation. A halted example's states reset to the learnable `H_init`
/ `L_init` on the next call, and its `current_data` slot refills from the
incoming batch, so one batch slot is reused as examples finish at
different times.

References:
    - Jolicoeur-Martineau, 2025. Less is More: Recursive Reasoning with Tiny
      Networks. (https://arxiv.org/abs/2510.04871)
    - Graves, 2016. Adaptive Computation Time for Recurrent Neural Networks.
      (https://arxiv.org/abs/1603.08983)
    - Wang et al., 2025. Hierarchical Reasoning Model.
      (https://arxiv.org/abs/2506.21734)
    - Banino et al., 2021. PonderNet: Learning to Ponder.
      (https://arxiv.org/abs/2107.05407)
    - Dehghani et al., 2018. Universal Transformers.
      (https://arxiv.org/abs/1807.03819)
"""

import keras
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
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.tiny_recursive_model.model")
class TRM(keras.Model):
    """
    Tiny Recursive Model (TRM) with Adaptive Computation Time (ACT).

    `call` runs a single step of the ACT loop: it takes the state `carry`
    forward one step through the `TRMInner` reasoning module and returns
    the updated carry. A training script owns the outer loop, calling this
    repeatedly until every sequence in the batch has halted.

    `carry` is a dict holding: `inner_carry` (the `z_H` / `z_L` latent
    states for `TRMInner`), `steps` (the per-item step count), `halted`
    (a boolean mask), and `current_data` (the input for non-halted items).

    :param vocab_size: Size of the vocabulary for token embeddings.
    :param hidden_size: Dimensionality of hidden states.
    :param num_heads: Number of attention heads in transformer layers.
    :param expansion: FFN intermediate-size multiplier.
    :param seq_len: Length of the input sequence, excluding the puzzle embedding.
    :param puzzle_emb_len: Length of the puzzle embedding prefix.
    :param h_layers: Number of layers in the H-level reasoning module.
    :param l_layers: Number of layers in the L-level reasoning module.
    :param halt_max_steps: Maximum number of ACT steps allowed.
    :param halt_exploration_prob: Probability of forcing extra exploration steps during training.
    :param no_act_continue: If True, halt on `q_halt > 0`; if False, use Q-learning halting (`q_halt > q_continue`).
    :param rope_theta: RoPE base frequency.
    :param attention_type: Attention mechanism. Default `'group_query'` with
        `num_kv_heads == num_heads`, plain multi-head attention that carries RoPE.
    :param ffn_type: Feed-forward network type. Default `'swiglu'`.
    :param normalization_type: Normalization layer type. Default `'rms_norm'`.
    :param normalization_position: `'pre'` or `'post'`. Default `'post'`.
    :param dropout_rate: Dropout rate for transformer layers.
    :param attention_dropout_rate: Dropout rate for attention.
    :param kwargs: Forwarded to `keras.Model`.
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
        # DECISION plan-2026-08-17T183311-79c63e38/D-007: 'group_query', not
        # 'multi_head' — see TRMReasoningModule.__init__ in components.py. See decisions.md.
        attention_type: AttentionType = 'group_query',
        ffn_type: FFNType = 'swiglu',
        normalization_type: NormalizationType = 'rms_norm',
        normalization_position: NormalizationPositionType = 'post',
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # --- Input validation ---
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        if halt_max_steps < 1:
            raise ValueError(
                f"halt_max_steps must be >= 1, got {halt_max_steps}."
            )
        if not (0.0 <= halt_exploration_prob <= 1.0):
            raise ValueError(
                f"halt_exploration_prob must be in [0, 1], got "
                f"{halt_exploration_prob}."
            )

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

        :param input_shape: Shape of the input. Not used since the inner layer handles its own shape inference.
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

        :param batch: A batch of input data containing: - `inputs` (keras.KerasTensor): Input token IDs with shape (batch_size, seq_len).

        :return: Dict[str, Any]: The initial `carry` dictionary containing: - `inner_carry`: Initial latent states (all zeros). - `steps`: Step counter initialized to 0. - `halted`: Boolean mask initialized to True (triggers reset on first step). - `current_data`: Data tensor initialized to zeros.
        """
        batch_size = keras.ops.shape(batch["inputs"])[0]
        full_shape = (
            batch_size,
            self.seq_len + self.puzzle_emb_len,
            self.hidden_size
        )

        return {
            "inner_carry": {
                "z_H": keras.ops.zeros(full_shape, dtype=self.compute_dtype),
                "z_L": keras.ops.zeros(full_shape, dtype=self.compute_dtype),
            },
            "steps": keras.ops.zeros((batch_size,), dtype="int32"),
            # Start with `halted` as True to trigger a reset on the first step.
            "halted": keras.ops.ones((batch_size,), dtype="bool"),
            "current_data": {k: keras.ops.zeros_like(v) for k, v in batch.items()},
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

        :param carry: The state from the previous step containing: - `inner_carry`: Latent states from previous step. - `steps`: Current step count. - `halted`: Boolean mask of halted sequences. - `current_data`: Current input data.
        :param batch: The current batch of data containing: - `inputs`: Input token IDs.
        :param training: Boolean flag for training mode. Affects halting behavior (training uses learned halting, inference uses max steps).

        :return: Tuple containing: - new_carry (Dict[str, Any]): The updated state for the next step. - outputs (Dict[str, keras.KerasTensor]): The model outputs for this step: - `logits`: Prediction logits. - `q_halt_logits`: Halting probability logits. - `q_continue_logits`: Continuation probability logits. - `target_q_continue` (optional): Target Q-value for Bellman update (only present during training with Q-learning).
        """
        inner_carry = carry["inner_carry"]
        halted = carry["halted"]

        # Reset inner state (z_H, z_L) for newly started sequences using
        # the initial state weights from the (now built) `inner` layer.
        # Broadcasting is handled by ops.where with appropriate expansion.
        reset_flag = keras.ops.expand_dims(halted, axis=(-1, -2))
        z_H = keras.ops.where(reset_flag, self.inner.H_init, inner_carry["z_H"])
        z_L = keras.ops.where(reset_flag, self.inner.L_init, inner_carry["z_L"])

        # Reset step counter for newly started sequences
        steps = keras.ops.where(halted, 0, carry["steps"])

        # Update the data for sequences that have not yet halted.
        # For halted sequences, use new batch data; for non-halted, keep current.
        current_data = {}
        for k, v in batch.items():
            # Expand halted mask to match data dimensions
            expand_dims = (1,) * (len(v.shape) - 1)
            halted_expanded = keras.ops.reshape(halted, (-1, *expand_dims))
            current_data[k] = keras.ops.where(halted_expanded, v, carry["current_data"][k])

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
            rand_val = keras.random.uniform(keras.ops.shape(q_halt))
            explore_halt = rand_val < self.halt_exploration_prob
            min_halt_steps = keras.ops.cast(explore_halt, "int32") * keras.random.randint(
                keras.ops.shape(steps), 2, self.halt_max_steps + 1
            )
            new_halted = new_halted & (steps >= min_halt_steps)

            if not self.no_act_continue:
                # Q-learning: compute target Q-value for Bellman update.
                # Lookahead one step in eval-mode (deterministic, no dropout)
                # so the bootstrap target is not corrupted by training-time
                # stochasticity. The target is also detached from the graph
                # via stop_gradient — HRMLoss consumes it as a Bellman TD
                # target (B-3 fix).
                _, _, (next_q_halt, next_q_continue) = self.inner(
                    new_inner_carry, current_data, training=False
                )
                # Target is the maximum Q-value at the next state
                target_q = keras.ops.where(
                    is_last_step,
                    next_q_halt,
                    keras.ops.maximum(next_q_halt, next_q_continue)
                )
                target_q = keras.ops.stop_gradient(target_q)
                outputs["target_q_continue"] = keras.ops.sigmoid(target_q)

        if not training:
            # DECISION plan_2026-05-10_e6309bd5/D-001: inference must halt on the
            # learned signal too, mirroring training minus exploration, not only on halt_max_steps. See decisions.md.
            if self.halt_max_steps > 1:
                if self.no_act_continue:
                    halt_signal = q_halt > 0
                else:
                    halt_signal = q_halt > q_continue
                new_halted = is_last_step | halt_signal
            else:
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

        :return: Dict[str, Any]: Configuration dictionary containing all parameters needed to reconstruct this model.
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


def create_trm(
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
    # DECISION plan-2026-08-17T183311-79c63e38/D-007: 'group_query', not 'multi_head'
    # — a 'multi_head' default here would re-impose the defect on every model this factory builds. See decisions.md.
    attention_type: AttentionType = 'group_query',
    ffn_type: FFNType = 'swiglu',
    normalization_type: NormalizationType = 'rms_norm',
    normalization_position: NormalizationPositionType = 'post',
    dropout_rate: float = 0.0,
    attention_dropout_rate: float = 0.0,
    name: Optional[str] = None,
) -> TRM:
    """Factory for constructing a built TRM model.

    Returns a TRM instance with its inner layer built so that ``H_init`` /
    ``L_init`` weights exist before the first ``call``. This mirrors the
    factory convention used elsewhere in ``dl_techniques.models``.

    :param vocab_size: Size of the vocabulary for token embeddings.
    :param hidden_size: Dimensionality of hidden states. Must be divisible by ``num_heads``.
    :param num_heads: Number of attention heads in transformer layers.
    :param expansion: Factor to determine FFN intermediate size.
    :param seq_len: Length of the input sequence (excluding puzzle embedding).
    :param puzzle_emb_len: Length of the puzzle embedding prefix. Default 16.
    :param h_layers: Number of layers in the H-level module. Default 2.
    :param l_layers: Number of layers in the L-level module. Default 2.
    :param halt_max_steps: Maximum ACT steps. Must be >= 1. Default 10.
    :param halt_exploration_prob: Probability of exploration during halting. Must be in [0, 1]. Default 0.1.
    :param no_act_continue: Use simple halting (True) vs Q-learning (False). Default True.
    :param rope_theta: Theta for Rotary Position Embedding. Default 10000.0.
    :param attention_type: Type of attention mechanism. Default 'group_query'; 'multi_head' carries no RoPE.
    :param ffn_type: Type of feed-forward network.
    :param normalization_type: Type of normalization layer.
    :param normalization_position: ``pre`` or ``post`` normalization.
    :param dropout_rate: Dropout rate for transformer layers.
    :param attention_dropout_rate: Dropout rate for attention.
    :param name: Optional Keras model name.

    :return: A built ``TRM`` instance.
    """
    model = TRM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        expansion=expansion,
        seq_len=seq_len,
        puzzle_emb_len=puzzle_emb_len,
        h_layers=h_layers,
        l_layers=l_layers,
        halt_max_steps=halt_max_steps,
        halt_exploration_prob=halt_exploration_prob,
        no_act_continue=no_act_continue,
        rope_theta=rope_theta,
        attention_type=attention_type,
        ffn_type=ffn_type,
        normalization_type=normalization_type,
        normalization_position=normalization_position,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        name=name,
    )
    model.build()
    return model

# ---------------------------------------------------------------------