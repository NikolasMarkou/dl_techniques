"""Tiny Recursive Model: one small network applied repeatedly under Adaptive
Computation Time, with optional Q-learned halting.

Defines :class:`TRM` and :func:`create_trm`. A feedforward network spends the same
compute on every input, so its budget has to fit the hardest case. TRM applies one
small network recursively instead, so depth is an iteration count rather than a
parameter count and each example in the batch picks its own. Inside a step,
``TRMInner`` updates the low-level state ``z_L`` from the previous ``z_L`` and the
token embeddings, then the high-level state ``z_H`` from the previous ``z_H`` and the
fresh ``z_L``, and a ``q_head`` reads two halting logits off ``z_H``'s first position.
Callers own the outer loop: ``call`` performs one step, taking a ``carry`` dict and
returning the updated one. The carry's latent states pass through ``stop_gradient``,
so gradients flow within a step but not across steps, which keeps memory flat in the
step count at the cost of a one-step-truncated approximation. A halted example
restarts from the learnable ``H_init``/``L_init`` and refills its ``current_data``
slot from the incoming batch, so batch slots are reused as examples finish at
different times. ``build`` has to run before the first ``call``, which
:func:`create_trm` does for you.

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
    """Run one Adaptive Computation Time step of the recursive reasoning module.

    ``call`` takes the state ``carry`` forward one step through ``TRMInner`` and
    returns the updated carry along with this step's outputs. A training script owns
    the outer loop, calling this repeatedly until every item in the batch has halted.

    One step:

    .. code-block:: text

        carry, batch
             │
             ▼
        ┌──────────────────────────────┐
        │ reset where halted           │
        │  z_H, z_L <- H_init, L_init  │
        │  steps <- 0                  │
        │  current_data <- batch       │
        └──────────────────────────────┘
             │
             ▼
        ┌──────────────────────────────┐
        │ inner  TRMInner              │
        │  z_L <- f(z_L, tokens)       │
        │  z_H <- g(z_H, z_L)          │
        └──────────────────────────────┘
             │
             ├──► logits
             ├──► q_halt_logits
             ├──► q_continue_logits
             │
             ▼
        ┌──────────────────────────────┐
        │ halting                      │
        │  steps + 1 vs halt_max_steps │
        │  learned signal, exploration │
        └──────────────────────────────┘
             │
             ├──► lookahead inner ──► target_q_continue
             │    (training and q-learning only)
             ▼
        new carry

    The lookahead is a side branch: its own carry is discarded.

    Halting rule:

    .. code-block:: text

              q_halt, q_continue, steps
                        │
                ┌───────┴───────┐
                ▼               ▼
           no_act_continue   q-learning
           q_halt > 0        q_halt > q_continue
                │               │
                └───────┬───────┘
                        ▼
             halt if signal or steps >= halt_max_steps
                        │
                        ▼
             training also needs steps >= min_halt_steps

    min_halt_steps is 0 unless exploration drew that example.

    Carry:

    .. code-block:: text

        carry
         ├─ inner_carry
         │    z_H, z_L    [B, puzzle_emb_len + seq_len, hidden_size]
         ├─ steps         [B] int32
         ├─ halted        [B] bool
         └─ current_data  one entry per batch key

    :param vocab_size: Size of the vocabulary for token embeddings.
    :param hidden_size: Dimensionality of hidden states. Must be divisible by `num_heads`.
    :param num_heads: Number of attention heads in transformer layers.
    :param expansion: FFN intermediate-size multiplier.
    :param seq_len: Length of the input sequence, excluding the puzzle embedding.
    :param puzzle_emb_len: Length of the puzzle embedding prefix.
    :param h_layers: Number of layers in the H-level reasoning module.
    :param l_layers: Number of layers in the L-level reasoning module.
    :param halt_max_steps: Maximum number of ACT steps allowed. Must be >= 1. With 1,
        every example halts after one step and no learned signal is read.
    :param halt_exploration_prob: Probability of forcing extra exploration steps during training.
    :param no_act_continue: If True, halt on `q_halt > 0`; if False, use Q-learning halting (`q_halt > q_continue`) and emit `target_q_continue` during training.
    :param rope_theta: RoPE base frequency.
    :param attention_type: Attention mechanism. Default `'group_query'` with
        `num_kv_heads == num_heads`, plain multi-head attention that carries RoPE.
    :param ffn_type: Feed-forward network type. Default `'swiglu'`.
    :param normalization_type: Normalization layer type. Default `'rms_norm'`.
    :param normalization_position: `'pre'` or `'post'`. Default `'post'`.
    :param dropout_rate: Dropout rate for transformer layers.
    :param attention_dropout_rate: Dropout rate for attention.
    :param **kwargs: Forwarded to `keras.Model`.

    :raises ValueError: If `hidden_size` is not divisible by `num_heads`, if
        `halt_max_steps` is below 1, or if `halt_exploration_prob` is outside [0, 1].
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
        # 'multi_head'; see TRMReasoningModule.__init__ in components.py. See decisions.md.
        attention_type: AttentionType = 'group_query',
        ffn_type: FFNType = 'swiglu',
        normalization_type: NormalizationType = 'rms_norm',
        normalization_position: NormalizationPositionType = 'post',
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

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
        """Build the inner layer, and with it the initial-state weights.

        ``call`` reads ``self.inner.H_init`` and ``self.inner.L_init`` for the reset,
        so those variables have to exist before the first call.

        :param input_shape: Shape of the input. Not used since the inner layer handles its own shape inference.
        """
        if not self.inner.built:
            self.inner.build()
        super().build(input_shape)

    def initial_carry(self, batch: Dict[str, keras.KerasTensor]) -> Dict[str, Any]:
        """Create the starting state for the ACT loop.

        Latent states, step counters and data slots all start at zero, and ``halted``
        starts True so the first ``call`` resets every item to the learned init states
        and pulls in the real batch.

        :param batch: A batch of input data. ``batch["inputs"]`` gives the batch size,
            and every key gets a zeroed slot in ``current_data``.

        :return: The initial ``carry``: ``inner_carry`` (zeroed ``z_H``/``z_L``),
            ``steps`` (0), ``halted`` (True) and ``current_data`` (zeros).
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
            "halted": keras.ops.ones((batch_size,), dtype="bool"),
            "current_data": {k: keras.ops.zeros_like(v) for k, v in batch.items()},
        }

    def call(
            self,
            carry: Dict[str, Any],
            batch: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Tuple[Dict[str, Any], Dict[str, keras.KerasTensor]]:
        """Perform one step of the ACT reasoning process.

        Resets the items that halted, runs the inner module once, then decides which
        items halt on this step.

        :param carry: The state from the previous step: ``inner_carry``, ``steps``,
            ``halted`` and ``current_data``.
        :param batch: The current batch, whose entries refill the ``current_data``
            slots of items that had halted.
        :param training: Training-mode flag. Both modes halt on the learned signal;
            training additionally forces extra steps for a random subset of items and,
            under Q-learning, computes the Bellman target.

        :return: ``(new_carry, outputs)``. ``outputs`` holds ``logits``,
            ``q_halt_logits`` and ``q_continue_logits``, plus a sigmoid-squashed
            ``target_q_continue`` when training with Q-learning halting.
        """
        inner_carry = carry["inner_carry"]
        halted = carry["halted"]

        # Items that halted restart from the learnable init states.
        reset_flag = keras.ops.expand_dims(halted, axis=(-1, -2))
        z_H = keras.ops.where(reset_flag, self.inner.H_init, inner_carry["z_H"])
        z_L = keras.ops.where(reset_flag, self.inner.L_init, inner_carry["z_L"])

        steps = keras.ops.where(halted, 0, carry["steps"])

        # A halted slot takes the incoming batch; a running slot keeps its own data.
        current_data = {}
        for k, v in batch.items():
            expand_dims = (1,) * (len(v.shape) - 1)
            halted_expanded = keras.ops.reshape(halted, (-1, *expand_dims))
            current_data[k] = keras.ops.where(halted_expanded, v, carry["current_data"][k])

        new_inner_carry, logits, (q_halt, q_continue) = self.inner(
            {"z_H": z_H, "z_L": z_L}, current_data, training=training
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt,
            "q_continue_logits": q_continue
        }

        steps = steps + 1

        is_last_step = steps >= self.halt_max_steps
        new_halted = is_last_step

        if training and self.halt_max_steps > 1:
            if self.no_act_continue:
                halt_signal = q_halt > 0
            else:
                halt_signal = q_halt > q_continue
            new_halted = new_halted | halt_signal

            # Forcing extra steps lets the halt head see states past an immediate halt.
            rand_val = keras.random.uniform(keras.ops.shape(q_halt))
            explore_halt = rand_val < self.halt_exploration_prob
            min_halt_steps = keras.ops.cast(explore_halt, "int32") * keras.random.randint(
                keras.ops.shape(steps), 2, self.halt_max_steps + 1
            )
            new_halted = new_halted & (steps >= min_halt_steps)

            if not self.no_act_continue:
                # The lookahead runs with training=False so dropout cannot corrupt the
                # bootstrap, and stop_gradient keeps it a target rather than a path.
                _, _, (next_q_halt, next_q_continue) = self.inner(
                    new_inner_carry, current_data, training=False
                )
                target_q = keras.ops.where(
                    is_last_step,
                    next_q_halt,
                    keras.ops.maximum(next_q_halt, next_q_continue)
                )
                target_q = keras.ops.stop_gradient(target_q)
                outputs["target_q_continue"] = keras.ops.sigmoid(target_q)

        if not training:
            # DECISION plan_2026-05-10_e6309bd5/D-001: inference halts on the learned
            # signal too, not on halt_max_steps alone. See decisions.md.
            if self.halt_max_steps > 1:
                if self.no_act_continue:
                    halt_signal = q_halt > 0
                else:
                    halt_signal = q_halt > q_continue
                new_halted = is_last_step | halt_signal
            else:
                new_halted = is_last_step

        new_carry = {
            "inner_carry": new_inner_carry,
            "steps": steps,
            "halted": new_halted,
            "current_data": current_data,
        }

        return new_carry, outputs

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Configuration dictionary containing every constructor argument.
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
    # DECISION plan-2026-08-17T183311-79c63e38/D-007: 'group_query', not 'multi_head';
    # a 'multi_head' default here would drop RoPE for every model built. See decisions.md.
    attention_type: AttentionType = 'group_query',
    ffn_type: FFNType = 'swiglu',
    normalization_type: NormalizationType = 'rms_norm',
    normalization_position: NormalizationPositionType = 'post',
    dropout_rate: float = 0.0,
    attention_dropout_rate: float = 0.0,
    name: Optional[str] = None,
) -> TRM:
    """Build a TRM and build its inner layer.

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
    :raises ValueError: For the same argument checks :class:`TRM` makes.
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
