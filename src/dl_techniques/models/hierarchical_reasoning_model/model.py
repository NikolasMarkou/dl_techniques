"""
Hierarchical Reasoning Model: a two-timescale recurrent reasoning core under Adaptive
Computation Time with Q-learned halting.

A fixed-depth transformer spends identical compute on every input. For multi-step
reasoning that is doubly wrong: easy instances are over-served, and hard ones are
capped by a depth chosen at training time. HRM makes depth a runtime quantity in two
independent ways. Within one step, a small core iterates its own latent states rather
than passing through distinct layers, so effective depth is `h_cycles * l_cycles`
applications of a fixed parameter set. Across steps, Adaptive Computation Time lets
each sequence in the batch decide for itself how many steps to take.

The hierarchy is a separation of timescales. Two latent states are maintained: a
low-level `z_l` that is refreshed against the input on every inner iteration, and a
high-level `z_h` that is refreshed only once per outer `h_cycle`, from the low-level
state that the inner loop just settled. `z_l` sees `z_h + input_emb`, so the
high-level state acts as a slowly-varying context that conditions fast local
computation, while the fast state supplies the evidence the slow state integrates.
This is the structure that lets a small network behave like a deep one without a deep
one's parameter count.

Gradients are truncated deliberately, and where they are truncated is the
non-obvious part. Every cycle *except the last* runs under `stop_gradient`; the
states are then detached and one final `l_reasoning` / `h_reasoning` pair runs with
gradients recorded. Backpropagation therefore covers a single update applied to a
converged state, not the whole `h_cycles * l_cycles` unroll. Memory is constant in
the cycle counts, which is what makes large cycle settings affordable, and the
assumption being made is that the earlier cycles are approximating a fixed point
whose gradient the last step adequately represents. One consequence is worth naming
because it looks like an omission: the core's `h_init` / `l_init` initial states are
NON-TRAINABLE. They are read only when a halted sequence's carry is reset, so they
enter as cycle 0's state and every path from them to the loss crosses that same
`stop_gradient`. They were declared `trainable=True` for a long time and never once
received a gradient; they are buffers in the reference implementation too.

Halting is learned as a two-action Q-value problem. A `q_head` reads `z_h`'s first
position and emits `q_halt` and `q_continue`; a sequence halts when
`q_halt > q_continue`, with `halt_max_steps` as a hard ceiling. The Q-values are
trained against a Bellman target formed by looking one step ahead and taking
`sigmoid(max(q_halt', q_continue'))`, or `sigmoid(q_halt')` when that step would be
the last. Note there is no discount factor: the target is the raw bootstrapped value,
not `gamma * max(...)`. The lookahead runs with `training=False` and the target is
`stop_gradient`-ed, so it is neither stochastic nor differentiable — without both, the
TD loss can be minimized by dragging the target toward the prediction, the standard
target-network collapse.

Training adds an exploration branch that forces a random subset of sequences to run
for at least a randomly drawn minimum number of steps, so the model observes states
that a greedy halting head would never reach. Inference uses the same learned halt
signal but no exploration. An earlier version halted at inference on `halt_max_steps`
alone, which made the trained `q_head` inert at exactly the time adaptivity is
claimed; that is fixed, and the two code paths are now deliberately symmetric apart
from exploration.

Two implementation details exist because this must run under `tf.function`, which is
the regime `fit()` uses. The complete-forward loop has a STATIC trip count of
`halt_max_steps` and no Python `break`: `all_finished` is a symbolic scalar, so an
`if` on it raises `OperatorNotAllowedInGraphError` and only ever worked eagerly.
Instead, once a sequence is done its carry and outputs are frozen by `ops.where`, which
reproduces the break exactly — continuing to step would reset the halted sequence
(a halted sequence's states are reset on the next step) and silently restart its
reasoning, returning a different answer. That freeze is keyed on the *per-sequence*
`carry["halted"]` mask, not on the batch-global `all_finished` scalar: gating on
`ops.all(halted)` meant an early-halting sequence kept being restarted until the
batch's slowest member finished, and the caller received its restarted partial run
instead of the answer it halted on. For the same reason `is_last_step` is used
inside an `ops.where` rather than a Python `if`: it is a per-sequence `(batch,)`
tensor and a truth-test on it is ambiguous eagerly and illegal in graph mode.

The carry is a dict of `inner_carry` (the two latent states), a per-sequence `steps`
counter, a `halted` mask, and `current_data`. That last field is what lets a batch
slot be recycled: when a sequence halts, its slot is refilled from the incoming batch
and its states are reset, so a batch is never blocked waiting for its slowest member.
The model exposes both a complete mode (`call(batch)`, run to halting) and a
single-step mode (`call((carry, batch))`) so an external training loop can drive the
recursion itself.

Positional information enters through attention, not through the input stream. With
`pos_encodings="rope"` (the default) the reasoning modules run grouped-query attention
with `num_kv_heads == num_heads` — arithmetically plain multi-head attention, chosen
because it is the only plain self-attention type in this repo's attention registry that
also applies RoPE to Q and K. `pos_encodings="learned"` is the alternative and works the
other way round: it adds a learned positional embedding to the token embedding and runs
position-blind attention. The core does NOT own a rotary-embedding layer of its own; it
once did, was never handed a Q or K tensor by any code path, and the model was measured
exactly permutation-equivariant while advertising RoPE.

Six preset variants scale layers, heads and cycle counts together, from 2+2 layers
with 2+2 cycles to 16+16 layers with 4+4 cycles.

References:
    - Wang et al., 2025. Hierarchical Reasoning Model.
      (https://arxiv.org/abs/2506.21734)
    - Graves, 2016. Adaptive Computation Time for Recurrent Neural Networks.
      (https://arxiv.org/abs/1603.08983)
    - Dehghani et al., 2018. Universal Transformers.
      (https://arxiv.org/abs/1807.03819)
    - Banino et al., 2021. PonderNet: Learning to Ponder.
      (https://arxiv.org/abs/2107.05407)
    - Bai et al., 2019. Deep Equilibrium Models.
      (https://arxiv.org/abs/1909.01377)
    - Mnih et al., 2015. Human-level control through deep reinforcement learning.
      Nature 518, 529-533.
"""

import keras
from typing import Optional, Union, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.losses.hrm_loss import StableMaxCrossEntropy
from dl_techniques.layers.reasoning.hrm_reasoning_core import HierarchicalReasoningCore

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class HierarchicalReasoningModel(keras.Model):
    """Hierarchical Reasoning Model with Adaptive Computation Time.

    This model wraps a hierarchical reasoning core with ACT mechanisms for dynamic
    computational depth allocation. It learns to perform variable numbers of reasoning
    steps based on input complexity, using Q-learning to determine optimal halting
    decisions while maintaining hierarchical processing capabilities.

    **Intent**: Provide a production-ready adaptive computation model that can
    efficiently handle reasoning tasks of varying complexity by learning to allocate
    computational resources dynamically, combining the benefits of hierarchical
    processing with adaptive computation time.

    **Architecture**:
    The model consists of a stateful wrapper around a HierarchicalReasoningCore,
    managing iterative computation through carry state and Q-learning based halting
    decisions. The dual call interface supports both complete reasoning and
    single-step execution for flexible training and inference scenarios.

    **Component Details**:
    - **HierarchicalReasoningCore**: Multi-level reasoning with attention mechanisms
    - **ACT Controller**: Q-learning based adaptive computation time management
    - **State Management**: Carry mechanism for iterative processing state
    - **Dual Interface**: Support for complete and step-by-step execution modes

    Args:
        vocab_size: Integer, size of the vocabulary for token embeddings.
            Must be positive.
        seq_len: Integer, maximum sequence length for input processing.
            Must be positive.
        embed_dim: Integer, embedding dimension for token representations.
            Must be positive and typically a multiple of num_heads.
        num_puzzle_identifiers: Integer, number of unique puzzle type identifiers.
            Must be positive.
        puzzle_emb_dim: Integer, embedding dimension for puzzle identifiers.
            Set to 0 to disable puzzle embeddings.
        batch_size: Integer, batch size for training and inference.
            Must be positive.
        h_layers: Integer, number of high-level reasoning layers.
            Must be positive.
        l_layers: Integer, number of low-level reasoning layers.
            Must be positive.
        h_cycles: Integer, number of high-level processing cycles per step.
            Must be positive.
        l_cycles: Integer, number of low-level processing cycles per step.
            Must be positive.
        num_heads: Integer, number of attention heads in each layer.
            Must be positive and divide evenly into embed_dim.
        ffn_expansion_factor: Integer, expansion factor for feed-forward networks.
            Typically 4 for standard transformer architectures.
        pos_encodings: String, type of positional encodings ("rope" or
            "learned"). "rope" (the default) runs the reasoning modules on
            grouped-query attention, which rotates Q and K by their own
            positions inside attention; "learned" adds a learned positional
            embedding to the input stream instead.
        rope_theta: Float, theta parameter for RoPE positional encodings.
            Only used when pos_encodings="rope".
        halt_max_steps: Integer, maximum computation steps before forced halt.
            Must be positive. Higher values allow more computation but risk instability.
        halt_exploration_prob: Float, probability of exploration in Q-learning halting.
            Must be between 0 and 1. Higher values encourage more varied computation depths.
        dropout_rate: Float, dropout rate applied throughout the model.
            Must be between 0 and 1.
        use_bias: Boolean, whether to use bias terms in linear transformations.
        embeddings_initializer: Initializer for embedding layers.
            Can be string name or Initializer instance.
        kernel_initializer: Initializer for linear layer kernels.
            Can be string name or Initializer instance.
        embeddings_regularizer: Optional regularizer for embedding layers.
        kernel_regularizer: Optional regularizer for linear layer kernels.
        name: Optional string name for the model.
        **kwargs: Additional keyword arguments for the Model base class.

    Input format:
        Dictionary with keys:
        - "token_ids": Integer tensor of shape (batch_size, seq_len)
        - "puzzle_ids": Integer tensor of shape (batch_size,)

    Output format:
        Dictionary with keys:
        - Standard reasoning outputs from the core
        - "q_halt_logits": Q-values for halting decisions
        - "q_continue_logits": Q-values for continuing computation
        - Additional ACT-related outputs during training

    Attributes:
        core: HierarchicalReasoningCore instance for reasoning computation.
        Configuration parameters as stored attributes.

    Raises:
        ValueError: If vocab_size, seq_len, embed_dim, or other size parameters are not positive.
        ValueError: If halt_exploration_prob is not in [0, 1].
        ValueError: If dropout_rate is not in [0, 1].

    Example:
        ```python
        # Standard reasoning model
        model = HierarchicalReasoningModel(
            vocab_size=32000,
            seq_len=512,
            embed_dim=768,
            num_puzzle_identifiers=1000,
            halt_max_steps=8,
            halt_exploration_prob=0.1
        )

        # High-capacity model with more exploration
        model = HierarchicalReasoningModel.from_variant(
            "large",
            vocab_size=50000,
            seq_len=1024,
            halt_exploration_prob=0.2
        )

        # Training with step-by-step control
        carry = model.initial_carry(batch)
        for step in range(max_steps):
            carry, outputs, finished = model((carry, batch))
            if finished:
                break
        ```

    Note:
        The model supports two calling modes: complete reasoning model(batch) and
        single-step reasoning model((carry, batch)). The choice depends on whether
        you need full control over the reasoning loop or prefer automatic execution.
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "micro": {
            "embed_dim": 256,
            "h_layers": 2,
            "l_layers": 2,
            "h_cycles": 2,
            "l_cycles": 2,
            "num_heads": 4,
            "halt_max_steps": 4
        },
        "tiny": {
            "embed_dim": 384,
            "h_layers": 4,
            "l_layers": 4,
            "h_cycles": 2,
            "l_cycles": 2,
            "num_heads": 6,
            "halt_max_steps": 6
        },
        "small": {
            "embed_dim": 512,
            "h_layers": 6,
            "l_layers": 6,
            "h_cycles": 2,
            "l_cycles": 3,
            "num_heads": 8,
            "halt_max_steps": 8
        },
        "base": {
            "embed_dim": 768,
            "h_layers": 8,
            "l_layers": 8,
            "h_cycles": 3,
            "l_cycles": 3,
            "num_heads": 12,
            "halt_max_steps": 10
        },
        "large": {
            "embed_dim": 1024,
            "h_layers": 12,
            "l_layers": 12,
            "h_cycles": 3,
            "l_cycles": 4,
            "num_heads": 16,
            "halt_max_steps": 12
        },
        "xlarge": {
            "embed_dim": 1536,
            "h_layers": 16,
            "l_layers": 16,
            "h_cycles": 4,
            "l_cycles": 4,
            "num_heads": 24,
            "halt_max_steps": 16
        }
    }

    def __init__(
            self,
            vocab_size: int,
            seq_len: int,
            embed_dim: int = 512,
            num_puzzle_identifiers: int = 1000,
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
            name: Optional[str] = "hierarchical_reasoning_model",
            **kwargs: Any
    ) -> None:
        """Initialize the Hierarchical Reasoning Model.

        Args:
            vocab_size: Size of vocabulary for token embeddings.
            seq_len: Maximum sequence length.
            embed_dim: Embedding dimension.
            num_puzzle_identifiers: Number of puzzle type identifiers.
            puzzle_emb_dim: Puzzle embedding dimension (0 to disable).
            batch_size: Batch size for processing.
            h_layers: Number of high-level reasoning layers.
            l_layers: Number of low-level reasoning layers.
            h_cycles: High-level processing cycles per step.
            l_cycles: Low-level processing cycles per step.
            num_heads: Number of attention heads.
            ffn_expansion_factor: Feed-forward network expansion factor.
            pos_encodings: Type of positional encodings.
            rope_theta: RoPE theta parameter.
            halt_max_steps: Maximum computation steps.
            halt_exploration_prob: Q-learning exploration probability.
            dropout_rate: Dropout rate.
            use_bias: Whether to use bias terms.
            embeddings_initializer: Embedding layer initializer.
            kernel_initializer: Linear layer kernel initializer.
            embeddings_regularizer: Embedding layer regularizer.
            kernel_regularizer: Linear layer kernel regularizer.
            name: Model name.
            **kwargs: Additional Model arguments.

        Raises:
            ValueError: If parameters are invalid.
        """
        # Validate parameters
        self._validate_parameters(
            vocab_size, seq_len, embed_dim, num_puzzle_identifiers,
            h_layers, l_layers, h_cycles, l_cycles, num_heads,
            halt_max_steps, halt_exploration_prob, dropout_rate
        )

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
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Create core reasoning model (following modern Keras 3 patterns)
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

        # Initialize the Model (Keras handles building automatically)
        super().__init__(name=name, **kwargs)

        logger.info(
            f"Initialized Hierarchical Reasoning Model with "
            f"h_layers={h_layers}, l_layers={l_layers}, "
            f"embed_dim={embed_dim}, halt_max_steps={halt_max_steps}"
        )

    def _validate_parameters(
            self,
            vocab_size: int,
            seq_len: int,
            embed_dim: int,
            num_puzzle_identifiers: int,
            h_layers: int,
            l_layers: int,
            h_cycles: int,
            l_cycles: int,
            num_heads: int,
            halt_max_steps: int,
            halt_exploration_prob: float,
            dropout_rate: float
    ) -> None:
        """Validate initialization parameters.

        Args:
            vocab_size: Vocabulary size.
            seq_len: Sequence length.
            embed_dim: Embedding dimension.
            num_puzzle_identifiers: Number of puzzle identifiers.
            h_layers: High-level layers.
            l_layers: Low-level layers.
            h_cycles: High-level cycles.
            l_cycles: Low-level cycles.
            num_heads: Number of attention heads.
            halt_max_steps: Maximum halting steps.
            halt_exploration_prob: Exploration probability.
            dropout_rate: Dropout rate.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_puzzle_identifiers <= 0:
            raise ValueError(f"num_puzzle_identifiers must be positive, got {num_puzzle_identifiers}")
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
        if halt_max_steps <= 0:
            raise ValueError(f"halt_max_steps must be positive, got {halt_max_steps}")
        if not (0.0 <= halt_exploration_prob <= 1.0):
            raise ValueError(f"halt_exploration_prob must be in [0, 1], got {halt_exploration_prob}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

    def build(self, input_shape: Any = None) -> None:
        """Explicitly build the reasoning core before any forward/reset_carry.

        `_forward_step` calls `self.core.reset_carry(...)` BEFORE the first
        `self.core(...)` call, but `reset_carry` is a plain method that does
        not trigger Keras auto-build, so `core.h_init`/`core.l_init` would
        otherwise stay None and the forward crashes. This override builds the
        core first so those add_weight states exist. Mirrors the SAM D-008
        build()-override precedent.
        """
        # DECISION plan_2026-06-16_c8f3e9ca/D-005: build reasoning core
        # (h_init/l_init via add_weight) before reset_carry; mirrors SAM D-008.
        # reset_carry is a plain method that does NOT trigger Keras auto-build.
        # Do NOT rely on auto-build via the first core(...) call: _forward_step
        # invokes core.reset_carry() first, which reads h_init/l_init. The
        # core's own build() ignores the input_shape content (it creates
        # fixed-shape weights from config), so any (None, seq_len) is fine.
        if self.built:
            return

        # call() takes a dict {token_ids, puzzle_ids}; input_shape may be a
        # dict of shapes. Extract token_ids shape if present (SAM D-008 pattern
        # for dict inputs); otherwise fall back to the config-derived shape.
        if isinstance(input_shape, dict) and "token_ids" in input_shape:
            core_input_shape = input_shape["token_ids"]
        else:
            core_input_shape = (None, self.seq_len)

        if not self.core.built:
            self.core.build(core_input_shape)

        super().build(input_shape)

    def initial_carry(self, batch: Dict[str, keras.KerasTensor]) -> Dict[str, keras.KerasTensor]:
        """Initialize carry state for a batch.

        Args:
            batch: Input batch dictionary with token_ids and puzzle_ids.

        Returns:
            Initial carry state dictionary.
        """
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

    def call(
            self,
            inputs: Union[
                Dict[str, keras.KerasTensor], Tuple[Dict[str, keras.KerasTensor], Dict[str, keras.KerasTensor]]],
            training: Optional[bool] = None
    ) -> Union[Dict[str, keras.KerasTensor], Tuple[
        Dict[str, keras.KerasTensor], Dict[str, keras.KerasTensor], keras.KerasTensor]]:
        """Forward pass through the model.

        This method supports two calling modes:
        1. Complete mode: call(batch) - runs until all sequences halt
        2. Single-step mode: call((carry, batch)) - executes one reasoning step

        Args:
            inputs: Either batch dictionary or (carry, batch) tuple.
            training: Whether in training mode.

        Returns:
            Complete mode: Final outputs dictionary
            Step mode: (new_carry, outputs, all_finished) tuple
        """
        if isinstance(inputs, dict):
            # Standard call - run until convergence
            return self._forward_complete(inputs, training=training)
        else:
            # Step call
            carry, batch = inputs
            return self._forward_step(carry, batch, training=training)

    def _forward_complete(
            self,
            batch: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """Run complete forward pass until all sequences halt.

        Args:
            batch: Input batch dictionary.
            training: Whether in training mode.

        Returns:
            Final outputs dictionary.
        """
        carry = self.initial_carry(batch)

        # The trip count is STATIC and the early exit is a tensor-valued freeze,
        # not a Python ``break``.
        #
        # The previous form was ``for _ in range(halt_max_steps * 2): ... if
        # all_finished: break``. ``all_finished`` is ``ops.all(halted)`` — a
        # symbolic scalar — so the ``if`` raises OperatorNotAllowedInGraphError
        # the moment this runs under ``tf.function``, which is the regime
        # ``fit()`` uses. It only ever worked eagerly.
        #
        # ``is_last_step`` forces every sequence to halt by ``halt_max_steps``,
        # so that is the true bound (the ``* 2`` safety factor was dead). Once
        # a sequence is done we freeze its carry and its outputs, which
        # reproduces the old break exactly: continuing to step would otherwise
        # RESET the halted sequence (``_forward_step`` resets on ``halted``) and
        # restart its reasoning, silently returning a different answer.
        #
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-031: the freeze predicate is
        # the PER-SEQUENCE ``carry["halted"]`` mask, NOT the batch-global
        # ``all_finished`` scalar (``ops.all(halted)``) that ``_forward_step``
        # returns as its third value. With the scalar, a sequence that halts at
        # step 2 in a batch whose slowest member runs to step 8 was reset and
        # re-run six more times, and the dict returned to the caller held, for
        # that sequence, the output of a RESTARTED PARTIAL RUN — not the answer
        # it halted on. Nothing raised; the batch-global gate is only correct
        # for a batch that halts in lockstep, which is exactly what an ACT model
        # is built not to do. ``all_finished`` is deliberately left in
        # ``_forward_step``'s return tuple: it is that method's PUBLIC step-mode
        # contract (``call`` returns it to external drivers), and narrowing the
        # tuple here would be an unrelated API break.
        #
        # Do NOT hoist ``ops.reshape(done, ...)`` out of ``_freeze``: the leaves
        # of ``carry``/``outputs`` have DIFFERENT ranks (``steps`` is
        # ``(batch,)``, ``logits`` is ``(batch, seq, vocab)``), so the broadcast
        # shape must be derived per leaf. This is the same idiom
        # ``_forward_step`` already uses for ``reset_mask``. See decisions.md
        # D-031.
        carry, outputs, _ = self._forward_step(carry, batch, training=training)
        done = carry["halted"]

        def _freeze(mask, held, fresh):
            broadcast = keras.ops.reshape(
                mask, [-1] + [1] * (len(held.shape) - 1))
            return keras.ops.where(broadcast, held, fresh)

        for _ in range(self.halt_max_steps - 1):
            next_carry, next_outputs, _ = self._forward_step(
                carry, batch, training=training)

            frozen = done
            carry = keras.tree.map_structure(
                lambda held, fresh: _freeze(frozen, held, fresh),
                carry, next_carry)
            outputs = keras.tree.map_structure(
                lambda held, fresh: _freeze(frozen, held, fresh),
                outputs, next_outputs)

            # Post-merge, ``carry["halted"]`` is ``where(frozen, True, fresh)``
            # for an already-frozen sequence and the fresh halt decision
            # otherwise — i.e. the accumulated per-sequence done mask.
            done = carry["halted"]

        return outputs

    def _forward_step(
            self,
            carry: Dict[str, keras.KerasTensor],
            batch: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Tuple[Dict[str, keras.KerasTensor], Dict[str, keras.KerasTensor], keras.KerasTensor]:
        """Execute single reasoning step with ACT logic.

        Args:
            carry: Current carry state.
            batch: Input batch dictionary.
            training: Whether in training mode.

        Returns:
            Tuple of (new_carry, outputs, all_finished).
        """
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

            # Exploration: random minimum halt steps.
            # ``keras.random.uniform`` REQUIRES a floating point dtype and raises
            # unconditionally on an integer one (keras/src/random/random.py:121),
            # so the integer draw must come from ``keras.random.randint``. This
            # mirrors tiny_recursive_model/model.py:341, which has always been
            # correct; HRM was the un-migrated copy.
            if self.halt_exploration_prob > 0:
                explore_mask = keras.random.uniform(keras.ops.shape(q_halt)) < self.halt_exploration_prob
                min_steps = keras.random.randint(
                    keras.ops.shape(new_steps),
                    minval=2,
                    maxval=self.halt_max_steps + 1
                )
                min_halt_steps = keras.ops.where(explore_mask, min_steps, 1)
                halted = halted & (new_steps >= min_halt_steps)

            # Compute target Q for bootstrapping (as in original).
            # NOTE: there is deliberately no ``if not is_last_step:`` guard here.
            # ``is_last_step`` is a per-sequence ``(batch,)`` bool tensor, so a
            # Python truth-test on it raises (ambiguous truth value eagerly,
            # OperatorNotAllowedInGraphError under tf.function). The per-sequence
            # branch is already expressed by the ``ops.where`` below, which is
            # what the guard was trying and failing to do.
            #
            # The lookahead runs with ``training=False`` and the target is
            # ``stop_gradient``-ed, so the Bellman target is neither stochastic
            # nor differentiable — otherwise the TD loss can be minimised by
            # dragging the target toward the prediction (standard target-network
            # collapse). Ported from tiny_recursive_model/model.py:346-363 (B-3).
            next_inner_carry, next_outputs = self.core(
                new_inner_carry,
                {"token_ids": new_current_data["token_ids"],
                 "puzzle_ids": new_current_data["puzzle_ids"]},
                training=False
            )

            next_q_halt = next_outputs["q_halt_logits"]
            next_q_continue = next_outputs["q_continue_logits"]

            # Target Q: if last step, use halt; otherwise use max
            target_q = keras.ops.where(
                is_last_step,
                keras.ops.sigmoid(next_q_halt),
                keras.ops.sigmoid(keras.ops.maximum(next_q_halt, next_q_continue))
            )
            outputs["target_q_continue"] = keras.ops.stop_gradient(target_q)

        if not training and self.halt_max_steps > 1:
            # Inference: halt on the learned signal OR max-steps, mirroring
            # training-mode halting minus the exploration branch. Previously
            # ``halted`` was ``is_last_step`` alone at inference, so ACT always
            # ran the full budget and the trained q_head was inert — which
            # contradicts the module docstring at :133 and the "Adaptive
            # Computation" claim at :45. Ported from
            # tiny_recursive_model/model.py:365-382 (B-5).
            halted = is_last_step | (
                outputs["q_halt_logits"] > outputs["q_continue_logits"]
            )

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

    @classmethod
    def from_variant(
            cls,
            variant: str,
            vocab_size: int,
            seq_len: int,
            num_puzzle_identifiers: int = 1000,
            **kwargs: Any
    ) -> "HierarchicalReasoningModel":
        """Create a Hierarchical Reasoning Model from a predefined variant.

        Args:
            variant: String, one of "micro", "tiny", "small", "base", "large", "xlarge"
            vocab_size: Integer, size of vocabulary
            seq_len: Integer, maximum sequence length
            num_puzzle_identifiers: Integer, number of puzzle identifiers
            **kwargs: Additional arguments passed to the constructor

        Returns:
            HierarchicalReasoningModel instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> # Mathematical reasoning model
            >>> model = HierarchicalReasoningModel.from_variant(
            ...     "base", vocab_size=32000, seq_len=512
            ... )
            >>> # Logic puzzle model with high exploration
            >>> model = HierarchicalReasoningModel.from_variant(
            ...     "large", vocab_size=50000, seq_len=1024,
            ...     halt_exploration_prob=0.2
            ... )
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()

        logger.info(f"Creating Hierarchical Reasoning Model-{variant.upper()}")
        logger.info(f"Architecture: {config}")

        # DECISION plan-2026-08-17T183311-79c63e38/D-025: MERGE, do not splat.
        # `**config, **kwargs` raises `TypeError: got multiple values for
        # keyword argument` for ANY override of a variant key, and every key in
        # MODEL_VARIANTS is one. `create_hierarchical_reasoning_model`'s own
        # Sudoku example (`variant="base", halt_max_steps=16`) is exactly this
        # call and could not run. Copied from `models/gpt2/gpt2.py:464`.
        config.update(kwargs)
        return cls(
            vocab_size=vocab_size,
            seq_len=seq_len,
            num_puzzle_identifiers=num_puzzle_identifiers,
            **config
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Returns:
            Dictionary containing the model configuration.
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

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HierarchicalReasoningModel":
        """Create model from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            HierarchicalReasoningModel instance.
        """
        # Handle serialized objects
        if "embeddings_initializer" in config and isinstance(config["embeddings_initializer"], dict):
            config["embeddings_initializer"] = keras.initializers.deserialize(
                config["embeddings_initializer"]
            )
        if "kernel_initializer" in config and isinstance(config["kernel_initializer"], dict):
            config["kernel_initializer"] = keras.initializers.deserialize(
                config["kernel_initializer"]
            )
        if "embeddings_regularizer" in config and config["embeddings_regularizer"]:
            config["embeddings_regularizer"] = keras.regularizers.deserialize(
                config["embeddings_regularizer"]
            )
        if "kernel_regularizer" in config and config["kernel_regularizer"]:
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )

        return cls(**config)

    def summary(self, **kwargs: Any) -> None:
        """Print model summary with additional HRM information.

        Args:
            **kwargs: Additional keyword arguments for summary.
        """
        super().summary(**kwargs)
        logger.info("Hierarchical Reasoning Model Configuration:")
        logger.info(f"  - Vocabulary size: {self.vocab_size:,}")
        logger.info(f"  - Sequence length: {self.seq_len}")
        logger.info(f"  - Embedding dimension: {self.embed_dim}")
        logger.info(f"  - High-level layers: {self.h_layers} (cycles: {self.h_cycles})")
        logger.info(f"  - Low-level layers: {self.l_layers} (cycles: {self.l_cycles})")
        logger.info(f"  - Attention heads: {self.num_heads}")
        logger.info(f"  - Max reasoning steps: {self.halt_max_steps}")
        logger.info(f"  - Exploration probability: {self.halt_exploration_prob}")
        logger.info(f"  - Total parameters: {self.count_params():,}")

    def __repr__(self) -> str:
        """Return string representation of the model.

        Returns:
            String representation including key parameters.
        """
        return (
            f"HierarchicalReasoningModel(vocab_size={self.vocab_size}, "
            f"embed_dim={self.embed_dim}, h_layers={self.h_layers}, "
            f"l_layers={self.l_layers}, halt_max_steps={self.halt_max_steps}, "
            f"name='{self.name}')"
        )


# ---------------------------------------------------------------------
# Factory function to create and configure HRM models
# ---------------------------------------------------------------------

def create_hierarchical_reasoning_model(
        vocab_size: int,
        seq_len: int,
        embed_dim: int = 512,
        num_puzzle_identifiers: int = 1000,
        variant: Optional[str] = None,
        optimizer: Union[str, keras.optimizers.Optimizer] = "adamw",
        learning_rate: float = 1e-4,
        loss: Optional[keras.losses.Loss] = None,
        **kwargs: Any
) -> HierarchicalReasoningModel:
    """Create and optionally compile a Hierarchical Reasoning Model.

    Factory function implementing the research architecture from Wang et al. (2025)
    with sensible defaults based on paper findings. The "base" variant matches the
    exact configuration that achieved 40.3% on ARC-AGI with only ~1000 training
    examples and 27M parameters.

    Research-Validated Configurations:
    - **base**: Paper configuration (27M params, 40.3% ARC-AGI)
    - **AdamW optimizer**: Scale-invariant optimization with bounded parameters
    - **Learning rate 1e-4**: Optimal for hierarchical convergence training
    - **Post-Norm architecture**: With RMSNorm, SwiGLU, and RoPE applied to Q/K
      inside the reasoning modules' grouped-query attention (Llama-style)

    Args:
        vocab_size: Size of vocabulary for token embeddings.
        seq_len: Maximum sequence length for input sequences.
        embed_dim: Embedding dimension (ignored if variant is specified).
        num_puzzle_identifiers: Number of puzzle type identifiers.
        variant: Optional model variant from research configurations:
            - "micro": 1.2M params, minimal reasoning
            - "base": 27M params, paper configuration, 40.3% ARC-AGI
            - "large": 156.7M params, high-capacity reasoning
        optimizer: Optimizer for compilation. Paper uses Adam-atan2 (scale-invariant).
            Pass `None` to skip compilation and drive the model from your own
            training loop (this is what `src/train/hrm/train_hrm.py` does).
        learning_rate: Learning rate. Paper uses 1e-4 with linear warmup.
        loss: Loss used when compiling. Defaults to
            `{"logits": StableMaxCrossEntropy()}`, so the compiled model trains
            with `fit(batch, {"logits": labels})`. This supervises the LM head
            ONLY — the Q-learning halting term couples two model *outputs* and
            cannot be expressed as a per-output Keras loss, so ACT supervision
            requires `src/train/hrm/train_hrm.py`'s loop (or your own `loss=`
            plus a `compute_loss` override). Ignored when `optimizer` is
            `None`.
        **kwargs: Additional arguments for HierarchicalReasoningModel constructor.

    Returns:
        HierarchicalReasoningModel instance, compiled unless `optimizer=None`.

    Example:
        >>> # Reproduce paper ARC-AGI results
        >>> model = create_hierarchical_reasoning_model(
        ...     vocab_size=32000,
        ...     seq_len=512,
        ...     variant="base",  # 27M params, matches paper
        ...     halt_exploration_prob=0.1  # Paper ACT setting
        ... )
        >>>
        >>> # Sudoku solver configuration
        >>> model = create_hierarchical_reasoning_model(
        ...     vocab_size=20,  # 0-9 digits + special tokens
        ...     seq_len=81,     # 9x9 grid flattened
        ...     variant="base",
        ...     halt_max_steps=16  # For backtracking search
        ... )
        >>>
        >>> # Maze pathfinding (30x30)
        >>> model = create_hierarchical_reasoning_model(
        ...     vocab_size=4,   # wall, empty, start, goal
        ...     seq_len=900,    # 30x30 maze flattened
        ...     variant="large",
        ...     halt_exploration_prob=0.2
        ... )
    """
    # Create model from variant or custom specification
    if variant is not None:
        model = HierarchicalReasoningModel.from_variant(
            variant,
            vocab_size=vocab_size,
            seq_len=seq_len,
            num_puzzle_identifiers=num_puzzle_identifiers,
            **kwargs
        )
    else:
        model = HierarchicalReasoningModel(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_puzzle_identifiers=num_puzzle_identifiers,
            **kwargs
        )

    # DECISION plan-2026-08-14T233721-d4f9beb2/D-032: this branch now actually
    # CALLS `model.compile(...)`. It used to resolve the optimizer, set its
    # learning rate, log it and DISCARD it, while both this function's summary
    # line and its `Returns:` promised an "optionally compiled" model — so a
    # caller who followed the docstring got an uncompiled model and
    # `ValueError: You must call compile() before using the model.` on `.fit()`
    # (measured). The default loss is a PER-OUTPUT dict keyed on `"logits"`, so
    # `fit(batch, {"logits": labels})` works with the stock `fit` and no custom
    # `train_step` / `compute_loss`.
    #
    # It is deliberately NOT the package's own `HRMLoss`. `HRMLoss.call` takes
    # the model's whole output DICT and a `{"labels": ...}` target, but Keras 3's
    # `CompileLoss` BROADCASTS a single `Loss` object across each output key and
    # then pairs it against a same-keyed target — measured, both
    # `fit(batch, {"labels": ...})` and `fit(batch, labels)` raise
    # `KeyError: The path: ('logits',) ... can't be found in either the model's
    # output (y_pred) or in the labels (y_true)`. The Q-learning term cannot be
    # written as a per-output Keras loss at all, because it couples
    # `q_halt_logits` with `target_q_continue` and BOTH are model outputs, not
    # labels. So ACT/Q supervision needs a loop that calls `HRMLoss` directly —
    # which is exactly why `src/train/hrm/train_hrm.py` exists and drives this
    # model with its own `create_hrm_loss` and `GradientTape`.
    #
    # Do NOT "upgrade" this default to `create_hrm_loss()` without also adding a
    # `compute_loss` override; it will compile and then fail at the first
    # `fit()` step. Do NOT restore the "log the optimizer and return" form to
    # keep the factory loss-agnostic: `optimizer=None` is already the
    # loss-agnostic route and returns an uncompiled model. See decisions.md
    # D-032.
    if optimizer is not None:
        if isinstance(optimizer, str):
            optimizer = keras.optimizers.get(optimizer)
        if hasattr(optimizer, 'learning_rate'):
            optimizer.learning_rate = learning_rate

        model.compile(
            optimizer=optimizer,
            loss=loss if loss is not None else {
                "logits": StableMaxCrossEntropy()
            },
        )
        logger.info(
            f"Created and compiled Hierarchical Reasoning Model with optimizer "
            f"{optimizer}"
        )

    else:
        logger.info("Created Hierarchical Reasoning Model (uncompiled)")

    # DECISION plan-2026-08-19T163559-499b6f0e/D-051
    # The factory BUILDS the model before returning it. `HierarchicalReasoningModel`
    # has a correct `build()` (it materialises the reasoning core's `h_init`/
    # `l_init` before any `reset_carry`), but nothing CALLED it, so this factory
    # handed back `built == False` and every public method that reads a
    # build-created quantity raised — measured: `src/train/hrm/train_hrm.py:61`
    # `self.model.count_params()` inside `HRMTrainer.__init__` died with
    # "count_params ... isn't built", i.e. the trainer module could not start.
    # `build()` here takes no shape (the model derives everything from
    # `seq_len` / `vocab_size` in its config), so this is safe for every caller.
    # Do NOT replace it with a dummy forward pass: a forward pass also advances
    # the ACT carry and, under a `StatelessScope`, would not persist the
    # variables it creates. See decisions.md D-051.
    if not model.built:
        model.build()

    return model

# ---------------------------------------------------------------------