"""
Neural Arithmetic Module (NAM) — full model.

Merges three architectures for arithmetic expression evaluation:

1. **Tree Transformer** (GroupAttention) — parses expression structure
2. **Neural Turing Machine** (memory + addressing) — a recurrent context vector
   for the controller
3. **Tiny Recursive Model** (ACT loop) — iterative re-scoring of the expression

Arithmetic operations are FIXED — the model learns parsing, routing, and halting.
Each operation outputs both a result and a validity flag (e.g., division by zero → invalid).

.. warning::

   **Scope: single-operator, integer-only.** MEASURED 2026-08-19, pinned by
   ``tests/test_models/test_nam/test_operand_derivation_through_call.py``.
   The two operands are assembled **exclusively** from the raw ``token_ids``
   (``cell.py`` step 5), split at ``argmax(reduction_weights)``, and
   ``NAM.call`` re-reads ``batch["input_ids"]`` unchanged on every ACT step.
   Nothing written to NTM memory ever re-enters ``left_val`` / ``right_val``.
   Two consequences, neither of which is reachable at ANY weights:

   * **Multi-operator expressions do not chain.** ``"1 + 2 * 3"`` splits at the
     ``+`` into ``1`` and the *concatenated* digits ``23``; at the ``*`` into
     ``12`` and ``3``. The answer 7 is not a candidate at any position.
     Parentheses (ids 18/19) are not digits, so they neither delimit nor group
     an operand: ``"( 1 + 2 ) * 3"`` gives ``(12, 3)`` at the ``*``.
   * **Decimals are dropped.** ``is_digit`` is ``4 <= id <= 13``; ``DOT_ID = 20``
     is tokenized and round-trips, but number assembly has no fractional
     branch, so ``"1.5 + 2"`` assembles ``15`` and ``2`` and computes 17 with
     ``valid = 1.0``. There is no error and no validity flag for this.

   Supporting either would be a new mechanism (an operand path from a previous
   step's result; a fractional branch in ``_assemble_number_from_tokens``), not
   a bug fix. Until then this docstring is the contract.

Architecture::

    Expression tokens ──► Embedding ──► TreeEncoder (parse structure)
                                              │
                                    ┌── NAMCell (recursive, weight-shared) ──┐
                                    │  1. Tree induction                     │
                                    │  2. Score sub-expressions              │
                                    │  3. Extract operands (NTM read)        │
                                    │  4. Classify operator                  │
                                    │  5. Fixed arithmetic → (result, valid) │
                                    │  6. Write result to memory             │
                                    │  7. Halt decision (ACT)                │
                                    └────────────────────────────────────────┘
                                              │
                                    Result Head → (scalar, validity)
"""

import keras
from keras import ops
from typing import Any, Dict, List, Optional, Tuple, Union

from dl_techniques.models.tree_transformer.model import (
    PositionalEncoding,
    TreeTransformerBlock,
)
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.utils.logger import logger

from .config import NAMConfig, NAM_VARIANTS
from .cell import NAMCell


@keras.saving.register_keras_serializable()
class NAM(keras.Model):
    """
    Neural Arithmetic Module.

    Evaluates arithmetic expressions by parsing them into tree structures
    and recursively reducing sub-expressions using fixed arithmetic operations.

    Each arithmetic operation produces two outputs:

    - **result**: The computed value.
    - **valid**: 1.0 if the operation is valid, 0.0 if invalid
      (e.g., division by zero).

    **Scope: single-operator, integer-only** — see the module docstring for the
    measurement. A multi-operator expression does not chain (step N's result
    cannot reach step N+1's operands at any weights) and the decimal point is
    tokenized but dropped by number assembly.

    The model uses Adaptive Computation Time (ACT) from TRM to dynamically
    decide how many ACT steps to run. Extra steps re-score the SAME token
    sequence — they do not reduce it, so a second step cannot consume the
    first step's answer. That halting is live at inference as well as during
    training: both branches halt on ``q_halt > 0`` (D-033). It was NOT true
    before 2026-08-15 — inference halted on ``halt_max_steps`` alone, so every
    sequence ran the full budget and the sentence above was false. Note that
    ``q_halt`` only earns its keep if the trainer supervises it;
    ``src/train/nam/train_nam.py`` does so via ``--w-halt`` (D-034), and
    ``--w-halt 0`` reproduces the untrained-head behaviour.

    .. note::

       **This model is driven by a training loop, not by Keras.** ``call``
       takes ``(carry, batch)`` rather than an ``inputs``-first signature, so
       ``predict()``, ``fit()`` and ``.save()``-with-build-config cannot drive
       it; ``src/train/nam/train_nam.py`` calls it directly in a hand-rolled
       ACT unroll. This is an accepted limitation, not an oversight: the
       recursion needs a carry no single input tensor can supply, and
       reordering the parameters would break the trainer and every saved build
       config for a benefit nothing here asks for.

    :param config: NAM configuration (dataclass or dict).
    :type config: Union[NAMConfig, Dict[str, Any]]

    **Inputs** (via ``call``)::

        carry: Dict from ``initial_carry()`` or previous step
        batch: Dict with "input_ids" key, shape (B, L)

    **Outputs** (from ``call``)::

        new_carry: Updated carry dict
        outputs: Dict with:
            - "result": (B, 1) predicted value
            - "valid": (B, 1) validity score
            - "q_halt_logits": (B,) halt signal
            - "q_continue_logits": (B,) continue signal
            - "step_results": list of per-step (result, valid)
    """

    def __init__(
        self,
        config: Union[NAMConfig, Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(config, dict):
            config = NAMConfig.from_dict(config)
        self.config = config

        h = config.hidden_size

        # --- Token embedding ---
        self.token_embedding = keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=h,
            name="token_embedding",
        )

        # --- Numeric value projection ---
        # Injects each token's numeric semantics (digit value, operator type)
        # directly into the embedding so the model doesn't have to learn
        # number extraction from scratch through deep transformer layers.
        # Input: (B, L, 3) — [digit_value, is_digit, operator_type]
        self.numeric_proj = keras.layers.Dense(
            h, name="numeric_proj",
        )

        # --- Positional encoding ---
        self.pos_encoding = PositionalEncoding(
            hidden_size=h,
            dropout_rate=config.hidden_dropout_rate,
            max_len=config.max_expression_len,
            name="pos_encoding",
        )

        # --- Tree encoder (parse expression structure) ---
        self.tree_blocks = [
            TreeTransformerBlock(
                hidden_size=h,
                num_heads=config.num_heads,
                intermediate_size=config.intermediate_size,
                hidden_dropout_rate=config.hidden_dropout_rate,
                attention_dropout_rate=config.attention_dropout_rate,
                normalization_type=config.normalization_type,
                ffn_type=config.ffn_type,
                hidden_act=config.hidden_act,
                layer_norm_eps=config.layer_norm_eps,
                name=f"tree_block_{i}",
            )
            for i in range(config.num_tree_layers)
        ]

        # --- Encoder output norm ---
        self.encoder_norm = create_normalization_layer(
            normalization_type=config.normalization_type,
            epsilon=config.layer_norm_eps,
            name="encoder_norm",
        )

        # --- NAMCell (reused across ACT steps) ---
        self.cell = NAMCell(config=config, name="nam_cell")

        # --- Result readout ---
        self.result_head = keras.layers.Dense(1, name="result_head")
        self.validity_head = keras.layers.Dense(
            1, activation="sigmoid", name="validity_head"
        )

        logger.info(
            f"NAM initialized: hidden_size={h}, "
            f"tree_layers={config.num_tree_layers}, "
            f"memory_size={config.memory_size}, "
            f"halt_max_steps={config.halt_max_steps}"
        )

    def build(self, input_shape: Optional[Any] = None) -> None:
        h = self.config.hidden_size
        L = self.config.max_expression_len

        self.token_embedding.build((None, L))
        self.numeric_proj.build((None, L, 3))

        emb_shape = (None, L, h)
        self.pos_encoding.build(emb_shape)

        mask_shape = (None, 1, L)
        scalar_shape = ()
        block_input = (emb_shape, mask_shape, scalar_shape)
        for block in self.tree_blocks:
            block.build(block_input)

        self.encoder_norm.build(emb_shape)
        self.cell.build(None)
        self.result_head.build((None, h + 2))  # pooled + cell_result + cell_valid
        self.validity_head.build((None, h))

        super().build(input_shape)

    def initial_carry(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create initial carry state for a batch.

        :param batch: Dict with "input_ids" key.
        :type batch: Dict[str, Any]
        :return: Initial carry dictionary.
        :rtype: Dict[str, Any]
        """
        input_ids = batch["input_ids"]
        batch_size = ops.shape(input_ids)[0]

        cell_carry = self.cell.initialize_carry(batch_size)

        return {
            "cell_carry": cell_carry,
            "steps": ops.zeros((batch_size,), dtype="int32"),
            "halted": ops.zeros((batch_size,), dtype="bool"),
        }

    def _encode(
        self,
        input_ids: Any,
        training: Optional[bool] = None,
    ) -> Tuple[Any, Any, List[Any]]:
        """
        Encode input tokens through the tree encoder.

        :param input_ids: Token IDs of shape (B, L).
        :type input_ids: Any
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Tuple of (hidden, mask, break_probs).
        :rtype: Tuple[Any, Any, List[Any]]
        """
        # Create padding mask
        mask = ops.cast(ops.not_equal(input_ids, 0), "int32")
        mask = ops.expand_dims(mask, axis=1)  # (B, 1, L)

        # --- Build numeric feature channels ---
        # Digit value: tokens 4-13 map to 0-9, everything else → 0
        ids_float = ops.cast(input_ids, self.compute_dtype)
        is_digit = ops.cast(
            ops.logical_and(
                ops.greater_equal(input_ids, 4),
                ops.less_equal(input_ids, 13),
            ),
            self.compute_dtype,
        )
        digit_value = (ids_float - 4.0) * is_digit  # 0-9 for digits, 0 otherwise

        # Operator type: +→1, -→2, *→3, /→4, else→0
        op_type = ops.zeros_like(ids_float)
        op_type = ops.where(ops.equal(input_ids, 14), ops.ones_like(op_type) * 1.0, op_type)
        op_type = ops.where(ops.equal(input_ids, 15), ops.ones_like(op_type) * 2.0, op_type)
        op_type = ops.where(ops.equal(input_ids, 16), ops.ones_like(op_type) * 3.0, op_type)
        op_type = ops.where(ops.equal(input_ids, 17), ops.ones_like(op_type) * 4.0, op_type)

        # Stack: (B, L, 3) — [digit_value, is_digit, operator_type]
        numeric_features = ops.stack(
            [digit_value, is_digit, op_type], axis=-1
        )

        # Embed tokens + inject numeric semantics
        import math
        x = self.token_embedding(input_ids)
        x = x * math.sqrt(self.config.hidden_size)
        x = x + self.numeric_proj(numeric_features)  # add numeric channel
        x = self.pos_encoding(x, training=training)

        # Run through tree encoder blocks
        group_prob = ops.convert_to_tensor(0.0, dtype=self.compute_dtype)
        all_break_probs = []

        for block in self.tree_blocks:
            x, group_prob, break_prob = block(
                (x, mask, group_prob), training=training
            )
            all_break_probs.append(break_prob)

        x = self.encoder_norm(x)
        return x, mask, all_break_probs

    def call(
        self,
        carry: Dict[str, Any],
        batch: Dict[str, Any],
        training: Optional[bool] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute one ACT step (one sub-expression reduction).

        This method is called repeatedly in an external loop until all
        sequences in the batch have halted.

        :param carry: Carry state from previous step or ``initial_carry()``.
        :type carry: Dict[str, Any]
        :param batch: Dict with "input_ids" (B, L) and "encoded" (B, L, D)
            if already encoded, plus "mask" (B, 1, L).
        :type batch: Dict[str, Any]
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Tuple of (new_carry, outputs).
        :rtype: Tuple[Dict[str, Any], Dict[str, Any]]
        """
        # Encode if not already done
        if "encoded" not in batch:
            encoded, mask, encoder_break_probs = self._encode(
                batch["input_ids"], training=training
            )
            batch = {
                **batch,
                "encoded": encoded,
                "mask": mask,
                "encoder_break_probs": encoder_break_probs,
            }

        hidden = batch["encoded"]
        mask = batch["mask"]

        # --- Run NAMCell ---
        # Pass token_ids for deterministic number assembly inside the cell.
        token_ids = batch["input_ids"]
        new_cell_carry, cell_outputs = self.cell(
            (carry["cell_carry"], hidden, mask, token_ids),
            training=training,
        )

        # Update hidden in batch for next step
        batch = {**batch, "encoded": cell_outputs["hidden"]}

        # --- Halt logic (from TRM) ---
        steps = carry["steps"] + 1
        q_halt = cell_outputs["q_halt"]
        q_continue = cell_outputs["q_continue"]

        is_last_step = ops.greater_equal(steps, self.config.halt_max_steps)

        if training is True:
            halt_signal = ops.greater(q_halt, ops.cast(0.0, q_halt.dtype))
            new_halted = ops.logical_or(is_last_step, halt_signal)

            # Exploration: randomly prevent early halting
            rand_val = keras.random.uniform(ops.shape(q_halt))
            explore = ops.less(rand_val, self.config.halt_exploration_prob)
            min_steps = ops.where(
                explore,
                ops.cast(
                    keras.random.randint(
                        ops.shape(steps),
                        minval=2,
                        maxval=self.config.halt_max_steps + 1,
                    ),
                    "int32",
                ),
                ops.ones_like(steps),
            )
            step_ok = ops.greater_equal(steps, min_steps)
            new_halted = ops.logical_and(new_halted, step_ok)
        else:
            # DECISION plan-2026-08-14T233721-d4f9beb2/D-033: inference consults
            # the LEARNED `q_halt` too, mirroring the training branch above minus
            # the exploration term. This used to be `new_halted = is_last_step`
            # alone, so every sequence ran the full `halt_max_steps` at eval and
            # the trained ACT head was INERT at exactly the moment adaptivity is
            # claimed — contradicting this class's own docstring ("simple
            # expressions like ``1 + 2`` take 1 step"). `halt_exploration_prob`
            # and `halt_max_steps` bought nothing measurable.
            #
            # The predicate is `q_halt > 0`, NOT `q_halt > q_continue`: that is
            # what the training branch uses, and the two branches must agree or
            # the head is trained against one rule and read under another. It is
            # the same choice `tiny_recursive_model` exposes as
            # `no_act_continue=True`. `q_continue` is therefore NOT on NAM's
            # halting path at all; do not "restore symmetry" by switching this
            # to a two-action comparison without also changing the training
            # branch and adding the Bellman arm to the trainer's `L_halt`.
            # See decisions.md D-033.
            halt_signal = ops.greater(q_halt, ops.cast(0.0, q_halt.dtype))
            new_halted = ops.logical_or(is_last_step, halt_signal)

        # --- Result readout ---
        # Pool hidden state for final prediction
        token_mask = ops.squeeze(mask, axis=1)  # (B, L)
        token_mask_float = ops.cast(token_mask, self.compute_dtype)
        pooled = ops.sum(
            cell_outputs["hidden"] * ops.expand_dims(token_mask_float, -1),
            axis=1,
        )
        pooled = pooled / (ops.sum(token_mask_float, axis=-1, keepdims=True) + 1e-9)

        # Concatenate the cell's intermediate arithmetic result as a hint
        # so the result head can refine it rather than compute from scratch.
        # Stop gradient: L_result trains ONLY the result_head weights, not
        # upstream sub-skill modules (number extraction, op classifier, etc.).
        # Without this, result loss gradients propagate backward through the
        # entire model and degrade sub-skill learning.
        result_input = ops.stop_gradient(
            ops.concatenate(
                [pooled, cell_outputs["result"], cell_outputs["valid"]], axis=-1
            )
        )

        final_result = self.result_head(result_input)  # (B, 1)
        final_valid = self.validity_head(ops.stop_gradient(pooled))  # (B, 1)

        # --- Build new carry ---
        new_carry = {
            "cell_carry": new_cell_carry,
            "steps": steps,
            "halted": new_halted,
        }

        outputs = {
            "result": final_result,
            "valid": final_valid,
            "step_result": cell_outputs["result"],
            "step_valid": cell_outputs["valid"],
            "op_logits": cell_outputs["op_logits"],
            "q_halt_logits": q_halt,
            "q_continue_logits": q_continue,
            "batch": batch,
            # Intermediate predictions for multi-task supervision
            "step_left_val": cell_outputs["left_val"],
            "step_right_val": cell_outputs["right_val"],
            "reduction_weights": cell_outputs["reduction_weights"],
        }

        return new_carry, outputs

    #: House-shape canonical name for the variant table. This is an ALIAS to
    #: ``config.NAM_VARIANTS`` -- the same dict object, not a copy -- because
    #: trainers and tests reference the module-level spelling.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = NAM_VARIANTS

    @classmethod
    def from_variant(
        cls,
        variant: str,
        **kwargs: Any,
    ) -> "NAM":
        """
        Create a NAM model from a preset variant.

        :param variant: One of "tiny", "small", "base".
        :type variant: str
        :param kwargs: Override config parameters.
        :return: NAM model instance.
        :rtype: NAM
        """
        if variant not in NAM_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(NAM_VARIANTS.keys())}"
            )
        config_dict = {**NAM_VARIANTS[variant], **kwargs}
        config = NAMConfig(**config_dict)
        return cls(config=config)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["config"] = self.config.to_dict()
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NAM":
        nam_config = config.pop("config")
        return cls(config=nam_config, **config)


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def create_nam(
    variant: str = "base",
    **kwargs: Any,
) -> NAM:
    """
    Create a NAM model from a preset variant.

    :param variant: One of ``NAM.MODEL_VARIANTS`` ("tiny", "small", "base").
    :type variant: str
    :param kwargs: Individual ``NAMConfig`` field overrides applied on top of
        the variant.
    :return: A configured NAM instance.
    :rtype: NAM
    :raises ValueError: If ``variant`` is not a known variant name, or if the
        resulting config fails ``NAMConfig`` validation.

    Example::

        model = create_nam("tiny", halt_max_steps=4)
        carry = model.initial_carry(batch)
        carry, outputs = model(carry, batch)
    """
    return NAM.from_variant(variant, **kwargs)

# ---------------------------------------------------------------------
