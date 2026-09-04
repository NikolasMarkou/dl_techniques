"""
Neural Arithmetic Module (NAM), the full model.

Merges three architectures for arithmetic expression evaluation: a tree
transformer (GroupAttention) parses the expression's structure, an NTM
(memory plus addressing) gives the controller a recurrent context vector,
and a tiny-recursive-model ACT loop re-scores the expression over several
steps. Arithmetic operations themselves are fixed; the model only learns
parsing, routing, and halting. Each operation outputs a result and a
validity flag (division by zero gives ``valid = 0.0``).

.. warning::

   **Scope: single-operator, integer-only.** This model handles one
   operator per expression and integers only, measured and pinned by
   ``tests/test_models/test_nam/test_operand_derivation_through_call.py``.
   Operands are assembled only from the raw ``token_ids`` (``cell.py`` step
   5), split at ``argmax(reduction_weights)``; ``NAM.call`` re-reads
   ``batch["input_ids"]`` unchanged on every ACT step, so nothing written
   to NTM memory ever re-enters an operand. As a result, a multi-operator
   expression does not chain — ``"1 + 2 * 3"`` splits at ``+`` into ``1``
   and the concatenated digits ``23``, and parentheses do not group
   (``"( 1 + 2 ) * 3"`` gives ``(12, 3)`` at the ``*``) — and a decimal
   point is tokenized but dropped by number assembly, so
   ``"1.5 + 2"`` computes ``15 + 2 = 17`` with ``valid = 1.0`` and no error.

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

References:
    - Wang et al., 2019. Tree Transformer: Integrating Tree Structures into
      Self-Attention. EMNLP 2019. (https://arxiv.org/abs/1909.06639) -- the
      GroupAttention that parses expression structure.
    - Graves et al., 2014. Neural Turing Machines.
      (https://arxiv.org/abs/1410.5401) -- the addressed external memory that
      supplies the controller's context vector.
    - Jolicoeur-Martineau, 2025. Less is More: Recursive Reasoning with Tiny
      Networks. (https://arxiv.org/abs/2510.04871) -- the Tiny Recursive Model
      whose ACT loop this re-scores expressions with.
    - Graves, 2016. Adaptive Computation Time for Recurrent Neural Networks.
      (https://arxiv.org/abs/1603.08983) -- the halting mechanism itself.
    - Trask et al., 2018. Neural Arithmetic Logic Units.
      (https://arxiv.org/abs/1808.00508) -- the prior line on learned
      arithmetic; NAM differs by FIXING the operators and learning only
      parsing, routing and halting.
"""

import numpy as np
import keras
from keras import ops
from typing import Any, Dict, List, Optional, Tuple, Union

from dl_techniques.models.language.tree_transformer.model import (
    PositionalEncoding,
    TreeTransformerBlock,
)
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.utils.logger import logger

from .config import NAMConfig, NAM_VARIANTS
from .cell import NAMCell
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.models.nam.model")
class NAM(keras.Model):
    """
    Neural Arithmetic Module.

    Evaluates arithmetic expressions by parsing them into tree structures
    and recursively reducing sub-expressions with fixed arithmetic
    operations. Each operation returns a result and a validity flag (0.0
    for an invalid operation such as division by zero). **Scope:
    single-operator, integer-only** — see the module docstring for what
    that means for multi-operator expressions and decimals.

    The model uses Adaptive Computation Time (ACT), from TRM, to decide how
    many steps to run. Extra steps re-score the same token sequence; they
    do not reduce it, so a later step cannot consume an earlier step's
    answer. Halting is live at both training and inference, on
    ``q_halt > 0`` (D-033), and only reflects anything the trainer taught
    it if training supervised ``q_halt`` — ``src/train/nam/train_nam.py``
    does this via ``--w-halt`` (D-034); ``--w-halt 0`` reproduces an
    untrained halting head.

    .. note::

       ``call`` takes ``(carry, batch)`` rather than an ``inputs``-first
       signature, so ``predict()``, ``fit()``, and ``.save()``-with-build-config
       cannot drive it directly; ``src/train/nam/train_nam.py`` calls it in
       a hand-rolled ACT unroll instead. The recursive carry has no single
       tensor it can be folded into.

    :param config: NAM configuration (dataclass or dict).
    :type config: Union[NAMConfig, Dict[str, Any]]

    Input shape:
        ``call`` takes ``(carry, batch)``: ``carry`` is a dict from
        :meth:`initial_carry` or a previous step, ``batch`` is a dict with
        an ``"input_ids"`` key of shape ``(B, L)``.

    Output shape:
        ``(new_carry, outputs)``: ``new_carry`` is the updated carry dict;
        ``outputs`` holds ``"result"`` and ``"valid"`` (each ``(B, 1)``),
        ``"q_halt_logits"`` and ``"q_continue_logits"`` (each ``(B,)``),
        and ``"step_results"``, a list of per-step ``(result, valid)``.
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
            # DECISION plan-2026-08-14T233721-d4f9beb2/D-033: inference must consult
            # the learned q_halt too (predicate q_halt > 0, matching training), not just is_last_step. See decisions.md.
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
        # DECISION plan-2026-08-19T163559-499b6f0e/D-050: floor must be
        # max(1e-9, finfo(dtype).tiny); np.float16(1e-9) is exactly 0.0, so a bare literal NaNs under mixed_float16. See decisions.md.
        mean_eps = max(1e-9, float(np.finfo(self.compute_dtype).tiny))
        pooled = pooled / (
            ops.sum(token_mask_float, axis=-1, keepdims=True) + mean_eps
        )

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
