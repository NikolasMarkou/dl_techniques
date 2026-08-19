"""
Neural Arithmetic Module Cell — single reduction step.

Performs one step of expression reduction:
1. Tree induction via GroupAttention (identifies sub-expression structure)
2. Sub-expression scoring (which sub-expression to reduce)
3. Operand assembly from the RAW token ids, split at that position
4. Operator classification (which fixed arithmetic op to apply)
5. Fixed arithmetic execution with validity tracking
6. Result writeback to NTM memory
7. Halt decision (is the expression fully reduced?)

Arithmetic operations are FIXED (not learned). The cell learns to parse,
route, and decide when to halt.

**Scope: single-operator, integer-only.** Step 3 reads ``token_ids`` and
nothing else — the NTM read heads (step 6/step 11) feed the *controller*, not
the operands, so the step-6 writeback can move WHICH position is selected but
can never supply a numeric operand. A multi-operator expression therefore
concatenates the far side's digits rather than reducing them (``"1 + 2 * 3"``
-> ``(1, 23)`` at the ``+``), and ``DOT_ID`` is excluded from ``is_digit`` so
decimals lose their point (``"1.5 + 2"`` -> ``(15, 2)``). See the ``NAM`` module
docstring and ``tests/test_models/test_nam/test_operand_derivation_through_call.py``.
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple

from dl_techniques.layers.memory.baseline_ntm import (
    NTMMemory,
    NTMReadHead,
    NTMWriteHead,
)
from dl_techniques.layers.memory.ntm_interface import (
    AddressingMode,
    MemoryState,
)
from dl_techniques.models.tree_transformer.model import (
    GroupAttention,
    TreeMHA,
)
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.ffn.factory import assemble_ffn_config

from .config import NAMConfig


# ── Fixed arithmetic operations (NOT learned) ──────────────────────────


def _fixed_add(a: Any, b: Any) -> Tuple[Any, Any]:
    """Fixed addition. Always valid."""
    return ops.add(a, b), ops.ones_like(a)


def _fixed_subtract(a: Any, b: Any) -> Tuple[Any, Any]:
    """Fixed subtraction. Always valid."""
    return ops.subtract(a, b), ops.ones_like(a)


def _fixed_multiply(a: Any, b: Any) -> Tuple[Any, Any]:
    """Fixed multiplication. Always valid."""
    return ops.multiply(a, b), ops.ones_like(a)


def _fixed_divide(a: Any, b: Any, epsilon: float = 1e-7) -> Tuple[Any, Any]:
    """
    Fixed division with validity check.

    Returns (0.0, 0.0) when the denominator is near zero.
    """
    abs_b = ops.abs(b)
    valid = ops.cast(ops.greater(abs_b, epsilon), a.dtype)
    safe_b = ops.where(ops.greater(abs_b, epsilon), b, ops.ones_like(b))
    result = ops.divide(a, safe_b)
    # zero out invalid results
    result = result * valid
    return result, valid


# ── Deterministic number assembly (ZERO learned parameters) ──────────


def _assemble_number_from_tokens(
    token_ids: Any,
    digit_mask: Any,
) -> Any:
    """Assemble a multi-digit number from token digit values.

    This is a DETERMINISTIC function — no learned parameters. The tokenizer
    encodes digits 0-9 as token IDs 4-13. Given a mask indicating which
    tokens belong to this number, we compute:

        value = sum(digit_value_i * 10^position_i)

    where position_i is the digit's place value (0=units, 1=tens, etc.),
    derived from the count of digits to its right within the number.

    :param token_ids: (B, L) int — raw token IDs from the tokenizer.
    :param digit_mask: (B, L) float — 1.0 for each digit belonging to
        this number, 0.0 elsewhere. Computed from is_digit AND side-of-operator.
    :return: (B, 1) float — the assembled scalar number value.
    """
    # Extract digit values: token 4='0', 5='1', ..., 13='9'
    is_digit = ops.cast(
        ops.logical_and(
            ops.greater_equal(token_ids, 4),
            ops.less_equal(token_ids, 13),
        ),
        "float32",
    )
    digit_values = ops.cast(token_ids - 4, "float32") * is_digit  # 0-9

    # Count digits to the right of each position (within this number).
    # cumsum_left[i] = number of this-number digits at positions <= i.
    # power_of_10[i] = total_digits - cumsum_left[i] (0 for rightmost = units).
    cumsum_left = ops.cumsum(digit_mask, axis=1)
    total_digits = ops.sum(digit_mask, axis=1, keepdims=True)  # (B, 1)
    power_of_10 = (total_digits - cumsum_left) * digit_mask  # (B, L)

    # Positional weights: 10^0=1, 10^1=10, 10^2=100, ...
    positional_weight = ops.power(
        ops.cast(10.0, "float32"), power_of_10
    ) * digit_mask

    # Assemble: sum(digit_value * positional_weight)
    value = ops.sum(digit_values * positional_weight, axis=1, keepdims=True)
    return value  # (B, 1)


# ── NAMCell ─────────────────────────────────────────────────────────────


@keras.saving.register_keras_serializable()
class NAMCell(keras.layers.Layer):
    """
    Single reduction step of the Neural Arithmetic Module.

    Each call to this cell reduces one sub-expression within the arithmetic
    expression. The cell combines:

    - **Tree Transformer** (GroupAttention + TreeMHA) for structural parsing
    - **NTM Memory** as the controller's recurrent context (it does NOT supply
      the operands — those come from the raw tokens; see the module docstring)
    - **Fixed arithmetic units** (add, sub, mul, div) with validity flags
    - **Halting head** for adaptive computation time

    **Architecture:**

    .. code-block:: text

        hidden_state ──► GroupAttention ──► tree structure
                              │
                              ▼
                         TreeMHA (structure-modulated attention)
                              │
                              ▼
                    ┌── Reduction Scorer ──► select sub-expression
                    │
                    ├── Token split at that position ──► operands (left, right)
                    │
                    ├── NTM Read Heads ──► controller context (NOT operands)
                    │
                    ├── Op Classifier ──► identify operator (+,-,*,/)
                    │
                    ├── Fixed Arithmetic ──► compute (result, valid)
                    │
                    ├── NTM Write Head ──► store result in memory
                    │
                    └── Halt Head ──► should we stop?

    :param config: NAM configuration.
    :type config: NAMConfig
    """

    def __init__(self, config: NAMConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if isinstance(config, dict):
            config = NAMConfig.from_dict(config)
        self.config = config
        h = config.hidden_size

        # --- Tree parsing layers ---
        self.group_attention = GroupAttention(
            hidden_size=h,
            normalization_type=config.normalization_type,
            name="group_attention",
        )
        self.tree_mha = TreeMHA(
            num_heads=config.num_heads,
            hidden_size=h,
            attention_dropout_rate=config.attention_dropout_rate,
            name="tree_mha",
        )
        self.attn_norm = create_normalization_layer(
            normalization_type=config.normalization_type,
            epsilon=config.layer_norm_eps,
            name="attn_norm",
        )
        self.attn_dropout = keras.layers.Dropout(config.hidden_dropout_rate)
        # DECISION plan-2026-07-30T140922-8af1028f/D-022
        # `activation`/`dropout_rate` are this cell's OWN generic defaults, so
        # they go through `assemble_ffn_config`, which intersects them with
        # what `config.ffn_type` actually accepts. Without it, `differential`,
        # `gelu_tanh`, `squared_relu` and `swiglu` each had `activation`
        # silently discarded by `create_ffn_layer` -- and that factory now
        # RAISES on a dropped key, so this filter is what keeps those four
        # types constructible here.
        # DISCARD, not rename: this site never enumerated FFN types, so
        # `hidden_act` is an unconditional default and not an expressed
        # per-type intent (contrast D-021, where `differential` was listed by
        # name in the injecting branch). A future consumer that genuinely
        # wants `hidden_act` on `DifferentialFFN.branch_activation` should
        # route this site through
        # `layers.transformers.transformer.build_transformer_ffn_config`,
        # which already owns that rename -- do NOT hand-add a fourth copy of
        # the per-type policy table here (D-018).
        self.ffn = create_ffn_layer(
            ffn_type=config.ffn_type,
            name="ffn",
            **assemble_ffn_config(
                config.ffn_type,
                {
                    "hidden_dim": config.intermediate_size,
                    "output_dim": h,
                    "activation": config.hidden_act,
                    "dropout_rate": config.hidden_dropout_rate,
                },
            ),
        )
        self.ffn_norm = create_normalization_layer(
            normalization_type=config.normalization_type,
            epsilon=config.layer_norm_eps,
            name="ffn_norm",
        )
        self.ffn_dropout = keras.layers.Dropout(config.hidden_dropout_rate)

        # --- Sub-expression scoring ---
        self.reduction_scorer = keras.layers.Dense(1, name="reduction_scorer")

        # Operand extraction is DETERMINISTIC — no learned projections needed.
        # See _assemble_number_from_tokens(): numbers are assembled from
        # tokenizer digit values + predicted operator position.
        # Previous Dense(1) heads (left_number_head, right_number_head) were
        # removed because Dense(D→1) cannot perform the nonlinear positional
        # multiplication (digit × 10^position) required for multi-digit assembly.

        # --- Operator classification (4 ops: +, -, *, /) ---
        self.op_classifier = keras.layers.Dense(4, name="op_classifier")

        # --- NTM memory ---
        self.memory = NTMMemory(
            memory_size=config.memory_size,
            memory_dim=h,
            epsilon=config.epsilon,
            name="ntm_memory",
        )
        # DECISION plan-2026-08-17T183311-79c63e38/D-014
        # CONTENT addressing is NAM's design, not an oversight — both heads have
        # been CONTENT since the package's first commit, and NAM's memory is a
        # content-keyed scratchpad for an expression evaluator, with no notion of
        # an adjacent slot for a circular shift to reach.
        #
        # WHAT NOT TO DO: do not restore `shift_range=config.shift_range` here.
        # Under CONTENT, `NTMReadHead.__init__` does not create `shift_dense`
        # (D-073), so the argument reached a branch NAM never takes and
        # `NAMConfig.shift_range` configured nothing at all. It has been deleted
        # from the config rather than reinstated. Switching these two sites to
        # `AddressingMode.HYBRID` to "use" it would be the opposite change: it
        # adds three trained projections and changes the memory semantics of a
        # shipped model to give a copy-pasted knob something to do.
        # See decisions.md D-014.
        self.read_heads = [
            NTMReadHead(
                memory_size=config.memory_size,
                memory_dim=h,
                addressing_mode=AddressingMode.CONTENT,
                name=f"read_head_{i}",
            )
            for i in range(config.num_read_heads)
        ]
        # DECISION plan-2026-08-18T140459-7991552f/D-035
        # ONE write head, deliberately, and there is no `config.num_write_heads`
        # to loop over any more. Do NOT re-add that field and wrap this in a
        # comprehension mirroring `read_heads` above: the field existed for
        # months as a default, three identical variant entries and a serialized
        # key while this line stayed a bare attribute, so every model that ever
        # advertised it wrote through exactly one head. Honouring it now would
        # silently change the memory semantics and the weight tree of every
        # shipped checkpoint to give a copy-pasted knob something to do — the
        # same trade D-014 refused for `shift_range`. `num_read_heads` above IS
        # live and stays. See decisions.md.
        self.write_head = NTMWriteHead(
            memory_size=config.memory_size,
            memory_dim=h,
            addressing_mode=AddressingMode.CONTENT,
            name="write_head",
        )

        # --- Controller (fuses tree output + memory reads) ---
        self.controller = keras.layers.Dense(h, activation="relu", name="controller")

        # --- Result encoder (scalar → embedding) ---
        self.result_encoder = keras.layers.Dense(h, name="result_encoder")

        # --- Validity encoder (scalar → embedding) ---
        self.validity_encoder = keras.layers.Dense(h, name="validity_encoder")

        # --- Hidden state update ---
        self.state_update = keras.layers.Dense(h, name="state_update")

        # --- Halt decision ---
        self.halt_head = keras.layers.Dense(2, name="halt_head")

    def build(self, input_shape: Any) -> None:
        h = self.config.hidden_size
        seq_shape = (None, self.config.max_expression_len, h)
        mask_shape = (None, 1, self.config.max_expression_len)
        scalar_shape = ()

        self.group_attention.build((seq_shape, mask_shape, scalar_shape))
        group_prob_shape = (None, self.config.max_expression_len, self.config.max_expression_len)
        mha_input = (seq_shape, seq_shape, seq_shape, group_prob_shape, mask_shape)
        self.tree_mha.build(mha_input)
        self.attn_norm.build(seq_shape)
        self.ffn.build(seq_shape)
        self.ffn_norm.build(seq_shape)

        self.reduction_scorer.build(seq_shape)
        # No left_proj/right_proj/number_head builds — number extraction is deterministic
        self.op_classifier.build(seq_shape)

        controller_input_dim = h + self.config.num_read_heads * h
        controller_shape = (None, controller_input_dim)
        self.controller.build(controller_shape)

        for head in self.read_heads:
            head.build((None, h))
        self.write_head.build((None, h))

        self.result_encoder.build((None, 2))  # result + valid
        self.validity_encoder.build((None, 1))
        # state_update input: controller(D) + result_emb(D) + validity(D) + post_read(num_read_heads * D)
        state_update_dim = h * (3 + self.config.num_read_heads)
        self.state_update.build((None, state_update_dim))

        self.halt_head.build((None, h))

        super().build(input_shape)

    def initialize_carry(self, batch_size: int) -> Dict[str, Any]:
        """
        Create initial carry state for a new expression.

        :param batch_size: Batch size.
        :type batch_size: int
        :return: Initial carry dictionary.
        :rtype: Dict[str, Any]
        """
        h = self.config.hidden_size
        m = self.config.memory_size

        memory_state = self.memory.initialize_state(batch_size)

        read_weights = [
            ops.ones((batch_size, m)) / m
            for _ in range(self.config.num_read_heads)
        ]
        write_weights = ops.ones((batch_size, m)) / m

        return {
            "memory": memory_state.memory,
            "memory_usage": memory_state.usage,
            "read_weights": read_weights,
            "write_weights": write_weights,
            "accumulated_result": ops.zeros((batch_size, 1)),
            "accumulated_valid": ops.ones((batch_size, 1)),
            "steps": ops.zeros((batch_size,), dtype="int32"),
        }

    def call(
        self,
        inputs: Tuple[Dict[str, Any], Any, Any, Any],
        training: Optional[bool] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute one reduction step.

        :param inputs: Tuple of (carry, hidden_state, mask, token_ids).
            - carry: Dictionary from previous step or initialize_carry().
            - hidden_state: (B, L, D) current expression representation.
            - mask: (B, 1, L) padding mask.
            - token_ids: (B, L) int — raw token IDs for deterministic
              number assembly.
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Tuple of (new_carry, outputs).
            - new_carry: Updated carry state (with stop_gradient).
            - outputs: Dict with 'result', 'valid', 'op_logits',
              'q_halt', 'q_continue', 'hidden', 'left_val', 'right_val'.
        :rtype: Tuple[Dict[str, Any], Dict[str, Any]]
        """
        carry, hidden, mask, token_ids = inputs
        h = self.config.hidden_size

        # --- 1. Tree induction ---
        group_prob, break_prob = self.group_attention(
            (hidden, mask, ops.convert_to_tensor(0.0, dtype=self.compute_dtype)),
            training=training,
        )

        # --- 2. Structure-modulated attention (pre-LN) ---
        hidden_norm = self.attn_norm(hidden)
        attn_out = self.tree_mha(
            (hidden_norm, hidden_norm, hidden_norm, group_prob, mask),
            training=training,
        )
        hidden = hidden + self.attn_dropout(attn_out, training=training)

        # --- 3. FFN (pre-LN) ---
        hidden_norm2 = self.ffn_norm(hidden)
        ffn_out = self.ffn(hidden_norm2, training=training)
        hidden = hidden + self.ffn_dropout(ffn_out, training=training)

        # --- 4. Score sub-expressions for reduction ---
        # squeeze mask for token-level ops: (B, L)
        token_mask = ops.squeeze(mask, axis=1)
        token_mask_float = ops.cast(token_mask, self.compute_dtype)

        scores = ops.squeeze(self.reduction_scorer(hidden), axis=-1)  # (B, L)
        # DECISION plan-2026-08-17T183311-79c63e38/D-024
        # dtype-aware mask sentinel. `-1e9` is below float16's finite floor
        # (~-6.55e4), so under `mixed_float16` it converts to `-inf` and this
        # line produced NaN TWICE OVER: a fully-masked row softmaxes
        # `[-inf, ...]` to NaN, and — the one a "check the masked positions"
        # guard misses — an UNMASKED position computes `0.0 * -inf`, which is
        # also NaN. The corruption therefore lands on the positions the mask is
        # meant to KEEP, in a row with a perfectly ordinary right-padding mask.
        #
        # The idiom is the LOCAL one from `tree_transformer/components.py:283`
        # (`GroupAttention`, D-001) and `:481`, not
        # `layers/attention/common.apply_attention_mask`. That helper has zero
        # call sites anywhere inside `models/`, and this is a reduction SCORER
        # rather than attention — importing it would open a models/ -> attention
        # internals dependency edge for a two-token change. `sequence_pooling`
        # made the same call for the same reason (SYSTEM.md). This cell already
        # composes the tree_transformer components that use this exact idiom, so
        # the two now agree instead of differing inside one call graph.
        neg_inf = -1e4 if self.compute_dtype == "float16" else -1e9
        scores = scores + (1.0 - token_mask_float) * neg_inf
        reduction_weights = ops.softmax(scores, axis=-1)  # (B, L)

        # --- 5. Deterministic number assembly from tokens ---
        # DECISION plan-2026-08-18T140459-7991552f/D-055
        # This split is the WHOLE numeric path, and it is single-operator,
        # integer-only BY CONSTRUCTION. Do NOT add a docstring, README line or
        # example that promises multi-step reduction or decimals without first
        # building the mechanism: `token_ids` is the raw input, re-read
        # unchanged by `NAM.call` on every ACT step, so a previous step's
        # result cannot become an operand at ANY weights, and `is_digit`
        # excludes DOT_ID so a decimal point is silently dropped
        # ("1.5 + 2" assembles 15). The docs were corrected to match rather
        # than the machine built (Assumption A1). Pinned by
        # tests/test_models/test_nam/test_documented_scope.py and measured by
        # test_operand_derivation_through_call.py. See decisions.md D-055.
        # The operator position is the argmax of reduction_weights (already
        # trained to 100% accuracy). Given the position, we split the tokens
        # into left-of-operator and right-of-operator digit masks, then
        # assemble each number as sum(digit_value * 10^position_in_number).
        # This is EXACT — no learned parameters, no Dense(1) bottleneck.
        op_pos = ops.argmax(reduction_weights, axis=-1)  # (B,)
        seq_len = ops.shape(token_ids)[1]
        positions = ops.cast(ops.arange(seq_len), "int32")  # (L,)
        op_pos_expanded = ops.expand_dims(ops.cast(op_pos, "int32"), axis=-1)  # (B, 1)

        is_digit = ops.cast(
            ops.logical_and(
                ops.greater_equal(token_ids, 4),
                ops.less_equal(token_ids, 13),
            ),
            "float32",
        )
        left_digit_mask = is_digit * ops.cast(
            ops.less(positions, op_pos_expanded), "float32"
        )  # (B, L)
        right_digit_mask = is_digit * ops.cast(
            ops.greater(positions, op_pos_expanded), "float32"
        )  # (B, L)

        left_val = _assemble_number_from_tokens(token_ids, left_digit_mask)   # (B, 1)
        right_val = _assemble_number_from_tokens(token_ids, right_digit_mask)  # (B, 1)

        # reduction_weights are still used for pooling hidden state for
        # the operator classifier and NTM addressing.
        rw = ops.expand_dims(reduction_weights, axis=-1)  # (B, L, 1)

        # --- 6. Pre-write read from NTM memory for context ---
        memory_state = MemoryState(
            memory=carry["memory"],
            usage=carry["memory_usage"],
        )

        pooled = ops.sum(hidden * rw, axis=1)  # (B, D)
        pre_read_vectors = []
        new_read_weights = []
        for i, head in enumerate(self.read_heads):
            weights, _ = head.compute_addressing(
                pooled, memory_state, carry["read_weights"][i]
            )
            rv = self.memory.read(memory_state, weights)
            pre_read_vectors.append(rv)
            new_read_weights.append(weights)

        # --- 7. Controller: fuse tree output + memory reads ---
        controller_input = ops.concatenate([pooled] + pre_read_vectors, axis=-1)
        controller_out = self.controller(controller_input)  # (B, D)

        # --- 8. Classify operator ---
        op_logits = self.op_classifier(controller_out)  # (B, 4)
        op_probs = ops.softmax(op_logits, axis=-1)  # (B, 4)

        # --- 9. Execute ALL fixed arithmetic, select by op_probs ---
        add_result, add_valid = _fixed_add(left_val, right_val)
        sub_result, sub_valid = _fixed_subtract(left_val, right_val)
        mul_result, mul_valid = _fixed_multiply(left_val, right_val)
        div_result, div_valid = _fixed_divide(
            left_val, right_val, epsilon=self.config.epsilon
        )

        # Stack results and validity: (B, 4, 1)
        all_results = ops.stack(
            [add_result, sub_result, mul_result, div_result], axis=1
        )
        all_valid = ops.stack(
            [add_valid, sub_valid, mul_valid, div_valid], axis=1
        )

        # Select arithmetic output:
        # - Training: soft-select (differentiable, gradients flow to op_classifier)
        # - Inference: hard-select (argmax, exact operation, no blending)
        if training is True:
            op_weights = ops.expand_dims(op_probs, axis=-1)  # (B, 4, 1)
            result = ops.sum(all_results * op_weights, axis=1)  # (B, 1)
            valid = ops.sum(all_valid * op_weights, axis=1)  # (B, 1)
        else:
            op_idx = ops.argmax(op_probs, axis=-1)  # (B,)
            op_one_hot = ops.one_hot(op_idx, 4)  # (B, 4)
            op_weights = ops.expand_dims(op_one_hot, axis=-1)  # (B, 4, 1)
            result = ops.sum(all_results * op_weights, axis=1)  # (B, 1)
            valid = ops.sum(all_valid * op_weights, axis=1)  # (B, 1)

        # --- 10. Write result to NTM memory ---
        # Log-compress the result before encoding into the internal pipeline.
        # Raw multiplication of large operands (e.g., 1e5 * 1e5 = 1e10) would
        # blow up Dense layers, memory, and state updates. Log-compression
        # maps any scale to a bounded range (~0-25) that the pipeline can handle.
        # The raw `result` is still returned as output for loss computation.
        result_compressed = ops.sign(result) * ops.log1p(ops.abs(result))
        result_embedding = self.result_encoder(
            ops.concatenate([result_compressed, valid], axis=-1)
        )  # (B, D)

        write_weights_new, write_state = self.write_head.compute_addressing(
            controller_out, memory_state, carry["write_weights"]
        )
        erase_vec = write_state.erase_vector
        # Combine the write head's learned add vector with the result embedding
        # so both the write head's add projection and the result encoder get gradients
        add_vec = write_state.add_vector + result_embedding
        memory_state = self.memory.write(
            memory_state, write_weights_new, erase_vec, add_vec
        )

        # --- 11. Post-write read: re-read from updated memory ---
        # This creates a gradient path through the write head:
        # write_head params → write weights/erase → memory.write →
        # updated memory → memory.read → post_read → state_update → loss
        post_read_vectors = []
        for i, head in enumerate(self.read_heads):
            rv = self.memory.read(memory_state, new_read_weights[i])
            post_read_vectors.append(rv)

        # --- 12. Update hidden state ---
        # Fuse controller output, result embedding, validity, and
        # post-write memory readback into the state update
        validity_emb = self.validity_encoder(valid)  # (B, D)
        update_input = ops.concatenate(
            [controller_out, result_embedding, validity_emb]
            + post_read_vectors,
            axis=-1,
        )
        state_delta = self.state_update(update_input)  # (B, D)
        state_delta_seq = ops.expand_dims(state_delta, axis=1)  # (B, 1, D)
        # Inject update weighted by reduction focus
        hidden = hidden + rw * state_delta_seq

        # --- 13. Halt decision ---
        halt_input = ops.sum(hidden * ops.expand_dims(token_mask_float, -1), axis=1)
        halt_input = halt_input / (ops.sum(token_mask_float, axis=-1, keepdims=True) + 1e-9)
        halt_logits = self.halt_head(halt_input)  # (B, 2)
        q_halt = halt_logits[..., 0]  # (B,)
        q_continue = halt_logits[..., 1]  # (B,)

        # --- Build new carry with stop_gradient ---
        new_carry = {
            "memory": ops.stop_gradient(memory_state.memory),
            "memory_usage": ops.stop_gradient(memory_state.usage),
            "read_weights": [ops.stop_gradient(w) for w in new_read_weights],
            "write_weights": ops.stop_gradient(write_weights_new),
            # DECISION plan-2026-08-17T183311-79c63e38/D-024 (second fp16
            # blocker, found by the mask fix's own RED proof and sitting AFTER
            # it in this same function). The deterministic number assembly above
            # pins itself to "float32" ON PURPOSE — exact digit arithmetic must
            # not run in fp16 — while Keras autocasts the incoming `carry` to
            # `compute_dtype`. Under `mixed_float16` the two met here and raised
            # `InvalidArgumentError: cannot compute AddV2 as input #1 was
            # expected to be a half tensor but is a float tensor`. Cast BOTH
            # sides to `compute_dtype` rather than casting only one: the carry is
            # float32 when this layer is called without autocast, so a one-sided
            # cast just moves the mismatch. `outputs["result"]` below keeps the
            # exact float32 value — only the accumulator is normalised.
            "accumulated_result": (
                ops.cast(carry["accumulated_result"], self.compute_dtype)
                + ops.cast(result, self.compute_dtype)
            ),
            "accumulated_valid": (
                ops.cast(carry["accumulated_valid"], self.compute_dtype)
                * ops.cast(valid, self.compute_dtype)
            ),
            "steps": carry["steps"] + 1,
        }

        outputs = {
            "result": result,
            "valid": valid,
            "op_logits": op_logits,
            "q_halt": q_halt,
            "q_continue": q_continue,
            "hidden": hidden,
            "break_prob": break_prob,
            "group_prob": group_prob,
            # Intermediate predictions for multi-task supervision
            "left_val": left_val,              # (B, 1) extracted left operand
            "right_val": right_val,            # (B, 1) extracted right operand
            "reduction_weights": reduction_weights,  # (B, L) sub-expression focus
        }

        return new_carry, outputs

    def compute_output_shape(self, input_shape: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Return the ``(new_carry, outputs)`` shape structure.

        Shapes are derived entirely from construction-time config (hidden_size
        ``D``, memory_size ``M``, max_expression_len ``L``, num_read_heads) and
        the batch axis ``B`` taken from the hidden-state input; they never
        depend on ``training`` (the soft/hard op-select branch produces the same
        ``result``/``valid`` shapes either way).

        :param input_shape: Tuple of (carry, hidden, mask, token_ids) shapes.
            ``B`` is read from the hidden-state shape ``(B, L, D)``.
        """
        h = self.config.hidden_size
        m = self.config.memory_size
        seq = self.config.max_expression_len
        # input_shape mirrors call() inputs: (carry, hidden, mask, token_ids)
        hidden_shape = input_shape[1]
        b = hidden_shape[0]

        new_carry = {
            "memory": (b, m, h),
            "memory_usage": (b, m),
            "read_weights": [(b, m) for _ in range(self.config.num_read_heads)],
            "write_weights": (b, m),
            "accumulated_result": (b, 1),
            "accumulated_valid": (b, 1),
            "steps": (b,),
        }
        outputs = {
            "result": (b, 1),
            "valid": (b, 1),
            "op_logits": (b, 4),
            "q_halt": (b,),
            "q_continue": (b,),
            "hidden": (b, seq, h),
            "break_prob": (b, seq, seq),
            "group_prob": (b, seq, seq),
            "left_val": (b, 1),
            "right_val": (b, 1),
            "reduction_weights": (b, seq),
        }
        return new_carry, outputs

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["config"] = self.config.to_dict()
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NAMCell":
        nam_config = config.pop("config")
        return cls(config=nam_config, **config)
