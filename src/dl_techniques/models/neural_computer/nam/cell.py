"""
NAMCell, one step of arithmetic expression reduction.

NAMCell is a Keras layer that takes a carry dict, a hidden state, a padding
mask and the raw token ids, and returns an updated carry plus a dict of
outputs. One call parses the expression with GroupAttention and TreeMHA,
scores which sub-expression to reduce, assembles the two operands from the
token ids at that position, classifies the operator, runs all four fixed
arithmetic units and selects one, writes the result into NTM memory, and
emits a halt decision. The arithmetic is fixed rather than learned; the
learned parts are the parse, the routing and the halting.

The cell handles one operator at a time and integers only. Operand assembly
reads only ``token_ids``, since the NTM read heads feed the controller
rather than the operands, so a multi-operator expression concatenates the
far side's digits instead of reducing them first (``"1 + 2 * 3"`` gives
operands ``(1, 23)`` at the ``+``), and ``DOT_ID`` falls outside
``is_digit``, so decimals lose their point (``"1.5 + 2"`` gives ``(15,
2)``). See the ``NAM`` module docstring.
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.memory.baseline_ntm import (
    NTMMemory,
    NTMReadHead,
    NTMWriteHead,
)
from dl_techniques.layers.memory.ntm_interface import (
    AddressingMode,
    MemoryState,
)
from dl_techniques.models.language.tree_transformer.model import (
    GroupAttention,
    TreeMHA,
)
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.ffn.factory import assemble_ffn_config

from dl_techniques.utils.dtype_policy import (
    accumulation_dtype,
    mask_sentinel,
    stability_floor,
)

from .config import NAMConfig
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Fixed arithmetic operations
# ---------------------------------------------------------------------

def _fixed_add(a: Any, b: Any) -> Tuple[Any, Any]:
    """Add, returning an all-ones validity flag."""
    return ops.add(a, b), ops.ones_like(a)


def _fixed_subtract(a: Any, b: Any) -> Tuple[Any, Any]:
    """Subtract, returning an all-ones validity flag."""
    return ops.subtract(a, b), ops.ones_like(a)


def _fixed_multiply(a: Any, b: Any) -> Tuple[Any, Any]:
    """Multiply, returning an all-ones validity flag."""
    return ops.multiply(a, b), ops.ones_like(a)


def _fixed_divide(a: Any, b: Any, epsilon: float = 1e-7) -> Tuple[Any, Any]:
    """
    Divide, flagging a near-zero denominator as invalid.

    :param a: Numerator.
    :param b: Denominator.
    :param epsilon: Threshold on ``abs(b)`` below which the division is
        invalid. ``NAMCell`` passes ``NAMConfig.epsilon`` here.
    :return: Tuple of (result, valid), both ``(B, 1)``. Where ``abs(b)`` is at
        or below ``epsilon`` the pair is ``(0.0, 0.0)``; the denominator is
        replaced by 1 before the divide so no infinity is produced.
    """
    abs_b = ops.abs(b)
    valid = ops.cast(ops.greater(abs_b, epsilon), a.dtype)
    safe_b = ops.where(ops.greater(abs_b, epsilon), b, ops.ones_like(b))
    result = ops.divide(a, safe_b)
    result = result * valid
    return result, valid

# ---------------------------------------------------------------------
# Deterministic number assembly
# ---------------------------------------------------------------------

def _assemble_number_from_tokens(
    token_ids: Any,
    digit_mask: Any,
) -> Any:
    """Assemble a multi-digit number from token digit values.

    The tokenizer encodes digits 0-9 as token ids 4-13, so a digit's value is
    its id minus 4. Each masked digit is weighted by its place value, taken
    from the number of masked digits at or to the right of it:

        value = sum(digit_value_i * 10^position_i)

    The function has no learned parameters. It re-derives digit values from
    the ids, so a mask position holding a non-digit token contributes nothing.
    An all-zero mask gives 0.0, which is indistinguishable from a literal 0.

    Place value:

    .. code-block:: text

        digit_mask        0    0    1    1
        cumsum_left       0    0    1    2
        total_digits      2
        power_of_10       0    0    1    0
        weight            .    .   10    1

        value = 2 * 10 + 3 * 1 = 23

    :param token_ids: (B, L) int — raw token IDs from the tokenizer.
    :param digit_mask: (B, L) float — 1.0 for each digit belonging to
        this number, 0.0 elsewhere. Computed from is_digit AND side-of-operator.
    :return: (B, 1) float32 — the assembled scalar number value. The dtype is
        float32 whatever the layer's compute dtype is.
    """
    is_digit = ops.cast(
        ops.logical_and(
            ops.greater_equal(token_ids, 4),
            ops.less_equal(token_ids, 13),
        ),
        "float32",
    )
    digit_values = ops.cast(token_ids - 4, "float32") * is_digit

    # The rightmost masked digit has all of total_digits behind it, so power 0.
    cumsum_left = ops.cumsum(digit_mask, axis=1)
    total_digits = ops.sum(digit_mask, axis=1, keepdims=True)
    power_of_10 = (total_digits - cumsum_left) * digit_mask

    positional_weight = ops.power(
        ops.cast(10.0, "float32"), power_of_10
    ) * digit_mask

    value = ops.sum(digit_values * positional_weight, axis=1, keepdims=True)
    return value

# ---------------------------------------------------------------------
# NAMCell
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.nam.cell")
class NAMCell(keras.layers.Layer):
    """
    Reduce one sub-expression of an arithmetic expression.

    The cell combines a tree transformer (GroupAttention plus TreeMHA) for
    structural parsing, NTM memory as the controller's recurrent context, four
    fixed arithmetic units with validity flags, and a halting head for adaptive
    computation time. The memory supplies context, not operands: those come
    from the raw tokens.

    Architecture:

    .. code-block:: text

        hidden [B, L, D], mask [B, 1, L], token_ids [B, L]
              │
              ▼
        ┌──────────────────────────────┐
        │ group attention              ├──► group_prob, break_prob
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ tree mha, pre-ln, residual   │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ ffn, pre-ln, residual        │
        └──────────────────────────────┘
              │ [B, L, D]
              ▼
        ┌──────────────────────────────┐
        │ reduction scorer             │
        │ mask sentinel, then softmax  │
        └──────────────────────────────┘
              │ reduction_weights [B, L]
              ├──► argmax op_pos ──► operand path ──► left, right
              ▼
        ┌──────────────────────────────┐
        │ pooled = sum(hidden * rw)    │
        └──────────────────────────────┘
              │ [B, D]
              ▼
        ┌──────────────────────────────┐
        │ read heads, before the write │
        └──────────────────────────────┘
              │ pooled and read vectors
              ▼
        ┌──────────────────────────────┐
        │ controller, relu             │
        └──────────────────────────────┘
              │ [B, D]
              ▼
        ┌──────────────────────────────┐
        │ op classifier                │
        └──────────────────────────────┘
              │ op_probs [B, 4]
              ▼
        ┌──────────────────────────────┐
        │ add, sub, mul, div           │◄── left, right
        │ select by op_probs           │
        └──────────────────────────────┘
              │ result, valid [B, 1]
              ▼
        ┌──────────────────────────────┐
        │ log compress, result encoder │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ memory write, then re-read   │  addressed from controller
        └──────────────────────────────┘
              │ post-write read vectors
              ▼
        ┌──────────────────────────────┐
        │ state update                 │
        └──────────────────────────────┘
              │ delta [B, D]
              ▼
        hidden = hidden + rw * delta
              │
              ▼
        ┌──────────────────────────────┐
        │ halt head on the masked mean │
        └──────────────────────────────┘
              │
              ▼
        q_halt, q_continue [B]

    Operand path:

    .. code-block:: text

        expression        1    +    2    3       op_pos = 1
        is_digit          1    0    1    1
        left mask         1    0    0    0   ──► left  = 1
        right mask        0    0    1    1   ──► right = 23

    The token at ``op_pos`` is in neither mask, so a digit landing there is
    dropped.

    Memory path:

    .. code-block:: text

        pooled ──► read head addressing ──► read weights
              │                                  │
              ▼                                  ▼
        controller(pooled, pre-write reads)   memory.read
              │
              ├──► write head ──► erase, add vectors
              │                        │
              │    add vector + result embedding
              ▼                        ▼
        memory.write ──► memory.read with the same read weights
                                       │
                                       ▼
                                post-write reads ──► state update

    Reading again after the write puts the write head's parameters on a
    gradient path to the loss.

    Operator selection:

    .. code-block:: text

        training is True   soft: sum(all_results * op_probs)
        anything else      hard: sum(all_results * one_hot(argmax))

    The test is identity against ``True``, so ``None`` and other truthy values
    take the hard branch.

    Note:
        Operand assembly and the fixed arithmetic run in float32 whatever the
        compute dtype is, and the carry entries are cast back to the compute
        dtype where they are combined. Every carry entry passes through
        ``stop_gradient``, so no gradient crosses between steps.

    :param config: NAM configuration. A plain dict is accepted and passed
        through ``NAMConfig.from_dict``.
    :type config: NAMConfig or dict
    :param kwargs: Additional arguments for the Layer base class.
    """

    def __init__(self, config: NAMConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if isinstance(config, dict):
            config = NAMConfig.from_dict(config)
        self.config = config
        h = config.hidden_size

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
        # DECISION plan-2026-07-30T140922-8af1028f/D-022: route through
        # assemble_ffn_config; a raw dict passes keys the ffn_type rejects. See
        # decisions.md.
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

        self.reduction_scorer = keras.layers.Dense(1, name="reduction_scorer")

        # One unit per operator, in the order add, sub, mul, div.
        self.op_classifier = keras.layers.Dense(4, name="op_classifier")

        # NTMMemory takes no epsilon; NAMConfig.epsilon reaches _fixed_divide.
        self.memory = NTMMemory(
            memory_size=config.memory_size,
            memory_dim=h,
            name="ntm_memory",
        )
        # DECISION plan-2026-08-17T183311-79c63e38/D-014: keep CONTENT addressing;
        # shift_range or HYBRID changes a shipped model's memory semantics. See
        # decisions.md.
        self.read_heads = [
            NTMReadHead(
                memory_size=config.memory_size,
                memory_dim=h,
                addressing_mode=AddressingMode.CONTENT,
                name=f"read_head_{i}",
            )
            for i in range(config.num_read_heads)
        ]
        # DECISION plan-2026-08-18T140459-7991552f/D-035: exactly one write head, as
        # every shipped checkpoint has. See decisions.md.
        self.write_head = NTMWriteHead(
            memory_size=config.memory_size,
            memory_dim=h,
            addressing_mode=AddressingMode.CONTENT,
            name="write_head",
        )

        self.controller = keras.layers.Dense(h, activation="relu", name="controller")

        self.result_encoder = keras.layers.Dense(h, name="result_encoder")

        self.validity_encoder = keras.layers.Dense(h, name="validity_encoder")

        self.state_update = keras.layers.Dense(h, name="state_update")

        self.halt_head = keras.layers.Dense(2, name="halt_head")

    def build(self, input_shape: Any) -> None:
        """Build every sub-layer from the config's fixed shapes.

        :param input_shape: Unused. All sub-layer shapes come from
            ``hidden_size``, ``memory_size``, ``max_expression_len`` and
            ``num_read_heads``, so the argument only reaches ``super().build``.
        :type input_shape: Any
        """
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
        self.op_classifier.build(seq_shape)

        controller_input_dim = h + self.config.num_read_heads * h
        controller_shape = (None, controller_input_dim)
        self.controller.build(controller_shape)

        for head in self.read_heads:
            head.build((None, h))
        self.write_head.build((None, h))

        # The result encoder reads the compressed result and its validity flag.
        self.result_encoder.build((None, 2))
        self.validity_encoder.build((None, 1))
        # controller(D) + result_emb(D) + validity(D) + post-write reads(n * D).
        state_update_dim = h * (3 + self.config.num_read_heads)
        self.state_update.build((None, state_update_dim))

        self.halt_head.build((None, h))

        super().build(input_shape)

    def initialize_carry(self, batch_size: int) -> Dict[str, Any]:
        """
        Create initial carry state for a new expression.

        Read and write weights start uniform over the ``memory_size`` slots, the
        accumulated result at 0 and the accumulated validity at 1, so the first
        step's multiply leaves it at that step's own flag.

        :param batch_size: Batch size.
        :type batch_size: int
        :return: Initial carry dictionary, with the same keys ``call`` returns.
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
        :param training: Whether in training mode. Only ``True`` itself selects
            the soft arithmetic branch; it also reaches the attention, FFN and
            dropout sub-layers.
        :type training: Optional[bool]
        :return: Tuple of (new_carry, outputs).
            - new_carry: 'memory', 'memory_usage', 'read_weights',
              'write_weights', 'accumulated_result', 'accumulated_valid' and
              'steps', each through stop_gradient except 'steps'.
            - outputs: Dict with 'result', 'valid', 'op_logits', 'q_halt',
              'q_continue', 'hidden', 'break_prob', 'group_prob', 'left_val',
              'right_val' and 'reduction_weights'. 'result' is the raw value,
              not the log-compressed one the pipeline uses.
        :rtype: Tuple[Dict[str, Any], Dict[str, Any]]
        """
        carry, hidden, mask, token_ids = inputs
        h = self.config.hidden_size

        group_prob, break_prob = self.group_attention(
            (hidden, mask, ops.convert_to_tensor(0.0, dtype=self.compute_dtype)),
            training=training,
        )

        hidden_norm = self.attn_norm(hidden)
        attn_out = self.tree_mha(
            (hidden_norm, hidden_norm, hidden_norm, group_prob, mask),
            training=training,
        )
        hidden = hidden + self.attn_dropout(attn_out, training=training)

        hidden_norm2 = self.ffn_norm(hidden)
        ffn_out = self.ffn(hidden_norm2, training=training)
        hidden = hidden + self.ffn_dropout(ffn_out, training=training)

        token_mask = ops.squeeze(mask, axis=1)
        token_mask_float = ops.cast(token_mask, self.compute_dtype)

        scores = ops.squeeze(self.reduction_scorer(hidden), axis=-1)
        # DECISION plan-2026-08-17T183311-79c63e38/D-024: a dtype-aware sentinel, not
        # a bare -1e9, which underflows to -inf under mixed_float16.
        # DECISION plan-2026-08-31T134711-6271592d/D-007: select the sentinel with
        # where(), do not add it. See decisions.md.
        keep = ops.cast(token_mask, "bool")
        sentinel = ops.cast(mask_sentinel(self.compute_dtype), scores.dtype)
        scores = ops.where(keep, scores, sentinel)
        reduction_weights = ops.softmax(scores, axis=-1)

        # DECISION plan-2026-08-18T140459-7991552f/D-055: this split is the whole
        # numeric path, single-operator and integer-only. See decisions.md.
        op_pos = ops.argmax(reduction_weights, axis=-1)
        seq_len = ops.shape(token_ids)[1]
        positions = ops.cast(ops.arange(seq_len), "int32")
        op_pos_expanded = ops.expand_dims(ops.cast(op_pos, "int32"), axis=-1)

        is_digit = ops.cast(
            ops.logical_and(
                ops.greater_equal(token_ids, 4),
                ops.less_equal(token_ids, 13),
            ),
            "float32",
        )
        # Strict comparisons, so the operator position joins neither side.
        left_digit_mask = is_digit * ops.cast(
            ops.less(positions, op_pos_expanded), "float32"
        )
        right_digit_mask = is_digit * ops.cast(
            ops.greater(positions, op_pos_expanded), "float32"
        )

        left_val = _assemble_number_from_tokens(token_ids, left_digit_mask)
        right_val = _assemble_number_from_tokens(token_ids, right_digit_mask)

        # The weights still serve as the pooling kernel for the controller path.
        rw = ops.expand_dims(reduction_weights, axis=-1)

        memory_state = MemoryState(
            memory=carry["memory"],
            usage=carry["memory_usage"],
        )

        pooled = ops.sum(hidden * rw, axis=1)
        pre_read_vectors = []
        new_read_weights = []
        for i, head in enumerate(self.read_heads):
            weights, _ = head.compute_addressing(
                pooled, memory_state, carry["read_weights"][i]
            )
            rv = self.memory.read(memory_state, weights)
            pre_read_vectors.append(rv)
            new_read_weights.append(weights)

        controller_input = ops.concatenate([pooled] + pre_read_vectors, axis=-1)
        controller_out = self.controller(controller_input)

        op_logits = self.op_classifier(controller_out)
        op_probs = ops.softmax(op_logits, axis=-1)

        add_result, add_valid = _fixed_add(left_val, right_val)
        sub_result, sub_valid = _fixed_subtract(left_val, right_val)
        mul_result, mul_valid = _fixed_multiply(left_val, right_val)
        div_result, div_valid = _fixed_divide(
            left_val, right_val, epsilon=self.config.epsilon
        )

        all_results = ops.stack(
            [add_result, sub_result, mul_result, div_result], axis=1
        )
        all_valid = ops.stack(
            [add_valid, sub_valid, mul_valid, div_valid], axis=1
        )

        # The soft branch is what carries gradient to the op classifier; the hard
        # branch keeps inference to one exact operation.
        if training is True:
            op_weights = ops.expand_dims(op_probs, axis=-1)
            result = ops.sum(all_results * op_weights, axis=1)
            valid = ops.sum(all_valid * op_weights, axis=1)
        else:
            op_idx = ops.argmax(op_probs, axis=-1)
            op_one_hot = ops.one_hot(op_idx, 4)
            op_weights = ops.expand_dims(op_one_hot, axis=-1)
            result = ops.sum(all_results * op_weights, axis=1)
            valid = ops.sum(all_valid * op_weights, axis=1)

        # Products of large operands (1e5 * 1e5) would saturate the Dense layers
        # and the memory, so the pipeline sees a log-compressed copy.
        result_compressed = ops.sign(result) * ops.log1p(ops.abs(result))
        result_embedding = self.result_encoder(
            ops.concatenate([result_compressed, valid], axis=-1)
        )

        write_weights_new, write_state = self.write_head.compute_addressing(
            controller_out, memory_state, carry["write_weights"]
        )
        erase_vec = write_state.erase_vector
        # Summing the two keeps both the add projection and the result encoder on
        # a gradient path.
        add_vec = write_state.add_vector + result_embedding
        memory_state = self.memory.write(
            memory_state, write_weights_new, erase_vec, add_vec
        )

        # Re-reading the written memory is what connects the write head's
        # parameters to the loss.
        post_read_vectors = []
        for i, head in enumerate(self.read_heads):
            rv = self.memory.read(memory_state, new_read_weights[i])
            post_read_vectors.append(rv)

        validity_emb = self.validity_encoder(valid)
        update_input = ops.concatenate(
            [controller_out, result_embedding, validity_emb]
            + post_read_vectors,
            axis=-1,
        )
        state_delta = self.state_update(update_input)
        state_delta_seq = ops.expand_dims(state_delta, axis=1)
        # Weighting by rw confines the update to the reduced position.
        hidden = hidden + rw * state_delta_seq

        halt_input = ops.sum(hidden * ops.expand_dims(token_mask_float, -1), axis=1)
        # DECISION plan-2026-08-31T134711-6271592d/D-016: use stability_floor, which
        # resolves bfloat16 through ml_dtypes; np.finfo("bfloat16") raises on numpy
        # 2.0.2. See decisions.md.
        halt_accum = accumulation_dtype(self.compute_dtype)
        halt_denominator = ops.cast(
            ops.sum(token_mask_float, axis=-1, keepdims=True), halt_accum
        ) + stability_floor(halt_accum, 1e-9)
        halt_input = ops.cast(
            ops.cast(halt_input, halt_accum) / halt_denominator,
            self.compute_dtype,
        )
        halt_logits = self.halt_head(halt_input)
        q_halt = halt_logits[..., 0]
        q_continue = halt_logits[..., 1]

        new_carry = {
            "memory": ops.stop_gradient(memory_state.memory),
            "memory_usage": ops.stop_gradient(memory_state.usage),
            "read_weights": [ops.stop_gradient(w) for w in new_read_weights],
            "write_weights": ops.stop_gradient(write_weights_new),
            # DECISION plan-2026-08-17T183311-79c63e38/D-024: cast both sides, not
            # one, or mixed_float16 raises on the add. See decisions.md.
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
            # The operands and the focus feed multi-task supervision.
            "left_val": left_val,
            "right_val": right_val,
            "reduction_weights": reduction_weights,
        }

        return new_carry, outputs

    def compute_output_shape(self, input_shape: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Return the ``(new_carry, outputs)`` shape structure.

        Shapes are derived entirely from construction-time config (hidden_size
        ``D``, memory_size ``M``, max_expression_len ``L``, num_read_heads) and
        the batch axis ``B`` taken from the hidden-state input; they never
        depend on ``training``, since the soft and hard op-select branches
        produce the same ``result`` and ``valid`` shapes.

        :param input_shape: Tuple of (carry, hidden, mask, token_ids) shapes,
            mirroring ``call``. ``B`` is read from the hidden-state shape
            ``(B, L, D)``.
        :type input_shape: Any
        :return: Tuple of (new_carry shapes, outputs shapes).
        :rtype: Tuple[Dict[str, Any], Dict[str, Any]]
        """
        h = self.config.hidden_size
        m = self.config.memory_size
        seq = self.config.max_expression_len
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
        """Return the layer configuration, with ``NAMConfig`` as a dict.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config["config"] = self.config.to_dict()
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NAMCell":
        """Rebuild the cell, letting ``__init__`` revive the config dict.

        :param config: Configuration dictionary produced by ``get_config``.
        :type config: Dict[str, Any]
        :return: Reconstructed cell instance.
        :rtype: NAMCell
        """
        nam_config = config.pop("config")
        return cls(config=nam_config, **config)

# ---------------------------------------------------------------------
