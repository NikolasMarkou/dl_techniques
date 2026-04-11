"""
Training script for the Differentiable Finite-State Arithmetic (DFSA) module.

A minimal, fully-differentiable neural network that imitates a finite-state
automaton for arithmetic evaluation. Designed to be frozen and embedded
into larger transformer models.

Architecture:
    token_ids → token_features(small MLP) → reduction_scorer(softmax)
    → soft left/right masks(cumsum) → differentiable number assembly
    → soft operator select → fixed arithmetic → result

Total learned parameters: ~100K (vs 5.6M in the full NAM).
Gradient flows end-to-end from the result back to token features.

Usage::

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m train.nam.train_dfsa \\
        --steps 5000 --batch-size 64 --lr 1e-4 --gpu 1

"""

import sys
import json
import argparse
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from keras import ops

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dl_techniques.models.nam.tokenizer import ArithmeticTokenizer
from dl_techniques.models.nam.cell import (
    _fixed_add,
    _fixed_subtract,
    _fixed_multiply,
    _fixed_divide,
)
from dl_techniques.models.tree_transformer.model import (
    TreeTransformerBlock,
    PositionalEncoding,
)
from dl_techniques.optimization.warmup_schedule import WarmupSchedule
from dl_techniques.utils.logger import logger
from train.common import setup_gpu
from train.nam.data_generator import (
    generate_curriculum_batch,
    prepare_per_step_labels,
    _safe_eval,
    DIFFICULTY_LEVELS,
    _curriculum_probs,
)


# ── Differentiable Finite-State Arithmetic Module ─────────────────────


RESULT_PLACEHOLDER_ID = 21  # Token ID for re-tokenized computed values


@keras.saving.register_keras_serializable()
class DifferentiableFSA(keras.Model):
    """
    Differentiable Finite-State Arithmetic evaluator.

    Imitates a finite-state transducer for arithmetic evaluation:
    1. Token features via small MLP (learned)
    2. Operator position via softmax (learned, differentiable)
    3. Left/right digit masks via cumulative softmax (differentiable)
    4. Number assembly via positional power-of-10 (differentiable)
    5. Operator classification via softmax (learned, differentiable)
    6. Fixed arithmetic with soft-select (differentiable)

    Gradient flows from the output result all the way back to the
    token features. When frozen, behaves as an exact FSA.

    :param hidden_size: Internal feature dimension.
    :param max_expression_len: Maximum token sequence length.
    :param vocab_size: Tokenizer vocabulary size.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        max_expression_len: int = 64,
        vocab_size: int = 22,  # 21 base + RESULT_PLACEHOLDER
        num_tree_layers: int = 2,
        num_heads: int = 4,
        intermediate_size: int = None,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.max_expression_len = max_expression_len
        self.vocab_size = vocab_size
        self.num_tree_layers = num_tree_layers

        if intermediate_size is None:
            intermediate_size = hidden_size * 2

        # --- Token features ---
        self.token_embedding = keras.layers.Embedding(
            vocab_size, hidden_size, name="token_embedding"
        )
        self.numeric_proj = keras.layers.Dense(
            hidden_size, name="numeric_proj"
        )
        self.pos_encoding = PositionalEncoding(
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            max_len=max_expression_len,
            name="pos_encoding",
        )

        # --- Shared tree transformer (CKY constituency parsing) ---
        # GroupAttention learns which adjacent tokens form sub-expressions
        # (e.g., "5 * 2" groups before "3 + ..."). This is the PEMDAS
        # inductive bias — tree structure = operator precedence.
        # Shared across ACT steps: same weights parse the expression at
        # each reduction level.
        self.tree_blocks = [
            TreeTransformerBlock(
                num_heads=num_heads,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                attention_dropout_rate=dropout_rate,
                hidden_dropout_rate=dropout_rate,
                name=f"tree_block_{i}",
            )
            for i in range(num_tree_layers)
        ]
        self.encoder_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="encoder_norm"
        )

        # --- Reduction scorer with skip connection ---
        # After tree encoding, op_type features get diluted by attention
        # mixing (all positions become weighted averages). The skip
        # connection feeds raw numeric features (op_type, is_operator)
        # DIRECTLY to the scorer so it can always see "this is a * not a +".
        # Input: tree-encoded features (D) + raw numeric features (4)
        self.reduction_scorer = keras.layers.Dense(
            1, name="reduction_scorer"
        )  # built on (D + 4) input

        # --- Operator classifier ---
        self.op_classifier = keras.layers.Dense(
            4, name="op_classifier"
        )

        # --- Result encoder (interface to host transformer) ---
        self.result_encoder = keras.layers.Dense(
            hidden_size, name="result_encoder"
        )

    def reduce_step(self, token_ids, training=None):
        """
        Perform ONE reduction step on the given token sequence.

        This is the core FSA step: find operator → extract operands →
        classify operator → compute result. Fully differentiable.

        :param token_ids: (B, L) int — tokenized expression at this step.
        :param training: Whether in training mode.
        :return: Dict with 'result', 'valid', 'left_val', 'right_val',
            'op_logits', 'reduction_weights'.
        """
        import math

        # Padding mask
        mask = ops.cast(ops.not_equal(token_ids, 0), "float32")  # (B, L)

        # Digit and operator features (deterministic from token IDs)
        ids_float = ops.cast(token_ids, "float32")
        is_digit = ops.cast(
            ops.logical_and(
                ops.greater_equal(token_ids, 4),
                ops.less_equal(token_ids, 13),
            ),
            "float32",
        )
        digit_values = (ids_float - 4.0) * is_digit  # 0-9

        op_type = ops.zeros_like(ids_float)
        op_type = ops.where(ops.equal(token_ids, 14), 1.0 * ops.ones_like(op_type), op_type)
        op_type = ops.where(ops.equal(token_ids, 15), 2.0 * ops.ones_like(op_type), op_type)
        op_type = ops.where(ops.equal(token_ids, 16), 3.0 * ops.ones_like(op_type), op_type)
        op_type = ops.where(ops.equal(token_ids, 17), 4.0 * ops.ones_like(op_type), op_type)

        is_operator = ops.cast(
            ops.logical_and(
                ops.greater_equal(token_ids, 14),
                ops.less_equal(token_ids, 17),
            ),
            "float32",
        )

        numeric_features = ops.stack(
            [digit_values, is_digit, op_type, is_operator], axis=-1
        )

        # Token features: embedding + numeric injection + positional encoding
        x = self.token_embedding(token_ids) * math.sqrt(self.hidden_size)
        x = x + self.numeric_proj(numeric_features)
        x = self.pos_encoding(x, training=training)

        # Tree transformer: CKY constituency parsing for PEMDAS
        # GroupAttention learns tree structure — g_attn[i][j] = probability
        # that tokens i and j belong to the same constituent.
        # "5 * 2" forms a tighter group than "3 + result" because PEMDAS
        # is constituency: higher-precedence ops bind their operands first.
        mask_3d = ops.expand_dims(mask, axis=1)  # (B, 1, L)
        group_prob = ops.convert_to_tensor(0.0, dtype=x.dtype)
        for block in self.tree_blocks:
            x, group_prob, _ = block(
                (x, mask_3d, group_prob), training=training
            )
        x = self.encoder_norm(x)
        # group_prob: (B, L, L) — constituency matrix from CKY tree induction

        # --- Reduction from parse tree ---
        # Extract "group tightness" at each position from the constituency
        # matrix: how strongly does each token group with its neighbors?
        # Operators that bind tighter (PEMDAS: * > +) will have higher
        # group_prob with their adjacent operands.
        #
        # --- Reduction: which operator to reduce? ---
        # PEMDAS precedence is encoded directly from token types:
        # * (tid=16) and / (tid=17) have HIGHER precedence than
        # + (tid=14) and - (tid=15). Same precedence: leftmost first.
        #
        # This is HARDCODED — the tree transformer provides structural
        # context for the op_classifier and soft masks, but reduction
        # order follows the deterministic PEMDAS rule.
        #
        # The learned scorer provides a REFINEMENT on top of the
        # hardcoded precedence (to allow learning custom precedence
        # rules when fine-tuned).
        high_prec = ops.cast(  # * and /
            ops.logical_or(ops.equal(token_ids, 16), ops.equal(token_ids, 17)),
            "float32",
        )
        low_prec = ops.cast(  # + and -
            ops.logical_or(ops.equal(token_ids, 14), ops.equal(token_ids, 15)),
            "float32",
        )
        # Precedence scores: * and / get 20.0, + and - get 15.0
        # Both must be high enough for sharp softmax (>0.999 operator
        # probability) so that cumsum-based soft masks produce accurate
        # number assembly. Score=5 only gives ~0.955 → 4.5% mask leakage
        # → 33% assembly error at 5 digits.  Score=15+ gives >0.9999.
        pemdas_scores = high_prec * 20.0 + low_prec * 15.0

        # Leftmost tie-breaking: subtract small position-dependent value
        seq_positions = ops.cast(ops.arange(ops.shape(token_ids)[1]), "float32")
        position_tiebreak = -0.01 * seq_positions  # prefer leftmost

        # Pure PEMDAS reduction — no learned component.
        # The tree encoder provides context for op_classifier and soft masks
        # but reduction order is deterministic from token types.
        scores = pemdas_scores + position_tiebreak
        scores = scores + (1.0 - mask) * (-1e9)  # mask padding
        reduction_weights = ops.softmax(scores, axis=-1)  # (B, L)

        # Soft left/right masks
        cum_op_prob = ops.cumsum(reduction_weights, axis=1)
        cum_before = cum_op_prob - reduction_weights
        left_mask = (1.0 - cum_op_prob) * is_digit
        right_mask = cum_before * is_digit

        # Differentiable number assembly
        left_val = self._soft_assemble(token_ids, left_mask, is_digit, digit_values)
        right_val = self._soft_assemble(token_ids, right_mask, is_digit, digit_values)

        # Operator classification
        pooled = ops.sum(x * ops.expand_dims(reduction_weights, -1), axis=1)
        op_logits = self.op_classifier(pooled)
        op_probs = ops.softmax(op_logits, axis=-1)

        # Fixed arithmetic with soft/hard select
        add_result, add_valid = _fixed_add(left_val, right_val)
        sub_result, sub_valid = _fixed_subtract(left_val, right_val)
        mul_result, mul_valid = _fixed_multiply(left_val, right_val)
        div_result, div_valid = _fixed_divide(left_val, right_val)

        all_results = ops.stack(
            [add_result, sub_result, mul_result, div_result], axis=1
        )
        all_valid = ops.stack(
            [add_valid, sub_valid, mul_valid, div_valid], axis=1
        )

        if training:
            op_weights = ops.expand_dims(op_probs, axis=-1)
            result = ops.sum(all_results * op_weights, axis=1)
            valid = ops.sum(all_valid * op_weights, axis=1)
        else:
            op_idx = ops.argmax(op_probs, axis=-1)
            op_one_hot = ops.one_hot(op_idx, 4)
            op_weights = ops.expand_dims(op_one_hot, axis=-1)
            result = ops.sum(all_results * op_weights, axis=1)
            valid = ops.sum(all_valid * op_weights, axis=1)

        return {
            "result": result,
            "valid": valid,
            "left_val": left_val,
            "right_val": right_val,
            "op_logits": op_logits,
            "reduction_weights": reduction_weights,
        }

    def call(self, token_ids, training=None):
        """
        Single-step evaluation (backward compatible).

        For multi-step evaluation, use ``reduce_step`` in a loop
        with updated token_ids per step.

        :param token_ids: (B, L) int — tokenized expression.
        :param training: Whether in training mode.
        :return: Dict with 'result', 'result_embedding', 'valid', etc.
        """
        out = self.reduce_step(token_ids, training=training)

        # Result encoder (learned interface)
        result_compressed = ops.sign(out["result"]) * ops.log1p(ops.abs(out["result"]))
        result_embedding = self.result_encoder(
            ops.concatenate([result_compressed, out["valid"]], axis=-1)
        )
        out["result_embedding"] = result_embedding
        return out

    @staticmethod
    def _soft_assemble(token_ids, soft_mask, is_digit, digit_values):
        """
        Differentiable number assembly from soft digit mask.

        When soft_mask is peaked (trained), this converges to exact
        integer assembly. But gradients flow through at all times via
        cumsum + power + sum.

        :param token_ids: (B, L) int
        :param soft_mask: (B, L) float — soft assignment to this number
        :param is_digit: (B, L) float — 1.0 for digit tokens
        :param digit_values: (B, L) float — 0-9 per digit token
        :return: (B, 1) float — assembled number value
        """
        # cumsum gives running count → position-in-number for power-of-10
        cumsum_left = ops.cumsum(soft_mask, axis=1)
        total_digits = ops.sum(soft_mask, axis=1, keepdims=True)  # (B, 1)
        power_of_10 = (total_digits - cumsum_left) * soft_mask  # (B, L)

        # 10^power is differentiable (d/dx 10^x = 10^x * ln(10))
        positional_weight = ops.power(10.0, power_of_10) * soft_mask

        # Assemble: sum(digit_value * positional_weight)
        value = ops.sum(
            digit_values * positional_weight, axis=1, keepdims=True
        )
        return value  # (B, 1)

    def multi_reduce(self, token_ids, max_steps=3, training=None):
        """
        Recursive reduction: reduce one op per step, re-tokenize, repeat.

        Each step selects the highest-precedence operator, extracts its
        ADJACENT operands via op_cumsum segmentation, computes the result,
        then replaces ``<left> <op> <right>`` with a RESULT_PLACEHOLDER
        token whose float value is stored in a gradient-carrying buffer.

        :param token_ids: (B, L) int — tokenized expression.
        :param max_steps: Maximum reduction steps (= max operators).
        :param training: Whether in training mode.
        :return: List of per-step output dicts.
        """
        B = ops.shape(token_ids)[0]
        L = self.max_expression_len

        current_ids = token_ids
        value_buffer = ops.zeros((B, L))
        buffer_active = ops.zeros((B, L))

        all_outputs = []
        # Track the last valid result for pass-through when no ops remain
        last_valid_result = ops.zeros((B, 1))

        for step in range(max_steps):
            # Check if operators remain in the current token sequence
            is_op_check = ops.cast(
                ops.logical_and(
                    ops.greater_equal(current_ids, 14),
                    ops.less_equal(current_ids, 17),
                ),
                "float32",
            )
            has_ops = ops.cast(
                ops.greater(
                    ops.sum(is_op_check, axis=-1, keepdims=True), 0.5
                ),
                "float32",
            )  # (B, 1) — 1.0 if operators remain

            out = self.reduce_step(
                current_ids, training=training,
                value_buffer=value_buffer,
                buffer_active=buffer_active,
            )

            # If no operators remain, pass through the prior result
            # This makes reduce_step idempotent on fully-reduced expressions
            out["result"] = has_ops * out["result"] + (
                1.0 - has_ops
            ) * last_valid_result
            last_valid_result = out["result"]

            all_outputs.append(out)

            # Re-tokenize for next step (only meaningful when ops remain)
            current_ids, value_buffer, buffer_active = self._retokenize(
                current_ids, value_buffer, buffer_active,
                out["op_pos_hard"],
                out["left_mask"], out["right_mask"],
                out["result"],
            )

        return all_outputs

    def reduce_step(self, token_ids, training=None,
                    value_buffer=None, buffer_active=None):
        """
        Perform ONE reduction step on the given token sequence.

        Uses op_cumsum segmentation for adjacent operand masking (exact
        for multi-op) and value_buffer for gradient-carrying result
        injection from prior steps.

        :param token_ids: (B, L) int — tokenized expression at this step.
        :param training: Whether in training mode.
        :param value_buffer: (B, L) float — prior step results at
            RESULT_PLACEHOLDER positions. None for first step.
        :param buffer_active: (B, L) float — 1.0 where value_buffer
            is valid. None for first step.
        :return: Dict with 'result', 'valid', 'left_val', 'right_val',
            'op_logits', 'reduction_weights', 'op_pos_hard',
            'left_mask', 'right_mask'.
        """
        import math

        B = ops.shape(token_ids)[0]
        L = self.max_expression_len

        if value_buffer is None:
            value_buffer = ops.zeros((B, L))
        if buffer_active is None:
            buffer_active = ops.zeros((B, L))

        # Padding mask
        mask = ops.cast(ops.not_equal(token_ids, 0), "float32")  # (B, L)

        # Digit and operator features (deterministic from token IDs)
        ids_float = ops.cast(token_ids, "float32")
        is_digit = ops.cast(
            ops.logical_and(
                ops.greater_equal(token_ids, 4),
                ops.less_equal(token_ids, 13),
            ),
            "float32",
        )
        digit_values = (ids_float - 4.0) * is_digit  # 0-9

        is_operator = ops.cast(
            ops.logical_and(
                ops.greater_equal(token_ids, 14),
                ops.less_equal(token_ids, 17),
            ),
            "float32",
        )

        op_type = ops.zeros_like(ids_float)
        op_type = ops.where(ops.equal(token_ids, 14), 1.0 * ops.ones_like(op_type), op_type)
        op_type = ops.where(ops.equal(token_ids, 15), 2.0 * ops.ones_like(op_type), op_type)
        op_type = ops.where(ops.equal(token_ids, 16), 3.0 * ops.ones_like(op_type), op_type)
        op_type = ops.where(ops.equal(token_ids, 17), 4.0 * ops.ones_like(op_type), op_type)

        numeric_features = ops.stack(
            [digit_values, is_digit, op_type, is_operator], axis=-1
        )

        # Token features: embedding + numeric injection + positional encoding
        x = self.token_embedding(token_ids) * math.sqrt(self.hidden_size)
        x = x + self.numeric_proj(numeric_features)
        x = self.pos_encoding(x, training=training)

        # Tree transformer
        mask_3d = ops.expand_dims(mask, axis=1)  # (B, 1, L)
        group_prob = ops.convert_to_tensor(0.0, dtype=x.dtype)
        for block in self.tree_blocks:
            x, group_prob, _ = block(
                (x, mask_3d, group_prob), training=training
            )
        x = self.encoder_norm(x)

        # --- PEMDAS reduction scoring (hardcoded) ---
        high_prec = ops.cast(
            ops.logical_or(ops.equal(token_ids, 16), ops.equal(token_ids, 17)),
            "float32",
        )
        low_prec = ops.cast(
            ops.logical_or(ops.equal(token_ids, 14), ops.equal(token_ids, 15)),
            "float32",
        )
        pemdas_scores = high_prec * 20.0 + low_prec * 15.0

        seq_positions = ops.cast(ops.arange(L), "float32")
        position_tiebreak = -0.01 * seq_positions

        scores = pemdas_scores + position_tiebreak
        scores = scores + (1.0 - mask) * (-1e9)
        # Mask non-operator positions to -inf
        scores = scores + (1.0 - is_operator) * (-1e9)
        # Safety: if no operators exist, add small uniform score to avoid
        # softmax(all -inf) = NaN. The has_ops check in multi_reduce
        # handles the result pass-through.
        any_op = ops.cast(
            ops.greater(
                ops.sum(is_operator, axis=-1, keepdims=True), 0.5
            ),
            "float32",
        )  # (B, 1)
        scores = scores + (1.0 - any_op) * mask * (-100.0)  # uniform when no ops
        reduction_weights = ops.softmax(scores, axis=-1)  # (B, L)

        # Hard operator position (for masking and re-tokenization)
        op_pos_hard = ops.argmax(reduction_weights, axis=-1)  # (B,)

        # --- Adjacent operand masking via op_cumsum ---
        is_value = ops.clip(is_digit + buffer_active, 0.0, 1.0)
        left_mask, right_mask = self._adjacent_masks(
            token_ids, is_digit, is_operator, buffer_active,
            op_pos_hard, L,
        )

        # --- Number assembly with value bypass ---
        left_val = self._assemble_with_bypass(
            token_ids, left_mask, is_digit, digit_values,
            value_buffer, buffer_active,
        )
        right_val = self._assemble_with_bypass(
            token_ids, right_mask, is_digit, digit_values,
            value_buffer, buffer_active,
        )

        # Operator classification
        pooled = ops.sum(x * ops.expand_dims(reduction_weights, -1), axis=1)
        op_logits = self.op_classifier(pooled)
        op_probs = ops.softmax(op_logits, axis=-1)

        # Fixed arithmetic with soft/hard select
        add_result, add_valid = _fixed_add(left_val, right_val)
        sub_result, sub_valid = _fixed_subtract(left_val, right_val)
        mul_result, mul_valid = _fixed_multiply(left_val, right_val)
        div_result, div_valid = _fixed_divide(left_val, right_val)

        all_results = ops.stack(
            [add_result, sub_result, mul_result, div_result], axis=1
        )
        all_valid = ops.stack(
            [add_valid, sub_valid, mul_valid, div_valid], axis=1
        )

        if training:
            op_weights = ops.expand_dims(op_probs, axis=-1)
            result = ops.sum(all_results * op_weights, axis=1)
            valid = ops.sum(all_valid * op_weights, axis=1)
        else:
            op_idx = ops.argmax(op_probs, axis=-1)
            op_one_hot = ops.one_hot(op_idx, 4)
            op_weights = ops.expand_dims(op_one_hot, axis=-1)
            result = ops.sum(all_results * op_weights, axis=1)
            valid = ops.sum(all_valid * op_weights, axis=1)

        return {
            "result": result,
            "valid": valid,
            "left_val": left_val,
            "right_val": right_val,
            "op_logits": op_logits,
            "reduction_weights": reduction_weights,
            "op_pos_hard": op_pos_hard,
            "left_mask": left_mask,
            "right_mask": right_mask,
        }

    @staticmethod
    def _adjacent_masks(token_ids, is_digit, is_operator, buffer_active,
                        op_pos, L):
        """
        Compute masks for digits immediately adjacent to the selected operator.

        Uses ``op_cumsum`` segmentation: the cumulative count of operators
        partitions the token sequence into segments. The left operand is
        the segment just before the selected operator; the right operand
        is the segment just after.

        :param token_ids: (B, L) int
        :param is_digit: (B, L) float
        :param is_operator: (B, L) float
        :param buffer_active: (B, L) float — 1 at RESULT_PLACEHOLDER positions
        :param op_pos: (B,) int — hard operator position
        :param L: int — sequence length
        :return: (left_mask, right_mask) each (B, L) float
        """
        # Segment boundaries from operator cumulative count
        op_cumsum = ops.cumsum(
            ops.cast(is_operator, "int32"), axis=1
        )  # (B, L)

        # Selected operator's cumsum value
        op_pos_expanded = ops.expand_dims(
            ops.cast(op_pos, "int32"), axis=-1
        )  # (B, 1)
        selected_cum = ops.take_along_axis(
            op_cumsum, op_pos_expanded, axis=1
        )  # (B, 1)

        # Positions that hold a value (digit or result placeholder)
        is_value = ops.clip(is_digit + buffer_active, 0.0, 1.0)

        # Left operand: value positions in segment (selected_cum - 1)
        in_left_segment = ops.cast(
            ops.equal(op_cumsum, selected_cum - 1), "float32"
        )
        left_mask = in_left_segment * is_value

        # Right operand: value positions in segment (selected_cum)
        # that are AFTER the operator position
        positions = ops.cast(ops.arange(L), "int32")  # (L,)
        after_op = ops.cast(
            ops.greater(positions, op_pos_expanded), "float32"
        )  # (B, L)
        in_right_segment = ops.cast(
            ops.equal(op_cumsum, selected_cum), "float32"
        )
        right_mask = in_right_segment * after_op * is_value

        return left_mask, right_mask

    @staticmethod
    def _assemble_with_bypass(token_ids, mask, is_digit, digit_values,
                              value_buffer, buffer_active):
        """
        Assemble a number from digit tokens, using value_buffer for
        RESULT_PLACEHOLDER positions (gradient bypass).

        If the masked region contains a buffer value (prior step result),
        use it directly. Otherwise, assemble from digit tokens via
        power-of-10 positional weighting.

        :param token_ids: (B, L) int
        :param mask: (B, L) float — which positions belong to this operand
        :param is_digit: (B, L) float
        :param digit_values: (B, L) float — 0-9 per digit
        :param value_buffer: (B, L) float — stored results from prior steps
        :param buffer_active: (B, L) float — 1 where buffer is valid
        :return: (B, 1) float — assembled value
        """
        # Check if this operand contains a result placeholder
        has_buffer = ops.sum(
            mask * buffer_active, axis=1, keepdims=True
        )  # (B, 1) — >0 if buffer present

        # Digit-only assembly (standard power-of-10)
        digit_mask = mask * is_digit * (1.0 - buffer_active)
        cumsum_d = ops.cumsum(digit_mask, axis=1)
        total_d = ops.sum(digit_mask, axis=1, keepdims=True)
        power_of_10 = (total_d - cumsum_d) * digit_mask
        positional_weight = ops.power(10.0, power_of_10) * digit_mask
        digit_val = ops.sum(
            digit_values * positional_weight, axis=1, keepdims=True
        )

        # Buffer value: read directly (carries gradients from prior step)
        buffer_val = ops.sum(
            mask * buffer_active * value_buffer, axis=1, keepdims=True
        )

        # Select: buffer if present, otherwise digits
        val = ops.where(has_buffer > 0.5, buffer_val, digit_val)
        return val

    def _retokenize(self, token_ids, value_buffer, buffer_active,
                    op_pos, left_mask, right_mask, result):
        """
        Replace ``<left> <op> <right>`` with a RESULT_PLACEHOLDER token.

        Clears consumed positions (left operand + operator + right operand
        + adjacent spaces) to PAD, places RESULT_PLACEHOLDER at the
        operator position, and stores the result float in value_buffer.

        Token modifications are non-differentiable (integer ops).
        Value buffer update IS differentiable (float assignment).

        :param token_ids: (B, L) int
        :param value_buffer: (B, L) float
        :param buffer_active: (B, L) float
        :param op_pos: (B,) int — operator position
        :param left_mask: (B, L) float — left operand positions
        :param right_mask: (B, L) float — right operand positions
        :param result: (B, 1) float — computed result (with gradients)
        :return: (new_token_ids, new_value_buffer, new_buffer_active)
        """
        L = self.max_expression_len
        op_pos_expanded = ops.expand_dims(
            ops.cast(op_pos, "int32"), axis=-1
        )  # (B, 1)

        # Operator one-hot
        op_one_hot = ops.cast(
            ops.one_hot(ops.cast(op_pos, "int32"), L), "float32"
        )  # (B, L)

        # Clear mask: left operand + operator + right operand
        clear_mask = ops.clip(left_mask + right_mask + op_one_hot, 0.0, 1.0)

        # Also clear spaces adjacent to cleared positions:
        # Shift clear_mask left/right by 1 and include space tokens
        is_space = ops.cast(ops.equal(token_ids, 3), "float32")
        # Positions adjacent to cleared region that are spaces
        clear_shifted_r = ops.concatenate(
            [ops.zeros_like(clear_mask[:, :1]), clear_mask[:, :-1]], axis=1
        )
        clear_shifted_l = ops.concatenate(
            [clear_mask[:, 1:], ops.zeros_like(clear_mask[:, :1])], axis=1
        )
        space_clear = is_space * ops.clip(
            clear_shifted_r + clear_shifted_l, 0.0, 1.0
        )
        clear_mask = ops.clip(clear_mask + space_clear, 0.0, 1.0)

        # New token_ids: clear to PAD, place RESULT_PLACEHOLDER at op_pos
        clear_int = ops.cast(clear_mask > 0.5, "int32")
        new_ids = token_ids * (1 - clear_int)  # zero out cleared positions
        # Place RESULT_PLACEHOLDER (ID=21) at op_pos
        placeholder_ids = ops.cast(op_one_hot, "int32") * RESULT_PLACEHOLDER_ID
        new_ids = new_ids + placeholder_ids

        # Update value buffer: store result at op_pos (gradient-carrying)
        result_broadcast = ops.squeeze(result, axis=-1)  # (B,) or (B,1)→(B,)
        if len(ops.shape(result_broadcast)) == 2:
            result_broadcast = result_broadcast[:, 0]
        new_buffer = value_buffer * (1.0 - clear_mask) + (
            ops.expand_dims(result_broadcast, -1) * op_one_hot
        )
        new_active = buffer_active * (1.0 - clear_mask) + op_one_hot

        return new_ids, new_buffer, new_active

    def call(self, token_ids, training=None, max_steps=1):
        """
        Forward pass: recursive multi-step reduction.

        :param token_ids: (B, L) int — tokenized expression.
        :param training: Whether in training mode.
        :param max_steps: Number of reduction steps.
        :return: Dict with final result and per-step outputs.
        """
        outputs = self.multi_reduce(token_ids, max_steps=max_steps,
                                    training=training)
        last = outputs[-1]

        # Result encoder (learned interface to host transformer)
        result_compressed = ops.sign(last["result"]) * ops.log1p(
            ops.abs(last["result"])
        )
        result_embedding = self.result_encoder(
            ops.concatenate([result_compressed, last["valid"]], axis=-1)
        )

        return {
            "result": last["result"],
            "result_embedding": result_embedding,
            "valid": last["valid"],
            "per_step": outputs,
        }

    @staticmethod
    def _soft_assemble(token_ids, soft_mask, is_digit, digit_values):
        """
        Differentiable number assembly from soft digit mask.

        :param token_ids: (B, L) int
        :param soft_mask: (B, L) float
        :param is_digit: (B, L) float
        :param digit_values: (B, L) float — 0-9
        :return: (B, 1) float
        """
        cumsum_left = ops.cumsum(soft_mask, axis=1)
        total_digits = ops.sum(soft_mask, axis=1, keepdims=True)
        power_of_10 = (total_digits - cumsum_left) * soft_mask

        positional_weight = ops.power(10.0, power_of_10) * soft_mask

        value = ops.sum(
            digit_values * positional_weight, axis=1, keepdims=True
        )
        return value

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "max_expression_len": self.max_expression_len,
            "vocab_size": self.vocab_size,
            "num_tree_layers": self.num_tree_layers,
        })
        return config


# ── Training ──────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Train DFSA (Differentiable FSA)")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-tree-layers", type=int, default=2,
                        help="Tree transformer blocks (CKY constituency for PEMDAS)")
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--act-steps", type=int, default=1,
                        help="Max ACT steps per expression (1=single-op, 4=multi-op)")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--w-operator", type=float, default=3.0)
    parser.add_argument("--w-reduction", type=float, default=20.0)
    parser.add_argument("--result-loss-weight", type=float, default=1.0)
    parser.add_argument("--curriculum-cap", type=float, default=0.8)
    parser.add_argument("--multiop-start-step", type=int, default=0,
                        help="Step at which multi-op levels are introduced. "
                        "Before this step, only single-op levels (0-7) are used. "
                        "Set to e.g. 20000 for staged training: single-op first "
                        "until reduction converges, then multi-op.")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args()


def _make_compiled_train_fn(
    model, optimizer, w_operator, w_reduction, result_loss_weight, max_act_steps
):
    """Compiled multi-step training function.

    Uses recursive reduction with re-tokenization (no teacher forcing).
    Each step: PEMDAS select → adjacent mask → assemble → arithmetic →
    re-tokenize → next step. Gradients flow through value_buffer.
    """
    max_expression_len = model.max_expression_len

    @tf.function(jit_compile=False)
    def _step(
        input_ids,            # (B, L) — original tokens
        targets,              # (B, 1) — final result
        target_validity,      # (B, 1) — validity
        per_step_token_ids,   # (B, max_act_steps, L) — NOT USED (kept for compat)
        per_step_op_index,    # (B, max_act_steps) — per-step operator type
        per_step_op_position, # (B, max_act_steps) — per-step operator position
        per_step_valid_mask,  # (B, max_act_steps) — 1.0 for real steps
    ):
        with tf.GradientTape() as tape:
            def _log_c(x):
                return tf.sign(x) * tf.math.log1p(tf.abs(x))

            # Recursive reduction (no teacher forcing)
            step_outputs = model.multi_reduce(
                input_ids, max_steps=max_act_steps, training=True
            )

            L_reduction = tf.constant(0.0)
            L_operator = tf.constant(0.0)

            for step_idx in range(max_act_steps):
                out = step_outputs[step_idx]
                step_valid = per_step_valid_mask[:, step_idx]
                step_op_idx = per_step_op_index[:, step_idx]
                step_op_pos = per_step_op_position[:, step_idx]

                # Per-step reduction loss
                step_pos_onehot = tf.one_hot(step_op_pos, max_expression_len)
                clamped_rw = tf.clip_by_value(
                    out["reduction_weights"], 1e-7, 1.0
                )
                per_sample_red = keras.losses.categorical_crossentropy(
                    step_pos_onehot, clamped_rw
                )
                L_reduction += tf.reduce_mean(per_sample_red * step_valid)

                # Per-step operator loss
                step_op_onehot = tf.one_hot(step_op_idx, 4)
                clamped_logits = tf.clip_by_value(
                    out["op_logits"], -30.0, 30.0
                )
                per_sample_op = keras.losses.categorical_crossentropy(
                    step_op_onehot, clamped_logits, from_logits=True
                )
                L_operator += tf.reduce_mean(per_sample_op * step_valid)

            n_steps = tf.cast(max_act_steps, tf.float32)
            L_reduction = L_reduction / n_steps
            L_operator = L_operator / n_steps

            # Result loss on the LAST step's output
            last_out = step_outputs[max_act_steps - 1]
            last_result = last_out["result"]
            L_result = tf.reduce_mean(
                keras.losses.huber(
                    _log_c(targets), _log_c(last_result), delta=2.0
                )
            )

            total_loss = (
                w_reduction * L_reduction
                + w_operator * L_operator
                + result_loss_weight * L_result
            )

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        grad_norms = tf.stack([tf.norm(g) for g in grads if g is not None])

        first_out = step_outputs[0]

        return {
            "total_loss": total_loss,
            "L_reduction": L_reduction,
            "L_operator": L_operator,
            "L_result": L_result,
            "result": last_result,
            "left_val": first_out["left_val"],
            "right_val": first_out["right_val"],
            "op_logits": first_out["op_logits"],
            "reduction_weights": first_out["reduction_weights"],
            "first_op_logits": first_out["op_logits"],
            "grad_mean": tf.reduce_mean(grad_norms),
            "grad_max": tf.reduce_max(grad_norms),
        }

    return _step


def _eval_digit_matrix(model, tokenizer, max_digits=10, samples_per_cell=4):
    """Evaluate model on digit-size grid (10% tolerance)."""
    ops_list = ["+", "-", "*", "/"]

    for op_sym in ops_list:
        acc_matrix = np.zeros((max_digits, max_digits))
        for d_left in range(1, max_digits + 1):
            for d_right in range(1, max_digits + 1):
                lo_l = max(1, 10 ** (d_left - 1))
                hi_l = 10 ** d_left - 1
                lo_r = max(1, 10 ** (d_right - 1))
                hi_r = 10 ** d_right - 1
                correct = 0
                total = 0
                for _ in range(samples_per_cell):
                    left = random.randint(lo_l, hi_l)
                    right = random.randint(lo_r, hi_r)
                    if op_sym == "/" and right == 0:
                        right = 1
                    expr = f"{left} {op_sym} {right}"
                    true_val, valid = _safe_eval(expr)
                    if not valid:
                        continue
                    ids = tokenizer.encode_batch([expr])
                    out = model(tf.constant(ids), training=False)
                    pred = float(out["result"][0, 0].numpy())
                    rel_err = abs(pred - true_val) / (abs(true_val) + 1e-8)
                    if rel_err < 0.10:
                        correct += 1
                    total += 1
                acc_matrix[d_left - 1, d_right - 1] = correct / total if total > 0 else 0.0

        header = f"  {'':>4s}" + "".join(f"{d:>6d}d" for d in range(1, max_digits + 1))
        logger.info(f"  ── {op_sym} accuracy (10% tol) ──")
        logger.info(header)
        for d_left in range(1, max_digits + 1):
            row = f"  {d_left:>3d}d"
            for d_right in range(1, max_digits + 1):
                val = acc_matrix[d_left - 1, d_right - 1]
                row += f"  {val:>4.0%} "
            logger.info(row)


def main():
    args = parse_args()
    setup_gpu(args.gpu)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.save_dir) / f"dfsa_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    model = DifferentiableFSA(
        hidden_size=args.hidden_size,
        max_expression_len=args.max_len,
        num_tree_layers=args.num_tree_layers,
        num_heads=args.num_heads,
    )
    tokenizer = ArithmeticTokenizer(max_len=args.max_len)

    # Warmup forward pass
    dummy = np.zeros((1, args.max_len), dtype=np.int32)
    _ = model(tf.constant(dummy), training=False)

    act_steps = args.act_steps
    param_count = sum(np.prod(v.shape) for v in model.trainable_variables)
    logger.info(f"DFSA model: hidden={args.hidden_size}, act_steps={act_steps}, params={param_count:,}")

    # Optimizer
    primary = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=max(1, args.steps - args.warmup_steps),
        alpha=0.0,
    )
    lr_schedule = WarmupSchedule(
        warmup_steps=args.warmup_steps,
        primary_schedule=primary,
        warmup_start_lr=1e-7,
    )
    optimizer = keras.optimizers.Adam(
        learning_rate=lr_schedule,
        weight_decay=args.weight_decay,
        global_clipnorm=args.clip_norm,
    )

    # Compile
    logger.info("Compiling training graph...")
    compiled_fn = _make_compiled_train_fn(
        model, optimizer, args.w_operator, args.w_reduction,
        args.result_loss_weight, act_steps,
    )

    # Helper to build per-step tensors
    def _build_per_step(expressions, input_ids):
        B, L = input_ids.shape
        all_ids = np.zeros((B, act_steps, L), dtype=np.int32)
        all_op_idx = np.zeros((B, act_steps), dtype=np.int32)
        all_op_pos = np.zeros((B, act_steps), dtype=np.int32)
        all_valid = np.zeros((B, act_steps), dtype=np.float32)
        for i, expr in enumerate(expressions):
            ps = prepare_per_step_labels(expr, tokenizer, act_steps)
            all_ids[i] = ps["per_step_token_ids"]
            all_op_idx[i] = ps["per_step_op_index"]
            all_op_pos[i] = ps["per_step_op_position"]
            all_valid[i] = ps["per_step_valid_mask"]
        return all_ids, all_op_idx, all_op_pos, all_valid

    # Warmup trace
    warmup_ids, warmup_t, warmup_v, warmup_exprs, _ = generate_curriculum_batch(
        args.batch_size, 0.0, tokenizer
    )
    w_ps_ids, w_ps_oi, w_ps_op, w_ps_vm = _build_per_step(warmup_exprs, warmup_ids)
    compiled_fn(
        tf.constant(warmup_ids),
        tf.constant(warmup_t),
        tf.constant(warmup_v),
        tf.constant(w_ps_ids),
        tf.constant(w_ps_oi),
        tf.constant(w_ps_op),
        tf.constant(w_ps_vm),
    )
    logger.info("Graph compiled. Training starts.")

    # Training loop
    metrics_log = []
    start_time = time.time()

    for step in range(1, args.steps + 1):
        progress = min(args.curriculum_cap, step / args.steps)

        # Staged training: single-op only until multiop_start_step.
        # Levels 0-7 are single-op, levels 8-10 are multi-op.
        # Cap progress so the curriculum never reaches multi-op levels
        # until the model has learned reduction on single-op.
        if args.multiop_start_step > 0 and step < args.multiop_start_step:
            # 8 single-op levels out of 11 total → cap progress at ~0.6
            # to keep sampling within levels 0-7 only
            progress = min(progress, 0.55)

        input_ids, targets, validity, expressions, labels = generate_curriculum_batch(
            args.batch_size, progress, tokenizer
        )

        ps_ids, ps_oi, ps_op, ps_vm = _build_per_step(expressions, input_ids)

        raw = compiled_fn(
            tf.constant(input_ids),
            tf.constant(targets),
            tf.constant(validity),
            tf.constant(ps_ids),
            tf.constant(ps_oi),
            tf.constant(ps_op),
            tf.constant(ps_vm),
        )

        # Numpy metrics
        pred_result = raw["result"].numpy()
        true_result = targets
        abs_true = np.abs(true_result) + 1e-8
        per_sample_rel = np.abs(pred_result - true_result) / abs_true

        # Op and reduction accuracy on STEP 0 (first reduction decision)
        pred_op = np.argmax(raw["first_op_logits"].numpy(), axis=-1)
        true_op = ps_oi[:, 0]  # step 0 operator target
        op_acc = float(np.mean(pred_op == true_op))

        pred_red = np.argmax(raw["reduction_weights"].numpy(), axis=-1)
        true_red = ps_op[:, 0]  # step 0 position target
        red_acc = float(np.mean(pred_red == true_red))

        # Number MSE (diagnostic)
        def _log_c(x):
            return np.sign(x) * np.log1p(np.abs(x))
        left_mse = float(np.mean(np.square(
            _log_c(raw["left_val"].numpy()) - _log_c(labels["left_operand"])
        )))
        right_mse = float(np.mean(np.square(
            _log_c(raw["right_val"].numpy()) - _log_c(labels["right_operand"])
        )))

        m = {
            "step": step,
            "total_loss": float(raw["total_loss"].numpy()),
            "L_reduction": float(raw["L_reduction"].numpy()),
            "L_operator": float(raw["L_operator"].numpy()),
            "L_result": float(raw["L_result"].numpy()),
            "op_acc": op_acc,
            "red_acc": red_acc,
            "step_1pct": float(np.mean(per_sample_rel < 0.01)),
            "step_5pct": float(np.mean(per_sample_rel < 0.05)),
            "step_10pct": float(np.mean(per_sample_rel < 0.10)),
            "num_mse": (left_mse + right_mse) / 2,
            "grad_mean": float(raw["grad_mean"].numpy()),
            "grad_max": float(raw["grad_max"].numpy()),
        }
        metrics_log.append(m)

        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Step {step}/{args.steps} | "
                f"loss={m['total_loss']:.4f} | "
                f"num_mse={m['num_mse']:.4f} | "
                f"op={m['op_acc']:.3f} red={m['red_acc']:.3f} | "
                f"1%={m['step_1pct']:.3f} 5%={m['step_5pct']:.3f} 10%={m['step_10pct']:.3f} | "
                f"grad={m['grad_mean']:.4f}/{m['grad_max']:.4f} | "
                f"{step/elapsed:.1f} s/s"
            )

        if step % args.eval_interval == 0:
            logger.info(f"  ── Digit accuracy matrix (step {step}) ──")
            _eval_digit_matrix(model, tokenizer)

        if step % args.save_interval == 0:
            ckpt_dir = output_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            model.save_weights(str(ckpt_dir / f"step_{step:06d}.weights.h5"))

    # Save final
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    model.save_weights(str(ckpt_dir / "final.weights.h5"))

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)

    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "hidden_size": args.hidden_size,
            "max_len": args.max_len,
            "steps": args.steps,
            "params": int(param_count),
        }, f, indent=2)

    logger.info(f"Training complete. Results in {output_dir}")
    logger.info(f"Final: loss={metrics_log[-1]['total_loss']:.4f} "
                f"op={metrics_log[-1]['op_acc']:.3f} red={metrics_log[-1]['red_acc']:.3f} "
                f"step_10%={metrics_log[-1]['step_10pct']:.3f}")


if __name__ == "__main__":
    main()
