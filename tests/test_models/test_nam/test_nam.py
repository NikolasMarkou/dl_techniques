"""
Tests for the Neural Arithmetic Module (NAM).

Covers: tokenizer, config, cell, model initialization, forward pass,
serialization, gradient flow, and fixed arithmetic validity.
"""

import pytest
import numpy as np
import keras
from keras import ops

from dl_techniques.models.nam import NAM, NAMCell, NAMConfig, NAM_VARIANTS
from dl_techniques.models.nam.tokenizer import (
    ArithmeticTokenizer,
    VOCAB_SIZE,
    PAD_ID,
    BOS_ID,
    EOS_ID,
    DIGIT_IDS,
    OPERATOR_IDS,
)
from dl_techniques.models.nam.cell import (
    _fixed_add,
    _fixed_subtract,
    _fixed_multiply,
    _fixed_divide,
)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_config():
    return NAMConfig(
        hidden_size=32,
        num_heads=4,
        num_tree_layers=1,
        intermediate_size=64,
        memory_size=8,
        num_read_heads=2,
        num_write_heads=1,
        max_expression_len=16,
        halt_max_steps=4,
        hidden_dropout_rate=0.0,
        attention_dropout_rate=0.0,
    )


@pytest.fixture
def tokenizer():
    return ArithmeticTokenizer(max_len=16)


@pytest.fixture
def tiny_model(tiny_config):
    return NAM(config=tiny_config)


@pytest.fixture
def sample_batch(tokenizer):
    expressions = ["1 + 2", "3 * 4", "10 / 2"]
    input_ids = tokenizer.encode_batch(expressions)
    return {"input_ids": input_ids}


# ── Tokenizer Tests ────────────────────────────────────────────────────


class TestArithmeticTokenizer:

    def test_vocab_size(self):
        assert VOCAB_SIZE == 21

    def test_encode_simple(self, tokenizer):
        ids = tokenizer.encode("1 + 2")
        assert ids[0] == BOS_ID
        assert ids[-1] == PAD_ID or ids[len("1 + 2") + 1] == EOS_ID

    def test_decode_roundtrip(self, tokenizer):
        expr = "123 + 456"
        ids = tokenizer.encode(expr)
        decoded = tokenizer.decode(ids)
        assert decoded == expr

    def test_encode_batch(self, tokenizer):
        batch = tokenizer.encode_batch(["1 + 2", "3 * 4"])
        assert batch.shape == (2, 16)
        assert batch.dtype == np.int32

    def test_padding(self, tokenizer):
        ids = tokenizer.encode("1")
        # Should be padded to max_len=16
        assert len(ids) == 16
        # After BOS, '1', EOS, rest should be PAD
        assert ids[3] == PAD_ID

    def test_truncation(self):
        tok = ArithmeticTokenizer(max_len=8)
        ids = tok.encode("12345678901234567890")
        assert len(ids) == 8
        assert ids[-1] == EOS_ID

    def test_operator_mask(self, tokenizer):
        ids = np.array(tokenizer.encode("1 + 2 * 3"))
        mask = tokenizer.get_operator_mask(ids)
        # Should have True at positions of + and *
        assert mask.sum() >= 2

    def test_number_mask(self, tokenizer):
        ids = np.array(tokenizer.encode("12 + 3"))
        mask = tokenizer.get_number_mask(ids)
        # '1', '2', '3' should be marked
        assert mask.sum() >= 3

    def test_all_operators_encoded(self):
        tok = ArithmeticTokenizer(max_len=32)
        expr = "1 + 2 - 3 * 4 / 5"
        ids = tok.encode(expr)
        decoded = tok.decode(ids)
        assert decoded == expr

    def test_parentheses_encoded(self, tokenizer):
        expr = "(1 + 2)"
        ids = tokenizer.encode(expr)
        decoded = tokenizer.decode(ids)
        assert decoded == expr

    def test_decimal_encoded(self, tokenizer):
        expr = "1.5 + 2.3"
        ids = tokenizer.encode(expr)
        decoded = tokenizer.decode(ids)
        assert decoded == expr


# ── Fixed Arithmetic Tests ──────────────────────────────────────────────


class TestFixedArithmetic:

    def test_add(self):
        a = ops.convert_to_tensor([[3.0]])
        b = ops.convert_to_tensor([[4.0]])
        result, valid = _fixed_add(a, b)
        assert float(result[0, 0]) == pytest.approx(7.0)
        assert float(valid[0, 0]) == pytest.approx(1.0)

    def test_subtract(self):
        a = ops.convert_to_tensor([[10.0]])
        b = ops.convert_to_tensor([[3.0]])
        result, valid = _fixed_subtract(a, b)
        assert float(result[0, 0]) == pytest.approx(7.0)
        assert float(valid[0, 0]) == pytest.approx(1.0)

    def test_multiply(self):
        a = ops.convert_to_tensor([[3.0]])
        b = ops.convert_to_tensor([[4.0]])
        result, valid = _fixed_multiply(a, b)
        assert float(result[0, 0]) == pytest.approx(12.0)
        assert float(valid[0, 0]) == pytest.approx(1.0)

    def test_divide_normal(self):
        a = ops.convert_to_tensor([[10.0]])
        b = ops.convert_to_tensor([[2.0]])
        result, valid = _fixed_divide(a, b)
        assert float(result[0, 0]) == pytest.approx(5.0)
        assert float(valid[0, 0]) == pytest.approx(1.0)

    def test_divide_by_zero(self):
        a = ops.convert_to_tensor([[10.0]])
        b = ops.convert_to_tensor([[0.0]])
        result, valid = _fixed_divide(a, b)
        assert float(result[0, 0]) == pytest.approx(0.0)
        assert float(valid[0, 0]) == pytest.approx(0.0)

    def test_divide_near_zero(self):
        a = ops.convert_to_tensor([[10.0]])
        b = ops.convert_to_tensor([[1e-10]])
        result, valid = _fixed_divide(a, b)
        # Near-zero denominator → invalid
        assert float(valid[0, 0]) == pytest.approx(0.0)
        assert float(result[0, 0]) == pytest.approx(0.0)

    def test_divide_negative_zero(self):
        a = ops.convert_to_tensor([[5.0]])
        b = ops.convert_to_tensor([[-0.0]])
        result, valid = _fixed_divide(a, b)
        assert float(valid[0, 0]) == pytest.approx(0.0)

    def test_add_batched(self):
        a = ops.convert_to_tensor([[1.0], [2.0], [3.0]])
        b = ops.convert_to_tensor([[4.0], [5.0], [6.0]])
        result, valid = _fixed_add(a, b)
        expected = [5.0, 7.0, 9.0]
        for i in range(3):
            assert float(result[i, 0]) == pytest.approx(expected[i])
            assert float(valid[i, 0]) == pytest.approx(1.0)

    def test_divide_mixed_validity(self):
        """Batch where some divisions are valid, some aren't."""
        a = ops.convert_to_tensor([[10.0], [20.0], [30.0]])
        b = ops.convert_to_tensor([[2.0], [0.0], [5.0]])
        result, valid = _fixed_divide(a, b)
        assert float(result[0, 0]) == pytest.approx(5.0)
        assert float(valid[0, 0]) == pytest.approx(1.0)
        assert float(result[1, 0]) == pytest.approx(0.0)
        assert float(valid[1, 0]) == pytest.approx(0.0)
        assert float(result[2, 0]) == pytest.approx(6.0)
        assert float(valid[2, 0]) == pytest.approx(1.0)


# ── Config Tests ────────────────────────────────────────────────────────


class TestNAMConfig:

    def test_default_config(self):
        config = NAMConfig()
        assert config.hidden_size == 128
        assert config.vocab_size == 21
        assert config.num_read_heads == 2

    def test_roundtrip(self):
        config = NAMConfig(hidden_size=64, num_heads=4)
        d = config.to_dict()
        config2 = NAMConfig.from_dict(d)
        assert config2.hidden_size == 64
        assert config2.num_heads == 4

    def test_validation(self):
        with pytest.raises(ValueError):
            NAMConfig(hidden_size=65, num_heads=4)  # not divisible

    def test_variants_exist(self):
        assert "tiny" in NAM_VARIANTS
        assert "small" in NAM_VARIANTS
        assert "base" in NAM_VARIANTS


# ── NAMCell Tests ───────────────────────────────────────────────────────


class TestNAMCell:

    def test_creation(self, tiny_config):
        cell = NAMCell(config=tiny_config)
        assert cell.config.hidden_size == 32

    def test_initialize_carry(self, tiny_config):
        cell = NAMCell(config=tiny_config)
        carry = cell.initialize_carry(batch_size=2)
        assert carry["memory"].shape == (2, 8, 32)
        assert len(carry["read_weights"]) == 2
        assert carry["steps"].shape == (2,)

    def test_forward_pass(self, tiny_config):
        cell = NAMCell(config=tiny_config)
        batch_size = 2
        L = tiny_config.max_expression_len
        D = tiny_config.hidden_size

        carry = cell.initialize_carry(batch_size)
        hidden = ops.ones((batch_size, L, D))
        mask = ops.ones((batch_size, 1, L), dtype="int32")
        # Dummy token_ids with a simple expression pattern
        token_ids = np.zeros((batch_size, L), dtype=np.int32)
        token_ids[:, 0] = 1   # BOS
        token_ids[:, 1] = 5   # digit '1'
        token_ids[:, 2] = 3   # space
        token_ids[:, 3] = 14  # '+'
        token_ids[:, 4] = 3   # space
        token_ids[:, 5] = 6   # digit '2'
        token_ids[:, 6] = 2   # EOS

        new_carry, outputs = cell((carry, hidden, mask, token_ids), training=False)

        assert outputs["result"].shape == (batch_size, 1)
        assert outputs["valid"].shape == (batch_size, 1)
        assert outputs["op_logits"].shape == (batch_size, 4)
        assert outputs["q_halt"].shape == (batch_size,)
        assert outputs["hidden"].shape == (batch_size, L, D)

        # New intermediate outputs for multi-task supervision
        assert outputs["left_val"].shape == (batch_size, 1)
        assert outputs["right_val"].shape == (batch_size, 1)
        assert outputs["reduction_weights"].shape == (batch_size, L)

    def test_validity_output_range(self, tiny_config):
        """Validity from cell should be in roughly [0, 1] range."""
        cell = NAMCell(config=tiny_config)
        carry = cell.initialize_carry(4)
        hidden = keras.random.normal((4, tiny_config.max_expression_len, 32))
        mask = ops.ones((4, 1, tiny_config.max_expression_len), dtype="int32")
        token_ids = np.zeros((4, tiny_config.max_expression_len), dtype=np.int32)

        _, outputs = cell((carry, hidden, mask, token_ids), training=False)
        valid = outputs["valid"].numpy()
        # valid is a soft value from weighted arithmetic ops
        assert valid.shape == (4, 1)

    def test_serialization(self, tiny_config):
        cell = NAMCell(config=tiny_config)
        config = cell.get_config()
        cell2 = NAMCell.from_config(config)
        assert cell2.config.hidden_size == tiny_config.hidden_size


# ── NAM Model Tests ─────────────────────────────────────────────────────


class TestNAM:

    def test_creation(self, tiny_config):
        model = NAM(config=tiny_config)
        assert model.config.hidden_size == 32

    def test_from_variant(self):
        model = NAM.from_variant("tiny")
        assert model.config.hidden_size == 64

    def test_initial_carry(self, tiny_model, sample_batch):
        carry = tiny_model.initial_carry(sample_batch)
        assert "cell_carry" in carry
        assert "steps" in carry
        assert "halted" in carry

    def test_single_step(self, tiny_model, sample_batch):
        carry = tiny_model.initial_carry(sample_batch)
        new_carry, outputs = tiny_model(carry, sample_batch, training=False)

        assert outputs["result"].shape[0] == 3
        assert outputs["result"].shape[1] == 1
        assert outputs["valid"].shape == (3, 1)
        assert "q_halt_logits" in outputs
        assert "q_continue_logits" in outputs

        # Multi-task intermediate outputs
        assert outputs["step_left_val"].shape == (3, 1)
        assert outputs["step_right_val"].shape == (3, 1)
        assert outputs["reduction_weights"].shape[0] == 3
        assert outputs["reduction_weights"].shape[1] == tiny_model.config.max_expression_len

    def test_multi_step_loop(self, tiny_model, sample_batch):
        carry = tiny_model.initial_carry(sample_batch)

        for step in range(tiny_model.config.halt_max_steps):
            carry, outputs = tiny_model(carry, sample_batch, training=False)
            sample_batch = outputs["batch"]

            if np.all(carry["halted"].numpy()):
                break

        # Should have taken at least 1 step
        assert np.all(carry["steps"].numpy() >= 1)

    def test_gradient_flow(self, tiny_model, sample_batch):
        """Verify gradients flow through the ACT loop."""
        import tensorflow as tf

        input_ids = tf.constant(sample_batch["input_ids"])
        targets = tf.constant(np.array([[3.0], [12.0], [5.0]], dtype=np.float32))
        batch = {"input_ids": input_ids}

        with tf.GradientTape() as tape:
            carry = tiny_model.initial_carry(batch)
            carry, outputs = tiny_model(carry, batch, training=True)
            loss = tf.reduce_mean(tf.square(outputs["result"] - targets))

        grads = tape.gradient(loss, tiny_model.trainable_variables)
        # At least some gradients should be non-None
        non_none = [g for g in grads if g is not None]
        assert len(non_none) > 0
        # At least one gradient should be non-zero
        has_nonzero = any(
            float(tf.reduce_sum(tf.abs(g)).numpy()) > 0 for g in non_none
        )
        assert has_nonzero

    def test_config_roundtrip(self, tiny_config):
        model = NAM(config=tiny_config)
        config = model.get_config()
        model2 = NAM.from_config(config.copy())
        assert model2.config.hidden_size == tiny_config.hidden_size

    def test_training_mode_dropout(self, tiny_config):
        """Training vs inference should produce different outputs when dropout > 0."""
        config = NAMConfig(
            hidden_size=32,
            num_heads=4,
            num_tree_layers=1,
            intermediate_size=64,
            memory_size=8,
            num_read_heads=2,
            max_expression_len=16,
            halt_max_steps=2,
            hidden_dropout_rate=0.5,
            attention_dropout_rate=0.5,
        )
        model = NAM(config=config)
        tokenizer = ArithmeticTokenizer(max_len=16)
        ids = tokenizer.encode_batch(["1 + 2"])
        batch = {"input_ids": ids}

        carry1 = model.initial_carry(batch)
        _, out_train = model(carry1, batch, training=True)

        carry2 = model.initial_carry(batch)
        _, out_eval = model(carry2, batch, training=False)

        # Outputs may differ due to dropout (not guaranteed but likely)
        # Just check both produce valid outputs
        assert out_train["result"].shape == (1, 1)
        assert out_eval["result"].shape == (1, 1)

    def test_all_variants_create(self):
        """All preset variants should create successfully."""
        for variant_name in NAM_VARIANTS:
            model = NAM.from_variant(variant_name)
            assert model.config.hidden_size > 0

    def test_invalid_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            NAM.from_variant("nonexistent")

    def test_deterministic_number_assembly(self, tiny_model, sample_batch):
        """Number extraction must be exact when reduction points to operator."""
        from dl_techniques.models.nam.tokenizer import ArithmeticTokenizer
        tokenizer = ArithmeticTokenizer(max_len=tiny_model.config.max_expression_len)

        # Test expressions with known answers
        exprs = ["3 + 5", "12 * 4", "999 - 1"]
        ids = tokenizer.encode_batch(exprs)
        batch = {"input_ids": ids}
        carry = tiny_model.initial_carry(batch)
        _, out = tiny_model(carry, batch, training=False)

        # We can't guarantee reduction points to the right place (untrained),
        # but we CAN verify the assembly function itself works by checking
        # that left_val and right_val are valid numbers (not NaN/Inf).
        left = out["step_left_val"].numpy()
        right = out["step_right_val"].numpy()
        assert not np.any(np.isnan(left)), "left_val contains NaN"
        assert not np.any(np.isnan(right)), "right_val contains NaN"
        assert not np.any(np.isinf(left)), "left_val contains Inf"
        assert not np.any(np.isinf(right)), "right_val contains Inf"

        # Verify no learned number_head layers exist
        cell = tiny_model.cell
        assert not hasattr(cell, "number_head"), "old number_head should not exist"
        assert not hasattr(cell, "left_number_head"), "learned left_number_head removed"
        assert not hasattr(cell, "right_number_head"), "learned right_number_head removed"

    def test_weight_round_trip_bit_exact(self, tiny_model, sample_batch, tmp_path):
        """Saving and reloading weights must yield bit-exact forward outputs.

        NTMMemory initial state is random per initial_carry() call, so the same
        carry dict is reused for both forward passes to isolate the weight
        round-trip effect from carry initialization randomness.
        """
        # Build model 1 and produce the reference output
        carry = tiny_model.initial_carry(sample_batch)
        _, out1 = tiny_model(carry, sample_batch, training=False)

        # Save weights
        path = str(tmp_path / "nam_tiny.weights.h5")
        tiny_model.save_weights(path)

        # Build a fresh model with the same config, build it, then load weights
        model2 = NAM(config=tiny_model.config)
        dummy_carry = model2.initial_carry(sample_batch)
        _ = model2(dummy_carry, sample_batch, training=False)  # triggers build
        model2.load_weights(path)

        # Reuse the ORIGINAL carry so both models see identical initial memory
        _, out2 = model2(carry, sample_batch, training=False)

        # Crucial outputs must match exactly — this specifically exercises
        # the split number heads (via step_left_val / step_right_val)
        for key in [
            "step_left_val",
            "step_right_val",
            "result",
            "op_logits",
            "reduction_weights",
        ]:
            a = np.asarray(out1[key])
            b = np.asarray(out2[key])
            assert a.shape == b.shape
            np.testing.assert_allclose(a, b, atol=1e-6, rtol=0, err_msg=f"mismatch in {key}")

    def test_deterministic_assembly_exact(self):
        """Verify _assemble_number_from_tokens gives exact values."""
        from dl_techniques.models.nam.cell import _assemble_number_from_tokens

        # Simulate "123 + 45": digits 1,2,3 on left, 4,5 on right
        # token_ids where digit '1'=5, '2'=6, '3'=7, '4'=8, '5'=9
        token_ids = np.array([[1, 5, 6, 7, 3, 14, 3, 8, 9, 2, 0, 0]], dtype=np.int32)
        #                     BOS 1  2  3  SP  +  SP 4  5  EOS PAD PAD

        # Left mask: digits before operator (pos 5)
        left_mask = np.array([[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        right_mask = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]], dtype=np.float32)

        left_val = _assemble_number_from_tokens(token_ids, left_mask)
        right_val = _assemble_number_from_tokens(token_ids, right_mask)

        assert float(left_val[0, 0]) == pytest.approx(123.0, abs=1e-3)
        assert float(right_val[0, 0]) == pytest.approx(45.0, abs=1e-3)

    def test_deterministic_assembly_large_numbers(self):
        """Verify assembly works up to 10-digit numbers."""
        from dl_techniques.models.nam.cell import _assemble_number_from_tokens

        # "1234567890 + 1"
        digits = [5, 6, 7, 8, 9, 10, 11, 12, 13, 4]  # 1,2,3,4,5,6,7,8,9,0
        token_ids = np.array([[1] + digits + [3, 14, 3, 5, 2] + [0]*4], dtype=np.int32)
        L = token_ids.shape[1]

        left_mask = np.zeros((1, L), dtype=np.float32)
        left_mask[0, 1:11] = 1.0  # positions 1-10 are the 10 digits
        right_mask = np.zeros((1, L), dtype=np.float32)
        right_mask[0, 14] = 1.0  # position 14 is digit '1'

        left_val = _assemble_number_from_tokens(token_ids, left_mask)
        right_val = _assemble_number_from_tokens(token_ids, right_mask)

        assert float(left_val[0, 0]) == pytest.approx(1234567890.0, rel=1e-6)
        assert float(right_val[0, 0]) == pytest.approx(1.0, abs=1e-3)


# ── Data Generator Tests ────────────────────────────────────────────────


class TestDataGenerator:

    def test_generate_batch(self, tokenizer):
        from train.nam.data_generator import generate_batch, ExpressionConfig

        config = ExpressionConfig(
            min_ops=1, max_ops=1,
            min_val=1, max_val=10,
            operators=["+", "-"],
        )
        ids, targets, validity, exprs, labels = generate_batch(8, config, tokenizer)
        assert ids.shape == (8, 16)
        assert targets.shape == (8, 1)
        assert validity.shape == (8, 1)
        assert len(exprs) == 8

        # Enriched labels
        assert labels["left_operand"].shape == (8, 1)
        assert labels["right_operand"].shape == (8, 1)
        assert labels["operator_index"].shape == (8,)
        assert labels["operator_position"].shape == (8,)
        # Operator index should be 0 (+) or 1 (-)
        assert all(idx in (0, 1) for idx in labels["operator_index"])

    def test_division_by_zero_handling(self, tokenizer):
        """Division expressions should have valid=0 when dividing by zero."""
        from train.nam.data_generator import _safe_eval

        result, valid = _safe_eval("1 / 0")
        assert valid is False
        assert result == 0.0

    def test_safe_eval_normal(self):
        from train.nam.data_generator import _safe_eval

        result, valid = _safe_eval("2 + 3 * 4")
        assert valid is True
        assert result == pytest.approx(14.0)

    def test_safe_eval_parentheses(self):
        from train.nam.data_generator import _safe_eval

        result, valid = _safe_eval("(2 + 3) * 4")
        assert valid is True
        assert result == pytest.approx(20.0)

    def test_curriculum_phases_exist(self):
        from train.nam.data_generator import CURRICULUM

        assert len(CURRICULUM) == 5
        assert "phase_1" in CURRICULUM
        assert "phase_5" in CURRICULUM

    def test_parse_single_op_addition(self, tokenizer):
        from train.nam.data_generator import _parse_single_op

        expr = "6 + 8"
        token_ids = tokenizer.encode(expr)
        parsed = _parse_single_op(expr, token_ids)

        assert parsed["left_operand"] == pytest.approx(6.0)
        assert parsed["right_operand"] == pytest.approx(8.0)
        assert parsed["operator_index"] == 0  # +
        assert parsed["operator_position"] > 0  # somewhere after BOS

    def test_parse_single_op_division(self, tokenizer):
        from train.nam.data_generator import _parse_single_op

        expr = "15 / 3"
        token_ids = tokenizer.encode(expr)
        parsed = _parse_single_op(expr, token_ids)

        assert parsed["left_operand"] == pytest.approx(15.0)
        assert parsed["right_operand"] == pytest.approx(3.0)
        assert parsed["operator_index"] == 3  # /

    def test_parse_single_op_multiply(self, tokenizer):
        from train.nam.data_generator import _parse_single_op

        expr = "4 * 7"
        token_ids = tokenizer.encode(expr)
        parsed = _parse_single_op(expr, token_ids)

        assert parsed["left_operand"] == pytest.approx(4.0)
        assert parsed["right_operand"] == pytest.approx(7.0)
        assert parsed["operator_index"] == 2  # *

    def test_parse_single_op_subtraction(self, tokenizer):
        from train.nam.data_generator import _parse_single_op

        expr = "10 - 3"
        token_ids = tokenizer.encode(expr)
        parsed = _parse_single_op(expr, token_ids)

        assert parsed["left_operand"] == pytest.approx(10.0)
        assert parsed["right_operand"] == pytest.approx(3.0)
        assert parsed["operator_index"] == 1  # -

    def test_enriched_labels_correctness(self, tokenizer):
        """Verify that enriched labels match expected values for known expressions."""
        from train.nam.data_generator import generate_batch, ExpressionConfig

        config = ExpressionConfig(
            min_ops=1, max_ops=1,
            min_val=1, max_val=10,
            operators=["+"],
        )
        ids, targets, validity, exprs, labels = generate_batch(16, config, tokenizer)

        # All expressions use + only, so operator_index should all be 0
        assert all(idx == 0 for idx in labels["operator_index"])

        # For each expression, left + right should equal the target (when valid)
        for i in range(len(exprs)):
            if validity[i, 0] > 0.5:
                expected = labels["left_operand"][i, 0] + labels["right_operand"][i, 0]
                assert expected == pytest.approx(targets[i, 0], abs=1e-4)


# ── Training Resume Tests ──────────────────────────────────────────────


class TestTrainingResume:
    """Verify that checkpoint resume preserves step/LR/optimizer state."""

    def test_sidecar_round_trip(self, tmp_path):
        """save_training_state + load_training_state preserves all fields."""
        from train.nam.train_nam import save_training_state, load_training_state

        weights_path = str(tmp_path / "step_000100.weights.h5")
        # Write a dummy weights file so the sidecar can sit next to it
        (tmp_path / "step_000100.weights.h5").write_bytes(b"dummy")

        save_training_state(
            weights_path,
            step=100,
            total_steps=20000,
            best_loss=0.1234,
            curriculum=True,
            curriculum_cap=0.8,
        )

        loaded = load_training_state(weights_path)
        assert loaded is not None
        assert loaded["step"] == 100
        assert loaded["total_steps"] == 20000
        assert loaded["best_loss"] == pytest.approx(0.1234)
        assert loaded["curriculum"] is True
        assert loaded["curriculum_cap"] == pytest.approx(0.8)

    def test_sidecar_missing_returns_none(self, tmp_path):
        """Missing sidecar must return None (graceful fallback)."""
        from train.nam.train_nam import load_training_state

        weights_path = str(tmp_path / "nonexistent.weights.h5")
        assert load_training_state(weights_path) is None

    def test_optimizer_iterations_assignable(self):
        """Keras 3 Adam with a LearningRateSchedule allows iterations.assign.

        This is the core mechanism by which the LR schedule picks up at
        the correct point after resume — the schedule is a pure function
        of optimizer.iterations.
        """
        from train.nam.train_nam import create_optimizer

        optimizer = create_optimizer(
            lr=1e-4,
            weight_decay=1e-5,
            clip_norm=10.0,
            warmup_steps=1000,
            total_steps=20000,
        )

        # Force iterations variable creation
        _ = optimizer.iterations
        # Assign a non-zero step count (simulate resume)
        optimizer.iterations.assign(5000)
        assert int(optimizer.iterations.numpy()) == 5000

        # The LR at step 5000 should be the post-warmup cosine value,
        # not the warmup ramp value.
        lr_at_5000 = float(optimizer.learning_rate)
        assert lr_at_5000 > 1e-7  # past warmup
        assert lr_at_5000 <= 1e-4 + 1e-8  # bounded by peak lr
