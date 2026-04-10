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
from dl_techniques.optimization.warmup_schedule import WarmupSchedule
from dl_techniques.utils.logger import logger
from train.common import setup_gpu
from train.nam.data_generator import (
    generate_curriculum_batch,
    _safe_eval,
    DIFFICULTY_LEVELS,
    _curriculum_probs,
)


# ── Differentiable Finite-State Arithmetic Module ─────────────────────


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
        vocab_size: int = 21,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.max_expression_len = max_expression_len
        self.vocab_size = vocab_size

        # --- Token feature extractor (small, learned) ---
        # Embeds token IDs + numeric features into a compact representation.
        # This is the "interface" that a host transformer would replace
        # with its own hidden states.
        self.token_embedding = keras.layers.Embedding(
            vocab_size, hidden_size, name="token_embedding"
        )
        self.numeric_proj = keras.layers.Dense(
            hidden_size, name="numeric_proj"
        )
        # Small 2-layer MLP to create per-token features
        self.feature_mlp = keras.layers.Dense(
            hidden_size, activation="gelu", name="feature_mlp"
        )
        self.feature_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="feature_norm"
        )

        # --- Reduction scorer (learned) ---
        # Predicts which token position is the operator to reduce.
        # Output: softmax over positions → soft operator position.
        self.reduction_scorer = keras.layers.Dense(
            1, name="reduction_scorer"
        )

        # --- Operator classifier (learned) ---
        # Predicts which of the 4 operators (+, -, *, /) is at the
        # reduction position.
        self.op_classifier = keras.layers.Dense(
            4, name="op_classifier"
        )

        # --- Result encoder (learned interface to host transformer) ---
        self.result_encoder = keras.layers.Dense(
            hidden_size, name="result_encoder"
        )

    def call(self, token_ids, training=None):
        """
        Evaluate arithmetic expression with full gradient flow.

        :param token_ids: (B, L) int — tokenized expression.
        :param training: Whether in training mode.
        :return: Dict with 'result', 'result_embedding', 'valid',
            'left_val', 'right_val', 'op_logits', 'reduction_weights'.
        """
        B = ops.shape(token_ids)[0]
        L = ops.shape(token_ids)[1]

        # --- 1. Token features ---
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

        # Operator type features
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

        # Numeric feature channels: (B, L, 4)
        numeric_features = ops.stack(
            [digit_values, is_digit, op_type, is_operator], axis=-1
        )

        # Learned features: embedding + numeric projection + MLP
        import math
        x = self.token_embedding(token_ids) * math.sqrt(self.hidden_size)
        x = x + self.numeric_proj(numeric_features)
        x = self.feature_mlp(x)
        x = self.feature_norm(x)  # (B, L, D)

        # --- 2. Reduction scorer → soft operator position ---
        scores = ops.squeeze(self.reduction_scorer(x), axis=-1)  # (B, L)
        scores = scores + (1.0 - mask) * (-1e9)  # mask padding
        reduction_weights = ops.softmax(scores, axis=-1)  # (B, L)

        # --- 3. Soft left/right masks (DIFFERENTIABLE) ---
        # cumsum of reduction_weights gives the cumulative probability
        # that the operator is at or before each position.
        # Before the operator: cum_prob ≈ 0 → left_mask ≈ is_digit
        # After the operator: cum_prob ≈ 1 → left_mask ≈ 0
        cum_op_prob = ops.cumsum(reduction_weights, axis=1)  # (B, L)

        # Shift by one: the operator position itself should NOT be in either mask.
        # cum_op_prob at the operator pos = the operator's own probability (≈1 when peaked).
        # We want: left_mask includes positions BEFORE the peak, right_mask AFTER.
        # Using (cum_op_prob - reduction_weights) gives the cum prob EXCLUDING current.
        cum_before = cum_op_prob - reduction_weights  # (B, L)

        left_mask = (1.0 - cum_op_prob) * is_digit       # (B, L) — digits before operator
        right_mask = cum_before * is_digit                 # (B, L) — digits after operator

        # --- 4. Differentiable number assembly ---
        left_val = self._soft_assemble(token_ids, left_mask, is_digit, digit_values)
        right_val = self._soft_assemble(token_ids, right_mask, is_digit, digit_values)

        # --- 5. Operator classification ---
        # Pool hidden features at the reduction position
        pooled = ops.sum(x * ops.expand_dims(reduction_weights, -1), axis=1)  # (B, D)
        op_logits = self.op_classifier(pooled)  # (B, 4)
        op_probs = ops.softmax(op_logits, axis=-1)  # (B, 4)

        # --- 6. Fixed arithmetic with soft-select ---
        add_result, add_valid = _fixed_add(left_val, right_val)
        sub_result, sub_valid = _fixed_subtract(left_val, right_val)
        mul_result, mul_valid = _fixed_multiply(left_val, right_val)
        div_result, div_valid = _fixed_divide(left_val, right_val)

        all_results = ops.stack(
            [add_result, sub_result, mul_result, div_result], axis=1
        )  # (B, 4, 1)
        all_valid = ops.stack(
            [add_valid, sub_valid, mul_valid, div_valid], axis=1
        )  # (B, 4, 1)

        if training:
            # Soft-select (differentiable)
            op_weights = ops.expand_dims(op_probs, axis=-1)  # (B, 4, 1)
            result = ops.sum(all_results * op_weights, axis=1)  # (B, 1)
            valid = ops.sum(all_valid * op_weights, axis=1)  # (B, 1)
        else:
            # Hard-select (exact)
            op_idx = ops.argmax(op_probs, axis=-1)
            op_one_hot = ops.one_hot(op_idx, 4)
            op_weights = ops.expand_dims(op_one_hot, axis=-1)
            result = ops.sum(all_results * op_weights, axis=1)
            valid = ops.sum(all_valid * op_weights, axis=1)

        # --- 7. Result encoder (learned interface) ---
        result_compressed = ops.sign(result) * ops.log1p(ops.abs(result))
        result_embedding = self.result_encoder(
            ops.concatenate([result_compressed, valid], axis=-1)
        )

        return {
            "result": result,
            "valid": valid,
            "result_embedding": result_embedding,
            "left_val": left_val,
            "right_val": right_val,
            "op_logits": op_logits,
            "reduction_weights": reduction_weights,
        }

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

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "max_expression_len": self.max_expression_len,
            "vocab_size": self.vocab_size,
        })
        return config


# ── Training ──────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Train DFSA (Differentiable FSA)")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--w-operator", type=float, default=3.0)
    parser.add_argument("--w-reduction", type=float, default=5.0)
    parser.add_argument("--result-loss-weight", type=float, default=1.0)
    parser.add_argument("--curriculum-cap", type=float, default=0.8)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args()


def _make_compiled_train_fn(model, optimizer, w_operator, w_reduction, result_loss_weight):
    """Compiled single-op training step."""

    max_expression_len = model.max_expression_len

    @tf.function(jit_compile=False)
    def _step(input_ids, targets, target_validity, true_op_index, true_op_position):
        with tf.GradientTape() as tape:
            out = model(input_ids, training=True)

            # Log-compress
            def _log_c(x):
                return tf.sign(x) * tf.math.log1p(tf.abs(x))

            # Reduction loss: CE on reduction_weights vs true operator position
            true_pos_onehot = tf.one_hot(true_op_position, max_expression_len)
            clamped_rw = tf.clip_by_value(out["reduction_weights"], 1e-7, 1.0)
            L_reduction = tf.reduce_mean(
                keras.losses.categorical_crossentropy(true_pos_onehot, clamped_rw)
            )

            # Operator loss: CE on op_logits vs true operator type
            true_op_onehot = tf.one_hot(true_op_index, 4)
            clamped_logits = tf.clip_by_value(out["op_logits"], -30.0, 30.0)
            L_operator = tf.reduce_mean(
                keras.losses.categorical_crossentropy(
                    true_op_onehot, clamped_logits, from_logits=True
                )
            )

            # Result loss: Huber in log-space (trains result_encoder)
            L_result = tf.reduce_mean(
                keras.losses.huber(_log_c(targets), _log_c(out["result"]), delta=2.0)
            )

            total_loss = (
                w_reduction * L_reduction
                + w_operator * L_operator
                + result_loss_weight * L_result
            )

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        grad_norms = tf.stack([tf.norm(g) for g in grads if g is not None])

        return {
            "total_loss": total_loss,
            "L_reduction": L_reduction,
            "L_operator": L_operator,
            "L_result": L_result,
            "result": out["result"],
            "left_val": out["left_val"],
            "right_val": out["right_val"],
            "op_logits": out["op_logits"],
            "reduction_weights": out["reduction_weights"],
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
    )
    tokenizer = ArithmeticTokenizer(max_len=args.max_len)

    # Warmup forward pass
    dummy = np.zeros((1, args.max_len), dtype=np.int32)
    _ = model(tf.constant(dummy), training=False)

    param_count = sum(np.prod(v.shape) for v in model.trainable_variables)
    logger.info(f"DFSA model: hidden={args.hidden_size}, params={param_count:,}")

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
        model, optimizer, args.w_operator, args.w_reduction, args.result_loss_weight
    )

    # Warmup trace
    warmup_ids, warmup_t, warmup_v, _, warmup_labels = generate_curriculum_batch(
        args.batch_size, 0.0, tokenizer
    )
    compiled_fn(
        tf.constant(warmup_ids),
        tf.constant(warmup_t),
        tf.constant(warmup_v),
        tf.constant(warmup_labels["operator_index"]),
        tf.constant(warmup_labels["operator_position"]),
    )
    logger.info("Graph compiled. Training starts.")

    # Training loop
    metrics_log = []
    start_time = time.time()

    for step in range(1, args.steps + 1):
        progress = min(args.curriculum_cap, step / args.steps)
        input_ids, targets, validity, expressions, labels = generate_curriculum_batch(
            args.batch_size, progress, tokenizer
        )

        raw = compiled_fn(
            tf.constant(input_ids),
            tf.constant(targets),
            tf.constant(validity),
            tf.constant(labels["operator_index"]),
            tf.constant(labels["operator_position"]),
        )

        # Numpy metrics
        pred_result = raw["result"].numpy()
        true_result = targets
        abs_true = np.abs(true_result) + 1e-8
        per_sample_rel = np.abs(pred_result - true_result) / abs_true

        pred_op = np.argmax(raw["op_logits"].numpy(), axis=-1)
        true_op = labels["operator_index"]
        op_acc = float(np.mean(pred_op == true_op))

        pred_red = np.argmax(raw["reduction_weights"].numpy(), axis=-1)
        true_red = labels["operator_position"]
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
