"""
Training script for the Neural Arithmetic Module (NAM).

Uses an external ACT loop (from TRM pattern) with curriculum learning.
Expressions are generated on-the-fly with increasing difficulty.

Usage::

    CUDA_VISIBLE_DEVICES=1 python -m train.nam.train_nam \\
        --variant small \\
        --phase phase_1 \\
        --steps 10000 \\
        --batch-size 64 \\
        --lr 1e-4

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

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dl_techniques.models.nam import NAM, NAMConfig, NAM_VARIANTS
from dl_techniques.models.nam.tokenizer import ArithmeticTokenizer
from dl_techniques.optimization.warmup_schedule import WarmupSchedule
from dl_techniques.utils.logger import logger
from train.common import setup_gpu
from train.nam.data_generator import (
    generate_batch,
    generate_curriculum_batch,
    _safe_eval,
    ExpressionConfig,
    CURRICULUM,
    DIFFICULTY_LEVELS,
    _curriculum_probs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NAM")
    parser.add_argument(
        "--variant",
        type=str,
        default="small",
        choices=list(NAM_VARIANTS.keys()),
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="phase_1",
        choices=list(CURRICULUM.keys()),
    )
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--min-val", type=int, default=None,
        help="Override minimum operand value (default: from phase config)",
    )
    parser.add_argument(
        "--max-val", type=int, default=None,
        help="Override maximum operand value (default: from phase config)",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--ponder-cost", type=float, default=0.01)
    parser.add_argument("--result-loss-weight", type=float, default=1.0)
    parser.add_argument("--valid-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--w-number", type=float, default=0.5,
        help="Weight for number extraction loss (S1). "
        "Keep low (0.5) — high values create large gradient norms that "
        "suppress operator and reduction learning via global clip.",
    )
    parser.add_argument(
        "--number-loss-delta", type=float, default=0.1,
        help="Huber delta for relative-error number loss (iteration 1). "
        "rel_err = (pred - target) / (|target| + 1). "
        "Below delta: quadratic. Above: linear. Smaller delta → more "
        "aggressive gradient on small errors.",
    )
    parser.add_argument(
        "--w-operator", type=float, default=3.0,
        help="Weight for operator classification CE loss (S2)",
    )
    parser.add_argument(
        "--w-reduction", type=float, default=5.0,
        help="Weight for reduction target CE loss (S3). "
        "High weight is critical — reduction must converge first for all "
        "downstream sub-skills (number extraction, operator) to work.",
    )
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=1000,
                        help="Run digit-accuracy matrix eval every N steps")
    parser.add_argument(
        "--act-steps", type=int, default=None,
        help="Override ACT depth (default: from model config). "
        "Use smaller values for early curriculum phases (e.g., 2-4 for phase_1).",
    )
    parser.add_argument(
        "--save-dir", type=str, default="results",
        help="Root results directory (default: results/)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU device ID")
    parser.add_argument(
        "--curriculum", action="store_true",
        help="Use smooth curriculum: difficulty increases over training, "
        "always mixing in easier examples to prevent forgetting. "
        "Overrides --phase/--min-val/--max-val.",
    )
    parser.add_argument(
        "--curriculum-cap", type=float, default=0.8,
        help="Cap curriculum progress at this value (default 0.8). "
        "At progress=1.0 the curriculum Gaussian concentrates ~67%% on "
        "the hardest 3 levels, which caused late-training operator "
        "regression in the 100K baseline (README §Final Run obs #4). "
        "Capping at 0.8 keeps the hardest levels at ~20%% and preserves "
        "sub-skill stability.",
    )
    parser.add_argument(
        "--log-grad-norms", action="store_true",
        help="Enable per-sub-skill gradient norm instrumentation. Uses a "
        "persistent GradientTape with 4 extra tape.gradient() calls per "
        "step (~5x backward cost). Adds grad_norm_number/operator/"
        "reduction/result to logged metrics. Recommended for probe/debug "
        "only — too expensive for 100K+ runs.",
    )
    return parser.parse_args()


def _sidecar_path(weights_path: str) -> str:
    """Sidecar JSON path for a given weights file (`*.weights.h5` -> `*.state.json`)."""
    return weights_path.replace(".weights.h5", ".state.json")


def save_training_state(
    weights_path: str,
    step: int,
    total_steps: int,
    best_loss: float,
    curriculum: bool,
    curriculum_cap: float,
) -> None:
    """Write sidecar JSON next to a weights checkpoint.

    DECISION D-006: checkpoint state is stored as a plain JSON sidecar
    (not Keras optimizer serialization) so resume survives Keras format
    drift and is backend-agnostic. Adam momentum is not preserved
    (re-warms in ~100 steps).
    """
    state = {
        "step": int(step),
        "total_steps": int(total_steps),
        "best_loss": float(best_loss),
        "curriculum": bool(curriculum),
        "curriculum_cap": float(curriculum_cap),
    }
    with open(_sidecar_path(weights_path), "w") as f:
        json.dump(state, f, indent=2)


def load_training_state(weights_path: str) -> dict | None:
    """Load sidecar JSON for a weights checkpoint. Returns None if missing."""
    p = _sidecar_path(weights_path)
    if not Path(p).exists():
        return None
    with open(p, "r") as f:
        return json.load(f)


def create_optimizer(
    lr: float,
    weight_decay: float,
    clip_norm: float,
    warmup_steps: int,
    total_steps: int,
) -> keras.optimizers.Optimizer:
    """Create Adam optimizer with linear warmup + cosine decay."""
    primary_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr,
        decay_steps=max(1, total_steps - warmup_steps),
        alpha=0.0,
    )
    lr_schedule = WarmupSchedule(
        warmup_steps=warmup_steps,
        primary_schedule=primary_schedule,
        warmup_start_lr=1e-7,
    )
    optimizer = keras.optimizers.Adam(
        learning_rate=lr_schedule,
        weight_decay=weight_decay,
        global_clipnorm=clip_norm,
    )
    return optimizer


def _classify_expressions(expressions: list[str]) -> dict[str, list[int]]:
    """
    Classify which operators appear in each expression.

    Returns a dict mapping operator symbol to list of sample indices
    that contain that operator.
    """
    op_indices: dict[str, list[int]] = {"+": [], "-": [], "*": [], "/": []}
    for i, expr in enumerate(expressions):
        for op in op_indices:
            if op in expr:
                op_indices[op].append(i)
    return op_indices


def _make_compiled_train_fn(
    model: NAM,
    optimizer: keras.optimizers.Optimizer,
    max_act_steps: int,
    ponder_cost: float,
    result_loss_weight: float,
    valid_loss_weight: float,
    w_number: float = 5.0,
    w_operator: float = 3.0,
    w_reduction: float = 1.0,
    number_loss_delta: float = 0.1,
    log_grad_norms: bool = False,
):
    """
    Create a compiled (tf.function) multi-task training function.

    Computes a combined loss supervising all sub-skills:
    - S1: Number extraction (Huber on relative error of left_val/right_val)
    - S2: Operator classification (CE on op_logits vs true operator)
    - S3: Reduction target (CE on reduction_weights vs true operator position)
    - S5: Final result (Huber on result vs true result, log-space)
    - Validity (BCE) and ponder cost (step penalty)

    All computed in ONE forward pass. ONE backward pass. ONE optimizer step.

    DECISION D-003 (plan_2026-04-09_aa9cac24): L_number uses relative-error
    Huber loss (per README §6), not log-compressed MSE. Rel-error directly
    optimizes the step_1% / step_10% metrics we care about; log-MSE plateaued
    around RMSE ~0.45-0.9 in the README 100K run, too coarse for the target.
    Log-compressed MSE is kept as a diagnostic metric in train_step().

    DECISION D-005: if ``log_grad_norms`` is True, a persistent tape is used
    with 4 extra ``tape.gradient()`` calls per step to compute per-sub-skill
    gradient norms. Cost is ~5x backward pass; only enable for probe/debug.
    """
    max_expression_len = model.config.max_expression_len

    @tf.function(jit_compile=False)
    def _compiled_step(
        input_ids,
        targets,
        target_validity,
        true_left,
        true_right,
        true_op_index,
        true_op_position,
    ):
        batch_data = {"input_ids": input_ids}

        with tf.GradientTape(persistent=log_grad_norms) as tape:
            carry = model.initial_carry(batch_data)

            # Pre-encode once (tree encoder), then reuse across ACT steps
            encoded, mask, _ = model._encode(input_ids, training=True)
            batch_data = {"input_ids": input_ids, "encoded": encoded, "mask": mask}

            # Fixed-length ACT unroll with Python range
            all_results = []
            all_step_results = []
            all_valids = []
            all_q_halt = []
            all_q_cont = []
            all_op_logits = []
            all_left_vals = []
            all_right_vals = []
            all_reduction_weights = []

            for step_idx in range(max_act_steps):
                carry, outputs = model(carry, batch_data, training=True)
                batch_data = outputs["batch"]
                all_results.append(outputs["result"])
                all_step_results.append(outputs["step_result"])
                all_valids.append(outputs["valid"])
                all_q_halt.append(outputs["q_halt_logits"])
                all_q_cont.append(outputs["q_continue_logits"])
                all_op_logits.append(outputs["op_logits"])
                all_left_vals.append(outputs["step_left_val"])
                all_right_vals.append(outputs["step_right_val"])
                all_reduction_weights.append(outputs["reduction_weights"])

            # --- Compute multi-task losses ---
            L_result = tf.constant(0.0)
            L_valid = tf.constant(0.0)
            L_number = tf.constant(0.0)
            L_operator = tf.constant(0.0)
            L_reduction = tf.constant(0.0)

            # One-hot encode operator targets for CE
            true_op_onehot = tf.one_hot(true_op_index, 4)
            # One-hot encode operator position for reduction CE
            true_pos_onehot = tf.one_hot(
                true_op_position, max_expression_len
            )

            # Log-compress: sign(x) * log(1 + |x|) — scale-invariant for
            # numbers spanning 1-digit to 10-digit. Defined once, reused.
            def _log_c(x):
                return tf.sign(x) * tf.math.log1p(tf.abs(x))

            for i in range(max_act_steps):
                # S5: Final result loss (Huber in log-space) — trains
                # result_head only (stop_gradient prevents backprop).
                L_result += tf.reduce_mean(
                    keras.losses.huber(
                        _log_c(targets), _log_c(all_results[i]), delta=2.0,
                    )
                )

                # Validity BCE — clamp predictions to prevent log(0) explosion
                clamped_valid = tf.clip_by_value(all_valids[i], 1e-7, 1.0 - 1e-7)
                L_valid += tf.reduce_mean(
                    keras.losses.binary_crossentropy(
                        target_validity, clamped_valid
                    )
                )

                # S1: Number extraction — Huber on relative error.
                # rel_err = (pred - target) / (|target| + 1.0). The +1 floor
                # prevents blow-up at target=0. Huber's quadratic region
                # covers small errors; linear region bounds large errors.
                # 0.5 coefficient on each operand normalizes to the
                # per-operand mean (matches the logged number_mse metric).
                zero_target = tf.zeros_like(all_left_vals[i])
                left_rel = (all_left_vals[i] - true_left) / (
                    tf.abs(true_left) + 1.0
                )
                right_rel = (all_right_vals[i] - true_right) / (
                    tf.abs(true_right) + 1.0
                )
                L_number += 0.5 * tf.reduce_mean(
                    keras.losses.huber(
                        zero_target, left_rel, delta=number_loss_delta,
                    )
                ) + 0.5 * tf.reduce_mean(
                    keras.losses.huber(
                        zero_target, right_rel, delta=number_loss_delta,
                    )
                )

                # S2: Operator classification CE — clamp logits to [-30, 30]
                # to prevent extreme confidence that causes CE explosion
                # when the prediction is wrong (CE of a 99.999% wrong
                # prediction = ~11.5, but unclamped can reach thousands).
                clamped_op_logits = tf.clip_by_value(
                    all_op_logits[i], -30.0, 30.0
                )
                L_operator += tf.reduce_mean(
                    keras.losses.categorical_crossentropy(
                        true_op_onehot, clamped_op_logits, from_logits=True
                    )
                )

                # S3: Reduction target CE — clamp probs away from 0
                clamped_rw = tf.clip_by_value(
                    all_reduction_weights[i], 1e-7, 1.0
                )
                L_reduction += tf.reduce_mean(
                    keras.losses.categorical_crossentropy(
                        true_pos_onehot, clamped_rw
                    )
                )

            # Normalize by number of steps
            n_steps = tf.cast(max_act_steps, tf.float32)
            L_result = L_result / n_steps
            L_valid = L_valid / n_steps
            L_number = L_number / n_steps
            L_operator = L_operator / n_steps
            L_reduction = L_reduction / n_steps

            # Ponder cost
            avg_steps = tf.reduce_mean(tf.cast(carry["steps"], tf.float32))
            L_ponder = ponder_cost * avg_steps

            # Combined loss
            total_loss = (
                w_number * L_number
                + w_operator * L_operator
                + w_reduction * L_reduction
                + result_loss_weight * L_result
                + valid_loss_weight * L_valid
                + L_ponder
            )

        grads = tape.gradient(total_loss, model.trainable_variables)

        # Per-sub-skill gradient norms (DECISION D-005, opt-in via flag).
        # The persistent tape lets us compute 4 extra gradient calls
        # to measure each sub-skill's contribution separately.
        # Cost: ~5x backward pass — only use for probe/debug.
        if log_grad_norms:
            num_grads = tape.gradient(L_number, model.trainable_variables)
            op_grads = tape.gradient(L_operator, model.trainable_variables)
            red_grads = tape.gradient(L_reduction, model.trainable_variables)
            res_grads = tape.gradient(L_result, model.trainable_variables)
            del tape

            def _global_norm(g_list):
                filtered = [g for g in g_list if g is not None]
                if not filtered:
                    return tf.constant(0.0)
                return tf.linalg.global_norm(filtered)

            grad_norm_number = _global_norm(num_grads)
            grad_norm_operator = _global_norm(op_grads)
            grad_norm_reduction = _global_norm(red_grads)
            grad_norm_result = _global_norm(res_grads)
        else:
            grad_norm_number = tf.constant(0.0)
            grad_norm_operator = tf.constant(0.0)
            grad_norm_reduction = tf.constant(0.0)
            grad_norm_result = tf.constant(0.0)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Return everything needed for metrics
        last_result = all_results[max_act_steps - 1]
        last_step_result = all_step_results[max_act_steps - 1]
        last_valid = all_valids[max_act_steps - 1]
        last_op_logits = all_op_logits[max_act_steps - 1]
        last_left_val = all_left_vals[max_act_steps - 1]
        last_right_val = all_right_vals[max_act_steps - 1]
        last_reduction_weights = all_reduction_weights[max_act_steps - 1]
        grad_norms = tf.stack([
            tf.norm(g) for g in grads if g is not None
        ])

        return {
            "total_loss": total_loss,
            "L_number": L_number,
            "L_operator": L_operator,
            "L_reduction": L_reduction,
            "L_result": L_result,
            "L_valid": L_valid,
            "last_result": last_result,
            "last_step_result": last_step_result,
            "last_valid": last_valid,
            "last_op_logits": last_op_logits,
            "last_left_val": last_left_val,
            "last_right_val": last_right_val,
            "last_reduction_weights": last_reduction_weights,
            "avg_steps": avg_steps,
            "grad_norm_mean": tf.reduce_mean(grad_norms),
            "grad_norm_max": tf.reduce_max(grad_norms),
            "grad_norm_number": grad_norm_number,
            "grad_norm_operator": grad_norm_operator,
            "grad_norm_reduction": grad_norm_reduction,
            "grad_norm_result": grad_norm_result,
        }

    return _compiled_step


def train_step(
    compiled_fn,
    input_ids: tf.Tensor,
    targets: tf.Tensor,
    target_validity: tf.Tensor,
    labels: dict,
    expressions: list[str],
) -> dict:
    """
    Execute one training step using the compiled function, then compute metrics.

    :param compiled_fn: Compiled training function from _make_compiled_train_fn.
    :param input_ids: Token IDs (B, L).
    :param targets: Ground truth results (B, 1).
    :param target_validity: Ground truth validity (B, 1).
    :param labels: Structured labels dict with left_operand, right_operand,
        operator_index, operator_position.
    :param expressions: List of expression strings (for per-operator metrics).
    :return: Dict of loss values and metrics.
    """
    raw = compiled_fn(
        input_ids,
        targets,
        target_validity,
        labels["left_operand"],
        labels["right_operand"],
        labels["operator_index"],
        labels["operator_position"],
    )

    # --- Compute metrics from raw tensors ---
    pred_result = raw["last_result"].numpy()
    true_result = targets.numpy()
    pred_valid = raw["last_valid"].numpy()
    true_valid = target_validity.numpy()
    op_logits = raw["last_op_logits"].numpy()

    # Relative error (result_head output)
    abs_true = np.abs(true_result) + 1e-8
    per_sample_rel_error = np.abs(pred_result - true_result) / abs_true
    rel_error = float(np.mean(per_sample_rel_error))

    # Exact accuracy (within 1% relative error)
    per_sample_exact = per_sample_rel_error < 0.01
    exact_acc = float(np.mean(per_sample_exact))

    # Step result metrics (cell's direct arithmetic output — the true
    # end-to-end pipeline: number_head → fixed_arithmetic → result)
    pred_step_result = raw["last_step_result"].numpy()
    per_sample_step_rel_error = np.abs(pred_step_result - true_result) / abs_true
    step_rel_error = float(np.mean(per_sample_step_rel_error))
    step_exact_acc = float(np.mean(per_sample_step_rel_error < 0.01))
    step_acc_5pct = float(np.mean(per_sample_step_rel_error < 0.05))
    step_acc_10pct = float(np.mean(per_sample_step_rel_error < 0.10))

    # Validity accuracy
    pred_valid_binary = (pred_valid > 0.5).astype(np.float32)
    valid_acc = float(np.mean(np.equal(pred_valid_binary, true_valid)))

    # Operator selection entropy
    op_probs = np.exp(op_logits - np.max(op_logits, axis=-1, keepdims=True))
    op_probs = op_probs / np.sum(op_probs, axis=-1, keepdims=True)
    op_entropy = float(np.mean(-np.sum(op_probs * np.log(op_probs + 1e-10), axis=-1)))
    op_entropy_normalized = op_entropy / np.log(4)

    # Operator classification accuracy (S2)
    pred_op = np.argmax(op_logits, axis=-1)
    true_op = labels["operator_index"].numpy() if hasattr(labels["operator_index"], "numpy") else labels["operator_index"]
    operator_acc = float(np.mean(pred_op == true_op))

    # Number extraction relative error (S1)
    pred_left = raw["last_left_val"].numpy()
    pred_right = raw["last_right_val"].numpy()
    true_left = labels["left_operand"].numpy() if hasattr(labels["left_operand"], "numpy") else labels["left_operand"]
    true_right = labels["right_operand"].numpy() if hasattr(labels["right_operand"], "numpy") else labels["right_operand"]
    # Log-compressed MSE (matches training loss)
    def _log_c(x):
        return np.sign(x) * np.log1p(np.abs(x))
    left_mse = float(np.mean(np.square(_log_c(pred_left) - _log_c(true_left))))
    right_mse = float(np.mean(np.square(_log_c(pred_right) - _log_c(true_right))))
    number_mse = (left_mse + right_mse) / 2.0

    # Reduction accuracy (S3)
    pred_reduction = raw["last_reduction_weights"].numpy()
    pred_reduction_pos = np.argmax(pred_reduction, axis=-1)
    true_reduction_pos = labels["operator_position"].numpy() if hasattr(labels["operator_position"], "numpy") else labels["operator_position"]
    reduction_acc = float(np.mean(pred_reduction_pos == true_reduction_pos))

    # Per-operator accuracy
    op_indices = _classify_expressions(expressions)
    per_op_rel_error = {}
    per_op_exact_acc = {}
    for op_sym, indices in op_indices.items():
        if indices:
            per_op_rel_error[op_sym] = float(np.mean(per_sample_rel_error[indices]))
            per_op_exact_acc[op_sym] = float(np.mean(per_sample_exact[indices]))
        else:
            per_op_rel_error[op_sym] = None
            per_op_exact_acc[op_sym] = None

    return {
        "total_loss": float(raw["total_loss"].numpy()),
        "L_number": float(raw["L_number"].numpy()),
        "L_operator": float(raw["L_operator"].numpy()),
        "L_reduction": float(raw["L_reduction"].numpy()),
        "L_result": float(raw["L_result"].numpy()),
        "L_valid": float(raw["L_valid"].numpy()),
        "rel_error": rel_error,
        "exact_acc": exact_acc,
        "step_exact_acc": step_exact_acc,
        "step_acc_5pct": step_acc_5pct,
        "step_acc_10pct": step_acc_10pct,
        "step_rel_error": step_rel_error,
        "valid_acc": valid_acc,
        "operator_acc": operator_acc,
        "number_mse": number_mse,
        "left_mse": left_mse,
        "right_mse": right_mse,
        "reduction_acc": reduction_acc,
        "avg_steps": float(raw["avg_steps"].numpy()),
        "grad_norm_mean": float(raw["grad_norm_mean"].numpy()),
        "grad_norm_max": float(raw["grad_norm_max"].numpy()),
        "grad_norm_number": float(raw["grad_norm_number"].numpy()),
        "grad_norm_operator": float(raw["grad_norm_operator"].numpy()),
        "grad_norm_reduction": float(raw["grad_norm_reduction"].numpy()),
        "grad_norm_result": float(raw["grad_norm_result"].numpy()),
        "op_entropy": op_entropy,
        "op_entropy_normalized": op_entropy_normalized,
        "per_op_rel_error": per_op_rel_error,
        "per_op_exact_acc": per_op_exact_acc,
    }


def _eval_digit_matrix(
    model: NAM,
    tokenizer: "ArithmeticTokenizer",
    act_steps: int = 2,
    max_digits: int = 10,
    samples_per_cell: int = 8,
) -> None:
    """
    Evaluate model on a [1..max_digits] × [1..max_digits] × 4-ops grid.

    Prints a per-operator accuracy matrix showing where the model succeeds
    and fails across different operand sizes (10% relative tolerance).
    """
    ops_list = ["+", "-", "*", "/"]

    for op_sym in ops_list:
        # acc_matrix[d_left][d_right] = fraction correct
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
                    batch = {"input_ids": tf.constant(ids)}
                    carry = model.initial_carry(batch)
                    for _ in range(act_steps):
                        carry, out = model(carry, batch, training=False)
                        batch = out["batch"]
                        if tf.reduce_all(carry["halted"]):
                            break

                    # Use step_result (cell arithmetic output)
                    pred = float(out["step_result"][0, 0].numpy())
                    rel_err = abs(pred - true_val) / (abs(true_val) + 1e-8)
                    if rel_err < 0.10:
                        correct += 1
                    total += 1

                acc_matrix[d_left - 1, d_right - 1] = (
                    correct / total if total > 0 else 0.0
                )

        # Print matrix
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

    # GPU setup
    setup_gpu(args.gpu)

    # Setup output directory: results/nam_<variant>-<phase>_<timestamp>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"nam_{args.variant}-{args.phase}_{timestamp}"
    output_dir = Path(args.save_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results directory: {output_dir}")

    # Create model
    logger.info(f"Creating NAM variant={args.variant}")
    model = NAM.from_variant(args.variant)
    tokenizer = ArithmeticTokenizer(max_len=model.config.max_expression_len)

    # Build model with a dummy forward pass
    dummy_ids = np.zeros((1, model.config.max_expression_len), dtype=np.int32)
    dummy_batch = {"input_ids": tf.constant(dummy_ids)}
    carry = model.initial_carry(dummy_batch)
    _ = model(carry, dummy_batch, training=False)

    param_count = sum(
        np.prod(v.shape) for v in model.trainable_variables
    )
    logger.info(f"Model parameters: {param_count:,}")

    # Load checkpoint if provided
    start_step = 0
    best_loss = float("inf")
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        model.load_weights(args.checkpoint)

        # DECISION D-006: restore training state from sidecar JSON if present
        saved_state = load_training_state(args.checkpoint)
        if saved_state is not None:
            start_step = saved_state["step"]
            best_loss = saved_state["best_loss"]
            logger.info(
                f"Resume state loaded from sidecar: "
                f"start_step={start_step} best_loss={best_loss:.4f} "
                f"(saved curriculum={saved_state['curriculum']} "
                f"cap={saved_state['curriculum_cap']})"
            )
            if start_step >= args.steps:
                logger.warning(
                    f"Resume start_step={start_step} >= --steps={args.steps}; "
                    "nothing to do."
                )
        else:
            logger.warning(
                f"No sidecar state found at {_sidecar_path(args.checkpoint)}; "
                "resuming from step 0 (LR warmup + curriculum progress reset)."
            )

    # Create optimizer
    optimizer = create_optimizer(
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip_norm=args.clip_norm,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
    )

    # Restore optimizer step counter so the LR schedule picks up at the
    # correct point (the schedule is a pure function of optimizer.iterations).
    if start_step > 0:
        # Touch optimizer.iterations to force variable creation on first access.
        _ = optimizer.iterations
        optimizer.iterations.assign(start_step)
        logger.info(f"Restored optimizer.iterations = {start_step}")

    # Get curriculum config
    use_curriculum = args.curriculum
    expr_config = CURRICULUM[args.phase]
    if args.min_val is not None:
        expr_config.min_val = args.min_val
    if args.max_val is not None:
        expr_config.max_val = args.max_val

    # Determine ACT depth: use override or model default
    act_steps = args.act_steps or model.config.halt_max_steps

    if use_curriculum:
        level_names = [l.name for l in DIFFICULTY_LEVELS]
        logger.info(
            f"Training with smooth curriculum ({len(DIFFICULTY_LEVELS)} levels): "
            f"{level_names}, act_steps={act_steps}, "
            f"curriculum_cap={args.curriculum_cap}"
        )
    else:
        logger.info(
            f"Training phase={args.phase}, "
            f"ops={expr_config.min_ops}-{expr_config.max_ops}, "
            f"vals={expr_config.min_val}-{expr_config.max_val}, "
            f"operators={expr_config.operators}, "
            f"parens={expr_config.allow_parentheses}, "
            f"act_steps={act_steps}"
        )

    # Compile training function (traces graph once — may take 30-60s)
    logger.info(
        f"Compiling training graph with act_steps={act_steps}... "
        f"(this is a one-time cost, please wait)"
    )
    compile_start = time.time()
    compiled_fn = _make_compiled_train_fn(
        model=model,
        optimizer=optimizer,
        max_act_steps=act_steps,
        ponder_cost=args.ponder_cost,
        result_loss_weight=args.result_loss_weight,
        valid_loss_weight=args.valid_loss_weight,
        w_number=args.w_number,
        w_operator=args.w_operator,
        w_reduction=args.w_reduction,
        number_loss_delta=args.number_loss_delta,
        log_grad_norms=args.log_grad_norms,
    )
    if args.log_grad_norms:
        logger.info(
            "Per-sub-skill grad norm instrumentation ENABLED "
            "(persistent tape, ~5x backward cost)."
        )

    # Warm up: run one step to trigger tracing
    if use_curriculum:
        warmup_ids, warmup_t, warmup_v, _, warmup_labels = generate_curriculum_batch(
            args.batch_size, 0.0, tokenizer
        )
    else:
        warmup_ids, warmup_t, warmup_v, _, warmup_labels = generate_batch(
            args.batch_size, expr_config, tokenizer
        )
    compiled_fn(
        tf.constant(warmup_ids),
        tf.constant(warmup_t),
        tf.constant(warmup_v),
        tf.constant(warmup_labels["left_operand"]),
        tf.constant(warmup_labels["right_operand"]),
        tf.constant(warmup_labels["operator_index"]),
        tf.constant(warmup_labels["operator_position"]),
    )
    compile_elapsed = time.time() - compile_start
    logger.info(f"Graph compiled in {compile_elapsed:.1f}s. Training starts now.")

    # Training loop
    metrics_log = []
    start_time = time.time()

    for step in range(start_step + 1, args.steps + 1):
        # Generate batch — curriculum mode shifts difficulty over training
        if use_curriculum:
            # DECISION D-002: cap progress at --curriculum-cap (default 0.8)
            # to keep hardest levels mixed with easier data throughout
            # training — prevents late op_acc regression.
            progress = min(args.curriculum_cap, step / args.steps)
            input_ids, targets, validity, expressions, labels = \
                generate_curriculum_batch(
                    args.batch_size, progress, tokenizer
                )
        else:
            input_ids, targets, validity, expressions, labels = generate_batch(
                args.batch_size, expr_config, tokenizer
            )

        input_ids_tf = tf.constant(input_ids)
        targets_tf = tf.constant(targets)
        validity_tf = tf.constant(validity)
        labels_tf = {
            "left_operand": tf.constant(labels["left_operand"]),
            "right_operand": tf.constant(labels["right_operand"]),
            "operator_index": tf.constant(labels["operator_index"]),
            "operator_position": tf.constant(labels["operator_position"]),
        }

        # Train step
        metrics = train_step(
            compiled_fn=compiled_fn,
            input_ids=input_ids_tf,
            targets=targets_tf,
            target_validity=validity_tf,
            labels=labels_tf,
            expressions=expressions,
        )
        metrics["step"] = step
        metrics_log.append(metrics)

        # Log
        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed

            # Main metrics line
            logger.info(
                f"Step {step}/{args.steps} | "
                f"loss={metrics['total_loss']:.4f} | "
                f"num_mse={metrics['number_mse']:.4f} | "
                f"op={metrics['operator_acc']:.3f} | "
                f"red={metrics['reduction_acc']:.3f} | "
                f"step_1%={metrics['step_exact_acc']:.3f} "
                f"5%={metrics['step_acc_5pct']:.3f} "
                f"10%={metrics['step_acc_10pct']:.3f} | "
                f"{steps_per_sec:.1f} s/s"
            )

            # Per-loss breakdown
            logger.info(
                f"  losses: L_num={metrics['L_number']:.4f} "
                f"L_op={metrics['L_operator']:.4f} "
                f"L_red={metrics['L_reduction']:.4f} "
                f"L_res={metrics['L_result']:.4f} "
                f"L_val={metrics['L_valid']:.4f}"
            )

            # Numbers detail
            logger.info(
                f"  numbers: left_mse={metrics['left_mse']:.4f} "
                f"right_mse={metrics['right_mse']:.4f}"
            )

            # Curriculum distribution (if active)
            if use_curriculum:
                probs = _curriculum_probs(min(args.curriculum_cap, step / args.steps))
                dist_str = " ".join(
                    f"{DIFFICULTY_LEVELS[i].name}={probs[i]:.0%}"
                    for i in range(len(DIFFICULTY_LEVELS))
                    if probs[i] > 0.03
                )
                logger.info(f"  curriculum: {dist_str}")

            # Computation & gradient health
            logger.info(
                f"  depth: avg_steps={metrics['avg_steps']:.1f} "
                f"act_steps={act_steps} | "
                f"grad: mean={metrics['grad_norm_mean']:.4f} "
                f"max={metrics['grad_norm_max']:.4f} | "
                f"op_entropy={metrics['op_entropy']:.3f} "
                f"({metrics['op_entropy_normalized']:.1%} of max)"
            )

            # Per-sub-skill gradient norms (only populated when --log-grad-norms)
            if args.log_grad_norms:
                logger.info(
                    f"  grad per-skill: num={metrics['grad_norm_number']:.4f} "
                    f"op={metrics['grad_norm_operator']:.4f} "
                    f"red={metrics['grad_norm_reduction']:.4f} "
                    f"res={metrics['grad_norm_result']:.4f}"
                )

            # Per-operator breakdown
            op_parts = []
            for op_sym in ["+", "-", "*", "/"]:
                acc = metrics["per_op_exact_acc"].get(op_sym)
                err = metrics["per_op_rel_error"].get(op_sym)
                if acc is not None:
                    op_parts.append(f"'{op_sym}'={acc:.3f}/{err:.3f}")
                else:
                    op_parts.append(f"'{op_sym}'=n/a")
            logger.info(f"  per-op (acc/err): {' | '.join(op_parts)}")

            # Show a few examples
            sample_ids = input_ids[:3]
            sample_batch = {"input_ids": tf.constant(sample_ids)}
            sample_carry = model.initial_carry(sample_batch)
            for _ in range(act_steps):
                sample_carry, sample_out = model(
                    sample_carry, sample_batch, training=False
                )
                sample_batch = sample_out["batch"]
                if tf.reduce_all(sample_carry["halted"]):
                    break

            for i in range(min(3, len(expressions))):
                pred = float(sample_out["result"][i, 0].numpy())
                val = float(sample_out["valid"][i, 0].numpy())
                logger.info(
                    f"  {expressions[i]} = {targets[i, 0]:.4f} "
                    f"(pred={pred:.4f}, valid={val:.3f})"
                )

        # Digit accuracy matrix evaluation
        if step % args.eval_interval == 0:
            logger.info(f"  ── Digit accuracy matrix (step {step}) ──")
            _eval_digit_matrix(model, tokenizer, act_steps=act_steps, max_digits=10, samples_per_cell=4)

        # Track best loss on every step so the sidecar reflects the
        # current best at save time (not a stale value from the prior save).
        new_best = metrics["total_loss"] < best_loss
        if new_best:
            best_loss = metrics["total_loss"]

        # Save checkpoint
        if step % args.save_interval == 0:
            ckpt_dir = output_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            ckpt_path = ckpt_dir / f"step_{step:06d}.weights.h5"
            model.save_weights(str(ckpt_path))
            save_training_state(
                str(ckpt_path),
                step=step,
                total_steps=args.steps,
                best_loss=best_loss,
                curriculum=args.curriculum,
                curriculum_cap=args.curriculum_cap,
            )
            logger.info(f"Saved checkpoint: {ckpt_path}")

            if new_best:
                best_path = ckpt_dir / "best.weights.h5"
                model.save_weights(str(best_path))
                save_training_state(
                    str(best_path),
                    step=step,
                    total_steps=args.steps,
                    best_loss=best_loss,
                    curriculum=args.curriculum,
                    curriculum_cap=args.curriculum_cap,
                )
                logger.info(f"New best model saved: {best_path}")

    # Save final model and metrics
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    final_path = ckpt_dir / "final.weights.h5"
    model.save_weights(str(final_path))
    save_training_state(
        str(final_path),
        step=args.steps,
        total_steps=args.steps,
        best_loss=best_loss,
        curriculum=args.curriculum,
        curriculum_cap=args.curriculum_cap,
    )

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)

    # Save training config for reproducibility
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "variant": args.variant,
                "phase": args.phase,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "clip_norm": args.clip_norm,
                "ponder_cost": args.ponder_cost,
                "result_loss_weight": args.result_loss_weight,
                "valid_loss_weight": args.valid_loss_weight,
                "w_number": args.w_number,
                "w_operator": args.w_operator,
                "w_reduction": args.w_reduction,
                "number_loss_delta": args.number_loss_delta,
                "curriculum": args.curriculum,
                "curriculum_cap": args.curriculum_cap,
                "model_config": model.config.to_dict(),
            },
            f,
            indent=2,
        )

    logger.info(f"Training complete. Results in {output_dir}")
    logger.info(
        f"Final: loss={metrics_log[-1]['total_loss']:.4f}, "
        f"exact_acc={metrics_log[-1]['exact_acc']:.4f}"
    )


if __name__ == "__main__":
    main()
