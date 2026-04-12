"""
Train DFSA with Straight-Through Estimator (STE) for gradient flow.

The DFSA arithmetic pipeline is fully deterministic (100% accuracy).
This script enables gradient flow from a host loss through the arithmetic
module via STE on the op_classifier:

    Forward:  hard operator selection (token_id - 14) — 100% correct
    Backward: soft operator classification (Dense(4)) — gradients flow
              to tree encoder, token embedding, numeric projection

Three training phases (layered strategy):

    Phase 1: Train op_classifier only — the tree encoder and embeddings
             are frozen. Only the Dense(4) learns to match the hard
             operator from tree-encoded features. This bootstraps the
             soft path quality before using it for STE gradients.

    Phase 2: Unfreeze all — with STE active, host result loss flows
             gradients through the soft op_classifier path back to the
             tree encoder. The forward pass remains 100% correct.

    Phase 3: (Future) Embed in host transformer — the DFSA produces
             result_embedding that the host can fine-tune via STE
             gradient flow.

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m train.nam.train_dfsa_ste \\
        --hidden-size 256 --num-tree-layers 3 --num-heads 8 \\
        --max-len 128 --act-steps 4 \\
        --steps 20000 --batch-size 64 --lr 1e-4 \\
        --curriculum-cap 0.8 --gpu 1
"""

import argparse
import time
import random
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from keras import ops

from dl_techniques.models.nam.tokenizer import ArithmeticTokenizer
from dl_techniques.optimization.warmup_schedule import WarmupSchedule
from train.common import setup_gpu
from train.nam.train_dfsa import (
    DifferentiableFSA,
    _fixed_add,
    _fixed_subtract,
    _fixed_multiply,
    _fixed_divide,
)
from train.nam.data_generator import (
    generate_curriculum_batch,
    prepare_per_step_labels,
    _safe_eval,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DFSA with STE gradient flow"
    )
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-tree-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--act-steps", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--result-loss-weight", type=float, default=1.0)
    parser.add_argument("--w-operator", type=float, default=3.0)
    parser.add_argument("--curriculum-cap", type=float, default=0.8)
    parser.add_argument(
        "--phase1-steps", type=int, default=5000,
        help="Steps for phase 1 (op_classifier only). 0 to skip.",
    )
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args()


def _make_ste_train_fn(model, optimizer, result_loss_weight, w_operator,
                       max_act_steps, freeze_mask=None):
    """Compiled training function with STE gradient flow.

    The result loss gradient flows through the STE on op_classifier
    back to the tree encoder. The operator loss provides direct
    supervision to the soft classifier path.

    :param freeze_mask: List of bool, same length as trainable_variables.
        True = freeze (zero gradient). None = train all.
    """
    train_vars = model.trainable_variables
    max_expression_len = model.max_expression_len

    @tf.function(jit_compile=False)
    def _step(
        input_ids,
        targets,
        per_step_op_index,
        per_step_op_position,
        per_step_valid_mask,
    ):
        with tf.GradientTape() as tape:
            def _log_c(x):
                return tf.sign(x) * tf.math.log1p(tf.abs(x))

            step_outputs = model.multi_reduce(
                input_ids, max_steps=max_act_steps, training=True
            )

            # Operator loss: trains the soft path of the STE to match
            # the hard operator. This is important — the soft path
            # quality determines the gradient quality for the host loss.
            L_operator = tf.constant(0.0)
            for step_idx in range(max_act_steps):
                out = step_outputs[step_idx]
                step_valid = per_step_valid_mask[:, step_idx]
                step_op_idx = per_step_op_index[:, step_idx]

                step_op_onehot = tf.one_hot(step_op_idx, 4)
                clamped_logits = tf.clip_by_value(
                    out["op_logits"], -30.0, 30.0
                )
                per_sample_op = keras.losses.categorical_crossentropy(
                    step_op_onehot, clamped_logits, from_logits=True
                )
                L_operator += tf.reduce_mean(per_sample_op * step_valid)
            L_operator = L_operator / tf.cast(max_act_steps, tf.float32)

            # Result loss: Huber in log-space. Gradient flows through
            # STE → op_classifier → tree encoder when use_ste=True.
            last_result = step_outputs[max_act_steps - 1]["result"]
            L_result = tf.reduce_mean(
                keras.losses.huber(
                    _log_c(targets), _log_c(last_result), delta=2.0
                )
            )

            total_loss = (
                w_operator * L_operator
                + result_loss_weight * L_result
            )

        grads = tape.gradient(total_loss, train_vars)
        # Replace None gradients with zeros
        grads = [
            tf.zeros_like(v) if g is None else g
            for g, v in zip(grads, train_vars)
        ]
        if freeze_mask is not None:
            grads = [
                tf.zeros_like(g) if frozen else g
                for g, frozen in zip(grads, freeze_mask)
            ]
        optimizer.apply_gradients(zip(grads, train_vars))

        grad_norms = tf.stack(
            [tf.norm(g) for g in grads if g is not None]
        )

        first_out = step_outputs[0]
        pred_op = tf.argmax(first_out["op_logits"], axis=-1)

        return {
            "total_loss": total_loss,
            "L_operator": L_operator,
            "L_result": L_result,
            "result": last_result,
            "pred_op": pred_op,
            "grad_mean": tf.reduce_mean(grad_norms),
            "grad_max": tf.reduce_max(grad_norms),
        }

    return _step


def _eval_quick(model, tokenizer):
    """Quick eval on key expression types. Returns (ok, total, details)."""
    random.seed(42)
    tests = {
        "single-op": [
            f"{random.randint(1,99)} {op} {random.randint(1,99)}"
            for op in ["+", "-", "*", "/"] for _ in range(10)
        ],
        "2-op-flat": [
            f"{random.randint(1,50)} {random.choice('+-*/')} "
            f"{random.randint(1,50)} {random.choice('+-*/')} "
            f"{random.randint(1,50)}"
            for _ in range(20)
        ],
        "2-op-paren": [
            f"({random.randint(1,50)} {random.choice('+-*/')} "
            f"{random.randint(1,50)}) {random.choice('+-*/')} "
            f"{random.randint(1,50)}"
            for _ in range(20)
        ],
        "3-op-paren": [
            f"({random.randint(1,30)} {random.choice('+-*/')} "
            f"{random.randint(1,30)}) {random.choice('+-*/')} "
            f"({random.randint(1,30)} {random.choice('+-*/')} "
            f"{random.randint(1,30)})"
            for _ in range(10)
        ],
    }
    results = {}
    grand_ok = grand_total = 0
    for name, exprs in tests.items():
        ok = total = 0
        for expr in exprs:
            tv, valid = _safe_eval(expr)
            if not valid or not np.isfinite(tv):
                continue
            ids = tokenizer.encode_batch([expr])
            out = model.multi_reduce(
                tf.constant(ids), max_steps=4, training=False
            )
            pred = float(out[-1]["result"].numpy()[0, 0])
            err = abs(pred - tv) / (abs(tv) + 1e-8)
            total += 1
            if err < 0.01:
                ok += 1
        results[name] = (ok, total)
        grand_ok += ok
        grand_total += total
    return grand_ok, grand_total, results


def main():
    args = parse_args()
    setup_gpu(args.gpu)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.save_dir) / f"dfsa_ste_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build model with STE enabled
    model = DifferentiableFSA(
        hidden_size=args.hidden_size,
        max_expression_len=args.max_len,
        num_tree_layers=args.num_tree_layers,
        num_heads=args.num_heads,
        use_ste=True,
    )
    tokenizer = ArithmeticTokenizer(max_len=args.max_len)

    # Warmup forward pass
    dummy = np.zeros((1, args.max_len), dtype=np.int32)
    _ = model(tf.constant(dummy), training=False, max_steps=args.act_steps)

    param_count = sum(np.prod(v.shape) for v in model.trainable_variables)
    logger.info(
        f"DFSA-STE model: hidden={args.hidden_size}, "
        f"act_steps={args.act_steps}, params={param_count:,}, use_ste=True"
    )

    # Verify 100% accuracy before training
    ok, total, _ = _eval_quick(model, tokenizer)
    logger.info(f"Pre-training accuracy: {ok}/{total} ({ok/total*100:.1f}%)")

    # Count gradient-receiving variables
    ids_t = tf.constant(tokenizer.encode_batch(["(3 + 5) * 2"]))
    with tf.GradientTape() as tape:
        out = model.multi_reduce(ids_t, max_steps=4, training=True)
        loss = tf.reduce_sum(out[-1]["result"])
    grads = tape.gradient(loss, model.trainable_variables)
    n_grad = sum(
        1 for g in grads
        if g is not None and tf.reduce_any(tf.not_equal(g, 0)).numpy()
    )
    logger.info(
        f"Gradient flow: {n_grad}/{len(model.trainable_variables)} "
        f"variables receive nonzero gradients"
    )

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

    # ── Phase 1: Train op_classifier only ────────────────────────────
    if args.phase1_steps > 0:
        logger.info(
            f"Phase 1: Training op_classifier only "
            f"({args.phase1_steps} steps)..."
        )
        op_classifier_var_ids = {
            id(v) for v in model.op_classifier.trainable_variables
        }
        freeze_mask = [
            id(v) not in op_classifier_var_ids
            for v in model.trainable_variables
        ]
        n_train = sum(1 for f in freeze_mask if not f)
        logger.info(
            f"  Phase 1 training {n_train} op_classifier variables, "
            f"freezing {len(freeze_mask) - n_train}"
        )

        compiled_fn = _make_ste_train_fn(
            model, optimizer, args.result_loss_weight, args.w_operator,
            args.act_steps, freeze_mask=freeze_mask,
        )

        # Warmup trace
        warmup_ids, warmup_t, _, warmup_exprs, _ = generate_curriculum_batch(
            args.batch_size, 0.0, tokenizer
        )
        ps = _build_per_step(warmup_exprs, tokenizer, args.act_steps,
                             warmup_ids)
        compiled_fn(
            tf.constant(warmup_ids), tf.constant(warmup_t),
            tf.constant(ps[1]), tf.constant(ps[2]), tf.constant(ps[3]),
        )
        logger.info("Phase 1 compiled.")

        for step in range(1, args.phase1_steps + 1):
            progress = min(args.curriculum_cap, step / args.steps)
            input_ids, targets, _, expressions, _ = \
                generate_curriculum_batch(args.batch_size, progress, tokenizer)
            ps = _build_per_step(expressions, tokenizer, args.act_steps,
                                 input_ids)
            raw = compiled_fn(
                tf.constant(input_ids), tf.constant(targets),
                tf.constant(ps[1]), tf.constant(ps[2]), tf.constant(ps[3]),
            )
            if step % args.log_interval == 0:
                logger.info(
                    f"  P1 step {step:>5d} | "
                    f"L_op={float(raw['L_operator'].numpy()):.4f} | "
                    f"L_res={float(raw['L_result'].numpy()):.4f} | "
                    f"grad={float(raw['grad_mean'].numpy()):.2f}"
                )

        logger.info("Phase 1 complete.")

    # ── Phase 2: Train all with STE gradient flow ────────────────────
    logger.info(
        f"Phase 2: Training all parameters with STE "
        f"({args.steps - args.phase1_steps} steps)..."
    )

    compiled_fn = _make_ste_train_fn(
        model, optimizer, args.result_loss_weight, args.w_operator,
        args.act_steps,
    )

    # Warmup trace for phase 2 (new graph with all vars trainable)
    warmup_ids, warmup_t, _, warmup_exprs, _ = generate_curriculum_batch(
        args.batch_size, 0.0, tokenizer
    )
    ps = _build_per_step(warmup_exprs, tokenizer, args.act_steps, warmup_ids)
    compiled_fn(
        tf.constant(warmup_ids), tf.constant(warmup_t),
        tf.constant(ps[1]), tf.constant(ps[2]), tf.constant(ps[3]),
    )
    logger.info("Phase 2 compiled.")

    start_time = time.time()
    for step in range(args.phase1_steps + 1, args.steps + 1):
        progress = min(args.curriculum_cap, step / args.steps)
        input_ids, targets, _, expressions, _ = \
            generate_curriculum_batch(args.batch_size, progress, tokenizer)
        ps = _build_per_step(expressions, tokenizer, args.act_steps,
                             input_ids)
        raw = compiled_fn(
            tf.constant(input_ids), tf.constant(targets),
            tf.constant(ps[1]), tf.constant(ps[2]), tf.constant(ps[3]),
        )

        pred_result = raw["result"].numpy()
        abs_true = np.abs(targets) + 1e-8
        per_sample_rel = np.abs(pred_result - targets) / abs_true

        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            sps = (step - args.phase1_steps) / elapsed if elapsed > 0 else 0
            logger.info(
                f"  P2 step {step:>5d} | "
                f"loss={float(raw['total_loss'].numpy()):.4f} | "
                f"L_op={float(raw['L_operator'].numpy()):.4f} | "
                f"L_res={float(raw['L_result'].numpy()):.4f} | "
                f"step_1%={float(np.mean(per_sample_rel < 0.01)):.3f} | "
                f"grad={float(raw['grad_mean'].numpy()):.2f} | "
                f"{sps:.1f} steps/s"
            )

        if step % args.eval_interval == 0:
            ok, total, details = _eval_quick(model, tokenizer)
            logger.info(f"  Eval: {ok}/{total} ({ok/total*100:.1f}%)")
            for name, (o, t) in details.items():
                logger.info(f"    {name}: {o}/{t}")

        if step % args.save_interval == 0:
            ckpt_dir = output_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            path = ckpt_dir / f"dfsa_ste_step{step:06d}.weights.h5"
            model.save_weights(str(path))
            logger.info(f"  Saved: {path}")

    # Final eval
    ok, total, details = _eval_quick(model, tokenizer)
    logger.info(f"Final accuracy: {ok}/{total} ({ok/total*100:.1f}%)")
    for name, (o, t) in details.items():
        logger.info(f"  {name}: {o}/{t}")

    # Save final weights
    final_path = output_dir / "dfsa_ste_final.weights.h5"
    model.save_weights(str(final_path))
    logger.info(f"Saved final weights: {final_path}")


def _build_per_step(expressions, tokenizer, act_steps, input_ids):
    """Build per-step label tensors."""
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


if __name__ == "__main__":
    main()
