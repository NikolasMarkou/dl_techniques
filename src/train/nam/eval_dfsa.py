"""Comprehensive eval for DFSA: single-op, multi-op, parenthesized, edge cases."""

import sys
import random
import numpy as np
import tensorflow as tf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dl_techniques.models.nam.tokenizer import ArithmeticTokenizer
from train.nam.train_dfsa import DifferentiableFSA
from train.nam.data_generator import _safe_eval


def eval_model(model, tokenizer, max_steps=3):
    """Run comprehensive eval, return (total_pass, total_tested, details)."""
    random.seed(42)
    np.random.seed(42)
    ops_list = ['+', '-', '*', '/']
    results = {}

    def check(expr, tol=0.01):
        tv, valid = _safe_eval(expr)
        if not valid or not np.isfinite(tv):
            return None
        ids = tokenizer.encode_batch([expr])
        out = model.multi_reduce(tf.constant(ids), max_steps=max_steps, training=False)
        pred = float(out[-1]['result'].numpy()[0, 0])
        err = abs(pred - tv) / (abs(tv) + 1e-8)
        return err < tol

    # ── TEST 1: Single-op (4 ops × 5 digit sizes × 15 samples = 300) ──
    name = "single-op"
    ok = total = 0
    for op in ops_list:
        for d in [1, 2, 3, 5, 8]:
            for _ in range(15):
                lo, hi = max(1, 10**(d-1)), 10**d - 1
                a, b = random.randint(lo, hi), random.randint(lo, hi)
                if op == '/' and b == 0: b = 1
                r = check(f"{a} {op} {b}")
                if r is not None:
                    total += 1
                    if r: ok += 1
    results[name] = (ok, total)

    # ── TEST 2: Two-op flat (16 combos × 20 = 320) ──
    name = "2-op-flat"
    ok = total = 0
    for op1 in ops_list:
        for op2 in ops_list:
            for _ in range(20):
                a, b, c = [random.randint(1, 99) for _ in range(3)]
                if op1 == '/' and b == 0: b = 1
                if op2 == '/' and c == 0: c = 1
                r = check(f"{a} {op1} {b} {op2} {c}")
                if r is not None:
                    total += 1
                    if r: ok += 1
    results[name] = (ok, total)

    # ── TEST 3: Three-op flat (200) ──
    name = "3-op-flat"
    ok = total = 0
    for _ in range(200):
        operands = [random.randint(1, 50) for _ in range(4)]
        operators = [random.choice(ops_list) for _ in range(3)]
        expr = str(operands[0])
        for i, op in enumerate(operators):
            expr += f" {op} {operands[i+1]}"
        r = check(expr)
        if r is not None:
            total += 1
            if r: ok += 1
    results[name] = (ok, total)

    # ── TEST 4: Parenthesized 2-op (16 combos × 20 = 320) ──
    name = "2-op-paren"
    ok = total = 0
    fails = []
    for op1 in ops_list:
        for op2 in ops_list:
            for _ in range(20):
                a, b, c = [random.randint(1, 99) for _ in range(3)]
                if op1 == '/' and b == 0: b = 1
                if op2 == '/' and c == 0: c = 1
                # Parenthesize first or second pair
                if random.random() < 0.5:
                    expr = f"({a} {op1} {b}) {op2} {c}"
                else:
                    expr = f"{a} {op1} ({b} {op2} {c})"
                tv, valid = _safe_eval(expr)
                if not valid or not np.isfinite(tv): continue
                ids = tokenizer.encode_batch([expr])
                out = model.multi_reduce(tf.constant(ids), max_steps=max_steps, training=False)
                pred = float(out[-1]['result'].numpy()[0, 0])
                err = abs(pred - tv) / (abs(tv) + 1e-8)
                total += 1
                if err < 0.01:
                    ok += 1
                elif len(fails) < 10:
                    fails.append(f"  {expr} = {tv:.4f} pred={pred:.4f} err={err:.1%}")
    results[name] = (ok, total)
    if fails:
        results["2-op-paren-fails"] = fails

    # ── TEST 5: Parenthesized 3-op (200) ──
    name = "3-op-paren"
    ok = total = 0
    fails = []
    for _ in range(200):
        operands = [random.randint(1, 50) for _ in range(4)]
        operators = [random.choice(ops_list) for _ in range(3)]
        # Random paren placement
        paren_idx = random.randint(0, 2)  # which op pair to wrap
        parts = []
        for i in range(4):
            if i == paren_idx: parts.append("(")
            parts.append(str(operands[i]))
            if i == paren_idx + 1: parts.append(")")
            if i < 3: parts.append(f" {operators[i]} ")
        expr = "".join(parts)
        tv, valid = _safe_eval(expr)
        if not valid or not np.isfinite(tv): continue
        ids = tokenizer.encode_batch([expr])
        out = model.multi_reduce(tf.constant(ids), max_steps=max_steps, training=False)
        pred = float(out[-1]['result'].numpy()[0, 0])
        err = abs(pred - tv) / (abs(tv) + 1e-8)
        total += 1
        if err < 0.01:
            ok += 1
        elif len(fails) < 10:
            fails.append(f"  {expr} = {tv:.4f} pred={pred:.4f} err={err:.1%}")
    results[name] = (ok, total)
    if fails:
        results["3-op-paren-fails"] = fails

    # ── TEST 6: PEMDAS vs paren ordering (must differ) ──
    name = "pemdas-vs-paren"
    cases = [
        ("3 + 5 * 2", "(3 + 5) * 2"),      # 13 vs 16
        ("10 - 2 * 3", "(10 - 2) * 3"),     # 4 vs 24
        ("8 / 4 + 1", "8 / (4 + 1)"),       # 3 vs 1.6
        ("6 + 3 * 4", "(6 + 3) * 4"),       # 18 vs 36
        ("20 - 5 * 3", "(20 - 5) * 3"),     # 5 vs 45
        ("100 / 5 - 3", "100 / (5 - 3)"),   # 17 vs 50
    ]
    ok = total = 0
    for flat, paren in cases:
        for expr in [flat, paren]:
            tv, _ = _safe_eval(expr)
            ids = tokenizer.encode_batch([expr])
            out = model.multi_reduce(tf.constant(ids), max_steps=max_steps, training=False)
            pred = float(out[-1]['result'].numpy()[0, 0])
            err = abs(pred - tv) / (abs(tv) + 1e-8)
            total += 1
            if err < 0.001:
                ok += 1
            else:
                print(f"  PEMDAS FAIL: {expr} = {tv:.4f} pred={pred:.4f} err={err:.1%}")
    results[name] = (ok, total)

    # ── TEST 7: Double parens and edge cases ──
    name = "edge-cases"
    edge = [
        "((3 + 5)) * 2",           # 16
        "(1 + 1) + (1 + 1)",       # 4
        "(99 * 99) + 1",           # 9802
        "1 + (0 * 99)",            # 1
        "(50 - 50) * 100",         # 0
        "1 * (2 + 3)",             # 5
        "(7 - 3) * (8 - 6)",       # 8
        "(100 / 10) / 2",          # 5
    ]
    ok = total = 0
    for expr in edge:
        tv, _ = _safe_eval(expr)
        ids = tokenizer.encode_batch([expr])
        out = model.multi_reduce(tf.constant(ids), max_steps=max_steps, training=False)
        pred = float(out[-1]['result'].numpy()[0, 0])
        err = abs(pred - tv) / (abs(tv) + 1e-8)
        total += 1
        if err < 0.01:
            ok += 1
        else:
            print(f"  EDGE FAIL: {expr} = {tv:.4f} pred={pred:.4f} err={err:.1%}")
    results[name] = (ok, total)

    # ── Summary ──
    grand_ok = sum(v[0] for k, v in results.items() if isinstance(v, tuple))
    grand_total = sum(v[1] for k, v in results.items() if isinstance(v, tuple))
    return grand_ok, grand_total, results


if __name__ == "__main__":
    import glob
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="results/nam/dfsa_paren_iter1")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-tree-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--act-steps", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    from train.common import setup_gpu
    setup_gpu(args.gpu)

    model = DifferentiableFSA(
        hidden_size=args.hidden_size,
        max_expression_len=args.max_len,
        num_tree_layers=args.num_tree_layers,
        num_heads=args.num_heads,
    )
    tokenizer = ArithmeticTokenizer(max_len=args.max_len)
    dummy = np.zeros((1, args.max_len), dtype=np.int32)
    _ = model(tf.constant(dummy), training=False, max_steps=args.act_steps)

    ckpt = args.checkpoint
    if ckpt is None:
        ckpts = sorted(glob.glob(f"{args.save_dir}/**/checkpoints/*.h5", recursive=True))
        ckpt = ckpts[-1] if ckpts else None

    if ckpt:
        model.load_weights(ckpt)
        print(f"Loaded: {ckpt}")
    else:
        print("WARNING: No checkpoint found, evaluating random model")

    grand_ok, grand_total, results = eval_model(model, tokenizer, max_steps=args.act_steps)

    print()
    print("=" * 60)
    print("COMPREHENSIVE EVAL RESULTS")
    print("=" * 60)
    for name, val in results.items():
        if isinstance(val, tuple):
            ok, total = val
            pct = ok / total * 100 if total > 0 else 0
            status = "PASS" if pct >= 99 else "FAIL"
            print(f"  {name:20s}: {ok:>4}/{total:<4} ({pct:5.1f}%) [{status}]")
        elif isinstance(val, list):
            print(f"  {name}:")
            for f in val:
                print(f"    {f}")
    print("-" * 60)
    pct = grand_ok / grand_total * 100
    print(f"  {'TOTAL':20s}: {grand_ok:>4}/{grand_total:<4} ({pct:5.1f}%)")
    print("=" * 60)
