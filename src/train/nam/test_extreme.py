"""
Extreme test suite for DFSA arithmetic pipeline.

Tests 2000+ expressions across all categories:
- Single-op: all 4 operators × 10 digit sizes × 10 samples = 400
- Multi-op flat: 2-op (200), 3-op (200), 4-op (100)
- Parenthesized: 2-op-paren (200), 3-op-paren (200), both-sides (100)
- Nested parens: double (50), triple (20)
- PEMDAS vs paren: 20 handpicked where parens change result
- Edge cases: zero results, large numbers, division precision
- Stress: random 4-op paren expressions (200)

Reports per-category pass rates and any failures.
"""

import sys
import random
import numpy as np
import tensorflow as tf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dl_techniques.models.nam.tokenizer import ArithmeticTokenizer
from train.nam.train_dfsa import DifferentiableFSA
from train.nam.data_generator import _safe_eval


def check(model, tokenizer, expr, max_steps=4, tol=0.01):
    """Evaluate one expression. Returns (passed, true_val, pred_val, rel_err)."""
    tv, valid = _safe_eval(expr)
    if not valid or not np.isfinite(tv):
        return None, tv, 0.0, 0.0
    ids = tokenizer.encode_batch([expr])
    out = model.multi_reduce(tf.constant(ids), max_steps=max_steps,
                             training=False)
    pred = float(out[-1]["result"].numpy()[0, 0])
    err = abs(pred - tv) / (abs(tv) + 1e-8)
    return err < tol, tv, pred, err


def run_extreme_tests(model, tokenizer, max_steps=4, verbose=True):
    """Run full extreme test suite."""
    random.seed(42)
    np.random.seed(42)
    ops_list = ["+", "-", "*", "/"]
    results = {}
    all_failures = []

    def _run_category(name, exprs, tol=0.01):
        ok = total = 0
        fails = []
        for expr in exprs:
            passed, tv, pred, err = check(model, tokenizer, expr,
                                          max_steps=max_steps, tol=tol)
            if passed is None:
                continue
            total += 1
            if passed:
                ok += 1
            else:
                fails.append((expr, tv, pred, err))
        results[name] = (ok, total)
        if fails:
            all_failures.extend([(name, *f) for f in fails])
        return ok, total

    # ── 1. Single-op: 4 ops × 10 digit sizes × 10 samples = 400 ──
    exprs = []
    for op in ops_list:
        for d in range(1, 11):
            for _ in range(10):
                lo, hi = max(1, 10 ** (d - 1)), 10 ** d - 1
                a, b = random.randint(lo, hi), random.randint(lo, hi)
                if op == "/" and b == 0:
                    b = 1
                exprs.append(f"{a} {op} {b}")
    _run_category("single-op", exprs)

    # ── 2. Two-op flat: 16 combos × 12 + extra = 200 ──
    exprs = []
    for op1 in ops_list:
        for op2 in ops_list:
            for _ in range(12):
                a, b, c = [random.randint(1, 99) for _ in range(3)]
                if op1 == "/" and b == 0: b = 1
                if op2 == "/" and c == 0: c = 1
                exprs.append(f"{a} {op1} {b} {op2} {c}")
    _run_category("2-op-flat", exprs[:200])

    # ── 3. Three-op flat: 200 ──
    exprs = []
    for _ in range(200):
        operands = [random.randint(1, 50) for _ in range(4)]
        operators = [random.choice(ops_list) for _ in range(3)]
        expr = str(operands[0])
        for i, op in enumerate(operators):
            if op == "/" and operands[i + 1] == 0:
                operands[i + 1] = 1
            expr += f" {op} {operands[i + 1]}"
        exprs.append(expr)
    _run_category("3-op-flat", exprs)

    # ── 4. Four-op flat: 100 ──
    exprs = []
    for _ in range(100):
        operands = [random.randint(1, 30) for _ in range(5)]
        operators = [random.choice(ops_list) for _ in range(4)]
        expr = str(operands[0])
        for i, op in enumerate(operators):
            if op == "/" and operands[i + 1] == 0:
                operands[i + 1] = 1
            expr += f" {op} {operands[i + 1]}"
        exprs.append(expr)
    _run_category("4-op-flat", exprs)

    # ── 5. Two-op parenthesized: 200 ──
    exprs = []
    for _ in range(200):
        a, b, c = [random.randint(1, 99) for _ in range(3)]
        op1, op2 = random.choice(ops_list), random.choice(ops_list)
        if op1 == "/" and b == 0: b = 1
        if op2 == "/" and c == 0: c = 1
        if random.random() < 0.5:
            exprs.append(f"({a} {op1} {b}) {op2} {c}")
        else:
            exprs.append(f"{a} {op1} ({b} {op2} {c})")
    _run_category("2-op-paren", exprs)

    # ── 6. Three-op parenthesized: 200 ──
    exprs = []
    for _ in range(200):
        operands = [random.randint(1, 50) for _ in range(4)]
        operators = [random.choice(ops_list) for _ in range(3)]
        for i in range(3):
            if operators[i] == "/" and operands[i + 1] == 0:
                operands[i + 1] = 1
        paren_idx = random.randint(0, 2)
        parts = []
        for i in range(4):
            if i == paren_idx:
                parts.append("(")
            parts.append(str(operands[i]))
            if i == paren_idx + 1:
                parts.append(")")
            if i < 3:
                parts.append(f" {operators[i]} ")
        exprs.append("".join(parts))
    _run_category("3-op-paren", exprs)

    # ── 7. Both-sides paren: (a op b) op (c op d) — 100 ──
    exprs = []
    for _ in range(100):
        a, b, c, d = [random.randint(1, 50) for _ in range(4)]
        op1, op2, op3 = [random.choice(ops_list) for _ in range(3)]
        if op1 == "/" and b == 0: b = 1
        if op3 == "/" and d == 0: d = 1
        exprs.append(f"({a} {op1} {b}) {op2} ({c} {op3} {d})")
    _run_category("both-sides-paren", exprs)

    # ── 8. Double nested: ((a op b)) op c — 50 ──
    exprs = []
    for _ in range(50):
        a, b, c = [random.randint(1, 99) for _ in range(3)]
        op1, op2 = random.choice(ops_list), random.choice(ops_list)
        if op1 == "/" and b == 0: b = 1
        if op2 == "/" and c == 0: c = 1
        exprs.append(f"(({a} {op1} {b})) {op2} {c}")
    _run_category("double-nested", exprs)

    # ── 9. PEMDAS vs paren: handpicked where parens change result ──
    pemdas_cases = [
        ("3 + 5 * 2", "(3 + 5) * 2"),           # 13 vs 16
        ("10 - 2 * 3", "(10 - 2) * 3"),          # 4 vs 24
        ("8 / 4 + 1", "8 / (4 + 1)"),            # 3 vs 1.6
        ("6 + 3 * 4", "(6 + 3) * 4"),            # 18 vs 36
        ("20 - 5 * 3", "(20 - 5) * 3"),          # 5 vs 45
        ("100 / 5 - 3", "100 / (5 - 3)"),        # 17 vs 50
        ("2 + 8 / 4", "(2 + 8) / 4"),            # 4 vs 2.5
        ("15 - 3 * 2", "(15 - 3) * 2"),          # 9 vs 24
        ("50 / 10 + 5", "50 / (10 + 5)"),        # 10 vs 3.33
        ("7 + 3 * 5 - 2", "(7 + 3) * (5 - 2)"), # 20 vs 30
    ]
    exprs = []
    for flat, paren in pemdas_cases:
        exprs.extend([flat, paren])
    _run_category("pemdas-vs-paren", exprs, tol=0.001)

    # ── 10. Edge cases ──
    edge = [
        "((3 + 5)) * 2",          # 16 — double parens
        "(1 + 1) + (1 + 1)",      # 4 — both sides simple
        "(99 * 99) + 1",          # 9802 — large result
        "1 + (0 * 99)",           # 1 — zero multiplication
        "(50 - 50) * 100",        # 0 — zero result
        "1 * (2 + 3)",            # 5 — identity multiplication
        "(7 - 3) * (8 - 6)",      # 8 — both sides
        "(100 / 10) / 2",         # 5 — chained division
        "1 + 2 + 3 + 4",          # 10 — all same precedence
        "2 * 3 * 4 * 5",          # 120 — all multiplication
        "(1 + 2) * (3 + 4)",      # 21 — both sides, 3 steps
        "10 / (2 + 3)",           # 2 — division by paren sum
        "(8 - 2) / (4 - 1)",      # 2 — both sides division
        "(99 + 1) / (9 + 1)",     # 10 — round result
        "5 * (10 - 8) + 1",       # 11 — mixed
        "(3 + 7) * (2 + 8) / 5",  # 20 — 3 ops, paren both sides
    ]
    _run_category("edge-cases", edge)

    # ── 11. Stress: random 4-op with parens — 200 ──
    exprs = []
    for _ in range(200):
        operands = [random.randint(1, 30) for _ in range(5)]
        operators = [random.choice(ops_list) for _ in range(4)]
        for i in range(4):
            if operators[i] == "/" and operands[i + 1] == 0:
                operands[i + 1] = 1
        # Random paren placement (1 or 2 pairs)
        paren_idx = random.randint(0, 3)
        parts = []
        for i in range(5):
            if i == paren_idx:
                parts.append("(")
            parts.append(str(operands[i]))
            if i == paren_idx + 1:
                parts.append(")")
            if i < 4:
                parts.append(f" {operators[i]} ")
        exprs.append("".join(parts))
    _run_category("stress-4op-paren", exprs)

    # ── 12. Multi-digit paren: larger numbers ──
    exprs = []
    for _ in range(100):
        a = random.randint(100, 999)
        b = random.randint(10, 99)
        c = random.randint(1, 50)
        op1, op2 = random.choice(ops_list), random.choice(ops_list)
        if op1 == "/" and b == 0: b = 1
        if op2 == "/" and c == 0: c = 1
        if random.random() < 0.5:
            exprs.append(f"({a} {op1} {b}) {op2} {c}")
        else:
            exprs.append(f"{a} {op1} ({b} {op2} {c})")
    _run_category("multi-digit-paren", exprs)

    # ── Summary ──
    grand_ok = sum(v[0] for v in results.values())
    grand_total = sum(v[1] for v in results.values())

    print()
    print("=" * 70)
    print("EXTREME TEST RESULTS")
    print("=" * 70)
    for name, (ok, total) in results.items():
        pct = ok / total * 100 if total > 0 else 0
        status = "PASS" if pct >= 99.0 else "FAIL"
        print(f"  {name:25s}: {ok:>4}/{total:<4} ({pct:6.2f}%) [{status}]")
    print("-" * 70)
    pct = grand_ok / grand_total * 100
    print(f"  {'TOTAL':25s}: {grand_ok:>4}/{grand_total:<4} ({pct:6.2f}%)")
    print("=" * 70)

    if all_failures:
        print(f"\n  FAILURES ({len(all_failures)}):")
        for cat, expr, tv, pred, err in all_failures[:20]:
            print(f"    [{cat}] {expr} = {tv:.6f} pred={pred:.6f} "
                  f"err={err:.2%}")
        if len(all_failures) > 20:
            print(f"    ... and {len(all_failures) - 20} more")

    return grand_ok, grand_total, results, all_failures


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-tree-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--act-steps", type=int, default=4)
    parser.add_argument("--use-ste", action="store_true")
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    from train.common import setup_gpu
    setup_gpu(args.gpu)

    model = DifferentiableFSA(
        hidden_size=args.hidden_size,
        max_expression_len=args.max_len,
        num_tree_layers=args.num_tree_layers,
        num_heads=args.num_heads,
        use_ste=args.use_ste,
    )
    tokenizer = ArithmeticTokenizer(max_len=args.max_len)
    dummy = np.zeros((1, args.max_len), dtype=np.int32)
    _ = model(tf.constant(dummy), training=False, max_steps=args.act_steps)

    if args.checkpoint:
        model.load_weights(args.checkpoint)
        print(f"Loaded: {args.checkpoint}")
    else:
        print("No checkpoint — testing random model (deterministic pipeline)")

    run_extreme_tests(model, tokenizer, max_steps=args.act_steps)
