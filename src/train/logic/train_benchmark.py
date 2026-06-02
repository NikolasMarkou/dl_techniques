"""
Benchmark + interpretability study of ``LearnableNeuralCircuit``.

Grid:

  Tasks (4):
    - parity_k6        — y = XOR(x_1..x_6). Linearly inseparable; the
                         classical XOR-generalization benchmark.
    - majority_k6      — y = 1 iff sum(x) >= 3. Linearly separable;
                         sanity-check task that linear-friendly methods
                         should solve trivially.
    - multiplexer_6    — 2 address bits + 4 data bits; y = d[addr]. Classical
                         UCI conditional-logic benchmark, famously hard for
                         small MLPs.
    - shift_xor_k8     — y[i] = x[i] XOR x[(i+1) mod 8]. Multi-output;
                         exercises ``selection_mode='per_channel'``.

  Models (per scalar task):
    - circuit          — Dense(channels)+LN -> LearnableNeuralCircuit(depth=2)
    - mlp_matched      — Dense-only baseline, param count within ~20% of circuit
    - mlp_large        — wider MLP (~3-4x circuit params)

  For shift_xor: circuit_per_channel vs mlp_large.

For every circuit run we ALSO run a **hard-extraction faithfulness test**:
after training, replace each inner op's ``operation_weights`` with a hard
one-hot at ``argmax`` and re-evaluate. Accuracy delta = how load-bearing the
soft mixture was.

Output:
  - ``results/logic_benchmark_<ts>/results.csv``  — one row per (task, model)
  - ``results/logic_benchmark_<ts>/report.md``    — human-readable narrative
  - All trained models saved under ``results/.../<task>_<model>.keras`` so a
    skeptic can reload and inspect.

Usage::

    MPLBACKEND=Agg .venv/bin/python -m train.logic.train_benchmark \\
        --gpu 1 --epochs 100

Plan: plan_2026-05-13_25774a34.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import keras
import numpy as np
from keras import ops

from dl_techniques.layers.logic import LearnableNeuralCircuit
from dl_techniques.layers.logic.arithmetic_operators import LearnableArithmeticOperator
from dl_techniques.layers.logic.logic_operators import LearnableLogicOperator
from dl_techniques.utils.logger import logger
from train.common import setup_gpu, set_seeds


# ---------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------

def gen_parity(n: int, k: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    x = rng.integers(0, 2, size=(n, k)).astype(np.float32)
    y = (x.sum(axis=1) % 2 == 1).astype(np.float32).reshape(-1, 1)
    return x, y


def gen_majority(n: int, k: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    x = rng.integers(0, 2, size=(n, k)).astype(np.float32)
    y = (x.sum(axis=1) >= (k / 2.0)).astype(np.float32).reshape(-1, 1)
    return x, y


def gen_multiplexer_6(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """6-bit MUX: bits 0-1 = address, bits 2-5 = data. y = d[addr]."""
    x = rng.integers(0, 2, size=(n, 6)).astype(np.float32)
    addr = (x[:, 0].astype(np.int32) + 2 * x[:, 1].astype(np.int32))  # 0..3
    data = x[:, 2:]  # (n, 4)
    y = data[np.arange(n), addr].reshape(-1, 1)
    return x, y


def gen_shift_xor(n: int, k: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """y[i] = x[i] XOR x[(i+1) mod k]. Multi-output."""
    x = rng.integers(0, 2, size=(n, k)).astype(np.float32)
    y = np.logical_xor(x.astype(bool), np.roll(x.astype(bool), -1, axis=1)).astype(np.float32)
    return x, y


TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "parity_k6": {
        "generator": lambda n, rng: gen_parity(n, 6, rng),
        "num_bits": 6, "num_outputs": 1, "multi_output": False,
        "description": "y = XOR(x_1..x_6) — linearly inseparable",
    },
    "majority_k6": {
        "generator": lambda n, rng: gen_majority(n, 6, rng),
        "num_bits": 6, "num_outputs": 1, "multi_output": False,
        "description": "y = 1 iff sum(x) >= 3 — linearly separable",
    },
    "multiplexer_6": {
        "generator": lambda n, rng: gen_multiplexer_6(n, rng),
        "num_bits": 6, "num_outputs": 1, "multi_output": False,
        "description": "2-bit addr + 4-bit data; y = d[addr] — conditional",
    },
    "shift_xor_k8": {
        "generator": lambda n, rng: gen_shift_xor(n, 8, rng),
        "num_bits": 8, "num_outputs": 8, "multi_output": True,
        "description": "y[i] = x[i] XOR x[(i+1) mod 8] — multi-output per_channel",
    },
}


# ---------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------

def build_circuit(
    num_bits: int,
    num_outputs: int,
    channels: int = 32,
    circuit_depth: int = 2,
    selection_mode: str = "global",
    lr: float = 3e-3,
) -> keras.Model:
    """Bit vector -> Dense embed -> LN -> LearnableNeuralCircuit -> Dense head.

    For multi-output tasks pass num_outputs > 1 and selection_mode='per_channel'.
    Arithmetic ops restricted to bounded set to avoid NaN at depth=2 (per
    LESSONS from plan_2026-05-13_d256b568).
    """
    inputs = keras.Input(shape=(num_bits,), name="bits")
    x = keras.layers.Dense(channels, activation="relu", name="embed")(inputs)
    x = keras.layers.LayerNormalization(name="embed_norm")(x)
    x = LearnableNeuralCircuit(
        circuit_depth=circuit_depth,
        num_logic_ops_per_depth=2,
        num_arithmetic_ops_per_depth=2,
        use_residual=True,
        use_layer_norm=True,
        selection_mode=selection_mode,
        arithmetic_op_types=["add", "max", "min"],
        name="neural_circuit",
    )(x)
    outputs = keras.layers.Dense(num_outputs, activation="sigmoid", name="head")(x)
    model = keras.Model(inputs, outputs, name=f"circuit_{num_bits}b_{num_outputs}o")
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mlp(
    num_bits: int,
    num_outputs: int,
    hidden_units: int,
    depth: int = 2,
    lr: float = 3e-3,
) -> keras.Model:
    """Plain MLP: depth-1 hidden Dense(relu) layers + sigmoid head."""
    inputs = keras.Input(shape=(num_bits,), name="bits")
    x = inputs
    for i in range(depth):
        x = keras.layers.Dense(hidden_units, activation="relu", name=f"hidden_{i}")(x)
    outputs = keras.layers.Dense(num_outputs, activation="sigmoid", name="head")(x)
    model = keras.Model(inputs, outputs, name=f"mlp_{hidden_units}h_{depth}d")
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def find_mlp_hidden_for_param_budget(
    num_bits: int, num_outputs: int, target_params: int, depth: int = 2
) -> int:
    """Binary-search a hidden_units value whose MLP has param count nearest
    to target_params. Returns hidden_units."""
    lo, hi = 1, 256
    best_h, best_diff = lo, float("inf")
    while lo <= hi:
        mid = (lo + hi) // 2
        m = build_mlp(num_bits, num_outputs, hidden_units=mid, depth=depth)
        p = m.count_params()
        diff = abs(p - target_params)
        if diff < best_diff:
            best_diff = diff
            best_h = mid
        if p < target_params:
            lo = mid + 1
        else:
            hi = mid - 1
    return best_h


# ---------------------------------------------------------------------
# Hard extraction (faithfulness test)
# ---------------------------------------------------------------------

def _iter_inner_ops(model: keras.Model):
    """Yield every LearnableLogicOperator and LearnableArithmeticOperator
    nested inside model (typically inside a LearnableNeuralCircuit)."""
    for layer in model._flatten_layers(include_self=True, recursive=True):
        if isinstance(layer, (LearnableLogicOperator, LearnableArithmeticOperator)):
            yield layer


# DECISION plan_2026-05-13_25774a34/D-002
# Hard-extraction snapshot pattern. Mutating `operation_weights` in place with
# a large-magnitude one-hot makes softmax numerically one-hot for any
# reasonable temperature (typical trained T in [0.5, 2.0]). We snapshot the
# original numpy array per op and restore via `.assign()` on the same
# keras.Variable instance — DO NOT use `set_weights()` (that re-bundles the
# whole layer's weights and breaks aliasing with the original instance).
_HARD_MAGNITUDE = 50.0


def extract_hard_inplace(model: keras.Model) -> List[Tuple[Any, np.ndarray]]:
    """Replace every inner op's operation_weights with a hard one-hot.

    Returns a list of (op_layer, original_weight_array) snapshots so the
    caller can call ``restore_soft_weights`` to undo.
    """
    snapshot: List[Tuple[Any, np.ndarray]] = []
    for op in _iter_inner_ops(model):
        if op.operation_weights is None:
            continue
        cur = np.array(ops.convert_to_numpy(op.operation_weights), copy=True)
        snapshot.append((op, cur))
        # operation_weights shape: (N,) global  OR  (C, N) per_channel.
        argmax = np.argmax(cur, axis=-1)
        hard = np.full_like(cur, -_HARD_MAGNITUDE)
        if cur.ndim == 1:
            hard[argmax] = _HARD_MAGNITUDE
        else:
            rows = np.arange(cur.shape[0])
            hard[rows, argmax] = _HARD_MAGNITUDE
        op.operation_weights.assign(hard)
    return snapshot


def restore_soft_weights(snapshot: List[Tuple[Any, np.ndarray]]) -> None:
    for op, original in snapshot:
        op.operation_weights.assign(original)


# ---------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------

def bitwise_accuracy(model: keras.Model, X: np.ndarray, Y: np.ndarray) -> float:
    """Bit-wise accuracy = fraction of correct bits across the test set."""
    pred = (model.predict(X, verbose=0) > 0.5).astype(np.float32)
    return float((pred == Y).mean())


def exact_enumeration_accuracy(
    model: keras.Model, task_name: str, num_bits: int
) -> Optional[float]:
    """For tasks where num_bits is small enough, enumerate all 2^k inputs."""
    if num_bits > 12:
        return None
    n = 1 << num_bits
    x = np.array(
        [[(i >> b) & 1 for b in range(num_bits)] for i in range(n)],
        dtype=np.float32,
    )
    gen = TASK_SPECS[task_name]["generator"]
    # Re-run the deterministic part of the generator on the enumerated x.
    if task_name == "parity_k6":
        y = (x.sum(axis=1) % 2 == 1).astype(np.float32).reshape(-1, 1)
    elif task_name == "majority_k6":
        y = (x.sum(axis=1) >= 3.0).astype(np.float32).reshape(-1, 1)
    elif task_name == "multiplexer_6":
        addr = (x[:, 0].astype(np.int32) + 2 * x[:, 1].astype(np.int32))
        y = x[np.arange(n), 2 + addr].reshape(-1, 1)
    elif task_name == "shift_xor_k8":
        y = np.logical_xor(x.astype(bool), np.roll(x.astype(bool), -1, axis=1)).astype(np.float32)
    else:
        return None
    pred = (model.predict(x, verbose=0) > 0.5).astype(np.float32)
    return float((pred == y).mean())


def roundtrip_check(model: keras.Model, X: np.ndarray, save_path: str) -> float:
    model.save(save_path)
    reloaded = keras.models.load_model(save_path)
    p1 = model.predict(X, verbose=0)
    p2 = reloaded.predict(X, verbose=0)
    return float(np.max(np.abs(p1 - p2)))


# ---------------------------------------------------------------------
# Single-run trainer
# ---------------------------------------------------------------------

def train_one(
    task_name: str,
    model_name: str,
    epochs: int,
    train_samples: int,
    test_samples: int,
    seed: int,
    results_dir: str,
) -> Dict[str, Any]:
    """Train one (task, model) combo end-to-end and return a metrics row."""
    spec = TASK_SPECS[task_name]
    gen = spec["generator"]
    num_bits = spec["num_bits"]
    num_outputs = spec["num_outputs"]

    set_seeds(seed)
    rng_train = np.random.default_rng(seed)
    rng_test = np.random.default_rng(seed + 1000)
    X_train, Y_train = gen(train_samples, rng_train)
    X_test, Y_test = gen(test_samples, rng_test)

    # Build target model.
    if model_name == "circuit":
        sel = "per_channel" if spec["multi_output"] else "global"
        model = build_circuit(num_bits, num_outputs, selection_mode=sel)
        circuit_params = model.count_params()
    elif model_name == "mlp_matched":
        # Param-match against a freshly-built circuit (cheap).
        ref = build_circuit(num_bits, num_outputs)
        target = ref.count_params()
        h = find_mlp_hidden_for_param_budget(num_bits, num_outputs, target)
        model = build_mlp(num_bits, num_outputs, hidden_units=h)
        circuit_params = None
    elif model_name == "mlp_large":
        model = build_mlp(num_bits, num_outputs, hidden_units=32)
        circuit_params = None
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    params = model.count_params()
    is_circuit = (model_name == "circuit")

    # Train.
    t0 = time.time()
    early = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=max(15, epochs // 5),
        restore_best_weights=True, verbose=0,
    )
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=epochs, batch_size=32, verbose=0,
        callbacks=[early],
    )
    wall_s = time.time() - t0
    epochs_used = len(history.history["loss"])

    # Evaluate.
    soft_test_acc = (
        bitwise_accuracy(model, X_test, Y_test)
        if spec["multi_output"] else
        float(model.evaluate(X_test, Y_test, verbose=0)[1])
    )
    exact_acc = exact_enumeration_accuracy(model, task_name, num_bits)

    # Save model + round-trip.
    save_path = os.path.join(results_dir, f"{task_name}_{model_name}.keras")
    rt_diff = roundtrip_check(model, X_test, save_path)

    # Hard-extraction faithfulness — circuit models only.
    hard_test_acc: Optional[float] = None
    hard_exact_acc: Optional[float] = None
    symbolic: Optional[str] = None
    if is_circuit:
        nc = model.get_layer("neural_circuit")
        try:
            symbolic = nc.to_symbolic(top_k=1)
        except Exception as e:
            symbolic = f"<to_symbolic failed: {e}>"
        snapshot = extract_hard_inplace(model)
        try:
            hard_test_acc = (
                bitwise_accuracy(model, X_test, Y_test)
                if spec["multi_output"] else
                float(model.evaluate(X_test, Y_test, verbose=0)[1])
            )
            hard_exact_acc = exact_enumeration_accuracy(model, task_name, num_bits)
        finally:
            restore_soft_weights(snapshot)
        # Sanity: round-trip again after restore to ensure no permanent mutation.
        rt2 = roundtrip_check(model, X_test, save_path)
        if rt2 > 1e-6:
            logger.warning(
                f"{task_name}/{model_name}: post-restore round-trip diff "
                f"{rt2:.2e} > 1e-6 (extraction restoration leaked)."
            )

    return {
        "task": task_name,
        "model": model_name,
        "params": params,
        "epochs_used": epochs_used,
        "wall_s": round(wall_s, 2),
        "soft_test_acc": round(soft_test_acc, 4),
        "exact_acc": round(exact_acc, 4) if exact_acc is not None else None,
        "hard_test_acc": round(hard_test_acc, 4) if hard_test_acc is not None else None,
        "hard_exact_acc": round(hard_exact_acc, 4) if hard_exact_acc is not None else None,
        "roundtrip_diff": rt_diff,
        "symbolic": symbolic,
    }


# ---------------------------------------------------------------------
# Benchmark loop + report writers
# ---------------------------------------------------------------------

CSV_COLUMNS = [
    "task", "model", "params", "epochs_used", "wall_s",
    "soft_test_acc", "exact_acc", "hard_test_acc", "hard_exact_acc",
    "roundtrip_diff",
]


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_COLUMNS})


def write_report_md(rows: List[Dict[str, Any]], path: str) -> None:
    """Render a human-readable markdown narrative."""
    tasks = sorted({r["task"] for r in rows})
    lines: List[str] = []
    lines.append("# LearnableNeuralCircuit Benchmark Report")
    lines.append("")
    lines.append(f"*Generated by plan_2026-05-13_25774a34. Rows: {len(rows)}*")
    lines.append("")

    # Headline summary.
    lines.append("## Headline")
    lines.append("")
    for t in tasks:
        sub = [r for r in rows if r["task"] == t]
        best = max(sub, key=lambda r: (r["soft_test_acc"] or 0.0))
        lines.append(
            f"- **{t}**: best soft test acc = **{best['soft_test_acc']:.4f}** "
            f"({best['model']}, {best['params']} params, {best['epochs_used']} epochs)"
        )
        circuit_row = next((r for r in sub if r["model"] == "circuit"), None)
        if circuit_row and circuit_row["hard_test_acc"] is not None:
            soft = circuit_row["soft_test_acc"]
            hard = circuit_row["hard_test_acc"]
            delta = hard - soft
            lines.append(
                f"  - Circuit faithfulness: soft={soft:.4f} hard={hard:.4f} "
                f"Δ={delta:+.4f} ({'FAITHFUL' if abs(delta) < 0.05 else 'LOSSY'})"
            )
    lines.append("")

    # Per-task tables.
    for t in tasks:
        sub = [r for r in rows if r["task"] == t]
        desc = TASK_SPECS[t]["description"]
        lines.append(f"## {t} — {desc}")
        lines.append("")
        lines.append("| Model | Params | Soft Test Acc | Exact Acc | Hard Test Acc | Hard Exact | Epochs | Wall (s) | Roundtrip Δ |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in sub:
            ea = f"{r['exact_acc']:.4f}" if r["exact_acc"] is not None else "—"
            ht = f"{r['hard_test_acc']:.4f}" if r["hard_test_acc"] is not None else "—"
            he = f"{r['hard_exact_acc']:.4f}" if r["hard_exact_acc"] is not None else "—"
            lines.append(
                f"| {r['model']} | {r['params']} | {r['soft_test_acc']:.4f} | {ea} | "
                f"{ht} | {he} | {r['epochs_used']} | {r['wall_s']} | {r['roundtrip_diff']:.2e} |"
            )
        # Symbolic readouts for circuit rows.
        for r in sub:
            if r["model"] == "circuit" and r.get("symbolic"):
                lines.append("")
                lines.append(f"**Symbolic readout (top-1 per inner op):**")
                lines.append("```")
                lines.append(r["symbolic"])
                lines.append("```")
        lines.append("")

    # Faithfulness summary.
    lines.append("## Faithfulness summary")
    lines.append("")
    lines.append("Hard extraction replaces each inner op's `operation_weights` with a one-hot at the argmax (LARGE × one_hot) so softmax is numerically one-hot. Re-evaluates without retraining. |Δ| < 0.05 ⇒ readout faithful.")
    lines.append("")
    lines.append("| Task | Soft | Hard | Δ | Verdict |")
    lines.append("|---|---|---|---|---|")
    for t in tasks:
        cr = next((r for r in rows if r["task"] == t and r["model"] == "circuit"), None)
        if cr is None or cr["hard_test_acc"] is None:
            continue
        d = cr["hard_test_acc"] - cr["soft_test_acc"]
        v = "FAITHFUL" if abs(d) < 0.05 else ("LOSSY" if d < -0.05 else "BOOSTED")
        lines.append(
            f"| {t} | {cr['soft_test_acc']:.4f} | {cr['hard_test_acc']:.4f} | "
            f"{d:+.4f} | {v} |"
        )
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------
# Grid + main
# ---------------------------------------------------------------------

# (task, model) pairs we actually run.
def grid() -> List[Tuple[str, str]]:
    g: List[Tuple[str, str]] = []
    for t in ["parity_k6", "majority_k6", "multiplexer_6"]:
        for m in ["circuit", "mlp_matched", "mlp_large"]:
            g.append((t, m))
    # Multi-output: only circuit_per_channel + mlp_large (matched would be tiny).
    g.append(("shift_xor_k8", "circuit"))
    g.append(("shift_xor_k8", "mlp_large"))
    return g


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark + faithfulness study.")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--train-samples", type=int, default=4096)
    p.add_argument("--test-samples", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir-prefix", type=str, default="logic_benchmark")
    p.add_argument("--only-task", type=str, default=None, help="Run only this task")
    p.add_argument("--only-model", type=str, default=None, help="Run only this model")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_gpu(args.gpu)
    ts = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"{args.results_dir_prefix}_{ts}")
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results dir: {results_dir}")

    rows: List[Dict[str, Any]] = []
    plan = grid()
    if args.only_task:
        plan = [p for p in plan if p[0] == args.only_task]
    if args.only_model:
        plan = [p for p in plan if p[1] == args.only_model]
    logger.info(f"Running {len(plan)} (task, model) combinations")

    for i, (task, model) in enumerate(plan, 1):
        logger.info(f"[{i}/{len(plan)}] {task} / {model}")
        try:
            row = train_one(
                task_name=task,
                model_name=model,
                epochs=args.epochs,
                train_samples=args.train_samples,
                test_samples=args.test_samples,
                seed=args.seed,
                results_dir=results_dir,
            )
        except Exception as e:
            logger.error(f"FAILED {task}/{model}: {e}")
            row = {
                "task": task, "model": model, "params": -1, "epochs_used": -1,
                "wall_s": 0.0, "soft_test_acc": 0.0, "exact_acc": None,
                "hard_test_acc": None, "hard_exact_acc": None,
                "roundtrip_diff": float("nan"), "symbolic": f"FAILED: {e}",
            }
        rows.append(row)
        logger.info(
            f"  -> soft_acc={row['soft_test_acc']:.4f} "
            f"hard_acc={row['hard_test_acc']} "
            f"params={row['params']} epochs={row['epochs_used']} wall={row['wall_s']}s"
        )

    csv_path = os.path.join(results_dir, "results.csv")
    md_path = os.path.join(results_dir, "report.md")
    write_csv(rows, csv_path)
    write_report_md(rows, md_path)
    logger.info(f"Wrote CSV: {csv_path}")
    logger.info(f"Wrote report: {md_path}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
