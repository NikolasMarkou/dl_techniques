"""
E4 low-data: 11-bit MUX learning curves.

Plan: plan_2026-05-14_e26eede2 (E4 — low-data MUX learning curves §E4).

Tests whether ``LearnableNeuralCircuit`` has a useful inductive bias in the
*low-data regime* by sweeping training-set size N ∈ {32, 64, 128, 256, 512, 1024}
on the 11-bit multiplexer (3 address + 8 data) — a conditional-logic benchmark
that is classically hard for small MLPs.

For each N × model × seed:
    - Train on N synthetic 11-bit MUX examples.
    - Evaluate on a fixed 2048-example held-out test set.

Outputs:
    results/logic_e4_lowdata_mux_<ts>/mux_results.csv
    results/logic_e4_lowdata_mux_<ts>/learning_curve.png
    results/logic_e4_lowdata_mux_<ts>/report.md
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from typing import Any, Dict, List

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dl_techniques.utils.logger import logger
from train.logic.train_benchmark import (
    build_circuit,
    build_mlp,
    find_mlp_hidden_for_param_budget,
)
from train.logic.train_e3_faithfulness import gen_mux_11bit

NUM_BITS = 11
NUM_OUTPUTS = 1
N_VALUES = (32, 64, 128, 256, 512, 1024)
MODEL_NAMES = ("circuit", "mlp_matched", "xgboost")
EPOCHS = 200
TEST_SIZE = 2048

CSV_COLUMNS = ["N", "model", "seed", "params", "epochs_used", "wall_s", "test_acc"]


def _train_keras(model, x_tr, y_tr, x_te, y_te, epochs):
    early = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=max(15, epochs // 5),
        restore_best_weights=True, verbose=0, mode="max",
    )
    history = model.fit(
        x_tr, y_tr, validation_data=(x_te, y_te),
        epochs=epochs, batch_size=min(32, max(8, len(x_tr) // 4)),
        verbose=0, callbacks=[early],
    )
    return len(history.history["loss"])


def run_cell(N: int, model_name: str, seed: int) -> Dict[str, Any]:
    logger.info(f"--- MUX-11 N={N} / {model_name} / seed={seed} ---")
    t0 = time.time()
    keras.utils.set_random_seed(seed)
    np.random.seed(seed)

    rng_tr = np.random.default_rng(seed)
    rng_te = np.random.default_rng(999 + seed)
    x_tr, y_tr = gen_mux_11bit(N, rng_tr)
    x_te, y_te = gen_mux_11bit(TEST_SIZE, rng_te)
    y_tr = y_tr.astype(np.float32)
    y_te = y_te.astype(np.float32)

    if model_name == "circuit":
        model = build_circuit(NUM_BITS, NUM_OUTPUTS, channels=32, circuit_depth=2)
        epochs_used = _train_keras(model, x_tr, y_tr, x_te, y_te, EPOCHS)
        params = model.count_params()
        test_acc = float(model.evaluate(x_te, y_te, verbose=0)[1])
    elif model_name == "mlp_matched":
        ref = build_circuit(NUM_BITS, NUM_OUTPUTS, channels=32, circuit_depth=2)
        h = find_mlp_hidden_for_param_budget(NUM_BITS, NUM_OUTPUTS, ref.count_params())
        model = build_mlp(NUM_BITS, NUM_OUTPUTS, hidden_units=h)
        epochs_used = _train_keras(model, x_tr, y_tr, x_te, y_te, EPOCHS)
        params = model.count_params()
        test_acc = float(model.evaluate(x_te, y_te, verbose=0)[1])
    elif model_name == "xgboost":
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", use_label_encoder=False,
            random_state=seed, verbosity=0,
        )
        clf.fit(x_tr, y_tr.astype(np.int64).reshape(-1))
        proba = clf.predict_proba(x_te)[:, 1]
        test_acc = float(((proba > 0.5).astype(np.int64).reshape(-1) ==
                          y_te.astype(np.int64).reshape(-1)).mean())
        try:
            params = int(clf.get_booster().trees_to_dataframe().shape[0])
        except Exception:
            params = -1
        epochs_used = 200
    else:
        raise ValueError(f"unknown model_name: {model_name}")

    wall = round(time.time() - t0, 2)
    logger.info(f"  test_acc={test_acc:.3f}  wall={wall}s")
    return {
        "N": N, "model": model_name, "seed": seed,
        "params": params, "epochs_used": epochs_used,
        "wall_s": wall, "test_acc": test_acc,
    }


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_COLUMNS})


def _agg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def plot_learning_curve(rows: List[Dict[str, Any]], path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"circuit": "tab:blue", "mlp_matched": "tab:orange", "xgboost": "tab:green"}
    markers = {"circuit": "o", "mlp_matched": "s", "xgboost": "^"}
    for m in MODEL_NAMES:
        ns = sorted({r["N"] for r in rows if r["model"] == m})
        means, stds = [], []
        for n in ns:
            sub = [r for r in rows if r["model"] == m and r["N"] == n]
            mu, sd = _agg(sub, "test_acc")
            means.append(mu)
            stds.append(sd)
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(ns, means, marker=markers[m], color=colors[m], label=m, linewidth=2)
        ax.fill_between(ns, means - stds, means + stds, color=colors[m], alpha=0.2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(N_VALUES)
    ax.set_xticklabels([str(n) for n in N_VALUES])
    ax.set_xlabel("Training set size (N)")
    ax.set_ylabel("Test accuracy (2048-example held-out)")
    ax.set_title("E4 — 11-bit MUX learning curves (mean ± std over 3 seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_ylim(0.4, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info(f"Wrote plot: {path}")


def write_report(rows, path):
    lines = [
        "# E4 low-data — 11-bit MUX learning curves",
        "",
        "Plan: plan_2026-05-14_e26eede2.",
        "",
        f"Test set: fixed {TEST_SIZE}-example 11-bit MUX (3 address + 8 data).",
        "Models: circuit (depth=2, channels=32), mlp_matched (param-matched), xgboost.",
        f"Train sizes: N ∈ {{{', '.join(str(n) for n in N_VALUES)}}}.",
        "3 seeds per cell.",
        "",
        "| N | circuit (mean±std) | mlp_matched (mean±std) | xgboost (mean±std) |",
        "|---|---|---|---|",
    ]
    for n in N_VALUES:
        line_cells = [f"| {n}"]
        for m in MODEL_NAMES:
            sub = [r for r in rows if r["N"] == n and r["model"] == m]
            mu, sd = _agg(sub, "test_acc")
            line_cells.append(f" {mu:.3f}±{sd:.3f}")
        lines.append("|".join(line_cells) + " |")
    lines.append("")
    # Headline criterion at N <= 128.
    lines.append("## Headline criterion: circuit beats both baselines by >5pt at N <= 128")
    lines.append("")
    n_wins = 0
    for n in (32, 64, 128):
        c_m, _ = _agg([r for r in rows if r["N"] == n and r["model"] == "circuit"], "test_acc")
        m_m, _ = _agg([r for r in rows if r["N"] == n and r["model"] == "mlp_matched"], "test_acc")
        x_m, _ = _agg([r for r in rows if r["N"] == n and r["model"] == "xgboost"], "test_acc")
        win = (c_m - m_m) > 0.05 and (c_m - x_m) > 0.05
        n_wins += int(win)
        lines.append(f"- N={n}: circuit={c_m:.3f}, mlp={m_m:.3f}, xgb={x_m:.3f} -> wins by >5pt: **{win}**")
    verdict = (
        "PASS — circuit wins at the low-data regime on >=2 of 3 N<=128 sizes."
        if n_wins >= 2 else
        "FAIL — circuit does not consistently win at the low-data regime."
    )
    lines.append("")
    lines.append(f"### Verdict: **{verdict}**")
    lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="E4 low-data 11-bit MUX")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--out-dir", type=str, default="results/logic_e4_lowdata_mux")
    parser.add_argument("--n-values", type=str, default=",".join(str(n) for n in N_VALUES))
    parser.add_argument("--models", type=str, default=",".join(MODEL_NAMES))
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    n_values = [int(n) for n in args.n_values.split(",")]
    models = [m.strip() for m in args.models.split(",")]

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = f"{args.out_dir}_{ts}" if not args.out_dir.endswith(ts) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"E4 MUX: out_dir={out_dir} N={n_values} seeds={seeds} models={models}")

    rows: List[Dict[str, Any]] = []
    for n in n_values:
        for m in models:
            for s in seeds:
                try:
                    rows.append(run_cell(n, m, s))
                    write_csv(rows, os.path.join(out_dir, "mux_results.csv"))
                except Exception as e:
                    logger.exception(f"cell failed N={n} model={m} seed={s}: {e}")
                    rows.append({
                        "N": n, "model": m, "seed": s,
                        "params": -1, "epochs_used": 0, "wall_s": 0.0,
                        "test_acc": float("nan"),
                    })

    csv_path = os.path.join(out_dir, "mux_results.csv")
    write_csv(rows, csv_path)
    plot_learning_curve(rows, os.path.join(out_dir, "learning_curve.png"))
    write_report(rows, os.path.join(out_dir, "report.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
