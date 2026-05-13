"""
E3: Non-saturation faithfulness study with circuit-native vs LIME vs SHAP
attributions on 3 new boolean tasks.

Plan: plan_2026-05-13_798d3a60 (decision D-001).

For each (task, model) pair we:
  1. Train to a band-checkpoint via ``StopOnAccuracyBand``.
  2. Compute hard-extraction Δ (the headline metric).
  3. Compute per-method attribution metrics on a sample of band-state
     validation inputs:
       - sufficiency / comprehensiveness AUCs (top-k mask sweep)
       - sparsity (normalized entropy of |attribution|)
       - stability (mean cosine similarity under 1-bit flips)
     for circuit / LIME / SHAP.
  4. Write a CSV row + a per-task report summary.

Tasks (3, new — not the train_benchmark.py grid):
  - ``mux_11bit`` (3 addr + 8 data) — non-trivial conditional logic
  - ``parity_k8``                   — symmetric, likely saturation cliff
  - ``random_dnf_8input_4term``     — structured noise, mixed difficulty

Usage::

    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train.logic.train_e3_faithfulness \\
        --tasks all --max-epochs 200 --band-low 0.70 --band-high 0.95 \\
        --out-dir results/logic_e3_<ts>
"""

# DECISION plan_2026-05-13_798d3a60/D-001
# Sibling trainer to train_e1_image.py. Both wrap (but do not modify)
# train_benchmark.py. The slight duplication of band-checkpoint and
# hard-extraction helpers is intentional under LESSONS L11 (N=2 sibling
# files is below the coupling crossover for a shared util).

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import keras
import numpy as np

from dl_techniques.utils.logger import logger
from train.common import setup_gpu
from train.logic.attributions import (
    circuit_attributions,
    lime_attributions,
    shap_attributions,
    sparsity,
    stability,
    suff_comp_aucs,
)
from train.logic.callbacks_band import StopOnAccuracyBand
from train.logic.train_benchmark import (
    build_circuit,
    build_mlp,
    extract_hard_inplace,
    restore_soft_weights,
    roundtrip_check,
)


# ---------------------------------------------------------------------
# Task generators (3 new tasks — disjoint from train_benchmark.py's set)
# ---------------------------------------------------------------------


def gen_mux_11bit(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """11-bit MUX: 3 address bits + 8 data bits; y = d[addr]."""
    x = rng.integers(0, 2, size=(n, 11)).astype(np.float32)
    addr = (
        x[:, 0].astype(np.int32)
        + 2 * x[:, 1].astype(np.int32)
        + 4 * x[:, 2].astype(np.int32)
    )
    data = x[:, 3:]  # (n, 8)
    y = data[np.arange(n), addr].reshape(-1, 1)
    return x, y


def gen_parity_k8(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    x = rng.integers(0, 2, size=(n, 8)).astype(np.float32)
    y = (x.sum(axis=1) % 2 == 1).astype(np.float32).reshape(-1, 1)
    return x, y


def _make_random_dnf_truth(num_terms: int, num_bits: int, seed: int) -> Callable[[np.ndarray], np.ndarray]:
    """Construct a random k-of-N DNF formula (positive/negative literals)
    and return a function evaluating it on a batch of bit vectors.
    """
    rng = np.random.default_rng(seed)
    terms = []  # list of (positive_mask, negative_mask) — each bit appears in zero or one polarity per term
    for _ in range(num_terms):
        # Each term: 3 random literals (sparse to ensure no trivial all-1 DNF).
        idxs = rng.choice(num_bits, size=3, replace=False)
        polarities = rng.integers(0, 2, size=3).astype(bool)
        pos = np.zeros(num_bits, dtype=bool)
        neg = np.zeros(num_bits, dtype=bool)
        for i, p in zip(idxs, polarities):
            if p:
                pos[i] = True
            else:
                neg[i] = True
        terms.append((pos, neg))

    def eval_dnf(x: np.ndarray) -> np.ndarray:
        x_b = x.astype(bool)
        out = np.zeros(x.shape[0], dtype=bool)
        for pos, neg in terms:
            term_val = np.ones(x.shape[0], dtype=bool)
            for i in np.where(pos)[0]:
                term_val &= x_b[:, i]
            for i in np.where(neg)[0]:
                term_val &= ~x_b[:, i]
            out |= term_val
        return out.astype(np.float32)

    return eval_dnf


_DNF_FN = _make_random_dnf_truth(num_terms=4, num_bits=8, seed=1234)


def gen_random_dnf(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Random 8-bit / 4-term DNF over a seeded formula (deterministic truth)."""
    x = rng.integers(0, 2, size=(n, 8)).astype(np.float32)
    y = _DNF_FN(x).reshape(-1, 1)
    return x, y


TASK_SPECS_E3: Dict[str, Dict[str, Any]] = {
    "mux_11bit": {
        "generator": lambda n, rng: gen_mux_11bit(n, rng),
        "num_bits": 11, "num_outputs": 1, "enum_size": 1 << 11,
        "description": "3-bit addr + 8-bit data; y = d[addr] — conditional",
    },
    "parity_k8": {
        "generator": lambda n, rng: gen_parity_k8(n, rng),
        "num_bits": 8, "num_outputs": 1, "enum_size": 1 << 8,
        "description": "y = XOR(x_1..x_8) — symmetric, saturation-prone",
    },
    "random_dnf_8input_4term": {
        "generator": lambda n, rng: gen_random_dnf(n, rng),
        "num_bits": 8, "num_outputs": 1, "enum_size": 1 << 8,
        "description": "Seeded random 4-term DNF over 8 bits — mixed difficulty",
    },
}


# ---------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------


def train_one_with_band(
    model: keras.Model,
    X: np.ndarray, Y: np.ndarray,
    Xv: np.ndarray, Yv: np.ndarray,
    band_low: float, band_high: float,
    max_epochs: int, batch_size: int = 64,
) -> Dict[str, Any]:
    band_cb = StopOnAccuracyBand("val_accuracy", band_low, band_high, "enter", verbose=1)
    t0 = time.time()
    h = model.fit(
        X, Y, validation_data=(Xv, Yv),
        epochs=max_epochs, batch_size=batch_size, verbose=0,
        callbacks=[band_cb],
    )
    return {
        "wall_s": time.time() - t0,
        "epochs_used": len(h.history.get("loss", [])),
        "band_entered": band_cb.fired,
        "band_acc": band_cb.band_value,
        "band_epoch": band_cb.band_epoch,
        "final_val_acc": float(h.history.get("val_accuracy", [float("nan")])[-1]),
    }


def evaluate_hard_extraction_circuit(
    model: keras.Model, Xv: np.ndarray, Yv: np.ndarray, save_path: str,
) -> Dict[str, Any]:
    soft_acc = float(model.evaluate(Xv, Yv, verbose=0)[1])
    rt_diff = roundtrip_check(model, Xv[:64], save_path)
    snapshot = extract_hard_inplace(model)
    try:
        hard_acc = float(model.evaluate(Xv, Yv, verbose=0)[1])
    finally:
        restore_soft_weights(snapshot)
    return {
        "soft_acc": round(soft_acc, 6),
        "hard_acc": round(hard_acc, 6),
        "delta_hard": round(hard_acc - soft_acc, 6),
        "roundtrip_diff": rt_diff,
    }


# ---------------------------------------------------------------------
# Attribution sweep
# ---------------------------------------------------------------------


def _attribution_metrics(
    name: str,
    model: keras.Model,
    X_sample: np.ndarray,
    attr_fn: Callable[[keras.Model, np.ndarray], np.ndarray],
    num_bits: int,
) -> Dict[str, Any]:
    """Run suff/comp/sparsity/stability for a single attribution method."""
    t0 = time.time()
    aucs = suff_comp_aucs(model, X_sample, attr_fn, k_range=list(range(0, num_bits + 1)))
    sparsities: List[float] = []
    stabilities: List[float] = []
    for x in X_sample:
        a = attr_fn(model, x)
        sparsities.append(sparsity(a))
        stabilities.append(stability(model, x, attr_fn, num_perturbations=min(num_bits, 8)))
    wall = time.time() - t0
    return {
        f"{name}_suff_auc": round(aucs["suff_auc"], 6),
        f"{name}_comp_auc": round(aucs["comp_auc"], 6),
        f"{name}_sparsity": round(float(np.mean(sparsities)), 6),
        f"{name}_stability": round(float(np.mean(stabilities)), 6),
        f"{name}_wall_s": round(wall, 2),
    }


# ---------------------------------------------------------------------
# CSV + report
# ---------------------------------------------------------------------

ATTRIBUTION_METHODS = ["circuit", "lime", "shap"]

BASE_COLUMNS = [
    "task", "model", "params",
    "band_low", "band_high", "band_acc", "band_epoch", "band_entered",
    "epochs_used", "wall_s",
    "soft_acc", "hard_acc", "delta_hard", "roundtrip_diff",
]

METRIC_SUFFIXES = ["_suff_auc", "_comp_auc", "_sparsity", "_stability", "_wall_s"]


def _csv_columns() -> List[str]:
    cols = list(BASE_COLUMNS)
    for m in ATTRIBUTION_METHODS:
        for s in METRIC_SUFFIXES:
            cols.append(f"{m}{s}")
    return cols


CSV_COLUMNS = _csv_columns()


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_COLUMNS})


def write_report_md(rows: List[Dict[str, Any]], path: str) -> None:
    lines: List[str] = []
    lines.append("# E3 Faithfulness Report")
    lines.append("")
    lines.append(f"*Plan: plan_2026-05-13_798d3a60. Rows: {len(rows)}.*")
    lines.append("")
    lines.append("## Headline (hard-extraction Δ + per-method faithfulness)")
    lines.append("")
    lines.append("| Task | Model | Params | Band | Soft | Hard | Δ | RT | Circuit suff/comp | LIME suff/comp | SHAP suff/comp |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        band = f"[{r['band_low']:.2f},{r['band_high']:.2f}]"
        ba = f"{r['band_acc']:.3f}" if r.get("band_acc") is not None else "—"
        sa = f"{r['soft_acc']:.3f}" if r.get("soft_acc") is not None else "—"
        ha = f"{r['hard_acc']:.3f}" if r.get("hard_acc") is not None else "—"
        dh = f"{r['delta_hard']:+.3f}" if r.get("delta_hard") is not None else "—"
        rt = f"{r['roundtrip_diff']:.1e}" if r.get("roundtrip_diff") is not None else "—"
        def cell(m):
            s = r.get(f"{m}_suff_auc")
            c = r.get(f"{m}_comp_auc")
            return f"{s:.3f}/{c:.3f}" if s is not None and c is not None else "—"
        lines.append(
            f"| {r['task']} | {r['model']} | {r['params']} | {band} ({ba}) | {sa} | {ha} | {dh} | {rt} "
            f"| {cell('circuit')} | {cell('lime')} | {cell('shap')} |"
        )
    lines.append("")
    lines.append("## Per-task notes")
    lines.append("")
    for r in rows:
        if not r.get("band_entered"):
            lines.append(f"- **{r['task']} / {r['model']}**: HONEST-NEGATIVE — band not entered (epochs_used={r['epochs_used']}, final_val_acc={r.get('final_val_acc')}).")
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E3 faithfulness study")
    p.add_argument("--tasks", type=str, default="all",
                   help="Comma-separated subset of TASK_SPECS_E3 keys, or 'all'.")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--band-low", type=float, default=0.70)
    p.add_argument("--band-high", type=float, default=0.95)
    p.add_argument("--train-samples", type=int, default=4096)
    p.add_argument("--test-samples", type=int, default=1024)
    p.add_argument("--num-attr-samples", type=int, default=32,
                   help="Number of validation samples on which to compute attribution metrics.")
    p.add_argument("--lime-num-samples", type=int, default=2000)
    p.add_argument("--shap-nsamples", type=int, default=128)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _resolve_tasks(tasks_arg: str) -> List[str]:
    if tasks_arg == "all":
        return list(TASK_SPECS_E3.keys())
    return [t.strip() for t in tasks_arg.split(",") if t.strip()]


def main() -> None:
    args = parse_args()
    setup_gpu(args.gpu)
    keras.utils.set_random_seed(args.seed)
    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.out_dir is None:
        out_dir = os.path.join("results", f"logic_e3_{ts}")
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"E3 run | tasks={args.tasks} | out_dir={out_dir}")

    rows: List[Dict[str, Any]] = []
    tasks = _resolve_tasks(args.tasks)

    for task_name in tasks:
        if task_name not in TASK_SPECS_E3:
            logger.warning(f"Unknown task {task_name!r}, skipping.")
            continue
        spec = TASK_SPECS_E3[task_name]
        num_bits = spec["num_bits"]
        num_outputs = spec["num_outputs"]

        rng_train = np.random.default_rng(args.seed)
        rng_test = np.random.default_rng(args.seed + 1000)
        X, Y = spec["generator"](args.train_samples, rng_train)
        Xv, Yv = spec["generator"](args.test_samples, rng_test)
        logger.info(f"[{task_name}] data: train {X.shape}, test {Xv.shape}")

        # ---- Circuit model only (E3 attribution comparison is on the
        # circuit; we keep an MLP comparison off the headline per plan
        # to avoid scope creep, but log it for context).
        circuit = build_circuit(num_bits, num_outputs)
        logger.info(f"[{task_name}] circuit params={circuit.count_params()}")
        train_res = train_one_with_band(
            circuit, X, Y, Xv, Yv,
            band_low=args.band_low, band_high=args.band_high,
            max_epochs=args.max_epochs, batch_size=args.batch_size,
        )
        logger.info(f"[{task_name}] training: {train_res}")
        save_path = os.path.join(out_dir, f"{task_name}_circuit.keras")
        hard_res = evaluate_hard_extraction_circuit(circuit, Xv, Yv, save_path)
        logger.info(f"[{task_name}] hard-extraction: {hard_res}")

        row: Dict[str, Any] = {
            "task": task_name, "model": "circuit", "params": circuit.count_params(),
            "band_low": args.band_low, "band_high": args.band_high,
            **train_res, **hard_res,
        }

        # Attribution sweep — only if model is in a usable state.
        if train_res["band_entered"] or hard_res["soft_acc"] > 0.55:
            # Sample some validation inputs.
            idxs = np.random.RandomState(args.seed).choice(
                Xv.shape[0], size=min(args.num_attr_samples, Xv.shape[0]), replace=False,
            )
            X_sample = Xv[idxs].astype(np.float32)

            # Circuit-native: integrated gradients
            row.update(_attribution_metrics(
                "circuit", circuit, X_sample,
                lambda m, x: circuit_attributions(m, x),
                num_bits=num_bits,
            ))

            # LIME
            try:
                row.update(_attribution_metrics(
                    "lime", circuit, X_sample,
                    lambda m, x: lime_attributions(m, x, num_samples=args.lime_num_samples),
                    num_bits=num_bits,
                ))
            except Exception as e:
                logger.warning(f"[{task_name}] LIME failed: {e}")

            # SHAP
            try:
                row.update(_attribution_metrics(
                    "shap", circuit, X_sample,
                    lambda m, x: shap_attributions(m, x, nsamples=args.shap_nsamples),
                    num_bits=num_bits,
                ))
            except Exception as e:
                logger.warning(f"[{task_name}] SHAP failed: {e}")
        else:
            logger.info(f"[{task_name}] skipping attribution sweep (band not entered and soft_acc <= 0.55).")

        rows.append(row)

    csv_path = os.path.join(out_dir, "benchmark_results.csv")
    md_path = os.path.join(out_dir, "report.md")
    write_csv(rows, csv_path)
    write_report_md(rows, md_path)
    logger.info(f"Wrote CSV: {csv_path}")
    logger.info(f"Wrote report: {md_path}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
