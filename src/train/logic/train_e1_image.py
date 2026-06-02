"""
E1: ``LearnableNeuralCircuit`` replication on MNIST + CIFAR-10.

Plan: plan_2026-05-13_798d3a60 (decision D-001).

Architecture A (Petersen-DLGN-style adaptation):
  Conv stem (1 stage MNIST, 3 stages CIFAR-10)
    -> LearnableNeuralCircuit (rank-4 spatial, depth=2)
    -> GlobalAvgPool2D
    -> Dense(num_classes, softmax)

The circuit hyperparameters are pinned by LESSONS L51 (NaN-safe):
  - ``circuit_depth=2``
  - ``arithmetic_op_types=['add','max','min']`` (bounded set)
  - ``apply_sigmoid_per_depth='first_only'``

Headline metric: hard-extraction Δ (soft − hard val_accuracy) at
non-saturation. ``StopOnAccuracyBand`` halts training the first epoch the
monitor enters the ``[band_low, band_high]`` window. At that checkpoint we
compute the soft accuracy, hard-extract the circuit, re-evaluate, and
restore the soft weights.

A matched-param CNN baseline trains under the same protocol for
comparison.

Usage::

    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train.logic.train_e1_image \\
        --dataset mnist --max-epochs 50 --band-low 0.70 --band-high 0.95 \\
        --out-dir results/logic_e1_mnist_<ts>
"""

# DECISION plan_2026-05-13_798d3a60/D-001
# This module + train_e3_faithfulness.py are sibling training scripts under
# src/train/logic/ that wrap (but do not modify) the FROZEN
# train_benchmark.py. See plans/plan_2026-05-13_798d3a60/decisions.md
# entry D-001 for rationale (N=2 is below the LESSONS L11 coupling
# crossover so no shared util module is extracted).

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np

from dl_techniques.layers.logic import LearnableNeuralCircuit
from dl_techniques.utils.logger import logger
from train.common import setup_gpu, load_dataset, set_seeds
from train.logic.callbacks_band import StopOnAccuracyBand
from train.logic.train_benchmark import (
    extract_hard_inplace,
    restore_soft_weights,
    roundtrip_check,
)


# ---------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------


def _conv_stage(
    x, filters: int, name: str, *, downsample: bool = True,
):
    x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    x = keras.layers.ReLU(name=f"{name}_relu")(x)
    if downsample:
        x = keras.layers.MaxPool2D(pool_size=2, name=f"{name}_pool")(x)
    return x


def build_image_circuit(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    circuit_depth: int = 2,
    stem_filters: Tuple[int, ...] = (32,),
    circuit_channels: int = 32,
    lr: float = 1e-3,
) -> keras.Model:
    """Conv stem + LearnableNeuralCircuit + GlobalAvgPool + Dense head.

    Conv stem has ``len(stem_filters)`` downsample stages, each
    Conv -> BN -> ReLU -> MaxPool. The output feature map (rank-4) is
    fed through ``LearnableNeuralCircuit`` directly (rank-4 path opened
    by plan_2aaad563).
    """
    inputs = keras.Input(shape=input_shape, name="image")
    x = inputs
    for i, f in enumerate(stem_filters):
        x = _conv_stage(x, f, name=f"stage{i}")
    # Project to circuit_channels if needed.
    if int(x.shape[-1]) != circuit_channels:
        x = keras.layers.Conv2D(
            circuit_channels, 1, padding="same", use_bias=False, name="project",
        )(x)
    x = LearnableNeuralCircuit(
        circuit_depth=circuit_depth,
        num_logic_ops_per_depth=2,
        num_arithmetic_ops_per_depth=2,
        use_residual=True,
        use_layer_norm=True,
        selection_mode="global",
        arithmetic_op_types=["add", "max", "min"],
        apply_sigmoid_per_depth="first_only",
        name="neural_circuit",
    )(x)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="head")(x)
    model = keras.Model(inputs, outputs, name=f"image_circuit_{'_'.join(map(str, stem_filters))}")
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_baseline(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    filters: Tuple[int, ...],
    head_units: int = 64,
    lr: float = 1e-3,
) -> keras.Model:
    """Plain CNN: Conv-BN-ReLU-Pool blocks + GAP + Dense head + Dense classifier."""
    inputs = keras.Input(shape=input_shape, name="image")
    x = inputs
    for i, f in enumerate(filters):
        x = _conv_stage(x, f, name=f"stage{i}")
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dense(head_units, activation="relu", name="dense_head")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="head")(x)
    model = keras.Model(inputs, outputs, name=f"cnn_{'_'.join(map(str, filters))}")
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def find_cnn_filters_for_param_budget(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    filter_pattern: Tuple[float, ...],
    target_params: int,
    head_units: int = 64,
    tol: float = 0.05,
) -> Tuple[Tuple[int, ...], int]:
    """Scale `filter_pattern` by a multiplicative factor s.t. the CNN's
    param count is within `tol` of `target_params`. Returns
    (filters_tuple, multiplier).
    """
    pattern = np.asarray(filter_pattern, dtype=np.float64)
    lo, hi = 1, 256
    best = None
    best_diff = float("inf")
    while lo <= hi:
        mid = (lo + hi) // 2
        filters = tuple(int(max(1, round(mid * (p / pattern[0])))) for p in pattern)
        m = build_cnn_baseline(input_shape, num_classes, filters=filters, head_units=head_units)
        p = m.count_params()
        diff = abs(p - target_params)
        if diff < best_diff:
            best = (filters, mid)
            best_diff = diff
        if p < target_params:
            lo = mid + 1
        else:
            hi = mid - 1
        # Cleanup big-graphs in keras backend if accumulation becomes an issue.
    assert best is not None
    rel = best_diff / max(target_params, 1)
    if rel > tol:
        logger.warning(
            f"find_cnn_filters_for_param_budget: closest match diff="
            f"{best_diff}, rel={rel:.3f} (> tol {tol})"
        )
    return best


# ---------------------------------------------------------------------
# Training + evaluation helpers
# ---------------------------------------------------------------------


def train_one_with_band_checkpoint(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    band_low: float,
    band_high: float,
    max_epochs: int,
    batch_size: int = 64,
    monitor: str = "val_accuracy",
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Train until `monitor` first enters [band_low, band_high] or `max_epochs`."""
    band_cb = StopOnAccuracyBand(
        monitor=monitor, low=band_low, high=band_high, mode="enter", verbose=1
    )
    t0 = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=max_epochs, batch_size=batch_size, verbose=2,
        callbacks=[band_cb],
    )
    wall_s = time.time() - t0
    epochs_used = len(history.history.get("loss", []))
    band_entered = band_cb.fired
    band_acc = band_cb.band_value if band_entered else None
    band_epoch = band_cb.band_epoch if band_entered else None
    if save_path and band_entered:
        try:
            model.save(save_path)
            logger.info(f"Saved band checkpoint to {save_path}")
        except Exception as e:
            logger.warning(f"Could not save band checkpoint: {e}")
    return {
        "wall_s": wall_s,
        "epochs_used": epochs_used,
        "band_entered": band_entered,
        "band_acc": band_acc,
        "band_epoch": band_epoch,
        "final_val_acc": float(history.history.get(monitor, [float("nan")])[-1]),
    }


def evaluate_hard_extraction(
    model: keras.Model,
    x_val: np.ndarray,
    y_val: np.ndarray,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Hard-extraction faithfulness: soft acc, hard acc, delta, roundtrip diff.

    Works for any model containing one or more inner ops (circuit models).
    For non-circuit models (CNN baseline) the snapshot is empty and we
    return only soft_acc with delta_hard=NaN.
    """
    soft_acc = float(model.evaluate(x_val, y_val, verbose=0)[1])
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), "_tmp_rt.keras")
        cleanup = True
    else:
        cleanup = False
    rt_diff = roundtrip_check(model, x_val[:32], save_path)
    snapshot = extract_hard_inplace(model)
    if not snapshot:
        if cleanup and os.path.exists(save_path):
            os.remove(save_path)
        return {
            "soft_acc": round(soft_acc, 6),
            "hard_acc": None,
            "delta_hard": None,
            "roundtrip_diff": rt_diff,
        }
    try:
        hard_acc = float(model.evaluate(x_val, y_val, verbose=0)[1])
    finally:
        restore_soft_weights(snapshot)
    rt2 = roundtrip_check(model, x_val[:32], save_path)
    if rt2 > 1e-6:
        logger.warning(
            f"evaluate_hard_extraction: post-restore round-trip diff "
            f"{rt2:.2e} > 1e-6 (restoration leaked)."
        )
    if cleanup and os.path.exists(save_path):
        os.remove(save_path)
    return {
        "soft_acc": round(soft_acc, 6),
        "hard_acc": round(hard_acc, 6),
        "delta_hard": round(hard_acc - soft_acc, 6),
        "roundtrip_diff": rt_diff,
    }


# ---------------------------------------------------------------------
# CSV + report writers
# ---------------------------------------------------------------------

CSV_COLUMNS = [
    "dataset", "model", "params",
    "band_low", "band_high", "band_acc", "band_epoch", "band_entered",
    "epochs_used", "wall_s",
    "soft_acc", "hard_acc", "delta_hard", "roundtrip_diff",
    "final_val_acc",
]


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_COLUMNS})


def write_report_md(rows: List[Dict[str, Any]], path: str, dataset: str) -> None:
    lines: List[str] = []
    lines.append(f"# E1 Hard-Extraction Δ Report — {dataset.upper()}")
    lines.append("")
    lines.append(f"*Plan: plan_2026-05-13_798d3a60. Rows: {len(rows)}.*")
    lines.append("")
    lines.append("## Headline (hard-extraction Δ at non-saturation)")
    lines.append("")
    lines.append("| Model | Params | Band [low,high] | Band Acc | Soft Acc | Hard Acc | Δ (hard-soft) | Roundtrip Δ | Verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        band = f"[{r['band_low']:.2f}, {r['band_high']:.2f}]"
        ba = f"{r['band_acc']:.4f}" if r.get("band_acc") is not None else "—"
        sa = f"{r['soft_acc']:.4f}" if r.get("soft_acc") is not None else "—"
        ha = f"{r['hard_acc']:.4f}" if r.get("hard_acc") is not None else "—"
        dh = f"{r['delta_hard']:+.4f}" if r.get("delta_hard") is not None else "—"
        rt = f"{r['roundtrip_diff']:.2e}" if r.get("roundtrip_diff") is not None else "—"
        if r.get("delta_hard") is None:
            verdict = "no-circuit"
        elif not r.get("band_entered"):
            verdict = "HONEST-NEGATIVE (band not entered)"
        else:
            d = r["delta_hard"]
            if abs(d) < 0.05:
                verdict = "FAITHFUL"
            elif d < -0.05:
                verdict = "LOSSY"
            else:
                verdict = "BOOSTED"
        lines.append(
            f"| {r['model']} | {r['params']} | {band} | {ba} | {sa} | {ha} | {dh} | {rt} | {verdict} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Headline metric is hard-extraction Δ on the **circuit** model at the **band-entry checkpoint** (val_accuracy ∈ [band_low, band_high]).")
    lines.append("- CNN baseline has no inner ops to hard-extract; its row reports soft accuracy only for matched-param comparison.")
    lines.append("- If `band_entered=False`, the model either saturated past the band in one epoch or never reached `band_low` within `max_epochs` — recorded as honest negative per the plan's Pre-Mortem (LESSONS L51 precedent).")
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1: LearnableNeuralCircuit on MNIST / CIFAR-10")
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--band-low", type=float, default=0.70)
    p.add_argument("--band-high", type=float, default=0.95)
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory. Defaults to results/logic_e1_<dataset>_<timestamp>/.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--circuit-channels", type=int, default=32)
    return p.parse_args()


def _dataset_stem(dataset: str) -> Tuple[int, ...]:
    """Return the stem-filter tuple for the circuit on a dataset."""
    if dataset == "mnist":
        return (32,)
    if dataset == "cifar10":
        return (32, 64, 128)
    raise ValueError(dataset)


def main() -> None:
    args = parse_args()
    setup_gpu(args.gpu)
    set_seeds(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.out_dir is None:
        out_dir = os.path.join("results", f"logic_e1_{args.dataset}_{ts}")
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"E1 run | dataset={args.dataset} | out_dir={out_dir}")

    (x_train, y_train), (x_test, y_test), input_shape, num_classes = load_dataset(args.dataset)

    rows: List[Dict[str, Any]] = []

    # ---- Circuit model
    logger.info("Building circuit model.")
    stem = _dataset_stem(args.dataset)
    circuit = build_image_circuit(
        input_shape=input_shape, num_classes=num_classes,
        stem_filters=stem, circuit_channels=args.circuit_channels,
    )
    circuit_params = circuit.count_params()
    logger.info(f"Circuit params: {circuit_params}")
    train_res = train_one_with_band_checkpoint(
        circuit, x_train, y_train, x_test, y_test,
        band_low=args.band_low, band_high=args.band_high,
        max_epochs=args.max_epochs, batch_size=args.batch_size,
        save_path=os.path.join(out_dir, "circuit_band.keras"),
    )
    logger.info(f"Circuit training done: {train_res}")
    hard_res = evaluate_hard_extraction(
        circuit, x_test, y_test,
        save_path=os.path.join(out_dir, "circuit_rt.keras"),
    )
    logger.info(f"Circuit hard-extraction: {hard_res}")
    rows.append({
        "dataset": args.dataset, "model": "circuit", "params": circuit_params,
        "band_low": args.band_low, "band_high": args.band_high,
        **train_res, **hard_res,
    })

    # ---- Matched-param CNN baseline
    logger.info("Building matched-param CNN baseline.")
    filter_pattern = stem
    filters, mult = find_cnn_filters_for_param_budget(
        input_shape, num_classes,
        filter_pattern=filter_pattern, target_params=circuit_params,
    )
    cnn = build_cnn_baseline(input_shape, num_classes, filters=filters)
    cnn_params = cnn.count_params()
    logger.info(f"CNN baseline filters={filters}, params={cnn_params}")
    train_res_c = train_one_with_band_checkpoint(
        cnn, x_train, y_train, x_test, y_test,
        band_low=args.band_low, band_high=args.band_high,
        max_epochs=args.max_epochs, batch_size=args.batch_size,
        save_path=os.path.join(out_dir, "cnn_band.keras"),
    )
    hard_res_c = evaluate_hard_extraction(
        cnn, x_test, y_test,
        save_path=os.path.join(out_dir, "cnn_rt.keras"),
    )
    logger.info(f"CNN baseline: train={train_res_c}, hard={hard_res_c}")
    rows.append({
        "dataset": args.dataset, "model": "cnn_matched", "params": cnn_params,
        "band_low": args.band_low, "band_high": args.band_high,
        **train_res_c, **hard_res_c,
    })

    csv_path = os.path.join(out_dir, "benchmark_results.csv")
    md_path = os.path.join(out_dir, "report.md")
    write_csv(rows, csv_path)
    write_report_md(rows, md_path, args.dataset)
    logger.info(f"Wrote CSV: {csv_path}")
    logger.info(f"Wrote report: {md_path}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
