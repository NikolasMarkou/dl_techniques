"""E5: CLEVR-Hans3 visual reasoning with ``LearnableNeuralCircuit``.

Plan: plan_2026-05-14_c95e848c (decision D-001).

Three configurations:

1. ``circuit``: ResNet50(imagenet, frozen) -> GAP -> Dense(64) -> rank-4 reshape
    -> LearnableNeuralCircuit -> GAP -> Dense(num_classes).
2. ``mlp``: ResNet50(imagenet, frozen) -> GAP -> Dense(64) -> Dense(num_classes).
3. ``oracle`` (perfect perception): scene-graph (max_objects, 18) -> Flatten
    -> Dense(64) -> rank-4 reshape -> LearnableNeuralCircuit -> GAP
    -> Dense(num_classes).

Headline metric: shortcut-gap = val_acc - test_acc (smaller = more
shortcut-resistant). Reasoning: CLEVR-Hans3 README confirms train+val keep the
confounders, test breaks them. A pure shortcut-learner has val_acc ≫ test_acc.

Wall-clock leashes (per E5 orchestrator):
- download: 2h (in ``clevr_hans_data.download_clevr_hans3``)
- per-model training: 6h (via ``WallClockLimit`` callback)
- total: 16h (run-level, orchestrator-enforced)

Usage::

    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m train.logic.train_e5_clevr_hans \\
        --data-dir data/clevr_hans3 --out-dir results/logic_e5_<ts> \\
        --max-epochs 30 --batch-size 32
"""

# DECISION plan_2026-05-14_c95e848c/D-001
# Three-way comparison (circuit / param-matched mlp / oracle).
# ResNet50 substitutes for "ResNet-18" (the only live pretrained Keras CNN).
# NS-CL replaced with perfect-perception oracle (scene-graph JSON -> circuit)
# to stay within the 16h budget. See plans/plan_2026-05-14_c95e848c/decisions.md.

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
from train.common import setup_gpu
from train.logic.clevr_hans_data import (
    FEATURE_WIDTH,
    build_image_dataset,
    build_symbolic_dataset,
    download_clevr_hans3,
    find_splits,
    infer_num_classes,
)


# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------


class WallClockLimit(keras.callbacks.Callback):
    """Stop training when wall-clock seconds since ``__init__`` exceed ``max_s``."""

    def __init__(self, max_s: float, *, name: str = "model") -> None:
        super().__init__()
        self.max_s = float(max_s)
        self.name = name
        self.t0 = time.time()
        self.fired = False

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        elapsed = time.time() - self.t0
        if elapsed >= self.max_s:
            logger.warning(
                f"WallClockLimit[{self.name}] fired at epoch {epoch} "
                f"(elapsed {elapsed:.1f}s >= leash {self.max_s:.1f}s)."
            )
            self.fired = True
            self.model.stop_training = True


# ---------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------


def _circuit_block(x, *, circuit_channels: int = 64, name_prefix: str = "circuit"):
    """rank-4 LearnableNeuralCircuit block (matches LESSONS L51 hparams)."""
    return LearnableNeuralCircuit(
        circuit_depth=2,
        num_logic_ops_per_depth=2,
        num_arithmetic_ops_per_depth=2,
        use_residual=True,
        use_layer_norm=True,
        selection_mode="global",
        arithmetic_op_types=["add", "max", "min"],
        apply_sigmoid_per_depth="first_only",
        name=f"{name_prefix}_circuit",
    )(x)


def build_resnet50_circuit(
    num_classes: int,
    image_size: int = 128,
    circuit_channels: int = 64,
    lr: float = 1e-3,
) -> keras.Model:
    """ResNet50(frozen) -> GAP -> Dense(64) -> circuit -> GAP -> Dense head."""
    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    backbone = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=inputs,
    )
    backbone.trainable = False
    feat = keras.layers.GlobalAveragePooling2D(name="backbone_gap")(backbone.output)
    feat = keras.layers.Dense(circuit_channels, activation="relu", name="embed")(feat)
    # Reshape to rank-4 (B,1,1,C) so the circuit's rank-4 path applies.
    x = keras.layers.Reshape((1, 1, circuit_channels), name="to_rank4")(feat)
    x = _circuit_block(x, circuit_channels=circuit_channels, name_prefix="img")
    x = keras.layers.GlobalAveragePooling2D(name="circuit_gap")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="head")(x)
    model = keras.Model(inputs, outputs, name="resnet50_circuit")
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_resnet50_mlp(
    num_classes: int,
    image_size: int = 128,
    hidden: int = 64,
    lr: float = 1e-3,
) -> keras.Model:
    """ResNet50(frozen) -> GAP -> Dense(64) -> Dense head. Param-matched baseline."""
    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    backbone = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=inputs,
    )
    backbone.trainable = False
    feat = keras.layers.GlobalAveragePooling2D(name="backbone_gap")(backbone.output)
    feat = keras.layers.Dense(hidden, activation="relu", name="embed")(feat)
    feat = keras.layers.Dense(hidden, activation="relu", name="hidden")(feat)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="head")(feat)
    model = keras.Model(inputs, outputs, name="resnet50_mlp")
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_symbolic_circuit(
    num_classes: int,
    max_objects: int = 10,
    feature_width: int = FEATURE_WIDTH,
    circuit_channels: int = 64,
    lr: float = 1e-3,
) -> keras.Model:
    """Perfect-perception oracle: (max_objects, FW) -> Dense(64) -> circuit -> head."""
    inputs = keras.Input(shape=(max_objects, feature_width), name="scene_graph")
    x = keras.layers.Flatten(name="flatten")(inputs)
    x = keras.layers.Dense(circuit_channels, activation="relu", name="embed")(x)
    x = keras.layers.Reshape((1, 1, circuit_channels), name="to_rank4")(x)
    x = _circuit_block(x, circuit_channels=circuit_channels, name_prefix="sym")
    x = keras.layers.GlobalAveragePooling2D(name="circuit_gap")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="head")(x)
    model = keras.Model(inputs, outputs, name="symbolic_circuit_oracle")
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------
# Train + evaluate one model
# ---------------------------------------------------------------------


def train_and_eval(
    model: keras.Model,
    train_ds,
    val_ds,
    test_ds,
    *,
    max_epochs: int,
    max_wall_s: float,
    name: str,
    patience: int = 5,
) -> Dict[str, Any]:
    """Fit a model, then evaluate on val (confounded) and test (clean)."""
    es_cb = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max", patience=patience,
        restore_best_weights=True, verbose=1,
    )
    wc_cb = WallClockLimit(max_wall_s, name=name)
    t0 = time.time()
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=max_epochs,
            callbacks=[wc_cb, es_cb],
            verbose=2,
        )
        wall_s = time.time() - t0
        epochs_used = len(history.history.get("loss", []))
    except Exception as e:
        wall_s = time.time() - t0
        logger.error(f"train_and_eval[{name}]: fit raised {type(e).__name__}: {e}")
        return {
            "model": name,
            "status": "fit_error",
            "error": f"{type(e).__name__}: {e}",
            "wall_s": round(wall_s, 1),
        }

    val_acc = None
    test_acc = None
    try:
        val_acc = float(model.evaluate(val_ds, verbose=0)[1])
    except Exception as e:
        logger.error(f"train_and_eval[{name}]: val eval failed: {e}")
    try:
        test_acc = float(model.evaluate(test_ds, verbose=0)[1])
    except Exception as e:
        logger.error(f"train_and_eval[{name}]: test eval failed: {e}")

    gap = (val_acc - test_acc) if (val_acc is not None and test_acc is not None) else None
    return {
        "model": name,
        "status": "leashed" if wc_cb.fired else "complete",
        "params": int(model.count_params()),
        "epochs_used": epochs_used,
        "wall_s": round(wall_s, 1),
        "val_acc": round(val_acc, 6) if val_acc is not None else None,
        "test_acc": round(test_acc, 6) if test_acc is not None else None,
        "shortcut_gap": round(gap, 6) if gap is not None else None,
    }


# ---------------------------------------------------------------------
# CSV + report writers
# ---------------------------------------------------------------------

CSV_COLUMNS = [
    "model", "status", "params", "epochs_used", "wall_s",
    "val_acc", "test_acc", "shortcut_gap", "error",
]


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_COLUMNS})


def write_report_md(
    rows: List[Dict[str, Any]],
    path: str,
    *,
    honest_negative_reason: Optional[str] = None,
) -> None:
    lines: List[str] = []
    lines.append("# E5 Report — CLEVR-Hans3 Visual Reasoning")
    lines.append("")
    lines.append("*Plan: plan_2026-05-14_c95e848c. Decision D-001 (3-way: circuit / mlp / oracle).*")
    lines.append("")
    if honest_negative_reason is not None:
        lines.append("## HONEST-NEGATIVE: dataset acquisition failed")
        lines.append("")
        lines.append(f"Reason: {honest_negative_reason}")
        lines.append("")
        lines.append("Code + unit tests still ship in `src/train/logic/clevr_hans_data.py`, "
                     "`src/train/logic/train_e5_clevr_hans.py`, and "
                     "`tests/test_train/test_logic/test_clevr_hans_data.py`.")
        lines.append("")
    lines.append("## Headline (shortcut-gap = val_acc - test_acc)")
    lines.append("")
    lines.append("| Model | Status | Params | Epochs | Wall (s) | Val Acc (confounded) | Test Acc (clean) | Shortcut Gap |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        va = f"{r['val_acc']:.4f}" if r.get("val_acc") is not None else "—"
        ta = f"{r['test_acc']:.4f}" if r.get("test_acc") is not None else "—"
        sg = f"{r['shortcut_gap']:+.4f}" if r.get("shortcut_gap") is not None else "—"
        params = r.get("params") if r.get("params") is not None else "—"
        ep = r.get("epochs_used") if r.get("epochs_used") is not None else "—"
        ws = r.get("wall_s") if r.get("wall_s") is not None else "—"
        lines.append(
            f"| {r['model']} | {r.get('status','?')} | {params} | {ep} | {ws} | {va} | {ta} | {sg} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- val split keeps the train confounders (per CLEVR-Hans3 README); test split breaks them.")
    lines.append("- Smaller `shortcut_gap` = more shortcut-resistant.")
    lines.append("- `oracle` row uses scene-graph JSON directly (perfect perception); isolates reasoning-head bias.")
    lines.append("- `mlp` is the param-matched baseline with identical ResNet50(frozen) backbone.")
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E5: CLEVR-Hans3 visual reasoning with LearnableNeuralCircuit")
    p.add_argument("--data-dir", type=str, default="data/clevr_hans3",
                   help="Directory containing CLEVR-Hans3 (download target).")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output dir. Defaults to results/logic_e5_<timestamp>/.")
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--symbolic-batch-size", type=int, default=128)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--max-wall-s-per-model", type=float, default=21600.0,
                   help="Hard wall-clock leash per model (default 6h).")
    p.add_argument("--download-timeout-s", type=int, default=7200,
                   help="Hard download leash (default 2h).")
    p.add_argument("--skip-download", action="store_true",
                   help="Use existing data on disk; do not attempt download.")
    p.add_argument("--download-only", action="store_true",
                   help="Download (if needed) and exit; do not train.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--skip-models", type=str, default="",
                   help="Comma-separated subset to skip: circuit,mlp,oracle.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_gpu(args.gpu)
    keras.utils.set_random_seed(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join("results", f"logic_e5_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"E5 run | data_dir={args.data_dir} | out_dir={out_dir}")

    skip = set(s.strip().lower() for s in args.skip_models.split(",") if s.strip())

    # --- Stage A: acquire data ---------------------------------------
    have_data = False
    if not args.skip_download:
        have_data = download_clevr_hans3(args.data_dir, timeout_s=args.download_timeout_s)
    else:
        have_data = os.path.isdir(os.path.join(args.data_dir, "CLEVR-Hans3")) or any(
            os.path.isdir(os.path.join(args.data_dir, sub)) for sub in ("images", "scenes")
        )

    if args.download_only:
        logger.info(f"download-only: have_data={have_data}; exiting.")
        return

    splits = find_splits(args.data_dir) if have_data else {}
    rows: List[Dict[str, Any]] = []

    if not (have_data and {"train", "val", "test"} <= set(splits)):
        reason = (
            "download or extraction failed; required splits not found "
            f"(found splits: {sorted(splits.keys())})"
        )
        logger.error(f"HONEST-NEGATIVE branch: {reason}")
        report_md = os.path.join(out_dir, "report.md")
        write_report_md(rows, report_md, honest_negative_reason=reason)
        write_csv(rows, os.path.join(out_dir, "results.csv"))
        logger.info(f"Wrote honest-negative report: {report_md}")
        return

    # --- Stage B: build datasets + train -----------------------------
    num_classes = max(
        infer_num_classes(splits["train"]),
        infer_num_classes(splits["val"]),
        infer_num_classes(splits["test"]),
    )
    if num_classes <= 0:
        reason = "scenes JSON has no class_id field; oracle path cannot proceed"
        logger.error(f"HONEST-NEGATIVE: {reason}")
        report_md = os.path.join(out_dir, "report.md")
        write_report_md(rows, report_md, honest_negative_reason=reason)
        write_csv(rows, os.path.join(out_dir, "results.csv"))
        return
    logger.info(f"Inferred num_classes={num_classes}")

    # Image datasets (for circuit + mlp).
    if "circuit" not in skip or "mlp" not in skip:
        try:
            img_train_ds, n_train = build_image_dataset(
                splits["train"], image_size=args.image_size, batch_size=args.batch_size,
                shuffle=True, seed=args.seed,
            )
            img_val_ds, n_val = build_image_dataset(
                splits["val"], image_size=args.image_size, batch_size=args.batch_size,
                shuffle=False,
            )
            img_test_ds, n_test = build_image_dataset(
                splits["test"], image_size=args.image_size, batch_size=args.batch_size,
                shuffle=False,
            )
            logger.info(f"Image datasets: train={n_train}, val={n_val}, test={n_test}")
        except Exception as e:
            logger.error(f"Image dataset build failed: {type(e).__name__}: {e}")
            img_train_ds = img_val_ds = img_test_ds = None
    else:
        img_train_ds = img_val_ds = img_test_ds = None

    # Symbolic datasets (for oracle).
    if "oracle" not in skip:
        try:
            sym_train_ds, _ = build_symbolic_dataset(
                splits["train"], batch_size=args.symbolic_batch_size,
                shuffle=True, seed=args.seed,
            )
            sym_val_ds, _ = build_symbolic_dataset(
                splits["val"], batch_size=args.symbolic_batch_size, shuffle=False,
            )
            sym_test_ds, _ = build_symbolic_dataset(
                splits["test"], batch_size=args.symbolic_batch_size, shuffle=False,
            )
        except Exception as e:
            logger.error(f"Symbolic dataset build failed: {type(e).__name__}: {e}")
            sym_train_ds = sym_val_ds = sym_test_ds = None
    else:
        sym_train_ds = sym_val_ds = sym_test_ds = None

    # ----- Circuit (image) -----
    if "circuit" not in skip and img_train_ds is not None:
        logger.info("=== Training: resnet50_circuit ===")
        m = build_resnet50_circuit(num_classes, image_size=args.image_size, lr=args.lr)
        rows.append(train_and_eval(
            m, img_train_ds, img_val_ds, img_test_ds,
            max_epochs=args.max_epochs, max_wall_s=args.max_wall_s_per_model,
            name="resnet50_circuit", patience=args.patience,
        ))
        del m
        keras.backend.clear_session()
    else:
        logger.info("Skipping circuit (skip-models or dataset unavailable).")

    # ----- MLP (image) -----
    if "mlp" not in skip and img_train_ds is not None:
        logger.info("=== Training: resnet50_mlp ===")
        m = build_resnet50_mlp(num_classes, image_size=args.image_size, lr=args.lr)
        rows.append(train_and_eval(
            m, img_train_ds, img_val_ds, img_test_ds,
            max_epochs=args.max_epochs, max_wall_s=args.max_wall_s_per_model,
            name="resnet50_mlp", patience=args.patience,
        ))
        del m
        keras.backend.clear_session()
    else:
        logger.info("Skipping mlp.")

    # ----- Oracle (symbolic) -----
    if "oracle" not in skip and sym_train_ds is not None:
        logger.info("=== Training: symbolic_circuit_oracle ===")
        m = build_symbolic_circuit(num_classes, lr=args.lr)
        rows.append(train_and_eval(
            m, sym_train_ds, sym_val_ds, sym_test_ds,
            max_epochs=args.max_epochs, max_wall_s=args.max_wall_s_per_model,
            name="symbolic_circuit_oracle", patience=args.patience,
        ))
        del m
        keras.backend.clear_session()
    else:
        logger.info("Skipping oracle.")

    csv_path = os.path.join(out_dir, "results.csv")
    md_path = os.path.join(out_dir, "report.md")
    write_csv(rows, csv_path)
    write_report_md(rows, md_path)
    logger.info(f"Wrote CSV: {csv_path}")
    logger.info(f"Wrote report: {md_path}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
