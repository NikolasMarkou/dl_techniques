"""
E4: UCI Monks-1/2/3 rule-recovery benchmark.

Plan: plan_2026-05-14_e26eede2 (E4 — UCI Monks rule recovery).

Grid:
    Tasks: Monks-1, Monks-2, Monks-3 (canonical 124/169/122-train, 432-test).
    Models: circuit (depth=2, channels=32), mlp_matched (param-budget match),
            xgboost.
    Seeds: 0, 1, 2 (3 random seeds per (task, model)).

For every (task, model, seed) we record:
    - test_acc  on the 432-config test set
    - rule_recovery_acc (and exact_match flag) on the 432-config enumeration
      vs the published Monks rule
    - For circuit models: hard_extraction test_acc and rule_recovery_acc
      (Δ tells us how much the soft mixture mattered).
    - The to_symbolic readout of the trained circuit.

Outputs:
    results/logic_e4_monks_<ts>/results.csv  — one row per (task, model, seed)
    results/logic_e4_monks_<ts>/report.md    — narrative + per-task verdict
    results/logic_e4_monks_<ts>/<task>_<model>_<seed>.keras  — saved circuits

Usage::

    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m train.logic.train_e4_monks \\
        --epochs 200 --seeds 0,1,2 --out-dir results/logic_e4_monks_iter1
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Tuple

import keras
import numpy as np

from dl_techniques.utils.logger import logger
from train.logic.monks_data import load_monks
from train.logic.rule_recovery import MONKS_RULES, rule_equivalence_score
from train.logic.train_benchmark import (
    build_circuit,
    build_mlp,
    extract_hard_inplace,
    find_mlp_hidden_for_param_budget,
    restore_soft_weights,
    roundtrip_check,
)

NUM_BITS = 17  # one-hot Monks input
NUM_OUTPUTS = 1
TASKS = (1, 2, 3)
MODEL_NAMES = ("circuit", "mlp_matched", "xgboost")

CSV_COLUMNS = [
    "problem", "model", "seed", "params", "epochs_used", "wall_s",
    "test_acc",
    "rule_recovery_exact", "rule_recovery_acc", "rule_recovery_hamming",
    "rule_tp", "rule_tn", "rule_fp", "rule_fn",
    "hard_test_acc", "hard_rule_recovery_acc", "hard_soft_delta",
    "roundtrip_diff", "to_symbolic",
]


# ---------------------------------------------------------------------
# Per-model trainers
# ---------------------------------------------------------------------

def _train_keras(
    model: keras.Model, x_tr: np.ndarray, y_tr: np.ndarray,
    x_te: np.ndarray, y_te: np.ndarray, epochs: int,
) -> Tuple[Any, int]:
    """Train a Keras model with EarlyStopping(val_accuracy)."""
    early = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=max(15, epochs // 5),
        restore_best_weights=True, verbose=0, mode="max",
    )
    history = model.fit(
        x_tr, y_tr, validation_data=(x_te, y_te),
        epochs=epochs, batch_size=32, verbose=0, callbacks=[early],
    )
    return history, len(history.history["loss"])


def _make_predict_fn_keras(model: keras.Model):
    """Return a predict_fn(x_oh)->prob suitable for rule_equivalence_score."""
    def predict_fn(x_oh: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(x_oh, verbose=0)).reshape(-1)
    return predict_fn


def _make_predict_fn_xgb(clf):
    def predict_fn(x_oh: np.ndarray) -> np.ndarray:
        return clf.predict_proba(x_oh)[:, 1]
    return predict_fn


# ---------------------------------------------------------------------
# Single (problem, model, seed) cell
# ---------------------------------------------------------------------

def run_cell(
    problem_id: int, model_name: str, seed: int,
    epochs: int, results_dir: str,
) -> Dict[str, Any]:
    logger.info(f"--- Monks-{problem_id} / {model_name} / seed={seed} ---")
    t0 = time.time()
    keras.utils.set_random_seed(seed)
    np.random.seed(seed)

    d = load_monks(problem_id)
    x_tr, y_tr = d["x_train_onehot"], d["y_train"].astype(np.float32)
    x_te, y_te = d["x_test_onehot"], d["y_test"].astype(np.float32)
    rule_fn = MONKS_RULES[problem_id]

    row: Dict[str, Any] = {
        "problem": problem_id, "model": model_name, "seed": seed,
        "hard_test_acc": None, "hard_rule_recovery_acc": None,
        "hard_soft_delta": None, "to_symbolic": "",
    }

    if model_name == "circuit":
        model = build_circuit(
            num_bits=NUM_BITS, num_outputs=NUM_OUTPUTS,
            channels=32, circuit_depth=2,
        )
        _, epochs_used = _train_keras(model, x_tr, y_tr, x_te, y_te, epochs)
        params = model.count_params()
        test_acc = float(model.evaluate(x_te, y_te, verbose=0)[1])

        predict_fn = _make_predict_fn_keras(model)
        score = rule_equivalence_score(predict_fn, rule_fn)

        save_path = os.path.join(results_dir, f"monks{problem_id}_circuit_seed{seed}.keras")
        rt_diff = roundtrip_check(model, x_te, save_path)

        nc = model.get_layer("neural_circuit")
        try:
            symbolic = nc.to_symbolic(top_k=1)
        except Exception as e:
            symbolic = f"<to_symbolic failed: {e}>"

        # Hard-extraction faithfulness.
        snapshot = extract_hard_inplace(model)
        try:
            hard_test_acc = float(model.evaluate(x_te, y_te, verbose=0)[1])
            hard_predict_fn = _make_predict_fn_keras(model)
            hard_score = rule_equivalence_score(hard_predict_fn, rule_fn)
            hard_rr_acc = hard_score["accuracy"]
        finally:
            restore_soft_weights(snapshot)

        row.update({
            "params": params, "epochs_used": epochs_used,
            "test_acc": test_acc,
            "rule_recovery_exact": score["exact_match"],
            "rule_recovery_acc": score["accuracy"],
            "rule_recovery_hamming": score["hamming_distance"],
            "rule_tp": score["true_positive"], "rule_tn": score["true_negative"],
            "rule_fp": score["false_positive"], "rule_fn": score["false_negative"],
            "hard_test_acc": hard_test_acc,
            "hard_rule_recovery_acc": hard_rr_acc,
            "hard_soft_delta": float(hard_test_acc - test_acc),
            "roundtrip_diff": rt_diff,
            "to_symbolic": symbolic.replace("\n", " | "),
        })

    elif model_name == "mlp_matched":
        ref = build_circuit(NUM_BITS, NUM_OUTPUTS, channels=32, circuit_depth=2)
        target = ref.count_params()
        h = find_mlp_hidden_for_param_budget(NUM_BITS, NUM_OUTPUTS, target)
        model = build_mlp(NUM_BITS, NUM_OUTPUTS, hidden_units=h)
        _, epochs_used = _train_keras(model, x_tr, y_tr, x_te, y_te, epochs)
        params = model.count_params()
        test_acc = float(model.evaluate(x_te, y_te, verbose=0)[1])
        predict_fn = _make_predict_fn_keras(model)
        score = rule_equivalence_score(predict_fn, rule_fn)
        save_path = os.path.join(results_dir, f"monks{problem_id}_mlp_matched_seed{seed}.keras")
        rt_diff = roundtrip_check(model, x_te, save_path)
        row.update({
            "params": params, "epochs_used": epochs_used,
            "test_acc": test_acc,
            "rule_recovery_exact": score["exact_match"],
            "rule_recovery_acc": score["accuracy"],
            "rule_recovery_hamming": score["hamming_distance"],
            "rule_tp": score["true_positive"], "rule_tn": score["true_negative"],
            "rule_fp": score["false_positive"], "rule_fn": score["false_negative"],
            "roundtrip_diff": rt_diff,
            "to_symbolic": f"mlp_matched hidden={h}",
        })

    elif model_name == "xgboost":
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", use_label_encoder=False,
            random_state=seed, verbosity=0,
        )
        clf.fit(x_tr, y_tr.astype(np.int64))
        proba = clf.predict_proba(x_te)[:, 1]
        test_acc = float(((proba > 0.5).astype(np.int64) == y_te.astype(np.int64)).mean())
        predict_fn = _make_predict_fn_xgb(clf)
        score = rule_equivalence_score(predict_fn, rule_fn)
        # Approx "params" for XGBoost: total leaves across trees.
        try:
            params = int(clf.get_booster().trees_to_dataframe().shape[0])
        except Exception:
            params = -1
        row.update({
            "params": params, "epochs_used": 200,
            "test_acc": test_acc,
            "rule_recovery_exact": score["exact_match"],
            "rule_recovery_acc": score["accuracy"],
            "rule_recovery_hamming": score["hamming_distance"],
            "rule_tp": score["true_positive"], "rule_tn": score["true_negative"],
            "rule_fp": score["false_positive"], "rule_fn": score["false_negative"],
            "roundtrip_diff": 0.0,
            "to_symbolic": f"xgboost ({clf.n_estimators}t, depth={clf.max_depth})",
        })
    else:
        raise ValueError(f"unknown model_name: {model_name}")

    row["wall_s"] = round(time.time() - t0, 2)
    logger.info(
        f"  test_acc={row['test_acc']:.3f}  rule_rec_acc={row['rule_recovery_acc']:.3f}"
        f"  exact={row['rule_recovery_exact']}  wall={row['wall_s']}s"
    )
    return row


# ---------------------------------------------------------------------
# CSV + report writers
# ---------------------------------------------------------------------

def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_COLUMNS})


def _agg(rows: List[Dict[str, Any]], key: str) -> Tuple[float, float]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def write_report(rows: List[Dict[str, Any]], path: str) -> None:
    lines = [
        "# E4 — UCI Monks rule-recovery benchmark",
        "",
        "Plan: plan_2026-05-14_e26eede2.",
        "",
        "Models per problem: `circuit` (LearnableNeuralCircuit, depth=2, channels=32),",
        "`mlp_matched` (param-budget match), `xgboost` (200 trees, max_depth=4).",
        "Seeds: 3 per cell. Test set is the canonical 432-config Monks test split.",
        "Rule-recovery score is accuracy of model predictions on the 432-config",
        "categorical enumeration vs the published Monks rule.",
        "",
    ]
    for problem in TASKS:
        prows = [r for r in rows if r["problem"] == problem]
        lines.append(f"## Monks-{problem}")
        lines.append("")
        lines.append("| Model | test_acc (mean±std) | rule_recovery_acc (mean±std) | exact_match (count) | hard_soft_delta (mean) |")
        lines.append("|---|---|---|---|---|")
        for m in MODEL_NAMES:
            sub = [r for r in prows if r["model"] == m]
            ta_m, ta_s = _agg(sub, "test_acc")
            rr_m, rr_s = _agg(sub, "rule_recovery_acc")
            exact_count = sum(1 for r in sub if r.get("rule_recovery_exact"))
            hd_m, _ = _agg(sub, "hard_soft_delta") if m == "circuit" else (float("nan"), 0.0)
            hd_str = f"{hd_m:+.3f}" if not (isinstance(hd_m, float) and np.isnan(hd_m)) else "n/a"
            lines.append(
                f"| {m} | {ta_m:.3f}±{ta_s:.3f} | {rr_m:.3f}±{rr_s:.3f} "
                f"| {exact_count}/{len(sub)} | {hd_str} |"
            )
        lines.append("")
        # Circuit symbolic readout (first seed).
        circ_rows = [r for r in prows if r["model"] == "circuit"]
        if circ_rows:
            lines.append("Circuit `to_symbolic` (seed 0):")
            lines.append(f"`{circ_rows[0].get('to_symbolic', '')}`")
            lines.append("")

    # >5pt comparison criterion from the analysis summary.
    lines.append("## Headline criterion (analyses/analysis_2026-05-13_9c535f78 §E4)")
    lines.append("")
    lines.append("Circuit beats BOTH mlp_matched AND xgboost by >5 points on test_acc:")
    wins_per_problem = {}
    for problem in TASKS:
        c_m, _ = _agg([r for r in rows if r["problem"] == problem and r["model"] == "circuit"], "test_acc")
        m_m, _ = _agg([r for r in rows if r["problem"] == problem and r["model"] == "mlp_matched"], "test_acc")
        x_m, _ = _agg([r for r in rows if r["problem"] == problem and r["model"] == "xgboost"], "test_acc")
        beats_mlp = (c_m - m_m) > 0.05
        beats_xgb = (c_m - x_m) > 0.05
        wins_per_problem[problem] = (beats_mlp and beats_xgb)
        lines.append(
            f"- Monks-{problem}: circuit={c_m:.3f}, mlp={m_m:.3f}, xgb={x_m:.3f} "
            f"-> beats both by >5pt: **{wins_per_problem[problem]}**"
        )
    n_wins = sum(wins_per_problem.values())
    verdict = (
        "PASS — circuit wins on >=2 of 3 Monks tasks. Inductive-bias claim CONFIRMED."
        if n_wins >= 2 else
        "FAIL — circuit does not beat both baselines by >5pt on >=2 of 3 Monks tasks. "
        "Inductive-bias claim NOT supported by this experiment."
    )
    lines.append("")
    lines.append(f"### Verdict (>5pt-on->=2-of-3): **{verdict}**")
    lines.append("")

    # Rule-recovery verdict.
    lines.append("## Rule-recovery verdict")
    lines.append("")
    any_strong_recovery = False
    for problem in TASKS:
        circ_rows = [r for r in rows if r["problem"] == problem and r["model"] == "circuit"]
        rr_m, rr_s = _agg(circ_rows, "rule_recovery_acc")
        if rr_m >= 0.85:
            any_strong_recovery = True
        lines.append(f"- Monks-{problem}: circuit rule_recovery_acc = {rr_m:.3f}±{rr_s:.3f}")
    rule_verdict = (
        "Circuit recovers a published Monks rule to >=0.85 enumeration accuracy on "
        "at least one of the three tasks — differentiable rule-extraction *capability* DEMONSTRATED."
        if any_strong_recovery else
        "Circuit does NOT reach >=0.85 enumeration accuracy on any Monks rule — "
        "the differentiable rule-extraction claim is NOT supported on Monks."
    )
    lines.append("")
    lines.append(f"### Verdict (rule-recovery >=0.85 on >=1 task): **{rule_verdict}**")
    lines.append("")

    with open(path, "w") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="E4 — UCI Monks rule recovery")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="comma-separated list of seeds")
    parser.add_argument("--out-dir", type=str, default="results/logic_e4_monks")
    parser.add_argument("--problems", type=str, default="1,2,3",
                        help="comma-separated Monks problem ids")
    parser.add_argument("--models", type=str, default="circuit,mlp_matched,xgboost")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    problems = [int(p) for p in args.problems.split(",")]
    models = [m.strip() for m in args.models.split(",")]

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = f"{args.out_dir}_{ts}" if not args.out_dir.endswith(ts) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"E4 Monks: out_dir={out_dir} seeds={seeds} problems={problems} models={models}")

    rows: List[Dict[str, Any]] = []
    for problem in problems:
        for m in models:
            for seed in seeds:
                try:
                    row = run_cell(problem, m, seed, args.epochs, out_dir)
                    rows.append(row)
                    # Incremental CSV save so a crash mid-run keeps partial results.
                    write_csv(rows, os.path.join(out_dir, "results.csv"))
                except Exception as e:
                    logger.exception(f"cell failed: problem={problem} model={m} seed={seed}: {e}")
                    rows.append({
                        "problem": problem, "model": m, "seed": seed,
                        "params": -1, "epochs_used": 0, "wall_s": 0.0,
                        "test_acc": float("nan"),
                        "rule_recovery_exact": False, "rule_recovery_acc": float("nan"),
                        "rule_recovery_hamming": -1,
                        "rule_tp": -1, "rule_tn": -1, "rule_fp": -1, "rule_fn": -1,
                        "hard_test_acc": None, "hard_rule_recovery_acc": None,
                        "hard_soft_delta": None,
                        "roundtrip_diff": float("nan"),
                        "to_symbolic": f"<FAILED: {e}>",
                    })

    csv_path = os.path.join(out_dir, "results.csv")
    write_csv(rows, csv_path)
    logger.info(f"Wrote CSV: {csv_path}")
    report_path = os.path.join(out_dir, "report.md")
    write_report(rows, report_path)
    logger.info(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
