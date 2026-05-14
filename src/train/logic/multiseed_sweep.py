"""Multi-seed robustness sweep over E1 (image), E3 (faithfulness), E5 (CLEVR-Hans3).

Drives the FROZEN training scripts as subprocesses across N seeds, aggregates
their CSV outputs, and writes a consolidated ``summary.md`` with mean ± std,
95% bootstrap CI on gap metrics, and a paired permutation p-value for the
E5 circuit-vs-MLP shortcut-gap difference.

Usage
-----
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 \\
        .venv/bin/python -m train.logic.multiseed_sweep \\
            --seeds 0,1,2,3,4 --experiments e1,e3,e5

Outputs
-------
    results/multiseed_<ts>/
        e1/seed_<s>/{mnist,cifar10}/benchmark_results.csv
        e3/seed_<s>/benchmark_results.csv
        e5/seed_<s>/results.csv
        e1_multiseed.csv
        e3_multiseed.csv
        e5_multiseed.csv
        summary.md
        sweep.log

Design
------
- Subprocess per (experiment, seed). Strictly serial. Each subprocess gets a
  clean TF/Keras initialization, eliminating cross-seed state contamination.
- Per-experiment wall-clock leashes; global cap defaults to 12 h.
- Honest-negative: any failed seed is logged and skipped; aggregation tolerates
  n < requested.

Plan: ``plans/plan_2026-05-14_9c6387a3``  (D-001, D-002, D-003).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from dl_techniques.utils.logger import logger
from train.logic.multiseed_stats import (
    bootstrap_ci,
    format_mean_std,
    mean_std,
    paired_permutation_test,
)


# ---------------------------------------------------------------------------
# Run specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSpec:
    """Specification for one (experiment, sub-config, seed) subprocess run."""

    experiment: str            # "e1" / "e3" / "e5"
    sub_label: str             # e.g. "mnist", "cifar10", "" for e3/e5
    seed: int
    module: str                # "train.logic.train_e1_image" etc.
    extra_args: Tuple[str, ...]
    out_dir: str
    csv_filename: str          # "benchmark_results.csv" or "results.csv"


def build_run_specs(
    *,
    experiments: Sequence[str],
    seeds: Sequence[int],
    sweep_root: str,
    skip_e5_if_no_data: bool,
    clevr_data_dir: str,
) -> List[RunSpec]:
    """Enumerate every subprocess we plan to launch."""
    specs: List[RunSpec] = []

    e5_available = (
        Path(clevr_data_dir).is_dir()
        and Path(clevr_data_dir, "CLEVR-Hans3").is_dir()
    )

    for seed in seeds:
        if "e1" in experiments:
            for ds, band in (("mnist", (0.70, 0.95)), ("cifar10", (0.50, 0.80))):
                out_dir = os.path.join(sweep_root, "e1", f"seed_{seed}", ds)
                specs.append(RunSpec(
                    experiment="e1",
                    sub_label=ds,
                    seed=seed,
                    module="train.logic.train_e1_image",
                    extra_args=(
                        "--dataset", ds,
                        "--seed", str(seed),
                        "--out-dir", out_dir,
                        "--band-low", str(band[0]),
                        "--band-high", str(band[1]),
                    ),
                    out_dir=out_dir,
                    csv_filename="benchmark_results.csv",
                ))

        if "e3" in experiments:
            out_dir = os.path.join(sweep_root, "e3", f"seed_{seed}")
            specs.append(RunSpec(
                experiment="e3",
                sub_label="",
                seed=seed,
                module="train.logic.train_e3_faithfulness",
                extra_args=(
                    "--tasks", "all",
                    "--seed", str(seed),
                    "--out-dir", out_dir,
                    # Reduced attribution sweep per LESSONS L67 (D-003).
                    "--num-attr-samples", "8",
                    "--lime-num-samples", "200",
                    "--shap-nsamples", "32",
                ),
                out_dir=out_dir,
                csv_filename="benchmark_results.csv",
            ))

        if "e5" in experiments:
            if not e5_available:
                if skip_e5_if_no_data:
                    logger.warning(
                        f"E5: CLEVR-Hans3 not found at {clevr_data_dir}; "
                        "skipping E5 (honest-negative)."
                    )
                    continue
                else:
                    raise RuntimeError(
                        f"E5: CLEVR-Hans3 not at {clevr_data_dir}; pass "
                        "--skip-e5-if-no-data to proceed without E5."
                    )
            out_dir = os.path.join(sweep_root, "e5", f"seed_{seed}")
            specs.append(RunSpec(
                experiment="e5",
                sub_label="",
                seed=seed,
                module="train.logic.train_e5_clevr_hans",
                extra_args=(
                    "--seed", str(seed),
                    "--out-dir", out_dir,
                    "--data-dir", clevr_data_dir,
                    "--skip-download",
                ),
                out_dir=out_dir,
                csv_filename="results.csv",
            ))

    return specs


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def _per_run_timeout_s(experiment: str, caps: Dict[str, float]) -> float:
    """Per-run wall-clock cap for a single subprocess invocation."""
    return float(caps.get(experiment, 3600.0))


def run_one(spec: RunSpec, timeout_s: float, env: Dict[str, str]) -> Dict[str, object]:
    """Launch one subprocess. Returns a status dict."""
    cmd = [".venv/bin/python", "-m", spec.module, *spec.extra_args]
    logger.info(
        f"[run] exp={spec.experiment} sub={spec.sub_label or '-'} "
        f"seed={spec.seed} cmd={' '.join(cmd)}"
    )
    Path(spec.out_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        cp = subprocess.run(
            cmd, check=False, timeout=timeout_s, env=env,
            cwd=str(Path(__file__).resolve().parents[3]),  # repo root
            capture_output=True, text=True,
        )
        dt = time.time() - t0
        ok = cp.returncode == 0
        status = "ok" if ok else f"exit_{cp.returncode}"
        if not ok:
            logger.warning(
                f"[run] non-zero exit={cp.returncode} for "
                f"{spec.experiment}/{spec.sub_label}/seed_{spec.seed}; "
                f"stderr tail: {cp.stderr[-500:] if cp.stderr else '(empty)'}"
            )
        else:
            logger.info(
                f"[run] OK exp={spec.experiment} sub={spec.sub_label or '-'} "
                f"seed={spec.seed} wall_s={dt:.1f}"
            )
        return {"status": status, "wall_s": dt, "stderr_tail": cp.stderr[-2000:] if cp.stderr else ""}
    except subprocess.TimeoutExpired:
        dt = time.time() - t0
        logger.warning(
            f"[run] TIMEOUT after {dt:.1f}s for "
            f"{spec.experiment}/{spec.sub_label}/seed_{spec.seed}"
        )
        return {"status": "timeout", "wall_s": dt, "stderr_tail": ""}
    except Exception as e:  # pragma: no cover (defensive)
        dt = time.time() - t0
        logger.exception(f"[run] UNEXPECTED error: {e}")
        return {"status": f"error:{type(e).__name__}", "wall_s": dt, "stderr_tail": str(e)}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def collect_csvs(
    specs: Sequence[RunSpec], experiment: str
) -> pd.DataFrame:
    """Glob this experiment's per-seed CSVs into one DataFrame with ``seed`` column."""
    frames: List[pd.DataFrame] = []
    for spec in specs:
        if spec.experiment != experiment:
            continue
        csv_path = os.path.join(spec.out_dir, spec.csv_filename)
        if not os.path.isfile(csv_path):
            logger.warning(
                f"[agg] missing CSV {csv_path} (exp={experiment} seed={spec.seed})"
            )
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.warning(f"[agg] failed to read {csv_path}: {e}")
            continue
        df["seed"] = spec.seed
        if spec.sub_label:
            # E1 needs an explicit sub_label column when out-dir has nested ds.
            # The CSV already carries `dataset`, so the sub_label is redundant
            # but we keep it for forensic clarity.
            df["sub_label"] = spec.sub_label
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

E1_METRICS = ["final_val_acc", "band_acc", "delta_hard", "roundtrip_diff", "hard_acc", "soft_acc"]
E3_METRICS = [
    "band_acc", "delta_hard",
    "circuit_suff_auc", "lime_suff_auc", "shap_suff_auc",
    "circuit_sparsity", "lime_sparsity", "shap_sparsity",
    "circuit_stability", "lime_stability", "shap_stability",
]
E5_METRICS = ["val_acc", "test_acc", "shortcut_gap"]

# Prior n=1 (seed=42) reference values for regression check.
PRIOR_N1: Dict[Tuple[str, str, str], Dict[str, float]] = {
    ("e1", "mnist", "circuit"):     {"final_val_acc": 0.7006, "delta_hard": 0.0008,  "band_acc": 0.7006, "roundtrip_diff": 0.0},
    ("e1", "mnist", "cnn_matched"): {"final_val_acc": 0.6057},
    ("e1", "cifar10", "circuit"):     {"final_val_acc": 0.5552, "delta_hard": -0.0009, "band_acc": 0.5552, "roundtrip_diff": 0.0},
    ("e1", "cifar10", "cnn_matched"): {"final_val_acc": 0.5208},
    ("e3", "", "circuit:mux_11bit"):              {"band_acc": 0.7031, "delta_hard": -0.0156},
    ("e3", "", "circuit:parity_k8"):              {"band_acc": 0.7246, "delta_hard": 0.0039},
    ("e3", "", "circuit:random_dnf_8input_4term"):{"band_acc": 0.7100, "delta_hard": 0.0},
    ("e5", "", "resnet50_circuit"):       {"val_acc": 0.8547, "test_acc": 0.6742, "shortcut_gap": 0.1804},
    ("e5", "", "resnet50_mlp"):           {"val_acc": 0.8533, "test_acc": 0.6858, "shortcut_gap": 0.1676},
    ("e5", "", "symbolic_circuit_oracle"):{"val_acc": 1.0000, "test_acc": 0.9978, "shortcut_gap": 0.0022},
}


def _group_stats(
    df: pd.DataFrame, group_cols: Sequence[str], metric_cols: Sequence[str],
    *, rng: np.random.Generator,
) -> pd.DataFrame:
    """Per-group mean ± std with bootstrap CI for each metric column."""
    rows: List[Dict[str, object]] = []
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(list(group_cols), dropna=False)
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row: Dict[str, object] = {c: v for c, v in zip(group_cols, key)}
        row["n_seeds"] = int(sub["seed"].nunique())
        for m in metric_cols:
            if m not in sub.columns:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_std"] = float("nan")
                row[f"{m}_ci_lo"] = float("nan")
                row[f"{m}_ci_hi"] = float("nan")
                continue
            vals = sub[m].to_numpy(dtype=float)
            mean, std = mean_std(vals)
            ci_lo, ci_hi = bootstrap_ci(vals, rng=rng)
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_ci_lo"] = ci_lo
            row[f"{m}_ci_hi"] = ci_hi
        rows.append(row)
    return pd.DataFrame(rows)


def _high_variance_flags(stats_df: pd.DataFrame, metrics: Sequence[str]) -> List[str]:
    """Return a list of '(key) metric: std=X > |mean|=Y' flags for the report."""
    flags: List[str] = []
    if stats_df.empty:
        return flags
    for _, row in stats_df.iterrows():
        # Build a key string from non-metric columns (e.g. dataset/model/task).
        key_cols = [c for c in stats_df.columns
                    if not (c.endswith("_mean") or c.endswith("_std")
                            or c.endswith("_ci_lo") or c.endswith("_ci_hi")
                            or c == "n_seeds")]
        key = "/".join(str(row[c]) for c in key_cols)
        for m in metrics:
            mean = row.get(f"{m}_mean")
            std = row.get(f"{m}_std")
            if mean is None or std is None:
                continue
            if not np.isfinite(mean) or not np.isfinite(std):
                continue
            if abs(mean) > 0 and std > abs(mean):
                flags.append(
                    f"  - HIGH-VARIANCE: {key} / {m}: "
                    f"std={std:.4f} > |mean|={abs(mean):.4f} → retract or qualify"
                )
            elif mean == 0.0 and std > 0.01:
                flags.append(
                    f"  - HIGH-VARIANCE: {key} / {m}: std={std:.4f} (mean=0.0)"
                )
    return flags


def _regression_flags(
    stats_df: pd.DataFrame, experiment: str, key_cols: Sequence[str]
) -> List[str]:
    """Compare each prior n=1 value vs new mean ± 2*std. Flag outliers."""
    flags: List[str] = []
    if stats_df.empty:
        return flags
    for _, row in stats_df.iterrows():
        if experiment == "e1":
            sub = str(row.get("dataset", ""))
            model = str(row.get("model", ""))
            key = (experiment, sub, model)
        elif experiment == "e3":
            task = str(row.get("task", ""))
            model = str(row.get("model", "circuit"))
            key = (experiment, "", f"{model}:{task}")
        else:  # e5
            model = str(row.get("model", ""))
            key = (experiment, "", model)
        priors = PRIOR_N1.get(key, {})
        if not priors:
            continue
        for metric, prior_val in priors.items():
            mean = row.get(f"{metric}_mean")
            std = row.get(f"{metric}_std")
            if mean is None or std is None:
                continue
            if not (np.isfinite(mean) and np.isfinite(std)):
                continue
            band_lo = mean - 2.0 * std
            band_hi = mean + 2.0 * std
            if prior_val < band_lo or prior_val > band_hi:
                flags.append(
                    f"  - REGRESSION: {'/'.join(map(str, key))} / {metric}: "
                    f"prior_n1={prior_val:.4f} outside new [mean ± 2*std]="
                    f"[{band_lo:.4f}, {band_hi:.4f}]"
                )
    return flags


def _table_for_md(
    stats_df: pd.DataFrame, key_cols: Sequence[str], metrics: Sequence[str]
) -> str:
    """Render a markdown table with one row per group, columns = mean ± std."""
    if stats_df.empty:
        return "*(no data)*\n"
    header_cols = list(key_cols) + ["n"] + list(metrics)
    lines = ["| " + " | ".join(header_cols) + " |",
             "|" + "|".join(["---"] * len(header_cols)) + "|"]
    for _, row in stats_df.iterrows():
        cells = [str(row[c]) for c in key_cols]
        cells.append(str(int(row["n_seeds"])))
        for m in metrics:
            mean = row.get(f"{m}_mean")
            std = row.get(f"{m}_std")
            cells.append(format_mean_std(mean, std))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _e5_permutation_section(
    e5_df: pd.DataFrame, rng: np.random.Generator
) -> str:
    """Paired permutation test on shortcut_gap differences (circuit - mlp)."""
    if e5_df.empty:
        return "*(no E5 data — section skipped)*\n"
    if "model" not in e5_df.columns or "shortcut_gap" not in e5_df.columns:
        return "*(E5 CSV missing model/shortcut_gap columns)*\n"

    circuit = e5_df[e5_df["model"] == "resnet50_circuit"][["seed", "shortcut_gap"]]
    mlp     = e5_df[e5_df["model"] == "resnet50_mlp"][["seed", "shortcut_gap"]]
    merged = circuit.merge(mlp, on="seed", suffixes=("_circuit", "_mlp")).sort_values("seed")
    if len(merged) == 0:
        return "*(no paired E5 seeds — section skipped)*\n"

    a = merged["shortcut_gap_circuit"].to_numpy(dtype=float)
    b = merged["shortcut_gap_mlp"].to_numpy(dtype=float)
    obs, p = paired_permutation_test(a, b, n_perm=10000, rng=rng)
    diffs = a - b
    diff_mean, diff_std = mean_std(diffs)
    diff_ci_lo, diff_ci_hi = bootstrap_ci(diffs, rng=rng)

    interp = "no significant difference"
    if p < 0.05:
        if obs > 0:
            interp = "circuit has LARGER shortcut_gap (worse) than MLP"
        else:
            interp = "circuit has SMALLER shortcut_gap (better) than MLP"
    elif p < 0.1:
        interp = "marginal — circuit-vs-MLP gap difference is borderline"

    return (
        f"### E5 paired permutation test (circuit_shortcut_gap - mlp_shortcut_gap)\n\n"
        f"- n_pairs: {len(merged)}\n"
        f"- mean(circuit - mlp): {format_mean_std(diff_mean, diff_std)}\n"
        f"- 95% bootstrap CI of mean diff: [{diff_ci_lo:.4f}, {diff_ci_hi:.4f}]\n"
        f"- Permutation test (B=10000, two-sided, Phipson-Smyth corrected): "
        f"observed_diff={obs:.4f}, **p={p:.4f}**\n"
        f"- Interpretation: {interp}\n"
    )


def _claim_survival_section(
    stats_e1: pd.DataFrame, stats_e3: pd.DataFrame, stats_e5: pd.DataFrame
) -> str:
    """Audit which prior n=1 claims now have non-degenerate error bars."""
    lines: List[str] = ["### Claim survival audit\n",
                        "Prior n=1 claims (seed=42) with new n>=2 error bars:\n"]
    lines.append("| Experiment | Key | Metric | Prior n=1 | New mean ± std | Status |")
    lines.append("|---|---|---|---|---|---|")
    for (exp, sub, model), priors in PRIOR_N1.items():
        if exp == "e1":
            df = stats_e1
            if df.empty:
                continue
            mask = (df.get("dataset") == sub) & (df.get("model") == model)
        elif exp == "e3":
            df = stats_e3
            if df.empty:
                continue
            # model="circuit:task" → split
            _, task = model.split(":", 1)
            mask = (df.get("task") == task)
        else:
            df = stats_e5
            if df.empty:
                continue
            mask = (df.get("model") == model)
        if not mask.any():
            continue
        row = df[mask].iloc[0]
        for metric, prior_val in priors.items():
            mean = row.get(f"{metric}_mean")
            std = row.get(f"{metric}_std")
            if mean is None or not np.isfinite(mean):
                status = "no data"
                cell = "—"
            else:
                cell = format_mean_std(mean, std)
                if std == 0.0 or not np.isfinite(std):
                    status = "saturated (std=0)"
                elif abs(prior_val - mean) <= 2.0 * std:
                    status = "WITHIN ± 2*std"
                else:
                    status = "OUTLIER"
            lines.append(
                f"| {exp} | {sub or '-'}/{model} | {metric} | "
                f"{prior_val:.4f} | {cell} | {status} |"
            )
    return "\n".join(lines) + "\n"


def write_summary(
    *,
    sweep_root: str,
    e1_df: pd.DataFrame, e3_df: pd.DataFrame, e5_df: pd.DataFrame,
    run_log: List[Dict[str, object]],
    seeds: Sequence[int],
    rng_seed: int = 20260514,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute aggregates and write summary.md. Returns the three stats DataFrames."""
    rng = np.random.default_rng(rng_seed)

    stats_e1 = _group_stats(
        e1_df, group_cols=["dataset", "model"], metric_cols=E1_METRICS, rng=rng
    )
    stats_e3 = _group_stats(
        e3_df, group_cols=["task", "model"], metric_cols=E3_METRICS, rng=rng
    )
    stats_e5 = _group_stats(
        e5_df, group_cols=["model"], metric_cols=E5_METRICS, rng=rng
    )

    md_path = os.path.join(sweep_root, "summary.md")
    parts: List[str] = []
    parts.append(f"# Multi-seed Sweep Summary\n")
    parts.append(f"Seeds requested: {list(seeds)}\n")
    parts.append(f"Sweep root: `{sweep_root}`\n")
    parts.append(f"RNG seed for bootstrap/permutation: {rng_seed}\n\n")

    parts.append("## Run status\n")
    parts.append("| experiment | sub | seed | status | wall_s |")
    parts.append("|---|---|---|---|---|")
    for r in run_log:
        parts.append(
            f"| {r['experiment']} | {r['sub_label'] or '-'} | {r['seed']} | "
            f"{r['status']} | {float(r['wall_s']):.1f} |"
        )
    parts.append("")

    parts.append("## E1 (MNIST / CIFAR-10) — mean ± std per (dataset, model)\n")
    parts.append(_table_for_md(stats_e1, ["dataset", "model"], E1_METRICS))

    parts.append("\n## E3 (boolean tasks) — mean ± std per (task, model)\n")
    parts.append(_table_for_md(stats_e3, ["task", "model"], E3_METRICS))

    parts.append("\n## E5 (CLEVR-Hans3) — mean ± std per model\n")
    parts.append(_table_for_md(stats_e5, ["model"], E5_METRICS))

    parts.append("\n## E5 inference\n")
    parts.append(_e5_permutation_section(e5_df, rng))

    parts.append("\n## Regression vs prior n=1 (seed=42)\n")
    reg_flags = (
        _regression_flags(stats_e1, "e1", ["dataset", "model"])
        + _regression_flags(stats_e3, "e3", ["task", "model"])
        + _regression_flags(stats_e5, "e5", ["model"])
    )
    if reg_flags:
        parts.append("\n".join(reg_flags) + "\n")
    else:
        parts.append("*(no prior n=1 value fell outside new mean ± 2*std)*\n")

    parts.append("\n## High-variance flags (std > |mean|)\n")
    hv_flags = (
        _high_variance_flags(stats_e1, E1_METRICS)
        + _high_variance_flags(stats_e3, E3_METRICS)
        + _high_variance_flags(stats_e5, E5_METRICS)
    )
    if hv_flags:
        parts.append("\n".join(hv_flags) + "\n")
    else:
        parts.append("*(no metric has std > |mean|)*\n")

    parts.append("\n" + _claim_survival_section(stats_e1, stats_e3, stats_e5))

    Path(md_path).write_text("\n".join(parts))
    logger.info(f"[agg] wrote {md_path}")
    return stats_e1, stats_e3, stats_e5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-seed robustness sweep (E1 + E3 + E5)."
    )
    p.add_argument("--seeds", type=str, default="0,1,2,3,4",
                   help="Comma-separated list of seeds.")
    p.add_argument("--experiments", type=str, default="e1,e3,e5",
                   help="Comma-separated subset of {e1,e3,e5}.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Sweep root. Defaults to results/multiseed_<ts>/.")
    p.add_argument("--clevr-data-dir", type=str, default="data/clevr_hans3",
                   help="Path to CLEVR-Hans3 root (must contain CLEVR-Hans3/).")
    p.add_argument("--skip-e5-if-no-data", action="store_true",
                   help="Skip E5 silently if CLEVR-Hans3 not on disk.")
    # Per-run timeouts (seconds). Generous; only fire on wedges.
    p.add_argument("--e1-timeout-s", type=float, default=3600.0,
                   help="Per-(seed, dataset) timeout for E1.")
    p.add_argument("--e3-timeout-s", type=float, default=3600.0,
                   help="Per-seed timeout for E3.")
    p.add_argument("--e5-timeout-s", type=float, default=7200.0,
                   help="Per-seed timeout for E5.")
    p.add_argument("--global-cap-s", type=float, default=43200.0,
                   help="Global hard cap on cumulative wall-clock (default 12h).")
    p.add_argument("--rng-seed-bootstrap", type=int, default=20260514,
                   help="Seed for bootstrap/permutation deterministic RNG.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]

    ts = time.strftime("%Y%m%d_%H%M%S")
    sweep_root = args.out_dir or os.path.join("results", f"multiseed_{ts}")
    Path(sweep_root).mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Multi-seed sweep ===")
    logger.info(f"seeds      : {seeds}")
    logger.info(f"experiments: {experiments}")
    logger.info(f"out_dir    : {sweep_root}")

    specs = build_run_specs(
        experiments=experiments, seeds=seeds, sweep_root=sweep_root,
        skip_e5_if_no_data=args.skip_e5_if_no_data,
        clevr_data_dir=args.clevr_data_dir,
    )
    logger.info(f"total subprocess runs planned: {len(specs)}")

    caps = {"e1": args.e1_timeout_s, "e3": args.e3_timeout_s, "e5": args.e5_timeout_s}
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # Ensure src/ is importable inside subprocess.
    repo_root = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = f"{repo_root / 'src'}:{env.get('PYTHONPATH', '')}"

    run_log: List[Dict[str, object]] = []
    t_global = time.time()
    for spec in specs:
        if time.time() - t_global > args.global_cap_s:
            logger.warning(
                f"GLOBAL CAP {args.global_cap_s}s hit; skipping remaining "
                f"{len(specs) - len(run_log)} runs."
            )
            break
        result = run_one(spec, _per_run_timeout_s(spec.experiment, caps), env)
        run_log.append({
            "experiment": spec.experiment, "sub_label": spec.sub_label,
            "seed": spec.seed, **result,
        })

    # Aggregation
    e1_df = collect_csvs(specs, "e1")
    e3_df = collect_csvs(specs, "e3")
    e5_df = collect_csvs(specs, "e5")
    if not e1_df.empty:
        e1_df.to_csv(os.path.join(sweep_root, "e1_multiseed.csv"), index=False)
    if not e3_df.empty:
        e3_df.to_csv(os.path.join(sweep_root, "e3_multiseed.csv"), index=False)
    if not e5_df.empty:
        e5_df.to_csv(os.path.join(sweep_root, "e5_multiseed.csv"), index=False)

    write_summary(
        sweep_root=sweep_root,
        e1_df=e1_df, e3_df=e3_df, e5_df=e5_df,
        run_log=run_log, seeds=seeds, rng_seed=args.rng_seed_bootstrap,
    )
    logger.info(f"DONE. Outputs in {sweep_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
