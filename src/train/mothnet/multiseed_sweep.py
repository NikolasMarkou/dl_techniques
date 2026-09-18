"""MothNet ``mb_units`` baseline-vs-candidate multiseed A/B sweep driver.

Drives ``train.mothnet.train_mothnet`` as subprocesses across N seeds x 2 arms
(baseline ``--mb-units``, candidate ``--mb-units``), aggregates each run's own
``training_log.csv``, and writes a consolidated ``summary.md`` with per-arm
mean +/- std, 95% bootstrap CI, and a paired permutation p-value on the
candidate-vs-baseline ``val_accuracy`` difference.

Usage
-----
    MPLBACKEND=Agg .venv/bin/python -m train.mothnet.multiseed_sweep \\
        --seeds 1,2,3,4,5,6,7,8 --mb-units-baseline 2000 --mb-units-candidate 16000

Outputs: ``results/mothnet_mb_units_sweep_<ts>/{raw_results.csv,summary.md}``, plus
one ``results/mothnet_sweep_<arm>_seed<seed>/`` per subprocess (the trainer's own
standard run-directory shape).

Reuse of an already-earned repo-wide pattern (``RunSpec``/``run_one``/
``collect_csvs``/a paired-permutation section/a ``summary.md`` writer), copied and
narrowed from ``src/train/logic/multiseed_sweep.py`` (663 lines, multi-experiment
generality) to ONE two-arm comparison — two existing call sites
(``src/train/logic/multiseed_sweep.py``, ``src/train/rms_variants_train/report.py``)
already earned this shape; no new abstraction is added here. Subprocess-per-seed
(not an in-process loop) is deliberate: each subprocess gets a clean TF/Keras
init, and a crashed/timed-out seed is logged and skipped, not fatal.

Plan: ``plans/plan-2026-09-18T060057-c1cfc3d3`` (Step 8).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from dl_techniques.utils.logger import logger
from train.common.stats import (
    bootstrap_ci,
    format_mean_std,
    mean_std,
    min_pairs_for_significance,
    min_reachable_p_signflip,
    paired_permutation_test,
)

# Matches train_mothnet.py's own REPO_ROOT derivation
# (src/train/mothnet/multiseed_sweep.py: [0] mothnet, [1] train, [2] src, [3] <repo>).
REPO_ROOT = Path(__file__).resolve().parents[3]

MB_UNITS_ARMS: Tuple[str, str] = ("baseline", "candidate")

# Canonical scale fixed by plan.md Step 9's decision rule — NOT exposed as flags,
# since varying them would change what the sweep measures.
_EPOCHS = 3
_NUM_TRAIN_SAMPLES = 2000
_NUM_VAL_SAMPLES = 500


@dataclass(frozen=True)
class RunSpec:
    """Specification for one (mb_units arm, seed) subprocess run."""

    arm: str            # "baseline" | "candidate"
    mb_units: int
    seed: int
    gpu: int
    experiment_name: str


def build_run_specs(
    seeds: Sequence[int], mb_units_baseline: int, mb_units_candidate: int, gpu: int,
) -> List[RunSpec]:
    """Enumerate the 2-arm x N-seed cross-product; baseline arm first, then candidate."""
    arm_values = {"baseline": mb_units_baseline, "candidate": mb_units_candidate}
    specs: List[RunSpec] = []
    for arm in MB_UNITS_ARMS:
        for seed in seeds:
            specs.append(RunSpec(
                arm=arm,
                mb_units=arm_values[arm],
                seed=seed,
                gpu=gpu,
                experiment_name=f"mothnet_sweep_{arm}_seed{seed}",
            ))
    return specs


def run_one(spec: RunSpec, timeout_s: float, env: Dict[str, str]) -> Dict[str, object]:
    """Launch one ``train.mothnet.train_mothnet`` subprocess.

    Non-zero exit and timeout are logged as WARNING and tolerated (the caller
    continues the sweep with the remaining runs). ``--viz-freq 0`` skips
    visualization overhead — not needed for the sweep.

    :return: ``{"status": "ok"|"exit_<code>"|"timeout"|"error:<Type>",
        "wall_s": float, "stderr_tail": str}``.
    """
    cmd = [
        ".venv/bin/python", "-m", "train.mothnet.train_mothnet",
        "--mb-units", str(spec.mb_units),
        "--seed", str(spec.seed),
        "--epochs", str(_EPOCHS),
        "--num-train-samples", str(_NUM_TRAIN_SAMPLES),
        "--num-val-samples", str(_NUM_VAL_SAMPLES),
        "--gpu", str(spec.gpu),
        "--experiment-name", spec.experiment_name,
        "--viz-freq", "0",
    ]
    logger.info(
        f"[run] arm={spec.arm} mb_units={spec.mb_units} seed={spec.seed} "
        f"cmd={' '.join(cmd)}"
    )
    t0 = time.time()
    try:
        cp = subprocess.run(
            cmd, check=False, timeout=timeout_s, env=env, cwd=str(REPO_ROOT),
            capture_output=True, text=True,
        )
        dt = time.time() - t0
        ok = cp.returncode == 0
        status = "ok" if ok else f"exit_{cp.returncode}"
        if not ok:
            logger.warning(
                f"[run] non-zero exit={cp.returncode} for arm={spec.arm} "
                f"seed={spec.seed}; stderr tail: "
                f"{cp.stderr[-500:] if cp.stderr else '(empty)'}"
            )
        else:
            logger.info(f"[run] OK arm={spec.arm} seed={spec.seed} wall_s={dt:.1f}")
        return {
            "status": status, "wall_s": dt,
            "stderr_tail": cp.stderr[-2000:] if cp.stderr else "",
        }
    except subprocess.TimeoutExpired:
        dt = time.time() - t0
        logger.warning(f"[run] TIMEOUT after {dt:.1f}s for arm={spec.arm} seed={spec.seed}")
        return {"status": "timeout", "wall_s": dt, "stderr_tail": ""}
    except Exception as e:  # pragma: no cover (defensive)
        dt = time.time() - t0
        logger.exception(f"[run] UNEXPECTED error: {e}")
        return {"status": f"error:{type(e).__name__}", "wall_s": dt, "stderr_tail": str(e)}


def collect_csvs(specs: Sequence[RunSpec], results_root: Path) -> pd.DataFrame:
    """Read each spec's final ``training_log.csv`` row into one flat DataFrame.

    One row per spec: columns ``arm, mb_units, seed, val_accuracy, status``. A
    missing/unreadable/empty CSV yields ``val_accuracy=NaN, status="failed"``
    rather than raising — a partially-failed sweep must not crash aggregation.
    """
    rows: List[Dict[str, object]] = []
    for spec in specs:
        csv_path = Path(results_root) / spec.experiment_name / "training_log.csv"
        val_accuracy = float("nan")
        status = "failed"
        if not csv_path.is_file():
            logger.warning(f"[agg] missing CSV {csv_path} (arm={spec.arm} seed={spec.seed})")
        else:
            try:
                run_df = pd.read_csv(csv_path)
                if run_df.empty or "val_accuracy" not in run_df.columns:
                    logger.warning(
                        f"[agg] {csv_path} has no val_accuracy row "
                        f"(arm={spec.arm} seed={spec.seed})"
                    )
                else:
                    val_accuracy = float(run_df["val_accuracy"].iloc[-1])
                    status = "ok"
            except Exception as e:
                logger.warning(f"[agg] failed to read {csv_path}: {e}")
        rows.append({
            "arm": spec.arm, "mb_units": spec.mb_units, "seed": spec.seed,
            "val_accuracy": val_accuracy, "status": status,
        })
    return pd.DataFrame(rows)


def run_permutation_section(df: pd.DataFrame, rng: np.random.Generator) -> Dict[str, object]:
    """Paired permutation test of ``val_accuracy``, candidate vs. baseline, by seed.

    Guard: ``len(a) < 2 or len(b) < 2 or len(a) != len(b)`` before calling
    ``paired_permutation_test`` (mirrors ``rms_variants_train/report.py``'s
    ``_add_paired_p`` shape) — a partially-failed sweep reports a reduced/
    insufficient ``n_pairs`` via ``result["warning"]`` rather than crashing or
    comparing misaligned arrays. Also flags an ``n_pairs`` below
    ``min_pairs_for_significance(family_size=1)``: below that floor the test
    cannot reject at ANY effect size, so a non-significant p is uninformative.

    :return: dict with per-arm ``mean``/``std``/``ci``, ``n_pairs``, ``mean_diff``,
        ``diff_std``, ``diff_ci``, ``p_value``, optional ``warning``.
    """
    ok = df[df["status"] == "ok"]
    baseline_vals = ok[ok["arm"] == "baseline"]["val_accuracy"].to_numpy(dtype=float)
    candidate_vals = ok[ok["arm"] == "candidate"]["val_accuracy"].to_numpy(dtype=float)

    baseline_mean, baseline_std = mean_std(baseline_vals)
    candidate_mean, candidate_std = mean_std(candidate_vals)
    baseline_ci = bootstrap_ci(baseline_vals, rng=rng)
    candidate_ci = bootstrap_ci(candidate_vals, rng=rng)

    result: Dict[str, object] = {
        "n_baseline": int(baseline_vals.size), "n_candidate": int(candidate_vals.size),
        "baseline_mean": baseline_mean, "baseline_std": baseline_std, "baseline_ci": baseline_ci,
        "candidate_mean": candidate_mean, "candidate_std": candidate_std, "candidate_ci": candidate_ci,
        "n_pairs": 0,
        "mean_diff": float("nan"), "diff_std": float("nan"),
        "diff_ci": (float("nan"), float("nan")), "p_value": float("nan"),
        "warning": None,
    }

    baseline_paired = ok[ok["arm"] == "baseline"][["seed", "val_accuracy"]]
    candidate_paired = ok[ok["arm"] == "candidate"][["seed", "val_accuracy"]]
    merged = baseline_paired.merge(
        candidate_paired, on="seed", suffixes=("_baseline", "_candidate"),
    ).sort_values("seed")
    a = merged["val_accuracy_candidate"].to_numpy(dtype=float)
    b = merged["val_accuracy_baseline"].to_numpy(dtype=float)

    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        result["warning"] = (
            f"insufficient/misaligned paired seeds (n_pairs={len(a)}) — "
            "paired_permutation_test not run"
        )
        logger.warning(f"[stats] {result['warning']}")
        return result

    obs, p = paired_permutation_test(a, b, n_perm=10000, rng=rng)
    diffs = a - b
    _, diff_std = mean_std(diffs)
    diff_ci = bootstrap_ci(diffs, rng=rng)

    result["n_pairs"] = len(merged)
    result["mean_diff"] = obs
    result["diff_std"] = diff_std
    result["diff_ci"] = diff_ci
    result["p_value"] = p

    floor = min_pairs_for_significance(family_size=1)
    if len(merged) < floor:
        result["warning"] = (
            f"n_pairs={len(merged)} is BELOW min_pairs_for_significance(family_size=1)"
            f"={floor} — the smallest two-sided p this test could ever report at this "
            f"n is min_reachable_p_signflip({len(merged)})="
            f"{min_reachable_p_signflip(len(merged)):.4f}; a non-significant result "
            "here is UNINFORMATIVE, not evidence of no effect."
        )
        logger.warning(f"[stats] {result['warning']}")

    return result


def write_summary(
    *,
    sweep_root: Path,
    df: pd.DataFrame,
    run_log: List[Dict[str, object]],
    seeds: Sequence[int],
    mb_units_baseline: int,
    mb_units_candidate: int,
    perm: Dict[str, object],
    rng_seed: int,
) -> None:
    """Write ``sweep_root/summary.md``: results table + permutation-test outcome."""
    parts: List[str] = []
    parts.append("# MothNet mb_units Sweep Summary\n")
    parts.append(f"Seeds: {list(seeds)}\n")
    parts.append(f"mb_units — baseline: {mb_units_baseline}, candidate: {mb_units_candidate}\n")
    parts.append(f"Sweep root: `{sweep_root}`\n")
    parts.append(f"RNG seed for bootstrap/permutation: {rng_seed}\n")

    parts.append("\n## Run status\n")
    parts.append("| arm | mb_units | seed | status | wall_s |")
    parts.append("|---|---|---|---|---|")
    for r in run_log:
        parts.append(
            f"| {r['arm']} | {r['mb_units']} | {r['seed']} | {r['status']} | "
            f"{float(r['wall_s']):.1f} |"
        )

    parts.append("\n## Results (final-epoch val_accuracy per run)\n")
    parts.append("| arm | mb_units | seed | val_accuracy | status |")
    parts.append("|---|---|---|---|---|")
    for _, row in df.iterrows():
        va = "nan" if not np.isfinite(row["val_accuracy"]) else f"{row['val_accuracy']:.4f}"
        parts.append(
            f"| {row['arm']} | {row['mb_units']} | {row['seed']} | {va} | {row['status']} |"
        )

    parts.append("\n## Paired permutation test (candidate - baseline, val_accuracy)\n")
    parts.append(
        f"- baseline (mb_units={mb_units_baseline}): n={perm['n_baseline']}, "
        f"{format_mean_std(perm['baseline_mean'], perm['baseline_std'])}, "
        f"95% CI [{perm['baseline_ci'][0]:.4f}, {perm['baseline_ci'][1]:.4f}]"
    )
    parts.append(
        f"- candidate (mb_units={mb_units_candidate}): n={perm['n_candidate']}, "
        f"{format_mean_std(perm['candidate_mean'], perm['candidate_std'])}, "
        f"95% CI [{perm['candidate_ci'][0]:.4f}, {perm['candidate_ci'][1]:.4f}]"
    )
    parts.append(f"- n_pairs: {perm['n_pairs']}")
    if perm["n_pairs"] > 0:
        parts.append(
            f"- mean(candidate - baseline): "
            f"{format_mean_std(perm['mean_diff'], perm['diff_std'])}"
        )
        parts.append(
            f"- 95% bootstrap CI of mean diff: "
            f"[{perm['diff_ci'][0]:.4f}, {perm['diff_ci'][1]:.4f}]"
        )
        parts.append(
            f"- Permutation test (B=10000, two-sided, Phipson-Smyth corrected): "
            f"observed_diff={perm['mean_diff']:.4f}, **p={perm['p_value']:.4f}**"
        )
    else:
        parts.append("- Permutation test: NOT RUN (see warning below)")

    if perm["warning"]:
        parts.append(f"\n**WARNING**: {perm['warning']}")

    interp = "no significant difference"
    if perm["n_pairs"] > 0 and np.isfinite(perm["p_value"]):
        if perm["p_value"] < 0.05:
            direction = "LARGER" if perm["mean_diff"] > 0 else "SMALLER"
            interp = f"candidate has {direction} val_accuracy than baseline (p < 0.05)"
        elif perm["p_value"] < 0.1:
            interp = "marginal — candidate-vs-baseline difference is borderline"
    parts.append(f"\n- Interpretation: {interp}\n")

    parts.append(
        "Decision rule (plan.md Step 9, pre-registered in decisions.md D-009): raise "
        "the CLI `--mb-units` default from 2000 to 16000 ONLY IF p < 0.05 (two-sided) "
        "AND |mean_diff(val_accuracy)| >= 0.07; otherwise leave the default unchanged "
        "and record the null/inconclusive result.\n"
    )

    md_path = Path(sweep_root) / "summary.md"
    md_path.write_text("\n".join(parts))
    logger.info(f"[agg] wrote {md_path}")


def parse_arguments(argv=None) -> argparse.Namespace:
    """Parse CLI arguments for the mb_units sweep driver."""
    parser = argparse.ArgumentParser(
        description=(
            "Run MothNet's mb_units baseline-vs-candidate A/B sweep across N paired "
            "seeds, as subprocess invocations of train.mothnet.train_mothnet."
        ),
    )
    parser.add_argument(
        "--seeds", type=str, default="1,2,3,4,5,6,7,8",
        help=(
            "Comma-separated seed list; one paired run per seed per arm "
            "(default: 8 seeds, above the family_size=1 statistical floor of 6)."
        ),
    )
    parser.add_argument(
        "--mb-units-baseline", type=int, default=2000,
        help=(
            "mb_units value for the baseline arm (default: 2000 -- train_mothnet.py's "
            "pre-D-009 CLI default; the current live default is 16000, see "
            "--mb-units-candidate)."
        ),
    )
    parser.add_argument(
        "--mb-units-candidate", type=int, default=16000,
        help="mb_units value for the candidate arm (default: 16000).",
    )
    parser.add_argument(
        "--rng-seed-bootstrap", type=int, default=20260918,
        help="Seed for bootstrap/permutation deterministic RNG (default: 20260918).",
    )
    parser.add_argument(
        "--timeout-s", type=float, default=120.0,
        help="Per-run subprocess wall-clock timeout in seconds (default: 120.0).",
    )
    parser.add_argument(
        "--gpu", type=int, default=1,
        help="GPU index forwarded to each subprocess's own --gpu flag (default: 1).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Entry point. First statement parses argv — ``--help`` exits 0, no side effects."""
    args = parse_arguments(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    specs = build_run_specs(
        seeds=seeds, mb_units_baseline=args.mb_units_baseline,
        mb_units_candidate=args.mb_units_candidate, gpu=args.gpu,
    )
    logger.info(
        f"=== MothNet mb_units sweep === seeds={seeds} "
        f"mb_units_baseline={args.mb_units_baseline} "
        f"mb_units_candidate={args.mb_units_candidate} "
        f"total subprocess runs planned: {len(specs)}"
    )

    results_root = REPO_ROOT / "results"
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"

    run_log: List[Dict[str, object]] = []
    for spec in specs:
        result = run_one(spec, args.timeout_s, env)
        run_log.append({"arm": spec.arm, "mb_units": spec.mb_units, "seed": spec.seed, **result})

    df = collect_csvs(specs, results_root)

    ts = time.strftime("%Y%m%d_%H%M%S")
    sweep_root = results_root / f"mothnet_mb_units_sweep_{ts}"
    sweep_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(sweep_root / "raw_results.csv", index=False)

    rng = np.random.default_rng(args.rng_seed_bootstrap)
    perm = run_permutation_section(df, rng)

    write_summary(
        sweep_root=sweep_root, df=df, run_log=run_log, seeds=seeds,
        mb_units_baseline=args.mb_units_baseline,
        mb_units_candidate=args.mb_units_candidate,
        perm=perm, rng_seed=args.rng_seed_bootstrap,
    )
    logger.info(f"DONE. Outputs in {sweep_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
