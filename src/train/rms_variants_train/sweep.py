"""Multi-experiment × multi-norm × multi-mode × multi-seed sweep driver.

Drives the FROZEN experiment trainers (e1..e5) as subprocesses across the
full Cartesian product, aggregates their per-cell CSVs into a single
multi-experiment frame, and then optionally invokes ``report.py`` for the
mean ± std / bootstrap CI / paired-permutation summary.

Subprocess-per-cell is the canonical pattern (LESSONS L74). Each cell gets
a clean TF/Keras init, so cross-cell state contamination is impossible.

Per-cell timeout + global wall-clock cap. Failed cells are logged but do not
poison the rest of the sweep; aggregation tolerates n < requested.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \\
        train.rms_variants_train.sweep \\
            --experiments e5 \\
            --norms rms_norm,band_rms,zero_centered_rms_norm,zero_centered_band_rms_norm \\
            --seeds 0,1 \\
            --mode oob \\
            --out-dir /tmp/sweep_smoke \\
            --global-cap-s 900

DECISION plan_2026-05-14_3764496e/D-001: structural copy of
``train.logic.multiseed_sweep`` — same RunSpec / build_specs / run_one /
collect pattern, adapted from (experiment, seed) to
(experiment, norm, mode, seed).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from dl_techniques.utils.logger import logger
from train.rms_variants_train.config import NORM_VARIANTS


# ---------------------------------------------------------------------
# RunSpec
# ---------------------------------------------------------------------

# Maps experiment ID to (module path, default per-cell timeout in seconds).
# Per-cell timeouts are generous for the full multi-epoch runs; smoke runs
# should override via --cell-timeout-s.
EXPERIMENT_REGISTRY = {
    "e1": ("train.rms_variants_train.experiments.e1_vit_cifar10", 3600),
    "e2": ("train.rms_variants_train.experiments.e2_resnet_cifar100", 5400),
    "e3": ("train.rms_variants_train.experiments.e3_tinytransformer_imdb", 2400),
    "e4": ("train.rms_variants_train.experiments.e4_deep_residual_reg", 2400),
    "e5": ("train.rms_variants_train.experiments.e5_norm_layer_microbench", 600),
}

# Modes each experiment supports. ViT/ResNet (E1/E2) do not plumb norm
# kwargs through their factories, so PARAM_MATCHED is skipped there
# (D-003). E3/E4/E5 build norms locally and support both modes.
EXPERIMENT_MODES = {
    "e1": ("oob",),
    "e2": ("oob",),
    "e3": ("oob", "param_matched"),
    "e4": ("oob", "param_matched"),
    "e5": ("oob", "param_matched"),
}

# Variants that meaningfully respond to PARAM_MATCHED mode (drop their
# per-feature scale). RMSNorm-family variants support a real `use_scale=False`
# toggle. BandRMS-family variants treat PARAM_MATCHED as a no-op (they always
# carry exactly 1 scalar), kept in this set for Phase 1 back-compat — the
# resulting cell duplicates the OOB cell but with mode="param_matched" label.
#
# The 3 Phase 2 variants (adaptive_band_rms, band_logit_norm, dynamic_tanh)
# do NOT expose `use_scale`. Listing them as PARAM_MATCHED-supported would
# produce duplicate cells with no scientific meaning. Excluded from this set.
VARIANT_SUPPORTS_PARAM_MATCHED: frozenset = frozenset({
    "rms_norm",
    "zero_centered_rms_norm",
    "band_rms",
    "zero_centered_band_rms_norm",
})


@dataclass(frozen=True)
class RunSpec:
    """Specification for one (experiment, norm, mode, seed) cell."""

    experiment: str
    norm_type: str
    mode: str
    seed: int
    module: str
    extra_args: Tuple[str, ...]
    out_dir: str
    csv_filename: str = "results.csv"


def build_run_specs(
    *,
    experiments: Sequence[str],
    norms: Sequence[str],
    modes: Sequence[str],
    seeds: Sequence[int],
    sweep_root: str,
    epochs_override: Optional[int],
) -> List[RunSpec]:
    """Enumerate every subprocess we plan to launch.

    Filters out (experiment, mode) pairs not supported per
    :data:`EXPERIMENT_MODES` and logs a one-line skip notice per filtered cell.
    """
    specs: List[RunSpec] = []
    for exp in experiments:
        if exp not in EXPERIMENT_REGISTRY:
            raise ValueError(
                f"Unknown experiment: {exp}. "
                f"Valid: {list(EXPERIMENT_REGISTRY.keys())}"
            )
        module, _ = EXPERIMENT_REGISTRY[exp]
        supported_modes = EXPERIMENT_MODES[exp]
        for norm in norms:
            for mode in modes:
                if mode not in supported_modes:
                    logger.info(
                        f"[sweep] skip {exp}/{norm}/{mode}/* "
                        f"(experiment supports only {supported_modes})"
                    )
                    continue
                if (
                    mode == "param_matched"
                    and norm not in VARIANT_SUPPORTS_PARAM_MATCHED
                ):
                    # The variant has no `use_scale` toggle (Phase 2 norms).
                    # Skip with a one-line log per (exp, norm) pair.
                    logger.info(
                        f"[sweep] skip {exp}/{norm}/param_matched/* "
                        f"(variant has no use_scale toggle; PM cells are "
                        f"not meaningful for this norm)"
                    )
                    continue
                for seed in seeds:
                    out_dir = os.path.join(
                        sweep_root, exp, norm, mode, f"seed_{seed}"
                    )
                    extra: List[str] = [
                        "--norm-type", norm,
                        "--seed", str(seed),
                        "--mode", mode,
                        "--out-dir", out_dir,
                    ]
                    if epochs_override is not None:
                        extra += ["--epochs", str(epochs_override)]
                    specs.append(RunSpec(
                        experiment=exp,
                        norm_type=norm,
                        mode=mode,
                        seed=seed,
                        module=module,
                        extra_args=tuple(extra),
                        out_dir=out_dir,
                    ))
    return specs


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------


def run_one(
    spec: RunSpec,
    *,
    python_exe: str,
    cell_timeout_s: int,
    deadline_s: float,
) -> Tuple[bool, str]:
    """Launch one subprocess. Returns (success, stderr_tail)."""
    now = time.time()
    remaining = deadline_s - now
    if remaining <= 0:
        return False, "global timeout reached before launch"
    timeout = min(cell_timeout_s, remaining)

    os.makedirs(spec.out_dir, exist_ok=True)
    log_path = os.path.join(spec.out_dir, "cell.log")
    cmd = [python_exe, "-m", spec.module, *spec.extra_args]
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    t0 = time.time()
    with open(log_path, "w") as log_f:
        log_f.write(f"# cmd: {' '.join(cmd)}\n# timeout: {timeout}s\n\n")
        log_f.flush()
        try:
            res = subprocess.run(
                cmd, env=env, timeout=timeout,
                stdout=log_f, stderr=subprocess.STDOUT,
            )
            wall_s = time.time() - t0
            if res.returncode != 0:
                with open(log_path) as f:
                    tail = f.read()[-1500:]
                logger.error(
                    f"[sweep] FAIL {spec.experiment}/{spec.norm_type}/{spec.mode}"
                    f"/seed_{spec.seed} rc={res.returncode} in {wall_s:.1f}s"
                )
                return False, tail
            logger.info(
                f"[sweep] OK   {spec.experiment}/{spec.norm_type}/{spec.mode}"
                f"/seed_{spec.seed} in {wall_s:.1f}s"
            )
            return True, ""
        except subprocess.TimeoutExpired:
            wall_s = time.time() - t0
            logger.error(
                f"[sweep] TIMEOUT {spec.experiment}/{spec.norm_type}/{spec.mode}"
                f"/seed_{spec.seed} after {wall_s:.1f}s (cap={timeout}s)"
            )
            return False, f"cell timeout after {timeout}s"


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------


def collect_csvs(
    sweep_root: str, experiments: Sequence[str],
) -> pd.DataFrame:
    """Find every per-cell ``results.csv`` and concatenate with metadata cols."""
    frames: List[pd.DataFrame] = []
    root = Path(sweep_root)
    for csv_path in root.rglob("results.csv"):
        try:
            df = pd.read_csv(csv_path)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            continue
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df_all = pd.concat(frames, ignore_index=True)
    # Restrict to requested experiments
    if "experiment" in df_all.columns:
        df_all = df_all[df_all["experiment"].isin(experiments)].reset_index(drop=True)
    return df_all


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _int_csv(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RMSNorm variants sweep driver")
    p.add_argument("--experiments", type=_csv_list, required=True,
                   help="Comma-separated subset of e1,e2,e3,e4,e5")
    p.add_argument("--norms", type=_csv_list, default=list(NORM_VARIANTS),
                   help=f"Default: all 7 — {','.join(NORM_VARIANTS)}")
    p.add_argument("--modes", type=_csv_list, default=["oob"],
                   help="Subset of {oob, param_matched}")
    p.add_argument("--seeds", type=_int_csv, default=[0, 1, 2, 3, 4])
    # Single-mode legacy alias (sweep --mode oob).
    p.add_argument("--mode", type=str, default=None,
                   help="Convenience: sets --modes to this single mode.")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=None,
                   help="Override per-trainer default epochs (smoke runs).")
    p.add_argument("--cell-timeout-s", type=int, default=None,
                   help="Default: per-experiment registry value.")
    p.add_argument("--global-cap-s", type=int, default=12 * 3600,
                   help="Total wall-clock cap (default 12h)")
    p.add_argument("--python-exe", type=str, default=sys.executable)
    p.add_argument("--no-report", action="store_true",
                   help="Skip the report.py invocation at the end.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.mode is not None:
        args.modes = [args.mode]

    os.makedirs(args.out_dir, exist_ok=True)
    sweep_log_path = os.path.join(args.out_dir, "sweep.log")
    logger.info(f"[sweep] out_dir={args.out_dir}, log={sweep_log_path}")

    specs = build_run_specs(
        experiments=args.experiments,
        norms=args.norms,
        modes=args.modes,
        seeds=args.seeds,
        sweep_root=args.out_dir,
        epochs_override=args.epochs,
    )
    logger.info(f"[sweep] {len(specs)} cells planned")
    if not specs:
        logger.error("[sweep] no cells planned; check --experiments/--modes filter.")
        return 2

    deadline = time.time() + args.global_cap_s
    successes = 0
    failures: List[Tuple[RunSpec, str]] = []
    for spec in specs:
        cell_timeout = args.cell_timeout_s or EXPERIMENT_REGISTRY[spec.experiment][1]
        ok, tail = run_one(
            spec,
            python_exe=args.python_exe,
            cell_timeout_s=cell_timeout,
            deadline_s=deadline,
        )
        if ok:
            successes += 1
        else:
            failures.append((spec, tail))
        if time.time() >= deadline:
            logger.error("[sweep] global timeout — aborting remaining cells")
            break

    df = collect_csvs(args.out_dir, args.experiments)
    merged_csv = os.path.join(args.out_dir, "all_runs.csv")
    df.to_csv(merged_csv, index=False)
    logger.info(
        f"[sweep] complete: {successes}/{len(specs)} cells succeeded; "
        f"merged → {merged_csv} ({len(df)} rows)"
    )

    if failures:
        with open(os.path.join(args.out_dir, "failures.log"), "w") as f:
            for spec, tail in failures:
                f.write(
                    f"=== {spec.experiment}/{spec.norm_type}/{spec.mode}/seed_{spec.seed} ===\n"
                )
                f.write(tail + "\n\n")

    if not args.no_report and not df.empty:
        try:
            from train.rms_variants_train.report import write_report
            write_report(df, out_dir=args.out_dir)
        except Exception as e:  # pragma: no cover
            logger.error(f"[sweep] report.py failed: {e}")
            return 1
    return 0 if successes == len(specs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
