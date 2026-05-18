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
    # E6 added at step 9 of plan_2026-05-18_6776f8ba. 14400s (4h) per-cell
    # timeout reflects the full Phase 3 v3 budget for the 4-epoch Wikipedia
    # 10k training; smoke runs MUST override via --cell-timeout-s.
    "e6": ("train.rms_variants_train.experiments.e6_clm_wiki", 14400),
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
    # E6 plumbs norm kwargs through TransformerLayer (Pattern-3); supports
    # both modes (OOB + param_matched). E6 has no scale-toggle confound for
    # RMSNorm variants but param_matched is still meaningful for the
    # use_scale=False ablation on rms_norm / zero_centered_rms_norm.
    "e6": ("oob", "param_matched"),
}

# Per-experiment regime support. Mirrors each trainer's `_REGIME_MAP` keys.
# A regime not in an experiment's tuple is skipped (with a one-line log)
# during build_run_specs — this keeps `--regimes` builds tight by dropping
# combinations the trainer would reject anyway. Update this dict whenever a
# trainer's `_REGIME_MAP` is extended.
EXPERIMENT_REGIMES: dict = {
    "e1": ("default", "lr_low", "lr_high", "mp_fp16"),
    "e2": ("default",),
    "e3": ("default", "mp_fp16"),
    "e4": ("default", "depth_12", "depth_48"),
    "e5": ("default", "bs_32", "bs_256", "lr_low", "lr_high"),
    # E6 supports default/mp_fp16/lr_extreme/wd_zero — mirrors E3's set,
    # minus bs_4 (tiny CLM batches <8 destabilise gradient estimates).
    "e6": ("default", "mp_fp16", "lr_extreme", "wd_zero"),
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
    """Specification for one (experiment, norm, mode, regime, seed) cell.

    The ``regime`` field defaults to ``"default"`` for backward compatibility
    with pre-plan_e1f12eab sweep invocations that did not pass ``--regimes``.
    """

    experiment: str
    norm_type: str
    mode: str
    seed: int
    module: str
    extra_args: Tuple[str, ...]
    out_dir: str
    regime: str = "default"
    csv_filename: str = "results.csv"


def build_run_specs(
    *,
    experiments: Sequence[str],
    norms: Sequence[str],
    modes: Sequence[str],
    seeds: Sequence[int],
    sweep_root: str,
    epochs_override: Optional[int],
    regimes: Sequence[str] = ("default",),
    max_cells: int = 1000,
) -> List[RunSpec]:
    """Enumerate every subprocess we plan to launch.

    Filters out (experiment, mode) pairs not supported per
    :data:`EXPERIMENT_MODES`, (experiment, regime) pairs not supported per
    :data:`EXPERIMENT_REGIMES`, and (PM, no-use_scale) combinations. Logs a
    one-line skip notice per filtered cell.

    Per plan_e1f12eab D-003 (anchored): raises ``ValueError`` at build time
    if the constructed cell count exceeds ``max_cells``. The error message
    includes the dimensions to help the user trim the build.

    :param regimes: Iterable of regime names. Cells are emitted for the
        Cartesian product of regimes × (other dims). Default ``("default",)``
        preserves pre-plan_e1f12eab behaviour.
    :param max_cells: Cell-count safety guard. Raises ``ValueError`` before
        any subprocess is launched if the build exceeds this. Default 1000.
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
        supported_regimes = EXPERIMENT_REGIMES.get(exp, ("default",))
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
                for regime in regimes:
                    if regime not in supported_regimes:
                        logger.info(
                            f"[sweep] skip {exp}/{norm}/{mode}/{regime}/* "
                            f"(experiment supports only "
                            f"regimes={supported_regimes})"
                        )
                        continue
                    for seed in seeds:
                        # Out-dir naming: include regime in the leaf path
                        # ONLY when a non-default regime is in play, to
                        # preserve the pre-plan_e1f12eab default layout
                        # (and keep RESULTS.md Phase 1 verdict-block paths
                        # stable — I2 invariant).
                        if regime == "default":
                            out_dir = os.path.join(
                                sweep_root, exp, norm, mode, f"seed_{seed}",
                            )
                        else:
                            out_dir = os.path.join(
                                sweep_root, exp, norm, mode,
                                f"regime_{regime}", f"seed_{seed}",
                            )
                        extra: List[str] = [
                            "--norm-type", norm,
                            "--seed", str(seed),
                            "--mode", mode,
                            "--regime", regime,
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
                            regime=regime,
                        ))

    # DECISION plan_2026-05-18_e1f12eab/D-003: --max-cells guard.
    # Reject oversized builds BEFORE any subprocess is launched, to prevent
    # the partial-sweep / inconsistent-results-dir failure mode (EC3 /
    # falsification scenario C). Raised at build time, not at run time.
    if len(specs) > max_cells:
        raise ValueError(
            f"Cell count {len(specs)} exceeds --max-cells={max_cells}. "
            f"Dimensions: {len(experiments)} experiments × {len(norms)} norms "
            f"× {len(modes)} modes × {len(regimes)} regimes × {len(seeds)} "
            f"seeds. Trim a dimension (e.g. fewer norms or seeds), bump "
            f"--max-cells, or chunk the sweep into multiple invocations."
        )
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
    gpu_id: int,
) -> Tuple[bool, str]:
    """Launch one subprocess. Returns (success, stderr_tail).

    The ``gpu_id`` argument is the single source of truth for the subprocess
    ``CUDA_VISIBLE_DEVICES`` value. It is **hard-set** (not ``setdefault``)
    so the parent shell's CVD value cannot leak through and silently re-bind
    the cell to the wrong GPU (the bug that killed plan_74a935a2's sweep,
    LESSONS L93).
    """
    now = time.time()
    remaining = deadline_s - now
    if remaining <= 0:
        return False, "global timeout reached before launch"
    timeout = min(cell_timeout_s, remaining)

    os.makedirs(spec.out_dir, exist_ok=True)
    log_path = os.path.join(spec.out_dir, "cell.log")
    cmd = [python_exe, "-m", spec.module, *spec.extra_args]
    env = os.environ.copy()
    # DECISION plan_2026-05-18_63121227/D-002: hard-set both env vars below
    # (was ``setdefault``, which silently honoured the parent shell). The
    # ``--gpu`` CLI flag is the canonical source of truth; parent CVD is
    # explicitly overridden. See LESSONS L93 for the failure mode.
    env["MPLBACKEND"] = "Agg"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    t0 = time.time()
    with open(log_path, "w") as log_f:
        log_f.write(
            f"# cmd: {' '.join(cmd)}\n"
            f"# CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n"
            f"# MPLBACKEND={env['MPLBACKEND']}\n"
            f"# timeout: {timeout}s\n\n"
        )
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
                   help=f"Default: all 8 — {','.join(NORM_VARIANTS)}")
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
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index. Hard-sets CUDA_VISIBLE_DEVICES for each "
                        "cell subprocess, overriding any value inherited from "
                        "the parent shell (LESSONS L93 / D-002). Default 0.")
    p.add_argument("--no-report", action="store_true",
                   help="Skip the report.py invocation at the end.")
    # Plan plan_e1f12eab Step 5 / D-003 additions.
    p.add_argument("--regimes", type=_csv_list, default=["default"],
                   help="Comma-separated regime names (per-trainer _REGIME_MAP "
                        "keys). Default: 'default'. Unsupported (exp, regime) "
                        "pairs are skipped with a log line. Cells multiply by "
                        "this dimension — combine with --max-cells.")
    p.add_argument("--max-cells", type=int, default=1000,
                   help="Hard cap on planned cell count. Raises an error "
                        "BEFORE any subprocess launches if exceeded (D-003 "
                        "guard against silent multi-hour partial sweeps). "
                        "Default 1000.")
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
        regimes=args.regimes,
        max_cells=args.max_cells,
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
            gpu_id=args.gpu,
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
