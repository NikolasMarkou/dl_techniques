"""Summary writer for the RMSNorm-variants sweep.

Reads the merged ``all_runs.csv`` produced by ``sweep.py`` and emits
``summary.md`` with:
- A headline table: ``experiment × norm × mode → metric mean ± std (CI) [p vs rms_norm]``.
- A probes table built from the per-cell probe CSVs (γ growth, output mean,
  per-sample RMS std, gradient norm) at the FINAL epoch only.
- High-variance flags where ``std > |mean|`` (LESSONS L76 — informational
  for "delta near zero" metrics like activation mean on zero-centered).
- An explicit PASS / FAIL / INDISTINGUISHABLE verdict per non-baseline
  variant, derived from a documented decision rule.

All statistical math is delegated to ``train.logic.multiseed_stats`` via the
local ``stats`` re-export.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dl_techniques.utils.logger import logger
from train.rms_variants_train.config import NORM_VARIANTS
from train.rms_variants_train.stats import (
    bootstrap_ci,
    format_mean_std,
    mean_std,
    paired_permutation_test,
)


# Headline metric per experiment.
HEADLINE_METRIC = {
    "e1": ("best_val_acc", "higher_is_better"),
    "e2": ("best_val_acc", "higher_is_better"),
    "e3": ("best_val_acc", "higher_is_better"),
    "e4": ("final_val_loss", "lower_is_better"),
    "e5": ("final_val_loss", "lower_is_better"),
}

BASELINE_NORM = "rms_norm"
RNG_SEED = 20260515


# ---------------------------------------------------------------------
# Headline aggregation
# ---------------------------------------------------------------------


def _aggregate_headline(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (experiment, norm_type, mode) and return mean ± std + CI."""
    rng = np.random.default_rng(RNG_SEED)
    rows: List[Dict] = []
    for exp, sub in df.groupby("experiment", sort=False):
        if exp not in HEADLINE_METRIC:
            continue
        metric, direction = HEADLINE_METRIC[exp]
        if metric not in sub.columns:
            continue
        for (norm, mode), grp in sub.groupby(["norm_type", "mode"], sort=False):
            values = grp[metric].astype(float).to_numpy()
            n = len(values)
            mean, std = mean_std(values)
            if n >= 2:
                lo, hi = bootstrap_ci(values, confidence=0.95, n_boot=2000, rng=rng)
            else:
                lo, hi = (mean, mean)
            rows.append({
                "experiment": exp,
                "norm_type": norm,
                "mode": mode,
                "metric": metric,
                "direction": direction,
                "n": n,
                "mean": mean,
                "std": std,
                "ci_low": lo,
                "ci_high": hi,
            })
    return pd.DataFrame(rows)


def _add_paired_p(headline: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add paired-permutation p-value vs ``rms_norm`` baseline per (exp, mode)."""
    rng = np.random.default_rng(RNG_SEED)
    p_values: List[Optional[float]] = []
    diff_values: List[Optional[float]] = []
    for _, row in headline.iterrows():
        exp = row["experiment"]
        norm = row["norm_type"]
        mode = row["mode"]
        metric = row["metric"]
        if norm == BASELINE_NORM:
            p_values.append(None)
            diff_values.append(0.0)
            continue
        a = (df[(df["experiment"] == exp)
               & (df["norm_type"] == norm)
               & (df["mode"] == mode)]
             .sort_values("seed")[metric].astype(float).to_numpy())
        b = (df[(df["experiment"] == exp)
               & (df["norm_type"] == BASELINE_NORM)
               & (df["mode"] == mode)]
             .sort_values("seed")[metric].astype(float).to_numpy())
        if len(a) < 2 or len(b) < 2 or len(a) != len(b):
            p_values.append(None)
            diff_values.append(None)
            continue
        diff, p = paired_permutation_test(a, b, n_perm=10000, rng=rng)
        p_values.append(float(p))
        diff_values.append(float(diff))
    headline["diff_vs_baseline"] = diff_values
    headline["p_vs_baseline"] = p_values
    return headline


def _verdict(row: pd.Series) -> str:
    """PASS / FAIL / INDISTINGUISHABLE rule per (norm, exp, mode)."""
    if row["norm_type"] == BASELINE_NORM:
        return "baseline"
    p = row["p_vs_baseline"]
    diff = row["diff_vs_baseline"]
    direction = row["direction"]
    if p is None or diff is None:
        return "n/a"
    if p >= 0.05:
        return "indistinguishable"
    better = diff > 0 if direction == "higher_is_better" else diff < 0
    return "pass" if better else "fail"


# ---------------------------------------------------------------------
# Probes aggregation (final-epoch snapshots)
# ---------------------------------------------------------------------


def _read_probe_csv(cell_dir: Path, name: str) -> Optional[pd.DataFrame]:
    p = cell_dir / name
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return None


def _final_probe_snapshot(out_dir: str, df: pd.DataFrame) -> pd.DataFrame:
    """Per (experiment, norm, mode, seed), pull the final-epoch row from each
    probe CSV (activation_stats / weight_norm / norm_internal / grad_norm)
    and average over layers.
    """
    rows: List[Dict] = []
    root = Path(out_dir)
    for _, r in df.iterrows():
        exp = r["experiment"]
        norm = r["norm_type"]
        mode = r["mode"]
        seed = int(r["seed"])
        cell_dir = root / exp / norm / mode / f"seed_{seed}"
        if not cell_dir.exists():
            continue

        act = _read_probe_csv(cell_dir, "activation_stats.csv")
        wnt = _read_probe_csv(cell_dir, "weight_norm.csv")
        inn = _read_probe_csv(cell_dir, "norm_internal.csv")
        grd = _read_probe_csv(cell_dir, "grad_norm.csv")

        def _last_epoch(d: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if d is None or d.empty or "epoch" not in d.columns:
                return None
            return d[d["epoch"] == d["epoch"].max()]

        row: Dict = {
            "experiment": exp, "norm_type": norm, "mode": mode, "seed": seed
        }
        last_act = _last_epoch(act)
        if last_act is not None and not last_act.empty:
            row["act_mean_abs"] = float(last_act["mean"].abs().mean())
            row["act_per_sample_rms_std_mean"] = float(last_act["per_sample_rms_std"].mean())
            row["act_per_sample_rms_mean_mean"] = float(last_act["per_sample_rms_mean"].mean())
            row["act_per_sample_rms_max_max"] = float(last_act["per_sample_rms_max"].max())
        last_wnt = _last_epoch(wnt)
        if last_wnt is not None and not last_wnt.empty:
            row["weight_l2_mean"] = float(last_wnt["l2"].mean())
            row["weight_l2_max"] = float(last_wnt["l2"].max())
        last_inn = _last_epoch(inn)
        if last_inn is not None and not last_inn.empty:
            row["internal_scale_or_raw_mean"] = float(
                last_inn["scale_l2_or_raw"].astype(float).mean()
            )
            ps = last_inn["post_sigmoid_scale"].astype(float)
            row["internal_post_sigmoid_mean"] = float(ps[~ps.isna()].mean()) if (~ps.isna()).any() else float("nan")
        last_grd = _last_epoch(grd)
        if last_grd is not None and not last_grd.empty:
            row["grad_norm_global"] = float(last_grd["grad_norm_global"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _probes_summary(probes_long: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the long-per-seed probe frame to mean ± std per cell."""
    if probes_long.empty:
        return pd.DataFrame()
    rows: List[Dict] = []
    for (exp, norm, mode), grp in probes_long.groupby(
        ["experiment", "norm_type", "mode"], sort=False
    ):
        row: Dict = {"experiment": exp, "norm_type": norm, "mode": mode, "n": len(grp)}
        for col in (
            "act_mean_abs",
            "act_per_sample_rms_std_mean",
            "act_per_sample_rms_mean_mean",
            "act_per_sample_rms_max_max",
            "weight_l2_mean",
            "weight_l2_max",
            "internal_scale_or_raw_mean",
            "internal_post_sigmoid_mean",
            "grad_norm_global",
        ):
            if col not in grp.columns:
                continue
            vals = grp[col].astype(float).to_numpy()
            m, s = mean_std(vals)
            row[f"{col}_mean"] = m
            row[f"{col}_std"] = s
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------


def _md_headline_table(headline: pd.DataFrame) -> str:
    if headline.empty:
        return "*(no headline rows — sweep produced no usable CSVs)*\n"
    out: List[str] = []
    out.append("| Experiment | Norm | Mode | n | Metric | Mean ± std | 95% CI | Δ vs baseline | p | Verdict |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, row in headline.iterrows():
        ci = f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        diff = row.get("diff_vs_baseline")
        p = row.get("p_vs_baseline")
        diff_s = "—" if diff is None or pd.isna(diff) else f"{diff:+.4f}"
        p_s = "—" if p is None or pd.isna(p) else (f"{p:.3f}" if p > 0.001 else "<0.001")
        out.append(
            f"| {row['experiment']} | {row['norm_type']} | {row['mode']} | "
            f"{int(row['n'])} | {row['metric']} | "
            f"{format_mean_std(row['mean'], row['std'])} | {ci} | "
            f"{diff_s} | {p_s} | {row.get('verdict', '?')} |"
        )
    return "\n".join(out) + "\n"


def _md_probes_table(probes: pd.DataFrame) -> str:
    if probes.empty:
        return "*(no probe rows)*\n"
    cols_of_interest = [
        ("act_mean_abs_mean", "|act.mean|"),
        ("act_per_sample_rms_std_mean_mean", "rms_std"),
        ("act_per_sample_rms_max_max_mean", "rms_max"),
        ("weight_l2_mean_mean", "weight_l2"),
        ("internal_scale_or_raw_mean_mean", "scale/raw"),
        ("internal_post_sigmoid_mean_mean", "post_sig"),
        ("grad_norm_global_mean", "grad_norm"),
    ]
    out: List[str] = []
    hdr = ["Experiment", "Norm", "Mode", "n"] + [d for _, d in cols_of_interest]
    out.append("| " + " | ".join(hdr) + " |")
    out.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for _, row in probes.iterrows():
        line = [row["experiment"], row["norm_type"], row["mode"], str(int(row["n"]))]
        for col, _ in cols_of_interest:
            v = row.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                line.append("—")
            else:
                line.append(f"{v:.4f}")
        out.append("| " + " | ".join(line) + " |")
    return "\n".join(out) + "\n"


def _high_variance_flags(headline: pd.DataFrame) -> str:
    if headline.empty:
        return "*(none)*\n"
    flagged = headline[
        (headline["std"].abs() > headline["mean"].abs())
        & (headline["std"] > 0)
        & (headline["mean"].abs() > 1e-12)
    ]
    if flagged.empty:
        return "*(none — all headline metrics have std ≤ |mean|)*\n"
    rows = ["| Experiment | Norm | Mode | Mean | Std |", "|---|---|---|---|---|"]
    for _, r in flagged.iterrows():
        rows.append(
            f"| {r['experiment']} | {r['norm_type']} | {r['mode']} | "
            f"{r['mean']:.4f} | {r['std']:.4f} |"
        )
    return "\n".join(rows) + "\n"


def _verdict_block(headline: pd.DataFrame) -> str:
    if headline.empty:
        return "*(no verdicts — sweep empty)*\n"
    out: List[str] = []
    for variant in NORM_VARIANTS:
        if variant == BASELINE_NORM:
            continue
        rows = headline[headline["norm_type"] == variant]
        if rows.empty:
            continue
        passes = (rows["verdict"] == "pass").sum()
        fails = (rows["verdict"] == "fail").sum()
        indis = (rows["verdict"] == "indistinguishable").sum()
        out.append(f"### `{variant}`")
        out.append(
            f"- PASS in {int(passes)} cells, FAIL in {int(fails)}, "
            f"INDISTINGUISHABLE in {int(indis)} (n={len(rows)} cells total)."
        )
        if fails:
            details = rows[rows["verdict"] == "fail"][
                ["experiment", "mode", "diff_vs_baseline", "p_vs_baseline"]
            ]
            out.append("- FAIL cells:")
            for _, r in details.iterrows():
                out.append(
                    f"  - {r['experiment']}/{r['mode']}: "
                    f"diff={r['diff_vs_baseline']:+.4f}, p={r['p_vs_baseline']:.3f}"
                )
        if passes >= 2 and fails == 0:
            out.append(f"- **Overall**: PASS — beats baseline in ≥2 cells with no failures.")
        elif fails > 0:
            out.append(f"- **Overall**: FAIL — direction opposite to claim in {int(fails)} cell(s).")
        else:
            out.append(f"- **Overall**: INDISTINGUISHABLE — within seed noise.")
        out.append("")
    return "\n".join(out) if out else "*(no non-baseline variants in sweep)*\n"


def write_report(df: pd.DataFrame, *, out_dir: str) -> None:
    """Write ``summary.md`` aggregating ``df`` (the merged ``all_runs.csv``)."""
    headline = _aggregate_headline(df)
    headline = _add_paired_p(headline, df)
    headline["verdict"] = headline.apply(_verdict, axis=1)

    probes_long = _final_probe_snapshot(out_dir, df)
    probes = _probes_summary(probes_long)

    md_lines: List[str] = []
    md_lines.append("# RMSNorm Variants — Sweep Summary\n")
    md_lines.append(f"*Output dir: `{out_dir}`*\n")
    md_lines.append(f"*Cells in merged frame: {len(df)}*\n")
    md_lines.append("## Headline metric per (experiment, norm, mode)\n")
    md_lines.append(_md_headline_table(headline))
    md_lines.append("## Mechanistic probes (final epoch, averaged over layers + seeds)\n")
    md_lines.append(_md_probes_table(probes))
    md_lines.append("## High-variance flags (std > |mean|)\n")
    md_lines.append(_high_variance_flags(headline))
    md_lines.append("## Verdict per variant\n")
    md_lines.append(_verdict_block(headline))
    md_lines.append(
        "\n*Verdict rule: at p < 0.05 (paired permutation B=10000) AND "
        "direction matches claim → PASS; opposite direction → FAIL; else → "
        "INDISTINGUISHABLE. Per-variant aggregate PASS requires ≥2 cells with "
        "no FAIL.*\n"
    )

    summary_path = os.path.join(out_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(md_lines))

    # Also persist the headline frame as CSV for downstream tooling.
    headline.to_csv(os.path.join(out_dir, "headline_summary.csv"), index=False)
    probes.to_csv(os.path.join(out_dir, "probes_summary.csv"), index=False)
    logger.info(f"[report] wrote {summary_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RMSNorm variants summary writer")
    p.add_argument("--in-dir", type=str, required=True,
                   help="Sweep root containing all_runs.csv + per-cell CSVs.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    merged = os.path.join(args.in_dir, "all_runs.csv")
    if not os.path.exists(merged):
        logger.error(f"[report] {merged} missing — run sweep.py first.")
        return 1
    df = pd.read_csv(merged)
    write_report(df, out_dir=args.in_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
