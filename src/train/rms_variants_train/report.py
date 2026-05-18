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
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd

from dl_techniques.utils.logger import logger
from train.rms_variants_train.config import NORM_VARIANTS
from train.rms_variants_train.hypotheses import (
    VARIANT_HYPOTHESES,
    evaluate_all as evaluate_all_hypotheses,
)
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


# DECISION plan_2026-05-18_74a935a2/D-002: per-variant verdict-rule
# parameterization. DyT does not normalize by RMS → act_mean_abs /
# per_sample_rms_* are mechanism-irrelevant. band_logit_norm normalizes by L2
# (not RMS) → per_sample_rms_* are irrelevant; additionally it is designed
# for classification logits, so residual-stream contexts (E2/E3/E4) are
# off-label and must NOT be compared head-to-head against the rms_norm
# baseline's headline metric.
#
# `applicable_probes` is an *informational* contract — it says which probe
# columns are meaningful for this variant's mechanism. The headline
# PASS/FAIL/INDISTINGUISHABLE rule on the `final_val_loss` / `best_val_acc`
# metric is unchanged for non-off-label cells (a fair head-to-head test).
#
# `off_label_contexts` is a *gating* contract — for any (variant,
# experiment) pair listed, the verdict is forced to "n/a (off-label)"
# regardless of the headline metric.
VARIANT_CRITERIA: Dict[str, Dict[str, Any]] = {
    "rms_norm": {
        "applicable_probes": ("grad_norm",),
        "off_label_contexts": frozenset(),
    },
    "band_rms": {
        "applicable_probes": ("grad_norm", "act_per_sample_rms_max"),
        "off_label_contexts": frozenset(),
    },
    "zero_centered_rms_norm": {
        "applicable_probes": ("grad_norm", "act_mean_abs"),
        "off_label_contexts": frozenset(),
    },
    "zero_centered_band_rms_norm": {
        "applicable_probes": (
            "grad_norm",
            "act_mean_abs",
            "act_per_sample_rms_max",
        ),
        "off_label_contexts": frozenset(),
    },
    "adaptive_band_rms": {
        "applicable_probes": ("grad_norm", "act_per_sample_rms_max"),
        "off_label_contexts": frozenset(),
    },
    "band_logit_norm": {
        "applicable_probes": ("grad_norm",),
        # Off-label on every residual-stream context. E1 (ViT) and E5
        # (microbench) are still scored — E1 includes a classification head
        # where band_logit can be argued to apply if used appropriately, and
        # E5 is the pure-mechanism microbenchmark.
        "off_label_contexts": frozenset({"e2", "e3", "e4"}),
    },
    "dynamic_tanh": {
        "applicable_probes": ("grad_norm",),
        "off_label_contexts": frozenset(),
    },
    "zero_centered_adaptive_band_rms_norm": {
        # Combines the zero-mean-output claim (zero_centered_*) with the
        # per-sample adaptive band-scaling claim (adaptive_band_rms). All
        # three relevant probes apply.
        "applicable_probes": (
            "grad_norm",
            "act_mean_abs",
            "act_per_sample_rms_max",
        ),
        "off_label_contexts": frozenset(),
    },
}


def _off_label_contexts(norm_type: str) -> FrozenSet[str]:
    """Return the set of experiment ids on which ``norm_type`` is off-label."""
    entry = VARIANT_CRITERIA.get(norm_type)
    if entry is None:
        return frozenset()
    return entry.get("off_label_contexts", frozenset())


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
    """PASS / FAIL / INDISTINGUISHABLE rule per (norm, exp, mode).

    Off-label cells (variant × experiment listed in
    ``VARIANT_CRITERIA[norm_type]['off_label_contexts']``) short-circuit to
    ``n/a (off-label)`` regardless of the headline metric. This prevents
    misleading FAIL/PASS verdicts when the variant's design context does not
    apply to the experiment (e.g. ``band_logit_norm`` on a residual stream).
    """
    if row["norm_type"] == BASELINE_NORM:
        return "baseline"
    # D-002 off-label gate.
    if row["experiment"] in _off_label_contexts(row["norm_type"]):
        return "n/a (off-label)"
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
    # Footnote for variants with non-empty off_label_contexts (D-002).
    off_label_summary: List[str] = []
    for v, criteria in VARIANT_CRITERIA.items():
        ctx = criteria.get("off_label_contexts", frozenset())
        if ctx:
            off_label_summary.append(
                f"`{v}` is treated as **off-label** on "
                f"{{ {', '.join(sorted(ctx))} }}: its design context does "
                f"not include those experiments, so the headline metric is "
                f"not a fair head-to-head test there."
            )
    if off_label_summary:
        out.append("> **Off-label notes** (D-002):")
        for line in off_label_summary:
            out.append(f"> - {line}")
        out.append("")
    for variant in NORM_VARIANTS:
        if variant == BASELINE_NORM:
            continue
        rows = headline[headline["norm_type"] == variant]
        if rows.empty:
            continue
        passes = (rows["verdict"] == "pass").sum()
        fails = (rows["verdict"] == "fail").sum()
        indis = (rows["verdict"] == "indistinguishable").sum()
        offlabel = (rows["verdict"] == "n/a (off-label)").sum()
        # Cells that are off-label do NOT count toward the rollup denominator.
        eligible = len(rows) - int(offlabel)
        out.append(f"### `{variant}`")
        if offlabel:
            out.append(
                f"- PASS in {int(passes)} cells, FAIL in {int(fails)}, "
                f"INDISTINGUISHABLE in {int(indis)}, OFF-LABEL in "
                f"{int(offlabel)} (n={len(rows)} cells; {eligible} eligible "
                f"for verdict)."
            )
        else:
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
        if eligible == 0:
            out.append(
                "- **Overall**: N/A — all cells off-label for this variant."
            )
        elif passes >= 2 and fails == 0:
            out.append(f"- **Overall**: PASS — beats baseline in ≥2 cells with no failures.")
        elif fails > 0:
            out.append(f"- **Overall**: FAIL — direction opposite to claim in {int(fails)} cell(s).")
        else:
            out.append(f"- **Overall**: INDISTINGUISHABLE — within seed noise.")
        out.append("")
    return "\n".join(out) if out else "*(no non-baseline variants in sweep)*\n"


def _load_per_epoch_frame(out_dir: str, df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort: walk per-cell directories, read ``history.csv``, stack.

    Returns a long-format DataFrame with columns
    ``[experiment, norm_type, mode, seed, regime, epoch, <metric columns>]``.
    Returns an empty frame if no cell directory has a ``history.csv``.
    """
    rows: List[pd.DataFrame] = []
    root = Path(out_dir)
    # Use df to enumerate (experiment, norm, mode, seed[, regime]) combinations.
    if df.empty:
        return pd.DataFrame()
    has_regime = "regime" in df.columns
    keys = ["experiment", "norm_type", "mode", "seed"]
    for _, r in df.drop_duplicates(subset=keys + (["regime"] if has_regime else [])).iterrows():
        # Cell-dir convention: out_dir/<exp>/<norm>/<mode>/seed_<seed>
        cell_dir = root / r["experiment"] / r["norm_type"] / r["mode"] / f"seed_{int(r['seed'])}"
        hist_path = cell_dir / "history.csv"
        if not hist_path.exists():
            continue
        try:
            h = pd.read_csv(hist_path)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            continue
        if h.empty:
            continue
        h["experiment"] = r["experiment"]
        h["norm_type"] = r["norm_type"]
        h["mode"] = r["mode"]
        h["seed"] = int(r["seed"])
        h["regime"] = r["regime"] if has_regime else "default"
        rows.append(h)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _compute_convergence_speed(
    df_per_epoch: pd.DataFrame,
    *,
    thresholds: Tuple[float, ...] = (0.5, 0.7, 0.9),
) -> pd.DataFrame:
    """Epochs-to-threshold per (experiment, norm_type, mode, seed, regime).

    Uses ``val_accuracy`` when present (classification). Falls back to
    normalized-improvement ``1 - val_loss / val_loss_initial`` (regression).
    Emits one row per (cell, threshold). Threshold not reached → epoch = -1.

    Schema: ``experiment, norm_type, mode, seed, regime, threshold, epoch``.
    """
    if df_per_epoch.empty:
        return pd.DataFrame(columns=[
            "experiment", "norm_type", "mode", "seed", "regime",
            "threshold", "epoch",
        ])
    has_val_acc = "val_accuracy" in df_per_epoch.columns
    has_val_loss = "val_loss" in df_per_epoch.columns
    keys = ["experiment", "norm_type", "mode", "seed", "regime"]
    out: List[dict] = []
    for key_tuple, sub in df_per_epoch.groupby(keys, sort=False):
        sub = sub.sort_values("epoch").reset_index(drop=True)
        if has_val_acc and sub["val_accuracy"].notna().any():
            series = sub["val_accuracy"].to_numpy(dtype=float)
        elif has_val_loss and sub["val_loss"].notna().any():
            v0 = float(sub["val_loss"].iloc[0])
            if v0 <= 0:
                continue
            series = 1.0 - sub["val_loss"].to_numpy(dtype=float) / v0
        else:
            continue
        for thr in thresholds:
            ok = np.where(series >= thr)[0]
            epoch_hit = int(ok[0]) if ok.size > 0 else -1
            row = dict(zip(keys, key_tuple))
            row["threshold"] = float(thr)
            row["epoch"] = epoch_hit
            out.append(row)
    return pd.DataFrame(out)


def _compute_late_stability(
    df_per_epoch: pd.DataFrame,
    *,
    last_frac: float = 0.25,
) -> pd.DataFrame:
    """Variance of ``val_loss`` over the last ``last_frac`` of epochs per cell.

    Schema: ``experiment, norm_type, mode, seed, regime,
    late_stability_var, n_epochs_used``.
    """
    cols = [
        "experiment", "norm_type", "mode", "seed", "regime",
        "late_stability_var", "n_epochs_used",
    ]
    if df_per_epoch.empty or "val_loss" not in df_per_epoch.columns:
        return pd.DataFrame(columns=cols)
    keys = ["experiment", "norm_type", "mode", "seed", "regime"]
    out: List[dict] = []
    for key_tuple, sub in df_per_epoch.groupby(keys, sort=False):
        sub = sub.sort_values("epoch")
        n = len(sub)
        if n < 2:
            continue
        k = max(2, int(round(n * last_frac)))
        tail = sub["val_loss"].to_numpy(dtype=float)[-k:]
        tail = tail[np.isfinite(tail)]
        if tail.size < 2:
            continue
        row = dict(zip(keys, key_tuple))
        row["late_stability_var"] = float(tail.var(ddof=1))
        row["n_epochs_used"] = int(tail.size)
        out.append(row)
    return pd.DataFrame(out, columns=cols)


def _compute_regime_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Δ(metric) vs ``regime='default'`` for each (experiment, norm_type, metric).

    Operates on the merged ``all_runs.csv`` (final-epoch headline rows).
    Aggregates over seeds within each (experiment, norm_type, mode, regime)
    cell using the mean. Emits one row per non-default regime, only when the
    ``default`` regime is present for that (experiment, norm_type, mode).

    Schema: ``experiment, norm_type, mode, regime, metric,
    default_mean, regime_mean, delta``.
    """
    cols = [
        "experiment", "norm_type", "mode", "regime", "metric",
        "default_mean", "regime_mean", "delta",
    ]
    if df.empty or "regime" not in df.columns:
        return pd.DataFrame(columns=cols)
    metric_candidates = [
        "best_val_acc", "final_val_acc", "final_val_loss", "final_val_mae",
    ]
    metrics = [m for m in metric_candidates if m in df.columns]
    if not metrics:
        return pd.DataFrame(columns=cols)
    out: List[dict] = []
    group_keys = ["experiment", "norm_type", "mode", "regime"]
    agg = df.groupby(group_keys, sort=False)[metrics].mean().reset_index()
    for (exp, norm, mode), sub in agg.groupby(
        ["experiment", "norm_type", "mode"], sort=False
    ):
        default_row = sub[sub["regime"] == "default"]
        if default_row.empty:
            continue
        d = default_row.iloc[0]
        for _, r in sub.iterrows():
            if r["regime"] == "default":
                continue
            for m in metrics:
                if pd.isna(r[m]) or pd.isna(d[m]):
                    continue
                out.append({
                    "experiment": exp,
                    "norm_type": norm,
                    "mode": mode,
                    "regime": r["regime"],
                    "metric": m,
                    "default_mean": float(d[m]),
                    "regime_mean": float(r[m]),
                    "delta": float(r[m]) - float(d[m]),
                })
    return pd.DataFrame(out, columns=cols)


# ---------------------------------------------------------------------
# Hypothesis verdicts (plan_e1f12eab Step 2 — additive to VARIANT_CRITERIA)
# ---------------------------------------------------------------------


def _compute_hypothesis_verdicts(
    df: pd.DataFrame, probes_long: pd.DataFrame,
) -> pd.DataFrame:
    """Per-cell hypothesis verdicts via ``hypotheses.evaluate_all``.

    Merges the per-seed probe snapshot (``probes_long``) into the merged
    headline frame (``df``) on ``(experiment, norm_type, mode, seed)`` so
    every cell has access to both headline columns (``best_val_acc``,
    ``final_val_loss``, ``generalization_gap``) and probe-aggregated columns
    (``act_per_sample_rms_max_max``, ``act_mean_abs``, ``grad_norm_global``).

    Returns the frame produced by ``evaluate_all_hypotheses`` (one row per
    ``(experiment, norm_type, mode)`` cell). Empty frame if ``df`` is empty.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "experiment", "norm_type", "mode",
            "hypothesis_verdict", "hypothesis_metric",
            "hypothesis_threshold", "hypothesis_observed",
        ])
    merged = df.copy()
    if not probes_long.empty:
        key_cols = ["experiment", "norm_type", "mode", "seed"]
        # Only merge probe columns the hypotheses actually reference, to
        # keep the merged frame tight.
        referenced = {spec.metric_column for spec in VARIANT_HYPOTHESES.values()}
        probe_cols = [c for c in probes_long.columns
                      if c in referenced and c not in merged.columns]
        if probe_cols:
            merged = merged.merge(
                probes_long[key_cols + probe_cols],
                on=key_cols, how="left",
            )
    try:
        return evaluate_all_hypotheses(merged)
    except (KeyError, ValueError) as e:
        logger.warning(f"[report] hypothesis evaluation failed: {e}")
        return pd.DataFrame(columns=[
            "experiment", "norm_type", "mode",
            "hypothesis_verdict", "hypothesis_metric",
            "hypothesis_threshold", "hypothesis_observed",
        ])


def _md_hypothesis_verdict_block(hyp: pd.DataFrame) -> str:
    """Render the hypothesis verdict block for ``summary.md``."""
    if hyp.empty:
        return "*(no hypothesis verdicts — sweep frame or registry empty)*\n"
    out: List[str] = []
    out.append(
        "*Verdicts derived from the falsifiable `VARIANT_HYPOTHESES` "
        "registry (cf. `train.rms_variants_train.hypotheses`). Each row: one "
        "`(experiment, norm_type, mode)` cell; the `observed` value is "
        "compared to the layer's design-claim threshold.*\n"
    )
    out.append(
        "| Experiment | Norm | Mode | Verdict | Metric | Observed | "
        "Threshold |"
    )
    out.append("|---|---|---|---|---|---|---|")
    for _, r in hyp.iterrows():
        observed = r["hypothesis_observed"]
        threshold = r["hypothesis_threshold"]
        obs_s = "—" if pd.isna(observed) else f"{float(observed):.4f}"
        thr_s = "—" if pd.isna(threshold) else f"{float(threshold):.4f}"
        out.append(
            f"| {r['experiment']} | {r['norm_type']} | {r['mode']} | "
            f"**{r['hypothesis_verdict']}** | {r.get('hypothesis_metric', '')} "
            f"| {obs_s} | {thr_s} |"
        )
    # Per-variant rollup.
    out.append("")
    out.append("### Per-variant rollup")
    for variant in NORM_VARIANTS:
        sub = hyp[hyp["norm_type"] == variant]
        if sub.empty:
            continue
        confirmed = (sub["hypothesis_verdict"] == "CONFIRMED").sum()
        rejected = (sub["hypothesis_verdict"] == "REJECTED").sum()
        incon = (sub["hypothesis_verdict"] == "INCONCLUSIVE").sum()
        na = (sub["hypothesis_verdict"] == "N/A").sum()
        out.append(
            f"- `{variant}`: CONFIRMED={int(confirmed)} "
            f"REJECTED={int(rejected)} INCONCLUSIVE={int(incon)} "
            f"N/A={int(na)} (n={len(sub)} cells)"
        )
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------
# Overall recommendation (Refinement B — plan_2026-05-18_6776f8ba step 3)
# ---------------------------------------------------------------------

# DECISION plan_2026-05-18_6776f8ba/D-002: pre-registered overall-recommendation
# rules. The 4-slot taxonomy (RECOMMENDED_DEFAULT / RECOMMENDED_NICHE / NULL /
# AVOID) and the thresholds below are FROZEN at plan-approval time. Any change
# to OVERALL_RULES after this plan is approved requires a new PIVOT entry in
# decisions.md — NOT a silent edit. Rationale: locking the rules pre-sweep
# eliminates curation bias (the temptation to tweak thresholds until the data
# tells the story we wanted). See plan.md Pre-Mortem scenario 3 for the
# falsification trigger (rules collapse / rules never recommend AVOID).
OVERALL_RULES: Dict[str, Any] = {
    # Headline-PASS over baseline at p < 0.05 with correct direction.
    "headline_pass_required_for_recommend": True,
    # Hypothesis registry CONFIRMED required for RECOMMENDED_DEFAULT.
    "hypothesis_confirm_required_for_default": True,
    # Compute overhead vs rms_norm — strict ceiling for RECOMMENDED_*.
    "overhead_ceiling_step_time_ratio": 1.5,
    # Robustness CRITERIA: calibration ECE delta vs rms_norm baseline must
    # NOT regress by more than this (positive = worse calibration).
    "calibration_ece_delta_max": 0.02,
    # Robustness CRITERIA: distribution-shift accuracy delta — recommended
    # variants must NOT lose more than this many points on shifted data.
    "robustness_shift_acc_delta_min": -0.05,
    # AVOID trigger: headline FAIL OR hypothesis REJECTED on a non-off-label
    # cell.
    "avoid_on_headline_fail": True,
    "avoid_on_hypothesis_rejected": True,
}


def compute_overall_recommendation(
    headline_df: pd.DataFrame,
    hypothesis_df: pd.DataFrame,
    calibration_df: Optional[pd.DataFrame] = None,
    robustness_df: Optional[pd.DataFrame] = None,
    overhead_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Produce a frozen-rule overall recommendation per ``norm_type``.

    Inputs (any may be empty / None — degrades to NULL gracefully):
    - ``headline_df``: rows from ``_aggregate_headline`` (with the ``verdict``
      column added). Columns: ``experiment, norm_type, mode, verdict``.
    - ``hypothesis_df``: rows from ``_compute_hypothesis_verdicts``. Columns:
      ``experiment, norm_type, mode, hypothesis_verdict``.
    - ``calibration_df``: optional. Columns: ``norm_type, ece_delta_vs_baseline``.
    - ``robustness_df``: optional. Columns: ``norm_type, shift_acc_delta_vs_baseline``.
    - ``overhead_df``: optional. From ``norm_overhead_bench``. Columns:
      ``norm, params, mean_step_ms_fp32`` (and others). The baseline is
      ``rms_norm``; ratio = norm_step / baseline_step.

    Returns a DataFrame indexed by ``norm_type`` with columns:
        norm_type, recommendation, reason
    where ``recommendation`` is one of:
        - ``RECOMMENDED_DEFAULT``: passes every gate including hypothesis CONFIRMED.
        - ``RECOMMENDED_NICHE``: passes headline + overhead, but hypothesis is
          INCONCLUSIVE (mechanism not confirmed; still a viable substitute).
        - ``NULL``: indistinguishable from baseline; no harm, no gain.
        - ``AVOID``: headline FAIL on any non-off-label cell, OR hypothesis
          REJECTED on any cell, OR overhead > ceiling without redeeming PASS.

    The function is pure: no I/O, no side effects.
    """
    rules = OVERALL_RULES
    norms_seen = set()
    if not headline_df.empty:
        norms_seen.update(headline_df["norm_type"].unique().tolist())
    if not hypothesis_df.empty:
        norms_seen.update(hypothesis_df["norm_type"].unique().tolist())
    for df in (calibration_df, robustness_df):
        if df is not None and not df.empty and "norm_type" in df.columns:
            norms_seen.update(df["norm_type"].unique().tolist())
    if overhead_df is not None and not overhead_df.empty and "norm" in overhead_df.columns:
        norms_seen.update(overhead_df["norm"].unique().tolist())

    # Baseline step time (overhead_df is keyed on `norm`, not `norm_type`).
    baseline_step = None
    if (
        overhead_df is not None
        and not overhead_df.empty
        and "norm" in overhead_df.columns
        and "mean_step_ms_fp32" in overhead_df.columns
    ):
        base_rows = overhead_df[overhead_df["norm"] == BASELINE_NORM]
        if not base_rows.empty:
            v = float(base_rows.iloc[0]["mean_step_ms_fp32"])
            if np.isfinite(v) and v > 0:
                baseline_step = v

    out: List[Dict[str, Any]] = []
    for norm in sorted(norms_seen):
        if norm == BASELINE_NORM:
            # Baseline is itself the reference — emit NULL with a clear reason.
            out.append({
                "norm_type": norm,
                "recommendation": "NULL",
                "reason": "baseline reference; not self-recommending",
            })
            continue

        reasons: List[str] = []

        # --- headline checks ---
        h_rows = headline_df[headline_df["norm_type"] == norm] if not headline_df.empty else pd.DataFrame()
        # Eligible (non-off-label) cells only.
        if not h_rows.empty:
            eligible = h_rows[h_rows["verdict"] != "n/a (off-label)"]
            n_pass = int((eligible["verdict"] == "pass").sum())
            n_fail = int((eligible["verdict"] == "fail").sum())
            n_indis = int((eligible["verdict"] == "indistinguishable").sum())
        else:
            n_pass = n_fail = n_indis = 0

        if rules["avoid_on_headline_fail"] and n_fail > 0:
            out.append({
                "norm_type": norm,
                "recommendation": "AVOID",
                "reason": f"headline FAIL in {n_fail} non-off-label cell(s)",
            })
            continue

        # --- hypothesis check ---
        hyp_rows = hypothesis_df[hypothesis_df["norm_type"] == norm] if not hypothesis_df.empty else pd.DataFrame()
        n_rejected = int((hyp_rows["hypothesis_verdict"] == "REJECTED").sum()) if not hyp_rows.empty else 0
        n_confirmed = int((hyp_rows["hypothesis_verdict"] == "CONFIRMED").sum()) if not hyp_rows.empty else 0

        if rules["avoid_on_hypothesis_rejected"] and n_rejected > 0:
            out.append({
                "norm_type": norm,
                "recommendation": "AVOID",
                "reason": f"hypothesis REJECTED in {n_rejected} cell(s)",
            })
            continue

        # --- overhead check ---
        overhead_ratio: Optional[float] = None
        if baseline_step is not None and overhead_df is not None and not overhead_df.empty:
            n_rows = overhead_df[overhead_df["norm"] == norm]
            if not n_rows.empty:
                v = float(n_rows.iloc[0]["mean_step_ms_fp32"])
                if np.isfinite(v) and v > 0:
                    overhead_ratio = v / baseline_step
        if (
            overhead_ratio is not None
            and overhead_ratio > rules["overhead_ceiling_step_time_ratio"]
        ):
            # Overhead too high. If headline still PASSes, we accept NULL
            # (the gain doesn't justify the cost); if no PASS, AVOID.
            if n_pass > 0:
                out.append({
                    "norm_type": norm,
                    "recommendation": "NULL",
                    "reason": (
                        f"overhead ratio {overhead_ratio:.2f}x exceeds ceiling "
                        f"{rules['overhead_ceiling_step_time_ratio']}x; "
                        f"headline gain insufficient to justify cost"
                    ),
                })
            else:
                out.append({
                    "norm_type": norm,
                    "recommendation": "AVOID",
                    "reason": (
                        f"overhead ratio {overhead_ratio:.2f}x exceeds ceiling "
                        f"and no headline PASS observed"
                    ),
                })
            continue

        # --- calibration / robustness regressions block recommendation ---
        cal_delta: Optional[float] = None
        if calibration_df is not None and not calibration_df.empty:
            r = calibration_df[calibration_df["norm_type"] == norm]
            if not r.empty and "ece_delta_vs_baseline" in r.columns:
                v = float(r.iloc[0]["ece_delta_vs_baseline"])
                if np.isfinite(v):
                    cal_delta = v
        if cal_delta is not None and cal_delta > rules["calibration_ece_delta_max"]:
            out.append({
                "norm_type": norm,
                "recommendation": "NULL",
                "reason": f"calibration ECE delta {cal_delta:+.4f} exceeds {rules['calibration_ece_delta_max']}",
            })
            continue

        rob_delta: Optional[float] = None
        if robustness_df is not None and not robustness_df.empty:
            r = robustness_df[robustness_df["norm_type"] == norm]
            if not r.empty and "shift_acc_delta_vs_baseline" in r.columns:
                v = float(r.iloc[0]["shift_acc_delta_vs_baseline"])
                if np.isfinite(v):
                    rob_delta = v
        if rob_delta is not None and rob_delta < rules["robustness_shift_acc_delta_min"]:
            out.append({
                "norm_type": norm,
                "recommendation": "NULL",
                "reason": f"robustness shift acc delta {rob_delta:+.4f} below floor {rules['robustness_shift_acc_delta_min']}",
            })
            continue

        # --- final recommendation slot ---
        if n_pass >= 2 and (not rules["hypothesis_confirm_required_for_default"] or n_confirmed >= 1):
            out.append({
                "norm_type": norm,
                "recommendation": "RECOMMENDED_DEFAULT",
                "reason": (
                    f"headline PASS in {n_pass} cells; hypothesis CONFIRMED "
                    f"in {n_confirmed}; overhead within ceiling"
                ),
            })
        elif n_pass >= 1:
            out.append({
                "norm_type": norm,
                "recommendation": "RECOMMENDED_NICHE",
                "reason": (
                    f"headline PASS in {n_pass} cell(s); hypothesis not "
                    f"CONFIRMED (n_confirmed={n_confirmed})"
                ),
            })
        else:
            out.append({
                "norm_type": norm,
                "recommendation": "NULL",
                "reason": (
                    f"no headline PASS (indis={n_indis}, fail={n_fail}); "
                    f"behaves like baseline"
                ),
            })

    return pd.DataFrame(out, columns=["norm_type", "recommendation", "reason"])


def write_report(df: pd.DataFrame, *, out_dir: str) -> None:
    """Write ``summary.md`` aggregating ``df`` (the merged ``all_runs.csv``)."""
    headline = _aggregate_headline(df)
    headline = _add_paired_p(headline, df)
    headline["verdict"] = headline.apply(_verdict, axis=1)

    probes_long = _final_probe_snapshot(out_dir, df)
    probes = _probes_summary(probes_long)

    # Phase 3 post-hoc derivations (best-effort; empty frames if data missing).
    try:
        df_per_epoch = _load_per_epoch_frame(out_dir, df)
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        logger.warning(f"[report] per-epoch frame load failed: {e}")
        df_per_epoch = pd.DataFrame()
    try:
        convergence = _compute_convergence_speed(df_per_epoch)
    except (KeyError, ValueError) as e:
        logger.warning(f"[report] convergence-speed derivation failed: {e}")
        convergence = pd.DataFrame()
    try:
        late_stab = _compute_late_stability(df_per_epoch)
    except (KeyError, ValueError) as e:
        logger.warning(f"[report] late-stability derivation failed: {e}")
        late_stab = pd.DataFrame()
    try:
        regime_delta = _compute_regime_delta(df)
    except (KeyError, ValueError) as e:
        logger.warning(f"[report] regime-delta derivation failed: {e}")
        regime_delta = pd.DataFrame()

    # Merge late-stability into headline (extend headline_summary.csv).
    if not late_stab.empty:
        agg_keys = ["experiment", "norm_type", "mode"]
        ls_agg = (
            late_stab.groupby(agg_keys, sort=False)["late_stability_var"]
            .mean()
            .reset_index()
        )
        headline = headline.merge(ls_agg, on=agg_keys, how="left")
    else:
        headline["late_stability_var"] = float("nan")

    # Hypothesis verdicts (plan_e1f12eab Step 2 — additive to PASS/FAIL).
    hyp = _compute_hypothesis_verdicts(df, probes_long)
    if not hyp.empty:
        agg_keys = ["experiment", "norm_type", "mode"]
        headline = headline.merge(
            hyp[agg_keys + ["hypothesis_verdict"]],
            on=agg_keys, how="left",
        )
    else:
        headline["hypothesis_verdict"] = "N/A"

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
    md_lines.append("## Hypothesis verdicts (falsifiable design-claim checks)\n")
    md_lines.append(_md_hypothesis_verdict_block(hyp))

    summary_path = os.path.join(out_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(md_lines))

    # Also persist the headline frame as CSV for downstream tooling.
    headline.to_csv(os.path.join(out_dir, "headline_summary.csv"), index=False)
    probes.to_csv(os.path.join(out_dir, "probes_summary.csv"), index=False)
    # Phase 3 derivation CSVs (always emitted — empty frames retain schema).
    convergence.to_csv(os.path.join(out_dir, "convergence_summary.csv"), index=False)
    regime_delta.to_csv(os.path.join(out_dir, "regime_delta_summary.csv"), index=False)
    hyp.to_csv(os.path.join(out_dir, "hypothesis_verdicts.csv"), index=False)

    # Refinement B: overall recommendation (frozen rules, D-002 anchor).
    # Best-effort: load overhead.csv / calibration_summary.csv / robustness_summary.csv
    # from out_dir if present; otherwise pass None and rules degrade gracefully.
    def _maybe_read(name: str) -> Optional[pd.DataFrame]:
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            return None
        try:
            return pd.read_csv(path)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            return None

    overhead_df = _maybe_read("overhead.csv")
    calibration_df = _maybe_read("calibration_summary.csv")
    robustness_df = _maybe_read("robustness_summary.csv")
    try:
        overall = compute_overall_recommendation(
            headline, hyp, calibration_df, robustness_df, overhead_df
        )
    except (KeyError, ValueError) as e:
        logger.warning(f"[report] compute_overall_recommendation failed: {e}")
        overall = pd.DataFrame(columns=["norm_type", "recommendation", "reason"])
    overall.to_csv(os.path.join(out_dir, "overall_recommendation.csv"), index=False)
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
