"""Aggregate an embeddings-study sweep into a report.

Reads the per-cell ``results.json`` files a sweep leaves behind and writes a
``summary.md`` plus machine-readable CSVs.

The statistics follow :mod:`train.rms_variants_train.report`: a bootstrap
confidence interval per cell group, and a **paired permutation test** against a
named baseline arm, pairing on seed so the comparison controls for
initialization. A fixed RNG seed makes the report reproducible.

Two disciplines carried over from that harness, both worth keeping:

- **Compare like with like.** A paired statistic is computed only across arms
  that share a variant AND a pooling strategy; averaging over those axes would
  compare an arm's best pooling to another's worst.
- **Report the spread, not just the mean.** A group whose standard deviation
  exceeds the absolute value of its mean is flagged, because a headline number
  from three noisy seeds is not a result.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from train.common.stats import (
    bootstrap_ci,
    format_mean_std,
    mean_std,
    paired_permutation_test,
)

from .config import BASELINE_MODEL

# ---------------------------------------------------------------------

__all__ = ["build_report", "main", "write_report"]

#: Deterministic RNG seed, so two runs of the report agree.
RNG_SEED = 20260828

#: Fewest seeds at which the paired test can reach p < 0.05 AT ALL.
#:
#: ``paired_permutation_test`` is a TWO-SIDED sign-flip test, so with ``n``
#: pairs the smallest reachable p-value is about ``2 / 2**n`` -- the observed
#: sign vector and its mirror. MEASURED against maximally separated arms:
#:
#:     n=2 p=0.502   n=3 p=0.248   n=4 p=0.125   n=5 p=0.063   n=6 p=0.031
#:
#: So at five seeds or fewer, NO effect size however large can be reported
#: significant. A sweep run with three seeds cannot conclude anything, and
#: would silently report every comparison as indistinguishable -- which reads
#: like a finding and is not one. Such comparisons are labelled UNDERPOWERED
#: rather than INDISTINGUISHABLE.
MIN_SEEDS_FOR_SIGNIFICANCE = 6

#: metric key -> (human label, direction). "min" means lower is better.
HEADLINE_METRICS: Dict[str, Tuple[str, str]] = {
    "mlm_val_loss_best": ("MLM val loss (best)", "min"),
    "mlm_val_accuracy_best": ("MLM val accuracy (best)", "max"),
    "contrastive_val_loss_best": ("Contrastive val loss (best)", "min"),
}

#: Axes that identify a comparison group. Arms are compared WITHIN a group.
GROUP_AXES = ("variant", "pooling")


def _load_records(sweep_root: str) -> List[Dict[str, Any]]:
    """Load the collected records, preferring the sweep's own summary file."""
    summary = os.path.join(sweep_root, "all_runs.json")
    if os.path.exists(summary):
        with open(summary) as handle:
            return json.load(handle)

    from .sweep import collect_results

    return collect_results(sweep_root)


def _dedupe(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep one record per cell, so overlapping chunked sweeps do not double-count."""
    seen: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for record in records:
        key = (
            record.get("model"),
            record.get("variant"),
            record.get("pooling"),
            record.get("seed"),
        )
        seen[key] = record
    return list(seen.values())


def build_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute the report's tables from the collected records.

    :param records: One record per cell.
    :type records: list[dict[str, Any]]
    :return: ``{"headline": [...], "paired": [...], "flags": [...]}``.
    :rtype: dict[str, Any]
    """
    records = _dedupe(records)
    rng = np.random.default_rng(RNG_SEED)

    by_cell: Dict[Tuple[Any, ...], Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = tuple(record.get(axis) for axis in GROUP_AXES)
        by_cell[(record.get("model"),) + key][record.get("seed")] = record

    headline: List[Dict[str, Any]] = []
    flags: List[str] = []
    for (model, *group), seeded in sorted(
        by_cell.items(), key=lambda item: tuple(str(x) for x in item[0])
    ):
        for metric, (label, direction) in HEADLINE_METRICS.items():
            values = [
                r[metric] for r in seeded.values() if r.get(metric) is not None
            ]
            if not values:
                continue
            mean, std = mean_std(values)
            row: Dict[str, Any] = {
                "model": model,
                **dict(zip(GROUP_AXES, group)),
                "metric": metric,
                "label": label,
                "direction": direction,
                "n": len(values),
                "mean": mean,
                "std": std,
                "ci_low": None,
                "ci_high": None,
                "parameters": next(
                    (r.get("parameters") for r in seeded.values()), None
                ),
            }
            if len(values) >= 2:
                low, high = bootstrap_ci(
                    values, confidence=0.95, n_boot=2000, rng=rng
                )
                row["ci_low"], row["ci_high"] = low, high
                if std > abs(mean) and mean != 0:
                    flags.append(
                        f"{model} {group} {metric}: std {std:.4g} exceeds "
                        f"|mean| {abs(mean):.4g} over n={len(values)}"
                    )
            headline.append(row)

    paired: List[Dict[str, Any]] = []
    groups = {tuple(k[1:]) for k in by_cell}
    for group in sorted(groups, key=lambda g: tuple(str(x) for x in g)):
        baseline_cells = by_cell.get((BASELINE_MODEL,) + group, {})
        if not baseline_cells:
            continue
        for (model, *other_group), seeded in by_cell.items():
            if model == BASELINE_MODEL or tuple(other_group) != group:
                continue
            for metric, (label, direction) in HEADLINE_METRICS.items():
                shared = sorted(
                    seed
                    for seed in seeded
                    if seed in baseline_cells
                    and seeded[seed].get(metric) is not None
                    and baseline_cells[seed].get(metric) is not None
                )
                if len(shared) < 2:
                    continue
                arm = [seeded[s][metric] for s in shared]
                base = [baseline_cells[s][metric] for s in shared]
                diff, p_value = paired_permutation_test(
                    arm, base, n_perm=10000, rng=rng
                )
                better = (diff < 0) if direction == "min" else (diff > 0)
                if len(shared) < MIN_SEEDS_FOR_SIGNIFICANCE:
                    # Not a finding: at this many pairs the test cannot reach
                    # p < 0.05 for ANY effect size.
                    verdict = "UNDERPOWERED"
                elif p_value < 0.05:
                    verdict = "BETTER" if better else "WORSE"
                else:
                    verdict = "INDISTINGUISHABLE"
                paired.append(
                    {
                        "model": model,
                        "baseline": BASELINE_MODEL,
                        **dict(zip(GROUP_AXES, group)),
                        "metric": metric,
                        "label": label,
                        "direction": direction,
                        "n_pairs": len(shared),
                        "diff_vs_baseline": diff,
                        "p_value": p_value,
                        "verdict": verdict,
                    }
                )

    return {"headline": headline, "paired": paired, "flags": flags}


def _markdown_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    """Render rows as a markdown table."""
    if not rows:
        return "_no rows_\n"
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                cells.append(f"{value:.4g}")
            else:
                cells.append("" if value is None else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(report: Dict[str, Any], out_dir: str) -> str:
    """Write ``summary.md`` and the CSVs.

    :param report: Output of :func:`build_report`.
    :type report: dict[str, Any]
    :param out_dir: Directory to write into.
    :type out_dir: str
    :return: Path to ``summary.md``.
    :rtype: str
    """
    os.makedirs(out_dir, exist_ok=True)

    headline_columns = (
        "model", *GROUP_AXES, "metric", "n", "mean", "std",
        "ci_low", "ci_high", "parameters",
    )
    paired_columns = (
        "model", "baseline", *GROUP_AXES, "metric", "n_pairs",
        "diff_vs_baseline", "p_value", "verdict",
    )

    for name, rows, columns in (
        ("headline_summary", report["headline"], headline_columns),
        ("paired_summary", report["paired"], paired_columns),
    ):
        path = os.path.join(out_dir, f"{name}.csv")
        with open(path, "w") as handle:
            handle.write(",".join(columns) + "\n")
            for row in rows:
                handle.write(
                    ",".join(
                        "" if row.get(c) is None else str(row.get(c))
                        for c in columns
                    )
                    + "\n"
                )

    summary_path = os.path.join(out_dir, "summary.md")
    with open(summary_path, "w") as handle:
        handle.write("# Embeddings study\n\n")
        handle.write(
            f"Baseline arm: `{BASELINE_MODEL}`. Paired permutation test "
            "(B=10000) against it, pairing on seed; bootstrap CI at 95% "
            f"(B=2000); RNG seed {RNG_SEED}.\n\n"
        )
        handle.write(
            "Arms are compared only WITHIN a matched (variant, pooling) group. "
            "Note that equal variant names do NOT mean equal parameter counts: "
            "the arms are depth- and width-matched, not parameter-matched, so "
            "read the `parameters` column alongside every result.\n\n"
        )
        handle.write("## Headline\n\n")
        handle.write(_markdown_table(report["headline"], headline_columns))
        handle.write("\n## Paired against the baseline\n\n")
        handle.write(_markdown_table(report["paired"], paired_columns))
        handle.write("\n## High-variance flags\n\n")
        if report["flags"]:
            for flag in report["flags"]:
                handle.write(f"- {flag}\n")
        else:
            handle.write("_none_\n")
        underpowered = [
            r for r in report["paired"] if r["verdict"] == "UNDERPOWERED"
        ]
        handle.write(
            "\n## Reading this report\n\n"
            f"The paired test is two-sided sign-flip, so with `n` pairs the "
            f"smallest reachable p-value is about `2/2**n`. At "
            f"{MIN_SEEDS_FOR_SIGNIFICANCE - 1} seeds or fewer, NO effect size "
            "however large can be reported significant (measured: n=3 gives "
            "p=0.248, n=5 gives p=0.063, n=6 gives p=0.031). Such comparisons "
            "are labelled UNDERPOWERED, not INDISTINGUISHABLE, because "
            "'no significant difference' from an underpowered test reads like "
            "a finding and is not one.\n\n"
            f"Run at least {MIN_SEEDS_FOR_SIGNIFICANCE} seeds for any "
            "comparison you intend to quote.\n"
        )
        if underpowered:
            handle.write(
                f"\n**{len(underpowered)} of {len(report['paired'])} "
                "comparisons in this report are UNDERPOWERED.**\n"
            )
    logger.info(f"Wrote {summary_path}")
    return summary_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.

    :param argv: Argument vector; ``None`` uses ``sys.argv[1:]``.
    :type argv: Sequence[str] | None
    :return: Process exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Aggregate an embeddings-study sweep into a report."
    )
    parser.add_argument("--in-dir", type=str, default="results/embeddings_study")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args(argv)

    records = _load_records(args.in_dir)
    if not records:
        logger.error(f"no cell results found under {args.in_dir}")
        return 1

    report = build_report(records)
    write_report(report, args.out_dir or args.in_dir)
    logger.info(
        f"{len(records)} cells, {len(report['paired'])} paired comparisons, "
        f"{len(report['flags'])} high-variance flags"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
