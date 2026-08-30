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
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from train.common.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    format_mean_std,
    holm_bonferroni,
    mean_std,
    min_pairs_for_significance,
    paired_permutation_test,
)

from .config import BASELINE_MODEL
from .paths import resolve_output_dir

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

class MetricSpec(NamedTuple):
    """How one metric is reported.

    ``direction`` stays at index 1 so existing positional access keeps working.

    ``role`` is what stops a diagnostic from being read as a quality claim:

    - ``primary``   the single pre-registered confirmatory endpoint. Only this
                    role produces a BETTER/WORSE verdict, corrected with Holm.
    - ``secondary`` reported with a BH-adjusted q-value and the verdict
                    ``SECONDARY``, never BETTER/WORSE.
    - ``diagnostic`` mean and CI only, NO paired test at all. A random
                    projection maximizes ``effective_rank``, minimizes
                    ``anisotropy`` and minimizes ``uniformity`` while retrieving
                    nothing, so "arm X is significantly better on anisotropy"
                    would be a false claim about quality. Diagnostics explain
                    WHY a primary number moved; they are not evidence that it
                    did.
    """

    label: str
    direction: Optional[str]
    role: str


#: metric key -> MetricSpec. Directions for history-derived keys must agree with
#: `metric_directions.direction_of`; a test asserts it.
HEADLINE_METRICS: Dict[str, MetricSpec] = {
    # --- primary: the one confirmatory endpoint -----------------------
    "eval_squad_mrr_at_10": MetricSpec(
        "SQuAD MRR@10 (context prefixes)", "max", "primary"
    ),
    # --- secondary ----------------------------------------------------
    "eval_squad_recall_at_1": MetricSpec("SQuAD R@1", "max", "secondary"),
    "eval_squad_recall_at_10": MetricSpec("SQuAD R@10", "max", "secondary"),
    "eval_sst2_probe_accuracy": MetricSpec(
        "SST-2 probe accuracy (padded corpus)", "max", "secondary"
    ),
    "mlm_val_loss_best": MetricSpec("MLM val loss (best)", "min", "secondary"),
    "mlm_val_accuracy_best": MetricSpec(
        "MLM val accuracy (best)", "max", "secondary"
    ),
    "contrastive_val_loss_best": MetricSpec(
        "Contrastive val loss (best)", "min", "secondary"
    ),
    # --- diagnostics: no verdict, no p-value --------------------------
    # Whitened retrieval is DIAGNOSTIC, not secondary, and the choice is
    # load-bearing. RESULTS.md asks for raw and whitened to be quoted together,
    # but promoting these to `secondary` would grow the BH family from 18 tests
    # to 24 and so RAISE the seed floor the README derives (18 tests -> 10
    # seeds). A post-hoc transform of an existing endpoint should not cost the
    # study statistical power it then has to buy back with GPU time.
    "eval_squad_whitened_recall_at_1": MetricSpec(
        "SQuAD R@1, ZCA-whitened", "max", "diagnostic"
    ),
    "eval_squad_whitened_mrr_at_10": MetricSpec(
        "SQuAD MRR@10, ZCA-whitened", "max", "diagnostic"
    ),
    "eval_squad_median_rank": MetricSpec(
        "SQuAD median rank", "min", "diagnostic"
    ),
    "eval_squad_ctx_anisotropy": MetricSpec(
        "Context anisotropy", None, "diagnostic"
    ),
    "eval_squad_ctx_effective_rank": MetricSpec(
        "Context effective rank", None, "diagnostic"
    ),
    "eval_squad_alignment": MetricSpec(
        "Alignment (question, context)", None, "diagnostic"
    ),
    "eval_squad_uniformity": MetricSpec("Uniformity", None, "diagnostic"),
    "eval_squad_ctx_norm_mean": MetricSpec(
        "Mean embedding L2 norm", None, "diagnostic"
    ),
    "eval_squad_ctx_pad_fraction": MetricSpec(
        "Context padding fraction", None, "diagnostic"
    ),
    "eval_sst2_pad_fraction": MetricSpec(
        "SST-2 padding fraction", None, "diagnostic"
    ),
}

#: Metrics whose chance level is known and must be printed beside the result,
#: so "low" can be told apart from "no better than chance".
CHANCE_BASELINE_KEYS: Dict[str, str] = {
    "eval_squad_recall_at_1": "eval_squad_chance_recall_at_1",
    "eval_sst2_probe_accuracy": "eval_sst2_majority_baseline",
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
        for metric, spec in HEADLINE_METRICS.items():
            label, direction = spec.label, spec.direction
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
                "role": spec.role,
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
            for metric, spec in HEADLINE_METRICS.items():
                # Diagnostics get no paired test at all. A BETTER/WORSE verdict
                # on one would be a false quality claim: a random projection
                # wins on anisotropy, effective rank and uniformity while
                # retrieving nothing.
                if spec.role == "diagnostic":
                    continue
                label, direction = spec.label, spec.direction
                shared = sorted(
                    seed
                    for seed in seeded
                    if seed in baseline_cells
                    and seeded[seed].get(metric) is not None
                    and baseline_cells[seed].get(metric) is not None
                )
                if len(shared) < 1:
                    continue
                # A single pair still produces a row, deliberately: it is
                # labelled UNDERPOWERED below. An ABSENT row reads as "no
                # difference found", which is a stronger claim than a
                # one-seed run can make.
                arm = [seeded[s][metric] for s in shared]
                base = [baseline_cells[s][metric] for s in shared]
                diff, p_value = paired_permutation_test(
                    arm, base, n_perm=10000, rng=rng
                )
                better = (diff < 0) if direction == "min" else (diff > 0)
                paired.append(
                    {
                        "model": model,
                        "baseline": BASELINE_MODEL,
                        **dict(zip(GROUP_AXES, group)),
                        "metric": metric,
                        "label": label,
                        "direction": direction,
                        "role": spec.role,
                        "n_pairs": len(shared),
                        "diff_vs_baseline": diff,
                        "p_value": p_value,
                        "better": better,
                    }
                )

    _apply_corrections(paired)
    return {"headline": headline, "paired": paired, "flags": flags}



def _apply_corrections(paired: List[Dict[str, Any]]) -> None:
    """Assign adjusted p-values and verdicts, in place.

    Two families, and the split is the substantive choice:

    - **Primary** -- the single pre-registered endpoint, across the
      non-baseline arms WITHIN one comparison group. Corrected with **Holm**
      (family-wise): the family is tiny, it backs one categorical claim, and
      Holm is valid under arbitrary dependence where BH needs positive
      dependency. This is the only family that yields BETTER/WORSE.
    - **Secondary** -- every secondary metric across the arms within one group.
      Corrected with **BH** (false-discovery rate): this family is exploratory
      and strongly positively dependent, since recall@1, recall@10 and MRR@10
      are three weightings of one recall curve. Reported as ``SECONDARY``,
      never BETTER/WORSE.

    Groups are separate families rather than pooled. A ``(variant, pooling)``
    cell is a separate question, and correcting across them would penalise the
    study for asking more questions rather than for asking one badly.

    The correction raises the seed floor, which is why it is worth stating:
    a sign-flip test over ``n`` pairs cannot produce a p below ``2/2**n``, so a
    family of size ``m`` needs ``n >= 1 + log2(m/alpha)`` -- 6 seeds
    uncorrected, 7 for a 3-arm primary family, 10 for an 18-test secondary one.
    """
    families: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in paired:
        group = tuple(row[axis] for axis in GROUP_AXES)
        if row["role"] == "primary":
            families[("primary", group, row["metric"])].append(row)
        else:
            families[("secondary", group)].append(row)

    for key, rows in families.items():
        kind = key[0]
        p_values = [r["p_value"] for r in rows]
        correct = holm_bonferroni if kind == "primary" else benjamini_hochberg
        _, adjusted = correct(p_values, alpha=0.05)
        family_size = len(rows)
        floor = min_pairs_for_significance(family_size)

        for row, p_adj in zip(rows, adjusted):
            row["p_adjusted"] = float(p_adj)
            row["family_size"] = family_size
            row["correction"] = "holm" if kind == "primary" else "bh"
            row["min_seeds_for_family"] = floor

            if row["n_pairs"] < floor:
                # Not a finding: below this many pairs the corrected test
                # cannot reject for ANY effect size.
                row["verdict"] = "UNDERPOWERED"
            elif kind == "secondary":
                row["verdict"] = "SECONDARY"
            elif p_adj < 0.05:
                row["verdict"] = "BETTER" if row["better"] else "WORSE"
            else:
                row["verdict"] = "INDISTINGUISHABLE"


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
        "model", *GROUP_AXES, "metric", "role", "n", "mean", "std",
        "ci_low", "ci_high", "parameters",
    )
    paired_columns = (
        "model", "baseline", *GROUP_AXES, "metric", "role", "n_pairs",
        "diff_vs_baseline", "p_value", "p_adjusted", "correction",
        "family_size", "min_seeds_for_family", "verdict",
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
        coverage = [r for r in report["headline"]]
        handle.write("\n## Protocol and caveats\n\n")
        handle.write(
            "- **Train/eval overlap.** SQuAD contexts ARE Wikipedia paragraphs "
            "and the MLM corpus is Wikipedia. Every arm shares the leak, so a "
            "RELATIVE comparison survives it; no absolute claim does.\n"
            "- **Prefix retrieval.** The position table is sized at the "
            "training length, so contexts (mean 780, median 705, p90 1166 "
            "characters) are truncated to the window. The task is matching a "
            "context PREFIX, not a passage.\n"
            "- **Padding.** Stage 1 trains packed and padding-free because one "
            "arm's block cannot honour a padding mask. Evaluation cannot pack, "
            "so the `*_pad_fraction` diagnostics report how much padding each "
            "corpus actually carried under length-sorted batching.\n"
            "- **Pooled, not projected.** Stage 2 saves the encoder, not the "
            "SimCSE wrapper, so these metrics are computed on `pooled_output`; "
            "the contrastive loss is measured in projection space and the two "
            "can move in opposite directions.\n"
            "- **Correction.** The primary endpoint is corrected with Holm "
            "across arms within a group; secondary metrics with "
            "Benjamini-Hochberg. Groups are SEPARATE families, not pooled -- a "
            "position a reader may disagree with. Diagnostics get no p-value "
            "at all, because a random projection wins on anisotropy, effective "
            "rank and uniformity while retrieving nothing.\n"
            "- **Chance levels** are reported as metrics "
            "(`eval_squad_chance_recall_at_1`, `eval_sst2_majority_baseline`) "
            "so a near-floor number is legible as near-floor.\n"
        )
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

    # Resolve against the REPO ROOT, matching the trainer and the sweep. The
    # documented invocation runs from `src/`, where a bare relative path would
    # silently point at a second, empty results tree.
    in_dir = resolve_output_dir(args.in_dir)
    records = _load_records(in_dir)
    if not records:
        logger.error(f"no cell results found under {in_dir}")
        return 1

    report = build_report(records)
    write_report(report, resolve_output_dir(args.out_dir) if args.out_dir else in_dir)
    logger.info(
        f"{len(records)} cells, {len(report['paired'])} paired comparisons, "
        f"{len(report['flags'])} high-variance flags"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
