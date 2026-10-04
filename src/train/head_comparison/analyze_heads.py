"""
Aggregate head-comparison runs into the pre-registered verdict table.

Walks an output directory of ``train_heads.py`` run dirs (each carrying
``test_metrics.json``, ``test_features.npz``, ``head_prototypes.npz`` and
``training_history.json``), aggregates per-arm means with bootstrap CIs,
paired arm differences with permutation tests, and representation
structure (silhouette / explained variance / grokking dynamics from
``dl_techniques.analyzer.representation_metrics``).

Writes ``comparison.json`` and ``comparison.md``. Hypotheses under test:

- H1 (parity): fine accuracy ties across arms.
- H2 (taxonomy): hierarchical arms win coarse accuracy and cluster purity.
- H3 (tails/calibration): harmonic arms win tail-decile accuracy and ECE.
- H5 (structure): harmonic arms show higher silhouette / lower
  parallelogram-style structure at accuracy parity. Parallelogram loss
  itself needs relational (input, output) pairs, which CIFAR-100 has no
  natural source for, so structure is scored with partition silhouette
  (true superclasses, and learned partitions where available) plus PCA
  explained variance.

Usage:
    python -m train.head_comparison.analyze_heads --runs-dir results --out comparison
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from train.head_comparison.train_heads import HEAD_ARMS
from train.common.stats import mean_std, bootstrap_ci, paired_permutation_test
from dl_techniques.analyzer.representation_metrics import (
    partition_silhouette,
    top_k_explained_variance,
    epochs_to_threshold,
    grokking_gap,
)
from dl_techniques.utils.logger import logger


def load_runs(runs_dir: str, prefix: str = "cmp_") -> List[Dict[str, Any]]:
    """Collect per-run payloads (metrics, features, prototypes, history).

    :param runs_dir: Directory holding one run directory per arm/seed.
    :type runs_dir: str
    :param prefix: Only aggregate run dirs starting with this prefix.
    :type prefix: str
    :return: List of dicts with ``arm``, ``seed``, ``metrics``,
        ``features`` (or None), ``prototypes`` (or None), ``history``
        (or None).
    :rtype: List[Dict[str, Any]]
    """
    runs = []
    for child in sorted(Path(runs_dir).iterdir()):
        metrics_path = child / "test_metrics.json"
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        with open(child / "config.json") as f:
            config = json.load(f)
        payload: Dict[str, Any] = {
            "arm": config.get("head", "unknown"),
            "seed": config.get("seed", -1),
            "metrics": metrics,
            "features": None,
            "prototypes": None,
            "history": None,
        }
        feat_path = child / "test_features.npz"
        if feat_path.exists():
            payload["features"] = dict(np.load(feat_path, allow_pickle=False))
        proto_path = child / "head_prototypes.npz"
        if proto_path.exists():
            payload["prototypes"] = dict(np.load(proto_path, allow_pickle=False))
        hist_path = child / "training_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                payload["history"] = json.load(f)
        runs.append(payload)
    logger.info(f"Loaded {len(runs)} runs from {runs_dir}")
    return runs


def summarize_metric(
        runs: List[Dict[str, Any]], key: str, rng: np.random.Generator
) -> Dict[str, Dict[str, float]]:
    """Per-arm mean/std/CI of a scalar metric key.

    :param runs: Payloads from :func:`load_runs`.
    :type runs: List[Dict[str, Any]]
    :param key: Metric key inside each run's ``metrics``.
    :type key: str
    :param rng: Caller-owned RNG for bootstrapping.
    :type rng: np.random.Generator
    :return: arm -> ``{mean, std, ci_low, ci_high, n}``.
    :rtype: Dict[str, Dict[str, float]]
    """
    table = {}
    for arm in HEAD_ARMS:
        values = [r["metrics"][key] for r in runs if r["arm"] == arm and key in r["metrics"]]
        mean, std = mean_std(values)
        ci_low, ci_high = bootstrap_ci(values, rng=rng)
        table[arm] = {
            "mean": mean, "std": std, "ci_low": ci_low, "ci_high": ci_high,
            "n": len(values),
        }
    return table


def paired_arm_differences(
        runs: List[Dict[str, Any]], key: str, rng: np.random.Generator
) -> Dict[str, Dict[str, float]]:
    """Permutation p-values for arm differences vs softmax, paired by seed.

    :param runs: Payloads from :func:`load_runs`.
    :type runs: List[Dict[str, Any]]
    :param key: Metric key.
    :type key: str
    :param rng: Caller-owned RNG.
    :type rng: np.random.Generator
    :return: ``"<arm>-softmax"`` -> ``{mean_diff, p_value}``.
    :rtype: Dict[str, Dict[str, float]]
    """
    base = {r["seed"]: r["metrics"].get(key, np.nan) for r in runs if r["arm"] == "softmax"}
    table = {}
    for arm in HEAD_ARMS:
        if arm == "softmax":
            continue
        pairs = [
            (r["metrics"].get(key, np.nan), base.get(r["seed"], np.nan))
            for r in runs if r["arm"] == arm
        ]
        pairs = [(a, b) for a, b in pairs if np.isfinite(a) and np.isfinite(b)]
        if not pairs:
            table[f"{arm}-softmax"] = {"mean_diff": float("nan"), "p_value": float("nan")}
            continue
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        _, p_value = paired_permutation_test(a, b, rng=rng)
        table[f"{arm}-softmax"] = {"mean_diff": float(np.mean(a - b)), "p_value": float(p_value)}
    return table


def representation_table(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Per-arm representation structure (silhouette, variance, grokking).

    :param runs: Payloads from :func:`load_runs`.
    :type runs: List[Dict[str, Any]]
    :return: arm -> averaged structure metrics (NaN where unavailable).
    :rtype: Dict[str, Dict[str, float]]
    """
    table: Dict[str, Dict[str, List[float]]] = {}

    def _quiet_mean(values: List[float]) -> float:
        finite = [v for v in values if np.isfinite(v)]
        return float(np.mean(finite)) if finite else float("nan")

    for run in runs:
        arm = run["arm"]
        entry = table.setdefault(arm, {
            "silhouette_coarse": [], "silhouette_fine": [],
            "top2_variance": [], "grokking_gap": [],
        })
        feats = run["features"]
        if feats is not None:
            features = np.asarray(feats["features"], dtype=np.float64)
            # Subsample for the O(N^2) silhouette (10k points -> 2k).
            sub = np.random.default_rng(0).choice(
                len(features), size=min(2000, len(features)), replace=False
            )
            entry["silhouette_coarse"].append(partition_silhouette(
                features[sub], np.asarray(feats["y_coarse"])[sub].tolist()))
            entry["silhouette_fine"].append(partition_silhouette(
                features[sub], np.asarray(feats["y_fine"])[sub].tolist()))
            entry["top2_variance"].append(top_k_explained_variance(features, k=2))
        hist = run["history"] or {}
        train_acc = hist.get("accuracy", [])
        val_acc = hist.get("val_accuracy", [])
        if train_acc and val_acc:
            gap = grokking_gap(train_acc, val_acc)
            entry["grokking_gap"].append(float("nan") if gap is None else gap)

    return {
        arm: {k: _quiet_mean(v) for k, v in vals.items()}
        for arm, vals in table.items()
    }


def write_markdown(path: str, scalar_tables: Dict[str, Any], rep: Dict[str, Any]) -> None:
    """Write the human-readable verdict table.

    :param path: Output ``.md`` path.
    :type path: str
    :param scalar_tables: Metric -> arm table from :func:`summarize_metric`.
    :type scalar_tables: Dict[str, Any]
    :param rep: Representation table from :func:`representation_table`.
    :type rep: Dict[str, Any]
    """
    lines = ["# Head comparison verdicts", ""]
    for metric, table in scalar_tables.items():
        lines.append(f"## {metric}")
        lines.append("| arm | mean ± std | 95% CI | n |")
        lines.append("|---|---|---|---|")
        for arm in HEAD_ARMS:
            row = table.get(arm, {})
            lines.append(
                f"| {arm} | {row.get('mean', float('nan')):.4f} ± "
                f"{row.get('std', float('nan')):.4f} | "
                f"[{row.get('ci_low', float('nan')):.4f}, "
                f"{row.get('ci_high', float('nan')):.4f}] | {row.get('n', 0)} |"
            )
        lines.append("")
    lines.append("## representation structure (mean over runs)")
    lines.append("| arm | silhouette_coarse | silhouette_fine | top2_variance | grokking_gap |")
    lines.append("|---|---|---|---|---|")
    for arm in HEAD_ARMS:
        row = rep.get(arm, {})
        lines.append(
            f"| {arm} | {row.get('silhouette_coarse', float('nan')):.4f} | "
            f"{row.get('silhouette_fine', float('nan')):.4f} | "
            f"{row.get('top2_variance', float('nan')):.4f} | "
            f"{row.get('grokking_gap', float('nan')):.1f} |"
        )
    lines += [
        "",
        "## provenance",
        "- Only run dirs starting with the `--prefix` (default `cmp_`) aggregate; "
        "pilot/smoke dirs are excluded.",
        "- ECE for the `softmax` / `harmonic_logits` arms is recomputed from "
        "softmaxed best-checkpoint probabilities: the harness originally read "
        "raw max-logit as confidence (since corrected in-train). Patched runs "
        "carry `ece_corrected_from_logits: true` in `test_metrics.json`.",
        "- Grokking gap reads NaN when the 0.9 threshold is never reached.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """CLI entry point: parse argv first, then aggregate."""
    parser = argparse.ArgumentParser(
        description="Aggregate head-comparison runs into verdict tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs-dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="cmp_",
                        help="Only aggregate run dirs starting with this prefix "
                        "(keeps pilot/smoke dirs out of the verdict table).")
    parser.add_argument("--out", type=str, default="comparison")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    runs = load_runs(args.runs_dir, prefix=args.prefix)
    result: Dict[str, Any] = {"n_runs": len(runs), "metrics": {}, "contrasts": {}}
    for key in ("fine_accuracy", "coarse_accuracy", "tail_decile_accuracy", "ece"):
        result["metrics"][key] = summarize_metric(runs, key, rng)
        result["contrasts"][key] = paired_arm_differences(runs, key, rng)
    result["representation"] = representation_table(runs)
    with open(f"{args.out}.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    write_markdown(f"{args.out}.md", result["metrics"], result["representation"])
    logger.info(f"Wrote {args.out}.json and {args.out}.md over {len(runs)} runs.")
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
