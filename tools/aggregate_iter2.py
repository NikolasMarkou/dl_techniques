"""Aggregate iter-2 sweep results into a single CSV + markdown table.

Reads every ``results/cliffordnet_downsampling_experiments_*/comparison.csv``
created at-or-after the sweep root mtime and produces:

* ``<sweep_root>/aggregated.csv`` — one row per variant-seed
* ``<sweep_root>/aggregated.md`` — group-by-variant means + std

Usage::

    .venv/bin/python tools/aggregate_iter2.py <sweep_log_root>
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from glob import glob


def _sweep_start_stamp(sweep_root: str) -> str:
    """Extract YYYYMMDD_HHMMSS timestamp from sweep root dirname."""
    base = os.path.basename(sweep_root.rstrip("/"))
    parts = base.split("_")
    return f"{parts[-2]}_{parts[-1]}" if len(parts) >= 2 else ""


def _find_runs(sweep_root: str, min_epochs: int = 50) -> list[tuple[str, dict]]:
    """Walk every per-run dir created at-or-after the sweep start.

    Filters out 3-epoch smoke runs by ``epochs_trained >= min_epochs``.
    """
    rows = []
    start_stamp = _sweep_start_stamp(sweep_root)
    for d in sorted(glob("results/cliffordnet_downsampling_experiments_*")):
        base = os.path.basename(d.rstrip("/"))
        ts = base.replace("cliffordnet_downsampling_experiments_", "")
        if ts < start_stamp:
            continue
        comp = os.path.join(d, "comparison.csv")
        if not os.path.exists(comp):
            continue
        with open(comp) as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    if int(r.get("epochs_trained", 0)) < min_epochs:
                        continue
                except ValueError:
                    continue
                seed = None
                for cfg in glob(os.path.join(d, "**/config.json"),
                                recursive=True):
                    try:
                        with open(cfg) as cf:
                            data = json.load(cf)
                        if data.get("variant") == r["variant"]:
                            seed = data.get("seed")
                            break
                    except Exception:
                        continue
                r["seed"] = seed
                r["run_root"] = d
                rows.append((d, r))
    return rows


def _write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        print(f"[aggregate] no rows to write to {path}")
        return
    fields = [
        "variant", "seed", "best_val_accuracy", "final_val_accuracy",
        "best_val_top5_accuracy", "parameters", "wall_seconds",
        "epochs_trained", "run_root", "results_dir",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[aggregate] wrote {len(rows)} rows -> {path}")


def _write_md(rows: list[dict], path: str) -> None:
    by_variant: dict[str, list[float]] = {}
    for r in rows:
        try:
            acc = float(r["best_val_accuracy"])
        except (TypeError, ValueError):
            continue
        by_variant.setdefault(r["variant"], []).append(acc)

    lines = ["# iter-2 sweep — aggregated results\n"]
    lines.append("| Variant | n | mean | std | min | max |")
    lines.append("|---------|--:|-----:|----:|----:|----:|")
    for v in sorted(by_variant):
        vals = by_variant[v]
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        lines.append(
            f"| {v} | {len(vals)} | {m:.4f} | {s:.4f} | "
            f"{min(vals):.4f} | {max(vals):.4f} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[aggregate] wrote markdown summary -> {path}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: aggregate_iter2.py <sweep_log_root>")
        return 2
    sweep_root = argv[1]
    found = _find_runs(sweep_root)
    rows = [r for _, r in found]
    _write_csv(rows, os.path.join(sweep_root, "aggregated.csv"))
    _write_md(rows, os.path.join(sweep_root, "aggregated.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
