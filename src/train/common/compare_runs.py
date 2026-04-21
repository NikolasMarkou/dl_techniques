"""Compare two training runs by their CSVLogger output.

Reads two ``results/<run>/training_log.csv`` + ``config.json`` pairs and emits:

- ``comparison.md`` — side-by-side best-metric table plus config diff.
- ``loss_curves.png`` — train / val loss over epochs for both runs.
- ``metric_curves.png`` — per-metric curves for every metric present in BOTH
  runs' CSVs.

CLI usage::

    python -m train.common.compare_runs <run_a_dir> <run_b_dir> \\
        [--labels A B] [--output results/compare_<ts>]

Library usage::

    from train.common.compare_runs import compare_runs
    out_dir = compare_runs("results/run_a", "results/run_b", labels=("a", "b"))
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dl_techniques.utils.logger import logger


def _load_csv(path: Path):
    """Load a training_log.csv — prefer pandas, fall back to stdlib csv."""
    try:
        import pandas as pd  # noqa
        return pd.read_csv(path), "pandas"
    except ImportError:
        import csv
        rows: List[Dict[str, float]] = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: float(v) for k, v in row.items() if v != ""})
        return rows, "csv"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning(f"Could not parse JSON at {path}; skipping.")
        return None


def _best_metric(df, column: str, mode: str = "min") -> Optional[float]:
    """Return min or max of a column across all epochs."""
    if column not in df.columns:
        return None
    series = df[column].dropna()
    if len(series) == 0:
        return None
    return float(series.min() if mode == "min" else series.max())


def _metric_columns(df) -> List[str]:
    """Return every numeric column except 'epoch'."""
    return [c for c in df.columns if c != "epoch"]


def _write_comparison_markdown(
    out_path: Path,
    df_a, df_b,
    label_a: str, label_b: str,
    config_a: Optional[Dict[str, Any]],
    config_b: Optional[Dict[str, Any]],
) -> None:
    """Write a markdown report with a side-by-side best-metric table."""
    lines: List[str] = []
    lines.append(f"# Run Comparison: {label_a} vs {label_b}\n")
    lines.append(f"- Run A: **{label_a}** ({len(df_a)} logged epochs)")
    lines.append(f"- Run B: **{label_b}** ({len(df_b)} logged epochs)\n")

    # Best metrics table — column intersection
    common = sorted(set(_metric_columns(df_a)) & set(_metric_columns(df_b)))
    if not common:
        lines.append("No overlapping metric columns between runs.\n")
    else:
        lines.append("## Best values (min for loss/error, max for accuracy/delta)\n")
        lines.append("| Metric | Run A | Run B | Delta (B − A) | Mode |")
        lines.append("|---|---:|---:|---:|:---:|")
        for col in common:
            # Heuristic: "acc" or "delta_" → max; everything else → min.
            mode = "max" if ("acc" in col or col.startswith("delta_") or col.startswith("val_delta")) else "min"
            a = _best_metric(df_a, col, mode)
            b = _best_metric(df_b, col, mode)
            if a is None or b is None:
                continue
            delta = b - a
            lines.append(f"| `{col}` | {a:.5g} | {b:.5g} | {delta:+.4g} | {mode} |")
        lines.append("")

    # Config diff (when both present)
    if config_a and config_b:
        lines.append("## Config diff (A → B)\n")
        keys = sorted(set(config_a.keys()) | set(config_b.keys()))
        any_diff = False
        lines.append("| Key | Run A | Run B |")
        lines.append("|---|---|---|")
        for k in keys:
            va = config_a.get(k, "<absent>")
            vb = config_b.get(k, "<absent>")
            if va != vb:
                lines.append(f"| `{k}` | `{va}` | `{vb}` |")
                any_diff = True
        if not any_diff:
            lines.append("| *no differences in recorded config keys* |  |  |")
        lines.append("")
    elif config_a or config_b:
        lines.append("## Config\n")
        lines.append(
            f"- Config A present: {bool(config_a)}; config B present: {bool(config_b)}.\n"
        )

    out_path.write_text("\n".join(lines))
    logger.info(f"Wrote comparison markdown to {out_path}")


def _plot_curves(
    df_a, df_b, label_a: str, label_b: str, output_dir: Path
) -> List[Path]:
    """Emit loss_curves.png and metric_curves.png. Returns list of saved paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plots.")
        return []

    saved: List[Path] = []

    # Loss curves
    loss_cols = [c for c in ("loss", "val_loss") if c in df_a.columns and c in df_b.columns]
    if loss_cols:
        n = len(loss_cols)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
        for ax, col in zip(axes[0], loss_cols):
            ax.plot(df_a["epoch"], df_a[col], label=label_a, linewidth=2)
            ax.plot(df_b["epoch"], df_b[col], label=label_b, linewidth=2)
            ax.set_xlabel("epoch")
            ax.set_ylabel(col)
            ax.set_title(col)
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        path = output_dir / "loss_curves.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

    # Metric curves — every metric column common to both runs, excluding loss
    skip = {"loss", "val_loss", "epoch"}
    metric_cols = sorted(
        (set(_metric_columns(df_a)) & set(_metric_columns(df_b))) - skip
    )
    if metric_cols:
        # Grid with max 3 cols
        ncols = min(3, len(metric_cols))
        nrows = (len(metric_cols) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
        for idx, col in enumerate(metric_cols):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]
            ax.plot(df_a["epoch"], df_a[col], label=label_a, linewidth=2)
            ax.plot(df_b["epoch"], df_b[col], label=label_b, linewidth=2)
            ax.set_xlabel("epoch")
            ax.set_ylabel(col)
            ax.set_title(col)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        # hide unused subplots
        for idx in range(len(metric_cols), nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r][c].axis("off")
        fig.tight_layout()
        path = output_dir / "metric_curves.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

    return saved


def compare_runs(
    run_a: str,
    run_b: str,
    labels: Optional[Tuple[str, str]] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """Compare two training runs and write a report directory.

    :param run_a: Path to a results directory containing ``training_log.csv``
        (and optionally ``config.json``).
    :param run_b: Same, for the second run.
    :param labels: ``(label_a, label_b)`` — defaults to each run's directory basename.
    :param output_dir: Where to write the comparison artifacts.  If ``None``,
        creates ``results/compare_<ts>`` under the current working directory.
    :returns: The output directory as a :class:`Path`.
    """
    run_a_path = Path(run_a)
    run_b_path = Path(run_b)
    csv_a = run_a_path / "training_log.csv"
    csv_b = run_b_path / "training_log.csv"
    if not csv_a.exists():
        raise FileNotFoundError(f"Missing training_log.csv in run A: {csv_a}")
    if not csv_b.exists():
        raise FileNotFoundError(f"Missing training_log.csv in run B: {csv_b}")

    df_a, backend_a = _load_csv(csv_a)
    df_b, backend_b = _load_csv(csv_b)
    if backend_a != "pandas" or backend_b != "pandas":
        raise RuntimeError(
            "compare_runs requires pandas for plotting and table alignment. "
            "Install pandas or file an issue."
        )

    label_a = (labels[0] if labels else None) or run_a_path.name
    label_b = (labels[1] if labels else None) or run_b_path.name

    config_a = _load_json(run_a_path / "config.json")
    config_b = _load_json(run_b_path / "config.json")

    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/compare_{ts}"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write_comparison_markdown(
        out / "comparison.md",
        df_a, df_b,
        label_a, label_b,
        config_a, config_b,
    )
    _plot_curves(df_a, df_b, label_a, label_b, out)

    logger.info(f"Comparison report written to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two training runs.")
    parser.add_argument("run_a", type=str, help="Run A directory (contains training_log.csv).")
    parser.add_argument("run_b", type=str, help="Run B directory.")
    parser.add_argument(
        "--labels", type=str, nargs=2, default=None,
        metavar=("LABEL_A", "LABEL_B"),
        help="Display labels for the two runs (default: directory basenames).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: results/compare_<timestamp>).",
    )
    args = parser.parse_args()
    compare_runs(args.run_a, args.run_b, tuple(args.labels) if args.labels else None, args.output)


if __name__ == "__main__":
    main()
