"""Step-based training metric plots.

Emits PNG plots of loss and accuracy versus training step from the CSV
written by ``StepCheckpointCallback``. Intended to be called periodically
during training so progress is visible without waiting for epoch end.
"""

import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dl_techniques.utils.logger import logger


def plot_step_metrics(
    csv_path: str,
    out_dir: str,
    smooth_window: int = 50,
) -> None:
    """Read a step-level training CSV and emit loss / accuracy plots.

    Plots are written to ``{out_dir}/step_loss.png`` and
    ``{out_dir}/step_accuracy.png``. Any column whose name contains
    ``loss`` or ``accuracy`` (case-insensitive) is plotted, including
    validation variants. If a column is sparse (e.g. ``val_*`` only
    present at epoch boundaries) the NaN gaps are skipped.

    :param csv_path: Path to the CSV (must have a ``step`` column).
    :param out_dir: Directory to write the plots into.
    :param smooth_window: Rolling-mean window for the smoothed overlay
        (set to ``0`` or ``1`` to disable smoothing).
    """
    if not os.path.isfile(csv_path):
        return
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, OSError) as e:
        logger.warning(f"plot_step_metrics: cannot read {csv_path}: {e}")
        return
    if df.empty or "step" not in df.columns:
        return

    os.makedirs(out_dir, exist_ok=True)

    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    acc_cols = [c for c in df.columns if "accuracy" in c.lower() or c.lower().endswith("_acc")]

    _plot_group(df, loss_cols, "loss", out_dir, smooth_window)
    _plot_group(df, acc_cols, "accuracy", out_dir, smooth_window)


def _plot_group(
    df: pd.DataFrame,
    cols: list,
    kind: str,
    out_dir: str,
    smooth_window: int,
) -> None:
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = df["step"].to_numpy()
    for col in cols:
        series = df[col]
        valid = series.notna()
        if not valid.any():
            continue
        ax.plot(
            steps[valid], series[valid].to_numpy(),
            alpha=0.35, linewidth=0.8, label=f"{col} (raw)",
        )
        if smooth_window and smooth_window > 1:
            smoothed = series.rolling(smooth_window, min_periods=1).mean()
            ax.plot(
                steps[valid], smoothed[valid].to_numpy(),
                linewidth=1.6, label=f"{col} (smooth)",
            )
    ax.set_xlabel("step")
    ax.set_ylabel(kind)
    ax.set_title(f"Training {kind} vs step")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"step_{kind}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
