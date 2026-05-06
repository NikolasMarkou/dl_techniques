"""Step-based training metric plots.

Emits PNG plots of loss and accuracy versus training step from a CSV
written during training. Intended to be called periodically so progress
is visible without waiting for epoch end.

Provides:
- :func:`plot_step_metrics` — read a CSV and emit the plots.
- :class:`StepPlotCallback` — a self-contained Keras callback that
  writes step-level metrics to ``training_log.csv`` and refreshes the
  plots every N steps. Use this in scripts that don't already have a
  ``StepCheckpointCallback``.
"""

import csv
import os

import keras
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

    Plots are written to ``{out_dir}/step_loss.png`` (log y-axis) and
    ``{out_dir}/step_accuracy.png`` (linear y-axis). Any column whose
    name contains ``loss`` or ``accuracy`` (case-insensitive) is
    plotted, including validation variants. NaN gaps are skipped.

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
    acc_cols = [
        c for c in df.columns
        if "accuracy" in c.lower() or c.lower().endswith("_acc")
    ]

    _plot_group(df, loss_cols, "loss", out_dir, smooth_window, log_y=True)
    _plot_group(df, acc_cols, "accuracy", out_dir, smooth_window, log_y=False)


def _plot_group(
    df: pd.DataFrame,
    cols: list,
    kind: str,
    out_dir: str,
    smooth_window: int,
    log_y: bool,
) -> None:
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = df["step"].to_numpy()
    for col in cols:
        series = df[col]
        valid = series.notna()
        if log_y:
            valid = valid & (series > 0)
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
    ylabel = f"{kind} (log scale)" if log_y else kind
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.set_title(f"Training {kind} vs step")
    ax.grid(True, which="both" if log_y else "major", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"step_{kind}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


class StepPlotCallback(keras.callbacks.Callback):
    """Self-contained step-level CSV logger + periodic plot refresher.

    Writes a row to ``{save_dir}/training_log.csv`` every
    ``log_every_steps`` training batches and re-emits
    ``step_loss.png`` / ``step_accuracy.png`` every ``plot_every_steps``
    batches. Also refreshes plots once on train end.

    Use this in training scripts that don't already maintain a step CSV
    via a ``StepCheckpointCallback`` (e.g. fine-tuning loops).

    :param save_dir: Output directory (created if missing).
    :param log_every_steps: CSV write interval.
    :param plot_every_steps: Plot refresh interval. ``0`` disables
        periodic refresh (final plot still emitted on train end).
    :param initial_step: Starting step count (for resume).
    """

    def __init__(
        self,
        save_dir: str,
        log_every_steps: int = 100,
        plot_every_steps: int = 25000,
        initial_step: int = 0,
    ) -> None:
        super().__init__()
        os.makedirs(save_dir, exist_ok=True)
        self._save_dir = save_dir
        self._csv_path = os.path.join(save_dir, "training_log.csv")
        self._log_every_steps = log_every_steps
        self._plot_every_steps = plot_every_steps
        self._global_step = initial_step
        self._csv_file = None
        self._csv_writer = None
        logger.info(
            f"StepPlotCallback: log every {log_every_steps} steps, "
            f"plot every {plot_every_steps} steps -> {save_dir}"
        )

    def on_train_batch_end(self, batch, logs=None):
        self._global_step += 1
        if self._global_step % self._log_every_steps == 0:
            self._log_metrics(logs)
        if (
            self._plot_every_steps > 0
            and self._global_step % self._plot_every_steps == 0
        ):
            self._plot_metrics()

    def on_epoch_end(self, epoch, logs=None):
        # Capture val_* metrics that Keras only emits at epoch end.
        self._log_metrics(logs)
        self._plot_metrics()

    def on_train_end(self, logs=None):
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        self._plot_metrics()

    def _log_metrics(self, logs):
        if logs is None:
            return
        row = {"step": self._global_step, **logs}
        if self._csv_writer is None:
            self._csv_file = open(self._csv_path, "a", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(row.keys()),
            )
            if self._csv_file.tell() == 0:
                self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def _plot_metrics(self):
        try:
            plot_step_metrics(self._csv_path, self._save_dir)
        except Exception as e:
            logger.warning(f"Step plot failed at step {self._global_step}: {e}")
