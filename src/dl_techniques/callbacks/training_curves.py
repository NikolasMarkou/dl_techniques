"""Training curves callback — save per-epoch metric curves as PNGs.

After every ``frequency`` epochs, scans ``logs`` (the dict Keras passes into
:meth:`on_epoch_end`) across the full training history, groups metrics into
- total / task losses,
- classification metrics,
- segmentation metrics,
- depth metrics,
- anything else,

and writes one PNG per group to ``{output_dir}/training_curves/``.  Files are
overwritten each time so the latest snapshot is always at a stable path
(e.g. ``training_curves/loss.png``, ``training_curves/classification.png``).

Model-agnostic — reacts only to the keys present in ``logs`` and their
``val_`` counterparts.  Safe to use alongside ``CSVLogger`` — both observe
the same dict.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional

import keras

from dl_techniques.utils.logger import logger


def _import_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _group_metrics(keys: List[str]) -> Dict[str, List[str]]:
    """Bucket metric keys into semantic groups based on name prefixes.

    Keys with a ``val_`` twin are detected automatically and plotted together.
    """
    train_keys = [k for k in keys if not k.startswith("val_") and k != "epoch"]
    groups: Dict[str, List[str]] = {
        "loss": [],
        "classification": [],
        "segmentation": [],
        "depth": [],
        "other": [],
    }
    for k in train_keys:
        if k == "loss" or k.endswith("_loss"):
            # total / per-task loss → "loss" group, except per-head metric-style
            # suffixes keep with their head.
            if "classification" in k:
                groups["classification"].append(k)
            elif "segmentation" in k:
                groups["segmentation"].append(k)
            elif "depth" in k:
                groups["depth"].append(k)
            else:
                groups["loss"].append(k)
        elif "classification" in k or any(
            tok in k for tok in ("macro_f1", "auc", "brier", "precision", "recall")
        ):
            groups["classification"].append(k)
        elif "segmentation" in k or "pix_acc" in k:
            groups["segmentation"].append(k)
        elif any(tok in k for tok in ("abs_rel", "sq_rel", "rmse", "delta_")):
            groups["depth"].append(k)
        else:
            groups["other"].append(k)

    return {name: cols for name, cols in groups.items() if cols}


class TrainingCurvesCallback(keras.callbacks.Callback):
    """Save metric/loss curves as PNGs every ``frequency`` epochs.

    :param output_dir: Directory to write PNGs into.  Created if missing.
    :param frequency: Save every N epochs.  Defaults to 1.
    :param file_prefix: PNG filename prefix (useful when running multiple
        concurrent callbacks).  Defaults to empty string.

    Emits one PNG per non-empty metric group:
      - ``loss.png`` — total & un-grouped losses
      - ``classification.png`` — classification head losses + metrics
      - ``segmentation.png`` — segmentation head losses + metrics
      - ``depth.png`` — depth-specific metrics
      - ``other.png`` — anything else
    """

    def __init__(
        self,
        output_dir: str,
        frequency: int = 1,
        file_prefix: str = "",
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frequency = int(frequency)
        self.file_prefix = file_prefix
        # Accumulated per-epoch history.  Keras doesn't hand us the full
        # history via logs; we build it up ourselves.
        self._history: List[Dict[str, float]] = []
        self._epochs: List[int] = []
        logger.info(
            f"TrainingCurvesCallback: every {frequency} epochs → {self.output_dir}"
        )

    def on_epoch_end(
        self, epoch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        if logs is None:
            return
        # Snapshot (shallow) so later mutations by other callbacks don't
        # retroactively affect our history.
        self._history.append({k: float(v) for k, v in logs.items()})
        self._epochs.append(epoch + 1)  # 1-indexed for display

        if (epoch + 1) % self.frequency != 0:
            return
        try:
            self._save_groups()
        except Exception as e:
            # Never crash training because of a visualization failure.
            logger.warning(f"TrainingCurvesCallback plot failed: {e}")
        finally:
            gc.collect()

    def _save_groups(self) -> None:
        plt = _import_mpl()
        # All keys ever seen (train and val) across history.
        all_keys = sorted({k for row in self._history for k in row.keys()})
        groups = _group_metrics(all_keys)

        for group_name, columns in groups.items():
            self._plot_group(plt, group_name, columns)

    def _plot_group(self, plt, group_name: str, columns: List[str]) -> None:
        if not columns:
            return
        ncols = min(3, len(columns))
        nrows = (len(columns) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False
        )
        for idx, col in enumerate(columns):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]
            train_y = [row.get(col) for row in self._history]
            val_col = f"val_{col}"
            val_y = [row.get(val_col) for row in self._history]

            xs = self._epochs
            # Mask None values so matplotlib doesn't crash.
            tx, ty = zip(*[(x, y) for x, y in zip(xs, train_y) if y is not None]) if any(
                v is not None for v in train_y
            ) else ([], [])
            vx, vy = zip(*[(x, y) for x, y in zip(xs, val_y) if y is not None]) if any(
                v is not None for v in val_y
            ) else ([], [])

            if tx:
                ax.plot(tx, ty, "-o", label="train", markersize=3, linewidth=1.5)
            if vx:
                ax.plot(vx, vy, "-s", label="val", markersize=3, linewidth=1.5)

            ax.set_xlabel("epoch")
            ax.set_ylabel(col)
            ax.set_title(col, fontsize=10)
            ax.grid(True, alpha=0.3)
            if tx or vx:
                ax.legend(fontsize=8)
        # hide unused subplots
        for idx in range(len(columns), nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r][c].axis("off")

        fig.suptitle(
            f"{group_name.capitalize()} curves (epoch {self._epochs[-1]})",
            fontsize=12,
        )
        fig.tight_layout()
        path = self.output_dir / f"{self.file_prefix}{group_name}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        logger.info(f"Saved {group_name} curves → {path}")
