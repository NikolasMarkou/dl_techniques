"""The end-of-run figures and report of a segmentation run.

The per-epoch grid is a Keras callback (``train.convunext.segmentation_viz.SegmentationGridCallback``,
which the task attaches through its ``figure_callbacks`` hook); this module writes what is drawn once,
after training, from the BEST weights: the predictions of the first test pairs, the per-class scores and the
report of the pooled test confusion, and the mIoU curve of the whole run. It is the segmentation sibling of
``train.common.run_summary.write_classification_figures`` and returns a record of the same shape.

Interface contract
------------------
``write_segmentation_figures(vis_dir, images, truths, predictions, confusion, history, best_epoch, class_names,
ignore_index, monitor)``
    :param vis_dir: the existing ``visualizations`` directory.
    :param images: ``(N, H, W, 3)`` float images in [0, 1] (the rows to draw).
    :param truths: ``(N, H, W)`` class ids of those images, void included.
    :param predictions: ``(N, H, W)`` argmax class ids of the best weights on those images.
    :param confusion: ``(C, C)`` pooled pixel counts over the whole test split, ``[true, predicted]``, void
        left out (``segmentation_run.confusion_from_dataset``).
    :param history: ``{"miou": [...], "val_miou": [...], ...}`` per epoch.
    :param best_epoch: the 1-based epoch the run kept.
    :param class_names: one name per class.
    :param ignore_index: the void value of ``truths``, or ``None``.
    :param monitor: the checkpoint metric, printed in the curve's label.
    :return: ``{"files": [names written, in order], "failed": [names that raised], "skipped": False}``. The
        files are ``predictions_best.png``, ``per_class_scores.png``, ``miou_curve.png`` and
        ``segmentation_report.json``. Each is independent: one that raises is logged as a warning and
        listed in ``failed``, and the others still render. The per-epoch grids are written by the callback
        and are not listed.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from dl_techniques.utils.logger import logger
from train.convunext import segmentation_viz as viz


def write_segmentation_figures(
    vis_dir: Path,
    images: np.ndarray,
    truths: np.ndarray,
    predictions: np.ndarray,
    confusion: np.ndarray,
    history: Dict[str, Sequence[float]],
    best_epoch: int,
    class_names: List[str],
    ignore_index: Optional[int],
    monitor: str,
) -> Dict[str, Any]:
    """Write the four end-of-run files; see the module contract."""
    record: Dict[str, Any] = {"files": [], "failed": [], "skipped": False}
    palette = viz.extended_palette(len(class_names))

    def attempt(name: str, draw: Callable[[], Any]) -> None:
        try:
            draw()
            if (vis_dir / name).is_file():
                record["files"].append(name)
        except Exception as exc:  # noqa: BLE001 - a figure must not fail the run
            logger.warning(f"Visualization {name} failed: {type(exc).__name__}: {exc}")
            record["failed"].append(name)

    attempt("predictions_best.png", lambda: viz.plot_segmentation_grid(
        images, truths, predictions, class_names, vis_dir / "predictions_best.png",
        title=f"best weights (epoch {best_epoch}, by {monitor}), first {len(images)} test pairs",
        palette=palette, ignore_index=ignore_index))
    attempt("per_class_scores.png", lambda: viz.plot_per_class_scores(
        confusion, class_names, vis_dir / "per_class_scores.png"))
    attempt("miou_curve.png", lambda: viz.plot_miou_curve(
        history, best_epoch, vis_dir / "miou_curve.png", monitor=monitor))
    attempt("segmentation_report.json", lambda: viz.write_segmentation_report(
        confusion, class_names, vis_dir / "segmentation_report.json"))
    return record
