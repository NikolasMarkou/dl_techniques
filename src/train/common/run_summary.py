"""Post-fit helpers shared by the classification trainers (PowerMLP, ConvNeXt).

Both trainers read the same things off a finished ``fit``: which epoch was best,
whether the history is finite, when ``ReduceLROnPlateau`` acted, what the reloaded
``best_model.keras`` scores, which figures rendered, and what the analyzer really wrote
to disk. These used to be trainer-local copies (D-015); each is defined once here and the
differences between the trainers are parameters (``monitor``, ``evaluate``, the
misclassification-grid arguments), not forks.

Interface contracts:

- :func:`best_epoch` -- ``(history, monitor) -> int`` (1-based); ``ValueError`` on a
  non-finite monitored history.
- :func:`non_finite_metrics` -- ``(history, monitor) -> list[str]``; never raises.
- :func:`lr_reduction_epochs` -- ``(lrs) -> list[int]`` (1-based); never raises.
- :func:`load_best_metrics` -- ``(run_dir, evaluate) -> (metrics | None, error | None)``;
  never raises.
- :func:`write_classification_figures` -- ``-> {"files", "ece", "failed"}``; never raises
  for a figure or report failure.
- :func:`read_analysis_status` -- ``-> {"status", "loss", "accuracy", "error", "path"}``;
  never raises. :func:`skipped_analysis_status` is the same schema for a run that did not
  call the analyzer.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np

from dl_techniques.utils.logger import logger
from train.common.callbacks import best_checkpoint_path, resolve_monitor_mode
from train.common.classification_viz import (
    plot_calibration,
    plot_confident_errors,
    plot_confusion_matrix,
    plot_per_class_metrics,
)
from train.common.config_io import json_numpy_default


def best_epoch(history: Dict[str, List[float]], monitor: str) -> int:
    """1-based epoch that is best under ``monitor`` (direction from ``resolve_monitor_mode``).

    Raises:
        ValueError: If any monitored value is NaN or infinite. ``np.argmin`` of a
            list holding a NaN returns the NaN's index, so a diverged run would
            otherwise report its NaN epoch as the best one.
    """
    values = np.asarray(history[monitor], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"cannot pick a best epoch: {monitor} has non-finite values {values.tolist()}")
    mode = resolve_monitor_mode(monitor)
    return int(np.argmin(values) if mode == "min" else np.argmax(values)) + 1


def non_finite_metrics(history: Dict[str, List[float]], monitor: str) -> List[str]:
    """Names among ``loss`` and ``monitor`` that are empty or hold a NaN / infinity."""
    return [
        key for key in dict.fromkeys(("loss", monitor))
        if not history.get(key) or not np.all(np.isfinite(history[key]))
    ]


def lr_reduction_epochs(lrs: Sequence[float]) -> List[int]:
    """1-based epochs whose learning rate is lower than the previous epoch's.

    ``lrs`` is the ``lr`` history (``LearningRateLogger`` writes the rate used
    DURING each epoch), so an entry ``e`` means ``ReduceLROnPlateau`` acted
    after epoch ``e - 1`` and epoch ``e`` was the first one trained at the lower
    rate. Keras prints that message to stdout, not to the logger, so this is how
    the run's own log and summary learn about it.

    Args:
        lrs: Per-epoch learning rates, epoch 1 first.

    Returns:
        The reduction epochs in ascending order (empty for a constant rate).
    """
    return [i + 1 for i in range(1, len(lrs)) if lrs[i] < lrs[i - 1]]


def load_best_metrics(
        run_dir: Path, evaluate: Callable[[keras.Model], Dict[str, float]]
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """Evaluate the reloaded ``best_model.keras`` with the trainer's own ``evaluate``.

    Args:
        run_dir: The run directory holding ``best_model.keras``.
        evaluate: ``model -> {metric: float}``; the trainer binds its data and batch size.

    Returns:
        ``(metrics, None)`` on success, ``(None, error_text)`` if the checkpoint
        is missing or does not load.
    """
    path = best_checkpoint_path(str(run_dir))
    try:
        return evaluate(keras.models.load_model(path)), None
    except Exception as e:  # noqa: BLE001 - reported, not fatal
        logger.warning(f"Could not evaluate best checkpoint {path}: {e}")
        return None, f"{type(e).__name__}: {e}"


def write_classification_figures(
        vis_dir: Path,
        x_test: np.ndarray,
        y_test: np.ndarray,
        probs: np.ndarray,
        class_names: List[str],
        image_shape: Optional[Tuple[int, ...]] = None,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Render the four end-of-run figures and the classification report.

    ``probs`` MUST be probabilities (softmax of the logits): calibration and confident
    errors read them as such. Each figure is independent: a failure logs a warning and
    the others still render.

    Args:
        vis_dir: Existing output directory.
        x_test: Test inputs, flat ``(N, H*W*C)`` (then ``image_shape`` is required) or
            images ``(N, H, W, C)``.
        y_test: Integer labels ``(N,)``.
        probs: Class probabilities ``(N, C)``.
        class_names: One name per class.
        image_shape, mean, std: Forwarded to ``plot_confident_errors`` (un-standardizing
            and reshaping the misclassification grid).

    Returns:
        ``{"files": [names of the post-fit figure and report files that exist on disk;
        the dashboard is written by the training callback and is not listed], "ece": float | None,
        "failed": [names]}``. A figure that raised is in ``failed``; one that legitimately
        wrote nothing (a perfect classifier has no misclassification grid) is in neither.
    """
    y_pred = np.argmax(probs, axis=-1)
    out: Dict[str, Any] = {"files": [], "ece": None, "failed": []}

    def _attempt(name: str, fn: Callable[[], Any]) -> Any:
        try:
            result = fn()
            if (vis_dir / name).is_file():
                out["files"].append(name)
            return result
        except Exception as e:  # noqa: BLE001 - a figure must not fail the run
            logger.warning(f"Visualization {name} failed: {e}")
            out["failed"].append(name)
            return None

    _attempt("confusion_matrix.png", lambda: plot_confusion_matrix(
        y_test, y_pred, class_names, vis_dir / "confusion_matrix.png"))
    report = _attempt("per_class_metrics.png", lambda: plot_per_class_metrics(
        y_test, y_pred, class_names, vis_dir / "per_class_metrics.png"))
    out["ece"] = _attempt("confidence_calibration.png", lambda: plot_calibration(
        y_test, probs, vis_dir / "confidence_calibration.png"))
    _attempt("misclassifications.png", lambda: plot_confident_errors(
        x_test, y_test, probs, vis_dir / "misclassifications.png",
        image_shape, mean, std, class_names=class_names))

    if report is not None:
        try:
            with open(vis_dir / "classification_report.json", "w") as f:
                json.dump(report, f, indent=2, default=json_numpy_default)
            out["files"].append("classification_report.json")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"classification_report.json failed: {e}")
            out["failed"].append("classification_report.json")
    return out


def read_analysis_status(run_dir: Path, model_name: str) -> Dict[str, Any]:
    """Read ``model_analysis/analysis_results.json`` back from disk.

    ``run_model_analysis`` swallows evaluation errors and logs "completed
    successfully" regardless, so the file is the only trustworthy source.

    Returns:
        ``{"status", "loss", "accuracy", "error", "path"}`` for ``model_name``;
        ``{"status": "missing", ...}`` if the file, the key or the JSON is unusable.
    """
    path = run_dir / "model_analysis" / "analysis_results.json"
    missing: Dict[str, Any] = {"status": "missing", "loss": None, "accuracy": None,
                               "error": None, "path": str(path)}
    try:
        with open(path) as f:
            metrics = json.load(f)["model_metrics"][model_name]
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not read analyzer status from {path}: {e}")
        missing["error"] = f"{type(e).__name__}: {e}"
        return missing
    return {
        "status": metrics.get("status", "missing"),
        "loss": metrics.get("loss"),
        "accuracy": metrics.get("accuracy"),
        "error": metrics.get("error"),
        "path": str(path),
    }


def skipped_analysis_status() -> Dict[str, Any]:
    """The :func:`read_analysis_status` schema for a run that did not call the analyzer."""
    return {"status": "skipped", "loss": None, "accuracy": None, "error": None, "path": None}
