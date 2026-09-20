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
- :func:`read_data_free_analysis_status` -- ``-> {"status", "analyzers", "error", "path"}``
  for a weights / spectral analysis (which leaves ``model_metrics`` empty, so
  :func:`read_analysis_status` calls it 'missing'); never raises for a file problem.
- :func:`run_data_free_analysis` -- ``(model, sample, labels, history, model_name, run_dir,
  enabled) -> {"status", "analyzers", "error", "path", "seconds"}`` (or the skipped block):
  runs the weights + spectral analysis and reads it back; never raises.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np
import tensorflow as tf
from threadpoolctl import threadpool_limits

from dl_techniques.analyzer import AnalysisConfig
from dl_techniques.utils.logger import logger
from train.common.callbacks import best_checkpoint_path, resolve_monitor_mode
from train.common.classification_viz import (
    plot_calibration,
    plot_confident_errors,
    plot_confusion_matrix,
    plot_per_class_metrics,
)
from train.common.config_io import json_numpy_default
from train.common.evaluation import run_model_analysis


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


# DECISION plan-2026-09-19T224205-49c8bf80/D-009: a DATA-FREE analysis (weights + spectral only,
# calibration / information flow / training dynamics off) leaves ``model_metrics`` EMPTY, so
# ``read_analysis_status`` reports it as 'missing' with ``KeyError: '<model>'`` although the
# analysis succeeded (MEASURED on a real ``run_model_analysis`` file; guard:
# ``test_data_free_status_reads_a_real_analyzer_file_that_read_analysis_status_calls_missing``).
# Do NOT "fix" this by loosening ``read_analysis_status`` (the ConvNeXt trainer relies on its
# strict ``model_metrics`` reading) and do NOT record 'ok' without reading the file.
DATA_FREE_ANALYZER_SECTIONS: Dict[str, str] = {
    "weights": "weight_stats",
    "spectral": "spectral_summary_per_model",
}


def read_data_free_analysis_status(
        run_dir: Path, model_name: str, expected: Sequence[str] = tuple(DATA_FREE_ANALYZER_SECTIONS),
) -> Dict[str, Any]:
    """Read a weights / spectral analysis back from ``model_analysis/analysis_results.json``.

    ``run_model_analysis`` swallows every exception and logs "completed successfully"
    regardless, so what is on disk is the only source of truth. A data-free analysis fills
    ``weight_stats[model_name]`` and ``spectral_summary_per_model[model_name]`` and leaves
    ``model_metrics`` empty, which is why :func:`read_analysis_status` cannot read it.
    Never raises for a file problem (a missing, empty or corrupt file is a returned status).

    Args:
        run_dir: The run directory holding ``model_analysis/``.
        model_name: The key the analysis was run under.
        expected: Analyzers that were requested, names from
            :data:`DATA_FREE_ANALYZER_SECTIONS` (``"weights"``, ``"spectral"``).

    Returns:
        ``{"status", "analyzers", "error", "path"}``. ``analyzers`` lists which of
        ``expected`` wrote a non-empty section for ``model_name`` (in ``expected`` order).
        ``status`` is ``"ok"`` when every expected analyzer wrote, ``"partial"`` when only
        some did, ``"missing"`` when the file is absent or nothing expected wrote,
        ``"unreadable"`` when the file is not a JSON object. ``error`` names the reason
        for every status but ``"ok"``.

    Raises:
        ValueError: If ``expected`` is empty or names an unknown analyzer (a caller bug).
    """
    path = run_dir / "model_analysis" / "analysis_results.json"
    out: Dict[str, Any] = {"status": "missing", "analyzers": [], "error": None, "path": str(path)}
    unknown = [name for name in expected if name not in DATA_FREE_ANALYZER_SECTIONS]
    if unknown or not expected:
        raise ValueError(f"expected must name analyzers from {sorted(DATA_FREE_ANALYZER_SECTIONS)}, got {list(expected)}")
    if not path.is_file():
        out["error"] = f"{path} does not exist"
        return out
    try:
        with open(path) as f:
            results = json.load(f)
        if not isinstance(results, dict):
            raise ValueError(f"top level is {type(results).__name__}, not an object")
    except Exception as e:  # noqa: BLE001 - reported, not fatal
        logger.warning(f"Could not read the analysis from {path}: {e}")
        out["status"] = "unreadable"
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    for name in expected:
        section = results.get(DATA_FREE_ANALYZER_SECTIONS[name])
        if isinstance(section, dict) and section.get(model_name):
            out["analyzers"].append(name)
    absent = [name for name in expected if name not in out["analyzers"]]
    if not absent:
        out["status"] = "ok"
    else:
        out["status"] = "partial" if out["analyzers"] else "missing"
        out["error"] = f"no results for {absent} under model {model_name!r} in {path.name}"
    return out


def skipped_analysis_status() -> Dict[str, Any]:
    """The :func:`read_analysis_status` schema for a run that did not call the analyzer."""
    return {"status": "skipped", "loss": None, "accuracy": None, "error": None, "path": None}


def data_free_analysis_config() -> AnalysisConfig:
    """Weights and spectral analyses only.

    Calibration, information flow and training dynamics read per-image labels and class
    probabilities of a classifier; on a dense output (a denoised image, per-pixel logits) they
    would measure nothing meaningful (D-005). One place for the choice, used by the
    end-of-run analysis of both ConvUNeXt trainers and the bfunet per-epoch ``--analyzer``.
    """
    return AnalysisConfig(
        analyze_weights=True, analyze_spectral=True, analyze_calibration=False,
        analyze_information_flow=False, analyze_training_dynamics=False, verbose=False,
    )


def run_data_free_analysis(
        model: keras.Model, sample: np.ndarray, labels: np.ndarray,
        history: "keras.callbacks.History", model_name: str, run_dir: Path, enabled: bool = True,
) -> Dict[str, Any]:
    """Run the end-of-run weights + spectral analysis and read back what reached the disk; never raises.

    ``run_model_analysis`` swallows its own exceptions and logs success regardless, so the
    status comes from :func:`read_data_free_analysis_status`, not from its return value.
    Anything raised on the way is recorded as status ``"error"``: the analysis cannot fail a
    finished run.

    Args:
        model: The fitted in-memory model (the LAST epoch's weights).
        sample: Input array handed to the analyzer as ``x`` (these analyses read weights, not data).
        labels: Its ``y`` (any array of the same length; unused by weights + spectral).
        history: The ``History`` of ``fit``.
        model_name: The key the analysis is stored and read back under.
        run_dir: The run directory; results land in ``<run_dir>/model_analysis/``.
        enabled: ``False`` returns :func:`skipped_analysis_status` without touching the disk.

    Returns:
        :func:`skipped_analysis_status` when disabled, else ``{"status", "analyzers", "error",
        "path", "seconds"}`` with ``status`` ``ok`` / ``partial`` / ``missing`` /
        ``unreadable`` (see :func:`read_data_free_analysis_status`) or ``error``; after an ``ok``
        analysis the library's empty ``summary_dashboard.png`` is deleted and the block gains
        ``"removed": ["summary_dashboard.png"]`` (the key is absent when nothing was removed).
    """
    if not enabled:
        logger.info("Analyzer skipped (--no-model-analysis)")
        return skipped_analysis_status()
    started = time.perf_counter()
    try:
        # DECISION plan-2026-09-19T224205-49c8bf80/D-046: the spectral fits run 12 BLAS threads that
        # oversubscribe the host (14.7-21.0 s against 6.1-6.6 s at one thread, same model and load).
        # Do NOT drop the limit, and do NOT set it process-wide: it is scoped to this one call.
        with threadpool_limits(limits=1):
            results = run_model_analysis(
                model, (sample, labels), history, model_name, str(run_dir), data_free_analysis_config())
        block = read_data_free_analysis_status(run_dir, model_name)
        if results is None and block["status"] != "ok":
            block["error"] = f"run_model_analysis failed (its exception is in run.log); {block['error']}"
        # DECISION plan-2026-09-19T224205-49c8bf80/D-027: the library always writes
        # ``summary_dashboard.png`` and, with weights + spectral only, all four panels read "No ...
        # data available" (194,788 B in every run). Delete THAT file, ours, in this run's own
        # ``model_analysis/``, after an "ok" read-back, and say so in ``removed``, but (D-040) only
        # while none of the four sections its panels read has data. Do NOT turn the analyzer's own
        # dashboard off (library layout) and do NOT delete by pattern or by file name alone.
        empty_dashboard = run_dir / "model_analysis" / "summary_dashboard.png"
        if block["status"] == "ok" and empty_dashboard.is_file() and results is not None and not any(
                (results.model_metrics, results.calibration_metrics, results.confidence_metrics, results.weight_pca)):
            empty_dashboard.unlink()
            block["removed"] = [empty_dashboard.name]
    except Exception as e:  # noqa: BLE001 - the analysis must not fail a finished run
        logger.warning(f"Analyzer raised: {type(e).__name__}: {e}")
        block = {"status": "error", "analyzers": [], "error": f"{type(e).__name__}: {e}", "path": None}
    block["seconds"] = time.perf_counter() - started
    logger.info(f"Analyzer status (read back from disk): {block['status']} in {block['seconds']:.1f}s")
    return block


# The config attributes every ConvUNeXt-side ``results_summary.json`` records under their own names.
SUMMARY_CONFIG_KEYS = (
    "experiment_name", "variant", "learning_rate", "warmup_epochs", "weight_decay", "batch_size", "seed")


def summary_head(config: Any, run_dir: Path, *, params: int, steps_per_epoch: int,
                 devices: Dict[str, Any]) -> Dict[str, Any]:
    """The summary keys the bfunet trainers and the segmenter share: ``SUMMARY_CONFIG_KEYS`` read
    off ``config`` (a missing attribute raises ``AttributeError``), ``epochs`` as ``epochs_requested``,
    the three ``devices`` keys of :func:`describe_devices`, the sizes, and the [0, 1] ``data_range``."""
    return {
        "run_dir": str(run_dir), "params": int(params), "steps_per_epoch": int(steps_per_epoch),
        "epochs_requested": config.epochs, "data_range": [0.0, 1.0],
        **{key: getattr(config, key) for key in SUMMARY_CONFIG_KEYS},
        **{key: devices[key] for key in ("gpu_name", "tf_visible_devices", "cuda_visible_devices")},
    }


def describe_devices() -> Dict[str, Any]:
    """Which GPU(s) TensorFlow sees in this process, for the run log and the summary.

    The first call enumerates devices, after which a changed ``CUDA_VISIBLE_DEVICES``
    no longer selects a different one; ``gpu_name`` is what TF actually sees, while
    ``cuda_visible_devices`` is only the environment value at call time.

    Returns:
        ``{"cuda_visible_devices", "tf_visible_devices", "gpu_names", "gpu_name"}``;
        ``gpu_name`` is the first visible GPU's name or ``None`` on a CPU-only process.
    """
    gpus = tf.config.list_physical_devices("GPU")
    names = [
        str(tf.config.experimental.get_device_details(gpu).get("device_name", gpu.name))
        for gpu in gpus
    ]
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf_visible_devices": [gpu.name for gpu in gpus],
        "gpu_names": names,
        "gpu_name": names[0] if names else None,
    }
