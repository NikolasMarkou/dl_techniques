"""
LightGlue homography evaluation (HPatches-style corner error on synthetic pairs)
====================================================================

Scores a trained `LightGlue` (loaded from the ``lightglue.keras`` or ``best_model.keras``
the trainer writes) against a mutual-nearest-neighbour descriptor baseline on the same
keypoints. Pairs are built from a folder of photographs (default COCO ``val2017``) with
the trainer's `make_pair_dataset`: a fixed seed, no photometric jitter by default,
homography ranges wider than SuperPoint's (glue-factory difficulty). A frozen SuperPoint
supplies keypoints and descriptors, matches are turned into a homography by
``cv2.findHomography`` (RANSAC), and the homography is scored by the mean displacement
of the four image corners.

Definitions transcribed from glue-factory (https://github.com/cvg/glue-factory, main):

* corner error: ``gluefactory/geometry/homography.py::homography_corner_error``; corners
  ``(0,0), (W,0), (W,H), (0,H)`` projected by the estimated and the ground-truth
  homography, mean of the four Euclidean distances (the same ``H0to1`` direction).
* AUC: ``gluefactory/utils/tools.py::cal_error_auc`` (called by ``AUCMetric`` in
  ``gluefactory/eval/utils.py::eval_poses``): errors sorted, recall curve
  ``(arange(n) + 1) / n`` with a ``(0, 0)`` point prepended, trapezoid area up to the
  threshold divided by the threshold, rounded to 4 decimals. Thresholds here are
  1, 3, 5 and 10 px (glue-factory's HPatches evaluation reports 1, 3 and 5).

One deliberate difference: glue-factory records a failed estimate as ``inf``; here it
is the configured ``--max-error`` (default 1000 px), so the MEAN error stays finite and
the strict summary json holds a number. Median and every AUC threshold below that value
are identical to the ``inf`` convention. A failure (fewer than 4 matches, a degenerate
or non-finite homography) is counted at that value, never skipped.

Match precision comes in two flavours, both reported. ``precision`` is the training
metric's definition (`KeypointMatchMetric`, D-013): a predicted pair ``(i, j)`` whose
keypoint ``i`` has label ``-2`` in image 0 or keypoint ``j`` has label ``-2`` in image 1
(ignored or padded: the labeller cannot say) is no prediction and leaves numerator and
denominator, so the number is comparable with the ``precision`` of the fit log.
``precision_strict`` counts such a pair as a false positive; it can only be lower or equal.
Recall (positives found over positive labels) is the same in both. ``--border`` is the
edge margin of the SuperPoint decode and defaults to the trainer's default, so a model
trained with another border must be evaluated with the same value.

``cv2`` (``opencv-python``) is imported lazily and only by the homography fit; without it
`main` refuses (exit 2) before writing anything.

Each run writes one directory under repo-root ``results/`` (or ``--output-dir``):
``config.json``, ``run.log`` and ``results_summary.json`` (strict JSON). A reused
``--experiment-name`` is refused.

Usage:
    python -m train.lightglue.eval_homography \
        --lightglue results/<run>/lightglue.keras \
        --superpoint-checkpoint results/<superpoint run>/final_model.keras
"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np

from dl_techniques.models.vision.keypoints.lightglue.model import LightGlue
from dl_techniques.utils.keypoint_extraction import decode_superpoint
from dl_techniques.utils.keypoint_matching import homography_matches
from dl_techniques.utils.logger import logger
from train.common import default_experiment_name, prepare_run_dir, setup_gpu
from train.common.run_artifacts import attach_run_log, refuse_existing_run, write_summary_json
from train.common.run_summary import describe_devices
from train.lightglue.data import list_images, make_pair_dataset
from train.lightglue.pipeline import load_superpoint
from train.lightglue.train_lightglue import LightGlueTrainConfig

# ---------------------------------------------------------------------

# `parents[3]` reaches the repo root from THIS file.
REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_NAME = "lightglue_eval"
DEFAULT_IMAGES_DIR = "/media/arxwn/data0_4tb/datasets/coco_2017/val2017"
DEFAULT_THRESHOLDS: Tuple[float, ...] = (1.0, 3.0, 5.0, 10.0)
DEFAULT_MAX_ERROR = 1000.0
# One source of truth: the trainer's edge margin (a dataclass field default).
DEFAULT_BORDER = LightGlueTrainConfig.border
RUN_ARTIFACTS = ("results_summary.json", "config.json", "run.log")
METHODS = ("lightglue", "mnn")

# ---------------------------------------------------------------------
# Pure numpy functions (no model, no cv2 except estimate_homography)
# ---------------------------------------------------------------------


def corner_error(
    H_est: Optional[np.ndarray],
    H_gt: np.ndarray,
    image_size: Sequence[float],
    max_error: float = DEFAULT_MAX_ERROR,
) -> float:
    """Mean displacement of the four image corners under ``H_est`` versus ``H_gt``.

    Interface contract: both homographies map image 0 to image 1 (``p1 = H @ p0``);
    ``image_size`` is ``(w, h)`` of image 0; the corners are ``(0,0), (w,0), (w,h), (0,h)``
    as in glue-factory's ``homography_corner_error``.

    :param H_est: Estimated ``(3, 3)`` homography, or ``None`` when estimation failed.
    :param H_gt: Ground-truth ``(3, 3)`` homography.
    :param image_size: ``(w, h)`` in pixels.
    :param max_error: Value returned for a missing, mis-shaped or non-finite ``H_est`` or a
        non-finite projection (a failure is counted, never skipped).
    :return: Mean corner distance in pixels.
    """
    if H_est is None:
        return float(max_error)
    est = np.asarray(H_est, dtype=np.float64)
    if est.shape != (3, 3) or not np.all(np.isfinite(est)):
        return float(max_error)
    width, height = float(image_size[0]), float(image_size[1])
    corners = np.array(
        [[0.0, 0.0, 1.0], [width, 0.0, 1.0], [width, height, 1.0], [0.0, height, 1.0]])

    def project(matrix: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            homogeneous = corners @ np.asarray(matrix, dtype=np.float64).T
            return homogeneous[:, :2] / homogeneous[:, 2:3]

    distances = np.sqrt(((project(est) - project(H_gt)) ** 2).sum(-1))
    value = float(distances.mean())
    return value if np.isfinite(value) else float(max_error)


def error_auc(errors: Sequence[float], thresholds: Sequence[float]) -> List[float]:
    """Area under the recall-versus-error curve per threshold (glue-factory ``cal_error_auc``).

    The errors are sorted, recall is ``(arange(n) + 1) / n``, a ``(0, 0)`` point is
    prepended, and the trapezoid area up to each threshold (closed with a flat segment at
    the last recall) is divided by the threshold. Rounded to 4 decimals like the source.

    :param errors: Per-pair errors (finite or ``inf``).
    :param thresholds: Positive thresholds (same unit as ``errors``).
    :return: One AUC in ``[0, 1]`` per threshold; empty ``errors`` give ``nan`` each.
    """
    values = np.asarray(errors, dtype=np.float64)
    if values.size == 0:
        return [float("nan")] * len(thresholds)
    errors_sorted = values[np.argsort(values)]
    recall = (np.arange(len(errors_sorted)) + 1) / len(errors_sorted)
    errors_sorted = np.r_[0.0, errors_sorted]
    recall = np.r_[0.0, recall]
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    aucs = []
    for threshold in thresholds:
        last_index = np.searchsorted(errors_sorted, threshold)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors_sorted[:last_index], threshold]
        aucs.append(float(np.round(trapezoid(r, x=e) / threshold, 4)))
    return aucs


def _import_cv2() -> Any:
    """Import cv2 lazily with an install hint.

    :raises ImportError: ``opencv-python`` is not installed.
    """
    try:
        import cv2
    except ImportError as error:
        raise ImportError(
            "eval_homography needs OpenCV for cv2.findHomography; install it with "
            "`pip install opencv-python-headless`"
        ) from error
    return cv2


def estimate_homography(
    kp0: np.ndarray,
    kp1: np.ndarray,
    matches: np.ndarray,
    method: str = "ransac",
    reproj_threshold: float = 3.0,
    max_iters: int = 2000,
    confidence: float = 0.995,
) -> Optional[np.ndarray]:
    """Fit the homography image 0 -> image 1 from matched keypoints with OpenCV.

    Interface contract: ``kp0`` ``(M, 2)`` and ``kp1`` ``(N, 2)`` are xy pixels,
    ``matches`` ``(S, 2)`` integer index pairs ``(i, j)``. Returns ``None`` (a failure the
    caller scores at the maximum error) when there are fewer than 4 matches, OpenCV returns
    no model, or the model is non-finite.

    :param kp0: Keypoints of image 0.
    :param kp1: Keypoints of image 1.
    :param matches: Index pairs into ``kp0`` and ``kp1``.
    :param method: ``"ransac"`` (``cv2.RANSAC``) or ``"dlt"`` (``cv2`` least squares on all
        matches).
    :param reproj_threshold: RANSAC inlier threshold in pixels.
    :param max_iters: RANSAC iteration cap.
    :param confidence: RANSAC confidence.
    :return: ``(3, 3)`` float64 homography or ``None``.
    :raises ValueError: Unknown ``method``.
    :raises ImportError: ``opencv-python`` is not installed.
    """
    if method not in ("ransac", "dlt"):
        raise ValueError(f"method must be 'ransac' or 'dlt', got {method!r}")
    matches = np.asarray(matches).reshape(-1, 2)
    if len(matches) < 4:
        return None
    cv2 = _import_cv2()
    pts0 = np.asarray(kp0, dtype=np.float64)[matches[:, 0]].reshape(-1, 1, 2)
    pts1 = np.asarray(kp1, dtype=np.float64)[matches[:, 1]].reshape(-1, 1, 2)
    try:
        if method == "ransac":
            H, _ = cv2.findHomography(
                pts0, pts1, cv2.RANSAC, float(reproj_threshold),
                maxIters=int(max_iters), confidence=float(confidence))
        else:
            H, _ = cv2.findHomography(pts0, pts1, 0)
    except cv2.error:
        return None
    if H is None or H.shape != (3, 3) or not np.all(np.isfinite(H)):
        return None
    return np.asarray(H, dtype=np.float64)


def mutual_nn_matches(
    desc0: np.ndarray,
    desc1: np.ndarray,
    mask0: Optional[np.ndarray] = None,
    mask1: Optional[np.ndarray] = None,
    ratio: Optional[float] = None,
) -> np.ndarray:
    """Mutual nearest neighbours by cosine similarity (the descriptor baseline).

    Interface contract: ``desc0`` ``(M, D)`` and ``desc1`` ``(N, D)``, masks ``(M,)`` /
    ``(N,)`` truthy for real keypoints (padded rows are never matched). A pair ``(i, j)``
    is kept when ``j`` is the best match of ``i`` and ``i`` the best of ``j``; with
    ``ratio`` it must also pass the ratio test on the Euclidean distance of the unit
    vectors (best distance below ``ratio`` times the second best, per direction).

    :param desc0: Descriptors of image 0.
    :param desc1: Descriptors of image 1.
    :param mask0: Real-keypoint mask of image 0, ``None`` for all real.
    :param mask1: Real-keypoint mask of image 1, ``None`` for all real.
    :param ratio: Optional ratio-test factor in ``(0, 1]``, ``None`` for no test.
    :return: ``(S, 2)`` int64 index pairs (possibly empty).
    """
    d0 = np.asarray(desc0, dtype=np.float64)
    d1 = np.asarray(desc1, dtype=np.float64)
    real0 = np.ones(len(d0), bool) if mask0 is None else np.asarray(mask0).astype(bool)
    real1 = np.ones(len(d1), bool) if mask1 is None else np.asarray(mask1).astype(bool)
    if not real0.any() or not real1.any():
        return np.zeros((0, 2), dtype=np.int64)
    n0 = d0 / np.maximum(np.linalg.norm(d0, axis=1, keepdims=True), 1e-12)
    n1 = d1 / np.maximum(np.linalg.norm(d1, axis=1, keepdims=True), 1e-12)
    sim = n0 @ n1.T
    sim = np.where(real0[:, None] & real1[None, :], sim, -np.inf)
    nn0 = sim.argmax(1)
    nn1 = sim.argmax(0)
    keep = real0 & (nn1[nn0] == np.arange(len(d0)))
    if ratio is not None:
        def passes(similarity: np.ndarray, best: np.ndarray) -> np.ndarray:
            if similarity.shape[1] < 2:
                return np.ones(len(best), bool)
            part = np.sort(similarity, axis=1)
            first, second = part[:, -1], part[:, -2]
            with np.errstate(invalid="ignore"):
                dist1 = np.sqrt(np.maximum(2.0 - 2.0 * first, 0.0))
                dist2 = np.sqrt(np.maximum(2.0 - 2.0 * second, 0.0))
            return np.where(np.isfinite(second), dist1 < ratio * dist2, True)
        keep &= passes(sim, nn0)
        ok1 = passes(sim.T, nn1)
        keep &= ok1[nn0]
    idx = np.nonzero(keep)[0]
    return np.stack([idx, nn0[idx]], axis=1).astype(np.int64)


def match_quality(
    matches: np.ndarray,
    labels0: np.ndarray,
    labels1: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Precision, strict precision and recall of predicted matches against the labels.

    ``labels0[i]`` is the ground-truth partner of keypoint ``i`` of image 0 (``-1``
    dustbin, ``-2`` ignored), the ``matches0`` of `homography_matches`; ``labels1`` is the
    same for image 1. A predicted pair ``(i, j)`` is correct iff ``labels0[i] == j``.

    * ``precision`` follows `KeypointMatchMetric` (D-013): a pair with ``labels0[i] == -2``
      or (when ``labels1`` is given) ``labels1[j] == -2`` is not a prediction and is
      removed from numerator and denominator.
    * ``precision_strict`` keeps every predicted pair in the denominator.
    * ``recall`` is correct pairs over positive labels.

    Each is ``nan`` for an empty denominator.

    :param matches: ``(S, 2)`` predicted index pairs.
    :param labels0: ``(M,)`` int labels of image 0.
    :param labels1: Optional ``(N,)`` int labels of image 1.
    :return: ``{"precision", "precision_strict", "recall"}``.
    """
    matches = np.asarray(matches).reshape(-1, 2)
    labels0 = np.asarray(labels0)
    correct = np.zeros(len(matches), dtype=bool)
    counted = np.zeros(len(matches), dtype=bool)
    if len(matches):
        correct = labels0[matches[:, 0]] == matches[:, 1]
        counted = labels0[matches[:, 0]] != -2
        if labels1 is not None:
            counted &= np.asarray(labels1)[matches[:, 1]] != -2
    true_positive = int(correct.sum())
    positives = int(np.sum(labels0 >= 0))
    return {
        "precision": true_positive / int(counted.sum()) if counted.any() else float("nan"),
        "precision_strict": true_positive / len(matches) if len(matches) else float("nan"),
        "recall": true_positive / positives if positives else float("nan"),
    }


def score_matches(
    kp0: np.ndarray,
    kp1: np.ndarray,
    matches: np.ndarray,
    H_gt: np.ndarray,
    image_size: Sequence[float],
    labels0: Optional[np.ndarray] = None,
    labels1: Optional[np.ndarray] = None,
    method: str = "ransac",
    reproj_threshold: float = 3.0,
    max_error: float = DEFAULT_MAX_ERROR,
) -> Dict[str, float]:
    """Score one pair's matches: fit a homography, measure the corner error and match quality.

    :param kp0: Keypoints of image 0 ``(M, 2)``.
    :param kp1: Keypoints of image 1 ``(N, 2)``.
    :param matches: ``(S, 2)`` index pairs.
    :param H_gt: Ground-truth ``(3, 3)`` homography image 0 -> image 1.
    :param image_size: ``(w, h)`` of image 0.
    :param labels0: Optional ground-truth ``matches0`` labels for precision / recall.
    :param labels1: Optional ground-truth ``matches1`` labels (ignored partners in image 1).
    :param method: Homography estimator, see :func:`estimate_homography`.
    :param reproj_threshold: RANSAC threshold in pixels.
    :param max_error: Error counted for a failed estimate.
    :return: ``{"error", "failed", "num_matches", "precision", "precision_strict",
        "recall"}`` (see :func:`match_quality`).
    """
    matches = np.asarray(matches).reshape(-1, 2)
    H = estimate_homography(kp0, kp1, matches, method=method, reproj_threshold=reproj_threshold)
    out = {
        "error": corner_error(H, H_gt, image_size, max_error),
        "failed": float(H is None),
        "num_matches": float(len(matches)),
        "precision": float("nan"),
        "precision_strict": float("nan"),
        "recall": float("nan"),
    }
    if labels0 is not None:
        out.update(match_quality(matches, labels0, labels1))
    return out


def summarize_records(
    records: Sequence[Dict[str, float]],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> Dict[str, Any]:
    """Aggregate per-pair records of one method.

    :param records: Dicts from :func:`score_matches` (optionally with ``"stop"``).
    :param thresholds: AUC thresholds in pixels.
    :return: Dict with ``pairs``, ``mean_error``, ``median_error``, ``auc`` (keyed
        ``"auc@<t>"``), ``failures``, ``mean_matches``, ``mean_precision``,
        ``mean_precision_strict``, ``mean_recall`` (over pairs where defined, ``None`` if none) and, when records carry
        ``"stop"``, ``mean_stop_layer``.
    """
    def mean_defined(key: str) -> Optional[float]:
        values = np.array([r[key] for r in records if key in r], dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(values.mean()) if values.size else None

    errors = [r["error"] for r in records]
    aucs = error_auc(errors, list(thresholds))
    summary: Dict[str, Any] = {
        "pairs": len(records),
        "mean_error": float(np.mean(errors)) if errors else None,
        "median_error": float(np.median(errors)) if errors else None,
        "auc": {f"auc@{t:g}": a for t, a in zip(thresholds, aucs)},
        "failures": int(sum(r["failed"] for r in records)),
        "mean_matches": mean_defined("num_matches"),
        "mean_precision": mean_defined("precision"),
        "mean_precision_strict": mean_defined("precision_strict"),
        "mean_recall": mean_defined("recall"),
    }
    if any("stop" in r for r in records):
        summary["mean_stop_layer"] = mean_defined("stop")
    return summary


# ---------------------------------------------------------------------
# Model glue
# ---------------------------------------------------------------------


def load_lightglue(path: str) -> LightGlue:
    """Load a LightGlue ``.keras`` file (``lightglue.keras`` or the trainer's ``best_model.keras``).

    :param path: Checkpoint file.
    :return: The model.
    :raises FileNotFoundError: ``path`` is not a file.
    :raises TypeError: The archive holds something other than a ``LightGlue``.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"LightGlue checkpoint not found: {path!r}. Train one with "
            "train.lightglue.train_lightglue (lightglue.keras or best_model.keras).")
    model = keras.saving.load_model(path, compile=False)
    if not isinstance(model, LightGlue):
        raise TypeError(f"{path!r} holds a {type(model).__name__}, not a LightGlue")
    return model


def evaluate_pairs(
    dataset: Any,
    superpoint: Any,
    lightglue: LightGlue,
    max_keypoints: int,
    nms_radius: int,
    detection_threshold: float,
    border: int = DEFAULT_BORDER,
    pos_threshold: float = 3.0,
    ransac_threshold: float = 3.0,
    max_error: float = DEFAULT_MAX_ERROR,
    mnn_ratio: Optional[float] = None,
) -> Dict[str, List[Dict[str, float]]]:
    """Run both matchers over a batch-size-1 pair dataset and score every pair.

    :param dataset: `make_pair_dataset` output with batch size 1.
    :param superpoint: Frozen float32 SuperPoint.
    :param lightglue: The matcher (its ``match()`` is used, batch 1).
    :param max_keypoints: Padded keypoints per image.
    :param nms_radius: Detection NMS radius.
    :param detection_threshold: Detection threshold.
    :param border: Edge margin without keypoints.
    :param pos_threshold: Pixel threshold of the ground-truth match labels.
    :param ransac_threshold: RANSAC inlier threshold in pixels.
    :param max_error: Error counted for a failed estimate.
    :param mnn_ratio: Optional ratio test of the baseline.
    :return: ``{"lightglue": [records], "mnn": [records]}``.
    """
    records: Dict[str, List[Dict[str, float]]] = {name: [] for name in METHODS}
    for batch in dataset:
        images = keras.ops.concatenate([batch["image0"], batch["image1"]], axis=0)
        decoded = decode_superpoint(
            superpoint(images, training=False), max_keypoints, detection_threshold,
            nms_radius, border)
        det0 = {k: v[:1] for k, v in decoded.items()}
        det1 = {k: v[1:] for k, v in decoded.items()}
        mask0 = keras.ops.cast(det0["mask"], "float32")
        mask1 = keras.ops.cast(det1["mask"], "float32")
        labels = homography_matches(
            det0["keypoints"], det1["keypoints"], mask0, mask1, batch["H0to1"],
            batch["image_size0"], batch["image_size1"],
            pos_threshold=pos_threshold, neg_threshold=pos_threshold)
        labels0 = np.asarray(labels["matches0"])[0]
        labels1 = np.asarray(labels["matches1"])[0]
        kp0 = np.asarray(det0["keypoints"])[0]
        kp1 = np.asarray(det1["keypoints"])[0]
        H_gt = np.asarray(batch["H0to1"])[0]
        size = np.asarray(batch["image_size0"])[0]

        result = lightglue.match({
            "keypoints0": det0["keypoints"], "keypoints1": det1["keypoints"],
            "descriptors0": det0["descriptors"], "descriptors1": det1["descriptors"],
            "mask0": mask0, "mask1": mask1,
            "image_size0": batch["image_size0"], "image_size1": batch["image_size1"],
        })
        lg_matches = np.asarray(result["matches"]).reshape(-1, 2)
        record = score_matches(kp0, kp1, lg_matches, H_gt, size, labels0, labels1,
                               reproj_threshold=ransac_threshold, max_error=max_error)
        record["stop"] = float(result["stop"])
        records["lightglue"].append(record)

        mnn = mutual_nn_matches(
            np.asarray(det0["descriptors"])[0], np.asarray(det1["descriptors"])[0],
            np.asarray(mask0)[0], np.asarray(mask1)[0], ratio=mnn_ratio)
        records["mnn"].append(score_matches(
            kp0, kp1, mnn, H_gt, size, labels0, labels1,
            reproj_threshold=ransac_threshold, max_error=max_error))
    return records


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Construct the evaluator's `argparse.ArgumentParser`."""
    parser = argparse.ArgumentParser(
        description="Evaluate LightGlue on fixed-seed homography pairs (corner-error AUC) "
                    "against a mutual-nearest-neighbour baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lightglue", type=str, required=True,
                        help="LightGlue .keras file (lightglue.keras or best_model.keras).")
    parser.add_argument("--superpoint-checkpoint", type=str, required=True,
                        help="Trained SuperPoint .keras file (fixes the image size).")
    parser.add_argument("--images-dir", type=str, default=DEFAULT_IMAGES_DIR,
                        help="Folder of photographs (jpg/png, not recursive).")
    parser.add_argument("--num-pairs", type=int, default=500,
                        help="Pairs to evaluate (the first N sorted images, one pair each).")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Square size, checked against the SuperPoint checkpoint.")
    parser.add_argument("--max-keypoints", type=int, default=512, help="Padded keypoints per image.")
    parser.add_argument("--nms-radius", type=int, default=4, help="Detection NMS radius in pixels.")
    parser.add_argument("--detection-threshold", type=float, default=0.005,
                        help="Minimum heatmap probability of a keypoint.")
    parser.add_argument("--border", type=int, default=DEFAULT_BORDER,
                        help="Pixels at the image edge without keypoints (use the value the "
                             "model was trained with).")
    parser.add_argument("--pos-threshold", type=float, default=3.0,
                        help="Pixel threshold of the ground-truth match labels (precision/recall).")
    parser.add_argument("--ransac-threshold", type=float, default=3.0,
                        help="RANSAC inlier threshold in pixels.")
    parser.add_argument("--max-error", type=float, default=DEFAULT_MAX_ERROR,
                        help="Corner error counted for a failed estimate (glue-factory: inf).")
    parser.add_argument("--mnn-ratio", type=float, default=None,
                        help="Optional ratio test of the baseline, in (0, 1].")
    parser.add_argument("--depth-confidence", type=float, default=None,
                        help="Override the checkpoint's early-exit confidence (-1 disables).")
    parser.add_argument("--width-confidence", type=float, default=None,
                        help="Override the checkpoint's point-pruning confidence (-1 disables).")
    parser.add_argument("--filter-threshold", type=float, default=None,
                        help="Override the checkpoint's match threshold, in [0, 1].")
    parser.add_argument("--seed", type=int, default=0, help="Pair sampling seed.")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index to use.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Base directory of the run (default: <repo>/results).")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Run directory name (default: lightglue_eval_<timestamp>).")
    return parser


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse and validate the CLI arguments (nothing is allocated before this returns).

    :param argv: Argument list; ``None`` defers to ``sys.argv[1:]``.
    :return: The parsed namespace.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    for flag in ("num_pairs", "max_keypoints"):
        if getattr(args, flag) < 1:
            parser.error(f"--{flag.replace('_', '-')} must be >= 1, got {getattr(args, flag)}")
    if args.image_size is not None and args.image_size < 1:
        parser.error(f"--image-size must be >= 1, got {args.image_size}")
    if args.nms_radius < 0:
        parser.error("--nms-radius must be >= 0")
    if args.border < 0:
        parser.error("--border must be >= 0")
    for flag in ("pos_threshold", "ransac_threshold", "max_error"):
        if getattr(args, flag) <= 0:
            parser.error(f"--{flag.replace('_', '-')} must be > 0, got {getattr(args, flag)}")
    if not 0.0 <= args.detection_threshold < 1.0:
        parser.error(f"--detection-threshold must be in [0, 1), got {args.detection_threshold}")
    if args.mnn_ratio is not None and not 0.0 < args.mnn_ratio <= 1.0:
        parser.error(f"--mnn-ratio must be in (0, 1], got {args.mnn_ratio}")
    if args.filter_threshold is not None and not 0.0 <= args.filter_threshold <= 1.0:
        parser.error(f"--filter-threshold must be in [0, 1], got {args.filter_threshold}")
    for flag in ("depth_confidence", "width_confidence"):
        value = getattr(args, flag)
        if value is not None and (value != -1 and not 0.0 < value <= 1.0):
            parser.error(f"--{flag.replace('_', '-')} must be -1 or in (0, 1], got {value}")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point of the evaluator.

    The first statement parses argv, so ``--help`` exits 0 with a ``usage:`` line before
    any GPU, dataset or model work.

    :param argv: Argument list; ``None`` defers to ``sys.argv[1:]``.
    :return: 0 on success; 2 when a preflight check (OpenCV, image folder, checkpoints,
        descriptor width) fails before any file is written.
    """
    args = parse_arguments(argv)
    setup_gpu(gpu_id=args.gpu)

    base_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "results"
    run_dir = base_dir / (args.experiment_name or default_experiment_name(EXPERIMENT_NAME))
    refuse_existing_run(run_dir, artifact_names=RUN_ARTIFACTS)

    try:
        _import_cv2()
        paths = list(list_images(args.images_dir))
        requested = None if args.image_size is None else (args.image_size, args.image_size)
        superpoint = load_superpoint(args.superpoint_checkpoint, image_size=requested)
        lightglue = load_lightglue(args.lightglue)
        if lightglue.input_dim != superpoint.descriptor_dim:
            raise ValueError(
                f"LightGlue input_dim {lightglue.input_dim} differs from the SuperPoint "
                f"descriptor_dim {superpoint.descriptor_dim}")
    except (ImportError, FileNotFoundError, ValueError, TypeError) as error:
        logger.error(f"Preflight failed, nothing written: {error}")
        return 2

    if args.depth_confidence is not None:
        lightglue.depth_confidence = args.depth_confidence
    if args.width_confidence is not None:
        lightglue.width_confidence = args.width_confidence
    if args.filter_threshold is not None:
        lightglue.filter_threshold = args.filter_threshold

    image_size = (superpoint.input_height, superpoint.input_width)
    selected = paths[: args.num_pairs]
    if len(selected) < args.num_pairs:
        logger.warning(f"only {len(selected)} images for --num-pairs {args.num_pairs}")

    prepare_run_dir(vars(args), output_dir=run_dir)
    with attach_run_log(run_dir):
        logger.info(f"Run directory: {run_dir}")
        logger.info(f"Config: {vars(args)}")
        dataset = make_pair_dataset(
            selected, image_size, 1, seed=args.seed, photometric_jitter=False,
            shuffle=False, drop_remainder=True, ignore_errors=False)
        start = time.time()
        records = evaluate_pairs(
            dataset, superpoint, lightglue, args.max_keypoints, args.nms_radius,
            args.detection_threshold, border=args.border, pos_threshold=args.pos_threshold,
            ransac_threshold=args.ransac_threshold, max_error=args.max_error,
            mnn_ratio=args.mnn_ratio)
        seconds = time.time() - start

        methods = {name: summarize_records(recs) for name, recs in records.items()}
        for name, stats in methods.items():
            logger.info(f"{name}: {stats}")
        summary: Dict[str, Any] = {
            "model": EXPERIMENT_NAME,
            "lightglue": args.lightglue,
            "superpoint_checkpoint": args.superpoint_checkpoint,
            "images_dir": args.images_dir,
            "image_size": list(image_size),
            "pairs": len(records["lightglue"]),
            "seed": args.seed,
            "border": args.border,
            "ransac_threshold": args.ransac_threshold,
            "max_error": args.max_error,
            "thresholds": list(DEFAULT_THRESHOLDS),
            "adaptive": {
                "depth_confidence": float(lightglue.depth_confidence),
                "width_confidence": float(lightglue.width_confidence),
                "filter_threshold": float(lightglue.filter_threshold),
            },
            "methods": methods,
            "devices": describe_devices(),
            "eval_seconds": seconds,
        }
        write_summary_json(run_dir, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------
