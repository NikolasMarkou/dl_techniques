"""Conformal prediction sets for a trained classifier, shared by the ImageNet trainers.

After training, :func:`run_conformal` runs the model once over the labelled validation
set, measures on random 50/50 splits whether the calibrated sets reach the target
coverage, fits the final threshold on all of the validation set, and exports the
classifier with a :class:`ConformalPredictionSet` on top (``conformal_model.keras``,
outputs ``(logits, probs, in_set)``) next to a ``conformal.json`` record. The score is
the deterministic APS rule, the one the exported layer computes, so the reported
numbers describe the exported model. The math is in
``dl_techniques.utils.conformal_classification``, including the caveat that the
validation set must not also have chosen the model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import keras
import numpy as np

from dl_techniques.layers.statistics.conformal import (
    ConformalPredictionSet,
    attach_conformal_sets,
)
from dl_techniques.utils.conformal_classification import (
    calibrate_aps,
    evaluate_sets,
    predict_sets,
    repeated_split_coverage,
)
from dl_techniques.utils.logger import logger

CONFORMAL_JSON_NAME = "conformal.json"
CONFORMAL_MODEL_NAME = "conformal_model.keras"
DEFAULT_ALPHA = 0.1
SPLIT_REPEATS = 20

CAVEAT = (
    "Coverage is marginal (on average over the validation distribution), not "
    "per-image or per-class. It holds when the calibration and deployment data are "
    "exchangeable and the model was not chosen on the calibration data; a "
    "checkpoint picked on this validation set weakens it. The split statistics "
    "calibrate on half of the validation set and measure on the other half; the "
    "exported threshold is fit on all of it."
)


def _alpha(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"alpha must be a number, got {text!r}")
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError(f"alpha must be in (0, 1), got {value}")
    return value


def add_conformal_arguments(parser: argparse.ArgumentParser, note: str = "") -> None:
    """Add ``--conformal-alpha`` and ``--no-conformal`` to ``parser``.

    ``note`` prefixes both help strings, e.g. ``"Used only when --dataset imagenet: "``.
    """
    parser.add_argument(
        "--conformal-alpha", type=_alpha, default=DEFAULT_ALPHA, dest="conformal_alpha",
        help=f"{note}miscoverage level of the conformal prediction sets fitted on the "
             f"validation set after training (default {DEFAULT_ALPHA}: sets that "
             "contain the true class about 90%% of the time)",
    )
    parser.add_argument(
        "--no-conformal", action="store_true", dest="no_conformal",
        help=f"{note}skip the post-training conformal calibration and export",
    )


def collect_logits(model: keras.Model, dataset: Any) -> Tuple[np.ndarray, np.ndarray]:
    """One ordered pass of ``model`` over a batched ``(x, y)`` dataset.

    :return: ``(logits (N, C) float32, labels (N,) int64)``, in dataset order.
    """
    logits, labels = [], []
    for x, y in dataset:
        out = model(x, training=False)
        if isinstance(out, (list, tuple, dict)):
            raise ValueError(
                "collect_logits needs a single-output model; reduce a "
                "deep-supervised model to its primary output first"
            )
        logits.append(keras.ops.convert_to_numpy(out).astype(np.float32))
        labels.append(keras.ops.convert_to_numpy(y).astype(np.int64).reshape(-1))
    if not logits:
        raise ValueError("the dataset is empty")
    return np.concatenate(logits), np.concatenate(labels)


def run_conformal(
        model: keras.Model,
        dataset: Any,
        alpha: float,
        seed: int,
        out_dir: Path,
        input_shape: Tuple[int, ...],
        model_name: str,
        weights: str,
) -> Dict[str, Any]:
    """Calibrate, check on random splits, export the model and write the record.

    :param model: Trained single-output classifier that emits logits.
    :param dataset: Batched, ordered ``(x, y)`` validation dataset (all of it).
    :param alpha: Miscoverage level in ``(0, 1)``.
    :param seed: Seed of the random splits.
    :param out_dir: Run directory; ``conformal.json`` and ``conformal_model.keras`` go here.
    :param input_shape: Per-sample input shape without the batch axis.
    :param model_name: Recorded in ``conformal.json``.
    :param weights: Which weights ``model`` holds (for example ``"last epoch"``), recorded.
    :return: The record that was written.
    """
    out_dir = Path(out_dir)
    logits, labels = collect_logits(model, dataset)
    n = int(labels.shape[0])
    logger.info(f"Conformal: {n} validation samples, alpha={alpha}")

    splits = repeated_split_coverage(
        logits, labels, alpha, repeats=SPLIT_REPEATS, seed=seed, randomized=False
    )
    tau = calibrate_aps(logits, labels, alpha, randomized=False)
    full = evaluate_sets(logits, labels, predict_sets(logits, tau, randomized=False))

    record = {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "score": "aps_deterministic",
        "tau": tau if np.isfinite(tau) else None,
        "tau_is_infinite": not bool(np.isfinite(tau)),
        "n_calibration": n,
        "model": model_name,
        "weights": weights,
        "accuracy": full["accuracy"],
        "repeated_splits": splits,
        "caveat": CAVEAT,
    }

    attach_conformal_sets(model, tau, input_shape).save(out_dir / CONFORMAL_MODEL_NAME)
    with open(out_dir / CONFORMAL_JSON_NAME, "w") as f:
        json.dump(record, f, indent=2, allow_nan=False)

    logger.info(
        f"Conformal: tau={tau:.6g}; on random half splits coverage "
        f"{splits['coverage_mean']:.4f} (min {splits['coverage_min']:.4f}, "
        f"target {1.0 - alpha:.2f}), mean set size {splits['mean_size']:.2f}, "
        f"empty sets {splits['empty_rate']:.4f}"
    )
    return record


def load_conformal_model(path: Path) -> keras.Model:
    """Load a ``conformal_model.keras`` written by :func:`run_conformal`."""
    return keras.models.load_model(
        path, custom_objects={"ConformalPredictionSet": ConformalPredictionSet}
    )
