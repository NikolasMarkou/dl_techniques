"""The scoring of the ``depth_estimation`` task: loss, metrics, the two constant baselines, per-crop Pearson and the
held-out report.

The loss and the metrics are the legacy trainer's (``train.depth_anything.train_depth_anything``): Depth Anything
V1's masked affine-invariant loss and five depth metrics aligned per image by scale and shift, each reading the
WHOLE target ``y = [disparity, mask]``. The constant and layout baselines are the ``val_loss`` of a weightless
:class:`train.common.pretrain_run.ConstantPredictor` evaluated under that compiled loss, so a baseline and
``val_loss`` share one reduction. A constant prediction scores exactly 1.0 under the masked loss (its normalized
map is 0, and the target's mean absolute normalized value is 1 by construction).

The held-out report scores the TEST crops once, from per-crop values: the model's loss and Pearson, the constant,
the layout and its Pearson, and the paired model-minus-layout difference with its standard error. Selection crops
select and are never reported as held-out.

Interface contract
------------------
``loss_and_metrics() -> (loss, metrics)``
    A fresh ``AffineInvariantLoss(masked=True)`` and the five metrics ``abs_rel``, ``sq_rel``, ``rmse``,
    ``rmse_log``, ``delta_1.25`` (on the normalized-disparity scale, not the paper's metric depth).
``constant_baseline(ds) -> float`` and ``layout_baseline(layout, ds) -> float``
    :param ds: one finite pass of ``(rgb, y)`` batches (``DepthBatches.select_ds`` or ``test_ds``).
    :param layout: the ``(S, S, 1)`` average layout map.
    :return: ``model.evaluate`` of the zero map, or of the layout map, under the compiled loss.
``selection_baselines(data) -> {"constant", "layout"}``
    Both baselines on ``data.select_ds``: the two terms of the bar besides the epoch-zero ``val_loss``.
``per_crop_loss(y, pred) -> (n,) float64`` and ``per_crop_pearson(y, pred) -> (n,) float64``
    :param y: ``(n, S, S, 2)`` targets; :param pred: ``(n, S, S, 1)`` predictions (a broadcast view is fine).
    Pearson is over the valid pixels of each crop, ``nan`` where the prediction or the disparity is constant there.
``held_out_report(model, data) -> dict``
    Plain floats and ints: ``n``; ``loss``, ``pearson``, ``layout``, ``layout_pearson``, ``model_minus_layout``,
    each a mean over the test crops with its ``*_sem`` (sample standard deviation over the square root of the
    count); ``pearson_n`` and ``layout_pearson_n`` (crops with a finite coefficient); ``constant``.

This module imports Keras and TensorFlow; ``train.api`` imports it inside functions only.
"""

import math
from typing import Any, Dict, List, Sequence, Tuple

import keras
import numpy as np

from train.common.pretrain_run import ConstantPredictor
from train.common.stats import mean_std
from train.depth_anything.train_depth_anything import build_depth_loss, build_depth_metrics

# ---------------------------------------------------------------------

#: Crops per call of the loss in :func:`per_crop_loss`.
LOSS_CHUNK = 32

# ---------------------------------------------------------------------


def loss_and_metrics() -> Tuple["keras.losses.Loss", List["keras.metrics.Metric"]]:
    """The compiled loss and metrics of a depth run, fresh objects each call; see the module contract."""
    return build_depth_loss(), build_depth_metrics()


def _map_shape(ds: Any) -> Tuple[int, ...]:
    """``(S, S, 1)``: the prediction shape of the crops of ``ds``."""
    return tuple(int(n) for n in ds.element_spec[1].shape[1:-1]) + (1,)


def _evaluate_constant(value: np.ndarray, ds: Any) -> float:
    """``val_loss`` of a model that outputs ``value`` for every crop, under the compiled masked loss."""
    model = ConstantPredictor(value, _map_shape(ds))
    model.compile(optimizer="sgd", loss=build_depth_loss(), jit_compile=False)
    return float(model.evaluate(ds, verbose=0, return_dict=True)["loss"])


def constant_baseline(ds: Any) -> float:
    """The zero map's loss on one pass of ``ds``; see the module contract."""
    # DECISION plan-2026-11-17T110935-9c7b9096/D-006: the constant term of the bar is MEASURED through the
    # compiled loss, never typed as 1.0. Do NOT replace it with the closed-form value or a numpy copy of the
    # loss: the bar must compare val_loss with a number made by the same reduction (a typed 1.0 would also
    # hide a change of the loss or of the mask channel that moves every val_loss).
    return _evaluate_constant(np.zeros(_map_shape(ds), np.float32), ds)


def layout_baseline(layout: np.ndarray, ds: Any) -> float:
    """The layout map's loss on one pass of ``ds``; see the module contract."""
    return _evaluate_constant(np.asarray(layout, np.float32), ds)


def selection_baselines(data: Any) -> Dict[str, float]:
    """The constant and the layout baselines on the SELECTION crops; see the module contract."""
    return {"constant": constant_baseline(data.select_ds), "layout": layout_baseline(data.layout, data.select_ds)}


def per_crop_loss(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """The masked loss of each crop (``AffineInvariantLoss.call`` keeps the batch axis), as float64."""
    loss = build_depth_loss()
    out = []
    for lo in range(0, len(y), LOSS_CHUNK):
        values = loss.call(keras.ops.convert_to_tensor(np.asarray(y[lo:lo + LOSS_CHUNK], np.float32)),
                           keras.ops.convert_to_tensor(np.asarray(pred[lo:lo + LOSS_CHUNK], np.float32)))
        out.append(keras.ops.convert_to_numpy(values))
    return np.concatenate(out).astype(np.float64).reshape(-1)


def per_crop_pearson(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Pearson of prediction and disparity over the valid pixels of each crop, float64; see the module contract."""
    out = np.full(len(y), np.nan, np.float64)
    for i in range(len(y)):
        valid = y[i, ..., 1] > 0
        a = np.asarray(pred[i, ..., 0], np.float64)[valid]
        b = np.asarray(y[i, ..., 0], np.float64)[valid]
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            continue
        a, b = a - a.mean(), b - b.mean()
        out[i] = float(np.dot(a, b) / math.sqrt(np.dot(a, a) * np.dot(b, b)))
    return out


def _mean_and_sem(values: Sequence[float]) -> Tuple[float, float, int]:
    """Mean, standard error and count of the finite values."""
    values = np.asarray(values, np.float64)
    n = int(np.isfinite(values).sum())
    mean, std = mean_std(values, ddof=1)
    return mean, (std / math.sqrt(n) if n else float("nan")), n


def held_out_report(model: "keras.Model", data: Any) -> Dict[str, Any]:
    """The test block, scored once on ``data.test_ds`` in order; see the module contract."""
    y = data.y_test
    pred = model.predict(data.test_ds, verbose=0)
    if pred.shape != y.shape[:-1] + (1,):
        raise ValueError(f"the model predicted {pred.shape} for {y.shape[0]} test crops of {y.shape[1:-1]}")
    layout = np.broadcast_to(data.layout, pred.shape)
    loss, layout_loss = per_crop_loss(y, pred), per_crop_loss(y, layout)
    report: Dict[str, Any] = {"n": int(len(y))}
    for key, values in (("loss", loss), ("pearson", per_crop_pearson(y, pred)), ("layout", layout_loss),
                        ("layout_pearson", per_crop_pearson(y, layout)), ("model_minus_layout", loss - layout_loss)):
        mean, sem, n = _mean_and_sem(values)
        report[key], report[f"{key}_sem"] = mean, sem
        if key.endswith("pearson"):
            report[f"{key}_n"] = n
    report["constant"] = constant_baseline(data.test_ds)
    return report
