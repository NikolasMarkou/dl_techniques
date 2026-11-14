"""Run pieces of the ``time_series_forecasting`` task: losses, metrics, the decode table, baselines, the report.

A forecasting model's OUTPUT is family-shaped: a tuple for nbeats, ``(B, H, F)`` or ``(B, H, F, Q)`` for prism,
``(B, H, Q)`` for tirex and for a quantile xlstm, ``(B, H, V, Q)`` for timestral, packed mixture parameters for
mdn. This module is the one place that says what each of them means, so the compiled metric, the post-hoc report,
the Predictor and the export read the same row (D-009). Everything is in STANDARDIZED units (context statistics,
``train.common.forecast_data``); this module never standardizes.

Interface contract
------------------
``DECODE``
    ``{row: DecodeRow}``, one row per family and head: ``nbeats``, ``prism_point``, ``prism_quantile``,
    ``tirex``, ``xlstm_point``, ``xlstm_quantile``, ``ets``, ``timestral``, ``mdn``. A row owns ``quantile`` (has
    a quantile head), ``flat_target`` (its loss wants the target as ``(B, H)``, not ``(B, H, C)``), ``objective``
    and how to read the levels and the median index off a model.

``decoder_for(family, model)``
    :return: a :class:`Decoder`, callable on the model's raw output (tensors or numpy, written with
        ``keras.ops`` so it runs inside a metric, a ``tf.function`` and an export):
        ``decoder(out) -> (point (B, H, C), quantiles (B, H, C, Q) or None)``; ``decoder.levels`` and
        ``decoder.median_index`` are ``None`` for a point head.
    :raises ConfigError: for a family with no row (``deepar`` and ``adaptive_ema`` are refused earlier, by name).
    The row equals ``predict_forecast`` for the six mixin families, ``TimestralForecaster`` with
    ``symmetric_averaging=False`` and ``clip_nonnegative=False`` for timestral (horizon <= patch size), and a
    mixture mean for mdn.

``ServedForecaster(model, family, scale_floor, static_batch=None)``
    A registered Keras layer, the function an exported file runs: ``batch`` is RAW windows ``(B, L, C)`` in the series'
    units; it standardizes with the context statistics (:func:`train.common.forecast_data.context_stats_ops`), runs
    ``model``, decodes with the family's row and de-standardizes with the same statistics.
    :return: the point ``(B, H, C)`` for a point head; ``{"point", "quantiles" (B, H, C, Q)}`` for a quantile head.
    ``static_batch`` is the batch a file of it is traced at (``None``: free; timestral packs rows by the batch and has
    no free one). It saves and reloads inside a ``keras.Model``; its config holds the serialized model.

``build_loss_and_metrics(decoder, model, loss_name=None)``
    :return: :class:`LossAndMetrics` ready for ``compile``; the metric is named ``mae`` for every family, so
        the watched key is ``val_mae``. ``loss_name`` (``mae``, ``mse``, ``mase``) applies to the point heads
        only; any other head with a name set is a :class:`ConfigError` naming ``loss_args.name``.

``beats_bar(reached, threshold, n_batches=1)``
    :return: whether ``reached`` (a ``val_mae``) is below ``threshold`` by more than ``tie_band(n_batches)`` float32 steps, both read
        as float32; ``tie_band(n_batches)`` is ``TIE_ULPS * max(1, ceil(sqrt(n_batches / 16)))`` (16 up to 16 batches, 192 at 2,048;
        ``TIE_ULPS`` is imported from ``train.common.segmentation_run``, not defined again) and ``n_batches`` is the batch count of
        the selection windows, which the task knows. ``epoch_zero_mae(sidecar_path)`` is the untrained model's ``mae`` as the fit sidecar holds it.

``baseline_maes(context, target)``, ``canonical_metrics(context, target, point, quantiles, levels)``
    numpy, float64 where it matters; ``context (N, L, C)``, ``target`` and ``point (N, H, C)``,
    ``quantiles (N, H, C, Q)``. ``canonical_report(model, decoder, ds)`` decodes a ``(x, y)`` dataset and calls
    ``canonical_metrics``. ``plot_forecasts(path, ...)`` writes one PNG.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np

from dl_techniques.losses.mase_loss import MASELoss
from dl_techniques.losses.quantile_loss import QuantileLoss
from dl_techniques.utils.matplotlib_backend import import_pyplot
from train.api.errors import ConfigError
from train.common import forecast_data
from train.common.resume_state import read_fit_state
from train.common.segmentation_run import TIE_ULPS
from train.common.timeseries import _plot_ts_forecast, compute_post_hoc_forecast_metrics

LOSS_NAMES = ("mae", "mse", "mase")
#: A context whose mean absolute one-step difference is not above this has no MASE scale (a flat window).
MASE_MIN_SCALE = 1e-12


# ---------------------------------------------------------------------
# baselines (numpy, data-free of any model)
# ---------------------------------------------------------------------


def seasonal_naive(context: np.ndarray, horizon: int, max_period: Optional[int] = None) -> np.ndarray:
    """Seasonal-naive forecast of (B, T) contexts, the period chosen per series from its context.

    Interface contract: for each series, every period m in [1, max_period] (default
    T // 2; m = 1 is the naive last value) is backtested on the context's last
    ``horizon`` steps, forecasting them from the ``horizon`` steps' own past; the period
    with the lowest backtest MAE forecasts the horizon, ``y[t] = x[T - m + (t mod m)]``.
    Only context values are read. Returns (B, horizon).
    """
    # DECISION plan-2026-11-13T105600-3a8319ef/D-015
    # This function MOVED here from src/train/time_series/timestral/train_timestral.py, body and signature
    # unchanged; that file imports it. Do NOT copy it back or import the trainer script from here (a script
    # that loads argparse and TensorFlow for one function, and a second definition that drifts): the bar of
    # the forecasting task and the legacy timestral held-out metrics must read one implementation.
    context = np.asarray(context, dtype=np.float64)
    b, t = context.shape
    max_period = max_period or t // 2
    if t < horizon + max_period:
        raise ValueError(f"context {t} is too short to backtest {max_period} periods over {horizon} steps")
    steps = np.arange(horizon)
    history, held = context[:, :t - horizon], context[:, t - horizon:]
    best_err = np.full(b, np.inf)
    best = np.zeros((b, horizon))
    for m in range(1, max_period + 1):
        backtest = history[:, history.shape[1] - m + steps % m]
        err = np.abs(backtest - held).mean(axis=1)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best[better] = context[better][:, t - m + steps % m]
    return best


def naive_last(context: np.ndarray, horizon: int) -> np.ndarray:
    """The last context value repeated: ``(N, L, C) -> (N, horizon, C)`` float64."""
    context = np.asarray(context, np.float64)
    return np.repeat(context[:, -1:, :], horizon, axis=1)


def context_mean(context: np.ndarray, horizon: int) -> np.ndarray:
    """The context's mean repeated: ``(N, L, C) -> (N, horizon, C)`` float64."""
    context = np.asarray(context, np.float64)
    return np.repeat(context.mean(axis=1, keepdims=True), horizon, axis=1)


def seasonal_naive_windows(context: np.ndarray, horizon: int) -> np.ndarray:
    """:func:`seasonal_naive` of every channel of ``(N, L, C)`` contexts, ``(N, horizon, C)`` float64.

    The longest period tried is what the context can back-test, ``min(L // 2, L - horizon)``; a context that
    cannot back-test even period 1 (``L <= horizon``) forecasts period 1, which IS the naive last value.
    """
    context = np.asarray(context, np.float64)
    n, t, c = context.shape
    longest = min(t // 2, t - horizon)
    if longest < 1:
        return naive_last(context, horizon)
    flat = context.transpose(0, 2, 1).reshape(n * c, t)
    return seasonal_naive(flat, horizon, longest).reshape(n, c, horizon).transpose(0, 2, 1)


def baseline_maes(context: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """MAE of naive-last, seasonal-naive and the context mean on the same windows (``context``, ``target``)."""
    target = np.asarray(target, np.float64)
    horizon = target.shape[1]

    def mae(forecast: np.ndarray) -> float:
        return float(np.mean(np.abs(target - forecast)))

    return {
        "naive_last": mae(naive_last(context, horizon)),
        "seasonal_naive": mae(seasonal_naive_windows(context, horizon)),
        "context_mean": mae(context_mean(context, horizon)),
        "n_windows": int(len(target)),
    }


def tie_band(n_batches: int) -> int:
    """The width, in float32 steps of the threshold, of the band below it that is still a miss: ``TIE_ULPS * max(1, ceil(sqrt(n / 16)))``.

    ``n_batches`` is the number of batches of the selection windows. The band is :data:`TIE_ULPS` (the segmentation task's, defined
    once in ``train.common.segmentation_run``) up to 16 batches and 192 at 2,048. Measured (plan step 12): the compiled ``mae`` of a
    model that outputs the naive-last forecast sits 1 float32 step BELOW the float64 numpy baseline in 2 of 8 seeds at the default
    sizes; (review, D-043): the float32 accumulation of that metric drifts with the number of terms, +20 steps at 262,144 windows
    (2,048 batches of 128), so the band widens with the square root of the batch count.
    """
    return TIE_ULPS * max(1, math.ceil(math.sqrt(n_batches / 16)))


def beats_bar(reached: float, threshold: float, n_batches: int = 1) -> bool:
    """Whether ``reached`` is below ``threshold`` by more than ``tie_band(n_batches)`` float32 steps (see the module contract)."""
    # DECISION plan-2026-11-13T105600-3a8319ef/D-008
    # DECISION plan-2026-11-13T105600-3a8319ef/D-043
    # STRICTLY below, in float32, and with a tie band. `<=` would call a model that merely EQUALS a baseline a success; a
    # float64 `reached` that rounds to the float32 threshold must be judged as the float32 number Keras computed; and the
    # baselines are float64 numpy while the compiled metric adds in float32, so a model that outputs the naive-last forecast
    # scores 1 step BELOW its own baseline on some seeds (MEASURED) and a bare `<` passes it. The band grows with the number of
    # select batches (D-043): a fixed 16 steps was measured too narrow at 2,048 batches (+20). Do NOT shrink the band to 0 "to be
    # strict", do NOT fix it at 16 again, and do NOT compare the Python floats.
    low = np.float32(threshold)
    return bool(np.float32(reached) < low - np.float32(tie_band(n_batches)) * np.spacing(low))


def epoch_zero_mae(sidecar_path: Any) -> float:
    """The untrained model's ``mae`` on the select windows as the fit sidecar holds it (see the module contract)."""
    # DECISION plan-2026-11-13T105600-3a8319ef/D-008
    # The epoch-zero term of the bar is read from the SIDECAR, always, never from an attribute the first process set: a
    # resume never calls `initial_loss_check`, and re-evaluating the weights a resume restored would measure the trained
    # model. Do NOT cache it on the trainer "to save a read".
    state = read_fit_state(sidecar_path)
    initial = state.get("initial")
    metrics = initial.get("metrics") if isinstance(initial, dict) else None
    value = metrics.get("mae") if isinstance(metrics, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"the epoch-zero val_mae is not in {sidecar_path}: the fit sidecar holds no initial.metrics.mae, so the success "
            "bar cannot be cut (a run writes it before its first epoch)"
        )
    return float(value)


# ---------------------------------------------------------------------
# DECODE
# ---------------------------------------------------------------------


def _mdn_layer(model: Any) -> Any:
    """The ``MDNLayer`` of an ``MDNModel`` or of a model that contains one (the Flatten stub)."""
    holder = model if hasattr(model, "mdn_layer") else next(
        (layer for layer in getattr(model, "layers", ()) if hasattr(layer, "mdn_layer")), None)
    if holder is None:
        raise ConfigError("the mdn decode row needs a model that holds an MDNModel")
    return holder.mdn_layer


def _first_of_tuple(model: Any, median: Optional[int]) -> Callable[[Any], Tuple[Any, None]]:
    """nbeats: ``(forecast (B, H, C), residual (B, L))``; the residual is NOT a forecast."""
    return lambda out: (out[0], None)


def _as_is(model: Any, median: Optional[int]) -> Callable[[Any], Tuple[Any, None]]:
    """prism and xlstm point heads and ets: the output is the forecast ``(B, H, C)``."""
    return lambda out: (out, None)


def _cube_with_channels(model: Any, median: Optional[int]) -> Callable[[Any], Tuple[Any, Any]]:
    """prism quantile ``(B, H, F, Q)`` and timestral ``(B, H, V, Q)``: the median slice is the point."""
    return lambda out: (out[..., median], out)


def _cube_without_channels(model: Any, median: Optional[int]) -> Callable[[Any], Tuple[Any, Any]]:
    """tirex and xlstm quantile ``(B, H, Q)``: a channel axis of 1 is added to the point and the cube."""
    ops = keras.ops
    return lambda out: (ops.expand_dims(out[..., median], -1), ops.expand_dims(out, 2))


def _mixture_mean(model: Any, median: Optional[int]) -> Callable[[Any], Tuple[Any, None]]:
    """mdn: ``sum_k softmax(pi)_k * mu_k`` over the mixture, ``(B, H, 1)``; the layer owns the packed layout."""
    layer = _mdn_layer(model)
    ops = keras.ops

    def decode(out: Any) -> Tuple[Any, None]:
        mu, _, logits = layer.split_mixture_params(ops.convert_to_tensor(out))
        mean = ops.sum(mu * ops.expand_dims(ops.softmax(logits, axis=-1), -1), axis=1)
        return ops.expand_dims(mean, -1), None

    return decode


def _attr_levels(model: Any) -> Tuple[float, ...]:
    return tuple(float(q) for q in model.quantile_levels)


def _knot_levels(model: Any) -> Tuple[float, ...]:
    return tuple(float(q) for q in model.config.quantile_knots)


def _nearest_to_half(model: Any, levels: Tuple[float, ...]) -> int:
    # prism picks the head nearest to 0.5 by value (its own ``predict_quantiles``), a tie going to the first.
    return int(np.argmin(np.abs(np.asarray(levels) - 0.5)))


def _half_or_middle(model: Any, levels: Tuple[float, ...]) -> int:
    # tirex and xlstm: the 0.5 level when trained, else ``len // 2`` (their own ``predict_quantiles``).
    return levels.index(0.5) if 0.5 in levels else len(levels) // 2


def _model_median(model: Any, levels: Tuple[float, ...]) -> int:
    return int(model.config.median_index)


@dataclass(frozen=True)
class DecodeRow:
    """One family and head of ``DECODE``.

    :param bind: ``(model, median_index) -> decode`` where ``decode(out) -> (point, quantiles or None)``.
    :param quantile: the head emits quantiles, so ``levels`` and ``median`` apply.
    :param flat_target: the loss wants ``y`` as ``(B, H)`` (tirex, xlstm quantile), the data gives ``(B, H, C)``.
    :param objective: ``point`` (a chosen loss), ``quantile`` (pinball), ``model`` (the model's own
        ``compute_loss``, no compiled loss) or ``nll`` (the MDN layer's negative log-likelihood).
    :param levels, median: read the quantile levels and the median index off a model, by the family's own rule.
    """

    bind: Callable[[Any, Optional[int]], Callable[[Any], Tuple[Any, Optional[Any]]]]
    quantile: bool
    flat_target: bool
    objective: str
    levels: Optional[Callable[[Any], Tuple[float, ...]]] = None
    median: Optional[Callable[[Any, Tuple[float, ...]], int]] = None


# DECISION plan-2026-11-13T105600-3a8319ef/D-009
# ONE decode row per family and head, shared by the compiled metric, the post-hoc report, the Predictor and the
# export. Do NOT call ``predict_forecast`` in those places (it is missing for timestral and mdn, shapes differ
# per family, and it is eager numpy so it cannot be an exported function), and do NOT let the consumers decode
# for themselves: a mismatch would then show up as a wrong number in one of them only. The tests hold each row
# equal to the model's own forecast function. timestral's ``call`` is the forecaster's first block only for
# horizon <= patch size, and only against the forecaster with symmetric averaging and the non-negative clip OFF.
DECODE: Dict[str, DecodeRow] = {
    "nbeats": DecodeRow(_first_of_tuple, False, False, "point"),
    "prism_point": DecodeRow(_as_is, False, False, "point"),
    "prism_quantile": DecodeRow(_cube_with_channels, True, False, "quantile", _attr_levels, _nearest_to_half),
    "tirex": DecodeRow(_cube_without_channels, True, True, "quantile", _attr_levels, _half_or_middle),
    "xlstm_point": DecodeRow(_as_is, False, False, "point"),
    "xlstm_quantile": DecodeRow(_cube_without_channels, True, True, "quantile", _attr_levels, _half_or_middle),
    "ets": DecodeRow(_as_is, False, False, "point"),
    "timestral": DecodeRow(_cube_with_channels, True, False, "model", _knot_levels, _model_median),
    "mdn": DecodeRow(_mixture_mean, False, False, "nll"),
}


@dataclass(frozen=True)
class Decoder:
    """A :class:`DecodeRow` bound to one model: callable on its raw output."""

    row: str
    fn: Callable[[Any], Tuple[Any, Optional[Any]]]
    levels: Optional[Tuple[float, ...]]
    median_index: Optional[int]
    flat_target: bool

    def __call__(self, out: Any) -> Tuple[Any, Optional[Any]]:
        return self.fn(out)


def row_key(family: str, model: Any) -> str:
    """The ``DECODE`` row of a family: prism and xlstm split by their head, the others are one row."""
    if family in ("prism", "xlstm"):
        return f"{family}_{'quantile' if model.use_quantile_head else 'point'}"
    if family in DECODE:
        return family
    raise ConfigError(f"no decode row for family {family!r}; rows: {', '.join(DECODE)}")


def decoder_for(family: str, model: Any) -> Decoder:
    """The decoder of ``model`` (see the module contract)."""
    key = row_key(family, model)
    row = DECODE[key]
    levels = row.levels(model) if row.levels else None
    median = row.median(model, levels) if row.median else None
    return Decoder(key, row.bind(model, median), levels, median, row.flat_target)


@keras.saving.register_keras_serializable(package="train_forecast")
class ServedForecaster(keras.layers.Layer):
    """Raw window in, forecast in the series' units out; see the module contract."""

    def __init__(self, model: Any, family: str, scale_floor: float, static_batch: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # DECISION plan-2026-11-13T105600-3a8319ef/D-026
        # A REGISTERED layer carries the standardization into a `.keras` file. Do NOT wrap the served function in a
        # `keras.layers.Lambda`: it saves but does not load under the default safe mode (measured, step 5(d)), and a
        # file that needs the caller to standardize is the trap D-011 exists to remove.
        self.model = model
        self.family = family
        self.scale_floor = float(scale_floor)
        self.static_batch = static_batch
        self.decoder = decoder_for(family, model)

    def call(self, batch: Any) -> Any:
        ops = keras.ops
        anchor, offset, scale = forecast_data.context_stats_ops(batch, self.scale_floor)
        # DECISION plan-2026-11-13T105600-3a8319ef/D-040
        # The anchor is subtracted FIRST (an exact difference of nearby float32 numbers) and added back LAST. Do NOT fold anchor and
        # offset into one `mean` and subtract that: at a level of 1e5 over a std of 8 the sum has lost the digits the standardized
        # window is made of, and the file forecasts 0.25 of a window std away from `predict`.
        point, cube = self.decoder(self.model(((batch - anchor) - offset) / scale, training=False))
        point = (point * scale + offset) + anchor
        if cube is None:
            return point
        return {"point": point, "quantiles": (cube * ops.expand_dims(scale, -1) + ops.expand_dims(offset, -1))
                + ops.expand_dims(anchor, -1)}

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "model": keras.saving.serialize_keras_object(self.model), "family": self.family,
                "scale_floor": self.scale_floor, "static_batch": self.static_batch}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ServedForecaster":
        return cls(**{**config, "model": keras.saving.deserialize_keras_object(config["model"])})


class DecodedMAE(keras.metrics.Metric):
    """MAE of the decoded POINT (the median for a quantile head) against the target, however the target is shaped.

    Stock ``keras.metrics.Metric`` state (a sum and a count); ``y_true`` is reshaped to the point's shape, so a
    flat ``(B, H)`` target of tirex and a ``(B, H, C)`` one both work. Not serializable on its own (the decoder
    is a closure): a model compiled with it reloads with ``compile=False``, which every reload of this task uses.
    """

    def __init__(self, decoder: Decoder, name: str = "mae", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.decoder = decoder
        self.total = self.add_variable(shape=(), initializer="zeros", name="total")
        self.count = self.add_variable(shape=(), initializer="zeros", name="count")

    def update_state(self, y_true: Any, y_pred: Any, sample_weight: Any = None) -> None:
        ops = keras.ops
        point, _ = self.decoder(y_pred)
        target = ops.reshape(ops.cast(y_true, "float32"), ops.shape(point))
        error = ops.abs(target - ops.cast(point, "float32"))
        self.total.assign(self.total + ops.sum(error))
        self.count.assign(self.count + ops.cast(ops.size(error), "float32"))

    def result(self) -> Any:
        return self.total / keras.ops.maximum(self.count, 1.0)

    def reset_state(self) -> None:
        self.total.assign(0.0)
        self.count.assign(0.0)


@dataclass(frozen=True)
class LossAndMetrics:
    """What ``compile`` takes: ``loss`` (``None`` for timestral, a list for nbeats) and ``metrics``."""

    loss: Any
    metrics: List[Any]


_FIXED_LOSS = {
    "quantile": "the pinball loss over its quantile levels",
    "model": "the objective the model owns (no compiled loss)",
    "nll": "the mixture negative log-likelihood",
}


def _point_loss(name: Optional[str]) -> Any:
    if name in (None, "mae"):
        return keras.losses.MeanAbsoluteError()
    if name == "mse":
        return keras.losses.MeanSquaredError()
    if name == "mase":
        return MASELoss()
    raise ConfigError(f"loss_args.name must be one of {', '.join(LOSS_NAMES)} or null, got {name!r}")


def build_loss_and_metrics(decoder: Decoder, model: Any, loss_name: Optional[str] = None) -> LossAndMetrics:
    """The compiled loss and the ``mae`` metric of a decoder's row (see the module contract)."""
    row = DECODE[decoder.row]
    if row.objective == "point":
        loss, mae = _point_loss(loss_name), keras.metrics.MeanAbsoluteError(name="mae")
        # nbeats has two outputs: the forecast is scored, the residual is not (D-007).
        return LossAndMetrics([loss, None], [[mae], []]) if decoder.row == "nbeats" else LossAndMetrics(loss, [mae])
    if loss_name is not None:
        raise ConfigError(
            f"loss_args.name {loss_name!r} is not accepted for {decoder.row}: its loss is {_FIXED_LOSS[row.objective]}, "
            f"not a choice")
    if row.objective == "quantile":
        loss = QuantileLoss(quantiles=list(decoder.levels))
    elif row.objective == "nll":
        loss = _mdn_layer(model).loss_func
    else:
        loss = None
    return LossAndMetrics(loss, [DecodedMAE(decoder)])


# ---------------------------------------------------------------------
# the post-hoc report
# ---------------------------------------------------------------------


def _canonical_mase(context: np.ndarray, target: np.ndarray, point: np.ndarray) -> Tuple[float, int]:
    """``(MASE, windows left out)``: per (window, channel) series, the horizon's MAE over the mean absolute
    one-step difference of its CONTEXT, averaged over the series that have a scale."""
    # DECISION plan-2026-11-13T105600-3a8319ef/D-008
    # Canonical MASE scales each series by ITS OWN context's naive one-step error. Do NOT take the scale from
    # the target horizon (that is MASELoss's batch approximation, and it divides the error by a quantity the
    # forecaster is asked to predict) and do NOT use one batch-global scale (``calculate_comprehensive_metrics``
    # does; a rough window then weighs the same as a smooth one). A flat context has scale 0: left out, counted.
    context, target, point = (np.asarray(a, np.float64) for a in (context, target, point))
    scale = np.abs(np.diff(context, axis=1)).mean(axis=1)
    error = np.abs(target - point).mean(axis=1)
    scored = scale > MASE_MIN_SCALE
    mase = float((error[scored] / scale[scored]).mean()) if scored.any() else float("nan")
    return mase, int(scored.size - scored.sum())


def canonical_metrics(
        context: np.ndarray,
        target: np.ndarray,
        point: np.ndarray,
        quantiles: Optional[np.ndarray] = None,
        levels: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """MAE, RMSE, sMAPE, rMAE, canonical MASE and ``MASE_excluded``; coverage, sharpness and
    ``coverage_nominal`` of the widest interval when ``quantiles`` and two or more ``levels`` are given.

    The shared block of ``compute_post_hoc_forecast_metrics`` supplies everything but MASE, whose batch-global
    scale it computes is replaced by the per-series canonical one.
    """
    block = compute_post_hoc_forecast_metrics(
        target, point, backcast=context, quantiles=quantiles,
        quantile_levels=None if levels is None else list(levels))
    block["MASE"], excluded = _canonical_mase(context, target, point)
    block["MASE_excluded"] = float(excluded)
    return block


def decode_dataset(model: Any, decoder: Decoder, ds: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Run ``model`` over a ``(x, y)`` dataset: ``x``, ``y`` (reshaped to the point's shape), point, quantiles.

    One model call per batch, so a static-batch family (timestral) sees only the full batches of the dataset.
    """
    xs, ys, points, cubes = [], [], [], []
    for x, y in ds:
        point, cube = decoder(model(x, training=False))
        point = keras.ops.convert_to_numpy(point)
        xs.append(np.asarray(x))
        ys.append(np.asarray(y).reshape(point.shape))
        points.append(point)
        if cube is not None:
            cubes.append(keras.ops.convert_to_numpy(cube))
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(points),
            np.concatenate(cubes) if cubes else None)


def canonical_report(model: Any, decoder: Decoder, ds: Any) -> Dict[str, float]:
    """:func:`canonical_metrics` of ``model`` on the windows of ``ds``."""
    x, y, point, cube = decode_dataset(model, decoder, ds)
    return canonical_metrics(x, y, point, cube, decoder.levels)


# ---------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------


def plot_forecasts(
        path: Any,
        context: np.ndarray,
        target: np.ndarray,
        point: np.ndarray,
        quantiles: Optional[np.ndarray] = None,
        levels: Optional[Sequence[float]] = None,
        count: int = 6,
        title: Optional[str] = None,
) -> str:
    """Write a grid of the first ``count`` windows (channel 0): context, target, point and, with ``quantiles``,
    the band from the lowest to the highest level. Returns ``str(path)``."""
    plt = import_pyplot()
    shown = min(count, len(context))
    cols = min(3, shown)
    rows = math.ceil(shown / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)
    low = high = None
    if quantiles is not None and levels is not None and len(levels) >= 2:
        low, high = int(np.argmin(levels)), int(np.argmax(levels))
    for i, ax in enumerate(axes.ravel()):
        if i >= shown:
            ax.axis("off")
            continue
        lower = None if low is None else quantiles[i, :, 0, low]
        upper = None if high is None else quantiles[i, :, 0, high]
        _plot_ts_forecast(ax, context[i, :, 0], target[i, :, 0], point[i, :, 0], lower, upper,
                          title=f"window {i}")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(str(path), dpi=100)
    plt.close(fig)
    return str(path)
