"""The forecasting run pieces (plan-2026-11-13T105600-3a8319ef, Step 9): baselines, decode table, metrics, report.

``train.common.forecast_run`` is the one place that knows what a forecasting model's OUTPUT means. This file pins,
on the CPU at tiny sizes, the part of it Step 9 shipped (the second and third parts below add the argument groups and
the family adapters; the task has its own file, ``test_forecast_task.py``):

* ``seasonal_naive`` MOVED from ``train_timestral.py`` and its outputs are the ones recorded from the original
  function before the move (``seasonal_naive_recorded.json``), exactly; the legacy file keeps no definition of
  its own, only the import;
* ``naive_last``, ``context_mean`` and the seasonal baseline (any channel count, a context too short to back-test
  falls back to period 1) against independent float64 oracles;
* the canonical post-hoc metrics: MAE, RMSE, sMAPE, rMAE, MASE with the scale from the CONTEXT's one-step
  differences, coverage and sharpness of a quantile cube, each against a float64 oracle;
* ``DECODE``, one row per family and head: equal to ``predict_forecast`` for the six mixin families, to
  ``TimestralForecaster`` (symmetric averaging and the non-negative clip OFF) for timestral, to a float64
  mixture-mean oracle for mdn; a wrong median index, the nbeats residual or a lost row each fail;
* the compiled ``mae`` metric: the stock MAE for a point head, the numpy MAE of the median for a quantile cube,
  the key ``val_mae`` for all nine families, and the loss builder (target layout, refusals).

Second part (Step 10): the argument groups of ``train.api.forecast_args``: strict parsing, the loader's layering, the
recipe-filled fields, every refusal that can be made BEFORE the run directory is claimed (each in a fresh interpreter
that must not have imported TensorFlow or keras afterwards), the watched metric and the loss names.

Third part (Step 11): ``train.api.forecast_models``: every adapter builds a BUILT model at ``(None, lookback, channels)`` whose
output the decode rows read (a static batch for timestral); the ``ALLOWED``, ``REFUSED``, ``FORCED`` and ``SUPPLIED`` tables are
held equal to the factory signatures by ``signature_census``; one normalization (``use_normalization`` forced off, a user
``true`` refused by name); the named refusals (a second channel for tirex, ets and mdn, a timestral horizon above its patch,
``deepar`` and ``adaptive_ema`` before any model code is imported, an sgd weight decay, ``jit_compile`` for ets); the recipes
against the legacy trainers' values READ from the legacy modules, each difference named; and the specs, which ``recipes.register_builtin_models``
registers with the task by one call.
"""

import ast
import dataclasses
import inspect
import json
import subprocess
import sys
from pathlib import Path

import keras
import numpy as np
import pytest

from train.api import forecast_args as fa
from train.api import forecast_models as fm
from train.api import recipes, registry
from train.api.args import AutoContext, OptimArgs
from train.api.config import validate
from train.api.errors import ConfigError, ConfigValidationError, UnknownModelError
from train.common import forecast_run as fr

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2] / "src"
RECORDED = json.loads((HERE / "seasonal_naive_recorded.json").read_text())

L, H, B = 64, 16, 4
#: Levels without 0.5: prism takes the nearest to 0.5 (index 1, a tie going to the first), tirex and xlstm take
#: ``len // 2`` (index 2): the two rules DISAGREE here, so a row that borrows the other family's rule is seen.
LEVELS = (0.1, 0.45, 0.55, 0.9)
ROWS = ("nbeats", "prism_point", "prism_quantile", "tirex", "xlstm_point", "xlstm_quantile", "ets", "timestral", "mdn")
MIXIN_ROWS = tuple(r for r in ROWS if r not in ("timestral", "mdn"))
FLAT_TARGET_ROWS = ("tirex", "xlstm_quantile")


# ---------------------------------------------------------------------
# tiny models, one per row (written out here; the adapters of Step 11 build the same shapes from the recipes)
# ---------------------------------------------------------------------

_CACHE = {}


def build_row(row: str):
    """``(family, model)`` of one decode row, built at ``(None, L, 1)`` with horizon ``H``, random weights."""
    if row in _CACHE:
        return _CACHE[row]
    keras.utils.set_random_seed(11)
    family = row.split("_")[0]
    if row == "nbeats":
        from dl_techniques.models.time_series.nbeats.nbeats import create_nbeats_model

        model = create_nbeats_model(backcast_length=L, forecast_length=H, hidden_layer_units=32,
                                    use_normalization=False, input_dim=1, output_dim=1)
        if not model.built:
            model.build((None, L, 1))
    elif row in ("prism_point", "prism_quantile"):
        from dl_techniques.models.time_series.prism.model import PRISM

        extra = {"use_quantile_head": True, "quantile_levels": list(LEVELS), "num_quantiles": len(LEVELS)} \
            if row == "prism_quantile" else {}
        model = PRISM.from_variant("tiny", context_len=L, forecast_len=H, num_features=1, **extra)
        model.build((None, L, 1))
    elif row == "tirex":
        from dl_techniques.models.time_series.tirex.model import create_tirex_by_variant

        model = create_tirex_by_variant("tiny", input_length=L, prediction_length=H, quantile_levels=LEVELS)
    elif row in ("xlstm_point", "xlstm_quantile"):
        from dl_techniques.models.time_series.xlstm.forecaster import xLSTMForecaster

        extra = {"use_quantile_head": row == "xlstm_quantile"}
        if row == "xlstm_quantile":
            extra["quantile_levels"] = list(LEVELS)
        model = xLSTMForecaster.from_variant("tiny", input_length=L, prediction_length=H, num_features=1,
                                             use_normalization=False, **extra)
        model.build((None, L, 1))
    elif row == "ets":
        from dl_techniques.models.time_series.ets.model import create_ets

        model = create_ets("ANN", horizon=H)
        model.build((None, L, 1))
    elif row == "timestral":
        from dl_techniques.models.time_series.timestral.model import create_timestral_by_variant

        model = create_timestral_by_variant("5M", prediction_length=H, patch_size=32)
        model.build((None, L, 1))
    elif row == "mdn":
        from dl_techniques.models.time_series.mdn.model import MDNModel

        inputs = keras.Input(shape=(L, 1))
        core = MDNModel(hidden_layers=[16, 16], output_dimension=H, num_mixtures=3, name="mdn")
        model = keras.Model(inputs, core(keras.layers.Flatten()(inputs)), name="mdn_stub")
    else:
        raise KeyError(row)
    _CACHE[row] = (family, model)
    return _CACHE[row]


def windows(n: int = B, seed: int = 5):
    """Standardized-looking float32 windows ``x (n, L, 1)`` and ``y (n, H, 1)``."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, L, 1)).astype("float32")
    y = rng.standard_normal((n, H, 1)).astype("float32")
    return x, y


def decoder_of(row: str):
    family, model = build_row(row)
    return fr.decoder_for(family, model), model


# ---------------------------------------------------------------------
# seasonal_naive was MOVED, not rewritten
# ---------------------------------------------------------------------


def _recorded_context(case):
    context = np.array(RECORDED["context"])
    return context.astype(np.float32) if case.get("float32_context") else context


@pytest.mark.parametrize("name", sorted(RECORDED["cases"]))
def test_the_moved_seasonal_naive_reproduces_the_recorded_outputs_exactly(name):
    case = RECORDED["cases"][name]
    got = fr.seasonal_naive(_recorded_context(case), case["horizon"], case["max_period"])
    assert got.shape == np.array(case["out"]).shape
    assert np.array_equal(got, np.array(case["out"])), f"{name}: max diff {np.abs(got - np.array(case['out'])).max()}"


def test_the_moved_seasonal_naive_refuses_a_context_too_short_to_back_test_with_the_recorded_message():
    short = RECORDED["short_context_raises"]
    context = np.array(RECORDED["context"])[:, :short["columns"]]
    with pytest.raises(ValueError) as caught:
        fr.seasonal_naive(context, short["horizon"])
    assert str(caught.value) == short["message"]


def test_the_legacy_timestral_trainer_imports_the_function_and_keeps_no_copy():
    path = SRC / "train" / "time_series" / "timestral" / "train_timestral.py"
    tree = ast.parse(path.read_text())
    defined = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert "seasonal_naive" not in defined, "a second definition of seasonal_naive stays in the legacy trainer"
    imported = [(n.module, a.name) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names]
    assert ("train.common.forecast_run", "seasonal_naive") in imported
    from train.time_series.timestral import train_timestral

    assert train_timestral.seasonal_naive is fr.seasonal_naive


# ---------------------------------------------------------------------
# baselines against float64 oracles
# ---------------------------------------------------------------------


def _oracle_seasonal(series: np.ndarray, horizon: int, max_period: int) -> np.ndarray:
    """Per-series period by back-test, written as plain loops (no vectorization shared with the function)."""
    series = np.asarray(series, np.float64)
    t = len(series)
    held = series[t - horizon:]
    best_err, best = np.inf, None
    for m in range(1, max_period + 1):
        backtest = [series[t - horizon - m + (k % m)] for k in range(horizon)]
        err = float(np.mean(np.abs(np.array(backtest) - held)))
        if err < best_err:
            best_err = err
            best = np.array([series[t - m + (k % m)] for k in range(horizon)])
    return best


def test_naive_last_repeats_the_last_context_value_in_float64():
    rng = np.random.default_rng(1)
    context = rng.standard_normal((5, 40, 2)).astype("float32")
    got = fr.naive_last(context, 7)
    assert got.shape == (5, 7, 2) and got.dtype == np.float64
    want = np.stack([np.stack([np.full(7, float(context[i, -1, c])) for c in range(2)], axis=-1) for i in range(5)])
    assert np.array_equal(got, want)
    assert not np.array_equal(got[:, 0, :], context[:, 0, :]), "the first context value is not the naive forecast"


def test_context_mean_repeats_the_float64_mean_of_the_context():
    rng = np.random.default_rng(2)
    context = (rng.standard_normal((5, 40, 2)) * 3 + 1).astype("float32")
    got = fr.context_mean(context, 6)
    assert got.shape == (5, 6, 2)
    want = np.broadcast_to(context.astype(np.float64).mean(axis=1, keepdims=True), (5, 6, 2))
    assert np.allclose(got, want, atol=1e-12, rtol=0)


def test_the_seasonal_baseline_continues_an_exact_cycle_and_matches_the_loop_oracle():
    cycle = np.array([1.0, 2.0, 4.0, 8.0, 3.0, 0.0, 7.0])
    series = np.tile(cycle, 16)
    context = np.stack([series[:84], series[3:87]])[..., None]  # two phases of a period-7 series, (2, 84, 1)
    got = fr.seasonal_naive_windows(context, 12)
    for i, start in enumerate((0, 3)):
        want = series[start + 84:start + 84 + 12]
        assert np.array_equal(got[i, :, 0], want), "an exact period-7 cycle is continued, not shifted"
    rng = np.random.default_rng(3)
    noisy = np.cumsum(rng.standard_normal((6, 96, 1)), axis=1) + np.sin(np.arange(96) * 2 * np.pi / 12)[None, :, None]
    got = fr.seasonal_naive_windows(noisy, 10)
    for i in range(6):
        assert np.allclose(got[i, :, 0], _oracle_seasonal(noisy[i, :, 0], 10, 96 // 2), atol=1e-12, rtol=0)


def test_the_seasonal_baseline_forecasts_each_channel_on_its_own():
    rng = np.random.default_rng(4)
    context = rng.standard_normal((3, 60, 2))
    both = fr.seasonal_naive_windows(context, 8)
    for c in range(2):
        alone = fr.seasonal_naive(context[:, :, c], 8)
        assert np.array_equal(both[:, :, c], alone)


@pytest.mark.parametrize("lookback, horizon", [(10, 12), (12, 12), (13, 12), (20, 12)])
def test_a_context_too_short_to_back_test_a_period_falls_back_to_the_naive_last_value(lookback, horizon):
    rng = np.random.default_rng(6)
    context = rng.standard_normal((3, lookback, 1))
    got = fr.seasonal_naive_windows(context, horizon)
    assert got.shape == (3, horizon, 1)
    if lookback <= horizon:
        assert np.array_equal(got, fr.naive_last(context, horizon)), "period 1 is the naive last value"
    else:
        assert np.isfinite(got).all()


def test_baseline_maes_are_float64_means_over_every_window_and_step():
    rng = np.random.default_rng(7)
    context = rng.standard_normal((9, 48, 1)).astype("float32")
    target = rng.standard_normal((9, 8, 1)).astype("float32")
    got = fr.baseline_maes(context, target)
    assert set(got) == {"naive_last", "seasonal_naive", "context_mean", "n_windows"} and got["n_windows"] == 9
    c64, t64 = context.astype(np.float64), target.astype(np.float64)
    assert got["naive_last"] == float(np.mean(np.abs(t64 - c64[:, -1:, :])))
    assert got["context_mean"] == float(np.mean(np.abs(t64 - c64.mean(axis=1, keepdims=True))))
    seasonal = np.stack([_oracle_seasonal(c64[i, :, 0], 8, 24) for i in range(9)])[..., None]
    assert abs(got["seasonal_naive"] - float(np.mean(np.abs(t64 - seasonal)))) < 1e-12


def test_a_model_that_outputs_naive_last_scores_the_naive_last_baseline():
    """The instrument's zero: the baseline the bar uses is what the forecast it names scores."""
    rng = np.random.default_rng(8)
    context = np.cumsum(rng.standard_normal((20, 48, 1)), axis=1).astype("float32")
    target = (context[:, -1:, :] + rng.standard_normal((20, 8, 1))).astype("float32")
    from_baseline = fr.baseline_maes(context, target)["naive_last"]
    scored = fr.canonical_metrics(context, target, fr.naive_last(context, 8))["MAE"]
    assert abs(from_baseline - scored) < 1e-6


# ---------------------------------------------------------------------
# canonical post-hoc metrics
# ---------------------------------------------------------------------


def _oracle_mase(context, target, point):
    """Per (window, channel) series: MAE of the horizon over the mean absolute one-step difference of its CONTEXT."""
    values = []
    for i in range(context.shape[0]):
        for c in range(context.shape[2]):
            series = np.asarray(context[i, :, c], np.float64)
            scale = np.mean([abs(series[t] - series[t - 1]) for t in range(1, len(series))])
            if scale > 1e-12:
                err = np.mean(np.abs(np.asarray(target[i, :, c], np.float64) - np.asarray(point[i, :, c], np.float64)))
                values.append(err / scale)
    return float(np.mean(values)), len(values)


def _point_case(seed=9, n=12, channels=2):
    rng = np.random.default_rng(seed)
    context = np.cumsum(rng.standard_normal((n, 48, channels)) * rng.uniform(0.2, 3.0, (n, 1, channels)), axis=1)
    target = rng.standard_normal((n, 8, channels)) * 4.0
    point = target + rng.standard_normal((n, 8, channels))
    return context.astype("float32"), target.astype("float32"), point.astype("float32")


def test_the_canonical_metrics_match_float64_oracles():
    context, target, point = _point_case()
    got = fr.canonical_metrics(context, target, point)
    t64, p64, c64 = (np.asarray(a, np.float64) for a in (target, point, context))
    mae = np.mean(np.abs(t64 - p64))
    assert abs(got["MAE"] - mae) < 1e-5 * mae
    assert abs(got["RMSE"] - np.sqrt(np.mean((t64 - p64) ** 2))) < 1e-5
    assert abs(got["sMAPE"] - 200 * np.mean(np.abs(t64 - p64) / (np.abs(t64) + np.abs(p64) + 1e-7))) < 1e-3
    naive = np.mean(np.abs(t64 - c64[:, -1:, :]))
    assert abs(got["rMAE"] - mae / naive) < 1e-5
    mase, scored = _oracle_mase(context, target, point)
    assert abs(got["MASE"] - mase) < 1e-5 * mase and got["MASE_excluded"] == 0.0 and scored == 24
    assert "coverage" not in got and "sharpness" not in got, "a point forecast has no interval"
    assert all(isinstance(v, float) for v in got.values())


def test_the_canonical_mase_scale_comes_from_the_context_never_from_the_target():
    context, target, point = _point_case(seed=10, n=8, channels=1)
    smooth_context = np.cumsum(np.full_like(context, 0.01), axis=1)   # one-step differences 0.01
    rough_target = np.cumsum(np.full_like(target, 5.0), axis=1)       # one-step differences 5.0
    got = fr.canonical_metrics(smooth_context, rough_target, point)["MASE"]
    want, _ = _oracle_mase(smooth_context, rough_target, point)
    assert abs(got - want) < 1e-5 * want
    assert got > 100, "dividing by the target's differences would give a number near 1"


def test_the_canonical_mase_is_per_window_and_not_the_batch_global_ratio():
    """One rough and one smooth window: the mean of the ratios differs from the ratio of the means."""
    rng = np.random.default_rng(11)
    smooth = np.cumsum(np.full((1, 40, 1), 0.1), axis=1)
    rough = np.cumsum(rng.choice([-5.0, 5.0], size=(1, 40, 1)), axis=1)
    context = np.concatenate([smooth, rough]).astype("float32")
    target = rng.standard_normal((2, 6, 1)).astype("float32")
    point = np.zeros_like(target)
    got = fr.canonical_metrics(context, target, point)["MASE"]
    want, _ = _oracle_mase(context, target, point)
    global_ratio = np.abs(target - point).mean() / np.abs(np.diff(context, axis=1)).mean()
    assert abs(got - want) < 1e-5 * want
    assert abs(want - global_ratio) > 0.1 * want


def test_a_flat_context_has_no_mase_scale_and_is_counted_not_hidden():
    context, target, point = _point_case(seed=12, n=6, channels=1)
    context[2] = 3.0
    context[5] = 0.0
    got = fr.canonical_metrics(context, target, point)
    want, scored = _oracle_mase(context, target, point)
    assert scored == 4 and got["MASE_excluded"] == 2.0
    assert abs(got["MASE"] - want) < 1e-5 * want and np.isfinite(got["MASE"])


def test_coverage_and_sharpness_are_the_widest_interval_over_the_last_axis_of_the_cube():
    rng = np.random.default_rng(13)
    n, h, c = 10, 6, 1
    levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    target = rng.standard_normal((n, h, c)).astype("float32")
    centre = rng.standard_normal((n, h, c, 1)).astype("float32") * 0.5
    spread = np.array([-2.0, -0.7, 0.0, 0.7, 2.0], "float32") * rng.uniform(0.5, 1.5, (n, 1, 1, 1)).astype("float32")
    cube = (centre + spread).astype("float32")
    context = rng.standard_normal((n, 30, c)).astype("float32")
    got = fr.canonical_metrics(context, target, cube[..., 2], cube, levels)
    inside = (target >= cube[..., 0]) & (target <= cube[..., -1])
    assert abs(got["coverage"] - float(inside.mean())) < 1e-6
    assert abs(got["sharpness"] - float((cube[..., -1] - cube[..., 0]).mean())) < 1e-5
    assert abs(got["coverage_nominal"] - 0.9) < 1e-9
    assert 0.0 < got["coverage"] < 1.0, "a case that covers everything or nothing cannot see a wrong axis"
    assert got["MAE"] == pytest.approx(float(np.abs(target - cube[..., 2]).mean()), rel=1e-5)


# ---------------------------------------------------------------------
# DECODE
# ---------------------------------------------------------------------


def test_decode_has_one_row_per_family_and_head():
    assert tuple(fr.DECODE) == ROWS
    assert {r for r in ROWS if fr.DECODE[r].flat_target} == set(FLAT_TARGET_ROWS)
    assert {r for r in ROWS if fr.DECODE[r].quantile} == {"prism_quantile", "tirex", "xlstm_quantile", "timestral"}


@pytest.mark.parametrize("row", ROWS)
def test_the_row_of_a_model_is_picked_from_its_family_and_head(row):
    decoder, _ = decoder_of(row)
    assert decoder.row == row
    assert (decoder.levels is not None) == fr.DECODE[row].quantile
    assert decoder.flat_target == (row in FLAT_TARGET_ROWS)


@pytest.mark.parametrize("row", MIXIN_ROWS)
def test_each_mixin_row_equals_predict_forecast(row):
    decoder, model = decoder_of(row)
    x, _ = windows()
    forecast = model.predict_forecast(x)
    point, quantiles = decoder(model(x, training=False))
    point = keras.ops.convert_to_numpy(point)
    assert point.shape == (B, H, 1)
    assert np.allclose(point.reshape(B, H, -1), np.asarray(forecast.point).reshape(B, H, -1), atol=1e-5), row
    if forecast.quantiles is None:
        assert quantiles is None and decoder.levels is None and decoder.median_index is None
    else:
        assert list(decoder.levels) == list(forecast.quantile_levels)
        quantiles = keras.ops.convert_to_numpy(quantiles)
        assert quantiles.shape == (B, H, 1, len(LEVELS))
        assert np.allclose(quantiles.reshape(B, H, -1), np.asarray(forecast.quantiles).reshape(B, H, -1), atol=1e-5)
        assert np.array_equal(point[..., 0], quantiles[:, :, 0, decoder.median_index])


def test_the_median_index_follows_each_family_s_own_rule_where_the_rules_differ():
    assert fr.decoder_for(*build_row("prism_quantile")).median_index == 1   # nearest to 0.5, the tie goes first
    assert fr.decoder_for(*build_row("tirex")).median_index == 2            # len // 2
    assert fr.decoder_for(*build_row("xlstm_quantile")).median_index == 2


def test_the_nbeats_row_returns_the_forecast_and_not_the_residual():
    decoder, model = decoder_of("nbeats")
    x, _ = windows()
    forecast, residual = model(x, training=False)
    point, quantiles = decoder([forecast, residual])
    assert quantiles is None and keras.ops.convert_to_numpy(point).shape == (B, H, 1)
    assert np.array_equal(keras.ops.convert_to_numpy(point), keras.ops.convert_to_numpy(forecast))
    assert tuple(residual.shape) == (B, L), "the residual has the context's shape: reading it is not a forecast"


def test_the_timestral_row_equals_the_forecaster_with_the_two_named_options_off():
    from dl_techniques.models.time_series.timestral.forecaster import TimestralForecaster

    decoder, model = decoder_of("timestral")
    x, _ = windows()
    point, quantiles = decoder(model(x, training=False))
    quantiles = keras.ops.convert_to_numpy(quantiles)
    assert quantiles.shape == (B, H, 1, model.config.num_quantiles)
    reference = TimestralForecaster(model).forecast(
        x[:, :, 0], H, symmetric_averaging=False, clip_nonnegative=False).quantiles   # (B, V, H, Q)
    assert np.array_equal(quantiles[:, :, 0, :], np.asarray(reference)[:, 0]), "equal to the last bit at horizon <= patch size"
    assert decoder.median_index == model.config.median_index
    assert tuple(decoder.levels) == tuple(model.config.quantile_knots)
    assert np.array_equal(keras.ops.convert_to_numpy(point)[..., 0], quantiles[:, :, 0, decoder.median_index])


def test_the_mdn_row_is_the_mixture_mean_of_a_float64_oracle():
    decoder, model = decoder_of("mdn")
    layer = model.get_layer("mdn").mdn_layer
    x, _ = windows()
    packed = keras.ops.convert_to_numpy(model(x, training=False))
    point, quantiles = decoder(packed)
    k, d = layer.num_mix, layer.output_dim
    assert quantiles is None and d == H
    mu = packed[:, :k * d].astype(np.float64).reshape(B, k, d)
    logits = packed[:, 2 * k * d:].astype(np.float64)
    pi = np.exp(logits - logits.max(axis=1, keepdims=True))
    pi /= pi.sum(axis=1, keepdims=True)
    oracle = (mu * pi[:, :, None]).sum(axis=1)[..., None]
    assert keras.ops.convert_to_numpy(point).shape == (B, H, 1)
    assert np.allclose(keras.ops.convert_to_numpy(point), oracle, atol=1e-5, rtol=0)
    assert np.abs(oracle - mu[:, 0, :, None]).max() > 1e-4, "a mixture mean is not the first component's mean"


def test_a_family_with_no_row_is_refused_by_name():
    with pytest.raises(ConfigError, match="no decode row"):
        fr.decoder_for("deepar", build_row("ets")[1])


# ---------------------------------------------------------------------
# the compiled metric and the losses
# ---------------------------------------------------------------------


def _quantile_cube(seed=14):
    rng = np.random.default_rng(seed)
    cube = np.sort(rng.standard_normal((B, H, 1, 4)), axis=-1).astype("float32")
    target = rng.standard_normal((B, H, 1)).astype("float32")
    return cube, target


def test_the_decoded_mae_of_a_quantile_cube_is_the_numpy_mae_of_the_median_index():
    decoder, _ = decoder_of("prism_quantile")
    cube, target = _quantile_cube()
    metric = fr.DecodedMAE(decoder)
    metric.update_state(target, cube)
    median = decoder.median_index
    assert metric.name == "mae"
    assert float(metric.result()) == pytest.approx(float(np.abs(target - cube[..., median]).mean()), rel=1e-6)
    other = (median + 1) % cube.shape[-1]
    assert abs(float(np.abs(target - cube[..., other]).mean()) - float(metric.result())) > 1e-3, "the index is seen"


def test_the_decoded_mae_reads_a_flat_target_and_accumulates_over_batches():
    decoder, _ = decoder_of("tirex")
    rng = np.random.default_rng(15)
    cube = np.sort(rng.standard_normal((2 * B, H, 4)), axis=-1).astype("float32")
    target = rng.standard_normal((2 * B, H)).astype("float32")
    metric = fr.DecodedMAE(decoder)
    assert float(metric.result()) == 0.0
    metric.update_state(target[:B], cube[:B])
    metric.update_state(target[B:], cube[B:])
    want = float(np.abs(target - cube[..., decoder.median_index]).mean())
    assert float(metric.result()) == pytest.approx(want, rel=1e-6)
    metric.reset_state()
    assert float(metric.result()) == 0.0


def test_the_decoded_mae_of_a_point_head_equals_the_stock_mae():
    decoder, _ = decoder_of("ets")
    rng = np.random.default_rng(16)
    out = rng.standard_normal((B, H, 1)).astype("float32")
    target = rng.standard_normal((B, H, 1)).astype("float32")
    ours, stock = fr.DecodedMAE(decoder), keras.metrics.MeanAbsoluteError()
    ours.update_state(target, out)
    stock.update_state(target, out)
    assert float(ours.result()) == pytest.approx(float(stock.result()), rel=1e-6)


@pytest.mark.parametrize("row", ROWS)
def test_every_row_compiles_to_the_metric_key_mae_and_the_history_key_val_mae(row):
    family, compiled = _fresh(row)
    decoder = fr.decoder_for(family, compiled)
    parts = fr.build_loss_and_metrics(decoder, compiled)
    compiled.compile(optimizer=keras.optimizers.AdamW(1e-3), loss=parts.loss, metrics=parts.metrics,
                     jit_compile=False)
    x, y = windows(2 * B)
    target = y[..., 0] if decoder.flat_target else y
    history = compiled.fit(x, target, batch_size=B, epochs=1, validation_data=(x, target), verbose=0)
    assert set(history.history) == {"loss", "mae", "val_loss", "val_mae"}, row
    assert np.isfinite(history.history["loss"]).all() and np.isfinite(history.history["val_mae"]).all()
    scores = compiled.evaluate(x, target, batch_size=B, verbose=0, return_dict=True)
    point = keras.ops.convert_to_numpy(decoder(compiled.predict(x, batch_size=B, verbose=0))[0])
    assert scores["mae"] == pytest.approx(float(np.abs(y - point).mean()), rel=1e-4), row


def _fresh(row):
    """A model of the row with its own weights (fit changes them; the cache holds the untouched one)."""
    _CACHE.pop(row, None)
    family, model = build_row(row)
    _CACHE.pop(row, None)
    return family, model


def test_the_point_losses_are_stock_losses_and_the_nbeats_loss_is_a_list():
    decoder, model = decoder_of("ets")
    for name, cls in (("mae", keras.losses.MeanAbsoluteError), ("mse", keras.losses.MeanSquaredError)):
        assert type(fr.build_loss_and_metrics(decoder, model, name).loss) is cls
    assert fr.build_loss_and_metrics(decoder, model, None).loss.__class__ is keras.losses.MeanAbsoluteError
    from dl_techniques.losses.mase_loss import MASELoss

    assert type(fr.build_loss_and_metrics(decoder, model, "mase").loss) is MASELoss
    nb_decoder, nb_model = decoder_of("nbeats")
    parts = fr.build_loss_and_metrics(nb_decoder, nb_model)
    assert isinstance(parts.loss, list) and len(parts.loss) == 2 and parts.loss[1] is None
    assert len(parts.metrics) == 2 and parts.metrics[1] == [] and len(parts.metrics[0]) == 1


@pytest.mark.parametrize("row", ("prism_quantile", "tirex", "xlstm_quantile", "timestral", "mdn"))
def test_a_loss_name_is_refused_for_a_head_whose_loss_is_not_chosen(row):
    decoder, model = decoder_of(row)
    for name in ("mae", "mse", "mase"):
        with pytest.raises(ConfigError, match="loss_args.name"):
            fr.build_loss_and_metrics(decoder, model, name)
    with pytest.raises(ConfigError, match="loss_args.name"):
        fr.build_loss_and_metrics(*decoder_of("ets"), "huber")


def test_the_quantile_loss_takes_a_flat_target_for_tirex_and_xlstm_and_the_window_target_for_prism():
    from dl_techniques.losses.quantile_loss import QuantileLoss

    rng = np.random.default_rng(17)
    for row in ("tirex", "xlstm_quantile", "prism_quantile"):
        decoder, model = decoder_of(row)
        loss = fr.build_loss_and_metrics(decoder, model).loss
        assert type(loss) is QuantileLoss and loss.quantiles == list(LEVELS)
        shape = (B, H, 1, len(LEVELS)) if row == "prism_quantile" else (B, H, len(LEVELS))
        pred = rng.standard_normal(shape).astype("float32")
        y = rng.standard_normal((B, H)).astype("float32")
        target = y[..., None] if row == "prism_quantile" else y
        assert decoder.flat_target == (row != "prism_quantile")
        pinball = np.stack([np.maximum(q * (y[..., None] - pred.reshape(B, H, -1)[..., i:i + 1]),
                                       (q - 1) * (y[..., None] - pred.reshape(B, H, -1)[..., i:i + 1]))
                            for i, q in enumerate(LEVELS)], axis=-1)
        assert float(loss(target, pred)) == pytest.approx(float(pinball.mean()), rel=1e-5), row


def test_mdn_trains_on_its_own_negative_log_likelihood_and_timestral_on_none():
    decoder, model = decoder_of("mdn")
    assert fr.build_loss_and_metrics(decoder, model).loss == model.get_layer("mdn").mdn_layer.loss_func
    decoder, model = decoder_of("timestral")
    assert fr.build_loss_and_metrics(decoder, model).loss is None


# ---------------------------------------------------------------------
# the post-hoc report and the figure
# ---------------------------------------------------------------------


def _dataset(x, y, flat=False):
    import tensorflow as tf

    return tf.data.Dataset.from_tensor_slices((x, y[..., 0] if flat else y)).batch(B, drop_remainder=True)


@pytest.mark.parametrize("row", ("ets", "prism_quantile", "tirex"))
def test_the_report_scores_the_windows_of_a_dataset_through_the_decode_row(row):
    decoder, model = decoder_of(row)
    x, y = windows(2 * B, seed=18)
    report = fr.canonical_report(model, decoder, _dataset(x, y, decoder.flat_target))
    cx, cy, point, quantiles = fr.decode_dataset(model, decoder, _dataset(x, y, decoder.flat_target))
    assert np.array_equal(cx, x) and np.array_equal(cy, y)
    assert point.shape == (2 * B, H, 1)
    want = fr.canonical_metrics(x, y, point, quantiles, decoder.levels)
    assert report == want
    assert ("coverage" in report) == (decoder.levels is not None)


def test_the_figure_is_written_with_and_without_a_band(tmp_path):
    rng = np.random.default_rng(19)
    x, y = windows(8, seed=20)
    point = y + rng.standard_normal(y.shape).astype("float32") * 0.1
    cube = np.sort(rng.standard_normal((8, H, 1, 3)), axis=-1).astype("float32")
    plain = fr.plot_forecasts(tmp_path / "point.png", x, y, point, count=4, title="point head")
    banded = fr.plot_forecasts(tmp_path / "band.png", x, y, point, cube, (0.1, 0.5, 0.9), count=20)
    for path in (plain, banded):
        assert Path(path).is_file() and Path(path).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert Path(plain).stat().st_size != Path(banded).stat().st_size, "the band changes the picture"
    two_channels = fr.plot_forecasts(tmp_path / "c2.png", np.repeat(x, 2, axis=-1), np.repeat(y, 2, axis=-1),
                                     np.repeat(point, 2, axis=-1), count=2)
    assert Path(two_channels).is_file()


# =====================================================================
# Second part: the argument groups (Step 10)
# =====================================================================

RECIPE_DATA = {"lookback": 32, "horizon": 8}
RECIPE = {"data": RECIPE_DATA, "metric_args": {"watch_metric": fa.WATCH_METRIC}}


def context(recipe=None, family="nbeats", dataset=fa.SYNTHETIC) -> AutoContext:
    """What the resolver passes: no class count, no image size, the recipe's tables under ``defaults``."""
    return AutoContext(family=family, variant="base", dataset=dataset, num_classes=None, image_size=None,
                       defaults=RECIPE if recipe is None else recipe)


def data(**given) -> fa.TimeSeriesData:
    return fa.load_forecast_data_config(given)


def resolved(**given) -> fa.TimeSeriesData:
    return data(**given).resolve_auto(context())


def write_csv(path: Path, rows: int, columns=("a", "b"), time_col=None, extra=None) -> str:
    """A csv of ``rows`` rows of smooth values; ``extra`` replaces cells as ``{(row, column): text}``."""
    cells = {(r, c): f"{np.sin(r / 7.0 + k):.6f}" for r in range(rows) for k, c in enumerate(columns)}
    cells.update(extra or {})
    names = ([time_col] if time_col else []) + list(columns)
    lines = [",".join(names)]
    for r in range(rows):
        lines.append(",".join(([cells.get((r, time_col), str(r))] if time_col else []) + [cells[(r, c)] for c in columns]))
    path.write_text("\n".join(lines) + "\n")
    return str(path)


# ---------------------------------------------------------------------
# the groups parse strictly
# ---------------------------------------------------------------------

GROUPS = (fa.TimeSeriesData, fa.ForecastTransformArgs, fa.ForecastMetricArgs, fa.ForecastLossArgs)


def test_every_group_field_has_help_text_and_the_data_fields_are_the_loaders_keywords():
    for group in GROUPS:
        for field in dataclasses.fields(group):
            assert field.metadata.get("help"), f"{group.__name__}.{field.name} has no help text"
    from train.common.forecast_data import load_forecast_batches

    keywords = set(inspect.signature(load_forecast_batches).parameters)
    fields = {f.name for f in dataclasses.fields(fa.TimeSeriesData)}
    # the loader takes every data key but `dataset` (it only checks it is `synthetic`), and the four keys the run
    # gives it from the rest of the config: the batch, the two seeds and the transform's floor
    assert fields - {"dataset"} <= keywords
    assert keywords - fields == {"batch_size", "seed", "stream_seed", "scale_floor"}
    assert {f.name for f in dataclasses.fields(fa.ForecastTransformArgs)} == {"scale_floor"}


def test_the_documented_defaults():
    d = fa.TimeSeriesData()
    assert (d.format, d.dataset, d.path, d.value_cols, d.time_col) == ("builtin", "synthetic", None, None, None)
    assert (d.lookback, d.horizon) == ("auto", "auto"), "the two numbers live in the family recipe, not here"
    assert (d.realizations, d.series_length, d.noise_level, d.patterns) == (4, 10_000, 0.1, None)
    assert (d.select_fraction, d.test_fraction) == (0.15, 0.15)
    assert (d.n_select, d.n_test, d.steps_per_epoch, d.max_samples) == (None,) * 4
    assert fa.ForecastTransformArgs().scale_floor == 1e-3
    assert fa.ForecastLossArgs().name is None
    assert fa.TimeSeriesData().problems("data") == []


@pytest.mark.parametrize("group", GROUPS)
def test_an_unknown_key_is_refused_with_a_suggestion(group):
    near = {fa.TimeSeriesData: "lookbak", fa.ForecastTransformArgs: "scale_flor", fa.ForecastMetricArgs: "watch_metrik",
            fa.ForecastLossArgs: "nam"}[group]
    with pytest.raises(ConfigValidationError, match=f"Unknown key '{near}'") as raised:
        validate(group, {near: 1})
    assert "Did you mean" in str(raised.value)
    if group is fa.TimeSeriesData:  # and the loader a command line and a file go through refuses it the same way
        with pytest.raises(ConfigValidationError, match=f"Unknown key '{near}'"):
            fa.load_forecast_data_config({"horizon": 4, near: 1})


@pytest.mark.parametrize(
    "group, given",
    [
        (fa.TimeSeriesData, {"lookback": "long"}),
        (fa.TimeSeriesData, {"lookback": 2.5}),
        (fa.TimeSeriesData, {"horizon": True}),
        (fa.TimeSeriesData, {"realizations": 2.5}),
        (fa.TimeSeriesData, {"noise_level": "high"}),
        (fa.TimeSeriesData, {"patterns": "sine"}),
        (fa.TimeSeriesData, {"patterns": [1]}),
        (fa.TimeSeriesData, {"value_cols": "a"}),
        (fa.TimeSeriesData, {"path": 3}),
        (fa.TimeSeriesData, {"format": "parquet"}),
        (fa.TimeSeriesData, {"steps_per_epoch": "many"}),
        (fa.ForecastTransformArgs, {"scale_floor": "small"}),
        (fa.ForecastMetricArgs, {"watch_mode": "up"}),
        (fa.ForecastLossArgs, {"name": 3}),
        (fa.ForecastLossArgs, {"name": "huber"}),
    ],
)
def test_a_wrong_type_or_choice_is_refused_naming_its_key(group, given):
    (key,) = given
    with pytest.raises(ConfigValidationError, match=key):
        validate(group, given)


@pytest.mark.parametrize(
    "given, key",
    [
        ({"lookback": 0}, "lookback"),
        ({"horizon": -3}, "horizon"),
        ({"realizations": 0}, "realizations"),
        ({"series_length": 0}, "series_length"),
        ({"noise_level": -0.1}, "noise_level"),
        ({"select_fraction": 0.0}, "select_fraction"),
        ({"test_fraction": 1.0}, "test_fraction"),
        ({"select_fraction": 0.5, "test_fraction": 0.5}, "select_fraction"),
        ({"n_select": 0}, "n_select"),
        ({"n_test": -1}, "n_test"),
        ({"steps_per_epoch": 0}, "steps_per_epoch"),
        ({"max_samples": 0}, "max_samples"),
        ({"patterns": []}, "patterns"),
        ({"dataset": "etth1"}, "dataset"),
        ({"path": "x.csv"}, "path"),
        ({"time_col": "t"}, "time_col"),
        ({"value_cols": ["a"]}, "value_cols"),
        ({"format": "csv", "path": "x.csv", "realizations": 8}, "realizations"),
        ({"format": "csv", "path": "x.csv", "patterns": ["sine"]}, "patterns"),
        ({"format": "csv", "path": "x.csv", "series_length": 500}, "series_length"),
        ({"format": "csv", "path": "x.csv", "noise_level": 0.2}, "noise_level"),
        ({"format": "csv", "path": "x.csv", "value_cols": []}, "value_cols"),
    ],
)
def test_the_data_group_names_every_option_out_of_range_at_its_dotted_path(given, key):
    (line,) = fa.TimeSeriesData(**given).problems("data")
    assert f"'data.{key}'" in line
    with pytest.raises(ConfigValidationError, match=f"'{key}'"):
        fa.load_forecast_data_config(given)


def test_a_key_of_the_other_source_is_refused_but_the_defaults_of_both_pass():
    assert fa.TimeSeriesData(format="csv", path="x.csv").problems("data") == []
    assert fa.TimeSeriesData(format="csv", path="x.csv", realizations=4, series_length=10_000).problems("data") == []
    assert fa.TimeSeriesData(patterns=["sine_wave"], realizations=2, series_length=999).problems("data") == []


def test_the_transform_and_loss_groups():
    assert fa.ForecastTransformArgs(scale_floor=0.0).problems("transform_args") == []
    (line,) = fa.ForecastTransformArgs(scale_floor=-1.0).problems("transform_args")
    assert "'transform_args.scale_floor'" in line
    for name in (None, "mae", "mse", "mase"):
        assert validate(fa.ForecastLossArgs, {"name": name}).name == name
    assert fa.LOSS_NAMES == fr.LOSS_NAMES


# ---------------------------------------------------------------------
# the loader layers a file under the keys
# ---------------------------------------------------------------------


def test_the_data_config_loads_from_a_mapping_an_instance_and_a_yaml_file(tmp_path):
    assert fa.load_forecast_data_config({}) == fa.TimeSeriesData()
    loaded = fa.load_forecast_data_config({"lookback": 48, "patterns": ["sine_wave"], "max_samples": 64})
    assert (loaded.lookback, loaded.patterns, loaded.max_samples, loaded.format) == (48, ["sine_wave"], 64, "builtin")
    assert fa.load_forecast_data_config(loaded) is loaded
    path = tmp_path / "data.yaml"
    path.write_text("lookback: 24\nhorizon: 6\nselect_fraction: 0.2\n")
    from_file = fa.load_forecast_data_config(str(path))
    assert (from_file.lookback, from_file.horizon, from_file.select_fraction, from_file.realizations) == (24, 6, 0.2, 4)
    path.write_text("lookbak: 24\n")
    with pytest.raises(ConfigValidationError, match="lookbak") as raised:
        fa.load_forecast_data_config(path)
    assert str(path) in str(raised.value) and "Did you mean" in str(raised.value)
    path.write_text("lookback: 0\n")
    with pytest.raises(ConfigValidationError, match="lookback") as raised:
        fa.load_forecast_data_config(path)
    assert str(path) in str(raised.value)
    with pytest.raises(ConfigError, match="mapping"):
        fa.load_forecast_data_config(3)
    with pytest.raises(ConfigError):
        fa.load_forecast_data_config(tmp_path / "absent.yaml")


def test_a_relative_csv_path_in_a_data_file_is_anchored_to_the_file(tmp_path):
    folder = tmp_path / "cfg"
    folder.mkdir()
    (folder / "d.yaml").write_text("format: csv\npath: series.csv\ntime_col: t\nvalue_cols: [a, b]\n")
    loaded = fa.load_forecast_data_config(str(folder / "d.yaml"))
    assert loaded.path == str(folder / "series.csv")
    assert (loaded.time_col, loaded.value_cols, loaded.format) == ("t", ["a", "b"], "csv")
    (folder / "abs.yaml").write_text(f"format: csv\npath: {tmp_path / 'elsewhere.csv'}\n")
    assert fa.load_forecast_data_config(str(folder / "abs.yaml")).path == str(tmp_path / "elsewhere.csv")


# ---------------------------------------------------------------------
# the recipe fills lookback and horizon
# ---------------------------------------------------------------------


def test_the_recipe_fills_the_window_and_an_explicit_value_wins():
    both = resolved()
    assert (both.lookback, both.horizon) == (32, 8)
    mine = resolved(lookback=64)
    assert (mine.lookback, mine.horizon) == (64, 8)
    explicit = resolved(lookback=20, horizon=5)
    assert (explicit.lookback, explicit.horizon) == (20, 5)
    other = {key: getattr(both, key) for key in ("format", "realizations", "series_length", "select_fraction")}
    assert other == {"format": "builtin", "realizations": 4, "series_length": 10_000, "select_fraction": 0.15}


def test_a_recipe_without_the_window_is_refused_by_name_and_a_typo_in_it_names_the_group():
    for missing in ({}, {"data": {"horizon": 8}}):
        with pytest.raises(ConfigError, match="'data.lookback'.*the 'nbeats' recipe has no lookback for 'synthetic'"):
            fa.TimeSeriesData().resolve_auto(context(missing))
    with pytest.raises(ConfigError, match="'data.horizon'.*the 'nbeats' recipe has no horizon for 'synthetic'"):
        fa.TimeSeriesData(lookback=16).resolve_auto(context({"data": {"lookback": 16}}))
    assert fa.TimeSeriesData(lookback=16, horizon=4).resolve_auto(context({})).horizon == 4
    with pytest.raises(ConfigError, match="lookbak"):
        fa.TimeSeriesData().resolve_auto(context({"data": {"lookbak": 16}}))
    with pytest.raises(ConfigError, match="data"):
        fa.TimeSeriesData().resolve_auto(context({"data": {"lookback": "long", "horizon": 8}}))


def test_the_facts_carry_no_class_and_no_image_size():
    facts = fa.forecast_facts(data())
    assert (facts.dataset, facts.num_classes, facts.image_size) == ("synthetic", None, None)
    assert fa.forecast_facts(data(format="csv", path="whatever.csv")).dataset == "synthetic"


# ---------------------------------------------------------------------
# every refusal that needs no tensor, in a fresh interpreter that must not have imported TensorFlow
# ---------------------------------------------------------------------

CHILD = """
import json, sys
from train.api import forecast_args as fa
from train.api.args import AutoContext
from train.api.errors import ConfigError

spec = json.loads(sys.argv[1])
refused = None
try:
    loaded = fa.load_forecast_data_config(spec["data"])
    context = AutoContext(family="nbeats", variant="base", dataset="synthetic", num_classes=None, image_size=None,
                          defaults={"data": spec["recipe"]})
    loaded.resolve_auto(context)
    fa.check_forecast_data(loaded.resolve_auto(context))
except ConfigError as exc:
    refused = [type(exc).__name__, str(exc)]
heavy = sorted(name for name in ("tensorflow", "keras", "train.common", "dl_techniques") if name in sys.modules)
print("@@" + json.dumps({"refused": refused, "heavy": heavy}))
"""


def fresh_check(tmp_path, given: dict, recipe=None):
    """``(refused, heavy)``: the pre-claim check of ``given`` run in a fresh interpreter, and what it imported."""
    env = {"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "", "HOME": str(tmp_path)}
    done = subprocess.run([sys.executable, "-c", CHILD, json.dumps({"data": given, "recipe": recipe or RECIPE_DATA})],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    answer = json.loads(done.stdout.rsplit("@@", 1)[1])
    return answer["refused"], answer["heavy"]


def refusal_cases(tmp_path):
    """``{name: (data mapping, text the refusal must contain)}``; every file is written under ``tmp_path``."""
    good = write_csv(tmp_path / "good.csv", 400)
    return {
        "a csv that does not exist": ({"format": "csv", "path": str(tmp_path / "absent.csv")}, "does not exist"),
        "no csv path at all": ({"format": "csv"}, "data.path"),
        "a missing value column": ({"format": "csv", "path": good, "value_cols": ["a", "nope"]}, "'nope' is missing"),
        "a missing time column": ({"format": "csv", "path": good, "time_col": "when"}, "'when' is missing"),
        "an unsorted time column": ({"format": "csv", "time_col": "t", "path": write_csv(
            tmp_path / "u.csv", 400, time_col="t", extra={(5, "t"): "2"})}, "strictly increasing"),
        "a lookback below 1": ({"lookback": 0}, "'lookback'"),
        "a horizon below 1": ({"horizon": 0}, "'horizon'"),
        "fractions that leave no train segment": ({"select_fraction": 0.5, "test_fraction": 0.5}, "leave a train segment"),
        "a segment shorter than lookback + horizon (builtin)": ({"series_length": 200, "lookback": 64, "horizon": 16}, "too short"),
        "a segment shorter than lookback + horizon (csv)": ({"format": "csv", "path": write_csv(tmp_path / "s.csv", 100),
                                                              "lookback": 48, "horizon": 16}, "too short"),
        "the test segment is the short one": ({"series_length": 300, "select_fraction": 0.5, "test_fraction": 0.1,
                                                "lookback": 30, "horizon": 10}, "its test segment has 30 rows"),
        "a dataset that is not the synthetic bank": ({"dataset": "etth1"}, "'dataset'"),
        "a non-numeric cell": ({"format": "csv", "path": write_csv(tmp_path / "n.csv", 400, extra={(3, "b"): "abc"})}, "not numeric"),
        "a non-finite cell": ({"format": "csv", "path": write_csv(tmp_path / "f.csv", 400, extra={(3, "b"): "nan"})}, "non-finite"),
    }


CASE_NAMES = (
    "a csv that does not exist", "no csv path at all", "a missing value column", "a missing time column",
    "an unsorted time column", "a lookback below 1", "a horizon below 1", "fractions that leave no train segment",
    "a segment shorter than lookback + horizon (builtin)", "a segment shorter than lookback + horizon (csv)",
    "the test segment is the short one", "a dataset that is not the synthetic bank", "a non-numeric cell",
    "a non-finite cell",
)


@pytest.mark.parametrize("name", CASE_NAMES)
def test_every_refusal_that_needs_no_tensor_is_named_and_loads_neither_tensorflow_nor_keras(tmp_path, name):
    cases = refusal_cases(tmp_path)
    assert tuple(cases) == CASE_NAMES
    given, text = cases[name]
    refused, heavy = fresh_check(tmp_path, given)
    assert refused is not None, f"{name}: the configuration was accepted"
    kind, message = refused
    assert kind in ("ConfigError", "ConfigValidationError")
    if text is not None:
        assert text in message, message
    assert heavy == [], f"{name}: the check imported {heavy}"


def test_a_good_configuration_passes_the_same_checks_without_tensorflow(tmp_path):
    good = write_csv(tmp_path / "ok.csv", 400, time_col="t")
    for given in ({}, {"series_length": 400, "lookback": 32, "horizon": 8},
                  {"format": "csv", "path": good, "time_col": "t", "value_cols": ["a", "b"]}):
        refused, heavy = fresh_check(tmp_path, given)
        assert refused is None, refused
        assert heavy == []


def test_the_csv_checks_run_through_the_one_reader_the_loader_uses(tmp_path):
    # no second reader: a refusal here is the loader's own text, so the two cannot drift apart
    from train.api import forecast_series

    path = write_csv(tmp_path / "x.csv", 400, extra={(3, "b"): "nan"})
    with pytest.raises(ConfigError) as in_check:
        fa.check_forecast_data(fa.TimeSeriesData(format="csv", path=path, lookback=32, horizon=8))
    with pytest.raises(ConfigError) as in_reader:
        forecast_series.read_csv_series(path, None, None)
    assert str(in_check.value) == str(in_reader.value)
    source = (SRC / "train" / "api" / "forecast_args.py").read_text()
    assert "read_csv(" not in source and "import pandas" not in source


#: The segment-too-short sentence for a 200-row synthetic series at lookback 64, horizon 16 (its select segment has
#: 30 rows against 80 needed). Pinned byte for byte: it is what both the pre-claim check and the loader say.
SEGMENT_SENTENCE = (
    "a synthetic series of 200 rows is too short: its select segment has 30 rows and one window needs "
    "lookback + horizon = 80; give it more rows or lower data.lookback, data.horizon or the fractions"
)


def _strings_and_calls(module):
    tree = ast.parse(Path(inspect.getsourcefile(module)).read_text())
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    calls = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    return strings, calls


def test_the_segment_check_is_one_function_both_callers_use():
    # D-037: the pre-claim check and the loader hold ONE check_segments (train.api.forecast_series); neither spells
    # the sentence or loops over the segments itself, so the two cannot say different things
    from train.api import forecast_series as fs
    from train.common import forecast_data as fd

    assert fa.check_segments is fs.check_segments and fd.check_segments is fs.check_segments
    for module in (fa, fd):
        strings, calls = _strings_and_calls(module)
        assert [s for s in strings if "rows is too short" in s or "one window needs" in s] == [], module.__name__
        assert "check_segments" in calls, module.__name__
    assert not hasattr(fd, "_check_segments")
    with pytest.raises(ConfigError) as err:
        fs.check_segments(fs.split_bounds(200, 0.15, 0.15), 200, 80, "a synthetic series")
    assert str(err.value) == SEGMENT_SENTENCE
    fs.check_segments(fs.split_bounds(1000, 0.15, 0.15), 1000, 80, "a synthetic series")


def test_the_segment_refusal_is_the_loaders_wording():
    # the loader raises the same sentence after the claim, the check before it: byte-equal, and the pinned one
    from train.common.forecast_data import load_forecast_batches

    kwargs = dict(format="builtin", dataset="synthetic", path=None, value_cols=None, time_col=None, lookback=64,
                  horizon=16, realizations=1, series_length=200, noise_level=0.1, patterns=["random_walk"],
                  select_fraction=0.15, test_fraction=0.15, n_select=None, n_test=None, steps_per_epoch=None,
                  max_samples=None, batch_size=4, seed=1, stream_seed=None, scale_floor=1e-3)
    with pytest.raises(ConfigError) as in_loader:
        load_forecast_batches(**kwargs)
    with pytest.raises(ConfigError) as in_check:
        fa.check_forecast_data(fa.TimeSeriesData(series_length=200, lookback=64, horizon=16))
    assert str(in_check.value) == str(in_loader.value) == SEGMENT_SENTENCE


def test_an_unresolved_window_is_refused_not_added():
    with pytest.raises(ConfigError, match="data.lookback is 'auto'"):
        fa.check_forecast_data(fa.TimeSeriesData())


def test_the_module_imports_nothing_heavy_at_module_scope():
    tree = ast.parse((SRC / "train" / "api" / "forecast_args.py").read_text())
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(("." * node.level) + (node.module or ""))
    assert not [n for n in names if n.split(".")[0] in ("tensorflow", "keras", "numpy", "pandas", "dl_techniques")
                or n.startswith("train.common") or n == "train"], names
    assert all(n.startswith(".") or n in ("dataclasses", "pathlib", "typing") for n in names), names


# ---------------------------------------------------------------------
# the watched metric and the loss
# ---------------------------------------------------------------------


def test_the_metric_group_watches_val_mae_and_minimises_it():
    got = fa.ForecastMetricArgs().resolve_auto(context())
    assert (got.watch_metric, got.watch_mode) == ("val_mae", "min")
    assert fa.ForecastMetricArgs(watch_metric="val_mae").resolve_auto(context()).watch_mode == "min"
    assert fa.ForecastMetricArgs(watch_metric="val_mae", watch_mode="min").resolve_auto(context()).watch_mode == "min"
    from train.common.callbacks import resolve_monitor_mode

    assert resolve_monitor_mode(fa.WATCH_METRIC) == fa.WATCH_MODE == "min", "the shared rule says val_mae is minimised"


@pytest.mark.parametrize(
    "given, key",
    [
        ({"watch_metric": "val_loss"}, "watch_metric"),
        ({"watch_metric": "val_mse"}, "watch_metric"),
        ({"watch_metric": "mae"}, "watch_metric"),
        ({"watch_metric": "val_mae", "watch_mode": "max"}, "watch_mode"),
        ({"watch_mode": "max"}, "watch_mode"),
    ],
)
def test_the_metric_group_refuses_what_the_bar_cannot_describe(given, key):
    (line,) = fa.ForecastMetricArgs(**given).problems("metric_args")
    assert f"'metric_args.{key}'" in line and "val_mae" in line and "min" in line
    with pytest.raises(ConfigValidationError, match=key):
        validate(fa.ForecastMetricArgs, given)
    for allowed in ({}, {"watch_metric": "val_mae"}, {"watch_mode": "min"}):
        assert fa.ForecastMetricArgs(**allowed).problems("metric_args") == [], allowed


def test_the_metric_watched_is_the_key_the_compiled_metric_writes():
    # the metric is compiled under the name `mae`; Keras prefixes the validation copy with `val_`
    name = inspect.signature(fr.DecodedMAE.__init__).parameters["name"].default
    assert fa.WATCH_METRIC == "val_" + name


# ---------------------------------------------------------------------
# Step 11: the family adapters, tables, refusals and recipes
# ---------------------------------------------------------------------

FAMILIES = tuple(fm.FAMILIES)
TABLES = ("ALLOWED", "REFUSED", "FORCED", "SUPPLIED", "FACTORIES", "RECIPES", "FAMILIES")


def test_every_table_has_exactly_the_seven_families_and_no_key_is_in_two_tables():
    assert FAMILIES == ("nbeats", "prism", "tirex", "xlstm", "ets", "timestral", "mdn")
    for name in TABLES:
        assert tuple(getattr(fm, name)) == FAMILIES, name
    for family in FAMILIES:
        classes = [set(fm.ALLOWED[family]), set(fm.REFUSED[family]), set(fm.FORCED[family]), set(fm.SUPPLIED[family])]
        assert sum(len(c) for c in classes) == len(set().union(*classes)), f"{family}: a key sits in two tables"
    assert {f for f, rows in fm.FORCED.items() if rows} == {"nbeats", "tirex", "xlstm"}
    assert all(rows == {"use_normalization": (False, fm._ONE_NORM)} for rows in fm.FORCED.values() if rows)
    assert set(fm.UNSUPPORTED) == {"deepar", "adaptive_ema"} and not set(fm.UNSUPPORTED) & set(FAMILIES)


@pytest.mark.parametrize("family", FAMILIES)
def test_the_tables_equal_the_factory_signatures(family):
    assert fm.signature_census(family) == {"unclassified": [], "stale": []}
    signatures = fm.factory_signatures(family)
    assert all(names for names in signatures.values()), "a factory with no parameters: the census would be vacuous"


@pytest.mark.parametrize("family", FAMILIES)
def test_the_census_names_a_new_factory_parameter_and_a_key_no_factory_has(family):
    signatures = fm.factory_signatures(family)
    first = next(iter(signatures))
    grown = {**signatures, first: [*signatures[first], "a_parameter_added_upstream"]}
    assert fm.signature_census(family, grown) == {"unclassified": ["a_parameter_added_upstream"], "stale": []}
    offered = sorted({*fm.ALLOWED[family], *fm.REFUSED[family], *fm.FORCED[family], *fm.SUPPLIED[family]})
    gone = offered[0]
    shrunk = {path: [n for n in names if n != gone] for path, names in signatures.items()}
    assert fm.signature_census(family, shrunk) == {"unclassified": [], "stale": [gone]}


def test_the_timestral_patch_constant_is_the_librarys_default():
    from dl_techniques.models.time_series.timestral.model import TimestralModel

    assert inspect.signature(TimestralModel.from_variant).parameters["patch_size"].default == fm.TIMESTRAL_PATCH


def build_adapter(family, variant, channels=1, **model_args):
    extra = {"batch_size": B} if family == "timestral" else {}
    return getattr(fm, f"create_{family}")(variant=variant, input_shape=(L, channels), horizon=H, **extra, **model_args)


#: (family, variant, channels, model_args, quantile levels expected in the decode)
ADAPTER_CASES = {
    "nbeats": ("nbeats", "tiny", 1, {}, None),
    "nbeats two channels": ("nbeats", "tiny", 2, {}, None),
    "prism point": ("prism", "tiny", 1, {}, None),
    "prism quantile": ("prism", "tiny", 1, {"quantile_levels": list(LEVELS)}, len(LEVELS)),
    "prism two channels": ("prism", "tiny", 2, {}, None),
    "tirex": ("tirex", "tiny", 1, {"quantile_levels": list(LEVELS)}, len(LEVELS)),
    "xlstm point": ("xlstm", "tiny", 1, {}, None),
    "xlstm quantile": ("xlstm", "tiny", 1, {"quantile_levels": list(LEVELS)}, len(LEVELS)),
    "xlstm two channels": ("xlstm", "tiny", 2, {}, None),
    "ets": ("ets", "ANN", 1, {}, None),
    "ets seasonal": ("ets", "AAA", 1, {"seasonal_period": 8}, None),
    "timestral": ("timestral", "5M", 1, {}, 21),
    "timestral two channels": ("timestral", "5M", 2, {}, 21),
    "mdn": ("mdn", "tiny", 1, {}, None),
}


@pytest.mark.parametrize("case", list(ADAPTER_CASES))
def test_every_adapter_builds_a_model_the_decode_row_reads(case):
    family, variant, channels, model_args, levels = ADAPTER_CASES[case]
    model = build_adapter(family, variant, channels, **model_args)
    assert model.built
    x = np.random.default_rng(3).standard_normal((B, L, channels)).astype("float32")
    point, quantiles = fr.decoder_for(family, model)(model(x))
    assert tuple(point.shape) == (B, H, channels)
    assert (quantiles is None) == (levels is None)
    if levels is not None:
        assert tuple(quantiles.shape) == (B, H, channels, levels)
    if family != "timestral":  # built at (None, L, C): another batch size runs
        other = fr.decoder_for(family, model)(model(x[:2]))[0]
        assert tuple(other.shape) == (2, H, channels)


def test_the_head_follows_quantile_levels_for_prism_and_xlstm():
    for family in ("prism", "xlstm"):
        point = build_adapter(family, "tiny")
        quantile = build_adapter(family, "tiny", quantile_levels=list(LEVELS))
        assert (point.use_quantile_head, quantile.use_quantile_head) == (False, True), family
        assert [round(q, 6) for q in quantile.quantile_levels] == list(LEVELS), family
    with pytest.raises(ValueError, match="use_quantile_head"):
        build_adapter("prism", "tiny", use_quantile_head=True)
    with pytest.raises(ValueError, match="quantile_levels"):
        build_adapter("prism", "tiny", num_quantiles=3)


@pytest.mark.parametrize("family, variant", [("nbeats", "tiny"), ("tirex", "tiny"), ("xlstm", "tiny")])
def test_the_model_normalization_is_forced_off_and_a_user_true_is_refused_as_double_normalization(family, variant):
    assert build_adapter(family, variant).get_config()["use_normalization"] is False
    assert build_adapter(family, variant, use_normalization=False).get_config()["use_normalization"] is False
    with pytest.raises(ValueError, match="normalize twice"):
        build_adapter(family, variant, use_normalization=True)
    with pytest.raises(ConfigValidationError, match="normalize twice"):
        fm.resolve_forecast_args(forecast_context(family), resolved_optim(family), {"use_normalization": True})


def test_the_mdn_model_is_a_flatten_stub_in_front_of_the_mixture_model_and_timestral_has_a_static_batch():
    mdn = build_adapter("mdn", "tiny")
    assert type(mdn.layers[1]).__name__ == "Flatten" and any(hasattr(layer, "mdn_layer") for layer in mdn.layers)
    timestral = build_adapter("timestral", "5M")
    assert timestral.built
    with pytest.raises(ValueError, match="static batch"):
        fm.create_timestral(variant="5M", input_shape=(L, 1), horizon=H, batch_size=0)


def test_ets_takes_the_librarys_tuple_and_a_seasonal_variant_needs_its_period():
    from dl_techniques.models.time_series.ets.model import ETS_VARIANTS

    assert registry.load_attr(fm.FAMILIES["ets"].variants) is ETS_VARIANTS and isinstance(ETS_VARIANTS, tuple)
    with pytest.raises(ValueError, match="Unknown variant 'BBB'"):
        build_adapter("ets", "BBB")
    with pytest.raises(ValueError):
        build_adapter("ets", "AAA")


def test_the_adapters_refuse_what_the_family_does_not_take():
    for call in (
        lambda: fm.create_nbeats(variant="tiny", num_classes=10, input_shape=(L, 1), horizon=H),
        lambda: fm.create_nbeats(variant="tiny", input_shape=(L, 1, 1), horizon=H),
        lambda: fm.create_nbeats(variant="tiny", input_shape=(L, 0), horizon=H),
        lambda: fm.create_nbeats(variant="tiny", input_shape=(L, 1), horizon=0),
        lambda: fm.create_nbeats(variant="huge", input_shape=(L, 1), horizon=H),
        lambda: fm.create_mdn(variant="huge", input_shape=(L, 1), horizon=H),
    ):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("family", ["tirex", "ets", "mdn"])
def test_a_second_channel_is_a_named_refusal_for_tirex_ets_and_mdn(family):
    assert fm.geometry_problems(family, 1, H, {}) == []
    (line,) = fm.geometry_problems(family, 2, H, {})
    assert family in line and "1 channel" in line and "data.value_cols" in line
    variant = {"tirex": "tiny", "ets": "ANN", "mdn": "tiny"}[family]
    with pytest.raises(ValueError, match="1 channel"):
        build_adapter(family, variant, channels=2)
    assert all(fm.geometry_problems(f, 5, H, {}) == [] for f in FAMILIES if f not in ("tirex", "ets", "mdn"))


def test_a_timestral_horizon_above_its_patch_is_refused_by_name_and_up_to_it_is_not():
    assert fm.geometry_problems("timestral", 1, fm.TIMESTRAL_PATCH, {}) == []
    (line,) = fm.geometry_problems("timestral", 1, fm.TIMESTRAL_PATCH + 1, {})
    assert "first block" in line and "data.horizon" in line and str(fm.TIMESTRAL_PATCH) in line
    assert fm.geometry_problems("timestral", 1, 24, {"patch_size": 16})[0].count("16") >= 2
    assert fm.geometry_problems("nbeats", 1, 500, {"patch_size": 16}) == []
    with pytest.raises(ValueError, match="first block"):
        fm.create_timestral(variant="5M", input_shape=(L, 1), horizon=fm.TIMESTRAL_PATCH + 1, batch_size=B)


_FRESH_REFUSAL = """
import sys
from train.api import forecast_models as fm
from train.api.errors import ConfigError
for model in sys.argv[1:]:
    try:
        fm.check_supported_family(model)
        print("@@", model, "accepted")
    except ConfigError as exc:
        print("@@", model, "refused", str(exc))
heavy = sorted(m for m in ("tensorflow", "keras", "dl_techniques", "train.common") if m in sys.modules)
print("@@ heavy", heavy)
"""


def test_deepar_and_adaptive_ema_are_refused_by_name_before_any_model_code_is_imported(tmp_path):
    src = str(SRC)
    env = {**__import__("os").environ, "PYTHONPATH": src, "CUDA_VISIBLE_DEVICES": "", "TF_CPP_MIN_LOG_LEVEL": "3"}
    out = subprocess.run([sys.executable, "-c", _FRESH_REFUSAL, "deepar/base", "adaptive_ema", "nbeats/base"],
                         capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=60)
    assert out.returncode == 0, out.stderr[-1500:]
    lines = [l[3:] for l in out.stdout.splitlines() if l.startswith("@@ ")]
    assert lines[0].startswith("deepar/base refused") and "teacher-forced" in lines[0] and "time_series_forecasting" in lines[0]
    assert lines[1].startswith("adaptive_ema refused") and "no horizon axis" in lines[1]
    assert lines[2] == "nbeats/base accepted"
    assert lines[3] == "heavy []", "the refusal imported a model module or TensorFlow"
    assert set(fm.UNSUPPORTED) == {"deepar", "adaptive_ema"}
    assert "deepar" not in {spec.family for spec in fm.builtin_forecast_models()}
    with pytest.raises(UnknownModelError):
        registry.parse_model_name("deepar/base")


def test_the_problem_lines_name_every_bad_key_at_once():
    lines = fm.problems_of("nbeats", {"backcast_length": 5, "kernel_regularizer": "l2", "use_normalization": True,
                                      "tree_depth": 3, "hidden_layer_units": 32})
    assert len(lines) == 4
    text = "\n".join(lines)
    assert "data.lookback" in text and "decays the weights" in text and "normalize twice" in text
    assert "tree_depth" in text and "hidden_layer_units" not in text
    assert "Did you mean" in fm.problems_of("nbeats", {"hidden_layer_unit": 32})[0]
    assert fm.problems_of("nbeats", {"use_normalization": False}) == []
    with pytest.raises(ConfigValidationError) as err:
        fm.resolve_forecast_args(forecast_context("prism"), resolved_optim("prism"), {"num_quantiles": 3, "nope": 1})
    assert len(err.value.args[0] if isinstance(err.value.args[0], list) else str(err.value).splitlines()) >= 2


def test_a_key_of_another_family_is_not_accepted_for_this_one():
    # every key a family's own tables offer is a key of another family's callables and not of this one's
    for family in FAMILIES:
        others = {k for f in FAMILIES if f != family for k in fm.ALLOWED[f]} - set(fm.ALLOWED[family]) \
            - set(fm.REFUSED[family]) - set(fm.FORCED[family]) - set(fm.SUPPLIED[family])
        assert others, family
        for key in sorted(others):
            assert fm.problems_of(family, {key: 1}), f"{family} accepts {key}, a key of another family"


def resolved_optim(family, **given):
    return OptimArgs(**given).resolve_auto(forecast_context(family))


def test_an_sgd_weight_decay_is_refused_as_the_other_resolvers_do():
    ctx = forecast_context("nbeats")
    with pytest.raises(ConfigError, match="optim_args.weight_decay"):
        fm.resolve_forecast_args(ctx, resolved_optim("nbeats", optimizer="sgd", lr=0.1, weight_decay=0.01), {})
    for given in ({"optimizer": "sgd", "lr": 0.1, "weight_decay": None}, {"optimizer": "adamw"}):
        assert fm.resolve_forecast_args(ctx, resolved_optim("nbeats", **given), {"hidden_layer_units": 16}) == {
            "hidden_layer_units": 16}


@pytest.mark.parametrize("family", FAMILIES)
def test_jit_true_is_refused_for_ets_alone(family):
    reason = fm.jit_problem(family, True)
    assert (reason is not None) == (family == "ets")
    assert fm.jit_problem(family, False) is None
    if reason:
        assert "gradient" in reason and "'jit_compile'" in reason


def forecast_context(family, variant="base"):
    spec = {s.family: s for s in fm.builtin_forecast_models()}[family]
    return AutoContext(family=family, variant=variant, dataset=fa.SYNTHETIC, num_classes=None, image_size=None,
                       defaults=spec.recipe(fa.SYNTHETIC, variant))


# ---------------------------------------------------------------------
# the recipes, against the legacy trainers' values READ from the legacy modules
# ---------------------------------------------------------------------

LEGACY = {
    "nbeats": ("train_nbeats", "NBeatsTrainingConfig"), "prism": ("train_prism", "PRISMTrainingConfig"),
    "tirex": ("train_tirex", "TiRexTrainingConfig"), "xlstm": ("train_xlstm", "XLSTMForecasterTrainingConfig"),
    "ets": ("train_ets", "ETSTrainingConfig"), "timestral": ("train_timestral", "TimestralTrainingConfig"),
    "mdn": ("train_mdn", "MDNTrainingConfig"),
}


def legacy_module(family):
    import importlib

    return importlib.import_module(f"train.time_series.{family}.{LEGACY[family][0]}")


def legacy_config(family):
    return getattr(legacy_module(family), LEGACY[family][1])()


def cli_default(module, flag):
    """The ``default=`` of ``parser.add_argument(flag, ...)`` in a legacy module's source."""
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args \
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == flag:
            return ast.literal_eval(next(k.value for k in node.keywords if k.arg == "default"))
    raise KeyError(flag)


#: Why a recipe value differs from the legacy trainer's, one sentence per (family, name); a difference without one fails.
WHY = {
    ("*", "epochs"): "10 epochs of 500 steps meet the bar with margin (D-032); the legacy 150 and 200 are for convergence studies",
    ("*", "warmup"): "one warmup epoch (500 steps), measured equal at 2,000 steps (D-034); the legacy 1000 steps is two epochs at 500",
    ("prism", "batch"): "one batch for every family, the shared legacy default 128 and what the GPU probe measured",
    ("mdn", "batch"): "one batch for every family, the shared legacy default 128 and what the GPU probe measured",
    ("timestral", "lookback"): "the legacy 1024 needs series of 7,500 points; 168 is every other family's",
    ("timestral", "horizon"): "a horizon above the patch size 32 is not the model's call (D-003), the legacy 96 is the multi-block forecaster's",
    ("mdn", "lookback"): "the legacy mdn is a one-step density over 120 points; 168 and 24 make its val_mae comparable with the bar",
    ("mdn", "horizon"): "the legacy mdn is a one-step density over 120 points; 168 and 24 make its val_mae comparable with the bar",
    ("nbeats", "optimizer"): "the legacy dataclass says adam and its CLI adamw; adamw with decoupled decay is the recipe (D-034)",
}


def recipe_rows():
    """``(family, name, recipe value, legacy value)`` for every number a recipe shares with a legacy trainer."""
    from train.common import forecast_data as fd
    from train.common.args import create_ts_argument_parser

    shared = vars(create_ts_argument_parser("x").parse_args([]))
    rows = []
    for family in FAMILIES:
        recipe, cfg, optim = fm.RECIPES[family], legacy_config(family), fm.RECIPES[family]["optim_args"]
        rows += [
            (family, "lr", optim["lr"], cfg.learning_rate),
            (family, "clip", optim["global_clipnorm"], cfg.gradient_clip_norm),
            (family, "batch", recipe["batch_size"], cfg.batch_size),
            (family, "epochs", recipe["epochs"], cfg.epochs),
            (family, "lookback", recipe["data"]["lookback"], getattr(cfg, "backcast_length", None) or cfg.input_length),
            (family, "horizon", recipe["data"]["horizon"], getattr(cfg, "forecast_length", None) or cfg.prediction_length),
            (family, "warmup", optim["warmup_epochs"] * fd.EPOCH_STEPS if cfg.use_warmup else 0,
             cfg.warmup_steps if cfg.use_warmup else 0),
            (family, "optimizer", optim["optimizer"], cfg.optimizer),
        ]
    rows.append(("nbeats", "hidden", fm.NBEATS_VARIANTS["base"]["hidden_layer_units"],
                 cli_default(legacy_module("nbeats"), "--hidden_layer_units")))
    rows.append(("nbeats", "optimizer, CLI", fm.RECIPES["nbeats"]["optim_args"]["optimizer"], shared["optimizer"]))
    return rows


def test_every_recipe_number_equals_the_legacy_trainers_or_names_why_it_does_not():
    differences = []
    for family, name, ours, theirs in recipe_rows():
        if ours == theirs:
            assert (family, name) not in WHY, f"{family} {name} is equal but has a reason"
        else:
            differences.append((family, name))
            assert WHY.get((family, name)) or WHY.get(("*", name)), f"{family} {name}: recipe {ours!r}, legacy {theirs!r}, no reason"
    assert ("nbeats", "lr") not in differences and ("ets", "lr") not in differences and ("mdn", "lr") not in differences
    # the stated differences are the ones that exist: a reason for an equal value is a stale reason
    present = {(f, n) for f, n in differences}
    assert {k for k in WHY if k[0] != "*"} <= present, sorted({k for k in WHY if k[0] != "*"} - present)
    assert {n for f, n in WHY if f == "*"} <= {n for _, n in present}


def test_the_legacy_disagreements_are_resolved_to_the_one_named_side():
    # nbeats hidden units: the dataclass says 128, the CLI 256; the recipe follows the CLI (what the trainer a user runs builds)
    assert legacy_config("nbeats").hidden_layer_units == 128
    assert cli_default(legacy_module("nbeats"), "--hidden_layer_units") == 256 == fm.NBEATS_VARIANTS["base"]["hidden_layer_units"]
    # tirex lookback: the dataclass says 168, the CLI 256; the recipe follows the dataclass, which is every other family's
    assert legacy_config("tirex").input_length == 168 and cli_default(legacy_module("tirex"), "--input_length") == 256
    assert fm.RECIPES["tirex"]["data"]["lookback"] == 168
    # AdamW's weight decay is what the legacy 'adamw' string built
    assert keras.optimizers.AdamW().weight_decay == fm.WEIGHT_DECAY


@pytest.mark.parametrize("family", FAMILIES)
def test_a_recipe_is_a_global_clip_adamw_with_warmup_in_epochs_and_no_stacked_l2(family):
    recipe = fm.RECIPES[family]
    optim = recipe["optim_args"]
    assert (recipe["epochs"], recipe["batch_size"], optim["optimizer"], optim["weight_decay"]) == (10, 128, "adamw", 0.004)
    assert optim["global_clipnorm"] == 1.0 and optim["clipnorm"] is None and optim["decay_over"] == "total"
    assert optim["warmup_epochs"] == (0 if family == "ets" else 1)
    assert optim["lr"] == {"ets": 5e-2, "mdn": 5e-4}.get(family, 1e-4)
    assert recipe["model_args"] == {} and not any("regulariz" in key for key in recipe["model_args"])
    assert recipe["metric_args"] == {"watch_metric": "val_mae"} and recipe["data"] == {"lookback": 168, "horizon": 24}
    assert set(optim) == {"optimizer", "lr", "weight_decay", "momentum", "beta_1", "beta_2", "epsilon", "schedule",
                          "warmup_epochs", "warmup_start_lr", "min_lr_ratio", "decay_over", "clipnorm", "global_clipnorm"}


@pytest.mark.parametrize("family", FAMILIES)
def test_the_recipe_fills_the_groups_of_the_task_and_leaves_no_auto(family):
    ctx = forecast_context(family)
    optim = OptimArgs().resolve_auto(ctx)
    assert (optim.lr, optim.global_clipnorm, optim.clipnorm) == (fm.RECIPES[family]["optim_args"]["lr"], 1.0, None)
    data = fa.TimeSeriesData(series_length=2000).resolve_auto(ctx)
    assert (data.lookback, data.horizon) == (168, 24)
    metric = fa.ForecastMetricArgs().resolve_auto(ctx)
    assert (metric.watch_metric, metric.watch_mode) == ("val_mae", "min")
    assert fm.RECIPES["nbeats"]["optim_args"]["lr"] == 1e-4 and fm.NBEATS_VARIANTS["base"]["hidden_layer_units"] == 256


# ---------------------------------------------------------------------
# the specs: built for one task each way, registered by one call
# ---------------------------------------------------------------------


def test_the_specs_are_built_for_the_forecasting_task_and_load_their_code():
    specs = fm.builtin_forecast_models()
    assert [s.family for s in specs] == list(FAMILIES)
    for spec in specs:
        assert spec.tasks == ("time_series_forecasting",) == (fa.TASK,) and spec.stride is None
        assert callable(spec.load_factory()) and callable(spec.load_resolver()) and len(spec.load_variants()) >= 1
        assert spec.load_factory().__module__ == "train.api.forecast_models"
    assert sorted(registry_variants("nbeats")) == ["base", "small", "tiny"] and sorted(registry_variants("mdn")) == ["base", "small", "tiny"]
    assert sorted(registry_variants("ets")) == ["AAA", "AAN", "ANN"] and "5M" in registry_variants("timestral")
    assert fm.FAMILIES["timestral"].static_batch and not any(fm.FAMILIES[f].static_batch for f in FAMILIES if f != "timestral")


def registry_variants(family):
    return {s.family: s for s in fm.builtin_forecast_models()}[family].load_variants()


def test_the_specs_are_registered_with_the_task_by_one_call_of_the_registry():
    # `builtin_forecast_models` only builds the specs; the one call in `register_builtin_models` registers them with the task
    # (a registered family with no task class breaks test_registry).
    recipes.register_builtin_models()
    assert set(FAMILIES) <= set(registry.MODELS)
    source = inspect.getsource(recipes.register_builtin_models)
    assert source.count("builtin_forecast_models") == 2, "one import and one use"


def test_a_forecaster_is_refused_by_the_other_tasks_and_a_classifier_by_this_one(monkeypatch):
    recipes.register_builtin_models()
    for spec in fm.builtin_forecast_models():
        monkeypatch.setitem(registry.MODELS, spec.family, spec)
    assert registry.parse_model_name("nbeats/base", task=fa.TASK) == ("nbeats", "base")
    assert registry.parse_model_name("ets/ANN", task=fa.TASK) == ("ets", "ANN")
    for task in ("image_classification", "image_segmentation", "pretrain"):
        with pytest.raises(ConfigError, match="trains time_series_forecasting"):
            registry.parse_model_name("nbeats/base", task=task)
    with pytest.raises(ConfigError) as err:
        registry.parse_model_name("vit/vit_pico", task=fa.TASK)
    assert "the families of 'time_series_forecasting' are" in str(err.value) and "nbeats" in str(err.value)
    with pytest.raises(ConfigError, match="trains"):
        registry.parse_model_name("convunext/tiny", task=fa.TASK)
    with pytest.raises(UnknownModelError):
        registry.parse_model_name("nbeats/huge", task=fa.TASK)
    assert registry.parse_model_name("vit/vit_pico", task="image_classification") == ("vit", "vit_pico")


def test_the_module_imports_no_model_code_tensorflow_or_keras_at_module_scope():
    tree = ast.parse((SRC / "train" / "api" / "forecast_models.py").read_text())
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(("." * node.level) + (node.module or ""))
    assert not [n for n in names if n.split(".")[0] in ("tensorflow", "keras", "numpy", "pandas", "dl_techniques")
                or n.startswith("train.common")], names
    assert all(n.startswith(".") or n in ("dataclasses", "typing") for n in names), names
