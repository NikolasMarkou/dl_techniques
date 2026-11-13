"""The forecasting run pieces (plan-2026-11-13T105600-3a8319ef, Step 9): baselines, decode table, metrics, report.

``train.common.forecast_run`` is the one place that knows what a forecasting model's OUTPUT means. This file pins,
on the CPU at tiny sizes, the part of it Step 9 ships (later steps add the argument groups, the family adapters
and the task to this same file):

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
"""

import ast
import json
from pathlib import Path

import keras
import numpy as np
import pytest

from train.api.errors import ConfigError
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
# tiny models, one per row (the adapters of Step 11 will build these; here they are written out)
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
