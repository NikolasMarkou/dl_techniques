"""Predict with a ``time_series_forecasting`` run: the record, ``ForecastPredictor`` and the ``predict`` command, and the export of a forecasting run (D-010, D-011, D-012, D-026).

First part (the export cells are the second part). Every case of ``forecast_run.DECODE`` is TRAINED for one
epoch at tiny sizes (lookback 32, horizon 8, batch 16, CPU) and the run directory it writes is read back:

* **The record.** The run writes a ``task.json`` whose keys are ``predictor.RECORD_KEYS`` for the task, in order, and whose
  values are what the run was told and what the model is: the window, the channels, the scale floor, the head, the quantile
  levels and the median index and the decode row of the model, and where the data came from. The record written BEFORE the
  model is built (a resume asks then) is the record written after.
* **The Predictor.** ``load_model`` returns a ``ForecastPredictor`` for every trained family. Its point forecast equals the
  family's decode row applied to the window standardized by the numpy twin of the data function and de-standardized with
  the window's own context statistics, to float32; a window shifted by a constant gives the forecast shifted by it and a
  window scaled by a constant the forecast scaled (the oracle that sees a missing or wrong de-standardization); a quantile
  head returns ``quantiles`` and ``quantile_levels`` and the point is the median slice, a point head returns ``None`` for
  both. An array ``(L, C)``, an array ``(N, L, C)``, a list and a csv path (the last ``lookback`` rows of the record's
  ``value_cols``) are the four inputs.
* **Refusals, all a named ``ConfigError``.** A window of the wrong length or channels, a NaN or infinite cell, a value that
  is not a number, a csv missing a value column or with too few rows, a record whose horizon, head, levels or decode row the
  model does not have, a ``.keras`` moved away from its ``task.json``, a record whose family does not train the task.
* **The command.** ``python -m train predict model=<run> data=<csv or npy> out=<new file>`` writes one JSON and a second call
  to the same ``out`` is refused; a directory, an unknown suffix and an ``.npz`` of several arrays are refused by name; the
  record is read BEFORE the data is listed (D-012), so an image run's order of errors changes only for a missing record
  together with an empty image directory.
* **Export** (second part, D-011 and D-026): see the section at the end of the file.

Everything writes under ``tmp_path``; nothing is deleted.
"""

import hashlib
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import keras
import numpy as np
import pandas as pd
import pytest

from train.api import cli, predictor
from train.api.errors import ConfigError
from train.api.predictor import RECORD_KEYS, TASK_TABLES, check_record, load_model, record_differences
from train.api.tasks.time_series_forecasting import train_time_series_forecasting
from train.common import forecast_data as fd
from train.common import forecast_run as fr

from ._forecast_fixtures import standardize_np, world, write_series_csv  # noqa: F401  (``world`` is a fixture)
from .test_forecast_task import CASES, DATA, QUANTILES, SRC, prepared, read_json

TASK = "time_series_forecasting"
BATCH, STEPS = 16, 4
SIZES = {**DATA, "n_select": 32, "n_test": 32, "steps_per_epoch": STEPS}
LOOKBACK, HORIZON = SIZES["lookback"], SIZES["horizon"]
CSV_COLUMNS, CSV_USED = 3, ["v0", "v2"]
CHUNK = 32

#: What a run's record says about the head of each case: ``(head, levels or None, median index or None)``. The levels of
#: tirex and timestral are the library's own, so they are read from the model and pinned only as "a quantile head".
HEADS = {
    "nbeats": ("point", None, None),
    "prism_point": ("point", None, None),
    "prism_quantile": ("quantile", QUANTILES, 1),
    "tirex": ("quantile", None, None),
    "xlstm_point": ("point", None, None),
    "xlstm_quantile": ("quantile", QUANTILES, 1),
    "ets": ("point", None, None),
    "timestral": ("quantile", None, None),
    "mdn": ("point", None, None),
}

_RUNS: Dict[str, SimpleNamespace] = {}


def train(root: Path, model: str, model_args: Dict[str, Any], data: Dict[str, Any]) -> Path:
    """One epoch of ``model`` on ``data`` into ``root / "run"``."""
    out = root / "run"
    train_time_series_forecasting(
        out=str(out), model=model, data=data, epochs=1, batch_size=BATCH, model_args=model_args,
        analysis_args={"model_analysis": False, "figures": False},
    )
    return out


def trained_case(case: str, factory) -> SimpleNamespace:
    """The run of ``case`` (trained once per module) and the predictor ``load_model`` makes of it."""
    if case not in _RUNS:
        model, model_args = CASES[case]
        run = train(factory.mktemp("predict_" + case), model, model_args, dict(SIZES))
        _RUNS[case] = SimpleNamespace(case=case, run=run, predictor=load_model(run), record=read_json(run / "task.json"))
    return _RUNS[case]


@pytest.fixture(scope="module", params=list(CASES))
def trained(request, world, tmp_path_factory):
    return trained_case(request.param, tmp_path_factory)


@pytest.fixture(scope="module")
def one(world, tmp_path_factory):
    """The nbeats run, for the tests that need one model and not nine."""
    return trained_case("nbeats", tmp_path_factory)


@pytest.fixture(scope="module")
def csv_run(world, tmp_path_factory):
    """nbeats on a csv of three value columns of which two are named: two channels, ``value_cols`` in the record."""
    root = tmp_path_factory.mktemp("predict_csv")
    csv = write_series_csv(root / "series.csv", rows=1200, columns=CSV_COLUMNS)
    data = {"format": "csv", "path": str(csv), "value_cols": CSV_USED, "time_col": "t", "lookback": LOOKBACK,
            "horizon": HORIZON, "steps_per_epoch": STEPS, "n_select": BATCH, "n_test": BATCH}
    run = train(root, "nbeats/tiny", {}, data)
    return SimpleNamespace(run=run, csv=csv, predictor=load_model(run), record=read_json(run / "task.json"))


def raw_windows(count: int, channels: int, seed: int = 0, lookback: int = LOOKBACK) -> np.ndarray:
    """Windows in a unit of their own: a level near 50, a season, a random walk (float64, ``(count, lookback, channels)``)."""
    rng = np.random.default_rng(seed)
    t = np.arange(lookback)[None, :, None]
    season = 8.0 * np.sin(2 * np.pi * t / 12.0 + rng.uniform(0, 6, (count, 1, channels)))
    return 50.0 + season + 0.3 * rng.normal(size=(count, lookback, channels)).cumsum(axis=1)


def reference(predictor_: Any, windows: np.ndarray):
    """The point and the quantiles the model gives for ``windows``, worked out here from the data module's numpy twin."""
    record = predictor_.record
    anchor, offset, scale = fd.context_stats_np(windows, record["scale_floor"])
    mean = anchor + offset
    x = standardize_np(
        np.concatenate([windows, np.zeros((len(windows), record["horizon"], windows.shape[-1]))], axis=1),
        record["lookback"], record["scale_floor"])[0]
    point, cube = fr.decoder_for(record["family"], predictor_.model)(predictor_.model(x, training=False))
    point = keras.ops.convert_to_numpy(point).astype(np.float64)
    if cube is None:
        return point * scale + mean, None, scale
    cube = keras.ops.convert_to_numpy(cube).astype(np.float64)
    return point * scale + mean, cube * scale[..., None] + mean[..., None], scale


# ---------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------


def test_the_tables_hold_the_fourth_task_and_it_has_a_writer_and_a_lister():
    assert predictor.FORECASTING == TASK
    for name, table in TASK_TABLES.items():
        assert TASK in table, name
    assert TASK in cli.PREDICT_WRITERS and TASK in cli.DATA_LISTERS
    assert sorted(cli.DATA_LISTERS) == sorted(cli.PREDICT_WRITERS), "a task that predicts lists its data and writes its answer"


def test_the_run_writes_a_record_with_the_keys_of_the_table_in_order(trained):
    record = trained.record
    assert list(record) == list(RECORD_KEYS[TASK])
    check_record(record, "task.json")
    assert record["schema_version"] == predictor.SCHEMA_VERSION and record["task"] == TASK
    model, _ = CASES[trained.case]
    assert record["model"] == model and record["family"] == model.partition("/")[0] and record["variant"] == model.partition("/")[2]


def test_the_record_holds_what_the_run_was_told_and_what_the_model_is(trained):
    record, config = trained.record, read_json(trained.run / "config.json")
    assert (record["lookback"], record["horizon"], record["channels"]) == (LOOKBACK, HORIZON, 1)
    assert record["value_cols"] is None and record["data_format"] == "builtin" and record["data_sha256"] is None
    assert record["scale_floor"] == config["transform_args"]["scale_floor"]
    head, levels, median = HEADS[trained.case]
    assert record["head"] == head and record["decode"] == trained.case
    if head == "point":
        assert record["quantile_levels"] is None and record["point_index"] is None
    else:
        assert isinstance(record["quantile_levels"], list) and len(record["quantile_levels"]) > 1
        assert record["point_index"] == trained.predictor.decoder.median_index
        assert record["quantile_levels"] == [float(q) for q in trained.predictor.decoder.levels]
        if levels is not None:
            assert record["quantile_levels"] == levels and record["point_index"] == median
    assert "variant_overrides" not in record, "a run that touches no key of its variant's table records no override"


def test_the_csv_record_names_the_columns_the_run_read_and_the_bytes_of_the_file(csv_run):
    record = csv_run.record
    assert record["channels"] == 2 and record["value_cols"] == CSV_USED and record["data_format"] == "csv"
    assert record["time_col"] == "t", "the record names the time column the run sorted by, so predict can check the order (D-042)"
    assert record["data_sha256"] == hashlib.sha256(csv_run.csv.read_bytes()).hexdigest()


def test_a_record_written_before_the_model_is_built_equals_the_one_written_after(world, tmp_path):
    """A resume asks for the record BEFORE ``build_model`` (``check_data_unchanged``); the answer must not depend on it."""
    trainer, resolved, context = prepared(tmp_path / "run")
    data = trainer.build_data(resolved, context)
    before = trainer.task_record(resolved, context, data)
    trainer.build_model(resolved, context, data)
    after = trainer.task_record(resolved, context, data)
    assert before == after and list(before) == list(RECORD_KEYS[TASK])
    check_record(before, "<built>")


def test_a_resume_compares_the_window_the_channels_the_columns_and_the_data(one):
    earlier = one.record
    assert record_differences(earlier, dict(earlier)) == []
    for key, value, shown in (("lookback", 64, "lookback: 32 -> 64"), ("horizon", 4, "horizon: 8 -> 4"),
                              ("channels", 2, "channels: 1 -> 2"), ("value_cols", ["a"], 'value_cols: null -> ["a"]'),
                              ("data_format", "csv", 'data_format: "builtin" -> "csv"'),
                              ("scale_floor", 0.5, "scale_floor: 1e-06 -> 0.5")):
        assert record_differences(earlier, {**earlier, key: value}) == [shown], key
    assert record_differences(earlier, {**earlier, "head": "quantile"}) == [], "the head is the model's, not the data's"


def bad(**changes):
    return changes  # a record's changed keys, for the table below


BAD_RECORDS = [
    (bad(lookback=0), "'lookback' must be a positive int"),
    (bad(lookback=True), "'lookback' must be a positive int"),
    (bad(horizon=2.5), "'horizon' must be a positive int"),
    (bad(channels=0), "'channels' must be a positive int"),
    (bad(scale_floor=-1.0), "'scale_floor' must be a finite number of at least 0"),
    (bad(scale_floor=float("nan")), "'scale_floor' must be a finite number of at least 0"),
    (bad(scale_floor="0.1"), "'scale_floor' must be a finite number of at least 0"),
    (bad(head="interval"), "'head' must be one of point, quantile"),
    (bad(head="quantile"), "'quantile_levels' must be a list for a quantile head"),
    (bad(quantile_levels=[0.1, 0.9]), "'quantile_levels' must be null for a point head"),
    (bad(point_index=0), "'point_index' must be null for a point head"),
    (bad(decode=""), "'decode' must be a non-empty text"),
    (bad(data_format="folder"), "'data_format' must be one of builtin, csv"),
    (bad(data_sha256="abc"), "'data_sha256' must be null for the builtin format"),
    (bad(value_cols=["a"]), "'value_cols' must be null for the builtin format"),
    (bad(data_format="csv", value_cols=None, data_sha256="0" * 64), "'value_cols' must be a list of 1 distinct texts for the csv format"),
    (bad(data_format="csv", value_cols=["a"], data_sha256="xyz"), "'data_sha256' must be the 64 hex digits"),
    (bad(data_format="csv", value_cols=["a", "a"], channels=2, data_sha256="0" * 64), "'value_cols' must be a list of 2 distinct texts"),
    (bad(time_col="t"), "'time_col' must be null for the builtin format"),
    (bad(data_format="csv", value_cols=["a"], data_sha256="0" * 64, time_col=""), "'time_col' must be a non-empty text or null"),
    (bad(data_format="csv", value_cols=["a"], data_sha256="0" * 64, time_col=3), "'time_col' must be a non-empty text or null"),
]


@pytest.mark.parametrize("changes, message", BAD_RECORDS, ids=[m for _, m in BAD_RECORDS])
def test_a_bad_record_is_refused_naming_the_key(one, changes, message):
    with pytest.raises(ConfigError, match=message):
        check_record({**one.record, **changes}, "task.json")


@pytest.mark.parametrize("key", [k for k in RECORD_KEYS.get(TASK, ("lookback",)) if k not in ("schema_version", "task")])
def test_a_record_missing_a_key_or_holding_another_tasks_key_is_refused(one, key):
    record = dict(one.record)
    del record[key]
    with pytest.raises(ConfigError, match=f"missing key.*{key}"):
        check_record(record, "task.json")
    with pytest.raises(ConfigError, match="unknown key.*class_names"):
        check_record({**one.record, "class_names": ["a"]}, "task.json")


def test_a_csv_record_with_its_columns_and_a_hash_is_valid(one):
    check_record({**one.record, "data_format": "csv", "value_cols": ["v0"], "data_sha256": "ab" * 32}, "task.json")
    check_record({**one.record, "data_format": "csv", "value_cols": ["v0"], "data_sha256": "ab" * 32, "time_col": "t"}, "task.json")


# ---------------------------------------------------------------------
# The Predictor
# ---------------------------------------------------------------------


def test_load_model_returns_a_forecast_predictor_for_every_trained_family(trained):
    from train.api.forecast_predictor import ForecastPredictor

    assert type(trained.predictor) is ForecastPredictor
    assert trained.predictor.record == trained.record


def test_the_point_is_the_decode_row_through_the_numpy_twin_to_float32(trained):
    windows = raw_windows(5, 1, seed=3)
    answer = trained.predictor.predict(windows)
    point, cube, scale = reference(trained.predictor, windows)
    assert answer["point"].shape == (5, HORIZON, 1)
    np.testing.assert_allclose(answer["point"], point, rtol=1e-6, atol=1e-6 * float(scale.max()))
    assert np.isfinite(answer["point"]).all()
    if cube is not None:
        np.testing.assert_allclose(answer["quantiles"], cube, rtol=1e-6, atol=1e-6 * float(scale.max()))


def test_the_forecast_is_in_the_units_of_the_window(trained):
    """The window sits near 50 with a spread near 6: a forecast left standardized sits near 0 and cannot pass."""
    windows = raw_windows(4, 1, seed=5)
    point = trained.predictor.predict(windows)["point"]
    spread = windows.std(axis=1).max()
    assert np.abs(point.mean(axis=1) - windows.mean(axis=1)).max() < 40 * spread, "a de-standardized forecast stays near the level"
    assert np.abs(point).min() > 0 and point.mean() > 5.0


@pytest.mark.parametrize("shift", [30.0, -125.0, 1e3, 1e5, 1e6])
def test_a_window_shifted_by_a_constant_gives_the_forecast_shifted_by_it(trained, shift):
    """D-040: up to a level of 1e6 over a std of about 8 (float64 windows; the scale no longer depends on the level)."""
    windows = raw_windows(4, 1, seed=7)
    plain = trained.predictor.predict(windows)
    shifted = trained.predictor.predict(windows + shift)
    _, _, scale = reference(trained.predictor, windows)
    tolerance = 1e-3 * scale
    np.testing.assert_allclose(shifted["point"] - shift, plain["point"], rtol=0, atol=float(tolerance.max()))
    if plain["quantiles"] is not None:
        np.testing.assert_allclose(shifted["quantiles"] - shift, plain["quantiles"], rtol=0, atol=float(tolerance.max()))


@pytest.mark.parametrize("factor", [7.0, 0.01])
def test_a_window_scaled_by_a_constant_gives_the_forecast_scaled_by_it(trained, factor):
    windows = raw_windows(4, 1, seed=9)
    plain = trained.predictor.predict(windows)
    scaled = trained.predictor.predict(windows * factor)
    _, _, scale = reference(trained.predictor, windows)
    tolerance = 1e-3 * scale * factor
    np.testing.assert_allclose(scaled["point"], plain["point"] * factor, rtol=0, atol=float(tolerance.max()))
    if plain["quantiles"] is not None:
        np.testing.assert_allclose(scaled["quantiles"], plain["quantiles"] * factor, rtol=0, atol=float(tolerance.max()))


def test_a_quantile_head_returns_quantiles_and_levels_and_a_point_head_returns_none_for_both(trained):
    answer = trained.predictor.predict(raw_windows(3, 1))
    head, _, _ = HEADS[trained.case]
    if head == "point":
        assert answer["quantiles"] is None and answer["quantile_levels"] is None
        return
    levels = trained.record["quantile_levels"]
    assert answer["quantile_levels"] == levels and answer["quantiles"].shape == (3, HORIZON, 1, len(levels))
    np.testing.assert_array_equal(answer["quantiles"][..., trained.record["point_index"]], answer["point"])


def test_the_four_inputs_give_one_answer(csv_run):
    predictor_, csv = csv_run.predictor, csv_run.csv
    import pandas as pd

    rows = pd.read_csv(csv)[CSV_USED].to_numpy(np.float64)
    last = rows[-LOOKBACK:]
    one_window = predictor_.predict(last)
    assert one_window["point"].shape == (1, HORIZON, 2)
    batch = np.stack([last, rows[-LOOKBACK - 5:-5]])
    stacked = predictor_.predict(batch)
    listed = predictor_.predict([last, rows[-LOOKBACK - 5:-5]])
    np.testing.assert_array_equal(stacked["point"], listed["point"])
    np.testing.assert_allclose(stacked["point"][:1], one_window["point"], rtol=1e-5, atol=1e-4)
    from_file = predictor_.predict(str(csv))
    np.testing.assert_array_equal(from_file["point"], one_window["point"])
    np.testing.assert_array_equal(predictor_.predict(Path(csv))["point"], one_window["point"])
    np.testing.assert_allclose(predictor_.predict([str(csv), last])["point"][1], one_window["point"][0], rtol=1e-5, atol=1e-4)


def test_a_batch_longer_than_a_chunk_is_the_chunks_in_order(one, monkeypatch):
    from train.api import forecast_predictor

    windows = raw_windows(7, 1, seed=11)
    whole = one.predictor.predict(windows)["point"]
    monkeypatch.setattr(forecast_predictor, "CHUNK", 3)
    np.testing.assert_allclose(one.predictor.predict(windows)["point"], whole, rtol=1e-5, atol=1e-5)


def test_predict_does_not_change_the_array_it_is_given(one):
    windows = raw_windows(3, 1, seed=13)
    before = windows.copy()
    one.predictor.predict(windows)
    np.testing.assert_array_equal(windows, before)


# ---------------------------------------------------------------------
# What the Predictor refuses
# ---------------------------------------------------------------------


def window_with(value, where=(2, 0)):
    windows = raw_windows(3, 1, seed=15)
    windows[(1, *where)] = value
    return windows


@pytest.mark.parametrize("make, message", [
    (lambda: raw_windows(2, 1, lookback=LOOKBACK + 1), r"Invalid window: each window must be \(32, 1\) \(lookback, channels\) but window 0 is \(33, 1\)"),
    (lambda: raw_windows(2, 1, lookback=LOOKBACK - 1), r"window 0 is \(31, 1\)"),
    (lambda: raw_windows(2, 2), r"window 0 is \(32, 2\)"),
    (lambda: raw_windows(1, 1)[0, :, 0], r"expected an array \(L, C\) or \(N, L, C\)"),
    (lambda: raw_windows(1, 1)[None], r"expected an array \(L, C\) or \(N, L, C\)"),
    (lambda: np.zeros((0, LOOKBACK, 1)), r"the array of windows is empty"),
    (lambda: [], r"the list of windows is empty"),
    (lambda: window_with(np.nan), r"window 1 holds a non-finite value \(NaN or infinity\) at row 2, column 0"),
    (lambda: window_with(np.inf, (7, 0)), r"window 1 holds a non-finite value \(NaN or infinity\) at row 7, column 0"),
    (lambda: np.array([["a"] * LOOKBACK], dtype=object).reshape(LOOKBACK, 1), r"the values must be numbers"),
    (lambda: [raw_windows(1, 1)[0], raw_windows(1, 1, lookback=LOOKBACK + 2)[0]], r"window 1 is \(34, 1\)"),
    (lambda: 3.5, r"expected an array, a list of windows or a csv path, got float"),
])
def test_a_window_the_model_cannot_take_is_a_named_error(one, make, message):
    with pytest.raises(ConfigError, match=message):
        one.predictor.predict(make())


def test_a_list_holding_a_window_with_a_nan_names_that_window(one):
    good = raw_windows(1, 1)[0]
    bad_window = good.copy()
    bad_window[3, 0] = np.nan
    with pytest.raises(ConfigError, match=r"window 2 holds a non-finite value"):
        one.predictor.predict([good, good, bad_window])


def test_a_csv_missing_a_column_short_absent_or_non_finite_is_a_named_error(csv_run, tmp_path):
    import pandas as pd

    frame = pd.read_csv(csv_run.csv)
    missing = tmp_path / "missing.csv"
    frame.drop(columns=["v2"]).to_csv(missing, index=False)
    with pytest.raises(ConfigError, match=r"Invalid value for 'data': .*column 'v2' is missing from missing.csv"):
        csv_run.predictor.predict(str(missing))
    short = tmp_path / "short.csv"
    frame.iloc[: LOOKBACK - 1].to_csv(short, index=False)
    with pytest.raises(ConfigError, match=rf"Invalid value for 'data': the csv has {LOOKBACK - 1} rows, fewer than the lookback {LOOKBACK}"):
        csv_run.predictor.predict(str(short))
    exact = tmp_path / "exact.csv"
    frame.iloc[:LOOKBACK].to_csv(exact, index=False)
    assert csv_run.predictor.predict(str(exact))["point"].shape == (1, HORIZON, 2), "exactly one lookback of rows is enough"
    with pytest.raises(ConfigError, match=r"does not exist"):
        csv_run.predictor.predict(str(tmp_path / "gone.csv"))
    inf = tmp_path / "inf.csv"
    bad_frame = frame.copy()
    bad_frame.loc[LOOKBACK + 1, "v0"] = np.inf
    bad_frame.to_csv(inf, index=False)
    with pytest.raises(ConfigError, match=r"Invalid value for 'data': .*non-finite"):
        csv_run.predictor.predict(str(inf))


def test_a_synthetic_run_reads_a_csv_whose_columns_are_all_numbers_and_counts_them(one, tmp_path):
    import pandas as pd

    numbers = tmp_path / "numbers.csv"
    pd.DataFrame(raw_windows(1, 1, lookback=60)[0], columns=["only"]).to_csv(numbers, index=False)
    assert one.predictor.predict(str(numbers))["point"].shape == (1, HORIZON, 1)
    two = tmp_path / "two.csv"
    pd.DataFrame(raw_windows(1, 2, lookback=60)[0], columns=["a", "b"]).to_csv(two, index=False)
    with pytest.raises(ConfigError, match=r"has 2 value columns \['a', 'b'\] but the model takes 1 channel"):
        one.predictor.predict(str(two))


def newest_first(source: Path, target: Path) -> Path:
    """The csv ``source`` with its rows in reverse order (a newest-first export of the same series)."""
    pd.read_csv(source).iloc[::-1].to_csv(target, index=False)
    return target


def test_predict_of_a_csv_checks_the_time_order_with_the_function_the_loader_uses(csv_run, tmp_path, monkeypatch, capsys):
    """Review W6: a newest-first csv was forecast from its OLDEST rows with no warning. The record names ``time_col`` and the
    check is ``forecast_series._check_time_column``, the loader's own (one implementation, D-042)."""
    from train.api import forecast_series

    reversed_file = newest_first(csv_run.csv, tmp_path / "newest_first.csv")
    with pytest.raises(ConfigError, match=r"data.time_col 't' must be strictly increasing with no empty cell: sort the file by time"):
        csv_run.predictor.predict(str(reversed_file))
    out = tmp_path / "forecast.json"
    code, _, stderr = run_command(["predict", f"model={csv_run.run}", f"data={reversed_file}", f"out={out}"], capsys)
    assert code == 2 and "strictly increasing" in stderr and not out.exists(), stderr[-600:]
    seen = []
    real = forecast_series._check_time_column
    monkeypatch.setattr(forecast_series, "_check_time_column", lambda column, name: (seen.append(name), real(column, name))[1])
    csv_run.predictor.predict(str(csv_run.csv))
    assert seen == ["t"], "predict reads the order through the loader's function, with the record's time column"


def test_a_csv_given_to_a_run_that_had_no_time_column_skips_the_order_check(csv_run, tmp_path):
    run = copy_run(csv_run.run, tmp_path / "run", time_col=None)
    answer = load_model(run).predict(str(newest_first(csv_run.csv, tmp_path / "newest_first.csv")))
    assert answer["point"].shape == (1, HORIZON, 2), "no time column was named at training, so the rows are the time order"


def copy_run(run: Path, where: Path, **changes) -> Path:
    """A run directory holding the best model and a task.json with ``changes``."""
    where.mkdir()
    shutil.copy(run / "best_model.keras", where / "best_model.keras")
    record = {**read_json(run / "task.json"), **changes}
    (where / "task.json").write_text(json.dumps(record))
    return where


def test_a_record_whose_horizon_the_model_does_not_forecast_is_refused(trained, tmp_path):
    run = copy_run(trained.run, tmp_path / "run", horizon=HORIZON + 1)
    with pytest.raises(ConfigError, match=rf"The model forecasts .* for one window but task.json names horizon {HORIZON + 1} and 1 channel"):
        load_model(run)


def test_a_record_that_names_another_head_levels_or_decode_row_is_refused(trained, tmp_path):
    head, _, _ = HEADS[trained.case]
    flipped = {"head": "quantile", "quantile_levels": [0.1, 0.5, 0.9], "point_index": 1} if head == "point" else {
        "head": "point", "quantile_levels": None, "point_index": None}
    with pytest.raises(ConfigError, match=r"the file and the model are not from one run"):
        load_model(copy_run(trained.run, tmp_path / "head", **flipped))
    with pytest.raises(ConfigError, match=rf"names the decode row 'other' but the model is {trained.case!r}"):
        load_model(copy_run(trained.run, tmp_path / "row", decode="other"))
    if head == "quantile":
        levels = list(trained.record["quantile_levels"])
        levels[0] = round(levels[0] / 2, 6)
        with pytest.raises(ConfigError, match=r"names the quantile levels"):
            load_model(copy_run(trained.run, tmp_path / "levels", quantile_levels=levels))
        with pytest.raises(ConfigError, match=r"names the point at level index"):
            load_model(copy_run(trained.run, tmp_path / "median", point_index=(trained.record["point_index"] + 1) % len(levels)))


@pytest.mark.parametrize("lookback", [LOOKBACK // 2, LOOKBACK + 8])
def test_a_record_whose_lookback_the_model_was_not_built_for_is_refused_naming_the_field_for_every_family(trained, tmp_path, lookback):
    """Review W5: four families (ets, xlstm, tirex, timestral) ran a window of the wrong length for a hand-edited lookback, because
    their forward pass takes any length. The record is held to the model's BUILT input shape (D-042), for all nine decode rows."""
    with pytest.raises(ConfigError, match=rf"lookback is {lookback} in task.json and {LOOKBACK} in the model"):
        load_model(copy_run(trained.run, tmp_path / "run", lookback=lookback))
    assert not [p for p in tmp_path.iterdir() if p.name != "run"]


def test_a_record_whose_channels_the_model_was_not_built_for_is_refused_naming_the_field_for_every_family(trained, tmp_path):
    with pytest.raises(ConfigError, match=r"channels is 2 in task.json and 1 in the model"):
        load_model(copy_run(trained.run, tmp_path / "run", channels=2))


def test_the_built_input_shape_is_read_from_the_model_the_way_each_family_keeps_it(trained):
    """A functional model (mdn) says ``input_shape``; every subclassed one keeps its build config (the batch of timestral is static)."""
    from train.api import forecast_predictor

    assert forecast_predictor.built_window(trained.predictor.model) == (LOOKBACK, 1)


def test_a_model_that_says_nothing_of_its_input_still_meets_the_zero_window_forward_pass(one, tmp_path, monkeypatch):
    """No input shape to compare (``built_window`` is ``None``): the zero window of the record is the check, as before the shape check."""
    from train.api import forecast_predictor

    class Silent:
        """A stand-in without ``input_shape`` and without a build config."""

    assert forecast_predictor.built_window(Silent()) is None
    monkeypatch.setattr(forecast_predictor, "built_window", lambda model: None)
    with pytest.raises(ConfigError, match=r"does not take the lookback"):
        load_model(copy_run(one.run, tmp_path / "run", lookback=LOOKBACK + 8))


def test_a_keras_file_moved_away_from_its_record_is_refused(one, tmp_path):
    moved = tmp_path / "moved.keras"
    shutil.copy(one.run / "best_model.keras", moved)
    with pytest.raises(ConfigError, match=r"has no record beside it"):
        load_model(moved)
    shutil.copy(one.run / "task.json", tmp_path / "moved.keras.task.json")
    assert type(load_model(moved)).__name__ == "ForecastPredictor", "the record the export writes beside a file is found"


def test_a_record_whose_family_does_not_train_the_task_is_refused(one, tmp_path):
    run = copy_run(one.run, tmp_path / "run", model="vit/vit_pico", family="vit", variant="vit_pico")
    with pytest.raises(ConfigError, match=r"the family 'vit' trains image_classification, not 'time_series_forecasting'"):
        load_model(run)


def test_the_predictor_refuses_a_record_of_another_task(one):
    from train.api.forecast_predictor import ForecastPredictor

    other = {"schema_version": 2, "task": "image_denoising", "model": "bfcnn/tiny", "family": "bfcnn", "variant": "tiny",
             "channels": 1, "size_multiple": 1, "data_range": predictor.unit_domain_stamp(), "output": "images"}
    with pytest.raises(ConfigError, match=r"this predictor serves 'time_series_forecasting' only"):
        ForecastPredictor(one.predictor.model, other)


# ---------------------------------------------------------------------
# The predict command
# ---------------------------------------------------------------------


def run_command(argv: List[str], capsys):
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def write_npy(path: Path, windows: np.ndarray) -> Path:
    np.save(path, windows)
    return path


def test_predict_writes_one_json_for_an_npy_of_windows_and_refuses_a_second_call_to_the_same_out(one, tmp_path, capsys):
    windows = raw_windows(3, 1, seed=17)
    data, out = write_npy(tmp_path / "w.npy", windows), tmp_path / "answer.json"
    code, stdout, stderr = run_command(["predict", f"model={one.run}", f"data={data}", f"out={out}"], capsys)
    assert code == 0, stderr[-1500:]
    assert stdout.strip() == str(out)
    answer = json.loads(out.read_text())
    assert list(answer) == ["task", "model", "data", "lookback", "horizon", "channels", "quantile_levels", "point", "quantiles"]
    assert answer["task"] == TASK and answer["model"] == "nbeats/tiny" and answer["data"] == str(data)
    assert (answer["lookback"], answer["horizon"], answer["channels"]) == (LOOKBACK, HORIZON, 1)
    assert answer["quantile_levels"] is None and answer["quantiles"] is None
    np.testing.assert_allclose(np.array(answer["point"]), one.predictor.predict(windows)["point"], rtol=1e-9)
    before = hashlib.sha256(out.read_bytes()).hexdigest()
    code, _, stderr = run_command(["predict", f"model={one.run}", f"data={data}", f"out={out}"], capsys)
    assert code == 2 and "already exists" in stderr and "Traceback" not in stderr
    assert hashlib.sha256(out.read_bytes()).hexdigest() == before


def test_predict_of_a_csv_gives_one_window_and_a_quantile_run_writes_its_levels(csv_run, tmp_path, capsys):
    out = tmp_path / "csv.json"
    code, _, stderr = run_command(["predict", f"model={csv_run.run}", f"data={csv_run.csv}", f"out={out}"], capsys)
    assert code == 0, stderr[-1500:]
    assert np.array(json.loads(out.read_text())["point"]).shape == (1, HORIZON, 2)


def test_predict_of_a_quantile_model_writes_levels_and_quantiles(world, tmp_path_factory, tmp_path, capsys):
    run = trained_case("prism_quantile", tmp_path_factory)
    data, out = write_npy(tmp_path / "w.npy", raw_windows(2, 1)), tmp_path / "q.json"
    code, _, stderr = run_command(["predict", f"model={run.run}", f"data={data}", f"out={out}"], capsys)
    assert code == 0, stderr[-1500:]
    answer = json.loads(out.read_text())
    assert answer["quantile_levels"] == QUANTILES and np.array(answer["quantiles"]).shape == (2, HORIZON, 1, 3)


def test_predict_reads_an_npz_of_one_array_and_refuses_one_of_several(one, tmp_path, capsys):
    windows = raw_windows(2, 1, seed=19)
    single, many = tmp_path / "one.npz", tmp_path / "many.npz"
    np.savez(single, windows=windows)
    np.savez(many, a=windows, b=windows)
    code, _, stderr = run_command(["predict", f"model={one.run}", f"data={single}", f"out={tmp_path / 'a.json'}"], capsys)
    assert code == 0, stderr[-1500:]
    code, _, stderr = run_command(["predict", f"model={one.run}", f"data={many}", f"out={tmp_path / 'b.json'}"], capsys)
    assert code == 2 and "holds 2 arrays ['a', 'b']" in stderr and not (tmp_path / "b.json").exists()


@pytest.mark.parametrize("make, message", [
    (lambda tmp: tmp, "is a directory, not a file: give a csv file, a .npy or a .npz of windows"),
    (lambda tmp: (tmp / "w.txt").write_text("1 2 3") and tmp / "w.txt", "has the suffix '.txt': give a csv file, a .npy or a .npz of windows"),
    (lambda tmp: (tmp / "x.png").write_bytes(b"x") and tmp / "x.png", "has the suffix '.png'"),
])
def test_predict_refuses_a_directory_and_an_unknown_suffix_by_name(one, tmp_path, capsys, make, message):
    data = make(tmp_path)
    out = tmp_path / "never.json"
    code, _, stderr = run_command(["predict", f"model={one.run}", f"data={data}", f"out={out}"], capsys)
    assert code == 2 and message in stderr and "holds no image" not in stderr and not out.exists()


def test_predict_refuses_a_window_of_the_wrong_length_with_the_predictors_sentence_and_writes_nothing(one, tmp_path, capsys):
    data, out = write_npy(tmp_path / "w.npy", raw_windows(1, 1, lookback=LOOKBACK + 3)), tmp_path / "w.json"
    code, _, stderr = run_command(["predict", f"model={one.run}", f"data={data}", f"out={out}"], capsys)
    assert code == 2 and "window 0 is (35, 1)" in stderr and not out.exists()


def test_predict_never_overwrites_an_out_that_appears_while_it_runs(one, tmp_path, monkeypatch):
    """``out`` is free when the command starts (the early check) and may be taken by the time the answer is written: the ``x`` mode."""
    from train.api import forecast_predictor

    data, out = write_npy(tmp_path / "w.npy", raw_windows(1, 1)), tmp_path / "taken.json"
    real = forecast_predictor.ForecastPredictor.predict

    def predict_and_take_the_out(self, x):
        out.write_text("someone else's answer")
        return real(self, x)

    monkeypatch.setattr(forecast_predictor.ForecastPredictor, "predict", predict_and_take_the_out)
    with pytest.raises(FileExistsError):
        cli.main(["predict", f"model={one.run}", f"data={data}", f"out={out}"])
    assert out.read_text() == "someone else's answer"


def test_a_forecast_that_is_not_finite_is_refused_before_anything_is_written(one, tmp_path, monkeypatch):
    """JSON has no NaN: a model that forecasts one is named, and ``out`` is not created."""
    from train.api import forecast_predictor

    real = forecast_predictor.ForecastPredictor.predict

    def predict_nan(self, x):
        answer = real(self, x)
        answer["point"][0, 0, 0] = np.nan
        return answer

    monkeypatch.setattr(forecast_predictor.ForecastPredictor, "predict", predict_nan)
    out = tmp_path / "nan.json"
    with pytest.raises(ConfigError, match=r"forecast a non-finite point.*nothing is written"):
        forecast_predictor.write_predictions(one.predictor, tmp_path / "w.npy", raw_windows(1, 1), out)
    assert not out.exists()


def test_predict_reads_the_record_before_it_lists_the_data(tmp_path, capsys):
    """D-012: a run with no ``task.json`` is said so first, whatever the data is (it used to be 'holds no image')."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "best_model.keras").write_bytes(b"not a model")
    empty = tmp_path / "empty"
    empty.mkdir()
    code, _, stderr = run_command(["predict", f"model={run}", f"data={empty}", f"out={tmp_path / 'p.json'}"], capsys)
    assert code == 2 and "task.json does not exist" in stderr and "holds no image" not in stderr
    code, _, stderr = run_command(["predict", f"model={run}", f"data={tmp_path / 'gone'}", f"out={tmp_path / 'p.json'}"], capsys)
    assert code == 2 and "Invalid value for 'data': " in stderr and "does not exist" in stderr, "a data path that is not there is still said first"


def test_the_help_of_predict_names_the_forecasting_inputs(capsys):
    code, stdout, _ = run_command(["predict", "help"], capsys)
    assert code == 0 and "csv" in stdout and ".npy" in stdout and "JSON" in stdout


# ---------------------------------------------------------------------
# Export: the served function, the rank-2 spec and the cells of the table
# ---------------------------------------------------------------------
#
# The file of a forecaster takes RAW windows ``(batch, lookback, channels)`` in the series' units and returns the forecast in the same
# units. Every trained family is exported in every format the table of ``forecast_predictor.EXPORT_REFUSALS`` does not refuse, and each
# file is run in ITS OWN runtime (Keras reload, ``tf.saved_model.load``, the TFLite interpreter, onnxruntime):
#
# * it equals ``ForecastPredictor`` on raw windows within ``TOLERANCE`` of the window's std (a share, per window and per output);
# * a window scaled by 1000 and shifted by 50 gives the forecast scaled and shifted, through the file (the oracle that sees a missing
#   or wrong de-standardization, and a file that standardizes with fixed numbers);
# * a quantile head gives ``{"point", "quantiles"}`` from ``saved_model`` and ``keras``, one tensor from ``tflite`` and ``onnx``.
#
# TOLERANCE is chosen from the gaps MEASURED on tiny untrained models at (64, 1), as a share of the window's std: saved_model
# 3.4e-6, tflite 5.3e-5, onnx 7.1e-5 (worst over the cells). saved_model and keras get 3e-5 (9 times the measured worst: the same
# graph in the same runtime, float32 against the Predictor's float64 statistics), tflite and onnx 5e-4 (9 and 7 times: another runtime's
# kernels, and a trained model's weights are not the untrained ones). Every defect the file can have is orders above it: a missing
# de-standardization is 1e1 to 1e2 of the std here, a floor on the wrong side 1e-2 or more, a standardization with fixed numbers 1e0.

TOLERANCE = {"keras": 3e-5, "saved_model": 3e-5, "tflite": 5e-4, "onnx": 5e-4}
FORMATS = ("keras", "saved_model", "tflite", "onnx")
NAMES = {"keras": "m.keras", "saved_model": "m_dir", "tflite": "m.tflite", "onnx": "m.onnx"}

#: The cells MEASURED as not exporting, each with a word of the measured reason: (decode row, format) -> text.
MEASURED_REFUSED = {
    ("timestral", "tflite"): "Cumsum",
    ("timestral", "onnx"): "ort_value_index",
    ("xlstm_point", "onnx"): "InvalidGraph",
    ("xlstm_quantile", "onnx"): "InvalidGraph",
    ("ets", "onnx"): "Cannot find node with output",
}
WORKING = [(case, fmt) for case in CASES for fmt in FORMATS if (case, fmt) not in MEASURED_REFUSED]
REFUSED = sorted(MEASURED_REFUSED)


def _onnx_toolchain_problem():
    """Why the onnx cells cannot run in this environment, or ``None``: both packages must import."""
    import importlib

    problems = []
    for name in ("tf2onnx", "onnxruntime"):
        try:
            importlib.import_module(name)
        except ImportError as exc:
            problems.append(f"{name}: {exc}")
    return "; ".join(problems) or None


ONNX_PROBLEM = _onnx_toolchain_problem()
needs_onnx = pytest.mark.skipif(ONNX_PROBLEM is not None, reason=f"the onnx cells are NOT run here, the toolchain does not import ({ONNX_PROBLEM})")


def cell_params(cells):
    return [pytest.param(case, fmt, id=f"{case}-{fmt}", marks=[needs_onnx] if fmt == "onnx" else []) for case, fmt in cells]


_EXPORTS: Dict[Any, Path] = {}


def export_cell(case: str, fmt: str, factory) -> Path:
    """The file ``python -m train export`` writes for the trained ``case`` in ``fmt`` (once per module)."""
    key = (case, fmt)
    if key not in _EXPORTS:
        out = factory.mktemp(f"export_{case}_{fmt}") / NAMES[fmt]
        code = cli.main(["export", f"model={trained_case(case, factory).run}", f"out={out}", f"format={fmt}"])
        assert code == 0 and out.exists(), f"{case} {fmt}: exit {code}"
        _EXPORTS[key] = out
    return _EXPORTS[key]


def run_file(path: Path, fmt: str, windows: np.ndarray, one_at_a_time: bool = False):
    """Run the file in its own runtime on raw ``windows``: ``(point, quantiles or None, answered_with_a_dict)``."""
    import tensorflow as tf

    x = np.asarray(windows, np.float32)
    if fmt == "keras":
        model = keras.saving.load_model(str(path))
        call = lambda batch: model(batch)  # noqa: E731
    elif fmt == "saved_model":
        loaded = tf.saved_model.load(str(path))  # kept alive: its endpoint holds only weak references to the variables
        call = lambda batch: loaded.serve(tf.constant(batch))  # noqa: E731
    elif fmt == "tflite":
        blob = path.read_bytes()
        call = lambda batch: fd_tflite(blob, batch)  # noqa: E731
    else:
        import onnxruntime

        session = onnxruntime.InferenceSession(path.read_bytes(), providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name
        call = lambda batch: session.run(None, {name: batch})[0]  # noqa: E731
    parts = [call(x[i:i + 1]) for i in range(len(x))] if one_at_a_time else [call(x)]
    as_dict = isinstance(parts[0], dict)
    if as_dict:
        return (np.concatenate([np.asarray(p["point"], np.float64) for p in parts]),
                np.concatenate([np.asarray(p["quantiles"], np.float64) for p in parts]), True)
    return np.concatenate([np.asarray(p, np.float64) for p in parts]), None, False


def fd_tflite(blob: bytes, batch: np.ndarray) -> np.ndarray:
    from train.api import exporting

    return exporting._tflite_logits(blob, batch)


def share_of_std(got: np.ndarray, want: np.ndarray, windows: np.ndarray) -> float:
    """The worst ``|got - want|`` as a share of the std of the window it forecasts (per window and channel)."""
    std = windows.std(axis=1)
    std = std[:, None, :, None] if got.ndim == 4 else std[:, None, :]
    return float((np.abs(got - want) / std).max())


def one_at_a_time(case: str) -> bool:
    return case == "timestral"  # a static batch of one window per call


@pytest.mark.parametrize("case, fmt", cell_params(WORKING))
def test_every_working_cell_equals_the_predictor_on_raw_windows(case, fmt, world, tmp_path_factory):
    path = export_cell(case, fmt, tmp_path_factory)
    run = trained_case(case, tmp_path_factory)
    windows = raw_windows(4, 1, seed=3)
    want = run.predictor.predict(windows)
    point, quantiles, as_dict = run_file(path, fmt, windows, one_at_a_time(case))
    assert point.shape == (4, HORIZON, 1) and np.isfinite(point).all()
    assert share_of_std(point, want["point"], windows) <= TOLERANCE[fmt], (case, fmt)
    has_quantiles = HEADS[case][0] == "quantile"
    assert as_dict == (has_quantiles and fmt in ("keras", "saved_model")), "saved_model and keras give a dict for a quantile head, tflite and onnx one tensor"
    if as_dict:
        assert quantiles.shape == want["quantiles"].shape
        assert share_of_std(quantiles, want["quantiles"], windows) <= TOLERANCE[fmt]


@pytest.mark.parametrize("case, fmt", cell_params(WORKING))
def test_every_working_cell_takes_raw_windows_so_a_scaled_and_shifted_window_gives_a_scaled_and_shifted_forecast(case, fmt, world, tmp_path_factory):
    path = export_cell(case, fmt, tmp_path_factory)
    windows = raw_windows(4, 1, seed=5)
    plain, plain_q, _ = run_file(path, fmt, windows, one_at_a_time(case))
    moved, moved_q, _ = run_file(path, fmt, windows * 1000.0 + 50.0, one_at_a_time(case))
    assert share_of_std(moved, plain * 1000.0 + 50.0, windows * 1000.0) <= TOLERANCE[fmt], (case, fmt)
    assert np.abs(plain - windows.mean(axis=1, keepdims=True)).max() < 40 * windows.std(axis=1).max(), "in the units of the window"
    if plain_q is not None:
        assert share_of_std(moved_q, plain_q * 1000.0 + 50.0, windows * 1000.0) <= TOLERANCE[fmt]


def test_the_working_cells_are_the_measured_matrix(world):
    """31 of 36 cells work (measured: keras 9, saved_model 9, tflite 8, onnx 5): the table refuses exactly the five that do not."""
    from train.api import forecast_predictor

    assert set(forecast_predictor.EXPORT_REFUSALS) == set(MEASURED_REFUSED)
    assert len(CASES) * len(FORMATS) == 36 and len(WORKING) == 31 and len(REFUSED) == 5
    per_format = {fmt: sum(1 for _, f in WORKING if f == fmt) for fmt in FORMATS}
    assert per_format == {"keras": 9, "saved_model": 9, "tflite": 8, "onnx": 5}
    assert sorted(CASES) == sorted(fr.DECODE), "a decode row without a case would have no cell tested"
    for cell, word in MEASURED_REFUSED.items():
        assert word in forecast_predictor.EXPORT_REFUSALS[cell], f"the reason of {cell} quotes the measured text"


@pytest.mark.parametrize("case, fmt", [pytest.param(c, f, id=f"{c}-{f}", marks=[needs_onnx] if f == "onnx" else []) for c, f in REFUSED])
def test_a_refused_cell_is_a_named_error_with_the_measured_reason_and_writes_nothing(case, fmt, world, tmp_path_factory, tmp_path, capsys):
    run = trained_case(case, tmp_path_factory)
    target = tmp_path / NAMES[fmt]
    code, _, stderr = run_command(["export", f"model={run.run}", f"out={target}", f"format={fmt}"], capsys)
    assert code == 2 and "Traceback" not in stderr, stderr[-800:]
    assert f"'{fmt}' of a {case} model is refused: " in stderr and MEASURED_REFUSED[(case, fmt)] in stderr, stderr[-800:]
    assert list(tmp_path.iterdir()) == [], "a refused cell writes nothing: no file, no .task.json, no .saved_model"


def test_a_refused_cell_is_raised_by_the_library_before_the_target_is_touched(one, tmp_path):
    from train.api import exporting

    record = dict(one.record, decode="timestral", family="nbeats")
    with pytest.raises(ConfigError, match="'tflite' of a timestral model is refused: .*Cumsum"):
        exporting.export(model=one.predictor.model, out=tmp_path / "m.tflite", format="tflite", input_shape=(LOOKBACK, 1), record=record)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("shape", [(LOOKBACK + 1, 1), (LOOKBACK, 2)])
def test_a_shape_that_is_not_the_records_window_is_refused_before_anything_is_written(one, tmp_path, shape):
    from train.api import exporting

    with pytest.raises(ConfigError, match=r"is not the window \(32, 1\) \(lookback, channels\)"):
        exporting.export(model=one.predictor.model, out=tmp_path / "m_dir", format="saved_model", input_shape=shape, record=one.record)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("case", ["nbeats", "xlstm_point"])
def test_the_library_keras_export_runs_the_shape_check_of_the_other_formats(case, world, tmp_path_factory, tmp_path):
    """Review W7: ``export(format="keras", input_shape=(64, 1))`` of an xlstm run trained at lookback 32 wrote a lookback-64 file
    beside a ``task.json`` that says 32, because the shape check ran only where a spec was traced (not for ``keras``). The CLI has no
    ``input_shape`` key, so the library function is the path (D-042)."""
    from train.api import exporting

    run = trained_case(case, tmp_path_factory)
    for shape in ((2 * LOOKBACK, 1), (LOOKBACK, 2)):
        with pytest.raises(ConfigError, match=r"is not the window \(32, 1\) \(lookback, channels\)"):
            exporting.export(model=run.predictor.model, out=tmp_path / "w.keras", format="keras", input_shape=shape, record=run.record)
        assert list(tmp_path.iterdir()) == [], f"a refused shape {shape} writes nothing"
    written = exporting.export(model=run.predictor.model, out=tmp_path / "ok.keras", format="keras",
                               input_shape=(LOOKBACK, 1), record=run.record)
    assert written.exists() and read_json(written.with_name("ok.keras.task.json")) == run.record


def test_the_keras_ops_standardization_is_the_numpy_twin_rule_for_rule():
    """The function a file runs is ``context_stats_ops``: the first-value anchor, the offset from it, the population std and the unit
    scale of a flat context equal ``context_stats_np`` (the rule the Predictor and the bar use), on the cases where a simpler rule
    differs (D-040): a level far above the spread, micro units, a flat context at a level and at zero."""
    floor = 1e-6
    base = np.random.default_rng(1).normal(size=(6, LOOKBACK, 2))
    cases = {
        "plain": base * 7 + 50, "micro units": base * 1e-6 + 2e-5, "huge": base * 1e6,
        "a level 1e5 over the spread": base * 8 + 1e5, "a level 1e6 over the spread": base * 8 + 1e6,
        "flat, not zero (the unit scale)": np.full((1, LOOKBACK, 2), 3.5),
        "flat zero (the unit scale)": np.zeros((1, LOOKBACK, 2)),
        "flat micro units": np.full((1, LOOKBACK, 2), 4e-7),
    }
    for name, window in cases.items():
        window = np.asarray(window, np.float32).astype(np.float64)
        anchor, offset, scale = fd.context_stats_np(window, floor)
        got = (np.asarray(t, np.float64) for t in fd.context_stats_ops(np.asarray(window, np.float32), floor))
        got_anchor, got_offset, got_scale = got
        np.testing.assert_array_equal(got_anchor, anchor, err_msg=name)
        np.testing.assert_allclose(got_offset, offset, rtol=1e-5, atol=1e-6 * float(np.abs(window - anchor).max()), err_msg=name)
        np.testing.assert_allclose(got_scale, scale, rtol=1e-4, atol=1e-30, err_msg=name)
    assert float(fd.context_stats_ops(np.zeros((1, LOOKBACK, 1), np.float32), floor)[2]) == fd.UNIT_SCALE


@pytest.mark.parametrize("case", ["nbeats", "prism_quantile"])
@pytest.mark.parametrize("fmt", ["keras", "saved_model"])
@pytest.mark.parametrize("level", [1e3, 1e5, 1e6])
def test_a_file_agrees_with_predict_at_a_level_far_above_the_spread(case, fmt, level, world, tmp_path_factory):
    """D-040 (review W1): the same float32 windows, std about 8, at a level of 1e3, 1e5 and 1e6 given to the file and to ``predict``.

    The agreement is 1e-4 of the window std PLUS one float32 step of the forecast itself: the file answers in float32, so at a level of
    1e5 its numbers sit on a grid of 0.0078 (1e-3 of this std) and at 1e6 on 0.0625, whatever the rule; that representation
    step is the floor of any float32 file, and it is the only slack allowed. Before D-040 the gap was 6e-4 of the std at 1e3, 0.25
    at 1e5 and 5.8 at 1e6 (prism quantile), 3 to 7 orders above this bound."""
    path = export_cell(case, fmt, tmp_path_factory)
    run = trained_case(case, tmp_path_factory)
    windows = np.asarray(raw_windows(4, 1, seed=3) - 50.0 + level, np.float32).astype(np.float64)
    want = run.predictor.predict(windows)
    point, quantiles, _ = run_file(path, fmt, windows, one_at_a_time(case))
    std = windows.std(axis=1)[:, None, :]
    grid = float(np.spacing(np.float32(abs(want["point"]).max())))
    assert np.isfinite(point).all()
    assert float((np.abs(point - want["point"]) - grid).max()) <= 1e-4 * float(std.min()), (case, fmt, level)
    if quantiles is not None:
        assert float((np.abs(quantiles - want["quantiles"]) - grid).max()) <= 1e-4 * float(std.min()), (case, fmt, level)


# --- the spec: rank 2 only for a rule whose shape_of returns it -------------------------------------


def test_a_rank_two_shape_is_accepted_only_where_the_rule_says_rank_two(one):
    from train.api import exporting

    spec = exporting._input_spec(one.predictor.model, (LOOKBACK, 1), rank=2)
    assert tuple(spec.shape) == (None, LOOKBACK, 1) and spec.dtype.name == "float32"
    assert tuple(exporting._input_spec(one.predictor.model, (LOOKBACK, 1), rank=2, batch=1).shape) == (1, LOOKBACK, 1)
    with pytest.raises(ConfigError, match=r"expected \(height, width, channels\) of positive ints"):
        exporting._input_spec(one.predictor.model, (LOOKBACK, 1), rank=3)
    with pytest.raises(ConfigError, match=r"expected \(height, width, channels\) of positive ints"):
        exporting._input_spec(one.predictor.model, (LOOKBACK, 1))
    with pytest.raises(ConfigError, match=r"expected \(lookback, channels\) of positive ints"):
        exporting._input_spec(one.predictor.model, (8, 8, 1), rank=2)
    with pytest.raises(ConfigError, match=r"expected \(lookback, channels\) of positive ints"):
        exporting._input_spec(one.predictor.model, (None, 1), rank=2)


def test_the_three_older_tasks_keep_no_served_function_and_the_forecaster_has_one():
    assert predictor.ExportRule._field_defaults["served"] is None, "the default is a no-op: the image tasks export the raw model"
    older = [task for task in predictor.EXPORT_RULES if task != TASK]
    assert sorted(older) == ["image_classification", "image_denoising", "image_segmentation"]
    assert [predictor.EXPORT_RULES[task].served for task in older] == [None, None, None]
    assert predictor.EXPORT_RULES[TASK].served is not None
    assert len(predictor.EXPORT_RULES[TASK].shape_of(one_record())) == 2
    assert all(len(predictor.EXPORT_RULES[task].shape_of({"input_shape": [8, 8, 1], "input_size": [8, 8], "channels": 1})) == 3 for task in older)


def one_record() -> Dict[str, Any]:
    return {"lookback": LOOKBACK, "channels": 1}


def test_a_static_batch_family_is_traced_at_one_window_and_the_others_at_a_free_batch(world, tmp_path_factory):
    from train.api import forecast_predictor

    for case in ("timestral", "nbeats"):
        run = trained_case(case, tmp_path_factory)
        layer = forecast_predictor.served(run.predictor.model, run.record)
        assert layer.static_batch == (forecast_predictor.STATIC_BATCH if case == "timestral" else None) and forecast_predictor.STATIC_BATCH == 1


def test_a_timestral_file_takes_one_window_per_call_and_refuses_two(world, tmp_path_factory):
    """The family packs rows by the batch: the file is traced at one window. (Several windows, one call each, equal ``predict`` of the
    batch in the cell tests above: the rows of a batch are independent.)"""
    import tensorflow as tf

    path = export_cell("timestral", "saved_model", tmp_path_factory)
    loaded = tf.saved_model.load(str(path))
    assert tuple(loaded.serve(tf.constant(raw_windows(1, 1).astype(np.float32)))["point"].shape) == (1, HORIZON, 1)
    with pytest.raises(TypeError, match=r"TensorSpec\(shape=\(1, 32, 1\)"):
        loaded.serve(tf.constant(raw_windows(2, 1).astype(np.float32)))


# --- the tflite check compares the SERVED function ----------------------------------------------------


@pytest.mark.parametrize("case", ["nbeats", "tirex"])
def test_a_wrong_tflite_file_is_refused_for_a_builtin_cell_and_for_a_select_ops_cell(case, world, tmp_path_factory, tmp_path, monkeypatch):
    """nbeats converts with built-in operators, tirex needs select TF operators; both are checked against the served function."""
    from train.api import exporting

    real = exporting._tflite_logits
    monkeypatch.setattr(exporting, "_tflite_logits", lambda blob, batch: real(blob, batch) + 0.5)
    run = trained_case(case, tmp_path_factory)
    out = tmp_path / "m.tflite"
    with pytest.raises(ConfigError, match=r"the tflite file converted with TensorFlow operators differs from the Keras model by"):
        exporting.export(model=run.predictor.model, out=out, format="tflite", input_shape=(LOOKBACK, 1), record=run.record)
    assert not out.exists() and not (tmp_path / "m.tflite.task.json").exists()


def test_the_tflite_check_reads_the_served_function_and_not_the_raw_model(one, world, tmp_path_factory):
    """The same file passes against the served function and is refused against the raw model: the check is not the old one."""
    from train.api import exporting, forecast_predictor

    path = export_cell("nbeats", "tflite", tmp_path_factory)
    spec = exporting._input_spec(one.predictor.model, (LOOKBACK, 1), rank=2)
    layer = forecast_predictor.served(one.predictor.model, one.record)
    exporting._check_tflite_against_keras(path.read_bytes(), one.predictor.model, spec, path, lambda batch: layer(batch))
    with pytest.raises(Exception):  # noqa: B017 - the raw model returns a tuple and standardized units: it cannot be compared
        exporting._check_tflite_against_keras(path.read_bytes(), one.predictor.model, spec, path)


# --- the keras file: a registered layer, reloaded in a fresh interpreter -----------------------------------

_CHILD = """
import sys
import keras
import train.common.forecast_run  # the module that registers ServedForecaster: the file names it, the process must be able to import it
import numpy as np

paths, data, out = sys.argv[1].split(","), np.load(sys.argv[2]), sys.argv[3]
answers = {}
for index, path in enumerate(paths):
    model = keras.saving.load_model(path)
    x = data[f"x{index}"]
    got = [model(x[i:i + 1]) if data[f"one{index}"] else model(x) for i in range(len(x) if data[f"one{index}"] else 1)]
    got = got if isinstance(got[0], dict) else [{"point": g} for g in got]
    answers[f"p{index}"] = np.concatenate([np.asarray(g["point"], np.float64) for g in got])
    if "quantiles" in got[0]:
        answers[f"q{index}"] = np.concatenate([np.asarray(g["quantiles"], np.float64) for g in got])
np.savez(out, **answers)
"""

#: A process in which the ``train`` package cannot be imported: every path entry that holds a ``train`` directory is dropped and one
#: directory holding only a link to ``dl_techniques`` is put in front (the wheel ships ``dl_techniques*`` and not ``train``).
_CHILD_WITHOUT_TRAIN = """
import os
import sys

sys.path = [p for p in sys.path if not os.path.isdir(os.path.join(p, "train"))]
sys.path.insert(0, sys.argv[2])
import keras

try:
    keras.saving.load_model(sys.argv[1])
except Exception as exc:
    sys.stdout.write(type(exc).__name__ + ": " + " ".join(str(exc).split()))
    sys.exit(3)
"""


def test_every_keras_file_reloads_in_a_fresh_interpreter_that_imports_the_train_module_and_keras(world, tmp_path_factory, tmp_path):
    """One child process imports ``train.common.forecast_run`` and ``keras`` and loads the nine files (D-041: the file needs the
    ``train`` package importable; it does not load without it, see the next test)."""
    import subprocess
    import sys

    cases = list(CASES)
    windows = raw_windows(3, 1, seed=21).astype(np.float32)
    arrays, files = {}, []
    for index, case in enumerate(cases):
        files.append(str(export_cell(case, "keras", tmp_path_factory)))
        arrays[f"x{index}"], arrays[f"one{index}"] = windows, np.asarray(one_at_a_time(case))
    np.savez(tmp_path / "in.npz", **arrays)
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    done = subprocess.run([sys.executable, str(script), ",".join(files), str(tmp_path / "in.npz"), str(tmp_path / "out.npz")],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-1500:]
    answers = np.load(tmp_path / "out.npz")
    for index, case in enumerate(cases):
        want = trained_case(case, tmp_path_factory).predictor.predict(windows)
        assert share_of_std(answers[f"p{index}"], want["point"], windows) <= TOLERANCE["keras"], case
        assert (f"q{index}" in answers.files) == (HEADS[case][0] == "quantile"), case


def test_an_exported_keras_file_does_not_load_in_a_process_where_the_train_package_cannot_be_imported(world, tmp_path_factory, tmp_path):
    """Review W3: the layer the file holds is a class of ``train``, which the wheel does not ship. The failure a user meets is Keras's
    own sentence, named by the class and the module (a keras version that words it otherwise fails this test, which is the signal)."""
    import subprocess
    import sys

    only = tmp_path / "only_dl_techniques"
    only.mkdir()
    (only / "dl_techniques").symlink_to(SRC / "dl_techniques", target_is_directory=True)
    script = tmp_path / "child.py"
    script.write_text(_CHILD_WITHOUT_TRAIN)
    done = subprocess.run([sys.executable, str(script), str(export_cell("nbeats", "keras", tmp_path_factory)), str(only)],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 3, done.stdout[-800:] + done.stderr[-800:]
    assert "ServedForecaster" in done.stdout and "train.common.forecast_run cannot be imported" in done.stdout, done.stdout[-800:]


#: The phrase that claimed a forecaster's keras file loads on its own: the call, the word ``alone``, across a line break and a code quote.
_ALONE = re.compile("load_model" + "`*" + r"\s+alone", re.S)


def test_no_text_claims_that_the_keras_call_alone_loads_an_exported_forecaster():
    """D-041: the README, the AGENTS file, every docstring, comment, help text and test under ``train`` and its tests say what the file
    needs (the ``train`` package importable); none says the Keras loader is enough."""
    roots = [SRC / "train", Path(__file__).resolve().parent.parent]
    texts = [path for root in roots for pattern in ("*.md", "*.py") for path in sorted(root.rglob(pattern))]
    assert len(texts) > 100, "vacuity guard: the scan reads the tree"
    found = [str(path.relative_to(SRC.parent)) for path in texts if _ALONE.search(path.read_text())]
    assert found == [], f"a text claims the Keras loader alone loads the forecaster's file: {found}"


def test_the_requirement_of_the_keras_file_is_stated_in_the_note_the_help_and_the_two_documents(one):
    from train.api import forecast_predictor

    needs = ("ServedForecaster", "train package", "importable", "does not ship")
    note = forecast_predictor.export_note(one.record)
    assert all(word in note for word in needs), note
    help_text = " ".join(" ".join(cli.EXPORT_NOTES).split())
    assert all(word in help_text for word in ("train package", "importable", "does not ship")), help_text
    for name in ("README.md", "AGENTS.md"):
        text = " ".join((SRC / "train" / name).read_text().split())
        assert all(word in text for word in needs), name


@pytest.mark.parametrize("case", list(CASES))
def test_an_exported_keras_file_is_not_a_run_and_load_model_refuses_it_by_name(case, world, tmp_path_factory):
    """``predict`` takes the run directory. The record beside the file is the run's, so the file is found and then refused: by the
    predictor's own sentence where the family check lets it through, by the family check (it holds a Functional) elsewhere."""
    path = export_cell(case, "keras", tmp_path_factory)
    assert (path.parent / "m.keras.task.json").exists()
    with pytest.raises(ConfigError, match=r"is an exported forecaster|but .*m\.keras holds a Functional"):
        load_model(path)


def test_the_export_writes_the_record_beside_every_file_and_logs_the_note(one, world, tmp_path_factory):
    from train.api import forecast_predictor

    path = export_cell("nbeats", "saved_model", tmp_path_factory)
    beside = path.with_name(path.name + ".task.json")
    assert read_json(beside) == one.record
    note = forecast_predictor.export_note(one.record)
    assert "RAW float32 windows (batch, 32, 1)" in note and "forecast (batch, 8, 1)" in note and "ONE window" not in note
    quantile = dict(one.record, head="quantile", decode="tirex")
    assert "saved_model and keras give {point, quantiles}" in forecast_predictor.export_note(quantile)
    assert "ONE window per call" in forecast_predictor.export_note(dict(one.record, decode="timestral"))


def test_the_export_help_names_forecasting(capsys):
    code, stdout, _ = run_command(["export", "help"], capsys)
    assert code == 0 and "forecasting model takes RAW float32 windows" in stdout
    for word in ("lookback", "units", "{point, quantiles}", "timestral", "tflite", "onnx"):
        assert word in stdout, word


# --- each refused cell is still refused for the reason it quotes ------------------------------------------------


def _traced(case: str, tmp_path_factory):
    """``(model, layer, spec)`` of the trained ``case``, as ``export`` builds them."""
    from train.api import exporting, forecast_predictor

    run = trained_case(case, tmp_path_factory)
    layer = forecast_predictor.served(run.predictor.model, run.record)
    spec = exporting._input_spec(run.predictor.model, (LOOKBACK, 1), rank=2, batch=layer.static_batch)
    return run.predictor.model, (lambda batch: layer(batch)["point"] if run.record["head"] == "quantile" else layer(batch)), spec


def test_the_tflite_of_timestral_fails_where_the_reason_says(world, tmp_path_factory, tmp_path):
    from train.api import exporting

    model, point, spec = _traced("timestral", tmp_path_factory)
    with pytest.raises(ConfigError, match=r"even with select TF operators.*Asinh"):
        exporting._export_tflite(model, tmp_path / "t.tflite", spec, point)
    assert not (tmp_path / "t.tflite").exists()


@needs_onnx
def test_the_onnx_of_ets_fails_in_the_converter_where_the_reason_says(world, tmp_path_factory, tmp_path):
    from train.api import exporting

    model, point, spec = _traced("ets", tmp_path_factory)
    with pytest.raises(ValueError, match="Cannot find node with output '[^']*ets_model_1/while:3'"):
        exporting._export_onnx(model, tmp_path / "e.onnx", spec, point)


@needs_onnx
@pytest.mark.parametrize("case, word", [("xlstm_point", "Pad"), ("xlstm_quantile", "Pad"), ("timestral", "ort_value_index")])
def test_the_onnx_of_xlstm_and_timestral_is_written_and_onnxruntime_refuses_it(case, word, world, tmp_path_factory, tmp_path):
    import onnxruntime

    from train.api import exporting

    model, point, spec = _traced(case, tmp_path_factory)
    out = tmp_path / "x.onnx"
    exporting._export_onnx(model, out, spec, point)
    with pytest.raises(Exception, match=word) as raised:  # noqa: B017 - onnxruntime's pybind exception types are not importable names
        onnxruntime.InferenceSession(out.read_bytes(), providers=["CPUExecutionProvider"])
    assert ("InvalidGraph" if case.startswith("xlstm") else "RUNTIME_EXCEPTION") in f"{type(raised.value).__name__} {raised.value}"
