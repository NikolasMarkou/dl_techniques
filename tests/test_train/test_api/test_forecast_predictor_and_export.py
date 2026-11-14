"""Predict with a ``time_series_forecasting`` run: the record, ``ForecastPredictor`` and the ``predict`` command (plan-2026-11-13T105600-3a8319ef step 15, D-010, D-012).

First part (step 15; the export cells are the second part, step 16). Every case of ``forecast_run.DECODE`` is TRAINED for one
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
* **Export** is not available until step 16: every format of the command is refused by name and nothing is written.

Everything writes under ``tmp_path``; nothing is deleted.
"""

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import keras
import numpy as np
import pytest

from train.api import cli, predictor
from train.api.errors import ConfigError
from train.api.predictor import RECORD_KEYS, TASK_TABLES, check_record, load_model, read_record, record_differences
from train.api.tasks.time_series_forecasting import train_time_series_forecasting
from train.common import forecast_data as fd
from train.common import forecast_run as fr

from .test_forecast_task import CASES, DATA, QUANTILES, prepared, read_json, world, write_series_csv  # noqa: F401

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
    mean, scale = fd.context_stats_np(windows, record["scale_floor"])
    x = fd.standardize_np(
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
                              ("scale_floor", 0.5, "scale_floor: 0.001 -> 0.5")):
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


@pytest.mark.parametrize("shift", [30.0, -125.0])
def test_a_window_shifted_by_a_constant_gives_the_forecast_shifted_by_it(trained, shift):
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


def test_a_record_whose_lookback_the_model_does_not_take_is_named_as_such(one, tmp_path):
    with pytest.raises(ConfigError, match=rf"does not take the lookback {LOOKBACK + 8} it names"):
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
# Export is the second part: every format is refused by name until it lands
# ---------------------------------------------------------------------


@pytest.mark.parametrize("format, name", [("keras", "m.keras"), ("saved_model", "m_dir"), ("tflite", "m.tflite"), ("onnx", "m.onnx")])
def test_every_export_format_of_a_forecasting_run_is_refused_by_name_and_writes_nothing(one, tmp_path, capsys, format, name):
    target = tmp_path / name
    code, _, stderr = run_command(["export", f"model={one.run}", f"out={target}", f"format={format}"], capsys)
    assert code == 2 and "Traceback" not in stderr
    assert "export of a time_series_forecasting model is not available yet" in stderr, stderr[-800:]
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("m")) == [], "a refused cell writes nothing"
