"""The resume primitives in ``train.common``: what a killed ``fit`` loses.

Keras 3.8 ``BackupAndRestore`` restores weights, optimizer and the next epoch
index, and nothing else (plan-2026-10-30T132432-a865fa08, findings section C):
the CSV rewrites itself empty, ``ModelCheckpoint`` forgets its best, ``EarlyStopping``
forgets ``best``, ``wait`` and ``best_weights``, and ``History`` holds only the new
epochs. One test per primitive, each with its defaults-unchanged arm, then one
test that composes them on a tiny model in the order the module documents and
compares the resumed run with an uninterrupted one.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

import keras
import numpy as np
import pytest

from dl_techniques.utils.logger import logger
from train.common import classification_viz as viz
from train.common import run_artifacts
from train.common.callbacks import EpochCSVLogger, create_callbacks
from train.common.resume_state import (
    FIT_STATE_NAME,
    FitStateSidecar,
    ResumePoint,
    RestoreFitState,
    copy_log_aside,
    merge_history,
    prior_rows,
    read_backup_epoch,
    read_fit_state,
    read_resume_point,
    rows_as_history,
    update_fit_state,
)


def _read(path):
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1:]


def _model():
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((3,)), keras.layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.SGD(0.05), loss="mse")
    return model


def _data():
    x = np.random.default_rng(0).standard_normal((8, 3)).astype("float32")
    return x, x.sum(axis=1, keepdims=True)


class Scripted(keras.callbacks.Callback):
    """Write a scripted ``val_loss`` into the epoch logs; goes FIRST in the list."""

    def __init__(self, values):
        super().__init__()
        self.values = values

    def on_epoch_end(self, epoch, logs=None):
        logs["val_loss"] = self.values[min(epoch, len(self.values) - 1)]


class KillAt(keras.callbacks.Callback):
    """Raise at the start of one epoch: a stand-in for SIGKILL between epochs."""

    def __init__(self, epoch):
        super().__init__()
        self.epoch = epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.epoch:
            raise RuntimeError("killed")


# ---- EpochCSVLogger(initial_rows)


def test_initial_rows_come_back_ahead_of_the_resumed_epochs(tmp_path):
    path = tmp_path / "log.csv"
    earlier = [(0, {"loss": 1.0, "val_loss": 0.5}), (1, {"loss": 0.9, "val_loss": 0.9})]
    logger_cb = EpochCSVLogger(str(path), initial_rows=earlier)
    logger_cb.on_train_begin()
    logger_cb.on_epoch_end(2, {"loss": 0.8, "val_loss": 0.9})
    header, rows = _read(path)
    assert header == ["epoch", "loss", "val_loss"]
    assert [row[0] for row in rows] == ["0", "1", "2"]
    assert [float(row[1]) for row in rows] == [1.0, 0.9, 0.8]


def test_initial_rows_leave_the_file_complete_before_any_new_epoch(tmp_path):
    path = tmp_path / "log.csv"
    cb = EpochCSVLogger(str(path), initial_rows=[(0, {"loss": 1.0})])
    cb.on_train_begin()
    cb.csv_file.flush()
    assert _read(path)[1] == [["0", "1.0"]]


def test_a_key_first_seen_after_the_earlier_rows_widens_the_header_once(tmp_path):
    path = tmp_path / "log.csv"
    cb = EpochCSVLogger(str(path), initial_rows=[(0, {"loss": 1.0})])
    cb.on_train_begin()
    cb.on_epoch_end(1, {"loss": 0.9, "val_loss": 0.4})
    header, rows = _read(path)
    assert header == ["epoch", "loss", "val_loss"]
    assert rows == [["0", "1.0", ""], ["1", "0.9", "0.4"]]


def test_the_default_has_no_initial_rows_and_append_is_still_refused(tmp_path):
    path = tmp_path / "log.csv"
    cb = EpochCSVLogger(str(path))
    cb.on_train_begin()
    cb.on_epoch_end(0, {"loss": 1.0})
    assert _read(path)[1] == [["0", "1.0"]]
    with pytest.raises(ValueError, match="does not append"):
        EpochCSVLogger(str(path), append=True, initial_rows=[(0, {"loss": 1.0})])


# ---- create_callbacks pass-through


def test_create_callbacks_hands_the_threshold_and_the_rows_to_their_callbacks(tmp_path):
    callbacks, _ = create_callbacks(
        "m", run_dir=str(tmp_path), monitor="val_loss", include_analyzer=False,
        initial_value_threshold=0.5, csv_initial_rows=[(0, {"val_loss": 0.5})])
    checkpoint = next(c for c in callbacks if isinstance(c, keras.callbacks.ModelCheckpoint))
    csv_logger = next(c for c in callbacks if isinstance(c, EpochCSVLogger))
    assert checkpoint.best == 0.5
    assert csv_logger._initial_rows == [(0, {"val_loss": 0.5})]


def test_create_callbacks_defaults_are_the_legacy_ones(tmp_path):
    callbacks, _ = create_callbacks("m", run_dir=str(tmp_path), monitor="val_loss",
                                    include_analyzer=False)
    checkpoint = next(c for c in callbacks if isinstance(c, keras.callbacks.ModelCheckpoint))
    csv_logger = next(c for c in callbacks if isinstance(c, EpochCSVLogger))
    assert checkpoint.best == float("inf")  # min mode, nothing recorded yet
    assert csv_logger._initial_rows == []


# ---- attach_run_log(mode)


def test_attach_run_log_appends_in_append_mode_and_truncates_by_default(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("killed segment line\n")
    with run_artifacts.attach_run_log(tmp_path, mode="a") as handler:
        assert handler.mode == "a"
        logger.info("resumed segment line")
    text = log.read_text()
    assert "killed segment line" in text and "resumed segment line" in text
    with run_artifacts.attach_run_log(tmp_path) as handler:
        assert handler.mode == "w"
    assert "killed segment line" not in log.read_text()


def test_attach_run_log_refuses_another_mode_before_touching_the_file(tmp_path):
    with pytest.raises(ValueError, match="'w' or 'a'"):
        with run_artifacts.attach_run_log(tmp_path, mode="r+"):
            pass
    assert not (tmp_path / "run.log").exists()


# ---- TrainingDashboardCallback(initial_history)


def test_a_resumed_dashboard_keeps_the_earlier_epochs_and_measures_no_baseline(tmp_path):
    calls = []
    cb = viz.TrainingDashboardCallback(
        tmp_path / "d.png", baseline_fn=lambda model: calls.append(model) or {"loss": 9.0},
        initial_history={"loss": [1.0, 0.9], "val_loss": [0.5, 0.6]},
        initial_baseline={"loss": 2.3, "accuracy": 0.1})
    cb.on_train_begin()
    assert calls == []
    assert cb.baseline == {"loss": 2.3, "accuracy": 0.1}
    cb.on_epoch_begin(2)
    cb.on_epoch_end(2, {"loss": 0.8, "val_loss": 0.7, "lr": 1e-3})
    assert cb.history["loss"] == [1.0, 0.9, 0.8]
    assert cb.history["val_loss"] == [0.5, 0.6, 0.7]
    assert len(cb.epoch_times) == 3 and math.isnan(cb.epoch_times[0]) and math.isfinite(cb.epoch_times[2])
    assert cb.out_path.exists()


def test_a_resumed_dashboard_draws_even_when_the_last_wall_time_is_clipped(tmp_path):
    cb = viz.TrainingDashboardCallback(tmp_path / "d.png", initial_history={"loss": [1.0, 0.9, 0.8]})
    cb.epoch_times.append(1.0)
    cb.epoch_times.append(50.0)
    cb.history["loss"] += [0.7, 0.6]
    cb._draw()
    assert cb.out_path.exists() and cb._drawn_epochs == 5


def test_the_default_dashboard_still_measures_its_baseline(tmp_path):
    calls = []
    cb = viz.TrainingDashboardCallback(
        tmp_path / "d.png", baseline_fn=lambda model: calls.append(model) or {"loss": 9.0})
    cb.on_train_begin()
    assert len(calls) == 1 and cb.baseline == {"loss": 9.0} and cb.epoch_times == []


# ---- read_backup_epoch, read_fit_state, update_fit_state


def test_read_backup_epoch_reads_the_completed_epoch_count(tmp_path):
    (tmp_path / "training_metadata.json").write_text(json.dumps({"epoch": 3, "batch": 0}))
    assert read_backup_epoch(tmp_path) == 3


def test_read_backup_epoch_refuses_a_missing_or_malformed_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_backup_epoch(tmp_path)
    (tmp_path / "training_metadata.json").write_text(json.dumps({"epoch": "3"}))
    with pytest.raises(ValueError, match="integer 'epoch'"):
        read_backup_epoch(tmp_path)


@pytest.mark.parametrize("text", ["[]", "3", '"a string"', "null"])
def test_a_sidecar_that_is_not_a_json_object_is_refused(tmp_path, text):
    """Mutant ``fit-state: non-object accepted``: a list would fail later on ``setdefault``."""
    path = tmp_path / FIT_STATE_NAME
    path.write_text(text)
    with pytest.raises(ValueError, match="is not a JSON object"):
        read_fit_state(path)
    path.write_text("{}")  # control: an empty object is the empty state
    assert read_fit_state(path) == {"epochs": {}, "fit_finished": False}


def test_a_missing_sidecar_reads_as_the_empty_state_and_update_merges_atomically(tmp_path):
    path = tmp_path / FIT_STATE_NAME
    assert read_fit_state(path) == {"epochs": {}, "fit_finished": False}
    update_fit_state(path, fit_finished=True, baseline={"loss": 2.3})
    state = read_fit_state(path)
    assert state["fit_finished"] is True and state["baseline"] == {"loss": 2.3}
    assert not (tmp_path / (FIT_STATE_NAME + ".tmp")).exists()


# ---- prior_rows, merge_history


def _write_csv(path, rows, header=("epoch", "loss", "val_loss")):
    path.write_text("\n".join([",".join(header)] + [",".join(r) for r in rows]) + "\n")


def test_prior_rows_returns_the_rows_below_the_start_and_drops_the_rest(tmp_path):
    path = tmp_path / "training_log.csv"
    _write_csv(path, [["0", "1.0", "0.5"], ["1", "0.9", ""], ["2", "0.8", "0.7"]])
    assert prior_rows(path, 2) == [(0, {"loss": 1.0, "val_loss": 0.5}), (1, {"loss": 0.9})]
    assert prior_rows(path, 0) == []


def test_prior_rows_ignores_a_torn_last_row_and_refuses_a_missing_epoch(tmp_path):
    path = tmp_path / "training_log.csv"
    _write_csv(path, [["0", "1.0", "0.5"], ["1", "0.9", "0.6"]])
    path.write_text(path.read_text() + "1")  # a torn row whose epoch cell reads as 1
    assert [epoch for epoch, _ in prior_rows(path, 2)] == [0, 1]
    with pytest.raises(ValueError, match="expected the row of epoch 2, found epoch 1"):
        prior_rows(path, 3)  # the torn row is read as a row of epoch 1: no third epoch
    _write_csv(path, [["0", "1.0", "0.5"], ["1", "0.9", "0.6"]])
    with pytest.raises(ValueError, match="holds 2 complete epoch rows"):
        prior_rows(path, 3)
    _write_csv(path, [["0", "1.0", "0.5"], ["2", "0.9", "0.6"]])
    with pytest.raises(ValueError, match="expected the row of epoch 1, found epoch 2"):
        prior_rows(path, 2)


def test_merge_history_prepends_the_earlier_epochs_in_place():
    x, y = _data()
    history = _model().fit(x, y, epochs=2, batch_size=4, verbose=0, validation_data=(x, y))
    held = history.history
    merge_history([(0, {"loss": 9.0, "val_loss": 8.0}), (1, {"loss": 7.0})], history)
    assert history.history is held
    assert held["loss"][:2] == [9.0, 7.0] and len(held["loss"]) == 4
    assert held["val_loss"][0] == 8.0 and len(held["val_loss"]) == 3
    assert history.epoch == [0, 1, 0, 1]


# ---- FitStateSidecar, RestoreFitState


def test_the_sidecar_records_best_and_wait_after_every_epoch(tmp_path):
    path = tmp_path / "checkpoints" / FIT_STATE_NAME
    stopper = keras.callbacks.EarlyStopping(monitor="val_loss", patience=9, mode="min")
    x, y = _data()
    _model().fit(x, y, epochs=3, batch_size=4, verbose=0,
                 callbacks=[Scripted([0.5, 0.9, 0.9]), stopper, FitStateSidecar(stopper, path)])
    epochs = read_fit_state(path)["epochs"]
    assert epochs == {
        "0": {"best": 0.5, "wait": 0, "best_epoch": 0},
        "1": {"best": 0.5, "wait": 1, "best_epoch": 0},
        "2": {"best": 0.5, "wait": 2, "best_epoch": 0},
    }
    assert not path.with_name(path.name + ".tmp").exists()


def test_the_sidecar_extends_an_earlier_segment_and_replaces_stale_entries(tmp_path):
    path = tmp_path / FIT_STATE_NAME
    update_fit_state(path, epochs={"0": {"best": 0.5, "wait": 0, "best_epoch": 0},
                                   "1": {"best": 0.5, "wait": 1, "best_epoch": 0},
                                   "2": {"best": 0.4, "wait": 0, "best_epoch": 2}},
                     baseline={"loss": 2.3})
    stopper = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min")
    stopper.best, stopper.wait, stopper.best_epoch = 0.5, 2, 0
    sidecar = FitStateSidecar(stopper, path)
    sidecar.on_train_begin()
    sidecar.on_epoch_end(2)  # a resume at epoch 2: the old entry of epoch 2 is stale
    state = read_fit_state(path)
    assert sorted(state["epochs"]) == ["0", "1", "2"]
    assert state["epochs"]["2"] == {"best": 0.5, "wait": 2, "best_epoch": 0}
    assert state["baseline"] == {"loss": 2.3}


def test_entries_of_the_resumed_epoch_and_every_later_one_are_dropped(tmp_path):
    """Mutant ``sidecar keeps stale entries``: a segment killed AFTER epoch 3's entry was written
    and BEFORE the backup claimed it leaves entries for epochs the resume trains again."""
    path = tmp_path / FIT_STATE_NAME
    update_fit_state(path, epochs={str(e): {"best": 0.5, "wait": e, "best_epoch": 0} for e in range(5)})
    stopper = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min")
    stopper.best, stopper.wait, stopper.best_epoch = 0.4, 0, 2
    sidecar = FitStateSidecar(stopper, path)
    sidecar.on_train_begin()
    sidecar.on_epoch_end(2)
    entries = read_fit_state(path)["epochs"]
    assert sorted(entries) == ["0", "1", "2"]
    assert entries["2"] == {"best": 0.4, "wait": 0, "best_epoch": 2}
    assert entries["1"]["wait"] == 1  # the earlier ones are kept as they were


def test_a_non_finite_best_is_recorded_as_null(tmp_path):
    path = tmp_path / FIT_STATE_NAME
    stopper = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min")
    stopper.best, stopper.wait, stopper.best_epoch = float("nan"), 0, 0
    sidecar = FitStateSidecar(stopper, path)
    sidecar.on_train_begin()
    sidecar.on_epoch_end(0)
    assert read_fit_state(path)["epochs"]["0"]["best"] is None


def test_a_restored_early_stopper_keeps_the_in_memory_best_weights(tmp_path):
    """Without ``best_weights`` a run that never improves ends on a worse epoch."""
    x, y = _data()
    model = _model()
    best_weights = [np.full_like(w, 3.0) for w in model.get_weights()]
    stopper = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=9, mode="min", restore_best_weights=True)
    model.fit(x, y, epochs=2, batch_size=4, verbose=0, callbacks=[
        Scripted([0.9, 0.9]), stopper,
        RestoreFitState(stopper, best=0.5, wait=1, best_weights=best_weights, best_epoch=0)])
    assert stopper.best == 0.5 and stopper.wait == 3 and stopper.best_epoch == 0
    for restored, expected in zip(model.get_weights(), best_weights):
        np.testing.assert_array_equal(restored, expected)


def test_a_restored_early_stopper_keeps_the_best_epoch_it_was_given(tmp_path):
    """Mutant ``restore: best_epoch not set``: ``EarlyStopping`` resets it to 0 at ``on_train_begin``,
    and the resumed run then reports the wrong best epoch (the default 0 hid the mutant above)."""
    x, y = _data()
    stopper = keras.callbacks.EarlyStopping(monitor="val_loss", patience=9, mode="min")
    _model().fit(x, y, epochs=1, batch_size=4, verbose=0, callbacks=[
        Scripted([0.9]), stopper, RestoreFitState(stopper, best=0.5, wait=1, best_weights=None, best_epoch=3)])
    assert stopper.best == 0.5 and stopper.best_epoch == 3


def test_the_restored_wait_decides_when_the_run_stops():
    x, y = _data()
    stopper = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min")
    # initial_epoch 4: Keras never stops at epoch index 0 (`epoch > 0`)
    history = _model().fit(x, y, epochs=9, initial_epoch=4, batch_size=4, verbose=0, callbacks=[
        Scripted([0.9]), stopper, RestoreFitState(stopper, best=0.5, wait=2, best_weights=None)])
    assert len(history.history["loss"]) == 1  # wait 2 + 1 worse epoch reaches patience 3 at once
    fresh = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min")
    history = _model().fit(x, y, epochs=9, initial_epoch=4, batch_size=4, verbose=0,
                           callbacks=[Scripted([0.9]), fresh])
    assert len(history.history["loss"]) == 4  # the first epoch sets `best`, then three worse ones


# ---- all of it, on a tiny model


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _segment(run, *, resume, kill_at=None, script=(0.5, 0.9), epochs=8):
    """One ``fit`` of a resumable run, with the callbacks in the documented order."""
    ckpt = run / "checkpoints"
    kwargs = {}
    start = 0
    best_weights = None
    state = {}
    if resume:
        start = read_backup_epoch(ckpt / "backup")
        state = read_fit_state(ckpt / FIT_STATE_NAME)["epochs"][str(start - 1)]
        kwargs = dict(initial_value_threshold=state["best"],
                      csv_initial_rows=prior_rows(run / "training_log.csv", start))
        best_weights = keras.models.load_model(run / "best_model.keras", compile=False).get_weights()
    callbacks, _ = create_callbacks(
        "tiny", run_dir=str(run), monitor="val_loss", monitor_mode="min", patience=3,
        include_analyzer=False, **kwargs)
    stopper = callbacks[0]
    ordered = [Scripted(script)] + callbacks
    if resume:
        ordered.append(RestoreFitState(stopper, state["best"], state["wait"], best_weights,
                                       state["best_epoch"]))
    ordered += [FitStateSidecar(stopper, ckpt / FIT_STATE_NAME),
                keras.callbacks.BackupAndRestore(str(ckpt / "backup"), delete_checkpoint=False)]
    if kill_at is not None:
        ordered.append(KillAt(kill_at))
    x, y = _data()
    model = _model()
    history = model.fit(x, y, epochs=epochs, batch_size=4, verbose=0, callbacks=ordered)
    if resume:
        merge_history(kwargs["csv_initial_rows"], history)
    return model, history, stopper


def test_a_killed_and_resumed_run_matches_an_uninterrupted_one(tmp_path):
    whole = tmp_path / "whole"
    killed = tmp_path / "killed"
    _, whole_history, whole_stopper = _segment(whole, resume=False)
    with pytest.raises(RuntimeError, match="killed"):
        _segment(killed, resume=False, kill_at=2)
    assert _read(killed / "training_log.csv")[1][-1][0] == "1"
    best_before = _sha(killed / "best_model.keras")

    model, history, stopper = _segment(killed, resume=True)

    epochs = [row[0] for row in _read(killed / "training_log.csv")[1]]
    assert epochs == [row[0] for row in _read(whole / "training_log.csv")[1]] == ["0", "1", "2", "3"]
    assert _sha(killed / "best_model.keras") == best_before  # a worse epoch did not overwrite it
    assert stopper.stopped_epoch == whole_stopper.stopped_epoch == 3
    assert history.epoch == whole_history.epoch == [0, 1, 2, 3]
    assert history.history["val_loss"] == whole_history.history["val_loss"]
    best = keras.models.load_model(killed / "best_model.keras", compile=False).get_weights()
    for restored, expected in zip(model.get_weights(), best):
        np.testing.assert_array_equal(restored, expected)


# ---- rows_as_history, read_resume_point, copy_log_aside


def test_rows_as_history_lists_a_key_only_for_the_epochs_that_logged_it():
    rows = [(0, {"loss": 1.0}), (1, {"loss": 0.9, "val_loss": 0.4})]
    assert rows_as_history(rows) == {"loss": [1.0, 0.9], "val_loss": [0.4]}
    assert rows_as_history([]) == {}


def _plant_run(tmp_path, start=2, entries=None):
    (tmp_path / "backup").mkdir()
    (tmp_path / "backup" / "training_metadata.json").write_text(json.dumps({"epoch": start}))
    (tmp_path / "log.csv").write_text(
        "epoch,loss\n" + "".join(f"{e},{1.0 - e / 10}\n" for e in range(start + 1)))
    update_fit_state(
        tmp_path / FIT_STATE_NAME,
        epochs={"0": {"best": 0.7, "wait": 0, "best_epoch": 0},
                "1": {"best": 0.7, "wait": 1, "best_epoch": 0}} if entries is None else entries,
        initial={"metrics": {"loss": 2.3}})


def test_read_resume_point_gathers_the_three_files(tmp_path):
    _plant_run(tmp_path)
    point = read_resume_point(tmp_path / "log.csv", tmp_path / "backup", tmp_path / FIT_STATE_NAME)
    assert isinstance(point, ResumePoint)
    assert (point.start, point.best, point.wait, point.best_epoch) == (2, 0.7, 1, 0)
    assert [epoch for epoch, _ in point.rows] == [0, 1]  # the row of epoch 2 is the surplus one
    assert point.state["initial"] == {"metrics": {"loss": 2.3}}


@pytest.mark.parametrize("start, entries, message", [
    (0, None, "records no finished epoch"),
    (2, {}, "holds no entry for epoch 1"),
    (3, None, "holds no entry for epoch 2"),
])
def test_read_resume_point_refuses_files_that_disagree(tmp_path, start, entries, message):
    _plant_run(tmp_path, start=start, entries=entries)
    with pytest.raises(ValueError, match=message):
        read_resume_point(tmp_path / "log.csv", tmp_path / "backup", tmp_path / FIT_STATE_NAME)


def test_copy_log_aside_never_overwrites_an_earlier_copy(tmp_path):
    path = tmp_path / "training_log.csv"
    path.write_text("first")
    one = copy_log_aside(path)
    path.write_text("second")
    two = copy_log_aside(path)
    assert (one.name, two.name) == ("training_log.pre_resume_1.csv", "training_log.pre_resume_2.csv")
    assert (one.read_text(), two.read_text(), path.read_text()) == ("first", "second", "second")
    with pytest.raises(FileNotFoundError):
        copy_log_aside(tmp_path / "missing.csv")
