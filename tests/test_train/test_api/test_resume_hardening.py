"""A resume survives what a kill does to the files, and refuses what it cannot make right (D-069).

Review-iter-3 killed real runs at named instants and found a resume that exits 0 with a
wrong run, a run that can never be resumed, and two processes in one directory. Each
concern has an arm here, built the way the kill really leaves the files:

* **C3** the best checkpoint is written to a temporary name and renamed
  (``AtomicModelCheckpoint``), so a kill inside ``model.save`` leaves the earlier best
  whole; a torn ``best_model.keras`` is a ``ConfigError`` naming the file on resume and on
  ``load_model``.
* **C4** ``BackupAndRestore(double_checkpoint=True)``: the files of a real run, written by
  Keras's own copy-and-save, with the latest weights cut in half afterwards. The run goes
  back one epoch and the CSV holds every epoch once.
* **C5** a kill between the backup's weights and its metadata leaves weights one epoch
  ahead of their metadata and a file that opens fine; the optimizer's step count catches it.
* **C6** other data than the run's (a renamed class, swapped class folders, added images)
  is refused naming what changed, before the CSV is touched.
* **C7** ``run.lock``, with two real processes.
* **ROP** ``ReduceLROnPlateau`` reduces at the epochs an uninterrupted run reduces at.
* **C13, C18** the help states the residual; two refusals say what to do.

The mechanism order (sidecar before backup, D-021) is pinned by a kill between them.
Tests write only under ``tmp_path``.
"""

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import keras
import numpy as np
import pytest
from PIL import Image

from train.api.errors import ConfigError
from train.api.run_dir import (
    LOCK_NAME,
    RESUME_UNCOMPARED_KEYS,
    check_model_file,
    process_stamp,
    read_lock,
    refuse_live_holder,
    request_differences,
    take_run_lock,
)
from train.api.tasks.image_classification import (
    ImageClassificationConfig,
    ImageClassificationTrainer,
    train_image_classification,
)
from train.common.callbacks import AtomicModelCheckpoint, create_callbacks
from train.common.resume_state import (
    BACKUP_METADATA_NAME,
    BACKUP_WEIGHTS_NAME,
    FIT_STATE_NAME,
    PREVIOUS_SUFFIX,
    backup_iterations,
    copy_log_aside,
    read_backup_epoch,
    read_fit_state,
    read_resume_point,
    settle_backup,
)

from .test_existing_run_is_refused import snapshot
from .test_resume_interrupted import (  # noqa: F401  (fixtures are used by name)
    REQUEST,
    KillAt,
    Scripted,
    killed,
    read_csv,
    sha,
    whole,
    world,
)
from .test_task_trainer import make_config



def cut_in_half(path: Path) -> None:
    """Leave the first half of ``path``, as a process killed inside the write does."""
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])


def copy_run(source: Path, target: Path, *, without=()) -> Path:
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(*without) if without else None)
    return target


def quiet_snapshot(run_dir: Path):
    """The tree as ``snapshot`` sees it, without the two files a resume that got past the lock writes."""
    return {k: v for k, v in snapshot(run_dir).items() if k not in (LOCK_NAME, "run.log")}


def resume(run_dir: Path, **changes):
    return Scripted(make_config(run_dir, resume_interrupted=True, **{**REQUEST, **changes})).run()


# ---------------------------------------------------------------------
# C3: the best checkpoint is written atomically
# ---------------------------------------------------------------------


def tiny_model():
    model = keras.Sequential([keras.layers.Input((4,)), keras.layers.Dense(3)])
    model.compile(optimizer="adam", loss="mse")
    return model


def tiny_fit(model, callbacks, epochs=2, steps=4):
    x = np.ones((steps * 4, 4), "float32")
    model.fit(x, np.zeros((steps * 4, 3), "float32"), batch_size=4, epochs=epochs,
              callbacks=callbacks, verbose=0)


def test_the_legacy_default_is_the_stock_checkpoint_and_the_option_is_the_atomic_one(tmp_path):
    stock, _ = create_callbacks("m", run_dir=str(tmp_path / "a"), include_analyzer=False)
    atomic, _ = create_callbacks("m", run_dir=str(tmp_path / "b"), include_analyzer=False,
                                 atomic_best_checkpoint=True)
    kinds = lambda callbacks: [c for c in callbacks if isinstance(c, keras.callbacks.ModelCheckpoint)]
    assert [type(c) for c in kinds(stock)] == [keras.callbacks.ModelCheckpoint]
    assert [type(c) for c in kinds(atomic)] == [AtomicModelCheckpoint]


def test_an_improvement_lands_whole_and_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "best.keras"
    tiny_fit(tiny_model(), [AtomicModelCheckpoint(str(path), monitor="loss", save_best_only=True)])
    check_model_file(path, "test")
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["best.keras"]


def test_an_epoch_that_does_not_improve_promotes_nothing_and_a_stale_partial_stays_a_partial(tmp_path):
    path = tmp_path / "best.keras"
    stale = tmp_path / "best.partial.keras"
    stale.write_bytes(b"left by a process killed mid-save")
    # A threshold no loss beats: no epoch improves, so nothing may be saved or renamed.
    tiny_fit(tiny_model(), [AtomicModelCheckpoint(
        str(path), monitor="loss", mode="min", save_best_only=True, initial_value_threshold=-1.0)])
    assert not path.exists()
    assert stale.read_bytes() == b"left by a process killed mid-save"


class RisingMonitor(keras.callbacks.Callback):
    """``val_accuracy`` rises every epoch (a new best each time); ``val_loss`` never moves (a plateau)."""

    def on_epoch_end(self, epoch, logs=None):
        logs["val_accuracy"] = 0.1 * (epoch + 1)
        logs["val_loss"] = 1.0


class Rising(ImageClassificationTrainer):
    """The real task with :class:`RisingMonitor`, optionally killed at the start of epoch ``kill_at``."""

    kill_at = None

    def _callbacks(self, *arguments):
        epoch_line, callbacks = super()._callbacks(*arguments)
        callbacks.insert(1, RisingMonitor())  # after the learning-rate logger, before EarlyStopping
        return epoch_line, callbacks

    def extra_callbacks(self, config, data):
        return [] if self.kill_at is None else [KillAt(self.kill_at)]


RISING = {**REQUEST, "epochs": 5, "checkpoint_args.early_stopping_patience": 50}


def test_a_kill_inside_the_third_best_save_leaves_the_second_best_whole(world, tmp_path, monkeypatch):
    """The stock callback writes the file in place: the earlier best dies with the write."""
    original = keras.Model.save
    calls = []
    seen = {}

    def torn_save(self, filepath, *arguments, **keywords):
        calls.append(str(filepath))
        best = Path(run_dir / "best_model.keras")
        if len(calls) == 3:
            seen["before"] = best.read_bytes() if best.exists() else None
        original(self, filepath, *arguments, **keywords)
        if len(calls) == 3:
            cut_in_half(Path(filepath))
            raise RuntimeError("killed inside the save")

    run_dir = tmp_path / "run"
    monkeypatch.setattr(keras.Model, "save", torn_save)
    with pytest.raises(RuntimeError, match="killed inside the save"):
        Rising(make_config(run_dir, **RISING)).run()
    monkeypatch.undo()
    assert len(calls) == 3 and seen["before"] is not None
    best = run_dir / "best_model.keras"
    check_model_file(best, "the earlier best must be whole")
    assert best.read_bytes() == seen["before"]
    assert [Path(call).name for call in calls] == ["best_model.partial.keras"] * 3


def test_a_torn_best_checkpoint_is_a_config_error_naming_the_file_on_resume(world, killed, tmp_path):
    run_dir = copy_run(killed, tmp_path / "torn")
    cut_in_half(run_dir / "best_model.keras")
    before = snapshot(run_dir)
    with pytest.raises(ConfigError, match=r"Cannot resume .*best_model\.keras is not a readable"):
        resume(run_dir)
    assert snapshot(run_dir) == before  # refused before the CSV was copied aside or a lock taken


def test_a_torn_best_checkpoint_is_a_config_error_naming_the_file_on_predict(world, killed, tmp_path):
    from train.api.predictor import load_model

    run_dir = copy_run(killed, tmp_path / "torn")
    cut_in_half(run_dir / "best_model.keras")
    with pytest.raises(ConfigError, match=r"best_model\.keras is not a readable \.keras archive"):
        load_model(run_dir)


def test_a_zip_that_keras_cannot_load_is_a_config_error_not_a_traceback(tmp_path):
    import zipfile

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with zipfile.ZipFile(run_dir / "best_model.keras", "w") as archive:
        archive.writestr("note.txt", "a zip, but not a Keras archive")
    with pytest.raises(ConfigError, match=r"best_model\.keras cannot be loaded"):
        ImageClassificationTrainer._best_weights_of(run_dir, SimpleNamespace(best=0.5))


def test_check_model_file_names_a_missing_a_cut_and_a_damaged_file(tmp_path):
    good = tmp_path / "good.keras"
    tiny_model().save(good)
    check_model_file(good, "x")
    cut = tmp_path / "cut.keras"
    cut.write_bytes(good.read_bytes()[:100])
    with pytest.raises(ConfigError, match=r"x: .*cut\.keras is not a readable"):
        check_model_file(cut, "x")
    with pytest.raises(ConfigError, match=r"missing\.keras is not a readable"):
        check_model_file(tmp_path / "missing.keras", "x")


# ---------------------------------------------------------------------
# C4, C5: the backup is a pair, and the pair is checked
# ---------------------------------------------------------------------


STEPS = 4


def real_backup(directory: Path, epochs: int = 3, policy: str = "float32") -> Path:
    """The backup directory of a REAL ``BackupAndRestore(double_checkpoint=True)`` after ``epochs`` epochs."""
    previous = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy(policy)
    try:
        tiny_fit(tiny_model(), [keras.callbacks.BackupAndRestore(
            str(directory), double_checkpoint=True, delete_checkpoint=False)], epochs=epochs, steps=STEPS)
    finally:
        keras.mixed_precision.set_global_policy(previous)
    return directory


def names(directory: Path):
    return sorted(entry.name for entry in directory.iterdir())


def test_keras_double_checkpoint_keeps_the_previous_pair_and_recovers_from_a_cut_file_itself(tmp_path):
    """The real Keras path, no harness: cut the latest weights after a real save and fit again."""
    directory = real_backup(tmp_path / "backup")
    assert names(directory) == sorted(
        [BACKUP_WEIGHTS_NAME, BACKUP_WEIGHTS_NAME + PREVIOUS_SUFFIX,
         BACKUP_METADATA_NAME, BACKUP_METADATA_NAME + PREVIOUS_SUFFIX])
    assert (read_backup_epoch(directory), read_backup_epoch(directory, previous=True)) == (3, 2)
    cut_in_half(directory / BACKUP_WEIGHTS_NAME)
    model = tiny_model()
    callback = keras.callbacks.BackupAndRestore(str(directory), double_checkpoint=True, delete_checkpoint=False)
    callback.set_model(model)
    callback.on_train_begin()
    assert model._initial_epoch == 2  # Keras went back to the previous pair on its own


def test_a_sound_latest_pair_is_kept_as_it_is(tmp_path):
    directory = real_backup(tmp_path / "backup")
    before = {name: (directory / name).read_bytes() for name in names(directory)}
    assert settle_backup(directory, STEPS) == ("latest", 3)
    assert {name: (directory / name).read_bytes() for name in names(directory)} == before


def test_a_cut_latest_weights_file_puts_the_previous_pair_in_its_place(tmp_path):
    directory = real_backup(tmp_path / "backup")
    cut_in_half(directory / BACKUP_WEIGHTS_NAME)
    assert settle_backup(directory, STEPS) == ("previous", 2)
    for name in (BACKUP_WEIGHTS_NAME, BACKUP_METADATA_NAME):
        assert (directory / name).read_bytes() == (directory / (name + PREVIOUS_SUFFIX)).read_bytes()
    assert backup_iterations(directory / BACKUP_WEIGHTS_NAME) == (2 * STEPS, False)
    assert not [name for name in names(directory) if name.endswith(".restoring")]


def test_weights_one_epoch_ahead_of_their_metadata_are_caught_by_the_step_count(tmp_path):
    """A kill between the weights write and the metadata write: the file opens, so Keras alone accepts it."""
    directory = real_backup(tmp_path / "backup")
    (directory / BACKUP_METADATA_NAME).write_text(json.dumps({"epoch": 2, "batch": 0}))
    assert backup_iterations(directory / BACKUP_WEIGHTS_NAME) == (3 * STEPS, False)
    assert settle_backup(directory, STEPS) == ("previous", 2)
    assert backup_iterations(directory / BACKUP_WEIGHTS_NAME) == (2 * STEPS, False)


def test_a_torn_latest_metadata_file_takes_the_previous_pair(tmp_path):
    directory = real_backup(tmp_path / "backup")
    (directory / BACKUP_METADATA_NAME).write_text('{"epo')
    assert settle_backup(directory, STEPS) == ("previous", 2)
    assert read_backup_epoch(directory) == 2


def test_two_unusable_copies_raise_naming_both_files_and_change_nothing(tmp_path):
    directory = real_backup(tmp_path / "backup")
    for name in (BACKUP_WEIGHTS_NAME, BACKUP_WEIGHTS_NAME + PREVIOUS_SUFFIX):
        cut_in_half(directory / name)
    before = {name: (directory / name).read_bytes() for name in names(directory)}
    with pytest.raises(ValueError) as caught:
        settle_backup(directory, STEPS)
    text = str(caught.value)
    assert f"{BACKUP_WEIGHTS_NAME}:" in text and f"{BACKUP_WEIGHTS_NAME}{PREVIOUS_SUFFIX}:" in text
    assert {name: (directory / name).read_bytes() for name in names(directory)} == before


def test_ahead_weights_with_no_previous_copy_say_how_many_steps_were_taken(tmp_path):
    directory = real_backup(tmp_path / "backup")
    (directory / BACKUP_METADATA_NAME).write_text(json.dumps({"epoch": 2, "batch": 0}))
    for name in (BACKUP_WEIGHTS_NAME + PREVIOUS_SUFFIX, BACKUP_METADATA_NAME + PREVIOUS_SUFFIX):
        (directory / name).unlink()
    with pytest.raises(ValueError, match=r"took 12 steps, but training_metadata\.json says 2 epochs x 4"):
        settle_backup(directory, STEPS)


def test_a_loss_scaled_optimizer_may_lag_the_batches_but_never_lead_them(tmp_path):
    """``LossScaleOptimizer`` skips a step with non-finite gradients, so its count is ``<=``."""
    directory = real_backup(tmp_path / "scaled", policy="mixed_float16")
    assert backup_iterations(directory / BACKUP_WEIGHTS_NAME) == (3 * STEPS, True)
    assert settle_backup(directory, STEPS) == ("latest", 3)
    (directory / BACKUP_METADATA_NAME).write_text(json.dumps({"epoch": 4, "batch": 0}))
    assert settle_backup(directory, STEPS)[0] == "latest"  # 12 steps, 4 epochs x 4 = 16: skipped steps
    (directory / BACKUP_METADATA_NAME).write_text(json.dumps({"epoch": 2, "batch": 0}))
    assert settle_backup(directory, STEPS) == ("previous", 2)  # 12 steps, 2 x 4 = 8: ahead
    plain = real_backup(tmp_path / "plain")
    (plain / BACKUP_METADATA_NAME).write_text(json.dumps({"epoch": 4, "batch": 0}))
    assert settle_backup(plain, STEPS) == ("previous", 2)  # the same lag is a mismatch without a loss scale


def plant_backup_with(run_dir: Path, latest_epoch=None, previous_epoch=None):
    """Write metadata files by hand, no weights: for the reading of the start epoch."""
    backup = run_dir / "checkpoints" / "backup"
    backup.mkdir(parents=True)
    if latest_epoch is not None:
        (backup / BACKUP_METADATA_NAME).write_text(latest_epoch)
    if previous_epoch is not None:
        (backup / (BACKUP_METADATA_NAME + PREVIOUS_SUFFIX)).write_text(json.dumps({"epoch": previous_epoch}))
    return backup


def test_the_start_epoch_comes_from_the_previous_metadata_when_the_latest_is_torn(tmp_path):
    backup = plant_backup_with(tmp_path, latest_epoch='{"epo', previous_epoch=2)
    csv_path = tmp_path / "training_log.csv"
    csv_path.write_text("epoch,loss\n0,1.0\n1,0.9\n2,0.8\n")  # the epoch after the previous pair is a row too
    sidecar = tmp_path / "checkpoints" / FIT_STATE_NAME
    sidecar.write_text(json.dumps({"epochs": {"0": {"best": 1.0, "wait": 0, "best_epoch": 0},
                                              "1": {"best": 1.0, "wait": 1, "best_epoch": 0}}}))
    point = read_resume_point(csv_path, backup, sidecar)
    assert point.start == 2 and [row[0] for row in point.rows] == [0, 1]


# ---------------------------------------------------------------------
# C4, C5 at the trainer: the files of a real killed run
# ---------------------------------------------------------------------


def test_every_run_backs_up_with_a_previous_copy_and_writes_its_best_atomically(whole):
    _, run_dir = whole
    backup = run_dir / "checkpoints" / "backup"
    assert (backup / (BACKUP_WEIGHTS_NAME + PREVIOUS_SUFFIX)).is_file()
    assert (backup / (BACKUP_METADATA_NAME + PREVIOUS_SUFFIX)).is_file()
    assert not list(run_dir.glob("*.partial.keras"))


def epochs_of(run_dir: Path):
    return [int(row["epoch"]) for row in read_csv(run_dir)]


def test_a_run_whose_latest_weights_are_cut_goes_back_one_epoch_and_the_csv_holds_each_epoch_once(
        world, whole, killed, tmp_path):
    """C4. ``killed`` made two REAL saves: latest after 2 epochs, previous after 1."""
    run_dir = copy_run(killed, tmp_path / "cut")
    assert read_backup_epoch(run_dir / "checkpoints" / "backup") == 2
    cut_in_half(run_dir / "checkpoints" / "backup" / BACKUP_WEIGHTS_NAME)
    resume(run_dir)
    summary = json.loads((run_dir / "results_summary.json").read_text())
    assert summary["resumed_after_epochs"] == 1  # one epoch went back, the other was trained again
    _, whole_dir = whole
    assert epochs_of(run_dir) == epochs_of(whole_dir) == [0, 1, 2, 3]
    assert [row["lr"] for row in read_csv(run_dir)] == [row["lr"] for row in read_csv(whole_dir)]
    assert backup_iterations(run_dir / "checkpoints" / "backup" / BACKUP_WEIGHTS_NAME) == backup_iterations(
        whole_dir / "checkpoints" / "backup" / BACKUP_WEIGHTS_NAME)


def test_two_cut_backup_files_are_refused_naming_both_before_the_csv_is_copied_aside(world, killed, tmp_path):
    run_dir = copy_run(killed, tmp_path / "both")
    backup = run_dir / "checkpoints" / "backup"
    cut_in_half(backup / BACKUP_WEIGHTS_NAME)
    cut_in_half(backup / (BACKUP_WEIGHTS_NAME + PREVIOUS_SUFFIX))
    before = quiet_snapshot(run_dir)
    with pytest.raises(ConfigError) as caught:
        resume(run_dir)
    text = str(caught.value)
    assert "latest.weights.h5:" in text and "latest.weights.h5.bkp:" in text
    assert "choose a new 'out'" in text
    assert not list(run_dir.glob("training_log.pre_resume_*.csv"))
    # Only the lock and the log may have moved (a resume takes the first before the GPU, the second
    # is appended to); the CSV and the backup did not.
    assert quiet_snapshot(run_dir) == before


@pytest.fixture(scope="module")
def ahead(world, tmp_path_factory):
    """The backup of a run killed between the weights write and the metadata write (REAL Keras steps)."""
    run_dir = tmp_path_factory.mktemp("ahead") / "run"
    originals = (keras.callbacks.BackupAndRestore._save_model, keras.Model.save_weights)
    holder = {}

    def save_model(self):
        holder["backup"] = self
        return originals[0](self)

    def save_weights(self, filepath, *arguments, **keywords):
        result = originals[1](self, filepath, *arguments, **keywords)
        backup = holder.get("backup")
        if backup is not None and backup._current_epoch == 3 and str(filepath) == backup._weights_path:
            raise RuntimeError("killed after the weights, before the metadata")
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(keras.callbacks.BackupAndRestore, "_save_model", save_model)
        patch.setattr(keras.Model, "save_weights", save_weights)
        with pytest.raises(RuntimeError, match="before the metadata"):
            Scripted(make_config(run_dir, **REQUEST)).run()
    return run_dir


def test_the_ahead_run_has_weights_that_open_and_a_stale_metadata_file(ahead):
    backup = ahead / "checkpoints" / "backup"
    assert read_backup_epoch(backup) == 2
    latest = backup_iterations(backup / BACKUP_WEIGHTS_NAME)[0]
    previous = backup_iterations(backup / (BACKUP_WEIGHTS_NAME + PREVIOUS_SUFFIX))[0]
    assert (latest, previous) == (3 * (previous // 2), previous)  # three epochs of weights, two of metadata


def test_weights_ahead_of_their_metadata_resume_from_the_previous_pair_with_the_schedule_intact(
        world, whole, ahead, tmp_path):
    """C5. Keras alone accepted this and skipped one epoch of the schedule, exit 0."""
    run_dir = copy_run(ahead, tmp_path / "ahead")
    resume(run_dir)
    summary = json.loads((run_dir / "results_summary.json").read_text())
    _, whole_dir = whole
    assert summary["resumed_after_epochs"] == 2  # the previous pair: epoch 2 of the stale metadata, the weights of 2 epochs
    assert epochs_of(run_dir) == epochs_of(whole_dir)
    assert [row["lr"] for row in read_csv(run_dir)] == [row["lr"] for row in read_csv(whole_dir)]
    assert backup_iterations(run_dir / "checkpoints" / "backup" / BACKUP_WEIGHTS_NAME) == backup_iterations(
        whole_dir / "checkpoints" / "backup" / BACKUP_WEIGHTS_NAME)


def test_ahead_weights_with_no_previous_copy_are_refused_not_resumed_with_a_skipped_schedule(
        world, ahead, tmp_path):
    run_dir = copy_run(ahead, tmp_path / "ahead_alone")
    backup = run_dir / "checkpoints" / "backup"
    for name in (BACKUP_WEIGHTS_NAME + PREVIOUS_SUFFIX, BACKUP_METADATA_NAME + PREVIOUS_SUFFIX):
        (backup / name).unlink()
    with pytest.raises(ConfigError, match=r"steps, but training_metadata\.json says 2 epochs"):
        resume(run_dir)
    assert not list(run_dir.glob("training_log.pre_resume_*.csv"))


# ---------------------------------------------------------------------
# The mechanism order (D-021) and the residual of the best checkpoint
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def before_sidecar(world, tmp_path_factory):
    """A run killed at the end of epoch index 2, after the checkpoint and CSV, before the sidecar and backup."""
    from train.common import resume_state

    run_dir = tmp_path_factory.mktemp("before_sidecar") / "run"
    original = resume_state.FitStateSidecar.on_epoch_end
    saved = {}
    checkpoint_save = keras.callbacks.ModelCheckpoint._save_model

    def remember(self, epoch, batch, logs):
        checkpoint_save(self, epoch, batch, logs)
        saved["checkpoint_best"] = float(self.best)

    def die(self, epoch, logs=None):
        if epoch == 2:
            raise RuntimeError("killed before the sidecar")
        return original(self, epoch, logs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(resume_state.FitStateSidecar, "on_epoch_end", die)
        patch.setattr(keras.callbacks.ModelCheckpoint, "_save_model", remember)
        with pytest.raises(RuntimeError, match="before the sidecar"):
            Rising(make_config(run_dir, **RISING)).run()
    return run_dir, saved["checkpoint_best"]


def test_the_sidecar_is_written_before_the_backup_so_a_kill_between_them_leaves_both_readable(
        before_sidecar):
    """Swap the two callbacks and the backup claims epoch 3 while the sidecar stops at epoch 1."""
    run_dir, _ = before_sidecar
    point = read_resume_point(
        run_dir / "training_log.csv", run_dir / "checkpoints" / "backup",
        run_dir / "checkpoints" / FIT_STATE_NAME)
    assert point.start == 2
    assert sorted(read_fit_state(run_dir / "checkpoints" / FIT_STATE_NAME)["epochs"]) == ["0", "1"]


def test_the_callback_list_puts_the_sidecar_directly_before_the_backup(world, tmp_path, monkeypatch):
    trainer = Rising(make_config(tmp_path / "order", **RISING))
    captured = {}
    original = Rising._callbacks

    def spy(self, *arguments):
        epoch_line, callbacks = original(self, *arguments)
        captured["kinds"] = [type(c).__name__ for c in callbacks]
        raise RuntimeError("stop before fit")

    monkeypatch.setattr(Rising, "_callbacks", spy)
    with pytest.raises(RuntimeError, match="stop before fit"):
        trainer.run()
    kinds = captured["kinds"]
    assert kinds[-2:] == ["FitStateSidecar", "BackupAndRestore"]
    assert kinds.index("AtomicModelCheckpoint") < kinds.index("EpochCSVLogger")


def test_the_best_file_of_a_kill_before_the_backup_is_one_epoch_later_than_the_record(before_sidecar):
    """The residual the help states: the checkpoint was saved for epoch index 2, the record ends at 1."""
    run_dir, checkpoint_best = before_sidecar
    entries = read_fit_state(run_dir / "checkpoints" / FIT_STATE_NAME)["epochs"]
    assert entries["1"]["best"] == pytest.approx(0.2)
    assert checkpoint_best == pytest.approx(0.3)  # the file is the weights of the epoch after the record
    check_model_file(run_dir / "best_model.keras", "whole")


def test_the_resume_help_states_the_checkpoint_window_and_what_comes_back():
    field = {f.name: f for f in dataclasses.fields(ImageClassificationConfig)}["resume_interrupted"]
    text = field.metadata["help"]
    for phrase in (
        "between the best checkpoint's save and the backup",
        "one epoch LATER than the resumed state's best",
        "never an earlier one",
        "ReduceLROnPlateau",
        "devices may differ",
        "other data than the run's",
        "a live process holding the run",
    ):
        assert phrase in text, phrase
    documented = " ".join(sys.modules[train_image_classification.__module__].__doc__.split())
    assert "one epoch LATER" in documented and "never an earlier one" in documented


# ---------------------------------------------------------------------
# ROP: ReduceLROnPlateau reduces at the epochs a clean run reduces at
# ---------------------------------------------------------------------

PLATEAU = {**RISING, "epochs": 12, "optim_args.schedule": "constant"}


class PlateauKilled(Rising):
    kill_at = 8


def test_reduce_lr_on_plateau_reduces_at_the_same_epochs_after_a_resume(world, tmp_path):
    clean = tmp_path / "clean"
    Rising(make_config(clean, **PLATEAU)).run()
    killed_dir = tmp_path / "killed"
    with pytest.raises(RuntimeError, match="killed"):
        PlateauKilled(make_config(killed_dir, **PLATEAU)).run()
    entry = read_fit_state(killed_dir / "checkpoints" / FIT_STATE_NAME)["epochs"]["7"]
    assert entry["reduce_lr"] == {"best": 1.0, "wait": 2, "cooldown_counter": 0}
    Rising(make_config(killed_dir, resume_interrupted=True, **PLATEAU)).run()
    clean_lr = [row["lr"] for row in read_csv(clean)]
    assert len(set(clean_lr)) == 3, clean_lr  # two reductions, so the test can tell
    assert [row["lr"] for row in read_csv(killed_dir)] == clean_lr
    lr_epochs = lambda run: json.loads((run / "results_summary.json").read_text())["lr_reduction_epochs"]
    assert lr_epochs(killed_dir) == lr_epochs(clean) and len(lr_epochs(clean)) == 2


# ---------------------------------------------------------------------
# C6: the data of the run
# ---------------------------------------------------------------------

COLOURS = {"apple": (200, 40, 40), "mango": (230, 170, 30), "zebra": (60, 60, 200)}


def make_tree(root: Path) -> SimpleNamespace:
    for split, count in (("train", 12), ("test", 3)):
        for name, base in COLOURS.items():
            (root / split / name).mkdir(parents=True)
            for index in range(count):
                shade = tuple(min(255, max(0, channel + 7 * index - 25)) for channel in base)
                Image.new("RGB", (20, 20), shade).save(root / split / name / f"{index:02d}.png")
    return SimpleNamespace(root=root, train=root / "train", test=root / "test")


def folder_config(out, tree, **changes):
    return make_config(out, **{
        "data": {"format": "folder", "train_dir": str(tree.train), "test_dir": str(tree.test)},
        "transform_args": {"image_size": 16}, "epochs": 4,
        "metric_args.success_threshold": 0.0, **changes})


class FolderKilled(ImageClassificationTrainer):
    def extra_callbacks(self, config, data):
        return [KillAt(2)]


@pytest.fixture(scope="module")
def folder_run(world, tmp_path_factory):
    root = tmp_path_factory.mktemp("folder")
    tree = make_tree(root / "data")
    run_dir = root / "run"
    with pytest.raises(RuntimeError, match="killed"):
        FolderKilled(folder_config(run_dir, tree)).run()
    return tree, run_dir


def rename(*pairs):
    for source, target in pairs:
        os.rename(source, target)


def resume_folder(tree, run_dir):
    return ImageClassificationTrainer(folder_config(run_dir, tree, resume_interrupted=True)).run()


def refuse_folder(tree, folder_run_dir, tmp_path):
    run_dir = copy_run(folder_run_dir, tmp_path / "copy")
    before = quiet_snapshot(run_dir)
    with pytest.raises(ConfigError) as caught:
        resume_folder(tree, run_dir)
    assert quiet_snapshot(run_dir) == before, "the refusal must come before the CSV or the backup is touched"
    return str(caught.value)


def test_a_renamed_class_is_refused_naming_the_class_names(folder_run, tmp_path):
    tree, run_dir = folder_run
    swap = [(tree.train / "zebra", tree.train / "aardvark"), (tree.test / "zebra", tree.test / "aardvark")]
    rename(*swap)
    try:
        text = refuse_folder(tree, run_dir, tmp_path)
    finally:
        rename(*[(target, source) for source, target in swap])
    assert "the data is not the data the run started with" in text
    assert "class_names: " in text and "aardvark" in text


def test_swapped_class_folders_are_refused_naming_the_statistics(folder_run, tmp_path):
    tree, run_dir = folder_run
    there = [(tree.train / "apple", tree.train / "tmp"), (tree.train / "zebra", tree.train / "apple"),
             (tree.train / "tmp", tree.train / "zebra"),
             (tree.test / "apple", tree.test / "tmp"), (tree.test / "zebra", tree.test / "apple"),
             (tree.test / "tmp", tree.test / "zebra")]
    back = [(tree.train / "apple", tree.train / "tmp"), (tree.train / "zebra", tree.train / "apple"),
            (tree.train / "tmp", tree.train / "zebra"),
            (tree.test / "apple", tree.test / "tmp"), (tree.test / "zebra", tree.test / "apple"),
            (tree.test / "tmp", tree.test / "zebra")]
    rename(*there)
    try:
        text = refuse_folder(tree, run_dir, tmp_path)
    finally:
        rename(*back)
    assert "the data is not the data the run started with" in text
    assert "class_names" not in text  # the names are the same; the images behind them are not
    assert "mean: " in text or "std: " in text


def test_added_images_are_refused_naming_the_sizes_and_the_statistics(folder_run, tmp_path):
    tree, run_dir = folder_run
    added = [tree.train / "apple" / f"extra{i}.png" for i in range(9)]
    for path in added:
        Image.new("RGB", (20, 20), (30, 200, 30)).save(path)
    try:
        text = refuse_folder(tree, run_dir, tmp_path)
    finally:
        for path in added:
            path.unlink()
    assert "n_train: " in text and "mean: " in text


def test_unchanged_data_resumes_and_a_resume_leaves_task_json_alone(folder_run, tmp_path):
    from train.api.predictor import read_record

    tree, folder_dir = folder_run
    run_dir = copy_run(folder_dir, tmp_path / "same")
    before = (run_dir / "task.json").read_bytes()
    resume_folder(tree, run_dir)
    assert (run_dir / "task.json").read_bytes() == before
    assert read_record(run_dir / "task.json")["class_names"] == ["apple", "mango", "zebra"]
    assert json.loads((run_dir / "results_summary.json").read_text())["resumed_after_epochs"] == 2


def test_the_record_comparison_tolerates_float_noise_and_names_every_difference():
    from train.api.predictor import record_differences

    base = {"class_names": ["a", "b"], "input_shape": [16, 16, 3], "normalize": "standardize",
            "eval_transform": {"kind": "resize", "source_channels": 3},
            "mean": [0.5, 0.4, 0.3], "std": [0.2, 0.2, 0.2]}
    assert record_differences(base, {**base, "mean": [0.5 + 1e-9, 0.4, 0.3]}) == []
    lines = record_differences(base, {**base, "class_names": ["a", "c"], "std": [0.2, 0.2, 0.25]})
    assert [line.split(":")[0] for line in lines] == ["class_names", "std"]
    assert '["a", "b"] -> ["a", "c"]' in lines[0]


# ---------------------------------------------------------------------
# `devices` is free at resume
# ---------------------------------------------------------------------


def test_devices_is_not_compared_at_resume_but_every_other_key_still_is():
    assert "devices" in RESUME_UNCOMPARED_KEYS
    earlier = {"out": "a", "devices": 0, "epochs": 3, "optim_args": {"lr": 0.1}}
    assert request_differences(earlier, {"out": "a", "devices": None, "epochs": 3,
                                         "optim_args": {"lr": 0.1}}) == []
    assert request_differences(earlier, {"out": "a", "devices": 1, "epochs": 4,
                                         "optim_args": {"lr": 0.1}}) == ["epochs: 3 -> 4"]


# ---------------------------------------------------------------------
# C7: the run lock, with two real processes
# ---------------------------------------------------------------------

HOLDER = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    from train.api.run_dir import take_run_lock

    handle = take_run_lock(Path(sys.argv[1]), resume=sys.argv[2] == "resume")  # the handle IS the lock
    print("held", flush=True)
    sys.stdin.read()
    """
)


def start_holder(directory: Path, mode: str = "fresh") -> subprocess.Popen:
    process = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(directory), mode],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert process.stdout.readline().strip() == "held", process.stderr.read()
    return process


def release(process: subprocess.Popen) -> None:
    process.stdin.close()
    assert process.wait(timeout=60) == 0
    process.stdout.close()
    process.stderr.close()


def test_a_fresh_lock_is_made_with_o_excl_and_names_this_process(tmp_path):
    with take_run_lock(tmp_path, resume=False):
        record = read_lock(tmp_path)
        assert record["pid"] == os.getpid() and record["start"] == process_stamp(os.getpid())
        with pytest.raises(ConfigError, match="claimed by another run just now"):
            take_run_lock(tmp_path, resume=False)


def test_a_resume_refuses_a_live_holder_and_replaces_the_lock_of_a_dead_one(tmp_path):
    holder = start_holder(tmp_path)
    try:
        record = read_lock(tmp_path)
        assert record["pid"] == holder.pid != os.getpid()
        before = (tmp_path / LOCK_NAME).read_bytes()
        with pytest.raises(ConfigError, match=rf"process {holder.pid} holds run\.lock and is still running"):
            refuse_live_holder(tmp_path)
        with pytest.raises(ConfigError, match=str(holder.pid)):
            take_run_lock(tmp_path, resume=True)
        assert (tmp_path / LOCK_NAME).read_bytes() == before  # refused: not touched
    finally:
        release(holder)
    # The holder exited: its lock is still there (nothing deletes it) and reads as stale.
    assert (tmp_path / LOCK_NAME).read_bytes() == before
    refuse_live_holder(tmp_path)  # nobody holds it: the file is just a label now
    with take_run_lock(tmp_path, resume=True):
        assert read_lock(tmp_path)["pid"] == os.getpid()
    assert names(tmp_path) == [LOCK_NAME]  # taken over in place: no temporary file, nothing deleted


def test_a_garbled_or_empty_lock_is_stale_and_this_process_is_never_its_own_holder(tmp_path):
    (tmp_path / LOCK_NAME).write_text('{"pid"')
    assert read_lock(tmp_path) == {}
    refuse_live_holder(tmp_path)
    with take_run_lock(tmp_path, resume=True):
        assert read_lock(tmp_path)["pid"] == os.getpid()


def test_a_resume_of_a_run_another_process_holds_is_refused_before_anything_is_written(world, killed, tmp_path):
    run_dir = copy_run(killed, tmp_path / "held", without=(LOCK_NAME,))
    holder = start_holder(run_dir)
    try:
        before = snapshot(run_dir)
        with pytest.raises(ConfigError, match=rf"process {holder.pid} holds run\.lock"):
            resume(run_dir)
        assert snapshot(run_dir) == before
    finally:
        release(holder)


def test_a_finished_run_keeps_its_lock(whole):
    _, run_dir = whole
    record = read_lock(run_dir)
    assert record["pid"] == os.getpid() and record["start"] == process_stamp(os.getpid())


# ---------------------------------------------------------------------
# C18: two refusals say what to do
# ---------------------------------------------------------------------


class KilledFirst(Scripted):
    kill_at = 0


def test_a_run_killed_in_its_first_epoch_says_nothing_can_be_resumed_and_to_choose_a_new_out(world, tmp_path):
    run_dir = tmp_path / "first"
    with pytest.raises(RuntimeError, match="killed"):
        KilledFirst(make_config(run_dir, **REQUEST)).run()
    assert not (run_dir / "checkpoints" / "backup").exists()
    before = quiet_snapshot(run_dir)
    with pytest.raises(ConfigError) as caught:
        resume(run_dir)
    text = str(caught.value)
    assert "killed before its first epoch ended" in text and "nothing was saved that a resume could continue from" in text
    assert "Choose a new 'out'" in text
    assert quiet_snapshot(run_dir) == before
    # A backup directory with no metadata in it (killed during the first save) says the same.
    (run_dir / "checkpoints" / "backup").mkdir()
    with pytest.raises(ConfigError, match="killed before its first epoch ended"):
        resume(run_dir)


def test_a_run_killed_just_after_a_resume_names_the_copy_that_holds_its_rows(world, killed, tmp_path):
    run_dir = copy_run(killed, tmp_path / "after_resume")
    kept = copy_log_aside(run_dir / "training_log.csv")
    assert kept.name == "training_log.pre_resume_1.csv"
    # What the stock CSV logger does when the resumed fit starts: it opens the file for writing.
    header = (run_dir / "training_log.csv").read_text().splitlines()[0]
    (run_dir / "training_log.csv").write_text(header + "\n")
    before = quiet_snapshot(run_dir)
    with pytest.raises(ConfigError) as caught:
        resume(run_dir)
    text = str(caught.value)
    assert "holds 0 complete epoch rows, the backup needs 2" in text
    assert "training_log.pre_resume_1.csv holds epochs 0..1" in text
    assert "copy it over training_log.csv and resume again" in text
    assert quiet_snapshot(run_dir) == before
    # Following the advice makes the files agree again.
    shutil.copyfile(kept, run_dir / "training_log.csv")
    point = read_resume_point(
        run_dir / "training_log.csv", run_dir / "checkpoints" / "backup",
        run_dir / "checkpoints" / FIT_STATE_NAME)
    assert point.start == 2


def test_the_hint_stays_quiet_when_no_copy_exists_and_says_so_when_none_is_complete(tmp_path):
    backup = plant_backup_with(tmp_path, latest_epoch=json.dumps({"epoch": 2}))
    sidecar = tmp_path / "checkpoints" / FIT_STATE_NAME
    sidecar.write_text(json.dumps({"epochs": {"1": {"best": 1.0, "wait": 0, "best_epoch": 0}}}))
    log = tmp_path / "training_log.csv"
    log.write_text("epoch,loss\n")
    with pytest.raises(ValueError) as quiet:
        read_resume_point(log, backup, sidecar)
    assert str(quiet.value).endswith("the backup needs 2")
    (tmp_path / "training_log.pre_resume_1.csv").write_text("epoch,loss\n0,1.0\n")
    with pytest.raises(ValueError, match=r"the copies training_log\.pre_resume_1\.csv do not hold them either"):
        read_resume_point(log, backup, sidecar)
