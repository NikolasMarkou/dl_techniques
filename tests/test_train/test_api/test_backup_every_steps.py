"""``checkpoint_args.backup_every_steps``: a killed epoch loses at most N batches of weights, coherently.

Step 19 of plan-2026-10-31T180033-2faf890a, decision D-038. Stock Keras 3.8
``BackupAndRestore(save_freq=N)`` cannot be used as it is (all MEASURED, the first group below
re-measures each on a tiny model so an upgrade that changes one fails here):

* it does not save at an epoch end;
* a fresh process keeps ``_current_epoch`` at 0, so the first mid-epoch save after a resume
  records epoch 0 over the weights of epoch k, and a second kill resumes from epoch 0;
* the restored optimizer keeps the step count of the killed batch, so every later epoch runs
  with the schedule ahead and the run ends on more than ``epochs * steps_per_epoch`` steps.

``train.common.resume_state.StepBackupAndRestore`` fixes those three: (a) it saves at the epoch
end, counting batches inside the epoch; (b) it sets ``_current_epoch`` from the restored
metadata; (c) it puts the optimizer's step count back by the batches the copy had taken. What
a resume then is: the killed epoch restarts from batch 0 on the weights of the last copy.

Four groups:

1. the Keras facts the subclass relies on (in process, a tiny model);
2. the subclass on that model: which batches save, the metadata, ``steps_per_execution`` above
   1, a loss-scaling optimizer, the argument checks, ``settle_backup`` on both copy forms and on
   a weights file newer than its metadata, ``read_resume_point`` inside the first epoch;
3. the configuration: the key, its refusals, the trainer's choice of callback;
4. REAL ``SIGKILL`` of a child process (``_backup_child.py``: the API fit of ``vit/vit_pico``
   on fake CIFAR-10, 18 batches an epoch, a copy every 5 batches) at real batches: inside the
   first epoch, inside a later one, between ``save_weights`` and the metadata write of a
   mid-epoch save and of an epoch-end save, just after a resume, and a boundary kill for the
   comparison. The CSV rows are unique, the epoch count is the asked one, the optimizer ends on
   ``epochs * steps_per_epoch`` steps, the learning rate of every epoch equals the unkilled
   run's, and an unkilled run with the key set equals one without it, weight for weight.

What the difference from an unkilled run is (the design predicts one, and the test requires it
to be small and non-zero): the killed epoch is trained again from its first batch on the weights
of batch k, in the data order epoch 0 had (a resumed run's data iterator is new), with Adam's
moments from batch k. So a resumed run is a valid run of the same recipe, with the same step
count and the same learning rate at every epoch, and NOT the unkilled run's trajectory; the
final weights differ by about 1% of their norm in this setup, against about 70% for another seed.

NOT checked: a kill during the weights write itself (``settle_backup`` has its own tests, a torn
file), a power loss (no ``fsync`` of the copies), a GPU, XLA, and an ImageNet-sized model.
"""

import csv
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import h5py
import keras
import numpy as np
import pytest

import train.api.recipes as recipes
from train.api import registry
from train.api.args import CheckpointArgs
from train.api.errors import ConfigError, ConfigValidationError
from train.api.tasks import register_builtin_tasks
from train.api.tasks.image_classification import (
    ImageClassificationConfig,
    ImageClassificationTrainer,
)
from train.common.resume_state import (
    BACKUP_METADATA_NAME,
    BACKUP_WEIGHTS_NAME,
    PREVIOUS_SUFFIX,
    StepBackupAndRestore,
    backup_iterations,
    read_backup_position,
    read_resume_point,
    settle_backup,
)

from tests.classification_fakes import use_tiny_raw

from .test_task_trainer import make_config

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CHILD = HERE / "_backup_child.py"

# ---------------------------------------------------------------------
# A tiny model, and a fit that a callback can stop
# ---------------------------------------------------------------------

S = 10  # batches per epoch of the tiny model
X = np.random.RandomState(0).randn(S * 8, 4).astype("float32")
Y = np.random.RandomState(1).randn(S * 8, 1).astype("float32")


def tiny(steps_per_execution=1):
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((4,)), keras.layers.Dense(3), keras.layers.Dense(1)])
    schedule = keras.optimizers.schedules.CosineDecay(0.1, 50)
    model.compile(keras.optimizers.Adam(schedule), "mse", steps_per_execution=steps_per_execution)
    return model


class Killed(Exception):
    """Raised by :class:`KillAt`: stands in for a SIGKILL between two batches."""


class KillAt(keras.callbacks.Callback):
    """Raise before batch ``done`` of ``epoch`` (``done`` batches of it have finished)."""

    def __init__(self, epoch, done):
        super().__init__()
        self.epoch, self.done, self.at = epoch, done, 0

    def on_epoch_begin(self, epoch, logs=None):
        self.at = epoch

    def on_train_batch_begin(self, batch, logs=None):
        if self.at == self.epoch and batch == self.done:
            raise Killed(f"killed after {self.done} batches of epoch {self.epoch}")


class Watch(keras.callbacks.Callback):
    """Record the optimizer's step count and the backup's epoch at each epoch start."""

    def __init__(self, backup):
        super().__init__()
        self.backup, self.starts = backup, []

    def on_epoch_begin(self, epoch, logs=None):
        optimizer = self.model.optimizer
        inner = getattr(optimizer, "inner_optimizer", optimizer)
        self.starts.append(
            (epoch, int(keras.ops.convert_to_numpy(inner.iterations)),
             int(keras.ops.convert_to_numpy(optimizer.iterations)), self.backup._current_epoch)
        )


def fit(model, callbacks, epochs=3):
    """Run a fit; return ``True`` when it was killed."""
    try:
        model.fit(X, Y, batch_size=8, epochs=epochs, shuffle=False, verbose=0, callbacks=callbacks)
    except Killed:
        return True
    return False


def metadata(directory):
    return json.loads((Path(directory) / BACKUP_METADATA_NAME).read_text())


def weights_steps(directory):
    return backup_iterations(Path(directory) / BACKUP_WEIGHTS_NAME)[0]


class Recording(StepBackupAndRestore):
    """The subclass, remembering ``(epoch, batch)`` of the metadata after each save."""

    def __init__(self, *arguments, **keywords):
        super().__init__(*arguments, **keywords)
        self.saved = []

    def _save_model(self):
        super()._save_model()
        written = metadata(self.backup_dir)
        self.saved.append((written["epoch"], written["batch"]))


def stock(directory, every=4):
    return keras.callbacks.BackupAndRestore(
        str(directory), save_freq=every, double_checkpoint=True, delete_checkpoint=False)


# ---------------------------------------------------------------------
# 1. What stock Keras does (the measurements the subclass is built on)
# ---------------------------------------------------------------------


def test_the_private_keras_names_the_subclass_relies_on_exist():
    import inspect

    callback = keras.callbacks.BackupAndRestore("unused", save_freq=3)
    for name in ("_current_epoch", "_last_batch_seen", "_batches_seen_since_last_saving",
                 "_training_metadata_path", "_weights_path"):
        assert hasattr(callback, name), f"keras renamed BackupAndRestore.{name}"
    assert list(inspect.signature(keras.callbacks.BackupAndRestore._save_model).parameters) == ["self"]
    assert list(inspect.signature(keras.callbacks.BackupAndRestore._should_save_on_batch).parameters) == [
        "self", "batch"]
    assert keras.__version__.split(".")[0] == "3", "the subclass was measured on Keras 3.x"
    assert StepBackupAndRestore.__mro__[1] is keras.callbacks.BackupAndRestore


def test_stock_save_freq_writes_no_copy_at_the_epoch_end(tmp_path):
    model = tiny()
    assert not fit(model, [stock(tmp_path / "b", every=4)], epochs=1)
    # Saves after batches 4 and 8 only; the epoch ended with 2 more batches and no copy.
    assert metadata(tmp_path / "b") == {"epoch": 0, "batch": 7}
    assert weights_steps(tmp_path / "b") == 8


def test_stock_counts_batches_across_epochs_so_a_copy_can_land_anywhere_in_the_next_epoch(tmp_path):
    seen = []

    class Recording(keras.callbacks.BackupAndRestore):
        def _save_model(self):
            super()._save_model()
            written = metadata(self._weights_path.rsplit("/", 1)[0])
            seen.append((written["epoch"], written["batch"]))

    assert not fit(tiny(), [Recording(str(tmp_path / "b"), save_freq=4)], epochs=2)
    # Epoch 0: after batches 4 and 8. The 2 left over run into epoch 1: its copies are after
    # ITS batches 2, 6 and 10 (the last one), not 4 and 8, and none is written by the epoch end.
    assert seen == [(0, 3), (0, 7), (1, 1), (1, 5), (1, 9)]


def test_stock_reports_the_first_batch_of_a_group_when_steps_per_execution_is_above_one():
    seen = []

    class Watching(keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            seen.append((batch, int(keras.ops.convert_to_numpy(self.model.optimizer.iterations))))

    assert not fit(tiny(4), [Watching()], epochs=1)
    # Groups of 4 start at batches 0, 4, 8; the step count is already past the group. So stock
    # metadata `batch` is not `steps - 1`, which the subclass computes from `steps_per_execution`.
    assert seen == [(0, 4), (4, 8), (8, 10)]


def test_stock_records_epoch_zero_after_a_resume_over_the_weights_of_a_later_epoch(tmp_path):
    directory = tmp_path / "b"
    assert fit(tiny(), [KillAt(1, 6), stock(directory)])
    assert metadata(directory) == {"epoch": 1, "batch": 5}, "the first segment names epoch 1 correctly"
    assert fit(tiny(), [KillAt(1, 6), stock(directory)])
    # The defect: the second segment saved during epoch 1 and wrote epoch 0 (a fresh
    # process keeps _current_epoch at 0) over weights that took 16 + 4 steps.
    assert metadata(directory) == {"epoch": 0, "batch": 3}
    assert weights_steps(directory) == S + 6 + 4


def test_stock_leaves_the_step_count_advanced_after_a_resume(tmp_path):
    directory = tmp_path / "b"
    assert fit(tiny(), [KillAt(1, 6), stock(directory)])
    steps_in_copy = weights_steps(directory)
    assert steps_in_copy == S + 6  # stock's count runs on across epochs: its last copy is after batch 6 of epoch 1
    backup = stock(directory)
    watch = Watch(backup)
    assert not fit(tiny(), [backup, watch], epochs=2)
    (epoch, inner, _outer, _cur), = watch.starts
    assert epoch == 1 and inner == steps_in_copy, "stock does not rewind: epoch 1 restarts at step 16, not 10"


# ---------------------------------------------------------------------
# 2. The subclass
# ---------------------------------------------------------------------


@pytest.mark.parametrize("every, expected", [
    (4, [(0, 3), (0, 7), (1, 0), (1, 3), (1, 7), (2, 0)]),
    # The last batch of an epoch is never a mid-epoch save: the epoch-end copy covers it.
    (5, [(0, 4), (1, 0), (1, 4), (2, 0)]),
    # The count restarts with every epoch: stock's runs on (3 left over would save at batch 0).
    (3, [(0, 2), (0, 5), (0, 8), (1, 0), (1, 2), (1, 5), (1, 8), (2, 0)]),
])
def test_it_saves_after_every_nth_batch_of_each_epoch_and_at_each_epoch_end(tmp_path, every, expected):
    backup = Recording(tmp_path / "b", every, S)
    assert not fit(tiny(), [backup], epochs=2)
    assert backup.saved == expected
    # The epoch-end copies are exactly the ones with batch 0: no mid-epoch copy can be mistaken for one.
    assert [pair for pair in backup.saved if pair[1] == 0] == [(1, 0), (2, 0)]
    assert weights_steps(tmp_path / "b") == 2 * S and metadata(tmp_path / "b") == {"epoch": 2, "batch": 0}


def test_a_copy_inside_an_epoch_has_the_steps_its_metadata_implies(tmp_path):
    directory = tmp_path / "b"
    assert fit(tiny(), [KillAt(1, 9), StepBackupAndRestore(directory, 4, S)])
    assert metadata(directory) == {"epoch": 1, "batch": 7}
    assert weights_steps(directory) == S + 7 + 1
    assert settle_backup(directory, S) == ("latest", 1)


def test_the_resumed_segment_names_its_epoch_and_restarts_the_epoch_on_the_nominal_step(tmp_path):
    directory = tmp_path / "b"
    assert fit(tiny(), [KillAt(1, 6), StepBackupAndRestore(directory, 4, S)])
    assert metadata(directory) == {"epoch": 1, "batch": 3}
    backup = Recording(directory, 4, S)
    watch = Watch(backup)
    model = tiny()
    assert not fit(model, [backup, watch])
    epoch, inner, _outer, current = watch.starts[0]
    assert (epoch, current) == (1, 1), "(b): the resumed segment must know it is in epoch 1"
    assert inner == S, "(c): epoch 1 restarts on step 10, the copy's 14 less its 4 batches"
    assert int(keras.ops.convert_to_numpy(model.optimizer.iterations)) == 3 * S
    # Its first mid-epoch save records epoch 1, not 0 (the stock defect).
    assert backup.saved[0] == (1, 3) and all(epoch >= 1 for epoch, _ in backup.saved)


def test_a_second_kill_after_the_resume_leaves_a_pair_that_still_resumes_into_the_same_epoch(tmp_path):
    directory = tmp_path / "b"
    assert fit(tiny(), [KillAt(1, 6), StepBackupAndRestore(directory, 4, S)])
    # Killed again two batches after the resume, before it saved anything: the pair is as it was.
    assert fit(tiny(), [KillAt(1, 2), StepBackupAndRestore(directory, 4, S)])
    assert metadata(directory) == {"epoch": 1, "batch": 3} and settle_backup(directory, S) == ("latest", 1)
    # Killed once more after its first new save: the new pair names epoch 1, not 0.
    assert fit(tiny(), [KillAt(1, 6), StepBackupAndRestore(directory, 4, S)])
    assert metadata(directory) == {"epoch": 1, "batch": 3}
    assert fit(tiny(), [KillAt(1, 9), StepBackupAndRestore(directory, 4, S)])
    assert metadata(directory) == {"epoch": 1, "batch": 7}
    assert settle_backup(directory, S) == ("latest", 1)
    final = tiny()
    assert not fit(final, [StepBackupAndRestore(directory, 4, S)])
    assert int(keras.ops.convert_to_numpy(final.optimizer.iterations)) == 3 * S


def test_plateau_and_early_stopping_state_come_back_across_a_mid_epoch_resume(tmp_path):
    """The sidecar holds finished epochs only, and a mid-epoch copy holds the rate the last epoch end left."""
    from train.common.resume_state import FitStateSidecar, RestoreFitState

    directory, sidecar = tmp_path / "b", tmp_path / "fit_state.json"

    def plain():
        keras.utils.set_random_seed(0)
        model = keras.Sequential([keras.Input((4,)), keras.layers.Dense(3), keras.layers.Dense(1)])
        model.compile(keras.optimizers.Adam(0.1), "mse")
        return model

    def parts(resume=None):
        stopper = keras.callbacks.EarlyStopping(monitor="loss", patience=20)
        # A huge min_delta: the first epoch improves on inf, no later one does, so the rate is
        # halved at the end of every epoch from the second on, whatever the losses are.
        plateau = keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=1, min_delta=1e9)
        callbacks = [stopper, plateau]
        if resume is not None:
            callbacks.append(RestoreFitState(
                stopper, resume.best, resume.wait, None, resume.best_epoch,
                reduce_lr=plateau, reduce_lr_state=resume.reduce_lr))
        callbacks += [FitStateSidecar(stopper, sidecar, reduce_lr=plateau), StepBackupAndRestore(directory, 4, S)]
        return stopper, plateau, callbacks

    class Rate(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.starts = []

        def on_epoch_begin(self, epoch, logs=None):
            self.starts.append((epoch, round(float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)), 6)))

    class Begin(keras.callbacks.Callback):
        """What the stopper and the plateau hold once every on_train_begin has run (last in the list)."""

        def __init__(self, stopper, plateau):
            super().__init__()
            self.stopper, self.plateau, self.held = stopper, plateau, None

        def on_train_begin(self, logs=None):
            self.held = (self.stopper.best, self.stopper.wait, self.plateau.best, self.plateau.wait)

    def run(kill=None, resume=None, epochs=4):
        stopper, plateau, callbacks = parts(resume)
        rate, begin = Rate(), Begin(stopper, plateau)
        killed = fit(plain(), [rate, *callbacks, begin] + ([KillAt(*kill)] if kill else []), epochs=epochs)
        return begin, rate, killed

    _, whole_rate, _ = run()
    assert whole_rate.starts == [(0, 0.1), (1, 0.1), (2, 0.05), (3, 0.025)]
    shutil.rmtree(directory)
    sidecar.unlink()

    _, _, killed = run(kill=(2, 6))
    assert killed and metadata(directory) == {"epoch": 2, "batch": 3}
    entry = json.loads(sidecar.read_text())["epochs"]["1"]
    assert entry["reduce_lr"]["wait"] == 0 and entry["reduce_lr"]["best"] is not None
    (tmp_path / "log.csv").write_text("epoch,loss\n0,1.0\n1,1.0\n")
    point = read_resume_point(tmp_path / "log.csv", directory, sidecar)
    assert point.start == 2 and point.reduce_lr == entry["reduce_lr"]

    begin, rate, killed = run(resume=point)
    assert not killed
    # Early stopping and the plateau start where epoch 1 left them (not at inf, or no halving at epoch 2's end).
    assert begin.held == (entry["best"], entry["wait"], entry["reduce_lr"]["best"], entry["reduce_lr"]["wait"])
    assert rate.starts == [(2, 0.05), (3, 0.025)], "the rate of the last epoch end, then the next halving on schedule"
    assert json.loads(sidecar.read_text())["epochs"]["3"]["reduce_lr"]["wait"] == 0


def test_a_resume_from_an_epoch_end_copy_changes_nothing(tmp_path):
    directory = tmp_path / "b"
    assert fit(tiny(), [KillAt(2, 0), StepBackupAndRestore(directory, 4, S)])
    assert metadata(directory) == {"epoch": 2, "batch": 0}
    backup = StepBackupAndRestore(directory, 4, S)
    watch = Watch(backup)
    assert not fit(tiny(), [backup, watch])
    assert watch.starts == [(2, 2 * S, 2 * S, 2)]


def test_steps_per_execution_above_one_saves_at_group_boundaries_and_records_the_last_batch(tmp_path):
    directory = tmp_path / "b"
    backup = Recording(directory, 4, S)
    assert fit(tiny(4), [KillAt(1, 8), backup])
    # Groups of 4 start at batches 0, 4, 8: copies after 4 and 8 batches, none after the last (10).
    assert backup.saved == [(0, 3), (0, 7), (1, 0), (1, 3), (1, 7)]
    assert metadata(directory) == {"epoch": 1, "batch": 7}
    assert weights_steps(directory) == S + 8
    assert settle_backup(directory, S) == ("latest", 1)
    final = tiny(4)
    assert not fit(final, [StepBackupAndRestore(directory, 4, S)])
    assert int(keras.ops.convert_to_numpy(final.optimizer.iterations)) == 3 * S


def test_a_loss_scaling_optimizer_is_rewound_on_its_inner_counter_and_never_below_zero(tmp_path):
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        directory = tmp_path / "b"
        assert fit(tiny(), [KillAt(1, 6), StepBackupAndRestore(directory, 2, S)])
        taken, scaled = backup_iterations(directory / BACKUP_WEIGHTS_NAME)
        assert scaled, "the vacuity guard: mixed_float16 compiles a LossScaleOptimizer"
        # The inner counter is the one that moved; it may lag the batches (steps with
        # non-finite gradients are skipped), and settle_backup accepts a lag, not a surplus.
        assert settle_backup(directory, S) == ("latest", 1)
        batch = metadata(directory)["batch"]
        backup = StepBackupAndRestore(directory, 2, S)
        watch = Watch(backup)
        assert fit(tiny(), [backup, watch, KillAt(1, 1)])
        (epoch, inner, outer, current), = watch.starts
        assert (epoch, current, outer) == (1, 1, 0)
        assert inner == max(0, taken - (batch + 1)) and inner <= S

        # Killed inside epoch 0 after one copy: the rewind cannot go below zero.
        first = tmp_path / "first"
        assert fit(tiny(), [KillAt(0, 5), StepBackupAndRestore(first, 4, S)])
        backup = StepBackupAndRestore(first, 4, S)
        watch = Watch(backup)
        assert fit(tiny(), [backup, watch, KillAt(0, 1)])
        assert watch.starts[0][1] == 0 and watch.starts[0][2] == 0
    finally:
        keras.mixed_precision.set_global_policy("float32")


@pytest.mark.parametrize("every, steps, message", [
    (1, S, ">= 2"), (0, S, ">= 2"), (-3, S, ">= 2"),
    (S, S, "< steps_per_epoch"), (S + 5, S, "< steps_per_epoch"),
    (4, None, "must be an int"), (4, True, "must be an int"), (True, S, "must be an int"),
    (4.0, S, "must be an int"), (4, -2, "positive count"), (4, 0, "positive count"),
])
def test_arguments_that_cannot_work_are_refused_at_construction(tmp_path, every, steps, message):
    with pytest.raises(ValueError, match=message):
        StepBackupAndRestore(tmp_path / "b", every, steps)
    assert not (tmp_path / "b").exists()


def test_an_unknown_length_is_refused_where_the_callback_is_built():
    import tensorflow as tf

    with pytest.raises(ValueError, match="positive count"):
        StepBackupAndRestore("unused", 4, tf.data.UNKNOWN_CARDINALITY)


def test_an_imagenet_epoch_of_9930_steps_saves_19_times_inside_the_epoch(tmp_path):
    backup = StepBackupAndRestore(tmp_path / "b", 500, 9930)
    backup.set_model(SimpleNamespace(steps_per_execution=1))
    saves = [batch for batch in range(9930) if backup._should_save_on_batch(batch)]
    assert saves == list(range(499, 9500, 500)) and len(saves) == 19
    assert StepBackupAndRestore(tmp_path / "c", 9929, 9930).save_every_steps == 9929
    with pytest.raises(ValueError, match="< steps_per_epoch"):
        StepBackupAndRestore(tmp_path / "d", 9930, 9930)


def _two_copies(tmp_path):
    """A backup directory after a kill with two copies in epoch 1: latest at batch 8, previous at batch 4."""
    directory = tmp_path / "b"
    assert fit(tiny(), [KillAt(1, 9), StepBackupAndRestore(directory, 4, S)])
    assert metadata(directory) == {"epoch": 1, "batch": 7}
    assert json.loads((directory / (BACKUP_METADATA_NAME + PREVIOUS_SUFFIX)).read_text()) == {"epoch": 1, "batch": 3}
    return directory


def test_an_epoch_end_copy_is_accepted_with_its_epoch_times_the_steps_and_no_more(tmp_path):
    directory = tmp_path / "b"
    assert not fit(tiny(), [StepBackupAndRestore(directory, 4, S)], epochs=2)
    assert metadata(directory) == {"epoch": 2, "batch": 0} and weights_steps(directory) == 2 * S
    assert settle_backup(directory, S) == ("latest", 2)
    # The same weights under a steps-per-epoch that does not divide them are refused (both copies).
    with pytest.raises(ValueError, match="different epochs"):
        settle_backup(directory, S + 1)


def test_weights_newer_than_their_metadata_are_detected_and_the_previous_pair_is_chosen(tmp_path):
    """A kill between ``save_weights`` and the metadata write of a mid-epoch save."""
    directory = _two_copies(tmp_path)
    # The state that kill leaves: the new weights (8 batches), the metadata of the copy before (4).
    shutil.copyfile(directory / (BACKUP_METADATA_NAME + PREVIOUS_SUFFIX), directory / BACKUP_METADATA_NAME)
    assert weights_steps(directory) == S + 8
    assert settle_backup(directory, S) == ("previous", 1)
    assert weights_steps(directory) == S + 4 and metadata(directory) == {"epoch": 1, "batch": 3}


def test_a_mid_epoch_metadata_over_the_epoch_end_weights_is_not_accepted(tmp_path):
    """A kill between the writes of an EPOCH-END save: epoch-end weights, the last mid-epoch metadata."""
    directory = tmp_path / "b"
    assert not fit(tiny(), [StepBackupAndRestore(directory, 4, S)], epochs=1)
    assert metadata(directory) == {"epoch": 1, "batch": 0} and weights_steps(directory) == S
    (directory / BACKUP_METADATA_NAME).write_text(json.dumps({"epoch": 0, "batch": 7}))
    (directory / (BACKUP_METADATA_NAME + PREVIOUS_SUFFIX)).write_text(json.dumps({"epoch": 0, "batch": 7}))
    shutil.copyfile(directory / BACKUP_WEIGHTS_NAME, directory / (BACKUP_WEIGHTS_NAME + PREVIOUS_SUFFIX))
    with pytest.raises(ValueError, match="different epochs"):
        settle_backup(directory, S)


def test_a_copy_that_claims_the_epoch_start_but_holds_mid_epoch_weights_is_not_accepted(tmp_path):
    directory = _two_copies(tmp_path)
    (directory / BACKUP_METADATA_NAME).write_text(json.dumps({"epoch": 1, "batch": 0}))
    # Latest says "epoch 1 not started" (10 steps) over weights of 18: refused; previous (14) is sound.
    assert settle_backup(directory, S) == ("previous", 1)


def test_the_resume_point_inside_the_first_epoch_is_a_fresh_early_stopping_state(tmp_path):
    backup = tmp_path / "backup"
    backup.mkdir()
    (backup / BACKUP_METADATA_NAME).write_text(json.dumps({"epoch": 0, "batch": 4}))
    sidecar = tmp_path / "fit_state.json"
    sidecar.write_text(json.dumps({"epochs": {}, "fit_finished": False, "initial": {"metrics": {}}}))
    point = read_resume_point(tmp_path / "training_log.csv", backup, sidecar)
    assert (point.start, point.best, point.wait, point.best_epoch, point.rows) == (0, None, 0, 0, ())
    assert point.reduce_lr is None and "initial" in point.state
    assert read_backup_position(backup) == (0, 4)
    # An epoch-end copy of no finished epoch (stock, or a mid-epoch copy cannot have batch 0) stays refused.
    (backup / BACKUP_METADATA_NAME).write_text(json.dumps({"epoch": 0, "batch": 0}))
    with pytest.raises(ValueError, match="no finished epoch"):
        read_resume_point(tmp_path / "training_log.csv", backup, sidecar)


def test_a_metadata_file_without_a_batch_counts_as_an_epoch_end_copy(tmp_path):
    (tmp_path / BACKUP_METADATA_NAME).write_text(json.dumps({"epoch": 3}))
    assert read_backup_position(tmp_path) == (3, 0)
    (tmp_path / BACKUP_METADATA_NAME).write_text(json.dumps({"epoch": 3, "batch": -1}))
    with pytest.raises(ValueError, match="batch"):
        read_backup_position(tmp_path)


# ---------------------------------------------------------------------
# 3. The configuration
# ---------------------------------------------------------------------


def test_the_key_defaults_to_off(tmp_path):
    assert CheckpointArgs().backup_every_steps is None
    assert make_config(tmp_path / "run").checkpoint_args.backup_every_steps is None
    assert make_config(tmp_path / "run", **{"checkpoint_args.backup_every_steps": 2}
                       ).checkpoint_args.backup_every_steps == 2


@pytest.mark.parametrize("value", [1, 0, -5, True, 2.5, "5"])
def test_a_value_that_cannot_work_is_refused_naming_the_key(tmp_path, value):
    with pytest.raises(ConfigValidationError, match="checkpoint_args.backup_every_steps"):
        make_config(tmp_path / "run", **{"checkpoint_args.backup_every_steps": value})


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        use_tiny_raw(patch, n_train=40, n_test=8)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


class StopBeforeFit(Exception):
    pass


class Peek(ImageClassificationTrainer):
    """The real task, which reports its last callback and stops before ``fit``."""

    seen = None

    def _callbacks(self, *arguments):
        epoch_line, callbacks = super()._callbacks(*arguments)
        Peek.seen = callbacks[-1]
        raise StopBeforeFit()


def test_the_trainer_builds_the_stock_callback_when_the_key_is_off_and_the_subclass_when_it_is_set(world, tmp_path):
    for out, every in (("off", None), ("on", 4)):
        Peek.seen = None
        with pytest.raises(StopBeforeFit):
            Peek(make_config(tmp_path / out, **{"checkpoint_args.backup_every_steps": every})).run()
        if every is None:
            assert type(Peek.seen) is keras.callbacks.BackupAndRestore
            assert Peek.seen.save_freq == "epoch" and Peek.seen.double_checkpoint
        else:
            assert type(Peek.seen) is StepBackupAndRestore
            assert (Peek.seen.save_every_steps, Peek.seen.steps_per_epoch) == (4, 9)
            assert Peek.seen.double_checkpoint and not Peek.seen.delete_checkpoint


class PeekResumed(ImageClassificationTrainer):
    """The real task whose callbacks are built as for a resume at ``start``; stops before ``fit``."""

    start = 0
    seen = None

    def _callbacks(self, *arguments):
        from train.common.resume_state import ResumePoint

        arguments = list(arguments)
        arguments[7] = ResumePoint(
            start=self.start, best=None, wait=0, best_epoch=0, rows=(), state={"initial": {}})
        PeekResumed.seen = super()._callbacks(*arguments)[1]
        raise StopBeforeFit()


@pytest.mark.parametrize("start, owes_epoch_one", [(0, True), (1, False)])
def test_a_resume_inside_the_first_epoch_still_owes_the_epoch_one_analysis(world, tmp_path, start, owes_epoch_one):
    """Epoch 0's analysis ran before the first batch; epoch 1's is due only if the killed epoch was epoch 0."""
    from dl_techniques.callbacks.analyzer_callback import EpochAnalyzerCallback

    PeekResumed.start = start
    config = make_config(tmp_path / "run", **{"analysis_args.epoch_analysis_every": 1})
    with pytest.raises(StopBeforeFit):
        PeekResumed(config).run()
    analyzer = next(c for c in PeekResumed.seen if isinstance(c, EpochAnalyzerCallback))
    assert analyzer.analyze_on_train_begin is False
    assert analyzer.analyze_first_epoch is owes_epoch_one


def test_a_value_not_below_the_steps_of_an_epoch_is_refused_before_anything_trains(world, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match=r"backup_every_steps.*9 steps of an epoch"):
        ImageClassificationTrainer(make_config(out, **{"checkpoint_args.backup_every_steps": 9})).run()
    assert not (out / "checkpoints" / "backup").exists()
    assert not (out / "final_model.keras").exists()


# ---------------------------------------------------------------------
# 4. Real SIGKILL
# ---------------------------------------------------------------------

EPOCHS, EVERY, STEPS = 3, 5, 18


def child_env():
    return dict(os.environ, PYTHONPATH=f"{REPO / 'src'}:{REPO}", CUDA_VISIBLE_DEVICES="-1",
                TF_CPP_MIN_LOG_LEVEL="3", MPLBACKEND="Agg")


def run_child(spec, log, timeout=900):
    """Run the child; SIGKILL its recorded PID when it prints a stop marker. Return the last marker."""
    with open(log, "wb") as out:
        child = subprocess.Popen([sys.executable, str(CHILD), json.dumps(spec)], stdout=out,
                                 stderr=subprocess.STDOUT, env=child_env(), cwd=REPO)
    pid = child.pid  # the only process this module ever signals
    marker, deadline = None, time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            text = Path(log).read_bytes().decode("utf8", "replace")
            lines = [line for line in text.replace("\r", "\n").splitlines() if line.startswith("@@")]
            marker = lines[-1] if lines else None
            if marker and marker != "@@DONE":
                os.kill(pid, signal.SIGKILL)
                break
            if child.poll() is not None:
                break
            time.sleep(0.2)
    finally:
        if child.poll() is None and marker != "@@DONE":
            os.kill(pid, signal.SIGKILL)
        child.wait(timeout=120)
    if marker is None or (marker == "@@DONE") != (child.returncode == 0):
        marker = f"{marker} (exit {child.returncode}): " + Path(log).read_bytes().decode("utf8", "replace")[-1500:]
    return marker


def snapshot(run, name, root):
    """Copy what a resume reads (the backup directory, sidecar and CSV) so a test can inspect a kill."""
    target = Path(root) / name
    shutil.copytree(Path(run) / "checkpoints", target / "checkpoints")
    shutil.copyfile(Path(run) / "training_log.csv", target / "training_log.csv")
    return target


def csv_epochs(run):
    with open(Path(run) / "training_log.csv", newline="") as handle:
        return [row["epoch"] for row in csv.DictReader(handle)]


def csv_column(run, name):
    with open(Path(run) / "training_log.csv", newline="") as handle:
        return [row[name] for row in csv.DictReader(handle)]


def final_state(run):
    """``(model arrays, optimizer arrays)`` of ``final_model.keras``, by dataset name."""
    with zipfile.ZipFile(Path(run) / "final_model.keras") as archive:
        data = archive.read("model.weights.h5")
    model, optimizer = {}, {}
    with h5py.File(io.BytesIO(data), "r") as handle:
        def visit(name, node):
            if isinstance(node, h5py.Dataset) and node.size:
                (optimizer if name.startswith("optimizer/") else model)[name] = node[()].astype(np.float64)
        handle.visititems(visit)
    optimizer.pop("optimizer/vars/0", None)
    return model, optimizer


def distance(first, second):
    """``(max abs difference, L2 difference / L2 norm of first)`` over the arrays of two states."""
    assert first.keys() == second.keys()
    largest = max(float(np.max(np.abs(first[k] - second[k]))) for k in first)
    relative = np.sqrt(sum(float(np.sum((first[k] - second[k]) ** 2)) for k in first)
                       / sum(float(np.sum(first[k] ** 2)) for k in first))
    return largest, float(relative)


class Group:
    """A sequence of child processes on one run directory."""

    def __init__(self, root, name):
        self.root, self.name = Path(root), name
        self.run = self.root / name / "run"
        self.snaps = {}
        self.steps = []

    def go(self, label, expect, **spec):
        spec = {"out": str(self.run), "epochs": EPOCHS, "every": EVERY, **spec}
        marker = run_child(spec, self.root / f"{self.name}_{label}.log")
        assert marker == expect, f"{self.name}/{label}: {marker}"
        self.steps.append((label, marker))
        if marker != "@@DONE":
            self.snaps[label] = snapshot(self.run, f"{self.name}_{label}", self.root)


def group_whole(root):
    group = Group(root, "whole")
    group.go("run", "@@DONE")
    return group


def group_plain(root):
    group = Group(root, "plain")
    group.go("run", "@@DONE", every=None)
    return group


def group_other_seed(root):
    group = Group(root, "seed")
    group.go("run", "@@DONE", seed=8)
    return group


def group_boundary(root):
    group = Group(root, "boundary")
    group.go("kill", "@@AT 2 0", stop_at=[2, 0])
    group.go("resume", "@@DONE", resume=True)
    return group


def group_chain(root):
    """Five processes on one run: four kills in four different places, then the finish."""
    group = Group(root, "chain")
    group.go("p1_first_epoch", "@@AT 0 7", stop_at=[0, 7])
    group.go("p2_resume_then_mid_epoch", "@@AT 1 9", resume=True, stop_at=[1, 9])
    # The resumed process writes metadata 1 (epoch 1, batch 5) and 2 (epoch 1, batch 10): killed
    # at the second write, after the weights of batch 10 and before their metadata.
    group.go("p3_between_the_writes", "@@AT_META 2", resume=True, stop_at_meta_write=2)
    group.go("p4_just_after_the_resume", "@@AT 1 2", resume=True, stop_at=[1, 2])
    group.go("p5_finish", "@@DONE", resume=True)
    return group


def group_epend(root):
    group = Group(root, "epend")
    # Writes of a whole process: epoch 0 has 3 mid-epoch copies and its epoch-end copy (1-4), epoch 1
    # the same (5-8): the 8th is the epoch-end copy of epoch 1.
    group.go("kill", "@@AT_META 8", stop_at_meta_write=8)
    group.go("resume", "@@DONE", resume=True)
    return group


@pytest.fixture(scope="module")
def matrix(tmp_path_factory):
    root = tmp_path_factory.mktemp("sigkill")
    groups = (group_whole, group_plain, group_other_seed, group_boundary, group_chain,
              group_epend)
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        futures = {fn.__name__[len("group_"):]: pool.submit(fn, root) for fn in groups}
        return SimpleNamespace(root=root, **{name: future.result() for name, future in futures.items()})


def test_the_child_trains_the_steps_per_epoch_this_file_assumes(matrix):
    log = (matrix.whole.run / "run.log").read_text()
    assert f"{STEPS} steps/epoch" in log and f"{EPOCHS} epochs" in log


def test_a_kill_inside_the_first_epoch_leaves_a_pair_that_names_epoch_zero_and_resumes(matrix):
    snap = matrix.chain.snaps["p1_first_epoch"]
    backup = snap / "checkpoints" / "backup"
    assert metadata(backup) == {"epoch": 0, "batch": EVERY - 1}
    assert csv_epochs(snap) == []  # no epoch had ended: the file is the header alone
    assert settle_backup(backup, STEPS) == ("latest", 0)
    log = (matrix.chain.run / "run.log").read_text()
    assert f"Resuming inside epoch 0 of {EPOCHS}" in log
    assert "no epoch had finished" in log


def test_a_kill_inside_a_later_epoch_and_the_resume_that_follows_it_leave_epoch_one(matrix):
    backup = matrix.chain.snaps["p2_resume_then_mid_epoch"] / "checkpoints" / "backup"
    assert metadata(backup) == {"epoch": 1, "batch": EVERY - 1}
    assert weights_steps(backup) == STEPS + EVERY
    assert settle_backup(backup, STEPS) == ("latest", 1)


def test_a_kill_between_the_weights_and_the_metadata_of_a_mid_epoch_copy_is_answered_by_the_previous_pair(matrix):
    snap = matrix.chain.snaps["p3_between_the_writes"]
    backup = snap / "checkpoints" / "backup"
    # The window itself: weights of 10 batches of epoch 1, metadata of the copy at 5.
    assert weights_steps(backup) == STEPS + 2 * EVERY
    assert metadata(backup) == {"epoch": 1, "batch": EVERY - 1}
    assert settle_backup(backup, STEPS) == ("previous", 1)
    assert weights_steps(backup) == STEPS + EVERY
    log = (matrix.chain.run / "run.log").read_text()
    assert "the previous copy (epoch 1) was put in its place" in log


def test_a_kill_just_after_a_resume_leaves_the_pair_the_resume_started_from(matrix):
    backup = matrix.chain.snaps["p4_just_after_the_resume"] / "checkpoints" / "backup"
    assert metadata(backup) == {"epoch": 1, "batch": EVERY - 1}, "stock Keras would name epoch 0 here"
    assert weights_steps(backup) == STEPS + EVERY


def test_a_kill_between_the_writes_of_an_epoch_end_copy_is_answered_by_the_previous_pair(matrix):
    backup = matrix.epend.snaps["kill"] / "checkpoints" / "backup"
    assert weights_steps(backup) == 2 * STEPS, "the weights of the end of epoch 1"
    assert metadata(backup) == {"epoch": 1, "batch": 3 * EVERY - 1}, "the metadata of the copy before"
    assert settle_backup(backup, STEPS) == ("previous", 1)
    assert weights_steps(backup) == STEPS + 3 * EVERY
    assert csv_epochs(matrix.epend.snaps["kill"]) == ["0", "1"], "epoch 1's row exists and is trained again"


@pytest.mark.parametrize("name", ["chain", "epend", "boundary", "whole"])
def test_every_run_ends_with_each_epoch_once_the_asked_number_of_epochs_and_the_nominal_steps(matrix, name):
    run = getattr(matrix, name).run
    assert csv_epochs(run) == [str(e) for e in range(EPOCHS)], "a CSV row repeats or is missing"
    summary = json.loads((run / "results_summary.json").read_text())
    assert summary["epochs_run"] == EPOCHS
    history = json.loads((run / "training_history.json").read_text())
    assert len(history["loss"]) == EPOCHS
    backup = run / "checkpoints" / "backup"
    assert metadata(backup) == {"epoch": EPOCHS, "batch": 0}
    assert backup_iterations(backup / BACKUP_WEIGHTS_NAME) == (EPOCHS * STEPS, False)
    sidecar = json.loads((run / "checkpoints" / "fit_state.json").read_text())
    assert sorted(sidecar["epochs"], key=int) == [str(e) for e in range(EPOCHS)] and sidecar["fit_finished"]


@pytest.mark.parametrize("name", ["chain", "epend", "boundary"])
def test_the_learning_rate_of_every_epoch_is_the_unkilled_runs(matrix, name):
    assert csv_column(getattr(matrix, name).run, "lr") == csv_column(matrix.whole.run, "lr")


def test_the_resumed_runs_say_where_they_restarted_the_optimizer(matrix):
    log = (matrix.chain.run / "run.log").read_text()
    assert f"restarts from its first batch on those weights, and the optimizer's step count goes from {STEPS + EVERY} back to {STEPS}" in log
    assert "saves inside the epoch took" in log, "the cost of the saves is logged"


def test_an_unkilled_run_with_the_key_set_equals_one_without_it_weight_for_weight(matrix):
    with_key, without = final_state(matrix.whole.run), final_state(matrix.plain.run)
    for first, second in zip(with_key, without):
        assert first.keys() == second.keys()
        assert all(np.array_equal(first[key], second[key]) for key in first)
    assert csv_column(matrix.whole.run, "val_loss") == csv_column(matrix.plain.run, "val_loss")


def test_a_run_resumed_from_a_mid_epoch_copy_is_a_valid_run_that_is_not_the_unkilled_trajectory(matrix):
    """What the design predicts, and what it means.

    The killed epoch is trained again in the data order of epoch 0, from the weights and Adam
    moments of batch k, so the final weights differ from the unkilled run's. The difference is
    small (about 1% of the norm here) against a different seed (about 70%), the step count and
    the learning rate of every epoch are the unkilled run's, and the Adam moments (which depend
    on the data order) are not.
    """
    whole, chain, boundary = final_state(matrix.whole.run), final_state(matrix.chain.run), final_state(matrix.boundary.run)
    seed = final_state(matrix.other_seed.run)
    chain_max, chain_relative = distance(whole[0], chain[0])
    boundary_max, boundary_relative = distance(whole[0], boundary[0])
    _, seed_relative = distance(whole[0], seed[0])
    assert chain_max > 0 and boundary_max > 0, "a resume that reproduced the unkilled run would be ignoring the data order"
    assert chain_relative < 0.1 * seed_relative and boundary_relative < 0.1 * seed_relative
    assert chain_max < 0.02
    assert distance(whole[1], chain[1])[0] > 0


# ---------------------------------------------------------------------
# 5. The README states what a save costs, from the measurement
# ---------------------------------------------------------------------

README = (REPO / "src" / "train" / "README.md").read_text()
EVIDENCE = json.loads((HERE / "backup_cost_evidence.json").read_text())
IMAGENET_TRAIN_IMAGES = 1_271_167
#: The lowest and highest images per second of a cold ImageNet training run (``memory_fit_evidence.json``).
COLD_RATES = (96.3, 106.9)


def readme_region(name):
    begin, end = f"<!-- {name}:begin -->", f"<!-- {name}:end -->"
    return README[README.index(begin) + len(begin):README.index(end)].strip()


def test_the_readme_table_of_the_cost_per_save_is_the_evidence():
    rows = [line for line in readme_region("backup-cost").splitlines()[2:]]
    expected = [
        f"| `{r['family']}/{r['variant']}` | {r['optimizer']} | {r['params']:,} | {r['file_mb']:.1f} | {r['median_later_s']:.2f} |"
        for r in EVIDENCE["rows"]]
    assert rows == expected
    for r in EVIDENCE["rows"]:
        assert r["median_later_s"] == sorted(r["later_s"])[1], "the median of saves 2 to 4"


def test_the_readme_sentence_for_500_batches_is_the_arithmetic_of_the_table_and_the_rates():
    seconds = {f"{r['family']}/{r['variant']}": r["median_later_s"] for r in EVIDENCE["rows"]}
    sentences = []
    for model, batch in (("vit/vit_pico", 128), ("resnet/resnet18", 256)):
        steps = IMAGENET_TRAIN_IMAGES // batch
        copies = (steps - 1) // 500
        low, high = (500 * IMAGENET_TRAIN_IMAGES / rate / steps / 60 for rate in reversed(COLD_RATES))
        sentences.append((model, batch, f"{steps:,}", copies, seconds[model], f"{copies * seconds[model]:.1f}",
                          f"{low:.1f}", f"{high:.1f}"))
    (m1, b1, steps1, copies1, s1, total1, low1, high1), (m2, b2, steps2, copies2, s2, total2, low2, high2) = sentences
    flat = " ".join(README.split())
    assert (f"`{m1}` at batch {b1} ({steps1} steps an epoch) takes {copies1} copies of {s1:.2f} s inside an epoch "
            f"({total1} s) and a kill loses at most {low1} to {high1} minutes of training") in flat
    assert (f"`{m2}` at batch {b2} ({steps2} steps) takes {copies2} copies of {s2:.2f} s ({total2} s) and loses at "
            f"most {low2} to {high2} minutes") in flat


def test_the_key_has_a_row_in_the_knob_table_and_the_section_stands_outside_the_knob_region():
    knobs = README[README.index("<!-- knobs:begin -->"):README.index("<!-- knobs:end -->")]
    assert "| `checkpoint_args.backup_every_steps` | `null` |" in knobs
    assert "Backing up inside an epoch" not in knobs and "<!-- backup-cost:begin -->" not in knobs
