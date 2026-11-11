"""``checkpoint_args.backup_every_steps``: a killed epoch loses at most N batches of weights AND of time.

Step 19 of plan-2026-10-31T180033-2faf890a, decision D-038; step 22.2, decision D-063, which makes
the time claim true (D-038 restarted the killed epoch from its first batch: the review measured
36 batches for a resume that should have run 21, and 69 real updates against a counter of 54).
Stock Keras 3.8 ``BackupAndRestore(save_freq=N)`` cannot be used as it is (all MEASURED, the first
group below re-measures each on a tiny model so an upgrade that changes one fails here):

* it does not save at an epoch end;
* a fresh process keeps ``_current_epoch`` at 0, so the first mid-epoch save after a resume
  records epoch 0 over the weights of epoch k, and a second kill resumes from epoch 0;
* a resume restarts the epoch at its first batch, so the batches of the copy are trained twice.

``train.common.resume_state.StepBackupAndRestore`` fixes those: (a) it saves at the epoch end,
counting batches inside the epoch; (b) it sets ``_current_epoch`` from the restored metadata; (c)
with its companion ``FinishKilledEpoch`` it makes the resumed epoch run only the batches the copy
had not taken, from a fresh pass of the data, and leaves the optimizer's step count alone. What a
resume then is: the epoch killed after k of its S batches runs ``S - k`` more (the copy's k
counted), the epochs after it run in full, the step count and the learning rate of every step are
the unkilled run's, the examples of the killed epoch are the first k of its pass in the killed
process plus the first ``S - k`` of a fresh pass, so some repeat and some are missed.

Four groups:

1. the Keras facts the subclass relies on (in process, a tiny model);
2. the subclass on that model: which batches save, the metadata, how many batches the resumed
   epoch runs, ``steps_per_execution`` above 1, a loss-scaling optimizer, the argument checks,
   ``settle_backup`` on both copy forms and on a weights file newer than its metadata,
   ``read_resume_point`` inside the first epoch;
3. the configuration: the key, its refusals, the trainer's choice and order of callbacks;
4. REAL ``SIGKILL`` of a child process (``_backup_child.py``: the API fit of ``vit/vit_pico``
   on fake CIFAR-10, 18 batches an epoch, a copy every 5 batches) at real batches: inside the
   first epoch, inside a later one, between ``save_weights`` and the metadata write of a
   mid-epoch save and of an epoch-end save, just after a resume, inside the shortened epoch, after
   it, and at the epoch end that follows it. Each child writes one trace line per finished batch, so
   the REAL number of updates is counted: it is the 54 of the unkilled run plus exactly the batches
   lost between each kill and the copy it resumed from, and the learning rate of every step is the
   unkilled run's. The CSV rows are unique and the epoch count is the asked one.

What the difference from an unkilled run is: the shortened epoch draws from a fresh pass, so a
resumed run is a valid run of the same recipe with the same step count and learning rate at every
step, and NOT the unkilled run's trajectory; the final weights differ by a few per cent of their
norm in this setup, against about 70% for another seed (the numbers are asserted below as bounds
and printed in the README).

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
from train.api.tasks.image_classification import ImageClassificationTrainer
from train.common.resume_state import (
    BACKUP_METADATA_NAME,
    BACKUP_WEIGHTS_NAME,
    PREVIOUS_SUFFIX,
    FinishKilledEpoch,
    FitStateSidecar,
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


def backed_up(directory, every=4, steps=S, cls=StepBackupAndRestore):
    """``(backup, [backup.finish, backup])``: the two callbacks in the order the trainer gives them."""
    backup = cls(directory, every, steps)
    return backup, [backup.finish, backup]


def steps_taken(model):
    optimizer = model.optimizer
    return int(keras.ops.convert_to_numpy(getattr(optimizer, "inner_optimizer", optimizer).iterations))


class Ran(keras.callbacks.Callback):
    """``(epoch, steps it ran, step count at its start)`` for every epoch of a fit."""

    def __init__(self):
        super().__init__()
        self.rows, self._start = [], 0

    def on_epoch_begin(self, epoch, logs=None):
        self._start = steps_taken(self.model)

    def on_epoch_end(self, epoch, logs=None):
        self.rows.append((epoch, steps_taken(self.model) - self._start, self._start))


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
    assert epoch == 1 and inner == steps_in_copy, "stock runs epoch 1 in full on the copy's count (16): 6 batches twice"


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


def test_the_resumed_epoch_runs_only_the_batches_the_copy_had_not_taken(tmp_path):
    directory = tmp_path / "b"
    assert fit(tiny(), [KillAt(1, 6), *backed_up(directory)[1]])
    assert metadata(directory) == {"epoch": 1, "batch": 3}
    backup, callbacks = backed_up(directory, cls=Recording)
    ran, model = Ran(), tiny()
    assert not fit(model, [*callbacks, ran])
    # Epoch 1 runs the 6 batches the copy (4 batches) had not taken, from step 14; epoch 2 runs in full.
    assert ran.rows == [(1, S - 4, S + 4), (2, S, 2 * S)]
    assert backup.resumed_inside == (1, 4)
    assert steps_taken(model) == 3 * S, "the counter is the unkilled run's: nothing trained twice, nothing rewound"
    # Its first copy is after 4 more batches, at the batch 8 of the WHOLE epoch, and names epoch 1 (b).
    assert backup.saved == [(1, 7), (2, 0), (2, 3), (2, 7), (3, 0)]
    assert backup.steps_left is None, "after the shortened epoch every epoch is a whole one"


def test_a_resume_from_a_copy_two_batches_before_the_end_runs_two_batches_and_one_more_epoch(tmp_path):
    directory = tmp_path / "b"
    assert fit(tiny(), [KillAt(1, 9), *backed_up(directory)[1]])
    assert metadata(directory) == {"epoch": 1, "batch": 7}
    ran = Ran()
    assert not fit(tiny(), [*backed_up(directory)[1], ran])
    assert ran.rows == [(1, 2, S + 8), (2, S, S + 10)]


def test_a_second_kill_after_the_resume_leaves_a_pair_that_still_resumes_into_the_same_epoch(tmp_path):
    directory = tmp_path / "b"
    assert fit(tiny(), [KillAt(1, 6), *backed_up(directory)[1]])
    # Killed again two batches after the resume, before it saved anything: the pair is as it was.
    assert fit(tiny(), [KillAt(1, 2), *backed_up(directory)[1]])
    assert metadata(directory) == {"epoch": 1, "batch": 3} and settle_backup(directory, S) == ("latest", 1)
    # Killed once more after its first new save (after 4 more batches): the pair names epoch 1,
    # batch 8 of the WHOLE epoch (the copy's 4 and the 4 new ones), not epoch 0.
    assert fit(tiny(), [KillAt(1, 5), *backed_up(directory)[1]])
    assert metadata(directory) == {"epoch": 1, "batch": 7}
    assert weights_steps(directory) == S + 8 and settle_backup(directory, S) == ("latest", 1)
    # Killed after one batch of the 2 that remain: the pair is the one it started from.
    assert fit(tiny(), [KillAt(1, 1), *backed_up(directory)[1]])
    assert metadata(directory) == {"epoch": 1, "batch": 7}
    ran, final = Ran(), tiny()
    assert not fit(final, [*backed_up(directory)[1], ran])
    assert ran.rows == [(1, 2, S + 8), (2, S, S + 10)]
    assert steps_taken(final) == 3 * S


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
        backup = StepBackupAndRestore(directory, 4, S)
        callbacks += [FitStateSidecar(stopper, sidecar, reduce_lr=plateau), backup]
        # FinishKilledEpoch directly before EarlyStopping, as the trainer orders them.
        callbacks.insert(callbacks.index(stopper), backup.finish)
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


@pytest.mark.parametrize("patience, plateau, kill, epochs", [
    # EarlyStopping stops when `wait` reaches 2: whole run after epoch 2; the resume must stop there too.
    (2, dict(patience=99), (2, 6), 6),
    # ReduceLROnPlateau halves the rate when its `wait` reaches 3, at the end of epoch 3.
    (99, dict(patience=3), (3, 6), 6),
    # After a halving at the end of epoch 1 it waits out a cooldown of 2 epochs (counter 2 -> 0 by the end of epoch 3).
    (99, dict(patience=1, cooldown=2), (2, 6), 6),
])
def test_the_wait_and_the_cooldown_of_the_callbacks_come_back_across_a_mid_epoch_resume(
        tmp_path, patience, plateau, kill, epochs):
    """The epochs a resumed fit runs and the rate each starts at are the unkilled run's (never improving: min_delta is huge)."""
    from train.common.resume_state import FitStateSidecar, RestoreFitState

    directory, sidecar = tmp_path / "b", tmp_path / "fit_state.json"

    def plain():
        keras.utils.set_random_seed(0)
        model = keras.Sequential([keras.Input((4,)), keras.layers.Dense(3), keras.layers.Dense(1)])
        model.compile(keras.optimizers.Adam(0.1), "mse")
        return model

    class Rate(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.starts = []

        def on_epoch_begin(self, epoch, logs=None):
            self.starts.append((epoch, round(float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)), 6)))

    def run(kill_at=None, resume=None):
        stopper = keras.callbacks.EarlyStopping(monitor="loss", patience=patience, min_delta=1e9)
        reducer = keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, min_delta=1e9, **plateau)
        backup = StepBackupAndRestore(directory, 4, S)
        callbacks = [stopper, reducer]
        if resume is not None:
            callbacks.append(RestoreFitState(stopper, resume.best, resume.wait, None, resume.best_epoch,
                                             reduce_lr=reducer, reduce_lr_state=resume.reduce_lr))
        callbacks += [FitStateSidecar(stopper, sidecar, reduce_lr=reducer), backup]
        callbacks.insert(callbacks.index(stopper), backup.finish)
        rate = Rate()
        killer = [KillAt(*kill_at)] if kill_at else []
        try:
            history = plain().fit(X, Y, batch_size=8, epochs=epochs, shuffle=False, verbose=0,
                                  callbacks=[rate, *callbacks, *killer])
        except Killed:
            return None, rate
        return history.epoch, rate

    whole_epochs, whole_rate = run()
    shutil.rmtree(directory)
    sidecar.unlink()
    killed, _ = run(kill_at=kill)
    assert killed is None and metadata(directory) == {"epoch": kill[0], "batch": 3}
    (tmp_path / "log.csv").write_text("epoch,loss\n" + "".join(f"{e},1.0\n" for e in range(kill[0])))
    point = read_resume_point(tmp_path / "log.csv", directory, sidecar)
    resumed_epochs, resumed_rate = run(resume=point)
    first = kill[0]
    assert resumed_epochs == [e for e in whole_epochs if e >= first], "the resume stops where the unkilled run stopped"
    assert resumed_rate.starts == [row for row in whole_rate.starts if row[0] >= first]
    assert len({rate for _, rate in whole_rate.starts}) > 1 or patience < 99, "the vacuity guard: the rate moves, or the stop is the point"


def test_a_resume_from_an_epoch_end_copy_changes_nothing(tmp_path):
    directory = tmp_path / "b"
    assert fit(tiny(), [KillAt(2, 0), *backed_up(directory)[1]])
    assert metadata(directory) == {"epoch": 2, "batch": 0}
    backup, callbacks = backed_up(directory)
    watch, ran = Watch(backup), Ran()
    assert not fit(tiny(), [*callbacks, watch, ran])
    assert watch.starts == [(2, 2 * S, 2 * S, 2)]
    assert ran.rows == [(2, S, 2 * S)] and backup.resumed_inside is None


def test_steps_per_execution_above_one_saves_at_group_boundaries_and_records_the_last_batch(tmp_path):
    directory = tmp_path / "b"
    backup, callbacks = backed_up(directory, cls=Recording)
    assert fit(tiny(4), [KillAt(1, 8), *callbacks])
    # Groups of 4 start at batches 0, 4, 8: copies after 4 and 8 batches, none after the last (10).
    assert backup.saved == [(0, 3), (0, 7), (1, 0), (1, 3), (1, 7)]
    assert metadata(directory) == {"epoch": 1, "batch": 7}
    assert weights_steps(directory) == S + 8
    assert settle_backup(directory, S) == ("latest", 1)
    # 2 batches are left and an execution takes 4: the epoch could not end where the copy says.
    with pytest.raises(ValueError, match="steps_per_execution is 4.*not a multiple"):
        fit(tiny(4), backed_up(directory)[1])


def test_steps_per_execution_that_divides_the_remaining_batches_resumes_exactly(tmp_path):
    directory = tmp_path / "b"
    assert fit(tiny(2), [KillAt(1, 6), *backed_up(directory)[1]])
    assert metadata(directory) == {"epoch": 1, "batch": 3}
    ran, final = Ran(), tiny(2)
    assert not fit(final, [*backed_up(directory)[1], ran])
    assert ran.rows == [(1, S - 4, S + 4), (2, S, 2 * S)]
    assert steps_taken(final) == 3 * S


def test_a_loss_scaling_optimizer_keeps_its_own_counter_and_the_resume_never_adds_to_it(tmp_path):
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        directory = tmp_path / "b"
        assert fit(tiny(), [KillAt(1, 6), *backed_up(directory, 2)[1]])
        taken, scaled = backup_iterations(directory / BACKUP_WEIGHTS_NAME)
        assert scaled, "the vacuity guard: mixed_float16 compiles a LossScaleOptimizer"
        # The inner counter is the one that moved; it may lag the batches (steps with
        # non-finite gradients are skipped), and settle_backup accepts a lag, not a surplus.
        assert settle_backup(directory, S) == ("latest", 1)
        batch = metadata(directory)["batch"]
        backup, callbacks = backed_up(directory, 2)
        watch = Watch(backup)
        assert fit(tiny(), [*callbacks, watch, KillAt(1, 1)])
        (epoch, inner, outer, current), = watch.starts
        assert (epoch, current, outer) == (1, 1, 0)
        assert inner == taken and inner <= S + batch + 1, "continued from the copy, not rewound"
        # A whole run ends on the batches it ran (a lag is allowed, a surplus is not).
        ran, final = Ran(), tiny()
        assert not fit(final, [*backed_up(directory, 2)[1], ran])
        assert ran.rows[0][1] <= S - batch - 1
        assert steps_taken(final) <= 3 * S

        # Killed inside epoch 0 after one copy: the resume continues from the copy's count, not from 0.
        first = tmp_path / "first"
        assert fit(tiny(), [KillAt(0, 5), *backed_up(first)[1]])
        copy_steps = backup_iterations(first / BACKUP_WEIGHTS_NAME)[0]
        backup, callbacks = backed_up(first)
        watch = Watch(backup)
        assert fit(tiny(), [*callbacks, watch, KillAt(0, 1)])
        assert watch.starts[0][1] == copy_steps and watch.starts[0][2] == 0
    finally:
        keras.mixed_precision.set_global_policy("float32")


class StopAt(keras.callbacks.Callback):
    """Ask Keras to stop: at a batch end (``batch`` given) or at an epoch end."""

    def __init__(self, epoch, batch=None):
        super().__init__()
        self.epoch, self.batch, self.at = epoch, batch, 0

    def on_epoch_begin(self, epoch, logs=None):
        self.at = epoch

    def on_train_batch_end(self, batch, logs=None):
        if self.batch is not None and self.at == self.epoch and batch == self.batch:
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        if self.batch is None and epoch == self.epoch:
            self.model.stop_training = True


def test_a_stop_request_made_before_the_companion_at_the_cut_batch_is_kept(tmp_path):
    """``TerminateOnNaN`` sits before ``FinishKilledEpoch``: its request at the last remaining batch must stand."""
    directory = tmp_path / "b"
    assert fit(tiny(), [KillAt(1, 6), *backed_up(directory)[1]])
    backup, callbacks = backed_up(directory)
    ran = Ran()
    # 6 batches remain: the request comes at the batch index 5 of the shortened epoch, the cut batch.
    model = tiny()
    history = model.fit(X, Y, batch_size=8, epochs=4, shuffle=False, verbose=0,
                        callbacks=[StopAt(1, 5), *callbacks, ran])
    assert history.epoch == [1] and ran.rows == [(1, S - 4, S + 4)], "the fit ended after the cut epoch"


def test_a_stop_request_made_at_the_end_of_the_cut_epoch_is_kept(tmp_path):
    """``EarlyStopping`` sits after ``FinishKilledEpoch``: the flag it sets at the epoch end must stand."""
    directory = tmp_path / "b"
    assert fit(tiny(), [KillAt(1, 6), *backed_up(directory)[1]])
    backup, callbacks = backed_up(directory)
    model = tiny()
    history = model.fit(X, Y, batch_size=8, epochs=4, shuffle=False, verbose=0,
                        callbacks=[*callbacks[:1], StopAt(1), *callbacks[1:]])
    assert history.epoch == [1]
    assert steps_taken(model) == 2 * S, "the shortened epoch ran 6 batches from step 14 and then training ended"


def test_the_flag_is_put_back_so_the_epochs_after_the_cut_one_run(tmp_path):
    directory = tmp_path / "b"
    assert fit(tiny(), [KillAt(1, 6), *backed_up(directory)[1]])
    model = tiny()
    history = model.fit(X, Y, batch_size=8, epochs=4, shuffle=False, verbose=0, callbacks=backed_up(directory)[1])
    assert history.epoch == [1, 2, 3] and model.stop_training is False


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
    assert point.batch == 4, "the batch the copy had taken: the resumed epoch is shortened by it"
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
    """The real task, which reports its last callback and its whole list, and stops before ``fit``."""

    seen = None
    listed = None

    def _callbacks(self, *arguments):
        epoch_line, callbacks = super()._callbacks(*arguments)
        Peek.seen, Peek.listed = callbacks[-1], list(callbacks)
        raise StopBeforeFit()


def test_the_trainer_builds_the_stock_callback_when_the_key_is_off_and_the_subclass_when_it_is_set(world, tmp_path):
    for out, every in (("off", None), ("on", 4)):
        Peek.seen = None
        with pytest.raises(StopBeforeFit):
            Peek(make_config(tmp_path / out, **{"checkpoint_args.backup_every_steps": every})).run()
        if every is None:
            assert type(Peek.seen) is keras.callbacks.BackupAndRestore
            assert Peek.seen.save_freq == "epoch" and Peek.seen.double_checkpoint
            assert not any(isinstance(c, FinishKilledEpoch) for c in Peek.listed)
        else:
            assert type(Peek.seen) is StepBackupAndRestore
            assert (Peek.seen.save_every_steps, Peek.seen.steps_per_epoch) == (4, 9)
            assert Peek.seen.double_checkpoint and not Peek.seen.delete_checkpoint


@pytest.mark.parametrize("terminate_on_nan", [False, True])
def test_the_companion_sits_directly_before_early_stopping_and_after_terminate_on_nan(world, tmp_path, terminate_on_nan):
    """FinishKilledEpoch reads TerminateOnNaN's request at the batch end and gives the flag back before EarlyStopping."""
    from train.common.callbacks import LearningRateLogger

    Peek.listed = None
    config = make_config(tmp_path / "run", **{"checkpoint_args.backup_every_steps": 4,
                                              "debug_args.terminate_on_nan": terminate_on_nan})
    with pytest.raises(StopBeforeFit):
        Peek(config).run()
    listed = Peek.listed
    companion = [c for c in listed if isinstance(c, FinishKilledEpoch)]
    assert len(companion) == 1 and companion[0] is Peek.seen.finish
    place = listed.index(companion[0])
    assert isinstance(listed[place + 1], keras.callbacks.EarlyStopping)
    before = listed[:place]
    assert isinstance(before[0], LearningRateLogger)
    assert [type(c) for c in before[1:]] == ([keras.callbacks.TerminateOnNaN] if terminate_on_nan else [])
    assert listed.index(Peek.seen) > max(i for i, c in enumerate(listed) if isinstance(c, FitStateSidecar)), (
        "the backup stays after the sidecar")


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


@pytest.mark.parametrize("batch_size, every, offered, not_offered", [
    (4, 9, "Choose a value from 2 to 8 or null", None),
    (12, 3, "Choose 2 or null", "from 2 to 2"),
    (16, 2, "has no value from 2 up that is below it, so choose null", "from 2 to 2"),
], ids=["nine_steps", "three_steps", "two_steps"])
def test_the_refusal_offers_only_values_that_would_work(world, tmp_path, batch_size, every, offered, not_offered):
    """Review-iter-2 concern 15: at 2 steps an epoch the refusal read "Choose a value from 2 to 2 or null",
    offering the value it had just refused. The data of ``world`` is 36 training rows, so the batch sets
    the steps of an epoch: 4 gives 9, 12 gives 3, 16 gives 2."""
    out = tmp_path / "run"
    with pytest.raises(ConfigError) as caught:
        ImageClassificationTrainer(make_config(
            out, batch_size=batch_size, **{"checkpoint_args.backup_every_steps": every})).run()
    assert offered in str(caught.value)
    if not_offered:
        assert not_offered not in str(caught.value)


# ---------------------------------------------------------------------
# 4. Real SIGKILL
# ---------------------------------------------------------------------

EPOCHS, EVERY, STEPS = 3, 5, 18


def child_env():
    return dict(os.environ, PYTHONPATH=f"{REPO / 'src'}:{REPO}", CUDA_VISIBLE_DEVICES="-1",
                TF_CPP_MIN_LOG_LEVEL="3", MPLBACKEND="Agg")


def run_child(spec, log, timeout=900, child_script=CHILD):
    """Run the child; SIGKILL its recorded PID when it prints a stop marker. Return the last marker.

    ``child_script`` is the script that speaks the ``@@`` marker protocol of ``_backup_child.py`` (the denoising
    resume test passes its own); the default is the classification child, so every other caller is unchanged.
    """
    with open(log, "wb") as out:
        child = subprocess.Popen([sys.executable, str(child_script), json.dumps(spec)], stdout=out,
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


def trace_of(path):
    """``[(epoch, batch, steps before, lr)]`` of a run's trace file, in the order the batches ran."""
    if not Path(path).exists():
        return []
    return [(int(e), int(b), int(i), float(lr)) for e, b, i, lr in (line.split() for line in Path(path).read_text().splitlines())]


class Group:
    """A sequence of child processes on one run directory, and what each one ran."""

    def __init__(self, root, name):
        self.root, self.name = Path(root), name
        self.run = self.root / name / "run"
        self.trace = self.root / name / "trace.txt"
        self.snaps = {}
        self.steps = []
        self.ran = []      # batches each process ran (lines it added to the trace)
        self.wall = {}     # seconds each process took, start to marker or exit

    def go(self, label, expect, **spec):
        self.trace.parent.mkdir(parents=True, exist_ok=True)
        spec = {"out": str(self.run), "epochs": EPOCHS, "every": EVERY, "trace": str(self.trace), **spec}
        before, started = len(trace_of(self.trace)), time.monotonic()
        marker = run_child(spec, self.root / f"{self.name}_{label}.log")
        self.wall[label] = time.monotonic() - started
        assert marker == expect, f"{self.name}/{label}: {marker}"
        self.steps.append((label, marker))
        self.ran.append(len(trace_of(self.trace)) - before)
        if marker != "@@DONE":
            self.snaps[label] = snapshot(self.run, f"{self.name}_{label}", self.root)

    def tear_latest_weights(self):
        """What a kill inside the weights write leaves: a latest.weights.h5 that is not a file HDF5 opens."""
        path = self.run / "checkpoints" / "backup" / BACKUP_WEIGHTS_NAME
        path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])


KILL_15 = dict(stop_at=[1, 15])  # 15 batches of epoch 1 done: the copy after batch 15 (N = 5) is the last


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
    # The resumed process continues epoch 1 from the copy after 5 batches (the copy's 5 are not run
    # again) and writes copy 1 after 5 more (batch 10 of the epoch) and copy 2 after 10 more (batch
    # 15): killed at the second write, after the weights of batch 15 and before their metadata.
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


def group_kill15(root):
    """The review's reproduction (probes/t1.py): killed after 15 of 18 batches of epoch 1, N = 5."""
    group = Group(root, "kill15")
    group.go("kill", "@@AT 1 15", **KILL_15)
    group.go("resume", "@@DONE", resume=True)
    return group


def group_nokey(root):
    """The same kill with the key unset: the stock callback reruns the whole killed epoch."""
    group = Group(root, "nokey")
    group.go("kill", "@@AT 1 15", every=None, **KILL_15)
    group.go("resume", "@@DONE", every=None, resume=True)
    return group


def group_short_begin(root):
    """Killed just after the shortened epoch's process began (before its first batch)."""
    group = Group(root, "short_begin")
    group.go("kill", "@@AT 1 15", **KILL_15)
    group.go("resume_killed_at_begin", "@@AT 1 0", resume=True, stop_at=[1, 0])
    group.go("resume", "@@DONE", resume=True)
    return group


def group_short_meta(root):
    """Killed between the weights and the metadata of the epoch-end copy the shortened epoch writes."""
    group = Group(root, "short_meta")
    group.go("kill", "@@AT 1 15", **KILL_15)
    group.go("resume_killed_in_the_epoch_end_write", "@@AT_META 1", resume=True, stop_at_meta_write=1)
    group.go("resume", "@@DONE", resume=True)
    return group


def group_second(root):
    """Killed in the epoch AFTER the shortened one, after its first copy; resumed again."""
    group = Group(root, "second")
    group.go("kill", "@@AT 1 15", **KILL_15)
    group.go("resume_killed_in_epoch_2", "@@AT 2 7", resume=True, stop_at=[2, 7])
    group.go("resume", "@@DONE", resume=True)
    return group


def group_second_begin(root):
    """Killed just after the shortened epoch ended, as the next epoch began."""
    group = Group(root, "second_begin")
    group.go("kill", "@@AT 1 15", **KILL_15)
    group.go("resume_killed_at_epoch_2", "@@AT 2 0", resume=True, stop_at=[2, 0])
    group.go("resume", "@@DONE", resume=True)
    return group


def group_torn(root):
    """Killed inside the weights write of the newest copy: the previous pair (batch 10) is the resume point."""
    group = Group(root, "torn")
    group.go("kill", "@@AT 1 15", **KILL_15)
    group.tear_latest_weights()
    group.go("resume", "@@DONE", resume=True)
    return group


def group_mixed_whole(root):
    """``mixed_float16``: the optimizer is a LossScaleOptimizer whose inner counter lags the batches."""
    group = Group(root, "mixed_whole")
    group.go("run", "@@DONE", precision="mixed_float16")
    return group


def group_mixed_kill15(root):
    group = Group(root, "mixed_kill15")
    group.go("kill", "@@AT 1 15", precision="mixed_float16", **KILL_15)
    group.go("resume", "@@DONE", precision="mixed_float16", resume=True)
    return group


GROUPS = (group_whole, group_plain, group_other_seed, group_boundary, group_chain, group_epend,
          group_kill15, group_nokey, group_short_begin, group_short_meta, group_second,
          group_second_begin, group_torn, group_mixed_whole, group_mixed_kill15)


@pytest.fixture(scope="module")
def matrix(tmp_path_factory):
    """Run every group of children (six at a time).

    ``BACKUP_MATRIX_ONLY=kill15,torn`` runs just those groups, for a developer who wants one test
    (``-k``) and not the eight minutes of the rest; a test that needs another group then fails with
    an ``AttributeError``. Unset, all of them run, which is what every guard run does.
    """
    root = tmp_path_factory.mktemp("sigkill")
    only = [name for name in os.environ.get("BACKUP_MATRIX_ONLY", "").split(",") if name]
    chosen = [fn for fn in GROUPS if not only or fn.__name__[len("group_"):] in only]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {fn.__name__[len("group_"):]: pool.submit(fn, root) for fn in chosen}
        return SimpleNamespace(root=root, **{name: future.result() for name, future in futures.items()})


#: Batches each process of a group ran, in order, and the batches the kills lost: the work done since
#: the copy the next process resumed from. The sum of ``ran`` is ``EPOCHS * STEPS + lost``.
#: kill15: 18 + 15 batches, then the 3 batches epoch 1 had left and the whole epoch 2 (18 + 3 = 21).
RAN = {
    "whole": ([54], 0), "plain": ([54], 0), "other_seed": ([54], 0),
    "boundary": ([36, 18], 0),
    "chain": ([7, 22, 10, 2, 26], 13),
    "epend": ([36, 21], 3),
    "kill15": ([33, 21], 0),
    "nokey": ([33, 36], 15),
    "short_begin": ([33, 0, 21], 0),
    "short_meta": ([33, 3, 21], 3),
    "second": ([33, 10, 13], 2),
    "second_begin": ([33, 3, 18], 0),
    "torn": ([33, 26], 5),
}
#: The step at which the pass that wrote each epoch's final CSV row began: its ``lr`` is the rate there.
ROW_STARTS = {
    "whole": (0, 18, 36), "plain": (0, 18, 36), "other_seed": (0, 18, 36),
    "boundary": (0, 18, 36),
    "chain": (5, 28, 36),
    "epend": (0, 33, 36),
    "kill15": (0, 33, 36),
    "nokey": (0, 18, 36),
    "short_begin": (0, 33, 36), "short_meta": (0, 33, 36), "second": (0, 33, 41),
    "second_begin": (0, 33, 36), "torn": (0, 28, 36),
}


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
    assert f"the backup was taken after batch {EVERY} of {STEPS} of epoch 0; that epoch runs its remaining {STEPS - EVERY} batches" in log


def test_a_kill_inside_a_later_epoch_and_the_resume_that_follows_it_leave_epoch_one(matrix):
    backup = matrix.chain.snaps["p2_resume_then_mid_epoch"] / "checkpoints" / "backup"
    assert metadata(backup) == {"epoch": 1, "batch": EVERY - 1}
    assert weights_steps(backup) == STEPS + EVERY
    assert settle_backup(backup, STEPS) == ("latest", 1)


def test_a_kill_between_the_weights_and_the_metadata_of_a_mid_epoch_copy_is_answered_by_the_previous_pair(matrix):
    snap = matrix.chain.snaps["p3_between_the_writes"]
    backup = snap / "checkpoints" / "backup"
    # The window itself: weights of batch 15 of epoch 1, metadata of the copy at batch 10.
    assert weights_steps(backup) == STEPS + 3 * EVERY
    assert metadata(backup) == {"epoch": 1, "batch": 2 * EVERY - 1}
    assert settle_backup(backup, STEPS) == ("previous", 1)
    assert weights_steps(backup) == STEPS + 2 * EVERY
    log = (matrix.chain.run / "run.log").read_text()
    assert "the previous copy (epoch 1) was put in its place" in log


def test_a_kill_just_after_a_resume_leaves_the_pair_the_resume_started_from(matrix):
    backup = matrix.chain.snaps["p4_just_after_the_resume"] / "checkpoints" / "backup"
    assert metadata(backup) == {"epoch": 1, "batch": 2 * EVERY - 1}, "stock Keras would name epoch 0 here"
    assert weights_steps(backup) == STEPS + 2 * EVERY
    log = (matrix.chain.run / "run.log").read_text()
    assert f"the backup was taken after batch {2 * EVERY} of {STEPS} of epoch 1; that epoch runs its remaining {STEPS - 2 * EVERY} batches" in log


def test_a_kill_between_the_writes_of_an_epoch_end_copy_is_answered_by_the_previous_pair(matrix):
    backup = matrix.epend.snaps["kill"] / "checkpoints" / "backup"
    assert weights_steps(backup) == 2 * STEPS, "the weights of the end of epoch 1"
    assert metadata(backup) == {"epoch": 1, "batch": 3 * EVERY - 1}, "the metadata of the copy before"
    assert settle_backup(backup, STEPS) == ("previous", 1)
    assert weights_steps(backup) == STEPS + 3 * EVERY
    assert csv_epochs(matrix.epend.snaps["kill"]) == ["0", "1"], "epoch 1's row exists; the resume writes it again, once"


def test_a_kill_in_the_shortened_epochs_own_epoch_end_write_resumes_the_same_remaining_batches(matrix):
    group = matrix.short_meta
    backup = group.snaps["resume_killed_in_the_epoch_end_write"] / "checkpoints" / "backup"
    assert weights_steps(backup) == 2 * STEPS, "the weights of the end of epoch 1, written by the shortened epoch"
    assert metadata(backup) == {"epoch": 1, "batch": 3 * EVERY - 1}, "the metadata of the copy it resumed from"
    assert settle_backup(backup, STEPS) == ("previous", 1)
    assert group.ran == [33, 3, 21], "the 3 batches are run again, and nothing more"


def test_a_kill_inside_the_weights_write_resumes_from_the_previous_pair_and_the_summary_says_where(matrix):
    group = matrix.torn
    # Latest (batch 15) is torn; the previous copy is batch 10: 8 batches of epoch 1 remain, not 3.
    assert group.ran == [33, 26]
    log = (group.run / "run.log").read_text()
    assert f"the backup was taken after batch {2 * EVERY} of {STEPS} of epoch 1; that epoch runs its remaining {STEPS - 2 * EVERY} batches" in log
    note = " ".join(json.loads((group.run / "results_summary.json").read_text())["notes"])
    assert f"epoch 1 was killed after {2 * EVERY} of its {STEPS} batches; the resume ran its remaining {STEPS - 2 * EVERY}" in note
    assert "the epoch's row (index 1) holds the train metrics" in note


@pytest.mark.parametrize("name", [fn.__name__[len("group_"):] for fn in GROUPS if "mixed" not in fn.__name__])
def test_every_run_ends_with_each_epoch_once_the_asked_number_of_epochs_and_the_nominal_steps(matrix, name):
    run = getattr(matrix, name).run
    assert csv_epochs(run) == [str(e) for e in range(EPOCHS)], "a CSV row repeats or is missing"
    summary = json.loads((run / "results_summary.json").read_text())
    assert summary["epochs_run"] == EPOCHS
    history = json.loads((run / "training_history.json").read_text())
    assert len(history["loss"]) == EPOCHS and len(history["val_loss"]) == EPOCHS
    backup = run / "checkpoints" / "backup"
    assert metadata(backup) == {"epoch": EPOCHS, "batch": 0}
    assert backup_iterations(backup / BACKUP_WEIGHTS_NAME) == (EPOCHS * STEPS, False)
    sidecar = json.loads((run / "checkpoints" / "fit_state.json").read_text())
    assert sorted(sidecar["epochs"], key=int) == [str(e) for e in range(EPOCHS)] and sidecar["fit_finished"]


@pytest.mark.parametrize("name", list(RAN))
def test_the_real_updates_are_the_asked_steps_plus_exactly_the_batches_the_kills_lost(matrix, name):
    """The review counted 69 updates against a counter of 54 for the kill at batch 15 (N = 5); now it is 54."""
    group = getattr(matrix, name)
    ran, lost = RAN[name]
    assert group.ran == ran, f"batches run by each process of {name}"
    assert sum(group.ran) == EPOCHS * STEPS + lost
    trace = trace_of(group.trace)
    assert len(trace) == sum(ran)
    # Every step number of the run was taken, and the step count never skipped one (the last lines are the final lineage).
    assert sorted({line[2] for line in trace}) == list(range(EPOCHS * STEPS))
    assert backup_iterations(group.run / "checkpoints" / "backup" / BACKUP_WEIGHTS_NAME)[0] == EPOCHS * STEPS


def test_under_mixed_float16_the_resume_runs_the_same_batches_and_the_loss_scale_counter_never_exceeds_the_steps(matrix):
    """The loss-scaling optimizer skips steps with non-finite gradients, so its inner counter lags the batches run."""
    final_count = {}
    for name in ("mixed_whole", "mixed_kill15"):
        group = getattr(matrix, name)
        taken, scaled = backup_iterations(group.run / "checkpoints" / "backup" / BACKUP_WEIGHTS_NAME)
        final_count[name] = taken
        assert scaled, "the vacuity guard: mixed_float16 compiles a LossScaleOptimizer"
        assert taken <= EPOCHS * STEPS, (name, taken)
        assert metadata(group.run / "checkpoints" / "backup") == {"epoch": EPOCHS, "batch": 0}
        assert csv_epochs(group.run) == [str(e) for e in range(EPOCHS)]
    assert matrix.mixed_whole.ran == [54] and matrix.mixed_kill15.ran == [33, 21], "21 batches, as in float32"
    log = (matrix.mixed_kill15.run / "run.log").read_text()
    assert "that epoch runs its remaining 3 batches" in log
    # The counter the resume continued from is the copy's, lag included: it never went back (no rewind).
    trace = trace_of(matrix.mixed_kill15.trace)
    killed_last, resumed_first = trace[32], trace[33]
    assert resumed_first[:2] == (1, 0) and resumed_first[2] >= killed_last[2], "the counter went on from the copy, not back"
    assert 33 - (EPOCHS * STEPS - final_count["mixed_kill15"]) <= resumed_first[2] <= 33, "and it lags the batches by the skipped steps only"


def test_without_the_key_the_same_kill_runs_the_whole_epoch_again(matrix):
    """The comparison the key is for: same kill, same N; 69 updates against a counter of 54, and 15 more batches run."""
    assert matrix.nokey.ran == [33, 36] and matrix.kill15.ran == [33, 21]
    assert sum(matrix.nokey.ran) - sum(matrix.kill15.ran) == 15
    assert backup_iterations(matrix.nokey.run / "checkpoints" / "backup" / BACKUP_WEIGHTS_NAME)[0] == EPOCHS * STEPS


def test_the_resume_of_the_review_reproduction_runs_the_three_batches_left_and_the_last_epoch(matrix):
    group = matrix.kill15
    assert group.ran[1] == (STEPS - 3 * EVERY) + STEPS == 21
    trace = trace_of(group.trace)[group.ran[0]:]
    assert [(e, b, i) for e, b, i, _ in trace[:3]] == [(1, 0, 33), (1, 1, 34), (1, 2, 35)], "the three batches left, from step 33"
    assert (trace[3][0], trace[3][1], trace[3][2]) == (2, 0, 36)


@pytest.fixture(scope="module")
def whole_lr_by_step(matrix):
    return {i: lr for _, _, i, lr in trace_of(matrix.whole.trace)}


@pytest.mark.parametrize("name", [n for n in RAN if n != "whole"])
def test_the_learning_rate_of_every_step_is_the_unkilled_runs(matrix, whole_lr_by_step, name):
    """Each batch of each process used the rate the unkilled run used at the same step (the schedule is per step)."""
    trace = trace_of(getattr(matrix, name).trace)
    assert trace and all(lr == whole_lr_by_step[step] for _, _, step, lr in trace)
    assert len({lr for lr in whole_lr_by_step.values()}) > 3, "the vacuity guard: the schedule moves"


@pytest.mark.parametrize("name", list(RAN))
def test_the_lr_column_of_every_epoch_is_the_rate_at_the_first_step_the_row_covers(matrix, whole_lr_by_step, name):
    column = [float(value) for value in csv_column(getattr(matrix, name).run, "lr")]
    assert column == [whole_lr_by_step[step] for step in ROW_STARTS[name]]


def test_the_resumed_runs_say_what_they_ran_and_what_the_saves_cost(matrix):
    log = (matrix.chain.run / "run.log").read_text()
    assert f"that epoch runs its remaining {STEPS - 2 * EVERY} batches from a fresh pass of the data (some examples of it repeat and some are missed) and ends; the optimizer's step count stays at {STEPS + 2 * EVERY}" in log
    assert "saves inside the epoch took" in log, "the cost of the saves is logged"
    note = " ".join(json.loads((matrix.kill15.run / "results_summary.json").read_text())["notes"])
    assert (f"epoch 1 was killed after {3 * EVERY} of its {STEPS} batches; the resume ran its remaining 3 from a fresh pass "
            "of the data, so some of its examples were seen twice and some not at all") in note
    assert "train metrics (loss, accuracy, ...), `lr` and `epoch_times` entry of those 3 batches only" in note


def test_the_epoch_row_of_a_resumed_epoch_covers_the_remaining_batches_and_the_whole_validation_set(matrix):
    """Epoch 1 of kill15 is one row written by the resumed process: 3 train batches, 1 validation pass."""
    rows = list(csv.DictReader(open(matrix.kill15.run / "training_log.csv", newline="")))
    assert [row["epoch"] for row in rows] == ["0", "1", "2"]
    # 3 batches of 4 images: the train accuracy of the row is a multiple of 1/12, a whole epoch's is of 1/72.
    accuracy = float(rows[1]["accuracy"])
    assert abs(accuracy * 12 - round(accuracy * 12)) < 1e-4
    whole = float(list(csv.DictReader(open(matrix.whole.run / "training_log.csv", newline="")))[1]["accuracy"])
    assert abs(whole * 72 - round(whole * 72)) < 1e-4


def test_an_unkilled_run_with_the_key_set_equals_one_without_it_weight_for_weight(matrix):
    with_key, without = final_state(matrix.whole.run), final_state(matrix.plain.run)
    for first, second in zip(with_key, without):
        assert first.keys() == second.keys()
        assert all(np.array_equal(first[key], second[key]) for key in first)
    assert csv_column(matrix.whole.run, "val_loss") == csv_column(matrix.plain.run, "val_loss")


def test_a_run_resumed_from_a_mid_epoch_copy_is_a_valid_run_that_is_not_the_unkilled_trajectory(matrix):
    """What the design predicts, and what it means.

    The shortened epoch draws its remaining batches from a fresh pass (the first batches of epoch 0's
    order, in a new process), so the final weights differ from the unkilled run's. The difference is
    small against a different seed, the step count and the learning rate of every step are the unkilled
    run's, and the Adam moments (which depend on the data order) differ.
    """
    whole, seed = final_state(matrix.whole.run), final_state(matrix.other_seed.run)
    _, seed_relative = distance(whole[0], seed[0])
    for name in ("chain", "kill15", "short_begin", "second", "torn"):
        state = final_state(getattr(matrix, name).run)
        largest, relative = distance(whole[0], state[0])
        assert largest > 0, f"{name}: a resume that reproduced the unkilled run would be ignoring the data order"
        assert relative < 0.1 * seed_relative, (name, relative, seed_relative)
        assert distance(whole[1], state[1])[0] > 0


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
    """What a kill costs: at most 500 batches of training, at the cold training rates. TRUE since D-063 (D-038 said it too
    while the resume reran the whole epoch: the review of iteration 2 measured 36 batches run for 21 owed)."""
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
    assert "the restart itself (process start, data load, compilation) comes on top" in flat


def test_the_readme_worked_example_of_a_resumed_epoch_is_the_arithmetic_of_the_formula():
    """A kill of the ``vit_pico`` ImageNet run at batch 128 after 7,200 of its 9,930 steps, with a copy every 500."""
    steps = IMAGENET_TRAIN_IMAGES // 128
    killed_after, every = 7200, 500
    copied = killed_after // every * every
    left = steps - copied
    share = left * copied / steps ** 2
    flat = " ".join(README.split())
    assert (f"after {killed_after:,} of the {steps:,} steps of an epoch the last copy is after step {copied:,}, the resume "
            f"runs the {left:,} steps that copy had not taken (not {steps:,})") in flat
    assert f"{100 * share:.1f}% of that epoch's examples are seen twice and the same share not at all" in flat
    assert "R(S - R)/S^2" in flat and "at most 25% (R = S/2)" in flat
    assert abs(left * (steps - left) / steps ** 2 - share) < 1e-12, "the share is symmetric in the two sides"
    # The exact count in the test setup (scratch_step22_2/logs/examples_seen.txt): 18 steps of 4 images, copy after 15.
    assert "10 of its 72 examples twice and 10 not at all" in flat


def test_the_readme_time_paragraph_is_the_arithmetic_of_the_resume_evidence():
    """Every second and count in the README's measured paragraph is a row of ``backup_resume_evidence.json`` or a difference of two."""
    evidence = json.loads((HERE / "backup_resume_evidence.json").read_text())
    flat = " ".join(README.split())
    cpu = evidence["cpu"]
    key = [r for r in cpu["rows"] if r["config"] == "key"]
    plain = [r for r in cpu["rows"] if r["config"] == "nokey"]
    assert {r["resumed_batches"] for r in key} == {75} and {r["resumed_batches"] for r in plain} == {225}
    fit_key, fit_plain = [r["fit_wall_seconds"] for r in key], [r["fit_wall_seconds"] for r in plain]
    proc_key, proc_plain = [r["process_seconds"] for r in key], [r["process_seconds"] for r in plain]
    saved = sum(fit_plain) / 2 - sum(fit_key) / 2
    assert (f"with the key the resume ran 75 batches (`fit` {fit_key[0]:.1f} and {fit_key[1]:.1f} s, the whole process "
            f"{proc_key[0]:.1f} and {proc_key[1]:.1f} s) against 225 without it ({fit_plain[0]:.1f} and {fit_plain[1]:.1f} s; "
            f"{proc_plain[0]:.1f} and {proc_plain[1]:.1f} s): {saved:.1f} s of `fit` saved, {100 * saved / (sum(fit_plain) / 2):.0f}% of it, "
            f"and the run executed {key[0]['updates_executed']} updates against {plain[0]['updates_executed']} for the {cpu['steps_asked']} asked") in flat
    gpu = {(r["model"], r["config"]): r for r in evidence["gpu"]["rows"]}
    for model, short in (("resnet/resnet50", "resnet50"), ("resnet/resnet18", "resnet18")):
        k, n = gpu[(model, "key")], gpu[(model, "nokey")]
        assert k["killed_with_batches_run"] == n["killed_with_batches_run"], "the same kill"
        assert k["updates_executed"] - evidence["gpu"]["steps_asked"] == (
            k["killed_with_batches_run"] - (k["backup_metadata_after_kill"]["epoch"] * evidence["gpu"]["steps_per_epoch"]
                                           + k["backup_metadata_after_kill"]["batch"] + 1)), "the batches lost are the ones after the last copy"
    r50k, r50n = gpu[("resnet/resnet50", "key")], gpu[("resnet/resnet50", "nokey")]
    assert (f"`resnet/resnet50` with a copy every {r50k['backup_every_steps']} (last copy after batch {r50k['backup_metadata_after_kill']['batch'] + 1}) resumed with "
            f"13 + 133 = {r50k['resumed_batches']} batches against {r50n['resumed_batches']} without the key, `fit` took {r50k['fit_wall_seconds']:.1f} s "
            f"against {r50n['fit_wall_seconds']:.1f} s ({r50n['fit_wall_seconds'] - r50k['fit_wall_seconds']:.1f} s saved) and the run executed "
            f"{r50k['updates_executed']} updates against {r50n['updates_executed']}") in flat
    r18k, r18n = gpu[("resnet/resnet18", "key")], gpu[("resnet/resnet18", "nokey")]
    assert (f"`resnet/resnet18` with a copy every {r18k['backup_every_steps']} (last copy after batch {r18k['backup_metadata_after_kill']['batch'] + 1}) resumed with "
            f"{r18k['resumed_batches']} batches against {r18n['resumed_batches']} and `fit` took {r18k['fit_wall_seconds']:.1f} s against "
            f"{r18n['fit_wall_seconds']:.1f} s (no saving)") in flat
    assert r18k["fit_wall_seconds"] >= r18n["fit_wall_seconds"], "the sentence says there is no saving: it must be true"
    # 13 = 133 - 120 (the last copy of the resnet50 run is after batch 120).
    assert evidence["gpu"]["steps_per_epoch"] - (r50k["backup_metadata_after_kill"]["batch"] + 1) == 13


def test_the_readme_states_the_review_reproduction_with_the_numbers_the_children_measured():
    flat = " ".join(README.split())
    assert "killed after 15 of 18 batches with a copy every 5, the resume ran 3 + 18 = 21 batches (36 without the key)" in flat
    assert "54 updates, the counter's 54 (69 without the key)" in flat


def test_the_key_has_a_row_in_the_knob_table_and_the_section_stands_outside_the_knob_region():
    knobs = README[README.index("<!-- knobs:begin -->"):README.index("<!-- knobs:end -->")]
    assert "| `checkpoint_args.backup_every_steps` | `null` |" in knobs
    assert "Backing up inside an epoch" not in knobs and "<!-- backup-cost:begin -->" not in knobs
