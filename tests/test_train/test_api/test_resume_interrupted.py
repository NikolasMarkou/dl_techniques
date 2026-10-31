"""``resume_interrupted=true``: a killed run continues with its rows, best and history intact.

Keras 3.8 ``BackupAndRestore`` alone truncates ``training_log.csv``, forgets the
best value so a worse epoch overwrites ``best_model.keras``, keeps a partial
history and trains an early-stopped run again (plan-2026-10-30T132432-a865fa08,
findings section C). ``train.common.resume_state`` holds the primitives
(``test_common_resume_state.py``); this file tests the TRAINER that composes
them, on real fits of ``vit/vit_pico`` on fake CIFAR-10 arrays.

The scripted monitor (``val_accuracy`` ``0.9, 0.2, 0.2, ...``, patience 3) makes
epoch 0 the best one and ``EarlyStopping`` stop after the fourth epoch, so a
resume that forgets the best value, the wait or the best weights is visible in
the files. Three fits are shared by the arms:

* **whole**: the run, uninterrupted (4 epochs);
* **killed**: the same request, a callback raises at the start of epoch index 2
  (a real SIGKILL is Step 30's);
* **resumed**: a COPY of ``killed`` resumed by a fresh ``TaskTrainer``, with a row
  planted in the CSV for epoch 2 (the kill that falls between a CSV row and the
  backup). The copy sits at another path, which also proves ``out`` is not part
  of the comparison.

Everything else is a refusal read off a planted run tree: nothing trains, and the
tree must be byte-identical afterwards. The tests write only under ``tmp_path``.
"""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import keras
import pytest

import train.api.recipes as recipes
from train.api import cli, registry
from train.api.errors import ConfigError
from train.api.run_dir import request_differences, write_requested
from train.api.tasks import register_builtin_tasks
from train.api.tasks.image_classification import ImageClassificationTrainer
from train.common.resume_state import FIT_STATE_NAME, update_fit_state

from tests.classification_fakes import use_tiny_raw

from .test_existing_run_is_refused import snapshot
from .test_task_trainer import files_of, make_config, read_json

EPOCHS = 6
PATIENCE = 3
#: The scripted ``val_accuracy`` of each epoch: the first is the best, so the run stops
#: after the fourth epoch (the best epoch, then three that are worse).
SCRIPT = (0.9, 0.2, 0.2, 0.2, 0.2, 0.2)
STOP_AFTER = 4
KILL_AT = 2

#: Fields of the request every run in this file shares.
REQUEST = {
    "epochs": EPOCHS,
    "metric_args.success_threshold": 0.0,
    "checkpoint_args.early_stopping_patience": PATIENCE,
}


class ScriptedMonitor(keras.callbacks.Callback):
    """Write the scripted ``val_accuracy`` into the epoch logs; sits before EarlyStopping."""

    def on_epoch_end(self, epoch, logs=None):
        logs["val_accuracy"] = SCRIPT[epoch]


class KillAt(keras.callbacks.Callback):
    """Raise at the start of epoch ``epoch``: a stand-in for a kill between two epochs."""

    def __init__(self, epoch):
        super().__init__()
        self.epoch = epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.epoch:
            raise RuntimeError("killed")


class Scripted(ImageClassificationTrainer):
    """The real task with a scripted monitor, optionally killed before ``kill_at``."""

    kill_at = None

    def _callbacks(self, *arguments):
        epoch_line, callbacks = super()._callbacks(*arguments)
        # Index 1: after the learning-rate logger, before EarlyStopping and ModelCheckpoint.
        callbacks.insert(1, ScriptedMonitor())
        return epoch_line, callbacks

    def extra_callbacks(self, config, data):
        return [] if self.kill_at is None else [KillAt(self.kill_at)]


class Killed(Scripted):
    kill_at = KILL_AT


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        use_tiny_raw(patch, n_train=40, n_test=8)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(run_dir):
    with open(run_dir / "training_log.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows


@pytest.fixture(scope="module")
def whole(world, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("whole") / "run"
    result = Scripted(make_config(run_dir, **REQUEST)).run()
    return result, run_dir


@pytest.fixture(scope="module")
def killed(world, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("killed") / "run"
    with pytest.raises(RuntimeError, match="killed"):
        Killed(make_config(run_dir, **REQUEST)).run()
    return run_dir


#: Written over the copy's task.json before the resume: a resume that rewrote the file would erase it.
PLANTED_TASK_JSON = '{"planted": "by the test"}\n'


@pytest.fixture(scope="module")
def resumed(world, killed, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("resumed") / "copy"
    shutil.copytree(killed, run_dir)
    (run_dir / "task.json").write_text(PLANTED_TASK_JSON)
    with open(run_dir / "training_log.csv", newline="") as handle:
        header = next(csv.reader(handle))
    # The kill that fell between a CSV row and the backup: epoch 2's row exists, its backup does not.
    with open(run_dir / "training_log.csv", "a", newline="") as handle:
        csv.writer(handle).writerow(["2"] + ["99.0"] * (len(header) - 1))
    result = Scripted(make_config(run_dir, resume_interrupted=True, **REQUEST)).run()
    return result, run_dir


# ---------------------------------------------------------------------
# The killed run, before any resume
# ---------------------------------------------------------------------


def test_the_killed_run_has_two_epochs_a_backup_and_an_unfinished_sidecar(killed):
    assert [row["epoch"] for row in read_csv(killed)] == ["0", "1"]
    assert (killed / "checkpoints" / "backup").is_dir()
    sidecar = json.loads((killed / "checkpoints" / FIT_STATE_NAME).read_text())
    assert sidecar["fit_finished"] is False
    assert sidecar["epochs"]["1"] == {"best": 0.9, "wait": 1, "best_epoch": 0}
    assert sidecar["initial"]["metrics"]["loss"] > 0
    assert not (killed / "results_summary.json").exists()


def test_the_comparison_sees_no_difference_between_a_request_and_its_own_file(killed):
    """Vacuity guard for the refusal arms below: the same request differs in nothing."""
    assert request_differences(read_json(killed, "config_requested.json"), make_config(
        killed, **REQUEST)) == []


# ---------------------------------------------------------------------
# The resumed run
# ---------------------------------------------------------------------


def test_the_csv_holds_each_epoch_once_and_the_planted_row_is_retrained(whole, resumed):
    _, whole_dir = whole
    _, run_dir = resumed
    rows = read_csv(run_dir)
    assert [row["epoch"] for row in rows] == ["0", "1", "2", "3"]
    assert [row["epoch"] for row in read_csv(whole_dir)] == ["0", "1", "2", "3"]
    assert [float(row["val_accuracy"]) for row in rows] == list(SCRIPT[:STOP_AFTER])
    assert all(float(row["loss"]) != 99.0 for row in rows)


def test_the_csv_as_it_was_is_kept_beside_it(resumed):
    _, run_dir = resumed
    kept = run_dir / "training_log.pre_resume_1.csv"
    assert [row["epoch"] for row in csv.DictReader(open(kept, newline=""))] == ["0", "1", "2"]
    assert "99.0" in kept.read_text()


def test_the_best_checkpoint_is_not_overwritten_by_a_worse_epoch(killed, resumed):
    _, run_dir = resumed
    assert sha(run_dir / "best_model.keras") == sha(killed / "best_model.keras")


def test_early_stopping_stops_after_the_same_epoch_as_the_uninterrupted_run(whole, resumed):
    _, whole_dir = whole
    _, run_dir = resumed
    summary = read_json(run_dir, "results_summary.json")
    assert summary["epochs_run"] == read_json(whole_dir, "results_summary.json")["epochs_run"] == STOP_AFTER
    assert summary["stopped_early"] is True
    assert summary["best_epoch"] == 1 and summary["resumed_after_epochs"] == KILL_AT


def test_the_in_memory_best_weights_are_the_checkpoints(resumed):
    """The resumed run never beat epoch 0, so its best weights came from ``best_model.keras``."""
    _, run_dir = resumed
    summary = read_json(run_dir, "results_summary.json")
    assert summary["best_checkpoint_load_error"] is None
    assert summary["best_checkpoint_max_abs_diff"] < 1e-5


def test_the_history_file_covers_every_epoch(whole, resumed):
    _, whole_dir = whole
    _, run_dir = resumed
    history = read_json(run_dir, "training_history.json")
    assert history["val_accuracy"] == read_json(whole_dir, "training_history.json")["val_accuracy"]
    assert all(len(values) == STOP_AFTER for values in history.values())


def test_the_learning_rate_at_the_resumed_epochs_is_the_uninterrupted_runs(whole, resumed):
    _, whole_dir = whole
    _, run_dir = resumed
    expected = [float(row["lr"]) for row in read_csv(whole_dir)]
    assert [float(row["lr"]) for row in read_csv(run_dir)] == pytest.approx(expected, rel=1e-6)
    assert len(set(expected)) == STOP_AFTER  # liveness: the schedule moves every epoch


def test_the_run_log_keeps_its_earlier_lines_and_continues(killed, resumed):
    _, run_dir = resumed
    log = (run_dir / "run.log").read_text()
    assert log.startswith((killed / "run.log").read_text())
    assert log.count("Untrained evaluate BEFORE fit") == 1  # the baseline is not measured again
    assert log.count("Run directory:") == 2
    assert log.index("Untrained evaluate") < log.index("Resuming after epoch 2 of 6")


def test_the_two_configs_are_not_written_again(killed, resumed):
    _, run_dir = resumed
    for name in ("config_requested.json", "config.json"):
        assert sha(run_dir / name) == sha(killed / name)


def test_a_killed_run_already_has_its_task_json_and_a_resume_leaves_it_alone(killed, resumed):
    from train.api.predictor import read_record

    assert read_record(killed / "task.json")["class_names"]  # written before `fit`, so a killed run is usable
    _, run_dir = resumed
    assert (run_dir / "task.json").read_text() == PLANTED_TASK_JSON


def test_the_baseline_comes_from_the_sidecar(killed, resumed):
    _, run_dir = resumed
    sidecar = json.loads((killed / "checkpoints" / FIT_STATE_NAME).read_text())
    summary = read_json(run_dir, "results_summary.json")
    assert summary["initial_loss_sanity_eval"]["loss"] == sidecar["initial"]["metrics"]["loss"]
    assert summary["initial_loss_ratio"] == sidecar["initial"]["ratio"]


def test_the_resumed_run_leaves_a_finished_sidecar_and_the_usual_files(resumed):
    _, run_dir = resumed
    assert json.loads((run_dir / "checkpoints" / FIT_STATE_NAME).read_text())["fit_finished"] is True
    found = files_of(run_dir)
    assert {"best_model.keras", "final_model.keras", "results_summary.json",
            "visualizations/training_dashboard.png", "training_log.pre_resume_1.csv"} <= found
    assert any(name.startswith("checkpoints/backup/") for name in found)


# ---------------------------------------------------------------------
# A run that cannot be resumed is refused, and nothing in it changes
# ---------------------------------------------------------------------

TOKENS = (
    "image_classification",
    "model=vit/vit_pico",
    "data.dataset=cifar10",
    "batch_size=4",
    "analysis_args.model_analysis=false",
    "metric_args.success_threshold=0",
    f"checkpoint_args.early_stopping_patience={PATIENCE}",
    "resume_interrupted=true",
)


def resume_by_cli(run_dir, epochs=EPOCHS):
    return cli.main([TOKENS[0], f"out={run_dir}", f"epochs={epochs}", *TOKENS[1:]])


def test_a_finished_run_is_refused_with_exit_2_and_left_alone(world, whole, tmp_path, capsys):
    _, whole_dir = whole
    run_dir = tmp_path / "finished"
    shutil.copytree(whole_dir, run_dir)
    before = snapshot(run_dir)
    assert resume_by_cli(run_dir) == 2
    assert "results_summary.json" in capsys.readouterr().err
    assert snapshot(run_dir) == before


def test_a_run_whose_fit_returned_is_refused_though_it_has_no_summary(world, whole, tmp_path, capsys):
    """An early-stopped run that crashed while reporting: only ``fit_finished`` says so (D-021)."""
    _, whole_dir = whole
    run_dir = tmp_path / "crashed_after_fit"
    shutil.copytree(whole_dir, run_dir)
    (run_dir / "results_summary.json").unlink()
    before = snapshot(run_dir)
    assert resume_by_cli(run_dir) == 2
    assert "fit_finished" in capsys.readouterr().err
    assert snapshot(run_dir) == before


def test_a_changed_request_is_refused_naming_the_key(world, killed, tmp_path, capsys):
    run_dir = tmp_path / "copy"
    shutil.copytree(killed, run_dir)
    before = snapshot(run_dir)
    assert resume_by_cli(run_dir, epochs=EPOCHS + 1) == 2
    error = capsys.readouterr().err
    assert "in epochs: 6 -> 7. Resume with the request" in error  # that key and no other
    with pytest.raises(ConfigError, match="optim_args.lr: 'auto' -> 0.01"):
        Scripted(make_config(run_dir, resume_interrupted=True, **REQUEST,
                             **{"optim_args.lr": 0.01})).run()
    assert snapshot(run_dir) == before


def test_resume_ignores_out_and_the_flag_and_reads_json_normal_form():
    earlier = {"out": "x", "resume_interrupted": False, "model_args": {"sizes": [1, 2]}, "epochs": 3}
    same = {"out": "elsewhere", "resume_interrupted": True, "model_args": {"sizes": (1, 2)}, "epochs": 3}
    assert request_differences(earlier, same) == []
    changed = {**same, "model_args": {"sizes": (1, 3), "extra": 1}, "epochs": 3.0}
    assert request_differences(earlier, changed) == [
        "epochs: 3 -> 3.0", "model_args.extra: <absent> -> 1", "model_args.sizes: [1, 2] -> [1, 3]"]


def plant_run(run_dir, config, *, start=2, wait=1, best=0.9, rows=None, best_model=True,
              sidecar_entries=True, request="written"):
    """Lay out the files of a run killed after ``start`` epochs, without training."""
    run_dir.mkdir(parents=True)
    if request == "written":
        write_requested(run_dir, config)
    else:
        (run_dir / "config_requested.json").write_text(request)
    backup = run_dir / "checkpoints" / "backup"
    backup.mkdir(parents=True)
    (backup / "training_metadata.json").write_text(json.dumps({"epoch": start}))
    entries = {str(e): {"best": best, "wait": wait, "best_epoch": 0} for e in range(start)}
    update_fit_state(
        run_dir / "checkpoints" / FIT_STATE_NAME,
        epochs=entries if sidecar_entries else {},
        initial={"metrics": {"loss": 2.3, "accuracy": 0.1}, "ratio": 1.0, "warned": False},
    )
    count = start if rows is None else rows
    with open(run_dir / "training_log.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "loss", "val_accuracy"])
        for epoch in range(count):
            writer.writerow([epoch, 1.0, 0.5])
    if best_model:
        (run_dir / "best_model.keras").write_bytes(b"x")
    return run_dir


@pytest.mark.parametrize(
    "overrides, plant, message",
    [
        ({"checkpoint_args.save_best": False}, {}, "save_best is false"),
        ({"epochs": 2}, {"start": 2}, "all 2 of its 2 epochs"),
        ({"checkpoint_args.early_stopping_patience": 1}, {"start": 2, "wait": 1},
         "had already decided to stop"),
        ({}, {"best_model": False}, "best_model.keras does not exist"),
        ({}, {"sidecar_entries": False}, "holds no entry for epoch 1"),
        ({}, {"rows": 1}, "holds 1 complete epoch rows"),
        ({}, {"request": ""}, "is empty, so the run died before it wrote its request"),
        ({}, {"request": "{not json"}, "not valid JSON"),
    ],
    ids=["save_best_off", "no_epoch_left", "stop_already_decided", "best_model_missing",
         "sidecar_without_entry", "csv_short", "empty_request", "request_not_json"],
)
def test_a_run_that_cannot_be_continued_is_refused_and_left_alone(
        world, tmp_path, overrides, plant, message):
    config = make_config(tmp_path / "run", resume_interrupted=True, **{**REQUEST, **overrides})
    run_dir = plant_run(tmp_path / "run", config, **plant)
    before = snapshot(run_dir)
    with pytest.raises(ConfigError, match=message):
        ImageClassificationTrainer(config).run()
    assert snapshot(run_dir) == before


def test_resuming_a_missing_run_creates_nothing(world, tmp_path):
    out = tmp_path / "missing"
    with pytest.raises(ConfigError, match="checkpoints/backup"):
        ImageClassificationTrainer(make_config(out, resume_interrupted=True)).run()
    assert not out.exists()
