"""A run whose training finished and whose reporting died is resumed to REPORT, not retrained (D-041).

The ImageNet gate (plan-2026-10-31T180033-2faf890a, step 21) trained ``cliffordnet/nano`` for
3 x 2,500 steps, then died in the figures' ``predict`` (256 images of 224 px asked 18 GB). The
sidecar said ``fit_finished`` and there was no summary, so the old rule refused the resume
(exit 2) and the only way back was to train again. Now a resume of that run skips ``fit``,
rebuilds the post-``fit`` state from the files and runs the reporting.

The crashed run here is real: ``vit/vit_pico`` on fake CIFAR-10 arrays, early-stopped after four
epochs by a scripted monitor, whose ``evaluate_and_report`` raises a stand-in for the OOM. The
arms copy it, resume the copy and read the files. What they pin:

* the exception is an ``ERROR`` line of ``run.log`` with its type and message, the CLI exits 1;
* the resume exits 0, writes the summary (``reporting_resumed``, ``resumed_after_epochs``, null
  ``epoch_times`` and ``fit_wall_seconds``), the sidecar counts it, ``training_log.csv`` is
  byte-for-byte the training's and no ``pre_resume`` copy exists;
* ``final_model.keras`` holds EXACTLY the weights ``LastEpochWeights`` held when the process
  died, and the model the reporting saw held ``best_model.keras``'s, and a stale or torn
  ``final_model.keras`` is written over;
* a second resume is refused (the summary exists) and changes nothing;
* the last epoch's weights are never guessed: a missing, torn, older or mid-epoch
  ``latest.weights.h5`` is a refusal (exit 2) that names it and changes NOTHING, the ``.bkp``
  copy included;
* a missing ``best_model.keras`` and other data are refusals; ``analysis_args.eval_batch_size``
  alone may differ from the earlier request;
* the sentence a failed reporting ends with is the resume's own answer (D-064): it promises a
  resume only when the resume would run, and says it would be refused, and why, when it would not;
* ``final_model.keras`` holds the last epoch's optimizer state too, as a whole run's does;
* a resume whose analyzer writes nothing says its ``analyzer.path`` is the earlier segment's file;
* an exception of ``fit`` is an ``ERROR`` line of ``run.log`` as well, and a ``SystemExit`` or a
  ``KeyboardInterrupt`` of either is not logged as a failure.

Tests write only under ``tmp_path``.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import keras
import numpy as np
import pytest

from train.api import cli
from train.api.errors import ConfigError
from train.api.run_dir import request_differences
from train.api.tasks.image_classification import ImageClassificationTrainer
from train.common.resume_state import BACKUP_METADATA_NAME, BACKUP_WEIGHTS_NAME, FIT_STATE_NAME

from .test_existing_run_is_refused import snapshot
from .test_resume_hardening import quiet_snapshot
from .test_resume_interrupted import (  # noqa: F401  (fixtures)
    REQUEST,
    killed,
    STOP_AFTER,
    TOKENS,
    Scripted,
    resume_by_cli,
    sha,
    whole,
    world,
)
from .test_task_trainer import make_config, read_json


class ResourceExhausted(RuntimeError):
    """Stands for ``tf.errors.ResourceExhaustedError``: a type of its own to find in the log."""


MESSAGE = "Out of memory while trying to allocate 18085848344 bytes"
REPO_ROOT = Path(__file__).resolve().parents[3]


class ReportingCrash(Scripted):
    """The real task, killed in the reporting after taking note of the weights it holds."""

    seen = None

    def evaluate_and_report(self, config, data, fit):
        type(self).seen = {
            "last": [np.array(w) for w in fit.last_weights.weights],
            "best": [np.array(w) for w in fit.model.get_weights()],
        }
        raise ResourceExhausted(MESSAGE)


@pytest.fixture(scope="module")
def crashed(world, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("crashed") / "run"
    with pytest.raises(ResourceExhausted):
        ReportingCrash(make_config(run_dir, **REQUEST)).run()
    return run_dir, ReportingCrash.seen


@pytest.fixture
def copy_of_crashed(crashed, tmp_path):
    run_dir = tmp_path / "copy"
    shutil.copytree(crashed[0], run_dir)
    return run_dir


def weights_of(path):
    return [np.array(w) for w in keras.models.load_model(path, compile=False).get_weights()]


def equal_lists(a, b):
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.fixture(scope="module")
def resumed(world, crashed, tmp_path_factory):
    """The crashed run resumed once, with a stale ``final_model.keras`` planted in front of it."""
    run_dir = tmp_path_factory.mktemp("resumed") / "run"
    shutil.copytree(crashed[0], run_dir)
    csv_before = sha(run_dir / "training_log.csv")
    (run_dir / "final_model.keras").write_bytes(b"torn by a kill in model.save")
    # `eval_batch_size` is the one key of the request that may differ (a card that cannot take
    # the first run's batch is the reason the reporting died).
    result = Scripted(make_config(
        run_dir, resume_interrupted=True, **REQUEST, **{"analysis_args.eval_batch_size": 8}
    )).run()
    return result, run_dir, csv_before


# ---------------------------------------------------------------------
# The crash: where it is written
# ---------------------------------------------------------------------


def test_the_crashed_run_has_a_finished_fit_and_no_summary(crashed):
    run_dir, seen = crashed
    sidecar = json.loads((run_dir / "checkpoints" / FIT_STATE_NAME).read_text())
    assert sidecar["fit_finished"] is True
    assert sorted(sidecar["epochs"], key=int) == [str(e) for e in range(STOP_AFTER)]
    assert not (run_dir / "results_summary.json").exists()
    assert not (run_dir / "final_model.keras").exists()
    assert (run_dir / "best_model.keras").is_file() and seen is not None


def test_a_reporting_exception_is_an_error_line_of_run_log_with_type_and_message(crashed):
    """Mutant ``no logger.error in the reporting handler``: the OOM was in stderr only."""
    run_dir, _ = crashed
    errors = [line for line in (run_dir / "run.log").read_text().splitlines() if " ERROR " in line]
    assert len(errors) == 1, errors
    assert "ResourceExhausted" in errors[0] and MESSAGE in errors[0]
    assert "resume_interrupted=true" in errors[0]  # the way back is named while it exists


#: A child process: the real ``python -m train`` with a reporting that raises, so the exit code
#: is the interpreter's for an uncaught exception, not a value the test chose.
DRIVER = f'''
import runpy
import sys

sys.path.insert(0, {str(REPO_ROOT)!r})
import train.common.classification_data as classification_data
from tests.classification_fakes import tiny_raw
from train.api.tasks.image_classification import ImageClassificationTrainer

classification_data._load_raw = tiny_raw(n_train=40, n_test=8)


class ResourceExhausted(RuntimeError):
    pass


def explode(self, config, data, fit):
    raise ResourceExhausted({MESSAGE!r})


ImageClassificationTrainer.evaluate_and_report = explode
runpy.run_module("train", run_name="__main__")
'''


def test_the_cli_exits_1_not_2_and_the_error_is_in_run_log_and_stderr(world, tmp_path):
    """Mutant ``the handler returns instead of raising``: exit code 0 would be a finished run."""
    (tmp_path / "cwd").mkdir()
    out = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, "-c", DRIVER, "image_classification", f"out={out}", "epochs=1",
         "model=vit/vit_pico", "data.dataset=cifar10", "batch_size=4",
         "analysis_args.model_analysis=false", "metric_args.success_threshold=0"],
        cwd=tmp_path / "cwd", capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 1, result.stderr[-2000:]
    assert "ResourceExhausted" in result.stderr and MESSAGE in result.stderr
    log = (out / "run.log").read_text()
    assert [line for line in log.splitlines() if " ERROR " in line and "ResourceExhausted" in line]
    assert not (out / "results_summary.json").exists()


def test_no_error_line_says_to_resume_when_a_summary_already_exists(crashed):
    """Control for the hint: it is given only while a resume can work (no summary)."""
    run_dir, _ = crashed
    assert not (run_dir / "results_summary.json").exists()


# ---------------------------------------------------------------------
# The resume
# ---------------------------------------------------------------------


def test_the_resume_writes_the_summary_and_says_it_only_reported(resumed):
    result, run_dir, _ = resumed
    summary = read_json(run_dir, "results_summary.json")
    assert summary["status"] == "ok" and result.summary_path == run_dir / "results_summary.json"
    assert summary["reporting_resumed"] is True
    assert summary["resumed_after_epochs"] == STOP_AFTER
    assert summary["epochs_run"] == STOP_AFTER and summary["stopped_early"] is True
    assert summary["fit_wall_seconds"] is None
    assert summary["epoch_times"] == [None] * STOP_AFTER
    assert any("reporting ran again from the files" in note for note in summary["notes"])
    assert read_json(run_dir, "checkpoints/" + FIT_STATE_NAME)["reporting_resumes"] == 1


def test_training_log_csv_is_byte_for_byte_the_trainings_and_no_copy_is_made(resumed):
    """Mutant ``copy_log_aside runs`` / ``the CSV logger is built``: the CSV is the training's."""
    _, run_dir, csv_before = resumed
    assert sha(run_dir / "training_log.csv") == csv_before
    assert not list(run_dir.glob("training_log.pre_resume_*"))


def test_the_summary_has_every_key_of_a_whole_runs_and_two_more(whole, resumed):
    whole_summary = read_json(whole[1], "results_summary.json")
    summary = read_json(resumed[1], "results_summary.json")
    assert set(summary) - set(whole_summary) == {"resumed_after_epochs", "reporting_resumed"}
    assert set(whole_summary) - set(summary) == set()
    assert summary["best_epoch"] == whole_summary["best_epoch"]
    assert summary["best_val_metrics"] == whole_summary["best_val_metrics"]
    assert summary["final_val_metrics"] == whole_summary["final_val_metrics"]


def test_final_model_is_exactly_the_last_epochs_weights_and_a_torn_file_is_written_over(
    crashed, resumed
):
    """Mutant ``best weights saved as final`` and ``the stale file is kept``: the planted
    ``final_model.keras`` was not a model at all, and the weights are LastEpochWeights'."""
    _, run_dir, _ = resumed
    _, seen = crashed
    final = weights_of(run_dir / "final_model.keras")
    assert equal_lists(final, seen["last"])
    assert not equal_lists(final, seen["best"])  # liveness: last and best differ in this run
    assert read_json(run_dir, "results_summary.json")["model_loading_validated"] is True
    assert not list(run_dir.glob("final_model.partial.keras"))


def test_the_model_the_reporting_saw_held_best_model_keras_exactly(crashed, resumed):
    _, run_dir, _ = resumed
    summary = read_json(run_dir, "results_summary.json")
    assert summary["best_checkpoint_max_abs_diff"] == 0.0
    assert equal_lists(weights_of(run_dir / "best_model.keras"), crashed[1]["best"])
    assert summary["test_metrics_best"] is not None and summary["test_metrics_final"] is not None


def test_the_figures_are_written_by_the_resume(resumed):
    _, run_dir, _ = resumed
    assert (run_dir / "visualizations" / "training_dashboard.png").is_file()
    assert read_json(run_dir, "results_summary.json")["visualizations"]["files"]


def test_a_second_resume_is_refused_with_exit_2_and_changes_nothing(resumed, capsys):
    _, run_dir, _ = resumed
    before = snapshot(run_dir)
    assert resume_by_cli(run_dir) == 2
    assert "results_summary.json" in capsys.readouterr().err
    assert snapshot(run_dir) == before


# ---------------------------------------------------------------------
# The last epoch's weights are never guessed
# ---------------------------------------------------------------------


def backup_of(run_dir):
    return run_dir / "checkpoints" / "backup"


def refused_and_untouched(run_dir, capsys):
    """Resume by the CLI, expect exit 2, and return the error text. ``run.lock`` and ``run.log``
    are left out of the comparison: a refusal that got past the lock labels the one and appends
    to the other (``quiet_snapshot``)."""
    before = quiet_snapshot(run_dir)
    assert resume_by_cli(run_dir, epochs=6) == 2
    error = capsys.readouterr().err
    assert quiet_snapshot(run_dir) == before, "a refusal must not change the run"
    return error


def test_the_control_arm_resumes_to_exit_0(world, copy_of_crashed):
    """Vacuity guard: every refusal below is that one run with ONE file damaged."""
    assert resume_by_cli(copy_of_crashed, epochs=6) == 0
    assert (copy_of_crashed / "results_summary.json").is_file()


def test_a_missing_latest_weights_file_is_refused_naming_it(world, copy_of_crashed, capsys):
    (backup_of(copy_of_crashed) / BACKUP_WEIGHTS_NAME).unlink()
    error = refused_and_untouched(copy_of_crashed, capsys)
    assert BACKUP_WEIGHTS_NAME in error and "does not exist" in error
    assert "best_model.keras is untouched" in error and "last epoch's weights" in error


def test_a_torn_latest_weights_file_is_refused_and_the_previous_copy_is_not_put_in_its_place(
    world, copy_of_crashed, capsys
):
    """Mutant ``settle_backup runs for a finished fit``: it would copy the .bkp pair over the
    latest, and the 'final model' would be an earlier epoch's."""
    backup = backup_of(copy_of_crashed)
    (backup / BACKUP_WEIGHTS_NAME).write_bytes((backup / BACKUP_WEIGHTS_NAME).read_bytes()[:200])
    previous = sha(backup / (BACKUP_WEIGHTS_NAME + ".bkp"))
    error = refused_and_untouched(copy_of_crashed, capsys)
    assert BACKUP_WEIGHTS_NAME in error and "cannot be read" in error
    assert sha(backup / (BACKUP_WEIGHTS_NAME + ".bkp")) == previous
    assert not (copy_of_crashed / "final_model.keras").exists()


def test_an_older_epochs_metadata_over_the_last_weights_is_refused(world, copy_of_crashed, capsys):
    """The weights and the metadata are from different epochs: neither is trusted."""
    metadata = backup_of(copy_of_crashed) / BACKUP_METADATA_NAME
    metadata.write_text(json.dumps({"epoch": STOP_AFTER - 1, "batch": 0}))
    # The sidecar also records STOP_AFTER epochs, so the files disagree where training ended.
    error = refused_and_untouched(copy_of_crashed, capsys)
    assert "do not agree on where the training ended" in error


def test_a_mid_epoch_copy_is_not_the_last_epochs_weights(world, copy_of_crashed, capsys):
    metadata = backup_of(copy_of_crashed) / BACKUP_METADATA_NAME
    metadata.write_text(json.dumps({"epoch": STOP_AFTER, "batch": 3}))
    error = refused_and_untouched(copy_of_crashed, capsys)
    assert BACKUP_WEIGHTS_NAME in error and "different epochs" in error


def test_a_sidecar_that_lost_an_epoch_is_refused(world, copy_of_crashed, capsys):
    path = copy_of_crashed / "checkpoints" / FIT_STATE_NAME
    state = json.loads(path.read_text())
    del state["epochs"]["1"]
    path.write_text(json.dumps(state))
    error = refused_and_untouched(copy_of_crashed, capsys)
    assert "number of epochs that ran is unknown" in error


def test_a_missing_best_checkpoint_is_refused(world, copy_of_crashed, capsys):
    (copy_of_crashed / "best_model.keras").unlink()
    error = refused_and_untouched(copy_of_crashed, capsys)
    assert "best_model.keras" in error and "does not exist" in error


def test_other_data_than_the_runs_is_refused_before_anything_is_written(
    world, copy_of_crashed, capsys
):
    task = read_json(copy_of_crashed, "task.json")
    task["class_names"] = list(reversed(task["class_names"]))
    (copy_of_crashed / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    error = refused_and_untouched(copy_of_crashed, capsys)
    assert "the data is not the data the run started with" in error


def test_only_the_evaluation_batch_may_differ_from_the_earlier_request(world, copy_of_crashed, capsys):
    """Mutant ``every analysis_args key is uncompared``: a changed ``figures`` is a different run."""
    earlier = read_json(copy_of_crashed, "config_requested.json")
    now = make_config(copy_of_crashed, resume_interrupted=True, **REQUEST,
                      **{"analysis_args.eval_batch_size": 8, "analysis_args.figures": False})
    assert request_differences(earlier, now) == [
        "analysis_args.figures: True -> False",
    ]
    assert request_differences(earlier, make_config(
        copy_of_crashed, resume_interrupted=True, **REQUEST,
        **{"analysis_args.eval_batch_size": 8})) == []
    before = quiet_snapshot(copy_of_crashed)
    with pytest.raises(ConfigError, match="analysis_args.figures: True -> False"):
        Scripted(now).run()
    assert quiet_snapshot(copy_of_crashed) == before


def test_the_tokens_of_the_cli_helper_are_the_requests_of_this_file():
    """Guard for ``resume_by_cli``: its tokens must describe the same run as ``REQUEST``."""
    assert "model=vit/vit_pico" in TOKENS and "analysis_args.model_analysis=false" in TOKENS


# ---------------------------------------------------------------------
# The sentence a failed reporting ends with (review-iter-2 concern 4, D-064)
# ---------------------------------------------------------------------


def error_lines(run_dir):
    return [line for line in (run_dir / "run.log").read_text().splitlines() if " ERROR " in line]


class DamagedThenCrash(Scripted):
    """The real task whose reporting damages ``latest.weights.h5`` (or not) and then dies."""

    damage = None

    def evaluate_and_report(self, config, data, fit):
        if self.damage == "unlink":
            (self.run_dir_of(config) / "checkpoints" / "backup" / BACKUP_WEIGHTS_NAME).unlink()
        raise ResourceExhausted(MESSAGE)

    @staticmethod
    def run_dir_of(config):
        return Path(config.out)


def test_the_hint_of_a_run_without_a_best_checkpoint_does_not_promise_a_resume(world, tmp_path):
    """Mutant ``rep_hint_always``: with ``save_best`` false the old line promised a resume that exited 2."""
    run_dir = tmp_path / "run"
    config = make_config(run_dir, **REQUEST, **{"checkpoint_args.save_best": False})
    with pytest.raises(ResourceExhausted):
        DamagedThenCrash(config).run()
    errors = error_lines(run_dir)
    assert len(errors) == 1 and "Reporting failed: ResourceExhausted" in errors[0]
    assert "would be refused" in errors[0] and "save_best is false" in errors[0]
    assert "run the same request with resume_interrupted=true" not in errors[0]
    # ... and the resume does refuse, for the reason the line gave.
    with pytest.raises(ConfigError, match="save_best is false"):
        Scripted(make_config(run_dir, resume_interrupted=True, **REQUEST,
                             **{"checkpoint_args.save_best": False})).run()


def test_the_hint_of_a_run_whose_last_epoch_backup_is_gone_says_the_resume_is_refused_and_names_the_file(
    world, tmp_path
):
    run_dir = tmp_path / "run"

    class Unlinks(DamagedThenCrash):
        damage = "unlink"

    with pytest.raises(ResourceExhausted):
        Unlinks(make_config(run_dir, **REQUEST)).run()
    (line,) = error_lines(run_dir)
    assert "would be refused" in line and BACKUP_WEIGHTS_NAME in line
    assert "run the same request with resume_interrupted=true" not in line
    with pytest.raises(ConfigError, match=BACKUP_WEIGHTS_NAME):
        Scripted(make_config(run_dir, resume_interrupted=True, **REQUEST)).run()


def test_the_hint_of_a_run_that_can_be_resumed_promises_it_and_the_resume_works(crashed, copy_of_crashed):
    """Control for both: the same sentence, and it is true."""
    (line,) = error_lines(crashed[0])
    assert "run the same request with resume_interrupted=true to run only the reporting again" in line
    assert "would be refused" not in line
    assert resume_by_cli(copy_of_crashed, epochs=6) == 0


# ---------------------------------------------------------------------
# SystemExit and KeyboardInterrupt are not failures; an exception of fit is logged too (concern 18)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("leaving", [SystemExit, KeyboardInterrupt])
def test_a_system_exit_in_the_reporting_is_not_logged_as_a_failure(world, tmp_path, leaving):
    """Mutant ``rep_logs_systemexit``: the context logged any exception, ``SystemExit`` included."""

    class Leaves(Scripted):
        def evaluate_and_report(self, config, data, fit):
            raise leaving(3)

    run_dir = tmp_path / "run"
    with pytest.raises(leaving):
        Leaves(make_config(run_dir, **REQUEST)).run()
    assert error_lines(run_dir) == []


@pytest.mark.parametrize("leaving", [SystemExit, KeyboardInterrupt])
def test_a_system_exit_in_fit_is_not_logged_as_a_failure(world, tmp_path, leaving):
    class Leaves(Scripted):
        def extra_callbacks(self, config, data):
            class Raises(keras.callbacks.Callback):
                def on_epoch_begin(self, epoch, logs=None):
                    raise leaving(3)

            return [Raises()]

    run_dir = tmp_path / "run"
    with pytest.raises(leaving):
        Leaves(make_config(run_dir, **REQUEST)).run()
    assert error_lines(run_dir) == []


def test_an_exception_of_fit_is_an_error_line_of_run_log_with_its_type_and_message(world, killed):
    """The review's reproduction: an exception in ``fit`` (there, ``on_train_end``) was in stderr only,
    0 lines of ``run.log`` against 2 of stderr. No hint: whether a resume works is another question."""
    (line,) = error_lines(killed)
    assert "Training failed: RuntimeError: killed" in line
    assert "resume_interrupted" not in line


# ---------------------------------------------------------------------
# What the resume writes that a whole run writes (concerns 13 and 14)
# ---------------------------------------------------------------------


def optimizer_variables(path):
    """``{name: array}`` of the optimizer's variables in a ``.keras`` archive or a ``.weights.h5`` file."""
    import io
    import zipfile

    import h5py

    path = Path(path)
    source = io.BytesIO(zipfile.ZipFile(path).read("model.weights.h5")) if path.suffix == ".keras" else path
    found = {}
    with h5py.File(source, "r") as handle:
        def visit(name, node):
            if name.startswith("optimizer") and isinstance(node, h5py.Dataset):
                found[name] = node[()]

        handle.visititems(visit)
    return found


def test_final_model_keras_of_a_reporting_resume_has_the_optimizer_state_of_the_last_epoch(crashed, resumed, whole):
    """Review-iter-2 concern 13: 1 optimizer variable and step count 0 against a whole run's 161 and 36."""
    _, run_dir, _ = resumed
    final = optimizer_variables(run_dir / "final_model.keras")
    backup = optimizer_variables(run_dir / "checkpoints" / "backup" / BACKUP_WEIGHTS_NAME)
    whole_final = optimizer_variables(whole[1] / "final_model.keras")
    assert len(final) > 1, "only the step count was saved: the optimizer was not built before the load"
    assert sorted(final) == sorted(whole_final) == sorted(backup)
    for name in final:
        assert np.array_equal(final[name], backup[name]), name
    steps = int(final["optimizer/vars/0"])
    assert steps > 0 and steps == int(whole_final["optimizer/vars/0"])


@pytest.fixture(scope="module")
def crashed_after_the_analysis(world, tmp_path_factory):
    """A run whose analyzer ran and wrote ``analysis_results.json`` before the reporting died."""

    class DiesAfter(Scripted):
        def evaluate_and_report(self, config, data, fit):
            super().evaluate_and_report(config, data, fit)
            raise ResourceExhausted(MESSAGE)

    run_dir = tmp_path_factory.mktemp("after_analysis") / "run"
    with pytest.raises(ResourceExhausted):
        DiesAfter(make_config(run_dir, **REQUEST, **{"analysis_args.model_analysis": True})).run()
    assert (run_dir / "model_analysis" / "analysis_results.json").is_file()
    return run_dir


def resume_with_analysis(crashed_run, tmp_path, analyzer_raises):
    run_dir = tmp_path / "run"
    shutil.copytree(crashed_run, run_dir)
    config = make_config(run_dir, resume_interrupted=True, **REQUEST, **{"analysis_args.model_analysis": True})
    with pytest.MonkeyPatch.context() as patch:
        if analyzer_raises:
            from train.common import evaluation

            def refuse(self, *arguments, **keywords):
                raise RuntimeError("injected analyzer failure")

            patch.setattr(evaluation.ModelAnalyzer, "analyze", refuse)
        Scripted(config).run()
    return read_json(run_dir, "results_summary.json")


def test_a_failed_analyzer_in_a_reporting_resume_says_its_path_is_the_earlier_segments(
    crashed_after_the_analysis, tmp_path
):
    """Review-iter-2 concern 14: ``status: partial`` and a ``path`` written before the resume began,
    with nothing saying it was older."""
    summary = resume_with_analysis(crashed_after_the_analysis, tmp_path, analyzer_raises=True)
    assert summary["analyzer"]["status"] in ("partial", "missing") or summary["analyzer"].get("failed_analyses")
    (note,) = [n for n in summary["notes"] if "which this reporting did not write" in n]
    assert summary["analyzer"]["path"] in note and "earlier segment" in note


def test_a_reporting_resume_whose_analyzer_wrote_its_file_has_no_such_note(crashed_after_the_analysis, tmp_path):
    summary = resume_with_analysis(crashed_after_the_analysis, tmp_path, analyzer_raises=False)
    assert summary["analyzer"]["status"] == "success"
    assert not [n for n in summary["notes"] if "which this reporting did not write" in n]
