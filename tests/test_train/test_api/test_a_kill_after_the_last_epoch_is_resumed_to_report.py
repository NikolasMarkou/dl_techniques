"""A kill after the last epoch ended and before ``fit`` returned is resumed to REPORT, not refused (D-064).

Review-iter-2 concern 2. The window between the end of the last epoch and ``fit_finished`` in the
sidecar holds the dashboard's last draw and the restore of EarlyStopping's best weights (1.6 s in
the parity log). A process killed there had trained everything, and the resume answered "all N of
its N epochs were trained and the run died after the last one; a run that crashed while reporting
is not finished by a resume (D-021)" (exit 2): the days of training were lost to a refusal, and
the message told a user whose reporting had never started that reporting had crashed.

The kills here are real: ``_backup_child.py`` runs the API on the CPU and blocks in the
``on_train_end`` of the LAST callback (``stop_at_train_end``), after every other callback has
ended its training and before ``fit`` returns, and this test SIGKILLs that recorded PID. Two runs
end that way:

* **all_epochs**: 2 of 2 epochs trained;
* **early_stopping**: a scripted monitor improves once and then waits out ``patience`` 2, so
  EarlyStopping decided to stop after epoch 3 of 6 (``stop_training`` was set in the epoch end
  that the kill outran).

For each, the resume must run only the reporting, from the files, and write what the unkilled run
(the same request in one process) writes: the same best epoch, the same test metrics for the best
and for the last epoch's weights, the same model and optimizer variables in ``final_model.keras``,
and a ``training_log.csv`` that is byte-for-byte the training's. Where the last epoch's weights
cannot be proven (a torn ``latest.weights.h5``; a backup that records another epoch than the
sidecar) the resume is refused with a message that names the file or the disagreement and says the
training is complete, not that a reporting crashed.

Tests write only under ``tmp_path``.
"""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from train.common.resume_state import BACKUP_METADATA_NAME, BACKUP_WEIGHTS_NAME, FIT_STATE_NAME

from .test_backup_every_steps import final_state, run_child

BASE = {"every": None, "n_train": 40}
SCENARIOS = {
    "all_epochs": {**BASE, "epochs": 2},
    "early_stopping": {**BASE, "epochs": 6, "script": [0.9, 0.2, 0.2, 0.2, 0.2, 0.2], "patience": 2},
}
#: The epochs that trained in each scenario.
EPOCHS_RAN = {"all_epochs": 2, "early_stopping": 3}
#: What the resume's log line and note say about why there was no epoch left.
REASON = {"all_epochs": "all 2 of its epochs ended", "early_stopping": "EarlyStopping had waited 2 epochs"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


class Scenario:
    """The unkilled run, the killed run, and a copy of the killed run that was resumed."""

    def __init__(self, root, name):
        self.root, self.name, self.spec = Path(root), name, SCENARIOS[name]
        self.whole = self.root / "whole"
        self.killed = self.root / "killed"
        self.resumed = self.root / "resumed"

    def go(self):
        self.child(self.whole, "whole", "@@DONE")
        self.child(self.killed, "killed", "@@AT_TRAIN_END", stop_at_train_end=True)
        # What a resume starts from, kept for the refusal arms, taken before anything resumes it.
        shutil.copytree(self.killed, self.root / "killed_untouched")
        self.csv_of_the_kill = sha(self.killed / "training_log.csv")
        shutil.copytree(self.killed, self.resumed)
        self.child(self.resumed, "resumed", "@@DONE", resume=True)
        return self

    def child(self, out, label, expect, **spec):
        marker = run_child({**self.spec, "out": str(out), **spec}, self.root / f"{label}.log")
        assert marker == expect, f"{self.name}/{label}: {marker}"


@pytest.fixture(scope="module", params=sorted(SCENARIOS))
def scenario(request, tmp_path_factory):
    return Scenario(tmp_path_factory.mktemp(request.param), request.param).go()


def refusal(scenario, tmp_path, damage):
    """Resume a damaged copy of the killed run; return the child's log text (the child exits 1)."""
    run = tmp_path / "run"
    shutil.copytree(scenario.root / "killed_untouched", run)
    damage(run)
    before = {p.name: sha(p) for p in run.iterdir() if p.is_file()}
    log = tmp_path / "refusal.log"
    marker = run_child({**scenario.spec, "out": str(run), "resume": True}, log)
    assert marker.startswith("None (exit 1)") or "(exit 1)" in marker, marker
    for name, digest in before.items():
        if name not in ("run.log", "run.lock"):
            assert sha(run / name) == digest, f"a refusal changed {name}"
    return log.read_text(errors="replace")


# ---------------------------------------------------------------------
# The kill, and the state it leaves
# ---------------------------------------------------------------------


def test_the_kill_leaves_every_epoch_trained_and_no_fit_finished(scenario):
    run = scenario.killed
    ran = EPOCHS_RAN[scenario.name]
    sidecar = read_json(run / "checkpoints" / FIT_STATE_NAME)
    assert not sidecar.get("fit_finished")
    assert sorted(sidecar["epochs"], key=int) == [str(e) for e in range(ran)]
    assert read_json(run / "checkpoints" / "backup" / BACKUP_METADATA_NAME) == {"epoch": ran, "batch": 0}
    assert not (run / "results_summary.json").exists() and not (run / "final_model.keras").exists()
    assert (run / "best_model.keras").is_file()
    assert len((run / "training_log.csv").read_text().strip().splitlines()) == ran + 1


# ---------------------------------------------------------------------
# The resume reports, and reports what the unkilled run reported
# ---------------------------------------------------------------------


def test_the_resume_runs_only_the_reporting_and_says_why_no_epoch_was_left(scenario):
    ran = EPOCHS_RAN[scenario.name]
    summary = read_json(scenario.resumed / "results_summary.json")
    assert summary["reporting_resumed"] is True and summary["resumed_after_epochs"] == ran
    assert summary["epochs_run"] == ran and summary["stopped_early"] is (scenario.name == "early_stopping")
    assert summary["fit_wall_seconds"] is None and summary["epoch_times"] == [None] * ran
    note = next(n for n in summary["notes"] if "reporting ran again from the files" in n)
    assert REASON[scenario.name] in note and "the training was complete" in note
    log = (scenario.resumed / "run.log").read_text()
    assert "whose training is complete" in log and REASON[scenario.name] in log
    assert read_json(scenario.resumed / "checkpoints" / FIT_STATE_NAME)["reporting_resumes"] == 1


def test_the_csv_is_the_trainings_and_no_copy_of_it_was_made(scenario):
    assert sha(scenario.resumed / "training_log.csv") == scenario.csv_of_the_kill
    assert not list(scenario.resumed.glob("training_log.pre_resume_*"))


def test_the_summary_says_what_the_unkilled_run_says(scenario):
    whole = read_json(scenario.whole / "results_summary.json")
    summary = read_json(scenario.resumed / "results_summary.json")
    for key in ("best_epoch", "epochs_run", "stopped_early", "test_metrics_best", "test_metrics_final",
                "best_val_metrics", "final_val_metrics", "success_value", "success"):
        assert summary[key] == whole[key], key
    assert set(summary) - set(whole) == {"resumed_after_epochs", "reporting_resumed"}
    assert set(whole) - set(summary) == set()


def test_final_model_keras_holds_the_last_epochs_weights_and_the_optimizer_state_of_the_unkilled_run(scenario):
    """Mutants ``optimizer not built before the load`` and ``best weights saved as final``."""
    from .test_backup_every_steps import distance

    whole_model, whole_optimizer = final_state(scenario.whole)
    model, optimizer = final_state(scenario.resumed)
    assert optimizer, "the resumed final_model.keras holds no optimizer variable"
    assert distance(whole_model, model) == (0.0, 0.0)
    assert distance(whole_optimizer, optimizer) == (0.0, 0.0)


# ---------------------------------------------------------------------
# Where the last epoch's weights cannot be proven, the resume is refused with a true message
# ---------------------------------------------------------------------


def test_a_torn_latest_weights_file_is_refused_naming_it_and_the_previous_copy_is_not_used(scenario, tmp_path):
    def tear(run):
        path = run / "checkpoints" / "backup" / BACKUP_WEIGHTS_NAME
        path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])

    text = refusal(scenario, tmp_path, tear)
    assert BACKUP_WEIGHTS_NAME in text and "its training is complete" in text
    assert "best_model.keras is untouched" in text
    # The old message blamed a reporting crash and cited a decision that said the opposite.
    assert "D-021" not in text and "crashed while reporting" not in text


def test_a_sidecar_that_lost_the_last_epoch_is_refused_not_reported_on(scenario, tmp_path):
    """The sidecar and the backup must agree on where the training ended."""
    ran = EPOCHS_RAN[scenario.name]

    def lose_the_last_entry(run):
        path = run / "checkpoints" / FIT_STATE_NAME
        state = read_json(path)
        del state["epochs"][str(ran - 1)]
        path.write_text(json.dumps(state))

    text = refusal(scenario, tmp_path, lose_the_last_entry)
    assert "do not agree" in text, text[-1200:]
    assert "D-021" not in text
