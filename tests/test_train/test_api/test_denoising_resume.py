"""``resume_interrupted=true`` for the ``image_denoising`` task, with a REAL SIGKILL, per family (plan d483ffab step 13).

The task adds three things a resume could get wrong, none of which the classification resume tests touch:

* **the curriculum**: ``NoiseSigmaCurriculumCallback`` keeps no state of its own (D-013's design relies on Keras
  passing the ABSOLUTE epoch to ``on_epoch_begin`` after a resume), so a resumed process that counted epochs from 0
  would train epoch 3 at the sigma of epoch 0 and nothing would fail loudly;
* **the validation set**: ``val_psnr`` is a function of the weights alone (stateless noise keyed on ``(val_seed, index)``),
  so the weights a resume restores must score what the killed run logged for the epoch they came from;
* **the selection window** (``selection_start_epoch``, D-013): the checkpoint saves nothing before the curriculum's last
  epoch, so a run killed before it has no ``best_model.keras`` and a run killed after it has one, and a resume on either
  side must finish with the window's best.

Each family runs one chain of three child processes (``_denoising_child.py``, the marker protocol and the kill of
``test_backup_every_steps.run_child``: the parent SIGKILLs the RECORDED PID at a ``@@AT`` marker):

* process 1 trains and is killed after 1 batch of epoch 1 (of 4; the curriculum is at sigma 0.1375 of its ramp
  0.025 -> 0.25, reached at epoch 2). The backup holds the end of epoch 0: no ``best_model.keras`` yet, and the resume
  starts at epoch 1, BEFORE the window (which starts at epoch 2);
* process 2 resumes, scores the weights it restored, trains epochs 1 and 2 (the window opens, ``best_model.keras``
  appears) and is killed after 1 batch of epoch 3: the second resume starts AT the window's first epoch's successor,
  AFTER the window start;
* process 3 resumes the same way and finishes.

What is read off the trace the children append to (one line per batch and per epoch, flushed, so a kill leaves exactly
what ran) and off the files: every epoch begins at its absolute index and its hand-computed sigma; the optimizer's step
count is ``epoch * steps`` at each epoch's start and ``epochs * steps`` at the end, and the real number of updates is that
plus exactly the batches the two kills lost; the CSV rows are unique and complete; the restored weights score the
``val_psnr`` the killed run logged; the summary reports the window and the epoch it resumed after. One more run (``bfcnn``,
``checkpoint_args.backup_every_steps=2``) kills INSIDE an epoch after a mid-epoch copy: the resume runs only the batches
the copy had not taken, at the sigma of the absolute epoch, and the step count is the unkilled run's.

NOT checked: a resumed run is not the unkilled run's trajectory (the killed epoch draws a fresh pass of the data; the
classification file measures that distance), a GPU, a kill during a weight write (``test_backup_every_steps``).

The tests write only under ``tmp_path``.
"""

import csv
import json
from pathlib import Path

import pytest

from train.common.resume_state import FIT_STATE_NAME

from .test_backup_every_steps import run_child
from .test_image_denoising_task import (
    BATCH,
    FAMILIES,
    PATCHES,
    TRAIN_FILES,
    VAL_FILES,
    mapping,
    write_images,
)

CHILD = Path(__file__).resolve().parent / "_denoising_child.py"

EPOCHS = 4
#: Batches of one epoch: ``files * patches_per_image // batch`` (the pipeline is finite and drops the remainder).
STEPS = TRAIN_FILES * PATCHES // BATCH
#: ``data.curriculum_epochs`` auto is ``round(0.8 * 4) = 3``: the upper bound ramps over epochs 0, 1, 2 and the selection
#: window opens at the ramp's last epoch (index 2).
WINDOW_START = 2
#: ``sigma_max`` of each epoch by hand: linear from 0.025 to 0.25 over 3 epochs (0.025 + 0.225 * epoch / 2), then flat.
SIGMA = (0.025, 0.1375, 0.25, 0.25)
#: Where the two kills land: ``(epoch, batches finished in it)``.
KILL_BEFORE_WINDOW, KILL_AFTER_WINDOW = (1, 1), (3, 1)


def read_trace(path):
    """The trace of a chain as a list of dicts, in the order the processes wrote them."""
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def csv_rows(path):
    """``[{column: value}]`` of a ``training_log.csv``."""
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


class Chain:
    """Child processes on one run directory: what each ran, and what each kill left behind."""

    def __init__(self, root: Path, family: str, folders, **request) -> None:
        self.root, self.family, self.folders, self.request = Path(root), family, folders, request
        self.run = self.root / "run"
        self.trace = self.root / "trace.jsonl"
        #: ``{label: {"csv": rows, "best_model": bool, "sidecar": dict}}`` read right after each kill.
        self.after = {}

    def go(self, label: str, expect: str, resume: bool = False, **spec) -> None:
        body = mapping(self.run, self.folders, self.family, epochs=EPOCHS, resume_interrupted=resume, **self.request)
        marker = run_child(
            {"mapping": body, "trace": str(self.trace), "label": label, **spec},
            self.root / f"{label}.log", child_script=CHILD,
        )
        assert marker == expect, f"{self.family}/{label}: {marker}"
        if marker != "@@DONE":
            self.after[label] = {
                "csv": csv_rows(self.run / "training_log.csv"),
                "best_model": (self.run / "best_model.keras").exists(),
                "sidecar": json.loads((self.run / "checkpoints" / FIT_STATE_NAME).read_text()),
            }

    def lines(self, kind: str, label: str = None):
        return [row for row in read_trace(self.trace) if row["kind"] == kind and label in (None, row["label"])]


@pytest.fixture(scope="module")
def folders(tmp_path_factory):
    root = tmp_path_factory.mktemp("denoise_resume")
    return write_images(root / "train", TRAIN_FILES, 0), write_images(root / "val", VAL_FILES, 1)


@pytest.fixture(scope="module", params=FAMILIES)
def chain(request, folders, tmp_path_factory):
    """Kill before the window, resume, kill after it, resume, finish: three processes of one family."""
    family = request.param
    run = Chain(tmp_path_factory.mktemp(f"resume_{family}"), family, folders)
    run.go("p1", f"@@AT {KILL_BEFORE_WINDOW[0]} {KILL_BEFORE_WINDOW[1]}", stop_at=list(KILL_BEFORE_WINDOW))
    run.go("p2", f"@@AT {KILL_AFTER_WINDOW[0]} {KILL_AFTER_WINDOW[1]}", resume=True, probe_restored=True,
           stop_at=list(KILL_AFTER_WINDOW))
    run.go("p3", "@@DONE", resume=True, probe_restored=True)
    return run


@pytest.fixture(scope="module")
def mid_epoch(folders, tmp_path_factory):
    """``bfcnn`` with a copy every 2 batches: killed after 2 batches of epoch 1, then resumed to the end."""
    run = Chain(tmp_path_factory.mktemp("resume_mid_epoch"), "bfcnn", folders,
                checkpoint_args={"backup_every_steps": 2})
    run.go("p1", "@@AT 1 2", stop_at=[1, 2])
    run.go("p2", "@@DONE", resume=True)
    return run


# ---------------------------------------------------------------------
# The kills are real, mid-epoch, with the curriculum running
# ---------------------------------------------------------------------


def test_each_kill_lands_between_two_batches_of_an_epoch_the_curriculum_is_running(chain):
    first, second = chain.lines("batch", "p1"), chain.lines("batch", "p2")
    assert (first[-1]["epoch"], first[-1]["batch"]) == (1, 0), "process 1 was killed after exactly 1 batch of epoch 1"
    assert (second[-1]["epoch"], second[-1]["batch"]) == (3, 0), "process 2 was killed after exactly 1 batch of epoch 3"
    # The kill is inside the ramp (0.1375 is between 0.025 and 0.25) for the first, at the full range for the second.
    assert first[-1]["sigma"] == pytest.approx(0.1375, abs=1e-6) and 0.025 < first[-1]["sigma"] < 0.25
    assert second[-1]["sigma"] == pytest.approx(0.25, abs=1e-6)
    assert not [r for r in chain.lines("epoch_end", "p1") if r["epoch"] == 1], "epoch 1 had not ended in process 1"
    assert not chain.lines("train_end", "p1") and not chain.lines("train_end", "p2"), "a killed process ends nothing"


def test_a_resume_starts_at_the_epoch_after_the_last_finished_one_and_the_killed_epoch_runs_again(chain):
    epochs = {label: [row["epoch"] for row in chain.lines("epoch_begin", label)] for label in ("p1", "p2", "p3")}
    assert epochs == {"p1": [0, 1], "p2": [1, 2, 3], "p3": [3]}


# ---------------------------------------------------------------------
# Sigma by absolute epoch
# ---------------------------------------------------------------------


def test_the_sigma_of_every_epoch_and_every_batch_is_the_one_of_its_absolute_epoch(chain):
    seen = chain.lines("epoch_begin") + chain.lines("batch")
    assert len(chain.lines("epoch_begin")) == 2 + 3 + 1, "every epoch start of the three processes is in the trace"
    for row in seen:
        assert row["sigma"] == pytest.approx(SIGMA[row["epoch"]], abs=1e-6), (
            f"{row['label']} epoch {row['epoch']} ({row['kind']}) read sigma_max {row['sigma']}, "
            f"the schedule says {SIGMA[row['epoch']]}"
        )


def test_the_resumed_epochs_do_not_read_the_sigma_of_a_restarted_counter(chain):
    """The tell of a curriculum that counts epochs from 0 in the new process: epoch 3 at the sigma of epoch 0."""
    for label, epoch in (("p2", 1), ("p3", 3)):
        first = chain.lines("epoch_begin", label)[0]
        assert first["epoch"] == epoch
        assert first["sigma"] == pytest.approx(SIGMA[epoch], abs=1e-6)
        assert abs(first["sigma"] - SIGMA[0]) > 0.1, f"{label} began at the sigma of epoch 0"


# ---------------------------------------------------------------------
# The step count, and the updates that really ran
# ---------------------------------------------------------------------


def test_the_optimizer_step_count_is_epoch_times_steps_at_every_epoch_start_and_the_unkilled_total_at_the_end(chain):
    for row in chain.lines("epoch_begin"):
        assert row["iterations"] == row["epoch"] * STEPS, f"{row['label']} epoch {row['epoch']} began at {row['iterations']}"
    for row in chain.lines("epoch_end"):
        assert row["iterations"] == (row["epoch"] + 1) * STEPS
    assert [row["iterations"] for row in chain.lines("train_end")] == [EPOCHS * STEPS]


def test_the_real_updates_are_the_asked_steps_plus_exactly_the_batches_the_kills_lost(chain):
    lost_in_p1 = KILL_BEFORE_WINDOW[1]   # epoch 1 of process 1: its batches are trained again after the resume
    lost_in_p2 = KILL_AFTER_WINDOW[1]    # epoch 3 of process 2
    batches = chain.lines("batch")
    assert len(batches) == EPOCHS * STEPS + lost_in_p1 + lost_in_p2
    # Every batch used the step count of its place in its epoch: no batch ran on a count that had been kept from the kill.
    for row in batches:
        assert row["before"] == row["epoch"] * STEPS + row["batch"], row


# ---------------------------------------------------------------------
# Rows, window, weights
# ---------------------------------------------------------------------


def test_the_csv_rows_are_unique_and_complete_and_the_earlier_rows_are_kept_as_they_were(chain):
    final = csv_rows(chain.run / "training_log.csv")
    assert [row["epoch"] for row in final] == [str(e) for e in range(EPOCHS)]
    killed_first, killed_second = chain.after["p1"]["csv"], chain.after["p2"]["csv"]
    assert [row["epoch"] for row in killed_first] == ["0"]
    assert [row["epoch"] for row in killed_second] == ["0", "1", "2"]
    assert final[:1] == killed_first and final[:3] == killed_second, "a resume rewrote a finished epoch's row"


def test_the_weights_a_resume_restores_score_the_val_psnr_the_killed_run_logged_for_them(chain):
    for label, from_row in (("p2", chain.after["p1"]["csv"][-1]), ("p3", chain.after["p2"]["csv"][-1])):
        restored = chain.lines("epoch_begin", label)[0]["restored"]
        logged = float(from_row["val_psnr"])
        # Equal to the last bit (measured for the five families): the same weights, the same fixed patches and the same
        # stateless noise, on one thread under op determinism, in another process.
        assert restored["psnr"] == logged, (
            f"{label}: the restored weights score {restored['psnr']!r} dB, epoch {from_row['epoch']} of the killed run "
            f"logged {logged!r}"
        )


def test_a_resume_before_the_window_has_no_best_model_yet_and_one_after_it_has_the_windows(chain):
    assert chain.after["p1"]["best_model"] is False, "epoch 0 is before the window: nothing may be saved there"
    assert chain.after["p2"]["best_model"] is True, "epochs 1 and 2 ran in process 2: epoch 2 opens the window"
    summary = json.loads((chain.run / "results_summary.json").read_text())
    assert summary["resumed_after_epochs"] == KILL_AFTER_WINDOW[0] and summary["epochs_run"] == EPOCHS
    selection = summary["selection"]
    assert selection["start_epoch"] == WINDOW_START and selection["window_epochs"] == EPOCHS - WINDOW_START
    assert WINDOW_START < selection["best_epoch"] <= EPOCHS, "best_epoch is 1-based and inside the window"
    assert selection["final_is_best"] is (selection["best_epoch"] == EPOCHS)
    assert (chain.run / "final_model.keras").is_file() and (chain.run / "best_model.keras").is_file()
    # What the sidecar kept across the second kill: nothing is "best" before the window, and epoch 2 (0-based) is the
    # window's first and so its best so far, with the val_psnr its CSV row holds.
    kept = chain.after["p2"]["sidecar"]["epochs"]
    assert kept["0"]["best"] is None and kept["1"]["best"] is None
    assert kept["2"]["best_epoch"] == WINDOW_START
    assert kept["2"]["best"] == pytest.approx(float(chain.after["p2"]["csv"][2]["val_psnr"]), rel=1e-6)


# ---------------------------------------------------------------------
# A kill inside an epoch, after a mid-epoch copy
# ---------------------------------------------------------------------


def test_a_resume_after_a_mid_epoch_copy_runs_only_the_batches_the_copy_had_not_taken(mid_epoch):
    first, second = mid_epoch.lines("epoch_begin", "p1"), mid_epoch.lines("epoch_begin", "p2")
    assert [row["epoch"] for row in first] == [0, 1] and [row["epoch"] for row in second] == [1, 2, 3]
    # The step count is the copy's (3 + 2), not the epoch's start (3): the 2 batches are not trained again.
    assert second[0]["iterations"] == STEPS + 2
    ran_epoch_1 = [row for row in mid_epoch.lines("batch", "p2") if row["epoch"] == 1]
    assert len(ran_epoch_1) == STEPS - 2
    assert len(mid_epoch.lines("batch")) == EPOCHS * STEPS, "no batch was lost: the kill came right after a copy"
    assert [row["iterations"] for row in mid_epoch.lines("train_end", "p2")] == [EPOCHS * STEPS]


def test_the_sigma_of_a_resumed_mid_epoch_is_the_one_of_its_absolute_epoch(mid_epoch):
    for row in mid_epoch.lines("epoch_begin") + mid_epoch.lines("batch"):
        assert row["sigma"] == pytest.approx(SIGMA[row["epoch"]], abs=1e-6), row


def test_the_mid_epoch_resume_leaves_each_epoch_one_row_and_a_window_best(mid_epoch):
    rows = csv_rows(mid_epoch.run / "training_log.csv")
    assert [row["epoch"] for row in rows] == [str(e) for e in range(EPOCHS)]
    summary = json.loads((mid_epoch.run / "results_summary.json").read_text())
    assert summary["resumed_after_epochs"] == 1 and summary["selection"]["start_epoch"] == WINDOW_START
