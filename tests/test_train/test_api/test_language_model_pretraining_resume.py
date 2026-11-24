"""``resume_interrupted=true`` for ``language_model_pretraining``, with a REAL SIGKILL: a resume CONTINUES the stream.

The train stream of this task is infinite and an epoch is ``steps_per_epoch`` batches by ``fit_kwargs``, so a resumed ``fit`` would
start a fresh iterator and replay the stream from its first batch. The image streams (``pretrain``, forecasting, depth) answer by
DERIVING another stream seed from ``(seed, completed steps)``: process 2 trains on other masks than process 1. A text stream
cannot do that without lying: epoch ``e`` is a permutation of the training documents keyed on ``(seed, e)``, the first process
consumed the documents in THAT order, and another order would visit some documents twice in an epoch and others not at all.
So ``resume_stream`` returns the SAME stream starting at ``completed_steps``, and this file pins what that means:

* **the resumed process reads the batch the uninterrupted run reads at that step**: the first batches of process 2 are, byte for
  byte, ``stream.batch_at(settled steps + i)``, and process 1's are ``batch_at(i)`` (the digests the children traced are those the
  parent computes from its own build of the data). The control: those two streams differ, so a replay would be seen;
* **the split never moves**: every process traces the same selection windows, and the weights a resume restores score exactly the
  ``val_loss`` the killed run logged for them (the selection windows are fixed by the split seed, one thread, op determinism);
* **the settled count reaches the stream**, at an epoch boundary (``start * steps``) and from a copy taken inside an epoch
  (``start * steps + batch + 1``, ``checkpoint_args.backup_every_steps=2``): the batch term is part of the count;
* ``resume_stream`` returns a stream, not ``data``, and the replacement keeps what the sequence checks (``steps_per_epoch``, the
  three counts, no fingerprint).

One chain of three child processes (``_language_child.py``; the marker protocol of ``test_backup_every_steps.run_child``, which
SIGKILLs the RECORDED PID at a ``@@AT`` marker) on 3 epochs: process 1 is killed after 2 batches of epoch 1, process 2 resumes,
finishes epoch 1 and is killed after 2 batches of epoch 2, process 3 resumes and finishes; and a second chain (process 1 killed
inside epoch 1 right after a mid-epoch copy, process 2 resumes). NOT checked: a resumed run is not the unkilled run's weights
(the optimizer's moving averages and the kill are real), a GPU, a kill during a weight write (``test_backup_every_steps``). The
tests write only under ``tmp_path``.
"""

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from . import _resume_protocol as protocol
from ._resume_protocol import csv_rows
from ._text_world import BATCH, EPOCHS, prepared, request, world  # noqa: F401  (``world`` is a fixture)

CHILD = Path(__file__).resolve().parent / "_language_child.py"

#: Where the two kills land: ``(epoch, batches finished in it)``.
KILL_1, KILL_2 = (1, 2), (2, 2)
FIRST = 3        # the batches whose digest is compared


def digest(batch) -> str:
    x, y = batch
    return hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()


class Chain(protocol.ResumeChain):
    """Child processes on one run directory: what each ran, and what each kill left behind."""

    child_script = CHILD
    first = FIRST

    def mapping(self):
        return request(self.root / "run", **self.changes)


@pytest.fixture(scope="module")
def parent(world, tmp_path_factory):  # noqa: F811
    """The parent's own build of the data of every child (a numpy stream: the same in every process)."""
    trainer, resolved, context = prepared(tmp_path_factory.mktemp("expected") / "run")
    data = trainer.build_data(resolved, context)
    steps = data.steps_per_epoch
    assert steps > KILL_2[1] + 1, "the control: an epoch is long enough for a kill inside it"

    def stream_from(count):
        return [digest(data.split.stream.batch_at(count + i)) for i in range(FIRST)]

    counts = {None: 0, "p2": steps, "p3": 2 * steps, "mid": steps + 2, "epoch_start": steps}
    select = digest(data.split.select)
    return dict(trainer=trainer, resolved=resolved, data=data, steps=steps, select=select,
                streams={key: stream_from(count) for key, count in counts.items()}, counts=counts)


@pytest.fixture(scope="module")
def chain(world, parent, tmp_path_factory):  # noqa: F811
    """Kill inside epoch 1, resume, kill inside epoch 2, resume, finish: three processes."""
    run = Chain(tmp_path_factory.mktemp("resume_language"), "language_model")
    run.go("p1", f"@@AT {KILL_1[0]} {KILL_1[1]}", stop_at=list(KILL_1))
    run.go("p2", f"@@AT {KILL_2[0]} {KILL_2[1]}", resume=True, probe_restored=True, stop_at=list(KILL_2))
    run.go("p3", "@@DONE", resume=True, probe_restored=True)
    return run


@pytest.fixture(scope="module")
def mid_epoch(world, parent, tmp_path_factory):  # noqa: F811
    """A copy every 2 batches: killed after 2 batches of epoch 1, then resumed to the end."""
    run = Chain(tmp_path_factory.mktemp("resume_language_mid"), "language_model",
                **{"checkpoint_args.backup_every_steps": 2})
    run.go("p1", "@@AT 1 2", stop_at=[1, 2])
    run.go("p2", "@@DONE", resume=True)
    return run


# ---------------------------------------------------------------------
# The kills are real and mid-epoch; the epochs and the step count are absolute
# ---------------------------------------------------------------------


def test_the_sizes_the_chain_counts_are_the_datas(chain, parent):
    assert parent["steps"] == parent["data"].n_train // BATCH
    # one line when a process builds its data, and one more for the stream a resume replaces it with
    assert [row["label"] for row in chain.lines("select")] == ["p1", "p2", "p2", "p3", "p3"]


def test_each_kill_lands_between_two_batches_of_an_epoch(chain):
    protocol.check_each_kill_lands_between_two_batches(chain, KILL_1, KILL_2)


def test_a_resume_starts_at_the_epoch_after_the_last_finished_one_and_the_killed_epoch_runs_again(chain):
    protocol.check_a_resume_starts_after_the_last_finished_epoch(chain)


def test_the_optimizer_step_count_is_epoch_times_steps_at_every_epoch_start_and_the_unkilled_total_at_the_end(chain, parent):
    protocol.check_the_step_count_is_absolute(chain, parent["steps"], EPOCHS, KILL_1, KILL_2)


def test_the_csv_rows_are_unique_and_complete_and_the_earlier_rows_are_kept_as_they_were(chain):
    protocol.check_the_csv_rows_are_unique_and_complete(chain, EPOCHS)


def test_the_weights_a_resume_restores_score_the_val_loss_the_killed_run_logged_for_them(chain):
    protocol.check_the_restored_weights_score_the_logged_value(chain, "loss")


def test_the_summary_says_the_run_resumed_and_finished(chain):
    summary = json.loads((chain.run / "results_summary.json").read_text())
    assert summary["resumed_after_epochs"] == KILL_2[0] and summary["epochs_run"] == EPOCHS
    assert (chain.run / "final_model.keras").is_file() and summary["success"] is True
    assert any("replays the order" not in note and "continues" in note for note in summary["notes"]), (
        "the note a resume writes says what a resume does to the stream of THIS task")


# ---------------------------------------------------------------------
# A resume continues the stream: it neither replays nor re-seeds
# ---------------------------------------------------------------------


def test_the_first_batches_of_each_process_are_the_streams_at_its_settled_count(chain, parent):
    # Process 1 is the fresh stream (count 0); process 2 resumes at the end of epoch 0 (steps), process 3 at the end of epoch 1
    # (2 * steps). The children traced what `fit` read; the parent computed `stream.batch_at(count + i)` in its own process.
    streams = {None: parent["streams"][None], parent["counts"]["p2"]: parent["streams"]["p2"],
               parent["counts"]["p3"]: parent["streams"]["p3"]}
    protocol.check_each_process_trains_on_the_stream_of_its_settled_count(
        chain, streams, {"p2": parent["counts"]["p2"], "p3": parent["counts"]["p3"]})


def test_no_process_replays_the_batches_of_the_process_before(chain):
    protocol.check_no_process_replays_the_one_before(chain)


def test_the_selection_windows_are_the_same_in_every_process_and_the_parents(chain, parent):
    for row in chain.lines("select"):
        assert row["digest"] == parent["select"], f"{row['label']} selects other windows: the split moved"
    assert {row["label"] for row in chain.lines("select")} == {"p1", "p2", "p3"}


# ---------------------------------------------------------------------
# A kill inside an epoch, after a mid-epoch copy
# ---------------------------------------------------------------------


def test_a_resume_after_a_mid_epoch_copy_runs_only_the_batches_the_copy_had_not_taken(mid_epoch, parent):
    protocol.check_a_mid_epoch_resume_runs_only_the_batches_the_copy_had_not_taken(
        mid_epoch, parent["steps"], EPOCHS, parent["counts"]["mid"])


def test_the_stream_after_a_mid_epoch_copy_is_the_one_of_the_count_the_copy_holds(mid_epoch, parent):
    # steps + 2 = 1 * steps + (1 + 1): the batch term is part of the settled count. `steps` (the epoch's start) is another stream.
    streams = {None: parent["streams"][None], parent["counts"]["mid"]: parent["streams"]["mid"],
               parent["counts"]["epoch_start"]: parent["streams"]["epoch_start"]}
    protocol.check_the_mid_epoch_stream(mid_epoch, streams, parent["steps"], parent["counts"]["mid"])


def test_the_mid_epoch_resume_leaves_each_epoch_one_row_and_keeps_the_split(mid_epoch, parent):
    assert [row["epoch"] for row in csv_rows(mid_epoch.run / "training_log.csv")] == [str(e) for e in range(EPOCHS)]
    selects = [row["digest"] for row in mid_epoch.lines("select")]
    assert len(selects) == 3 and set(selects) == {parent["select"]}, "p1 built once, p2 built and replaced its stream"


# ---------------------------------------------------------------------
# The seam itself
# ---------------------------------------------------------------------


def test_resume_stream_returns_a_stream_that_starts_at_the_count_and_keeps_what_the_sequence_checks(parent):
    trainer, resolved, data = parent["trainer"], parent["resolved"], parent["data"]
    count = parent["steps"] + 1
    replacement = trainer.resume_stream(resolved, data, count)
    assert replacement is not data and replacement.train_ds is not data.train_ds, "None or `data` would be the replay"
    assert (replacement.steps_per_epoch, replacement.n_train, replacement.n_select, replacement.n_test) == (
        data.steps_per_epoch, data.n_train, data.n_select, data.n_test)
    assert getattr(replacement, "fingerprint", None) is None and getattr(data, "fingerprint", None) is None
    assert replacement.select_ds is data.select_ds and replacement.test_ds is data.test_ds, "the fixed sets are not rebuilt"
    first = next(iter(replacement.train_ds))
    assert digest((first[0].numpy(), first[1].numpy())) == digest(data.split.stream.batch_at(count))
    again = next(iter(data.train_ds))
    assert digest((again[0].numpy(), again[1].numpy())) == digest(data.split.stream.batch_at(0)), "the original starts at 0"
    assert digest(data.split.stream.batch_at(count)) != digest(data.split.stream.batch_at(0)), "the control: they differ"
    assert replacement.split.stream.start_step == count and data.split.stream.start_step == 0
    np.testing.assert_array_equal(replacement.split.select[0], data.split.select[0])
    assert dataclasses.is_dataclass(replacement)
    assert trainer.resume_order and "continue" in trainer.resume_order
