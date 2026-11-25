"""``resume_interrupted=true`` for ``text_classification``, with a REAL SIGKILL: a resume CONTINUES the stream, the rows and the split.

The twin of ``test_language_model_pretraining_resume.py`` (plan-2026-11-24T092348-7806a290 step 10, Pre-Mortem 3) for the classifier,
through the same child script and the same protocol (``_language_child.py`` takes the task as a key; ``_resume_protocol.py`` holds the
assertions). The train stream is infinite and an epoch is ``steps_per_epoch`` batches, so a resumed ``fit`` would replay it from its
first batch; epoch ``e`` is the train rows in an order keyed on ``(seed, e)``, so ``resume_stream`` returns the SAME stream starting at
the settled step and what it reads next is what the unkilled run reads next (``resume_stream_exact``: the four shared sentences about
a resume from a copy taken inside an epoch say so). Pinned, on a chain of three child processes (killed after 2 batches of epoch 1 and
of epoch 2, then finished) and a second chain (killed right after a mid-epoch copy):

* the first batches of each process, rows AND labels, are ``stream.batch_at(settled steps + i)`` as the parent computes them, and no
  process replays the one before (the control: the three streams differ);
* the selection rows, labels included, are the same in every process, and the weights a resume restores score exactly the
  ``val_loss`` the killed run logged for them;
* the settled count is epoch times steps at an epoch boundary and ``steps + batch + 1`` from a copy taken inside an epoch.

Tests write only under ``tmp_path``.
"""

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from . import _resume_protocol as protocol
from ._resume_protocol import csv_rows
from ._text_world import BATCH, classifier_prepared, classifier_request, world  # noqa: F401  (``world`` is a fixture)

CHILD = Path(__file__).resolve().parent / "_language_child.py"
EPOCHS = 3
#: Where the two kills land: ``(epoch, batches finished in it)``; the number of batches whose digest is compared.
KILL_1, KILL_2, FIRST = (1, 2), (2, 2), 3


def digest(batch) -> str:
    x, y = batch
    return hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()


class Chain(protocol.ResumeChain):
    """Child processes on one run directory: what each ran, and what each kill left behind."""

    child_script = CHILD
    first = FIRST

    def mapping(self):
        return classifier_request(self.root / "run", epochs=EPOCHS, **self.changes)

    def child_spec(self):
        return {"task": "text_classification"}


@pytest.fixture(scope="module")
def parent(world, tmp_path_factory):  # noqa: F811
    """The parent's own build of the data of every child (a numpy stream: the same in every process)."""
    trainer, resolved, context = classifier_prepared(tmp_path_factory.mktemp("expected") / "run", epochs=EPOCHS)
    data = trainer.build_data(resolved, context)
    steps = data.steps_per_epoch
    assert steps > KILL_2[1] + 1, "the control: an epoch is long enough for a kill inside it"
    counts = {None: 0, "p2": steps, "p3": 2 * steps, "mid": steps + 2, "epoch_start": steps}
    return dict(trainer=trainer, resolved=resolved, data=data, steps=steps, select=digest(data.split.select), counts=counts,
                streams={key: [digest(data.split.stream.batch_at(count + i)) for i in range(FIRST)] for key, count in counts.items()})


@pytest.fixture(scope="module")
def chain(world, parent, tmp_path_factory):  # noqa: F811
    """Kill inside epoch 1, resume, kill inside epoch 2, resume, finish: three processes."""
    run = Chain(tmp_path_factory.mktemp("resume_classifier"), "text_classification")
    run.go("p1", f"@@AT {KILL_1[0]} {KILL_1[1]}", stop_at=list(KILL_1))
    run.go("p2", f"@@AT {KILL_2[0]} {KILL_2[1]}", resume=True, probe_restored=True, stop_at=list(KILL_2))
    run.go("p3", "@@DONE", resume=True, probe_restored=True)
    return run


@pytest.fixture(scope="module")
def mid_epoch(world, parent, tmp_path_factory):  # noqa: F811
    """A copy every 2 batches: killed after 2 batches of epoch 1, then resumed to the end."""
    run = Chain(tmp_path_factory.mktemp("resume_classifier_mid"), "text_classification", **{"checkpoint_args.backup_every_steps": 2})
    run.go("p1", "@@AT 1 2", stop_at=[1, 2])
    run.go("p2", "@@DONE", resume=True)
    return run


def test_the_kills_are_real_and_the_epochs_and_the_step_count_are_absolute(chain, parent):
    assert parent["steps"] == parent["data"].n_train // BATCH
    assert [row["label"] for row in chain.lines("select")] == ["p1", "p2", "p2", "p3", "p3"]
    protocol.check_each_kill_lands_between_two_batches(chain, KILL_1, KILL_2)
    protocol.check_a_resume_starts_after_the_last_finished_epoch(chain)
    protocol.check_the_step_count_is_absolute(chain, parent["steps"], EPOCHS, KILL_1, KILL_2)
    protocol.check_the_csv_rows_are_unique_and_complete(chain, EPOCHS)


def test_the_weights_a_resume_restores_score_the_val_loss_the_killed_run_logged_for_them(chain):
    protocol.check_the_restored_weights_score_the_logged_value(chain, "loss")


def test_the_summary_says_the_run_resumed_and_what_a_resume_does_to_the_stream(chain):
    summary = json.loads((chain.run / "results_summary.json").read_text())
    assert summary["resumed_after_epochs"] == KILL_2[0] and summary["epochs_run"] == EPOCHS
    assert (chain.run / "final_model.keras").is_file()
    assert any("replays the order" not in note and "continues" in note for note in summary["notes"])


def test_the_first_batches_of_each_process_are_the_streams_at_its_settled_count_rows_and_labels(chain, parent):
    streams = {None: parent["streams"][None], parent["counts"]["p2"]: parent["streams"]["p2"],
               parent["counts"]["p3"]: parent["streams"]["p3"]}
    protocol.check_each_process_trains_on_the_stream_of_its_settled_count(
        chain, streams, {"p2": parent["counts"]["p2"], "p3": parent["counts"]["p3"]})
    protocol.check_no_process_replays_the_one_before(chain)


def test_the_selection_rows_and_their_labels_are_the_same_in_every_process_and_the_parents(chain, parent):
    assert {row["label"] for row in chain.lines("select")} == {"p1", "p2", "p3"}
    for row in chain.lines("select"):
        assert row["digest"] == parent["select"], f"{row['label']} selects other rows: the split moved"


def test_a_resume_after_a_mid_epoch_copy_continues_the_stream_at_the_count_the_copy_holds(mid_epoch, parent):
    protocol.check_a_mid_epoch_resume_runs_only_the_batches_the_copy_had_not_taken(
        mid_epoch, parent["steps"], EPOCHS, parent["counts"]["mid"])
    streams = {None: parent["streams"][None], parent["counts"]["mid"]: parent["streams"]["mid"],
               parent["counts"]["epoch_start"]: parent["streams"]["epoch_start"]}
    protocol.check_the_mid_epoch_stream(mid_epoch, streams, parent["steps"], parent["counts"]["mid"])
    selects = [row["digest"] for row in mid_epoch.lines("select")]
    assert len(selects) == 3 and set(selects) == {parent["select"]} and len(csv_rows(mid_epoch.run / "training_log.csv")) == EPOCHS


def test_the_sentences_a_mid_epoch_resume_writes_say_the_stream_continues_exactly(mid_epoch):
    """The ``Resume:`` log line and the killed-epoch note of the real resumed process (D-050): none says a fresh pass."""
    log = (mid_epoch.run / "run.log").read_text()
    notes = json.loads((mid_epoch.run / "results_summary.json").read_text())["notes"]
    for sentence in (next(line for line in log.splitlines() if "Resume: the backup was taken after batch" in line),
                     next(note for note in notes if note.startswith("epoch 1 was killed after "))):
        assert "the batches the unkilled run reads next" in sentence and "fresh pass" not in sentence, sentence


def test_resume_stream_returns_a_stream_that_starts_at_the_count_and_keeps_what_the_sequence_checks(parent):
    trainer, resolved, data = parent["trainer"], parent["resolved"], parent["data"]
    count = parent["steps"] + 1
    replacement = trainer.resume_stream(resolved, data, count)
    assert replacement is not data and replacement.train_ds is not data.train_ds, "None or `data` would be the replay"
    assert (replacement.steps_per_epoch, replacement.n_train, replacement.n_select, replacement.n_test) == (
        data.steps_per_epoch, data.n_train, data.n_select, data.n_test)
    assert replacement.fingerprint == data.fingerprint and len(data.fingerprint) == 64, "the corpus digest, kept by the replacement"
    assert replacement.select_ds is data.select_ds and replacement.test_ds is data.test_ds, "the fixed sets are not rebuilt"
    first = next(iter(replacement.train_ds))
    assert digest((first[0].numpy(), first[1].numpy())) == digest(data.split.stream.batch_at(count))
    assert digest(data.split.stream.batch_at(count)) != digest(data.split.stream.batch_at(0)), "the control: they differ"
    assert replacement.split.stream.start_step == count and data.split.stream.start_step == 0 and dataclasses.is_dataclass(replacement)
    np.testing.assert_array_equal(replacement.split.select[0], data.split.select[0])
    assert trainer.resume_stream_exact is True and "continue" in trainer.resume_order
