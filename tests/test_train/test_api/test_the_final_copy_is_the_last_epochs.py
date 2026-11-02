"""``final_copy_problem`` accepts the latest backup copy only when it IS the last epoch's weights (D-041).

A reporting-only resume evaluates ``checkpoints/backup/latest.weights.h5`` as the FINAL model, so
the rule "the latest copy is the epoch-end copy of the epochs that ran, or the resume is refused"
is what stands between a finished training and a summary that calls earlier weights the final
ones. It has two halves and each was removable with the whole of ``tests/test_train/test_api``
green (review-iter-2 concern 5, mutants ``fc_batch_ignored`` and ``fc_epoch_ignored``):

* ``epoch != epochs_run``: an EPOCH-END copy of an earlier epoch. Its optimizer count is
  consistent with its metadata (``epoch x steps``), so the optimizer check passes it; only the
  epoch comparison refuses it.
* ``batch != 0``: a copy written INSIDE an epoch. Its optimizer count is consistent too
  (``epoch x steps + batch + 1``), and its epoch can equal ``epochs_run``; only the batch
  comparison refuses it.

The copies are real: a tiny model is fitted under the trainer's own callbacks
(``StepBackupAndRestore``), and a kill is a callback that raises.
"""

import json

import pytest

from train.common.resume_state import (
    BACKUP_METADATA_NAME,
    BACKUP_WEIGHTS_NAME,
    final_copy_problem,
)

from .test_backup_every_steps import S, KillAt, StepBackupAndRestore, fit, metadata, tiny, weights_steps


def test_the_epoch_end_copy_of_the_epochs_that_ran_is_the_final_weights(tmp_path):
    """Control: the one state that is accepted, so every refusal below is that state damaged."""
    assert not fit(tiny(), [StepBackupAndRestore(tmp_path, 4, S)], epochs=2)
    assert metadata(tmp_path) == {"epoch": 2, "batch": 0} and weights_steps(tmp_path) == 2 * S
    assert final_copy_problem(tmp_path, 2, S) is None


def test_an_epoch_end_copy_of_an_earlier_epoch_is_not_the_final_weights(tmp_path):
    """Mutant ``fc_epoch_ignored``: a kill in epoch 2 leaves the copy of epoch 1's end, sound by its
    optimizer count (``1 x S``) and by its batch (0); the run trained 2 epochs."""
    assert fit(tiny(), [KillAt(1, 0), StepBackupAndRestore(tmp_path, 4, S)], epochs=3)
    assert metadata(tmp_path) == {"epoch": 1, "batch": 0} and weights_steps(tmp_path) == S
    problem = final_copy_problem(tmp_path, 2, S)
    assert problem is not None
    assert BACKUP_METADATA_NAME in problem and "epoch 1 batch 0" in problem and "end of epoch 2" in problem
    assert final_copy_problem(tmp_path, 1, S) is None  # the same copy IS the end of epoch 1


def test_a_copy_taken_inside_the_next_epoch_is_not_the_final_weights_even_with_a_consistent_count(tmp_path):
    """Mutant ``fc_batch_ignored``: the copy after batch 4 of epoch 1 says ``epoch 1`` (so it equals
    ``epochs_run`` for a run that trained one epoch) and its optimizer took ``S + 4`` steps, which is
    exactly what that position implies; only ``batch != 0`` tells it from the end of epoch 1."""
    assert fit(tiny(), [KillAt(1, 5), StepBackupAndRestore(tmp_path, 4, S)], epochs=3)
    assert metadata(tmp_path) == {"epoch": 1, "batch": 3} and weights_steps(tmp_path) == S + 4
    problem = final_copy_problem(tmp_path, 1, S)
    assert problem is not None and "epoch 1 batch 3" in problem and "end of epoch 1 (batch 0)" in problem


@pytest.mark.parametrize("damage, fragment", [
    (lambda d: (d / BACKUP_WEIGHTS_NAME).unlink(), f"{BACKUP_WEIGHTS_NAME} does not exist"),
    (lambda d: (d / BACKUP_WEIGHTS_NAME).write_bytes(b"torn"), f"{BACKUP_WEIGHTS_NAME} cannot be read"),
    (lambda d: (d / BACKUP_METADATA_NAME).unlink(), f"{BACKUP_METADATA_NAME} does not exist"),
    (lambda d: (d / BACKUP_METADATA_NAME).write_text(json.dumps({"epoch": 1, "batch": 0})),
     "different epochs"),
], ids=["no_weights", "torn_weights", "no_metadata", "metadata_of_another_epoch"])
def test_a_copy_that_cannot_be_read_or_whose_pair_disagrees_is_not_the_final_weights(tmp_path, damage, fragment):
    assert not fit(tiny(), [StepBackupAndRestore(tmp_path, 4, S)], epochs=2)
    assert final_copy_problem(tmp_path, 2, S) is None
    damage(tmp_path)
    problem = final_copy_problem(tmp_path, 2, S)
    assert problem is not None and fragment in problem


def test_the_previous_copy_is_never_put_in_the_latest_ones_place(tmp_path):
    """READ ONLY: refusing must not run ``settle_backup``'s fallback (which copies ``.bkp`` over the latest)."""
    assert fit(tiny(), [KillAt(1, 5), StepBackupAndRestore(tmp_path, 4, S)], epochs=3)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert final_copy_problem(tmp_path, 1, S) is not None
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
