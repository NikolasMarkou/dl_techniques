"""``resume_interrupted=true`` for ``beit_tokenizer``, with a REAL SIGKILL: a resume trains on another stream and never on a selection row.

The train stream of this task is infinite and an epoch is ``steps_per_epoch`` batches by ``fit_kwargs``, so a resumed ``fit`` would start a
fresh iterator and replay the stream from its first batch. The task answers as the other image-stream tasks do (D-014): ``resume_stream``
is ``data.with_stream(completed_steps)``, the same request on a stream seed derived from ``(seed, completed steps)``. The shuffle and the
augmentation continue on a fresh stream, so a resume is NOT the unkilled run's next batches (``resume_stream_exact`` is false). What
this file pins, over a chain of REAL processes:

* **a resume must not replay**: process 2 trains on another stream than process 1 and process 3 on another than process 2;
* **a resume must not leak**: the split keeps the split seed, so no train row of any process is a selection row, the selection rows are the
  same in every process, and the weights a resume restores score the ``val_reconstruction_loss`` the killed run logged for them;
* **the settled count reaches the stream**: the stream of a process is the one ``TokenizerBatches.with_stream`` gives for exactly the count
  its copy held (``start * steps + batch + 1`` for a copy taken inside an epoch);
* ``resume_stream`` returns a stream, not ``data``, and keeps what the sequence checks.

One chain of three child processes (``_tokenizer_child.py``; the marker protocol of ``test_backup_every_steps.run_child``, which SIGKILLs the
RECORDED PID at a ``@@AT`` marker) on 3 epochs of 3 batches: process 1 is killed after 2 batches of epoch 1 (the backup holds the end of epoch
0), process 2 resumes, finishes epoch 1 and is killed after 2 batches of epoch 2, process 3 resumes and finishes. One more run
(``checkpoint_args.backup_every_steps=2``) is killed INSIDE epoch 1 right after a mid-epoch copy: the resume runs the one batch the copy had
not taken, at ``completed_steps = 3 + 2``. The data is un-augmented (the augmentation seeds are re-randomised on every iterator) and the
identity fake CIFAR-10. NOT checked: a resumed run is not the unkilled run's trajectory, a GPU, a kill during a weight write
(``test_backup_every_steps``). The tests write only under ``tmp_path``.
"""

import json
from pathlib import Path

import pytest

from tests.classification_fakes import identities_of, use_identity_cifar10
from train.api import recipes, registry
from train.api.config import validate
from train.api.tasks import register_builtin_tasks
from train.api.tasks.beit_tokenizer import BeitTokenizerConfig, BeitTokenizerTrainer
from train.common.seed import set_seeds

from . import _resume_protocol as protocol
from ._resume_protocol import csv_rows

CHILD = Path(__file__).resolve().parent / "_tokenizer_child.py"

BATCH, EPOCHS = 16, 3
#: Batches of one epoch: the fake cifar10's 54 fit rows over the batch (asserted against the data by ``test_the_sizes...``).
STEPS = 54 // BATCH
#: Where the two kills land: ``(epoch, batches finished in it)``.
KILL_1, KILL_2 = (1, 2), (2, 2)
#: ``completed_steps`` of each resume: the end of epoch 0, the end of epoch 1; and a copy 2 batches into epoch 1.
COMPLETED = {"p2": STEPS, "p3": 2 * STEPS}
MID_COMPLETED = STEPS + 2
FIRST = 3        # the batches whose digest is compared


def request(**changes):
    """The ``mapping`` of a tiny run on the fake CIFAR-10 (a dotted key sets a field of a group); ``out`` is added by the chain."""
    asked = {
        "model": "vq_vae_rotation/small", "data": {"dataset": "cifar10"}, "epochs": EPOCHS, "batch_size": BATCH,
        "transform_args": {"augment": False},
        "model_args": {"num_embeddings": 64, "embedding_dim": 8, "hidden_channels": 16, "num_res_blocks": 1},
        "analysis_args": {"model_analysis": False, "figures": False},
    }
    return protocol.merge_changes(asked, changes)


class Chain(protocol.ResumeChain):
    """Child processes on one run directory: what each ran, and what each kill left behind."""

    child_script = CHILD
    first = FIRST

    def mapping(self):
        return request(**self.changes)


@pytest.fixture(scope="module")
def world():
    """The identity fake CIFAR-10 and an empty registry holding every built-in family and task."""
    with pytest.MonkeyPatch.context() as patch:
        use_identity_cifar10(patch)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


def build(tmp_path):
    """``(trainer, resolved config, context, data)`` of this request in THIS process, seeded as the run seeds before it builds its data."""
    config = validate(BeitTokenizerConfig, {**request(), "out": str(tmp_path / "expected")})
    trainer = BeitTokenizerTrainer(config)
    family, variant = registry.parse_model_name(config.model, task=trainer.task)
    resolved, context = trainer._resolve(trainer.describe_data(config), family, variant)
    # `tf.data`'s shuffle reads the global seed when the dataset is made: seeded the same way, the parent's streams are the children's.
    set_seeds(resolved.seed)
    return trainer, resolved, context, trainer.build_data(resolved, context)


def stream_digests(ds):
    """The digest of each of the first ``FIRST`` batches of ``ds`` (the child's: the image bytes)."""
    import hashlib

    return [hashlib.sha256(image.numpy().tobytes()).hexdigest() for image, _ in ds.take(FIRST)]


def expected(tmp_path, counts):
    """``{count: digests}`` of the stream ``with_stream(count)`` gives (``None``: the fresh stream), and the select identities."""
    trainer, resolved, context, data = build(tmp_path)
    streams = {None: stream_digests(data.train_ds)}
    for count in counts:
        set_seeds(resolved.seed)
        streams[count] = stream_digests(data.with_stream(count).train_ds)
    return streams, sorted(identities_of("cifar10", data.select_ds, 0).tolist()), data


@pytest.fixture(scope="module")
def chain(world, tmp_path_factory):
    """Kill inside epoch 1, resume, kill inside epoch 2, resume, finish: three processes."""
    run = Chain(tmp_path_factory.mktemp("resume_tokenizer"), "beit_tokenizer")
    run.go("p1", f"@@AT {KILL_1[0]} {KILL_1[1]}", stop_at=list(KILL_1))
    run.go("p2", f"@@AT {KILL_2[0]} {KILL_2[1]}", resume=True, probe_restored=True, stop_at=list(KILL_2))
    run.go("p3", "@@DONE", resume=True, probe_restored=True)
    run.expected, run.select, run.data = expected(run.root, COMPLETED.values())
    return run


@pytest.fixture(scope="module")
def mid_epoch(world, tmp_path_factory):
    """A copy every 2 batches: killed after 2 batches of epoch 1, then resumed to the end."""
    run = Chain(tmp_path_factory.mktemp("resume_tokenizer_mid"), "beit_tokenizer", **{"checkpoint_args.backup_every_steps": 2})
    run.go("p1", "@@AT 1 2", stop_at=[1, 2])
    run.go("p2", "@@DONE", resume=True)
    run.expected, run.select, run.data = expected(run.root, (STEPS, MID_COMPLETED))
    return run


# ---------------------------------------------------------------------
# The kills are real and mid-epoch; the epochs and the step count are absolute
# ---------------------------------------------------------------------


def test_the_sizes_the_chain_counts_are_the_datas(chain):
    assert chain.data.steps_per_epoch == STEPS and chain.data.n_train == 54 and chain.data.request["batch_size"] == BATCH


def test_each_kill_lands_between_two_batches_of_an_epoch(chain):
    protocol.check_each_kill_lands_between_two_batches(chain, KILL_1, KILL_2)


def test_a_resume_starts_at_the_epoch_after_the_last_finished_one_and_the_killed_epoch_runs_again(chain):
    protocol.check_a_resume_starts_after_the_last_finished_epoch(chain)


def test_the_optimizer_step_count_is_epoch_times_steps_at_every_epoch_start_and_the_unkilled_total_at_the_end(chain):
    protocol.check_the_step_count_is_absolute(chain, STEPS, EPOCHS, KILL_1, KILL_2)


# ---------------------------------------------------------------------
# Rows, weights, summary
# ---------------------------------------------------------------------


def test_the_csv_rows_are_unique_and_complete_and_the_earlier_rows_are_kept_as_they_were(chain):
    protocol.check_the_csv_rows_are_unique_and_complete(chain, EPOCHS)


def test_the_weights_a_resume_restores_score_the_val_reconstruction_loss_the_killed_run_logged_for_them(chain):
    protocol.check_the_restored_weights_score_the_logged_value(chain, "reconstruction_loss")


def test_the_summary_says_the_run_resumed_and_finished_and_says_the_stream_is_a_fresh_one(chain):
    summary = json.loads((chain.run / "results_summary.json").read_text())
    assert summary["resumed_after_epochs"] == KILL_2[0] and summary["epochs_run"] == EPOCHS
    assert (chain.run / "final_model.keras").is_file() and (chain.run / "best_model.keras").is_file()
    assert summary["status"] == "ok" and summary["best_checkpoint_load_error"] is None
    assert any("fresh stream" in note and "resumed after epoch" in note for note in summary["notes"]), summary["notes"]


# ---------------------------------------------------------------------
# Not a replay, and the stream is the one of the settled count
# ---------------------------------------------------------------------


def test_the_first_batches_of_each_process_differ_from_the_process_before(chain):
    protocol.check_no_process_replays_the_one_before(chain)


def test_the_stream_of_a_process_is_the_one_of_its_settled_completed_steps(chain):
    # Process 1 is the fresh stream; process 2 and 3 are `with_stream` of the count their copy held. The parent built these in its own
    # process: equal digests mean the same shuffle and the same images.
    protocol.check_each_process_trains_on_the_stream_of_its_settled_count(chain, chain.expected, COMPLETED)


# ---------------------------------------------------------------------
# Disjointness
# ---------------------------------------------------------------------


def test_the_selection_set_is_the_same_in_every_process_and_the_one_the_parent_derives(chain):
    selects = {row["label"]: row["rows"] for row in chain.lines("select")}
    assert len(chain.select) == chain.data.n_select == 6, "the control: a real selection set"
    assert set(selects) == {"p1", "p2", "p3"}
    for label, rows in selects.items():
        assert rows == chain.select, f"{label} selects other rows: the split moved with the stream seed"


def test_no_train_row_of_any_process_is_a_selection_row(chain):
    select = set(chain.select)
    for label in ("p1", "p2", "p3"):
        seen = {row for batch in chain.lines("train_batch", label) for row in batch["rows"]}
        assert len(seen) >= STEPS * BATCH // 2, f"the control: {label} read a real part of the pool, not a few rows"
        assert not seen & select, f"{label} trained on selection rows {sorted(seen & select)}"
        assert seen <= set(range(60)) - select, "every train row is a row of the fit pool"


# ---------------------------------------------------------------------
# A kill inside an epoch, after a mid-epoch copy
# ---------------------------------------------------------------------


def test_a_resume_after_a_mid_epoch_copy_runs_only_the_batches_the_copy_had_not_taken(mid_epoch):
    protocol.check_a_mid_epoch_resume_runs_only_the_batches_the_copy_had_not_taken(mid_epoch, STEPS, EPOCHS, MID_COMPLETED)


def test_the_stream_after_a_mid_epoch_copy_is_the_one_of_the_count_the_copy_holds(mid_epoch):
    # 5 = 1 * 3 + (1 + 1): the batch term is part of the settled count. 3 (the epoch's start) is another stream.
    protocol.check_the_mid_epoch_stream(mid_epoch, mid_epoch.expected, STEPS, MID_COMPLETED)


def test_the_mid_epoch_resume_leaves_each_epoch_one_row_and_keeps_the_split_and_its_note_says_a_fresh_pass(mid_epoch):
    assert [row["epoch"] for row in csv_rows(mid_epoch.run / "training_log.csv")] == [str(e) for e in range(EPOCHS)]
    selects = [row["rows"] for row in mid_epoch.lines("select")]
    assert selects and all(rows == mid_epoch.select for rows in selects)
    notes = json.loads((mid_epoch.run / "results_summary.json").read_text())["notes"]
    killed = next(note for note in notes if note.startswith("epoch 1 was killed after "))
    assert "a fresh pass of the data" in killed and "unkilled run reads next" not in killed, "this stream is NOT the exact one"


# ---------------------------------------------------------------------
# The seam itself
# ---------------------------------------------------------------------


def test_resume_stream_returns_a_stream_that_is_not_the_fresh_one_and_keeps_what_the_sequence_checks(world, tmp_path):
    trainer, resolved, context, data = build(tmp_path)
    replacement = trainer.resume_stream(resolved, data, STEPS + 1)
    assert replacement is not data and replacement.train_ds is not data.train_ds, "None or `data` would be the replay"
    assert (replacement.steps_per_epoch, replacement.n_train, replacement.n_select, replacement.n_test) == (
        data.steps_per_epoch, data.n_train, data.n_select, data.n_test)
    assert replacement.seed == data.seed and replacement.stream_seed != data.stream_seed
    assert sorted(identities_of("cifar10", replacement.select_ds, 0).tolist()) == sorted(identities_of("cifar10", data.select_ds, 0).tolist())
    set_seeds(resolved.seed)
    assert stream_digests(replacement.train_ds) != stream_digests(data.train_ds)
    assert trainer.resume_order and "fresh stream" in trainer.resume_order and trainer.resume_stream_exact is False
