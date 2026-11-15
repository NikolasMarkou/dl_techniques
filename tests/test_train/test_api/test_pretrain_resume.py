"""``resume_interrupted=true`` for the ``pretrain`` task, with a REAL SIGKILL, per family (plan 46e5318f step 13).

The task is a stream task: its train stream is infinite, an epoch is ``steps_per_epoch`` batches by ``fit_kwargs``, and a
resumed ``fit`` starts a fresh iterator, which would replay the stream from its first element. The stock resume tests
(classification, denoising) cannot see the three things this task adds, because none of them has such a stream:

* **a resume must not replay** (D-004): the stream seed is derived from ``(seed, completed_steps)``, so process 2 trains on
  another stream than process 1 and process 3 on another than process 2;
* **a resume must not leak** (D-004, D-005): the split keeps the split seed, so no train row of any process is a selection
  row, and the selection set is the same in every process; the select masks are keyed on the split seed, so the weights a
  resume restores score the ``val_loss`` the killed run logged for them;
* **the settled count reaches the stream**: ``resume_stream`` is given ``completed_steps``, the count the copy holds
  (``start * steps + batch + 1`` for a copy taken inside an epoch), so the stream of a process is the one
  ``PretrainBatches.with_stream`` gives for exactly that count.

Each family (``energy_transformer`` with the fake CIFAR-10, ``beit`` with a tiny tokenizer) runs one chain of three child
processes (``_pretrain_child.py``; the marker protocol of ``test_backup_every_steps.run_child``, which SIGKILLs the RECORDED
PID at a ``@@AT`` marker) on 3 epochs of 3 batches: process 1 is killed after 2 batches of epoch 1 (the backup holds the end
of epoch 0), process 2 resumes, finishes epoch 1 and is killed after 2 batches of epoch 2, process 3 resumes and finishes.
(Killing the same epoch twice would resume at the same count and so on the same stream, by design: the stream is a function
of the count.) One more run (``energy_transformer``, ``checkpoint_args.backup_every_steps=2``) is killed INSIDE epoch 1 right
after a mid-epoch copy: the resume runs the one batch the copy had not taken, at ``completed_steps = 3 + 2``.

The data is un-augmented: the augmentation seeds are re-randomised on every iterator, so two iterators of one stream differ in
their pixels by themselves, and the first batches of a replay would not be equal. The shuffle and the train mask are keyed on
the stream seed, so the first iterator of a stream is the same in every process; the parent builds that stream itself
(``expected``) and the children are compared with it, and with each other. What is read: the trace the children append to (the
digest and the row identities of every train batch a fit read, the identities of the selection set, one line per batch and per
epoch) and the files. NOT checked: a resumed run is not the unkilled run's trajectory, a GPU, a kill during a weight write
(``test_backup_every_steps``). The tests write only under ``tmp_path``.
"""

import json
from pathlib import Path

import pytest

from tests.classification_fakes import identities_of
from train.common import pretrain_run
from train.common.seed import set_seeds

from . import _resume_protocol as protocol
from ._resume_protocol import csv_rows
from .test_pretrain_task import (  # noqa: F401  (``world`` is a fixture)
    BEIT_MODEL, BEIT_SMALL, ET_MODEL, LOW_FLOOR, TOKENIZER_KEY, batch_digest, make_tokenizer, prepared, world,
)

CHILD = Path(__file__).resolve().parent / "_pretrain_child.py"

BATCH, EPOCHS = 16, 3
#: Batches of one epoch: the fake cifar10's 54 fit rows over the batch (asserted against the data by ``test_the_sizes...``).
STEPS = 54 // BATCH
FAMILIES = ("energy_transformer", "beit")
#: Where the two kills land: ``(epoch, batches finished in it)``.
KILL_1, KILL_2 = (1, 2), (2, 2)
#: ``completed_steps`` of each resume: the end of epoch 0, the end of epoch 1; and a copy 2 batches into epoch 1.
COMPLETED = {"p2": STEPS, "p3": 2 * STEPS}
MID_COMPLETED = STEPS + 2
FIRST = 3        # the batches whose digest is compared


def request(family, tokenizer, **changes):
    """The ``mapping`` of a run of ``family`` (a dotted key sets a field of a group); ``out`` is added by the chain."""
    asked = {
        "model": ET_MODEL if family == "energy_transformer" else BEIT_MODEL,
        "data": {"dataset": "cifar10"}, "epochs": EPOCHS, "batch_size": BATCH,
        "transform_args": {"augment": False},
        "model_args": {"num_steps": 2} if family == "energy_transformer" else {TOKENIZER_KEY: tokenizer, **BEIT_SMALL},
        "analysis_args": {"model_analysis": False, "figures": family == "energy_transformer"},
    }
    return protocol.merge_changes(asked, changes)


class Chain(protocol.ResumeChain):
    """Child processes on one run directory: what each ran, and what each kill left behind."""

    child_script = CHILD
    first = FIRST

    def __init__(self, root: Path, family: str, tokenizer: str, **changes) -> None:
        super().__init__(root, family, **changes)
        self.tokenizer = tokenizer

    @property
    def family(self) -> str:
        return self.name

    def mapping(self):
        return request(self.family, self.tokenizer, **self.changes)

    def child_spec(self):
        return {"floor": LOW_FLOOR if self.family == "beit" else None}


def stream_digests(ds):
    """The digest of each of the first ``FIRST`` batches of ``ds`` (the child's: image bytes then mask bytes)."""
    out = []
    for (image, mask), _, _ in ds.take(FIRST):
        out.append(batch_digest(image, mask))
    return out


def expected(tmp_path, family, tokenizer, counts):
    """``{count: digests}`` of the stream ``with_stream(count)`` gives (``None``: the fresh stream), and the select identities."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pretrain_run, "MIN_TOKENIZER_PERPLEXITY", LOW_FLOOR)
        asked = request(family, tokenizer)
        trainer, resolved, context = prepared(
            tmp_path / "expected", model=asked["model"], model_args=asked["model_args"], batch_size=BATCH,
            **{"transform_args.augment": False})
        # The run seeds TensorFlow before it builds its data (`set_seeds(config.seed)`), and `tf.data`'s shuffle reads the
        # global seed when the dataset is made: seeded the same way, the parent's streams are the children's.
        set_seeds(resolved.seed)
        data = trainer.build_data(resolved, context)
        streams = {None: stream_digests(data.train_ds)}
        for count in counts:
            set_seeds(resolved.seed)
            streams[count] = stream_digests(data.with_stream(count).train_ds)
        return streams, sorted(identities_of("cifar10", data.select_ds, 0).tolist()), data


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    return make_tokenizer(tmp_path_factory.mktemp("tokenizer") / "tokenizer.keras")


@pytest.fixture(scope="module", params=FAMILIES)
def chain(request, world, tokenizer, tmp_path_factory):
    """Kill inside epoch 1, resume, kill inside epoch 2, resume, finish: three processes of one family."""
    family = request.param
    run = Chain(tmp_path_factory.mktemp(f"resume_{family}"), family, tokenizer)
    run.go("p1", f"@@AT {KILL_1[0]} {KILL_1[1]}", stop_at=list(KILL_1))
    run.go("p2", f"@@AT {KILL_2[0]} {KILL_2[1]}", resume=True, probe_restored=True, stop_at=list(KILL_2))
    run.go("p3", "@@DONE", resume=True, probe_restored=True)
    run.expected, run.select, run.data = expected(run.root, family, tokenizer, COMPLETED.values())
    return run


@pytest.fixture(scope="module")
def mid_epoch(world, tokenizer, tmp_path_factory):
    """``energy_transformer`` with a copy every 2 batches: killed after 2 batches of epoch 1, then resumed to the end."""
    run = Chain(tmp_path_factory.mktemp("resume_mid_epoch"), "energy_transformer", tokenizer,
                **{"checkpoint_args.backup_every_steps": 2})
    run.go("p1", "@@AT 1 2", stop_at=[1, 2])
    run.go("p2", "@@DONE", resume=True)
    run.expected, run.select, run.data = expected(run.root, "energy_transformer", tokenizer, (STEPS, MID_COMPLETED))
    return run


# ---------------------------------------------------------------------
# The kills are real and mid-epoch; the epochs and the step count are absolute
# ---------------------------------------------------------------------


def test_the_sizes_the_chain_counts_are_the_data_s(chain):
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


def test_the_weights_a_resume_restores_score_the_val_loss_the_killed_run_logged_for_them(chain):
    protocol.check_the_restored_weights_score_the_logged_value(chain, "loss")


def test_the_summary_says_the_run_resumed_and_finished(chain):
    summary = json.loads((chain.run / "results_summary.json").read_text())
    assert summary["resumed_after_epochs"] == KILL_2[0] and summary["epochs_run"] == EPOCHS
    assert (chain.run / "final_model.keras").is_file() and not (chain.run / "task.json").exists()


def test_the_energy_trace_has_one_header_and_one_row_per_finished_epoch(chain):
    path = chain.run / "visualizations" / "energy_trace.csv"
    if chain.family != "energy_transformer":
        assert not path.exists()
        return
    lines = path.read_text().splitlines()
    assert [line.split(",")[0] for line in lines] == ["epoch"] + [str(e) for e in range(EPOCHS)], (
        "three processes appended to one file: a header per process, or a row twice, shows here")


# ---------------------------------------------------------------------
# Not a replay, and the stream is the one of the settled count
# ---------------------------------------------------------------------


def test_the_first_batches_of_each_process_differ_from_the_process_before(chain):
    protocol.check_no_process_replays_the_one_before(chain)


def test_the_stream_of_a_process_is_the_one_of_its_settled_completed_steps(chain):
    # Process 1 is the fresh stream; process 2 and 3 are `with_stream` of the count their copy held. The parent built these
    # in its own process: equal digests mean the same shuffle, the same images, the same masks.
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


def test_the_mid_epoch_resume_leaves_each_epoch_one_row_and_keeps_the_split(mid_epoch):
    assert [row["epoch"] for row in csv_rows(mid_epoch.run / "training_log.csv")] == [str(e) for e in range(EPOCHS)]
    selects = [row["rows"] for row in mid_epoch.lines("select")]
    assert selects and all(rows == mid_epoch.select for rows in selects)
    lines = (mid_epoch.run / "visualizations" / "energy_trace.csv").read_text().splitlines()
    assert [line.split(",")[0] for line in lines] == ["epoch"] + [str(e) for e in range(EPOCHS)]
