"""``resume_interrupted=true`` for ``dino``, with a REAL SIGKILL: a resume trains on another stream, restores the centre and never replays.

A DINO run has three pieces of state that the weights file does not carry or that a restart could replay: the infinite train stream (a resumed
``fit`` starts a fresh iterator), the loss's centre (a variable the loss moves in ``call``; written to a file at each epoch end, D-027), and the
teacher EMA's position (read from ``optimizer.iterations``, which the weights file restores, D-024). One chain of two child processes
(``_dino_child.py``, the marker protocol of ``test_backup_every_steps.run_child``, which SIGKILLs the RECORDED PID at a ``@@AT`` marker) on 2 epochs
of 3 batches: process 1 is killed after 2 batches of epoch 1 (the backup holds the end of epoch 0), process 2 resumes and finishes. What this file
pins:

* **a resume must not replay**: process 2 trains on the stream ``DinoBatches.with_stream`` gives for exactly the count its copy held, not on process 1's;
* **a resume must not leak**: the selection rows and the two probe streams (bank and queries) are the same in both processes;
* **the weights, the centre and the step are the backup's**: the digests process 2 starts from are process 1's at the end of epoch 0, and the centre is
  the file's; a restored run's EMA updated at every batch (a private counter restarting at 0 would replay the warmup and update none);
* **the CSV has one row per epoch** with the probe columns, and the rows of the first process are kept as they were;
* ``resume_stream`` returns a stream and not ``data``, and keeps what the sequence checks.

NOT checked: a resumed run is not the unkilled run's trajectory (``resume_stream_exact`` is false), a GPU, a kill during a weight write
(``test_backup_every_steps``). The tests write only under ``tmp_path``.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests.classification_fakes import identities_of, use_identity_cifar10
from train.api import recipes, registry
from train.api.config import validate
from train.api.tasks import register_builtin_tasks
from train.api.tasks.dino import CENTRE_FILE, PROBE_KEYS, DinoConfig, DinoTrainer
from train.common.seed import set_seeds

from . import _resume_protocol as protocol
from ._resume_protocol import csv_rows

CHILD = Path(__file__).resolve().parent / "_dino_child.py"

BATCH, EPOCHS = 16, 2
#: Batches of one epoch: the fake cifar10's 54 fit rows over the batch (asserted against the data by the first test).
STEPS = 54 // BATCH
KILL = (1, 2)
COMPLETED = {"p2": STEPS}
FIRST = 3        # the batches whose digest is compared


def request(**changes):
    """The ``mapping`` of a tiny run on the fake CIFAR-10; ``out`` is added by the chain."""
    asked = {
        "model": "dino_v1/tiny", "data": {"dataset": "cifar10", "patch_size": 8, "n_local_crops": 2}, "epochs": EPOCHS, "batch_size": BATCH,
        "model_args": {"dino_out_dim": 64}, "optim_args": {"lr": 1e-3, "warmup_epochs": 0},
        "analysis_args": {"model_analysis": False, "figures": False},
    }
    return protocol.merge_changes(asked, changes)


class Chain(protocol.ResumeChain):
    child_script = CHILD
    first = FIRST

    def mapping(self):
        return request(**self.changes)


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        use_identity_cifar10(patch)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


def build(tmp_path):
    """``(trainer, resolved config, context, data)`` of this request in THIS process, seeded as the run seeds before it builds its data."""
    config = validate(DinoConfig, {**request(), "out": str(tmp_path / "expected")})
    trainer = DinoTrainer(config)
    family, variant = registry.parse_model_name(config.model, task=trainer.task)
    resolved, context = trainer._resolve(trainer.describe_data(config), family, variant)
    set_seeds(resolved.seed)
    return trainer, resolved, context, trainer.build_data(resolved, context)


def stream_digests(ds):
    return [hashlib.sha256(np.asarray(views).tobytes()).hexdigest() for views, _ in ds.take(FIRST)]


@pytest.fixture(scope="module")
def chain(world, tmp_path_factory):
    """Kill inside epoch 1 after 2 batches, resume, finish: two processes."""
    run = Chain(tmp_path_factory.mktemp("resume_dino"), "dino")
    run.go("p1", f"@@AT {KILL[0]} {KILL[1]}", stop_at=list(KILL))
    run.go("p2", "@@DONE", resume=True)
    trainer, resolved, context, data = build(run.root)
    run.expected = {None: stream_digests(data.train_ds)}
    set_seeds(resolved.seed)
    run.expected[STEPS] = stream_digests(data.with_stream(STEPS).train_ds)
    run.data = data
    run.select = sorted(identities_of("cifar10", data.select_ds, 0).tolist())
    return run


def test_the_sizes_the_chain_counts_are_the_datas(chain):
    assert chain.data.steps_per_epoch == STEPS and chain.data.n_train == 54 and chain.data.n_select == 6


def test_the_kill_lands_between_two_batches_of_epoch_one_and_the_resume_starts_there(chain):
    last = chain.lines("batch", "p1")[-1]
    assert (last["epoch"], last["batch"]) == (KILL[0], KILL[1] - 1) and not chain.lines("train_end", "p1")
    epochs = {label: [row["epoch"] for row in chain.lines("epoch_begin", label)] for label in ("p1", "p2")}
    assert epochs == {"p1": [0, 1], "p2": [1]}


def test_the_step_count_is_absolute_in_both_processes(chain):
    for row in chain.lines("epoch_begin"):
        assert row["iterations"] == row["epoch"] * STEPS, row
    assert [row["iterations"] for row in chain.lines("train_end")] == [EPOCHS * STEPS]
    for row in chain.lines("batch"):
        assert row["before"] == row["epoch"] * STEPS + row["batch"], row


def test_the_csv_has_one_row_per_epoch_with_the_probe_columns_and_the_first_rows_are_kept(chain):
    final = csv_rows(chain.run / "training_log.csv")
    assert [row["epoch"] for row in final] == [str(e) for e in range(EPOCHS)]
    assert set(PROBE_KEYS) <= set(final[0]) and not [name for name in final[0] if name.startswith("val_")]
    assert [row["epoch"] for row in chain.after["p1"]["csv"]] == ["0"] and final[:1] == chain.after["p1"]["csv"], "a resume rewrote a finished epoch's row"


def test_the_summary_says_the_run_resumed_and_finished_and_says_the_stream_is_a_fresh_one(chain):
    summary = json.loads((chain.run / "results_summary.json").read_text())
    assert summary["resumed_after_epochs"] == KILL[0] and summary["epochs_run"] == EPOCHS and summary["status"] == "ok"
    assert (chain.run / "final_model.keras").is_file() and (chain.run / "best_model.keras").is_file() and summary["best_checkpoint_load_error"] is None
    assert any("fresh stream" in note and "resumed after epoch" in note for note in summary["notes"]), summary["notes"]


def test_the_first_batches_of_process_two_differ_from_process_one_and_are_the_stream_of_its_settled_count(chain):
    one, two = chain.digests("p1"), chain.digests("p2")
    assert len(one) == len(two) == FIRST and len(set(one)) == FIRST and not set(one) & set(two), "the control: real batches, no replay"
    protocol.check_each_process_trains_on_the_stream_of_its_settled_count(chain, chain.expected, COMPLETED)   # M36: `resume_stream` returns `data`


def test_the_selection_rows_and_both_probe_streams_are_the_same_in_every_process(chain):
    for kind in ("select", "bank", "query"):
        rows = {row["label"]: row["rows"] for row in chain.lines(kind)}
        assert set(rows) == {"p1", "p2"} and rows["p1"] == rows["p2"] and len(rows["p1"]) >= 6, kind
    assert {row["label"]: row["rows"] for row in chain.lines("select")}["p1"] == chain.select
    bank, select = {row for row in chain.lines("bank", "p1")[0]["rows"]}, set(chain.select)
    assert not bank & select, "the bank is fit rows and the queries are selection rows"


def test_the_weights_the_centre_and_the_step_a_resume_starts_from_are_the_backups_and_the_centre_files(chain):
    ended = {row["epoch"]: row for row in chain.lines("epoch_state", "p1")}
    started = chain.lines("restored", "p2")[0]
    assert set(ended) == {0}, "process 1 was killed inside epoch 1"
    assert started["weights"] == ended[0]["weights"], "the weights are the end of epoch 0 (student, teacher and head)"
    assert started["centre"] == ended[0]["centre"] and started["iterations"] == STEPS
    saved = np.load(chain.run / "checkpoints" / CENTRE_FILE)
    after_p2 = {row["epoch"]: row for row in chain.lines("epoch_state", "p2")}
    assert hashlib.sha256(np.ascontiguousarray(saved).tobytes()).hexdigest() == after_p2[1]["centre"], "the file holds the last epoch end's centre"
    assert after_p2[1]["centre"] != ended[0]["centre"], "the control: the centre moved while process 2 trained"
    assert started["centre"] != hashlib.sha256(np.zeros_like(saved).tobytes()).hexdigest(), "the control: it is not a zero centre"      # M15


def test_the_teacher_ema_of_a_resumed_run_updates_at_every_batch_because_its_position_is_the_optimizer_step(chain):
    updates = {row["label"]: row for row in chain.lines("ema")}
    # The warmup is one epoch (3 steps). Process 2 starts at step 3 and its 3 batches are steps 4 to 6, all after the warmup: 3 updates. A callback
    # that kept its own counter would restart at 0 in every process, replay the warmup freeze and update NONE of them (M10).
    assert updates["p2"]["updates"] == STEPS and updates["p2"]["traces"] == 1 and "p1" not in updates


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
