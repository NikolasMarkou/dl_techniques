"""``resume_interrupted=true`` for the ``depth_estimation`` task, with a REAL SIGKILL.

The task is a stream task: its train stream is an infinite ``PyDataset`` whose batch ``i`` is a pure function of the stream
seed and ``i``, an epoch is ``steps_per_epoch`` batches by ``fit_kwargs``, and a resumed ``fit`` asks the stream for index 0
again, which would replay the first batches. What a resume could get wrong, and the stock resume tests cannot see:

* **a resume must not replay** (D-004, D-014): ``resume_stream`` returns ``DepthBatches.with_stream(completed_steps)``, so
  process 2 trains on another stream than process 1 and process 3 on another than process 2;
* **a resume must not move the fixed crops** (D-004, D-005): the selection and test crops are keyed on ``(seed, salt, pool
  index)`` in scenes that never train, so they are the same in every process, the weights a resume restores score the
  ``val_loss`` the killed run logged for them (the crops are in memory, so the number is exact), and no train draw of any
  process touches a selection or test scene;
* **the optimizer is restored, not rebuilt** (D-019): the backup is loaded into an optimizer that is already built, so its
  step count and every slot variable are the killed run's, and on a pretrained start it is the ``LRMultiplierAdamW`` with
  the DPT head at 10x in every process.

One chain of three child processes (``_depth_child.py``; the marker protocol of ``test_backup_every_steps.run_child``, which
SIGKILLs the RECORDED PID at a ``@@AT`` marker): ``vit_s`` at 28 px on the synthetic MegaDepth tree of the data library's
tests, 3 epochs of 3 batches of 4. Process 1 is killed after 2 batches of epoch 1 (the backup holds the end of epoch 0),
process 2 resumes, finishes epoch 1 and is killed after 2 batches of epoch 2, process 3 resumes and finishes.

The start. The chain starts from the official DINOv2 weights when the vit_s file is on this machine (the 10x table is
empty from random weights, so only that start can show the table restored), and from random weights otherwise, where the
table test is skipped by name and every other test runs. One chain, not two: a vit_s process is the unit of cost (the
first ``fit`` traces the train step twice, about 21,000 graph ops each, on pure-python protobuf), and the official start
costs a weights read on top of it. TensorBoard, the figures, the analysis, the final model and the warm-up are off: the
chain checks none of them, and each costs seconds per process.

What is read: the trace the children append to (each batch the train stream handed out, by stream seed and index, with
its digest and every file its draws tried; the fixed sets of each data object; the optimizer at each process's first
epoch begin and at every epoch end) and the files. NOT checked: a resumed run is not the unkilled run's trajectory, a
mid-epoch copy (``backup_every_steps``; ``test_forecast_resume.py`` holds the shared rule), a GPU. The tests write only
under ``tmp_path``.
"""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from train.api import depth_models
from train.api.errors import ConfigError
from train.common.pretrain_data import derive_stream_seed

from . import _resume_protocol as protocol
from .test_depth_task import SIZE, megadepth_tree, prepared, world  # noqa: F401  (fixtures)

CHILD = Path(__file__).resolve().parent / "_depth_child.py"

BATCH, EPOCHS, STEPS = 4, 3, 3
#: Two batches of fixed crops each, every train file of the tree, the stream read in the caller (no worker processes).
DATA = {"train_split": 0.6, "n_select": 2 * BATCH, "n_test": 2 * BATCH, "layout_samples": 16, "steps_per_epoch": STEPS,
        "workers": 0, "max_train_files": None}
#: Where the two kills land: ``(epoch, batches finished in it)``.
KILL_1, KILL_2 = (1, 2), (2, 2)
#: ``completed_steps`` of each resume: the end of epoch 0, the end of epoch 1.
COMPLETED = {"p2": STEPS, "p3": 2 * STEPS}
FIRST = 3  # the train batches whose digest is compared
HEAD_MULTIPLIER = 10.0


def sha256_of(*arrays) -> str:
    """The digest the child writes (``_depth_child.sha256_of``): the bytes of each array, in order."""
    return hashlib.sha256(b"".join(array.tobytes() for array in arrays)).hexdigest()


def official_start() -> bool:
    """Whether the official DINOv2 vit_s file is on this machine (the start of the chain)."""
    try:
        depth_models.check_weights("vit_s", {})
    except ConfigError:
        return False
    return True


def request(tree, pretrained):
    """The ``mapping`` of the chain; ``out`` and ``resume_interrupted`` are added by the chain."""
    return {
        "model": "depth_anything/vit_s", "data": {"root": str(tree), **DATA}, "epochs": EPOCHS, "batch_size": BATCH,
        "model_args": {} if pretrained else {"pretrained": False}, "transform_args": {"image_size": SIZE},
        "analysis_args": {"model_analysis": False, "figures": False}, "logger_args": {"tensorboard": None},
        "checkpoint_args": {"save_last": False}, "optim_args": {"warmup_epochs": 0},
    }


class Chain(protocol.ResumeChain):
    """Child processes on one run directory: what each ran, and what each kill left behind."""

    child_script = CHILD
    first = FIRST

    def __init__(self, root: Path, tree: Path, pretrained: bool) -> None:
        super().__init__(root, "depth")
        self.tree, self.pretrained = tree, pretrained

    def mapping(self):
        return request(self.tree, self.pretrained)

    def fit_stream_seed(self, label: str) -> int:
        """The stream seed of the data the process's ``fit`` read: the last data object it built or was handed."""
        return self.lines("select", label)[-1]["stream_seed"]

    def handed_out(self, label: str):
        """``{index: {digests}}`` of the batches the fit's stream handed out (an index read twice holds one digest)."""
        seed, seen = self.fit_stream_seed(label), {}
        for row in self.lines("train_batch", label):
            if row["stream_seed"] == seed:
                seen.setdefault(row["index"], set()).add(row["digest"])
        return seen

    def digests(self, label: str):
        """The digests of the fit's first ``FIRST`` batches, by global index (Keras reads the first batches once for the
        shape before the fit reads them again, indices 0 and 1 in the measured trace, and the queue reads ahead), so the
        order is the stream's and not the order the reads reached the trace."""
        seen = self.handed_out(label)
        return [sorted(seen[index])[0] for index in range(self.first) if index in seen]

    def optimizer(self, label: str, at: str):
        """The ``optimizer`` lines of ``label`` written at ``at`` (``begin`` or ``end``)."""
        return [row for row in self.lines("optimizer", label) if row["at"] == at]


def stream_digests(stream):
    """The digest of each of the first ``FIRST`` batches of a train stream (the child's: x bytes then y bytes)."""
    return [sha256_of(*stream[index]) for index in range(FIRST)]


def expected(root, tree, pretrained, counts):
    """The data the parent builds itself, and ``{count: digests}`` of ``with_stream(count)`` (``None``: the fresh stream)."""
    trainer, resolved, context = prepared(Path(root) / "expected", tree, **request(tree, pretrained))
    data = trainer.build_data(resolved, context)
    streams = {None: stream_digests(data.train_ds)}
    for count in counts:
        streams[count] = stream_digests(data.with_stream(count).train_ds)
    return SimpleNamespace(streams=streams, data=data, seed=resolved.seed)


@pytest.fixture(scope="module")
def chain(world, megadepth_tree, tmp_path_factory):
    """Kill inside epoch 1, resume, kill inside epoch 2, resume, finish: three processes."""
    run = Chain(tmp_path_factory.mktemp("resume_depth"), megadepth_tree, official_start())
    run.go("p1", f"@@AT {KILL_1[0]} {KILL_1[1]}", stop_at=list(KILL_1))
    run.go("p2", f"@@AT {KILL_2[0]} {KILL_2[1]}", resume=True, probe_restored=True, stop_at=list(KILL_2))
    run.go("p3", "@@DONE", resume=True, probe_restored=True)
    run.expect = expected(run.root, megadepth_tree, run.pretrained, COMPLETED.values())
    return run


# ---------------------------------------------------------------------
# The kills are real and mid-epoch; the epochs and the step count are absolute
# ---------------------------------------------------------------------


def test_the_sizes_the_chain_counts_are_the_data_s(chain):
    data = chain.expect.data
    assert (data.steps_per_epoch, data.n_train, data.batch_size) == (STEPS, STEPS * BATCH, BATCH)
    assert data.n_select == data.n_test == 2 * BATCH and data.image_size == SIZE and data.workers == 0
    resolved = json.loads((chain.run / "config.json").read_text())
    assert resolved["used"]["model_args"] == {"pretrained": chain.pretrained}, "the chain ran the start it says it ran"


def test_each_kill_lands_between_two_batches_of_an_epoch(chain):
    protocol.check_each_kill_lands_between_two_batches(chain, KILL_1, KILL_2)


def test_a_resume_starts_at_the_epoch_after_the_last_finished_one_and_the_killed_epoch_runs_again(chain):
    protocol.check_a_resume_starts_after_the_last_finished_epoch(chain)


def test_the_optimizer_step_count_is_epoch_times_steps_at_every_epoch_start_and_the_unkilled_total_at_the_end(chain):
    protocol.check_the_step_count_is_absolute(chain, STEPS, EPOCHS, KILL_1, KILL_2)


# ---------------------------------------------------------------------
# Rows, weights, optimizer, summary
# ---------------------------------------------------------------------


def test_the_csv_rows_are_unique_and_complete_and_the_earlier_rows_are_kept_as_they_were(chain):
    protocol.check_the_csv_rows_are_unique_and_complete(chain, EPOCHS)


def test_the_weights_a_resume_restores_score_the_val_loss_the_killed_run_logged_for_them(chain):
    protocol.check_the_restored_weights_score_the_logged_value(chain, "loss")


def test_the_optimizer_a_resume_restores_is_the_killed_run_s_variable_for_variable(chain):
    # A fresh fit builds its optimizer inside the first train step, so before it the optimizer holds its step count only.
    (fresh,) = chain.optimizer("p1", "begin")
    assert (fresh["iterations"], fresh["variables"], fresh["nonzero"]) == (0, 1, 0), "the control: a fresh optimizer"
    for label, killed, epoch in (("p2", "p1", KILL_1[0] - 1), ("p3", "p2", KILL_2[0] - 1)):
        (saved,) = [row for row in chain.optimizer(killed, "end") if row["epoch"] == epoch]
        (restored,) = chain.optimizer(label, "begin")
        assert restored["epoch"] == epoch + 1 and restored["iterations"] == (epoch + 1) * STEPS
        assert restored["variables"] == saved["variables"] > 100, f"{label}: the optimizer was not built before the restore"
        # Every slot variable, not only the step count: an optimizer built AFTER the restore takes none of the copy's
        # variables and starts from zeros (D-019 measured 0 of 469 restored that way), which `nonzero` would show.
        assert restored["nonzero"] == saved["nonzero"] > 0, f"{label} restored an optimizer of zeros"
        assert restored["digest"] == saved["digest"], f"{label} holds other optimizer values than {killed} backed up"
    ends = [row["digest"] for row in chain.lines("optimizer") if row["at"] == "end"]
    assert len(ends) == EPOCHS and len(set(ends)) == EPOCHS, "the control: the optimizer moved every epoch"


def test_the_optimizer_of_every_process_is_lr_multiplier_adamw_with_the_head_at_10x(chain):
    if not chain.pretrained:
        pytest.skip("the chain started from random weights (no official DINOv2 vit_s file here): the head table is empty")
    rows = chain.lines("optimizer")
    assert [row["label"] for row in rows] == ["p1", "p1", "p2", "p2", "p3", "p3"]
    head = rows[0]["head"]
    assert len(head) > 10, "the control: the DPT head has trainable variables"
    for row in rows:
        assert row["cls"] == "LRMultiplierAdamW", f"{row['label']} trained with {row['cls']}"
        assert row["table"] == {path: HEAD_MULTIPLIER for path in head}, f"{row['label']} has another table"
        assert row["head"] == head and row["table_paths_are_the_heads"], (
            f"{row['label']}: the table's keys are not the paths of this process's head variables")


def test_the_summary_says_the_run_resumed_and_finished(chain):
    summary = json.loads((chain.run / "results_summary.json").read_text())
    assert summary["resumed_after_epochs"] == KILL_2[0] and summary["epochs_run"] == EPOCHS
    assert summary["status"] == "ok" and (chain.run / "best_model.keras").is_file()


# ---------------------------------------------------------------------
# Not a replay, and the stream is the one of the settled count
# ---------------------------------------------------------------------


def test_an_index_the_stream_hands_out_twice_is_the_same_batch(chain):
    for label in ("p1", "p2", "p3"):
        seen = chain.handed_out(label)
        assert set(range(STEPS)) <= set(seen), f"the control: {label} read the first batches of its stream"
        assert all(len(digests) == 1 for digests in seen.values()), f"{label}: one index gave two batches"


def test_the_first_batches_of_each_process_differ_from_the_process_before(chain):
    protocol.check_no_process_replays_the_one_before(chain)


def test_the_stream_of_a_process_is_the_one_of_its_settled_completed_steps(chain):
    # Process 1 is the fresh stream; process 2 and 3 are `with_stream` of the count their copy held. The parent built these
    # in its own process: equal digests mean the same files, crops and flips, standardized the same way.
    protocol.check_each_process_trains_on_the_stream_of_its_settled_count(chain, chain.expect.streams, COMPLETED)


# ---------------------------------------------------------------------
# The fixed crops do not move; no train draw reaches a selection or test scene
# ---------------------------------------------------------------------


def test_the_fixed_crops_are_the_same_in_every_process_and_the_ones_the_parent_derives(chain):
    data, seed = chain.expect.data, chain.expect.seed
    selects = chain.lines("select")
    assert [row["label"] for row in selects] == ["p1", "p2", "p2", "p3", "p3"], "a build per process and one per resume"
    wanted = {"select": sha256_of(data.x_select, data.y_select), "test": sha256_of(data.x_test, data.y_test),
              "layout": sha256_of(data.layout), "fingerprint": data.fingerprint}
    for row in selects:
        assert {key: row[key] for key in wanted} == wanted, f"{row['label']}: the fixed sets moved with the stream seed"
        assert row["select_files"] == [os.path.relpath(path, chain.tree) for path in data.select_files]
        assert row["test_files"] == [os.path.relpath(path, chain.tree) for path in data.test_files]
    streams = [(row["label"], row["stream_seed"]) for row in selects]
    assert streams == [("p1", seed), ("p2", seed), ("p2", derive_stream_seed(seed, COMPLETED["p2"])),
                       ("p3", seed), ("p3", derive_stream_seed(seed, COMPLETED["p3"]))]


def test_no_train_draw_of_any_process_reads_a_file_of_a_selection_or_test_scene(chain):
    data = chain.expect.data
    train, held = set(data.train_scenes), set(data.select_scenes) | set(data.test_scenes)
    assert train and data.select_scenes and data.test_scenes and not train & held, "the control: three disjoint groups"
    scenes = set()
    for label in ("p1", "p2", "p3"):
        batches = chain.lines("train_batch", label)
        assert len(batches) >= STEPS, f"the control: {label} read a real part of the stream"
        tried = [path for batch in batches for sample in batch["files"] for path in sample]
        assert len(tried) >= STEPS * BATCH
        scenes |= {Path(path).parts[0] for path in tried}
    assert scenes <= train, f"a train draw read a file of {sorted(scenes - train)}, a selection or test scene"
    assert len(scenes) > 1, "the control: the draws reach more than one train scene"
    for row in chain.lines("select"):
        assert {Path(path).parts[0] for path in row["select_files"]} <= set(data.select_scenes)
        assert {Path(path).parts[0] for path in row["test_files"]} <= set(data.test_scenes)
