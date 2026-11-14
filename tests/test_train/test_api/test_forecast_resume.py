"""``resume_interrupted=true`` for the ``time_series_forecasting`` task, with a REAL SIGKILL (plan 3a8319ef step 14).

The task is a stream task: its train stream is infinite, an epoch is ``steps_per_epoch`` batches by ``fit_kwargs``, and a
resumed ``fit`` starts a fresh iterator, which would replay the stream from its first element. The data layer adds three
things a resume could get wrong, and the stock resume tests (classification, denoising) cannot see any of them:

* **a resume must not replay** (D-004, D-013): the stream seed is derived from ``(seed, completed_steps)``, so process 2
  trains on another stream than process 1 and process 3 on another than process 2;
* **a resume must not leak** (D-004, D-006): the bank and the select and test windows keep the SPLIT seed, so no train
  window of any process lies in the select segment, the selection set is the same in every process, and the weights a resume
  restores score the ``val_mae`` the killed run logged for them (the windows are fixed, so the number is exact);
* **the settled count reaches the stream**: ``resume_stream`` is given ``completed_steps``, the count the copy holds
  (``start * steps + batch + 1`` for a copy taken inside an epoch), so the stream of a process is the one
  ``ForecastBatches.with_stream`` gives for exactly that count.

Three families run one chain of three child processes (``_forecast_child.py``; the marker protocol of
``test_backup_every_steps.run_child``, which SIGKILLs the RECORDED PID at a ``@@AT`` marker): ``nbeats`` (a point head),
``tirex`` (a quantile head, a flat target) and ``nbeats`` on a csv, on 3 epochs of 10 batches of 32. Process 1 is killed after
2 batches of epoch 1 (the backup holds the end of epoch 0), process 2 resumes, finishes epoch 1 and is killed after 2 batches
of epoch 2, process 3 resumes and finishes. (Killing the same epoch twice would resume at the same count and so on the same
stream, by design: the stream is a function of the count.) One more run (``nbeats``, ``checkpoint_args.backup_every_steps=2``)
is killed INSIDE epoch 1 right after a mid-epoch copy: the resume runs the 8 batches the copy had not taken, at
``completed_steps = 10 + 2``.

What is read: the trace the children append to (the digest and the ``(series, start)`` of every train window a fit read, found
by looking the window up in a table of every window of the bank, so the identity does not rest on the module's own keys;
the same for the selection windows, one line per batch and per epoch) and the files, and the log line of the data wrapper
that states the stream seed and the digest of the first train batch. NOT checked: a resumed run is not the unkilled run's
trajectory, a GPU, a kill during a weight write (``test_backup_every_steps``). The tests write only under ``tmp_path``.
"""

import csv
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from train.common.pretrain_data import derive_stream_seed
from train.common.resume_state import FIT_STATE_NAME

from .test_backup_every_steps import run_child
from .test_forecast_task import (  # noqa: F401  (``world`` is a fixture)
    DATA, HORIZON, LOOKBACK, csv_data, prepared, write_series_csv, world,
)

CHILD = Path(__file__).resolve().parent / "_forecast_child.py"

BATCH, EPOCHS, STEPS = 32, 3, 10
#: Fixed windows of each split: two batches, so the select pass is cheap and ``validation_steps`` is 2.
SPLIT_WINDOWS = 2 * BATCH
SYNTHETIC = {**DATA, "realizations": 2, "n_select": SPLIT_WINDOWS, "n_test": SPLIT_WINDOWS, "steps_per_epoch": STEPS}
#: ``case: (model, model_args, csv?)``.
CASES = {"nbeats": ("nbeats/tiny", {}, False), "tirex": ("tirex/tiny", {}, False), "csv": ("nbeats/tiny", {}, True)}
#: Where the two kills land: ``(epoch, batches finished in it)``.
KILL_1, KILL_2 = (1, 2), (2, 2)
#: ``completed_steps`` of each resume: the end of epoch 0, the end of epoch 1; and a copy 2 batches into epoch 1.
COMPLETED = {"p2": STEPS, "p3": 2 * STEPS}
MID_COMPLETED = STEPS + 2
FIRST = 3        # the train batches whose digest is compared
NOT_FOUND = 1e-4  # the largest distance at which a window counts as found in the bank
LOGGED = re.compile(r"forecast train stream: stream seed (\d+), first train batch sha256 ([0-9a-f]{64})")


def read_trace(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def csv_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def request(case, source, **changes):
    """The ``mapping`` of a run of ``case`` (a dotted key sets a field of a group); ``out`` is added by the chain."""
    model, model_args, from_csv = CASES[case]
    asked = {
        "model": model, "data": csv_data(source, steps_per_epoch=STEPS) if from_csv else dict(SYNTHETIC),
        "epochs": EPOCHS, "batch_size": BATCH, "model_args": model_args,
        "analysis_args": {"model_analysis": False, "figures": False},
    }
    for key, value in changes.items():
        head, _, tail = key.partition(".")
        asked[head] = {**asked.get(head, {}), tail: value} if tail else value
    return asked


class Chain:
    """Child processes on one run directory: what each ran, and what each kill left behind."""

    def __init__(self, root: Path, case: str, source: Path, **changes) -> None:
        self.root, self.case, self.source, self.changes = Path(root), case, source, changes
        self.run, self.trace = self.root / "run", self.root / "trace.jsonl"
        #: ``{label: {"csv": rows, "sidecar": dict}}`` read right after each kill.
        self.after = {}

    def go(self, label: str, expect: str, resume: bool = False, **spec) -> None:
        body = {**request(self.case, self.source, **self.changes), "out": str(self.run), "resume_interrupted": resume}
        marker = run_child({"mapping": body, "trace": str(self.trace), "label": label, **spec},
                           self.root / f"{label}.log", child_script=CHILD)
        assert marker == expect, f"{self.case}/{label}: {marker}"
        if marker != "@@DONE":
            self.after[label] = {
                "csv": csv_rows(self.run / "training_log.csv"),
                "sidecar": json.loads((self.run / "checkpoints" / FIT_STATE_NAME).read_text()),
            }

    def lines(self, kind: str, label: str = None):
        return [row for row in read_trace(self.trace) if row["kind"] == kind and label in (None, row["label"])]

    def digests(self, label: str):
        """The digests of the first ``FIRST`` train batches the process's fit read, in the order it read them."""
        return [row["digest"] for row in self.lines("train_batch", label)][:FIRST]

    def logged(self, label: str):
        """``[(stream seed, first-batch digest)]`` the data wrapper logged in the process, in order."""
        return [(int(seed), digest) for seed, digest in LOGGED.findall((self.root / f"{label}.log").read_text())]


def stream_digests(ds):
    """The digest of each of the first ``FIRST`` batches of ``ds`` (the child's: x bytes then y bytes)."""
    return [hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest() for x, y in ds.take(FIRST)]


def expected(root, case, source, counts):
    """The data the parent builds itself, and ``{count: digests}`` of ``with_stream(count)`` (``None``: the fresh stream)."""
    asked = request(case, source)
    trainer, resolved, context = prepared(Path(root) / "expected", **asked)
    data = trainer.build_data(resolved, context)
    streams = {None: stream_digests(data.train_ds)}
    for count in counts:
        streams[count] = stream_digests(data.with_stream(count).train_ds)
    return SimpleNamespace(streams=streams, data=data, seed=resolved.seed)


@pytest.fixture(scope="module")
def csv_file(tmp_path_factory) -> Path:
    return write_series_csv(tmp_path_factory.mktemp("resume_csv") / "series.csv")


@pytest.fixture(scope="module", params=list(CASES))
def chain(request, world, csv_file, tmp_path_factory):
    """Kill inside epoch 1, resume, kill inside epoch 2, resume, finish: three processes of one case."""
    case = request.param
    run = Chain(tmp_path_factory.mktemp(f"resume_{case}"), case, csv_file)
    run.go("p1", f"@@AT {KILL_1[0]} {KILL_1[1]}", stop_at=list(KILL_1))
    run.go("p2", f"@@AT {KILL_2[0]} {KILL_2[1]}", resume=True, probe_restored=True, stop_at=list(KILL_2))
    run.go("p3", "@@DONE", resume=True, probe_restored=True)
    run.expect = expected(run.root, case, csv_file, COMPLETED.values())
    return run


@pytest.fixture(scope="module")
def mid_epoch(world, csv_file, tmp_path_factory):
    """``nbeats`` with a copy every 2 batches: killed after 2 batches of epoch 1, then resumed to the end."""
    run = Chain(tmp_path_factory.mktemp("resume_mid_epoch"), "nbeats", csv_file, **{"checkpoint_args.backup_every_steps": 2})
    run.go("p1", "@@AT 1 2", stop_at=[1, 2])
    run.go("p2", "@@DONE", resume=True)
    run.expect = expected(run.root, "nbeats", csv_file, (STEPS, MID_COMPLETED))
    return run


# ---------------------------------------------------------------------
# The kills are real and mid-epoch; the epochs and the step count are absolute
# ---------------------------------------------------------------------


def test_the_sizes_the_chain_counts_are_the_data_s(chain):
    data = chain.expect.data
    assert data.steps_per_epoch == STEPS and data.n_train == STEPS * BATCH and data.batch_size == BATCH
    assert data.n_select == SPLIT_WINDOWS or chain.case == "csv", "the control: the select windows are two batches"


def test_each_kill_lands_between_two_batches_of_an_epoch(chain):
    for label, (epoch, done) in (("p1", KILL_1), ("p2", KILL_2)):
        last = chain.lines("batch", label)[-1]
        assert (last["epoch"], last["batch"]) == (epoch, done - 1), f"{label} was killed after exactly {done} batches of epoch {epoch}"
        assert not [row for row in chain.lines("epoch_end", label) if row["epoch"] == epoch], "the killed epoch had not ended"
        assert not chain.lines("train_end", label), "a killed process ends nothing"


def test_a_resume_starts_at_the_epoch_after_the_last_finished_one_and_the_killed_epoch_runs_again(chain):
    epochs = {label: [row["epoch"] for row in chain.lines("epoch_begin", label)] for label in ("p1", "p2", "p3")}
    assert epochs == {"p1": [0, 1], "p2": [1, 2], "p3": [2]}


def test_the_optimizer_step_count_is_epoch_times_steps_at_every_epoch_start_and_the_unkilled_total_at_the_end(chain):
    for row in chain.lines("epoch_begin"):
        assert row["iterations"] == row["epoch"] * STEPS, f"{row['label']} epoch {row['epoch']} began at {row['iterations']}"
    for row in chain.lines("epoch_end"):
        assert row["iterations"] == (row["epoch"] + 1) * STEPS
    assert [row["iterations"] for row in chain.lines("train_end")] == [EPOCHS * STEPS]
    batches = chain.lines("batch")
    assert len(batches) == EPOCHS * STEPS + KILL_1[1] + KILL_2[1], "the asked steps plus exactly the batches the kills lost"
    for row in batches:
        assert row["before"] == row["epoch"] * STEPS + row["batch"], row


# ---------------------------------------------------------------------
# Rows, weights, summary
# ---------------------------------------------------------------------


def test_the_csv_rows_are_unique_and_complete_and_the_earlier_rows_are_kept_as_they_were(chain):
    final = csv_rows(chain.run / "training_log.csv")
    assert [row["epoch"] for row in final] == [str(e) for e in range(EPOCHS)]
    first, second = chain.after["p1"]["csv"], chain.after["p2"]["csv"]
    assert [row["epoch"] for row in first] == ["0"] and [row["epoch"] for row in second] == ["0", "1"]
    assert final[:1] == first and final[:2] == second, "a resume rewrote a finished epoch's row"


def test_the_weights_a_resume_restores_score_the_val_mae_the_killed_run_logged_for_them(chain):
    for label, from_row in (("p2", chain.after["p1"]["csv"][-1]), ("p3", chain.after["p2"]["csv"][-1])):
        restored, logged = chain.lines("epoch_begin", label)[0]["restored"]["mae"], float(from_row["val_mae"])
        # Exact (the same weights, the select windows fixed by (split seed, salt, index), one thread under op determinism),
        # so a select window that followed the stream seed shows here: the same weights would score another number.
        assert np.float32(restored) == np.float32(logged), (
            f"{chain.case}/{label}: the restored weights score {restored!r}, epoch {from_row['epoch']} logged {logged!r}")


def test_the_summary_says_the_run_resumed_and_finished(chain):
    summary = json.loads((chain.run / "results_summary.json").read_text())
    assert summary["resumed_after_epochs"] == KILL_2[0] and summary["epochs_run"] == EPOCHS
    assert (chain.run / "final_model.keras").is_file() and (chain.run / "best_model.keras").is_file()


# ---------------------------------------------------------------------
# Not a replay, and the stream is the one of the settled count
# ---------------------------------------------------------------------


def test_the_first_batches_of_each_process_differ_from_the_process_before(chain):
    first, second, third = (chain.digests(label) for label in ("p1", "p2", "p3"))
    assert len(first) == len(second) == len(third) == FIRST and len(set(first)) == FIRST, "the control: three real batches each"
    assert second != first, "process 2 trained on the stream of process 1 again: a replay"
    assert third != second, "process 3 trained on the stream of process 2 again: a replay"
    assert third != first
    assert not set(first) & set(second) and not set(second) & set(third)


def test_the_stream_of_a_process_is_the_one_of_its_settled_completed_steps(chain):
    # Process 1 is the fresh stream; process 2 and 3 are `with_stream` of the count their copy held. The parent built these
    # in its own process: equal digests mean the same windows, in the same order, standardized the same way.
    streams = chain.expect.streams
    assert chain.digests("p1") == streams[None]
    for label, count in COMPLETED.items():
        assert chain.digests(label) == streams[count], f"{label} did not train on the stream of {count} completed steps"
    assert len({tuple(digests) for digests in streams.values()}) == 1 + len(COMPLETED), "the control: every count is its own stream"


def test_the_logged_first_batch_digest_of_the_data_wrapper_is_the_one_the_fit_read_and_names_its_stream_seed(chain):
    seed = chain.expect.seed
    wanted = {"p1": seed, "p2": derive_stream_seed(seed, COMPLETED["p2"]), "p3": derive_stream_seed(seed, COMPLETED["p3"])}
    assert len(set(wanted.values())) == 3, "the control: three stream seeds"
    for label, stream_seed in wanted.items():
        logged = chain.logged(label)
        # A fresh process logs the stream of `build_data`; a resume logs a second one, the one `resume_stream` made.
        assert logged and logged[-1][0] == stream_seed, f"{label} logged {logged}, its stream seed is {stream_seed}"
        assert logged[-1][1] == chain.digests(label)[0], f"{label}: the logged digest is not the first batch its fit read"
    assert [seed for seed, _ in chain.logged("p1")] == [seed] and len(chain.logged("p2")) == len(chain.logged("p3")) == 2


# ---------------------------------------------------------------------
# Disjointness: the split does not move, no train window is a select window
# ---------------------------------------------------------------------


def test_the_bank_and_the_selection_windows_are_the_same_in_every_process_and_the_ones_the_parent_derives(chain):
    data = chain.expect.data
    wanted = sorted(map(list, np.asarray(data.select_windows).tolist()))
    bank = hashlib.sha256(data.bank.tobytes()).hexdigest()
    selects = chain.lines("select")
    assert [row["label"] for row in selects] == ["p1", "p2", "p2", "p3", "p3"], "a build per process and one per resume"
    for row in selects:
        assert row["bank"] == bank, f"{row['label']} built another bank: the split moved with the stream seed"
        assert row["windows"] == wanted, f"{row['label']} selects other windows: the split moved with the stream seed"
        assert row["distance"] < NOT_FOUND, "every selection window is a window of the bank"
        assert {k: tuple(v) for k, v in row["segments"].items()} == data.segments


def test_the_split_seed_stays_and_only_the_stream_seed_moves_on_a_resume(chain):
    seed = chain.expect.seed
    streams = [(row["label"], row["stream_seed"]) for row in chain.lines("select")]
    assert streams == [("p1", seed), ("p2", seed), ("p2", derive_stream_seed(seed, COMPLETED["p2"])),
                       ("p3", seed), ("p3", derive_stream_seed(seed, COMPLETED["p3"]))]


def test_no_train_window_of_any_process_lies_in_a_selection_or_test_segment(chain):
    train_hi, width = chain.expect.data.segments["train"][1], LOOKBACK + HORIZON
    select_lo = chain.expect.data.segments["select"][0]
    assert train_hi == select_lo
    starts = []
    for label in ("p1", "p2", "p3"):
        batches = chain.lines("train_batch", label)
        windows = [tuple(window) for batch in batches for window in batch["windows"]]
        assert len(batches) >= STEPS and len(windows) >= STEPS * BATCH, f"the control: {label} read a real part of the pool"
        assert max(batch["distance"] for batch in batches) < NOT_FOUND, f"{label} read a window that is not one of the bank"
        assert all(start + width <= train_hi for _, start in windows), (
            f"{label} trained on a window that reaches into the selection segment (rows from {train_hi})")
        starts += [start for _, start in windows]
    series = {series for label in ("p1", "p2", "p3") for batch in chain.lines("train_batch", label) for series, _ in batch["windows"]}
    assert len(series) == chain.expect.data.n_series or chain.case == "csv", "the control: every series is trained on"
    assert max(starts) > (train_hi - width) // 2, "the control: the train windows reach into the second half of the segment"
    for row in chain.lines("select"):
        assert all(start >= select_lo and start + width <= chain.expect.data.segments["test"][0] for _, start in row["windows"])


# ---------------------------------------------------------------------
# A kill inside an epoch, after a mid-epoch copy
# ---------------------------------------------------------------------


def test_a_resume_after_a_mid_epoch_copy_runs_only_the_batches_the_copy_had_not_taken(mid_epoch):
    first, second = mid_epoch.lines("epoch_begin", "p1"), mid_epoch.lines("epoch_begin", "p2")
    assert [row["epoch"] for row in first] == [0, 1] and [row["epoch"] for row in second] == [1, 2]
    assert second[0]["iterations"] == MID_COMPLETED, "the step count is the copy's (10 + 2), not the epoch's start"
    assert second[1]["iterations"] == 2 * STEPS
    assert len([row for row in mid_epoch.lines("batch", "p2") if row["epoch"] == 1]) == STEPS - 2
    assert len(mid_epoch.lines("batch")) == EPOCHS * STEPS, "no batch was lost: the kill came right after a copy"
    assert [row["iterations"] for row in mid_epoch.lines("train_end")] == [EPOCHS * STEPS]


def test_the_stream_after_a_mid_epoch_copy_is_the_one_of_the_count_the_copy_holds(mid_epoch):
    # 12 = 1 * 10 + (1 + 1): the batch term is part of the settled count. 10 (the epoch's start) is another stream.
    streams = mid_epoch.expect.streams
    assert mid_epoch.digests("p1") == streams[None]
    assert mid_epoch.digests("p2") == streams[MID_COMPLETED]
    assert streams[MID_COMPLETED] != streams[STEPS] != streams[None]
    assert mid_epoch.logged("p2")[-1][0] == derive_stream_seed(mid_epoch.expect.seed, MID_COMPLETED)


def test_the_mid_epoch_resume_leaves_each_epoch_one_row_and_keeps_the_split(mid_epoch):
    assert [row["epoch"] for row in csv_rows(mid_epoch.run / "training_log.csv")] == [str(e) for e in range(EPOCHS)]
    data = mid_epoch.expect.data
    selects = mid_epoch.lines("select")
    assert selects and all(row["windows"] == sorted(map(list, np.asarray(data.select_windows).tolist())) for row in selects)
    assert all(row["bank"] == hashlib.sha256(data.bank.tobytes()).hexdigest() for row in selects)
    train_hi, width = data.segments["train"][1], LOOKBACK + HORIZON
    windows = [window for row in mid_epoch.lines("train_batch") for window in row["windows"]]
    assert windows and all(start + width <= train_hi for _, start in windows)
