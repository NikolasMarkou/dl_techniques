"""The resume protocol the stream tasks share: the child half and the parent half of a REAL SIGKILL chain.

``pretrain`` and ``time_series_forecasting`` are stream tasks (an infinite train stream, an epoch of ``steps_per_epoch``
batches, a resumed ``fit`` that would replay the stream), and both are tested the same way: a chain of three child
processes on one run directory, the first two killed at a ``@@AT`` marker by the parent (``test_backup_every_steps.run_child``
SIGKILLs the RECORDED PID), the third finished. This module holds what the two tests have in common, so the protocol has one
written home: the children (``_pretrain_child.py``, ``_forecast_child.py``) are a few lines each (what they WATCH on the stream),
and the two parents (``test_pretrain_resume.py``, ``test_forecast_resume.py``) keep what is the task's own (what a window or a row
IS, the selection set, the files) and call the ``check_*`` functions below for the assertions that are the protocol's.

Child half (no TensorFlow, keras or pytest at module scope: the child script sets its environment first)
    ``child_environment()``
        Set the CPU-only defaults; the first call of a child script, before TensorFlow is imported.
    ``start_tensorflow()``
        :return: ``tensorflow`` with op determinism on and one thread per pool (the golden run's determinism).
    ``run_child(spec, trainer_class, config_class, watcher)``
        :param spec: the JSON the parent passed (``mapping``, ``trace``, ``label``, ``stop_at``, ``probe_restored``).
        :param watcher: ``watcher(write) -> watched`` where ``write(kind, **fields)`` appends a JSON line to the trace and
            ``watched(data, config) -> data`` returns the task's data with its selection set written to the trace and its
            train stream read through a recorder (``train_batch`` lines): what the fit really read, in the order it read it.
        Runs ``trainer_class(validate(config_class, spec["mapping"]))`` with a ``Recorder`` (``epoch_begin``, ``batch``,
        ``epoch_end``, ``train_end``, and the scores of the restored weights when ``probe_restored``) and a ``Stopper`` (prints
        ``@@AT <epoch> <done>`` and sleeps) added after the callbacks the task builds; prints ``@@DONE`` at the end.

Parent half
    ``read_trace(path)``, ``csv_rows(path)``, ``ResumeChain`` (child processes on one run directory: what each ran and what each
    kill left behind), and the assertions of the protocol: ``check_each_kill_lands_between_two_batches``,
    ``check_a_resume_starts_after_the_last_finished_epoch``, ``check_the_step_count_is_absolute``,
    ``check_the_csv_rows_are_unique_and_complete``, ``check_the_restored_weights_score_the_logged_value``,
    ``check_no_process_replays_the_one_before``, ``check_each_process_trains_on_the_stream_of_its_settled_count``,
    ``check_a_mid_epoch_resume_runs_only_the_batches_the_copy_had_not_taken`` and ``check_the_mid_epoch_stream``. Each takes the
    chain and the constants of the test file, and fails with the message that names the property.
"""

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# the child half
# ---------------------------------------------------------------------


def child_environment() -> None:
    """Default a child to the CPU with quiet TensorFlow and a headless matplotlib; call before TensorFlow is imported."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("MPLBACKEND", "Agg")


def start_tensorflow() -> Any:
    """Import TensorFlow with op determinism and one thread per pool, and return it."""
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    return tf


def steps_of(model: Any) -> int:
    """The optimizer's step count (the inner optimizer of a loss-scaled one)."""
    import keras

    optimizer = model.optimizer
    return int(keras.ops.convert_to_numpy(getattr(optimizer, "inner_optimizer", optimizer).iterations))


def _callback_classes(spec: Dict[str, Any], write: Callable[..., None]) -> Tuple[Any, Any]:
    import keras

    class Recorder(keras.callbacks.Callback):
        """Append what the run did to the trace; sits last in the list, after the callback that restores the backup."""

        def __init__(self, data: Any) -> None:
            super().__init__()
            self.data, self.epoch, self.first = data, 0, True

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch
            fields = {"epoch": epoch, "iterations": steps_of(self.model)}
            if self.first and spec.get("probe_restored"):
                # The weights a resume restored, scored before any step of this process: the selection set is fixed by the
                # split seed, so these are the numbers the killed run logged for the epoch they came from.
                # `float(v)`: a model with a `test_step` of its own (the tokenizer's) returns tensors, which the trace cannot hold.
                fields["restored"] = {k: float(v) for k, v in self.model.evaluate(self.data.select_ds, verbose=0, return_dict=True).items()}
            self.first = False
            write("epoch_begin", **fields)

        def on_train_batch_end(self, batch, logs=None):
            write("batch", epoch=self.epoch, batch=batch, before=steps_of(self.model) - 1)

        def on_epoch_end(self, epoch, logs=None):
            write("epoch_end", epoch=epoch, iterations=steps_of(self.model))

        def on_train_end(self, logs=None):
            write("train_end", iterations=steps_of(self.model))

    class Stopper(keras.callbacks.Callback):
        """Block when ``done`` batches of ``epoch`` are finished (runs before the next batch)."""

        def __init__(self, epoch: int, done: int) -> None:
            super().__init__()
            self.target, self.epoch = (epoch, done), 0

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch

        def on_train_batch_begin(self, batch, logs=None):
            if self.target == (self.epoch, batch) and batch > 0:
                # The marker the parent reads off the log (``test_backup_every_steps.run_child``), not a log line.
                print(f"\n@@AT {self.epoch} {batch}", flush=True)
                time.sleep(3600)

    return Recorder, Stopper


def run_child(spec: Dict[str, Any], trainer_class: Any, config_class: Any, watcher: Callable[[Callable[..., None]], Callable[[Any, Any], Any]]) -> None:
    """Run one watched fit of ``trainer_class`` and print ``@@DONE``; see the module contract."""
    from train.api.config import validate

    def write(kind: str, **fields: Any) -> None:
        with open(spec["trace"], "a") as handle:
            handle.write(json.dumps({"label": spec["label"], "kind": kind, **fields}) + "\n")
            handle.flush()

    watched = watcher(write)
    Recorder, Stopper = _callback_classes(spec, write)

    class Child(trainer_class):
        def build_data(self, config, context):
            return watched(super().build_data(config, context), config)

        def resume_stream(self, config, data, completed_steps):
            replacement = super().resume_stream(config, data, completed_steps)
            # None is the task saying "keep the data": the fit then reads the stream `build_data` made.
            return None if replacement is None else watched(replacement, config)

        def _callbacks(self, config, data, *arguments):
            epoch_line, callbacks = super()._callbacks(config, data, *arguments)
            callbacks.append(Recorder(data))
            if spec.get("stop_at"):
                callbacks.append(Stopper(*spec["stop_at"]))
            return epoch_line, callbacks

    Child(validate(config_class, spec["mapping"])).run()
    print("\n@@DONE", flush=True)  # the marker of a finished child (see above)


# ---------------------------------------------------------------------
# the parent half
# ---------------------------------------------------------------------


def read_trace(path: Any) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def csv_rows(path: Any) -> List[Dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def merge_changes(asked: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    """Apply ``changes`` to a request mapping: a dotted key (``"checkpoint_args.backup_every_steps"``) sets a field of a group."""
    for key, value in changes.items():
        head, _, tail = key.partition(".")
        asked[head] = {**asked.get(head, {}), tail: value} if tail else value
    return asked


class ResumeChain:
    """Child processes on one run directory: what each ran, and what each kill left behind.

    A subclass supplies :meth:`mapping` (the request, without ``out`` and ``resume_interrupted``), may add keys to the child's
    spec in :meth:`child_spec`, and sets ``child_script`` (the script the children run) and ``first`` (how many train batches
    :meth:`digests` returns).
    """

    child_script: Path
    first: int

    def __init__(self, root: Path, name: str, **changes: Any) -> None:
        self.root, self.name, self.changes = Path(root), name, changes
        self.run, self.trace = self.root / "run", self.root / "trace.jsonl"
        #: ``{label: {"csv": rows, "sidecar": dict}}`` read right after each kill.
        self.after: Dict[str, Dict[str, Any]] = {}

    def mapping(self) -> Dict[str, Any]:
        raise NotImplementedError

    def child_spec(self) -> Dict[str, Any]:
        return {}

    def go(self, label: str, expect: str, resume: bool = False, **spec: Any) -> None:
        from train.common.resume_state import FIT_STATE_NAME

        from .test_backup_every_steps import run_child as run_the_child

        body = {**self.mapping(), "out": str(self.run), "resume_interrupted": resume}
        marker = run_the_child({"mapping": body, "trace": str(self.trace), "label": label, **self.child_spec(), **spec},
                               self.root / f"{label}.log", child_script=self.child_script)
        assert marker == expect, f"{self.name}/{label}: {marker}"
        if marker != "@@DONE":
            self.after[label] = {
                "csv": csv_rows(self.run / "training_log.csv"),
                "sidecar": json.loads((self.run / "checkpoints" / FIT_STATE_NAME).read_text()),
            }

    def lines(self, kind: str, label: Optional[str] = None) -> List[Dict[str, Any]]:
        return [row for row in read_trace(self.trace) if row["kind"] == kind and label in (None, row["label"])]

    def digests(self, label: str) -> List[str]:
        """The digests of the first ``self.first`` train batches the process's fit read, in the order it read them."""
        return [row["digest"] for row in self.lines("train_batch", label)][:self.first]


def check_each_kill_lands_between_two_batches(chain: ResumeChain, kill_1: Sequence[int], kill_2: Sequence[int]) -> None:
    for label, (epoch, done) in (("p1", kill_1), ("p2", kill_2)):
        last = chain.lines("batch", label)[-1]
        assert (last["epoch"], last["batch"]) == (epoch, done - 1), f"{label} was killed after exactly {done} batches of epoch {epoch}"
        assert not [row for row in chain.lines("epoch_end", label) if row["epoch"] == epoch], "the killed epoch had not ended"
        assert not chain.lines("train_end", label), "a killed process ends nothing"


def check_a_resume_starts_after_the_last_finished_epoch(chain: ResumeChain) -> None:
    epochs = {label: [row["epoch"] for row in chain.lines("epoch_begin", label)] for label in ("p1", "p2", "p3")}
    assert epochs == {"p1": [0, 1], "p2": [1, 2], "p3": [2]}


def check_the_step_count_is_absolute(chain: ResumeChain, steps: int, epochs: int, kill_1: Sequence[int], kill_2: Sequence[int]) -> None:
    for row in chain.lines("epoch_begin"):
        assert row["iterations"] == row["epoch"] * steps, f"{row['label']} epoch {row['epoch']} began at {row['iterations']}"
    for row in chain.lines("epoch_end"):
        assert row["iterations"] == (row["epoch"] + 1) * steps
    assert [row["iterations"] for row in chain.lines("train_end")] == [epochs * steps]
    batches = chain.lines("batch")
    assert len(batches) == epochs * steps + kill_1[1] + kill_2[1], "the asked steps plus exactly the batches the kills lost"
    for row in batches:
        assert row["before"] == row["epoch"] * steps + row["batch"], row


def check_the_csv_rows_are_unique_and_complete(chain: ResumeChain, epochs: int) -> None:
    final = csv_rows(chain.run / "training_log.csv")
    assert [row["epoch"] for row in final] == [str(e) for e in range(epochs)]
    first, second = chain.after["p1"]["csv"], chain.after["p2"]["csv"]
    assert [row["epoch"] for row in first] == ["0"] and [row["epoch"] for row in second] == ["0", "1"]
    assert final[:1] == first and final[:2] == second, "a resume rewrote a finished epoch's row"


def check_the_restored_weights_score_the_logged_value(chain: ResumeChain, metric: str) -> None:
    """The weights a resume restores score the ``val_<metric>`` the killed run logged for them, to the float32 bit."""
    import numpy as np

    for label, from_row in (("p2", chain.after["p1"]["csv"][-1]), ("p3", chain.after["p2"]["csv"][-1])):
        restored, logged = chain.lines("epoch_begin", label)[0]["restored"][metric], float(from_row[f"val_{metric}"])
        # Exact (the same weights, the selection set fixed by the split seed, one thread under op determinism), so a selection
        # set that followed the stream seed shows here: the same weights would score another number.
        assert np.float32(restored) == np.float32(logged), (
            f"{chain.name}/{label}: the restored weights score {restored!r}, epoch {from_row['epoch']} logged {logged!r}")


def check_no_process_replays_the_one_before(chain: ResumeChain) -> None:
    one, two, three = (chain.digests(label) for label in ("p1", "p2", "p3"))
    assert len(one) == len(two) == len(three) == chain.first and len(set(one)) == chain.first, "the control: three real batches each"
    assert two != one, "process 2 trained on the stream of process 1 again: a replay"
    assert three != two, "process 3 trained on the stream of process 2 again: a replay"
    assert three != one
    assert not set(one) & set(two) and not set(two) & set(three)


def check_each_process_trains_on_the_stream_of_its_settled_count(
        chain: ResumeChain, streams: Dict[Optional[int], List[str]], completed: Dict[str, int]) -> None:
    """Process 1 is the fresh stream; processes 2 and 3 are ``with_stream`` of the count their copy held."""
    assert chain.digests("p1") == streams[None]
    for label, count in completed.items():
        assert chain.digests(label) == streams[count], f"{label} did not train on the stream of {count} completed steps"
    assert len({tuple(digests) for digests in streams.values()}) == 1 + len(completed), "the control: every count is its own stream"


def check_a_mid_epoch_resume_runs_only_the_batches_the_copy_had_not_taken(
        mid: ResumeChain, steps: int, epochs: int, mid_completed: int) -> None:
    first, second = mid.lines("epoch_begin", "p1"), mid.lines("epoch_begin", "p2")
    assert [row["epoch"] for row in first] == [0, 1] and [row["epoch"] for row in second] == [1, 2]
    assert second[0]["iterations"] == mid_completed, f"the step count is the copy's ({steps} + 2), not the epoch's start"
    assert second[1]["iterations"] == 2 * steps
    assert len([row for row in mid.lines("batch", "p2") if row["epoch"] == 1]) == steps - 2
    assert len(mid.lines("batch")) == epochs * steps, "no batch was lost: the kill came right after a copy"
    assert [row["iterations"] for row in mid.lines("train_end")] == [epochs * steps]


def check_the_mid_epoch_stream(
        mid: ResumeChain, streams: Dict[Optional[int], List[str]], steps: int, mid_completed: int) -> None:
    """The batch term is part of the settled count (``steps + 2``): the epoch's start (``steps``) is another stream."""
    assert mid.digests("p1") == streams[None]
    assert mid.digests("p2") == streams[mid_completed]
    assert streams[mid_completed] != streams[steps] != streams[None]
