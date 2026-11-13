"""The stream contract of the shared sequence: ``fit_kwargs`` and the refusal of a stream with no length.

Plan 46e5318f step 3 (D-003). Keras 3.8 trains an epoch of an infinite ``train_ds`` forever when ``fit``
is given neither ``steps_per_epoch`` nor a dataset of known cardinality, so a task whose stream repeats
(the self-supervised image streams do) must say how long an epoch is, through ``fit_kwargs``. The sequence
refuses what would hang or what disagrees with the epoch it computed everything else from:

* a ``steps_per_epoch`` that is not a plain int, or differs from ``data.steps_per_epoch`` (the schedule, the
  backup arithmetic and ``settle_backup`` read that number, so ``fit`` must run the same one);
* a ``train_ds`` of infinite or unknown cardinality with no ``steps_per_epoch`` (a ``ConfigError`` that names
  ``fit_kwargs`` and ``steps_per_epoch`` and carries the claimed-run note, instead of a hang);
* a ``validation_steps`` that is not a plain int equal to the cardinality of ``select_ds``: Keras keeps ONE validation
  iterator, so a shorter count scores a different part of the stream each epoch and ``val_loss`` drifts; an explicit
  ``validation_steps=None`` is Keras' own default (the whole stream) and is read as absent;
* a ``select_ds`` of infinite cardinality, whether or not ``validation_steps`` is given: a fixed ``val_loss`` needs a
  fixed stream (D-026);
* with ``steps_per_epoch`` given the same stream trains, and ONE iterator persists across epochs: epoch
  1 continues epoch 0's stream (Keras 3.8, MEASURED in the plan's findings), so a counter-keyed stream does
  not restart.

Step 4 (D-004) adds the second hook, ``TaskTrainer.resume_stream(config, data, completed_steps)``: a resumed
``fit`` starts a fresh iterator, so a stream keyed on a counter would replay from 0. The sequence calls the
hook once, for a resume that trains, directly after ``settle_backup_pair`` (the count is final only then: a
torn latest copy sends the run back one epoch), with the optimizer steps the settled copy holds; a returned
object replaces the run's data, ``None`` leaves it, and a replacement whose ``steps_per_epoch``, counts or
fingerprint differ from the original's is refused by name. The runs are real classification runs killed in
process by a callback (an epoch end, a batch inside an epoch with ``backup_every_steps``) and resumed from a
copy of the directory; the count each hook call receives is compared with the optimizer step count the
settled backup file itself holds, read at the moment of the call.

The data are the real classification pipeline on fake CIFAR-10 arrays; the stream under test is that
pipeline's own ``train_ds`` made infinite (``repeat``) or unknown (``filter``) inside ``build_data``. Every
test that could hang if the guard were gone carries a callback that stops the run, so a broken guard is a
named failure and not a stuck suite. The tests write only under ``tmp_path``.
"""

import dataclasses
import shutil
from pathlib import Path

import keras
import numpy as np
import pytest
import tensorflow as tf

import train.api.recipes as recipes
from train.api import registry
from train.api.errors import ConfigError
from train.api.tasks import register_builtin_tasks
from train.api.tasks.image_classification import ImageClassificationTrainer
from train.common.resume_state import BACKUP_WEIGHTS_NAME, backup_iterations

from tests.classification_fakes import use_tiny_raw

from .test_task_trainer import make_config

EPOCHS = 2
#: More train batches in one epoch than any test below may ask for (its epochs are a few steps): past this a run without a bound has hung.
RUNAWAY = 40


class RanWithoutABound(RuntimeError):
    """Raised by :class:`Runaway`: the infinite stream trained past :data:`RUNAWAY` batches."""


class Runaway(keras.callbacks.Callback):
    """Stop a run that would otherwise never end (an infinite stream, no ``steps_per_epoch``)."""

    def on_train_batch_end(self, batch, logs=None):
        if batch >= RUNAWAY:
            raise RanWithoutABound(f"trained {batch} batches of one epoch")


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        use_tiny_raw(patch, n_train=40, n_test=8)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


def config_for(out, **changes):
    return make_config(out, **{"epochs": EPOCHS, "analysis_args.figures": False, **changes})


class Streaming(ImageClassificationTrainer):
    """The real task with its ``train_ds`` replaced by ``stream(train_ds)``, and ``fit_kwargs`` by ``options``.

    :cvar stream: the replacement of ``train_ds`` (a function of the real one); ``None`` keeps it.
    :cvar options: what ``fit_kwargs`` returns, given the data (a function), or ``None`` for the default hook.
        Both are set through :func:`streaming`, which wraps them as static methods.
    """

    stream = None
    options = None

    built = None

    def build_data(self, config, context):
        """Build the real data, keep it as ``built`` (the real ``steps_per_epoch``), swap the stream."""
        data = self.built = super().build_data(config, context)
        if self.stream is None:
            return data
        return dataclasses.replace(data, train_ds=self.stream(data.train_ds))

    def fit_kwargs(self, config, data):
        return {} if self.options is None else self.options(data)

    def extra_callbacks(self, config, data):
        return [Runaway()]


def streaming(**attributes):
    """Return a ``Streaming`` subclass with ``stream`` and ``options`` set from the keywords."""
    return type("Streaming", (Streaming,), {k: staticmethod(v) if callable(v) else v for k, v in attributes.items()})


def infinite(train_ds):
    return train_ds.repeat()


def unknown(train_ds):
    # ``filter`` hides the length without changing a row: cardinality UNKNOWN (-2), finite.
    return train_ds.filter(lambda *element: True)


def steps_only(data):
    return {"steps_per_epoch": data.steps_per_epoch}


# ---------------------------------------------------------------------
# steps_per_epoch must be the epoch the sequence computed
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [lambda n: n + 1, lambda n: n - 1, lambda n: float(n), lambda n: True, lambda n: None, lambda n: str(n)],
    ids=["one more", "one fewer", "a float", "a bool", "None", "a string"],
)
def test_a_steps_per_epoch_that_is_not_the_data_s_own_is_refused(world, tmp_path, bad):
    task = streaming(stream=infinite, options=lambda data: {"steps_per_epoch": bad(data.steps_per_epoch)})
    run_dir = tmp_path / "run"
    with pytest.raises(ConfigError, match=r"steps_per_epoch") as raised:
        task(config_for(run_dir)).run()
    message = str(raised.value)
    assert "Streaming" in message and "was claimed before this failed" in message
    assert not (run_dir / "training_log.csv").exists(), "refused before fit"


def test_the_steps_per_epoch_of_the_data_is_accepted(world, tmp_path):
    task = streaming(stream=infinite, options=steps_only)(config_for(tmp_path / "run", epochs=1))
    task.run()  # control: the same stream with the right number trains (a refusal above is the number's)


# ---------------------------------------------------------------------
# validation_steps must be the selection stream's own length
# ---------------------------------------------------------------------


def with_validation_steps(change):
    """A task whose ``fit_kwargs`` gives the right ``steps_per_epoch`` and ``change(select cardinality)`` validation steps."""

    def options(data):
        return {"steps_per_epoch": data.steps_per_epoch, "validation_steps": change(int(data.select_ds.cardinality()))}

    return streaming(stream=infinite, options=options)


@pytest.mark.parametrize(
    "bad",
    [lambda n: n + 1, lambda n: n - 1, lambda n: float(n), lambda n: True, lambda n: str(n)],
    ids=["one more", "one fewer", "a float", "a bool", "a string"],
)
def test_a_validation_steps_that_is_not_the_selection_streams_length_is_refused(world, tmp_path, bad):
    run_dir = tmp_path / "run"
    with pytest.raises(ConfigError, match=r"validation_steps") as raised:
        with_validation_steps(bad)(config_for(run_dir)).run()
    message = str(raised.value)
    assert "Streaming" in message and "was claimed before this failed" in message
    assert "select_ds" in message, "the message names the stream the number is compared with"
    assert not (run_dir / "training_log.csv").exists(), "refused before fit"


def test_a_selection_stream_of_unknown_length_is_not_judged_by_validation_steps(tmp_path):
    # Nothing to compare the count with, so the guard stands aside (the pretrain data wrapper refuses such a stream itself).
    # A real run cannot show it: the classification task's own epoch-zero evaluate needs the length, so the guard is called
    # on a stand-in run, as the sequence calls it.
    from types import SimpleNamespace

    from train.api.run_sequence import _checked_fit_options

    class Task:
        def fit_kwargs(self, config, data):
            return {"steps_per_epoch": 3, "validation_steps": 5}

    def run_over(select_ds):
        data = SimpleNamespace(train_ds=tf.data.Dataset.range(6).batch(2), select_ds=select_ds)
        return SimpleNamespace(trainer=Task(), config=None, data=data, run_dir=tmp_path, steps_per_epoch=3)

    known = tf.data.Dataset.range(10).batch(2)                         # 5 batches: the count is right
    assert _checked_fit_options(run_over(known)) == {"steps_per_epoch": 3, "validation_steps": 5}
    assert _checked_fit_options(run_over(unknown(known))) == {"steps_per_epoch": 3, "validation_steps": 5}
    with pytest.raises(ConfigError, match="validation_steps=5"):       # the instrument can see: a known length of 4
        _checked_fit_options(run_over(tf.data.Dataset.range(8).batch(2)))


def stand_in_run(tmp_path, select_ds, options):
    """A stand-in run for :func:`_checked_fit_options`: a known 3-batch train stream, ``select_ds``, ``options``."""
    from types import SimpleNamespace

    class Task:
        def fit_kwargs(self, config, data):
            return options

    data = SimpleNamespace(train_ds=tf.data.Dataset.range(6).batch(2), select_ds=select_ds)
    return SimpleNamespace(trainer=Task(), config=None, data=data, run_dir=tmp_path, steps_per_epoch=3)


@pytest.mark.parametrize(
    "options",
    [{"steps_per_epoch": 3, "validation_steps": 5}, {"steps_per_epoch": 3}, {"steps_per_epoch": 3, "validation_steps": None}],
    ids=["a count that matches the batches of one pass", "no validation_steps", "an explicit None"],
)
def test_an_endless_selection_stream_is_refused_by_name_with_or_without_validation_steps(tmp_path, options):
    # D-026: a fixed ``val_loss`` needs a fixed stream. ``.repeat()`` of the right batches makes Keras' ONE validation
    # iterator continue where the last epoch stopped (a drifting ``val_loss``) when a count is given, and never finish
    # when none is. A real run cannot show it (the task's own epoch-zero evaluate needs the length): stand-in run.
    from train.api.run_sequence import _checked_fit_options

    endless = infinite(tf.data.Dataset.range(10).batch(2))
    assert int(endless.cardinality()) == tf.data.INFINITE_CARDINALITY, "vacuity guard: the stream is endless"
    with pytest.raises(ConfigError) as raised:
        _checked_fit_options(stand_in_run(tmp_path, endless, options))
    message = str(raised.value)
    assert "select_ds" in message and "endless" in message and "cardinality -1" in message
    assert "fixed" in message and "was claimed before this failed" in message


def test_a_selection_stream_of_unknown_length_with_no_validation_steps_is_not_judged(tmp_path):
    # The refusal above is for -1 only: -2 is a finite stream whose length is not stated (the same as the unknown-length test).
    from train.api.run_sequence import _checked_fit_options

    unstated = unknown(tf.data.Dataset.range(10).batch(2))
    assert int(unstated.cardinality()) == tf.data.UNKNOWN_CARDINALITY
    assert _checked_fit_options(stand_in_run(tmp_path, unstated, {"steps_per_epoch": 3})) == {"steps_per_epoch": 3}


def test_an_explicit_none_for_validation_steps_is_keras_own_default_and_is_not_refused(tmp_path):
    # ``validation_steps=None`` is what Keras does when the key is absent (the whole stream): read as absent, passed on as given.
    from train.api.run_sequence import _checked_fit_options

    known = tf.data.Dataset.range(10).batch(2)
    options = {"steps_per_epoch": 3, "validation_steps": None}
    assert _checked_fit_options(stand_in_run(tmp_path, known, options)) == options


def test_the_validation_steps_of_the_selection_stream_are_accepted(world, tmp_path):
    with_validation_steps(lambda n: n)(config_for(tmp_path / "run", epochs=1)).run()  # control: the right number trains


def test_a_validation_steps_of_none_trains_on_the_whole_selection_stream(world, tmp_path):
    run_dir = tmp_path / "run"
    with_validation_steps(lambda n: None)(config_for(run_dir, epochs=1)).run()  # Keras' default, so a fixed val_loss
    assert (run_dir / "training_log.csv").is_file()


# ---------------------------------------------------------------------
# A stream with no length and no steps_per_epoch
# ---------------------------------------------------------------------


@pytest.mark.parametrize("stream", [infinite, unknown], ids=["infinite", "unknown cardinality"])
def test_a_stream_with_no_length_and_no_steps_per_epoch_is_refused_instead_of_hanging(world, tmp_path, stream):
    run_dir = tmp_path / "run"
    with pytest.raises(ConfigError) as raised:
        streaming(stream=stream)(config_for(run_dir)).run()
    message = str(raised.value)
    assert "fit_kwargs" in message and "steps_per_epoch" in message
    assert "Streaming" in message and "was claimed before this failed" in message
    # What is true: the stream does not state its length, so Keras cannot size an epoch of it. A finite stream of unknown
    # length does not "never end" (it ends in Keras' own error in a later epoch), so the sentence does not say it does.
    assert "does not state its length" in message and "cannot size an epoch" in message
    assert "never ends" not in message
    assert not (run_dir / "training_log.csv").exists(), "refused before fit"


def test_the_guard_reads_the_cardinality_the_stream_states(world):
    # The instrument can see: the two streams above have the cardinalities the guard refuses, the
    # shipped pipeline's own has none of them.
    probe = tf.data.Dataset.range(6).batch(2, drop_remainder=True)
    assert int(infinite(probe).cardinality()) == tf.data.INFINITE_CARDINALITY
    assert int(unknown(probe).cardinality()) == tf.data.UNKNOWN_CARDINALITY
    assert int(probe.cardinality()) == 3


def test_the_shipped_pipeline_has_a_known_length_so_no_steps_per_epoch_is_asked_of_it(world, tmp_path):
    run_dir = tmp_path / "run"
    task = streaming()(config_for(run_dir, epochs=1))
    task.run()
    assert (run_dir / "training_log.csv").is_file()


def test_a_stream_with_no_cardinality_attribute_is_not_judged_by_it(world, tmp_path):
    """A ``keras.utils.PyDataset`` has a length and no ``cardinality``: the guard skips it."""

    class Batches(keras.utils.PyDataset):
        def __init__(self, x, y, **kwargs):
            super().__init__(**kwargs)
            self.x, self.y = x, y

        def __len__(self):
            return len(self.x) // 4

        def __getitem__(self, index):
            rows = slice(index * 4, (index + 1) * 4)
            return self.x[rows], self.y[rows]

    def as_pydataset(train_ds):
        x, y = next(iter(train_ds.unbatch().batch(36)))
        pydataset = Batches(x.numpy(), y.numpy())
        assert not hasattr(pydataset, "cardinality"), "vacuity guard: the stand-in has no cardinality"
        return pydataset

    run_dir = tmp_path / "run"
    streaming(stream=as_pydataset)(config_for(run_dir, epochs=1)).run()
    assert (run_dir / "training_log.csv").is_file()


# ---------------------------------------------------------------------
# With steps_per_epoch the same stream trains, on ONE iterator
# ---------------------------------------------------------------------


def counted(produced):
    """Return a ``train_ds`` transform that records, in ``produced``, the counter of each element made."""

    def record(index):
        produced.append(int(index))
        return index

    def transform(train_ds):
        def tag(index, element):
            x, y = element
            seen = tf.numpy_function(record, [index], tf.int64)
            seen.set_shape(())
            # The recorded value feeds the pixels (by zero), or the stateful op would be pruned from the graph.
            return x + tf.cast(seen - seen, x.dtype), y

        return train_ds.repeat().enumerate().map(tag)

    return transform


def test_an_infinite_stream_with_steps_per_epoch_trains_on_one_iterator_across_epochs(world, tmp_path):
    produced = []
    task = streaming(stream=counted(produced), options=steps_only)(config_for(tmp_path / "run"))
    task.run()
    steps = task.built.steps_per_epoch
    # Every counter appears once and in order: a second iterator (one per epoch) would produce 0 again.
    assert produced == list(range(len(produced))), "a counter restarted: epoch 1 began a new iterator"
    # Read-ahead may run past the steps asked, never short of them.
    assert len(produced) >= EPOCHS * steps
    assert len(produced) < EPOCHS * steps + RUNAWAY, "the stream was read without a bound"


def test_the_persistent_iterator_probe_sees_a_restart(world):
    # The instrument can see: two fits of one model over a dataset each start a fresh iterator and
    # the counter shows it (this is what a resumed ``fit`` does).
    produced = []
    dataset = counted(produced)(tf.data.Dataset.from_tensor_slices(
        (np.zeros((8, 2), np.float32), np.zeros(8, np.int64))).batch(2, drop_remainder=True))
    model = keras.Sequential([keras.layers.Input((2,)), keras.layers.Dense(1)])
    model.compile(optimizer="sgd", loss="mse")
    model.fit(dataset, epochs=1, steps_per_epoch=4, verbose=0)
    model.fit(dataset, epochs=1, steps_per_epoch=4, verbose=0)
    assert produced != list(range(len(produced))) and produced.count(0) == 2


# ---------------------------------------------------------------------
# Seam 2: resume_stream, called once after settle_backup_pair
# ---------------------------------------------------------------------

RESUME_EPOCHS = 4
#: Batches in an epoch of the fake CIFAR-10 at batch 4 (36 fit rows); each resume test reads it back from ``data.steps_per_epoch``.
SPE = 9
#: ``backup_every_steps`` of the mid-epoch arm: copies after batches 4 and 8 of an epoch, and at each epoch end.
EVERY = 4


class Killed(RuntimeError):
    """Raised by :class:`KillBefore`: a stand-in for a kill between two batches."""


class KillBefore(keras.callbacks.Callback):
    """Raise before batch ``batch`` of epoch ``epoch`` (``batch`` batches of it have finished)."""

    def __init__(self, epoch, batch):
        super().__init__()
        self.epoch, self.batch, self.at = epoch, batch, 0

    def on_epoch_begin(self, epoch, logs=None):
        self.at = epoch

    def on_train_batch_begin(self, batch, logs=None):
        if self.at == self.epoch and batch == self.batch:
            raise Killed(f"killed before batch {batch} of epoch {self.epoch}")


class Resuming(ImageClassificationTrainer):
    """The real task with ``resume_stream`` recording its arguments, and a ``fit`` spy.

    :cvar kill: ``(epoch, batch)`` of a :class:`KillBefore`, or ``None``.
    :cvar replace: a function of the data the hook was given returning what it returns (``None`` returns
        ``None``); set through :func:`resuming`, which wraps it as a static method.
    :ivar calls: ``(completed_steps, data, steps_in_the_settled_backup)`` of every ``resume_stream`` call;
        the third is read from the backup weights file inside the call, so it is the settled pair's
        (``None`` when the file cannot be read).
    :ivar fit_x: the ``x`` of every ``fit`` call.
    :ivar returned: what the hook last returned.
    """

    kill = None
    replace = None

    def __init__(self, config):
        super().__init__(config)
        self.calls, self.fit_x, self.returned = [], [], None

    def build_model(self, config, context, data):
        model = super().build_model(config, context, data)
        real_fit = model.fit

        def spy_fit(*args, **kwargs):
            self.fit_x.append(args[0])
            return real_fit(*args, **kwargs)

        model.fit = spy_fit
        return model

    def extra_callbacks(self, config, data):
        return [] if self.kill is None else [KillBefore(*self.kill)]

    def resume_stream(self, config, data, completed_steps):
        weights = Path(config.out) / "checkpoints" / "backup" / BACKUP_WEIGHTS_NAME
        try:
            in_the_backup = backup_iterations(weights)[0]
        except OSError:  # no copy yet (a fresh run) or a torn one: the assertions on the count say which
            in_the_backup = None
        self.calls.append((completed_steps, data, in_the_backup))
        self.returned = None if self.replace is None else self.replace(data)
        return self.returned


def resuming(**attributes):
    """Return a ``Resuming`` subclass with the keyword attributes set (a callable becomes a static method)."""
    return type("Resuming", (Resuming,), {k: staticmethod(v) if callable(v) else v for k, v in attributes.items()})


def resume_config(out, **changes):
    return config_for(out, epochs=RESUME_EPOCHS, **changes)


def cut_in_half(path):
    """Leave the first half of ``path``, as a process killed inside the write does."""
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])


def killed_run(tmp_path_factory, name, kill, **changes):
    run_dir = tmp_path_factory.mktemp(name) / "run"
    with pytest.raises(Killed):
        resuming(kill=kill)(resume_config(run_dir, **changes)).run()
    return run_dir


@pytest.fixture(scope="module")
def killed_at_an_epoch_end(world, tmp_path_factory):
    """Killed before the first batch of epoch 3: epochs 0 to 2 done, the latest copy after 3 epochs, the previous after 2."""
    return killed_run(tmp_path_factory, "epoch_end", (3, 0))


@pytest.fixture(scope="module")
def killed_inside_an_epoch(world, tmp_path_factory):
    """Killed before batch 6 of epoch 1 with a copy every 4 batches: the latest copy is after 4 batches of epoch 1."""
    return killed_run(tmp_path_factory, "mid_epoch", (1, 6), **{"checkpoint_args.backup_every_steps": EVERY})


@pytest.fixture(scope="module")
def finished(world, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("finished") / "run"
    task = resuming()(resume_config(run_dir))
    task.run()
    return task, run_dir


def resumed(killed, tmp_path, task_class=None, name="copy", **changes):
    run_dir = tmp_path / name
    shutil.copytree(killed, run_dir)
    task = (task_class or resuming())(resume_config(run_dir, resume_interrupted=True, **changes))
    task.run()
    return task, run_dir


def test_a_fresh_run_never_calls_the_hook(finished):
    task, _ = finished
    assert task.calls == [], "resume_stream is for a resume"


def test_a_reporting_only_resume_never_calls_the_hook(finished, tmp_path):
    _, whole = finished
    run_dir = tmp_path / "crashed_after_fit"
    shutil.copytree(whole, run_dir)
    (run_dir / "results_summary.json").unlink()
    task = resuming()(resume_config(run_dir, resume_interrupted=True))
    task.run()
    assert (run_dir / "results_summary.json").is_file(), "vacuity guard: the run was resumed and reported"
    assert task.calls == [] and task.fit_x == [], "no fit and no new stream for a run with no epoch left"


def test_a_resume_killed_at_an_epoch_end_gets_the_steps_of_the_whole_epochs(killed_at_an_epoch_end, tmp_path):
    task, _ = resumed(killed_at_an_epoch_end, tmp_path)
    assert len(task.calls) == 1, "once"
    completed, data, in_the_backup = task.calls[0]
    # 3 epochs done, batch 0: start * steps_per_epoch, and no batch of an epoch in progress.
    assert data.steps_per_epoch == SPE
    assert completed == 3 * SPE == in_the_backup


def test_a_resume_from_a_mid_epoch_copy_counts_the_batches_of_the_copy(killed_inside_an_epoch, tmp_path):
    task, _ = resumed(killed_inside_an_epoch, tmp_path, **{"checkpoint_args.backup_every_steps": EVERY})
    assert len(task.calls) == 1
    completed, data, in_the_backup = task.calls[0]
    # Epoch 1, 4 batches taken (the metadata names the last one, batch index 3): 9 + (3 + 1).
    assert completed == 1 * SPE + EVERY == in_the_backup
    assert data.steps_per_epoch == SPE


def test_the_count_is_the_settled_one_when_the_latest_copy_is_torn(killed_at_an_epoch_end, tmp_path):
    run_dir = tmp_path / "torn"
    shutil.copytree(killed_at_an_epoch_end, run_dir)
    cut_in_half(run_dir / "checkpoints" / "backup" / BACKUP_WEIGHTS_NAME)
    task = resuming()(resume_config(run_dir, resume_interrupted=True))
    task.run()
    completed, _, in_the_backup = task.calls[0]
    # The torn latest copy (27 steps) is replaced by the previous one: the run goes back to epoch 2 (18 steps).
    assert completed == 2 * SPE == in_the_backup != 3 * SPE


def test_a_returned_object_replaces_the_data_the_fit_sees(killed_at_an_epoch_end, tmp_path):
    def other_stream(data):
        return dataclasses.replace(data, train_ds=data.train_ds.map(lambda x, y: (x, y)))

    task, _ = resumed(killed_at_an_epoch_end, tmp_path, resuming(replace=other_stream))
    _, original, _ = task.calls[0]
    assert task.returned is not None and task.returned.train_ds is not original.train_ds
    assert len(task.fit_x) == 1 and task.fit_x[0] is task.returned.train_ds


def test_none_leaves_the_data_as_it_was(killed_at_an_epoch_end, tmp_path):
    task, _ = resumed(killed_at_an_epoch_end, tmp_path)
    _, original, _ = task.calls[0]
    assert task.returned is None
    assert len(task.fit_x) == 1 and task.fit_x[0] is original.train_ds


def test_the_default_hook_returns_none_for_any_arguments(world, tmp_path):
    task = ImageClassificationTrainer(resume_config(tmp_path / "run"))
    assert task.resume_stream(task.config, object(), 27) is None


@pytest.mark.parametrize(
    "field, change",
    [
        ("steps_per_epoch", lambda data: data.steps_per_epoch + 1),
        ("n_train", lambda data: data.n_train + 1),
        ("n_select", lambda data: data.n_select - 1),
        ("n_test", lambda data: data.n_test + 1),
        ("fingerprint", lambda data: "0" * 64),
    ],
)
def test_a_replacement_that_changes_the_data_it_replaces_is_refused_by_name(
    killed_at_an_epoch_end, tmp_path, field, change
):
    def changed(data):
        return dataclasses.replace(data, **{field: change(data)})

    run_dir = tmp_path / "copy"
    shutil.copytree(killed_at_an_epoch_end, run_dir)
    task = resuming(replace=changed)(resume_config(run_dir, resume_interrupted=True))
    with pytest.raises(ConfigError, match=rf"resume_stream of Resuming.*\b{field}\b") as raised:
        task.run()
    assert "was claimed before this failed" in str(raised.value)
    assert task.fit_x == [], "refused before fit"
