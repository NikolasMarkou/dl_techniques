"""The stream contract of the shared sequence: ``fit_kwargs`` and the refusal of a stream with no length.

Plan 46e5318f step 3 (D-003). Keras 3.8 trains an epoch of an infinite ``train_ds`` forever when ``fit``
is given neither ``steps_per_epoch`` nor a dataset of known cardinality, so a task whose stream repeats
(the self-supervised image streams do) must say how long an epoch is, through ``fit_kwargs``. The sequence
refuses what would hang or what disagrees with the epoch it computed everything else from:

* a ``steps_per_epoch`` that is not a plain int, or differs from ``data.steps_per_epoch`` (the schedule, the
  backup arithmetic and ``settle_backup`` read that number, so ``fit`` must run the same one);
* a ``train_ds`` of infinite or unknown cardinality with no ``steps_per_epoch`` (a ``ConfigError`` that names
  ``fit_kwargs`` and ``steps_per_epoch`` and carries the claimed-run note, instead of a hang);
* with ``steps_per_epoch`` given the same stream trains, and ONE iterator persists across epochs: epoch
  1 continues epoch 0's stream (Keras 3.8, MEASURED in the plan's findings), so a counter-keyed stream does
  not restart.

The data are the real classification pipeline on fake CIFAR-10 arrays; the stream under test is that
pipeline's own ``train_ds`` made infinite (``repeat``) or unknown (``filter``) inside ``build_data``. Every
test that could hang if the guard were gone carries a callback that stops the run, so a broken guard is a
named failure and not a stuck suite. The tests write only under ``tmp_path``.
"""

import dataclasses

import keras
import numpy as np
import pytest
import tensorflow as tf

import train.api.recipes as recipes
from train.api import registry
from train.api.errors import ConfigError
from train.api.tasks import register_builtin_tasks
from train.api.tasks.image_classification import ImageClassificationTrainer

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
