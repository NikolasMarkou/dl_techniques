"""``TaskTrainer.validation_stream(config, data)``: the one hook a task with no validation needs.

Plan e3381378 (Phase 4k) step 3, D-007. A DINO loss moves its centre inside ``call()``, so every batch Keras
evaluates through ``fit(validation_data=...)`` is a second, unwanted centre update; the shared sequence used to
hand ``data.select_ds`` to ``fit`` always. The hook answers what ``validation_data`` is:

* the DEFAULT returns the very ``select_ds`` object the sequence passed before (identity, not equality), so no
  earlier task moves a byte, and the keyword set of the one ``fit`` call is still
  ``{validation_data, epochs, callbacks, verbose}``;
* a task that answers ``None`` reaches ``fit`` as ``validation_data=None``: Keras then runs no evaluation, the CSV
  holds no ``val_`` column, and no ``ReduceLROnPlateau`` is built for a decaying schedule (it would watch a
  ``val_loss`` that does not exist);
* through the real sequence a DINO-shaped task advances the loss centre exactly ``steps_per_epoch`` times per
  epoch: the legacy README Rule 1 (``tests/test_train/test_dino/test_train_dino.py::TestNoValidationData``)
  states the failure this prevents (a validation set doubled the updates, MEASURED 4 instead of 2). The counter is
  a ``DINOLoss`` subclass that adds one to an in-graph variable per execution of ``call`` (a Python counter would
  count traces), and a control with a real validation stream shows the counter DOES see the extra batches.

The tests write only under ``tmp_path``; the world is fake CIFAR-10 arrays on the CPU.
"""

import csv
import dataclasses
import types

import keras
import numpy as np
import pytest
import tensorflow as tf

import train.api.recipes as recipes
from train.api import registry
from train.api.tasks import register_builtin_tasks
from train.api.tasks.image_classification import ImageClassificationTrainer
from train.api.trainer import TaskTrainer

from tests.classification_fakes import use_tiny_raw

from .test_task_trainer import make_config
from .test_the_sequence_seams_default_to_nothing import FIT_KWARGS

EPOCHS = 2
#: ``watch_metric`` "loss": the one metric a run with no validation stream reports under every task.
REQUEST = {
    "epochs": EPOCHS,
    "metric_args.watch_metric": "loss",
    "metric_args.success_threshold": 0.0,
    "analysis_args.figures": False,
}
#: The tasks that override ``validation_stream``. The dino task (plan-2026-11-26T224507-e3381378 step 6) is the one of them.
OVERRIDERS = {"DinoTrainer"}  # class names

IMAGE = 32
PATCH = 16
OUT_DIM = 16
N_LOCAL = 1
N_VIEWS = 2 + N_LOCAL


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
    return make_config(out, **{**REQUEST, **changes})


class Stop(RuntimeError):
    """Raised by :class:`StopAtTheEnd`: the fit is over and the rest of the sequence is not under test."""


class StopAtTheEnd(keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        raise Stop("fit finished")


class Spying(ImageClassificationTrainer):
    """The real classification task whose model records every ``fit`` call; ``stream`` says what the hook answers.

    :cvar stream: ``"default"`` keeps the shared hook; ``"none"`` answers ``None``.
    :ivar fit_calls: the ``(args, kwargs)`` of every ``fit`` call of the model.
    :ivar callbacks: the callback list of the last fit.
    """

    stream = "default"
    #: The classification success value reads ``val_accuracy``, which a run with no validation never reports.
    success_metric = "accuracy"

    def __init__(self, config):
        super().__init__(config)
        self.fit_calls = []
        self.callbacks = []

    def build_model(self, config, context, data):
        model = super().build_model(config, context, data)
        real_fit = model.fit

        def spy_fit(*args, **kwargs):
            self.fit_calls.append((args, dict(kwargs)))
            return real_fit(*args, **kwargs)

        model.fit = spy_fit
        return model

    def validation_stream(self, config, data):
        return None if self.stream == "none" else super().validation_stream(config, data)

    def _callbacks(self, config, data, run_dir, *rest):
        epoch_line, callbacks = super()._callbacks(config, data, run_dir, *rest)
        self.callbacks = list(callbacks)
        return epoch_line, callbacks


def with_stream(stream):
    return type(f"Spying_{stream}", (Spying,), {"stream": stream})


# ---------------------------------------------------------------------
# The default is the object the sequence passed before
# ---------------------------------------------------------------------


def test_the_default_returns_the_very_selection_stream_object():
    sentinel, data = object(), types.SimpleNamespace()
    data.select_ds = sentinel
    assert TaskTrainer.validation_stream(object.__new__(TaskTrainer), None, data) is sentinel


def test_the_hook_takes_the_config_and_the_data_and_nothing_else():
    import inspect

    assert list(inspect.signature(TaskTrainer.validation_stream).parameters) == ["self", "config", "data"]


def shipped_trainers():
    """Every ``TaskTrainer`` subclass defined in a module of ``train.api.tasks`` (the tests' own subclasses are not)."""
    import importlib
    import pkgutil

    import train.api.tasks as tasks

    for module in pkgutil.iter_modules(tasks.__path__):
        importlib.import_module(f"{tasks.__name__}.{module.name}")
    found, queue = {}, list(TaskTrainer.__subclasses__())
    while queue:
        cls = queue.pop()
        queue.extend(cls.__subclasses__())
        if cls.__module__.startswith(tasks.__name__ + "."):
            found[cls.__name__] = cls
    return found


def test_no_shipped_task_overrides_the_hook_but_the_ones_named():
    trainers = shipped_trainers()
    assert len(trainers) >= 9, sorted(trainers)
    overriding = {name for name, cls in trainers.items() if cls.validation_stream is not TaskTrainer.validation_stream}
    assert overriding == OVERRIDERS, "a task that overrides the hook joins OVERRIDERS in the step that adds it"


def test_the_seams_file_pins_the_fit_keyword_set_this_hook_must_not_change():
    assert FIT_KWARGS == {"validation_data", "epochs", "callbacks", "verbose"}


def test_a_default_task_hands_fit_the_selection_stream_and_the_same_keywords(world, tmp_path):
    task = with_stream("default")(config_for(tmp_path / "run", **{"metric_args.watch_metric": "accuracy"}))
    seen = {}

    def spy_build_data(config, context, real=task.build_data):
        data = seen["data"] = real(config, context)
        return data

    task.build_data = spy_build_data
    task.run()
    (args, kwargs), = task.fit_calls
    assert len(args) == 1 and set(kwargs) == FIT_KWARGS
    assert kwargs["validation_data"] is seen["data"].select_ds
    with open(tmp_path / "run" / "training_log.csv") as handle:
        header = next(csv.reader(handle))
    assert "val_accuracy" in header and "val_loss" in header


# ---------------------------------------------------------------------
# A task that answers None
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def stub_run(world, tmp_path_factory):
    """A real fit through the real sequence with ``validation_stream`` answering ``None``."""
    run_dir = tmp_path_factory.mktemp("stub") / "run"
    task = with_stream("none")(config_for(run_dir, **{"metric_args.watch_metric": "accuracy"}))
    task.run()
    with open(run_dir / "training_log.csv") as handle:
        rows = list(csv.DictReader(handle))
    return task, rows


def test_a_none_answer_reaches_fit_as_no_validation_data_and_the_keywords_stay(stub_run):
    task, _ = stub_run
    (args, kwargs), = task.fit_calls
    assert len(args) == 1 and set(kwargs) == FIT_KWARGS
    assert kwargs["validation_data"] is None


def test_a_none_answer_leaves_no_val_column_in_the_csv(stub_run):
    _, rows = stub_run
    assert len(rows) == EPOCHS
    columns = list(rows[0])
    assert "loss" in columns and "accuracy" in columns
    assert [name for name in columns if name.startswith("val_")] == []


def test_a_none_answer_builds_no_plateau_callback_for_a_decaying_schedule(stub_run):
    task, _ = stub_run
    assert task.config.optim_args.schedule != "constant"
    assert not [c for c in task.callbacks if isinstance(c, keras.callbacks.ReduceLROnPlateau)]
    assert [c for c in task.callbacks if isinstance(c, keras.callbacks.EarlyStopping)], "the stopper is still there"


# ---------------------------------------------------------------------
# A DINO-shaped task: the centre advances once per training step and never for a validation batch
# ---------------------------------------------------------------------


def counting_loss_class():
    from dl_techniques.losses.dino_loss import DINOLoss

    class CountingDINOLoss(DINOLoss):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.executions = keras.Variable(0, dtype="int32", trainable=False, name="executions")

        def call(self, y_true, y_pred):
            self.executions.assign_add(1)
            return super().call(y_true, y_pred)

    return CountingDINOLoss


class DinoShaped(Spying):
    """A classification run whose model is a ``DINOTrainingModel`` fed by three views of each image.

    Only what the claim needs: the data's ``train_ds`` and ``select_ds`` yield ``(views, label)`` with rank-5 views, the model is the
    library's, the loss counts its executions, and the run stops (``Stop``) when ``fit`` ends: the classification
    reporting that follows reads rank-4 test batches and is not under test. ``validate`` is what the hook answers:
    ``"none"`` or ``"selection"`` (the shared default, the control that has to move the counter).
    """

    stream = "none"
    success_metric = "loss"
    losses = []

    def build_data(self, config, context):
        data = super().build_data(config, context)

        def views(images, labels):
            return tf.stack([images, images[:, ::-1], images[:, :, ::-1]], axis=1), labels

        return dataclasses.replace(data, train_ds=data.train_ds.map(views), select_ds=data.select_ds.map(views))

    def build_model(self, config, context, data):
        from dl_techniques.models.vision.dino.training import create_dino_training_model

        keras.utils.set_random_seed(0)
        model = create_dino_training_model(
            "tiny", image_size=IMAGE, patch_size=PATCH, n_local_views=N_LOCAL, dino_out_dim=OUT_DIM
        )
        model.build((None, N_VIEWS, IMAGE, IMAGE, 3))
        real_fit = model.fit

        def spy_fit(*args, **kwargs):
            self.fit_calls.append((args, dict(kwargs)))
            return real_fit(*args, **kwargs)

        model.fit = spy_fit
        return model

    def metric_names(self, config, facts):
        return []

    def build_loss_and_metrics(self, config, data):
        loss = counting_loss_class()(out_dim=OUT_DIM)
        self.losses.append(loss)
        return loss, []

    def initial_loss_check(self, config, model, data):
        return {"loss": 1.0}, 1.0, False

    def extra_callbacks(self, config, data):
        return [StopAtTheEnd()]


def dino_run(tmp_path, stream):
    task_class = type(f"Dino_{stream}", (DinoShaped,), {"stream": stream, "losses": []})
    task = task_class(config_for(tmp_path / "run"))
    seen = {}
    real = task.build_data
    task.build_data = lambda config, context: seen.setdefault("data", real(config, context))
    with pytest.raises(Stop):
        task.run()
    loss, = task.losses
    return task, seen["data"], loss


def test_a_dino_shaped_fit_advances_the_centre_exactly_once_per_training_step(world, tmp_path):
    task, data, loss = dino_run(tmp_path, "none")
    (args, kwargs), = task.fit_calls
    assert kwargs["validation_data"] is None
    assert data.steps_per_epoch > 1
    assert int(loss.executions.numpy()) == EPOCHS * data.steps_per_epoch
    assert float(np.abs(keras.ops.convert_to_numpy(loss.center)).max()) > 0.0, "the centre moved off its zero start"


def test_the_control_with_a_validation_stream_moves_the_centre_more_often(world, tmp_path):
    """The counter sees validation batches: with the shared default the same fit executes the loss more often."""
    task, data, loss = dino_run(tmp_path, "default")
    (args, kwargs), = task.fit_calls
    assert kwargs["validation_data"] is data.select_ds
    assert int(loss.executions.numpy()) > EPOCHS * data.steps_per_epoch
