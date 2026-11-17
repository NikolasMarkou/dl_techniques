"""The five seams of the shared sequence default to nothing and take effect when a task overrides them.

Plan d483ffab step 3 (D-004, D-011, D-012, D-013) gave ``TaskTrainer`` and its sequence the room a
denoising task needs without changing what classification and segmentation write:

* ``ModelSpec.properties`` (default ``()``), checked against ``KNOWN_PROPERTIES``;
* ``TaskTrainer.compile_options(config, spec)`` (default ``{}``), spread into the one ``compile`` call;
* ``TaskTrainer.selection_start_epoch(config)`` (default 0): ``AtomicModelCheckpoint`` saves nothing
  before it, ``EarlyStopping`` counts nothing before it, and the epoch the summary calls best is the
  best one from it on;
* ``TaskTrainer.config_stamps(config, spec)`` (default ``{}``), merged at the top level of
  ``config.json`` by ``write_resolved`` and never overwriting a key;
* ``Evaluation.extras`` (merged into ``results_summary.json``, never over a key that is there) and
  ``Evaluation.failed_checks`` (a non-empty tuple makes ``success`` false and is written as
  ``success_blocked_by``).

Plan 46e5318f step 3 (D-003) adds two more hooks, ``TaskTrainer.fit_kwargs(config, data)`` (default
``{}``, spread into the one ``fit`` call and stored on ``Run.fit_options``) and
``TaskTrainer.resume_stream(config, data, completed_steps)`` (default ``None``; step 4, D-004, makes the
sequence call it once for a resume that trains): the default arm below reads the keywords the one ``fit``
call of a shipped task carries, and the positive arm shows a task's ``fit_kwargs`` reaching ``fit``. The
refusals that need a stream (a mismatched ``steps_per_epoch``, an infinite ``train_ds``) and every case of
the ``resume_stream`` call, among them that a fresh run never asks for a new stream, that a ``None`` answer
leaves the data and that the default hook answers ``None``, are in ``test_the_stream_contract.py``.

Plan 9c7b9096 step 2 (D-003) adds ``TaskTrainer.optimizer_options(config, model)`` (default ``{}``), called
once with the built model and handed to ``build_optimizer_and_schedule(..., options=)``, which merges a
closed key set (``lr_multipliers`` only) into the builder's config. The default arm spies on
``optimizer_builder`` and compares the config it receives with literals read at the commit before the
seam (f6f15bedf); the positive arm gets an ``LRMultiplierAdamW`` holding the task's table; another key,
or a table with an optimizer other than adamw, is refused by name.

Every seam is proved twice, because a default that is "almost" nothing moves a byte in a golden and a
seam that does nothing passes every default test: the default arm is a real fit of ``vit/vit_pico`` on
fake CIFAR-10 arrays whose compile call, callback list, ``config.json`` keys and summary keys are read
back; the positive arm is the same fit with a task that overrides every seam. A resume on either side of
the selection window's start is run for real (a callback raises at the start of an epoch, as in
``test_resume_interrupted.py``). The byte level proof is the golden harness of the plan, not this file.
The tests write only under ``tmp_path``.
"""

import contextlib
import copy
import dataclasses
import hashlib
import json
import re
import shutil
from pathlib import Path

import keras
import pytest

import train.api.recipes as recipes
import train.api.run_sequence as run_sequence
from train.api import registry
from train.api.args import OptimArgs
from train.api.errors import ConfigError
from train.api.optim import build_optimizer_and_schedule
from train.api.registry import KNOWN_PROPERTIES, ModelSpec
from train.api.run_dir import write_resolved
from train.api.run_sequence import Run
from train.api.tasks import register_builtin_tasks
from train.api.tasks.image_classification import ImageClassificationConfig, ImageClassificationTrainer
from train.api.tasks.image_denoising import ImageDenoisingTrainer
from train.api.tasks.image_segmentation import ImageSegmentationTrainer
from train.api.tasks.pretrain import PretrainTrainer
from train.api.tasks.time_series_forecasting import TimeSeriesForecastingTrainer
from train.api.trainer import Evaluation, TaskTrainer
from train.common.callbacks import AtomicModelCheckpoint

from tests.classification_fakes import use_tiny_raw

from .test_task_trainer import make_config, read_json

EPOCHS = 4
#: ``val_accuracy`` of each epoch: epoch 0 is the best of the whole run, epoch 2 the best from epoch 2 on.
SCRIPT = (0.9, 0.8, 0.3, 0.2)
START = 2
REQUEST = {
    "epochs": EPOCHS,
    "metric_args.success_threshold": 0.0,
    "checkpoint_args.early_stopping_patience": EPOCHS,
    "analysis_args.figures": False,
}
BASE_KWARGS = {"optimizer", "loss", "metrics"}
#: What the one ``fit`` call of a shipped task carries besides the positional ``x``.
FIT_KWARGS = {"validation_data", "epochs", "callbacks", "verbose"}
#: The names of ``model.fit`` the sequence passes itself (or owns): a task's ``fit_kwargs`` may not name them.
RESERVED_FIT_NAMES = ("x", "y", "validation_data", "validation_split", "epochs", "callbacks", "verbose",
                      "initial_epoch")


class ScriptedMonitor(keras.callbacks.Callback):
    """Write the scripted ``val_accuracy`` into the epoch logs; sits before EarlyStopping."""

    def on_epoch_end(self, epoch, logs=None):
        logs["val_accuracy"] = SCRIPT[epoch]


class KillAt(keras.callbacks.Callback):
    """Raise at the start of epoch ``epoch``: a stand-in for a kill between two epochs."""

    def __init__(self, epoch):
        super().__init__()
        self.epoch = epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.epoch:
            raise RuntimeError("killed")


class SeeBest(keras.callbacks.Callback):
    """Record, after every epoch, whether ``best_model.keras`` exists (placed after the checkpoint)."""

    def __init__(self, run_dir, seen):
        super().__init__()
        self.path = Path(run_dir) / "best_model.keras"
        self.seen = seen

    def on_epoch_end(self, epoch, logs=None):
        self.seen[epoch] = self.path.is_file()


class Scripted(ImageClassificationTrainer):
    """The real task with a scripted monitor and recorders; it overrides NO seam (the default arm).

    ``seen_callbacks`` is the callback list of the last fit, ``seen_best`` the epochs after which
    ``best_model.keras`` existed, ``compile_calls`` and ``fit_calls`` the ``(args, kwargs)`` of every
    ``compile`` and ``fit`` call of the model.
    """

    kill_at = None

    def __init__(self, config):
        super().__init__(config)
        self.seen_callbacks = []
        self.seen_best = {}
        self.compile_calls = []
        self.fit_calls = []

    def build_model(self, config, context, data):
        """Spy on the model's own ``compile`` and ``fit`` (instance attributes: Keras refuses a
        class-level replacement at load time, because it reads an overridden ``compile`` as a custom one)."""
        model = super().build_model(config, context, data)
        real, real_fit = model.compile, model.fit

        def spy(*args, **kwargs):
            self.compile_calls.append((args, dict(kwargs)))
            return real(*args, **kwargs)

        def spy_fit(*args, **kwargs):
            self.fit_calls.append((args, dict(kwargs)))
            return real_fit(*args, **kwargs)

        model.compile, model.fit = spy, spy_fit
        return model

    def _callbacks(self, config, data, run_dir, *rest):
        epoch_line, callbacks = super()._callbacks(config, data, run_dir, *rest)
        callbacks.insert(1, ScriptedMonitor())
        callbacks.append(SeeBest(run_dir, self.seen_best))
        self.seen_callbacks = list(callbacks)
        return epoch_line, callbacks

    def extra_callbacks(self, config, data):
        return [] if self.kill_at is None else [KillAt(self.kill_at)]


class Overriding(Scripted):
    """``Scripted`` with every seam overridden; subclasses set the class attributes below."""

    options = {}
    start = 0
    stamps = {}
    extras = {}
    failed = ()

    def compile_options(self, config, spec):
        return dict(self.options)

    def selection_start_epoch(self, config):
        return self.start

    def config_stamps(self, config, spec):
        return dict(self.stamps)

    def evaluate_and_report(self, config, data, fit):
        evaluation = super().evaluate_and_report(config, data, fit)
        return dataclasses.replace(evaluation, extras=dict(self.extras), failed_checks=tuple(self.failed))


class Windowed(Overriding):
    start = START


def killed_at(epoch):
    return type(f"KilledAt{epoch}", (Windowed,), {"kill_at": epoch})


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        use_tiny_raw(patch, n_train=40, n_test=8)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextlib.contextmanager
def spying_on_the_optimizer_builder():
    """Yield a list that receives a deep copy of every config ``optimizer_builder`` is called with.

    ``build_optimizer_and_schedule`` imports the builder from ``dl_techniques.optimization`` at call
    time, so replacing the package attribute is enough; the real builder still builds the optimizer.
    """
    import dl_techniques.optimization as optimization

    seen, real = [], optimization.optimizer_builder

    def spy(config, schedule):
        seen.append(copy.deepcopy(config))
        return real(config, schedule)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(optimization, "optimizer_builder", spy)
        yield seen


def config_for(out, **changes):
    return make_config(out, **{**REQUEST, **changes})


# ---------------------------------------------------------------------
# Seam 1: ModelSpec.properties
# ---------------------------------------------------------------------


def spec(**changes):
    return ModelSpec(family="fam", factory="a.b:make", variants_attr="a.b:TABLE", tasks=("t",), **changes)


def test_the_known_properties_are_pinned():
    assert KNOWN_PROPERTIES == ("bias_free",)


def test_a_spec_has_no_property_unless_it_names_one(world):
    assert spec().properties == ()
    assert registry.MODELS, "vacuity guard: the built-in families are registered"
    # Step 7 (plan d483ffab): the three bias-free denoisers name the property; no classifier, segmenter or biased family does.
    assert {family: s.properties for family, s in registry.MODELS.items() if s.properties} == {
        "bfcnn": ("bias_free",), "bfunet": ("bias_free",), "bfconvunext": ("bias_free",),
    }


def test_a_known_property_is_accepted():
    assert spec(properties=("bias_free",)).properties == ("bias_free",)


@pytest.mark.parametrize(
    "bad",
    [("unknown",), "bias_free", ["bias_free"], ("bias_free", "bias_free"), (None,)],
    ids=["unknown name", "a bare string", "a list", "a duplicate", "a non-string"],
)
def test_an_unknown_or_malformed_property_is_refused(bad):
    with pytest.raises(ValueError, match="properties"):
        spec(properties=bad)


# ---------------------------------------------------------------------
# The five hooks and the two fields default to nothing
# ---------------------------------------------------------------------


@pytest.mark.parametrize("task_class", [ImageClassificationTrainer, ImageSegmentationTrainer])
@pytest.mark.parametrize("hook", ["compile_options", "selection_start_epoch", "config_stamps"])
def test_neither_existing_task_overrides_a_seam(task_class, hook):
    assert getattr(task_class, hook) is getattr(TaskTrainer, hook)


# The denoiser (plan d483ffab) overrides the first three seams by design; the two stream hooks are
# nobody's yet, and a task that overrides one is the first to need it.
@pytest.mark.parametrize(
    "task_class", [ImageClassificationTrainer, ImageSegmentationTrainer, ImageDenoisingTrainer]
)
@pytest.mark.parametrize("hook", ["fit_kwargs", "resume_stream"])
def test_no_existing_task_overrides_a_stream_hook(task_class, hook):
    assert getattr(task_class, hook) is getattr(TaskTrainer, hook)


def test_the_hook_defaults_are_nothing(world, tmp_path):
    task = ImageClassificationTrainer(config_for(tmp_path / "run"))
    family = registry.MODELS["vit"]
    assert task.compile_options(task.config, family) == {}
    assert task.selection_start_epoch(task.config) == 0
    assert task.config_stamps(task.config, family) == {}
    assert task.fit_kwargs(task.config, None) == {}


def test_a_run_holds_no_fit_option_until_the_task_names_one():
    field = {f.name: f for f in dataclasses.fields(Run)}["fit_options"]
    assert field.default_factory() == {}
    assert field.default_factory() is not field.default_factory()


def test_an_evaluation_carries_no_extras_and_no_failed_check_by_default():
    evaluation = Evaluation(None, None, None, None, {}, {}, {}, None)
    assert evaluation.extras == {} and evaluation.failed_checks == ()
    # Two evaluations never share one mapping.
    other = Evaluation(None, None, None, None, {}, {}, {}, None)
    assert evaluation.extras is not other.extras


# ---------------------------------------------------------------------
# The default arm: a real classification fit, read back
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def default_run(world, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("default") / "run"
    task = Scripted(make_config(run_dir, **{**REQUEST, "analysis_args.figures": True}))
    with spying_on_the_optimizer_builder() as seen:
        result = task.run()
    task.builder_configs = seen
    return task, result, run_dir


def test_a_default_task_compiles_with_exactly_the_three_arguments(default_run):
    task, _, _ = default_run
    assert len(task.compile_calls) == 1
    args, kwargs = task.compile_calls[0]
    assert args == () and set(kwargs) == BASE_KWARGS


def test_the_default_callback_list_has_no_selection_start(default_run):
    task, _, _ = default_run
    stopper = next(c for c in task.seen_callbacks if isinstance(c, keras.callbacks.EarlyStopping))
    checkpoint = next(c for c in task.seen_callbacks if isinstance(c, AtomicModelCheckpoint))
    assert stopper.start_from_epoch == 0 and checkpoint.start_from_epoch == 0
    assert [type(c).__name__ for c in task.seen_callbacks] == [
        "LearningRateLogger", "ScriptedMonitor", "TerminateOnNaN", "EarlyStopping", "AtomicModelCheckpoint",
        "EpochCSVLogger", "TensorBoard", "LastEpochWeights", "EpochLogLine", "TrainingDashboardCallback",
        "FitStateSidecar", "BackupAndRestore", "SeeBest",
    ]


def test_a_default_task_calls_fit_with_exactly_the_arguments_it_always_passed(default_run):
    task, _, _ = default_run
    assert len(task.fit_calls) == 1
    args, kwargs = task.fit_calls[0]
    # ``x`` is the one positional argument; ``steps_per_epoch`` and ``validation_steps`` are NOT there
    # (``steps_per_epoch=None`` would be a value to Keras, and an extra key moves nothing visible but this).
    assert len(args) == 1 and set(kwargs) == FIT_KWARGS


def test_the_default_config_json_holds_the_config_fields_and_used_only(default_run):
    task, _, run_dir = default_run
    fields = {f.name for f in dataclasses.fields(ImageClassificationConfig)}
    assert set(read_json(run_dir, "config.json")) == fields | {"used"}


def test_the_default_summary_has_no_extra_key_and_is_not_blocked(default_run):
    _, result, run_dir = default_run
    summary = read_json(run_dir, "results_summary.json")
    assert "success_blocked_by" not in summary
    assert summary["success"] is True and result.success is True
    # Epoch 0 is the best of the whole run: the window is the whole run.
    assert summary["best_epoch"] == 1 and summary["final_is_best"] is False
    keys = list(summary)
    assert keys.index("success") < keys.index("success_metric") < keys.index("success_threshold") \
        < keys.index("success_value")


# ---------------------------------------------------------------------
# The positive arm: every seam overridden at once
# ---------------------------------------------------------------------


class Everything(Windowed):
    options = {"jit_compile": False}
    stamps = {"data_range": "[0,1]", "stamped_patch": 64}
    extras = {"held_out": {"kodak24": {"15": 31.5}}, "homogeneity": "not_applicable"}
    failed = ("homogeneity: numeric probe 0.2 above 0.001",)


@pytest.fixture(scope="module")
def everything(world, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("everything") / "run"
    task = Everything(config_for(run_dir))
    result = task.run()
    return task, result, run_dir


def test_compile_options_are_spread_into_the_one_compile_call(everything):
    task, _, _ = everything
    assert len(task.compile_calls) == 1
    _, kwargs = task.compile_calls[0]
    assert set(kwargs) == BASE_KWARGS | {"jit_compile"} and kwargs["jit_compile"] is False


def test_the_selection_start_reaches_both_callbacks(everything):
    task, _, _ = everything
    stopper = next(c for c in task.seen_callbacks if isinstance(c, keras.callbacks.EarlyStopping))
    checkpoint = next(c for c in task.seen_callbacks if isinstance(c, AtomicModelCheckpoint))
    assert stopper.start_from_epoch == START and checkpoint.start_from_epoch == START


def test_no_best_checkpoint_exists_before_the_window_and_one_does_from_its_first_epoch(everything):
    task, _, _ = everything
    assert task.seen_best == {0: False, 1: False, 2: True, 3: True}


def test_the_best_epoch_is_the_best_one_from_the_window_on_and_is_the_saved_one(everything):
    _, _, run_dir = everything
    summary = read_json(run_dir, "results_summary.json")
    # Epoch 1 (index 0) scored 0.9, but no checkpoint may be taken there: the saved best is epoch 3.
    assert summary["best_epoch"] == START + 1
    assert summary["best_checkpoint_load_error"] is None
    assert summary["best_checkpoint_max_abs_diff"] < 1e-5


def test_the_stamps_are_top_level_keys_of_config_json(everything):
    _, _, run_dir = everything
    config = read_json(run_dir, "config.json")
    assert config["data_range"] == "[0,1]" and config["stamped_patch"] == 64
    fields = {f.name for f in dataclasses.fields(ImageClassificationConfig)}
    assert set(config) == fields | {"used", "data_range", "stamped_patch"}


def test_the_extras_are_keys_of_the_summary(everything):
    _, _, run_dir = everything
    summary = read_json(run_dir, "results_summary.json")
    assert summary["held_out"] == {"kodak24": {"15": 31.5}}
    assert summary["homogeneity"] == "not_applicable"


def test_a_failed_check_makes_the_run_unsuccessful_whatever_the_bar_says(everything):
    _, result, run_dir = everything
    summary = read_json(run_dir, "results_summary.json")
    # The bar is 0.0, so it is met; the failed check alone blocks the run.
    assert summary["success_value"] >= summary["success_threshold"]
    assert summary["success"] is False and result.success is False
    assert summary["success_blocked_by"] == list(Everything.failed)


class Spreading(Scripted):
    """A task that names ``steps_per_epoch`` and ``validation_steps`` through ``fit_kwargs``."""

    asked = None

    def fit_kwargs(self, config, data):
        self.asked = {"steps_per_epoch": data.steps_per_epoch, "validation_steps": int(data.select_ds.cardinality())}
        return dict(self.asked)


@pytest.fixture(scope="module")
def spreading(world, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("spreading") / "run"
    task = Spreading(config_for(run_dir, epochs=2))
    task.run()
    return task


def test_the_fit_options_of_a_task_are_spread_into_the_one_fit_call(spreading):
    assert len(spreading.fit_calls) == 1
    args, kwargs = spreading.fit_calls[0]
    assert len(args) == 1 and set(kwargs) == FIT_KWARGS | {"steps_per_epoch", "validation_steps"}
    # Both values are the data's own, and both are positive: a spy that saw ``None`` or 0 would pass an equality.
    assert {k: kwargs[k] for k in ("steps_per_epoch", "validation_steps")} == spreading.asked
    assert all(isinstance(v, int) and v > 0 for v in spreading.asked.values())
    assert kwargs["epochs"] == 2 and kwargs["verbose"] == 1


# ---------------------------------------------------------------------
# Each collision and each bad value raises
# ---------------------------------------------------------------------


@pytest.mark.parametrize("key", RESERVED_FIT_NAMES)
def test_a_fit_option_may_not_replace_what_the_sequence_passes(world, tmp_path, key):
    class Reserved(Scripted):
        def fit_kwargs(self, config, data):
            return {key: None}

    run_dir = tmp_path / "run"
    with pytest.raises(ConfigError, match=rf"fit_kwargs of Reserved names \['{key}'\]") as raised:
        Reserved(config_for(run_dir, epochs=1)).run()
    assert "was claimed before this failed" in str(raised.value)
    assert not (run_dir / "training_log.csv").exists(), "refused before fit"


def test_the_reserved_names_are_the_arguments_of_fit_the_sequence_owns():
    # A name here that ``keras.Model.fit`` does not take would refuse nothing real; one the sequence passes
    # but this lacks would let a task collide with it.
    import inspect

    taken = set(inspect.signature(keras.Model.fit).parameters)
    assert set(RESERVED_FIT_NAMES) <= taken
    assert FIT_KWARGS | {"x"} <= set(RESERVED_FIT_NAMES)
    assert tuple(run_sequence._RESERVED_FIT_NAMES) == RESERVED_FIT_NAMES


@pytest.mark.parametrize("bad", [[("steps_per_epoch", 3)], 3, None], ids=["a list", "an int", "None"])
def test_a_fit_kwargs_that_is_not_a_mapping_is_refused_by_name(world, tmp_path, bad):
    class NotAMapping(Scripted):
        def fit_kwargs(self, config, data):
            return bad

    with pytest.raises(ConfigError, match="fit_kwargs of NotAMapping.*mapping"):
        NotAMapping(config_for(tmp_path / "run", epochs=1)).run()


# ---------------------------------------------------------------------
# Each collision and each bad value raises (compile and the rest)
# ---------------------------------------------------------------------


def test_a_stamp_equal_to_a_config_key_is_refused_and_nothing_is_written(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = config_for(run_dir)
    for key in ("epochs", "model_args", "used"):
        with pytest.raises(ConfigError, match=key):
            write_resolved(run_dir, config, {"a": 1}, {key: 1})
        assert not (run_dir / "config.json").exists()
    # Control: a stamp that is not a key writes the file, with the stamp at the top level.
    write_resolved(run_dir, config, {"a": 1}, {"data_range": "[0,1]"})
    assert read_json(run_dir, "config.json")["data_range"] == "[0,1]"


def test_write_resolved_without_stamps_writes_what_it_always_wrote(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = config_for(run_dir)
    write_resolved(run_dir, config, {"a": 1})
    assert read_json(run_dir, "config.json") == {**dataclasses.asdict(config), "used": {"a": 1}}


class CollidingStamp(Overriding):
    stamps = {"seed": 1}


def test_a_stamp_colliding_with_a_config_key_stops_the_run_before_training(world, tmp_path):
    run_dir = tmp_path / "run"
    with pytest.raises(ConfigError, match="seed"):
        CollidingStamp(config_for(run_dir)).run()
    assert not (run_dir / "config.json").exists() and not (run_dir / "training_log.csv").exists()


class CollidingExtras(Overriding):
    extras = {"task": "image_denoising"}


def test_an_extras_key_colliding_with_a_summary_key_is_refused(world, tmp_path):
    run_dir = tmp_path / "run"
    with pytest.raises(ConfigError, match="task"):
        CollidingExtras(config_for(run_dir, epochs=1)).run()
    assert not (run_dir / "results_summary.json").exists()


class CollidingSuccessExtras(Overriding):
    extras = {"success": True}


def test_an_extras_key_may_not_replace_the_verdict_either(world, tmp_path):
    with pytest.raises(ConfigError, match="success"):
        CollidingSuccessExtras(config_for(tmp_path / "run", epochs=1)).run()


@pytest.mark.parametrize("key", ["optimizer", "loss", "metrics"])
def test_a_compile_option_may_not_replace_what_the_sequence_passes(world, tmp_path, key):
    class Reserved(Overriding):
        options = {key: None}

    with pytest.raises(ConfigError, match=key):
        Reserved(config_for(tmp_path / "run", epochs=1)).run()


@pytest.mark.parametrize("bad", [-1, True, 1.5, EPOCHS, EPOCHS + 3])
def test_a_selection_start_outside_the_run_is_refused(world, tmp_path, bad):
    class Bad(Overriding):
        start = bad

    run_dir = tmp_path / "run"
    with pytest.raises(ValueError, match="selection_start_epoch"):
        Bad(config_for(run_dir)).run()
    assert not (run_dir / "results_summary.json").exists()


def test_the_checkpoint_callback_refuses_a_negative_start_and_skips_before_a_positive_one(tmp_path):
    with pytest.raises(ValueError, match="start_from_epoch"):
        AtomicModelCheckpoint(str(tmp_path / "best.keras"), start_from_epoch=-1)
    callback = AtomicModelCheckpoint(str(tmp_path / "best.keras"), monitor="val_loss", start_from_epoch=2)
    callback._save_model(0, None, {"val_loss": 1.0})
    callback._save_model(1, None, {"val_loss": 0.5})
    assert not list(tmp_path.iterdir()) and callback.best == float("inf")


# ---------------------------------------------------------------------
# A resume on either side of the window's start
# ---------------------------------------------------------------------


def killed_run(world, tmp_path_factory, name, epoch):
    run_dir = tmp_path_factory.mktemp(name) / "run"
    with pytest.raises(RuntimeError, match="killed"):
        killed_at(epoch)(config_for(run_dir)).run()
    return run_dir


@pytest.fixture(scope="module")
def killed_before_the_window(world, tmp_path_factory):
    """Killed at the start of epoch index 1: one epoch done, none inside the window."""
    return killed_run(world, tmp_path_factory, "before", 1)


@pytest.fixture(scope="module")
def killed_inside_the_window(world, tmp_path_factory):
    """Killed at the start of epoch index 3: epoch index 2, the window's first, is done and saved."""
    return killed_run(world, tmp_path_factory, "inside", 3)


def resume(killed, tmp_path_factory, name):
    run_dir = tmp_path_factory.mktemp(name) / "copy"
    shutil.copytree(killed, run_dir)
    task = Windowed(config_for(run_dir, resume_interrupted=True))
    return task, task.run(), run_dir


def test_a_run_killed_before_the_window_has_no_best_value_and_no_best_checkpoint(killed_before_the_window):
    sidecar = json.loads((killed_before_the_window / "checkpoints" / "fit_state.json").read_text())
    assert sidecar["epochs"]["0"]["best"] is None
    assert not (killed_before_the_window / "best_model.keras").exists()


def test_a_resume_before_the_window_selects_inside_it(killed_before_the_window, tmp_path_factory):
    task, _, run_dir = resume(killed_before_the_window, tmp_path_factory, "resumed_before")
    summary = read_json(run_dir, "results_summary.json")
    assert summary["resumed_after_epochs"] == 1
    assert summary["best_epoch"] == START + 1
    assert summary["best_checkpoint_load_error"] is None
    assert summary["best_checkpoint_max_abs_diff"] < 1e-5
    # Epoch indices 0 and 1 were not saved by this segment either, and the window's first was.
    assert task.seen_best == {1: False, 2: True, 3: True}


def test_a_resume_inside_the_window_keeps_the_best_epoch_and_its_file(killed_inside_the_window, tmp_path_factory):
    before = sha(killed_inside_the_window / "best_model.keras")
    _, _, run_dir = resume(killed_inside_the_window, tmp_path_factory, "resumed_inside")
    summary = read_json(run_dir, "results_summary.json")
    assert summary["resumed_after_epochs"] == START + 1
    assert summary["best_epoch"] == START + 1
    # The resumed epoch scored 0.2, below the window's 0.3: the file is the earlier segment's, untouched.
    assert sha(run_dir / "best_model.keras") == before
    assert summary["best_checkpoint_max_abs_diff"] < 1e-5


# ---------------------------------------------------------------------
# The optimizer seam (plan 9c7b9096 step 2, D-003)
# ---------------------------------------------------------------------

#: What ``optimizer_builder`` received at f6f15bedf, the commit before the seam, in key order: the
#: default arm's classification fit, and four ``OptimArgs`` arms that reach every key the mapping writes.
#: Read by a spy at that commit (``derive_base_configs.py`` of the plan), not typed from the code.
BASE_DEFAULT_RUN_CONFIG = [("type", "adamw"), ("gradient_clipping_by_norm", 1.0), ("weight_decay", 0.05)]
OPTIM_ARMS = {
    "adamw_global_clip": (
        {},
        [("type", "adamw"), ("gradient_clipping_by_norm", 1.0), ("weight_decay", 0.05)],
    ),
    "adamw_no_decay_local_clip_betas": (
        {"weight_decay": None, "clipnorm": 0.5, "global_clipnorm": None, "beta_1": 0.8, "beta_2": 0.95,
         "epsilon": 1e-7},
        [("type", "adamw"), ("gradient_clipping_by_norm_local", 0.5), ("beta_1", 0.8), ("beta_2", 0.95),
         ("epsilon", 1e-07), ("weight_decay", 0.0)],
    ),
    "sgd_momentum": (
        {"optimizer": "sgd", "lr": 0.1, "weight_decay": 1e-4, "momentum": 0.9, "global_clipnorm": None},
        [("type", "sgd"), ("momentum", 0.9)],
    ),
    "adam_constant": (
        {"optimizer": "adam", "weight_decay": None, "schedule": "constant", "warmup_epochs": 0},
        [("type", "adam"), ("gradient_clipping_by_norm", 1.0)],
    ),
}
OPTIM_BASE = OptimArgs(
    optimizer="adamw", lr=3e-4, weight_decay=0.05, momentum=None, beta_1=None, beta_2=None, epsilon=None,
    schedule="cosine_decay", warmup_epochs=5, warmup_start_lr=1e-8, min_lr_ratio=0.01, decay_over="total",
    clipnorm=None, global_clipnorm=1.0,
)
#: ``options`` values that must all leave the builder's config as it was before the seam; ``"absent"`` is
#: the call without the keyword. An empty table is no table (the depth task's from-scratch start has none).
NO_OPTIONS = {"absent": "absent", "None": None, "empty": {}, "empty table": {"lr_multipliers": {}}}


def optim_arm(name):
    changes, expected = OPTIM_ARMS[name]
    return dataclasses.replace(OPTIM_BASE, **changes), expected


@pytest.mark.parametrize(
    "task_class",
    [ImageClassificationTrainer, ImageSegmentationTrainer, ImageDenoisingTrainer, PretrainTrainer,
     TimeSeriesForecastingTrainer],
)
def test_no_existing_task_overrides_the_optimizer_options(task_class):
    assert getattr(task_class, "optimizer_options") is getattr(TaskTrainer, "optimizer_options")


def test_the_optimizer_options_default_to_nothing(world, tmp_path):
    task = ImageClassificationTrainer(config_for(tmp_path / "run"))
    assert task.optimizer_options(task.config, None) == {}


def test_a_default_task_hands_the_optimizer_builder_the_config_it_did_before_the_seam(default_run):
    task, _, _ = default_run
    assert len(task.builder_configs) == 1
    assert list(task.builder_configs[0].items()) == BASE_DEFAULT_RUN_CONFIG


@pytest.mark.parametrize("arm", list(OPTIM_ARMS))
@pytest.mark.parametrize("options", list(NO_OPTIONS))
def test_no_options_leave_the_builder_config_as_it_was_before_the_seam(arm, options):
    args, expected = optim_arm(arm)
    keywords = {} if NO_OPTIONS[options] == "absent" else {"options": NO_OPTIONS[options]}
    with spying_on_the_optimizer_builder() as seen:
        optimizer, _ = build_optimizer_and_schedule(args, 351, 10, **keywords)
    assert [list(config.items()) for config in seen] == [expected]
    assert type(optimizer).__name__ != "LRMultiplierAdamW"


class Multiplied(Scripted):
    """A task whose ``optimizer_options`` gives the model's last trainable variable a 10x rate."""

    def optimizer_options(self, config, model):
        self.model_was_built = model.built
        self.table = {model.trainable_variables[-1].path: 10.0}
        self.other = model.trainable_variables[0].path
        return {"lr_multipliers": dict(self.table)}


@pytest.fixture(scope="module")
def multiplied(world, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("multiplied") / "run"
    task = Multiplied(config_for(run_dir, epochs=1))
    with spying_on_the_optimizer_builder() as seen:
        task.run()
    task.builder_configs = seen
    return task


def test_a_task_rate_table_reaches_an_lr_multiplier_adamw(multiplied):
    from dl_techniques.optimization import LRMultiplierAdamW

    assert multiplied.model_was_built is True
    assert len(multiplied.compile_calls) == 1
    optimizer = multiplied.compile_calls[0][1]["optimizer"]
    assert isinstance(optimizer, LRMultiplierAdamW)
    assert optimizer.lr_multipliers == multiplied.table
    (path,) = multiplied.table
    assert optimizer.multiplier_for(path) == 10.0
    # Only the named variable moves: the first trainable variable keeps the base rate.
    assert multiplied.other != path and optimizer.multiplier_for(multiplied.other) == 1.0
    # The table is the one key the options add: everything else is what the builder got before.
    (config,) = multiplied.builder_configs
    assert config["lr_multipliers"] == multiplied.table
    assert list(config)[-1] == "lr_multipliers"


@pytest.mark.parametrize(
    "options, named",
    [({"layer_rates": {"a": 2.0}}, "layer_rates"),
     ({"lr_multipliers": {"a": 2.0}, "momentum": 0.9}, "momentum"),
     ({"lr_multipliers": [("a", 2.0)]}, "lr_multipliers"),
     ([("lr_multipliers", {"a": 2.0})], "mapping")],
    ids=["an unknown key", "a known key beside an unknown one", "a table that is not a mapping",
         "options that are not a mapping"],
)
def test_an_option_outside_the_closed_set_is_refused_by_name(options, named):
    args, _ = optim_arm("adamw_global_clip")
    with spying_on_the_optimizer_builder() as seen:
        with pytest.raises(ConfigError, match=named):
            build_optimizer_and_schedule(args, 351, 10, options=options)
    assert seen == [], "refused before the builder is called"


#: Table entries ``LRMultiplierAdamW`` refuses; each must be a ``ConfigError`` naming the entry's key and
#: value, raised before the builder is called, not the optimizer's own ``ValueError``.
BAD_ENTRIES = {
    "a key that is not a string": (3, 2.0),
    "an empty key": ("", 2.0),
    "a key that is only a slash": ("/", 2.0),
    "a value that is a string": ("m/dec", "10"),
    "a value that is a bool": ("m/dec", True),
    "a value that is None": ("m/dec", None),
    "a zero multiplier": ("m/dec", 0.0),
    "a negative multiplier": ("m/dec", -1.0),
    "an infinite multiplier": ("m/dec", float("inf")),
    "a nan multiplier": ("m/dec", float("nan")),
}


@pytest.mark.parametrize("entry", list(BAD_ENTRIES))
def test_a_bad_rate_table_entry_is_refused_naming_its_key_and_value(entry):
    key, value = BAD_ENTRIES[entry]
    args, _ = optim_arm("adamw_global_clip")
    with spying_on_the_optimizer_builder() as seen:
        with pytest.raises(ConfigError) as raised:
            build_optimizer_and_schedule(args, 351, 10, options={"lr_multipliers": {"m/enc": 0.5, key: value}})
    text = str(raised.value)
    assert re.search(rf"lr_multipliers.*{re.escape(repr(key))}", text), text
    assert repr(value) in text, text
    assert "'m/enc'" not in text, "a good entry beside the bad one is not named"
    assert seen == [], "refused before the builder is called"


def test_a_task_handing_a_bad_rate_table_entry_stops_the_run_with_a_config_error(world, tmp_path):
    class BadEntry(Scripted):
        def optimizer_options(self, config, model):
            return {"lr_multipliers": {model.trainable_variables[-1].path: 0.0}}

    run_dir = tmp_path / "run"
    with pytest.raises(ConfigError, match=r"lr_multipliers.*0\.0"):
        BadEntry(config_for(run_dir, epochs=1)).run()
    assert not (run_dir / "training_log.csv").exists(), "refused before fit"


@pytest.mark.parametrize("arm", ["sgd_momentum", "adam_constant"])
def test_a_rate_table_with_an_optimizer_other_than_adamw_is_refused_by_name(arm):
    args, _ = optim_arm(arm)
    with spying_on_the_optimizer_builder() as seen:
        with pytest.raises(ConfigError, match=rf"lr_multipliers.*{args.optimizer}"):
            build_optimizer_and_schedule(args, 351, 10, options={"lr_multipliers": {"a": 10.0}})
    assert seen == []


def test_a_task_naming_an_unknown_optimizer_option_stops_the_run_before_fit(world, tmp_path):
    class Unknown(Scripted):
        def optimizer_options(self, config, model):
            return {"layer_rates": {"a": 2.0}}

    run_dir = tmp_path / "run"
    with pytest.raises(ConfigError, match="layer_rates"):
        Unknown(config_for(run_dir, epochs=1)).run()
    assert not (run_dir / "training_log.csv").exists(), "refused before fit"
