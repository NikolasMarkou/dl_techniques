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

Plan 9c7b9096 (D-003) adds ``TaskTrainer.optimizer_options(config, model)`` (default ``{}``), called
once with the built model and handed to ``build_optimizer_and_schedule(..., options=)``, which merges a
closed key set (``lr_multipliers`` only) into the builder's config. The default arm spies on
``optimizer_builder`` and compares the config it receives with literals read at the commit before the
seam (f6f15bedf); the positive arm gets an ``LRMultiplierAdamW`` holding the task's table; another key,
or a table with an optimizer other than adamw, is refused by name.

Plan 7806a290 step 5 (D-008) adds the two seams the language task's export needs, both defaulting to what every earlier task did:
``ExportRule.sample`` (default ``None``: the tflite check feeds the random float batch it always fed) and the rank-keyed sentence of
``exporting._input_spec`` (ranks 3 and 2 keep their words, rank 1 names ``(seq_len,)``). Each has its own test over EVERY task's row.

Plan 7806a290 step 7.2 (D-050) adds ``TaskTrainer.resume_stream_exact`` (default ``False``), which says that a resume from a copy
taken inside an epoch reads the batches the unkilled run reads next. Four shared sentences choose their wording from it (the help
of ``resume_interrupted`` and of ``checkpoint_args.backup_every_steps``, the ``Resume:`` log line, the summary note); the last section
compares each earlier task's four sentences with the text they had at 585af294b, character for character.

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
import importlib
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


def test_without_init_nothing_is_pinned_and_no_init_block_is_held():
    """plan-2026-11-18T072904-0fd54b91 D-010, D-014: ``AutoContext.pinned`` defaults to empty, so ``merge_model_args``
    lays the defaults, the computed values and the caller's keys as it did; a run holds no ``init`` block."""
    from train.api.args import AutoContext

    field = {f.name: f for f in dataclasses.fields(AutoContext)}["pinned"]
    assert field.default_factory() == {} and field.default_factory() is not field.default_factory()
    assert {f.name: f for f in dataclasses.fields(Run)}["init_block"].default is None
    context = AutoContext(family="vit", variant="vit_pico", dataset="cifar10", num_classes=10, image_size=32,
                          defaults={"model_args": {"depth": 1, "patch_size": 2}})
    merged = recipes.merge_model_args(context, OptimArgs(optimizer="adamw"), {"depth": 3}, {"patch_size": 4, "x": 5})
    assert merged == {"depth": 3, "patch_size": 4, "x": 5}


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
# The optimizer seam (plan 9c7b9096, D-003)
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


def test_the_depth_task_is_the_one_task_that_overrides_the_optimizer_options():
    """Plan 9c7b9096 (D-003): the seam's one user is the depth task (the DPT head at 10x). Read over every
    built-in task, so a seventh task that overrides it is a new row to review, not a silent second user."""
    from train.api.tasks import BUILTIN_TASKS
    from train.api.tasks.depth_estimation import DepthEstimationTrainer

    users = set()
    for spec in BUILTIN_TASKS:
        module = importlib.import_module(spec.function.split(":")[0])
        trainers = [value for value in vars(module).values()
                    if isinstance(value, type) and issubclass(value, TaskTrainer) and value.__module__ == module.__name__]
        assert len(trainers) == 1, (spec.name, trainers)
        if trainers[0].optimizer_options is not TaskTrainer.optimizer_options:
            users.add(trainers[0])
    assert users == {DepthEstimationTrainer}


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


# ---------------------------------------------------------------------
# The two export seams of the language task (plan 7806a290 step 5, D-008)
# ---------------------------------------------------------------------

EXPORT_SHAPE = (4, 4, 3)


def tflite_blob_of_a_dense_model(bias_shift: float = 0.0) -> tuple:
    """``(model, blob, spec)`` of a dense model over ``EXPORT_SHAPE`` and its TFLite file (built-in operators)."""
    import tensorflow as tf

    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.layers.Input(EXPORT_SHAPE), keras.layers.Flatten(), keras.layers.Dense(5)])
    return model, tf.lite.TFLiteConverter.from_keras_model(model).convert(), tf.TensorSpec((None, *EXPORT_SHAPE), tf.float32)


def what_the_check_runs_the_file_on(monkeypatch, *arguments, **keywords):
    """The batches ``exporting._tflite_logits`` is given while ``_check_tflite_against_keras(*arguments, **keywords)`` runs."""
    from train.api import exporting

    real, seen = exporting._tflite_logits, []

    def spy(blob, batch):
        seen.append(batch.copy())
        return real(blob, batch)

    monkeypatch.setattr(exporting, "_tflite_logits", spy)
    exporting._check_tflite_against_keras(*arguments, **keywords)
    return seen


def test_the_tflite_check_feeds_the_random_float_batch_it_always_fed_when_no_sample_is_given(monkeypatch, tmp_path):
    import numpy as np

    model, blob, spec = tflite_blob_of_a_dense_model()
    seen = what_the_check_runs_the_file_on(monkeypatch, blob, model, spec, tmp_path / "a.tflite")
    # written out, not read from the module: the literal of the check before the seam existed (a free batch is two rows)
    assert len(seen) == 1 and np.array_equal(seen[0], np.random.default_rng(0).random((2, *EXPORT_SHAPE)).astype(np.float32))
    import tensorflow as tf

    fixed = tf.TensorSpec((3, *EXPORT_SHAPE), tf.float32)
    seen = what_the_check_runs_the_file_on(monkeypatch, blob, model, fixed, tmp_path / "a.tflite", None, None)
    assert np.array_equal(seen[0], np.random.default_rng(0).random((3, *EXPORT_SHAPE)).astype(np.float32))


def test_a_sample_replaces_the_check_batch_and_is_asked_for_the_shape_of_the_batch(monkeypatch, tmp_path):
    import numpy as np

    model, blob, spec = tflite_blob_of_a_dense_model()
    asked, marker = [], np.full((2, *EXPORT_SHAPE), 0.25, np.float32)

    def sample(shape):
        asked.append(shape)
        return marker

    seen = what_the_check_runs_the_file_on(monkeypatch, blob, model, spec, tmp_path / "a.tflite", None, sample)
    assert asked == [(2, *EXPORT_SHAPE)] and len(seen) == 1 and np.array_equal(seen[0], marker)


def test_an_export_rule_names_no_sample_unless_it_is_a_text_rule():
    from train.api import predictor

    assert predictor.ExportRule._field_defaults["sample"] is None, "the default is a no-op: today's random float batch"
    with_sample = sorted(task for task, rule in predictor.EXPORT_RULES.items() if rule.sample is not None)
    assert with_sample == ["language_model_pretraining", "text_classification"], "every earlier task's export check still feeds random floats"
    assert sorted(predictor.EXPORT_RULES) == sorted(predictor.TASK_TABLES["RECORD_KEYS"])


def test_export_hands_the_rules_sample_to_the_tflite_writer_bound_to_the_record_and_hands_nothing_for_a_rule_without_one(
        monkeypatch, tmp_path):
    import numpy as np

    from train.api import exporting, predictor
    from train.api.text_models import create_gpt2_lm
    from train.api.text_tokenizer import tokenizer_spec

    task = "language_model_pretraining"
    model = create_gpt2_lm(variant="tiny", input_shape=(16,), vocab_size=101, embed_dim=32, depth=1, num_heads=2)
    record = predictor.build_language_model_record(
        model="gpt2_lm/tiny", family="gpt2_lm", variant="tiny", seq_len=16, tokenizer=tokenizer_spec("ascii95").record())
    handed = []
    monkeypatch.setattr(exporting, "_export_tflite", lambda *args: handed.append(args))
    exporting.export(model=model, out=tmp_path / "a.tflite", format="tflite", input_shape=(16,), record=record)
    sample = handed[0][4]
    assert callable(sample) and np.array_equal(sample((2, 16)), predictor.EXPORT_RULES[task].sample(record, (2, 16)))
    monkeypatch.setitem(predictor.EXPORT_RULES, task, predictor.EXPORT_RULES[task]._replace(sample=None))
    handed.clear()
    exporting.export(model=model, out=tmp_path / "b.tflite", format="tflite", input_shape=(16,), record=record)
    assert handed[0][4] is None, "a rule that names no sample hands the writer none"


#: ``input_shape`` is asked of a rule in the shape its own ``shape_of`` returns; a record that serves every row's ``shape_of``.
SHAPE_RECORD = {"input_shape": [8, 8, 3], "input_size": [8, 8], "channels": 3, "lookback": 8, "seq_len": 8}
#: The sentence of ``_input_spec`` before the rank seam existed: rank 3 named the image shape, any other rank the series shape.
BEFORE_THE_SEAM = (
    "Invalid value for 'input_shape': expected {wanted} of positive ints, got {given!r} (a height or width may be None only for a "
    "model that itself takes any size)"
)


def test_every_earlier_tasks_input_spec_sentence_is_the_one_it_always_was_and_the_text_rank_names_its_own():
    from train.api import exporting, predictor

    ranks = {task: len(rule.shape_of(SHAPE_RECORD)) for task, rule in predictor.EXPORT_RULES.items()}
    assert ranks == {
        "image_classification": 3, "image_segmentation": 3, "image_denoising": 3, "time_series_forecasting": 2,
        "depth_estimation": 3, "pretrain": 3, "language_model_pretraining": 1, "text_classification": 1,
    }
    for task, rank in ranks.items():
        bad = (0,) * rank
        with pytest.raises(ConfigError) as caught:
            exporting._input_spec(None, bad, rank=rank)
        if task not in ("language_model_pretraining", "text_classification"):
            wanted = "(height, width, channels)" if rank == 3 else "(lookback, channels)"
            assert str(caught.value) == BEFORE_THE_SEAM.format(wanted=wanted, given=bad), task
        else:
            assert str(caught.value) == BEFORE_THE_SEAM.format(wanted="(seq_len,)", given=bad)
    spec = exporting._input_spec(None, (8,), rank=1)
    assert tuple(spec.shape) == (None, 8) and spec.dtype.name == "float32"
    with pytest.raises(ConfigError, match=r"expected \(seq_len,\) of positive ints"):
        exporting._input_spec(None, (8, 8), rank=1)


# ---------------------------------------------------------------------
# Plan 7806a290 step 7.2 (D-050): TaskTrainer.resume_stream_exact, and the four sentences that read it
# ---------------------------------------------------------------------

EARLIER_TASKS = ("image_classification", "image_segmentation", "image_denoising", "pretrain", "time_series_forecasting",
                 "depth_estimation")
#: The words the four sentences of a task whose resumed stream is exact share.
EXACT_PHRASE = "the batches the unkilled run reads next"
#: Each sentence's clause about the batches of the killed epoch the copy had not taken, as it was at 585af294b (``OLD``) and as a
#: task that sets the flag says it (``NEW``): the help of ``resume_interrupted`` (1), of ``backup_every_steps`` (2), the log line
#: (3) and the summary note (4). The rest of each sentence is the shared text and is the same on both sides.
OLD = {
    1: "from a fresh pass of the data, so some examples of that epoch repeat and some are missed",
    2: "from a fresh pass of the data (some examples of that epoch are seen twice and some not at all)",
    3: "from a fresh pass of the data (some examples of it repeat and some are missed)",
    4: "from a fresh pass of the data, so some of its examples were seen twice and some not at all",
}
NEW = {
    1: "from the stream at the step that backup holds (the batches the unkilled run reads next), so no example of that epoch "
       "repeats and none is missed",
    2: "from the stream at the step that backup holds (the batches the unkilled run reads next: no example of that epoch is seen "
       "twice and none is missed)",
    3: "from the stream at the step the backup holds (the batches the unkilled run reads next: none of its examples repeats or "
       "is missed)",
    4: "from the stream at the step the backup holds (the batches the unkilled run reads next), so none of its examples was seen "
       "twice or missed",
}
#: The rendered help line of ``checkpoint_args.backup_every_steps`` at 585af294b: one text for every task (the group is shared).
BEFORE_BACKUP_EVERY_STEPS_LINE = (
    "  checkpoint_args.backup_every_steps  (int or null)  also back the run up every this many batches INSIDE an epoch (an int >= 2 "
    "and below the steps per epoch), so a killed epoch loses at most that many batches of weights; null backs up at each epoch end "
    "only. A resume from such a backup continues the killed epoch on the weights of the last backup and runs only the batches of it "
    "that backup had not taken, from a fresh pass of the data (some examples of that epoch are seen twice and some not at all), then "
    "the remaining epochs in full; the optimizer's step count and the learning rate of every step are the unkilled run's. That "
    "epoch's row holds the train metrics of its remaining batches only. Each backup writes the weights and optimizer state twice "
    "(the previous copy is kept): src/train/README.md has the measured cost per save [default: None]"
)
#: sha256 of the rendered ``resume_interrupted`` line of each task at 585af294b (it differs per task: its epoch cost and its order).
BEFORE_RESUME_LINE_SHA256 = {
    "image_classification": "77c3adfe69c61edd4df41156d688b87594d8be79828bbcec50fd83434ecfb388",
    "image_segmentation": "d3fc00fad9037082137a21a36885228fb36f8800bfe246490ae6b7bf7e6677df",
    "image_denoising": "d3fc00fad9037082137a21a36885228fb36f8800bfe246490ae6b7bf7e6677df",
    "pretrain": "ab0f9158b8e2bd2a8ed26e4d7aaee819e38bb9f6eaf1f72bbbf25f7326d6a8d1",
    "time_series_forecasting": "65eec1749a11cbe55e2cbf1bcf85e10bda0494fdd19ec5dae526c4de5f2446e6",
    "depth_estimation": "8b1b6e8869121b170bf2f442f433510f4e687c07b5cf8ae15e12f3a624d2fa23",
    "language_model_pretraining": "fbb39c201c4ede0c53cdfccbfabad75f6eff279689267be1b5a05a7762e4ac9c",
}
#: The ``Resume:`` line of a resume from a copy taken after batch 4 of 10 of epoch 1 (``resume_line`` below), and the summary note
#: of the killed epoch (``killed_epoch_note``), both at 585af294b.
BEFORE_RESUME_LOG_LINE = (
    "Resume: the backup was taken after batch 4 of 10 of epoch 1; that epoch runs its remaining 6 batches from a fresh pass of the "
    "data (some examples of it repeat and some are missed) and ends; the optimizer's step count stays at 14"
)
BEFORE_KILLED_EPOCH_NOTE = (
    "epoch 1 was killed after 4 of its 10 batches; the resume ran its remaining 6 from a fresh pass of the data, so some of its "
    "examples were seen twice and some not at all, and the epoch's row (index 1) holds the train metrics (loss, accuracy, ...), "
    "`lr` and `epoch_times` entry of those 6 batches only and the validation metrics of the whole validation set"
)


def task_row(name):
    from train.api.tasks import BUILTIN_TASKS

    return next(row for row in BUILTIN_TASKS if row.name == name)


def trainer_class_of(row):
    """The one ``TaskTrainer`` the module of the task defines."""
    module = importlib.import_module(row.function.split(":")[0])
    found = [value for value in vars(module).values()
             if isinstance(value, type) and issubclass(value, TaskTrainer) and value.__module__ == module.__name__]
    assert len(found) == 1, (row.name, found)
    return found[0]


def rendered(name, key):
    """The line ``python -m train <task> help`` prints for ``key`` (the real renderer)."""
    from train.api import cli

    lines = [line for line in cli._field_help_lines(task_row(name).load_config_class()) if line.startswith(f"  {key}  (")]
    assert len(lines) == 1, (name, key, lines)
    return lines[0]


def sha256_of(text):
    return hashlib.sha256(text.encode()).hexdigest()


#: The tasks whose ``resume_stream`` continues the SAME stream at the settled step: the two text tasks.
EXACT_TASKS = ("language_model_pretraining", "text_classification")


def test_the_flag_is_false_by_default_and_the_two_text_tasks_are_the_tasks_that_set_it():
    """A task whose ``resume_stream`` continues the SAME stream at the settled step is the one that sets it; the seam's second
    user (the text classifier, plan 7806a290 step 10) is a row reviewed here, not a silent change."""
    from train.api.tasks import BUILTIN_TASKS

    assert TaskTrainer.resume_stream_exact is False
    assert {row.name: trainer_class_of(row).resume_stream_exact for row in BUILTIN_TASKS} == {
        **{name: False for name in EARLIER_TASKS}, **{name: True for name in EXACT_TASKS}}


@pytest.mark.parametrize("name", EARLIER_TASKS)
def test_the_two_helps_of_every_earlier_task_are_what_they_always_were(name):
    assert sha256_of(rendered(name, "resume_interrupted")) == BEFORE_RESUME_LINE_SHA256[name], rendered(name, "resume_interrupted")
    assert rendered(name, "checkpoint_args.backup_every_steps") == BEFORE_BACKUP_EVERY_STEPS_LINE


def test_a_task_that_sets_the_flag_changes_the_clause_of_both_helps_and_nothing_else():
    lm = "language_model_pretraining"
    resume, backup = rendered(lm, "resume_interrupted"), rendered(lm, "checkpoint_args.backup_every_steps")
    assert NEW[1] in resume and NEW[2] in backup, (resume, backup)
    assert "fresh pass" not in resume + backup
    assert sha256_of(resume.replace(NEW[1], OLD[1])) == BEFORE_RESUME_LINE_SHA256[lm]
    assert backup.replace(NEW[2], OLD[2]) == BEFORE_BACKUP_EVERY_STEPS_LINE


@pytest.mark.parametrize("name", [*EARLIER_TASKS, *EXACT_TASKS])
def test_the_helps_of_every_task_follow_its_flag(name):
    """The helps are text of the config class and the flag a class variable of the trainer, declared apart (as ``resume_order``
    and the help's ``order`` are): this test holds them equal for every task, so a task cannot set one and forget the other."""
    exact = trainer_class_of(task_row(name)).resume_stream_exact
    for key in ("resume_interrupted", "checkpoint_args.backup_every_steps"):
        if name == "depth_estimation" and key == "resume_interrupted":
            continue  # a task that refuses the key says so instead of what a resume from a copy reads
        text = rendered(name, key)
        assert (EXACT_PHRASE in text, "from a fresh pass of the data" in text) == (exact, not exact), (name, key, text)


def resume_line(monkeypatch, tmp_path, **keywords):
    """The ``Resume:`` line the real callback logs when a fit resumes from a copy taken after batch 4 of 10 of epoch 1."""
    from train.common import resume_state

    from .test_backup_every_steps import KillAt, S, backed_up, fit, tiny

    tmp_path.mkdir(parents=True, exist_ok=True)
    directory = tmp_path / "b"
    assert fit(tiny(), [KillAt(1, 6), *backed_up(directory)[1]]) and S == 10
    lines = []
    monkeypatch.setattr(resume_state.logger, "info", lines.append)
    second = resume_state.StepBackupAndRestore(directory, 4, S, **keywords)
    assert not fit(tiny(), [second.finish, second])
    return next(line for line in lines if line.startswith("Resume:"))


def test_the_resume_line_is_what_it_always_was_and_the_flag_changes_its_clause_only(monkeypatch, tmp_path):
    assert resume_line(monkeypatch, tmp_path / "default") == BEFORE_RESUME_LOG_LINE
    line = resume_line(monkeypatch, tmp_path / "exact", exact_stream=True)
    assert line == BEFORE_RESUME_LOG_LINE.replace(OLD[3], NEW[3]) and EXACT_PHRASE in line


def killed_epoch_note(trainer_class, monkeypatch):
    """The second note of the real ``TaskTrainer._notes`` (the task's own lines left out) after 4 of 10 batches of epoch 1."""
    from types import SimpleNamespace

    from train.api import resume_decisions

    monkeypatch.setattr(trainer_class, "task_notes", lambda *args: [])
    monkeypatch.setattr(resume_decisions, "training_complete_reason", lambda resume, config: None)
    config = SimpleNamespace(debug_args=SimpleNamespace(smoke=False))
    return object.__new__(trainer_class)._notes(
        config, {}, 1.0, False, SimpleNamespace(steps_per_epoch=10), SimpleNamespace(start=1, batch=3))[1]


@pytest.mark.parametrize("name", EARLIER_TASKS)
def test_the_killed_epoch_note_of_every_earlier_task_is_what_it_always_was(name, monkeypatch):
    assert killed_epoch_note(trainer_class_of(task_row(name)), monkeypatch) == BEFORE_KILLED_EPOCH_NOTE


def test_the_killed_epoch_note_of_a_task_that_sets_the_flag_changes_its_clause_only(monkeypatch):
    note = killed_epoch_note(trainer_class_of(task_row("language_model_pretraining")), monkeypatch)
    assert note == BEFORE_KILLED_EPOCH_NOTE.replace(OLD[4], NEW[4]) and EXACT_PHRASE in note


@pytest.mark.parametrize("exact", [False, True])
def test_the_callback_list_hands_the_trainers_flag_to_the_backup_callback(world, tmp_path, exact):
    """``build_callbacks`` reads the flag off the trainer: a real run with ``backup_every_steps`` set holds the callback."""
    from train.common.resume_state import StepBackupAndRestore

    task_class = type("Flagged", (Scripted,), {"resume_stream_exact": exact})
    task = task_class(config_for(tmp_path / "run", epochs=1, **{"checkpoint_args.backup_every_steps": 2}))
    task.run()
    backups = [callback for callback in task.seen_callbacks if isinstance(callback, StepBackupAndRestore)]
    assert [backup.exact_stream for backup in backups] == [exact]
