"""The ``depth_estimation`` task: its argument groups, its one family's adapter, tables and recipe.

First part (the groups and the family, built but not registered): the data, transform and metric groups parse
strictly and name every bad option; an ``image_size`` that is not a multiple of the DINOv2 patch 14 is refused; the
pre-claim facts refuse a missing root and a tree that deals fewer than 3 validation scenes; the adapter returns a
BUILT ``DepthAnything`` at ``(None, s, s, 3)`` with the four forced keys; the refusal tables equal the library's
signatures (``signature_census``); a missing official weights file is a ``ConfigError`` naming the path and the
environment variable, raised by the resolver before any model is built (and, in a fresh interpreter, before
TensorFlow has touched a device); the recipe numbers are the legacy trainer's, read from
``train.depth_anything.train_depth_anything``; the learning rate resolves to the from-scratch rate and the head
table to ``{}`` when ``model_args.pretrained`` is false.

Every model here is built with ``pretrained=False`` at 28 px (a 2 x 2 patch grid), except where a test says so.
"""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

import dl_techniques
import train
from train.api import depth_args, depth_models
from train.api.args import AutoContext, OptimArgs
from train.api.config import validate
from train.api.depth_args import (
    DepthData,
    DepthMetricArgs,
    DepthOptimArgs,
    DepthTransformArgs,
    depth_facts,
    load_depth_data_config,
    pretrained_start,
)
from train.api.errors import ConfigError, ConfigValidationError
from train.api.registry import ModelSpec

WT_SRC = str(Path(train.__file__).resolve().parents[1])
SIZE = 28
WEIGHTS_ENV = "DL_TECHNIQUES_DINOV2_WEIGHTS_DIR"


def write_scene_tree(root: Path, scenes: int, pairs: int = 2) -> Path:
    """A MegaDepth-shaped listing with empty files: discovery matches names, it reads no pixel."""
    for s in range(scenes):
        for kind, suffix in (("imgs", ".png"), ("depths", ".h5")):
            folder = root / f"s{s:02d}" / "dense0" / kind
            folder.mkdir(parents=True, exist_ok=True)
            for p in range(pairs):
                (folder / f"p{p}{suffix}").touch()
    return root


def context(model_args=None, variant="vit_s"):
    spec = depth_models.builtin_depth_models()[0]
    return AutoContext(
        family=depth_models.FAMILY, variant=variant, dataset=depth_args.MEGADEPTH, num_classes=None,
        image_size=None, defaults=spec.recipe(depth_args.MEGADEPTH, variant), epochs=30,
        model_args=dict(model_args or {}))


def resolved_optim(model_args=None, **given):
    return DepthOptimArgs(**given).resolve_auto(context(model_args))


@pytest.fixture(scope="module")
def tiny():
    """One from-scratch vit_s at 28 px through the adapter, shared by the tests that only read it."""
    return depth_models.create_depth_anything(
        variant="vit_s", input_shape=(SIZE, SIZE, 3), pretrained=False)


# ---------------------------------------------------------------------
# The groups
# ---------------------------------------------------------------------


def test_the_groups_parse_strictly_and_hold_the_measured_defaults():
    data = load_depth_data_config({})
    assert (data.format, data.dataset, data.n_select, data.n_test, data.layout_samples) == (
        "builtin", "megadepth", 128, 256, 512)
    assert (data.max_train_files, data.steps_per_epoch, data.workers, data.train_split) == (10_000, 100, 8, 0.9)
    assert os.path.isabs(data.root)
    with pytest.raises(ConfigValidationError, match="n_selct"):
        load_depth_data_config({"n_selct": 4})
    with pytest.raises(ConfigValidationError, match="augment"):
        validate(DepthTransformArgs, {"augment": "rotate"})
    assert validate(DepthTransformArgs, {}).augment == "flip"
    lines = DepthMetricArgs(watch_metric="val_abs_rel", watch_mode="max").problems("metric_args")
    assert len(lines) == 2 and all("val_loss" in line for line in lines)
    assert DepthMetricArgs().problems("metric_args") == []


def test_a_data_file_anchors_a_relative_root_to_the_file(tmp_path):
    (tmp_path / "data.yaml").write_text("root: tree\nn_select: 16\n")
    data = load_depth_data_config(str(tmp_path / "data.yaml"))
    assert data.root == str(tmp_path / "tree") and data.n_select == 16


def test_every_bad_data_option_is_named():
    bad = DepthData(dataset="nyu", n_select=0, n_test=-1, layout_samples=0, min_valid_ratio=0.0, steps_per_epoch=0,
                    workers=-1, max_train_files=0, max_samples=0, train_split=1.0)
    text = "\n".join(bad.problems("data"))
    for name in ("dataset", "n_select", "n_test", "layout_samples", "min_valid_ratio", "steps_per_epoch", "workers",
                 "max_train_files", "max_samples", "train_split"):
        assert f"'data.{name}'" in text, name
    assert DepthData().problems("data") == []


@pytest.mark.parametrize("size", [250, 13, 0])
def test_an_image_size_that_is_not_a_multiple_of_the_patch_is_refused(size):
    lines = DepthTransformArgs(image_size=size).problems("transform_args")
    assert len(lines) == 1 and "'transform_args.image_size'" in lines[0] and "14" in lines[0]


def test_the_image_size_resolves_from_the_recipe_and_a_multiple_of_14_is_kept():
    assert DepthTransformArgs().resolve_auto(context()).image_size == 252
    assert DepthTransformArgs(image_size=28).resolve_auto(context()).image_size == 28
    assert DepthTransformArgs(image_size=28).problems("transform_args") == []


def test_the_patch_is_the_models_own():
    from dl_techniques.models.vision.depth_anything import model

    assert depth_args.DINOV2_PATCH == model._DINOV2_PATCH_SIZE


# ---------------------------------------------------------------------
# The facts before the claim
# ---------------------------------------------------------------------


def test_the_facts_refuse_a_missing_root(tmp_path):
    with pytest.raises(ConfigError, match=r"'data\.root'.*does_not_exist"):
        depth_facts(DepthData(root=str(tmp_path / "does_not_exist")))


def test_the_facts_refuse_a_tree_that_deals_fewer_than_three_validation_scenes(tmp_path):
    root = write_scene_tree(tmp_path / "small", scenes=4)
    with pytest.raises(ConfigError, match=r"'data\.(root|train_split)'.*2 validation scenes.*at least 3"):
        depth_facts(DepthData(root=str(root), train_split=0.5))


def test_the_facts_of_a_tree_that_deals_three_validation_scenes(tmp_path):
    root = write_scene_tree(tmp_path / "ok", scenes=6)
    facts = depth_facts(DepthData(root=str(root), train_split=0.5))
    assert (facts.dataset, facts.num_classes, facts.image_size) == ("megadepth", None, None)


# ---------------------------------------------------------------------
# The adapter and its tables
# ---------------------------------------------------------------------


def test_the_adapter_returns_a_built_model_with_the_forced_keys(tiny):
    from dl_techniques.models.vision.depth_anything.model import DepthAnything

    assert isinstance(tiny, DepthAnything) and tiny.built
    assert tuple(tiny.image_shape) == (SIZE, SIZE, 3)
    config = tiny.get_config()
    for key, (value, _) in depth_models.FORCED.items():
        assert config[key] == value, key
    import numpy as np

    assert tuple(tiny(np.zeros((2, SIZE, SIZE, 3), "float32")).shape) == (2, SIZE, SIZE, 1)


def test_the_tables_are_held_to_the_library_signatures():
    assert depth_models.signature_census() == {"unclassified": [], "stale": []}
    assert depth_models.ALLOWED == ("pretrained",)
    assert {key: value for key, (value, _) in depth_models.FORCED.items()} == {
        "use_feature_alignment": False, "include_top": True, "input_value_range": None, "encoder_kind": "real"}
    grown = {**depth_models.factory_signatures(), "stand_in:new": ["a_new_knob"]}
    assert depth_models.signature_census(grown) == {"unclassified": ["a_new_knob"], "stale": []}


@pytest.mark.parametrize("key, value, why", [
    ("encoder_kind", "placeholder", "sets it to 'real'"),
    ("use_feature_alignment", True, "sets it to False"),
    ("enable_semi_supervised", True, "semi-supervised"),
    ("cutmix_prob", 0.1, "semi-supervised"),
    ("image_shape", [28, 28, 3], "transform_args.image_size"),
    ("dpt_features", 32, "not offered"),
    ("pretrained", "yes", "true or false"),
    ("no_such_key", 1, "no_such_key"),
])
def test_a_key_the_family_does_not_take_is_refused_by_name(key, value, why):
    lines = depth_models.problems_of({key: value})
    assert len(lines) == 1 and f"'model_args.{key}'" in lines[0] and why in lines[0], lines


def test_a_forced_key_at_its_value_and_the_allowed_key_are_accepted():
    assert depth_models.problems_of({"encoder_kind": "real", "use_feature_alignment": False, "pretrained": False}) == []


def test_the_adapter_refuses_classes_a_bad_shape_and_a_refused_key():
    for kwargs, match in (
        ({"num_classes": 3}, "no classes"),
        ({"input_shape": (28, 42, 3)}, "square"),
        ({"input_shape": (30, 30, 3)}, "14"),
        ({"input_shape": (28, 28, 1)}, "3 channels"),
        ({"encoder_kind": "placeholder"}, "encoder_kind"),
    ):
        call = {"variant": "vit_s", "input_shape": (SIZE, SIZE, 3), "pretrained": False, **kwargs}
        with pytest.raises(ValueError, match=match):
            depth_models.create_depth_anything(**call)


# ---------------------------------------------------------------------
# The weights before the claim
# ---------------------------------------------------------------------


def adamw():
    return resolved_optim()


def test_a_missing_weights_file_is_a_config_error_naming_the_path_and_the_variable(tmp_path, monkeypatch):
    monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path))
    expected = tmp_path / "dinov2-small" / "model.safetensors"
    with pytest.raises(ConfigError) as caught:
        depth_models.resolve_depth_args(context(), adamw(), {})
    text = str(caught.value)
    assert str(expected) in text and WEIGHTS_ENV in text and "pretrained=false" in text


def test_a_from_scratch_request_needs_no_weights_file(tmp_path, monkeypatch):
    monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path))
    kwargs = depth_models.resolve_depth_args(context({"pretrained": False}), adamw(), {"pretrained": False})
    assert kwargs == {"pretrained": False}


def test_the_weights_refusal_comes_before_any_device_or_model(tmp_path):
    """A fresh interpreter: the refusal imports the library's path rule and nothing touches a device or builds a model."""
    code = (
        "import sys, dl_techniques, train\n"
        "print('origin', dl_techniques.__file__, train.__file__)\n"
        "from train.api import depth_models\n"
        "from train.api.errors import ConfigError\n"
        "from test_depth_task import context, adamw\n"
        "try:\n"
        "    depth_models.resolve_depth_args(context(), adamw(), {})\n"
        "except ConfigError as exc:\n"
        "    print('refused', 'dinov2-small' in str(exc))\n"
        "from tensorflow.python.eager import context as eager\n"
        "print('device context initialized', eager.context()._initialized)\n"
        "print('model module', 'dl_techniques.models.vision.depth_anything.model' in sys.modules)\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([WT_SRC, str(Path(__file__).parent)]),
           "CUDA_VISIBLE_DEVICES": "", WEIGHTS_ENV: str(tmp_path)}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert f"origin {WT_SRC}" in out.stdout
    assert "refused True" in out.stdout
    assert "device context initialized False" in out.stdout


# ---------------------------------------------------------------------
# The recipe
# ---------------------------------------------------------------------


def test_the_recipe_numbers_are_the_legacy_trainers():
    from train.depth_anything import train_depth_anything as legacy

    spec = depth_models.builtin_depth_models()[0]
    recipe = spec.recipe(depth_args.MEGADEPTH, "vit_s")
    optim = recipe["optim_args"]
    assert optim["optimizer"] == legacy.DepthAnythingTrainingConfig().optimizer_type == "adamw"
    assert optim["lr"] == legacy.PRETRAINED_LEARNING_RATE == 5e-6
    assert depth_args.FROM_SCRATCH_LR == legacy.FROM_SCRATCH_LEARNING_RATE == 2e-4
    assert optim["weight_decay"] == legacy.DepthAnythingTrainingConfig().weight_decay == 1e-5
    assert optim["global_clipnorm"] == legacy.DepthAnythingTrainingConfig().gradient_clipping == 1.0
    assert optim["clipnorm"] is None
    assert (optim["schedule"], optim["warmup_epochs"], optim["min_lr_ratio"], optim["decay_over"]) == (
        "cosine_decay", 1, 0.01, "total")
    assert (recipe["epochs"], recipe["batch_size"], recipe["transform_args"]["image_size"]) == (30, 8, 252)
    assert recipe["model_args"] == {"pretrained": True}
    assert recipe["metric_args"] == {"watch_metric": "val_loss"}
    assert depth_models.JIT_COMPILE is False


@pytest.mark.parametrize("variant", ["vit_s", "vit_b", "vit_l"])
def test_every_variant_trains_at_batch_8_at_252_px(variant):
    recipe = depth_models.builtin_depth_models()[0].recipe(depth_args.MEGADEPTH, variant)
    assert (recipe["batch_size"], recipe["transform_args"]["image_size"]) == (8, 252)


def test_the_one_family_spec_is_built_for_the_task():
    (spec,) = depth_models.builtin_depth_models()
    assert isinstance(spec, ModelSpec)
    assert (spec.family, spec.tasks) == ("depth_anything", ("depth_estimation",))
    assert sorted(spec.load_variants()) == ["vit_b", "vit_l", "vit_s"]
    assert spec.load_factory() is depth_models.create_depth_anything
    assert spec.load_resolver() is depth_models.resolve_depth_args


def test_the_lr_follows_the_pretrained_start():
    assert resolved_optim().lr == 5e-6
    assert resolved_optim({"pretrained": True}).lr == 5e-6
    assert resolved_optim({"pretrained": False}).lr == 2e-4
    assert resolved_optim({"pretrained": False}, lr=1e-3).lr == 1e-3
    assert dataclasses.replace(resolved_optim({"pretrained": False}), lr=5e-6) == resolved_optim()


def test_the_head_table_is_10x_on_a_pretrained_start_and_empty_from_scratch(tiny):
    from train.depth_anything.train_depth_anything import DECODER_LR_MULTIPLIER, decoder_lr_multipliers

    assert pretrained_start({}) is True and pretrained_start({"pretrained": False}) is False
    assert decoder_lr_multipliers(tiny, pretrained_start({"pretrained": False})) == {}
    table = decoder_lr_multipliers(tiny, pretrained_start({}))
    head = {v.path for v in tiny.decoder.trainable_variables}
    assert set(table) == head and set(table.values()) == {DECODER_LR_MULTIPLIER} == {10.0}


def test_a_non_adamw_optimizer_on_a_pretrained_start_is_refused():
    adam = validate(OptimArgs, {**dataclasses.asdict(adamw()), "optimizer": "adam", "weight_decay": None})
    with pytest.raises(ConfigError, match=r"'optim_args\.optimizer'.*adamw"):
        depth_models.resolve_depth_args(context(), adam, {"pretrained": True})
    assert depth_models.resolve_depth_args(context(), adam, {"pretrained": False}) == {"pretrained": False}


def test_an_sgd_weight_decay_is_refused_as_for_every_task_with_no_model_regularizer():
    sgd = validate(OptimArgs, {**dataclasses.asdict(adamw()), "optimizer": "sgd", "momentum": 0.9,
                               "beta_1": None, "beta_2": None, "epsilon": None, "weight_decay": 1e-4})
    with pytest.raises(ConfigError, match="weight_decay"):
        depth_models.resolve_depth_args(context({"pretrained": False}), sgd, {"pretrained": False})


def test_the_evidence_imports_the_worktree():
    assert dl_techniques.__file__.startswith(WT_SRC) and train.__file__.startswith(WT_SRC)


# ---------------------------------------------------------------------
# Second part: the task, its hooks, its bar and its registration
# ---------------------------------------------------------------------
#
# The task module is imported inside the helpers, so each test of this part fails on its own until it exists.

import importlib
import inspect
import math
from types import SimpleNamespace

import numpy as np

from train.api import recipes, registry
from train.api.trainer import REQUIRED_HOOKS, TaskTrainer
from train.common import forecast_run
from train.common.pretrain_data import derive_stream_seed
from train.common.resume_state import FIT_STATE_NAME, update_fit_state

TASK = "depth_estimation"
TINY_TREE = dict(train_split=0.6, n_select=8, n_test=8, layout_samples=16, steps_per_epoch=3, workers=0,
                 max_train_files=None)


def task_module():
    return importlib.import_module("train.api.tasks.depth_estimation")


def bare_trainer():
    """A trainer with no config, for the pure hooks (``object.__new__``: the constructor needs a config)."""
    trainer = object.__new__(task_module().DepthEstimationTrainer)
    trainer.config = None
    trainer._baselines, trainer._epoch_zero, trainer._select_batches = None, None, None
    return trainer


@pytest.fixture(scope="module")
def world():
    """An empty registry holding every built-in family and task."""
    from train.api.tasks import register_builtin_tasks

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


@pytest.fixture(scope="module")
def megadepth_tree(tmp_path_factory):
    """The synthetic MegaDepth tree of the data library's tests: 9 scenes of 6 pairs at 40 px."""
    from tests.test_train.test_common_depth_data import write_megadepth_tree

    return write_megadepth_tree(tmp_path_factory.mktemp("megadepth"))


def prepared(out, root, **changes):
    """``(trainer, resolved config, context)`` resolved the way ``run`` resolves, at 28 px and batch 4."""
    module = task_module()
    asked = {"out": str(out), "model": "depth_anything/vit_s", "data": {"root": str(root), **TINY_TREE},
             "epochs": 2, "batch_size": 4, "transform_args": {"image_size": SIZE},
             "analysis_args": {"model_analysis": False, "figures": False}, **changes}
    config = validate(module.DepthEstimationConfig, asked)
    trainer = module.DepthEstimationTrainer(config)
    family, variant = registry.parse_model_name(config.model, task=trainer.task)
    resolved, ctx = trainer._resolve(trainer.describe_data(config), family, variant)
    return trainer, resolved, ctx


@pytest.fixture(scope="module")
def built(world, megadepth_tree, tmp_path_factory):
    """The data of a from-scratch request on the synthetic tree, built once by the task's own ``build_data``."""
    trainer, resolved, ctx = prepared(tmp_path_factory.mktemp("built") / "run", megadepth_tree,
                                      model_args={"pretrained": False})
    data = trainer.build_data(resolved, ctx)
    return SimpleNamespace(trainer=trainer, resolved=resolved, context=ctx, data=data)


#: The optional hooks the task overrides, and why each one is needed.
OPTIONAL_HOOKS = (
    "fit_kwargs",          # the train stream is infinite: an epoch is steps_per_epoch batches
    "resume_stream",       # a resumed fit starts a fresh iterator: another stream, not a replay
    "bar_is_met",          # a loss: strictly below, in float32, with a tie band (the default is >=)
    "compile_options",     # jit_compile False, explicitly
    "optimizer_options",   # the DPT head at 10x the encoder rate from a pretrained start
    "check_model_args",    # the weights file, the optimizer and the refused keys before the claim
    "figure_callbacks",    # the prediction grid of 8 fixed selection crops
    "run_analysis",        # data-free (the output is disparity, not class probabilities)
    "summary_fields",      # the scenes, the crops and the stream rule
    "used_arguments",      # the model keywords the run was built with
    "unjudged_note",       # why init_scale_warning is null
    "task_notes",          # the lines of the summary's notes
)


def test_the_class_names_its_task_and_its_metric():
    cls = task_module().DepthEstimationTrainer
    assert (cls.task, cls.success_metric) == (TASK, "val_loss")
    assert issubclass(cls, TaskTrainer)
    assert cls.resume_order, "a resume trains on another stream and the summary says so"


@pytest.mark.parametrize("hook", REQUIRED_HOOKS)
def test_every_required_hook_is_overridden(hook):
    cls = task_module().DepthEstimationTrainer
    assert getattr(cls, hook) is not getattr(TaskTrainer, hook), hook


@pytest.mark.parametrize("hook", OPTIONAL_HOOKS)
def test_every_optional_hook_the_task_relies_on_is_overridden(hook):
    cls = task_module().DepthEstimationTrainer
    assert getattr(cls, hook) is not getattr(TaskTrainer, hook), hook


def test_the_task_does_not_name_a_stage():
    own = {name for name in vars(task_module().DepthEstimationTrainer) if not name.startswith("__")}
    assert not own & {"run", "_resolve", "_callbacks", "_summary_head", "_notes", "_refuse_before_the_claim"}


def test_the_function_takes_exactly_the_fields_of_the_config():
    module = task_module()
    fields = [f.name for f in dataclasses.fields(module.DepthEstimationConfig)]
    params = list(inspect.signature(module.train_depth_estimation).parameters)
    assert params == fields
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in inspect.signature(
        module.train_depth_estimation).parameters.values())


def test_the_metric_names_are_the_compiled_metrics_names_and_build_nothing():
    from train.common.depth_run import loss_and_metrics

    names = bare_trainer().metric_names(None, None)
    assert names == [m.name for m in loss_and_metrics()[1]]
    assert all(isinstance(name, str) for name in names)


# -- the bar: strictly below, in float32, with the tie band of the selection batches ----------------


def test_a_value_equal_to_the_threshold_is_a_miss():
    trainer = bare_trainer()
    assert trainer.bar_is_met(None, 0.5, 0.5) is False
    assert trainer.bar_is_met(None, 0.4999, 0.5) is True
    assert trainer.bar_is_met(None, 0.5001, 0.5) is False


def test_a_float64_value_that_rounds_to_the_float32_threshold_is_a_miss():
    trainer = bare_trainer()
    threshold = float(np.float32(0.7))
    reached = threshold - 1e-10
    assert reached < threshold and np.float32(reached) == np.float32(threshold)
    assert trainer.bar_is_met(None, reached, threshold) is False


def test_the_bar_is_the_forecasting_rule_with_the_selection_batch_count():
    trainer = bare_trainer()
    threshold = np.float32(1.0)
    step = np.spacing(threshold)
    for batches in (None, 2, 16, 2048):
        trainer._select_batches = batches
        band = forecast_run.tie_band(batches or 1)
        for ulps in (0, 1, band, band + 1, band + 40):
            reached = float(threshold - np.float32(ulps) * step)
            assert trainer.bar_is_met(None, reached, float(threshold)) is forecast_run.beats_bar(
                reached, float(threshold), batches or 1), (batches, ulps)
        assert trainer.bar_is_met(None, float(threshold - np.float32(band) * step), float(threshold)) is False
        assert trainer.bar_is_met(None, float(threshold - np.float32(band + 1) * step), float(threshold)) is True


def test_the_success_value_is_the_minimum_from_the_selection_start(monkeypatch):
    trainer = bare_trainer()
    hist = {"val_loss": [0.4, 0.9, 0.6, 0.7], "loss": [0.1, 0.1, 0.1, 0.1]}
    assert trainer.success_value(hist) == 0.4
    monkeypatch.setattr(trainer, "selection_start_epoch", lambda config: 2)
    assert trainer.success_value(hist) == 0.6


def write_sidecar(run_dir: Path, loss: float) -> Path:
    from train.api.run_dir import CHECKPOINTS_DIR

    path = run_dir / CHECKPOINTS_DIR / FIT_STATE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    update_fit_state(path, initial={"metrics": {"loss": loss, "abs_rel": 0.3}, "ratio": loss, "warned": None})
    return run_dir


@pytest.mark.parametrize("constant, layout, epoch_zero, expected", [
    (1.0, 1.02, 1.55, 1.0),     # Step 1 measured: the constant binds
    (1.0, 0.93, 1.55, 0.93),    # the layout binds
    (1.0, 1.02, 0.81, 0.81),    # the untrained model binds
])
def test_the_threshold_is_the_minimum_of_the_three_terms(tmp_path, constant, layout, epoch_zero, expected):
    trainer = bare_trainer()
    trainer._baselines = {"constant": constant, "layout": layout}
    trainer.read_epoch_zero(write_sidecar(tmp_path, epoch_zero))
    assert trainer.success_threshold(None) == expected


def test_a_resume_reads_the_same_epoch_zero_value_from_the_sidecar(tmp_path):
    """A resumed process never runs ``initial_loss_check``: a fresh trainer reads the first process's value from the file."""
    run_dir = write_sidecar(tmp_path, 0.77)
    first, resumed = bare_trainer(), bare_trainer()
    for trainer in (first, resumed):
        trainer._baselines = {"constant": 1.0, "layout": 1.02}
        assert trainer.read_epoch_zero(run_dir) == 0.77
    assert first.success_threshold(None) == resumed.success_threshold(None) == 0.77


def test_a_threshold_asked_before_the_data_or_the_sidecar_is_refused_by_name(tmp_path):
    trainer = bare_trainer()
    trainer._epoch_zero = 0.8
    with pytest.raises(RuntimeError, match="baselines"):
        trainer.success_threshold(None)
    trainer._baselines, trainer._epoch_zero = {"constant": 1.0, "layout": 1.0}, None
    with pytest.raises(RuntimeError, match="epoch-zero"):
        trainer.success_threshold(None)
    with pytest.raises(ValueError, match="epoch-zero"):
        trainer.read_epoch_zero(tmp_path / "no_run")


# -- the hooks against the data the task builds -------------------------------------------------------


def test_build_data_hands_every_field_to_the_loader_and_measures_both_baselines(built):
    data, trainer = built.data, built.trainer
    assert (data.n_select, data.n_test, data.steps_per_epoch, data.batch_size, data.image_size) == (8, 8, 3, 4, SIZE)
    assert data.augment is True and data.workers == 0 and data.seed == built.resolved.seed
    assert trainer._select_batches == data.n_select // data.batch_size == 2
    assert set(trainer._baselines) == {"constant", "layout"}
    assert abs(trainer._baselines["constant"] - 1.0) < 1e-5
    assert math.isfinite(trainer._baselines["layout"])


def test_the_smoke_cap_and_the_file_cap_cut_the_train_list_and_never_the_fixed_sets(world, megadepth_tree, tmp_path):
    trainer, resolved, ctx = prepared(tmp_path / "run", megadepth_tree, model_args={"pretrained": False},
                                      data={"root": str(megadepth_tree), **TINY_TREE, "max_samples": 5})
    data = trainer.build_data(resolved, ctx)
    assert len(data.train_rgb) == 5 and (data.n_select, data.n_test) == (8, 8)
    trainer, resolved, ctx = prepared(tmp_path / "run2", megadepth_tree, model_args={"pretrained": False},
                                      data={"root": str(megadepth_tree), **TINY_TREE, "max_train_files": 7,
                                            "max_samples": 9})
    assert len(trainer.build_data(resolved, ctx).train_rgb) == 7


def test_the_fit_keywords_are_the_datas(built):
    data = built.data
    assert built.trainer.fit_kwargs(built.resolved, data) == {
        "steps_per_epoch": data.steps_per_epoch, "validation_steps": data.n_select // data.batch_size}


def test_a_resume_trains_on_the_stream_of_the_completed_steps_and_keeps_the_fixed_sets(built):
    data = built.data
    resumed = built.trainer.resume_stream(built.resolved, data, 7)
    assert resumed.stream_seed == derive_stream_seed(data.seed, 7) != data.stream_seed
    assert resumed.x_select is data.x_select and resumed.y_test is data.y_test and resumed.layout is data.layout
    assert resumed.fingerprint == data.fingerprint


def test_jit_compile_is_passed_explicitly_false(built):
    assert built.trainer.compile_options(built.resolved, registry.MODELS["depth_anything"]) == {"jit_compile": False}


def test_the_figure_is_the_prediction_grid_of_eight_selection_crops(built, tmp_path):
    from dl_techniques.callbacks.depth_visualization import DepthPredictionGridCallback
    from train.common.megadepth import standardize_imagenet

    (grid,) = built.trainer.figure_callbacks(built.resolved, built.data, tmp_path)
    assert isinstance(grid, DepthPredictionGridCallback) and grid.output_dir == tmp_path
    n = min(8, built.data.n_select)
    np.testing.assert_array_equal(np.asarray(grid.val_rgb), built.data.x_select[:n])
    np.testing.assert_array_equal(np.asarray(grid.val_depth), built.data.y_select[:n, ..., :1])
    np.testing.assert_array_equal(np.asarray(grid.val_mask), built.data.y_select[:n, ..., 1:])
    np.testing.assert_allclose(standardize_imagenet(np.asarray(grid.display_rgb)), built.data.x_select[:n], atol=1e-5)


@pytest.mark.parametrize("total, picked", [
    (12, [0, 1, 3, 4, 6, 7, 9, 10]),
    (32, [0, 4, 8, 12, 16, 20, 24, 28]),
    (128, [0, 16, 32, 48, 64, 80, 96, 112]),
])
def test_the_grid_crops_are_spread_evenly_over_the_selection_set(tmp_path, total, picked):
    """The selection crops are in scene order (about 13 per scene at the defaults), so the first 8 would all be one
    scene; crop ``i * total // 8`` is taken instead, deterministic and in order."""
    from train.common import depth_run

    x = np.broadcast_to(np.arange(total, dtype=np.float32)[:, None, None, None], (total, SIZE, SIZE, 3)).copy()
    y = np.concatenate([x[..., :1] + 0.5, np.ones_like(x[..., :1])], axis=-1)
    grid = depth_run.prediction_grid(SimpleNamespace(n_select=total, x_select=x, y_select=y), tmp_path)
    assert np.asarray(grid.val_rgb)[:, 0, 0, 0].tolist() == picked
    assert np.asarray(grid.val_depth)[:, 0, 0, 0].tolist() == [i + 0.5 for i in picked]
    assert np.asarray(grid.val_mask).min() == 1.0


def test_the_data_holds_no_classes(built):
    from train.api.data_args import DataFacts

    trainer = built.trainer
    trainer.input_check(DataFacts(dataset="megadepth", num_classes=None, image_size=None), built.data)
    with pytest.raises(RuntimeError, match="holds no classes"):
        trainer.input_check(DataFacts(dataset="megadepth", num_classes=3, image_size=None), built.data)


# -- the optimizer seam: the head at 10x from a pretrained start, nothing from scratch ----------------


def test_the_optimizer_options_give_the_head_10x_and_nothing_else_from_a_pretrained_start(world, megadepth_tree,
                                                                                          tmp_path):
    from train.depth_anything.train_depth_anything import DECODER_LR_MULTIPLIER

    trainer, resolved, ctx = prepared(tmp_path / "run", megadepth_tree)
    data = SimpleNamespace(input_shape=(SIZE, SIZE, 3))
    model = trainer.build_model(resolved, ctx, data)
    options = trainer.optimizer_options(resolved, model)
    assert set(options) == {"lr_multipliers"}
    table = options["lr_multipliers"]
    head = {v.path for v in model.decoder.trainable_variables}
    assert set(table) == head and set(table.values()) == {DECODER_LR_MULTIPLIER} == {10.0}
    encoder = {v.path for v in model.trainable_variables} - head
    assert encoder and not encoder & set(table), "no encoder variable is in the table"
    scratch, resolved, ctx = prepared(tmp_path / "run2", megadepth_tree, model_args={"pretrained": False})
    assert scratch.optimizer_options(resolved, scratch.build_model(resolved, ctx, data)) == {}


def test_the_optimizer_options_are_asked_after_the_model_was_built():
    with pytest.raises(RuntimeError, match="build_model"):
        bare_trainer().optimizer_options(None, None)


# -- the refusals of the task ---------------------------------------------------------------------------


def test_a_missing_weights_file_is_refused_by_check_model_args(world, megadepth_tree, tmp_path, monkeypatch):
    monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path / "no_weights"))
    with pytest.raises(ConfigError, match=WEIGHTS_ENV):
        prepared(tmp_path / "run", megadepth_tree)
    assert not (tmp_path / "run").exists()


def test_a_conformal_level_and_a_mixed_precision_are_refused_by_name(megadepth_tree, tmp_path):
    module = task_module()
    asked = {"out": str(tmp_path / "run"), "model": "depth_anything/vit_s", "data": {"root": str(megadepth_tree)}}
    with pytest.raises(ConfigValidationError, match=r"'analysis_args\.conformal_alpha'.*disparity"):
        validate(module.DepthEstimationConfig, {**asked, "analysis_args": {"conformal_alpha": 0.1}})
    with pytest.raises(ConfigValidationError, match=r"'precision'.*float32"):
        validate(module.DepthEstimationConfig, {**asked, "precision": "mixed_float16"})


# -- the registration ----------------------------------------------------------------------------------


def test_the_task_and_its_family_are_registered(world):
    from train.api.tasks import BUILTIN_TASKS

    (spec,) = [spec for spec in BUILTIN_TASKS if spec.name == TASK]
    assert spec.load_config_class() is task_module().DepthEstimationConfig
    assert spec.load_function() is task_module().train_depth_estimation
    assert TASK in registry.TASKS
    assert registry.MODELS["depth_anything"].tasks == (TASK,)


def test_the_lazy_surface_names_the_four_depth_rows():
    import train.api as api

    for name in ("DepthData", "load_depth_data_config", "DepthEstimationConfig", "train_depth_estimation"):
        assert name in api.__all__, name
        assert getattr(api, name) is not None


def run_cli(*words):
    env = {**os.environ, "PYTHONPATH": WT_SRC, "CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg"}
    return subprocess.run([sys.executable, "-m", "train", *words], env=env, capture_output=True, text=True,
                          timeout=300)


def test_list_tasks_and_list_models_name_the_task_and_its_three_variants():
    tasks = run_cli("list_tasks")
    assert tasks.returncode == 0, tasks.stderr[-2000:]
    assert TASK in tasks.stdout
    models = run_cli("list_models", f"task={TASK}")
    assert models.returncode == 0, models.stderr[-2000:]
    assert models.stdout.split() == ["depth_anything/vit_b", "depth_anything/vit_l", "depth_anything/vit_s"], models.stdout


# ---------------------------------------------------------------------
# Third part: stubbed end to end, the real weights, a constant model and the refusals
# ---------------------------------------------------------------------
#
# One request (from random weights, 28 px, E2E_EPOCHS epochs of E2E_STEPS steps, CPU, the synthetic tree) runs twice:
# through ``train_depth_estimation`` with the real vit_s, then through ``cli.main`` with a constant-output stand-in
# installed as the family's adapter (the run renamed between the two). The stand-in changes the model, not the request,
# so the two ``config.json`` must still be byte-equal, and the command line must exit 1 for the reason a constant model
# misses: it scores the constant baseline. One 1-epoch run starts from the official DINOv2 weights. The refusals run
# through ``cli.main`` and claim nothing.
#
# Cost. A vit_s fit is the unit of cost of this file: Keras traces its train step twice in the first fit (about 21,000
# graph ops each), on pure-python protobuf (no C extension in this venv), with the warm-up schedule rebuilt for each of
# the 234 per-variable learning-rate reads of the weight decay in each trace; a whole 2-epoch run took 135 s at a machine load of 4, and two took 440 s at a
# load of 17 to 37 (measured 2026-11-17). So the file trains exactly two vit_s models, each in one module fixture.
#
# On the synthetic tree (a smooth depth gradient plus noise) the average layout scores about 0.33, far below the constant's
# 1.0, so the layout term sets the bar and a 2-epoch from-scratch run misses it: the tests read the verdict, they do not
# expect a pass.

import json

import keras

from train.api import cli

E2E_EPOCHS, E2E_STEPS, E2E_BATCH = 2, 3, 4
E2E_DATA = {"train_split": 0.6, "n_select": 8, "n_test": 8, "layout_samples": 16, "steps_per_epoch": E2E_STEPS,
            "workers": 0}
#: A fit that runs more batches than this has an epoch with no length (an infinite stream and no ``steps_per_epoch``):
#: the spy stops it, so the missing keyword is reported by the tests instead of hanging the file.
RUNAWAY_BATCHES = 4 * E2E_EPOCHS * E2E_STEPS


class FitSpy(keras.callbacks.Callback):
    """Counts the batches of each epoch of one ``fit`` and stops a fit whose epoch never ends (reported, not hung)."""

    def __init__(self):
        super().__init__()
        self.last_batch, self.total, self.epoch = {}, 0, 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        self.total += 1
        self.last_batch[self.epoch] = batch
        if self.total > RUNAWAY_BATCHES:
            self.model.stop_training = True


def unscoped(path):
    """A variable path without the model's own name (``depth_anything_5/block_0/...`` -> ``block_0/...``)."""
    return path.partition("/")[2]


def spying_fit(fits, snapshot_encoder=False):
    """A replacement of ``keras.Model.fit`` recording its keywords, the optimizer it trains with and a :class:`FitSpy`."""
    real_fit = keras.Model.fit

    def fit(self, *args, **keywords):
        spy = FitSpy()
        fits.append({
            "keywords": {k: keywords.get(k) for k in ("steps_per_epoch", "validation_steps")}, "spy": spy,
            "model": self, "optimizer": self.optimizer,
            "encoder": [(unscoped(v.path), v.numpy()) for v in self.encoder.weights] if snapshot_encoder else None,
        })
        return real_fit(self, *args, **{**keywords, "callbacks": [*keywords.get("callbacks", []), spy]})

    return fit


def request(root, tree, **changes):
    asked = {"out": str(root / "run"), "model": "depth_anything/vit_s", "data": {"root": str(tree), **E2E_DATA},
             "epochs": E2E_EPOCHS, "batch_size": E2E_BATCH, "model_args": {"pretrained": False},
             "transform_args": {"image_size": SIZE}, "analysis_args": {"model_analysis": False, "figures": True}}
    return {**asked, **changes}


def cli_tokens(asked):
    from tests.test_train.test_api.test_forecast_task import flatten_request

    return [TASK, *flatten_request(asked)]


def read_json(path):
    return json.loads(Path(path).read_text())


def names(path):
    return {child.name for child in Path(path).iterdir()}


@keras.saving.register_keras_serializable(package="test_depth_task")
class ConstantDepthModel(keras.Model):
    """A constant (zero) disparity map as a real, savable model: one trainable scalar times zero, so the optimizer has
    a variable and a gradient and the output never changes. It IS the constant baseline at every epoch."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.idle = None

    def build(self, input_shape):
        self.idle = self.add_weight(name="idle", shape=(), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        return keras.ops.zeros_like(inputs[..., :1]) + 0.0 * self.idle


def create_constant_depth(*, variant, num_classes=None, input_shape, **model_args):
    """The depth adapter's keywords; a :class:`ConstantDepthModel` built at ``(None, s, s, 3)``."""
    model = ConstantDepthModel()
    model.build((None, *input_shape))
    return model


@pytest.fixture(scope="module")
def ended(world, megadepth_tree, tmp_path_factory):
    """The request through the function (a real vit_s) and through ``cli.main`` (the constant stand-in), in this process."""
    root = tmp_path_factory.mktemp("end_to_end")
    asked = request(root, megadepth_tree)
    fits = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(keras.Model, "fit", spying_fit(fits))
        result = task_module().train_depth_estimation(**asked)
        (root / "run").rename(root / "function")
        patch.setattr(depth_models, "create_depth_anything", create_constant_depth)
        code = cli.main(cli_tokens(asked))
        (root / "run").rename(root / "cli")
    return SimpleNamespace(root=root, asked=asked, result=result, code=code, fits=fits, function=root / "function",
                           cli=root / "cli")


#: The files of a finished depth run, from the code: the sequence writes the config pair, the log and the lock
#: (``run_sequence``), the callbacks the csv, the best checkpoint, the backup, the sidecar and tensorboard
#: (``fit_callbacks``), the reporting the history, the final model and the summary (``report_data_free_run``,
#: ``write_summary_json``), and the sequence ``task.json`` before ``fit`` (``task_record``, the depth record).
RUN_FILES = {
    "best_model.keras", "config.json", "config_requested.json", "final_model.keras", "results_summary.json",
    "run.lock", "run.log", "task.json", "training_history.json", "training_log.csv",
}
#: ``analysis_args.model_analysis=false`` writes no ``model_analysis/``.
RUN_DIRS = {"checkpoints", "tensorboard", "visualizations"}
#: The dashboard of the callbacks; the prediction grid draws every ``monitor_every_n_epochs`` epochs of the legacy
#: config, which a 2-epoch run does not reach.
FIGURES = {"training_dashboard.png"}


@pytest.mark.parametrize("which", ["function", "cli"])
def test_the_run_directory_holds_the_listing_the_code_writes(ended, which):
    run_dir = getattr(ended, which)
    assert names(run_dir) == RUN_FILES | RUN_DIRS, "an added or a missing file is a change of the contract"
    assert names(run_dir / "checkpoints") == {"backup", "fit_state.json"}
    assert names(run_dir / "tensorboard") == {"train", "validation"}
    assert names(run_dir / "visualizations") == FIGURES
    summary = read_json(run_dir / "results_summary.json")
    assert summary["visualizations"]["files"] == sorted(FIGURES) and summary["analyzer"]["status"] == "skipped"


def test_the_function_and_the_cli_write_the_same_config_bytes(ended):
    for name in ("config.json", "config_requested.json"):
        assert (ended.function / name).read_bytes() == (ended.cli / name).read_bytes(), name
    resolved = read_json(ended.cli / "config.json")
    assert (resolved["model"], resolved["epochs"], resolved["batch_size"]) == ("depth_anything/vit_s", E2E_EPOCHS,
                                                                                E2E_BATCH)
    assert resolved["transform_args"]["image_size"] == SIZE
    assert resolved["optim_args"]["lr"] == depth_args.FROM_SCRATCH_LR and resolved["model_args"] == {"pretrained": False}
    assert ended.result.success is read_json(ended.function / "results_summary.json")["success"]


def test_an_epoch_is_steps_per_epoch_steps_and_fit_sees_both_lengths(ended):
    summary = read_json(ended.function / "results_summary.json")
    assert summary["steps_per_epoch"] == E2E_STEPS and summary["n_train"] == E2E_STEPS * E2E_BATCH
    assert len(ended.fits) == 2, "one fit per run: the function and the command line"
    for fit in ended.fits:
        assert fit["keywords"] == {"steps_per_epoch": E2E_STEPS, "validation_steps": E2E_DATA["n_select"] // E2E_BATCH}
        # Observed, not read back: each epoch ran batches 0..STEPS-1; a stream with no length would have run away.
        assert fit["spy"].last_batch == {epoch: E2E_STEPS - 1 for epoch in range(E2E_EPOCHS)}, "an epoch has the wrong length"


@pytest.fixture(scope="module")
def oracle(ended, megadepth_tree, tmp_path_factory):
    """The request's data built again (a pure function of the request) and the function run's best checkpoint."""
    from train.common import depth_run

    trainer, resolved, ctx = prepared(tmp_path_factory.mktemp("oracle") / "run", megadepth_tree,
                                      data={"root": str(megadepth_tree), **E2E_DATA}, model_args={"pretrained": False})
    data = trainer.build_data(resolved, ctx)
    best = keras.saving.load_model(ended.function / "best_model.keras", compile=False)
    return SimpleNamespace(data=data, baselines=depth_run.selection_baselines(data),
                           held_out=depth_run.held_out_report(best, data))


@pytest.mark.parametrize("which", ["function", "cli"])
def test_the_summary_holds_the_selection_block_and_the_test_block_of_the_best_weights(ended, oracle, which):
    run_dir = getattr(ended, which)
    summary = read_json(run_dir / "results_summary.json")
    wanted = {"status", "success_metric", "monitor", "task", "baselines", "epoch_zero", "success_threshold",
              "success_value", "success", "held_out", "test_metrics_best", "scenes", "stream"}
    assert wanted <= set(summary), f"the summary lacks {sorted(wanted - set(summary))}"
    assert (summary["status"], summary["success_metric"], summary["monitor"], summary["task"]) == (
        "ok", "val_loss", "val_loss", TASK)
    # The selection block: the two baselines measured again on the same crops, the epoch-zero value of the sidecar, the
    # threshold the minimum of the three, the value the minimum val_loss, and the verdict the strict float32 bar.
    assert summary["baselines"] == pytest.approx(oracle.baselines, rel=1e-6)
    assert abs(summary["baselines"]["constant"] - 1.0) < 1e-5
    epoch_zero = read_json(run_dir / "checkpoints" / FIT_STATE_NAME)["initial"]["metrics"]["loss"]
    assert summary["epoch_zero"]["val_loss"] == epoch_zero
    assert summary["success_threshold"] == min(summary["baselines"]["constant"], summary["baselines"]["layout"], epoch_zero)
    assert summary["success_threshold"] == summary["baselines"]["layout"], "on the synthetic tree the layout binds"
    column = [float(v) for v in read_json(run_dir / "training_history.json")["val_loss"]]
    assert len(column) == E2E_EPOCHS and summary["success_value"] == pytest.approx(min(column), rel=1e-6)
    batches = E2E_DATA["n_select"] // E2E_BATCH
    assert summary["success"] is forecast_run.beats_bar(summary["success_value"], summary["success_threshold"], batches)
    # The test block: every key of the held-out report, n the test crops, the constant 1.0.
    assert set(summary["held_out"]) == set(oracle.held_out)
    assert summary["held_out"]["n"] == E2E_DATA["n_test"] and summary["held_out"]["constant"] == pytest.approx(1.0, abs=1e-5)
    assert summary["test_metrics_best"]["loss"] == pytest.approx(summary["held_out"]["loss"], rel=1e-5)
    assert summary["scenes"] == {"train": len(oracle.data.train_scenes), "select": len(oracle.data.select_scenes),
                                 "test": len(oracle.data.test_scenes)}


def test_the_test_block_is_the_held_out_report_of_the_best_checkpoint(ended, oracle):
    # Scored again here from the function run's best_model.keras on the rebuilt test crops: the summary scored the BEST
    # weights (EarlyStopping restored them before the report), not the last epoch's.
    summary = read_json(ended.function / "results_summary.json")
    assert summary["held_out"] == pytest.approx(oracle.held_out, rel=1e-5, nan_ok=True)


@pytest.mark.parametrize("which", ["function", "cli"])
def test_the_best_checkpoint_reloads_with_a_gap_of_exactly_zero(ended, which):
    summary = read_json(getattr(ended, which) / "results_summary.json")
    assert summary["best_checkpoint_load_error"] is None and summary["best_checkpoint_max_abs_diff"] == 0.0
    assert summary["model_loading_validated"] is True
    assert summary["test_metrics_best"] is not None and summary["test_metrics_final"] is not None


def test_the_function_run_reports_its_verdict_and_the_cli_exits_by_it(ended):
    function = read_json(ended.function / "results_summary.json")
    assert ended.result.success is function["success"]
    cli_summary = read_json(ended.cli / "results_summary.json")
    assert ended.code == (0 if cli_summary["success"] else 1)


def test_a_constant_output_model_exits_one_because_it_scores_the_constant_and_never_beats_the_bar(ended):
    summary = read_json(ended.cli / "results_summary.json")
    assert summary["status"] == "ok", "vacuity guard: the run finished and was judged"
    constant = summary["baselines"]["constant"]
    band = 8 * float(np.spacing(np.float32(constant)))
    # The model is the constant baseline: its val_loss is the constant's to float32 rounding, at epoch zero and after.
    assert abs(summary["success_value"] - constant) <= band
    assert abs(summary["epoch_zero"]["val_loss"] - constant) <= band
    assert summary["success_value"] >= summary["success_threshold"] and summary["success"] is False
    assert ended.code == 1
    assert summary["held_out"]["loss"] == pytest.approx(constant, abs=band), "the test block scores the constant too"


# -- the official weights, one epoch: the pretrained path and the 10x head table in a real run ----------------------


@pytest.fixture(scope="module")
def pretrained_run(world, megadepth_tree, tmp_path_factory):
    """One 1-epoch run from the official weights through the function; skipped where the weights file is absent.

    TensorBoard, the two model files and the warm-up are off: this arm proves the start and the optimizer, and those
    cost seconds a fit (TensorBoard writes the 21,000-op graph; the warm-up schedule builds a ``tf.cond`` for each of the
    234 per-variable learning-rate reads of the weight decay, in each of the two traces of the step).
    """
    try:
        depth_models.check_weights("vit_s", {})
    except ConfigError as exc:
        pytest.skip(f"the official DINOv2 vit_s weights are not on this machine: {exc}")
    root = tmp_path_factory.mktemp("pretrained")
    asked = request(root, megadepth_tree, epochs=1, model_args={},
                    analysis_args={"model_analysis": False, "figures": False}, logger_args={"tensorboard": None},
                    checkpoint_args={"save_best": False, "save_last": False}, optim_args={"warmup_epochs": 0})
    fits = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(keras.Model, "fit", spying_fit(fits, snapshot_encoder=True))
        result = task_module().train_depth_estimation(**asked)
    return SimpleNamespace(result=result, fits=fits, run=root / "run")


def test_a_pretrained_run_trains_the_official_encoder_with_the_head_at_10x(pretrained_run):
    from dl_techniques.optimization.lr_multiplier_adamw import LRMultiplierAdamW
    from train.depth_anything.train_depth_anything import DECODER_LR_MULTIPLIER

    (fit,) = pretrained_run.fits
    model, optimizer = fit["model"], fit["optimizer"]
    assert isinstance(optimizer, LRMultiplierAdamW)
    head = {v.path for v in model.decoder.trainable_variables}
    assert optimizer.lr_multipliers == {path: DECODER_LR_MULTIPLIER for path in head} and DECODER_LR_MULTIPLIER == 10.0
    # The encoder fit started from is the official one: the library's own strict load, built here at the same size.
    reference = depth_models.create_depth_anything(variant="vit_s", input_shape=(SIZE, SIZE, 3))
    official = [(unscoped(v.path), v.numpy()) for v in reference.encoder.weights]
    assert [path for path, _ in fit["encoder"]] == [path for path, _ in official] and len(official) > 100
    for (path, started), (_, value) in zip(fit["encoder"], official):
        np.testing.assert_array_equal(started, value, err_msg=path)
    resolved = read_json(pretrained_run.run / "config.json")
    # The request asked nothing of the model; the recipe's start is what the run was built with (`used`).
    assert resolved["model_args"] == {} and resolved["used"]["model_args"] == {"pretrained": True}
    assert resolved["optim_args"]["lr"] == depth_models.PRETRAINED_LR == 5e-6
    summary = read_json(pretrained_run.run / "results_summary.json")
    assert summary["status"] == "ok" and summary["epochs_run"] == 1
    assert pretrained_run.result.success is summary["success"]
    assert fit["spy"].last_batch == {0: E2E_STEPS - 1}


# -- the refusals: exit 2 through the command line and no run directory ---------------------------------------------


REFUSALS = {
    "missing_weights": ({"model_args": {}}, (WEIGHTS_ENV, "dinov2-small")),
    "size_not_a_multiple_of_14": ({"transform_args": {"image_size": 30}}, ("'transform_args.image_size'", "14")),
    "mixed_precision": ({"precision": "mixed_float16"}, ("'precision'", "float32")),
    "placeholder_encoder": ({"model_args": {"pretrained": False, "encoder_kind": "placeholder"}},
                            ("'model_args.encoder_kind'",)),
    "feature_alignment": ({"model_args": {"pretrained": False, "use_feature_alignment": True}},
                          ("'model_args.use_feature_alignment'",)),
    "semi_supervised_key": ({"model_args": {"pretrained": False, "enable_semi_supervised": True}},
                            ("'model_args.enable_semi_supervised'", "semi-supervised")),
    "conformal": ({"analysis_args": {"conformal_alpha": 0.1}}, ("'analysis_args.conformal_alpha'", "disparity")),
    "sgd_on_a_pretrained_start": (
        {"model_args": {}, "optim_args": {"optimizer": "sgd", "lr": 0.01, "weight_decay": None}},
        ("'optim_args.optimizer'", "adamw")),
    "fewer_than_3_validation_scenes": ({"data": {**E2E_DATA, "train_split": 0.8}}, ("at least 3",)),
    "backup_inside_an_epoch": ({"checkpoint_args": {"backup_every_steps": 2}},
                               ("'checkpoint_args.backup_every_steps'", "PyDataset", "epoch end")),
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_a_refused_request_exits_two_and_claims_no_run_directory(world, megadepth_tree, tmp_path, monkeypatch, capsys,
                                                                 case):
    changes, fragments = REFUSALS[case]
    if case == "missing_weights":
        monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path / "no_weights"))
    if "data" in changes:
        changes = {**changes, "data": {"root": str(megadepth_tree), **changes["data"]}}
    asked = request(tmp_path, megadepth_tree, **changes)
    code = cli.main(cli_tokens(asked))
    error = capsys.readouterr().err
    assert code == 2, error[-2000:]
    assert not (tmp_path / "run").exists(), "a refusal claims no run directory"
    for fragment in fragments:
        assert fragment in error, (case, fragment, error[-2000:])


def test_a_backup_inside_an_epoch_is_refused_by_the_function_before_the_claim(world, megadepth_tree, tmp_path):
    asked = request(tmp_path, megadepth_tree, checkpoint_args={"backup_every_steps": 2})
    with pytest.raises(ConfigValidationError, match=r"'checkpoint_args\.backup_every_steps'.*PyDataset"):
        task_module().train_depth_estimation(**asked)
    assert not (tmp_path / "run").exists(), "a refusal claims no run directory"


# -- why a backup inside an epoch is refused: Keras 3.8 cannot cut an epoch of a PyDataset ------------------------


class _Killed(Exception):
    """The in-process stand-in for the SIGKILL of the first process."""


class _KillInsideEpochOne(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        if self.epoch == 1 and batch == 3:
            raise _Killed()


class _BatchesPerEpoch(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.counts, self.ran = {}, 0

    def on_epoch_begin(self, epoch, logs=None):
        self.ran = 0

    def on_train_batch_end(self, batch, logs=None):
        self.ran += 1

    def on_epoch_end(self, epoch, logs=None):
        self.counts[epoch] = self.ran


class _InfiniteBatches(keras.utils.PyDataset):
    """An infinite PyDataset read in this process, the shape of the depth train stream with ``data.workers=0``."""

    def __init__(self):
        super().__init__(workers=1, use_multiprocessing=False, max_queue_size=4)

    @property
    def num_batches(self):
        return None

    def __getitem__(self, index):
        rng = np.random.default_rng(int(index))
        return rng.normal(size=(4, 3)).astype("float32"), rng.normal(size=(4, 1)).astype("float32")


def _repeated_dataset():
    import tensorflow as tf

    rng = np.random.default_rng(0)
    return tf.data.Dataset.from_tensor_slices((rng.normal(size=(24, 3)).astype("float32"),
                                               rng.normal(size=(24, 1)).astype("float32"))).batch(4).repeat()


def _kill_inside_an_epoch_and_resume(backup_dir, make_data):
    """Process 1 backs up every 2 of 6 batches and is killed after batch 4 of epoch 1; process 2 resumes from that
    copy with the shared callbacks in the order the sequence uses (``finish`` first, the backup last)."""
    from train.common.resume_state import StepBackupAndRestore

    def fit(callbacks):
        model = keras.Sequential([keras.Input((3,)), keras.layers.Dense(1)])
        model.compile(keras.optimizers.SGD(0.01), "mse", jit_compile=False)
        model.fit(make_data(), epochs=3, steps_per_epoch=6, verbose=0, callbacks=callbacks)

    backup = StepBackupAndRestore(backup_dir, save_every_steps=2, steps_per_epoch=6)
    with pytest.raises(_Killed):
        fit([backup.finish, backup, _KillInsideEpochOne()])
    backup, counted = StepBackupAndRestore(backup_dir, save_every_steps=2, steps_per_epoch=6), _BatchesPerEpoch()
    try:
        fit([backup.finish, counted, backup])
    finally:
        assert backup.resumed_inside == (1, 4), backup.resumed_inside
    return counted.counts


def test_keras_fails_the_epoch_after_a_cut_epoch_of_a_pydataset_which_is_why_the_backup_inside_an_epoch_is_refused(
        tmp_path):
    """Pins the Keras behaviour the refusal of ``checkpoint_args.backup_every_steps`` for this task rests on.

    A resume from a backup taken inside an epoch ends that epoch early (``FinishKilledEpoch`` sets ``stop_training``
    after the batches the copy had not taken). Keras 3.8's fit loop then leaves the epoch with ``break``, so the
    ``PyDataset`` adapter's ``on_epoch_end`` never runs and the next epoch raises. A ``tf.data`` stream, the train data
    of the other tasks, takes the same cut and goes on. The depth train stream is a ``PyDataset``. When this test
    fails (Keras or the shared callback changed so the cut epoch of a ``PyDataset`` is followed by a whole one), the
    refusal in ``DepthEstimationConfig.problems`` can be lifted, with its refusal tests and the README sentence.
    """
    from train.common.depth_data import DepthStream

    assert issubclass(DepthStream, keras.utils.PyDataset)
    assert _kill_inside_an_epoch_and_resume(tmp_path / "tf_data", _repeated_dataset) == {1: 2, 2: 6}
    with pytest.raises(ValueError, match="on_epoch_begin` was called twice without `on_epoch_end"):
        _kill_inside_an_epoch_and_resume(tmp_path / "py_dataset", _InfiniteBatches)
