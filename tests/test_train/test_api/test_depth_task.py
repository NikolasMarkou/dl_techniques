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
