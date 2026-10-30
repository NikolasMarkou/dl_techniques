"""``"auto"`` fields are filled from the family recipe and the data, and only those.

The recipe here is a small fake. The real family tables arrive with the
registry; what is pinned now is the mechanism: literals for the success
threshold, the mode read from the metric name, an explicit value surviving
(D-010), and a missing recipe entry being reported by dotted path.
"""

import dataclasses
from typing import Any, Mapping

import pytest

from train.api.args import (
    AnalysisArgs,
    AutoContext,
    CheckpointArgs,
    DebugArgs,
    LoggerArgs,
    MetricArgs,
    OptimArgs,
    TransformArgs,
)
from train.api.config import AUTO, StrictConfig, assert_resolved, validate
from train.api.errors import (
    ConfigError,
    ConfigValidationError,
    UnresolvedAutoError,
)

GROUPS = (
    OptimArgs,
    TransformArgs,
    MetricArgs,
    CheckpointArgs,
    LoggerArgs,
    AnalysisArgs,
    DebugArgs,
)

FAKE_RECIPE: Mapping[str, Mapping[str, Any]] = {
    "optim_args": {
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 0.05,
        "momentum": None,
        "beta_1": None,
        "beta_2": None,
        "epsilon": None,
        "schedule": "cosine_decay",
        "warmup_epochs": 5,
        "warmup_start_lr": 1e-8,
        "min_lr_ratio": 0.01,
        "decay_over": "total",
        "clipnorm": None,
        "global_clipnorm": 1.0,
    },
    "transform_args": {"augment": "cifar_light", "random_erasing_prob": 0.0},
    # top_k and save_best are not AUTO fields: the recipe may carry values for
    # them, and they must never win over the class default or an explicit value.
    "metric_args": {"watch_metric": "val_accuracy", "top_k": 3},
    "checkpoint_args": {"early_stopping_patience": 15, "save_best": False},
}


def context(num_classes: int = 10, defaults: Mapping[str, Any] = None) -> AutoContext:
    return AutoContext(
        family="fake",
        variant="fake_tiny",
        dataset="cifar10",
        num_classes=num_classes,
        image_size=32,
        defaults=FAKE_RECIPE if defaults is None else defaults,
    )


@dataclasses.dataclass(frozen=True)
class Task(StrictConfig):
    """A stand-in task config: the seven groups and nothing else."""

    optim_args: OptimArgs = dataclasses.field(default_factory=OptimArgs)
    transform_args: TransformArgs = dataclasses.field(default_factory=TransformArgs)
    metric_args: MetricArgs = dataclasses.field(default_factory=MetricArgs)
    checkpoint_args: CheckpointArgs = dataclasses.field(default_factory=CheckpointArgs)
    logger_args: LoggerArgs = dataclasses.field(default_factory=LoggerArgs)
    analysis_args: AnalysisArgs = dataclasses.field(default_factory=AnalysisArgs)
    debug_args: DebugArgs = dataclasses.field(default_factory=DebugArgs)


def resolve_all(task: Task, ctx: AutoContext) -> Task:
    return dataclasses.replace(
        task,
        **{
            field.name: getattr(task, field.name).resolve_auto(ctx)
            for field in dataclasses.fields(task)
        },
    )


def resolved(overrides: Mapping[str, Any] = None, ctx: AutoContext = None) -> Task:
    return resolve_all(validate(Task, overrides or {}), ctx or context())


# ---------------------------------------------------------------------
# The mechanism
# ---------------------------------------------------------------------


def test_every_group_field_is_documented():
    for group in GROUPS:
        for field in dataclasses.fields(group):
            assert field.metadata.get("help"), f"{group.__name__}.{field.name}"


def test_a_default_run_resolves_to_the_recipe_and_the_data():
    task = resolved()
    assert_resolved(task)
    assert task.optim_args.lr == 3e-4
    assert task.optim_args.optimizer == "adamw"
    assert task.optim_args.global_clipnorm == 1.0
    assert task.optim_args.clipnorm is None
    assert task.transform_args.augment == "cifar_light"
    assert task.transform_args.image_size == 32
    assert task.checkpoint_args.early_stopping_patience == 15
    assert task.metric_args.watch_metric == "val_accuracy"


def test_resolve_returns_a_new_instance_and_leaves_the_original():
    original = OptimArgs()
    result = original.resolve_auto(context())
    assert result is not original
    assert original.lr == AUTO
    assert result.lr == 3e-4


@pytest.mark.parametrize(
    "num_classes, expected", [(10, 0.2), (100, 0.05), (2, 0.95), (1000, 0.05)]
)
def test_success_threshold_literals(num_classes, expected):
    metric = MetricArgs().resolve_auto(context(num_classes))
    assert metric.success_threshold == expected


@pytest.mark.parametrize(
    "metric, mode",
    [("val_accuracy", "max"), ("val_loss", "min"), ("val_top5_accuracy", "max")],
)
def test_watch_mode_is_read_from_the_metric_name(metric, mode):
    recipe = {"metric_args": {"watch_metric": metric}}
    assert MetricArgs().resolve_auto(context(defaults=recipe)).watch_mode == mode


def test_an_explicit_watch_metric_decides_the_mode_too():
    metric = MetricArgs(watch_metric="val_loss").resolve_auto(context())
    assert (metric.watch_metric, metric.watch_mode) == ("val_loss", "min")


def test_an_explicit_watch_mode_is_kept():
    metric = MetricArgs(watch_mode="min").resolve_auto(context())
    assert metric.watch_mode == "min"
    assert metric.watch_metric == "val_accuracy"


# ---------------------------------------------------------------------
# An explicit value is never replaced (D-010)
# ---------------------------------------------------------------------


def test_an_explicit_lr_equal_to_the_recipe_survives_and_so_does_a_different_one():
    assert OptimArgs(lr=3e-4).resolve_auto(context()).lr == 3e-4
    assert OptimArgs(lr=1e-3).resolve_auto(context()).lr == 1e-3


def test_an_explicit_value_equal_to_the_class_default_beats_the_recipe():
    # top_k=5 and save_best=True are the class defaults; the fake recipe
    # carries 3 and False for them. Neither field is AUTO, so the recipe must
    # not touch them, however the caller came to hold the class default.
    task = resolved({"metric_args": {"top_k": 5}, "checkpoint_args": {"save_best": True}})
    assert task.metric_args.top_k == 5
    assert task.checkpoint_args.save_best is True
    untouched = resolved()
    assert untouched.metric_args.top_k == 5
    assert untouched.checkpoint_args.save_best is True


def test_an_explicit_none_survives():
    assert OptimArgs(global_clipnorm=None).resolve_auto(context()).global_clipnorm is None


def test_an_explicit_clipnorm_does_not_collide_with_the_recipe_global_clip():
    optim = OptimArgs(clipnorm=0.5).resolve_auto(context())
    assert (optim.clipnorm, optim.global_clipnorm) == (0.5, None)


def test_an_explicit_global_clipnorm_does_not_collide_with_the_recipe_clip():
    recipe = {"optim_args": {**FAKE_RECIPE["optim_args"], "clipnorm": 1.0, "global_clipnorm": None}}
    optim = OptimArgs(global_clipnorm=2.0).resolve_auto(context(defaults=recipe))
    assert (optim.clipnorm, optim.global_clipnorm) == (None, 2.0)


def test_an_explicit_augment_is_kept():
    assert TransformArgs(augment="flip").resolve_auto(context()).augment == "flip"


def test_an_explicit_image_size_is_kept():
    assert TransformArgs(image_size=64).resolve_auto(context()).image_size == 64


# ---------------------------------------------------------------------
# What cannot be filled is reported
# ---------------------------------------------------------------------


def test_a_missing_recipe_entry_stays_auto_and_is_named_by_path():
    recipe = {k: dict(v) for k, v in FAKE_RECIPE.items()}
    del recipe["optim_args"]["lr"]
    del recipe["checkpoint_args"]["early_stopping_patience"]
    task = resolved(ctx=context(defaults=recipe))
    assert task.optim_args.lr == AUTO
    with pytest.raises(UnresolvedAutoError) as caught:
        assert_resolved(task)
    message = str(caught.value)
    assert "optim_args.lr" in message
    assert "checkpoint_args.early_stopping_patience" in message
    assert "optim_args.optimizer" not in message


def test_an_empty_recipe_leaves_the_recipe_fields_and_none_of_the_data_fields():
    task = resolved(ctx=context(defaults={}))
    with pytest.raises(UnresolvedAutoError) as caught:
        assert_resolved(task)
    message = str(caught.value)
    for path in ("optim_args.lr", "transform_args.augment", "metric_args.watch_metric"):
        assert path in message
    # Data decides these two, so they resolve even with no recipe.
    assert task.transform_args.image_size == 32
    assert task.metric_args.success_threshold == 0.2
    assert "transform_args.image_size" not in message
    assert "metric_args.success_threshold" not in message


def test_an_unknown_augment_preset_is_refused_naming_the_choices():
    recipe = {"transform_args": {"augment": "cifar_heavy"}}
    with pytest.raises(ConfigError) as caught:
        TransformArgs().resolve_auto(context(defaults=recipe))
    assert "transform_args.augment" in str(caught.value)
    assert "cifar_light" in str(caught.value)


def test_an_unknown_explicit_augment_is_refused_too():
    with pytest.raises(ConfigError, match="not a preset"):
        TransformArgs(augment="flpi").resolve_auto(context())


def test_every_real_preset_is_accepted():
    from train.common.image_augment import PRESETS

    for name in PRESETS:
        assert TransformArgs(augment=name).resolve_auto(context()).augment == name


def test_a_recipe_value_of_the_wrong_type_names_the_group():
    recipe = {"optim_args": {"lr": "3e-4"}}
    with pytest.raises(ConfigValidationError) as caught:
        OptimArgs().resolve_auto(context(defaults=recipe))
    assert caught.value.errors == (
        "Family recipe for 'optim_args': Invalid value for 'lr': expected float, "
        "got '3e-4' (str)",
    )


def test_a_recipe_int_for_a_float_field_becomes_a_float():
    recipe = {"optim_args": {"lr": 1}}
    assert OptimArgs().resolve_auto(context(defaults=recipe)).lr == 1.0
    assert isinstance(OptimArgs().resolve_auto(context(defaults=recipe)).lr, float)


def test_zero_classes_are_refused():
    with pytest.raises(ConfigError, match="0 classes"):
        MetricArgs().resolve_auto(context(num_classes=0))


# ---------------------------------------------------------------------
# The groups themselves
# ---------------------------------------------------------------------


def test_the_seven_groups_default_to_their_documented_values():
    assert MetricArgs().top_k == 5
    assert CheckpointArgs().save_best is True and CheckpointArgs().save_last is True
    assert LoggerArgs().tensorboard == {}
    assert AnalysisArgs() == AnalysisArgs(
        epoch_analysis_every=0, model_analysis=True, figures=True, conformal_alpha=None
    )
    assert DebugArgs() == DebugArgs(terminate_on_nan=True, smoke=False)
    assert TransformArgs().normalize == "standardize"


def test_a_tensorboard_dict_with_keys_is_refused_and_null_is_accepted():
    assert LoggerArgs(tensorboard=None).tensorboard is None
    with pytest.raises(ConfigValidationError, match="logger_args.tensorboard"):
        LoggerArgs(tensorboard={"update_freq": "batch"})
    with pytest.raises(ConfigValidationError, match="logger_args.tensorboard"):
        validate(Task, {"logger_args": {"tensorboard": {"update_freq": "batch"}}})


def test_the_group_fields_refuse_text_from_a_python_caller():
    with pytest.raises(ConfigValidationError) as caught:
        validate(Task, {"optim_args": {"lr": "abc"}, "metric_args": {"top_k": "5"}})
    assert caught.value.errors == (
        "Invalid value for 'optim_args.lr': expected float, got 'abc' (str)",
        "Invalid value for 'metric_args.top_k': expected int, got '5' (str)",
    )
