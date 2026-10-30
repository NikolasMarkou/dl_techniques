"""``"auto"`` fields are filled from the family recipe and the data, and only those.

The recipe here is a small fake. The real family tables arrive with the
registry; what is pinned now is the mechanism: literals for the success
threshold, the mode read from the metric name, an explicit value surviving
(D-010), and a missing recipe entry being reported by dotted path.
"""

import dataclasses
from typing import Any, Mapping

import pytest

import train.api.args as args_module
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

from ._groups import args_groups, task_class

GROUPS = tuple(args_groups())

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


# The stand-in task config is built from train.api.args, so a group added there
# is resolved, documented and checked by every test below (see _groups.py).
Task = task_class()


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
    "num_classes, expected",
    [(10, 0.2), (100, 0.05), (2, 0.95), (1000, 0.05), (1, 0.95), (3, 2 / 3), (7, 2 / 7)],
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


@pytest.mark.parametrize("value", [0.5, 0.0, 0, 1.0])
def test_an_explicit_success_threshold_is_kept_and_the_formula_is_not_applied(value):
    # The formula gives 0.2 for ten classes; none of these may become it.
    metric = MetricArgs(success_threshold=value).resolve_auto(context(10))
    assert metric.success_threshold == value
    assert type(metric.success_threshold) is float


ZERO_RECIPE = {
    "optim_args": {"warmup_epochs": 5, "warmup_start_lr": 1e-8, "min_lr_ratio": 0.01, "lr": 3e-4},
    "transform_args": {"random_erasing_prob": 0.25, "augment": "flip"},
    "checkpoint_args": {"early_stopping_patience": 15},
}


@pytest.mark.parametrize(
    "recipe_key, group, key, zero",
    [
        ("optim_args", OptimArgs, "warmup_epochs", 0),
        ("optim_args", OptimArgs, "warmup_start_lr", 0.0),
        ("optim_args", OptimArgs, "min_lr_ratio", 0.0),
        ("optim_args", OptimArgs, "lr", 0.0),
        ("transform_args", TransformArgs, "random_erasing_prob", 0.0),
        ("checkpoint_args", CheckpointArgs, "early_stopping_patience", 0),
    ],
)
def test_an_explicit_zero_beats_a_recipe_value_that_is_not_zero(recipe_key, group, key, zero):
    # 0 is falsy and equals False and 0.0; none of that may let the recipe in.
    assert ZERO_RECIPE[recipe_key][key] != 0
    result = group(**{key: zero}).resolve_auto(context(defaults=ZERO_RECIPE))
    assert getattr(result, key) == zero
    assert type(getattr(result, key)) is type(zero)


# The random-erasing probability belongs to one preset (D-041).

ERASING_RECIPE = {"transform_args": {"augment": "none", "random_erasing_prob": 0.25}}
AUTOAUGMENT_RECIPE = {
    "transform_args": {"augment": "cifar_autoaugment", "random_erasing_prob": 0.25}
}


@pytest.mark.parametrize(
    "recipe, explicit_augment, expected",
    [
        # Where the preset comes from does not matter, only what is in force.
        (ERASING_RECIPE, AUTO, 0.0),
        (ERASING_RECIPE, "cifar_autoaugment", 0.25),
        (ERASING_RECIPE, "flip", 0.0),
        (ERASING_RECIPE, "cifar_light", 0.0),
        (AUTOAUGMENT_RECIPE, AUTO, 0.25),
        (AUTOAUGMENT_RECIPE, "none", 0.0),
        (AUTOAUGMENT_RECIPE, "photometric", 0.0),
    ],
)
def test_an_auto_erasing_probability_follows_the_preset_in_force(
    recipe, explicit_augment, expected
):
    transform = TransformArgs(augment=explicit_augment).resolve_auto(context(defaults=recipe))
    assert transform.random_erasing_prob == expected
    assert type(transform.random_erasing_prob) is float


def test_an_explicit_erasing_probability_is_not_moved_by_the_preset_rule():
    # D-010: the rule fills AUTO only. 0.4 under a preset that takes it stays 0.4,
    # whether the preset is the recipe's or the caller's.
    kept = TransformArgs(random_erasing_prob=0.4).resolve_auto(
        context(defaults=AUTOAUGMENT_RECIPE)
    )
    assert kept.random_erasing_prob == 0.4
    explicit = TransformArgs(augment="cifar_autoaugment", random_erasing_prob=0.4).resolve_auto(
        context(defaults=ERASING_RECIPE)
    )
    assert (explicit.augment, explicit.random_erasing_prob) == ("cifar_autoaugment", 0.4)


@pytest.mark.parametrize(
    "recipe, augment",
    [
        (ERASING_RECIPE, AUTO),  # the recipe's preset, "none"
        (ERASING_RECIPE, "flip"),
        (AUTOAUGMENT_RECIPE, "cifar_light"),
        (AUTOAUGMENT_RECIPE, "photometric"),
    ],
)
def test_an_explicit_erasing_probability_under_another_preset_is_a_config_error(recipe, augment):
    # D-042: build_augment would raise a ValueError at the loader (a traceback, exit 1);
    # the refusal is made here, names the key, and exits 2 through the CLI.
    with pytest.raises(ConfigError) as caught:
        TransformArgs(augment=augment, random_erasing_prob=0.4).resolve_auto(
            context(defaults=recipe)
        )
    message = str(caught.value)
    assert message.startswith("Invalid value for 'transform_args.random_erasing_prob': 0.4")
    assert "cifar_autoaugment" in message


def test_an_explicit_zero_erasing_probability_is_fine_under_any_preset():
    # The control of the arm above: only a NONZERO probability is refused.
    result = TransformArgs(augment="flip", random_erasing_prob=0.0).resolve_auto(
        context(defaults=ERASING_RECIPE)
    )
    assert (result.augment, result.random_erasing_prob) == ("flip", 0.0)


# The two clipping fields exclude each other, and the rule that keeps them
# apart acts on a number only (D-028).


def test_an_explicit_null_clipnorm_does_not_clear_the_recipe_global_clip():
    recipe = {"optim_args": {"clipnorm": None, "global_clipnorm": 1.0}}
    optim = OptimArgs(clipnorm=None).resolve_auto(context(defaults=recipe))
    assert (optim.clipnorm, optim.global_clipnorm) == (None, 1.0)


def test_an_explicit_null_global_clipnorm_does_not_clear_the_recipe_clipnorm():
    recipe = {"optim_args": {"clipnorm": 1.0, "global_clipnorm": None}}
    optim = OptimArgs(global_clipnorm=None).resolve_auto(context(defaults=recipe))
    assert (optim.clipnorm, optim.global_clipnorm) == (1.0, None)


def test_two_explicit_clips_both_survive_resolution_and_the_refusal_is_not_ours():
    # optim.py refuses the pair later; resolution must not repair it silently.
    optim = OptimArgs(clipnorm=0.5, global_clipnorm=2.0).resolve_auto(context())
    assert (optim.clipnorm, optim.global_clipnorm) == (0.5, 2.0)


def test_an_explicit_clip_resolves_the_other_to_none_whatever_the_recipe_says():
    recipe = {"optim_args": {"clipnorm": 3.0, "global_clipnorm": 4.0}}
    one = OptimArgs(clipnorm=0.5).resolve_auto(context(defaults=recipe))
    other = OptimArgs(global_clipnorm=0.5).resolve_auto(context(defaults=recipe))
    assert (one.clipnorm, one.global_clipnorm) == (0.5, None)
    assert (other.clipnorm, other.global_clipnorm) == (None, 0.5)


def test_the_recipe_cannot_decide_the_mode_the_threshold_or_the_image_size():
    # Each of the three has its own rule: the metric's name, the class count, the data.
    recipe = {
        "metric_args": {"watch_metric": "val_accuracy", "watch_mode": "min", "success_threshold": 0.9},
        "transform_args": {"image_size": 224, "augment": "flip"},
    }
    metric = MetricArgs().resolve_auto(context(10, defaults=recipe))
    assert (metric.watch_metric, metric.watch_mode, metric.success_threshold) == ("val_accuracy", "max", 0.2)
    assert TransformArgs().resolve_auto(context(defaults=recipe)).image_size == 32


def test_a_recipe_mode_with_no_recipe_metric_leaves_the_mode_auto_to_be_reported():
    # The mode follows the resolved metric's name; with no metric to read it
    # from, the recipe's mode is not a substitute (it may belong to another metric).
    recipe = {"metric_args": {"watch_mode": "min"}}
    metric = MetricArgs().resolve_auto(context(defaults=recipe))
    assert (metric.watch_metric, metric.watch_mode) == (AUTO, AUTO)
    with pytest.raises(UnresolvedAutoError, match="metric_args.watch_mode"):
        assert_resolved(resolve_all(validate(Task, {}), context(defaults=recipe)))


def test_the_context_is_frozen():
    ctx = context()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.num_classes = 3


@pytest.mark.parametrize(
    "group",
    [
        OptimArgs(
            optimizer="sgd", lr=0.1, weight_decay=None, momentum=0.9, beta_1=None, beta_2=None,
            epsilon=None, schedule="constant", warmup_epochs=0, warmup_start_lr=0.0,
            min_lr_ratio=0.0, decay_over="total", clipnorm=None, global_clipnorm=None,
        ),
        TransformArgs(image_size=16, augment="flip", random_erasing_prob=0.0),
        MetricArgs(watch_metric="val_loss", watch_mode="min", success_threshold=0.5),
        CheckpointArgs(early_stopping_patience=4),
    ],
    ids=lambda group: type(group).__name__,
)
def test_a_group_with_nothing_to_fill_still_comes_back_as_a_new_equal_instance(group):
    result = group.resolve_auto(context())
    assert result == group
    assert result is not group


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


def test_classes_the_data_did_not_report_are_refused_as_the_data_s_fault():
    with pytest.raises(ConfigError, match="reports None classes"):
        MetricArgs().resolve_auto(context(num_classes=None))


def test_the_success_threshold_is_built_from_its_named_constants(monkeypatch):
    assert args_module.SUCCESS_CHANCE_MULTIPLE == 2
    assert args_module.SUCCESS_THRESHOLD_FLOOR == 0.05
    assert args_module.SUCCESS_THRESHOLD_CEILING == 0.95
    threshold = lambda n: MetricArgs().resolve_auto(context(num_classes=n)).success_threshold
    monkeypatch.setattr(args_module, "SUCCESS_CHANCE_MULTIPLE", 3)
    assert threshold(10) == 0.3
    monkeypatch.setattr(args_module, "SUCCESS_THRESHOLD_FLOOR", 0.4)
    assert threshold(10) == 0.4
    monkeypatch.setattr(args_module, "SUCCESS_THRESHOLD_CEILING", 0.35)
    assert threshold(10) == 0.35


# ---------------------------------------------------------------------
# A bad recipe or bad data is blamed on its source (D-034)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("table", [None, ["lr"], "lr", 3])
def test_a_recipe_group_that_is_not_a_mapping_gives_a_named_error(table):
    with pytest.raises(ConfigValidationError) as caught:
        OptimArgs().resolve_auto(context(defaults={"optim_args": table}))
    (line,) = caught.value.errors
    assert line.startswith("Family recipe for 'optim_args': expected a mapping of OptimArgs fields")


@pytest.mark.parametrize(
    "group, key, field",
    [
        (OptimArgs, "optim_args", "learning_rate"),
        (OptimArgs, "optim_args", "lrr"),
        (TransformArgs, "transform_args", "augmnt"),
        (MetricArgs, "metric_args", "tok_k"),
        (CheckpointArgs, "checkpoint_args", "early_stoping_patience"),
    ],
)
def test_a_recipe_key_that_is_not_a_field_is_an_error(group, key, field):
    with pytest.raises(ConfigValidationError) as caught:
        group().resolve_auto(context(defaults={key: {field: 1}}))
    (line,) = caught.value.errors
    assert line.startswith(f"Family recipe for {key!r}: Unknown key '{key}.{field}'")


def test_a_recipe_typo_is_an_error_even_when_the_field_it_resembles_is_not_auto():
    # save_best is a plain bool, never AUTO: the old code dropped its typos.
    recipe = {"checkpoint_args": {"early_stopping_patience": 15, "save_bset": False}}
    with pytest.raises(ConfigValidationError, match="Did you mean 'checkpoint_args.save_best'"):
        CheckpointArgs().resolve_auto(context(defaults=recipe))


def test_every_unknown_recipe_key_is_named_at_once():
    with pytest.raises(ConfigValidationError) as caught:
        OptimArgs().resolve_auto(context(defaults={"optim_args": {"aa": 1, "bb": 2}}))
    assert len(caught.value.errors) == 2


@pytest.mark.parametrize("size", [None, 0, -4, True, "32", 2.5])
def test_an_image_size_the_data_did_not_give_blames_the_data(size):
    ctx = dataclasses.replace(context(), image_size=size)
    with pytest.raises(ConfigError) as caught:
        TransformArgs().resolve_auto(ctx)
    text = str(caught.value)
    assert not isinstance(caught.value, ConfigValidationError)
    assert f"The data reports image_size {size!r}" in text
    assert "Family recipe" not in text


def test_an_explicit_image_size_does_not_need_the_data():
    ctx = dataclasses.replace(context(), image_size=None)
    assert TransformArgs(image_size=64).resolve_auto(ctx).image_size == 64


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


@pytest.mark.parametrize(
    "group, field",
    [
        ("optim_args", "weight_decay"),
        ("optim_args", "momentum"),
        ("optim_args", "beta_1"),
        ("optim_args", "beta_2"),
        ("optim_args", "epsilon"),
        ("optim_args", "clipnorm"),
        ("optim_args", "global_clipnorm"),
        ("analysis_args", "conformal_alpha"),
    ],
)
def test_the_fields_documented_as_null_for_none_accept_null_and_keep_it(group, field):
    task = validate(Task, {group: {field: None}})
    assert getattr(getattr(task, group), field) is None
    # An explicit null also survives resolution, even where the recipe has a number.
    recipe = {"optim_args": {field: 0.5}} if group == "optim_args" else FAKE_RECIPE
    kept = resolve_all(task, context(defaults={**FAKE_RECIPE, **recipe}))
    assert getattr(getattr(kept, group), field) is None


@pytest.mark.parametrize("bad", ["avg", "MAX", "", 1, True, None])
def test_watch_mode_refuses_everything_but_min_max_and_auto(bad):
    with pytest.raises(ConfigValidationError, match="metric_args.watch_mode"):
        validate(Task, {"metric_args": {"watch_mode": bad}})


@pytest.mark.parametrize("good", ["min", "max", "auto"])
def test_watch_mode_accepts_min_max_and_auto(good):
    assert validate(Task, {"metric_args": {"watch_mode": good}}).metric_args.watch_mode == good


def test_a_tensorboard_dict_with_keys_is_refused_and_null_is_accepted():
    assert LoggerArgs(tensorboard=None).tensorboard is None
    with pytest.raises(ConfigValidationError, match="logger_args.tensorboard"):
        validate(Task, {"logger_args": {"tensorboard": {"update_freq": "batch"}}})
    assert validate(Task, {"logger_args": {"tensorboard": None}}).logger_args.tensorboard is None


def test_an_empty_tensorboard_dict_is_the_accepted_default_however_it_is_given():
    # {} turns TensorBoard on; only a dict WITH keys is refused. A group built
    # from a mapping runs the check, so the empty cases must pass it.
    for given in ({"logger_args": {}}, {"logger_args": {"tensorboard": {}}}, {"logger_args": LoggerArgs()}):
        assert validate(Task, given).logger_args.tensorboard == {}
    assert LoggerArgs().problems("logger_args") == []
    assert LoggerArgs(tensorboard=None).problems("logger_args") == []


@dataclasses.dataclass(frozen=True)
class Elsewhere(StrictConfig):
    """A config that carries a LoggerArgs under a name other than ``logger_args``."""

    sinks: LoggerArgs = dataclasses.field(default_factory=LoggerArgs)
    epochs: int = 1


def test_the_tensorboard_refusal_is_collected_with_every_other_problem():
    # It used to raise from __post_init__ and hide the rest of the walk (D-034).
    with pytest.raises(ConfigValidationError) as caught:
        validate(
            Task,
            {
                "logger_args": {"tensorboard": {"x": 1}},
                "optim_args": {"lr": "abc"},
                "debug_args": {"smoke": "yes"},
            },
        )
    errors = caught.value.errors
    assert len(errors) == 3, errors
    assert any(line.startswith("Invalid value for 'logger_args.tensorboard'") for line in errors)
    assert any("'optim_args.lr'" in line for line in errors)
    assert any("'debug_args.smoke'" in line for line in errors)


def test_the_tensorboard_refusal_names_the_path_of_the_group_that_holds_it():
    with pytest.raises(ConfigValidationError) as caught:
        validate(Elsewhere, {"sinks": {"tensorboard": {"x": 1}}, "epochs": "zz"})
    errors = caught.value.errors
    assert len(errors) == 2, errors
    assert any(line.startswith("Invalid value for 'sinks.tensorboard'") for line in errors)
    assert not any("logger_args" in line for line in errors)
    with pytest.raises(ConfigValidationError) as bare:
        validate(LoggerArgs, {"tensorboard": {"x": 1}})
    assert bare.value.errors[0].startswith("Invalid value for 'tensorboard'")


def test_an_instance_whose_tensorboard_dict_was_edited_in_place_is_caught_at_validate():
    sinks = LoggerArgs()
    sinks.tensorboard["update_freq"] = "batch"
    with pytest.raises(ConfigValidationError, match="logger_args.tensorboard"):
        validate(Task, {"logger_args": sinks})


def test_the_group_fields_refuse_text_from_a_python_caller():
    with pytest.raises(ConfigValidationError) as caught:
        validate(Task, {"optim_args": {"lr": "abc"}, "metric_args": {"top_k": "5"}})
    assert caught.value.errors == (
        "Invalid value for 'optim_args.lr': expected float, got 'abc' (str)",
        "Invalid value for 'metric_args.top_k': expected int, got '5' (str)",
    )
