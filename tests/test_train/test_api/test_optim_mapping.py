"""``OptimArgs`` reaches the optimizer and the schedule, field by field.

Every arm reads the BUILT object. The mapping has silent traps: the builder
ignores a literal ``clipnorm`` key, ignores ``warmup_epochs`` and ``min_lr_ratio``
as names, and gives AdamW 0.004 of weight decay when no value is passed. A test
over the config dict would pass through all three; a test over the optimizer
cannot.

The last-step learning rates are checked twice: against literals measured from
the built schedule, and against the cosine formula written out here.
"""

import dataclasses
import math
from typing import Any, Mapping

import pytest

from train.api.args import AutoContext, OptimArgs
from train.api.errors import ConfigValidationError
from train.api.optim import build_optimizer_and_schedule

PEAK = 3e-4
STEPS_PER_EPOCH = 351

#: A resolved adamw recipe, the vit shape: global clip, cosine over the whole run.
ADAMW = OptimArgs(
    optimizer="adamw",
    lr=PEAK,
    weight_decay=0.05,
    momentum=None,
    beta_1=None,
    beta_2=None,
    epsilon=None,
    schedule="cosine_decay",
    warmup_epochs=5,
    warmup_start_lr=1e-8,
    min_lr_ratio=0.01,
    decay_over="total",
    clipnorm=None,
    global_clipnorm=1.0,
)

#: A resolved sgd recipe, the resnet shape.
SGD = dataclasses.replace(
    ADAMW, optimizer="sgd", lr=0.1, weight_decay=1e-4, momentum=0.9, global_clipnorm=None
)


def adamw(**changes: Any) -> OptimArgs:
    return dataclasses.replace(ADAMW, **changes)


def sgd(**changes: Any) -> OptimArgs:
    return dataclasses.replace(SGD, **changes)


def build(args: OptimArgs, epochs: int = 10):
    return build_optimizer_and_schedule(args, STEPS_PER_EPOCH, epochs)


def cosine(step: int, steps: int, alpha: float = 0.01) -> float:
    """The cosine-decay formula, written out, for a step in ``[0, steps]``."""
    progress = min(step, steps) / steps
    return PEAK * ((1 - alpha) * 0.5 * (1 + math.cos(math.pi * progress)) + alpha)


# ---------------------------------------------------------------------
# Clipping: the silent drop
# ---------------------------------------------------------------------


def test_clipnorm_clips_each_variable_and_not_the_whole():
    optimizer, _ = build(adamw(clipnorm=1.0, global_clipnorm=None))
    assert optimizer.clipnorm == 1.0
    assert optimizer.global_clipnorm is None


def test_global_clipnorm_clips_the_whole_and_not_each_variable():
    optimizer, _ = build(adamw(clipnorm=None, global_clipnorm=2.0))
    assert optimizer.global_clipnorm == 2.0
    assert optimizer.clipnorm is None


def test_no_clip_gives_neither():
    optimizer, _ = build(adamw(clipnorm=None, global_clipnorm=None))
    assert optimizer.clipnorm is None
    assert optimizer.global_clipnorm is None


# ---------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "epochs, warmup, decay_over, literal",
    [
        (50, 5, "total", 1.0276e-05),
        (10, 5, "total", 1.5163e-04),
        (10, 5, "post_warmup", 3.0002e-06),
    ],
)
def test_the_last_step_learning_rate(epochs, warmup, decay_over, literal):
    _, schedule = build(adamw(warmup_epochs=warmup, decay_over=decay_over), epochs)
    last = STEPS_PER_EPOCH * epochs - 1
    warmup_steps = warmup * STEPS_PER_EPOCH
    length = STEPS_PER_EPOCH * epochs - (warmup_steps if decay_over == "post_warmup" else 0)
    assert float(schedule(last)) == pytest.approx(literal, rel=1e-3)
    assert float(schedule(last)) == pytest.approx(cosine(last - warmup_steps, length), rel=1e-4)


def test_the_two_decay_lengths_differ_by_about_fifty_times_at_ten_epochs():
    """The regime the parity runs sit in; if the arms ever agree, one is not wired."""
    _, whole = build(adamw(decay_over="total"), 10)
    _, after = build(adamw(decay_over="post_warmup"), 10)
    last = STEPS_PER_EPOCH * 10 - 1
    assert float(whole(last)) / float(after(last)) > 40


def test_warmup_starts_at_the_start_rate_and_reaches_the_peak_after_its_steps():
    _, schedule = build(adamw(warmup_epochs=2, warmup_start_lr=1e-6))
    assert float(schedule(0)) == pytest.approx(1e-6, rel=1e-4)
    assert float(schedule(2 * STEPS_PER_EPOCH)) == pytest.approx(PEAK, rel=1e-4)
    assert float(schedule(STEPS_PER_EPOCH)) == pytest.approx((1e-6 + PEAK) / 2, rel=1e-3)


def test_min_lr_ratio_is_the_floor_of_the_cosine():
    _, schedule = build(adamw(warmup_epochs=0, min_lr_ratio=0.2, decay_over="total"), 4)
    assert float(schedule(4 * STEPS_PER_EPOCH)) == pytest.approx(PEAK * 0.2, rel=1e-4)


def test_zero_warmup_starts_at_the_peak():
    _, schedule = build(adamw(warmup_epochs=0))
    assert float(schedule(0)) == pytest.approx(PEAK, rel=1e-6)


def test_post_warmup_decay_is_refused_when_nothing_is_left_after_warmup():
    with pytest.raises(ConfigValidationError, match="post_warmup"):
        build(adamw(warmup_epochs=10, decay_over="post_warmup"), 10)


def test_total_decay_allows_a_warmup_as_long_as_the_run():
    """vit trains inside its warmup in legacy; the recipe says so with ``total``."""
    _, schedule = build(adamw(warmup_epochs=5, decay_over="total"), 3)
    assert float(schedule(3 * STEPS_PER_EPOCH - 1)) < PEAK


def test_a_constant_schedule_is_a_float_the_optimizer_holds():
    optimizer, schedule = build(adamw(schedule="constant", warmup_epochs=0, lr=2e-3))
    assert isinstance(schedule, float) and schedule == 2e-3
    assert float(optimizer.learning_rate) == pytest.approx(2e-3)


def test_a_constant_schedule_with_a_warmup_is_refused_not_silently_dropped():
    with pytest.raises(ConfigValidationError, match="optim_args.warmup_epochs"):
        build(adamw(schedule="constant", warmup_epochs=3))


# ---------------------------------------------------------------------
# The optimizer
# ---------------------------------------------------------------------


def test_adamw_takes_the_weight_decay_and_the_betas():
    optimizer, _ = build(adamw(weight_decay=0.07, beta_1=0.8, beta_2=0.99, epsilon=1e-6))
    assert type(optimizer).__name__ == "AdamW"
    assert optimizer.weight_decay == 0.07
    assert (optimizer.beta_1, optimizer.beta_2, optimizer.epsilon) == (0.8, 0.99, 1e-6)


def test_adamw_with_no_weight_decay_gets_none_and_not_the_builders_0_004():
    optimizer, _ = build(adamw(weight_decay=None))
    assert optimizer.weight_decay == 0.0


def test_adam_takes_the_betas_and_has_no_decay():
    optimizer, _ = build(adamw(optimizer="adam", weight_decay=None, beta_1=0.85))
    assert type(optimizer).__name__ == "Adam"
    assert optimizer.beta_1 == 0.85
    assert optimizer.weight_decay is None


def test_sgd_takes_the_momentum_and_never_the_weight_decay():
    """Its ``weight_decay`` is the L2 strength on the model; the optimizer gets none."""
    optimizer, _ = build(sgd())
    assert type(optimizer).__name__ == "SGD"
    assert optimizer.momentum == 0.9
    assert optimizer.weight_decay is None


def test_an_optimizer_name_is_read_without_case_or_padding():
    optimizer, _ = build(adamw(optimizer=" AdamW "))
    assert type(optimizer).__name__ == "AdamW"


# ---------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes, field",
    [
        ({"optimizer": "lion"}, "optim_args.optimizer"),
        ({"schedule": "linear"}, "optim_args.schedule"),
        ({"decay_over": "never"}, "optim_args.decay_over"),
        ({"lr": 0.0}, "optim_args.lr"),
        ({"lr": -1.0}, "optim_args.lr"),
        ({"lr": float("nan")}, "optim_args.lr"),
        ({"lr": float("inf")}, "optim_args.lr"),
        ({"lr": True}, "optim_args.lr"),
        ({"warmup_epochs": -1}, "optim_args.warmup_epochs"),
        ({"warmup_start_lr": -1e-8}, "optim_args.warmup_start_lr"),
        ({"min_lr_ratio": 1.5}, "optim_args.min_lr_ratio"),
        ({"min_lr_ratio": -0.1}, "optim_args.min_lr_ratio"),
        ({"weight_decay": -0.1}, "optim_args.weight_decay"),
        ({"global_clipnorm": 0.0}, "optim_args.global_clipnorm"),
        ({"global_clipnorm": -1.0}, "optim_args.global_clipnorm"),
        ({"epsilon": 0.0}, "optim_args.epsilon"),
        ({"beta_1": 1.0}, "optim_args.beta_1"),
        ({"momentum": 0.9}, "optim_args.momentum"),
    ],
)
def test_a_value_the_optimizer_cannot_use_is_refused_naming_its_key(changes, field):
    with pytest.raises(ConfigValidationError) as caught:
        build(adamw(**changes))
    assert f"'{field}'" in str(caught.value)


def test_both_clips_together_are_refused():
    with pytest.raises(ConfigValidationError) as caught:
        build(adamw(clipnorm=0.5, global_clipnorm=1.0))
    assert "clipnorm" in str(caught.value) and "not both" in str(caught.value)


@pytest.mark.parametrize("option", ["beta_1", "beta_2", "epsilon"])
def test_sgd_with_an_adam_option_is_refused(option):
    with pytest.raises(ConfigValidationError, match=f"optim_args.{option}"):
        build(sgd(**{option: 0.5 if option != "epsilon" else 1e-8}))


def test_adam_with_a_weight_decay_is_refused():
    with pytest.raises(ConfigValidationError, match="optim_args.weight_decay"):
        build(adamw(optimizer="adam", weight_decay=0.01))


def test_adam_with_a_momentum_is_refused():
    with pytest.raises(ConfigValidationError, match="optim_args.momentum"):
        build(adamw(optimizer="adam", weight_decay=None, momentum=0.9))


def test_every_problem_is_reported_at_once():
    with pytest.raises(ConfigValidationError) as caught:
        build(adamw(lr=-1.0, schedule="linear", momentum=0.9, clipnorm=1.0))
    assert len(caught.value.errors) >= 4
    text = str(caught.value)
    for field in ("lr", "schedule", "momentum", "clipnorm"):
        assert f"optim_args.{field}" in text


def test_an_unresolved_field_is_refused_by_path():
    with pytest.raises(ConfigValidationError) as caught:
        build(OptimArgs(lr=1e-3))
    text = str(caught.value)
    assert "optim_args.optimizer" in text and "optim_args.warmup_epochs" in text
    assert "optim_args.lr" not in text


@pytest.mark.parametrize("steps, epochs", [(0, 10), (-1, 10), (351, 0), (351, True), (3.5, 10)])
def test_the_run_length_must_be_positive_ints(steps, epochs):
    with pytest.raises(ConfigValidationError):
        build_optimizer_and_schedule(ADAMW, steps, epochs)


# ---------------------------------------------------------------------
# Switching the optimizer under a recipe (D-028's open interaction)
# ---------------------------------------------------------------------

ADAMW_RECIPE: Mapping[str, Any] = {
    "optimizer": "adamw",
    "lr": 1e-3,
    "weight_decay": 0.1,
    "momentum": None,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epsilon": 1e-8,
    "schedule": "cosine_decay",
    "warmup_epochs": 0,
    "warmup_start_lr": 1e-8,
    "min_lr_ratio": 0.01,
    "decay_over": "post_warmup",
    "clipnorm": None,
    "global_clipnorm": None,
}
SGD_RECIPE: Mapping[str, Any] = {
    **ADAMW_RECIPE,
    "optimizer": "sgd",
    "lr": 0.1,
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "beta_1": None,
    "beta_2": None,
    "epsilon": None,
}


def resolve(recipe: Mapping[str, Any], **explicit: Any) -> OptimArgs:
    context = AutoContext(
        family="fake",
        variant="fake_tiny",
        dataset="cifar10",
        num_classes=10,
        image_size=32,
        defaults={"optim_args": recipe},
    )
    return OptimArgs(**explicit).resolve_auto(context)


def test_sgd_on_an_adamw_recipe_builds_and_takes_none_of_the_adamw_options():
    args = resolve(ADAMW_RECIPE, optimizer="sgd", lr=0.05)
    assert (args.beta_1, args.beta_2, args.epsilon, args.weight_decay) == (None, None, None, None)
    assert args.lr == 0.05
    assert args.warmup_epochs == ADAMW_RECIPE["warmup_epochs"], "the schedule still comes from the recipe"
    optimizer, _ = build(args)
    assert type(optimizer).__name__ == "SGD"
    assert optimizer.weight_decay is None


def test_adam_on_an_adamw_recipe_keeps_the_betas_and_drops_the_decay():
    args = resolve(ADAMW_RECIPE, optimizer="adam", lr=1e-3)
    assert (args.beta_1, args.beta_2, args.epsilon) == (0.9, 0.999, 1e-8)
    assert args.weight_decay is None
    optimizer, _ = build(args)
    assert (optimizer.beta_1, optimizer.epsilon) == (0.9, 1e-8)


def test_adamw_on_an_sgd_recipe_takes_neither_the_momentum_nor_the_l2_strength():
    """The recipe's ``weight_decay`` is an L2 strength there, not decoupled decay."""
    args = resolve(SGD_RECIPE, optimizer="adamw", lr=1e-3)
    assert args.momentum is None
    assert args.weight_decay is None
    optimizer, _ = build(args)
    assert optimizer.weight_decay == 0.0


def test_the_recipe_own_optimizer_keeps_every_recipe_option():
    args = resolve(ADAMW_RECIPE, optimizer="adamw")
    assert (args.beta_1, args.weight_decay) == (0.9, 0.1)
    assert resolve(SGD_RECIPE).momentum == 0.9
    assert resolve(SGD_RECIPE).weight_decay == 1e-4


def test_an_explicit_option_survives_the_switch_and_is_then_refused_by_the_builder():
    args = resolve(ADAMW_RECIPE, optimizer="sgd", lr=0.05, beta_1=0.8)
    assert args.beta_1 == 0.8
    with pytest.raises(ConfigValidationError, match="optim_args.beta_1"):
        build(args)


def test_an_explicit_sgd_weight_decay_survives_and_is_the_l2_strength():
    args = resolve(ADAMW_RECIPE, optimizer="sgd", lr=0.05, weight_decay=5e-4, momentum=0.8)
    assert (args.weight_decay, args.momentum) == (5e-4, 0.8)
    build(args)


def test_a_constant_schedule_over_a_warmup_recipe_resolves_to_no_warmup():
    recipe = {**ADAMW_RECIPE, "warmup_epochs": 5}
    args = resolve(recipe, schedule="constant")
    assert args.warmup_epochs == 0
    optimizer, schedule = build(args)
    assert schedule == args.lr


def test_an_explicit_warmup_with_a_constant_schedule_survives_resolution_and_is_refused():
    recipe = {**ADAMW_RECIPE, "warmup_epochs": 5}
    args = resolve(recipe, schedule="constant", warmup_epochs=2)
    assert args.warmup_epochs == 2
    with pytest.raises(ConfigValidationError, match="optim_args.warmup_epochs"):
        build(args)
