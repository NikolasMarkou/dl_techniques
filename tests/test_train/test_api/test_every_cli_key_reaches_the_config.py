"""Every group key typed on the command line lands in the config field it names.

The idea is the one in ``tests/test_train/_cli_contract.py``, which binds to
argparse and so cannot see a dotlist: drive the key through the real parse,
coerce and validate path, check the field holds the expected value, then drive
the same line WITHOUT the fragment and check the field differs. A key the
parser forgot to forward fails the first check; a key whose value equals the
default would pass the first check for the wrong reason, which the second
check refuses.

``Task`` is a stand-in built from ``train.api.args`` itself (``_groups.py``),
one field per group class the module defines, so an eighth group appears in it
the moment it is written and fails the completeness test until it has rows. A
hand-written ``Task`` could not notice. The real task config (Step 16) extends
this table with its own top-level keys.
"""

import dataclasses
from typing import Any, Dict, Tuple

import pytest

from train.api.cli import coerce_mapping, parse_dotlist
from train.api.config import AUTO, StrictConfig, validate

from ._groups import args_groups, task_class

Task = task_class()


#: (dotted key, command-line text, expected value). One row per group field.
ROWS: Tuple[Tuple[str, str, Any], ...] = (
    ("optim_args.optimizer", "sgd", "sgd"),
    ("optim_args.lr", "2.5e-4", 2.5e-4),
    ("optim_args.weight_decay", "0.07", 0.07),
    ("optim_args.momentum", "0.85", 0.85),
    ("optim_args.beta_1", "0.81", 0.81),
    ("optim_args.beta_2", "0.991", 0.991),
    ("optim_args.epsilon", "1e-7", 1e-7),
    ("optim_args.schedule", "constant", "constant"),
    ("optim_args.warmup_epochs", "3", 3),
    ("optim_args.warmup_start_lr", "2e-9", 2e-9),
    ("optim_args.min_lr_ratio", "0.02", 0.02),
    ("optim_args.decay_over", "post_warmup", "post_warmup"),
    ("optim_args.clipnorm", "0.6", 0.6),
    ("optim_args.global_clipnorm", "1.7", 1.7),
    ("transform_args.image_size", "48", 48),
    ("transform_args.augment", "cifar_light", "cifar_light"),
    ("transform_args.random_erasing_prob", "0.35", 0.35),
    ("transform_args.normalize", "unit", "unit"),
    ("metric_args.watch_metric", "val_loss", "val_loss"),
    ("metric_args.watch_mode", "min", "min"),
    ("metric_args.top_k", "7", 7),
    ("metric_args.success_threshold", "0.33", 0.33),
    ("checkpoint_args.save_best", "false", False),
    ("checkpoint_args.save_last", "false", False),
    ("checkpoint_args.early_stopping_patience", "11", 11),
    ("logger_args.tensorboard", "null", None),
    ("analysis_args.epoch_analysis_every", "4", 4),
    ("analysis_args.model_analysis", "false", False),
    ("analysis_args.figures", "false", False),
    ("analysis_args.conformal_alpha", "0.12", 0.12),
    ("debug_args.terminate_on_nan", "false", False),
    ("debug_args.smoke", "true", True),
)


def build(tokens) -> Task:
    _config_path, raw = parse_dotlist(tokens)
    return validate(Task, coerce_mapping(Task, raw))


def read(task: Task, key: str) -> Any:
    group, name = key.split(".")
    return getattr(getattr(task, group), name)


def every_group_key() -> set:
    return {
        f"{group.name}.{field.name}"
        for group in dataclasses.fields(Task)
        for field in dataclasses.fields(group.type)
    }


def test_the_stand_in_carries_every_group_the_args_module_defines():
    carried = [field.type for field in dataclasses.fields(Task)]
    assert carried == args_groups()
    assert len(carried) >= 7, "Phase 1 ships seven groups; a derivation that finds fewer is broken"
    assert issubclass(Task, StrictConfig)


def test_the_table_covers_every_group_field_exactly():
    keys = [row[0] for row in ROWS]
    assert len(keys) == len(set(keys)), "a key has two rows"
    assert set(keys) == every_group_key()
    assert len(keys) >= 32, "the seven groups hold 32 fields; a floor against an emptied table"


def test_the_expected_values_are_distinct_across_rows():
    # A shared value would let a key that lands in the wrong field pass.
    seen: Dict[Any, str] = {}
    for key, _text, expected in ROWS:
        if isinstance(expected, bool) or expected is None:
            continue
        assert (type(expected), expected) not in seen, (
            f"{key} repeats the value of {seen.get((type(expected), expected))}"
        )
        seen[(type(expected), expected)] = key


@pytest.mark.parametrize("key, text, expected", ROWS, ids=[row[0] for row in ROWS])
def test_the_key_reaches_its_field_and_only_its_field(key, text, expected):
    with_fragment = build([f"{key}={text}"])
    without_fragment = build([])
    assert read(with_fragment, key) == expected
    assert type(read(with_fragment, key)) is type(expected)
    assert read(without_fragment, key) != expected, (
        f"{key}={text} equals the default, so this row proves nothing"
    )
    # No other group field moved.
    for other in every_group_key() - {key}:
        assert read(with_fragment, other) == read(without_fragment, other), other


def test_every_row_at_once_reaches_all_the_fields():
    task = build([f"{key}={text}" for key, text, _ in ROWS])
    for key, _text, expected in ROWS:
        assert read(task, key) == expected, key
    assert AUTO not in {
        read(task, key) for key in every_group_key()
    }, "every AUTO field was given a value"


def test_an_unknown_group_key_gets_a_suggestion():
    from train.api.errors import ConfigValidationError

    with pytest.raises(ConfigValidationError) as caught:
        build(["optim_args.warmup_epoch=3"])
    assert caught.value.errors == (
        "Unknown key 'optim_args.warmup_epoch'. "
        "Did you mean 'optim_args.warmup_epochs'?",
    )


def test_a_bad_value_names_its_dotted_key():
    from train.api.errors import ConfigValidationError

    with pytest.raises(ConfigValidationError) as caught:
        build(["optim_args.lr=abc", "metric_args.top_k=5.0"])
    assert caught.value.errors == (
        "Invalid value for 'optim_args.lr': expected float, got 'abc' (str)",
        "Invalid value for 'metric_args.top_k': expected int, got '5.0' (str)",
    )
