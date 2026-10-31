"""Every key of the image-classification task, typed on the command line, lands in the field it names.

The idea is the one in ``tests/test_train/_cli_contract.py``, which binds to
argparse and so cannot see a dotlist. Each row of the table is driven through the
real ``cli.main`` into a stand-in task function that takes the same config class
as the real task, and is checked at two places:

1. **the validated config** holds the expected value, of the expected type, and
   no other key moved (the CLI door: parse, coerce, validate);
2. **the resolved config**, after ``TaskTrainer._resolve`` (the code the real run
   uses to fill ``"auto"``), still holds it: an explicit value survives the
   recipe, even where the recipe holds another (D-010).

The same line WITHOUT the fragment is driven too, and the field must differ from
the expected value at both places, so a row whose value equals the default or
the recipe's proves nothing and fails.

The task config is the real ``ImageClassificationConfig`` (Step 16); the seven
groups come from ``train.api.args`` through ``_groups.args_groups`` and must all
be in it, so an eighth group fails the completeness test until it has rows.
"""

import dataclasses
import typing
from types import SimpleNamespace
from typing import Any, Dict, List, NamedTuple, Tuple

import pytest

import train.api.recipes as recipes
from train.api import cli, registry
from train.api.cli import coerce_mapping, parse_dotlist
from train.api.config import AUTO, StrictConfig, validate
from train.api.errors import ConfigValidationError
from train.api.tasks.image_classification import (
    ImageClassificationConfig as Config,
    ImageClassificationTrainer,
)

from ._groups import args_groups

#: The tokens every line starts with; a row's fragment may replace one of them.
BASE: Dict[str, str] = {
    "out": "base_out",
    "model": "vit/vit_pico",
    "data.dataset": "cifar10",
}

CAPTURE_TASK = "capture_classification"
CAPTURED: List[Tuple[Config, Config]] = []


class Row(NamedTuple):
    key: str
    text: str
    expected: Any
    companions: Dict[str, str] = {}


IMAGENET = {"data.dataset": "imagenet"}
IMAGENETTE = {"data.dataset": "imagenette"}

#: One row per leaf key of the task config. A row's ``companions`` are tokens the
#: key needs to be legal (an imagenet option needs imagenet); they are in BOTH lines.
ROWS: Tuple[Row, ...] = (
    Row("out", "other_out", "other_out"),
    Row("model", "vit/vit_tiny", "vit/vit_tiny"),
    Row("epochs", "6", 6),
    Row("batch_size", "9", 9),
    Row("seed", "13", 13),
    Row("devices", "1", 1),
    Row("precision", "mixed_float16", "mixed_float16"),
    Row("resume_interrupted", "true", True),
    Row("checkpoint", "weights.keras", "weights.keras"),
    Row("model_args", '{"dropout_rate": 0.3}', {"dropout_rate": 0.3}),
    Row("data.dataset", "cifar100", "cifar100"),
    Row("data.validation_split", "0.2", 0.2),
    Row("data.holdout_per_class", "5", 5, IMAGENET),
    Row("data.max_samples", "100", 100),
    Row("data.eval_samples", "20", 20, IMAGENETTE),
    Row("data.imagenet_root", "/data/imagenet", "/data/imagenet", IMAGENET),
    Row("data.devkit_dir", "/data/devkit", "/data/devkit", IMAGENET),
    # A switch away from the recipe's optimizer needs an explicit lr (D-045).
    Row("optim_args.optimizer", "sgd", "sgd", {"optim_args.lr": "0.05"}),
    Row("optim_args.lr", "2.5e-4", 2.5e-4),
    Row("optim_args.weight_decay", "0.07", 0.07),
    Row(
        "optim_args.momentum",
        "0.85",
        0.85,
        {"optim_args.optimizer": "sgd", "optim_args.lr": "0.05"},
    ),
    Row("optim_args.beta_1", "0.81", 0.81),
    Row("optim_args.beta_2", "0.991", 0.991),
    Row("optim_args.epsilon", "1e-7", 1e-7),
    Row("optim_args.schedule", "constant", "constant"),
    Row("optim_args.warmup_epochs", "3", 3),
    Row("optim_args.warmup_start_lr", "2e-9", 2e-9),
    Row("optim_args.min_lr_ratio", "0.02", 0.02),
    Row("optim_args.decay_over", "post_warmup", "post_warmup"),
    Row("optim_args.clipnorm", "0.6", 0.6),
    Row("optim_args.global_clipnorm", "1.7", 1.7),
    Row("transform_args.image_size", "48", 48, IMAGENETTE),
    Row("transform_args.augment", "flip", "flip"),
    # The probability belongs to one preset (D-042): the pair is legal, either alone is not.
    Row("transform_args.random_erasing_prob", "0.35", 0.35, {"transform_args.augment": "cifar_autoaugment"}),
    Row("transform_args.normalize", "unit", "unit"),
    Row("metric_args.watch_metric", "val_loss", "val_loss"),
    Row("metric_args.watch_mode", "min", "min"),
    Row("metric_args.top_k", "7", 7),
    Row("metric_args.success_threshold", "0.33", 0.33),
    Row("checkpoint_args.save_best", "false", False),
    Row("checkpoint_args.save_last", "false", False),
    Row("checkpoint_args.early_stopping_patience", "11", 11),
    Row("logger_args.tensorboard", "null", None),
    Row("analysis_args.epoch_analysis_every", "4", 4),
    Row("analysis_args.model_analysis", "false", False),
    Row("analysis_args.figures", "false", False),
    Row("analysis_args.conformal_alpha", "0.12", 0.12),
    Row("debug_args.terminate_on_nan", "false", False),
    Row("debug_args.smoke", "true", True),
)

#: Keys with a single legal value: no second value exists to drive, so no row can differ.
SINGLE_VALUED = ("data.format",)


def leaf_keys(cls: type, prefix: str = "") -> List[str]:
    """Every dotted leaf key of a config class: groups are walked, anything else is a leaf."""
    hints = typing.get_type_hints(cls)
    keys = []
    for field in dataclasses.fields(cls):
        annotation = hints[field.name]
        key = f"{prefix}.{field.name}" if prefix else field.name
        if typing.get_origin(annotation) is typing.Union and all(
            isinstance(m, type) and issubclass(m, StrictConfig) for m in typing.get_args(annotation)
        ):
            # A union of groups (``data``) is walked as its FIRST member, the default
            # format (``builtin``); the folder and csv keys are driven by
            # ``test_folder_and_csv_data`` (Step 25) and get rows here in Step 28.
            annotation = typing.get_args(annotation)[0]
        if isinstance(annotation, type) and issubclass(annotation, StrictConfig):
            keys += leaf_keys(annotation, key)
        else:
            keys.append(key)
    return keys


def read(config: Any, key: str) -> Any:
    for part in key.split("."):
        config = getattr(config, part)
    return config


def tokens(row: Row, with_fragment: bool) -> List[str]:
    merged = {**BASE, **row.companions}
    if with_fragment:
        merged[row.key] = row.text
    return [f"{key}={value}" for key, value in merged.items()]


def capture(**given: Any) -> SimpleNamespace:
    """The stand-in task function: validate as the real one does, resolve as the real run does."""
    config = validate(Config, given)
    trainer = ImageClassificationTrainer(config)
    family, variant = registry.parse_model_name(config.model)
    resolved, _context = trainer._resolve(trainer.describe_data(config), family, variant)
    CAPTURED.append((config, resolved))
    return SimpleNamespace(success=True)


@pytest.fixture(autouse=True)
def three_visible_gpus(monkeypatch):
    """``devices`` is checked against ``CUDA_VISIBLE_DEVICES`` before the run is claimed.

    Nothing here touches TensorFlow's devices (the stand-in task function only
    resolves), so a list of three entries is enough for the row ``devices=1``.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")


@pytest.fixture(scope="module", autouse=True)
def world():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        registry.register_task(
            registry.TaskSpec(
                name=CAPTURE_TASK,
                config_class="train.api.tasks.image_classification:ImageClassificationConfig",
                function=f"{__name__}:capture",
                summary="validates and resolves the task config, trains nothing",
            )
        )
        yield


def drive(arguments: List[str]) -> Tuple[Config, Config]:
    """Run one line through ``cli.main`` and return the validated and the resolved config."""
    CAPTURED.clear()
    assert cli.main([CAPTURE_TASK, *arguments]) == 0
    assert len(CAPTURED) == 1, "the stand-in task function was not reached exactly once"
    return CAPTURED[0]


def build(arguments: List[str]) -> Config:
    """The CLI door alone: parse, coerce, validate. No resolution."""
    _config_path, raw = parse_dotlist(arguments)
    return validate(Config, coerce_mapping(Config, raw))


# ---------------------------------------------------------------------
# The table covers the config
# ---------------------------------------------------------------------


def test_the_config_carries_every_group_the_args_module_defines():
    hints = typing.get_type_hints(Config)
    carried = [hints[field.name] for field in dataclasses.fields(Config)]
    assert [group for group in carried if group in args_groups()] == sorted(
        args_groups(), key=lambda group: carried.index(group)
    )
    assert set(args_groups()) <= set(carried)
    assert len(args_groups()) >= 7, "Phase 1 ships seven groups; a derivation that finds fewer is broken"


def test_the_table_covers_every_leaf_key_of_the_task_config_exactly():
    keys = [row.key for row in ROWS]
    assert len(keys) == len(set(keys)), "a key has two rows"
    assert set(keys) | set(SINGLE_VALUED) == set(leaf_keys(Config))
    assert not set(keys) & set(SINGLE_VALUED)
    assert len(leaf_keys(Config)) >= 48, "32 group keys, 10 top-level keys, 8 data keys"


def test_a_single_valued_key_has_exactly_one_legal_value():
    members = typing.get_args(typing.get_type_hints(Config)["data"])
    assert [typing.get_args(typing.get_type_hints(m)["format"]) for m in members] == [
        ("builtin",),
        ("folder",),
        ("csv",),
    ]


def test_the_companions_are_legal_tokens_of_the_base_line():
    for row in ROWS:
        assert set(row.companions) <= set(leaf_keys(Config)), row.key
        assert row.key not in row.companions, row.key


def test_the_expected_values_are_distinct_across_rows():
    # A shared value would let a key that lands in the wrong field pass.
    seen: Dict[Any, str] = {}
    for row in ROWS:
        if isinstance(row.expected, bool) or row.expected is None:
            continue
        identity = (type(row.expected), repr(row.expected))
        assert identity not in seen, f"{row.key} repeats the value of {seen.get(identity)}"
        seen[identity] = row.key


# ---------------------------------------------------------------------
# Each key, through cli.main
# ---------------------------------------------------------------------


@pytest.mark.parametrize("row", ROWS, ids=[row.key for row in ROWS])
def test_the_key_reaches_its_field_and_only_its_field(row):
    config, _ = drive(tokens(row, with_fragment=True))
    control, _ = drive(tokens(row, with_fragment=False))
    assert read(config, row.key) == row.expected
    assert type(read(config, row.key)) is type(row.expected)
    assert read(control, row.key) != row.expected, (
        f"{row.key}={row.text} equals the default, so this row proves nothing"
    )
    for other in set(leaf_keys(Config)) - {row.key}:
        assert read(config, other) == read(control, other), other


@pytest.mark.parametrize("row", ROWS, ids=[row.key for row in ROWS])
def test_the_key_survives_resolve_auto(row):
    _, resolved = drive(tokens(row, with_fragment=True))
    _, control = drive(tokens(row, with_fragment=False))
    assert read(resolved, row.key) == row.expected, (
        f"{row.key}={row.text} was replaced when the config was resolved"
    )
    assert type(read(resolved, row.key)) is type(row.expected)
    assert read(control, row.key) != row.expected, (
        f"the recipe already holds {row.expected!r} for {row.key}, so surviving proves nothing"
    )
    # Resolution leaves nothing automatic behind.
    assert AUTO not in [read(resolved, key) for key in leaf_keys(Config)]


def test_every_group_row_at_once_reaches_all_the_fields():
    rows = [row for row in ROWS if "." in row.key and not row.key.startswith("data.")]
    task = build([f"{key}={value}" for key, value in BASE.items()] + [f"{r.key}={r.text}" for r in rows])
    for row in rows:
        assert read(task, row.key) == row.expected, row.key


# ---------------------------------------------------------------------
# The lines a bad key prints
# ---------------------------------------------------------------------


def base_tokens() -> List[str]:
    return [f"{key}={value}" for key, value in BASE.items()]


def test_an_unknown_group_key_gets_a_suggestion():
    with pytest.raises(ConfigValidationError) as caught:
        build(base_tokens() + ["optim_args.warmup_epoch=3"])
    assert caught.value.errors == (
        "Unknown key 'optim_args.warmup_epoch'. "
        "Did you mean 'optim_args.warmup_epochs'?",
    )


def test_an_unknown_top_level_key_gets_a_suggestion():
    with pytest.raises(ConfigValidationError) as caught:
        build(base_tokens() + ["batch_sise=3"])
    assert caught.value.errors == ("Unknown key 'batch_sise'. Did you mean 'batch_size'?",)


def test_a_bad_value_names_its_dotted_key():
    with pytest.raises(ConfigValidationError) as caught:
        build(base_tokens() + ["optim_args.lr=abc", "metric_args.top_k=5.0"])
    assert caught.value.errors == (
        "Invalid value for 'optim_args.lr': expected float, got 'abc' (str)",
        "Invalid value for 'metric_args.top_k': expected int, got '5.0' (str)",
    )
