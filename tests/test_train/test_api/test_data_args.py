"""``BuiltinImageData``: the options it accepts, the facts it gives, the loader it calls.

Four things are pinned:

* the dataset list the config carries equals the loader's table (the config
  cannot import it, so a test holds the copy);
* every option that does not apply to the chosen dataset is refused with the
  dotted key, and the same option on a dataset it applies to is accepted
  (the control that makes the refusal row mean something);
* ``builtin_facts`` answers the class count and size without loading, and
  refuses to resize an in-memory dataset;
* ``load_builtin_data`` hands the loader exactly the arguments that apply, and
  reads the recipe's ``loader`` mapping (``channels`` for mnist).
"""

import pytest

import train.common.classification_data as classification_data
from dl_techniques.datasets.vision.classification import CLASSIFICATION_DATASETS
from train.api import data_args
from train.api.config import validate
from train.api.data_args import (
    DATASETS,
    BuiltinImageData,
    builtin_facts,
    load_builtin_data,
    load_data_config,
)
from train.api.errors import ConfigError, ConfigValidationError

from tests.classification_fakes import use_tiny_raw

# ---------------------------------------------------------------------
# The dataset list
# ---------------------------------------------------------------------


def test_the_dataset_list_is_the_loaders_table():
    assert len(DATASETS) == len(set(DATASETS))
    assert set(DATASETS) == set(CLASSIFICATION_DATASETS)


def test_the_in_memory_list_is_the_loaders_in_memory_datasets():
    in_memory = {name for name, spec in CLASSIFICATION_DATASETS.items() if spec.in_memory}
    assert set(data_args.IN_MEMORY_DATASETS) == in_memory


def test_an_unknown_dataset_is_refused_with_the_choices():
    with pytest.raises(ConfigValidationError, match="cifar10"):
        validate(BuiltinImageData, {"dataset": "cifar11"})


def test_the_dataset_is_required():
    with pytest.raises(ConfigValidationError, match="Missing required key 'dataset'"):
        validate(BuiltinImageData, {})


# ---------------------------------------------------------------------
# What does not apply is refused
# ---------------------------------------------------------------------

#: (option, value, a dataset it applies to, a dataset that refuses it)
OPTIONS = [
    ("validation_split", 0.2, "cifar10", "imagenet"),
    ("holdout_per_class", 5, "imagenet", "cifar10"),
    ("eval_samples", 100, "imagenette", "mnist"),
    ("eval_samples", 100, "imagenet", "cifar100"),
    ("imagenet_root", "/data/imagenet", "imagenet", "imagenette"),
    ("devkit_dir", "/data/devkit", "imagenet", "cifar10"),
]


@pytest.mark.parametrize("option, value, applies, refuses", OPTIONS)
def test_an_option_is_accepted_where_it_applies_and_refused_elsewhere(
    option, value, applies, refuses
):
    accepted = validate(BuiltinImageData, {"dataset": applies, option: value})
    assert getattr(accepted, option) == value
    with pytest.raises(ConfigValidationError) as raised:
        validate(BuiltinImageData, {"dataset": refuses, option: value})
    assert f"'{option}'" in str(raised.value)


def test_every_problem_is_reported_at_once():
    with pytest.raises(ConfigValidationError) as raised:
        validate(
            BuiltinImageData,
            {"dataset": "cifar10", "holdout_per_class": 5, "devkit_dir": "x", "eval_samples": 3},
        )
    assert len(raised.value.errors) == 3


@pytest.mark.parametrize(
    "mapping",
    [
        {"dataset": "cifar10", "validation_split": 0.0},
        {"dataset": "cifar10", "validation_split": 1.0},
        {"dataset": "cifar10", "max_samples": 0},
        {"dataset": "imagenet", "holdout_per_class": 0},
        {"dataset": "imagenette", "eval_samples": -1},
    ],
)
def test_an_out_of_range_number_is_refused(mapping):
    with pytest.raises(ConfigValidationError):
        validate(BuiltinImageData, mapping)


def test_the_error_lines_carry_the_group_prefix_inside_a_task():
    # `validate` of an enclosing config passes the prefix; the group names its key as the user typed it.
    lines = BuiltinImageData(dataset="cifar10", devkit_dir="x").problems("data")
    assert len(lines) == 1 and "'data.devkit_dir'" in lines[0]


# ---------------------------------------------------------------------
# load_data_config
# ---------------------------------------------------------------------


def test_a_mapping_without_format_is_builtin():
    data = load_data_config({"dataset": "mnist"})
    assert data == BuiltinImageData(dataset="mnist")
    assert data.format == "builtin"


def test_an_instance_is_returned_as_it_is():
    data = BuiltinImageData(dataset="cifar100", max_samples=64)
    assert load_data_config(data) is data


def test_an_unknown_format_is_refused_naming_the_choices():
    with pytest.raises(ConfigError, match="'data.format'.*builtin"):
        load_data_config({"dataset": "mnist", "format": "folder"})


@pytest.mark.parametrize("value", [3, ["cifar10"], None])
def test_something_that_is_no_mapping_or_path_is_refused(value):
    with pytest.raises(ConfigError, match="'data'"):
        load_data_config(value)


def test_a_yaml_file_is_loaded_and_its_errors_name_the_file(tmp_path):
    good = tmp_path / "data.yaml"
    good.write_text("dataset: cifar10\nmax_samples: 64\n")
    assert load_data_config(good) == BuiltinImageData(dataset="cifar10", max_samples=64)
    assert load_data_config(str(good)) == load_data_config(good)
    bad = tmp_path / "bad.yaml"
    bad.write_text("dataset: cifar10\nholdout_per_class: 5\n")
    with pytest.raises(ConfigValidationError, match="bad.yaml.*holdout_per_class"):
        load_data_config(bad)
    with pytest.raises(ConfigError, match="Cannot read"):
        load_data_config(tmp_path / "missing.yaml")


# ---------------------------------------------------------------------
# builtin_facts
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "dataset, size, expected",
    [
        ("mnist", None, (10, 28)),
        ("cifar10", None, (10, 32)),
        ("cifar100", 32, (100, 32)),
        ("imagenette", None, (10, 224)),
        ("imagenet", None, (1000, 224)),
        ("imagenet", 160, (1000, 160)),
        ("imagenette", 128, (10, 128)),
    ],
)
def test_the_facts_come_from_the_table_without_loading(dataset, size, expected, monkeypatch):
    def refuse(name):
        raise AssertionError("builtin_facts must not read any data")

    monkeypatch.setattr(classification_data, "_load_raw", refuse)
    facts = builtin_facts(BuiltinImageData(dataset=dataset), size)
    assert (facts.num_classes, facts.image_size) == expected
    assert facts.dataset == dataset


@pytest.mark.parametrize("dataset", ["mnist", "cifar10", "cifar100"])
def test_an_in_memory_dataset_is_not_resized(dataset):
    with pytest.raises(ConfigError, match="transform_args.image_size.*not resized"):
        builtin_facts(BuiltinImageData(dataset=dataset), 64)


# ---------------------------------------------------------------------
# load_builtin_data
# ---------------------------------------------------------------------


def load(data, loader=None, **overrides):
    options = dict(batch_size=4, seed=1, augment=None, normalize="standardize", image_size=32)
    options.update(overrides)
    return load_builtin_data(data, loader=loader, **options)


def test_the_recipe_loader_mapping_reaches_the_loader(monkeypatch):
    use_tiny_raw(monkeypatch, n_train=40, n_test=8)
    native = load(BuiltinImageData(dataset="mnist"), image_size=28)
    repeated = load(BuiltinImageData(dataset="mnist"), {"channels": 3}, image_size=28)
    assert native.input_shape == (28, 28, 1)
    assert repeated.input_shape == (28, 28, 3)
    assert repeated.mean.shape == (3,)


def test_an_unknown_loader_key_is_a_recipe_error(monkeypatch):
    use_tiny_raw(monkeypatch)
    with pytest.raises(ConfigError, match="Family recipe for 'loader'.*'chanels'"):
        load(BuiltinImageData(dataset="mnist"), {"chanels": 3})


def test_max_samples_and_validation_split_change_the_splits(monkeypatch):
    use_tiny_raw(monkeypatch, n_train=40, n_test=8)
    base = load(BuiltinImageData(dataset="cifar10"))
    capped = load(BuiltinImageData(dataset="cifar10", max_samples=20))
    wide = load(BuiltinImageData(dataset="cifar10", validation_split=0.5))
    assert capped.n_train < base.n_train
    assert wide.n_select > base.n_select
    assert (base.n_train, base.n_select) != (wide.n_train, wide.n_select)


def test_the_loader_gets_only_the_arguments_that_apply(monkeypatch):
    calls = []

    def record(name, **kwargs):
        calls.append((name, kwargs))
        return "data"

    monkeypatch.setattr(classification_data, "load_classification_data", record)
    load(BuiltinImageData(dataset="cifar10", validation_split=0.3, max_samples=9))
    load(
        BuiltinImageData(
            dataset="imagenet", holdout_per_class=7, imagenet_root="/r", devkit_dir="/d",
            eval_samples=11,
        ),
        {"imagenet_augment_color": True},
        image_size=160,
    )
    load(BuiltinImageData(dataset="imagenette"), image_size=128)
    (_, memory), (_, tree), (_, records) = calls
    assert memory["validation_split"] == 0.3 and memory["max_samples"] == 9
    assert not {"image_size", "eval_samples", "holdout_per_class", "imagenet_root"} & set(memory)
    assert tree["holdout_per_class"] == 7
    assert (tree["imagenet_root"], tree["devkit_dir"]) == ("/r", "/d")
    assert (tree["image_size"], tree["eval_samples"]) == (160, 11)
    assert tree["imagenet_augment_color"] is True
    assert "validation_split" not in tree
    assert records["image_size"] == 128
    assert records["eval_samples"] == classification_data.DEFAULT_EVAL_SAMPLES
    assert "holdout_per_class" not in records and records["validation_split"] is None
