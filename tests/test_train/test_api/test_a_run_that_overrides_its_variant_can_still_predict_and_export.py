"""A run whose ``model_args`` override keys of its variant's table can be loaded, predicted with and exported.

Step 14.7. D-028 made ``load_model`` refuse a model that differs from the variant its ``task.json``
names, and accepted that a run trained with ``model_args.<variant-table key>=...`` could then neither
``predict`` nor ``export``. Measured end to end (``scratch_step14e/e2e_before_clean.jsonl``): on
resnet, convnext_v1, convnext_v2, cliffordnet and cliffordnet_hier such a run trained, then both
commands exited 2. The run now records the keys it changed in an optional ``variant_overrides`` key of
``task.json`` (written only when non-empty), ``export`` carries it into ``<out>.task.json``, and
``check_model_against_record`` compares the model with the variant's table UPDATED by them.

* per family, a REAL tiny CPU run with one override trains, records the override, loads, predicts, and
  exports to a file that loads and predicts the same;
* the same run's files are refused under a wrong variant (with the override recorded), under a record
  that lacks the override, under an override the model does not have, and under a key the variant has no;
* a run with no override writes no ``variant_overrides`` and its record keys are exactly ``RECORD_KEYS``;
* a ViT's ``scale`` cannot be overridden (its factory raises ``TypeError``): refused before the claim.

Everything writes under ``tmp_path``.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import train.api.recipes as recipes
from train.api import registry
from train.api.errors import ConfigError
from train.api.exporting import export
from train.api.predictor import (
    OPTIONAL_RECORD_KEYS,
    RECORD_KEYS,
    check_record,
    load_model,
    overrides_of_variant,
)
from train.api.tasks.image_classification import train_image_classification

#: family -> (model name, model_args, a variant of the family whose table differs from the smallest
#: variant in a key that the override does NOT touch, or None).
CASES = {
    "resnet": ("resnet/resnet18", {"blocks_per_stage": [1, 1, 1, 1]}, "resnet50"),
    "resnet_two_keys": ("resnet/resnet18", {"blocks_per_stage": [1, 1, 1, 1], "filters_per_stage": [16, 32, 64, 128]}, "resnet50"),
    "convnext_v1": ("convnext_v1/cifar10", {"depths": [1, 1]}, "tiny"),
    "convnext_v2": ("convnext_v2/atto", {"depths": [1, 1, 1, 1]}, "femto"),
    "cliffordnet": ("cliffordnet/nano", {"depth": 2}, "lite"),
    "cliffordnet_hier": ("cliffordnet_hier/hier_345", {"stage_depths": [1, 1, 1]}, None),
}
CLASSES = ["ant", "moth", "zebra"]


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        yield


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("tree")
    rng = np.random.default_rng(3)
    for k, name in enumerate(CLASSES):
        for index in range(6):
            path = root / name / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((rng.integers(0, 120, (32, 32, 3)) + 40 * k).astype(np.uint8)).save(path)
    return root


def train(out, tree, model, model_args):
    return train_image_classification(
        out=str(out), model=model, data={"format": "folder", "train_dir": str(tree)}, epochs=1, batch_size=4,
        transform_args={"image_size": 32}, analysis_args={"model_analysis": False, "figures": False},
        metric_args={"success_threshold": 0.0}, model_args=model_args,
    )


@pytest.fixture(scope="module", params=sorted(CASES))
def run(request, world, tree, tmp_path_factory):
    """One trained run per case: ``(name, run dir, model_args, wrong variant)``."""
    model, model_args, wrong = CASES[request.param]
    out = tmp_path_factory.mktemp(request.param) / "run"
    train(out, tree, model, model_args)
    return request.param, out, model_args, wrong


def record_of(directory):
    return json.loads((Path(directory) / "task.json").read_text())


def copy_with(run_dir, destination, **changes):
    """The run's best model with a ``task.json`` edited by ``changes`` (a value of ``DROP`` removes the key)."""
    destination.mkdir(parents=True)
    shutil.copy(run_dir / "best_model.keras", destination / "best_model.keras")
    record = record_of(run_dir)
    for key, value in changes.items():
        if value is DROP:
            record.pop(key, None)
        else:
            record[key] = value
    (destination / "task.json").write_text(json.dumps(record))
    return destination


DROP = object()


def test_the_run_records_exactly_the_keys_it_overrode(run):
    name, out, model_args, _ = run
    record = record_of(out)
    assert record["variant_overrides"] == model_args
    check_record(record, out)  # the record the trainer wrote is valid
    assert list(record)[: len(RECORD_KEYS["image_classification"])] == list(RECORD_KEYS["image_classification"])
    assert list(record)[len(RECORD_KEYS["image_classification"]):] == ["variant_overrides"]


def test_a_run_that_overrides_its_variant_predicts_and_exports(run, tree, tmp_path):
    name, out, model_args, _ = run
    image = str(tree / "ant" / "0.png")
    predictor = load_model(out)  # was a ConfigError naming the variant before step 14.7
    answer = predictor.predict(image)
    assert sorted(answer["labels"]) == CLASSES
    target = export(model=predictor.model, out=tmp_path / "exported.keras", format="keras",
                    input_shape=predictor.record["input_shape"], record=predictor.record)
    sibling = json.loads(Path(f"{target}.task.json").read_text())
    assert sibling["variant_overrides"] == model_args  # carried into <out>.task.json
    again = load_model(target).predict(image)
    assert again["labels"] == answer["labels"]
    np.testing.assert_allclose(again["scores"], answer["scores"], atol=1e-5)


def test_a_wrong_variant_is_refused_even_with_the_overrides_recorded(run, tmp_path):
    name, out, model_args, wrong = run
    if wrong is None:
        pytest.skip("this family has one variant")
    family = record_of(out)["family"]
    directory = copy_with(out, tmp_path / "wrong", variant=wrong, model=f"{family}/{wrong}")
    with pytest.raises(ConfigError, match=rf"names the variant '{wrong}' but .* is not that variant .*with the recorded overrides"):
        load_model(directory)


def test_a_record_without_the_override_is_refused_naming_the_key(run, tmp_path):
    name, out, model_args, _ = run
    directory = copy_with(out, tmp_path / "bare", variant_overrides=DROP)
    key = next(iter(model_args))
    with pytest.raises(ConfigError, match=rf"is not that variant \({key} is "):
        load_model(directory)


def test_an_override_the_model_does_not_have_is_refused(run, tmp_path):
    name, out, model_args, _ = run
    key, value = next(iter(model_args.items()))
    other = [v + 1 for v in value] if isinstance(value, list) else value + 1
    directory = copy_with(out, tmp_path / "lying", variant_overrides={**model_args, key: other})
    with pytest.raises(ConfigError, match=rf"is not that variant .*{key} is "):
        load_model(directory)
    foreign = copy_with(out, tmp_path / "foreign", variant_overrides={**model_args, "no_such_key": 1})
    with pytest.raises(ConfigError, match=r"variant_overrides names 'no_such_key', which .* does not have"):
        load_model(foreign)


def test_a_run_with_no_override_writes_no_variant_overrides_key(world, tree, tmp_path):
    train(tmp_path / "run", tree, "resnet/resnet18", {})
    record = record_of(tmp_path / "run")
    assert list(record) == list(RECORD_KEYS["image_classification"])
    assert "variant_overrides" not in record
    # Restating a table value, or setting a key that is not in the table, is not an override either.
    train(tmp_path / "again", tree, "resnet/resnet18", {"blocks_per_stage": [2, 2, 2, 2], "stem_type": "cifar"})
    assert list(record_of(tmp_path / "again")) == list(RECORD_KEYS["image_classification"])


def test_a_vit_scale_is_refused_before_the_directory_is_claimed(world, tree, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match=r"'scale' is fixed by the variant 'vit_pico' of 'vit' and cannot be overridden") as caught:
        train(out, tree, "vit/vit_pico", {"scale": "tiny"})
    assert "'out' is now used" not in str(caught.value)
    assert not out.exists()


# ---------------------------------------------------------------------
# The helper and the schema
# ---------------------------------------------------------------------


def test_overrides_of_variant_keeps_only_table_keys_with_another_value():
    row = {"depth": 12, "shifts": [1, 2], "cli_mode": "full"}
    assert overrides_of_variant(row, {"depth": 2, "dropout_rate": 0.1}) == {"depth": 2}
    assert overrides_of_variant(row, {"depth": 12, "shifts": (1, 2)}) == {}  # restated, a tuple for a list
    assert overrides_of_variant(row, {"shifts": (1, 3), "cli_mode": "inner"}) == {"shifts": [1, 3], "cli_mode": "inner"}
    assert list(overrides_of_variant(row, {"cli_mode": "x", "depth": 1})) == ["depth", "cli_mode"]  # the table's order
    assert overrides_of_variant(row, {}) == {}


def a_record(**changes):
    from train.api.predictor import build_task_record

    base = build_task_record(
        task="image_classification", model="resnet/resnet18", family="resnet", variant="resnet18", class_names=CLASSES,
        input_shape=(32, 32, 3), normalize="unit", mean=[0.0] * 3, std=[1.0] * 3, data_kind="folder", source_channels=3,
    )
    base.update(changes)
    return base


def test_the_optional_key_is_in_the_schema_and_written_only_when_there_is_something_to_write():
    from train.api.predictor import build_task_record

    assert OPTIONAL_RECORD_KEYS == ("variant_overrides",)
    assert "variant_overrides" not in a_record()
    for empty in (None, {}):
        assert "variant_overrides" not in build_task_record(
            task="t", model="a/b", family="a", variant="b", class_names=["x"], input_shape=(8, 8, 3), normalize="unit",
            mean=[0, 0, 0], std=[1, 1, 1], data_kind="folder", source_channels=3, variant_overrides=empty)
    check_record(a_record(variant_overrides={"blocks_per_stage": [1, 1, 1, 1], "block_type": "basic", "scale": None}), "x")


@pytest.mark.parametrize("bad", [
    {}, [], "depth", 3, None, {"depth": float("nan")}, {"depth": float("inf")}, {"depth": object()}, {"": 1},
    {"depth": {"inner": {1, 2}}}, {"depth": [1, object()]},
])
def test_a_malformed_variant_overrides_is_a_config_error(bad):
    with pytest.raises(ConfigError, match="'variant_overrides' must be a non-empty object of text keys and JSON values"):
        check_record(a_record(variant_overrides=bad), "task.json")


def test_an_unknown_key_is_still_refused_beside_the_optional_one():
    with pytest.raises(ConfigError, match="unknown key.*extra"):
        check_record(a_record(variant_overrides={"depth": 1}, extra=1), "task.json")


class Reports:
    """A stand-in model: ``get_config`` reports ``kept`` and nothing else."""

    def __init__(self, kept):
        self.kept = kept

    def get_config(self):
        return dict(self.kept)


class Owner(Reports):
    VARIANTS = {"small": {"stages": [1, 2], "width": 8}}


def test_an_override_of_a_key_the_model_does_not_report_is_refused(tmp_path):
    """The key is in the variant's table, so it is not foreign; the model's config simply lacks it, so
    nothing says the override is true of it and it is refused rather than trusted."""
    from train.api.predictor import check_model_against_record

    spec = registry.ModelSpec(
        family="owner", factory=f"{__name__}:Owner", variants_attr=f"{__name__}:Owner.VARIANTS",
        tasks=("image_classification",),
    )
    files = (tmp_path / "m.keras", tmp_path / "task.json")
    record = a_record(family="owner", variant="small", model="owner/small", variant_overrides={"width": 16})
    with pytest.raises(ConfigError, match=r"names 'width' but .* does not report that key"):
        check_model_against_record(Owner({"stages": [1, 2]}), record, spec, *files)
    # Control: the model that reports it with the recorded value passes; with another value it does not.
    check_model_against_record(Owner({"stages": [1, 2], "width": 16}), record, spec, *files)
    with pytest.raises(ConfigError, match=r"width is 8 in the model, 16 in 'small' with the recorded overrides"):
        check_model_against_record(Owner({"stages": [1, 2], "width": 8}), record, spec, *files)
