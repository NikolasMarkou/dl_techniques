"""Step 24: ``task.json`` is schema version 2, conditional by task, and a version 1 record is refused.

What changed, and what must not have:

* ``SCHEMA_VERSION`` is 2. A classification record has EXACTLY the keys and key order it had at
  version 1 (``train.api.predictor.RECORD_KEYS["image_classification"]``); only the number moved.
  This is the one named exception to byte-identical classification output (D-004).
* A segmentation record carries ``input_size`` ``[height, width]`` (non-square allowed) and
  ``channels`` in place of ``input_shape``, plus ``ignore_index``, ``mask_threshold`` and
  ``output``. A key of the other task's layout is an unknown key; a missing one is named.
* A version 1 record is refused, in one sentence that names the file and says the run has to be
  trained again, at the three places a record is read: ``predict`` (``load_model``), ``export`` and
  ``resume``. There is no reader for it.

The anti-vacuity arms: the control of each refusal is the same file at version 2.
"""

import copy
import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

import train.api.recipes as recipes
from train.api import cli, registry
from train.api.errors import ConfigError
from train.api.predictor import (
    DATA_KEYS,
    OPTIONAL_RECORD_KEYS,
    RECORD_KEYS,
    SCHEMA_VERSION,
    build_segmentation_record,
    build_task_record,
    check_record,
    load_model,
    read_record,
    record_differences,
)
from train.api.tasks import register_builtin_tasks
from train.api.tasks.image_classification import train_image_classification

from tests.classification_fakes import use_tiny_raw

CLASSIFICATION = "image_classification"
SEGMENTATION = "image_segmentation"
DENOISING = "image_denoising"

#: The keys and order of a version 1 classification record, written out (not read from the module).
V1_KEYS = [
    "schema_version", "task", "model", "family", "variant", "class_names", "input_shape",
    "channels", "normalize", "mean", "std", "eval_transform",
]


def classification(**changes):
    value = build_task_record(
        task=CLASSIFICATION, model="vit/vit_pico", family="vit", variant="vit_pico",
        class_names=["a", "b", "c"], input_shape=(8, 8, 3), normalize="standardize",
        mean=[0.4, 0.5, 0.6], std=[0.2, 0.25, 0.3], data_kind="folder", source_channels=3,
    )
    value.update(changes)
    return value


def segmentation(**changes):
    value = build_segmentation_record(
        model="convunext/tiny", family="convunext", variant="tiny", class_names=["pet", "background", "border"],
        input_size=(128, 256), channels=3, normalize="unit", mean=[0.0] * 3, std=[1.0] * 3,
        ignore_index=255,
    )
    value.update(changes)
    return value


def without(value, *keys):
    value = dict(value)
    for key in keys:
        del value[key]
    return value


# ---------------------------------------------------------------------
# The schema
# ---------------------------------------------------------------------


def test_the_version_is_two_and_the_tables_are_the_three_tasks():
    assert SCHEMA_VERSION == 2
    assert sorted(RECORD_KEYS) == sorted([CLASSIFICATION, SEGMENTATION, DENOISING])
    assert sorted(DATA_KEYS) == sorted(RECORD_KEYS)
    assert OPTIONAL_RECORD_KEYS == ("variant_overrides",)


def test_a_classification_record_is_the_version_1_record_with_the_number_changed():
    record = classification()
    assert list(record) == V1_KEYS == list(RECORD_KEYS[CLASSIFICATION])
    assert record["schema_version"] == 2
    assert {**record, "schema_version": 1} == {
        "schema_version": 1, "task": CLASSIFICATION, "model": "vit/vit_pico", "family": "vit",
        "variant": "vit_pico", "class_names": ["a", "b", "c"], "input_shape": [8, 8, 3], "channels": 3,
        "normalize": "standardize", "mean": [0.4, 0.5, 0.6], "std": [0.2, 0.25, 0.3],
        "eval_transform": {"kind": "resize", "source_channels": 3},
    }


def test_a_segmentation_record_has_its_own_keys_and_a_non_square_size():
    record = segmentation()
    assert list(record) == list(RECORD_KEYS[SEGMENTATION]) == [
        "schema_version", "task", "model", "family", "variant", "class_names", "input_size", "channels",
        "normalize", "mean", "std", "ignore_index", "mask_threshold", "output",
    ]
    assert record["input_size"] == [128, 256] and record["channels"] == 3
    assert record["ignore_index"] == 255 and record["mask_threshold"] is None
    assert record["output"] == "masks" and record["task"] == SEGMENTATION
    assert "input_shape" not in record and "eval_transform" not in record


@pytest.mark.parametrize("make", [classification, segmentation], ids=[CLASSIFICATION, SEGMENTATION])
def test_each_task_round_trips_through_a_file_and_passes_its_own_check(make, tmp_path):
    record = make()
    check_record(record, "<built>")
    path = tmp_path / "task.json"
    path.write_text(json.dumps(record, indent=2))
    assert read_record(path) == record == json.loads(path.read_text())


def test_the_optional_key_rides_on_either_task():
    for make in (classification, segmentation):
        with_overrides = make(variant_overrides={"depth": 4})
        check_record(with_overrides, "<built>")
    assert "variant_overrides" not in segmentation()
    built = build_segmentation_record(
        model="convunext/tiny", family="convunext", variant="tiny", class_names=["a", "b"], input_size=[32, 32],
        channels=3, normalize="unit", mean=[0.0] * 3, std=[1.0] * 3, variant_overrides={"depth": 4},
    )
    assert built["variant_overrides"] == {"depth": 4}
    check_record(built, "<built>")


@pytest.mark.parametrize(
    "ignore_index, threshold, classes",
    [(None, None, 3), (3, None, 3), (255, None, 150), (None, 127.5, 2), (None, 0, 2)],
)
def test_a_good_pair_of_segmentation_values_is_accepted(ignore_index, threshold, classes):
    record = segmentation(
        class_names=[f"c{i}" for i in range(classes)], ignore_index=ignore_index, mask_threshold=threshold
    )
    check_record(record, "<built>")


# ---------------------------------------------------------------------
# What a segmentation record refuses
# ---------------------------------------------------------------------

BAD_SEGMENTATION = [
    (segmentation(input_size=[128]), "'input_size' must be"),
    (segmentation(input_size=[128, 0]), "'input_size' must be"),
    (segmentation(input_size=[128, 256, 3]), "'input_size' must be"),
    (segmentation(input_size=[128.0, 256]), "'input_size' must be"),
    (segmentation(input_size=[True, 256]), "'input_size' must be"),
    (segmentation(input_size=128), "'input_size' must be"),
    (segmentation(channels=0), "'channels' must be"),
    (segmentation(channels=True), "'channels' must be"),
    (segmentation(channels=1), "'mean' holds 3 numbers for 1 channels"),
    (segmentation(ignore_index=True), "'ignore_index'"),
    (segmentation(ignore_index=-1), "'ignore_index'"),
    (segmentation(ignore_index=2.0), "'ignore_index'"),
    (segmentation(ignore_index="255"), "'ignore_index'"),
    (segmentation(ignore_index=1), "outside the class indices 0..2"),
    (segmentation(ignore_index=2), "outside the class indices 0..2"),
    (segmentation(mask_threshold=127), "'mask_threshold' needs exactly 2 classes"),
    (segmentation(class_names=["a", "b"], ignore_index=255, mask_threshold=127), "'mask_threshold' and 'ignore_index'"),
    (segmentation(class_names=["a", "b"], ignore_index=None, mask_threshold=True), "'mask_threshold' must be"),
    (segmentation(class_names=["a", "b"], ignore_index=None, mask_threshold=float("nan")), "'mask_threshold' must be"),
    (segmentation(class_names=["a", "b"], ignore_index=None, mask_threshold="127"), "'mask_threshold' must be"),
    (segmentation(output="images"), "'output' must be one of masks, logits"),
    (segmentation(output=None), "'output' must be one of masks, logits"),
    (segmentation(class_names=[]), "class_names"),
    (segmentation(class_names=["a", "a", "b"]), "'class_names' repeats 'a'"),
    (segmentation(family="accunet"), "'model' is 'convunext/tiny' but 'family' and 'variant' give 'accunet/tiny'"),
    (segmentation(normalize="unit", mean=[0.1, 0.1, 0.1]), "unit means zeros and ones"),
    (segmentation(std=[0.0, 0.2, 0.2]), "'std' must be positive"),
    (segmentation(mean=[1e30, 0.5, 0.5]), r"'mean' must lie in \[0, 1\]"),
    (segmentation(variant_overrides={}), "'variant_overrides' must be a non-empty object"),
    (without(segmentation(), "input_size"), "missing key.*input_size"),
    (without(segmentation(), "output"), "missing key.*output"),
    (without(segmentation(), "ignore_index", "mask_threshold"), "missing key.*ignore_index.*mask_threshold"),
    (segmentation(extra=1), "unknown key.*extra"),
    ({**segmentation(), "input_shape": [8, 8, 3]}, "unknown key.*input_shape"),
    ({**segmentation(), "eval_transform": {"kind": "resize", "source_channels": 3}}, "unknown key.*eval_transform"),
]


@pytest.mark.parametrize("record, message", BAD_SEGMENTATION, ids=[m for _, m in BAD_SEGMENTATION])
def test_a_bad_segmentation_record_is_refused_naming_the_key(record, message):
    with pytest.raises(ConfigError, match=message):
        check_record(record, "task.json")


def test_a_classification_record_does_not_take_the_segmentation_keys_and_the_reverse():
    with pytest.raises(ConfigError, match="unknown key.*output"):
        check_record(classification(output="masks"), "task.json")
    with pytest.raises(ConfigError, match="unknown key.*input_size"):
        check_record(classification(input_size=[8, 8]), "task.json")
    with pytest.raises(ConfigError, match=r"missing key.*input_shape.*eval_transform"):
        check_record({**without(segmentation(), "input_size", "ignore_index", "mask_threshold", "output"),
                      "task": CLASSIFICATION}, "task.json")


# A name no task will ever take (the task this literal used to be, image_denoising, is a real task
# by now); the tasks the reader names are read from the table, never typed here.
@pytest.mark.parametrize("task", ["a_task_no_one_will_take", "", None, 3, ["image_classification"]])
def test_a_task_this_reader_does_not_know_is_named_and_never_crashes(task):
    knows = re.escape(", ".join(RECORD_KEYS))
    with pytest.raises(ConfigError, match=rf"'task' is .*; this reader knows {knows}"):
        check_record(classification(task=task), "task.json")


# ---------------------------------------------------------------------
# What a resume compares
# ---------------------------------------------------------------------


def test_a_resume_compares_the_segmentation_data_keys_and_not_the_output():
    earlier = segmentation()
    assert record_differences(earlier, copy.deepcopy(earlier)) == []
    assert record_differences(earlier, segmentation(output="logits")) == []
    cases = {
        "input_size": [128, 128],
        "ignore_index": None,
        "mask_threshold": 3.0,
        "class_names": ["pet", "border", "background"],
        "normalize": "standardize",
        "mean": [0.5, 0.5, 0.5],
    }
    for key, value in cases.items():
        lines = record_differences(earlier, segmentation(**{key: value}))
        assert [line.partition(":")[0] for line in lines][:1] == [key], (key, lines)


def test_a_resume_that_changes_the_task_is_told_so_and_nothing_else():
    lines = record_differences(classification(), segmentation())
    assert lines == [f'task: "{CLASSIFICATION}" -> "{SEGMENTATION}"']


def test_the_classification_comparison_is_the_one_it_was():
    earlier = classification()
    assert record_differences(earlier, copy.deepcopy(earlier)) == []
    assert record_differences(earlier, classification(input_shape=[16, 16, 3], channels=3))[0].startswith("input_shape")
    assert record_differences(earlier, classification(eval_transform={"kind": "resize_centre_crop", "source_channels": 3}))[0].startswith("eval_transform")


# ---------------------------------------------------------------------
# A version 1 record is refused at predict, export and resume
# ---------------------------------------------------------------------

READ_ONCE = {
    "model": "vit/vit_pico",
    "data": {"dataset": "cifar10"},
    "epochs": 1,
    "batch_size": 4,
    "analysis_args": {"model_analysis": False},
    "metric_args": {"success_threshold": 0.0},
}


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        use_tiny_raw(patch, n_train=40, n_test=12)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


@pytest.fixture(scope="module")
def run_dir(world, tmp_path_factory):
    out = tmp_path_factory.mktemp("trained") / "run"
    result = train_image_classification(out=str(out), **READ_ONCE)
    assert result.success
    return out


def version_one(run_dir, destination):
    """A copy of the finished run whose ``task.json`` says version 1 (and is otherwise the same)."""
    shutil.copytree(run_dir, destination)
    path = destination / "task.json"
    record = json.loads(path.read_text())
    assert record["schema_version"] == 2  # what the trainer writes now
    record["schema_version"] = 1
    path.write_text(json.dumps(record, indent=2))
    return destination


def sha_of_tree(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file() and p.name != "run.lock"
    }


def test_a_run_writes_a_version_2_classification_record(run_dir):
    record = json.loads((run_dir / "task.json").read_text())
    assert record["schema_version"] == 2 and list(record) == V1_KEYS
    check_record(record, "task.json")


def test_predict_refuses_a_version_1_run_naming_the_file_and_the_retraining(run_dir, tmp_path):
    old = version_one(run_dir, tmp_path / "old")
    with pytest.raises(ConfigError) as caught:
        load_model(old)
    text = str(caught.value)
    assert str(old / "task.json") in text
    assert "'schema_version' is 1" in text and "understands 2 only" in text
    assert "cannot upgrade it" in text and "trained again" in text
    assert text.count(". ") == 0 and "\n" not in text  # one sentence
    # the control: the same directory at version 2 loads
    assert load_model(run_dir).class_names


def test_the_predict_command_exits_2_with_that_one_line(run_dir, tmp_path, capsys):
    old = version_one(run_dir, tmp_path / "old")
    pictures = tmp_path / "pictures"
    pictures.mkdir()
    from PIL import Image

    Image.new("RGB", (32, 32), (10, 20, 30)).save(pictures / "a.png")
    code = cli.main(["predict", f"model={old}", f"data={pictures}", f"out={tmp_path / 'answers.json'}"])
    err = capsys.readouterr().err
    assert code == 2 and "Traceback" not in err
    lines = [line for line in err.splitlines() if line.startswith("error:")]
    assert len(lines) == 1 and "trained again" in lines[0] and str(old / "task.json") in lines[0]
    assert not (tmp_path / "answers.json").exists()


def test_export_refuses_a_version_1_run_and_writes_nothing(run_dir, tmp_path, capsys):
    old = version_one(run_dir, tmp_path / "old")
    out = tmp_path / "exported.keras"
    code = cli.main(["export", f"model={old}", f"out={out}", "format=keras"])
    err = capsys.readouterr().err
    assert code == 2 and "'schema_version' is 1" in err and "trained again" in err
    assert not out.exists() and not Path(str(out) + ".task.json").exists()
    # the control: version 2 exports
    good = tmp_path / "good.keras"
    assert cli.main(["export", f"model={run_dir}", f"out={good}", "format=keras"]) == 0
    assert good.is_file() and json.loads(Path(str(good) + ".task.json").read_text())["schema_version"] == 2


def test_the_export_function_refuses_a_record_of_another_version_and_writes_nothing(run_dir, tmp_path):
    import keras

    from train.api.exporting import export

    model = keras.Sequential([keras.layers.Input((4,)), keras.layers.Dense(2)])
    stale = {**json.loads((run_dir / "task.json").read_text()), "schema_version": 1}
    out = tmp_path / "m.keras"
    with pytest.raises(ConfigError, match="'schema_version' is 1.*trained again"):
        export(model=model, out=out, format="keras", record=stale)
    assert not out.exists() and not Path(str(out) + ".task.json").exists()


def as_interrupted(directory):
    """Make a finished run's directory look like a killed one to ``open_run``: a backup directory and no summary.

    The resume itself would find nothing in that empty backup and say so; the version check comes
    before the sidecar is read, which is what the tests below observe.
    """
    from train.api.run_dir import BACKUP_DIR
    from train.common.run_artifacts import RESULTS_SUMMARY_NAME

    (directory / BACKUP_DIR).mkdir(parents=True, exist_ok=True)
    (directory / RESULTS_SUMMARY_NAME).unlink()
    return directory


def test_resume_refuses_a_version_1_run_before_the_data_and_changes_no_file(world, run_dir, tmp_path):
    old = as_interrupted(version_one(run_dir, tmp_path / "old"))
    before = sha_of_tree(old)
    with pytest.raises(ConfigError) as caught:
        train_image_classification(out=str(old), resume_interrupted=True, **READ_ONCE)
    text = str(caught.value)
    assert text.startswith(f"Cannot resume {old}")
    assert str(old / "task.json") in text and "'schema_version' 1" in text and "trained again" in text
    assert "\n" not in text
    assert sha_of_tree(old) == before


def test_the_version_is_checked_before_the_sidecar_is_read(world, run_dir, tmp_path):
    """A run directory that lost its sidecar AND is version 1 is told about the version first."""
    from train.common.resume_state import FIT_STATE_NAME

    old = as_interrupted(version_one(run_dir, tmp_path / "old"))
    (old / "checkpoints" / FIT_STATE_NAME).unlink()
    with pytest.raises(ConfigError, match=r"Cannot resume .*'schema_version' 1.*trained again"):
        train_image_classification(out=str(old), resume_interrupted=True, **READ_ONCE)
    # the control: at version 2 the same directory is refused for the sidecar it lacks
    current = as_interrupted(shutil.copytree(run_dir, tmp_path / "current"))
    (current / "checkpoints" / FIT_STATE_NAME).unlink()
    with pytest.raises(ConfigError) as caught:
        train_image_classification(out=str(current), resume_interrupted=True, **READ_ONCE)
    assert "schema_version" not in str(caught.value)


def test_the_same_directory_at_version_2_resumes(world, run_dir, tmp_path):
    """The control: the sidecar of the finished run says its fit returned, so the resume only reports."""
    current = as_interrupted(shutil.copytree(run_dir, tmp_path / "current"))
    result = train_image_classification(out=str(current), resume_interrupted=True, **READ_ONCE)
    assert result.success
    assert json.loads((current / "task.json").read_text())["schema_version"] == 2


def test_a_resume_whose_task_json_is_another_tasks_is_refused_as_changed_data(world, run_dir, tmp_path):
    """Since version 2 the task word is part of what a resume compares (it was never compared before)."""
    other = as_interrupted(shutil.copytree(run_dir, tmp_path / "other"))
    earlier = json.loads((other / "task.json").read_text())
    planted = build_segmentation_record(
        model=earlier["model"], family=earlier["family"], variant=earlier["variant"],
        class_names=earlier["class_names"], input_size=(32, 32), channels=3, normalize="unit",
        mean=[0.0] * 3, std=[1.0] * 3,
    )
    (other / "task.json").write_text(json.dumps(planted, indent=2))
    with pytest.raises(ConfigError, match=r"data is not the data the run started with.*task: \"image_segmentation\" -> \"image_classification\""):
        train_image_classification(out=str(other), resume_interrupted=True, **READ_ONCE)


def test_a_version_1_check_does_not_fire_on_a_run_that_has_no_record_or_a_current_one(tmp_path):
    from train.api.predictor import refuse_old_record

    refuse_old_record(tmp_path / "missing.json", "Cannot resume x")  # nothing to read: not this check's business
    current = tmp_path / "task.json"
    current.write_text(json.dumps(classification()))
    refuse_old_record(current, "Cannot resume x")
    current.write_text("{not json")
    refuse_old_record(current, "Cannot resume x")  # unreadable is named by read_record, not here
    current.write_text(json.dumps({**classification(), "schema_version": 3}))
    refuse_old_record(current, "Cannot resume x")  # a NEWER record is read_record's refusal
    current.write_text(json.dumps({**classification(), "schema_version": 1}))
    with pytest.raises(ConfigError, match="Cannot resume x"):
        refuse_old_record(current, "Cannot resume x")


def test_other_version_values_keep_the_plain_refusal():
    for value, shown in ((3, "3"), (True, "True"), ("1", "'1'"), (0, "0")):
        with pytest.raises(ConfigError) as caught:
            check_record(classification(schema_version=value), "task.json")
        text = str(caught.value)
        assert f"'schema_version' is {shown}; this reader understands 2 only" in text
        assert "trained again" not in text
    with pytest.raises(ConfigError, match="schema_version"):
        check_record(without(classification(), "schema_version"), "task.json")


def test_a_predictor_is_not_built_for_a_task_it_does_not_serve():
    """A segmentation record would fail at the first prediction on a missing ``eval_transform``; it is refused here."""
    import keras

    from train.api.predictor import Predictor

    model = keras.Sequential([keras.layers.Input((4,)), keras.layers.Dense(3)])
    with pytest.raises(ConfigError, match="the task is 'image_segmentation' but this predictor serves 'image_classification' only"):
        Predictor(model, segmentation())
    assert Predictor(model, classification(input_shape=[4, 4, 3])).class_names == ["a", "b", "c"]


def test_a_record_of_version_1_with_every_key_valid_is_still_refused_there_is_no_reader():
    """D-004: the old layout is not read, however well-formed (nothing else in the file is wrong)."""
    check_record(classification(), "task.json")
    with pytest.raises(ConfigError, match="'schema_version' is 1, an older layout"):
        check_record(classification(schema_version=1), "task.json")
