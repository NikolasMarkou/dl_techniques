"""A data file and the command line's dotted keys are two layers of ONE group (plan 98b866fb 7.1, D-021).

``data=data.yaml data.max_samples=100`` used to be refused as a repeated key, so a capped or tweaked
run needed a copy of the data file. The rule, for EVERY task and every group that takes a file
(``cli.path_form_fields``): the lowest layer is the ``config=`` file, then a group's file
(``data: data.yaml`` in the config file, then ``data=data.yaml`` on the command line, which replaces the
config file's whole group), then the dotted keys, which win key by key; the group's ``format`` that the
dotted keys do not state is the lower layers'. A key given twice on the command line itself stays refused,
and so does a file together with a JSON object for the same group.

What this file pins, through the real ``cli.main`` into a stand-in function that takes the real task config:
the merge for both tasks, the precedence, the format taken from the file (a csv file and ``data.image_column=``,
which the first member of the union would refuse), the config file's own ``data: path`` surviving a dotted
key (the old merge dropped it), the replacement of a config file's group by a command-line file (the replaced
file is never read), the refusals, and that the result equals the explicit merged copy (what a user used to
write by hand, so ``config.json`` is the same).
"""

import dataclasses
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from PIL import Image

from train.api import cli, registry
from train.api.cli import FileLayer, parse_dotlist, path_form_fields
from train.api.errors import ConfigValidationError
from train.api.tasks.image_classification import ImageClassificationConfig
from train.api.tasks.image_segmentation import ImageSegmentationConfig

CLASSIFICATION = "capture_layered_classification"
SEGMENTATION = "capture_layered_segmentation"
SEEN: List[Dict[str, Any]] = []


def _capture(**kwargs: Any) -> Any:
    SEEN.append(kwargs)
    return dataclasses.make_dataclass("Result", [("success", bool)])(True)


@pytest.fixture(autouse=True)
def tasks(monkeypatch):
    SEEN.clear()
    specs = {
        CLASSIFICATION: ImageClassificationConfig,
        SEGMENTATION: ImageSegmentationConfig,
    }
    monkeypatch.setattr(
        registry,
        "TASKS",
        {
            **registry.TASKS,
            **{
                name: registry.TaskSpec(
                    name=name,
                    config_class=f"{cls.__module__}:{cls.__name__}",
                    function=f"{__name__}:_capture",
                    summary="records its keywords",
                )
                for name, cls in specs.items()
            },
        },
    )


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> Dict[str, Path]:
    """Small real files: a class-per-folder tree and csv listings of images and masks."""
    root = tmp_path_factory.mktemp("layers")
    rng = np.random.default_rng(0)
    out: Dict[str, Path] = {"root": root}
    for klass in ("ant", "moth"):
        for index in range(2):
            path = root / "tree" / klass / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 255, size=(20, 30, 3), dtype=np.uint8)).save(path)
    pairs = []
    for index in range(4):
        image, mask = root / "seg" / f"i{index}.png", root / "seg" / f"m{index}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, size=(20, 30, 3), dtype=np.uint8)).save(image)
        Image.fromarray(rng.integers(0, 2, size=(20, 30), dtype=np.uint8)).save(mask)
        pairs.append((image, mask))
    (root / "seg.csv").write_text(
        "pic,lab\n" + "".join(f"{i},{m}\n" for i, m in pairs)
    )
    out["tree"] = root / "tree"
    out["seg_csv"] = root / "seg.csv"
    return out


def write(path: Path, text: str) -> str:
    path.write_text(text)
    return str(path)


def run(task: str, *tokens: str) -> int:
    return cli.main([task, *tokens])


BASE_C = ("out=x", "model=vit/vit_pico")
BASE_S = ("out=x", "model=convunext/tiny", "transform_args.image_size=32")


# ---------------------------------------------------------------------
# The parser: what is a layer and what is a repeated key
# ---------------------------------------------------------------------


def test_a_file_and_dotted_keys_make_one_layered_group_in_either_order():
    for tokens in (["data=a.yaml", "data.max_samples=5"], ["data.max_samples=5", "data=a.yaml"]):
        _, tree = parse_dotlist(tokens, ["data"])
        assert isinstance(tree["data"], FileLayer)
        assert tree["data"].file == "a.yaml"
        assert dict(tree["data"]) == {"max_samples": "5"}
    # A file alone is a layer with nothing over it.
    _, alone = parse_dotlist(["data=a.yaml"], ["data"])
    assert alone["data"].file == "a.yaml" and dict(alone["data"]) == {}


@pytest.mark.parametrize(
    "tokens, wording",
    [
        (["data=a.yaml", "data=b.yaml"], "Repeated key 'data'"),
        (["data=a.yaml", 'data={"dataset": "cifar10"}'], "Repeated key 'data'"),
        (['data={"dataset": "cifar10"}', "data=a.yaml"], "Repeated key 'data'"),
        (["data=a.yaml", "data.max_samples=1", "data.max_samples=2"], "Repeated key 'data.max_samples'"),
        (['data={"dataset": "cifar10"}', "data.max_samples=1"], "Repeated key 'data'"),
        (["data.max_samples=1", 'data={"dataset": "cifar10"}'], "Repeated key 'data'"),
    ],
)
def test_a_key_given_twice_on_the_command_line_stays_refused(tokens, wording):
    with pytest.raises(ConfigValidationError) as caught:
        parse_dotlist(tokens, ["data"])
    assert any(wording in line for line in caught.value.errors), caught.value.errors


def test_a_caller_with_no_file_group_still_refuses_a_value_and_sub_keys():
    for tokens in (["data=a.yaml", "data.max_samples=5"], ["data.max_samples=5", "data=a.yaml"]):
        with pytest.raises(ConfigValidationError) as caught:
            parse_dotlist(tokens)
        assert any("Repeated key 'data'" in line for line in caught.value.errors)


def test_only_the_path_form_groups_take_a_file():
    assert [f.name for f in path_form_fields(ImageClassificationConfig)] == ["data"]
    assert [f.name for f in path_form_fields(ImageSegmentationConfig)] == ["data"]
    # `optim_args=file.yaml` is no file form: it is a bare word where a group is, and stays refused.
    with pytest.raises(ConfigValidationError):
        parse_dotlist(["optim_args=file.yaml", "optim_args.lr=1"], ["data"])


# ---------------------------------------------------------------------
# Both tasks, through cli.main
# ---------------------------------------------------------------------


def test_classification_a_file_with_a_cap_equals_the_file_with_the_cap_written_in(files, tmp_path):
    layered = write(tmp_path / "a.yaml", "dataset: cifar100\nvalidation_split: 0.2\n")
    merged = write(tmp_path / "b.yaml", "dataset: cifar100\nvalidation_split: 0.2\nmax_samples: 64\n")
    assert run(CLASSIFICATION, *BASE_C, f"data={layered}", "data.max_samples=64") == 0
    assert run(CLASSIFICATION, *BASE_C, "data.max_samples=64", f"data={layered}") == 0
    assert run(CLASSIFICATION, *BASE_C, f"data={merged}") == 0
    assert SEEN[0] == SEEN[1] == SEEN[2]
    assert SEEN[0]["data"] == {"dataset": "cifar100", "validation_split": 0.2, "max_samples": 64}


def test_segmentation_a_file_with_a_cap_equals_the_file_with_the_cap_written_in(files, tmp_path):
    base = (
        f"format: csv\ntrain_csv: {files['seg_csv']}\nimage_column: pic\nmask_column: lab\n"
        "class_names: [a, b]\n"
    )
    layered = write(tmp_path / "a.yaml", base)
    merged = write(tmp_path / "b.yaml", base + "max_samples: 4\nmask_threshold: 1.5\n")
    assert run(SEGMENTATION, *BASE_S, f"data={layered}", "data.max_samples=4", "data.mask_threshold=1.5") == 0
    assert run(SEGMENTATION, *BASE_S, f"data={merged}") == 0
    assert SEEN[0] == SEEN[1]
    assert SEEN[0]["data"]["max_samples"] == 4 and SEEN[0]["data"]["format"] == "csv"


def test_the_command_line_wins_over_the_file_key_by_key(files, tmp_path):
    data_file = write(tmp_path / "d.yaml", "dataset: cifar10\nmax_samples: 5\nvalidation_split: 0.3\n")
    assert run(CLASSIFICATION, *BASE_C, f"data={data_file}", "data.max_samples=7") == 0
    assert SEEN[0]["data"] == {"dataset": "cifar10", "max_samples": 7, "validation_split": 0.3}


def test_the_format_of_the_file_decides_how_the_dotted_keys_are_read(files, tmp_path):
    # `data.path_column` is a key of the csv members only. Read as the FIRST member (builtin, the
    # default format) it is an unknown key; read as the file's format it is a column name.
    data_file = write(
        tmp_path / "c.yaml",
        f"format: csv\ntrain_csv: {files['seg_csv']}\npath_column: pic\nlabel_column: lab\n",
    )
    assert run(CLASSIFICATION, *BASE_C, f"data={data_file}", "data.path_column=other", "transform_args.image_size=24") == 0
    assert SEEN[0]["data"]["format"] == "csv" and SEEN[0]["data"]["path_column"] == "other"
    # A folder file and a folder-only key.
    folder = write(tmp_path / "f.yaml", f"format: folder\ntrain_dir: {files['tree']}\n")
    other = files["root"] / "elsewhere"
    other.mkdir(exist_ok=True)
    assert run(CLASSIFICATION, *BASE_C, f"data={folder}", f"data.test_dir={other}", "transform_args.image_size=24") == 0
    assert SEEN[1]["data"]["test_dir"] == str(other) and SEEN[1]["data"]["train_dir"] == str(files["tree"])


def test_a_dotted_format_over_a_file_of_another_format_is_refused_naming_both_formats(files, tmp_path, capsys):
    # The merge would keep the file's keys under the new format and validation would then refuse one of
    # them as unknown ("Unknown key 'data.train_dir'"), a key the user never typed (review-iter-1 concern 7).
    folder = write(tmp_path / "f.yaml", f"format: folder\ntrain_dir: {files['tree']}\n")
    assert run(CLASSIFICATION, *BASE_C, f"data={folder}", "data.format=csv") == 2
    error = capsys.readouterr().err
    assert "data.format=csv is given over a folder data" in error
    assert "train_dir" not in error and "Unknown key" not in error
    assert not SEEN


def test_the_same_refusal_for_segmentation_a_csv_file_under_a_folder_format_and_a_config_files_mapping(files, tmp_path, capsys):
    csv_file = write(
        tmp_path / "c.yaml", f"format: csv\ntrain_csv: {files['seg_csv']}\nimage_column: pic\nmask_column: lab\n")
    tokens = ("data.format=folder", f"data.train_images={files['root']}", f"data.train_masks={files['root']}")
    assert run(SEGMENTATION, *BASE_S, f"data={csv_file}", *tokens) == 2
    error = capsys.readouterr().err
    assert "data.format=folder is given over a csv data" in error
    assert "train_csv" not in error and "Unknown key" not in error
    config_file = write(
        tmp_path / "m.yaml", f"out: x\nmodel: convunext/tiny\ndata:\n  format: csv\n  train_csv: {files['seg_csv']}\n")
    assert run(SEGMENTATION, f"config={config_file}", *tokens) == 2
    assert "data.format=folder is given over a csv data" in capsys.readouterr().err
    # The same format restated, and a format over a file that states none, are not this refusal.
    assert run(SEGMENTATION, *BASE_S, f"data={csv_file}", "data.format=csv", "data.num_classes=2") == 0
    assert SEEN[0]["data"]["format"] == "csv" and SEEN[0]["data"]["num_classes"] == 2


def test_a_config_files_own_data_path_survives_a_dotted_key(files, tmp_path):
    # The old merge replaced the file's text with the dotted keys' mapping: `data: path` was lost.
    data_file = write(tmp_path / "d.yaml", "dataset: mnist\nvalidation_split: 0.25\n")
    config_file = write(tmp_path / "c.yaml", f"out: x\nmodel: vit/vit_pico\ndata: {data_file}\n")
    assert run(CLASSIFICATION, f"config={config_file}", "data.max_samples=32") == 0
    assert SEEN[0]["data"] == {"dataset": "mnist", "validation_split": 0.25, "max_samples": 32}


def test_a_config_files_data_mapping_is_read_by_its_format_too(files, tmp_path):
    config_file = write(
        tmp_path / "c.yaml",
        f"out: x\nmodel: vit/vit_pico\ndata:\n  format: csv\n  train_csv: {files['seg_csv']}\n"
        "transform_args:\n  image_size: 24\n",
    )
    assert run(CLASSIFICATION, f"config={config_file}", "data.path_column=pic", "data.label_column=lab") == 0
    assert SEEN[0]["data"]["format"] == "csv" and SEEN[0]["data"]["label_column"] == "lab"


def test_a_command_line_file_replaces_the_config_files_group_and_its_file_is_never_read(files, tmp_path):
    never = tmp_path / "never_written.yaml"
    config_file = write(tmp_path / "c.yaml", f"out: x\nmodel: vit/vit_pico\ndata: {never}\n")
    data_file = write(tmp_path / "d.yaml", "dataset: cifar10\n")
    assert run(CLASSIFICATION, f"config={config_file}", f"data={data_file}", "data.max_samples=9") == 0
    assert SEEN[0]["data"] == {"dataset": "cifar10", "max_samples": 9}


def test_a_file_that_is_missing_or_holds_a_bad_key_is_exit_2_naming_it(files, tmp_path, capsys):
    missing = tmp_path / "nope.yaml"
    assert run(CLASSIFICATION, *BASE_C, f"data={missing}", "data.max_samples=3") == 2
    assert "nope.yaml" in capsys.readouterr().err
    bad = write(tmp_path / "bad.yaml", "dataset: cifar10\nmax_sample: 5\n")
    assert run(CLASSIFICATION, *BASE_C, f"data={bad}", "data.max_samples=3") == 2
    assert "max_sample" in capsys.readouterr().err
    assert run(CLASSIFICATION, *BASE_C, f"data={bad}", "data.no_such_key=3") == 2
    assert not SEEN


def test_a_dotted_key_the_format_does_not_have_is_refused_by_name(files, tmp_path, capsys):
    data_file = write(tmp_path / "d.yaml", "dataset: cifar10\n")
    assert run(CLASSIFICATION, *BASE_C, f"data={data_file}", "data.train_dir=x") == 2
    assert "train_dir" in capsys.readouterr().err


def test_a_key_twice_exits_2_through_the_real_cli(files, tmp_path, capsys):
    data_file = write(tmp_path / "d.yaml", "dataset: cifar10\n")
    for tokens in (
        (f"data={data_file}", f"data={data_file}"),
        (f"data={data_file}", "data.max_samples=1", "data.max_samples=2"),
        (f"data={data_file}", 'data={"dataset": "mnist"}'),
    ):
        assert run(SEGMENTATION, *BASE_S, *tokens) == 2
        assert "Repeated key" in capsys.readouterr().err
    assert not SEEN
