"""Folder and csv data: the listing, the split, the statistics, the refusals and a run.

The data is PNG files under ``tmp_path`` of mixed sizes, one colour per image: the red
channel is the image's id and the green channel its class, so a pixel read back from a
pipeline names the file it came from. That makes the claims checkable by looking at what
the pipelines yield, not at the code that cut them:

* the class names are the SORTED directory names, whatever order the directories were
  made in, and a file's label is its class's index;
* the fit, selection and test splits are disjoint, the first two cover the train files,
  the third is the test directory, and the cut is a function of the seed;
* the standardization statistics are the per-channel mean and standard deviation of the
  FIT images alone, equal to a literal computed here from the colours of the files the
  fit pipeline yielded (and different from the mean of every train file, so the literal
  can tell a fit-only pass from an all-data one);
* the refusals are ``ConfigError`` lines naming the dotted key, raised by the same
  listing that runs before the run directory is claimed, with no tensor made;
* a YAML file's relative paths start at the file, a mapping's at the working directory;
* a run with no test split skips what needs test images and says so in its summary.

``tests/test_train/test_api/test_every_cli_key_reaches_the_config.py`` drives the ``builtin``
keys through ``cli.main``; the folder and csv keys are typed here through
``coerce_mapping`` (Step 28 adds them to that table).
"""

import csv
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pytest
from PIL import Image

import train.api.recipes as recipes
import train.api.tasks.image_classification as task_module
from train.api import cli, registry
from train.api.cli import coerce_mapping, parse_dotlist
from train.api.config import validate
from train.api.data_args import (
    FORMATS,
    BuiltinImageData,
    CsvImageData,
    DataFacts,
    FolderImageData,
    image_facts,
    load_data_config,
    load_file_data,
    read_data_file,
)
from train.api.errors import ConfigError, ConfigValidationError
from train.api.tasks.image_classification import (
    ImageClassificationConfig,
    ImageClassificationTrainer,
    train_image_classification,
)
from train.common.classification_data import (
    DataSourceError,
    load_csv_data,
    load_folder_data,
    read_image_csv,
    scan_image_folder,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Created in this order on purpose: the sorted order is different.
CREATED_IN_ORDER = ("zebra", "ant", "moth")
SORTED = ["ant", "moth", "zebra"]
TRAIN_PER_CLASS = 14
TEST_PER_CLASS = 4
SIZES = [(40, 30), (30, 40), (36, 36)]  # (width, height): mixed on purpose
GREEN = {"ant": 60, "moth": 120, "zebra": 180}


def colour(name: str, red: int, index: int):
    """The colour of one image: its id in red, its class in green, noise in blue."""
    return (red, GREEN[name], (index * 37) % 180 + 20)


def write_png(path: Path, size, rgb) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, rgb).save(path)


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """A train and a test directory, and the colour and class of every file in them."""
    root = tmp_path_factory.mktemp("tree")
    colours: Dict[str, tuple] = {}
    classes: Dict[int, str] = {}
    train_reds: set = set()
    test_reds: set = set()
    train_id = 0
    test_id = 0
    for name in CREATED_IN_ORDER:
        for index in range(TRAIN_PER_CLASS):
            rgb = colour(name, 20 + 4 * train_id, index)
            path = root / "train" / name / f"img_{index:02d}.png"
            write_png(path, SIZES[index % 3], rgb)
            colours[str(path)] = rgb
            classes[rgb[0]] = name
            train_reds.add(rgb[0])
            train_id += 1
        for index in range(TEST_PER_CLASS):
            rgb = colour(name, 200 + 3 * test_id, index)
            path = root / "test" / name / f"img_{index:02d}.png"
            write_png(path, SIZES[index % 3], rgb)
            colours[str(path)] = rgb
            classes[rgb[0]] = name
            test_reds.add(rgb[0])
            test_id += 1
    return SimpleNamespace(
        root=root,
        train_dir=root / "train",
        test_dir=root / "test",
        colours=colours,
        class_of_red=classes,
        train_reds=train_reds,
        test_reds=test_reds,
        n_train=train_id,
        n_test=test_id,
    )


def load(tree, **changes):
    """Load the tree unit-normalized (so a pixel reads back as value / 255)."""
    arguments = dict(
        batch_size=4,
        seed=7,
        image_size=16,
        validation_split=0.25,
        normalize="unit",
        eval_samples=100,
        test_dir=str(tree.test_dir),
    )
    arguments.update(changes)
    return load_folder_data(str(tree.train_dir), **arguments)


def reds_and_labels(dataset):
    """Return ``(sorted red ids, labels in the same order)`` of one pass over ``dataset``."""
    reds: List[int] = []
    labels: List[int] = []
    for images, batch_labels in dataset.as_numpy_iterator():
        reds.extend(np.rint(images[:, 0, 0, 0] * 255).astype(int).tolist())
        labels.extend(batch_labels.tolist())
    order = np.argsort(reds)
    return np.asarray(reds)[order].tolist(), np.asarray(labels)[order].tolist()


# ---------------------------------------------------------------------
# The listing
# ---------------------------------------------------------------------


def test_the_class_names_are_the_sorted_directory_names_and_a_label_is_the_index(tree):
    listing = scan_image_folder(str(tree.train_dir), str(tree.test_dir))
    assert listing.class_names == SORTED
    assert listing.num_classes == 3
    assert len(listing.train_paths) == tree.n_train and len(listing.test_paths) == tree.n_test
    for path, label in zip(listing.train_paths, listing.train_labels):
        assert Path(path).parent.name == SORTED[label]
    for path, label in zip(listing.test_paths, listing.test_labels):
        assert Path(path).parent.name == SORTED[label]


def test_a_hidden_entry_and_a_file_of_another_suffix_are_not_images(tmp_path):
    for name in ("a", "b"):
        write_png(tmp_path / name / "one.png", (8, 8), (1, 2, 3))
    (tmp_path / "a" / "notes.txt").write_text("not an image")
    (tmp_path / "a" / ".hidden.png").write_bytes(b"")
    (tmp_path / ".cache").mkdir()
    (tmp_path / "loose.png").write_bytes(b"")
    listing = scan_image_folder(str(tmp_path))
    assert listing.class_names == ["a", "b"]
    assert [Path(p).name for p in listing.train_paths] == ["one.png", "one.png"]
    assert listing.test_paths is None and listing.test_labels is None


def test_the_labels_the_pipelines_yield_are_the_class_indices(tree):
    data = load(tree)
    for dataset in (data.train_ds, data.select_ds, data.test_ds):
        reds, labels = reds_and_labels(dataset)
        assert labels == [SORTED.index(tree.class_of_red[red]) for red in reds]


# ---------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------


def test_the_fit_selection_and_test_splits_are_disjoint_and_cover_the_files(tree):
    data = load(tree)
    fit, _ = reds_and_labels(data.train_ds)
    select, _ = reds_and_labels(data.select_ds)
    test, _ = reds_and_labels(data.test_ds)
    # The sizes: 42 train files, a quarter (10) held out; 32 fit files in 8 batches of 4.
    assert (data.n_train, data.n_select, data.n_test) == (32, 10, 12)
    assert data.steps_per_epoch == 8
    assert len(fit) == 32 and len(set(fit)) == 32
    assert len(select) == 10 and len(test) == 12
    assert not set(fit) & set(select)
    assert set(fit) | set(select) == tree.train_reds
    assert set(test) == tree.test_reds
    assert not (set(fit) | set(select)) & set(test)
    assert data.test_split == "the test files"
    assert data.selection_split == "validation_split=0.25 of the train files (seeded permutation)"


def test_the_cut_is_a_function_of_the_seed(tree):
    select_a, _ = reds_and_labels(load(tree, seed=7).select_ds)
    select_b, _ = reds_and_labels(load(tree, seed=7).select_ds)
    select_c, _ = reds_and_labels(load(tree, seed=8).select_ds)
    assert select_a == select_b
    assert select_a != select_c


def test_max_samples_caps_the_train_pool_and_the_test_split(tree):
    data = load(tree, max_samples=10)
    assert (data.n_train, data.n_select, data.n_test) == (8, 2, 10)


def test_the_train_pipeline_reshuffles_and_the_evaluation_pipelines_do_not(tree):
    data = load(tree)
    first = [images[:, 0, 0, 0].tolist() for images, _ in data.train_ds.as_numpy_iterator()]
    second = [images[:, 0, 0, 0].tolist() for images, _ in data.train_ds.as_numpy_iterator()]
    assert first != second
    assert [b[:, 0, 0, 0].tolist() for b, _ in data.select_ds.as_numpy_iterator()] == [
        b[:, 0, 0, 0].tolist() for b, _ in data.select_ds.as_numpy_iterator()
    ]


def test_images_of_mixed_sizes_come_out_at_the_input_side(tree):
    data = load(tree, image_size=24)
    assert data.input_shape == (24, 24, 3)
    for images, _ in data.train_ds.take(2).as_numpy_iterator():
        assert images.shape == (4, 24, 24, 3)
    assert data.x_fig.shape == (12, 24, 24, 3)


# ---------------------------------------------------------------------
# The statistics
# ---------------------------------------------------------------------


def literal_statistics(tree, reds):
    """Per-channel mean and (population) standard deviation of the colours with these reds."""
    rows = np.asarray(
        [rgb for rgb in tree.colours.values() if rgb[0] in set(reds)], np.float64
    ) / 255.0
    assert len(rows) == len(reds)
    return rows.mean(axis=0), rows.std(axis=0)


def test_the_statistics_are_the_mean_and_std_of_the_fit_images_alone(tree):
    unit = load(tree)
    fit, _ = reds_and_labels(unit.train_ds)
    mean, std = literal_statistics(tree, fit)
    all_mean, all_std = literal_statistics(tree, tree.train_reds)
    # The control that gives the literal its meaning: an all-data pass is a different number.
    assert np.max(np.abs(all_mean - mean)) > 1e-3
    assert np.max(np.abs(all_std - std)) > 1e-3

    data = load(tree, normalize="standardize")
    np.testing.assert_allclose(data.mean, mean, atol=2e-5)
    np.testing.assert_allclose(data.std, std, atol=2e-5)
    assert data.mean.dtype == np.float32 and data.std.dtype == np.float32
    # The pipelines are standardized with them: a selection image read back is (x - mean) / std.
    raw = np.concatenate([b for b, _ in unit.select_ds.as_numpy_iterator()])
    standardized = np.concatenate([b for b, _ in data.select_ds.as_numpy_iterator()])
    np.testing.assert_allclose(standardized, (raw - mean) / std, atol=1e-3)


def test_unit_normalization_records_no_shift_and_no_scale(tree):
    data = load(tree)
    assert data.mean.tolist() == [0.0, 0.0, 0.0] and data.std.tolist() == [1.0, 1.0, 1.0]


def test_a_hook_that_needs_the_mean_receives_the_fit_mean(tree):
    seen = []

    def factory(mean):
        seen.append(mean)
        return None

    unit = load(tree)
    fit, _ = reds_and_labels(unit.train_ds)
    mean, _ = literal_statistics(tree, fit)
    load(tree, augment_from_mean=factory)  # unit mode: the pass runs because a hook asks
    assert len(seen) == 1 and len(seen[0]) == 3
    np.testing.assert_allclose(seen[0], mean, atol=2e-5)
    with pytest.raises(ValueError, match="augment or augment_from_mean"):
        load(tree, augment=lambda image, label: (image, label), augment_from_mean=factory)


def test_images_with_no_variation_cannot_be_standardized(tmp_path):
    for name in ("a", "b"):
        for index in range(8):
            write_png(tmp_path / name / f"{index}.png", (8, 8), (255, 255, 255))
    with pytest.raises(ValueError, match="no variation"):
        load_folder_data(str(tmp_path), batch_size=2, seed=0, image_size=8)


# ---------------------------------------------------------------------
# No test split
# ---------------------------------------------------------------------


def test_a_folder_without_a_test_directory_has_no_test_split(tree):
    data = load(tree, test_dir=None)
    assert data.test_ds is None
    assert data.n_test == 0
    assert data.x_fig.shape == (0, 16, 16, 3) and data.y_fig.shape == (0,)
    assert (data.n_train, data.n_select) == (32, 10)
    assert data.test_split == "none: the data has no test split"
    # The control: the same call with the directory has all of it.
    assert load(tree).test_ds is not None


# ---------------------------------------------------------------------
# csv
# ---------------------------------------------------------------------


def write_csv(path: Path, rows, header=("path", "label")):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def folder_rows(tree, directory, relative_to=None):
    rows = []
    listing = scan_image_folder(str(directory))
    for path, label in zip(listing.train_paths, listing.train_labels):
        shown = os.path.relpath(path, relative_to) if relative_to else path
        rows.append((shown, listing.class_names[label]))
    return rows


def test_a_csv_gives_the_record_the_folder_gives(tree, tmp_path):
    train_csv, test_csv = tmp_path / "train.csv", tmp_path / "test.csv"
    write_csv(train_csv, folder_rows(tree, tree.train_dir, tmp_path))  # relative to the csv
    write_csv(test_csv, folder_rows(tree, tree.test_dir))  # absolute
    by_folder = load(tree)
    by_csv = load_csv_data(
        str(train_csv), test_csv=str(test_csv), batch_size=4, seed=7, image_size=16,
        validation_split=0.25, normalize="unit", eval_samples=100,
    )
    assert by_csv.class_names == by_folder.class_names == SORTED
    assert (by_csv.n_train, by_csv.n_select, by_csv.n_test) == (32, 10, 12)
    for name in ("train_ds", "select_ds", "test_ds"):
        assert reds_and_labels(getattr(by_csv, name)) == reds_and_labels(getattr(by_folder, name))
    assert by_csv.test_split == "the test files"


def test_a_csv_takes_its_column_names_and_orders_integer_classes_numerically(tmp_path):
    rows = []
    for label in (2, 9, 10):  # text order would be 10, 2, 9
        for index in range(6):
            write_png(tmp_path / "pics" / f"{label}_{index}.png", (8, 8), (label * 20, index * 30, 9))
            rows.append((f"pics/{label}_{index}.png", label))
    write_csv(tmp_path / "all.csv", [(label, path) for path, label in rows], ("class", "file"))
    listing = read_image_csv(
        str(tmp_path / "all.csv"), path_column="file", label_column="class"
    )
    assert listing.class_names == ["2", "9", "10"]
    assert listing.train_labels.tolist() == [0] * 6 + [1] * 6 + [2] * 6
    assert listing.test_paths is None
    # Labels that are not all integers sort as text; other integers sort by value.
    words = {2: "b", 9: "a", 10: "c"}
    write_csv(tmp_path / "words.csv", [(p, words[l]) for p, l in rows])
    assert read_image_csv(str(tmp_path / "words.csv")).class_names == ["a", "b", "c"]
    numbers = {2: 5, 9: 100, 10: 20}
    write_csv(tmp_path / "ints.csv", [(p, numbers[l]) for p, l in rows])
    assert read_image_csv(str(tmp_path / "ints.csv")).class_names == ["5", "20", "100"]


# ---------------------------------------------------------------------
# What is refused, with the key
# ---------------------------------------------------------------------


def make_classes(root: Path, names=("a", "b"), count=2):
    for name in names:
        for index in range(count):
            write_png(root / name / f"{index}.png", (8, 8), (index, 1, 2))
    return root


def folder_data(**changes):
    return FolderImageData(**changes)


def test_a_class_directory_with_no_image_is_refused_naming_the_key(tmp_path):
    make_classes(tmp_path / "train")
    (tmp_path / "train" / "empty").mkdir()
    with pytest.raises(ConfigError, match=r"'data.train_dir'.*class directory 'empty'.*holds no images"):
        image_facts(folder_data(train_dir=str(tmp_path / "train")), 8)


def test_fewer_than_two_classes_and_a_missing_directory_are_refused(tmp_path):
    make_classes(tmp_path / "one", names=("a",))
    with pytest.raises(ConfigError, match=r"'data.train_dir'.*at least two"):
        image_facts(folder_data(train_dir=str(tmp_path / "one")), 8)
    with pytest.raises(ConfigError, match=r"'data.train_dir'.*is not a directory"):
        image_facts(folder_data(train_dir=str(tmp_path / "missing")), 8)


def test_test_classes_that_differ_from_the_train_classes_are_refused(tmp_path):
    make_classes(tmp_path / "train", names=("a", "b", "c"))
    make_classes(tmp_path / "test", names=("a", "b", "d"))
    with pytest.raises(ConfigError) as caught:
        image_facts(folder_data(train_dir=str(tmp_path / "train"), test_dir=str(tmp_path / "test")), 8)
    assert str(caught.value).startswith("Invalid value for 'data.test_dir'")
    assert "only in train ['c'], only in test ['d']" in str(caught.value)
    # The control: the same test directory with the right classes is accepted.
    make_classes(tmp_path / "good", names=("a", "b", "c"))
    facts = image_facts(
        folder_data(train_dir=str(tmp_path / "train"), test_dir=str(tmp_path / "good")), 8
    )
    assert facts == DataFacts(dataset="folder", num_classes=3, image_size=8)


def test_an_unset_image_size_is_refused_because_files_have_no_native_size(tmp_path):
    make_classes(tmp_path / "train")
    for data in (
        folder_data(train_dir=str(tmp_path / "train")),
        CsvImageData(train_csv=str(tmp_path / "unused.csv")),
    ):
        with pytest.raises(ConfigError, match=r"'transform_args.image_size'.*no native size"):
            image_facts(data, None)


def csv_for(tmp_path: Path, rows, name="data.csv", header=("path", "label")):
    path = tmp_path / name
    write_csv(path, rows, header)
    return path


@pytest.mark.parametrize(
    "case, pattern",
    [
        ("missing_path_column", r"'data.path_column'.*column 'path' is not in"),
        ("missing_label_column", r"'data.label_column'.*column 'label' is not in"),
        ("missing_image", r"'data.train_csv'.*line 2: .*nope.png' does not exist"),
        ("bad_suffix", r"'data.train_csv'.*line 2: 'notes.txt' is not one of"),
        ("empty_cell", r"'data.train_csv'.*line 2 has an empty"),
        ("no_rows", r"'data.train_csv'.*holds no rows"),
        ("one_class", r"'data.train_csv'.*1 class\(es\)"),
        ("no_such_csv", r"'data.train_csv'.*is not a file"),
        ("test_classes", r"'data.test_csv'.*only in train \['b'\], only in test \['z'\]"),
    ],
)
def test_a_bad_csv_is_refused_naming_the_key(tmp_path, case, pattern):
    make_classes(tmp_path / "pics")
    good = [("pics/a/0.png", "a"), ("pics/b/0.png", "b")]
    data = {"train_csv": str(csv_for(tmp_path, good))}
    if case == "missing_path_column":
        data["train_csv"] = str(csv_for(tmp_path, good, header=("file", "label")))
    elif case == "missing_label_column":
        data["train_csv"] = str(csv_for(tmp_path, good, header=("path", "class")))
    elif case == "missing_image":
        data["train_csv"] = str(csv_for(tmp_path, [("nope.png", "a")] + good))
    elif case == "bad_suffix":
        (tmp_path / "notes.txt").write_text("x")
        data["train_csv"] = str(csv_for(tmp_path, [("notes.txt", "a")] + good))
    elif case == "empty_cell":
        data["train_csv"] = str(csv_for(tmp_path, [("pics/a/0.png", "")] + good))
    elif case == "no_rows":
        data["train_csv"] = str(csv_for(tmp_path, []))
    elif case == "one_class":
        data["train_csv"] = str(csv_for(tmp_path, good[:1]))
    elif case == "no_such_csv":
        data["train_csv"] = str(tmp_path / "absent.csv")
    elif case == "test_classes":
        data["test_csv"] = str(
            csv_for(tmp_path, [("pics/a/0.png", "a"), ("pics/b/0.png", "z")], name="test.csv")
        )
    with pytest.raises(ConfigError, match=pattern):
        image_facts(CsvImageData(**data), 8)


def test_the_loader_refuses_with_a_value_error_a_trainer_turns_into_a_config_error(tree):
    with pytest.raises(ValueError, match="exceeds the 32 train samples"):
        load(tree, batch_size=64)
    with pytest.raises(ValueError, match="strictly inside"):
        load(tree, validation_split=0.0)
    with pytest.raises(ValueError, match="holds out 0 of"):
        load(tree, validation_split=0.01)
    with pytest.raises(DataSourceError) as caught:
        scan_image_folder(str(tree.root / "nowhere"))
    assert caught.value.key == "train_dir" and isinstance(caught.value, ValueError)


def test_a_recipe_loader_mapping_is_refused_for_file_data(tree):
    data = FolderImageData(train_dir=str(tree.train_dir))
    with pytest.raises(ConfigError, match=r"'loader'.*built-in datasets only"):
        load_file_data(
            data, batch_size=4, seed=0, augment=None, augment_from_mean=None,
            normalize="unit", image_size=16, loader={"channels": 3},
        )


# ---------------------------------------------------------------------
# The facts, before anything is loaded
# ---------------------------------------------------------------------


def test_the_facts_come_from_the_listing(tree, tmp_path):
    folder = FolderImageData(train_dir=str(tree.train_dir), test_dir=str(tree.test_dir))
    assert image_facts(folder, 16) == DataFacts(dataset="folder", num_classes=3, image_size=16)
    train_csv = csv_for(tmp_path, folder_rows(tree, tree.train_dir))
    assert image_facts(CsvImageData(train_csv=str(train_csv)), 24) == DataFacts(
        dataset="csv", num_classes=3, image_size=24
    )
    assert image_facts(BuiltinImageData(dataset="cifar10"), None) == DataFacts(
        dataset="cifar10", num_classes=10, image_size=32
    )


CHILD = textwrap.dedent(
    f"""
    import json
    import sys

    sys.path.insert(0, {str(REPO_ROOT)!r})
    from tensorflow.python.eager import context as eager

    import keras
    import train.common.gpu as gpu
    from train.api.tasks.image_classification import train_image_classification

    out, data, mode = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
    seen = {{"at_start": eager.context()._context_handle is not None, "calls": []}}


    class Recorded(Exception):
        pass


    def recorder(gpu_id=None, flag="--gpu"):
        seen["calls"].append(eager.context()._context_handle is not None)
        raise Recorded


    gpu.setup_gpu = recorder
    if mode == "control":
        keras.ops.zeros(1)
    try:
        train_image_classification(
            out=out, model="vit/vit_pico", data=data, epochs=1, batch_size=4,
            transform_args={{"image_size": 32}},
        )
    except Recorded:
        pass
    print(json.dumps(seen))
    """
)


def run_child(tmp_path: Path, data: dict, mode: str = "real") -> dict:
    """Run the driver in a child; each call claims its own directory (the recorder stops the run after the claim)."""
    run = tmp_path / f"run_{len(list(tmp_path.glob('run_*')))}"
    env = {**os.environ, "MPLBACKEND": "Agg", "TF_CPP_MIN_LOG_LEVEL": "3", "CUDA_VISIBLE_DEVICES": "-1"}
    done = subprocess.run(
        [sys.executable, "-c", CHILD, str(run), json.dumps(data), mode],
        capture_output=True, text=True, timeout=600, cwd=tmp_path, env=env,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_describing_a_folder_or_a_csv_makes_no_tensor_before_setup_gpu(tree, tmp_path):
    train_csv = csv_for(tmp_path, folder_rows(tree, tree.train_dir))
    for data in (
        {"format": "folder", "train_dir": str(tree.train_dir), "test_dir": str(tree.test_dir)},
        {"format": "csv", "train_csv": str(train_csv)},
    ):
        seen = run_child(tmp_path, data)
        assert seen == {"at_start": False, "calls": [False]}, (data["format"], seen)
    # The control: an op before the call makes the recorder see it.
    control = run_child(tmp_path, {"format": "folder", "train_dir": str(tree.train_dir)}, "control")
    assert control["calls"] == [True]


# ---------------------------------------------------------------------
# The config: the format tells the three apart
# ---------------------------------------------------------------------


def test_the_format_names_are_the_classes_and_the_recipe_knows_the_file_ones():
    assert list(FORMATS) == ["builtin", "folder", "csv"]
    assert registry.FILE_DATASETS == tuple(name for name in FORMATS if name != "builtin")
    assert [FORMATS[name]  for name in FORMATS] == [BuiltinImageData, FolderImageData, CsvImageData]
    for name, cls in FORMATS.items():
        assert validate(cls, _minimal(name)).format == name


def _minimal(name):
    return {
        "builtin": {"dataset": "cifar10"},
        "folder": {"train_dir": "x"},
        "csv": {"train_csv": "x.csv"},
    }[name]


def task_mapping(data):
    return {"out": "o", "model": "vit/vit_pico", "data": data}


@pytest.mark.parametrize("name", list(FORMATS))
def test_the_task_config_builds_the_class_the_format_names(name):
    config = validate(ImageClassificationConfig, task_mapping({"format": name, **_minimal(name)}))
    assert type(config.data) is FORMATS[name]
    untagged = validate(ImageClassificationConfig, task_mapping(_minimal("builtin")))
    assert type(untagged.data) is BuiltinImageData  # no format: the default one


def test_an_unknown_format_is_one_line_naming_the_choices():
    with pytest.raises(ConfigValidationError) as caught:
        validate(ImageClassificationConfig, task_mapping({"format": "parquet", "dataset": "cifar10"}))
    assert caught.value.errors == (
        "Invalid value for 'data.format': expected one of 'builtin', 'folder', 'csv', "
        "got 'parquet' (str)",
    )


def test_a_key_of_another_format_is_unknown_not_a_format_error():
    with pytest.raises(ConfigValidationError) as caught:
        validate(
            ImageClassificationConfig,
            task_mapping({"format": "folder", "train_dir": "x", "dataset": "cifar10"}),
        )
    assert caught.value.errors == ("Unknown key 'data.dataset'.",)


def test_the_command_line_types_the_keys_of_the_format_it_names():
    _, raw = parse_dotlist(
        ["out=o", "model=vit/vit_pico", "data.format=folder", "data.train_dir=/d/train",
         "data.test_dir=/d/test", "data.validation_split=0.2", "data.max_samples=50"]
    )
    config = validate(ImageClassificationConfig, coerce_mapping(ImageClassificationConfig, raw))
    assert config.data == FolderImageData(
        train_dir="/d/train", test_dir="/d/test", validation_split=0.2, max_samples=50
    )
    _, raw = parse_dotlist(
        ["out=o", "model=vit/vit_pico", "data.format=csv", "data.train_csv=t.csv",
         "data.path_column=file", "data.eval_samples=5"]
    )
    config = validate(ImageClassificationConfig, coerce_mapping(ImageClassificationConfig, raw))
    assert config.data == CsvImageData(train_csv="t.csv", path_column="file", eval_samples=5)
    _, raw = parse_dotlist(["out=o", "model=vit/vit_pico", "data.format=parquet"])
    with pytest.raises(ConfigValidationError) as caught:
        coerce_mapping(ImageClassificationConfig, raw)
    assert caught.value.errors == (
        "Invalid value for 'data.format': expected one of 'builtin', 'folder', 'csv', "
        "got 'parquet' (str)",
    )
    _, raw = parse_dotlist(["out=o", "data.format=folder", "data.dataset=cifar10"])
    with pytest.raises(ConfigValidationError, match="Unknown key 'data.dataset'"):
        coerce_mapping(ImageClassificationConfig, raw)


def test_the_help_lists_the_keys_of_all_three_formats():
    lines = "\n".join(cli._field_help_lines(ImageClassificationConfig))
    for key in ("data.dataset", "data.train_dir", "data.test_dir", "data.train_csv",
                "data.path_column", "data.label_column"):
        assert f"  {key}  " in lines


@pytest.mark.parametrize(
    "data, key",
    [
        (FolderImageData(train_dir="x", validation_split=1.5), "data.validation_split"),
        (FolderImageData(train_dir="x", max_samples=0), "data.max_samples"),
        (FolderImageData(train_dir=" "), "data.train_dir"),
        (CsvImageData(train_csv="x.csv", eval_samples=-1), "data.eval_samples"),
        (CsvImageData(train_csv="x.csv", path_column=""), "data.path_column"),
    ],
)
def test_an_option_out_of_range_names_its_key(data, key):
    lines = data.problems("data")
    assert len(lines) == 1 and f"'{key}'" in lines[0]


# ---------------------------------------------------------------------
# Relative paths
# ---------------------------------------------------------------------


def test_a_yaml_files_relative_paths_start_at_the_file(tree, tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    relative = os.path.relpath(tree.train_dir, configs)
    file = configs / "data.yaml"
    file.write_text(f"format: folder\ntrain_dir: {relative}\ntest_dir: {os.path.relpath(tree.test_dir, configs)}\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # not the file's directory: a cwd-relative reading would miss
    data = load_data_config(file)
    assert data == FolderImageData(train_dir=str(tree.train_dir), test_dir=str(tree.test_dir))
    assert os.path.isabs(data.train_dir)
    assert load_data_config(str(file)) == data
    assert image_facts(data, 16).num_classes == 3
    # The command line's data=file.yaml reads it the same way.
    expanded = cli._expand_path_forms(ImageClassificationConfig, {"data": str(file)})
    assert expanded["data"] == read_data_file(file)
    assert expanded["data"]["train_dir"] == str(tree.train_dir)
    # An absolute path stays as written.
    file.write_text(f"format: folder\ntrain_dir: {tree.train_dir}\n")
    assert load_data_config(file).train_dir == str(tree.train_dir)


def test_a_mappings_relative_paths_start_at_the_working_directory(tree, monkeypatch):
    monkeypatch.chdir(tree.root)
    data = load_data_config({"format": "folder", "train_dir": "train", "test_dir": "test"})
    assert (data.train_dir, data.test_dir) == ("train", "test")  # kept as written
    assert image_facts(data, 16).num_classes == 3
    monkeypatch.chdir(tree.root / "train")
    with pytest.raises(ConfigError, match=r"'data.train_dir'.*is not a directory"):
        image_facts(data, 16)


def test_a_yaml_file_of_csv_data_anchors_its_csv_paths(tree, tmp_path, monkeypatch):
    train_csv = csv_for(tmp_path, folder_rows(tree, tree.train_dir), name="train.csv")
    file = tmp_path / "csv.yaml"
    file.write_text("format: csv\ntrain_csv: train.csv\nlabel_column: label\n")
    monkeypatch.chdir(tree.root)
    data = load_data_config(file)
    assert data == CsvImageData(train_csv=str(train_csv))
    assert image_facts(data, 16).num_classes == 3


# ---------------------------------------------------------------------
# The recipe and the augmentation of file data
# ---------------------------------------------------------------------


def folder_config(out, tree, **changes):
    mapping = {
        "out": str(out),
        "model": "vit/vit_pico",
        "data": {"format": "folder", "train_dir": str(tree.train_dir), "test_dir": str(tree.test_dir)},
        "batch_size": 4,
        "transform_args": {"image_size": 16},
        "analysis_args": {"model_analysis": False},
    }
    for key, value in changes.items():
        mapping[key] = {**mapping.get(key, {}), **value} if isinstance(value, dict) else value
    return validate(ImageClassificationConfig, mapping)


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        yield


def resolve(config):
    trainer = ImageClassificationTrainer(config)
    family, variant = registry.parse_model_name(config.model)
    resolved, context = trainer._resolve(trainer.describe_data(config), family, variant)
    return trainer, resolved, context


def test_the_augment_default_of_file_data_is_flip_and_an_explicit_value_wins(world, tree, tmp_path):
    for fmt in ("folder", "csv"):
        spec = registry.MODELS["vit"]
        assert spec.recipe(fmt, "vit_pico")["transform_args"]["augment"] == "flip"
    assert registry.MODELS["vit"].recipe("cifar10", "vit_pico")["transform_args"]["augment"] == "cifar_light"
    _, resolved, context = resolve(folder_config(tmp_path / "a", tree))
    assert resolved.transform_args.augment == "flip"
    assert context.dataset == "folder" and context.num_classes == 3 and context.image_size == 16
    _, kept, _ = resolve(folder_config(tmp_path / "b", tree, transform_args={"augment": "none"}))
    assert kept.transform_args.augment == "none"
    # No unrelated recipe value moved: the optimizer is the family's base recipe.
    base = registry.MODELS["vit"].recipe("folder", "vit_pico")["optim_args"]["lr"]
    assert resolved.optim_args.lr == base


def test_the_trainer_hands_the_loader_the_hook_or_the_function_that_needs_the_mean(
    world, tree, tmp_path, monkeypatch
):
    import train.common.image_augment as augment_module

    captured = {}
    monkeypatch.setattr(task_module, "load_image_data", lambda data, **kw: captured.update(kw))
    built = []
    real = augment_module.build_augment

    def recording(name, **kwargs):
        built.append((name, kwargs))
        return real(name, **kwargs)

    monkeypatch.setattr(augment_module, "build_augment", recording)

    trainer, resolved, context = resolve(folder_config(tmp_path / "flip", tree))
    trainer.build_data(resolved, context)
    assert callable(captured["augment"]) and captured["augment_from_mean"] is None
    assert built[-1][0] == "flip"

    erasing = {"augment": "cifar_autoaugment", "random_erasing_prob": 0.25, "image_size": 16}
    trainer, resolved, context = resolve(
        folder_config(tmp_path / "erase", tree, transform_args=erasing)
    )
    trainer.build_data(resolved, context)
    assert captured["augment"] is None
    captured["augment_from_mean"]((0.1, 0.2, 0.3))  # the loader calls it with the fit mean
    assert built[-1][0] == "cifar_autoaugment"
    assert built[-1][1]["fill"] == (0.1, 0.2, 0.3) and built[-1][1]["random_erasing_prob"] == 0.25


# ---------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------


def run_folder(out, tree, **changes):
    arguments = dict(
        out=str(out),
        model="vit/vit_pico",
        data={"format": "folder", "train_dir": str(tree.train_dir)},
        epochs=1,
        batch_size=4,
        transform_args={"image_size": 32},
        analysis_args={"model_analysis": False},
        metric_args={"success_threshold": 0.0},
    )
    arguments.update(changes)
    return train_image_classification(**arguments)


@pytest.fixture(scope="module")
def no_test_run(world, tree, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("no_test") / "run"
    result = run_folder(run_dir, tree, analysis_args={"model_analysis": True, "conformal_alpha": 0.2})
    return result, run_dir, json.loads((run_dir / "results_summary.json").read_text())


@pytest.fixture(scope="module")
def with_test_run(world, tree, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("with_test") / "run"
    data = {"format": "folder", "train_dir": str(tree.train_dir), "test_dir": str(tree.test_dir)}
    result = run_folder(run_dir, tree, data=data)
    return result, run_dir, json.loads((run_dir / "results_summary.json").read_text())


def test_a_run_with_no_test_split_skips_what_needs_test_images_and_says_so(no_test_run):
    result, run_dir, summary = no_test_run
    assert summary["status"] == "ok"
    assert summary["has_test_split"] is False
    assert summary["n_test"] == 0 and summary["n_val"] == 4 and summary["n_train"] == 38
    assert summary["dataset"] == "folder" and summary["data_format"] == "folder"
    assert summary["test_split"] == "none: the data has no test split"
    assert summary["test_metrics_best"] is None and summary["test_metrics_final"] is None
    assert result.test_metrics_best is None and result.test_metrics_final is None
    assert summary["conformal"] is None and summary["ece"] is None
    assert summary["visualizations"] == {"files": [], "ece": None, "failed": [], "skipped": True}
    assert summary["analyzer"]["status"] == "skipped"
    assert any("no test split" in note for note in summary["notes"])
    assert summary["figure_samples"] == 0
    # What does not need test images still ran: the reload check of the best checkpoint
    # (on the selection split) and the final model's load check.
    assert summary["best_checkpoint_load_error"] is None
    assert summary["best_checkpoint_max_abs_diff"] is not None
    assert summary["model_loading_validated"] is True
    # The files: no figure but the dashboard, no analysis directory, no conformal output.
    written = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file())
    assert "visualizations/training_dashboard.png" in written
    assert not [p for p in written if p.startswith("visualizations/") and p != "visualizations/training_dashboard.png"]
    assert not [p for p in written if p.startswith("model_analysis") or "conformal" in p]
    assert {"best_model.keras", "final_model.keras", "results_summary.json"} <= set(written)
    log = (run_dir / "run.log").read_text()
    assert "conformal prediction is skipped" in log
    assert "Analyzer skipped (the data has no test split)" in log
    assert "Selection results (reload check only)" in log


def test_the_resolved_config_records_the_data_it_was_given(no_test_run, tree):
    _, run_dir, _ = no_test_run
    config = json.loads((run_dir / "config.json").read_text())
    assert config["data"]["format"] == "folder"
    assert config["data"]["train_dir"] == str(tree.train_dir)
    assert config["data"]["test_dir"] is None
    assert config["transform_args"]["augment"] == "flip"
    assert config["transform_args"]["image_size"] == 32
    assert config["used"]["loader"] == {}


def test_the_control_run_with_a_test_split_reports_everything(with_test_run):
    result, run_dir, summary = with_test_run
    assert summary["has_test_split"] is True
    assert summary["n_test"] == 12 and summary["test_split"] == "the test files"
    assert set(summary["test_metrics_best"]) >= {"accuracy", "loss"}
    assert set(summary["test_metrics_final"]) >= {"accuracy", "loss"}
    assert result.test_metrics_best == summary["test_metrics_best"]
    assert summary["figure_samples"] == 12
    assert summary["visualizations"]["files"] and not summary["visualizations"].get("skipped")
    assert (run_dir / "visualizations" / "confusion_matrix.png").is_file()
    assert summary["analyzer"]["status"] == "skipped"  # model_analysis=False in the helper
    assert summary["best_checkpoint_load_error"] is None


def test_an_unusable_folder_is_refused_before_the_run_directory_is_claimed(world, tree, tmp_path):
    bad = tmp_path / "bad_train"
    make_classes(bad)
    (bad / "empty").mkdir()
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match=r"'data.train_dir'.*class directory 'empty'"):
        train_image_classification(
            out=str(out), model="vit/vit_pico",
            data={"format": "folder", "train_dir": str(bad)},
            transform_args={"image_size": 32}, batch_size=4,
        )
    assert not out.exists()
    with pytest.raises(ConfigError, match=r"'transform_args.image_size'.*no native size"):
        train_image_classification(
            out=str(out), model="vit/vit_pico",
            data={"format": "folder", "train_dir": str(tree.train_dir)}, batch_size=4,
        )
    assert not out.exists()


def test_a_batch_larger_than_the_fit_split_is_a_config_error_after_the_claim(world, tree, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match=r"exceeds the 38 train samples"):
        run_folder(out, tree, batch_size=64)
    assert (out / "config_requested.json").is_file()  # claimed, as the message says
