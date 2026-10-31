"""``task.json`` and the predictor: a trained run reloads and predicts what the loader fed it.

The claim under test is that ``Predictor.predict`` gives, for an image, the softmax of what the
model computes on the SAME input the training loader produced for that image: same decode, same
resize, same standardization. It is checked against the loader's own batch, never against a
re-statement of the preprocessing:

* a real folder run (non-square PNG noise of mixed sizes, so a resize changes the pixels, and a
  fit-split mean and std that no constant could equal) is reloaded from its run directory, and a
  path, a PIL image, a ``uint8`` array and a float array of every test file reproduce the
  probabilities of ``model(loader batch)`` within 1e-5;
* the same equality holds for built-in data (CIFAR-10 arrays; MNIST, whose single channel the
  convnext recipe repeats three times) through the trainer's own ``task_record``;
* the statistics in ``task.json`` are LIVE: a copy of the model file with a wrong mean, or a wrong
  std, predicts something else, and the unmodified copy still matches;
* a run with no test split writes the same ``task.json``;
* the schema, every refusal, and the fresh-process reload (with ``train`` importable, and the model
  file alone with it blocked, D-051) are tested on files and on hand-made records.

The pure tests use a hand-made linear model: its logits are known, so ordering, ``top_k`` and
``threshold`` are checked against numbers. Everything writes under ``tmp_path``.
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import keras
import numpy as np
import pytest
from PIL import Image

import train.api.recipes as recipes
from train.api import predictor, registry
from train.api.config import validate
from train.api.errors import ConfigError
from train.api.predictor import (
    EVAL_KINDS,
    RECORD_KEYS,
    SCHEMA_VERSION,
    Predictor,
    build_task_record,
    check_record,
    load_model,
)
from train.api.tasks.image_classification import (
    ImageClassificationConfig,
    ImageClassificationTrainer,
    train_image_classification,
)
from dl_techniques.datasets.vision.splits import seeded_split_indices
from train.common.classification_data import (
    DEFAULT_VALIDATION_SPLIT,
    load_folder_data,
    scan_image_folder,
)
from train.common.seed import DEFAULT_SEED

from tests.classification_fakes import use_tiny_raw

CREATED_IN_ORDER = ("zebra", "ant", "moth")
CLASS_NAMES = ["ant", "moth", "zebra"]
SIZES = [(40, 30), (30, 45), (36, 36)]  # (width, height); two of three are not square
IMAGE_SIZE = 32
BATCH = 4
ATOL = 1e-5


# ---------------------------------------------------------------------
# Data and runs
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        yield


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """Noise PNGs of mixed, non-square sizes; the class sets the brightness."""
    root = tmp_path_factory.mktemp("tree")
    rng = np.random.default_rng(3)
    for split, per_class in (("train", 14), ("test", 4)):
        for k, name in enumerate(CREATED_IN_ORDER):
            for index in range(per_class):
                width, height = SIZES[index % 3]
                pixels = rng.integers(0, 120, size=(height, width, 3)) + 40 * CLASS_NAMES.index(name)
                path = root / split / name / f"img_{index:02d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels.astype(np.uint8)).save(path)
    return SimpleNamespace(root=root, train_dir=root / "train", test_dir=root / "test")


def run_folder(out, tree, with_test=True, **changes):
    data = {"format": "folder", "train_dir": str(tree.train_dir)}
    if with_test:
        data["test_dir"] = str(tree.test_dir)
    arguments = dict(
        out=str(out),
        model="vit/vit_pico",
        data=data,
        epochs=2,
        batch_size=BATCH,
        transform_args={"image_size": IMAGE_SIZE},
        analysis_args={"model_analysis": False},
        metric_args={"success_threshold": 0.0},
    )
    arguments.update(changes)
    return train_image_classification(**arguments)


@pytest.fixture(scope="module")
def run_dir(world, tree, tmp_path_factory):
    path = tmp_path_factory.mktemp("with_test") / "run"
    run_folder(path, tree)
    return path


@pytest.fixture(scope="module")
def no_test_run_dir(world, tree, tmp_path_factory):
    path = tmp_path_factory.mktemp("no_test") / "run"
    run_folder(path, tree, with_test=False, epochs=1)
    return path


def read_task(directory):
    return json.loads((Path(directory) / "task.json").read_text())


def listed_test_paths(tree):
    """The test files in the order the loader's test pipeline yields them (a seeded order)."""
    listing = scan_image_folder(str(tree.train_dir), str(tree.test_dir))
    _, _, order = seeded_split_indices(
        len(listing.train_paths), len(listing.test_paths), DEFAULT_SEED, DEFAULT_VALIDATION_SPLIT, None
    )
    return [listing.test_paths[i] for i in order], np.asarray(listing.test_labels)[order]


@pytest.fixture(scope="module")
def truth(run_dir, tree):
    """``(paths, probabilities)``: the loader's own normalized test batch through the saved model."""
    data = load_folder_data(
        str(tree.train_dir),
        test_dir=str(tree.test_dir),
        batch_size=BATCH,
        seed=DEFAULT_SEED,
        image_size=IMAGE_SIZE,
    )
    paths, listed_labels = listed_test_paths(tree)
    batch = np.concatenate([x for x, _ in data.test_ds.as_numpy_iterator()])
    labels = np.concatenate([y for _, y in data.test_ds.as_numpy_iterator()])
    assert labels.tolist() == listed_labels.tolist()  # the order is the loader's
    model = keras.models.load_model(run_dir / "best_model.keras", compile=False)
    logits = np.asarray(model(batch, training=False), np.float64)
    return SimpleNamespace(paths=paths, probabilities=softmax(logits), data=data)


def softmax(logits):
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def in_class_order(result, names=CLASS_NAMES):
    """``(N, classes)`` array from the ``{"labels", "scores"}`` of several images."""
    rows = []
    for labels, scores in zip(result["labels"], result["scores"]):
        assert sorted(labels) == sorted(names)
        by_name = dict(zip(labels, scores))
        rows.append([by_name[name] for name in names])
    return np.asarray(rows)


# ---------------------------------------------------------------------
# task.json
# ---------------------------------------------------------------------


def test_task_json_holds_the_schema_and_what_the_loader_reports(run_dir, truth):
    record = read_task(run_dir)
    assert list(record) == list(RECORD_KEYS)
    assert record["schema_version"] == SCHEMA_VERSION == 1
    assert (record["task"], record["model"], record["family"], record["variant"]) == (
        "image_classification",
        "vit/vit_pico",
        "vit",
        "vit_pico",
    )
    assert record["class_names"] == CLASS_NAMES
    assert record["input_shape"] == [IMAGE_SIZE, IMAGE_SIZE, 3] and record["channels"] == 3
    assert record["normalize"] == "standardize"
    assert record["eval_transform"] == {"kind": "resize", "source_channels": 3}
    # What training standardized with: the statistics of the FIT split (D-060), not a constant.
    assert record["mean"] == [float(v) for v in truth.data.mean]
    assert record["std"] == [float(v) for v in truth.data.std]
    summary = json.loads((run_dir / "results_summary.json").read_text())
    assert summary["input_normalization"] == {"mean": record["mean"], "std": record["std"]}
    from dl_techniques.datasets.vision.classification import get_spec

    assert record["mean"] != list(get_spec("imagenet").mean)


def test_a_run_with_no_test_split_still_writes_the_same_task_json(no_test_run_dir, run_dir):
    summary = json.loads((no_test_run_dir / "results_summary.json").read_text())
    assert summary["has_test_split"] is False
    assert read_task(no_test_run_dir) == read_task(run_dir)


# ---------------------------------------------------------------------
# The predictor equals the model on the loader's own input
# ---------------------------------------------------------------------


def forms(paths):
    """The same test images as each input form the predictor takes."""
    as_arrays = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    return {
        "path": [str(p) for p in paths],
        "pathlib": [Path(p) for p in paths],
        "pil": [Image.open(p) for p in paths],
        "uint8": as_arrays,
        "float64": [a.astype(np.float64) / 255.0 for a in as_arrays],
        "float32": [(a / np.float32(255)).astype(np.float32) for a in as_arrays],
    }


def test_the_test_images_are_not_square_so_a_resize_changes_the_pixels(truth):
    sizes = {Image.open(p).size for p in truth.paths}
    assert any(w != h for w, h in sizes) and (IMAGE_SIZE, IMAGE_SIZE) not in sizes


@pytest.mark.parametrize("form", ["path", "pathlib", "pil", "uint8", "float64", "float32"])
def test_predict_equals_the_model_on_the_loaders_batch(run_dir, truth, form):
    predictor_ = load_model(run_dir)
    result = predictor_.predict(forms(truth.paths)[form])
    assert in_class_order(result).shape == truth.probabilities.shape
    np.testing.assert_allclose(in_class_order(result), truth.probabilities, atol=ATOL, rtol=0)


def test_one_image_gives_flat_lists_and_a_list_gives_one_list_per_image(run_dir, truth):
    predictor_ = load_model(run_dir)
    one = predictor_.predict(truth.paths[0])
    assert isinstance(one["labels"][0], str) and isinstance(one["scores"][0], float)
    assert len(one["labels"]) == len(one["scores"]) == 3
    several = predictor_.predict([truth.paths[0]])
    assert several["labels"] == [one["labels"]] and several["scores"] == [one["scores"]]
    assert abs(sum(one["scores"]) - 1.0) < 1e-6
    assert one["scores"] == sorted(one["scores"], reverse=True)


def test_a_mixed_list_of_forms_is_predicted_like_the_paths(run_dir, truth):
    predictor_ = load_model(run_dir)
    pieces = [forms(truth.paths[i : i + 1]) for i in range(4)]
    mixed = [pieces[0]["path"][0], pieces[1]["pil"][0], pieces[2]["uint8"][0], pieces[3]["float32"][0]]
    result = predictor_.predict(mixed)
    np.testing.assert_allclose(in_class_order(result), truth.probabilities[:4], atol=ATOL, rtol=0)


def test_more_images_than_a_chunk_are_all_predicted_in_order(run_dir, truth, monkeypatch):
    monkeypatch.setattr(predictor, "CHUNK", 5)  # 12 test images -> chunks of 5, 5, 2
    result = load_model(run_dir).predict(truth.paths)
    np.testing.assert_allclose(in_class_order(result), truth.probabilities, atol=ATOL, rtol=0)


def copy_with_record(run_dir, destination, source="best_model.keras", **changes):
    """A directory with the model file and a ``task.json`` edited by ``changes``."""
    destination.mkdir(parents=True)
    shutil.copy(run_dir / source, destination / source)
    record = read_task(run_dir)
    record.update(changes)
    (destination / "task.json").write_text(json.dumps(record))
    return destination / source


def test_the_recorded_mean_and_std_are_live(run_dir, truth, tmp_path):
    same = load_model(copy_with_record(run_dir, tmp_path / "same")).predict(truth.paths)
    np.testing.assert_allclose(in_class_order(same), truth.probabilities, atol=ATOL, rtol=0)  # the control
    record = read_task(run_dir)
    wrong_mean = copy_with_record(run_dir, tmp_path / "mean", mean=[m + 0.3 for m in record["mean"]])
    wrong_std = copy_with_record(run_dir, tmp_path / "std", std=[s * 2.0 for s in record["std"]])
    for path in (wrong_mean, wrong_std):
        moved = in_class_order(load_model(path).predict(truth.paths))
        assert np.abs(moved - truth.probabilities).max() > 1e-3, path.parent.name


def test_a_kind_changes_the_pixels(run_dir, truth, tmp_path):
    other = copy_with_record(
        run_dir,
        tmp_path / "crop",
        eval_transform={"kind": "resize_centre_crop", "source_channels": 3},
    )
    moved = in_class_order(load_model(other).predict(truth.paths))
    assert np.abs(moved - truth.probabilities).max() > 1e-3


# ---------------------------------------------------------------------
# Built-in data, through the trainer's own record
# ---------------------------------------------------------------------


def trainer_record(model, dataset, **given):
    """``(record, data)`` the trainer would write for ``model`` on a built-in dataset."""
    config = validate(
        ImageClassificationConfig,
        {"out": "unused", "model": model, "data": {"dataset": dataset}, "batch_size": BATCH, **given},
    )
    trainer = ImageClassificationTrainer(config)
    family, variant = registry.parse_model_name(model)
    resolved, context = trainer._resolve(trainer.describe_data(config), family, variant)
    data = trainer.build_data(resolved, context)
    return trainer.task_record(resolved, context, data), data


def linear_model(shape, classes, seed=0):
    initializer = keras.initializers.RandomNormal(stddev=0.02, seed=seed)
    inputs = keras.Input(shape)
    outputs = keras.layers.Dense(classes, kernel_initializer=initializer)(keras.layers.Flatten()(inputs))
    return keras.Model(inputs, outputs)


def loader_probabilities(model, dataset):
    batch = np.concatenate([x for x, _ in dataset.as_numpy_iterator()])
    return softmax(np.asarray(model(batch, training=False), np.float64))


@pytest.mark.parametrize("normalize", ["standardize", "unit"])
def test_cifar10_arrays_are_predicted_like_the_loaders_batch(world, monkeypatch, normalize):
    from dl_techniques.datasets.vision.classification import get_spec

    with monkeypatch.context() as patch:
        use_tiny_raw(patch, n_train=40, n_test=12)
        record, data = trainer_record(
            "vit/vit_pico", "cifar10", transform_args={"normalize": normalize}
        )
    spec = get_spec("cifar10")
    if normalize == "standardize":
        assert record["mean"] == pytest.approx(list(spec.mean)) and record["std"] == pytest.approx(list(spec.std))
    else:
        assert record["mean"] == [0.0] * 3 and record["std"] == [1.0] * 3
    assert record["eval_transform"] == {"kind": "resize", "source_channels": 3}
    assert record["class_names"] == list(spec.class_names)
    model = linear_model((32, 32, 3), 10)
    expected = loader_probabilities(model, data.test_ds)
    images = data.arrays.x_test  # float32 in [0, 1], the native 32 px
    predictor_ = Predictor(model, record)
    for given in (list(images), [np.rint(i * 255).astype(np.uint8) for i in images], images):
        result = predictor_.predict(given)
        np.testing.assert_allclose(
            in_class_order(result, record["class_names"]), expected, atol=ATOL, rtol=0
        )


def test_mnists_one_channel_is_read_as_one_and_repeated_to_the_models_three(world, monkeypatch, tmp_path):
    with monkeypatch.context() as patch:
        use_tiny_raw(patch, n_train=40, n_test=12)
        record, data = trainer_record("convnext_v1/cifar10", "mnist")
    assert record["input_shape"] == [28, 28, 3] and record["channels"] == 3
    assert record["eval_transform"]["source_channels"] == 1  # the repeated channel is recorded
    assert len(record["mean"]) == len(record["std"]) == 3
    model = linear_model((28, 28, 3), 10)
    expected = loader_probabilities(model, data.test_ds)
    gray = data.arrays.x_test  # (N, 28, 28, 3): the loader already repeated the channel; float in [0, 1]
    predictor_ = Predictor(model, record)
    files = []
    for index, image in enumerate(gray):
        path = tmp_path / f"digit_{index}.png"
        Image.fromarray(np.rint(image[..., 0] * 255).astype(np.uint8)).save(path)
        files.append(str(path))
    # Forms that are already one channel are exact; a colour image of grey pixels goes through a
    # luma conversion (weights summing to 0.9999 for an array, integer-rounded for PIL), which is
    # not the loader's grey read, so it is held to a looser bar.
    exact = (list(gray[..., :1]), [i[..., 0] for i in gray], [Image.open(f) for f in files], files)
    colour = ([Image.open(f).convert("RGB") for f in files], list(gray), gray)
    for given, atol in [(g, ATOL) for g in exact] + [(g, 1e-3) for g in colour]:
        result = predictor_.predict(given)
        np.testing.assert_allclose(
            in_class_order(result, record["class_names"]), expected, atol=atol, rtol=0
        )


def test_each_data_source_records_the_transform_its_evaluation_pipeline_applies():
    kinds = predictor.EVAL_KIND_OF_DATA
    assert {kinds[n] for n in ("mnist", "cifar10", "cifar100", "folder", "csv")} == {"resize"}
    assert kinds["imagenette"] == "resize_centre_crop"
    assert kinds["imagenet"] == "imagenet_centre_crop"
    assert set(kinds.values()) == set(EVAL_KINDS)  # every recorded kind is one the predictor reads
    from train.api.data_args import DATASETS, FORMATS

    assert set(kinds) == set(DATASETS) | {f for f in FORMATS if f != "builtin"}


# ---------------------------------------------------------------------
# A hand-made model with known logits: order, top_k, threshold, inputs
# ---------------------------------------------------------------------


def record(**changes):
    base = build_task_record(
        task="image_classification",
        model="vit/vit_pico",
        family="vit",
        variant="vit_pico",
        class_names=["a", "b", "c"],
        input_shape=(8, 8, 3),
        normalize="standardize",
        mean=[0.4, 0.5, 0.6],
        std=[0.2, 0.25, 0.3],
        data_kind="folder",
        source_channels=3,
    )
    base.update(changes)
    return base


def constant_model(bias, shape=(8, 8, 3)):
    """A model whose logits are ``bias`` for any input."""
    inputs = keras.Input(shape)
    outputs = keras.layers.Dense(len(bias), kernel_initializer="zeros", bias_initializer=keras.initializers.Constant(0.0))(
        keras.layers.Flatten()(inputs)
    )
    model = keras.Model(inputs, outputs)
    model.layers[-1].set_weights([np.zeros((int(np.prod(shape)), len(bias)), np.float32), np.asarray(bias, np.float32)])
    return model


def blank():
    return np.zeros((8, 8, 3), np.uint8)


def test_the_best_class_comes_first_and_scores_are_the_softmax_of_the_logits():
    result = Predictor(constant_model([0.0, 2.0, 1.0]), record()).predict(blank())
    expected = softmax(np.array([0.0, 2.0, 1.0]))
    assert result["labels"] == ["b", "c", "a"]
    np.testing.assert_allclose(result["scores"], expected[[1, 2, 0]], atol=1e-6, rtol=0)


def test_equal_scores_keep_the_class_order():
    result = Predictor(constant_model([0.0, 0.0, 0.0]), record()).predict(blank())
    assert result["labels"] == ["a", "b", "c"]


def test_equal_scores_keep_the_class_order_beyond_the_size_numpy_sorts_by_insertion():
    """Mutant ``stable sort off``: above 16 elements ``np.argsort`` is a quicksort and orders
    ties as it likes; three score levels over 40 classes make 13 or 14 ties at each level."""
    count = 40
    names = [f"c{index:02d}" for index in range(count)]
    predictor_ = Predictor(constant_model([float(i % 3) for i in range(count)]), record(class_names=names))
    expected = sorted(range(count), key=lambda i: (-(i % 3), i))
    if list(np.argsort(-(np.arange(count) % 3).astype(np.float64))) == expected:
        pytest.skip("this numpy happens to order these ties stably, so no mutant is visible")
    result = predictor_.predict(blank())
    assert result["labels"] == [names[i] for i in expected]


def test_a_class_exactly_at_the_threshold_is_kept():
    """Mutant ``threshold > not >=``: the docstring drops classes BELOW the threshold only."""
    predictor_ = Predictor(constant_model([0.0] * 4), record(class_names=list("abcd")))
    assert predictor_.predict(blank())["scores"] == [0.25] * 4  # exact in float32
    assert predictor_.predict(blank(), threshold=0.25)["labels"] == list("abcd")
    assert predictor_.predict(blank(), threshold=0.2500001)["labels"] == []


def test_top_k_keeps_the_best_k_and_a_larger_k_keeps_all():
    predictor_ = Predictor(constant_model([0.0, 2.0, 1.0]), record())
    assert predictor_.predict(blank(), top_k=1)["labels"] == ["b"]
    assert predictor_.predict(blank(), top_k=2)["labels"] == ["b", "c"]
    assert predictor_.predict(blank(), top_k=2)["scores"] == predictor_.predict(blank())["scores"][:2]
    assert predictor_.predict(blank(), top_k=99)["labels"] == ["b", "c", "a"]
    assert predictor_.predict([blank(), blank()], top_k=1)["labels"] == [["b"], ["b"]]


def test_threshold_drops_the_classes_below_it_and_may_leave_none():
    predictor_ = Predictor(constant_model([0.0, 2.0, 1.0]), record())  # 0.665, 0.245, 0.090
    assert predictor_.predict(blank(), threshold=0.2)["labels"] == ["b", "c"]
    assert predictor_.predict(blank(), threshold=0.0)["labels"] == ["b", "c", "a"]
    assert predictor_.predict(blank(), threshold=0.5, top_k=2)["labels"] == ["b"]
    assert predictor_.predict(blank(), threshold=1.0) == {"labels": [], "scores": []}
    assert predictor_.predict([blank()], threshold=1.0) == {"labels": [[]], "scores": [[]]}


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "2"])
def test_a_bad_top_k_is_refused(top_k):
    with pytest.raises(ConfigError, match="top_k"):
        Predictor(constant_model([0.0, 1.0, 2.0]), record()).predict(blank(), top_k=top_k)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, True, "0.5", float("nan")])
def test_a_bad_threshold_is_refused(threshold):
    with pytest.raises(ConfigError, match="threshold"):
        Predictor(constant_model([0.0, 1.0, 2.0]), record()).predict(blank(), threshold=threshold)


def test_a_4d_array_is_several_images_and_a_3d_array_is_one():
    predictor_ = Predictor(constant_model([0.0, 2.0, 1.0]), record())
    batch = np.zeros((3, 8, 8, 3), np.uint8)
    assert predictor_.predict(batch)["labels"] == [["b", "c", "a"]] * 3
    assert predictor_.predict(batch[0])["labels"] == ["b", "c", "a"]
    assert predictor_.predict(batch[0, ..., 0])["labels"] == ["b", "c", "a"]  # (H, W) is grey, repeated


@pytest.mark.parametrize(
    "given, pattern",
    [
        (np.full((8, 8, 3), 1.5, np.float32), r"\[0, 1\]"),
        (np.full((8, 8, 3), -0.1, np.float32), r"\[0, 1\]"),
        (np.full((8, 8, 3), np.nan, np.float32), r"\[0, 1\]"),
        (np.zeros((8, 8, 3), np.int64), "uint8 or floating"),
        (np.zeros((8, 8, 3), bool), "uint8 or floating"),
        (np.zeros((8, 8, 2), np.uint8), "shape"),
        (np.zeros((8, 8, 5), np.uint8), "shape"),
        (np.zeros((0, 8, 8, 3), np.uint8), "empty"),
        (np.zeros((0, 8, 3), np.uint8), "shape"),
        (np.zeros((8,), np.uint8), "got ndarray of shape"),
        (np.zeros((2, 2, 8, 8, 3), np.uint8), "got ndarray of shape"),
        ([], "empty"),
        (object(), "got object"),
        (3, "got int"),
        ("/no/such/file.png", "is not a file"),
    ],
    ids=["above", "below", "nan", "int64", "bool", "two-channels", "five-channels", "no-images",
         "no-rows", "1d", "5d", "empty", "object", "int", "missing"],
)
def test_an_unusable_image_is_refused_naming_why(given, pattern):
    with pytest.raises(ConfigError, match=pattern):
        Predictor(constant_model([0.0, 1.0, 2.0]), record()).predict(given)


def test_a_file_that_is_not_an_image_is_refused(tmp_path):
    path = tmp_path / "notes.png"
    path.write_text("not a png")
    with pytest.raises(ConfigError, match="not an image"):
        Predictor(constant_model([0.0, 1.0, 2.0]), record()).predict(str(path))


def test_a_rgb_array_for_a_one_channel_model_is_turned_grey_and_grey_is_repeated_for_rgb(tmp_path):
    model = linear_model((8, 8, 3), 3, seed=5)
    grey_record = record(eval_transform={"kind": "resize", "source_channels": 1})
    rgb = np.random.default_rng(0).integers(0, 255, (8, 8, 3)).astype(np.uint8)
    got = Predictor(model, grey_record).predict(rgb)
    import tensorflow as tf
    from PIL import Image

    from train.common.classification_data import decode_image_file

    # The expected grey is the LOADER's: the same pixels stored as a PNG and decoded with
    # channels=1 (not `tf.image.rgb_to_grayscale`, which is a level off in a quarter of them).
    Image.fromarray(rgb).save(tmp_path / "rgb.png")
    grey = decode_image_file(str(tmp_path / "rgb.png"), 1)
    batch = (tf.repeat(grey, 3, axis=-1)[None] - np.asarray(grey_record["mean"], np.float32)) / np.asarray(
        grey_record["std"], np.float32
    )
    expected = softmax(np.asarray(model(batch), np.float64))[0]
    np.testing.assert_allclose(in_class_order({"labels": [got["labels"]], "scores": [got["scores"]]}, ["a", "b", "c"])[0], expected, atol=ATOL, rtol=0)
    one = rgb[..., :1]
    again = Predictor(model, record()).predict(one)  # a 3-channel record reading a 1-channel array
    repeated = Predictor(model, record()).predict(np.repeat(one, 3, axis=-1))
    assert again == repeated


def test_the_centre_crop_kind_is_the_loaders_own_function():
    import tensorflow as tf

    from dl_techniques.datasets.vision.crops import resize_and_centre_crop

    model = linear_model((8, 8, 3), 3, seed=7)
    image = np.random.default_rng(1).integers(0, 255, (20, 33, 3)).astype(np.uint8)
    crop = Predictor(model, record(eval_transform={"kind": "resize_centre_crop", "source_channels": 3}))
    squash = Predictor(model, record())
    unit = tf.image.convert_image_dtype(tf.constant(image), tf.float32)
    mean, std = np.asarray(record()["mean"], np.float32), np.asarray(record()["std"], np.float32)
    batch = (resize_and_centre_crop(unit, 8)[None] - mean) / std
    expected = softmax(np.asarray(model(batch), np.float64))[0]
    np.testing.assert_allclose(crop.probabilities([unit])[0], expected, atol=ATOL, rtol=0)
    assert np.abs(squash.probabilities([unit])[0] - expected).max() > 1e-4  # the kind is live


def imagenet_loader_batch(path, side):
    """The batch ImageNet's own evaluation pipeline yields for one file (its ``_make_dataset``)."""
    from dl_techniques.datasets.vision import imagenet2012

    config = imagenet2012.ImageNet2012Config(
        image_size=side, batch_size=1, normalize="imagenet", augment=False, shuffle=False
    )
    dataset = imagenet2012._make_dataset(np.asarray([str(path)]), np.asarray([0], np.int32), False, config)
    (batch, _), = list(dataset.as_numpy_iterator())
    return batch


@pytest.mark.parametrize("extension", ["png", "jpg"])
@pytest.mark.parametrize("shape", [(60, 150), (150, 60), (120, 120)])
def test_an_imagenet_run_is_predicted_through_the_loaders_own_crop(tmp_path, shape, extension):
    from dl_techniques.datasets.vision.imagenet2012 import IMAGENET_MEAN, IMAGENET_STD

    side = 16
    pixels = np.random.default_rng(5).integers(0, 256, (*shape, 3)).astype(np.uint8)
    path = tmp_path / f"photo.{extension}"
    Image.fromarray(pixels).save(path)
    expected = imagenet_loader_batch(path, side)
    assert expected.shape == (1, side, side, 3)

    imagenet = build_task_record(
        task="image_classification", model="vit/vit_pico", family="vit", variant="vit_pico",
        class_names=["a", "b", "c"], input_shape=(side, side, 3), normalize="standardize",
        mean=IMAGENET_MEAN, std=IMAGENET_STD, data_kind="imagenet", source_channels=3,
    )
    check_record(imagenet, "<built>")  # the record the trainer writes is one the reader accepts
    assert imagenet["eval_transform"]["kind"] == "imagenet_centre_crop"
    model = linear_model((side, side, 3), 3, seed=11)
    given = Predictor(model, imagenet)

    # The preprocessing itself: the loader's normalized tensor, from the file, a PIL image and an array.
    mean, std = np.asarray(IMAGENET_MEAN, np.float32), np.asarray(IMAGENET_STD, np.float32)
    # (PIL's JPEG decoder differs from TensorFlow's by a few levels, so the PIL and array forms
    # are built from the pixels the loader decodes, and only the path form reads the file.)
    import tensorflow as tf

    decoded = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False).numpy()
    for form in (str(path), Image.fromarray(decoded), decoded):
        image = given._one(form)
        got = (np.asarray(given._shaped(image)) - mean) / std
        np.testing.assert_allclose(got, expected[0], atol=1e-4, rtol=0)

    # And through the model: the probabilities of the loader's batch.
    truth = softmax(np.asarray(model(expected, training=False), np.float64))[0]
    result = given.predict(str(path))
    by_name = dict(zip(result["labels"], result["scores"]))
    np.testing.assert_allclose([by_name[n] for n in "abc"], truth, atol=ATOL, rtol=0)

    # Liveness: the antialiased ImageNet crop is not the Imagenette crop, nor a plain resize.
    for other in ("resize_centre_crop", "resize"):
        moved = Predictor(model, {**imagenet, "eval_transform": {"kind": other, "source_channels": 3}})
        wrong = (np.asarray(moved._shaped(moved._one(str(path)))) - mean) / std
        assert np.abs(wrong - expected[0]).max() > 1e-2, other


def test_a_model_and_a_record_that_disagree_on_the_class_count_are_refused():
    with pytest.raises(ConfigError, match="names 3 classes"):
        Predictor(constant_model([0.0, 1.0, 2.0, 3.0]), record()).predict(blank())


def test_the_class_count_is_checked_at_load_not_at_the_first_predict(run_dir, tmp_path):
    two = copy_with_record(run_dir, tmp_path / "two", class_names=["ant", "moth"])
    with pytest.raises(ConfigError, match=r"outputs \(3,\) per image but task\.json names 2 classes"):
        load_model(two)
    assert load_model(copy_with_record(run_dir, tmp_path / "three")).class_names == CLASS_NAMES  # control


# ---------------------------------------------------------------------
# Arrays: RGBA, and nothing at all
# ---------------------------------------------------------------------


def test_an_rgba_array_is_read_like_an_rgba_pil_image_and_its_alpha_is_dropped():
    model = linear_model((8, 8, 3), 3, seed=7)
    predictor_ = Predictor(model, record())
    pixels = np.random.default_rng(1).integers(0, 255, (8, 8, 4)).astype(np.uint8)
    from_pil = predictor_.predict(Image.fromarray(pixels, "RGBA"))
    from_array = predictor_.predict(pixels)
    assert from_array == from_pil
    assert from_array == predictor_.predict(pixels[..., :3])  # the rgb part alone
    opaque = pixels.copy()
    opaque[..., 3] = 255 - opaque[..., 3]
    assert predictor_.predict(opaque) == from_array  # alpha is not used
    assert predictor_.predict((pixels.astype(np.float32) / 255.0)) == predictor_.predict(pixels.astype(np.float32)[..., :3] / 255.0)
    assert predictor_.predict(np.stack([pixels, pixels]))["labels"] == [from_array["labels"]] * 2
    grey = Predictor(model, record(eval_transform={"kind": "resize", "source_channels": 1}))
    assert grey.predict(pixels) == grey.predict(pixels[..., :3])  # grey from the rgb part, alpha unused


def test_an_empty_array_of_images_is_a_config_error_not_a_numpy_one():
    with pytest.raises(ConfigError, match=r"the array of images is empty \(shape \(0, 8, 8, 3\)\)"):
        Predictor(constant_model([0.0, 1.0, 2.0]), record()).predict(np.zeros((0, 8, 8, 3), np.float32))


# ---------------------------------------------------------------------
# A .keras file and the record beside it
# ---------------------------------------------------------------------


def test_a_keras_file_reads_its_name_task_json_then_its_stem_task_json_then_task_json(run_dir, tmp_path):
    record_ = read_task(run_dir)
    order = ["m.keras.task.json", "m.task.json", "task.json"]
    classes = {"m.keras.task.json": ["ant", "moth", "zebra"], "m.task.json": ["ant", "moth", "yak"], "task.json": ["ant", "moth", "wolf"]}
    for start, expected in enumerate(order):
        folder = tmp_path / f"beside{start}"  # holds this record and the ones after it in the order
        folder.mkdir()
        shutil.copy(run_dir / "best_model.keras", folder / "m.keras")
        for name in order[start:]:
            (folder / name).write_text(json.dumps({**record_, "class_names": classes[name]}))
        assert load_model(folder / "m.keras").class_names == classes[expected], expected
    bare = tmp_path / "bare"
    bare.mkdir()
    shutil.copy(run_dir / "best_model.keras", bare / "m.keras")
    with pytest.raises(ConfigError, match="has no record beside it"):
        load_model(bare / "m.keras")


# ---------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------


def test_a_record_built_by_the_trainer_passes_its_own_check():
    check_record(record(), "<test>")
    assert list(record()) == list(RECORD_KEYS)


def test_the_mean_and_std_of_the_loader_survive_json_exactly():
    mean = np.array([0.4914, 0.4822, 0.4465], np.float32)
    std = np.array([0.2470, 0.2435, 0.2616], np.float32)
    written = json.loads(json.dumps(record(mean=[float(v) for v in mean], std=[float(v) for v in std])))
    assert np.array_equal(np.asarray(written["mean"], np.float32), mean)
    assert np.array_equal(np.asarray(written["std"], np.float32), std)


def without(key):
    value = record()
    del value[key]
    return value


BAD_RECORDS = [
    (record(schema_version=2), r"'schema_version' is 2; this reader understands 1 only"),
    (record(schema_version=True), "schema_version"),
    (record(schema_version="1"), "schema_version"),
    (without("schema_version"), "schema_version"),
    (without("class_names"), "missing key.*class_names"),
    (record(extra=1), "unknown key.*extra"),
    (record(class_names=[]), "class_names"),
    (record(class_names=[1, 2, 3]), "class_names"),
    (record(model=""), "'model'"),
    (record(class_names=["a", "b", "a"]), "'class_names' repeats 'a'"),
    (record(family="resnet"), "'model' is 'vit/vit_pico' but 'family' and 'variant' give 'resnet/vit_pico'"),
    (record(variant="vit_tiny"), "'model' is 'vit/vit_pico' but"),
    (record(normalize="unit"), "'normalize' is 'unit'.*unit means zeros and ones"),
    (record(normalize="unit", mean=[0.0, 0.0, 0.0], std=[1.0, 2.0, 1.0]), "unit means zeros and ones"),
    (record(input_shape=[8, 8]), "input_shape"),
    (record(input_shape=[8, 9, 3]), "input_shape"),
    (record(input_shape=[0, 0, 3]), "input_shape"),
    (record(channels=1), "'channels' is 1 but 'input_shape' has 3"),
    (record(normalize="whiten"), "normalize"),
    (record(mean=[0.1, 0.2]), "'mean' holds 2 numbers for 3 channels"),
    (record(std=[0.1, 0.2, "x"]), "'std' must be a list of finite numbers"),
    (record(mean=[0.1, float("nan"), 0.3]), "'mean' must be a list of finite numbers"),
    (record(std=[0.2, 0.0, 0.3]), "'std' must be positive"),
    (record(eval_transform={"kind": "resize"}), "eval_transform"),
    (record(eval_transform={"kind": "resize", "source_channels": 2}), "source_channels"),
    (
        record(eval_transform={"kind": "stretch", "source_channels": 3}),
        "does not reproduce",
    ),
]


@pytest.mark.parametrize("bad, pattern", BAD_RECORDS, ids=[str(i) for i in range(len(BAD_RECORDS))])
def test_a_bad_record_is_refused_naming_the_key(bad, pattern):
    with pytest.raises(ConfigError, match=pattern):
        check_record(bad, "/some/task.json")
    with pytest.raises(ConfigError, match=pattern):
        Predictor(constant_model([0.0, 1.0, 2.0]), bad)


def test_a_unit_record_of_zeros_and_ones_and_a_standardize_record_with_any_numbers_pass():
    check_record(record(normalize="unit", mean=[0, 0, 0], std=[1, 1.0, 1]), "<test>")
    check_record(record(normalize="standardize", mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]), "<test>")


def test_a_record_that_is_not_an_object_is_refused():
    with pytest.raises(ConfigError, match="not a JSON object"):
        check_record([1, 2], "/some/task.json")


# ---------------------------------------------------------------------
# load_model: the forms it takes and what it refuses
# ---------------------------------------------------------------------


def test_a_run_directory_loads_its_best_model_and_a_keras_file_loads_beside_its_task_json(run_dir, truth):
    from_dir = load_model(run_dir)
    assert isinstance(from_dir.model, keras.Model)
    assert from_dir.class_names == CLASS_NAMES and from_dir.record == read_task(run_dir)
    best = keras.models.load_model(run_dir / "best_model.keras", compile=False)
    for weights, expected in zip(from_dir.model.get_weights(), best.get_weights()):
        assert np.array_equal(weights, expected)
    for name in ("best_model.keras", "final_model.keras"):
        from_file = load_model(run_dir / name)
        assert from_file.record == from_dir.record
        assert isinstance(from_file.predict(truth.paths[0])["labels"][0], str)
    assert load_model(str(run_dir)).class_names == CLASS_NAMES  # a str path


def test_a_loaded_model_is_not_compiled(run_dir):
    assert not load_model(run_dir).model.compiled


def test_a_path_that_does_not_exist_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="neither a run directory nor a file"):
        load_model(tmp_path / "nothing")


def test_a_model_file_without_a_task_json_beside_it_is_refused(run_dir, tmp_path):
    (tmp_path / "bare").mkdir()
    shutil.copy(run_dir / "best_model.keras", tmp_path / "bare" / "best_model.keras")
    with pytest.raises(ConfigError, match=r"has no record beside it.*best_model\.keras\.task\.json.*task\.json"):
        load_model(tmp_path / "bare" / "best_model.keras")


def test_a_run_directory_without_a_best_model_points_at_the_final_one(run_dir, tmp_path):
    other = tmp_path / "nobest"
    other.mkdir()
    shutil.copy(run_dir / "task.json", other / "task.json")
    shutil.copy(run_dir / "final_model.keras", other / "final_model.keras")
    with pytest.raises(ConfigError, match=r"holds no best_model\.keras.*final_model\.keras"):
        load_model(other)
    assert load_model(other / "final_model.keras").class_names == CLASS_NAMES


def test_an_unknown_schema_version_in_the_file_is_refused(run_dir, tmp_path):
    path = copy_with_record(run_dir, tmp_path / "v2", schema_version=2)
    with pytest.raises(ConfigError, match=r"schema_version.*understands 1 only"):
        load_model(path)


def test_a_task_json_that_is_not_json_is_refused(run_dir, tmp_path):
    path = copy_with_record(run_dir, tmp_path / "junk")
    (path.parent / "task.json").write_text("{not json")
    with pytest.raises(ConfigError, match="not JSON"):
        load_model(path)


def test_a_family_that_is_not_registered_is_refused(run_dir, tmp_path):
    path = copy_with_record(run_dir, tmp_path / "family", family="nosuchnet", model="nosuchnet/vit_pico")
    with pytest.raises(ConfigError, match="'nosuchnet' is not registered"):
        load_model(path)


def test_class_names_that_do_not_match_the_model_are_refused_when_it_is_loaded(run_dir, tmp_path, truth):
    path = copy_with_record(run_dir, tmp_path / "wide", class_names=CLASS_NAMES + ["extra"])
    with pytest.raises(ConfigError, match="names 4 classes"):
        load_model(path).predict(truth.paths[0])


# ---------------------------------------------------------------------
# A fresh process
# ---------------------------------------------------------------------

#: With ``train`` importable: the family's model code must be imported BEFORE Keras reads the file.
RELOAD = textwrap.dedent(
    """
    import json, sys
    import keras

    seen = []
    real = keras.models.load_model

    def checked(*args, **kwargs):
        seen.append("dl_techniques.models.vision.vit.model" in sys.modules)
        return real(*args, **kwargs)

    keras.models.load_model = checked
    from train.api.predictor import load_model

    result = load_model(sys.argv[1]).predict(sys.argv[2])
    print(json.dumps({"family_imported_first": seen, **result}))
    """
)

#: With ``train`` blocked: the model file alone loads with default arguments (D-051).
BLOCKED = textwrap.dedent(
    """
    import json, sys
    sys.modules["train"] = None
    import numpy as np
    import keras
    import dl_techniques

    model = keras.models.load_model(sys.argv[1])
    out = model(np.zeros((2, 32, 32, 3), "float32"), training=False)
    print(json.dumps({"shape": list(out.shape), "train_blocked": sys.modules["train"] is None}))
    """
)


def child(code, *arguments, tmp_path):
    done = subprocess.run(
        [sys.executable, "-c", code, *map(str, arguments)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "MPLBACKEND": "Agg", "TF_CPP_MIN_LOG_LEVEL": "3"},
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_a_fresh_process_reloads_the_run_with_the_family_imported_first(run_dir, truth, tmp_path):
    seen = child(RELOAD, run_dir, truth.paths[0], tmp_path=tmp_path)
    assert seen["family_imported_first"] == [True]
    here = load_model(run_dir).predict(truth.paths[0])
    assert seen["labels"] == here["labels"]
    np.testing.assert_allclose(seen["scores"], here["scores"], atol=ATOL, rtol=0)


@pytest.mark.parametrize("name", ["best_model.keras", "final_model.keras"])
def test_the_model_file_loads_with_train_blocked(run_dir, tmp_path, name):
    seen = child(BLOCKED, run_dir / name, tmp_path=tmp_path)
    assert seen == {"shape": [2, 3], "train_blocked": True}
