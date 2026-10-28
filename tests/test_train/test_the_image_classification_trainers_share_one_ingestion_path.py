"""MNIST, CIFAR and ImageNet reach a trainer through ``train.common.classification_data``.

Before it, every labelled-image trainer ingested its data its own way: thirteen called
``train.common.load_dataset``, six more called ``keras.datasets`` themselves, about eleven
passed the TEST set as ``validation_data`` (the checkpoint was chosen on the number then
reported), and ImageNet's mean/std lived twice. This module holds the structure, by AST over
``src/train``:

* ``keras.datasets.<name>.load_data`` is called only by ``_load_raw`` (the seam every trainer
  test replaces) and by the named exceptions below, each with its reason;
* ``train.common`` no longer offers ``load_dataset`` and nothing imports one from it;
* no ``fit(...)`` is given a test split as ``validation_data``.

The constants' single home is guarded by ``tests/test_datasets/test_the_dataset_statistics_have_one_home.py``.
"""

import ast
from pathlib import Path

import pytest

import train.common as common

SRC_TRAIN = Path(common.__file__).resolve().parents[1]

#: ``keras.datasets`` callers outside the seam, each with the reason it stays.
KERAS_DATASET_EXCEPTIONS = {
    "common/datasets.py": "build_raw_image_dataset's CIFAR-10 branch: the self-supervised "
                          "(dino, beit, energy_transformer) pipeline with its own crop policies",
    "head_comparison/train_heads.py": "CIFAR-100's coarse labels, a second label set aligned "
                                      "through the loader's row indices",
    "rms_variants_train/experiments/e3_tinytransformer_imdb.py": "IMDB text, not images",
}

#: ``fit(validation_data=<test split>)`` callers, each with the reason it stays.
TEST_AS_VALIDATION_EXCEPTIONS = {
    "logic/train_boolean_circuit.py": "a synthetic boolean-circuit task, not an image dataset",
    "logic/train_benchmark.py": "synthetic boolean tasks (parity, majority, multiplexer), "
                                "not an image dataset",
}

TEST_NAMES = ("x_test", "y_test", "test_ds", "test_x", "X_test")


def _trees():
    for path in sorted(SRC_TRAIN.rglob("*.py")):
        yield str(path.relative_to(SRC_TRAIN)), ast.parse(path.read_text(encoding="utf-8"))


def _keras_dataset_calls(tree):
    return [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load_data"
        and ast.unparse(node.func).startswith("keras.datasets.")
    ]


def _test_split_validation_calls(tree):
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "fit":
            for keyword in node.keywords:
                if keyword.arg == "validation_data" and any(
                        name in ast.unparse(keyword.value) for name in TEST_NAMES):
                    hits.append(node.lineno)
    return hits


def test_the_census_sees_the_seam() -> None:
    # A broken walk would make every check below pass vacuously.
    calls = {name: _keras_dataset_calls(tree) for name, tree in _trees()}
    assert len(calls["common/classification_data.py"]) == 3
    assert sum(1 for lines in calls.values() if lines) >= 4


def test_only_the_seam_and_the_named_exceptions_read_keras_datasets() -> None:
    offenders = [
        f"{name}:{line}" for name, tree in _trees()
        for line in _keras_dataset_calls(tree)
        if name != "common/classification_data.py" and name not in KERAS_DATASET_EXCEPTIONS
    ]
    assert offenders == [], f"read keras.datasets outside the seam: {offenders}"


def test_every_exception_still_reads_keras_datasets() -> None:
    calls = {name: _keras_dataset_calls(tree) for name, tree in _trees()}
    stale = [name for name in KERAS_DATASET_EXCEPTIONS if not calls.get(name)]
    assert stale == [], f"exceptions that no longer need their entry: {stale}"


def test_train_common_no_longer_offers_load_dataset() -> None:
    assert not hasattr(common, "load_dataset") and "load_dataset" not in common.__all__
    importers = [
        f"{name}:{node.lineno}" for name, tree in _trees() for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("train.common")
        and any(alias.name == "load_dataset" for alias in node.names)
    ]
    assert importers == []


def test_no_fit_is_given_a_test_split_as_validation_data() -> None:
    offenders = [
        f"{name}:{line}" for name, tree in _trees()
        for line in _test_split_validation_calls(tree)
        if name not in TEST_AS_VALIDATION_EXCEPTIONS
    ]
    assert offenders == [], f"test split used as validation_data: {offenders}"


def test_every_validation_exception_still_applies() -> None:
    hits = {name: _test_split_validation_calls(tree) for name, tree in _trees()}
    stale = [name for name in TEST_AS_VALIDATION_EXCEPTIONS if not hits.get(name)]
    assert stale == [], f"exceptions that no longer need their entry: {stale}"


@pytest.mark.parametrize("source, expected", [
    ("keras.datasets.mnist.load_data()", 1),
    ("x = keras.datasets.cifar10.load_data()", 1),
    ("other.load_data()", 0),
])
def test_the_census_detects_what_it_claims(source, expected) -> None:
    assert len(_keras_dataset_calls(ast.parse(source))) == expected
    assert len(_test_split_validation_calls(
        ast.parse("model.fit(x, y, validation_data=(x_test, y_test))"))) == 1
    assert _test_split_validation_calls(
        ast.parse("model.fit(x, y, validation_data=(x_val, y_val))")) == []
