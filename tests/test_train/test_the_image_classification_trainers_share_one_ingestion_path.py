"""MNIST, CIFAR, Imagenette and ImageNet reach a trainer through ``train.common``'s loaders.

Before it, every labelled-image trainer ingested its data its own way: thirteen called
``train.common.load_dataset``, six more called ``keras.datasets`` themselves, about eleven
passed the TEST set as ``validation_data`` (the checkpoint was chosen on the number then
reported), and ImageNet's mean/std lived twice. This module holds the structure, by AST over
``src/train``:

* ``keras.datasets.<name>.load_data`` is called only by ``_load_raw`` (the seam every trainer
  test replaces) and by the named exceptions below, each with its reason;
* ``train.common`` no longer offers ``load_dataset`` and nothing imports one from it;
* no ``fit(...)`` is given a test split as ``validation_data``, directly or through a name
  bound to one (the old self-supervised pipeline validated on ``val_ds``, the labelled test
  split, and the name-only check could not see it);
* nothing under ``src/train`` names Imagenette's TFDS records: they are the library's, read
  through ``dl_techniques.datasets.vision.imagenette``;
* the self-supervised raw-image pipeline ``build_raw_image_dataset`` is gone, and its four
  trainers (dino, BEiT MIM and tokenizer, ET masked completion) read through
  ``train.common.image_streams.load_image_streams``, whose selection split is cut from train.

The constants' single home is guarded by ``tests/test_datasets/test_the_dataset_statistics_have_one_home.py``.
"""

import ast
from pathlib import Path

import pytest

import train.common as common

SRC_TRAIN = Path(common.__file__).resolve().parents[1]

#: ``keras.datasets`` callers outside the seam, each with the reason it stays.
KERAS_DATASET_EXCEPTIONS = {
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

#: The self-supervised trainers and the module whose loader they must call.
SELF_SUPERVISED_TRAINERS = (
    "dino/train_dino.py",
    "beit/train_mim.py",
    "beit/train_tokenizer.py",
    "energy_transformer/train_masked_completion.py",
)

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


def _mentions_a_test_split(source: str) -> bool:
    return any(name in source for name in TEST_NAMES)


def _names_bound_to_a_test_split(scope) -> set:
    """Names assigned (or tuple-unpacked) inside ``scope`` from an expression that mentions a
    test split."""
    tainted = set()
    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple) \
                    and len(target.elts) == len(node.value.elts):
                pairs = list(zip(target.elts, node.value.elts))
            else:
                pairs = [(target, node.value)]
            for element, value in pairs:
                if _mentions_a_test_split(ast.unparse(value)):
                    tainted.update(
                        n.id for n in ast.walk(element) if isinstance(n, ast.Name))
    return tainted


def _enclosing_scopes(tree):
    """Yield ``(call, scope)`` for every ``.fit(...)`` call, ``scope`` being the innermost
    function that contains it (the module when there is none). A name bound in another
    function is a different variable, so the taint never crosses a function boundary."""
    def visit(node, scope):
        for child in ast.iter_child_nodes(node):
            inner = child if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else scope
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) \
                    and child.func.attr == "fit":
                yield child, scope
            yield from visit(child, inner)

    yield from visit(tree, tree)


def _test_split_validation_calls(tree):
    hits = []
    for call, scope in _enclosing_scopes(tree):
        tainted = _names_bound_to_a_test_split(scope)
        for keyword in call.keywords:
            if keyword.arg != "validation_data":
                continue
            names = {n.id for n in ast.walk(keyword.value) if isinstance(n, ast.Name)}
            if _mentions_a_test_split(ast.unparse(keyword.value)) or names & tainted:
                hits.append(call.lineno)
    return hits


def test_the_census_sees_the_seam() -> None:
    # A broken walk would make every check below pass vacuously.
    calls = {name: _keras_dataset_calls(tree) for name, tree in _trees()}
    assert len(calls["common/classification_data.py"]) == 3
    assert sum(1 for lines in calls.values() if lines) >= 3


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


def _names_imagenette_records(tree) -> bool:
    return any(
        (isinstance(node, ast.Name) and node.id == "IMAGENETTE_TFDS_NAME")
        or (isinstance(node, ast.Attribute) and node.attr == "IMAGENETTE_TFDS_NAME")
        or (isinstance(node, ast.alias) and node.name == "IMAGENETTE_TFDS_NAME")
        for node in ast.walk(tree)
    )


def _mentions(tree, name: str):
    """Lines that define, import or reference ``name``."""
    return [
        getattr(node, "lineno", 0) for node in ast.walk(tree)
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
        or (isinstance(node, ast.Name) and node.id == name)
        or (isinstance(node, ast.Attribute) and node.attr == name)
        or (isinstance(node, ast.alias) and node.name == name)
    ]


def test_nothing_under_src_train_names_the_imagenette_records() -> None:
    offenders = [name for name, tree in _trees() if _names_imagenette_records(tree)]
    assert offenders == [], f"name the Imagenette records outside the library: {offenders}"


def test_the_old_raw_image_pipeline_is_gone() -> None:
    offenders = [
        f"{name}:{line}" for name, tree in _trees()
        for line in _mentions(tree, "build_raw_image_dataset")
    ]
    assert offenders == [], f"build_raw_image_dataset is deleted, yet: {offenders}"
    assert not hasattr(common, "build_raw_image_dataset")


def test_the_self_supervised_trainers_read_through_image_streams() -> None:
    trees = dict(_trees())
    missing = [name for name in SELF_SUPERVISED_TRAINERS
               if not _mentions(trees[name], "load_image_streams")]
    assert missing == [], f"these trainers do not call load_image_streams: {missing}"


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
    assert _names_imagenette_records(ast.parse("tfds.builder(IMAGENETTE_TFDS_NAME)"))
    assert not _names_imagenette_records(ast.parse("tfds.builder('mnist')"))


@pytest.mark.parametrize("source, expected", [
    # the pattern the old self-supervised trainers had: a name bound to the test split
    ("val_ds = streams.test_ds\nmodel.fit(x, validation_data=val_ds)", 1),
    ("train_ds, val_ds = a.train_ds, a.test_ds\nmodel.fit(x, validation_data=val_ds)", 1),
    ("val_ds = x_test\nmodel.fit(x, validation_data=(val_ds, y))", 1),
    # selection is fine, and so is a name that merely sits beside a test split
    ("val_ds = streams.select_ds\nmodel.fit(x, validation_data=val_ds)", 0),
    ("train_ds, val_ds = a.train_ds, a.select_ds\nmodel.fit(x, validation_data=val_ds)", 0),
    ("test_ds = a.test_ds\nmodel.fit(x, validation_data=val_ds)", 0),
])
def test_the_census_follows_a_name_bound_to_a_test_split(source, expected) -> None:
    assert len(_test_split_validation_calls(ast.parse(source))) == expected


def test_the_census_detects_the_removed_pipeline_and_the_stream_loader() -> None:
    assert _mentions(ast.parse("from train.common import build_raw_image_dataset"),
                     "build_raw_image_dataset")
    assert _mentions(ast.parse("def build_raw_image_dataset(): pass"), "build_raw_image_dataset")
    assert _mentions(ast.parse("x = load_image_streams('cifar10')"), "load_image_streams")
    assert not _mentions(ast.parse("x = load_classification_data('cifar10')"), "load_image_streams")
