"""``dl_techniques.datasets.vision.classification.CIFAR100_CLASS_NAMES`` and ``DatasetSpec.class_names_for`` (audit H3).

CIFAR-100 figures carried ``class_N`` placeholders. The names are the fine labels in the
order of ``keras.datasets.cifar100`` (alphabetical); the spot checks pin that order, so a
sorted-differently or mistyped tuple fails.
"""

import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest  # noqa: E402

from dl_techniques.datasets.vision.classification import CIFAR100_CLASS_NAMES, get_spec  # noqa: E402


def test_there_are_100_distinct_names_in_alphabetical_order() -> None:
    assert len(CIFAR100_CLASS_NAMES) == 100
    assert len(set(CIFAR100_CLASS_NAMES)) == 100
    assert list(CIFAR100_CLASS_NAMES) == sorted(CIFAR100_CLASS_NAMES)


KERAS_CIFAR100_META = Path.home() / ".keras/datasets/cifar-100-python-target/cifar-100-python/meta"


@pytest.mark.skipif(not KERAS_CIFAR100_META.is_file(), reason="Keras CIFAR-100 cache is absent")
def test_the_whole_tuple_equals_the_fine_labels_in_the_keras_cache_meta_file() -> None:
    """The ground truth is on disk: 100 pinned names instead of 10 spot checks (review-iter-3 C3).

    A sort-order-preserving typo such as ``sweetpepper`` for ``sweet_pepper`` keeps the tuple
    sorted, distinct and 100 long, so only a comparison with the dataset's own metadata sees it.
    """
    with open(KERAS_CIFAR100_META, "rb") as handle:
        meta = pickle.load(handle, encoding="latin1")
    assert list(CIFAR100_CLASS_NAMES) == list(meta["fine_label_names"])


@pytest.mark.parametrize("index,name", [
    (0, "apple"), (1, "aquarium_fish"), (3, "bear"), (19, "cattle"), (47, "maple_tree"),
    (58, "pickup_truck"), (69, "rocket"), (88, "tiger"), (98, "woman"), (99, "worm"),
])
def test_the_keras_fine_label_order(index, name) -> None:
    assert CIFAR100_CLASS_NAMES[index] == name


def test_the_spec_returns_the_cifar100_names() -> None:
    names = get_spec("cifar100").class_names_for(100)
    assert names == list(CIFAR100_CLASS_NAMES)
    assert isinstance(names, list), "callers treat it as a list; the tuple must not leak"
    assert get_spec("CIFAR100").class_names_for(100)[47] == "maple_tree"


@pytest.mark.parametrize("num_classes", [20, 99, 101])
def test_a_wrong_class_count_falls_back_to_placeholders(num_classes) -> None:
    assert get_spec("cifar100").class_names_for(num_classes) == [
        f"class_{i}" for i in range(num_classes)]


def test_the_other_datasets_name_their_classes() -> None:
    assert get_spec("mnist").class_names_for(10) == [str(i) for i in range(10)]
    assert get_spec("cifar10").class_names_for(10)[:3] == ["airplane", "automobile", "bird"]
    assert get_spec("imagenet").class_names_for(1000)[:2] == ["class_0", "class_1"]
