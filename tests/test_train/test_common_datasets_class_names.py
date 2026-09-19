"""``train.common.datasets.CIFAR100_CLASS_NAMES`` and ``get_class_names`` (audit H3).

CIFAR-100 figures carried ``class_N`` placeholders. The names are the fine labels in the
order of ``keras.datasets.cifar100`` (alphabetical); the spot checks pin that order, so a
sorted-differently or mistyped tuple fails.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest  # noqa: E402

from train.common.datasets import CIFAR100_CLASS_NAMES, get_class_names  # noqa: E402


def test_there_are_100_distinct_names_in_alphabetical_order() -> None:
    assert len(CIFAR100_CLASS_NAMES) == 100
    assert len(set(CIFAR100_CLASS_NAMES)) == 100
    assert list(CIFAR100_CLASS_NAMES) == sorted(CIFAR100_CLASS_NAMES)


@pytest.mark.parametrize("index,name", [
    (0, "apple"), (1, "aquarium_fish"), (3, "bear"), (19, "cattle"), (47, "maple_tree"),
    (58, "pickup_truck"), (69, "rocket"), (88, "tiger"), (98, "woman"), (99, "worm"),
])
def test_the_keras_fine_label_order(index, name) -> None:
    assert CIFAR100_CLASS_NAMES[index] == name


def test_get_class_names_returns_the_cifar100_names() -> None:
    names = get_class_names("cifar100", 100)
    assert names == list(CIFAR100_CLASS_NAMES)
    assert isinstance(names, list), "callers treat it as a list; the tuple must not leak"
    assert get_class_names("CIFAR100", 100)[47] == "maple_tree"


@pytest.mark.parametrize("num_classes", [20, 99, 101])
def test_a_wrong_class_count_falls_back_to_placeholders(num_classes) -> None:
    assert get_class_names("cifar100", num_classes) == [f"class_{i}" for i in range(num_classes)]


def test_the_other_datasets_are_unchanged() -> None:
    assert get_class_names("cifar10", 10)[3] == "cat"
    assert get_class_names("mnist", 10) == [str(i) for i in range(10)]
    assert get_class_names("unknown", 3) == ["class_0", "class_1", "class_2"]
