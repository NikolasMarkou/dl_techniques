"""The CIFAR-100 channel statistics have one home, ``train.common.datasets``.

Guard for plan-2026-10-14T062616-81938e81 step 13.6 (c) (D-112). Measured at
e259d9977: three CIFAR-100 mean/std pairs existed. ``train.common`` held the
data-derived pair (D-103, used by AlexNet); ``train_cliffordnet._CIFAR_STATS``
and the round-trip check in ``train_downsampling_techniques`` each hard-coded
the often-quoted (0.5071, 0.4867, 0.4408) / (0.2675, 0.2565, 0.2761), which the
data does not reproduce. All three trainers now import the one pair.
"""

import re
from pathlib import Path

import keras
import numpy as np
import pytest

import train.alexnet.train_alexnet as alexnet
import train.cliffordnet.train_cliffordnet as cliffordnet
import train.cliffordnet.train_downsampling_techniques as downsampling
import train.common.datasets as datasets

SRC_TRAIN = Path(datasets.__file__).resolve().parents[1]

CONSUMERS = {
    "train/alexnet/train_alexnet.py": alexnet,
    "train/cliffordnet/train_cliffordnet.py": cliffordnet,
    "train/cliffordnet/train_downsampling_techniques.py": downsampling,
}

#: Both CIFAR-100 variants' digits, to four places: the data-derived pair and
#: the often-quoted one. A copy of either outside the home is a second source.
STATISTIC_LITERAL = re.compile(
    r"\b0\.(5071|4865|4867|4409|4408|2673|2675|2564|2565|2762|2761)")


@pytest.mark.parametrize("path", sorted(CONSUMERS))
def test_each_consumer_holds_the_one_object(path):
    module = CONSUMERS[path]
    assert module.CIFAR100_MEAN is datasets.CIFAR100_MEAN
    assert module.CIFAR100_STD is datasets.CIFAR100_STD


def test_no_copy_of_the_statistics_lives_outside_the_home():
    home = Path(datasets.__file__).resolve()
    copies = [
        f"{path.relative_to(SRC_TRAIN)}:{number}"
        for path in sorted(SRC_TRAIN.rglob("*.py")) if path.resolve() != home
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if STATISTIC_LITERAL.search(line)
    ]
    assert copies == []


def test_the_cliffordnet_pipeline_normalises_with_them():
    mean, std = cliffordnet._normalization_stats("cifar100", None)
    np.testing.assert_array_equal(mean, np.asarray(datasets.CIFAR100_MEAN, np.float32))
    np.testing.assert_array_equal(std, np.asarray(datasets.CIFAR100_STD, np.float32))


def test_the_cifar100_statistics_are_the_training_set_statistics():
    # Re-derived from the data (D-103): the population per-channel mean/std of
    # the 50,000 training images in [0, 1], float64; the constant keeps 4 places.
    (x, _), _ = keras.datasets.cifar100.load_data()
    x = x.reshape(-1, 3).astype(np.float64) / 255.0
    np.testing.assert_allclose(datasets.CIFAR100_MEAN, x.mean(axis=0), rtol=0, atol=5e-5)
    np.testing.assert_allclose(datasets.CIFAR100_STD, x.std(axis=0), rtol=0, atol=5e-5)
