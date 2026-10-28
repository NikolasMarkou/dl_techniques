"""The labelled-image datasets the classification trainers share, and their constants.

One :class:`DatasetSpec` per dataset (``mnist``, ``cifar10``, ``cifar100``, ``imagenet``)
carries the class count, the native input shape, the per-channel mean and standard
deviation of the [0, 1]-scaled training images and the class names. This module is the
ONE home of the CIFAR and MNIST statistics; the ImageNet statistics are defined once in
:mod:`dl_techniques.datasets.vision.imagenet2012` and referenced here. A trainer imports
the names, it does not keep a literal copy (a guard scans for one).

The statistics are the population mean and standard deviation of every pixel of the
training images scaled to [0, 1], computed in float64 from ``keras.datasets``
(CIFAR-100 on 2026-10-16, CIFAR-10 and MNIST on 2026-10-28; a test recomputes them).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from dl_techniques.datasets.vision.imagenet2012 import (
    IMAGENET2012_NUM_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from dl_techniques.datasets.vision.imagenette import (
    IMAGENETTE_CLASS_NAMES,
    IMAGENETTE_NUM_CLASSES,
    IMAGENETTE_TRAIN_SIZE,
)

# ---------------------------------------------------------------------
# Per-channel normalisation constants. The CIFAR-10 pair is computed from the CIFAR-10
# training set. These are DISTINCT from the OpenAI CLIP IMAGE_MEAN/IMAGE_STD in
# train/common/image_text.py (those are [0.48145466, ...]) and from the ImageNet pair:
# do not conflate them. Plain tuples, so each call site wraps with np.array(...) when it
# needs array broadcasting.
# ---------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# CIFAR-100 per-channel mean/std, derived from the data the same way as the
# CIFAR-10 pair above (which it reproduces to 4 places): the population mean
# and standard deviation of every pixel of the 50,000 training images scaled
# to [0, 1], computed in float64 (2026-10-16, `keras.datasets.cifar100`):
# mean 0.50707516 0.48654887 0.44091784, std 0.26733429 0.25643846 0.27615047.
# The often-quoted (0.2675, 0.2565, 0.2761) / 0.4867 do not come out of this
# computation.
# DECISION plan-2026-10-14T062616-81938e81/D-112
# The ONE home of the CIFAR-100 statistics: trainers import these names, they
# do NOT keep a literal copy (a guard scans for one), and do NOT
# "correct" them to the often-quoted values above.
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

# MNIST is one channel (60,000 training images, 2026-10-28: mean 0.13066048, std
# 0.30810780); a trainer that feeds a three-channel model repeats the channel.
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


# ---------------------------------------------------------------------
# Class names
# ---------------------------------------------------------------------

CIFAR10_CLASS_NAMES: Tuple[str, ...] = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

MNIST_CLASS_NAMES: Tuple[str, ...] = tuple(str(i) for i in range(10))

# The 100 CIFAR-100 fine label names, in the order of ``keras.datasets.cifar100``
# (``label_mode="fine"`` integer labels are indices into this tuple, and that order is
# alphabetical).
CIFAR100_CLASS_NAMES: Tuple[str, ...] = (
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle",
    "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle",
    "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch",
    "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom",
    "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road",
    "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle",
    "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
)


# ---------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """What a trainer needs to know about a labelled-image dataset before loading it.

    :param name: Dataset key, as used by every trainer's ``--dataset``.
    :param num_classes: Number of classes.
    :param input_shape: Native ``(H, W, C)``. ImageNet's ``(224, 224, 3)`` is the default
        crop; the loader takes ``image_size``.
    :param mean: Per-channel mean of the [0, 1]-scaled training images.
    :param std: Per-channel standard deviation of the same.
    :param class_names: One name per class, or ``None`` when they come from the data
        (ImageNet: the wnids of the extracted tree).
    :param source: Where the images come from: ``"arrays"`` (read whole into memory: MNIST,
        CIFAR), ``"records"`` (prepared TFDS records: Imagenette) or ``"tree"`` (an extracted
        directory tree: ImageNet).
    :param train_size: Number of training images (before any held-out slice).
    """

    name: str
    num_classes: int
    input_shape: Tuple[int, int, int]
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    class_names: Optional[Tuple[str, ...]]
    source: str
    train_size: int

    @property
    def in_memory(self) -> bool:
        """``True`` when the dataset is read whole into arrays (MNIST, CIFAR)."""
        return self.source == "arrays"

    def class_names_for(self, num_classes: int) -> List[str]:
        """The class names, or ``class_N`` placeholders when ``num_classes`` is not the
        dataset's own (a subset or a different label mode must not be mislabelled)."""
        if self.class_names is not None and num_classes == len(self.class_names):
            return list(self.class_names)
        return [f"class_{i}" for i in range(num_classes)]


CLASSIFICATION_DATASETS: Dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        "mnist", 10, (28, 28, 1), MNIST_MEAN, MNIST_STD, MNIST_CLASS_NAMES, "arrays", 60000),
    "cifar10": DatasetSpec(
        "cifar10", 10, (32, 32, 3), CIFAR10_MEAN, CIFAR10_STD, CIFAR10_CLASS_NAMES, "arrays", 50000),
    "cifar100": DatasetSpec(
        "cifar100", 100, (32, 32, 3), CIFAR100_MEAN, CIFAR100_STD, CIFAR100_CLASS_NAMES, "arrays", 50000),
    "imagenet": DatasetSpec(
        "imagenet", IMAGENET2012_NUM_CLASSES, (224, 224, 3), IMAGENET_MEAN, IMAGENET_STD,
        None, "tree", 1281167),
    # Imagenette's images are ImageNet's, so ImageNet's statistics apply.
    "imagenette": DatasetSpec(
        "imagenette", IMAGENETTE_NUM_CLASSES, (224, 224, 3), IMAGENET_MEAN, IMAGENET_STD,
        IMAGENETTE_CLASS_NAMES, "records", IMAGENETTE_TRAIN_SIZE),
}


def get_spec(name: str) -> DatasetSpec:
    """The :class:`DatasetSpec` of ``name`` (case-insensitive).

    :param name: A key of :data:`CLASSIFICATION_DATASETS`.
    :raises ValueError: If ``name`` is not one of them.
    """
    key = name.lower()
    if key not in CLASSIFICATION_DATASETS:
        raise ValueError(
            f"dataset must be one of {tuple(CLASSIFICATION_DATASETS)}, got {name!r}"
        )
    return CLASSIFICATION_DATASETS[key]
