"""The per-channel dataset statistics have one home each.

Guard for plan-2026-10-14T062616-81938e81 step 13.6 (c) (D-112), widened. Measured at
e259d9977: three CIFAR-100 mean/std pairs existed, two of them the often-quoted
(0.5071, 0.4867, 0.4408) / (0.2675, 0.2565, 0.2761) that the data does not reproduce.
Later, ImageNet's pair existed twice with two types (lists in ``train.common.datasets``,
tuples in ``imagenet2012``) and MNIST's only in ``power_mlp``.

The homes: CIFAR-10, CIFAR-100 and MNIST in ``dl_techniques.datasets.vision.classification``;
ImageNet in ``dl_techniques.datasets.vision.imagenet2012`` (the classification spec
references it). A trainer imports the names. This module checks that the consumers hold
the one object, that no literal copy of any pair lives elsewhere under ``src/``, and that
the constants are the training-set statistics.
"""

import ast
from pathlib import Path

import keras
import numpy as np
import pytest

import dl_techniques.datasets.vision.classification as classification
import dl_techniques.datasets.vision.imagenet2012 as imagenet2012
import train.cliffordnet.train_cliffordnet as cliffordnet
import train.cliffordnet.train_downsampling_techniques as downsampling
import train.clip.train_clip as clip
import train.common.megadepth as megadepth
import train.power_mlp.train_power_mlp as power_mlp
import train.rad_convnet.data as rad_data

SRC = Path(classification.__file__).resolve().parents[3]

HOMES = {
    Path(classification.__file__).resolve(),
    Path(imagenet2012.__file__).resolve(),
}

#: Files that hold ImageNet's numbers for another purpose and must keep their own copy
#: (they sit below ``train`` in the import graph, or normalize a pretrained backbone's
#: input): the reason is the point of listing them.
ALLOWED_COPIES = {
    "dl_techniques/losses/lpips_loss.py": "VGG input normalization of a loss; a loss does "
                                          "not import dataset code",
    "dl_techniques/losses/image_restoration_loss.py": "VGG input normalization of a loss",
    "dl_techniques/datasets/pusht_hdf5.py": "its own tested constant "
                                            "(tests/test_datasets/test_pusht_hdf5.py)",
}

#: Four-place values of every pair (data-derived and often-quoted variants alike).
VALUES = {
    "cifar10": {0.4914, 0.4822, 0.4465, 0.2470, 0.2435, 0.2616},
    "cifar100": {0.5071, 0.4865, 0.4867, 0.4409, 0.4408, 0.2673, 0.2675, 0.2564, 0.2565,
                 0.2762, 0.2761},
    "mnist": {0.1307, 0.3081},
    "imagenet": {0.485, 0.456, 0.406, 0.229, 0.224, 0.225},
}


def _copies(tree: ast.AST, dataset: str) -> list:
    """Line numbers of a literal copy of the pair: a tuple or list holding two or more of
    its values (three channels), or, for one-channel MNIST, a bare float equal to one.
    Docstrings and comments are not code and are not scanned."""
    values = VALUES[dataset]
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Tuple, ast.List)):
            held = [e for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, float)
                    and e.value in values]
            if len(held) >= 2:
                hits.append(node.lineno)
        elif (dataset == "mnist" and isinstance(node, ast.Constant)
              and isinstance(node.value, float) and node.value in values):
            hits.append(node.lineno)
    return hits


CONSUMERS = [
    ("cifar100", cliffordnet, "CIFAR100"),
    ("cifar100", downsampling, "CIFAR100"),
    ("cifar10", cliffordnet, "CIFAR10"),
    ("cifar10", power_mlp, "CIFAR10"),
    ("mnist", power_mlp, "MNIST"),
]


@pytest.mark.parametrize("dataset, module, prefix", CONSUMERS,
                         ids=[f"{d}-{m.__name__}" for d, m, _ in CONSUMERS])
def test_each_consumer_holds_the_one_object(dataset, module, prefix):
    assert getattr(module, f"{prefix}_MEAN") is getattr(classification, f"{prefix}_MEAN")
    assert getattr(module, f"{prefix}_STD") is getattr(classification, f"{prefix}_STD")


@pytest.mark.parametrize("module", [clip, megadepth, rad_data],
                         ids=lambda m: m.__name__)
def test_every_imagenet_consumer_holds_the_one_object(module):
    assert module.IMAGENET_MEAN is imagenet2012.IMAGENET_MEAN
    assert module.IMAGENET_STD is imagenet2012.IMAGENET_STD


def test_the_specs_reference_the_one_object():
    spec = classification.CLASSIFICATION_DATASETS
    assert spec["imagenet"].mean is imagenet2012.IMAGENET_MEAN
    assert spec["imagenet"].std is imagenet2012.IMAGENET_STD
    # Imagenette's images are ImageNet's, so it normalizes with the same pair.
    assert spec["imagenette"].mean is imagenet2012.IMAGENET_MEAN
    assert spec["imagenette"].std is imagenet2012.IMAGENET_STD
    for name, prefix in (("cifar10", "CIFAR10"), ("cifar100", "CIFAR100"), ("mnist", "MNIST")):
        assert spec[name].mean is getattr(classification, f"{prefix}_MEAN")
        assert spec[name].std is getattr(classification, f"{prefix}_STD")


@pytest.mark.parametrize("dataset", sorted(VALUES))
def test_no_copy_of_a_pair_lives_outside_its_home(dataset):
    copies = []
    for path in sorted(SRC.rglob("*.py")):
        relative = str(path.relative_to(SRC))
        if path.resolve() in HOMES or relative in ALLOWED_COPIES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        copies += [f"{relative}:{number}" for number in _copies(tree, dataset)]
    assert copies == []


def test_the_copy_detector_can_fail():
    tree = ast.parse("a = np.array([0.485, 0.456, 0.406])\nb = (0.2470, 0.2435, 0.2616)\n"
                     "c = 0.1307\nd = [0.2, 0.3]\n")
    assert _copies(tree, "imagenet") == [1]
    assert _copies(tree, "cifar10") == [2]
    assert _copies(tree, "mnist") == [3]
    assert _copies(ast.parse('"""[0.485, 0.456, 0.406] 0.1307"""'), "imagenet") == []


def test_the_allowed_copies_exist_and_hold_what_they_claim():
    for relative in ALLOWED_COPIES:
        tree = ast.parse((SRC / relative).read_text(encoding="utf-8"))
        assert _copies(tree, "imagenet"), f"{relative} no longer holds a copy"


def test_the_cliffordnet_pipeline_normalises_with_the_cifar100_pair():
    mean, std = cliffordnet._normalization_stats("cifar100", None)
    np.testing.assert_array_equal(mean, np.asarray(classification.CIFAR100_MEAN, np.float32))
    np.testing.assert_array_equal(std, np.asarray(classification.CIFAR100_STD, np.float32))


@pytest.mark.parametrize("name, loader, channels", [
    ("cifar10", keras.datasets.cifar10.load_data, 3),
    ("cifar100", keras.datasets.cifar100.load_data, 3),
    ("mnist", keras.datasets.mnist.load_data, 1),
])
def test_the_constants_are_the_training_set_statistics(name, loader, channels):
    # The population per-channel mean/std of the training images in [0, 1], float64
    # (D-103); each constant keeps 4 places.
    (x, _), _ = loader()
    x = x.reshape(-1, channels).astype(np.float64) / 255.0
    spec = classification.CLASSIFICATION_DATASETS[name]
    np.testing.assert_allclose(spec.mean, x.mean(axis=0), rtol=0, atol=5e-5)
    np.testing.assert_allclose(spec.std, x.std(axis=0), rtol=0, atol=5e-5)


def test_the_specs_are_internally_consistent():
    for name, spec in classification.CLASSIFICATION_DATASETS.items():
        assert spec.name == name
        assert len(spec.mean) == len(spec.std) == spec.input_shape[-1] or name == "mnist"
        if spec.class_names is not None:
            assert len(spec.class_names) == spec.num_classes
            assert len(set(spec.class_names)) == spec.num_classes
    assert classification.get_spec("CIFAR10") is classification.CLASSIFICATION_DATASETS["cifar10"]
    with pytest.raises(ValueError, match="got 'svhn'"):
        classification.get_spec("svhn")


def test_unknown_class_counts_fall_back_to_placeholders():
    spec = classification.CLASSIFICATION_DATASETS["cifar100"]
    assert spec.class_names_for(100)[0] == "apple"
    assert spec.class_names_for(20) == [f"class_{i}" for i in range(20)]


# ---------------------------------------------------------------------
# The crop recipe's constants have one home too
# ---------------------------------------------------------------------

CROPS_HOME = (SRC / "dl_techniques" / "datasets" / "vision" / "crops.py").resolve()


def _crop_constant_copies(tree: ast.AST) -> list:
    """Lines of a literal ``0.875`` or of the ``(0.08, 1.0)`` area range."""
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 0.875:
            hits.append(node.lineno)
        elif isinstance(node, (ast.Tuple, ast.List)):
            values = [e.value for e in node.elts if isinstance(e, ast.Constant)]
            if values == [0.08, 1.0]:
                hits.append(node.lineno)
    return hits


def test_the_crop_recipe_constants_are_defined_only_in_crops():
    offenders = []
    for root in (SRC / "dl_techniques" / "datasets", SRC / "train"):
        for path in sorted(root.rglob("*.py")):
            if path.resolve() == CROPS_HOME:
                continue
            offenders += [f"{path.relative_to(SRC)}:{line}"
                          for line in _crop_constant_copies(ast.parse(path.read_text()))]
    assert offenders == [], f"literal copies of the crop recipe outside crops.py: {offenders}"


def test_the_crop_census_sees_the_definition_and_a_copy():
    assert _crop_constant_copies(ast.parse(CROPS_HOME.read_text()))
    assert _crop_constant_copies(ast.parse("x = 0.875")) == [1]
    assert _crop_constant_copies(ast.parse("area_range=(0.08, 1.0)")) == [1]
    assert _crop_constant_copies(ast.parse("x = (0.08, 0.12)")) == []
