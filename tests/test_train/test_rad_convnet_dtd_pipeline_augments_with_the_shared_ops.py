"""``train.rad_convnet.data.make_dtd_dataset`` crops and flips through ``train.common.image_augment``.

A 2-class fake DTD tree: the train pipeline resizes to 1.15x, takes a random window and flips
it; the val pipeline resizes only. Nothing pinned either before the crop and flip moved to the
shared ops.
"""

import numpy as np
import pytest
from PIL import Image

from train.rad_convnet.data import make_dtd_dataset

SIZE = 16


@pytest.fixture(scope="module")
def dtd(tmp_path_factory):
    root = tmp_path_factory.mktemp("dtd")
    names = []
    rng = np.random.default_rng(0)
    for cls in ("banded", "dotted"):
        (root / "images" / cls).mkdir(parents=True)
        for i in range(4):
            Image.fromarray((rng.random((40, 40, 3)) * 255).astype(np.uint8)).save(
                root / "images" / cls / f"{cls}_{i}.jpg")
            names.append(f"{cls}/{cls}_{i}.jpg")
    (root / "labels").mkdir()
    (root / "labels" / "train1.txt").write_text("\n".join(names))
    (root / "labels" / "val1.txt").write_text("\n".join(names[:4]))
    return str(root)


def _first(dataset):
    return next(iter(dataset.as_numpy_iterator()))


def test_the_train_pipeline_yields_cropped_augmented_batches(dtd):
    ds, classes, n = make_dtd_dataset(dtd, "train", SIZE, 4, augment=True, shuffle_buffer=8)
    images, labels = _first(ds)
    assert images.shape == (4, SIZE, SIZE, 3) and n == 8 and classes == ["banded", "dotted"]
    again = _first(make_dtd_dataset(dtd, "train", SIZE, 4, augment=True, shuffle_buffer=8)[0])[0]
    assert images.shape == again.shape


def test_two_passes_over_the_same_image_crop_differently(dtd):
    ds, _, _ = make_dtd_dataset(dtd, "train", SIZE, 8, augment=True, shuffle_buffer=1)
    first = np.concatenate([x for x, _ in ds.as_numpy_iterator()])
    second = np.concatenate([x for x, _ in ds.as_numpy_iterator()])
    assert not np.array_equal(first, second)


def test_the_val_pipeline_only_resizes_and_is_repeatable(dtd):
    ds, _, _ = make_dtd_dataset(dtd, "val", SIZE, 4)
    a, b = _first(ds)[0], _first(ds)[0]
    np.testing.assert_array_equal(a, b)
    assert a.shape == (4, SIZE, SIZE, 3)


def test_no_augmentation_makes_the_train_pipeline_deterministic_up_to_the_shuffle(dtd):
    ds, _, _ = make_dtd_dataset(dtd, "train", SIZE, 8, augment=False, shuffle_buffer=1)
    a = np.concatenate([x for x, _ in ds.as_numpy_iterator()])
    b = np.concatenate([x for x, _ in ds.as_numpy_iterator()])
    np.testing.assert_array_equal(a, b)
