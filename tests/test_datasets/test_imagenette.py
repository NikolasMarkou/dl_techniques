"""``dl_techniques.datasets.vision.imagenette``: the Imagenette splits and pipelines.

The record reader (``read_imagenette_records``, the only TFDS touch) is replaced by a few
in-memory jpegs, so the split protocol is audited image by image: every train image has the
mean colour ``4 * i`` (with a horizontal ramp) and decodes back to its identity. The tests that need the real
prepared records (``imagenette/320px-v2`` under ``$TFDS_DATA_DIR``) skip when they are
absent; they never download, and a skip is not a pass.
"""

import os
from glob import glob

import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.datasets.vision.imagenet2012 import IMAGENET_MEAN, IMAGENET_STD
from dl_techniques.datasets.vision.imagenette import (
    IMAGENETTE_CLASS_NAMES,
    IMAGENETTE_NUM_CLASSES,
    IMAGENETTE_TFDS_NAME,
    IMAGENETTE_WNIDS,
    ImagenetteConfig,
    load_imagenette,
    read_imagenette_records,
)
from dl_techniques.datasets.vision.splits import seeded_test_rows
from tests.classification_fakes import (
    IMAGENETTE_FAKE_TEST,
    IMAGENETTE_FAKE_TRAIN,
    use_fake_imagenette,
)

N_TRAIN, N_TEST = IMAGENETTE_FAKE_TRAIN, IMAGENETTE_FAKE_TEST


@pytest.fixture(autouse=True)
def fake_records(monkeypatch):
    return use_fake_imagenette(monkeypatch)


def _config(**kwargs):
    settings = dict(image_size=24, batch_size=6, crop_policy="resize", augment=False,
                    normalize="unit", seed=3)
    settings.update(kwargs)
    return ImagenetteConfig(**settings)


def _ids(split, config, offset=0):
    """Identities of every image of a split, recovered from the decoded colour."""
    pixels = np.concatenate([x.numpy() for x, _ in load_imagenette(split, config).dataset])
    return np.rint((pixels.mean(axis=(1, 2, 3)) * 255.0 - offset) / 4.0).astype(int)


def test_constants_are_consistent() -> None:
    assert len(IMAGENETTE_WNIDS) == len(IMAGENETTE_CLASS_NAMES) == IMAGENETTE_NUM_CLASSES == 10
    assert len(set(IMAGENETTE_WNIDS)) == 10 and IMAGENETTE_TFDS_NAME == "imagenette/320px-v2"


def test_train_select_and_test_sizes() -> None:
    config = _config()
    train, select, test = (load_imagenette(s, config) for s in ("train", "select", "test"))
    assert (train.num_samples, select.num_samples, test.num_samples) == (54, 6, N_TEST)
    assert train.steps_per_epoch == 54 // 6 and select.steps_per_epoch == 1
    assert train.num_classes == 10 and train.class_names == list(IMAGENETTE_CLASS_NAMES)


def test_selection_is_disjoint_from_train_and_together_they_are_the_train_records() -> None:
    config = _config(batch_size=3)
    train, select = _ids("train", config), _ids("select", config)
    assert not set(train) & set(select)
    assert sorted(set(train) | set(select)) == list(range(N_TRAIN))
    assert len(select) == 6


def test_the_test_split_is_the_validation_records_and_never_the_train_ones(fake_records) -> None:
    config = _config(batch_size=4)
    ids = _ids("test", config, offset=2)
    assert sorted(ids) == list(range(N_TEST))
    assert set(fake_records) == {"train", "validation"} or set(fake_records) == {"validation"}


def test_the_two_crop_policies_have_different_evaluation_geometry() -> None:
    """Whole image resized vs shorter side resized then centre-cropped: different pixels."""
    resize = next(iter(load_imagenette("test", _config(crop_policy="resize")).dataset))[0]
    centre = next(iter(load_imagenette(
        "test", _config(crop_policy="random_resized_crop")).dataset))[0]
    assert not np.array_equal(resize.numpy(), centre.numpy())


def test_the_class_names_follow_the_wnid_order() -> None:
    pairs = dict(zip(IMAGENETTE_WNIDS, IMAGENETTE_CLASS_NAMES))
    assert pairs["n01440764"] == "tench" and pairs["n03000684"] == "chain saw"
    assert pairs["n03425413"] == "gas pump" and pairs["n03888257"] == "parachute"


def test_the_split_is_deterministic_in_the_seed_and_depends_on_it() -> None:
    a, b, c = (_ids("select", _config(seed=s)) for s in (1, 1, 2))
    np.testing.assert_array_equal(a, b)
    assert sorted(a) != sorted(c)


def test_max_images_caps_the_train_pool_and_the_test_split() -> None:
    config = _config(max_images=15, batch_size=3)
    train, select, test = (load_imagenette(s, config) for s in ("train", "select", "test"))
    assert (train.num_samples, select.num_samples, test.num_samples) == (14, 1, 15)
    rows = seeded_test_rows(N_TEST, 3, 15)
    assert sorted(_ids("test", config, offset=2)) == sorted(rows.tolist())


def test_train_drops_the_ragged_batch_and_the_others_keep_every_row() -> None:
    config = _config(batch_size=5)
    sizes = [len(y) for _, y in load_imagenette("train", config).dataset]
    assert sizes == [5] * (54 // 5)
    assert sum(len(y) for _, y in load_imagenette("select", config).dataset) == 6
    assert sum(len(y) for _, y in load_imagenette("test", config).dataset) == N_TEST


@pytest.mark.parametrize("policy", ["resize", "random_resized_crop"])
@pytest.mark.parametrize("augment", [False, True])
def test_every_policy_yields_square_float_images_and_int_labels(policy, augment) -> None:
    data = load_imagenette("train", _config(crop_policy=policy, augment=augment))
    x, y = next(iter(data.dataset))
    assert x.shape == (6, 24, 24, 3) and x.dtype == tf.float32 and y.dtype == tf.int32


def test_selection_and_test_are_never_augmented() -> None:
    for split in ("select", "test"):
        first = next(iter(load_imagenette(split, _config(augment=True, seed=1)).dataset))[0]
        again = next(iter(load_imagenette(split, _config(augment=True, seed=1)).dataset))[0]
        np.testing.assert_array_equal(first.numpy(), again.numpy())


def test_training_augmentation_changes_the_pixels() -> None:
    plain = next(iter(load_imagenette("train", _config(augment=False, seed=1)).dataset))[0]
    augmented = next(iter(load_imagenette(
        "train", _config(augment=True, crop_policy="random_resized_crop", seed=1)).dataset))[0]
    assert not np.array_equal(plain.numpy(), augmented.numpy())


def test_normalization_imagenet_is_the_unit_image_standardized() -> None:
    unit = next(iter(load_imagenette("test", _config(normalize="unit")).dataset))[0].numpy()
    std = next(iter(load_imagenette("test", _config(normalize="imagenet")).dataset))[0].numpy()
    assert 0.0 <= unit.min() and unit.max() <= 1.0 + 1e-6
    np.testing.assert_allclose(std, (unit - IMAGENET_MEAN) / IMAGENET_STD, atol=1e-5)


@pytest.mark.parametrize("kwargs, message", [
    (dict(crop_policy="centre"), "Unknown crop_policy"),
    (dict(normalize="whiten"), "normalize must be one of"),
    (dict(image_size=0), "image_size must be positive"),
    (dict(batch_size=0), "batch_size must be positive"),
    (dict(validation_split=0.0), "strictly inside"),
    (dict(validation_split=1.0), "strictly inside"),
    (dict(validation_split=0.001, max_images=30), "holds out 0 of 30"),
])
def test_a_bad_setting_is_refused_naming_it(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        load_imagenette("train", _config(**kwargs))


def test_an_unknown_split_is_refused() -> None:
    with pytest.raises(ValueError, match="split must be one of"):
        load_imagenette("validation", _config())


def test_absent_records_are_a_loud_file_not_found_naming_the_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.undo()
    with pytest.raises(FileNotFoundError, match="nothing is downloaded"):
        read_imagenette_records("train", str(tmp_path))


# ---------------------------------------------------------------------
# The real records
# ---------------------------------------------------------------------

_DATA_DIR = os.environ.get("TFDS_DATA_DIR", os.path.expanduser("~/tensorflow_datasets"))
_RECORD_DIR = os.path.join(_DATA_DIR, *IMAGENETTE_TFDS_NAME.split("/"), "1.0.0")
needs_records = pytest.mark.skipif(
    not glob(os.path.join(_RECORD_DIR, "imagenette-validation.tfrecord-*")),
    reason=f"imagenette TFDS records are not prepared under {_DATA_DIR!r}; this test never "
           "downloads. A skip is not a pass.",
)


@needs_records
def test_the_real_records_have_the_documented_sizes_and_label_order(monkeypatch) -> None:
    monkeypatch.undo()
    images, labels = read_imagenette_records("validation")
    assert len(images) == len(labels) == 3925 and labels.min() == 0 and labels.max() == 9
    with open(os.path.join(_RECORD_DIR, "label.labels.txt")) as handle:
        assert handle.read().split() == list(IMAGENETTE_WNIDS)
    assert len(read_imagenette_records("train")[0]) == 9469


@needs_records
def test_the_real_evaluation_geometry_is_resize_then_centre_crop_then_imagenet_norm(monkeypatch) -> None:
    """Independent oracle from the raw records: shorter side to size / 0.875, centre crop."""
    monkeypatch.undo()
    size = 64
    config = ImagenetteConfig(image_size=size, batch_size=4, crop_policy="random_resized_crop",
                              augment=False, seed=0)
    images, labels = read_imagenette_records("validation")
    rows = seeded_test_rows(len(images), 0)[:4]
    got_x, got_y = next(iter(load_imagenette("test", config).dataset))
    for i, row in enumerate(rows):
        image = tf.cast(tf.io.decode_jpeg(images[row], channels=3), tf.float32) / 255.0
        h, w = image.shape[:2]
        scale = (size / 0.875) / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        with tf.device("/CPU:0"):  # where tf.data runs; the GPU kernel rounds differently
            image = tf.image.resize(image, (nh, nw), method="bilinear").numpy()
        top, left = (nh - size) // 2, (nw - size) // 2
        expected = (image[top:top + size, left:left + size] - IMAGENET_MEAN) / IMAGENET_STD
        assert int(got_y[i]) == int(labels[row])
        np.testing.assert_allclose(
            got_x[i].numpy(), expected, rtol=0.0,
            atol=4 * np.finfo(np.float32).eps * np.abs(expected).max())
