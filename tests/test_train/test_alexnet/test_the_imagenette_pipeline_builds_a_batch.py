"""The AlexNet trainer defaults to Imagenette and its pipeline yields a correct batch.

Guard for plan-2026-10-14T062616-81938e81 step 13.5 (D-104). Before it the
trainer's default ``--dataset imagenet`` needed ImageNet class directories that
are not on this machine, so a default run could not start. Imagenette
(``imagenette/320px-v2``, a 10-class ImageNet subset) is on disk as TFDS
records; the trainer reads it through the shared ``build_raw_image_dataset``
with ``crop_policy="random_resized_crop"`` at the model's own 227 input.

The batch tests read REAL records and never download: they skip when the
records are absent, and a skip is not a pass.
"""

import os
from glob import glob

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_datasets as tfds

from train.alexnet import train_alexnet as module
from train.common import IMAGENETTE_TFDS_NAME, build_raw_image_dataset

_TFDS_DATA_DIR = os.environ.get(
    "TFDS_DATA_DIR", os.path.expanduser("~/tensorflow_datasets"))
_RECORDS = sorted(glob(os.path.join(
    _TFDS_DATA_DIR, *IMAGENETTE_TFDS_NAME.split("/"), "1.0.0",
    "imagenette-validation.tfrecord-*")))
needs_records = pytest.mark.skipif(
    not _RECORDS,
    reason=f"imagenette TFDS records are not prepared under {_TFDS_DATA_DIR!r}; "
           f"this test never downloads. A skip is not a pass.",
)

# Literal ImageNet statistics, not the module constant, so a change to the
# constant is seen here.
_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])
_SIZE = 227
_BATCH = 4


def test_the_default_dataset_is_imagenette_at_the_models_input():
    args = module.parse_arguments([])
    assert args.dataset == "imagenette"
    assert args.image_size == _SIZE
    assert module.DATASET_CLASSES[args.dataset] == 10
    config = module.TrainingConfig()
    assert (config.dataset, config.num_classes, config.image_size) == (
        "imagenette", 10, _SIZE)


def test_every_dataset_choice_has_a_class_count():
    assert sorted(module.DATASET_CLASSES) == [
        "cifar10", "cifar100", "imagenet", "imagenette"]


@needs_records
def test_a_training_batch_has_the_right_shape_labels_and_range():
    config = module.TrainingConfig(batch_size=_BATCH)
    train_ds, _val_ds, steps, val_steps = module.create_imagenette_dataset(config)
    assert (steps, val_steps) == (9469 // _BATCH, -(-3925 // _BATCH))
    images, labels = next(iter(train_ds))
    assert images.shape == (_BATCH, _SIZE, _SIZE, 3)
    assert images.dtype == tf.float32
    assert labels.dtype == tf.int32
    assert np.all((labels.numpy() >= 0) & (labels.numpy() < 10))
    # Normalised with the ImageNet statistics: undoing them gives [0, 1]
    # pixels (a bilinear resize of [0, 1] values stays in [0, 1]).
    pixels = images.numpy() * _STD + _MEAN
    eps = 1e-5
    assert pixels.min() >= -eps and pixels.max() <= 1.0 + eps


@needs_records
def test_the_validation_batch_is_resize_then_centre_crop_then_imagenet_norm():
    config = module.TrainingConfig(batch_size=_BATCH)
    _train_ds, val_ds, _steps, _val_steps = module.create_imagenette_dataset(config)
    images, labels = next(iter(val_ds))
    assert images.shape == (_BATCH, _SIZE, _SIZE, 3)

    # Independent oracle from the raw records, in their unshuffled order:
    # shorter side to 227 / 0.875, centre 227 crop, ImageNet mean/std.
    raw = tfds.builder(IMAGENETTE_TFDS_NAME).as_dataset(
        split="validation", shuffle_files=False)
    for i, element in enumerate(raw.take(_BATCH)):
        image = tf.cast(element["image"], tf.float32) / 255.0
        h, w = image.shape[:2]
        scale = (_SIZE / 0.875) / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        # On the CPU, where tf.data runs: the GPU bilinear kernel rounds
        # differently (measured up to 3.6e-5 after the divide by std).
        with tf.device("/CPU:0"):
            image = tf.image.resize(image, (nh, nw), method="bilinear").numpy()
        top, left = (nh - _SIZE) // 2, (nw - _SIZE) // 2
        expected = (image[top:top + _SIZE, left:left + _SIZE] - _MEAN) / _STD
        assert int(labels[i]) == int(element["label"])
        # One subtract and one divide in float32 against a float64 expectation.
        np.testing.assert_allclose(
            images[i].numpy(), expected, rtol=0.0,
            atol=4 * np.finfo(np.float32).eps * np.abs(expected).max(),
        )


def test_the_random_resized_crop_policy_is_refused_off_imagenette():
    with pytest.raises(ValueError, match="imagenette only"):
        build_raw_image_dataset(
            "cifar10", 64, 2, is_training=True, crop_policy="random_resized_crop")


def test_an_unknown_crop_policy_is_refused():
    with pytest.raises(ValueError, match="Unknown crop_policy"):
        build_raw_image_dataset("imagenette", 64, 2, is_training=True,
                                crop_policy="centre")
