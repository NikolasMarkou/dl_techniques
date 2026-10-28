"""The AlexNet trainer defaults to Imagenette and its pipeline yields a correct batch.

Guard for plan-2026-10-14T062616-81938e81 step 13.5 (D-104). Before it the
trainer's default ``--dataset imagenet`` needed ImageNet class directories that
are not on this machine, so a default run could not start. Imagenette
(``imagenette/320px-v2``, a 10-class ImageNet subset) is on disk as TFDS
records; the trainer reads it through ``train.common.classification_data`` with
``crop_policy="random_resized_crop"`` at the model's own 227 input.

The batch tests read REAL records and never download: they skip when the
records are absent, and a skip is not a pass.
"""

import os
from glob import glob

import numpy as np
import pytest
import tensorflow as tf

from train.alexnet import train_alexnet as module
from dl_techniques.datasets.vision.imagenette import IMAGENETTE_TFDS_NAME

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
    data = module.create_imagenette_data(config)
    assert (data.n_train, data.n_select, data.n_test) == (8523, 946, 3925)
    assert data.steps_per_epoch == 8523 // _BATCH
    images, labels = next(iter(data.train_ds))
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
def test_the_selection_and_test_batches_are_evaluation_geometry():
    """Never augmented: two builds of the same seed give the same selection batch."""
    config = module.TrainingConfig(batch_size=_BATCH, seed=3)
    first = next(iter(module.create_imagenette_data(config).select_ds))[0].numpy()
    second = next(iter(module.create_imagenette_data(config).select_ds))[0].numpy()
    np.testing.assert_array_equal(first, second)
    assert first.shape[1:] == (_SIZE, _SIZE, 3)


def test_the_trainer_has_no_second_imagenette_reader():
    import inspect
    source = inspect.getsource(module)
    assert "build_raw_image_dataset" not in source and "create_imagenette_dataset" not in source
